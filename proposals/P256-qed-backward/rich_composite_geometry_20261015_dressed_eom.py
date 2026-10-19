"""NEW exact unmasked R/complexTrU/Pphi local-source EOM/current consumer.
[H,F B]Omega=F[H,B]Omega+[H,F]B Omega BEFORE count slicing. Includes
mechanical/SU3electric/plaquette and shared noncommuting hop-field derivatives.
ALL reachable output counts are a memory partition, never source quotas.
"""
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any
import gc,json,resource,sys,time
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_cloud_probe import neutral_state,inner,add,LocalElectric
from rich_composite_geometry_20261015_source_eom import source_operator,commutator_terms,bare_state


def parts(primitive,state):
    result={}
    for key,field in state.items():result.setdefault(primitive.label(key)[:-1],{})[key]=field
    return result


def delta(operations):
    value=[0]*16
    for index,create in operations:
        site,f,_,member=model.unpack(index);value[4*site+2*f+int(member==3)]+=1 if create else -1
    return tuple(value)


def difference(left,right):return tuple(a-b for a,b in zip(left,right))
def plus(left,right):return tuple(a+b for a,b in zip(left,right))


def pw_matrix(maps):
    result=np.zeros((19,19),complex)
    for column,entries in enumerate(maps):
        for row,value in entries:result[row,column]=value
    return result


def field_maps(matrix):return [[(int(row),complex(matrix[row,column])) for row in np.flatnonzero(abs(matrix[:,column])>1e-14)] for column in range(19)]


def field_add(target,key,field,pw,rotation,phi,coefficient=1.):
    value=rotation@field@phi.T
    for index,weight in pw[key[-1]]:add(target,key[:-1]+(index,),coefficient*weight,value)


def field_apply(state,factors):
    pw,rotation,phi=factors;result={}
    for key,field in state.items():field_add(result,key,field,pw,rotation,phi)
    return result


def combine(target,other,coefficient=1.):
    for key,field in other.items():add(target,key,coefficient,field)


def responses():return {'D':{},'J':{},'contact':{},'finite':{}}


def add_response(target,key,field,coefficient,edge,charge,direction,theta):
    phase=direction*charge if edge==3 else 0.
    for name,factor in (('D',1.),('J',-1j*phase),('contact',-phase*phase),('finite',np.exp(1j*theta*phase))):
        if factor:add(target[name],key,coefficient*factor,field)


def bare_sector(primitive,omega_parts,templates,target_counts,scalar,rotations,theta):
    result=responses();p=primitive.p
    for (edge,flavour),operators in templates.items():
        change=delta(operators[0][2]);state=omega_parts.get(difference(target_counts,change),{})
        for key,field in state.items():
            fields=[p['scalar_coupling']*primitive.graph['weights'][edge]*field*scalar['grid'][None,:]]
            fields.extend(1j*p['vector_velocity']*primitive.graph['weights'][edge]*(rotations[edge,i]@field) for i in range(3))
            for component,dagger,operations,maps,coefficient in operators:
                factor=-1 if component and dagger else 1
                for target,car in primitive.coordinate_operations(key,operations).items():
                    for pw,wilson in maps[key[-1]]:add_response(result,target[:-1]+(pw,),fields[component],factor*coefficient*car*wilson,edge,2/3 if flavour==0 else -1/3,-1 if dagger else 1,theta)
    return result


def density_add(accumulator,state,nphi):
    bodies=dict.fromkeys(key[:-1] for key in state)
    for body in bodies:
        field=np.zeros((19,10,nphi),complex)
        for pw in range(19):
            value=state.get(body+(pw,))
            if value is not None:field[pw]=value
        accumulator[0]+=np.einsum('prf,prg->fg',field,field.conj());accumulator[1]+=np.einsum('prf,psf->rs',field,field.conj());flat=field.reshape(19,-1);accumulator[2]+=flat@flat.conj().T


def field_readout(density,primitive,scalar):
    phi,rotation,pw=density;norm=float(np.trace(phi).real)
    return {'raw_norm2':norm,'Phi2':float(np.trace(phi@np.diag(scalar['grid']**2)).real/norm),'Phi_energy':float(np.trace(phi@scalar['H']).real/norm),
        'R_j1_weight':float(np.trace(rotation[1:,1:]).real/norm),'loop_ReTrU':float(np.trace(pw@primitive.loop['B']).real/norm),'PW_Casimir':float(np.trace(pw@primitive.loop['C']).real/norm)}


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time();data=load(config['neutral_basis_snapshot']);primitive:Any=primitive_for(data);p=data['p'];coupling=load(config['neutral_coupling_snapshot']);eigen=np.load(config['neutral_states_snapshot']);omega=neutral_state(data,coupling,eigen['states'][:,0]);reference_energy=float(eigen['energies'][0]);del data,coupling,eigen;gc.collect()
    scalar:dict[str,Any];rotor:dict[str,Any]
    scalar,rotor=model.field_and_rotor(p);nf=p['scalar_points'];identity_pw=np.eye(19);identity_r=np.eye(10);identity_phi=np.eye(nf);axes=np.asarray(primitive.graph['axes']);weights=np.asarray(primitive.graph['weights']);theta=config['finite_material_EM_theta']
    rotations={(edge,i):sum((axes[edge,a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for edge,i in product(range(4),range(3))}
    right=angular.right_angular(rotor);kinetic=sum((primitive.graph['inertia_inverse'][a,b]*(right[a]@right[b]).toarray()/2 for a,b in np.ndindex(3,3)),np.zeros((10,10),complex));wilson=sum((primitive.loop['U'][a,a] for a in range(3)),np.zeros((19,19),complex))
    families=['bare','vector','wilson','momentum'];source_specs={1:source_operator(0,'half_triplet',1,False),-1:source_operator(0,'half_triplet',-1,False)};bare={m:bare_state(primitive,omega,terms) for m,terms in source_specs.items()};bare_parts={m:parts(primitive,state) for m,state in bare.items()};omega_parts=parts(primitive,omega);templates={m:commutator_terms(primitive,terms) for m,terms in source_specs.items()}
    # Each piece is a literal F_pw tensor F_R tensor F_phi times B_m on ALL inputs.
    pieces=[('bare',1,identity_pw,identity_r,identity_phi),('vector',1,identity_pw,rotations[0,2],identity_phi),('vector',-1,identity_pw,rotations[0,0]+1j*rotations[0,1],identity_phi),('wilson',1,wilson,identity_r,identity_phi),('momentum',1,identity_pw,identity_r,scalar['P'])]
    sources={name:{} for name in families};compiled=[];electric=LocalElectric(primitive);path=np.r_[0.,np.cumsum(np.asarray(primitive.graph['lengths'])[:3])];length_total=float(sum(primitive.graph['lengths']));matrix_roundoff=0.
    for family,m,pw,rotation,phi in pieces:
        factors=(field_maps(pw),rotation,phi);combine(sources[family],field_apply(bare[m],factors));local=[];colour=[];hops=[]
        for component,fieldpw,fieldr,fieldphi in (
            ('Phi_kinetic',pw,rotation,scalar['H']@phi-phi@scalar['H']),
            ('R_Noether_kinetic',pw,kinetic@rotation-rotation@kinetic,phi),
            ('plaquette',-p['plaquette']*(primitive.loop['B']@pw-pw@primitive.loop['B']),rotation,phi),
            ('electric_loop',p['electric']*length_total*(primitive.loop['C']@pw-pw@primitive.loop['C'])/2,rotation,phi)):
            size=float(np.linalg.norm(fieldpw)*np.linalg.norm(fieldr)*np.linalg.norm(fieldphi))
            if size>2e-14:local.append((component,field_maps(fieldpw),fieldr,fieldphi))
            else:matrix_roundoff+=size*size
        for site,a in product(range(1,4),range(8)):
            comm=p['electric']*path[site]*(primitive.loop['R'][a]@pw-pw@primitive.loop['R'][a]);size=float(np.linalg.norm(comm)*np.linalg.norm(rotation)*np.linalg.norm(phi))
            if size>2e-14:colour.append((site,a,field_maps(comm),rotation,phi))
            else:matrix_roundoff+=size*size
        for (edge,flavour,iswhite,component,dagger),transitions in primitive.maps.items():
            hr=identity_r if component==0 else (-1 if dagger else 1)*1j*p['vector_velocity']*weights[edge]*rotations[edge,component-1]
            hp=p['scalar_coupling']*weights[edge]*scalar['phi'] if component==0 else identity_phi
            fr_common=hr@rotation-rotation@hr if family=='vector' else hr
            fp_common=hp@phi-phi@hp if family=='momentum' else hp
            for operations,spin,maps in transitions:
                hw=pw_matrix(maps)
                if family=='vector':fw=hw;fr=fr_common;fp=fp_common
                elif family=='wilson':fw=hw@pw-pw@hw;fr=fr_common;fp=fp_common
                elif family=='momentum':fw=hw;fr=fr_common;fp=fp_common
                else:continue
                size=float(np.linalg.norm(fw)*np.linalg.norm(fr)*np.linalg.norm(fp))
                if size>2e-14:
                    charge=(0. if flavour==0 else -1.) if iswhite else (2/3 if flavour==0 else -1/3)
                    hops.append((delta(operations),operations,spin,field_maps(fw),fr,fp,edge,charge,-1 if dagger else 1))
                else:matrix_roundoff+=size*size
        compiled.append((family,m,factors,local,colour,hops))
    source_parts={name:parts(primitive,state) for name,state in sources.items()};source_gram=np.array([[inner(sources[a],sources[b]) for b in families] for a in families]);targets=set()
    for _,m,_,_,_,hops in compiled:
        targets.update(bare_parts[m])
        for change,_,_,_,_,_,_,_,_ in hops:
            for counts in bare_parts[m]:targets.add(plus(counts,change))
        for operators in templates[m].values():
            change=delta(operators[0][2]);targets.update(plus(counts,change) for counts in omega_parts)
    targets=sorted(targets);grams={name:np.zeros((4,4),complex) for name in ('D','J','contact','finite')};cross=np.zeros((4,4),complex);components={name:defaultdict(float) for name in families};component_interference={name:defaultdict(complex) for name in families};density={name:[np.zeros((nf,nf),complex),np.zeros((10,10),complex),np.zeros((19,19),complex)] for name in families};maximum_rows=0;calls=0
    for sector_number,target_counts in enumerate(targets):
        derivatives={name:responses() for name in families};component_states={name:{} for name in families};transforms={};bare_derivatives={m:bare_sector(primitive,omega_parts,templates[m],target_counts,scalar,rotations,theta) for m in source_specs}
        for family,m,factors,local,colour,hops in compiled:
            for response,state in bare_derivatives[m].items():combine(derivatives[family][response],field_apply(state,factors))
            component_states[family].setdefault('F_bare_EOM',{})
            combine(component_states[family]['F_bare_EOM'],field_apply(bare_derivatives[m]['D'],factors))
            state=bare_parts[m].get(target_counts,{})
            for component,pw,rotation,phi in local:
                part={}
                for key,field in state.items():field_add(part,key,field,pw,rotation,phi)
                combine(component_states[family].setdefault(component,{}),part)
                for response in ('D','finite'):combine(derivatives[family][response],part)
            for site,a,pw,rotation,phi in colour:
                part={}
                for key,field in state.items():
                    for column,charge in electric.charge(key[site],a).items():
                        target=key[:site]+(column,)+key[site+1:];field_add(part,target,field,pw,rotation,phi,charge)
                combine(component_states[family].setdefault('electric_colour_cross',{}),part)
                for response in ('D','finite'):combine(derivatives[family][response],part)
            for change,operations,spin,pw,rotation,phi,edge,charge,direction in hops:
                state=bare_parts[m].get(difference(target_counts,change),{})
                for key,field in state.items():
                    transform_key=(m,id(rotation),id(phi),key);value=transforms.get(transform_key)
                    if value is None:value=rotation@field@phi.T;transforms[transform_key]=value
                    calls+=1
                    for target,car in primitive.coordinate_operations(key,operations).items():
                        for index,wilson_value in pw[key[-1]]:
                            result_key=target[:-1]+(index,);coefficient=spin*car*wilson_value;add_response(derivatives[family],result_key,value,coefficient,edge,charge,direction,theta);add(component_states[family].setdefault('shared_hop_field_commutator',{}),result_key,coefficient,value)
        for response in grams:
            states=[derivatives[name][response] for name in families];grams[response]+=np.array([[inner(a,b) for b in states] for a in states])
        for a,name in enumerate(families):
            state=derivatives[name]['D'];maximum_rows=max(maximum_rows,len(state));density_add(density[name],state,nf)
            for b,target in enumerate(families):cross[b,a]+=inner(source_parts[target].get(target_counts,{}),state)
            names=list(component_states[name])
            for component in names:components[name][component]+=float(inner(component_states[name][component],component_states[name][component]).real)
            for left,right in product(names,names):component_interference[name][left+'|'+right]+=inner(component_states[name][left],component_states[name][right])
        if sector_number%8==0:print(json.dumps({'stage':'exact_R_U_Pphi_source_EOM_all_output_counts','sector':sector_number,'all_sectors':len(targets),'maximum_sector_source_rows':maximum_rows,'hop_input_calls':calls}),flush=True)
        del derivatives,component_states,bare_derivatives,transforms;gc.collect()
    result={'kind':'EXACT_UNMASKED_R_COMPLEXW_PPHI_SOURCE_EOM_WITH_ALL_FIELD_DERIVATIVES','families':families,'parameters':p,'reference_selected_energy':reference_energy,'source_Gram_real':source_gram.real.tolist(),'source_Gram_imag':source_gram.imag.tolist(),'K_real':cross.real.tolist(),'K_imag':cross.imag.tolist(),'Gdot_norm':float(np.linalg.norm(1j*(cross-cross.conj().T))),
        'all_reachable_output_count_sectors':len(targets),'maximum_one_sector_source_rows':maximum_rows,'shared_field_hop_input_calls':calls,'matrix_roundoff_discarded_norm2_sum':matrix_roundoff,'sources':{},
        'scope':'Original actual hot onehop Omega, no neutral/source quota; ALL outputcount sectors only partition memory. Literal F B and D_F=[H,F B] on ALL inputs. FullPhi7/R10/PW19 and all generated localCAR sea components. Finite-cut shared-field commutators retained, including remote quark/white hops; not full continuum field closure. This is source EOM/current/field response, NOT a pole/mass/decay/lifetime or finite-time mobility proof. No fixed-G Hermitization or orthogonal-connection omission.',
        'derivatives':'F[H,B]+[Hphi,F]B+[Hrot,F]B+[plaquette,F]B+[HE,F]B+sum_h[A_h,F]C_h B. HE cross=e sum_s,a path_s Q_s^a[R_colour^a,F] plus(e/2)sumell[C_loop,F]. FULL localFock Q acts before final contraction. Corrected creation index(mu,m),complexTrU, actualPphi. Field source is EM-independent; J=-dD/dtheta/contact=d²D/dtheta² use microscopic physical quark/white charge, all actual chord hop terms.'}
    for i,name in enumerate(families):
        g=float(source_gram[i,i].real);entry=cross[i,i];norm=float(grams['D'][i,i].real);orth=norm-abs(entry)**2/g
        result['sources'][name]={'raw_strength':g,'full_EOM_norm2':norm,'source_frequency_real':float((entry/g).real),'source_frequency_imag':float((entry/g).imag),'orthogonal_source_cloud_norm2':float(orth),'orthogonal_EOM_cloud_RMS':float(np.sqrt(orth/g)),
            'actual_current_commutator_norm2':float(grams['J'][i,i].real),'actual_contact_commutator_norm2':float(grams['contact'][i,i].real),'finite_theta_EOM_norm2':float(grams['finite'][i,i].real),'EOM_field_response':field_readout(density[name],primitive,scalar),'component_norm2':dict(components[name]),'component_interference_sum_real':float(sum(component_interference[name].values()).real)}
    np.savez(out+'_raw.npz',source_Gram=source_gram,K=cross,D_Gram=grams['D'],J_Gram=grams['J'],contact_Gram=grams['contact'],finite_Gram=grams['finite'],**{name+'_Phi_density':density[name][0] for name in families},**{name+'_R_density':density[name][1] for name in families},**{name+'_PW_density':density[name][2] for name in families})
    result['local_receipt']=primitive.local.receipt();result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
