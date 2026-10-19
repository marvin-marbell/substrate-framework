"""UNMASKED actual local source [H,Bdag]Omega, not hot-sea RMS subtraction.
Bare source operator extension is explicit at all inputs. Derive CAR commutator
BEFORE acting: genuinely commuting spectator terms cancel as OPERATORS.
Changed quantum field states, whole occupied sea and C partner retained.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import product
from pathlib import Path
import gc,json,resource,sys,time
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_compile import save
from rich_composite_geometry_20261015_cloud_probe import neutral_state,inner,add


def source_operator(site,kind,m2,removal):
    terms=[]
    for operations,coefficient in model.source_terms([site]*3,kind,m2):
        if removal:
            mapped=[]
            for index,_ in operations:
                v,f,s,c=model.unpack(index);coefficient*=(-1)**(v+s);mapped.append((model.mode(v,f,1-s,c),False))
            operations=mapped
        terms.append((operations,coefficient))
    return terms


def canonical(operations):
    indices=[index for index,_ in operations]
    if len(set(indices))!=len(indices):return None,0
    sign=(-1)**sum(indices[i]>indices[j] for i in range(len(indices)) for j in range(i+1,len(indices)))
    return tuple(sorted(operations)),sign


def commutator_terms(primitive,source_terms):
    combined=defaultdict(complex)
    for (edge,flavour,white,component,dagger),hops in primitive.maps.items():
        for (creation,annihilation),spin,maps in hops:
            for operations,source_coefficient in source_terms:
                for slot,(index,create) in enumerate(operations):
                    match=annihilation[0] if create else creation[0]
                    if index!=match:continue
                    replacement=(creation[0],True) if create else (annihilation[0],False)
                    changed=operations[:slot]+[replacement]+operations[slot+1:];monomial,sign=canonical(changed)
                    if monomial is None:continue
                    field_key=tuple(tuple(row) for row in maps)
                    combined[edge,flavour,white,component,dagger,monomial,field_key]+=source_coefficient*spin*sign*(1 if create else -1)
    groups=defaultdict(list)
    for (edge,flavour,white,component,dagger,operations,maps),coefficient in combined.items():
        if abs(coefficient)>2e-14:groups[edge,flavour].append((component,dagger,operations,maps,coefficient))
    return dict(groups)


def apply_terms(primitive,omega,terms,scalar,rotor):
    p=primitive.p;axes=np.asarray(primitive.graph['axes']);weights=np.asarray(primitive.graph['weights'])
    rotation={(edge,i):sum((axes[edge,a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for edge,i in product(range(4),range(3))}
    groups={}
    for branch,operators in terms.items():
        edge,flavour=branch;result={}
        for row,(key,field) in enumerate(omega.items()):
            fields=[p['scalar_coupling']*weights[edge]*field*scalar['grid'][None,:]]
            fields.extend(1j*p['vector_velocity']*weights[edge]*(rotation[edge,i]@field) for i in range(3))
            for component,dagger,operations,maps,coefficient in operators:
                factor=-1 if component and dagger else 1
                for target,car in primitive.coordinate_operations(key,operations).items():
                    for pw,wilson in maps[key[-1]]:add(result,target[:-1]+(pw,),factor*coefficient*car*wilson,fields[component])
        groups[branch]=result
        print(json.dumps({'stage':'actual_unprojected_source_EOM_branch','branch':branch,'operator_monomials':len(operators),'output_rows':len(result)}),flush=True)
    return groups


def conjugate(primitive,state):
    result={}
    for key,field in state.items():
        for target,coefficient in primitive.conjugate({key:1.+0j}).items():add(result,target,coefficient,field)
    return result


def bare_state(primitive,omega,terms):
    result={}
    for operations,coefficient in terms:
        for key,field in omega.items():
            for target,car in primitive.coordinate_operations(key,operations).items():add(result,target,coefficient*car,field)
    return result


def field_densities(groups,coefficients):
    bodies=sorted({key[:-1] for state in groups for key in state});nf=next(iter(groups[0].values())).shape[1]
    phi=np.zeros((nf,nf),complex);rotation=np.zeros((10,10),complex);pw=np.zeros((19,19),complex)
    for body in bodies:
        field=np.zeros((19,10,nf),complex)
        for index in range(19):
            key=body+(index,)
            for state,c in zip(groups,coefficients):
                if key in state:field[index]+=c*state[key]
        phi+=np.einsum('prf,prg->fg',field,field.conj());rotation+=np.einsum('prf,psf->rs',field,field.conj());flat=field.reshape(19,-1);pw+=flat@flat.conj().T
    return phi,rotation,pw


def field_readout(densities,primitive,scalar):
    phi,rotation,pw=densities;strength=float(np.trace(phi).real)
    return {'raw_strength':strength,'Phi_mean':float(np.trace(phi@np.diag(scalar['grid'])).real/strength),'Phi2':float(np.trace(phi@np.diag(scalar['grid']**2)).real/strength),
        'Phi_energy':float(np.trace(phi@scalar['H']).real/strength),'loop_ReTrU':float(np.trace(pw@primitive.loop['B']).real/strength),'PW_Casimir':float(np.trace(pw@primitive.loop['C']).real/strength),
        'SO3_j1_weight':float(np.trace(rotation[1:,1:]).real/strength),'Phi_boundary_weight':float((phi[0,0]+phi[-1,-1]).real/strength)}


def run_source(primitive,omega,p,scalar,rotor,config,removal,out):
    started=time.time();site=config['source_site'];kind=config['source_kind'];m2=config['source_twice_m'];operator=source_operator(site,kind,m2,removal)
    b=bare_state(primitive,omega,operator);g=float(inner(b,b).real);terms=commutator_terms(primitive,operator);groups=apply_terms(primitive,omega,terms,scalar,rotor);branches=list(groups);states=[groups[key] for key in branches]
    gram=np.array([[inner(left,right) for right in states] for left in states]);row=np.array([inner(b,state) for state in states]);total=np.sum(gram);entry=np.sum(row);frequency=entry/g
    direction=-1 if removal else 1;charges=np.array([2/3 if flavour==0 else -1/3 for edge,flavour in branches]);chord=np.array([edge==3 for edge,flavour in branches]);theta=config['finite_material_EM_theta']
    finite=np.exp(1j*direction*charges*chord*theta);current=-1j*direction*charges*chord;contact=-charges**2*chord
    source_overlaps={}
    for target_site in range(4):
        target=bare_state(primitive,omega,source_operator(target_site,kind,m2,removal));overlap=sum(inner(target,state) for state in states)
        source_overlaps[str(target_site)]={'raw_source_Gram':float(inner(target,target).real),'EOM_overlap_real':float(overlap.real),'EOM_overlap_imag':float(overlap.imag)}
    densities0=field_densities(states,np.ones(len(states)));densities_finite=field_densities(states,finite);densities_b=field_densities([b],[1.])
    orthogonal=float(total.real-abs(entry)**2/g)
    result={'source_charge':-1 if removal else 1,'source_delta_N':-3 if removal else 3,'raw_source_strength':g,'full_source_EOM_norm2':float(total.real),'EOM_norm2_imag_roundoff':float(total.imag),
        'source_frequency_mean_real':float(frequency.real),'source_frequency_mean_imag':float(frequency.imag),'source_strength_time_derivative':float(-2*entry.imag),
        'orthogonal_source_cloud_norm2':orthogonal,'orthogonal_EOM_cloud_RMS_energy':float(np.sqrt(orthogonal/g)),
        'orthogonal_norm_semantics':'Signed source-orthogonal norm retained without clamp; not a probability/current expectation clamp, mass recenter or lifetime.',
        'branches':[list(key) for key in branches],'source_target_overlaps':source_overlaps,
        'source_EM_commutator_current_norm2':float(np.vdot(current,gram@current).real),'source_EM_commutator_contact_norm2':float(np.vdot(contact,gram@contact).real),
        'finite_EM_EOM_norm2':float(np.vdot(finite,gram@finite).real),'finite_EM_EOM_source_overlap_real':float(np.dot(row,finite).real),'finite_EM_EOM_source_overlap_imag':float(np.dot(row,finite).imag),
        'field_on_bare_source':field_readout(densities_b,primitive,scalar),'field_on_EOM_source':field_readout(densities0,primitive,scalar),'field_on_finite_EM_EOM_source':field_readout(densities_finite,primitive,scalar),
        'raw_EOM_rows_by_branch':{str(key):len(groups[key]) for key in branches},'operator_terms_by_branch':{str(key):len(terms[key]) for key in branches},'elapsed_seconds':time.time()-started,
        'source_operator_extension':'Literal canonical local epsilon/CG cubic, unprojected at ALL input states. No W best-state extension ambiguity, no neutral count quota, no source/receiver fit.',
        'EOM_meaning':'D=[H,Bdag]Omega=Hc BdagOmega-Bdag HnOmega EXACT operator cancellation before action, not RMS subtraction. Source mean frequency/cloud spread on hot prepared sea, NOT pole/mass/decay/lifetime or binding. Orthogonal component removes the measured source direction diagnostically, NOT Hamiltonian energy recentering.',
        'field_derivatives':'Chosen Bdag has literal field coefficient1, so [Hphi,B]=[Hrot,B]=[plaquette,B]=0; local colour singlet gives [HE,B]=0 before Gauss reduction. Only actually incident outgoing quark hops remain; white/distant bilinears commute with this source. B carries physical CAR spin; total finite spatial J=L_R+S_CAR. A future F(Phi,R,U)B source MUST retain [Hfield,F], colour/Gauss and noncommuting projected field coefficient terms. No Phi/R-independent assumption is extended to dressed sources.',
        'EM_scope':'Actual primitive material chord Peierls derivative -d[Htheta,B]/dtheta and second derivative, with microscopic quark fractional charges, not baryon charge pasted on each hop. Finite theta changes EOM source at fixed initial Omega; NOT time-domain sea evolution, lab3D A/rotor minimal coupling, photons or COM.'}
    save(out+'_operator.pkl',{'source_terms':operator,'commutator_terms':terms,'p':p})
    np.savez(out+'_data.npz',EOM_branch_Gram=gram,source_EOM_row=row,current_coefficients=current,contact_coefficients=contact,finite_coefficients=finite,
        source_Phi_density=densities_b[0],EOM_Phi_density=densities0[0],EOM_R_density=densities0[1],EOM_PW_density=densities0[2],finite_EOM_Phi_density=densities_finite[0],finite_EOM_PW_density=densities_finite[2])
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time();data=load(config['neutral_basis_snapshot']);primitive=primitive_for(data);p=data['p'];coupling=load(config['neutral_coupling_snapshot']);eigen=np.load(config['neutral_states_snapshot'])
    omega=neutral_state(data,coupling,eigen['states'][:,0]);energy=float(eigen['energies'][0]);scalar,rotor=model.field_and_rotor(p);del data,coupling,eigen;gc.collect()
    particle=run_source(primitive,omega,p,scalar,rotor,config,False,out+'_B');anti_omega=conjugate(primitive,omega);anti=run_source(primitive,anti_omega,p,scalar,rotor,config,True,out+'_antiB')
    result={'kind':'EXACT_OPERATOR_SOURCE_EOM_ON_OWN_EVOLVING_SEA_NOT_HOT_RMS_SUBTRACTION','parameters':p,'reference_selected_energy':energy,'reference_norm2':float(inner(omega,omega).real),'C_reference_norm2':float(inner(anti_omega,anti_omega).real),
        'B':particle,'antiB_on_actual_C_Omega':anti,'local_receipt':primitive.local.receipt(),'resource':{'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started},
        'C_semantics':'Actual unitary CAR particle-hole/spin-flip plus PW3<->3bar transformation of OWN Omega; anti-uud charge-1,N29 source, NOT electron/white lepton or proton claim. Source strengths are raw canonical CAR strengths, not normalized outcome probabilities.',
        'reference_limit':'Selected correlated one-hop quantum sea/Phi7/PW19/R10 reference, known full-H RMS5.905. No claim of full-law stationary vacuum. EOM uses ALL its quark/white/field support without count masks; actual whole64 CAR before adaptive local Fock representation, no charged native/W projection. No fitted mass/spin penalty or coupling change.'}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
