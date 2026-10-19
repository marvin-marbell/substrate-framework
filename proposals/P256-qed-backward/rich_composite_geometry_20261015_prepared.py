"""Prepared-source dynamics on the EXISTING full native/Phi/SO3 family.
No charged poles, new family constructor, effective baryon hop or spin fit.
Disk Arnoldi vectors are numerical integration data, not a physical state cut.
"""
from __future__ import annotations
import gc,json,resource,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import expm
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_sources import embedding,raw_source
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load


def source_state(neutral,charged,primitive,coupling,omega,spec):
    keys,matrix=embedding(neutral['basis'])
    targets,operator=raw_source(primitive,keys,spec)
    basis=charged['basis'];plans={};nf=neutral['p']['scalar_points']
    for row,key in enumerate(targets):
        lab=primitive.label(key) if hasattr(primitive,'local') else model.label(key[0])
        block=basis.storage.get(lab)
        if block is not None:
            target=block.lookup.get(key)
            if target is not None:plans.setdefault(lab,[]).append((row,target))
    result=np.zeros((len(basis.vectors),10,nf),complex);full=0.
    for r,u in enumerate(coupling['U']):
        raw=operator@(matrix@(u@omega.reshape(coupling['dimension'],nf)))
        full+=float(np.linalg.norm(raw)**2)
        for lab,indices in plans.items():
            block=basis.storage[lab];source_rows,target_rows=map(np.asarray,zip(*indices))
            rhs=np.zeros((len(block.keys),nf),complex);rhs[target_rows]=raw[source_rows]
            result[np.asarray(block.columns),r,:]=np.conjugate(block.matrix.T@np.conjugate(rhs))
    return result,full


def count_data(basis):
    counts=np.empty((len(basis.vectors),16),np.int8);blocks=[]
    for lab,block in basis.storage.items():
        rows=np.asarray(block.columns,dtype=np.int64);counts[rows]=lab[:-1];blocks.append(rows)
    _,groups=np.unique(counts,axis=0,return_inverse=True)
    counts=counts.reshape(-1,4,4);colour=counts[:,:,0]+counts[:,:,2];white=counts[:,:,1]+counts[:,:,3]
    compact=np.all(np.sort(colour,axis=1)==[6,6,6,9],axis=1)
    fragmented=np.all(np.sort(colour,axis=1)==[6,7,7,7],axis=1)
    return {'counts':counts,'colour':colour,'white':white,'Q':2*counts[:,:,0]/3-counts[:,:,2]/3-counts[:,:,3],
        'B':(colour-6)/3,'compact':compact,'fragmented':fragmented,
        'site':compact[:,None]&(colour==9),'groups':groups,'blocks':blocks}


def rotation_add(target,state,rotation,coefficient,buffer):
    matrix=sparse.coo_matrix(rotation)
    for r,c,value in zip(matrix.row,matrix.col,matrix.data):
        np.multiply(state[:,c,:],coefficient*value,out=buffer)
        np.add(target[:,r,:],buffer,out=target[:,r,:])


class NativeHamiltonian:
    """Body-major native x ALL10 orientation x ALL7 quantum scalar action."""
    def __init__(self,operators,p,scalar,rotor):
        self.operators=operators;self.p=p;self.scalar=scalar;self.rotor=rotor;self.calls=0
        generators=angular.right_angular(rotor);inverse=np.asarray(model.geometry(p)['inertia_inverse'],dtype=float)
        self.kinetic=sparse.csr_matrix((10,10),dtype=complex)
        for a,b in np.ndindex(3,3):self.kinetic+=inverse[a,b]*(generators[a]@generators[b])/2

    def apply(self,state,mode='full'):
        self.calls+=1;n,_,nf=state.shape;flat=state.reshape(n,-1)
        result=state@self.scalar['H'].T;buffer=np.empty((n,nf),complex)
        rotation_add(result,state,self.kinetic,1.,buffer)
        if mode!='colour_electric_off':
            raw=self.operators['electric']@flat;result+=raw.reshape(state.shape);del raw
        raw=self.operators['B']@flat;raw*=-self.p['plaquette'];result+=raw.reshape(state.shape);del raw
        if mode!='ALL_hops_off':
            raw=(self.operators['scalar']@flat).reshape(state.shape)
            np.multiply(raw,(self.p['scalar_coupling']*self.scalar['grid'])[None,None,:],out=raw)
            result+=raw;del raw
            for i,a in np.ndindex(3,3):
                raw=(self.operators[f'vector_{i}_{a}']@flat).reshape(state.shape)
                rotation_add(result,raw,self.rotor['coordinate'][i][a],self.p['vector_velocity'],buffer)
                del raw
        return result


def pvm(state,data):
    weights=np.sum(abs(state)**2,axis=(1,2))
    return weights,np.bincount(data['groups'],weights=weights)


def readout(state,data,scalar,operators,initial_pvm):
    weights,joint=pvm(state,data);colour=data['colour'];white=data['white'];baryon=data['B']
    rho=np.einsum('nrf,nrg->fg',state.conj(),state);field=np.real(np.diag(rho))
    average=float(field@scalar['grid']);flat=state.reshape(len(weights),-1)
    result={'norm':float(weights.sum()),'compact_channel':float(weights@data['compact']),
        'fragmented_channel':float(weights@data['fragmented']),'compact_site':(weights@data['site']).tolist(),
        'ncolour_site_mean':(weights@colour).tolist(),'nwhite_site_mean':(weights@white).tolist(),
        'nwhite_site_variance':(weights@(white**2)-(weights@white)**2).tolist(),
        'Q_em_site_mean':(weights@data['Q']).tolist(),'baryon_site_mean':(weights@baryon).tolist(),
        'baryon_site_covariance':((baryon.T*weights)@baryon-np.outer(weights@baryon,weights@baryon)).tolist(),
        'material_count_PVM_TV_from_preparation':float(np.linalg.norm(joint-initial_pvm,ord=1)/2),
        'phi':average,'phi_variance':float(field@(scalar['grid']**2)-average**2),
        'scalar_reduced_purity':float(np.trace(rho@rho).real)}
    for name in ('electric','B','C'):
        result['loop_ReTrU' if name=='B' else 'PW_Casimir' if name=='C' else 'electric_energy']=float(np.vdot(flat,operators[name]@flat).real)
    return result


def nohop_trajectory(initial,data,hamiltonian,times,out):
    body=hamiltonian.operators['electric']-hamiltonian.p['plaquette']*hamiltonian.operators['B']
    initial_pvm=pvm(initial,data)[1];active=[rows for rows in data['blocks'] if np.any(initial[rows])]
    events=[]
    for t in times:
        state=np.zeros_like(initial);rotation=expm(-1j*t*hamiltonian.kinetic.toarray());field=expm(-1j*t*hamiltonian.scalar['H'])
        for rows in active:
            block=(expm(-1j*t*body[rows][:,rows].toarray())@initial[rows].reshape(len(rows),-1)).reshape(len(rows),10,-1)
            state[rows]=np.einsum('rs,nsf->nrf',rotation,block)@field.T
        events.append({'t':t,**readout(state,data,hamiltonian.scalar,hamiltonian.operators,initial_pvm)})
        Path(out+'_ALL_hops_off.json').write_text(json.dumps({'events':events,'meaning':'Exact small native physical-count/Sz blocks and fullPhi/SO3 exponentials; ALL original hops removed, no clamp. Material PVM must stay fixed; lab rotation need not.'},indent=2)+'\n')
    return events


def trajectory(initial,data,hamiltonian,times,out,mode,maximum,checkpoints):
    shape=initial.shape;n=initial.size;small=np.zeros((maximum+1,maximum),complex)
    files=[Path(out+'_prepared.npy')];initial_pvm=pvm(initial,data)[1]
    previous=None;records=[];second_pass=0.;started=time.time()
    for j in range(maximum):
        column=np.load(files[j],mmap_mode='r');value=hamiltonian.apply(column.reshape(shape),mode).reshape(-1)
        for repeat in range(2):
            for i,path in enumerate(files):
                old=column if i==j else np.load(path,mmap_mode='r')
                coefficient=np.vdot(old.reshape(-1),value);small[i,j]+=coefficient
                if repeat:second_pass=max(second_pass,float(abs(coefficient)))
                value-=coefficient*old.reshape(-1)
                if i!=j:del old
        del old,column
        beta=float(np.linalg.norm(value));small[j+1,j]=beta;m=j+1
        breakdown=beta<=1e-13*max(1.,float(np.linalg.norm(small[:m,j])))
        if m<maximum and not breakdown:
            value/=beta;path=Path(out+f'_{mode}_q{m:03d}.npy');np.save(path,value.reshape(shape));files.append(path)
        del value
        np.savez(out+f'_{mode}_Arnoldi.npz',Hessenberg=small[:m+1,:m],times=np.asarray(times),dimension=m)
        print(json.dumps({'stage':'prepared_actual_full_family_H_action','mode':mode,'Arnoldi_dimension':m,'matvec_count':hamiltonian.calls,'beta':beta}),flush=True)
        if m not in checkpoints and m!=maximum and not breakdown:continue
        projected=small[:m,:m];events=[];coefficients=[]
        for index,t in enumerate(times):
            coefficient=expm(-1j*t*projected)[:,0];coefficients.append(coefficient)
            state=np.zeros(shape,complex);difference=np.zeros(shape,complex) if previous is not None else None
            previous_coefficients=previous[index] if previous is not None else ()
            for i,path in enumerate(files[:m]):
                q=np.load(path,mmap_mode='r');state+=coefficient[i]*q
                if difference is not None:
                    old_coefficient=previous_coefficients[i] if i<len(previous_coefficients) else 0.
                    difference+=(coefficient[i]-old_coefficient)*q
                del q
            event={'t':t,**readout(state,data,hamiltonian.scalar,hamiltonian.operators,initial_pvm),
                'Arnoldi_residual_norm_at_time':float(beta*abs(coefficient[-1])),
                'native_state_difference_from_previous_prefix':float(np.linalg.norm(difference)) if difference is not None else None}
            events.append(event);del state,difference
        previous=coefficients
        record={'mode':mode,'dimension':m,'events':events,'projected_H_Hermiticity_error':float(np.linalg.norm(projected-projected.conj().T)),
            'largest_second_MGS_pass_coefficient':second_pass,'exact_breakdown_to_numerical_tolerance':breakdown,
            'elapsed_seconds':time.time()-started,
            'integration_scope':'Actual finite-time Arnoldi approximation on the UNCHANGED full declared native/Phi/SO3 family, not a physical low-state replacement. Prefix state differences and residuals are numerical evidence, NOT a certified global error bound. No expectation clamping or state renormalization.'}
        records.append(record);Path(out+f'_{mode}_checkpoint.json').write_text(json.dumps({'prefixes':records},indent=2)+'\n')
        np.savez(out+f'_{mode}_coefficients.npz',times=np.asarray(times),coefficients=np.asarray(coefficients))
        print(json.dumps({'stage':'actual_prepared_compact_survival_and_transfer','mode':mode,'dimension':m,'events':events}),flush=True)
        if breakdown:break
    return records


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    neutral=load(config['neutral_basis_snapshot']);charged=load(config['charged_basis_snapshot']);p=neutral['p']
    if charged['p']!=p:raise ValueError('Prepared source must use the SAME actual Hamiltonian/reference')
    primitive=primitive_for(charged);primitive_for(neutral,primitive)
    coupling=load(config['neutral_coupling_snapshot']);states=np.load(config['neutral_states_snapshot'])
    omega=states['states'][:,0];reference_energy=float(states['energies'][0]);spec=config['source_spec']
    initial,full=source_state(neutral,charged,primitive,coupling,omega,spec);retained=float(np.linalg.norm(initial)**2)
    data=count_data(charged['basis']);initial*=data['site'][:,0,None,None]
    support=float(np.linalg.norm(initial)**2)
    preparation={'source_spec':spec,'reference_energy':reference_energy,'full_raw_source_strength':full,
        'retained_source_strength':retained,'compact_site0_raw_strength':support,
        'conditional_compact_site0_fraction':support/retained if retained else None,
        'meaning':'Actual local canonical epsilon/CG source on its OWN quantum neutral stationary reference; then physical MATERIAL colour-count conditioning. Raw strengths are not probabilities. Prepared state is NOT a charged pole or a ground-spin claim.'}
    Path(out+'_preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    if support==0.:raise ValueError('Physical source has NO compact preparation support; no replacement source/state')
    initial/=np.sqrt(support);np.save(out+'_prepared.npy',initial)
    del neutral,charged,primitive,coupling,states,omega;gc.collect()
    operators=load(config['operators_snapshot']);operators.relocate(Path(config['operators_snapshot']).resolve().parent)
    required={'scalar','electric','B','C'}|{f'vector_{i}_{a}' for i,a in np.ndindex(3,3)}
    if set(operators)!=required:raise ValueError('Complete owning original native operators required; no field/state fallback')
    scalar,rotor=model.field_and_rotor(p);hamiltonian=NativeHamiltonian(operators,p,scalar,rotor)
    times=p['times'];maximum=int(config['maximum_Arnoldi_dimension']);checkpoints=tuple(config['Arnoldi_checkpoints'])
    result={'kind':'ACTUAL_PREPARED_OCCUPIED_COMPOSITE_SHORT_DYNAMICS_NOT_POLE_OR_BINDING_THEOREM','parameters':p,
        'preparation':preparation,'native_body_dimension':initial.shape[0],'full_Phi_SO3_dimension':initial.size,
        'numerical_maximum_Arnoldi_dimension':maximum,'ALL_hops_off':nohop_trajectory(initial,data,hamiltonian,times,out),'trajectories':{}}
    Path(out+'_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    for mode in ('full','colour_electric_off'):
        result['trajectories'][mode]=trajectory(initial,data,hamiltonian,times,out,mode,maximum,checkpoints)
        Path(out+'_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    result['limits']='Prepared finite-time test, not stationary binding/longevity or a proton. C/F refer to whole-sea quark-occupation channels, not3-added-particle positions/EMcharge/colour8. FullPhi7/PW19/SO3j0+j1 and ALL current native family retained. Severe occupied/pair-family cut and native-H/current range omissions remain. Colour-off is not guaranteed nonbinding; exact quadratic free control separate. No new spin/force/baryon hop, photon, freeCOM or continuum claim. Broad pole/Ward/source diagnostics deferred, not declared paid.'
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started,'owning_H_matvec_count':hamiltonian.calls}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
