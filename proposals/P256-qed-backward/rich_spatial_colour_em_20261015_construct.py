"""NEW extended spatial H/reference; exact factor checkpoint BEFORE expensive science."""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from rich_spatial_colour_em_20261015_model import build,assemble,angular,source,conjugation


def save_sparse(stored,prefix,matrix):
    matrix=sparse.csr_matrix(matrix)
    for key in ['data','indices','indptr']:stored[prefix+'_'+key]=getattr(matrix,key)
    stored[prefix+'_shape']=matrix.shape


def save_factor(stored,prefix,operator):
    stored[prefix+'_terms']=len(operator.terms);stored[prefix+'_include_scalar']=operator.include_scalar
    for index,(matter,rq,phi) in enumerate(operator.terms):
        save_sparse(stored,prefix+'_A'+str(index),matter);save_sparse(stored,prefix+'_B'+str(index),rq);stored[prefix+'_phi'+str(index)]=phi


def lowest(h,p):
    count=min(p['low_eigenstates'],h.shape[0]-2);ncv=min(h.shape[0],max(2*count+8,32))
    seed=np.random.default_rng(2221015).normal(size=h.shape[0])
    e,v=eigsh(h,k=count,ncv=ncv,which='SA',tol=p['eigen_tolerance'],maxiter=24000,v0=seed)
    u,s,_=np.linalg.svd(v,full_matrices=False);basis=u[:,s>1e-10*s[0]]
    small=basis.conj().T@(h@basis);e,w=np.linalg.eigh((small+small.conj().T)/2)
    return e,basis@w


def main():
    inp,out=map(Path,sys.argv[1:3]);p=json.loads(inp.read_text());started=time.time();model=build(p);base=model['base'];field=model['field'];rotor=base['rotor'];scalar=base['scalar']
    stored=dict(scalar_H=scalar['H'],scalar_grid=scalar['grid'],rotor_R=rotor['R'],rotor_weights=rotor['weights'],rotor_basis=rotor['basis'],field_basis=np.array(field['basis']),field_HE=field['H_E'],field_C=field['C'],field_boundary=field['boundary'],route_coefficients=model['coefficients']['values'],route_body_coefficients=model['coefficients']['body'],electric_raw_Gram=model['coefficients']['raw_electric_gram'],vertices=base['geometry']['vertices'])
    for axis in range(3):save_sparse(stored,'rotor_L'+str(axis),rotor['L'][axis]);stored['field_F'+str(axis)]=field['F'][axis]
    for index,path in enumerate(base['geometry']['paths']):stored['path'+str(index)]=path
    for index in range(6):stored['field_Q'+str(index)]=field['Q'][index];stored['field_Pi'+str(index)]=field['Pi'][index]
    startpath=inp.with_name(inp.name.replace('_input.json','_start.json'))
    recipe=json.loads(startpath.read_text()) if startpath.exists() else {'source_input_binding':'Direct invocation; no capture start receipt'}
    results={};laws={};spectra={}
    def checkpoint(stage):
        result=dict(kind='NEW_SPATIAL_COLOUR_EM_WHOLE_H_CHECKPOINT_NOT_BASE_STATE_SPLICE',parameters=p,stage=stage,source_input_recipe=recipe,results=results,V_self=model['V_self'],V_cross=model['V_cross'],elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,state_order=['CAR_PW_physical','scalar','material_SO3','quantum_field'])
        out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)
    for name,sector in base['sectors'].items():
        op=base['operators'][name];h=assemble(model,name);laws[name]=h
        save_factor(stored,name+'_H',h);save_sparse(stored,name+'_P',sector['P']);stored[name+'_masks']=np.array(sector['masks'],np.uint64)
        for key in ['Q','rho0','rho1','density_delta','density_delta2','electric','C','B','positive','N','PW_boundary','Q1_C','electric_cross','tree_C']:
            save_sparse(stored,name+'_'+key,op[key])
        for axis in range(3):save_sparse(stored,name+'_S'+str(axis),op['S'][axis])
        results[name]=dict(dimension=h.shape[0],matter_dimension=sector['dimension'],raw_Fock_masks=len(sector['masks']),raw_dimension=sector['raw_dimension'],colour_Gauss=sector['gauge_error'],term_count=len(h.terms),Q_em_mean=float(op['Q'].diagonal().real.mean()))
        checkpoint('constructed_'+name)
    for name in ['neutral','lepton','baryon']:
        h=laws[name];e,v=lowest(h,p);spectra[name]=(e,v);stored[name+'_e']=e;stored[name+'_v']=v
        state=v[:,0];results[name]['ground_energy']=float(e[0]);results[name]['eigen_residual']=float(np.linalg.norm(h@state-e[0]*state));results[name]['Gram_error']=float(np.linalg.norm(v.conj().T@v-np.eye(v.shape[1])))
        generators=[angular(model,name,axis) for axis in range(3)];results[name]['ground_J2']=float(sum(np.linalg.norm(j@state)**2 for j in generators))
        random=np.random.default_rng(1015).normal(size=h.shape[0])+1j*np.random.default_rng(1510).normal(size=h.shape[0]);random/=np.linalg.norm(random)
        results[name]['HJ_random_commutators']=[float(np.linalg.norm(h@(j@random)-j@(h@random))) for j in generators]
        checkpoint('solved_'+name)
        print(json.dumps(dict(stage='solved',sector=name,result=results[name])),flush=True)
    reference=spectra['neutral'][1][:,0];stored['neutral_reference']=reference;reference_energy=float(spectra['neutral'][0][0]);source_results={};c_audits={}
    for name in ['lepton','baryon']:
        c=conjugation(model,name);cn=conjugation(model,'neutral');anti='anti'+name;e,v=spectra[name];av=c@v;spectra[anti]=(e,av);stored[anti+'_e']=e;stored[anti+'_v']=av
        results[anti]['ground_energy']=float(e[0]);results[anti]['eigen_residual']=float(np.linalg.norm(laws[anti]@av[:,0]-e[0]*av[:,0]))
        probes=v[:,:min(3,v.shape[1])];c_audits[name]=dict(H_on_poles=float(np.linalg.norm(laws[anti]@(c@probes)-c@(laws[name]@probes))),independent_source_C={})
        for placement in [0,1]+([(0,0,1)] if name=='baryon' else []):
            for spin in ([.5,1.5] if name=='baryon' else [.5]):
                key=str(placement)+':'+str(spin);s=source(model,name,placement,spin);sa=source(model,anti,placement,spin)
                error=float(np.linalg.norm(sa@(cn@reference)-c@(s@reference)));c_audits[name]['independent_source_C'][key]=error
        checkpoint('conjugated_'+name)
    for name in ['lepton','baryon','antilepton','antibaryon']:
        e,v=spectra[name];source_results[name]={};kind='baryon' if 'baryon' in name else 'lepton'
        for placement in [0,1]+([(0,0,1)] if kind=='baryon' else []):
            for spin in ([.5,1.5] if kind=='baryon' else [.5]):
                key=str(placement)+':'+str(spin);s=source(model,name,placement,spin);raw=s@reference;amplitudes=v.conj().T@raw;weights=abs(amplitudes)**2;bright=np.flatnonzero(weights>p['bright_tolerance'])
                prefix=name+'_source_'+key;save_sparse(stored,prefix+'_matter',s.matrix);stored[prefix+'_raw']=raw
                entry=dict(raw_retained_strength=float(np.linalg.norm(raw)**2),source_mean_energy=float(np.vdot(raw,laws[name]@raw).real/np.linalg.norm(raw)**2-reference_energy),low_gaps=(e-reference_energy).tolist(),low_raw_weights=weights.tolist(),outside_low_window_strength=float(np.linalg.norm(raw)**2-weights.sum()))
                if len(bright):
                    index=bright[0];pole=abs(e-e[index])<p['degeneracy_tolerance'];state=v[:,pole]@amplitudes[pole];state/=np.linalg.norm(state);stored[prefix+'_pole']=state
                    entry.update(lowest_computed_bright_gap=float(e[index]-reference_energy),pole_raw_strength=float(weights[pole].sum()),pole_J2=float(sum(np.linalg.norm(angular(model,name,axis)@state)**2 for axis in range(3))),pole_eigen_residual=float(np.linalg.norm(laws[name]@state-e[index]*state)))
                else:entry['lowest_computed_bright_gap']=None
                source_results[name][key]=entry
        results[name]['sources']=source_results[name];checkpoint('sources_'+name)
    results['C_audits']=c_audits;checkpoint('complete')
    print(json.dumps(dict(stage='complete',reference_energy=reference_energy,sources=source_results,C=c_audits,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)


if __name__=='__main__':main()
