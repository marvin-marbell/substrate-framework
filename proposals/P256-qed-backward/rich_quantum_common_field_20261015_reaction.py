"""Additional actual SAME-H quantum field event, reading only this branch's exact state capsule.
No G/D splice or old producer replay: the saved CSR whole H itself is propagated.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, gmres, LinearOperator
from rich_quantum_common_field_20261015_model import rotor, compressed, operators, law, vector_maps, signatures, signature_layers, energy_projection_residual, omitted_vector_norm, conjugate_sector
from rich_common_colour_lepton_20261015_model import scalar_field, gauss_operators, masks_for
from rich_colour_composite_20261014_model import loop_field
from rich_quantum_common_field_20261015_feedback import source_response, frozen_control, field_overflow, retarded_current
from rich_quantum_common_field_20261015_calculate import physical_joint_receiver, neutral_C, save_sparse
from rich_quantum_common_field_20261015_sources import compressed_source
from rich_quantum_common_field_20261015_boundary import omitted_population, full_source_strength


def load_sparse(data,prefix):
    return sparse.csr_matrix((data[prefix+'_data'],data[prefix+'_indices'],data[prefix+'_indptr']),shape=tuple(data[prefix+'_shape']))


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent
    statepath=root/request['state_npz'];resultpath=root/request['state_json']
    loaded=json.loads(resultpath.read_text());p=loaded['parameters'];data=np.load(statepath)
    scalar=scalar_field(p);loop=loop_field();field=rotor(p);size=len(field['electric']);nf=len(scalar['grid'])
    u=sparse.diags(np.ones(size-1),-1,shape=(size,size),format='csr');ux=sparse.kron(u,sparse.eye(size));uy=sparse.kron(sparse.eye(size),u)
    generator=(ux+ux.conj().T)/2+(uy-uy.conj().T)/(4j);results={};stored={}
    neutral_blocks=[];neutral_offset=0
    for i in range(loaded['results']['neutral']['retained_gauge_blocks']):
        prefix='neutral_block'+str(i);v=data[prefix+'_V'];projection=load_sparse(data,prefix+'_P')
        neutral_blocks.append(dict(sector=dict(masks=data[prefix+'_masks'].tolist(),P=projection),V=v,offset=neutral_offset,dimension=v.shape[1],signature=tuple(data[prefix+'_signature'])))
        neutral_offset+=v.shape[1]
    neutral=dict(blocks=neutral_blocks,dimension=neutral_offset)
    reflection=sparse.csr_matrix(np.fliplr(np.eye(field['size'])))
    cneutral=sparse.kron(neutral_C(neutral,scalar),reflection,format='csr')
    neutral_reference=data['neutral_stationary'];neutral_vectors=data['neutral_v'];neutral_energies=data['neutral_e']
    neutral_ops={key:sparse.kron(sparse.eye(neutral_offset),field['E'][axis],format='csr') for axis,key in enumerate(['E_x','E_y'])}
    neutral_ops.update(J_x=load_sparse(data,'neutral_J0'),J_y=load_sparse(data,'neutral_J1'))
    neutral_mode=dict(gaps=(neutral_energies-neutral_energies[0]).tolist(),raw_residues={key:(abs(neutral_vectors.conj().T@(op@neutral_reference))**2).tolist() for key,op in neutral_ops.items()},meaning='Computed low-window actual mixed whole-H poles with physical field/current residues. Positive bare compact electric gap alone is NOT an induced or travelling photon.')
    for name in ['lepton','baryon']:
        blocks=[];offset=0
        for i in range(loaded['results'][name]['retained_gauge_blocks']):
            prefix=name+'_block'+str(i);masks=data[prefix+'_masks'].tolist();projection=load_sparse(data,prefix+'_P');q,g=gauss_operators(masks,loop)
            sector=dict(masks=masks,P=projection,Q=q,G=g,dimension=projection.shape[1],raw_dimension=projection.shape[0],gauge_error=max(float(sparse.linalg.norm(generator@projection)) for generator in g))
            op=operators(sector,loop);h0=law(op,scalar,p)[0];v=data[prefix+'_V'];dimension=v.shape[1]
            blocks.append(dict(sector=sector,op=op,H0=h0,V=v,offset=offset,dimension=dimension,signature=tuple(data[prefix+'_signature'])))
            offset+=dimension
        sector=dict(blocks=blocks,dimension=offset)
        delta,spin=((0,0,0,1),1) if name=='lepton' else ((2,1,0,0),1)
        sector['omitted']=signature_layers({signatures(mask) for mask in masks_for(delta,spin)},p['population_depth'])[1]
        sector['anti']=False
        h=load_sparse(data,name+'_H');source=sparse.kron(compressed_source(neutral,sector,name,scalar),field['identity'],format='csr')
        old_vectors=data[name+'_v'];q,_=np.linalg.qr(old_vectors)
        energies,rotation=np.linalg.eigh(q.conj().T@(h@q));vectors=q@rotation
        raw=source@data['neutral_stationary'];weights=abs(vectors.conj().T@raw)**2
        first=np.flatnonzero(weights>p['bright_threshold'])[0];pole=abs(energies-energies[first])<1e-7
        state=vectors[:,pole]@(vectors[:,pole].conj().T@raw);state/=np.linalg.norm(state)
        estate=float(np.vdot(state,h@state).real)
        pole_evidence=dict(input_vector_Gram_error=float(np.linalg.norm(old_vectors.conj().T@old_vectors-np.eye(old_vectors.shape[1]))),orthonormal_Gram_error=float(np.linalg.norm(vectors.conj().T@vectors-np.eye(vectors.shape[1]))),source_gap=float(energies[first]-loaded['results']['neutral']['ground_energy']),low_raw_weights=weights.tolist(),eigen_residual=float(np.linalg.norm(h@state-estate*state)),state_difference_from_initial_consumer=float(np.linalg.norm(state-data[name+'_stationary'])))
        neutral['target_blocks']=blocks
        source_evidence=full_source_strength(neutral,name,data['neutral_stationary'],scalar,field,loop)
        source_evidence['retained_raw_strength']=float(np.linalg.norm(raw)**2)
        source_evidence['total_omitted_source_norm']=float(np.sqrt(max(0,source_evidence['full_unprojected_CAR_strength']-np.linalg.norm(raw)**2)))
        source_evidence['low_window_raw_strength']=float(weights.sum())
        source_evidence['outside_low_window_strength']=float(np.linalg.norm(raw)**2-weights.sum())
        source_evidence['complete_projected_G']=[]
        for frequency in p['source_frequencies']:
            matrix=(frequency+loaded['results']['neutral']['ground_energy']+1j*p['eta'])*sparse.eye(h.shape[0])-h
            diagonal=matrix.diagonal();preconditioner=LinearOperator(matrix.shape,matvec=lambda x:x/diagonal)
            solution,info=gmres(matrix,raw,M=preconditioner,rtol=1e-10,atol=1e-12,restart=100,maxiter=1000)
            value=np.vdot(raw,solution)
            source_evidence['complete_projected_G'].append(dict(frequency=frequency,real=float(value.real),imag=float(value.imag),solver_info=int(info),residual=float(np.linalg.norm(matrix@solution-raw))))
        kicked=expm_multiply(-1j*request['field_kick']*generator,state.reshape(offset,field['size']).T).T.ravel()
        obs={key:sparse.kron(compressed(sector,key,scalar),field['identity']) for key in ['phi','phi2','C','B','positive']}
        obs.update(J_x=load_sparse(data,name+'_J0'),J_y=load_sparse(data,name+'_J1'))
        baseline=[];reaction=[];base_states=[];pulse_states=[]
        sector['vector']=vector_maps(sector,scalar)
        for t in request['times']:
            base_state=expm_multiply(-1j*t*(h-estate*sparse.eye(h.shape[0])),state)
            pulse=expm_multiply(-1j*t*(h-estate*sparse.eye(h.shape[0])),kicked)
            base_states.append(base_state);pulse_states.append(pulse)
            baseline.append({key:float(np.vdot(base_state,op@base_state).real) for key,op in obs.items()})
            reaction.append({key:float(np.vdot(pulse,op@pulse).real) for key,op in obs.items()})
        results[name]=dict(source=source_response(source,state,kicked,pulse_states,base_states),excess={key:[r[key]-b[key] for r,b in zip(reaction,baseline)] for key in obs},frozen_paths=frozen_control(sector,state,kicked,scalar,field,p,request['times'][-1]),electric_overflow={label:field_overflow(sector,physical,scalar,field,p) for label,physical in [('stationary',state),('final',pulse_states[-1])]},norm_error=float(max(abs(np.linalg.norm(x)-1) for x in pulse_states)))
        results[name]['orthonormal_source_pole']=pole_evidence
        results[name]['full_original_positive_source']=source_evidence
        results[name]['signed_current_response']=retarded_current(sector,state,h,[obs['J_x'],obs['J_y']],[load_sparse(data,name+'_K0'),load_sparse(data,name+'_K1')],field,p)
        results[name]['joint_receiver']=[physical_joint_receiver(sector,x,scalar,field).tolist() for x in pulse_states]
        results[name]['joint_normalization_error']=float(max(abs(np.array(x).sum()-1) for x in results[name]['joint_receiver']))
        stored[name+'_stationary']=state
        results[name]['final_energy_projection_residual']=energy_projection_residual(sector,pulse_states[-1],scalar,field,p)
        results[name]['final_population_residual']=omitted_population(sector,pulse_states[-1],scalar,field,p)
        charge=sparse.kron(compressed(sector,'Q',scalar),field['identity'],format='csr');chosen=-1 if name=='lepton' else 1
        results[name]['charge_semantics']=dict(chosen_physical_Q=chosen,commutator_HQ=float(sparse.linalg.norm(h@charge-charge@h)),source_Q_Ward=float(sparse.linalg.norm(charge@source-chosen*source)),colour_Gauss=max(b['sector']['gauge_error'] for b in blocks))
        ca=sparse.kron(sparse.eye(offset),reflection,format='csr');anti_source=ca@source@cneutral.conj().T
        anti_blocks=[dict(b,sector=conjugate_sector(b['sector'],loop)) for b in blocks]
        independent_anti_source=sparse.kron(compressed_source(neutral,dict(blocks=anti_blocks),name,scalar,removal=True),field['identity'],format='csr')
        ah=load_sparse(data,'anti'+name+'_H');anti_state=ca@state
        results[name]['full_source_C']=dict(anti_retained_raw_strength=float(np.linalg.norm(anti_source@data['neutral_stationary'])**2),C_H=float(sparse.linalg.norm(ah@ca-ca@h)),anti_pole_residual=float(np.linalg.norm(ah@anti_state-estate*anti_state)),C_neutral_projection_unitarity=float(sparse.linalg.norm(cneutral.conj().T@cneutral-sparse.eye(cneutral.shape[0]))),source_C_identity=float(sparse.linalg.norm(anti_source@cneutral-ca@source)))
        results[name]['full_source_C']['independent_raw_removal_difference']=float(sparse.linalg.norm(independent_anti_source-anti_source))
        results[name]['full_source_C']['independent_anti_raw_strength']=float(np.linalg.norm(independent_anti_source@data['neutral_stationary'])**2)
        save_sparse(stored,name+'_full_source',source);save_sparse(stored,'anti'+name+'_full_source',anti_source)
        stored[name+'_states']=np.array(pulse_states)
        print(json.dumps(dict(sector=name,result=results[name])),flush=True)
    result=dict(kind='ACTUAL_SAME_BRANCH_SAVED_WHOLE_H_FIELD_BACKREACTION_NOT_ROOT_SPLICE',input_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [statepath,resultpath]},request=request,neutral_mixed_modes=neutral_mode,results=results)
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)


if __name__=='__main__':main()
