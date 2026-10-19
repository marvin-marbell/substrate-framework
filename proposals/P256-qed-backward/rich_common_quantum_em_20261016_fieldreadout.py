"""Actual colour/Phi/frame observers on saved NEW-H rich source states.
No trajectory replay or foreign solved-state splice. Unchanged primitive weak
observables are reused in the identical isometries, with their actual Gram.
"""
from pathlib import Path
import json,pickle,sys,time,resource,gc
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as primitive_model
import rich_composite_geometry_20261015_angular as angular
from rich_common_quantum_em_20261016_operator import Model


def observers(model,prepared,sector,out):
    inverse=sparse.load_npz(prepared+'_inversehalf.npz')
    if sector=='neutral':
        old='rich_composite_geometry_20261015_correlated_onehop_reference1'
        with open(old+'_J0_coupling.pkl','rb')as stream:coupling=pickle.load(stream)
        raw={name:sparse.load_npz(old+'_op_'+key+'.npz')for name,key in(('colour_ReTrU','B'),('colour_loop_right_Casimir','C'),('colour_electric_energy','electric'))};weak={name:angular.project_product(coupling,value,sparse.eye(10))for name,value in raw.items()};scalar,rotor=primitive_model.field_and_rotor(model.body['parameters']);k=angular.right_angular(rotor);metric=np.asarray(primitive_model.geometry(model.body['parameters'])['inertia_inverse']);kinetic=sum((metric[a,b]*(k[a]@k[b])/2 for a,b in np.ndindex(3,3)),sparse.csr_matrix((10,10),dtype=complex));weak['material_frame_kinetic_energy']=angular.project_product(coupling,sparse.eye(raw['colour_ReTrU'].shape[0]),kinetic)
    else:
        old='rich_common_native_species_20261016_joint1';weak={name:sparse.load_npz(old+'_op_'+key+'.npz')for name,key in(('colour_ReTrU','loopB'),('colour_loop_right_Casimir','loopC'),('colour_electric_energy','electric'),('material_frame_kinetic_energy','rotation'))}
    result={name:(inverse@value@inverse).tocsr()for name,value in weak.items()}
    for name,value in result.items():sparse.save_npz(out+'_'+sector+'_observer_'+name+'.npz',value)
    return result


def expectations(model,matrices,state):
    x=state.reshape(model.n,len(model.grid),len(model.flux));flat=x.reshape(model.n,-1);result={name:float(np.vdot(flat,value@flat).real)for name,value in matrices.items()};result['Phi_energy_actual_quantum_kinetic_plus_negative_input_potential']=float(np.einsum('npm,pq,nqm->',x.conj(),model.hphi,x,optimize=True).real);return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=config['out'];run=config['dynamics_prefix'];started=time.time();records={};sources={};timings=[(0,0.),(4,1.)]
    for sector,prepared in(('neutral',config['neutral_matter_prefix']),('pair',config['joint_matter_prefix'])):
        model=Model(prepared);matrices=observers(model,prepared,sector,out);records[sector]={};sources[sector]={'observer_Hermiticity':{name:float(np.linalg.norm((value-value.conj().T).data))for name,value in matrices.items()},'basis_dimension':model.n}
        for law in('full_live_quantum_links','TRANSVERSE_link_identity_active_matter_source_control'):
            records[sector][law]={}
            for suffix in('','_white'):
                for index,t in timings:
                    path=run+'_'+sector+'_'+law+suffix+'_t'+str(index)+'_states.npz'
                    with np.load(path)as data:
                        states=data['states'];labels=data['labels']
                        for column,label in enumerate(labels):records[sector][law].setdefault(str(label),[]).append({'time':t,**expectations(model,matrices,states[:,column])})
                    del states;gc.collect();print(json.dumps({'stage':'actual_saved_NEW_state_colour_Phi_frame_readout','sector':sector,'law':law,'pulse_suffix':suffix,'time':t}),flush=True)
        del model,matrices;gc.collect()
    differences={}
    for law in records['pair']:
        differences[law]={}
        for label in('field_kick_y','field_kick_z','white_pulse'):
            rows=[]
            for index in range(2):
                row={'time':timings[index][1]}
                for name in records['pair'][law][label][index]:
                    if name=='time':continue
                    pp=records['pair'][law][label][index][name]-records['pair'][law]['baseline'][index][name];nn=records['neutral'][law][label][index][name]-records['neutral'][law]['baseline'][index][name];row[name]=pp-nn
                rows.append(row)
            differences[law][label]=rows
    result={'kind':'SAME_NEW_NATIVE_STATE_ACTUAL_COLOUR_LOOP_PHI_FRAME_RESPONSE','records':records,'added_neutral_pair_source_response':differences,'observables':sources,'meaning':'ReTrU_colour and SU3 loop-right Casimir are NOT charge flux or NoetherL. Colour electric is original full-before-target colour-square energy; new charge longitudinal/transverse energies are separately in dynamics result. Quantum Phi energy retains primitive negative potential. Fieldkick-to-colour/Phi/frame response has same-age paired and neutral baselines; transverse-identity control retains all other rich interactions. No allcolourOFF, binding or unique-mediator inference.','old_states_or_energies_used':False,'resources':{'seconds':time.time()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'actual_rich_quantum_field_readout_complete',**result['resources']}),flush=True)


if __name__=='__main__':main()
