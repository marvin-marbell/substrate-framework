"""ONE full primitive-H residual per actual charged/reference state.
Exact output count-sector slicing bounds memory, not the physical state space.
Every reachable hopping count and full generated local Fock image is retained.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import product
from pathlib import Path
import gc,json,resource,sys,time
import numpy as np
from scipy import sparse
from scipy.linalg import solve
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_cloud_probe import LocalElectric,neutral_state,inner,add
from rich_composite_geometry_20261015_science import load


def local_metric(primitive):
    maximum=0.
    for indices in primitive.local.blocks.values():
        keys=sorted({key for column in indices for key in primitive.local.vectors[column]});lookup={key:row for row,key in enumerate(keys)}
        matrix=np.zeros((len(keys),len(indices)),complex)
        for column,index in enumerate(indices):
            for key,value in primitive.local.vectors[index].items():matrix[lookup[key],column]=value
        maximum=max(maximum,float(np.linalg.norm(matrix.conj().T@matrix-np.eye(len(indices)))))
    return {'full_local16_Fock_Gram_max_block_Frobenius':maximum,'four_site_tensor_Gram_operator_upper':float(np.expm1(4*np.log1p(maximum))),
        'meaning':'Physical local Fock embedding metric only, NOT a full-H range/gap bound.'}


def reconstruct(lookup,coordinate):
    result={}
    for key,(columns,fields) in lookup.items():
        value=np.tensordot(coordinate[columns],fields,axes=(0,0))
        if np.any(value):result[key]=value
    return result


def residual(primitive,state,energy,scalar,rotor,out,name,W_lookup=None,W_gram=None):
    p=primitive.p;groups={}
    for key,field in state.items():groups.setdefault(primitive.label(key)[:-1],{})[key]=field
    incoming=defaultdict(list)
    for counts in groups:
        for edge,(start,end) in enumerate(model.EDGES):
            for flavour,iswhite,dagger in product(range(2),(False,True),(False,True)):
                category=2*flavour+int(iswhite);origin=end if dagger else start;destination=start if dagger else end
                if counts[4*origin+category]==0 or counts[4*destination+category]>=(2 if iswhite else 6):continue
                target=list(counts);target[4*origin+category]-=1;target[4*destination+category]+=1
                incoming[tuple(target)].append((counts,edge,flavour,iswhite,dagger))
    targets=sorted(set(groups)|set(incoming));weights=np.asarray(primitive.graph['weights']);axes=np.asarray(primitive.graph['axes'])
    inverse=np.asarray(primitive.graph['inertia_inverse'],float);generators=angular.right_angular(rotor);kinetic=np.zeros((10,10),complex)
    for a,b in np.ndindex(3,3):kinetic+=inverse[a,b]*(generators[a]@generators[b]).toarray()/2
    rotations={(edge,i):sum((axes[edge,a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for edge,i in product(range(4),range(3))}
    electric=LocalElectric(primitive);component_names=('field','electric','plaquette','quark_hop','white_hop','minus_E_state')
    total_gram=np.zeros((6,6),complex);rows=[];dual=np.zeros(W_gram.shape[0],complex) if W_gram is not None else None
    total_norm2=0.;source_support_norm2=0.;W_count_norm2=0.;mean=0.+0j;maximum_rows=0;phi_density=np.zeros((p['scalar_points'],)*2,complex)
    rotor_strength=np.zeros(10);pw_strength=np.zeros(19);calls=0;started=time.time();W_counts={primitive.label(key)[:-1] for key in W_lookup} if W_lookup is not None else set()
    for index,counts in enumerate(targets):
        components={key:{} for key in component_names}
        for key,field in groups.get(counts,{}).items():
            add(components['field'],key,1.,kinetic@field+field@scalar['H'].T)
            add(components['minus_E_state'],key,-energy,field)
            for target,value in electric.action(key).items():add(components['electric'],target,value,field)
            for pw,value in primitive.field_maps['B'][key[-1]]:add(components['plaquette'],key[:-1]+(pw,),-p['plaquette']*value,field)
        for source_counts,edge,flavour,iswhite,dagger in incoming.get(counts,[]):
            selected=components['white_hop' if iswhite else 'quark_hop']
            for key,field in groups[source_counts].items():
                transformed=[p['scalar_coupling']*weights[edge]*field*scalar['grid'][None,:]]
                transformed.extend(1j*p['vector_velocity']*weights[edge]*(rotations[edge,i]@field) for i in range(3))
                for component in range(4):
                    image=primitive.hop({key:1.+0j},edge,flavour,iswhite,component,dagger);calls+=1
                    factor=-1 if component and dagger else 1
                    for target,value in image.items():add(selected,target,factor*value,transformed[component])
        matrix=np.zeros((6,6),complex)
        for a in range(6):
            for b in range(a,6):
                value=inner(components[component_names[a]],components[component_names[b]])
                matrix[a,b]=value;matrix[b,a]=np.conjugate(value)
        total_gram+=matrix
        combined={}
        for component in components.values():
            for key,value in component.items():add(combined,key,1.,value)
        norm2=float(inner(combined,combined).real);on_source=sum(float(np.linalg.norm(value)**2) for key,value in combined.items() if key in state)
        total_norm2+=norm2;source_support_norm2+=on_source
        if counts in W_counts:W_count_norm2+=norm2
        mean+=inner(state,combined);maximum_rows=max(maximum_rows,len(combined))
        for key,field in combined.items():
            phi_density+=field.T@field.conj();rotor_strength+=np.sum(abs(field)**2,axis=1);pw_strength[key[-1]]+=float(np.linalg.norm(field)**2)
            if dual is not None:
                present=W_lookup.get(key)
                if present is not None:
                    columns,bra=present;dual[columns]+=np.einsum('brf,rf->b',bra.conj(),field)
        c=np.asarray(counts).reshape(4,4);colour=c[:,0]+c[:,2];white=c[:,1]+c[:,3]
        rows.append({'counts':counts,'residual_norm2':norm2,'raw_output_rows':len(combined),'same_initial_raw_key_support_norm2':on_source,
            'inside_W_material_counts':(counts in W_counts) if W_lookup is not None else None,'ncolour':colour.tolist(),'nwhite':white.tolist(),
            'regional_quark_deficit_relative6':int(np.maximum(6-colour,0).sum()),'regional_white_deficit_relative2':int(np.maximum(2-white,0).sum()),
            'classification':'Regional occupation deficits, NOT negative-band pair count or tagged-particle position.'})
        Path(out+'_'+name+'_checkpoint.json').write_text(json.dumps({'processed_output_count_sectors':index+1,'all_reachable_count_sectors':len(targets),
            'accumulated_full_residual_norm2':total_norm2,'maximum_one_output_sector_rows':maximum_rows,'primitive_hop_calls':calls,'rows':rows},indent=2)+'\n')
        print(json.dumps({'stage':'ONE_FULL_PRIMITIVE_H_RANGE_CLUE','state':name,'count_sector':index+1,'count_sectors':len(targets),'sector_rows':len(combined),'sector_residual_norm2':norm2,'accumulated_norm2':total_norm2}),flush=True)
        del components,combined;gc.collect()
    norm=float(inner(state,state).real);projected=None
    if dual is not None:projected=float(np.vdot(dual,solve(W_gram,dual,assume_a='her')).real)
    result={'state':name,'energy_used':energy,'raw_state_norm2':norm,'full_primitive_residual_norm2':total_norm2,
        'normalized_residual_RMS_energy':float(np.sqrt(total_norm2/norm)),
        'mean_residual_real':float(mean.real/norm),'mean_residual_imag':float(mean.imag/norm),
        'full_H_energy_variance':float(total_norm2/norm-abs(mean/norm)**2),
        'projected_W_residual_norm2':projected,
        'outside_W_residual_norm2_signed':total_norm2-projected if projected is not None else None,
        'inside_W_material_counts_residual_norm2':W_count_norm2 if W_lookup is not None else None,
        'outside_W_material_counts_residual_norm2':total_norm2-W_count_norm2 if W_lookup is not None else None,
        'same_initial_raw_key_support_norm2':source_support_norm2,'new_raw_key_support_norm2':total_norm2-source_support_norm2,
        'largest_count_channels':sorted(rows,key=lambda value:value['residual_norm2'],reverse=True)[:20],
        'component_names':component_names,'primitive_hop_calls':calls,'input_raw_rows':len(state),'input_count_sectors':len(groups),
        'all_reachable_output_count_sectors':len(targets),'maximum_one_output_sector_rows':maximum_rows,
        'elapsed_seconds':time.time()-started,
        'interpretation':'ONE actual FULL primitive H action, sliced by ALL reachable exact material counts solely to bound output memory. No native/W projection or physical sector dropped. Phi7/R10/PW19 primitive cuts remain. minus_E_state is the residual definition with the PREEXISTING candidate eigenenergy, NOT an additive Hamiltonian mass/recenter. Count/channel and component interference data distinguish escaping sea/field support. RMS is NOT an exponential decay rate or compact-particle lifetime.'}
    np.savez(out+'_'+name+'_residual_data.npz',component_Gram=total_gram,Phi_density=phi_density,SO3_strength=rotor_strength,PW_strength=pw_strength,W_dual=dual if dual is not None else np.zeros(0,complex))
    Path(out+'_'+name+'_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    basis=load(config['W_basis_snapshot']);primitive=basis['primitive'];p=basis['p'];lookup=basis['lookup'];labels=basis['labels']
    summary=json.loads(Path(config['W_result_snapshot']).read_text());j=min((1,3),key=lambda value:summary['sectors'][str(value)]['minimum_energy'])
    eigen=np.load(config[f'W_J{j}_states_snapshot']);keep=np.flatnonzero(np.asarray([row['twice_J'] for row in labels])==j)
    coordinate=np.zeros(len(labels),complex);coordinate[keep]=eigen['vectors'][:,0];candidate=reconstruct(lookup,coordinate);energy=float(eigen['energies'][0])
    with np.load(config['W_matrices_snapshot']) as matrices:gram=matrices['Gram']
    scalar,rotor=model.field_and_rotor(p)
    result={'kind':'ONE_ACTUAL_FULL_PRIMITIVE_RANGE_CLUE_FOR_BEST_W_CANDIDATE_AND_OWN_REFERENCE','parameters':p,
        'candidate_selection':'Lowest ACTUAL charged W energy across both required spin sectors, NOT receiver/halfspin fitted. Ground density at1/2 is not called moving COM.','chosen_twice_J':j}
    result['charged']=residual(primitive,candidate,energy,scalar,rotor,out,'charged',lookup,gram)
    del candidate,lookup,basis;gc.collect()
    neutral=load(config['neutral_basis_snapshot']);coupling=load(config['neutral_coupling_snapshot']);states=np.load(config['neutral_states_snapshot'])
    omega=neutral_state(neutral,coupling,states['states'][:,0]);reference_energy=float(states['energies'][0]);del neutral,coupling,states;gc.collect()
    result['neutral_reference']=residual(primitive,omega,reference_energy,scalar,rotor,out,'neutral_reference')
    result['local_physical_metric']=local_metric(primitive)
    result['limits']='One state/range clue, not an N-uniform theorem or full continuum stability. Charged/reference RMS values live in different N sectors and MUST NOT be subtracted as a decay rate/mass correction. Projected unitary compactness alone is insufficient; large escaping field/sea channels require a MATERIAL ansatz change. No all-basis cloud storage, selective field return, numerical expectation clamp, spin/force fit or issue closure.'
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
