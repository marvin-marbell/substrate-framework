"""Selected raw CAR/PW/R source and primitive checks, never1328-block assembly."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_cloud_probe import LocalElectric,neutral_state,inner,add
from rich_composite_geometry_20261015_source_eom import source_operator,bare_state
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_runtime import primitive_for
ROOT=Path(__file__).resolve().parent
P='rich_native_body_review_20261016_'
RUN='rich_composite_geometry_20261015_correlated_dressing3'


def column_state(block,column=0):
    vector=block.matrix[:,column].toarray().ravel();state={}
    for key,value in zip(block.keys,vector):
        if value:
            body,r=key[:-1],key[-1]
            if body not in state:state[body]=np.zeros(10,complex)
            state[body][r]+=value
    return state


def project(block,state,nphi=None):
    shape=(len(block.keys),) if nphi is None else (len(block.keys),nphi)
    vector=np.zeros(shape,complex)
    for key,field in state.items():
        for r in range(10):
            row=block.lookup.get(key+(r,))
            if row is not None:vector[row]+=field[r]
    return block.matrix.conj().T@vector


def apply_J(primitive,state,rotor,axis):
    out={};left=sparse.csr_matrix(rotor['L'][axis]).toarray()
    for key,field in state.items():
        add(out,key,1.,left@field)
        for target,value in primitive.spin({key:1.+0j},axis).items():add(out,target,value,field)
    return out


def audit(meta,ops):
    storage=meta['storage'];primitive=meta['primitive'];scalar,rotor=model.field_and_rotor(meta['p'])
    source_label=next(iter(storage.meta));j,counts=source_label;changed=list(counts);changed[0]-=1;changed[4]+=1;target_label=(j,tuple(changed))
    source_block=storage.load_readonly(source_label);target_block=storage.load_readonly(target_label)
    state=column_state(source_block);source_column=source_block.columns[0];cols=source_block.columns;target_cols=target_block.columns
    gram=(source_block.matrix.conj().T@source_block.matrix).toarray();errors={'independent_body_Gram':float(np.linalg.norm(gram-ops['Gram'][cols][:,cols].toarray()))}
    generators=angular.right_angular(rotor);inverse=np.asarray(primitive.graph['inertia_inverse']);kinetic=sum((inverse[a,b]*(generators[a]@generators[b]).toarray()/2 for a,b in np.ndindex(3,3)),np.zeros((10,10),complex))
    field_image={key:kinetic@field for key,field in state.items()};electric_image={};plaquette={};electric=LocalElectric(primitive)
    for key,field in state.items():
        for target,value in electric.action(key).items():add(electric_image,target,value,field)
        for pw,value in primitive.field_maps['B'][key[-1]]:add(plaquette,key[:-1]+(pw,),-meta['p']['plaquette']*value,field)
    raw={}
    for name,image in [('rotation',field_image),('electric',electric_image),('plaquette',plaquette)]:
        value=project(source_block,image);expected=ops[name][cols,source_column].toarray().ravel();raw[name]=value;errors[name]=float(np.linalg.norm(value-expected))
    scalar_image={};vector_image={};axis=np.asarray(primitive.graph['axes'])[0];weight=primitive.graph['weights'][0]
    rotations=[sum((axis[a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for i in range(3)]
    for key,field in state.items():
        for component in range(4):
            image=primitive.hop({key:1.+0j},0,0,False,component,False)
            transformed=meta['p']['scalar_coupling']*weight*field if component==0 else 1j*meta['p']['vector_velocity']*weight*(rotations[component-1]@field)
            selected=scalar_image if component==0 else vector_image
            for target,value in image.items():add(selected,target,value,transformed)
    for name,image in [('quark_scalar',scalar_image),('quark_vector',vector_image)]:
        value=project(target_block,image);expected=ops[name][target_cols,source_column].toarray().ravel();raw[name]=value;errors['tree01_forward_u_'+name]=float(np.linalg.norm(value-expected))
    angular_receipts={}
    for spin in (1,3):
        label=next(label for label in storage.meta if label[0]==spin);block=source_block if spin==1 else storage.load_readonly(label);v=column_state(block);j2={}
        for axis_index in range(3):
            image=apply_J(primitive,apply_J(primitive,v,rotor,axis_index),rotor,axis_index)
            for key,field in image.items():add(j2,key,1.,field)
        eigen=spin*(spin+2)/4
        for key,field in v.items():add(j2,key,-eigen,field)
        angular_receipts[str(spin)]=dict(J2_eigenvalue=eigen,raw_residual_norm2=float(inner(j2,j2).real),raw_norm2=float(inner(v,v).real),body_block=storage.paths[label])
    config=json.loads((ROOT/(RUN+'_input.json')).read_text());neutral=load(ROOT/config['neutral_basis_snapshot']);pr=primitive_for(neutral);coupling=load(ROOT/config['neutral_coupling_snapshot']);eigen=np.load(ROOT/config['neutral_states_snapshot'])
    if any(vector!=primitive.local.vectors[i] for i,vector in enumerate(pr.local.vectors)):raise AssertionError('Neutral/native canonical local dictionary prefix mismatch')
    omega=neutral_state(neutral,coupling,eigen['states'][:,0]);bare=bare_state(primitive,omega,source_operator(0,'half_triplet',1,False));probe=next(probe for probe in meta['probes'] if probe['descriptor']==dict(twice_J=1,spec=([0,0,0],'half_triplet',1),family='bare'))
    raw_source=project(source_block,bare,7);expected_source=ops['Gram'][cols][:,cols]@probe['coordinates'][cols]
    errors['actual_unmasked_source_covector_selected_block']=float(np.linalg.norm(raw_source-expected_source))
    result=dict(selected_blocks=[storage.paths[source_label],storage.paths[target_label]],selected_primitive_column=source_column,independent_errors=errors,angular=angular_receipts,actual_source_raw_strength=float(inner(bare,bare).real),declared_probe_strength=probe['raw_strength'],scope='Only selected source/count blocks and one raw primitive column. Phi scalar/vector/E/R retained before projection. No1328-block reconstruction, full-H range/lifetime bound or independent full-neutral ground.')
    if max(errors.values())>2e-9:raise AssertionError(('Native primitive/source mismatch',result))
    if max(row['raw_residual_norm2'] for row in angular_receipts.values())>1e-16:raise AssertionError(('Chosen physical angular representation mismatch',angular_receipts))
    np.savez(ROOT/(P+'primitive_raw.npz'),Gram=gram,source_covector=raw_source,declared_source_covector=expected_source,**raw)
    (ROOT/(P+'primitive_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(stage='selected_raw_primitive_audit',result=result)),flush=True)
    return result
