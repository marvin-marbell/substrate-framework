"""One all-J factor compilation shared by current and interaction consumers.

This is data packaging for the already-required same-A response, not a second
physical construction or a new eigenspectrum experiment.
"""
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from rich_common_carrier_motion_20261015_model import build,law,fibre,source,observable,prepared_reference
from rich_common_carrier_motion_20261015_factors import compile
from rich_common_carrier_motion_20261015_operators import transverse_off,field_observable
from rich_common_carrier_motion_20261015_physical import audit
from rich_frame_field_join_20261015_calculate import m_isometry


def main():
    p=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);start=time.monotonic();motionpath=Path(p['own_motion_result']);motion=json.loads(motionpath.read_text())
    if 'first_discriminator' not in motion:raise ValueError('Actual first motion discriminator must precede all-J response compilation')
    model=build(p);result={'kind':'IMMUTABLE_CORRECTED_CAR36_ALL_J_FACTOR_AND_ORIGINAL_PREPARATION_SUPPLIER','parameters':p,'model_sha256':hashlib.sha256(Path('rich_common_carrier_motion_20261015_model.py').read_bytes()).hexdigest(),'motion_result_sha256':hashlib.sha256(motionpath.read_bytes()).hexdigest(),'snapshots':{},'sectors':{},'physical_source_audit':audit(model)};arrays={};vs=m_isometry(model,'neutral',0.);fresh=prepared_reference(model);arrays['neutral_prepared_t0']=vs.conj().T@fresh
    states=np.load(p['own_motion_states']);old=fibre(model,'neutral',0.,0.)@states['neutral_t0'];result['original_preparation']={'meaning':'EXACT original nonstationary occupied prepared reference at t0, NOT evolved source/response state','fresh_vs_actual2_original_L2':float(np.linalg.norm(fresh-old)),'M0_embedding_loss':float(np.linalg.norm(fresh-vs@arrays['neutral_prepared_t0'])),'norm_squared':float(np.linalg.norm(arrays['neutral_prepared_t0'])**2)}
    def checkpoint(stage):
        result['stage']=stage;out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**arrays);print(stage,flush=True)
    def store(label,operator,v,source_v=None):
        compiled,receipt=compile(operator,v,source_v);result['snapshots'][label]=compiled.save(out.with_name(out.name+'_'+label+'.pkl'));result['snapshots'][label]['compile']=receipt;return compiled
    checkpoint('CORRECTED_PHYSICAL_SOURCE_AUDIT_AND_T0_AVAILABLE')
    for name,M,J in [('neutral',0.,0.),('lepton',.5,.5),('antilepton',.5,.5)]:
        v=m_isometry(model,name,M);op=model['sectors'][name]['op'];result['sectors'][name]={'M':M,'dimension':v.shape[1],'angular_scope':'ALL compatible J in physical fixed M; no ground-J response restriction'}
        h=law(model,name);store(name+'_H_zero',h,v);store(name+'_H_full_transverse_off',transverse_off(model,name),v)
        test=fresh if name=='neutral' else source(model,name,0,.5)@fresh;vj=fibre(model,name,J,M);image=h@test
        result['sectors'][name]['same_H_prepared_state_outside_total_J_norm']=float(np.linalg.norm(image-vj@(vj.conj().T@image)))
        result['sectors'][name]['angular_check_scope']='Actual prepared source/reference H action, not a whole-space commutator theorem'
        for axis in [2,5]:
            store(name+'_Q'+str(axis),field_observable(model,name,'Q',axis),v);store(name+'_Pi'+str(axis),field_observable(model,name,'Pi',axis),v)
        store(name+'_field_HE',field_observable(model,name,'HE'),v);store(name+'_material_Lz',field_observable(model,name,'material_L',2),v);store(name+'_field_Fz',field_observable(model,name,'F',2),v)
        for i,a in enumerate(op['rho']):store(name+'_rho'+str(i),observable(model,name,matter=a),v)
        store(name+'_LEFT_z',observable(model,name,matter=op['S'][2]),v);store(name+'_matrix_r2',observable(model,name,power=2),v);store(name+'_colour_loop_B',observable(model,name,matter=op['B']),v)
        for axis in [2,5]:store(name+'_J'+str(axis)+'_zero',law(model,name,derivative=axis),v)
        if name!='neutral':
            for site in range(4):store(name+'_source_site'+str(site),source(model,name,site,.5),v,vs)
        original=states['full_neutral_t' if name=='neutral' else 'full_'+name+'_t'];arrays[name+'_moving_at_t025']=v.conj().T@(fibre(model,name,J,M)@original)
        checkpoint(name+'_ALL_J_FACTORS_COMPLETE')
    result['limits']=['Trusted local numeric pickle snapshots only, never load untrusted data','Only neutralM0 and chargedM+.5 exported here; no unexecuted M-.5/all-M exports claimed','Physical TEz/TMz probes preserve M, source m+.5; ALL J retained within these fibres','Original neutral t0 and evolved moving source at t.25 are DIFFERENT clearly named arrays','Full-transverse-OFF removes BOTH charged mechanical AND Peierls feedback, retains Coulomb/colour/radial/hops/HE; not bridge-off'];result['resource']={'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};checkpoint('NATURAL_COMPLETION')
if __name__=='__main__':main()
