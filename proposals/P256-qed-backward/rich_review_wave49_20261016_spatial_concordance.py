"""Check only wrapper/typing-invalidated API layer against own preserved physical states."""
from pathlib import Path
import hashlib
import json
import numpy as np
import rich_spatial_frame_photon_20261016_model as law
from rich_review_wave49_20261016_spatial_consumer import observe,evolve,external_action

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    source=ROOT/(P+'spatial_consumer_raw.npz');raw=np.load(source)
    previous=json.loads((ROOT/(P+'spatial_consumer_result.json')).read_text());records={}
    for branch,number in (('addition',5),('removal',3)):
        m=law.Model(number);state=raw[branch+'_final'];observed=observe(m,state);expected=previous['records'][branch]['final']
        energy=abs(observed['energy']-expected['energy'])
        current=float(np.max(abs(np.array(observed['physical_current_qin_qout'])-expected['physical_current_qin_qout'])))
        direct=external_action(m,state,np.zeros(8));action=float(np.linalg.norm(direct-m.apply(state)))
        step=2e-4;direction=np.eye(8)[0]
        plus=external_action(m,state,step*direction);minus=external_action(m,state,-step*direction)
        first=(plus-minus)/(2*step);second=(plus+minus-2*direct)/step**2
        vertex=m.vertex(state,0);contact=m.vertex(state,0,0)
        first_error=float(np.linalg.norm(first-vertex)/max(np.linalg.norm(vertex),1e-14))
        second_error=float(np.linalg.norm(second-contact)/max(np.linalg.norm(contact),1e-14))
        assert first_error<1e-6 and second_error<1e-4,(first_error,second_error)
        next_state=evolve(m,state,.0001)
        assert energy<1e-11 and current<1e-11 and action<1e-10,(energy,current,action)
        records[branch]=dict(energy_concordance_error=energy,physical_source_concordance_error=current,independent_minimal_source_base_action_error=action,new_wrapper_path_norm2=float(np.vdot(next_state,next_state).real))
        records[branch].update(current_geometry_rebuild_FD_relative=first_error,current_geometry_rebuild_contact_relative=second_error)
    result=dict(kind='CURRENT48_APPLY_API_CONCORDANCE_NO_PHYSICAL_LAW_RERUN',raw_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),records=records,limits='Current corrected-cubature own states. Only wrapper/typing/Gaunt-factorization/private-cache layers are compared here; physical matrices/kinetic products are unchanged on this corrected grid. Includes independent geometry source FD after private hopping cache removal, current apply and own LinearOperator path. Existing full physical evolution remains bound to exact executed capsules, not relabeled with new bytes.')
    (ROOT/(P+'spatial_concordance_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
