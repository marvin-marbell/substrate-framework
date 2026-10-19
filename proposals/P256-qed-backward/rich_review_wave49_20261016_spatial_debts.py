"""Own-state postprocessing: finite-cut canonical/Noether claims remain bounded."""
from pathlib import Path
import hashlib
import json
import numpy as np
import rich_spatial_frame_photon_20261016_model as law

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    source=ROOT/(P+'spatial_consumer_raw.npz');raw=np.load(source);records={};arrays={}
    for branch,number in (('addition',5),('removal',3)):
        m=law.Model(number);state=raw[branch+'_final'];h=m.apply(state);checks={}
        for axis in (0,4):
            pi=m.field(state,axis,'p');q=m.field(state,axis,'q')
            # These defects must not be silently identified with physical source.
            source_image=-m.vertex(state,axis)
            pidot=1j*(m.apply(pi)-m.field(h,axis,'p'))
            qdot=1j*(m.apply(q)-m.field(h,axis,'q'))
            defect=pidot-source_image;velocity=qdot-pi/m.parameters['cavity_inertia']
            checks[str(axis)]=dict(physical_source_image_norm=float(np.linalg.norm(source_image)),cut_Pi_commutator_norm=float(np.linalg.norm(pidot)),cut_Pi_vs_raw_source_debt=float(np.linalg.norm(defect)),cut_q_vs_Pi_over_inertia_debt=float(np.linalg.norm(velocity)))
            arrays[branch+f'_mode{axis}_source']=source_image;arrays[branch+f'_mode{axis}_cut_pidot']=pidot;arrays[branch+f'_mode{axis}_defect']=defect
        records[branch]=checks
    result=dict(kind='OWN_ACTUAL48_STATE_CANONICAL_REGULATOR_DEBT',raw_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),records=records,limits='Uses review49 own freshly evolved states only. Projected Pi/q commutators are not uncut canonical identities; raw original-A physical current stays independently defined. No exact global SO3 generator exists for the selected non-rotation-closed 8real q-set. This audit does not bound angular/radial/full field-H exterior or assert convergence.')
    np.savez_compressed(ROOT/(P+'spatial_debts_raw.npz'),**arrays)
    (ROOT/(P+'spatial_debts_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
