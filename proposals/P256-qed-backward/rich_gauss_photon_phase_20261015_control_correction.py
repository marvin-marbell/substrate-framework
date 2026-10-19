"""Correct ONLY control-current semantics from this consumer's saved endpoints.

No eigensolve or time evolution. Frozen raw run/capsule remains untouched.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy import sparse
from rich_open_gauss_carrier_20261015_model import NeutralCube
from rich_gauss_photon_phase_20261015_consume import measure, maximum, comm

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_gauss_photon_phase_20261015_'


def digest(name: str) -> str:
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def main() -> None:
    start = time.monotonic()
    names = [PREFIX+'control_correction.py',PREFIX+'consume.py',PREFIX+'actual_states.npz',
             PREFIX+'actual_result.json',PREFIX+'actual_exit.json','rich_open_gauss_carrier_20261015_model.py']
    pins = {name:digest(name) for name in names}
    model = NeutralCube()
    saved = np.load(ROOT/(PREFIX+'actual_states.npz'))
    if not np.array_equal(saved['old_indices'],model.old_indices):
        raise ValueError('Own saved trajectory basis mismatch')
    initial = saved['initial']
    cut_edges = np.concatenate((model.left_edges,model.right_edges))
    corrected = dict(off_before=measure(model,initial,model.Hoff,model.HRoff,()),
                     off_after=measure(model,saved['off_endpoint'],model.Hoff,model.HRoff,()),
                     cut_before=measure(model,initial,model.Hcut,model.HRcut,cut_edges),
                     cut_after=measure(model,saved['cut_endpoint'],model.Hcut,model.HRcut,cut_edges))
    algebra: dict[str,Any] = {}
    for name,h,edges in (('off',model.Hoff,()),('cut',model.Hcut,cut_edges)):
        motion,power = [],[]
        for e in range(12):
            E = sparse.diags(model.electric[:,e].astype(float),format='csr')
            J = model.currents[e] if e in edges else sparse.csr_matrix(model.H.shape,dtype=complex)
            motion.append(maximum(1j*comm(h,E)-J))
            power.append(maximum(1j*comm(h,E @ E/32)-(E @ J+J @ E)/32))
        algebra[name] = dict(own_flux_motion_max=max(motion),own_I16_electric_power_max=max(power))
    summary = {}
    for law in ('off','cut'):
        first,last = corrected[law+'_before'],corrected[law+'_after']
        summary[law] = dict(endpoint_delta_ER=last['ER'][1]-last['ER'][0],
                            endpoint_delta_own_HR=last['own_HR'][1]-last['own_HR'][0],
                            receiver_ER_excess_change=(last['ER'][1]-last['ER'][0])-(first['ER'][1]-first['ER'][0]),
                            receiver_own_HR_excess_change=(last['own_HR'][1]-last['own_HR'][0])-(first['own_HR'][1]-first['own_HR'][0]))
    result = dict(kind='CORRECTED_OWN_H_CONTROL_CURRENT_PROJECTION_NO_NEW_EVOLUTION',source_and_own_run_input_SHA256=pins,
                  sources_and_inputs_unchanged=all(digest(n)==p for n,p in pins.items()),
                  correction='Raw actual_result control energies were already own-H and correct. Raw control currents/J2/sym_EJ/right_powerER used full primitive-J probes even when hopping was removed; they are NOT own-H motion/power. This artifact supersedes all raw CONTROL observables with own-H active-edge currents. Full-H packet/spectral output unchanged.',
                  input_semantics='Only this consumer natural-exit own actual_states.npz initial/off_endpoint/cut_endpoint are reused as declared postprocessing INPUT. No old model/state/law splice, no regrounding, no borrowed evolution, no repeated producer/checker CLI.',
                  controls=corrected,summary=summary,exact_own_H_current_and_power_algebra=algebra,
                  elapsed_seconds=time.monotonic()-start)
    (ROOT/(PREFIX+'control_correction_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(kind=result['kind'],summary=summary,algebra=algebra,unchanged=result['sources_and_inputs_unchanged'])),flush=True)


if __name__ == '__main__':
    main()
