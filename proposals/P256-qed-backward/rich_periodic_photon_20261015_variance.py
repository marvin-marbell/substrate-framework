"""Exact full-H external residual of the new selected correlated references.

This is own-H operator action, not a borrowed trajectory or exact-ground claim.
All amplitudes, including coherent cancellations, enter the omitted-row sum.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from rich_periodic_photon_20261015_model import PREFIX, neighbors

ROOT = Path(__file__).resolve().parent


def main():
    previous = ROOT/(PREFIX+'full_H_variance.json')
    result = json.loads(previous.read_text()) if previous.exists() else {}
    for kind in ('hop1', 'loop_dressed', 'multi_pair4'):
        if kind in result:
            continue
        start = time.time()
        capsule = np.load(ROOT/(PREFIX+kind+'_state.npz'))
        keys = [int(k) for k in capsule['keys']]
        selected = set(keys)
        external = {}
        for key, amplitude in zip(keys, capsule['state']):
            for target, value, _, _ in neighbors(key):
                if target not in selected:
                    external[target] = external.get(target, 0.0)+value*float(amplitude)
        norm2 = float(sum(value*value for value in external.values()))
        result[kind] = dict(external_rows=len(external), full_H_external_residual_squared=norm2,
                            full_H_external_residual=norm2**.5,
                            meaning='Exact norm of Q H psi for Q=1-P. Add projected residual squared for full-H energy variance; not a gap bound.',
                            seconds=time.time()-start)
        print(json.dumps(dict(kind=kind,**result[kind])),flush=True)
        del external
    (ROOT/(PREFIX+'full_H_variance.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
