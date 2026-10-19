from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROPOSAL = Path(__file__).resolve().parents[1] / "proposals/P256-qed-backward"


@pytest.mark.parametrize("spacing", [1., .001])
def test_temporal_green_matches_entire_car_path_and_rejects_wrong_kernels(spacing: float) -> None:
    # This is the conditional Wick kernel before canonical/Gauss projection,
    # not a grandcanonical substitute for the physical neutral ground.
    code = """
import json, sys
import numpy as np
from scipy.linalg import expm
from original_charged_principal import original_frame
from original_uniform_charge_gap import (
    exterior_hamiltonian, green_kernel_controls, temporal_green,
    time_evolution, tree_onebody,
)
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE
band=RotationBand(**{**BARE,'a':float(sys.argv[1])})
_, frame=original_frame(band,5,np.asarray([np.pi/2]*3),True)
hs=[tree_onebody(frame,band,.47),tree_onebody(frame,band,-.69)]
a0=np.asarray([[.11,.25],[-.32,.09]])/band.a
durations=band.a*np.asarray([.23,.31])
u,prefixes,_=time_evolution(hs,a0,durations)
kernels=temporal_green(u,prefixes)
green_kernel_controls(u,prefixes,kernels)

# Independent many-body time evolution on EVERY one of the 2^8 CAR states.
# Temporal site phases are number operators in each exact exterior sector.
bases={}; before={}; after={}; total={}
for n in range(9):
    h1,basis=exterior_hamiltonian(hs[0],n)
    h2,_=exterior_hamiltonian(hs[1],n)
    bases[n]=basis
    phases=[np.asarray([sum(np.repeat(a,4)[j] for j in range(8) if state & (1<<j))
                        for state in basis]) for a in a0]
    before[n]=expm(durations[0]*(-h1+1j*np.diag(phases[0])))
    after[n]=expm(durations[1]*(-h2+1j*np.diag(phases[1])))
    total[n]=after[n]@before[n]
partition=sum(np.trace(total[n]) for n in range(9))
correlation=np.zeros((8,8),dtype=complex)
for n in range(8):
    lookup={state:i for i,state in enumerate(bases[n+1])}
    creators=[]
    for mode in range(8):
        c=np.zeros((len(bases[n+1]),len(bases[n])),dtype=complex)
        for col,state in enumerate(bases[n]):
            if not state & (1<<mode):
                c[lookup[state | (1<<mode)],col]=(-1)**((state & ((1<<mode)-1)).bit_count())
        creators.append(c)
    for i in range(8):
        for j in range(8):
            correlation[i,j]+=np.trace(after[n]@creators[i].conj().T@before[n+1]@creators[j])
correlation/=partition
error=float(np.linalg.norm(correlation-kernels[1]))

# The constant identity formerly satisfied the native loose decay checks.
# The constant identity violates the source jump; reversed multiplication
# satisfies that jump but violates evolution in this actual noncommuting path.
wrong_identity=[np.eye(8,dtype=complex)]*len(prefixes)
wrong_order=[kernels[0]@prefix for prefix in prefixes]
rejections=[]
for wrong in (wrong_identity,wrong_order):
    try:
        green_kernel_controls(u,prefixes,wrong)
    except AssertionError:
        rejections.append(True)
    else:
        raise AssertionError('non-Green kernel accepted')
print(json.dumps({'full_CAR_correlation_error':error,'rejections':rejections,
                 'noncommuting_path':float(np.linalg.norm(hs[0]@hs[1]-hs[1]@hs[0]))}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(spacing)], cwd=PROPOSAL,
        env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                          "MKL_NUM_THREADS": "1"},
        check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    assert result["full_CAR_correlation_error"] < 1e-10
    assert result["noncommuting_path"] > 1e-4
    assert result["rejections"] == [True, True]
