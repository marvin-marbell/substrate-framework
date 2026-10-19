from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROPOSAL = Path(__file__).resolve().parents[1] / "proposals/P256-qed-backward"


@pytest.mark.parametrize("spacing,g2", [(1., 1e-6), (1e-8, .09)])
def test_entire_car_gap_resolves_supported_coupling_and_cutoff_scales(spacing: float, g2: float) -> None:
    code = """
import json, sys
import numpy as np
from original_charged_principal import original_frame
from bulk_frame_compton import reconstruct_frame
from original_quantum_mass_diagnostic import full_car_benchmark
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE
spacing,g2=map(float,sys.argv[1:])
bare=RotationBand(**{**BARE,'a':spacing,'e':1.})
_,frame=original_frame(bare,5,np.zeros(3),True)
band,unitary,_=reconstruct_frame(bare,frame)
result=full_car_benchmark(bare,band,frame,unitary,g2)
# Independent exact pi/y occupied-empty coefficients, not the solver's formula.
expected=dict(contact=g2*band.r/(16*spacing),
              virtual=g2*band.v**2/(16*spacing*(spacing*band.m+band.r+band.v)),
              Coulomb=g2*band.v/(16*spacing))
print(json.dumps({'actual':result['exact_components'],'expected':expected,
                  'gap':result['exact_gap_order_g2'],
                  'reference_self':result['opposite_static_reference_self'],
                  'adverse':result['adverse']}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(spacing), str(g2)], cwd=PROPOSAL,
        env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                          "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    for name, expected in result["expected"].items():
        assert result["actual"][name] == pytest.approx(expected, rel=2e-7, abs=0.)
    assert result["gap"] == pytest.approx(sum(result["expected"].values()), rel=2e-7, abs=0.)
    assert result["reference_self"] == pytest.approx(result["expected"]["Coulomb"], rel=1e-12, abs=0.)
    for name in ("missing_blocked_negative", "missing_contact", "missing_Coulomb_transition"):
        assert result["adverse"][name]["rejected_by_same_acceptance"]
