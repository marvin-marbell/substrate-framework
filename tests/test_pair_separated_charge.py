from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROPOSAL = Path(__file__).resolve().parents[1] / "proposals/P256-qed-backward"


@pytest.mark.parametrize("stiffness", [2.0, 4.0])
def test_large_lambda_kinetic_bound_retains_its_positive_growth(stiffness: float) -> None:
    # At these scales the trial excess rounds away in a total energy. Its
    # positive kinetic bound must still grow with the derived Lambda^(5/6)
    # law; recovering it from rounded total energies instead gives a constant.
    code = """
import json
import sys
import numpy as np
from full_band_gauss_section import FullBandGaussSection
from pair_full_field_concentration import constants
from pair_separated_charge_transfer import finite_bound
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE, ALIGNMENT, pair_clifford_reference
stiffness = float(sys.argv[1])
model = FullBandGaussSection(RotationBand(**BARE), heavy_scale=1.)
frame, _ = pair_clifford_reference(model, stiffness=stiffness)
law = constants(model, np.tile(frame, (2, 1, 1, 1)), stiffness=stiffness,
                alignment=ALIGNMENT, ue4=.3, ue1=.4, magnetic=.7, chart_radius=.001)
rows = [finite_bound(L, law, stiffness=stiffness, alignment=ALIGNMENT,
                     ue4=.3, ue1=.4) for L in (1e108, 1e120)]
print(json.dumps(rows))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(stiffness)], cwd=PROPOSAL,
        env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                          "MKL_NUM_THREADS": "1"},
        check=True, capture_output=True, text=True,
    )
    lower, upper = json.loads(completed.stdout)
    growth = (upper["Lambda"] / lower["Lambda"]) ** (5 / 6)
    assert upper["exact_neutral_total_kinetic_upper"] == pytest.approx(
        lower["exact_neutral_total_kinetic_upper"] * growth, rel=1e-8, abs=0,
    )
