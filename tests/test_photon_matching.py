from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROPOSAL = Path(__file__).resolve().parents[1] / "proposals/P256-qed-backward"


def test_physical_weak_coupling_pole_and_current_remain_valid() -> None:
    # A real positive subthreshold pole can have arbitrarily small sea/contact
    # effects. The old absolute root and minimum-effect gates rejected it.
    code = """
import json
from bulk_frame_compton import actual_sea_pole, reconstruct_frame
from photon_matched_compton import forward_at_pole
from photon_matching import match_photon
from rotation_band import RotationBand
from sea_selection import SELECTED
bare = RotationBand(a=.001, m=.1, r=.06, v=.1, e=1., improved=False)
band, _, _ = reconstruct_frame(bare, SELECTED)
pole = match_photon(8, band, electric_coefficient=3e-8, magnetic_coefficient=3e10)
raw = actual_sea_pole(8, bare, SELECTED, band, pole)
response = forward_at_pole(band, raw)
print(json.dumps(dict(pole=pole, raw=raw, response=response)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=PROPOSAL,
        env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                          "MKL_NUM_THREADS": "1"},
        check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    pole, raw, response = result["pole"], result["raw"], result["response"]
    assert 0 < pole["photon_energy"] < pole["pair_threshold"]
    assert pole["link_residue_inverse"] > pole["lagrangian_epsilon"]
    assert pole["relative_pole_residual"] < 1e-10
    assert all(pole["checks"].values())
    assert all(raw["checks"].values())
    assert all(response["checks"].values())
    assert response["pauli_forward_squared"] == pytest.approx(
        response["independent_forward_squared"], rel=1e-10, abs=0,
    )
    # Reach the branches rejected by both former minimum-effect gates.
    assert response["no_contact_ward_residual"] < 1e-8
    assert abs(response["bare_source_squared"] / response["pauli_forward_squared"] - 1) < 1e-4
    # Small is not zero: dropping the contact fails this relative Ward check,
    # even though its absolute defect would pass the former correctness gate.
    assert response["ward_residual"] < 1e-8 * response["no_contact_ward_residual"]


@pytest.mark.parametrize("electric_coefficient", [1e-6, 1e-12])
def test_contact_dominated_current_keeps_a_valid_certified_pole(electric_coefficient: float) -> None:
    # Weak nonlinear dressing formerly lost the root bracket; a small but
    # genuine virtual-pair correction also failed the universal size gate.
    code = """
import json
import sys
from bulk_frame_compton import actual_sea_pole, reconstruct_frame
from photon_matched_compton import forward_at_pole
from photon_matching import match_photon
from rotation_band import RotationBand
from sea_selection import SELECTED
electric = float(sys.argv[1])
bare = RotationBand(a=1., m=.1, r=.1, v=.001, e=1., improved=False)
band, _, _ = reconstruct_frame(bare, SELECTED)
pole = match_photon(8, band, electric_coefficient=electric,
                    magnetic_coefficient=1e-6/electric)
raw = actual_sea_pole(8, bare, SELECTED, band, pole)
print(json.dumps(forward_at_pole(band, raw)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(electric_coefficient)], cwd=PROPOSAL,
        env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                          "MKL_NUM_THREADS": "1"},
        check=True, capture_output=True, text=True,
    )
    response = json.loads(completed.stdout)
    pole = response["photon_pole"]
    assert 0 < pole["photon_energy"] < pole["pair_threshold"]
    assert pole["link_residue_inverse"] >= pole["lagrangian_epsilon"] > 0
    assert pole["rigorous_pole_lower"] <= pole["photon_energy"] <= pole["low_frequency_pole_upper"]
    assert all(response["checks"].values())
    assert response["pauli_forward_squared"] == pytest.approx(
        response["independent_forward_squared"], rel=1e-10, abs=0,
    )
    assert abs(response["no_virtual_pairs_squared"] / response["pauli_forward_squared"] - 1) < .01
    assert response["ward_residual"] < 1e-8 * response["no_contact_ward_residual"]


def test_actual_approximate_frame_pole_can_be_delivered_as_json() -> None:
    code = """
import json
from pathlib import Path
from bulk_frame_compton import actual_sea_pole, read_bulk, reconstruct_frame
from photon_matched_compton import forward_at_pole
from photon_matching import match_photon
bare, frame, _ = read_bulk(Path('periodic_bulk_frame_receipt.json'),
                          'environmental_Clifford_L3')
band, _, _ = reconstruct_frame(bare, frame)
reduced = match_photon(16, band, electric_coefficient=.4, magnetic_coefficient=.7)
raw = actual_sea_pole(16, bare, frame, band, reduced)
print(json.dumps(forward_at_pole(band, raw)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=PROPOSAL,
        env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                          "MKL_NUM_THREADS": "1"},
        check=True, capture_output=True, text=True,
    )
    response = json.loads(completed.stdout)
    pole = response["photon_pole"]
    assert 0 < pole["photon_energy"] < pole["pair_threshold"]
    assert all(pole["checks"].values())
    assert response["pauli_forward_squared"] == pytest.approx(
        response["independent_forward_squared"], rel=1e-10, abs=0,
    )
