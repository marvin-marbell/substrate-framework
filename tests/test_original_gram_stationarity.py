from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROPOSAL = Path(__file__).resolve().parents[1] / "proposals/P256-qed-backward"


@pytest.mark.parametrize("spacing", [1., .001])
def test_full_native_gram_is_stationary_for_every_centre(spacing: float) -> None:
    # The N5 (0,0,2) solve stalled at gradient1.62e-9: its true Newton
    # descent5.16e-19 was below the1.11e-16 objective subtraction resolution.
    # Check the independent original field consumer, not solver metadata.
    code = """
import itertools, json, sys
import numpy as np
from original_charged_principal import OriginalUniform, original_frame
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis
from unrestricted_field_minimum import BARE, coordinates
band=RotationBand(**{**BARE,'a':float(sys.argv[1])})
basis=matrix_unit_basis()
maximum=0.
for label in itertools.product(range(4), repeat=3):
    angles=np.pi/2*np.asarray(label)
    _, frame=original_frame(band,5,angles,True)
    model=OriginalUniform(band,5,basis,angles)
    _, gradient=model.evaluate(coordinates(frame,basis).reshape(-1))
    maximum=max(maximum,band.a*float(np.linalg.norm(gradient)))
angles=np.asarray([0.,0.,np.pi])
_, imported=original_frame(band,5,angles,False)
_, wrong_gradient=OriginalUniform(band,5,basis,angles).evaluate(
    coordinates(imported,basis).reshape(-1))
print(json.dumps({'stationarity_error':maximum,
    'unshifted_import_error':band.a*float(np.linalg.norm(wrong_gradient))}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(spacing)], cwd=PROPOSAL,
        env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                          "MKL_NUM_THREADS": "1"},
        check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    assert result["stationarity_error"] < 1e-9
    # An unshifted/cubic import is not stationary on this actual native grid.
    # This wrong construction reaches and fails the same physical criterion.
    assert result["unshifted_import_error"] > 1e-6
