from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROPOSAL = Path(__file__).resolve().parents[1] / "proposals/P256-qed-backward"


def test_flux_selection_and_gap_are_covariant_under_energy_rescaling(tmp_path: Path) -> None:
    # Both Hamiltonian coefficients scale together: no flux preference can change.
    # The former absolute tolerance admitted all 1024 classes at the small scale;
    # the former absolute cubic residual rejected the legitimate large scale.
    results = []
    environment = os.environ | {
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"
    }
    scales = (1.0, 1e-8, 1e8)
    for scale in scales:
        receipt = tmp_path / f"flux-{scale:g}.json"
        subprocess.run(
            [sys.executable, "centre_flux_sea.py", "--heavy-scale", str(scale),
             "--K4", str(1e-4 * scale), "--output", str(receipt)],
            cwd=PROPOSAL, env=environment, check=True, capture_output=True, text=True,
        )
        results.append(json.loads(receipt.read_text(encoding="utf-8")))
    reference = results[0]
    assert reference["next_energy_gap"] > 0
    for scale, result in zip(scales[1:], results[1:]):
        assert result["minimizer_indices"] == reference["minimizer_indices"]
        assert result["next_energy_gap"] == pytest.approx(
            scale * reference["next_energy_gap"], rel=1e-8, abs=0
        )
        assert result["minimum_energy"] == pytest.approx(
            scale * reference["minimum_energy"], rel=1e-12, abs=0
        )
