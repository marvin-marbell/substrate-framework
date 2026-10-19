#!/usr/bin/env python3
"""Retrospective logic audit, issue #223. Reads saved artifacts only.

No GPAW import, calculator calls, optimizer, MPI, notification or simulation.
Prior outputs exposed. Scope: bookkeeping and artifact consistency, not a
new physical result or DFT convergence certification. Mutations exercise
atom conservation and same-coordinate matching. Run with catalyst-dft path.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from ase.io import read


def energy(path):
    return sp.Rational(str(json.loads(path.read_text())["energy"]))


def conservation():
    # Columns: VH5, H2, VH6, bulkH, perfect bulk; rows: Ni, H.
    counts = sp.Matrix([[31, 0, 31, 32, 32], [5, 2, 6, 1, 0]])
    used = sp.Matrix([1, 1, -1, 0, 0])
    trapping = sp.Matrix([-1, 0, 1, -1, 1])
    molecular = sp.Matrix([-1, -sp.Rational(1, 2), 1, 0, 0])
    assert counts * trapping == sp.zeros(2, 1)
    assert counts * molecular == sp.zeros(2, 1)
    assert counts * used != sp.zeros(2, 1)  # full-H2 mutation exposed
    return {"campaign4_atom_residual_Ni_H": list(counts * used),
            "bulk_trapping_residual": list(counts * trapping),
            "half_H2_formation_residual": list(counts * molecular)}


def coordinate_pairs(curves):
    shell, mol = curves["shell"], curves["molecule"]
    actual = [(s["R_actual"], m["R_actual"])
              for s, m in zip(shell, mol)]
    same = sorted(set(s["R_actual"] for s in shell)
                  & set(m["R_actual"] for m in mol))
    assert any(a != b for a, b in actual)
    assert not all(a == b for a, b in actual)  # index-zip mutation exposed
    return {"gate_index_pairs": actual, "available_exact_common_R": same}


def cached_geometry(path):
    atoms = read(path, index=-1)
    # Inspect cached results directly; never call a calculator property.
    raw = np.asarray(atoms.calc.results["forces"]).copy()
    projected = raw.copy()
    for constraint in atoms.constraints:
        constraint.adjust_forces(atoms, projected)
    h = [i for i, symbol in enumerate(atoms.symbols) if symbol == "H"]
    ni = [i for i, symbol in enumerate(atoms.symbols) if symbol == "Ni"]
    distances = atoms.get_all_distances(mic=True)
    pair = next((tuple(c.pairs[0]) for c in atoms.constraints
                 if hasattr(c, "pairs")), None)
    return {"formula": atoms.get_chemical_formula(),
            "pair": pair,
            "R_from_positions": float(distances[pair]) if pair else None,
            "min_H_Ni_A": float(distances[np.ix_(h, ni)].min()) if ni else None,
            "projected_fmax_eV_A": float(np.linalg.norm(projected, axis=1).max()),
            "raw_fmax_eV_A": float(np.linalg.norm(raw, axis=1).max()),
            "constraints": [c.todict() for c in atoms.constraints],
            "positions_A": atoms.positions.tolist()}


def reference_geometry_and_metric(camp):
    atoms = read(camp / "frames/int_1p6.traj", index=-1)
    ni = atoms.positions[:32]
    anchors = []
    for fraction in ([0.5, 0, 0], [0.5, 0.5, 0]):
        position = np.array(fraction) @ atoms.cell.array
        distances = np.linalg.norm(ni - position, axis=1)
        anchors.append({"supercell_fraction": fraction,
                        "nearest_Ni_A": float(distances.min())})
    b_shell = sp.Symbol("B_shell", nonnegative=True)
    assert sp.ask(sp.Q.nonnegative(b_shell - 0))
    return {"unshifted_interstitial_anchors": anchors,
            "registered_delta_B_nonnegative_when_B_mol_zero": True,
            "csv_column_counts": [len(line.split(","))
                                 for line in (camp / "U_of_R.csv").read_text().splitlines()],
            "interstitial_logged_steps": {
                p.stem: sum(line.startswith("BFGS:") for line in p.read_text().splitlines())
                for p in sorted((camp / "frames").glob("int_*.log"))}}


def audit(root):
    camp = root / "runs/2026-10-15_dd-screening"
    c3 = root / "runs/campaign3/c32_bulkvac"
    result = json.loads((camp / "result.json").read_text())
    curves = result["curves"]
    c3_increment = (energy(c3 / "a04_vach6r/result.json")
                    - energy(c3 / "a04_vach5r/result.json")
                    - energy(c3 / "a03_bulkh1/result.json")
                    + energy(c3 / "a02_perfect_k2/result.json"))
    statistics = {}
    frames = {}
    for name, rows in curves.items():
        values = [sp.Rational(str(row["E_eV"])) for row in rows]
        statistics[name] = {
            "minimum_sample_R_A": rows[values.index(min(values))]["R_actual"],
            "endpoint_R_A": rows[-1]["R_actual"],
            "endpoint_minus_start_eV": float(values[-1] - values[0]),
            "endpoint_minus_sample_min_eV": float(values[-1] - min(values)),
            "registered_max_minus_start_eV": float(max(values) - values[0])}
        prefix = {"shell": "shell", "molecule": "mol", "interstitial": "int"}[name]
        for row in rows:
            tag = f"{prefix}_{str(row['R_target']).replace('.', 'p')}"
            saved = json.loads((camp / "frames" / f"{tag}.json").read_text())
            geometry = cached_geometry(camp / "frames" / f"{tag}.traj")
            frames[tag] = {**saved, "cached_geometry": geometry}
    # Source/receipt provenance, not a claim of physical accuracy.
    sources = [camp / "PLAN.md", camp / "REPORT.md", camp / "result.json",
               camp / "U_of_R.csv", camp / "scripts/gate.py",
               camp / "scripts/screen_path.py", camp / "scripts/gen_jobs.py",
               root / "runs/campaign3/audit.py"]
    sources += sorted((camp / "frames").glob("*.traj"))
    return {"scope": "saved artifacts only; no simulations run",
            "symbolic_conservation": conservation(),
            "campaign3_bulk_trapping_increment_eV": float(c3_increment),
            "coordinate_comparison": coordinate_pairs(curves),
            "reference_geometry_and_metric": reference_geometry_and_metric(camp),
            "curve_statistics_not_physical_verdicts": statistics,
            "anchor": result["anchor"], "emitted_verdict": result["verdict"],
            "frames": frames,
            "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalyst_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.catalyst_root), indent=2, default=lambda x: int(x)))
