"""Selected incoming-zero flux Feshbach boundary; not a phase/pole supplier."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
PINS = {
    'analytic_flux_time_actual3_20261011_receipt.json': '346458add630716bb0d71a70c0b6d38026736a040a8e33792393b9649767d58c',
    'charged_cell_heat_interface_20261011.json': '41d524542ded29c183426646bbdecf2f34b613bee7dcd13bf98e2a427c9c330f',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve(matrix: list[list[F]], rhs: list[F]) -> list[F]:
    """Exact Gaussian elimination for the independent finite Schur control."""
    a = [row[:] + [value] for row, value in zip(matrix, rhs)]
    n = len(rhs)
    for j in range(n):
        pivot = next(i for i in range(j, n) if a[i][j])
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [v / scale for v in a[j]]
        for i in range(n):
            if i != j:
                scale = a[i][j]
                a[i] = [v - scale * w for v, w in zip(a[i], a[j])]
    return [row[-1] for row in a]


def schur_controls() -> list[dict[str, Any]]:
    controls = []
    for z in (F(0), F(64)):
        lam = F(1, 4)
        a, b, c = (F(v) + z for v in (2, 3, 5))
        hq = [[a, lam, F(0)], [lam, b, lam], [F(0), lam, c]]
        # P has two endpoints; PQ couples endpoint0 to Q0 and endpoint1 to Q2.
        # H_eff(-z) = PHP - PHQ (QHQ + z)^-1 QHP.
        column = solve(hq, [F(0), F(0), lam])
        actual_offdiag = -lam * column[0]
        determinant = a * b * c - lam * lam * (a + c)
        independent_formula = -lam**4 / determinant
        assert actual_offdiag == independent_formula < 0
        fourth_order = -F(1) / (a * b * c)
        # R0 V R0 V R0 gives the unique Q0-Q1-Q2 path. Outer Schur minus remains.
        neumann_path = -F(1, 1) / a / b / c
        assert fourth_order == neumann_path
        wrong_positive = -fourth_order
        assert wrong_positive != fourth_order
        omitted_middle = -F(1) / a / c
        assert omitted_middle != fourth_order
        controls.append({
            'positive_resolvent_energy_shift_exact': str(z),
            'lambda_exact': str(lam),
            'full_finite_schur_offdiag_exact': str(actual_offdiag),
            'fourth_order_coefficient_exact': str(fourth_order),
            'wrong_positive_Schur_sign_rejected': True,
            'wrong_omitted_middle_resolvent_rejected': True,
            'scope': 'Independent five-state P2/Q3 finite method control, not native spectral data.',
        })
    return controls


def box(row: dict[str, Any], key: str) -> tuple[F, F]:
    lo, hi = row[key]['exact']
    pair = F(lo), F(hi)
    assert pair[0] <= pair[1]
    return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    assert {n: sha(ROOT / n) for n in PINS} == PINS
    receipt = json.loads((ROOT / next(iter(PINS))).read_bytes())
    heat = json.loads((ROOT / 'charged_cell_heat_interface_20261011.json').read_bytes())
    assert F(heat['source_norm_squared_exact']) >= 1
    assert F(heat['rigorous_residual_exact']) == F(1, 65536)
    assert F(heat['formation_included_in_residual_exact']) == F(1, 262144)
    # Every proper subset of a distinct square has nonzero divergence.
    # Hence it leaves the neutral product-ground/closed-external-flow P_S space.
    # This does not replace any intermediate neutral four-point propagator by P0.
    subset_divergences = []
    for mask in range(1, 15):
        div = tuple(int(bool(mask & (1 << v))) - int(bool(mask & (1 << ((v - 1) % 4)))) for v in range(4))
        assert any(div)
        subset_divergences.append({'mask': mask, 'divergence': div})
    orientation_groups: dict[str, Any] = {}
    for z in ('0', '64'):
        rows = receipt['orders']
        for role, copies in (('active', 1), ('aux', 63)):
            for orientation in (-1, 1):
                chosen = [r for r in rows if r['role'] == role and r['orientation'] == orientation]
                assert {tuple(r['order']) for r in chosen} == set(itertools.permutations(range(4)))
                assert all(r['count'] == copies for r in chosen)
                value = {key: tuple(sum((copies * box(r['values'][z]['alltime_coefficient_interval'], key)[j] for r in chosen), F(0)) for j in (0, 1)) for key in ('re', 'im')}
                orientation_groups[f'{z}/{role}/{orientation}'] = {k: [str(x) for x in v] for k, v in value.items()}
        for key in ('re', 'im'):
            actual = tuple(sum((r['count'] * box(r['values'][z]['alltime_coefficient_interval'], key)[j] for r in rows), F(0)) for j in (0, 1))
            assert actual == box(receipt['coefficient_intervals'][z], key)
    c0 = box(receipt['coefficient_intervals']['0'], 're')
    c64 = box(receipt['coefficient_intervals']['64'], 're')
    assert c0[0] > c64[1] > 0
    separation = c0[0] - c64[1]
    result = {
        'kind': 'SELECTED_FLUX_ENERGY_DEPENDENCE_AND_FESHBACH_SIGN_PARENT_COMPOSITION',
        'own_source_sha256': sha(Path(__file__)),
        'input_sha256': PINS,
        'positive_resolvent_energy_shift_semantics': 'z is in each positive denominator H0-Eground+z. It is NOT photon frequency, momentum or an inferred pole.',
        'selected_C0_minus_C64_real_lower_exact': str(separation),
        'selected_C0_minus_C64_real_lower_display': float(separation),
        'energy_independent_selected_fourhop_coefficient_rejected': True,
        'formal_fourth_order_Feshbach_term': '-C(z), because H_eff(E)=PHP-PHQ(QHQ-E)^-1QHP; E=Eground-z.',
        'formal_minusC_real_intervals_exact': {z: [str(-box(receipt['coefficient_intervals'][z], 're')[1]), str(-box(receipt['coefficient_intervals'][z], 're')[0])] for z in ('0', '64')},
        'projection_boundary': 'P_S is the frozen neutral product g times external divergence-free integer-flow space; Q_S=1-P_S. H0 reduces it. Every proper square-edge subset has nonzero divergence, so all three intermediate Q_S act identically on this selected four-hop word. This is not the entire ground block of the physical interacting lattice.',
        'proper_subset_exact_divergences': subset_divergences,
        'orientation_role_intervals_exact': orientation_groups,
        'independent_finite_Schur_controls': schur_controls(),
        'source_norm_cap_binding': 'THIS pinned raw source norm squared >=1 makes eta(1+sqrt(rawN)) at least 2eta. The heat moments use normalized s; rawN is not a source adoption or a ground norm.',
        'nonclaims': ['No all-path or all-order Feshbach operator assembled.', 'No energy-independent magnetic Hamiltonian adopted.', 'No incoming nonzero-flow value inferred.', 'No physicalPhi/lattice ground, stiffness, physical1 phase, massless IR atom, pole, commoncone or originalSOURCE admission.'],
        'elapsed_seconds': time.monotonic() - started,
    }
    assert {n: sha(ROOT / n) for n in PINS} == PINS
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('kind', 'selected_C0_minus_C64_real_lower_exact', 'selected_C0_minus_C64_real_lower_display', 'formal_minusC_real_intervals_exact', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
