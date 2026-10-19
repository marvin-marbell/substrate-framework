"""Exact five-state vacuum-subtraction boundary control; not native phase data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import sympy as s


def exact_matrix(matrix: Any) -> list[list[str]]:
    return [[str(s.factor(matrix[i, j])) for j in range(matrix.cols)] for i in range(matrix.rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    root = Path(__file__).resolve().parent
    assert all(hashlib.sha256((root / n).read_bytes()).hexdigest() == h for n, h in request['pins'].items())
    z = s.symbols('z', positive=True)
    hp = s.diag(0, 2)
    hq = s.diag(3, 3, 7)
    j = s.Matrix([[6, 0], [0, 6], [s.Rational(1, 2), s.Rational(1, 2)]])
    h = hp.row_join(j.T).col_join(j.row_join(hq))
    w = j.T * (hq + z * s.eye(3)).inv() * j
    vacuum = w[0, 0]
    connected = w - vacuum * s.eye(2)
    schur = hp + z * s.eye(2) - w
    normalized_schur = hp + (z - vacuum) * s.eye(2) - connected
    assert schur == normalized_schur
    assert connected[1, 1] == 0
    assert s.simplify(connected[0, 1] - 1 / (4 * (7 + z))) == 0
    derivative = schur.diff(z)
    metric = s.eye(2) + j.T * (hq + z * s.eye(3)).inv() ** 2 * j
    assert s.simplify(derivative - metric) == s.zeros(2)
    rows = []
    for value in (5, 64):
        inverse = (h + value * s.eye(5)).inv()
        selected = inverse.extract([0, 1], [0, 1])
        svalue = schur.subs(z, value)
        assert selected == svalue.inv()
        assert svalue[0, 0] > 0 and svalue.det() > 0
        dropped = s.diag(svalue[0, 0], svalue[1, 1]).inv()
        wrong_frozen = (hp + value * s.eye(2) - w.subs(z, 0)).inv()
        assert selected[0, 1] != dropped[0, 1]
        assert selected != wrong_frozen
        assert svalue.det() - svalue[0, 0] * svalue[1, 1] == -svalue[0, 1] ** 2 < 0
        rows.append({'positive_resolvent_shift': value, 'whole_H_plus_shift_positive_by_Schur': True,
                     'exact_projected_resolvent': exact_matrix(selected),
                     'drop_connected_offdiagonal_defect_exact': str(s.factor(selected[0, 1])),
                     'frozen_energy_kernel_is_wrong': True,
                     'missing_offdiagonal_secular_term_exact': str(s.factor(-svalue[0, 1] ** 2))})
    result = {'kind': 'EXACT_VACUUM_CONNECTED_ENERGY_DEPENDENT_SCHUR_METHOD_BOUNDARY_CONTROL',
              'method_only_not_native': True, 'H_exact': exact_matrix(h),
              'P_dimension': 2, 'Q_dimension': 3, 'coupling_parameter': 1,
              'vacuum_scalar': str(s.factor(vacuum)), 'connected_operator': exact_matrix(connected),
              'scalar_subtraction_identity': 'S(z)=K+(z-w0(z))I-(W(z)-w0(z)I)',
              'connected_QP_offdiagonal_retained': True,
              'zero_excited_diagonal_is_not_zero_connected_operator': True,
              'absolute_W0_exceeds_reference_gap': bool(w.subs(z, 0)[0, 0] > 2),
              'shift_derivative_metric_exact': exact_matrix(metric),
              'derivative_sign': 'W_prime(z)=-Jdag(HQ+z)^-2 J; S_prime=I-W_prime, not I+W_prime',
              'finite_positive_shift_controls': rows,
              'source_or_ground_or_physical_law_not_adopted': True,
              'stationary_native_ground_or_physical1_phase_or_pole': False,
              'z_is_resolvent_energy_shift_not_photon_frequency': True,
              'campaign_complete': False, 'elapsed_seconds': time.monotonic() - started}
    assert all(hashlib.sha256((root / n).read_bytes()).hexdigest() == h for n, h in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'five_state_exact_identity': True, 'missing_offdiagonal_rejected': True,
                      'energy_independent_kernel_rejected': True, 'native_phase': False}), flush=True)


if __name__ == '__main__':
    main()
