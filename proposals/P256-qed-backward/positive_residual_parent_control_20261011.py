"""Exact noncommuting all-vector Galerkin residual control, not native phase evidence."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any, cast

import sympy as sp


def equal(a: Any, b: Any) -> bool:
    return all(sp.simplify(z) == 0 for z in a - b)


def psd(a: Any) -> bool:
    assert equal(a, a.H)
    return all(cast(Any, sp.factor(a.extract(rows, rows).det())) >= 0
               for size in range(1, a.rows + 1)
               for rows in combinations(range(a.rows), size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    import hashlib
    root = Path(__file__).resolve().parent
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == pin
               for name, pin in request['pins'].items())
    K = sp.diag(0, 2, 5)
    X = sp.Matrix([[8, 1, 0, 1], [1, 6, 1, 0], [0, 1, 7, 1], [1, 0, 1, 9]])
    J = sp.Matrix([[1, 2, 0], [0, 1, 1], [2, 0, 1], [1, -1, 2]])
    G = J.H * J
    a = sp.Rational(7)
    assert not equal(G*K, K*G)
    T = X*J - J*K - a*J
    B = (J.H*T + T.H*J)/2
    rows = []
    for E in (sp.Rational(-1), sp.Rational(0), sp.Rational(1, 4)):
        R = (X-E*sp.eye(4)).inv()
        F = (K+(a-E)*sp.eye(3)).inv()
        W = J.H*R*J
        Y = J*F
        variational = Y.H*J + J.H*Y - Y.H*(X-E*sp.eye(4))*Y
        galerkin = (G*F+F*G)/2 - F*B*F
        residual = F*T.H*R*T*F
        assert equal(variational, galerkin)
        assert equal(W, galerkin+residual)
        assert psd(residual) and psd(W-galerkin)
        # X>=4I independently by exact principal minors: bound residual as
        # an OPERATOR before passing to the excitation energy form.
        assert psd(X-4*sp.eye(4))
        upper = galerkin + F*T.H*T*F/(4-E)
        assert psd(upper-W)
        Q = sp.Matrix([[0, 0], [1, 0], [0, 1]])
        w0 = W[0, 0]
        C = W-w0*sp.eye(3)
        lo = Q.H*(galerkin-w0*sp.eye(3))*Q
        hi = Q.H*(upper-w0*sp.eye(3))*Q
        assert psd(Q.H*C*Q-lo) and psd(hi-Q.H*C*Q)
        wrong_order = (G*F+F*G)/2 - B*F*F
        assert not equal(wrong_order, wrong_order.H)
        assert not equal(wrong_order+residual, W)
        assert not equal(galerkin, W)  # Positive residual is not zero.
        rows.append({'E': str(E), 'exact_W': [[str(z) for z in W.row(i)] for i in range(3)],
                     'Galerkin_lower_and_positive_remainder': True,
                     'whole_operator_upper': True, 'entire_Q_energy_form': True,
                     'wrong_commuting_order_rejected': True,
                     'omitted_positive_residual_rejected': True})
    # One-sided connected stability cannot silently become an absolute
    # Mayer contraction: -5K is bounded above by K but fails |C|<=K.
    Kq = sp.diag(2, 5)
    Cq = -5*Kq
    assert psd(Kq-Cq) and not psd(Kq+Cq)
    result = {'kind': 'EXACT_ALLVECTOR_POSITIVE_RESIDUAL_METHOD_CONTROL_NOT_NATIVE_PHASE',
              'identity': 'W=Sym(GF)-F Sym(Jdag T) F+F Tdag R T F',
              'rows': rows, 'one_sided_not_absolute_contraction': True,
              'native_g_or_physical_ground_computed': False,
              'physical1_stiffness_or_photon': False, 'campaign_complete': False}
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == pin
               for name, pin in request['pins'].items())
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({'noncommuting_identity': True, 'positive_residual': True,
                      'all_vector_two_sided_enclosure': True, 'native_phase': False}))


if __name__ == '__main__':
    main()
