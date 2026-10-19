"""Independent exact NONNATIVE matrix controls, not a native ground backend."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import sympy as sp


def clean(value: Any) -> Any:
    if isinstance(value, sp.MatrixBase):
        return value.applyfunc(sp.simplify)
    return sp.simplify(value)


def encode(value: Any) -> Any:
    if isinstance(value, sp.MatrixBase):
        return [[str(clean(value[i, j])) for j in range(value.cols)] for i in range(value.rows)]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, sp.Basic):
        return str(clean(value))
    return value


def matrix_control() -> dict[str, Any]:
    q = sp.symbols('q', positive=True)
    H = sp.Matrix([[2, sp.I, 1, 0], [-sp.I, 3, 0, 1], [1, 0, 4, sp.I], [0, 1, -sp.I, 5]])
    # Positive leading principal minors establish a positive-energy toy band.
    minors = [cast(Any, H[:k, :k]).det() for k in range(1, 5)]
    assert all(x > 0 for x in minors)
    W, U = sp.eye(4)[:, :2], sp.eye(4)[:, 2:]
    T = sp.Matrix([[1, sp.I/2], [-sp.I/2, 1]])
    X = sp.sqrt(q)*W*T
    S = clean(X.H*X)
    support_inverse_root = T.inv()/sp.sqrt(q)
    assert clean(X*support_inverse_root) == W
    L, K = q*H, H**2
    f = X/q
    e = sp.sqrt(q)*sp.Matrix([[1, 0], [0, 2], [1, 1], [0, 1]])/128
    j = -sp.I*L*e
    d = 128*e-2*sp.I*L*f
    force = 64*q*j+sp.I*q*L*d/2
    assert clean(force-L**2*X) == sp.zeros(4, 2)
    assert clean((L**2-q**2*sp.eye(4))*X-(force-q**2*X)) == sp.zeros(4, 2)
    current = clean(support_inverse_root*X.H*(64*q*j)*support_inverse_root/q**2)
    seam = clean(support_inverse_root*X.H*(sp.I*q*L*d/2)*support_inverse_root/q**2)
    M = clean(W.H*K*W)
    assert clean(current+seam) == M and M == M.H
    assert current != current.H and seam != seam.H
    R = sp.eye(4)-W*W.H
    B = clean(U.H*force*support_inverse_root/q**2)
    bath = clean(U.H*K*U)
    assert B == clean(U.H*K*W)
    z = -1
    memory = clean(B.H*(z*sp.eye(2)-bath).inv()*B)
    schur = clean((z*sp.eye(2)-M-memory).inv())
    projected = clean(W.H*(z*sp.eye(4)-K).inv()*W)
    assert schur == projected
    wrong_total_propagator = clean(U.H*(z*sp.eye(4)-K).inv()*U)
    wrong_memory = clean(B.H*wrong_total_propagator*B)
    assert wrong_memory != memory
    assert clean((z*sp.eye(2)-M-wrong_memory).inv()) != projected
    entrywise_real = M.applyfunc(sp.re)
    polarization = sp.Matrix([1, sp.I])
    full_quadratic = clean((polarization.H*M*polarization)[0])
    real_quadratic = clean((polarization.H*entrywise_real*polarization)[0])
    assert full_quadratic != real_quadratic
    assert clean(S/q) != sp.eye(2) and current != M
    assert clean(R*force/q**2) == sp.sqrt(q)*U*B*T
    # Absolute force vanishes as sqrt(q), while its normalized force B does not.
    assert B != sp.zeros(2)
    return {'positive_H_minors': minors, 'S_over_q': clean(S/q), 'Gram_eigenvalues': list((T**2).eigenvals()),
            'M': M, 'current_nonHermitian': current, 'parallel_seam_nonHermitian': seam,
            'B': B, 'compressed_bath': bath, 'irreducible_memory': memory,
            'wrong_total_return_memory': wrong_memory, 'projected_resolvent': projected,
            'complex_u_full_value': full_quadratic, 'wrong_entrywise_real_value': real_quadratic,
            'full_identity_zero': True, 'source_returns_rejected_at_resolvent': True,
            'unnormalized_force_small_does_not_make_B_small': True}


def adverse_boundaries() -> dict[str, Any]:
    q = sp.symbols('q', positive=True)
    W = sp.eye(4)[:, :2]
    T = sp.Matrix([[1, sp.I/2], [-sp.I/2, 1]])
    H = sp.diag(2, 2, 3, 4)
    X = sp.sqrt(q)*W*T
    M = W.H*H**2*W
    B = (sp.eye(4)-W*W.H)*H**2*W
    residual = clean(((q*H)**2-q**2*sp.eye(4))*X/q**sp.Rational(5, 2))
    assert B == sp.zeros(4, 2) and M == 4*sp.eye(2)
    assert residual == 3*W*T
    # A vanishing source cheats the residual criterion without Gram lower order.
    weakened_X = q**2*W*T
    weak_S = clean(weakened_X.H*weakened_X/q)
    assert weak_S == q**3*T**2
    assert clean(((q*H)**2-q**2*sp.eye(4))*weakened_X/q**sp.Rational(5, 2)) == 3*q**sp.Rational(3, 2)*W*T
    # Rank-one Gram: pseudoinverse isometry only on its positive support.
    Xnull = cast(Any, sp.sqrt(q)*sp.Matrix([[1, sp.I], [0, 0], [0, 0], [0, 0]]))
    Snull = clean(Xnull.H*Xnull)
    support = clean(Snull/(2*q))
    Wnull = clean(Xnull*support/sp.sqrt(2*q))
    assert clean(Wnull.H*Wnull) == support
    null = sp.Matrix([-sp.I, 1])
    assert Xnull*null == sp.zeros(4, 1) and support*null == sp.zeros(2, 1)
    assert Snull.det() == 0 and support != sp.eye(2)
    # Whole stationary block, not only the vacuum vector, must be removed.
    Lzero = sp.diag(0, 0, 1)
    source = sp.Matrix([0, 1, 1])
    rankone_Q = sp.diag(0, 1, 1)
    whole_Q = sp.diag(0, 0, 1)
    assert (rankone_Q*source).norm()**2 == 2 and (whole_Q*source).norm()**2 == 1
    assert Lzero*rankone_Q*source == Lzero*whole_Q*source
    return {'parallel_missing_B_zero_M': M, 'parallel_missing_scaled_residual': residual,
            'weak_source_Gram_over_q': weak_S, 'weak_source_false_residual_limit_zero': True,
            'null_Gram': Snull, 'null_support': support, 'two_polarizations_not_certified': True,
            'wholeP0_stationary_mass_rankone_2_vs_whole_1': True}


def atom_transfer_control() -> dict[str, Any]:
    q = sp.symbols('q', positive=True)
    x = sp.symbols('x', real=True)
    T = sp.Matrix([[1, sp.I/2], [-sp.I/2, 1]])
    G = T**2
    # NONNATIVE absolutely continuous uniform velocity law on [1-q,1+q].
    # For 0<q<=1/2 it is a positive band with kappa=2. There is NO finite atom.
    variance = sp.simplify(sp.integrate((x**2-1)**2, (x, 1-q, 1+q))/(2*q))
    electric = clean(sp.integrate(x**2, (x, 1-q, 1+q))/(2*q)*G/4096)
    assert clean(variance-(4*q**2/3+q**4/5)) == 0
    assert variance.subs(q, 0) == 0
    assert clean(electric.subs(q, 0)) == G/4096
    epsilon_squared = sp.Rational(9, 4)*variance
    eta = sp.symbols('eta', positive=True)
    chebyshev_upper = epsilon_squared/eta**2
    assert clean(chebyshev_upper.subs(q, 0)) == 0
    return {'magnetic_Gram_limit': G, 'residual_squared_over_q5': variance*G,
            'epsilon_squared_operator_upper': epsilon_squared,
            'Chebyshev_outside_velocity_square_gap': chebyshev_upper,
            'electric_Gram_over_q': electric, 'electric_limit': G/4096,
            'seam_d': 0, 'every_finite_q_atomless': True,
            'limit_velocity_atom_only': 1, 'finite_isolated_pole_inference_rejected': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    observed = {}
    for name, expected in request['input_sha256'].items():
        actual = hashlib.sha256((args.request.parent/name).read_bytes()).hexdigest()
        assert actual == expected, (name, actual, expected)
        observed[name] = actual
    result = {'kind': 'EXACT_NONNATIVE_CURVATURE_MEMORY_METHOD_CONTROLS_NOT_NATIVE_GROUND',
              'input_hashes': observed, 'matrix': matrix_control(), 'adverse': adverse_boundaries(),
              'conditional_limit_atom': atom_transfer_control(),
              'native_ground_or_fullH_computed': False, 'scientific_admission': False,
              'phase_or_positive_native_Gram_or_pole_admitted': False}
    args.output.write_text(json.dumps(encode(result), indent=2, sort_keys=True)+'\n')
    print('Exact complex two-column supported curvature Schur, adverse normalization/seam/source-return/null/wholeP0 and diffuse limit-atom controls reached.')


if __name__ == '__main__':
    main()
