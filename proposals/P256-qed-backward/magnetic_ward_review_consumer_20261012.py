"""Independent exact Ward/circle/Gram/energy controls; never native ground data."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent
Cell = tuple[int, int, int]
Edge = tuple[Cell, int]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def shift(x: Cell, axis: int, n: int) -> Cell:
    y = list(x)
    y[axis] = (y[axis] + 1) % n
    return y[0], y[1], y[2]


def boundary(x: Cell, normal: int, n: int) -> dict[Edge, int]:
    j, k = (normal + 1) % 3, (normal + 2) % 3
    return {(x, j): 1, (shift(x, j, n), k): 1,
            (shift(x, k, n), j): -1, (x, k): -1}


def div(flow: dict[Edge, Any], n: int) -> dict[Cell, Any]:
    result: dict[Cell, Any] = {}
    for (x, axis), coefficient in flow.items():
        y = shift(x, axis, n)
        result[x] = result.get(x, 0) + coefficient
        result[y] = result.get(y, 0) - coefficient
    return {x: value for x, value in result.items() if value}


def native_controls(n: int) -> dict[str, Any]:
    x: Cell = (0, 0, 0)
    c = boundary(x, 2, n)
    adjacent = boundary(shift(x, 0, n), 2, n)
    assert not div(c, n) and not div(adjacent, n)
    overlap = sum(k * adjacent.get(e, 0) for e, k in c.items())
    assert overlap == -1 and sum(k * k for k in c.values()) == 4
    # A different native boundary crossing tests AP rather than hiding it at origin.
    wrap = boundary((n-1, 0, 0), 2, n)
    frequencies: dict[str, int] = {}
    ap_nonzero = 0
    charged_paths = 0
    for site in product(range(n), repeat=3):
        for axis in range(3):
            for length in (1, 2):
                end = site
                frequency = Fraction(0)
                ap = 1
                for _ in range(length):
                    frequency += Fraction(wrap.get((end, axis), 0), 2)
                    if end[axis] == n-1:
                        ap = -ap
                    end = shift(end, axis, n)
                for orientation in (-1, 1):
                    nu = orientation * frequency
                    frequencies[str(nu)] = frequencies.get(str(nu), 0) + 64
                    charged_paths += 64
                    if ap == -1 and nu:
                        ap_nonzero += 64
    assert charged_paths == 64 * 12 * n**3 and ap_nonzero > 0
    assert frequencies.get('1/2', 0) == frequencies.get('-1/2', 0)
    # A cube boundary is in ker Cdag, but nonlinear smooth flux is not null there.
    cube = [(boundary(shift(x, axis, n), axis, n), 1) for axis in range(3)]
    cube += [(boundary(x, axis, n), -1) for axis in range(3)]
    linear: dict[Edge, Fraction] = {}
    weighted: dict[Edge, Fraction] = {}
    angles = {(x, 1): 1, (shift(x, 1, n), 2): 2}
    cosine = (Fraction(1), Fraction(0), Fraction(-1), Fraction(0))
    for flow, sign in cube:
        quarter = sum(v * angles.get(e, 0) for e, v in flow.items()) % 4
        cos = cosine[quarter]
        slope = (cos + Fraction(1, 2)) / (Fraction(5, 4) + cos)
        for e, v in flow.items():
            linear[e] = linear.get(e, Fraction(0)) + sign * v
            weighted[e] = weighted.get(e, Fraction(0)) + sign * v * slope
    assert not any(linear.values()) and any(weighted.values())
    assert not div(weighted, n)
    return {'N': n, 'closed_faces': True, 'unit_face_Gram_det': '15/16',
            'overlap': overlap, 'charged64_literal_AP_path_count': charged_paths,
            'AP_nonzero_current_count': ap_nonzero, 'frequency_histogram': frequencies,
            'neutral_charge0_and_onsite_frequency': '0; retained in full H and ground',
            'cube_linear_boundary_zero': True,
            'nonlinear_cube_gradient_squared': str(sum(v*v for v in weighted.values())),
            'nonlinear_gradient_Gauss_closed': True,
            'native_factor_geometry_NOT_ground': True}


def ward_controls() -> dict[str, Any]:
    L = sp.diag(0, 0, 1, 3, 7)
    omega = sp.eye(5)[:, 0]
    P0 = sp.diag(1, 1, 0, 0, 0)
    Q = sp.eye(5) - P0
    e = sp.Matrix([0, 0, 1+sp.I, 2, 0])
    f = sp.Matrix([0, 1, 2, sp.I, 1])
    E = e * omega.H + omega * e.H
    F = f * omega.H + omega * f.H
    J = sp.simplify(-sp.I * (L * E - E * L))
    assert E == E.H and F == F.H and J == J.H
    assert P0 * E * omega == sp.zeros(5, 1)
    assert sp.simplify(L * E * omega - sp.I * J * omega) == sp.zeros(5, 1)
    eps = sp.symbols('epsilon', positive=True)
    C = (f.H * Q * e)[0]
    K = (f.H * Q * (L + eps*sp.eye(5)).inv() * Q * J * omega)[0]
    rhs = -sp.I*C + sp.I*eps*(f.H * (L+eps*sp.eye(5)).inv()*Q*e)[0]
    assert sp.simplify(K-rhs) == 0
    assert sp.simplify(sp.limit(K, eps, 0)+sp.I*C) == 0
    derivative = sp.simplify((sp.I*omega.H*(E*F-F*E)*omega)[0])
    assert derivative == 2*sp.im(C)
    # This is a GENERAL method wrongcase, not an allowed native closed-source ground.
    Ebad = E + sp.I*sp.eye(5)[:, 1]*omega.H - sp.I*omega*sp.eye(5)[:, 1].H
    Zbad = (f.H*P0*Ebad*omega)[0]
    derivative_bad = sp.simplify((sp.I*omega.H*(Ebad*F-F*Ebad)*omega)[0])
    assert derivative_bad == derivative + 2*sp.im(Zbad) and Zbad != 0
    V = sp.eye(5)[:, 1]*omega.H + omega*sp.eye(5)[:, 1].H
    assert Q*V*omega == sp.zeros(5, 1) and P0*V*omega != sp.zeros(5, 1)
    t, tau = sp.symbols('t tau', real=True, positive=True)
    log_four = sp.diff(sp.log(sp.cosh(tau*t)), t, 4).subs(t, 0)
    assert log_four == -2*tau**4
    # Electric moments alone do not buy the vacuum graph. Bounded swaps E_n,
    # vacuum amplitudes n^-2, and L excited levels n^2 have finite E4 but ||Le||^2=inf.
    norm = sp.zeta(4)
    assert norm == sp.pi**4/90
    return {'closed_source_P0E_zero': True, 'native_specialization_Z': '0',
            'cross': str(C), 'K_minus_epsilon': str(K), 'real_static_response': str(derivative),
            'general_only_Z_counter': str(Zbad), 'general_transport_response': str(derivative_bad),
            'arbitrary_V_stationary_block_survives': True, 'log_connected_fourth': str(log_four),
            'electric_E4_without_LE_graph_wrongcase': {'e_norm_squared': 'zeta(4)',
                'Le_norm_squared': 'sum_{n>=1} 1 = infinity', 'E_operator_norm': '1'},
            'nonnative_exact_method_NOT_declared_native_ground': True}


def gram_controls() -> dict[str, Any]:
    i = sp.I
    X = sp.Matrix([[1, 0, i], [0, 1, 0], [0, 0, 0]])
    Y = sp.Matrix([[1/sp.sqrt(2), 0, i/sp.sqrt(2)], [0, 1, 0],
                   [1/sp.sqrt(2), 0, i/sp.sqrt(2)]])
    S, T, C = X.H*X, Y.H*Y, X.H*Y
    assert S == T and S.rank() == 2
    support = sp.Matrix([[sp.Rational(1, 2), 0, i/2], [0, 1, 0], [-i/2, 0, sp.Rational(1, 2)]])
    invroot = sp.Matrix([[1/(2*sp.sqrt(2)), 0, i/(2*sp.sqrt(2))],
                        [0, 1, 0], [-i/(2*sp.sqrt(2)), 0, 1/(2*sp.sqrt(2))]])
    WX, WY = X*invroot, Y*invroot
    assert sp.simplify(WX.H*WX-support) == sp.zeros(3)
    assert sp.simplify(WY.H*WY-support) == sp.zeros(3)
    null = sp.Matrix([-i, 0, 1])
    assert T*null == sp.zeros(3, 1) and C*null == sp.zeros(3, 1)
    schur = sp.simplify(S-C*T.pinv()*C.H)
    assert all(value >= 0 for value in schur.eigenvals())
    L = sp.diag(1, 3, 7)
    damp = L*(L+sp.eye(3)).inv()
    khat = -i*WX.H*damp*WY
    defect = sp.simplify(support-khat.H*khat)
    assert all(value >= 0 for value in defect.eigenvals())
    u = sp.Matrix([1, 0, -i])
    full = (u.H*S*u)[0]
    deleted = (u.H*S.applyfunc(sp.re)*u)[0]
    assert full == 4 and deleted == 2
    Pband = sp.diag(1, 0, 0)
    Sband = X.H*Pband*X
    assert Sband.rank() == 1 and S.rank() == 2
    wrong_isometry = sp.simplify(invroot*Sband*invroot)
    assert wrong_isometry != support
    return {'support_rank': 2, 'ambient_columns': 3, 'full_S': str(S),
            'supported_isometry_NOT_I3': str(support), 'Schur_remainder': str(schur),
            'Abel_contraction_defect': str(defect), 'null_column_annihilation': True,
            'complex_u_full_quadratic': str(full), 'wrong_real_deleted': str(deleted),
            'band_rank': 1, 'unfiltered_Gram_wrong_for_band': True,
            'nonnative_exact_method': True}


def circle_controls() -> dict[str, Any]:
    theta = sp.symbols('theta', real=True)
    alpha = sp.Rational(1, 3)
    # Actual Hilbert-valued H1 model with nontrivial circle holonomy.
    psi = sp.exp(sp.I*alpha*theta)*sp.Matrix([1+sp.exp(sp.I*theta), sp.exp(sp.I*theta)])/sp.sqrt(6*sp.pi)
    density = (1 + sp.Rational(2, 3)*sp.cos(theta))/(2*sp.pi)
    assert sp.simplify(sp.expand_complex((psi.H*psi)[0])-density) == 0
    holonomy = sp.exp(2*sp.pi*sp.I*alpha)
    assert sp.simplify(psi.subs(theta, 2*sp.pi)-holonomy*psi.subs(theta, 0)) == sp.zeros(2, 1)
    norm = sp.integrate(density, (theta, 0, 2*sp.pi))
    assert norm == 1
    kinetic = (alpha**2 + 2*(alpha+1)**2)/3
    assert kinetic == sp.Rational(11, 9)
    tv = sp.Rational(4, 3)/sp.pi
    assert tv**2 < 4*kinetic
    sup = sp.Rational(5, 6)/sp.pi
    assert sup < 1/(2*sp.pi) + 2*sp.sqrt(kinetic)
    # Fourier convolution, with all geometric tails summed exactly.
    r = sp.symbols('r', positive=True)
    P = (1-r*r)/(1+2*r*sp.cos(theta)+r*r)
    bprime = 2*(r*sp.cos(theta)+r*r)/(1+2*r*sp.cos(theta)+r*r)
    assert sp.simplify(bprime-(1-P)) == 0
    a0 = (1+r*r)/(1-r*r)
    a1 = -2*r/(1-r*r)
    rows = []
    for radius in (sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(31, 32), sp.Rational(255, 256)):
        mean = 1-sp.Rational(2, 3)*radius
        second = (a0+sp.Rational(2, 3)*a1).subs(r, radius)
        haar_second = a0.subs(r, radius)
        assert 0 <= mean <= 1+4*sp.pi*sp.sqrt(kinetic)
        assert second <= (1+4*sp.pi*sp.sqrt(kinetic))*haar_second
        assert second >= mean**2
        rows.append({'r': str(radius), 'twisted_H1_mean_P': str(mean),
                     'twisted_H1_second_P': str(second), 'Haar_K0_second_P': str(haar_second),
                     'real_normalized_derivative_factor': str(1/(2*radius))})
    assert sp.limit(a0, r, 1, dir='-') == sp.oo
    return {'holonomy': str(holonomy), 'density': str(density), 'integral_density': str(norm),
            'K': str(kinetic), 'TV': str(tv), 'density_sup': str(sup),
            'monodromy_norm_density_periodic': True, 'rows': rows,
            'Haar_K0_seam_second_diverges': True, 'local_H1_NOT_IR_order': True,
            'analytic_fibre_method_NOT_native_ground_measurement': True}


def energy_controls() -> dict[str, Any]:
    L = sp.diag(0, 0, sp.Rational(1, 4), 4)
    root = sp.diag(0, 0, sp.Rational(1, 2), 2)
    Q = sp.diag(0, 0, 1, 1)
    X = sp.Matrix([[0, 0], [1, 2], [2, 0], [0, sp.Rational(1, 2)]])
    assert root*root == L
    T = root*Q*X
    assert T.H*T == sp.eye(2)
    B = sp.diag(0, 0, 2, sp.Rational(1, 2))*T
    assert B == Q*X and B.H*B == sp.diag(4, sp.Rational(1, 4))
    assert all(v >= 0 for v in (4*sp.eye(2)-B.H*B).eigenvals())
    selected_Q = sp.diag(0, 1, 1, 1)
    assert X.H*selected_Q*X != X.H*Q*X
    # One single infinite diagonal operator: bounded energy map, unbounded total map.
    n = sp.symbols('n', positive=True, integer=True)
    E = 1/n**2
    source = n
    assert sp.simplify(sp.sqrt(E)*source) == 1
    assert sp.limit(source, n, sp.oo) == sp.oo
    r = sp.Rational(1, 2)
    real = 768/(1-r)**2
    complex_bound = 2*real
    assert real == 3072 and complex_bound == 6144
    q, alpha = sp.symbols('q alpha', positive=True)
    sigma2 = 1/q**2
    b2 = sp.simplify(q**2*6144/(alpha*q)*sigma2)
    assert b2 == 6144/(alpha*q)
    return {'same_L_root_identity': True, 'energy_T_Gram': str(T.H*T),
            'positive_band_source_Gram': str(B.H*B), 'delta': '1/4', 'band_bound_squared': '4',
            'wholeP0_not_selected_ground': True, 'infinite_energy_map_T': 'identity on l2',
            'infinite_total_map': 'diag(n); unbounded', 'energy_levels': '1/n^2 -> 0',
            'complex_safe_constant': str(complex_bound), 'real_constant': str(real),
            'b_mass_upper': str(b2), 'S_b_over_q_upper': str(b2/q),
            'common_ae_vs_point': 'A density changed on {p0} represents the same operator measure; no selected point value follows.',
            'smallq_lower_atom_phase': False, 'nonnative_exact_method': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {name: digest(Path(name) if Path(name).is_absolute() else ROOT/name)
              for name in request['pins']}
    assert before == request['pins']
    proof = json.loads((ROOT/request['proof']).read_text())
    assert proof['owner'] == 'MagneticWardProofReview'
    result = {'kind': 'INDEPENDENT_EXACT_WARD_CIRCLE_GRAM_ENERGY_REVIEW_CONTROLS_NOT_ADMISSION',
              'native_geometry': [native_controls(n) for n in (4, 6, 8)],
              'Ward': ward_controls(), 'supported_Gram': gram_controls(),
              'circle': circle_controls(), 'energy': energy_controls(),
              'source_input_sha256_before': before,
              'source_input_sha256_after': {name: digest(Path(name) if Path(name).is_absolute() else ROOT/name)
                                           for name in request['pins']},
              'native_ground_computed': False, 'original_producer_executed': False,
              'scientific_admission': False, 'IR_lower_or_atom': False, 'campaign_complete': False}
    assert result['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('Independent exact native factor and labelled Ward/circle/Gram/energy wrongcases completed; no phase admission.')


if __name__ == '__main__':
    main()
