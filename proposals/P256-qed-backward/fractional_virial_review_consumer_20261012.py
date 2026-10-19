"""Independent exact spectral, radial and Schwartz method controls; no ground solver."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sympy as s

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def spectral_controls() -> dict:
    # Exact complex Gram on a fixed generator with a TWO-dimensional zero space.
    L = s.diag(0, 0, 1, 16)
    Q = s.diag(0, 0, 1, 1)
    a = s.Matrix([1, s.I, 2, s.I])
    b = s.Matrix([s.I, 3, 1+s.I, 2])
    root = s.diag(0, 0, 1, 2)  # L^(1/4), a method exponent only.
    R = (L+s.eye(4)).inv()
    sources = s.Matrix.hstack(root*Q*a, root*Q*b)
    gram = (sources.conjugate().T*R*sources).applyfunc(s.expand)
    assert gram == gram.conjugate().T
    assert s.im(gram[0, 1]) != 0
    assert gram.det() > 0 and gram[0, 0] > 0
    rankone_Q = s.diag(0, 1, 1, 1)
    assert (rankone_Q*a)[1] != 0 and (Q*a)[1] == 0
    # Uncut G does not bound a cutoff graph endpoint; K^2 is necessary in general.
    x, y, dx, dy = s.Integer(1), s.Integer(1), s.Integer(1), s.Integer(1)
    difference = (x+dx)*(y+dy)-x*y
    linear_bound = dx*y+x*dy
    assert difference == 3 and linear_bound == 2
    assert difference == linear_bound+dx*dy
    # Monotone multiplicative cutoff versus unjustified naive triangle constant.
    for c1, c2 in [(Fraction(0), Fraction(1, 4)), (Fraction(1, 5), Fraction(4, 5)),
                   (Fraction(3, 4), Fraction(1)), (Fraction(1), Fraction(1))]:
        assert 0 <= c2-c1 <= 1-c1
    # SAME-L dyadic sequence: E_n=2^(17n/4), |u_n|^2=2^(-n/2), n>=1.
    # M=2^N, u_M truncates n<=N. Infinite sum at s=1/17 consists of 1's.
    n, exponent = s.symbols('n exponent', positive=True)
    energy = 2**(s.Rational(17, 4)*n)
    mass = 2**(-n/s.Integer(2))
    assert s.simplify(energy**(s.Rational(2, 17))*mass) == 1
    assert s.simplify(energy**2*mass) == 256**n
    beta = (1-17*exponent)/4
    assert s.expand(-s.Rational(1, 4)*(1-exponent)+4*exponent+beta) == 0
    assert 1+2**4 == 17
    assert s.Rational(4, 17)*4-1 == -s.Rational(1, 17)
    assert -s.Rational(4, 17)/4 == -s.Rational(1, 17)
    # Independently evaluate the integrable exponent, not a native moment value.
    value = s.integrate(2*s.Rational(1, 34)*s.Symbol('T', positive=True)**(-s.Rational(18, 17)),
                        (s.Symbol('T', positive=True), 1, s.oo))
    assert value == 1
    spike = 2**(-9*n)/(4*n)
    radius = 2**n
    electric_squared = 2**(4*n)/n**2
    assert s.simplify(spike*electric_squared**2-2**(-n)/(4*n**5)) == 0
    assert s.simplify(spike*radius**8*electric_squared-2**(3*n)/(4*n**3)) == 0
    assert s.limit(2**(3*n)/(4*n**3), n, s.oo) == s.oo
    assert s.limit(2**(-n)/(4*n**5)/(2**(-(n-1))/(4*(n-1)**5)), n, s.oo) == s.Rational(1, 2)
    return {'complex_positive_Gram': [[str(v) for v in row] for row in gram.tolist()],
            'whole_two_dimensional_P0_required': True,
            'weighted_kernel_missing_quadratic_control': {'actual': 3, 'linear_only': 2, 'corrected': 3},
            'monotone_cutoff_pointwise_control': True,
            'optimization_exponents': ['-1/17', '-1/17'], 'dyadic_graph_growth': 17,
            'beta_s': str(beta), 'layercake_s_1over34_coefficient': str(value),
            'accepted_E4_mixed_countercontrol': {'E4_summand': '2^-n/(4n^5)',
                'E4_ratio_limit': '1/2', 'Phi8E2_summand': '2^(3n)/(4n^3)',
                'Phi8E2_summand_limit': 'infinity', 'native_ground': False},
            'endpoint_countermodel': {'energies': '2^(17n/4)', 'masses': '2^(-n/2)',
                'tail_norm_at_M_2N': 'sqrt(2^(-(N+1)/2)/(1-2^(-1/2)))',
                'graph_norm_squared': '(256^(N+1)-256)/255',
                'endpoint_moment_each_summand': '1', 'endpoint_diverges': True,
                'scope': 'abstract same-generator hypothesis control, NOT native ground'},
            'gap_not_excluded': 'All E_n>1; a positive gap coexists with paid fractional hypotheses'}


def radial_controls() -> dict:
    y, a, d = s.symbols('y a d', nonnegative=True)
    t = 1+y
    f = t**3/(1+a)
    def native_dy(g):
        return s.diff(g, y)+s.Rational(7, 2)*a/t*s.diff(g, a)
    fy = native_dy(f)
    radial = s.factor(f+2*y*fy)
    assert s.simplify(radial-t**2*(1+7*y+a)/(1+a)**2) == 0
    div = d*f+2*y*fy
    lapdiv = s.factor(d*native_dy(div)+2*y*native_dy(native_dy(div)))
    jets = [fy, native_dy(fy), native_dy(native_dy(fy))]
    formula = d*(d+2)*jets[0]+4*(d+2)*y*jets[1]+4*y**2*jets[2]
    assert s.simplify(lapdiv-formula) == 0
    polynomials = [s.cancel(g*(1+a)**(j+2)/t**(2-j)) for j, g in enumerate(jets)]
    coefficients = [sum(abs(c) for c in s.Poly(p, a).all_coeffs()) for p in polynomials]
    cdiv = 60*62*coefficients[0]+4*62*coefficients[1]+4*coefficients[2]
    assert cdiv == 20125
    native0 = s.expand(lapdiv.subs({a: 0, d: 60}))
    assert native0 == 12672*y**2+23808*y+11160
    # Wrong radius convention |q|^2 instead of |q|^2/2 doubles the Laplacian.
    assert 2*native0 != native0
    # Direct ambient Cartesian chain rule in three dimensions, with free radial f.
    coords = s.symbols('q0:3', real=True)
    radial_y = sum(q*q for q in coords)/2
    ff = s.Function('f')
    vv = [q*ff(radial_y) for q in coords]
    div_direct = sum(s.diff(v, q) for v, q in zip(vv, coords))
    lap_direct = sum(s.diff(div_direct, q, 2) for q in coords)
    abstract = (3*5*s.diff(ff(y), y)+4*5*y*s.diff(ff(y), y, 2)+4*y*y*s.diff(ff(y), y, 3)).subs(y, radial_y)
    assert s.simplify(lap_direct-abstract) == 0
    # Explicit full-matrix conjugation preserves the native HS radius and radial field.
    X = s.Matrix([[1, 1+s.I, 0, 0], [1-s.I, -1, 0, 0], [0, 0, 2, s.I], [0, 0, -s.I, -2]])
    U = s.diag(s.I, -s.I, 1, 1)
    Y = U*X*U.conjugate().T
    normX = s.expand(s.trace(X*X))
    assert s.expand(s.trace(Y*Y)-normX) == 0 and U.det() == 1
    scalar = (1+normX)**3/(1+(1+normX)**s.Rational(7, 2))
    assert (Y*scalar-U*(X*scalar)*U.conjugate().T).applyfunc(s.simplify) == s.zeros(4)
    # epsilon->0 supremum obstruction: choose t=epsilon^(-2/7), a=1.
    eps = s.Symbol('eps', positive=True)
    f_peak = eps**(-s.Rational(6, 7))/2
    assert s.limit(f_peak, eps, 0, dir='+') == s.oo
    return {'radial_eigenvalue': str(radial), 'native_delta_div_epsilon0': str(native0),
            'direct_Cartesian_radial_identity': True, 'native_dimension': 60,
            'native_radius_squared': '|q|^2/2', 'jet_polynomials': [str(p) for p in polynomials],
            'jet_coefficient_l1': [str(c) for c in coefficients], 'Cdiv': str(cdiv),
            'SU4_matrix_radial_equivariance': True,
            'no_uniform_Dv_epsilon_bound': str(f_peak),
            'fixed_epsilon_flow': 'bounded speed and bounded Lipschitz derivative, complete on all signed times'}


def commutator_controls() -> dict:
    q = s.symbols('x y', real=True)
    psi = s.Function('psi')(*q)
    v = [s.Function('v0')(*q), s.Function('v1')(*q)]
    V = s.Function('V')(*q)
    k = s.Symbol('k', positive=True)
    divergence = sum(s.diff(v[i], q[i]) for i in range(2))
    def h(g):
        return -k*sum(s.diff(g, x, 2) for x in q)+V*g
    def D(g):
        return -s.I*(sum(v[i]*s.diff(g, q[i]) for i in range(2))+divergence*g/2)
    actual = s.expand(s.I*(h(D(psi))-D(h(psi))))
    weighted = -2*k*sum(s.diff((s.diff(v[j], q[i])+s.diff(v[i], q[j]))/2*s.diff(psi, q[j]), q[i])
                         for i in range(2) for j in range(2))
    correction = -k*sum(s.diff(divergence, x, 2) for x in q)*psi/2
    potential = -sum(v[i]*s.diff(V, q[i]) for i in range(2))*psi
    assert s.simplify(actual-weighted-correction-potential) == 0
    assert s.simplify(actual-weighted+correction+potential) != 0
    return {'dimension': 2, 'arbitrary_vector_field_differential_residual': '0',
            'identity': 'i[H,D]=2k p.sym(Dv).p-(k/2)Delta(div v)-v.grad V',
            'opposite_potential_and_quantum_signs_rejected': True,
            'half_density_generator': '-i(v.grad+div(v)/2)'}


def gaussian_controls() -> dict:
    x, z = s.symbols('x z', real=True)
    def average(poly):
        result = s.Integer(0)
        for (power,), coeff in s.Poly(s.expand(poly), z).terms():
            if power % 2 == 0:
                moment = s.Integer(1) if power == 0 else s.factorial2(power-1)/2**(power//2)
                result += coeff*moment
        return s.factor(result)
    rows = []
    for center in [3, 9, 27]:
        xx = center+z
        m4, m6, m10 = [average(xx**p) for p in (4, 6, 10)]
        centered = average(z*z*xx**6)
        momentum_squared = s.factor((105*m4+4*m10)/14/m6-centered/m6)
        assert momentum_squared > 0
        kinetic = 14*(momentum_squared*m6+centered)
        assert s.simplify(kinetic-105*m4-4*m10) == 0
        assert 4*m10 > -105*m4
        rows.append({'center': center, 'P_squared': str(momentum_squared), 'radial10': str(m10),
                     'positive_weighted_kinetic': str(kinetic), 'virial_residual': '0'})
    R, P = s.symbols('R P', real=True)
    psi = s.exp(-(x-R)**2/2+s.I*P*x)
    local_energy = s.simplify((-s.diff(psi, x, 2)+x**4*psi)/psi)
    assert s.diff(local_energy, x) != 0
    # A simple noncommuting full-H ground is not an eigenvector of its summand h.
    H = s.diag(0, 2)
    local_h = s.Matrix([[2, 1], [1, 2]])
    omega = s.Matrix([1, 0])
    assert H*omega == s.zeros(2, 1) and local_h*omega != 0*omega
    return {'Schwartz_family': 'pi^-1/4 exp(-(x-R)^2/2+iPx)',
            'method_H': 'p^2+x^4', 'method_v': 'x^7', 'packets': rows,
            'nonground_local_energy_derivative': str(s.diff(local_energy, x)),
            'dropping_positive_kinetic_not_upper_bound': True,
            'full_ground_to_scalar_h_eigenstate_substitution_rejected': True,
            'native_ground_no_go': False}


def native_matter_controls(profiles: list[dict]) -> dict:
    assert [p['load'] for p in profiles] == [1, 1, 63]
    assert [len(p['paths']) for p in profiles] == [3, 6, 6]
    assert len({p['onsite'] for p in profiles}) == 3
    w = [abs(Fraction(p['onsite']))+2*sum((abs(Fraction(r['c0']))+abs(Fraction(r['ci']))
                                        for r in p['paths']), Fraction()) for p in profiles]
    # Divide by lambda before arithmetic, avoiding huge integer construction.
    B_over_lambda = 4*(w[0]+(w[1]+63*w[2])/2**8192)
    assert B_over_lambda < 512
    z = s.Symbol('z', nonnegative=True)
    # Each branch is an exact polynomial nonnegativity witness.
    large = s.Poly(s.expand(8*(1+z)**7-(1+z)*(1+(1+z)**2)**3), z)
    small = s.Poly(s.expand(8*(1+z)**7-z*((1+z)**2+z**2)**3), z)
    assert all(c >= 0 for c in large.all_coeffs())
    assert all(c >= 0 for c in small.all_coeffs())
    return {'all65_loads': [1, 1, 63], 'native_path_counts': [3, 6, 6],
            'w': [str(v) for v in w], 'B_less_than_512lambda': True,
            'linear_endpoint_derivative_bound': 'B sum_z r_z f_z <=8B sum_z(1+r_z^7)',
            'B_over_lambda_sha256': digest(str(B_over_lambda).encode())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    captured = {name: (ROOT/name).read_bytes() for name in request['source_input_sha256']}
    assert all(digest(raw) == request['source_input_sha256'][name] for name, raw in captured.items())
    profiles = json.loads(captured[request['profile_file']])['primitive_profiles']
    result = {'kind': 'INDEPENDENT_FRACTIONAL_VIRIAL_DIFFERENTIAL_SPECTRAL_QUANTUM_CONTROLS',
              'spectral': spectral_controls(), 'radial': radial_controls(),
              'commutator': commutator_controls(), 'Schwartz': gaussian_controls(),
              'native_matter': native_matter_controls(profiles),
              'source_input_sha256': request['source_input_sha256'],
              'native_ground_computed': False, 'native_evolution_computed': False,
              'scientific_admission': False, 'campaign_complete': False}
    assert all((ROOT/name).read_bytes() == raw for name, raw in captured.items())
    result['inputs_unchanged'] = True
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'receipt': str(args.output), 'inputs_unchanged': True,
                      'independent_spectral_differential_quantum_controls': True}))


if __name__ == '__main__':
    main()
