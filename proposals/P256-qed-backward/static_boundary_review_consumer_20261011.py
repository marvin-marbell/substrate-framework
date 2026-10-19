"""Independent frozen static-boundary review controls; no original producer import."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import zipfile

import mpmath as mp
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def matrix_control() -> dict:
    x = sp.Matrix([[0, 1], [1, 0]])
    y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    z = sp.diag(1, -1)
    gamma = [sp.kronecker_product(x, sp.eye(2))]
    gamma += [sp.kronecker_product(z, p) for p in (x, y, z)]
    basis = []
    for degree in range(1, 5):
        for word in itertools.combinations(range(4), degree):
            b = sp.eye(4)
            for j in word:
                b *= gamma[j]
            basis.append((sp.I if degree in (2, 3) else 1)*b/sp.sqrt(8))
    metric = sp.Matrix([[sp.simplify(sp.trace(a*b)) for b in basis] for a in basis])
    assert metric == sp.eye(15)/2
    fields = [sum((sp.Rational(((mu+2)*(j+3)) % 13-6, 17)*b
                   for j, b in enumerate(basis)), sp.zeros(4)) for mu in range(4)]
    s = sp.simplify(sum(sp.trace(a*a) for a in fields))
    jordan = sp.simplify(4*sum(sp.trace((a*b+b*a)**2) for a in fields for b in fields))
    assert s*s <= jordan <= 16*s*s
    u = sp.diag(1, 1, 1, 1)
    u[:2, :2] = sp.Matrix([[sp.Rational(3, 5), sp.Rational(4, 5)],
                           [-sp.Rational(4, 5), sp.Rational(3, 5)]])
    assert u.H*u == sp.eye(4) and u.det() == 1
    neighbour = [sp.Rational(2, 3)*fields[(j+1) % 4] for j in range(4)]
    align = sp.simplify(sum(sp.trace((a-u*b*u.H)**2) for a, b in zip(fields, neighbour)))
    doubled = sp.simplify(sum(sp.trace((2*a-u*(2*b)*u.H)**2)
                             for a, b in zip(fields, neighbour)))
    assert align > 0 and doubled == 4*align
    # All four fields and all 16 ordered pairs, not a Gram-only or 3-field crop.
    return {'basis_count': len(basis), 'Cart_metric': 'delta/2', 'dimension': 60,
            's': str(s), 'Jordan_div_lambda': str(jordan),
            'Jordan_over_s2': str(sp.cancel(jordan/s**2)),
            'alignment_div_alpha': str(align), 'doubling_alignment_factor': 4,
            'arithmetic': 'exact SymPy rational/radical; selected arbitrary full60 point, not ground',
            'analytic_Jordan_bound': 'lambda*s^2 <= J <= 16*lambda*s^2 from PSD4 Gram and HS anticommutator',
            'radius_units': 'r=sqrt(s)=|q|/sqrt(2); not |q|'}


def tail_control(ctx, request: dict) -> dict:
    shape = 30
    rows = []
    for m in request['cutoffs']:
        v = ctx.mpf(m*m)/30
        high_moment = Fraction(31, 30)
        # Integer-shape upper incomplete Gamma uses finite exact-expression sums,
        # not the original scipy gamma implementation.
        a = ctx.mpf('7.5')
        upper = ctx.exp(-a)*sum(a**j/ctx.factorial(j) for j in range(32))
        b = ctx.mpf(30*m*m)/16
        neighbour = 1-ctx.exp(-b)*sum(b**j/ctx.factorial(j) for j in range(30))
        lower = ctx.mpf(31)/480*upper*neighbour
        # At the declared HS cutoff r>M, require s_x>M^2, not M^2/4.
        correct_upper = ctx.exp(-30)*sum(ctx.mpf(30)**j/ctx.factorial(j) for j in range(32))
        correct_b = ctx.mpf(30*m*m)/4
        correct_neighbour = 1-ctx.exp(-correct_b)*sum(correct_b**j/ctx.factorial(j) for j in range(30))
        correct_lower = ctx.mpf(31)/480*correct_upper*correct_neighbour
        assert correct_lower > 0
        assert lower > 0 and high_moment == Fraction(31, 30)
        rows.append({'M': m, 'mixture_weight': str(Fraction(1, m**4)),
                     'high_variance': str(Fraction(m*m, 30)),
                     'high_s2_contribution': str(high_moment),
                     'target_event_r_greater_Mover2_lower_div_alpha2': ctx.nstr(lower, 60),
                     'corrected_event_r_greater_M_lower_div_alpha2': ctx.nstr(correct_lower, 60),
                     'complete_high_alignment_action_squared_div_alpha2':
                     str(Fraction(31, 30)*Fraction((m*m+1)**2, m**4))})
    exponent_rows = [{'p': p, 'union_exponent_at_R_M2': str(Fraction(3)-Fraction(p, 2)),
                     'quadratic_action_triangle_exponent': str(Fraction(8)-Fraction(p, 2))}
                    for p in (4, 6, 16, 18)]
    assert Fraction(8)-Fraction(16, 2) == 0
    assert Fraction(8)-Fraction(18, 2) < 0
    return {'rows': rows, 'moment_power_controls': exponent_rows,
            'cone_vs_union': {'cone_min_a': '2', 'union_requires_a_lt': '4/3',
                              'at_a2_norm_union_exponent': '1'},
            'nonstationary': True, 'actual_ground_failure': False,
            'exact_scope': 'bounded fourth radial moments do not imply uniform single-alignment action tails; no full-H lower bound',
            'source': 'full60 invariant Gaussians, normalized Haar, all65 exterior4-even/empty-odd singlets each2V, Ref4 inert; k_axis0=64parity, Dk=128parity',
            'kinetic_budget': 'Kphi=60chi/v; high mixture contribution=1800chi/M^6; baseline finite',
            'fullE_moment_control': 'closed plaquette shifts mb: weight m^-2 bounds second moment but fourth action diverges; m^-4 yields nonvanishing fourth action tail'}


def projection_control(ctx, request: dict) -> dict:
    va, vb = ctx.mpf(1)/30, ctx.mpf(2)/30
    overlap = (2*ctx.sqrt(va*vb)/(va+vb))**30
    norm2 = 2*(1+overlap**2)
    gram = ctx.matrix([[1, overlap], [overlap, 1]])
    def density(s, v):
        return s**29*ctx.exp(-s/v)/(ctx.factorial(29)*v**30)
    def quantities(s):
        pa, pb = density(s, va), density(s, vb)
        a = ctx.matrix([ctx.sqrt(pa), ctx.sqrt(pb)])
        ca, cb = 60-(2+2/va)*s, 60-(2+2/vb)*s
        c = ctx.matrix([ca*a[0], cb*a[1]])*ctx.exp(-s)
        denominator = (a.T*gram*a)[0]
        if denominator == 0:
            return ctx.mpf(0), ctx.mpf(0), ctx.mpf(0)
        full = (c.T*gram*c)[0]/norm2
        fitted = (a.T*gram*c)[0]**2/denominator/norm2
        determinant = 4*s*s*ctx.exp(-2*s)*(1/va-1/vb)**2*(1-overlap**2)*pa*pb/denominator/norm2
        assert abs(full-fitted-determinant) < ctx.mpf('1e-75')
        return full, fitted, determinant
    knots = [0, ctx.mpf('0.5'), 1, ctx.mpf('1.5'), 2, 3, 5, 10, 20]
    complete = ctx.quad(lambda s: quantities(s)[0], knots)
    fitted = ctx.quad(lambda s: quantities(s)[1], knots)
    residual = ctx.quad(lambda s: quantities(s)[2], knots)
    assert residual > 2 and complete > residual and abs(complete-fitted-residual) < ctx.mpf('1e-70')
    def tail(v, degree, cutoff):
        rate = 1/v+2
        return ctx.gammainc(30+degree, rate*cutoff, ctx.inf)/(ctx.factorial(29)*v**30*rate**(30+degree))
    action_tail = (1+overlap)/norm2*sum(7200*tail(v, 0, 20)
                      +2*(2+2/v)**2*tail(v, 2, 20) for v in (va, vb))
    residual_tail = 4*(1/va-1/vb)**2*(1-overlap**2)*tail(vb, 2, 20)/norm2
    # A deliberately inadequate crop is rejected by a paid analytic action tail.
    crop2 = ctx.quad(lambda s: quantities(s)[0], [0, ctx.mpf('0.5'), 1, ctx.mpf('1.5'), 2])
    omitted2 = ctx.quad(lambda s: quantities(s)[0], [2, 3, 5, 10, 20])
    assert omitted2 > ctx.mpf('0.01') and abs(crop2+omitted2-complete) < ctx.mpf('1e-70')
    vc = 2*va*vb/(va+vb)
    local_s2 = 930*(va*va+vb*vb+2*overlap**2*vc*vc)/norm2
    local_kinetic = (15/va+15/vb+60*overlap**2/(va+vb))/norm2
    return {'digits': ctx.dps, 'overlap': ctx.nstr(overlap, 70),
            'source_norm2_before_division': ctx.nstr(norm2, 70), 'source_norm_after_division': '1 exactly',
            'full_action_squared_div_kphi2': ctx.nstr(complete, 70),
            'optimal_local_multiplier_fitted_squared_div_kphi2': ctx.nstr(fitted, 70),
            'residual_squared_div_kphi2': ctx.nstr(residual, 70),
            'action_tail_s20_analytic_upper_numeric_value': ctx.nstr(action_tail, 60),
            'residual_tail_s20_analytic_upper_numeric_value': ctx.nstr(residual_tail, 60),
            's2': ctx.nstr(local_s2, 60), 'kinetic_div_kphi': ctx.nstr(local_kinetic, 60),
            'wrong_crop_s2_omitted_action_squared': ctx.nstr(omitted2, 60),
            'drop_kinetic_derivative_wrong_residual': '0: remaining scalar action is exactly a local multiplier',
            'wrong_core_enlargement': 'arbitrary bounded local B(H) can map f_a to c_a f_a and f_b to c_b f_b using bounded rank2 dual-basis operator; it annihilates this residual but is not THIS MW core',
            'exact_positivity': '(1-z^2)>0, (va^-1-vb^-1)^2>0, pa*pb>0 on s>0; determinant residual strictly positive',
            'numeric_limitation': '90-digit mpmath quadrature is not interval-certified; analytic positivity and Gamma-tail formulas are separate exact statements',
            'distance_controls': [{'R': r, 'Y': r+5, 'even_N': 4*(r+8)} for r in request['radii']],
            'nonstationary': True, 'actual_ground_failure': False}


def electric_control(n: int) -> dict:
    sites = list(itertools.product(range(n), repeat=3))
    index = {x: i for i, x in enumerate(sites)}
    parity = np.array([(-1)**sum(x) for x in sites])
    k0 = np.zeros((n**3, 3)); k0[:, 0] = 64*parity
    kl = np.repeat((64*parity/3)[:, None], 3, axis=1)
    loop = np.zeros_like(k0)
    loop[index[(n-1, 0, 0)], 0] = 1
    loop[index[(0, 0, 0)], 1] = 1
    loop[index[(n-1, 1, 0)], 0] = -1
    loop[index[(n-1, 0, 0)], 1] = -1
    def div(field):
        out = np.sum(field, axis=1).copy()
        for x, row in zip(sites, field):
            for axis in range(3):
                y = list(x); y[axis] = (y[axis]+1) % n
                out[index[tuple(y)]] -= row[axis]
        return out
    assert np.max(abs(div(loop))) == 0
    assert np.max(abs(div(k0+loop)-128*parity)) == 0
    theta = np.array([2**(-sum(min(j, n-j) for j in x)/4) for x in sites])
    cross0 = 128*np.sum(theta[:, None]*(k0-kl)*kl)
    cross = 128*np.sum(theta[:, None]*(k0+loop-kl)*kl)
    expected = 8192/3*(1-1/math.sqrt(2))
    assert abs(cross0) < 1e-7 and abs(cross-expected) < 1e-7
    # Independent Fourier inversion of Poisson on a NEW N; no producer/solver rerun.
    rho = np.zeros((n, n, n)); rho[0, 0, 0] = -1; rho[1, 0, 0] = 1
    modes = np.arange(n)
    one = 4*np.sin(math.pi*modes/n)**2
    eigen = one[:, None, None]+one[None, :, None]+one[None, None, :]
    transformed = np.fft.fftn(rho); transformed[0, 0, 0] = 0
    eigen[0, 0, 0] = 1
    potential = np.fft.ifftn(transformed/eigen).real
    tl = np.stack([potential-np.roll(potential, -1, axis=a) for a in range(3)], axis=-1).reshape(-1, 3)
    path = np.zeros_like(k0); path[0, 0] = -1
    tt = path-tl
    exterior = np.ones_like(path, dtype=bool); exterior[0, 0] = False
    assert np.max(abs(div(tl)-rho.reshape(-1))) < 1e-12
    assert np.linalg.norm(tl[exterior]) > 0.4
    assert np.max(abs((tt+tl)[exterior])) == 0
    k = k0+loop; kt = k-kl
    direct = 64*(2*np.sum(k*path)+np.sum(path*path))
    split = 64*(2*np.sum(kt*tt)+np.sum(tt*tt)+2*np.sum(kl*tl)+np.sum(tl*tl))
    assert abs(direct-split) < 1e-7
    wrong_weighted = 64*np.sum(theta[:, None]*(kt*kt+kl*kl))
    full_weighted = 64*np.sum(theta[:, None]*k*k)
    assert abs(full_weighted-wrong_weighted-cross) < 1e-7
    return {'N': n, 'each65_number': 2*n**3, 'exact_checkerboard_divergence': '128parity',
            'closed_circulation_preserves_Gauss': True, 'cross0': float(cross0),
            'weighted_cross': float(cross), 'analytic_cross': '(8192/3)*(1-1/sqrt(2))>0',
            'path_fullE_delta': float(direct), 'split_delta': float(split),
            'exterior_longitudinal_norm': float(np.linalg.norm(tl[exterior])),
            'complete_exterior_cancellation': 0,
            'scope': 'native fullE geometry and weighted cross, not physical ground/dynamics or phase'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {n: digest((ROOT/n).read_bytes()) for n in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['snapshot']) as bundle:
        observed = {n: digest(bundle.read(n)) for n in request['target_pins']}
    assert observed == request['target_pins']
    ctx = mp.mp.clone(); ctx.dps = request['digits']
    result = {'kind': 'INDEPENDENT_STATIC_BOUNDARY_REVIEW_NOT_ADMISSION',
              'matrix': matrix_control(), 'tail': tail_control(ctx, request),
              'projection': projection_control(ctx, request),
              'fullE': [electric_control(n) for n in request['N']],
              'input_hashes_before': before, 'target_pins': observed,
              'same_static_GNS_exterior_closed': False, 'same_static_range': False,
              'history_splice_admitted': False, 'ground_or_native_U_computed': False,
              'runtime': {'mpmath': mp.__version__, 'sympy': sp.__version__, 'numpy': np.__version__}}
    result['input_hashes_after'] = {n: digest((ROOT/n).read_bytes()) for n in request['pins']}
    assert result['input_hashes_after'] == before
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'residual_div_kphi2': result['projection']['residual_squared_div_kphi2'],
                      'independent_controls': ['exact_full60', 'tail', 'Gram_projection', 'Fourier_fullE'],
                      'scientific_admission': False}))


if __name__ == '__main__':
    main()
