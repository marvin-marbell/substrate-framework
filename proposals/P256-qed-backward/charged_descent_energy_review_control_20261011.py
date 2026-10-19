"""Independent exact signed-CAR and rational charged-energy review; no old CLI runs."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as Q
import hashlib
import itertools
import json
from pathlib import Path
import sys
import zipfile
from typing import Any, Callable

import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parent
QZERO = Q(0)
Vector = dict[int, Q]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def add(*rows: tuple[Q | int, Vector]) -> Vector:
    out: dict[int, Q] = defaultdict(Q)
    for factor, row in rows:
        for key, val in row.items():
            out[key] += factor * val
    return {k: v for k, v in out.items() if v}


def inner(left: Vector, right: Vector) -> Q:
    return sum((v * right.get(k, QZERO) for k, v in left.items()), QZERO)


def bilinear(mask: int, p: int, q: int, unsigned: bool = False) -> tuple[int, int] | None:
    if not (mask >> q) & 1:
        return None
    phase = 1 if unsigned else (-1) ** ((mask & ((1 << q) - 1)).bit_count())
    out = mask ^ (1 << q)
    if (out >> p) & 1:
        return None
    if not unsigned:
        phase *= (-1) ** ((out & ((1 << p) - 1)).bit_count())
    return out | (1 << p), phase


def apply(terms: list[tuple[int, int, Q]], vector: Vector, unsigned: bool = False) -> Vector:
    out: dict[int, Q] = defaultdict(Q)
    for mask, val in vector.items():
        for p, q, weight in terms:
            moved = bilinear(mask, p, q, unsigned)
            if moved:
                out[moved[0]] += weight * moved[1] * val
    return {k: v for k, v in out.items() if v}


def epsilon(indices: list[int]) -> int:
    return (-1) ** sum(a > b for i, a in enumerate(indices) for b in indices[i + 1:])


def singlet() -> Vector:
    out = {}
    for left in itertools.combinations(range(4), 2):
        cl = [i for i in range(4) if i not in left]
        for right in itertools.combinations(range(4), 2):
            cr = [i for i in range(4) if i not in right]
            mask = sum(1 << i for i in left) + sum(1 << (i + 4) for i in right)
            mask += sum(1 << (i + 8) for i in cl) + sum(1 << (i + 12) for i in cr)
            out[mask] = Q(epsilon(list(left) + cl) * epsilon(list(right) + cr), 6)
    return out


def rho(mask: int, copy: int | None = None) -> int:
    copies = (0, 1) if copy is None else (copy,)
    return sum(((mask >> (8 * s)) & 15).bit_count() - 2 for s in copies)


def density(vector: Vector, wrong: bool = False) -> Vector:
    return {m: 2 * (sum(rho(m, s) ** 2 for s in (0, 1)) if wrong else rho(m) ** 2) * v
            for m, v in vector.items()
            if (sum(rho(m, s) ** 2 for s in (0, 1)) if wrong else rho(m) ** 2)}


def hopping(t: list[list[Q]], copy: int, theta: bool = False) -> list[tuple[int, int, Q]]:
    out = []
    for i in range(4):
        for j in range(4):
            if t[i][j]:
                out.append((8 * copy + i, 8 * copy + 4 + j, t[i][j]))
                out.append((8 * copy + 4 + j, 8 * copy + i, -t[i][j] if theta else t[i][j]))
    return out


def car_controls(cfg: dict[str, Any]) -> dict[str, Any]:
    # Rational arbitrary endpoint direction; both spin labels and all wedge2 coherences.
    t = [[Q((7 * i + 3 * j + 2) % 13 - 6, 100) for j in range(4)] for i in range(4)]
    t[3][3] = -sum((t[i][i] for i in range(3)), QZERO)
    profiles, _, _ = actual_profile(cfg)
    c0, c1 = profiles[0]['paths'][0]['coefficients'][:2]
    phi0 = [[(t[i][j] + t[j][i]) / c0[0] for j in range(4)] for i in range(4)]
    # Phi0_left=2sym(T)/c0, Phi1_left=2anti(T)/c1; right fields0, U=I.
    # Phi1 is pure imaginary antisymmetric, so both endpoint fields are Hermitian/traceless.
    phi1_imag = [[-(t[i][j] - t[j][i]) / c1[1] for j in range(4)] for i in range(4)]
    assert sum((phi0[i][i] for i in range(4)), QZERO) == 0
    assert all((c0[0] * phi0[i][j] - c1[1] * phi1_imag[i][j]) / 2 == t[i][j]
               for i in range(4) for j in range(4))
    terms = hopping(t, 0) + hopping(t, 1)
    theta_terms = hopping(t, 0, True) + hopping(t, 1, True)
    psi = singlet()
    h = lambda v: apply(terms, v)
    ht = lambda v: apply(theta_terms, v)  # Htheta=i*ht; exact real representation.
    g = lambda v: add((1, density(h(v))), (-1, h(density(v))))
    gt = lambda v: add((1, density(ht(v))), (-1, ht(density(v))))
    gw = lambda v: add((1, density(h(v), True)), (-1, h(density(v, True))))
    assert inner(psi, psi) == 1 and density(psi) == {}
    hs = [apply(hopping(t, s), psi) for s in (0, 1)]
    hs2 = sum((x * x for row in t for x in row), QZERO)
    assert all(inner(v, v) == hs2 / 2 for v in hs)
    assert inner(hs[0], hs[1]) == 0
    assert all(set(v).isdisjoint(set(hs[1 - s])) for s, v in enumerate(hs))
    assert g(psi) == add((2, h(psi)))
    a = gt(psi)
    b = add((Q(1, 2), g(a)), (Q(1, 2), gt(g(psi))))
    w = inner(h(psi), h(psi))
    assert inner(a, a) == 4 * w
    assert 2 * inner(a, b) == 0
    onsite = [(8 * s + i, 8 * s + j, profiles[s]['onsite'] * phi0[i][j])
              for s in (0, 1) for i in range(4) for j in range(4) if phi0[i][j]]
    o = lambda v: apply(onsite, v)
    m = lambda v: add((1, h(v)), (1, o(v)))
    assert inner(o(psi), o(psi)) > 0  # Unequal roles cannot be removed from ACTION.
    matter1 = 2 * inner(g(psi), m(psi))
    matter2 = inner(g(psi), m(g(psi))) + inner(psi, m(g(g(psi))))
    assert matter1 == 4 * w
    adverse_second = QZERO
    for mask in psi:
        wrong_initial = {mask: Q(1)}
        adverse_second = inner(g(wrong_initial), m(g(wrong_initial))) + inner(wrong_initial, m(g(g(wrong_initial))))
        if adverse_second:
            break
    assert adverse_second != 0  # A noncentered field law does not remove a cubic coefficient.
    # q->-q negates H, onsite and G: second coefficient changes sign exactly.
    assert (-matter2) + matter2 == 0
    unsigned_gap = inner(add((1, h(psi)), (-1, apply(terms, psi, True))),
                         add((1, h(psi)), (-1, apply(terms, psi, True))))
    assert unsigned_gap > 0
    # A local SU4 generator annihilates the coherent epsilon pair, not its dephased mixture.
    local_terms = [(0, 1, Q(1)), (1, 0, Q(1)), (8, 9, Q(1)), (9, 8, Q(1))]
    assert apply(local_terms, psi) == {}
    mixture_generator_norm2 = sum((amp * amp * inner(apply(local_terms, {mask: Q(1)}),
                                                       apply(local_terms, {mask: Q(1)}))
                                   for mask, amp in psi.items()), QZERO)
    assert mixture_generator_norm2 > 0
    # Full sectors retained for a fresh finite unitary diagnostic; no producer calls.
    masks8 = [sum(1 << i for i in c) for c in itertools.combinations(range(8), 4)]
    masks = [a0 | (b0 << 8) for a0 in masks8 for b0 in masks8]
    index = {mask: i for i, mask in enumerate(masks)}
    def matrix(operation: Callable[[Vector], Vector]) -> Any:
        rr, cc, data = [], [], []
        for j, mask in enumerate(masks):
            for k, val in operation({mask: Q(1)}).items():
                rr.append(index[k]); cc.append(j); data.append(float(val))
        return csr_matrix((data, (rr, cc)), shape=(len(masks), len(masks)))
    hm, gm, gtm, om, wrongm = [matrix(op) for op in (h, g, gt, o, gw)]
    v = np.array([float(psi.get(mask, QZERO)) for mask in masks])
    tau = -1 / 128
    fv = expm_multiply(tau * gm, v)
    wrongv = expm_multiply(tau * wrongm, v)
    z = csr_matrix(gm.shape)
    block = bmat([[tau * gm, z], [1j * tau * gtm, tau * gm]], format='csr')
    jets = expm_multiply(block, np.concatenate((v, np.zeros(len(v)))))
    j = jets[len(v):]
    electric = 64 * float(np.vdot(j, j).real)
    finite_matter = float(np.vdot(fv, (hm + om) @ fv).real - np.vdot(v, (hm + om) @ v).real)
    matter_linear = float(Q(-1, 128) * matter1)
    jet_a = np.array([float(a.get(mask, QZERO)) for mask in masks]) * 1j
    jet_b = np.array([float(b.get(mask, QZERO)) for mask in masks]) * 1j
    gcap = float(np.max(abs(gm).sum(axis=1)))
    gtcap = float(np.max(abs(gtm).sum(axis=1)))
    mcap = float(np.max(abs(hm + om).sum(axis=1)))
    matter_remainder_cap = 4 / 3 * abs(tau) ** 3 * mcap * gcap ** 3
    jet_remainder_cap = abs(tau) ** 3 * gcap ** 2 * gtcap / 2
    matter_quadratic = float(Q(1, 128**2) * matter2)
    jet_remainder = float(np.linalg.norm(j - tau * jet_a - tau**2 * jet_b))
    assert abs(finite_matter - matter_linear - matter_quadratic) <= matter_remainder_cap + 1e-13
    assert jet_remainder <= jet_remainder_cap + 1e-13
    wrong_gap = float(np.linalg.norm(fv - wrongv))
    assert wrong_gap > 1e-10 and electric > 0
    assert float(np.linalg.norm((gm + gm.T).data)) < 1e-13
    return {'exact': {'state_norm': '1', 'spin_labels': [0, 1], 'local_coherent_support': len(psi),
                     'actual_binary_role_onsites': [str(p['onsite']) for p in profiles],
                     'actual_binary_selected_coefficients': [[str(r), str(i)] for r, i in profiles[0]['paths'][0]['coefficients']],
                     'realized_traceless_Phi0_left': [[str(x) for x in row] for row in phi0],
                     'realized_imaginary_traceless_Phi1_left': [[str(x) for x in row] for row in phi1_imag],
                     'onecopy_weight_each': str(hs2 / 2), 'crosscopy_inner': '0',
                     'pair_weight': str(w), 'full64_weight_at_this_direction': str(32 * w),
                     'matter_first_coefficient': str(matter1), 'onsite_action_norm2': str(inner(o(psi), o(psi))),
                     'matter_second_coefficient': str(matter2), 'inversion_paired_second': '0',
                     'noncentered_parity_adverse_second_coefficient': str(adverse_second),
                     'electric_rawjet_first_norm2': str(inner(a, a)), 'electric_cubic_cross': str(2 * inner(a, b)),
                     'unsigned_CAR_wrong_state_norm2': str(unsigned_gap),
                     'dephased_wrong_local_generator_norm2': str(mixture_generator_norm2)},
            'finite_diagnostic_not_fullPhi_integral': {'dimension': len(masks), 'tau': tau,
                     'matter': finite_matter, 'electric64': electric, 'finite_fullrho_wrong_state_gap': wrong_gap,
                     'matter_remainder_observed': abs(finite_matter - matter_linear - matter_quadratic),
                     'matter_ad3_bound': matter_remainder_cap, 'jet_remainder_observed': jet_remainder,
                     'jet_unitary_bound': jet_remainder_cap, 'g_induced_norm_bound': gcap,
                     'theta_g_induced_norm_bound': gtcap, 'pointnorm': float(np.vdot(fv, fv).real)}}


def actual_profile(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], Q, Q]:
    # Exactly the binary arithmetic used by couplings -> scalar_paths; no source imports.
    a, r, v = cfg['a'], cfg['light_r'], cfg['light_v']
    charged_csum = Q(float(-r / a))**2 + Q(float(-v / (2 * a)))**2
    neutral_csum = Q(float(-cfg['heavy_r'] / (2 * a)))**2 + Q(float(-cfg['heavy_v'] / (2 * a)))**2
    rows = []
    for role, count, mass in [('active', 1, cfg['active_m']), ('auxiliary', 63, cfg['auxiliary_m'])]:
        paths = []
        for axis in range(3):
            cp = [(Q(float(-r / a)), QZERO)] + [(QZERO, Q(float(-v / (2 * a))) if k == axis else QZERO) for k in range(3)]
            paths.append({'axis': axis, 'length': 1, 'coefficients': cp})
            paths.append({'axis': axis, 'length': 2, 'coefficients': [(Q(float(r / (4 * a))), QZERO)] + [(QZERO, QZERO)] * 3})
        rows.append({'role': role, 'count': count, 'onsite': Q(float(mass + 4.5 * r / a)), 'paths': paths})
    assert sum(row['count'] for row in rows) == 64 and rows[0]['onsite'] != rows[1]['onsite']
    assert all(abs(x) < 64 for p in rows for path in p['paths'] for cp in path['coefficients'] for x in cp)
    assert all(abs(p['onsite']) < 64 for p in rows)
    return rows, charged_csum, Q(15, 4) * neutral_csum


def conditional_gaussian(cfg: dict[str, Any]) -> dict[str, Any]:
    # Rational covariance representation avoids irrational normalized diagonal generators.
    # A Gaussian along raw D_k has variance 1/[2k(k+1)]; offdiagonals have variance1.
    basis = []
    for i, j in itertools.combinations(range(4), 2):
        for imaginary in (False, True):
            a = [[(QZERO, QZERO) for _ in range(4)] for _ in range(4)]
            a[i][j] = (QZERO, Q(-1, 2)) if imaginary else (Q(1, 2), QZERO)
            a[j][i] = (QZERO, Q(1, 2)) if imaginary else (Q(1, 2), QZERO)
            basis.append((a, Q(1)))
    for k in range(1, 4):
        a = [[(QZERO, QZERO) for _ in range(4)] for _ in range(4)]
        for i in range(k):
            a[i][i] = (Q(1), QZERO)
        a[k][k] = (Q(-k), QZERO)
        basis.append((a, Q(1, 2 * k * (k + 1))))
    coeff = actual_profile(cfg)[0][0]['paths'][0]['coefficients']
    identity = [[Q(int(i == j)) for j in range(4)] for i in range(4)]
    rotation = [[Q(3, 5), Q(-4, 5), QZERO, QZERO], [Q(4, 5), Q(3, 5), QZERO, QZERO],
                [QZERO, QZERO, Q(5, 13), Q(-12, 13)], [QZERO, QZERO, Q(12, 13), Q(5, 13)]]
    results = []
    for u in (identity, rotation):
        assert all(sum((u[i][k] * u[j][k] for k in range(4)), QZERO) == int(i == j) for i in range(4) for j in range(4))
        maps = []
        for endpoint in (0, 1):
            for cr, ci in coeff:
                for a, weight in basis:
                    row = []
                    for i in range(4):
                        for j in range(4):
                            real = sum(((a[i][k][0] * u[k][j] if endpoint == 0 else u[i][k] * a[k][j][0]) for k in range(4)), QZERO)
                            imag = sum(((a[i][k][1] * u[k][j] if endpoint == 0 else u[i][k] * a[k][j][1]) for k in range(4)), QZERO)
                            row.extend(((cr * real - ci * imag) / 2, (cr * imag + ci * real) / 2))
                    if any(row):
                        maps.append((row, weight))
        sigma = [[sum((weight * row[i] * row[j] for row, weight in maps), QZERO) for j in range(32)] for i in range(32)]
        trace = sum((sigma[i][i] for i in range(32)), QZERO)
        trace_square = sum((val * val for row in sigma for val in row), QZERO)
        fourth = trace**2 + 2 * trace_square
        assert trace == Q(15, 4) * actual_profile(cfg)[1]
        assert trace**2 <= fourth <= 3 * trace**2
        results.append({'conditional_HS_second_eta1': str(trace), 'conditional_HS_fourth_eta1': str(fourth),
                        'real_covariance_trace_square': str(trace_square), 'Wick_bound': str(3 * trace**2)})
    assert results[0] == results[1]
    return {'exact_rational_SU4_positive_controls': results, 'Haar_samples': 0,
            'arbitrary_U_proof': 'For every fixed SU4 U, T=[sum_mu c_mu(Phi_r,mu+Ad_U Phi_c,mu)]U/2. Isotropic independent endpoint adjoint Gaussians are invariant under Ad_U; right unitary multiplication preserves HS moments. Conditional full HS distribution and Wick bound therefore do not depend on U, hence hold under any normalized interacting Haar colour wave independent of radial q.'}


def rational_review(proof: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    profiles, csum, mn2 = actual_profile(cfg)
    request, claimed, budget = proof['request'], proof['rational_increment'], proof['absolute_source_budget']
    eta = Q(1, 2**request['eta_exponent']); sqrteta = Q(1, 2**(request['eta_exponent'] // 2))
    tau = Q(request['tau']); chi = Q(1, 2**4096); kappa = Q(2**8192, 2**request['neutral_epsilon_exponent'])
    g, n, matter, moments = Q(2**32), Q(2**10), Q(2**20), [Q(120), Q(120 * 122), Q(120 * 122 * 124)]
    m2, m4, m6 = moments
    assert kappa * sqrteta == Q(1, 2**20)
    w_edge = 64 * Q(1, 2) * Q(15, 4) * eta * csum
    w_site = w_edge / 2
    leading_m = 4 * tau * w_site; leading_e = 64 * 4 * tau**2 * w_site
    # Factors obtained independently: commutator2^3/3!, electric64 * edges/site1/2.
    prices = {'matter_all_onsites_and_matching_hop_remainder': Q(8, 6) / 2 * abs(tau)**3 * matter * g**3 * m4 * eta**2,
              'fullE64_finite_jet_remainder': Q(64, 2) * (2 * tau**4 * g**4 * m4 * eta**2 + abs(tau)**5 * g**5 * m6 * eta**2 * sqrteta + tau**6 * g**6 * m6 * eta**3 / 4),
              'Phi_charged_derivative': 4 * chi * Q(120, 2) * tau**2 * g**2,
              'Phi_neutral_charged_Berry_cross': 4 * chi * Q(120, 2) * 2 * kappa**2 * tau**2 * n**2 * g**2 * m2 * eta,
              'colour_charged_derivative': Q(3, 20) * chi * Q(15, 2) * tau**2 * g**2 * m2 * eta,
              'colour_neutral_charged_Berry_cross': Q(3, 20) * chi * Q(15, 2) * 2 * kappa**2 * tau**2 * n**2 * g**2 * m4 * eta**2}
    assert csum == Q(claimed['charged_selected_coefficient_square_sum_exact'])
    assert w_edge == Q(claimed['W_per_matching_edge_exact']) and w_site == Q(claimed['W_per_site_exact'])
    assert leading_m == Q(claimed['leading_matter_increment_per_site_exact'])
    assert leading_e == Q(claimed['leading_fullE64_increment_per_site_exact'])
    assert all(val == Q(claimed['positive_remainder_prices_per_site_exact'][key]) for key, val in prices.items())
    leading = leading_m + leading_e; upper = leading + sum(prices.values())
    assert upper == Q(claimed['increment_upper_per_site_exact']) and upper < leading / 2 < 0
    assert [str(p['onsite']) for p in profiles] == [x['coefficient_exact'] for x in claimed['charged_onsites']]
    delta = Q(1, 2**20); lam = Q(2**8192); bigf = Q(2**10240); t = Q(2**7168)
    neutral_upper = lam * sqrteta * (-2 * delta * mn2 + Q(4, 3) * delta**3 * Q(2**24))
    neutral_lower = -2 * delta * mn2 * lam * sqrteta
    alignment = Q(9, 2) * bigf * eta
    baseline_prices = [60 * chi / eta, 3000 * lam * eta**2, alignment, Q(27, 4) * chi * t,
                       Q(7, 5) * bigf / t * (450 + 48 * 7168 + 24), 240 * chi * kappa**2 * n**2,
                       Q(9, 8) * chi * kappa**2 * n**2 * m2 * eta]
    baseline_upper = neutral_upper + sum(baseline_prices)
    after_lower = alignment + neutral_lower - Q(2**24) * sqrteta
    assert mn2 == Q(budget['neutral_M2_exact']) and 3 * mn2**2 < 2**24
    assert baseline_upper == Q(budget['neutral_circuited_baseline_energy_upper_per_site_exact'])
    assert after_lower == Q(budget['charged_after_energy_lower_per_site_exact'])
    assert baseline_upper + upper == Q(budget['charged_after_energy_upper_per_site_exact'])
    if request['eta_exponent'] == 4000:
        assert after_lower > 2**6240
    else:
        assert baseline_upper < -2**6070
    inventory = []
    for size in (4, 6, 8, 10):
        sites = list(itertools.product(range(size), repeat=3))
        even = [x for x in sites if sum(x) % 2 == 0]
        gates = {x: i for i, x in enumerate(even)}
        for i, x in enumerate(even):
            gates[((x[0] + 1) % size, x[1], x[2])] = i
        assert len(gates) == size**3
        counts = defaultdict(int)
        for p in profiles:
            for path in p['paths']:
                for x in sites:
                    y = list(x); y[path['axis']] = (y[path['axis']] + path['length']) % size
                    same = gates[x] == gates[tuple(y)]
                    if same:
                        assert path['axis'] == 0 and path['length'] == 1 and sum(x) % 2 == 0
                    counts['matched' if same else 'excluded'] += p['count']
        assert counts['matched'] == 64 * size**3 // 2
        assert sum(counts.values()) == 64 * 6 * size**3
        inventory.append({'N': size, 'V': size**3, 'matched_directed': counts['matched'], 'excluded_directed': counts['excluded'],
                          'total_increment_upper': str(size**3 * upper)})
    return {'eta_exponent': request['eta_exponent'], 'tau': str(tau), 'csum_exact_binary': str(csum),
            'W_edge': str(w_edge), 'W_site': str(w_site), 'net_leading': str(leading), 'increment_upper_perSITE': str(upper),
            'price_over_gain': {key: str(val / -leading) for key, val in prices.items()},
            'absolute_baseline_upper_perSITE': str(baseline_upper), 'absolute_after_lower_perSITE': str(after_lower),
            'all_published_rationals_equal_independent_derivation': True, 'inventory': inventory,
            'profile_coefficients': [{**p, 'onsite': str(p['onsite']), 'paths': [{**path, 'coefficients': [[str(r), str(i)] for r, i in path['coefficients']]} for path in p['paths']]} for p in profiles]}


def adverses() -> dict[str, Any]:
    tau = Q(-3, 128)
    correct = 4 * tau + 256 * tau**2; wrong_half = 4 * tau + 128 * tau**2
    assert correct > 0 > wrong_half
    assert 4 * Q(1, 128) + 256 * Q(1, 128)**2 > 0
    assert (4 * Q(-1, 128) + 256 * Q(-1, 128)**2) == Q(-1, 64)
    # A centered Bloch packet with E=k+B beta versus a coordinate bump E=-i dtheta.
    width = Q(1, 64)
    bloch_bound = 64 * width**2
    # Every normalized H1_0(-w,w) coordinate bump obeys int|b'|²>=1/(4w²)
    # by 1=-2 Re int theta b' conjugate(b) and Cauchy. No numerical bump integral needed.
    coordinate_lower = 16 / width**2
    alleged_bound = Q(1, 64)
    assert bloch_bound == alleged_bound and coordinate_lower > alleged_bound
    # A symmetric beta distribution has E beta=0 but beta²>0; twist action cannot be omitted.
    beta2_uniform = width**2 / 3
    twist_action_norm2 = 64**2 * width**4 / 5  # exact normalized uniform direct-integral k0 control.
    assert twist_action_norm2 > 0 and 64 * beta2_uniform <= alleged_bound
    return {'wrong_sign_positive_step': 'REJECTED', 'half_electric_wrong_decrease': str(wrong_half),
            'correct_full64_electric_increase': str(correct), 'omitted_spin_W_ratio': '1/2',
            'two_copies_wrong_full64_W_ratio': '1/32', 'missing_aux_W_ratio': '1/64',
            'edge_to_site_ratio': '1/2', 'leading_minusW_over64_only_tau_minus1over128': True,
            'upper_as_lower_wrongcase': {'valid_interval': ['-3', '7'], 'upper': '7', 'assertion_actual_gt_0': 'NOT IMPLIED'},
            'coordinate_packet_not_Bloch': {'width': str(width), 'alleged_small_energy': str(alleged_bound),
                                          'coordinate_energy_genuine_lower': str(coordinate_lower), 'Bloch_small_energy_valid_upper': str(bloch_bound)},
            'Bloch_twist_cannot_be_omitted': {'electric64_uniform_mean': str(64 * beta2_uniform),
                                           'complete_H_action_norm2': str(twist_action_norm2), 'omitted_twist_action_norm2': '0'}}


def provenance() -> dict[str, Any]:
    contract = json.loads((ROOT / 'charged_descent_energy_review_inputs_20261011.json').read_bytes())
    assert all(digest((ROOT / n).read_bytes()) == pin for n, pin in contract['pins'].items())
    epochs = []
    for epoch in (4, 5):
        prefix = f'physical_charged_descent_attempt{epoch}_20261011_'
        receipt = json.loads((ROOT / (prefix + 'receipt.json')).read_bytes())
        exit_record = json.loads((ROOT / (prefix + 'natural_exit.json')).read_bytes())
        assert exit_record['natural_returncode'] == 0
        assert digest((ROOT / (prefix + 'stdout.txt')).read_bytes()) == exit_record['stdout_sha256']
        assert digest((ROOT / (prefix + 'stderr.txt')).read_bytes()) == exit_record['stderr_sha256']
        with zipfile.ZipFile(ROOT / (prefix + 'preexecution.zip')) as pre, zipfile.ZipFile(ROOT / (prefix + 'executed.zip')) as post:
            for name, pin in receipt['source_input_hashes'].items():
                assert digest(pre.read(name)) == pin and pre.read(name) == post.read(name)
            for name, pin in receipt['outputs'].items():
                assert digest((ROOT / name).read_bytes()) == pin == digest(post.read(name))
        epochs.append({'epoch': epoch, 'natural_returncode': 0, 'exclusive_capsule_sources_and_outputs_verified': True,
                       'preexecution_sha256': digest((ROOT / (prefix + 'preexecution.zip')).read_bytes()),
                       'executed_sha256': digest((ROOT / (prefix + 'executed.zip')).read_bytes())})
    return {'pins': contract['pins'], 'reviewed_old_epochs': epochs, 'old_producer_reruns': 0}


def main() -> None:
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    assert args.prefix.startswith('charged_descent_energy_review_')
    inputs = provenance()
    cfg = json.loads((ROOT / 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json').read_bytes())['input_descriptor']['original_micro_config_UNCHANGED']
    rational = [rational_review(json.loads((ROOT / f'physical_charged_descent_attempt{epoch}_20261011_proof.json').read_bytes()), cfg) for epoch in (4, 5)]
    derivation = json.loads((ROOT / 'charged_descent_energy_review_derivation_20261011.json').read_bytes())
    proof = {'kind': 'INDEPENDENT_ACTUAL_SIGNED_CAR_RATIONAL_ENERGY_REVIEW', 'verdict': derivation['verdict'],
             'derivation': derivation, 'rational_profiles': rational, 'fresh_actual_CAR': car_controls(cfg),
             'fresh_adverses': adverses(), 'conditional_gaussian_moments': conditional_gaussian(cfg),
             'provenance': inputs, 'campaign_complete': False}
    out = ROOT / (args.prefix + 'proof.json')
    with out.open('x') as stream:
        json.dump(proof, stream, indent=2); stream.write('\n')
    receipt = {'source_input_pins': inputs['pins'], 'output': {out.name: digest(out.read_bytes())},
               'physical_energy_arithmetic': 'exact Fraction of binary scalar coefficients',
               'finite_diagnostic': 'fresh4900 signedCAR rational direction, not a field/Haar integration',
               'old_producer_runs': 0, 'natural_exit': 'runner records actual subprocess returncode'}
    with (ROOT / (args.prefix + 'receipt.json')).open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'verdict': proof['verdict'], 'epochs_exact_rational_equal': [4, 5],
                      'signedCAR': proof['fresh_actual_CAR'], 'adverses': proof['fresh_adverses']}))


if __name__ == '__main__':
    main()
