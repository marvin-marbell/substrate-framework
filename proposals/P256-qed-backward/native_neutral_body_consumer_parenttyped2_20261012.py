"""Exact complete-canonical body bounds and a full-field polynomial-filter branch.

No optimizer, numerical ground vector, frozen-field quantum state or phase claim.
The second construction is a physical Gaussian/Haar/CAR section, not a BO ansatz.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, cast

import sympy as sp

ROOT = Path(__file__).resolve().parent


def dyadic(exponent: int) -> Q:
    return Q(2**exponent) if exponent >= 0 else Q(1, 2**(-exponent))


def log_upper(value: Q) -> int:
    assert value > 0
    return value.numerator.bit_length() - value.denominator.bit_length() + 1


def matrices() -> tuple[sp.Matrix, list[sp.Matrix]]:
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.diag(1, -1)
    return cast(sp.Matrix, sp.kronecker_product(sz, sp.eye(2))), [cast(sp.Matrix, sp.kronecker_product(sx, a)) for a in (sx, sy, sz)]


def jordan(fields: list[sp.Matrix]) -> sp.Expr:
    return sp.expand(4 * sum(sp.trace((a*b+b*a)**2) for a, b in itertools.product(fields, repeat=2)))


def canonical_controls(profile: dict[str, Any]) -> dict[str, Any]:
    m = sp.Rational(profile['onsite'])
    c0 = sp.Rational(profile['paths'][0]['c0'])
    ci = sp.Rational(profile['paths'][0]['ci'])
    assert len(profile['paths']) == 3
    assert all(p['length'] == 1 and sp.Rational(p['c0']) == c0 and sp.Rational(p['ci']) == ci for p in profile['paths'])
    beta, alpha = matrices()
    s, r = sp.Rational(1, 2), sp.Rational(1, 3)
    clifford = [s*beta, *[r*a for a in alpha]]
    commuting = [s*beta, *[r*beta for _ in alpha]]
    jc, jm = jordan(clifford), jordan(commuting)
    diagonal_only = 4 * sum(sp.trace((2*a*a)**2) for a in commuting)
    assert jc == 64*(s**4+3*r**4)
    assert jm == 64*(s*s+3*r*r)**2 and jm > diagonal_only == jc
    mmin = m - 6*abs(c0)
    assert mmin > sp.Rational(1, 8)
    # Complete canonical2V means a GLOBAL Ky Fan sum, not two levels at each k.
    # D is traceless Hermitian but non-Clifford; there are only V positive levels.
    d = sp.diag(3, -1, -1, -1)
    sd = sp.Rational(1, 4)
    masses = [m+c0*sp.sqrt(2)*j for j in (3, 1, -1, -3)]
    counts = [8, 24, 24, 8]
    assert all(a > 0 for a in masses)
    assert all(masses[i] < masses[i+1] for i in range(3))
    bands: list[sp.Expr] = []
    for a, count in zip(reversed(masses), reversed(counts)):
        bands.extend([3*sd*a]*count)
    for a, count in zip(masses, counts):
        bands.extend([-sd*a]*(3*count))
    sea = sp.simplify(cast(sp.Expr, sum(bands[:128])))
    expected = 128*sd*m - 112*sd*c0*sp.sqrt(2)
    assert sp.simplify(sea-expected) == 0
    per_k_wrong = 128*sd*m
    assert sea > per_k_wrong
    assert len([a for a in bands[:128] if a < 0]) == 64
    nonclifford_j = jordan([sd*d, sp.zeros(4), sp.zeros(4), sp.zeros(4)])
    assert nonclifford_j == 1344*sd**4
    # Staggered Phi0 has EVERY neutral hopping zero by endpoint averaging,
    # but a strictly positive enormous alignment price. AP signs cannot change it.
    staggered_leading = 64*s**4 - 2*m*s + sp.Rational(6, 5)*2**2048*s**2
    assert staggered_leading > 0
    assert 64*s**4-2*m*s < 0
    # Every N4 flat centre twist permutes the AP momentum set, not one sea point.
    momenta = tuple(range(1, 8, 2))
    centre_permutations = {str(shift): sorted((k+shift) % 8 for k in momenta) for shift in (0, 2, 4, 6)}
    assert all(a == list(momenta) for a in centre_permutations.values())
    # A complete canonical Clifford trial upper follows without a band optimizer:
    # epsilon(k)>=s mass(k)>0 and mean mass=m for every AP N>=4.
    leading_upper = jc - 2*s*m
    assert leading_upper < -16
    c = max(m*m+6*c0*c0, 2*ci*ci)
    assert c < 1024
    # Universal all-field lower is -48; overlap is a FAILED global sharpness gate.
    assert leading_upper > -48
    return {'canonical_particles_N4': 128, 'canonical_positive_levels_in_nonClifford_control': 64,
            'canonical_negative_levels_also_required': 64,
            'nonClifford_global_KyFan_sum_exact': str(sea),
            'wrong_two_per_momentum_sum_exact': str(per_k_wrong),
            'wrong_positive_sheet_only_and_per_momentum_occupation_rejected': True,
            'physical_minus_Lambda_energy_exact': str(-sea),
            'wrong_plus_Lambda_same_occupation_energy_exact': str(sea),
            'ordered_Clifford_J_exact': str(jc), 'ordered_commuting_J_exact': str(jm),
            'wrong_diagonal_only_commuting_J_exact': str(diagonal_only),
            'ordered_cross_omission_rejected': True,
            'nonClifford_J_per_site_exact': str(nonclifford_j),
            'nonClifford_complete_energy_over_Lambda_per_site_exact': str(sp.simplify(nonclifford_j-sea/64)),
            'inhomogeneous_staggered_alignment_over_F_per_site_exact': '3/10',
            'inhomogeneous_drop_alignment_false_negative_rejected': True,
            'N4_flat_centre_AP_permutations': centre_permutations,
            'all_momenta_mass_floor_exact': str(mmin),
            'all_field_trace_square_constant_exact': str(c),
            'leading_global_lower_over_Lambda_per_site': '-48',
            'leading_global_trial_upper_over_Lambda_per_site_exact': str(leading_upper),
            'global_sharpness_gate': 'FAILED: certified intervals overlap; no actual minimizer identification',
            'Clifford_window_certified': False}


def car(mask: int, p: int, q: int) -> tuple[int, int] | None:
    if not mask & (1 << q):
        return None
    sign = (-1)**((mask & ((1 << q)-1)).bit_count())
    mask ^= 1 << q
    if mask & (1 << p):
        return None
    sign *= (-1)**((mask & ((1 << p)-1)).bit_count())
    return mask | (1 << p), sign


def lift(one: sp.Matrix, masks: list[int]) -> sp.Matrix:
    index = {mask: i for i, mask in enumerate(masks)}
    out = sp.zeros(len(masks))
    for j, mask in enumerate(masks):
        for p in range(8):
            for q in range(8):
                if one[p, q] == 0:
                    continue
                result = car(mask, p, q)
                if result is not None:
                    target, sign = result
                    out[index[target], j] += sign*one[p, q]
    return out


def polynomial_controls(profile: dict[str, Any]) -> dict[str, Any]:
    beta, alpha = matrices()
    # Inhomogeneous actual endpoint fields, NOT a Clifford ray; exact SU4 link.
    left = [sp.diag(3, -1, -1, -1)/7, alpha[0]/5, alpha[1]/11, alpha[2]/13]
    right = [beta/3, (beta+alpha[0])/17, alpha[1]/19, (beta+alpha[2])/23]
    u = sp.diag(sp.I, -sp.I, 1, 1)
    assert u.det() == 1 and u*u.H == sp.eye(4)
    c0 = sp.Rational(profile['paths'][0]['c0'])
    ci = sp.Rational(profile['paths'][0]['ci'])
    f = c0*(left[0]*u+u*right[0])/2 + sp.I*ci*(left[1]*u+u*right[1])/2
    hop = sp.zeros(8)
    hop[:4, 4:] = f
    hop[4:, :4] = f.H
    onsite = sp.diag(sp.Rational(profile['onsite'])*left[0], sp.Rational(profile['onsite'])*right[0])
    masks = [sum(1 << i for i in inds) for inds in itertools.combinations(range(8), 4)]
    h = lift(hop, masks)
    full = h+lift(onsite, masks)
    p = sp.zeros(70, 1)
    p[masks.index(15)] = 1
    w = h*p
    b = sp.simplify((w.H*w)[0])
    assert b == sp.simplify(sp.trace(f*f.H)) and b > 0
    assert (p.H*w)[0] == 0 and (p.H*full*p)[0] == 0
    # Positive field / negative field pair is the actual centred Gaussian odd rule.
    tau = sp.Rational(1, 1024)
    plus, minus = p+tau*w, p-tau*w
    norm = sp.simplify((plus.H*plus)[0])
    assert norm == 1+tau*tau*b
    numerator = sp.simplify(((plus.H*full*plus)[0]+(minus.H*(-full)*minus)[0])/2)
    assert sp.simplify(numerator-2*tau*b) == 0
    correct = sp.simplify(-numerator/norm)
    assert correct < 0 and -correct > 0
    # Fermionic sign witness; this is not a distinguishable-particle tensor hop.
    sign = car(15, 4, 1)
    assert sign == (29, 1)
    return {'canonical_four_particle_pair_dimension': 70, 'full_local_onsite_hopping_retained': True,
            'inhomogeneous_nonClifford_endpoint_SU4_exact': True,
            'filter_norm_exact': str(norm), 'B_pair_exact': str(b),
            'Gaussian_reflection_matter_numerator_exact': str(numerator),
            'physical_minus_Lambda_filter_energy_exact_at_control': str(correct),
            'wrong_scale_or_filter_sign_rejected': True,
            'canonical_number_preserved': True, 'CAR_sign_witness': list(cast(tuple[int, int], sign)),
            'filter_not_exponential_or_frozen_band_state': True}


def quantum_prices(profile: dict[str, Any]) -> dict[str, Any]:
    eta, strength, colour_t = dyadic(-7168), dyadic(3072), dyadic(7168)
    chi, lam, big_f = dyadic(-4096), dyadic(8192), dyadic(10240)
    c0, ci = Q(profile['paths'][0]['c0']), Q(profile['paths'][0]['ci'])
    b = Q(15, 4)*(c0*c0+ci*ci)
    cap = max(c0*c0, ci*ci)/4
    x = strength*strength*b*eta
    z = 1+x
    assert cap*124*strength*strength*eta < 1
    assert b > 256 and x < dyadic(-1000)
    gain = lam*strength*b*eta/z
    # Cartesian expression is exact. Other positive prices are rigorous uppers.
    cart = chi/eta*(60+x/z)
    prices = {'Phi_Cartesian': cart,
              'Jordan_ordered': 3000*lam*eta*eta*(1+124*cap*strength*strength*eta)/z,
              'alignment': Q(93, 10)*big_f*eta,
              'Haar_scalar_plus_cross': Q(27, 4)*chi*colour_t,
              'Haar_filter_plus_cross': 270*2**20*chi*strength*strength*eta/z,
              # log2<1 gives a rational outward Wilson-colour bound.
              'colour_plaquette': Q(7, 5)*big_f/colour_t*(450+48*7168+24),
              'full_shared_U1_HE_plus_HC_once': Q(262144)}
    upper = sum(prices.values(), Q(0))-gain
    assert gain > dyadic(4096)
    assert sum(prices.values(), Q(0)) < dyadic(3092)
    assert upper < -dyadic(4095)
    return {'eta': '2^-7168', 'neutral_polynomial_strength': '2^3072', 'colour_Haar_t': '2^7168',
            'pair_B_expectation_over_eta_exact': str(b),
            'pair_B_Cart_norm_cap_exact': str(cap),
            'pair_normalizer': 'Z_pair=1+t_neutral^2*b*eta',
            'x_log2_upper': log_upper(x),
            'negative_gain_log2_lower': gain.numerator.bit_length()-gain.denominator.bit_length()-1,
            'positive_prices_log2_upper': {k: log_upper(v) for k, v in prices.items()},
            'Phi_Cartesian_exact_formula': '(chi/eta)*(60+x/(1+x)) per site',
            'quantum_energy_upper_per_site': 'strictly less than -2^4095',
            'same_source_before_filter_energy_lower_per_site': '262144',
            'strict_fullH_variational_gain': True,
            'uniform_all_even_N_ge4': True,
            'quantum_trial_Berry_Haar_and_Gauss_prices_paid': True,
            'ground_BO_Berry_error_paid': False,
            'ground_Clifford_or_local_gap_support_paid': False}


def main() -> None:
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interface', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for locator, digest in request['pins'].items():
        path = Path(locator)
        if not path.is_absolute():
            path = ROOT/path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, locator
    profiles = json.loads((ROOT/request['profiles']).read_text())
    profile = profiles['primitive_profiles'][0]
    assert profile['role'] == 'neutral heavy -Lambda' and profile['load'] == 1
    leading = canonical_controls(profile)
    filter_controls = polynomial_controls(profile)
    quantum = quantum_prices(profile)
    receipt = {'kind': 'EXACT_COMPLETE_CANONICAL_LEADING_BOUNDS_AND_EXECUTED_FULL_FIELD_POLYNOMIAL_QUANTUM_BRANCH',
               'pins': request['pins'], 'leading': leading, 'polynomial_controls': filter_controls,
               'quantum': quantum, 'law_changed': False, 'ground_vector_required': False,
               'scientific_admission': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    interface = {'kind': 'COMPLETE_BODY_BOUNDS_WITH_FAILED_GLOBAL_IDENTIFICATION_AND_PHYSICAL_QUANTUM_VARIATIONAL_BRANCH',
                 'authoritative_contract': request['contract'], 'pins': request['pins'],
                 'proof': request['proof'], 'receipt': args.output.name,
                 'full_module': 'L2(R^(60V),Cart) tensor L2(SU4^(3V),Haar) tensor native noncompact-U1 Bloch fibre/packets tensor neutral canonical2V tensor active1+aux63 canonical2V each tensor full Ref, restricted by quantum SU4/U1 Gauss',
                 'leading_functional': 'Lambda*[4 sum_ordered x,mu,nu Tr({Phi_mu,Phi_nu}^2)-KyFan_top_2V(h_neutral(Phi,U))]+(F/40)A+(7F/10)C',
                 'actual_ground_authority': 'Inherited finite global minimizing character/ground family, translation averaged; NOT the trial filter or point sea',
                 'global_leading_bounds': leading,
                 'quantum_fullH_variational_comparison': quantum,
                 'ground_energy_lower_per_site': '-8192 Lambda inherited fullH coercivity',
                 'body_support': {'actual_Clifford_window': None, 'quantum_BO_remainder': None,
                                  'ground_local_moments': 's_x=sum_mu||Phi_mu,x||HS^2: <s_x^2><16384; <w_x><2^-2035 from inherited coercivity and new negative upper',
                                  'point_uniform_neutral_gap': 'Conditional Clifford s>=1/2: separation >= m_min>1/8, multiplied by Lambda; NOT fullH gap',
                                  'mass_floor_exact': leading['all_momenta_mass_floor_exact']},
                 'verified_windows': [],
                 'debts': ['No classification of non-Clifford/inhomogeneous minimizing fields or flat SU4 centre/character winners; universal bounds cover but do not identify them.',
                           'No quantum BO approximation close to the global leading infimum; trial derivative prices are not an actual-ground Berry correction.',
                           'No actual originating family support in the charged conditional Clifford window; no controlled ground fast-neutral gap.',
                           'No U1 stiffness, uniform fullH gap, photon atom, phase or original SOURCE adoption.'],
                 'law_changed': False, 'scientific_admission': False, 'campaign_complete': False}
    args.interface.write_text(json.dumps(interface, indent=2)+'\n')
    print(json.dumps({'global_gate': leading['global_sharpness_gate'], 'quantum_energy_upper': quantum['quantum_energy_upper_per_site'], 'all_even_N_ge4': True, 'complete_canonical_competing_field_control': True}))


if __name__ == '__main__':
    main()
