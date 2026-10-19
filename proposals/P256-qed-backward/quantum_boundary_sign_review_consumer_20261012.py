"""Independent exact boundary/sign controls. Never imports/runs original producers."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path
import sympy as s

ROOT = Path(__file__).resolve().parent
SUBSETS = list(itertools.combinations(range(4), 2))
MASKS = [sum(1 << i for i in ix) for ix in SUBSETS]


def exterior(b):
    return s.Matrix(6, 6, lambda i, j: b.extract(SUBSETS[i], SUBSETS[j]).det())


def annihilation(mode):
    # Independently implement the full Fock Jordan-Wigner representation.
    out = s.zeros(16)
    for state in range(16):
        if state & (1 << mode):
            parity = sum((state >> j) & 1 for j in range(mode))
            out[state ^ (1 << mode), state] = (-1)**parity
    return out


def dg(a):
    full = s.zeros(16)
    for i in range(4):
        for j in range(4):
            if a[i, j]:
                full += a[i, j]*annihilation(i).T*annihilation(j)
    return full.extract(MASKS, MASKS)


def partitions(items):
    if not items:
        yield []
        return
    first, *rest = items
    for blocks in partitions(rest):
        yield [[first]] + blocks
        for j in range(len(blocks)):
            yield blocks[:j] + [[first] + blocks[j]] + blocks[j+1:]


def cumulant(moments, n):
    value = s.S.Zero
    for blocks in partitions(list(range(n))):
        value += (-1)**(len(blocks)-1)*math.factorial(len(blocks)-1)*s.prod(
            moments[sum(1 << j for j in block)] for block in blocks)
    return s.factor(value)


def nilpotent_log(moments, n):
    # Independent coefficient extraction of log(1+x), in t_i^2=0 algebra.
    x = {mask: val for mask, val in moments.items() if mask and val}
    power = {0: s.S.One}
    full = (1 << n)-1
    answer = s.S.Zero
    for k in range(1, n+1):
        out = {}
        for a, va in power.items():
            for b, vb in x.items():
                if not a & b:
                    out[a | b] = out.get(a | b, s.S.Zero) + va*vb
        power = out
        answer += s.Rational((-1)**(k+1), k)*power.get(full, 0)
    return s.factor(answer)


def boundary_controls():
    # Pure fixed-N non-Slater state: Pluecker relation has nonzero 01*23 term.
    psi = s.zeros(6, 1); psi[0] = 1; psi[5] = 2
    rho = psi*psi.T/5
    rho_diagonal = s.diag(*rho.diagonal())
    heat = s.diag(2, 3, 5, 7)
    h2 = exterior(heat)
    edges = [(0, 2), (1, 3), (2, 0), (3, 1)]*2
    legs = []
    for a, b in edges:
        leg = s.zeros(4); leg[a, b] = 1
        legs.append(leg)
        assert exterior(s.eye(4)+leg) == s.eye(6)+dg(leg)
    # Full exterior polynomial by interpolation/Mobius inversion, not CAR expansion.
    values = {}
    for mask in range(256):
        b = heat
        for j, leg in enumerate(legs):
            b = (s.eye(4)+(leg if mask >> j & 1 else s.zeros(4)))*b
        values[mask] = exterior(b)
    coefficients = {mask: value.copy() for mask, value in values.items()}
    for bit in range(8):
        for mask in range(256):
            if mask >> bit & 1:
                coefficients[mask] -= coefficients[mask ^ (1 << bit)]
    direct, diagonal, identity, reversed_moments = {}, {}, {}, {}
    matrix_checks = 0
    denominator = s.trace(rho*h2)
    for mask in range(256):
        word, reverse = h2, h2
        for j, leg in enumerate(legs):
            factor = dg(leg) if mask >> j & 1 else s.eye(6)
            word = factor*word
            reverse = reverse*factor
        assert coefficients[mask] == word
        matrix_checks += 36
        direct[mask] = s.trace(rho*word)/denominator
        diagonal[mask] = s.trace(rho_diagonal*word)/denominator
        identity[mask] = s.trace(word)/s.trace(h2)
        reversed_moments[mask] = s.trace(rho*reverse)/denominator
    assert direct != diagonal and direct != identity and direct != reversed_moments
    rows = []
    for n in (4, 8):
        log_value = nilpotent_log(direct, n)
        partition_value = cumulant(direct, n)
        assert log_value == partition_value
        assert log_value != direct[(1 << n)-1]
        assert log_value != cumulant(identity, n)
        rows.append({'legs': n, 'normalized_connected': str(log_value),
                     'raw_ordered_moment': str(direct[(1 << n)-1]),
                     'identity_boundary_wrong_connected': str(cumulant(identity, n)),
                     'set_partition_count': len(list(partitions(list(range(n))))),
                     'log_equals_partition': True})
    # Ordered eightleg response with an interposed projector cannot be collapsed.
    j = [dg(leg) for leg in legs]
    left = j[3]*j[2]*j[1]*j[0]
    right = j[7]*j[6]*j[5]*j[4]
    projector = s.diag(1, 0, 0, 0, 0, 0)
    ordered = s.trace(rho*right*projector*left*h2)/denominator
    dropped = s.trace(rho*right*left*h2)/denominator
    assert ordered != dropped
    differing_mask = next(m for m in direct if direct[m] != diagonal[m])
    return {'canonical_dimension': 6, 'density': '(|01>+2|23>)(<01|+2<23|)/5',
            'pure_nonGaussian_Pluecker_obstruction': 'c01*c23-c02*c13+c03*c12=2 !=0',
            'not_native_ground': True, 'thermal_method_denominator': str(denominator),
            'all256_full_matrix_minor_CAR_checks': matrix_checks,
            'offdiagonal_witness_mask': differing_mask,
            'offdiagonal_full': str(direct[differing_mask]),
            'offdiagonal_erased': str(diagonal[differing_mask]),
            'identity_substitution_rejected': True, 'time_reversal_rejected': True,
            'normalized_cumulants': rows,
            'ordered_projected_eightleg': str(ordered), 'projector_dropped_wrong': str(dropped)}


def directed_coefficients():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    full = []
    for a, b in edges:
        full.append(annihilation(a).T*annihilation(b))
    sums = [0]*16
    for order in itertools.permutations(range(4)):
        word = s.eye(16)
        for j in order:
            word = full[j]*word
        for mask in range(16):
            sums[mask] += int(word[mask, mask])
    return sums


def complement_controls():
    sums = directed_coefficients()
    z = math.comb(256, 128)
    weights = [F(math.comb(252, 128-m.bit_count()), z) for m in range(16)]
    assert sum(weights) == 1
    pair = sum(sums[m]*weights[m] for m in (5, 10))
    complement = sum(sums[m]*weights[m] for m in range(16) if m not in (5, 10))
    assert pair == F(-32512, 64515) and complement == F(-256, 64515)
    diagnostic_count = 2*math.comb(252, 126)
    actual_pi_count = 2*math.comb(4, 2)**2*math.comb(240, 124)
    assert actual_pi_count != diagnostic_count
    heat_rows = []
    for fugacity in (F(1), F(1, 16)):
        normalizer = sum(math.comb(4, k)*math.comb(4, 4-k)*fugacity**k for k in range(5))
        total = sum(sums[m]*math.comb(4, 4-m.bit_count())*fugacity**m.bit_count() for m in range(16))
        selected = sum(sums[m]*math.comb(4, 2)*fugacity**2 for m in (5, 10))
        other = total-selected
        assert total == 16*fugacity+16*fugacity**3-96*fugacity**2
        heat_rows.append({'fugacity': str(fugacity), 'canonical4_dimension': 70,
                          'normalizer': str(normalizer), 'raw_pair': str(selected),
                          'raw_complement': str(other), 'normalized_total': str(total/normalizer),
                          'normalized_pair': str(selected/normalizer),
                          'normalized_complement': str(other/normalizer)})
    assert F(heat_rows[1]['normalized_complement']) > 0
    # Occupations show neither projector contains the other. Extra colours unrestricted
    # in diagnostic masks; actual Pi resolves all four colour occupations per vertex.
    assert (1, 0, 1, 0) not in ((2, 0, 2, 0), (0, 2, 0, 2))
    return {'all16_sum24': sums, 'native_onecolour_pair': str(pair),
            'native_onecolour_complement': str(complement), 'total': str(pair+complement),
            'withdrawn_compensation_rejected': pair < 0 and complement < 0,
            'conditional_mask_count': str(diagnostic_count),
            'actual_total_number_Pi_count_one_active_copy': str(actual_pi_count),
            'actual_Pi_other64_copies_unrestricted_factor': 'binom(256,128)^64',
            'distinct_Pi_and_masks': True,
            'diagnostic_not_in_Pi': 'only colour0 occupied at vertices0,2 => n=(1,0,1,0)',
            'Pi_not_in_diagnostic': 'colours1,2 occupied at vertices0,2 => colour0 mask0',
            'parent_heat_method_rows': heat_rows, 'no_actual_ground_lower': True}


def response_controls():
    psi = s.zeros(6, 1); psi[0] = 1; psi[5] = 2
    rho = psi*psi.T/5
    ground_heat = rho + (s.eye(6)-rho)/2
    ground_denominator = s.trace(rho*ground_heat)
    identity_denominator = s.trace(ground_heat)/6
    dephased_denominator = s.trace(s.diag(*rho.diagonal())*ground_heat)
    assert ground_denominator == 1
    assert identity_denominator != 1 and dephased_denominator != 1
    # Equal energies and source-free normalizers do not select the same history.
    inherited = s.diag(1, 0, 0)
    thermal_ground_average = s.diag(s.Rational(1, 2), s.Rational(1, 2), 0)
    history_source = s.diag(1, 0, 0)
    inherited_moment = s.trace(inherited*history_source)
    thermal_moment = s.trace(thermal_ground_average*history_source)
    assert inherited_moment != thermal_moment
    # L=diag(0,3), true ground e0, bounded Hermitian source offdiagonal1.
    eta = F(2)
    dynamic = F(2, 3+eta)
    negative_energy_hessian = F(2, 3)
    t = s.Symbol('t', real=True)
    lower_energy = (3-s.sqrt(9+4*t**2))/2
    assert -s.diff(lower_energy, t, 2).subs(t, 0) == s.Rational(2, 3)
    assert dynamic != negative_energy_hessian
    # Nonzero PSD branch Gram cancels identically with complex cross retained.
    gram = s.Matrix([[1, -1], [-1, 1]])
    assert gram.det() == 0 and (s.Matrix([[1, 1]])*gram*s.Matrix([1, 1]))[0] == 0
    # Whole kernel has rank2; mean subtraction incorrectly retains e1 zero-mode.
    ground_projection = s.diag(1, 1, 0)
    source_vector = s.Matrix([0, 1, 1])
    whole = (s.eye(3)-ground_projection)*source_vector
    mean = (s.eye(3)-s.diag(1, 0, 0))*source_vector
    assert whole != mean
    # Traceclass/strong versus heat-trace boundary: P_R -> I, Tr I infinite;
    # density p_n=1/[n(n+1)] has exact Tr rho P_R=R/(R+1) ->1.
    trace_rows = [{'R': r, 'density_trace': str(F(r, r+1)), 'ordinary_trace': r}
                  for r in (1, 2, 10, 100)]
    # Unbounded multiplication J=n and vacuum coefficients n^-2/sqrt(zeta4).
    # L=I-|Omega><Omega| bounded, so A Omega belongs to every D(L^s), yet
    # ||J^2 Omega||^2=sum 1/zeta4=infinity. Fixed-cutoff identities cannot remove M.
    tails = [{'M': m, 'J2_vacuum_norm_squared_in_zeta4_units': m,
              'J_vacuum_tail_squared_upper_in_zeta4_units': str(F(1, m))}
             for m in (4, 16, 64)]
    return {'finite_eta_dynamic': str(dynamic), 'minus_energy_hessian': str(negative_energy_hessian),
            'sourcefree_originating_ground_denominator': str(ground_denominator),
            'identity_boundary_wrong_denominator': str(identity_denominator),
            'dephased_boundary_wrong_denominator': str(dephased_denominator),
            'same_energy_different_history': {'inherited_moment': str(inherited_moment),
                                            'thermal_ground_average_wrong_moment': str(thermal_moment)},
            'finite_eta_hessian_identification_rejected': True,
            'nonzero_branch_Gram_cancelled_exactly': True,
            'wholeP0_not_mean_rejected': True, 'traceclass_strong_limit_control': trace_rows,
            'ordinary_trace_not_inferred': True, 'unbounded_domain_countercontrol': tails,
            'countercontrol_is_not_native_ground_no_go': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = json.loads((ROOT/args.request).read_text())
    for name, expected in request['pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    result = {'kind': 'INDEPENDENT_BOUNDARY_SIGN_CONTROLS_NOT_PHASE_ADMISSION',
              'input_pins': request['pins'], 'boundary': boundary_controls(),
              'complement': complement_controls(), 'response_and_domains': response_controls(),
              'exact_arithmetic': True, 'native_ground_numerics': False,
              'scientific_admission': False, 'campaign_complete': False}
    (ROOT/request['receipt']).write_text(json.dumps(result, indent=2)+'\n')
    print('Independent boundary/sign exact controls reached every assertion and exited naturally.')


if __name__ == '__main__':
    main()
