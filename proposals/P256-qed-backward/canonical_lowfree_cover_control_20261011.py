from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import cast

import sympy as s

ROOT = Path(__file__).resolve().parent
BITS = 160
Q = 1 << BITS
PROFILE = 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'
PINS = {
    PROFILE: '102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
    'neutral_return_lower_control_20261011.py': 'f2a3ff7efce4960ee38a32055e9daf6ae97e465ab23514c7bee206b60a32338f',
    'neutral_return_lower_boundary_actual2_20261011_receipt.json': '7c0ae50f0790e502161e5e925e5a707b5130a15ef3648f8c9cb40ff957ac77bb',
    'neutral_return_image_API_20261011.json': '861595a1c0432fc3e965fe35472a0386217fd57b75bba7ab868918e57e4197e1',
    'neutral_return_image_interface_20261011.json': 'a0b4d7e530f770d549d4f022e41f2134715e850b93a046449b64e3fe61dd2a43',
    'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json': '2f019dd49873a5e13f130c65ef6739b9af78a16f85138dc6c6ab6e32e65c7a12',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sqrt_interval(value):
    value = Fraction(value)
    n = math.isqrt(value.numerator * Q * Q // value.denominator)
    assert Fraction(n, Q) ** 2 <= value <= Fraction(n + 1, Q) ** 2
    return Fraction(n, Q), Fraction(n + 1, Q)


def interval_json(value):
    return [str(x) for x in value]


def combine(e, c, pa, pu, qc):
    return tuple(pa * e[0][j] + pu * e[1][j] + qc * c for j in (0, 1))


def local_types():
    # Every one of the sixteen local CAR eigenvectors occurs once, including the
    # scalar and the vector Cartan zero modes as distinct orthogonal states.
    out = [('n0', 0, -2, 2, 0), ('n4', 4, 2, 2, 0)]
    for number in (1, 3):
        for index, (p, q) in enumerate(((1, 1), (1, -1), (3, 1), (3, -1))):
            out.append((f'n{number}_s{index}', number, number - 2, p, q))
    for index, (p, q) in enumerate(((2, 2), (2, 0), (2, -2), (4, 0)), 1):
        out.append((f'n2_v{index}', 2, 0, p, q))
    out.append(('n2_scalar', 2, 0, 2, 0))
    assert len(out) == 15 and len({x[0] for x in out}) == 15
    return out


TYPES = local_types()


def signature(word):
    t = sum(count for _, kind, count in word if TYPES[kind][1] in (0, 4))
    odd = sum(count for _, kind, count in word if TYPES[kind][1] in (1, 3))
    scalar = sum(count for _, kind, count in word if TYPES[kind][0] == 'n2_scalar')
    vector = sum(count for _, kind, count in word if TYPES[kind][0].startswith('n2_v'))
    return t, odd, scalar, vector


def covered(word, mode):
    t, odd, scalar, vector = signature(word)
    modified2 = scalar + vector
    if mode == 'old_shortcut':
        return t == 0 and ((odd == 0 and modified2 <= 1) or (odd == 2 and modified2 == 0))
    if t == 0 and ((odd == 0 and modified2 <= 2) or (odd == 2 and modified2 == 0)):
        return True
    if mode == 'old_domains':
        return False
    if t == 0 and ((odd == 4 and modified2 == 0) or (odd == 2 and modified2 <= 1)):
        return True
    if mode == 'literal_new_domains':
        return False
    assert mode == 'with_scalar_transfer'
    return t == 1 and odd == 2 and modified2 == 0


def labelled_multiplicity(word):
    product = 1
    for role, slots in enumerate((2, 126)):
        counts = [count for r, _, count in word if r == role]
        ground = slots - sum(counts)
        assert ground >= 0
        product *= math.factorial(slots) // (math.factorial(ground) * math.prod(math.factorial(x) for x in counts))
    return product


def enumerate_actual(e, c, budget):
    kinds = [(role, k) for role in (0, 1) for k in range(15)]
    costs = []
    for role, k in kinds:
        p, q = TYPES[k][3:]
        costs.append(combine(e, c, p if role == 0 else 0, p if role == 1 else 0, q))
    smallest = min(x[0] for x in costs)
    assert smallest > 0
    maximum = budget[1] // smallest
    assert maximum == 5
    records = []
    cuts = {'strict_cost_lower_above_budget_upper': 0, 'charge_nonzero': 0, 'capacity': 0}
    visited = 0

    def visit(start, capacity, remaining, delta, pa, pu, qc, word):
        nonlocal visited
        visited += 1
        cost = combine(e, c, pa, pu, qc)
        if delta == 0:
            assert cost[1] < budget[0] or cost[0] > budget[1], ('unresolved radical comparison', word)
            if cost[1] < budget[0]:
                labels = [{'role': ('A', 'U')[r], 'state': TYPES[k][0], 'count': n} for r, k, n in word]
                records.append({'word': labels, 'coefficients_eA_eU_c': [pa, pu, qc], 'charge': delta,
                                'signature_emptyfull_odd_scalar_vector': list(signature(word)),
                                'cost_interval_exact': interval_json(cost),
                                'labelled_free_basis_multiplicity': str(labelled_multiplicity(word)),
                                'covered': {mode: covered(word, mode) for mode in
                                            ('old_shortcut', 'old_domains', 'literal_new_domains', 'with_scalar_transfer')}})
        else:
            cuts['charge_nonzero'] += 1
        if remaining == 0:
            return
        for j in range(start, len(kinds)):
            role, kind = kinds[j]
            if capacity[role] == 0:
                cuts['capacity'] += 1
                continue
            name, number, charge, p, q = TYPES[kind]
            newcost = combine(e, c, pa + (p if role == 0 else 0), pu + (p if role == 1 else 0), qc + q)
            if newcost[0] > budget[1]:
                cuts['strict_cost_lower_above_budget_upper'] += 1
                continue
            assert newcost[1] < budget[0], ('unresolved partial cost', name)
            capacities = list(capacity)
            capacities[role] -= 1
            if word and word[-1][:2] == (role, kind):
                nextword = word[:-1] + [(role, kind, word[-1][2] + 1)]
            else:
                nextword = word + [(role, kind, 1)]
            visit(j, tuple(capacities), remaining - 1, delta + charge,
                  pa + (p if role == 0 else 0), pu + (p if role == 1 else 0), qc + q, nextword)

    visit(0, (2, 126), int(maximum), 0, 0, 0, 0, [])
    words = [json.dumps(x['word'], sort_keys=True) for x in records]
    assert len(words) == len(set(words))
    assert all(x['covered']['with_scalar_transfer'] for x in records)
    summary = {}
    for record in records:
        key = ','.join(map(str, record['signature_emptyfull_odd_scalar_vector']))
        entry = summary.setdefault(key, {'histograms': 0, 'labelled_dimension': 0})
        entry['histograms'] += 1
        entry['labelled_dimension'] += int(record['labelled_free_basis_multiplicity'])
    return {'method': 'All nondecreasing multisets of all15 local excited CAR states, both exact roles, bounded by five strictly positive local costs; capacity and charge filters. No optimizer and no small-n extrapolation.',
            'excited_slots_maximum_before_neutrality': int(maximum), 'visited_partial_multisets': visited,
            'proved_cuts': cuts, 'neutral_negative_histograms': len(records),
            'negative_free_dimension_including_all_labelled_nonsymmetric_vectors': str(sum(int(x['labelled_free_basis_multiplicity']) for x in records)),
            'summary': summary, 'records': records}


def exterior(matrix, number):
    masks = [sum(1 << p for p in word) for word in itertools.combinations(range(4), number)]
    output = s.zeros(len(masks))
    for j, mask in enumerate(masks):
        for q in range(4):
            if not mask >> q & 1:
                continue
            sign = (-1) ** ((mask & ((1 << q) - 1)).bit_count())
            tail = mask ^ (1 << q)
            for p in range(4):
                if not tail >> p & 1:
                    target = tail | (1 << p)
                    sign2 = (-1) ** ((tail & ((1 << p) - 1)).bit_count())
                    output[masks.index(target), j] += sign * sign2 * matrix[p, q]
    return output, masks


def joint_transfer_controls():
    X = s.Matrix([[0, 1], [1, 0]])
    A = s.diag(-1, 1, -1, 1)
    B = s.kronecker_product(X, s.diag(1, -1))
    C = s.zeros(4)
    C[:2, 2:] = s.I * X
    C[2:, :2] = -s.I * X
    rho = s.diag(s.Rational(1, 2), s.Rational(1, 2), -s.Rational(1, 2), -s.Rational(1, 2))
    matrices = {'A': A, 'B': B, 'C': C, 'rho': rho}
    _, masks3 = exterior(A, 3)
    _, masks2 = exterior(A, 2)
    J = s.zeros(4)
    for p in range(4):
        J[masks3.index(15 ^ (1 << p)), p] = (-1) ** p
    K = s.diag(1, -1, 1, -1) * s.eye(4)[::-1, :]
    T3 = J * K
    assert T3.H * T3 == s.eye(4)
    scalar = s.zeros(6, 1)
    scalar[masks2.index(9)] = 1 / s.sqrt(2)
    scalar[masks2.index(6)] = -1 / s.sqrt(2)
    controls = {}
    for name, matrix in matrices.items():
        one, _ = exterior(matrix, 1)
        three, _ = exterior(matrix, 3)
        two, _ = exterior(matrix, 2)
        empty, _ = exterior(matrix, 0)
        full, _ = exterior(matrix, 4)
        assert s.trace(matrix) == 0 and empty == s.zeros(1) and full == s.zeros(1)
        assert two * scalar == s.zeros(6, 1)
        assert s.simplify(T3.H * three * T3 - one) == s.zeros(4)
        controls[name] = {'trace_zero': True, 'n0_n4_exact_zero': True, 'n2_scalar_exact_zero': True,
                          'number3_joint_same_number1': True}
    # Compare complete three-slot H, with unequal onsite profiles and one total
    # density squared. These are 16x16 exact rational forms, not fit spectra.
    ha = 3 * A - B + 4 * C
    hu = 12 * A - B + 5 * C
    D1, _ = exterior(ha, 1)
    D3, _ = exterior(hu, 3)
    R1, _ = exterior(rho, 1)
    R3, _ = exterior(rho, 3)
    def product_h(leftD, rightD, leftR, rightR):
        R = s.kronecker_product(leftR, s.eye(4)) + s.kronecker_product(s.eye(4), rightR)
        return s.kronecker_product(leftD, s.eye(4)) + s.kronecker_product(s.eye(4), rightD) + 64 * R * R
    standard = product_h(ha, hu, rho, rho)
    balanced = product_h(D1, D3, R1, R3)
    D3a, _ = exterior(ha, 3)
    empty33 = product_h(D3a, D3, R3, R3)
    full11 = standard
    balanced_map = cast(s.MatrixBase, s.kronecker_product(s.eye(4), T3))
    empty_map = cast(s.MatrixBase, s.kronecker_product(T3, T3))
    assert s.simplify(balanced_map.H * balanced * balanced_map - standard) == s.zeros(16)
    assert s.simplify(empty_map.H * empty33 * empty_map - standard) == s.zeros(16)
    assert full11 == standard
    wrong = cast(s.MatrixBase, s.kronecker_product(J, J))
    assert s.simplify(wrong.H * empty33 * wrong - standard) != s.zeros(16)
    assert s.kronecker_product(rho, rho) != s.zeros(16)
    assert standard != s.kronecker_product(ha + 64 * rho * rho, s.eye(4)) + s.kronecker_product(s.eye(4), hu + 64 * rho * rho)
    return {'basis_order_number3_masks': masks3, 'basis_order_number2_masks': masks2,
            'Jhole': [[str(x) for x in J.row(j)] for j in range(4)],
            'K': [[str(x) for x in K.row(j)] for j in range(4)],
            'T3_Jhole_K': [[str(x) for x in T3.row(j)] for j in range(4)],
            'scalar6': [str(x) for x in scalar], 'native_generator_controls': controls,
            'all16_exact_unequal_profile_three_slot_H_entries_match': True,
            'total_rho_squared_once_including_cross_terms': True,
            'uncorrected_hole_map_rejected': True, 'dropping_cross_density_rejected': True,
            'transfer_total_numbers': {'balanced_n2_n1_n3': 6, 'empty_n0_n3_n3': 6, 'full_n4_n1_n1': 6},
            'spectator_number': 'Every unchanged vector5 slot has number2; all three patterns give 6+2*125=256.',
            'role_rule': 'The trivial slot retains its role and label, both spinors retain their respective roles and labels, and both symmetric spectator counts are identical. No energy trace or density affine offset is added.'}


def adverse_controls(e, c, budget, baseline, enumeration):
    byname = {x[0]: i for i, x in enumerate(TYPES)}
    def word(items):
        return sorted((role, byname[name], count) for role, name, count in items)
    cases = {
        'two_vector_Cartan_AA': word([(0, 'n2_v1', 2)]),
        'four_balanced_odd_AAUU': word([(0, 'n1_s0', 1), (0, 'n3_s0', 1), (1, 'n1_s0', 1), (1, 'n3_s0', 1)]),
        'scalar_A_plus_UU_odd_pair': word([(0, 'n2_scalar', 1), (1, 'n1_s0', 1), (1, 'n3_s0', 1)]),
        'empty_A_plus_AU_number3_pair': word([(0, 'n0', 1), (0, 'n3_s0', 1), (1, 'n3_s0', 1)]),
        'full_A_plus_AU_number1_pair': word([(0, 'n4', 1), (0, 'n1_s0', 1), (1, 'n1_s0', 1)]),
        'positive_one_vector_U': word([(1, 'n2_v1', 1)]),
        'positive_balanced_odd_AA': word([(0, 'n1_s0', 1), (0, 'n3_s0', 1)]),
    }
    out = {}
    for name, w in cases.items():
        pa = sum(TYPES[k][3] * n for r, k, n in w if r == 0)
        pu = sum(TYPES[k][3] * n for r, k, n in w if r == 1)
        qc = sum(TYPES[k][4] * n for _, k, n in w)
        charge = sum(TYPES[k][2] * n for _, k, n in w)
        cost = combine(e, c, pa, pu, qc)
        assert charge == 0 and cost[1] < budget[0]
        assert covered(w, 'with_scalar_transfer')
        out[name] = {'word': [{'role': ('A', 'U')[r], 'state': TYPES[k][0], 'count': n} for r, k, n in w],
                     'charge_exact': charge, 'cost_interval_exact': interval_json(cost),
                     'D_minus_z_interval_exact': interval_json((cost[0] - budget[1], cost[1] - budget[0])),
                     'coverage': {m: covered(w, m) for m in ('old_shortcut', 'old_domains', 'literal_new_domains', 'with_scalar_transfer')}}
    assert not out['two_vector_Cartan_AA']['coverage']['old_shortcut']
    assert not out['four_balanced_odd_AAUU']['coverage']['old_domains']
    assert not out['scalar_A_plus_UU_odd_pair']['coverage']['old_domains']
    assert not out['empty_A_plus_AU_number3_pair']['coverage']['literal_new_domains']
    assert not out['full_A_plus_AU_number1_pair']['coverage']['literal_new_domains']
    for name in ('positive_one_vector_U', 'positive_balanced_odd_AA'):
        assert out[name]['coverage']['old_shortcut']
    excluded_cost = combine(e, c, 4, 0, 0)
    assert excluded_cost[0] > budget[1]
    out['positive_free_exclusion_empty_full_AA_and_single_v4_A'] = {
        'cost_interval_exact': interval_json(excluded_cost),
        'D_minus_z_interval_exact': interval_json((excluded_cost[0] - budget[1], excluded_cost[1] - budget[0])),
        'negative_classifier_rejects': True, 'empty_full_AA_is_neutral': True,
        'single_v4_A_is_neutral': True}
    # Concrete nonsymmetric spectator control in the actual128 labelled space.
    # v1 at auxiliary label2 and v2 at label3, minus the exchanged configuration.
    nonsym_cost = combine(e, c, 0, 4, 2)
    assert nonsym_cost[1] < budget[0]
    swap = s.Matrix([[0, 1], [1, 0]])
    antisym = s.Matrix([1, -1])
    sym = s.Matrix([1, 1])
    assert swap * antisym == -antisym and (sym.H * antisym)[0] == 0
    assert (sym.H * sym)[0] == 2 and (antisym.H * antisym)[0] == 2
    out['nonsymmetric_UU_two_different_vector_excitations'] = {
        'canonical_vector': '|v1_2 v2_3 g_else>-|v2_2 v1_3 g_else>',
        'cost_interval_exact': interval_json(nonsym_cost), 'norm_squared_exact': '2',
        'aux_swap_eigenvalue': '-1', 'orthogonal_to_role_Sym126': True,
        'literal_harmonic_source_only_coverage_rejected': True,
        'each_term_in_same_labelled_two_full6_UU_image': True,
        'positive_symmetric_control_norm_squared': '2'}
    out['neutral_filter_and_capacity_positive_controls'] = {
        'one_low_odd_is_free_negative_but_not_totalnumber256': True,
        'three_active_low_odds_have_insufficient_actual_active_slots': True,
        'enumerator_uses_actual_active2_aux126': True}
    assert enumeration['proved_cuts']['charge_nonzero'] > 0
    assert enumeration['proved_cuts']['capacity'] > 0
    return out


def theorem(e, c, baseline, budget):
    delta = (e[0][0] + c, e[0][1] + c)
    assert e[1][0] > e[0][1] and c < 0 and delta[0] > 0
    assert e[0][0] + 3 * c > 0
    sixdelta_minus_four_eA = (2 * e[0][0] + 6 * c, 2 * e[0][1] + 6 * c)
    complement = (-252 * e[1][1], -252 * e[1][0])
    margin = (complement[0] + 5123, complement[1] + 5123)
    assert margin[0] > 0
    return {
        'scope': 'Exact fullcanonical neutral local free-D coverage theorem; H0 ground admission is CONDITIONAL on parent integration of every domain bound.',
        'native_space': '(exterior CAR4)^tensor128 at totalnumber256; role multiplicities2,126; D=sum dGamma(h_role), H0=D+64(sum rho_slot)^2.',
        'full_local_spectra': {'number0_and4': ['0'], 'number1_and3': ['-e+c', '-e-c', 'e+c', 'e-c'],
                               'number2_vector5': ['-2e', '2c', '0', '-2c', '2e'], 'number2_scalar1': ['0']},
        'all_local_excitation_costs': {'number0_and4_and_number2_scalar': '2e',
                                      'number1_and3': ['e+c', 'e-c', '3e+c', '3e-c'],
                                      'number2_excited_vector': ['2(e+c)', '2e', '2(e-c)', '4e']},
        'free_baseline_interval_exact': interval_json(baseline),
        'negative_free_budget_interval_exact': interval_json(budget),
        'least_local_cost_interval_exact': interval_json(delta),
        'six_minimum_costs_minus_four_eA_exact_interval': interval_json(sixdelta_minus_four_eA),
        'negative_classes_exhaustive': [
            'No odd or empty/full slots: zero, one or two excited number2 scalar/vector slots.',
            'Exactly two odd slots, number1+number3: zero or one excited number2 scalar/vector slot.',
            'Exactly four odd slots: two number1 and two number3, all other slots at vector free minimum.',
            'Exactly one empty plus two number3, or one full plus two number1, all other slots at vector free minimum.'
        ],
        'proof_exhaustiveness': [
            'Each of the128 actual slots independently has one of16 joint free eigenvectors; no source/harmonic assumption enters. Every nonground slot costs at least delta=eA+c>0. Six such costs exceed4eA, while the budget is strictly below4eA.',
            'Let t=count(n0/n4), o=count(odd), j=count(excited n2 scalar/vector). Neutrality is2(n4-n0)+(n3-n1)=0, so o is even. The excitation cost is at least2eA*t+delta*o+2delta*j.',
            't>=2 costs>=4eA. If t=1 neutrality requires o>=2; o>=4 or j>=1 costs>=2eA+4delta>4eA, using eA+2c>0. Thus t=1 forces o=2,j=0 and both odds compensate its charge.',
            'For t=0, o+2j>=6 costs>=6delta>4eA, using eA+3c>0. The remaining even o cases are(o=0,j<=2),(o=2,j<=1),(o=4,j=0). These include all physical labels and all local scalar/vector/spinor indices; the exact multiset enumeration decides which of their energies are actually below-5123.',
            'Role inequalities, delta>0, eA+3c>0 and4eA>budget are proved by160-bit rational outward bounds of the exact pinned radicals. Unresolved comparisons fail loudly; there are none in the naturally-exited enumeration.'
        ],
        'invariant_domain_images': [
            'For every actual ordered labelled choice of one/two distinguished slots, embed full wedge2=scalar1+vector5 on them and Sym^(remaining active)(5) tensor Sym^(remaining auxiliary)(5) on spectators. These are invariant under D and every rho generator, hence H0. Include the unmodified selected Sym2(5) tensor Sym126(5) domain.',
            'For every labelled neutral number1/3 pair and optional distinguished full6 slot, embed spinor4 tensor spinor4 [tensor full6] and the same role-symmetric vector spectator spaces; these are invariant. For four odd slots use all labelled two-number1/two-number3 patterns and fourspinor4 with the actual spectator counts.',
            'For each triple additionally embed the SCALAR RESTRICTION of the mixed6+spinor+spinor domain into n0+3+3 and n4+1+1 via explicit joint T3=Jhole*K. These are additional physical embeddings, not an assertion that the old wedge2 embedding already contained n0/n4. The full6 vector summand cannot be mapped to a one-dimensional empty/full slot.',
            'Every classified free basis vector belongs to one such image: put every nonground n2 slot among distinguished slots, every odd slot among spinor slots, the empty/full slot in the transferred scalar, and every untouched slot in its vector free-minimum state. The latter spectator product is symmetric within each role. Linear span therefore contains the ENTIRE negative free spectral subspace, including nonsymmetric combinations of label choices; it does not demand the vector itself be symmetric.',
            'This is an all128-slot spanning argument. No fixed-l, highest-harmonic projection or independent spectator state replaces Sym^n(5); its decomposition includes every trace harmonic. The finite antisymmetric label control proves why role-Sym source alone is insufficient.'
        ],
        'orthogonal_complement': {
            'union': 'U=sum of the labelled invariant domain images INCLUDING scalar transfers.',
            'D_lower_bound_exact_formula': 'D|Uperp >= baseline+4eA = -252eU',
            'D_lower_bound_exact_interval': interval_json(complement),
            'D_minus_minus5123_lower_interval': interval_json(margin),
            'H0_lower_bound_same': True,
            'proof': 'Every free eigenvector of cost<4eA belongs to U by the exhaustive class argument. Each image reduces the selfadjoint D and rho, so U and Uperp reduce D,rho,H0. The D spectral subspace below baseline+4eA is contained in U; hence D on Uperp has the displayed lower bound. H0-D=64rho_all^2 is positive. Equality for D is witnessed by n0 at active slot0 and n4 at active slot1 with all126 auxiliary vector-minimum spectators; its conserved number pattern is orthogonal to every stated image.',
            'sharp_free_complement_witness': 'n0_A0 tensor n4_A1 tensor g_U^tensor126, totalnumber0+4+252=256.'
        },
        'coherent_overlap_and_same_ground': [
            'All images land in ONE fixed ordered labelled canonical Fock space. Symmetric extraction is an isometry b_i/sqrt(n) (and ordered twice b_j b_i/sqrt(n(n-1))); the selected exact g has the SAME canonical embedding in every overlapping pure-vector image. There is never one ground copy per representation or per label.',
            'Pure-vector one/two-full6 images overlap; do not add their norms or inertia counts. Distinct scalar-label, odd-number-label and empty/full-label patterns are orthogonal, but corresponding histories with identical labels remain coherent.',
            'Conditional integration theorem: if each original one/two-full6 image has below-z spectral space exactly SAME span(g), and every odd/mixed/fourodd image is strictly above-z, then the union U has below-z spectral space span(g). For a selfadjoint H each invariant image reduces every spectral projector; applying P_(H<z) to their algebraic sum yields the sum of their projected images, not a sum of quadratic inequalities. Scalar restrictions and their unitary transfers inherit the mixed positivity.',
            'Together with H0|Uperp>z this would identify unique selected g as the fullcanonical local ground. This artifact does NOT assert those new domain inequalities; their natural-exit certificates and parent reconciliation are REQUIRED. FullPhi ground, lattice stationarity, linked phase, stiffness and photon pole remain OPEN.'
        ],
        'literal_coverage_failure_without_transfer': 'Smallest newly unpaid physical number class has three changed slots: n0+3+3 or n4+1+1. Fourspinor and balanced mixed wedge2 embeddings preserve different local CAR numbers and cannot contain it. Scalar-restricted representation reuse repairs coverage only when parent explicitly integrates the additional embeddings and applicable mixed bounds.',
        'nonclaims': ['fullcanonical H0 ground admitted here', 'fullPhi ground', 'lattice ground', 'linked phase', 'stiffness', 'sharp photon', 'common cone', 'originalSOURCE adoption'],
        'parent_checks': ['Selected Pyright/LSP diagnostics on new source and epoch files after sibling integration.', 'Review every actual mixed and four-odd certificate; integrate ordered scalar transfers before any fullcanonical claim.'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    assert args.tag.startswith('canonical_lowfree_cover_')
    actual_pins = {name: digest(ROOT / name) for name in PINS}
    assert actual_pins == PINS, ('immutable input mismatch', actual_pins)
    profiles = json.loads((ROOT / PROFILE).read_text())
    assert profiles['corrected_source_unitarity_H_density_identities']
    assert [x['spin_replicas'] for x in profiles['profiles']] == [2, 126]
    e = []
    for p in profiles['profiles']:
        m, d = Fraction(p['m_exact']), Fraction(p['d_exact'])
        assert Fraction(p['e_squared_exact']) == m * m + d * d
        e.append(sqrt_interval(p['e_squared_exact']))
    c = Fraction(profiles['profiles'][0]['c0_exact'])
    assert c == Fraction(profiles['profiles'][1]['c0_exact'])
    baseline = (-4 * e[0][1] - 252 * e[1][1], -4 * e[0][0] - 252 * e[1][0])
    budget = (Fraction(-5123) - baseline[1], Fraction(-5123) - baseline[0])
    proof = theorem(e, c, baseline, budget)
    enumeration = enumerate_actual(e, c, budget)
    controls = adverse_controls(e, c, budget, baseline, enumeration)
    transfer = joint_transfer_controls()
    receipt = {'kind': 'EXACT_FULLCANONICAL_FREE_CLASSIFICATION_AND_INVARIANT_COVER_WITH_EXPLICIT_SCALAR_TRANSFER_NOT_H_GROUND_ADMISSION',
               'source_sha256': digest(Path(__file__)), 'input_sha256': actual_pins,
               'radical_enclosure_bits': BITS, 'z_exact': '-5123', 'proof': proof,
               'enumeration': enumeration, 'controls': controls, 'joint_scalar_transfer': transfer,
               'inputs_unchanged': all(digest(ROOT / f) == h for f, h in PINS.items())}
    output = ROOT / (args.tag + '_receipt.json')
    with output.open('x') as file:
        json.dump(receipt, file, indent=2)
        file.write('\n')
    print(json.dumps({'receipt': output.name, 'neutral_negative_histograms': enumeration['neutral_negative_histograms'],
                      'negative_free_dimension': enumeration['negative_free_dimension_including_all_labelled_nonsymmetric_vectors'],
                      'all_negative_free_basis_states_covered_with_explicit_scalar_transfer': True,
                      'fullcanonical_H_ground_admitted': False}), flush=True)


if __name__ == '__main__':
    main()
