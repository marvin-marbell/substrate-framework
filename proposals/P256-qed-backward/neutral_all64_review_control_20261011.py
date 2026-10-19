"""Independent exact integer 48-mode CAR contrary controls; no target imports/runs.

Three labelled copies, two physical spins each, four Dirac modes per slot,
two cells. A role-symmetric but correlated source tests all seven classes.
All sparse arithmetic is Python integer, including full mixed-role onsite H.
The real symmetric H is deliberately generic: the free commutator identity
is polynomial in arbitrary onsite entries and does not require native phases.
Native complex-density orientation is tested separately with exact SymPy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import sympy as sy

ROOT = Path(__file__).resolve().parent
PREFIX = 'neutral_all64_review_'
COPIES, SPINS, SLOTS, CELL_MODES = 3, 2, 6, 24
M = [[2, 1, 0, -1], [0, -2, 1, 0], [1, 0, 3, 1], [0, 1, 0, -1]]
PROFILES = [
    [[-3, 0, 2, 1], [0, 3, 1, -2], [2, 1, -3, 0], [1, -2, 0, 3]],
    [[-7, 0, 5, 2], [0, 7, 2, -5], [5, 2, -7, 0], [2, -5, 0, 7]],
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean(v):
    return {mask: c for mask, c in v.items() if c}


def add(*parts):
    out = defaultdict(int)
    for scale, vector in parts:
        for mask, c in vector.items():
            out[mask] += scale*c
    return clean(out)


def inner(left, right):
    return sum(c*right.get(mask, 0) for mask, c in left.items())


def word(vector, factors, signed=True):
    """Independent global occupation-mask CAR; factors are written left-to-right."""
    for mode, creation in reversed(factors):
        out = defaultdict(int)
        for mask, c in vector.items():
            if bool(mask & (1 << mode)) != creation:
                parity = (mask & ((1 << mode)-1)).bit_count() if signed else 0
                out[mask ^ (1 << mode)] += (-1)**parity*c
        vector = clean(out)
    return vector


def make_cell():
    # Coefficients depend only on active/auxiliary occupation counts, NOT copies.
    weights = {(0, 0): 1, (2, 4): 2, (1, 2): 3, (0, 4): -1, (2, 0): 4}
    out = {}
    for labels in itertools.product((5, 3), repeat=SLOTS):
        key = (sum(x == 3 for x in labels[:2]), sum(x == 3 for x in labels[2:]))
        if key in weights:
            out[sum(x << (4*i) for i, x in enumerate(labels))] = weights[key]
    return out


def mode(copy, entry, side):
    return (2*copy+entry//2, 2*side+entry % 2)


def generate_terms(f, g=None):
    terms = []
    if g is None:
        for i, k in itertools.product(range(4), repeat=2):
            ca = sum(M[i][j]*M[k][j] for j in range(4))
            cb = sum(M[j][i]*M[j][k] for j in range(4))
            if ca:
                terms.append((ca, (*mode(f, i, 1), *mode(f, k, 1)), None))
            if cb:
                terms.append((cb, None, (*mode(f, i, 0), *mode(f, k, 0))))
        g = f
    for i, j, k, l in itertools.product(range(4), repeat=4):
        c = -2*M[i][j]*M[k][l]
        if c:
            terms.append((c, (*mode(f, i, 1), *mode(g, k, 1)),
                          (*mode(g, l, 0), *mode(f, j, 0))))
    return terms


def feature_word(feature, side=0, dagger=False):
    if feature is None:
        return []
    si, p, sk, q = feature
    a, b = side*CELL_MODES+4*si+p, side*CELL_MODES+4*sk+q
    return [(b, True), (a, False)] if dagger else [(a, True), (b, False)]


def action(terms, vector):
    result = defaultdict(int)
    for c, A, B in terms:
        for mask, value in word(vector, feature_word(A)+feature_word(B, 1)).items():
            result[mask] += c*value
    return clean(result)


def hopping(vector, dagger=False, signed=True, selected=None):
    result = defaultdict(int)
    for f, i, j in itertools.product(range(COPIES), range(4), range(4)):
        if selected is not None and f not in selected:
            continue
        coefficient = M[i][j]
        if not coefficient:
            continue
        ai = mode(f, i, 1)
        bj = mode(f, j, 0)
        a, b = 4*ai[0]+ai[1], CELL_MODES+4*bj[0]+bj[1]
        factors = [(b, True), (a, False)] if dagger else [(a, True), (b, False)]
        for mask, value in word(vector, factors, signed).items():
            result[mask] += coefficient*value
    return clean(result)


def direct_return(vector, signed=True):
    return add((1, hopping(hopping(vector, False, signed), True, signed)),
               (1, hopping(hopping(vector, True, signed), False, signed)))


def rho2(mask, side):
    return sum((1 if p < 2 else -1) for slot, p in itertools.product(range(SLOTS), range(4))
               if mask & (1 << (side*CELL_MODES+4*slot+p)))


def H(vector, sides=(0, 1), wrong_common_profile=False, electric=True):
    result = defaultdict(int)
    for side, slot, p, q in itertools.product(sides, range(SLOTS), range(4), range(4)):
        role = 0 if slot < 2 or wrong_common_profile else 1
        coefficient = PROFILES[role][p][q]
        if coefficient:
            offset = side*CELL_MODES+4*slot
            for mask, value in word(vector, [(offset+p, True), (offset+q, False)]).items():
                result[mask] += coefficient*value
    if electric:
        for mask, value in vector.items():
            # 64 rho_all^2, not sum of slotwise squares: rho2=2rho.
            result[mask] += 16*sum(rho2(mask, side)**2 for side in sides)*value
    return clean(result)


def commutator_terms(terms, sides=(0, 1), wrong_common_profile=False):
    result = []
    for c, A, B in terms:
        for side, feature in enumerate((A, B)):
            if feature is None or side not in sides:
                continue
            si, p, sk, q = feature
            hi = PROFILES[0 if si < 2 or wrong_common_profile else 1]
            hk = PROFILES[0 if sk < 2 or wrong_common_profile else 1]
            for r in range(4):
                for coefficient, replacement in (
                    (hi[r][p], (si, r, sk, q)), (-hk[q][r], (si, p, sk, r))
                ):
                    if coefficient:
                        result.append((c*coefficient, replacement if side == 0 else A,
                                       B if side == 0 else replacement))
    return result


def reduced_density(cell, selected):
    """Unnormalized rho[ket,bra], tracing ONE shared spectator string."""
    groups = defaultdict(dict)
    other = tuple(slot for slot in range(SLOTS) if slot not in selected)
    for mask, c in cell.items():
        labels = tuple((mask >> (4*slot)) & 15 for slot in range(SLOTS))
        ket = tuple(labels[slot] for slot in selected)
        spectator = tuple(labels[slot] for slot in other)
        groups[spectator][ket] = c
    result = defaultdict(int)
    for block in groups.values():
        for (ket, a), (bra, b) in itertools.product(block.items(), repeat=2):
            result[ket, bra] += a*b
    return dict(result)


def density_kernel(cell):
    selected = {(0,): (0,), (1,): (2,), (0, 0): (0, 1), (1, 1): (2, 3),
                (0, 1): (0, 2), (1, 0): (2, 0)}
    densities = {roles: reduced_density(cell, slots) for roles, slots in selected.items()}

    @lru_cache(maxsize=None)
    def moment(left, right):
        factors = []
        for feature, dagger in ((left, True), (right, False)):
            if feature is not None:
                si, p, sk, q = feature
                factors += [(sk, q, True), (si, p, False)] if dagger else [(si, p, True), (sk, q, False)]
        if not factors:
            return inner(cell, cell)
        balance = Counter()
        for slot, _, creation in factors:
            balance[slot] += 1 if creation else -1
        if any(balance.values()):
            return 0
        slots = sorted(balance)
        assert len(slots) <= 2
        parity = sum(factors[i][0] > factors[j][0] for i in range(len(factors)) for j in range(i+1, len(factors)))
        roles = tuple(0 if slot < 2 else 1 for slot in slots)
        result = 0
        for (ket, bra), density in densities[roles].items():
            element = 1
            for index, slot in enumerate(slots):
                local = [(p, creation) for owner, p, creation in factors if owner == slot]
                element *= word({ket[index]: 1}, local).get(bra[index], 0)
            result += (-1)**parity*element*density
        return result

    def gram(left, right):
        return sum(c*d*moment(A, C)*moment(B, D) for c, A, B in left for d, C, D in right)
    return densities, gram


def archival_and_caps():
    target_exit = json.loads((ROOT/'parent_neutral_return_all64_attempt3_20261011_exit.json').read_text())
    target_receipt = json.loads((ROOT/'parent_neutral_return_all64_attempt3_20261011_receipt.json').read_text())
    assert target_exit['returncode'] == 0 and target_exit['natural_exit']
    assert target_receipt['own_sha256'] == '25bd1c4824b4c5058ecd95d8bc2ff699b0304f53b015d8a9c20e35e6f75ba8f0'
    for name, pin in target_exit['input_sha256'].items():
        assert sha(ROOT/name) == pin
    audited = {}
    for suffix in ('preexecution', 'executed'):
        name = f'parent_neutral_return_all64_attempt3_20261011_{suffix}.zip'
        with zipfile.ZipFile(ROOT/name) as archive:
            members = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}
        assert all(members[n] == pin for n, pin in target_exit['input_sha256'].items())
        if suffix == 'executed':
            for artifact in ('receipt.json', 'exit.json', 'stdout.txt', 'stderr.txt', 'request.json'):
                member = 'parent_neutral_return_all64_attempt3_20261011_'+artifact
                assert members[member] == sha(ROOT/member)
        audited[name] = {'sha256': sha(ROOT/name), 'member_sha256': members}
    assert audited[target_exit['preexecution_zip']]['sha256'] == target_exit['preexecution_zip_sha256']
    assert (ROOT/'parent_neutral_return_all64_attempt3_20261011_stderr.txt').stat().st_size == 0
    caps = json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_text())
    profiles = json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles']
    c, d = [Fraction(profiles[0][key]) for key in ('c0_exact', 'd_exact')]
    assert all(Fraction(p['c0_exact']) == c and Fraction(p['d_exact']) == d for p in profiles)
    eta = Fraction(caps['normalized_exact_source_residual_upper_exact'])
    formation = Fraction(caps['coefficient_and_action_error_upper_exact'])
    infidelity = Fraction(caps['selected_ground_infidelity_upper_exact'])
    shift = Fraction(caps['Rayleigh_interval_exact'][1])-Fraction(caps['ground_energy_interval_exact'][0])
    L = 131072*(c*c+d*d)
    assert eta == Fraction(1, 65536) and formation == Fraction(1, 262144)
    assert 2*infidelity < Fraction(1, 60000**2) and infidelity > 0
    assert Fraction(target_receipt['all64_C_norm_upper_exact']) == L
    assert Fraction(target_receipt['Rayleigh_to_ground_reference_shift_upper_exact']) == shift
    assert Fraction(target_receipt['missing_source_residual_action_norm_upper_exact']) == 2*(eta+shift)*L
    assert Fraction(target_receipt['source_to_true_ground_Q_return_vector_norm_error_upper_exact']) == L/30000
    return {'audited_archives': audited, 'target_natural_exit': True,
            'frozen_target_sha256': target_receipt['own_sha256'], 'eta': str(eta),
            'formation_already_inside_eta': str(formation), 'reference_shift': str(shift),
            'I_is_squared_infidelity': str(infidelity), 'two_I_less_than_1_over_60000_squared': True,
            'operator_bound_L': str(L), 'source_action_cap': str(2*(eta+shift)*L),
            'no_shift_wrong_cap_difference': str(2*shift*L),
            'double_added_formation_wrong_cap_difference': str(2*formation*L),
            'source_to_true_ground_Q_vector_cap': str(L/30000)}


def complex_orientation_control():
    ann = []
    for p in range(4):
        A = sy.zeros(16)
        for mask in range(16):
            if mask & (1 << p):
                A[mask ^ (1 << p), mask] = (-1)**((mask & ((1 << p)-1)).bit_count())
        ann.append(A)
    z = sy.zeros(16, 1)
    z[3], z[5] = 1, sy.I
    R = z.conjugate()*z.T  # producer R_ij=<E_i,E_j>; density is R.T
    O = ann[1].T*ann[2]
    direct = (z.conjugate().T*O*z)[0]
    correct = sy.trace(O*R.T)
    wrong = sy.trace(O*R)
    assert direct == correct and wrong != correct
    # Exact phase-aligned approximation and spectral versus source-P witness.
    source = sy.Matrix([sy.Rational(3, 5), sy.Rational(4, 5)])
    ground = sy.Matrix([1, 0])
    C = sy.diag(1, 2)
    Psource, Pground = source*source.T, ground*ground.T
    source_return = (sy.eye(2)-Psource)*C*source
    spectral_return = (sy.eye(2)-Pground)*C*source
    I = 1-(source.T*Pground*source)[0]
    vector_square = ((source-ground).T*(source-ground))[0]
    assert source_return != spectral_return and vector_square > I**2
    return {'globally_signed_native4_mode_direct_moment': str(direct),
            'producer_Gram_transpose_correct': str(correct), 'missing_transpose_wrong': str(wrong),
            'source_Q_norm_squared': str((source_return.T*source_return)[0]),
            'spectral_Q_norm_squared': str((spectral_return.T*spectral_return)[0]),
            'squared_infidelity_I': str(I), 'phase_aligned_vector_distance_squared': str(vector_square),
            'using_I_as_vector_distance_rejected': True}


def density_artifact_observations():
    # Independent reading/arithmetic only; does not invoke the target CLI/kernel.
    U = np.zeros((25, 15))
    col = 0
    for i in range(5):
        for j in range(i, 5):
            U[5*i+j, col] = 1 if i == j else 1/np.sqrt(2)
            if i != j:
                U[5*j+i, col] = 1/np.sqrt(2)
            col += 1
    with np.load(ROOT/'connected_root_exterior_actual1_20261011_moments.npz', allow_pickle=False) as data:
        shape = {key: list(data[key].shape) for key in data.files}
        AA = U@data['active_pair'][:15, :15].T@U.T
        UU = U@data['auxiliary_pair'][:15, :15].T@U.T
        AU = data['mixed_pair'][:25, :25].T
    left = lambda x: np.einsum('ijkj->ik', x.reshape(5, 5, 5, 5))
    right = lambda x: np.einsum('ijil->jl', x.reshape(5, 5, 5, 5))
    swap = lambda x: x.reshape(5, 5, 5, 5).transpose(1, 0, 3, 2).reshape(25, 25)
    result = {'NPZ_shapes': shape, 'traces_FLOAT': [float(np.trace(x)) for x in (AA, UU, AU)],
              'AA_swap_defect_FLOAT': float(np.linalg.norm(AA-swap(AA))),
              'UU_swap_defect_FLOAT': float(np.linalg.norm(UU-swap(UU))),
              'AU_active_partial_defect_FLOAT': float(np.linalg.norm(left(AU)-left(AA))),
              'AU_aux_partial_defect_FLOAT': float(np.linalg.norm(right(AU)-left(UU))),
              'AA_product_wrongcase_defect_FLOAT': float(np.linalg.norm(AA-np.kron(left(AA), left(AA)))),
              'UU_product_wrongcase_defect_FLOAT': float(np.linalg.norm(UU-np.kron(left(UU), left(UU)))),
              'role_swap_wrongcase_defect_FLOAT': float(np.linalg.norm(AU-swap(AU))),
              'floating_density_not_interval': True}
    assert max(abs(x-1) for x in result['traces_FLOAT']) < 2e-9
    assert result['UU_swap_defect_FLOAT'] < 2e-9 and result['AA_swap_defect_FLOAT'] < 2e-9
    assert result['AA_product_wrongcase_defect_FLOAT'] > 1e-5
    assert result['UU_product_wrongcase_defect_FLOAT'] > 1e-5
    return result


def run():
    start = time.monotonic()
    archival = archival_and_caps()
    cell = make_cell()
    source = {a+(b << CELL_MODES): x*y for a, x in cell.items() for b, y in cell.items()}
    S = [generate_terms(f) for f in range(COPIES)]
    D = {(f, g): generate_terms(f, g) for f, g in itertools.product(range(COPIES), repeat=2) if f != g}
    complete = sum(S, [])+sum(D.values(), [])
    direct = direct_return(source)
    coherent = action(complete, source)
    assert coherent == direct
    vectors = [action(terms, source) for terms in S]
    sectors = {pair: action(terms, source) for pair, terms in D.items()}
    assert all(inner(left, right) == 0 for left in vectors for right in sectors.values())
    assert all(inner(sectors[pair], sectors[other]) == 0 for pair in D for other in D if pair != other)
    seven = [inner(vectors[0], vectors[0]), inner(vectors[1], vectors[1]),
             inner(vectors[0], vectors[1]), inner(vectors[1], vectors[2]),
             inner(sectors[0, 1], sectors[0, 1]), inner(sectors[1, 0], sectors[1, 0]),
             inner(sectors[1, 2], sectors[1, 2])]
    counts3 = [1, 2, 4, 2, 2, 2, 2]
    total = inner(direct, direct)
    assert sum(a*b for a, b in zip(seven, counts3)) == total
    assert inner(vectors[1], vectors[2]) != 0
    densities, gram = density_kernel(cell)
    representatives = [(S[0], S[0]), (S[1], S[1]), (S[0], S[1]), (S[1], S[2]),
                       (D[0, 1], D[0, 1]), (D[1, 0], D[1, 0]), (D[1, 2], D[1, 2])]
    density_seven = [gram(left, right) for left, right in representatives]
    assert density_seven == seven
    direct_D = add((1, H(direct)), (-1, direct_return(H(source))))
    comm_terms = commutator_terms(complete)
    comm_vector = action(comm_terms, source)
    assert direct_D == comm_vector
    # All ordered self pairs, including BOTH orientations for mixed A/U and U/U.
    full_first = inner(direct, direct_D)
    full_second = inner(direct_D, direct_D)
    contractions = []
    for left, right in representatives:
        DL, DR = commutator_terms(left), commutator_terms(right)
        contractions.append((gram(left, DR), gram(DL, DR)))
    reverse_AU = (gram(S[1], commutator_terms(S[0])),
                  gram(commutator_terms(S[1]), commutator_terms(S[0])))
    reverse_UU = (gram(S[2], commutator_terms(S[1])),
                  gram(commutator_terms(S[2]), commutator_terms(S[1])))
    assert reverse_UU == contractions[3]
    first = contractions[0][0]+2*contractions[1][0]+2*(contractions[2][0]+reverse_AU[0])+2*contractions[3][0]+2*sum(x[0] for x in contractions[4:])
    second = contractions[0][1]+2*contractions[1][1]+2*(contractions[2][1]+reverse_AU[1])+2*contractions[3][1]+2*sum(x[1] for x in contractions[4:])
    assert first == full_first and second == full_second
    wrong_decoherent = sum(inner(v, v) for v in vectors)+sum(inner(v, v) for v in sectors.values())
    wrong_selfonly = inner(add(*[(1, v) for v in vectors]), add(*[(1, v) for v in vectors]))
    wrong_unsigned = direct_return(source, signed=False)
    wrong_common = action(commutator_terms(complete, wrong_common_profile=True), source)
    wrong_onecell = action(commutator_terms(complete, sides=(0,)), source)
    wrong_half_cross = first-contractions[3][0]
    assert wrong_decoherent != total and wrong_selfonly != total
    assert wrong_unsigned != direct and wrong_common != direct_D and wrong_onecell != direct_D
    # rho commutes INITIAL return, but not generated free commutator.
    def rho_action(v, side):
        return clean({mask: rho2(mask, side)*c for mask, c in v.items()})
    assert all(rho_action(direct, side) == direct_return(rho_action(source, side)) for side in (0, 1))
    nested = add((1, rho_action(direct_D, 0)), (-1, action(comm_terms, rho_action(source, 0))))
    assert nested
    counts64 = [1, 63, 126, 63*62, 63, 63, 63*62]
    # Complete labelled classification, not empirical extrapolation from copies3.
    classified = Counter()
    for f, g in itertools.product(range(64), repeat=2):
        if f == g:
            classified['same_active' if f == 0 else 'same_auxiliary'] += 1
        else:
            classified['self_active_auxiliary_cross' if f == 0 or g == 0 else 'self_distinct_auxiliary_cross'] += 1
            classified['different_active_auxiliary' if f == 0 else 'different_auxiliary_active' if g == 0 else 'different_auxiliary_auxiliary'] += 1
    assert list(classified[k] for k in ('same_active', 'same_auxiliary', 'self_active_auxiliary_cross',
                                      'self_distinct_auxiliary_cross', 'different_active_auxiliary',
                                      'different_auxiliary_active', 'different_auxiliary_auxiliary')) == counts64
    result = {'kind': 'NEW_INDEPENDENT_EXACT_INTEGER_SIGNED_CAR_AND_CAP_REVIEW_NOT_TARGET_RERUN',
              'own_sha256': sha(__file__), 'elapsed_seconds': time.monotonic()-start,
              'archival_and_caps': archival, 'density_artifact_FLOAT': density_artifact_observations(),
              'complex_orientation_and_projection_control': complex_orientation_control(),
              'exact_integer_control': {'global_modes': 48, 'copies': 3, 'physical_spins_per_copy': 2,
              'four_Dirac_modes_per_slot': True, 'correlated_role_symmetric_cell_source_terms': len(cell),
              'unnormalized_cell_source_norm_squared': inner(cell, cell),
              'unnormalized_product_source_norm_squared': inner(source, source),
              'all_seven_kernel_integers': seven, 'counts_three_copies': counts3, 'counts_64_exact_label_classes': counts64,
              'global_C_norm_squared': total, 'global_C_equals_contact_exchange': True,
              'factorized_correlated_density_seven_equals_globalCAR': True,
              'self_cross_overlap_retained': seven[2:4],
              'complete_commutator_first': full_first, 'complete_commutator_second': full_second,
              'seven_commutator_first_second': contractions, 'AU_reverse_first_second': reverse_AU,
              'UU_reverse_equals_forward_first_second': reverse_UU,
              'direct_full_HA_HB_rho_squares_commutator_equals_free_generated_features': True,
              'wrong_decoherent_norm_squared': wrong_decoherent, 'wrong_drop_crosscopy_norm_squared': wrong_selfonly,
              'wrong_unsigned_C_vector_error_squared': inner(add((1, wrong_unsigned), (-1, direct)), add((1, wrong_unsigned), (-1, direct))),
              'wrong_common_role_H_commutator_error_squared': inner(add((1, wrong_common), (-1, direct_D)), add((1, wrong_common), (-1, direct_D))),
              'wrong_one_cell_H_commutator_error_squared': inner(add((1, wrong_onecell), (-1, direct_D)), add((1, wrong_onecell), (-1, direct_D))),
              'wrong_UU_halfcross_first': wrong_half_cross,
              'generated_commutator_does_not_commute_rho2_error_squared': inner(nested, nested),
              'native_complex_profiles_scope': 'Integer generic real mixed onsite controls polynomial commutator rule; actual native A/B/C phases audited in frozen source, not substituted here.',
              'exact_integer_arithmetic': True},
              'all_assertions_reached': True, 'target_imported_or_run': False,
              'ground_or_interval_or_phase_claim': False}
    output = ROOT/(PREFIX+'actual1_20261011_receipt.json')
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
