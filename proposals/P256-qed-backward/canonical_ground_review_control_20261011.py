from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path
from typing import cast

import numpy as np
import sympy as s

ROOT = Path(__file__).resolve().parent
PREFIX = 'canonical_ground_review_20261011'
INPUTS = [
    'neutral_return_lower_boundary_actual2_20261011_receipt.json',
    'neutral_return_lower_control_20261011.py',
    'canonical_four_odd_lower_20261011.py',
    'canonical_four_odd_actual2_20261011_receipt.json',
    'canonical_odd_vector_control_20261011.py',
    'canonical_odd_vector_actual2_20261011_receipt.json',
    'canonical_odd_vector_proof_20261011.json',
    'canonical_odd_vector_scalar_transfer_20261011.py',
    'canonical_odd_vector_scalar_actual1_20261011_receipt.json',
    'canonical_lowfree_cover_control_20261011.py',
    'canonical_lowfree_cover_actual1_20261011_receipt.json',
    'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json',
    'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json',
]


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text())


def inertia(matrix):
    """Independent rational congruence elimination, not target Bareiss."""
    a = [[F(x) for x in row] for row in matrix]
    signs = []
    while a:
        pivot = a[0][0]
        assert pivot != 0
        signs.append(1 if pivot > 0 else -1)
        a = [[a[i][j] - a[i][0] * a[0][j] / pivot
              for j in range(1, len(a))] for i in range(1, len(a))]
    return {'negative': signs.count(-1), 'positive': signs.count(1), 'pivot_signs': signs}


def matrix_check(block, bindings):
    for kind in ('matrix', 'witness'):
        name = block[kind + '_path']
        assert digest(name) == block[kind + '_sha256']
        bindings[name] = digest(name)
    cert = load(block['matrix_path'])
    assert cert['bits'] == 100
    n = len(cert['center_integer'])
    radius, residual = cert['radius_integer'], cert['residual_squared_upper_integer']
    assert radius >= 0 and residual >= 0
    for i in range(n):
        for j in range(n):
            assert cert['center_integer'][i][j] == cert['center_integer'][j][i]
            assert cert['lower_integer'][i][j] == cert['center_integer'][i][j] - (radius if i == j else 0)
            assert cert['upper_integer'][i][j] == cert['center_integer'][i][j] + (radius + 64*residual if i == j else 0)
    lo, hi = inertia(cert['lower_integer']), inertia(cert['upper_integer'])
    assert lo == cert['inertia_lower'] == block['lower_inertia']
    assert hi == cert['inertia_upper'] == block['upper_inertia']
    assert F(block['radius_upper_exact']) == F(radius, 1 << 100)
    assert F(block['residual_squared_upper_exact']) == F(residual, 1 << 100)
    return {'matrix': block['matrix_path'], 'matrix_sha256': bindings[block['matrix_path']],
            'lower': lo, 'upper': hi, 'bits': 100}


def radical(value):
    value = F(value)
    q = 1 << 192
    k = math.isqrt(value.numerator*q*q // value.denominator)
    assert F(k*k, q*q) <= value < F((k+1)*(k+1), q*q)
    return F(k, q), F(k+1, q)


def enumeration(profile, cover):
    e = [radical(p['e_squared_exact']) for p in profile['profiles']]
    c = F(profile['profiles'][0]['c0_exact'])
    assert c == F(profile['profiles'][1]['c0_exact']) < 0
    assert e[0][1] < e[1][0]
    baseline = (-4*e[0][1]-252*e[1][1], -4*e[0][0]-252*e[1][0])
    budget = (-5123-baseline[1], -5123-baseline[0])
    delta = e[0][0]+c
    assert delta > 0 and 6*delta > 4*e[0][1] > budget[1]
    assert e[0][0]+3*c > 0 and e[0][0]+2*c > 0
    # Derive all 16 local eigenstates as exterior sums of four one-particle
    # eigenvalues; no hand-entered list of 15 excitation classes is reused.
    one = [(-1,1), (-1,-1), (1,1), (1,-1)]
    local = []
    for number in range(5):
        for occupied in itertools.combinations(range(4), number):
            pe = 2+sum(one[j][0] for j in occupied)
            pc = sum(one[j][1] for j in occupied)
            if (number, occupied) != (2, (0,1)):
                local.append((number, occupied, pe, pc))
    assert len(local) == 15
    kinds = [(role, state) for role in (0,1) for state in local]
    cache = {}
    def cost(a,u,b):
        key = a,u,b
        if key not in cache:
            cache[key] = (a*e[0][0]+u*e[1][0]+b*c, a*e[0][1]+u*e[1][1]+b*c)
        return cache[key]
    histograms = collections.Counter()
    dimension = 0
    witness = None
    tested = 0
    # Unpruned combinations enumeration differs from target's cost-pruned tree.
    for length in range(6):
        for word in itertools.combinations_with_replacement(range(30), length):
            tested += 1
            roles = [kinds[j][0] for j in word]
            if roles.count(0) > 2:
                continue
            numbers = [kinds[j][1][0] for j in word]
            if sum(n-2 for n in numbers):
                continue
            a = sum(kinds[j][1][2] for j in word if kinds[j][0] == 0)
            u = sum(kinds[j][1][2] for j in word if kinds[j][0] == 1)
            b = sum(kinds[j][1][3] for j in word)
            lo, hi = cost(a,u,b)
            assert hi < budget[0] or lo > budget[1], ('unresolved', word)
            if lo > budget[1]:
                continue
            t = sum(n in (0,4) for n in numbers)
            o = sum(n in (1,3) for n in numbers)
            j = sum(n == 2 for n in numbers)
            assert (t == 0 and ((o == 0 and j <= 2) or (o == 2 and j <= 1) or (o == 4 and j == 0))) or (t == 1 and o == 2 and j == 0)
            histograms[t,o,j] += 1
            multiplicity = 1
            counts = collections.Counter(word)
            for role, slots in enumerate((2,126)):
                used = roles.count(role)
                multiplicity *= math.factorial(slots)//math.factorial(slots-used)
                for k,n in counts.items():
                    if kinds[k][0] == role:
                        multiplicity //= math.factorial(n)
            dimension += multiplicity
            if t == 1 and witness is None:
                witness = {'slots': [{'role': ['A','U'][kinds[k][0]], 'number': kinds[k][1][0], 'occupied_free_modes': kinds[k][1][1]} for k in word],
                           'free_H_minus_z_upper_exact': str(baseline[1]+hi+5123),
                           'absent_without_additional_scalar_transfers': True}
    count = sum(histograms.values())
    assert count == cover['enumeration']['neutral_negative_histograms'] == 111
    assert dimension == int(cover['enumeration']['negative_free_dimension_including_all_labelled_nonsymmetric_vectors']) == 332468225
    assert witness is not None and F(witness['free_H_minus_z_upper_exact']) < 0
    complement = -252*e[1][1]+5123
    assert complement > 0
    return {'histograms': count, 'labelled_dimension': str(dimension), 'unpruned_words_tested': tested,
            'histogram_classes': {str(k):v for k,v in histograms.items()},
            'missing_transfer_adverse_witness': witness, 'complement_H_minus_z_lower_exact': str(complement),
            'inequalities_192bit': {'delta_positive': str(delta), 'six_delta_minus_four_eA_lower': str(6*delta-4*e[0][1]),
                                   'four_eA_minus_budget_lower': str(4*e[0][0]-budget[1]), 'eA_plus_three_c_lower': str(e[0][0]+3*c)}}


def adverse_joint_and_overlap():
    # Exact native CAR operator construction with deliberately independent
    # generator probes. The wrong hole map passes D but reverses rho.
    rho = s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))
    D = s.diag(-6,-4,4,6)
    R = s.Matrix([[0,s.Rational(3,10),0,-s.Rational(2,5)],
                  [s.Rational(3,10),0,s.Rational(2,5),0],
                  [0,s.Rational(2,5),0,s.Rational(3,10)],
                  [-s.Rational(2,5),0,s.Rational(3,10),0]])
    S = s.diag(1,-1,1,-1)
    assert S*D*S == D and S*R*S == -R != R
    # One wrong signed spinor in a coherent total density changes cross terms,
    # although its isolated density square and its free spectrum remain equal.
    total = s.kronecker_product(R,s.eye(4))+s.kronecker_product(s.eye(4),R)
    wrong = s.kronecker_product(-R,s.eye(4))+s.kronecker_product(s.eye(4),R)
    difference = cast(s.MatrixBase, 64*(wrong*wrong-total*total))
    assert difference != s.zeros(16)
    i,j = next((i,j) for i in range(16) for j in range(16) if difference[i,j] != 0)
    assert R*R == (-R)*(-R)
    # Two different invariant images overlap on exactly the same physical g.
    H = s.diag(-2,1,3)
    E1 = s.Matrix([[1,0],[0,1],[0,0]])
    E2 = s.Matrix([[1,0],[0,0],[0,1]])
    ground = s.Matrix([1,0])
    assert E1*ground == E2*ground
    merged = E1.row_join(E2)
    assert merged.rank() == 3 and s.Matrix.hstack(E1*ground,E2*ground).rank() == 1
    null = s.Matrix([1,0,-1,0])
    assert merged*null == s.zeros(3,1)
    assert (E1.T*H*E1)[0,0] == (E2.T*H*E2)[0,0] == -2
    # Nonsymmetric low-free labels are missed by role-Sym alone.
    psi = s.Matrix([0,1,-1,0])
    swap = s.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    assert swap*psi == -psi
    return {'incorrect_phase_rejected': {'energy_pair_same': True, 'isolated_density_square_same': True,
                                        'coherent_H_difference_entry': [i,j,str(difference[i,j])]},
            'duplicate_ground_rejected': {'formal_domain_ground_copies': 2, 'physical_ground_rank': 1,
                                           'physical_union_rank': 3, 'redundant_coordinate_kernel': [null[i,0] for i in range(null.rows)]},
            'role_Sym_only_rejected': {'antisymmetric_label_vector': [psi[i,0] for i in range(psi.rows)], 'swap_eigenvalue': -1}}


def main():
    manifest = load(PREFIX+'_preexecution.json')
    bindings = manifest['input_sha256']
    assert digest(Path(__file__).name) == manifest['source_sha256']
    assert all(digest(n) == h for n,h in bindings.items())
    boundary = load(INPUTS[0])
    four = load('canonical_four_odd_actual2_20261011_receipt.json')
    mixed = load('canonical_odd_vector_actual2_20261011_receipt.json')
    families = [(f['kind'],load(f['receipt'])) for f in boundary['families']]
    families += [('four_odd',f) for f in four['families']]
    families += [('mixed',f) for f in mixed['role_assignments']]
    summaries = []
    final_scalar = []
    for kind, family in families:
        assert F(family['floor_exact']) == -5123
        assert family['support_budget'] > family['output_budget']+82-1
        assert sum(family['spectator_counts']) + len(family.get('roles', family.get('roles_vector_number1_number3'))) == 128
        checks = [matrix_check(b, bindings) for b in family['parity_blocks']]
        low = sum(c['lower']['negative'] for c in checks)
        high = sum(c['upper']['negative'] for c in checks)
        assert low == high == (1 if kind == 'vector' else 0)
        summaries.append({'kind':kind,'roles':family.get('roles',family.get('roles_vector_number1_number3')),
                          'output_budget':family['output_budget'],'support_budget':family['support_budget'],
                          'negative_lower':low,'negative_upper':high,'blocks':checks})
        if kind == 'mixed':
            assert (family['output_budget'],family['support_budget']) == (240,322)
            blocks = []
            for block in family['parity_blocks']:
                cert = load(block['matrix_path'])
                with np.load(ROOT/block['witness_path']) as w:
                    positions = [i for i,j in enumerate(w['negative']) if w['states'][j,10] == 5]
                il = inertia([[cert['lower_integer'][i][j] for j in positions] for i in positions])
                iu = inertia([[cert['upper_integer'][i][j] for j in positions] for i in positions])
                assert il['negative'] == iu['negative'] == 0
                blocks.append({'parity':block['parity'],'positions':positions,'lower':il,'upper':iu})
            final_scalar.append({'roles':family['roles_vector_number1_number3'],'blocks':blocks,'transferred_numbers':[[0,3,3],[4,1,1]]})
    assert len(final_scalar) == 7
    cover = load('canonical_lowfree_cover_actual1_20261011_receipt.json')
    free = enumeration(load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'), cover)
    cap = load('local_so5_ground_upper_complete_state_consumer_20261011_receipt.json')
    gap = -5123-F(cap['ground_energy_interval_exact'][1])
    assert gap == F('46955312929/34359738368') == F(cap['selected_ground_gap_lower_exact'])
    controls = adverse_joint_and_overlap()
    # SymPy values are serialized exactly, not approximated.
    result = {'kind':'INDEPENDENT_CANONICAL_COMPOSITION_CONTRARY_CONTROL',
              'verdict':'CONDITIONAL', 'conditional_only_on':['Accepted selected-ground analytic caps under standard binary64 correct rounding',
                  'Reviewed exact native representation intertwiners and outward interval Schur proof, not a new recomputation of all Schur residuals'],
              'mathematical_composition':'VALID under these explicit hypotheses',
              'scope':'ENTIRE frozen local canonical N256 CAR4^tensor128, roles2/126, H0=D+64rho_all^2; not Phi/lattice',
              'source_sha256':manifest['source_sha256'],'input_sha256':bindings,
              'families':summaries,'final240_scalar_restrictions':final_scalar,'free_classification':free,
              'new_adverse_controls':controls,'gap_lower_exact':str(gap),
              'historical_mixed120_prose_not_used':True,'old_scalar_actual1_not_used_for_final_mixed_bound':True,
              'inputs_unchanged':all(digest(n)==h for n,h in bindings.items()),
              'source_unchanged':digest(Path(__file__).name)==manifest['source_sha256']}
    out = ROOT/(PREFIX+'_control_receipt.json')
    with out.open('x') as f:
        json.dump(result,f,indent=2,default=str)
        f.write('\n')
    print(json.dumps({'receipt':out.name,'sha256':digest(out.name),'verdict':result['verdict'],
                      'families':len(summaries),'parity_blocks':sum(len(x['blocks']) for x in summaries),
                      'histograms':free['histograms'],'labelled_dimension':free['labelled_dimension'],
                      'adverse_controls_rejected':3,'gap_lower_exact':str(gap)}),flush=True)


if __name__ == '__main__':
    main()
