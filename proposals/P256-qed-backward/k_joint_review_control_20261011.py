"""Independent exact frozen-scope review: no producer main, root or eigensolver.

Only the pinned native inertia helper is called. Native candidate matrices,
row certificates and prices are checked by separate implementations.
"""
from __future__ import annotations

from fractions import Fraction as F
from math import comb
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile
from typing import Any, cast

import numpy as np
import sympy as sp
import vacuum_connected_native_scientist_20261011 as native

ROOT = Path(__file__).resolve().parent
GAMMA = F(46955312929, 34359738368)
ZERO = (F(0), F(0))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT / name).read_bytes())


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def mul(a, b):
    return a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]


def scale(a, c):
    return a[0]*c, a[1]*c


def decode(rows):
    return [[tuple(F(x) for x in z) for z in row] for row in rows]


def norm(z):
    return abs(z[0]) + abs(z[1])


def matmul(a, b):
    n = len(a)
    return [[sum_complex(mul(a[i][k], b[k][j]) for k in range(n))
             for j in range(n)] for i in range(n)]


def sum_complex(values):
    result = ZERO
    for z in values:
        result = add(result, z)
    return result


def characteristic(a):
    """Faddeev-LeVerrier in exact Q(i), independently of LDL factorization."""
    n = len(a)
    b = [[(F(int(i == j)), F(0)) for j in range(n)] for i in range(n)]
    coefficients = [F(1)]
    for k in range(1, n + 1):
        b = matmul(a, b)
        trace = sum_complex(b[i][i] for i in range(n))
        assert trace[1] == 0
        c = -trace[0] / k
        coefficients.append(c)
        for i in range(n):
            b[i][i] = add(b[i][i], (c, F(0)))
    assert all(z == ZERO for row in b for z in row)
    return coefficients


def variations(values):
    signs = [1 if x > 0 else -1 for x in values if x]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def polynomial_negative_count(coefficients, shift):
    """All roots are real: Descartes variations of p(shift-t) are exact.

    No numerical spectral estimates, eigenvectors or old bisection are used.
    A real-rooted polynomial with nonzero constant has exact positive-root
    count equal to its coefficient sign variations, with multiplicity.
    """
    n = len(coefficients) - 1
    ascending = list(reversed(coefficients))
    translated = [sum((ascending[j]*comb(j, k)*shift**(j-k)
                       for j in range(k, n + 1)), F(0)) * (-1)**k
                  for k in range(n + 1)]
    assert translated[0] != 0
    return variations(reversed(translated))


def independently_certify_role(role):
    h = decode(role['pencil_onebody_exact'])
    assert len(h) == 8 and all(len(r) == 8 for r in h)
    assert all(h[i][j] == (h[j][i][0], -h[j][i][1])
               for i in range(8) for j in range(8))
    no_cross = all(h[i][j] == ZERO and h[j][i] == ZERO
                   for i in range(4) for j in range(4, 8))
    repeated = no_cross and all(h[i][j] == h[i+4][j+4]
                               for i in range(4) for j in range(4))
    cert = role['certificate']
    assert cert['exact_identical_spin_blocks'] == repeated
    # Each actual labelled block is independently retained, even when unequal.
    blocks = [h] if not no_cross else [h[:4], h[4:]]
    if no_cross:
        blocks = [[row[:4] for row in h[:4]], [row[4:] for row in h[4:]]]
    polynomials = [characteristic(a) for a in blocks]
    intervals = [tuple(F(x) for x in r['exact']) for r in cert['intervals']]
    effective_intervals = intervals * 2 if repeated else intervals
    assert len(effective_intervals) == 8
    cache = {}
    endpoints = []
    for rank, ((lo, hi), recorded) in enumerate(zip(intervals, cert['endpoint_inertias'])):
        assert 0 < hi-lo <= F(1, 256)
        for endpoint in (lo, hi):
            if endpoint not in cache:
                independent = sum(polynomial_negative_count(c, endpoint) for c in polynomials)
                observed_native = native.inertia(h, endpoint)
                assert independent == observed_native
                cache[endpoint] = independent
        count = (cache[lo], cache[hi])
        expected = tuple(x*2 for x in recorded) if repeated else tuple(recorded)
        assert count == expected
        # Identical-spin certificates index four local ranks; otherwise eight.
        full_rank = 2*rank if repeated else rank
        assert count[0] <= full_rank and count[1] >= full_rank+1
        endpoints.append(dict(interval_exact=[str(lo), str(hi)],
                              polynomial_and_native_counts=list(count)))
    return h, effective_intervals, dict(role=role['role'], count=role['count'],
        identical_spin_blocks=repeated, offspin_zero=no_cross,
        characteristic_polynomials=[[str(x) for x in c] for c in polynomials],
        independently_exercised_endpoints=len(cache), endpoints=endpoints)


def freeframe():
    out = []
    for row in load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json')['profiles']:
        m, c, d = (F(row[k]) for k in ('m_exact', 'c0_exact', 'd_exact'))
        block = [[ZERO for _ in range(4)] for _ in range(4)]
        for i in range(4):
            block[i][i] = ((-1)**(i+1)*m, F(0))
        for i in (0, 1):
            block[i][i+2] = block[i+2][i] = ((-1)**i*c, F(0))
            block[i][3-i] = (F(0), d)
            block[3-i][i] = (F(0), -d)
        out.append([[block[i % 4][j % 4] if i//4 == j//4 else ZERO
                     for j in range(8)] for i in range(8)])
    return out


def scalar_upper(u, v, A, weight, g, m):
    assert u >= 0 and A > 0 and GAMMA > 0 and g >= 0
    edges = [GAMMA * F(17, 16)**j for j in range(230)]
    if weight == 'S':
        edges.insert(0, F(0))
    assert weight in ('Q', 'S')
    intervals = []
    for left, right in zip(edges, edges[1:]):
        wl = left + (GAMMA if weight == 'S' else 0)
        wr = right + (GAMMA if weight == 'S' else 0)
        n = u*right + v
        denominator = (left+A)**2*wl if n >= 0 else (right+A)**2*wr
        intervals.append(n/denominator - g*left/(A*(right+A)*wr)
                         + m/(wl if m >= 0 else wr))
    last = edges[-1]
    tail = (u + max(v, F(0))/last)/(last+A)**2 + max(m, F(0))/last
    return max([tail, *intervals]), dict(intervals=len(intervals),
        last_cut_exact=str(last), tail_upper_exact=str(tail),
        winning_interval_or_tail=[*intervals, tail].index(max([tail, *intervals])))


def occupation_costs(a, weights):
    """Build actual CAR transitions column->row, then weighted row sums.

    This orientation differs from the producer's occupied-row enumeration.
    """
    occupations = [[i for i in range(8) if mask >> i & 1] for mask in range(256)]
    diagonal_weights = [np.prod([weights[i] for i in occ]) for occ in occupations]
    diagonals = [sum((a[i][i][0] for i in occ), F(0)) for occ in occupations]
    radii = [F(0) for _ in range(256)]
    hopping = 0
    for column, occupied in enumerate(occupations):
        for j in occupied:
            removed = column ^ (1 << j)
            for i in range(8):
                if i in occupied:
                    continue
                row = removed | (1 << i)
                sign = (-1)**((column & ((1 << j)-1)).bit_count()
                              + (removed & ((1 << i)-1)).bit_count())
                coefficient = scale(a[i][j], sign)
                radii[row] += norm(coefficient)*diagonal_weights[column]/diagonal_weights[row]
                hopping += coefficient != ZERO
    records = []
    costs = {}
    for mask, occupied in enumerate(occupations):
        p = sum(i % 4 < 2 for i in occupied)
        m = len(occupied)-p
        exact = diagonals[mask]+radii[mask]
        rounded = -((-1024*exact.numerator)//exact.denominator)
        assert exact <= F(rounded, 1024) < exact+F(1, 1024)
        costs[p+m, p] = max(costs.get((p+m, p), rounded), rounded)
        records.append((p, m, exact, rounded))
    return costs, records, hopping


def row_candidate(candidate, operators, hfree, eg, error):
    u = F(candidate['slope_exact'])
    weights = [F(x) for x in candidate['weights_exact']]
    # Independent coordinates are (totalN, Nplus), not producer (Nplus,Nminus).
    sentinel = -(1 << 61)
    dp = np.full((257, 257), sentinel, dtype=np.int64)
    dp[0, 0] = 0
    scalar = F(0)
    hopping_rows = 0
    max_round = F(0)
    for role in candidate['roles']:
        r, mult = role['role'], role['count']
        c, d = operators[r]
        a = [[add(d[i][j], scale(hfree[r][i][j], -u)) for j in range(8)] for i in range(8)]
        costs, records, hopping = occupation_costs(a, weights)
        hopping_rows += hopping
        assert mult == (1 if r == 0 else 63)
        assert len(role['local_whole_Fock_rows']) == 256
        for mask, record in enumerate(role['local_whole_Fock_rows']):
            p, m, exact, rounded = records[mask]
            assert record['occupation_mask'] == mask
            assert (record['nplus'], record['nminus']) == (p, m)
            assert F(record['cost_exact']) == exact and record['outward_integer'] == rounded
        assert max(abs(x) for x in costs.values())*64 < (1 << 58)
        max_round += mult*max(F(x[3], 1024)-x[2] for x in records)
        scalar += mult*c
        for _ in range(mult):
            target = np.full_like(dp, sentinel)
            for (n, p), cost in costs.items():
                old = dp[:257-n, :257-p]
                possible = np.where(old == sentinel, sentinel, old+cost)
                np.maximum(target[n:, p:], possible, out=target[n:, p:])
            dp = target
    assert max_round < F(64, 1024)
    values = []
    for p, record in enumerate(candidate['whole_totalrho_profile']):
        assert record['Nplus'] == p and record['Nminus'] == 256-p
        assert int(dp[256, p]) != sentinel
        rho = F(p-128)
        value = scalar + F(int(dp[256, p]), 1024) - 64*u*rho*rho + u*eg + error
        assert F(record['TOTALrho_exact']) == rho
        assert F(record['total_row_upper_exact']) == value
        values.append(value)
    assert F(candidate['intercept_exact']) == max(values)
    assert candidate['worst_Nplus'] == values.index(max(values))
    return dict(slope=str(u), weights=[str(x) for x in weights],
                all512_occupation_rows_independent=True, local_rows=512,
                nonzero_CAR_hopping_transitions=hopping_rows,
                collective_charge_profiles=len(values),
                observed_rigorous_rounding_upper_exact=str(max_round),
                intercept_exact=str(max(values)), independent_DP_coordinates='(totalN,Nplus)')


def adversarial():
    K = sp.diag(1, 10)
    D = sp.Matrix([[9, 3], [3, 1]])
    A = sp.Rational(4)
    H = K+A*sp.eye(2)
    Fm = H.inv()
    assert K*D != D*K and D.det() == 0 and D.trace() == 10
    # Exact maxima are lambda_max(K)=lambda_max(D)=10; their ratio=1 fails.
    independent_slack = K-D
    assert independent_slack[0, 0] == -8 and independent_slack.det() == -81
    u, v = sp.Rational(10), sp.Rational(1)
    slack = u*K+v*sp.eye(2)-D
    transformed = Fm*slack*Fm
    assert slack[0, 0] > 0 and slack.det() > 0
    assert transformed[0, 0] > 0 and transformed.det() > 0
    P, V = D/10, 5*Fm
    positive_product = (P*V+V*P)/2
    assert positive_product.det() < 0
    X = sp.Matrix([[8, 1], [1, 11]])
    J = sp.Matrix([[1, 2], [-1, 1]])
    T = X*J-J*H
    G = J.T*J
    B = (J.T*T+T.T*J)/2
    W = J.T*X.inv()*J
    residual = Fm*T.T*X.inv()*T*Fm
    assert W == (G*Fm+Fm*G)/2-Fm*B*Fm+residual
    assert residual[0, 0] > 0 and residual.det() > 0
    assert B*Fm**2 != Fm*B*Fm
    assert B*Fm**2 != (B*Fm**2).T
    # Entire native-sized canonical occupation witness, not independent copies.
    masks = [51, 204]+[85]*62
    charges = [F(sum((1 if i % 4 < 2 else -1) for i in range(8) if mask >> i & 1), 2)
               for mask in masks]
    assert sum(mask.bit_count() for mask in masks) == 256
    assert sum(charges) == 0 and sum(q*q for q in charges) == 8
    rho_wrong = -64*10*sum(q*q for q in charges)
    assert rho_wrong == -5120 < -64*10*sum(charges)**2
    fluctuating = [255, 0]+[85]*62
    assert sum(mask.bit_count() for mask in fluctuating) == 256
    # A row certificate deleting hopping already fails on a two-mode exact CAR.
    hopping = sp.Matrix([[0, 1], [1, 0]])
    witness = sp.Matrix([1, 1])
    assert (witness.T*hopping*witness)[0] > 0
    return dict(K=[[str(x) for x in row] for row in K.tolist()],
        D=[[str(x) for x in row] for row in D.tolist()],
        noncommuting=True, exact_independent_max_ratio='1',
        wrong_ratio_slack_determinant=str(independent_slack.det()),
        correct_joint_slack_determinant=str(slack.det()),
        correct_congruence_slack_determinant=str(transformed.det()),
        positive_product_determinant=str(positive_product.det()),
        retained_residual_determinant=str(residual.det()),
        wrong_B_F_squared_rejected=True, omitted_positive_residual_rejected=True,
        canonical256_collective_rho='0', independent_sum_rho_square='8',
        wrong_independent_square_slack_exact=str(rho_wrong),
        labelled_N_fluctuation_retained=[8, 0, 4],
        hopping_deletion_rejected=True, native_canonical_dimension=str(comb(512, 256)))


def run(request_name, receipt_name):
    began = time.monotonic()
    request = load(request_name)
    assert all(sha(ROOT/n) == pin for n, pin in request['input_sha256'].items())
    capsule_checks = []
    for name in ('k_resolved_actual1_20261011_capsule.json', 'k_resolved_refined_actual1_20261011_capsule.json'):
        capsule = load(name)
        expected = {'request': capsule['request_sha256'],
                    'preexecution_archive': capsule['preexecution_sha256'],
                    'executed_archive': capsule['executed_sha256'],
                    'receipt': capsule['receipt_sha256'], 'exit': capsule['exit_sha256']}
        request_path = 'k_resolved_'+capsule['epoch']+'_request.json'
        assert sha(ROOT/request_path) == expected['request']
        for key in ('preexecution_archive', 'executed_archive', 'receipt', 'exit'):
            assert sha(ROOT/capsule[key]) == expected[key]
        exit_record = load(capsule['exit'])
        assert capsule['natural_exit'] and capsule['returncode'] == 0
        assert exit_record['natural_exit'] and exit_record['returncode'] == 0
        oldrequest = load(request_path)
        with zipfile.ZipFile(ROOT/capsule['preexecution_archive']) as archive:
            assert archive.read('request.json') == (ROOT/request_path).read_bytes()
            for n, pin in oldrequest['input_sha256'].items():
                with archive.open('inputs/'+n) as member:
                    assert hashlib.file_digest(cast(Any, member), 'sha256').hexdigest() == pin
        with zipfile.ZipFile(ROOT/capsule['executed_archive']) as archive:
            for n, pin in capsule['input_sha256'].items():
                with archive.open('inputs/'+n) as member:
                    assert hashlib.file_digest(cast(Any, member), 'sha256').hexdigest() == pin
            with archive.open(capsule['receipt']) as member:
                assert hashlib.file_digest(cast(Any, member), 'sha256').hexdigest() == capsule['receipt_sha256']
        capsule_checks.append(dict(capsule=name, all_archived_inputs_hash_bound=True,
                                   historical_natural_exit_observed=0))
    old = load('positive_residual_actual1_20261011_receipt.json')
    primary = load('k_resolved_actual1_20261011_receipt.json')
    refined = load('k_resolved_refined_actual1_20261011_receipt.json')
    centered = load('k_resolved_centered_actual1_20261011_receipt.json')
    eg = F(load('local_so5_ground_upper_complete_state_consumer_20261011_receipt.json')['ground_energy_interval_exact'][1])
    formation = old['formation']
    assert formation['accepted_residual_exact'] == '1/65536'
    assert formation['included_formation_exact'] == '1/262144'
    infidelity, eta = F(formation['infidelity_exact']), F(formation['eta_sqrt_transfer_exact'])
    assert infidelity == (F(1, 65536)/GAMMA)**2 and eta*eta >= 2*infidelity
    assert F(formation['projected_operator_price_exact']) == F(formation['density_price_exact'])+F(formation['beta_exact'])+2*eta
    assert formation == primary['formation'] == refined['formation']
    accounts = 0
    for axis in old['axes']:
        for row in axis['projection_accounting']:
            account = row['accounting']
            assert F(account['formation_L1_exact']) == max(F(account['pre_normal_order_term_pair_L1_exact']), F(account['post_normal_order_pre_outer_collection_L1_exact']))
            assert row['physical_count'] == (1 if row['role'] == 0 else 63)
            assert account['whole_central_labelled_number_unrestricted'] and account['scalar_retained']
            accounts += 1
    hfree = freeframe()
    pencils, rows, selection = [], [], []
    assert [a['axis'] for a in old['axes']] == [a['axis'] for a in primary['axes']] == [a['axis'] for a in refined['axes']] == [a['axis'] for a in centered['axes']]
    for ai, oldaxis in enumerate(old['axes']):
        original = oldaxis['scalar_anchor_sweep'][3]
        g0lo, g0hi = (F(x) for x in oldaxis['G_ground_expectation']['exact'])
        for ri, oldrow in enumerate(original['energies']):
            E, A = F(oldrow['E_exact']), F(original['anchor_exact'])-F(oldrow['E_exact'])
            error = F(oldrow['joint_operator_error_exact'])
            operators = [(F(r['scalar_exact']), decode(r['onebody_exact'])) for r in oldrow['joint_operator_onebody_exact']]
            de = F(old['domains']['whole_operator_two_cell_floor_exact'])-E
            assert de > 0 and A > 0 and old['domains']['physical_hopping'] == '1'
            assert error == F(original['signed_B_error_exact'])+F(original['TdagT_error_exact'])/de
            for role in range(2):
                br, tr = original['signed_B_onebody_exact'][role], original['TdagT_onebody_exact'][role]
                bm, tm = decode(br['onebody_exact']), decode(tr['onebody_exact'])
                assert operators[role][0] == -F(br['scalar_exact'])+F(tr['scalar_exact'])/de
                assert operators[role][1] == [[add(scale(bm[i][j], -1), scale(tm[i][j], 1/de)) for j in range(8)] for i in range(8)]
            mismatch = g0hi/A-F(oldrow['vacuum_w0']['exact'][0])
            for family, receipt in (('primary', primary), ('refined', refined)):
                output = receipt['axes'][ai]['energies'][ri]
                assert output['E_exact'] == oldrow['E_exact']
                candidate_prices = {'Q': [], 'S': []}
                for ci, candidate in enumerate(output['candidates']):
                    u, v = F(candidate['slope_exact']), F(candidate['intercept_exact'])
                    assert u >= 0
                    if candidate['design'] == 'JOINT_FREE_PENCIL_WITH_RETAINED_COLLECTIVE_SQUARE':
                        z = F(candidate['rho_shift_exact'])
                        levels, scalar, controls = [], F(0), []
                        for role in candidate['roles']:
                            r, mult = role['role'], role['count']
                            assert mult == (1 if r == 0 else 63)
                            h, intervals, certificate = independently_certify_role(role)
                            expected = [[add(operators[r][1][i][j], scale(hfree[r][i][j], -u)) for j in range(8)] for i in range(8)]
                            for i in range(8):
                                expected[i][i] = add(expected[i][i], (-64*u*z*(1 if i % 4 < 2 else -1), F(0)))
                            assert h == expected
                            levels.extend([x[1] for x in intervals]*mult)
                            scalar += mult*operators[r][0]
                            controls.append(certificate)
                        assert len(levels) == 512
                        assert v == scalar+sum(sorted(levels, reverse=True)[:256])+u*eg+64*u*z*z+error
                        assert F(candidate['coefficient_error_exact']) == error
                        pencils.append(dict(family=family, axis=oldaxis['axis'], E=str(E),
                            slope=str(u), intercept_exact=str(v), all512_global256=True,
                            coefficient_error_added_once=True, role_controls=controls))
                    else:
                        assert family == 'primary' and E == 0
                        rows.append(dict(axis=oldaxis['axis'], **row_candidate(candidate, operators, hfree, eg, error)))
                    for weight in ('Q', 'S'):
                        price, price_control = scalar_upper(u, v, A, weight, g0lo, mismatch)
                        candidate_prices[weight].append(price)
                        key = 'joint_and_scalar_'+weight+'_exact' if family == 'refined' else 'every_Q_joint_plus_scalar_price_exact'
                        if weight == 'Q' or family == 'refined':
                            assert F(candidate[key]) == price
                center = centered['axes'][ai]['energies'][ri]
                for weight in ('Q', 'S'):
                    best_index = min(range(len(output['candidates'])), key=lambda i: candidate_prices[weight][i])
                    best = output['candidates'][best_index]
                    assert output['best_'+weight+'_slope_exact'] == best['slope_exact']
                    price = candidate_prices[weight][best_index]
                    if family == 'refined':
                        price = min(price, F(center['joint_and_negative_scalar_'+weight+'_exact']))
                        assert F(output['joint_and_negative_scalar_'+weight+'_exact']) == price
                        component = F(center['centered_G_'+weight+'_upper_exact'])
                        upper_key = 'every_Q_upper_exact' if weight == 'Q' else 'allcentral_S_upper_exact'
                        assert F(output[upper_key]) == component+price
                        lower_key = 'every_Q_lower_exact' if weight == 'Q' else 'allcentral_S_lower_exact'
                        assert output[lower_key] == center[lower_key]
                        absolute_key = 'every_Q_absolute_exact' if weight == 'Q' else 'allcentral_S_absolute_exact'
                        assert F(output[absolute_key]) == max(F(output[upper_key]), F(output[lower_key]))
                    else:
                        assert F(output['K_resolved_joint_and_negative_scalar_'+weight+'_exact']) == price
                    selection.append(dict(family=family, axis=oldaxis['axis'], E=str(E), weight=weight,
                        domain='k>=gamma on Q' if weight == 'Q' else 'k>=0 whole Pg/Qg; denominator k+gamma',
                        selected_slope=best['slope_exact'], exact_scalar_upper=str(price),
                        tail_control=scalar_upper(F(best['slope_exact']), F(best['intercept_exact']), A, weight, g0lo, mismatch)[1]))
    assert all(r['selected_slope'] == '88' for r in selection if r['family'] == 'refined')
    physical = [dict(axis=a['axis'], upper_exact=a['energies'][1]['every_Q_upper_exact'],
                     upper_display=a['energies'][1]['every_Q_upper_display'],
                     absolute_exact=a['energies'][1]['every_Q_absolute_exact'],
                     absolute_display=a['energies'][1]['every_Q_absolute_display']) for a in refined['axes']]
    assert all(F(r['upper_exact']) > 1 and F(r['absolute_exact']) > 2 for r in physical)
    assert all(sha(ROOT/n) == pin for n, pin in request['input_sha256'].items())
    result = dict(kind='NEW_INDEPENDENT_EXACT_K_JOINT_REVIEW_CONTROL_NOT_PHYSICAL_ADMISSION',
        governing_plan=request['governing_plan'], source_sha256=sha(Path(__file__)),
        input_sha256=request['input_sha256'], capsule_checks=capsule_checks,
        exact_pencil_controls=pencils, exact_collective_row_controls=rows,
        exact_selections=selection, formation_accounts_checked=accounts,
        source_cap='1/65536', formation_cap='1/262144', infidelity_exact=str(infidelity),
        eta_squared_is_outward=True, eta_exact=str(eta), adverse=adversarial(),
        physical1=physical, failed_sufficient_guards_not_phase_no_go=True,
        centered_price_only_interface_checked=True, conditional_ground_and_operation_premises=True,
        no_old_producer_main_eigensolver_root_checker=True, targets_unchanged=True,
        campaign_complete=False, elapsed_seconds=time.monotonic()-began)
    (ROOT/receipt_name).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(pencils=len(pencils), row_certificates=len(rows), selections=len(selection),
        polynomial_native_endpoint_comparisons=sum(r['independently_exercised_endpoints'] for p in pencils for r in p['role_controls']),
        physical1=[dict(axis=r['axis'], upper=r['upper_display'], absolute=r['absolute_display']) for r in physical],
        targets_unchanged=True)), flush=True)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
