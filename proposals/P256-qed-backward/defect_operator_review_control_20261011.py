from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile
from typing import cast

import sympy as s

ROOT = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reject(condition: bool, label: str) -> str:
    if condition:
        raise AssertionError('Adverse assertion unexpectedly accepted: ' + label)
    return label


def exterior_control(profiles: list[dict]) -> list[dict]:
    # Full annihilation matrices on all sixteen ordered Fock masks, rather
    # than importing the target's number-sector lifting routine.
    annihilators = []
    for mode in range(4):
        a = s.zeros(16)
        for mask in range(16):
            if mask & (1 << mode):
                a[mask ^ (1 << mode), mask] = (-1)**((mask & ((1 << mode)-1)).bit_count())
        annihilators.append(a)
    identity = s.eye(16)
    for i, j in itertools.product(range(4), repeat=2):
        assert annihilators[i]*annihilators[j].H + annihilators[j].H*annihilators[i] == (identity if i == j else s.zeros(16))
    X = s.Matrix([[0, 1], [1, 0]])
    A = s.diag(-1, 1, -1, 1)
    B = s.kronecker_product(X, s.diag(1, -1))
    C = s.kronecker_product(s.Matrix([[0, s.I], [-s.I, 0]]), X)
    R = s.diag(s.Rational(1, 2), s.Rational(1, 2), -s.Rational(1, 2), -s.Rational(1, 2))
    lifted_r = cast(s.Matrix, sum((R[i, j]*annihilators[i].H*annihilators[j] for i, j in itertools.product(range(4), repeat=2)), s.zeros(16)))
    result = []
    for profile in profiles:
        m, c, d = [s.Rational(profile[key+'_exact']) for key in ('m', 'c0', 'd')]
        h = m*A+c*B+d*C
        lifted_h = cast(s.Matrix, sum((h[i, j]*annihilators[i].H*annihilators[j] for i, j in itertools.product(range(4), repeat=2)), s.zeros(16)))
        sectors = []
        for number in range(5):
            masks = [mask for mask in range(16) if mask.bit_count() == number]
            hn, rn = lifted_h.extract(masks, masks), lifted_r.extract(masks, masks)
            x, e = s.symbols('x e')
            levels = [-e+c, -e-c, e+c, e-c]
            poly = s.prod(x-sum(levels[j] for j in indices) for indices in itertools.combinations(range(4), number))
            expected = s.rem(s.expand(poly), e*e-m*m-d*d, e)
            assert s.expand(hn.charpoly(x).as_expr()-expected) == 0
            u = s.Rational(1, 2)
            actual = hn+64*(rn+u*s.eye(len(masks)))**2
            wrong = hn+64*(rn**2+u*u*s.eye(len(masks)))
            assert actual-wrong == 128*u*rn
            if number in (1, 2, 3):
                reject(actual == wrong, 'dropping coherent affine cross n='+str(number))
            sectors.append({'number': number, 'dimension': len(masks), 'affine_cross_nonzero': actual != wrong})
        # A physical mask supported wholly on the first site has r=1/2.
        witness = 1
        assert (lifted_h+64*(lifted_r+s.Rational(1, 2)*identity)**2)[witness, witness]-(lifted_h+64*(lifted_r**2+s.Rational(1, 4)*identity))[witness, witness] == 32
        result.append({'role': profile['role'], 'complete_CAR_and_free_polynomials': True, 'sectors': sectors, 'affine_cross_witness_exact': '32'})
    return result


def projector_control() -> dict:
    g = s.Matrix([s.Rational(3, 5), s.Rational(4, 5)])
    excited = s.Matrix([-s.Rational(4, 5), s.Rational(3, 5)])
    G = g*g.H
    H = s.eye(2)-G
    rho = s.diag(s.Rational(1, 2), -s.Rational(1, 2))
    shifted = H+128*rho+64*s.eye(2)
    assert G*H == H*G
    reject(G*rho == rho*G, 'bare ground projector commutes with rho')
    reject(G*shifted == shifted*G, 'bare ground projector commutes on affine block')
    # Two cells share the same flow label. On nonzero flow S_v=0 at BOTH.
    local0 = s.kronecker_product(H, s.eye(2))+s.kronecker_product(s.eye(2), H)
    local1 = s.kronecker_product(shifted, s.eye(2))+s.kronecker_product(s.eye(2), H-128*rho+64*s.eye(2))+64*s.eye(4)
    full = s.diag(local0, local1)
    P0 = s.diag(s.kronecker_product(G, s.eye(2)), s.zeros(4))
    P1 = s.diag(s.kronecker_product(s.eye(2), G), s.zeros(4))
    flow0 = s.diag(s.eye(4), s.zeros(4))
    ground = P0*P1
    count = 2*s.eye(8)-P0-P1
    assert P0*P1 == P1*P0
    assert full*P0 == P0*full and full*P1 == P1*full
    neutral = cast(s.Matrix, s.kronecker_product(excited, g)).col_join(s.zeros(4, 1))
    assert flow0*neutral == neutral and ground*neutral == s.zeros(8, 1)
    assert (s.eye(8)-ground)*neutral == neutral and (s.eye(8)-flow0)*neutral == s.zeros(8, 1)
    assert count*neutral == neutral
    reject(flow0 == ground, 'ground-flow conflation')
    # A serialized approximate vector is not an exact spectral projector.
    proxy = s.Matrix([1, 0])*s.Matrix([[1, 0]])
    reject(proxy*H == H*proxy, 'approximate source projector is reducing')
    reject(s.diag(1, 0)*H == H*s.diag(1, 0), 'rho-zero/coordinate proxy is reducing')
    # A true global gap does not pay two instantaneous defects.
    gap_only = s.diag(0, 1, 1)
    wrong_count = s.diag(0, 1, 2)
    reject(all(value >= 0 for value in (gap_only-wrong_count).diagonal()), 'global gap implies additive count')
    assert local0 == 2*s.eye(4)-s.kronecker_product(G, s.eye(2))-s.kronecker_product(s.eye(2), G)
    return {'scope': 'EXACT_RATIONAL_SMALL_OPERATOR_CONTROL_NOT_ACTUAL_G', 'bare_G_shifted_commutator_nonzero': True, 'masked_P_commutation': True, 'shared_flow_projectors_commute': True, 'neutral_flow0_defects': 1, 'ground_flow_conflation_rejected': True, 'source_proxy_rejected': True}


def geometry_control(n: int) -> dict:
    sites = list(itertools.product(range(n), repeat=3))
    def next_site(x, axis, distance=1):
        return tuple((value+distance) % n if j == axis else value for j, value in enumerate(x))
    def cell(x):
        return (x[0]//2, x[1], x[2])
    def matching(x, axis):
        return axis == 0 and x[0] % 2 == 0
    external = [(x, axis) for x in sites for axis in range(3) if not matching(x, axis)]
    incidences = Counter()
    pairs = Counter()
    for x, axis in external:
        y = next_site(x, axis)
        incidences[cell(x)] += 1
        incidences[cell(y)] += 1
        pairs[tuple(sorted((cell(x), cell(y))))] += 1
    assert len(external) == 5*n**3//2 and set(incidences.values()) == {10}
    # Actual opposite-flow parallel raw edges: Q=0,u=+/-1,w=2.
    parallel = {((0, 0, 0), 1): 1, ((1, 0, 0), 1): -1}
    def evaluate(flow):
        d = Counter()
        w = Counter()
        for (x, axis), k in flow.items():
            assert not matching(x, axis)
            y = next_site(x, axis)
            d[x] -= k
            d[y] += k
            w[cell(x)] += k*k
            w[cell(y)] += k*k
        assert 32*sum(w.values()) == 64*sum(k*k for k in flow.values())
        assert sum(d.values()) == 0
        states = []
        for v in incidences:
            left = (2*v[0], v[1], v[2])
            right = next_site(left, 0)
            Q, u = d[left]+d[right], Fraction(d[right]-d[left], 2)
            assert 0 <= 256+Q <= 512
            assert Q*Q <= 10*w[v] and (2*u)**2 <= 10*w[v]
            assert (2*u-Q) % 2 == 0
            # Explicit canonical Fock mask occupancy witness at each site.
            q0 = max(-128, Q-128)
            q1 = Q-q0
            rho = Fraction(q0-q1, 2)
            E = -(rho+u)
            assert E.denominator == 1
            assert q0 == d[left]-E and q1 == d[right]+E
            if w[v]:
                states.append({'cell': v, 'Q': Q, 'u': str(u), 'w': w[v]})
        return states
    parallel_states = evaluate(parallel)
    assert sorted((v['Q'], v['u'], v['w']) for v in parallel_states) == [(0, '-1', 2), (0, '1', 2)]
    reject(sum(parallel.values())**2 == sum(k*k for k in parallel.values()), 'collapsing parallel RAW flows')
    inventory = Counter()
    for x, axis, length in itertools.product(sites, range(3), (1, 2)):
        raw_path = [(next_site(x, axis, j), axis) for j in range(length)]
        flow = {edge: 1 for edge in raw_path if not matching(*edge)}
        external_count = len(flow)
        inventory[axis, length, x[0] % 2, external_count] += 1
        if external_count:
            states = evaluate(flow)
            assert sum(v['w'] for v in states) == 2*external_count
    assert sum(inventory.values()) == 6*n**3
    assert sum(value for key, value in inventory.items() if key[-1] == 0) == n**3//2
    # Histories cancel or revisit edges; flow, not history length, is state.
    chosen = external[0]
    cancelled = evaluate({chosen: 1-1})
    repeated = evaluate({chosen: 1+1})
    assert not cancelled and sum(v['w'] for v in repeated) == 8
    reject(bool(cancelled), 'cancelled two-hop history forces two defects')
    reject(sum(v['w'] for v in repeated) == 2, 'repeated traversal treated as one unit raw flow')
    return {'N': n, 'cells': n**3//2, 'external_raw_edges': len(external), 'raw_incidence_degree': 10, 'parallel_pair_excess': sum(v-1 for v in pairs.values()), 'parallel_opposite_flow_states': parallel_states, 'all_actual_positive_paths': 6*n**3, 'path_classes': [{'axis': k[0], 'length': k[1], 'tail_parity': k[2], 'external_edges': k[3], 'count': v} for k, v in sorted(inventory.items())], 'simultaneous_allocation_checked_per_path': True, 'history_count_rejected': True}


def domain_controls() -> dict:
    # An actual divergence-free external cycle permits unbounded multiplicity
    # with n=256 at every cell; no occupancy bound cuts off this cycle.
    # The series tail demonstrates why finite support is a form core, not
    # the whole domain, and why arbitrary l2 is not the form domain.
    m = s.symbols('m', integer=True, positive=True)
    finite_energy_tail = s.summation(2**(-3*m), (m, 1, s.oo))
    assert finite_energy_tail == s.Rational(1, 7)
    finite_norm = s.summation(4**(-m), (m, 1, s.oo))
    infinite_form = s.summation(4**m * 4**(-m), (m, 1, s.oo))
    assert finite_norm == s.Rational(1, 3) and infinite_form == s.oo
    reject(infinite_form.is_finite is True, 'every l2 vector is in the electric form domain')
    return {'unbounded_cycle': 'all positive axis1 edges on one periodic line with amplitude m; divergence0, canonical cells, m arbitrary integer', 'form_core_example': 'k_m=2^m,a_m=2^(-5m/2); electric form summand2^(-3m), total1/7; truncations converge in form norm', 'l2_not_form_domain': 'k_m=2^m,a_m=2^(-m); norm sum4^(-m)<infinity but each electric form summand1, sum infinity', 'finite_support_not_full_domain': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, default=ROOT / 'defect_operator_review_request_20261011.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'defect_operator_review_receipt_20261011.json')
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    for name, expected in request['input_sha256'].items():
        assert sha((ROOT/name).read_bytes()) == expected, name
    with ZipFile(ROOT/request['frozen_archive']) as archive:
        for name, expected in request['archive_member_sha256'].items():
            assert sha(archive.read(name)) == expected, name
        profiles = json.loads(archive.read('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'))['profiles']
        old_verdict = json.loads(archive.read('reference_h0_mathreview_verdict_20261011.json'))
        current_root = sha(archive.read('root_pair_reference_parent_20261011.py'))
    with ZipFile(ROOT/request['older_source_archive']) as archive:
        original_root = sha(archive.read('root_pair_reference_parent_20261011.py'))
    assert original_root == old_verdict['binding']['root_pair_reference_parent_20261011.py']
    assert current_root != original_root
    result = {'kind': 'INDEPENDENT_EXACT_MATCHING_OPERATOR_PROJECTOR_DOMAIN_CONTROLS', 'source_sha256': sha(Path(__file__).read_bytes()), 'input_sha256': request['input_sha256'], 'archive_member_sha256': request['archive_member_sha256'], 'source_identity': {'older_reviewed_root_source': original_root, 'frozen_matching_epoch_root_source': current_root, 'same_bytes': False, 'no_rebinding': True}, 'full_CAR': exterior_control(profiles), 'projectors': projector_control(), 'geometry': [geometry_control(n) for n in (8, 10)], 'domain': domain_controls(), 'target_reruns': 0, 'physical_claims': False}
    for name, expected in request['input_sha256'].items():
        assert sha((ROOT/name).read_bytes()) == expected, name
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'sha256': sha(args.output.read_bytes()), 'target_reruns': 0}), flush=True)


if __name__ == '__main__':
    main()