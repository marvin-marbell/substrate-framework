"""Independent exact body/physical-probability consumer, never a ground solver."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as Q
import hashlib
import itertools
import json
from math import isqrt
from pathlib import Path
from typing import cast
import sympy as s


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_upper(value: Q, bits: int = 512) -> Q:
    scale = 1 << bits
    return Q(isqrt(value.numerator * scale * scale // value.denominator) + 1, scale)


def hs2(a: s.Matrix) -> s.Expr:
    return s.simplify(s.trace(a.H*a))


def car_density(states: list[tuple[int, int, s.Rational]], modes: int) -> s.SparseMatrix:
    """Trace environment after annihilation; no Slater refill or local rank imposed."""
    groups: dict[tuple[int, int], list[tuple[int, s.Expr]]] = defaultdict(list)
    for environment, mask, amplitude in states:
        for j in range(modes):
            if mask & (1 << j):
                sign = (-1)**((mask & ((1 << j)-1)).bit_count())
                groups[(environment, mask ^ (1 << j))].append((j, sign*amplitude))
    entries: dict[tuple[int, int], s.Expr] = defaultdict(lambda: s.S.Zero)
    for values in groups.values():
        for i, ai in values:
            for j, aj in values:
                entries[(i, j)] += s.conjugate(ai)*aj
    return s.SparseMatrix(modes, modes, entries)


def native_partial_trace(profiles: list[dict]) -> dict:
    n = 4
    sites = list(itertools.product(range(n), repeat=3))
    index = {v: i for i, v in enumerate(sites)}
    sx = s.Matrix([[0, 1], [1, 0]])
    sy = s.Matrix([[0, -s.I], [s.I, 0]])
    sz = s.diag(1, -1)
    g = [s.kronecker_product(sz, s.eye(2))] + [s.kronecker_product(sx, p) for p in (sx, sy, sz)]
    body = [g[0]/3+g[1]/7, g[1]/4+g[0]/8, g[2]/5, g[3]/6]
    phase = s.Rational(3, 5)+4*s.I/5
    twist = s.diag(phase, s.conjugate(phase), 1, 1)
    fields = [[b*(1+s.Rational(sum(x), 32)) for b in body] for x in sites]
    edges: dict[tuple[int, int], s.Matrix] = {}
    for x in sites:
        for axis in range(3):
            y = list(x); y[axis] = (y[axis]+1) % n
            edges[(index[x], axis)] = twist if x == (0, 0, 0) and axis == 0 else s.eye(4)
    assert s.simplify(twist.H*twist) == s.eye(4) and s.simplify(twist.det()) == 1
    # A native contractible face is curved; no all-links tree flattening claim.
    assert edges[(0, 0)] != s.eye(4)
    def assemble(profile: dict) -> s.SparseMatrix:
        entries: dict[tuple[int, int], s.Expr] = defaultdict(lambda: s.S.Zero)
        for x in sites:
            ix = index[x]
            onsite = cast(s.Matrix, s.Rational(profile['onsite'])*fields[ix][0])
            for i in range(4):
                for j in range(4):
                    entries[(4*ix+i, 4*ix+j)] += onsite[i, j]
            for path in profile['paths']:
                axis, length = path['axis'], path['length']
                y = list(x); U = s.eye(4)
                for _ in range(length):
                    U = U*edges[(index[tuple(y)], axis)]
                    y[axis] = (y[axis]+1) % n
                iy = index[tuple(y)]
                ap = -1 if x[axis]+length >= n else 1
                c0, ci = s.Rational(path['c0']), s.Rational(path['ci'])
                f = ap*(c0*(fields[ix][0]*U+U*fields[iy][0])+s.I*ci*(fields[ix][axis+1]*U+U*fields[iy][axis+1]))/2
                for i in range(4):
                    for j in range(4):
                        entries[(4*ix+i, 4*iy+j)] += f[i, j]
                        entries[(4*iy+j, 4*ix+i)] += s.conjugate(f[i, j])
        h = s.SparseMatrix(4*len(sites), 4*len(sites), entries)
        assert h == h.H and s.trace(h) == 0
        return h
    h = assemble(profiles[0])
    charged = [assemble(p) for p in profiles[1:]]
    V, modes = len(sites), 4*len(sites)
    base = sum(3 << (4*x) for x in range(V))
    swap0 = base ^ 1 ^ (1 << 6)
    swap1 = base ^ (1 << 4) ^ (1 << 10)
    states = cast(list[tuple[int, int, s.Rational]], [(0, base, s.Rational(9, 25)), (0, swap0, s.Rational(12, 25)),
              (1, base, s.Rational(12, 25)), (1, swap1, s.Rational(16, 25))])
    assert all(mask.bit_count() == 2*V for _, mask, _ in states)
    assert sum(a*a for _, _, a in states) == 1
    Gamma = car_density(states, modes)
    p0 = car_density(cast(list[tuple[int, int, s.Rational]], [(0, base, s.Rational(3, 5)), (0, swap0, s.Rational(4, 5))]), modes)
    p1 = car_density(cast(list[tuple[int, int, s.Rational]], [(0, base, s.Rational(3, 5)), (0, swap1, s.Rational(4, 5))]), modes)
    assert p0*p0 == p0 and p1*p1 == p1
    assert Gamma == s.Rational(9, 25)*p0+s.Rational(16, 25)*p1
    assert s.trace(Gamma) == 2*V and Gamma*Gamma != Gamma
    assert Gamma*h != h*Gamma
    # Convex combination of projectors proves 0<=Gamma<=I exactly.
    local_traces = [s.trace(Gamma[4*x:4*x+4, 4*x:4*x+4]) for x in range(V)]
    assert local_traces[0] != 2 and local_traces[1] != 2
    energy = s.simplify(s.trace(Gamma*h))
    charged_energy = [s.simplify(s.trace(Gamma*c)) for c in charged]
    xq = [s.trace(b*b)/4 for b in body]
    J = s.expand(4*sum(s.trace((a*b+b*a)**2) for a in body for b in body))
    delta = s.expand(J-64*sum(x*x for x in xq))
    self_terms = sum(hs2(b*b-x*s.eye(4)) for b, x in zip(body, xq))
    anti_terms = sum(hs2(cast(s.Matrix, body[i]*body[j]+body[j]*body[i])) for i in range(4) for j in range(4) if i != j)
    assert s.expand(delta-16*self_terms-4*anti_terms) == 0 and delta > 0
    assert J-64*sum((4*x)**2 for x in xq) != delta
    # Noncommuting exact square/anti transfer identity at rational perturbation.
    Phi = [b+g[(i+1)%4]/1024 for i, b in enumerate(body)]
    for i in range(4):
        d = Phi[i]-body[i]
        assert cast(s.Matrix, Phi[i]*d+d*body[i]-Phi[i]**2+body[i]**2).applyfunc(s.simplify) == s.zeros(4)
        for j in range(4):
            e = Phi[j]-body[j]
            assert cast(s.Matrix, d*Phi[j]+body[i]*e+e*Phi[i]+body[j]*d-Phi[i]*Phi[j]-Phi[j]*Phi[i]+body[i]*body[j]+body[j]*body[i]).applyfunc(s.simplify) == s.zeros(4)
    return {'N':n, 'modes':modes, 'canonical':2*V, 'environment_branches':2,
            'density_from_actual_sparse_CAR_partial_trace':True, 'density_not_projector':True,
            'density_noncommuting':True, 'first_site_traces':[str(t) for t in local_traces[:2]],
            'native_neutral_expectation':str(energy), 'charged_all_AP1_AP2_expectations':[str(t) for t in charged_energy],
            'charged_loads':[p['load'] for p in profiles[1:]], 'quarter_norms':[str(x) for x in xq],
            'Jordan_excess':str(delta), 'ordered_pairs':16, 'wrong_raw_trace_rejected':True,
            'scope':'Exact algebra consumer in actual native CAR/endpoint/AP1/AP2 module; configurations are NOT sampled actual ground and do not prove a ground state.'}


def all_partition_volume_bias() -> dict:
    n = 8; lengths = (3, 5); V = n**3
    sites = list(itertools.product(range(n), repeat=3))
    # One HS-unit traceless matrix; scalar amplitudes give rational radial caps.
    amplitudes = {x:Q(1+sum((j+1)*v for j, v in enumerate(x)) % 5) for x in sites}
    capped = {x:min(v, Q(2)) for x, v in amplitudes.items()}
    fourth = sum(v**4 for v in amplitudes.values())/V
    capped_second = sum(v*v for v in capped.values())/V
    original_second = sum(v*v for v in amplitudes.values())/V
    body2 = unweighted = deviation = 0
    fixed0_body2 = 0
    translated_fixed_root = Q()
    volumes = set()
    for shift in sites:
        axes = [[[(shift[i]+t) % n for t in range(ell_start, ell_start+ell)]
                 for ell_start, ell in ((0, lengths[0]), (lengths[0], lengths[1]))] for i in range(3)]
        unweighted_shift = 0
        root_values = {}
        for rectangle in itertools.product(*axes):
            box = list(itertools.product(*rectangle)); volume = len(box); volumes.add(volume)
            mean = sum(capped[x] for x in box)/volume
            body2 += Q(volume, V)*mean**2
            unweighted_shift += mean**2/8
            deviation += sum((amplitudes[x]-mean)**2 for x in box)/V
            if (0, 0, 0) in box: fixed0_body2 += mean**2
            for root_site in box:
                root_values[root_site] = mean**2
        unweighted += unweighted_shift
        assert len(root_values) == V
        translated_fixed_root += sum(root_values.values())/V
    body2 /= V; unweighted /= V; deviation /= V; fixed0_body2 /= V
    assert body2 <= capped_second <= original_second
    assert body2 != unweighted
    # Enumerate roots induced by translating the field AND partition jointly.
    translated_fixed_root /= V
    assert translated_fixed_root == body2
    tail_second = sum((amplitudes[x]-capped[x])**2 for x in sites)/V
    assert tail_second <= fourth/4
    return {'N':n,'R':3,'interval_lengths':lengths,'all_partition_shifts':V,'box_volumes':sorted(volumes),
            'volume_biased_mean_Bnorm2':str(body2),'wrong_unweighted_mean_Bnorm2':str(unweighted),
            'fixed_unaveraged_root_mean_Bnorm2':str(fixed0_body2),'translation_averaged_fixed_root_equals_uniform_root':True,
            'physical_s_average':str(original_second),'cap_second_tail':str(tail_second),
            'site_to_capped_body_squared_deviation':str(deviation),
            'scope':'Exact all-shift odd-rectangle probability algebra; method control, not native ground sampling.'}


def integrated_gate(profiles: list[dict], upper: dict) -> dict:
    m, c0, ci = (Q(profiles[0]['onsite']), Q(profiles[0]['paths'][0]['c0']), Q(profiles[0]['paths'][0]['ci']))
    b = abs(c0)+abs(ci); D = 2*abs(m)+12*b
    dch = sum(p['load']*(2*abs(Q(p['onsite']))+4*sum(abs(Q(t['c0']))+abs(Q(t['ci'])) for t in p['paths'])) for p in profiles[1:])
    assert dch == Q(855444290449274477, 35184372088832)
    N, R, L, M = 1 << 257, 1 << 256, 1 << 64, Q(1 << 14)
    a, c = Q(40, 1 << 2035), Q(10, 7*(1 << 2035))
    prices = {'field':(1024*L**3+D)*root_upper(48*R**4*a),
              'curvature':4*b*L*root_upper(1728*R**5*c),
              'cut_sew_BZ':(18+Q(66, 7))*b*L/R, 'tail':D*M/L**3}
    # 512-bit sqrt enclosure directly preserves all small gates here.
    lower_error = sum(prices.values())
    eps_quantum = Q(upper['epsilon_quantum'])
    # Independently compose the supplier's stated fullH prices at unchanged law.
    expected_quantum = Q(2)**(-3304)+Q(2)**(-4820)+2*Q(2)**(-4840)+Q(2)**(-8173)
    assert eps_quantum == expected_quantum and eps_quantum < Q(2)**(-3000)
    assert upper['quarter_normalization'] == 'x_mu=Tr(B_mu^2)/4'
    assert upper['f'] == '64*sum(x_mu^2)-2*BZmean(sqrt(M^2*x0+sum(D_i^2*x_i)))'
    assert upper['xstar'] == 'unique argmin on R_+^4 of f_infinity(x)'
    eps = Q(1 << 16, N)+eps_quantum
    charged = 12*dch/Q(1 << 8192)
    eta = lower_error+eps+charged
    assert lower_error < Q(1, 1 << 160) and eps < Q(1, 1 << 240) and eta < Q(1, 1 << 159)
    deviation = 2*M/L**2+96*R**4*a
    assert deviation < Q(1, 1 << 112)
    norm2 = eta/64
    site_L1 = 6*root_upper(deviation)+root_upper(norm2)
    assert site_L1 < Q(1, 1 << 52)
    assert 256*site_L1 < Q(1, 1 << 44)
    failures = [eta/(64*Q(1, 1 << 128)), eta/Q(1, 1 << 80), M/Q(1 << 32), deviation/Q(1, 1 << 84)]
    bounds = [Q(1,1 << k) for k in (37,79,18,28)]
    assert failures[0] < bounds[0] and failures[1] < bounds[1] and failures[2] == bounds[2] and failures[3] < bounds[3]
    assert sum(bounds) < Q(1, 1 << 17)
    assert 8*(2+Q(1, 1 << 64)) < 25  # ||B||tuple^2=4sum x<=8||x||2.
    self_bound = 261*Q(1,1 << 42)+Q(1,1 << 42)+2*Q(1,1 << 64)
    anti_bound = 522*Q(1,1 << 42)+Q(1,1 << 40)
    assert self_bound < Q(1,1 << 32) and anti_bound < Q(1,1 << 31)
    assert Q((2*R-1)**3)*M/L**4 > 1
    return {'Dch':str(dch),'N_min':str(N),'R':str(R),'L':str(L),
            'prices_exact_outward':{k:str(v) for k,v in prices.items()},'epsilon_upper':str(eps),
            'charged_price_per_Lambda_site':str(charged),'eta_upper':str(eta),'eta_lt_2power_minus159':True,
            'norm_distance_mean_square_upper':str(norm2),'DeltaJ_mean_upper':str(eta),
            'uncapped_site_deviation_upper':str(deviation),'quarter_norm_site_L1_upper':str(site_L1),
            'site_component_lower_window_failure_upper':str(256*site_L1),
            'four_joint_failure_outward_bounds':[str(x) for x in bounds], 'failure_sum':str(sum(bounds)),
            'self_defect_bound':str(self_bound),'ordered_anticommutator_bound':str(anti_bound),
            'box_simultaneous_cutoff_rejected':True,'strictness':'M<2^14 makes the cutoff bound strict despite outward equality',
            'retained':'Q/(Lambda V)+(F/Lambda)<W>, and full charged -Dch sqrt(<s>)/Lambda before optional NONNEGATIVE Q/F drop',
            'conditional':'Same physical minimizing family/moments and independently valid SAME-fstar full quantum supplier; no old -21 substitution.'}


def fourth_tail_family() -> dict:
    family = []
    for radius in (2,4,16,256,65536):
        mass = Q(1,radius**4)
        second = mass*radius**2
        fourth = mass*radius**4
        tail = fourth if radius > 1 else Q(0)
        assert second == Q(1,radius**2) and fourth == tail == 1
        family.append({'r':radius,'mass':str(mass),'remaining_atom_zero':str(1-mass),
                       'second_transfer':str(second),'fourth_moment':str(fourth),'fourth_tail_above_one':str(tail)})
    return {'family':family,'analytic':'For every threshold T choose r>T: E[radius^4 1(radius>T)]=1; sup_r tail never tends to zero. E radius^2=r^-2 tends to zero.',
            'verdict':'Bounded fourth moment and vanishing second-transfer do not imply fourth-tail uniform integrability.',
            'scope':'NONNATIVE scalar probability FAMILY, method-only falsifier; not a native physical ground, phase obstruction, or no-go.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, pin in request['pins'].items():
        if digest(Path(pin['path'])) != pin['sha256']:
            raise ValueError(f'Pinned input changed: {name}')
    profiles = json.loads(Path(request['pins']['neutral_body_parent_primitive_profiles_20261012.json']['path']).read_text())['primitive_profiles']
    upper = json.loads(Path(request['pins']['quantum_minimum_sea_actual3_20261012_interface.json']['path']).read_text())
    receipt = {'kind':'INDEPENDENT_EXECUTED_ACTUAL_MODULE_BODY_PROBABILITY_REVIEW_NOT_GROUND',
               'source_sha256':digest(Path(__file__)),'request_sha256':digest(args.request),'pins':request['pins'],
               'native_CAR_and_matrices':native_partial_trace(profiles),
               'all_partitions_volume_bias':all_partition_volume_bias(),
               'same_fstar_integrated_gate':integrated_gate(profiles,upper),
               'fourth_tail_UI_adverse':fourth_tail_family(),
               'scientific_admission':False,'law_changed':False,'campaign_complete':False}
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print('NEW exact native CAR partial trace/AP1/AP2/all partitions/volume bias/quarter and point-defect probability consumer passed; eta<2^-159 conditional, NONNATIVE UI family only; no ground/phase/sharp-atom adoption.', flush=True)


if __name__ == '__main__':
    main()
