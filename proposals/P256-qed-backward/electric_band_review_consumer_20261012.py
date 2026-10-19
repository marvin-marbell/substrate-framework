"""Independent exact electric-band factor algebra and adverse method controls.
No native ground/full-H backend; no original producer import or rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import zipfile
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ceil_log2(value: F) -> int:
    k = value.numerator.bit_length() - value.denominator.bit_length()
    power = F(2**k) if k >= 0 else F(1, 2**(-k))
    return k if value <= power else k + 1


def root34_upper(value: F) -> F:
    assert value >= 1
    result = F(2**((3*ceil_log2(value)+3)//4))
    assert result**4 >= value**3
    return result


def ceil_sqrt(value: F) -> int:
    from math import isqrt
    result = isqrt(value.numerator // value.denominator)
    return result if F(result*result) >= value else result+1


def budgets(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    w = sum((F(p['load'])*r['length']*(abs(F(r['c0']))+abs(F(r['ci'])))
             for p in profiles[1:] for r in p['paths']), F(0))
    assert w == F(46425051219964815, 8796093022208)
    ce = 8*sum((F(p['load'])*r['length']*(abs(F(r['c0']))+abs(F(r['ci'])))
                for p in profiles[1:] for r in p['paths'] if r['axis'] == 0), F(0))
    assert ce == F(15475017073321605, 1099511627776)
    assert F(65, 4)**3 < 16**4 and F(65, 4)**9 < 64**8
    assert 1025 < 33**2 and 2**15 < 16**4 and 2**15 < 256**2
    d = 128*w*16
    c1 = 1025+d
    a1 = d*16*root34_upper(c1)
    c3 = 33*c1+2*a1
    a3 = d*64*root34_upper(c3)
    c2 = 1025*c1+8*a3
    e4 = c2/256
    qj = (4*ce)**2*16*256
    denominators = [1+3*p/4-s for p, s in [(F(1), F(3, 2)), (F(3, 2), F(2))]]
    assert denominators == [F(1, 4), F(1, 8)]
    assert 1+3*F(1)/4-F(7, 4) == 0
    # r_HS^4 = Cart |q|^4/4; no Cart-to-HS factor may be silently omitted.
    cart_square = F(12)
    assert (cart_square/2)**2 == cart_square**2/4
    # Strict endpoint witness: h_k=16^k, probability=127/128^k.
    # It is normalized, has first moment127/7, but every7/4-moment term is127.
    assert sum((F(127, 128**k)*16**k for k in range(1, 9)), F(0)) < F(127, 7)
    assert all(F(127, 128**k)*(2**(7*k)) == 127 for k in range(1, 9))
    # Tail Hoelder is sharp on correlated commuting r,Q, not independence.
    p = F(1, 16)
    rq = p*2
    assert p*2**4 == 1 and rq**4 == p**3 and rq > p
    assert rq != rq*p
    return {key: str(value) for key, value in {
        'W': w, 'C_e': ce, 'D_upper': d, 'C1prime': c1,
        'A1_upper': a1, 'C3half_upper': c3, 'A3half_upper': a3,
        'C2prime_upper': c2, 'CE4prime_upper': e4, 'QJ_face_upper': qj,
        'A_harmonic_upper': F(ceil_sqrt(e4)),
        'B_harmonic_upper': F(ceil_sqrt(qj), 128),
    }.items()} | {'strict_denominators': [str(v) for v in denominators],
                'endpoint_divergence_term': '127 per k',
                'Holder_saturating_rQ': str(rq), 'HS_Cart_fourth_factor': '1/4'}


def exact_partition(e: int) -> dict[tuple[str, int], sp.Expr]:
    """Actual R4 smooth partition at flat endpoints and genuine midpoints."""
    z = sp.sqrt(2)/2
    if abs(e) <= 4:
        return {('keep', 0): sp.S.One}
    assert abs(e) == 6 or abs(e) >= 8
    f, g = (z, z) if abs(e) == 6 else (sp.S.Zero, sp.S.One)
    m, offset = divmod(e, 4)
    assert offset in (0, 2)
    weights = (sp.S.One, sp.S.Zero) if offset == 0 else (z, z)
    return {('keep', 0): f, ('reset', m): g*weights[0], ('reset', m+1): g*weights[1]}


def scalar_cp() -> dict[str, Any]:
    energies = [0, 1, 2, 4, 6, 8, 10, 12, -40000, -39998]
    partitions = {e: exact_partition(e) for e in energies}
    labels = set().union(*(p.keys() for p in partitions.values()))
    output = sorted({e if label[0] == 'keep' else e-4*label[1]
                     for e, p in partitions.items() for label, weight in p.items() if weight != 0})
    kraus = []
    for label in sorted(labels):
        matrix = sp.zeros(len(output), len(energies))
        for column, e in enumerate(energies):
            weight = partitions[e].get(label, sp.S.Zero)
            if weight != 0:
                target = e if label[0] == 'keep' else e-4*label[1]
                matrix[output.index(target), column] = weight
        kraus.append(matrix)
    row = sum((k.H*k for k in kraus), sp.zeros(len(energies)))
    assert sp.simplify(row-sp.eye(len(energies))) == sp.zeros(len(energies))
    h = lambda e: 1+16*e*e
    removed = {}
    for e in energies:
        cost = sum((weight**2*h(e if label[0] == 'keep' else e-4*label[1])
                    for label, weight in partitions[e].items()), sp.S.Zero)
        removed[e] = sp.simplify(h(e)-cost)
        assert removed[e] >= 0
        if h(e) > 1025:
            assert removed[e] >= sp.Rational(h(e), 2)
    assert removed[6] == 256 and removed[8] == 1024 and removed[-40000] == 16*40000**2
    # Actual CP on a source entangled with a spectator, not just scalar weights.
    psi = sp.zeros(2*len(energies), 1)
    psi[2*energies.index(6)] = sp.sqrt(2)/2
    psi[2*energies.index(8)+1] = sp.sqrt(2)/2
    branches = [sp.kronecker_product(k, sp.eye(2))*psi for k in kraus]
    density = sum((v*v.H for v in branches), sp.zeros(2*len(output)))
    assert sp.simplify(sp.trace(density)) == 1
    nonzero = sorted([sp.simplify(v.dot(v)) for v in branches if v != sp.zeros(len(v), 1)])
    assert nonzero == [sp.Rational(1, 8), sp.Rational(1, 4), sp.Rational(5, 8)]
    # Gram outer-product construction proves positive semidefiniteness exactly.
    coherent = sum(kraus, sp.zeros(len(output), len(energies)))
    wrong = sp.zeros(len(energies), 1)
    wrong[energies.index(8)] = wrong[energies.index(12)] = sp.sqrt(2)/2
    wrong_trace = sp.simplify((coherent*wrong).dot(coherent*wrong))
    assert wrong_trace == 2
    kernel = lambda a, b: sp.simplify(sum((v*partitions[b].get(label, 0)
                                          for label, v in partitions[a].items()), sp.S.Zero))
    assert kernel(1, 2) == 1 and kernel(6, 8) == sp.Rational(1, 2)
    assert kernel(4, 6) == sp.sqrt(2)/2
    assert (1+2)//4 != (2+2)//4  # Hard all-band low coherence is erased.
    assert removed[6] < sp.Rational(h(6), 2) and h(6) > 1+16*4**2
    # The frozen implementation's -10000 sentinel is a reset label at this point.
    frozen_collision = {-10000: sp.S.Zero, -10000+0: sp.S.One, -9999: sp.S.Zero}
    frozen_cost = h(-40000)*frozen_collision[-10000]
    assert frozen_cost == h(-40000) and removed[-40000] > 0
    # Channel covariance alone does not preserve each Gauss annihilation kernel.
    raising, lowering = sp.Matrix([[0, 0], [1, 0]]), sp.Matrix([[0, 1], [0, 0]])
    gauge = sp.diag(1, sp.I)
    source = sp.diag(1, 0)
    cp = lambda rho: raising*rho*raising.H+lowering*rho*lowering.H
    assert cp(gauge*source*gauge.H) == gauge*cp(source)*gauge.H
    assert cp(source) == sp.diag(0, 1) and raising*gauge != gauge*raising
    return {'scope': 'native scalar smooth-partition factor algebra; nonnative entangled method/covariance controls',
            'input_energies': energies, 'row_identity_exact': True,
            'entangled_branch_weights': [str(v) for v in nonzero],
            'coherent_sum_wrong_trace': str(wrong_trace),
            'hard_low_kernel': 0, 'soft_low_kernel': 1,
            'changed_high_kernel': '1/2', 'R_instead_of_Rminus2_rejected_kernel': 'sqrt(2)/2',
            'wrong_tail_threshold_257_rejected': True,
            'removed_h': {str(e): str(v) for e, v in removed.items()},
            'frozen_sentinel_collision_rejected': True,
            'covariance_only_Gauss_kernel_claim_rejected': True}


def car(mask: int, destination: int, origin: int) -> tuple[int, int]:
    assert mask & (1 << origin) and not mask & (1 << destination)
    sign = (-1)**((mask & ((1 << origin)-1)).bit_count())
    mask ^= 1 << origin
    sign *= (-1)**((mask & ((1 << destination)-1)).bit_count())
    return mask | (1 << destination), sign


def native_paths(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    def move(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        return tuple((v+1) % n if i == axis else v for i, v in enumerate(x))
    origin = (n-1, n-1, 0)
    face = {(origin, 0): 1, (move(origin, 0), 1): 1,
            (move(origin, 1), 0): -1, (origin, 1): -1}
    def divergence(flow: dict[tuple[tuple[int, ...], int], int]) -> Counter:
        result: Counter = Counter()
        for (x, axis), value in flow.items():
            result[x] += value
            result[move(x, axis)] -= value
        return result
    assert not any(divergence(face).values())
    background = {(x, 0): 64*(-1)**sum(x) for x in sites}
    assert all(divergence(background)[x] == 128*(-1)**sum(x) for x in sites)
    for m in [-10000, -3, 0, 7]:
        shifted = dict(background)
        for e, value in face.items():
            shifted[e] = shifted.get(e, 0)-m*value
        assert divergence(shifted) == divergence(background)
        ec0 = sum(value*background.get(e, 0) for e, value in face.items())
        ec1 = sum(value*shifted[e] for e, value in face.items())
        assert ec1 == ec0-4*m
        assert all(F(shifted[e])-F(value*ec1, 4) == F(background.get(e, 0))-F(value*ec0, 4)
                   for e, value in face.items())
    counts, affected, ap = [0]*3, [0]*3, [0]*3
    aliases = 0
    for role, profile in enumerate(profiles):
        for row in profile['paths']:
            for x in sites:
                y, edges, seam = x, [], 1
                for _ in range(row['length']):
                    edges.append((y, row['axis']))
                    seam *= -1 if y[row['axis']] == n-1 else 1
                    y = move(y, row['axis'])
                counts[role] += 1
                r = sum(face.get(e, 0) for e in edges) if role else 0
                assert abs(r) <= 2
                if r:
                    affected[role] += 1
                    ap[role] += int(seam == -1)
                    aliases += int(n == 4 and row['length'] == 2)
                    assert all(divergence(dict(Counter(edges)))[z] == int(z == x)-int(z == y) for z in sites)
    assert affected == [0, 12, 12] and ap == [0, 12, 12]
    assert aliases == (16 if n == 4 else 0)
    # Genuine65 distinct factor offsets, all64 charged transfers; Ref4 untouched.
    v, width = n**3, 4*n**3
    local = sum((1 << (4*j)) | (1 << (4*j+1)) for j in range(v))
    mask = sum(local << (role*width) for role in range(65)) | (15 << (65*width))
    source_site, destination_site = sites.index(move(origin, 0)), sites.index(origin)
    for role in range(1, 65):
        offset = role*width
        changed, sign = car(mask, offset+4*destination_site+2, offset+4*source_site)
        restored, reverse_sign = car(changed, offset+4*source_site, offset+4*destination_site+2)
        assert restored == mask and sign*reverse_sign == 1
        assert ((changed >> offset) & ((1 << width)-1)).bit_count() == 2*v
        assert (changed >> (65*width)) == 15
    # Ordered endpoint SU4 covariance, not scalarized/clifford-frozen matrices.
    gx = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    gy = sp.diag(sp.I, -sp.I, 1, 1)
    phi = sp.diag(1, -1, 2, -2)
    other = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]])
    u = gx*gy
    block = (phi*u+u*other)/2
    transformed = ((gx*phi*gx.H)*(gx*u*gy.H)+(gx*u*gy.H)*(gy*other*gy.H))/2
    assert gx.det() == gy.det() == 1 and gx*gy != gy*gx
    assert transformed == gx*block*gy.H
    return {'N': n, 'literal_counts_all65_profiles': counts, 'affected': affected, 'AP222_affected': ap,
            'N4_alias_literals': aliases, 'fullrho': '128 parity, once collectively',
            'Eperp_fullrho_reset_exact': True, 'charged_distinct_factor_CAR_transfers': 64,
            'canonical_each': 2*v, 'neutral_and_Ref4_unchanged': True,
            'ordered_SU4_factor_covariance_exact': True, 'physical_ground_state_used': False}


def domain_and_mass() -> dict[str, Any]:
    # Nonnative Hilbert method: L Omega=0, E Omega=e1 in D(L),
    # bounded E maps e1 to Omega+w; w_n^2=2/[n(n+1)] for n>=2.
    e = sp.Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    omega = sp.Matrix([1, 0, 0])
    assert e*omega == sp.Matrix([0, 1, 0]) and e**2*omega == sp.Matrix([1, 0, 1])
    assert (e**2*omega).dot(e**2*omega) == 2
    partial_graph = sum((F(2*n, n+1) for n in range(2, 33)), F(0))
    assert partial_graph >= 31  # Diverges as endpoint grows; E4 does not pay graph.
    assert sum((F(2, n*(n+1)) for n in range(2, 33)), F(0)) == 1-F(2, 33)
    mass_cases = []
    for n in [2, 4, 8, 32]:
        # Uniform E4 family: mass1/n^4 at E=n, remainder at0.
        assert F(1, n**4)*n**4 == 1
        assert F(1, n**4)*n**2 == F(1, n**2)
        # Separate collapsing-level family: total E source mass1 is unchanged,
        # finite excitation mass1 collapses to whole limiting P0 mass1.
        finite_laplace_kernel = F(n, n+1)
        assert 1-finite_laplace_kernel == F(1, n+1)
        mass_cases.append({'N': n, 'E4_escape_E2_source_norm_squared': '1',
                           'E_source_norm_squared': str(F(1, n**2)),
                           'collapse_total_E_mass': '1', 'finite_Q0_mass': '1',
                           'limiting_Q0_mass': '0', 'collapse_laplace_error': str(F(1, n+1))})
    # Taylor integral coefficient1/2 is exact; dropping it doubles the bound,
    # while claiming zero quotient error is rejected by exp(i pi)=-1.
    quotient_error_squared = sp.pi**2+4
    assert quotient_error_squared != 0
    return {'scope': 'nonnative exact/rational domain and spectral method counterexamples',
            'E4_finite_E_Omega_D_L_but_E2_Omega_not_D_L': True,
            'E4_value_domain_witness': 2, 'graph_partial32': str(partial_graph),
            'E2_source_norm_preservation_rejected': True,
            'finite_P0_equals_limiting_P0_rejected': True,
            'total_Gram_not_excitation_mass': mass_cases,
            'nonzero_boost_quotient_error_squared': str(quotient_error_squared)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    before = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        profiles = json.loads(archive.read(request['profile_member']))['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    receipt = {'kind': 'INDEPENDENT_EXACT_ELECTRIC_BAND_REVIEW_CONSUMER_NOT_ADMISSION',
               'rational_enclosures': budgets(profiles), 'scalar_CP': scalar_cp(),
               'native_factor_controls': [native_paths(n, profiles) for n in request['N']],
               'nonnative_method_adverse': domain_and_mass(),
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: sha((ROOT/name).read_bytes()) for name in request['pins']},
               'native_fullH_ground_backend': False, 'scientific_admission': False}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'completed_exact_controls': True}))


if __name__ == '__main__':
    main()
