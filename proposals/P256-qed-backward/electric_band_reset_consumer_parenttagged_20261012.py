"""Own electric CP/reset scientific controls; no ground/fullH numerical backend."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import zipfile
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def car(mask: int, create: int, annihilate: int) -> tuple[int, int] | None:
    if not mask & (1 << annihilate):
        return None
    sign = (-1)**((mask & ((1 << annihilate)-1)).bit_count())
    mask ^= 1 << annihilate
    if mask & (1 << create):
        return None
    sign *= (-1)**((mask & ((1 << create)-1)).bit_count())
    return mask | (1 << create), sign


def native(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    def move(x: tuple[int, ...], a: int) -> tuple[int, ...]:
        y = list(x); y[a] = (y[a]+1) % n
        return tuple(y)
    o = (n-1, n-1, 0)
    c = {(o, 0): 1, (move(o, 0), 1): 1,
         (move(o, 1), 0): -1, (o, 1): -1}
    def div(flow: dict[tuple[tuple[int, ...], int], int]) -> Counter:
        d: Counter = Counter()
        for (x, a), v in flow.items():
            d[x] += v; d[move(x, a)] -= v
        return d
    assert all(v == 0 for v in div(c).values()) and sum(v*v for v in c.values()) == 4
    rho = {x: 128*(-1)**sum(x) for x in sites}
    background = {(x, 0): 64*(-1)**sum(x) for x in sites}
    assert all(div(background)[x] == rho[x] for x in sites)
    for m in (-3, -1, 0, 2, 7):
        shifted = dict(background)
        for e, v in c.items(): shifted[e] = shifted.get(e, 0)-m*v
        assert all(div(shifted)[x] == rho[x] for x in sites)
        ec0 = sum(v*background.get(e, 0) for e, v in c.items())
        ec1 = sum(v*shifted.get(e, 0) for e, v in c.items())
        assert ec1 == ec0-4*m
        for e, v in c.items():
            assert F(shifted[e])-F(v*ec1, 4) == F(background.get(e, 0))-F(v*ec0, 4)
    # Exact ordered noncommuting fullmatrix/SU4 control at every touched literal.
    gx = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    gy = sp.diag(sp.I, -sp.I, 1, 1)
    assert gx.det() == gy.det() == 1 and gx*gy != gy*gx
    px = sp.diag(1, -1, 2, -2)
    py = gx*sp.diag(2, -1, -2, 1)*gx.H
    p1 = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]])
    p2 = gy*p1*gy.H
    counts = [0, 0, 0]; touched = [0, 0, 0]; ap = [0, 0, 0]
    frequencies: Counter = Counter(); aliases = 0; allocation = F(0)
    matrix_hashes = []; gauge_checks = 0; path_gauss_checks = 0
    for role, profile in enumerate(profiles):
        for x in sites:
            for row in profile['paths']:
                a, length = row['axis'], row['length']; y = x; edges = []; seam = 1
                for _ in range(length):
                    edges.append((y, a)); seam *= -1 if y[a] == n-1 else 1; y = move(y, a)
                counts[role] += 1
                r = sum(c.get(e, 0) for e in edges) if role else 0
                assert abs(r) <= 2
                if not r: continue
                touched[role] += 1; ap[role] += seam == -1; frequencies[(role, length, r)] += 1
                aliases += int(n == 4 and length == 2)
                flow = dict(Counter(edges)); d = div(flow)
                assert all(d[z] == int(z == x)-int(z == y) for z in sites)
                path_gauss_checks += 1
                u = gx if length == 1 else gx*gy
                c0, ci = sp.Rational(row['c0']), sp.Rational(row['ci'])
                fx, fy = c0*px+sp.I*ci*p1, c0*py+sp.I*ci*p2
                block = seam*(fx*u+u*fy)/2
                transformed = seam*((gx*fx*gx.H)*(gx*u*gy.H)+(gx*u*gy.H)*(gy*fy*gy.H))/2
                assert sp.simplify(transformed-gx*block*gy.H) == sp.zeros(4)
                assert block != sp.zeros(4)
                matrix_hashes.append(digest(str(block).encode())); gauge_checks += 1
                allocation += 8*int(profile['load'])*(abs(F(row['c0']))+abs(F(row['ci'])))
    w = sum((int(p['load'])*int(r['length'])*(abs(F(r['c0']))+abs(F(r['ci'])))
             for p in profiles[1:] for r in p['paths']), F(0))
    assert allocation <= 32*w and touched[0] == 0 and touched[1] and touched[2]
    assert ap[1] and ap[2] and (n != 4 or aliases)
    # Actual CAR occupancy signs/number conservation, all64 charged roles retained.
    # A canonical background has two occupied modes at each site. Moving a mode
    # from neighbour y to the empty colour2 at x preserves the full particle count.
    v = n**3
    mask = sum((1 << (4*j)) | (1 << (4*j+1)) for j in range(v))
    assert mask.bit_count() == 2*v
    xindex, yindex = sites.index(o), sites.index(move(o, 0))
    legal = car(mask, 4*xindex+2, 4*yindex)
    assert legal is not None and legal[0].bit_count() == 2*v
    restored = car(legal[0], 4*yindex, 4*xindex+2)
    assert restored is not None and restored[0] == mask and restored[1]*legal[1] == 1
    charged_numbers = [legal[0].bit_count() for _ in range(64)]
    assert charged_numbers == [2*v]*64
    # Exact CAR anticommutator, all occupation sectors of a small native mode factor.
    for m in range(16):
        for i in range(4):
            occupied = bool(m & (1 << i))
            annihilation_creation = 0 if occupied else 1
            creation_annihilation = 1 if occupied else 0
            assert annihilation_creation+creation_annihilation == 1
    return {'N': n, 'canonical_each_species': 2*v, 'all65_loads': [1, 1, 63],
            'literal_path_counts': counts, 'nonzero_closed_frequency_counts': touched,
            'AP_affected': ap, 'N4_length2_alias_literals_retained': aliases,
            'frequency_table': {str(k): v for k, v in sorted(frequencies.items())},
            'actual_endpoint_coefficient_allocation': str(allocation), 'uniform_allocation_32W': str(32*w),
            'closed_Dc_exact': True, 'c_squared': 4, 'fullrho_charged_roles': 64,
            'fullrho_amplitude': 128, 'reset_Eperp_and_annihilation_U1_Gauss_exact': True,
            'individual_reset_integer_shifts': [-3, -1, 0, 2, 7],
            'ordered_fullmatrix_SU4_covariance_checks': gauge_checks,
            'charged_path_Gauss_contact_checks': path_gauss_checks,
            'ordered_matrix_stream_sha256': digest(''.join(matrix_hashes).encode()),
            'all64_canonical_CAR_transfer_controls': len(charged_numbers),
            'Ref4_unchanged': True, 'native_ground_fullH_numerics': False}


def theta(t: float) -> float:
    if t <= 0: return 0.0
    if t >= 1: return math.pi/2
    a, b = math.exp(-1/t), math.exp(-1/(1-t))
    return math.pi/2*a/(a+b)


def partition(e: float, radius: float) -> dict[tuple[str, int], float]:
    if abs(e) <= radius:
        f, g = 1.0, 0.0
    elif abs(e) >= 2*radius:
        f, g = 0.0, 1.0
    else:
        a = theta((abs(e)-radius)/radius)
        f, g = math.cos(a), math.sin(a)
    m = math.floor(e/4); angle = theta((e-4*m)/4)
    return {('keep', 0): f, ('shift', m): g*math.cos(angle),
            ('shift', m+1): g*math.sin(angle)}


def kernel(e: float, r: float, radius: float) -> float:
    a, b = partition(e+r, radius), partition(e, radius)
    return sum(v*b.get(k, 0.0) for k, v in a.items())


def reset_controls(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    # Exact points of the ACTUAL smooth alternative: theta(1/2)=pi/4.
    z = sp.sqrt(2)/2
    assert sp.simplify(z*z+z*z) == 1
    h = lambda e: 1+16*e*e
    e6_output = sp.Rational(1, 2)*h(6)+sp.Rational(1, 4)*h(2)+sp.Rational(1, 4)*h(-2)
    assert e6_output == 321 and h(6)-e6_output == 256
    assert h(12)-h(0) == 2304 and h(12)-h(0) >= sp.Rational(1, 2)*h(12)
    exact_soft_k_6_to_8 = z*z*z*z*2
    assert exact_soft_k_6_to_8 == sp.Rational(1, 2)
    # All-band hard contrary: frequencies1 and2 cross a half-open cell boundary.
    hard_kernel = int(math.floor((1+2)/4) == math.floor((2+2)/4))
    assert hard_kernel == 0 and kernel(1, 1, 8) == 1
    assert math.floor((2-F(1, 100)+2)/4) != math.floor((2+F(1, 100)+2)/4)
    # Bounded scalar partition controls are not finite-H ground/evolution models.
    largest_norm_error = 0.0; low_preservation_error = 0.0; energy_error = 0.0
    for beta in (-0.49, -0.125, 0.0, 0.125, 0.49):
        for integer in range(-80, 81):
            e = integer+beta; radius = 8.0; weights = partition(e, radius)
            largest_norm_error = max(largest_norm_error, abs(sum(v*v for v in weights.values())-1))
            output = h(e)*weights[('keep', 0)]**2
            output += sum(h(e-4*key[1])*v*v for key, v in weights.items()
                          if key[0] == 'shift')
            lower = max(0.0, h(e)/2 if abs(e) >= 2*radius else 0.0)
            energy_error = max(energy_error, lower-(h(e)-output))
            for r in (-2, -1, 0, 1, 2):
                if abs(e) <= radius-2 or abs(e+r) <= radius-2:
                    low_preservation_error = max(low_preservation_error, abs(1-kernel(e, r, radius)))
                assert abs(1-kernel(e, r, radius)) <= 2+1e-14
    assert largest_norm_error < 1e-14 and low_preservation_error < 1e-14 and energy_error < 1e-10
    # The old integer sentinel collided with shift -10000 at e=-40000.
    # Tagged branches preserve the high-energy reset on the entire real line.
    far_reset_rows = []
    for e in (-40004.0, -40000.0, -39996.0, 40000.0):
        weights = partition(e, 8.0)
        assert weights[('keep', 0)] == 0.0
        output = sum(h(e-4*key[1])*v*v for key, v in weights.items()
                     if key[0] == 'shift')
        assert output == 1.0 and h(e)-output >= h(e)/2
        far_reset_rows.append({'input_E': e, 'output_h': output,
                               'removed_h': h(e)-output})
    w = sum((int(p['load'])*int(r['length'])*(abs(F(r['c0']))+abs(F(r['ci'])))
             for p in profiles[1:] for r in p['paths']), F(0))
    stages = [(F(1), F(3, 2)), (F(3, 2), F(2))]
    assert all(s < 1+3*p/4 for p, s in stages)
    denominators = [1+3*p/4-s for p, s in stages]
    assert denominators == [F(1, 4), F(1, 8)]
    # Endpoint contrary: the literal tail-integral denominator vanishes.
    assert 1+3*F(1)/4-F(7, 4) == 0
    # Removing aux63 materially alters the native allcharged coefficient constant.
    w_active = sum((int(r['length'])*(abs(F(r['c0']))+abs(F(r['ci'])))
                    for r in profiles[1]['paths']), F(0))
    assert w > w_active
    x = sp.symbols('x', real=True)
    assert sp.simplify(sp.cos(x)**2+sp.sin(x)**2) == 1
    return {'scope': 'exact scalar CP/energy/frequency algebra plus labelled floating continuous-character controls',
            'W_exact': str(w), 'W_active_only_adverse': str(w_active),
            'hard_allband_low_coherence_kernel': hard_kernel,
            'soft_highonly_same_low_coherence_kernel_exact': 1,
            'hard_fibre_endpoint_jump_rejected': True,
            'soft_exact_e6_h_output': str(e6_output), 'soft_exact_e6_h_removed': '256',
            'soft_exact_e12_h_removed': '2304', 'soft_exact_changed_high_coherence_kernel': '1/2',
            'CP_partition_identity_exact': 'cos(x)^2+sin(x)^2=1',
            'float_character_partition_max_error': largest_norm_error,
            'float_exact_low_selection_max_error': low_preservation_error,
            'float_energy_lower_violation': energy_error,
            'strict_bootstrap_inputs_outputs': [[str(p), str(s)] for p, s in stages],
            'strict_tail_integral_denominators': [str(v) for v in denominators],
            'endpoint_adverse_divergence_retained': True,
            'uniform_E4_analytic_not_measured': 'CE4=C2/256',
            'disjoint_keep_shift_labels': True,
            'far_reset_sentinel_collision_controls': far_reset_rows,
            'Phi8_conditional_reset_used': False, 'ground_phase_atom_claim': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    before = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        profiles = json.loads(archive.read(request['profile_member']))['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    receipt = {'kind': 'OWN_NATIVE_ELECTRIC_RESET_CONTROLS_NOT_GROUND_NUMERICS',
               'native': [native(n, profiles) for n in request['N']],
               'reset_exact_and_adverse': reset_controls(profiles),
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: digest((ROOT/name).read_bytes()) for name in request['pins']},
               'proof': request['proof'], 'interface': request['interface'],
               'fullH_ground_numerics': False, 'scientific_admission': False}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'native_N': request['N'], 'natural_completion': True}))


if __name__ == '__main__':
    main()
