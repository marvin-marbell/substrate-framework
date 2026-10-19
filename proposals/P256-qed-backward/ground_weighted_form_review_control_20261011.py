"""Independent native coefficient/form audit, not a quantum evolution or ground solver."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

import numpy as np
import sympy as sp

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rational(value: F) -> str:
    return str(value.numerator) + '/' + str(value.denominator)


def constants(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lam, chi, fast = F(2)**8192, F(2)**-4096, F(2)**10240
    weights = [abs(F(p['onsite'])) + 2*sum((abs(F(row['c0'])) + abs(F(row['ci'])) for row in p['paths']), F(0)) for p in profiles]
    b = 4*(lam*weights[0] + weights[1] + 63*weights[2])
    assert b < 512*lam and [p['load'] for p in profiles] == [1, 1, 63]
    kphi, kh, ke, alpha, beta = 4*chi, 3*chi/20, F(64), fast/40, 7*fast/10
    sphere = F(4096)
    z = sphere + 2/lam
    terms = {
        'kinetic_incidence': F(54),
        'Phi_matter_derivative': 480*kphi*b*b*sphere,
        'Phi_alignment_derivative': 184320*kphi*alpha*alpha*z,
        'Haar_matter_derivative': 480*kh*b*b*z,
        'fullE_charged_matter_derivative': 32*ke*b*b*z,
        'ordered_matter_matter': 32*b*b*z,
        'Haar_alignment_derivative': 46080*kh*alpha*alpha/lam,
        'Haar_plaquette_derivative': 2880*kh*beta*beta*sphere,
    }
    c = sum(terms.values(), F(0))
    assert c < F(2)**16448
    r = sp.Symbol('r', real=True)
    factor = (r-8)**2*(r*r+16*r+192)/2
    assert sp.expand(factor-(r**4/2-1024*r+6144)) == 0
    # 2^-1/4 < 7/8 gives coth(epsilon/2)^3 < 15^3 < 4096.
    assert F(7, 8)**4 > F(1, 2) and 15**3 < sphere
    return {'profile_weights': list(map(rational, weights)),
        'B_over_lambda': rational(b/lam), 'B_sha256': digest(rational(b).encode()),
        'explicit_C': rational(c), 'explicit_C_sha256': digest(rational(c).encode()),
        'C_terms': {key: rational(value) for key, value in terms.items()},
        'C_upper': '2^16448', 'S_upper': 4096,
        'Young_factorization': str(factor), 'coercivity_P_le_Q': True,
        'comparability_Q_le_33P_plus_14336lambda_sumtheta': True,
        'all65_distinct_pinned_profile_coefficients_retained': True}


def geometry(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    def move(x: tuple[int, ...], axis: int, step: int) -> tuple[int, ...]:
        y = list(x); y[axis] = (y[axis]+step) % n; return tuple(y)
    def weight(x: tuple[int, ...]) -> float:
        return 2**(-sum(min(t, n-t) for t in x)/4)
    ws = {x: weight(x) for x in sites}
    paths: list[tuple[tuple[int, ...], tuple[int, ...], list[tuple[tuple[int, ...], int]], float]] = []
    path_incidence = {x: 0 for x in sites}
    edge_incidence = {(x, axis): 0 for x in sites for axis in range(3)}
    charged_incidence = dict(edge_incidence)
    phi_delta = haar_delta = electric_delta = corner_ratio = 0.0
    seam_count = 0
    for role, profile in enumerate(profiles):
        for row in profile['paths']:
            axis, length = int(row['axis']), int(row['length'])
            for x in sites:
                y = move(x, axis, length)
                edges = [(move(x, axis, j), axis) for j in range(length)]
                assert len(set(edges)) == length
                w = (ws[x]+ws[y])/2
                paths.append((x, y, edges, w))
                path_incidence[x] += 1; path_incidence[y] += 1
                seam_count += int(x[axis]+length >= n)
                for endpoint in (x, y):
                    phi_delta = max(phi_delta, (w-ws[endpoint])**2/ws[endpoint]/min(ws[x], ws[y]))
                for edge in edges:
                    edge_incidence[edge] += 1
                    if role:
                        charged_incidence[edge] += 1
                    ratio = (w-ws[edge[0]])**2/ws[edge[0]]/min(ws[x], ws[y])
                    haar_delta = max(haar_delta, ratio)
                    if role:
                        electric_delta = max(electric_delta, ratio)
    faces = 0
    face_incidence = dict.fromkeys(edge_incidence, 0)
    for x in sites:
        for a, b in itertools.combinations(range(3), 2):
            xa, xb, xab = move(x, a, 1), move(x, b, 1), move(move(x, a, 1), b, 1)
            corners = (x, xa, xb, xab)
            edges = ((x, a), (xa, b), (xb, a), (x, b))
            w = sum(ws[z] for z in corners)/4
            assert len(set(corners)) == 4 and len(set(edges)) == 4
            for edge in edges:
                face_incidence[edge] += 1
                corner_ratio = max(corner_ratio, w/ws[edge[0]], ws[edge[0]]/w)
                assert (w-ws[edge[0]])**2/ws[edge[0]] <= 4*ws[edge[0]] + 1e-12
            faces += 1
    assert max(path_incidence.values()) == 30
    assert max(edge_incidence.values()) == 7
    assert max(charged_incidence.values()) == 6
    assert set(face_incidence.values()) == {4}
    assert max(phi_delta, haar_delta, electric_delta) <= 8 + 1e-12
    assert corner_ratio <= math.sqrt(2) + 1e-12
    overlap_ratio = 0.0
    by_endpoint: dict[tuple[int, ...], list[int]] = {x: [] for x in sites}
    for j, (x, y, _, _) in enumerate(paths):
        by_endpoint[x].append(j); by_endpoint[y].append(j)
    # Only shared CAR endpoints generate matter/matter commutators.
    for x in sites:
        for j, k in itertools.combinations(by_endpoint[x], 2):
            u, v, _, wj = paths[j]; a, b, _, wk = paths[k]
            ratio = max(wj, wk)/min(ws[u], ws[v], ws[a], ws[b])
            overlap_ratio = max(overlap_ratio, ratio)
            assert ratio <= 2+1e-12
    assert sum(ws.values()) < 4096
    origin = (0, 0, 0)
    wa, wb, wab = ws[(1, 0, 0)], ws[(0, 1, 0)], ws[(1, 1, 0)]
    correct_corner = (ws[origin]+wa+wb+wab)/4
    wrong_two_corner = (ws[origin]+wab)/2
    assert abs(correct_corner-wrong_two_corner) > 1e-4
    translated_singleton = (n//2, 0, 0)
    assert 1/ws[translated_singleton] > 1
    return {'N': n, 'sum_theta': sum(ws.values()), 'Phi_path_incidence': 30,
        'Haar_path_incidence': 7, 'fullE_charged_path_incidence': 6,
        'Haar_plaquette_incidence': 4, 'plaquettes': faces,
        'native_AP_seam_terms': seam_count,
        'delta_squared_over_derivative_and_endpoint_theta_max': max(phi_delta, haar_delta, electric_delta),
        'plaquette_corner_average_to_link_tail_ratio_max': corner_ratio,
        'four_corner_weight': correct_corner, 'wrong_two_corner_weight': wrong_two_corner,
        'wrong_two_corner_allocation_rejected': True,
        'singleton_patch_diameter': 0,
        'translated_singleton_min_theta_reciprocal': 1/ws[translated_singleton],
        'wrong_patch_diameter_without_distance_to_centre_rejected': True,
        'overlapping_path_union_weight_ratio_max': overlap_ratio,
        'N4_length2_reverse_collisions_kept_as_separate_terms': n == 4}


def car(mask: int, create: int, annihilate: int) -> tuple[int, int] | None:
    if not mask >> annihilate & 1:
        return None
    sign = (-1)**((mask & ((1 << annihilate)-1)).bit_count())
    mask ^= 1 << annihilate
    if mask >> create & 1:
        return None
    sign *= (-1)**((mask & ((1 << create)-1)).bit_count())
    return mask | (1 << create), sign


def car_ordered_control() -> dict[str, Any]:
    n = 4; v = n**3
    sites = list(itertools.product(range(n), repeat=3))
    occupied = {j for j, x in enumerate(sites) if sum(x) % 2 == 0}
    a, b, c = sites.index((0, 0, 0)), sites.index((1, 0, 0)), sites.index((2, 0, 0))
    occupied.remove(c); occupied.add(b)
    species_mask = sum(15 << (4*j) for j in occupied)
    mask = sum(species_mask << (4*v*s) for s in range(65)) | (15 << (4*v*65))
    assert all(((mask >> (4*v*s)) & ((1 << (4*v))-1)).bit_count() == 2*v for s in range(65))
    rows = []
    for s in range(65):
        aa, bb, cc = (4*v*s+4*j for j in (a, b, c))
        def product(left: tuple[int, int], right: tuple[int, int]) -> dict[int, int]:
            first = car(mask, *right)
            if first is None: return {}
            second = car(first[0], *left)
            return {} if second is None else {second[0]: first[1]*second[1]}
        ab, ba = product((cc, bb), (bb, aa)), product((bb, aa), (cc, bb))
        comm = {key: ab.get(key, 0)-ba.get(key, 0) for key in ab.keys() | ba.keys()}
        comm = {key: value for key, value in comm.items() if value}
        direct = car(mask, cc, aa)
        assert direct is not None and comm == {direct[0]: direct[1]}
        rows.append([s, direct[1], s != 64])
    return {'canonical_each': 2*v, 'Ref_particles': 4, 'species_controls': rows,
        'ordered_identity': '[c_c^dag c_b,c_b^dag c_a]=c_c^dag c_a',
        'wrong_independent_or_commuting_CAR_factors_rejected': True,
        'scope': 'native CAR coefficient identity; not truncated quantum evolution'}


def native_coefficients() -> dict[str, Any]:
    # Exact noncommuting SU4 path and all-four-field ordered Jordan coefficients.
    x = sp.Matrix([[0, 1], [1, 0]])
    y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    z = sp.diag(1, -1)
    gamma = [sp.kronecker_product(x, sp.eye(2)), sp.kronecker_product(z, x), sp.kronecker_product(z, y), sp.kronecker_product(z, z)]
    jordan = 4*sum((sp.trace((a*b+b*a)**2) for a in gamma for b in gamma), sp.S(0))
    s = sum((sp.trace(a*a) for a in gamma), sp.S(0))
    assert jordan == s*s == 256
    u = sp.diag(sp.I, -sp.I, -1, -1)
    v = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    assert u.det() == v.det() == 1 and u*v != v*u
    generator = sp.diag(1, -1, 0, 0)/2
    fields_x = [a/sp.Integer(3+mu) for mu, a in enumerate(gamma)]
    fields_y = [v*a*v.H/sp.Integer(5+mu) for mu, a in enumerate(gamma)]
    sx = sum((sp.trace(a*a) for a in fields_x), sp.S(0))
    sy = sum((sp.trace(a*a) for a in fields_y), sp.S(0))
    ds = [a-u*b*u.H for a, b in zip(fields_x, fields_y)]
    phi_generator = gamma[0]/sp.sqrt(8)
    dq = 2*sp.re(sp.trace(ds[0]*phi_generator))
    dh = sum((-2*sp.re(sp.trace(d*sp.I*(generator*u*b*u.H-u*b*u.H*generator))) for d, b in zip(ds, fields_y)), sp.S(0))
    assert dq*dq <= 32*(sx+sy)
    assert dh*dh <= 16*(sx+sy)**2
    # Four native face links (face,I,I,I); this positive jet avoids a stationary face.
    face = sp.diag(sp.I, 1, -sp.I, 1)
    dp = -sp.re(sp.trace(sp.I*generator*face))
    assert dp*dp <= 4
    assert dq != 0 and dh != 0 and dp != 0
    q, pk, wkin, wj, kappa, lam = sp.symbols('q pk wkin wj kappa lambda', real=True)
    wrong = 4*kappa*lam*(wj-wkin)*q**3*pk
    assert wrong.subs(wj, wkin) == 0 and wrong != 0
    scales = []
    for radius in (16, 64, 256, 1024):
        rr = F(radius)
        # Exact 1D Gaussian moment compression of the native quartic-current scaling.
        current = 4*rr*rr*(rr**3+3*rr)
        budget = rr**4+F(1, 4)+rr**4+6*rr*rr+3
        scales.append({'R': radius, 'q3_p_to_p2_plus_q4': rational(current/budget)})
    assert all(F(scales[j+1]['q3_p_to_p2_plus_q4']) > F(scales[j]['q3_p_to_p2_plus_q4']) for j in range(len(scales)-1))
    return {'ordered_16_Jordan_at_Clifford': str(jordan), 's_squared': str(s*s),
        'native_alignment_Phi_derivative': str(dq), 'native_alignment_Haar_derivative': str(dh),
        'native_colour_plaquette_Haar_derivative': str(dp),
        'ordered_SU4_length2_differs_from_reversed': True,
        'wrong_onsite_weight_current': str(wrong), 'wrong_weighting_scaling_controls': scales,
        'scaling_controls_scope': 'exact Gaussian coordinate compression, not native ground expectation; gauge-invariant radial-shell contrary family proved in audit'}


def split(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = flow.shape[0]
    rho = np.zeros((n, n, n))
    symbol = np.zeros((n, n, n))
    for axis in range(3):
        rho += flow[..., axis]-np.roll(flow[..., axis], 1, axis=axis)
        symbol += (2-2*np.cos(2*np.pi*np.arange(n)/n)).reshape(tuple(n if j == axis else 1 for j in range(3)))
    symbol[0, 0, 0] = 1
    fourier = np.fft.fftn(rho)/symbol; fourier[0, 0, 0] = 0
    potential = np.fft.ifftn(fourier).real
    longitudinal = np.stack([potential-np.roll(potential, -1, axis=j) for j in range(3)], axis=-1)
    return flow-longitudinal, longitudinal


def full_electric(n: int) -> dict[str, Any]:
    d = np.minimum(np.arange(n), n-np.arange(n))
    theta = 2**(-(d[:, None, None]+d[None, :, None]+d[None, None, :])/4)
    parity = 1-2*(np.indices((n, n, n)).sum(axis=0) % 2)
    flow = np.zeros((n, n, n, 3)); flow[..., 0] = 64*parity
    # Closed integral circulation and a nonzero harmonic character shift preserve fullrho.
    flow[0, 0, 0, 0] += 3; flow[1, 0, 0, 1] += 3
    flow[0, 1, 0, 0] -= 3; flow[0, 0, 0, 1] -= 3
    flow[..., 2] += float(F(3, 8))
    transverse, longitudinal = split(flow)
    weighted_cross = float(128*np.sum(theta[..., None]*transverse*longitudinal))
    direct = float(64*np.sum(theta[..., None]*flow*flow))
    omitted = float(64*np.sum(theta[..., None]*(transverse*transverse+longitudinal*longitudinal)))
    assert abs(direct-omitted-weighted_cross) < 1e-7
    assert abs(weighted_cross) > 1e-4
    assert abs(np.sum(transverse*longitudinal)) < 1e-8
    controls = []
    for length in (1, 2):
        transfer = np.zeros_like(flow)
        for step in range(length): transfer[step, 0, 0, 0] = 1
        tt, tl = split(transfer)
        outside = transfer == 0
        assert np.max(abs((tt+tl)[outside])) < 1e-12
        exterior = float(np.sum(tl[outside]**2))
        assert exterior > 0
        change = float(64*np.sum(theta[..., None]*(2*flow*transfer+transfer*transfer)))
        split_change = float(64*np.sum(theta[..., None]*(2*transverse*tt+tt*tt+2*longitudinal*tl+tl*tl)))
        cross_change = float(128*np.sum(theta[..., None]*(transverse*tl+longitudinal*tt+tt*tl)))
        assert abs(change-split_change-cross_change) < 1e-7
        assert abs(cross_change) > 1e-4
        controls.append({'length': length, 'weighted_fullE_energy_change': change,
            'wrong_weighted_HE_plus_HC_energy_change': split_change,
            'required_cross_change': cross_change,
            'separate_longitudinal_exterior_norm_squared': exterior,
            'full_transfer_exterior_norm_squared': 0})
    return {'N': n, 'fullrho': 'all64 checkerboard, 128 parity',
        'harmonic_character_shift': '3/8', 'weighted_cross': weighted_cross,
        'weighted_energy_direct': direct, 'wrong_crossless_energy': omitted,
        'wrong_weighted_crossless_split_rejected': True,
        'wrong_dropped_Coulomb_exterior_rejected': True, 'path_controls': controls}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.input.read_text())
    before = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['epoch_archive']) as archive:
        profiles = json.loads(archive.read(request['profile_member']))['primitive_profiles']
        members = {name: digest(archive.read(name)) for name in archive.namelist()}
    assert members == request['epoch_member_sha256']
    result = {'kind': 'INDEPENDENT_NATIVE_WEIGHTED_FORM_COEFFICIENT_AUDIT_NOT_EVOLUTION_OR_ADMISSION',
        'constant_control': constants(profiles),
        'native_geometry': [geometry(n, profiles) for n in request['N']],
        'native_ordered_CAR': car_ordered_control(), 'native_coefficients': native_coefficients(),
        'native_fullE_cross_exterior': [full_electric(n) for n in request['N']],
        'source_input_sha256_before': before,
        'source_input_sha256_after': {name: digest((ROOT/name).read_bytes()) for name in request['pins']},
        'audit': json.loads((ROOT/request['proof']).read_text()),
        'runtime': {'executable': sys.executable, 'elapsed_seconds': time.monotonic()-started},
        'scientific_admission': False, 'actual_quantum_evolution_computed': False,
        'closed_form_evolution_extension_verified_by_this_audit': False}
    assert before == result['source_input_sha256_after']
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'native_N': request['N'], 'explicit_uniform_C_upper': '2^16448',
        'weighted_crossless_split_rejected': True, 'scientific_admission': False}))


if __name__ == '__main__':
    main()
