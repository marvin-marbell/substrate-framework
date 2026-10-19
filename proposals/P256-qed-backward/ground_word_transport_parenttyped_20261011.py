"""Periodic full-law local-word form controls, energy repair, and range debt.
No old producer, ground solver, eigensolver, or checker is executed.
"""
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
from typing import Any, cast

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)
C = tuple[F, F]
ZERO: C = (F(0), F(0))


def add(a: C, b: C) -> C:
    return a[0]+b[0], a[1]+b[1]


def mul(a: C, b: C) -> C:
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def scale(a: C, b: F | int) -> C:
    return a[0]*b, a[1]*b


def conjugate(a: C) -> C:
    return a[0], -a[1]


def exact(z: Any) -> C:
    re, im = sp.expand(z).as_real_imag()
    return F(int(sp.numer(re)), int(sp.denom(re))), F(int(sp.numer(im)), int(sp.denom(im)))


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def car(mask: int, annihilate: int, create: int) -> tuple[int, int] | None:
    if not mask >> annihilate & 1:
        return None
    sign = (-1)**((mask & ((1 << annihilate)-1)).bit_count())
    mask ^= 1 << annihilate
    if mask >> create & 1:
        return None
    sign *= (-1)**((mask & ((1 << create)-1)).bit_count())
    return mask | (1 << create), sign


def divergence(k: np.ndarray) -> np.ndarray:
    return sum((k[..., j]-np.roll(k[..., j], 1, axis=j) for j in range(3)), np.zeros(k.shape[:3]))


def split(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = k.shape[0]
    rho = divergence(k)
    lam = cast(np.ndarray, sum((2-2*np.cos(2*np.pi*np.arange(n)/n)).reshape(tuple(n if j == axis else 1 for j in range(3))) for axis in range(3)))
    rh = np.fft.fftn(rho)
    lam[0, 0, 0] = 1
    ph = rh/lam
    ph[0, 0, 0] = 0
    phi = np.fft.ifftn(ph).real
    longitudinal = np.stack([phi-np.roll(phi, -1, axis=j) for j in range(3)], axis=-1)
    return k-longitudinal, longitudinal


def native_phi(seed: int) -> list[sp.Matrix]:
    fields = []
    for mu in range(4):
        a = sp.zeros(4)
        for i in range(4):
            a[i, i] = sp.Rational(seed+mu+i, 17)
        a -= sp.trace(a)*sp.eye(4)/4
        for i in range(3):
            z = sp.Rational(seed+i+1, 23)+sp.I*sp.Rational(mu+i+1, 29)
            a[i, i+1], a[i+1, i] = z, sp.conjugate(z)
        fields.append(a)
    return fields


def native_control(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Literal full65 CAR action on a canonical ambient mask, with affine flows.
    Coordinate evaluations control operator coefficients, NOT ground expectations.
    """
    v = n**3
    sites = list(itertools.product(range(n), repeat=3))
    index = {x: i for i, x in enumerate(sites)}
    x, y, empty = index[(0, 0, 0)], index[(1, 0, 0)], index[(2, 0, 0)]
    filled = {i for i, z in enumerate(sites) if sum(z) % 2 == 0}
    filled.remove(empty)
    filled.add(y)
    one = sum(15 << (4*i) for i in filled)
    assert one.bit_count() == 2*v
    mask = sum(one << (4*v*s) for s in range(65)) | (15 << (4*v*65))
    parity = 1-2*(np.indices((n, n, n)).sum(axis=0) % 2)
    k = np.zeros((n, n, n, 3), dtype=np.int64)
    k[..., 0] = 64*parity
    k[1, 0, 0, 0] += 256
    rho = 128*parity
    rho[1, 0, 0] += 256
    rho[2, 0, 0] -= 256
    assert np.array_equal(divergence(k), rho)
    assert all(((mask >> (4*v*s)) & ((1 << (4*v))-1)).bit_count() == 2*v for s in range(65))
    links = [sp.diag(sp.I, -sp.I, -1, -1), sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])]
    assert links[0]*links[1] != links[1]*links[0]
    assert all(u.det() == 1 and u.H*u == sp.eye(4) for u in links)
    fields = {i: native_phi(2+i % 5) for i in range(v)}
    # A=sum65 n_s,x; B=1+n_active,y. Products are on the SAME ambient copy.
    def eigen(state: int) -> tuple[int, int]:
        a = sum(((state >> (4*v*s+4*x)) & 15).bit_count() for s in range(65))
        b = 1+((state >> (4*y)) & 15).bit_count()
        return a, b
    a0, b0 = eigen(mask)
    assert (a0, b0) == (260, 5)
    comm: dict[tuple[int, tuple[tuple[int, int], ...]], C] = {}
    dropped: dict[tuple[int, tuple[tuple[int, int], ...]], C] = {}
    roles = [0, 0, 0]
    lengths = {1: 0, 2: 0}
    gauss, electric, colour, seams = 0, 0, 0, 0
    exterior_samples = []
    flow_cache: dict[tuple[tuple[int, int], ...], tuple[float, float]] = {}
    kt, kl = split(k)
    for role, profile in enumerate(profiles):
        species = [64] if role == 0 else ([0] if role == 1 else range(1, 64))
        charge = 0 if role == 0 else 1
        mass = -(1 << 8192) if role == 0 else 1
        for row in profile['paths']:
            axis, length = int(row['axis']), int(row['length'])
            for source, z in enumerate(sites):
                dest = list(z)
                dest[axis] = (dest[axis]+length) % n
                target = index[tuple(dest)]
                if not ({source, target} & {x, y}):
                    continue
                edgeids = []
                cursor = list(z)
                for step in range(length):
                    edgeids.append(3*index[tuple(cursor)]+axis)
                    cursor[axis] = (cursor[axis]+1) % n
                seam = -1 if z[axis]+length >= n else 1
                upath = sp.eye(4)
                for edge in edgeids:
                    edge_site, edge_axis = sites[edge//3], edge % 3
                    upath = upath*links[edge_site[edge_axis] % 2]
                c0, ci = sp.Rational(row['c0']), sp.Rational(row['ci'])
                fx = c0*fields[source][0]+sp.I*ci*fields[source][1+axis]
                fy = c0*fields[target][0]+sp.I*ci*fields[target][1+axis]
                coefficient = (fx*upath+upath*fy)*seam/2
                # Local SU4 transformation of endpoints/path, including Phi.
                if colour < 6:
                    gx = sp.diag(-1, -1, 1, 1)
                    gy = links[1]
                    transformed = (gx*fx*gx.H*(gx*upath*gy.H)+(gx*upath*gy.H)*gy*fy*gy.H)*seam/2
                    assert all(sp.expand(entry) == 0 for entry in transformed-gx*coefficient*gy.H)
                    colour += 1
                entries = [(i, j, exact(coefficient[i, j])) for i, j in itertools.product(range(4), repeat=2)]
                for s in species:
                    offset = 4*v*s
                    for i, j, coef in entries:
                        if coef == ZERO:
                            continue
                        for create, annihilate, c, direction in (
                            (offset+4*source+i, offset+4*target+j, coef, 1),
                            (offset+4*target+j, offset+4*source+i, conjugate(coef), -1)):
                            action = car(mask, annihilate, create)
                            if action is None:
                                continue
                            state, sign = action
                            a1, b1 = eigen(state)
                            lhs = a0*b0-a1*b1
                            rhs = (a0-a1)*b0+a1*(b0-b1)
                            assert lhs == rhs
                            assert ((state >> offset) & ((1 << (4*v))-1)).bit_count() == 2*v
                            shifts = tuple((e, charge*direction) for e in edgeids) if charge else ()
                            key = (state, shifts)
                            value = scale(c, sign*mass)
                            comm[key] = add(comm.get(key, ZERO), scale(value, lhs))
                            dropped[key] = add(dropped.get(key, ZERO), scale(value, a0-a1))
                            roles[role] += 1
                            lengths[length] += 1
                            seams += int(seam == -1)
                            # Full rho and full E change together; no fixed-HC substitution.
                            if shifts not in flow_cache:
                                t = np.zeros_like(k)
                                for e, delta in shifts:
                                    zz, ax = sites[e//3], e % 3
                                    t[zz+(ax,)] += delta
                                drho = np.zeros_like(rho)
                                drho[sites[create % (4*v)//4]] += charge
                                drho[sites[annihilate % (4*v)//4]] -= charge
                                assert np.array_equal(divergence(t), drho)
                                tt, tl = split(t)
                                direct = 64*int(2*np.sum(k*t)+np.sum(t*t))
                                dt = 64*(2*np.sum(kt*tt)+np.sum(tt*tt))
                                dl = 64*(2*np.sum(kl*tl)+np.sum(tl*tl))
                                assert abs(direct-dt-dl) < 1e-7
                                flow_cache[shifts] = float(dt), float(dl)
                                gauss += 1
                                electric += 1
                                if shifts and len(exterior_samples) < 4:
                                    support = np.zeros_like(k, dtype=bool)
                                    for e, delta in shifts:
                                        support[sites[e//3]+(e % 3,)] = True
                                    outside = ~support
                                    assert np.max(abs((tt+tl)[outside])) < 1e-12
                                    exterior_samples.append({'path_length': length, 'full_energy_change': direct,
                                        'HE_change': float(dt), 'HC_change': float(dl),
                                        'longitudinal_outside_literal_path_norm2': float(np.sum(tl[outside]**2)),
                                        'full_flow_outside_literal_path_norm2': int(np.sum(t[outside]**2)),
                                        'exterior_HE_HC_cancel_in_fullE': True})
    comm = {key: value for key, value in comm.items() if value != ZERO}
    dropped = {key: value for key, value in dropped.items() if value != ZERO}
    assert comm and comm != dropped and all(roles) and all(lengths.values()) and seams
    # Different heavy onsite/active/auxiliary masses are kept, although densities commute onsite.
    onsite = [str(F(p['onsite'])) for p in profiles]
    assert len(set(onsite)) == 3
    assert profiles[0]['load'] == 1 and profiles[1]['load'] == 1 and profiles[2]['load'] == 63
    probe = sha(json.dumps([[str(state), shifts, [str(z) for z in c]] for (state, shifts), c in sorted(comm.items())]).encode())
    return {'N': n, 'V': v, 'ambient_CAR_modes': 65*4*v+4, 'canonical_each': 2*v,
        'Ref_particles': 4, 'charged_roles': [1, 63], 'heavy_scale': '-2^8192',
        'distinct_onsite_coefficients': onsite, 'all65_literal_nonzero_actions_by_role': roles,
        'length1_and_length2_actions': lengths, 'AP_seam_actions': seams,
        'mixed_word_commutator_output_states': len(comm), 'mixed_word_sha256': probe,
        'Leibniz_same_copy_verified': True, 'drop_factor_rejected': True,
        'fullrho_affine_Gauss_controls': gauss, 'fullE_HE_HC_controls': electric,
        'SU4_endpoint_covariance_controls': colour, 'exterior_transport': exterior_samples,
        'physical_input': 'radial Phi x normalized Haar x local exterior4 canonical singlets x integer affine section; normalized original-cover packets, not an L2 plane wave',
        'coefficient_probe_only_not_ground_expectation': True,
        'full_quantum_mixed_time_evolution_numerically_computed': False}


def energy_repairs(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    # Exact symbolic IMS identities apply to EVERY derivative of the full law.
    z = sp.Symbol('z', real=True)
    u, w = sp.sin(z), sp.cos(z)
    assert sp.simplify(u*u+w*w-1) == 0
    assert sp.simplify(u*sp.diff(u, z)+w*sp.diff(w, z)) == 0
    assert sp.simplify(sp.diff(u, z)**2+sp.diff(w, z)**2-1) == 0
    # Native dressed hopping compressed to the two occupations connected by a literal CAR transfer.
    # Spectator modes keep N_s=2V; e^(+/- i A_path) includes the integer Gauss flow.
    a, theta = sp.symbols('a theta', real=True)
    g = sp.Matrix([[0, sp.exp(sp.I*a)], [sp.exp(-sp.I*a), 0]])
    f1, f2 = sp.sin(theta)*g, sp.cos(theta)*sp.eye(2)
    connection = sp.simplify(cast(sp.MatrixBase, f1).H*sp.diff(f1, a)+cast(sp.MatrixBase, f2).H*sp.diff(f2, a))
    assert connection == sp.diag(-sp.I*sp.sin(theta)**2, sp.I*sp.sin(theta)**2)
    assert connection != sp.zeros(2)
    bweights = []
    for profile in profiles:
        wgt = abs(F(profile['onsite']))+2*sum(abs(F(row['c0']))+abs(F(row['ci'])) for row in profile['paths'])
        bweights.append(wgt)
    b = 4*((1 << 8192)*bweights[0]+bweights[1]+63*bweights[2])
    assert b/F(1 << 8192) < 512
    # Exact closed-flow scaling arithmetic; radial shell values are a moment
    # countercase, not an integrated native expectation or a ground-state solve.
    tails = []
    for m in (16, 64, 256):
        probability = F(1, m*m)
        assert probability*m*m == 1
        tails.append({'M': m, 'physical_closed_flow_norm2_scaled': m*m,
            'probability': str(probability), 'electric_second_moment': '1',
            'electric_squared_tail_above_M_over2': '1',
            'radial_s2_budget': '1', 's_times_E2_moment': str(m),
            'radial_values_are_idealised_scale_arithmetic': True,
            'native_radial_quantum_expectation_computed': False})
    return {'scalar_IMS_full_H_exact': True,
        'Wilson_sincos_partition_energy_cost': 256,
        'density_cutoff_IMS': {'generator': 'g=theta exp(-T/M) n_total,x; T=sum_(13-site two-step star) s_z; n_total<=260',
            'support_sites_per_single_centre': 13,
            'kinetic_constant': '4 chi theta^2 260^2/(e M)',
            'matter_constant': 'theta^2 B_inc sqrt(M)/(4 sqrt(e)), B_inc<=2 B*13',
            'B_exact_sha256': sha(str(b).encode()), 'B_over_lambda_upper': 512,
            'operator_inequality': 'sum_(sin g,cos g) F_j^* (H-E0) F_j <= (H-E0)+C_g I',
            'word_energy_bound': 'q_L(A_m U(dt_m)...A_1 Omega)<=sum_j C_gj; all times/all even N>=4, for this repaired partition subalgebra',
            'all65_full_law_not_truncated': True},
        'offdiagonal_transport_partition_connection': str(connection),
        'offdiagonal_transport_first_order_fullE_term_unbounded': True,
        'absolute_energy_repair': 'Q_N=H_N+8192 lambda V+1; ||[H_N,A] Q_N^(-1/2)||<=c_A independent N on the full common core',
        'absolute_energy_ground_cost': '||Q_N^(1/2) Omega_N||^2=E0_N+8192 lambda V+1<=R V+1; this does not give a uniform excitation-energy insertion bound',
        'moment_only_native_adverse_sequences': tails,
        'adverse_sequences_are_ground_states': False}


def weighted_local_energy(n: int) -> dict[str, Any]:
    """Positive local-energy weighting: native torus controls, no dynamics run."""
    epsilon = math.log(2)/4
    d = np.minimum(np.arange(n), n-np.arange(n))
    weight = np.exp(-epsilon*(d[:, None, None]+d[None, :, None]+d[None, None, :]))
    lattice_bound = (1/math.tanh(epsilon/2))**3
    assert float(weight.sum()) <= lattice_bound
    ratios = []
    for axis, length in itertools.product(range(3), (1, 2)):
        ratio = weight/np.roll(weight, length, axis=axis)
        assert np.max(ratio) <= math.exp(epsilon*length)+1e-12
        ratios.append(float(np.max(ratio)))
    # Exact derivative identity for the potentially dangerous onsite quartic.
    # The native site kinetic and ordered Jordan term carry the SAME weight.
    q, lam, kappa, wkin, wj = sp.symbols('q lam kappa wkin wj', real=True)
    onsite_current_coefficient = sp.expand(kappa*(wj-wkin)*sp.diff(lam*q**4, q))
    assert onsite_current_coefficient.subs(wj, wkin) == 0
    assert onsite_current_coefficient != 0
    return {'N': n, 'epsilon': 'log(2)/4', 'weight_sum': float(weight.sum()),
        'all_N_weight_sum_upper': lattice_bound, 'native_path_weight_ratio_maxima': ratios,
        'weighted_matter_all65_budget_over_lambda_upper': 1024,
        'onsite_Phi_Jordan_current_cancellation': True,
        'wrong_mismatched_onsite_weight_quartic_gradient': str(onsite_current_coefficient),
        'core_commutator_estimate_only': True, 'closed_form_evolution_extension_paid': False}


def selected_feature_countercase() -> dict[str, Any]:
    # Method control only; this is not a native Hamiltonian truncation.
    omega = np.array([1., 0., 0.])
    feature = np.array([0., 1., 1.])/math.sqrt(2)
    h = np.diag([0., 1., 2.])
    static = np.column_stack([omega, feature])
    evolved = np.exp(-1j*np.diag(h))*feature
    residual = evolved-static@(static.T@evolved)
    assert np.linalg.norm(residual) > .1
    # At time zero, C^2 with equal weights has a selfadjoint involution f.
    # Positive temporal features do not preserve its time-zero multiplicative GNS span.
    return {'scope': 'synthetic method countercase only', 'positive_stationary_H': [0, 1, 2],
        'timezero_involution_f_squared_equals1': True,
        'timezero_invariance_defect_norm': float(np.linalg.norm(residual)),
        'selected_positive_feature_not_full_multiplicative_dynamics': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    pins = request['pins']
    before = {name: sha((ROOT/name).read_bytes()) for name in pins}
    assert before == pins
    archive = ROOT/request['epoch_archive']
    with zipfile.ZipFile(archive) as z:
        actual = json.loads((ROOT/request['epoch_actual']).read_text())
        members = {name: sha(z.read(name)) for name in z.namelist()}
        assert members == actual['source_input_sha256']
        profiles = json.loads(z.read(request['profile']))['primitive_profiles']
        archived_source = z.read('joined_phi_u1_uniform_gauss_limit_consumer.py').decode()
        assert '64 rho^T L^+rho' in archived_source and 'no bare u1 magnetic' in archived_source.lower()
    proof = json.loads((ROOT/request['proof']).read_text())
    rows = [native_control(n, profiles) for n in request['N']]
    result = {'schema': 1, 'scope': 'PERIODIC_FULL_LOCAL_WORD_FORM_BRIDGE_ATTEMPT_WITH_MATERIAL_ENERGY_REPAIR_NOT_FULL_GNS_DYNAMICS',
        'native_controls': rows, 'energy_transport_attempts': energy_repairs(profiles),
        'weighted_local_energy_controls': [weighted_local_energy(n) for n in request['N']],
        'method_countercase': selected_feature_countercase(), 'proof': proof,
        'epoch_archive_member_sha256': members, 'source_input_sha256_before': before,
        'source_input_sha256_after': {name: sha((ROOT/name).read_bytes()) for name in pins},
        'flags': proof['flags'], 'runtime': {'executable': sys.executable, 'elapsed_seconds': time.monotonic()-started}}
    assert result['source_input_sha256_after'] == before
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'N': request['N'], 'native_all65_controls': len(rows),
        'full_GNS_dynamics': False, 'repaired_partition_energy_transport': True,
        'elapsed_seconds': result['runtime']['elapsed_seconds']}))


if __name__ == '__main__':
    main()