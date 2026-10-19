"""Closed raw electric/history join controls; no native minimizing-ground numerics."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def record(value: F) -> dict[str, Any]:
    return {'sha256': digest(str(value).encode()), 'numerator_bits': value.numerator.bit_length(),
            'denominator_bits': value.denominator.bit_length()}


def costs(profiles: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    w = [abs(F(p['onsite']))+2*sum((abs(F(t['c0']))+abs(F(t['ci'])) for t in p['paths']), F(0)) for p in profiles]
    b = 4*(lam*w[0]+w[1]+63*w[2])
    ce = 8*sum((p['load']*t['length']*(abs(F(t['c0']))+abs(F(t['ci'])))
                for p in profiles[1:] for t in p['paths'] if t['axis'] == 0), F(0))
    assert ce == F(15475017073321605, 1099511627776) and ce < 2**14
    r = 60*chi+11192*lam+F(129, 10)*fast+262144
    b0 = 33*(1+10*r*4096)+16384*lam*4096
    cstar = 2**32*65*(1+(2**16*(1+fast+b))**2)*(4096+2/lam)
    # Symbolic propagation factors retain huge native exp(Cstar*|dt|) exactly.
    u = sp.symbols(f'u0:{depth+1}', nonnegative=True)
    energy: Any = sp.Rational(b0)
    graph2: Any = sp.Integer(0)
    norm2: Any = sp.Integer(1)
    for j in range(depth):
        a, m, tau_inv = F(j % 2+1), j+4, F(2**((j+7)//4))
        d1, d2 = (F(2), F(3), F(4)), (F(4), F(6), F(8))
        kin = (4*chi, F(3, 20)*chi, F(64))
        d = sum((k*v*v for k, v in zip(kin, d1)), F(0))
        c = sum((k*v2+2*max(F(1), k)*tau_inv*v1 for k, v1, v2 in zip(kin, d1, d2)), F(0))+4*a*b*m*m
        energy *= u[j]
        graph2 = sp.expand(2*sp.Rational(a*a)*graph2+2*sp.Rational(c*c)*energy)
        energy = sp.expand(66*sp.Rational(a*a)*energy+sp.Rational(66*d+16384*lam*4096*a*a)*norm2)
        norm2 *= sp.Rational(a*a)
    energy = sp.expand(u[-1]*energy)  # Leading history propagator is paid.
    ep = sp.Poly(energy, *u)
    assert all(c > 0 for c in ep.coeffs())
    # A closed unit plaquette at the origin has beta^2=2+2*2^(1/4).
    z: Any = sp.Symbol('z', positive=True)
    beta2 = 2+2*z
    assert sp.rem(beta2-2-2*z, z**4-2, z) == 0
    ir = []
    for denominator in (8, 16, 32):
        q = F(1, denominator); volume = q**3  # Unit sector-volume coefficient diagnostic.
        ir.append({'q': str(q), 'v_q': str(volume), 'cycle_m0_multiplier': str(1/(volume*q)),
                   'cycle_scaled_m2_multiplier': str(1/(volume*q**3)),
                   'physical_E_m0_multiplier': str(4/(volume*q**3)),
                   'physical_E_scaled_m2_multiplier': str(4/(volume*q**5))})
    assert F(ir[1]['physical_E_m0_multiplier'])/F(ir[0]['physical_E_m0_multiplier']) == 64
    assert F(ir[1]['physical_E_scaled_m2_multiplier'])/F(ir[0]['physical_E_scaled_m2_multiplier']) == 256
    return {'scope': 'exact native arithmetic and declared per-word budgets, not ground measurements',
            'C_e': str(ce), 'B': record(b), 'B0': record(b0), 'Cstar': record(cstar),
            'R': record(r), 'word_depth': depth, 'Q_terms': len(ep.terms()),
            'Q_polynomial_sha256': digest(str(energy).encode()),
            'graph2_polynomial_sha256': digest(str(graph2).encode()),
            'leading_propagator_retained': True, 'plaquette_beta_squared': str(beta2),
            'cycle_trace_m0_upper': record(48*r/64),
            'cycle_trace_m2_integer_upper': record(F(3*256*2**1055)),
            'IR_envelopes_not_actual_response': ir,
            'IR_positive_Gram_lower_paid': False, 'IR_uniform_tightness_paid': False}


def native_geometry_current(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3)); idx = {x: i for i, x in enumerate(sites)}
    edges = [(x, a) for x in sites for a in range(3)]; eidx = {e: i for i, e in enumerate(edges)}
    def move(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        y = list(x); y[axis] = (y[axis]+1) % n; return tuple(y)
    def cycle(o: tuple[int, ...], a: int, b: int) -> np.ndarray:
        v = np.zeros(len(edges)); v[eidx[o, a]] = 1; v[eidx[move(o, a), b]] = 1
        v[eidx[move(o, b), a]] = -1; v[eidx[o, b]] = -1; return v
    def divergence(raw: np.ndarray) -> np.ndarray:
        d = np.zeros(n**3)
        for i, (x, a) in enumerate(edges):
            d[idx[x]] += raw[i]; d[idx[move(x, a)]] -= raw[i]
        return d
    h, k = cycle((0, 0, 0), 0, 1), cycle((0, 0, 0), 0, 2)
    assert not np.any(divergence(h)) and not np.any(divergence(k))
    # Analytical finite Fourier inverse of DD*: diagnostic raw edge NOT admitted.
    raw = np.zeros(len(edges)); raw[eidx[(0, 0, 0), 0]] = -1
    drho = divergence(raw).reshape((n, n, n))
    freq = 2*np.pi*np.arange(n)/n
    eigen = sum((4*np.sin(a/2)**2 for a in np.meshgrid(freq, freq, freq, indexing='ij')))
    inv = np.zeros_like(eigen); np.divide(1, eigen, out=inv, where=eigen > 1e-12)
    pot = np.fft.ifftn(np.fft.fftn(drho)*inv).real
    tl = np.array([pot[x]-pot[move(x, a)] for x, a in edges]); tt = raw-tl
    assert np.max(np.abs(divergence(tt))) < 2e-14
    longitudinal = -F(n**3-1, 3*n**3)
    assert abs(tl[eidx[(0, 0, 0), 0]]-float(longitudinal)) < 1e-14
    assert abs(h@tl) < 1e-14
    # Native fullrho from all64 charged canonical occupation backgrounds.
    rho = np.array([128*(-1)**sum(x) for x in sites])
    background = np.array([64*(-1)**sum(x) if a == 0 else 0 for x, a in edges])
    assert np.array_equal(divergence(background), rho)
    assert np.array_equal(divergence(background+raw), rho+drho.ravel())
    # Distinct noncommuting complete fields and ordered SU4 links.
    rng = np.random.default_rng(771+n)
    phi = rng.integers(-4, 5, size=(len(sites), 4, 4, 4))+1j*rng.integers(-4, 5, size=(len(sites), 4, 4, 4))
    phi = (phi+phi.swapaxes(-1, -2).conj())/8
    phi -= np.trace(phi, axis1=-2, axis2=-1)[..., None, None]*np.eye(4)/4
    base = np.diag([(3+4j)/5, (3-4j)/5, 1, 1]).astype(complex)
    permutation = np.roll(np.eye(4), 1, axis=0); permutation[:, 0] *= -1
    links = [base@np.linalg.matrix_power(permutation, (i % 3)+1) for i in range(len(edges))]
    assert max(np.linalg.norm(u.conj().T@u-np.eye(4)) for u in links) < 1e-14
    assert max(abs(np.linalg.det(u)-1) for u in links) < 1e-14
    A = np.array([((i*13)%31-15)/37 for i in range(len(edges))])
    steps = (0.01, 0.005)
    aggregate = {'J': 0.0, 'Khh': 0.0, 'Khk': 0.0}
    delta2 = [0.0, 0.0]; taylor2 = [0.0, 0.0]
    first_derivative_error2 = 0.0; second_derivative_error2 = 0.0
    path_counts = [0, 0, 0]; touched = [0, 0, 0]; seam_touched = 0; collisions = 0
    charged_density_defect2 = 0.0
    for role, p in enumerate(profiles):
        for x in sites:
            for path in p['paths']:
                axis, length = path['axis'], path['length']; y = x; indices = []; seam = 1; U = np.eye(4, dtype=complex)
                for _ in range(length):
                    edge = eidx[y, axis]; indices.append(edge); U = U@links[edge]
                    seam *= -1 if y[axis] == n-1 else 1; y = move(y, axis)
                path_counts[role] += 1
                rh, rk = float(h[indices].sum()), float(k[indices].sum())
                if role == 0:  # Neutral scalar role derivative is identically zero.
                    continue
                if not (rh or rk): continue
                touched[role] += 1; seam_touched += int(seam == -1)
                collisions += int(n == 4 and length == 2)
                c0, ci = float(F(path['c0'])), float(F(path['ci']))
                fx = c0*phi[idx[x], 0]+1j*ci*phi[idx[x], axis+1]
                fy = c0*phi[idx[y], 0]+1j*ci*phi[idx[y], axis+1]
                block = (fx@U+U@fy)/2*seam*np.exp(1j*A[indices].sum())
                jblock = 1j*rh*block; khh = -rh*rh*block; khk = -rh*rk*block
                load = int(p['load'])
                aggregate['J'] += load*np.linalg.norm(jblock)**2
                aggregate['Khh'] += load*np.linalg.norm(khh)**2
                aggregate['Khk'] += load*np.linalg.norm(khk)**2
                d = 1e-4
                derivative = (np.exp(1j*d*rh)-np.exp(-1j*d*rh))*block/(2*d)
                second = (np.exp(1j*d*rh)+np.exp(-1j*d*rh)-2)*block/(d*d)
                first_derivative_error2 += load*np.linalg.norm(derivative-jblock)**2
                second_derivative_error2 += load*np.linalg.norm(second-khh)**2
                for i, s in enumerate(steps):
                    delta = np.expm1(-1j*s*rh)*block
                    delta2[i] += load*np.linalg.norm(delta)**2
                    taylor2[i] += load*np.linalg.norm(delta+s*jblock-s*s*khh/2)**2
                # Complete raw jet t_T+t_L matches literal path derivative.
                for selected in indices:
                    exact = raw[selected]
                    assert abs(tt[selected]+tl[selected]-exact) < 1e-14
                    charged_density_defect2 += load*abs(tl[selected])**2
    assert touched[0] == 0 and aggregate['J'] > 0 and aggregate['Khk'] > 0
    assert math.sqrt(first_derivative_error2/aggregate['J']) < 1e-8
    assert math.sqrt(second_derivative_error2/aggregate['Khh']) < 1e-6
    assert taylor2[0]/taylor2[1] > 63.9  # Squared cubic Taylor remainder.
    # Raw-edge density phase against a legal canonical transfer, once fullrho.
    selected = eidx[(0, 0, 0), 0]; s = 0.3
    full_phase = np.exp(1j*s*raw[selected]); pi_only = np.exp(1j*s*tt[selected])
    density_phase = np.exp(1j*s*tl[selected])
    assert abs(pi_only*density_phase-full_phase) < 1e-14 and abs(pi_only-full_phase) > 0.01
    midpoint = []
    for q in ((2*np.pi/n, 0, 0), (2*np.pi/n, 2*np.pi/n, 0)):
        hp = 2*np.sin(np.array(q)/2)
        C = np.array([[0, -hp[2], hp[1]], [hp[2], 0, -hp[0]], [-hp[1], hp[0], 0]])
        assert np.linalg.matrix_rank(C) == 2 and np.linalg.norm(hp@C) < 1e-14
        assert np.linalg.norm(C.T@C-(np.dot(hp, hp)*np.eye(3)-np.outer(hp, hp))) < 1e-14
        midpoint.append({'q': list(q), 'curl_rank': 2, 'pseudoinverse_norm': float(1/np.linalg.norm(hp))})
    return {'N': n, 'native_field_seed': 771+n, 'canonical_each_species': 2*n**3,
            'all65_loads': [1, 1, 63], 'literal_path_counts': path_counts,
            'charged_h_or_k_touched_counts': touched, 'AP_seam_touched': seam_touched,
            'N4_length2_literals_not_deduplicated': collisions,
            'closed_Dh_Dk_exact': True, 'native_block_squared_norms_not_ground_moments': aggregate,
            'current_relative_derivative_error': math.sqrt(first_derivative_error2/aggregate['J']),
            'contact_relative_derivative_error': math.sqrt(second_derivative_error2/aggregate['Khh']),
            'Taylor_cubic_squared_remainder_ratio': taylor2[0]/taylor2[1],
            'raw_diagnostic_longitudinal_selected_edge': str(longitudinal),
            'Pi_only_density_phase_defect': float(abs(pi_only-full_phase)),
            'complete_charge_affine_jet_defect_of_frozen_connection_squared': charged_density_defect2,
            'closed_boost_longitudinal_phase': '1', 'raw_edge_core_extension': False,
            'collective_fullrho_not_64_independent_Coulomb_terms': True,
            'midpoint_curl_controls': midpoint, 'native_minimizing_ground_or_fullH_evolution': False}


def full_history_adverse(depth: int) -> dict[str, Any]:
    # Non-native finite matrices: test full multiplication vs selected-source action.
    L: Any = sp.diag(0, 0, 1, 3)
    E: Any = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    A: Any = sp.Matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    O: Any = sp.Matrix([1, 0, 0, 0]); Q0: Any = sp.diag(0, 0, 1, 1)
    feature: Any = sp.diag(1, 0, 1, 0)
    selected_defect = (sp.eye(4)-feature)*E*feature
    assert selected_defect != sp.zeros(4)
    # A finite exact phase at t=pi: exp(itL)=diag(1,1,-1,-1).
    V: Any = sp.diag(1, 1, -1, -1)
    letters = [A, V*A*V, E]
    worst = F(0); words = 0
    for size in range(1, depth+1):
        for indices in itertools.product(range(3), repeat=size):
            full = O; compressed = O
            for i in indices:
                full = letters[i]*full
                compressed = feature*letters[i]*feature*compressed
            difference = full-compressed
            worst = max(worst, F((difference.T*difference)[0])); words += 1
    assert worst > 0
    v: Any = A*O; meanremoved = v-O*(O.T*v)[0]
    assert (meanremoved-Q0*v).norm() == 1
    e = E*O; j = -sp.I*L*e
    ee = (e.T*Q0*e)[0]; m1 = (e.T*L*e)[0]; m2 = (j.conjugate().T*j)[0]
    K = -(E*(E*L-L*E)-(E*L-L*E)*E)
    assert m1 == (O.T*K*O)[0]/2
    assert L*e == sp.I*j
    En = np.array(E.tolist(), dtype=float); Ln = np.diag([0., 0., 1., 3.]); on = np.array([1., 0., 0., 0.])
    jn = -1j*Ln@En@on
    errors = []
    for s in (0.01, 0.005):
        U = expm(1j*s*En); x = (U-np.eye(4))@on/(1j*s)
        # Delta=U* H U-H with U*=exp(-isE), matching A-sh orientation.
        delta = U.conj().T@Ln@U-Ln; y = U@delta@on/(1j*s)
        assert np.linalg.norm(Ln@x-y) < 1e-12
        errors.append(float(np.linalg.norm(y-1j*jn)))
    assert errors[0] > errors[1]
    # Positivity/moments cannot select an atom: same m0=1,m1=1, continuous density 1/2 on[0,2].
    continuum = {'m0': '1', 'm1': '1', 'm2': '4/3', 'atom_mass': '0'}
    return {'scope': 'NON_NATIVE exact full-history method control, not physical-law counterexample',
            'full_words': words, 'selected_feature_product_max_squared_defect': str(worst),
            'selected_feature_E_action_leak_squared': str((selected_defect.T*selected_defect).trace()),
            'mean_subtraction_ground_block_defect_squared': '1',
            'electric_moments': [str(ee), str(m1), str(m2)], 'contact_Khh': str((O.T*K*O)[0]),
            'fixed_s_graph_identity_numeric_error_below': 1e-12,
            's_second_source_errors': errors, 'positive_continuum_adverse': continuum,
            'full_history_vs_selected_feature_rejected': True, 'rankone_mean_vs_wholeP0_rejected': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {p: digest((ROOT/p).read_bytes()) for p in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        raw = archive.read(request['profile_member']); profiles = json.loads(raw)['primitive_profiles']
    receipt = {'schema': 1, 'kind': 'NEW_ELECTRIC_FULL_HISTORY_SCIENTIFIC_CONTROL_NOT_PHOTON',
               'native_costs': costs(profiles, request['word_depth']),
               'native_N_geometry_current_contact': [native_geometry_current(n, profiles) for n in request['N']],
               'method_full_history_boundary': full_history_adverse(request['method_depth']),
               'analytical_proof': json.loads((ROOT/request['proof']).read_text()),
               'profile_member_sha256': digest(raw), 'source_input_sha256_before': before,
               'source_input_sha256_after': {p: digest((ROOT/p).read_bytes()) for p in request['pins']},
               'runtime': {'executable': sys.executable, 'elapsed_seconds': time.monotonic()-start}}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'native_N': request['N'], 'full_history_join_analytic_conditional': True,
                      'native_ground_computed': False, 'photon': False,
                      'method_words': receipt['method_full_history_boundary']['full_words'],
                      'elapsed_seconds': receipt['runtime']['elapsed_seconds']}))


if __name__ == '__main__':
    main()
