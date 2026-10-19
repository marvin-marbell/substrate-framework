"""Native operator/interval construction; no ground or evolution backend.

The controls below exercise the full-law analytic identities in the accompanying
proof. Factor/CAR and spectral controls are not evaluations of the ground state.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

W = Fraction(46425051219964815, 8796093022208)
CE = Fraction(15475017073321605, 1099511627776)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(value: Any) -> Any:
    return sp.simplify(sp.expand_complex(value))


def face(x: tuple[int, int, int], n: int) -> dict[Any, int]:
    def shift(axis: int) -> tuple[int, int, int]:
        y = list(x)
        y[axis] = (y[axis] + 1) % n
        return tuple(y)
    return {(x, 0): 1, (shift(0), 1): 1, (shift(1), 0): -1, (x, 1): -1}


def contiguous_geometry(n: int) -> dict[str, Any]:
    a, b = face((0, 0, 0), n), face((1, 0, 0), n)
    d = {e: a.get(e, 0) - b.get(e, 0) for e in a.keys() | b.keys()}
    d = {e: k for e, k in d.items() if k}
    overlap = sum(k * b.get(e, 0) for e, k in a.items())
    d2 = sum(k*k for k in d.values())
    div: dict[Any, int] = {}
    for (x, axis), k in d.items():
        y = list(x); y[axis] = (y[axis] + 1) % n; y = tuple(y)
        div[x] = div.get(x, 0) + k
        div[y] = div.get(y, 0) - k
    assert overlap == -1 and d2 == 10 and not any(div.values())
    roles = [('neutral', 1, 0, (1,)), ('active', 1, 1, (1, 2)),
             ('auxiliary', 63, 1, (1, 2))]
    histogram: dict[str, dict[str, int]] = {}
    charged_touched = neutral_touched = ap_touched = 0
    for role, copies, charge, lengths in roles:
        hist: dict[str, int] = {}
        for x in itertools.product(range(n), repeat=3):
            for axis in range(3):
                for length in lengths:
                    y, frequency, touched, ap = x, 0, False, 1
                    for _ in range(length):
                        k = d.get((y, axis), 0)
                        frequency += charge*k; touched = touched or bool(k)
                        z = list(y)
                        if z[axis] == n-1: ap *= -1
                        z[axis] = (z[axis] + 1) % n; y = tuple(z)
                    # Forward and adjoint are different literal Hamiltonian terms.
                    for orientation in (1, -1):
                        key = str(orientation*frequency)
                        hist[key] = hist.get(key, 0) + copies
                    if touched:
                        if charge: charged_touched += 2*copies
                        else: neutral_touched += 2*copies
                        if ap == -1: ap_touched += 2*copies
        if charge == 0: assert set(hist) == {'0'}
        histogram[role] = hist
    assert histogram['auxiliary'] == {k: 63*v for k, v in histogram['active'].items()}
    # The actual noncompact Bianchi holds on the native raw incidence; the
    # relative integer Wilson character is physical without adjoining raw B.
    return {'N': n, 'ca_dot_cb': overlap, 'd_squared': d2, 'D_d': 0,
            'shared_edge_retained': True, 'charged_touched_terms': charged_touched,
            'neutral_touched_but_unchanged': neutral_touched, 'AP_touched_terms': ap_touched,
            'all_forward_adjoint_frequency_counts': histogram,
            'each_species_particles': 2*n**3, 'Ref_particles': 4,
            'fullrho': 'sum_active1_aux63(n_s-2)+(NRef-4)delta0',
            'M1_neighbor': '-64*omega(cos(Ba-Bb)); actual contact, not Gamma_spin'}


def contiguous_cp() -> dict[str, Any]:
    D = sp.symbols('D', real=True)
    psi = sp.Function('psi')(D)
    row = [(1 + sign*sp.exp(sp.I*orientation*D))/(2*sp.sqrt(2))
           for orientation in (1, -1) for sign in (1, -1)]
    tp = exact(sum(sp.conjugate(k)*k for k in row))
    connection = exact(sum(sp.conjugate(k)*sp.diff(k, D) for k in row))
    jets = exact(sum(sp.conjugate(sp.diff(k, D))*sp.diff(k, D) for k in row))
    increment = sp.simplify(sum(sp.conjugate(k)*(-640*sp.diff(k*psi, D, 2)) for k in row)
                            + 640*sp.diff(psi, D, 2))
    assert (tp, connection, jets) == (1, 0, sp.Rational(1, 2))
    assert sp.simplify(increment - 320*psi) == 0
    # Choose alpha=pi/10 so feedback flips the RELATIVE Wilson character.
    dual = exact(sum(sp.conjugate(k)*sp.exp(sp.I*D)*k*(1 if sign == 1 else -1)
                     for orientation in (1, -1) for sign in (1, -1)
                     for k in [(1+sign*sp.exp(sp.I*orientation*D))/(2*sp.sqrt(2))]))
    assert exact(dual-(1+sp.exp(2*sp.I*D))/2) == 0
    alpha, nu = sp.symbols('alpha nu', real=True)
    dual_path = exact(sum(sp.conjugate(k)*k*(1 if sign == 1 else sp.exp(-sp.I*alpha*nu))
                          for orientation in (1, -1) for sign in (1, -1)
                          for k in [(1+sign*sp.exp(sp.I*orientation*D))/(2*sp.sqrt(2))]))
    desired = sp.cos(D/2)**2 + sp.exp(-sp.I*alpha*nu)*sp.sin(D/2)**2
    assert exact(dual_path-desired) == 0
    # Two face allocations <=64W. Difference multiplier <=2. Ground r mean<16.
    envelope = 128*W*16
    needed_K = Fraction(320, 1)/Fraction(1, 16)
    assert -envelope < 0 < needed_K < envelope
    return {'TP': str(tp), 'connection': str(connection), 'jet_square': str(jets),
            'complete_HE_HC_increment': 320, 'relative_Wilson_dual': '(1+U_d^2)/2',
            'charged_path_dual': str(desired), 'physical_hopping': 1,
            'complete_carrier_gain_interval': [str(-envelope), str(envelope)],
            'ground_variational_intersection': [str(-envelope), '320'],
            'conditional_defect_target': 'omega(sin^2(D/2))<=1/16 requires K>=5120+16*epsilon',
            'K_required_epsilon0': str(needed_K), 'state_independent_positive_K_paid': False,
            'first_gate': 'FAILED_SUFFICIENT_SIGNED_CARRIER_COERCIVITY_NOT_NO_GO',
            'no_scalar_spin_trace_or_field_state_evaluation': True}


def mixed_carrier_control() -> dict[str, Any]:
    # Full transfer identity on all 256 two-site, four-colour Fock columns.
    # The finite CAR module is an operator identity control, NOT a full quantum ground.
    modes = 8
    phi = [1, 2, -1, -2]  # A traceless Hermitian full-rank primitive factor.
    transfer = np.zeros((256, 256), dtype=np.int64)
    def ladder(mask: int, mode: int, create: bool) -> tuple[int, int] | None:
        if bool(mask & (1 << mode)) == create: return None
        return mask ^ (1 << mode), (-1)**((mask & ((1 << mode)-1)).bit_count())
    for mask in range(1 << modes):
        for colour in range(4):
            r = ladder(mask, 4+colour, False)
            if r is None: continue
            s = ladder(r[0], colour, True)
            if s is not None: transfer[s[0], mask] += phi[colour]*r[1]*s[1]
    carrier = transfer + transfer.T
    assert int(np.trace(carrier)) == 0 and np.any(carrier)
    parity = np.array([(-1)**((mask & 15).bit_count()) for mask in range(256)])
    assert np.array_equal(parity[:, None]*carrier*parity[None, :], -carrier)
    # Canonical TOTAL occupation does not remove these signs: use four particles.
    sector = [mask for mask in range(256) if mask.bit_count() == 4]
    c4 = carrier[np.ix_(sector, sector)]
    assert int(np.trace(c4)) == 0 and np.any(c4)
    # V_e is the SUM on each leg, so the four-leg expansion has 64^4 assignments.
    patterns = {}
    for assignment in itertools.product(range(2), repeat=4):
        aux_legs = sum(assignment)
        key = ''.join(map(str, assignment))
        patterns[key] = 63**aux_legs
    assert sum(patterns.values()) == 64**4
    mixed = sum(v for k, v in patterns.items() if k not in ('0000', '1111'))
    assert mixed > 0
    return {'full_Fock_columns': 256, 'canonical_four_particle_dimension': len(sector),
            'trace_zero_nonzero_Hermitian_transfer': True,
            'exact_parity_sign_partner': True, 'uniform_positive_carrier_form_rejected': True,
            'control_is_not_ground_or_SU4_Gauss_state': True,
            'all_fourleg_species_assignments': 64**4, 'active_aux_profile_pattern_weights': patterns,
            'mixed_profile_assignments': mixed,
            'complete_current': 'J_d=i sum_s,path (d.t_path)(T_s,path-T_s,path^dag)',
            'neutral_quantum_coupling': 'neutral -Lambda dGamma(h(Phi,U)) retained inside H and ground/memory',
            'shared_field_two_equal_unitflows_energy': 256,
            'wrong_independent_field_energy': 128,
            'two_opposite_unitflows_energy': 0}


def dynamic_route() -> dict[str, Any]:
    # These are the new full quantum source identities, not a substituted rotor.
    q, L, v = sp.symbols('q L v', positive=True, real=True)
    e, f, j = sp.symbols('e f j', commutative=True)
    b = q*f
    d = 128*e - 2*sp.I*L*f
    # L e=i j is the paid vacuum source graph. No LJ or LE^2 domain is needed.
    Ld = 128*sp.I*j - 2*sp.I*L**2*f
    curvature_residual = 64*q*j-v**2*q**2*b
    identity = exact((L**2-v**2*q**2)*b-curvature_residual-sp.I*q*Ld/2)
    assert identity == 0
    # Frequency-shell projection makes all L powers BOUNDED and pays graph.
    # Chebyshev gives scaled atom concentration when this residual is o(q^5/2).
    threshold = sp.Rational(1, 16)
    upper_rows = []
    for n in (4, 6, 8, 16, 64):
        qmin = 2*sp.sin(sp.pi/n)
        # ||b||<=pi sqrt(3V)<6 sqrt(V); qmin>=4/N gives sqrt(V/qmin)<=N^2/2.
        seam_scaled_upper = sp.sqrt(128) + 6*n*n
        residual_scaled_upper = 8*sp.sqrt(2) + 6*n*n
        upper_rows.append({'N': n, 'qN_exact': str(qmin),
                           'b_norm_upper': f'6*sqrt({n**3})',
                           'seam_over_sqrtq_upper': str(seam_scaled_upper),
                           'curvature_residual_over_q5half_upper': str(residual_scaled_upper),
                           'target_residual': str(threshold), 'certifies_target': False})
    # Precise NONNATIVE adverse controls exercise criterion, not native moments.
    weights = [sp.Rational(1, 256), sp.Rational(1, 512)]
    energies = [sp.Rational(1), sp.Rational(3, 2)]
    s = sum(weights)
    normalized_curvature = sum(w*x*x for w, x in zip(weights, energies))/s
    residual = sum(w*(x*x-normalized_curvature)**2 for w, x in zip(weights, energies))/s
    assert residual > 0
    K = sp.diag(*(x*x for x in energies))
    wb = sp.Matrix([sp.sqrt(w/s) for w in weights])
    rb = sp.Matrix([-wb[1], wb[0]])
    M = exact((wb.T*K*wb)[0])
    B = exact((rb.T*K*wb)[0])
    Dmemory = exact((rb.T*K*rb)[0])
    assert exact(M-normalized_curvature) == 0 and exact(B*B-residual) == 0
    assert exact((wb.T*wb)[0]-1) == 0 and exact((rb.T*wb)[0]) == 0
    zmemory = sp.Rational(-1)
    lhs = exact((wb.T*(zmemory*sp.eye(2)-K).inv()*wb)[0])
    rhs = exact(1/(zmemory-M-B*B/(zmemory-Dmemory)))
    assert lhs == rhs
    assert exact((wb.T*(zmemory*sp.eye(2)-K).inv()*wb)[0]*s) != lhs
    assert residual == sp.Rational(25, 72)
    # Wrong use of the reducible full resolvent in memory is an exact discriminator.
    H = sp.Matrix([[2, 1], [1, 3]])
    z = sp.Rational(-1)
    irr = 1/(z-3)
    reducible = (z*sp.eye(2)-H).inv()[1, 1]
    assert irr != reducible
    return {'new_operator_identity': 'P_I(L^2-v^2 q^2)b = 64q P_I J_h-v^2 q^2 b+(i q/2)L d',
            'identity_residual_exact': str(identity),
            'b_definition': 'q P_I Q0 F_sigma Omega, C^T sigma=h, ||h||=1, ||sigma||=1/q',
            'd_definition': '128 P_I E_h Omega-2i L P_I Q0 F_sigma Omega',
            'current_source': 'P_I J_h Omega = -i L P_I E_h Omega, full all64 Phi/SU4/rho Hamiltonian',
            'normalized_connected_curvature': 'A_q=64q Re(b^* P_I J_h)/s_q; s_q=b^*b; wholeP0, not mean subtraction',
            'irreducible_force': 'R_b[64q P_I J_h+(i q/2)Ld]/(q^2 sqrt(s_q)); R_b=P_I-W_b W_b^*, K_q=(L/q)^2 on band, W_b=b S_b^-1/2',
            'source_normalization_is_paid_only_on_positive_Gram_support': True,
            'atom_concentration_criterion': 'c I2<=S_b/q<=C I2; full M_q (current plus signed parallel seam) -> v^2 I2 and B_q ->0, equivalently ||(L^2-v^2 q^2)b||=o(q^(5/2)); uniform finiteALLq then compatible common-ae limit',
            'Faraday_transfer_criterion': '||d||<=delta sqrt(q), S_b/q>=c I2, band velocities>=vminus => electric sigma(I)>= (vminus sqrt(c)/64-delta/128)^2 I2',
            'paid_envelope_rows': upper_rows,
            'second_gate': 'FAILED_AVAILABLE_CONNECTED_CURVATURE_AND_SEAM_ENVELOPE_NOT_NO_GO',
            'positive_Sb_or_signed_current_lower_paid': False,
            'spectral_control_not_native': {'two_atom_normalized_curvature': str(normalized_curvature),
                                           'nonzero_curvature_variance': str(residual),
                                           'secondorder_source_Gram': str(s), 'normalized_M': str(M),
                                           'normalized_B': str(B), 'compressed_D': str(Dmemory),
                                           'exact_normalized_memory_resolvent': str(lhs),
                                           'irreducible_memory': str(irr), 'wrong_reducible_memory': str(reducible)},
            'finite_isolated_atom_or_phase_claim': False}


def all_mode_scope(n: int) -> dict[str, Any]:
    maximum = 0.0
    for indices in itertools.product(range(n), repeat=3):
        p = 2*np.pi*np.array(indices)/n
        hp = 2*np.sin(p/2)
        q2 = float(hp@hp)
        if q2 == 0: continue
        PT = np.eye(3)-np.outer(hp, hp)/q2
        maximum = max(maximum, float(np.max(np.abs(PT@PT-PT))),
                      float(np.max(np.abs(PT@hp))))
        assert abs(float(np.trace(PT))-2) < 1e-12
    assert maximum < 1e-12
    return {'N': n, 'every_discrete_mode': n**3, 'two_transverse_modes_offzero': True,
            'geometry_float_control_max_error': maximum, 'qN': float(2*np.sin(np.pi/n)),
            'zero_mode': 'three torons, excluded from photon criterion',
            'native_AP_fermion_momenta_not_inserted_into_gauge_q': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    request = json.loads(request_path.read_text())
    root = request_path.parent
    for name, expected in request['inputs'].items():
        assert sha(root/name) == expected, name
    assert request['physical_hopping'] == 1
    assert sha(Path(request['contract_path'])) == request['contract_sha256']
    result = {'kind': 'EXECUTED_NATIVE_CONTIGUOUS_CP_AND_NORMALIZED_CONNECTED_CURVATURE_ATTEMPTS_NOT_ADMISSION',
              'source_sha256': sha(Path(__file__)), 'request_sha256': sha(request_path),
              'inputs': request['inputs'], 'contiguous_geometry': [contiguous_geometry(n) for n in (4, 6, 8)],
              'first_gate': contiguous_cp(), 'mixed_carrier': mixed_carrier_control(),
              'second_gate': dynamic_route(), 'finite_ALLq_geometry': [all_mode_scope(n) for n in (4, 6, 8)],
              'actual_ground_moments_numerically_evaluated': False,
              'no_old_entrypoint_import_or_execution': True, 'law_root_boundary_adoption': False,
              'scientific_admission': False, 'campaign_complete': False}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print('Native contiguous full-H CP and distinct filtered connected-curvature gates exercised; neither pays physical1 IR/phase.')


if __name__ == '__main__':
    main()
