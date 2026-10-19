"""Independent electric-history mathematical controls, not a native ground solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)
F = Fraction


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def norm2(a: Any) -> Any:
    return sp.simplify((a.conjugate().T * a).trace())


def exact_record(x: F) -> dict[str, Any]:
    return {'sha256': sha(str(x).encode()), 'numerator_bits': x.numerator.bit_length(),
            'denominator_bits': x.denominator.bit_length()}


def native_constants(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    ce = [8*sum((F(p['load'])*t['length']*(abs(F(t['c0']))+abs(F(t['ci'])))
                for p in profiles[1:] for t in p['paths'] if t['axis'] == axis), F(0))
          for axis in range(3)]
    assert ce == [F(15475017073321605, 1099511627776)]*3
    weights = [abs(F(p['onsite']))+2*sum((abs(F(t['c0']))+abs(F(t['ci']))
               for t in p['paths']), F(0)) for p in profiles]
    b = 4*(lam*weights[0]+weights[1]+63*weights[2])
    r = 60*chi+11192*lam+F(129, 10)*fast+262144
    b0 = 33*(1+10*r*4096)+16384*lam*4096
    cstar = 2**32*65*(1+(2**16*(1+fast+b))**2)*(4096+2/lam)
    assert b < 512*lam and 2*r/lam < 2**2053 and ce[0] < 2**14
    # Squaring eliminates the only radical in the claimed convenient current bound.
    assert ce[0]**4*(2*r/lam) < 2**2110
    h1, m, tau_inv = 4, 16, F(8)
    tail = (h1*ce[0])**2 * 2*m*b0*tau_inv/lam
    delta2 = F(1, 4096)
    cutoff = math.ceil(tail/delta2)
    assert tail/cutoff <= delta2
    active = 8*sum((t['length']*(abs(F(t['c0']))+abs(F(t['ci'])))
                   for t in profiles[1]['paths'] if t['axis'] == 0), F(0))
    assert ce[0] == 64*active and ce[0]**2 == 4096*active**2
    return {'scope': 'EXACT_NATIVE_COEFFICIENT_AND_BOUND_ARITHMETIC_NOT_MEASUREMENTS',
            'C_e_all_axes': [str(x) for x in ce], 'roles': [p['role'] for p in profiles],
            'loads': [p['load'] for p in profiles], 'B': exact_record(b),
            'R': exact_record(r), 'B0': exact_record(b0), 'Cstar': exact_record(cstar),
            'all64_sum_before_square_factor': 4096, 'wrong_independent_square_factor': 64,
            'wrong_independent_variance_underestimate_factor': 64,
            'radial_and_QJ_power_claims_exactly_satisfied': True,
            'dense_word_tail_coefficient': exact_record(tail),
            'declared_h_l1': h1, 'declared_site_m': m, 'declared_tau_inverse': str(tau_inv),
            'requested_tail_error_squared': str(delta2), 'sufficient_cutoff': exact_record(F(cutoff)),
            'tail_upper_at_cutoff': exact_record(tail/cutoff)}


def native_paths(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    def step(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        return tuple((v+int(j == axis)) % n for j, v in enumerate(x))
    def face(x: tuple[int, ...], a: int, b: int) -> dict[tuple[Any, int], int]:
        return {(x, a): 1, (step(x, a), b): 1, (step(x, b), a): -1, (x, b): -1}
    origin = (n-1, 0, n-1)
    h = face(origin, 0, 1)
    k = face(origin, 0, 2)
    def div(flow: dict[tuple[Any, int], int]) -> dict[Any, int]:
        out: dict[Any, int] = {}
        for (x, a), value in flow.items():
            y = step(x, a)
            out[x] = out.get(x, 0)+value
            out[y] = out.get(y, 0)-value
        return out
    assert all(v == 0 for v in div(h).values()) and all(v == 0 for v in div(k).values())
    I: Any = sp.I
    diagonal: Any = sp.diag(I, -I, 1, 1)
    permutation: Any = sp.zeros(4)
    for j in range(4):
        permutation[(j+1) % 4, j] = -1 if j == 0 else 1
    assert diagonal.det() == permutation.det() == 1
    assert diagonal.conjugate().T*diagonal == permutation.conjugate().T*permutation == sp.eye(4)
    phi0: Any = sp.diag(3, 1, -1, -3)/7
    phi1: Any = sp.Matrix([[0, 1, 0, 0], [1, 0, I, 0], [0, -I, 0, 1], [0, 0, 1, 0]])/5
    def link(x: tuple[int, ...], a: int) -> Any:
        return diagonal if (sum(x)+a) % 2 else permutation
    rows = [dict(J=sp.Integer(0), Khh=sp.Integer(0), Khk=sp.Integer(0)) for _ in profiles]
    touches = [0]*len(profiles)
    seams = aliases = wrong_order = mixed = 0
    records: list[Any] = []
    for role, p in enumerate(profiles):
        charge = int(role != 0)
        for x in sites:
            for t in p['paths']:
                a, length = int(t['axis']), int(t['length'])
                y, flow, us, ap = x, [], [], 1
                for _ in range(length):
                    flow.append((y, a)); us.append(link(y, a))
                    ap *= -1 if y[a] == n-1 else 1
                    y = step(y, a)
                rh = charge*sum(h.get(e, 0) for e in flow)
                rk = charge*sum(k.get(e, 0) for e in flow)
                if not (rh or rk):
                    continue
                touches[role] += 1; seams += int(ap == -1)
                aliases += int(n == 4 and length == 2)
                mixed += int(rh*rk != 0)
                u: Any = sp.eye(4)
                for ue in us:
                    u *= ue
                c0, ci = sp.Rational(t['c0']), sp.Rational(t['ci'])
                fx: Any = c0*phi0+I*ci*phi1
                fy: Any = c0*(phi0+phi1)/2+I*ci*phi0
                block: Any = ap*(fx*u+u*fy)/2
                forward: Any = sp.zeros(8)
                forward[:4, 4:] = block
                hermitian: Any = forward+forward.conjugate().T
                j: Any = I*rh*(forward-forward.conjugate().T)
                khh: Any = -rh*rh*hermitian
                khk: Any = -rh*rk*hermitian
                assert norm2(j-j.conjugate().T) == 0 and norm2(khk-khk.conjugate().T) == 0
                # Formal charged scalar phase derivatives; no finite-difference cancellation.
                dhp: Any = -(rh+rk)**2*hermitian
                dhm: Any = -(rh-rk)**2*hermitian
                assert norm2(dhp-dhm-4*khk) == 0
                for key, value in [('J', j), ('Khh', khh), ('Khk', khk)]:
                    rows[role][key] += int(p['load'])*norm2(value)
                if length == 2:
                    wrong: Any = us[1]*us[0]
                    wrong_order += int(norm2(fx*wrong+wrong*fy-fx*u-u*fy) > 0)
                records.append([role, x, y, a, length, rh, rk, ap, str(norm2(block))])
    assert touches[0] == 0 and touches[1] == touches[2] > 0 and mixed > 0 and seams > 0
    assert wrong_order > 0 and (aliases > 0 if n == 4 else aliases == 0)
    return {'scope': 'EXACT_NATIVE_ORDERED_LITERAL_COEFFICIENT_IDENTITIES_NOT_GROUND_STATE',
            'N': n, 'each_canonical_number': 2*n**3, 'Ref_inert': 4,
            'closed_h_and_k_exact': True, 'charged_touch_counts': touches,
            'AP_seam_touches': seams, 'N4_length2_literal_touches': aliases,
            'wrong_SU4_length2_order_nonzero_defects': wrong_order,
            'nonzero_mixed_contact_literals': mixed,
            'current_contact_block_norms': [{key: str(value) for key, value in row.items()} for row in rows],
            'literal_record_sha256': sha(json.dumps(records).encode()),
            'reverse_ordered_adjoint_and_real_contact_polarization_exact': True,
            'native_ground_or_fullH_evolution': False}


def method_domains() -> dict[str, Any]:
    # l2({0,1,...}): L=diag(0,1,2,...), e_n=n^-2, Omega=e_0.
    # E=L+|e><Omega|+|Omega><e| is selfadjoint on D(L) by bounded perturbation.
    # E Omega=e, LE Omega=(0,1,1/2,...), so J Omega=-i LE Omega is not in D(L)=D(E).
    rows = []
    for k in (8, 32, 128):
        e_mass = sum((F(1, n**4) for n in range(1, k+1)), F(0))
        j_mass = sum((F(1, n*n) for n in range(1, k+1)), F(0))
        ej_graph2 = F(k)+sum((F(1, n**3) for n in range(1, k+1)), F(0))**2
        assert ej_graph2 >= k and e_mass < 2 and j_mass < 2
        # Cutoff triangle proves strong U_E(s)J Omega continuity, without J Omega in D(E).
        # Tail sum n>K n^-2 <=1/K, and ||E j_K||^2<=K+4.
        continuity_upper = (2/math.sqrt(k)+math.sqrt(k+4)/k)**2
        rows.append({'K': k, 'e_partial_mass': str(e_mass), 'j_partial_mass': str(j_mass),
                     'E_J_cutoff_graph_squared': str(ej_graph2),
                     's': str(F(1, k)), 'strong_group_J_difference_squared_upper_float': continuity_upper,
                     'current_tail_squared_exact_upper': str(F(1, k))})
    assert rows[-1]['strong_group_J_difference_squared_upper_float'] < rows[0]['strong_group_J_difference_squared_upper_float']
    # Small form convergence does not pay closed L graph convergence.
    form_graph = [{'K': k, 'norm_squared': str(F(1, k**3)),
                   'L_form': str(F(1, k)), 'L_graph_squared': str(F(k))}
                  for k in (2, 4, 8, 16)]  # psi_K=e_(K^2)/K^(3/2)
    omega: Any = sp.Matrix([1, 0])
    p: Any = sp.Matrix([[0, 1], [0, 0]])
    b: Any = sp.Matrix([[0, 1], [1, 0]])
    assert p*omega == b*p*omega == sp.zeros(2, 1)
    assert norm2(p*b*omega) == 1
    return {'scope': 'NON_NATIVE_INFINITE_DOMAIN_AND_EXACT_NULL_IDEAL_CONTROLS',
            'construction': 'E=L+rank_two(e,Omega), e_n=n^-2; J Omega=-i L e, n>=1',
            'e_in_D_L_and_Omega_in_D_E': True, 'J_Omega_in_D_L_or_D_E': False,
            'strong_U_E_on_J_without_generator_domain': rows,
            'small_form_not_graph': form_graph,
            'bounded_left_null_preservation_exact': True, 'two_sided_null_claim_rejected_squared': '1',
            'current_closure_scope': 'symmetric cutoff limits on dense words are closable; no essential selfadjointness claim'}


def method_contact_ground_compression() -> dict[str, Any]:
    root2: Any = sp.sqrt(2)
    omega: Any = sp.ones(4, 1)/2
    u: Any = sp.Matrix([1, -1, 0, 0])/root2
    v: Any = sp.Matrix([0, 0, 1, -1])/root2
    w: Any = sp.Matrix([1, 1, -1, -1])/2
    uv: Any = sp.Matrix.hstack(u, v)
    L: Any = uv*sp.Matrix([[2, 1+sp.I], [1-sp.I, 3]])*uv.conjugate().T
    eh: Any = sp.diag(1, -1, 0, 0)
    ek: Any = sp.diag(0, 0, 1, -1)
    assert eh*ek == ek*eh and L*omega == L*w == sp.zeros(4, 1)
    xh, xk = eh*omega, ek*omega
    m1hk: Any = (xh.conjugate().T*L*xk)[0]
    comm = lambda a, b: a*b-b*a
    khk: Any = -(comm(eh, comm(ek, L))+comm(ek, comm(eh, L)))/2
    khh: Any = -comm(eh, comm(eh, L))
    assert (omega.conjugate().T*khk*omega)[0] == 2*sp.re(m1hk) == 1
    assert (omega.conjugate().T*khh*omega)[0] == 2*(xh.conjugate().T*L*xh)[0]
    q0: Any = uv*uv.conjugate().T
    source: Any = omega+w+xh
    meanremoved: Any = source-omega*(omega.conjugate().T*source)[0]
    assert norm2(meanremoved-q0*source) == 1
    h: Any = sp.diag(0, 0, 0, 0, 0)
    h[2:, 2:] = sp.Matrix([[3, 1, 1], [1, 4, 2], [1, 2, 6]])
    xi: Any = sp.zeros(5, 2)
    xi[2, 0] = 1; xi[2, 1] = 1; xi[3, 1] = 2
    s: Any = xi.T*xi
    sqrt_s: Any = (s+2*sp.eye(2))/sp.sqrt(10)
    assert sqrt_s*sqrt_s == s
    W: Any = sp.simplify(xi*sqrt_s.inv())
    assert sp.simplify(W.T*W) == sp.eye(2)
    P: Any = W*W.T
    R: Any = sp.diag(0, 0, 1, 1, 1)-P
    M: Any = sp.simplify(W.T*h*W)
    B: Any = sp.simplify(R*h*W)
    bath: Any = sp.Matrix([B[4, :]])
    z: Any = sp.Symbol('z')
    sigma: Any = bath.T*bath/(z-6)
    gram_schur: Any = sqrt_s*(z*sp.eye(2)-M-sigma).inv()*sqrt_s
    direct: Any = xi.T*(z*sp.eye(5)-h).inv()*xi
    assert sp.simplify(gram_schur-direct) == sp.zeros(2)
    assert sp.simplify(sp.diff(sigma, z)+bath.T*bath/(z-6)**2) == sp.zeros(2)
    full_jj: Any = W.T*h*h*W
    orthogonal: Any = B.T*B
    assert norm2(full_jj-orthogonal) > 0
    return {'scope': 'NON_NATIVE_EXACT_COMMUTING_E_CONTACT_WHOLE_P0_AND_NONORTHOGONAL_SCHUR',
            'mixed_m1': str(m1hk), 'real_contact_Khk': '1', 'imaginary_cross_moment_retained': '1/2',
            'mean_vs_whole_P0_squared_defect': '1', 'source_Gram': str(s),
            'selfadjoint_finite_compression_Schur_exact': True,
            'orthogonal_force_vs_fullJJ_squared_defect': str(norm2(full_jj-orthogonal)),
            'residue_denominator_sign': 'I-Sigma_prime=I+B*(e-C)^-2 B',
            'real_isolated_atom_or_uniform_Gram_claim': False}


def method_ordered_limits_packets() -> dict[str, Any]:
    losses = []
    for n in (2, 4, 8, 16):
        vacuum: Any = sp.Matrix([1, 0])
        a: Any = sp.Matrix([sp.Rational(1, n), sp.sqrt(1-sp.Rational(1, n*n))])
        E: Any = n*a*a.T
        L: Any = sp.diag(0, sp.Rational(1, n*n))
        j: Any = -sp.I*L*E*vacuum
        assert norm2(E*vacuum) == 1 and norm2(j) == F(n*n-1, n**6)
        losses.append({'n': n, 'finite_E_source_mass': '1',
                       'fixed_s_group_difference_squared_upper': str(F(4, n*n)),
                       'J_source_mass': str(norm2(j)), 'ordered_limiting_E_source_mass': '0'})
    packets = []
    for k in (4, 16, 64):
        # Spectrum L_k={0,1/k}, Omega=(1,0), Q_k=diag(1,k^4).
        # At fixed t=1 the omitted rightmost propagation produces Q cost ~k^2/4.
        propagated_Q = 1+(k**4-1)*math.sin(1/(2*k))**2
        packets.append({'k': k, 'support_eta': str(F(1, k)), 'initial_Q': '1',
                        'rightmost_time': '1', 'propagated_Q_float': propagated_Q,
                        'Hilbert_change_norm_upper': str(F(1, k)),
                        'graph_norm_upper': str(F(1, k)),
                        'Q_change_squared_lower_float': k*k/math.pi**2})
    assert packets[-1]['propagated_Q_float'] > packets[0]['propagated_Q_float']
    return {'scope': 'NON_NATIVE_ORDERED_E_TAIL_LOSS_AND_PACKET_BOUNDARY_NOT_TARGET_COUNTEREXAMPLE',
            'electric_source_mass_loss_without_uniform_E_squared_tails': losses,
            'shrinking_packet_rightmost_propagation': packets,
            'accepted_parent_packet_correction_already_present': True,
            'exact_ground_U_t1_Omega_equals_Omega': True,
            'packet_required_fixed_s_residual': '(W_s-I)L Omega/(i s), bounded by 2 eta/abs(s)',
            'packet_contact_cross_terms': 'retain at fixed s, vanish as eta->0 before s->0',
            'universal_native_Q_constant_invented_for_method_family': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {key: sha(Path(row['path']).read_bytes()) for key, row in request['inputs'].items()}
    assert before == {key: row['sha256'] for key, row in request['inputs'].items()}
    with zipfile.ZipFile(request['profile_archive']) as archive:
        raw = archive.read(request['profile_member'])
        profiles = json.loads(raw)['primitive_profiles']
    with zipfile.ZipFile(request['snapshot']) as archive:
        frozen = {name: sha(archive.read(name)) for name in request['target_pins']}
        assert frozen == request['target_pins']
        literal_targets = {name: json.loads(archive.read(name)) for name in request['literal_targets']}
    receipt = {'schema': 1, 'kind': 'INDEPENDENT_ELECTRIC_HISTORY_SCIENTIFIC_REVIEW_CONTROLS_NOT_PHOTON',
               'frozen_snapshot_sha256': sha(Path(request['snapshot']).read_bytes()),
               'frozen_target_members_sha256': frozen, 'full_literal_target_proofs': literal_targets,
               'primitive_profile_member_sha256': sha(raw),
               'native_constants': native_constants(profiles),
               'native_literal_controls': [native_paths(n, profiles) for n in request['N']],
               'domain_null_current_controls': method_domains(),
               'contact_ground_compression_controls': method_contact_ground_compression(),
               'ordered_limits_packet_controls': method_ordered_limits_packets(),
               'source_input_sha256_before': before,
               'source_input_sha256_after': {key: sha(Path(row['path']).read_bytes()) for key, row in request['inputs'].items()},
               'scientific_claims': {'native_ground_computed': False, 'native_fullH_evolved': False,
                                     'conditional_analytic_theorem': True, 'photon': False, 'scientific_admission': False},
               'runtime': {'executable': sys.executable, 'elapsed_seconds': time.monotonic()-started}}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'N': request['N'], 'independent_domain_contact_compression_controls': True,
                      'native_ground_computed': False, 'photon': False}))


if __name__ == '__main__':
    main()
