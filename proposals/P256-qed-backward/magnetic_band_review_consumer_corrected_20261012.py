"""Independent native factor algebra and nonnative method falsifiers, not a ground solver."""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import sympy as s

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def audit_originals(request: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audits = []
    profiles = []
    for epoch in request['original_epochs']:
        cap = json.loads((ROOT/epoch['capsule']).read_bytes())
        with zipfile.ZipFile(ROOT/cap['preexecution']['path']) as pre, zipfile.ZipFile(ROOT/cap['executed']['path']) as exe:
            assert digest((ROOT/cap['preexecution']['path']).read_bytes()) == cap['preexecution']['sha256']
            assert digest((ROOT/cap['executed']['path']).read_bytes()) == cap['executed']['sha256']
            members = {}
            for name, expected in cap['source_input_sha256_before'].items():
                member = cap['archive_member_map'][name]
                raw = pre.read(member)
                assert digest(raw) == expected == cap['source_input_sha256_after'][name]
                assert raw == exe.read(member)
                members[member] = digest(raw)
            req_name = epoch['request']
            req_raw = pre.read(req_name)
            assert req_raw == exe.read(req_name)
            assert digest(req_raw) == cap['request_sha256_before'] == cap['request_sha256_after']
            original_request = json.loads(req_raw)
            assert original_request['pins'] == cap['source_input_sha256_before']
            outputs = {}
            for name, expected in cap['outputs'].items():
                assert digest(exe.read(name)) == expected
                outputs[name] = expected
            receipt = json.loads(exe.read(epoch['receipt']))
            exit_data = json.loads(exe.read(epoch['exit']))
            assert exit_data == cap['runtime'] and exit_data['returncode'] == 0 and exit_data['natural_exit'] is True
            assert cap['sources_inputs_unchanged'] is True
            if not profiles:
                nested_raw = exe.read(original_request['profile_archive'])
                with zipfile.ZipFile(io.BytesIO(nested_raw)) as nested:
                    profile_raw = nested.read(original_request['profile_member'])
                assert digest(profile_raw) == receipt['profile_member_sha256']
                profiles = json.loads(profile_raw)['primitive_profiles']
            audits.append({'epoch': cap['epoch'], 'source_member_hashes': members, 'request_sha256': digest(req_raw),
                           'output_member_hashes': outputs, 'reported_runtime_bound_to_archived_exit': exit_data,
                           'original_request': original_request, 'all_snapshots_equal_pre_and_executed': True,
                           'original_consumer_executed_by_reviewer': False})
    assert [p['load'] for p in profiles] == [1, 1, 63]
    return {'scope': 'read-only independent immutable epoch audit', 'epochs': audits}, profiles


def move(x: tuple[int, ...], axis: int, n: int) -> tuple[int, ...]:
    y = list(x)
    y[axis] = (y[axis]+1) % n
    return tuple(y)


def native_factors(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    origin = (0, 0, 0)
    c = {(origin, 0): 1, (move(origin, 0, n), 1): 1,
         (move(origin, 1, n), 0): -1, (origin, 1): -1}
    divergence = {x: 0 for x in sites}
    for (x, axis), v in c.items():
        divergence[x] += v
        divergence[move(x, axis, n)] -= v
    assert not any(divergence.values()) and sum(v*v for v in c.values()) == 4
    # Actual collective64 canonical density; longitudinal fullE is a gradient.
    parity = {x: (-1)**sum(x) for x in sites}
    occupations = {x: 2+2*parity[x] for x in sites}
    assert sum(occupations.values()) == 2*n**3
    rho = {x: 64*(occupations[x]-2) for x in sites}
    longitudinal = {(x, a): F(64, 3)*parity[x] for x in sites for a in range(3)}
    assert all(sum(longitudinal[(x, a)]-longitudinal[(move(x, a, n), a)] for a in range(3)) == rho[x] for x in sites)
    assert sum(v*longitudinal[e] for e, v in c.items()) == 0
    t, harmonic = s.symbols('t nUV', real=True)
    transverse = {e: t*c.get(e, 0) for e in longitudinal}
    full_e = {e: transverse[e]+s.Rational(v.numerator, v.denominator) for e, v in longitudinal.items()}
    kinetic = 64*sum(v*v for v in full_e.values())
    split = 64*sum(v*v for v in transverse.values())+64*sum(s.Rational(v.numerator, v.denominator)**2 for v in longitudinal.values())
    assert s.expand(kinetic-split) == 0
    shifted = 64*sum((v+harmonic*c.get(e, 0))**2 for e, v in full_e.items())
    ec = sum(v*full_e[e] for e, v in c.items())
    assert s.expand(shifted-kinetic-128*harmonic*ec-256*harmonic**2) == 0
    assert s.expand(shifted-kinetic-64*harmonic*ec-128*harmonic**2) != 0
    # Noncommuting exact SU4 matrices, not a commuting scalar replacement.
    permutation = s.zeros(4)
    for j in range(4):
        permutation[(j+1) % 4, j] = 1
    su4 = permutation*s.diag(-1, 1, 1, 1)
    diagonal = s.diag(-1, -1, 1, 1)
    assert su4.det() == diagonal.det() == 1
    assert su4.H*su4 == diagonal.H*diagonal == s.eye(4)
    assert su4*diagonal != diagonal*su4
    def link(x: tuple[int, ...], axis: int) -> Any:
        return su4**((sum(x)+axis) % 4)*diagonal**((x[axis]+axis) % 2)
    def gauge(x: tuple[int, ...]) -> Any:
        return diagonal**(sum(x) % 2)*su4**((x[0]+2*x[1]+x[2]) % 4)
    def phi(x: tuple[int, ...], mu: int) -> Any:
        base = s.diag(*(s.Rational((j+1)*(mu+1)+sum(x), 7) for j in range(4)))
        off = s.zeros(4)
        for j in range(3):
            off[j, j+1] = off[j+1, j] = s.Rational(mu+1, 11)
        return base+off
    counts = [0, 0, 0]
    affected = [0, 0, 0]
    derivative_count = 0
    ap_affected = 0
    n4_literals = 0
    frequencies: dict[int, int] = {}
    literal_records = []
    selected_block = None
    for role_index, role in enumerate(profiles):
        for x in sites:
            for row_index, row in enumerate(role['paths']):
                y = x
                edges = []
                ap = 1
                for _ in range(row['length']):
                    edges.append((y, row['axis']))
                    if y[row['axis']] == n-1:
                        ap *= -1
                    y = move(y, row['axis'], n)
                counts[role_index] += 1
                frequency = 0 if role_index == 0 else sum(c.get(e, 0) for e in edges)
                frequencies[frequency] = frequencies.get(frequency, 0)+int(role['load'])
                literal_records.append([role_index, list(x), row_index, list(y), ap, frequency])
                assert abs(frequency) <= 2
                if frequency == 0:
                    continue  # derivative zero; original literal is still counted and hashed.
                affected[role_index] += 1
                ap_affected += int(ap == -1)
                n4_literals += int(n == 4 and row['length'] == 2)
                transport = s.eye(4)
                for tail, axis in edges:
                    transport = transport*link(tail, axis)
                c0, ci = s.Rational(row['c0']), s.Rational(row['ci'])
                fx = c0*phi(x, 0)+s.I*ci*phi(x, row['axis']+1)
                fy = c0*phi(y, 0)+s.I*ci*phi(y, row['axis']+1)
                block = ap*(fx*transport+transport*fy)/2
                assert block != s.zeros(4)
                gx, gy = gauge(x), gauge(y)
                transformed = ap*((gx*fx*gx.H)*(gx*transport*gy.H)+(gx*transport*gy.H)*(gy*fy*gy.H))/2
                assert transformed == gx*block*gy.H
                angle = s.Symbol('s', real=True)
                forward = s.exp(-s.I*frequency*angle)*block
                reverse = s.exp(s.I*frequency*angle)*block.H
                assert forward.H == reverse
                assert forward.diff(angle).subs(angle, 0) == -s.I*frequency*block
                assert reverse.diff(angle).subs(angle, 0) == s.I*frequency*block.H
                assert forward.diff(angle, 2).subs(angle, 0) == -frequency**2*block
                assert reverse.diff(angle, 2).subs(angle, 0) == -frequency**2*block.H
                derivative_count += 4
                selected_block = block
    assert affected[0] == 0 and affected[1:] == [12, 12] and ap_affected == 8
    assert n4_literals == (16 if n == 4 else 0)
    assert selected_block is not None
    ce = 8*sum((F(role['load'])*row['length']*(abs(F(row['c0']))+abs(F(row['ci'])))
                 for role in profiles[1:] for row in role['paths'] if row['axis'] == 0), F())
    w = sum((F(role['load'])*row['length']*(abs(F(row['c0']))+abs(F(row['ci'])))
             for role in profiles[1:] for row in role['paths']), F())
    assert ce == F(15475017073321605, 1099511627776) and w == 3*ce/8
    # Full four-colour CAR bilinear in an actual two-site canonical4 sector.
    masks = [mask for mask in range(256) if mask.bit_count() == 4]
    index = {mask: k for k, mask in enumerate(masks)}
    entries = {}
    for mask in masks:
        for a in range(4):
            for b in range(4):
                source, target = b+4, a
                if not (mask >> source) & 1 or (mask >> target) & 1:
                    continue
                removed = mask ^ (1 << source)
                sign = (-1)**((mask & ((1 << source)-1)).bit_count()+(removed & ((1 << target)-1)).bit_count())
                new = removed | (1 << target)
                entries[(index[new], index[mask])] = sign*selected_block[a, b]
    hopping = s.SparseMatrix(len(masks), len(masks), entries)
    number_x = s.diag(*[(mask & 15).bit_count() for mask in masks])
    assert number_x*hopping-hopping*number_x == hopping
    assert hopping.H != hopping and hopping != s.zeros(len(masks))
    b, r, ee = s.symbols('B r E_c', real=True)
    f = sum((-1)**(k+1)*r**(k-1)*s.sin(k*b)/k for k in range(1, 7))
    poisson = 1+2*sum((-1)**k*r**k*s.cos(k*b) for k in range(1, 7))
    assert s.expand(2*r*s.diff(f, b)-1+poisson) == 0
    dotf = 128*ee*s.diff(f, b)-256*s.I*s.diff(f, b, 2)
    anticomm = 2*ee*poisson-4*s.I*s.diff(poisson, b)
    assert s.expand(2*r*dotf-128*ee+64*anticomm) == 0
    assert s.expand(2*r*(-4*s.I*s.diff(f, b))+4*s.I*(1-poisson)) == 0
    return {'N': n, 'scope': 'native complete rational factor/geometry/CAR algebra, NOT minimizing ground/fullH dynamics',
            'roles': [p['role'] for p in profiles], 'loads': [p['load'] for p in profiles],
            'each_species_canonical': 2*n**3, 'collective_rho_used_once': True,
            'closed_cycle_affine_Gauss_and_fullHE_HC_split': True, 'shift_coefficients': [128, 256],
            'HE_half_coefficient_mutation_rejected': True, 'literal_path_counts': counts,
            'affected_literal_paths': affected, 'AP_affected': ap_affected, 'N4_length2_separate_literals': n4_literals,
            'all65_load_weighted_frequencies': frequencies, 'literal_enumeration_sha256': digest(json.dumps(literal_records).encode()),
            'noncommuting_SU4_order_and_covariance_checked': True, 'exact_forward_reverse_derivatives': derivative_count,
            'canonical_four_colour_CAR_bilinear_dimension': len(masks), 'Ce': str(ce), 'W': str(w),
            'fullE_Faraday_contact_and_first_harmonic_one': True,
            'unchanged_Phi_Jordan_Haar_colour_onsite_neutral_Ref_terms': 'scalar U commutes these factors; retained in analytic H, not numerically evolved'}


def spectral_controls() -> dict[str, Any]:
    energies = [0, 0, s.Rational(1, 3), 19, 128, 129, 600]
    lam = s.diag(*energies)
    x = s.Matrix([2, 3, -1, 4, 2, -3, 1])/11
    w = s.Matrix([1, -2, 3, 1, -1, 2, 4])/9
    rows = []
    for n in (-4, -1, 1, 4):
        a, b, omega = 256*n*n, 128*n, 128*n*n
        z = (lam-a*s.eye(7))*x/b
        j = ((lam-a*s.eye(7))*z-b*w)/s.I
        p = s.diag(*[int(bool(0 < e <= omega)) for e in energies])
        resolvent = s.diag(*[s.Rational(1)/(e-a) if 0 < e <= omega else 0 for e in energies])
        force = b*b*w+s.I*b*j
        assert p*x == resolvent**2*force
        assert p*x != resolvent*force and p*x != resolvent**2*(b*b*w-s.I*b*j)
        assert max(abs(resolvent[k, k]) for k in range(7)) <= s.Rational(1, 128*n*n)
        lhs = s.simplify((p*x).norm()**2)
        upper = 2*w.norm()**2/n**4+2*j.norm()**2/(128**2*n**6)
        assert s.simplify(upper-lhs) >= 0
        assert x[1] != 0 and (p*x)[1] == 0
        rows.append({'nUV': n, 'omega': omega, 'source_squared_norm': str(lhs),
                     'two_source_equations_exact': True, 'one_resolvent_and_wrong_J_sign_rejected': True,
                     'squared_Minkowski_upper': str(upper)})
    # Projection is essential at resonance; a global inverse need not exist.
    resonance = s.diag(0, 128, 256)
    p = s.diag(0, 1, 0)
    bounded_r = s.diag(0, -s.Rational(1, 128), 0)
    assert (resonance-256*s.eye(3)).det() == 0
    assert bounded_r*(resonance-256*s.eye(3)) == p
    return {'scope': 'nonnative exact source-equation method controls; no native Weyl/ground realization',
            'rows': rows, 'entire_degenerate_P0_removed_not_mean_centered': True,
            'positive_band_inverse_exists_despite_global_resonance': True,
            'finite_positive_level_collapse': {'L_N': 'diag(0,1/N)', 'source': '(0,1)', 'finite_excitation_mass': 1,
                                                'limiting_wholeP0_excitation_mass': 0}}


def domain_controls() -> dict[str, Any]:
    assert s.simplify(s.zeta(2)/s.zeta(4)-15/s.pi**2) == 0
    rows = []
    for m in (4, 16, 64):
        # E multiplies by k. J_k=1/k is Hilbert but neither EJ nor LJ for L=k.
        head_tail = F(1, m)+F(4, m)
        assert head_tail == F(5, m)
        # At k=m^2, psi=e_k/m^3, L=k^2: small form, divergent graph.
        norm2, form, graph2 = F(1, m**6), F(1, m**2), m**2
        assert norm2 < form < 1 < graph2
        # Uniform fourth moment does not preserve direct E^2-source Gram.
        weight = F(1, m**4)
        e2, e4 = weight*m*m, weight*m**4
        assert e2 == F(1, m*m) and e4 == 1
        rows.append({'M': m, 'W_1_over_M_J_strong_error_squared_upper': str(head_tail),
                     'J_E_derivative_squared_partial_norm': m,
                     'small_form_example_norm2': str(norm2), 'small_form_example_form': str(form),
                     'small_form_example_L_graph_norm2': graph2,
                     'escaping_E_moment2': str(e2), 'escaping_E_moment4': e4})
    return {'scope': 'nonnative unbounded selfadjoint multiplication/closed-domain adverse witnesses',
            'E_second_not_fourth_moment': {'Omega_k': 'k^-2/sqrt(zeta(4))', 'E': 'diag(k)',
                                          'E_second_moment': '15/pi^2', 'E_fourth_moment': 'infinite'},
            'E4_not_E6': {'Omega_k': 'k^-3/sqrt(zeta(6))', 'E_fourth_moment': 'zeta(2)/zeta(6)',
                           'E_sixth_moment': 'infinite'},
            'J_not_in_D_E_or_D_L_but_W_strong': {'J_k': '1/k', 'E_and_L': 'diag(k)',
                                                'J_norm2': 'zeta(2)', 'EJ_and_LJ_norm2': 'infinite',
                                                'head_tail_bound': '|| (W_s-I)J ||^2 <= s^2 M+4/M'},
            'small_form_not_closed_L_graph': 'L=diag(k^2), psi_M=e_(M^2)/M^3',
            'uniform_E4_preserves_E_source_not_E_squared_source': 'mu_M=(1-M^-4)delta0+M^-4deltaM; bounded quotient limit is0, m4=1',
            'rows': rows, 'second_quotient_Fatou_symbol': '0<=4sin^2(se/2)/s^2<=e^2; square<=e^4',
            'ordered_source_graph_required': 'fixed-s cutoff and fixed-s t->0, THEN s->0 with both vector and L-image convergence'}


def principal_and_phase_controls() -> dict[str, Any]:
    r = s.Symbol('r', positive=True)
    # At angle pi/2 the exact Abel sum is atan(r)/r, not twice this source.
    assert s.limit(s.atan(r)/r, r, 1, dir='-') == s.pi/4
    assert s.pi/4 != s.pi/2
    exact_radius = 1-F(1, 2**4101)
    assert exact_radius < 1 and float(exact_radius) == 1.0
    rows = []
    e = s.Matrix([1, 0])
    for den in (8, 16, 32):
        q = s.Rational(1, den)
        signed = []
        for sign in (-1, 1):
            f = sign*s.I*e
            lf = q*f
            cross = (e.H*lf)[0]
            defect = 128*e-2*s.I*lf
            gram = 16384*e.norm()**2+4*lf.norm()**2+512*s.im(cross)
            assert s.simplify(defect.norm()**2-gram) == 0
            signed.append({'Im_E_LF': str(s.im(cross)), 'FF': '1', 'defect_norm': str(128+2*sign*q)})
        bad = -64*s.I*e/q
        assert bad.norm()**2 > s.Rational(22, 7)**2
        weak_lower = 128-s.Rational(44, 7)*q
        half_principal_lower = 128-s.Rational(22, 7)*q
        assert min(128-2*q, 128+2*q) >= half_principal_lower >= weak_lower > 0
        full_e = s.Matrix([1, 1])
        full_f = s.Matrix([-s.I, s.I])
        assert full_f.norm()**2 == 2 < s.Rational(3, 2)**2
        assert s.im((full_e.H*full_f)[0]) == s.im((full_e.H*(-full_f))[0]) == 0
        lam = s.diag(q, 1)
        assert (lam*full_f).norm()**2 == (lam*(-full_f)).norm()**2
        assert s.im((s.Matrix([1, 0]).H*lam*full_f)[0]) != s.im((s.Matrix([1, 0]).H*lam*(-full_f))[0])
        rows.append({'qIR': str(q), 'fixed_E_norm2': '1', 'bounded_half_principal_phase_pairs': signed,
                     'actual1_local_cancellation_FF': str(bad.norm()**2), 'actual1_local_realization_rejected': True,
                     'requested_piS_lower_using_pi_upper': str(weak_lower),
                     'half_principal_sharper_lower_using_pi_upper': str(half_principal_lower),
                     'equal_total_contact_and_full_positive_diagonals_do_not_fix_lowband_phase': True})
    packet_rows = []
    for m in (4, 16, 64):
        # These are actual spatial coefficient arrays, not inferred annulus multipliers.
        concentrated = [F(1)]+[F(0)]*(m*m-1)
        spread = [F(1, m)]*(m*m)
        assert sum(v*v for v in concentrated) == sum(v*v for v in spread) == 1
        assert sum(abs(v) for v in concentrated) == 1 and sum(abs(v) for v in spread) == m
        packet_rows.append({'spatial_support': m*m, 'both_l2_norm2': '1', 'actual_l1_concentrated': '1',
                            'actual_l1_spread': str(m), 'cauchy_bound_saturated_by_spread': True})
    # Diffuse energy measure on(0,1), finite moments and band weight but no atoms.
    energy = s.Symbol('epsilon', positive=True)
    assert s.integrate(energy, (energy, 0, 1)) == s.Rational(1, 2)
    assert s.integrate(energy**2, (energy, 0, 1)) == s.Rational(1, 3)
    assert s.integrate((1/(energy+1))**2, (energy, 0, 1)) == s.Rational(1, 2)
    return {'scope': 'nonnative bounded-source signed/phase/spatial/continuum adverses with native norm constraint',
            'normalization': {'first_harmonic': '1 for every exact r<1', 'F': 'principal_angle(U)/2',
                               'operator_norm_upper': 'pi/2', 'full_principal_angle_norm_upper': 'pi',
                               'angle_pi_over2_Abel_limit': 'pi/4', 'doubling_source_rejected': True,
                               'exact_radius_vs_binary64_adverse': True},
            'rows': rows, 'spatial_coefficients': packet_rows,
            'no_atom_from_filter': {'measure': 'Lebesgue on(0,1)', 'm0': '1', 'm1': '1/2', 'm2': '1/3',
                                   'filter': '1/(epsilon+1)', 'filtered_mass': '1/2', 'atomic_mass': '0'},
            'nUV_distinct_from_qIR': True, 'native_lowband_measure_or_phase_computed': False}


def tail_controls() -> list[dict[str, Any]]:
    rows = []
    for m in (1, 3, 12, 48):
        bounds = {k: F(1, (k-1)*m**(k-1)) for k in (2, 3, 4)}
        for k, bound in bounds.items():
            finite_sum = sum((F(1, j**k) for j in range(m+1, m+257)), F())
            assert finite_sum <= bound
        rows.append({'M_nUV': m, 'Poisson_A': str(2*bounds[2]), 'Poisson_B': str(2*bounds[3]),
                     'normalized_F_A': str(bounds[3]), 'normalized_F_B': str(bounds[4]),
                     'infinite_tail_justification': 'positive decreasing integral comparison, not extrapolation of finite sum'})
    return rows

def native_constant_controls() -> dict[str, Any]:
    ce = F(15475017073321605, 1099511627776)
    ratio = 2*(F(60, 2**12288)+11192+F(129, 10)*2**2048+F(262144, 2**8192))
    assert ce**4*ratio < 2**2110
    assert F(3, 2) < F(7, 4) and 2 < F(17, 8)
    assert 1+F(3, 4)-F(3, 2) == F(1, 4)
    assert 1+F(9, 8)-2 == F(1, 8)
    return {'scope': 'native rational constants, conditional E4 premise not numerically measured',
            'QJ_face': '(4Ce)^2*16*sqrt(2R/Lambda)',
            'QJ_face_less_than': '256*2^1055', 'QJ_upper_exact_squared_comparison': True,
            'A': 'sqrt(CE4)=sqrt(C2)/16', 'B': 'sqrt(QJ_face)/128',
            'strict_bootstrap_denominators': ['1/4', '1/8'],
            'optional_parent_sharp_budget': 'not adopted here; sibling independent reset review owns it'}



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    request = json.loads(args.request.read_bytes())
    before = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    assert all(before[name] == expected for name, expected in request['frozen_target_pins'].items())
    audit, profiles = audit_originals(request)
    result = {'schema': 1, 'kind': 'INDEPENDENT_MAGNETIC_BAND_LITERAL_REVIEW_EXACT_DISCRIMINATING_CONSUMER_NOT_ADMISSION',
              'original_immutable_audit': audit, 'native_complete_factor_checks': [native_factors(n, profiles) for n in request['N']],
              'nonnative_unbounded_domain_adverses': domain_controls(), 'nonnative_spectral_two_resolvent_wholeP0': spectral_controls(),
              'nonnative_bounded_norm_phase_and_atom_adverses': principal_and_phase_controls(),
              'projected_source_tail_arithmetic': tail_controls(),
              'native_conditional_constant_arithmetic': native_constant_controls(),
              'literal_review_proof': json.loads((ROOT/request['proof']).read_bytes()),
              'literal_original_targets': {name: json.loads((ROOT/name).read_bytes()) for name in request['literal_targets']},
              'input_sha256_before': before, 'input_sha256_after': {name: digest((ROOT/name).read_bytes()) for name in request['pins']},
              'runtime': {'elapsed_seconds': time.monotonic()-start, 'sympy_version': s.__version__},
              'native_numerical_ground_or_fullH_backend': False, 'scientific_admission': False,
              'phase_IR_atom_and_campaign_complete': False}
    assert result['input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'native_factor_N': request['N'],
                      'independent_nonnative_adverses': True, 'scientific_admission': False}))


if __name__ == '__main__':
    main()
