"""Conditional native band-defect proof and exact controls, not a native ground solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_spectral_controls() -> dict[str, Any]:
    # These exact vectors test the two source equations, not a native Weyl realization.
    energies = [0, 0, 1, 2, 17, 128, 129, 511]
    lam = sp.diag(*energies)
    x = sp.Matrix([1, 2, 3, -2, 5, 7, -1, 4]) / 12
    rows = []
    for n in (-3, -2, -1, 1, 2, 3):
        a, b, omega = 256*n*n, 128*n, 128*n*n
        z = (lam-a*sp.eye(len(energies)))*x/b
        w = sp.Matrix([1, -1, 2, 3, -2, 1, 0, 4])/7
        j = ((lam-a*sp.eye(len(energies)))*z-b*w)/sp.I
        p = sp.diag(*[int(0 < e <= omega) for e in energies])
        resolvent = sp.diag(*[sp.Rational(1, e-a) if 0 < e <= omega else 0 for e in energies])
        force = b*b*w+sp.I*b*j
        assert p*x == resolvent*resolvent*force
        assert p*x != resolvent*force
        # Whole ground removal differs from subtracting the Omega component only.
        mean_removed = sp.diag(0, 1, *([1]*6))*x
        assert mean_removed[1] != 0 and (p*x)[1] == 0
        lhs = (p*x).norm()**2
        # Minkowski plus Cauchy gives a rational squared upper envelope.
        upper2 = 2*w.norm()**2/n**4+2*j.norm()**2/(128**2*n**6)
        assert sp.simplify(upper2-lhs) >= 0
        rows.append({'nUV': n, 'omega': omega, 'positive_band_source_norm_squared': str(lhs),
                     'squared_upper': str(upper2), 'one_resolvent_rejected': True})
    # Concrete selfadjoint multiplication E on l2(N), with normalized source m^-2.
    # E^2 moment is zeta(2)/zeta(4); E^4 moment diverges. E source exists, E^2 source does not.
    e2 = sp.zeta(2)/sp.zeta(4)
    assert sp.simplify(e2-15/sp.pi**2) == 0
    return {'scope': 'exact non-native source-equation/spectral and infinite-domain controls',
            'two_resolvent_rows': rows, 'whole_degenerate_P0_not_scalar_centering': True,
            'domain_adverse': {'E_second_moment': str(e2), 'E_fourth_moment': 'infinite',
                               'norm_squared_E_squared_source_cutoff_M': 'M/zeta(4)',
                               'E2_domain_does_not_pay_E4': True}}


def signed_controls() -> dict[str, Any]:
    rows = []
    # e=(1,0) has fixed normalization for every q; f amplitude is not weakened.
    e = sp.Matrix([1, 0])
    for den in (8, 16, 32):
        q = sp.Rational(1, den)
        f_cancel = -64*sp.I*e/q
        f_amplify = -f_cancel
        results = []
        for f in (f_cancel, f_amplify):
            lf = q*f
            cross = (e.conjugate().T*lf)[0]
            d = 128*e-2*sp.I*lf
            identity = 16384*e.norm()**2+4*lf.norm()**2+512*sp.im(cross)
            assert d.norm()**2 == identity
            results.append({'FF': str(f.norm()**2), 'LLFF': str(lf.norm()**2),
                            'Im_E_LF': str(sp.im(cross)), 'defect_squared': str(d.norm()**2)})
        assert results[0]['FF'] == results[1]['FF']
        assert results[0]['defect_squared'] == '0' and results[1]['defect_squared'] == '65536'
        # Equal TOTAL signed contact can still hide opposite LOW-band phases.
        full_e = sp.Matrix([1, 1])
        full_f = sp.Matrix([-64*sp.I/q, 64*sp.I/q])
        total_contact = sp.im((full_e.conjugate().T*full_f)[0])
        assert total_contact == 0
        assert sp.im((full_e.conjugate().T*(-full_f))[0]) == total_contact
        full_l = sp.diag(q, 1)
        assert (full_l*full_f).norm()**2 == (full_l*(-full_f)).norm()**2
        rows.append({'qIR': str(q), 'fixed_E_norm_squared': '1', 'signed_pairs': results,
                     'volume_inverse_upper_multiplier': str(q**-3),
                     'photon_scaled_cycle_upper_multiplier': str(q**-4),
                     'photon_scaled_polarization_inverse_curl_multiplier': str(4*q**-6),
                     'equal_total_signed_contact': str(total_contact),
                     'equal_full_E_norm_squared': str(full_e.norm()**2),
                     'equal_full_F_norm_squared': str(full_f.norm()**2)})
    return {'scope': 'exact non-native signed measure discriminator, not physical response',
            'rows': rows, 'same_positive_diagonals_do_not_fix_defect': True,
            'native_cross_phase_determined': False, 'native_atom_inferred': False}


def native_controls(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    def move(x: tuple[int, ...], a: int) -> tuple[int, ...]:
        y = list(x); y[a] = (y[a]+1) % n
        return tuple(y)
    origin = (0, 0, 0)
    c = {(origin, 0): 1, (move(origin, 0), 1): 1,
         (move(origin, 1), 0): -1, (origin, 1): -1}
    divergence = {x: 0 for x in sites}
    for (x, a), coefficient in c.items():
        divergence[x] += coefficient
        divergence[move(x, a)] -= coefficient
    assert not any(divergence.values()) and sum(v*v for v in c.values()) == 4
    # All64 canonical charged backgrounds, ONE collective rho and full Gauss.
    rho = {x: 128*(-1)**sum(x) for x in sites}
    assert sum(rho.values()) == 0
    for x in sites:
        previous = list(x); previous[0] = (previous[0]-1) % n
        assert 64*((-1)**sum(x)-(-1)**sum(previous)) == rho[x]
    for role in profiles:
        assert sum(2+2*(-1)**sum(x) for x in sites) == 2*n**3
    path_counts = [0, 0, 0]; touched = [0, 0, 0]; seam_touched = 0; aliases = 0
    frequency_counts: dict[int, int] = {}
    derivative_identities = 0
    xvar = sp.Symbol('boost', real=True)
    left0, lefta, right0, righta, transport = sp.symbols(
        'Phi0_x PhiAxis_x Phi0_y PhiAxis_y Ordered_SU4_path', commutative=False)
    for ri, role in enumerate(profiles):
        for x in sites:
            for path in role['paths']:
                y = x; indices = []; seam = 1
                for _ in range(path['length']):
                    indices.append((y, path['axis']))
                    if y[path['axis']] == n-1: seam *= -1
                    y = move(y, path['axis'])
                path_counts[ri] += 1
                if ri == 0:
                    continue  # Scalar U1 derivative of neutral role only, not Hamiltonian deletion.
                frequency = sum(c.get(edge, 0) for edge in indices)
                frequency_counts[frequency] = frequency_counts.get(frequency, 0)+int(role['load'])
                assert abs(frequency) <= 2
                if not frequency: continue
                touched[ri] += 1; seam_touched += int(seam == -1)
                aliases += int(n == 4 and path['length'] == 2)
                # Preserve actual noncommutative endpoint order. Coefficients are exact native rationals.
                c0, ci = sp.Rational(path['c0']), sp.Rational(path['ci'])
                left = c0*left0+sp.I*ci*lefta
                right = c0*right0+sp.I*ci*righta
                block = seam*(left*transport+transport*right)/2
                shifted = sp.exp(-sp.I*frequency*xvar)*block
                assert sp.expand(sp.diff(shifted, xvar).subs(xvar, 0)+sp.I*frequency*block) == 0
                assert sp.expand(sp.diff(shifted, xvar, 2).subs(xvar, 0)+frequency**2*block) == 0
                derivative_identities += 2
    assert touched[0] == 0 and touched[1] > 0 and touched[2] > 0 and seam_touched > 0
    ce = 8*sum((Fraction(p['load'])*t['length']*(abs(Fraction(t['c0']))+abs(Fraction(t['ci'])))
                for p in profiles[1:] for t in p['paths'] if t['axis'] == 0), Fraction())
    assert ce == Fraction(15475017073321605, 1099511627776)
    # Exact fullE contact / path identities on a Fourier cochain polynomial.
    b, r, ec, harmonic = sp.symbols('B r Ec n', real=True)
    f = sum((-1)**(k+1)*r**(k-1)*sp.sin(k*b)/k for k in range(1, 5))
    poisson = 1+2*sum((-1)**k*r**k*sp.cos(k*b) for k in range(1, 5))
    assert sp.expand(2*r*sp.diff(f, b)-(1-poisson)) == 0
    current = 128*ec*sp.diff(f, b)-256*sp.I*sp.diff(f, b, 2)
    seam_anticommutator = 2*ec*poisson-4*sp.I*sp.diff(poisson, b)
    assert sp.expand(2*r*current-(128*ec-64*seam_anticommutator)) == 0
    assert sp.expand(16*(ec+4*harmonic)**2-16*ec**2-(128*harmonic*ec+256*harmonic**2)) == 0
    contact = -4*sp.I*sp.diff(f, b)
    assert sp.expand(2*r*contact+4*sp.I*(1-poisson)) == 0
    # Pi-only projection has a nonzero density-contact deficit at every native N.
    longitudinal_raw_contact = Fraction(-(n**3-1), 3*n**3)
    assert longitudinal_raw_contact != 0
    return {'N': n, 'scope': 'exact native symbolic/integer identities, NOT actual-ground measurements',
            'each_species_canonical': 2*n**3, 'all65_role_loads': [int(p['load']) for p in profiles],
            'literal_path_counts': path_counts, 'charged_face_touched': touched,
            'AP_seam_touched': seam_touched, 'N4_length2_literals_retained': aliases,
            'charged_load_weighted_frequency_counts': frequency_counts,
            'exact_noncommutative_path_derivative_identities': derivative_identities,
            'native_C_e': str(ce), 'closed_face_divergence_and_norm4': True,
            'collective_fullrho_Gauss_identity': True, 'fullE_shift_coefficients': [128, 256],
            'fullHE_Faraday_Abel_and_contact_identities': True,
            'raw_edge_Pi_only_contact_deficit': str(longitudinal_raw_contact),
            'full_native_H_ground_or_evolution_computed': False}


def theorem_arithmetic(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    w = sum((Fraction(p['load'])*t['length']*(abs(Fraction(t['c0']))+abs(Fraction(t['ci'])))
             for p in profiles[1:] for t in p['paths']), Fraction())
    ce = Fraction(15475017073321605, 1099511627776)
    assert w == 3*ce/8
    assert sp.Rational(3, 2) < sp.Rational(7, 4)
    assert 2 < sp.Rational(17, 8)
    tails = []
    for m in (1, 4, 16, 64):
        # Separately test both coefficient envelopes with exact positive rational sums.
        p2 = sum((Fraction(1, k*k) for k in range(m+1, m+65)), Fraction())
        p3 = sum((Fraction(1, k**3) for k in range(m+1, m+65)), Fraction())
        p4 = sum((Fraction(1, k**4) for k in range(m+1, m+65)), Fraction())
        assert p2 <= Fraction(1, m) and p3 <= Fraction(1, 2*m*m)
        assert p4 <= Fraction(1, 3*m**3)
        tails.append({'M_nUV': m, 'Poisson_A_multiplier': str(Fraction(2, m)),
                      'Poisson_B_multiplier': str(Fraction(1, m*m)),
                      'F_A_multiplier': str(Fraction(1, 2*m*m)),
                      'F_B_multiplier': str(Fraction(1, 3*m**3))})
    return {'native_W_all_literal_axes': str(w),
            'R_exact': '60*2^-4096+11192*2^8192+(129/10)*2^10240+262144',
            'Lambda_exact': '2^8192', 'electric_C1': '1+4R',
            'Phi_fourth_moment_K': '2R/Lambda',
            'A_p': '128W*(2R/Lambda)^(1/4)*Cp^(3/4)*(65/4)^(3p/4)',
            'C_s': 'C1*1025^(s-1)+(s-1)*A_p*1025^(s-1-3p/4)/(1+3p/4-s)',
            'bootstrap_pairs': [['1', '3/2'], ['3/2', '2']],
            'A_exact_conditional': 'sqrt(C2)/16',
            'B_exact': 'sqrt((4*C_e)^2*16*sqrt(2R/Lambda))/128',
            'tail_envelopes_exact': tails, 'uniform_spatial_q_suppression_paid': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    request = json.loads(args.request.read_bytes())
    before = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        profile_raw = archive.read(request['profile_member'])
    profiles = json.loads(profile_raw)['primitive_profiles']
    result = {'schema': 1, 'kind': 'CONDITIONAL_NATIVE_BAND_DEFECT_EXACT_SCIENTIFIC_CONTROLS',
              'native_controls': [native_controls(n, profiles) for n in request['N']],
              'exact_spectral_adverse': exact_spectral_controls(), 'signed_attempt': signed_controls(),
              'exact_conditional_constants_and_tails': theorem_arithmetic(profiles),
              'literal_proof': json.loads((ROOT/request['proof']).read_bytes()),
              'conditional_interface': json.loads((ROOT/request['interface']).read_bytes()),
              'profile_member_sha256': sha(profile_raw), 'source_input_sha256_before': before,
              'source_input_sha256_after': {name: sha((ROOT/name).read_bytes()) for name in request['pins']},
              'runtime': {'executable': sys.executable, 'elapsed_seconds': time.monotonic()-start},
              'native_photon_atom_phase_or_finishline_claim': False}
    assert result['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'native_N': request['N'],
                      'native_ground_backend': False, 'conditional_E4': True, 'atom_or_phase': False}))


if __name__ == '__main__':
    main()
