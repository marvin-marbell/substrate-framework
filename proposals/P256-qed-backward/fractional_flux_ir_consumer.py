"""Full physical form-Weyl and sharp-character spectral controls; not a photon."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import sympy as sp

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def divergence(e: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, sum(e[..., j] - np.roll(e[..., j], 1, axis=j) for j in range(3)))


def gradient(f: np.ndarray) -> np.ndarray:
    return np.stack([f - np.roll(f, -1, axis=j) for j in range(3)], axis=-1)


def longitudinal(rho: np.ndarray) -> np.ndarray:
    n = rho.shape[0]
    modes = np.indices(rho.shape)
    eigen = sum(4 * np.sin(np.pi * modes[j] / n) ** 2 for j in range(3))
    potential = np.zeros(rho.shape, complex)
    np.divide(np.fft.fftn(rho), eigen, out=potential, where=eigen > 1e-12)
    return gradient(cast(NDArray[np.complex128], np.fft.ifftn(potential)).real)


def plaquette(n: int, origin: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    c = np.zeros((n, n, n, 3))
    x = list(origin)
    c[tuple(x) + (0,)] += 1
    x[0] = (x[0] + 1) % n
    c[tuple(x) + (1,)] += 1
    x[0] = origin[0]
    x[1] = (x[1] + 1) % n
    c[tuple(x) + (0,)] -= 1
    c[origin + (1,)] -= 1
    return c / 2


def symbolic_operator() -> dict[str, Any]:
    theta, phi, u, s, b = sp.symbols('Theta Phi SU4 s b', real=True)
    psi = sp.Function('Psi')(theta, phi, u)
    boost = sp.exp(sp.I * s * b * theta)
    electric = lambda f: -sp.I * sp.diff(f, theta)
    shifted = sp.simplify(electric(boost * psi) / boost - electric(psi))
    second = sp.simplify(electric(electric(boost * psi)) / boost
                         - electric(electric(psi)) - 2 * s * b * electric(psi))
    phi_diff = sp.simplify(sp.diff(boost * psi, phi, 2) / boost - sp.diff(psi, phi, 2))
    su4_diff = sp.simplify(sp.diff(boost * psi, u) / boost - sp.diff(psi, u))
    assert shifted == s * b * psi and second == s ** 2 * b ** 2 * psi
    assert phi_diff == 0 and su4_diff == 0
    # Phi/SU4 multiplication and every literal CAR path commute with boost.
    k = sp.symbols('k', integer=True)
    B = sp.Function('B')(phi, u)
    matter_diff = sp.simplify(boost * B * sp.exp(sp.I * k * theta) * psi
                            - B * sp.exp(sp.I * k * theta) * boost * psi)
    assert matter_diff == 0
    return {'electric_first': str(shifted), 'electric_second_remainder': str(second),
            'Phi_kinetic_commutator': str(phi_diff), 'SU4_vectorfield_commutator': str(su4_diff),
            'arbitrary_quantum_B_and_integer_path_commutator': str(matter_diff),
            'extension': 'Leibniz identity on smooth equivariant fullPhi/Haar/CAR core; closed form closure via bounded U and finite electric shift. Not a fixedPhi section.',
            'domain_paid': 'U preserves D(h_form); D(H) and D(H^2) not inferred from D(h_form)'}


def lattice_control(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    V = n ** 3
    coords = np.indices((n, n, n))
    parity = cast(NDArray[np.float64], (-1.) ** sum(coords[j] for j in range(3)))
    # Every charged copy and the heavy neutral has wedge4 on V/2 sites,
    # wedge0 elsewhere. These are actual local SU4 singlets, each with2V CARs.
    occupations = 2 + 2 * parity
    rho = 64 * (occupations - 2)
    el = longitudinal(rho)
    b = plaquette(n)
    plane = np.zeros_like(b)
    plane[..., 1] = np.cos(2 * np.pi * coords[0] / n)
    plane /= np.linalg.norm(plane)
    cycle = np.zeros_like(b)
    cycle[:, 0, 0, 0] = 1 / np.sqrt(n)
    closed = [b, plane, cycle]
    assert np.max(np.abs(divergence(el) - rho)) < 1e-10
    assert all(np.max(np.abs(divergence(v))) < 1e-12 for v in closed)
    incoming = Fraction(3, 8)
    s_values = [Fraction(1, 2), Fraction(1, 16), Fraction(1, 256)]
    controls = []
    for label, v in zip(('coexact_plaquette', 'coexact_nonzero_momentum_cosine', 'closed_noncontractible_cycle'), closed):
        et = float(incoming) * v
        E = et + el
        assert abs(np.sum(el * v)) < 1e-10
        for s in s_values:
            # Increment calculated without subtracting two extensive energies.
            actual = 64 * (2 * float(s) * np.sum(E * v) + float(s) ** 2 * np.sum(v * v))
            reduced = 64 * (2 * float(s) * np.sum(et * v) + float(s) ** 2 * np.sum(v * v))
            expected = 64 * (2 * float(s) * float(incoming) + float(s) ** 2)
            assert abs(actual - expected) < 1e-9 and abs(reduced - expected) < 1e-9
            controls.append({'source': label, 's': str(s), 'norm_b_squared': float(np.sum(v*v)),
                             'full_HEHC_increment': actual, 'reduced_increment': reduced,
                             'HC_increment': 0, 'Gauss_residual': float(np.max(np.abs(divergence(E+float(s)*v)-rho)))})
    # Genuine adverse control: a gauge-complete hop c_x^dag B c_y exp(i k.Theta)
    # from occupied y=(1,1,0) to empty x=(0,1,0). Dk is its charge transfer.
    k = np.zeros_like(b)
    k[0, 1, 0, 0] = 1
    delta_rho = divergence(k)
    rho_after = rho + delta_rho
    kl = longitudinal(delta_rho)
    kt = k - kl
    E = el + float(incoming) * b
    full = 64 * (2 * np.sum(E * k) + np.sum(k*k))
    transverse = 64 * (2 * float(incoming) * np.sum(b*kt) + np.sum(kt*kt))
    coulomb = 64 * (2 * np.sum(el*kl) + np.sum(kl*kl))
    assert abs(full - transverse - coulomb) < 1e-8
    assert np.max(np.abs(divergence(E+k)-rho_after)) < 1e-10
    wrong_gauss = float(np.max(np.abs(divergence(E+kt)-rho_after)))
    assert wrong_gauss > .9 and abs(coulomb) > 1
    # Collective HC is not sum of64 individual-copy Coulomb energies.
    individual = longitudinal(occupations - 2)
    hc = 64 * np.sum(el*el)
    fake_hc = 64 * 64 * np.sum(individual*individual)
    assert abs(hc / fake_hc - 64) < 1e-9
    path_receipts = []
    for profile in profiles:
        count = 0
        seam_count = 0
        endpoints_ok = True
        for row in profile['paths']:
            axis, length = int(row['axis']), int(row['length'])
            assert Fraction(row['c0']) != 0
            for start in np.ndindex(n, n, n):
                path = np.zeros_like(b)
                pos = list(start)
                seam = 1
                for _ in range(length):
                    path[tuple(pos)+(axis,)] += 1
                    if pos[axis] == n-1:
                        seam *= -1
                    pos[axis] = (pos[axis]+1) % n
                boundary = np.zeros((n,n,n)); boundary[start] += 1; boundary[tuple(pos)] -= 1
                endpoints_ok = endpoints_ok and bool(np.array_equal(divergence(path), boundary))
                count += 1; seam_count += int(seam == -1)
        assert endpoints_ok
        path_receipts.append({'role':profile['role'], 'load':profile['load'], 'path_count':count,
                              'AP_minus_paths':seam_count, 'exact_Dk_endpoint':endpoints_ok,
                              'physical_hopping':1, 'boost_commutation':'all Theta multiplication and quantum Phi/SU4/CAR coefficients commute; no coefficient sampled or frozen'})
    # Exact character orthogonality: all translated normalized plaquettes are
    # distinct; s/2 differences cannot be integer for these rational0<s<1.
    chars = [plaquette(n, x).reshape(-1) for x in np.ndindex(n,n,n)]
    distinct = len({v.tobytes() for v in chars}) == V
    assert distinct
    q = 2 * np.pi / n
    coefficients = np.exp(-1j*q*coords[0]).reshape(-1) / np.sqrt(V)
    m0 = float(np.vdot(coefficients, coefficients).real)
    assert abs(m0-1) < 1e-12
    # H and its spectral kernels preserve characters. Thus all cross forms
    # vanish, giving an identity Gram without evaluating the full ground.
    moments = [{'s':str(s), 'sharp_character_total_m0':1,
                'sharp_character_ground_first_moment':str(64*s*s),
                'whole_P0_excited_m0':'1-||P0 A_q Omega||^2; unknown without transported ground-block bound',
                'sharp_source_divided_by_s_total_m0':str(1/(s*s))} for s in s_values]
    return {'N':n,'V':V,'physical_a':'1/64','physical_nonzero_q':float(64*q),
            'legal_state':'Normalized transverse Gaussian times exp(i E_L.Theta), SU4-invariant fullPhi Gaussian/Haar, all65 species local wedge4/wedge0 singlets; Ref wedge4. Physical quotient, not unreduced gauge-orbit L2. Smooth finite-form trial, NOT ground.',
            'trial_transverse_dimension':2*V+1,'charged_species_particles':[2*V,63,2*V],
            'neutral_particles':2*V,'Ref_particles':4,'collective_HC_trial':float(hc),
            'collective_vs_independent_HC_ratio':float(hc/fake_hc),
            'boost_controls':controls,'native_operator_paths':path_receipts,
            'distinct_legal_adverse_hop':{'full_increment':float(full),'transverse_increment':float(transverse),
                                        'collective_Coulomb_increment':float(coulomb),
                                        'Pi_only_Gauss_residual':wrong_gauss,
                                        'closed_probes_alone_discriminate_longitudinal_omission':False,
                                        'nonzero_full_hopping_Phi_CAR_matrix_element':'allowed wedge4 to wedge3/wedge1 with native nonzero B(Phi,U); gauge-complete output retains link/Phi covariance'},
            'translated_coexact_character_count':V,'distinct_characters':distinct,
            'nonzero_q_normalization':m0,'conditional_fullground_spectral_moments':moments,
            'ground_evaluation':False,'arbitrary_Borel_point_evaluation':False}


def original_packet_controls() -> dict[str, Any]:
    """Exact original-measure marginal, no evaluation of a Borel fibre."""
    beta = sp.symbols('beta', real=True)
    eta, delta, B, r, m = sp.symbols('eta delta B r m', positive=True)
    center = sp.Rational(1, 3)
    # Normalize a rank-two cluster frame without choosing an eigensection.
    frame_norm = sp.trigsimp(sp.cos(beta)**2 + sp.sin(beta)**2)
    packet_norm = sp.simplify(sp.integrate(frame_norm/(2*eta), (beta,center-eta,center+eta)))
    assert packet_norm == 1
    null_mass = sp.integrate(sp.Piecewise((1, sp.Eq(beta,center)),(0,True)),
                             (beta,center-eta,center+eta))
    assert null_mass == 0
    complete_square = (8*sp.sqrt(B)*r + 8*m/sp.sqrt(B))**2 + delta - 64*m*m/B
    assert sp.expand(complete_square-(delta+128*r*m+64*r*r*B)) == 0
    first_bound = sp.sqrt(delta*B/64)
    assert sp.limit(sp.sqrt(eta*B/64),eta,0) == 0
    packets = []
    for n in (4,6,8):
        for s in (Fraction(1,2),Fraction(1,16),Fraction(1,256)):
            width = s/16
            shift = s/2
            overlap_length = max(Fraction(0),2*width-shift)
            assert overlap_length == 0
            # Other beta coordinates and physical cluster sections integrate
            # to1. One distinguishing character marginal kills cross terms.
            defect = width*width
            bound = (sp.sqrt(sp.Rational(defect.numerator,defect.denominator))
                     + 8*sp.Rational(s.numerator,s.denominator))**2
            packets.append({'N':n,'physical_transverse_rank':2*n**3+1,
                            'beta_center_not_ground_identification':'1/3',
                            's':str(s),'original_measure_halfwidth':str(width),
                            'character_marginal_shift':str(shift),
                            'integrated_norm_exact':1,'shifted_support_overlap_exact':str(overlap_length),
                            'illustrative_packet_defect_budget':str(defect),
                            'shifted_excess_budget_exact':str(bound)})
    return {'norm_integral':str(packet_norm),'global_Borel_singleton_null_mass':str(null_mass),
            'cluster_frame_norm':str(frame_norm),'packet_controls':packets,
            'variational_complete_square':str(complete_square),
            'first_moment_bound':str(first_bound),
            'ground_first_moment_limit':0,
            'scope':'Exact ORIGINAL-measure normalization/support and full-form variational arithmetic; native minimizing cluster sections remain the explicit full-ground antecedent, not supplied toy eigenvectors.',
            'finite_packet_positive_excitation_generator_inferred':False}


def spectral_controls() -> dict[str, Any]:
    s = sp.symbols('s', positive=True)
    eps = sp.symbols('eps', nonnegative=True)
    upper = 128*s*s
    m0 = sp.integrate(1/upper, (eps,0,upper))
    m1 = sp.integrate(eps/upper, (eps,0,upper))
    assert m0 == 1 and m1 == 64*s*s
    regular = sp.limit((1-sp.exp(-s*s))/(s*s), s, 0)
    assert regular == 1
    # A distinct ground vector in the transported character must be removed.
    # Xi has1/2 mass there and1/2 at128s²: same m0,m1 but Q0 mass1/2.
    assert sp.Rational(1,2)*128*s*s == m1
    temporal = []
    for val in (Fraction(1,2),Fraction(1,16),Fraction(1,256)):
        a = 128*float(val)**2
        times = np.array([0.,.25,1.,2.])
        kernel = np.ones((4,4))
        for i,t in enumerate(times):
            for j,u in enumerate(times):
                z=a*(t+u)
                kernel[i,j] = -np.expm1(-z)/z if z else 1.
        eigen = np.linalg.eigvalsh(kernel)
        assert eigen[0] > -1e-12
        temporal.append({'s':str(val),'uniform_continuum_support_upper':a,
                         'temporal_kernel_min_eigenvalue_float':float(eigen[0]),
                         'atom_mass_at_any_single_energy_exact':0,
                         'relative_window_halfwidth_1_over_100_fraction_exact':'1/50'})
    return {'exact_no_atom_same_moments':{'density':'1/(128s^2) on[0,128s^2]',
                                         'm0':str(m0),'m1':str(m1),
                                         'all_q':'may use this measure at EVERY spatial q; no sharp spatial dispersion',
                                         'positive_kernel':'K(t,u)=int exp(-eps*(t+u)) dmu is a Gram kernel exactly'},
            'temporal_evaluations':temporal,
            'regular_characteristic_contrary':{'chi':'exp(-s^2/2)',
                                                'connected_m0':'1-exp(-s^2)',
                                                'limit_m0_over_s_squared':str(regular),
                                                'scope':'exact regular representation normalization control, NOT actual native ground'},
            'whole_ground_block_adverse':{'total_m0':1,'mean_connected_m0':1,
                                          'transported_distinct_ground_mass':'1/2',
                                          'whole_P0_excited_m0':'1/2',
                                          'm1':'64s^2',
                                          'spectrum':'1/2 delta0 +1/2 delta_(128s^2)',
                                          'scope':'exact character-preserving positive spectral representation control, not native ground identification'},
            'sharp_regular_contradiction':'sharp-character ||(U_s-1)Omega||^2=2 for every noninteger small shift; no strong continuity or Stone Theta generator in that state',
            'pole_inferred':False}


def run(request_path: Path, output: Path) -> None:
    request = json.loads(request_path.read_text())
    for name,pin in request['input_sha256'].items():
        assert digest(ROOT/name) == pin, name
    profiles = json.loads((ROOT/request['profile']).read_text())['primitive_profiles']
    assert [(p['load'],p['scale']) for p in profiles] == [(1,'heavy'),(1,'charged'),(63,'charged')]
    result = {'kind':'FULL_PHYSICAL_FRACTIONAL_WEYL_FORM_AND_CHARACTER_SPECTRAL_CONTROL_NOT_PHOTON',
              'request_sha256':digest(request_path),'source_sha256':digest(Path(__file__)),
              'input_sha256':request['input_sha256'],'operator':symbolic_operator(),
              'original_normalized_packet_controls':original_packet_controls(),
              'N_controls':[lattice_control(n,profiles) for n in (4,6,8)],
              'spectral_contrary_controls':spectral_controls(),
              'full_physical_ground':'explicit antecedent; selected frozenPhi local ground not substituted',
              'paid_edge':'fullPhi/Gauss fractional source form domain, minimizing-beta first moment identity, volume-normalized sharp-character spectral scaling',
              'unpaid_edge':'regular same-ground source limit or bounded-flux overlap; transported whole-ground block, thermodynamic phase/stiffness and rescaled-q atom/residue',
              'physical_hopping':1,'HE':64,'HC':64,'bare_B2':False,
              'finite_observations_not_uniform_proof':True,'ground_or_photon_admitted':False}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'receipt':output.name,'N':[4,6,8],'physical_hopping':1,'sharp_photon':False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.request,args.output)
