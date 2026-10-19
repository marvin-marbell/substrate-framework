"""NEW restricted real uniform Phi35 law: finite CAR, not PH bosonization.

Hilbert space L2(R,dphi) tensor exterior(C^(65*256+4)).  All 65 species
have the native N4 antiperiodic paths; the neutral Hamiltonian is -Lambda h.
Ref4 is static. Theta/Pi/local SU4/Noether and the other Phi coordinates are
explicitly omitted in this NEW law, NOT removed from the native primitive law.
The physical source is E=.5 sum_alpha c_active(0,alpha)^dagger R_alpha.

The CLI constructs literal matrices/CAR and outward exact-algebraic prices.
It certifies the untruncated real-scalar SAME-H Heisenberg source by a global
Kato-frame Duhamel comparison, retaining the common sea/oscillator phases in
both legs. No interacting ground-state or original SOURCE claim is made.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.abc
import importlib.machinery
from itertools import combinations, product
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES: dict[str, bytes] = {Path(__file__).name: Path(__file__).read_bytes()}


class CapturedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes()
        SOURCE_BYTES[Path(self.path).name] = raw
        return self.source_to_code(raw, self.path)


class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.origin is not None:
            source = Path(spec.origin)
            if source.parent == ROOT and source.suffix == '.py':
                spec.loader = CapturedLoader(fullname, str(source))
                return spec
        return None


_finder = CapturedFinder()
sys.meta_path.insert(0, _finder)
import induced_phi_uniform_quantum_backreaction as uniform
import induced_phi_uniform_global_gauss as global_gauss
sys.meta_path.remove(_finder)
SOURCE_BYTES.update(uniform.SOURCE_BYTES)
import mpmath as mp
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye as sparse_eye

iv: Any = uniform.iv
exact = uniform.exact
endpoint = uniform.endpoint
supplier = uniform.supplier
UNIFORM_SHA = '0c3d65dce2d22181073cc9bdd373a7ba7c99977f895ffab1a097e7ae070c9da7'
RECEIPT_SHA = 'ee8f8ebd6b20db73b7a5412ca828bc9d2687fc28b1c7bfb5fb3cd27eb5843166'
GAUSS_SOURCE_SHA = 'd0e8707178a99bd742a9f56c1de50c4d4f69f5a35abe3a0f711c7e1a05a4455c'
GAUSS_RECEIPT_SHA = '59d3655442ee61ac48e4a244b9496bc63243fd86411608b2d62e333413407c5f'
DIGITS = 4400


def decimal_edge(value: Fraction, upper: bool, digits: int = 65) -> str:
    """Directed decimal rounding, using integers, never a floating log."""
    if not value:
        return '0'
    positive = abs(value)
    exponent = len(str(positive.numerator)) - len(str(positive.denominator))
    power = Fraction(10)**exponent
    if positive < power:
        exponent -= 1
    elif positive >= 10*power:
        exponent += 1
    step_exponent = exponent-digits+1
    q = value / Fraction(10)**step_exponent
    floor = q.numerator//q.denominator
    integer = floor + int(upper and q.denominator != 1)
    return f'{integer}e{step_exponent}'


def pack(value: Any) -> dict[str, str]:
    return dict(lower=decimal_edge(endpoint(value, 0), False),
                upper=decimal_edge(endpoint(value, 1), True),
                encoding='exact outward decimal endpoints')


def cap(value: Any) -> Any:
    return exact(endpoint(value, 1))


def absolute_cap(value: Any) -> Any:
    return exact(max(abs(endpoint(value, 0)), abs(endpoint(value, 1))))


def maximum(values: list[Any]) -> Any:
    return exact(max(endpoint(value, 1) for value in values))


def car_annihilator(mode: int, count: int) -> csr_matrix:
    rows, cols, data = [], [], []
    for state in range(1 << count):
        if state & (1 << mode):
            rows.append(state ^ (1 << mode)); cols.append(state)
            data.append((-1)**((state & ((1 << mode)-1)).bit_count()))
    return csr_matrix((data, (rows, cols)), shape=(1 << count, 1 << count), dtype=complex)


def exterior_unitary(U: np.ndarray) -> np.ndarray:
    """Exact finite CAR exterior representation; no determinant inverses."""
    result = np.zeros((16, 16), complex)
    for number in range(5):
        subsets = list(combinations(range(4), number))
        for left in subsets:
            for right in subsets:
                result[sum(1 << i for i in left), sum(1 << i for i in right)] = np.linalg.det(U[np.ix_(left, right)]) if number else 1
    return result


def literal_car_source(projector: np.ndarray) -> dict[str, Any]:
    """Physical eight-mode E and the actual four-mode reduced Slater density."""
    annihilators = [car_annihilator(i, 8) for i in range(8)]
    I = sparse_eye(256, format='csr')
    defect = 0.
    for i, a in enumerate(annihilators):
        for j, b in enumerate(annihilators):
            difference = a@b.conj().T+b.conj().T@a-(I if i == j else 0*I)
            defect = max(defect, float(np.max(abs(difference.data))) if difference.nnz else 0.)
    E = annihilators[0].conj().T@annihilators[4]/2
    for i in range(1,4):
        E = E+annihilators[i].conj().T@annihilators[4+i]/2
    occupations, U = np.linalg.eigh(projector)
    if occupations.min() < -1e-12 or occupations.max() > 1+1e-12:
        raise ArithmeticError('Actual local reduced Slater occupations invalid')
    weights = np.array([np.prod([occupations[i] if state & (1 << i) else 1-occupations[i] for i in range(4)]) for state in range(16)])
    wedge = exterior_unitary(U); local_rho = (wedge*weights)@wedge.conj().T
    rho = np.zeros((256, 256), complex); rho[240:,240:] = local_rho
    source_rho = E@rho@E.conj().T
    norm = float(np.trace(source_rho).real)
    nref = np.array([((state >> 4) & 15).bit_count() for state in range(256)])
    wrong_ref_weight = float(np.trace(source_rho[np.ix_(nref != 3, nref != 3)]).real)
    operator_norm = float(np.sqrt(np.linalg.eigvalsh((E.conj().T@E).toarray())[-1]))
    if defect or abs(norm-.5) > 2e-12 or abs(wrong_ref_weight) > 1e-14:
        raise ArithmeticError('Literal physical CAR/Ref4 source failed')
    return dict(ordered_CAR_defect=defect, reduced_Slater_trace=float(np.trace(local_rho).real),
                physical_E_norm_squared=norm, reference_holes_outside_Ref3_weight=wrong_ref_weight,
                actual_full_eight_mode_E_operator_norm=operator_norm,
                proof_operator_norm_upper=2, source_normalization='RAW .5, no whitening',
                reduced_occupation_eigenvalues=occupations.tolist())


def literal_construction(mean: dict[str, Any], profiles: list[dict[str, Any]], points: list[list[dict[str, Any]]]) -> dict[str, Any]:
    native = supplier.quantum.build_native(mean, 4)
    V = native['V']; zero = np.zeros(3*V)
    mode = dict(supplier.pauli.magnetic_shape(native)); mode['mode'] = np.ones(V)/np.sqrt(V)
    gamma = supplier.primitive.SELECTED
    T = 1j*gamma[0]@gamma[1]@gamma[2]
    gamma5 = gamma[0]@gamma[1]@gamma[2]@gamma[3]
    if np.linalg.norm(mode['delta']-T/np.sqrt(8)) > 1e-12:
        raise ArithmeticError('Actual native Cart Phi35 convention differs')
    geometry = uniform.uniform_geometry(native, mode)
    indices = np.asarray(list(product(range(4), repeat=3)))
    momenta = (2*np.pi*indices+np.pi)/4
    spatial = np.exp(1j*native['sites']@momenta.T)/np.sqrt(V)
    fourier = np.kron(spatial, np.eye(4)); rows = []
    active_spectral = None
    for profile, pts in zip(profiles, points, strict=True):
        sign = -1 if profile['scale'] == 'heavy' else 1
        if sign == 1:
            physical = next(p for p in native['Pprofiles'] if p['label'] == profile['role'])
        else:
            paths = []
            for displacement, coefficients in profile['raw_paths']:
                support = [p for p in native['Pprofiles'][0]['paths'] if np.array_equal(p['displacement'], displacement)]
                if len(support) != V:
                    raise ArithmeticError('Missing actual heavy AP ordered path')
                paths.extend({**p, 'scalars': coefficients, 'hop': p['seam']*np.einsum('m,mij->ij', coefficients, native['frame'])} for p in support)
            physical = dict(label=profile['role'], load=1, model=profile['model'], onsite=float(profile['onsite'])*native['frame'][0], paths=paths)
        h = sign*supplier.quantum.matter_hamiltonian(native, physical, zero)
        F = sign*supplier.pauli.operator(native, physical, zero, mode)
        energies, U = np.linalg.eigh(h)
        if not energies[127] < 0 < energies[128]:
            raise ArithmeticError('Actual all-species negative Slater filling failed')
        W, Z = U[:,:128], U[:,128:]
        gaps = energies[None,128:]-energies[:128,None]
        weights = abs(W.conj().T@F@Z)**2
        fs = float(np.sum(weights/gaps**2)); bubble = float(np.sum(2*weights/gaps))
        expected_h, expected_F = np.zeros_like(h), np.zeros_like(F)
        algebraic_fs, algebraic_bubble = iv.mpf(0), iv.mpf(0)
        kato_errors = []; displacement_errors = []; source_rows = []
        for j, point in enumerate(pts):
            coefficients = [float(endpoint(v, 0)) for v in point['h']]
            r = np.linalg.norm(coefficients[:3]); z = abs(coefficients[3]); b = float(endpoint(point['xi0']/(8*iv.sqrt(exact(8))), 0))
            block_h = sum(coefficients[i]*gamma[i] for i in range(4))
            block_F = b*T
            expected_h[4*j:4*j+4,4*j:4*j+4] = sign*block_h
            expected_F[4*j:4*j+4,4*j:4*j+4] = sign*block_F
            algebraic_fs += point['weight']/point['delta']**2
            algebraic_bubble += 2*point['weight']/point['delta']
            D = sum(coefficients[i]*gamma[i] for i in range(3))/r
            Zgamma = np.sign(coefficients[3])*gamma[3]
            C = D@Zgamma; S = T@D
            for phi in (-12., .125, 4.):
                rotation = np.zeros((4,4), complex); diagonal = np.zeros((4,4), complex); connection = np.zeros((4,4), complex)
                for branch in (-1, 1):
                    Q = (np.eye(4)+branch*S)/2
                    y = r+branch*b*phi; energy = np.sqrt(z*z+y*y)
                    angle = np.arctan(y/z)-np.arctan(r/z)
                    rotation += expm(angle*C/2)@Q
                    diagonal += energy/np.sqrt(r*r+z*z)*block_h@Q
                    connection += branch*b*z/(z*z+y*y)*C@Q/2
                hp = block_h+phi*block_F
                P = (np.eye(4)-rotation@(block_h/np.sqrt(r*r+z*z))@rotation.conj().T)/2
                Pprime = connection@P-P@connection
                kato_errors.append(float(np.linalg.norm(Pprime@P-P@Pprime-connection)))
                displacement_errors.append(float(np.linalg.norm(hp-rotation@diagonal@rotation.conj().T)))
                kato_errors.append(float(abs(np.trace(P@connection))))
                if np.linalg.norm(hp@gamma5+gamma5@hp) > 1e-10:
                    raise ArithmeticError('Literal all-phi grade1/grade3 chiral symmetry failed')
        matrix_errors = dict(full256_h_Bloch=float(np.max(abs(fourier.conj().T@h@fourier-expected_h))),
                             full256_F_Bloch=float(np.max(abs(fourier.conj().T@F@fourier-expected_F))),
                             Kato_projector_connection_and_Berry=float(max(kato_errors)),
                             global_spectral_rotation=float(max(displacement_errors)),
                             mean_FS_metric=abs(fs-float(endpoint(algebraic_fs, 0))),
                             mean_static_PH_bubble=abs(bubble-float(endpoint(algebraic_bubble, 0))))
        for phi in (-12., 0., .125, 4.):
            ev, vectors = np.linalg.eigh(h+phi*F)
            prediction = []
            for point in pts:
                parallel = float(endpoint(iv.sqrt(sum(c*c for c in point['h'][:3])), 0))
                z = float(endpoint(absolute_cap(point['h'][3]), 0))
                b = float(endpoint(point['xi0']/(8*iv.sqrt(exact(8))), 0))
                pair = [np.sqrt(z*z+(parallel+branch*b*phi)**2) for branch in (-1,1)]
                prediction.extend([-pair[0],-pair[1],pair[0],pair[1]])
            matrix_errors[f'full256_global_spectrum_phi_{phi}'] = float(np.max(abs(ev-np.sort(prediction))))
            if profile['role'] == 'active filled charged':
                Psite = vectors[:4,:128]@vectors[:4,:128].conj().T
                source_rows.append(dict(phi=phi, **literal_car_source(Psite)))
        if max(matrix_errors.values()) > 3e-10:
            raise ArithmeticError('Independent literal CAR/Kato/full-spectrum calculation failed')
        if profile['role'] == 'active filled charged':
            positive_site_weights = np.sum(abs(U[:4,128:])**2, axis=0)/4
            t = 1/4096
            active_spectral = complex(np.sum(positive_site_weights*np.exp(-1j*energies[128:]*t)))
        rows.append(dict(role=profile['role'], multiplicity=profile['load'], physical_h_sign=sign,
                         physical_scale='Lambda' if sign == -1 else '1', filled_negative_rank=128,
                         measured_FS_metric=fs, measured_static_PH_bubble=bubble,
                         literal_errors=matrix_errors, physical_CAR_source_controls=source_rows))
    if active_spectral is None:
        raise ArithmeticError('Actual active source consumer missing')
    return dict(actual_native_mean=dict(g0=native['g0'], gs=native['gs']),
                geometry=geometry, all65_species=rows,
                free_reference_literal_source_time_1over4096=dict(real=active_spectral.real, imag=active_spectral.imag),
                finite_CAR_modes=65*256+4, prepared_total_particles=65*128+4,
                source_particle_transition=dict(active=[128,129], Ref=[4,3], auxiliary_each=[128,128], heavy=[128,128]),
                exact_particle_conservation='All 8324 particles retained in BOTH source and vacuum legs')


def exact_proof(profiles: list[dict[str, Any]], points: list[list[dict[str, Any]]], g0: Fraction) -> dict[str, Any]:
    Lambda = exact(Fraction(2)**8192); G = exact(8*Fraction(2)**-4096)
    kappa = Lambda*exact(g0)*32; primitive4 = Lambda/exact(64)
    sea0, bubble, sea4, remainder6, q0 = [iv.mpf(0) for _ in range(5)]
    C1, C2, Qvariation, Qfourth, global_curvature_bubble, active_C1 = [iv.mpf(0) for _ in range(6)]
    active_b = []; species_rows = []; active_points = None
    for profile, pts in zip(profiles, points, strict=True):
        scale = Lambda if profile['scale'] == 'heavy' else exact(1)
        load = exact(profile['load']); local = dict(sea0=iv.mpf(0), bubble=iv.mpf(0), sea4=iv.mpf(0), remainder6=iv.mpf(0), q0=iv.mpf(0), C1=iv.mpf(0), C2=iv.mpf(0), Qvariation=iv.mpf(0), Qfourth=iv.mpf(0), global_curvature_bubble=iv.mpf(0))
        gap_lower = []
        for point in pts:
            # h3 is an exact nonzero AP-quarter sine algebraic number. Using its
            # LOWER modulus, not its upper interval, is essential in gap prices.
            z = exact(min(abs(endpoint(point['h'][3], 0)), abs(endpoint(point['h'][3], 1))))
            if endpoint(z, 0) <= 0:
                raise ArithmeticError('Actual global all-phi AP h3 gap vanishes')
            b = absolute_cap(point['xi0']/(8*iv.sqrt(exact(8))))
            local['sea0'] -= 2*scale*point['E']
            local['bubble'] += scale*2*point['weight']/point['delta']
            local['sea4'] += scale*point['static_filled_sea_quartic_coefficient']
            local['remainder6'] += scale*exact(Fraction(21,8))*b**6/z**5
            local['q0'] += point['weight']/point['delta']**2
            local['C1'] += b/z
            local['C2'] += 2*b*b/(z*z)
            local['Qvariation'] += 7*b**4/z**4
            local['Qfourth'] += 960*b**6/z**6
            local['global_curvature_bubble'] += 2*scale*b*b/z
            gap_lower.append(2*scale*z)
            if profile['role'] == 'active filled charged':
                active_b.append(b)
        sea0 += load*local['sea0']; bubble += load*local['bubble']; sea4 += load*local['sea4']
        remainder6 += load*local['remainder6']; q0 += load*local['q0']
        C1 += load*local['C1']; C2 += load*local['C2']; Qvariation += load*local['Qvariation']
        Qfourth += load*local['Qfourth']; global_curvature_bubble += load*local['global_curvature_bubble']
        if profile['role'] == 'active filled charged':
            active_C1 = local['C1']; active_points = pts
        species_rows.append(dict(role=profile['role'], load=profile['load'], global_all_phi_PH_gap_lower=pack(exact(min(endpoint(x, 0) for x in gap_lower))),
                                 exact_filled_sea0=pack(load*local['sea0']), exact_static_bubble=pack(load*local['bubble']),
                                 exact_sea_quartic=pack(load*local['sea4']), all_tail_sixth_remainder_coefficient_upper=pack(cap(load*local['remainder6'])),
                                 exact_mean_FS_metric=pack(load*local['q0']), CAR_connection_norm_upper=pack(cap(load*local['C1'])),
                                 CAR_connection_derivative_norm_upper=pack(cap(load*local['C2']))))
    effective = kappa-bubble; quartic = primitive4+sea4
    if endpoint(effective, 0) <= 0 or endpoint(quartic, 0) <= 0 or active_points is None:
        raise ArithmeticError('Actual scalar Gaussian variational law unstable')
    omega = iv.sqrt(G*effective); harmonic_x = G/(2*omega)
    epsilon = 12*quartic*harmonic_x/effective
    # Optimize the ACTUAL full Gaussian Rayleigh functional, not a surrogate:
    # heat identity d_x E[g(phi)]=E[gsecond]/2 gives |Rsea'|<=45 R6 x2,
    # |EBorn'|<=G Qvariation/2. Their derivative price encloses its local root.
    root_error = cap(2*(45*remainder6*(2*harmonic_x)**2+G*Qvariation/2)/effective)
    initial_lower = Fraction(1)-2*endpoint(epsilon+root_error,1)
    bracket = uniform.enclosure(initial_lower,Fraction(1))
    lower_score = -1/exact(initial_lower)**2+1+epsilon*exact(initial_lower)
    upper_score = epsilon
    if endpoint(lower_score+root_error,1) >= 0 or endpoint(upper_score-root_error,0) <= 0:
        raise ArithmeticError('Actual full Gaussian variational root signs failed')
    # qfourth<=960 sum b6/z6; hence |EBornsecond|<=G Qfourth/8.
    # The normalized exact derivative is strictly increasing throughout x<=2x0.
    root_slope_lower = exact(Fraction(1,4))-cap(2*harmonic_x*(90*remainder6*(2*harmonic_x)+G*Qfourth/8)/effective)
    if endpoint(root_slope_lower,0) <= 0:
        raise ArithmeticError('Actual full Gaussian local minimum not unique')
    correction = uniform.enclosure(-endpoint(root_error,1),endpoint(root_error,1))
    for _ in range(8):
        image = 1/iv.sqrt(1+epsilon*bracket+correction)
        bracket = uniform.enclosure(max(endpoint(bracket,0),endpoint(image,0)),min(endpoint(bracket,1),endpoint(image,1)))
    x = harmonic_x*bracket
    gaussian_energy = G/(8*x)+effective*x/2+3*quartic*x*x
    BH_mean = G*q0/2
    BH_error = cap(G*Qvariation*x/2)
    sea_mean_error = cap(15*remainder6*x**3)
    energy_error = BH_error+sea_mean_error
    full_energy = sea0+gaussian_energy+BH_mean+uniform.enclosure(-endpoint(energy_error,1),endpoint(energy_error,1))
    # Any harmonic evolution of this optimized real Gaussian remains Gaussian:
    # X=max(x,xharm^2/x), P=max(1/(4x), kappa_eff*x/G).
    X = maximum([x,harmonic_x**2/x]); P = maximum([1/(4*x),effective*x/G])
    potential_price = cap(absolute_cap(quartic)*iv.sqrt(exact(105))*X**2+remainder6*iv.sqrt(exact(10395))*X**3)
    derivative_price = cap(G*C1*iv.sqrt(P))
    second_price = cap(G*(C2+C1*C1)/2)
    D = potential_price+derivative_price+second_price
    B = maximum(active_b)
    charged_price = cap(B*iv.sqrt(X))
    Eoperator = exact(2); p = exact(Fraction(1,2)); rootp = iv.sqrt(p)
    endpoint0 = cap(2*Eoperator*active_C1*iv.sqrt(x))
    endpoint_t = cap(2*Eoperator*active_C1*iv.sqrt(X)*rootp)
    # Residual at the exact symbolic Rayleigh centre. Its huge common sea0 is
    # retained. Centering phi4 has variance 96*x4, not the uncentred 105*x4.
    squeeze = cap(iv.sqrt(exact(2))*x*absolute_cap(effective/2-G/(8*x*x)))
    centered_quartic = cap(absolute_cap(quartic)*iv.sqrt(exact(96))*x*x)
    residual = squeeze+centered_quartic+cap(remainder6*(iv.sqrt(exact(10395))+15)*x**3)+derivative_price+second_price+cap(BH_mean)+BH_error
    times = []
    for t in (Fraction(0),Fraction(1,4096),Fraction(4096)):
        reference_real = sum(iv.cos(point['E']*exact(t)) for point in active_points)/128
        reference_imag = -sum(iv.sin(point['E']*exact(t)) for point in active_points)/128
        vacuum_error = cap(exact(t)*D)
        source_error = cap(exact(t)*rootp*(D+charged_price))
        # K=<U Psi, E^dagger U E Psi>. BOTH U factors carry the SAME raw H.
        # Etilde at both endpoints and all nonlinear/connection terms priced.
        error = cap(Eoperator*rootp*vacuum_error+Eoperator*(source_error+endpoint0)+endpoint_t)
        if t == 0:
            # Exact CAR/chiral identity, independently exercised above, sharpens
            # the general endpoint estimate. No fitted/whitened normalization.
            error = exact(0)
        source_norm_deviation = cap(Eoperator*vacuum_error+2*Eoperator*active_C1*iv.sqrt(X)) if t else exact(0)
        source_norm_lower = max(Fraction(0),endpoint(rootp-source_norm_deviation,0))
        evolved_source_probability = uniform.enclosure(source_norm_lower**2,min(Fraction(1),endpoint((rootp+source_norm_deviation)**2,1)))
        rr = uniform.enclosure(endpoint(reference_real,0)-endpoint(error,1),endpoint(reference_real,1)+endpoint(error,1))
        ii = uniform.enclosure(endpoint(reference_imag,0)-endpoint(error,1),endpoint(reference_imag,1)+endpoint(error,1))
        magnitude_lower = min(abs(endpoint(reference_real,0)),abs(endpoint(reference_real,1))) if endpoint(reference_real,0)*endpoint(reference_real,1)>0 else Fraction(0)
        imag_lower = min(abs(endpoint(reference_imag,0)),abs(endpoint(reference_imag,1))) if endpoint(reference_imag,0)*endpoint(reference_imag,1)>0 else Fraction(0)
        lower = iv.sqrt(exact(magnitude_lower*magnitude_lower+imag_lower*imag_lower))-error
        positive_lower = max(Fraction(0),endpoint(lower,0))
        probability_low = positive_lower**2
        probability_high = min(Fraction(1,2),endpoint((iv.sqrt(reference_real**2+reference_imag**2)+error)**2,1))
        times.append(dict(time=str(t), raw_free_source_real=pack(reference_real), raw_free_source_imag=pack(reference_imag),
                          raw_same_H_Heisenberg_source_real=pack(rr), raw_same_H_Heisenberg_source_imag=pack(ii),
                          absolute_complex_error_upper=pack(error), vacuum_leg_Duhamel_norm_upper=pack(vacuum_error),
                          source_leg_Duhamel_norm_upper=pack(source_error), actual_nonzero_modulus_lower=pack(exact(positive_lower)),
                          raw_source_return_probability_interval=pack(uniform.enclosure(probability_low,probability_high)),
                          physical_E_transfer_probability_on_evolved_vacuum=pack(evolved_source_probability),
                          interpretation='|K(t)|^2 is the RAW compound probability: E preparation, same-H evolution, E^dagger return and projection on U(t)Psi. E:Ref4->Ref3 and E^dagger:Ref3->Ref4 are contractions since E^dagger E on Ref4 is N_empty_site/4<=1. No division by p=.5, no whitening, no exclusive scattering claim.'))
    if endpoint(exact(Fraction(times[-1]['actual_nonzero_modulus_lower']['lower'])),0) <= 0:
        raise ArithmeticError('Actual 4096 raw physical source nonzero discrimination failed')
    return dict(all65_exact_species=species_rows, raw_same_H_sea_energy_retained=pack(sea0),
                physical_G=pack(G), physical_kappa=pack(kappa), primitive_quartic_coefficient=pack(primitive4),
                exact_all65_static_PH_bubble=pack(bubble), positive_kappa_eff_divided_Lambda=pack(effective/Lambda),
                exact_all65_sea_quartic=pack(sea4), positive_total_quartic=pack(quartic),
                all_phi_sixth_remainder_upper=pack(cap(remainder6)), exact_mean_all65_FS_metric=pack(q0),
                all_phi_FS_metric_variation_coefficient_upper=pack(cap(Qvariation)),
                global_CAR_connection_norm_upper=pack(cap(C1)), global_CAR_connection_derivative_norm_upper=pack(cap(C2)),
                active_source_CAR_connection_norm_upper=pack(cap(active_C1)), active_added_band_lipschitz_upper=pack(B),
                optimized_variance_sigma_squared=pack(x), free_harmonic_variance=pack(harmonic_x),
                optimized_width_leading_stationarity_residual=pack(1/bracket**2-1-epsilon*bracket),
                optimized_width_full_Rayleigh_derivative_remainder_upper=pack(root_error),
                optimized_width_exact_root_slope_lower=pack(root_slope_lower),
                optimization='sigma_squared is the unique local minimum of the ACTUAL Gaussian Rayleigh functional in the certified harmonic neighbourhood: exact kinetic, primitive potential, all65 instantaneous filled spectra and Born-Huang energy. Outward derivative remainders from global sixth/qfourth derivatives enclose that root; no fit/PH surrogate. No global Gaussian minimum or true ground claim.',
                adverse_global_convexity_sufficient_bound_divided_Lambda=pack((kappa-global_curvature_bubble)/Lambda),
                exact_Gaussian_moments=dict(phi2='x', phi4='3 x^2', phi6='15 x^3', phi8='105 x^4', phi12='10395 x^6', derivative2='1/(4 x)'),
                prepared_quantum_Rayleigh_energy=pack(full_energy), Rayleigh_components=dict(sea0=pack(sea0), kinetic_plus_effective_potential_plus_quartic=pack(gaussian_energy), exact_mean_Born_Huang=pack(BH_mean), Born_Huang_variation_upper=pack(BH_error), all_tail_mean_sixth_remainder_upper=pack(sea_mean_error)),
                actual_prepared_Rayleigh_residual_norm_upper=pack(cap(residual)), residual_squeeze_price=pack(squeeze),
                residual_claim='Actual normalized f_sigma(phi) times all-species instantaneous negative Slater lies in D(H). Residual relative to its exact symbolic Rayleigh centre is bounded above; NOT claimed true ground or stationary vacuum.',
                harmonic_evolution_variance_upper=pack(X), harmonic_evolution_derivative_squared_upper=pack(P),
                Duhamel_prices=dict(primitive_PLUS_exact_sea_nonlinear=pack(potential_price), connection_first_derivative=pack(derivative_price), connection_derivative_PLUS_square=pack(second_price), active_added_band_variation=pack(charged_price), source_endpoint0=pack(endpoint0), source_endpoint_t=pack(endpoint_t)),
                raw_source_times=times,
                algebraic_derivation=dict(spectrum='T=iGamma0Gamma1Gamma2, D=H012/r, S=T D. S commutes H012,Gamma3,T. On S=tau: h(phi)=(r+tau b phi)D+|h3| Z, b=xi0/(8 sqrt8), Z=sign(h3)Gamma3. Eigenvalues +-sqrt(h3^2+(r+-b phi)^2).',
                    projector='Pminus(phi)=sum_tau (I+tau S)/2 [I-h(phi)/E_tau(phi)]/2; physical heavy fills Pplus of h because its Hamiltonian is -Lambda h. Every negative rank is128. Global gap>=2 scale |h3| at EVERY real phi, not only a packet cutoff.',
                    Kato='C=D Z, theta_tau=atan((r+tau b phi)/|h3|); W=sum_tau exp[(theta_tau(phi)-theta_tau(0)) C/2] (I+tau S)/2. W^dagger Wprime=[Pprime,P] in this commuting rotation frame, Tr P A=0. CAR lift is the exact exterior unitary, no Slater-to-boson substitution.',
                    derivative_bounds='One-particle A eigenvalues +-i theta_tauprime/2, |theta_tauprime|<=b/|h3|; ||dGamma A||<=sum_blocks b/|h3|=C1, ||dGamma Aprime||<=sum_blocks 2 b^2/h3^2=C2, ALL finite CAR sectors and all65 loads. FS q(phi)=sum b^2 h3^2/4 [Eplus^-4+Eminus^-4], q0=sum_PH |F|^2/Delta^2; Berry=0.',
                    nonlinear='For e(y)=sqrt(z^2+y^2): e6=(45 z6-540 z4 y2+360 z2 y4)/(z2+y2)^(11/2), |e6|<=945/|z|5 globally. Filled pair is -scale[e(r+b phi)+e(r-b phi)], so |sea-sea0+B phi2/2-sea4 phi4|<=sum load scale(21/8)b6/|z|5 |phi|6. Exact fourth derivative, no curve fit. |q(phi)-q0|<=7 sum load b4/z4 phi2 from global |qsecond| bounds.',
                    exact_width_optimization='For real Gaussian variance x, d_x E[g]=E[gsecond]/2 and d_x2 E[g]=E[gfourth]/4. Sea remainder derivative<=45 R6 x2, second<=90 R6 x. |qsecond|<=2Qvariation, |qfourth|<=960 sum b6/z6=Qfourth. Actual derivative normalized by kappa_eff/2 is -1/y2+1+epsilon*y+eta, |eta|<=root_error. Observed opposite endpoint signs and positive derivative lower give one exact local Rayleigh minimizer; inclusion iteration encloses its x, including all tails/Born terms.',
                    source='Gamma5 anticommutes both grade1 h and grade3 F. At every phi, each 4x4 occupied projector has trace2, hence onsite Tr Q(0,0)=2 by AP Fourier completeness. Literal CAR E^dagger E=.25 sum_alpha,beta Ralpha^dagger c_alpha c_beta^dagger Rbeta and filled Ref4 gives p=.25 Tr Qsite=.5 exactly. Restricted to Ref4, E^dagger E=N_empty_site/4<=1; E and its reverse on Ref3 are physical contractions. E is an SU4 scalar but no SU4 law is silently added.',
                    raw_common_phase='Href=-G/2 d2+kappa_eff phi2/2+sum_s dGamma(h_s(0)), INCLUDING Esea0. Reference K=<Uref Psi,E0^dagger Uref E0 Psi>; both legs carry identical sea0 and oscillator unitary, cancelling by the exact inner product. This is physical Heisenberg common pairing under SAMEH, not normal ordering or a privately subtracted vacuum phase. Kref=(1/(2V))sum_AP64 exp(-i Eactive(k)t).',
                    Duhamel='Exact conjugate Htilde=-G/2(partial+A_CAR)^2+primitive potential+diagonal finite fermion energies. On Uref f Omega: vacuum difference norm<=D; on Uref E0 f Omega<=sqrt(p)(D+B sqrtX). Both endpoints Etilde-E0<=2||E|| Cactive |phi|. Thus |K-Kref|<=||E||sqrt(p)tD+||E||(sqrt(p)t(D+B sqrtX)+eta0)+eta_t, ||E||<=2.',
                    domain='Finite Hermitian polynomial matrix potential with positive phi4 dominates its finite linear fermion norm, so Friedrichs H is selfadjoint and semibounded on full real line. Smooth Kato rotations have globally bounded first/second derivatives from h3 gap; Gaussian and all polynomial moments put both prepared vacuum/source in D(H). Duhamel uses this domain, unitary norm conservation, exact Gaussian harmonic evolution; no Fock/field/phi-tail cutoff.'))


def global_gauss_extension(proof: dict[str, Any], g0: Fraction, gs: Fraction, path: Path) -> dict[str, Any]:
    """Separately declared global Gauss law; the scalar-only proof stays intact."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != GAUSS_RECEIPT_SHA or hashlib.sha256(SOURCE_BYTES[Path(global_gauss.__file__).name]).hexdigest() != GAUSS_SOURCE_SHA:
        raise ArithmeticError('Completed frozen global Gauss source/receipt required')
    receipt = json.loads(raw)
    descriptor = receipt['input_descriptor']
    if descriptor['declared_actual_POINT_g0'] != str(g0) or descriptor['declared_actual_POINT_gs'] != str(gs) or descriptor['conserved_total_finite_CAR_particles'] != 8324:
        raise ArithmeticError('Global Gauss point/species/source epoch mismatch')
    sigma_upper = Fraction(proof['optimized_variance_sigma_squared']['upper'])
    harmonic_variance_upper = Fraction(proof['harmonic_evolution_variance_upper']['upper'])
    initial = global_gauss.global_gauss_prices(g0,gs,64,8324,sigma_upper)
    # The exact optimized Gaussian is squeezed relative to Href. Use its
    # all-time harmonic variance envelope, NOT an untrue stationary assumption.
    reference = global_gauss.global_gauss_prices(g0,gs,64,8324,harmonic_variance_upper)
    chi = exact(Fraction(2)**-4096)
    eta = cap(chi*exact(reference['total_reference_packet_residual_divided_chi_upper']))
    rotational = cap(chi*exact(initial['rotational_operator_norm_divided_chi_upper']))
    geometric = cap(chi*exact(initial['geometric_packet_action_norm_divided_chi_upper']))
    rows = []
    for scalar in proof['raw_source_times']:
        t = Fraction(scalar['time'])
        additional = cap(8*exact(t)*eta)
        total_error = cap(exact(Fraction(scalar['absolute_complex_error_upper']['upper']))+additional)
        real = uniform.enclosure(Fraction(scalar['raw_free_source_real']['lower']),Fraction(scalar['raw_free_source_real']['upper']))
        imag = uniform.enclosure(Fraction(scalar['raw_free_source_imag']['lower']),Fraction(scalar['raw_free_source_imag']['upper']))
        rr = uniform.enclosure(endpoint(real,0)-endpoint(total_error,1),endpoint(real,1)+endpoint(total_error,1))
        ii = uniform.enclosure(endpoint(imag,0)-endpoint(total_error,1),endpoint(imag,1)+endpoint(total_error,1))
        modulus = iv.sqrt(real*real+imag*imag)
        lower = max(Fraction(0),endpoint(modulus-total_error,0))
        norm_deviation = cap(2*exact(t)*eta)
        # Scalar evolved-source probability is already enclosed. This is the
        # extra vacuum-leg perturbation under the explicitly NEW Gauss law.
        scalar_p = scalar['physical_E_transfer_probability_on_evolved_vacuum']
        p_lower = max(Fraction(0),endpoint(iv.sqrt(exact(Fraction(scalar_p['lower'])))-norm_deviation,0))**2
        p_upper = min(Fraction(1),endpoint((iv.sqrt(exact(Fraction(scalar_p['upper'])))+norm_deviation)**2,1))
        rows.append(dict(time=str(t), raw_same_H_global_Gauss_source_real=pack(rr),raw_same_H_global_Gauss_source_imag=pack(ii),
                         additional_Noether_volume_complex_error_upper=pack(additional),total_scalar_PLUS_Gauss_complex_error_upper=pack(total_error),
                         actual_nonzero_modulus_lower=pack(exact(lower)),
                         raw_compound_source_return_probability=pack(uniform.enclosure(lower*lower,min(Fraction(1,2),endpoint((modulus+total_error)**2,1)))),
                         physical_E_transfer_probability_on_evolved_global_Gauss_vacuum=pack(uniform.enclosure(p_lower,p_upper))))
    if Fraction(rows[-1]['actual_nonzero_modulus_lower']['lower']) <= 0:
        raise ArithmeticError('Global Gauss actual4096 physical source nonzero discrimination failed')
    return dict(law='NEW_SCALAR_UNIFORM_PHI35_EXACT_CAR_PLUS_GLOBAL_SU4_GAUSS', scalar_only_law_UNCHANGED=True,
                frozen_companion_receipt=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),input_sha256=receipt['input_sha256']),
                actual_Born_sigma_squared_upper=str(sigma_upper),actual_alltime_reference_variance_upper=str(harmonic_variance_upper),
                exact_initial_Gauss_prices={k:pack(exact(v)) for k,v in initial.items()},
                exact_harmonic_reference_Gauss_prices={k:pack(exact(v)) for k,v in reference.items()},
                additional_reference_residual_norm_upper=pack(eta),
                additional_Rayleigh_energy_interval=pack(uniform.enclosure(-endpoint(geometric,1),endpoint(rotational+geometric,1))),
                actual_global_Gauss_prepared_Rayleigh_residual_norm_upper=pack(cap(exact(Fraction(proof['actual_prepared_Rayleigh_residual_norm_upper']['upper']))+2*(rotational+geometric))),
                global_Gauss_raw_source_times=rows,
                physical_reconstruction=receipt['actual_global_Gauss_Noether_scalar_source_prices']['Gauss_reconstruction'],
                proof='Use the frozen all-phi orbit gap and actual finite-CAR Noether operator bounds, with THIS actual Born packet and its squeezed harmonic envelope. The positive rotational operator and signed volume potential are included on both Duhamel legs; extra source error<=2t||E||^2 eta=8t eta. Source is the same physical all4 Ref contraction and commutes all15 global generators. No common vacuum phase is removed.',
                excluded_admissions=['local SU4','full90 Phi/colour','Theta/Pi','Coulomb','original SOURCE','true ground','common cone','scattering'])


def run(path: Path, gauss_path: Path) -> dict[str, Any]:
    started = time.monotonic(); raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECEIPT_SHA or hashlib.sha256(SOURCE_BYTES[Path(uniform.__file__).name]).hexdigest() != UNIFORM_SHA:
        raise ArithmeticError('Authoritative frozen uniform0c3 source/ee8 receipt required')
    frozen = json.loads(raw); mean = frozen['input_descriptor']['actual_root_mean_input_UNCHANGED']
    iv.dps = DIGITS; mp.mp.dps = 80
    g0, gs = uniform.fast.point_mean(mean)
    point = frozen['input_descriptor']['declared_NEW_POINT_parameters']
    if str(g0) != point['g0_exact_binary64'] or str(gs) != point['gs_exact_binary64']:
        raise ArithmeticError('Authoritative uniform binary point mismatch')
    profiles = uniform.fast.primitive_profiles(mean)
    points = [uniform.uniform_bloch(p,g0,gs) for p in profiles]
    actual = literal_construction(mean,profiles,points); proof = exact_proof(profiles,points,g0)
    control = actual['free_reference_literal_source_time_1over4096']; reference = proof['raw_source_times'][1]
    difference = abs(complex(control['real'],control['imag'])-complex(float(Fraction(reference['raw_free_source_real']['lower'])),float(Fraction(reference['raw_free_source_imag']['lower']))))
    if difference > 1e-11:
        raise ArithmeticError('Literal physical source/free-reference spectral consumer differs')
    gauss = global_gauss_extension(proof,g0,gs,gauss_path)
    desc = dict(plan='issue-222-induced-photon-delta13-continuation-20261008', assigned_revision=uniform.ASSIGNED_REVISION,
                bound_scientific_revision=uniform.BOUND_REVISION, template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
                law='NEW_SCALAR_UNIFORM_PHI35_EXACT_FINITE_CAR_REAL_QUANTUM', source='E0=.5 sum4 c_active_site0_alpha^dagger R_alpha; Ref4 static, no whitening',
                omitted_coordinates=['other44 normalPhi','all local SU4/Noether','Theta','Pi','Coulomb'],
                native_primitive_law_UNCHANGED=True, original_SOURCE_admission=False,
                authoritative_uniform_receipt=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),
                separately_declared_global_Gauss_extension_input=gauss['frozen_companion_receipt'],
                unchanged_actual_root_mean_input=mean, exact_binary_POINT=dict(g0=str(g0),gs=str(gs)),
                parameters=dict(N=4,V=64,a='1/64',UEphys=1,u=64,Lambda='2^8192',chi='2^-4096',all_species='neutral heavy1, active charged1 m.2, auxiliary charged63 m2',heavy_v=.3,charged_v=.4990360130634525,r=.12),
                all_phase_law='No Hamiltonian normal ordering/common vacuum phase removal. SAME raw H physical Heisenberg source pairing.', precision_decimal_digits=DIGITS,
                representation='Full finite exterior CAR and L2(R), actual normalized Kato Gaussian preparation; NO RPA/PH boson stand-in')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),
                actual_literal_fermionic_scalar_and_source=actual, actual_exact_unitary_source_proof=proof,
                actual_separate_global_Gauss_unitary_source_proof=gauss,
                literal_free_source_reference_difference=difference,
                source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
                source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__),elapsed_seconds=time.monotonic()-started,
                adverse_facts=['Restricted scalar NEW law, not original full H or original SOURCE.','Gaussian prepared state is not certified true ground.','Return probability is raw compound E/evolution/E^dagger/moving-vacuum projection, not exclusive receiver scattering.','Native mean is exact declared binary POINT, not a newly proved selfconsistent interacting mean.','Duhamel intervals prove dynamics without diagonalizing a truncated scalar oscillator.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uniform', type=Path, required=True)
    parser.add_argument('--global-gauss', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = run(args.uniform,args.global_gauss)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Scientific source changed during actual execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    proof = result['actual_exact_unitary_source_proof']
    print(json.dumps(dict(input_sha256=result['input_sha256'], sigma_squared=proof['optimized_variance_sigma_squared'],
                          exact_FS_metric=proof['exact_mean_all65_FS_metric'], Rayleigh_residual=proof['actual_prepared_Rayleigh_residual_norm_upper'],
                          physical_raw_source_times=proof['raw_source_times'], global_Gauss_times=result['actual_separate_global_Gauss_unitary_source_proof']['global_Gauss_raw_source_times'], elapsed_seconds=result['elapsed_seconds']),indent=2))
