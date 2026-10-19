"""NEW full45 uniform real-Phi quantum law with all65 finite fermion CAR.

The centre is the proved finiteN4 implicit stationary Gram root. The reference
Gaussian uses EXACT A=sqrt(Keff/G) as matrix functional calculus, not displayed
NumPy eigenvectors/covariance. The actual preparation is its normalized compact
radial p=6 bump times the exact radial occupied-projector Kato Slater frame.
All45 coordinates remain jointly coupled. No scalar h3 global-gap argument is
used: a genuine gap ball, a smooth global unitary retraction, and positive
Gaussian tail bounds price the SAME-H physical Heisenberg source at4096.

The scalar45-CAR law and its separately declared NEW uniform60-CAR globalGauss
tubular law are both priced. Spatial Phi, localGauss, Theta/Pi and Coulomb stay
explicitly omitted, not removed from the native primitive law.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES: dict[str, bytes] = {Path(__file__).name: Path(__file__).read_bytes()}


class CapturedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes(); SOURCE_BYTES[Path(self.path).name] = raw
        return self.source_to_code(raw,self.path)


class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.origin is not None:
            source = Path(spec.origin)
            if source.parent == ROOT and source.suffix == '.py':
                spec.loader = CapturedLoader(fullname,str(source)); return spec
        return None


_finder = CapturedFinder(); sys.meta_path.insert(0,_finder)
import induced_phi_uniform_fermionic_source_unitary as scalar
import induced_phi_uniform_full_normal_quantum as full
import induced_phi_uniform_selfconsistent_full_normal_quantum as stationary
import induced_phi_full_normal_global_gauss as gauss
sys.meta_path.remove(_finder)
SOURCE_BYTES.update(scalar.SOURCE_BYTES); SOURCE_BYTES.update(full.SOURCE_BYTES)
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

iv: Any = scalar.iv
exact, endpoint, pack, cap = scalar.exact, scalar.endpoint, scalar.pack, scalar.cap
absolute_cap, maximum = scalar.absolute_cap, scalar.maximum
enclosure = scalar.uniform.enclosure
FULL_SOURCE_SHA = '2832ab998db0bcef0f21fc984a40d63f4e953d36674a63058563a9612260db80'
SCALAR_SOURCE_SHA = 'a592f31ac7d78db94b0bbc17a59924cc09e87cb04ab51049d25b50c87a161196'
STATIONARY_SOURCE_SHA = 'f63c1590af4fa9922757615b1bfccb788f430760d401d7c66372e8656163251c'
ROOT_RECEIPT_SHA = '4691ffaa9cd81902942cbc228fc47fafff14518be6b5df2e84d08c29c7da4756'
FULL_RECEIPT_SHA = '5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354'
GAUSS_SOURCE_SHA = 'ab8c54180797378516aed2056192d71ffc2be10e99368bf064acb2a21325d294'
DIMENSION = 45
BUMP_POWER = 6
TAIL_BITS = 32768
DIGITS = 3000


def interval(record: dict[str, str]) -> Any:
    return enclosure(F(record['lower']),F(record['upper']))


def midpoint(value: Any) -> F:
    return (endpoint(value,0)+endpoint(value,1))/2


def radial_moment(cmax: Any, degree: int) -> Any:
    """sqrt E|eta|^(2degree), dominated by exact isotropic Gaussian covariance."""
    coefficient = 1
    for j in range(degree):
        coefficient *= DIMENSION+2*j
    return cap(iv.sqrt(exact(coefficient)*cmax**degree))


def polynomial_cutoff(t: float) -> tuple[float, float]:
    """Integral of 12012 t6(1-t)6, C6 at both endpoints."""
    if t <= 0:
        return 0.,0.
    if t >= 1:
        return 1.,0.
    coefficients = (1,-6,15,-20,15,-6,1)
    S = 12012*sum(c*t**(7+j)/(7+j) for j,c in enumerate(coefficients))
    primitive = 12012*sum(c*t**(8+j)/((7+j)*(8+j)) for j,c in enumerate(coefficients))
    return S,primitive


def radial_retraction(eta: np.ndarray, radius: float) -> np.ndarray:
    r = float(np.linalg.norm(eta))
    if r <= radius:
        return eta.copy()
    if r >= 2*radius:
        return eta*(1.5*radius/r)
    t = (r-radius)/radius
    _,primitive = polynomial_cutoff(t)
    rho = radius*(1+t-primitive)
    return eta*(rho/r)


def bloch_points(profiles: list[dict[str, Any]], g0: Any, gs: Any) -> list[list[dict[str, Any]]]:
    roots = [iv.sqrt(g0),iv.sqrt(gs),iv.sqrt(gs),iv.sqrt(gs)]
    half = iv.sqrt(exact(2))/2
    cos1,sin1,sin2 = (1,-1,-1,1),(1,1,-1,-1),(1,-1,1,-1)
    all_points = []
    for profile in profiles:
        points = []
        for k in product(range(4),repeat=3):
            xi = [exact(F(profile['onsite'])),exact(0),exact(0),exact(0)]
            for path in profile['paths']:
                axis,L = path['axis'],path['length']
                cosine = cos1[k[axis]]*half if L == 1 else exact(0)
                sine = sin1[k[axis]]*half if L == 1 else exact(sin2[k[axis]])
                xi[0] += 2*exact(F(path['c0']))*cosine
                xi[axis+1] -= 2*exact(F(path['ci']))*sine
            h = [r*x for r,x in zip(roots,xi,strict=True)]
            E = iv.sqrt(sum(x*x for x in h))
            # Sum_m Tr(deltaPhi_m^2)=|eta|²/2, uniform f=1/8.
            D = iv.sqrt(sum(x*x for x in xi))/(8*iv.sqrt(exact(2)))
            points.append(dict(k=k,xi=xi,h=h,E=E,D=D))
        all_points.append(points)
    return all_points


def projector_connection(h: np.ndarray, derivative: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    energies,U = np.linalg.eigh(h)
    if not energies[1] < 0 < energies[2]:
        raise ArithmeticError('Literal full45 occupied/unoccupied gap lost')
    P = U[:,:2]@U[:,:2].conj().T
    Fblock = U.conj().T@derivative@U
    derivative_P = np.zeros((4,4),complex)
    derivative_P[:2,2:] = -Fblock[:2,2:]/(energies[None,2:]-energies[:2,None])
    derivative_P[2:,:2] = derivative_P[:2,2:].conj().T
    derivative_P = U@derivative_P@U.conj().T
    # Only the occupied/empty gap is divided by. Never split a degenerate band.
    return P,derivative_P@P-P@derivative_P


def literal_geometry(root: dict[str, Any], full_receipt: dict[str, Any]) -> tuple[dict[str, Any],dict[str, Any]]:
    g0,gs = interval(root['parameters']['g0']),interval(root['parameters']['gs'])
    mean = dict(microscopic_config=full_receipt['input_descriptor']['original_micro_config_UNCHANGED'])
    mean['centre_enclosure'] = {name+'_'+side:dict(numerator=str(value.numerator),denominator=str(value.denominator)) for name,value in zip(('g0','gs'),(midpoint(g0),midpoint(gs)),strict=True) for side in ('lower','upper')}
    native = full.supplier.quantum.build_native(mean,4)
    geom = full.normal_geometry(native,[g0,gs,gs,gs])
    if geom['labels'] != full_receipt['actual_full45_geometry']['labels']:
        raise ArithmeticError('Exact normal45 basis/consumer ordering mismatch')
    return native,geom


def literal_full45(root: dict[str, Any], full_receipt: dict[str, Any], points: list[list[dict[str, Any]]], radius: F) -> dict[str, Any]:
    native,geom = literal_geometry(root,full_receipt)
    gamma = full.supplier.primitive.SELECTED
    gamma5 = gamma[0]@gamma[1]@gamma[2]@gamma[3]
    rng = np.random.default_rng(22245)
    directions = np.eye(45)[[0,10,20,30,44]].tolist()
    random = rng.normal(size=45); random /= np.linalg.norm(random); directions.append(random.tolist())
    coordinates = [np.zeros(45),*[float(radius)*.8*np.asarray(d) for d in directions]]
    source_controls = []; species_controls = []; kato_errors = []; positive_gap = []; chiral_breaking = []
    for species_index,point_rows in enumerate(points):
        sign = -1 if species_index == 0 else 1
        matrices = np.asarray([sign*sum(float(midpoint(c))*gamma[j] for j,c in enumerate(point['h'])) for point in point_rows])
        derivatives = np.asarray([sign*np.einsum('m,amij->aij',[float(midpoint(x)) for x in point['xi']],geom['deltas'])/8 for point in point_rows])
        sphere_norm_bound = max(float(endpoint(point['D'],1)) for point in point_rows)
        for eta in coordinates:
            h = matrices+np.einsum('a,kaij->kij',eta,derivatives)
            eigenvalues,vectors = np.linalg.eigh(h)
            if not np.all((eigenvalues[:,1]<0)&(eigenvalues[:,2]>0)):
                raise ArithmeticError('Literal certifiedball BOTH filled bands failed')
            positive_gap.append(float(np.min(eigenvalues[:,2]-eigenvalues[:,1])))
            Pmom = vectors[:,:,:2]@vectors[:,:,:2].conj().transpose(0,2,1)
            norms = np.linalg.norm(h-matrices,ord=2,axis=(1,2))
            if float(np.max(norms)) > sphere_norm_bound*float(np.linalg.norm(eta))+1e-11:
                raise ArithmeticError('Actual ALL45 joint directional norm exceeds analytical bound')
            chiral_breaking.append(float(np.max(abs(h@gamma5+gamma5@h))))
            if species_index == 1:
                Psite = np.mean(Pmom,axis=0)
                source_controls.append(dict(coordinate_norm=float(np.linalg.norm(eta)),actual_ALL64_momentum_negative_ranks=[int(np.sum(row<0)) for row in eigenvalues],literal_source=scalar.literal_car_source(Psite)))
        # Actual radial Kato ODE, no within-band eigenvector transport/division.
        eta = coordinates[-1]; j = 0; Fdirection = np.einsum('a,aij->ij',eta,derivatives[j])
        P0,_ = projector_connection(matrices[j],Fdirection)
        def ode(t: float,y: np.ndarray) -> np.ndarray:
            _,B = projector_connection(matrices[j]+t*Fdirection,Fdirection)
            return (B@y.reshape(4,4)).reshape(16)
        trajectory = solve_ivp(ode,(0.,1.),np.eye(4,dtype=complex).reshape(16),rtol=1e-11,atol=1e-13)
        if not trajectory.success:
            raise ArithmeticError(trajectory.message)
        U = trajectory.y[:,-1].reshape(4,4); P1,_ = projector_connection(matrices[j]+Fdirection,Fdirection)
        kato_errors.append(dict(projector_transport=float(np.linalg.norm(U@P0@U.conj().T-P1)),unitarity=float(np.linalg.norm(U.conj().T@U-np.eye(4)))))
        species_controls.append(dict(species_prototype=species_index,physical_scale='Lambda' if sign<0 else '1',load=(1,1,63)[species_index],literal_ALL45_Bloch_operator_shape=list(derivatives.shape)))
    if max(v for row in kato_errors for v in row.values()) > 2e-10:
        raise ArithmeticError('Literal actual full45 radial Kato transport failed')
    retraction_rows = []
    for r in (0.,.5,1.,1.25,1.75,2.,3.):
        eta = np.zeros(45); eta[0] = r*float(radius)
        transformed = radial_retraction(eta,float(radius))
        retraction_rows.append(dict(input_radius_units_R=r,output_radius_units_R=float(np.linalg.norm(transformed))/float(radius)))
    return dict(geometry_errors=geom['errors'],species=species_controls,actual_source_controls=source_controls,
                actual_radial_occupied_projector_Kato_ODE=kato_errors,minimum_measured_BOTHband_gap=min(positive_gap),
                full45_Gamma5_anticommutator_max=max(chiral_breaking),global_smooth_retraction_observations=retraction_rows,
                source_norm_proof='Not a chiral assumption: every momentum block has exactly rank2 throughout the certified uniform-field ball by gap continuity. AP translation makes Tr Q_site00=(1/64)sum_k(4-rank P_k)=2. Literal physical CAR/Ref4 therefore gives p=.5 on this compact preparation; grade2 chiral-breaking is actually observed. Source endpoint changes are separately priced.')


def constructive_prices(root: dict[str, Any], quantum: dict[str, Any], profiles: list[dict[str, Any]], points: list[list[dict[str, Any]]]) -> dict[str, Any]:
    Lambda = exact(F(2)**8192); G = exact(8*F(2)**-4096)
    lower = exact(F(quantum['certified_eigenvalue_lower_divided_Lambda']['lower']))
    error = interval(quantum['eigen_reconstruction_operator_error'])
    upper = cap(exact(F.from_float(float(max(quantum['actual_full45_effective_eigenvalues_divided_Lambda']))))+error)
    if endpoint(lower,0) <= 0:
        raise ArithmeticError('Actual joint full45 Hessian is not certified SPD')
    cmax = cap(iv.sqrt(G/(Lambda*lower))/2)
    cmin = exact(endpoint(iv.sqrt(G/(Lambda*upper))/2,0))
    amax = cap(1/(2*cmin))
    records = []; C1,C2,sea3,sea4 = [exact(0) for _ in range(4)]
    radius_candidates = [F(1,8192)]; active_D = None; active_C1 = None; active_Emax = None; active_delta = None
    for profile,rows in zip(profiles,points,strict=True):
        scale = Lambda if profile['scale']=='heavy' else exact(1)
        load = exact(profile['load'])
        Emin = exact(min(endpoint(r['E'],0) for r in rows)); Emax = maximum([r['E'] for r in rows]); D = maximum([r['D'] for r in rows])
        radius_candidates.append(endpoint(Emin/(16*D),0))
        delta = Emin/2; d = cap(D/delta)
        # Pderivative_n <=n! (D/delta)^n from the imaginary-axis sign resolvent.
        # Radial Kato: B1<=5d, B2<=20d², W2<=70d², Asecond<=100d².
        C1 += load*64*20*d
        C2 += load*64*400*d*d
        sea3 += load*64*scale*exact(F(4,3))*D**3/delta**2
        sea4 += load*64*scale*D**4/delta**3
        if profile['role']=='active filled charged':
            active_D,active_C1,active_Emax,active_delta = D,64*20*d,Emax,delta
        records.append(dict(role=profile['role'],load=profile['load'],mean_Emin=pack(Emin),mean_Emax=pack(Emax),joint_ALL45_operator_Lipschitz=pack(D),all_ball_projector_derivative_unit=pack(d),ball_PH_gap_lower=pack(Emin),CAR_first_derivative_norm_per_coordinate_upper=pack(cap(64*20*d)),sea_cubic_remainder_upper=pack(cap(64*scale*exact(F(4,3))*D**3/delta**2))))
    if active_D is None or active_C1 is None or active_Emax is None or active_delta is None:
        raise ArithmeticError('Physical active source profile absent')
    R = min(radius_candidates); Rv = exact(R)
    # Retraction is identity on R, maps allspace into1.5R, ||Dtheta||<=1,
    # ||D²theta||<=6/R. This adds W1*D²theta to the global A derivative cap.
    global_C2 = cap(C2+6*C1/Rv)
    g0,gs = interval(root['parameters']['g0']),interval(root['parameters']['gs'])
    gmax = maximum([g0,gs])
    primitive3 = cap(Lambda*iv.sqrt(gmax)/(2*exact(F(1,64))*iv.sqrt(exact(8*64))))
    primitive4 = Lambda/exact(64)
    M = [radial_moment(cmax,j) for j in range(1,5)]
    moment1,moment2,moment3,moment4 = M
    Zlower = exact(1)-cap(2*BUMP_POWER*DIMENSION*cmax/(Rv*Rv))
    if endpoint(Zlower,0) <= 0:
        raise ArithmeticError('Physical compact packet normalizer bound is vacuous')
    normalization = cap(1/iv.sqrt(Zlower))
    preparation_distance = cap(2*BUMP_POWER*moment2/(Rv*Rv))
    tail_log_margin = Rv*Rv/(8*cmax)-exact(F(TAIL_BITS)+F(DIMENSION,4))*iv.ln(exact(2))
    if endpoint(tail_log_margin,0) <= 0:
        raise ArithmeticError('Positive ALLtails dyadic price not established')
    tau = exact(F(2)**-TAIL_BITS)
    tail_moments = [tau,*[tau*radial_moment(2*cmax,j) for j in range(1,5)]]
    gradient = cap(iv.sqrt(exact(DIMENSION)*amax/2))
    derivative = cap(G*iv.sqrt(exact(DIMENSION))*C1*gradient)
    second = cap(G*DIMENSION*(C2+C1*C1)/2)
    nonlinear = cap((primitive3+sea3)*moment3+primitive4*moment4)
    active_lipschitz = cap(active_D+4*active_Emax*active_D/active_delta)
    charged = cap(normalization*active_lipschitz*moment1)
    endpoint_price = cap(normalization*4*iv.sqrt(exact(DIMENSION))*active_C1*moment1)
    bump_gradient = cap(2*G*BUMP_POWER*iv.sqrt(exact(DIMENSION*(DIMENSION+2)))/(Rv*Rv))
    bump_laplacian = cap(G*(BUMP_POWER*DIMENSION/(Rv*Rv)+2*BUMP_POWER*(BUMP_POWER-1)*moment2/(Rv**4)))
    bump_Kato = cap(2*G*BUMP_POWER*iv.sqrt(exact(DIMENSION))*C1*moment1/(Rv*Rv))
    reference_residual = cap(normalization*(nonlinear+derivative+second+bump_gradient+bump_laplacian+bump_Kato))
    bump_energy = cap(2*G*BUMP_POWER**2*DIMENSION*cmax/(Rv**4*Zlower))
    mean_potential_shift = cap(normalization**2*(primitive4+sea4)*moment2**2)
    born_upper = cap(G*DIMENSION*C1*C1/2)
    energy_shift = cap(bump_energy+mean_potential_shift+born_upper)
    rayleigh_residual = cap(reference_residual+energy_shift)
    # Weighted ORIGINAL-body derivatives, including CAR Kato and compact bump,
    # used by the separate full45/globalSU4 angular/metric companion.
    B1 = cap(normalization*(amax*moment2+2*BUMP_POWER*moment2/Rv**2+iv.sqrt(exact(DIMENSION))*C1*moment1))
    B2 = cap(normalization*(amax**2*moment4+iv.sqrt(exact(DIMENSION))*amax*moment2
         +2*BUMP_POWER*iv.sqrt(exact(DIMENSION))*moment2/Rv**2
         +4*BUMP_POWER*(BUMP_POWER-1)*moment4/Rv**4+4*BUMP_POWER*amax*moment4/Rv**2
         +exact(DIMENSION)*(C1*C1+C2)*moment2
         +2*iv.sqrt(exact(DIMENSION))*C1*(amax+2*BUMP_POWER/Rv**2)*moment3))
    primitive_centre = Lambda*exact(4096)*(g0*g0+3*gs*gs)
    occupied_centre = sum(-2*exact(profile['load'])*(Lambda if profile['scale']=='heavy' else exact(1))*sum(row['E'] for row in rows) for profile,rows in zip(profiles,points,strict=True))
    centre_energy = primitive_centre+occupied_centre
    harmonic_energy = enclosure(endpoint(DIMENSION*iv.sqrt(G*Lambda*lower)/2,0),endpoint(DIMENSION*iv.sqrt(G*Lambda*upper)/2,1))
    reference_energy = centre_energy+harmonic_energy
    prepared_energy = reference_energy+enclosure(-endpoint(energy_shift,1),endpoint(energy_shift,1))
    times = []; p = exact(F(1,2)); rootp = iv.sqrt(p)
    for t in (F(0),F(1,4096),F(4096)):
        re = sum(iv.cos(row['E']*exact(t)) for row in points[1])/128
        im = -sum(iv.sin(row['E']*exact(t)) for row in points[1])/128
        vac_error = cap(exact(t)*reference_residual)
        src_error = cap(endpoint_price+exact(t)*rootp*(reference_residual+charged))
        error = cap(2*rootp*vac_error+2*src_error+rootp*endpoint_price)
        if not t:
            error = exact(0)
        rr = enclosure(endpoint(re,0)-endpoint(error,1),endpoint(re,1)+endpoint(error,1))
        ii = enclosure(endpoint(im,0)-endpoint(error,1),endpoint(im,1)+endpoint(error,1))
        modulus = iv.sqrt(re*re+im*im); lower_K = max(F(0),endpoint(modulus-error,0))
        times.append(dict(time=str(t),raw_free_reference_real=pack(re),raw_free_reference_imag=pack(im),raw_actual_same_H_source_real=pack(rr),raw_actual_same_H_source_imag=pack(ii),complex_error_upper=pack(error),nonzero_modulus_lower=pack(exact(lower_K)),raw_compound_return_probability=pack(enclosure(lower_K**2,min(F(1,2),endpoint((modulus+error)**2,1)))),vacuum_leg_norm_price=pack(vac_error),source_leg_norm_price=pack(src_error)))
    if F(times[-1]['nonzero_modulus_lower']['lower']) <= 0:
        raise ArithmeticError('Actual joint full45 raw source nonzero at4096 failed')
    return dict(radius_exact=str(R),radius=pack(Rv),dimension=DIMENSION,bump_power=BUMP_POWER,
                exact_reference_wave='A=(Keff/G)^(1/2), Cov=(1/2)A^-1, at the unique implicit finiteN4 root; EXACT joint matrix functional, not rounded modes. Keff=primitive full45 Hessian minus ALL65 actual static PH trace. COMPARISON Href=Ephi_exact I_field+dGamma(hmean) fixes the actual compact packet; Ephi=G Tr A/2 is NOT silently substituted into physical H.',
                certified_Keff_eigen_lower_divided_Lambda=pack(lower),certified_Keff_eigen_upper_divided_Lambda=pack(upper),covariance_eigen_min=pack(cmin),covariance_eigen_max=pack(cmax),
                all65_species_gap_and_connection=records,compact_normalizer_Z_lower=pack(Zlower),compact_packet_distance_to_joint_Gaussian_upper=pack(preparation_distance),
                ALLGaussian_tail_L2_factor_positive_upper=pack(tau),tail_log_margin=pack(tail_log_margin),weighted_Gaussian_tail_norm_bounds=[pack(x) for x in tail_moments],
                global_CAR_connection_first_per_coordinate=pack(cap(C1)),global_CAR_connection_second_per_coordinate=pack(global_C2),
                support_CAR_connection_second_per_coordinate=pack(cap(C2)),
                weighted_body_r_gradient_norm_upper=pack(B1),weighted_body_r_squared_Hessian_norm_upper=pack(B2),
                primitive_cubic_coefficient_upper=pack(primitive3),primitive_quartic_coefficient_upper=pack(primitive4),actual_all65_sea_cubic_remainder_upper=pack(cap(sea3)),actual_all65_sea_quartic_mean_remainder_upper=pack(cap(sea4)),
                residual_prices=dict(primitive_PLUS_sea_cubic_and_primitive_quartic=pack(nonlinear),Kato_first=pack(derivative),Kato_second_PLUS_square=pack(second),active_added_band=pack(charged),physical_source_endpoint=pack(endpoint_price),bump_gradient=pack(bump_gradient),bump_laplacian=pack(bump_laplacian),bump_Kato=pack(bump_Kato)),
                actual_prepared_reference_energy_residual_norm_upper=pack(reference_residual),actual_prepared_Rayleigh_residual_norm_upper=pack(rayleigh_residual),actual_prepared_energy_shift_from_exact_reference_upper=pack(energy_shift),
                raw_primitive_centre_energy=pack(primitive_centre),raw_ALL65_occupied_centre_energy=pack(occupied_centre),
                raw_exact_harmonic_trace_energy_enclosure=pack(harmonic_energy),raw_same_H_reference_energy_enclosure=pack(reference_energy),
                actual_prepared_Rayleigh_energy_enclosure=pack(prepared_energy),
                raw_same_H_reference_energy='Ecentre=sum_actual_occupied h_s(0)+full primitive background Lambda V/a(g0²+3gs²), PLUS exact harmonic G Tr A/2. ALL common phases retained in both Heisenberg legs.',
                source_probability_exact_prepared='1/2 from certified uniform momentum-rank2 and literal CAR, NOT assumed Gamma5 symmetry away from mean. Source/receiver are physical Ref4↔3 contractions; compound probability |K|², no whitening.',
                raw_source_times=times,
                construction_proof=dict(gap='||delta h_s(k)||<=D_s||eta|| jointly for all45 exact normal coordinates. R<=Emin/(16D); the entire2R ball has at least half the mean distance to zero for every actual species. Heavy scaling cancels in projector bounds and is retained in energy bounds.',
                    Kato='Pnegative is defined by the sign-resolvent on the separated ball. ||directional Pn||<=n!(D/delta)^n. Radial transport d_t W=[eta·grad P(teta),P(teta)]W uniquely maps occupied subspaces, no eigenvector choice or within-band division. Its CAR lift is the full finite exterior unitary, all65 species plus static Ref4.',
                    retraction='theta(eta)=rho(r)eta/r, rho=r insideR; on[R,2R], rho=R[1+t-integral0t S(u)du], S(t)=12012 integral0t u6(1-u)6du; rho=1.5R after2R. C6 matching, Dtheta<=1 and D²theta<=6/R. W(theta) is a global smooth unitary extending the actual gap-ball frame, solely for global operator equivalence/domain. Reference compact packet stays fixed under comparison Href, so no offball spectral or harmonic-propagation assumption is used.',
                    compact_domain='psi=Z^-1/2 f_A(eta)(1-r²/R²)^6 insideR, zero outside. This is C5/H4 and lies in D(H²), unlike a C1 cutoff. Its Kato lift has bounded derivatives on support and finite polynomial moments. Normalizer Z>=1-12 E r²/R²>0, distance to normalized Gaussian<=12 sqrt(E r4)/R². No Gaussian clipping without a price.',
                    derivatives='Radial Kato generator derivatives B1<=5d, B2<=20d² imply W2<=70d² and Ajprime<=100d². Each four-mode CAR block costs at most four times its one-particle norm;64momentum blocks and loads1+1+63 sum into C1,C2. Retraction contributes6 C1/R. All45 kinetic terms -G sum Aj partialj and -G/2 sum(Ajprime+Aj²) are retained.',
                    nonlinear='Primitive Gram: ||Jlinear||F<=2sqrt(gmax)r/sqrt(8V), ||Dquadratic||F<=r²/(8V). Therefore cubic<=Lambda sqrt(gmax)/(2a sqrt(8V)) r³, quartic<=Lambda/(64aV) r4. True filled-sea Ethird=Tr(Psecond F), Efourth=Tr(Pthird F), giving perblock cubic4D³/(3delta²), fourth D4/delta³. Mean stationarity is exact at implicit root, not a counterterm; cubic expectations vanish by even packet parity but cubic residual is retained.',
                    tails='For exact Gaussian Cov<=cmax I, Chernoff tilt gives ||r^m 1_r>R f||<=exp[-R²/(8cmax)]2^(45/4)(2cmax)^(m/2)sqrt(product_j(45+2j)). Actual positive log margin proves factor<=2^-32768; finite positive dyadic, NEVERzero. Normalizer/moment and compact-vs-Gaussian comparison are priced; the actual prepared packet is a deliberate smooth compact state, not a secretly clipped Gaussian.',
                    Heisenberg='K=<U psi,Ephi^dag U Ephi psi> under SAME raw full45 scalar-CAR H. Comparison Href=Ephi_exact I_field+dGamma(hmean) fixes THIS normalized compact joint packet. Its common Ecentre+Ephi phase is retained in both referencelegs and cancels only in the physical paired inner product. Kref=(1/128)sum_AP64 exp(-i Eactive(k)t). Actual physical H keeps kinetic, full primitive, exact finite CAR and Kato connection; its discrepancy includes the explicitly nonzero compact-bump harmonic residual. Duhamel prices both legs, charged block variation and both physical source endpoints. No physical fallback, phase subtraction, clock rescaling or claim of an exact bump ground state.'))


def global_gauss_extension(root: dict[str, Any], full_receipt: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct actual60Cart kinetic on the SAME prepared joint45 CAR packet."""
    params = root['parameters']; R = F(base['radius_exact'])
    B1 = F(base['weighted_body_r_gradient_norm_upper']['upper'])
    B2 = F(base['weighted_body_r_squared_Hessian_norm_upper']['upper'])
    prices = gauss.full_normal_gauss_prices(F(params['g0']['lower']),F(params['gs']['lower']),F(params['g0']['upper']),F(params['gs']['upper']),64,8324,R,B1,B2)
    residual = exact(F(2)**-4096*prices['total_reference_packet_Noether_and_metric_residual_divided_chi_upper'])
    native,geom = literal_geometry(root,full_receipt)
    chart = gauss.literal_chart(native,geom['Qcart'],R)
    times = []
    for row in base['raw_source_times']:
        t = F(row['time']); additional = exact(8*abs(t))*residual
        error = interval(row['complex_error_upper'])+additional
        re,im = interval(row['raw_free_reference_real']),interval(row['raw_free_reference_imag'])
        modulus = iv.sqrt(re*re+im*im); lower_K = max(F(0),endpoint(modulus-error,0))
        times.append(dict(time=str(t),additional_global_Noether_metric_complex_error_upper=pack(additional),
            total_complex_error_upper=pack(error),
            raw_actual_global60_same_H_source_real=pack(enclosure(endpoint(re,0)-endpoint(error,1),endpoint(re,1)+endpoint(error,1))),
            raw_actual_global60_same_H_source_imag=pack(enclosure(endpoint(im,0)-endpoint(error,1),endpoint(im,1)+endpoint(error,1))),
            nonzero_modulus_lower=pack(exact(lower_K)),
            raw_compound_return_probability=pack(enclosure(lower_K**2,min(F(1,2),endpoint((modulus+error)**2,1))))))
    if F(times[-1]['nonzero_modulus_lower']['lower']) <= 0:
        raise ArithmeticError('Actual reconstructed uniform60 globalGauss raw source failed at4096')
    prepared_energy = interval(base['actual_prepared_Rayleigh_energy_enclosure'])+enclosure(-endpoint(residual,1),endpoint(residual,1))
    return dict(law='NEW_UNIFORM60_REAL_PHI_EXACT_FINITE_CAR_GLOBAL_SU4_GAUSS',
        physical_field_coordinates=60,normal_coordinates=45,global_orbit_coordinates=15,
        exact_root_inputs=params,actual_body_weighted_r_gradient_upper=str(B1),actual_body_weighted_r2_Hessian_upper=str(B2),
        actual_exact_rational_Noether_metric_prices={name:pack(exact(value)) for name,value in prices.items()},
        actual_new_stationary_mean_full60_tubular_chart=chart,additional_reference_packet_H_residual_norm_upper=pack(residual),
        actual_prepared_Rayleigh_energy_enclosure=pack(prepared_energy),
        actual_prepared_Rayleigh_residual_norm_upper=pack(interval(base['actual_prepared_Rayleigh_residual_norm_upper'])+2*residual),
        raw_source_times=times,
        physical_state='Psi(U,eta)=U_CAR[W_Kato(eta)Omega f_A(eta)b6(eta)]/sqrt(J(eta)), normalized Haar SU4/Z4 and the SAME normalized joint45 compact body; Ref4 included, n=8324 mod4=0. Exact globalGauss equivariance, physical E0 co-localSU4singlet, unchanged source normalization1/2. D(H²) follows compactC5 regular injective tubular chart and smooth finiteCAR lift.',
        quantum_metric_proof='Actual D=[N,O(eta)] gives M=I/8,B=N^TO,H=O^TO, S=M-BH^-1B^T. Reduced cometric A=S^-1, not frozen8I. A-8I=O(r²), dA=O(r), normal-orbit cross=O(r). All45 Hessian/gradient action is priced using actual Born/Bump/CAR B1,B2. Exact angular/cross Noether and conjugated flatJ volume potential are all retained. Full source/reference states have same n and body derivative bounds. Additional physical twoleg Duhamel error<=8|t|chi total; no private sea/common-phase subtraction.',
        excluded=['local/nonuniform Gauss','nonuniform Phi','all90 collective fields','Theta/Pi','Coulomb','originalSOURCE/eR/commoncone/QED/scattering admission'])


def run(root_path: Path, full_path: Path) -> dict[str, Any]:
    started = time.monotonic(); root_raw = root_path.read_bytes(); full_raw = full_path.read_bytes()
    root_receipt = json.loads(root_raw); full_receipt = json.loads(full_raw)
    if hashlib.sha256(root_raw).hexdigest() != ROOT_RECEIPT_SHA or hashlib.sha256(full_raw).hexdigest() != FULL_RECEIPT_SHA or hashlib.sha256(SOURCE_BYTES[Path(stationary.__file__).name]).hexdigest() != STATIONARY_SOURCE_SHA:
        raise ArithmeticError('Completed frozen NEW stationary root/full45 producer required')
    if hashlib.sha256(SOURCE_BYTES[Path(full.__file__).name]).hexdigest() != FULL_SOURCE_SHA or hashlib.sha256(SOURCE_BYTES[Path(scalar.__file__).name]).hexdigest() != SCALAR_SOURCE_SHA:
        raise ArithmeticError('Frozen2832 full geometry and a592 scalar CAR helpers required')
    if hashlib.sha256(SOURCE_BYTES[Path(gauss.__file__).name]).hexdigest() != GAUSS_SOURCE_SHA:
        raise ArithmeticError('Frozen NEW full45normal/globalSU4 metric supplier required')
    if not all(root_receipt['source_unchanged_since_import'].values()) or not all(full_receipt['source_unchanged_since_import'].values()):
        raise ArithmeticError('Upstream scientific source epoch incomplete')
    root = root_receipt['actual_stationary_root']
    if root['parameters'] != full_receipt['actual_NEW_finiteN4_stationary_mean']['parameters']:
        raise ArithmeticError('Mean/full45 stiffness root epochs differ')
    iv.dps = DIGITS; mp.mp.dps = 80
    profiles = root_receipt['input_descriptor']['exact_profile_paths']
    if [p['load'] for p in profiles] != [1,1,63]:
        raise ArithmeticError('Actual ALL65 species contract changed')
    g0,gs = interval(root['parameters']['g0']),interval(root['parameters']['gs'])
    points = bloch_points(profiles,g0,gs)
    prices = constructive_prices(root,full_receipt['actual_full45_coupled_quantum_consumer'],profiles,points)
    actual = literal_full45(root,full_receipt,points,F(prices['radius_exact']))
    global_proof = global_gauss_extension(root,full_receipt,prices)
    descriptor = dict(law='NEW_FULL45_SCALAR_CAR_AND_NEW_UNIFORM60_GLOBAL_SU4_GAUSS',plan='issue-222-induced-photon-delta13-continuation-20261008',assigned_scientific_revision=scalar.uniform.ASSIGNED_REVISION,
                      parameters=dict(N=4,V=64,a='1/64',UEphys=1,u=64,Lambda='2^8192',chi='2^-4096',G='8chi',all_bulk_species=65,staticRef=4,finite_CAR_modes=16644,total_particles=8324),
                      source='E0=.5 sum4 c_active0alpha^dagger Ralpha, physical CAR/staticRef4->3, no whitening',
                      root_input=dict(path=str(root_path),sha256=hashlib.sha256(root_raw).hexdigest(),input_sha256=root_receipt['input_sha256']),
                      full45_input=dict(path=str(full_path),sha256=hashlib.sha256(full_raw).hexdigest(),input_sha256=full_receipt['input_sha256']),
                      exact_root_parameters=root['parameters'],exact_matrix_functional_wave_ABI=full_receipt['actual_exact_functional_Gaussian_wave_ABI'],
                      full60_metric_supplier=dict(source_sha256=GAUSS_SOURCE_SHA,body_B1_upper=global_proof['actual_body_weighted_r_gradient_upper'],body_B2_upper=global_proof['actual_body_weighted_r2_Hessian_upper']),
                      omitted_terms=['nonuniform Phi','local/nonuniform Gauss','Theta','Pi','Coulomb'],original_source_and_native_primitive_law_UNCHANGED=True)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
                actual_joint_full45_finite_CAR_source_proof=prices,actual_literal_full45_Kato_CAR_consumers=actual,
                actual_uniform60_global_Gauss_finite_CAR_source_proof=global_proof,
                source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
                source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},input_unchanged=dict(root=root_path.read_bytes()==root_raw,full45=full_path.read_bytes()==full_raw),
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__),elapsed_seconds=time.monotonic()-started,
                adverse_facts=['NEW uniform45 scalar-CAR and NEW reconstructeduniform60 globalGauss laws, not full90/localGauss/originalSOURCE.','Exact matrix-functional Gaussian is a preparation, not a true interacting ground.','Compact smooth bump has positive priced normalizer/tails and genuine D(H²) domain.','Gamma5 symmetry DOES break away frommean; source normalization uses uniform fixedrank plus literal CAR, not a chiral hardcode.','All45 coupling/Hessian and full60 variablecometric retained; not45 independent scalar copies or frozenmetric.','No electromagnetic/current/eR/commoncone/scattering admission.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--full45',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); result = run(args.root,args.full45)
    if not all(result['source_unchanged_since_import'].values()) or not all(result['input_unchanged'].values()):
        raise ArithmeticError('Scientific source/input changed during actual full45 quantum source execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    proof = result['actual_uniform60_global_Gauss_finite_CAR_source_proof']
    print(json.dumps(dict(input_sha256=result['input_sha256'],Rayleigh_residual=proof['actual_prepared_Rayleigh_residual_norm_upper'],raw_source_times=proof['raw_source_times'],elapsed_seconds=result['elapsed_seconds']),indent=2))
