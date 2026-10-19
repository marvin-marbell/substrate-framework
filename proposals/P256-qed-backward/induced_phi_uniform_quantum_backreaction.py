"""Uniform q0 Cart-normal Phi35 plus all64/heavy PH quantum Gaussian.

NEW exact-binary-point scalar control. Primitive Lambda/chi/clock unchanged;
no sin(0), no alignment gradient, no full Phi/SU4/original SOURCE promotion.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
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
import induced_phi_quantum_fast_transfer as fast
sys.meta_path.remove(_finder)
SOURCE_BYTES.update(fast.SOURCE_BYTES)
import numpy as np

supplier = fast.supplier
iv: Any = fast.iv
exact = fast.exact
endpoint = fast.endpoint
bounds = fast.bounds
enclosure = fast.enclosure
DIGITS = 1250
ASSIGNED_REVISION = 'd86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7'
BOUND_REVISION = '452ad8053dd6efcbc7d30b55443c1da57c4e6b7e6866b58b11fc61284c6d0961'
FAST_SHA = '8916c7da2d96d3bf10100a7839cc4ba19fed8cb5ec646ea5dee8cccafaa8b0a9'


def uniform_bloch(profile: dict[str, Any], g0: Fraction, gs: Fraction) -> list[dict[str, Any]]:
    root0, roots = iv.sqrt(exact(g0)), iv.sqrt(exact(gs))
    halfroot = iv.sqrt(exact(2))/2
    cos1, sin1, sin2 = (1, -1, -1, 1), (1, 1, -1, -1), (1, -1, 1, -1)
    rows = []
    for k in product(range(4), repeat=3):
        xi0 = exact(profile['onsite']); hs = [iv.mpf(0) for _ in range(3)]
        for p in profile['paths']:
            axis, length = p['axis'], p['length']
            cosine = cos1[k[axis]]*halfroot if length == 1 else exact(0)
            sine = sin1[k[axis]]*halfroot if length == 1 else exact(sin2[k[axis]])
            xi0 += 2*exact(p['c0'])*cosine
            hs[axis] -= 2*exact(p['ci'])*sine
        h = [root0*xi0, *(roots*x for x in hs)]
        E = iv.sqrt(sum(x*x for x in h))
        weight = xi0*xi0*h[3]*h[3]/(4*64*E*E)
        if endpoint(weight, 0) <= 0:
            raise ArithmeticError('Uniform exact PH weight lost strict positivity')
        parallel2 = sum(x*x for x in h[:3])
        # Fourth derivative of the ACTUAL two filled eigenvalues of h+t F.
        quartic = -xi0**4*h[3]**2*(4*parallel2-h[3]**2)/(256*64**2*E**7)
        rows.append(dict(k=k, h=h, xi0=xi0, E=E, delta=2*E, weight=weight,
                         static_filled_sea_quartic_coefficient=quartic))
    return rows


def uniform_geometry(native: dict[str, Any], mode: dict[str, Any]) -> dict[str, Any]:
    primitive, force = supplier.primitive, supplier.static.force
    basis = primitive.matrix_unit_basis(); geom = force.physical_normal_geometry(native)
    conversion = np.real(np.einsum('bij,aji->ba', basis, force.T))/4
    S = np.kron(np.eye(4), conversion); E = S@geom['normal']; O = S@geom['orbit']
    delta = np.zeros((4,4,4), complex); delta[0] = mode['delta']
    cart = force.cart_coordinates(delta).reshape(60); z = S@cart
    gram_linear = np.real(np.einsum('mij,nji->mn', delta, native['frame']))/4
    gram_linear += gram_linear.T
    H = primitive.elastic_hessian(native['frame'], basis, 2/native['a'], 0.)
    restricted = float(z@H@z); expected = native['g0']/(2*native['a'])
    errors = dict(Cart_normal_metric=float(np.max(abs(E.T@E-np.eye(45)/8))),
                  normal_orbit=float(np.linalg.norm(E.T@O)),
                  Gram_linear=float(np.linalg.norm(gram_linear)),
                  spatial_unit=float(abs(np.sum(mode['mode']**2)-1)),
                  restricted_quartic=float(abs(restricted-expected)))
    if max(errors.values()) > 1e-10:
        raise ArithmeticError('Uniform primitive Cartnormal/inertia control failed')
    shifted = native['frame'].copy(); step = .5
    shifted[0] += step*mode['delta']/8
    gram0 = np.real(np.einsum('mij,nji->mn', native['frame'], native['frame']))/4
    gram1 = np.real(np.einsum('mij,nji->mn', shifted, shifted))/4
    literal = 64/native['a']*float(np.sum(gram1**2)-np.sum(gram0**2))
    predicted = expected*step**2/2 + step**4/(64*native['a']*64)
    if abs(literal-predicted) > 1e-10:
        raise ArithmeticError('Literal primitive Gram-square fourth derivative mismatch')
    # A uniform field has identical endpoint matrices on EVERY periodic bond.
    gradient = max(float(np.linalg.norm(mode['mode'][x]-mode['mode'][p['target']]))
                   for x in range(64) for p in native['Pprofiles'][0]['paths'] if p['source'] == x)
    if gradient != 0:
        raise ArithmeticError('Hidden uniform q0 alignment gradient')
    return dict(errors=errors, actual_Cart_to_primitiveB_Pauli_norm_squared=float(z@z),
                actual_kappa_divided_Lambda=restricted, actual_inertia='G=8 chi from E^T E=I45/8',
                actual_uniform_f='1/sqrt64=1/8', actual_qtheta=[0,0,0],
                actual_ALL_ordered_path_alignment_gradient=gradient,
                actual_alignment_Lambda_to_5over4_stiffness=0,
                primitive_energy='Vquartic=(Lambda/a) sum_x sum_mn [Tr(Phi_m Phi_n)/4]^2',
                literal_primitive_energy_increment_divided_Lambda=literal,
                algebraic_primitive_energy_increment_divided_Lambda=predicted,
                primitive_Phi_fourth_derivative_divided_Lambda=24/(64*native['a']*64))


def literal_species(mean: dict[str, Any], profiles: list[dict[str, Any]], points: list[list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    native = supplier.quantum.build_native(mean, 4); V = native['V']; zero = np.zeros(3*V)
    mode = dict(supplier.pauli.magnetic_shape(native)); mode['mode'] = np.ones(V)/np.sqrt(V)
    gamma = native['frame']/np.sqrt([native['g0']]+[native['gs']]*3)[:,None,None]
    if np.linalg.norm(mode['delta']-1j*gamma[0]@gamma[1]@gamma[2]/np.sqrt(8)) > 1e-12:
        raise ArithmeticError('T35 Clifford convention changed')
    geometry = uniform_geometry(native, mode)
    density_e, density_u = np.linalg.eigh(native['Lplus']); positive = density_e > 1e-12
    if int(np.sum(positive)) != 63:
        raise ArithmeticError('Actual native density rank changed')
    density = density_u[:,positive]*np.sqrt(2*native['UE']/native['a']*density_e[positive])[None,:]
    rho = [np.diag(np.repeat(f,4)).astype(complex) for f in density.T]
    rho_uniform = np.eye(4*V, dtype=complex)/np.sqrt(V)
    directions = []
    for axis in range(3):
        direction = np.zeros(3*V); direction[axis::3] = mode['mode']; directions.append(direction)
    indices = np.asarray(list(product(range(4),repeat=3))); momenta = (2*np.pi*indices+np.pi)/4
    spatial_U = np.exp(1j*native['sites']@momenta.T)/np.sqrt(V)
    Ufourier = np.kron(spatial_U, np.eye(4)); rows = []
    for p, pts in zip(profiles, points, strict=True):
        sign = -1 if p['scale'] == 'heavy' else 1
        if sign == 1:
            profile = next(s for s in native['Pprofiles'] if s['label'] == p['role'])
        else:
            paths = []
            for displacement, coefficients in p['raw_paths']:
                support = [s for s in native['Pprofiles'][0]['paths'] if np.array_equal(s['displacement'],displacement)]
                if len(support) != V:
                    raise ArithmeticError('Actual heavy ordered path support changed')
                paths.extend({**s,'scalars':coefficients,'hop':s['seam']*np.einsum('m,mij->ij',coefficients,native['frame'])} for s in support)
            profile = dict(model=p['model'], load=1, label=p['role'], onsite=float(p['onsite'])*native['frame'][0], paths=paths)
        h = sign*supplier.quantum.matter_hamiltonian(native,profile,zero)
        F = sign*supplier.pauli.operator(native,profile,zero,mode)
        energy, U = np.linalg.eigh(h)
        if not energy[127] < 0 < energy[128]:
            raise ArithmeticError('Both physical negative bands must be occupied')
        W, Z = U[:,:128], U[:,128:]; delta = energy[None,128:]-energy[:128,None]
        amplitude = W.conj().T@F@Z; weights = abs(amplitude)**2
        currents = [native['a']*supplier.quantum.path_derivative(native,profile,zero,d) for d in directions] if sign == 1 else [np.zeros_like(h) for _ in directions]
        others = currents+[rho_uniform,*rho]
        mixed_amplitudes = np.stack([W.conj().T@operator@Z for operator in others])
        bubbles = []
        for om in (0,1,10,100):
            mixed = np.einsum('ij,aij,ij->a',amplitude,mixed_amplitudes.conj(),1/(delta-1j*om)) + np.einsum('ij,aij,ij->a',amplitude.conj(),mixed_amplitudes,1/(delta+1j*om))
            bubbles.append(dict(unscaled_Euclidean_frequency=om, Phi_to_ALL3_Aholonomy_real=np.real(mixed[:3]).tolist(), Phi_to_ALL3_Aholonomy_imag=np.imag(mixed[:3]).tolist(), Phi_to_uniform_rho_abs=float(abs(mixed[3])), Phi_to_ALL63_rho_abs_max=float(np.max(abs(mixed[4:])))))
        contact = []
        if sign == 1:
            for d in directions:
                _,_,DF,_ = supplier.control.ward.vertices(native,profile,zero,mode,d)
                contact.append(native['a']*np.trace(W.conj().T@DF@W))
        else:
            contact = [0j]*3
        actual_h = Ufourier.conj().T@h@Ufourier; actual_F = Ufourier.conj().T@F@Ufourier
        expected_h = np.zeros_like(h); expected_F = np.zeros_like(F)
        energies = []
        for j, row in enumerate(pts):
            sl = slice(4*j,4*j+4)
            expected_h[sl,sl] = sign*sum(float(endpoint(c,0))*gamma[i] for i,c in enumerate(row['h']))
            expected_F[sl,sl] = sign*float(endpoint(row['xi0'],0))*mode['delta']/8
            E = float(endpoint(row['E'],0)); energies.extend([-E,-E,E,E])
        errors = dict(full256_h_Bloch=float(np.max(abs(actual_h-expected_h))), full256_F_literal_Bloch=float(np.max(abs(actual_F-expected_F))), both_negative_bands_eigh=float(np.max(abs(energy-np.sort(energies)))))
        observed = dict(weight=float(weights.sum()), bubble=float(np.sum(2*weights/delta)), first=float(np.sum(weights*delta)))
        algebraic = dict(weight=sum(r['weight'] for r in pts), bubble=sum(2*r['weight']/r['delta'] for r in pts), first=sum(r['weight']*r['delta'] for r in pts))
        comparisons = {key: abs(observed[key]-float(endpoint(value,0)))/abs(observed[key]) for key,value in algebraic.items()}
        if max(errors.values()) > 2e-10 or max(comparisons.values()) > 2e-11:
            raise ArithmeticError('Independent uniform literal/Bloch PH comparison failed')
        # Literal filled 256-matrix energy finite displacement checks the static
        # fourth derivative, independently of any bosonization/vacuum analogy.
        quadratic = -observed['bubble']/2; base = float(np.sum(energy[:128])); quartics = []
        for t in (.4,.2,.1):
            ep = np.linalg.eigvalsh(h+t*F); em = np.linalg.eigvalsh(h-t*F)
            quartics.append((float(np.sum(ep[:128])+np.sum(em[:128]))/2-base-quadratic*t*t)/t**4)
        fit = float(np.polynomial.polynomial.polyfit(np.array([.4,.2,.1])**2,quartics,2)[0])
        exact_fourth = sum(r['static_filled_sea_quartic_coefficient'] for r in pts)
        fourth_error = abs(fit-float(endpoint(exact_fourth,0)))
        if fourth_error > 3e-5:
            raise ArithmeticError('Literal static filled-sea fourth derivative disagrees')
        rows.append(dict(role=p['role'], load=p['load'], physical_scale='Lambda' if sign == -1 else '1', physical_occupied_bands='negative -Lambda h = upper h BOTH bands' if sign == -1 else 'negative h BOTH bands', PH_shape=[128,128], observed_unscaled=observed, exact_algebraic_unscaled={key:bounds(value) for key,value in algebraic.items()}, relative_agreement= comparisons, literal_Bloch_errors=errors,
                         actual_derived_mixed_bubbles=bubbles, actual_Phi_Aholonomy_contact_real=np.real(contact).tolist(), actual_Phi_Aholonomy_contact_imag=np.imag(contact).tolist(), neutralheavy_EM_law='charge0, NOTheta in physical heavy Hamiltonian' if sign == -1 else 'actual physical Aholonomy derivative a*d/dTheta; all ordered paths',
                         mixed_selection_derivation='At same k, Pminus=(I-h/E)/2 and Pplus=(I+h/E)/2. T35 is Clifford grade3, J_A is grade1, h J_A h is grade1: trace Pminus T35 Pplus J_A=0 by grade traces, separately for both spectral directions. Tr Pminus T35=0 also kills mixed contact. Uniform rho is I and Pplus Pminus=0; nonuniform rho carries momentum q!=0 whereas F carries q0, killing the trace by momentum conservation. These identities are independently OBSERVED in literal ALL3 currents/ALL63 density rows; numerical residuals are not set to zero.',
                         actual_static_fourth_derivative_control=dict(literal_finite_displacement_quartic_fit=fit, algebraic_quartic=bounds(exact_fourth), absolute_difference=fourth_error, scope='Static filled-band operator energy, not full quantum fermion-Phi energy.')))
    return native, rows, geometry


def coupled_quantum(profiles: list[dict[str, Any]], points: list[list[dict[str, Any]]], g0: Fraction) -> dict[str, Any]:
    Lambda = exact(Fraction(2)**8192); G = exact(8*Fraction(2)**-4096)
    kappa = Lambda*exact(g0)*32; A = G*kappa; free_omega = iv.sqrt(A); free_var = G/(2*free_omega)
    modes = []
    for p, pts in zip(profiles,points,strict=True):
        scale = Lambda if p['scale'] == 'heavy' else exact(1)
        for r in pts:
            d = scale*r['delta']; w = p['load']*scale**2*r['weight']
            modes.append(dict(delta=d,d2=d*d,numerator=2*d*w,scale=p['scale']))
    heavy = [m for m in modes if m['scale'] == 'heavy']; charged = [m for m in modes if m['scale'] == 'charged']
    Bh = sum(m['numerator']/m['d2'] for m in heavy); Bc = sum(m['numerator']/m['d2'] for m in charged)
    B = Bh+Bc; effective = kappa-B; epsilon = B/kappa
    if endpoint(effective,0) <= 0:
        raise ArithmeticError('Actual uniform Gaussian adverse instability; integration ABI repair required')
    effective_omega = iv.sqrt(G*effective); static_variance_upper = G/(2*effective_omega)
    heavy_gap2 = min(endpoint(m['d2'],0) for m in heavy); charged_top2 = max(endpoint(m['d2'],1) for m in charged)
    ratio = A/exact(heavy_gap2)
    lower = 1-endpoint(2*Bh/kappa/(1-ratio),1)
    if lower <= 0 or endpoint(ratio,1) >= 1 or charged_top2 >= endpoint(A*exact(lower),0):
        raise ArithmeticError('Uniform separated positive collective pole failed')
    bracket = enclosure(lower,Fraction(1))
    def fixed(z: Any) -> Any:
        return 1-sum(m['numerator']/(m['d2']-A*z) for m in modes)/kappa
    def derivative(z: Any) -> Any:
        return G*sum(m['numerator']/(m['d2']-A*z)**2 for m in modes)
    low_sign = fixed(exact(lower))-exact(lower); top_sign = fixed(exact(1))-1
    contraction = derivative(bracket)
    if endpoint(low_sign,0) <= 0 or endpoint(top_sign,1) >= 0 or endpoint(contraction,1) >= Fraction(1,2):
        raise ArithmeticError('Outward existence/uniqueness pole signs failed')
    z = bracket; iterations = []
    for iteration in range(12):
        image = fixed(z); new = enclosure(max(endpoint(z,0),endpoint(image,0)), min(endpoint(z,1),endpoint(image,1)))
        iterations.append(dict(iteration=iteration+1,width=bounds(new.b-new.a)))
        unchanged = endpoint(new.b-new.a,1) == endpoint(z.b-z.a,1); z = new
        if unchanged:
            break
    omega = iv.sqrt(A*z); residue = G/(1+derivative(z)); fastweight = residue/(2*omega)
    tail_upper = endpoint(static_variance_upper,1)-endpoint(fastweight,0)
    if tail_upper < 0:
        raise ArithmeticError('Uniform positive all-other-poles spectral measure failed')
    tail = exact(tail_upper)
    total_variance = enclosure(max(endpoint(free_var,0),endpoint(fastweight,0)),endpoint(static_variance_upper,1))
    # Exact determinant-integral energy relative to SAME uncoupled PH vacuum.
    # Static Schur replacement is a lower bound. log(1-r)-log(1-r_static)
    # <=G(Bstatic-B(iy))/(y^2+G kappa_eff); integrate each actual pole.
    adiabatic_shift = (effective_omega-free_omega)/2
    energy_error = G/4*sum((m['numerator']/m['d2'])/(m['delta']+effective_omega) for m in modes)
    energy_shift = enclosure(endpoint(adiabatic_shift,0),endpoint(adiabatic_shift+energy_error,1))
    times = []
    for t in (Fraction(0),Fraction(1,4096),Fraction(4096)):
        phase = omega*exact(t); re = fastweight*iv.cos(phase); im = -fastweight*iv.sin(phase)
        real = re+enclosure(-tail_upper,tail_upper); imag = im+enclosure(-tail_upper,tail_upper)
        if t == 0:
            real = total_variance; imag = exact(0)
        times.append(dict(time=str(t),omega_t_outward=bounds(phase),phase_width_price=bounds(phase.b-phase.a),fast_cos=bounds(iv.cos(phase)),fast_minus_sin=bounds(-iv.sin(phase)),fast_raw_real=bounds(re),fast_raw_imag=bounds(im),actual_raw_kernel_real=bounds(real),actual_raw_kernel_imag=bounds(imag),ALL_other_poles_complex_disk_radius_upper=bounds(tail),common_phase='K(t)=<Omega|Phi exp[-it(H-E_Omega)] Phi|Omega> for ONE full coupled neutral Gaussian vacuum',normalization='Actual raw Cart coordinate. No Gamma whitening, charged own phase, calibration or free-oscillator substitution.'))
        free_phase = free_omega*exact(t)
        wrong_real = free_var*iv.cos(free_phase)
        wrong_imag = -free_var*iv.sin(free_phase)
        distance = iv.sqrt((wrong_real-real)**2+(wrong_imag-imag)**2)
        times[-1]['WRONG_standalone_free_Phi_substitution'] = dict(
            raw_real=bounds(wrong_real), raw_imag=bounds(wrong_imag),
            absolute_complex_kernel_difference=bounds(distance),
            difference_relative_to_free_variance=bounds(distance/free_var),
            interpretation='Counterfactual free covariance/frequency only, NOT the physical kernel or a separately chosen physical neutral phase.')
    primitive_quartic = Lambda/exact(64*Fraction(1,64)*64)
    sea_quartic = sum(p['load']*(Lambda if p['scale'] == 'heavy' else exact(1))*sum(r['static_filled_sea_quartic_coefficient'] for r in pts) for p,pts in zip(profiles,points,strict=True))
    qeff = primitive_quartic+sea_quartic
    nonlinear = dict(actual_primitive_quartic_coefficient=bounds(primitive_quartic),actual_static_filled_sea_Pauli_quartic_coefficient=bounds(sea_quartic),actual_static_BO_total_quartic_coefficient=bounds(qeff),actual_coupled_Gaussian_marginal_Phi_fourth_moment=bounds(3*total_variance**2),primitive_quartic_expectation_in_ACTUAL_coupled_Gaussian=bounds(3*primitive_quartic*total_variance**2),static_BO_leading_quartic_expectation=bounds(3*qeff*total_variance**2),scope='Primitive quartic expectation uses actual coupled Gaussian marginal. Sea coefficient is literal static filled-band fourth derivative; its marginal average is only a leading static BO perturbative price, NOT a full interacting nonlinear quantum vacuum/error enclosure or omitted Phi bdag b dynamics.')
    return dict(G=bounds(G),kappa=bounds(kappa),q0_alignment_stiffness=0,heavy_static_bubble=bounds(Bh),all64_charged_static_bubble=bounds(Bc),epsilon_static=bounds(epsilon),positive_kappa_effective=bounds(effective),positive_kappa_effective_divided_Lambda=bounds(effective/Lambda),bare_frequency=bounds(free_omega),free_Phi_variance=bounds(free_var),actual_coupled_ground_Phi_variance=bounds(total_variance),free_packet_relative_variance_wrongness=bounds(total_variance/free_var-1),actual_fullGaussian_vacuum_energy_shift=bounds(energy_shift),actual_static_Schur_energy_shift=bounds(adiabatic_shift),ALL_actual_poles_energy_nonadiabatic_error_upper=bounds(energy_error),initial_z_bracket=bounds(bracket),F_lower_minus_lower=bounds(low_sign),F_one_minus_one=bounds(top_sign),contraction_absolute_derivative=bounds(contraction),intersection_iterations=iterations,zroot_outward=bounds(z),collective_omega=bounds(omega),coordinate_residue_Z=bounds(residue),fast_Phi_spectralweight=bounds(fastweight),ALL_other_poles_spectralweight_upper=bounds(tail),tail_relative_to_free_variance=bounds(tail/free_var),full_times_rawcoordinate_transfer=times,leading_nonlinear_primitive_and_static_Pauli_price=nonlinear,
                actual_uncoupled_free_packet_energy_excess_above_coupled_ground=bounds(-energy_shift),
                free_packet_energy_excess_proof='In the uncoupled Phi Gaussian times all actual PH vacua, the mean linear Phi-PH term is exactly0. Its energy is the uncoupled zero point; excess over the actual coupled Gaussian ground is -Eshift, not merely a covariance mismatch.',
                Hamiltonian='H=G PiPhi^2/2+kappa Phi^2/2+sum Delta bdag b+Phi sum sqrt(w)(b+bdag); all64 charged load and BOTH negative bands, physical heavy Delta=Lambda*delta, w=Lambda^2*w0. PH vacuum zeroed equally before/after coupling.',
                spectral_proof='D00=Gkappa,Djj=Delta^2,D0j=sqrt(2G Delta w); kappa-Bstatic>0 iff D>0. Schur R00=G/[Gkappa-s-Gsum 2Delta w/(Delta^2-s)]. Isolated collective pole between charged and heavy bands; Z=G/(1+Gsum2Delta w/(Delta^2-s)^2). Phi covariance=sum Zj/(2omegaj). Euclidean B(iy)<=Bstatic implies total variance<=G/(2sqrt(Gkappa_eff)); subtract collective weight LOWER to bound EVERY other pole. K(t)=sum positive weightj exp(-iomegaj t).',
                energy_proof='Eshift=(1/(2pi))integral_0^infty log(1-G B(iy)/(y^2+Gkappa))dy. Static replacement lower Ead=(sqrt(Gkappa_eff)-sqrt(Gkappa))/2. Difference<=G/4 sum B_j/(Delta_j+sqrt(Gkappa_eff)); includes all actual PH poles. This is exact Gaussian ground energy enclosure, not second-order frozen-sea expectation.')


def run(path: Path) -> dict[str, Any]:
    started = time.monotonic(); raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != fast.FROZEN_RECEIPT_SHA:
        raise ArithmeticError('Frozen e7a scalar receipt input required')
    if hashlib.sha256(SOURCE_BYTES[Path(fast.__file__).name]).hexdigest() != FAST_SHA or hashlib.sha256(SOURCE_BYTES[Path(supplier.__file__).name]).hexdigest() != fast.FROZEN_SUPPLIER_SHA:
        raise ArithmeticError('Frozen 8916/404d captured source epochs required')
    frozen = json.loads(raw); mean = frozen['input_descriptor']['actual_root_mean_input']; iv.dps = DIGITS
    g0, gs = fast.point_mean(mean); profiles = fast.primitive_profiles(mean)
    points = [uniform_bloch(p,g0,gs) for p in profiles]
    native, observed, geometry = literal_species(mean,profiles,points)
    physical = coupled_quantum(profiles,points,g0)
    desc = dict(plan=frozen['input_descriptor']['plan'],scientific_execution_revision=ASSIGNED_REVISION,observed_current_governing_plan_revision=BOUND_REVISION,template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',input_receipts=dict(frozen_oscillator=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),declared_NEW_POINT_parameters=dict(g0_exact_binary64=str(g0),gs_exact_binary64=str(gs),g0_float=float(g0),gs_float=float(gs),native_frame_squared_trace=dict(g0=native['g0'],gs=native['gs']),interpretation='Exact float(midpoint frozen constructor enclosure), square roots thereafter outward algebraic. NEW scalar Gaussian point, NOT promoted true interacting mean.'),actual_root_mean_input_UNCHANGED=mean,actual_parameters=dict(Lambda='2^8192',chi='2^-4096',a='1/64',UEphys='1',g_Pi_AND_Coul='64',N=4,clock_UNCHANGED='4096'),uniform_mode='f=1/sqrtV (NOT sin0, NOT sqrt(2/V)cos0), deltaPhi0=i Gamma0 Gamma1 Gamma2 f/sqrt8, Cart unit normal',actual_source_vertex='same k F=xi0*T35/sqrtV; trace Pminus F Pplus F=xi0^2*h3^2/(4V E^2), Delta=2E. Heavy physical -Lambda h, both upper h bands occupied; amplitude Lambda, gap Lambda.',exact_AP_trig='k=(2pi n+pi)/4; signed cos1/sin1=sqrt2/2, cos2=0, signed sin2=1. Every kinetic uses coefficients[axis+1].',primitive_binary_point_profiles=[dict(role=p['role'],load=p['load'],onsite=str(p['onsite']),paths=[{**s,'c0':str(s['c0']),'ci':str(s['ci'])} for s in p['paths']]) for p in profiles],scope='NEW uniform scalarPhi35/all64+neutralheavy linear PH Gaussian; frozen otherPhi/SU4/Pi/Coul, NO full mean/vacuum/originalSOURCE promotion',outward_decimal_precision=DIGITS,physical_times=['0','1/4096','4096'])
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_uniform_primitive_Cartnormal_geometry=geometry,actual_uniform_ALLspecies_literal_operator_consumers=observed,actual_uniform_coupled_quantum_backreaction_and_realtime_transfer=physical,exact_algebraic_uniform_PH_points=[dict(role=p['role'],load=p['load'],rows=[dict(k=r['k'],unscaled_delta=bounds(r['delta']),unscaled_spin_PH_weight=bounds(r['weight']),static_filled_sea_quartic_coefficient=bounds(r['static_filled_sea_quartic_coefficient'])) for r in pts]) for p,pts in zip(profiles,points,strict=True)],source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=fast.mp.__version__,interval_digits=iv.dps),elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oscillator',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); result = run(args.oscillator)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source changed during actual uniform quantum execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    physical = result['actual_uniform_coupled_quantum_backreaction_and_realtime_transfer']
    print(json.dumps(dict(input_sha256=result['input_sha256'],positive_kappa_eff_divided_Lambda=physical['positive_kappa_effective_divided_Lambda'],relative_free_packet_wrongness=physical['free_packet_relative_variance_wrongness'],energy_shift=physical['actual_fullGaussian_vacuum_energy_shift'],times=[dict(time=t['time'],real=t['actual_raw_kernel_real'],imag=t['actual_raw_kernel_imag'],phase_width=t['phase_width_price']) for t in physical['full_times_rawcoordinate_transfer']],tail_relative=physical['tail_relative_to_free_variance'],elapsed_seconds=result['elapsed_seconds']),indent=2))
