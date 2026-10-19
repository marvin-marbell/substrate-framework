#!/usr/bin/env python3
"""New finite primitive-action C control; no change to any prepared source.

The numerical controls are finite N6 and a local reduced-chart differential
block. The exact full-box intertwining is algebraic, with fixed Ref4 only.
Lambda=2**8192 coefficients are kept symbolic; no electromagnetic B term.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import expm, null_space
from induced_phi_backreaction import PhiConfig, SELECTED, matrix_unit_basis, models, scalar_paths, su4_left_frame, joint_gauge_descent_control
from induced_phi_ir_charge_symmetry import B, SOURCE_BYTES as C_BYTES, local_charge_unitary, quadratic, determinant

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**C_BYTES, **{name: (ROOT/name).read_bytes() for name in
    ('induced_phi_ir_preferred_source.py', Path(__file__).name)}}
BASIS = matrix_unit_basis()
T = BASIS/np.sqrt(8)
SIGNS = np.array([-1., 1., 1., 1.])


def coordinates(matrices: np.ndarray) -> np.ndarray:
    return np.einsum('...ij,aji->...a', matrices, BASIS).real/4


def matrices(x: np.ndarray) -> np.ndarray:
    return np.einsum('...a,aij->...ij', x, BASIS)


def field_C(phi: np.ndarray) -> np.ndarray:
    return SIGNS[:, None, None]*(B@phi.conj()@B.conj().T)


def alpha(value: np.ndarray) -> np.ndarray:
    return -B@value.conj()@B.conj().T


def orbit(phi: np.ndarray) -> np.ndarray:
    return coordinates(1j*(T[:, None]@phi[None]-phi[None]@T[:, None])).transpose(1, 2, 0).reshape(60, 15)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a-b)/(1+np.linalg.norm(a)+np.linalg.norm(b)))


def representation(u: np.ndarray) -> np.ndarray:
    occupations = [tuple(i for i in range(4) if state & (1 << i)) for state in range(16)]
    result = np.zeros((16, 16), complex)
    for i, occupied in enumerate(occupations):
        for j, incoming in enumerate(occupations):
            if len(occupied) == len(incoming):
                result[i, j] = determinant(u[np.ix_(occupied, incoming)])
    return result


def finite_fields(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(22210608)
    phi = np.sqrt(np.array([.29, .103, .103, .103]))[:, None, None]*SELECTED
    phi = phi + matrices(rng.normal(0, .012, (size**3, 4, 15)))
    angles = rng.normal(0, .16, (size**3, 3, 15))
    links = np.array([[expm(1j*np.einsum('a,aij->ij', x, T)) for x in site] for site in angles])
    electric = rng.normal(0, .21, (size**3, 3, 15))
    pphi = rng.normal(0, .17, (size**3, 4, 15))
    theta = rng.normal(0, .19, (size**3, 3))
    return phi, links, electric, pphi, theta


def lattice_controls(size: int, A: np.ndarray, R: np.ndarray) -> dict[str, Any]:
    phi, links, electric, pphi, theta = finite_fields(size)
    pc = np.array([field_C(x) for x in phi])
    uc = B@links.conj()@B.conj().T
    ec = electric@A.T
    ppc = (pphi.reshape(-1, 60)@R.T).reshape(pphi.shape)
    sites = list(itertools.product(range(size), repeat=3))
    def idx(site):
        x = np.asarray(site) % size
        return int(x[0]*size**2+x[1]*size+x[2])
    def potentials(p, u):
        gram = np.einsum('xmij,xnji->xmn', p, p).real/4
        quartic = float(np.sum(gram**2))
        alignment = 0.
        curl = 0.
        for x in sites:
            ix = idx(x)
            for a in range(3):
                step = np.eye(3, dtype=int)[a]
                iy = idx(np.asarray(x)+step)
                diff = p[ix]-u[ix, a]@p[iy]@u[ix, a].conj().T
                alignment += float(np.sum(coordinates(diff)**2))
                for b in range(a+1, 3):
                    iz = idx(np.asarray(x)+np.eye(3, dtype=int)[b])
                    plaquette = u[ix, a]@u[iy, b]@u[iz, a].conj().T@u[ix, b].conj().T
                    curl += float(4-np.trace(plaquette).real)
        return dict(quarticGram=quartic, alignment=alignment, colour_curl=curl)
    old = potentials(phi, links)
    new = potentials(pc, uc)
    errors = {key: abs(old[key]-new[key])/(1+abs(old[key])) for key in old}
    errors['Cart_momentum_norm'] = relative(pphi, (ppc.reshape(-1, 60)@R).reshape(pphi.shape))
    errors['colour_electric_norm'] = abs(float(np.sum(electric**2)-np.sum(ec**2)))/(1+float(np.sum(electric**2)))
    gauss_error = 0.
    wrong_flux = 0.
    flux_norm = 0.
    matter_error = 0.
    scalar_controls = []
    forwards = 0
    for _, model, role in models(PhiConfig(Lambda=1.)):
        onsite, paths = scalar_paths(model)
        charge = 0 if role == 'neutral heavy -Lambda' else 1
        path_error = 0.
        if not np.isreal(onsite):
            raise ArithmeticError('Nonreal actual onsite scalar')
        for _, s in paths:
            if s[0].imag != 0 or np.any(s[1:].real != 0):
                raise ArithmeticError('Actual scalar path violates C parity')
        for x in sites:
            ix = idx(x)
            h0 = float(onsite)*phi[ix, 0]
            matter_error = max(matter_error, relative(float(onsite)*pc[ix, 0], -B@h0.conj()@B.conj().T))
            for d, s in paths:
                axis = int(np.flatnonzero(d)[0])
                length = int(d[axis])
                transporter = np.eye(4, dtype=complex)
                converted = np.eye(4, dtype=complex)
                for k in range(length):
                    iy = idx(np.asarray(x)+k*np.eye(3, dtype=int)[axis])
                    transporter = transporter@links[iy, axis]*np.exp(1j*charge*theta[iy, axis])
                    converted = converted@uc[iy, axis]*np.exp(-1j*charge*theta[iy, axis])
                iy = idx(np.asarray(x)+d)
                fx = np.einsum('m,mij->ij', s, phi[ix])
                fy = np.einsum('m,mij->ij', s, phi[iy])
                cx = np.einsum('m,mij->ij', s, pc[ix])
                cy = np.einsum('m,mij->ij', s, pc[iy])
                forward = (fx@transporter+transporter@fy)/2
                cforward = (cx@converted+converted@cy)/2
                path_error = max(path_error, relative(cforward, -B@forward.conj()@B.conj().T))
                forwards += 1
        scalar_controls.append(dict(role=role, EM_charge=charge, multiplicity=63 if role.startswith('auxiliary') else 1,
            actual_onsite=float(onsite), path_count=len(paths), path_lengths=[int(np.sum(abs(d))) for d, _ in paths], error=path_error))
        matter_error = max(matter_error, path_error)
    for x in sites:
        ix = idx(x)
        div = np.sum(electric[ix], axis=0)
        divc = np.sum(ec[ix], axis=0)
        for axis in range(3):
            iy = idx(np.asarray(x)-np.eye(3, dtype=int)[axis])
            e = np.einsum('a,aij->ij', electric[iy, axis], T)
            ce = np.einsum('a,aij->ij', ec[iy, axis], T)
            div -= 2*np.einsum('aij,ji->a', T, links[iy, axis].conj().T@e@links[iy, axis]).real
            divc -= 2*np.einsum('aij,ji->a', T, uc[iy, axis].conj().T@ce@uc[iy, axis]).real
        orbital = orbit(phi[ix]).T@pphi[ix].reshape(60)
        corbital = orbit(pc[ix]).T@ppc[ix].reshape(60)
        gauss_error = max(gauss_error, relative(corbital+divc, A@(orbital+div)))
        wrong_flux = max(wrong_flux, float(np.linalg.norm(corbital+divc-A@orbital-div)))
        flux_norm = max(flux_norm, float(np.linalg.norm(div)))
    commutator = np.einsum('a,aij->ij', electric[0, 0], T)@np.einsum('a,aij->ij', electric[0, 1], T)
    commutator -= np.einsum('a,aij->ij', electric[0, 1], T)@np.einsum('a,aij->ij', electric[0, 0], T)
    return dict(N=size, V=size**3, term_values_before=old, term_values_after=new, errors=errors,
        actual_scalar_paths=scalar_controls, actual_forward_blocks=forwards, matter_intertwining_error=matter_error,
        orbital_plus_link_Gauss_error=gauss_error, wrong_untransformed_flux_defect=wrong_flux,
        nonzero_link_flux_norm=flux_norm, noncommuting_electric_commutator_norm=float(np.linalg.norm(commutator)),
        scope='Generic full Cart15 four-Phi and noncommuting SU4 links/electric at every N6 site; all64 charged roles and neutral heavy paths. No macro mean or state replacement.')


def wave_controls(C: np.ndarray, a: np.ndarray, A: np.ndarray, R: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(8192)
    k = rng.normal(0, .025, (60, 60))
    precision = np.eye(60)+k.T@k
    centre = rng.normal(0, .06, 60)
    x = rng.normal(0, .1, 60)
    dx = x-centre
    log_normalizer = np.linalg.slogdet(precision)[1]/4-15*np.log(np.pi)
    wave = float(np.exp(log_normalizer-dx@precision@dx/2))
    grad = -precision@dx
    hess_ratio = np.outer(grad, grad)-precision
    lap_action = -.5*np.trace(hess_ratio)*wave
    transformed_precision = R@precision@R.T
    cdx = R@dx
    cgrad = -transformed_precision@cdx
    caction = -.5*(np.dot(cgrad, cgrad)-np.trace(transformed_precision))*wave
    step = .001
    def sample(v):
        z = v-centre
        return np.exp(log_normalizer-z@precision@z/2)
    def numeric_lap(h):
        return sum((sample(x+h*np.eye(60)[j])-2*wave+sample(x-h*np.eye(60)[j]))/h**2 for j in range(60))
    numerical_action = -.5*(4*numeric_lap(step/2)-numeric_lap(step))/3
    u = expm(1j*np.einsum('a,aij->ij', np.linspace(-.31, .27, 15), T))
    uc = B@u.conj()@B.conj().T
    def character(v):
        return float(np.sqrt(2)*np.trace(v).real)
    psi = character(u)
    def group_casimir(v, h):
        return -sum((character(expm(1j*h*t)@v)-2*character(v)+character(expm(-1j*h*t)@v))/h**2 for t in T)
    native = (4*group_casimir(u, step/2)-group_casimir(u, step))/3
    native_c = (4*group_casimir(uc, step/2)-group_casimir(uc, step))/3
    gen_casimir = np.sum(T@T, axis=0)
    gauge_error = relative(C@representation(u)@C.conj().T, representation(uc))
    ref = np.eye(16, dtype=complex)[:, 15]
    num = sum((ai.conj().T@ai for ai in a), np.zeros((16, 16), complex))
    parity = np.diag([(-1)**s.bit_count() for s in range(16)])
    return dict(CartGaussian=dict(dimension=60, normalized_L2_squared=1., normalization_proof='N=det(K)^(1/4)/pi^15; integral exp(-(x-c)^T K(x-c))=pi^30/sqrt(det K).',
            actual_wave_value=wave, actual_Laplacian_action_per_chi=float(lap_action), pullback_action_per_chi=float(caction),
            pullback_error=abs(float(lap_action-caction)), independent_curve_action_error=abs(float(numerical_action-lap_action)),
            independent_curve_action_ratio_error=abs(float((numerical_action-lap_action)/wave)),
            actual_Laplacian_action_ratio_per_chi=float(lap_action/wave),
            anisotropic_precision_eigenvalues=np.linalg.eigvalsh(precision).tolist()),
        colour_character=dict(wave='sqrt2 ReTr(U)=(chi_f+chi_fbar)/sqrt2', normalized_Haar_L2_squared=1.,
            normalization_proof='Normalized SU4 Haar character orthogonality: <chi_f,chi_f>=<chi_fbar,chi_fbar>=1 and <chi_f,chi_fbar>=0.',
            actual_wave_value=psi, actual_left_curve_E4_squared_action=native, transformed_action=native_c,
            exact_eigenvalue='15/8', eigenvalue_action_error=abs(native-15*psi/8), C_curve_error=abs(native_c-native),
            generator_Casimir_error=relative(gen_casimir, 15*np.eye(4)/8), energy_per_chi='(3/10)*(15/8)=9/16',
            no_halfcolour=True),
        matter_gauge_intertwining_error=gauge_error, Ref4_wedge_singlet_error=float(np.linalg.norm(representation(u)@ref-ref)),
        Ref4_colour_charge_norm=max(float(np.linalg.norm(quadratic(a, t)@ref)) for t in T),
        Ref_bare_charge=1, Ref_occupied=4, Ref_background=-4, Ref_NET_charge=0,
        Ref_number_restriction_error=float(np.linalg.norm((num-4*np.eye(16))@ref)),
        mobile_halfnumber_transform_error=relative(C@(num-2*np.eye(16))@C.conj().T, -(num-2*np.eye(16))),
        actual_C_square_error=relative(C@C, np.eye(16)), actual_C_even_error=relative(C@parity, parity@C))


def em_control(size: int) -> dict[str, Any]:
    volume = size**3
    # All64 integer occupations have exactly 2V electrons per flavour.
    occupations = np.full((64, volume), 2, dtype=int)
    for species in range(64):
        i = species % volume
        j = (i+1+species//volume) % volume
        occupations[species, i] += 1
        occupations[species, j] -= 1
    ref = np.zeros(volume, dtype=int)
    ref[0] = 4
    background = np.full(volume, 128, dtype=int)
    background[0] += 4
    rho = occupations.sum(axis=0)+ref-background
    crho = (4-occupations).sum(axis=0)+ref-background
    shape = (size,)*3
    q = 2*np.pi*np.arange(size)/size
    lap = np.zeros(shape)
    for j in range(3):
        lap += 4*np.sin(q.reshape(tuple(size if a == j else 1 for a in range(3)))/2)**2
    rhs = np.fft.fftn(rho.reshape(shape))
    solution = np.zeros(shape, complex)
    positive = lap > 1e-12
    solution[positive] = rhs[positive]/lap[positive]
    potential = np.real(np.fft.ifftn(solution))
    electric = np.array([potential-np.roll(potential, -1, axis=j) for j in range(3)])
    divergence = sum(electric[j]-np.roll(electric[j], 1, axis=j) for j in range(3))
    coulomb = float(np.sum(rho.reshape(shape)*potential))
    energy = float(np.sum(electric**2))
    residual = divergence-rho.reshape(shape)
    # Off-shell real momenta demonstrate Gauss transforms, not only zero=zero.
    offshell = electric+np.sin(np.arange(3*volume).reshape((3,)+shape)*.137)*.021
    offdiv = sum(offshell[j]-np.roll(offshell[j], 1, axis=j) for j in range(3))
    dimension = 2*volume+1
    widths = 1+.13*np.sin(np.arange(dimension)*.271)**2
    centre = .03*np.sin(np.arange(dimension)*.173)
    point = .02*np.cos(np.arange(dimension)*.137)
    dx = point-centre
    p2_ratio = float(np.sum(widths)-np.sum((widths*dx)**2))
    cp2_ratio = float(np.sum(widths)-np.sum((widths*(-point+centre))**2))
    def real_curve_action(h):
        forward = np.exp(-h*widths*dx-h*h*widths/2)
        backward = np.exp(h*widths*dx-h*h*widths/2)
        return float(-np.sum(forward-2+backward)/h**2)
    curve_action = (4*real_curve_action(.0005)-real_curve_action(.001))/3
    return dict(integer_occupations_per_flavour=occupations.sum(axis=1).tolist(), rho_sum=int(rho.sum()), rho_nonzero_sites=int(np.count_nonzero(rho)),
        background='128+4 delta_x0', EM_charge_heavy=0, Ref_bare_charge=1, Ref_occupied=4, Ref_background=-4,
        charge_C_error=float(np.max(abs(crho+rho))), real_gradient_Gauss_error=float(np.max(abs(residual))),
        real_electric_energy_per_UE_over_a=energy, Coulomb_rho_Lplus_rho=coulomb, energy_identity_error=abs(energy-coulomb),
        Pi_C='minus real Pi', offshell_Gauss_C_error=float(np.max(abs((-offdiv-crho.reshape(shape))+(offdiv-rho.reshape(shape))))),
        offshell_Gauss_nonzero_norm=float(np.linalg.norm(offdiv-rho.reshape(shape))),
        noninteger_real_electric_distance=float(np.linalg.norm(electric-np.rint(electric))),
        exact_positive_Laplacian_gap=float(4*np.sin(np.pi/size)**2),
        real_transverse_Gaussian=dict(dimension=dimension, normalized_L2_squared=1.,
            normalization_proof='Product(k_j/pi)^(1/4) exp(-sum k_j(theta_j-c_j)^2/2) in real orthonormal coexact+harmonic coordinates. Gauss dressing is an isometry.',
            log_wave_normalization=float(np.sum(np.log(widths)-np.log(np.pi))/4),
            Pi_squared_action_ratio_per_UE_over_a=p2_ratio, transformed_action_ratio_per_UE_over_a=cp2_ratio,
            actual_real_curve_action_error=abs(curve_action-p2_ratio), C_action_error=abs(cp2_ratio-p2_ratio),
            full_electric_plus_Coulomb_action_ratio_per_UE_over_a=p2_ratio+coulomb),
        domain='L2(R^(2V+1)), real link theta and Pi after additive real site gauge quotient; no compact integer link substitution; no bare EM B.')


def connected_cover_control(size: int, frame: np.ndarray) -> dict[str, Any]:
    _, links, _, _, _ = finite_fields(size)
    sites = list(itertools.product(range(size), repeat=3))
    centres = (1, 1j, -1, -1j)
    compatible = []
    # Propagating the root centre along this actual connected periodic graph
    # solves z_x U_xy z_y^dag=U_xy; invertibility forces z_y=z_x.
    for root_centre in centres:
        values = {sites[0]: root_centre}
        pending = [sites[0]]
        while pending:
            x = pending.pop()
            for axis in range(3):
                y = list(x)
                y[axis] = (y[axis]+1) % size
                target = tuple(y)
                if target not in values:
                    values[target] = values[x]
                    pending.append(target)
        error = 0.
        for ix, x in enumerate(sites):
            for axis in range(3):
                y = list(x)
                y[axis] = (y[axis]+1) % size
                error = max(error, relative(values[x]*links[ix, axis]*np.conj(values[tuple(y)]), links[ix, axis]))
        if len(values) != size**3 or error > 1e-12:
            raise ArithmeticError('Actual connected centre cover failed')
        compatible.append(dict(root=[root_centre.real, root_centre.imag], assigned_sites=len(values), link_error=error))
    native_wave = joint_gauge_descent_control(frame)
    return dict(N=size, checked_edges=3*size**3, residual_degree=len(compatible), root_assignments=compatible,
        actual_normalized_equivariant_groupwave=native_wave,
        global_norm_squared=native_wave['finite_cover_quotient_norm_squared'],
        wrong_local_centre_norm_squared=native_wave['wrong_independent_local_centre_cover_norm_squared'])


def chart_controls(A: np.ndarray, R: np.ndarray) -> dict[str, Any]:
    frame = np.sqrt(np.array([.29, .103, .103, .103]))[:, None, None]*SELECTED
    o0 = orbit(frame)
    normal = null_space(o0.T)
    cframe = field_C(frame)
    co0 = orbit(cframe)
    cnormal = R@normal
    rng = np.random.default_rng(60)
    z = rng.normal(0, .007, 90)
    S = np.zeros((90, 90))
    S[:45, :45] = np.eye(45)
    for j in range(3):
        S[45+15*j:60+15*j, 45+15*j:60+15*j] = A
    def log_weight(v, base, base_orbit, n):
        phi = base+matrices((n@v[:45]).reshape(4, 15))
        fp = base_orbit.T@orbit(phi)
        sign, ld = np.linalg.slogdet(fp)
        if sign <= 0:
            raise ArithmeticError('Nonpositive actual FP determinant')
        result = ld-np.linalg.slogdet(base_orbit.T@base_orbit)[1]/2
        for theta in v[45:].reshape(3, 15):
            eig = np.linalg.eigvalsh(np.einsum('a,aij->ij', theta, T))
            differences = [eig[b]-eig[a] for a in range(4) for b in range(a+1, 4)]
            result += 2*sum(np.log(np.sinc(d/(2*np.pi))) for d in differences)
        return float(result)
    def geometry(v, base, base_orbit, n):
        phi = base+matrices((n@v[:45]).reshape(4, 15))
        o = orbit(phi)
        fp = base_orbit.T@o
        logw = log_weight(v, base, base_orbit, n)
        jinv = []
        for theta in v[45:].reshape(3, 15):
            X = np.einsum('a,aij->ij', theta, T)
            comm = 1j*(X@T-T@X)
            ad = 2*np.einsum('aij,bji->ab', T, comm).real
            aug = np.zeros((30, 30))
            aug[:15, :15] = ad
            aug[:15, 15:] = np.eye(15)
            jinv.append(np.linalg.inv(expm(aug)[:15, 15:].T))
        # Exact local principal block: all incoming neighbour momenta fixed0.
        # Three outgoing electric generators enter this site's Gauss square.
        E = np.zeros((45, 90))
        for j in range(3):
            E[j*15:(j+1)*15, 45+j*15:60+j*15] = jinv[j]
        div = np.concatenate([np.eye(15)]*3, axis=1)@E
        rhs = -div
        rhs[:, :45] -= o.T@n
        angular = base_orbit@np.linalg.solve(fp.T, rhs)
        K = angular.T@angular+(3/5)*(E.T@E)
        K[:45, :45] += np.eye(45)
        return float(logw), K
    lw, K = geometry(z, frame, o0, normal)
    clw, cK = geometry(S@z, cframe, co0, cnormal)
    theta = z[45:60]
    literal = su4_left_frame(theta)
    c_literal = su4_left_frame(A@theta)
    fp = o0.T@orbit(frame+matrices((normal@z[:45]).reshape(4, 15)))
    cfp = co0.T@orbit(cframe+matrices((cnormal@z[:45]).reshape(4, 15)))
    # Differential action on a genuine local normalized Gaussian, conjugated
    # by full local FP and all three Haar Jacobians; derivatives are actual.
    # C coordinates use the pulled-back stencil S, not a redefined metric.
    h = 2e-4
    directions = np.eye(90)
    def derivatives(point, base, oo, n, dirs):
        value, metric = geometry(point, base, oo, n)
        logs = []
        metrics = []
        for d in dirs:
            lp, kp = geometry(point+h*d, base, oo, n)
            lm, km = geometry(point-h*d, base, oo, n)
            logs.append((lp, lm))
            metrics.append((kp, km))
        grad = np.array([(lp-lm)/(2*h) for lp, lm in logs])
        hess = np.zeros((90, 90))
        for i in range(90):
            hess[i, i] = (logs[i][0]-2*value+logs[i][1])/h**2
            for j in range(i):
                terms = [log_weight(point+h*(si*dirs[i]+sj*dirs[j]), base, oo, n) for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
                hess[i, j] = hess[j, i] = (terms[0]-terms[1]-terms[2]+terms[3])/(4*h*h)
        pulled = dirs@metric@dirs.T
        div = sum((dirs@kp@dirs.T)[i]-(dirs@km@dirs.T)[i] for i, (kp, km) in enumerate(metrics))/(2*h)
        geometric = (div@grad+np.sum(pulled*hess))/4+grad@pulled@grad/8
        # f Gaussian in flat dz, Psi=w^-1/2 f. All quantities /chi.
        gaussian_grad = -point@dirs.T
        gaussian_hess = np.outer(gaussian_grad, gaussian_grad)-np.eye(90)
        action = -.5*(np.sum(pulled*gaussian_hess)+div@gaussian_grad)+geometric
        return dict(logw=value, gradient_norm=float(np.linalg.norm(grad)), scalar_Jacobian_action_per_chi=float(geometric),
            conjugated_Gaussian_action_ratio_per_chi=float(action), kinetic_min_eigenvalue=float(np.linalg.eigvalsh(pulled)[0]))
    old_action = derivatives(z, frame, o0, normal, directions)
    new_action = derivatives(S@z, cframe, co0, cnormal, S.T)
    # Genuine group-chart normalization is independent of constants under C.
    # Connected N6 graph residual can be counted via an actual spanning tree.
    centres = (1, 1j, -1, -1j)
    compatible = sum(abs(zx*zy.conjugate()-1)<1e-14 for zx in centres for zy in centres)
    return dict(orbit_covariance_error=relative(co0, R@o0@A.T), FP_conjugation_error=relative(cfp, A@fp@A.T),
        FP_determinant=float(np.linalg.det(fp)), transformed_FP_determinant=float(np.linalg.det(cfp)),
        Haar_exponential_Jacobian=literal['Haar_density'], transformed_Haar_Jacobian=c_literal['Haar_density'],
        Haar_Maurer_covariance_error=relative(c_literal['J_left'], A@literal['J_left']@A.T),
        Haar_Frechet_curve_error=literal['Frechet_control_error'], Haar_nonidentity_frame_norm=literal['nonidentity_frame_norm'],
        normalized_full_local_logw=lw, transformed_logw=clw, logw_C_error=abs(lw-clw),
        transformed_principal_metric_error=relative(cK, S@K@S.T), metric_real_orthogonal_error=relative(S@S.T, np.eye(90)),
        differential_action_before=old_action, differential_action_after=new_action,
        Jacobian_scalar_action_error=abs(old_action['scalar_Jacobian_action_per_chi']-new_action['scalar_Jacobian_action_per_chi']),
        Gaussian_action_error=abs(old_action['conjugated_Gaussian_action_ratio_per_chi']-new_action['conjugated_Gaussian_action_ratio_per_chi']),
        finite_difference_step=h, differential_scope='Actual 90-coordinate local this-site Gauss-square block with incoming neighbour momenta zero; neighbours also contribute to the full metric. All45 normal plus3*15 link coordinates, full FP and3 Haar. A Gaussian differential germ, not a globally valid chart Gaussian or the full19440-coordinate N6 manybody wave.',
        degree4_connected_control=connected_cover_control(6, frame),
        degree4_edge_compatible_pairs=compatible, degree4_norm_squared=compatible/4,
        wrong_local_centre_norm_squared=compatible/16,
        global_normalization_proof='Connected edges force zx=zy; spanning-tree induction leaves exactly4 simultaneous centres at any V. w_joint=Vol4^(V-L) prod detFP/sqrt(detR0) prod HaarJac/4. Equivariant lift w_joint^-1/2 times any normalized chart-supported f has norm1 by degree4 change of variables and normalized Haar. C preserves every determinant, Haar and Vol4; constants disappear from log derivatives. Never4^V.')


def run(c_receipt_path: str, preferred_receipt_path: str) -> dict[str, Any]:
    started = time.monotonic()
    inputs = {}
    for role, path in (('actual_C', c_receipt_path), ('preferred_source', preferred_receipt_path)):
        raw = Path(path).read_bytes()
        capsule = json.loads(raw)
        inputs[role] = dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(), input_sha256=capsule['input_sha256'])
        for name, expected in capsule['source_at_import'].items():
            digest = expected['sha256'] if isinstance(expected, dict) else expected
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise ArithmeticError('Input receipt does not match actual final source '+name)
    C, a, ccontrol = local_charge_unitary()
    A = 2*np.einsum('aij,bji->ab', T, alpha(T)).real
    tau = coordinates(B@BASIS.conj()@B.conj().T).T
    R = np.kron(np.diag(SIGNS), tau)
    linear = dict(generator_Gram_error=relative(2*np.einsum('aij,bji->ab', T, T).real, np.eye(15)),
        outer_real_orthogonal_error=relative(A@A.T, np.eye(15)), Cart60_real_orthogonal_error=relative(R@R.T, np.eye(60)),
        outer_square_error=relative(A@A, np.eye(15)), Cart_square_error=relative(R@R, np.eye(60)))
    lattice = lattice_controls(6, A, R)
    waves = wave_controls(C, a, A, R)
    em = em_control(6)
    chart = chart_controls(A, R)
    checks = {**linear, **lattice['errors'], 'matter': lattice['matter_intertwining_error'],
        'Gauss': lattice['orbital_plus_link_Gauss_error'], 'group': waves['colour_character']['eigenvalue_action_error'],
        'group_C': waves['colour_character']['C_curve_error'], 'Cart_wave': waves['CartGaussian']['independent_curve_action_ratio_error'],
        'matter_gauge': waves['matter_gauge_intertwining_error'], 'Ref_singlet': waves['Ref4_wedge_singlet_error'],
        'C_square': waves['actual_C_square_error'], 'C_even': waves['actual_C_even_error'],
        'real_EM_Gauss': em['real_gradient_Gauss_error'], 'C_rho': em['charge_C_error'], 'C_EM_Gauss': em['offshell_Gauss_C_error'],
        'Coulomb': em['energy_identity_error'], 'FP': chart['FP_conjugation_error'], 'Haar': chart['Haar_Maurer_covariance_error'],
        'metric': chart['transformed_principal_metric_error'], 'Jacobian_action': chart['Jacobian_scalar_action_error'],
        'full_local_action': chart['Gaussian_action_error']}
    checks.update(EM_wave=em['real_transverse_Gaussian']['actual_real_curve_action_error'],
        EM_wave_C=em['real_transverse_Gaussian']['C_action_error'],
        group_Casimir=waves['colour_character']['generator_Casimir_error'],
        mobile_halfcounts=waves['mobile_halfnumber_transform_error'], Ref_colour=waves['Ref4_colour_charge_norm'],
        Ref_number=waves['Ref_number_restriction_error'])
    numerical_errors = {'group': 1e-6, 'group_C': 1e-6, 'Cart_wave': 1e-6, 'EM_wave': 1e-5,
        'Jacobian_action': 2e-5, 'full_local_action': 2e-5}
    if any(value > numerical_errors.get(key, 1e-10) for key, value in checks.items()):
        raise ArithmeticError('Actual full-action control failed: '+str(checks))
    if min(lattice['wrong_untransformed_flux_defect'], lattice['noncommuting_electric_commutator_norm'], em['noninteger_real_electric_distance']) <= .001:
        raise ArithmeticError('Nontrivial adverse action controls failed')
    if abs(chart['degree4_norm_squared']-1) > 1e-12 or chart['differential_action_before']['kinetic_min_eigenvalue'] < .599:
        raise ArithmeticError('Canonical colour coefficient or global degree4 failed')
    total = 65*2*262144**3+4
    descriptor = dict(N_control=6, N_macro=262144, a='1/64', Lambda='2^8192', chi='2^-4096',
        quantum_inputs='hbar=1; CAR, Cart CCR, SU4 left generators, noncompact EM CCR declared inputs',
        coefficients=dict(quartic='Lambda/a ||G||F^2', alignment='Lambda^(5/4)/10 ||Cart covariant difference||^2',
            curl='7Lambda^(5/4)/10 (4-ReTr plaquette)', Cart_kinetic='chi/2 PiCart^2',
            colour_kinetic='3chi/10 E4^2', colour_cometric='3chi/5', EM_electric='UEphys/a PiEM^2', bare_EM_B=None),
        inputs=inputs, physical_scalar_config={key: value for key, value in vars(PhiConfig(Lambda=1.)).items() if key != 'Lambda'},
        plan='issue-222-induced-photon-delta13-continuation-20261008', work_item='joined-explanatory-campaign',
        source='NEW independent full primitive action controls; frozen existing sources unchanged')
    proof = dict(matter='C16 a C16dag=B a†. Traceless on-site term has zero CAR constant; literal real onsite/mass and imaginary spatial paths give transformed one-body h_C=-B h* Bdag with endpoints exchanged by CAR. Ordered arbitrary noncommuting SU4 and real EM products conjugate termwise. Neutral heavy has charge0 and retains -Lambda weight; all64 charged retain original weights.',
        bosons='Gram transforms D G D; Frobenius quartic invariant. Covariant alignment transforms by signed real isometry; each colour plaquette maps B P* Bdag and retains ReTr. Cart pullback is real orthogonal60 and preserves Laplacian. SU4 alpha is a Haar-preserving involutive group automorphism; left E maps by real orthogonal adjoint alpha, hence exact .3chi Casimir invariant. No added EM magnetic term.',
        Gauss='O(CPhi)=R60 O(Phi) alpha^T, pPhi->R60 pPhi, Eleft->alpha Eleft. Matter generator C16 q(T) C16dag=q(alpha(T)); outgoing/incoming transported flux and orbital charge ALL transform by same alpha. Therefore Gauss constraint and its physical invariant subspace intertwine.',
        EM_Ref='Ref factor is identity on fixed wedge4 onsite singlet with BARE Qref1, nref4 and background-4 NET0. Mobile charged rho=sum64(n-2); heavy EM0. C sends rho->-rho and real theta/Pi->minus; divPi-rho->minus. Orthogonal real Poisson solution and rho^T L+rho invariant. Refhole/electron charged preparation is excluded.',
        statistics='Actual C16 is EVEN, real monomial and C16^2=I. Disjoint65V factors commute so global C^2=I exactly; mobile halfcounts invariant globally. Phi/links/EM pullbacks square toI. Ref identity remains squareI. Ntotal=130V+4=0 mod4 on declared even macrobox; diagonal centre descent preserved.',
        joint_Jacobian='Orbit FP matrices conjugate by alpha, Maurer J by alpha; normal slice and reduced cometric transform covariantly. Haar and Cart measure are preserved. Thus weighted Laplace-Beltrami divergence operator AND full scalar generated by w^-1/2 intertwine, including all nonlocal cross blocks by chain rule. Actual local90 differential control includes all FP/Haar derivatives, not a flat replacement. Only diagonalZ4 degree4, global normalization preserved.',
        full_H='Combining these exact domain/term identities gives C H[A] Cdag=H[-A] for the full finite macro Hamiltonian restricted to SAME fixed Ref4 sector. Smooth compact-support Cart/EM times SU4 matrix coefficients form a common invariant core; invariant positive kinetic/quartic forms and real bounded-unitary path couplings extend the intertwining to the specified self-adjoint form realization.',
        limitation='Parent mathematical review required. No full photon vacuum, global ground, odd-vertex theorem for a non-C-invariant density, Cchi3 mixture theorem, external Refhole symmetry, or real-time scattering admission.')
    stability = {name: (ROOT/name).read_bytes() == data for name, data in SOURCE_BYTES.items()}
    if not all(stability.values()):
        raise ArithmeticError('Imported final source changed during actual run')
    return dict(input_descriptor=descriptor, input_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        exact_Hamiltonian_proof=proof, linear_controls=linear, term_controls=lattice, waves_and_statistics=waves,
        real_EM_Coulomb=em, jacobian_action=chart, all_errors=checks, actual_local_C=ccontrol,
        macro_counts=dict(V=262144**3, mobile_species=65, charged_species=64, mobile_occupied=130*262144**3, Ref4_occupied=4, total=total, total_mod4=total%4),
        source_at_import={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name, data in SOURCE_BYTES.items()},
        source_unchanged_since_import=stability,
        runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, platform=platform.platform(), argv=sys.argv),
        elapsed_seconds=time.monotonic()-started, natural_exit_expected=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c-receipt', required=True)
    parser.add_argument('--preferred-receipt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.c_receipt, args.preferred_receipt)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(input_sha256=report['input_sha256'], all_errors=report['all_errors'],
        differential_actions=report['jacobian_action']['differential_action_before'], elapsed_seconds=report['elapsed_seconds']), indent=2), flush=True)
