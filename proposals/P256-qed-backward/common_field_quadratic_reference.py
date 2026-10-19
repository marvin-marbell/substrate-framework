"""Mean-quadratic reference, not a replacement Hamiltonian or prepared ground.

Coordinates are the original mass-horizontal normal coordinates.  The ordinary
held Fock density is integrated from the unchanged two-bank polynomial packet.
No occupied-spectrum response, harmonic field law, or phase reset is introduced.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, cast

import numpy as np
from scipy.linalg import eigh, qr
from scipy.special import gammaincc, gammaln

from local_orbital_stability import elastic_hessian
from unrestricted_field_minimum import ALIGNMENT, STIFFNESS


def seed_packet(model: Any, data: dict[str, Any]) -> dict[str, Any]:
    coefficient = np.asarray(data['preparation_state14'], complex).reshape(2, 7)
    a, b = coefficient[:, 0], coefficient[:, 1:]
    w = model.original.directions @ b.T
    bank = np.abs(a)**2 + np.sum(np.abs(b)**2, axis=1)
    z = float(bank.sum())
    rho = sum(bank[i] * model.fibre.projectors[q] for i, q in enumerate((1, -1))) / z
    sigma = float(model.sigma)
    v = w @ a.conj()
    mean_q = math.sqrt(2)*sigma*v.real/z
    mean_p = math.sqrt(2)*v.imag/(sigma*z)
    return dict(a=a, b=b, w=w, bank_weights=bank/z, original_norm_squared=z,
                rho_light=rho, mean_q=mean_q, mean_p=mean_p,
                sigma=sigma, radius=float(model.radius), n=model.chart.normal_dim,
                boost_Omega_at_actual_cache_restore=float(model.boost['Omega']))


def fixed_seed_potential(model: Any, packet: dict[str, Any], q: np.ndarray) -> dict[str, float]:
    point = model.chart.embedding(q)
    result = model.original.classical_potential(point)
    h = model.law.onebody(point['phi'], point['links'], np.zeros((model.law.volume, 3)))
    result['light_fixed_seed'] = float(np.trace(packet['rho_light'] @ h).real)
    result['reference_onsite'] = 0.
    result['fixed_coulomb'] = float(np.sum(packet['bank_weights'] * cast(Any,np.diag(model.HC['neutral'])).real))
    return result


def _add(matrix: np.ndarray, ids: np.ndarray, block: np.ndarray) -> None:
    matrix[np.ix_(ids, ids)] += block


def _onebody_contact(law: Any, phi: np.ndarray, rho: np.ndarray, factor: float,
                     hessian: np.ndarray, gradient: np.ndarray) -> None:
    """Literal directed endpoint paths, including the improved two-link path.

    The +adjoint and endpoint /2 cancel in the real expectation.  Same-edge
    exponential second derivatives are anticommutators; distinct edges retain
    their original path order.  There are no occupied/empty denominators.
    """
    basis, generators = law.basis, law.generators
    products = np.einsum('aij,bjk->abik', generators, generators)
    sym = (products + products.swapaxes(0, 1))/2
    for x in range(law.volume):
        sx = slice(4*x, 4*x+4)
        gradient[105*x:105*x+15] += factor*law.onsite*np.einsum('ij,aji->a', rho[sx, sx], basis).real
        for displacement, scalar in law.paths:
            axis = int(np.flatnonzero(displacement)[0])
            length = int(displacement[axis])
            if length <= 0:
                raise ValueError('Original positive directed paths required')
            y, edges = x, []
            for _ in range(length):
                edges.append((y, axis))
                y = int(law.target[y, axis])
            c = rho[4*y:4*y+4, sx]
            fx = np.einsum('m,mij->ij', scalar, phi[x])
            fy = np.einsum('m,mij->ij', scalar, phi[y])
            endpoint = fx + fy
            for site in (x, y):
                for m in range(4):
                    gradient[105*site+15*m:105*site+15*(m+1)] += factor*(scalar[m]*np.einsum('ij,aji->a', c, basis)).real
            edge_ids = [np.arange(105*s+60+15*j, 105*s+75+15*j) for s, j in edges]
            for ie, ids in enumerate(edge_ids):
                first = 1j*(np.einsum('ij,ajk->aik', fx, generators)+np.einsum('aij,jk->aik', generators, fy))
                gradient[ids] += factor*np.einsum('ij,aji->a', c, first).real
                for site, left in ((x, True), (y, False)):
                    for m in range(4):
                        pids = np.arange(105*site+15*m, 105*site+15*(m+1))
                        mixed = (np.einsum('aij,bjk->abik', basis, 1j*generators) if left
                                 else np.einsum('bij,ajk->abik', 1j*generators, basis))
                        block = factor*(scalar[m]*np.einsum('ij,abji->ab', c, mixed)).real
                        hessian[np.ix_(pids, ids)] += block
                        hessian[np.ix_(ids, pids)] += block.T
                for je in range(ie, len(edge_ids)):
                    second = -sym if ie == je else -products
                    jet = np.einsum('ij,abjk->abik', fx, second)+np.einsum('abij,jk->abik', second, fy)
                    block = factor*np.einsum('ij,abji->ab', c, jet).real
                    hessian[np.ix_(ids, edge_ids[je])] += block
                    if je != ie:
                        hessian[np.ix_(edge_ids[je], ids)] += block.T


def mean_reference(model: Any, packet: dict[str, Any]) -> dict[str, Any]:
    """Build the all-2430 original-law Hessian, then its physical normal modes."""
    chart, law = model.chart, model.law
    phi, links = chart.mean_phi, chart.mean_links
    flat_defect = float(np.linalg.norm(links-np.eye(4)))
    uniform_defect = float(np.linalg.norm(phi-phi[0]))
    if max(flat_defect, uniform_defect) > 1e-12:
        raise ValueError('This controlled first reference requires the ACTUAL flat uniform mean')
    nr, volume = chart.raw_dim, law.volume
    raw = np.zeros((nr, nr))
    gradient = np.zeros(nr)
    lam = float(model.Lambda)
    pieces: dict[str, Any] = {}
    basis = law.basis
    for x in range(volume):
        ids = np.arange(105*x, 105*x+60)
        _add(raw, ids, lam*elastic_hessian(phi[x], basis, STIFFNESS, 0.))
        coefficients = np.einsum('mij,aji->ma', phi[x], basis).real/4
        gram = np.einsum('mij,nji->mn', phi[x], phi[x]).real/4
        gradient[ids] += (2*lam*STIFFNESS*gram@coefficients).ravel()
    pieces['quartic'] = dict(trace_raw=float(np.trace(raw)), gradient_norm=float(np.linalg.norm(gradient)))
    before = float(np.trace(raw))
    identity_phi = basis.reshape(15, 16).T
    for x in range(volume):
        for j in range(3):
            y = int(law.target[x, j])
            ids = np.r_[np.arange(105*x, 105*x+60), np.arange(105*y, 105*y+60),
                        np.arange(105*x+60+15*j, 105*x+75+15*j)]
            derivative = np.zeros((64, 135), complex)
            for m in range(4):
                derivative[16*m:16*(m+1), 15*m:15*(m+1)] = identity_phi
                derivative[16*m:16*(m+1), 60+15*m:60+15*(m+1)] = -identity_phi
                commutator = -1j*(np.einsum('aij,jk->aik', law.generators, phi[y, m])
                                   -np.einsum('ij,ajk->aik', phi[y, m], law.generators))
                derivative[16*m:16*(m+1), 120:] = commutator.reshape(15, 16).T
            _add(raw, ids, lam**1.25*ALIGNMENT/4*cast(Any,derivative.conj().T@derivative).real)
    pieces['alignment'] = dict(trace_raw=float(np.trace(raw))-before, residual_at_mean=0.)
    before = float(np.trace(raw))
    trace_generators = np.einsum('aij,bji->ab', law.generators, law.generators).real
    for x in range(volume):
        for j in range(3):
            for k in range(j+1, 3):
                y, z = int(law.target[x, j]), int(law.target[x, k])
                edges = ((x, j, 1), (y, k, 1), (z, j, -1), (x, k, -1))
                for s, axis, sign in edges:
                    ids = np.arange(105*s+60+15*axis, 105*s+75+15*axis)
                    for t, other, sign2 in edges:
                        js = np.arange(105*t+60+15*other, 105*t+75+15*other)
                        raw[np.ix_(ids, js)] += lam**1.25*model.K4*sign*sign2*trace_generators
    pieces['colour_plaquette'] = dict(trace_raw=float(np.trace(raw))-before, residual_at_mean=0.)
    for name, operator_law, density, factor in (
            ('heavy_fixed_seed', model.original.heavy_law, model.original.heavy_projector, -lam),
            ('light_fixed_seed', law, packet['rho_light'], 1.)):
        before = float(np.trace(raw))
        old_gradient = gradient.copy()
        _onebody_contact(operator_law, phi, density, factor, raw, gradient)
        pieces[name] = dict(trace_raw=float(np.trace(raw))-before,
                            gradient_norm=float(np.linalg.norm(gradient-old_gradient)), multiplier=factor)
    symmetry = float(np.max(np.abs(raw-raw.T)))
    if symmetry > 1e-6:
        raise AssertionError('Analytic scalar Hessian lost mixed-derivative symmetry')
    # The full J=[B,D] inverse, not an inverse(normal_metric), is used here.
    mean_point = chart.evaluate(np.zeros(chart.normal_dim))
    normal_cometric = mean_point['cometric'][:chart.normal_dim, :chart.normal_dim]
    metric_defects = dict(full_normal_cometric_identity=float(np.linalg.norm(normal_cometric-np.eye(chart.normal_dim))),
                         normal_gauge_metric=float(np.linalg.norm(mean_point['normal_gauge_metric'])),
                         normal_gauge_cometric=float(np.linalg.norm(mean_point['normal_gauge_cometric'])),
                         full_raw_dimension=nr, normal_dimension=chart.normal_dim,
                         colour_mass=float(chart.mass[60]), chi=float(model.chi))
    if metric_defects['full_normal_cometric_identity'] > 1e-9:
        raise ValueError('Mean cometric is not the original mass-horizontal identity: cannot copy chi')
    eigenvalues, modes, spectral = _normal_modes(raw, chart, law)
    normal_gradient = chart.mean_basis.T@gradient
    potential = fixed_seed_potential(model, packet, np.zeros(chart.normal_dim))
    return dict(eigenvalues=eigenvalues, modes=modes, gradient=normal_gradient,
                E0=sum(potential.values()), mean_potential_parts=potential,
                metric_defects=metric_defects, raw_symmetry_defect=symmetry,
                flat_defect=flat_defect, uniform_defect=uniform_defect,
                pieces=pieces, spectral=spectral, chi=float(model.chi), raw_hessian=raw,
                raw_gradient=gradient)


def _normal_modes(raw: np.ndarray, chart: Any, law: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fourier reduction ONLY after actual Hessian translation-covariance check."""
    volume, size, n = law.volume, law.size, chart.normal_dim
    blocks = raw.reshape(volume, 105, volume, 105)
    maximum = 0.
    for x, site in enumerate(law.sites):
        offsets = [law.index[tuple((s-site)%size)] for s in law.sites]
        maximum = max(maximum, float(np.max(np.abs(blocks[x]-blocks[0, :, offsets, :].transpose(1, 0, 2)))))
    scale = max(1., float(np.max(np.abs(raw))))
    if maximum > 1e-11*scale:
        normal = chart.mean_basis.T@raw@chart.mean_basis
        values, modes = eigh(normal)
        return values, modes, dict(method='dense exact normal projection; translation premise failed',
                                  translation_defect=maximum, translation_relative_defect=maximum/scale)
    mass = chart.mass[:105]
    gauge = chart.mean_gauge.reshape(volume, 105, volume, 15)
    basis = chart.mean_basis.reshape(volume, 105, n)
    values_list, modes_list, records = [], [], []
    seen: set[tuple[int, ...]] = set()
    for kt in itertools.product(range(size), repeat=3):
        if kt in seen:
            continue
        opposite = tuple((-k)%size for k in kt)
        seen.update((kt, opposite))
        phase = np.exp(2j*np.pi*(law.sites@np.asarray(kt))/size)
        hk = np.einsum('xaj,x->aj', blocks[0].transpose(1, 0, 2), phase)
        dk = np.einsum('xaj,x->aj', gauge[0].transpose(1, 0, 2), phase)
        q, r = cast(tuple[np.ndarray,np.ndarray],qr(np.sqrt(mass)[:, None]*dk, mode='full'))
        if np.min(np.abs(np.diag(r[:15]))) < 1e-10:
            raise ValueError('Nonregular Fourier gauge block')
        bk = q[:, 15:]/np.sqrt(mass)[:, None]
        projected = bk.conj().T@hk@bk
        defect = float(np.max(np.abs(projected-projected.conj().T)))
        relative_defect=defect/max(1.,float(np.max(np.abs(projected))))
        if relative_defect > 1e-12:
            raise AssertionError('Fourier scalar Hessian Hermitian defect exceeds floating roundoff: '+repr((kt,defect,relative_defect)))
        kvals, vectors = eigh((projected+projected.conj().T)/2)
        transform = np.einsum('xrn,x,r->nr', basis, phase/math.sqrt(volume), mass)
        complex_modes = transform@bk@vectors
        if opposite == kt:
            if np.linalg.norm(complex_modes.imag) > 1e-9:
                # Real QR/eigensolve avoids arbitrary complex phases at the real block.
                q0, _ = cast(tuple[np.ndarray,np.ndarray],qr(np.sqrt(mass)[:, None]*dk.real, mode='full'))
                b0 = q0[:, 15:]/np.sqrt(mass)[:, None]
                kvals, vectors0 = eigh(b0.T@hk.real@b0)
                real_modes = transform.real@b0@vectors0
            else:
                real_modes = complex_modes.real
            values_list.append(kvals); modes_list.append(real_modes)
        else:
            values_list.extend((kvals, kvals))
            modes_list.extend((math.sqrt(2)*complex_modes.real, math.sqrt(2)*complex_modes.imag))
        records.append(dict(momentum=list(kt), multiplicity=1 if opposite == kt else 2,
                            smallest=float(kvals[0]), largest=float(kvals[-1]), Hermitian_defect=defect))
    values, modes = np.concatenate(values_list), np.column_stack(modes_list)
    order = np.argsort(values)
    values, modes = values[order], modes[:, order]
    orthogonality = float(np.linalg.norm(modes.T@modes-np.eye(n)))
    if len(values) != n or orthogonality > 1e-8:
        raise AssertionError('Fourier physical normal modes did not span the whole real space')
    action_defects=[]
    for column in (0,n//2,n-1):
        vector=modes[:,column]
        literal=chart.mean_basis.T@raw@(chart.mean_basis@vector)
        spectral_vector=values[column]*vector
        action_defects.append(float(np.linalg.norm(literal-spectral_vector))/max(1.,abs(values[column])))
    if max(action_defects)>1e-9:
        raise AssertionError('Fourier physical spectrum disagrees with original raw Hessian action')
    return values, modes, dict(method='verified translation-covariant full normal Fourier blocks',
                              translation_defect=maximum, translation_relative_defect=maximum/scale,
                              literal_raw_Hessian_eigen_action_relative_defects=action_defects,
                              normal_mode_orthogonality=orthogonality, blocks=records)


def cutoff_graph_bounds(packet: dict[str, Any]) -> dict[str, float]:
    """Bounds the ORIGINAL compact packet minus its uncut computational core.

    This is not permission to change the initial function.  In particular the
    momentum bound includes eta', unlike a Gram-only cutoff loss estimate.
    """
    n, sigma, radius = packet['n'], packet['sigma'], packet['radius']
    alpha, z0 = n/2, (radius/(2*sigma))**2
    def tail_moment(power: int) -> float:
        return math.exp(power*math.log(sigma*sigma)+float(gammaln(alpha+power)-gammaln(alpha)))*float(gammaincc(alpha+power, z0))
    a = float(np.sum(np.abs(packet['a'])**2))
    l = math.sqrt(2)*packet['w']/sigma
    trace_t = float(np.sum(np.abs(l)**2))
    r0, r1, r2 = (tail_moment(i) for i in range(3))
    norm2 = a*r0+trace_t*r1/n
    position2 = a*r1+trace_t*r2/n
    eta_prime_bound = 160/(9*radius)
    # |grad[(eta-1)P g]|² <= 2|grad(Pg)|² + 2|eta' P g|² on r>radius/2.
    core_gradient2 = 2*trace_t*r0+2*(a*r1+trace_t*r2/n)/sigma**4
    momentum2 = 2*core_gradient2+2*eta_prime_bound**2*norm2
    return dict(flat_region_radius=radius/2, Gaussian_radial_tail=r0,
                state_difference_norm_squared_upper=norm2,
                q_graph_difference_squared_upper=position2,
                p_graph_difference_squared_upper=momentum2,
                eta_prime_global_upper=eta_prime_bound,
                original_initial_function_unchanged=True)


def modal_packet(reference: dict[str, Any], packet: dict[str, Any], y0: np.ndarray) -> dict[str, Any]:
    modes = reference['modes']
    return dict(w=modes.T@packet['w'], mean_q=modes.T@packet['mean_q'],
                mean_p=modes.T@packet['mean_p'], gradient=modes.T@reference['gradient'],
                Y0=modes.T@y0)


def canonical_flow(eigenvalues: np.ndarray, chi: float, t: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Entire real symplectic flow, retaining zero and negative Hessian modes."""
    omega2 = chi*eigenvalues
    c, s = np.ones_like(omega2), np.full_like(omega2, t)
    positive, negative = omega2 > 0, omega2 < 0
    wp, wn = np.sqrt(omega2[positive]), np.sqrt(-omega2[negative])
    c[positive], s[positive] = np.cos(wp*t), np.sin(wp*t)/wp
    c[negative], s[negative] = np.cosh(wn*t), np.sinh(wn*t)/wn
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
        raise FloatingPointError('Actual unstable reference overflow; no clipped Hessian or covariance')
    integrals = []
    small = np.abs(omega2*t*t) < .1
    for k in range(4):
        j = np.zeros_like(omega2)
        term = np.full_like(omega2, t**(k+2)/((k+1)*(k+2)))
        j[small] = term[small]
        for m in range(1, 16):
            term *= -omega2*t*t/((k+2*m+1)*(k+2*m+2))
            j[small] += term[small]
        if k == 0:
            j[~small] = (1-c[~small])/omega2[~small]
        elif k == 1:
            j[~small] = (t-s[~small])/omega2[~small]
        elif k == 2:
            j[~small] = (t*t-2*integrals[0][~small])/omega2[~small]
        else:
            j[~small] = (t**3-6*integrals[1][~small])/omega2[~small]
        integrals.append(j)
    return c, s, integrals


def radial_reference_observables(reference: dict[str, Any], packet: dict[str, Any], modal: dict[str, Any],
                                 t: float, clock: float, include_Y0_pulse: bool=True) -> dict[str, Any]:
    """Trace covariance and mean translation are separate, in original q units."""
    k, chi, sigma, z = reference['eigenvalues'], reference['chi'], packet['sigma'], packet['original_norm_squared']
    c, s, j = canonical_flow(k, chi, t)
    gradient, y0 = modal['gradient'], modal['Y0']
    forced_q = -chi*gradient*j[0]
    forced_p = -gradient*s
    u = t/clock
    omega = 60*math.pi/clock*(u*u-2*u**3+u**4) if include_Y0_pulse else 0.
    if include_Y0_pulse:
        forced_q -= y0*(120*math.pi*j[1]/clock**3-360*math.pi*j[2]/clock**4+240*math.pi*j[3]/clock**5)
        # p=(qdot+Omega Y0)/chi; integrate the pulse against cos directly.
        # For p response to -Omega Yp: k*int Omega(s) sin(w(t-s))/w ds.
        _, _, higher = _pulse_integrals(k, chi, t)
        forced_p += k*y0*60*math.pi*(higher[2]/clock**3-2*higher[3]/clock**4+higher[4]/clock**5)
    mq = c*modal['mean_q']+chi*s*modal['mean_p']+forced_q
    mp = -k*s*modal['mean_q']+c*modal['mean_p']+forced_p
    diagonal_m = np.sum(np.abs(modal['w'])**2, axis=1)/z
    initial_qq = sigma*sigma*(.5+diagonal_m)
    initial_pp = (.5+diagonal_m)/(sigma*sigma)
    # The diagonal sym(qp) is zero for the original degree-one polynomial.
    qq_unforced = c*c*initial_qq+(chi*s)**2*initial_pp
    pp_unforced = (k*s)**2*initial_qq+c*c*initial_pp
    initial_poly_q = c*modal['mean_q']+chi*s*modal['mean_p']
    initial_poly_p = -k*s*modal['mean_q']+c*modal['mean_p']
    trace_q_cov = float(np.sum(qq_unforced)-initial_poly_q@initial_poly_q)
    trace_p_cov = float(np.sum(pp_unforced)-initial_poly_p@initial_poly_p)
    mean2 = float(mq@mq)
    radial2 = trace_q_cov+mean2
    covariance_energy = float(.5*np.sum(chi*(pp_unforced-initial_poly_p**2)+k*(qq_unforced-initial_poly_q**2)))
    bounds = cutoff_graph_bounds(packet)
    # Unitarity preserves the state remainder; q(t)=Cq+chi Sp+affine.
    graph_error = (float(np.max(np.abs(c)))*math.sqrt(bounds['q_graph_difference_squared_upper'])
                   +chi*float(np.max(np.abs(s)))*math.sqrt(bounds['p_graph_difference_squared_upper'])
                   +float(np.linalg.norm(forced_q))*math.sqrt(bounds['state_difference_norm_squared_upper']))
    core_qnorm = math.sqrt(z*radial2)
    q_second_error = 2*core_qnorm*graph_error+graph_error*graph_error
    denominator=c+1j*chi*s/(sigma*sigma)
    linear=math.sqrt(2)*modal['w']/(sigma*denominator[:,None])
    gaussian_variance=sigma*sigma/2*c*c+(chi*s)**2/(2*sigma*sigma)
    trace_d=float(gaussian_variance.sum())
    trace_d2=float(gaussian_variance@gaussian_variance)
    diagonal_t=np.sum(np.abs(linear)**2,axis=1)
    trace_td2=float(diagonal_t@(gaussian_variance**2))
    trace_td3=float(diagonal_t@(gaussian_variance**3))
    f=(linear@packet['a'].conj()).real
    centre2=float(forced_q@forced_q)
    base_fourth=(trace_d+centre2)**2+2*trace_d2+4*float(forced_q@(gaussian_variance*forced_q))
    quadratic_mean=float(np.sum(np.abs(linear.T@(gaussian_variance*forced_q))**2))
    fourth=base_fourth+(4*(trace_d+centre2)*trace_td2+8*trace_td3+8*quadratic_mean
        +8*((trace_d+centre2)*float(f@(gaussian_variance*forced_q))
             +2*float(f@(gaussian_variance**2*forced_q))))/z
    lower=0. if radial2<=packet['radius']**2 else (radial2-packet['radius']**2)**2/fourth
    delta=math.sqrt(bounds['state_difference_norm_squared_upper']/z)
    compact_lower=(max(0.,math.sqrt(lower)-delta)/(1+delta))**2
    return dict(time=t, pulse_Omega=omega, Tr_q_covariance=trace_q_cov,
                mean_q_squared=mean2, radial_q_second_moment=radial2,
                Tr_p_covariance=trace_p_cov, covariance_energy=covariance_energy,
                radius_squared=packet['radius']**2,
                Markov_radial_exit_probability_upper=min(1.,radial2/packet['radius']**2),
                radial_fourth_moment=fourth,
                Paley_Zygmund_radial_exit_probability_lower=lower,
                ORIGINAL_compact_reference_radial_exit_probability_lower=compact_lower,
                core_radial_second_moment_over_radius_squared=radial2/packet['radius']**2,
                compact_q_second_moment_numerator_error_upper=q_second_error,
                compact_state_unitary_remainder_squared_upper=bounds['state_difference_norm_squared_upper'],
                maximum_single_coordinate_sigma_NOT_radial=float(np.sqrt(np.max(qq_unforced))),
                forced_mean_norm=float(np.linalg.norm(forced_q)),
                original_polynomial_mean_norm=float(np.linalg.norm(initial_poly_q)),
                mode_mean_q=mq, mode_mean_p=mp)


def _pulse_integrals(k: np.ndarray, chi: float, t: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    c, s, first = canonical_flow(k, chi, t)
    w2 = chi*k
    j4 = np.zeros_like(k)
    small = np.abs(w2*t*t)<.1
    term = np.full_like(k, t**6/30)
    j4[small] = term[small]
    for m in range(1,16):
        term *= -w2*t*t/((2*m+5)*(2*m+6))
        j4[small] += term[small]
    j4[~small]=(t**4-12*first[2][~small])/w2[~small]
    return c,s,first+[j4]


def gaussian_polynomial_abi(reference: dict[str, Any], packet: dict[str, Any], modal: dict[str, Any],
                            t: float, clock: float) -> dict[str, Any]:
    """Exact metaplectic core parameters plus the nonzero original cutoff remainder.

    Wave = exp(i phase+i pcentre.x) det-denominator^(-1/2)
           exp(i x.Z.x/2) [a + ell.x].  The phase is continuous from t=0;
    consumers must transport the compact remainder, not silently discard it.
    """
    k, chi, sigma = reference['eigenvalues'], reference['chi'], packet['sigma']
    c,s,_=canonical_flow(k,chi,t)
    denominator=c+1j*chi*s/(sigma*sigma)
    precision=(-k*s+1j*c/(sigma*sigma))/denominator
    linear=math.sqrt(2)*modal['w']/(sigma*denominator[:,None])
    moments=radial_reference_observables(reference,packet,modal,t,clock)
    centre_q,centre_p=_affine_centres(reference,modal,t,clock)
    phase=continuous_reference_phase(reference,packet,modal,t,clock)
    return dict(modal_q_precision=precision, modal_linear_coefficients=linear,
                original_constant_coefficients=packet['a'].copy(),
                modal_gaussian_centre_q=centre_q,modal_gaussian_centre_p=centre_p,
                c=c,s=s,chi=chi,original_norm_squared=packet['original_norm_squared'],
                moments=moments,cutoff_remainder=cutoff_graph_bounds(packet),
                phase=phase,
                affine_action_derivative_contract='(chi*|pcentre|²-qcentre.K.qcentre)/2-g.qcentre-E0',
                determinant_phase_contract='continuous log(c+i chi s/sigma²) from t=0; no principal-log reset')
 
 
def _affine_centres(reference: dict[str, Any], modal: dict[str, Any],
                    t: float, clock: float) -> tuple[np.ndarray,np.ndarray]:
    k,chi=reference['eigenvalues'],reference['chi']
    _,s,j=_pulse_integrals(k,chi,t)
    q=-chi*modal['gradient']*j[0]-modal['Y0']*(120*math.pi*j[1]/clock**3
         -360*math.pi*j[2]/clock**4+240*math.pi*j[3]/clock**5)
    p=-modal['gradient']*s+k*modal['Y0']*60*math.pi*(j[2]/clock**3
         -2*j[3]/clock**4+j[4]/clock**5)
    return q,p
 
 
def continuous_reference_phase(reference: dict[str, Any],packet: dict[str, Any],
                               modal: dict[str, Any],t: float,clock: float) -> dict[str,float]:
    """Continuous determinant/Maslov phase and actual affine classical action.

    Low-frequency short intervals use convergent quadrature; longer intervals
    use analytic polynomial/trigonometric integrals, not phase wrapping or a
    many-period sampling quadrature.
    """
    k,chi=reference['eigenvalues'],reference['chi']
    w2=chi*k
    c,s,_=canonical_flow(k,chi,t)
    phase=np.arctan2(chi*s/packet['sigma']**2,c)
    positive=w2>0
    frequency=np.sqrt(w2[positive])
    angle=frequency*t
    ratio=chi/(packet['sigma']**2*frequency)
    phase[positive]=angle+np.arctan2((ratio-1)*np.sin(angle)*np.cos(angle),
                                    np.cos(angle)**2+ratio*np.sin(angle)**2)
    if np.max(np.sqrt(np.abs(w2))*abs(t))<.25:
        nodes,weights=np.polynomial.legendre.leggauss(24)
        action=0.
        for node,weight in zip(nodes,weights):
            q,p=_affine_centres(reference,modal,(node+1)*t/2,clock)
            action+=weight*(float((chi*p@p-q@(k*q))/2-modal['gradient']@q)-reference['E0'])*t/2
    else:
        action=float(np.sum(_analytic_modal_action(k,chi,modal['gradient'],modal['Y0'],t,clock)).real)-reference['E0']*t
    log_amplitude=-.25*float(np.log(c*c+(chi*s/packet['sigma']**2)**2).sum())
    return dict(affine_classical_action=action,
                continuous_determinant_phase=-.5*float(phase.sum()),
                total_common_phase=action-.5*float(phase.sum()),
                log_amplitude_prefactor=log_amplitude,phase_reset=False)
 
 
def _analytic_modal_action(k: np.ndarray,chi: float,g: np.ndarray,y: np.ndarray,
                           t: float,clock: float) -> np.ndarray:
    """Integral of (chi p²-k q²)/2-gq for the exact quintic pulse."""
    answer=np.zeros_like(k,dtype=complex)
    zero=k==0
    answer[zero]=chi*g[zero]**2*t**3/3+g[zero]*y[zero]*(
        5*math.pi*t**4/clock**3-6*math.pi*t**5/clock**4+2*math.pi*t**6/clock**5)
    nonzero=~zero
    kk,gg,yy=k[nonzero],g[nonzero],y[nonzero]
    frequency=np.sqrt((chi*kk).astype(complex));v=chi*kk
    qp=np.zeros((len(kk),5),complex)
    qp[:,0]=-gg/kk-yy*720*math.pi/(clock**4*v**2)
    qp[:,1]=-yy*(120*math.pi/(clock**3*v)-1440*math.pi/(clock**5*v**2))
    qp[:,2]=yy*360*math.pi/(clock**4*v)
    qp[:,3]=-yy*240*math.pi/(clock**5*v)
    aq=gg/kk+yy*720*math.pi/(clock**4*v**2)
    bq=yy*(120*math.pi/(clock**3*v)-1440*math.pi/(clock**5*v**2))/frequency
    pp=np.zeros_like(qp)
    for degree in range(1,5):
        pp[:,degree-1]=degree*qp[:,degree]/chi
    pp[:,2]+=yy*60*math.pi/(chi*clock**3)
    pp[:,3]-=yy*120*math.pi/(chi*clock**4)
    pp[:,4]+=yy*60*math.pi/(chi*clock**5)
    ap,bp=frequency*bq/chi,-frequency*aq/chi
    result=np.zeros(len(kk),complex)
    for i in range(5):
        result-=gg*qp[:,i]*t**(i+1)/(i+1)
        for j in range(5):
            result+=(chi*pp[:,i]*pp[:,j]-kk*qp[:,i]*qp[:,j])/2*t**(i+j+1)/(i+j+1)
    def trig_integral(degree: int) -> tuple[np.ndarray,np.ndarray]:
        exponent=np.exp(1j*frequency*t)
        plus=np.expm1(1j*frequency*t)/(1j*frequency)
        minus=np.expm1(-1j*frequency*t)/(-1j*frequency)
        for j in range(1,degree+1):
            plus=t**j*exponent/(1j*frequency)-j*plus/(1j*frequency)
            minus=t**j/ exponent/(-1j*frequency)-j*minus/(-1j*frequency)
        return (plus+minus)/2,(plus-minus)/(2j)
    for degree in range(5):
        cosine,sine=trig_integral(degree)
        result+=(chi*pp[:,degree]*ap-kk*qp[:,degree]*aq-(gg*aq if degree==0 else 0))*cosine
        result+=(chi*pp[:,degree]*bp-kk*qp[:,degree]*bq-(gg*bq if degree==0 else 0))*sine
    result+=(chi*(ap*ap+bp*bp)-kk*(aq*aq+bq*bq))*t/4
    result+=(chi*(ap*ap-bp*bp)-kk*(aq*aq-bq*bq))*np.sin(2*frequency*t)/(8*frequency)
    result+=(chi*ap*bp-kk*aq*bq)*(1-np.cos(2*frequency*t))/(4*frequency)
    answer[nonzero]=result
    return answer


def current_six_projection(reference: dict[str, Any], packet: dict[str, Any], y0: np.ndarray,
                           order: int=4) -> dict[str, Any]:
    """Exact strong Href pressure outside the ACTUAL neutral12 trial space.

    Four Phi multipliers per held bank are integrated by Gaussian quadrature;
    all other normal directions are analytically integrated.  The original two
    EM multipliers are orthogonal to Phi Href pressure, not missing field axes.
    This is a subspace projection diagnostic, never a replacement H or IC.
    """
    n,sigma,z=packet['n'],packet['sigma'],packet['original_norm_squared']
    modes,k=reference['modes'],reference['eigenvalues']
    unit=y0/np.linalg.norm(y0)
    candidates=np.column_stack((packet['w'].real,packet['w'].imag,unit,reference['gradient']))
    lengths=np.linalg.norm(candidates,axis=0)
    candidates=candidates[:,lengths>0]/lengths[lengths>0]
    u, singular, _=np.linalg.svd(candidates,full_matrices=False)
    rank=int(np.sum(singular>1e-12*singular[0]));active=u[:,:rank]
    m=active.T@modes
    adjusted=k-reference['chi']/sigma**4
    aa=(m*adjusted)@m.T
    aau=(m*adjusted**2)@m.T
    trace_perp=float(adjusted.sum()-np.trace(aa))
    frob_perp=float(adjusted@adjusted-2*np.trace(aau)+np.sum(aa*aa))
    cross=aau-aa@aa
    dof=n-rank;variance=sigma*sigma/2
    radial_coefficient=2/(sigma*sigma*math.sqrt(2*n))
    cov_q_radial=2*variance**2*trace_perp
    l=math.sqrt(2)*active.T@packet['w']/sigma
    direction=active.T@unit
    force=active.T@reference['gradient']
    x,weights=np.polynomial.hermite.hermgauss(order)
    locations=np.asarray(list(itertools.product(range(order),repeat=rank)))
    nodes=sigma*x[locations]
    weights=np.prod(weights[locations]/math.sqrt(math.pi),axis=1)
    coordinate=nodes@direction/sigma
    radial=(2*np.sum(nodes*nodes,axis=1)/sigma**2-rank)/math.sqrt(2*n)
    multipliers=np.column_stack((np.ones(len(nodes)),math.sqrt(2)*coordinate,
                                 (2*coordinate**2-1)/math.sqrt(2),radial))
    polynomials=packet['a'][None,:]+nodes@l
    eq=np.einsum('bi,ij,bj->b',nodes,aa,nodes)+variance*trace_perp
    varq=4*variance*np.einsum('bi,ij,bj->b',nodes,cross,nodes)+2*variance**2*frob_perp
    # Omit only the E0 + chi*n/(2sigma²) multiple of the unchanged input:
    # it lies EXACTLY in the trial space and cancels from the outside residual.
    action=polynomials*(.5*eq+nodes@force)[:,None]+reference['chi']/sigma**2*(nodes@l)
    results=[]
    for bank in range(2):
        density=np.abs(polynomials[:,bank])**2
        gram=np.einsum('b,bi,bj->ij',weights*density,multipliers,multipliers)
        gram[3,3]+=radial_coefficient**2*2*dof*variance**2*float(weights@density)
        rhs=np.einsum('b,bi,b->i',weights,multipliers,polynomials[:,bank].conj()*action[:,bank])
        rhs[3]+=.5*radial_coefficient*cov_q_radial*float(weights@density)
        norm=float(weights@(np.abs(action[:,bank])**2+.25*density*varq))
        captured=float(np.vdot(rhs,np.linalg.solve(gram,rhs)).real)
        without_radial=float(np.vdot(rhs[:3],np.linalg.solve(gram[:3,:3],rhs[:3])).real)
        outside=norm-captured
        tolerance=1e-10*max(norm,1.)
        if outside < -tolerance:
            raise AssertionError('Strong reference subspace norm became negative beyond roundoff')
        results.append(dict(bank=bank,Gram=gram,rhs=rhs,action_norm_squared=norm,
                            captured_norm_squared=captured,outside_norm_squared=outside,
                            outside_if_actual_radial_mode_omitted_squared=norm-without_radial,
                            outside_rate_over_original_norm=math.sqrt(max(0.,outside)/z)))
    return dict(actual_modes=['1','sqrt2 Y0.q/(|Y0|sigma)','Y0 quadratic',
                              'ALL2430 normal radial','first harmonic','ALL52 coexact radial'],
                exact_Phi_active_rank=rank,Gaussian_rule_order=order,
                analytically_integrated_normal_complement=dof,
                unchanged_original_norm_squared=z,banks=results,
                outside_strong_Href_rate=math.sqrt(max(0.,sum(b['outside_norm_squared'] for b in results))/z),
                common_identity_energy_removed_ONLY_in_outside_projection=True)
