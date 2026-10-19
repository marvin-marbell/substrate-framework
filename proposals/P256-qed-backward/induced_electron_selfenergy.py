#!/usr/bin/env python3
"""NEW regulated one-loop charged-band discriminator in the continuous-time law.

The photon field includes its positive particle-hole continuum, not an extrapolated
static Maxwell pole. Only an explicitly integrated physical momentum shell is
claimed. No frozen photon supplier or source/receiver bytes are changed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time
import sys
from typing import Any, cast

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ('induced_electron_selfenergy.py', 'induced_photon_response.py',
                'induced_photon_mode_frame.py', 'induced_phi_backreaction.py',
                'induced_compton_transfer.py', 'improved_phi_band.py',
                'rotation_band.py', 'sea_selection.py', 'soft_budget.py',
                'local_orbital_stability.py', 'compton.py')
SOURCE_SNAPSHOTS = {name: (ROOT/name).read_text() for name in SOURCE_NAMES}
SOURCE_HASHES = {name: hashlib.sha256(text.encode()).hexdigest()
                 for name, text in SOURCE_SNAPSHOTS.items()}

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh
from scipy.optimize import brentq
from induced_photon_response import ResponseConfig, SpectralResponse, native_path_proof
from induced_photon_mode_frame import compute_mode, transverse_frame
from induced_phi_backreaction import PhiConfig, uniform

for _name in SOURCE_NAMES:
    if hashlib.sha256((ROOT/_name).read_bytes()).hexdigest() != SOURCE_HASHES[_name]:
        raise RuntimeError(f'Source changed while loading {_name}')

I4 = np.eye(4, dtype=complex)
EQUATIONS = {
    'photon': 'H=((PT+ST*p)^2+(SL*p)^2)/(2*eps_b)+sum Delta*(x^2+p^2)/2+AT*K_T*AT/2; S=sqrt(W_physical)/Delta',
    'Gauss': 'SL=qhat_unit.S; physical bare longitudinal kernel 1/(eps_b*|qhat/a|^2), not old Maxwell/Coulomb supplier',
    'spectrum': 'Diagonalize sqrt(Kx)*Gp*sqrt(Kx); Ai=Omega*(Kx^-1/2 U)_field; A0=-(SL*sqrt(Delta)*U_bath)/(eps_b*|qhat/a|)',
    'selfenergy': 'integral dq/(2pi)^3 sum_r [Vem Pplus Vem/(w-Ek-Omega)+Vabs Pminus Vabs/(w+Ek+Omega)]/(2Omega)+1/2 K<AA>+Vbare*(I/2-Pminus)',
    'vertices': 'Vem=A0*I-Jphysical.A; Vabs=-A0*I-Jphysical.A; Jphysical=a*literal Peierls current; literal contact=a^2*path second derivative',
    'CAR_Coulomb': 'Adding one tagged electron to the half-filled active sea gives Vbare*(I/2-Pfilled), with Hartree/background q=0 excluded; no extra flavor',
    'Ward': 'd_p Sigma from matrix Frechet electron triangle + literal mixed Peierls contact + third insertion tadpole; compared with finite difference; d_w Sigma supplies temporal vertex',
    'scope': 'Regulated leading 1/N / one-loop quantum tangent; not full finite-N interacting pole, global compact phase or full Root/Phi/Gauss admission',
}


def sqrt_positive(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = eigh((matrix+matrix.T)/2)
    if np.min(values) <= 0:
        raise ArithmeticError(f'Nonpositive canonical position Hessian: {np.min(values)}')
    return ((vectors*np.sqrt(values))@vectors.T,
            (vectors/np.sqrt(values))@vectors.T)


def photon_spectrum(model: SpectralResponse, data: dict[str, Any], bins: int,
                    *, compare_isolated_poles: bool = True) -> dict[str, Any]:
    """Full positive frequency spectrum, including the density/Gauss PH sector."""
    cfg = model.config
    factor = cfg.flavors/cfg.a
    delta = data['delta']
    weights = factor*data['weights']
    edges = np.geomspace(float(np.min(delta))*(1-1e-12),
                         float(np.max(delta))*(1+1e-12), bins+1)
    index = np.clip(np.searchsorted(edges, delta)-1, 0, bins-1)
    matrices = np.empty((bins, 3, 3))
    electric_matrices = np.empty_like(matrices)
    for i in range(3):
        for j in range(3):
            matrices[:, i, j] = np.bincount(index, weights=weights[:, i, j], minlength=bins)
            electric_matrices[:, i, j] = np.bincount(
                index, weights=weights[:, i, j]/delta**3, minlength=bins)
    gaps, columns, intervals = [], [], []
    psd_roundoff = 0.
    for number, (matrix, moment) in enumerate(zip(matrices, electric_matrices)):
        amount = float(np.trace(matrix))
        if amount == 0:
            continue
        values, vectors = eigh((matrix+matrix.T)/2)
        if np.min(values) < -2e-11*max(1., amount):
            raise ArithmeticError('Actual spectral current measure is not positive')
        # Only roundoff-negative weights are clipped; never floor a frequency.
        psd_roundoff = max(psd_roundoff, float(max(0., -np.min(values))))
        keep = values > 0
        root_weight = vectors[:, keep]*np.sqrt(values[keep])
        inverse_weight = vectors[:, keep]/np.sqrt(values[keep])
        electric_values, electric_vectors = eigh(inverse_weight.T@moment@inverse_weight)
        if np.min(electric_values) <= 0:
            raise ArithmeticError('Nonpositive exact electric moment in occupied gap bin')
        # Matrix-valued moment matching, not a fitted low pole:
        # sum W_r=W_bin and sum W_r/Delta_r^3=E_bin simultaneously.
        for value, vector in zip(electric_values, electric_vectors.T):
            gap = float(value**(-1/3))
            if not edges[number]*(1-1e-10) <= gap <= edges[number+1]*(1+1e-10):
                raise ArithmeticError('Moment-matched gap leaves its actual spectral interval')
            gaps.append(gap)
            columns.append(root_weight@vector/gap)
        intervals.append((matrix, moment, float(edges[number]*(1-1e-10)),
                          float(edges[number+1]*(1+1e-10))))
    gaps = np.asarray(gaps)
    s = np.stack(columns, axis=1)
    qhat = np.asarray(data['qhat'])
    qnorm = float(np.linalg.norm(qhat))/cfg.a
    direction = qhat/np.linalg.norm(qhat)
    _, _, vh = np.linalg.svd(direction[None, :])
    transverse = vh[1:].T
    st, sl = transverse.T@s, direction@s
    kt = factor*transverse.T@data['static']@transverse
    bare = 1/(2*cfg.inertia)
    size = len(gaps)+2
    root_field,inverse_field=sqrt_positive(kt)
    sqrt_gaps=np.sqrt(gaps)
    momentum = np.zeros((size,size))
    momentum[:2, :2] = np.eye(2)/bare
    momentum[:2, 2:] = st/bare
    momentum[2:, :2] = st.T/bare
    momentum[2:, 2:] = np.diag(gaps)+(st.T@st+np.outer(sl, sl))/bare
    # The position form is exactly block diagonal (2x2 K_T, diagonal gaps).
    # Do not diagonalize it densely or multiply two dense block-diagonal roots.
    frequency_matrix=np.empty_like(momentum)
    frequency_matrix[:2,:2]=root_field@momentum[:2,:2]@root_field
    frequency_matrix[:2,2:]=(root_field@momentum[:2,2:])*sqrt_gaps
    frequency_matrix[2:,:2]=frequency_matrix[:2,2:].T
    frequency_matrix[2:,2:]=sqrt_gaps[:,None]*momentum[2:,2:]*sqrt_gaps[None,:]
    omega2, vectors = eigh(frequency_matrix)
    if np.min(omega2) <= 0:
        raise ArithmeticError('Unresolved/nonpositive photon frequency; no floor is permitted')
    omega = np.sqrt(omega2)
    spatial = transverse@(inverse_field@vectors[:2]*omega)
    scalar = -(sl*np.sqrt(gaps))@vectors[2:]/(bare*qnorm)
    legs = np.vstack((scalar, spatial)).T
    static_screened = 1/(bare*qnorm**2+cfg.flavors/cfg.a**3*data['chi'])
    compressed_screened = 1/(qnorm**2*(bare+np.sum(sl**2/gaps)))
    # A UV PH continuum need not contain two sharp subthreshold photons.
    # Still retain ALL positive canonical modes, A0 legs and original-spectrum
    # Euclidean prices; this is not a fallback to an IR Maxwell propagator.
    actual_modes=None;native_omega=None;retarded_probe=None
    if compare_isolated_poles:
        actual_modes=compute_mode(cfg,np.asarray(data['q']),model=model,data=data)
        native_omega=np.asarray([row['omega'] for row in actual_modes['two_polarizations']])
        retarded_probe=float(np.max(native_omega))
    sumrule = spatial@spatial.T
    expected = transverse@transverse.T/bare
    longitudinal_sum = float(np.sum(scalar**2/omega2))
    covariance = (spatial/omega)@spatial.T/2
    static_covariance = (spatial/omega2)@spatial.T
    interval_price = resolvent_price(model,data,gaps,s,intervals,retarded_probe)
    scalar_errors = []
    for mode in range(2 if compare_isolated_poles else 0):
        x = omega2[mode]
        chi = bare*qnorm**2+cfg.flavors/cfg.a**3*np.sum(
            data['density']*delta/(delta**2-x))
        d = cfg.flavors/cfg.a**2*np.einsum(
            'bi,b->i', data['mixed_weights'], 1/(delta**2-x))
        native_scalar = omega[mode]*float(d@spatial[:, mode])/chi
        scalar_errors.append(abs(float(scalar[mode])-native_scalar))
    controls: dict[str,Any] = dict(
        positive_frequency_min=float(np.min(omega)),
        psd_roundoff_weight=psd_roundoff,
        canonical_residual=float(np.linalg.norm(frequency_matrix@vectors-vectors*omega2)/max(1.,np.linalg.norm(frequency_matrix))),
        transverse_sumrule_error=float(np.linalg.norm(sumrule-expected)),
        longitudinal_sumrule_error=abs(longitudinal_sum-(1/(bare*qnorm**2)-compressed_screened)),
        native_lowpole_relative=(float(np.max(np.abs(omega[:2]/native_omega-1)))
                                 if native_omega is not None else None),
        native_lowresidue_relative=(float(max(abs(np.sum(spatial[:,i]**2)/row['residue']-1)
            for i,row in enumerate(actual_modes['two_polarizations']))) if actual_modes is not None else None),
        screened_static_relative=abs(compressed_screened/static_screened-1),
        scalar_momentum_leg_native_error=max(scalar_errors) if scalar_errors else None,
        scalar_spatial_quadrature_sumrule=float(np.linalg.norm((scalar/omega)@spatial.T)),
        static_transverse_error=float(np.linalg.norm(transverse.T@static_covariance@transverse-np.linalg.inv(kt))),
        vertex_Ward=float(data['vertex_error']), density_Ward=float(data['spectral_ward']),
        mixed_Ward=float(data['mixed_ward']),
        longitudinal_static_raw=float(qhat@data['static_raw']@qhat),
        longitudinal_static_identity=float(data['longitudinal_static']),
        native_mode=actual_modes,
        isolated_photon_comparison=compare_isolated_poles,
        kappa_static_min=float(np.min(np.linalg.eigvalsh(kt))/qnorm**2),
        no_longitudinal_bath_static_defect=float(abs(1/(bare*qnorm**2)-static_screened)),
        missing_contact_min=float(np.min(np.linalg.eigvalsh(
            kt-factor*transverse.T@np.diag(data['contact'])@transverse))),
    )
    controls['resolvent_interval_price'] = interval_price
    if controls['transverse_sumrule_error'] > 2e-6 or controls['canonical_residual'] > 2e-10:
        raise ArithmeticError('Full photon spectrum fails canonical positive control')
    return dict(q=np.asarray(data['q']), omega=omega, legs=legs, covariance=covariance,
                static_screened=compressed_screened, bare_coulomb=1/(bare*qnorm**2),
                controls=controls, bins=bins, modes=len(omega))


def resolvent_price(model, data, gaps, s, intervals, omega):
    """Original-spectrum interval certificate, separate from generated Ward."""
    cfg = model.config
    delta = data['delta']
    weights = cfg.flavors/cfg.a*data['weights']
    exact_electric = np.einsum('bij,b->ij', weights, 1/delta**3)
    compressed_electric = (s/gaps)@s.T
    electric_error = float(np.linalg.norm(exact_electric-compressed_electric, 2))
    native_bubble = cfg.flavors/cfg.a*data['bubble']
    compressed_bubble = (s*gaps)@s.T
    contact_difference = float(np.linalg.norm(compressed_bubble-native_bubble, 2))
    # A secant of f(y)=y^(1/3), with |f''|<=2*u^5/9, bounds BOTH
    # matrix-positive measures that match W and W/Delta^3 on [l,u].
    contact_bound = sum(float(np.linalg.norm(w,2))*u**5*(l**-3-u**-3)**2/36
                        for w,_,l,u in intervals)
    rows = []
    q = np.asarray(data['qhat'])/cfg.a
    probes=[('Euclidean',x) for x in (0.,cfg.m*math.sqrt(cfg.g0),1.,1/cfg.a)]
    if omega is not None:
        probes.insert(0,('retarded',omega))
    for kind,frequency in probes:
        x = frequency**2
        sign = -1. if kind == 'retarded' else 1.
        if kind == 'retarded' and frequency >= float(np.min(delta)):
            raise ArithmeticError('Retarded interval proof outside the actual spectral gap')
        native = np.einsum('bij,b->ij', weights, 1/(delta*(delta**2+sign*x)))
        compressed = (s*(gaps/(gaps**2+sign*x)))@s.T
        bound = electric_error+x*sum(float(np.linalg.norm(e,2))*(
            1/(l*l+sign*x)-1/(u*u+sign*x)) for _,e,l,u in intervals)
        measured = float(np.linalg.norm(native-compressed,2))
        # Original independent density and signed mixed-current measures.
        chi_density = cfg.flavors/cfg.a**3*np.sum(
            data['density']*delta/(delta**2+sign*x))
        mixed = cfg.flavors/cfg.a**2*np.einsum(
            'bi,b->i', data['mixed_weights'], 1/(delta**2+sign*x))
        density_mismatch = abs(float(chi_density-q@native@q))
        mixed_mismatch = float(np.linalg.norm(mixed-native@q))
        if measured > bound+2e-10*max(1.,np.linalg.norm(native,2)):
            raise ArithmeticError('Original spectral resolvent escapes its interval bound')
        rows.append(dict(kind=kind,frequency=frequency,current_resolvent_error=measured,
            operator_norm_error_upper=bound,original_density_Ward_mismatch=density_mismatch,
            original_signed_mixed_Ward_mismatch=mixed_mismatch,
            compressed_density_error_upper=float(q@q)*bound+density_mismatch,
            compressed_mixed_error_upper=float(np.linalg.norm(q))*bound+mixed_mismatch))
    return dict(electric_matrix_match_error=electric_error,
        native_contact_compression_difference=contact_difference,
        native_contact_compression_error_upper=contact_bound,
        rows=rows,scope='Rigorous gap-interval resolvent bound at every listed Euclidean frequency and subthreshold retarded pole; the same formula applies to all Euclidean frequencies. Not merely bin convergence, not a full loop-speed error certificate.',
        formula='P_E=E-x sum Eweights/(Delta^2+x); P_R=E+x sum Eweights/(Delta^2-x); ||error||<=moment_roundoff+x sum ||Ebin||*(1/(l^2+-x)-1/(u^2+-x))',
        approximation='Generated rho=q.J/Delta_bin is a compressed Ward law; ORIGINAL density and signed mixed-current mismatches above are independently checked. Effective contact differs from original by the explicitly priced bubble moment.')


def rotations(rep: np.ndarray, direction: np.ndarray) -> np.ndarray:
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.,1.), repeat=3):
            matrix = np.diag(signs)@np.eye(3)[list(permutation)]
            if np.max(np.abs(matrix@rep-direction)) < 1e-12:
                return matrix
    raise ArithmeticError('Direction is not in the actual cubic orbit')


def angular_rule(order: int):
    groups = []
    if order == 14:
        weights = (1/15, 3/40, 0.)
    elif order == 26:
        weights = (1/21, 9/280, 4/105)
    else:
        raise ValueError('Actual Lebedev 14 or 26 direction rule required')
    axes = [sign*np.eye(3)[axis] for axis in range(3) for sign in (-1.,1.)]
    corners = [np.asarray(signs)/math.sqrt(3) for signs in itertools.product((-1.,1.),repeat=3)]
    edges = []
    for zero in range(3):
        axes_nonzero = [axis for axis in range(3) if axis != zero]
        for signs in itertools.product((-1.,1.),repeat=2):
            vector = np.zeros(3)
            vector[axes_nonzero] = signs
            edges.append(vector/math.sqrt(2))
    for rep, members, weight in zip((np.array([1.,0.,0.]),np.ones(3)/math.sqrt(3),
                                   np.array([1.,1.,0.])/math.sqrt(2)),
                                  (axes,corners,edges),weights):
        if weight:
            groups.append((rep,members,4*math.pi*weight))
    return groups


def rotate_spectrum(spectrum: dict[str, Any], rotation: np.ndarray) -> dict[str, Any]:
    result = dict(spectrum)
    result['q'] = rotation@spectrum['q']
    result['legs'] = spectrum['legs'].copy()
    result['legs'][:, 1:] = spectrum['legs'][:, 1:]@rotation.T
    result['covariance'] = rotation@spectrum['covariance']@rotation.T
    return result


class ElectronLoop:
    def __init__(self, config: ResponseConfig):
        self.config = config
        self.model = SpectralResponse(config, 2)

    def elementary(self, p: np.ndarray, q: np.ndarray):
        cfg = self.config
        h = self.model.h((cfg.a*(p-q))[None, :])[0]
        values, vectors = np.linalg.eigh(h)
        energy = float(np.mean(values[2:]))
        upper, lower = vectors[:,2:],vectors[:,:2]
        pp, pm = upper@upper.conj().T,lower@lower.conj().T
        j, _ = self.model.vertices((cfg.a*(p-q/2))[None, :],cfg.a*q)
        _, contact = self.model.vertices((cfg.a*p)[None, :],cfg.a*q)
        derivatives = np.zeros((3,3,4,4),complex)
        third = np.zeros((3,3,4,4),complex)
        dh = np.zeros((3,4,4),complex)
        for axis,length,t in self.model.paths:
            path_sum = sum(np.exp(1j*cfg.a*q[axis]*(ell-(length-1)/2))
                           for ell in range(length)).real
            middle = np.exp(1j*length*cfg.a*(p[axis]-q[axis]/2))
            phase = np.exp(1j*length*cfg.a*p[axis])
            internal = np.exp(1j*length*cfg.a*(p[axis]-q[axis]))
            derivatives[axis,axis] -= cfg.a**2*length*path_sum*(middle*t+middle.conjugate()*t.conj().T)
            third[axis,axis] -= 1j*cfg.a**3*length*path_sum**2*(phase*t-phase.conjugate()*t.conj().T)
            dh[axis] += 1j*cfg.a*length*(internal*t-internal.conjugate()*t.conj().T)
        return energy,pp,pm,cfg.a*j[0],cfg.a**2*contact[0],derivatives,third,dh

    def sigma(self, w: float, p: np.ndarray, fields: list[tuple[float,dict[str,Any]]],
              *, low_only: bool = False, drop_contact: bool = False,
              derivatives: bool = False) -> tuple[np.ndarray,dict[str,Any]]:
        total = np.zeros((4,4),complex)
        components = dict(radiative=np.zeros_like(total),contact=np.zeros_like(total),coulomb=np.zeros_like(total))
        dw = np.zeros_like(total)
        dp = np.zeros((3,4,4),complex)
        dp_without_contact = np.zeros_like(dp)
        denominator_min = math.inf
        for measure,field in fields:
            q = field['q']
            energy,pp,pm,j,c, dj,third,dh = self.elementary(p,q)
            count = 2 if low_only else len(field['omega'])
            omega,legs = field['omega'][:count],field['legs'][:count]
            spatial = np.einsum('ri,ijk->rjk',legs[:,1:],j)
            vp = legs[:,0,None,None]*I4-spatial
            vm = -legs[:,0,None,None]*I4-spatial
            plus,minus = w-energy-omega,w+energy+omega
            denominator_min=min(denominator_min,float(np.min(np.abs(plus))),float(np.min(np.abs(minus))))
            if np.any(plus >= 0) or np.any(minus <= 0):
                raise ArithmeticError('Electron contour crosses emission/occupied-band pole; use explicit width/Cherenkov calculation')
            coeff=measure/(2*omega)
            rp=vp@pp@vp
            rm=vm@pm@vm
            loop=np.einsum('rij,r->ij',rp,coeff/plus)+np.einsum('rij,r->ij',rm,coeff/minus)
            covariance = ((legs[:,1:]/omega[:,None]).T@legs[:,1:])/2
            tadpole=measure/2*np.einsum('i,ijk->jk',np.diag(covariance),c)
            coulomb=measure*field['bare_coulomb']*(I4/2-pm)
            components['radiative']+=loop
            if not drop_contact:
                components['contact']+=tadpole
            components['coulomb']+=coulomb
            if derivatives:
                dw-=np.einsum('rij,r->ij',rp,coeff/plus**2)+np.einsum('rij,r->ij',rm,coeff/minus**2)
                for axis in range(3):
                    dv=-np.einsum('ri,ijk->rjk',legs[:,1:],dj[axis])
                    triangle=np.zeros_like(total)
                    endcontacts=np.zeros_like(total)
                    for projector,other,vertex,denominator,sign in ((pp,pm,vp,plus,1.),(pm,pp,vm,minus,-1.)):
                        same=projector@dh[axis]@projector
                        cross=(projector@dh[axis]@other+other@dh[axis]@projector)*sign/(2*energy)
                        frechet=same[None,:,:]/denominator[:,None,None]**2+cross[None,:,:]/denominator[:,None,None]
                        triangle+=np.einsum('rij,r->ij',vertex@frechet@vertex,coeff)
                        endcontacts+=np.einsum('rij,r->ij',dv@projector@vertex+vertex@projector@dv,coeff/denominator)
                    dpm=-(pp@dh[axis]@pm+pm@dh[axis]@pp)/(2*energy)
                    dcoulomb=-measure*field['bare_coulomb']*dpm
                    dtadpole=measure/2*np.einsum('i,ijk->jk',np.diag(covariance),third[axis])
                    dp_without_contact[axis]+=triangle+dcoulomb
                    dp[axis]+=triangle+endcontacts+dcoulomb+(0 if drop_contact else dtadpole)
        for value in components.values():
            total+=value
        return total,dict(components=components,frequency_derivative=dw,
                          momentum_derivative=dp,without_mixed_contact=dp_without_contact,
                          denominator_min=denominator_min)

    def bare_energy(self,p):
        values=np.linalg.eigvalsh(self.model.h((self.config.a*np.asarray(p))[None,:])[0])
        return float(np.mean(values[2:]))

    def pole(self,p,fields,*,low_only=False,drop_contact=False,particle=1):
        p=np.asarray(p)
        if particle not in (-1,1):
            raise ValueError('Positive particle or negative filled-band hole branch required')
        occupied=slice(2,None) if particle==1 else slice(None,2)
        bare_h=self.model.h((self.config.a*p)[None,:])[0]
        bare=self.bare_energy(p)
        def inverse(w):
            sigma,_=self.sigma(w,p,fields,low_only=low_only,drop_contact=drop_contact)
            return w-float(np.mean(np.linalg.eigvalsh(bare_h+sigma)[occupied]))
        omega_min=min(float(np.min(field['omega'])) for _,field in fields)
        edge=min(self.bare_energy(p-field['q'])+float(np.min(field['omega']))
                 for _,field in fields)*(1-1e-10)
        near=max(0.,bare-min(omega_min, .1*bare))
        lower,upper=(near,edge) if particle==1 else (-edge,-near)
        if inverse(lower)*inverse(upper) >= 0:
            raise ArithmeticError('No isolated regulated one-loop charged pole in actual contour domain')
        root=cast(float,brentq(inverse,lower,upper,xtol=1e-13,rtol=np.float64(1e-11),full_output=False))
        sigma,details=self.sigma(root,p,fields,low_only=low_only,drop_contact=drop_contact,derivatives=True)
        values,vectors=np.linalg.eigh(bare_h+sigma)
        electron=vectors[:,occupied]
        residue=float(np.mean(np.linalg.eigvalsh(np.eye(2)-electron.conj().T@details['frequency_derivative']@electron)))
        return dict(energy=particle*root,signed_frequency=root,particle=particle,
                    bare_energy=bare,residue=1/residue,
                    mass_shift=particle*root-bare,components={key:particle*float(np.trace(electron.conj().T@value@electron).real/2)
                        for key,value in details['components'].items()},denominator_min=details['denominator_min'])


def make_fields(config, spectral_order, bins, radial_order, angular_order, qmin, qmax, cache):
    nodes,weights=leggauss(radial_order)
    radii=(qmax+qmin)/2+(qmax-qmin)*nodes/2
    radial_weights=(qmax-qmin)*weights/2
    fields=[]
    controls=[]
    for radius,weight in zip(radii,radial_weights):
        for representative,members,angular_weight in angular_rule(angular_order):
            key=(spectral_order,bins,float(radius),tuple(representative))
            if key not in cache:
                model=SpectralResponse(config,spectral_order)
                data=model.spectral(radius*representative)
                cache[key]=photon_spectrum(model,data,bins)
            spectrum=cache[key]
            controls.append(spectrum['controls'])
            measure=float(weight*radius**2*angular_weight/(2*math.pi)**3)
            for member in members:
                fields.append((measure,rotate_spectrum(spectrum,rotations(representative,member))))
    return fields,controls


def make_hard_bz_fields(config,spectral_order,bins,radial_order,face_order,qcut,cache):
    """Exact cube-ray geometry, not an inscribed spherical UV cutoff."""
    z,w=leggauss(face_order);nodes=(z+1)/2;weights=w/2
    rz,rw=leggauss(radial_order)
    model=SpectralResponse(config,spectral_order)
    fields=[];controls=[];volume=0.
    for s,ws in zip(nodes,weights):
        for v,wv in zip(nodes,weights):
            ray=np.array([1.,s,s*v])
            lower=qcut/float(np.linalg.norm(ray));upper=math.pi/config.a
            if lower>=upper:
                raise ValueError('Hard sphere reaches a face of the actual BZ')
            panel=lower
            while panel<upper:
                end=min(2*panel,upper)
                for zrad,wrad in zip(rz,rw):
                    u=(panel+end)/2+(end-panel)*zrad/2
                    q=u*ray
                    key=('hard',spectral_order,bins,tuple(q))
                    if key not in cache:
                        data=model.spectral(q)
                        cache[key]=photon_spectrum(model,data,bins,compare_isolated_poles=False)
                    spectrum=cache[key]
                    measure=float(ws*wv*s*(end-panel)*wrad*u*u/(2*(2*math.pi)**3))
                    # q1>q2>q3>0 throughout the Gauss nodes: all48 signed
                    # permutations are distinct. Rotate physical vector legs,
                    # not a sorted eigenvalue branch or a norm-per-leg flux.
                    for permutation in itertools.permutations(range(3)):
                        for signs in itertools.product((-1.,1.),repeat=3):
                            rotation=np.diag(signs)@np.eye(3)[list(permutation)]
                            fields.append((measure,rotate_spectrum(spectrum,rotation)))
                    controls.append(dict(q=q.tolist(),modes=spectrum['modes'],**spectrum['controls']))
                    volume+=48*measure
                panel=end
    exact_volume=config.a**-3-qcut**3/(6*math.pi**2)
    geometry=dict(parameterization='q=u*(1,s,s*v), s,v in(0,1); 48 signed permutations; physical Jacobian u^2*s; u from qcut/|ray| to pi/a',
                  integrated_measure=volume,exact_hard_BZ_measure=exact_volume,
                  volume_relative_error=abs(volume/exact_volume-1),
                  omitted_ball_radius=qcut,shape='ENTIRE actual cubic BZ minus the stated spherical IR ball')
    return fields,controls,geometry


def hard_bare_shell(loop,fields) -> dict[str,Any]:
    """Actual first-order quantities; do not resum a large UV shift into a pole."""
    p0=np.zeros(3);mass=loop.bare_energy(p0)
    _,vectors=np.linalg.eigh(loop.model.h(p0[None,:])[0])
    upper,lower=vectors[:,2:],vectors[:,:2]
    def project(matrix):
        return float(np.trace(upper.conj().T@matrix@upper).real/2)
    sigma,details=loop.sigma(mass,p0,fields,derivatives=True)
    delta=project(sigma)
    negative,_=loop.sigma(-mass,p0,fields)
    hole_delta=-float(np.trace(lower.conj().T@negative@lower).real/2)
    cm=loop.config.v*math.sqrt(loop.config.gs)
    steps=[]
    for step in (.025*mass/cm,.0125*mass/cm):
        p=np.array([step,0.,0.]);energy=loop.bare_energy(p)
        _,up=np.linalg.eigh(loop.model.h((loop.config.a*p)[None,:])[0])
        up=up[:,2:]
        shifted,_=loop.sigma(energy,p,fields)
        shift=float(np.trace(up.conj().T@shifted@up).real/2)
        # Expand c_R^2 to FIRST order, including the rest-mass change.
        # Multiplying a huge M+deltaM by a shifted dispersion would silently
        # include uncontrolled powers of this one-loop UV counterterm.
        correction=2*mass*(shift-delta)/step**2+cm**2*delta/mass
        steps.append(dict(momentum=step,energy_shift=shift,velocity_squared_correction=correction))
    correction=(4*steps[1]['velocity_squared_correction']-steps[0]['velocity_squared_correction'])/3
    return dict(bare_mass=mass,mass_shift=delta,mass_shift_fraction=delta/mass,
        components={name:project(matrix) for name,matrix in details['components'].items()},
        reference_hole_mass_shift=hole_delta,charge_conjugate_mass_error=abs(hole_delta-delta),
        frequency_derivative_positive_projection=project(details['frequency_derivative']),
        denominator_min=details['denominator_min'],curvature_steps=steps,
        first_order_velocity_squared_correction=correction,
        first_order_velocity_squared_relative_correction=correction/cm**2,
        signed_critical_mass_candidate=dict(physical_counterterm=-delta,
            primitive_m_counterterm=-delta/math.sqrt(loop.config.g0),
            primitive_m_after_hard_counterterm=loop.config.m-delta/math.sqrt(loop.config.g0),
            scope='Measured HARD-shell first-order candidate only. Not an implemented/tuned bare law, not the complete critical mass: missing soft ball, interacting filling/vacuum/Phi feedback and higher orders remain. Negative bare m is not admitted by the frozen positive-m primitive.'),
        scope='Native continuous-time fixed-mean hard-shell one-loop expansion on the BARE shell. Large mass/velocity ratios reject a small-correction promotion; these numbers are not resummed physical poles or a renormalized Compton law.')


def compute_uv_case(label,args) -> dict[str,Any]:
    cfg,mean=selected_config(label,args.mean_order);loop=ElectronLoop(cfg)
    orders=args.uv_orders;bins=args.uv_bins;radial=args.uv_radial;faces=args.uv_faces
    specs=[(orders[0],bins[0],radial[0],faces[0]),
           (orders[0],bins[-1],radial[0],faces[0]),
           (orders[0],bins[-1],radial[-1],faces[0]),
           (orders[0],bins[-1],radial[-1],faces[-1]),
           (orders[-1],bins[-1],radial[-1],faces[-1])]
    cache={};rows=[];selected_fields=[];selected_controls=[]
    for order,bin_count,radial_order,face_order in dict.fromkeys(specs):
        fields,controls,geometry=make_hard_bz_fields(cfg,order,bin_count,radial_order,face_order,args.qmax,cache)
        measured=hard_bare_shell(loop,fields)
        rows.append(dict(fermion_BZ_order=order,bins=bin_count,radial=radial_order,face=face_order,
                         bare_shell=measured,geometry=geometry,independent_native_spectra=len(controls),
                         cubic_orbit_samples=len(fields)))
        selected_fields,selected_controls=fields,controls
        print(json.dumps({'UVcase':label,'specification':rows[-1]}) ,flush=True)
    selected=rows[-1]
    ward=ward_proof(loop,selected_fields)
    return dict(label=label,config=asdict(cfg),mean_provenance=mean,
        hard_full_BZ_convergence=rows,selected_hard_bare_shell=selected['bare_shell'],ward=ward,
        all_sampled_static_coexact_positive=bool(min(c['kappa_static_min'] for c in selected_controls)>0),
        selected_controls=selected_controls,
        integration_error_scope='Separated actual bin/radial/cube-face/fermion-BZ differences, plus original current/density/signed-mixed Euclidean resolvent interval bounds at EVERY independent native spectrum. No certified integrated self-energy/velocity error or unsampled whole-BZ positivity.',
        physical_scope='No IR two-pole propagator extrapolation: ALL positive Hamiltonian PH/Gauss modes and A0/mixed legs at every UV sample. Full microscopic nonlinear photon/filled-source/Phi corrections are not supplied by this Gaussian leading1/N branch.')


def measured_dispersion(loop,fields):
    mass=loop.bare_energy(np.zeros(3))
    cm=loop.config.v*math.sqrt(loop.config.gs)
    step=.025*mass/cm
    rest=loop.pole(np.zeros(3),fields)
    rows=[]
    for h in (step,step/2):
        shifted=loop.pole(np.array([h,0.,0.]),fields)
        c2=2*rest['energy']*(shifted['energy']-rest['energy'])/h**2
        rows.append(dict(momentum=h,energy=shifted['energy'],velocity_squared=c2,
                         speed=math.sqrt(c2),speed_ratio=math.sqrt(c2)/cm))
    extrapolated=(4*rows[1]['velocity_squared']-rows[0]['velocity_squared'])/3
    return dict(rest=rest,curvature=rows,bare_speed=cm,
                extrapolated_velocity_squared=extrapolated,
                extrapolated_speed=math.sqrt(extrapolated),
                extrapolated_speed_ratio=math.sqrt(extrapolated)/cm,
                curvature_step_relative=abs(rows[0]['velocity_squared']/rows[1]['velocity_squared']-1))


def ward_proof(loop,fields):
    mass=loop.bare_energy(np.zeros(3))
    p=np.array([.12,-.07,.04])*mass/(loop.config.v*math.sqrt(loop.config.gs))
    w=loop.bare_energy(p)
    _,derivatives=loop.sigma(w,p,fields,derivatives=True)
    h=1e-4*mass/(loop.config.v*math.sqrt(loop.config.gs))
    spatial=[]
    for axis in range(3):
        shift=np.eye(3)[axis]*h
        upper,_=loop.sigma(w,p+shift,fields)
        lower,_=loop.sigma(w,p-shift,fields)
        finite=(upper-lower)/(2*h)
        spatial.append(dict(axis=axis,absolute=float(np.linalg.norm(finite-derivatives['momentum_derivative'][axis])),
                            relative=float(np.linalg.norm(finite-derivatives['momentum_derivative'][axis])/max(1e-20,np.linalg.norm(finite))),
                            missing_mixed_contact=float(np.linalg.norm(finite-derivatives['without_mixed_contact'][axis]))))
    hw=1e-4*mass
    upper,_=loop.sigma(w+hw,p,fields)
    lower,_=loop.sigma(w-hw,p,fields)
    finite=(upper-lower)/(2*hw)
    temporal=float(np.linalg.norm(finite-derivatives['frequency_derivative'])/max(1e-20,np.linalg.norm(finite)))
    q=fields[len(fields)//2][1]['q']
    j=loop.config.a*loop.model.vertices((loop.config.a*(p-q/2))[None,:],loop.config.a*q)[0][0]
    qhat=2*np.sin(loop.config.a*q/2)/loop.config.a
    difference=loop.model.h((loop.config.a*p)[None,:])[0]-loop.model.h((loop.config.a*(p-q))[None,:])[0]
    primitive=float(np.linalg.norm(np.einsum('i,ijk->jk',qhat,j)-difference))
    if max(row['relative'] for row in spatial)>1e-4 or temporal>1e-4:
        raise ArithmeticError('Independent one-loop Ward/Peierls insertion derivative failed')
    return dict(primitive_longitudinal=primitive,spatial=spatial,temporal_relative=temporal,
                source='Independent matrix Frechet electron triangle plus literal mixed contact and third insertion; not a derivative-defined fake vertex')


def cubic_transfer_proof(model, q, bins):
    rotation = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    original = model.spectral(q)
    rotated = model.spectral(rotation@q)
    first = photon_spectrum(model, original, bins)
    second = photon_spectrum(model, rotated, bins)
    a = first['covariance']
    b = second['covariance']
    spatial = float(np.linalg.norm(rotation@a@rotation.T-b)/np.linalg.norm(a))
    static = float(np.linalg.norm(rotation@original['static']@rotation.T-rotated['static']))
    electric = float(np.linalg.norm(rotation@original['epsilon_link']@rotation.T-rotated['epsilon_link']))
    frequency = float(np.max(np.abs(first['omega']/second['omega']-1)))
    if max(spatial, frequency) > 2e-5:
        raise ArithmeticError('Independent actual cubic spectrum transfer failed')
    return dict(rotation=rotation.tolist(),q=np.asarray(q).tolist(),
                covariance_relative=spatial,full_spectrum_relative=frequency,
                static_absolute=static,electric_absolute=electric,
                scope='Signed-permutation symmetry of literal uniform cubic paths, independently checked against actual rotated 3D BZ response; no angular interpolation')


def cherenkov_proof(loop, spectral_order, qmin):
    model = SpectralResponse(loop.config,spectral_order)
    soft_q = np.array([qmin/4,0.,0.])
    soft = compute_mode(loop.config, soft_q, model=model)
    photon_speed = soft['omega']/np.linalg.norm(soft_q)
    mass = loop.bare_energy(np.zeros(3))
    cm = loop.config.v*math.sqrt(loop.config.gs)
    def velocity(p):
        momentum = np.array([p,0.,0.])
        h = loop.model.h((loop.config.a*momentum)[None,:])[0]
        j = loop.config.a*loop.model.vertices(
            (loop.config.a*momentum)[None,:],np.zeros(3))[0][0,0]
        return float(np.trace(h@j).real/(4*loop.bare_energy(momentum)))
    upper = min(math.pi/(4*loop.config.a),max(10*mass/cm, .3))
    soft_threshold = (cast(float,brentq(lambda p:velocity(p)-photon_speed,0.,upper,full_output=False))
                      if velocity(upper)>photon_speed else None)
    q = 3*qmin*np.array([math.cos(.1),math.sin(.1),0.])
    actual = compute_mode(loop.config, q, model=model)
    rows = []
    for photon in actual['two_polarizations']:
        omega = photon['omega']
        def inverse(p):
            incoming = np.array([p,0.,0.])
            return loop.bare_energy(incoming)-loop.bare_energy(incoming-q)-omega
        if inverse(upper)<=0:
            rows.append(dict(onset=None,scope='No forward on-shell root in stated native momentum bracket'))
            continue
        onset = cast(float,brentq(inverse,0.,upper,xtol=1e-12,full_output=False))
        incoming = np.array([onset,0.,0.])
        outgoing = incoming-q
        _,uin = np.linalg.eigh(loop.model.h((loop.config.a*incoming)[None,:])[0])
        _,uout = np.linalg.eigh(loop.model.h((loop.config.a*outgoing)[None,:])[0])
        j = loop.config.a*loop.model.vertices(
            (loop.config.a*(incoming-q/2))[None,:],loop.config.a*q)[0][0]
        vertex = photon['scalar_polarization']*I4-np.einsum('i,ijk->jk',photon['polarization'],j)
        qhat = 2*np.sin(loop.config.a*q/2)/loop.config.a
        longitudinal = np.einsum('i,ijk->jk',qhat,j)
        ward = uout[:,2:].conj().T@(omega*I4-longitudinal)@uin[:,2:]
        missing_density = uout[:,2:].conj().T@(-longitudinal)@uin[:,2:]
        rows.append(dict(onset=onset,energy_residual=inverse(onset),
            current_matrix_norm=float(np.linalg.norm(uout[:,2:].conj().T@vertex@uin[:,2:])),
            emission_longitudinal_Ward=float(np.linalg.norm(ward)),
            missing_density_Ward=float(np.linalg.norm(missing_density)),photon=photon))
    return dict(soft_q=soft_q.tolist(),soft_native_mode=soft,
        soft_group_velocity_threshold=soft_threshold,soft_threshold_velocity=(
            velocity(soft_threshold) if soft_threshold is not None else None),
        finite_q=q.tolist(),onsets=rows,
        scope='Native leading bare-band Cherenkov onset with actual retarded photon residues/current/density, not a dressed decay-rate claim. Regulated loop curvature at rest cannot be extrapolated to this momentum.')


def selected_config(label: str, mean_order: int):
    if label in ('base','half'):
        name=('induced_phi_backreaction_receipt.json' if label=='base' else 'induced_phi_backreaction_half_a_receipt.json')
        raw=(ROOT/name).read_bytes()
        capsule=json.loads(raw)
        mean=capsule['uniform'][-1]
        cfg=ResponseConfig(**capsule['response']['config'])
        provenance=dict(kind='Actual source-bound classical BULK uniform mean; NOT finiteN6 quantum mean',
                        receipt=name,sha256=hashlib.sha256(raw).hexdigest(),row=dict(size=mean['size'],
                        quadrature_order=mean['quadrature_order'],g0=mean['g0'],gs=mean['gs']),
                        full10_gradient_max=mean['full10_gradient_max'])
    elif label in ('lighter','lightest'):
        mass=.02 if label=='lighter' else .002
        phi=PhiConfig(active_m=mass)
        measured=uniform(phi,0,mean_order)
        coarse=uniform(phi,0,max(2,mean_order-2))
        g=measured['G']
        cfg=ResponseConfig(a=phi.a,m=mass,r=phi.light_r,v=phi.light_v,
            g0=float(g[0,0]),gs=float(np.trace(g[1:,1:])/3),species=((1,mass),(63,phi.auxiliary_m)))
        provenance=dict(kind='NEW actual bulk mean at changed active mass, not inherited stationarity',
            config=asdict(phi),size=0,quadrature_order=mean_order,G=g.tolist(),
            cubic_encoding_error=float(np.linalg.norm(g-np.diag([cfg.g0,cfg.gs,cfg.gs,cfg.gs]))),
            full10_gradient_norm=float(np.linalg.norm(measured['derivative'])),
            full10_hessian_min=float(np.min(np.linalg.eigvalsh(measured['h10']))),
            lower_mean_order=max(2,mean_order-2),
            mean_G_order_difference=float(np.linalg.norm(g-coarse['G'])),
            quadrature_stationary_G_error_upper=float(np.linalg.norm(measured['derivative'])/phi.stiffness),
            stationary_scope='Global strong convexity in G of this quadrature energy; not a quantum or continuum quadrature-error theorem')
    else:
        raise ValueError('Named actual base/half/lighter/lightest law profile required')
    return cfg,provenance

def literal_ir_integrands(model,p):
    """Independent literal-matrix electric and magnetic q=0 integrands."""
    a=model.config.a
    h=model.h(a*p);current,contact=model.vertices(a*p,np.zeros(3))
    current*=a;contact*=a**2
    energy=np.sqrt(np.einsum('bij,bji->b',h,h).real/4)
    n=h/energy[:,None,None];pm=(I4-n)/2;pp=(I4+n)/2
    electric_weight=sum(2*np.trace(pm@current[:,axis]@pp@current[:,axis],axis1=1,axis2=2).real/3
                        for axis in range(3))
    hx=current[:,0];hxx=contact[:,0];jy=current[:,1]
    ex=np.einsum('bij,bji->b',h,hx).real/(4*energy)
    exx=(np.einsum('bij,bji->b',hx,hx).real/4+
          np.einsum('bij,bji->b',h,hxx).real/4-ex**2)/energy
    nx=hx/energy[:,None,None]-h*(ex/energy**2)[:,None,None]
    nxx=hxx/energy[:,None,None]-2*hx*(ex/energy**2)[:,None,None]-h*(exx/energy**2)[:,None,None]+2*h*(ex**2/energy**3)[:,None,None]
    w0=2*np.trace(pm@jy@pp@jy,axis1=1,axis2=2).real
    w2=np.trace(-nxx@jy@pp@jy+pm@jy@nxx@jy+nx@jy@nx@jy,axis1=1,axis2=2).real/8
    error=float(np.max(abs(h@h-energy[:,None,None]**2*I4)))
    return electric_weight/(2*energy)**3,w0*exx/(16*energy**2)-w2/(2*energy),error


def native_dirac_patch(loop,angular=26,radial=16) -> dict[str,Any]:
    """Subtract/add data for the singular IR part, derived from literal paths.

    The patch is NOT a standalone gauge theory: its finite magnetic boundary
    term belongs to the complementary native BZ integral. Only its universal
    logarithm is shared by epsilon and kappa/c_m^2.
    """
    cfg=loop.config;cm=cfg.v*math.sqrt(cfg.gs);mass=loop.bare_energy(np.zeros(3))
    h0=loop.model.h(np.zeros((1,3)))[0]
    j0=cfg.a*loop.model.vertices(np.zeros((1,3)),np.zeros(3))[0][0]
    matrices=[h0/mass,*[j/cm for j in j0]]
    clifford=max(float(np.linalg.norm(a@b+b@a-(2*I4 if i==j else 0)))
                 for i,a in enumerate(matrices) for j,b in enumerate(matrices))
    pcut=.1/cfg.a;mu=cm*pcut
    coefficient=1/(6*math.pi**2*cm)
    fraction=mu/math.sqrt(mu**2+mass**2)
    electric=(2*(math.asinh(mu/mass)-fraction)/3+fraction**3/9)/(4*math.pi**2*cm)
    magnetic=cm**2*electric+cm*(fraction**3-fraction**5/5)/(24*math.pi**2)
    theta=.1
    h3=cfg.r*math.sqrt(cfg.g0)/(4*cm);j3=cfg.r*math.sqrt(cfg.g0)/cm
    hmax=theta**2/6+h3*theta**3;jmax=theta**2/2+j3*theta**3
    aj=2*(2+jmax)/(1-hmax)**3
    ah=(2*(2+jmax)*(1+jmax)+3)/(1-hmax)**4
    electric_remainder=(aj*(theta**2/4+j3*theta**3/3)+
                        ah*(theta**2/12+h3*theta**3/3))/(4*math.pi**2*cm)
    def integrate(nrad,nang):
        nodes,weights=leggauss(nrad)
        scale=min(pcut,mass/cm);edges=[0.,scale]
        while edges[-1]<pcut:
            edges.append(min(pcut,2*edges[-1]))
        directions=[];angles=[]
        for _,members,weight in angular_rule(nang):
            directions.extend(members);angles.extend([weight]*len(members))
        directions=np.asarray(directions);angles=np.asarray(angles)
        epsilon=0.;kappa=0.;matrix_error=0.
        for left,right in zip(edges[:-1],edges[1:]):
            radius=(left+right)/2+(right-left)*nodes/2
            radial_weights=(right-left)*weights/2
            p=(radius[:,None,None]*directions[None,:,:]).reshape(-1,3)
            measure=(radial_weights[:,None]*radius[:,None]**2*angles[None,:]/(2*math.pi)**3).ravel()
            native_electric,native_magnetic,error=literal_ir_integrands(loop.model,p)
            epsilon+=float(np.sum(measure*native_electric))
            kappa+=float(np.sum(measure*native_magnetic))
            matrix_error=max(matrix_error,error)
        return dict(epsilon=epsilon,kappa=kappa,clifford_h_squared_error=matrix_error,
                    radial=nrad,angular=nang,radial_panels=len(edges)-1)
    coarse=integrate(max(4,radial-4),14);fine=integrate(radial,angular)
    active=cfg.species[0][0] if cfg.species else cfg.flavors
    return dict(native_mass=mass,native_speed=cm,native_Clifford_error=clifford,
        cutoff=dict(theta_radius=.1,physical_radius=pcut,kinetic_energy_scale=mu),
        per_active_flavor_log_coefficients=dict(epsilon=coefficient,kappa=cm**2*coefficient),
        active_multiplicity=active,Dirac_patch=dict(epsilon=electric,kappa=magnetic,
            native_electric_lattice_error_upper_uniform_mass=electric_remainder,
            error_proof='|delta h|/E_D <= theta^2/6+r sqrt(g0)theta^3/(4c); |delta J_j|/c <= theta^2/2+r sqrt(g0)theta^3/c. Normalize h/E with E_native >=(1-hmax) E_D, bound literal interband trace and E^-3, integrate theta^2,theta^3 against dp/p. Finite uniformly as M tends zero.',
            finite_boundary_kappa_minus_c2epsilon=magnetic-cm**2*electric),
        native_patch=dict(coarse=coarse,fine=fine,
            epsilon_order_difference=abs(fine['epsilon']-coarse['epsilon']),
            kappa_order_difference=abs(fine['kappa']-coarse['kappa']),
            epsilon_minus_Dirac=fine['epsilon']-electric,kappa_minus_Dirac=fine['kappa']-magnetic),
        derivation=dict(Clifford='beta=h_native(0)/M; alpha_j=a*literal_J_j(0)/c_m, checked rather than imposed',
            electric='Wjj=2 Re Tr(Pminus Jj Pplus Jj)=4 c_m^2(1-c_m^2 pj^2/E^2); epsilon_patch=1/(4pi^2 c_m) int t^2/E^3*(1-t^2/(3E^2)) dt',
            magnetic='Native h_x=a Jx and h_xx=a^2 Cxx generate n_x,n_xx. W2=Tr(-n_xx Jy Pplus Jy+Pminus Jy n_xx Jy+n_x Jy n_x Jy)/8. kappa_patch=int [W0 E_xx/(16E^2)-W2/(2E)] dp/(2pi)^3. Transverse contact is q_x-independent on these actual axial paths.',
            log='Electric coefficient 1/(6pi^2 c_m), magnetic c_m/(6pi^2): derived from the angular averages of literal filled-doublet traces, not copied QED matching',
            asymptotic='epsilon_Dirac=C[log(2mu/M)-5/6]+O(M^2/mu^2); finite patch magnetic surface term tends c_m/(30pi^2)'),
        scope='Controlled-design subtract/add ingredients at actual finite masses and native paths. Patch quadrature convergence is recorded; no unmeasured full-BZ or ultra-small-mass result and no critical radiative mass tuning is claimed.')

def native_bulk_ir_coefficients(config,order) -> dict[str,Any]:
    """Independent q=0 derivatives, no finite-q speed fit or bare curl term."""
    model=SpectralResponse(config,order)
    total_electric=1/(2*config.inertia);total_magnetic=0.;records=[]
    parts=model.species_models or [(config.flavors,model)]
    for count,child in parts:
        epsilon=0.;kappa=0.;matrix_error=0.;probes=0
        for k,weight in child.chunks():
            _,_,h,_,j,contact,_=child.coefficient_fields(k,np.zeros(3))
            j*=config.a;contact*=config.a**2
            energy=np.sqrt(np.sum(h*h,axis=1));n=h/energy[:,None]
            j_squared=np.sum(j*j,axis=2)
            n_dot_j=np.einsum('bm,bim->bi',n,j)
            electric=np.mean((j_squared-n_dot_j**2)/(2*energy[:,None]**3),axis=1)
            hx=j[:,0];hxx=contact[:,0];jy=j[:,1]
            ex=np.sum(h*hx,axis=1)/energy
            exx=(np.sum(hx*hx+h*hxx,axis=1)-ex**2)/energy
            nx=hx/energy[:,None]-h*(ex/energy**2)[:,None]
            nxx=hxx/energy[:,None]-2*hx*(ex/energy**2)[:,None]-h*(exx/energy**2)[:,None]+2*h*(ex**2/energy**3)[:,None]
            ny=np.sum(n*jy,axis=1);nxy=np.sum(nx*jy,axis=1)
            nxxy=np.sum(nxx*jy,axis=1);ny2=np.sum(jy*jy,axis=1)
            w0=4*(ny2-ny**2)
            w2=nxy**2-nxxy*ny-np.sum(nx*nx,axis=1)*ny2
            magnetic=w0*exx/(16*energy**2)-w2/(2*energy)
            indices=np.linspace(0,len(k)-1,8,dtype=int)
            literal_e,literal_k,_=literal_ir_integrands(child,k[indices]/config.a)
            error=max(float(np.max(abs(electric[indices]-literal_e)/np.maximum(1.,abs(literal_e)))),
                      float(np.max(abs(magnetic[indices]-literal_k)/np.maximum(1.,abs(literal_k)))))
            matrix_error=max(matrix_error,error);probes+=len(indices)
            if error>2e-10:
                raise ArithmeticError('Native q=0 derivative coefficients disagree with literal matrix/contact traces')
            epsilon+=float(np.sum(weight*electric))/config.a**3
            kappa+=float(np.sum(weight*magnetic))/config.a**3
        total_electric+=count*epsilon;total_magnetic+=count*kappa
        records.append(dict(count=count,m=child.config.m,epsilon_per_flavor=epsilon,
                            kappa_per_flavor=kappa,literal_matrix_relative_error=matrix_error,
                            literal_matrix_probes=probes))
    if min(total_electric,total_magnetic)<=0:
        raise ArithmeticError('Native independent q=0 electric/magnetic derivatives are not stable')
    cm=config.v*math.sqrt(config.gs)
    return dict(order=order,epsilon=total_electric,kappa=total_magnetic,
        photon_speed=math.sqrt(total_magnetic/total_electric),matter_speed=cm,
        photon_over_matter_speed=math.sqrt(total_magnetic/total_electric)/cm,
        Delta=total_magnetic-cm**2*total_electric,species=records,
        magnetic_derivative='Same full periodic contact-complete identity: transverse contact is q_x-independent; differentiate B(0)-B(q_x) at q=0 with native h_x=aJx and h_xx=a^2Cxx.',
        scope='Actual full BZ zero-frequency derivatives at fixed classical mean, quadrature order recorded. No photon speed calibration and no claim of finiteN/full interacting phase.')



def neutral_soft_dressing(loop,fields,dispersion,soft_mode,qmin):
    """NEW neutral CAR trial bookkeeping and its conditional leading IR cloud."""
    cfg=loop.config;size=1<<40;length=cfg.a*size
    qir=2*math.pi/length;qhatmin=2*math.sin(math.pi/size)/cfg.a
    momentum=float(dispersion['curvature'][-1]['momentum'])
    quantum=round(momentum/qir);quantized=quantum*qir
    p=np.array([momentum,0.,0.]);w=float(dispersion['curvature'][-1]['energy'])
    sigma,derivative=loop.sigma(w,p,fields,derivatives=True)
    _,vectors=np.linalg.eigh(loop.model.h((cfg.a*p)[None,:])[0]+sigma)
    electron=vectors[:,2:]
    current=cfg.a*loop.model.vertices((cfg.a*p)[None,:],np.zeros(3))[0][0,0]
    kinetic=electron.conj().T@(current+derivative['momentum_derivative'][0])@electron
    norm=electron.conj().T@(I4-derivative['frequency_derivative'])@electron
    velocity=float(np.trace(np.linalg.solve(norm,kinetic)).real/2)
    speed=float(soft_mode['omega']/np.linalg.norm(soft_mode['q']));z=float(soft_mode['residue'])
    if abs(velocity)>=speed:
        raise ArithmeticError('Soft dressing ray enters Cherenkov cone; no stationary charged-packet cloud')
    coefficients=[]
    for order in (32,64):
        mu,weights=leggauss(order)
        angular=2*math.pi*float(np.sum(weights*velocity**2*(1-mu**2)/(speed-velocity*mu)**2))
        coefficients.append(z*angular/(2*speed*(2*math.pi)**3))
    hole=loop.pole(np.zeros(3),fields,particle=-1)
    h_tag=loop.model.h(np.array([[cfg.a*quantized,0.,0.]]))[0]
    h_hole=loop.model.h(np.zeros((1,3)))[0]
    _,tagvectors=np.linalg.eigh(h_tag);_,holevectors=np.linalg.eigh(h_hole)
    tag=tagvectors[:,2];reference=holevectors[:,0]
    return dict(reference='NEW c_plus(p).dagger c_minus(0)|filled active sea>, NOT frozen Ref4 source',
        charge=dict(delta_tag=1,delta_reference_hole=-1,total=0,
                    origin='Relative to the declared filled-sea charge origin; no added fractional or uniform compensator'),
        CAR=dict(tag_empty=True,reference_filled=True,norm_squared=float(np.vdot(tag,tag).real*np.vdot(reference,reference).real)),
        even_box=dict(size=size,volume=size**3,physical_length=length,smallest_q=qir,
            tag_momentum_quantum=quantum,quantized_tag_momentum=quantized,
            momentum_rounding_error=abs(quantized-momentum),
            SU4_center_positive=bool((130*size**3)%4==0),
            scope='Even-volume centre condition is necessary only. BULK mean is not claimed stationary in this arbitrary bookkeeping box; no local Haar/Gauss or Root admission is inferred.'),
        reference_hole_regulated_pole=hole,
        connected_pair_CAR_Coulomb_norm_upper=1/(length**3*(1/(2*cfg.inertia))*qhatmin**2),
        connected_pair_scope='Exact finite-box CAR occupation cross term bounded by bare nonzero-q kernel / physical volume. Transverse/annihilation and actual source-map admission remain separate.',
        leading_soft_cloud=dict(qir=qir,qmin=qmin,regulated_Ward_group_velocity=velocity,
            native_IR_photon_speed=speed,native_IR_residue=z,log_coefficient=coefficients[-1],
            angular_order_difference=abs(coefficients[0]-coefficients[1]),
            photon_number=coefficients[-1]*math.log(qmin/qir),
            energy=coefficients[-1]*speed*(qmin-qir),
            formula='n_soft=Z/[2c(2pi)^3] log(qmin/qIR) int dOmega [v^2-(v.n)^2]/(c-v.n)^2; rest reference hole has zero leading transverse velocity',
            scope='Leading asymptotic soft dressing, not an exact finite-window cloud or a transferred Ref4 wavepacket. Nonzero velocity gives log-divergent Fock photon number at infinite volume; no sharp infinite-volume charged pole claim.'),
        unpaid='Sea charge-origin/compact Gauss realization, full dressed interacting source/receiver, finiteN Phi/SU4/Root phase, and full hard-shell radiative mass/velocity remain unpaid.')



def ultraviolet_static_probe(config,orders) -> dict[str,Any]:
    """Selected full-zone coexact controls, not a whole-BZ positivity proof."""
    rows=[]
    for order in dict.fromkeys(orders):
        model=SpectralResponse(config,order)
        for fraction in (.5,1.):
            for direction in (np.array([1.,0.,0.]),np.array([1.,1.,0.]),np.ones(3)):
                theta=math.pi*fraction*direction;q=theta/config.a
                data=model.spectral(q);frame,_=transverse_frame(q,data['qhat'])
                factor=config.flavors/config.a
                values=np.linalg.eigvalsh(factor*(frame.T@data['static']@frame))
                raw=np.linalg.eigvalsh(factor*(frame.T@data['static_raw']@frame))
                qphysical2=float(data['qhat']@data['qhat'])/config.a**2
                rows.append(dict(order=order,qtheta=theta.tolist(),q=q.tolist(),
                    coexact_eigenvalues=values.tolist(),raw_coexact_eigenvalues=raw.tolist(),
                    kappa_min=float(np.min(values)/qphysical2),
                    raw_alias_operator_norm=float(np.linalg.norm(factor*(data['static']-data['static_raw']),2)),
                    native_vertex_Ward=float(data['vertex_error']),density_Ward=float(data['spectral_ward']),
                    signed_mixed_Ward=float(data['mixed_ward']),
                    longitudinal_identity_link=float(data['longitudinal_static']),
                    pair_threshold=float(np.min(data['delta'])),species=data.get('species',[]),
                    quadrature_points=model.quadrature_points))
    return dict(rows=rows,all_selected_positive=bool(min(min(row['coexact_eigenvalues']) for row in rows)>0),
                selected_kappa_min=min(row['kappa_min'] for row in rows),
                scope='Actual axes/faces/body corners at pi/2 and pi with mass-resolved filled auxiliaries, both BZ orders and raw aliases. Positive selected controls do NOT prove whole-BZ RPA positivity. A negative selected mode rejects the global zero-field vacuum, not all induced constructions.')


def compute_case(label,args) -> dict[str,Any]:
    cfg,mean=selected_config(label,args.mean_order)
    cfg=replace(cfg,orders=tuple(args.orders))
    loop=ElectronLoop(cfg)
    ultraviolet=ultraviolet_static_probe(cfg,args.orders)
    cache={}
    rows=[]
    specifications = (
        (args.orders[-1],args.bins[0],args.radial[0],args.angular[0]),
        (args.orders[-1],args.bins[-1],args.radial[0],args.angular[0]),
        (args.orders[-1],args.bins[-1],args.radial[-1],args.angular[0]),
        (args.orders[-1],args.bins[-1],args.radial[-1],args.angular[-1]),
        (args.orders[0],args.bins[-1],args.radial[-1],args.angular[-1]))
    for spectral_order,bins,radial,angular in specifications:
        fields,controls=make_fields(cfg,spectral_order,bins,radial,angular,args.qmin,args.qmax,cache)
        measured=measured_dispersion(loop,fields)
        rows.append(dict(spectral_order=spectral_order,bins=bins,radial=radial,angular=angular,
                         integration_points=len(fields),dispersion=measured,
                         photon_controls_max={key:max(float(c[key]) for c in controls)
                             for key in ('native_lowpole_relative','native_lowresidue_relative','screened_static_relative',
                                         'transverse_sumrule_error','longitudinal_sumrule_error','canonical_residual')}))
    final_fields,final_controls=make_fields(cfg,args.orders[-1],args.bins[-1],args.radial[-1],args.angular[-1],args.qmin,args.qmax,cache)
    selected=rows[-2]['dispersion']
    bare_mass=loop.bare_energy(np.zeros(3))
    low_sigma,_=loop.sigma(bare_mass,np.zeros(3),final_fields,low_only=True)
    _,rest_vectors=np.linalg.eigh(loop.model.h(np.zeros((1,3)))[0])
    low_shift=float(np.trace(rest_vectors[:,2:].conj().T@low_sigma@rest_vectors[:,2:]).real/2)
    low=dict(on_bare_shell_shift=low_shift,
             scope='Deliberately omitted high-frequency PH/density continuum: a shift discriminator, NOT a solved dressed pole')
    no_contact=loop.pole(np.zeros(3),final_fields,drop_contact=True)
    ward=ward_proof(loop,final_fields)
    # Independent lower BZ quadrature at one actual off-axis transfer.
    q=np.array([.04,.03,.02])
    lower_model=SpectralResponse(cfg,args.orders[0]);upper_model=SpectralResponse(cfg,args.orders[-1])
    lower=compute_mode(cfg,q,model=lower_model);upper=compute_mode(cfg,q,model=upper_model)
    cherenkov=cherenkov_proof(loop,args.orders[-1],args.qmin)
    photon_speed=float(cherenkov['soft_native_mode']['omega']/np.linalg.norm(cherenkov['soft_native_mode']['q']))
    mass=loop.bare_energy(np.zeros(3));cm=cfg.v*math.sqrt(cfg.gs)
    symmetry=cubic_transfer_proof(upper_model,q,args.bins[-1])
    qmin=args.qmin;qmax=args.qmax;bare=1/(2*cfg.inertia)
    # Quantitative unconditional Coulomb part; transverse hard-shell stability is unpaid.
    sinc=2*math.sin(cfg.a*qmin/2)/(cfg.a*qmin)
    omitted_soft_coulomb_upper=qmin/(4*math.pi**2*bare*sinc**2)
    patch=native_dirac_patch(loop)
    bulk=[native_bulk_ir_coefficients(cfg,order) for order in dict.fromkeys(args.orders)]
    physical=bulk[-1]
    active=patch['active_multiplicity']
    subtract_add=dict(epsilon_regular=physical['epsilon']-active*patch['Dirac_patch']['epsilon'],
        kappa_regular=physical['kappa']-active*patch['Dirac_patch']['kappa'],
        scope='Exact finite-mass numerical subtract/add on the native full BZ result; regular includes auxiliaries, primitive bare electric inertia, lattice and patch-boundary terms. No imported Maxwell matching.')
    delta=physical['Delta'];target=.01
    needed=abs(delta)/(cm**2*(2*target-target**2 if delta<0 else 2*target+target**2))
    hierarchy=dict(target_relative_cone_error=target,required_epsilon_if_regular_Delta_fixed=needed,
        added_active_IR_log=max(0.,(needed-physical['epsilon'])/(active*patch['per_active_flavor_log_coefficients']['epsilon'])),
        scope='Conditional leading-log design discriminator, NOT an observed ultra-small-mass result. Fixed-mean/UV assumption and signed critical radiative mass/filled-source obligations are not solved by lowering bare m.')
    neutral=neutral_soft_dressing(loop,final_fields,selected,cherenkov['soft_native_mode'],qmin)
    jbound=2*cfg.a*sum(length*np.linalg.norm(matrix,2) for _,length,matrix in loop.model.paths)
    cbound=2*cfg.a**2*sum(length**2*np.linalg.norm(matrix,2) for _,length,matrix in loop.model.paths)
    hard_assumed=min(min(c['kappa_static_min'] for c in final_controls),ultraviolet['selected_kappa_min'])
    qhat_cut=2*math.sin(cfg.a*qmax/2)/cfg.a
    hard_upper=cfg.a**-3*((5/(2*bare)+4*jbound**2/hard_assumed)/qhat_cut**2+
                         cbound/(2*math.sqrt(bare*hard_assumed)*qhat_cut))
    hard_price=dict(assumed_hard_kappa_lower=hard_assumed,
        assumption='NOT certified: extrapolate the minimum of sampled IR AND selected UV stiffness ratios to the ENTIRE hard BZ. This is a conditional bound, not an actual hard-shell error certificate.',
        native_current_operator_upper=float(jbound),native_contact_operator_upper=float(cbound),
        conditional_rest_on_bare_shell_mass_norm_upper=float(hard_upper),
        relative_to_mass=float(hard_upper/mass),
        derivation='At rest E(q)>=M, electronic emission/filled denominators >=Omega. Full PH sumrules bound A0^2/Omega^2<=Vbare and sum |Ai|^2/Omega^2=Tr K^-1; CAR Coulomb adds Vbare/2; seagull covariance Cauchy bound. Integrand <=(5/(2eps_b)+4Jmax^2/kappa_h)/qhat^2+Cmax/(2sqrt(eps_b*kappa_h)|qhat|). Whole BZ measure <=a^-3.',
        scope='No hard velocity-derivative or loop gap-bin speed bound. The deliberately conservative magnitude is exposed rather than silently omitting the shell.')
    return dict(label=label,config=asdict(cfg),mean_provenance=mean,
        shell=dict(qmin=qmin,qmax=qmax,measure='physical d^3q/(2pi)^3; spherical shell inside actual lattice BZ',
                   frequency='ALL positive compressed canonical frequencies including PH/density continuum; no frequency-cut low-pole extrapolation'),
        convergence=rows,selected_dispersion=selected,ward=ward,native_paths=native_path_proof(cfg),
        native_bulk_IR=bulk,native_Dirac_patch=patch,subtract_add=subtract_add,
        material_hierarchy_discriminator=hierarchy,neutral_reference_and_soft_dressing=neutral,
        ultraviolet_static_discriminator=ultraviolet,
        regulated_matter_over_photon_IR_speed=selected['extrapolated_speed']/physical['photon_speed'],
        photon_BZ_order_difference=dict(omega_relative=abs(lower['omega']/upper['omega']-1),
                                      residue_relative=abs(lower['residue']/upper['residue']-1),q=q.tolist()),
        lowpole_only_rest=low,drop_contact_rest=no_contact,
        Cherenkov=cherenkov,cubic_symmetry=symmetry,
        omitted=dict(soft_q0_to_qmin_Coulomb_mass_norm_upper=omitted_soft_coulomb_upper,
                     soft_relative_to_mass=omitted_soft_coulomb_upper/mass,
                     hard_qmax_to_BZ='Not silently dropped: no full-q stiffness/positive spectral certificate available from these IR probes. Conditional bound requires min hard stiffness and electronic emission denominator; no finite full speed bound is claimed.',
                     hard_Coulomb_crude_norm_upper=(2*math.pi/cfg.a)**3/(2*(2*math.pi)**3*bare*(2*math.sin(cfg.a*qmax/2)/cfg.a)**2),
                     hard_full_PH_conditional=hard_price,
                     quantum='FiniteN/nonlinear, full Phi/SU4/Root state and radiative vacuum backreaction not transferred'),
        photon_records=[dict(q=field['q'].tolist(),bins=field['bins'],modes=field['modes'],controls=field['controls'])
                        for field in cache.values()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',default='base,half,lighter,lightest')
    parser.add_argument('--orders',default='6,8')
    parser.add_argument('--bins',default='32,128')
    parser.add_argument('--radial',default='3,5')
    parser.add_argument('--angular',default='14,26')
    parser.add_argument('--mean-order',type=int,default=10)
    parser.add_argument('--qmin',type=float,default=.01)
    parser.add_argument('--qmax',type=float,default=.12)
    parser.add_argument('--uv-only',action='store_true')
    parser.add_argument('--uv-orders',default='4,6')
    parser.add_argument('--uv-bins',default='32,64')
    parser.add_argument('--uv-radial',default='2,3')
    parser.add_argument('--uv-faces',default='2,3')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for name in ('orders','bins','radial','angular','uv_orders','uv_bins','uv_radial','uv_faces'):
        setattr(args,name,tuple(int(value) for value in getattr(args,name).split(',')))
    if not 0<args.qmin<args.qmax:
        raise ValueError('Positive physical momentum shell required')
    started=time.monotonic()
    cases=[]
    for label in args.cases.split(','):
        cases.append(compute_uv_case(label,args) if args.uv_only else compute_case(label,args))
        if not args.uv_only:
            print(json.dumps({'case':label,'rest':cases[-1]['selected_dispersion']['rest'],
                'curvature':cases[-1]['selected_dispersion']['curvature'],
                'bulk_IR':cases[-1]['native_bulk_IR'][-1]},indent=2),flush=True)
    inputs=dict(cases=args.cases,orders=args.orders,bins=args.bins,radial=args.radial,angular=args.angular,
                mean_order=args.mean_order,qmin=args.qmin,qmax=args.qmax,
                uv_only=args.uv_only,uv_orders=args.uv_orders,uv_bins=args.uv_bins,
                uv_radial=args.uv_radial,uv_faces=args.uv_faces,
                physical_cases=[dict(label=case['label'],config=case['config'],
                                     mean_provenance=case['mean_provenance']) for case in cases])
    capsule: dict[str,Any] = dict(branch='NEW regulated native continuous-time radiative discriminator; frozen490 kernels plus verified physical paired-mode API',
        assigned_plan_epoch='03b34911221ba1898315373edaa7c4c3bcac718462e7fe51a842560eb6933f87',
        governing_plan_id='issue-222-induced-photon-delta13-continuation-20261008',
        governing_plan_revision='cb2914f24b938d376070cc10a9681bd8dfb89b1c198628f01b8dffa7b4ca7bae',
        equations=EQUATIONS,inputs=inputs,input_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        cases=cases,source_hashes=SOURCE_HASHES,source_snapshots=SOURCE_SNAPSHOTS,
        source_unchanged_since_import={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
                                       for name,value in SOURCE_HASHES.items()},
        runtime=dict(python=platform.python_version(),numpy=np.__version__,executable=sys.executable),
        elapsed_seconds=time.monotonic()-started)
    if any(not capsule['source_unchanged_since_import'][name] for name in
           ('induced_electron_selfenergy.py','induced_photon_response.py','induced_photon_mode_frame.py')):
        raise RuntimeError('Owned radiative or frozen photon/provider source changed during execution')
    args.output.write_text(json.dumps(capsule,indent=2,allow_nan=False)+'\n')
    print(f'Natural exit: {args.output}; actual regulated loop, no full cone repair claim',flush=True)


if __name__=='__main__':
    main()
