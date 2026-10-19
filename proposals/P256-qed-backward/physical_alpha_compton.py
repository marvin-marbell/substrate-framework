#!/usr/bin/env python3
"""NEW physical-alpha scientific forks; never the ultraweak baseline law.

Native Ref0 controls retain positive two-site Wilson/Haar-reference transfer,
primitive Floquet residues and exact CAR/current experiments. NEW macro RefQ
probes use all-label kinetic-momentum torons, literal integer Coulomb flows,
actual Wilson determinant jets/local all-grade conditionals and causal signed poles.
Finite recoil secant masses are labelled separately from infinitesimal curvature.
Each capsule distinguishes executed quantum controls, conditional reference
probabilities and the still-unpaid full interacting source/higher-grade debt.
The selected Q/sqrt(beta) is BARE charge; photon/current normalization is needed
before calling any consumer coefficient a measured physical/Thomson alpha.
Neutral Ref4 unitary charge conjugation is a control, not a charged Ref3 symmetry.
The explicit --cartesian-c born/native branches declare a NEW hC/Aeven potential,
with physical-component matrix-log currents/contacts, analytic ALL60/site Phi
first/mixed/quadratic-norm jets and scalar Wilson-path jets. The declared
positive completion is exponentially quasilocal, not spatially ultralocal.
Cartesian outside Phi/Fock couplings are linear and separately C-completed.
Native runs use an amplified ONE normal oscillator and induced-Gaussian photons;
they do not certify full60/SU4/all-Naux source admission or measured QED.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import hashlib
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import types
from typing import Any, cast

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy
import mpmath as mp
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.integrate import quad
from scipy.special import ive
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {name: (ROOT / name).read_bytes() for name in
                ('physical_alpha_compton.py', 'sea_selection.py', 'rotation_band.py')}
SOURCE_HASHES = {name: hashlib.sha256(value).hexdigest() for name, value in SOURCE_BYTES.items()}
from sea_selection import SELECTED, matrix_unit_basis


@dataclass(frozen=True)
class Parameters:
    inverse_alpha: str = '137.035999084'
    beta_power: int = 150
    kappa_power: int = 250
    mass: float = 1.
    spacing: float = .01
    orders: tuple[int, ...] = (3, 3, 4)
    toron_cutoffs: tuple[int, ...] = (1, 3, 5)
    observation_times: tuple[float, ...] = (.25, .5, 1., 2.)
    toron_packet_width: float = .4
    dense_limit: int = 2000
    flux_fraction: str = '0'


def charge_contract(p: Parameters) -> dict[str, Any]:
    # 110 decimal digits resolve the enormous integer choice, not merely e.
    with localcontext() as context:
        context.prec = 110
        pi = Decimal('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651')
        beta = Decimal(10) ** p.beta_power
        target = (4*pi*beta/Decimal(p.inverse_alpha)).sqrt()
        q = int(target.to_integral_value())
        alpha = Decimal(q*q)/(4*pi*beta)
        e = Decimal(q)/beta.sqrt()
        rounding = abs(alpha*Decimal(p.inverse_alpha)-1)
    naux = 10**(p.beta_power+4*p.kappa_power)//16
    volume = 2**72
    with mp.workdps(180):
        theta = mp.mpf('.23')/mp.sqrt(mp.mpf(str(beta)))
        link = mp.exp(mp.j*theta)
        active = link**q
        expected = mp.exp(mp.j*mp.mpf(str(e))*mp.mpf('.23'))
        character_error = str(abs(active-expected))
        periodic_error = str(abs(mp.exp(mp.j*(theta+2*mp.pi))**q-active))
        aux_phase = str(abs(link-1))
    kappa = 1/(2*(4+math.expm1(p.spacing*p.mass)))
    return dict(Q=str(q), auxiliary_charge=1, reference_primitive_charge=0,
                bare_e=float(e), bare_alpha=str(alpha), relative_integer_rounding=str(rounding),
                charge_definition='e0=Q/sqrt(beta) is the CHOSEN canonically normalized BARE charge, not measured eR. Integer rounding certifies only bare alpha0.',
                inverse_alpha_parameter_meaning='Chosen bare inverse alpha0; physical Thomson charge requires the SAME photon residue and Ward-normalized current.',
                Naux=str(naux), SU4_species_count=str(naux+1), active_filled_vacuum_count=1,
                beta=str(beta), selected_integer_charge_is_input=True,
                charges='rho=Q(n_active-2)+sum_aux(n_aux-2); Ref contributes zero. Source has +Q and needs a compensating -Q charge and electric-flux dressing.',
                background_active_per_site=str(2*q), auxiliary_background_per_site=2,
                neutralizing_background='Charged sector has integer -Q static compensator at site1. Integer x-link flux E0-E1=Q(n0-2) gives exact Gauss; Ref remains charge0. Photon projection does not replace this charge ledger.',
                exact_link='U=exp(i theta), theta=A/sqrt(beta); active U**Q=exp(i Q theta), aux U. No noninteger compact character.',
                literal_compact_character_probe=dict(decimal_digits=180,
                    primitive_theta=mp.nstr(theta, 12), active_power_error=character_error,
                    integer_periodicity_error=periodic_error, auxiliary_link_change=aux_phase),
                active_kappa=kappa, full_Wilson_B_lower=1-6*kappa,
                Ref_modes=4, Ref_initial_occupation=4, Ref_after_source_occupation=3,
                inherited_chart_Q_delta=float(e)*10**(p.beta_power/2-50),
                inherited_global_state_budget=32*4*volume*2**20*math.log(1+8/math.expm1(p.spacing*p.mass)),
                inherited_bounds_rejected=True,
                actual_compact_typical_phase_scale=float(e),
                microscopic_parameters_are_inputs=['Grassmann/Haar quantum law','Q','m0=expm1(a mass)','a','Naux','kaux','selected Clifford/Phi frame'])


def gamma_frame() -> np.ndarray:
    beta, alpha = SELECTED[0], SELECTED[1:]
    gamma = np.array([beta] + [1j*beta@x for x in alpha])
    if max(np.linalg.norm(gamma[i]@gamma[j]+gamma[j]@gamma[i]
                          -(2*np.eye(4) if i == j else 0))
           for i in range(4) for j in range(4)) > 1e-12:
        raise ArithmeticError('Captured frame is not Clifford')
    return gamma


def clifford_time_reversal(gamma: np.ndarray) -> np.ndarray:
    """Solve the actual four-component quaternionic intertwiner, in SU4."""
    constraints = np.vstack([np.kron(g, np.eye(4))-np.kron(np.eye(4), g.conj().T) for g in gamma])
    _, _, vectors = np.linalg.svd(constraints)
    left, _, right = np.linalg.svd(vectors[-1].conj().reshape(4, 4))
    intertwiner = left@right
    intertwiner *= np.exp(-.25j*np.angle(np.linalg.det(intertwiner)))
    if max(np.linalg.norm(intertwiner@g.conj()@intertwiner.conj().T-g) for g in gamma) > 2e-10:
        raise ArithmeticError('Actual Clifford time-reversal intertwiner failed')
    return intertwiner


def spatial_blocks(gamma: np.ndarray, p: Parameters, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Actual two-site Wilson links, including transverse self-links."""
    b = (1+math.expm1(p.spacing*p.mass)+3)*np.eye(8, dtype=complex)
    c = np.zeros((8, 8), complex)
    for x in range(2):
        for j in range(3):
            y = 1-x if j == 0 else x
            sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
            link = np.exp(1j*phases[x, j])
            b[sx, sy] -= .5*link*np.eye(4)
            b[sy, sx] -= .5*link.conjugate()*np.eye(4)
            c[sx, sy] += .5*link*gamma[j+1]
            c[sy, sx] -= .5*link.conjugate()*gamma[j+1]
    return b, c


def one_transfer(gamma: np.ndarray, p: Parameters, phases: np.ndarray) -> tuple[np.ndarray, float]:
    b, c = spatial_blocks(gamma, p, phases)
    _, basis = np.linalg.eigh(gamma[0])
    plus = np.kron(np.eye(2), basis[:, 2:])
    minus = np.kron(np.eye(2), basis[:, :2])
    bp, bm = plus.conj().T@b@plus, minus.conj().T@b@minus
    f = plus.conj().T@c@minus
    l = np.block([[np.eye(4), -f], [np.zeros((4, 4)), np.eye(4)]])
    t = l.conj().T@scipy.linalg.block_diag(np.linalg.inv(bp), bm)@l
    if np.linalg.eigvalsh(t)[0] <= 0 or np.linalg.eigvalsh(bp)[0] <= 0:
        raise ArithmeticError('Wilson transfer lost positivity')
    return t, float(np.linalg.slogdet(bp)[1])


def matrix_log_jet(matrix: np.ndarray, first: np.ndarray, second: np.ndarray,
                   mixed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Actual Frechet log derivatives; ordered blocks retain BOTH contact terms."""
    values, frame = eigh(matrix)
    if values[0] <= 0:
        raise ArithmeticError('Completed-law logarithm requires a positive matrix')
    logarithm = (frame*np.log(values))@frame.conj().T
    size, zero = len(matrix), np.zeros_like(matrix)
    def derivative(direction):
        block = np.block([[matrix, direction], [zero, matrix]])
        return scipy.linalg.logm(block)[:size, size:]
    def ordered(left, right):
        block = np.block([[matrix, left, zero], [zero, matrix, right],
                          [zero, zero, matrix]])
        return scipy.linalg.logm(block)[:size, 2*size:]
    dx, dy = derivative(first), derivative(second)
    dxy = derivative(mixed)+ordered(first, second)+ordered(second, first)
    outputs = (logarithm, dx, dy, dxy)
    if max(np.linalg.norm(value-value.conj().T) for value in outputs) > 2e-9:
        raise ArithmeticError('Frechet logarithm lost Hermitian derivative')
    return tuple((value+value.conj().T)/2 for value in outputs)


class CartesianCWilson:
    """NEW log-completed law, in physical SITE-MAJOR four-component coordinates.

    This replaces the non-Clifford potential, not Root931's connection or law.
    Targets describe a finite periodic spatial graph; no auxiliary multiplicity
    is allocated. Phi normal coordinates and SU4 links remain independent inputs.
    """
    def __init__(self, gamma: np.ndarray, parameters: Parameters, targets: np.ndarray):
        self.gamma, self.parameters = gamma, parameters
        self.targets = np.asarray(targets, dtype=int)
        self.sites = len(self.targets)
        if self.targets.shape != (self.sites, 3) or any(
                sorted(self.targets[:, j]) != list(range(self.sites)) for j in range(3)):
            raise ValueError('Each periodic target direction must be a permutation')
        _, beta_frame = np.linalg.eigh(gamma[0])
        self.plus = np.kron(np.eye(self.sites), beta_frame[:, 2:])
        self.minus = np.kron(np.eye(self.sites), beta_frame[:, :2])
        self.change = np.column_stack((self.plus, self.minus))
        spin_R = -1j*gamma[1]@gamma[2]@gamma[3]
        self.CH = spin_R@clifford_time_reversal(gamma)
        self.C = np.kron(np.eye(self.sites), self.CH)
        self.kappa = 1/(2*(4+math.expm1(parameters.spacing*parameters.mass)))
        self.B_global_lower = 1/(2*self.kappa)-3-self.kappa/4
        if self.B_global_lower <= 0:
            raise ValueError('This completed candidate requires globally positive B')
        if np.linalg.norm(self.CH@self.CH.conj()-np.eye(4)) > 2e-10:
            raise ArithmeticError('Cartesian C must be an actual involution')

    def _reflect_fields(self, fields: np.ndarray) -> np.ndarray:
        reflected = np.einsum('ab,xmbc,cd->xmad', self.CH,
                              fields.transpose(0, 1, 3, 2), self.CH.conj().T)
        reflected[:, 0] *= -1
        return reflected

    def reflect(self, remainder: np.ndarray, links: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        colour = np.einsum('ab,xjbc,cd->xjad', self.CH, links.conj(), self.CH.conj().T)
        return self._reflect_fields(remainder), colour

    def _base(self, phases: np.ndarray, remainder: np.ndarray, links: np.ndarray,
              directions: tuple[np.ndarray, np.ndarray] | None = None,
              phi_directions: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
        sites, size, kappa = self.sites, 4*self.sites, self.kappa
        if remainder.shape != (sites, 4, 4, 4) or links.shape != (sites, 3, 4, 4):
            raise ValueError('Full four-component Phi and three SU4 links are required at EACH site')
        if max(np.linalg.norm(y-y.conj().T) for y in remainder.reshape(-1, 4, 4)) > 2e-10:
            raise ValueError('Cartesian Phi coordinates must be Hermitian')
        if max(max(np.linalg.norm(u.conj().T@u-np.eye(4)), abs(np.linalg.det(u)-1))
               for u in links.reshape(-1, 4, 4)) > 2e-10:
            raise ValueError('Colour coordinates must be actual SU4 matrices')
        orders = ((0, 0),) if directions is None else ((0, 0), (1, 0), (0, 1), (1, 1))
        blocks = []
        for first, second in orders:
            b, c = np.zeros((size, size), complex), np.zeros((size, size), complex)
            if first+second == 0:
                for x in range(sites):
                    sx = slice(4*x, 4*x+4)
                    b[sx, sx] = (1/(2*kappa)+kappa*np.linalg.norm(remainder[x])**2)*np.eye(4)+kappa*remainder[x, 0]
                    c[sx, sx] = kappa*remainder[x, 0]
            elif phi_directions is not None:
                for x in range(sites):
                    sx = slice(4*x, 4*x+4)
                    if first+second == 1:
                        delta = phi_directions[second][x]
                        b[sx, sx] = kappa*(delta[0]+2*np.vdot(remainder[x], delta).real*np.eye(4))
                        c[sx, sx] = kappa*delta[0]
                    else:
                        b[sx, sx] = 2*kappa*np.vdot(phi_directions[0][x], phi_directions[1][x]).real*np.eye(4)
            for j in range(3):
                u = np.zeros((size, size), complex)
                g = scipy.linalg.block_diag(*(self.gamma[j+1]+kappa*remainder[x, j+1] for x in range(sites)))
                for x in range(sites):
                    factor = np.exp(1j*phases[x, j])
                    if directions is not None:
                        factor *= (1j*directions[0][x, j])**first*(1j*directions[1][x, j])**second
                    y = self.targets[x, j]
                    u[4*x:4*x+4, 4*y:4*y+4] = factor*links[x, j]
                b -= (u+u.conj().T)/2
                c += (g@u-u.conj().T@g)/2
                if phi_directions is not None:
                    for delta, left, right in (
                            ((phi_directions[0], first-1, second),) if first else ()) + (
                            ((phi_directions[1], first, second-1),) if second else ()):
                        assert directions is not None
                        dg = scipy.linalg.block_diag(*(kappa*delta[x, j+1] for x in range(sites)))
                        du = np.zeros_like(u)
                        for x in range(sites):
                            factor = np.exp(1j*phases[x, j])
                            factor *= (1j*directions[0][x, j])**left*(1j*directions[1][x, j])**right
                            y = self.targets[x, j]
                            du[4*x:4*x+4, 4*y:4*y+4] = factor*links[x, j]
                        c += (dg@du-du.conj().T@dg)/2
            blocks.append((self.plus.conj().T@b@self.plus,
                           self.minus.conj().T@b@self.minus,
                           self.plus.conj().T@c@self.minus))
        bp, bm, f = blocks[0]
        if min(np.linalg.eigvalsh(bp)[0], np.linalg.eigvalsh(bm)[0]) <= 0:
            raise ArithmeticError('Declared positive Phi completion lost B')
        inverse, other_inverse = np.linalg.inv(bp), np.linalg.inv(bm)
        zero, identity = np.zeros_like(f), np.eye(len(f))
        l = np.block([[identity, -f], [zero, identity]])
        d = scipy.linalg.block_diag(inverse, bm)
        t = self.change@(l.conj().T@d@l)@self.change.conj().T
        scalar = .5*(np.linalg.slogdet(bp)[1]+np.linalg.slogdet(bm)[1])
        if directions is None:
            values, vectors = eigh(t)
            if values[0] <= 0:
                raise ArithmeticError('Base positive transfer has unresolved spectrum')
            return dict(h=(vectors*np.log(values))@vectors.conj().T, scalar=float(scalar), T=t)
        jets, scalar_jets = [], []
        for dbp, dbm, df in blocks[1:3]:
            dl = np.block([[zero, -df], [zero, zero]])
            dd = scipy.linalg.block_diag(-inverse@dbp@inverse, dbm)
            dt = dl.conj().T@d@l+l.conj().T@dd@l+l.conj().T@d@dl
            jets.append((dl, dd, self.change@dt@self.change.conj().T))
            scalar_jets.append(.5*np.trace(inverse@dbp+other_inverse@dbm).real)
        dxbp, dxbm, _ = blocks[1]
        dybp, dybm, _ = blocks[2]
        dbp, dbm, df = blocks[3]
        dl = np.block([[zero, -df], [zero, zero]])
        dd = scipy.linalg.block_diag(inverse@dxbp@inverse@dybp@inverse
             +inverse@dybp@inverse@dxbp@inverse-inverse@dbp@inverse, dbm)
        lx, dx, tx = jets[0]
        ly, dy, ty = jets[1]
        mixed = (dl.conj().T@d@l+l.conj().T@d@dl+l.conj().T@dd@l
                 +lx.conj().T@dy@l+ly.conj().T@dx@l
                 +l.conj().T@dx@ly+l.conj().T@dy@lx
                 +lx.conj().T@d@ly+ly.conj().T@d@lx)
        logarithm, hx, hy, hxy = matrix_log_jet(t, tx, ty, self.change@mixed@self.change.conj().T)
        scalar_xy = .5*np.trace(inverse@dbp-inverse@dybp@inverse@dxbp
                     +other_inverse@dbm-other_inverse@dybm@other_inverse@dxbm).real
        return dict(h=logarithm, scalar=float(scalar), hx=hx, hy=hy, hxy=hxy,
                    scalar_x=float(scalar_jets[0]), scalar_y=float(scalar_jets[1]),
                    scalar_xy=float(scalar_xy), T=t)

    def potential(self, phases: np.ndarray, remainder: np.ndarray, links: np.ndarray,
                  directions: tuple[np.ndarray, np.ndarray] | None = None,
                  phi_directions: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
        """Own analytic Peierls/full60-Phi first and mixed matrix-log jets."""
        if phi_directions is not None and directions is None:
            directions = (np.zeros_like(phases), np.zeros_like(phases))
        reflected, reflected_links = self.reflect(remainder, links)
        reversed_directions = None if directions is None else (-directions[0], -directions[1])
        reversed_fields = None if phi_directions is None else (
            self._reflect_fields(phi_directions[0]), self._reflect_fields(phi_directions[1]))
        original = self._base(phases, remainder, links, directions, phi_directions)
        reverse = self._base(-phases, reflected, reflected_links, reversed_directions, reversed_fields)
        result = {}
        for suffix in ('',) if directions is None else ('', 'x', 'y', 'xy'):
            hkey = 'h'+suffix
            skey = 'scalar'+('_'+suffix if suffix else '')
            h = .5*(original[hkey]-self.C@reverse[hkey].T@self.C.conj().T)
            h = (h+h.conj().T)/2
            result[hkey] = h
            result['pref'+suffix] = float(.5*(original[skey]+reverse[skey])-.5*np.trace(h).real)
        return result

    def transfer(self, phases: np.ndarray, remainder: np.ndarray,
                 links: np.ndarray) -> tuple[np.ndarray, float]:
        potential = self.potential(phases, remainder, links)
        values, frame = eigh(potential['h'])
        return (frame*np.exp(values))@frame.conj().T, potential['pref']

    def locality_bound(self, radius: float) -> dict[str, float]:
        """Combes weighted-resolvent bound on log T, NOT strict ultralocality."""
        bmin = self.B_global_lower
        bmax = 1/(2*self.kappa)+3+self.kappa*(radius+radius*radius)
        fmax = 3+4*self.kappa*radius
        lmax, dmax = 1+fmax, max(1/bmin, bmax)
        tmin, tmax = min(bmin, 1/bmax)/lmax**2, dmax*lmax**2
        eta = .1
        while True:
            db = 3*math.expm1(eta)
            df = 3*(1+self.kappa*radius)*math.expm1(eta)
            if db < bmin:
                dd = max(db, db/(bmin*(bmin-db)))
                dt = (2*lmax*df+df*df)*dmax+(lmax+df)**2*dd
                if dt <= tmin/2:
                    break
            eta /= 2
        return dict(B_lower=bmin, B_upper=bmax, F_norm_upper=fmax,
                    T_lower=tmin, T_upper=tmax, exponential_rate=eta,
                    log_kernel_prefactor=max(abs(math.log(tmin)), abs(math.log(tmax)))+math.log(2),
                    weighted_T_perturbation_upper=dt)

    def outside_potential(self, phases: np.ndarray, phi: np.ndarray,
                          links: np.ndarray, couplings: tuple[float, float, float],
                          directions: tuple[np.ndarray, np.ndarray] | None = None,
                          phi_directions: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
        """NEW all-Cartesian LINEAR Phi/Fock outside Hamiltonian and its jets.

        hxy=(-w Phi0(x)-i v Phi_j(x)) Uxy/2, plus its adjoint;
        onsite=m Phi0. The normal-ordered Fock lift is dGamma(h)-Tr(h)/2.
        Positive auxiliary E n, if present, C-completes to the actual constant
        2 E per site/species; it is not silently dropped or retagged.
        """
        if phi_directions is not None and directions is None:
            directions = (np.zeros_like(phases), np.zeros_like(phases))
        orders = ((0, 0),) if directions is None else ((0, 0), (1, 0), (0, 1), (1, 1))
        def base(a, fields, colour, tangent, field_tangent):
            outputs = []
            mass, wilson, current = couplings
            for first, second in orders:
                h = np.zeros((4*self.sites, 4*self.sites), complex)
                if first+second == 0:
                    for x in range(self.sites):
                        h[4*x:4*x+4, 4*x:4*x+4] = mass*fields[x, 0]
                elif first+second == 1 and field_tangent is not None:
                    for x in range(self.sites):
                        h[4*x:4*x+4, 4*x:4*x+4] = mass*field_tangent[second][x, 0]
                for x in range(self.sites):
                    sx = slice(4*x, 4*x+4)
                    for j in range(3):
                        y = self.targets[x, j]
                        sy = slice(4*y, 4*y+4)
                        factor = np.exp(1j*a[x, j])
                        if tangent is not None:
                            factor *= (1j*tangent[0][x, j])**first*(1j*tangent[1][x, j])**second
                        hop = (-wilson*fields[x, 0]-1j*current*fields[x, j+1])@colour[x, j]*factor/2
                        if field_tangent is not None:
                            for delta, left, right in (
                                    ((field_tangent[0], first-1, second),) if first else ()) + (
                                    ((field_tangent[1], first, second-1),) if second else ()):
                                assert tangent is not None
                                factor = np.exp(1j*a[x, j])
                                factor *= (1j*tangent[0][x, j])**left*(1j*tangent[1][x, j])**right
                                hop += (-wilson*delta[x, 0]-1j*current*delta[x, j+1])@colour[x, j]*factor/2
                        h[sx, sy] += hop
                        h[sy, sx] += hop.conj().T
                outputs.append(h)
            return outputs
        reflected, reflected_links = self.reflect(phi, links)
        reversed_directions = None if directions is None else (-directions[0], -directions[1])
        reversed_fields = None if phi_directions is None else (
            self._reflect_fields(phi_directions[0]), self._reflect_fields(phi_directions[1]))
        original = base(phases, phi, links, directions, phi_directions)
        reverse = base(-phases, reflected, reflected_links, reversed_directions, reversed_fields)
        result = {}
        for suffix, h, reflected_h in zip(
                ('',) if directions is None else ('', 'x', 'y', 'xy'), original, reverse, strict=True):
            completed = .5*(h-self.C@reflected_h.T@self.C.conj().T)
            result['H'+suffix] = (completed+completed.conj().T)/2
            result['offset'+suffix] = -.5*float(np.trace(completed).real)
        return result

    def source_packet_jet(self, phases: np.ndarray, links: np.ndarray,
                          directions: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
        """Actual native Wilson source path 0->Ref1, including its Peierls jets."""
        if self.sites != 2 or self.targets[0, 0] != 1:
            raise ValueError('This explicit scalar source path is the native 0->Ref1 edge')
        w = np.exp(1j*phases[0, 0])*links[0, 0]
        zero = np.zeros_like(w)
        x, y = directions[0][0, 0], directions[1][0, 0]
        return dict(psi=np.vstack((w, np.eye(4)))/2,
                    psi_x=np.vstack((1j*x*w, zero))/2,
                    psi_y=np.vstack((1j*y*w, zero))/2,
                    psi_xy=np.vstack((-x*y*w, zero))/2)


def cartesian_C_normal_geometry(law: CartesianCWilson,
                               phi_reference: np.ndarray) -> dict[str, Any]:
    """Actual sixty real coordinates and ALL forty-five normal directions."""
    basis = matrix_unit_basis()/2
    coordinates = np.zeros((60, 4, 4, 4), complex)
    for mu in range(4):
        coordinates[15*mu:15*(mu+1), mu] = basis
    tangent = np.array([[1j*(a@phi-phi@a) for phi in phi_reference] for a in basis])
    orbit = np.einsum('amij,bmji->ab', coordinates, tangent).real
    left, singular, _ = np.linalg.svd(orbit, full_matrices=True)
    if np.sum(singular > 1e-10) != 15:
        raise ArithmeticError('Reference Phi does not have the selected SU4 orbit')
    reflected = np.einsum('ij,amjk,kl->amil', law.CH,
                          coordinates.transpose(0, 1, 3, 2), law.CH.conj().T)
    reflected[:, 0] *= -1
    jacobian = np.einsum('amij,bmji->ab', coordinates, reflected).real
    normal = left[:, 15:]
    trial = normal@np.sin(.31*np.arange(1, 46))
    odd = trial-jacobian@trial
    odd /= np.linalg.norm(odd)
    direction = np.einsum('a,amij->mij', odd, coordinates)
    phi_image = np.einsum('ij,mjk,kl->mil', law.CH,
                         phi_reference.transpose(0, 2, 1), law.CH.conj().T)
    phi_image[0] *= -1
    return dict(direction=direction, cartesian_jacobian=jacobian,
        normal_jacobian=normal.T@jacobian@normal,
        normal_to_orbit_defect=float(np.linalg.norm(orbit.T@normal)),
        normal_descent_defect=float(np.linalg.norm((np.eye(60)-normal@normal.T)@jacobian@normal)),
        Cartesian_isometry_defect=float(np.linalg.norm(jacobian.T@jacobian-np.eye(60))),
        Cartesian_involution_defect=float(np.linalg.norm(jacobian@jacobian-np.eye(60))),
        selected_reference_fixed_defect=float(np.linalg.norm(phi_image-phi_reference)))


def matrix_exp_jet(generator: np.ndarray, first: np.ndarray, second: np.ndarray,
                   mixed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exponentiate the NEW Fock generator and its actual mixed contact."""
    size, zero = len(generator), np.zeros_like(generator)
    def ordered(left, right):
        block = np.block([[generator, left, zero], [zero, generator, right],
                          [zero, zero, generator]])
        return scipy.linalg.expm(block)[:size, 2*size:]
    value = scipy.linalg.expm(generator)
    dx = scipy.linalg.expm_frechet(generator, first, compute_expm=False)
    dy = scipy.linalg.expm_frechet(generator, second, compute_expm=False)
    dxy = (scipy.linalg.expm_frechet(generator, mixed, compute_expm=False)
           +ordered(first, second)+ordered(second, first))
    return tuple((a+a.conj().T)/2 for a in (value, dx, dy, dxy))


def cartesian_C_carrier(law: CartesianCWilson, number: int, bare_e: float,
                        normal_order: int, normal_width: float,
                        normal_direction: np.ndarray) -> dict[str, Any]:
    """Complete native Phi/toron carrier; currents are derivatives of NEW log T.

    Coexact photons are decoupled ONLY at the explicit Born reference lambda=0.
    The normal oscillator and Coulomb/compact winding remain quantum/interacting.
    No free-band sea, fitted mass, old Dirac vertex or old contact is imported.
    """
    nodes, transform, _ = oscillator(normal_order, 1.)
    nodes *= normal_width*math.sqrt(2*math.sinh(1.))
    half = transform.T@np.diag(np.exp(-.5*np.arange(normal_order)))@transform
    torons, angles = np.arange(-1, 2), 2*np.pi*np.arange(24)/24
    characters = np.exp(-1j*np.outer(torons, angles))/math.sqrt(24)
    _, masks = second_quantized(np.eye(8), number)
    size = len(masks)
    electric_half = np.exp(-.5*integer_electric_energy(masks, bare_e, torons, 0.))
    local = {key: [] for key in ('T', 'Ty', 'Tz', 'Tyz')}
    dy, dz = np.zeros((2, 3)), np.zeros((2, 3))
    dy[:, 1] = bare_e*np.array([1., -1.])/math.sqrt(2)
    dz[:, 2] = bare_e*np.array([1., -1.])/math.sqrt(2)
    links = np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy()
    minimum = math.inf
    for node in nodes:
        blocks = {key: [] for key in local}
        remainder = node*np.broadcast_to(normal_direction, (2, 4, 4, 4))
        for angle in angles:
            phases = np.zeros((2, 3));phases[1, 0] = angle
            potential = law.potential(phases, remainder, links, (dy, dz))
            generators = []
            for suffix in ('', 'x', 'y', 'xy'):
                h = law.change.conj().T@potential['h'+suffix]@law.change
                generator, _ = second_quantized(h, number)
                generator += potential['pref'+suffix]*np.eye(size)
                generators.append(generator)
            value, first, second, mixed = matrix_exp_jet(*generators)
            minimum = min(minimum, float(np.linalg.eigvalsh(value)[0]))
            for key, block in zip(blocks, (value, first, second, mixed), strict=True):
                blocks[key].append(block)
        for key in local:
            local[key].append(blocks[key])
    dimension = normal_order*3*size
    matrices = {}
    for key in local:
        kernel_blocks = np.array(local[key])
        def apply(argument):
            state = argument.reshape(normal_order, 3, size, -1)
            state = np.einsum('ab,bmfc->amfc', half, state, optimize=True)
            state *= electric_half[None, :, :, None]
            angular = np.einsum('mq,amfc->aqfc', characters.conj(), state, optimize=True)
            angular = np.einsum('aqfg,aqgc->aqfc', kernel_blocks, angular, optimize=True)
            state = np.einsum('mq,aqfc->amfc', characters, angular, optimize=True)
            state *= electric_half[None, :, :, None]
            state = np.einsum('ab,bmfc->amfc', half, state, optimize=True)
            return state.reshape(dimension, -1)
        matrix = np.empty((dimension, dimension), complex)
        for start in range(0, dimension, 32):
            stop = min(dimension, start+32)
            columns = np.zeros((dimension, stop-start))
            columns[np.arange(start, stop), np.arange(stop-start)] = 1.
            matrix[:, start:stop] = apply(columns)
        matrices[key] = (matrix+matrix.conj().T)/2
    eigenvalues, eigenvectors = eigh(matrices['T'])
    if eigenvalues[0] <= 0:
        raise ArithmeticError('Actual full native Phi/toron carrier lost positivity')
    matrices.update(values=eigenvalues, vectors=eigenvectors, masks=masks,
                    normal_nodes=nodes, normal_transform=transform,
                    dimension=dimension, lower=minimum*math.exp(
                        -(normal_order-1)-float(np.max(-2*np.log(electric_half)))))
    return matrices


def log_divided_first(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Confluent log divided difference, without subtracting near-equal logs."""
    a, b = np.broadcast_arrays(a, b)
    difference = a-b
    return np.divide(np.log1p(difference/b), difference, out=1/b,
                     where=difference != 0)


def log_divided_second(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Symmetric second difference; near confluent pair has relative error<3e-11."""
    a, b, c = np.broadcast_arrays(a, b, c)
    centre = (a+c)/2
    close = abs(a-c) <= 1e-5*centre
    result = np.empty_like(centre)
    result[~close] = ((log_divided_first(a[~close], b[~close])
                      -log_divided_first(b[~close], c[~close]))/(a[~close]-c[~close]))
    ratio = (b[close]-centre[close])/centre[close]
    value = np.empty_like(ratio)
    small = abs(ratio) < .01
    value[small] = sum((-1.)**(k+1)*ratio[small]**k/(k+2) for k in range(13))
    value[~small] = (np.log1p(ratio[~small])-ratio[~small])/ratio[~small]**2
    result[close] = value/centre[close]**2
    return result


def cartesian_C_quantum_vertices(carrier: dict[str, Any], spacing: float) -> dict[str, np.ndarray]:
    """Frechet -log of the FULL photon/carrier transfer, not one-body forwarding."""
    values, frame = carrier['values'], carrier['vectors']
    ky, kz, kyz = [frame.conj().T@carrier[key]@frame for key in ('Ty', 'Tz', 'Tyz')]
    omega = 2*math.asinh(1.)
    qnorm = math.sqrt(2*math.sinh(omega))
    a, c = values[:, None], values[None, :]
    raising_factor = -math.exp(-omega/2)*log_divided_first(math.exp(-omega)*a, c)/(spacing*qnorm)
    vy, vz = raising_factor*ky, raising_factor*kz
    contact = log_divided_first(a, c)*kyz
    for k, value in enumerate(values):
        contact += math.exp(omega)*log_divided_second(a, math.exp(omega)*value, c)*np.outer(kz[:, k], ky[k])
        contact += math.exp(-omega)*log_divided_second(a, math.exp(-omega)*value, c)*np.outer(ky[:, k], kz[k])
    contact *= -1/(spacing*qnorm*qnorm)
    return dict(raise_y=vy, raise_z=vz, contact_zy=contact,
                matter_energies=-np.log(values)/spacing,
                photon_frequency=np.array(omega/spacing))


def ordered_time_integral(a: np.ndarray, b: np.ndarray, time_value: float) -> np.ndarray:
    """Entire finite-time Dyson integral; no energy-delta or resonance division."""
    a, b = np.broadcast_arrays(a, b)
    def first(value):
        return time_value*np.exp(.5j*value*time_value)*np.sinc(value*time_value/(2*math.pi))
    close = abs(b*time_value) < 1e-4
    out = np.empty(a.shape, complex)
    out[~close] = (first(a[~close]+b[~close])-first(a[~close]))/(1j*b[~close])
    z = a[close]*time_value
    moments = []
    for order in (1, 2, 3):
        value = np.empty(z.shape, complex)
        small = abs(z) < .5
        value[small] = sum((1j*z[small])**k/(math.factorial(k)*(order+k+1)) for k in range(21))
        large = z[~small]
        integral = np.exp(.5j*large)*np.sinc(large/(2*math.pi))
        for degree in range(1, order+1):
            integral = (np.exp(1j*large)-degree*integral)/(1j*large)
        value[~small] = integral
        moments.append(value)
    bt = b[close]*time_value
    out[close] = time_value**2*(moments[0]+.5j*bt*moments[1]-bt*bt*moments[2]/6)
    return out


def cartesian_C_Born_matrix(vertices: dict[str, np.ndarray], time_value: float) -> dict[str, np.ndarray]:
    """ALL carrier intermediate eigenstates in particle/temporal contact words."""
    energy = vertices['matter_energies']
    omega = float(vertices['photon_frequency'])
    external = energy+omega
    vy, vz = vertices['raise_y'], vertices['raise_z']
    phase = np.exp(-1j*external*time_value)[:, None]
    difference = external[:, None]-external[None, :]
    contact = (-1j*phase*time_value*np.exp(.5j*difference*time_value)
               *np.sinc(difference*time_value/(2*math.pi))*vertices['contact_zy'])
    s, u = np.zeros_like(contact), np.zeros_like(contact)
    for k, middle in enumerate(energy):
        a, b = external[:, None]-middle, middle-external[None, :]
        s -= phase*np.outer(vz[:, k], vy.conj().T[k])*ordered_time_integral(a, b, time_value)
        middle += 2*omega
        a, b = external[:, None]-middle, middle-external[None, :]
        u -= phase*np.outer(vy.conj().T[:, k], vz[k])*ordered_time_integral(a, b, time_value)
    return dict(contact=contact, s=s, u=u, total=contact+s+u)


def cartesian_C_source_Born(p: Parameters, contract: dict[str, Any],
                            critical: dict[str, Any], phi_reference: np.ndarray,
                            normal_order: int = 2, normal_width: float = 5e-6,
                            times: tuple[float, ...] = (.005, .02, .05)) -> dict[str, Any]:
    """Execute the NEW coupled Phi/toron source and actual finite connected Born."""
    parameters = Parameters(**dict(asdict(p), mass=critical['declared_critical_bare_mass_parameter']))
    law = CartesianCWilson(gamma_frame(), parameters, np.array([[1, 0, 0], [0, 1, 1]]))
    geometry = cartesian_C_normal_geometry(law, phi_reference)
    carriers = {number: cartesian_C_carrier(law, number, contract['bare_e'],
                normal_order, normal_width, geometry['direction']) for number in (4, 5)}
    vacuum, charged = carriers[4], carriers[5]
    ground = vacuum['vectors'][:, -1]
    reference = vacuum['values'][-1]
    lookup = {mask: i for i, mask in enumerate(charged['masks'])}
    path_source = law.source_packet_jet(np.zeros((2, 3)),
        np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy(), (np.zeros((2, 3)), np.zeros((2, 3))))
    spatial = law.change.conj().T@path_source['psi']
    sources = np.zeros((4, 56, 70), complex)
    for column, mask in enumerate(vacuum['masks']):
        for bit in range(8):
            if not mask & (1 << bit):
                sources[:, lookup[mask | (1 << bit)], column] += (
                    (-1)**((mask & ((1 << bit)-1)).bit_count())*spatial[bit])
    acted = np.einsum('rfg,amg->amfr', sources, ground.reshape(normal_order, 3, 70)).reshape(charged['dimension'], 4)
    raw = float(np.linalg.norm(acted)**2)
    frame = charged['vectors']
    source_half = frame@(np.sqrt(charged['values']/reference)[:, None]*(frame.conj().T@acted))
    half_gram = float(np.linalg.norm(source_half)**2)
    incoming = source_half/math.sqrt(half_gram)
    gram = source_half.conj().T@source_half
    gram_values, gram_frame = eigh(gram)
    if gram_values[0] <= 0:
        raise ArithmeticError('Actual all-four-spin receiver Gram is not positive')
    receiver = source_half@(gram_frame/np.sqrt(gram_values))
    receiver_defect = float(np.linalg.norm(receiver.conj().T@receiver-np.eye(4)))
    incoming_eigen = frame.conj().T@incoming
    receiver_eigen = frame.conj().T@receiver
    vertices = {number: cartesian_C_quantum_vertices(carriers[number], p.spacing) for number in (4, 5)}
    for value in vertices.values():
        value['matter_energies'] += math.log(reference)/p.spacing
    source_weights = np.sum(abs(incoming_eigen)**2, axis=1)
    source_mean = float(source_weights@vertices[5]['matter_energies'])
    source_variance = float(source_weights@((vertices[5]['matter_energies']-source_mean)**2))
    rows = []
    for observation in times:
        words = cartesian_C_Born_matrix(vertices[5], observation)
        vacuum_words = cartesian_C_Born_matrix(vertices[4], observation)
        vacuum_amplitude = vacuum_words['total'][-1, -1]
        vacuum_ratio = vacuum_amplitude*np.exp(1j*float(vertices[4]['photon_frequency'])*observation)
        propagation = np.exp(-1j*(vertices[5]['matter_energies']
                                    +vertices[5]['photon_frequency'])*observation)
        spectator = receiver_eigen.conj().T@(propagation[:, None]*incoming_eigen)
        amplitudes = {name: receiver_eigen.conj().T@word@incoming_eigen for name, word in words.items()}
        connected = amplitudes['total']-vacuum_ratio*spectator
        no_contact = amplitudes['s']+amplitudes['u']-vacuum_ratio*spectator
        rows.append(dict(time_physical=observation,
            actual_connected_lambda4_probability_coefficient=float(np.linalg.norm(connected)**2),
            unsubtracted_lambda4_probability_coefficient=float(np.linalg.norm(amplitudes['total'])**2),
            dropped_new_quantum_contact_adverse_coefficient=float(np.linalg.norm(no_contact)**2),
            actual_vacuum_disconnected_amplitude_squared=float(abs(vacuum_amplitude)**2),
            connected_amplitude_real=connected.real.tolist(), connected_amplitude_imag=connected.imag.tolist(),
            s_amplitude_norm=float(np.linalg.norm(amplitudes['s'])),
            u_amplitude_norm=float(np.linalg.norm(amplitudes['u'])),
            new_quantum_contact_amplitude_norm=float(np.linalg.norm(amplitudes['contact']))))
    exact_gauss = 0
    q = int(contract['Q'])
    for number in (4, 5):
        for mask in carriers[number]['masks']:
            n0 = sum(bool(mask & (1 << i)) for i in (0, 1, 4, 5))
            n1 = number-n0
            for m in (-1, 0, 1):
                e0, e1 = q*(n0-2+m), q*m
                exact_gauss = max(exact_gauss, abs(e0-e1-q*(n0-2)),
                                  abs(e1-e0-q*(n1-2)+(q if number == 5 else 0)))
    return dict(NEW_law='CartesianCWilson hC/Aeven; actual -log of photon/carrier transfer supplies every current/contact.',
        input=dict(parameters=asdict(parameters), Q=contract['Q'], bare_e=contract['bare_e'],
                   normal_order=normal_order, native_normal_width=normal_width,
                   Phi_reference_real=np.real(phi_reference).tolist(), Phi_reference_imag=np.imag(phi_reference).tolist()),
        geometry={key: value for key, value in geometry.items() if not isinstance(value, np.ndarray)},
        exact_integer_Gauss_defect=str(exact_gauss),
        source='S=sum_site,alpha f_site cdag_sitealpha Wsite,Ref^Q ralpha/sqrt2; all4 spin/Ref3 channels, filledRef4, f=(1,1)/sqrt2. No fixed beta projector.',
        actual_unfiltered_Ref4_scalar_Gram=raw,
        actual_full_charged_positive_halfstep_Gram=half_gram,
        actual_all_four_receiver_Gram_eigenvalues=gram_values.tolist(),
        actual_all_four_receiver_whitening_defect=receiver_defect,
        actual_prepared_source_energy_mean_physical=source_mean,
        actual_prepared_source_energy_std_physical=math.sqrt(source_variance),
        actual_reference_charged_neutral_gap_physical=float(math.log(reference/charged['values'][-1])/p.spacing),
        actual_Carrier_dimensions={str(n): carriers[n]['dimension'] for n in (4, 5)},
        all_positive_halfstep_spectral_counts={str(n): len(carriers[n]['values']) for n in (4, 5)},
        actual_Carrier_minimum_eigenvalues={str(n): float(carriers[n]['values'][0]) for n in (4, 5)},
        actual_Carrier_positive_lower_bounds={str(n): carriers[n]['lower'] for n in (4, 5)},
        actual_NEW_quantum_vertex_norms={str(n): dict(
            raising_y=float(np.linalg.norm(vertices[n]['raise_y'], 2)),
            raising_z=float(np.linalg.norm(vertices[n]['raise_z'], 2)),
            contact_zy=float(np.linalg.norm(vertices[n]['contact_zy'], 2))) for n in (4, 5)},
        probabilities=rows,
        preparation_scope='Reference lambda=0 decouples only the coexact photons. The Phi oscillator, literal electric cost, quantum compact toron/winding and unquenched Fock matter remain. All positive carrier eigenvalues define the halfsteps. No Gaussian/free-band sea is promoted to this vacuum.',
        receiver_scope='Full whitened four-spin electron packet projector tensor identity on orthogonal Ref3 holes. Every positive receiver Gram eigenvalue is retained. Exact neutral vacuum photon coefficient is subtracted at amplitude level with the spectator carrier propagation.',
        C_domain_warning='Ref4 neutral C does NOT make Ref3/Gauss a C domain: C G Cinv=-G-2Q(nRef-4). Charged halfstep is executed separately.',
        limit='Conditional finite two-site induced-Gaussian photon/one-normal-mode reference, not full60 Phi/SU4 or all-Naux microscopic law. Born is an actual lambda4 coefficient with every carrier intermediate state, NOT an interacting finitepacket probability or measured QED cross section. The amplified native Phi width is not the macro width and no energy-window/finiteSCREEN gate is changed.')


def cartesian_C_native_transfer(law: CartesianCWilson, number: int, bare_e: float,
                               photon_order: int, normal_order: int,
                               normal_width: float, direction: np.ndarray) -> dict[str, Any]:
    """Actual coupled photon/Phi/toron positive transfer, ALL native eigenstates."""
    omega = 2*math.asinh(1.)
    nodes, transform, annihilator = oscillator(photon_order, omega)
    normal_nodes, normal_transform, _ = oscillator(normal_order, 1.)
    normal_nodes *= normal_width*math.sqrt(2*math.sinh(1.))
    ph_basis = np.kron(transform, transform)
    ph_numbers = np.array([i+j for i in range(photon_order) for j in range(photon_order)])
    ph_half = ph_basis.T@np.diag(np.exp(-.5*omega*ph_numbers))@ph_basis
    normal_half = normal_transform.T@np.diag(np.exp(-.5*np.arange(normal_order)))@normal_transform
    field_half = np.kron(ph_half, normal_half)
    torons, angles = np.arange(-1, 2), 2*np.pi*np.arange(24)/24
    characters = np.exp(-1j*np.outer(torons, angles))/math.sqrt(24)
    _, masks = second_quantized(np.eye(8), number)
    size = len(masks)
    electric_half = np.exp(-.5*integer_electric_energy(masks, bare_e, torons, 0.))
    colour = np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy()
    local, minimum = [], math.inf
    for ay, az, y in itertools.product(nodes, nodes, normal_nodes):
        blocks = []
        remainder = y*np.broadcast_to(direction, (2, 4, 4, 4))
        for angle in angles:
            phases = np.zeros((2, 3))
            phases[1, 0] = angle
            phases[:, 1] = bare_e*ay*np.array([1., -1.])/math.sqrt(2)
            phases[:, 2] = bare_e*az*np.array([1., -1.])/math.sqrt(2)
            potential = law.potential(phases, remainder, colour)
            h = law.change.conj().T@potential['h']@law.change
            generator, _ = second_quantized(h, number)
            generator += potential['pref']*np.eye(size)
            values, frame = eigh(generator)
            minimum = min(minimum, math.exp(float(values[0])))
            blocks.append((frame*np.exp(values))@frame.conj().T)
        local.append(blocks)
    blocks = np.array(local)
    field_size, dimension = len(local), len(local)*3*size
    def apply(argument):
        state = argument.reshape(field_size, 3, size, -1)
        state = np.einsum('ab,bmfc->amfc', field_half, state, optimize=True)
        state *= electric_half[None, :, :, None]
        angular = np.einsum('mq,amfc->aqfc', characters.conj(), state, optimize=True)
        angular = np.einsum('aqfg,aqgc->aqfc', blocks, angular, optimize=True)
        state = np.einsum('mq,aqfc->amfc', characters, angular, optimize=True)
        state *= electric_half[None, :, :, None]
        state = np.einsum('ab,bmfc->amfc', field_half, state, optimize=True)
        return state.reshape(dimension, -1)
    matrix = np.empty((dimension, dimension), complex)
    for start in range(0, dimension, 32):
        stop = min(dimension, start+32)
        columns = np.zeros((dimension, stop-start))
        columns[np.arange(start, stop), np.arange(stop-start)] = 1.
        matrix[:, start:stop] = apply(columns)
    values, vectors = eigh((matrix+matrix.conj().T)/2)
    if values[0] <= 0:
        raise ArithmeticError('FULL coupled native positive spectrum unresolved')
    lower = minimum*math.exp(-2*omega*(photon_order-1)-(normal_order-1)
                              -float(np.max(-2*np.log(electric_half))))
    return dict(values=values, vectors=vectors, apply=apply, masks=masks,
                dimension=dimension, lower=lower, transform=transform,
                normal_transform=normal_transform, annihilator=annihilator)


def cartesian_C_native_source(p: Parameters, contract: dict[str, Any],
                              critical: dict[str, Any], phi_reference: np.ndarray,
                              normal_order: int = 2, normal_width: float = 5e-6,
                              photon_orders: tuple[int, ...] = (2, 3)) -> dict[str, Any]:
    """Execute the NEW fully coupled native source, halfstep and detector event."""
    parameters = Parameters(**dict(asdict(p), mass=critical['declared_critical_bare_mass_parameter']))
    law = CartesianCWilson(gamma_frame(), parameters, np.array([[1, 0, 0], [0, 1, 1]]))
    geometry = cartesian_C_normal_geometry(law, phi_reference)
    _, masks4 = second_quantized(np.eye(8), 4)
    _, masks5 = second_quantized(np.eye(8), 5)
    lookup4, lookup5 = {m: i for i, m in enumerate(masks4)}, {m: i for i, m in enumerate(masks5)}
    canonical_C = law.change.conj().T@law.C@law.change.conj()
    occupied = [[j for j in range(8) if m & (1 << j)] for m in masks4]
    Gamma_C = np.array([[np.linalg.det(canonical_C[np.ix_(r, c)])
                        for c in occupied] for r in occupied])
    PH = np.zeros((70, 70), complex)
    for column, original in enumerate(masks4):
        mask, sign = original, 1
        for bit in range(7, -1, -1):
            sign *= (-1)**((mask & ((1 << bit)-1)).bit_count())
            mask ^= 1 << bit
        PH[lookup4[mask], column] = sign
    unitary_C = Gamma_C@PH
    path_source = law.source_packet_jet(np.zeros((2, 3)),
        np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy(), (np.zeros((2, 3)), np.zeros((2, 3))))
    spatial = law.change.conj().T@path_source['psi']
    sources = np.zeros((4, 56, 70), complex)
    for column, mask in enumerate(masks4):
        for bit in range(8):
            if not mask & (1 << bit):
                sources[:, lookup5[mask | (1 << bit)], column] += (
                    (-1)**((mask & ((1 << bit)-1)).bit_count())*spatial[bit])
    records = []
    for order in photon_orders:
        sectors = {n: cartesian_C_native_transfer(law, n, contract['bare_e'], order,
                     normal_order, normal_width, geometry['direction']) for n in (4, 5)}
        neutral, charged = sectors[4], sectors[5]
        nv, nf = neutral['values'], neutral['vectors']
        cv, cf = charged['values'], charged['vectors']
        ground, reference = nf[:, -1:], nv[-1]
        py = neutral['transform'].T@np.diag((-1.)**np.arange(order))@neutral['transform']
        pn = neutral['normal_transform'].T@np.diag((-1.)**np.arange(normal_order))@neutral['normal_transform']
        def conjugate(state):
            tensor = state.reshape(order, order, normal_order, 3, 70, -1)[:, :, :, ::-1]
            return np.einsum('ai,bj,dk,uv,ijkmvc->abdmuc', py, py, pn,
                             unitary_C, tensor).reshape(neutral['dimension'], -1)
        probe = (np.sin(np.arange(neutral['dimension'])*.37)
                 +1j*np.cos(np.arange(neutral['dimension'])*.13))[:, None]
        covariance = float(np.linalg.norm(neutral['apply'](conjugate(probe))
                           -conjugate(neutral['apply'](probe)))/np.linalg.norm(neutral['apply'](probe)))
        creator = neutral['transform'].T@neutral['annihilator'].T@neutral['transform']
        field_y = np.kron(np.kron(creator, np.eye(order)), np.eye(normal_order))
        field_z = np.kron(np.kron(np.eye(order), creator), np.eye(normal_order))
        field_size = order*order*normal_order
        def photon(field):
            return np.einsum('ab,bmfc->amfc', field, ground.reshape(field_size, 3, 70, -1)).reshape(neutral['dimension'], -1)
        y, z = photon(field_y), photon(field_z)
        circular = y+1j*z
        circular /= np.linalg.norm(circular)
        prepared = nf@(np.sqrt(nv/reference)[:, None]*(nf.conj().T@circular))
        photon_gram = float(np.linalg.norm(prepared)**2)
        if photon_gram < neutral['lower']/reference*(1-2e-10):
            raise ArithmeticError('FULL photon halfstep violates positive lower bound')
        prepared /= math.sqrt(photon_gram)
        c_overlap = np.vdot(prepared, conjugate(prepared))
        aligned = float(np.linalg.norm(conjugate(prepared)-prepared*c_overlap/abs(c_overlap)))
        def source_state(state):
            acted = np.einsum('rfg,amgc->amfrc', sources, state.reshape(field_size, 3, 70, -1))
            acted = acted.reshape(charged['dimension'], 4)
            half = cf@(np.sqrt(cv/reference)[:, None]*(cf.conj().T@acted))
            return acted, half
        acted, half = source_state(prepared)
        raw, half_gram = float(np.linalg.norm(acted)**2), float(np.linalg.norm(half)**2)
        incoming_ph = nf@(np.sqrt(nv/reference)[:, None]*(nf.conj().T@y))
        incoming_ph /= np.linalg.norm(incoming_ph)
        outgoing_ph = nf@(np.sqrt(nv/reference)[:, None]*(nf.conj().T@z))
        outgoing_ph /= np.linalg.norm(outgoing_ph)
        _, incoming = source_state(incoming_ph)
        _, outgoing = source_state(outgoing_ph)
        incoming /= np.linalg.norm(incoming)
        gram_values, gram_vectors = eigh(outgoing.conj().T@outgoing)
        if gram_values[0] <= 0:
            raise ArithmeticError('Coupled all-four receiver Gram is not positive')
        receiver = outgoing@(gram_vectors/np.sqrt(gram_values))
        ie, re = cf.conj().T@incoming, cf.conj().T@receiver
        energies = -np.log(cv/reference)/p.spacing
        events = []
        for observation in (0., .005, .02, .05):
            amplitude = re.conj().T@(np.exp(-1j*energies*observation)[:, None]*ie)
            events.append(dict(time_physical=observation,
                exact_coupled_prepared_packet_projector_probability=float(np.linalg.norm(amplitude)**2)))
        weights = np.sum(abs(cf.conj().T@half)**2, axis=1)/half_gram
        mean = float(weights@energies)
        records.append(dict(photon_order=order, normal_order=normal_order,
            actual_neutral_dimension=neutral['dimension'], actual_charged_dimension=charged['dimension'],
            FULL_positive_halfstep_spectral_counts=dict(neutral=len(nv), charged=len(cv)),
            actual_positive_minimum_eigenvalues=dict(neutral=float(nv[0]), charged=float(cv[0])),
            actual_positive_lower_bounds=dict(neutral=neutral['lower'], charged=charged['lower']),
            actual_neutral_unitary_C_covariance=covariance,
            actual_same_full_halfstep_photon_C_density_trace_distance=2*aligned*math.sqrt(max(0., 1-aligned*aligned/4)),
            actual_full_positive_halfstep_photon_Gram=photon_gram,
            proven_native_halfstep_photon_Gram_lower=neutral['lower']/reference,
            actual_primitive_Ref4_scalar_source_raw_Gram=raw,
            actual_charged_source_halfstep_Gram=half_gram,
            actual_all_four_receiver_Gram_eigenvalues=gram_values.tolist(),
            actual_source_energy_mean_physical=mean,
            actual_source_energy_std_physical=math.sqrt(float(weights@((energies-mean)**2))),
            actual_coupled_charged_neutral_gap_physical=float(math.log(reference/cv[-1])/p.spacing),
            detector_events=events))
        if max(covariance, aligned, abs(raw-1)) > 2e-9:
            raise ArithmeticError('NEW coupled full-halfstep scalar C/source proof failed')
    return dict(NEW_law='CartesianCWilson completed log/prefactor, coupled coexact photons and one normal mode.',
        Q=contract['Q'], bare_e=contract['bare_e'], native_normal_width=normal_width,
        actual_bare_mass=parameters.mass, records=records,
        source='Unprojected full4-spin scalar cdag W^Q r/sqrt2, filledRef4 to four orthogonal Ref3 holes. Literal integer electric source and all retained winding.',
        domain='NeutralRef4 C ONLY; actual chargedRef3 halfstep separately executed. No C G Cinv=-G assumption on Ref3.',
        detector='Positive full-four-spin packet projector tensor identityRef3, built from the SAME interacting photon-inserted full-halfstep states. Probability includes initial dressed overlap and disconnected conversion; not labelled connected scattering.',
        limit='Finite amplified one-normal-mode/two-site induced-Gaussian photon/toron reference, not full60 Phi/SU4 or microscopic all-Naux photon law. ALL positive native spectra are used, no TOP2, Gaussian ground promotion, fitted mass or changed finiteSCREEN/source energy gate.')


def second_quantized(h: np.ndarray, number: int) -> tuple[np.ndarray, list[int]]:
    masks = [sum(1 << j for j in occupied) for occupied in itertools.combinations(range(8), number)]
    lookup = {mask: i for i, mask in enumerate(masks)}
    out = np.zeros((len(masks), len(masks)), complex)
    for column, mask in enumerate(masks):
        for j in range(8):
            if not mask & (1 << j):
                continue
            removed = mask ^ (1 << j)
            first = (-1)**((mask & ((1 << j)-1)).bit_count())
            for i in range(8):
                if removed & (1 << i):
                    continue
                sign = first*(-1)**((removed & ((1 << i)-1)).bit_count())
                out[lookup[removed | (1 << i)], column] += sign*h[i, j]
    return out, masks


def integer_flux_residue(p: Parameters, q: int) -> int:
    fraction = Fraction(p.flux_fraction)
    return (q*fraction.numerator//fraction.denominator) % q


def integer_electric_energy(masks: list[int], e: float, torons: np.ndarray,
                            fraction: float) -> np.ndarray:
    # Exact primitive E0=Q(r0+m)+r,E1=Qm+r. Fraction is r/Q, not a charge.
    # Static -Q at site1 in K5 ensures rho1=-rho0; native Ref remains charge0.
    site0 = (0, 1, 4, 5)
    r0 = np.array([sum(bool(mask & (1 << i)) for i in site0)-2 for mask in masks])
    shifted = torons[:, None]+fraction
    return .5*e*e*((r0[None, :]+shifted)**2+shifted**2)


def oscillator(order: int, omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    annihilator = np.diag(np.sqrt(np.arange(1, order)), 1)
    coordinate = (annihilator+annihilator.T)/math.sqrt(2*math.sinh(omega))
    nodes, transform = np.linalg.eigh(coordinate)
    return nodes, transform, annihilator


def native_sector(gamma: np.ndarray, p: Parameters, e: float, order: int, number: int,
                  drop_prefactor: bool = False, toron_cutoff: int = 1) -> dict[str, Any]:
    """Positive finite Fourier/Gauss projection of detB Gamma(T1).

    Exact integer electric flux includes the charged source string and winding.
    A normalized Haar Fourier quadrature gives a positive Toeplitz compression.
    Photon DVR remains a finite model, not an infinite-Fock truncation theorem.
    """
    omega = 2*math.asinh(1.)
    nodes, transform, annihilator = oscillator(order, omega)
    photon_basis = np.kron(transform, transform)
    occupations = np.array([i+j for i in range(order) for j in range(order)])
    half_photon = photon_basis.T@np.diag(np.exp(-.5*omega*occupations))@photon_basis
    _, masks = second_quantized(np.eye(8), number)
    fock_size = len(masks)
    torons = np.arange(-toron_cutoff, toron_cutoff+1)
    size = fock_size*len(torons)
    contract = charge_contract(p)
    primitive_q = int(contract['Q'])
    residue = integer_flux_residue(p, primitive_q)
    fraction = residue/primitive_q
    electric = integer_electric_energy(masks, e, torons, fraction).reshape(-1)
    quadrature = max(24, 4*toron_cutoff+4)
    angles = 2*np.pi*np.arange(quadrature)/quadrature
    characters = np.exp(-1j*np.outer(torons, angles))/math.sqrt(quadrature)
    local_blocks = []
    minimum = float('inf')
    local_minimum = float('inf')
    winding_norm = 0.
    spin_x = -.5j*gamma[2]@gamma[3]
    _, basis = np.linalg.eigh(gamma[0])
    change = np.column_stack((basis[:, 2:], basis[:, :2]))
    site_spin = np.kron(np.eye(2), change.conj().T@spin_x@change)
    permutation = [0, 1, 4, 5, 2, 3, 6, 7]
    site_spin = site_spin[np.ix_(permutation, permutation)]
    free_transfer, _ = one_transfer(gamma, p, np.zeros((2, 3)))
    _, free_vectors = eigh(free_transfer)
    free_sea = free_vectors[:, 4:]
    # A continuous local occupied frame, not arbitrary eigensolver phases.
    free_positive = free_vectors[:, 2:4]
    _, free_spin = eigh(free_positive.conj().T@site_spin@free_positive)
    free_electron = free_positive@free_spin[:, -1]
    signed_angles = (angles+math.pi) % (2*math.pi)-math.pi
    angular_packet = np.exp(-signed_angles**2/(2*p.toron_packet_width**2))
    angular_packet /= np.linalg.norm(angular_packet)
    photon_vacuum = photon_basis[0]
    local_wave = np.zeros((order*order, quadrature, fock_size), complex)
    gram_minimum = 1.
    for coordinate_index, (ay, az) in enumerate(itertools.product(nodes, repeat=2)):
        coordinate_blocks = []
        winding = np.zeros((fock_size, fock_size), complex)
        for index, holonomy in enumerate(angles):
            phases = np.zeros((2, 3))
            # Tree gauge theta_x0=0, Q theta_x1=holonomy; character m means
            # primitive integer electric flux Qm, never noninteger charge.
            phases[1, 0] = holonomy if e else 0.
            phases[:, 1] = e*ay*np.array([1., -1.])/math.sqrt(2)
            phases[:, 2] = e*az*np.array([1., -1.])/math.sqrt(2)
            t, logdet_b = one_transfer(gamma, p, phases)
            eigenvalues, one_vectors = eigh(t)
            occupied = one_vectors[:, 4:]@one_vectors[:, 4:].conj().T
            mapped = occupied@free_sea
            gram_values, gram_vectors = eigh(mapped.conj().T@mapped)
            gram_minimum = min(gram_minimum, float(gram_values[0]))
            if gram_values[0] <= 1e-12:
                raise ArithmeticError('Local sea chart fails before packet preparation')
            sea = mapped@(gram_vectors/np.sqrt(gram_values))@gram_vectors.conj().T
            if number == 5:
                electron = free_electron-occupied@free_electron
                electron /= np.linalg.norm(electron)
                columns = np.column_stack((sea, electron))
            else:
                columns = sea
            local_fock = np.array([np.linalg.det(columns[
                [bit for bit in range(8) if mask & (1 << bit)], :]) for mask in masks])
            local_wave[coordinate_index, index] = (
                photon_vacuum[coordinate_index]*angular_packet[index]*local_fock)
            minimum = min(minimum, float(eigenvalues[0]))
            logarithm = (one_vectors*np.log(eigenvalues))@one_vectors.conj().T
            lifted, _ = second_quantized(logarithm, number)
            if not drop_prefactor:
                lifted += logdet_b*np.eye(fock_size)
            eigen, spin = eigh(lifted)
            local = (spin*np.exp(eigen))@spin.conj().T
            local_minimum = min(local_minimum, float(math.exp(eigen[0])))
            coordinate_blocks.append(local)
            winding += np.exp(-1j*holonomy)*local/quadrature
        winding_norm = max(winding_norm, float(np.linalg.norm(winding)))
        local_blocks.append(coordinate_blocks)
    blocks = np.array(local_blocks)
    electric_half = np.exp(-.5*electric).reshape(len(torons), fock_size)
    dimension = order*order*size
    def apply_kernel(argument: np.ndarray) -> np.ndarray:
        single = argument.ndim == 1
        state = argument.reshape(order*order, len(torons), fock_size, -1)
        state = np.einsum('ab,bmfc->amfc', half_photon, state, optimize=True)
        state *= electric_half[None, :, :, None]
        phase_state = np.einsum('mq,amfc->aqfc', characters.conj(), state, optimize=True)
        phase_state = np.einsum('aqfg,aqgc->aqfc', blocks, phase_state, optimize=True)
        state = np.einsum('mq,aqfc->amfc', characters, phase_state, optimize=True)
        state *= electric_half[None, :, :, None]
        state = np.einsum('ab,bmfc->amfc', half_photon, state, optimize=True)
        return state.reshape(dimension) if single else state.reshape(dimension, -1)
    # SciPy's runtime factory accepts callbacks; its abstract-class stub does not.
    kernel = cast(Any, LinearOperator)((dimension, dimension),
                 matvec=apply_kernel, matmat=apply_kernel, dtype=np.dtype(complex))
    transfer_lower = local_minimum*math.exp(-2*omega*(order-1)-float(max(electric)))
    if dimension <= p.dense_limit:
        transfer = np.empty((dimension, dimension), complex)
        for start in range(0, dimension, 32):
            stop = min(dimension, start+32)
            unit_columns = np.zeros((dimension, stop-start))
            unit_columns[np.arange(start, stop), np.arange(stop-start)] = 1.
            transfer[:, start:stop] = apply_kernel(unit_columns)
        values, vectors = eigh((transfer+transfer.conj().T)/2, driver='evd')
        if values[0] <= 0:
            raise ArithmeticError('Native full quantum transfer is not positive')
        transfer_lower = float(values[0])
    else:
        # Installed SciPy stub incorrectly types fractional tolerance as int.
        values, vectors = cast(Any, eigsh)(kernel, k=min(160, dimension-2),
             which='LA', tol=1e-12, v0=np.linspace(.1, 1., dimension).astype(complex))
        orthonormal, _ = np.linalg.qr(vectors)
        projected = orthonormal.conj().T@apply_kernel(orthonormal)
        values, rotations = eigh((projected+projected.conj().T)/2)
        vectors = orthonormal@rotations
    eigen_residual = 0.
    for start in range(max(0, len(values)-160), len(values), 32):
        chosen = vectors[:, start:start+32]
        residual = apply_kernel(chosen)-chosen*values[None, start:start+32]
        eigen_residual = max(eigen_residual, float(max(np.linalg.norm(residual, axis=0))))
    energies = -np.log(values/values[-1])
    ground = vectors[:, -1]
    ground_dimension = int(np.count_nonzero(energies < 1e-9))
    # Diagonalize physical Jx in the degenerate charged dressed ground space.
    spin_fock, _ = second_quantized(site_spin, number)
    if number == 5 and ground_dimension >= 2:
        subspace = vectors[:, -ground_dimension:]
        tensor = subspace.reshape(order*order*len(torons), fock_size, ground_dimension)
        spin_applied = np.einsum('fg,agc->afc', spin_fock, tensor).reshape(dimension, ground_dimension)
        spins, rotations = eigh(subspace.conj().T@spin_applied)
        incoming_ground = subspace@rotations[:, -1]
        outgoing_ground = incoming_ground
    else:
        spins = np.array([0.])
        incoming_ground = outgoing_ground = ground
    creator = transform.T@annihilator.T@transform
    def create_packet(state: np.ndarray, axis: int) -> np.ndarray:
        tensor = state.reshape(order, order, size)
        return (np.einsum('ij,jkf->ikf', creator, tensor) if axis == 0
                else np.einsum('ij,kjf->kif', creator, tensor)).reshape(-1)
    initial = create_packet(incoming_ground, 0)
    final = create_packet(outgoing_ground, 1)
    initial -= incoming_ground*np.vdot(incoming_ground, initial)
    final -= outgoing_ground*np.vdot(outgoing_ground, final)
    initial /= np.linalg.norm(initial)
    # Detector orthogonal to initial: exclusive connected channel, no t=0 echo.
    final -= initial*np.vdot(initial, final)
    final /= np.linalg.norm(final)
    vi, vf = vectors.conj().T@initial, vectors.conj().T@final
    initial_missing = max(0., 1-float(np.vdot(vi, vi).real))
    final_missing = max(0., 1-float(np.vdot(vf, vf).real))
    amplitude_error = math.sqrt(initial_missing*final_missing)
    probabilities, amplitudes = [], []
    for observation in p.observation_times:
        amplitude = np.vdot(vf, np.exp(-1j*observation*energies)*vi)
        amplitudes.append([float(amplitude.real), float(amplitude.imag)])
        probabilities.append(float(abs(amplitude)**2))
    # High-shell probability is a discriminant, not a truncation certificate.
    edge = np.array([i == order-1 or j == order-1 for i in range(order) for j in range(order)])
    initial_number = (photon_basis@initial.reshape(order*order, size)).reshape(-1)
    shell = float(np.sum(abs(initial_number.reshape(order*order, size)[edge])**2))
    toron_state = initial_number.reshape(order*order, len(torons), fock_size)
    toron_edge = float(np.sum(abs(toron_state[:, [0, -1], :])**2))
    prepared = np.einsum('mq,aqf->amf', characters, local_wave).reshape(-1)
    projection_weight = float(np.vdot(prepared, prepared).real)
    prepared /= math.sqrt(projection_weight)
    local_initial = create_packet(prepared, 0)
    local_final = create_packet(prepared, 1)
    local_initial -= prepared*np.vdot(prepared, local_initial)
    local_final -= prepared*np.vdot(prepared, local_final)
    local_initial /= np.linalg.norm(local_initial)
    local_final -= local_initial*np.vdot(local_initial, local_final)
    local_final /= np.linalg.norm(local_final)
    local_vi, local_vf = vectors.conj().T@local_initial, vectors.conj().T@local_final
    local_error = math.sqrt(max(0., 1-float(np.vdot(local_vi, local_vi).real))*
                            max(0., 1-float(np.vdot(local_vf, local_vf).real)))
    local_amplitudes = [np.vdot(local_vf, np.exp(-1j*t*energies)*local_vi)
                        for t in p.observation_times]
    local_probabilities = [float(abs(amplitude)**2) for amplitude in local_amplitudes]
    local_intervals = [[max(0., abs(amplitude)-local_error)**2,
                       min(1., (abs(amplitude)+local_error)**2)]
                      for amplitude in local_amplitudes]
    # No common energy phase removes variance or stabilizes a prepared toron.
    # Price real evolution of the same packet and its actual electric energy.
    prepared_vi = vectors.conj().T@prepared
    prepared_weights = abs(prepared_vi)**2
    prepared_missing = max(0., 1-float(sum(prepared_weights)))
    maximum_energy = math.log(values[-1]/transfer_lower)
    energy_mean = float(prepared_weights@energies)
    energy_second = float(prepared_weights@(energies**2))
    energy_mean_interval = [energy_mean, energy_mean+prepared_missing*maximum_energy]
    energy_variance_interval = [
        max(0., energy_second-energy_mean_interval[1]**2),
        max(0., energy_second+prepared_missing*maximum_energy**2-energy_mean**2)]
    electric_weights = np.sum(abs(prepared.reshape(order*order, size))**2, axis=0)
    electric_mean = float(electric_weights@electric)
    electric_variance = max(0., float(electric_weights@(electric**2))-electric_mean**2)
    packet_missing = max(0., 1-float(np.vdot(local_vi, local_vi).real))
    window_error = 2*math.sqrt(packet_missing)+packet_missing
    toron_window = abs(signed_angles) <= 2*p.toron_packet_width
    window_survival = []
    prepared_survival = []
    for observation in p.observation_times:
        phase = np.exp(-1j*observation*energies)
        evolved = (vectors@(phase*local_vi)).reshape(order*order, len(torons), fock_size)
        angular = np.einsum('mq,amf->aqf', characters.conj(), evolved)
        probability = float(np.sum(abs(angular[:, toron_window, :])**2))
        window_survival.append([max(0., probability-window_error), min(1., probability+window_error)])
        overlap = abs(prepared_weights@phase)
        prepared_survival.append([max(0., overlap-prepared_missing)**2,
                                  min(1., (overlap+prepared_missing)**2)])
    return dict(order=order, toron_cutoff=toron_cutoff, active_number=number, dimension=dimension,
                retained_spectral_dimension=len(values), observed_eigen_residual=eigen_residual,
                transfer_minimum_lower_bound=transfer_lower, transfer_maximum=float(values[-1]),
                one_body_transfer_minimum=minimum, log_ground_transfer=float(math.log(values[-1])),
                local_periodic_preparation=dict(
                    angular_packet_width=p.toron_packet_width,
                    integer_Fourier_projection_weight=projection_weight,
                    local_sea_overlap_gram_minimum=gram_minimum,
                    prepared_energy_mean_lattice_interval=energy_mean_interval,
                    prepared_energy_variance_lattice_interval=energy_variance_interval,
                    prepared_energy_variance_physical_interval=[
                        value/p.spacing**2 for value in energy_variance_interval],
                    integer_electric_energy_mean_lattice=electric_mean,
                    integer_electric_energy_variance_lattice=electric_variance,
                    integer_electric_energy_mean_physical=electric_mean/p.spacing,
                    integer_electric_energy_std_physical=math.sqrt(electric_variance)/p.spacing,
                    prepared_state_survival_intervals=prepared_survival,
                    incoming_active_holonomy_window_survival_intervals=window_survival,
                    window='|phi=Q theta modulo2pi|<=2 packet_width. This Q-cell preparation is not primitive theta localization; time is lattice time.',
                    exclusive_probabilities=local_probabilities,
                    amplitudes=[[float(z.real), float(z.imag)] for z in local_amplitudes],
                    finite_spectral_amplitude_error=local_error,
                    finite_spectral_probability_intervals=local_intervals,
                    integer_toron_edge_probability=float(np.sum(abs(
                        prepared.reshape(order*order, len(torons), fock_size)[:, [0, -1]])**2)),
                    meaning='Prepared local periodic Gaussian around holonomy0 with actual field-dependent filled Wilson Slater and one projected charged mode. SAME positive native transfer, no trap, no mass fit, no global-ground substitution. Finite observation, not stationary dressed photon/electron poles.'),
                dressed_ground_degeneracy=ground_dimension, dressed_spin_eigenvalues=spins.tolist(),
                dressed_excitation_gaps=sorted(energies[energies > 1e-9].tolist())[:8],
                photon_spectral_measure=[dict(gap=float(energies[index]),
                                              normalized_packet_residue=float(abs(vi[index])**2))
                                         for index in np.argsort(abs(vi)**2)[-8:][::-1]],
                initial_final_overlap=float(abs(np.vdot(initial, final))),
                initial_highest_photon_shell_probability=shell,
                observation_times=list(p.observation_times), amplitudes=amplitudes,
                exclusive_probabilities=probabilities,
                finite_spectral_amplitude_error=amplitude_error,
                finite_spectral_probability_intervals=[
                    [max(0., math.sqrt(probability)-amplitude_error)**2,
                     min(1., (math.sqrt(probability)+amplitude_error)**2)]
                    for probability in probabilities],
                error_meaning='Cauchy-Schwarz bound for omitted spectral subspace of THIS finite matrix, excluding floating-point/ARPACK residual error and infinite-toron/photon truncation. Dense cases have all eigenpairs.',
                norm_residual=float(abs(np.vdot(vi, vi)-1)),
                toron_highest_shell_probability=toron_edge,
                nonzero_winding_transition_norm=winding_norm,
                compact_Haar_quadrature_points=quadrature,
                exact_integer_Gauss=True,
                integer_Floquet_residue=str(residue), integer_Floquet_denominator=str(primitive_q),
                primitive_flux='E0=Q(n0-2+m)+r,E1=Qm+r; characters e^(i(Qm+r)theta). Common r phase cancels in field multiplication, but NOT in electric transfer.',
                _states=dict(masks=masks, ground=ground, charged_ground=incoming_ground,
                             ground_space=vectors[:, -ground_dimension:]),
                source='Charged intertwiner with neutral Ref and -Q static compensator at site1. Integer string on x0 and ALL retained integer winding characters; norm conditional on dressed spectral preparation.',
                meaning='Exact finite projected quantum spectral probability for same-spin y-to-z packet on interacting charged dressed ground. Vacuum polarization conversion is the disconnected control; not a full-box cross section or unprojected error certificate.')


def determinant_response(gamma: np.ndarray, p: Parameters, e: float) -> dict[str, Any]:
    """Full unquenched active determinant on a time-dependent finite link slab."""
    nt = 6
    profile = np.sin(2*np.pi*np.arange(nt)/nt)
    def dirac(amplitude: float) -> tuple[np.ndarray, np.ndarray, float]:
        d = np.zeros((8*nt, 8*nt), complex)
        product = np.eye(8, dtype=complex)
        prefactor = 0.
        plus, minus = (np.eye(4)+gamma[0])/2, (np.eye(4)-gamma[0])/2
        for t in range(nt):
            phases = np.zeros((2, 3))
            phases[:, 1] = amplitude*e*profile[t]*np.array([1., -1.])/math.sqrt(2)
            b, c = spatial_blocks(gamma, p, phases)
            sx = slice(8*t, 8*t+8)
            d[sx, sx] = b+c
            next_t = (t+1) % nt
            sy = slice(8*next_t, 8*next_t+8)
            sign = -1 if t == nt-1 else 1
            d[sx, sy] -= sign*np.kron(np.eye(2), minus)
            d[sy, sx] -= sign*np.kron(np.eye(2), plus)
            transfer, logdet = one_transfer(gamma, p, phases)
            product = transfer@product
            prefactor += logdet
        return d, product, prefactor
    amplitude = .23
    d, product, scalar = dirac(amplitude)
    _, direct = np.linalg.slogdet(d)
    _, recurrence = np.linalg.slogdet(np.eye(8)+product)
    defect = abs(direct-scalar-recurrence)
    if defect > 1e-9:
        raise ArithmeticError(f'Active determinant/positive transfer mismatch {defect}')
    h = 1e-4
    low, _, _ = dirac(amplitude-h)
    high, _, _ = dirac(amplitude+h)
    derivative = (high-low)/(2*h)
    contact = (high-2*d+low)/(h*h)
    inverse = np.linalg.inv(d)
    current = -np.trace(inverse@derivative).real
    bubble = np.trace(inverse@derivative@inverse@derivative).real
    seagull = -np.trace(inverse@contact).real
    action_curvature = float(bubble+seagull)
    curvature_check = (-np.linalg.slogdet(high)[1]+2*direct-np.linalg.slogdet(low)[1])/(h*h)
    # Full primitive temporal current/contact: an actual gauge orbit changes
    # temporal AND spatial links. The Ward Hessian must vanish, not just its
    # transverse response. Dropping the temporal contact gives a false signal.
    gauge = np.repeat(np.array([math.sin(2*math.pi*t/nt)+.31*(-1)**x
                                for t in range(nt) for x in range(2)]), 4)
    delta = gauge[:, None]-gauge[None, :]
    gauge_current = 1j*e*delta*d
    gauge_contact = -e*e*delta**2*d
    temporal_mask = np.repeat(np.arange(nt), 8)[:, None] != np.repeat(np.arange(nt), 8)[None, :]
    ward_bubble = np.trace(inverse@gauge_current@inverse@gauge_current).real
    ward_contact = -np.trace(inverse@gauge_contact).real
    temporal_contact = -np.trace(inverse@(gauge_contact*temporal_mask)).real
    ward_residual = abs(ward_bubble+ward_contact)
    if ward_residual > 1e-10 or abs(temporal_contact) < 1e-5:
        raise ArithmeticError('Temporal-contact control does not discriminate')
    return dict(time_slices=nt, spatial_box=[2, 1, 1], field_amplitude=amplitude,
                direct_active_logdet=float(direct), transfer_logdet=float(scalar+recurrence),
                determinant_identity_defect=float(defect), omitted_detB_error=float(abs(scalar)),
                actual_current=float(current), current_current_bubble=float(bubble),
                actual_contact=float(seagull), active_radiative_curvature=action_curvature,
                finite_difference_curvature=float(curvature_check),
                curvature_disagreement=float(abs(curvature_check-action_curvature)),
                missing_contact_curvature=float(bubble),
                primitive_gauge_Ward_hessian=float(ward_bubble+ward_contact),
                primitive_gauge_current_bubble=float(ward_bubble),
                primitive_gauge_contact=float(ward_contact),
                actual_temporal_contact=float(temporal_contact),
                omitted_temporal_contact_gauge_defect=float(abs(temporal_contact)),
                interpretation='One filled active determinant only; not multiplied again into native Fock transfer. Exact time-dependent Wilson action retains both transfer bands and temporal Grassmann hoppings; finite transverse response is not zero-momentum alpha renormalization.')


def dressed_source_price(charged: dict[str, Any], vacuum: dict[str, Any],
                         p: Parameters, q: int) -> dict[str, Any]:
    """Actual compact integer string intertwiner c0dagger times Ux0**Q."""
    charged_states, vacuum_states = charged.pop('_states'), vacuum.pop('_states')
    cm, vm = charged_states['masks'], vacuum_states['masks']
    lookup = {mask: i for i, mask in enumerate(cm)}
    source = np.zeros((len(cm), len(vm)), complex)
    # Addressed rank-one neutral Ref supplies the missing fermion; external
    # apparatus adds -Q at site1, so total charge is zero in every component.
    for column, mask in enumerate(vm):
        if not mask & 1:
            source[lookup[mask | 1], column] = 1.
    replicas = charged['order']**2*(2*charged['toron_cutoff']+1)
    acted = (vacuum_states['ground'].reshape(replicas, len(vm))@source.T).reshape(-1)
    norm_squared = float(np.vdot(acted, acted).real)
    ground_weight = float(np.linalg.norm(charged_states['ground_space'].conj().T@acted)**2)
    gauss_defects = []
    for mask in cm:
        n0 = sum(bool(mask & (1 << i)) for i in (0, 1, 4, 5))
        n1 = 5-n0
        for m in range(-charged['toron_cutoff'], charged['toron_cutoff']+1):
            residue = integer_flux_residue(p, q)
            e0, e1 = q*(n0-2+m)+residue, q*m+residue
            rho0, rho1 = q*(n0-2), q*(n1-2)-q
            gauss_defects.extend((e0-e1-rho0, e1-e0-rho1))
    return dict(source_norm_squared=norm_squared,
                dressed_ground_preparation_weight=ground_weight/norm_squared,
                addressed_spin_ground_overlap=float(abs(np.vdot(charged_states['charged_ground'], acted))**2/norm_squared),
                exact_integer_Gauss_defect=str(max(abs(x) for x in gauss_defects)),
                external_reference_charge=str(-q), source_charge=str(q),
                reference_operator_charge=0, reference_distance=p.spacing,
                external_sector='One static spin-zero -Q worldline at site1; no fictitious boundary.',
                compact_dressing='Integer Ux0**Q string. Tree-gauge source shifts E0 byQ, leaves E1=Qm+r, preserving the selected primitive Floquet residue. ALL retained winding transitions.',
                self_energy='Included as e²[(r0+m+r/Q)²+(m+r/Q)²]/2 before transfer. No subtracted Coulomb/self-energy.',
                preparation_scope='Conditional spectral preparation has the reported nonunit probability; not an unprepared charged-packet experiment.')


def run(p: Parameters) -> dict[str, Any]:
    contract = charge_contract(p)
    gamma = gamma_frame()
    e = contract['bare_e']
    response = determinant_response(gamma, p, e)
    native = []
    for order, cutoff in zip(p.orders, p.toron_cutoffs, strict=True):
        charged = native_sector(gamma, p, e, order, 5, toron_cutoff=cutoff)
        vacuum = native_sector(gamma, p, e, order, 4, toron_cutoff=cutoff)
        charged['source_preparation'] = dressed_source_price(charged, vacuum, p, int(contract['Q']))
        charged['dressed_mass_gap_lattice'] = vacuum['log_ground_transfer']-charged['log_ground_transfer']
        charged['dressed_mass_gap_physical'] = charged['dressed_mass_gap_lattice']/p.spacing
        native.append(dict(charged=charged, vacuum=vacuum))
    adverse = native_sector(gamma, p, e, p.orders[0], 5, drop_prefactor=True)
    zero = native_sector(gamma, p, 0., p.orders[0], 5)
    adverse.pop('_states')
    zero.pop('_states')
    differences = [max(abs(np.array(b['charged']['exclusive_probabilities'])
                           -np.array(a['charged']['exclusive_probabilities'])))
                   for a, b in zip(native, native[1:])]
    return dict(parameters=asdict(p), charge_contract=contract,
                active_unquenched_response=response, native_positive_transfer=native,
                adverse_controls=dict(drop_detB=adverse, zero_coupling=zero),
                truncation_sequence_probability_differences=differences,
                photon_kernel='Same induced beta, canonical coexact oscillator: omega=2asinh(khat/2), khat=2. Vacuum variance1/(2sinh omega). Symmetric positive photon/Fock/Coulomb transfer; no imported Thomson or KN coefficient.',
                material_repairs=['Reject Q delta<<1 and free-global-state closeness; retain full nonlinear compact character at all DVR nodes.',
                                  'Replace free sea boundary by exact interacting dressed charged/vacuum ground states, then create/detect polarized photon packets. First spinflip probe failed by forward-channel selection; repaired to same-spin polarization conversion.',
                                  'Reject fixed minimizing integer-flux state: it kills compact hopping. Retain integer toron m, exact winding Fourier transitions, actual electric cost and addressed remote source.',
                                  'Reject global holonomy ground mass93.7 as physical mass1: prepare a periodic Gaussian around holonomy0 with field-dependent filled sea and real charged local packet, propagated by the unchanged transfer. Preparation and spectral truncation prices remain explicit.',
                                  'Increase native photon/toron cutoffs after edge-shell/convergence failure; determinant scalar retained at every node.'],
                rejected_historical_capsules=dict(
                    first_probe='Spinflip forward selection gives zero; free/fractional background construction withdrawn.',
                    repaired_probe='Finite photon cutoff converged but minimized toron string incompatible with x hopping; not a compact source result.',
                    sparse_probe='Complex ARPACK degeneracy gave nonorthogonal retained vectors and false clamped spectral error. Withdrawn; QR plus Hermitian Rayleigh-Ritz is now used.'),
                conditional_packet='Normalized finite standing-wave y/z photons on the two-site coexact pi mode; projected Gaussian oscillator/reference. NOT the baseline125-mode spatial Gaussian packet or full-box continuum scattering.',
                error_status='Finite spectral-subspace Cauchy-Schwarz intervals only, with numerical eigen residual observed separately. Successive cutoffs/highest-shell probability are NOT rigorous infinite-toron/photon or omitted-mode bounds. No unprojected probability/error or observed physical-QED claim.',
                continuum_and_renormalization_debt=['Two-site box has k=pi/a, NOT continuum/IR Compton. No continuum/lattice/finite-volume extrapolation paid.',
                    'Gaussian induced photon kernel imported from the large-box supplier as a conditional reference. The auxiliary law CANNOT be rerun on two sites: length-two winding would dominate Naux kaux². Thus this native diagnostic is not literally the same full microscopic lattice law.',
                    'One selected primitive Floquet coset in the native quotient; charge1 auxiliaries mix other cosets in the full law. Native diagnostics do not certify that full mixing.',
                    'Bare alpha=Q^2/(4pi beta); measured zero-momentum renormalized alpha not obtained from finite UV response.',
                    'Mass means finite dressed charged-neutral transfer gap, not infinite-volume infraparticle pole; bare mass remains input.',
                    'No measured velocity renormalization with only two momenta; no fitted common speed.',
                    'Full compact Haar transfer, all omitted coexact/toron modes and rigorous infinite-Fock remainder remain missing. Integer Gauss/source dressing and its Coulomb energy are retained in the native quotient.',
                    'Normal/Phi/colour transport conditional; Naux+1 count unchanged. Native diagnostic is not full SU4 gauge-projected source probability.'])

def run_floquet(p: Parameters) -> dict[str, Any]:
    """Discriminate native missing primitive cosets, never a macro promotion."""
    contract = charge_contract(p)
    gamma, native = gamma_frame(), []
    for order, cutoff in zip(p.orders, p.toron_cutoffs, strict=True):
        charged = native_sector(gamma, p, contract['bare_e'], order, 5, toron_cutoff=cutoff)
        vacuum = native_sector(gamma, p, contract['bare_e'], order, 4, toron_cutoff=cutoff)
        charged['source_preparation'] = dressed_source_price(charged, vacuum, p, int(contract['Q']))
        charged['dressed_mass_gap_lattice'] = vacuum['log_ground_transfer']-charged['log_ground_transfer']
        charged['dressed_mass_gap_physical'] = charged['dressed_mass_gap_lattice']/p.spacing
        native.append(dict(charged=charged, vacuum=vacuum))
    return dict(parameters=asdict(p), charge_contract=contract,
        native_positive_transfer=native, truncation_sequence_probability_differences=[],
        material_discriminator='Class0 omits the split E1=-Q/2 for evenQ. Select a genuine integer residue r, keep the SAME positive transfer and all retained winding, and compare charged/vacuum in the SAME residue.',
        limit='Native induced-Wilson-reference control only. Auxiliary charge1 can mix ALL primitive classes in the true macro law; no native->macro source/mass/probability shortcut. Native Ref0/externalcompensator is NOT the new macro RefQ contract.')


def bounded_integer_rounding(edges: list[tuple[int, int, int]], optimum: list[Any],
                             rho: list[int]) -> list[int]:
    """Unit-capacity integral circulation rounds a rational feasible flow."""
    size = len(rho)
    source, sink = size, size+1
    graph: list[list[list[int]]] = [[] for _ in range(size+2)]
    def edge(a: int, b: int, capacity: int) -> tuple[int, int]:
        forward = [b, len(graph[b]), capacity]
        reverse = [a, len(graph[a]), 0]
        graph[a].append(forward);graph[b].append(reverse)
        return a, len(graph[a])-1
    floor = [int(x.p//x.q) for x in optimum]
    divergence = [0]*size
    positions = []
    for (a, b, _), value in zip(edges, floor, strict=True):
        divergence[a] += value;divergence[b] -= value
        positions.append(edge(a, b, 1))
    supply = 0
    for a in range(size):
        residual = rho[a]-divergence[a]
        if residual > 0:
            edge(source, a, residual);supply += residual
        elif residual < 0:
            edge(a, sink, -residual)
    flow = 0
    while flow < supply:
        parent: list[tuple[int, int] | None] = [None]*(size+2)
        queue = [source]
        for a in queue:
            for j, (b, _, capacity) in enumerate(graph[a]):
                if capacity and b != source and parent[b] is None:
                    parent[b] = a, j
                    queue.append(b)
            if parent[sink] is not None:
                break
        if parent[sink] is None:
            raise ArithmeticError('Fractional feasible flow lost integral circulation')
        b = sink
        while b != source:
            a, j = cast(tuple[int, int], parent[b])
            target, reverse, _ = graph[a][j]
            graph[a][j][2] -= 1;graph[target][reverse][2] += 1
            b = a
        flow += 1
    rounded = [value+1-graph[a][j][2] for value, (a, j) in zip(floor, positions, strict=True)]
    divergence = [0]*size
    for (a, b, _), value in zip(edges, rounded, strict=True):
        divergence[a] += value;divergence[b] -= value
    if divergence != rho or any(abs(y-x) > 1 for x, y in zip(optimum, rounded, strict=True)):
        raise ArithmeticError('Integral circulation violates Gauss or edge bound')
    return rounded


def literal_integer_flow(p: Parameters) -> dict[str, Any]:
    """Exact rational native discriminator plus a uniform full-family bound."""
    import sympy as sp
    contract = charge_contract(p)
    q = int(contract['Q'])
    length = 4
    sites = list(itertools.product(range(length), repeat=3))
    lookup = {x: i for i, x in enumerate(sites)}
    edges = []
    laplacian = sp.zeros(len(sites))
    for i, x in enumerate(sites):
        for axis in range(3):
            y = list(x);y[axis] = (y[axis]+1) % length
            j = lookup[tuple(y)]
            edges.append((i, j, axis))
            laplacian[i, i] += 1;laplacian[j, j] += 1
            laplacian[i, j] -= 1;laplacian[j, i] -= 1
    inverse = laplacian[1:, 1:].inv(method='DM')
    edge_lookup = {(a, b): i for i, (a, b, _) in enumerate(edges)}
    parent = {}
    for i, x in enumerate(sites):
        if i:
            axis = next(j for j in range(3) if x[j])
            y = list(x);y[axis] -= 1
            parent[i] = lookup[tuple(y)]
    native = []
    for aux in (0, 7):
        rho = [0]*len(sites)
        rho[lookup[(0, 0, 0)]] += q
        rho[lookup[(2, 1, 1)]] -= q
        rho[lookup[(1, 2, 1)]] += aux
        rho[lookup[(3, 1, 2)]] -= aux
        potential = [sp.Rational(0)]+list(inverse*sp.Matrix(rho[1:]))
        optimum = [potential[a]-potential[b] for a, b, _ in edges]
        balanced = bounded_integer_rounding(edges, optimum, rho)
        balanced_increment = sum(sp.Rational(y)*y-x*x for x, y in zip(optimum, balanced, strict=True))
        if balanced_increment < 0 or balanced_increment > len(edges):
            raise ArithmeticError('Bounded rounding energy increment failed')
        rounded = [int((2*x.p+x.q)//(2*x.q)) for x in optimum]
        divergence = [0]*len(sites)
        for (a, b, _), value in zip(edges, rounded, strict=True):
            divergence[a] += value;divergence[b] -= value
        residual = [rho[i]-divergence[i] for i in range(len(sites))]
        residual_max = max(abs(x) for x in residual)
        corrected = rounded.copy()
        for i in sorted(parent, key=lambda j: sum(sites[j]), reverse=True):
            a = parent[i]
            corrected[edge_lookup[(a, i)]] -= residual[i]
            residual[a] += residual[i]
        actual = [0]*len(sites)
        for (a, b, _), value in zip(edges, corrected, strict=True):
            actual[a] += value;actual[b] -= value
        defect = max(abs(x-y) for x, y in zip(actual, rho, strict=True))
        minimum = sum(x*x for x in optimum)
        increment = sp.Rational(sum(x*x for x in corrected))-minimum
        orthogonality = sum(x*(y-x) for x, y in zip(optimum, corrected, strict=True))
        if defect or orthogonality or increment < 0 or residual_max > 3:
            raise ArithmeticError('Exact integer-flow/Coulomb orthogonality failed')
        with mp.workdps(180):
            primitive_angles = [mp.mpf(i % 11-5)*mp.mpf('1e-75') for i in range(len(edges))]
            phase = mp.mpf(0);integer_character = mp.mpc(1)
            for value, angle in zip(corrected, primitive_angles, strict=True):
                phase += value*angle
                integer_character *= mp.exp(mp.j*angle)**value
            character_defect = str(abs(integer_character-mp.exp(mp.j*phase)))
            first = corrected[0]
            periodic_defect = str(abs(mp.exp(mp.j*(primitive_angles[0]+2*mp.pi))**first
                                     -mp.exp(mp.j*primitive_angles[0])**first))
        native.append(dict(spatial_size=length, total_integer_charge=str(sum(rho)),
            active_charge=str(q), Ref_operator_charge=str(q), Ref_hole_charge=str(-q),
            extra_auxiliary_charge1_pairs=aux,
            exact_integer_Gauss_defect=str(defect),
            maximum_rounding_divergence_residual=residual_max,
            exact_Coulomb_cycle_orthogonality=str(orthogonality),
            exact_squared_flow_energy_increment=str(increment),
            bounded_rounding_squared_energy_increment_exact=str(balanced_increment),
            bounded_rounding_squared_error_upper=len(edges),
            bounded_rounding_integer_fluxes=[str(x) for x in balanced],
            universal_tree_squared_error_upper=3*len(sites)*(3*len(sites)+.5)**2,
            literal_integer_character_defect=character_defect,
            literal_primitive_2pi_periodicity_defect=periodic_defect,
            integer_edge_fluxes=[str(x) for x in corrected],
            scope='Exact finite-graph kinematic source receipt; NOT the microscopic auxiliary photon law at L4 or a thermal admission promotion.'))
    volume, beta, L = 2**72, 10.**p.beta_power, 2**24
    tree_error = 3*volume*(3*volume+.5)**2
    V4 = volume*2**20
    field_bound = 2e20
    m0 = math.expm1(p.spacing*p.mass)
    determinant_ratio_log = 4*V4*math.log((m0+16)/m0)
    qmin = 2*math.sin(math.pi/L)
    coth = 1/math.tanh(qmin*2**20/2)
    prior_tail_log = math.log(32*V4)+18*math.log(field_bound*math.sqrt(qmin/coth))-qmin*field_bound**2/(128*coth)
    active_weighted_tail_log10 = (prior_tail_log+determinant_ratio_log)/math.log(10)
    compact_action_error = 64*V4*field_bound**4/beta
    return dict(parameters=asdict(p), charge_contract=dict(contract,
        reference_primitive_charge=str(q),
        neutralizing_background='NEW RefoperatorQ, neutral Ref4 background4Q; no external compensator'),
        exact_native_integer_receipts=native, runtime_sympy=sp.__version__,
        full_family_all_rho_operator='For EVERY actual active/auxiliary/Ref integer rho with sumrho0, solve E*=gradG rho, round EACH edge, then repair residual divergence by oriented tree subtree flows. W=exp(i sum Eint_e A_e) is a literal compact character. Source is the difference Eint(rho+Q(delta_x-delta_Ref))-Eint(rho), with density-controlled phases; all auxiliary rho components/canonical projectors remain.',
        all_rho_rounding_residual_per_vertex_upper=3,
        all_rho_tree_correction_per_edge_upper=3*volume,
        all_rho_tree_squared_flow_error_upper=tree_error,
        all_rho_tree_energy_increment_physical_upper=tree_error/(2*beta*p.spacing),
        proof='Each edge rounding error<=1/2, degree6 gives residual|div|<=3; tree correction is a subtree sum<=3V. Corrected-minus-optimal is divergence-free, so gradG rho is exactly orthogonal to it. Energy increment is ||Delta||²/(2 beta a), independent of how large actual auxiliary rho is. No Q1 ledger is reused.',
        bounded_rounding_improvement='Incidence is totally unimodular: residual integer divergence has a feasible fractional flow in capacities0..1 (edge fractions), hence an integral feasible flow. Executed native unit-capacity max-flow gives |Eint-E*|<=1 peredge and squared error<=3V. Three primitive toron-cycle adjustments make |sum_axis E/L|<=1/2 without pinning theta; the adjustment stays literal integer.',
        bounded_rounding_energy_increment_physical_upper=3*volume/(2*beta*p.spacing),
        harmonic_adjusted_squared_error_upper=6*volume+6*L*(3*volume/L+1)**2,
        harmonic_adjusted_energy_increment_physical_upper=(6*volume+6*L*(3*volume/L+1)**2)/(2*beta*p.spacing),
        harmonic_adjusted_coexact_phase_error_on_canonical_chart_upper=32*volume*2e20/math.sqrt(beta),
        physical_Q_nonlinear_D_chart_density_ratio_log_upper=determinant_ratio_log,
        actual_active_D_global_chart_tail_log10_upper=active_weighted_tail_log10,
        compact_prior_chart_action_error_upper=compact_action_error,
        strong_chart_proof='For m0>0 the exact unitary-link Wilson D has Hermitian part>=m0 and ||D||<=m0+16. Thus determinant max/min<=exp[4V4 log((m0+16)/m0)], independent ofQ; retain D(U^Q) NONLINEARLY on the entire chart. Multiply the supplied thermal induced-Gaussian prior tail, not a small-Qtheta perturbation. Compact plaquette quartic price<=64 V4 B^4/beta. This Euclidean field-density result is conditional on the induced-photon prior supplier, not a source-Tmin or real-time scattering claim.',
        chart_assumption='This last phase bound is CONDITIONAL on |sqrt(beta) A_coexact,e|<=2e20. Torons are handled separately with all-label Bloch/harmonic wavefunctions; never substitute that chart for Haar torons.',
        normalization_limit='Literal integer source and uniform all-rho Coulomb rounding price, NOT the actual interacting charged-halfT Gram. Positive canonical full thermal/Gauss preparation must still be retained.')


def macro_toron_source(p: Parameters, contract: dict[str, Any]) -> dict[str, Any]:
    """Large-box kinetic-momentum source, with every compact Floquet class.

    This evaluates the harmonic-background occupied minor, not the interacting
    all-coexact source Gram. Neither its charge nor its Ref law is the native one.
    """
    L, Nt, volume = 2**24, 2**20, 2**72
    q, e = int(contract['Q']), contract['bare_e']
    m0 = math.expm1(p.spacing*p.mass)
    kappa = 1/(2*(4+m0))
    sigma = .001
    results = []
    for order in (8, 12, 16):
        node, weight = hermgauss(order)
        momenta = np.array(list(itertools.product(math.sqrt(2)*sigma*node, repeat=3)))
        weights = np.array([math.prod(x) for x in itertools.product(weight/math.sqrt(math.pi), repeat=3)])
        b = 1+m0+np.sum(1-np.cos(momenta), axis=1)
        energy = np.arccosh((1+b*b+np.sum(np.sin(momenta)**2, axis=1))/(2*b))
        gram = float(weights@np.exp(-energy))
        results.append(dict(order=order, charged_half_transfer_Gram=gram,
                            positive_source_energy_lattice=float(weights@energy),
                            energy_variance_lattice=float(weights@energy**2-(weights@energy)**2),
                            detB_log_prefactor_per_momentum=(2*np.log(b)).tolist()))
    # Signed bare Wilson B>=1+m0>0; retain actual signed matrix/source bands.
    # Bounds below pay the actual B floor, not an invalid B>=1 assumption.
    b_floor = min(1., 1+m0)
    derivative_factor = b_floor**-3
    second_factor = b_floor**-4
    gap = 2*math.sinh(abs(p.spacing*p.mass))
    carrier_distance = round(1/p.spacing)
    derivative_squared = 3*(1/(4*sigma*sigma)+512*derivative_factor**2/gap**2)+carrier_distance**2
    second_derivative_squared = 3*(1/sigma**4+(64*second_factor/gap+1024*derivative_factor**2/gap**2)**2
                                  +4096*derivative_factor**2/(sigma*sigma*gap*gap))+carrier_distance**4
    toron_energy = e*e*derivative_squared/(2*volume)
    commutator_error = Nt**2*(e**4*second_derivative_squared/(2*volume**2)
                              +2*e*e*derivative_squared/(volume*Nt))
    fractions = [(j, denominator) for denominator in (7, 11, 13) for j in range(denominator)]
    relabel = []
    for j, denominator in fractions:
        quotient, residue = divmod(q*j, denominator)
        relabel.append(dict(theta_over_2pi=f'{j}/{denominator}',
                            integer_Bloch_offset=str(-quotient),
                            fractional_kinetic_grid_shift=f'{residue}/{denominator}',
                            Floquet_residue_mod_Q=str((-quotient) % q),
                            large_gauge_relabel_defect=str((-quotient-q)+q-(-quotient))))
    if 8*kappa < 1:
        winding_log = (math.log10(8*volume*Nt)+L*math.log10(8*kappa)
                       -math.log10(L)-math.log10(1-8*kappa))
        winding_proof = 'Actual positive-bare free harmonic determinant: convergent8k hopping series.'
    else:
        strip = abs(m0)/100
        logarithm_bound = 4*(math.log(2*(abs(m0)+17)/abs(m0))+math.pi)
        winding_log = math.log10(32*volume*Nt*logarithm_bound)-strip*L/math.log(10)
        winding_proof = 'FREE harmonic background ONLY: DdagD=m0²+2(1+m0)sum x_mu+2sum_mu<nu x_mu x_nu>=m0² for m0>-1. Complex shifts |Im p_mu|<=|m0|/100 perturb D by<=8(exp(strip)-1)<|m0|/2. Analytic logdet Fourier images give32VNt Clog exp(-strip L), uniformly over every EM/statistical toron. This replaces free spatial-winding control, NOT interacting8k/accretiveD admission.'
    thermal_log = math.log10(4*volume)-Nt*abs(p.spacing*p.mass)/math.log(10)
    alias_exponent = .5*(L*sigma)**2
    return dict(spatial_size=L, time_slices=Nt,
        Ref_operator_charge=str(q), Ref_initial_background=str(4*q),
        Ref_initial_occupation=4, Ref_after_source_occupation=3,
        body_charge=str(q), Ref_hole_charge=str(-q), external_compensator=False,
        toron_density='Haar before source, all three coordinates theta_j=L A_hj; no coordinate pinning',
        wavefunction='f_n(theta)=periodized exp[-|((2pi n+Q theta)/L)-p_c|²/(4 sigma²)], n in (Z/LZ)^3; all labels, with n->n-Q on theta->theta+2pi',
        statistical_histories='ALL8 Pauli torons retained independently; shift pi*epsilon/L in kinetic momentum',
        momentum_width_lattice=sigma, full_integer_Floquet_examples=relabel,
        normalized_gaussian_derivative_squared_upper=derivative_squared,
        toron_kinetic_energy_lattice_upper=toron_energy,
        toron_kinetic_energy_physical_upper=toron_energy/p.spacing,
        source_toron_thermal_commutator_error_squared_upper=commutator_error,
        source_toron_derivative_proof='T1 blocks:1/B,-F/B,B+FdagF/B; B>=b_floor=min(1,1+m0)>0 and ||F||<=sqrt3 give ||T1prime||<=11/b_floor³, ||T1second||<=32/b_floor⁴. Spectral projector derivatives pay actual positive eigen-gap2sinh(|aM|), while signed m0 and actual filled/empty bands are NOT replaced. Ref-scalar source Pi_empty/sqrt2. Thermal ptheta²<=beta L/Nt by Poisson theta identity. Purified thermal commutator controls relative source clock.',
        toron_kinetic_derivation='H_h=-sum d_theta²/(2 beta L). d_theta f=(Q/L)d_p f, hence price=e² ||d_p f||²/(2 L³), including band-frame derivatives. No L2 Qtheta pin or huge incoming-state variance.',
        momentum_alias_amplitude_upper_log10=(math.log(12)-alias_exponent)/math.log(10),
        active_full_thermal_det_spatial_winding_log10_upper=winding_log,
        active_free_spatial_winding_proof=winding_proof,
        active_thermal_population_log10_upper=thermal_log,
        nonlinear_harmonic_determinant='det D(U^Q)=product_p B(p)^(2 Nt) exp(2 Nt E(p)) (1+exp(-Nt E(p)))^4; shift p=(2pi n+Qtheta)/L. This includes detB and the active filled sea once, not a quenched source.',
        positive_transfer='T1=Ldag diag(Bplus^-1,Bminus)L; eigenvalues exp(+-E), each twice. Canonical Nactive=2V and exact thermal occupied minors.',
        harmonic_source_Gram_sequence=results,
        harmonic_source_Gram_quadrature_difference=abs(results[-1]['charged_half_transfer_Gram']-results[-2]['charged_half_transfer_Gram']),
        numerical_limit='Hermite differences are observed numerical convergence, NOT rigorous integration error. Harmonic-background source Gram is not the full interacting charged-T source admission.')


def macro_full_label_scalar_C(p: Parameters, contract: dict[str, Any]) -> dict[str, Any]:
    """Execute coherent Gaussian/Wilson source covariance, without Q1 absorption."""
    L, sigma, q = 2**24, .001, int(contract['Q'])
    step, radius = 2*math.pi/L, math.ceil(16*.001*L/(2*math.pi))
    positions = np.array([[0, 0, 0], [100, 0, 0], [L-100, 0, 0], [0, L-100, 100]], dtype=np.int64)
    reference = np.array([100, 0, 0], dtype=np.int64)
    displacements = (positions-reference+L//2) % L-L//2
    seams = (displacements-(positions-reference))//L
    gamma = gamma_frame()
    C_H = -1j*gamma[1]@gamma[2]@gamma[3]@clifford_time_reversal(gamma)
    source_C = np.kron(np.eye(len(positions)), C_H)
    offsets = np.arange(-radius-1, radius+2, dtype=np.int64)
    centre_phases = np.array([0, 1, 2, 3])
    profiles = (((0, 1),)*3, ((1, 7), (3, 11), (5, 13)), ((6, 7), (8, 11), (12, 13)))
    def source(eta, profile, inverse=False, large_gauge=False, obsolete_three_modes=False):
        wave = np.ones(len(positions), complex)
        labels = []
        for axis, (numerator, denominator) in enumerate(profile):
            numerator = -numerator if inverse else numerator
            numerator += denominator if large_gauge else 0
            if obsolete_three_modes:
                centre, modes = 0, np.array([-1, 0, 1], dtype=np.int64)
            else:
                centre = (-2*q*numerator-eta[axis]*denominator)//(2*denominator)
                fractional = float(Fraction(2*denominator*centre+eta[axis]*denominator+2*q*numerator, 2*denominator))
                modes = offsets[abs(offsets+fractional) <= radius]
            kinetic = step*(modes+float(Fraction(2*denominator*centre+eta[axis]*denominator+2*q*numerator, 2*denominator)))
            weights = np.exp(-kinetic*kinetic/(4*sigma*sigma))
            # Reduce the LITERAL integer labels before floating exponentiation.
            twice_label = (2*(centre % L+modes)+eta[axis]) % (2*L)
            phase = math.pi/L*((twice_label[:, None]*positions[None, :, axis]) % (2*L))
            wave *= weights@np.exp(1j*phase)/math.sqrt(L*float(weights@weights))
            labels.append(str(centre))
        wilson = []
        for point in range(len(positions)):
            phase = sum((Fraction(q*((-n if inverse else n)+(d if large_gauge else 0))*int(displacements[point, axis]), d*L)
                         for axis, (n, d) in enumerate(profile)), Fraction())
            fraction = Fraction(phase.numerator % phase.denominator, phase.denominator)
            colour = (-1 if inverse else 1)*centre_phases[point]/4
            sign = (-1.)**int(np.dot(eta, seams[point]))
            wilson.append(sign*np.exp(2j*math.pi*(float(fraction)+colour)))
        wave *= np.array(wilson)
        return np.kron(np.outer(wave, wave.conj()), np.eye(4)), labels
    records = []
    for eta in itertools.product((0, 1), repeat=3):
        for profile in profiles:
            actual, labels = source(eta, profile)
            inverse, _ = source(eta, profile, inverse=True)
            gauge, shifted_labels = source(eta, profile, large_gauge=True)
            scale = np.linalg.norm(actual)
            defect = float(np.linalg.norm(source_C@inverse.T@source_C.conj().T-actual)/scale)
            gauge_defect = float(np.linalg.norm(gauge-actual)/scale)
            if max(defect, gauge_defect) > 2e-11:
                raise ArithmeticError('Actual coherent full-label Q/Wilson source covariance failed')
            cross = actual[0, 4]/math.sqrt(float(actual[0, 0].real*actual[4, 4].real))
            records.append(dict(statistical_history=eta, rational_torons=profile,
                exact_integer_Floquet_centres=labels, large_gauge_centres=shifted_labels,
                coherent_full_source_C_matrix_relative_defect=defect,
                coherent_source_large_gauge_matrix_relative_defect=gauge_defect,
                actual_spatial_cross_phase=dict(real=float(cross.real), imag=float(cross.imag))))
    obsolete, _ = source((1, 0, 0), profiles[0], obsolete_three_modes=True)
    obsolete_inverse, _ = source((1, 0, 0), profiles[0], inverse=True, obsolete_three_modes=True)
    adverse = float(np.linalg.norm(source_C@obsolete_inverse.T@source_C.conj().T-obsolete)/np.linalg.norm(obsolete))
    u = (radius-1)*step/sigma
    tail = 2*(1+sigma/(step*u))*math.exp(-u*u/2)/(math.sqrt(2*math.pi)*sigma/step-2)
    return dict(active_integer_charge=str(q), momentum_width_lattice=sigma, spatial_size=L,
        records=records, one_dimensional_unperiodized_numeric_tail_squared_upper=tail,
        obsolete_unclosed_eta1_three_mode_matrix_covariance_defect=adverse,
        proof='ALL labels: k_n=(2π(n+eta/2)+Qtheta)/L. n→−n−eta at theta→−theta gives k→−k; n→n−Q at theta→theta+2π is an exact INTEGER relabel. Periodized Gaussian coefficients and ALL8 normalizers are even. Actual σ Wilson signs are REAL; U^Q and centre Wilson paths conjugate. Keep coherent W_x W_y†; common Ref orientation cancels but individual paths do NOT. In the ideal constant Clifford frame C_H A(τ)^T C_Hdag=A, A=ΨΨdag. Global canonical PH phases give SdagS=2−dΓ(A)/2 and C SdagS Cinv=2−SdagS, provided the FULL neutral kernel commutes C.',
        sampling='Four spatial evaluation points test the coherent operator coefficient, NOT a four-point replacement source. Numerical Fourier truncation has a reported Gaussian tail; the declared source remains periodized ALL modes, no preparation/receiver gate.',
        limits='Source-matrix covariance only. Full Root connection AND Phi-transfer compatibility must be fixed in the SAME coordinate convention; no full macro vacuum/CP normalization, Pe/charged-halfstep, pole or probability admission. The different three-mode preparedField recipe and Ref3 charged domain do NOT inherit this identity.')


def macro_integer_coulomb(p: Parameters, contract: dict[str, Any]) -> dict[str, Any]:
    """Price an admissible all-cycle Gaussian rather than a fractional character."""
    L, volume, Nt = 2**24, 2**72, 2**20
    e2 = contract['bare_e']**2
    distance = round(1/p.spacing)
    def integrand(t):
        i0 = float(ive(0, 2*t))
        return i0*i0*(i0-float(ive(distance, 2*t)))
    value, numerical_error = quad(integrand, 0, np.inf, epsabs=2e-11, epsrel=2e-10, limit=400)
    green0, green0_error = quad(lambda t: float(ive(0, 2*t))**3, 0, np.inf,
                               epsabs=2e-11, epsrel=2e-10, limit=400)
    self_energy = e2*green0/(2*p.spacing)
    mutual_energy = -e2*(green0-value)/p.spacing
    # Bound finite-torus images and both neutral-pair heat tails separately.
    c = math.sqrt(math.pi)/4
    image_distance = L-distance
    image_denominator = 1-math.exp(-3*image_distance**2/(8*L*L))
    image_error = 8*c*(1+c)**2*math.sqrt(8*math.pi)/(image_distance*image_denominator)
    image_error += 1e-300
    infinite_pair_tail = math.pi**3.5*distance**2/(1536*L**3)
    finite_pair_tail = (math.pi**2*distance**2*(1+c)**2/(4*L**3)
                        *math.exp(-16)/(1-math.exp(-48)))
    torus_error = e2*(image_error+infinite_pair_tail+finite_pair_tail)/p.spacing
    qmin = 2*math.sin(math.pi/L)
    # A fundamental integer-cycle matrix has at most 6V rows/columns and
    # entries 0,+-1: ||B||² <=(6V)². All harmonic directions are included.
    # The source shifts only the Poisson phase, not this absolute theta tail.
    dual_exponent = 2*math.pi**2*10.**p.beta_power*min(qmin, 1/Nt)/(36*volume**2)
    tail_log = (math.log(4*(2*volume+1))-dual_exponent)/math.log(10)
    return dict(reference_distance_physical=distance*p.spacing,
        integer_reference_distance_lattice=distance,
        source_pair='body+Q, static Refhole-Q; Ref operatorQ and initially neutral Ref4',
        exact_admissible_wavefunction='For EVERY integer occupation charge rho, choose integer E_tree with div E_tree=rho. Sum exp[-(E-E_C)^t Omega^-1(E-E_C)/(2 beta)] exp(i E.A) over ALL integer E=E_tree+B n, n in Z^(2V+1), including the three harmonic cycles. E_C=grad G rho is a coefficient centre, never a noninteger character.',
        Gauss_identity='div(E_tree+B n)=rho exactly; div B=0. No E divisibleQ restriction. Auxiliary charge1, activeQ and RefQ relative charges all enter rho.',
        cycle_rank=2*volume+1, cycle_basis_operator_norm_squared_upper=(6*volume)**2,
        minimum_scaled_electric_frequency=min(qmin, 1/Nt),
        Poisson_dual_minimum_exponent=dual_exponent,
        absolute_integer_periodization_tail_log10_upper=tail_log,
        Poisson_proof='Precision on cycle coefficients is B^t Omega^-1 B/beta, lambda_max<=(6V)^2/(beta omega_min). Dual Gaussian exponent>=2pi² beta omega_min/(6V)^2. The Z^d theta tail is <=4d exp(-exponent); rho shifts only phases. Toron thermal precision Nt/(beta L) is included, never frozen.',
        infinite_lattice_Coulomb_pair_energy_physical=e2*value/p.spacing,
        infinite_lattice_Green0=green0, infinite_lattice_GreenR=green0-value,
        body_Coulomb_self_energy_physical=self_energy,
        static_Refhole_Coulomb_self_energy_physical=self_energy,
        mutual_Coulomb_potential_energy_physical=mutual_energy,
        separated_self_energy_numeric_error=e2*green0_error/(2*p.spacing),
        source_clock_convention='Measured neutral-to-Ref3/body source clock includes BOTH Coulomb self terms plus mutual potential. Calibrated static-Ref counterrotation removes only its common phase, not its Hamiltonian energy. These harmonic Coulomb terms are not separately added to the Feynman-gauge electron self-energy, which already contains its longitudinal contribution.',
        infinite_lattice_integral_numeric_error=e2*numerical_error/p.spacing,
        literal_integer_string_energy_physical=e2*distance/(2*p.spacing),
        finite_torus_analytic_image_error_physical_upper=torus_error,
        finite_torus_pair_energy_reference_interval=[e2*value/p.spacing-torus_error,
                                                    e2*value/p.spacing+torus_error],
        finite_torus_bound_proof='Exact image heat kernels K_L(d,t)=sum_n exp(-2t) I_(d+nL)(2t). Shifted-contour saddle gives exponent>=min(n²/(8t),|n|/4) and prefactor min(1,sqrt(pi)/(4sqrt(t))). For t<=L², K_L0<=(1+c)/sqrt(t), images at least n(L-R), geometric image tail denominator>=1-exp[-3(L-R)²/(8L²)], integral t^-3/2 exp[-(L-R)²/(8t)]<=sqrt(8pi)/(L-R). Both neutral-pair tails beyondL² are bounded separately by Fourier 1-cos(Rk)<=R²k²/2. Bounds quoted separately from numerical infinite-integral error.',
        finite_torus_energy_formula='e²/(a L³) sum_{k!=0} (1-cos(k.R))/sum_j 4sin²(k_j/2)',
        energy_limit='Finite-torus analytic interval uses the separately reported infinite-integral numerical quadrature value/error. Includes body AND Ref self-energy; no subtraction or fit to mass1. Gaussian norm/cycle price is not an interacting unquenched-source admission theorem.')


def locked_native_scalar_source(p: Parameters, contract: dict[str, Any],
                                critical: dict[str, Any]) -> dict[str, Any]:
    """Execute primitive Ref4 scalar CAR on an unquenched positive native fibre."""
    parameters = Parameters(**dict(asdict(p), mass=critical['declared_critical_bare_mass_parameter']))
    gamma, e = gamma_frame(), contract['bare_e']
    _, beta_basis = np.linalg.eigh(gamma[0])
    plus, minus = np.kron(np.eye(2), beta_basis[:, 2:]), np.kron(np.eye(2), beta_basis[:, :2])
    change = np.column_stack((plus, minus))
    spin_R = -1j*gamma[1]@gamma[2]@gamma[3]
    R = change.conj().T@np.kron(np.eye(2), spin_R)@change
    time_reversal = clifford_time_reversal(gamma)
    canonical_TR = change.conj().T@np.kron(np.eye(2), time_reversal)@change.conj()
    charge_conjugation = R@canonical_TR
    _, vacuum_masks = second_quantized(np.eye(8), 4)
    _, charged_masks = second_quantized(np.eye(8), 5)
    vacuum_lookup = {mask: i for i, mask in enumerate(vacuum_masks)}
    charged_lookup = {mask: i for i, mask in enumerate(charged_masks)}
    occupied = [[j for j in range(8) if mask & (1 << j)] for mask in vacuum_masks]
    Gamma_R = np.array([[np.linalg.det(R[np.ix_(row, column)]) for column in occupied] for row in occupied])
    PH = np.zeros((70, 70), complex)
    for column, original in enumerate(vacuum_masks):
        mask, sign = original, 1
        for bit in range(7, -1, -1):
            sign *= (-1)**((mask & ((1 << bit)-1)).bit_count())
            mask ^= 1 << bit
        PH[vacuum_lookup[mask], column] = sign
    CAR_C = Gamma_R@PH
    Gamma_charge_C = np.array([[np.linalg.det(charge_conjugation[np.ix_(row, column)])
                               for column in occupied] for row in occupied])
    unitary_C = Gamma_charge_C@PH
    # Filled Ref4 leaves four orthogonal Ref3-hole channels; no rank-one spin address.
    spatial_source = change.conj().T@np.vstack((np.eye(4), np.eye(4)))/2
    sources = np.zeros((4, 56, 70), complex)
    for column, mask in enumerate(vacuum_masks):
        for bit in range(8):
            if not mask & (1 << bit):
                row = charged_lookup[mask | (1 << bit)]
                sources[:, row, column] += (-1)**((mask & ((1 << bit)-1)).bit_count())*spatial_source[bit]
    records = []
    for order in (2, 3):
        omega = 2*math.asinh(1.)
        nodes, transform, annihilator = oscillator(order, omega)
        photon_basis = np.kron(transform, transform)
        photons = np.array([i+j for i in range(order) for j in range(order)])
        half_photon = photon_basis.T@np.diag(np.exp(-.5*omega*photons))@photon_basis
        torons, angles = np.arange(-1, 2), 2*math.pi*np.arange(24)/24
        characters = np.exp(-1j*np.outer(torons, angles))/math.sqrt(24)
        gamma_blocks, one_defect = {4: [], 5: []}, 0.
        minima = {4: math.inf, 5: math.inf}
        unitary_one_defect = 0.
        for ay, az in itertools.product(nodes, repeat=2):
            local_blocks = {4: [], 5: []}
            for angle in angles:
                phases = np.zeros((2, 3))
                phases[1, 0] = angle
                phases[:, 1] = e*ay*np.array([1., -1.])/math.sqrt(2)
                phases[:, 2] = e*az*np.array([1., -1.])/math.sqrt(2)
                T1, logdet_B = one_transfer(gamma, parameters, phases)
                one_defect = max(one_defect, float(np.linalg.norm(R@T1@R.conj().T-np.linalg.inv(T1))))
                reversed_T1, _ = one_transfer(gamma, parameters, -phases)
                unitary_one_defect = max(unitary_one_defect, float(np.linalg.norm(
                    charge_conjugation@np.linalg.inv(reversed_T1).T@charge_conjugation.conj().T-T1)))
                values, vectors = eigh(T1)
                logarithm = (vectors*np.log(values))@vectors.conj().T
                for number in (4, 5):
                    lifted, _ = second_quantized(logarithm, number)
                    spectrum, frame = eigh(lifted+logdet_B*np.eye(len(lifted)))
                    local_blocks[number].append((frame*np.exp(spectrum))@frame.conj().T)
                    minima[number] = min(minima[number], math.exp(float(spectrum[0])))
            for number in (4, 5):
                gamma_blocks[number].append(local_blocks[number])
        if one_defect > 2e-10:
            raise ArithmeticError('Actual locked R/transfer intertwining failed')
        actions, eigenpairs, lower = {}, {}, {}
        for number, masks in ((4, vacuum_masks), (5, charged_masks)):
            blocks, size = np.array(gamma_blocks[number]), len(masks)
            electric = integer_electric_energy(masks, e, torons, 0.)
            electric_half = np.exp(-.5*electric)
            dimension = order*order*3*size
            def apply(argument, blocks=blocks, size=size, electric_half=electric_half):
                state = argument.reshape(order*order, 3, size, -1)
                state = np.einsum('ab,bmfc->amfc', half_photon, state, optimize=True)
                state *= electric_half[None, :, :, None]
                angular = np.einsum('mq,amfc->aqfc', characters.conj(), state, optimize=True)
                angular = np.einsum('aqfg,aqgc->aqfc', blocks, angular, optimize=True)
                state = np.einsum('mq,aqfc->amfc', characters, angular, optimize=True)
                state *= electric_half[None, :, :, None]
                state = np.einsum('ab,bmfc->amfc', half_photon, state, optimize=True)
                return state.reshape(order*order*3*size, -1)
            actions[number] = apply
            matrix = np.empty((dimension, dimension), complex)
            for start in range(0, dimension, 32):
                stop = min(dimension, start+32)
                columns = np.zeros((dimension, stop-start))
                columns[np.arange(start, stop), np.arange(stop-start)] = 1.
                matrix[:, start:stop] = apply(columns)
            eigenpairs[number] = eigh((matrix+matrix.conj().T)/2)
            lower[number] = minima[number]*math.exp(-2*omega*(order-1)-float(np.max(electric)))
        def conjugate(state):
            tensor = state.reshape(order*order, 3, 70, -1)
            return np.einsum('ij,amjc->amic', CAR_C, tensor[:, ::-1].conj()).reshape(order*order*210, -1)
        photon_parity = transform.T@np.diag((-1.)**np.arange(order))@transform
        def unitary_conjugate(state):
            tensor = state.reshape(order, order, 3, 70, -1)[:, :, ::-1]
            return np.einsum('ai,bj,kl,ijmlc->abmkc', photon_parity, photon_parity,
                             unitary_C, tensor).reshape(order*order*210, -1)
        ground_values, ground_vectors = eigenpairs[4]
        ground = ground_vectors[:, -1:]
        probe = np.sin(np.arange(len(ground))*.37)[:, None]+1j*np.cos(np.arange(len(ground))*.13)[:, None]
        covariance = np.linalg.norm(actions[4](conjugate(probe))-conjugate(actions[4](probe)))
        covariance /= np.linalg.norm(actions[4](probe))
        if covariance > 2e-10:
            raise ArithmeticError('Actual native quantum C covariance failed')
        unitary_covariance = np.linalg.norm(actions[4](unitary_conjugate(probe))-unitary_conjugate(actions[4](probe)))
        unitary_covariance /= np.linalg.norm(actions[4](probe))
        naive_tensor = probe.reshape(order, order, 3, 70, -1)[::-1, ::-1, ::-1]
        naive_C = np.einsum('ij,abmjc->abmic', unitary_C, naive_tensor).reshape(len(probe), -1)
        acted_tensor = actions[4](probe).reshape(order, order, 3, 70, -1)[::-1, ::-1, ::-1]
        naive_after = np.einsum('ij,abmjc->abmic', unitary_C, acted_tensor).reshape(len(probe), -1)
        naive_covariance = float(np.linalg.norm(actions[4](naive_C)-naive_after)/np.linalg.norm(actions[4](probe)))
        if max(unitary_one_defect, unitary_covariance) > 2e-10:
            raise ArithmeticError('Actual UNITARY charge conjugation failed')
        creator = transform.T@annihilator.T@transform
        states = [('neutral_actual_ground', ground)]
        packet = np.einsum('ij,jkmc->ikmc', creator, ground.reshape(order, order, 210, -1)).reshape(len(ground), -1)
        packet /= np.linalg.norm(packet)
        states.append(('same_real_y_one_photon_excitation', packet))
        packet_z = np.einsum('ij,kjmc->kimc', creator, ground.reshape(order, order, 210, -1)).reshape(len(ground), -1)
        circular = packet+1j*packet_z/np.linalg.norm(packet_z)
        circular /= np.linalg.norm(circular)
        states.append(('complex_y_plus_i_z_photon_same_source', circular))
        half_prepared = ground_vectors@(np.sqrt(ground_values/ground_values[-1])[:, None]
                                        *(ground_vectors.conj().T@circular))
        half_photon_Gram = float(np.linalg.norm(half_prepared)**2)
        minimum_half_photon_Gram = lower[4]/ground_values[-1]
        if half_photon_Gram < minimum_half_photon_Gram*(1-2e-10):
            raise ArithmeticError('FULL positive halfstep photon norm violates actual transfer lower bound')
        half_prepared /= math.sqrt(half_photon_Gram)
        states.append(('same_positive_halfstep_complex_photon', half_prepared))
        state_records = []
        for name, state in states:
            acted = np.einsum('rfg,amgc->ramfc', sources, state.reshape(order*order, 3, 70, -1))
            acted = acted.transpose(1, 2, 3, 0, 4).reshape(order*order*168, -1)
            raw = float(np.linalg.norm(acted)**2)
            half = float(np.sum(np.real(acted.conj()*actions[5](acted)))/ground_values[-1])
            overlap = float(np.linalg.norm(eigenpairs[5][1][:, -2:].conj().T@acted)**2/raw)
            spectrum, frame = eigenpairs[5]
            energies = -np.log(spectrum/ground_values[-1])/p.spacing
            weights = np.sum(abs(frame.conj().T@acted)**2, axis=1)/raw
            mean = float(weights@energies)
            variance = float(weights@((energies-mean)**2))
            same_photon_reference = float(np.sum(np.real(state.conj()*actions[4](state))))
            unitary_acted = unitary_conjugate(state)
            unitary_overlap = np.vdot(state, unitary_acted)
            aligned = float(np.linalg.norm(unitary_acted-state*unitary_overlap/abs(unitary_overlap)))
            anti_overlap = np.vdot(state, conjugate(state))
            if name == 'same_positive_halfstep_complex_photon' and aligned > 2e-10:
                raise ArithmeticError('SAME full positive-halfstep photon state lost unitary C covariance')
            state_records.append(dict(state=name, actual_primitive_Ref4_scalar_raw_Gram=raw,
                charged_positive_halfT_Gram_relative_to_same_vacuum=half,
                same_prepared_photon_charge_halfT_Gram=half*ground_values[-1]/same_photon_reference,
                actual_source_energy_mean_physical=mean,
                actual_source_energy_standard_deviation_physical=math.sqrt(variance),
                actual_source_energy_spectral_normalization=float(np.sum(weights)),
                actual_prepared_density_unitary_C_trace_distance=2*aligned*math.sqrt(max(0., 1-aligned*aligned/4)),
                actual_prepared_density_antiPH_trace_distance=2*math.sqrt(max(0., 1-abs(anti_overlap)**2)),
                actual_charged_lowest_doublet_source_weight=overlap))
        records.append(dict(photon_order=order, native_neutral_dimension=len(ground),
            actual_signed_bare_mass=parameters.mass, source_states=state_records,
            actual_R_T1_R_inverse_defect=one_defect, actual_full_quantum_C_covariance_relative_defect=float(covariance),
            actual_unitary_charge_conjugation_T1_defect=unitary_one_defect,
            actual_unitary_charge_conjugation_quantum_covariance=float(unitary_covariance),
            naive_unsigned_DVR_coordinate_permutation_covariance_adverse=naive_covariance,
            actual_positive_transfer_lower=lower,
            actual_neutral_positive_halfstep_complex_photon_Gram=half_photon_Gram,
            native_positive_transfer_halfstep_photon_Gram_lower=minimum_half_photon_Gram,
            actual_native_interacting_charge_gap_physical=float(math.log(ground_values[-1]/eigenpairs[5][0][-1])/p.spacing),
            actual_neutral_thermal_excitation_log10_upper=math.log10(len(ground)-1)-2**20*math.log(ground_values[-1]/ground_values[-2])/math.log(10)))
    return dict(active_integer_charge=contract['Q'], Ref_operator_charge=contract['Q'],
        initial_Ref_number=4, relative_Ref_background_charge=0, records=records,
        source='S=Σsites,alpha f_site cdag_sitealpha Wsite,Ref^Q ralpha/sqrt2, f=(1,1)/sqrt2. In the exact integer-Gauss basis source atsite0 automatically shiftsE0 byQ; atsite1 it is local. Same toronm is retained, Ref3 hole contributes−Q, not an external apparatus.',
        C_proof='Locked spinR=−i gamma1 gamma2 gamma3 identifies plus/minus bases. R T1 Rdag=T1^-1 when Bplus=Bminus and Fdag=−F, including noncommuting spatial B/F. CAR antiunitary Gamma(R)PH K preserves detB Gamma(T1); K reverses integer toronm, PH reverses centred active rho. Ref4 is unchanged in the neutral vacuum. Scalar source SdagS maps to2−SdagS, so a C-invariant neutral state has rawGram1.',
        unitary_C_proof='Solve U_T gamma_mu* U_Tdag=gamma_mu in the ACTUAL selected frame. C_H=R U_T gives C_H T1(-A,zdag)^(-T) C_Hdag=T1(A,z) for EVERY integerQ; no centre absorption. Gamma(C_H)PH plus coordinate inversion is UNITARY, so a complex photon creation packet maps to MINUS itself, not its adjoint. Use the ACTUAL DVR parity transform.T diag((-1)^n) transform, not an unsigned coordinate-index reversal. SAME positive halfstep photon density is invariant in this native neutralRef4 domain.',
        domain_warning='Both tested C maps leave Ref4 fixed ONLY in the neutral block. On Ref3, C G Cinv=-G-2Q(nRef-4); they are NOT full charged-domain symmetries. Full PH on Ref would need Ref4↔0 and background4Q↔0, which is NOT performed. Full macro W/connection/filter/Source proof remains separate.',
        scope='Executed unquenched compact-Haar/native quantum-photon POSITIVE CONTROL at actual negativeκ; exact integer flux and filledRef4, not a full SU4/centre/Phi/macroscopic admission. Finite photon/toron/Haar compressions are stated, not an infinite truncation theorem. Real native photon excitation is C invariant; moving full-box hard packet and all-centre connection still require their own SAME-source C/P proof.')


def macro_packet_reference_energy(p: Parameters, contract: dict[str, Any],
                                  coulomb: dict[str, Any], critical: dict[str, Any],
                                  resummed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Price declared target/signed bands; include a skeleton ONLY if computed."""
    sigma, distance = .001, coulomb['integer_reference_distance_lattice']
    e2_over_a = contract['bare_e']**2/p.spacing
    scalar_green = math.erf(math.sqrt(2)*sigma*distance)/(4*math.pi*distance)
    Fourier_mass = (2*sigma*sigma/math.pi)**1.5
    lattice_error = math.pi**2*Fourier_mass/48
    reference_self = coulomb['static_Refhole_Coulomb_self_energy_physical']
    scalar_mutual = -e2_over_a*scalar_green
    L, cutoff = 2**24, 2**21
    max_distance = cutoff+distance
    c = math.sqrt(math.pi)/4
    image_distance = L-max_distance
    image = 8*c*(1+c)**2*math.sqrt(8*math.pi)/(image_distance*(1-math.exp(-3*image_distance**2/(8*L*L))))
    infinite_tail = math.pi**3.5*max_distance**2/(1536*L**3)
    finite_tail = math.pi**2*max_distance**2*(1+c)**2/(4*L**3)*math.exp(-16)/(1-math.exp(-48))
    torus_error = 3*e2_over_a*(image+infinite_tail+finite_tail)
    branches = []
    reference_masses = [('nominal_positive_reference', p.mass),
                        ('actual_critical_bare_reference', critical['declared_critical_bare_mass_parameter'])]
    if resummed is not None:
        reference_masses.insert(1, ('positive_bare_skeleton_reference', resummed['resummed_rest_mass_physical']))
    for name, mass in reference_masses:
        b0 = math.exp(p.spacing*mass)
        energy0 = abs(p.spacing*mass)
        band_metric = 3/(4*b0*b0*math.sinh(energy0)**2)
        spin_error = band_metric*math.pi**2*Fourier_mass/4
        error = e2_over_a*(lattice_error+spin_error)+coulomb['separated_self_energy_numeric_error']
        strip = abs(math.expm1(p.spacing*mass))/100
        analytic_Bfloor = min(1., b0)-(math.exp(strip)-1)
        analytic_T_defect = 11*strip*math.exp(strip)/analytic_Bfloor**3
        if b0 < math.sqrt(2)-1 or analytic_T_defect >= math.sinh(energy0)/2:
            raise ArithmeticError('Band packet analytic-strip/projector margin failed')
        packet_tail_log10 = math.log10(1024)+(strip*strip/(2*sigma*sigma)-strip*cutoff)/math.log(10)
        torus_packet_error = torus_error+4*e2_over_a*10.**packet_tail_log10
        branches.append(dict(branch=name, actual_signed_reference_mass=mass,
            infinite_lattice_band_projected_mutual_reference_interval=[scalar_mutual-error, scalar_mutual+error],
            source_clock_addition_beyond_dressed_body_pole_interval=[reference_self+scalar_mutual-error-torus_packet_error,
                                                                   reference_self+scalar_mutual+error+torus_packet_error],
            harmonic_pair_Coulomb_reference_interval=[2*reference_self+scalar_mutual-error,
                                                     2*reference_self+scalar_mutual+error],
            actual_band_projector_overlap_error_bound=spin_error))
        branches[-1].update(finite_torus_pair_reference_interval=[2*reference_self+scalar_mutual-error-torus_packet_error,
                                                                 2*reference_self+scalar_mutual+error+torus_packet_error],
            finite_torus_packet_image_error_bound=torus_packet_error,
            packet_position_tail_log10_upper=packet_tail_log10,
            executed_complex_T_perturbation_bound=analytic_T_defect,
            actual_minimum_T_eigen_gap=2*math.sinh(energy0))
    return dict(momentum_width_lattice=sigma, position_coordinate_standard_deviation_lattice=1/(2*sigma),
        reference_hole_position_relative_to_body_carrier_lattice=[distance, 0, 0],
        scalar_packet_convolved_Green_reference=scalar_green,
        scalar_packet_mutual_potential_reference_physical=scalar_mutual,
        static_Refhole_self_energy_physical=reference_self,
        scalar_lattice_Green_discretization_error_bound=lattice_error,
        Gaussian_Brillouin_alias_Green_error_log10_upper=math.log10(8)-math.pi**2/(8*sigma*sigma*math.log(10)),
        finite_torus_packet_proof='Choose coordinate window|x_j|<=L/8. The complex shiftdelta=|m0|/100 changes T1 by<=11delta exp(delta)/(b_floor-(exp(delta)-1))³<gap/4, so its rank2 Riesz projector norm<=2. Parseval for the shifted periodic Gaussian and absolute periodization yields tail<=1024 exp(delta²/(2sigma²)-delta L/8), uniform twisted Floquet/statistical phases. Inside the window, neutral-pair heat-image bound pays THREE directions and Rmax=L/8+Refdistance; tails pay3Rmax². Outside, |G_L|<=G_L0<1 by integer-shell Fourier bound, giving cost<=4e²/a times tail. Bodyself is counted once; all intervals retain separately reported numerical G0 error.',
        band_packet_definition='Canonical compensated Ref4 scalar-source amplitude is f(p)Pi_empty(p)/sqrt2, f normalized full3D Gaussian. Its density Fourier factor is exp(-k²/(8sigma²)) times Gaussian-p average[(1+n(p+k/2).n(p-k/2))/2], using the ACTUAL signed T1 band.',
        Green_proof='Continuum convolution is erf(sqrt2 sigma R)/(4pi R). For|k_j|<=pi, khat²>=4k²/pi² and k²-khat²<=k⁴/12, so reciprocal lattice-continuum error<=pi²/48 times the Gaussian Fourier mass. Brillouin Gaussian aliases/tails are exponentially small in1/sigma².',
        band_proof='T1=tI+v.Gamma. |nprime_k|²<=(Bprime²+|sprime|²)/(B² sinh²E)<=3|k|²/(b0² sinh²E0). Hence overlap defect<=3k²/(4b0² sinh²E0), and its Green integral<=that coefficient*pi²/4*GaussianFourierMass. E(p)>=E0 because b0>=sqrt2-1 for every exercised signed reference; no absolute-mass replacement inside T1.',
        branches=branches,
        clock_convention='A dressed electron pole already includes its own Coulomb self term within its full Feynman-gauge self-energy. Add ONLY calibrated staticRef self plus actual mutual reference potential, not bodyself again. Ref counterrotation changes common phase, NOT Hamiltonian energy.',
        scope='Harmonic signed-band packet reference bounds, not a measured full interacting source clock. Finite-torus body/Ref prices are renewed for the actual packet, not the pointR100 approximation; mixed auxiliary/active source correlations and all postcharge-Gram weights require their own same-kernel proof.')


def macro_active_loop(p: Parameters, contract: dict[str, Any], *,
                      external_momentum: float = .005) -> dict[str, Any]:
    """Actual Wilson det current/contact at chosen bare Q, resolving the mass."""
    m0, e2 = math.expm1(p.spacing*p.mass), contract['bare_e']**2
    external = external_momentum
    records = []
    for order in (6, 8, 10, 12):
        node, weights = leggauss(order)
        totals = np.zeros(4)
        outer = math.pi
        boxes = []
        while outer > abs(m0)/16:
            inner = outer/2
            for axis in range(4):
                low = np.zeros(4);high = np.full(4, outer)
                high[:axis] = inner;low[axis] = inner
                boxes.append((low, high))
            outer = inner
        boxes.append((np.zeros(4), np.full(4, outer)))
        for low, high in boxes:
            points = [low[j]+(node+1)*(high[j]-low[j])/2 for j in range(4)]
            wp = [weights*(high[j]-low[j])/2/math.pi for j in range(4)]
            momentum = np.array(np.meshgrid(*points, indexing='ij')).reshape(4, -1).T
            weight = np.prod(np.array(np.meshgrid(*wp, indexing='ij')), axis=0).reshape(-1)
            sine = np.sin(momentum);cosine = np.cos(momentum)
            b = m0+np.sum(1-cosine, axis=1)
            den = b*b+np.sum(sine*sine, axis=1)
            shifted = momentum.copy();shifted[:, 3] += external/2
            sp, cp = np.sin(shifted), np.cos(shifted)
            bp = m0+np.sum(1-cp, axis=1)
            shifted[:, 3] -= external
            sm, cm = np.sin(shifted), np.cos(shifted)
            bm = m0+np.sum(1-cm, axis=1)
            denom = (bp*bp+np.sum(sp*sp, axis=1))*(bm*bm+np.sum(sm*sm, axis=1))
            dot = np.sum(sp*sm, axis=1)
            for output, axis in ((0, 1), (2, 3)):
                d, c = sine[:, axis], cosine[:, axis]
                contact = 4*(b*c-sine[:, axis]**2)/den
                bubble0 = 4*(d*d*(b*b-np.sum(sine*sine, axis=1))
                              +4*b*d*c*sine[:, axis]
                              -c*c*(b*b+np.sum(sine*sine, axis=1)-2*sine[:, axis]**2))/den**2
                bubble = 4*(d*d*(bp*bm-dot)+2*d*c*(bp*sm[:, axis]+bm*sp[:, axis])
                             -c*c*(bp*bm+dot-2*sp[:, axis]*sm[:, axis]))/denom
                totals[output] += float(weight@(bubble-bubble0))
                totals[output+1] += float(weight@(bubble0-contact))
        khat2 = (2*math.sin(external/2))**2
        records.append(dict(order=order, boxes=len(boxes),
            transverse_inverse_photon_grade2=e2*totals[0]/khat2,
            longitudinal_Ward_grade2=e2*totals[2]/khat2,
            zero_momentum_current_contact_Ward=[e2*totals[1], e2*totals[3]],
            charge_current_scale=contract['bare_e'], contact_scale=e2))
    shift = records[-1]['transverse_inverse_photon_grade2']
    return dict(external_momentum_lattice=external, bare_mass_coefficient=m0,
        exact_D='D(p)=m0+sum_mu(1-cos p_mu)+i sum_mu gamma_mu sin p_mu; detD=(b²+sum sin²p)^2',
        current='Q/sqrt(beta) [sin p_i+i gamma_i cos p_i]',
        contact='Q²/beta [cos p_i-i gamma_i sin p_i]',
        determinant_Hessian='delta²(-logdetD)=Tr(G Dprime G Dprime)-Tr(G Dsecond). Exact finite-q Wilson vertices, NOT a continuum polarization imported as data.',
        integration='Positive-orthant nested cubes down to |m0|/16 with actual SIGNED Wilson b; every shell is partitioned into four disjoint boxes. Full4D Brillouin zone and actual Wilson doublers retained.',
        numerical_sequence=records,
        numerical_transverse_successive_difference=abs(shift-records[-2]['transverse_inverse_photon_grade2']),
        one_loop_finite_q_consumer_inverse_alpha=float(p.inverse_alpha)*(1+shift),
        stable_one_loop_photon_inverse_kernel=1+shift,
        normalization='The loop is the grade2 jet of the ACTIVE unquenched detD, not a second active species or a determinant multiplier of an already integrated Fock state.',
        limit='Executed one-loop determinant jet only. Numerical differences/longitudinal Ward residual are not all-orders truncation bounds or a measured renormalized alpha.')


def macro_charge_renormalization(p: Parameters, contract: dict[str, Any],
                                 loop: dict[str, Any], selfenergy: dict[str, Any]) -> dict[str, Any]:
    """Execute a joint ONE-LOOP charge/mass condition; never retag the input."""
    e0_squared = contract['bare_e']**2
    rows = [loop]+[macro_active_loop(p, contract, external_momentum=q)
                   for q in (.0025, .00125, .000625)]
    q2 = np.array([row['external_momentum_lattice']**2 for row in rows])
    coefficients = np.array([(row['stable_one_loop_photon_inverse_kernel']-1)/e0_squared for row in rows])
    coefficient = float(np.polyfit(q2[-3:], coefficients[-3:], 1)[1])
    two_point = float(np.polyfit(q2[-2:], coefficients[-2:], 1)[1])
    with localcontext() as context:
        context.prec = 110
        pi = Decimal('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651')
        beta = Decimal(10)**p.beta_power
        target_e2 = 4*pi/Decimal(p.inverse_alpha)
        chosen_e2 = target_e2/(1-target_e2*Decimal(str(coefficient)))
        target_Q = (beta*chosen_e2).sqrt()
        residue = int(contract['Q']) % 4
        new_Q = residue+4*int(((target_Q-residue)/4).to_integral_value())
        new_e2 = Decimal(new_Q*new_Q)/beta
        new_inverse_alpha = 4*pi/new_e2
        new_e = float(Decimal(new_Q)/beta.sqrt())
    renormalized_parameters = Parameters(**dict(asdict(p), inverse_alpha=str(new_inverse_alpha)))
    renormalized_contract = charge_contract(renormalized_parameters)
    actual_candidate_Q = int(renormalized_contract['Q'])
    if actual_candidate_Q != new_Q:
        raise ArithmeticError('candidate integer charge was not preserved by its declared bare alpha')
    actual_e2 = renormalized_contract['bare_e']**2
    sigma_target = selfenergy['numerical_sequence'][-1]['poles'][0]
    sigma_plus = sigma_target['on_shell_energy_shift_lattice']*sigma_target['pole_residue_E']
    candidate_sigma_plus = sigma_plus*actual_e2/e0_squared
    candidate_m0 = math.expm1(p.spacing*p.mass)-candidate_sigma_plus
    candidate_kappa = 1/(2*(4+candidate_m0))
    candidate_loop = macro_active_loop(renormalized_parameters, renormalized_contract,
                                        external_momentum=.000625)
    candidate_photon_inverse = 1+actual_e2*coefficient
    return dict(original_chosen_bare_charge=contract['bare_e'],
        original_chosen_bare_inverse_alpha=p.inverse_alpha,
        q_to_zero_reference_determinant_sequence=rows,
        extrapolated_reference_vacuum_polarization_coefficient=coefficient,
        observed_three_vs_two_q_extrapolation_difference=abs(coefficient-two_point),
        original_one_loop_dielectric_normalized_inverse_alpha=float(p.inverse_alpha)*(1+e0_squared*coefficient),
        declared_Thomson_inverse_alpha_target=p.inverse_alpha,
        charge_condition='eR²=e0²/Zgamma, Zgamma=1+e0² C(0). Choose e0²=eRtarget²/(1-eRtarget² C(0)); Q is the nearest integer in the ORIGINAL modulo4 class. This is a new declared microscopic fork, not a baseline relabel.',
        candidate_primitive_Q=str(new_Q), candidate_Q_mod4=new_Q % 4,
        candidate_canonical_bare_e=new_e,
        candidate_bare_inverse_alpha=str(new_inverse_alpha),
        candidate_reference_photon_inverse_kernel=candidate_photon_inverse,
        candidate_reference_photon_residue=1/candidate_photon_inverse,
        candidate_one_loop_current_normalized_inverse_alpha=4*math.pi*candidate_photon_inverse/actual_e2,
        executed_candidate_small_q_determinant=candidate_loop,
        candidate_Sigma_plus_at_target_mass_lattice=candidate_sigma_plus,
        candidate_bare_m0=candidate_m0,
        candidate_bare_mass_parameter=math.log1p(candidate_m0)/p.spacing,
        candidate_primitive_kappa=candidate_kappa,
        candidate_all_unitary_Wilson_B_margin=1-6*candidate_kappa,
        candidate_primitive_character=renormalized_contract['literal_compact_character_probe'],
        same_species_count=renormalized_contract['SU4_species_count'],
        current_normalization='Zero-momentum Ward vertex must be the derivative of the SAME dressed inverse fermion operator; its residue/current factors cancel, leaving e0 before photon normalization. This REQUIRED normalization is not replaced by a free-pole residue or a speed fit.',
        scope='Executed one-loop renormalized-reference photon/current condition and self-energy-derived bare-mass candidate. Full microscopic Thomson measurement, vertex/source Gram, nonlinear active remainder, physicalQ centre transport and higher-loop mass/charge corrections are NOT certified.')


def macro_local_linked_measure(p: Parameters, contract: dict[str, Any],
                               critical: dict[str, Any]) -> dict[str, Any]:
    """All-grade active determinant of ONE local link, not a quenched jet."""
    gamma = gamma_frame()[1]
    eye = np.eye(4)
    projector_plus, projector_minus = (eye+gamma)/2, (eye-gamma)/2
    e2 = contract['bare_e']**2
    phase_variance = e2/6
    records = []
    for name, m0 in (('positive_bare', math.expm1(p.spacing*p.mass)),
                     ('actual_critical_bare', critical['declared_critical_bare_m0'])):
        sequence = []
        for order in (6, 8, 10, 12):
            node, weights = leggauss(order)
            outer, boxes = math.pi, []
            while outer > abs(m0)/16:
                inner = outer/2
                for axis in range(4):
                    low, high = np.zeros(4), np.full(4, outer)
                    high[:axis] = inner;low[axis] = inner
                    boxes.append((low, high))
                outer = inner
            boxes.append((np.zeros(4), np.full(4, outer)))
            coefficients = np.zeros(3)
            for low, high in boxes:
                points = [low[j]+(node+1)*(high[j]-low[j])/2 for j in range(4)]
                wp = [weights*(high[j]-low[j])/2/math.pi for j in range(4)]
                k = np.array(np.meshgrid(*points, indexing='ij')).reshape(4, -1).T
                weight = np.prod(np.array(np.meshgrid(*wp, indexing='ij')), axis=0).reshape(-1)
                sine, cosine = np.sin(k), np.cos(k)
                b = m0+np.sum(1-cosine, axis=1)
                denominator = b*b+np.sum(sine*sine, axis=1)
                coefficients += [float(weight@(b/denominator)),
                                 float(weight@(cosine[:, 1]*b/denominator)),
                                 float(weight@(sine[:, 1]**2/denominator))]
            g0, g1, gs = coefficients
            coefficient = float(g1-gs)
            amplitude = 2*coefficient*(1+coefficient)
            gaussian_normalizer = ((1+amplitude)**2
                -2*amplitude*(1+amplitude)*math.exp(-phase_variance/2)
                +amplitude**2*(1+math.exp(-2*phase_variance))/2)
            second_moment = ((1+amplitude)**2
                -2*amplitude*(1+amplitude)*(1-phase_variance)*math.exp(-phase_variance/2)
                +amplitude**2*(1+(1-4*phase_variance)*math.exp(-2*phase_variance))/2)
            sequence.append(dict(order=order, Green0=float(g0),
                Green_neighbor_scalar=float(g1), Green_neighbor_gamma=float(gs),
                exact_linked_character_coefficient=coefficient,
                unquenched_Gaussian_conditional_normalizer=gaussian_normalizer,
                unquenched_conditional_canonical_link_variance=second_moment/gaussian_normalizer))
        green = np.block([[g0*eye, g1*eye-gs*gamma],
                         [g1*eye+gs*gamma, g0*eye]])
        probes = []
        for phase in (0., .1, .3, 1., math.pi):
            delta, first, second = [np.zeros((8, 8), complex) for _ in range(3)]
            plus_phase, minus_phase = np.exp(1j*phase), np.exp(-1j*phase)
            delta[:4, 4:] = -projector_minus*(plus_phase-1)
            delta[4:, :4] = -projector_plus*(minus_phase-1)
            first[:4, 4:] = -1j*projector_minus*plus_phase
            first[4:, :4] = 1j*projector_plus*minus_phase
            second[:4, 4:] = projector_minus*plus_phase
            second[4:, :4] = projector_plus*minus_phase
            linked = np.eye(8)+green@delta
            determinant = np.linalg.det(linked)
            current = np.linalg.solve(linked, green@first)
            contact = np.linalg.solve(linked, green@second)
            hessian = np.trace(contact-current@current)
            denominator = 1+amplitude*(1-math.cos(phase))
            exact_hessian = 2*amplitude*(math.cos(phase)*denominator
                                           -amplitude*math.sin(phase)**2)/denominator**2
            probes.append(dict(active_link_phase=phase,
                actual_rank8_unquenched_determinant_ratio=float(determinant.real),
                determinant_imaginary_defect=float(abs(determinant.imag)),
                exact_all_grade_character_ratio=denominator**2,
                actual_current=float(np.trace(current).real),
                actual_bubble_and_contact_Hessian=float(hessian.real),
                all_grade_Hessian_discrepancy=float(abs(hessian-exact_hessian))))
        strip = abs(m0)/100
        records.append(dict(branch=name, actual_m0=m0, all_momentum_Green_sequence=sequence,
            actual_nonperturbative_link_probes=probes,
            exact_all_phase_determinant_ratio_lower=min(1., (1+2*coefficient)**4),
            uniform_free_torus_Green_image_error_log10_upper=math.log10(64/abs(m0))-strip*(2**20)/math.log(10),
            normalizer_observed_quadrature_difference=abs(sequence[-1]['unquenched_Gaussian_conditional_normalizer']
                                                        -sequence[-2]['unquenched_Gaussian_conditional_normalizer'])))
    return dict(actual_active_species=1, active_integer_charge=contract['Q'],
        conditioning='ONE primitive spatial link varies; every other gauge link and Phi is fixed at the free Clifford background. This is an actual all-grade LOCAL CONDITIONAL of the induced-action reference, NOT a new physical prepared state or the full photon measure.',
        full_macro_resolvent='G0 integral over entire4D Wilson BZ, with actual signed m0. Finite8x8 determinant lemma det(D0+deltaD)/detD0=det(I8+G_R deltaD), no huge NF/Fock allocations, quenching or second active species.',
        exact_character_identity='For c=Gneighbor_scalar-Gneighbor_gamma, det ratio=[1+2c(1+c)(1-cos(Qtheta))]^2. Primitive Qtheta remains an INTEGER character at every angle. Current/Hessian use actual inverse rank8 matrix, including Dsecond contact.',
        derivative_coordinates='Matrix derivatives are with respect to active phase Qtheta; canonical A derivatives multiply current by e and Hessian by e². Auxiliaries are charge1, neverQ.',
        conditional_prior='NO bare gauge input: leading aux-induced beta plaquette action gives6beta cos(theta) for this local link. x=sqrt(6beta)theta has unit Gaussian reference; active phase=Qtheta=e x/sqrt6.',
        compact_to_Gaussian_error_log10_order=-p.beta_power+5,
        compact_error_proof='On|x|<=40 use x=2sqrt(6beta)sin(theta/2): Jacobian deviation<=40²/(12beta), active-phase deviation<=e40³/(144sqrt6 beta). Outside use the weighted Gaussian40-sigma tail; the all-grade determinant is globally bounded by the displayed character polynomial. Numerical BZ errors are separately observed, NOT absorbed in this asymptotic estimate.',
        toron_control='For actual free D0, DdagD>=m0²; shifts |Im p_mu|<=|m0|/100 remain invertible. Poisson images bounded by64/|m0| exp(-strip Nt), uniformly for ALL8 statistical twists and EM Floquet/Bloch labels. This is free-background control, not full interacting winding.',
        branches=records,
        limit='All-grade unquenched local normalization/current/contact probe. It does NOT integrate the other photon coordinates, provide the full charged-halfT/source Gram, measure renormalized alpha, or certify a finitepacket scattering probability.')


def macro_body_selfenergy(p: Parameters, contract: dict[str, Any], *,
                          external_spatial_momentum: float = .001) -> dict[str, Any]:
    """Wilson tadpole AND rainbow, projected on the actual on-shell pole."""
    m0, e2 = math.expm1(p.spacing*p.mass), contract['bare_e']**2
    if external_spatial_momentum <= 0:
        raise ValueError('External recoil momentum must be positive')
    records = []
    for order in (6, 8, 10, 12):
        node, weights = leggauss(order)
        outer, boxes = math.pi, []
        while outer > abs(m0)/16:
            inner = outer/2
            for axis in range(4):
                low = np.zeros(4);high = np.full(4, outer)
                high[:axis] = inner;low[axis] = inner
                boxes.append((low, high))
            outer = inner
        boxes.append((np.zeros(4), np.full(4, outer)))
        shifts = []
        for momentum_x in (0., external_spatial_momentum):
            momentum = np.array([0., momentum_x, 0., 0.], complex)
            b_spatial = 1+m0+1-math.cos(momentum_x)
            energy = math.acosh((1+b_spatial*b_spatial+math.sin(momentum_x)**2)/(2*b_spatial))
            momentum[0] = 1j*energy
            s_external = np.sin(momentum)
            b_pole = m0+np.sum(1-np.cos(momentum))
            residue = math.sinh(energy)*(1+math.cosh(energy)/b_pole.real)
            total = 0.
            tadpole_total = 0.
            contour_lower_margin = math.inf
            contour_upper_margin = math.inf
            for low, high in boxes:
                points = [low[j]+(node+1)*(high[j]-low[j])/2 for j in range(4)]
                wp = [weights*(high[j]-low[j])/2/math.pi for j in range(4)]
                k = np.array(np.meshgrid(*points, indexing='ij'), dtype=complex).reshape(4, -1).T
                weight = np.prod(np.array(np.meshgrid(*wp, indexing='ij')), axis=0).reshape(-1)
                photon_energy = 2*np.arcsinh(np.sqrt(np.sum((2*np.real(np.sin(k[:, 1:]/2)))**2, axis=1))/2)
                contour_height = photon_energy/2
                k[:, 0] += 1j*contour_height
                photon_den = np.sum((2*np.sin(k/2))**2, axis=1)
                projected_tadpole = .5*np.sum(np.cos(momentum)-s_external*s_external/b_pole)
                tadpole_total += float(weight@np.real(projected_tadpole/photon_den))
                for sign in (-1, 1):
                    signed_k = k.copy();signed_k[:, 1] *= sign
                    spatial_internal = np.real(momentum[None, 1:]-signed_k[:, 1:])
                    internal_B = 1+m0+np.sum(1-np.cos(spatial_internal), axis=1)
                    internal_E = np.arccosh((1+internal_B**2+np.sum(np.sin(spatial_internal)**2, axis=1))/(2*internal_B))
                    contour_lower_margin = min(contour_lower_margin, float(np.min(contour_height-energy+internal_E)))
                    contour_upper_margin = min(contour_upper_margin, float(np.min(energy+internal_E-contour_height)))
                    internal = momentum-signed_k
                    sine = np.sin(internal)
                    b = m0+np.sum(1-np.cos(internal), axis=1)
                    den = b*b+np.sum(sine*sine, axis=1)
                    d, c = np.sin(momentum-signed_k/2), np.cos(momentum-signed_k/2)
                    scalar = np.sum(b[:, None]*(d*d-c*c)+2*d*c*sine, axis=1)
                    vector = (-np.sum(d*d+c*c, axis=1)[:, None]*sine
                              +2*c*c*sine+2*b[:, None]*d*c)
                    projected_rainbow = (scalar+np.sum(vector*s_external/b_pole, axis=1))/den
                    total += .5*float(weight@np.real((projected_tadpole-projected_rainbow)/photon_den))
            if min(contour_lower_margin, contour_upper_margin) <= 0:
                raise ArithmeticError('on-shell contour crosses a fermion pole')
            delta_energy = e2*total/residue
            shifts.append(dict(momentum_x=momentum_x, free_pole_energy=energy,
                on_shell_energy_shift_lattice=delta_energy,
                tadpole_energy_shift_lattice=e2*tadpole_total/residue,
                pole_residue_E=residue,
                photon_pole_contour_fraction=.5,
                actual_quadrature_lower_fermion_pole_margin=contour_lower_margin,
                actual_quadrature_upper_fermion_pole_margin=contour_upper_margin,
                dressed_energy_one_loop_lattice=energy+delta_energy))
        curvature = 2*(shifts[1]['dressed_energy_one_loop_lattice']
                       -shifts[0]['dressed_energy_one_loop_lattice'])/external_spatial_momentum**2
        records.append(dict(order=order, poles=shifts,
            dressed_rest_mass_physical_one_loop=shifts[0]['dressed_energy_one_loop_lattice']/p.spacing,
            dressed_finite_recoil_secant_mass_physical_one_loop=1/(p.spacing*curvature)))
    return dict(numerical_sequence=records,
        rest_mass_successive_difference=abs(records[-1]['dressed_rest_mass_physical_one_loop']
                                           -records[-2]['dressed_rest_mass_physical_one_loop']),
        photon_gauge='Feynman gauge, ALL lattice momenta; this is a pole self-energy diagnostic, not a physical bare-source state.',
        integrand='deltaD=e² integral [sum_mu Dsecond_mu/2 - sum_mu V_mu G(p-k) V_mu]/khat²; V_mu=sin(p_mu-k_mu/2)+i gamma_mu cos(p_mu-k_mu/2). Left/right exact Wilson-pole projector and residue, not a rest-mass fitted constant.',
        temporal_contact_retained=True, selected_bare_mass=p.mass,
        causal_continuation='q0=u+i omega_photon(qsp)/2, strictly below the photon pole and between the actual fermion poles at i(Eexternal+-Einternal). Nonzero spatial external momentum otherwise moves a fermion pole across the real q0 contour. Evaluate the FULL complex photon denominator before taking real parts; signed spatial momenta retained.',
        counterterm_or_mass_fit=False,
        limit='One-loop on-shell evaluation and observed quadrature convergence only; infraparticle/all-orders mass and kinetic covariance not certified. No subtraction of93.7 or tuning this number back to1.')


def mass_conditioned_recoil_reference(p: Parameters, selfenergy: dict[str, Any]) -> dict[str, Any]:
    """Pay the SAME scalar counterterm at both actual external pole momenta."""
    records = []
    b0 = math.exp(p.spacing*p.mass)
    exact_Wilson_curvature = 2*b0*b0*math.sinh(p.spacing*p.mass)/(p.spacing*(b0*b0+2*b0-1))
    for row in selfenergy['numerical_sequence']:
        rest, moving = row['poles']
        scalar = rest['on_shell_energy_shift_lattice']*rest['pole_residue_E']
        energy = moving['free_pole_energy']+moving['on_shell_energy_shift_lattice']-scalar/moving['pole_residue_E']
        curvature = 2*(energy-rest['free_pole_energy'])/moving['momentum_x']**2
        records.append(dict(order=row['order'], conditioned_rest_reference=p.mass,
            finite_recoil_secant_mass_reference=1/(p.spacing*curvature),
            exact_Wilson_finite_recoil_secant_reference=moving['momentum_x']**2/(2*p.spacing*(moving['free_pole_energy']-rest['free_pole_energy'])),
            continuum_finite_recoil_secant_reference=(math.sqrt(p.mass*p.mass+(moving['momentum_x']/p.spacing)**2)+p.mass)/2,
            external_momentum_lattice=moving['momentum_x'],
            exact_Wilson_infinitesimal_kinetic_mass_reference=exact_Wilson_curvature,
            same_scalar_mass_counterterm=-scalar,
            actual_moving_residue=moving['pole_residue_E']))
    return dict(records=records,
        scope='One-loop SAME scalar mass condition, NO speed fit. Values are FINITE RECOIL SECANTS, not infinitesimal kinetic masses: even relativistic sqrt(M²+k²) gives (sqrt(M²+k²)+M)/2 at finite k. Compare exact Wilson recoil, then decrease momentum before inferring curvature. Actual momentum-dependent residue is paid; no constant energy subtraction.')


def macro_one_loop_Ward_current(p: Parameters, contract: dict[str, Any]) -> dict[str, Any]:
    """Independent triangle/contact diagrams versus self-energy pole derivative."""
    gamma, identity = gamma_frame(), np.eye(4)
    m0, energy, epsilon = math.expm1(p.spacing*p.mass), abs(p.spacing*p.mass), 1e-8
    if energy <= 0:
        raise ValueError('Ward on-shell projection needs a nonzero signed Wilson mass')
    chirality = 1. if m0 > 0 else -1.
    projector = (identity+chirality*gamma[0])/2
    free_current = chirality*math.exp(chirality*energy)
    def projected(matrix):
        return np.einsum('ij,nji->n', projector, matrix)/2
    records = []
    for order in (6, 8, 10, 12):
        node, weights = leggauss(order)
        outer, boxes = math.pi, []
        while outer > abs(m0)/16:
            inner = outer/2
            for axis in range(4):
                low, high = np.zeros(4), np.full(4, outer)
                high[:axis] = inner;low[axis] = inner
                boxes.append((low, high))
            outer = inner
        boxes.append((np.zeros(4), np.full(4, outer)))
        self_values = np.zeros(2, complex)
        triangle, contact, tadpole = 0j, 0j, 0j
        for low, high in boxes:
            points = [low[j]+(node+1)*(high[j]-low[j])/2 for j in range(4)]
            wp = [weights*(high[j]-low[j])/2/math.pi for j in range(4)]
            k = np.array(np.meshgrid(*points, indexing='ij'), complex).reshape(4, -1).T
            weight = np.prod(np.array(np.meshgrid(*wp, indexing='ij')), axis=0).reshape(-1)
            omega = 2*np.arcsinh(np.sqrt(np.sum((2*np.sin(np.real(k[:, 1:])/2))**2, axis=1))/2)
            k[:, 0] += 1j*omega/2
            photon_den = np.sum((2*np.sin(k/2))**2, axis=1)
            def data(p0):
                external = np.array([p0, 0., 0., 0.])
                internal = external-k
                sine = np.sin(internal)
                b = m0+np.sum(1-np.cos(internal), axis=1)
                den = b*b+np.sum(sine*sine, axis=1)
                G = (b[:, None, None]*identity-1j*np.einsum('nj,jab->nab', sine, gamma))/den[:, None, None]
                half = external-k/2
                V = [np.sin(half[:, j])[:, None, None]*identity+1j*np.cos(half[:, j])[:, None, None]*gamma[j]
                     for j in range(4)]
                return external, internal, G, V
            for index, shift in enumerate((-epsilon, epsilon)):
                external, _, G, V = data(1j*energy+shift)
                second = sum((np.cos(external[j])*identity-1j*np.sin(external[j])*gamma[j]
                              for j in range(4)), np.zeros((4, 4), complex))/2
                numerator = np.repeat(second[None], len(k), axis=0)
                for j in range(4):
                    numerator -= V[j]@G@V[j]
                self_values[index] += weight@(projected(numerator)/photon_den)
            external, internal, G, V = data(1j*energy)
            internal_current = np.sin(internal[:, 0])[:, None, None]*identity+1j*np.cos(internal[:, 0])[:, None, None]*gamma[0]
            propagated = G@internal_current@G
            tri = np.zeros_like(G)
            for j in range(4):
                tri += V[j]@propagated@V[j]
            second = np.cos((external-k/2)[:, 0])[:, None, None]*identity-1j*np.sin((external-k/2)[:, 0])[:, None, None]*gamma[0]
            con = -(second@G@V[0]+V[0]@G@second)
            external_current = np.sin(external[0])*identity+1j*np.cos(external[0])*gamma[0]
            triangle += weight@(projected(tri)/photon_den)
            contact += weight@(projected(con)/photon_den)
            tadpole += np.trace(projector@(-external_current/2))/2*(weight@(1/photon_den))
        e2 = contract['bare_e']**2
        finite = e2*(self_values[1]-self_values[0])/(2*epsilon)
        total = e2*(triangle+contact+tadpole)
        inverse_residue = float(free_current+finite.imag)
        records.append(dict(order=order,
            actual_triangle_temporal_current_correction=float(e2*triangle.imag),
            actual_two_photon_contact_current_correction=float(e2*contact.imag),
            actual_three_photon_tadpole_current_correction=float(e2*tadpole.imag),
            actual_total_temporal_current_correction=float(total.imag),
            independent_selfenergy_finite_difference_temporal_correction=float(finite.imag),
            three_point_Ward_defect=float(abs(total.imag-finite.imag)),
            same_dressed_inverse_pole_residue_temporal=inverse_residue,
            independently_residue_normalized_current=float((free_current+total.imag)/inverse_residue),
            bare_residue_wrong_normalized_current=float((free_current+total.imag)/free_current)))
    return dict(records=records, finite_difference_epsilon=epsilon,
        actual_signed_Wilson_mass_parameter=p.mass,
        actual_charged_rest_pole_beta_eigenvalue=chirality,
        signed_free_temporal_current_inverse_residue=free_current,
        diagrams='Rainbow triangle V G Dprime0 G V; contact−(Dsecond0 G V0+V0 G Dsecond0), NOhalf; tadpole−Dprime0/2. Actual full4D Wilson propagator and FULLcomplex photon denominator on q0=u+iomega/2. External p shifts both charged legs, so derivative of V(p−q/2) is Dsecond, not Dsecond/2.',
        independent_normalization='Pole residue comes from separate finite difference of selfenergy, not from the triangle result; current numerator is separately executed3point graphs including BOTH contacts.',
        IR_limit='Temporal Z/current graphs separately contain massless on-shell IR logarithms. Their quadrature-order drift is NOT a converged finite-volume residue. The cancellation/observed Ward defects refer to identical finite quadrature regulators; all-box discrete modes and Coulomb/source dressing must still be matched.',
        scope='Actual signed-bare OR declared positive-target one-loop on-shell reference, with signed m0 and ACTUAL rest chirality. Negative mass is NOT replaced inside D. Identical quadrature IR regulator only; not all-orders current, prepared source Gram or a measured microscopic Thomson alpha.')


def macro_selfconsistent_mass(p: Parameters, contract: dict[str, Any]) -> dict[str, Any]:
    """A genuine skeleton gap equation, not a fitted target-mass counterterm."""
    bare = math.expm1(p.spacing*p.mass)
    probes = []
    mass = p.mass
    result = {}
    for iteration in range(12):
        reference = Parameters(**dict(asdict(p), mass=mass))
        result = macro_body_selfenergy(reference, contract)
        pole = result['numerical_sequence'][-1]['poles'][0]
        scalar_shift = pole['on_shell_energy_shift_lattice']*pole['pole_residue_E']
        next_mass = math.log1p(bare+scalar_shift)/p.spacing
        probes.append(dict(iteration=iteration, internal_reference_mass=mass,
                           computed_scalar_selfenergy=scalar_shift,
                           next_mass=next_mass, fixed_point_defect=abs(next_mass-mass)))
        if abs(next_mass-mass) < 1e-8:
            mass = next_mass
            break
        mass = next_mass
    reference_bare = math.expm1(p.spacing*mass)
    scalar_shift = reference_bare-bare
    poles = result['numerical_sequence'][-1]['poles']
    # Db=Dref-(mref-mbare)I. The compensating term below is thus an
    # algebraic reorganization of the SAME computed skeleton, not a subtraction
    # of a native toy mass or a fit to a requested physical mass.
    energies = [x['free_pole_energy']+x['on_shell_energy_shift_lattice']
                -scalar_shift/x['pole_residue_E'] for x in poles]
    curvature = 2*(energies[1]-energies[0])/poles[1]['momentum_x']**2
    return dict(iteration_receipts=probes,
        gap_equation='exp(a Mref)-1=m0bare+Sigma_plus(Mref), with internal Wilson D(Mref) and its actual one-loop tadpole/rainbow; original m0bare fixed.',
        bare_mass_input=p.mass, resummed_rest_mass_physical=mass,
        resummed_finite_recoil_secant_mass_physical=1/(p.spacing*curvature),
        computed_scalar_mass_shift_lattice=scalar_shift,
        last_reference_selfenergy=result,
        residual_pole_energies_lattice=energies,
        converged=probes[-1]['fixed_point_defect']<1e-8,
        identity='Dbare=Dreference-(mreference-mbare)I. Residual dispersion uses [Sigma_projected(p)-(mreference-mbare)]/actual_pole_residue(p); the mass shift stays in Dreference.',
        material_repair='The additive first-order dispersion has negative kinetic curvature because its rest correction exceeds the bare mass. A self-consistent positive-mass skeleton retains that radiative mass instead of retuning to1.',
        limit='Executed one-loop self-consistent skeleton, not an exact microscopic gap equation, higher-loop error bound or interacting source normalization. No input mass or spectral identity is retagged.')


def critical_bare_mass_receipt(p: Parameters, contract: dict[str, Any],
                               selfenergy: dict[str, Any]) -> dict[str, Any]:
    """Declared critical bare choice from the actual macro self-energy."""
    pole = selfenergy['numerical_sequence'][-1]['poles'][0]
    sigma = pole['on_shell_energy_shift_lattice']*pole['pole_residue_E']
    critical = math.expm1(p.spacing*p.mass)-sigma
    if critical <= -1:
        raise ArithmeticError('Computed critical mass violates B positivity')
    kappa = 1/(2*(4+critical))
    gamma = gamma_frame()
    _, beta_basis = np.linalg.eigh(gamma[0])
    plus, minus = beta_basis[:, 2:], beta_basis[:, :2]
    records = []
    _, masks = second_quantized(np.eye(8), 6)
    source_matrix = np.zeros((8, 8), complex)
    for i in range(4):
        source_matrix[i, i+4] = 1/math.sqrt(2)
    source, _ = second_quantized(source_matrix, 6)
    for momentum in (np.zeros(3), np.array([.001, 0., 0.]), np.array([math.pi, 0., 0.])):
        b = 1+critical+float(np.sum(1-np.cos(momentum)))
        f = sum((1j*math.sin(momentum[j])*(plus.conj().T@gamma[j+1]@minus)
                 for j in range(3)), np.zeros((2, 2), complex))
        l = np.block([[np.eye(2), -f], [np.zeros((2, 2)), np.eye(2)]])
        transfer = l.conj().T@scipy.linalg.block_diag(np.eye(2)/b, b*np.eye(2))@l
        values, vectors = np.linalg.eigh(transfer)
        log_transfer = (vectors*np.log(values))@vectors.conj().T
        generator, _ = second_quantized(scipy.linalg.block_diag(log_transfer, np.zeros((4, 4))), 6)
        full_transfer = b*b*scipy.linalg.expm(generator)
        orbitals = np.zeros((8, 6), complex)
        orbitals[:4, :2] = vectors[:, 2:]
        orbitals[4:, 2:] = np.eye(4)
        vacuum = np.array([np.linalg.det(orbitals[[i for i in range(8) if mask & (1 << i)], :])
                           for mask in masks])
        acted = source@vacuum
        obsolete_matrix = source_matrix.copy()
        obsolete_matrix[2:, :] = 0.
        obsolete_source, _ = second_quantized(obsolete_matrix, 6)
        obsolete_acted = obsolete_source@vacuum
        vacuum_step = float(np.vdot(vacuum, full_transfer@vacuum).real)
        charged_step = float(np.vdot(acted, full_transfer@acted).real)
        records.append(dict(momentum=momentum.tolist(), Wilson_B=b,
            actual_T1_eigenvalues=values.tolist(), actual_detBplus=b*b,
            actual_Fock_transfer_minimum=float(np.linalg.eigvalsh(full_transfer)[0]),
            filled_active_sea_number=2, Ref_initial_number=4,
            actual_primitive_CAR_source_norm_squared=float(np.vdot(acted, acted).real),
            obsolete_beta_plus_projected_source_norm_squared=float(np.vdot(obsolete_acted, obsolete_acted).real),
            actual_charged_halfT_Gram_relative_to_vacuum=charged_step/vacuum_step,
            actual_free_bare_pole_energy=math.log(values[-1]),
            beta_plus_bare_positive_pole_overlap=float(np.trace(vectors[:2, :2]@vectors[:2, :2].conj().T).real/2)))
    return dict(renormalization_condition='m0critical=expm1(a Mtarget)-Sigma_plus(Mtarget); Sigma is the executed on-shell Wilson tadpole+rainbow, not a fitted native gap.',
        target_reference_mass_input=p.mass, computed_Sigma_plus_lattice=sigma,
        declared_critical_bare_m0=critical,
        declared_critical_bare_mass_parameter=math.log1p(critical)/p.spacing,
        primitive_kappa=kappa, actual_all_unitary_fields_B_lower=1-6*kappa,
        active_8k_hopping_series_ratio=8*kappa,
        actual_positive_Wilson_RefQ_source_receipts=records,
        unchanged_SU4_species_count=contract['SU4_species_count'],
        limits=['One-loop renormalization CONDITION, not an exact mass/gap certificate.',
            'Negative bare m0 flips the free positive-band beta orientation; actual source Gram uses actual bands, not an invalid beta-plus frame.',
            '8k>1 invalidates inherited active winding/thermal hopping ledger; m0<0 invalidates positive-accretive D/global chart determinant argument.',
            'B>0 and detB Gamma(T1)>0 DO survive for all unitary links. Full Gauss prepared state and interacting pole/current must use this actual new bare law, not its free pole as targetM.'])


def signed_skeleton_recovery(p: Parameters, contract: dict[str, Any],
                             critical: dict[str, Any]) -> dict[str, Any]:
    """Track the actual signed reference bands when the bare sea switches."""
    bare = critical['declared_critical_bare_m0']
    signed_mass = critical['declared_critical_bare_mass_parameter']
    records = []
    for iteration in range(12):
        reference = Parameters(**dict(asdict(p), mass=signed_mass))
        result = macro_body_selfenergy(reference, contract)
        pole = result['numerical_sequence'][-1]['poles'][0]
        sigma_projected = pole['on_shell_energy_shift_lattice']*pole['pole_residue_E']
        next_mass = math.log1p(bare+sigma_projected)/p.spacing
        b = math.exp(p.spacing*signed_mass)
        t = np.diag([1/b, 1/b, b, b])
        values, vectors = np.linalg.eigh(t)
        filled = vectors[:, 2:]@vectors[:, 2:].conj().T
        positive = np.eye(4)-filled
        records.append(dict(iteration=iteration, signed_reference_mass=signed_mass,
            actual_T1_eigenvalues=values.tolist(), actual_detBplus=b*b,
            beta_plus_filled_sea_overlap=float(np.trace(filled[:2, :2]).real/2),
            beta_plus_positive_pole_overlap=float(np.trace(positive[:2, :2]).real/2),
            actual_RefQ_CAR_source_Gram_before_boundary=float(np.trace(positive).real/2),
            actual_harmonic_charged_halfT_Gram=float(np.trace(positive@t).real/2),
            actual_signed_pole_residue=pole['pole_residue_E'],
            actual_signed_projected_Sigma=sigma_projected,
            next_signed_mass=next_mass, fixed_point_defect=abs(next_mass-signed_mass)))
        if abs(next_mass-signed_mass) < 1e-8:
            signed_mass = next_mass
            break
        signed_mass = next_mass
    return dict(bare_m0_fixed=bare, actual_reference_branch_iterations=records,
        recovered_signed_reference_mass=signed_mass,
        converged=records[-1]['fixed_point_defect']<1e-8,
        occupied_number_per_Bloch_cell=2, Ref_initial_number=4,
        actual_active_filled_sea_count=1, total_species_count=contract['SU4_species_count'],
        exact_onsite_Gauss='Q(nactive-2)+Q(nRef-4)=Q(Ntotal-6); CAR source a†r preserves Ntotal6',
        mass_fit_or_abs_mass_substitution=False,
        construction='Use actual signed b=m0+sum(1-cos p) and actual left/right pole projector/residue inside BOTH tadpole and rainbow. Iterate from negative bare branch, rebuilding filled/empty T1 bands and RefQ scalar source each time. No old positive-beta projector is promoted to the bare vacuum.',
        limit='Executed signed ONE-LOOP skeleton and actual harmonic CAR minors, not the microscopic all-coexact interacting vacuum/source or an all-orders gap error.')


def macro_soft_vertex_control(p: Parameters, contract: dict[str, Any]) -> dict[str, Any]:
    """Actual finite-frequency one-soft vertex and all-momentum norm control."""
    path = ROOT/'covariant_induced_photon.py'
    captured = path.read_bytes()
    supplier = types.ModuleType('_physical_alpha_soft_vertex_epoch')
    supplier.__file__ = str(path)
    sys.modules[supplier.__name__] = supplier
    exec(compile(captured, str(path), 'exec'), supplier.__dict__)
    m0 = math.expm1(p.spacing*p.mass)
    b_min, b_max = 1+m0, 7+m0
    b_floor = min(1., b_min)
    inverse_eigenvalue_bound = max(b_min+4/b_min, b_max+4/b_max)
    static_bound = math.sqrt(3)/b_min
    finite_q_Lipschitz = 3*(64*inverse_eigenvalue_bound/b_floor**4
                             +.5*inverse_eigenvalue_bound**2*(15/b_floor**3)**2)
    L, volume, Nt = 2**24, 2**72, 2**20
    step = 2*math.pi/L
    probes, budgets = [], []
    polarizations = (np.array([0., 1., 0.]), np.array([0., 0., 1.]))
    momenta = (np.zeros(3), np.array([.002, .001, .003]),
               np.array([.1, .2, .3]), np.array([math.pi, 0., 0.]),
               np.full(3, math.pi))
    for radius in (1, 2, 4, 8, 26):
        q = np.array([step*radius, 0., 0.])
        omega = 2*math.asinh(step*radius/2)
        bound = (1+2*math.expm1(omega))*(static_bound+finite_q_Lipschitz*step*radius)
        for pin, polarization in itertools.product(momenta, polarizations):
            emission = supplier.soft_photon_transfer_vertex(gamma_frame(), p.spacing,
                         p.mass, pin, q, polarization, emission=True)
            absorption = supplier.soft_photon_transfer_vertex(gamma_frame(), p.spacing,
                         p.mass, pin-q, q, polarization, emission=False)
            vertex = emission['vertex_without_oscillator']
            physical = emission['final_canonical_frame']@vertex@emission['initial_canonical_frame'].conj().T
            reverse = (absorption['final_canonical_frame']@absorption['vertex_without_oscillator']
                       @absorption['initial_canonical_frame'].conj().T)
            norm = float(np.linalg.norm(vertex, ord=2))
            if not math.isfinite(norm) or norm > bound*(1+1e-12):
                raise ArithmeticError('actual finite-frequency soft vertex violates its norm control')
            probes.append(dict(integer_soft_radius=radius, momentum=pin.tolist(),
                polarization=polarization.tolist(), actual_photon_energy=emission['photon_energy'],
                full_four_band_vertex_norm_without_oscillator=norm,
                proved_uniform_bound_without_oscillator=bound,
                exact_oscillator_normalization=emission['oscillator_normalization'],
                emission_vs_reversed_absorption_physical_frame_defect=float(np.linalg.norm(physical-reverse.conj().T)),
                finite_frequency_vs_static_log_vertex_defect=float(np.linalg.norm(vertex-emission['static_log_vertex_without_oscillator'])),
                actual_band_frame_convention=emission['frame_convention']))
        labels = np.array(list(itertools.product(range(-radius, radius+1), repeat=3)))
        squared = np.sum(labels*labels, axis=1)
        labels = labels[(squared > 0) & (squared <= radius*radius)]
        wavevectors = step*labels
        frequencies = 2*np.arcsinh(np.sqrt(np.sum((2*np.sin(wavevectors/2))**2, axis=1))/2)
        variance = float(np.sum(1/(np.sinh(frequencies)*np.tanh(Nt*frequencies/2)))/volume)
        for duration in (32, 128, 1024):
            x = contract['bare_e']*duration*bound*math.sqrt(variance)
            budgets.append(dict(integer_soft_radius=radius, duration_lattice=duration,
                actual_scaled_thermal_coexact_variance=variance, connected_open_line_word_parameter=x,
                Gaussian_soft_linear_word_norm_upper=math.sqrt(math.expm1(4*x*x)*(math.sqrt(2)-1))))
    return dict(captured_supplier_source_sha256=hashlib.sha256(captured).hexdigest(),
        captured_supplier_source_snapshot=captured.decode(),
        captured_supplier_startup_sources=supplier.SOURCE_AT_IMPORT,
        captured_loaded_own_bytes_match=supplier.SOURCE_AT_IMPORT['covariant_induced_photon.py']==hashlib.sha256(captured).hexdigest(),
        captured_source_unchanged_at_exit=path.read_bytes()==captured,
        actual_signed_mass_parameter=p.mass, actual_B_min=b_min,
        exact_all_momentum_static_one_body_vertex_norm_upper=static_bound,
        finite_q_cross_log_Lipschitz_upper=finite_q_Lipschitz,
        actual_nonzero_frequency_current_probes=probes, renewed_soft_linear_word_budgets=budgets,
        static_proof='T1=tI+v.Gamma, t²-|v|²=1. h=-E n.Gamma; derivative norm²=Eprime²+E²|nprime|²<=Eprime²+sinh²E|nprime|²=|vprime|²-tprime²=(Bprime²+|sprime|²)/B². For unit spatial polarization this is<=3/B². SINGLE connected open fermion line has this one-body norm, not8 times a full-Fock majorant.',
        finite_q_proof='Actual B/F jet gives Tprime<=15/bfloor³ and mixedTsecond<=64/bfloor⁴. Cross-log resolvent integral changes by <=lambda_min^-1||deltaTprime||+||deltaT||||Tprime||/(2lambda_min²). lambda_min^-1<=max(Bmin+4/Bmin,Bmax+4/Bmax); pay3 for the two l1-to-l2 direction sums.',
        frequency_proof='For sqrt(T_F)Kphsqrt(T_F), K1_fi=deltaT_fi(sqrteta_i+exp(-omega)sqrteta_f)/(sqrteta_i+sqrteta_f). Actual -log divided difference at(eta_f exp(-omega),eta_i) gives each-entry multiplier relative to static in[exp(-omega),exp(omega)]. Difference Frobenius<=2expm1(omega)||Jstatic||, so operator bound factor is1+2expm1(omega). Oscillator1/sqrt(2sinhomega) is paid by the measured Gaussian variance, never discarded.',
        limit='Actual one-soft canonical positive-transfer/current control and Gaussian open-line linear-word budgets ONLY. A probability bound additionally needs rotation-safe hard operator envelopes and shifted-momentum controls; matching-index hard sums alone do not provide that. Source Gram, closed active loops, nonlinear contacts and remaining UV are separate.')


def soft_open_line_discriminator(contract: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    """Finite-volume Gaussian soft words, priced against unsigned hard words."""
    L, volume, Nt = 2**24, 2**72, 2**20
    step = 2*math.pi/L
    cuts = []
    for radius in (1, 2, 4, 8, 26):
        labels = np.array(list(itertools.product(range(-radius, radius+1), repeat=3)))
        squared = np.sum(labels*labels, axis=1)
        labels = labels[(squared > 0) & (squared <= radius*radius)]
        momenta = step*labels
        omega = 2*np.arcsinh(np.sqrt(np.sum((2*np.sin(momenta/2))**2, axis=1))/2)
        variance = float(np.sum(1/(np.sinh(omega)*np.tanh(Nt*omega/2)))/volume)
        for hard in packet['probability_sequence']:
            time = hard['duration_lattice']
            x = contract['bare_e']*time*(8*30720)*math.sqrt(variance)
            log_expression = 4*x*x
            word_error = (math.sqrt(math.expm1(log_expression)*(math.sqrt(2)-1))
                          if log_expression < 4 else 6.)
            relative = min(6., word_error)*hard['unsigned_word_to_observed_response_ratio']
            cuts.append(dict(integer_soft_radius=radius, soft_modes=len(labels),
                duration_lattice=time, electron_Hermite_order=hard['electron_Hermite_order'],
                actual_scaled_thermal_coexact_variance=variance,
                connected_open_line_gaussian_word_parameter=x,
                soft_linear_words_amplitude_norm_upper=min(6., word_error),
                coarse_matching_hard_word_ratio=relative,
                matching_word_discriminator_below_one=bool(relative < 1)))
    return dict(executed_exact_finite_soft_mode_sums=cuts,
        proof='Coarse candidate uses Wick ordered-word norm and8*30720 times the actual MATCHING-index hard-word envelope. It correctly refuses signed cancellation as an error majorant; it does NOT control arbitrary soft rotations between unequal virtual indices.',
        limit='Adverse matching-word discriminator, NOT a probability lower-bound certificate. Rotation-safe hard operator envelopes, shifted momenta, actual source Gram, closed active loops/nonlinear contacts and remaining UV are separate.')


def macro_connected_reference(p: Parameters, contract: dict[str, Any],
                              loop: dict[str, Any]) -> dict[str, Any]:
    """Full-box connected occupied-minor probe, with an explicit RPA fork."""
    photon_path = ROOT/'covariant_induced_photon.py'
    photon_bytes = photon_path.read_bytes()
    supplier = types.ModuleType('_physical_alpha_photon_epoch')
    supplier.__file__ = str(photon_path)
    sys.modules[supplier.__name__] = supplier
    exec(compile(photon_bytes, str(photon_path), 'exec'), supplier.__dict__)
    native_consumer = supplier.WilsonConsumer
    def reference_consumer(gamma, spacing, mass):
        return native_consumer(gamma, p.spacing, p.mass)
    setattr(supplier, 'WilsonConsumer', reference_consumer)
    consumer = supplier.WilsonConsumer(gamma_frame(), p.spacing, p.mass)
    L, volume = 2**24, 2**72
    step = 2*math.pi/L
    incoming = step*np.rint(np.array([0., 0., 2*math.asin(math.sinh(.005/2))])/step)
    direction = np.array([math.sin(1.2), 0., math.cos(1.2)])
    from scipy.optimize import brentq
    radius = cast(float, brentq(lambda r: consumer.energy(incoming-r*direction)+consumer.photon_energy(r*direction)
                                -consumer.energy(np.zeros(3))-consumer.photon_energy(incoming),
                                0., .01, xtol=1e-15))
    outgoing = step*np.rint(radius*direction/step)
    packet = np.exp(-np.arange(-1, 2)**2/4)
    packet /= np.linalg.norm(packet)
    sigma, records = .001, []
    frame_conventions: set[str] = set()
    for order in (2, 3):
        node, weight = hermgauss(order)
        for duration in (32, 128, 1024):
            parameters = supplier.Parameters(time_slices=duration)
            response = np.zeros((2, 4), dtype=np.complex128)
            contact = np.zeros_like(response)
            source_gram = 0.
            unsigned_envelope = 0.
            rotation_safe_envelope = 0.
            for indices in itertools.product(range(order), repeat=3):
                electron = math.sqrt(2)*sigma*np.array([node[x] for x in indices])
                electron = step*np.rint(electron/step)
                electron_weight = math.prod(weight[x]/math.sqrt(math.pi) for x in indices)
                source_gram += electron_weight*math.exp(-consumer.energy(electron))
                for i, j in itertools.product(range(3), repeat=2):
                    qin = incoming+(i-1)*np.array([step, 0., 0.])
                    qout = outgoing+(j-1)*np.array([step, 0., 0.])
                    # Centre the product f(p)f(p+delta) before Hermite integration.
                    mismatch = (i-j)*step
                    kinetic = electron-np.array([mismatch/2, 0., 0.])
                    kernel = supplier.connected_transfer_kernel(parameters, gamma_frame(), kinetic, qin, qout)
                    frame_conventions.add(json.dumps(kernel['canonical_spin_frame_convention'], sort_keys=True))
                    amplitude_weight = electron_weight*packet[i]*packet[j]*math.exp(-mismatch*mismatch/(8*sigma*sigma))
                    ref = kernel['initial_Ref_spin_frame'].conj().T/math.sqrt(2)
                    response += amplitude_weight*kernel['response']@ref
                    contact += amplitude_weight*kernel['contact']@ref
                    unsigned_envelope += amplitude_weight*kernel['absolute_time_word_envelope_over_g2_times_V']
                    rotation_safe_envelope += amplitude_weight*kernel['rotation_safe_hard_Cauchy_envelope_over_g2_times_V']
            scale = contract['bare_e']**2/volume
            probability = scale**2*float(np.linalg.norm(response)**2)
            inverse = loop['stable_one_loop_photon_inverse_kernel']
            records.append(dict(electron_Hermite_order=order, duration_lattice=duration,
                connected_grade2_response_over_e2_times_V_real=response.real.tolist(),
                connected_grade2_response_over_e2_times_V_imag=response.imag.tolist(),
                physical_Q_connected_leading_probability=probability,
                RPA_reference_probability=probability/inverse**2,
                harmonic_source_boundary_Gram=source_gram,
                contact_operator_norm_over_e2_times_V=float(np.linalg.norm(contact))))
            records[-1]['absolute_hard_time_word_envelope_over_e2_times_V'] = float(unsigned_envelope)
            records[-1]['unsigned_word_to_observed_response_ratio'] = float(unsigned_envelope/np.linalg.norm(response))
            records[-1]['rotation_safe_hard_Cauchy_envelope_over_e2_times_V'] = float(rotation_safe_envelope)
            records[-1]['rotation_safe_hard_word_to_observed_response_ratio'] = float(rotation_safe_envelope/np.linalg.norm(response))
    return dict(captured_supplier_source_sha256=hashlib.sha256(photon_bytes).hexdigest(),
        captured_supplier_source_snapshot=photon_bytes.decode(),
        captured_supplier_startup_sources=supplier.SOURCE_AT_IMPORT,
        captured_supplier_loaded_own_bytes_match=supplier.SOURCE_AT_IMPORT['covariant_induced_photon.py']==hashlib.sha256(photon_bytes).hexdigest(),
        captured_supplier_unchanged_at_exit=photon_path.read_bytes()==photon_bytes,
        declared_reference_consumer_mass=p.mass,
        declared_reference_consumer_spacing=p.spacing,
        reference_consumer_override='Captured kernel uses its exact WilsonConsumer implementation with explicit (a,Mreference) above. No peer source file is edited or baseline input retagged.',
        input_electron_gaussian_momentum_width_lattice=sigma,
        incoming_photon_centre=incoming.tolist(), outgoing_photon_centre=outgoing.tolist(),
        incoming_and_outgoing_photon_modes=3, packet_norm=float(packet@packet),
        actual_positive_Wilson_connected_CAR_and_log_transfer=True,
        temporal_contact_and_negative_band_paths_retained=True,
        actual_canonical_spin_frame_conventions=sorted(frame_conventions),
        receiver_spin_transport='Outgoing two-spin receiver uses the supplier COMMON signed rest/chirality seed frame. Initial Ref4 hole uses the ACTUAL returned four-component initial_Ref_spin_frame. No fixed beta-plus projector is imposed on negative bare bands.',
        canonical_boundary_precompensation='Within this harmonic reference, multiply bare source amplitude by exp(E(p)/2) and incoming photon creator by exp(omega(q)/2), then normalize. Positive half transfer produces the stated normalized canonical packets. This is bounded on the lattice BZ; it is not an interacting-source admission oracle.',
        probability_sequence=records,
        source='NEW RefQ Coulomb/coexact-factorized reference; all integer-Gauss cycles. No external photon attachment to a longitudinal Coulomb source. Toron source is covariant kinetic-momentum, not fixed canonical labels.',
        linked_repair='Evaluate actual physical-Q determinant Hessian; resum its positive quadratic photon kernel (RPA) rather than use the divergent full-Fock volume majorant. Output RPA and unresummed probabilities separately.',
        limit='Finite Hermite approximation to the full-box leading connected occupied-minor coefficient and a declared one-loop Gaussian RPA reference, NOT the full all-coexact interacting finitepacket probability. RPA stability is not a higher-grade/source-admission proof.')


def rotation_safe_soft_reference(soft: dict[str, Any], hard: dict[str, Any]) -> dict[str, Any]:
    """Compare actual soft-current words to rotation-safe hard reference channels."""
    records = []
    for budget in soft['renewed_soft_linear_word_budgets']:
        for response in hard['probability_sequence']:
            if response['duration_lattice'] != budget['duration_lattice']:
                continue
            ratio = budget['Gaussian_soft_linear_word_norm_upper']*response['rotation_safe_hard_word_to_observed_response_ratio']
            records.append(dict(integer_soft_radius=budget['integer_soft_radius'],
                duration_lattice=budget['duration_lattice'],
                electron_Hermite_order=response['electron_Hermite_order'],
                soft_linear_word_norm_upper=budget['Gaussian_soft_linear_word_norm_upper'],
                actual_rotation_safe_hard_to_response_ratio=response['rotation_safe_hard_word_to_observed_response_ratio'],
                unshifted_rotation_safe_linear_reference_comparison=ratio,
                comparison_below_one=bool(ratio < 1)))
    return dict(records=records,
        hard_supplier_epoch=hard['captured_supplier_source_sha256'],
        soft_supplier_epoch=soft['captured_supplier_source_sha256'],
        method='Cauchy complete virtual2spin row and column norms per actual particle/hole s/u channel, not matched-M products. Triangle over actual packet modes and Hermite electron weights; primitive Ref4 factor is conservatively overpaid.',
        scope='NEW unshifted Gaussian soft-LINEAR reference discriminator, NOT a probability theorem. It permits arbitrary rotations between virtual indices, but shifted hard vertices/momenta, nonlinear soft contacts, actual source/receiver/current normalization and non-Gaussian/full-UV grades are still unpaid.')


def renewed_root_charge_budget(p: Parameters, contract: dict[str, Any],
                               critical: dict[str, Any]) -> dict[str, Any]:
    """Execute the supplier's Q moments; renew the negative-bare margin."""
    path = ROOT/'covariant_root_transfer.py'
    captured = path.read_bytes()
    supplier = types.ModuleType('_physical_alpha_root_epoch')
    supplier.__file__ = str(path)
    sys.modules[supplier.__name__] = supplier
    exec(compile(captured, str(path), 'exec'), supplier.__dict__)
    root_parameters = supplier.RootParameters(normal_width_power=1500, tube_radius_power=1600)
    photon_parameters = supplier.Parameters(beta_power=p.beta_power, kappa_power=p.kappa_power,
                                            masses=(p.mass,), spacings=(p.spacing,))
    charge = int(contract['Q'])
    records = [supplier.charge_weighted_phi_budget(root_parameters, photon_parameters,
                charge=charge, electromagnetic_order=n) for n in range(3)]
    kappa = critical['primitive_kappa']
    margin = critical['actual_all_unitary_fields_B_lower']
    critical_records = [supplier.charge_weighted_phi_budget(root_parameters, photon_parameters,
                charge=charge, electromagnetic_order=n, active_kappa=kappa) for n in range(3)]
    return dict(captured_source_sha256=hashlib.sha256(captured).hexdigest(),
        captured_source_snapshot=captured.decode(),
        supplier_startup_dependency_hashes=supplier.SOURCE_HASHES_AT_IMPORT,
        captured_own_source_matches_supplier_receipt=supplier.SOURCE_HASHES_AT_IMPORT['covariant_root_transfer.py']==hashlib.sha256(captured).hexdigest(),
        captured_source_unchanged_at_exit=path.read_bytes()==captured,
        actual_reference_inputs=dict(Root=asdict(root_parameters), photon=asdict(photon_parameters)),
        captured_physical_component_connection_law=records[0]['physical_component_connection_law'],
        executed_positive_bare_Q_derivative_budgets=records,
        executed_critical_bare_Q_derivative_budgets=critical_records,
        actual_critical_B_scaled_lower=margin/(2*kappa),
        proof='Execute the captured API at actual critical kappa, with explicit beta/kaux matching THIS fork. Auxiliaries retain their tiny-kappa constant and charge1; never inflate Naux byQ^n. Returned projective coefficient/Schatten arithmetic is CONDITIONAL on a compatible physical Phi/transfer/connection and the specified prepared measure.',
        limit='This budget does NOT certify the NEW C-completed kernel or source. Historical bd4d/931 PLUS-connection joined geometry/density claims stay WITHDRAWN; the captured successor physical Dminus law is identified separately, never retagged backward. Its coefficient ledger does not automatically price changed hC/Aeven, outside/cross, source, filters or currents. No active8k winding/accretiveD or actual physical-Q source/probability admission.')


def all_momentum_signed_source(p: Parameters, critical: dict[str, Any]) -> dict[str, Any]:
    """Actual bare-band source minors across the whole Gaussian momentum packet."""
    supplier = sys.modules['_physical_alpha_root_epoch']
    gamma = gamma_frame()
    bare_mass = critical['declared_critical_bare_mass_parameter']
    sigma = .001
    records = []
    for order in (8, 12, 16):
        node, weight = hermgauss(order)
        actual_gram = 0.
        stale_gram = 0.
        actual_boundary = 0.
        stale_boundary = 0.
        for indices in itertools.product(range(order), repeat=3):
            momentum = math.sqrt(2)*sigma*np.array([node[i] for i in indices])
            w = math.prod(weight[i]/math.sqrt(math.pi) for i in indices)
            t_bare, _, _ = supplier.wilson_step(gamma, p.spacing, bare_mass, momentum)
            t_old, _, _ = supplier.wilson_step(gamma, p.spacing, p.mass, momentum)
            eig, basis = np.linalg.eigh(t_bare)
            _, old_basis = np.linalg.eigh(t_old)
            empty = basis[:, :2]@basis[:, :2].conj().T
            stale = old_basis[:, :2]@old_basis[:, :2].conj().T
            actual_gram += w*float(np.trace(empty).real/2)
            stale_gram += w*float(np.trace(stale@empty).real/2)
            actual_boundary += w*float(np.trace(empty@t_bare).real/2)
            stale_boundary += w*float(np.trace(stale@empty@t_bare@empty@stale).real/2)
        records.append(dict(order=order,
            actual_refilled_bare_sea_source_Gram=actual_gram,
            stale_positive_mass_source_Gram_in_ACTUAL_bare_sea=stale_gram,
            actual_charged_halfT_Gram=actual_boundary,
            stale_charged_halfT_Gram=stale_boundary))
    return dict(momentum_width_lattice=sigma, actual_bare_mass_parameter=bare_mass,
        sigma_reference_mass=p.mass, full_packet_Gaussian_source_sequence=records,
        physical_branch='ACTUAL low/high eigenvectors of signed-bare T1, each p. Ref4 scalar source into unoccupied actual band. Occupied number remains2 at every momentum; no |mass| replacement or old beta-plus vacuum.',
        zero_mode='For statistical history0, p0 bare and positive-reference filled bands are orthogonal: the obsolete positive-reference source is Pauli-blocked exactly there. Other7 histories use actual shifts pi*h/L; no Q multiplication.',
        limit='Executed all-momentum Gaussian harmonic source minors, not the full interacting signed-bare thermal vacuum or all-coexact postchargeT Gram.')


def macro_run(p: Parameters) -> dict[str, Any]:
    """NEW macro RefQ fork; archived native Ref0 identities stay unchanged."""
    contract = charge_contract(p)
    q = int(contract['Q'])
    contract = dict(contract, reference_primitive_charge=str(q),
        charges='rho=Q(nactive-2)+sum_aux(naux-2)+Q(nRef-4) at Ref0; source body+Q and Refhole-Q',
        neutralizing_background='Ref operator has chargeQ, neutral filledRef4 background4Q. NO external compensator.',
        branch='NEW macro RefQ, not a retagged native Ref0 control')
    toron = macro_toron_source(p, contract)
    scalar_C_contract = macro_full_label_scalar_C(p, contract)
    coulomb = macro_integer_coulomb(p, contract)
    loop = macro_active_loop(p, contract)
    selfenergy = macro_body_selfenergy(p, contract)
    recoil_condition = mass_conditioned_recoil_reference(p, selfenergy)
    Ward_current = macro_one_loop_Ward_current(p, contract)
    charge_condition = macro_charge_renormalization(p, contract, loop, selfenergy)
    resummed_mass = macro_selfconsistent_mass(p, contract)
    critical_mass = critical_bare_mass_receipt(p, contract, selfenergy)
    critical_Ward_current = macro_one_loop_Ward_current(
        Parameters(**dict(asdict(p), mass=critical_mass['declared_critical_bare_mass_parameter'])), contract)
    scalar_source_control = locked_native_scalar_source(p, contract, critical_mass)
    packet_reference_energy = macro_packet_reference_energy(p, contract, coulomb, critical_mass, resummed_mass)
    local_linked = macro_local_linked_measure(p, contract, critical_mass)
    signed_recovery = signed_skeleton_recovery(p, contract, critical_mass)
    root_budget = renewed_root_charge_budget(p, contract, critical_mass)
    signed_packet_source = all_momentum_signed_source(p, critical_mass)
    connected = macro_connected_reference(p, contract, loop)
    mass_parameters = Parameters(**dict(asdict(p), mass=resummed_mass['resummed_rest_mass_physical']))
    resummed_connected = macro_connected_reference(mass_parameters, contract, loop)
    soft_vertices = macro_soft_vertex_control(mass_parameters, contract)
    soft_rotation_reference = rotation_safe_soft_reference(soft_vertices, resummed_connected)
    critical_parameters = Parameters(**dict(asdict(p), mass=critical_mass['declared_critical_bare_mass_parameter']))
    critical_loop = macro_active_loop(critical_parameters, contract)
    critical_connected = macro_connected_reference(critical_parameters, contract, critical_loop)
    integer_source = literal_integer_flow(p)
    soft = soft_open_line_discriminator(contract, resummed_connected)
    # Price the actually inherited majorant before attempting a linked repair.
    L, Nt, volume = 2**24, 2**20, 2**72
    field, body = 2e6, 8*30720.
    inherited_ratio = contract['bare_e']*Nt*body*volume*field*64*math.exp(2*math.log(8)+2*math.asinh(math.sqrt(3)))
    return dict(parameters=asdict(p), charge_contract=contract,
        macro_harmonic_prepared_source=toron, macro_integer_Coulomb_dressing=coulomb,
        actual_band_packet_Ref_and_Coulomb_reference=packet_reference_energy,
        actual_locked_native_Ref4_scalar_source_control=scalar_source_control,
        full_label_coherent_Ref4_source_C_matrix_contract=scalar_C_contract,
        physical_Q_unquenched_current_contact=loop,
        physical_Q_all_grade_local_linked_measure=local_linked,
        one_loop_joint_charge_mass_condition=charge_condition,
        one_loop_same_mass_condition_recoil_reference=recoil_condition,
        one_loop_independent_Ward_current=Ward_current,
        actual_signed_bare_independent_Ward_current=critical_Ward_current,
        physical_Q_Wilson_body_selfenergy=selfenergy,
        physical_Q_selfconsistent_mass_repair=resummed_mass,
        physical_Q_connected_reference=connected,
        critical_bare_mass_declared_fork=critical_mass,
        critical_bare_signed_source_recovery=signed_recovery,
        actual_signed_bare_Gaussian_packet_source=signed_packet_source,
        actual_critical_bare_photon_determinant_jet=critical_loop,
        actual_critical_bare_connected_reference=critical_connected,
        physical_Q_soft_connected_discriminator=soft,
        actual_soft_vertex_and_renewed_word_control=soft_vertices,
        actual_rotation_safe_soft_linear_reference=soft_rotation_reference,
        physical_Q_mass_resummed_connected_reference=resummed_connected,
        literal_integer_full_rho_source=integer_source,
        actual_Q_Root_charge_and_margin_budget=root_budget,
        inherited_full_Fock_higher_grade_ratio=inherited_ratio,
        inherited_physical_Q_remainder_rejected=inherited_ratio>=1,
        material_repair='All-label covariant kinetic momentum and unpinned Haar torons repair the native L2 holonomy artifact; all integer cycle/harmonic Gaussian sums repair literal-string UV energy. Full physical-Q active determinant current/contact was renewed, not borrowed from Q1.',
        exact_full_density_contract='rho=PiNs P_G T^Nt P_G PiNs/Z on actual nonzero positive-transfer range; T contains detB Gamma(T1), full compact photon/Coulomb, Root field and centre completion. S uses RefQ and the above exact integer-Gauss wavefunction. Charged preparation is Tchhalf S rho_ph Sdag Tchhalf /Tr(...).',
        remaining_debt=['Full interacting charged T source Gram, not the evaluated harmonic-background minor.',
            'Physical-Q connected linked/resummed higher grades and actual hard-photon probability/error.',
            'Higher-loop microscopic body/Ref self-energies; one-loop skeleton is executed but not certified.',
            'Root moments/margins are renewed; actual normal/colour/full-source state admission remains separate.'],
        admission='Actual macro harmonic-sector prices and active determinant jet, NOT a completed full macro observable or exact physical-Q scattering certificate.')



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--orders', default='3,3,4')
    parser.add_argument('--toron-cutoffs', default='1,3,5')
    parser.add_argument('--dense-limit', type=int, default=2000)
    parser.add_argument('--macro', action='store_true')
    parser.add_argument('--floquet-only', action='store_true')
    parser.add_argument('--flux-fraction', default='0')
    parser.add_argument('--integer-flow-only', action='store_true')
    parser.add_argument('--cartesian-c', choices=('born', 'native'))
    parser.add_argument('--bare-mass', type=float)
    parser.add_argument('--phi-gram')
    parser.add_argument('--normal-order', type=int, default=2)
    parser.add_argument('--normal-width', type=float, default=5e-6)
    args = parser.parse_args()
    p = Parameters(orders=tuple(int(x) for x in args.orders.split(',')),
                   toron_cutoffs=tuple(int(x) for x in args.toron_cutoffs.split(',')),
                   dense_limit=args.dense_limit, flux_fraction=args.flux_fraction)
    started = time.time()
    new_inputs = None
    if args.cartesian_c:
        if args.bare_mass is None or args.phi_gram is None:
            parser.error('--cartesian-c requires explicit --bare-mass and --phi-gram')
        if args.macro or args.floquet_only or args.integer_flow_only:
            parser.error('--cartesian-c is a separate NEW law branch')
        phi_gram = tuple(float(value) for value in args.phi_gram.split(','))
        if len(phi_gram) != 4 or min(phi_gram) <= 0:
            parser.error('--phi-gram requires four positive entries')
        phi_reference = np.sqrt(phi_gram)[:, None, None]*SELECTED
        critical = dict(declared_critical_bare_mass_parameter=args.bare_mass)
        contract = charge_contract(p)
        new_inputs = dict(bare_mass=args.bare_mass, Phi_reference_gram=phi_gram,
                         normal_order=args.normal_order, native_normal_width=args.normal_width)
        branch = 'NEW_cartesian_C_'+args.cartesian_c
        if args.cartesian_c == 'born':
            result = cartesian_C_source_Born(p, contract, critical, phi_reference,
                args.normal_order, args.normal_width)
        else:
            result = cartesian_C_native_source(p, contract, critical, phi_reference,
                args.normal_order, args.normal_width, p.orders)
    else:
        branch = ('literal_integer_flow_RefQ' if args.integer_flow_only else
                  ('macro_RefQ' if args.macro else ('native_Floquet_Ref0' if args.floquet_only else 'native_Ref0')))
        result = (literal_integer_flow(p) if args.integer_flow_only else
                  (macro_run(p) if args.macro else (run_floquet(p) if args.floquet_only else run(p))))
    inputs = dict(parameters=asdict(p), branch=branch)
    if new_inputs is not None:
        inputs['NEW_law_inputs'] = new_inputs
    input_bytes = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
    result['capsule'] = dict(source_at_import=SOURCE_HASHES,
        source_snapshot_at_import={name: content.decode() for name, content in SOURCE_BYTES.items()},
        input_sha256=hashlib.sha256(input_bytes).hexdigest(), input_json=input_bytes.decode(),
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, mpmath=mp.__version__,
                     executable=sys.executable, platform=platform.platform()),
        runtime_seconds=time.time()-started, startup_source_unchanged={
            name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
            for name, digest in SOURCE_HASHES.items()})
    path = Path(args.output)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(path), source_sha256=SOURCE_HASHES['physical_alpha_compton.py'],
                         input_sha256=result['capsule']['input_sha256'], runtime_seconds=result['capsule']['runtime_seconds'],
                         branch=branch,
                         evidence=((dict(raw_Gram=result['actual_unfiltered_Ref4_scalar_Gram'],
                                         Born=result['probabilities']) if args.cartesian_c == 'born'
                                     else dict(native=result['records'])) if args.cartesian_c else
                                   dict(integer_Gauss=[x['exact_integer_Gauss_defect'] for x in result['exact_native_integer_receipts']])
                                   if args.integer_flow_only else dict(toron_energy_upper=result['macro_harmonic_prepared_source']['toron_kinetic_energy_physical_upper'],
                                        loop_inverse_kernel=result['physical_Q_unquenched_current_contact']['stable_one_loop_photon_inverse_kernel'])
                                   if args.macro else dict(probabilities=[x['charged']['exclusive_probabilities'] for x in result['native_positive_transfer']],
                                                           truncation_differences=result['truncation_sequence_probability_differences'])))))


if __name__ == '__main__':
    main()
