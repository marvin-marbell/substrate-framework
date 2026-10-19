#!/usr/bin/env python3
"""Same primitive Peierls-band Compton in a NEW induced large-N gauge branch.

No frozen source, receiver or probability is reused. Photon input is a mandatory
source-bound induced_photon_response capsule, not the old fixed-speed reference.
External charged states are normalized positive-band states above a filled sea;
this leading 1/N tree consumer is not an infrared-complete interacting S matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from typing import Any, cast
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from scipy.optimize import brentq
from improved_phi_band import ImprovedPhiLaw
from rotation_band import RotationBand
from sea_selection import SELECTED
from compton import amplitude as qed_amplitude, klein_nishina_squared

ROOT = Path(__file__).resolve().parent
I4 = np.eye(4, dtype=complex)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SOURCE_NAMES = ('induced_compton_transfer.py', 'improved_phi_band.py', 'rotation_band.py',
                'local_orbital_stability.py', 'sea_selection.py', 'compton.py',
                'induced_photon_response.py', 'induced_photon_mode_frame.py')
EXECUTED_SOURCE_HASHES = {name: sha(ROOT/name) for name in SOURCE_NAMES}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def species_profile(capsule: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = capsule['config']
    species = cfg['species'] or [[cfg['flavors'], cfg['m']]]
    require(float(species[0][1]) == float(cfg['m']), 'first charged species includes the active filled flavor')
    require(sum(int(count) for count, _ in species) == int(cfg['flavors']),
            'total charged flavors include active flavor exactly once')
    masses: dict[float, int] = {}
    for count, mass in species:
        require(int(count) == count and int(count) > 0 and float(mass) > 0,
                'positive integer actual species count and gap input')
        masses[float(mass)] = masses.get(float(mass), 0)+int(count)
    return [dict(m=mass, count=count) for mass, count in sorted(masses.items())]


class SameCurrent:
    """Fourier transform of actual ImprovedPhiLaw one/two-link derivatives.

    The link field is sampled at each link midpoint. S_l(theta) sums EVERY
    insertion location on a length-l path, including all l*l contact pairs.
    All h/J/K use the very same primitive Phi-weighted path coefficients.
    """
    def __init__(self, capsule: dict[str, Any], dynamic: bool = False):
        self.capsule = capsule
        cfg = capsule['config']
        self.species = species_profile(capsule)
        self.dynamic = dynamic
        self.mode_cache: dict[tuple[float, ...], dict[str, Any]] = {}
        if dynamic:
            from induced_photon_response import ResponseConfig, SpectralResponse
            self.response_config = ResponseConfig(**cfg)
            self.response_model = SpectralResponse(self.response_config, self.response_config.orders[-1])
        band = capsule['band']
        self.a = float(cfg['a'])
        require(float(band['a']) == self.a and float(band['e']) == 1.,
                'same lattice spacing and unit path charge as the real supplier')
        require(all(float(band[key]) == float(cfg[key]) for key in ('m', 'r', 'v')),
                'supplier band/config physical parameter agreement')
        self.band = RotationBand(a=self.a, m=float(band['m']), r=float(band['r']),
                                 v=float(band['v']), improved=bool(band['improved']))
        metric = capsule['metric']
        g0, gs = float(metric['g0']), float(metric['gs'])
        require(g0 == float(cfg['g0']) and gs == float(cfg['gs']),
                'actual supplier mean metric, not a manufactured endpoint frame')
        self.frame = np.sqrt([g0, gs, gs, gs])[:, None, None] * SELECTED
        law = ImprovedPhiLaw(3, self.band)
        self.onsite = law.onsite * self.frame[0]
        self.paths = [(np.asarray(d, dtype=float), np.einsum('m,mij->ij', s, self.frame))
                      for d, s in law.paths]
        self.epsilon = float(capsule['epsilon'])
        self.kappa = float(capsule['kappa'])
        self.c = float(capsule['photon_speed'])
        self.z = float(capsule['transverse_residue'])
        require(self.epsilon > 0 and self.kappa > 0, 'stable induced electric/magnetic response')
        require(abs(self.c**2-self.kappa/self.epsilon) < 1e-10*self.c**2,
                'dispersion and response must remain coupled')
        require(abs(self.z*self.epsilon-1) < 1e-10, 'photon residue is inverse electric coefficient')
        self.mass = self.energy(np.zeros(3))
        self.cm = self.band.v * math.sqrt(gs)
        _, vectors = np.linalg.eigh(self.h(np.zeros(3)))
        # Rest positive doublet, spin diagonalized along the primitive z generator.
        base = vectors[:, 2:]
        spin = -.5j * SELECTED[1] @ SELECTED[2]
        _, rotation = np.linalg.eigh(base.conj().T @ spin @ base)
        self.rest = base @ rotation

    def h(self, p: np.ndarray) -> np.ndarray:
        result = self.onsite.astype(complex).copy()
        for d, t in self.paths:
            phase = np.exp(1j*self.a*np.dot(p, d))
            result += phase*t + phase.conjugate()*t.conj().T
        return result

    def energy(self, p: np.ndarray) -> float:
        h = self.h(p)
        return math.sqrt(float(np.trace(h@h).real/4))

    def velocity(self, p: np.ndarray) -> np.ndarray:
        h = self.h(p)
        result = np.zeros(3)
        for axis in range(3):
            derivative = np.zeros((4, 4), complex)
            for d, t in self.paths:
                phase = np.exp(1j*self.a*np.dot(p, d))
                derivative += 1j*self.a*d[axis]*(phase*t-phase.conjugate()*t.conj().T)
            result[axis] = np.trace(h@derivative).real/(4*self.energy(p))
        return result

    def spinors(self, p: np.ndarray) -> np.ndarray:
        q = (I4+self.h(p)/self.energy(p))/2
        raw = q @ self.rest
        values, vectors = np.linalg.eigh(raw.conj().T @ raw)
        require(min(values) > 1e-12, 'positive-spin chart remains nonsingular')
        return raw @ ((vectors/np.sqrt(values)) @ vectors.conj().T)

    def qhat(self, q: np.ndarray) -> np.ndarray:
        return 2*np.sin(self.a*q/2)/self.a

    @staticmethod
    def path_sum(length: int, theta: float) -> float:
        return float(sum(np.cos((j-(length-1)/2)*theta) for j in range(length)))

    def current(self, pout: np.ndarray, pin: np.ndarray) -> np.ndarray:
        result = np.zeros((3, 4, 4), complex)
        midpoint, q = (pout+pin)/2, pout-pin
        for d, t in self.paths:
            axis = int(np.flatnonzero(d)[0])
            length = int(d[axis])
            phase = np.exp(1j*self.a*np.dot(midpoint, d))
            s = self.path_sum(length, self.a*q[axis])
            result[axis] += 1j*self.a*s*(phase*t-phase.conjugate()*t.conj().T)
        return result

    def vertex(self, pout: np.ndarray, pin: np.ndarray, eps: np.ndarray) -> np.ndarray:
        return eps[0]*I4 - np.einsum('i,ijk->jk', eps[1:], self.current(pout, pin))

    def contact(self, pout: np.ndarray, pin: np.ndarray, ein: np.ndarray,
                eout: np.ndarray, k: np.ndarray, kout: np.ndarray) -> np.ndarray:
        result = np.zeros((4, 4), complex)
        midpoint = (pout+pin)/2
        for d, t in self.paths:
            axis = int(np.flatnonzero(d)[0])
            length = int(d[axis])
            phase = np.exp(1j*self.a*np.dot(midpoint, d))
            s = self.path_sum(length, self.a*k[axis])*self.path_sum(length, self.a*kout[axis])
            result -= self.a**2*ein[axis+1]*eout[axis+1]*s*(phase*t+phase.conjugate()*t.conj().T)
        return result

    def mode(self, k: np.ndarray, velocity: bool = False) -> dict[str, Any]:
        from induced_photon_mode_frame import compute_mode
        key = tuple(float(x) for x in np.round(k, 13))
        if key not in self.mode_cache or (velocity and
                'group_velocity' not in self.mode_cache[key]['two_polarizations'][0]):
            result = compute_mode(self.response_config, np.asarray(k, float), velocity=velocity,
                                  model=self.response_model)
            require(all(0 < float(p['omega']) < float(result['pair_threshold']) and
                        float(p['residue']) > 0 for p in result['two_polarizations']),
                    'two stable subthreshold spectral photon modes')
            require(float(result['canonical_mode_gram_error']) < 1e-7,
                    'paired low-energy photon states orthonormal in the full field-plus-PH norm')
            self.mode_cache[key] = result
        return self.mode_cache[key]

    def omega(self, k: np.ndarray, index: int = 0) -> float:
        if np.linalg.norm(k) == 0:
            return 0.
        return (float(self.mode(k)['two_polarizations'][index]['omega']) if self.dynamic
                else self.c*float(np.linalg.norm(self.qhat(k))))

    def residue(self, k: np.ndarray, index: int = 0) -> float:
        return float(self.mode(k)['two_polarizations'][index]['residue']) if self.dynamic else self.z

    def photon_velocity(self, k: np.ndarray, index: int = 0) -> np.ndarray:
        if self.dynamic:
            return np.asarray(self.mode(k, velocity=True)['two_polarizations'][index]['group_velocity'], float)
        return self.c**2*np.sin(self.a*k)/(self.a*self.omega(k))

    def photon_radial_velocity(self, k: np.ndarray, index: int = 0) -> tuple[float, float]:
        """Ray derivative for the outgoing energy Jacobian, not incoming flux.

        Both finite-difference resolutions evaluate actual spectral poles; there
        is no interpolation or phase-speed substitution. Incoming flux retains
        the full Cartesian group-velocity norm.
        """
        radius = float(np.linalg.norm(k))
        require(radius > 0, 'nonzero outgoing momentum')
        normal = k/radius
        if not self.dynamic:
            return float(np.dot(self.photon_velocity(k, index), normal)), 0.
        step = min(radius/4, max(radius*1e-3, 1e-5))
        coarse = (self.omega(k+step*normal, index)-self.omega(k-step*normal, index))/(2*step)
        fine = (self.omega(k+step*normal/2, index)-self.omega(k-step*normal/2, index))/step
        error = abs(fine-coarse)/max(abs(fine), np.finfo(float).tiny)
        require(error < 1e-5, 'outgoing spectral ray derivative converges')
        return float(fine), float(error)

    def polarizations(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.dynamic:
            poles = self.mode(k)['two_polarizations']
            return (np.r_[poles[0]['scalar_polarization'], poles[0]['polarization']],
                    np.r_[poles[1]['scalar_polarization'], poles[1]['polarization']])
        normal = self.qhat(k)/np.linalg.norm(self.qhat(k))
        seed = np.eye(3)[int(np.argmin(abs(normal)))]
        first = np.cross(normal, seed); first /= np.linalg.norm(first)
        return np.r_[0., first], np.r_[0., np.cross(normal, first)]

    def kinematics(self, x: float, theta: float, direction: np.ndarray,
                   incoming_index: int = 0, outgoing_index: int = 0) -> tuple[np.ndarray, ...]:
        require(x > 0 and 0 < theta < math.pi, 'positive energy and nonendpoint angle')
        nin = np.asarray(direction, float); nin = nin/np.linalg.norm(nin)
        seed = np.eye(3)[int(np.argmin(abs(nin)))]
        tangent = seed-np.dot(seed, nin)*nin; tangent /= np.linalg.norm(tangent)
        nout = math.cos(theta)*nin+math.sin(theta)*tangent
        target = x*self.mass
        high = min(math.pi/(4*self.a), 2*target/self.c)
        require(self.omega(high*nin, incoming_index) > target, 'input exceeds selected low-Brillouin photon branch')
        incoming_norm = cast(float, brentq(lambda t: self.omega(t*nin, incoming_index)-target,
                                          0., high, xtol=1e-13))
        k = incoming_norm*nin
        balance = lambda t: self.energy(k-t*nout)+self.omega(t*nout, outgoing_index)-self.mass-target
        require(balance(0.) < 0 and balance(incoming_norm*1.1) > 0,
                'same-band two-body recoil has a root in selected branch')
        outgoing_norm = cast(float, brentq(balance, 0., incoming_norm*1.1, xtol=1e-14))
        kout = outgoing_norm*nout
        pf = k-kout
        require(abs(self.energy(pf)+self.omega(kout, outgoing_index)-self.mass-target) < 2e-12,
                'exact band energy conservation')
        return np.zeros(3), k, pf, kout

    def propagator(self, energy: float, momentum: np.ndarray, positive_only: bool = False) -> np.ndarray:
        h, e = self.h(momentum), self.energy(momentum)
        q = (I4+h/e)/2
        return q/(energy-e) + (0 if positive_only else (I4-q)/(energy+e))

    def kernel(self, p: np.ndarray, k: np.ndarray, pf: np.ndarray, kout: np.ndarray,
               ein: np.ndarray, eout: np.ndarray, control: str = '',
               incoming_index: int = 0, outgoing_index: int = 0) -> np.ndarray:
        eo = np.conjugate(eout)
        gs = self.propagator(self.energy(p)+self.omega(k, incoming_index), p+k, control == 'positive_poles')
        gu = self.propagator(self.energy(p)-self.omega(kout, outgoing_index), p-kout, control == 'positive_poles')
        result = (self.vertex(pf, p+k, eo) @ gs @ self.vertex(p+k, p, ein)
                  + self.vertex(pf, p-kout, ein) @ gu @ self.vertex(p-kout, p, eo))
        if control != 'no_contact':
            result += self.contact(pf, p, ein, eo, k, kout)
        return result

    @staticmethod
    def unpolarized_surface_flux(direction: np.ndarray,
                                 current_expectations: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Tr[(I/2) n.V]: one physical surface and equal photon-number density.

        Expectations are the two canonical prepared-state current diagonals,
        not eigenvalues of mutually noncommuting velocity operators. Their
        average is linear and basis invariant; averaging their norms is not.
        """
        normal = np.asarray(direction, dtype=float)
        normal = normal/np.linalg.norm(normal)
        mean_current = np.mean(current_expectations, axis=0)
        flux = float(np.dot(normal, mean_current))
        require(flux > 0., 'positive incoming number current through the fixed physical surface')
        return flux, normal, mean_current

    def observe(self, x: float, theta: float, direction: np.ndarray,
                control: str = '') -> dict[str, Any]:
        channels = []
        total, rate, ward = 0., 0., 0.
        qed_recoil = 1/(1+x*(1-math.cos(theta)))
        qed_soft = (1+math.cos(theta)**2)/(32*math.pi**2*self.mass**2)
        incoming_momenta = [self.kinematics(x, theta, direction, i, 0)[1] for i in range(2)]
        incoming_currents = np.asarray([self.photon_velocity(k, i)
                                       for i, k in enumerate(incoming_momenta)])
        flux, incident_normal, mean_incident_current = self.unpolarized_surface_flux(
            direction, incoming_currents)
        for i in range(2):
            for j in range(2):
                p, k, pf, kout = self.kinematics(x, theta, direction, i, j)
                spin_in, spin_out = self.spinors(p), self.spinors(pf)
                norm_error = max(float(np.linalg.norm(u.conj().T@u-np.eye(2)))
                                 for u in (spin_in, spin_out))
                shell_error = max(float(np.linalg.norm((self.h(momentum)-self.energy(momentum)*I4)@u)/
                                        self.energy(momentum))
                                  for momentum, u in ((p, spin_in), (pf, spin_out)))
                require(norm_error < 1e-10 and shell_error < 1e-10,
                        'unit-normalized spin states on the actual positive charged-band shell')
                ei, eo = self.polarizations(k)[i], self.polarizations(kout)[j]
                kernel = self.kernel(p, k, pf, kout, ei, eo, control, i, j)
                amplitudes = spin_out.conj().T @ kernel @ spin_in
                squared = float(np.sum(abs(amplitudes)**2))/4
                total += squared
                wi, wo = self.omega(k, i), self.omega(kout, j)
                normal = kout/np.linalg.norm(kout)
                radial_velocity, radial_error = self.photon_radial_velocity(kout, j)
                jac = abs(radial_velocity-float(np.dot(self.velocity(pf), normal)))
                zin, zout = ((1., 1.) if control == 'unit_residue' else
                             (self.residue(k, i), self.residue(kout, j)))
                channel_rate = float(zin*zout*squared*np.dot(kout, kout)/(16*math.pi**2*wi*wo*flux*jac))
                rate += channel_rate
                # Comparator ONLY: project lattice eigenvectors onto continuum
                # transverse directions. This map is declared, not an endpoint supplier.
                nin = k/np.linalg.norm(k)
                pin_ref = ei[1:]-np.dot(ei[1:], nin)*nin
                pout_ref = eo[1:]-np.dot(eo[1:], normal)*normal
                pin_ref /= np.linalg.norm(pin_ref)
                pout_ref /= np.linalg.norm(pout_ref)
                rp = np.r_[self.mass, np.zeros(3)]
                rk = np.r_[wi, wi*nin]
                rkout = np.r_[wi*qed_recoil, wi*qed_recoil*normal]
                rpf = rp+rk-rkout
                reference_squared = sum(abs(qed_amplitude(
                    rp, rk, rpf, rkout, np.r_[0., pin_ref], np.r_[0., pout_ref],
                    s, sf, m=self.mass))**2 for s in range(2) for sf in range(2))/4
                reference_rate = reference_squared*qed_recoil**2/(64*math.pi**2*self.mass**2)
                for gauge_in, gauge_out in ((np.r_[wi, self.qhat(k)], eo),
                                            (ei, np.r_[wo, self.qhat(kout)])):
                    gauge = self.kernel(p, k, pf, kout, gauge_in, gauge_out, control, i, j)
                    defect = np.max(abs(spin_out.conj().T @ gauge @ spin_in))
                    ward = max(ward, float(defect)/(self.cm**2*wi/self.mass))
                channel = dict(in_polarization=i, out_polarization=j,
                               incoming_polarization=ei[1:].tolist(), outgoing_polarization=eo[1:].tolist(),
                               incoming_scalar_polarization=float(ei[0]), outgoing_scalar_polarization=float(eo[0]),
                               charged_state_norm_error=norm_error, charged_state_on_shell_error=shell_error,
                               spin_amplitudes_real=np.real(amplitudes).tolist(),
                               spin_amplitudes_imag=np.imag(amplitudes).tolist(),
                               k=k.tolist(), kout=kout.tolist(), final_band_energy=self.energy(pf),
                               photon_omega=wi, outgoing_photon_omega=wo, recoil_ratio=wo/wi,
                               recoil_relative_error=abs(wo/wi/qed_recoil-1), flux=flux,
                               incoming_state_current=incoming_currents[i].tolist(),
                               incoming_state_normal_current=float(np.dot(incident_normal, incoming_currents[i])),
                               radial_energy_jacobian=jac, radial_photon_group_velocity=radial_velocity,
                               radial_derivative_resolution_error=radial_error,
                               incoming_photon_residue=zin,
                               outgoing_photon_residue=zout, differential_cross_section=channel_rate)
                channel['QED_polarized_rate_over_Thomson'] = float(reference_rate/qed_soft)
                if self.dynamic:
                    channel['incoming_pair_threshold'] = self.mode(k)['pair_threshold']
                    channel['outgoing_pair_threshold'] = self.mode(kout)['pair_threshold']
                    channel['incoming_polarization_splitting'] = self.mode(k)['polarization_speed_splitting']
                    channel['outgoing_polarization_splitting'] = self.mode(kout)['polarization_speed_splitting']
                    channel['canonical_photon_mode_gram_error'] = max(
                        float(self.mode(k)['canonical_mode_gram_error']),
                        float(self.mode(kout)['canonical_mode_gram_error']))
                    channel['incoming_mode_frame_proof'] = self.mode(k)['frame_proof']
                    channel['outgoing_mode_frame_proof'] = self.mode(kout)['frame_proof']
                channels.append(channel)
        soft = self.z**2*self.cm**4*(1+math.cos(theta)**2)/(32*math.pi**2*self.mass**2*self.c**4)
        qed_shape = klein_nishina_squared(x, theta)*qed_recoil**2/(2*(1+math.cos(theta)**2))
        polarized_error = max(abs(channel['differential_cross_section']/soft -
                                  channel['QED_polarized_rate_over_Thomson']) for channel in channels)
        return dict(x=x, theta=theta, direction=np.asarray(direction).tolist(),
                    QED_recoil_comparator=qed_recoil,
                    angular_measure='outgoing wave-vector solid angle',
                    incident_flux_surface_normal=incident_normal.tolist(),
                    incident_density_matrix=[[.5, 0.], [0., .5]],
                    incident_mean_number_current=mean_incident_current.tolist(),
                    incident_number_flux=flux,
                    incident_flux_semantics='Tr[(I/2) n.V], common fixed incident surface; not a sum of per-state current norms',
                    recoil_relative_error=max(channel['recoil_relative_error'] for channel in channels),
                    reduced_spin_polarization_squared=total, channels=channels,
                    differential_cross_section=rate, Thomson_prediction=soft,
                    rate_over_Thomson=rate/soft, QED_rate_over_Thomson=qed_shape,
                    finite_energy_shape_error=abs(rate/soft/qed_shape-1),
                    polarization_shape_error=polarized_error,
                    polarization_comparator_map='spatial qhat-transverse mode projected onto q-transverse QED state; spin averaged',
                    normalized_Ward_residual=ward)


def load_capsule(path: Path) -> dict[str, Any]:
    capsule = json.loads(path.read_text())
    for key in ('config', 'band', 'epsilon', 'kappa', 'photon_speed',
                'transverse_residue', 'source_hashes', 'validity', 'branch'):
        require(key in capsule, f'missing real supplier field: {key}')
    # These are the response's executed microscopic dependencies. The original
    # bosonic/reference files in old capsules annotate the historical law being
    # replaced; they are not dependencies of this induced response calculation.
    provider_sources = ('induced_photon_response.py', 'improved_phi_band.py',
                        'rotation_band.py', 'local_orbital_stability.py', 'sea_selection.py')
    for name in provider_sources + (('induced_photon_mode_frame.py',)
                                   if 'induced_photon_mode_frame.py' in capsule['source_hashes'] else ()):
        require(name in capsule['source_hashes'], f'missing executed supplier provenance: {name}')
        source = ROOT/name
        require(source.is_file() and sha(source) == capsule['source_hashes'][name],
                f'stale executed supplier source: {name}')
    return capsule


def transfer(capsule: dict[str, Any], dynamic: bool = False) -> dict[str, Any]:
    model = SameCurrent(capsule, dynamic=dynamic)
    directions = (np.array([0., 0., 1.]), np.ones(3)/math.sqrt(3))
    energies = (.1, .4) if dynamic else (.01, .1, .4)
    angles = (.7, 1.3) if dynamic else (.7, 1.3, 2.1)
    rows = [model.observe(x, theta, direction) for x in energies
            for theta in angles for direction in directions]
    require(max(row['normalized_Ward_residual'] for row in rows) < 1e-7,
            'same paths, both signs of poles and contact: physical on-shell Ward cancellation')
    soft = SameCurrent(capsule).observe(1e-4, 1.3, directions[0])
    require(abs(soft['rate_over_Thomson']-1) < .005, 'derived-residue inertial Thomson limit')
    probe = model.observe(.4, 1.3, directions[1])
    controls = {}
    for name in ('no_contact', 'positive_poles', 'unit_residue'):
        bad = model.observe(.4, 1.3, directions[1], name)
        defect = (bad['normalized_Ward_residual'] if name != 'unit_residue' else
                  abs(bad['differential_cross_section']/probe['differential_cross_section']-1))
        require(defect > 1e-7, f'{name} must reach and fail the physical assertion')
        controls[name] = dict(defect=defect, rejected=True, observation=bad)
    p, q = np.array([.13, -.21, .17]), np.array([.07, .11, -.09])
    defect = np.linalg.norm(np.einsum('i,ijk->jk', model.qhat(q), model.current(p+q, p))
                            -(model.h(p+q)-model.h(p)))
    require(defect < 2e-12, 'off-shell finite-q primitive path identity')
    density_current_defect = 0.
    for momentum in (p, p*2, -p):
        initial_values, initial_vectors = np.linalg.eigh(model.h(momentum))
        final_values, final_vectors = np.linalg.eigh(model.h(momentum+q))
        density = final_vectors.conj().T @ initial_vectors
        longitudinal = final_vectors.conj().T @ np.einsum(
            'i,ijk->jk', model.qhat(q), model.current(momentum+q, momentum)) @ initial_vectors
        expected = (final_values[:, None]-initial_values[None, :])*density
        density_current_defect = max(density_current_defect, float(np.max(abs(longitudinal-expected))))
    require(density_current_defect < 2e-12, 'mixed density-current spectral Ward identity')
    radial_probe = {}
    if dynamic:
        _, _, _, outgoing = model.kinematics(.4, 1.3, directions[1], 0, 0)
        ray, ray_error = model.photon_radial_velocity(outgoing, 0)
        full = model.photon_velocity(outgoing, 0)
        projected = float(np.dot(full, outgoing/np.linalg.norm(outgoing)))
        relative_error = abs(ray-projected)/max(abs(projected), np.finfo(float).tiny)
        require(relative_error < 1e-5, 'actual full-gradient projection agrees with outgoing ray derivative')
        radial_probe = dict(ray_velocity=ray, full_group_velocity=full.tolist(),
                            full_gradient_projection=projected,
                            ray_resolution_error=ray_error,
                            relative_projection_error=relative_error,
                            incorrect_norm_substitution_defect=abs(np.linalg.norm(full)/ray-1))
    orientation = [dict(x=x, theta=theta,
                        relative_rate_spread=abs(rows[2*i]['differential_cross_section']/
                                                 rows[2*i+1]['differential_cross_section']-1))
                   for i, (x, theta) in enumerate((x, theta) for x in energies for theta in angles)]
    return dict(a=model.a, rest_gap=model.mass, matter_linear_speed=model.cm,
                induced_photon_speed=model.c, cone_relative_mismatch=abs(model.c/model.cm-1),
                epsilon=model.epsilon, kappa=model.kappa, residue=model.z,
                charged_species=model.species,
                supplier_capsule_hash=canonical_hash(capsule), supplier_validity=capsule['validity'],
                primitive_current_identity_defect=float(defect), soft=soft,
                mixed_density_current_Ward_defect=density_current_defect,
                kinematics=rows, adverse_controls=controls,
                orientation_discriminator=orientation,
                outgoing_radial_gradient_control=radial_probe,
                finite_energy_unpolarized_shape_within_one_percent=max(r['finite_energy_shape_error'] for r in rows) < .01,
                polarization_shape_max_error=max(r['polarization_shape_error'] for r in rows),
                canonical_photon_mode_gram_max_error=(
                    max(c['canonical_photon_mode_gram_error'] for r in rows for c in r['channels'])
                    if dynamic else 0.),
                mode_queries=len(model.mode_cache),
                scope=('actual spectral RPA photon poles/residues and same microscopic tree current'
                       if dynamic else 'leading local quadratic induced photon; not full dynamic pole'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplier', type=Path, required=True)
    parser.add_argument('--family', type=Path, nargs='*', default=[])
    parser.add_argument('--dynamic', action='store_true', help='Use actual paired canonical spectral photon modes')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    inputs = [args.supplier, *args.family]
    input_hashes = {str(path): sha(path) for path in inputs}
    capsules = [load_capsule(path) for path in inputs]
    results = [transfer(capsule, dynamic=args.dynamic) for capsule in capsules]
    if len(capsules) > 1:
        fixed = ('m', 'r', 'v', 'g0', 'gs', 'flavors', 'inertia')
        for capsule in capsules[1:]:
            require(all(capsule['config'][key] == capsules[0]['config'][key] for key in fixed),
                    'material repair keeps active mass, hopping, charge, Gram, total flavor count and inertia; no speed/Thomson re-fit')
    changes = [dict(from_a=before['a'], to_a=after['a'],
                    from_species=before['charged_species'], to_species=after['charged_species'],
                    kind=('a-only continuum path repair' if before['charged_species'] == after['charged_species']
                          else 'explicit inducing-species mass and lattice-spacing repair'))
               for before, after in zip(results, results[1:])]
    family = dict(repair='actual path/filled-species family with fixed active parameters',
                  lattice_spacings=[row['a'] for row in results],
                  material_changes=changes,
                  cone_mismatches=[row['cone_relative_mismatch'] for row in results],
                  maximum_finite_energy_shape_errors=[
                      max(case['finite_energy_shape_error'] for case in row['kinematics']) for row in results],
                  no_velocity_refit=True,
                  result='calculated bounded family, not a proof of a common-cone limit')
    require(all(sha(ROOT/name) == digest for name, digest in EXECUTED_SOURCE_HASHES.items()),
            'source changed during scientific execution; no retrospective hash claim')
    require(all(sha(path) == input_hashes[str(path)] for path in inputs),
            'supplier input changed during scientific execution')
    result = dict(branch='NEW primitive link inertia/no barecurl filled-sea largeN branch',
                  scientific_execution_epoch='03b34911221ba1898315373edaa7c4c3bcac718462e7fe51a842560eb6933f87',
                  input_hashes=input_hashes,
                  source_hashes=EXECUTED_SOURCE_HASHES,
                  runtime=dict(python=sys.version, executable=sys.executable,
                               executable_sha256=sha(Path(sys.executable).resolve()),
                               numpy=np.__version__, platform=platform.platform()),
                  transfers=results,
                  material_repair=family,
                  assumptions=[
                      'Unit-normalized positive-band spin doublet above the filled negative sea; canonical CAR only in admitted all8 flat P sectors.',
                      'Selected uniform meanPhi Gram is assumed, not a proof of normal/Root transport or exact interacting electron.',
                      'Tagged positive excitation of one of N filled charged flavors; RPA already contains every sea bubble, never reinserted in this tree amplitude.',
                      'Primitive link electric inertia retained, bare magnetic curl removed; original heavy neutral. Canonical quantum algebra and charge quantum declared.',
                      'Leading 1/N amplitude (two vertices plus Peierls contact), rate order1/N²; self-energy, vertex loops and inclusive infrared dressing are not computed.',
                      'Dynamic mode uses paired field-plus-PH canonical states, actual spectral roots/residues/A0 and fixed-state energy-current velocities; local mode is explicitly a quadratic truncation.',
                  ], natural_exit=0)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
