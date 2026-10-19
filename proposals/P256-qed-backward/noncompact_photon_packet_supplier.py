#!/usr/bin/env python3
"""Actual coupled photon preparations and residual-certified finite bands.

Full-space source Krylov candidates are certified only by their WHOLE-H
residuals. No dense Q spectrum, parity restriction or uncomputed-pole claim.
The native3D packet theorem and the finite two-vertex execution have distinct
coefficients and are explicitly labelled in the companion construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from bulk_quantum_selection import BulkQuantumLaw
from noncompact_maxwell_phase import curl, native_jets, phase_constants
from noncompact_native_current import Coupled, accept, adverse, source_operators
from noncompact_photon_response import F, coordinate, encoded
from original_charged_principal import original_frame
from rotation_band import RotationBand
from noncompact_charged_packet_transport import (
    graph, green, packet_seed, source_marks, currents, divergence, symbol, theorem_bounds, weak_family,
)
from noncompact_wilson_photon_envelope import even_loop_family

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
PLAN_REVISION = '14d4836b213e2bdca4c489ec23d8489fdb5ffcb76ebfa1a4fad87f7216b9e8c0'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def hashes() -> dict[str, str]:
    paths = {Path(__file__).resolve(), ROOT / 'noncompact_photon_packet_supplier.md'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.parent == ROOT and path.suffix == '.py':
                paths.add(path)
                if path.with_suffix('.md').exists():
                    paths.add(path.with_suffix('.md'))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def ladder(model: Coupled, vector: np.ndarray, creation: bool) -> tuple[np.ndarray, float]:
    wave = vector.reshape(model.n, model.l, model.d)
    out = np.zeros_like(wave)
    for level in range(model.n - 1):
        if creation:
            out[level + 1] = math.sqrt(level + 1) * wave[level]
        else:
            out[level] = math.sqrt(level + 1) * wave[level + 1]
    leakage = math.sqrt(model.n) * float(np.linalg.norm(wave[-1])) if creation else 0.
    return out.ravel(), leakage


def finite_band_candidate(model: Coupled, E0: float, q: np.ndarray,
                          steps: int, delta: float) -> dict[str, Any]:
    """Reorthogonalized full-space source filter, not a spectral completeness claim."""
    norm = float(np.linalg.norm(q))
    basis = np.zeros((model.size, steps), dtype=np.complex128)
    images = np.zeros_like(basis)
    basis[:, 0] = q / norm
    count = 0
    for column in range(steps):
        image = model.action(basis[:, column]) - E0 * basis[:, column]
        images[:, column] = image
        count = column + 1
        if count == steps:
            break
        active = basis[:, :count]
        remainder = image.copy()
        for _ in range(2):
            remainder -= active @ (active.conj().T @ remainder)
        beta = float(np.linalg.norm(remainder))
        if beta < 2e-13:
            break
        basis[:, count] = remainder / beta
    basis, images = basis[:, :count], images[:, :count]
    small = basis.conj().T @ images
    small = (small + small.conj().T) / 2
    values, vectors = np.linalg.eigh(small)
    overlaps = vectors.conj().T @ (basis.conj().T @ q)
    bright = int(np.argmax(abs(overlaps)**2))
    u = basis @ vectors[:, bright]
    u /= np.linalg.norm(u)
    energy = float(np.vdot(u, model.action(u)).real)
    residual = float(np.linalg.norm(model.action(u) - energy * u))
    overlap = float(abs(np.vdot(u, q)) / norm)
    lower = max(0., overlap - residual / delta)**2
    if lower < .90:
        raise AssertionError(f'Actual full-H finite band certificate is not useful: {lower}')
    ground_bound = model.full_residual_bound(q / norm, E0 + model.omega)
    candidate_bound = model.full_residual_bound(u, energy)
    infinite_residual = candidate_bound['full_infinite_comparison_residual_upper']
    infinite_lower = max(0., overlap - infinite_residual / delta)**2
    return dict(filter_dimension=count, full_dimension=model.size,
                candidate_energy_absolute=energy, candidate_excitation=energy-E0,
                whole_H_candidate_residual=residual, normalized_source_overlap=overlap,
                band_half_width=delta, energy_interval_relative_to_finite_E0=[energy-E0-delta, energy-E0+delta],
                entire_finite_H_Q_band_mass_fraction_lower=lower,
                all_other_finite_H_Q_weight_fraction_upper=1-lower,
                total_connected_projected_Q_weight=norm**2,
                Q_band_weight_lower=lower*norm**2,
                infinite_comparison_candidate_bound=candidate_bound,
                embedded_projected_Q_infinite_band_fraction_lower=infinite_lower,
                infinite_band_energy_origin='finite E0 only; actual infinite ground energy not certified',
                projected_Q_preparation_infinite_residual=ground_bound,
                quadrature_nodes=values.tolist(), quadrature_weights=(abs(overlaps)**2).tolist(),
                quadrature_status='filter diagnostics only; no assertion about uncomputed eigenvalues or residues',
                certificate='spectral theorem: ||P_out u||<=r/delta; |<u,q>|<=||P_band q||+r||q||/delta')


def joined_source_case(neutral: Coupled, E0: float, psi: np.ndarray,
                       args: argparse.Namespace, frame: np.ndarray,
                       neutral_photon: list[dict[str, Any]]) -> dict[str, Any]:
    """Continue the selected finite inclusive same-current charge/photon join."""
    charged = Coupled(neutral.n,neutral.M,args.zeta,5,args.UE,args.K,frame)
    operators = source_operators(neutral,charged)
    wave = neutral.reshape(psi)
    sources = [np.array([wave@t.T for t in label]) for label in operators]
    alpha = np.array([1.,.3j,-.2+.4j,.6])
    alpha /= np.linalg.norm(alpha)
    mixed = np.einsum('a,abcd->bcd',alpha,np.asarray(sources))
    norm = float(np.linalg.norm(mixed))
    accept(abs(norm**2-2), 'joined actual complex quartet full-reference Gram', 2e-7)
    chi = mixed.reshape(4,charged.size).T/norm
    created = np.column_stack([ladder(charged,chi[:,b],True)[0] for b in range(4)])
    photon_norm = float(np.linalg.norm(created))
    joined = created/photon_norm
    annihilated = np.column_stack([ladder(charged,chi[:,b],False)[0] for b in range(4)])
    top = sum(float(np.linalg.norm(chi[:,b].reshape(charged.n,charged.l,charged.d)[-1])**2)
              for b in range(4))
    ccr_defect = charged.n*top
    ccr_error = accept(abs(photon_norm**2-np.linalg.norm(annihilated)**2-(1-ccr_defect)),
                       'joined source paid truncated CCR', 2e-8)
    creation_leak = math.sqrt(ccr_defect)
    trace = charged.d*np.sum(charged.phenergy)+charged.phdim*(np.sum(charged.coulomb)+np.trace(charged.onsite).real)
    outgoing = np.ones(3)
    rows = []
    for time, neutral_row in zip(args.times,neutral_photon):
        propagated = chi if time==0 else expm_multiply(-1j*time*charged.operator,chi,
                                                       traceA=-1j*time*trace)
        propagated_joined = joined if time==0 else expm_multiply(-1j*time*charged.operator,joined,
                                                                traceA=-1j*time*trace)
        comparison = np.exp(-1j*time*charged.omega)*np.column_stack(
            [ladder(charged,propagated[:,b],True)[0] for b in range(4)])/photon_norm
        comparison_norm = float(np.linalg.norm(comparison))
        detector = comparison/comparison_norm
        receiver = float(abs(np.vdot(detector,propagated_joined))**2)
        photon_history_error = float(np.linalg.norm(propagated_joined-comparison))
        branches = []
        for state in (propagated,propagated_joined):
            accept(abs(np.vdot(state,state).real-1), 'actual joined full-reference full-H norm', 2e-8)
            current = sum(np.vdot(state[:,b],charged.current(state[:,b],F)) for b in range(4))
            flux = sum(np.vdot(state[:,b],charged.current(state[:,b],outgoing)) for b in range(4))
            density = sum(float(np.sum(abs(charged.reshape(state[:,b]))**2@charged.nx)) for b in range(4))-2
            energy = sum(np.vdot(state[:,b],charged.action(state[:,b])).real for b in range(4))-E0
            branches.append(dict(transverse_current=encoded(current),outgoing_current=encoded(flux),
                                 charge_x=density,charge_R=1-density,actual_Rayleigh_excess=float(energy)))
        connected_current = (sum(np.vdot(propagated_joined[:,b],charged.current(propagated_joined[:,b],F))
                                 -np.vdot(propagated[:,b],charged.current(propagated[:,b],F)) for b in range(4))
                             -complex(neutral_row['actual_same_current']['real'],neutral_row['actual_same_current']['imag'])
                             +np.vdot(psi,neutral.current(psi,F)))
        rows.append(dict(time=time,charge_only=branches[0],charge_plus_photon=branches[1],
                         actual_four_history_connected_transverse_current=encoded(connected_current),
                         actual_joined_vs_photon_on_propagated_charge_norm_error=photon_history_error,
                         actual_joined_receiver_probability=receiver,
                         charged_minus_neutral_photon_receiver_probability=receiver-neutral_row['normalized_prepared_detector_probability'],
                         free_packet_comparison_norm=comparison_norm))
    return dict(source_coefficients=encoded(alpha),complete_references=4,charged_dimension=charged.size,
                actual_source_norm_squared=norm**2,actual_photon_on_charge_norm_squared=photon_norm**2,
                source_truncated_CCR_defect=ccr_defect,source_truncated_CCR_error=ccr_error,
                exact_creation_output_leakage=creation_leak,propagation=rows,
                scope='executed finite inclusive same-current joined histories/receiver; no 3D nonforward recoil or S-matrix')


def coupled_case(n: int, M: int, args: argparse.Namespace, frame: np.ndarray) -> dict[str, Any]:
    model = Coupled(n, M, args.zeta, 4, args.UE, args.K, frame)
    values, vectors, ground_residuals = model.solve(2)
    E0, psi = float(values[0]), vectors[:, 0]
    if values[1] - E0 < 1e-7:
        raise AssertionError('Actual finite neutral ground requires complete degenerate projector')
    Qpsi, qleak = coordinate(model, psi)
    qmean = np.vdot(psi, Qpsi)
    q = Qpsi - qmean * psi
    created, creation_leak = ladder(model, psi, True)
    annihilated, _ = ladder(model, psi, False)
    cnorm, anorm = float(np.linalg.norm(created)), float(np.linalg.norm(annihilated))
    top = float(np.linalg.norm(psi.reshape(n, model.l, model.d)[-1])**2)
    ccr_defect = n * top
    ccr_error = accept(abs(cnorm**2 - anorm**2 - (1-ccr_defect)), 'actual truncated CCR expectation')
    Jpsi = model.current(psi, F)
    jmean = np.vdot(psi, Jpsi)
    jvariance = float(np.vdot(Jpsi, Jpsi).real - abs(jmean)**2)
    s = math.sqrt(model.UE / model.omega)
    Hpsi = model.action(psi)
    ground_error = Hpsi - E0 * psi
    plus = model.action(created) - (E0 + model.omega) * created
    minus = model.action(annihilated) - (E0 - model.omega) * annihilated
    # These defects are measured, NOT set to zero by a cutoff CCR assumption.
    plus_boundary = plus - s * Jpsi - ladder(model, ground_error, True)[0]
    minus_boundary = minus + s * Jpsi - ladder(model, ground_error, False)[0]
    plus_boundary_norm = float(np.linalg.norm(plus_boundary))
    minus_boundary_norm = float(np.linalg.norm(minus_boundary))
    canonical_error = accept(np.linalg.norm(Qpsi-s*(created+annihilated)), 'actual projected canonical coordinate')
    direct_residual = float(np.linalg.norm(plus)) / cnorm
    duhamel_bound = (s*float(np.linalg.norm(Jpsi))+plus_boundary_norm
                    +float(np.linalg.norm(ladder(model, ground_error, True)[0]))) / cnorm
    accept(max(0., direct_residual-duhamel_bound), 'paid current plus Galerkin boundary propagation envelope')
    wrong_sign = adverse(np.linalg.norm(plus+s*Jpsi), 'actual creation identity with paid boundary',
                         max(2e-8, 2*plus_boundary_norm))
    initial = created / cnorm
    trace = model.d * np.sum(model.phenergy) + model.phdim * (np.sum(model.coulomb)+np.trace(model.onsite).real)
    propagation = []
    for time in args.times:
        evolved = initial if time == 0 else expm_multiply(-1j*time*model.operator, initial,
                                                         traceA=-1j*time*trace)
        bare_history = np.exp(-1j*time*(E0+model.omega))*initial
        defect = float(np.linalg.norm(evolved-bare_history))
        accept(max(0., defect-abs(time)*duhamel_bound), 'actual full-H photon Duhamel propagation', 2e-8)
        norm_error = accept(abs(np.vdot(evolved,evolved).real-1), 'actual full-H photon preparation norm')
        detection = float(abs(np.vdot(bare_history,evolved))**2)
        current = np.vdot(evolved, model.current(evolved,F))
        energy = float(np.vdot(evolved,model.action(evolved)).real-E0)
        propagation.append(dict(time=time, full_H_vs_free_single_mode_norm_error=defect,
                                paid_Duhamel_norm_bound=abs(time)*duhamel_bound,
                                normalized_prepared_detector_probability=detection,
                                actual_same_current=encoded(current), actual_excitation_energy=energy,
                                norm_error=norm_error))
    band = finite_band_candidate(model,E0,q,args.filter_steps,args.band_delta)
    joined = joined_source_case(model,E0,psi,args,frame,propagation)
    return dict(n=n,M=M,occupation=4,dimension=model.size,zeta=args.zeta,omega_bare=model.omega,
                E0=E0,actual_ground_gap=float(values[1]-E0),ground_residuals=ground_residuals,
                ground_full_infinite_comparison=model.full_residual_bound(psi,E0),
                neutral_state_sha256=array_hash(psi), neutral_state_owner='same full coupled finite H; not product vacuum',
                projected_Q_mean=encoded(qmean),connected_Q_norm_squared=float(np.vdot(q,q).real),
                exact_unbounded_Q_output_leakage=qleak,
                actual_current_mean=encoded(jmean),actual_current_variance=jvariance,
                actual_current_norm_squared=float(np.vdot(Jpsi,Jpsi).real),
                creation_norm_squared=cnorm**2,annihilation_norm_squared=anorm**2,
                exact_unbounded_creation_output_leakage=creation_leak,
                truncated_CCR_defect=ccr_defect,truncated_CCR_identity_error=ccr_error,
                projected_coordinate_identity_error=canonical_error,
                measured_creation_displacement_boundary=plus_boundary_norm,
                measured_annihilation_displacement_boundary=minus_boundary_norm,
                current_only_preparation_residual=s*float(np.linalg.norm(Jpsi)),
                paid_full_finite_H_preparation_residual=direct_residual,
                created_state_full_infinite_comparison=model.full_residual_bound(initial,E0+model.omega),
                wrong_creation_current_sign=wrong_sign,
                photon_band=band,actual_preparation_propagation=propagation,selected_joined_successor=joined,
                scope='actual two-vertex rank-one curl coupled neutral source/current/preparation/detector/time evolution; not native3D propagation')


def native_source(size: int, centre: tuple[int,int,int]) -> dict[str, Any]:
    """Native source/metric/Wick anchor, not a coupled quantum vacuum sample."""
    band = RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    selected, frame = original_frame(band,size,np.asarray(centre)*math.pi/2,True)
    law = BulkQuantumLaw(size,band=band)
    fields = np.tile(frame,(law.volume,1,1,1))
    links = law.toron_links(np.asarray(centre)*math.pi/2)
    h = np.asarray(law.onebody(fields,links,np.zeros((law.volume,3))),complex)
    source = np.zeros((law.volume,3))
    source[:,1] = np.cos(2*math.pi*law.sites[:,0]/size)
    source /= np.linalg.norm(source)
    divergence = np.zeros(law.volume)
    for axis in range(3):
        divergence += source[:,axis]
        np.add.at(divergence,law.target[:,axis],-source[:,axis])
    q = 2*math.sin(math.pi/size)
    accept(np.linalg.norm(divergence), 'actual native normalized transverse photon source')
    accept(abs(np.linalg.norm(curl(law,source))**2-q*q), 'native curl eigenmode frequency normalization')
    first,contact = native_jets(law,h,source)
    energies,vectors = np.linalg.eigh(h)
    occupied = vectors[:,energies<0]
    empty = vectors[:,energies>0]
    wick = float(np.linalg.norm(empty.conj().T@first@occupied)**2)
    eps = 1e-5
    hp = np.asarray(law.onebody(fields,links,eps*source),complex)
    hm = np.asarray(law.onebody(fields,links,-eps*source),complex)
    current_error = accept(np.linalg.norm((hp-hm)/(2*eps)-first), 'native original same-current first jet')
    constants = phase_constants(.01,.1)
    return dict(N=size,centre=centre,true_Gram=np.asarray(selected['G']).tolist(),
                full_native_modes=4*law.volume,whole_half_number=int(np.sum(energies<0)),
                source_sha256=array_hash(source),native_h_sha256=array_hash(h),
                normalized_transverse_source_norm=float(np.linalg.norm(source)),curl_eigenvalue=q*q,
                native_bare_packet_frequency=constants['bare_speed']*q,
                actual_native_current_jet_error=current_error,
                static_whole_sea_equal_time_Wick_current_variance=wick,
                native_contact_expectation=float(np.trace(occupied.conj().T@contact@occupied).real),
                phase_family=constants,
                scope='native physical endpoint/true metric/source/current and exact whole-sea conditional Wick anchor; not numeric coupled native3D ground variance')


def photon_beam(law: Any, axis: int, centre: np.ndarray, speed: float) -> dict[str, Any]:
    """Finite-width transverse beam on the actual periodic curl dispersion."""
    signed = (law.sites+law.size//2)%law.size-law.size//2
    spacing = 2*math.pi/law.size
    central = np.eye(3)[axis]*(2*math.pi/5)
    entries = []
    for mode in signed:
        if mode[axis]<=0 or mode[axis]<=max(abs(mode[j]) for j in range(3) if j!=axis):
            continue
        k = spacing*mode
        distance = float(np.linalg.norm(k-central)/(2*math.pi/5))
        if distance>2.01:
            continue
        d = 1-np.exp(-1j*k)
        q = float(np.linalg.norm(d))
        if not .08<=speed*q<=.26:
            continue
        polar = np.array([0.,0.,1.],complex)-d.conj()*d[2]/q**2
        if np.linalg.norm(polar)<1e-10:
            continue
        polar /= np.linalg.norm(polar)
        coefficient = math.exp(-distance**2/(2*.9**2))*np.exp(-1j*k@centre)
        entries.append(dict(mode=mode.copy(),k=k,polar=polar,omega=speed*q,
                            coefficient=coefficient,frequency_sign=1))
    entries = sorted(entries,key=lambda row:abs(row['coefficient']),reverse=True)[:8]
    norm = math.sqrt(sum(abs(row['coefficient'])**2 for row in entries))
    for row in entries:
        row['coefficient'] /= norm
    velocity = sum(abs(row['coefficient'])**2*speed*np.sin(row['k'])/(row['omega']/speed)
                   for row in entries)
    return dict(modes=entries,axis=axis,centre=centre,mean_group_velocity=velocity,
                omega_min=min(row['omega'] for row in entries),
                omega_max=max(row['omega'] for row in entries),coefficient_norm=1.)


def beam_field(law: Any, beam: dict[str, Any], time: float, coordinate: bool) -> np.ndarray:
    out = np.zeros((law.volume,3),complex)
    for row in beam['modes']:
        coefficient = row['coefficient']*np.exp(-1j*row['omega']*row['frequency_sign']*time)
        if coordinate:
            coefficient /= math.sqrt(row['omega'])
        out += coefficient*np.exp(1j*law.sites@row['k'])[:,None]*row['polar'][None,:]/math.sqrt(law.volume)
    return out


def real_beam(beam: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in beam['modes']:
        rows.append({**row,'coefficient':row['coefficient']/math.sqrt(2)})
        rows.append({**row,'mode':-row['mode'],'k':-row['k'],'polar':row['polar'].conj(),
                     'coefficient':row['coefficient'].conjugate()/math.sqrt(2),'frequency_sign':-1})
    return {**beam,'modes':rows}


class NativeMomentum:
    """Exact full4V native harmonic frame, not a positive-band Hamiltonian."""
    def __init__(self, law: Any, h: np.ndarray, frame: np.ndarray, centre: tuple[int,...],
                 harmonic: np.ndarray):
        self.law = law
        self.momenta = 2*math.pi*law.sites/law.size
        angles = np.asarray(centre)*math.pi/2+law.size*harmonic
        self.blocks = symbol(frame,self.momenta,angles,law.size)
        self.values,self.vectors = np.linalg.eigh(self.blocks)
        self.occupied = (self.values<0)
        self.endpoints = [h[:4,4*int(law.target[0,j]):4*int(law.target[0,j])+4] for j in range(3)]
        accept(np.linalg.norm(np.ravel_multi_index(law.sites.T,(law.size,)*3)-np.arange(law.volume)),
               'native coordinate enumeration for exact FFT')
        rng = np.random.default_rng(2221061+law.size)
        trial = rng.normal(size=(law.volume,4,1))+1j*rng.normal(size=(law.volume,4,1))
        trial /= np.linalg.norm(trial)
        exact = self.to_momentum(h@self.to_position(trial).reshape(4*law.volume,1))
        accept(np.linalg.norm(exact-np.einsum('pab,pbr->par',self.blocks,trial)),
               'actual native endpoint including seams versus entire momentum H')

    def to_momentum(self, wave: np.ndarray) -> np.ndarray:
        shaped = wave.reshape((self.law.size,)*3+(4,-1))
        return (np.fft.fftn(shaped,axes=(0,1,2))/math.sqrt(self.law.volume)).reshape(self.law.volume,4,-1)

    def to_position(self, wave: np.ndarray) -> np.ndarray:
        shaped = wave.reshape((self.law.size,)*3+(4,-1))
        return (np.fft.ifftn(shaped,axes=(0,1,2))*math.sqrt(self.law.volume)).reshape(self.law.volume,4,-1)

    def evolve(self, wave: np.ndarray, time: float) -> np.ndarray:
        coefficients = np.einsum('pba,pbr->par',self.vectors.conj(),wave)
        coefficients *= np.exp(-1j*time*self.values)[:,:,None]
        return np.einsum('pab,pbr->par',self.vectors,coefficients)

    def project(self, wave: np.ndarray, occupied: bool) -> np.ndarray:
        coefficients = np.einsum('pba,pbr->par',self.vectors.conj(),wave)
        coefficients *= (self.occupied if occupied else ~self.occupied)[:,:,None]
        return np.einsum('pab,pbr->par',self.vectors,coefficients)

    def shift(self, wave: np.ndarray, mode: np.ndarray) -> np.ndarray:
        shaped = wave.reshape((self.law.size,)*3+wave.shape[1:])
        return np.roll(shaped,tuple(int(k) for k in mode),axis=(0,1,2)).reshape(wave.shape)

    def current(self, beam: dict[str, Any], time: float, wave: np.ndarray) -> np.ndarray:
        out = np.zeros_like(wave)
        for row in beam['modes']:
            coefficient = row['coefficient']*np.exp(-1j*row['frequency_sign']*row['omega']*time)
            coefficient /= math.sqrt(self.law.volume*row['omega'])
            added = np.zeros_like(wave)
            for axis,T in enumerate(self.endpoints):
                forward = np.einsum('ab,pbr->par',T,wave)*np.exp(1j*self.momenta[:,axis])[:,None,None]
                reverse = np.einsum('ab,pbr->par',T.conj().T,wave)*np.exp(-1j*(self.momenta[:,axis]+row['k'][axis]))[:,None,None]
                added += 1j*row['polar'][axis]*(forward-reverse)
            out += coefficient*self.shift(added,row['mode'])
        return out

    def contact(self, first: dict[str, Any], second: dict[str, Any], time: float,
                wave: np.ndarray) -> np.ndarray:
        out = np.zeros_like(wave)
        for a in first['modes']:
            for b in second['modes']:
                mode,k = a['mode']+b['mode'],a['k']+b['k']
                coefficient = (a['coefficient']*b['coefficient']
                               *np.exp(-1j*time*(a['frequency_sign']*a['omega']+b['frequency_sign']*b['omega']))
                               /(self.law.volume*math.sqrt(a['omega']*b['omega'])))
                added = np.zeros_like(wave)
                for axis,T in enumerate(self.endpoints):
                    forward = np.einsum('ab,pbr->par',T,wave)*np.exp(1j*self.momenta[:,axis])[:,None,None]
                    reverse = np.einsum('ab,pbr->par',T.conj().T,wave)*np.exp(-1j*(self.momenta[:,axis]+k[axis]))[:,None,None]
                    added -= a['polar'][axis]*b['polar'][axis]*(forward+reverse)
                out += coefficient*self.shift(added,mode)
        return out

    def interaction_current(self, beam: dict[str, Any], time: float, wave: np.ndarray) -> np.ndarray:
        return self.evolve(self.current(beam,time,self.evolve(wave,time)),-time)


def pulse(time: float, duration: float) -> float:
    return math.sin(math.pi*time/duration)**2


def receiver_pulse(time: float, duration: float) -> float:
    return pulse(time,duration)*math.exp(-((time-.8*duration)/(.16*duration))**2)


def window_kernel(momentum: NativeMomentum, incoming: dict[str, Any], outgoing: dict[str, Any],
                  source: np.ndarray, duration: float, nodes: int) -> dict[str, Any]:
    """Exact full-CAR charge-minus-neutral retarded current/contact coefficient."""
    x,w = np.polynomial.legendre.leggauss(nodes)
    times,weights = (x+1)*duration/2,w*duration/2
    contact = np.zeros_like(source)
    ordered = np.zeros_like(source)
    reverse = np.zeros_like(source)
    for time,weight in zip(times,weights):
        dtweight = weight*receiver_pulse(float(time),duration)
        evolved = momentum.evolve(source,float(time))
        contact += dtweight*pulse(float(time),duration)*momentum.evolve(
            momentum.contact(incoming,outgoing,float(time),evolved),-float(time))
        jt = momentum.interaction_current(outgoing,float(time),source)
        inner_times,inner_weights = (x+1)*time/2,w*time/2
        for earlier,inner_weight in zip(inner_times,inner_weights):
            js = momentum.interaction_current(incoming,float(earlier),source)
            scale = dtweight*inner_weight*pulse(float(earlier),duration)
            ordered += scale*momentum.interaction_current(outgoing,float(time),js)
            reverse += scale*momentum.interaction_current(incoming,float(earlier),jt)
    full = contact-1j*(ordered-reverse)
    return dict(full=full,contact=contact,first=ordered,second=reverse,
                scalar=complex(np.vdot(source,full)),
                scalar_contact=complex(np.vdot(source,contact)),
                scalar_first=complex(np.vdot(source,-1j*ordered)),
                scalar_second=complex(np.vdot(source,1j*reverse)))


def physical_drive_response(momentum: NativeMomentum, h: np.ndarray, incoming: dict[str, Any],
                            outgoing: dict[str, Any], source: np.ndarray,
                            duration: float, steps: int, epsilon: float) -> float:
    """Independent REAL Peierls drive of entire native onebody, source-traced CAR."""
    law = momentum.law
    initial = momentum.to_position(source).reshape(4*law.volume,-1)
    endpoints = [(x,int(law.target[x,j]),j) for x in range(law.volume) for j in range(3)]
    dt = duration/steps
    values = []
    for sign in (1,-1):
        state = initial.copy()
        observable = 0.
        for step in range(steps):
            time = (step+.5)*dt
            field = np.real(beam_field(law,incoming,time,True))*pulse(time,duration)
            detector = np.real(beam_field(law,outgoing,time,True))*receiver_pulse(time,duration)
            shifted_h = h.copy()
            for a,b,j in endpoints:
                sa,sb = slice(4*a,4*a+4),slice(4*b,4*b+4)
                shifted_h[sa,sb] = h[sa,sb]*np.exp(1j*sign*epsilon*field[a,j])
                shifted_h[sb,sa] = shifted_h[sa,sb].conj().T
            operator = csr_matrix(shifted_h)
            middle = expm_multiply(-.5j*dt*operator,state,traceA=-.5j*dt*np.trace(shifted_h))
            current,_ = native_jets(law,shifted_h,detector)
            observable += dt*float(np.vdot(middle,current@middle).real)
            state = expm_multiply(-1j*dt*operator,state,traceA=-1j*dt*np.trace(shifted_h))
        values.append(observable)
    return (values[0]-values[1])/(2*epsilon)


def native_receivers_case(size: int, args: argparse.Namespace) -> dict[str, Any]:
    centre = (1,2,3)
    band = RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    selected,frame = original_frame(band,size,np.asarray(centre)*math.pi/2,True)
    law,h,baseline = graph(size,centre,frame,band)
    fields = np.tile(frame,(law.volume,1,1,1))
    links = law.toron_links(np.asarray(centre)*math.pi/2)
    g,_,_ = green(law)
    eta,preparation = packet_seed(law,.8,np.array([1.2,0.,0.]))
    origin = np.asarray(preparation['origin'])
    reference = int(law.index[tuple((origin+np.array([-2,-2,-2]))%size)])
    harmonic = np.zeros(3)
    marks = source_marks(law,fields,links,eta,reference,baseline,harmonic,g)
    momentum = NativeMomentum(law,h,frame,centre,harmonic)
    alpha = np.array([1.,.3j,-.2+.4j,.6])
    alpha /= np.linalg.norm(alpha)
    source = np.einsum('a,air->ir',alpha,marks)
    source = momentum.project(momentum.to_momentum(source),False)
    norm = float(np.linalg.norm(source))
    accept(abs(norm**2-2), 'complete reference harmonic-covariant native source norm')
    source /= norm
    constants = weak_family(.01,.1,1.,'conservative')
    incoming = photon_beam(law,0,origin,constants['bare_speed'])
    duration = 1/float(incoming['mean_group_velocity'][0])
    forward_centre = (origin+np.array([1,0,0]))%size
    recoil_centre = (origin+np.array([0,1,0]))%size
    forward = photon_beam(law,0,forward_centre,constants['bare_speed'])
    outgoing = photon_beam(law,1,recoil_centre,constants['bare_speed'])
    field = beam_field(law,incoming,0.,False)
    accept(abs(np.linalg.norm(field)-1), 'actual normalized native transverse finite-width photon packet')
    accept(np.linalg.norm(divergence(law,field)), 'actual curl transverse beam including seams')
    real_in,real_out = real_beam(incoming),real_beam(outgoing)
    witness = momentum.evolve(source,.37)
    jfield = beam_field(law,real_in,.37,True)
    j,_ = native_jets(law,h,jfield)
    exact_j = momentum.to_momentum(j@momentum.to_position(witness).reshape(4*law.volume,-1))
    j_error = float(np.linalg.norm(exact_j-momentum.current(real_in,.37,witness)))
    accept(j_error, 'packet current full native endpoint versus momentum-shift vertex')
    cfield = beam_field(law,real_out,.37,True)
    cross = np.zeros_like(h)
    for x in range(law.volume):
        for axis in range(3):
            y = int(law.target[x,axis])
            sx,sy = slice(4*x,4*x+4),slice(4*y,4*y+4)
            cross[sx,sy] = -jfield[x,axis]*cfield[x,axis]*h[sx,sy]
            cross[sy,sx] = -jfield[x,axis]*cfield[x,axis]*h[sy,sx]
    exact_c = momentum.to_momentum(cross@momentum.to_position(witness).reshape(4*law.volume,-1))
    c_error = float(np.linalg.norm(exact_c-momentum.contact(real_in,real_out,.37,witness)))
    accept(c_error, 'packet cross-contact full native endpoint versus momentum-shift vertex')
    coarse = window_kernel(momentum,real_in,real_out,source,duration,args.window_nodes)
    fine = window_kernel(momentum,real_in,real_out,source,duration,2*args.window_nodes)
    quadrature = abs(fine['scalar']-coarse['scalar'])
    fd = physical_drive_response(momentum,h,real_in,real_out,source,duration,args.drive_steps,1e-4)
    fd_fine = physical_drive_response(momentum,h,real_in,real_out,source,duration,2*args.drive_steps,1e-4)
    fd_error = abs(fd-fd_fine)
    tolerance = max(2e-7,4*fd_error+4*quadrature)
    accept(abs(fine['scalar'].real-fd_fine), 'two-window full-CAR current kernel versus actual Peierls-drive derivative',tolerance)
    missing_contact = adverse(abs(fine['scalar'].real-fine['scalar_contact'].real-fd_fine),
                              'two-window full-CAR current kernel versus actual Peierls-drive derivative',tolerance)
    transfer = 2*math.pi/size*np.array([1.,-1.,0.])
    recoil_eta = eta*np.exp(1j*(law.sites-law.sites[reference])@transfer)
    recoil_marks = source_marks(law,fields,links,recoil_eta,reference,baseline,harmonic,g)
    recoil_source = momentum.project(momentum.to_momentum(np.einsum('a,air->ir',alpha,recoil_marks)),False)
    recoil_norm = float(np.linalg.norm(recoil_source))
    recoil_source /= recoil_norm
    recoil_response = complex(np.vdot(recoil_source,fine['full']))
    flux = []
    for time in sorted(set([0.,.25,.5,.75,1.,duration])):
        moved = momentum.to_position(momentum.evolve(source,time))
        density = np.sum(abs(moved)**2,axis=(1,2))
        current = currents(law,h,moved.reshape(4*law.volume,4))
        selected_region = law.sites[:,0] == int(forward_centre[0])
        outward = float(np.sum(divergence(law,current)[selected_region]))
        derivative = momentum.to_position(-1j*np.einsum('pab,pbr->par',momentum.blocks,momentum.evolve(source,time)))
        density_derivative = 2*np.real(np.sum(moved.conj()*derivative,axis=(1,2)))
        accept(np.linalg.norm(density_derivative+divergence(law,current)), 'native complete-reference physical charge/current continuity')
        flux.append(dict(time=time,receiver_plane_charge=float(np.sum(density[selected_region])),
                         outgoing_plane_flux=outward,reference_charge=1.,
                         excess_total_charge=float(density.sum()),
                         density_sha256=array_hash(density),current_sha256=array_hash(current)))
    beam_flights = []
    for time in [0.,duration]:
        propagated = beam_field(law,incoming,time,False)
        forward_field = beam_field(law,forward,0.,False)
        outgoing_field = beam_field(law,outgoing,0.,False)
        beam_flights.append(dict(time=time,forward_free_packet_probability=float(abs(np.vdot(forward_field,propagated))**2),
                                 nonforward_free_packet_probability=float(abs(np.vdot(outgoing_field,propagated))**2)))
    central = 2*math.pi/size*np.array([1.,0.,0.])
    energy_in = float(np.vdot(source,np.einsum('pab,pbr->par',momentum.blocks,source)).real)
    energy_recoil = float(np.vdot(recoil_source,np.einsum('pab,pbr->par',momentum.blocks,recoil_source)).real)
    return dict(N=size,modes=4*law.volume,whole_neutral_number=2*law.volume,
                centre=centre,true_Gram=np.asarray(selected['G']),harmonic=harmonic,
                origin=origin,reference_site=law.sites[reference],prepared_coefficients=encoded(alpha),
                source_norm_squared=norm**2,recoil_conditional_source_norm_squared=recoil_norm**2,
                recoil_actual_ground_metric='actual CT norm2, not inferred from conditional free source norm',
                incoming_photon=incoming,outgoing_receiver=outgoing,
                receiver_centres=[forward_centre.tolist(),recoil_centre.tolist()],
                actual_lattice_kinematics=dict(incoming_k=central,outgoing_k=central[[1,0,2]],
                    recoil_transfer=transfer,central_photon_energy=constants['bare_speed']*np.linalg.norm(1-np.exp(1j*central)),
                    charged_recoil_energy_mismatch=energy_recoil-energy_in,finite_time_energy_resolution=2*math.pi/duration,
                    on_shell_point_required=False,flight_time=duration),
                current_packet_jet_error=j_error,cross_contact_error=c_error,
                conditional_whole_CAR_two_window_response=dict(coarse=encoded(coarse['scalar']),fine=encoded(fine['scalar']),
                    contact=encoded(fine['scalar_contact']),first_ordered=encoded(fine['scalar_first']),
                    second_ordered=encoded(fine['scalar_second']),Peierls_drive_fd=fd_fine,
                    drive_discretization_error=fd_error,quadrature_difference=quadrature,
                    acceptance_tolerance=tolerance,missing_contact_rejected=missing_contact,
                    recoil_marked_coefficient=encoded(recoil_response),
                    physical_two_photon_coupling=constants['UE'],
                    physical_current_response_coefficient=encoded(constants['UE']*fine['scalar'])),
                complete_reference_charge_flux=flux,finite_width_photon_receiver_flight=beam_flights,
                actual_coupled_Omega_bounds=theorem_bounds(constants,law,eta,reference,duration),
                native_family=constants,
                scope='native exact lattice/source/current/contact/whole-CAR conditional finite-window and recoil comparator; actual coupled Omega packet and receiver lift supplied analytically, not a numeric coupled3D ground simulation')


def bounded_native_packet_certificate(size: int, time: float, omega_min: float = .08,
                                      omega_max: float = .26) -> dict[str, Any]:
    """Finite native full-CAR operator bound, valid on every coupled state."""
    constants = weak_family(.01,.1,1.,'conservative')
    band = RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    _,frame = original_frame(band,size,np.array([1,2,3])*math.pi/2,True)
    # Frobenius triangle bounds each full4x4 endpoint for every unitary link/phase.
    tau = max((.12*np.linalg.norm(frame[0],'fro')+.3*np.linalg.norm(frame[j],'fro'))/2
              for j in range(1,4))
    cj = 192*tau*tau*size**3
    debt = constants['UE']*cj/omega_min**3
    residual = math.sqrt(constants['UE']*cj/omega_min)
    return dict(N=size,a=1.,UE=constants['UE'],K=constants['K'],speed=.1,
                native_frame_sha256=array_hash(frame),endpoint_operator_norm_upper=float(tau),
                current_operator_squared_upper=float(cj),omega_min=omega_min,omega_max=omega_max,
                annihilation_debt_upper=float(debt),creation_Gram_lower=1.,
                creation_Gram_upper_offset=float(debt),observation_time=time,
                normalized_neutral_packet_defect_upper=2*abs(time)*residual,
                normalized_joint_insertion_vs_exact_charge_defect_upper=2*abs(time)*residual,
                photon_on_charge_initial_depletion_upper=2*debt,
                external_coordinate_scale=math.sqrt(constants['UE']),
                two_external_photon_leg_factor=constants['UE'],
                premise='||J_f||<=8 tau ||f||l1<=8 tau sqrt(3V)||f||l2 on full CAR; ANY coupled state',
                infinite_volume_uniform=False,phase_onset_required=False,
                normal_W_error='separate weighted legal-Lambda lift; no uniform scalar-normal operator bound',
                numerical_coupled_ground_simulated=False)


def actual_native_packet_certificate(time: float, omega_min: float = .08,
                                     omega_max: float = .26) -> dict[str, Any]:
    """Coupled-ground current/packet capability at the SAME conservative a1 UE,K."""
    constants = weak_family(.01,.1,1.,'conservative')
    inherited = even_loop_family(.4,.1,tight=True)
    ratio = constants['UE']/inherited['UE']
    if not 0<ratio<=1:
        raise AssertionError('Inherited score-current monotonicity needs weaker UE at the same speed')
    alpha,gamma = inherited['alpha'],inherited['gamma']
    point,jet,H = inherited['point_kernel_bound'],inherited['native_jet'],inherited['H']
    wick = 128*point**2*jet**2*3*math.exp(4*alpha)*(1+2/math.expm1(alpha))**3
    x2 = max(0.,2/alpha-1)
    p2 = (1+x2)**2*math.exp(-alpha*x2)
    score = 2*(8*H)**2*constants['UE']*((2*math.sqrt(3)/.1)*(128*p2)**2+1/(math.e*gamma))
    onset = inherited['N0_seam']
    log_seam = (math.log(constants['UE'])+math.log(onset/(4*.1)+onset**2/8)
                +2*inherited['log_Rstar']-2*(alpha/64)*onset)
    seam = math.exp(log_seam)
    rho = inherited['rho']*ratio
    cj = wick+(score+seam)/(1-rho)
    debt = constants['UE']*cj/omega_min**3
    residual = math.sqrt(constants['UE']*cj/omega_min)
    return dict(a=1.,UE=constants['UE'],K=constants['K'],speed=.1,
                inherited_anchor_admission=dict(s_floor=.4,G00_upper=.43,Gjj_upper=.065,tilt_upper=.002,N_metric=25),
                required_N=max(25,onset),required_inverse_temperature=inherited['tau'],
                actual_coupled_current_variance_upper=cj,omega_min=omega_min,omega_max=omega_max,
                annihilation_debt=debt,creation_Gram_lower=1.,creation_Gram_upper_offset=debt,
                creation_Gram_upper_expression='1 + annihilation_debt (tiny correction retained separately)',
                observation_time=time,unnormalized_packet_defect=time*residual,
                normalized_packet_defect=2*time*residual,
                matched_receiver_probability_error_upper=4*time*residual+4*debt,
                photon_on_charge_initial_depletion_upper=2*debt,
                same_native_family_preserved=True,
                premise='all-background instantaneous Wick/BL incidence bound plus paid zeroT spatial seam; not independent review or numerical coupled ground',
                current_monotonicity='C_Wick unchanged; score/seam scale by UE; Hessian ratio decreases by UE; K increases at fixed speed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs',nargs='+',default=['3:36','5:44'])
    parser.add_argument('--UE',type=float,default=.0001)
    parser.add_argument('--K',type=float,default=28.361421309064075)
    parser.add_argument('--zeta',type=float,default=0.)
    parser.add_argument('--times',nargs='+',type=float,default=[0.,.25,.5,.75,1.])
    parser.add_argument('--filter-steps',type=int,default=80)
    parser.add_argument('--band-delta',type=float,default=.001)
    parser.add_argument('--native-size',type=int,default=5)
    parser.add_argument('--native-receivers-only',action='store_true')
    parser.add_argument('--native-receiver-sizes',nargs='+',type=int,default=[5,7])
    parser.add_argument('--window-nodes',type=int,default=12)
    parser.add_argument('--drive-steps',type=int,default=32)
    parser.add_argument('--native-packet-bound-only',action='store_true')
    parser.add_argument('--observation-time',type=float,default=17.)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.native_packet_bound_only:
        before = hashes()
        bound = actual_native_packet_certificate(args.observation_time)
        finite_bounds = [bounded_native_packet_certificate(size,args.observation_time)
                         for size in args.native_receiver_sizes]
        after = hashes()
        if before!=after:
            raise AssertionError('Packet-current capability source changed during construction')
        result = dict(plan=PLAN,mode='native_actual_Omega_packet_bound',source_sha256=after,
                      runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                                   executable=sys.executable,argv=sys.argv),capability=bound,
                      finite_native_capabilities=finite_bounds,numerical_coupled_ground_simulated=False)
        rendered = json.dumps(result,indent=2,allow_nan=False,default=encoded)
        args.output.write_text(rendered+'\n')
        print(rendered)
        return
    if args.native_receivers_only:
        before = hashes()
        spatial = [native_receivers_case(size,args) for size in args.native_receiver_sizes]
        after = hashes()
        if before!=after:
            raise AssertionError('Native receiver supplier/imported bytes changed during consumer')
        result = dict(plan=PLAN,plan_revision=PLAN_REVISION,mode='native3D_two_window_recoil',
                      source_sha256=after,runtime=dict(python=sys.version,numpy=np.__version__,
                      scipy=scipy.__version__,executable=sys.executable,argv=sys.argv),
                      results=spatial,native3D_coupled_ground_simulated=False,
                      exact_current_contact_conditional_consumer=True,campaign_complete=False)
        rendered = json.dumps(result,indent=2,allow_nan=False,default=encoded)
        args.output.write_text(rendered+'\n')
        print(rendered)
        return
    if args.band_delta<=0 or args.filter_steps<4:
        raise ValueError('positive detector bandwidth and nontrivial full-space filter required')
    band = RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    selected,frame = original_frame(band,5,np.zeros(3),True)
    before = hashes()
    native = native_source(args.native_size,(1,2,3))
    cutoffs = [tuple(int(v) for v in item.split(':')) for item in args.cutoffs]
    if any(len(item)!=2 for item in cutoffs):
        raise ValueError('cutoffs use n:M')
    results = [coupled_case(item[0],item[1],args,frame) for item in cutoffs]
    after = hashes()
    if before != after:
        raise AssertionError('Executed supplier/imported source bytes changed during scientific consumer')
    result = dict(plan=PLAN,plan_revision=PLAN_REVISION,template_revision=TEMPLATE_REVISION,
                  source_sha256=after,runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                                                  platform=platform.platform(),executable=sys.executable,argv=sys.argv),
                  finite_coupled_regime=dict(UE=args.UE,K=args.K,zeta=args.zeta,eta_for_join=.1,
                                            photon_source=F.tolist(),reference='complete bar4 charged counterpart; R is second vertex',
                                            times=args.times,nativeN5_identity_frame_Gram=np.asarray(selected['G']).tolist()),
                  native3D_source_metric_current=native,results=results,
                  distinct_regimes_not_identified=True,isolated_photon_pole_required=False,
                  native3D_coupled_quantum_simulation=False,full_dense_Q_spectrum_run=False,
                  campaign_complete=False)
    rendered=json.dumps(result,indent=2,allow_nan=False)
    args.output.write_text(rendered+'\n')
    print(rendered)


if __name__=='__main__':
    main()
