#!/usr/bin/env python3
"""Original-Phi photon graphs and a normalized noncompact Bloch packet.

The native executable is a Cartesian-slice/oscillator/Fourier finite field
approximation. The full-domain all-mode and selected-family bounds are derived
separately. No finite matrix is called the huge-volume microscopic vacuum.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, cast

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh, expm_multiply

from noncompact_native_current import Coupled, HermitianAction, accept, adverse, exterior
from noncompact_microscopic_charge_window import endpoint_blocks
from noncompact_centre_source_energy import haar_nodes, orientation_cloud
from noncompact_photon_response import encoded
from noncompact_quantitative_normal_budget import (
    native_normal_budget, ceil_log2, sqrt_upper, sqrt_lower, neutral_trial_upper)
from original_charged_principal import original_frame
from rotation_band import RotationBand
from sea_selection import SELECTED, matrix_unit_basis

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
PLAN_REVISION = '1fe2e204d3de07a7abf374db2dc9f0f089c449898797d488a9082386c5c8ee61'


def require(ok: bool, label: str) -> None:
    if not ok:
        raise AssertionError(label)


def source_hashes() -> dict[str, str]:
    paths = {Path(__file__).resolve(), ROOT/'noncompact_original_mic_photon.md'}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and Path(name).resolve().parent == ROOT:
            paths.add(Path(name).resolve())
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def dyadic_upper(x: F) -> str:
    return f'2^{ceil_log2(x)}' if x > 0 else '0'


def selected_certificate(normal: dict[str, Any], scattering: dict[str, Any]) -> dict[str, Any]:
    """Rational outward constants; no allocation proportional to selected V."""
    n = 5*2**96
    v = n**3
    ue = F.from_float(1.5429442570866347e-148)/2**4096
    c = F(1130202988249777, 9007199254740992)
    k = c*c/(2*ue)
    b = native_normal_budget(n)
    # Full-domain quartic: Lambda*a*rho^4 - (Lambda*BH+BL)*rho.
    # Young is applied to the scalar form, NOT to an operator norm of rho^4.
    a = F(1, 4*v)
    cx = sqrt_upper(F(92, 100)**2+3*F(3, 10)**2)
    bh, bl = 4*v*cx, 2*(2*v+1)*cx
    beta = bh+bl
    radius = F(2**max(0, (ceil_log2(2*beta/a)+2)//3))
    supplied = normal['full_Phi_and_photon_graph_capability']
    require(F(supplied['A4']) == a, 'actual selected quartic coefficient')
    cd = max(beta*radius, F(supplied['CD']))
    # The unchanged global tower exceeds the displayed elementary floor.
    A = 100*(100000*v*v)**2
    log_floor = 640*(A+1)
    requested = 524288
    required = max(16*(ceil_log2(4*b.residual_kinetic)+requested),
                   16*(ceil_log2(4*b.residual_light)+requested),
                   16*(b.cutoff_log2+requested+2),
                   80*(ceil_log2(64*(b.gap_loss+1)/b.frequency)+requested),
                   8*(ceil_log2(b.vacuum)+2), ceil_log2(neutral_trial_upper(b, ue, k))+2,
                   16*(ceil_log2(b.current_lipschitz*b.soft_moment)+requested))
    require(log_floor > required, 'unchanged global Lambda pays stronger graph caps')
    d, G = F(1, 2**requested), F(2**requested)
    eps = d/G
    # E0/Lambda <= 2 at this SAME Lambda (negative scalar baseline may be dropped
    # only for this upper bound). <rho^4> <= 2*(CD+2)/a.
    rho4 = 2*(cd+2)/a
    M2 = sqrt_upper(rho4)  # ||rho^2 Omega||, not an operator norm.
    M1 = sqrt_upper(M2)
    CJ = sqrt_upper(F(12))*(2*v+1)*F(42, 100)
    omega = 4*c/n  # sin(pi/N)>=2/N, omega=c*sqrt(q^2).
    photon_number = ue*CJ*CJ*M1*M1/omega**3
    unorm = CJ*M1/(omega*sqrt_lower(omega))
    qunorm = CJ*M1/sqrt_lower(omega)
    rho2_U = sqrt_upper(2*(cd+3)/a)*unorm
    birth_norm = sqrt_upper(1+photon_number)
    birth_Q = c*sqrt_upper(F(12))*birth_norm+sqrt_upper(ue/omega)*CJ*M1
    rho2_birth = sqrt_upper(2/a*(birth_norm*birth_Q+(cd+2)*birth_norm*birth_norm))
    trial = neutral_trial_upper(b, ue, k)
    S = F(supplied['Lambda_independent_photon_form_shift'])
    hem_z2 = d*d/G+S*eps*eps
    birth_mismatch = sqrt_upper(eps*eps+hem_z2/omega)
    # Current source complement plus image cap. The graph residual is paid on
    # the existing source graph multiplier, with the actual nonzero d retained.
    graph = F(2**131072)
    half_graph = scattering['microscopic_sandwich']['actual_global_receiver_energy_half_upper']
    actual_half = F(int(half_graph['numerator']), int(half_graph['denominator']))
    require(64*actual_half*actual_half*2**2547 < graph, 'actual source graph and six jets covered')
    rho_z = F(supplied['C_rho'])*eps+F(supplied['C_Q'])*d/F(2**(requested+3*requested//4))
    current_mismatch = CJ*rho_z+d
    U_mismatch = (current_mismatch+d*graph)/(omega*sqrt_lower(omega))
    photon_cloud_mismatch = sqrt_upper(ue)*U_mismatch
    source_value = d*graph
    all_source_mismatch = 8*graph*(birth_mismatch+photon_cloud_mismatch+source_value)
    target = ue/(v*2**40)
    require(all_source_mismatch < target, 'actual weighted original source/photon cap below far target')
    # Three-dimensional compact even envelope; rho_kappa<=sqrt3*sigma.
    # Phase is charged/nonforward hard-photon selected BEFORE its phase bound.
    sigma = F(1, 2**262144)
    D = ue/v
    dispersion = 2*sqrt_upper(D*(S+3*D*sigma*sigma))*sqrt_upper(F(3))*sigma+3*D*sigma*sigma
    R = F(2**131072)  # supplied full-Q source word plus finite source jets.
    source_rho = 16*F(supplied['C_rho'])+F(supplied['C_Q'])*sqrt_upper(actual_half)
    hard_omega = c/2
    history_needed = 4096*v*(1+CJ*CJ)*(1+source_rho)**2/hard_omega
    require(history_needed < R, 'actual opposite-leg hard-current history phase majorant')
    phase_rescaled = 6*R*dispersion
    require(phase_rescaled < F(1, 2**80), 'hard UE/V selected paired-packet phase debt')
    return dict(N=n, V=v, UE=str(ue), K=str(k), selected_time=6,
        law_unchanged=True, original_scattering_eta=18, original_alpha='2^371',
        actual_global_Lambda_floor_log2=str(log_floor), required_log2_upper=str(required),
        stronger_cap_on_same_Lambda=dict(residual_upper=dyadic_upper(d),
            current_image_upper=dyadic_upper(d), complement_gap_lower=dyadic_upper(G),
            nonzero_residual=True),
        rho_fourth_expectation_upper=dyadic_upper(rho4),
        rho_squared_vector_norm_upper=dyadic_upper(M2),
        quartic_operator_norm_claimed=False,
        all_mode_photon_number_upper=dyadic_upper(photon_number),
        U_norm_upper=dyadic_upper(unorm), QU_norm_upper=dyadic_upper(qunorm),
        rho_squared_U_graph_upper=dyadic_upper(rho2_U),
        source_weighted_rho_squared_U_upper=dyadic_upper(16*rho2_U),
        actual_photon_birth_Q_first_graph_upper=dyadic_upper(birth_Q),
        source_weighted_rho_squared_birth_upper=dyadic_upper(16*rho2_birth),
        original_source_birth_mismatch_upper=dyadic_upper(birth_mismatch),
        original_source_cloud_mismatch_upper=dyadic_upper(photon_cloud_mismatch),
        weighted_original_source_transfer_upper=dyadic_upper(all_source_mismatch),
        requested_far_target=dyadic_upper(target), strictly_below_target=True,
        uniform_harmonic_kinetic_energy_upper=dyadic_upper(S),
        packet=dict(dimension=3, normalized=True, symmetric=True,
            envelope='product_j sqrt(15/(16 sigma))*(1-(kappa_j/sigma)^2)_+',
            sigma='2^-262144', finite_energy_upper=dyadic_upper(trial+3*D*sigma*sigma),
            not_global_eigen_ground=True, global_band_minimum_assumed=False,
            two_sided_dispersion_upper=dyadic_upper(dispersion),
            nonforward_rescaled_history_majorant=dyadic_upper(R),
            T6_phase_rescaled_error_upper=dyadic_upper(phase_rescaled),
            physical_phase_error='(UE/V)*T6_phase_rescaled_error_upper'),
        comparison_scope='original versus Gamma cloud on projected quasimode; no Gamma vacuum gap or closeness assertion',
        source_graph_domain='actual quartic form, balanced large-gauge Bloch boundary, coexact CCR smooth core',
        source_first_graph='K0*(||Q U||+s||U||)+||[H,B_t]U||; positive-half residual NOT used')


class FullFieldFibre:
    """Finite Cartesian normal slice, exact endpoint Phi, two quantum photons.

    Heavy top-four energy is the exact whole-heavy BO scalar on this slice;
    this native probe is NOT a full heavy/colour quantum discretization.
    """
    def __init__(self, frame: np.ndarray, kappa: float, occupation: int = 4,
                 external: np.ndarray | None = None):
        self.n = 3
        self.M = 2
        self.l = 5
        self.nodes = 3
        self.UE, self.K = .07, .8
        self.kappa = kappa
        self.band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
        base = Coupled(self.n, self.M, kappa, occupation, self.UE, self.K, frame)
        self.rows, self.d = base.rows, base.d
        self.size = self.nodes*self.n*self.n*self.l*self.d
        self.omegas = np.array([base.omega, .63])
        self.ell = base.ell
        self.coulomb = base.coulomb
        self.scalar_kinetic = .22*np.array([[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]])
        rng = np.random.default_rng(2221006)
        basis = matrix_unit_basis()
        deformation = np.einsum('xma,aij->xmij', rng.normal(size=(2, 4, 15)), basis)/15
        self.fields = np.array([np.tile(frame, (2, 1, 1, 1))+(x+.17)*deformation
                                for x in (-.4, 0., .4)])
        if kappa < 0:
            self.fields = -self.fields
        self.links = np.array([expm(1j*(.17+j*.07)*basis[j+3]) for j in range(3)])
        self.onsite, self.hops, potential = [], [], []
        for phi in self.fields:
            local, hops = endpoint_blocks(phi, self.links, self.band)
            self.onsite.append(exterior(local, self.rows))
            self.hops.append([exterior(h, self.rows) for h in hops])
            heavy_h = local+sum((h+h.conj().T for h in hops), np.zeros_like(local))
            heavy = -3*float(np.linalg.eigvalsh(heavy_h)[4:].sum())
            gram = np.einsum('xmij,xnji->xmn', phi, phi).real/4
            quartic = 3*float(np.sum(gram*gram))
            alignment = .2/8*sum(float(np.linalg.norm(phi[0]-u@phi[1]@u.conj().T)**2)
                                  for u in self.links)
            potential.append(quartic+alignment+heavy)
        self.potential = np.array(potential)
        self.rho2 = np.einsum('sxmij,sxmji->s', self.fields, self.fields).real/4
        from noncompact_native_current import displacement
        f = np.array([[1., -1., 0.], [1., 1., -2.]])
        f /= np.linalg.norm(f, axis=1)[:, None]
        self.phase = []
        self.parameters = f*np.sqrt(self.UE/self.omegas)[:, None]
        external = np.zeros(3) if external is None else external
        for edge, nu in enumerate((1, 1, -2)):
            shift = np.zeros((self.l, self.l), complex)
            for j in range(self.l):
                if 0 <= j+nu < self.l:
                    shift[j+nu, j] = 1
            z = np.kron(np.kron(displacement(self.n, self.parameters[0, edge]),
                                displacement(self.n, self.parameters[1, edge])), shift)
            self.phase.append(np.exp(1j*external[edge])*z)
        levels = np.indices((self.n, self.n, self.l))
        self.phenergy = (self.omegas[0]*(levels[0]+.5)+self.omegas[1]*(levels[1]+.5)
                         +self.UE*(self.ell[levels[2]]+kappa)**2/6).ravel()
        self.phdim = len(self.phenergy)
        self.operator = HermitianAction(self.size, self.action)

    def reshape(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v).reshape(self.nodes, self.phdim, self.d)

    def action(self, v: np.ndarray) -> np.ndarray:
        wave = self.reshape(v)
        out = np.asarray((self.potential[:, None, None]+self.phenergy[None, :, None]
                          +self.coulomb[None, None, :])*wave, dtype=complex)
        out += np.einsum('st,tpd->spd', self.scalar_kinetic, wave)
        for s in range(self.nodes):
            out[s] += wave[s]@self.onsite[s].T
            for z, t in zip(self.phase, self.hops[s]):
                out[s] += z@wave[s]@t.T+z.conj().T@wave[s]@t.conj()
        return out.ravel()

    def current(self, v: np.ndarray, weights: np.ndarray, contact: bool = False) -> np.ndarray:
        wave = self.reshape(v)
        out = np.zeros_like(wave)
        for s in range(self.nodes):
            for w, z, t in zip(weights, self.phase, self.hops[s]):
                if contact:
                    out[s] -= w*w*(z@wave[s]@t.T+z.conj().T@wave[s]@t.conj())
                else:
                    out[s] += 1j*w*(z@wave[s]@t.T-z.conj().T@wave[s]@t.conj())
        return out.ravel()

    def photon(self, v: np.ndarray, mode: int, create: bool = False) -> np.ndarray:
        w = np.asarray(v).reshape(self.nodes, self.n, self.n, self.l, self.d)
        out = np.zeros_like(w)
        for j in range(self.n-1):
            source: list[slice | int] = [slice(None)]*5
            target = source.copy()
            source[mode+1], target[mode+1] = (j, j+1) if create else (j+1, j)
            out[tuple(target)] += math.sqrt(j+1)*w[tuple(source)]
        return out.ravel()


def source_matrices(neutral: FullFieldFibre, charged: FullFieldFibre,
                    frame: np.ndarray) -> np.ndarray:
    """Actual finite product-Haar integrand, all quartet/all four references."""
    samples = haar_nodes(128, 2, 222611)
    samples[:4] = np.tile(np.eye(4), (4, 2, 1, 1))
    gamma5 = np.linalg.multi_dot(list(SELECTED))
    samples = np.concatenate((samples, samples@gamma5))
    labels = [np.eye(4)]+[SELECTED[0]@SELECTED[j] for j in range(1, 4)]
    alpha = np.array([1., .3j, -.2+.4j, .6])
    alpha /= np.linalg.norm(alpha)
    q = np.asarray(sum(z*m for z, m in zip(alpha, labels)))
    lookup = {row: j for j, row in enumerate(charged.rows)}
    creators = np.zeros((8, charged.d, neutral.d), complex)
    for j, row in enumerate(neutral.rows):
        for mode in range(8):
            if mode not in row:
                creators[mode, lookup[tuple(sorted((*row, mode)))], j] = (-1)**sum(t < mode for t in row)
    result = []
    for phi in neutral.fields:
        mark = orientation_cloud(phi, neutral.links[:1], q, 18., float(2**371),
                                 samples, np.tile(frame, (2, 1, 1, 1)))['mean']
        local_mark = orientation_cloud(phi[1:], np.empty((0, 4, 4)), q, 18., float(2**371),
            samples[:, 1:], frame[None])['mean']
        pair = []
        for output in range(2):
            seed = np.zeros((8, 4), complex)
            seed[:4] = .8*mark
            seed[4:] = .6*np.exp(output*.71j)*local_mark
            pair.append(np.einsum('ir,icd->rcd', seed, creators))
        result.append(pair)
    return np.asarray(result).swapaxes(0, 1)


def apply_source(neutral: FullFieldFibre, charged: FullFieldFibre,
                 matrices: np.ndarray, vector: np.ndarray, recoil: bool = False) -> np.ndarray:
    wave = neutral.reshape(vector)
    out = np.einsum('spd,srcd->rspc', wave, matrices[int(recoil)])
    return out.reshape(4, charged.size)


def fibre_case(frame: np.ndarray, kappa: float, detailed: bool,
               paired: np.ndarray | None = None) -> dict[str, Any]:
    neutral = FullFieldFibre(frame, kappa)
    charged = FullFieldFibre(frame, kappa, 5)
    if paired is None:
        rng = np.random.default_rng(222012)
        solver = cast(Callable[..., Any], eigsh)
        ev, vec = solver(neutral.operator, k=1, which='SA', tol=2e-10,
                         v0=rng.normal(size=neutral.size))
        E0, omega = float(ev[0]), vec[:, 0]
    else:
        from noncompact_neutral_supplier import particle_hole
        C = particle_hole(neutral.rows, 8)
        wave = paired.reshape(neutral.nodes, neutral.n, neutral.n, neutral.l, neutral.d)
        omega = (wave[:, :, :, ::-1].conj()@C.T).ravel()
        E0 = float(np.vdot(omega, neutral.action(omega)).real)
    residual = accept(np.linalg.norm(neutral.action(omega)-E0*omega), 'new full-field fibre residual', 2e-7)
    matrices = source_matrices(neutral, charged, frame)
    sources = [apply_source(neutral, charged, matrices, omega, i == 1) for i in range(2)]
    outputs = [np.array([charged.photon(w, i, True) for w in sources[i]])/math.sqrt(2) for i in range(2)]
    gram = np.array([[np.vdot(a, b) for b in outputs] for a in outputs])
    cloud = np.empty((2, 2), complex)
    bare = np.zeros((2, 2), complex)
    for i in range(2):
        bare[i, i] = np.vdot(sources[i], sources[i])/2
        for j in range(2):
            left = apply_source(neutral, charged, matrices, neutral.photon(omega, j), i == 1)
            right = apply_source(neutral, charged, matrices, neutral.photon(omega, i), j == 1)
            cloud[i, j] = np.vdot(left, right)/2
    # Exact finite CCR includes top-level number defect; pay it, do not erase it.
    ccr_defect = gram-bare-cloud
    predicted_ccr = np.zeros((2, 2), complex)
    for i in range(2):
        w = sources[i].reshape(4, charged.nodes, charged.n, charged.n, charged.l, charged.d)
        predicted_ccr[i, i] = -charged.n*np.linalg.norm(np.take(w, charged.n-1, axis=i+2))**2/2
    accept(np.linalg.norm(ccr_defect-predicted_ccr), 'crossed cloud and exact finite CCR boundary', 2e-9)
    require(float(np.linalg.norm(cloud)) > 1e-7, 'actual fullPhi photon cloud retained')
    eig, unit = np.linalg.eigh(gram)
    require(float(eig[0]) > 0, 'true global2x2 receiver Gram')
    inverse = (unit/np.sqrt(eig))@unit.conj().T
    receiver = inverse.T@np.stack([w.ravel() for w in outputs])
    accept(np.linalg.norm(receiver.conj()@receiver.T-np.eye(2)), 'GLOBAL2x2 output whitening', 2e-8)
    initial = outputs[0].T/np.linalg.norm(outputs[0])
    evolved = expm_multiply(-6j*charged.operator, initial, traceA=0.)
    norm = accept(abs(np.vdot(evolved, evolved).real-1), 'T6 full-reference actual field/photon source norm', 2e-7)
    overlaps = np.array([np.vdot(w.T, evolved) for w in outputs])
    amplitude = inverse@overlaps
    row: dict[str, Any] = dict(kappa=kappa, E0=E0, residual=residual,
        norm_T6_error=norm, original_two_output_Gram=encoded(gram),
        actual_crossed_cloud_Gram=encoded(cloud), finite_CCR_defect=encoded(ccr_defect),
        source_reference_count=4, output_count=2, global_whitened_amplitude_T6=encoded(amplitude),
        actual_rho_fourth_expectation=float(np.sum(neutral.rho2[:, None, None]**2*abs(neutral.reshape(omega))**2)),
        photon_number=float(sum(np.linalg.norm(neutral.photon(omega, j))**2 for j in range(2))))
    row['_section'] = omega
    row['_packet_raw'] = dict(gram=gram, cloud=cloud,
        overlaps=overlaps*np.linalg.norm(outputs[0]), input_norm_squared=float(gram[0, 0].real))
    if detailed:
        weights = np.array([.7, -.4, .2])
        delta = 2e-5
        plus = FullFieldFibre(frame, kappa, external=delta*weights)
        minus = FullFieldFibre(frame, kappa, external=-delta*weights)
        derivative = (plus.action(omega)-minus.action(omega))/(2*delta)
        second = (plus.action(omega)+minus.action(omega)-2*neutral.action(omega))/delta**2
        current_error = accept(np.linalg.norm(derivative-neutral.current(omega, weights)), 'nonlinear fullPhi Peierls current', 2e-8)
        contact_error = accept(np.linalg.norm(second-neutral.current(omega, weights, True)), 'nonlinear fullPhi contact', 3e-5)
        plain = Coupled(neutral.n, neutral.M, kappa, 4, neutral.UE, neutral.K, frame)
        wrongphi = np.linalg.norm(neutral.hops[0][0]-plain.hops[0])
        wrong_source = abs(np.vdot(outputs[0][0], outputs[0][0])-np.vdot(outputs[0], outputs[0]))
        commutator = np.linalg.norm(apply_source(neutral, charged, matrices, neutral.photon(omega, 0))-np.array([charged.photon(s, 0) for s in sources[0]]))
        accept(commutator, 'exact coexact B commutator', 2e-12)
        wrongcurrent = np.linalg.norm(derivative)
        uncrossed = np.vdot(
            apply_source(neutral, charged, matrices, neutral.photon(omega, 0)),
            apply_source(neutral, charged, matrices, neutral.photon(omega, 1), True))
        row['wrong_uncrossed_cloud'] = adverse(abs(cloud[0, 1]-uncrossed/2),
            'literal crossed cloud receiver', 1e-9)
        row['new_discriminants'] = dict(
            wrong_Phi=adverse(wrongphi, 'actual fullPhi endpoint', 1e-5),
            wrong_source=adverse(wrong_source, 'all4-reference source', 1e-5),
            wrong_domain=adverse(abs(np.exp(2j*math.pi*.31)-1), 'integer neutral Bloch deck source', 1e-10),
            missing_current=adverse(wrongcurrent, 'nonlinear fullPhi Peierls current', 2e-8),
            erased_cloud=adverse(np.linalg.norm(cloud), 'actual photon-cloud Gram', 1e-7))
        row['nonlinear_current_error'] = current_error
        row['nonlinear_contact_error'] = contact_error
        row['exact_coexact_source_commutator_error'] = float(commutator)
        row['nonClifford_defect'] = float(np.linalg.norm(neutral.fields[0, 0, 0]@neutral.fields[0, 0, 1]+neutral.fields[0, 0, 1]@neutral.fields[0, 0, 0]))
    return row


def packet_consumer() -> dict[str, Any]:
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    _, frame = original_frame(band, 5, np.zeros(3), True)
    nodes, quadrature = np.polynomial.legendre.leggauss(4)
    sigma = .08
    # Exactly integrates the polynomial density of the continuum envelope.
    density = 15/16*(1-nodes*nodes)**2
    weights = quadrature*density
    accept(abs(sum(weights)-1), 'normalized continuous paired Bloch envelope', 2e-14)
    positive = {j: fibre_case(frame, float(sigma*nodes[j]), j == 2) for j in (2, 3)}
    negative = {j: fibre_case(frame, float(sigma*nodes[j]), False,
                             positive[3-j]['_section']) for j in (0, 1)}
    cases = [negative[j] if j in negative else positive[j] for j in range(4)]
    central = fibre_case(frame, 0., False)
    energies = np.array([row['E0'] for row in cases])
    dispersion = float(np.max(abs(energies-central['E0'])))
    phase_norm = float(np.sqrt(np.sum(weights*abs(np.exp(-6j*(energies-central['E0']))-1)**2)))
    accept(max(0., phase_norm-6*dispersion), 'full normalized packet true phase evolution bound', 2e-12)
    paired_error = float(max(abs(cases[j]['E0']-cases[-1-j]['E0']) for j in range(2)))
    accept(paired_error, 'C-related plus/minus Bloch finite slice bands', 2e-7)
    for row in [*cases, central]:
        row.pop('_section')
    from scipy.integrate import quad
    D = .07/6
    spectator_phase = complex(
        quad(lambda u: 15/16*(1-u*u)**2*math.cos(6*D*sigma*sigma*u*u), -1, 1)[0],
        -quad(lambda u: 15/16*(1-u*u)**2*math.sin(6*D*sigma*sigma*u*u), -1, 1)[0])
    full_phase_overlap = np.sum(weights*np.exp(-6j*(energies-central['E0'])))*spectator_phase**2
    full_phase_norm = math.sqrt(max(0., 2-2*float(full_phase_overlap.real)))
    full_phase_bound = 6*(dispersion+2*D*sigma*sigma)
    accept(max(0., full_phase_norm-full_phase_bound), 'full R3 packet phase evolution', 2e-12)
    raw = [row.pop('_packet_raw') for row in cases]
    central.pop('_packet_raw')
    global_gram = np.asarray(sum(w*row['gram'] for w, row in zip(weights, raw)))
    global_cloud = np.asarray(sum(w*row['cloud'] for w, row in zip(weights, raw)))
    global_overlap = np.asarray(sum(w*row['overlaps'] for w, row in zip(weights, raw)))*spectator_phase**2
    values, vectors = np.linalg.eigh(global_gram)
    require(float(values[0]) > 0, 'full packet actual GLOBAL2x2 Gram')
    global_inverse = (vectors/np.sqrt(values))@vectors.conj().T
    global_input_norm = math.sqrt(float(global_gram[0, 0].real))
    global_amplitude = global_inverse@global_overlap/global_input_norm
    accept(np.linalg.norm(global_inverse@global_gram@global_inverse-np.eye(2)),
           'full packet GLOBAL2x2 whitening without per-fibre rescaling', 2e-9)
    require(float(abs(global_cloud[0, 1])) > 1e-7, 'full packet genuinely crossed two-column photon cloud')
    return dict(scope='new quantum Cartesian-slice/two-photon/Fourier finite approximation; exact full-domain derivations separate',
        scalar_nodes=3, neutral_dimension=3*3*3*5*70, charged_dimension=3*3*3*5*56,
        heavy_scope='exact whole top4 BO scalar per Phi node; full heavy/colour continuum not executed',
        fibres=cases, central_fibre=central, synthesized_packet=dict(
            envelope='sqrt(15/(16 sigma))*(1-(kappa/sigma)^2)_+', sigma=sigma,
            quadrature_nodes=(sigma*nodes).tolist(), quadrature_weights=weights.tolist(),
            continuum_norm_squared=1., packet_energy_quadrature=float(weights@energies)+2*D*sigma*sigma/7,
            spectator_harmonic_directions=2, spectator_envelopes='same normalized even density, free D*kappa_j^2 phases',
            T6_neutral_phase_norm=full_phase_norm, T6_bound=full_phase_bound,
            active_fibre_phase_norm=phase_norm, full_packet_GLOBAL2x2_Gram=encoded(global_gram),
            full_packet_crossed_cloud=encoded(global_cloud), actual_global_input_norm=global_input_norm,
            full_packet_amplitude_T6=encoded(global_amplitude), per_fibre_rescaling=False,
            finite_slice_plus_minus_band_difference=paired_error,
            original_domain_pairing='explicit C-related plus/minus Phi slices and Bloch bands; all fields retained inside each slice',
            finite_energy=True, global_eigen_ground_claimed=False,
            normalized_source_evolution=True, prepared_photon_cloud_erased=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--normal', required=True)
    parser.add_argument('--scattering', required=True)
    parser.add_argument('--input', action='append', default=[])
    args = parser.parse_args()
    sys.set_int_max_str_digits(0)
    normal = json.loads(Path(args.normal).read_text())
    scattering = json.loads(Path(args.scattering).read_text())
    before = source_hashes()
    inputs = {str(Path(name)): hashlib.sha256(Path(name).read_bytes()).hexdigest()
              for name in [args.normal, args.scattering, *args.input]}
    result: dict[str, Any] = dict(topic='Original fullPhi microscopic photon/source and symmetric noncompact packet',
                  plan=PLAN, plan_revision=PLAN_REVISION, selected=selected_certificate(normal, scattering),
                  observed=packet_consumer(), input_sha256=inputs, source_sha256=before)
    require(before == source_hashes(), 'owned source unchanged during natural execution')
    result['source_unchanged_during_run'] = True
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=args.output, selected=result['selected'],
                         observed=result['observed']), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
