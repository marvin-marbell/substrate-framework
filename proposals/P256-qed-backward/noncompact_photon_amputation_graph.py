#!/usr/bin/env python3
"""Mixed charged-source/neutral-photon resolvent graph, not a receiver copy."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.sparse.linalg import cg

from noncompact_native_current import Coupled, HermitianAction, accept, adverse
from local_orbital_stability import scalar_paths
from original_normal_gap_bound import peierls_derivatives
from noncompact_charged_packet_transport import (
    GREEN_3D_UPPER, graph, packet_seed, theorem_bounds, weak_family)
from original_charged_principal import original_frame
from rotation_band import RotationBand
from sea_selection import SELECTED

ROOT = Path(__file__).resolve().parent
PLAN_REVISION = 'ad6a94bfefcf9a2e3d10d60a51d344c6ac559f67a5dc41d46d7ae9f87be9f9ff'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def hashes() -> dict[str, str]:
    paths = {Path(__file__).resolve(), ROOT / 'noncompact_photon_amputation_graph.md'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().parent == ROOT:
            paths.add(Path(filename).resolve())
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def annihilate(model: Coupled, vector: np.ndarray) -> np.ndarray:
    wave = vector.reshape(model.n, model.l, model.d)
    out = np.zeros_like(wave)
    out[:-1] = np.sqrt(np.arange(1, model.n))[:, None, None] * wave[1:]
    return out.ravel()


def creators(neutral: Coupled, charged: Coupled) -> np.ndarray:
    result = np.zeros((8, charged.d, neutral.d), complex)
    lookup = {row: i for i, row in enumerate(charged.rows)}
    for column, row in enumerate(neutral.rows):
        for mode in range(8):
            if mode not in row:
                result[mode, lookup[tuple(sorted((*row, mode)))], column] = (-1)**sum(k < mode for k in row)
    return result


class HarmonicSource:
    """All eight onebody modes, all four references, genuine Fourier shifts."""
    def __init__(self, neutral: Coupled, charged: Coupled, time: float,
                 source_shift: int, modes: int = 24, samples: int = 128):
        self.neutral, self.charged = neutral, charged
        self.time, self.source_shift = time, source_shift
        alpha = np.array([1., .3j, -.2+.4j, .6]); alpha /= np.linalg.norm(alpha)
        labels = [np.eye(4)] + [SELECTED[0] @ SELECTED[j] for j in range(1, 4)]
        seed = np.zeros((8, 4), complex)
        seed[:4] = sum(alpha[j]*labels[j] for j in range(4))
        points = 2*math.pi*np.arange(samples)/samples
        marks, dots, sea = [], [], []
        self.onebody_hops = []
        for edge in range(3):
            hop = np.zeros((8, 8), complex)
            hop[:4, 4:] = -(.12*neutral.frame[0]+1j*.3*neutral.frame[edge+1])/2
            self.onebody_hops.append(hop)
        onsite = np.kron(np.eye(2), (.2+3*.12)*neutral.frame[0])
        for coordinate in points:
            h = onsite.copy()
            for hop, nu, external in zip(self.onebody_hops, (1, 1, -2), neutral.external):
                z = np.exp(1j*(nu*coordinate+external))
                h += z*hop + z.conjugate()*hop.conj().T
            values, vectors = np.linalg.eigh(h)
            # No positive-source filter. Negative components remain in the mark.
            mark = (vectors*np.exp(-1j*time*values)) @ vectors.conj().T @ seed
            mark *= np.exp(-1j*source_shift*coordinate)
            marks.append(mark); dots.append(h @ mark)
            sea.append(float(values[values < 0].sum()))
        fourier = np.fft.fft(np.asarray(marks), axis=0)/samples
        derivative = np.fft.fft(np.asarray(dots), axis=0)/samples
        self.orders = np.arange(-modes, modes+1)
        self.marks = fourier[self.orders % samples]
        self.idot_marks = derivative[self.orders % samples]
        creation = creators(neutral, charged)
        self.coefficients = np.einsum('qir,icd->qrcd', self.marks, creation)
        self.idot_coefficients = np.einsum('qir,icd->qrcd', self.idot_marks, creation)
        norm = np.linalg.norm(self.marks, axis=(1, 2))
        self.jet_l1 = [float(np.sum(abs(self.orders)**power*norm)) for power in range(3)]
        # A complex-strip Cauchy bound pays unsampled Fourier orders AND aliases.
        strip = 1.
        hop_bound = sum(2*float(np.linalg.norm(h, 2))*math.exp(abs(nu)*strip)
                        for h, nu in zip(self.onebody_hops, (1, 1, -2)))
        strip_h = float(np.linalg.norm(onsite, 2)) + hop_bound
        strip_m = float(np.linalg.norm(seed))*math.exp(abs(source_shift)*strip+abs(time)*strip_h)
        first, ratio = modes+1, math.exp(-strip)
        sums = (1/(1-ratio), first/(1-ratio)+ratio/(1-ratio)**2,
                first*first/(1-ratio)+2*first*ratio/(1-ratio)**2
                +ratio*(1+ratio)/(1-ratio)**3)
        self.tail_jets = [2*strip_m*ratio**first*value for value in sums]
        alias = 2*strip_m*math.exp(-strip*(samples-modes))/(1-math.exp(-strip*samples))
        self.alias_jets = [float(alias*np.sum(abs(self.orders)**power)) for power in range(3)]
        self.idot_tail = strip_h*self.tail_jets[0]
        self.idot_alias = strip_h*self.alias_jets[0]
        self.scalar_sea_range = [min(sea), max(sea)]
        self.scalar_sea_sha256 = hashlib.sha256(np.asarray(sea).tobytes()).hexdigest()

    def apply(self, vector: np.ndarray, idot: bool = False) -> np.ndarray:
        source = vector.reshape(self.neutral.n, self.neutral.l, self.neutral.d)
        out = np.zeros((4, self.charged.n, self.charged.l, self.charged.d), complex)
        coefficients = self.idot_coefficients if idot else self.coefficients
        for order, coefficient in zip(self.orders, coefficients):
            if order >= 0:
                left, right = slice(order, self.neutral.l), slice(0, self.neutral.l-order)
            else:
                left, right = slice(0, self.neutral.l+order), slice(-order, self.neutral.l)
            out[:, :, left] += np.einsum('nld,rcd->rnlc', source[:, right], coefficient, optimize=True)
        return out.reshape(4, self.charged.size)

    def residual(self, vector: np.ndarray) -> np.ndarray:
        created = self.apply(vector)
        action = np.array([self.charged.action(row) for row in created])
        return self.apply(vector, True)-action+self.apply(self.neutral.action(vector))

    def components(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        created = self.apply(vector).reshape(4, self.charged.phdim, self.charged.d)
        wave = self.neutral.reshape(vector)
        coulomb = -created*self.charged.coulomb + self.apply(
            (wave*self.neutral.coulomb).ravel()).reshape(created.shape)
        harmonic_energy = self.neutral.UE*(self.neutral.ell+self.neutral.zeta)**2/6
        kinetic = np.tile(harmonic_energy, self.neutral.n)
        harmonic = -created*kinetic[None, :, None] + self.apply(
            (wave*kinetic[:, None]).ravel()).reshape(created.shape)
        total = self.residual(vector).reshape(created.shape)
        return (total-coulomb-harmonic).reshape(4, -1), coulomb.reshape(4, -1), harmonic.reshape(4, -1)


def embedding_defect(model: Coupled, vector: np.ndarray, target: Coupled) -> dict[str, float]:
    """Whole-H embedding defect, including energy-weighted infinite phase tails."""
    embedded = model.embed(model.reshape(vector), target).ravel()
    visible = target.action(embedded)-model.embed(model.reshape(model.action(vector)), target).ravel()
    weights = 1+target.phenergy-target.omega/2
    weighted_visible = float(np.linalg.norm(np.sqrt(weights)[:, None]*target.reshape(visible)))
    tail, weighted_tail = 0., 0.
    wave = target.reshape(embedded)
    for edge, (phase, hop, nu) in enumerate(zip(target.phases, target.hops, (1, 1, -2))):
        parameter = (1., -1., 0.)[edge]/math.sqrt(2)*math.sqrt(model.UE/model.omega)
        for adjoint in (False, True):
            p = phase.conj().T if adjoint else phase
            t = hop.conj().T if adjoint else hop
            shift, s = (-nu, -parameter) if adjoint else (nu, parameter)
            f = wave @ t.T
            kept = p @ f
            total = float(np.vdot(f, f).real)
            lower = annihilate(target, f.ravel()).reshape(f.shape)
            kinetic = float(np.linalg.norm(lower+1j*s*f)**2)
            harmonic = np.tile((target.ell+shift+target.zeta)**2, target.n)
            total_weight = total+target.omega*kinetic+target.UE/6*float(np.sum(harmonic[:, None]*abs(f)**2))
            roundoff = 1e-13*max(total_weight, total)
            tail += math.sqrt(max(0., total-float(np.vdot(kept, kept).real))+roundoff)
            kept_weight = float(np.sum(weights[:, None]*abs(kept)**2))
            weighted_tail += math.sqrt(max(0., total_weight-kept_weight)+roundoff)
    return dict(visible_embedding_defect=float(np.linalg.norm(visible)),
                infinite_displacement_tail_upper=tail,
                full_infinite_embedding_defect_upper=float(np.linalg.norm(visible))+tail,
                energy_weighted_visible_defect=weighted_visible,
                energy_weighted_infinite_tail_upper=weighted_tail,
                energy_weighted_full_embedding_defect_upper=weighted_visible+weighted_tail)


def native_graph_certificate(size: int = 5, time: float = 6., omega_min: float = .08,
                             omega_max: float = .26, phase_envelope: str = 'conservative',
                             source_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Finite-N full-CAR bound; NO native coupled vacuum is numerically assumed."""
    if size < 5 or time < 0 or not 0 < omega_min <= omega_max:
        raise ValueError('finite native graph, time and nonzero photon band required')
    if source_data is None:
        band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
        selected, frame = original_frame(band, size, np.array([1, 2, 3])*math.pi/2, True)
        gram = selected['G']
        law, _, _ = graph(size, (1, 2, 3), frame, band)
        eta, preparation = packet_seed(law, 1.6, np.array([1.2, 0., 0.]))
        reference = law.index[tuple((np.asarray(preparation['origin'])-2) % size)]
        constants = weak_family(.01, .1, 1., phase_envelope)
    else:
        law, eta, reference = (source_data[key] for key in ('law', 'eta', 'reference'))
        constants, gram = source_data['constants'], source_data['gram']
        if law.size != size:
            raise ValueError('source data graph differs from requested native size')
    supplier = theorem_bounds(constants, law, eta, reference, time)
    ue, k, tau, b = (constants[key] for key in ('UE', 'K', 'endpoint_norm_upper', 'B_upper'))
    volume, number = size**3, 2*size**3
    qmin = 4*math.sin(math.pi/size)**2
    coulomb_max = ue*4*volume/qmin
    # Exact Esea(a) is retained by the scalar variational/Kato argument.
    # Its range is used only for a deliberately coarse finite energy domination.
    ph_zero = constants['bare_speed']*float(np.sum(np.sqrt(
        np.sum(abs(1-np.exp(2j*math.pi*law.sites/size))**2, axis=1))))
    neutral_shift = 2*number*b+coulomb_max+ue*3/(2*volume)+ph_zero
    cj = 192*tau*tau*volume
    l1 = supplier['packet_matrix_L1_time_upper']
    d1 = supplier['harmonic_source_first_HS_derivative_upper']
    d2 = supplier['harmonic_source_Laplacian_HS_upper']
    # Packet heat smoothing, unlike a scalar-frequency spectral shortcut.
    heat_constant = math.sqrt(math.pi/(2*math.e))
    def mixed(rhs: float) -> dict[str, float]:
        norm = rhs/omega_min**1.5
        energy = rhs*heat_constant/omega_min
        positive = math.hypot(energy, math.sqrt(neutral_shift)*norm)
        field = math.sqrt(2*GREEN_3D_UPPER/k)*positive
        light = 6*tau*l1*field
        electric = 2*math.sqrt(ue/volume)*d1*positive+ue/volume*d2*norm
        # Entire compensated Coulomb form: [HC,Bx]=UE Bx(2 qx.G rho+qx.G qx).
        force = 4*math.sqrt(4*volume*GREEN_3D_UPPER/qmin)+13/4
        coulomb = ue*force*l1*norm
        return dict(resolvent_packet_norm_upper=norm, relative_energy_half_norm_upper=energy,
                    positive_photon_harmonic_graph_upper=positive, coexact_field_source_moment_upper=field,
                    light_graph_term_upper=light, Coulomb_reference_term_upper=coulomb,
                    harmonic_source_electric_term_upper=electric,
                    normalized_RB_annihilation_cloud_upper=light+coulomb+electric)
    actual = mixed(math.sqrt(cj))
    # Constructive positive particle-hole current jet; no backward-energy exponential.
    mu, jet = constants['mu_lower'], 2*tau
    p1, p2 = jet/(2*mu), jet/(2*mu)+2*jet*jet/(mu*mu)
    j0, j1, j2 = 2*math.sqrt(6)*tau, 2*math.sqrt(6)*tau, 2*math.sqrt(6)*tau
    decay = omega_min+2*mu
    integrals = [math.factorial(power)/decay**(power+1) for power in range(3)]
    i0 = j0*integrals[0]
    i1 = (j1+4*p1*j0)*integrals[0]+2*jet*j0*integrals[1]
    i2 = (j2+8*p1*j1+(4*p2+32*p1*p1)*j0)*integrals[0]
    i2 += (4*jet*j1+(2*jet+16*p1*jet)*j0)*integrals[1]+4*jet*jet*j0*integrals[2]
    ground_positive = math.sqrt(neutral_shift)
    sigma = math.sqrt(2*GREEN_3D_UPPER/k)*ground_positive
    current_light = 4*number*tau*(3*volume)*sigma*i0
    current_coulomb = 2*coulomb_max*number*i0
    current_electric = (2*math.sqrt(ue/volume)*ground_positive*math.sqrt(3)*number*i1
                        +ue/volume*3*number*i2)
    leakage = math.sqrt(min(1., volume*supplier['neutral_excitation_density_upper']/2))
    initial_mismatch = (math.sqrt(cj)+number*j0)*leakage
    # J_f(A) does not become j_f(a) just because the CAR state is the full sea.
    # Pay its literal Peierls phase difference BEFORE the particle-hole comparison.
    initial_vertex = 8*tau*math.sqrt(3*volume)*sigma
    rhs_error = initial_vertex+initial_mismatch+current_light+current_coulomb+current_electric
    comparison = mixed(rhs_error)
    return dict(N=size, time=time, omega_min=omega_min, omega_max=omega_max, UE=ue, K=k,
                native_true_Gram=np.asarray(gram).tolist(), actual_full_CAR_number=number,
                all_state_current_squared_upper=cj, retained_scalar_Esea=True,
                full_positive_energy_domination_shift=neutral_shift,
                photon_zero_energy_upper=ph_zero, entire_Coulomb_operator_upper=coulomb_max,
                source_supplier=supplier, mixed_source_graph=actual,
                conditional_positive_particle_hole_resolvent=dict(
                    pair_gap=2*mu, initial_full_sea_mismatch_upper=initial_mismatch,
                    initial_coexact_current_vertex_replacement_upper=initial_vertex,
                    integrated_current_light_residual_upper=current_light,
                    integrated_current_Coulomb_residual_upper=current_coulomb,
                    integrated_current_harmonic_electric_residual_upper=current_electric,
                    total_resolvent_rhs_error_upper=rhs_error,
                    mixed_source_error_graph=comparison,
                    coexact_conditional_sea_current='zero by native translation/nonzero momentum; scalar Esea itself NOT flattened'),
                finite_time_amputation_source_error_upper=time*actual['normalized_RB_annihilation_cloud_upper'],
                conditional_boundary_coefficient_error_upper=rhs_error/omega_min**1.5,
                preparation_boundary_is_leading=True, physical_two_leg_factor=ue,
                native_numeric_coupled_ground=False, infinite_volume_uniform=False,
                legal_Lambda_scope='fixed-N existence, weighted component/source/normal majorants; no numerical threshold')


def quartic_current_majorant(volume: int, strong: float, spacing: float,
                            neutral_energy_upper: float, state_norm: float,
                            relative_energy_upper: float) -> dict[str, float]:
    """Original slow law: a state form bound, never a fixed-Gamma operator norm."""
    if volume <= 0 or strong < 1 or spacing <= 0 or state_norm < 0 or relative_energy_upper < 0:
        raise ValueError('positive original-law graph/coefficients and actual unnormalized moments')
    band = RotationBand(a=spacing, m=.2, r=.12, v=.3, improved=False)
    onsite, paths = scalar_paths(band)
    coefficient_norms = [float(np.linalg.norm(coefficients)) for _, coefficients in paths]
    incidence = 2*len(paths)
    endpoint = max(coefficient_norms)
    local_matter = 8*abs(onsite)+16*sum(coefficient_norms)
    kappa = 2/spacing
    linear = (strong+1)*local_matter
    young_per_site = .75*linear*(4*linear/(strong*kappa))**(1/3)
    debt = volume*young_per_site
    shift = neutral_energy_upper+debt
    if shift < 0:
        raise ValueError('neutral upper contradicts the original quartic lower form')
    energy = relative_energy_upper+shift*state_norm**2
    rho_fourth = 16*energy/(strong*kappa)
    current_squared = 128*endpoint**2*incidence*math.sqrt(volume)*state_norm*math.sqrt(rho_fourth)
    return dict(quartic_rho_fourth_coefficient=strong*kappa/16,
                original_full_CAR_linear_coefficient=local_matter,
                Young_lower_form_debt=debt, positive_energy_shift=shift,
                scalar_rho_fourth_moment_upper=rho_fourth,
                original_Phi_momentum_squared_upper=2*math.sqrt(strong)*energy,
                photon_harmonic_Coulomb_form_upper=energy,
                unnormalized_current_squared_upper=current_squared,
                primitive_path_incidence=incidence, primitive_path_coefficient_norm=endpoint)


def quartic_current_consumer(size: int, frame: np.ndarray) -> dict[str, Any]:
    """Pointwise full-field inequality, with whole-CAR Slater witnesses."""
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    law, _, _ = graph(size, (0, 0, 0), frame, band)
    rng = np.random.default_rng(2220610)
    field = rng.normal(size=(law.volume, 4, 4, 4))+1j*rng.normal(size=(law.volume, 4, 4, 4))
    field = (field+field.conj().swapaxes(-1, -2))/2
    field -= np.trace(field, axis1=-2, axis2=-1)[..., None, None]*np.eye(4)/4
    diagonal = rng.uniform(-1., 1., size=(law.volume, 3, 3))
    diagonal = np.concatenate((diagonal, -diagonal.sum(axis=2, keepdims=True)), axis=2)
    links = np.eye(4)*np.exp(1j*diagonal[..., None, :])
    phase = rng.uniform(-math.pi, math.pi, size=(law.volume, 3))
    direction = rng.normal(size=(law.volume, 3)); direction /= np.linalg.norm(direction)
    reference_bound = quartic_current_majorant(law.volume, 16., 1., 0., 1., 0.)
    endpoint, incidence = (reference_bound[key] for key in ('primitive_path_coefficient_norm', 'primitive_path_incidence'))
    tau = max((.12*np.linalg.norm(frame[0], 'fro')+.3*np.linalg.norm(frame[j], 'fro'))/2 for j in range(1, 4))
    fixed_body = 192*tau*tau*law.volume
    rows = []
    for scale in (1., 1000.):
        fields = scale*field
        h = law.onebody(fields, links, phase)
        heavy = law.onebody(fields, links, np.zeros_like(phase))
        values, vectors = np.linalg.eigh(h)
        occupied = vectors[:, :2*law.volume]
        current, _ = peierls_derivatives(law, fields, links, phase, direction)
        mapped = current@occupied
        mean = np.trace(occupied.conj().T@mapped)
        fluctuation = mapped-occupied@(occupied.conj().T@mapped)
        actual_squared = float(abs(mean)**2+np.linalg.norm(fluctuation)**2)
        gram = np.einsum('xmij,xnji->xmn', fields, fields).real/4
        rho_squared = np.trace(gram, axis1=1, axis2=2)
        rho_fourth = float(np.sum(rho_squared**2))
        upper = 128*endpoint**2*incidence*math.sqrt(law.volume*rho_fourth)
        accept(max(0., actual_squared-upper), 'unbounded full-field current quartic form bound')
        potential = 16*float(np.sum(gram**2))-16*float(np.linalg.eigvalsh(heavy)[-2*law.volume:].sum())+float(values[:2*law.volume].sum())
        lower = 2*rho_fourth-reference_bound['Young_lower_form_debt']
        accept(max(0., lower-potential), 'actual quartic plus whole-heavy/light Young coercivity')
        row: dict[str, Any] = dict(field_scale=scale, scalar_rho_fourth=rho_fourth,
                   actual_whole_CAR_current_norm_squared=actual_squared,
                   quartic_current_squared_upper=upper, original_scalar_heavy_light_potential=potential,
                   Young_potential_lower=lower)
        if scale == 1000.:
            row['wrong_fixed_Gamma_bound'] = adverse(max(0., actual_squared-fixed_body),
                'unbounded full-field current quartic form bound')
        rows.append(row)
    first = quartic_current_majorant(law.volume, 16., 1., 0., 1., 3.)
    scaled = quartic_current_majorant(law.volume, 16., 1., 0., .01, .0003)
    scaling = accept(abs(scaled['unnormalized_current_squared_upper']/
                         first['unnormalized_current_squared_upper']-.0001),
                     'unnormalized resolvent current retains cloud norm factor')
    return dict(N=size, strong=16., unrestricted_real_Phi_coordinates=60*law.volume,
                whole_heavy_number=2*law.volume, whole_light_number=2*law.volume,
                genuine_unitary_colour_links=True, no_field_or_CAR_projection=True,
                rows=rows, unnormalized_cloud_scaling_error=scaling,
                scope='pointwise full-field/whole-CAR graph-form positive control, not a sampled microscopic ground')


def coupled_case(n: int, M: int, args: argparse.Namespace, frame: np.ndarray) -> dict[str, Any]:
    neutral = Coupled(n, M, args.zeta, 4, args.UE, args.K, frame)
    charged = Coupled(n, M, args.zeta, 5, args.UE, args.K, frame)
    values, vectors, residuals = neutral.solve(2)
    e0, psi = float(values[0]), vectors[:, 0]
    if values[1]-e0 < 1e-7:
        raise AssertionError('complete neutral ground projector required')
    f = np.array([1., -1., 0.])/math.sqrt(2)
    current = neutral.current(psi, f)
    s = math.sqrt(args.UE/neutral.omega)
    operator = HermitianAction(neutral.size, lambda v: neutral.action(v)+(neutral.omega-e0)*v)
    solver = cast(Callable[..., Any], cg)
    u, status = solver(operator, current, rtol=2e-12, atol=0., maxiter=6000)
    if status != 0:
        raise AssertionError(f'actual whole-H current resolvent status {status}')
    u = np.asarray(u, dtype=complex)
    solve_error = accept(np.linalg.norm(neutral.action(u)+(neutral.omega-e0)*u-current),
                         'whole-H mixed current resolvent', 2e-8)
    apsi = annihilate(neutral, psi)
    pull_defect = neutral.action(apsi)+(neutral.omega-e0)*apsi+s*current
    cloud_error = float(np.linalg.norm(apsi+s*u))
    accept(max(0., cloud_error-np.linalg.norm(pull_defect)/neutral.omega), 'paid cutoff pull-through')
    qenergy = float(np.vdot(u, neutral.action(u)-e0*u).real)
    cj = float(np.vdot(current, current).real)
    accept(max(0., qenergy-cj/(4*neutral.omega)), 'actual mixed resolvent energy moment')
    rows = []
    for time in args.times:
        source = HarmonicSource(neutral, charged, time, args.source_shift)
        ru, ra = source.residual(u), source.residual(apsi)
        rpsi = source.residual(psi)
        ar = np.array([annihilate(charged, row) for row in rpsi])
        commutator = np.array([charged.current(row, f) for row in source.apply(psi)])-source.apply(current)
        def defect(model: Coupled, v: np.ndarray) -> np.ndarray:
            return model.action(annihilate(model, v))-annihilate(model, model.action(v))+model.omega*annihilate(model, v)+s*model.current(v, f)
        dcomm = np.array([defect(charged, row) for row in source.apply(psi)])-source.apply(defect(neutral, psi))
        identity = accept(np.linalg.norm(ar-ra+s*commutator-dcomm), 'both source graph and local excess plus measured cutoff', 2e-8)
        light, coulomb, harmonic = source.components(u)
        decomposition = accept(np.linalg.norm(ru-light-coulomb-harmonic), 'entire source electric/reference/light decomposition')
        ground_only = adverse(np.linalg.norm(ar+s*commutator-dcomm), 'both source graph and local excess plus measured cutoff', 2e-8)
        dropped_electric = adverse(np.linalg.norm(coulomb+harmonic), 'entire source electric/reference/light decomposition', 2e-8)
        missing_reference = adverse(np.linalg.norm(ra[1:]), 'complete four-reference source graph', 2e-8)
        frozen = HarmonicSource(neutral, charged, time, 0)
        wrong_harmonic = adverse(np.linalg.norm(ru-frozen.residual(u)), 'actual source/reference harmonic dressing', 2e-8)
        moment_p = float(np.linalg.norm((neutral.reshape(u)*np.tile(neutral.ell+neutral.zeta, n)[:, None])))
        hbound = args.UE/6*(2*source.jet_l1[1]*moment_p+source.jet_l1[2]*np.linalg.norm(u))
        cbound = 5*args.UE/3*source.jet_l1[0]*float(np.linalg.norm(u))
        tail0, tail1, tail2 = np.asarray(source.tail_jets)+np.asarray(source.alias_jets)
        onebody_upper = float(np.linalg.norm((.2+3*.12)*frame[0], 2))+sum(
            2*float(np.linalg.norm(hop, 2)) for hop in source.onebody_hops)
        fourier_graph_tail = ((source.idot_tail+source.idot_alias+9*onebody_upper*tail0+5*args.UE/3*tail0)*np.linalg.norm(u)
                              +args.UE/6*(2*tail1*moment_p+tail2*np.linalg.norm(u)))
        accept(max(0., np.linalg.norm(harmonic)-hbound), 'source derivative harmonic mixed moment bound')
        accept(max(0., np.linalg.norm(coulomb)-cbound), 'full reference Coulomb mixed moment bound')
        target_n = Coupled(n+6, M+26, args.zeta, 4, args.UE, args.K, frame)
        target_c = Coupled(n+6, M+26, args.zeta, 5, args.UE, args.K, frame)
        target_source = HarmonicSource(target_n, target_c, time, args.source_shift)
        lifted_u = neutral.embed(neutral.reshape(u), target_n).ravel()
        lifted_ru = np.array([charged.embed(charged.reshape(row), target_c).ravel() for row in ru])
        rb_embedding = float(np.linalg.norm(target_source.residual(lifted_u)-lifted_ru))
        rows.append(dict(time=time, actual_RB_aOmega_norm=float(np.linalg.norm(ra)),
                         current_resolvent_RB_norm=float(np.linalg.norm(ru)),
                         source_ground_residual_norm=float(np.linalg.norm(rpsi)),
                         source_resolvent_Fourier_graph_tail_upper=float(fourier_graph_tail),
                         normalized_RB_aOmega_norm=float(np.linalg.norm(ra))/math.sqrt(args.UE),
                         normalized_current_resolvent_graph_norm=float(np.linalg.norm(ru))/math.sqrt(neutral.omega),
                         resolvent_cutoff_RB_difference=float(np.linalg.norm(ra+s*ru)),
                         source_commutator_identity_error=identity, source_decomposition_error=decomposition,
                         light_source_graph_norm=float(np.linalg.norm(light)),
                         Coulomb_reference_graph_norm=float(np.linalg.norm(coulomb)),
                         harmonic_electric_source_graph_norm=float(np.linalg.norm(harmonic)),
                         harmonic_momentum_current_mixed_norm=moment_p,
                         harmonic_derivative_graph_upper=float(hbound), Coulomb_graph_upper=cbound,
                         source_fourier_tail_HS_jets_upper=source.tail_jets,
                         source_fourier_alias_HS_jets_upper=source.alias_jets,
                         source_idot_fourier_tail_upper=source.idot_tail,
                         source_scalar_Esea_range=source.scalar_sea_range,
                         source_scalar_Esea_sha256=source.scalar_sea_sha256,
                         enlarged_RB_current_source_embedding_difference=rb_embedding,
                         current_resolvent_embedding=embedding_defect(neutral, u, target_n),
                         charged_source_embeddings=[embedding_defect(charged, row, target_c) for row in source.apply(u)],
                         adverse=dict(ground_current_only=ground_only, dropped_source_electric=dropped_electric,
                                      missing_reference=missing_reference, dropped_harmonic_source=wrong_harmonic)))
    return dict(n=n, M=M, zeta=args.zeta, neutral_dimension=neutral.size, charged_dimension=charged.size,
                E0=e0, actual_neutral_gap=float(values[1]-e0), ground_residuals=residuals,
                neutral_state_sha256=hashlib.sha256(psi.tobytes()).hexdigest(),
                omega=neutral.omega, actual_current_norm_squared=cj, resolvent_solve_error=solve_error,
                current_resolvent_norm=float(np.linalg.norm(u)), current_resolvent_relative_energy=qenergy,
                annihilation_cloud_norm=float(np.linalg.norm(apsi)), paid_pullthrough_defect=float(np.linalg.norm(pull_defect)),
                annihilation_resolvent_difference=cloud_error,
                neutral_ground_infinite_embedding=neutral.full_residual_bound(psi, e0), rows=rows,
                scope='actual coupled two-vertex fullCAR/allref/wholeH selected harmonic fibre; not numeric native3D or full-real ground')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', nargs='+', default=['3:36', '5:44'])
    parser.add_argument('--UE', type=float, default=.0001)
    parser.add_argument('--K', type=float, default=28.361421309064075)
    parser.add_argument('--zeta', type=float, default=0.)
    parser.add_argument('--times', nargs='+', type=float, default=[0., 1.])
    parser.add_argument('--source-shift', type=int, default=1)
    parser.add_argument('--native-size', type=int, default=5)
    parser.add_argument('--observation-time', type=float, default=6.)
    parser.add_argument('--capability-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.UE <= 0 or args.K <= 0 or args.source_shift == 0:
        raise ValueError('positive coupled coefficients and actual distinct source/reference required')
    before = hashes()
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    _, frame = original_frame(band, args.native_size, np.zeros(3), True)
    cases = []
    if not args.capability_only:
        for cutoff in args.cutoffs:
            n, harmonic = (int(value) for value in cutoff.split(':'))
            cases.append(coupled_case(n, harmonic, args, frame))
    result = dict(plan='issue-222-synthesis-continuation-20261006', assigned_revision=PLAN_REVISION,
                  template_revision=TEMPLATE_REVISION, source_hashes=before,
                  environment=dict(python=sys.version, scipy=scipy.__version__, numpy=np.__version__, platform=platform.platform(), argv=sys.argv),
                  coupled_cases=cases, native_graph_capability=native_graph_certificate(args.native_size, args.observation_time),
                  full_field_quartic_current_consumer=quartic_current_consumer(args.native_size, frame),
                  campaign_complete=False, two_photon_coefficient_claimed=False)
    if hashes() != before:
        raise AssertionError('source/dependency bytes changed during consumer execution')
    result['source_bindings_unchanged'] = True
    rendered = json.dumps(result, indent=2, allow_nan=False)
    args.output.write_text(rendered+'\n')
    print(rendered)


if __name__ == '__main__':
    main()
