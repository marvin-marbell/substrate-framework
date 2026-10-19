#!/usr/bin/env python3
"""Original full-state coincident-reference charge upper; no periodic ground computed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import original_frame, portable
from original_slow_kinetic_bo import source_hashes as bo_hashes
from original_uniform_charge_gap import (car_action, exterior_hamiltonian,
                                         residual, require, sector_basis)
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE

Matrix = NDArray[np.complex128]
ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'fdd8d586e9506148da75d9a41626f3bbadec4a4c88133582afc041e903bbe8a6'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def source_hashes() -> dict[str, str]:
    return {**bo_hashes(), **{name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('original_local_charge_upper.py', 'original_local_charge_upper.md',
                     'pair_full_field_charge_transfer.py', 'pair_full_field_charge_transfer.md',
                     'pair_separated_charge_transfer.md')}}


def field_constants(band: RotationBand) -> dict[str, float]:
    c = max((band.m+3*band.r/band.a)**2+3*band.r**2/(2*band.a**2),
            band.v**2/(2*band.a**2))
    kappa = 2/band.a
    moment = (32*np.sqrt(c)/kappa)**(2/3)
    return dict(kappa=kappa, endpoint_trace_constant=c,
                exact_ground_Q2_upper_if_E0_nonpositive=float(moment),
                compensated_charge_upper_if_E0_nonpositive=float(np.sqrt(c*moment)))


def insertion(site: int, charge: int) -> tuple[Matrix, list[int], list[int]]:
    """Full neutral exterior sector -> full target sector times actual reference4."""
    require(site in (0, 1) and charge in (-1, 1), 'actual tree reference site and charge')
    neutral = sector_basis(8, 4)
    target = sector_basis(8, 4+charge)
    lookup = {state: i for i, state in enumerate(target)}
    result = np.zeros((4*len(target), len(neutral)), dtype=np.complex128)
    for column, state in enumerate(neutral):
        for alpha in range(4):
            acted = car_action(state, 4*site+alpha, charge == 1)
            if acted is not None:
                final, sign = acted
                result[4*lookup[final]+alpha, column] = sign
    return result, neutral, target


def occupations(basis: list[int], site: int) -> NDArray[np.float64]:
    return np.asarray([sum((state>>(4*site+a)) & 1 for a in range(4))
                       for state in basis], dtype=np.float64)


def tree_reference_controls(h: Matrix, generators: Matrix) -> dict[str, Any]:
    """Exact one-edge Gauss elimination; no independent compact rotor cutoff."""
    h0, neutral = exterior_hamiltonian(h, 4)
    left0 = occupations(neutral, 0)-2
    rows = []
    for site in (0, 1):
        for charge in (-1, 1):
            b, basis0, target = insertion(site, charge)
            target_h, _ = exterior_hamiltonian(h, 4+charge)
            hq = np.kron(target_h, np.eye(4, dtype=np.complex128))
            n0 = occupations(basis0, site)
            target_n = np.repeat(occupations(target, site), 4)
            norm = float(np.linalg.norm(b.conj().T@b-np.diag(n0 if charge == -1 else 4-n0)))
            residual(norm, 'full reference creator has actual occupation norm')
            em = float(np.linalg.norm((target_n-charge)[:, None]*b-b*n0[None, :]))
            residual(em, 'reference insertion preserves local U1 source')
            wrong_em = float(np.linalg.norm(target_n[:, None]*b-b*n0[None, :]))
            colour = 0.
            wrong_colour = 0.
            for axis in range(len(generators)):
                t = generators[axis]
                local = np.zeros((8, 8), dtype=np.complex128)
                local[4*site:4*site+4, 4*site:4*site+4] = t
                q0, _ = exterior_hamiltonian(local, 4)
                qt, _ = exterior_hamiltonian(local, 4+charge)
                reference = t if charge == -1 else -t.T
                total = np.kron(qt, np.eye(4))+np.kron(np.eye(len(target)), reference)
                colour = max(colour, float(np.linalg.norm(total@b-b@q0)))
                wrong_reference = -t.T if charge == -1 else t
                wrong_total = np.kron(qt, np.eye(4))+np.kron(np.eye(len(target)), wrong_reference)
                wrong_colour = max(wrong_colour, float(np.linalg.norm(wrong_total@b-b@q0)))
            residual(colour, 'all15 whole-CAR colour and actual opposite reference generators')
            projector = np.zeros((8, 8), dtype=np.complex128)
            projector[4*site:4*site+4, 4*site:4*site+4] = np.eye(4)
            # The row/column cost is non-Hermitian; keep the owner's Hermitian contract.
            anticommutator, _ = exterior_hamiltonian(projector@h+h@projector, 4)
            commutator = (n0[:, None]-n0[None, :])*h0
            expected = -(anticommutator+(commutator if charge == -1 else -commutator))/2
            if charge == 1:
                expected += np.trace(projector@h)*np.eye(len(neutral))
            cost_identity = float(np.linalg.norm(b.conj().T@(hq@b-b@h0)-expected))
            residual(cost_identity/max(1., np.linalg.norm(expected)), 'actual CAR row-column charge cost')
            flux_target = np.repeat(occupations(target, 0)-2-charge*int(site == 0), 4)
            flux = float(np.linalg.norm(flux_target[:, None]*b-b*left0[None, :]))
            residual(flux, 'entire tree Gauss integer flux unchanged by local reference insertion')
            electric_rows = []
            for electric in (.001, .1, 1000.):
                hn = h0+np.diag(electric*left0**2)
                hc = hq+np.diag(electric*flux_target**2)
                commutator = b.conj().T@(hc@b-b@hn)
                error = float(np.linalg.norm(commutator-expected)/max(1., np.linalg.norm(expected)))
                residual(error, 'full quantum Gauss Hamiltonian local cost with nonzero electric')
                energies0 = np.linalg.eigvalsh(hn)
                charged = np.linalg.eigvalsh(hc)[0]
                electric_rows.append(dict(electric=electric, full_neutral_dimension=len(neutral),
                    full_charged_reference_dimension=len(hc), ground_gap=float(charged-energies0[0]),
                    common_kinetic_commutator_error=error))
            require(wrong_em > 1 and wrong_colour > 1,
                    'omitted charge offset and wrong colour representation are genuine witnesses')
            for wrong, label in ((wrong_em, 'reference insertion preserves local U1 source'),
                                 (wrong_colour, 'all15 whole-CAR colour and actual opposite reference generators')):
                try:
                    residual(wrong, label)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('same Gauss accept boundary must reject wrong reference')
            rows.append(dict(reference_site=site, charge=charge, norm_identity_error=norm,
                U1_source_error=em, all15_colour_error=colour, CAR_cost_identity_error=cost_identity,
                exact_flux_error=flux, omitted_reference_U1_error=wrong_em,
                wrong_colour_reference_error=wrong_colour, quantum_electric_rows=electric_rows))
    return dict(all_original_8_mode_sectors=True, tree_rotor_eliminated_by_exact_Gauss=True,
                no_rotor_cutoff=True, reference_rows=rows)


def _separated_path_controls(links: Matrix, phases: NDArray[np.float64],
                             generators: Matrix, vertices: list[int]) -> dict[str, Any]:
    """Native two-edge path; entire endpoint CAR, full spectator identities."""
    first, second = links
    path = first@second
    angle = float(np.sum(phases))
    rows = []
    for charge in (-1, 1):
        bare, neutral, target = insertion(0, charge)
        def dress(colour: Matrix, photon_angle: float) -> Matrix:
            rotation = colour.conj().T if charge == -1 else colour.T
            return np.exp(1j*charge*photon_angle)*np.einsum(
                'ba,sac->sbc', rotation, bare.reshape(len(target), 4, len(neutral))).reshape(bare.shape)
        b = dress(path, angle)
        n0 = occupations(neutral, 0)
        norm_error = float(np.linalg.norm(b.conj().T@b-np.diag(n0 if charge == -1 else 4-n0)))
        residual(norm_error, 'remote path retains the actual endpoint occupation norm')
        colour_error = 0.
        omitted_colour_error = 0.
        internal_error = 0.
        for t in generators/np.sqrt(8):
            left = np.zeros((8, 8), dtype=np.complex128)
            right = np.zeros_like(left)
            left[:4, :4] = t
            right[4:, 4:] = t
            ql0, _ = exterior_hamiltonian(left, 4)
            qr0, _ = exterior_hamiltonian(right, 4)
            qlt, _ = exterior_hamiltonian(left, 4+charge)
            qrt, _ = exterior_hamiltonian(right, 4+charge)
            ql = np.kron(qlt, np.eye(4))
            reference = t if charge == -1 else -t.T
            qr = np.kron(qrt, np.eye(4))+np.kron(np.eye(len(target)), reference)
            # D=-i derivative along gL W or W gR^dagger; do not drop J_Phi.
            if charge == -1:
                dl_rotation, dr_rotation = -path.conj().T@t, t@path.conj().T
            else:
                dl_rotation, dr_rotation = (t@path).T, -(path@t).T
            def derivative(rotation: Matrix) -> Matrix:
                return np.exp(1j*charge*angle)*np.einsum(
                    'ba,sac->sbc', rotation, bare.reshape(len(target), 4, len(neutral))).reshape(bare.shape)
            colour_error = max(colour_error,
                float(np.linalg.norm(ql@b-b@ql0-derivative(dl_rotation))),
                float(np.linalg.norm(qr@b-b@qr0-derivative(dr_rotation))))
            # A genuine intermediate-site gauge move acts on BOTH native links.
            middle = expm(.23j*t)
            moved = (first@middle.conj().T)@(middle@second)
            internal_error = max(internal_error, float(np.linalg.norm(dress(moved, angle)-b)))
            wrong = dress(np.eye(4, dtype=np.complex128), angle)
            omitted_colour_error = max(omitted_colour_error,
                float(np.linalg.norm(ql@wrong-wrong@ql0)),
                float(np.linalg.norm(qr@wrong-wrong@qr0)))
        residual(colour_error, 'remote all30 colour Gauss endpoint intertwiners')
        residual(internal_error, 'remote intermediate-site colour Gauss cancellation')
        em_error = 0.
        omitted_em_error = 0.
        h = 1e-5
        for site, orientation in ((0, 1), (1, -1)):
            nt = np.repeat(occupations(target, site)-2-charge*int(site == 1), 4)
            nn = occupations(neutral, site)-2
            matter = nt[:, None]*b-b*nn[None, :]
            numerical = -1j*(dress(path, angle+orientation*h)-dress(path, angle-orientation*h))/(2*h)
            em_error = max(em_error, float(np.linalg.norm(numerical-matter)))
            omitted_em_error = max(omitted_em_error, float(np.linalg.norm(matter)))
        middle_phases_plus = phases+np.array([-h, h])
        middle_phases_minus = phases-np.array([-h, h])
        middle_derivative = -1j*(dress(path, float(np.sum(middle_phases_plus)))
            -dress(path, float(np.sum(middle_phases_minus))))/(2*h)
        em_error = max(em_error, float(np.linalg.norm(middle_derivative)))
        residual(em_error, 'remote allthree U1 Gauss source derivatives')
        for error, label in ((omitted_colour_error, 'remote all30 colour Gauss endpoint intertwiners'),
                             (omitted_em_error, 'remote allthree U1 Gauss source derivatives')):
            try:
                residual(error, label)
            except AssertionError:
                pass
            else:
                raise AssertionError('omitted remote dressing must fail the same physical Gauss assertion')
        rows.append(dict(charge=charge, full_neutral_dimension=len(neutral),
            full_charged_reference_dimension=len(b), occupation_norm_error=norm_error,
            all30_colour_error=colour_error, intermediate_colour_error=internal_error,
            allthree_U1_error=em_error, omitted_colour_error=omitted_colour_error,
            omitted_EM_phase_error=omitted_em_error, wrong_dressings_rejected=True))
    return dict(native_vertices=vertices, native_links=links, native_phases=phases,
        integer_EM_path=[1, 1], endpoint_CAR_modes=8, full_endpoint_NL=4,
        heavy_and_intermediate_CAR='entire identity spectator factors, not projected',
        Cartesian_angular_terms='commute with multiplication; remain in Gauss and kinetic energy',
        compact_rotor_cutoff=False, periodic_quantum_density_sampled=False,
        photon_marginal_equality='analytic for any original C-invariant neutral density; not a numerical ground claim',
        separated_uniform_energy_bound_proved=False, rows=rows)

def _coherent_profile_controls(paths: Matrix, angles: NDArray[np.float64],
                                generators: Matrix, sites: list[int],
                                reference_frame: Matrix, reference_derivatives: Matrix) -> dict[str, Any]:
    """Complete endpoint consumer for an even constant-trace reference frame."""
    profile = np.asarray([np.sqrt(.37), np.sqrt(.63)*np.exp(.61j)], dtype=np.complex128)
    maps: dict[int, Matrix] = {}
    same_site_maps: dict[int, Matrix] = {}
    rows = []
    frames = [path@reference_frame for path in paths]
    frame_gram_error = float(np.linalg.norm(reference_frame.conj().T@reference_frame-np.eye(4)))
    residual(float(abs(np.vdot(reference_frame, reference_frame)-4)), 'reference frame trace norm')
    for charge in (-1, 1):
        bare, neutral, target = insertion(0, charge)
        local_reference, _, _ = insertion(1, charge)
        coefficients = profile.conj() if charge == -1 else profile
        def dress(source: Matrix, frame: Matrix, angle: float) -> Matrix:
            rotation = frame.conj().T if charge == -1 else frame.T
            return np.exp(1j*charge*angle)*np.einsum(
                'ba,sac->sbc', rotation, source.reshape(len(target), 4, len(neutral))).reshape(source.shape)
        remote = dress(bare, frames[0], 0.)
        local = dress(local_reference, reference_frame, 0.)
        def coherent(angle: float) -> Matrix:
            return coefficients[0]*np.exp(1j*charge*angle)*remote+coefficients[1]*local
        b = coherent(float(angles[0]))
        maps[charge] = b
        same_site_maps[charge] = (coefficients[0]*np.exp(1j*charge*float(angles[0]))*remote
                                 +coefficients[1]*dress(bare, frames[1], float(angles[1])))
        colour_error = 0.
        for t, dm in zip(generators/np.sqrt(8), reference_derivatives, strict=True):
            left = np.zeros((8, 8), dtype=np.complex128)
            right = np.zeros_like(left)
            left[:4, :4] = t
            right[4:, 4:] = t
            ql0, _ = exterior_hamiltonian(left, 4)
            qr0, _ = exterior_hamiltonian(right, 4)
            qlt, _ = exterior_hamiltonian(left, 4+charge)
            qrt, _ = exterior_hamiltonian(right, 4+charge)
            ql = np.kron(qlt, np.eye(4))
            reference = t if charge == -1 else -t.T
            qr = np.kron(qrt, np.eye(4))+np.kron(np.eye(len(target)), reference)
            def derivative(source: Matrix, delta_frame: Matrix) -> Matrix:
                rotation = -delta_frame.conj().T if charge == -1 else delta_frame.T
                return np.einsum('ba,sac->sbc', rotation,
                    source.reshape(len(target), 4, len(neutral))).reshape(source.shape)
            factor = coefficients[0]*np.exp(1j*charge*float(angles[0]))
            dl = factor*derivative(bare, t@frames[0])
            dr = factor*derivative(bare, -paths[0]@t@reference_frame+paths[0]@dm)
            dr += coefficients[1]*derivative(local_reference, dm)
            colour_error = max(colour_error,
                float(np.linalg.norm(ql@b-b@ql0-dl)),
                float(np.linalg.norm(qr@b-b@qr0-dr)))
        residual(colour_error, 'coherent distinct-site all30 colour Gauss intertwiners')
        em_error = 0.
        h = 1e-5
        for site, orientation in ((0, 1), (1, -1)):
            nt = np.repeat(occupations(target, site)-2-charge*int(site == 1), 4)
            nn = occupations(neutral, site)-2
            matter = nt[:, None]*b-b*nn[None, :]
            numerical = -1j*(coherent(float(angles[0])+orientation*h)
                -coherent(float(angles[0])-orientation*h))/(2*h)
            em_error = max(em_error, float(np.linalg.norm(numerical-matter)))
        residual(em_error, 'coherent distinct-site U1 Gauss source derivatives')
        rows.append(dict(charge=charge, all30_colour_error=colour_error, U1_source_error=em_error))
    nminus = maps[-1].conj().T@maps[-1]
    nplus = maps[1].conj().T@maps[1]
    complement = float(np.linalg.norm(nminus+nplus-4*np.eye(len(neutral))))
    residual(complement, 'four-mode coherent CAR trace complement')
    # Actual ordered light CAR particle-hole action; value conjugation is separate.
    particle_hole = np.zeros_like(nminus)
    lookup = {mask: i for i, mask in enumerate(neutral)}
    for column, mask in enumerate(neutral):
        state, sign = mask, 1
        for orbital in reversed(range(8)):
            acted = car_action(state, orbital, not bool((state>>orbital) & 1))
            assert acted is not None
            state, step = acted
            sign *= step
        particle_hole[lookup[state], column] = sign
    ph_error = float(np.linalg.norm(particle_hole@nminus.conj()@particle_hole.conj().T-nplus))
    residual(ph_error, 'coherent profile particle-hole and value-conjugation norm')
    wrong = float(np.linalg.norm(same_site_maps[-1].conj().T@same_site_maps[-1]
        +same_site_maps[1].conj().T@same_site_maps[1]-4*np.eye(len(neutral))))
    try:
        residual(wrong, 'four-mode coherent CAR trace complement')
    except AssertionError:
        pass
    else:
        raise AssertionError('same-site multipath sum must reject automatic canonical-profile claim')
    return dict(profile_sites=sites, profile=profile, profile_probabilities=abs(profile)**2,
        norm_complement_error=complement, particle_hole_value_conjugation_error=ph_error,
        reference_mode_Gram_error=frame_gram_error, canonical_four_modes=frame_gram_error < 1e-10,
        coherent_Gauss_rows=rows, same_site_multipath_noncanonical_error=wrong,
        wrong_automatic_isometry_claim_rejected=True,
        neutral_photon_marginal='analytic exact equality for C-invariant density, not a sampled quantum ground',
        ground_profile_isometry='C-even constant trace gives 2||f||² vacuum norm, not arbitrary-state isometry',
        actual_invariant_particle_dynamics_proved=False, uniform_separated_energy_proved=False)


def _reference_spin_frame(fields: Matrix, epsilon: float) -> Matrix:
    """Globally smooth even spin-bivector filter with constant squared trace."""
    h = -1j*(fields[1]@fields[2]-fields[2]@fields[1])
    values, vectors = np.linalg.eigh(h)
    weights = (1+values/np.hypot(values, epsilon))/2
    filtered = np.asarray((vectors*weights)@vectors.conj().T, dtype=np.complex128)
    return 2*filtered/np.linalg.norm(filtered)


def _reference_spin_profile_controls(paths: Matrix, angles: NDArray[np.float64],
                                     generators: Matrix, sites: list[int],
                                     reference_fields: Matrix) -> dict[str, Any]:
    epsilon = .05
    frame = _reference_spin_frame(reference_fields, epsilon)
    even_error = float(np.linalg.norm(_reference_spin_frame(-reference_fields, epsilon)-frame))
    residual(even_error, 'actual full-field reference frame inversion')
    zero_error = float(np.linalg.norm(_reference_spin_frame(np.zeros_like(reference_fields), epsilon)-np.eye(4)))
    residual(zero_error, 'unrestricted zero-field reference frame')
    derivatives = []
    covariance_error = 0.
    step = 1e-5
    for t in generators/np.sqrt(8):
        rotation = expm(1j*step*t)
        plus = np.asarray([rotation@field@rotation.conj().T for field in reference_fields],
                          dtype=np.complex128)
        minus = np.asarray([rotation.conj().T@field@rotation for field in reference_fields],
                           dtype=np.complex128)
        derivative = -1j*(_reference_spin_frame(plus, epsilon)
                         -_reference_spin_frame(minus, epsilon))/(2*step)
        covariance_error = max(covariance_error, float(np.linalg.norm(derivative-(t@frame-frame@t))))
        derivatives.append(derivative)
    residual(covariance_error, 'all15 actual Cartesian reference Gauss derivatives')
    result = _coherent_profile_controls(paths, angles, generators, sites, frame,
        np.asarray(derivatives, dtype=np.complex128))
    try:
        residual(result['reference_mode_Gram_error'], 'four canonical individual mode anticommutators')
    except AssertionError:
        pass
    else:
        raise AssertionError('nonunitary spin filter must reject the canonical-four-mode claim')
    return dict(**result, epsilon=epsilon, reference_frame=frame,
        field_inversion_error=even_error, zero_field_frame_error=zero_error,
        all15_Cartesian_gauge_derivative_error=covariance_error,
        wrong_four_canonical_modes_claim_rejected=True,
        mechanical_Noether_half_spin_proved=False, new_frame_uniform_energy_upper_proved=False)



def native_field_controls(law: BulkQuantumLaw, fields: Matrix, links: Matrix,
                          phases: NDArray[np.float64], constants: dict[str, float]) -> tuple[dict[str, float], Matrix]:
    h = law.onebody(fields, links, phases)
    grams = law.grams(fields)
    q2 = float(np.trace(grams, axis1=1, axis2=2).mean())
    trace2 = float(np.vdot(h, h).real)
    bound = 4*law.volume*constants['endpoint_trace_constant']*q2
    require(trace2 <= bound*(1+1e-12), 'actual full-field endpoint Frobenius bound')
    trace = float(abs(np.trace(h)))
    residual(trace, 'actual whole onebody trace zero')
    elastic = constants['kappa']/2*float(np.sum(grams**2))
    quartic_lower = constants['kappa']*law.volume*q2**2/8
    require(elastic >= quartic_lower*(1-1e-12), 'actual Gram quartic dominates whole field moment')
    values = np.linalg.eigvalsh(h)
    any_car_norm = float(np.abs(values).sum()/2)
    car_bound = 2*law.volume*np.sqrt(constants['endpoint_trace_constant']*q2)
    require(any_car_norm <= car_bound*(1+1e-12), 'whole Fock norm bound without occupation omission')
    return dict(actual_Q2=q2, actual_Gram_quartic=elastic, quartic_Q2_lower=quartic_lower,
                actual_onebody_trace_squared=trace2, endpoint_trace_squared_upper=bound,
                entire_Fock_norm=any_car_norm, entire_Fock_norm_upper=float(car_bound),
                whole_onebody_trace_error=trace), h


def run(size: int = 5, spacing: float = 1.) -> dict[str, Any]:
    require(size >= 5 and 0 < spacing <= 1, 'native original N>=5 and 0<a<=1')
    hashes = source_hashes()
    band = RotationBand(**{**BARE, 'a': spacing})
    law = BulkQuantumLaw(size, band=band)
    constants = field_constants(band)
    rng = np.random.default_rng(2221007)
    rows = []
    tree_h: Matrix | None = None
    path_inputs: tuple[Matrix, NDArray[np.float64], list[int]] | None = None
    coherent_inputs: tuple[Matrix, NDArray[np.float64], list[int]] | None = None
    coherent_reference_fields: Matrix | None = None
    for centre in ((0, 0, 0), (1, 0, 0), (1, 2, 3), (2, 2, 2)):
        selected, frame = original_frame(band, size, np.pi/2*np.asarray(centre), True)
        fields = np.tile(frame, (law.volume, 1, 1, 1))
        links = law.toron_links(np.pi/2*np.asarray(centre))
        for perturbed in (False, True):
            actual_fields = fields
            actual_links = links
            if perturbed:
                actual_fields = fields+np.einsum('xma,aij->xmij',
                    .08*rng.normal(size=(law.volume, 4, 15)), law.basis)
                actual_links = np.asarray([expm(.19j*np.einsum('a,aij->ij', coefficients, law.basis))
                    for coefficients in rng.normal(size=(law.volume*3, 15))]).reshape(law.volume, 3, 4, 4)@links
            phases = rng.uniform(-np.pi, np.pi, (law.volume, 3))
            diagnostics, h = native_field_controls(law, actual_fields, actual_links, phases, constants)
            if perturbed and path_inputs is None:
                middle = int(law.target[0, 0])
                end = int(law.target[middle, 1])
                path_inputs = (np.asarray([actual_links[0, 0], actual_links[middle, 1]], dtype=np.complex128),
                               np.asarray([phases[0, 0], phases[middle, 1]]), [0, middle, end])
                alternate_middle = int(law.target[0, 1])
                require(int(law.target[alternate_middle, 0]) == end, 'both native paths reach the same reference')
                coherent_inputs = (np.asarray([
                    actual_links[0, 0]@actual_links[middle, 1],
                    actual_links[0, 1]@actual_links[alternate_middle, 0]], dtype=np.complex128),
                    np.asarray([phases[0, 0]+phases[middle, 1],
                                phases[0, 1]+phases[alternate_middle, 0]]), [0, end])
                coherent_reference_fields = actual_fields[end]
            rows.append(dict(centre=centre, native_full10_Gram=selected['G'],
                             arbitrary_full_field_noncentral=perturbed, **diagnostics))
            if tree_h is None and not perturbed:
                other = int(law.target[0, 0])
                indices = np.r_[np.arange(4), 4*other+np.arange(4)]
                tree_h = h[np.ix_(indices, indices)]
    assert tree_h is not None
    tree = tree_reference_controls(tree_h, np.asarray(law.basis, dtype=np.complex128))
    assert path_inputs is not None
    separated = _separated_path_controls(path_inputs[0], path_inputs[1],
        np.asarray(law.basis, dtype=np.complex128), path_inputs[2])
    assert coherent_inputs is not None and coherent_reference_fields is not None
    coherent = _coherent_profile_controls(coherent_inputs[0], coherent_inputs[1],
        np.asarray(law.basis, dtype=np.complex128), coherent_inputs[2],
        np.eye(4, dtype=np.complex128), np.zeros_like(law.basis, dtype=np.complex128))
    oriented = _reference_spin_profile_controls(coherent_inputs[0], coherent_inputs[1],
        np.asarray(law.basis, dtype=np.complex128), coherent_inputs[2], coherent_reference_fields)
    require(source_hashes() == hashes, 'source/proof bindings unchanged throughout native consumer')
    return dict(governing_plan=PLAN, assignment_revision=PLAN_REVISION,
        template_revision=TEMPLATE_REVISION, source_sha256=hashes, source_bindings_unchanged=True,
        inputs=dict(size=size, spacing=spacing, full_fields=60*law.volume,
                    full_colour_Gauss=15*law.volume, original_NH=2*law.volume),
        field_coercivity_constants=constants, actual_field_rows=rows,
        full_CAR_local_reference=tree, all_checks_pass=True,
        separated_path_operator_controls=separated,
        coherent_profile_operator_controls=coherent,
        field_oriented_profile_operator_controls=oriented,
        analytic_statement='E_q-E0<=-<Hlight>0/(2V)<=U_a if E0<=0, Lambda>=1; native BO trajectory supplies premise and positive lower',
        periodic_quantum_ground_computed=False, native_N_globality_certified=False,
        carrier_reference_separation_prescribed=False, isolated_mobile_electron_proved=False,
        campaign_complete=False, limits=[
            'Local insertion at the actual static opposite reference; not deconfinement or separated mass.',
            'Full original Cartesian/gauge/CAR domains; scalar moment bound uses true ground expectation, not BO norm closeness.',
            'Uniform fixeda compensated lower/upper along Lambda(N), not fixedLambda uniform BO or interacting UV limit.',
            'Finite tree quantum diagnostics are not a full periodic photon ground, physical spin or scattering proof.'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=ROOT/'original_local_charge_upper_receipt.json')
    args = parser.parse_args()
    result = run(args.size, args.spacing)
    args.output.write_text(json.dumps(portable(result), indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=result['all_checks_pass'],
                         isolated_mobile_electron_proved=result['isolated_mobile_electron_proved'])))


if __name__ == '__main__':
    main()
