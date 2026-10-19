#!/usr/bin/env python3
"""Original-heavy noncompact compensated window and finite lift discriminators.

The theorem is full-domain; the executed quantum comparator has two vertices,
complete fixed-number CAR and a moving whole-heavy Slater line. It is not a
simulation of the full scalar/colour field or a measurement of BO remainders.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import block_diag, expm

from bulk_quantum_selection import BulkQuantumLaw
from local_orbital_stability import scalar_paths
from noncompact_charge_threshold import constants
from noncompact_native_current import Coupled, accept, adverse, exterior, source_operators
from noncompact_neutral_supplier import uniform_row, wedge
from original_charged_principal import OriginalSpatial, original_frame, portable
from original_slow_kinetic_bo import (
    deformed_links, jet_controls, normal_directions, source_hashes as bo_hashes,
    whole_heavy,
)
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis
from unrestricted_field_minimum import coordinates

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'ed42a4078d9dca07c06716c3774d2b67477eb55266b92f03b60c53bd0ece3c3c'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def source_hashes() -> dict[str, str]:
    names = (
        'noncompact_microscopic_charge_window.py', 'noncompact_microscopic_charge_window.md',
        'noncompact_charge_threshold.py', 'noncompact_charge_threshold.md',
        'noncompact_neutral_supplier.py', 'noncompact_neutral_supplier.md',
        'noncompact_native_current.py', 'noncompact_coulomb_carrier.md',
        'bulk_quantum_selection.py', 'local_orbital_stability.py',
        'rotation_band.py', 'sea_selection.py', 'unrestricted_field_minimum.py',
    )
    return {**bo_hashes(), **{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in names}}


def endpoint_blocks(fields, links, band):
    """Actual original endpoint formula on the comparator's three directed edges.

    This is the Coupled two-vertex multigraph, not a periodic N^3 field model.
    scalar_paths owns every coefficient; no 27-selector method is used.
    """
    onsite, paths = scalar_paths(band)
    local = block_diag(onsite * fields[0, 0], onsite * fields[1, 0])
    hops = []
    for displacement, scalar in paths:
        axes = np.flatnonzero(displacement)
        if len(axes) != 1 or displacement[axes[0]] != 1:
            raise ValueError('Only the original unimproved nearest-axis source is admitted')
        axis = int(axes[0])
        endpoints = np.einsum('m,xmij->xij', scalar, fields)
        hop = np.zeros((8, 8), dtype=complex)
        hop[:4, 4:] = (endpoints[0] @ links[axis] + links[axis] @ endpoints[1]) / 2
        hops.append(hop)
    if len(hops) != 3:
        raise ValueError('The comparator retains all three actual spatial source coefficients')
    return local, hops


class MovingFibre(Coupled):
    """True quadratic photon/Coulomb with arbitrary actual Phi and SU4 endpoints."""

    def __init__(self, n, radius, zeta, occupation, UE, K, frame, fields, links, band):
        super().__init__(n, radius, zeta, occupation, UE, K, frame)
        local, hops = endpoint_blocks(fields, links, band)
        self.onsite = exterior(local, self.rows)
        self.hops = [exterior(hop, self.rows) for hop in hops]


def slater_line(h, rows):
    data = whole_heavy(h, 4)
    occupied = data['vectors'][:, data['cut']:]
    omega = np.array([np.linalg.det(occupied[np.asarray(row), :]) for row in rows])
    accept(abs(np.vdot(omega, omega).real - 1), 'whole-heavy Slater norm', 2e-10)
    residual = exterior(-h, rows) @ omega - data['energy'] * omega
    accept(np.linalg.norm(residual), 'actual whole-heavy eigenline', 2e-10)
    return omega, data


def transform_wave(wave, light, reference):
    return np.einsum('dc,pcr,sr->pds', light, wave, reference, optimize=True)


def reference_action(model, wave):
    return np.stack([model.reshape(model.action(wave[:, :, r].ravel()))
                     for r in range(wave.shape[2])], axis=2)


def electric_matrix(n, radius, zeta, UE, K):
    """Exact compression of UE(p_coexact^2+p_harmonic^2), not total H_photon."""
    omega = 2 * math.sqrt(UE * K)
    electric = np.diag(omega / 4 * (2 * np.arange(n) + 1)).astype(complex)
    for j in range(n - 2):
        electric[j, j + 2] = electric[j + 2, j] = -omega / 4 * math.sqrt((j + 1) * (j + 2))
    ell = np.arange(-radius, radius + 1)
    return np.kron(electric, np.eye(len(ell))) + np.diag(
        np.tile(UE * (ell + zeta) ** 2 / 6, n))


def photon_action(matrix, wave):
    return (matrix @ wave.reshape(len(matrix), -1)).reshape(wave.shape)


def electric_boundary(actual, expected):
    return float(np.linalg.norm(actual - expected))


def coulomb_boundary(coulomb, lift):
    # Entire fixed-number CAR accept boundary, not a chosen eigenvector.
    return float(np.linalg.norm(coulomb[:, None] * lift - lift * coulomb[None, :]))


def shift_harmonic(wave, n, radius, shift):
    """Exact e^(i*shift*y) on a finite-support smooth harmonic core.

    The output cell basis is enlarged, never wrapped/truncated; integer shift
    preserves the same Bloch boundary condition and the same heavy line.
    """
    larger = radius + abs(shift)
    out = np.zeros((n, 2 * larger + 1, *wave.shape[1:]), dtype=complex)
    start = larger - radius + shift
    out[:, start:start + 2 * radius + 1] = wave.reshape(n, 2 * radius + 1, *wave.shape[1:])
    return out.reshape(n * (2 * larger + 1), *wave.shape[1:])


def point_source(neutral, target, link, q):
    if q == 1:
        primitive = source_operators(neutral, target)[0]
        coefficients = link.T
    else:
        primitive = source_operators(target, neutral)[0].conj().transpose(0, 2, 1)
        coefficients = link.conj().T
    return np.einsum('ab,bij->aij', coefficients, primitive)


def finite_lift_consumer(frame, band, n, radius, UE, K, zeta):
    basis = matrix_unit_basis()
    rng = np.random.default_rng(2221031)
    dfields = np.einsum('xma,aij->xmij', rng.normal(size=(2, 4, 15)), basis) / 12
    generators = np.einsum('ea,aij->eij', rng.normal(size=(3, 15)), basis) / 5
    heavy_rows = tuple(itertools.combinations(range(8), 4))
    base_fields = np.tile(frame, (2, 1, 1, 1))
    local, hops = endpoint_blocks(base_fields, np.tile(np.eye(4), (3, 1, 1)), band)
    hbase = local + sum((h + h.conj().T for h in hops), np.zeros((8, 8), complex))
    omega_base, _ = slater_line(hbase, heavy_rows)
    rows = []
    for t in (-.035, 0., .035):
        fields = base_fields + t * dfields
        links = np.array([expm(1j * t * T) for T in generators])
        gauge_generators = np.array([fields[0, 0] + .7 * fields[0, 1],
                                     fields[1, 0] - .4 * fields[1, 2]])
        gauge = np.array([expm(1j * (.31 + t) * T) for T in gauge_generators])
        onebody_gauge = block_diag(*gauge)
        moved_fields = np.array([gauge[x] @ fields[x] @ gauge[x].conj().T for x in range(2)])
        moved_links = np.array([gauge[0] @ u @ gauge[1].conj().T for u in links])
        local, hops = endpoint_blocks(fields, links, band)
        heavy_h = local + sum((h + h.conj().T for h in hops), np.zeros((8, 8), complex))
        local_g, hops_g = endpoint_blocks(moved_fields, moved_links, band)
        heavy_g = local_g + sum((h + h.conj().T for h in hops_g), np.zeros((8, 8), complex))
        omega, heavy = slater_line(heavy_h, heavy_rows)
        omega_actual, _ = slater_line(heavy_g, heavy_rows)
        heavy_lift = wedge(onebody_gauge, heavy_rows)
        lifted_heavy = heavy_lift @ omega
        heavy_line_error = accept(1 - abs(np.vdot(omega_actual, lifted_heavy)) ** 2,
                                  'actual real-Phi/U whole-heavy colour line', 2e-10)
        heavy_residual = exterior(-heavy_g, heavy_rows) @ lifted_heavy - heavy['energy'] * lifted_heavy
        accept(np.linalg.norm(heavy_residual), 'moving whole-heavy eigenline after local colour lift', 2e-10)
        neutral = MovingFibre(n, radius, zeta, 4, UE, K, frame, fields, links, band)
        neutral_g = MovingFibre(n, radius, zeta, 4, UE, K, frame, moved_fields, moved_links, band)
        energies, vectors, residuals = neutral.solve(2)
        E0 = float(energies[0])
        wave = neutral.reshape(vectors[:, 0])[:, :, None]
        F0 = wedge(onebody_gauge, neutral.rows)
        wave_g = transform_wave(wave, F0, np.eye(1))
        light_residual = reference_action(neutral_g, wave_g) - E0 * wave_g
        # Exact norm of r_H tensor psi + Omega_H tensor r_L. No 4900 x photon
        # dense Kronecker operator or duplicated heavy/reference tensor needed.
        joint_squared = (np.vdot(heavy_residual, heavy_residual).real * np.vdot(wave_g, wave_g).real
                         + np.vdot(lifted_heavy, lifted_heavy).real * np.vdot(light_residual, light_residual).real
                         + 2 * (np.vdot(heavy_residual, lifted_heavy) * np.vdot(wave_g, light_residual)).real)
        joint_residual = accept(math.sqrt(max(0., float(joint_squared))),
                                'actual heavy+quantum-photon/light lifted eigenpair', 5e-8)
        E = electric_matrix(n, radius, zeta, UE, K)
        quantum_rows = []
        for q in (-1, 1):
            target = MovingFibre(n, radius, zeta, 4 + q, UE, K, frame, fields, links, band)
            target_g = MovingFibre(n, radius, zeta, 4 + q, UE, K, frame, moved_fields, moved_links, band)
            F = wedge(onebody_gauge, target.rows)
            reference = gauge[1].conj() if q == 1 else gauge[1]
            source = point_source(neutral, target, links[0], q)
            primitive_g = point_source(neutral_g, target_g, moved_links[0], q)
            transported = np.einsum('dc,acj,kj->adk', F, source, F0.conj())
            transported = np.einsum('ba,aij->bij', reference, transported)
            source_error = accept(np.linalg.norm(primitive_g - transported),
                                  'complete-reference original source colour covariance', 2e-9)
            source_norm_operator = np.einsum('aij,aik->jk', source.conj(), source)
            norm_target = np.diag(4 - neutral.nx if q == 1 else neutral.nx)
            car_error = accept(np.linalg.norm(source_norm_operator - norm_target),
                               'complete-reference unfiltered source CAR norm operator', 2e-9)
            prepared = np.einsum('aij,pj->pia', source, wave[:, :, 0])
            prepared_g = transform_wave(prepared, F, reference)
            independently_prepared_g = np.einsum('aij,pj->pia', primitive_g, wave_g[:, :, 0])
            accept(np.linalg.norm(prepared_g - independently_prepared_g),
                   'same moving heavy-line full-reference prepared source', 2e-9)
            number_error = accept(np.linalg.norm(target.nx[:, None] * F - F * target.nx[None, :]),
                                  'every spatial light number preserved by the colour BO lift', 2e-9)
            C_label = 'same reduced Coulomb operator intertwines the lift'
            C_error = accept(coulomb_boundary(target.coulomb, F), C_label, 2e-9)
            mixing_onebody = np.zeros((8, 8), complex)
            mixing_onebody[0, 4] = mixing_onebody[4, 0] = 1
            false_F = expm(.27j * exterior(mixing_onebody, target.rows)) @ F
            false_C = adverse(coulomb_boundary(target.coulomb, false_F), C_label, 2e-9)
            E_label = 'same unbounded electric operator intertwines the lift on its smooth core'
            electric_expected = transform_wave(photon_action(E, prepared), F, reference)
            E_error = accept(electric_boundary(photon_action(E, prepared_g), electric_expected),
                             E_label, 2e-9)
            # A photon-dependent phase preserves the exact same heavy line and
            # all n_x; only the actual electric boundary must reject it.
            shifted = shift_harmonic(prepared_g, n, radius, 1)
            Elarge = electric_matrix(n, radius + 1, zeta, UE, K)
            shifted_expected = shift_harmonic(electric_expected, n, radius, 1)
            false_E_error = electric_boundary(photon_action(Elarge, shifted), shifted_expected)
            false_E = adverse(false_E_error, E_label, 2e-9)
            shifted_momenta = np.tile(np.arange(-radius, radius + 1) + zeta, n)
            predicted_shift = shift_harmonic(
                UE * (2 * shifted_momenta[:, None, None] + 1) * prepared_g / 6,
                n, radius, 1)
            accept(np.linalg.norm(photon_action(Elarge, shifted) - shifted_expected - predicted_shift),
                   'independent actual photon-phase electric product rule', 2e-9)
            norm = float(np.vdot(prepared, prepared).real)
            if norm <= 0:
                raise AssertionError('The actual complete-reference preparation is zero')
            actual_H = reference_action(target, prepared)
            lifted_H = reference_action(target_g, prepared_g)
            H_error = accept(np.linalg.norm(lifted_H - transform_wave(actual_H, F, reference)),
                             'full true quadratic-photon/Coulomb/CAR source Hamiltonian intertwining', 5e-9)
            rayleigh = float(np.vdot(prepared, actual_H).real / norm - E0)
            lifted_rayleigh = float(np.vdot(prepared_g, lifted_H).real / norm - E0)
            accept(abs(rayleigh - lifted_rayleigh), 'nontrivial actual source Rayleigh transfer', 2e-9)
            em_coulomb = (target.phenergy[:, None, None] + target.coulomb[None, :, None]) * prepared
            em_coulomb_g = (target_g.phenergy[:, None, None] + target_g.coulomb[None, :, None]) * prepared_g
            photon_coulomb_energy = float(np.vdot(prepared, em_coulomb).real / norm)
            accept(np.linalg.norm(em_coulomb_g - transform_wave(em_coulomb, F, reference)),
                   'entire quadratic photon plus physical Coulomb intertwining', 2e-9)
            quantum_rows.append(dict(
                q=q, NL=4 + q, static_reference_charge=-q, reference_dimension=4,
                complete_light_CAR_dimension=target.d, complete_heavy_CAR_dimension=len(heavy_rows),
                factored_full_heavy_light_reference_photon_dimension=4 * len(heavy_rows) * target.size,
                source_norm=norm, source_CAR_error=car_error, source_covariance_error=source_error,
                spatial_number_error=number_error, Coulomb_intertwining_error=C_error,
                spatial_number_mixing_false_lift_rejected=false_C,
                electric_intertwining_error=E_error, photon_phase_electric_defect=false_E_error,
                photon_dependent_same_heavy_line_false_lift_rejected=false_E,
                full_source_H_intertwining_error=H_error,
                actual_source_Rayleigh_excess=rayleigh, lifted_source_Rayleigh_excess=lifted_rayleigh,
                actual_photon_plus_Coulomb_source_energy=photon_coulomb_energy,
            ))
        frozen_error = float(np.linalg.norm(exterior(-heavy_h, heavy_rows) @ omega_base
                                            - heavy['energy'] * omega_base))
        frozen_rejection = None if t == 0 else adverse(frozen_error, 'actual whole-heavy eigenline', 2e-10)
        rows.append(dict(
            actual_field_coordinate=t, whole_heavy_gap=heavy['gap'], whole_heavy_energy=heavy['energy'],
            heavy_line_departure_from_base=float(max(0., 1 - abs(np.vdot(omega_base, omega)) ** 2)),
            actual_colour_line_error=heavy_line_error, frozen_heavy_line_rejected=frozen_rejection,
            neutral_coupled_E0=E0, neutral_eigenpair_residuals=residuals,
            actual_lifted_joint_eigenpair_residual=joint_residual,
            full_scalar_colour_field_kinetic_simulated=False, charged=quantum_rows,
        ))
    return dict(
        graph='two vertices with three directed endpoint edges and the existing Coupled cochain quotient',
        oscillator_levels=n, harmonic_radius=radius, Bloch_zeta=zeta,
        actual_whole_heavy_NH=4, entire_heavy_CAR_dimension=70, rows=rows,
        scope='finite true photon Galerkin plus complete CAR/moving Phi,U fibres; not the full-field BO isometry or its remainders',
    )


def endpoint_difference_bound(law, fields, links, moved_fields, moved_links):
    bounds = np.array([abs(law.onsite) * np.linalg.norm(moved_fields[x, 0] - fields[x, 0], 2)
                       for x in range(law.volume)])
    for x in range(law.volume):
        for axis, scalar in law.paths:
            y = law.target[x, axis]
            old = np.einsum('m,xmij->xij', scalar, fields[[x, y]])
            new = np.einsum('m,xmij->xij', scalar, moved_fields[[x, y]])
            a = (old[0] @ links[x, axis] + links[x, axis] @ old[1]) / 2
            b = (new[0] @ moved_links[x, axis] + moved_links[x, axis] @ new[1]) / 2
            weight = np.linalg.norm(b - a, 2)
            bounds[x] += weight
            bounds[y] += weight
    return float(bounds.max())


def native_controls(size, spacing):
    band = RotationBand(a=spacing, m=.2, r=.12, v=.3, improved=False)
    theta = np.asarray([1, 2, 3]) * math.pi / 2
    selected, frame = original_frame(band, size, theta, True)
    law = BulkQuantumLaw(size, band=band)
    model = OriginalSpatial(band, size, law.basis, coordinates(frame, law.basis).reshape(-1), theta)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(theta)
    rng = np.random.default_rng(2221032)
    direction = normal_directions(model, law, rng)
    jets, _ = jet_controls(model, law, fields, links, direction)
    amplitude = .003
    moved_fields = fields + amplitude * (direction['soft_fields'] + direction['stiff_fields'])
    moved_links = deformed_links(links, direction['generators'], amplitude)
    bound = endpoint_difference_bound(law, fields, links, moved_fields, moved_links)
    deviations = []
    for scale in (0., 3., 11.):
        A = rng.normal(size=(law.volume, 3)) * scale
        difference = law.onebody(moved_fields, moved_links, A) - law.onebody(fields, links, A)
        actual = float(np.max(np.abs(np.linalg.eigvalsh(difference))))
        accept(max(0., actual - bound), 'native phase-uniform actual endpoint deviation bound', 2e-9)
        deviations.append(dict(real_A_scale=scale, actual_onebody_deviation=actual))
    mu, _, schur = constants(selected['G'], frame)
    all64 = cast(dict[str, Any], uniform_row(spacing, size))
    for row in all64['actual_all64']:
        G = np.asarray(row['true10Gram'])
        s = float(G[0, 0] - G[0, 1:] @ np.linalg.solve(G[1:, 1:], G[1:, 0]))
        if s <= 0:
            raise ValueError('A sampled native metric lacks a positive Schur mass')
        row['actual_Schur'] = s
        row['sampled_mu'] = band.m * math.sqrt(s)
    all64['min_sampled_native_mu'] = min(row['sampled_mu'] for row in all64['actual_all64'])
    return frame, band, dict(
        size=size, volume=law.volume, centre=[1, 2, 3], actual_all10_Gram=selected['G'],
        actual_Schur=schur, sampled_mu=mu, original_whole_heavy_jets=jets,
        deterministic_deformation_amplitude=amplitude, phase_independent_endpoint_row_bound=bound,
        actual_real_phase_deviations=deviations, all64_uniform_native_row=all64,
        observed_deviations_are_BO_remainders=False, N0_or_N1_numerically_certified=False,
        added_27_selector_methods_used=False,
    )


def theorem_contract(spacing, UE, K):
    Cx = math.sqrt((.2 + 6 * .12 / spacing) ** 2 + 3 * (.3 / spacing) ** 2)
    kappa = 2 / spacing
    gram_bound = (4 * math.sqrt(2) * Cx / kappa) ** (2 / 3)
    Bstar = Cx * math.sqrt(2 * gram_bound)
    return dict(
        exact_scope='original full R^(60V), SU4^(3V), NH=2V and entire NL=2V+q CAR; real-link quadratic photon quotient; opposite static reference retained',
        microscopic_quantum_CAR_is_assumed=True, fixed_a=spacing, fixed_UE=UE, fixed_K=K,
        Bstar=Bstar, uniform_effective_point_upper=13 * UE / 4 + Bstar,
        lower='mu_min(N)-3*d_q(N,Lambda)-d_0(N,Lambda)',
        upper='13*UE/4+Bstar(a)+d_q(N,Lambda)+3*d_0(N,Lambda)+2*epsilon+epsilon_packet',
        actual_g_requirement='g_s(N,Lambda)>T_s(N) for s=0,+1,-1; T_s is the finite effective witness in proof section 4',
        scalar_offsets='all64 c_theta=E_theta-E_min retained in every neutral/charged infimum and supplier',
        uniform_quantifiers='for fixed a,UE,K>0 and N>=max(5,N0(a),N1(a)), there exists finite legal Lambda(N) with all residuals <=delta and actual g_s>T_s; 0<4delta<mu_floor(a)',
        uniform_floor='mu_floor(a)=m*sqrt(g0_infinity(a)/2)>0, not inferred from a floating N5 sample',
        uniform_window='[mu_floor-4*delta,13*UE/4+Bstar+5*delta] with 2epsilon+epsilon_packet<=delta',
        residual_values_numerically_certified=False, microscopic_window_numerically_observed=False,
        no_claims=['numerical N0/N1', 'fixed-Lambda uniform-N BO', 'reference-free/mobile mass',
                   'full-field quantum numerical spectrum', 'photon phase', 'dispersion',
                   'physical spin', 'Lorentz symmetry', 'Compton or campaign completion'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--UE', type=float, default=.37)
    parser.add_argument('--K', type=float, default=.9)
    parser.add_argument('--zeta', type=float, default=.23)
    args = parser.parse_args()
    if args.native_size < 5 or args.n < 3 or args.M < 1:
        raise ValueError('Native N>=5, oscillator levels>=3 and harmonic radius>=1 required')
    if not all(math.isfinite(x) and x > 0 for x in (args.spacing, args.UE, args.K)) or not math.isfinite(args.zeta):
        raise ValueError('Finite positive a,UE,K and finite Bloch label required')
    identities = source_hashes()
    frame, band, native = native_controls(args.native_size, args.spacing)
    finite = finite_lift_consumer(frame, band, args.n, args.M, args.UE, args.K, args.zeta)
    if source_hashes() != identities:
        raise RuntimeError('Source/proof dependencies changed during the actual consumer execution')
    result: dict[str, Any] = dict(
        governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION, template_revision=TEMPLATE_REVISION,
        source_sha256=identities, source_bindings_unchanged=True, inputs=vars(args),
        exact_theorem=theorem_contract(args.spacing, args.UE, args.K), native_source_controls=native,
        finite_actual_lift_consumer=finite, all_checks_pass=True, campaign_complete=False,
        scientific_execution_owner='parent; no numerical BO residual or full field simulation claimed',
    )
    print(json.dumps(portable(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
