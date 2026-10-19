#!/usr/bin/env python3
"""Original itinerant-heavy toron bound, without setting flat links to identity.

The endpoint matrix helper is reused from bulk_quantum_selection, but its
added local CAR selector and changed heavy coefficient are NOT used here.
All occupations below are the original whole-total-half heavy occupation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import block_diag, expm

from bulk_quantum_selection import BulkQuantumLaw
from local_orbital_stability import periodic_grid
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, gram, rank_report
from soft_budget import select_gram
from unrestricted_field_minimum import BARE, STIFFNESS

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'e1874737294f261dfc305be3f8d054bd1223a1a3142bdf1a70c6d6cc136b76a4'


def check(error, label, tolerance=2e-9):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')


def source_hashes():
    names = ('bulk_flat_holonomy_bound.py', 'bulk_flat_holonomy_bound.md',
        'bulk_quantum_selection.py', 'bulk_quantum_selection.md',
        'bulk_global_gram.py', 'bulk_global_gram.md', 'periodic_bulk_frame.py',
        'local_orbital_stability.py', 'sea_selection.py', 'soft_budget.py',
        'rotation_band.py', 'unrestricted_field_minimum.py')
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def error_bound(frame, size, band):
    """Exact uniform periodic cell-integration error bound in source units."""
    return float(math.pi/(band.a*size) * (3*abs(band.r)*np.linalg.norm(frame[0])
        + abs(band.v)*sum(np.linalg.norm(frame[axis]) for axis in (1, 2, 3))))


def gram_functional(frame, momenta, band, stiffness):
    matrix = np.einsum('pm,mij->pij', coefficients(band, momenta), frame)
    radii = np.sqrt(np.sum(abs(matrix)**2, axis=(1, 2))/4)
    return float(stiffness/2*np.sum(gram(frame)**2)-2*radii.mean())


def selected_gram(band, momenta, stiffness):
    """Solve in aH units; the physical minimum value divides by a."""
    solution = select_gram(band.a*coefficients(band, momenta), band.a*stiffness, 0.)
    solution['value'] /= band.a
    return solution


def bloch_spectrum(frame, blocks, size, band):
    momenta = 2*np.pi*periodic_grid(size)/(size*band.a)
    matrix = np.zeros((len(momenta), 4, 4), dtype=complex)
    for indices, angles in blocks:
        x = coefficients(band, momenta+np.asarray(angles)/(size*band.a))
        block = frame[:, indices][:, :, indices]
        matrix[:, np.asarray(indices)[:, None], np.asarray(indices)[None, :]] = np.einsum('pm,mij->pij', x, block)
    return np.linalg.eigvalsh(matrix)


def flat_links(law, blocks):
    links = law.identity_links()
    for axis in range(3):
        phases = np.zeros(4)
        for indices, angles in blocks:
            phases[indices] = angles[axis]
        seam = np.diag(np.exp(1j*phases))
        check(abs(np.linalg.det(seam)-1), 'actual SU4 toron determinant')
        for x, site in enumerate(law.sites):
            if site[axis] == law.size-1:
                links[x, axis] = seam
    return links


def assess(frame, blocks, size, band, stiffness, reference_momenta, reference_size, *, native):
    values = bloch_spectrum(frame, blocks, size, band)
    count = size**3
    flat = values.reshape(-1)
    order = np.argsort(flat, kind='stable')
    ordered = flat[order]
    occupied_per_momentum = np.bincount(order[2*count:]//4, minlength=count)
    elastic = float(stiffness/2*np.sum(gram(frame)**2))
    energy = elastic-float(ordered[2*count:].sum()/count)
    wrong = elastic-float(values[:, 2:].sum()/count)
    check(band.a*abs(float(ordered.sum()/count)), 'aH whole source trace zero', 2e-10)
    reference = gram_functional(frame, reference_momenta, band, stiffness)
    debt = error_bound(frame, size, band)
    reference_debt = error_bound(frame, reference_size, band)
    lower = reference-debt-reference_debt
    if band.a*(energy-lower) < -2e-9:
        raise AssertionError('Original flat toron/global-half spectral integration bound failed')
    report: dict[str, Any] = dict(size=size, whole_total_half_energy_per_cell=energy,
        wrong_two_per_momentum_energy_per_cell=wrong, forced_block_error=wrong-energy,
        reference_finite_Gram_functional=reference, native_grid_error_upper=debt,
        reference_grid_error_upper=reference_debt, exact_finite_comparison_lower=lower,
        comparison_remainder=energy-lower, Gram=gram(frame).tolist(),
        occupation_histogram={str(number): int(np.sum(occupied_per_momentum==number))
                              for number in range(5)})
    if native:
        law = BulkQuantumLaw(size, band=band)
        fields = np.tile(frame, (law.volume, 1, 1, 1))
        links = flat_links(law, blocks)
        W, P = law.link_defects(fields, links)
        check(W, 'all native covariant alignment directions vanish')
        check(P, 'all native colour plaquettes flat')
        matrix = law.onebody(fields, links)
        native_values = np.linalg.eigvalsh(matrix)
        spectral_error = float(np.linalg.norm(native_values-ordered))
        check(band.a*spectral_error, 'aH full native endpoint seam/Bloch whole spectrum')
        check(band.a*abs(elastic-native_values[2*count:].sum()/count-energy), 'aH actual full native whole-half consumer')
        rng = np.random.default_rng(222810+size)
        local = np.array([expm(1j*np.einsum('a,aij->ij', rng.normal(size=15)*.1, law.generators))
                          for _ in range(law.volume)])
        moved_fields = local[:, None]@fields@local[:, None].conj().swapaxes(-1, -2)
        moved_links = np.array([[local[x]@links[x, axis]@local[law.target[x, axis]].conj().T
                                for axis in range(3)] for x in range(law.volume)])
        full = np.asarray(block_diag(*local))
        covariance = float(np.linalg.norm(law.onebody(moved_fields, moved_links)-full@matrix@full.conj().T))
        check(band.a*covariance, 'aH full native local colour covariance')
        report['native'] = dict(full_matrix_dimension=len(matrix), W=W, P=P,
            seam_Bloch_spectrum_error=spectral_error, local_colour_covariance_error=covariance,
            aH_seam_Bloch_spectrum_error=band.a*spectral_error,
            aH_local_colour_covariance_error=band.a*covariance)
    return report


def physical_trial_orbit_control(size, frame, band):
    """Actual complete-colour orbit regularity and whole-Fock isotropy character."""
    law = BulkQuantumLaw(size, band=band)
    derivatives = np.array([1j*(generator@frame-frame@generator) for generator in law.generators])
    # Gauge field-coordinate chart is HS/4. Its full rank already bounds the
    # full field/link orbit Jacobian from below; incidence adds a positive form.
    orbit_gram = np.einsum('amij,bmij->ab', derivatives.conj(), derivatives).real/4
    smallest = float(np.linalg.eigvalsh(orbit_gram)[0])
    if smallest <= 0:
        raise AssertionError('Actual full-rank Clifford orbit not regular modulo discrete centre')
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    values, vectors = np.linalg.eigh(law.onebody(fields, law.identity_links()))
    if values[2*law.volume-1] >= 0 or values[2*law.volume] <= 0:
        raise AssertionError('Actual whole-heavy positive Slater has no neutral half-filled gap')
    heavy = vectors[:, 2*law.volume:]
    _, rest = np.linalg.eigh(frame[0])
    light = np.asarray(block_diag(*[rest[:, :2] for _ in range(law.volume)]))
    phase_h = np.linalg.det(heavy.conj().T@(1j*heavy))
    phase_l = np.linalg.det(light.conj().T@(1j*light))
    check(abs(phase_h*phase_l-1), 'full physical trial global Z4 isotropy character')
    if size % 2 and abs(phase_h+1) > 1e-10:
        raise AssertionError('Heavy-only wrong-isotropy control does not reach odd-volume obstruction')
    return dict(full_local_colour_gauge_rank=15*law.volume,
        field_orbit_Gram_lower_eigenvalue=smallest,
        complete_graph_stabilizer='global Z4 only; all colour Lie generators have nonzero field tangent',
        heavy_whole_positive_gap=float(values[2*law.volume]-values[2*law.volume-1]),
        heavy_global_i_centre_phase=[float(phase_h.real), float(phase_h.imag)],
        light_global_i_centre_phase=[float(phase_l.real), float(phase_l.imag)],
        paired_isotropy_residual=float(abs(phase_h*phase_l-1)),
        light_site_occupation=2, light_EM_background=2,
        scope='regular compact orbit admits an equivariant full-Gauss tubular trial; no numerical quantum ground')


def run(*, sizes=(3, 4, 8, 16, 32), reference_size=32, spacing=1.):
    if not np.isfinite(spacing) or not 0 < spacing <= 1:
        raise ValueError('The source-selected cutoff family requires 0<a<=1')
    identities = source_hashes()
    band = RotationBand(**{**BARE, 'a': spacing})
    stiffness = STIFFNESS/spacing
    reference_momenta = 2*np.pi*periodic_grid(reference_size)/(reference_size*band.a)
    reference = selected_gram(band, reference_momenta, stiffness)
    richness = rank_report(coefficients(band, reference_momenta))
    if richness['quadratic_rank'] != 10:
        raise AssertionError('Actual source reference must exercise all10 Gram equality directions')
    target = np.sqrt(np.diag(reference['G']))[:, None, None]*SELECTED
    rng = np.random.default_rng(222810)
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    reducible = np.array([block_diag(np.einsum('a,aij->ij', rng.normal(size=3)*.25, pauli),
                                    np.einsum('a,aij->ij', rng.normal(size=3)*.25, pauli)) for _ in range(4)])
    reducible[0] += np.diag([.1, .1, -.1, -.1])
    cartan = np.array([np.diag(row-row.mean()) for row in rng.normal(size=(4, 4))*.12])
    cartan[0] = np.diag([5., 1., 1., -7.])
    two_blocks = [([0, 1], [.7, -.4, 1.2]), ([2, 3], [-.7, .4, -1.2])]
    cartan_angles = np.array([[.5, -.3, .7], [-.2, .6, -.5], [.8, -.9, .1]])
    cartan_angles = np.vstack([cartan_angles, -cartan_angles.sum(axis=0)])
    cartan_blocks = [([index], angles.tolist()) for index, angles in enumerate(cartan_angles)]
    rows = []
    trial_controls = []
    for size in sizes:
        if size < 3:
            raise ValueError('Native distinct-nearest-path periodic box requires N>=3')
        momenta = 2*np.pi*periodic_grid(size)/(size*band.a)
        central = []
        for angle in (0., math.pi):
            solution = selected_gram(band, momenta+angle/(size*band.a), stiffness)
            frame = np.sqrt(np.diag(solution['G']))[:, None, None]*SELECTED
            central.append(dict(angle=angle, **assess(frame, [([0, 1, 2, 3], [angle]*3)], size, band,
                stiffness, reference_momenta, reference_size, native=size in (3, 4))))
        reduced = assess(reducible, two_blocks, size, band, stiffness, reference_momenta, reference_size, native=size in (3, 4))
        transferred = assess(cartan, cartan_blocks, size, band, stiffness, reference_momenta, reference_size, native=size in (3, 4))
        if band.a*transferred['forced_block_error'] <= 1e-5:
            raise AssertionError('Actual original whole-half occupation-transfer boundary not reached')
        rows.append(dict(size=size, central=central, reducible_two_by_two=reduced, cartan_occupation_transfer=transferred))
        if size in (3, 4):
            trial_controls.append(dict(size=size, **physical_trial_orbit_control(size, target, band)))
    Cx = math.sqrt((abs(band.m)+6*abs(band.r)/band.a)**2+3*(band.v/band.a)**2)
    maximum_Gram_norm = (4*math.sqrt(2)*Cx/stiffness)**(2/3)
    C_D = 2*math.pi/band.a*math.sqrt(9*band.r**2+3*band.v**2)*math.sqrt(2*maximum_Gram_norm)
    if source_hashes() != identities:
        raise RuntimeError('Original toron bound source/proof changed during execution')
    return dict(source_sha256=identities, source_bindings_unchanged=True,
        governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION,
        bare_original={**BARE, 'a': spacing}, stiffness=stiffness, rows=rows, physical_trial_orbit_controls=trial_controls,
        reference=dict(grid=reference_size, Gram=reference['G'].tolist(), value=reference['value'], richness=richness,
            status='finite source-grid estimate, not interval continuum Gstar'),
        coercive_minimizer_Gram_norm_upper=maximum_Gram_norm, uniform_grid_error_prefactor=C_D,
        exact_classical_statement='Flat/covariant-uniform original whole-heavy class: kappa/2 distance-to-exact-continuum-Gstar squared plus continuum spectral slack <=2 C_D/N at any global minimizer.',
        quantum_extension='Declared original-heavy Lambda law with positive alignment/colour-plaquette Lambda^(5/4), full coordinate/Gauss/Fock domains, iterated Lambda then N shape-selection theorem in proof; not a numerical ground.',
        quantum_packet_powers=dict(normal_width=-3/16, field_kinetic=7/8,
            colour_kinetic=3/8, stiff_potential=7/8, leading_Lipschitz_remainder=13/16),
        local_selector_law_used=False, quantum_ground_computed=False, all_checks_pass=True, campaign_complete=False,
        limits=['Bare alignment/colour-plaquette trajectory is explicitly changed; original itinerant source/whole-CAR sectors retained.',
            'Iterated largeLambda then N limit, not uniform simultaneous volume limit or fixed-coupling theorem.',
            'No adiabatic light pole, charged lower mass, physical spin, compact photon phase, radiative scattering or Lorentz/UV closure.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[3, 4, 8, 16, 32])
    parser.add_argument('--reference-grid', type=int, default=32)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=ROOT/'bulk_flat_holonomy_bound_receipt.json')
    args = parser.parse_args()
    result = run(sizes=args.sizes, reference_size=args.reference_grid, spacing=args.spacing)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=result['all_checks_pass'],
        quantum_ground_computed=False, sizes=args.sizes)))


if __name__ == '__main__':
    main()
