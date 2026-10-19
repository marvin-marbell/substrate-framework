#!/usr/bin/env python3
"""Original full-domain angular stress: native Gauss/flow and IR discriminators.

The analytic whole-filter theorem is in the companion proof. This producer
never simulates the scalar quantum vacuum or asserts a compact photon phase.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, null_space

from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import OriginalSpatial, original_frame, plaquette_hessian, portable
from original_intrinsic_angular_transfer import MASS, source_hashes as angular_hashes
from original_normal_gap_bound import peierls_derivatives
from original_physical_rotation_lift import check, recover_gamma
from original_uniform_charge_gap import exterior_hamiltonian
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE, coordinates

Matrix = NDArray[np.complex128]
RealArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
ASSIGNMENT = 'b9c426960085aecd28d51c57e27e9cc42092d05a6bf09762578f27840a6eb11d'
TEMPLATE = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def source_hashes() -> dict[str, str]:
    return {**angular_hashes(), **{name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('original_total_angular_transfer.py', 'original_total_angular_transfer.md',
                     'original_uniform_charge_gap.py', 'original_normal_gap_bound.py')}}


def oriented_step(law: BulkQuantumLaw, x: int, axis: int, sign: int,
                  links: Matrix, phases: RealArray) -> tuple[int, Matrix, float]:
    if sign == 1:
        return int(law.target[x, axis]), links[x, axis], float(phases[x, axis])
    site = law.sites[x].copy()
    site[axis] = (site[axis]-1) % law.size
    y = law.index[tuple(site)]
    return y, links[y, axis].conj().T, -float(phases[y, axis])


def clover(law: BulkQuantumLaw, links: Matrix, phases: RealArray) -> tuple[Matrix, RealArray]:
    """All oriented loops are anchored; reverse seams use the actual inverse edge."""
    colour = np.zeros((law.volume, 3, 3, 4, 4), dtype=np.complex128)
    electromagnetic = np.zeros((law.volume, 3, 3), dtype=np.float64)
    for x in range(law.volume):
        for j, i in itertools.combinations(range(3), 2):
            for s, t in itertools.product((-1, 1), repeat=2):
                y, product, angle = x, np.eye(4, dtype=np.complex128), 0.
                for axis, sign in ((j, s), (i, t), (j, -s), (i, -t)):
                    y, edge, phase = oriented_step(law, y, axis, sign, links, phases)
                    product = product@edge
                    angle += phase
                if y != x:
                    raise AssertionError('Native oriented plaquette did not close')
                imaginary = (product-product.conj().T)/(2j)
                imaginary -= np.trace(imaginary)/4*np.eye(4)
                colour[x, j, i] += s*t*imaginary/(4*law.band.a**2)
                electromagnetic[x, j, i] += s*t*np.sin(angle)/(4*law.band.a**2)
            colour[x, i, j] = -colour[x, j, i]
            electromagnetic[x, i, j] = -electromagnetic[x, j, i]
    return colour, electromagnetic


def covariant_field_difference(law: BulkQuantumLaw, fields: Matrix, links: Matrix) -> Matrix:
    result = np.zeros((law.volume, 3, 4, 4, 4), dtype=np.complex128)
    zeros = np.zeros((law.volume, 3), dtype=np.float64)
    for x in range(law.volume):
        for j in range(3):
            y, u, _ = oriented_step(law, x, j, 1, links, zeros)
            z, w, _ = oriented_step(law, x, j, -1, links, zeros)
            result[x, j] = (u@fields[y]@u.conj().T-w@fields[z]@w.conj().T)/(2*law.band.a)
    return result


def primitive_flow(law: BulkQuantumLaw, fields: Matrix, links: Matrix, phases: RealArray,
                   velocity: RealArray, intrinsic: RealArray) -> tuple[Matrix, Matrix, RealArray]:
    differences = covariant_field_difference(law, fields, links)
    field_flow = -np.einsum('xj,xjmab->xmab', velocity, differences)
    # z-angular Cartesian-vector flow: Phi_1 -> -Phi_2, Phi_2 -> Phi_1.
    field_flow[:, 1] -= intrinsic[:, None, None]*fields[:, 2]
    field_flow[:, 2] += intrinsic[:, None, None]*fields[:, 1]
    colour, electromagnetic = clover(law, links, phases)
    link_flow = np.zeros_like(links)
    angle_flow = np.zeros_like(phases)
    for x in range(law.volume):
        for i in range(3):
            y, u = law.target[x, i], links[x, i]
            for j in range(3):
                v = (velocity[x, j]+velocity[y, j])/2
                b4 = (colour[x, j, i]+u@colour[y, j, i]@u.conj().T)/2
                b1 = (electromagnetic[x, j, i]+electromagnetic[y, j, i])/2
                link_flow[x, i] -= 1j*law.band.a*v*b4@u
                angle_flow[x, i] -= law.band.a*v*b1
    return field_flow, link_flow, angle_flow


def scalar_orbital(law: BulkQuantumLaw, links: Matrix, phases: RealArray,
                   velocity: RealArray) -> Matrix:
    """Hermitian full one-body covariant transport; no Phi/spin matrix is inserted."""
    result = np.zeros((4*law.volume, 4*law.volume), dtype=np.complex128)
    for x in range(law.volume):
        sx = slice(4*x, 4*x+4)
        for j in range(3):
            y = law.target[x, j]
            sy = slice(4*y, 4*y+4)
            edge = ((velocity[x, j]+velocity[y, j])/2
                    *links[x, j]*np.exp(1j*phases[x, j])/(2j*law.band.a))
            result[sx, sy] += edge
            result[sy, sx] += edge.conj().T
    return result


def endpoint_flow(law: BulkQuantumLaw, fields: Matrix, links: Matrix, phases: RealArray,
                  flow: tuple[Matrix, Matrix, RealArray]) -> Matrix:
    df, du, da = flow
    first, _ = peierls_derivatives(law, fields, links, phases, da)
    for x in range(law.volume):
        sx = slice(4*x, 4*x+4)
        first[sx, sx] += law.onsite*df[x, 0]
        for axis, scalar in law.paths:
            y = law.target[x, axis]
            sy = slice(4*y, 4*y+4)
            fx = np.einsum('m,mij->ij', scalar, fields[x])
            fy = np.einsum('m,mij->ij', scalar, fields[y])
            dfx = np.einsum('m,mij->ij', scalar, df[x])
            dfy = np.einsum('m,mij->ij', scalar, df[y])
            edge = (dfx@links[x, axis]+links[x, axis]@dfy
                    +fx@du[x, axis]+du[x, axis]@fy)/2*np.exp(1j*phases[x, axis])
            first[sx, sy] += edge
            first[sy, sx] += edge.conj().T
    return first


def gauge_move(law: BulkQuantumLaw, fields: Matrix, links: Matrix, phases: RealArray,
               gauge: Matrix, angles: RealArray) -> tuple[Matrix, Matrix, RealArray]:
    return (gauge[:, None]@fields@gauge[:, None].conj().swapaxes(-1, -2),
            gauge[:, None]@links@gauge[law.target].conj().swapaxes(-1, -2),
            phases+angles[:, None]-angles[law.target])


def native_flow_case(law: BulkQuantumLaw, frame: Matrix, centre: tuple[int, ...],
                     rng: np.random.Generator) -> dict[str, Any]:
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    fields += .025*np.einsum('xma,aij->xmij', rng.normal(size=(law.volume, 4, 15)), law.basis)
    links = law.toron_links(np.pi/2*np.asarray(centre))
    for x, j in itertools.product(range(law.volume), range(3)):
        links[x, j] = expm(.08j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))@links[x, j]
    phases = rng.uniform(-np.pi, np.pi, (law.volume, 3))
    positions = law.band.a*((law.sites+law.size//2) % law.size-law.size//2)
    velocity = np.column_stack((-positions[:, 1], positions[:, 0], np.zeros(law.volume)))
    weight = np.ones(law.volume, dtype=np.float64)
    flow = primitive_flow(law, fields, links, phases, velocity, weight)
    derivative = endpoint_flow(law, fields, links, phases, flow)
    h = law.onebody(fields, links, phases)
    orbital = scalar_orbital(law, links, phases, velocity)
    ward = derivative+1j*(orbital@h-h@orbital)
    gauge = np.asarray([expm(1j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))
                        for _ in range(law.volume)])
    angles = rng.uniform(-1., 1., law.volume)
    moved_fields, moved_links, moved_phases = gauge_move(law, fields, links, phases, gauge, angles)
    g = np.zeros_like(h)
    for x in range(law.volume):
        g[4*x:4*x+4, 4*x:4*x+4] = np.exp(1j*angles[x])*gauge[x]
    moved_flow = primitive_flow(law, moved_fields, moved_links, moved_phases, velocity, weight)
    flow_errors = [float(np.linalg.norm(moved_flow[0]-gauge[:, None]@flow[0]@gauge[:, None].conj().swapaxes(-1, -2))),
                   float(np.linalg.norm(moved_flow[1]-gauge[:, None]@flow[1]@gauge[law.target].conj().swapaxes(-1, -2))),
                   float(np.linalg.norm(moved_flow[2]-flow[2]))]
    for error in flow_errors:
        check(law.band.a*error, 'primitive full-field/link flow local gauge covariance', 2e-8)
    moved_orbital = scalar_orbital(law, moved_links, moved_phases, velocity)
    orbital_error = float(np.linalg.norm(moved_orbital-g@orbital@g.conj().T))
    check(law.band.a*orbital_error, 'full covariant scalar orbital local Gauss', 2e-8)
    moved_h = law.onebody(moved_fields, moved_links, moved_phases)
    moved_ward = (endpoint_flow(law, moved_fields, moved_links, moved_phases, moved_flow)
                  +1j*(moved_orbital@moved_h-moved_h@moved_orbital))
    ward_error = float(np.linalg.norm(moved_ward-g@ward@g.conj().T))
    check(law.band.a*ward_error, 'actual native interacting rotation Ward covariance', 3e-8)
    wrong = scalar_orbital(law, law.identity_links(), np.zeros_like(phases), velocity)
    wrong_defect = float(np.linalg.norm(wrong-g@wrong@g.conj().T))
    if law.band.a*wrong_defect <= 1e-4:
        raise AssertionError('Bare ordinary derivative did not discriminate local Gauss failure')
    df, du, da = flow
    errors = []
    for step in (1e-4, 5e-5):
        changed = []
        for sign in (-1, 1):
            altered_links = np.empty_like(links)
            for x, j in itertools.product(range(law.volume), range(3)):
                tangent = du[x, j]@links[x, j].conj().T
                altered_links[x, j] = expm(sign*step*tangent)@links[x, j]
            changed.append(law.onebody(fields+sign*step*df, altered_links, phases+sign*step*da))
        errors.append(float(np.linalg.norm((changed[1]-changed[0])/(2*step)-derivative)))
    relative = law.band.a*errors[-1]/max(1., law.band.a*float(np.linalg.norm(derivative)))
    check(relative, 'exact current-derived primitive endpoint flow', 2e-6)
    return dict(centre=centre, nonuniform_full60_fields=True, noncentral_SU4_links=True,
        arbitrary_compact_EM=True, primitive_flow_gauge_errors=flow_errors,
        covariant_orbital_Gauss_error=orbital_error, bare_orbital_Gauss_defect=wrong_defect,
        actual_interacting_Ward_gauge_error=ward_error, actual_finite_a_Ward_norm=float(np.linalg.norm(ward)),
        exact_endpoint_derivative_errors=errors, endpoint_derivative_relative_error=relative,
        continuous_conservation_claimed=False, chart_seam_defect_retained=True)


def full_car_gauss_control(law: BulkQuantumLaw, rng: np.random.Generator) -> dict[str, Any]:
    """An actual bond, every exterior sector; no non-Hermitian API workaround."""
    u = expm(1j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))
    angle, direction = .73, -.51
    edge = u*np.exp(1j*angle)/(2j*law.band.a)
    h = np.zeros((8, 8), dtype=np.complex128)
    h[:4, 4:], h[4:, :4] = edge, edge.conj().T
    derivative = np.zeros_like(h)
    derivative[:4, 4:] = 1j*direction*edge
    derivative[4:, :4] = derivative[:4, 4:].conj().T
    q = np.diag(np.r_[np.full(4, direction), np.zeros(4)]).astype(np.complex128)
    rows = []
    for particles in range(9):
        lifted, basis = exterior_hamiltonian(h, particles)
        lifted_derivative, derivative_basis = exterior_hamiltonian(derivative, particles)
        charge, charge_basis = exterior_hamiltonian(q, particles)
        if basis != derivative_basis or basis != charge_basis:
            raise AssertionError('Actual exterior sector bases disagree')
        error = float(np.linalg.norm(lifted_derivative-1j*(charge@lifted-lifted@charge)))
        check(law.band.a*error, 'entire exterior CAR photon/number Gauss Ward')
        rows.append(dict(particles=particles, dimension=len(basis), Gauss_Ward_error=error))
    if sum(row['dimension'] for row in rows) != 256:
        raise AssertionError('Every two-site CAR occupation must be exercised')
    return dict(full_two_site_CAR_dimension=256, all_number_sectors=rows,
        full_graph_fibre_claim='analytic exterior functor, no global Fock truncation or per-site occupation')


def all_centre_stars(law: BulkQuantumLaw) -> tuple[list[dict[str, Any]], dict[tuple[int, ...], tuple[dict[str, Any], Matrix]]]:
    rows, data = [], {}
    phases = np.zeros((law.volume, 3), dtype=np.float64)
    for centre in itertools.product(range(4), repeat=3):
        angles = np.pi/2*np.asarray(centre)
        selected, frame = original_frame(law.band, law.size, angles, True)
        fields = np.tile(frame, (law.volume, 1, 1, 1))
        links = law.toron_links(angles)
        colour, _ = clover(law, links, phases)
        difference = covariant_field_difference(law, fields, links)
        check(float(np.linalg.norm(difference)), 'all64 actual covariantly constant field differences')
        check(float(np.linalg.norm(colour)), 'all64 central-star colour stress with reverse seams')
        k = 2*np.pi*law.sites/law.size+angles/law.size
        x = np.column_stack((law.band.m+law.band.r/law.band.a*np.sum(1-np.cos(k), axis=1),
                             law.band.v/law.band.a*np.sin(k)))
        blocks = np.einsum('km,mij->kij', x, frame)
        radii = np.sqrt(np.einsum('km,mn,kn->k', x, selected['G'], x))
        projectors = (np.eye(4)[None]+blocks/radii[:, None, None])/2
        trace_error = float(np.max(np.abs(np.trace(projectors, axis1=1, axis2=2)-2)))
        offsite = [abs(np.mean(np.exp(1j*k[:, j])*np.trace(projectors, axis1=1, axis2=2))) for j in range(3)]
        check(trace_error, 'actual whole-heavy Bloch projector trace2')
        check(max(offsite), 'whole-heavy scalar nearest-neighbour offsite trace, every centre')
        data[centre] = selected, frame
        rows.append(dict(centre=centre, native_full10_Gram=selected['G'],
            source_stationarity=selected['gradient_residual'],
            covariant_field_difference_norm=float(np.linalg.norm(difference)),
            anchored_colour_stress_norm=float(np.linalg.norm(colour)),
            actual_heavy_projector_trace_error=trace_error, scalar_offsite_trace_errors=offsite,
            finite_native_isometry_error=float(np.linalg.norm(selected['G']-np.diag(
                [selected['G'][0, 0], *[np.trace(selected['G'][1:, 1:])/3]*3])))))
    return rows, data


def normal_stiff_control(law: BulkQuantumLaw, frame: Matrix, centre: tuple[int, ...]) -> dict[str, Any]:
    model = OriginalSpatial(law.band, law.size, law.basis, coordinates(frame, law.basis).reshape(-1),
                            np.pi/2*np.asarray(centre))
    q = np.zeros(3, dtype=np.float64)
    gauge = model.gauge_columns(q)
    horizontal = null_space((np.sqrt(MASS)[:, None]*gauge).conj().T)/np.sqrt(MASS)[:, None]
    strong = model.alignment(q)+plaquette_hessian(q)
    b = horizontal.conj().T@strong@horizontal
    values, vectors = np.linalg.eigh((b+b.conj().T)/2)
    tolerance = 1e-9*max(1., float(values[-1]))
    kernel = horizontal@vectors[:, np.abs(values) <= tolerance]
    if kernel.shape != (105, 45):
        raise AssertionError('The actual complete uniform physical kernel must have45 directions')
    link_error = float(np.linalg.norm(kernel[60:]))
    mass_error = float(np.linalg.norm(gauge.conj().T@(MASS[:, None]*horizontal)))
    check(link_error, 'all45 soft directions have zero link tangent', 3e-8)
    check(mass_error, 'all15 complete mass-horizontal Gauss orthogonality', 3e-8)
    # Uniform field changes with no link tangent leave D Phi and B exactly zero.
    # This tests the entire kernel, not a handpicked Gram or Clifford tangent.
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.pi/2*np.asarray(centre))
    soft_errors = []
    for column in range(kernel.shape[1]):
        change = np.einsum('ma,aij->mij', kernel[:60, column].reshape(4, 15), law.basis)
        shifted = fields+.01*change[None]
        soft_errors.append(float(np.linalg.norm(covariant_field_difference(law, shifted, links))))
    check(max(soft_errors), 'complete physical soft family leaves orbital coefficient zero', 3e-8)
    return dict(centre=centre, gauge_generators=15, mass_horizontal_dimension=horizontal.shape[1],
        full_uniform_soft_dimension=kernel.shape[1], soft_link_tangent_norm=link_error,
        all45_soft_orbital_null_error=max(soft_errors), actual_mass_horizontal_error=mass_error,
        stiff_dimension_at_q0=horizontal.shape[1]-kernel.shape[1],
        no_Euclidean_or_field_only_slice=True)


def ir_rotation_control(law: BulkQuantumLaw, selected: dict[str, Any], frame: Matrix) -> dict[str, Any]:
    gram = np.asarray(selected['G'])
    gs = float(np.trace(gram[1:, 1:])/3)
    isotropic = np.diag([gram[0, 0], gs, gs, gs])
    check(float(np.linalg.norm(gram-isotropic)), 'invariant000 native isometric frame for IR control')
    gamma = recover_gamma(gram, frame)
    spin = -1j*gamma[1]@gamma[2]/2
    direction = np.array([.37, -.53, .41], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    mu, speed, wilson = law.band.m*np.sqrt(gram[0, 0]), law.band.v*np.sqrt(gs), law.band.r*np.sqrt(gram[0, 0])
    rows = []
    for scale in (.16, .08, .04, .02):
        p = scale*direction
        ap = law.band.a*p
        h = gamma[0]*(mu+wilson/law.band.a*np.sum(1-np.cos(ap)))
        h += speed/law.band.a*np.einsum('i,ijk->jk', np.sin(ap), gamma[1:])
        derivatives = np.asarray([wilson*np.sin(ap[j])*gamma[0]+speed*np.cos(ap[j])*gamma[j+1]
                                  for j in range(3)])
        remainder = -1j*(p[0]*derivatives[1]-p[1]*derivatives[0])+spin@h-h@spin
        # Explicit safe operator-norm bound from sine/cosine Taylor remainders.
        bound = 2*speed*law.band.a**2*float(np.linalg.norm(p))**3+2*wilson*law.band.a**3*float(np.linalg.norm(p))**4
        error = float(np.linalg.norm(remainder, ord=2))
        check(max(0., error-bound), 'actual source rotation cubic/quartic moment bound')
        rows.append(dict(momentum=p, actual_rotation_commutator_norm=error,
                         analytic_cubic_quartic_bound=bound, error_over_p_cubed=error/scale**3))
    if not all(rows[i+1]['actual_rotation_commutator_norm'] < rows[i]['actual_rotation_commutator_norm']/4
               for i in range(len(rows)-1)):
        raise AssertionError('Actual native nonaxis rotation remainder did not reach the long-wave boundary')
    return dict(derived_from_native_vector_tangent=True, free_band_only=True,
        primitive_raw_CAR_spin_assigned=False, no_momentum_cut=True, rows=rows,
        interacting_photon_state_or_spin_claim=False)


def harmonic_cross_discriminator() -> dict[str, Any]:
    rows = []
    for heavy in (1e8, 1e16, 1e32, 1e48):
        chi = heavy**(-.5)
        soft_frequency, stiff_frequency = heavy**.25, heavy**.375
        # Independent exact whole-axis Gaussian moments; no rotor/Fock cutoff.
        dangerous = np.sqrt(chi/(2*soft_frequency)*stiff_frequency/(2*chi))
        reversed_order = np.sqrt(chi/(2*stiff_frequency)*soft_frequency/(2*chi))
        rows.append(dict(Lambda=heavy, soft_coordinate_stiff_momentum_norm=dangerous,
                         stiff_coordinate_soft_momentum_norm=reversed_order,
                         coefficient_of_Lambda_one_sixteenth=dangerous/heavy**(1/16)))
    return dict(rows=rows, naive_uniform_cross_norm_disproved=True,
        material_repair='selfadjoint C-side weighted-momentum compression and real-vacuum means, not bare BO norm or discarded cross term',
        photon_repair='exact full-associated-bundle photon-fibre block comparison retaining all scalar vacuum offsets')


def run(*, size: int = 5, spacing: float = 1., seed: int = 223017) -> dict[str, Any]:
    if size < 5 or not np.isfinite(spacing) or not 0 < spacing <= 1:
        raise ValueError('Need native N>=5 and0<a<=1; global BO threshold remains existence-only')
    identities = source_hashes()
    law = BulkQuantumLaw(size, band=RotationBand(**{**BARE, 'a': spacing}))
    rows, data = all_centre_stars(law)
    rng = np.random.default_rng(seed)
    cases = ((0, 0, 0), (1, 0, 0), (1, 2, 3), (2, 2, 2))
    native = [native_flow_case(law, data[c][1], c, rng) for c in cases]
    normal = [normal_stiff_control(law, data[c][1], c) for c in cases]
    car = full_car_gauss_control(law, rng)
    ir = ir_rotation_control(law, *data[(0, 0, 0)])
    if source_hashes() != identities:
        raise RuntimeError('Angular proof/source owner bindings changed during execution')
    return dict(status='original_total_covariant_mechanical_stress_transfer_NOT_continuous_finite_cube_Noether_or_localized_spin',
        plan=PLAN, assignment_revision=ASSIGNMENT, template_revision=TEMPLATE,
        source_sha256=identities, source_bindings_unchanged=True,
        inputs=dict(size=size, spacing=spacing, seed=seed, all64_classes=True),
        all64_flat_stars=rows, native_current_Gauss_flow=native, full_mass_horizontal_stiff_control=normal,
        complete_bond_CAR=car, conditional_IR_rotation=ir, cross_norm_discriminator=harmonic_cross_discriminator(),
        checks=dict(all64_original_full10=True, actual_reverse_seams_and_clover=True,
            full15_mass_horizontal_Gauss=True, actual_native_current_and_covariant_flow=True,
            complete_bond_CAR_All256=True, no_changed27_selector_or_hard_cut=True,
            source_bindings_unchanged=True),
        scope=dict(whole_filter_transfer_and_photon_form='analytic companion proof, not numerical infinite-dimensional spectrum',
            native_N5_global_quantum_selection_proved=False, continuous_finite_cube_Noether=False,
            localized_interacting_halfspin=False, reference_removal=False, photon_phase=False,
            particle_or_infraparticle=False, spacetime_or_UV_closure=False, scattering=False,
            campaign_complete=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=223017)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(size=args.size, spacing=args.spacing, seed=args.seed)
    payload = json.dumps(portable(result), indent=2)
    if args.output:
        args.output.write_text(payload+'\n')
    else:
        print(payload)


if __name__ == '__main__':
    main()
