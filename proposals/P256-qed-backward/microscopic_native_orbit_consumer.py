#!/usr/bin/env python3
"""Source-bound full N3 microscopic Pauli/current/auxiliary consumer.

Runs the unchanged cached initial FUNCTION at its actual physical-G point. It
executes all eight flat-history Pauli words and actual off-code syndrome images.
The bounded auxiliary construction is not full interacting particle propagation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from fractions import Fraction
from typing import Any
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import numpy as np
from scipy.linalg import expm
from microscopic_native_orbit_lift import (NativePauliGraph, PauliWord, quadratic_norm_squared,
    encoded_slater_preparation, defect_commutator, selected_syndrome_bundle, conditional_fullfield_window)
from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_source_relative_embedding import construct
from prepared_neutral_full_onebody import full_coordinate_light_operators
from original_charged_principal import portable
from improved_phi_band import GAP_LOG2
from bosonic_original_law_join import quartic_vacuum_coefficients_from_l1


def operator_fingerprint(terms: Any) -> str:
    digest = hashlib.sha256()
    for term in terms:
        digest.update(repr((term.target, term.source, term.coefficient, term.path,
                            [(coefficient, word.x, word.z, word.phase) for coefficient, word in term.words])).encode())
    return digest.hexdigest()


def polynomial_defect_control(graph: NativePauliGraph, terms: Any) -> int:
    """Verify literal Z conjugation of every polynomial word, not just matrices."""
    errors = 0
    for term in terms:
        for edge in graph.spatial_edges:
            z = PauliWord(z=1 << edge)
            sign = 2 if term.target != term.source and graph.path_A(term.path).x & (1 << edge) else 0
            for _, word in term.words:
                if z @ word @ z != word.scale_i(sign): errors += 1
    return errors


def source_terms(graph: NativePauliGraph, plus: np.ndarray, minus: np.ndarray) -> Any:
    n = 4*graph.volume
    light = tuple(range(n, 2*n))
    return ([graph.bilinear(light[i], graph.ref[r], complex(plus[i, r].item())) for i, r in zip(*np.nonzero(plus))]
            + [graph.bilinear(graph.ref[r], light[i], complex(minus[r, i].item())) for r, i in zip(*np.nonzero(minus))])


def source_auxiliary_current_bundle(graph: NativePauliGraph, model: Any, current_terms: Any,
                                    source_action: np.ndarray, auxiliary: dict[str, Any]) -> dict[str, Any]:
    """Full precompression current image of the explicit off-code SOURCE bundle."""
    n = model.fibre.n
    base = graph.decode(current_terms)[n:2*n, n:2*n]
    original_product = model.fibre.bilinear_product(base.conj().T, base)
    original_norm2 = float(np.vdot(source_action, original_product @ source_action).real)
    total = defect_current_norm2 = 0.
    for edge, amplitude in zip(graph.spatial_edges, auxiliary['Delta_scaled_first_amplitudes']):
        shifted = graph.decode(current_terms, defect_edge=edge)[n:2*n, n:2*n]
        difference = base-shifted
        if np.any(difference):
            second = model.fibre.bilinear_product(shifted.conj().T, shifted)
            shifted_norm2 = float(np.vdot(source_action, second @ source_action).real)
            second_difference = model.fibre.bilinear_product(difference.conj().T, difference)
            defect_current_norm2 += graph.epsilon**2*float(np.vdot(source_action, second_difference @ source_action).real)
        else:
            shifted_norm2 = original_norm2
        total += abs(amplitude)**2*shifted_norm2
    return dict(original_J_SOURCE_full_PH_norm_squared=original_norm2,
                actual_D_J_on_SOURCE_auxQ_norm_squared=defect_current_norm2,
                Delta_squared_times_J_first_bundle_norm_squared=float(total),
                point_current_remainder_upper='58*||J(point)||*remainder_coefficient/actual_Delta^2',
                point_current_operator_norm_upper=float(58*np.linalg.norm(base, 2)),
                precompression_PH_retained=True, not_a_fullfield_current_graph=True)


def projectors_at_cached_seed(model: Any, graph: NativePauliGraph) -> tuple[np.ndarray, np.ndarray]:
    n = model.fibre.n
    result = []
    for charge in (1, -1):
        p = np.zeros((graph.modes, graph.modes), complex)
        p[:n, :n] = model.heavy_projector
        p[n:2*n, n:2*n] = model.fibre.projectors[charge]
        p[2*n:, 2*n:] = np.eye(4) if charge == 1 else np.zeros((4, 4))
        result.append(p)
    return tuple(result)

def renewed_native_fullfield_constants(model: Any) -> dict[str, Any]:
    """Actual changed path coefficients enter the all-aux quartic lower bound."""
    def cap(law):
        value = 4*abs(Fraction(float(law.onsite)))
        for _, scalar in law.paths:
            value += 8*sum((abs(Fraction(float(z.real)))+abs(Fraction(float(z.imag)))
                            for z in scalar), Fraction())
        return value
    heavy, light = cap(model.heavy_law), cap(model.law)
    coefficients = quartic_vacuum_coefficients_from_l1(model.law.volume, 2, heavy, light)
    lam = Fraction(model.Lambda)
    B = coefficients[0]*lam+coefficients[1]+coefficients[2]/lam
    upper = model.law.volume*(300*model.Lambda+(132/5)*model.Lambda**1.25+15*model.chi+4*model.c)
    return dict(heavy_l1=str(heavy), light_l1=str(light), B_N_coefficients=[str(x) for x in coefficients],
                all_aux_H0_lower=float(-B), code_vacuum_trial_upper=float(upper),
                Lambda=model.Lambda, original_c=model.c,
                trial_scope='original Gaussian/Haar onsite H2L2Ref4 colour singlet trial, vacuum upper only; never substituted for fixed Psi',
                fullfield_window_for_original_function=None, original_source_graph_certified=False,
                photon='52 coexact oscillators,3 free harmonic coordinates; no harmonic oscillator potential')



def consumer(cache: Path, manifest: Path) -> dict[str, Any]:
    started = time.monotonic()
    model, data = load_shared_native_cache(cache, manifest)
    epoch = loaded_source_epoch()
    q, em, _ = source_point(model, data)
    sample = model.basis_at(q, em)
    graph = NativePauliGraph(model.law, model.reference_site)
    algebra = graph.algebra_report()
    if algebra['primitive_algebra_errors'] or algebra['all8_local_flat_errors']: raise AssertionError(algebra)
    n = model.fibre.n
    light_modes = tuple(range(n, 2*n)); heavy_modes = tuple(range(n))
    phi, links, a = sample['point']['phi'], sample['point']['links'], sample['a']
    heavy_terms = graph.path_terms(model.heavy_law, phi, links, np.zeros_like(a), 0)
    light_terms = graph.path_terms(model.law, phi, links, a, 1)
    heavy_matrix = graph.decode(heavy_terms)[:n, :n]
    ordinary_matter = graph.decode(light_terms)-model.Lambda*graph.decode(heavy_terms)
    light_matrix = graph.decode(light_terms)[n:2*n, n:2*n]
    heavy_error = float(np.linalg.norm(heavy_matrix-model.massive_onebody(phi, links)))
    light_error = float(np.linalg.norm(light_matrix-model.law.onebody(phi, links, a)))
    if max(heavy_error, light_error) > 1e-10: raise AssertionError('Full current path lift mismatch')
    preparation = []
    for label, occupied, modes in [('heavy54', model.heavy_orbitals, heavy_modes),
                                  ('light54', model.fibre.occupied[1], light_modes),
                                  ('light58', model.fibre.occupied[-1], light_modes)]:
        prepared = encoded_slater_preparation(graph, occupied, modes)
        decoded = graph.decode(prepared['encoded_generator'])[np.ix_(modes, modes)]
        recovered = expm(decoded)[:, :occupied.shape[1]]
        error = float(np.linalg.norm(recovered-occupied))
        if error > 1e-9: raise AssertionError('Encoded quadratic preparation changed the cached seed')
        preparation.append(dict(label=label, rank=occupied.shape[1], error=error,
                                unitary_error=prepared['unitary_error'], generator_error=prepared['generator_error'],
                                Pauli_generator_sha256=operator_fingerprint(prepared['encoded_generator']),
                                seed_occupied=prepared['seed_occupied']))
    projectors = projectors_at_cached_seed(model, graph)
    neutral = sample['neutral'] @ data['neutral_state']
    packets = (model.photon_basis.packets @ data['packet_templates'].T).T
    legs = [(model.em['legs'] @ packet).reshape(model.law.volume, 3)/np.sqrt(2) for packet in packets]
    first, second = legs[0], legs[2].conj()
    operators = full_coordinate_light_operators(model.law, phi, links, a, first, second)
    current_terms = graph.path_terms(model.law, phi, links, a, 1, first)
    contact_terms = graph.path_terms(model.law, phi, links, a, 1, first, second)
    Jfull, Cfull = graph.decode(current_terms), graph.decode(contact_terms)
    J, contact = Jfull[n:2*n, n:2*n], Cfull[n:2*n, n:2*n]
    errors = dict(heavy=heavy_error, light=light_error,
                  positive_original_J=float(np.linalg.norm(J-operators['J'])),
                  original_contact=float(np.linalg.norm(contact-operators['C'])))
    if max(errors.values()) > 1e-10: raise AssertionError(errors)
    # Range-two omission changes the original photon current, not only a path count.
    dropped = graph.decode(graph.path_terms(model.law, phi, links, a, 1, first, drop_two=True))
    wrong_range_current_norm = quadratic_norm_squared(Jfull-dropped, projectors[0])
    histories = []
    for history in graph.central_bases:
        shifted = graph.decode(current_terms, history)
        histories.append(dict(history=history,
            current_difference_on_original_plus_neutral_norm_squared=quadratic_norm_squared(shifted-Jfull, projectors[0]),
            light_onebody_spectrum=np.linalg.eigvalsh(graph.decode(light_terms, history)[n:2*n, n:2*n])))
    profiles = {}
    literal_z_errors = polynomial_defect_control(graph, current_terms)
    if literal_z_errors: raise AssertionError('Pauli defect conjugation mismatch')
    for profile in ('source19', 'SOFT'):
        relative = construct(model, data, profile, 2.**-19)
        coherent = relative['literal_coherent_row']
        Z = relative['channel_metadata']['global_SOURCE_normalizer']
        source = model.source_at(phi, links, sample['theta'], a, profile)
        S, T = source['full108_plus_coefficients'], source['full108_minus_coefficients']
        Bterms = source_terms(graph, S, T)
        Bfull = graph.decode(Bterms)
        relative_marks = coherent['CAR_commutator_marks']
        derivatives = coherent['derivative_marks']
        rS = -1j*derivatives['plus']+relative_marks['plus']
        rT = -1j*derivatives['minus']+relative_marks['minus']
        rterms = source_terms(graph, rS, rT)
        rfull = graph.decode(rterms)
        r_marks = {1:rfull[n:2*n, 2*n:].reshape(model.law.volume, 4, 4),
                   -1:rfull[2*n:, n:2*n].reshape(4, model.law.volume, 4).transpose(1, 0, 2)}
        lifted_relative = model.fibre.source_vectors_from_marks(r_marks)['source_matrix']
        r_error = float(np.linalg.norm(lifted_relative-coherent['relative_matrix']))
        source_action = source['source_matrix'] @ neutral / Z
        original_source = sample['charged'] @ data[profile+'_normalized_source_state']
        source_error = float(np.linalg.norm(source_action-original_source))
        current_comm = Jfull @ Bfull-Bfull @ Jfull
        physical_pairs = []
        for term in current_terms:
            if term.source < n or term.target < n: continue
            match = next((other for other in Bterms if other.target == term.source), None)
            if match is not None:
                physical_pairs.append((graph.bilinear(term.target, term.source, 1., term.path),
                                       graph.bilinear(match.target, match.source, 1., match.path)))
            if len(physical_pairs) == 32: break
        Pauli_comm_error = max(graph.bilinear_commutator_error(left, right) for left, right in physical_pairs)
        if Pauli_comm_error > 1e-12: raise AssertionError('Literal fullgraph Pauli [J,B] algebra failed')
        contact_comm = Cfull @ Bfull-Bfull @ Cfull
        current_source_norm2 = sum(abs(neutral[i]/Z)**2*quadratic_norm_squared(current_comm, p)
                                   for i,p in enumerate(projectors))
        contact_source_norm2 = sum(abs(neutral[i]/Z)**2*quadratic_norm_squared(contact_comm, p)
                                   for i,p in enumerate(projectors))
        # Creator is actual original coefficient action at the same full52 point.
        photon_images = []
        for index, packet in enumerate(packets):
            creator = sum(data['packet_templates'][index, j]*data['charged_creators'][j] for j in range(4))
            actual = sample['charged'] @ (creator @ data[profile+'_normalized_source_state'])
            full_image = np.sqrt(2)*(packet @ em[:52])*source_action
            error = float(np.linalg.norm(actual-full_image))
            photon_images.append(dict(packet=index, creator_point_error=error,
                                       source_image_norm_squared=float(np.vdot(actual,actual).real)))
            if error > 1e-8*max(1.,float(np.linalg.norm(actual))): raise AssertionError('Original full52 photon image mismatch')
        if max(r_error, source_error) > 1e-8: raise AssertionError('Original source/rG binding mismatch')
        D_source = defect_commutator(graph, Bterms, projectors, neutral/Z)
        D_relative = defect_commutator(graph, rterms, projectors, neutral/Z)
        native_source_norm = float(np.vdot(source_action,source_action).real)
        auxiliary = selected_syndrome_bundle(graph, native_source_norm, GAP_LOG2, .1)
        if auxiliary['Delta_times_exact_auxQ_norm_lower'] <= 0: raise AssertionError('Nonzero auxiliary evolution not discriminated')
        auxiliary_current = source_auxiliary_current_bundle(graph, model, current_terms, source_action, auxiliary)
        profiles[profile] = dict(globalZ=Z, original_initial_FUNCTION_error=source_error,
            literal_SOURCE_lift_error=float(np.linalg.norm(Bfull[n:2*n,2*n:]-S)+np.linalg.norm(Bfull[2*n:,n:2*n]-T)),
            literal_rG_lift_error=r_error, original_source_norm_squared=native_source_norm,
            rG_image_norm_squared=float(np.vdot(lifted_relative@neutral/Z,lifted_relative@neutral/Z).real),
            original_current_relative_source_norm_squared=current_source_norm2,
            original_contact_relative_source_norm_squared=contact_source_norm2,
            literal_Pauli_J_SOURCE_commutator_error=Pauli_comm_error,
            literal_Pauli_commutator_paths_exercised=len(physical_pairs),
            actual_auxiliary_current_bundle=auxiliary_current,
            neutral_RIGHT_omission_error=coherent['projected_before_RIGHT_wrong_error'],
            corrected_neutral_RIGHT_error=coherent['corrected_projected_commutator_error'],
            actual_D_SOURCE_commutator=D_source, actual_D_rG_commutator=D_relative,
            actual_nonzero_syndrome_dressing=auxiliary, original_full52_incoming_images=photon_images,
            SOURCE_Pauli_sha256=operator_fingerprint(Bterms), rG_Pauli_sha256=operator_fingerprint(rterms))
    D_current = defect_commutator(graph, current_terms, projectors, neutral)
    if D_current['norm_squared'] <= 0 or wrong_range_current_norm <= 0: raise AssertionError('Wrong-case controls failed')
    # Point matrix graphs are NOT full-field operator-domain or spectral-window certificates.
    fullfield_constants = renewed_native_fullfield_constants(model)
    admission = conditional_fullfield_window(graph, float(np.linalg.norm(J)),
        fullfield_constants['all_aux_H0_lower'], fullfield_constants['code_vacuum_trial_upper'], None, 1., .1)
    result = dict(scope='NEW full native N3 edge-Pauli code/source/current/photon and actual bounded auxiliary dynamics; not full particle',
        graph=algebra, path_law='ImprovedPhiLaw.terms, ordered full one/two links and summed original U1 phase',
        Pauli_lift_errors=errors, actual_Z_conjugation_polynomial_errors=literal_z_errors,
        encoded_frozen_Slater_preparation=preparation, all8_flat_histories=histories,
        wrong_drop_range_two_current_on_original_neutral_norm_squared=wrong_range_current_norm,
        actual_D_original_current_commutator=D_current, profiles=profiles,
        photon_packet_Gram=packets.conj()@packets.T, photon_EM_coordinate_count=len(em),
        original_field_point_q=q, original_field_point_em55=em,
        source_paths='actual Ref attachment and canonical graph tree; field/source marks retain original physical Wilson tree separately',
        physical_charges='heavy EM neutral; light+Ref58, heavy54: total112; Ref charge Nref-4, onsite energy zero',
        conditional_fullfield_window=admission,
        renewed_fullfield_constants=fullfield_constants,
        ordinary_matter_law='dGamma(h_light)-Lambda*dGamma(h_heavy); Ref onsite zero, physical Coulomb retained by original fibre',
        ordinary_matter_on_original_neutral_norm_squared=[
            quadratic_norm_squared(ordinary_matter, p) for p in projectors],
        selected_state_window_paid=False, driven_source_window_paid=False, harmonic_position_coercive=False,
        auxQ_separate_from_PH_and_fieldoutside=True,
        fullfield_limitations=['Bounded Duhamel proof evolves Delta local cycles+D only, not ordinary full field/matter H.',
            'Unchanged-function and driven SOURCE spectral tails were not supplied; no fullfield admission.',
            'Phi graph scales gap^-3/4; coexact graph gap^-1/2; harmonic free coordinate has no oscillator potential.',
            'All8 code algebra exact and cached Slater functions isometric; no independent 2^355 evolution.',
            'Full interacting propagation, dressed mass gap, physical spatial half-spin and scattering/spacetime joins remain open.'],
        source_epochs=epoch, loaded_sources_stable=epoch==loaded_source_epoch(),
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        Pauli_source_sha256=hashlib.sha256(Path(__file__).with_name('microscopic_native_orbit_lift.py').read_bytes()).hexdigest(),
        consumer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-started)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = consumer(args.cache_npz, args.cache_manifest)
    text = json.dumps(portable(result), sort_keys=True)
    args.output.write_text(text)
    os.chmod(args.output, 0o600)
    print(text)


if __name__ == '__main__': main()
