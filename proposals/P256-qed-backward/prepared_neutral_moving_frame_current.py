#!/usr/bin/env python3
"""Pointwise full108 photon/SOURCE graphs on once-prepared moving matter.

No preparation, H/G compilation, instantaneous sea, or whole-field integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np

from prepared_neutral_full_onebody import full_coordinate_light_operators
from prepared_neutral_weyl_hard_fock import packet_templates


def source_from_coefficients(fibre: Any, plus: np.ndarray, minus: np.ndarray) -> dict[str, Any]:
    """Actual even light/Ref bilinear, including both physical charge channels."""
    return fibre.source_vectors_from_marks({
        1: plus.reshape(fibre.law.volume, 4, 4),
        -1: minus.reshape(4, fibre.law.volume, 4).transpose(1, 0, 2)})


def current_graph(fibre: Any, body: np.ndarray, source: dict[str, Any],
                  neutral: np.ndarray) -> dict[str, Any]:
    """Full Wick BEFORE compression, actual filtered point coefficients."""
    vector = source['source_matrix'] @ neutral
    weak = fibre.onebody(body)
    second = fibre.bilinear_product(body.conj().T, body)
    represented = weak @ vector
    full2 = float(np.vdot(vector, second @ vector).real)
    projected2 = float(np.vdot(represented, represented).real)
    # The exact even-CAR commutator has J S and -T J physical marks.
    plus, minus = source['full108_plus_coefficients'], source['full108_minus_coefficients']
    relative = source_from_coefficients(fibre, body @ plus, -minus @ body)['source_matrix']
    small_right = source['source_matrix'] @ fibre.onebody(body, 'neutral')
    outside_right = np.zeros_like(relative)
    outside_right[fibre.charged_slices[1], 0] = (
        -fibre.empty.conj().T @ body @ fibre.projectors[1] @ plus).T.reshape(-1)
    outside_right[fibre.charged_slices[-1], 1] = (
        minus @ (np.eye(fibre.n) - fibre.projectors[-1]) @ body @ fibre.occupied[-1]).reshape(-1)
    projected_difference = weak @ source['source_matrix'] - small_right
    identity_error = float(np.linalg.norm(projected_difference - outside_right - relative))
    if identity_error > 1e-10*(np.linalg.norm(relative)+np.linalg.norm(outside_right)):
        raise AssertionError('Full physical current commutator lost neutral RIGHT')
    relative_value = relative @ neutral
    return dict(full_norm_squared=full2, projected_norm_squared=projected2,
                unrepresented_PH_norm_squared=full2-projected2,
                response=represented, source_vector=vector, relative_map=relative,
                relative_value=relative_value,
                relative_norm_squared=float(np.vdot(relative_value, relative_value).real),
                neutral_RIGHT_projected_value=small_right @ neutral,
                neutral_RIGHT_outside_value=outside_right @ neutral,
                neutral_RIGHT_outside_norm_squared=float(np.linalg.norm(outside_right @ neutral)**2),
                full_neutral_RIGHT_identity_error=identity_error,
                wrong_recaptured_first_order_projected_substitute=projected_difference @ neutral,
                wrong_recaptured_substitute_error=float(np.linalg.norm(outside_right @ neutral)),
                wrong_project_before_current_lost_norm_squared=full2-projected2)


def moving_current_control(model: Any, moved: Any, data: dict[str, Any],
                           sample: dict[str, Any], light_unitary: np.ndarray,
                           reference_unitary: np.ndarray) -> dict[str, Any]:
    """Transport primitive operators independently into the fixed seed fibre.

    sample must be moved.basis_at at the actual selected off-mean point. Its
    neutral@data.neutral_state, never fibre.raw_neutral, is used for every graph.
    """
    ul, ur = np.asarray(light_unitary), np.asarray(reference_unitary)
    point, a = sample['point'], sample['a']
    neutral = sample['neutral'] @ data['neutral_state']
    templates = np.asarray(data['packet_templates'])
    recipe = packet_templates(model.law, model.photon_basis)
    packet_error = float(np.linalg.norm(templates-recipe['creator_hard_coefficients']))
    if packet_error > 1e-12:
        raise AssertionError('Accepted literal photon templates changed')
    all52 = (model.photon_basis.packets @ templates.T).T
    rows = []
    sources = {name: model.source_at(point['phi'], point['links'], sample['theta'], a, name)
               for name in ('source19', 'SOFT')}
    moving_sources = {name: moved.source_at(point['phi'], point['links'], sample['theta'], a, name)
                      for name in sources}
    for packet_name, packet in zip(('gi', 'gfwd', 'gout'), all52):
        first = (model.em['legs'] @ packet).reshape(model.law.volume, 3)/math.sqrt(2)
        independent_leg = (model.em['vectors'] @ (np.sqrt(model.ue/model.em['omega'])*packet)).reshape(first.shape)
        leg_error = float(np.linalg.norm(first-independent_leg))
        primitive = full_coordinate_light_operators(model.law, point['phi'], point['links'], a, first)
        j = primitive['J']  # Actual cached light Hamiltonian coefficient is ONE.
        pullback_j = ul.conj().T @ j @ ul
        operator_error = float(np.linalg.norm(moved.fibre._rotate(j)-model.fibre._rotate(pullback_j)))
        for name, source in sources.items():
            moving_source = moving_sources[name]
            s = ul.conj().T @ source['full108_plus_coefficients'] @ ur
            t = ur.conj().T @ source['full108_minus_coefficients'] @ ul
            pulled_source = source_from_coefficients(model.fibre, s, t)
            moving = current_graph(moved.fibre, j, moving_source, neutral)
            independent = current_graph(model.fibre, pullback_j, pulled_source, neutral)
            frozen = current_graph(model.fibre, j, source, neutral)
            # Deliberately keep the ORIGINAL coefficient J in moving coordinates:
            # its physical representative is UL J UL†, not the actual J.
            wrong_frozen_j = current_graph(moved.fibre, ul @ j @ ul.conj().T,
                                          moving_source, neutral)
            source_error = float(np.linalg.norm(moving_source['source_matrix']-pulled_source['source_matrix']))
            response_error = float(np.linalg.norm(moving['response']-independent['response']))
            relative_error = float(np.linalg.norm(moving['relative_value']-independent['relative_value']))
            relative_map_error = float(np.linalg.norm(moving['relative_map']-independent['relative_map']))
            outside_error = abs(moving['unrepresented_PH_norm_squared']-independent['unrepresented_PH_norm_squared'])
            norm_error = abs(moving['full_norm_squared']-independent['full_norm_squared'])
            comparisons = ((source_error, np.linalg.norm(pulled_source['source_matrix'])),
                (response_error, np.linalg.norm(independent['response'])),
                (relative_error, np.linalg.norm(independent['relative_value'])),
                (relative_map_error, np.linalg.norm(independent['relative_map'])),
                (norm_error, abs(independent['full_norm_squared'])),
                (outside_error, abs(independent['unrepresented_PH_norm_squared'])),
                (operator_error, np.linalg.norm(model.fibre._rotate(pullback_j))),
                (leg_error, np.linalg.norm(independent_leg)))
            # Relative checks matter even for the tiny SOFT restriction; no floor,
            # clipping, renormalization or absolute tolerance can hide its signal.
            if any(error > 1e-10*scale for error, scale in comparisons):
                raise AssertionError('Actual moving current/SOURCE provider differs from independent pullback')
            rows.append(dict(packet=packet_name, profile=name, photon_leg_error=leg_error,
                source_pullback_error=source_error, current_pullback_response_error=response_error,
                relative_pullback_error=relative_error, full_Wick_pullback_error=norm_error,
                current_pullback_operator_error=operator_error, relative_pullback_map_error=relative_map_error,
                unrepresented_PH_pullback_error=outside_error,
                full_norm_squared=moving['full_norm_squared'], projected_norm_squared=moving['projected_norm_squared'],
                unrepresented_PH_norm_squared=moving['unrepresented_PH_norm_squared'],
                relative_norm_squared=moving['relative_norm_squared'],
                neutral_RIGHT_outside_norm_squared=moving['neutral_RIGHT_outside_norm_squared'],
                neutral_RIGHT_identity_error=moving['full_neutral_RIGHT_identity_error'],
                wrong_project_before_current_lost_norm_squared=moving['wrong_project_before_current_lost_norm_squared'],
                wrong_recaptured_first_order_projected_substitute_error=moving['wrong_recaptured_substitute_error'],
                wrong_frozen_matter_response_error=float(np.linalg.norm(moving['response']-frozen['response'])),
                wrong_frozen_current_response_error=float(np.linalg.norm(moving['response']-wrong_frozen_j['response'])),
                wrong_frozen_matter_full_norm_squared=frozen['full_norm_squared']))
    return dict(scope='POINTWISE off-mean local-frozen interaction frame; not whole-field L2, U2pi, or Born',
        photon_templates='literal gi/gfwd/gout centre phases/time6; ALL52 creator coefficients',
        photon_template_error=packet_error, point_neutral_norm_squared=float(np.vdot(neutral,neutral).real),
        filtered_neutral_default_difference=float(np.linalg.norm(neutral-model.fibre.raw_neutral)),
        physical_current='first=(em.legs@packet52)/sqrt2; lightH coefficient1; no heavyU1/Ref EM onsite',
        relative_current='full physical [dGamma(J),B]: plus J S, minus -T J; actual neutralRIGHT retained',
        wrong_recaptured_control='explicit first-order PROJECTED substitute; no state reset/eigh/instantground',
        ancestor='2.261381336887e-7 full / 2.216987466982e-7 outsidePH was POINTWISE, not whole-field cause',
        rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch
    from prepared_neutral_moving_matter_frame import construct_point_frame
    start = time.perf_counter()
    model, data = load_shared_native_cache(args.cache, args.manifest)
    frame = construct_point_frame(model, data)
    result = moving_current_control(model, frame['model'], data, frame['sample'],
                                    frame['unitaries']['light'], frame['unitaries']['reference'])
    result['bindings'] = dict(cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result['source_bindings'] = loaded_source_epoch()
    result['point_frame'] = dict(local_step=frame['local_step'], pulse=frame['pulse'],
        normal_point_norm=float(np.linalg.norm(frame['sample']['q'])),
        coexact_point_norm=float(np.linalg.norm(frame['sample']['em'][:52])),
        theta=frame['sample']['theta'].tolist())
    result['wall_seconds'] = time.perf_counter()-start
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
