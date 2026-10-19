"""NEW physical current/contact response and fixed-step larger-domain consumer.
Original actual1 sources/capsule remain immutable. Pure model helpers only.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh
from rich_extensive_photon_20261014_model import matter, line_integrals, molecular_response, polarizations, exact_consumer


def plain(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    raise TypeError(type(obj).__name__)


def physical_modes(mat, cfg, q, dw, bulk=True, contact_scale=1., current_scale=1., normalization_scale=1.):
    phases = [0., np.pi/2, np.pi, 3*np.pi/2] if bulk else [cfg['standing_phase']]
    contact = np.zeros((2, 2)); couplings = []; gaps = []
    for phase in phases:
        beta = line_integrals(mat, q, phase) * cfg['charge'] * np.sqrt(normalization_scale / (cfg['epsilon'] * mat['volume']))
        own = molecular_response(mat, cfg, beta, dw)
        contact += own['contact'] / len(phases)
        couplings.append(current_scale * own['transition'].imag * np.sqrt(2 * own['gaps'])[None, :] / np.sqrt(len(phases)))
        gaps.extend(own['gaps'])
    coupling = np.concatenate(couplings, axis=1); gaps = np.array(gaps); contact *= contact_scale
    kernel = np.diag(np.r_[np.zeros(2), gaps ** 2]); kernel[:2, :2] = contact
    kernel[:2, 2:] = coupling; kernel[2:, :2] = coupling.T
    squared, vectors = eigh(kernel)
    static = contact - (coupling / gaps[None, :] ** 2) @ coupling.T
    result = dict(q=np.asarray(q), polarizations=polarizations(q), stable=bool(squared[0] > 1.e-12),
                  minimum_squared_frequency=squared[0], static_tensor=static, contact_tensor=contact,
                  retained_correlated_transitions=len(gaps), modes=[])
    if not result['stable']:
        # Negative eigenvalues have neither stable frequencies nor vacuum
        # residues. Never report max(lambda,0) as a massless photon.
        result['unstable_squared_frequencies'] = squared[squared < -1.e-12]
        return result
    frequencies = np.sqrt(squared); amplitude = vectors[:2] / np.sqrt(2 * frequencies)[None, :]
    paramagnetic = -coupling @ vectors[2:] / np.sqrt(2 * frequencies)[None, :]
    diamagnetic = -contact @ vectors[:2] / np.sqrt(2 * frequencies)[None, :]
    physical_current = paramagnetic + diamagnetic
    result['current_equation_defect'] = float(np.max(abs(physical_current + frequencies[None, :] ** 2 * amplitude)))
    result['covariance'] = amplitude @ amplitude.T
    for index, frequency in enumerate(frequencies):
        if np.sum(amplitude[:, index] ** 2) > 1.e-7:
            result['modes'].append(dict(energy=frequency, coordinate_residue=amplitude[:, index] ** 2,
                physical_current_weight=physical_current[:, index] ** 2,
                paramagnetic_amplitude=paramagnetic[:, index], contact_current_amplitude=diamagnetic[:, index],
                field_projector=np.outer(vectors[:2, index], vectors[:2, index])))
    result['reaction'] = []
    for time in np.linspace(0, cfg['reaction_time'], cfg['reaction_steps']):
        q_expectation = cfg['field_kick'] * (vectors[:2] * (np.sin(frequencies * time) / frequencies)[None, :]) @ vectors[0]
        j_expectation = cfg['field_kick'] * (-(contact @ vectors[:2] + coupling @ vectors[2:]) * (np.sin(frequencies*time)/frequencies)[None, :]) @ vectors[0]
        result['reaction'].append(dict(time=time, Q=q_expectation, physical_current=j_expectation))
    return result


def main():
    inp, out = map(Path, sys.argv[1:3]); request = json.loads(inp.read_text()); root = inp.parent
    original = root / request['original_result']; data = original.read_bytes()
    if hashlib.sha256(data).hexdigest() != request['original_result_sha256']:
        raise ValueError('Actual1 own-state input hash changed')
    first = json.loads(data); cfg = json.loads((root / request['physical_input']).read_text()); mat = matter(cfg)
    dw = np.array(first['bulk']['feedback']['dw'])
    result = dict(schema=1, original_own_state=original.name, original_sha256=request['original_result_sha256'],
                  correction='Actual1 collective mode current_weight was a default zero for an unrequested current argument, NOT a computed physical zero. These explicit current/contact modes supersede that display field. Negative squared modes have no stable residues.',
                  bulk_current_modes=[], finite_gaussian_current_comparison=[])
    for point in first['bulk']['dispersion']:
        q = np.array(point['q']); own = physical_modes(mat, cfg, q, dw)
        result['bulk_current_modes'].append(dict(returned=own,
            omitted_contact=physical_modes(mat, cfg, q, dw, contact_scale=0.),
            doubled_current=physical_modes(mat, cfg, q, dw, current_scale=2.),
            wrong_missing_inverse_sqrt_V=physical_modes(mat, cfg, q, dw, normalization_scale=mat['volume'])))
    for point in first['exact']:
        finite_dw = np.array(point['finite_gaussian_feedback']['dw'])
        gaussian = physical_modes(mat, cfg, np.array(point['q']), finite_dw, bulk=False)
        exact_current = np.array([x['current'] for x in point['reaction']]); approx_current = np.array([x['physical_current'] for x in gaussian['reaction']])
        result['finite_gaussian_current_comparison'].append(dict(q=point['q'], gaussian=gaussian,
            exact_first_two_poles=point['bright_groups'][:2], exact_covariance=point['covariance'],
            current_relative_L2_error=float(np.linalg.norm(exact_current-approx_current)/np.linalg.norm(exact_current)),
            exact_peak_current=np.max(abs(exact_current), axis=0), gaussian_peak_current=np.max(abs(approx_current), axis=0),
            exact_double_occupancy=point['baseline_double_occupancy'], gaussian_double_occupancy=point['finite_gaussian_returned']['molecular_double_occupancy']))
    # New discriminating domain consumer: enlarge Q wall at IDENTICAL finite
    # difference step. This isolates the domain effect absent from actual1's
    # fixed-point-count wall contrast (which also changed discretization).
    refined = dict(cfg); refined['field_points'] = request['field_points']; refined['field_wall'] = request['field_wall']
    q = np.array(request['q']); print(json.dumps(dict(stage='fixed_step_larger_domain', q=q.tolist(), field_points=refined['field_points'], wall=refined['field_wall'])), flush=True)
    exact, states = exact_consumer(mat, refined, q)
    result['fixed_step_domain'] = exact
    result['fixed_step_domain']['baseline_step'] = 2*cfg['field_wall']/(cfg['field_points']+1)
    result['fixed_step_domain']['new_step'] = 2*refined['field_wall']/(refined['field_points']+1)
    # Per-polarization spectral moments, complete eigenbasis, no selected pole.
    for exact_point in first['exact'] + [exact]:
        energies = np.array(exact_point['gaps']); coordinate = np.array(exact_point['coordinate_weights']); current=np.array(exact_point['current_weights'])
        moments = dict(q=exact_point['q'], covariance_sum=coordinate.sum(axis=1), current_equal_time_sum=current.sum(axis=1),
                       first_moment=(coordinate*energies[None,:]).sum(axis=1))
        result.setdefault('exact_spectral_moments', []).append(moments)
    np.savez_compressed(out.with_suffix('.npz'), fixed_step_energies=states['energies'], field_probability=states['field_probability'])
    out.write_text(json.dumps(result, default=plain, indent=2)+'\n')
    print(json.dumps(dict(stage='complete', dimension=exact['dimension'], domain_first_gaps=exact['gaps'][:3],
                         current_errors=[x['current_relative_L2_error'] for x in result['finite_gaussian_current_comparison']])), flush=True)


if __name__ == '__main__':
    main()
