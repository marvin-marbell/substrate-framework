"""Physical native-charge receiver on exact NEW reviewer trajectories.

The first consumer's labels[0] is a multiplet ID, NOT site charge. Preserve that
adverse output, reject its charge interpretation, and consume its exact states
here without rerunning any propagation or author solve/evolve/observe routine.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse
from rich_frame_field_join_20261015_model import build, law
from rich_review_wave23_20261015_consume import norm, save, all_m, physical_receiver_setup, charge_conjugation

ROOT = Path(__file__).resolve().parent


def charge_labels(sector):
    spin = sector['spin']
    physical = np.asarray(sector['native']['charge3_site0'])
    charge = np.empty(len(spin['j']), int)
    for multiplet in spin['multiplets']:
        charge[multiplet['start']:multiplet['stop']] = multiplet['charge3']
    u = spin['U']
    matrix = u.conj().T@sparse.diags(physical)@u
    error = float(sparse.linalg.norm(matrix-sparse.diags(charge)))
    density_error = float(sparse.linalg.norm(matrix/3-sector['op']['rho0']))
    resolution = sum((u.conj().T@sparse.diags((physical==q).astype(float))@u for q in sorted(set(physical))), sparse.csr_matrix(matrix.shape))
    if error > 1e-9 or density_error > 1e-9:
        raise ValueError('Physical native site-charge PVM does not match spin multiplet charge3 metadata')
    return charge, dict(physical_charge3_labels=sorted(set(charge.tolist())), native_PVM_vs_multiplet_charge3_error=error, physical_rho0_equals_charge3_over3_error=density_error, native_PVM_resolution_error=float(sparse.linalg.norm(resolution-sparse.eye(len(charge)))), wrong_multiplet_ID_is_not_charge_count=int(np.count_nonzero(charge != np.array([label[0] for label in spin['labels']]))))


def joint_probabilities(model, name, projector, state, charges, effects, radial):
    sector = model['sectors'][name]
    spin = sector['spin']
    x = (projector@state).reshape(len(spin['j']), 280, model['radial'].count)
    x = (model['rqU']@x.transpose(1, 0, 2).reshape(280, -1)).reshape(280, len(spin['j']), -1).transpose(1, 0, 2).reshape(len(spin['j']), 10, 28, model['radial'].count)
    labels = sorted(set(charges.tolist()))
    probabilities = np.zeros((len(labels), 2, 2, 2))
    for iq, charge in enumerate(labels):
        for side in range(2):
            density = np.zeros((28, 28), complex)
            for j in set(spin['j']):
                selected = np.flatnonzero((charges==charge)&(spin['j']==j))
                if len(selected):
                    block = x[selected]
                    moved = np.einsum('iren,mn->irem', block, radial[float(j)][side], optimize=True)
                    density += np.einsum('iren,irfn->ef', block.conj(), moved, optimize=True)
            for (te, tm), effect in effects.items():
                probabilities[iq, side, te, tm] = float(np.sum(density*effect).real)
    return dict(site_charge3_labels=labels, probabilities=probabilities, total=float(probabilities.sum()), minimum=float(probabilities.min()), norm_squared=norm(state)**2)


def form_norm_bound(operator):
    # Frobenius factor bounds and triangle inequality, not a fitted stiffness.
    bound = 0.
    for rows, cols, matrix, angle, radius in operator.blocks:
        bound += norm(matrix.data)*norm(angle)*norm(radius)
    if operator.field is not None:
        bound += max(float(np.linalg.norm(x, 2)) for x in operator.field.values())
    return bound


def main():
    inp, output = map(Path, sys.argv[1:3])
    p = json.loads(inp.read_text())
    started = time.time()
    for key in ['state', 'result']:
        path = ROOT/p[key+'_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != p[key+'_sha256']:
            raise ValueError('Exact reviewer '+key+' changed')
    states = np.load(ROOT/p['state_file'])
    metadata = json.loads((ROOT/p['result_file']).read_text())
    model = build(p)
    effects, radial, diagnostics = physical_receiver_setup(model, p['matrix_amplitude_receiver_threshold'])
    nodes, weights = np.polynomial.legendre.leggauss(320)
    threshold = p['matrix_amplitude_receiver_threshold']
    coordinates = (nodes+1)*threshold/2
    weights = weights*threshold/2*coordinates**3
    radial_errors = []
    for j in radial:
        values = model['radial'].values(j, coordinates)
        refined = values.T@(weights[:, None]*values)
        radial_errors.append(norm(refined-radial[j][0]))
        radial[j] = [refined, np.eye(model['radial'].count)-refined]
    diagnostics['radial_interval_160_vs320_max_error'] = max(radial_errors)
    diagnostics['minimum_radial_effect_eigenvalue'] = min(float(np.linalg.eigvalsh(e)[0]) for pair in radial.values() for e in pair)
    result = dict(kind='CORRECTED_PHYSICAL_NATIVE_CHARGE_JOINT_RECEIVER_ON_NEW_EXACT_REVIEWER_STATES', rejected_observable='actual1 receiver treated spin labels[0] (multiplet ID) as charge3. Its time/source/current/C/geometry data are separate and unchanged; its site-charge interpretation is rejected, not silently repaired.', state_file=p['state_file'], state_sha256=p['state_sha256'], underlying_receiver=diagnostics, events={})
    for name in ['lepton', 'antilepton']:
        vm = all_m(model, name, .5)
        sector = model['sectors'][name]
        charges, charge_diagnostics = charge_labels(sector)
        # Keep the exact primary readouts; do not repeat valid propagation or costly current actions.
        bounds = {key:float(np.sqrt(sparse.linalg.norm(matrix, 1)*sparse.linalg.norm(matrix, np.inf))) for key, matrix in [('LEFT_z', sector['op']['S'][2]), ('colour_loop_B', sector['op']['B']), ('rho0', sector['op']['rho0'])]}
        bounds['matrix_r2'] = max(float(np.linalg.norm(model['radial'].operators(float(j))['r2'], 2)) for j in set(sector['spin']['j']))
        for axis in [2, 5]:
            bounds['current_'+str(axis)] = form_norm_bound(law(model, name, derivative=(axis,)))
        event = dict(charge_PVM=charge_diagnostics, dimension=vm.shape[1], measurements={}, observable_operator_norm_upper_bounds=bounds)
        for tag in ['seed', 'kicked', 'full_no_kick', 'full_kicked', 'off_no_kick', 'off_kicked']:
            state = states[name+'_'+tag]
            vector_error = 0. if tag in ['seed', 'kicked'] else metadata['events'][name]['trajectories'][tag]['propagation']['vector_error_bound']
            source_values = metadata['events'][name]['t0_before'] if tag == 'seed' else metadata['events'][name]['t0_after'] if tag == 'kicked' else metadata['events'][name]['trajectories'][tag]['observables']
            values = {key:source_values[key] for key in bounds}
            uncertainties = {key:bound*(2*norm(state)*vector_error+vector_error**2) for key, bound in bounds.items()}
            event['measurements'][tag] = dict(values=values, conditional_numerical_expectation_error_bounds=uncertainties, propagation_vector_error=vector_error, receiver=joint_probabilities(model, name, vm, state, charges, effects, radial))
        measurements = event['measurements']
        event['t0_matter_and_radial_changes'] = {key:measurements['kicked']['values'][key]-measurements['seed']['values'][key] for key in ['LEFT_z', 'colour_loop_B', 'rho0', 'matrix_r2']}
        event['full_intervention'] = {key:dict(value=measurements['full_kicked']['values'][key]-measurements['full_no_kick']['values'][key], conditional_numerical_absolute_error_bound=measurements['full_kicked']['conditional_numerical_expectation_error_bounds'][key]+measurements['full_no_kick']['conditional_numerical_expectation_error_bounds'][key]) for key in bounds}
        event['proper_complete_off_intervention'] = {key:dict(value=measurements['off_kicked']['values'][key]-measurements['off_no_kick']['values'][key], conditional_numerical_absolute_error_bound=measurements['off_kicked']['conditional_numerical_expectation_error_bounds'][key]+measurements['off_no_kick']['conditional_numerical_expectation_error_bounds'][key]) for key in bounds}
        for prefix in ['full', 'off']:
            delta = measurements[prefix+'_kicked']['receiver']['probabilities']-measurements[prefix+'_no_kick']['receiver']['probabilities']
            event[prefix+'_receiver_intervention'] = dict(joint_TV=float(np.abs(delta).sum()/2), physical_charge_and_matrix_amplitude_marginal_TV=float(np.abs(delta.sum(axis=(2, 3))).sum()/2), physical_charge_only_marginal_TV=float(np.abs(delta.sum(axis=(1, 2, 3))).sum()/2), interpretation='Field joint TV need not vanish off: field-only preparation changes field outcomes even without matter feedback. Only matter/radial marginal is the off-causal discriminator.')
        result['events'][name] = event
        save(output, result)
        print(name, 'physical_charge_receiver_complete', flush=True)
    c = charge_conjugation(model, 'lepton')
    vl = all_m(model, 'lepton', .5)
    va = all_m(model, 'antilepton', .5)
    phase = complex(*metadata['C']['phase'])
    pairs = []
    for tag in ['seed', 'kicked', 'full_no_kick', 'full_kicked', 'off_no_kick', 'off_kicked']:
        left = vl@states['lepton_'+tag]
        right = va@states['antilepton_'+tag]
        state_defect = norm(c@left-phase*right)
        a = result['events']['lepton']['measurements'][tag]['receiver']
        b = result['events']['antilepton']['measurements'][tag]['receiver']
        mirrored = np.array([b['probabilities'][b['site_charge3_labels'].index(-q), :, ::-1, ::-1] for q in a['site_charge3_labels']])
        pairs.append(dict(state=tag, unitary_C_source_and_trajectory_error=state_defect, physical_charge_field_sign_mirrored_receiver_L1_error=float(np.abs(a['probabilities']-mirrored).sum())))
    result['C_source_and_physical_receiver_trajectories'] = pairs
    result['elapsed_seconds'] = time.time()-started
    result['scope'] = 'Only physical receiver correction and error-controlled finite-model observables on exact new Wave23 states, not propagation replay, author solve, smaller cut or future science expansion.'
    save(output, result)
    print(json.dumps(dict(natural_completion=True, seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
