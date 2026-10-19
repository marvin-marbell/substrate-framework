#!/usr/bin/env python3
"""Corrective NEW charged calculation: complete orthonormal finite eigenframes.

The first reviewer eigsh run exposed nonorthogonal degenerate charge eigenvectors
and impossible captured weights > source norms. Preserve that receipt; here use
full Hermitian eigh, complete source matrices and full spectral sum. No threshold
or immutable author artifact is edited. Original and anchored observables remain
distinct and their patch choice is measured, not silently repaired in publication.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import eigh
import rich_electron_20261014_hubbard_charged as electron
import rich_photon_20261014_loop as photon
from rich_review_exec_20261014_consumer import creation_map, large_gauge, spin_squared, data, chi


def run(request, output):
    ng = request['flux_grid']; parameters = photon.Parameters()
    sectors = {}; errors = {}
    for n in (1, 2, 3):
        h, basis, index = electron.coupled_matrix(n, parameters, ng)
        energies, states = eigh(h.toarray(), driver='evd')
        sectors[n] = energies, states, basis, index
        errors[n] = {'dimension': len(energies),
                     'eigenframe_orthonormal_error': float(np.linalg.norm(states.conj().T@states-np.eye(len(energies)))),
                     'eigen_residual_max': float(np.max(np.linalg.norm(h@states-states*energies, axis=0)))}
    energy = sectors[2][0][0]; neutral = sectors[2][1][:, 0].reshape(ng, -1)
    frequencies = [.0, .4, 1.1, 2.2]; eta = request['z'][1]
    rows = []; raw_spectra = {}
    for k in (0., np.pi/2):
        source_families = {}
        for anchored in (False, True):
            contributions = {}; source_sums = np.zeros((2, 2), complex); spin_values = []
            for n, creation in ((3, True), (1, False)):
                energies, states, basis, _ = sectors[n]; sources = []
                for spin in (0, 1):
                    op, pieces = creation_map(sectors[2][2], basis, spin, k, creation)
                    sign = 1 if creation else -1
                    vector = []
                    for grid, neutral_part in enumerate(neutral):
                        operator = sum(np.exp(sign*1j*site*2*np.pi*grid/ng/4)*piece for site, piece in enumerate(pieces)) if anchored else op
                        vector.append(operator@neutral_part)
                    vector = np.concatenate(vector); sources.append(vector)
                    shaped = vector.reshape(ng, -1)
                    s2 = spin_squared(basis)
                    spin_values.append(float(np.einsum('qi,ij,qj->', shaped.conj(), s2, shaped).real/np.vdot(vector, vector).real))
                source = np.column_stack(sources); transitions = states.conj().T@source
                norm_matrix = source.conj().T@source
                weight_matrix = transitions.conj().T@transitions
                # Addition: <c_s c_t^dag>; removal transposes <c_s^dag c_t>.
                spectral = np.einsum('ns,nt->nst', transitions.conj(), transitions)
                if not creation:
                    spectral = spectral.transpose(0, 2, 1)
                    norm_matrix = norm_matrix.T; weight_matrix = weight_matrix.T
                source_sums += norm_matrix
                poles = energies-energy if creation else energy-energies
                green = np.array([np.einsum('nst,n->st', spectral, 1/(omega+1j*eta-poles)) for omega in frequencies])
                label = 'addition' if creation else 'removal'
                contributions[label] = {'pole_min': float(poles.min()), 'pole_max': float(poles.max()),
                    'source_norm_matrix': data(norm_matrix), 'complete_residue_sum_matrix': data(weight_matrix),
                    'complete_residue_source_error': float(np.linalg.norm(weight_matrix-norm_matrix)),
                    'G_spin_matrix': data(green)}
                raw_spectra[f'k{k}_anchored{anchored}_{label}_poles'] = poles
                raw_spectra[f'k{k}_anchored{anchored}_{label}_spectral_spin_matrix'] = spectral
            add_g = np.array(contributions['addition']['G_spin_matrix']['real'])+1j*np.array(contributions['addition']['G_spin_matrix']['imag'])
            rem_g = np.array(contributions['removal']['G_spin_matrix']['real'])+1j*np.array(contributions['removal']['G_spin_matrix']['imag'])
            total = add_g+rem_g
            source_families['anchored' if anchored else 'original'] = {'contributions': contributions,
                'G_spin_matrix': data(total), 'CAR_source_sum_matrix': data(source_sums),
                'CAR_sum_error': float(np.linalg.norm(source_sums-np.eye(2))),
                'up_down_G_difference_max': float(np.max(np.abs(total[:, 0, 0]-total[:, 1, 1]))),
                'offdiagonal_spin_G_max': float(np.max(np.abs(total[:, 0, 1]))),
                'actual_source_S2': spin_values}
        original = source_families['original']['G_spin_matrix']; anchored = source_families['anchored']['G_spin_matrix']
        differences = np.abs((np.array(original['real'])+1j*np.array(original['imag']))-(np.array(anchored['real'])+1j*np.array(anchored['imag'])))
        rows.append({'k': k, 'sources': source_families, 'source_choice_spin_G_frobenius_difference': np.linalg.norm(differences, axis=(1, 2)).tolist()})
    _, _, _, transfers = electron.matter_matrices(2, parameters)
    jq = np.zeros((ng*len(sectors[2][2]),)*2, complex)
    for grid in range(ng):
        phase = np.exp(1j*2*np.pi*grid/ng/4); d = len(sectors[2][2])
        jq[grid*d:(grid+1)*d, grid*d:(grid+1)*d] = -.25j*np.sum(phase*transfers-phase.conjugate()*transfers.conj().transpose(0, 2, 1), axis=0)
    en, un, _, _ = sectors[2]
    current_response = [chi(en, un, un[:, 0], jq, jq, omega+1j*eta) for omega in frequencies]
    mu_plus = float(sectors[3][0][0]-energy); mu_minus = float(energy-sectors[1][0][0])
    result = {'kind': 'corrective_complete_finite_charged_spectral_consumer', 'input': request,
        'correction': 'Full scipy.linalg.eigh EVD orthonormal frames replace nonorthogonal selected ARPACK columns. No global QR mixing distinct energies; all states kept.',
        'runtime': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'frames': errors, 'mu_plus': mu_plus, 'mu_minus': mu_minus,
        'canonical_gap': mu_plus-mu_minus, 'midgap_mu': (mu_plus+mu_minus)/2,
        'frequencies': frequencies, 'eta': eta, 'rows': rows,
        'same_N2_state_flux_current_retarded': data(current_response),
        'retarded_convention': 'G(z)=sum w_add/(z-(E3-E2))+sum w_rem/(z-(E2-E1)). At chemical potential mu substitute z+mu.',
        'scope': 'Complete spectra of finite 32-grid one-global-flux law; anchored source has compensating large-gauge patch. Not local-Gauss photon field or relativistic positron. Source S2 is primitive SU2; no spatial rotation derivation.',
        'supersedes': 'Only raw charged Lehmann weights/current truncation/G in rich_review_exec_20261014_result.json; energies, other three physical branches and exact source norms were not invalidated.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'), **raw_spectra)
    print(json.dumps({'output': str(output), 'canonical_gap': result['canonical_gap'],
        'max_CAR_error': max(x['sources'][s]['CAR_sum_error'] for x in rows for s in ('original', 'anchored')),
        'max_complete_residue_error': max(x['sources'][s]['contributions'][c]['complete_residue_source_error'] for x in rows for s in ('original', 'anchored') for c in ('addition', 'removal'))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(json.loads(args.input.read_text()), args.output)
