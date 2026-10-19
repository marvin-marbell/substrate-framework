#!/usr/bin/env python3
"""Finite compact-photon Lehmann response of the actual Gauss Pauli/rotor square.

Run this file for JSON evidence; no Gaussian sea or tensor-factor ground is used.
Energy units have hbar=1. The source term is -j(t) B with dimensionless B=sin(phi).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

from code_phase import CoupledSquare


@dataclass(frozen=True)
class LehmannResponse:
    """Positive inelastic spectral measure, including every nonzero-energy pole."""

    energies: np.ndarray
    weights: np.ndarray

    def susceptibility(self, frequency, broadening=0.02):
        """delta<B>=chi*j for H_source=-jB; positive absorptive Im chi at w>0."""
        if broadening <= 0:
            raise ValueError('Retarded broadening must be positive')
        z = np.asarray(frequency, dtype=complex)[..., None] + 1j * broadening
        return np.sum(2 * self.energies * self.weights /
                      (self.energies ** 2 - z ** 2), axis=-1)

    def time_kernel(self, times):
        """chi(t)=2 theta(t) sum_a w_a sin(epsilon_a t), units dimensionless."""
        times = np.asarray(times, dtype=float)
        return np.where(times >= 0, 2 * np.sum(
            self.weights * np.sin(times[..., None] * self.energies), axis=-1), 0)

    def rows(self, threshold=1e-10):
        total = float(self.weights.sum())
        return [{'energy': float(e), 'weight': float(w),
                 'inelastic_fraction': float(w / total) if total else 0.0}
                for e, w in zip(self.energies, self.weights) if w > threshold]


def loop_operators(model):
    """Construct exactly the oriented W used by CoupledSquare.operators()."""
    W = np.zeros((model.dimension, model.dimension), dtype=complex)
    shifts = model.shifts(model.cycle, closing=True)
    for column, (q, fields) in enumerate(model.raw):
        shifted = model.shift(fields, shifts)
        if shifted is not None:
            model.insert(W, column, q, shifted, 1)
    B = (W - W.conj().T) / (2j)
    # [n,W]=W; Gauss allows affine charged electric configurations as well.
    n = np.diag([sum(s * e for s, e in zip(shifts, fields)) / 4
                 for _, fields in model.raw]).astype(complex)
    return W, B, n


def spectral_measure(H, O, energies, vectors):
    ground = vectors[:, 0]
    gaps = energies - energies[0]
    amplitudes = vectors.conj().T @ (O @ ground)
    weights = np.abs(amplitudes) ** 2
    zero = gaps <= 1e-10
    positive = ~zero
    response = LehmannResponse(gaps[positive], weights[positive])
    expectation = np.vdot(ground, O @ ground)
    variance = float(np.vdot(O @ ground, O @ ground).real - abs(expectation) ** 2)
    elastic_connected = float(weights[zero].sum() - abs(expectation) ** 2)
    commutator = H @ O - O @ H
    double = O @ commutator - commutator @ O
    moment = float(np.dot(response.energies, response.weights))
    double_value = float(np.vdot(ground, double @ ground).real / 2)
    reports = {
        'expectation_real': float(expectation.real),
        'expectation_imag': float(expectation.imag),
        'connected_equal_time': variance,
        'zero_energy_connected_weight': elastic_connected,
        'inelastic_weight': float(response.weights.sum()),
        'equal_time_sum_rule_residual': abs(variance - elastic_connected - response.weights.sum()),
        'energy_weighted_sum': moment,
        'half_double_commutator': double_value,
        'energy_sum_rule_residual': abs(moment - double_value),
        'static_susceptibility': float(np.sum(2 * response.weights / response.energies)),
        'minimum_weight': float(response.weights.min()) if len(response.weights) else 0.0,
        'poles': response.rows(),
    }
    if reports['equal_time_sum_rule_residual'] > 2e-9 or reports['energy_sum_rule_residual'] > 2e-8:
        raise AssertionError('Actual finite spectral sum rule failed')
    return response, reports


def gaussian_prediction(electric, magnetic):
    variance = math.sqrt(electric / magnetic)
    frequency = 2 * math.sqrt(electric * magnetic)
    # <m|(a+a^dagger)^4|m>=3(2m^2+2m+1), phi=sqrt(variance)(a+a^dagger).
    quartic_shifts = [-magnetic / 24 * 3 * (2 * m * m + 2 * m + 1) * variance ** 2
                     for m in (0, 1)]
    gap_correction = quartic_shifts[1] - quartic_shifts[0]
    # Exact sin(phi) on a harmonic oscillator is distinct from its linearization.
    odd = np.arange(1, 16, 2)
    weights = np.array([math.exp(-variance) * variance ** int(m) / math.factorial(int(m))
                        for m in odd])
    return {
        'omega': frequency, 'angle_variance': variance,
        'linear_B_single_pole_weight': variance,
        'quartic_level_shifts_0_1': quartic_shifts,
        'quartic_gap_correction': gap_correction,
        'quartic_predicted_gap': frequency + gap_correction,
        'harmonic_sine_poles': [{'energy': float(m * frequency), 'weight': float(w)}
                                for m, w in zip(odd, weights)],
        'harmonic_sine_static_susceptibility': float(np.sum(2 * weights / (odd * frequency))),
    }


def solve(cutoff, magnetic=100 / 9, electric=0.09, hopping=0.6, improved=0.19):
    model = CoupledSquare(cutoff=cutoff, magnetic=magnetic, electric=electric,
                          delta=3 * magnetic, hopping=hopping, improved=improved)
    H, S, cosine, matter = model.operators()
    W, B, n = loop_operators(model)
    hermitian = float(np.linalg.norm(H - H.conj().T))
    if hermitian > 1e-10 or np.linalg.norm(cosine - (W + W.conj().T) / 2) > 1e-10:
        raise AssertionError('Hamiltonian/operator construction mismatch')
    energies, vectors = np.linalg.eigh(H)  # ALL raw Gauss states, including both code sectors.
    ground = vectors[:, 0]
    response, B_report = spectral_measure(H, B, energies, vectors)
    _, n_report = spectral_measure(H, n, energies, vectors)
    boundary = np.array([any(abs(e) == cutoff for e in fields) for _, fields in model.raw])
    active = response.weights > 1e-10
    gaps = energies - energies[0]
    photon_gaps = response.energies[active]
    dominant = int(np.argmax(response.weights))
    # Partial trace over the ACTUAL raw Pauli/rotor embedding, not an assumed sea product.
    fields = list(dict.fromkeys(fields for _, fields in model.raw))
    field_index = {f: k for k, f in enumerate(fields)}
    amplitudes = np.zeros((16, len(fields)), dtype=complex)
    for coefficient, (q, f) in zip(ground, model.raw):
        amplitudes[q, field_index[f]] = coefficient
    density_matter = amplitudes @ amplitudes.conj().T
    occupations = {}
    for coefficient, (q, _) in zip(ground, model.raw):
        name = ','.join(map(str, model.words.occupation(q)))
        occupations[name] = occupations.get(name, 0.0) + float(abs(coefficient) ** 2)
    frequencies = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    chi = response.susceptibility(frequencies)
    report = {
        'cutoff': cutoff, 'K': magnetic, 'UE': electric, 'Delta': model.delta,
        'hopping': hopping, 'improved': improved, 'phase': 0.23,
        'raw_Gauss_dimension': model.dimension, 'ground_energy': float(energies[0]),
        'ground_degeneracy': int(np.count_nonzero(gaps <= 1e-10)),
        'smallest_overall_positive_gap': float(gaps[gaps > 1e-10].min()),
        'first_B_active_gap': float(photon_gaps.min()) if len(photon_gaps) else None,
        'dominant_B_gap': float(response.energies[dominant]),
        'dominant_B_weight_fraction': float(response.weights[dominant] / response.weights.sum()),
        'ground_S': float(np.vdot(ground, S @ ground).real),
        'ground_matter_rotor_purity': float(np.trace(density_matter @ density_matter).real),
        'ground_occupation_probabilities': occupations,
        'ground_cutoff_shell_probability': float(np.sum(np.abs(ground[boundary]) ** 2)),
        'ground_raise_loss': float(1 - np.linalg.norm(W @ ground) ** 2),
        'ground_lower_loss': float(1 - np.linalg.norm(W.conj().T @ ground) ** 2),
        'hermiticity_residual': hermitian,
        'Gauss_violations': model.constraint_violations,
        'B_response': B_report, 'loop_electric_response': n_report,
        'gaussian_alternative': gaussian_prediction(electric, magnetic),
        'retarded_samples': [{'frequency': float(f), 'real': float(c.real), 'imag': float(c.imag)}
                             for f, c in zip(frequencies, chi)],
        'causal_time_samples': response.time_kernel([-1, 0, 0.5, 1, 2]).tolist(),
        'full_spectrum': energies.tolist(),
    }
    return model, H, ground, report


def neutral_quartic_report(cutoff, electric):
    """An actual invariant neutral-background S=+ block, not the full ground claim."""
    magnetic = 1 / electric  # omega=2 fixed while weak angle variance=UE decreases.
    model = CoupledSquare(cutoff=cutoff, magnetic=magnetic, electric=electric,
                          delta=3 * magnetic, hopping=0, improved=0)
    H, _, _, _ = model.operators()
    W, B, _ = loop_operators(model)
    basis, labels = model.sector_basis(1)
    columns = [i for i, (occupation, _) in enumerate(labels) if occupation == (0, 2)]
    P = basis[:, columns]
    block = P.conj().T @ H @ P
    invariant_residual = float(np.linalg.norm(H @ P - P @ block))
    if invariant_residual > 1e-9:
        raise AssertionError('Neutral zero-transfer block is not invariant')
    energies, vectors = np.linalg.eigh(block)
    response, report = spectral_measure(block, P.conj().T @ B @ P, energies, vectors)
    dominant = int(np.argmax(response.weights))
    gap = float(response.energies[dominant])
    prediction = gaussian_prediction(electric, magnetic)
    shifts = model.shifts(model.cycle, closing=True)
    # Every neutral raw field is one integer loop; verify the kinetic coefficient.
    kinetic_residual = 0.0
    for occupation, fields in (labels[i] for i in columns):
        n = sum(s * e for s, e in zip(shifts, fields)) / 4
        kinetic_residual = max(kinetic_residual, abs(electric / 2 * sum(e * e for e in fields)
                                                   - 2 * electric * n ** 2))
    ground = P @ vectors[:, 0]
    shell = np.array([any(abs(e) == cutoff for e in fields) for _, fields in model.raw])
    return {
        'domain': 'neutral occupation=(0,2), S=+, hopping=improved=0; NOT full-ground projection',
        'cutoff': cutoff, 'UE': electric, 'K': magnetic, 'dimension': len(columns),
        'gap': gap, 'gaussian': prediction,
        'gap_minus_harmonic': gap - prediction['omega'],
        'correction_divided_by_UE': (gap - prediction['omega']) / electric,
        'quartic_prediction_error': gap - prediction['quartic_predicted_gap'],
        'quartic_scaled_remainder': (gap - prediction['quartic_predicted_gap']) /
                                    (electric * math.sqrt(electric / magnetic)),
        'ground_shell_probability': float(np.sum(np.abs(ground[shell]) ** 2)),
        'kinetic_coefficient_residual': kinetic_residual,
        'invariant_block_residual': invariant_residual,
        'B_response': report,
    }


def cutoff_comparison(previous, current):
    small, _, small_ground, small_report = previous
    large, H_large, _, large_report = current
    embedded = np.zeros(large.dimension, dtype=complex)
    for coefficient, state in zip(small_ground, small.raw):
        embedded[large.index[state]] = coefficient
    energy = small_report['ground_energy']
    residual = float(np.linalg.norm(H_large @ embedded - energy * embedded))
    old, new = small_report['B_response'], large_report['B_response']
    if large_report['ground_energy'] > energy + 2e-9:
        raise AssertionError('Nested Galerkin ground energy violated variational monotonicity')
    return {'from': small.cutoff, 'to': large.cutoff,
            'ground_energy_change': large_report['ground_energy'] - energy,
            'embedded_ground_eigen_residual': residual,
            'rigorous_large_spectrum_distance_bound': residual,
            'dominant_B_gap_change': large_report['dominant_B_gap'] - small_report['dominant_B_gap'],
            'B_inelastic_weight_change': new['inelastic_weight'] - old['inelastic_weight'],
            'B_energy_moment_change': new['energy_weighted_sum'] - old['energy_weighted_sum'],
            'B_static_susceptibility_change': new['static_susceptibility'] - old['static_susceptibility']}


def run(cutoffs=(2, 4, 6), weak_cutoff=8):
    if len(cutoffs) < 2 or any(c < 1 for c in cutoffs) or sorted(set(cutoffs)) != list(cutoffs):
        raise ValueError('Use at least two strictly increasing positive cutoffs')
    cases, comparisons, final = [], [], {}
    for hopping, improved in ((0, 0), (0.6, 0), (0.6, 0.19)):
        previous = None
        for cutoff in cutoffs:
            result = solve(cutoff, hopping=hopping, improved=improved)
            cases.append(result[3])
            if previous is not None:
                comparisons.append({'hopping': hopping, 'improved': improved,
                                    **cutoff_comparison(previous, result)})
            previous = result
        assert previous is not None  # cutoffs contains at least two entries.
        final[hopping, improved] = previous[3]
    baseline = final[0, 0]
    modifications = []
    for key in ((0.6, 0), (0.6, 0.19)):
        case = final[key]
        modifications.append({
            'hopping': key[0], 'improved': key[1],
            'dominant_B_gap_change_from_zero_transfer': case['dominant_B_gap'] - baseline['dominant_B_gap'],
            'B_weight_change_from_zero_transfer': case['B_response']['inelastic_weight'] - baseline['B_response']['inelastic_weight'],
            'B_static_change_from_zero_transfer': case['B_response']['static_susceptibility'] - baseline['B_response']['static_susceptibility'],
            'ground_matter_rotor_purity': case['ground_matter_rotor_purity'],
        })
    return {'scope': 'Finite one-plaquette Hamiltonian response, not a 3D phase or QED cross section',
            'units': 'hbar=1; K,UE,Delta,hopping,j,omega energies; phi,n,B dimensionless; chi(omega) inverse energy',
            'source': 'H_source(t)=-j(t) B; B=(W-W^dagger)/(2i), W actual Gauss loop shift',
            'cases': cases, 'cutoff_comparisons': comparisons,
            'charged_matter_spectral_modifications': modifications,
            'neutral_quartic_comparisons': [neutral_quartic_report(weak_cutoff, e)
                                           for e in (0.2, 0.1, 0.09)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[2, 4, 6])
    parser.add_argument('--weak-cutoff', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(tuple(args.cutoffs), args.weak_cutoff), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
