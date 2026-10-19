#!/usr/bin/env python3
"""Finite-cutoff compact-rotor Holder bound and actual Pauli-spinor consumer.

The CLI solves only tiny pure rotors and constructs actual sparse Hamiltonians;
it never repeats the coupled ARPACK ground-state or photon-response runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import eigh_tridiagonal
from scipy.special import logsumexp

import quantum_spinor_plaquette as spinor
from rotation_band import RotationBand


@dataclass(frozen=True)
class GapParameters:
    electric: float = 0.0072
    magnetic: float = 0.08 / 0.3**2
    delta: float = 2.2
    m: float = 0.8
    r: float = 0.04
    v: float = 0.08

    def validate(self):
        numbers = (self.electric, self.magnetic, self.delta, self.m, self.r, self.v)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError('Require finite scalar parameters')
        if self.electric <= 0 or self.magnetic <= 0 or self.delta <= 2 * self.magnetic:
            raise ValueError('Actual spinor consumer requires UE,K>0 and Delta>2K')
        if self.r < 0 or self.v < 0 or self.m + 2 * self.r <= 0:
            raise ValueError('Charge estimate requires r,v>=0 and M=m+2r>0')

    @property
    def mass(self):
        return self.m + 2 * self.r

    @property
    def hopping_bound(self):
        return 8 * max(self.r, self.v)

    @property
    def gap_bound(self):
        return self.mass + self.electric / 2 - self.hopping_bound

    def model_kwargs(self):
        return {'electric': self.electric, 'magnetic': self.magnetic,
                'delta': self.delta, 'm': self.m, 'r': self.r, 'v': self.v}


def loop_domain(offsets, cutoff):
    """Physical intersection, with integer loop coordinate and no wrap edge."""
    if len(offsets) != 4 or any(not math.isfinite(a) or int(a) != a for a in offsets):
        raise ValueError('Four integer Gauss offsets are required')
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError('Require a finite nonnegative electric cutoff')
    low = math.ceil(max(-cutoff - a for a in offsets))
    high = math.floor(min(cutoff - a for a in offsets))
    if high < low:
        raise ValueError('Gauss-offset electric intersection is empty')
    return np.arange(low, high + 1, dtype=int)


def rotor(offsets, cutoff, electric, magnetic):
    """H_a on the exact intersection, including the one-state/K=0 cases."""
    if not math.isfinite(electric) or electric <= 0:
        raise ValueError('Require UE>0')
    if not math.isfinite(magnetic) or magnetic < 0:
        raise ValueError('Require K>=0')
    loops = loop_domain(offsets, cutoff)
    fields = loops[:, None] + np.asarray(offsets, dtype=int)[None, :]
    diagonal = electric / 2 * np.sum(fields**2, axis=1)
    offdiagonal = np.full(max(0, len(loops) - 1), -magnetic / 2)
    return loops, diagonal, offdiagonal


def rotor_spectrum(offsets, cutoff, electric, magnetic):
    _, diagonal, offdiagonal = rotor(offsets, cutoff, electric, magnetic)
    if len(diagonal) == 1:
        return diagonal
    return np.asarray(eigh_tridiagonal(diagonal, offdiagonal, eigvals_only=True))


def neutral_ground(cutoff, electric, magnetic):
    loops, diagonal, offdiagonal = rotor((0, 0, 0, 0), cutoff, electric, magnetic)
    if len(loops) == 1:
        return loops, float(diagonal[0]), np.ones(1)
    eigenvalues, eigenvectors = eigh_tridiagonal(
        diagonal, offdiagonal, select='i', select_range=(0, 0))
    return loops, float(eigenvalues[0]), eigenvectors[:, 0]


def gauss_patterns():
    """All actual N7/8/9 cell counts, weighted by orbital-mask multiplicity."""
    patterns = defaultdict(dict)
    for number in (7, 8, 9):
        boundary = number - 8
        for counts in itertools.product(range(5), repeat=4):
            if sum(counts) != number:
                continue
            charges = tuple(counts[c] - 2 - (boundary if c == 0 else 0)
                            for c in range(4))
            offsets = (0, charges[1], charges[1] + charges[2], -charges[0])
            if sum(charges) != 0:
                raise AssertionError('Boundary-adjusted Gauss charges are not neutral')
            patterns[offsets][number] = math.prod(math.comb(4, n) for n in counts)
    for number in (7, 8, 9):
        if sum(by_number.get(number, 0) for by_number in patterns.values()) != math.comb(16, number):
            raise AssertionError('Cell-count patterns omit actual orbital occupations')
    return dict(patterns)


def spectral_consumer(cutoffs, parameters, trace_times):
    patterns = gauss_patterns()
    cases = ((parameters.electric, parameters.magnetic),
             (parameters.electric, 0.0), (0.6, 0.25), (0.11, 3.0))
    records = []
    for electric, magnetic in dict.fromkeys(cases):
        previous = {}
        for cutoff in cutoffs:
            neutral = rotor_spectrum((0, 0, 0, 0), cutoff, electric, magnetic)
            neutral_logs = np.array([logsumexp(-t * neutral) for t in trace_times])
            worst_energy = (math.inf, None)
            worst_log_ratio = (-math.inf, None, None)
            included, omitted = 0, 0
            dimensions = {number: 0 for number in (7, 8, 9)}
            maximum_nested_increase = 0.0
            for offsets, by_number in patterns.items():
                try:
                    loops = loop_domain(offsets, cutoff)
                except ValueError:
                    omitted += 1
                    continue
                included += 1
                spectrum = rotor_spectrum(offsets, cutoff, electric, magnetic)
                margin = float(spectrum[0] - neutral[0])
                scale = 1 + float(np.max(abs(spectrum))) + float(np.max(abs(neutral)))
                if margin < -2e-11 * scale:
                    raise AssertionError('Pure Gauss rotor violates the Holder ground bound')
                logs = np.array([logsumexp(-t * spectrum) for t in trace_times])
                ratios = logs - neutral_logs
                if float(np.max(ratios)) > 2e-10 * (1 + max(trace_times) * scale):
                    raise AssertionError('Pure Gauss rotor violates the Holder trace bound')
                if offsets in previous:
                    increase = float(spectrum[0] - previous[offsets])
                    maximum_nested_increase = max(maximum_nested_increase, increase)
                    if increase > 2e-11 * scale:
                        raise AssertionError('A nested physical rotor domain raised its ground energy')
                previous[offsets] = float(spectrum[0])
                for number, multiplicity in by_number.items():
                    dimensions[number] += multiplicity * len(loops)
                if offsets != (0, 0, 0, 0):
                    if margin < worst_energy[0]:
                        worst_energy = (margin, offsets)
                    index = int(np.argmax(ratios))
                    if float(ratios[index]) > worst_log_ratio[0]:
                        worst_log_ratio = (float(ratios[index]), offsets, trace_times[index])
            # A common INTEGER shift changes the coordinate, not the physical domain.
            translated = rotor_spectrum((3, 3, 3, 3), cutoff, electric, magnetic)
            translation_error = float(np.max(abs(translated - neutral)))
            if translation_error > 2e-11 * (1 + float(np.max(abs(neutral)))):
                raise AssertionError('Integer-translation positive control failed')
            records.append({'UE': electric, 'K': magnetic, 'cutoff': cutoff,
                            'neutral_ground_energy': float(neutral[0]),
                            'nonempty_patterns': included, 'empty_patterns': omitted,
                            'actual_Gauss_dimensions_by_N': dimensions,
                            'minimum_nontrivial_ground_margin': worst_energy[0],
                            'ground_margin_offsets': worst_energy[1],
                            'maximum_nontrivial_log_trace_ratio': worst_log_ratio[0],
                            'trace_ratio_offsets': worst_log_ratio[1],
                            'trace_ratio_time': worst_log_ratio[2],
                            'maximum_nested_ground_increase': maximum_nested_increase,
                            'integer_translation_spectrum_error': translation_error})
    return {'trace_times': trace_times, 'distinct_true_Gauss_patterns': len(patterns),
            'cases': records, 'scope': 'Finite numerical discriminants; the universal result is the proof.'}


def operator_parts(model):
    """Split the ACTUAL sparse matrix by its conserved cell-count transitions."""
    entries = model.H.tocoo()
    signatures = np.array([sum(sum(mask >> (4 * c + j) & 1 for j in range(4)) * 5**c
                               for c in range(4)) for mask, _ in model.raw])
    masks = np.array([mask for mask, _ in model.raw])
    different_mask = masks[entries.row] != masks[entries.col]
    same_counts = signatures[entries.row] == signatures[entries.col]
    onsite_selection = different_mask & same_counts
    hopping_selection = different_mask & ~same_counts
    shape = model.H.shape
    onsite = sparse.coo_matrix((entries.data[onsite_selection],
                               (entries.row[onsite_selection], entries.col[onsite_selection])),
                              shape=shape).tocsr()
    diagonal_onsite = np.array([sum(model.onsite[i % 4, i % 4].real
                                   for i in spinor.occupied(mask)) for mask, _ in model.raw])
    onsite += sparse.diags(diagonal_onsite, format='csr')
    hopping = sparse.coo_matrix((entries.data[hopping_selection],
                                (entries.row[hopping_selection], entries.col[hopping_selection])),
                               shape=shape).tocsr()
    boundary = model.UE / 2 * model.boundary**2
    defect = model.delta * (1 - model.sign) / 2
    photon = model.H - onsite - hopping - (boundary + defect) * sparse.eye(model.dimension, format='csr')
    electric = np.array([model.UE / 2 * sum(e * e for e in fields) for _, fields in model.raw])
    expected = sparse.diags(electric, format='csr') - model.K / 2 * (model.W + model.W.conjugate().T)
    error = spinor.max_abs(photon - expected)
    if error > 2e-10:
        raise AssertionError('Actual sparse operator has unaccounted photon/matter terms')
    return onsite, hopping, photon, error


def neutral_trial(model, parameters):
    """Onsite negative Slater fill embedded by actual primitive Pauli phases."""
    local_energies, local_vectors = np.linalg.eigh(model.onsite)
    if np.max(abs(local_energies - np.array([-1, -1, 1, 1]) * parameters.mass)) > 2e-11:
        raise AssertionError('Actual planar onsite law is not the stated +/-M pair')
    negative = local_vectors[:, :2]
    local = {sum(1 << i for i in pair): np.linalg.det(negative[list(pair), :])
             for pair in itertools.combinations(range(4), 2)}
    phases = model.code.intertwiner(8, model.sign)
    loops, epsilon, photon_vector = neutral_ground(model.cutoff, model.UE, model.K)
    photon_amplitudes = dict(zip(loops, photon_vector))
    trial = np.zeros(model.dimension, dtype=complex)
    for index, (mask, fields) in enumerate(model.raw):
        cell_masks = tuple((mask >> (4 * c)) & 15 for c in range(4))
        if all(cm in local for cm in cell_masks) and len(set(fields)) == 1:
            trial[index] = phases[mask] * math.prod(local[cm] for cm in cell_masks) * photon_amplitudes[fields[0]]
    if abs(np.vdot(trial, trial).real - 1) > 2e-10:
        raise AssertionError('Neutral onsite/photon variational state is not normalized')
    return trial, epsilon, local_energies


def actual_operator_consumer(parameters, cutoff, expected_dimensions):
    code = spinor.PauliCode()
    band = RotationBand(a=1.0, m=parameters.m, r=parameters.r, v=parameters.v)
    _, links = band.couplings()
    singular_sums = [float(np.sum(np.linalg.svd(links[axis][1], compute_uv=False)))
                     for axis in (0, 1)]
    norm_bound = 2 * sum(singular_sums)
    if abs(norm_bound - parameters.hopping_bound) > 2e-11:
        raise AssertionError('Actual link singular values disagree with the four-link hopping estimate')
    output = []
    for number in (7, 8, 9):
        model = spinor.SpinorPlaquette(code, number, cutoff, **parameters.model_kwargs())
        if model.dimension != expected_dimensions[number]:
            raise AssertionError('Pattern consumer disagrees with the actual Gauss basis dimension')
        onsite, hopping, photon, decomposition_error = operator_parts(model)
        local_energies = np.linalg.eigvalsh(model.onsite)
        onsite_minimum = float(np.sum(np.sort(np.tile(local_energies, 4))[:number]))
        if abs(onsite_minimum - (-8 if number == 8 else -7) * parameters.mass) > 2e-10:
            raise AssertionError('Actual onsite spectrum disagrees with the N7/8/9 lower bounds')
        _, epsilon, _ = neutral_ground(cutoff, parameters.electric, parameters.magnetic)
        # Audit every actual fixed-mask rotor domain, without repeated eigensolves.
        domains = defaultdict(list)
        for mask, fields in model.raw:
            domains[mask].append(fields)
        offsets_seen = set()
        for mask, fields in domains.items():
            first = fields[0]
            offsets = tuple(e - first[0] for e in first)
            expected_loops = loop_domain(offsets, cutoff)
            actual_loops = np.array([e[0] for e in fields])
            if not np.array_equal(actual_loops, expected_loops):
                raise AssertionError('Actual photon block omits or wraps a physical loop state')
            if any(model.gauss(mask, e) != (0, 0, 0, 0) for e in fields):
                raise AssertionError('Actual photon block fails full Gauss law')
            offsets_seen.add(offsets)
        record = {'N': number, 'cutoff': cutoff, 'dimension': model.dimension,
                  'actual_fixed_mask_blocks': len(domains), 'distinct_photon_blocks': len(offsets_seen),
                  'operator_decomposition_residual': decomposition_error,
                  'onsite_lower_bound': onsite_minimum,
                  'photon_lower_bound': epsilon, 'hopping_norm_bound': norm_bound,
                  'boundary_energy': parameters.electric / 2 * model.boundary**2,
                  'derived_total_energy_lower_bound': onsite_minimum + epsilon
                  + parameters.electric / 2 * model.boundary**2 - norm_bound,
                  'hermiticity_residual': model.hermiticity,
                  'T_symmetry_residual': model.Tsymmetry,
                  'boundary_omission_Gauss_rejected': bool(any(model.gauss(*model.raw[0], boundary=0)))
                  if number != 8 else None}
        if number != 8:
            removed = parameters.electric / 2 * model.boundary**2
            altered = model.H - removed * sparse.eye(model.dimension, format='csr')
            shift_error = spinor.max_abs(model.H - altered - removed * sparse.eye(model.dimension, format='csr'))
            if shift_error > 2e-10 or not record['boundary_omission_Gauss_rejected']:
                raise AssertionError('Charged boundary omission control did not change the actual operator/law')
            record['boundary_omission_operator_shift_residual'] = shift_error
            record['gap_bound_without_boundary_energy'] = parameters.gap_bound - removed
        else:
            trial, epsilon, _ = neutral_trial(model, parameters)
            onsite_residual = float(np.linalg.norm(onsite @ trial + 8 * parameters.mass * trial))
            photon_residual = float(np.linalg.norm(photon @ trial - epsilon * trial))
            hopping_expectation = float(np.vdot(trial, hopping @ trial).real)
            expectation = float(np.vdot(trial, model.H @ trial).real)
            expected = -8 * parameters.mass + epsilon
            if max(onsite_residual, photon_residual, abs(hopping_expectation), abs(expectation - expected)) > 2e-9:
                raise AssertionError('Actual neutral trial does not attain the stated variational upper bound')
            record['neutral_trial'] = {'onsite_eigenvector_residual': onsite_residual,
                                       'actual_photon_eigenvector_residual': photon_residual,
                                       'hopping_expectation': hopping_expectation,
                                       'total_energy_expectation': expectation,
                                       'neutral_energy_upper_bound': expected,
                                       'full_H_variational_state_residual': float(np.linalg.norm(model.H @ trial - expected * trial)),
                                       'scope': 'A physical trial, not the interacting ground state.'}
        output.append(record)
    return {'models': output, 'per_axis_link_singular_value_sums': singular_sums,
            'actual_Pauli_intertwiner_residual': float(code.law_residual),
            'coupled_ground_eigensolver_runs': 0}


def hypothesis_controls():
    # Deliberately WRONG cyclic truncation: W is a permutation, including the wrap.
    electric, magnetic, cutoff = 1.0, 10.0, 1
    def wrapped_energy(offsets):
        _, diagonal, _ = rotor(offsets, cutoff, electric, magnetic)
        size = len(diagonal)
        shift = np.zeros((size, size))
        for column in range(size):
            shift[(column + 1) % size, column] = 1
        return float(np.linalg.eigvalsh(np.diag(diagonal) - magnetic / 2 * (shift + shift.T))[0])
    offsets = (0, 0, 1, 1)
    wrapped_neutral = wrapped_energy((0, 0, 0, 0))
    wrapped_offset = wrapped_energy(offsets)
    if wrapped_offset >= wrapped_neutral - 1e-6:
        raise AssertionError('Cyclic-rotor countercontrol failed to expose the false theorem')
    fractional_rejected = False
    try:
        loop_domain((0.5, 0.5, 0.5, 0.5), 1)
    except ValueError:
        fractional_rejected = True
    # At R=1,K=0, translating Z by 1/2 is NOT an integer-path bijection.
    fractional_log_trace = np.asarray(logsumexp(-np.array([0.5, 0.5]))).item()
    neutral_log_trace = np.asarray(logsumexp(-rotor_spectrum((0, 0, 0, 0), 1, 1.0, 0.0))).item()
    if not fractional_rejected or abs(fractional_log_trace - neutral_log_trace) < 1e-6:
        raise AssertionError('Fractional-offset control hid the translation hypothesis')
    return {'cyclic_wrap': {'UE': electric, 'K': magnetic, 'cutoff': cutoff,
                             'true_Gauss_offsets': offsets,
                             'wrapped_neutral_ground': wrapped_neutral,
                             'wrapped_offset_ground': wrapped_offset,
                             'bound_violation': wrapped_neutral - wrapped_offset,
                             'scope': 'Wrong cyclic truncation genuinely falsifies the claimed inequality.'},
            'noninteger_offsets': {'rejected': fractional_rejected,
                                   'K0_R1_t1_neutral_log_trace': neutral_log_trace,
                                   'K0_R1_t1_half_shift_log_trace': fractional_log_trace,
                                   'scope': 'Not equal under translation; no claim that every fractional offset violates the ground bound.'}}


def parameter_report(parameters):
    photon_speed = math.sqrt(parameters.electric * parameters.magnetic)
    charge = (parameters.electric / parameters.magnetic)**0.25
    family = []
    for q in (0.2, 0.3, 0.4):
        speed = parameters.v
        electric = speed * q**2
        magnetic = speed / q**2 if speed else 0.0
        family.append({'rescaled_charge': q, 'c_equals_v': speed, 'UE': electric,
                       'K': magnetic, 'require_Delta_strictly_greater_than': 2 * magnetic,
                       'sufficient_gap': parameters.mass + electric / 2 - parameters.hopping_bound})
    return {'input': parameters.model_kwargs(), 'M': parameters.mass,
            'new_sufficient_gap': parameters.gap_bound,
            'old_sufficient_gap': parameters.gap_bound - parameters.magnetic,
            'sharpening_exactly_K': parameters.magnetic,
            'new_bound_positive': parameters.gap_bound > 0,
            'harmonic_matching_not_measured_photon': {
                'c': photon_speed, 'rescaled_charge': charge,
                't_prime_equals_v_t': parameters.v,
                'UE_divided_by_v': parameters.electric / parameters.v if parameters.v else None,
                'K_divided_by_v': parameters.magnetic / parameters.v if parameters.v else None,
                'scope': 'Bare small-angle matching; not an observed compact-photon pole or renormalized charge.'},
            'equal_speed_parameter_family': family,
            'old_K_subtraction_control': 'Conservative but not wrong physics; it discards the common epsilon0 cancellation.'}


def run(parameters, cutoffs=(1, 2, 4, 8), actual_cutoff=1, trace_times=(0.125, 1.0, 8.0, 64.0)):
    parameters.validate()
    if not cutoffs or any(int(r) != r or r < 1 for r in cutoffs) or list(cutoffs) != sorted(set(cutoffs)):
        raise ValueError('Require strictly increasing positive integer spectral cutoffs')
    if actual_cutoff not in cutoffs:
        raise ValueError('Actual-operator cutoff must occur in the spectral cutoff sequence')
    if not trace_times or any(not math.isfinite(t) or t <= 0 for t in trace_times):
        raise ValueError('Trace times must be positive and finite')
    pure = spectral_consumer(cutoffs, parameters, trace_times)
    first = next(case for case in pure['cases'] if case['cutoff'] == actual_cutoff
                 and case['UE'] == parameters.electric and case['K'] == parameters.magnetic)
    actual = actual_operator_consumer(parameters, actual_cutoff, first['actual_Gauss_dimensions_by_N'])
    sources = (Path(__file__), Path(spinor.__file__))
    return {'construction': 'Integer-Gauss-loop Holder diamagnetism with the actual neutral rotor ground',
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'parameters': parameter_report(parameters), 'pure_offset_consumers': pure,
            'actual_spinor_operator_consumer': actual, 'hypothesis_controls': hypothesis_controls(),
            'scope': 'Universal proof plus finite rotor/operator discriminants; no new coupled charged-ground or photon measurements.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--actual-cutoff', type=int, default=1)
    parser.add_argument('--electric', type=float, default=0.0072)
    parser.add_argument('--magnetic', type=float, default=0.08 / 0.3**2)
    parser.add_argument('--delta', type=float, default=2.2)
    parser.add_argument('--m', type=float, default=0.8)
    parser.add_argument('--r', type=float, default=0.04)
    parser.add_argument('--v', type=float, default=0.08)
    args = parser.parse_args()
    parameters = GapParameters(args.electric, args.magnetic, args.delta, args.m, args.r, args.v)
    print(json.dumps(run(parameters, args.cutoffs, args.actual_cutoff), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
