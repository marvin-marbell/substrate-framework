#!/usr/bin/env python3
"""New v3W neutral-endpoint scattering functional, not an old-state receipt.

The microscopic Hamiltonian/state and their transfer bounds remain A-owned.
This consumer evaluates that transfer's centred whole-light comparison using
literal endpoint coefficients, finite-time neutral tangents and one GLOBAL
receiver metric. It does not substitute this comparison for the physical law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from flint import arb, ctx

import noncompact_harmonic_scattering_fiber as fibre
import noncompact_legal_volume_bloch as legal
from improved_physical_compton import ImprovedBloch, decode, encode
from prepared_neutral_centred_current import PreparedCentredCurrent
from sea_selection import SELECTED

ROOT = Path(__file__).resolve().parent
CLOCK = 6.0
DEPENDENCIES = (
    'physical_fullfield_scattering.py', 'prepared_neutral_centred_current.py',
    'prepared_neutral_full_field.py', 'improved_phi_band.py',
    'improved_physical_compton.py', 'noncompact_harmonic_scattering_fiber.py',
    'noncompact_legal_volume_bloch.py', 'local_orbital_stability.py',
)


def source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in DEPENDENCIES}


def fixed_binary_receiver(columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ONE joint actual endpoint Gram; return frozen F and the global Gram.

    The COMPLETE lab effects are (I-|F2><F2|, |F2><F2|), not the
    receiver-span sub-POVM and not a receiver-span-conditioned probability.
    """
    columns = np.asarray(columns, dtype=complex)
    if columns.ndim != 2 or columns.shape[1] != 2:
        raise ValueError('Exactly two actual geometric/internal columns required')
    gram = columns.conj().T @ columns
    values, vectors = np.linalg.eigh(gram)
    if np.min(values) <= 0:
        raise ArithmeticError('Actual joint receiver Gram is not positive')
    inverse = (vectors * values**-.5) @ vectors.conj().T
    receiver = columns @ inverse
    receiver.setflags(write=False)
    gram.setflags(write=False)
    return receiver, gram


def binary_probabilities(second: np.ndarray, normalized_charged: np.ndarray) -> tuple[float, float]:
    """Unconditional NON-SUCCESS and SECOND probabilities; no normalization."""
    success = float(abs(np.vdot(second, normalized_charged))**2)
    return 1-success, success


def weighted_global_shadow(amplitudes: np.ndarray, raw_C: np.ndarray, raw_G: np.ndarray,
                           raw_m: np.ndarray, input_norm2: np.ndarray,
                           endpoint_diagonal: np.ndarray) -> dict[str, Any]:
    """ONE global shadow norm/metric from actual512 complex sector amplitudes.

    The amplitudes must include the actual ONE-floor filter weights, initial
    minimizer/tie amplitudes, and ordinary phases. No equal64 default, sector
    source normalization, receiver whitening, or averaging calibrated mu exists.
    Orthogonal m/s sectors and their preserved SOURCE/current words are the
    shadow premise; actual beta/Q/normal transfers are separately owned.
    """
    amplitudes = np.asarray(amplitudes, complex)
    coefficients = [np.asarray(value, complex) for value in (raw_C, raw_G, raw_m)]
    norms = np.asarray(input_norm2, float)
    diagonal = np.asarray(endpoint_diagonal, float)
    if (amplitudes.shape != (64, 8) or norms.shape != (64, 8)
            or diagonal.shape != (64, 8, 2)
            or any(value.shape != (64, 8) for value in coefficients)):
        raise ValueError('All64 colour/all8 histories and exactly two endpoint diagonals required')
    weight = abs(amplitudes)**2
    neutral_norm2 = float(np.sum(weight))
    if neutral_norm2 <= 0 or np.any(norms < 0) or np.any(diagonal < 0):
        raise ValueError('Actual global shadow preparation/norms must be positive')
    probability = weight/neutral_norm2
    source_norm2 = float(np.sum(probability*norms))
    endpoint = np.sum(probability[:, :, None]*diagonal, axis=(0, 1))
    if source_norm2 <= 0 or np.min(endpoint) <= 0:
        raise ArithmeticError('Actual once-global source/receiver metric lost positivity')
    C, G, m = (complex(np.sum(probability*value)) for value in coefficients)
    first, second = np.sqrt(endpoint)
    derivative = 1/(first*second*(first+second))
    mu = (C/second-derivative*G*m)/math.sqrt(source_norm2)
    return dict(actual_sector_amplitudes=amplitudes, global_neutral_norm2=neutral_norm2,
        global_source_norm2=source_norm2, global_endpoint_Gram0_diagonal=endpoint,
        global_raw_C=C, global_raw_G=G, global_raw_m=m,
        offdiagonal_inverse_sqrt_derivative=-derivative,
        GLOBAL_shadow_leading_coefficient=mu, actual_weights_supplied=True,
        equal64_assumed=False, sector_input_whitened=False, per_sector_receiver_reset=False,
        ordinary_phases_retained_in_amplitudes=True,
        physical_probability_certified=False,
        scope='orthogonal512-sector mean shadow; exact global floor/filter/ordinary phases supplied '
              'by owner; beta/Q/normal/coupled physical transfer not inferred')


def paired_bank_global_shadow(amplitudes: np.ndarray, raw_C: np.ndarray, raw_G: np.ndarray,
                              raw_m: np.ndarray, input_norm2: np.ndarray,
                              endpoint_diagonal: np.ndarray) -> dict[str, Any]:
    """ONE physical paired SOURCE receiver; independent globally normalized inputs.

    Bank0 is sea/Ref4; bank1 is seaPlus4/Ref0 on the SAME fixed qRef=nRef-4.
    Each of the TWO templates is B_jplus+B_jminus, not a branch measurement.
    G/G0 average over BOTH banks before ONE inverse root. C/m/input norms
    retain the charge's physically branch-selected input, without whitening.
    """
    amplitudes = np.asarray(amplitudes, complex)
    coefficients = [np.asarray(value, complex) for value in (raw_C, raw_G, raw_m)]
    norms = np.asarray(input_norm2, float)
    diagonal = np.asarray(endpoint_diagonal, float)
    if (amplitudes.shape != (64, 8, 2) or norms.shape != (64, 8, 2)
            or diagonal.shape != (64, 8, 2, 2)
            or any(value.shape != (64, 8, 2) for value in coefficients)):
        raise ValueError('All64/all8/TWO physical banks and exactly TWO templates required')
    weights = abs(amplitudes)**2
    neutral_norm2 = float(np.sum(weights))
    if neutral_norm2 <= 0 or np.any(norms < 0) or np.any(diagonal < 0):
        raise ValueError('Actual coherent-bank global preparation must have positive norms')
    probability = weights/neutral_norm2
    bank_weights = np.sum(probability, axis=(0, 1))
    source_norm2 = np.sum(probability*norms, axis=(0, 1))
    endpoint = np.sum(probability[:, :, :, None]*diagonal, axis=(0, 1, 2))
    if np.min(source_norm2) <= 0 or np.min(endpoint) <= 0:
        raise ArithmeticError('Both actual charged inputs and the ONE global receiver must exist')
    raw_c, raw_g, raw_overlap = coefficients
    C = np.sum(probability*raw_c, axis=(0, 1))
    G = complex(np.sum(probability*raw_g))
    m = np.sum(probability*raw_overlap, axis=(0, 1))
    first, second = np.sqrt(endpoint)
    derivative = 1/(first*second*(first+second))
    mu = (C/second-derivative*G*m)/np.sqrt(source_norm2)
    return dict(actual_sector_bank_amplitudes=amplitudes,
        global_neutral_norm2=neutral_norm2, actual_bank_weights=bank_weights,
        globally_normalized_charge_input_norm2=source_norm2,
        ONE_global_endpoint_Gram0_diagonal=endpoint, ONE_global_raw_G=G,
        charge_raw_C=C, charge_raw_m=m,
        offdiagonal_inverse_sqrt_derivative=-derivative,
        physical_bank_shadow_leading_coefficients=mu,
        receiver='TWO columns from (Bplus_j+Bminus_j) Un SAME Xi; ONE global F/E2',
        separate_charge_receiver_calibrated=False, bank_balance_assumed=False,
        ordinary_bank_energies_and_phases_retained=True, input_whitened=False,
        fixed_reference_charge_operator='n_ref-4',
        physical_probability_certified=False,
        scope='paired-bank mean leading coefficient; actual whole-field/filter/Root/photon/Noether transfer required')


def paired_physical_binary_enclosure(
        descriptor: dict[str, Any], mean_coefficient_real_lower: str,
        normalized_input_error_upper: str, endpoint_matrix_error_upper: str,
        mean_Gram0_lower: str, mean_endpoint_operator_upper: str) -> dict[str, Any]:
    """Transfer a SAME-H paired amplitude to its ONE frozen physical binary effect.

    Errors are absolute norms of the normalized charged input and TWO-column
    endpoint matrix, supplied by the actual source/weighted/Root proofs. The
    mean coefficient already includes its leading-relative structured and
    finite-coupling Weyl remainder; photon spectral phase is a different unit.
    No old single-bank probability or separate-charge root enters this map.
    """
    ue = Fraction(descriptor['physical_family']['UE'])
    leading = (arb(ue.numerator)/ue.denominator)/int(descriptor['N'])**3
    coefficient = arb(mean_coefficient_real_lower)
    error_x, error_y = arb(normalized_input_error_upper), arb(endpoint_matrix_error_upper)
    gram_lower, y_upper = arb(mean_Gram0_lower), arb(mean_endpoint_operator_upper)
    if not bool(error_x >= 0) or not bool(error_y >= 0) or not bool(gram_lower > 0):
        raise ValueError('Actual norm errors and positive joint Gram lower required')
    gram_error = 2*y_upper*error_y+error_y**2
    physical_gram_lower = gram_lower-gram_error
    if not bool(physical_gram_lower > 0):
        raise ValueError('Supplied actual endpoint error does not preserve receiver rank')
    # Integral representation of x^-1/2 on positive matrices: norm Lipschitz
    # <= 1/(2 min(lambda_min G,lambda_min Gmean)^(3/2)).
    root_error = gram_error/(2*physical_gram_lower**arb('1.5'))
    effect_vector_error = error_y/physical_gram_lower.sqrt()+y_upper*root_error
    amplitude_error = error_x+effect_vector_error
    relative_error = amplitude_error/leading
    amplitude_lower = leading*(coefficient-relative_error)
    if not bool(amplitude_lower > 0):
        raise ValueError('Actual physical transfer does not prove a positive Born amplitude')
    probability_lower = amplitude_lower**2
    return dict(scope='ordinary T6 actual-source/weighted/Root proof transfer to SAME paired STATIC binary; spatial Noether loop is distinct',
        leading_amplitude_scale_UE_over_V=leading.str(65),
        physical_joint_Gram_lower=physical_gram_lower.str(65),
        input_absolute_norm_error=error_x.str(65),
        endpoint_absolute_operator_error=error_y.str(65),
        global_inverse_root_error=root_error.str(65),
        frozen_F2_absolute_norm_error=effect_vector_error.str(65),
        corrected_mean_amplitude_coefficient_real_lower=coefficient.str(65),
        additional_source_normal_Root_amplitude_relative_error=relative_error.str(65),
        final_physical_amplitude_coefficient_real_lower=(coefficient-relative_error).str(65),
        physical_second_probability_lower=probability_lower.str(65),
        second_probability_lower_relative_to_UE2_over_V2=(probability_lower/leading**2).str(65),
        complete_binary_complement_upper_exact='1 - lower_endpoint('+probability_lower.str(65)+')',
        complete_binary_effects='I-|F2><F2| and |F2><F2|; unconditional',
        separate_charge_receiver=False, input_whitened=False,
        endpoint_error_and_Gram_units='original physical raw SOURCE Y; NO column/bank rescaling',
        spectral_phase_error_not_used_as_Born_relative=True)


def gaussian_source_embedding(view: Any, profile: np.ndarray) -> dict[str, Any]:
    """SOURCE-complete native108 embedding; never a full-H/current cutover.

    Keep the owner's arbitrary active70 state, tensor its documented fifty
    occupied Gamma0-negative complement modes, and represent the original
    SOURCE's harmonic normal coordinate by all27 coherent packets. Returned
    packet amplitudes/operators specify the same state for downstream consumers.
    No source normalization, light-state Gaussian approximation, or filter occurs.
    """
    from prepared_neutral_canonical_source import canonical_coordinate_marks, literal_car_transfers
    from prepared_neutral_full_field import BETA

    if view.law.volume != 27 or tuple(view.shape_neutral) != (9, 56, 6, 2, 70):
        raise ValueError('This embedding binds the historical native N3/PC9/CAR8 state only')
    charge = view.charge
    profile = np.asarray(profile, dtype=complex)
    if charge not in (-1, 1) or profile.shape != (27,):
        raise ValueError('Native charge and full27-site profile required')
    neutral = view.xi.reshape(view.shape_neutral)
    if np.count_nonzero(neutral[:, 1:]) != 0:
        raise ValueError('Exact initial coexact/harmonic vacuum required; no reset permitted')
    initial = neutral[:, 0]
    output_shape = (9, 6, 2, 224)
    creators = literal_car_transfers(view, charge)
    packets = np.zeros((4, 27, math.prod(output_shape)), complex)
    fields = np.zeros((4, 27, 9, 9, 4, 4), complex)
    eta = profile if charge == 1 else profile.conjugate()
    for basis in range(4):
        nodes = canonical_coordinate_marks(view, np.eye(4)[basis])
        field = np.einsum('qa,qxij,qb->xabij', view.Q2, nodes, view.Q2)
        fields[basis] = field
        for x in range(27):
            output = np.zeros(output_shape, complex)
            for orbital in range(2):
                for i in range(4):
                    for reference in range(4):
                        matrix = (field[x, :, :, i, reference] if charge == 1
                                  else field[x, :, :, reference, i])
                        moved = np.einsum('ab,bhri->ahri', matrix, initial, optimize=True)
                        car = creators[4*orbital+i, reference]
                        output += eta[x]*view.orbitals[x, orbital]*np.asarray(
                            (car @ moved.reshape(-1, 70).T).T).reshape(output_shape)
            packets[basis, x] = output.reshape(-1)
    displacement = view.law.sites-view.law.sites[view.reference_site]
    harmonic_words = -charge*displacement/view.law.size
    word_norms = np.sum(harmonic_words**2, axis=1)
    full_kernel = np.exp(-np.sum(
        (harmonic_words[:, None]-harmonic_words[None, :])**2, axis=-1)/4)
    vacuum_kernel = np.exp(-(word_norms[:, None]+word_norms[None, :])/4)
    grade1_kernel = vacuum_kernel*(1+harmonic_words @ harmonic_words.T/2)
    root_vectors = initial.reshape(9, -1)
    root_factor = np.linalg.qr(root_vectors.T, mode='r').T
    root_error = float(np.linalg.norm(
        root_vectors @ root_vectors.conj().T-root_factor @ root_factor.conj().T))
    complement = np.eye(27)-view.orbitals @ view.orbitals.conj().T
    occupied_spin = (np.eye(4)-SELECTED[0])/2
    source_spin = (np.eye(4)+SELECTED[0])/2 if charge == 1 else occupied_spin.T
    if root_error > 1e-12 or abs(np.trace(complement).real-25) > 1e-10:
        raise ArithmeticError('Actual arbitrary-state marginal/complement embedding lost')
    phi_fields = fields if charge == 1 else fields.swapaxes(-1, -2)
    outside_packets = np.einsum(
        'axbcir,ct->axirbt', phi_fields, root_factor, optimize=True)
    outside_packets *= eta[None, :, None, None, None, None]

    def gram(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        active = np.einsum('axd,xy,byd->ab',
                           packets.conj(), kernel, packets, optimize=True)
        outside = np.einsum('axiruv,xy,ij,byjruv->ab',
            outside_packets.conj(), complement*kernel, source_spin,
            outside_packets, optimize=True)
        return active, outside

    active, outside = gram(full_kernel)
    active1, outside1 = gram(grade1_kernel)
    active0, outside0 = gram(vacuum_kernel)
    full, cap1, vacuum = active+outside, active1+outside1, active0+outside0
    coefficients = BETA if charge == 1 else BETA.conj()

    def norm2(matrix: np.ndarray) -> float:
        return float(np.vdot(coefficients, matrix @ coefficients).real)

    if np.linalg.eigvalsh(outside)[0] < -1e-10 or np.linalg.eigvalsh(full-cap1)[0] < -1e-10:
        raise ArithmeticError('SOURCE complement/normal-tail positivity lost')
    return dict(
        charge=charge, Hermitian_G=full, active_G=active, outside_G=outside,
        normal_grade1_G=cap1, normal_vacuum_G=vacuum,
        eigenvalues=np.linalg.eigvalsh(full), raw_coefficients=coefficients,
        full_raw_creator_norm2=norm2(full), normal_grade1_raw_norm2=norm2(cap1),
        normal_vacuum_raw_norm2=norm2(vacuum),
        missing_SOURCE_outside_old8_active_orbitals_norm2=norm2(outside),
        missing_initial_SOURCE_normal_ge2_norm2=norm2(full-cap1),
        full_unclipped_incoming_one_hard_photon_norm2=norm2(full),
        old_input_totalgrade1_andCAR8_norm2=norm2(active0),
        total_incoming_norm2_omitted_by_old_consumer=norm2(full)-norm2(active0),
        root_marginal_QR_reconstruction_error=root_error,
        active_light_state_arbitrary_nonGaussian70=True,
        complement_occupied_count=50, complement_empty_count=50,
        initial_whole_light_count=54, charged_whole_light_count=55 if charge == 1 else 53,
        Ref_neutral=4 if charge == 1 else 0, Ref_charged=3 if charge == 1 else 1,
        initial_state_reprepared=False, input_whitened=False,
        whole_negative_cluster_assumed=False, current_and_whole_H_complete=False,
        physical_MinSet_transfer=False, T6_Born_executed=False,
        state_embedding=dict(
            initial_active_state=initial, source_packet_fields=fields,
            active_packet_amplitudes=packets, profile=eta,
            spatial_complement_projector=complement, source_spin_projector=source_spin,
            harmonic_words=harmonic_words, harmonic_kernel=full_kernel,
            coexact_words=np.zeros((27, 52)),
            recipe='arbitrary active70 Xi tensor occupied50 Gamma0-negative complement; '
                   'allRef and original per-edge SOURCE; distinct orthogonal active/outside sectors'),
        scope='SOURCE-only full108 CAR contraction and entire harmonic-normal Gaussian input; '
              'not full108 Hamiltonian/current/dynamics, whole negative cluster, or legal-N MinSet')


def run_native_source_embedding(descriptor_path: Path, state_path: Path) -> dict[str, Any]:
    """Exercise the shared SOURCE API on both charge views of ONE saved Xi."""
    from prepared_neutral_canonical_source import CanonicalPreparedSpatialChargeTransfer
    from prepared_neutral_spatial_gauss_well import PreparedSpatialGaussQuantumSlice

    start = time.perf_counter()
    descriptor_bytes, state_bytes = descriptor_path.read_bytes(), state_path.read_bytes()
    descriptor = decode(json.loads(descriptor_bytes))
    owner = PreparedSpatialGaussQuantumSlice(state_capsule=state_path)
    signature = owner.operator_signature()
    profile = np.zeros(owner.law.volume, complex)
    for offset, eta in zip(descriptor['compact_displacement'], descriptor['source_eta']):
        position = tuple((np.asarray(offset, dtype=int)+1) % 3)
        profile[owner.law.index[position]] += eta
    results = []
    for charge in (1, -1):
        view = CanonicalPreparedSpatialChargeTransfer(owner, charge)
        result = gaussian_source_embedding(view, profile)
        # CLI emits decision-ready contractions, not the large reconstructible state bundle.
        results.append({key: value for key, value in result.items() if key != 'state_embedding'})
    if (owner.operator_signature() != signature or descriptor_path.read_bytes() != descriptor_bytes
            or state_path.read_bytes() != state_bytes):
        raise RuntimeError('Native SOURCE consumer changed the same owner H/state/input')
    return dict(topic='NEW same-state SOURCE-complete native108/fullGaussian input',
        charge_views=results, state_sha256=hashlib.sha256(state_bytes).hexdigest(),
        descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        owner_signature=signature, initial_EM_vacuum_exact=True,
        state_reprepared=False, source_normal_domain='all Gaussian Hermite grades via27 coherent packets',
        geometry='N3/PC9; SOURCE19 folded27-site profile; not selected legalN modes',
        current_and_whole_H_complete=False, physical_MinSet_transfer=False,
        T6_Born_executed=False, natural_exit=0, actual_seconds=time.perf_counter()-start)


class NeutralEndpointBloch(ImprovedBloch):
    """Finite neutral-endpoint words, with no Euclidean ground preparation."""

    def __init__(self, points: np.ndarray, current: PreparedCentredCurrent):
        super().__init__(points, current.frame, CLOCK, current.band)
        self.current_owner = current
        self.literal_coefficients = current.coefficients
        # These are the owner's literal range-one/range-two Phi endpoint
        # coefficients, not independently assigned Dirac or SU(2) matrices.
        coefficients = {tuple(d): F for d, F in current.coefficients}
        self.endpoint = coefficients[(0, 0, 1)]
        self.second = coefficients[(0, 0, 2)]

    def info(self, shift: np.ndarray) -> dict[str, Any]:
        key = tuple(float(q) for q in shift)
        if key not in self.cache:
            points = self.points + shift
            h = np.broadcast_to(self.current_owner.onsite * self.frame[0],
                                (len(points), 4, 4)).copy()
            for d, F in self.literal_coefficients:
                phase = fibre.bounded_phase(points @ d)[:, None, None]
                h += phase * F + phase.conj() * F.conj().T
            energy = np.sqrt(np.trace(h @ h, axis1=1, axis2=2).real / 4)
            if float(np.min(energy)) <= 0:
                raise ArithmeticError('Centred whole-light comparison gap closed')
            positive = (self.identity + h / energy[:, None, None]) / 2
            self.cache[key] = dict(h=h, E=energy, Q=positive,
                                   P=self.identity - positive)
        return self.cache[key]

    def finite_neutral_pair(self, incoming: dict[str, Any],
                            outgoing: dict[str, Any]) -> np.ndarray:
        zero = np.zeros(3)
        ki, ko = incoming['k'], outgoing['k']
        first, final = self.info(zero), self.info(ki + ko)
        ao = self.integral(incoming, ko)
        b0 = self.integral(outgoing, zero)
        # Direct full-sea Dyson word plus the ACTUAL finite neutral endpoint.
        # The latter cancels one direct occupied-sea term and adds the
        # negative-final source boundary. Initial v3W photon cloud is a
        # separate supplied error, not the old Euclidean vacuum tangent.
        return (final['Q'] @ self.mixed(incoming, outgoing) @ first['Q']
                - final['Q'] @ ao @ self.info(ko)['P'] @ b0 @ first['Q']
                + final['P'] @ ao @ self.info(ko)['Q'] @ b0 @ first['Q'])

    def finite_endpoint_gram(self, forward: dict[str, Any],
                             outgoing: dict[str, Any]) -> np.ndarray:
        zero = np.zeros(3)
        ki, ko = forward['k'], outgoing['k']
        first, final = self.info(zero), self.info(ki + ko)
        a0, ao = self.integral(forward, zero), self.integral(forward, ko)
        b0, bi = self.integral(outgoing, zero), self.integral(outgoing, ki)
        gram = (final['Q'] @ bi @ self.info(ki)['P'] @ a0 @ first['Q']
                - final['P'] @ ao @ self.info(ko)['Q'] @ b0 @ first['P'])
        free_final = self.free(ki + ko)
        free_initial = self.free(zero)
        photon_phase = fibre.bounded_phase(np.asarray(
            CLOCK * (forward['omega'] - outgoing['omega'])))
        return photon_phase * (free_final @ gram
                               @ free_initial.conj().swapaxes(-2, -1))

    def free(self, shift: np.ndarray) -> np.ndarray:
        info = self.info(shift)
        return (fibre.bounded_phase(-CLOCK * info['E'])[:, None, None] * info['Q']
                + fibre.bounded_phase(CLOCK * info['E'])[:, None, None] * info['P'])


class HoleNeutralEndpointBloch(NeutralEndpointBloch):
    """Exact light CAR particle-hole transform, with the SAME photon modes."""

    def __init__(self, points: np.ndarray, current: PreparedCentredCurrent):
        super().__init__(points, current)
        self.frame = np.asarray([-current.frame[0].T]
                                + [matrix.T for matrix in current.frame[1:]])
        self.literal_coefficients = [(d, -F.conj()) for d, F in current.coefficients]
        coefficients = {tuple(d): F for d, F in self.literal_coefficients}
        self.endpoint = coefficients[(0, 0, 1)]
        self.second = coefficients[(0, 0, 2)]

    def vertex(self, row: dict[str, Any], shift: np.ndarray) -> np.ndarray:
        # SAME physical A: h_h(p,A)=-h_e(-p,A)^T. Relative to the
        # F_h=-F_e* backend, A changes sign: first vertex minus, contact plus.
        return -super().vertex(row, shift)


def photon_packets(descriptor: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exact identities/norms; represented q0 angles require a splitting debt."""
    incoming, outgoing = legal.represented_packets(float(Fraction(descriptor['physical_family']['c'])))
    for packet, key in ((incoming, 'incoming'), (outgoing, 'outgoing_annihilation')):
        for row, exact in zip(packet['modes'], descriptor[key]):
            row['mode_id'] = tuple(int(k) for k in exact['mode_id'])
    identities = [row['mode_id'] for packet in (incoming, outgoing)
                  for row in packet['modes']]
    if len(set(identities)) != 4:
        raise ValueError('The four photon modes must remain exactly distinct')
    for packet in (incoming, outgoing):
        if abs(sum(abs(row['coefficient'])**2 for row in packet['modes']) - 1) > 1e-12:
            raise ValueError('Physical photon norm is the sum over distinct modes')
    return incoming, outgoing


def comparison_means(descriptor: dict[str, Any], current: PreparedCentredCurrent,
                     quadrature: int, branch: str = 'plus') -> dict[str, Any]:
    if quadrature < 4:
        raise ValueError('At least four periodic samples per coordinate required')
    if branch not in ('plus', 'minus'):
        raise ValueError('The actual source branch must be plus or minus')
    incoming, outgoing = photon_packets(descriptor)
    # Bilinearity combines LOCAL field coefficients only. This is not photon
    # mode identification or a replacement normalization. Exact mode IDs and
    # the nonzero 2pi/N debt remain in the output contract.
    first = dict(incoming['modes'][0], coefficient=sum(
        row['coefficient'] for row in incoming['modes']))
    last = dict(outgoing['modes'][0], coefficient=sum(
        row['coefficient'] for row in outgoing['modes']))
    static = dict(last, coefficient=last['coefficient'] * complex(
        fibre.bounded_phase(np.asarray(-CLOCK * last['omega']))))
    forward = dict(first, coefficient=first['coefficient'] * complex(
        fibre.bounded_phase(np.asarray(-first['k'][0]))))
    bare_photon = complex(fibre.bounded_phase(np.asarray(
        first['k'][0] - CLOCK * first['omega'])))
    matrix = np.asarray(descriptor['source_matrix'], dtype=complex)
    ds = np.asarray(descriptor['compact_displacement'])
    eta = np.asarray(descriptor['source_eta'], dtype=complex)
    if branch == 'minus':
        # SOURCE_minus=c_ref^dagger v^dagger c_light. In hole coordinates
        # the created amplitude is v*, not a guessed conjugate probability.
        matrix, eta = matrix.conj(), eta.conj()
    source_norm2 = float(np.linalg.norm(matrix)**2 * np.vdot(eta, eta).real / 4)
    if source_norm2 <= 0:
        raise ValueError('Actual comparison source norm must be positive')
    chirality = np.linalg.multi_dot(list(SELECTED))
    density = matrix @ matrix.conj().T
    density_even_error = float(np.linalg.norm(chirality @ density - density @ chirality))
    if density_even_error > 1e-10:
        raise ValueError('Comparison half-HS norm requires actual even source density')
    totals = np.zeros(3, complex)
    count = quadrature**3
    for start in range(0, count, 2048):
        indices = np.arange(start, min(start + 2048, count))
        points = 2 * math.pi * np.column_stack(
            np.unravel_index(indices, (quadrature,) * 3)) / quadrature
        kernel_type = NeutralEndpointBloch if branch == 'plus' else HoleNeutralEndpointBloch
        kernel = kernel_type(points, current)
        source = fibre.compact_fourier(points, ds, eta)
        detector = fibre.compact_fourier(points + first['k'] + last['k'], ds, eta)
        raw = kernel.free(first['k'] + last['k']) @ kernel.finite_neutral_pair(first, static)
        gram = kernel.finite_endpoint_gram(forward, last)
        info = kernel.info(np.zeros(3))
        bare = fibre.bounded_phase(-CLOCK * info['E'])[:, None, None] * info['Q']
        for j, block in enumerate((raw, gram, bare)):
            trace = np.trace(matrix.conj().T @ block @ matrix, axis1=1, axis2=2) / 2
            weight = detector.conj() * source if j < 2 else abs(source)**2 * bare_photon
            totals[j] += np.sum(weight * trace)
    # Exact FAR-RIGHT D/N=1/2 recoil, before floating arithmetic.
    c, g, m = -totals[0] / count, -totals[1] / count, totals[2] / count
    c, g, m = c / source_norm2, g / source_norm2, m / source_norm2
    global_word = c - .5 * g * m
    return dict(raw_C=complex(c), offGram=complex(g), bare_m=complex(m),
                GLOBAL_new_finite_endpoint_word=complex(global_word),
                # Unchanged LAW05: UE Pi^2 + (K/2) curl^2, leg sqrt(UE/(V omega)).
                amplitude_coefficient_at_UE_over_V=complex(global_word),
                EM_law='UE Pi_A^2 + (K/2)||curl A||^2; c=sqrt(2 UE K)',
                photon_mode_leg='sqrt(UE/(V omega)); exact independent mode operators',
                comparison_source_norm2=source_norm2,
                source_density_even_error=density_even_error,
                quadrature=quadrature, exact_mode_ids=[row['mode_id']
                    for packet in (incoming, outgoing) for row in packet['modes']],
                photon_norms=[sum(abs(row['coefficient'])**2 for row in packet['modes'])
                              for packet in (incoming, outgoing)],
                representative_mode_splitting_paid=False,
                physical_probability_certified=False,
                all_reference_colours=4, input_whitened=False,
                source_branch=branch, neutral_reference=4 if branch == 'plus' else 0,
                charged_reference=3 if branch == 'plus' else 1,
                reference_charge_operator='n_ref-4' if branch == 'plus' else 'n_ref',
                reference_scope='independent reference-sector/operator presentation; '
                                'minus Ref0/1 is NOT Gauss-neutral under fixed plus n_ref-4 hardware',
                common_physical_Xi_or_static_across_charge_E2_proved=False,
                target_outcome='SECOND nonforward GLOBAL orthonormal receiver only',
                ordinary_clock='actual neutral endpoint, not initial-Xi static lab effect')


C_WORDS = ((1., ('Q', 'c', 'Q')), (-1., ('Q', 'a', 'P', 'b', 'Q')),
           (1., ('P', 'a', 'Q', 'b', 'Q')))
G_WORDS = ((1., ('Q', 'b', 'P', 'a', 'Q')), (-1., ('P', 'a', 'Q', 'b', 'P')))


def word_norm(words: tuple, limits: dict[str, Any]) -> Any:
    return sum(abs(scale) * math.prod(limits[name] for name in word)
               for scale, word in words)


def word_error(words: tuple, limits: dict[str, float],
               errors: dict[str, float]) -> float:
    return float(sum(abs(scale) * errors[name]
        * math.prod(limits[prior] + errors[prior] for prior in word[:index])
        * math.prod(limits[later] for later in word[index+1:])
        for scale, word in words for index, name in enumerate(word)))


def comparison_enclosure(descriptor: dict[str, Any], current: PreparedCentredCurrent,
                         means: dict[str, Any], metric_radius: float,
                         velocity_radius: float = 0.) -> dict[str, Any]:
    """New finite-clock word enclosure, conditional on the supplied core box.

    Source/metric intervals bound the TRUE new leading coefficient, not a
    falsely small post-Born physical-state error. A's actual state/source
    transfer and B's physical loop remain separate obligations.
    """
    ctx.prec = 256
    gram = np.einsum('mij,nji->mn', current.frame, current.frame).real / 4
    g0, gs = float(gram[0, 0]), float(gram[1, 1])
    if not 0 <= metric_radius < min(g0, gs) / 4:
        raise ValueError('An admitted finite-N positive Gram box is required')
    m, r, v = current.band.m, current.band.r, current.band.v
    mu = m * math.sqrt(g0-metric_radius-metric_radius**2/(2*(gs-metric_radius)))
    t1 = r*math.sqrt(g0+metric_radius)+(v+velocity_radius)*math.sqrt(gs+metric_radius)/2
    t2 = r*math.sqrt(g0+metric_radius)/4
    B = (m+4.5*r)*math.sqrt(g0+metric_radius)+6*(t1+t2)
    incoming, outgoing = photon_packets(descriptor)
    fi, fo = (abs(sum(row['coefficient'] for row in packet['modes']))
              / math.sqrt(packet['modes'][0]['omega']) for packet in (incoming, outgoing))
    ji, jo, k = 2*(t1+2*t2)*fi, 2*(t1+2*t2)*fo, 2*(t1+4*t2)*fi*fo
    limits = dict(P=1., Q=1., a=CLOCK*ji, b=CLOCK*jo,
                  c=CLOCK*k+CLOCK**2*ji*jo)
    cn, gn = float(word_norm(C_WORDS, limits)), float(word_norm(G_WORDS, limits))
    eta, ds = np.asarray(descriptor['source_eta']), np.asarray(descriptor['compact_displacement'])
    l1 = float(np.sum(abs(eta)))
    source_factor = 2*l1*l1
    e, u = 4e-10, 2.**-52
    eh = 8*B*e
    ee = B*eh/mu+100*u*B*B/mu+e
    ep = (eh+B*ee/mu)/(2*mu)+100*u*(1+B/mu)
    eji, ejo, ek = (j*(4*e+100*u) for j in (ji, jo, k))
    ef, eo = 24*e+36*ee, 300*36*e+4*216*ee/3
    ea = 4*(6*(2*ep*ji+eji)+ji*ef+600*u*ji)
    eb = 4*(6*(2*ep*jo+ejo)+jo*ef+600*u*jo)
    ec = (4*(6*(2*ep*k+ek)+k*ef)
          +16*(18*(3*ep*ji*jo+eji*jo+ji*ejo)+ji*jo*eo+14400*u*ji*jo))
    errors = dict(P=ep, Q=ep, a=ea, b=eb, c=ec)
    free_error = 4*e+24*ee+4*ep+1000*u
    rounding = 1000*u+4*e+400*u*means['quadrature']**3
    ce, ge = word_error(C_WORDS, limits, errors), word_error(G_WORDS, limits, errors)
    arithmetic_c = 1.01*source_factor*(ce+free_error*(cn+ce)+cn*rounding)
    arithmetic_g = 1.01*source_factor*(ge+(2*free_error+free_error**2)*(gn+ge)+gn*rounding)
    arithmetic_m = 1.01*source_factor*(ep+free_error+rounding)

    sigma = arb('.3')
    sh, ch = sigma.sinh(), sigma.cosh()
    g0b, gsb = legal.binary64_ball(g0), legal.binary64_ball(gs)
    rb, vb, mb = (legal.binary64_ball(x) for x in (r, v, m))
    vb += legal.binary64_ball(velocity_radius)
    distance = legal.binary64_ball(metric_radius)
    rootlip = 1/(math.sqrt(gs)+math.sqrt(gs-metric_radius))
    h_lip = (m+12*r+3*(v+velocity_radius))*rootlip
    metric_dh = h_lip*metric_radius+3*math.sqrt(gs+metric_radius)*velocity_radius
    perturb = arb(2*metric_dh)*(2*sigma).exp()
    cbar = ((g0b-distance).sqrt()*(mb-3*rb*sh**2)
            -vb*(3*(gsb+distance)).sqrt()*sh-perturb)
    if not cbar > 0:
        raise ArithmeticError('Centred whole-light complex strip is not accretive')
    bc = ((g0b+distance).sqrt()*(mb+3*rb*(1+ch)**2)
          +3*vb*(gsb+distance).sqrt()*ch+perturb)
    imh = (6*rb*(g0b+distance).sqrt()*(1+ch)*sh
           +vb*(3*(gsb+distance)).sqrt()*sh+perturb)
    projector, flight = (1+bc/cbar)/2, (CLOCK*imh).exp()
    strip_ji, strip_jo = (2*(arb(t1)*sigma.exp()+2*arb(t2)*(2*sigma).exp())*field
                         for field in (fi, fo))
    strip_k = 2*(arb(t1)*sigma.exp()+4*arb(t2)*(2*sigma).exp())*fi*fo
    strip = dict(P=projector, Q=projector, a=CLOCK*flight**2*strip_ji,
                 b=CLOCK*flight**2*strip_jo,
                 c=CLOCK*flight**2*strip_k+CLOCK**2*flight**4*strip_ji*strip_jo)
    strip_l1 = sum((arb(float(abs(a)))*(sigma*int(np.sum(abs(d)))).exp()
                    for a, d in zip(eta, ds)), arb(0))
    alias = (legal.alias_factor((-sigma*means['quadrature']).exp())
             +legal.alias_factor(legal.capped_decay(sigma*int(descriptor['N']))))
    periodic_c = 2*strip_l1**2*flight*word_norm(C_WORDS, strip)*alias
    periodic_g = 2*strip_l1**2*flight**2*word_norm(G_WORDS, strip)*alias
    periodic_m = 2*strip_l1**2*flight*projector*alias

    mark_delta = (arb(descriptor['finite_alpha_input_normalized_matrix_error_interval'])
                  +arb(descriptor['finite_alpha_output_whitened_matrix_error_interval']))
    mark_product = 2*arb(2).sqrt()*mark_delta+mark_delta**2
    profile_delta = (4*abs(arb(descriptor['source_raw_profile_log_c_interval']))).expm1()
    source_c = arb(cn)*mark_product+source_factor*cn*profile_delta
    source_g = arb(gn)*mark_product+source_factor*gn*profile_delta
    source_m = mark_product+source_factor*profile_delta

    def propagated_error(dh: float, dj: float, dk: float, frequency: float = 0.) -> tuple[float, float, float]:
        da = CLOCK*dj+CLOCK**2*ji*(dh+frequency/2)
        db = CLOCK*dj+CLOCK**2*jo*(dh+frequency/2)
        dc = (CLOCK*dk+CLOCK**2*k*(dh+frequency)
              +CLOCK**2*((ji+jo)*dj+dj*dj)
              +2*CLOCK**3*ji*jo*(dh+frequency))
        changed = dict(P=4*dh/mu, Q=4*dh/mu, a=da, b=db, c=dc)
        df = CLOCK*dh
        wc, wg = word_error(C_WORDS, limits, changed), word_error(G_WORDS, limits, changed)
        return (1.01*source_factor*(wc+df*(cn+wc)),
                1.01*source_factor*(wg+(2*df+df*df)*(gn+wg)),
                1.01*source_factor*(4*dh/mu+df))

    metric_dj = (2*max(fi, fo)*(1.5*r+v/2)*rootlip*metric_radius
                 +max(fi, fo)*math.sqrt(gs+metric_radius)*velocity_radius)
    metric_dk = (2*fi*fo*(2*r+v/2)*rootlip*metric_radius
                 +fi*fo*math.sqrt(gs+metric_radius)*velocity_radius)
    metric_c, metric_g, metric_m = propagated_error(metric_dh, metric_dj, metric_dk)
    delta = 2*arb.pi()/int(descriptor['N'])
    # A strict dyadic upper, never binary64 underflow or an exact-mode merger.
    delta_upper = math.nextafter(float(delta.upper()), math.inf)
    omega = min(row['omega'] for packet in (incoming, outgoing) for row in packet['modes'])
    speed = float(Fraction(descriptor['physical_family']['c']))
    coefficient_change = delta_upper*(1+speed/(2*omega))
    dh = 12*(t1+2*t2)*delta_upper
    split_c, split_g, split_m = propagated_error(
        dh, max(ji, jo)*coefficient_change, k*2*coefficient_change, 2*speed*delta_upper)
    split_c += 8*l1*l1*cn*delta_upper+source_factor*cn*CLOCK*speed*delta_upper
    split_g += 8*l1*l1*gn*delta_upper+source_factor*gn*2*CLOCK*speed*delta_upper
    split_m += source_factor*(1+CLOCK*speed)*delta_upper

    ec_total = periodic_c+arithmetic_c+source_c+metric_c+split_c
    eg_total = periodic_g+arithmetic_g+source_g+metric_g+split_g
    em_total = periodic_m+arithmetic_m+source_m+metric_m+split_m
    c, g, m0 = (means[key] for key in ('raw_C', 'offGram', 'bare_m'))
    radius = ec_total+arb('.5')*(arb(abs(g))*em_total+arb(abs(m0))*eg_total+eg_total*em_total)
    reference = c-.5*g*m0
    norm = (arb(reference.real)**2+arb(reference.imag)**2).sqrt()
    positive = bool(norm > radius)
    return dict(metric_box_radius=metric_radius, velocity_radius=velocity_radius,
                sigma=sigma.str(65),
                complex_accretivity_lower=cbar.str(65),
                real_word_norms=dict(C=cn, G=gn),
                periodic=dict(C=periodic_c.str(65), G=periodic_g.str(65), m=periodic_m.str(65)),
                arithmetic=dict(C=arithmetic_c, G=arithmetic_g, m=arithmetic_m),
                finite_Haar_leading_interval=dict(C=source_c.str(65), G=source_g.str(65), m=source_m.str(65)),
                actual_core_metric=dict(C=metric_c, G=metric_g, m=metric_m),
                exact_distinct_mode_splitting=dict(C=split_c, G=split_g, m=split_m),
                GLOBAL_disk_radius=radius.str(65),
                new_leading_positive=positive,
                new_amplitude_coefficient_lower=(legal.positive_lower(norm-radius) if positive else None),
                current_functional_enclosure_only=True, physical_probability_certified=False,
                assumptions='supplied actual finite-N Gram box and fixed-source mark intervals; actual v3W state/source/receiver transfer not inferred')


def harmonic_leading_enclosure(descriptor: dict[str, Any], current: PreparedCentredCurrent,
                               metric_radius: float, velocity_radius: float = 0.,
                               *, ordinary_shift_l1_upper: int = 0) -> dict[str, Any]:
    """Structured Gaussian-harmonic POINT comparison, not whole-state Duhamel.

    Initial P0 is fixed while h(theta/N) evolves it. The once-global endpoint
    zeroth Gram is allowed to change; no instantaneous-sea reset or per-theta
    receiver calibration occurs. Actual Phi/filter/Root and harmonic-kinetic
    transfer must separately be supplied by the full-state owner.
    """
    ctx.prec = 256
    if ordinary_shift_l1_upper < 0:
        raise ValueError('Nonnegative ordinary momentum-shift cap required')
    gram = np.einsum('mij,nji->mn', current.frame, current.frame).real/4
    g0, gs = float(gram[0, 0]), float(gram[1, 1])
    if not 0 <= metric_radius < min(g0, gs)/4:
        raise ValueError('Actual finite-N point metric box required')
    # Conservative slack encloses binary64 evaluation of these analytic caps.
    t1 = 1.01*(current.band.r*math.sqrt(g0+metric_radius)
               +(current.band.v+velocity_radius)*math.sqrt(gs+metric_radius)/2)
    t2 = 1.01*current.band.r*math.sqrt(g0+metric_radius)/4
    incoming, outgoing = photon_packets(descriptor)
    fi, fo = (1.01*sum(abs(row['coefficient']) for row in packet['modes'])
              / math.sqrt(packet['modes'][0]['omega']) for packet in (incoming, outgoing))
    ji, jo = 2*(t1+2*t2)*fi, 2*(t1+2*t2)*fo
    contact = 2*(t1+4*t2)*fi*fo
    limits = dict(P=1., Q=1., a=CLOCK*ji, b=CLOCK*jo,
                  c=CLOCK*contact+CLOCK**2*ji*jo)
    cn, gn = float(word_norm(C_WORDS, limits)), float(word_norm(G_WORDS, limits))
    # All errors are linear in S/N, S=sum_j |theta_j|. Both exact and
    # shifted words obey the SAME caps, eliminating spurious volume factors.
    dh = 2*(t1+2*t2)
    dji, djo = 2*(t1+4*t2)*fi, 2*(t1+4*t2)*fo
    dk = 2*(t1+8*t2)*fi*fo
    da, db = CLOCK*dji+CLOCK**2*ji*dh, CLOCK*djo+CLOCK**2*jo*dh
    dc = (CLOCK*dk+CLOCK**2*(contact*dh+ji*djo+jo*dji)
          +2*CLOCK**3*ji*jo*dh)
    # If requested, also transport the INITIAL filled projector from kappa=0
    # to the ordinary sector kappa=pi(m/2+s)/N. Full10 G>0 gives the real
    # gap m*sqrt(lambda_min(G)); fourfold Davis-Kahan slack is sufficient.
    gap = current.band.m*math.sqrt(min(g0, gs)-metric_radius)
    projector_delta = arb(4*dh*ordinary_shift_l1_upper/gap)/int(descriptor['N'])
    wc = dc+2*(limits['a']*db+limits['b']*da)
    wg = 2*(limits['a']*db+limits['b']*da)
    eta = np.asarray(descriptor['source_eta'], complex)
    eta_norm2 = cast(arb, sum(legal.binary64_ball(z.real)**2+legal.binary64_ball(z.imag)**2 for z in eta))
    l1 = sum((legal.binary64_ball(z.real)**2+legal.binary64_ball(z.imag)**2).sqrt() for z in eta)
    factor = 2*l1**2/eta_norm2
    dmax = int(np.max(abs(np.asarray(descriptor['compact_displacement'], int))))
    # Vacuum density exp(-|theta|^2): E S=3/sqrt(pi)<3, no theta cutoff.
    mean_shift = arb(3+ordinary_shift_l1_upper)/int(descriptor['N'])
    errors = dict(C=factor*((wc+CLOCK*dh*cn+2*dmax*cn)*mean_shift
                           +(2*limits['c']+8*limits['a']*limits['b'])*projector_delta),
                  G=factor*((wg+2*CLOCK*dh*gn+2*dmax*gn)*mean_shift
                           +8*limits['a']*limits['b']*projector_delta),
                  m=factor*((CLOCK*dh+2*dmax)*mean_shift+projector_delta))
    # ||U_theta P0 U_theta^dagger-P0||<=2T dh S/N; source trace gives factor2.
    delta = 4*CLOCK*dh*mean_shift+4*projector_delta
    if not bool(delta < arb('.5')):
        raise ArithmeticError('Positive once-global zeroth receiver Gram required')
    inverse_root = (1-delta)**arb('-.5')
    off_derivative = arb('.5')*(1-delta)**arb('-1.5')
    ccap, gcap, mcap = factor*cn, factor*gn, factor
    coefficient_error = (
        errors['C']*inverse_root+ccap*(inverse_root-1)
        +off_derivative*(gcap*errors['m']+mcap*errors['G']+errors['G']*errors['m'])
        +gcap*mcap*(off_derivative-arb('.5')))
    return dict(
        relative_leading_errors={key: value.str(65) for key, value in errors.items()},
        GLOBAL_leading_relative_error=coefficient_error.str(65),
        zeroth_GLOBAL_Gram_operator_error=delta.str(65),
        once_GLOBAL_Gram_lower=(1-delta).str(65),
        offdiagonal_inverse_sqrt_derivative_bound=off_derivative.str(65),
        actual_eta_norm2=eta_norm2.str(65), gaussian_mean_L1_bound=3,
        ordinary_shift_L1_upper=ordinary_shift_l1_upper,
        ordinary_initial_projector_error=projector_delta.str(65),
        ordinary_full10_real_gap_lower=gap,
        common_birth_phase='exp[-i D theta_x/N] retained; matched SOURCE factors cancel only in contractions',
        filled_sea_Ward='uniform h/P_t Fourier diagonal: disjoint exact photon IDs force extensive closed-loop trace zero',
        initial_projector='P0 fixed; P_t=U_theta P0 U_theta^dagger, NOT instantaneous P_theta',
        receiver='ONE GLOBAL theta-averaged Gram at T6; no per-theta calibration',
        conditional_scope='uniform point quasi-free Gaussian-harmonic comparison; '
                          'actual Phi/normal/filter/Root, Coulomb, harmonic kinetic, photon moments remain owner transfers',
        physical_probability_certified=False, frozen_q128_promoted=False)


def original_source_operator_bound(profile: np.ndarray, quartet: np.ndarray,
                                   *, carrier_scope: str) -> arb:
    """Ordinary CAR/flatRootP bound, NOT a fullAuxQ claim or creator lower bound.

    Primitive labels are Hilbert-Schmidt orthogonal with trace4; Fx/Fr are
    unitary polar charts and |centre_scalar|<=1 by its positive seed weights.
    Each reference CAR annihilator has norm1, so ||B||<=2||nu||_HS.
    Encoded fullRootQ may have path-dependent ghost-face flux. This small
    bound must not replace its owner's CBQ<=16sqrt(V), beta/G leakage, or
    missing global odd-CAR/tree proof.
    """
    ctx.prec = 256
    if carrier_scope not in ('ordinary-CAR', 'flat-Root-P'):
        raise ValueError('Bound3 requires ordinary CAR or an independently established flat Root P')

    def squared_norm(vector: np.ndarray) -> arb:
        return cast(arb, sum(legal.binary64_ball(z.real)**2+legal.binary64_ball(z.imag)**2
                   for z in np.asarray(vector, complex).reshape(-1)))

    if np.asarray(quartet).shape != (4,):
        raise ValueError('Literal raw four-component quartet required')
    return 2*(squared_norm(profile)*squared_norm(quartet)).sqrt()


def coulomb_leading_enclosure(descriptor: dict[str, Any], current: PreparedCentredCurrent,
                             metric_radius: float, velocity_radius: float = 0.) -> dict[str, Any]:
    """True full-CAR HC; leading hard-mode insertion bound on the mean tube.

    HC is photon-number preserving: the nonforward transition requires the
    two current legs/contact. Do not treat its naked source phase as a Born
    amplitude error. Phi/normal/Root transfer and higher photon flow stay open.
    """
    from prepared_neutral_full_field import BETA

    ctx.prec = 256
    N, V = int(descriptor['N']), int(descriptor['V'])
    weak = Fraction(descriptor['physical_family']['UE'])
    ue = arb(weak.numerator)/weak.denominator
    source_bound = original_source_operator_bound(
        descriptor['source_eta'], BETA, carrier_scope='ordinary-CAR')
    if not bool(source_bound < arb(3)):
        raise ArithmeticError('Actual Source19 raw operator bound must be below3')
    # SAME fixed qref=nref-4; neutral Ref0 bank contains four extra electrons.
    # |rho_FAR|<=6, so sum rho²<=4(V-1)+36=4V+32.
    hc = ue*N**2*(4*V+32)/16
    gram = np.einsum('mij,nji->mn', current.frame, current.frame).real/4
    g0, gs = float(gram[0, 0]), float(gram[1, 1])
    if not 0 <= metric_radius < min(g0, gs)/4:
        raise ValueError('Actual finite-N point metric box required')
    t1 = 1.01*(current.band.r*math.sqrt(g0+metric_radius)
               +(current.band.v+velocity_radius)*math.sqrt(gs+metric_radius)/2)
    t2 = 1.01*current.band.r*math.sqrt(g0+metric_radius)/4
    incoming, outgoing = photon_packets(descriptor)
    fi, fo = (1.01*sum(abs(row['coefficient']) for row in packet['modes'])
              / math.sqrt(packet['modes'][0]['omega']) for packet in (incoming, outgoing))
    occupation = 2*V+4
    ji, jo = occupation*2*(t1+2*t2)*fi, occupation*2*(t1+2*t2)*fo
    contact = occupation*2*(t1+4*t2)*fi*fo
    # Four hard external modes only at this leading order. The contour/photon
    # combinatorics are overbounded by128/512, not unrepresented-mode return.
    factor = arb(3)**2/arb('.25')**2
    ccap = 128*factor*(CLOCK*contact+CLOCK**2*ji*jo)
    gcap, mcap = ccap, 4*factor
    ec = 512*factor*hc*(CLOCK**2*contact+CLOCK**3*ji*jo)
    eg, em = ec, 32*factor*hc*CLOCK
    delta = 64*factor*hc*CLOCK
    if not bool(delta < arb('.25')):
        raise ArithmeticError('Coulomb endpoint Gram comparison left the positive tube')
    # Before HC: mean harmonic Gram >=1/2. After HC: >=1/4.
    # On d1,d2>=1/4: inverse-root Lipschitz<=4; k<=4, |delta k|<=24 delta.
    total = (2*ec+4*ccap*delta
             +4*(gcap*em+mcap*eg+eg*em)+24*gcap*mcap*delta)
    return dict(
        ordinary_CAR_or_flatRootP_SOURCE_operator_norm_upper=source_bound.str(65),
        full_aux_Root_Q_small_SOURCE_bound_proved=False,
        full_aux_Root_Q_source_majorant='CBQ<=16sqrt(V) retained; owner beta/G/current/SOURCE leakage required',
        HC_full_CAR_operator_norm_upper=hc.str(65),
        leading_C_relative_error=ec.str(65), leading_G_relative_error=eg.str(65),
        bare_overlap_relative_error=em.str(65),
        zeroth_GLOBAL_Gram_error=delta.str(65),
        GLOBAL_leading_relative_error=total.str(65),
        rho='n_light(x)-2 +(n_ref-4)delta_FAR; n in[0,4], n_ref in[0,4], total0',
        charge_choices='SAME fixed law: plus2V+1/Ref3; physicalhole2V+3/Ref1; '
                       'coherent neutral2V/Ref4 + (2V+4)/Ref0',
        SOURCE_proof='unitary polar charts; convex centre scalar; raw beta/global2; all4 Ref CAR columns',
        source_norm_lower_assumption='actual normalized source/photon input norm>=1/4; owner must bind it',
        receiver_lower_assumption='uniform mean+Gaussian-harmonic zeroth GLOBAL Gram>=1/2 before HC',
        no_photon_transition_without_current=True, full_quartic_HC_not_squared_after_projection=True,
        higher_photon_remainder_paid=False,
        conditional_scope='ordinary CAR/independently flatRootP Coulomb correction to leading mean-tube coefficients; '
                          'actual Phi/filter/Root/current tube/source lower/receiver transfer and fullQ budget remain owner bound',
        physical_probability_certified=False, frozen_q128_promoted=False)


def finite_rank_spectator_enclosure(descriptor: dict[str, Any], current: PreparedCentredCurrent,
                                   metric_radius: float,
                                   velocity_radius: float = 0.) -> dict[str, Any]:
    """Four positive Bloch spectators: relative POINT anti-current correction.

    Use the SAME physical electron H: h_h=-h_e^T, P_h=Q_sea^T-R^T.
    R comprises complete positive spin pairs at two distinct plane momenta.
    No absolute Slater-distance estimate is substituted for a Born error.
    """
    ctx.prec = 256
    V = int(descriptor['V'])
    gram = np.real(np.einsum('mij,nji->mn', current.frame, current.frame))/4
    g0, gs = float(gram[0, 0]), float(gram[1, 1])
    t1 = 1.01*(current.band.r*math.sqrt(g0+metric_radius)
               +(current.band.v+velocity_radius)*math.sqrt(gs+metric_radius)/2)
    t2 = 1.01*current.band.r*math.sqrt(g0+metric_radius)/4
    incoming, outgoing = photon_packets(descriptor)
    fi, fo = (1.01*sum(abs(row['coefficient']) for row in packet['modes'])
              / math.sqrt(packet['modes'][0]['omega']) for packet in (incoming, outgoing))
    ji, jo = 2*(t1+2*t2)*fi, 2*(t1+2*t2)*fo
    contact = 2*(t1+4*t2)*fi*fo
    a, b = CLOCK*ji, CLOCK*jo
    c = CLOCK*contact+CLOCK**2*ji*jo
    eta = np.asarray(descriptor['source_eta'], complex)
    norm2 = cast(arb, sum(legal.binary64_ball(z.real)**2
                         +legal.binary64_ball(z.imag)**2 for z in eta))
    l1 = sum((legal.binary64_ball(z.real)**2+legal.binary64_ball(z.imag)**2).sqrt()
             for z in eta)
    factor = 2*l1**2/norm2
    # Each normalized plane has amplitude1/sqrt(V). Fourier SOURCE norm
    # <=||eta||1; R has four spin-plane modes. Both sides of each inserted R
    # contribute1/sqrt(V). Slack covers full4Ref source/current contractions.
    density = arb('8.08')*l1**2/(V*norm2)
    ec, eg, em = density*(2*c+6*a*b), density*6*a*b, density
    ccap, gcap, mcap = factor*(c+2*a*b), factor*2*a*b, factor
    # All source/endpoint norm increments are positive R traces <=density.
    # The joint metric is differentiated AFTER raw averaging; conservative
    # inverse-root/derivative Lipschitz constants use the common >=1/2 tube.
    total = (4*ec+8*ccap*density
             +4*(gcap*em+mcap*eg+eg*em)
             +48*gcap*mcap*density+4*density*(ccap+4*gcap*mcap))
    return dict(spectator_rank=4, spectator_planes=2,
        actual_hole_generator='-h_e(A)^T on SAME fixed qRef=nRef-4 law',
        initial_hole_projector='P_h=Q_sea^T-R^T; Q_h=P_sea^T+R^T',
        Fourier_source_relative_density_upper=density.str(65),
        source_profile_squared_slack='1.01 covers existing normalized c18/profile interval; not a profile renormalization',
        initial_SOURCE_norm2_increment='positive; BASE+Tr(nu†R nu), not normalization per column',
        raw_C_relative_error=ec.str(65), raw_G_relative_error=eg.str(65),
        bare_m_relative_error=em.str(65),
        source_and_zeroth_GLOBAL_Gram_relative_increment_upper=density.str(65),
        GLOBAL_leading_relative_error=total.str(65),
        extensive_offmode_sea_loop='zero: initialP andR andallfreeU are momentumdiagonal; exactincoming/outgoing IDs distinct',
        spectator_momenta='legal +(N/4,0,0), -(N/4,0,0), bothpositive spinpairs; outside p0(.02,0,0),radius.01 SOFT',
        bank_amplitude='owner actual1/sqrt2 RAW fork andONEglobalfn weights/phases retained separately',
        no_absolute_Slater_distance_Born_payment=True,
        current_scope='uniformPOINT/SOURCE19 raw current/metric perturbation; '
                      'pairedbank STATICreceiver andactualfilter/normal/Root/photon transfer still required',
        physical_probability_certified=False)

def mean_allmode_born_remainder_enclosure(
        descriptor: dict[str, Any], current: PreparedCentredCurrent,
        metric_radius: float, velocity_radius: float = 0.) -> dict[str, Any]:
    """Exact Weyl/Duhamel remainder AFTER the fixed-Phi mean-CAR transfer.

    This does not infer a coupled-Phi cubic moment from a Phi fourth moment.
    Photon-number-preserving HC/full harmonic kinetics remain in H0. Every
    coexact Peierls difference is a bounded CAR coefficient times (W(gv)-I).
    Third iterated Duhamel has the full unitary ONLY at its left endpoint.
    """
    ctx.prec = 256
    N = int(descriptor['N'])
    if N < 3:
        raise ValueError('The three-dimensional shell bound requires N>=3')
    V = N**3
    ue = Fraction(descriptor['physical_family']['UE'])
    speed = Fraction(descriptor['physical_family']['c'])
    UE, c = arb(ue.numerator)/ue.denominator, arb(speed.numerator)/speed.denominator
    gram = np.real(np.einsum('mij,nji->mn', current.frame, current.frame))/4
    g0, gs = float(gram[0, 0]), float(gram[1, 1])
    t1 = 1.01*(current.band.r*math.sqrt(g0+metric_radius)
               +(current.band.v+velocity_radius)*math.sqrt(gs+metric_radius)/2)
    t2 = 1.01*current.band.r*math.sqrt(g0+metric_radius)/4
    # Three directions, both orientations, all4 internal matrix singular
    # values; no norm of an unbounded photon coordinate enters this sum.
    coefficient_sum = 24*arb(V)*(arb(t1)+arb(t2))
    displacement = 8/c.sqrt()
    scale = 1+CLOCK*coefficient_sum*displacement
    coupling = UE.sqrt()
    small = coupling*scale
    if not bool(small < arb(2)**-16):
        raise ArithmeticError('The exact Weyl receiver remainder left its small-coupling tube')
    # One-photon input / vacuum neutral endpoint, SOURCE ordinary norm<=3.
    # X1<=3L, X2<=16L², RX<=2^10 g³L³; TWO-column Y0<=8,
    # Y1<=32L, Y2<=64L², RY<=2^13 g³L³. Photon parity gives
    # G1=A0=A1=0. RG<=2^18 g³L³, G2<=2^11 L².
    # G0>=1/2 and ||G-G0||<=2^12(gL)² give the inverse-root
    # remainder<=2^21 g³L³, RF<=2^25 g³L³, RA<2^30 g³L³.
    absolute = arb(2)**30*coupling**3*scale**3
    relative = absolute/(UE/V)
    return dict(mean_Phi_fixed=True, all_coexact_modes_retained=True,
        photon_leg='sqrt(UE/(V omega)); omega=c sqrt(lattice_laplacian)',
        harmonic_zero_mode_excluded_from_shell_and_retained_separately=True,
        coexact_inverse_frequency_shell_average_upper='4/3',
        shell_proof='j=max|n_i|, shell<=24j²+2, sqrt(lap)>=4j/N; '
                    'V^-1 sum1/sqrt(lap)<=(3N²+7N)/(4N²)<=4/3 forN>=3',
        range2_Weyl_displacement_squared_over_UE_upper=(64/c).str(65),
        bounded_mean_CAR_Peierls_coefficient_sum=coefficient_sum.str(65),
        mean_source_operator_upper=3,
        paired_source_norm_proof='derived Bminus=-Bplus†; ordinary even-CAR '
                                'dGamma offdiagonal skew matrix has norm<=sum singular(nu)'
                                '<=2||nu||HS<3; NOT fullauxQ CQ3',
        third_Duhamel_scale=scale.str(65), small_coupling_parameter=small.str(65),
        exact_receiver_Gram0_eigenvalue_premise='>=1/2 after common SOURCE factor',
        allmode_receiver_and_amplitude_remainder=absolute.str(65),
        leading_UE_over_V_relative_remainder=relative.str(65),
        receiver='TWO paired SOURCE columns, ONE actual GLOBAL inverse root/E2',
        source_coexact_number_commutator='zero: original PURE Dirac Coulomb phase',
        endpoint_creator_graph='[a_g,H_mean] bounded by g S d; '
                               'weighted full-unitary endpoint graph retained',
        initial_domain='mean neutral coexact vacuum and normalized one-photon SOURCE input; '
                       'actual prepared initial cloud/normal/Root transfer separately required',
        no_coupled_Phi_marginal_to_cubic_inference=True,
        physical_probability_certified=False)


def regular_allmode_photon_graph_enclosure(
        descriptor: dict[str, Any], current: PreparedCentredCurrent,
        physical_Phi_operator_upper: float,
        initial_annihilator_tensor_bounds: tuple[float, ...],
        duration: float = CLOCK) -> dict[str, Any]:
    """Ordinary regular-domain Touchard N² graph, with actual initial bounds.

    A_k²=||sqrt(N falling k) psi||². Exact Weyl commutators give
    A'_k <= S sum_{j=1}^k binom(k,j) d_g^j A_{k-j}. Massive
    normal momenta do not enter S. This is NOT a driven-Noether photon bound.
    """
    ctx.prec = 256
    if (len(initial_annihilator_tensor_bounds) != 5
            or min(initial_annihilator_tensor_bounds) < 0
            or physical_Phi_operator_upper <= 0 or duration < 0):
        raise ValueError('Actual A0..4, positive regular physical Phi bound and ordinary duration required')
    N = int(descriptor['N'])
    V = N**3
    ue = Fraction(descriptor['physical_family']['UE'])
    speed = Fraction(descriptor['physical_family']['c'])
    UE, c = arb(ue.numerator)/ue.denominator, arb(speed.numerator)/speed.denominator
    scalar_sum = 1.01*sum(float(np.sum(abs(s))) for _, s in current.paths)
    # Both orientations and four singular values. The actual two-Phi
    # endpoint average is bounded by the SAME physical Phi operator cap.
    S = 8*arb(V)*arb(physical_Phi_operator_upper)*arb(scalar_sum)
    displacement = 8*(UE/c).sqrt()
    z = S*arb(duration)
    stirling = ((1,), (0, 1), (0, 1, 1), (0, 1, 3, 1), (0, 1, 7, 6, 1))
    bell = [sum((arb(coef)*z**power for power, coef in enumerate(row)), arb(0))
            for row in stirling]
    initial = [arb(value) for value in initial_annihilator_tensor_bounds]
    bounds = [sum((math.comb(k, ell)*initial[ell]*displacement**(k-ell)*bell[k-ell]
                   for ell in range(k+1)), arb(0)) for k in range(5)]
    graph = (bounds[0]**2+15*bounds[1]**2+25*bounds[2]**2
             +10*bounds[3]**2+bounds[4]**2).sqrt()
    return dict(scope='ordinary exact regular/Dirichlet H_D allmode graph; '
                      'whole-to-Dir weighted interpolation and Noether flow separate',
        duration=duration, physical_Phi_operator_upper=physical_Phi_operator_upper,
        coefficient_sum_upper=S.str(65),
        range2_displacement_over_sqrt_UE_upper=(8/c.sqrt()).str(65),
        initial_A0_to_A4=[value.str(65) for value in initial],
        propagated_A0_to_A4=[value.str(65) for value in bounds],
        Nplus1_squared_graph_upper=graph.str(65),
        graph_domain='D((Ncoexact+1)^2), no photon truncation; regular Dirichlet '
                     'boundary/cutoff depend only on Phi/SU4 and commute every a_m',
        hierarchy='A_k(t)<=sum_l binom(k,l) A_l(0) d_g^(k-l) Touchard_(k-l)(S t)',
        exact_commutator='[a_tensor^k,H_Peierls]=sum_j binom(k,j) '
                         'F_tensor^(j) a_tensor^(k-j), ||F^(j)||<=S d_g^j; '
                         'free oscillator frequencies removed unitarily',
        massive_fast_Pi_graph_used_as_photon_bound=False,
        source_transfer='[a_m,Bplus/minus]=0 exactly; '
                        'A_k(normalized BXi)<=CQ/Z A_k(Xi), CQ16sqrtV retained',
        true_creator_transfer='A_k(a_gdagger psi)<=sqrt(A_(k+1)^2+A_k^2)'
                              '+k A_(k-1), then divide actual creator norm',
        weighted_normal_Cauchy='||q sqrt(N+1)psi||^2<=||qpsi|| r ||(N+1)psi||; '
                              '||chi Pi sqrt(N+1)psi||^2<=chi^2 '
                              '||Pi^2 psi|| ||(N+1)psi||',
        no_unbounded_operator_applied_to_norm_residual=True,
        Lambda_independent_when_actual_initial_Ak_and_Phi_cap_are=True,
        physical_probability_certified=False)



def harmonic_kinetic_leading_enclosure(descriptor: dict[str, Any], current: PreparedCentredCurrent,
                                      metric_radius: float,
                                      velocity_radius: float = 0.) -> dict[str, Any]:
    """Unbounded p_theta² insertion on the full-CAR Gaussian mean graph.

    This is a graph-domain estimate, not ||H_light(theta)-H_light(0)|| Duhamel.
    Actual filtered/Phi/Root initial graph transfer remains separately owned.
    """
    ctx.prec = 256
    N = int(descriptor['N'])
    V = N**3
    ue = Fraction(descriptor['physical_family']['UE'])
    UE = arb(ue.numerator)/ue.denominator
    gram = np.real(np.einsum('mij,nji->mn', current.frame, current.frame))/4
    g0, gs = float(gram[0, 0]), float(gram[1, 1])
    t1 = 1.01*(current.band.r*math.sqrt(g0+metric_radius)
               +(current.band.v+velocity_radius)*math.sqrt(gs+metric_radius)/2)
    t2 = 1.01*current.band.r*math.sqrt(g0+metric_radius)/4
    occupation = 2*V+4
    derivative_h = arb(occupation)*2*(t1+2*t2)/N
    derivative2_h = arb(occupation)*2*(t1+4*t2)/N**2
    # At most eight T6 flight lengths, two source words and two current/contact
    # factors on a leading contour. SOURCE theta words have |d_j/N|<=1;
    # the order-one common Birth-to-Ref phase is NOT discarded.
    first = 4+8*CLOCK*derivative_h+arb(16)/N
    second = (4+8*CLOCK*derivative2_h+64*CLOCK**2*derivative_h**2
              +first**2+arb(16)/N**2)
    # ||p² g||=sqrt(15)/2 and ||theta_j g||=1/sqrt(2).
    graph = arb(15).sqrt()/2+3*arb(2).sqrt()*first+3*second
    word_error = 8*CLOCK*UE*graph/N
    incoming, outgoing = photon_packets(descriptor)
    fi, fo = (1.01*sum(abs(row['coefficient']) for row in packet['modes'])
              / math.sqrt(packet['modes'][0]['omega']) for packet in (incoming, outgoing))
    ji, jo = occupation*2*(t1+2*t2)*fi, occupation*2*(t1+2*t2)*fo
    contact = occupation*2*(t1+4*t2)*fi*fo
    factor = arb(3)**2/arb('.25')**2
    ccap = 128*factor*(CLOCK*contact+CLOCK**2*ji*jo)
    gcap, mcap = ccap, 4*factor
    ec, eg, em = 16*word_error*ccap, 16*word_error*gcap, 16*word_error*mcap
    delta = 32*word_error*factor
    if not bool(delta < arb('.25')):
        raise ArithmeticError('Harmonic kinetic graph comparison left the positive receiver tube')
    total = (2*ec+4*ccap*delta
             +4*(gcap*em+mcap*eg+eg*em)+24*gcap*mcap*delta)
    return dict(theta_coordinate_law='theta=N A_h; H_harm=UE/N p_theta², no theta potential',
        full_CAR_occupation=occupation,
        mean_gaussian_p2_graph_norm_upper=graph.str(65),
        harmonic_kinetic_word_relative_error=word_error.str(65),
        leading_C_relative_error=ec.str(65), leading_G_relative_error=eg.str(65),
        bare_overlap_relative_error=em.str(65), zeroth_GLOBAL_Gram_error=delta.str(65),
        GLOBAL_leading_relative_error=total.str(65),
        common_Birth_to_Ref_phase_derivatives_retained=True,
        photon_number_preserving=True, no_incoming_outgoing_transition_without_current=True,
        ordinary_CAR_or_flatRootP_SOURCE_upper=3, full_aux_Root_Q_bound3_proved=False,
        actual_filtered_initial_graph_bound_proved=False,
        conditional_scope='Gaussian harmonic mean full-CAR graph with actual SOURCE words '
                          '|Birth_j-Ref_j|/N<=1 and charged source norm>=1/4; '
                          'owner Phi/normal/filter/Root/initial graph and all-mode photon transfers required',
        physical_probability_certified=False)


def run_coulomb_comparison(descriptor_path: Path, metric_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    descriptor_bytes, metric_bytes = descriptor_path.read_bytes(), metric_path.read_bytes()
    descriptor = decode(json.loads(descriptor_bytes))
    current = PreparedCentredCurrent()
    core = metric_box(metric_bytes, current)
    result = coulomb_leading_enclosure(descriptor, current, core['metric_radius'], core['velocity_radius'])
    from improved_phi_band import ImprovedPhiLaw
    law = ImprovedPhiLaw(3, current.band)
    laplacian = np.zeros((law.volume, law.volume))
    for x in range(law.volume):
        for axis in range(3):
            y = law.target[x, axis]
            laplacian[x, x] += 1
            laplacian[y, y] += 1
            laplacian[x, y] -= 1
            laplacian[y, x] -= 1
    eigenvalues, basis = np.linalg.eigh(laplacian)
    positive = eigenvalues > 1e-10
    green = (basis[:, positive]/eigenvalues[positive]) @ basis[:, positive].T
    # Exact HC-invariant density sector: all12 CAR modes on three sites,
    # other24 sites n=2. This is NOT an H_light-invariant frozen spectator.
    active_sites = [law.index[(x, 1, 1)] for x in range(3)]
    occupations = np.indices((5,)*3, dtype=np.int8).reshape(3, -1).T
    spectrum = []
    bound = law.size**2*(4*law.volume+32)/16
    for reference_number, light_extra in ((4, 0), (0, 4), (3, 1), (1, 3)):
        selected = occupations[occupations.sum(axis=1) == 6+light_extra]
        rho = np.zeros((len(selected), law.volume))
        rho[:, active_sites] = selected.astype(float)-2
        missing_reference = float(np.min(abs(rho.sum(axis=1))))
        rho[:, law.index[(0, 1, 1)]] += reference_number-4
        energy = np.einsum('nx,xy,ny->n', rho, green, rho, optimize=True)
        if (np.max(abs(rho.sum(axis=1))) != 0 or np.min(energy) < -1e-11
                or np.max(energy) > bound+1e-11 or missing_reference != light_extra):
            raise AssertionError('Native fixed-bank full-Green Coulomb/reference bound failed')
        spectrum.append(dict(reference_number=reference_number,
            reference_charge=reference_number-4, light_number=2*law.volume+light_extra,
            complete_density_configurations=len(selected), HC_over_UE_min=float(np.min(energy)),
            HC_over_UE_max=float(np.max(energy)), rigorous_majorant=bound,
            missing_reference_Gauss_total_charge=missing_reference))
    if descriptor_path.read_bytes() != descriptor_bytes or metric_path.read_bytes() != metric_bytes:
        raise RuntimeError('Legal Coulomb comparison inputs changed during execution')
    return dict(topic='NEW full-CAR HC leading-relative correction; NOT physical Born promotion',
        coulomb_leading_enclosure=result,
        native_fullGreen_invariant_density_spectrum=spectrum,
        native_Coulomb_scope='N3 full27-site Green/108CAR HC factor; all12 active-site CAR, '
                            'other24 sites fixed n=2 invariant ONLY for HC; not full light H/dynamics',
        descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        metric_sha256=hashlib.sha256(metric_bytes).hexdigest(),
        physical_probability_certified=False, natural_exit=0, actual_seconds=time.perf_counter()-start)


def metric_box(packet_bytes: bytes, current: PreparedCentredCurrent) -> dict[str, Any]:
    """Bind the actual finite-N heavy core and exact common-c light coefficient."""
    certificate = json.loads(packet_bytes)['metric_certificate']
    box = certificate['scalar_box']
    gram = np.einsum('mij,nji->mn', current.frame, current.frame).real / 4
    errors = []
    for index, name in enumerate(('g0', 'gs', 'gs', 'gs')):
        trial = Fraction.from_float(float(gram[index, index]))
        lo, hi = Fraction(box[name+'_lower']), Fraction(box[name+'_upper'])
        errors.append(max(abs(trial-lo), abs(trial-hi)))
    alias = Fraction(certificate['full10_G_N_minus_G_infinity_upper'])
    radius = math.nextafter(float(sum(errors)+alias), math.inf)
    radius += 64*math.ulp(float(np.max(abs(gram))))
    ctx.prec = 256
    speed = arb(1130202988249777)/9007199254740992
    gslo, gshi = Fraction(box['gs_lower']), Fraction(box['gs_upper'])
    vl = speed/(arb(gshi.numerator)/gshi.denominator).sqrt()
    vu = speed/(arb(gslo.numerator)/gslo.denominator).sqrt()
    vt = legal.binary64_ball(current.band.v)
    velocity = abs(vt-vl)+abs(vt-vu)
    return dict(metric_radius=radius,
                velocity_radius=math.nextafter(float(velocity.upper()), math.inf),
                source_sha256=hashlib.sha256(packet_bytes).hexdigest(),
                actual_core='finite-N full10 G_N, NOT G_infinity assigned as exact centre',
                light_velocity='exact c/sqrt(G_infinity,11), numeric trial enclosed',
                scalar_box=box)


def native_harmonic_ward(descriptor: dict[str, Any], current: PreparedCentredCurrent) -> dict[str, Any]:
    """Full108 literal mean-law/Wick stress, NOT the cached native Xi or full H."""
    from improved_phi_band import ImprovedPhiLaw
    from noncompact_screened_gauss_dressing import screened_marks
    from prepared_neutral_full_field import BETA

    law = ImprovedPhiLaw(3, current.band)
    phi = np.tile(current.frame, (law.volume, 1, 1, 1))
    links = np.tile(np.eye(4), (law.volume, 3, 1, 1))
    zero = np.zeros((law.volume, 3))
    theta = np.array([.7, -.4, .3])
    harmonic = np.tile(theta/law.size, (law.volume, 1))
    h0, h = law.onebody(phi, links, zero), law.onebody(phi, links, harmonic)
    values0, basis0 = np.linalg.eigh(h0)
    occupied = basis0[:, values0 < 0]
    P0 = occupied @ occupied.conj().T
    values, basis = np.linalg.eigh(h)
    free = (basis*np.exp(-1j*CLOCK*values)) @ basis.conj().T
    Pt = free @ P0 @ free.conj().T
    Qt = np.eye(len(h))-Pt
    modes = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))
    currents = []
    for mode in modes:
        phase = 2*math.pi*(law.sites @ np.asarray(mode))/law.size
        cosine, sine = np.zeros_like(zero), np.zeros_like(zero)
        cosine[:, 2], sine[:, 2] = np.cos(phase), np.sin(phase)
        currents.append(law.onebody(phi, links, harmonic, cosine)
                        +1j*law.onebody(phi, links, harmonic, sine))
    loops = np.array([[np.trace(Pt @ left.conj().T @ Qt @ right)
                       for right in currents] for left in currents])
    off = float(np.max(abs(loops[:2, 2:])))
    diagonal = float(np.min(np.real(np.diag(loops))))
    noncommuting = float(np.linalg.norm(h @ P0-P0 @ h))
    if off > 1e-11 or diagonal <= 1e-7 or noncommuting <= 1e-4:
        raise AssertionError('Full108 distinct-mode Ward/nonstationary positive control failed')
    profile = np.zeros(law.volume, complex)
    for offset, eta in zip(descriptor['compact_displacement'], descriptor['source_eta']):
        profile[law.index[tuple((np.asarray(offset, int)+1) % 3)]] += eta
    reference = law.index[(0, 1, 1)]
    marks = screened_marks(law, phi, links, zero, profile, reference, 18.,
                           harmonic=theta/law.size).reshape(4, law.volume, 4, 4)/2
    source = np.einsum('a,axir->xir', BETA, marks).reshape(4*law.volume, 4)
    input_norm = float(np.trace(source.conj().T @ (np.eye(len(h))-P0) @ source).real)
    endpoint_norm = float(np.trace(source.conj().T @ Qt @ source).real)
    instantaneous = basis[:, values < 0] @ basis[:, values < 0].conj().T
    instantaneous_error = float(np.linalg.norm(Pt-instantaneous))
    if input_norm <= 0 or instantaneous_error <= 1e-4:
        raise AssertionError('Actual raw source/nonstationary projector control failed')
    return dict(
        exact_mode_ids=modes, light_dimension=len(h), occupied_count=len(occupied.T),
        clock=CLOCK, theta=theta, extensive_closed_loop_Gram=loops,
        distinct_in_out_extensive_loop_max=off, same_mode_positive_control_min=diagonal,
        h_theta_P0_commutator_norm=noncommuting,
        wrong_instantaneous_P_theta_error=instantaneous_error,
        original_FULLRef_raw_SOURCE_input_norm2=input_norm,
        original_FULLRef_raw_SOURCE_endpoint_norm2=endpoint_norm,
        SOURCE_zeroth_GLOBAL_Gram_ratio=endpoint_norm/input_norm,
        common_birth_phase_retained_in_original_SOURCE=True,
        scope='full108 literal uniform-Phi mean h/current and original SOURCE at frozen theta; '
              'evolving initial whole negative h0 cluster, NOT actual cached Gamma0-complement Xi, '
              'Coulomb/Phi kinetics/normal feedback/coupled T6 or physical probability')


def run_harmonic_comparison(descriptor_path: Path, metric_path: Path) -> dict[str, Any]:
    """New analytic legal harmonic payment plus full108 nonstationary Ward stress."""
    start = time.perf_counter()
    descriptor_bytes, metric_bytes = descriptor_path.read_bytes(), metric_path.read_bytes()
    descriptor = decode(json.loads(descriptor_bytes))
    current = PreparedCentredCurrent()
    core = metric_box(metric_bytes, current)
    enclosure = harmonic_leading_enclosure(
        descriptor, current, core['metric_radius'], core['velocity_radius'])
    native = native_harmonic_ward(descriptor, current)
    if descriptor_path.read_bytes() != descriptor_bytes or metric_path.read_bytes() != metric_bytes:
        raise RuntimeError('Legal harmonic comparison inputs changed during execution')
    return dict(topic='NEW structured harmonic mean comparison; NOT physical Born promotion',
        legal_harmonic_leading_enclosure=enclosure, full108_native_Ward=native,
        descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        metric_sha256=hashlib.sha256(metric_bytes).hexdigest(),
        actual_core_box=core, physical_probability_certified=False,
        frozen_q128_promoted=False, natural_exit=0, actual_seconds=time.perf_counter()-start)


def run(descriptor_path: Path, quadrature: int,
        metric_packet: Path | None = None, branch: str = 'plus') -> dict[str, Any]:
    start = time.perf_counter()
    before = source_hashes()
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor = decode(json.loads(descriptor_bytes))
    metric_bytes = metric_packet.read_bytes() if metric_packet else None
    current = PreparedCentredCurrent()
    means = comparison_means(descriptor, current, quadrature, branch)
    enclosure, core_box = None, None
    if metric_bytes is not None:
        core_box = metric_box(metric_bytes, current)
        enclosure = comparison_enclosure(descriptor, current, means,
            core_box['metric_radius'], core_box['velocity_radius'])
        if metric_packet and metric_packet.read_bytes() != metric_bytes:
            raise RuntimeError('Finite-N metric input changed during execution')
    after = source_hashes()
    if before != after:
        raise RuntimeError('Owned current or consumer dependencies changed during execution')
    if descriptor_path.read_bytes() != descriptor_bytes:
        raise RuntimeError('Source19 geometry input changed during execution')
    return dict(topic='NEW v3W finite-neutral-endpoint same-current scattering',
                owned_consumer='physical_fullfield_scattering.py',
                current_owner=current.metadata(), comparison=means,
                actual_core_box=core_box, new_current_enclosure=enclosure,
                clock=CLOCK, source_sites=len(descriptor['compact_displacement']),
                source_word=f"D+{int(descriptor['source_coordinate_tree_word_length_max'])-int(descriptor['N'])//2}",
                eta=descriptor['screened_eta'], alpha=descriptor['screened_alpha_expression'],
                receiver_transition='NEW STATIC GLOBAL relative-protocol receiver within '
                                    'each specified reference/operator family; not old E or P',
                common_physical_reference_hardware_proved=False,
                missing_physical_claim='actual v3W state/source/current/receiver and physical generator '
                                       'consumer must be joined; one Ref4-slot bank uses different '
                                       'plus/minus charge operators and does not prove common Xi/static E2',
                source_hashes=before,
                descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
                natural_exit=0, actual_seconds=time.perf_counter()-start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--quadrature', type=int, default=32)
    parser.add_argument('--metric-packet', type=Path)
    parser.add_argument('--native-state-capsule', type=Path,
                        help='SOURCE108/fullGaussian native input only; no scattering/H cutover')
    parser.add_argument('--harmonic-point-bound', action='store_true',
                        help='New structured mean-law harmonic bound/Ward stress, not physical Born')
    parser.add_argument('--coulomb-point-bound', action='store_true',
                        help='True full-CAR Coulomb leading-relative bound, not physical Born')
    parser.add_argument('--branch', choices=('plus', 'minus'), default='plus')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.coulomb_point_bound:
        if args.native_state_capsule or args.harmonic_point_bound or args.metric_packet is None:
            parser.error('--coulomb-point-bound requires --metric-packet and excludes other native/proof modes')
        result = run_coulomb_comparison(args.descriptor, args.metric_packet)
    elif args.harmonic_point_bound:
        if args.native_state_capsule or args.metric_packet is None:
            parser.error('--harmonic-point-bound requires --metric-packet and excludes --native-state-capsule')
        result = run_harmonic_comparison(args.descriptor, args.metric_packet)
    elif args.native_state_capsule:
        result = run_native_source_embedding(args.descriptor, args.native_state_capsule)
    else:
        result = run(args.descriptor, args.quadrature, args.metric_packet, args.branch)
    encoded = json.dumps(result, default=encode, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
