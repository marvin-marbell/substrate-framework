#!/usr/bin/env python3
"""Source-weighted bounded receiver frames; no global bare-Haar CCR.

The exact API acts on Hilbert-Schmidt purifiers, so microscopic field/Fock
coherence, spectator traces and all source phases survive BEFORE the Gram.
The primitive bounded-field commutator is evaluated, never replaced by delta.
Gaussian comparison data and actual microscopic data have separate labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent
OWN_BYTES = Path(__file__).read_bytes()
Action = Callable[[int, np.ndarray], np.ndarray]


def joint_receiver_frame(xi: np.ndarray, source: Action, annihilate: Action,
                         create: Action, channels: int) -> dict[str, Any]:
    """xi MUST already be the actual U_N(t) Xi0, not a returned vacuum.

    source(j,xi) includes the original microscopic source and declared physical
    spin/Ref/output-electron effect, including both original charged boundaries.
    create/annihilate are actual adjoints on the common full output codomain.
    No energy/PH/range/ground projection is inserted by this function.
    """
    v = [source(j, xi) for j in range(channels)]
    y = [create(j, v[j]) for j in range(channels)]
    gram = np.array([[np.vdot(a, b) for b in y] for a in y])
    # N_jk = <A_k v_j, A_j v_k>; generally NOT positive as a matrix.
    number = np.array([[np.vdot(annihilate(k, v[j]), annihilate(j, v[k]))
                        for k in range(channels)] for j in range(channels)])
    commutator = np.array([[np.vdot(v[j], annihilate(j, create(k, v[k]))
                                   - create(k, annihilate(j, v[k])))
                           for k in range(channels)] for j in range(channels)])
    return dict(vectors=y, Gram=gram, N=number, actual_commutator_insertion=commutator,
                CCR_decomposition_defect=float(np.linalg.norm(gram-commutator-number)),
                source_Gram=np.array([[np.vdot(a, b) for b in v] for a in v]),
                primitive_global_CCR_asserted=False)


def normalized_receiver_born(frame: dict[str, Any], actual_target: np.ndarray,
                             original_input_Z: float) -> dict[str, Any]:
    """One GLOBAL receiver whitening, same target/source/dynamics as Gram."""
    g = frame['Gram']; values, vectors = np.linalg.eigh(g)
    if values[0] <= 0 or original_input_Z <= 0:
        raise ArithmeticError('Actual receiver Gram/input norm must be positive')
    inverse_root = (vectors/np.sqrt(values))@vectors.conj().T
    r = np.array([np.vdot(y, actual_target) for y in frame['vectors']])
    amplitudes = inverse_root@r/original_input_Z
    return dict(raw_CTC=r, original_input_Z=original_input_Z,
                joint_Gram_eigenvalues=values, one_GLOBAL_inverse_root=inverse_root,
                probabilities=abs(amplitudes)**2,
                whitening_defect=float(np.linalg.norm(inverse_root@g@inverse_root-np.eye(len(g)))))


def source_weighted_frame_lower(reference_vectors: list[np.ndarray],
                                error_Gram_upper: np.ndarray) -> dict[str, Any]:
    """Given E†E <= supplied bound, sigma_min(Y)>=sigma_min(Yref)-||E||.

    E_j=(actual receiver_j-reference receiver_j) ON THE SAME MOVED SOURCE
    purifier, with ALL field/source phases. An unconditioned norm times a
    normalized global state, pointwise RN ratio or input M0/M1 is NOT this debt.
    The upper matrix must be established by the operator/source provider.
    This theorem neither assumes N>=0 nor discards N to get G>=D.
    """
    reference = np.array([[np.vdot(a, b) for b in reference_vectors]
                          for a in reference_vectors])
    a = float(np.linalg.eigvalsh(reference)[0])
    error = float(np.linalg.eigvalsh(error_Gram_upper)[-1])
    if error < 0:
        raise ArithmeticError('Error Gram upper bound cannot be negative')
    singular_lower = max(0., math.sqrt(max(0., a))-math.sqrt(error))
    return dict(reference_Gram=reference, reference_Gram_floor=a,
                source_weighted_family_error_norm_upper=math.sqrt(error),
                actual_Gram_floor_lower=singular_lower**2,
                conditional_on_supplied_error_Gram_bound=True,
                theorem='||Ya|| >= ||Yref a||-||Ea|| for every ||a||=1',
                no_RN_to_Loewner_or_N_drop=True)


def source_moment_receiver_lower(moments: np.ndarray, witness_Gram: np.ndarray,
                                 whitened_moment_error_upper: float = 0.) -> dict[str, Any]:
    """Finite source-moment representation; no new physical outcome gate.

    moments[a,j]=<w_a,Y_j>, WdagW=witness_Gram, for full actual moved Y.
    G >= moments† (WdagW)^-1 moments since W(WdagW)^-1Wdag is a
    proof projection <=I, NOT a filter on the state or receiver. Thus a
    source provider can establish an output floor from a small full-rank
    mixed moment matrix without supplying the full inserted Haar Gram.
    An optional error is an actual bound on whitened SOURCE MOMENTS,
    not a global state norm or input source admission.
    """
    eig, vectors = np.linalg.eigh(witness_Gram)
    if eig[0] <= 0 or whitened_moment_error_upper < 0:
        raise ArithmeticError('Independent witnesses and nonnegative moment debt required')
    whitened = ((vectors/np.sqrt(eig))@vectors.conj().T)@moments
    if whitened.shape[0] < whitened.shape[1]:
        raise ArithmeticError('Too few source moments to bound the whole receiver frame')
    singular = float(np.linalg.svd(whitened, compute_uv=False)[-1])
    lower = max(0., singular-whitened_moment_error_upper)**2
    return dict(moment_matrix=moments, witness_Gram=witness_Gram,
                whitened_moment_sigma_min=singular,
                actual_receiver_Gram_floor_lower=lower,
                source_weighted_moment_error_upper=whitened_moment_error_upper,
                lower_matrix=whitened.conj().T@whitened,
                projection_is_proof_only_not_PH_ground_or_range_gate=True)


def full125_receiver_contract(capsule: dict[str, Any], history: int) -> dict[str, Any]:
    """Literal output/input descriptor; do not replace 125 by selected slice5."""
    packet = capsule['connected_finite_packets_all_statistics'][history]
    return dict(input_descriptor=capsule['input_descriptor'],
                statistical_history=packet['statistical_history'],
                input_modes=packet['incoming_photon_modes'],
                outgoing_photon_modes=packet['outgoing_photon_modes'],
                receiving_electron_modes=packet['receiving_electron_modes'],
                physical_output_spins=2, primitive_Ref_holes=packet['primitive_Ref_modes'],
                carrier_centres=packet['actual_carrier_centres'],
                phase_contract=packet['carrier_phase_contract'],
                source_selection=packet['source_selection'],
                output_PH_sector_is_lower_effect_not_state_gate=True,
                actual_macro_provider_required=['U_N(t)Xi0', 'original source S and both charged halves',
                    'bounded outgoing packet/spin/Ref actions', 'same charged evolved target',
                    'original input normalization', 'source-weighted receiver error Gram OR full-rank mixed source moments'],
                Gaussian_comparison_NOT_actual_micro_provider=True)


def native_receiver_discriminator(provider: Any, gamma: np.ndarray, *, nodes: int = 7,
                                  steps: int = 12) -> dict[str, Any]:
    """Native normal/integerGauss model, NOT macro125 transfer.

    Output field acts on the extended physical integer-flux codomain BEFORE
    Gram; K-null output tails are retained. Ref spin effects are physical Ref4
    hole columns, not a new preparation. The original source is unchanged.
    """
    prepared = provider.native_normal_preparation(gamma, steps, nodes, True)
    rt = prepared['runtime']; summary = prepared['summary']
    neutral = provider.native_range(provider.two_spin_frame(gamma), .09, 32, 2, 0)
    nn = len(neutral['physical']); un = rt['neutral_range_isometry']
    hol = np.zeros((nn, nn), complex)
    for column, (occupied, cycle, _) in enumerate(neutral['physical']):
        row = neutral['basis_index'].get((occupied, cycle+1))
        if row is not None:
            hol[row, column] = 1
    curl = (hol-hol.conj().T)/(2j)
    ground = np.outer(rt['normal_ground'], rt['normal_ground'])
    pn = un.conj().T@np.kron(ground, np.eye(nn))@un
    ln = rt['neutral_eigenvalues']; lc = rt['charged_eigenvalues']
    weights = ln**steps; weights /= weights.sum()
    half = np.diag(np.sqrt(ln)); b = un.conj().T@np.kron(np.eye(nodes), curl)@un
    x0 = half@pn@np.diag(np.sqrt(weights))/math.sqrt(summary['normal_preparation_Gram'])
    xph = half@b.conj().T@x0/math.sqrt(summary['photon_retained_Gram'])
    density_defect = float(np.linalg.norm(xph@xph.conj().T-rt['photon_density']))
    charged = dict(states=rt['charged_Fock_occupations'])
    physical = rt['charged_physical_basis']; nc = len(physical)
    extended = provider.physical_basis(charged['states'], 4, 1)
    index = {(i, n): j for j, (i, n, _) in enumerate(extended)}; ne = len(extended)
    rows = []; cols = []; data = []
    for column, (occupied, cycle, _) in enumerate(extended):
        row = index.get((occupied, cycle+1))
        if row is not None:
            rows.append(row); cols.append(column); data.append(1.)
    shift = coo_matrix((data, (rows, cols)), shape=(ne, ne)).tocsr()
    field = (shift-shift.conj().T)/(2j)
    embedding = np.zeros((ne, nc))
    for column, (occupied, cycle, _) in enumerate(physical):
        embedding[index[occupied, cycle], column] = 1
    uc = np.kron(np.eye(2), np.kron(np.eye(nodes), embedding))@rt['charged_range_isometry']
    raw_charged = dict(states=charged['states'], body_charge=1, physical=extended,
                       basis_index=index)
    single, gauss_defect = provider.source_matrix(neutral, raw_charged,
                                                 rt['source_coefficients'])
    if gauss_defect != 0:
        raise ArithmeticError('Raw same-source Gauss intertwiner changed')
    raw_source_operator = np.zeros((2*nodes*ne, nodes*nn), complex)
    for ref in range(2):
        for node in range(nodes):
            raw_source_operator[(ref*nodes+node)*ne:(ref*nodes+node+1)*ne,
                                node*nn:(node+1)*nn] = single[ref*ne:(ref+1)*ne]
    raw_source_operator = raw_source_operator@un
    def photon(_channel: int, x: np.ndarray) -> np.ndarray:
        return np.stack([field@part for part in x.reshape(2*nodes, ne, -1)]).reshape(x.shape)
    def source(channel: int, xi: np.ndarray) -> np.ndarray:
        z = uc@(rt['source_half']@xi)
        # Fixed physical Ref4-hole effects transported through original embedding.
        z = z.reshape(2, nodes*ne, -1).copy(); z[1-channel] = 0
        return z.reshape(2*nodes*ne, -1)
    z_original = math.sqrt(summary['source_retained_Gram'])
    initial = rt['source_half']@xph
    rows_out = []
    static = joint_receiver_frame(xph, source, photon, photon, 2)
    # Fix proof witnesses ONCE at the original clock; never change outcomes.
    witness_coordinates = [int(np.argmax(abs(y).ravel())) for y in static['vectors']]
    for time in (0., 2., 8., float(steps)):
        moved = np.exp(1j*np.log(ln)*time)[:, None]*xph
        target = uc@(np.exp(1j*np.log(lc)*time)[:, None]*initial)
        frame = joint_receiver_frame(moved, source, photon, photon, 2)
        born = normalized_receiver_born(frame, target, z_original)
        moments = np.array([[y.ravel()[coordinate] for y in frame['vectors']]
                            for coordinate in witness_coordinates])
        moment_bound = source_moment_receiver_lower(moments, np.eye(2))
        witness_remainder = frame['Gram']-moment_bound['lower_matrix']
        if float(np.linalg.eigvalsh(witness_remainder)[0]) < -1e-12:
            raise ArithmeticError('Full receiver fails the source-moment projection inequality')
        errors = [a-bb for a, bb in zip(frame['vectors'], static['vectors'])]
        error_gram = np.array([[np.vdot(a, bb) for bb in errors] for a in errors])
        bound = source_weighted_frame_lower(static['vectors'], error_gram)
        delta_D = np.diag(np.diag(frame['source_Gram']))
        raw_source_norm = float(np.linalg.norm(raw_source_operator@moved)**2)
        pauli_dropped_norm = float(np.linalg.norm(rt['source_coefficients'])**2
                                   *np.linalg.norm(moved)**2)
        rows_out.append(dict(time=time, Gram=frame['Gram'], N=frame['N'],
            actual_commutator_insertion=frame['actual_commutator_insertion'],
            exact_bounded_decomposition_defect=frame['CCR_decomposition_defect'],
            wrong_global_CCR_Gram_defect=float(np.linalg.norm(frame['Gram']-delta_D-frame['N'])),
            wrong_N_drop_Gram_defect=float(np.linalg.norm(frame['Gram']-delta_D)),
            wrong_static_reference_Gram_defect=float(np.linalg.norm(frame['Gram']-static['Gram'])),
            moved_neutral_purifier_difference=float(np.linalg.norm(moved-xph)),
            raw_same_source_M0=raw_source_norm,
            wrong_negative_Pauli_drop_source_norm=pauli_dropped_norm,
            actual_negative_Pauli_blocking=pauli_dropped_norm-raw_source_norm,
            source_weighted_error_bound=bound, Born=born,
            fixed_source_moment_witness_indices=witness_coordinates,
            source_moment_receiver_lower=moment_bound,
            full_Gram_minus_witness_lower_min=float(np.linalg.eigvalsh(witness_remainder)[0]),
            raw_receiver_vectors_physical_rows=2*nodes*ne,
            photon_true_field_not_nonzero_range_compressed=True))
    if max(row['wrong_static_reference_Gram_defect'] for row in rows_out) <= 1e-10:
        raise ArithmeticError('Moved actual receiver does not discriminate static reference')
    if max(row['wrong_global_CCR_Gram_defect'] for row in rows_out) <= 1e-8:
        raise ArithmeticError('Bounded primitive does not discriminate fictitious CCR')
    if min(row['actual_negative_Pauli_blocking'] for row in rows_out) <= 1e-8:
        raise ArithmeticError('Native moved source cannot discriminate negative Pauli drop')
    if min(row['source_moment_receiver_lower']['actual_receiver_Gram_floor_lower']
           for row in rows_out) <= 0:
        raise ArithmeticError('Fixed source moments fail positive moving receiver floor')
    return dict(backend='ACTUAL native .09 normal+canonical integerGauss Wilson/Haar',
        macro125_provider=False, source_summary=summary, observations=rows_out,
        original_input_Z=z_original, reconstructed_original_photon_density_defect=density_defect,
        full_Ref4_hole_embedding=rt['Ref4_hole_embedding'], fullCAR_negative_Pauli_retained=True,
        photon_number_coherence='Both literal flux shift orientations retained before Gram',
        native_only_output_effects='Two physical Ref-hole components, SAME original scalar curl; NOT declared macro outgoing3/receive7 effects',
        source_input_admission_NOT_receiver_normalization=True, no_return_vacuum=True,
        no_new_ground_pole_PH_gate=True)


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-sources', type=Path, required=True)
    parser.add_argument('--nodes', type=int, default=7)
    args = parser.parse_args()
    import covariant_prepared_gauss_state as provider
    from covariant_induced_photon import Parameters, phi_frame
    capsule_bytes = args.capsule.read_bytes(); capsule = json.loads(capsule_bytes)
    gamma, _ = phi_frame(Parameters(kappa_power=400))
    native = native_receiver_discriminator(provider, gamma, nodes=args.nodes)
    args.runtime_sources.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for name, digest in provider.COMPILED.items():
        data = provider.SNAPSHOT[name]
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError('Executed provider source binding disagrees: '+name)
        path = args.runtime_sources/(name+'.py'); path.write_bytes(data)
        snapshots[name] = dict(path=str(path), sha256=digest)
    owned = args.runtime_sources/Path(__file__).name; owned.write_bytes(OWN_BYTES)
    result = dict(scope='Exact bounded receiver consumer API + moved-source native discriminator; actual macro125 provider OPEN',
        full125_contracts=[full125_receiver_contract(capsule, h) for h in range(8)], native=native,
        input_capsule_sha256=hashlib.sha256(capsule_bytes).hexdigest(),
        owned_source_sha256=hashlib.sha256(OWN_BYTES).hexdigest(),
        compiled_project_sources=provider.COMPILED,
        executed_source_snapshots=snapshots,
        provider_loaded_code_matches_snapshot=provider.MAIN_CODE_MATCH,
        microscopic_macro_receiver_Gram=None, microscopic_macro_positive_lower=None,
        physical_Born_Maxwell_closure='OPEN')
    args.output.write_text(json.dumps(json_value(result), indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), native_rows=len(native['observations']),
                         physical_closure='OPEN', natural_exit=0)))


if __name__ == '__main__':
    main()
