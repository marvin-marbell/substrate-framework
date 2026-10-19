#!/usr/bin/env python3
"""Actual ALL36 paired particle/removal photon-source and current consumer.

Uses captured native UE1/cut1 quantum eigenvectors and original ALL sigma,
with operator-action recovered sector tags; no repeated SVD, tail pruning,
new Reference flavour, fitted speed or current-to-mass interpretation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
import scipy
import covariant_native_collective_primitive_electric as base
import covariant_collective_spectral_recovery as recovery
import covariant_collective_paired_car as car

ROOT = Path(__file__).resolve().parent
SOURCE_CAP_PATH = ROOT / 'covariant_native_collective_primitive_electric_cut1_UE1_capsule_20261008.json'
SOURCE_CAP_BYTES_AT_IMPORT = SOURCE_CAP_PATH.read_bytes()
SOURCE_CAP_SHA = '6000a6b4a6d9e6e91c9b1f0c607a3b4708c41c878a7aa24cf08ff17182412b05'
if hashlib.sha256(SOURCE_CAP_BYTES_AT_IMPORT).hexdigest() != SOURCE_CAP_SHA:
    raise RuntimeError('Frozen actual N2 source capsule bytes changed')
PLAN = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT = PLAN.read_bytes()
ARGUMENT_INPUT_BYTES_AT_IMPORT: dict[str, bytes] = {}
for i, arg in enumerate(sys.argv[:-1]):
    if arg in ('--labels', '--car-proof'):
        ARGUMENT_INPUT_BYTES_AT_IMPORT[arg] = Path(sys.argv[i + 1]).read_bytes()
SOURCE_BYTES_AT_IMPORT: dict[str, bytes] = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if isinstance(filename, str):
        path = Path(filename)
        if path.suffix == '.py' and path.resolve().parent == ROOT:
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()


def complex_array(value: np.ndarray) -> dict[str, Any]:
    return dict(real=np.real(value).tolist(), imag=np.imag(value).tolist())


def photon_state(n: base.Spectrum, neutral: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    vacuum = np.zeros(len(n.q), dtype=complex); vacuum[0] = 1.
    Omega = n.wave(vacuum)
    qn = n.q / n.q[0]
    projected = n.project(base.collective.native.curl(neutral, 3) @ Omega)
    ph = qn * projected
    Z = float(np.vdot(ph, ph).real)
    if Z <= 0:
        raise ArithmeticError('Actual ALL36 neutral photon source vanished')
    return Omega, n.wave(ph / math.sqrt(Z)), Z


def current_case(c: base.Spectrum, physical: dict[str, Any], S: Any, Sout: Any,
                 Omega: np.ndarray, incoming: np.ndarray, lambdaN: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """j_e=-i[H_transfer,E_e], full field and its Gauss density connection."""
    qc = c.q / math.sqrt(lambdaN)
    energy = -2 * np.log(qc)
    raw = (S @ incoming).reshape(4, c.dimension)
    Xi = qc[None, :] * np.array([c.project(w) for w in raw])
    M0 = float(np.vdot(Xi, Xi).real)
    if M0 <= 0:
        raise ArithmeticError('Actual coupled source/current preparation vanished')
    sourcewave = np.array([c.wave(w) for w in Xi])
    Hsourcewave = np.array([c.wave(energy * w) for w in Xi])
    target = (Sout @ Omega).reshape(4, c.dimension)
    B = base.collective.native.curl(physical, 3)
    receiver = np.array([c.project(B @ w) for w in target])
    electric = np.array([E for _, _, E in physical['physical']])
    currents = []
    for edge in range(4):
        Efield = np.tile(electric[:, edge], 3)
        EXi = np.array([c.project(Efield * w) for w in sourcewave])
        EHXi = np.array([c.project(Efield * w) for w in Hsourcewave])
        currents.append(-1j * (energy[None, :] * EXi - EHXi))
    currents_array = np.array(currents)
    continuity = []
    rho = np.array([base.collective.density(light, aux1, aux2, physical['charge'])
                    for state, _, _ in physical['physical']
                    for light, aux1, aux2 in (physical['states'][state],)])
    for site in range(4):
        density = np.tile(rho[:, site], 3)
        rhoXi = np.array([c.project(density * w) for w in sourcewave])
        rhoHXi = np.array([c.project(density * w) for w in Hsourcewave])
        dotrho = 1j * (energy[None, :] * rhoXi - rhoHXi)
        residual = dotrho + currents_array[site] - currents_array[(site - 1) % 4]
        continuity.append(float(np.linalg.norm(residual) / math.sqrt(M0)))
    vertices = np.einsum('rj,erj->ej', Xi.conj(), currents_array) / M0
    times = []
    for elapsed in (0., .5, 1., 2.):
        phase = np.exp(-1j * elapsed * energy)
        receiver_insertions, source_insertions = [], []
        for current, vertex in zip(currents_array, vertices):
            receiver_insertions.append(complex(np.vdot(receiver, current * phase[None, :]) / math.sqrt(M0)))
            source_insertions.append(complex(np.sum(vertex * phase)))
        times.append(dict(time=elapsed,
                          coherent_all4Ref_receiver_current_insertions=complex_array(np.array(receiver_insertions)),
                          normalized_source_current_insertions=complex_array(np.array(source_insertions))))
    result = dict(actual_M0=M0, actual_ALL_charged_energies=energy.tolist(),
                  actual_ALL4edge_ALL4Ref_current_source_amplitudes=complex_array(currents_array),
                  actual_current_source_spectral_vertices=complex_array(vertices),
                  actual_ALL4site_Gauss_continuity_relative_residuals=continuity,
                  actual_same_literal_receiver_current_times=times,
                  current_contract='j_e=-i[H_transfer,Ephys_e], H=-log(KC/lambdaN), ALL4edge Gauss integerE including longitudinal density connection. Current insertion at source in one-sided transfer GF/receiver, NOT full Dirac CAR retarded function or bare-electron velocity.',
                  vertex_contract='W_e,j=sum_ref conj(Xi_ref,j) (j_e Xi)_ref,j /M0; no spectral row, tail or Reference channel dropped')
    return result, dict(Xi=Xi, currents=currents_array, sourcewave=sourcewave, receiver=receiver)


def source_boundaries(gamma: np.ndarray, embedding: np.ndarray, neutral: dict[str, Any],
                      addition: dict[str, Any], removal: dict[str, Any], Omega: np.ndarray,
                      incoming: np.ndarray, packet: np.ndarray, receiver: np.ndarray) -> dict[str, Any]:
    """Complete literal-Wilson image of FIXED cut1 seeds in cut2; not Omega∞."""
    neutral2 = base.space(0, 2)
    nr1, nr2 = len(neutral['physical']), len(neutral2['physical'])
    neutral_rows = np.array([neutral2['index'][(s, n)] for s, n, _ in neutral['physical']])
    for old, new in zip(neutral['physical'], neutral_rows):
        if neutral2['physical'][new][2] != old[2]:
            raise ArithmeticError('Larger source-image regulator changed a fixed neutral electric state')
    seeds = {}
    for name, wave in (('fixed_actual_Omega_cut1', Omega), ('fixed_actual_photon_seed_cut1', incoming)):
        expanded = np.zeros((3, nr2), dtype=complex)
        expanded[:, neutral_rows] = wave.reshape(3, nr1)
        seeds[name] = expanded.reshape(-1)
    Ebar = car.antiparticle_embedding(embedding)
    results = []
    for charge, small in ((1, addition), (-1, removal)):
        large = base.space(charge, 2)
        nc1, nc2 = len(small['physical']), len(large['physical'])
        retained = np.array([large['index'][(s, n)] for s, n, _ in small['physical']])
        outside = np.ones(nc2, dtype=bool); outside[retained] = False
        for packet_name, coeff in (('ORIGINAL_source_packet', packet), ('ORIGINAL_receiver_packet', receiver)):
            if charge == 1:
                band = base.collective.native.source_coefficients(gamma, coeff)
                S1 = base.collective.source(neutral, small, band, embedding)
                S2 = base.collective.source(neutral2, large, band, embedding)
            else:
                band = car.negative_coefficients(gamma, coeff)
                S1 = car.annihilator(neutral, small, band, Ebar)
                S2 = car.annihilator(neutral2, large, band, Ebar)
            for seed_name, expanded in seeds.items():
                original_seed = Omega if seed_name == 'fixed_actual_Omega_cut1' else incoming
                original_image = (S1 @ original_seed).reshape(4, 3, nc1)
                image = (S2 @ expanded).reshape(4, 3, nc2)
                total = float(np.vdot(image, image).real)
                if total <= 0:
                    raise ArithmeticError('Complete literal source image unexpectedly has zero norm')
                lost = image[:, :, outside]
                results.append(dict(active_number=4 + charge, packet=packet_name, seed=seed_name,
                                    complete_cut2_source_image_norm_squared=total,
                                    source_boundary_loss_norm_squared=float(np.vdot(lost, lost).real),
                                    source_boundary_loss_fraction=float(np.vdot(lost, lost).real / total),
                                    retained_cut1_operator_image_absolute_defect=float(np.linalg.norm(image[:, :, retained] - original_image)),
                                    complete_charged_image_base_dimension=3 * nc2))
    return dict(rows=results,
                contract='Fixed actual neutral cut1 Omega/photon vectors, not recomputed atcut2. Every original Wilson path changes each edge by atmost1, so cut2 contains their COMPLETE image. Boundary loss is before charged halfstep, not projected Knull, not a bound on Omega∞, and not an energy/cutoff-convergence claim.')


def run(labels_path: Path, car_proof_path: Path) -> dict[str, Any]:
    label_bytes = ARGUMENT_INPUT_BYTES_AT_IMPORT['--labels']
    car_bytes = ARGUMENT_INPUT_BYTES_AT_IMPORT['--car-proof']
    # Inputs are the bytes captured at IMPORT, not a later completion snapshot.
    if labels_path.read_bytes() != label_bytes or car_proof_path.read_bytes() != car_bytes:
        raise RuntimeError('Input proof/label artifact changed after IMPORT')
    n, plus, recovery_controls = recovery.load_recovered(SOURCE_CAP_PATH, labels_path)
    gamma, embedding, nodes, heat, Hphi, metric = base.left.frame()
    neutral, addition, removal = base.space(0, 1), base.space(1, 1), base.space(-1, 1)
    Cbody = car.physical_conjugation(removal, addition, False)
    Cquartet = car.physical_conjugation(removal, addition, True)
    Cneutral = car.physical_conjugation(neutral, neutral, False)
    # This is the ALL-column conjugate physical kernel, not copied observables.
    minus_blocks = [dict(V=b['V'], q=b['q'], P=Cbody.T @ b['P'], sign=b['sign']) for b in plus.blocks]
    minus = base.Spectrum(minus_blocks, 3 * len(removal['physical']))
    packet = np.array([.1, .8, .4, .1]); packet /= np.linalg.norm(packet)
    outpacket = np.array([.5, .4, .1, .6]); outpacket /= np.linalg.norm(outpacket)
    Ebar = car.antiparticle_embedding(embedding)
    Sa = base.collective.source(neutral, addition, base.collective.native.source_coefficients(gamma, packet), embedding)
    Sa_out = base.collective.source(neutral, addition, base.collective.native.source_coefficients(gamma, outpacket), embedding)
    Sh = car.annihilator(neutral, removal, car.negative_coefficients(gamma, packet), Ebar)
    Sh_out = car.annihilator(neutral, removal, car.negative_coefficients(gamma, outpacket), Ebar)
    print(json.dumps(dict(stage='executing_actual_ALL36_Ref0_to1_reverse_Wilson_source_and_time_receiver',
                          complete_ranks=[len(n.q), len(plus.q), len(minus.q)])), flush=True)
    plus_result = base.observe(n, plus, neutral, addition, Sa, Sa_out, nodes, Hphi)
    minus_result = base.observe(n, minus, neutral, removal, Sh, Sh_out, nodes, Hphi)
    Omega, incoming, Z = photon_state(n, neutral)
    lambdaN = float(n.q[0] ** 2)
    plus_current, plus_state = current_case(plus, addition, Sa, Sa_out, Omega, incoming, lambdaN)
    minus_current, minus_state = current_case(minus, removal, Sh, Sh_out, Omega, incoming, lambdaN)
    neutral_overlap = complex(np.vdot(Omega, Cneutral @ Omega))
    neutral_orthogonal = float(np.linalg.norm(Cneutral @ Omega - neutral_overlap * Omega))
    mapped_sourcewave = (Cquartet @ minus_state['sourcewave'].reshape(-1)).reshape(4, plus.dimension)
    mapped_source = np.array([plus.project(w) for w in mapped_sourcewave])
    source_defect = float(np.linalg.norm(mapped_source - neutral_overlap * plus_state['Xi']) / math.sqrt(plus_result['actual_M0']))
    current_defects = []
    for plus_j, minus_j in zip(plus_state['currents'], minus_state['currents']):
        physical_j = np.array([minus.wave(w) for w in minus_j])
        mapped_j = (Cquartet @ physical_j.reshape(-1)).reshape(4, plus.dimension)
        projected_j = np.array([plus.project(w) for w in mapped_j])
        current_defects.append(float(np.linalg.norm(projected_j + neutral_overlap * plus_j) / math.sqrt(plus_result['actual_M0'])))
    original = json.loads(SOURCE_CAP_BYTES_AT_IMPORT)['actual_primitive_electric_ALL36_transfer']
    reproduction: dict[str, Any] = {key:dict(original=original[key], actual_recovered_consumer=plus_result[key],
                             absolute_difference=abs(plus_result[key] - original[key]))
                    for key in ('actual_photon_Zph', 'actual_photon_energy_mean', 'actual_M0', 'actual_E1', 'actual_R1',
                                'actual_energy_mean', 'actual_transfer_Rayleigh_mean')}
    reproduction['coherent_timed_receiver_probabilities'] = [dict(time=new['time'], original=old['coherent_Ref4_receiver_probability'],
            actual_recovered_consumer=new['coherent_Ref4_receiver_probability'],
            absolute_difference=abs(new['coherent_Ref4_receiver_probability'] - old['coherent_Ref4_receiver_probability']))
            for old, new in zip(original['actual_same_literal_source_consumer_rows'], plus_result['actual_same_literal_source_consumer_rows'])]
    descriptor = dict(plan_id=PLAN.stem, raw_plan_sha256_at_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest(),
        source_N2_capsule_sha256=SOURCE_CAP_SHA, source_N2_input_sha256='763dd7e68928c2ea03c02bb6e18f619c75ee708041e556eb7daa2f8c4182c42d',
        recovered_label_capsule_sha256=hashlib.sha256(label_bytes).hexdigest(),
        actual_ALL36_CAR_operator_proof_capsule_sha256=hashlib.sha256(car_bytes).hexdigest(),
        total_dynamic_charged_flavours_including_tagged_active=3, auxiliary_copies=2,
        auxiliary_modes_each=4, auxiliary_number_each=2, ALL_auxiliary_states=36,
        BOTH_exact_copy_sectors=[21,15], active_modes=8, active_numbers=[3,4,5], active_states=[56,70,56],
        primitive_electric_input=1., NEW_electric_cutoff=1, actual_field_grid=32,
        kappa_active=.09, kappa_aux_each=.04, Phi3='three levels ONE coordinate, SAME oldconditional κ400 mean/metric',
        particle_reference='SAME4RefCAR; oldneutralRef4/background4→Ref3 atstatic0, originalSf/source unchanged',
        removal_reference='SAME4RefCAR; NEWneutralRef0/background0→Ref1 atstatic0; r† W_inverse a_negative; no creation onfilledRef4',
        neutral_identity='Ref occupation minus background zero in BOTHneutral protocols; SAME neutralbody/aux/Phi/electric kernel',
        removal_kernel='ALL-column full20CAR Hodge/bodyspin-swap/globalE reversal intertwines N3/N5 primitiveelectric/Phi3 kernels, SAME actual lambdaN. Actual N3 source and receiver amplitudes independentlyevaluated; no echoed N5 observables.',
        source_packet=packet.tolist(), receiver_packet=outpacket.tolist(),
        original_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        removal_paths=[[0,0,0,0],[1,0,0,0],[1,1,0,0],[0,0,0,-1]],
        antiparticle_embedding='Ebar=E.conj()@sigma_x', Pe=[.25,2.,4.], times=[0.,.5,1.,2.],
        current_definition='-i[H_transfer,Ephys_e], H=-log(KC/lambdaN), ALL4edge longitudinal/globalGauss retained',
        actual_original_ALL_sigma_retained=True, no_repeated_SVD=True, no_spectral_clipping_or_pruning=True,
        scope='Native compactone-cycle/all36CAR/finitecut/Phi3 selectedCliff2; NOT originalrealEM/fullCartesianPhi/SU4/largeN/massless/commoncone/mass/dispersion/CARanticommutator admission',
        source_image_Galerkin_cutoff=2,
        source_image_role='COMPLETE literal-Wilson image of FIXED cut1 seeds only; no cut2 K/Omega or global regulator convergence inference',
        source_boundary_and_cutoff_bias_NOT_inferred_from_tiny_projected_Knull=True)
    boundaries = source_boundaries(gamma, embedding, neutral, addition, removal, Omega, incoming, packet, outpacket)
    return dict(input_descriptor=descriptor,
                input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
                source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
                source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
                plan_bytes_at_import=PLAN_BYTES_AT_IMPORT.decode(), physical_mean_descriptor=metric,
                runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                actual_recovery_controls=recovery_controls,
                actual_same_source_reproduced_N5_consumer=reproduction,
                actual_complete_fixed_seed_source_boundary_images=boundaries,
                actual_particle_photon_source_and_timed_receiver=plus_result,
                actual_Ref0_to1_removal_photon_source_and_timed_receiver=minus_result,
                actual_particle_full_field_current=plus_current,
                actual_removal_full_field_current=minus_current,
                actual_neutral_C_overlap=complex_array(np.array(neutral_overlap)),
                actual_neutral_C_orthogonal_residual=neutral_orthogonal,
                actual_C_even_source_relative_defect=source_defect,
                actual_ALL4edge_C_odd_current_relative_defects=current_defects,
                actual_same_neutral_photon_Zph=Z,
                induced_electric_or_massless_commoncone_macro_admitted=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--car-proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result = run(args.labels, args.car_proof)
    result['runtime_seconds'] = time.monotonic() - start
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(source_at_import=result['source_at_import'], input_descriptor_sha256=result['input_descriptor_sha256'],
                         runtime_seconds=result['runtime_seconds'],
                         particle={key:result['actual_particle_photon_source_and_timed_receiver'][key] for key in ('actual_M0','actual_E1','actual_R1','actual_energy_mean')},
                         removal={key:result['actual_Ref0_to1_removal_photon_source_and_timed_receiver'][key] for key in ('actual_M0','actual_E1','actual_R1','actual_energy_mean')},
                         actual_C_even_source_relative_defect=result['actual_C_even_source_relative_defect'],
                         actual_ALL4edge_C_odd_current_relative_defects=result['actual_ALL4edge_C_odd_current_relative_defects'])), flush=True)


if __name__ == '__main__':
    main()
