#!/usr/bin/env python3
"""ALL-rank fixed-frame current/contact decomposition and diagnostic spectral tails.

analyze is regulator-generic. Energy partitions are observations, never gates or
bounds. Positive imaginary-time external conjugation is not on-shell scattering.
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
from scipy.sparse.linalg import expm_multiply
import covariant_native_collective_primitive_electric as base
import covariant_collective_paired_car as car
import covariant_collective_paired_source_current as current
import covariant_collective_fixed_frame_contact_typed as contact
import covariant_collective_cut2_neutral_loader as neutral_input
import covariant_collective_cut2_charged as charged_input
import covariant_collective_cut2_source_consumer_serialized as consumer

ROOT = Path(__file__).resolve().parent
THRESHOLDS = np.array([.25, 2., 4., 6., 8., 12., 20., 40.])
TIMES = (0., .5, 1., 2.)
INPUTS = {
    'neutral': ('covariant_collective_cut2_neutral_capsule_20261009.json', '5393d53572aea55179dcc4bd2f5a32cddb4544b7f94343c4ebd1fb851b5f81d0'),
    'charged': ('covariant_collective_cut2_charged_capsule_20261009.json', '0cbccb384238585209c373130d966da835aea3d95afc1dc73fa5105456ef6415'),
    'car': ('covariant_collective_paired_car_capsule_20261009.json', '517a539a51759fe1cb5e45668b3c760bbcf1201abfc83ea2cbf4884748900acf'),
    'source': ('covariant_collective_cut2_source_consumer_serialized_capsule_20261009.json', 'ccf99b7a57b593335b2714952cbcfdd9fd144f542fefe778d639a8d18bfa617f'),
    'cut1': ('covariant_collective_fixed_frame_contact_capsule_20261009.json', '8dd4526ed0be2e615648cd9fa9a27a2af631a8ab0e86da2124f69b0459d83cd2')}
INPUT_BYTES_AT_IMPORT = {name: (ROOT / filename).read_bytes() for name, (filename, _) in INPUTS.items()}
SOURCE_BYTES_AT_IMPORT = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if filename:
        path = Path(filename).resolve()
        if path.parent == ROOT and path.suffix == '.py':
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()


def signed_comparison(reference: Any, actual: Any) -> dict[str, Any]:
    a, r = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    return dict(reference=r.tolist(), actual=a.tolist(), signed_delta=(a-r).tolist(),
                absolute_delta=abs(a-r).tolist(), relative_delta_over_signed_reference=((a-r)/r).tolist())


def _group(values: np.ndarray) -> np.ndarray:
    return np.array([np.sum(values[:4]), np.sum(values[4:])])


def _contact_action(c: base.Spectrum, energy: np.ndarray, field: np.ndarray,
                    X: np.ndarray, physical_X: np.ndarray,
                    physical_HX: np.ndarray) -> np.ndarray:
    EX = contact.project(c, field[:, None] * physical_X)
    E2X = contact.project(c, field[:, None]**2 * physical_X)
    E2HX = contact.project(c, field[:, None]**2 * physical_HX)
    EHEX = contact.project(c, field[:, None] * contact.wave(c, energy[:, None] * EX))
    return -E2HX + 2*EHEX - energy[:, None]*E2X


def analyze(c: base.Spectrum, physical: dict[str, Any], source_state: dict[str, Any],
            common_neutral_lambda: float, current_result: dict[str, Any],
            post_source_pulse: float = .17, *, independent_frechet: bool = True) -> dict[str, Any]:
    """Use actual current_case state, common lambda, every sigma and four Ref.

    N3 may set independent_frechet=False only with explicit exact-C carrier and
    fresh N3 source/current/contact observations supplied by its integration owner.
    """
    energy = -2*np.log(c.q/math.sqrt(common_neutral_lambda))
    direction = np.array([.1, .2, .3, .4])
    electric = np.tile(np.array([E for _, _, E in physical['physical']]), (3, 1))
    field = electric @ direction
    Xi = source_state['Xi']
    M0 = float(np.vdot(Xi, Xi).real)
    original = source_state['sourcewave'].T/math.sqrt(M0)
    sources = np.concatenate((original, np.exp(-1j*post_source_pulse*field)[:, None]*original), axis=1)
    X = contact.project(c, sources)
    EX = contact.project(c, field[:, None]*sources)
    E2X = contact.project(c, field[:, None]**2*sources)
    thermal = np.exp(-.5*energy)[:, None]
    # Keep the two variants separate, summing only the original four Ref each.
    def rows(values: np.ndarray) -> np.ndarray:
        return np.array([np.sum(values[:, :4], axis=1), np.sum(values[:, 4:], axis=1)])
    G0rows = rows(X.conj()*thermal*X)
    G1rows = rows(1j*(EX.conj()*thermal*X-X.conj()*thermal*EX))
    G2rows = rows(-E2X.conj()*thermal*X+2*EX.conj()*thermal*EX-X.conj()*thermal*E2X)
    g0, g1, g2 = (np.real(np.sum(value, axis=1)) for value in (G0rows, G1rows, G2rows))
    disconnected = (g1/g0)**2
    connected = g2/g0-disconnected
    connected_rows = G2rows/g0[:, None]-(g1/g0)[:, None]*G1rows/g0[:, None]
    finite = []
    for step in (.01, .005):
        exact_plus = _group(contact.exact_pulse(c, energy, field, sources, .5, step))
        exact_minus = _group(contact.exact_pulse(c, energy, field, sources, .5, -step))
        nonlinear = (np.log(exact_plus)-2*np.log(g0)+np.log(exact_minus))/step**2
        row: dict[str, Any] = dict(step=step, exact_nonlinear_LOG_pulse=nonlinear.tolist(),
            exact_analytic_LOG_response=connected.tolist(), nonlinear_vs_analytic_absolute_defect=abs(nonlinear-connected).tolist())
        if independent_frechet:
            values: dict[str, list[np.ndarray]] = {'current': [], 'contact': []}
            calls = []
            for kind in values:
                for sign in (-1, 1):
                    operator = contact.PulseHamiltonian(c, energy, field, sign*step, kind)
                    evolved = expm_multiply(-.5*operator, X, traceA=-.5*float(np.sum(energy)))
                    values[kind].append(_group(np.real(np.sum(X.conj()*evolved, axis=0))))
                    calls.append(dict(kind=kind, sign=sign, batch_calls=operator.actual_operator_batch_calls))
            ordered = (values['current'][0]-2*g0+values['current'][1])/step**2
            seagull = (values['contact'][1]-values['contact'][0])/(2*step)
            assembled = (ordered+seagull)/g0-disconnected
            row.update(current_only_d2=ordered.tolist(), contact_only_d1=seagull.tolist(),
                current_only_d2_over_G0=(ordered/g0).tolist(), contact_only_d1_over_G0=(seagull/g0).tolist(),
                disconnected_subtraction=disconnected.tolist(), assembled_LOG_response=assembled.tolist(),
                assembled_vs_analytic_absolute_defect=abs(assembled-connected).tolist(),
                seagull_omission_error=abs(ordered/g0-disconnected-connected).tolist(),
                disconnected_omission_error=abs((ordered+seagull)/g0-connected).tolist(), actual_operator_calls=calls)
            print(json.dumps(dict(stage='actual_Frechet_step', charge=physical['charge'], step=step, assembled=assembled.tolist(), exact=connected.tolist())), flush=True)
        finite.append(row)
    # Original source only: amplitudes retain original M0, normalization explicit.
    rawX = Xi.T
    rawphysical = source_state['sourcewave'].T
    rawHphysical = contact.wave(c, energy[:, None]*rawX)
    Ledges = np.array([_contact_action(c, energy, electric[:, edge], rawX, rawphysical, rawHphysical).T for edge in range(4)])
    Ldirection = _contact_action(c, energy, field, rawX, rawphysical, rawHphysical).T
    jedges = source_state['currents']
    jdirection = np.einsum('e,erj->rj', direction, jedges)
    Lvertices = np.einsum('rj,erj->ej', Xi.conj(), Ledges)/M0
    jvertices = np.einsum('rj,erj->ej', Xi.conj(), jedges)/M0
    receiver = source_state['receiver']
    receiver_jrows = np.einsum('rj,erj->ej', receiver.conj(), jedges)/math.sqrt(M0)
    receiver_Lrows = np.einsum('rj,erj->ej', receiver.conj(), Ledges)/math.sqrt(M0)
    d_source_Lrows = np.sum(Xi.conj()*Ldirection, axis=0)/M0
    d_receiver_Lrows = np.sum(receiver.conj()*Ldirection, axis=0)/math.sqrt(M0)
    weights = np.sum(abs(Xi)**2, axis=0)/M0
    jnorm = np.sum(abs(jedges)**2, axis=1)/M0
    Lnorm = np.sum(abs(Ledges)**2, axis=1)/M0
    jdnorm = np.sum(abs(jdirection)**2, axis=0)/M0
    Ldnorm = np.sum(abs(Ldirection)**2, axis=0)/M0
    bins = np.searchsorted(THRESHOLDS, energy, side='left')
    partitions = []
    for index in range(len(THRESHOLDS)+1):
        mask = bins == index
        times = []
        for elapsed in TIMES:
            phase = np.exp(-1j*elapsed*energy[mask])
            times.append(dict(time=elapsed, source_current=current.complex_array(jvertices[:, mask]@phase),
                receiver_current=current.complex_array(receiver_jrows[:, mask]@phase),
                source_contact=current.complex_array(Lvertices[:, mask]@phase),
                receiver_contact=current.complex_array(receiver_Lrows[:, mask]@phase),
                source_direction_contact=current.complex_array(np.array(d_source_Lrows[mask]@phase)),
                receiver_direction_contact=current.complex_array(np.array(d_receiver_Lrows[mask]@phase))))
        partitions.append(dict(lower_exclusive=None if index == 0 else float(THRESHOLDS[index-1]),
            upper_inclusive=None if index == len(THRESHOLDS) else float(THRESHOLDS[index]), count=int(np.sum(mask)),
            source_norm_squared=float(np.sum(weights[mask])),
            current_source_norm_squared_per_edge=np.sum(jnorm[:, mask], axis=1).tolist(),
            contact_source_norm_squared_per_edge=np.sum(Lnorm[:, mask], axis=1).tolist(),
            direction_current_source_norm_squared=float(np.sum(jdnorm[mask])),
            direction_contact_source_norm_squared=float(np.sum(Ldnorm[mask])),
            connected_analytic_propagation_row_response=current.complex_array(np.sum(connected_rows[:, mask], axis=1)), time_insertions=times))
    times = []
    for elapsed in TIMES:
        phase = np.exp(-1j*elapsed*energy)
        times.append(dict(time=elapsed, source_contact=current.complex_array(Lvertices@phase),
            receiver_contact=current.complex_array(receiver_Lrows@phase),
            source_direction_contact=current.complex_array(np.array(d_source_Lrows@phase)),
            receiver_direction_contact=current.complex_array(np.array(d_receiver_Lrows@phase))))
    joint = _group(contact.exact_pulse(c, energy, field, sources, .5, .2, True))
    return dict(common_neutral_lambda=common_neutral_lambda, original_M0=M0, charge=physical['charge'],
        ALL_columns=len(c.q), ALL_Ref=4, direction=direction.tolist(), duration_positive_imaginary_time=.5,
        variants=['ORIGINALcoupledphoton_source', 'SEPARATEdeclared_external_postsource_drive'],
        separate_external_postsource_drive=post_source_pulse, original_source_norm_squared=float(np.sum(weights)),
        source_survival=g0.tolist(), LOG_first_derivative=(g1/g0).tolist(), LOG_connected_second_derivative=connected.tolist(),
        disconnected_subtraction=disconnected.tolist(), independent_Frechet_executed=independent_frechet,
        Frechet_and_nonlinear_rows=finite, joint_source_H_basischange_absolute_defect=abs(joint-g0).tolist(),
        source_projection_absolute_defect=float(np.linalg.norm(contact.wave(c, X)-sources)),
        actual_ALL4edge_contact_source_amplitudes=current.complex_array(Ledges),
        actual_direction_contact_source_amplitudes=current.complex_array(Ldirection),
        actual_ALL4edge_contact_source_vertices=current.complex_array(Lvertices),
        actual_original_contact_time_insertions=times, actual_current_case=current_result,
        spectral_energies=energy.tolist(), receiver_current_spectral_rows_at_time0=current.complex_array(receiver_jrows),
        receiver_contact_spectral_rows_at_time0=current.complex_array(receiver_Lrows),
        source_norm_squared_by_row=weights.tolist(), current_source_norm_squared_by_edge_and_row=jnorm.tolist(),
        contact_source_norm_squared_by_edge_and_row=Lnorm.tolist(),
        analytic_G0_G1_G2_rows=[current.complex_array(v) for v in (G0rows, G1rows, G2rows)],
        connected_analytic_propagation_rows=current.complex_array(connected_rows),
        connected_row_exactsum_absolute_defect=abs(np.sum(connected_rows, axis=1)-connected).tolist(),
        diagnostic_partitions=partitions, partition_source_norm_sum_absolute_defect=abs(sum(p['source_norm_squared'] for p in partitions)-float(np.sum(weights))),
        partition_contract='DISJOINT diagnostic energy bins ONLY, no source gates/pruning/bounds/convergence inference. Norms squared are divided by original M0. Current/contact complex receiver and source rows retain signs; phases reconstruct every reported time. Connected analytic rows label PROPAGATION row energy, NOT the intermediate energy of a two-current insertion. No intermediate-energy cutoff bound is inferred.',
        law='H(eps)=exp(i eps E_d) H exp(-i eps E_d); j_d=-i[H,E_d], L_dd=-[E_d,[E_d,H]]; commuting full physical ALL4edge E including longitudinal Gauss rho',
        independent_Frechet_contract='Actual independent exp(-t(H+h j)) second derivative plus exp(-t(H+h L)) first derivative minus disconnected; not inherited from cut1.' if independent_frechet else 'No redundant N3 Frechet: exact C-carrier only; fresh N3 source/current/contact/tail/nonlinear observations computed here. Integration must report actual C controls.')


def run() -> dict[str, Any]:
    for name, (filename, expected) in INPUTS.items():
        data = INPUT_BYTES_AT_IMPORT[name]
        if hashlib.sha256(data).hexdigest() != expected or (ROOT/filename).read_bytes() != data:
            raise RuntimeError(f'Actual frozen input changed: {name}')
    n, nc = neutral_input.load_spectrum(ROOT/INPUTS['neutral'][0])
    plus, cc = charged_input.load_spectrum(ROOT/INPUTS['charged'][0])
    gamma, embedding, nodes, heat, Hphi, metric = base.left.frame()
    neutral, addition, removal = base.space(0, 2), base.space(1, 2), base.space(-1, 2)
    Cbody = car.physical_conjugation(removal, addition, False)
    Cquartet = car.physical_conjugation(removal, addition, True)
    Cn = car.physical_conjugation(neutral, neutral, False)
    minus = base.Spectrum([dict(V=b['V'], q=b['q'], P=Cbody.T@b['P'], sign=b['sign']) for b in plus.blocks], 3*len(removal['physical']))
    packet = np.array([.1, .8, .4, .1]); packet /= np.linalg.norm(packet)
    receiver = np.array([.5, .4, .1, .6]); receiver /= np.linalg.norm(receiver)
    Sa = base.collective.source(neutral, addition, base.collective.native.source_coefficients(gamma, packet), embedding)
    Sout = base.collective.source(neutral, addition, base.collective.native.source_coefficients(gamma, receiver), embedding)
    Sh = car.annihilator(neutral, removal, car.negative_coefficients(gamma, packet), car.antiparticle_embedding(embedding))
    Hout = car.annihilator(neutral, removal, car.negative_coefficients(gamma, receiver), car.antiparticle_embedding(embedding))
    Omega, photon, Z = current.photon_state(n, neutral)
    lambdaN = float(n.q[0]**2)
    jp, ps = current.current_case(plus, addition, Sa, Sout, Omega, photon, lambdaN)
    jh, hs = current.current_case(minus, removal, Sh, Hout, Omega, photon, lambdaN)
    print(json.dumps(dict(stage='actual_cut2_contact_tail', ranks=[len(n.q), len(plus.q), len(minus.q)], lambdaN=lambdaN)), flush=True)
    p = analyze(plus, addition, ps, lambdaN, jp, .17)
    h = analyze(minus, removal, hs, lambdaN, jh, -.17, independent_frechet=False)
    eta = complex(np.vdot(Omega, Cn@Omega))
    def mapped(values: np.ndarray) -> np.ndarray:
        wave = np.array([minus.wave(v) for v in values])
        return np.array([plus.project(w) for w in (Cquartet@wave.reshape(-1)).reshape(4, plus.dimension)])
    C_source = float(np.linalg.norm(mapped(hs['Xi'])-eta*ps['Xi'])/math.sqrt(p['original_M0']))
    C_j = [float(np.linalg.norm(mapped(v)+eta*w)/math.sqrt(p['original_M0'])) for v, w in zip(hs['currents'], ps['currents'])]
    def decode(value: dict[str, Any]) -> np.ndarray:
        return np.array(value['real'])+1j*np.array(value['imag'])
    C_L = [float(np.linalg.norm(mapped(v)-eta*w)/math.sqrt(p['original_M0'])) for v,w in zip(decode(h['actual_ALL4edge_contact_source_amplitudes']), decode(p['actual_ALL4edge_contact_source_amplitudes']))]
    cut1 = json.loads(INPUT_BYTES_AT_IMPORT['cut1'])
    prior = json.loads(INPUT_BYTES_AT_IMPORT['source'])
    comparison = {'LOG_connected': signed_comparison(cut1['exact_LOG_connected_second_derivative'], p['LOG_connected_second_derivative'])}
    for old, new in zip(cut1['actual_Frechet_current_contact_rows'], p['Frechet_and_nonlinear_rows']):
        comparison[str(new['step'])] = dict(current_d2=signed_comparison(old['ALL_two_ordered_current_contribution'], new['current_only_d2']),
            contact_d1=signed_comparison(old['actual_seagull_Frechet_contribution'], new['contact_only_d1']),
            disconnected=signed_comparison(old['actual_disconnected_contribution'], new['disconnected_subtraction']),
            assembled_LOG=signed_comparison(old['assembled_connected_log_response'], new['assembled_LOG_response']))
    descriptor = dict(input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in INPUT_BYTES_AT_IMPORT.items()},
        input_source_descriptor_sha256=prior['input_descriptor_sha256'], cut2_source_descriptor=prior['input_descriptor'],
        cut1_descriptor=cut1['input_descriptor'], neutral_lambda=lambdaN,
        law=p['law'], regulator='compact electric/Fourier cutoff2 ALL4edges UE1, NOT Hermite grade',
        spacing='transfer-step units as frozen base H=-log(K/commonlambda); no physical spatial/time rescaling introduced',
        root_geometry='selectedCliff2 compact onecycle ALL36 Phi3; canonical Gauss rows serialized losslessly',
        source_packet=packet.tolist(), receiver_packet=receiver.tolist(), source_embedding=current.complex_array(embedding),
        negative_embedding=current.complex_array(car.antiparticle_embedding(embedding)),
        partition_thresholds=THRESHOLDS.tolist(), times=list(TIMES), no_source_gate_changes=True,
        provenance='Historical plan/input source hashes retained as historical provenance, NOT live physics gate',
        exclusions='No RootW/0226/450a/on-shell binding; no massless/commoncone/full4Dirac/fullCartesianPhi/SU4/realEM promotion')
    return dict(input_descriptor=descriptor, input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        source_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={k:v.decode() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        physical_mean_descriptor=metric, neutral_controls=nc,
        charged_controls={**cc, 'physical': consumer.canonical_physical_descriptor(cc['physical'])},
        root_geometries={name:consumer.canonical_physical_descriptor(value) for name,value in [('neutral',neutral),('addition',addition),('removal',removal)]},
        actual_photon_Z=Z, particle=p, proper_N3_removal=h, cut1_to_cut2=comparison,
        actual_C_controls=dict(source_relative_defect=C_source, all4edge_current_Codd_relative_defects=C_j,
            all4edge_contact_Ceven_relative_defects=C_L, LOG_first_Codd_absolute_defect=abs(np.array(p['LOG_first_derivative'])+np.array(h['LOG_first_derivative'])).tolist(),
            LOG_second_Ceven_absolute_defect=abs(np.array(p['LOG_connected_second_derivative'])-np.array(h['LOG_connected_second_derivative'])).tolist()),
        original_source_vs_frozen_cut2_M0_absolute_defect=abs(p['original_M0']-prior['actual_particle_full_field_current']['actual_M0']),
        original_LOG_vs_frozen_cut2_absolute_defect=abs(np.array(p['LOG_connected_second_derivative'])-np.array(prior['actual_particle_connected_fixed_frame_source_response']['LOG_connected_second_derivative'])).tolist(),
        full_SVD_rerun=False, originalrealEM_or_onshell_admitted=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    result = run()
    result['runtime_seconds'] = time.time()-started
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        input_descriptor_sha256=result['input_descriptor_sha256'], runtime_seconds=result['runtime_seconds'],
        final_Frechet=result['particle']['Frechet_and_nonlinear_rows'][-1], C_controls=result['actual_C_controls'])), flush=True)


if __name__ == '__main__':
    main()
