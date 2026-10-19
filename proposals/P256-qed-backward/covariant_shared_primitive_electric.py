#!/usr/bin/env python3
"""New UE=1 primitive-electric shared split-route finite quantum construction.

The electric law is a declared input, not induced electromagnetism. All canonical
columns are retained; the complete E=0 columns of the same factor give its Haar
endpoint. Face packet q is not translation momentum or continuum dispersion.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import scipy
from scipy.linalg import eigh, svd

ROOT = Path(__file__).resolve().parent
PLAN_PATH = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT = PLAN_PATH.read_bytes()
from covariant_shared_split_route_photon import split_route_half
from covariant_shared_plaquette_photon import graph, curl, apply_field, source, apply_source, quantum_loads
from covariant_native_source_gram_moments import matched_tree_half
import covariant_native_haar_range_jets as native
from covariant_induced_photon import Parameters, phi_frame

SOURCE_BYTES_AT_IMPORT = {Path(module.__file__).name: Path(module.__file__).read_bytes()
    for module in tuple(sys.modules.values())
    if getattr(module, '__file__', None) and Path(module.__file__).suffix == '.py'
    and Path(module.__file__).resolve().parent == ROOT}
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()


def space(charge: int, cutoff: int) -> dict[str, Any]:
    light = list(itertools.combinations(range(8), 4 + charge))
    auxiliary = list(itertools.combinations(range(4), 2))
    states = list(itertools.product(light, auxiliary))
    rho = np.array([(*native.density(l, a, charge), 0, 0) for l, a in states])
    incidence, _ = graph()
    physical = []
    for state, density in enumerate(rho):
        for kL in range(-cutoff, cutoff + 1):
            for kR in range(-cutoff, cutoff + 1):
                electric = (kL + int(density[0]), kL + int(density[0] + density[1]),
                    kL + int(density[0] + density[1] + density[2]), kL + kR, -kR, -kR, -kR)
                if max(abs(e) for e in electric) > cutoff:
                    continue
                if np.any(incidence @ np.array(electric) != density):
                    raise ArithmeticError('Six-node total Gauss violation')
                physical.append((state, kL, kR, electric))
    return dict(charge=charge, light=light, auxiliary=auxiliary, states=states, rho=rho,
        physical=physical, index={(s, l, r): j for j, (s, l, r, _) in enumerate(physical)},
        state_index={s: j for j, s in enumerate(states)})


def canonical(half: np.ndarray, prefactor: float, occupied: np.ndarray) -> np.ndarray:
    return prefactor * np.linalg.det(half[occupied[:, None, :, None], occupied[None, :, None, :]])


def validate_light_identity(gamma: np.ndarray, nodes: np.ndarray) -> list[dict[str, Any]]:
    """Actual full point matrices at nonzero angles, before Fourier shortcut use."""
    rows = []
    auxiliary = np.array(list(itertools.combinations(range(4), 2)))
    for charge in (0, 1):
        occupied = np.array(list(itertools.combinations(range(8), 4 + charge)))
        n3 = np.sum(occupied >= 6, axis=1)
        for normal in nodes:
            actual = gamma.copy()
            actual[1] *= 1 + float(normal)
            mean = canonical(*matched_tree_half(actual, 0., 0.), occupied)
            for thetaL, thetaR in ((.37, -.61), (-.83, .29)):
                light = canonical(*matched_tree_half(actual, thetaL, 0.), occupied)
                predicted = mean * np.exp(1j * thetaL * (n3[:, None] - n3[None, :]))
                phaseB = complex((np.exp(1j * thetaL) + 1) / 2)
                phaseF = complex((np.exp(1j * thetaL) + np.exp(1j * (thetaL - thetaR))) / 2)
                aux = canonical(*split_route_half(actual, .04, phaseB, phaseF), auxiliary)
                direct = np.kron(light, aux)
                shortcut = np.kron(predicted, aux)
                defect = float(np.linalg.norm(direct - shortcut) / np.linalg.norm(direct))
                if defect > 1e-11:
                    raise ArithmeticError('Full canonical nonzero-angle light-tree identity failed')
                rows.append(dict(charge=charge, Phi_normal=float(normal), thetaL=thetaL, thetaR=thetaR,
                    actual_full_pointcanonical_dimension=len(direct), actual_relative_identity_defect=defect))
    return rows


def full_factor(gamma: np.ndarray, nodes: np.ndarray, heat: np.ndarray,
                physical: dict[str, Any], grid: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    occupied = np.array(physical['light'])
    auxiliary = np.array(physical['auxiliary'])
    light_lookup = {s: j for j, s in enumerate(physical['light'])}
    aux_lookup = {s: j for j, s in enumerate(physical['auxiliary'])}
    trip = np.array([(light_lookup[l], aux_lookup[a]) for l, a in physical['states']])
    states = np.array([s for s, _, _, _ in physical['physical']])
    lr, ar = trip[states].T
    fluxL = np.array([l for _, l, _, _ in physical['physical']])
    fluxR = np.array([r for _, _, r, _ in physical['physical']])
    n3 = np.sum(occupied >= 6, axis=1)[lr]
    total_electric = np.array([sum(e * e for e in E) for _, _, _, E in physical['physical']])
    electric_half = np.exp(-total_electric / 2)
    nr = len(states)
    factor = np.zeros((3 * nr, 3 * nr), complex, order='F')
    lower = []
    defects = []
    angles = 2 * math.pi * np.arange(grid) / grid
    for p, normal in enumerate(nodes):
        actual = gamma.copy()
        actual[1] *= 1 + float(normal)
        mean = canonical(*matched_tree_half(actual, 0., 0.), occupied)
        samples = np.zeros((grid, grid, 6, 6), complex)
        point_min = math.inf
        for i, thetaL in enumerate(angles):
            for j, thetaR in enumerate(angles):
                phaseB = complex((np.exp(1j * thetaL) + 1) / 2)
                phaseF = complex((np.exp(1j * thetaL) + np.exp(1j * (thetaL - thetaR))) / 2)
                samples[i, j] = canonical(*split_route_half(actual, .04, phaseB, phaseF), auxiliary)
                point_min = min(point_min, float(eigh(samples[i, j], eigvals_only=True)[0]))
        lower.append(float(eigh(mean, eigvals_only=True)[0]) * point_min)
        coefficients = np.fft.fft2(samples, axes=(0, 1)) / grid ** 2
        SF = np.empty((nr, nr), complex)
        for start in range(0, nr, 128):
            stop = min(nr, start + 128)
            deltaL = fluxL[start:stop, None] - fluxL[None, :] - n3[start:stop, None] + n3[None, :]
            deltaR = fluxR[start:stop, None] - fluxR[None, :]
            SF[start:stop] = mean[lr[start:stop, None], lr[None, :]] * coefficients[
                deltaL % grid, deltaR % grid, ar[start:stop, None], ar[None, :]]
        defects.append(float(np.linalg.norm(SF - SF.conj().T) / np.linalg.norm(SF)))
        if defects[-1] > 1e-11 or lower[-1] <= 0:
            raise ArithmeticError('Projected full canonical SF positivity/Hermitian failure')
        for out in range(3):
            factor[out * nr:(out + 1) * nr, p * nr:(p + 1) * nr] = heat[out, p] * SF * electric_half[None, :]
    middle = np.array([j for j, (_, _, _, E) in enumerate(physical['physical']) if all(e == 0 for e in E)])
    columns = np.concatenate([middle + p * nr for p in range(3)])
    haar = factor[:, columns].copy(order='F')
    return factor, haar, dict(actual_complete_factor_dimension=3 * nr, actual_complete_E0_columns=len(columns),
        actual_electric_half_min=float(electric_half.min()), actual_sum_ALL7edges_E2_max=int(total_electric.max()),
        actual_pointcanonical_positive_lower_bounds=lower, actual_projected_SF_Hermitian_relative_defects=defects)


def spectrum(factor: np.ndarray) -> dict[str, Any]:
    vectors, singulars, right = svd(factor, full_matrices=False, overwrite_a=True, lapack_driver='gesdd')
    del right
    if singulars[-1] <= 0:
        raise ArithmeticError('Positive factor lost a singular direction; no Gram clipping permitted')
    return dict(V=vectors, q=singulars)


def loads(wave: np.ndarray, physical: dict[str, Any], Hphi: np.ndarray) -> dict[str, Any]:
    result = quantum_loads(wave, physical, Hphi)
    w = wave.reshape(-1, 3, len(physical['physical']))
    probability = np.sum(abs(w) ** 2, axis=(0, 1)) / float(np.vdot(w, w).real)
    E = np.array([electric for _, _, _, electric in physical['physical']])
    _, cycles = graph()
    transverse = E @ cycles @ np.linalg.inv(cycles.T @ cycles) @ cycles.T
    longitudinal = E - transverse
    result.update(actual_ALL7edge_electric_square_means=(probability @ (E ** 2)).tolist(),
        actual_total_primitive_electric_energy=float(probability @ np.sum(E ** 2, axis=1)),
        actual_longitudinal_Coulomb_energy=float(probability @ np.sum(longitudinal ** 2, axis=1)),
        actual_transverse_electric_energy=float(probability @ np.sum(transverse ** 2, axis=1)))
    return result


def observe(n: dict[str, Any], c: dict[str, Any], neutral: dict[str, Any], charged: dict[str, Any],
            S: list[Any], Sout: list[Any], Hphi: np.ndarray) -> dict[str, Any]:
    Vn, Vc = n['V'], c['V']
    lambdaN = float(n['q'][0] ** 2)
    qn, qc = n['q'] / math.sqrt(lambdaN), c['q'] / math.sqrt(lambdaN)
    En, Ec = -2 * np.log(qn), -2 * np.log(qc)
    Omega = Vn[:, 0]
    BL, BR = curl(neutral, 0), curl(neutral, 1)
    CL, CR = curl(charged, 0), curl(charged, 1)
    rawL, rawR = apply_field(BL, Omega), apply_field(BR, Omega)
    bL, bR = Vn.conj().T @ rawL, Vn.conj().T @ rawR
    target = apply_source(Sout, Omega)
    cases = []
    for q in (0., math.pi):
        mode = np.exp(1j * q)
        b = (bL + mode * bR) / math.sqrt(2)
        rawph = (rawL + mode * rawR) / math.sqrt(2)
        ph = qn * b
        Zph = float(np.vdot(ph, ph).real)
        if Zph <= 0:
            raise ArithmeticError('Actual physical face photon preparation zero')
        ph /= math.sqrt(Zph)
        incoming = Vn @ ph
        raw = apply_source(S, incoming)
        projected = np.array([Vc.conj().T @ w for w in raw])
        Xi = qc[None, :] * projected
        M0 = float(np.vdot(Xi, Xi).real)
        if M0 <= 0:
            raise ArithmeticError('Actual literal source absent')
        E1 = float(np.sum(abs(Xi) ** 2 * Ec[None, :]))
        R1 = float(np.sum(abs(Xi) ** 2 * qc[None, :] ** 2))
        source_null = np.array([w - Vc @ z for w, z in zip(raw, projected)])
        photon_null = rawph - Vn @ b
        outgoing = np.array([(apply_field(CL, w) + mode * apply_field(CR, w)) / math.sqrt(2) for w in target])
        receiver = np.array([Vc.conj().T @ w for w in outgoing])
        consumers = []
        for elapsed in (0., .5, 1., 2.):
            evolved = Xi * np.exp(-1j * elapsed * Ec)[None, :] / math.sqrt(M0)
            amplitude = np.vdot(receiver, evolved)
            consumers.append(dict(time=elapsed, coherent_Ref4_receiver_real=float(amplitude.real),
                coherent_Ref4_receiver_imag=float(amplitude.imag), coherent_Ref4_receiver_probability=float(abs(amplitude) ** 2),
                actual_evolved_source_norm_squared=float(np.vdot(evolved, evolved).real)))
        sourcewave = np.array([Vc @ z for z in Xi])
        cases.append(dict(actual_face_packet_q=q, q_is_packet_label_not_translation_eigenvalue=True,
            actual_photon_Zph=Zph, actual_photon_energy_mean=float(np.sum(abs(ph) ** 2 * En)),
            actual_M0=M0, actual_E1=E1, actual_R1=R1, actual_energy_mean=E1 / M0, actual_transfer_Rayleigh_mean=R1 / M0,
            actual_raw_photon_Knull_norm_squared=float(np.vdot(photon_null, photon_null).real),
            actual_raw_source_Knull_norm_squared=float(np.vdot(source_null, source_null).real),
            actual_charged_source_band_rows=[dict(absolute_threshold=t, source_fraction=float(np.sum(abs(Xi[:, Ec <= t]) ** 2) / M0)) for t in (.25, 2., 4.)],
            actual_consumer=consumers, actual_quantum_photon_loads=loads(incoming, neutral, Hphi),
            actual_quantum_source_loads=loads(sourcewave, charged, Hphi),
            actual_neutral_photon_spectral_rows=[dict(energy=float(e), weight=float(w)) for e, w in zip(En, abs(ph) ** 2)],
            actual_charged_source_spectral_rows=[dict(energy=float(e), absolute_weight=float(w)) for e, w in zip(Ec, np.sum(abs(Xi) ** 2, axis=0))]))
    cross = []
    for elapsed in (0., .5, 1., 2.):
        value = np.vdot(bL, bR * np.exp(-1j * elapsed * En))
        cross.append(dict(time=elapsed, actual_positive_sector_retarded_cross_face_response=2 * float(value.imag),
            actual_positive_sector_cross_spectral_real=float(value.real), actual_positive_sector_cross_spectral_imag=float(value.imag)))
    return dict(actual_common_neutral_lambdaN=lambdaN, actual_positive_ranks=[len(qn), len(qc)],
        actual_ALL_neutral_singular_values=n['q'].tolist(), actual_ALL_charged_singular_values=c['q'].tolist(),
        actual_normalized_min_positive_singular_values=[float(qn[-1]), float(qc[-1])],
        actual_neutral_ground_loads=loads(Omega, neutral, Hphi), actual_face_packet_cases=cases,
        actual_positive_sector_retarded_cross_face_rows=cross)


def run(cutoff: int, grid: int) -> dict[str, Any]:
    gamma4, metric = phi_frame(Parameters(kappa_power=400))
    values, U = eigh(1j * gamma4[0] @ gamma4[1] @ gamma4[2])
    embedding = U[:, values > 0]
    _, frame = eigh(embedding.conj().T @ gamma4[0] @ embedding)
    embedding = embedding @ frame[:, ::-1]
    gamma = np.array([embedding.conj().T @ g @ embedding for g in gamma4[:3]])
    nodes = np.array([-.09, 0., .09])
    sigma, radius, dy = .03, .12, .09
    k = np.exp(-nodes ** 2 / (4 * sigma ** 2) - sigma ** 2 * nodes ** 2 / (radius ** 2 * (radius ** 2 - nodes ** 2)))
    k /= np.linalg.norm(k)
    D = np.zeros((2, 3))
    for j in range(2):
        D[j, j] = math.sqrt(k[j + 1] / k[j]) / dy
        D[j, j + 1] = -math.sqrt(k[j] / k[j + 1]) / dy
    Hphi = 2 * sigma ** 2 * D.T @ D
    values, U = eigh(Hphi)
    heat = (U * np.exp(-values / 2)) @ U.T
    identity = validate_light_identity(gamma, nodes)
    print(json.dumps(dict(stage='validated_actual_nonzero_fullcanonical_identity', max_defect=max(r['actual_relative_identity_defect'] for r in identity))), flush=True)
    neutral, charged = space(0, cutoff), space(1, cutoff)
    spectra, haar_spectra, diagnostics = [], [], {}
    for label, physical in (('neutral', neutral), ('charged', charged)):
        print(json.dumps(dict(stage='fullcanonical_factor_and_SVD', sector=label, dimension=3 * len(physical['physical']))), flush=True)
        factor, haar, diagnostic = full_factor(gamma, nodes, heat, physical, grid)
        diagnostics[label] = diagnostic
        haar_spectra.append(spectrum(haar))
        del haar
        spectra.append(spectrum(factor))
        del factor
    packet = np.array([.1, .8, .4, .1])
    packet /= np.linalg.norm(packet)
    outpacket = np.array([.5, .4, .1, .6])
    outpacket /= np.linalg.norm(outpacket)
    S = source(neutral, charged, native.source_coefficients(gamma, packet), embedding)
    Sout = source(neutral, charged, native.source_coefficients(gamma, outpacket), embedding)
    full = observe(*spectra, neutral, charged, S, Sout, Hphi)
    haar = observe(*haar_spectra, neutral, charged, S, Sout, Hphi)
    _, cycles = graph()
    descriptor = dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',
        plan_sha256_at_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest(),
        pinned_general='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        law='NEW primitive electric INPUT UE FIXED1, SAME split_route_half SF6b0243, no standardWilson or electric induction claim',
        F='PhiheatHalf SF_fullcanonical exp[-sum_ALL7edges E²/2]; K=F Fdag in ONE total-density globalGauss physicalspace',
        light_tree_identity='Ltheta=exp[i thetaL(n3out-n3in)] Lmean; SF= Lmean*auxFourier[delta_kL-delta_light_n3,delta_kR]',
        Haar_endpoint='UE→infinity projects complete physical E=0 input columns of SAME F, totalrho0; native reference only',
        primitive_electric_input=1, NEW_declared_electric_cutoff_ALL7edges=cutoff, NOT_historical_E12_retag=True,
        physical_edges=[[0,1],[1,2],[2,3],[3,0],[3,4],[4,5],[5,0]], cycle_columns=cycles.tolist(),
        E='(kL+rho0,kL+rho0+rho1,kL+rho0+rho1+rho2,kL+kR,-kR,-kR,-kR)',
        longitudinal_Coulomb_retained=True, full_active_CAR_states=[70,56], all_auxiliary_CAR_states=6,
        active_kappa=.09, auxiliary_kappa=.04, light_homotopy=0., originalRef=4, preparedRef=3,
        Phi_nodes=nodes.tolist(), Phi_sigma=sigma, Phi_radius=radius, actual_field_grid_each_cycle=grid,
        source_packet=packet.tolist(), receiver_packet=outpacket.tolist(),
        source_paths=[[0]*7,[-1,0,0,0,0,0,0],[-1,-1,0,0,0,0,0],[0,0,0,1,0,0,0]],
        source_Pe_absolute_thresholds=[.25,2.,4.], consumer_times=[0.,.5,1.,2.],
        photon='physical Bq=(sin(thetaL)+exp(iq)sin(thetaR))/sqrt2, q0/pi facepacket NOT translationdispersion',
        moment_contract='lambdaN largest Fn singular²; KN/KC same lambdaN; Xi_ph=sqrt(KN/lambdaN)Bq Omega/sqrtZph; Xi_ch=sqrt(KC/lambdaN)S_literal Xi_ph; M0=norm²Xi; E1=<Xi,-log(KC/lambdaN)Xi>/tau; R1=<Xi,KC/lambdaN Xi>; tau1 ONEwholechargedhalfpervector',
        quantum_postulates_and_microscopic_parameters='Declared inputs, not newly derived or a permission gate',
        SVD='ALL positive factor singulars retained, no Gram squaring/clipping/occupation pruning',
        scope='Declared finite electric/Gauss/Pauli/Phi3 model only; no continuum, fullCartesianPhi/SU4, massless, macro, commoncone, imported speed/mass/quarterfit',
        no_bare_magnetic_action=True, no_source_Pe_retrofit=True)
    return dict(input_descriptor=descriptor, input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        source_at_import={name: hashlib.sha256(data).hexdigest() for name, data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name: data.decode() for name, data in SOURCE_BYTES_AT_IMPORT.items()},
        plan_bytes_at_import=PLAN_BYTES_AT_IMPORT.decode(),
        runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        physical_mean_descriptor=metric, actual_nonzero_pointcanonical_identity_rows=identity,
        actual_fullcolumn_factor_diagnostics=diagnostics, actual_UE1_fullpositive=full, actual_same_factor_Haar_E0_endpoint=haar)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--electric-cutoff', type=int, required=True, choices=(1, 2))
    parser.add_argument('--field-grid', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.field_grid < 32:
        parser.error('Actual field grid must retain at least 32² points')
    start = time.monotonic()
    result = run(args.electric_cutoff, args.field_grid)
    result['runtime_seconds'] = time.monotonic() - start
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(source_at_import=result['source_at_import'], input_descriptor_sha256=result['input_descriptor_sha256'], runtime_seconds=result['runtime_seconds'])), flush=True)
    print(json.dumps({label: dict(ranks=result[label]['actual_positive_ranks'], cases=[{key: row[key] for key in ('actual_face_packet_q','actual_photon_energy_mean','actual_M0','actual_E1','actual_R1','actual_energy_mean','actual_raw_source_Knull_norm_squared')} for row in result[label]['actual_face_packet_cases']]) for label in ('actual_UE1_fullpositive','actual_same_factor_Haar_E0_endpoint')}), flush=True)


if __name__ == '__main__':
    main()
