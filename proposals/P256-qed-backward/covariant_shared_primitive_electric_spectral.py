#!/usr/bin/env python3
"""Final UE1 shared-face driver exporting every neutral 2x2 spectral matrix.

Uses the frozen initial all-column electric construction without modifying it.
No translation momentum, pole, massless or common-cone promotion is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh
import covariant_shared_primitive_electric as core

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT = {Path(module.__file__).name: Path(module.__file__).read_bytes()
    for module in tuple(sys.modules.values())
    if getattr(module, '__file__', None) and Path(module.__file__).suffix == '.py'
    and Path(module.__file__).resolve().parent == ROOT}
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()


def spectral_observe(n: dict[str, Any], c: dict[str, Any], neutral: dict[str, Any],
                     charged: dict[str, Any], S: list[Any], Sout: list[Any], Hphi: np.ndarray) -> dict[str, Any]:
    result = core.observe(n, c, neutral, charged, S, Sout, Hphi)
    vectors = n['V']
    qn = n['q'] / n['q'][0]
    energies = -2 * np.log(qn)
    Omega = vectors[:, 0]
    b = np.array([vectors.conj().T @ core.apply_field(core.curl(neutral, axis), Omega) for axis in (0, 1)])
    rows = []
    prepared = np.array([qn * (b[0] + np.exp(1j * q) * b[1]) / math.sqrt(2) for q in (0., math.pi)])
    Z = np.sum(abs(prepared) ** 2, axis=1)
    for j, energy in enumerate(energies):
        matrix = np.conjugate(b[:, j, None]) * b[None, :, j]
        rows.append(dict(neutral_singular_index=j, actual_neutral_energy=float(energy),
            actual_bL_real=float(b[0, j].real), actual_bL_imag=float(b[0, j].imag),
            actual_bR_real=float(b[1, j].real), actual_bR_imag=float(b[1, j].imag),
            actual_W_face_order=['L', 'R'], actual_W_real=matrix.real.tolist(), actual_W_imag=matrix.imag.tolist(),
            actual_prepared_q0_absolute_weight=float(abs(prepared[0, j]) ** 2),
            actual_prepared_qpi_absolute_weight=float(abs(prepared[1, j]) ** 2),
            actual_prepared_q0_normalized_weight=float(abs(prepared[0, j]) ** 2 / Z[0]),
            actual_prepared_qpi_normalized_weight=float(abs(prepared[1, j]) ** 2 / Z[1])))
    result['actual_every_neutral_energy_full_2x2_face_spectral_matrix_rows'] = rows
    result['face_spectral_matrix_contract'] = 'b_r=Vndag B_r Omega; W_rs=conj(b_r)b_s at EVERY neutral energy/singular index, without binning/dropping. Prepared q weights include qn² and same sqrt2/Zph. Degenerate rows may be basis-dependent; their summed W is invariant.'
    result['actual_face_spectral_matrix_sum_real'] = (b.conj() @ b.T).real.tolist()
    result['actual_face_spectral_matrix_sum_imag'] = (b.conj() @ b.T).imag.tolist()
    return result


def run(cutoff: int, grid: int) -> dict[str, Any]:
    gamma4, metric = core.phi_frame(core.Parameters(kappa_power=400))
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
    identity = core.validate_light_identity(gamma, nodes)
    print(json.dumps(dict(stage='validated_actual_nonzero_fullcanonical_identity', max_defect=max(r['actual_relative_identity_defect'] for r in identity))), flush=True)
    neutral, charged = core.space(0, cutoff), core.space(1, cutoff)
    spectra, haar_spectra, diagnostics = [], [], {}
    for label, physical in (('neutral', neutral), ('charged', charged)):
        print(json.dumps(dict(stage='fullcanonical_factor_and_SVD', sector=label, dimension=3 * len(physical['physical']))), flush=True)
        factor, haar, diagnostic = core.full_factor(gamma, nodes, heat, physical, grid)
        diagnostics[label] = diagnostic
        haar_spectra.append(core.spectrum(haar))
        del haar
        spectra.append(core.spectrum(factor))
        del factor
    packet = np.array([.1, .8, .4, .1])
    packet /= np.linalg.norm(packet)
    outpacket = np.array([.5, .4, .1, .6])
    outpacket /= np.linalg.norm(outpacket)
    S = core.source(neutral, charged, core.native.source_coefficients(gamma, packet), embedding)
    Sout = core.source(neutral, charged, core.native.source_coefficients(gamma, outpacket), embedding)
    full = spectral_observe(*spectra, neutral, charged, S, Sout, Hphi)
    haar = spectral_observe(*haar_spectra, neutral, charged, S, Sout, Hphi)
    descriptor = dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',
        plan_sha256_at_IMPORT=hashlib.sha256(core.PLAN_BYTES_AT_IMPORT).hexdigest(),
        pinned_general='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        primitive_electric_input=1, NEW_declared_electric_cutoff_ALL7edges=cutoff, NOT_historical_E12_retag=True,
        actual_field_grid_each_cycle=grid, Phi_nodes=nodes.tolist(), Phi_sigma=sigma, Phi_radius=radius,
        Phi3_is_three_levels_of_ONE_normal_coordinate=True,
        active_modes=8, active_particle_numbers=[4,5], active_CAR_states=[70,56],
        auxiliary_modes=4, auxiliary_particle_number=2, auxiliary_CAR_states=6,
        original_Ref=4, prepared_Ref=3, active_kappa=.09, auxiliary_kappa=.04,
        law='SAME frozen SF6b0243 split-route B/F mean; ONE globalGauss after full light and all6aux matrices; F=PhiheatHalf SF_fullcanonical exp[-sum_ALL7edges E²/2]; K=F Fdag',
        electric_heat_includes_longitudinal_Coulomb=True,
        light_identity='exp[i thetaL(n3out-n3in)] Lmean validated BEFORE shortcut against full canonical point matrices including each proper gammaPhi3',
        Haar='Complete physical E=0 columns extracted from SAME F; UE→infinity endpoint, native reference only',
        spectral_matrix='W_rs(E)=conj(b_r)b_s, bL=Vndag BL Omega, bR=Vndag BR Omega, EVERY neutral energy; export real/imag and prepared q0/pi weights',
        source_packet=packet.tolist(), receiver_packet=outpacket.tolist(),
        source_paths=[[0]*7,[-1,0,0,0,0,0,0],[-1,-1,0,0,0,0,0],[0,0,0,1,0,0,0]],
        Pe=[.25,2.,4.], consumer_times=[0.,.5,1.,2.], q=[0.,math.pi],
        moment_contract='SAME lambdaN largest Fn singular² for KN/KC; Omega actual neutral ground; Xi_ph=sqrt(KN/lambdaN)BqOmega/sqrtZph; Xi_ch=sqrt(KC/lambdaN)S_literal Xi_ph; M0=norm²Xi; E1=<Xi,-log(KC/lambdaN)Xi>/tau1; R1=<Xi,KC/lambdaN Xi>; ONE charged whole halfstep pervector; coherent Ref4 before abs²; physical outgoingBq and independentreceiver',
        SVD='Full positive factor ALL singulars; no Gram squaring/clipping/pruning',
        quantum_law_and_parameters='Declared inputs; no permission gate or derivation claim',
        scope='Finitefacepacket NOT translationdispersion/pole/massless/commoncone/macro/untruncatedEM; no fullCartesianPhi/SU4, no speed/mass/quarterfit, no bare magnetic action, no inducedE claim')
    return dict(input_descriptor=descriptor, input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        source_at_import={name: hashlib.sha256(data).hexdigest() for name, data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name: data.decode() for name, data in SOURCE_BYTES_AT_IMPORT.items()},
        plan_bytes_at_import=core.PLAN_BYTES_AT_IMPORT.decode(),
        runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=core.scipy.__version__, platform=core.platform.platform()),
        physical_mean_descriptor=metric, actual_nonzero_pointcanonical_identity_rows=identity,
        actual_fullcolumn_factor_diagnostics=diagnostics, actual_UE1_fullpositive=full, actual_same_factor_Haar_E0_endpoint=haar)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--electric-cutoff', type=int, required=True, choices=(1,2))
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
    print(json.dumps({label: dict(ranks=result[label]['actual_positive_ranks'], matrix_rows=len(result[label]['actual_every_neutral_energy_full_2x2_face_spectral_matrix_rows']), cases=[{key: row[key] for key in ('actual_face_packet_q','actual_photon_energy_mean','actual_M0','actual_E1','actual_R1','actual_energy_mean','actual_raw_source_Knull_norm_squared')} for row in result[label]['actual_face_packet_cases']]) for label in ('actual_UE1_fullpositive','actual_same_factor_Haar_E0_endpoint')}), flush=True)


if __name__ == '__main__':
    main()
