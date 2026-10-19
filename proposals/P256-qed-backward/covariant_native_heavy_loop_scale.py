#!/usr/bin/env python3
"""Full-canonical heavy-kappa discrimination of the frozen N1 two-path law.

Only the declared auxiliary kappa changes. All six canonical occupations,
Phi3, literal Gauss/source/receiver and physical sin(theta) are retained.
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
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = (
    'covariant_native_heavy_loop_scale.py', 'covariant_native_closed_auxiliary_loop.py',
    'covariant_native_source_gram_moments.py', 'covariant_native_haar_range_jets.py',
    'covariant_source_band_halfstep_control.py', 'covariant_induced_photon.py',
    'covariant_photon_energy_control.py', 'sea_selection.py', 'rotation_band.py')
SOURCE_BYTES_AT_IMPORT = {name: (ROOT / name).read_bytes() for name in SOURCE_NAMES}
PLAN_PATH = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA256_AT_IMPORT = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
from covariant_native_closed_auxiliary_loop import auxiliary_path_half
from covariant_native_source_gram_moments import matched_tree_half
import covariant_native_haar_range_jets as native
from covariant_induced_photon import Parameters, phi_frame


def reflected_factor(gamma: np.ndarray, y: np.ndarray, heat_half: np.ndarray,
                     physical: dict[str, Any], grid: int, kaux: float) -> np.ndarray:
    """Exact same N1 factor construction with kaux explicit at every heavy call."""
    middle = [physical['states'][i] for i in physical['middle']]
    ls = sorted(set(l for l, _ in middle)); ars = sorted(set(a for _, a in middle))
    li = {l: i for i, l in enumerate(ls)}; ai = {a: i for i, a in enumerate(ars)}
    lc = np.array([li[l] for l, _ in middle]); ac = np.array([ai[a] for _, a in middle])
    lr = np.array(physical['light']); la = np.array(ls)
    ar = np.array(physical['auxiliary']); aa = np.array(ars)
    rows = np.array([i for i, _, _ in physical['physical']])
    flux = np.array([n % grid for _, n, _ in physical['physical']])
    blocks = []; tail = 0.; sample_norm = 0.
    for normal in y:
        actual = gamma.copy(); actual[1] *= 1 + float(normal)
        samples = []
        for theta in 2 * math.pi * np.arange(grid) / grid:
            half, pref = matched_tree_half(actual, float(theta), 0.)
            light = pref * np.linalg.det(half[lr[:, None, :, None], la[None, :, None, :]])
            heavy, hpref = auxiliary_path_half(actual, kaux, complex((1 + np.exp(1j * theta)) / 2))
            aux = hpref * np.linalg.det(heavy[ar[:, None, :, None], aa[None, :, None, :]])
            combined = light[:, lc][:, None, :] * aux[None, :, ac]
            samples.append(combined.reshape(len(physical['states']), len(middle)))
        coefficients = np.fft.fft(np.array(samples), axis=0) / grid
        blocks.append(coefficients[flux, rows, :])
        high = [j for j in range(grid) if min(j, grid-j) > 12]
        tail = max(tail, float(np.linalg.norm(coefficients[high])))
        sample_norm = max(sample_norm, float(np.linalg.norm(coefficients)))
    nr = len(rows); nc = len(middle)
    result = np.zeros((3 * nr, 3 * nc), complex)
    for out in range(3):
        for inp in range(3):
            result[out*nr:(out+1)*nr, inp*nc:(inp+1)*nc] = heat_half[out, inp] * blocks[inp]
    physical['Fourier_tail_above12'] = tail
    physical['Fourier_total_norm'] = sample_norm
    physical['actual_kaux_under_all_kernel_factors'] = kaux
    return result


def spectral_support(energies: np.ndarray, weights: np.ndarray, gap: float) -> dict[str, Any]:
    total = float(np.sum(weights))
    if total <= 0:
        raise ArithmeticError('No nonzero spectral state to normalize')
    fractions = weights / total
    ordering = np.argsort(energies); cumulative = np.cumsum(fractions[ordering])
    quantiles = {str(p): float(energies[ordering[min(int(np.searchsorted(cumulative, p)), len(ordering)-1)]])
                 for p in (.01, .1, .5, .9, .99)}
    return dict(energies=energies.tolist(), normalized_weights=fractions.tolist(),
                mean=float(np.sum(fractions * energies)), energy_quantiles=quantiles,
                fractions_below_absolute_threshold={str(t): float(np.sum(fractions[energies <= t])) for t in (.25, 2., 4.)},
                fraction_below_actual_auxiliary_PH_gap=float(np.sum(fractions[energies < gap])),
                mean_over_actual_auxiliary_PH_gap=float(np.sum(fractions * energies) / gap),
                positive_rank=len(energies), minimum_positive_transfer_energy=float(np.min(energies)),
                maximum_positive_transfer_energy=float(np.max(energies)))


def run(grid: int, kaux: float) -> dict[str, Any]:
    if grid not in (32, 64) or kaux not in (.04, .02, .01):
        raise ValueError('This physical discriminator declares only Haar32/64 and kaux .04/.02/.01')
    gamma4, metric = phi_frame(Parameters(kappa_power=400))
    values, U = eigh(1j * gamma4[0] @ gamma4[1] @ gamma4[2]); embedding = U[:, values > 0]
    _, frame = eigh(embedding.conj().T @ gamma4[0] @ embedding); embedding = embedding @ frame[:, ::-1]
    gamma = np.array([embedding.conj().T @ g @ embedding for g in gamma4[:3]])
    links = np.array([.17, -.23, .31, .47]); site_gauge = np.array([.13, -.37, .29, .41])
    incidence = np.zeros((4, 4), int)
    for e in range(4): incidence[e, e] = 1; incidence[(e+1) % 4, e] = -1
    def phase(A: np.ndarray) -> complex:
        return complex((np.exp(1j*A[3]) + np.exp(-1j*np.sum(A[:3]))) / 2)
    old_mean, old_pref = native.auxiliary_half(gamma, kaux, 0.)
    mean, pref = auxiliary_path_half(gamma, kaux, 1.+0j)
    mean_defect = float(np.linalg.norm(old_mean-mean, 2) + abs(old_pref-pref))
    h, _ = auxiliary_path_half(gamma, kaux, phase(links))
    hg, _ = auxiliary_path_half(gamma, kaux, phase(links+incidence.T@site_gauge))
    G = np.diag(np.repeat(np.exp(1j*site_gauge[[0, 3]]), 2))
    gauge_defect = float(np.linalg.norm(hg-G@h@G.conj().T, 2))
    if max(mean_defect, gauge_defect) > 1e-11:
        raise ArithmeticError('Actual parameter-dependent positive half/gauge control failed')
    onebody = np.sort(-2*np.log(eigh(mean, eigvals_only=True)))
    ph_gap = float(onebody[2]-onebody[1])
    canonical = np.sort(np.array([sum(onebody[list(pair)])-2*math.log(pref)
                                  for pair in itertools.combinations(range(4), 2)]))
    canonical -= canonical[0]
    y = np.array([-.09, 0., .09]); sigma = .03; radius = .12; dy = .09
    logk = -y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y))
    k = np.exp(logk); k /= np.linalg.norm(k); D = np.zeros((2, 3))
    for j in range(2):
        D[j, j] = math.sqrt(k[j+1]/k[j])/dy; D[j, j+1] = -math.sqrt(k[j]/k[j+1])/dy
    Hphi = 2*sigma*sigma*D.T@D; ev, evec = eigh(Hphi)
    heat_half = (evec*np.exp(-ev/2))@evec.T
    neutral = native.space(0, 12); charged = native.space(1, 12)
    Fn = reflected_factor(gamma, y, heat_half, neutral, grid, kaux)
    Fc = reflected_factor(gamma, y, heat_half, charged, grid, kaux)
    Vn, qn, _ = np.linalg.svd(Fn, full_matrices=False)
    Vc, qc, _ = np.linalg.svd(Fc, full_matrices=False)
    lambdaN = float(qn[0]**2); qn /= math.sqrt(lambdaN); qc /= math.sqrt(lambdaN)
    if min(qn[-1], qc[-1]) <= 0:
        raise ArithmeticError('Full canonical positive range lost rank')
    En = -2*np.log(qn); Ec = -2*np.log(qc); Omega = Vn[:, 0]
    del Fn, Fc
    Bn = native.curl(neutral, 3); Bc = native.curl(charged, 3)
    raw_ph = Bn@Omega; projected_ph = Vn.conj().T@raw_ph
    ph_null = raw_ph-Vn@projected_ph
    photon = qn*projected_ph; Zph = float(np.vdot(photon, photon).real)
    # Conservative floating-point amplitude guard, not a physical support cut.
    noise_amplitude_guard = 256*np.finfo(float).eps*max(Vn.shape)
    if math.sqrt(Zph) <= noise_amplitude_guard:
        raise ArithmeticError('Prepared photon amplitude is unresolved against numerical Fourier/SVD noise; no normalized state manufactured')
    photon /= math.sqrt(Zph); incoming = Vn@photon
    prepared_ph_null = incoming-Vn@(Vn.conj().T@incoming)
    packet = np.array([.1, .8, .4, .1]); packet /= np.linalg.norm(packet)
    outpacket = np.array([.5, .4, .1, .6]); outpacket /= np.linalg.norm(outpacket)
    S = native.source(neutral, charged, native.source_coefficients(gamma, packet), embedding, 3)
    Sout = native.source(neutral, charged, native.source_coefficients(gamma, outpacket), embedding, 3)
    raw = (S@incoming).reshape(4, Vc.shape[0])
    projections = np.array([Vc.conj().T@row for row in raw])
    source = projections*qc[None, :]
    M0 = float(np.vdot(source, source).real)
    if M0 <= noise_amplitude_guard**2:
        raise ArithmeticError('Prepared charged state is unresolved; no normalized consumer manufactured')
    weights = np.sum(abs(source)**2, axis=0)
    E1 = float(np.sum(weights*Ec)); R1 = float(np.sum(weights*qc**2))
    raw_null = np.array([row-Vc@projection for row, projection in zip(raw, projections)])
    prepared_rows = source@Vc.T
    prepared_null = np.array([row-Vc@(Vc.conj().T@row) for row in prepared_rows])
    target = (Sout@Omega).reshape(4, Vc.shape[0])
    receiver = np.array([Vc.conj().T@(Bc@row) for row in target])
    consumers = []
    for elapsed in (0., .5, 1., 2.):
        evolved = source*np.exp(-1j*elapsed*Ec)[None, :]/math.sqrt(M0)
        amplitude = np.vdot(receiver, evolved)
        consumers.append(dict(time=elapsed, coherent_Ref4_receiver_real=float(amplitude.real),
                              coherent_Ref4_receiver_imag=float(amplitude.imag),
                              coherent_Ref4_receiver_probability=float(abs(amplitude)**2),
                              evolved_source_norm_squared=float(np.vdot(evolved, evolved).real)))
    positive_curl = Vn.conj().T@(Bn@Vn)
    Eavg = np.tile(np.array([sum(e)/4 for _, _, e in neutral['physical']]), 3)
    static = np.tile(np.array([np.array(native.density(neutral['states'][i][0], neutral['states'][i][1], 0))@np.array([0., -.25, -.5, .25])
                              for i, _, _ in neutral['physical']]), 3)
    independent_E = Vn.conj().T@((Eavg-static)[:, None]*Vn)
    gauss_defect = 0
    for sector in (neutral, charged):
        for state, _, electric in sector['physical']:
            rho = native.density(sector['states'][state][0], sector['states'][state][1], sector['charge'])
            gauss_defect = max(gauss_defect, max(abs(electric[x]-electric[(x-1)%4]-rho[x]) for x in range(4)))
    descriptor = dict(governing_plan_id='issue-222-induced-photon-delta13-continuation-20261008',
        plan_sha256_at_IMPORT=PLAN_SHA256_AT_IMPORT,
        pinned_general='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        primitive_ABI='NEW-native-N1-two-path-heavy-kappa-discriminator-20261008',
        frozen_primitive_sha256='c6e5eda6241ba7929da2f1e203930c9b5af59807883bffa0f6b9b7cac867a0d2',
        auxiliary_kappa=kaux, actual_kaux_under_all_kernel_factors={'neutral':neutral['actual_kaux_under_all_kernel_factors'],
            'charged':charged['actual_kaux_under_all_kernel_factors'], 'mean_gap':kaux, 'gauge_controls':kaux},
        active_kappa=.09, light_homotopy=0., alternative_path_weights=[.5, .5],
        auxiliary_sites=[0, 3], auxiliary_orbitals=4, auxiliary_canonical_number=2, auxiliary_canonical_states=6,
        active_canonical_numbers=[4, 5], original_Ref=4, prepared_Ref=3,
        integer_active_auxiliary_charges=[1, 1], integer_Ref_hole_charge=-1,
        source_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        source_packet=packet.tolist(), receiver_packet=outpacket.tolist(),
        photon_operator='physical sin(theta), independent of light matter homotopy',
        normal_nodes=y.tolist(), normal_sigma=sigma, normal_radius=radius,
        transfer_duration=1., spatial_spacing=1., Haar_quadrature_grid=grid, integer_electric_Galerkin_cutoff=12,
        absolute_source_band_thresholds=[.25,2.,4.], common_neutral_normalization=True,
        operator_contract='Xi_ph=sqrt(KN/lambdaN) sin(theta) Omega/sqrtZph; Xi_ch=sqrt(KC/lambdaN) S Xi_ph. M0=norm2 Xi_ch; E1=<Xi_ch,-log(KC/lambdaN)Xi_ch>; R1=<Xi_ch,KC/lambdaN Xi_ch>.',
        no_static_Slater_or_auxiliary_occupation_truncation=True, no_source_or_Pe_cut_change=True,
        no_mass_speed_fit=True, no_bare_gauge_action=True)
    return dict(input_descriptor=descriptor,
        input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        runtime_packages={'python':sys.version,'executable':sys.executable,'numpy':np.__version__, 'scipy':scipy.__version__, 'platform':platform.platform()},
        physical_mean_descriptor=metric, actual_auxiliary_mean_equals_original_mean_defect=mean_defect,
        actual_positive_auxiliary_half_gauge_covariance_defect=gauge_defect, actual_local_integer_Gauss_defect=gauss_defect,
        actual_auxiliary_mean_onebody_energies=onebody.tolist(), actual_auxiliary_mean_PH_gap=ph_gap,
        actual_all6_auxiliary_mean_canonical_excitation_energies=canonical.tolist(),
        actual_common_neutral_lambda_N=lambdaN, actual_neutral_photon_Zph=Zph,
        numerical_preparation_amplitude_guard=noise_amplitude_guard,
        photon_amplitude_over_numerical_guard=math.sqrt(Zph)/noise_amplitude_guard,
        actual_positive_range_physical_sin_theta_norm=float(np.linalg.norm(positive_curl,2)),
        actual_positive_range_Etheta_minus_matter_density_connection_norm=float(np.linalg.norm(independent_E,2)),
        actual_source_M0_including_whole_charged_halfstep=M0,
        actual_energy_moment_Htransfer=E1, actual_transfer_moment_Kcharged_over_lambda_neutral=R1,
        actual_energy_mean=E1/M0, actual_transfer_Rayleigh_mean=R1/M0,
        actual_photon_spectral_support=spectral_support(En,abs(photon)**2,ph_gap),
        actual_source_spectral_support=spectral_support(Ec,weights,ph_gap),
        actual_raw_photon_Knull_norm_squared=float(np.vdot(ph_null,ph_null).real),
        actual_raw_photon_norm_squared=float(np.vdot(raw_ph,raw_ph).real),
        actual_prepared_photon_Knull_norm_squared=float(np.vdot(prepared_ph_null,prepared_ph_null).real),
        actual_raw_source_Knull_norm_squared=float(np.vdot(raw_null,raw_null).real),
        actual_raw_source_norm_squared=float(np.vdot(raw,raw).real),
        actual_prepared_source_Knull_norm_squared=float(np.vdot(prepared_null,prepared_null).real),
        actual_prepared_source_Knull_fraction=float(np.vdot(prepared_null,prepared_null).real)/M0,
        actual_same_literal_source_consumer_rows=consumers,
        all_canonical_positive_ranks={'neutral':len(qn),'charged_per_Ref':len(qc)},
        actual_minimum_normalized_positive_transfer_eigenvalues={'neutral':float(qn[-1]**2),'charged':float(qc[-1]**2)},
        Fourier_above12={s:{'norm':d['Fourier_tail_above12'],'relative_total':d['Fourier_tail_above12']/d['Fourier_total_norm']}
                        for s,d in (('neutral',neutral),('charged',charged))},
        massless_photon_or_common_cone_or_macro_source_admitted=False,
        scope='Finite Haar32/64 and electricGalerkin12, mean invariant2spin Phi3, distinct coarse3link primitive. PH comparison is actual MEAN auxiliary gap, not exact interacting excitation labeling; no 4D Wilson/fullCartesianPhi/SU4/macroscopic pole or phase closure.')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaux', type=float, required=True)
    parser.add_argument('--haar-grid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    result = run(args.haar_grid, args.kaux); result['runtime_seconds'] = time.monotonic()-start
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    fields = ('input_descriptor_sha256','actual_auxiliary_mean_PH_gap','actual_neutral_photon_Zph',
              'actual_source_M0_including_whole_charged_halfstep','actual_energy_mean',
              'actual_transfer_Rayleigh_mean','actual_same_literal_source_consumer_rows','runtime_seconds')
    summary = {key:result[key] for key in fields}
    summary['actual_kaux'] = args.kaux
    summary['photon_energy_mean'] = result['actual_photon_spectral_support']['mean']
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
