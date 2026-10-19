#!/usr/bin/env python3
"""Declared UE/auxiliary-mass discriminator of frozen 3a53 full quantum SF.

All Phi3, 70/56 active and all six canonical auxiliary occupations, physical
Gauss columns, literal source and Ref4 coherent consumer remain unchanged.
Primitive electric heat is an INPUT, not an induced Maxwell explanation.
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
INITIAL_BYTES = {p.name: p.read_bytes() for p in ROOT.glob('*.py')}
PLAN = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA_AT_IMPORT = hashlib.sha256(PLAN.read_bytes()).hexdigest()
BASELINE_NAMES = ('covariant_native_electric_heat_cut2_UE1_capsule_20261008.json',
                  'covariant_native_electric_heat_cut3_UE1_capsule_20261008.json',
                  'covariant_native_heavy_loop_scale_k04_grid32_capsule_20261008.json',
                  'covariant_native_heavy_loop_scale_k02_grid32_capsule_20261008.json',
                  'covariant_native_heavy_loop_scale_k01_grid32_capsule_20261008.json')
BASELINE_BYTES = {name: (ROOT/name).read_bytes() for name in BASELINE_NAMES}
from covariant_native_electric_heat_comparison import prepare_spectrum
from covariant_native_closed_auxiliary_loop import auxiliary_path_half
from covariant_native_source_gram_moments import matched_tree_half
from covariant_induced_photon import Parameters, phi_frame
import covariant_native_haar_range_jets as native
FROZEN_SHA = '3a53c515002dda7c9ed7fb3667574dac94976ad717d0a5f14b681d107416187b'
if hashlib.sha256(INITIAL_BYTES['covariant_native_electric_heat_comparison.py']).hexdigest() != FROZEN_SHA:
    raise RuntimeError('Frozen positive primitive-electric supplier revision differs')


def phi_sf_factor(gamma: np.ndarray, y: np.ndarray, heat: np.ndarray,
                  physical: dict[str, Any], grid: int, kaux: float) -> dict[str, Any]:
    """Same frozen SF with explicit kaux; E weighting occurs only downstream."""
    light = np.array(physical['light']); aux = np.array(physical['auxiliary'])
    rows = np.array([state for state, _, _ in physical['physical']])
    flux = np.array([n for _, n, _ in physical['physical']]); nr = len(rows)
    F = np.zeros((3*nr, 3*nr), complex); lower = []; defects = []
    for p, normal in enumerate(y):
        actual = gamma.copy(); actual[1] *= 1+float(normal)
        samples = []; point_lower = []
        for theta in 2*math.pi*np.arange(grid)/grid:
            half, pref = matched_tree_half(actual, float(theta), 0.)
            L = pref*np.linalg.det(half[light[:, None, :, None], light[None, :, None, :]])
            half, pref = auxiliary_path_half(actual, kaux, complex((1+np.exp(1j*theta))/2))
            A = pref*np.linalg.det(half[aux[:, None, :, None], aux[None, :, None, :]])
            point_lower.append(float(eigh(L, eigvals_only=True, subset_by_index=[0, 0])[0]*
                                     eigh(A, eigvals_only=True, subset_by_index=[0, 0])[0]))
            samples.append(np.kron(L, A))
        coeff = np.fft.fft(np.array(samples), axis=0)/grid
        SF = coeff[(flux[:, None]-flux[None, :]) % grid, rows[:, None], rows[None, :]]
        defects.append(float(np.linalg.norm(SF-SF.conj().T)/np.linalg.norm(SF)))
        lower.append(min(point_lower))
        for out in range(3):
            F[out*nr:(out+1)*nr, p*nr:(p+1)*nr] = heat[out, p]*SF
    middle = np.array([j for j, (_, _, e) in enumerate(physical['physical']) if all(x == 0 for x in e)], dtype=int)
    return dict(F=F, middle_columns=np.concatenate([middle+p*nr for p in range(3)]),
                squared_ALL4edge_electric=np.tile([sum(e*e for e in electric) for _, _, electric in physical['physical']], 3),
                diagnostics=dict(auxiliary_kappa_at_every_half_and_pref=kaux,
                    actual_point_fullcanonical_SF_positive_lower_bounds=lower,
                    actual_projected_SF_hermitian_relative_defects=defects,
                    actual_complete_physical_columns=3*nr, actual_totalrho0_Haar_columns=3*len(middle)))


def observe(n: dict[str, Any], c: dict[str, Any], neutral: dict[str, Any], charged: dict[str, Any],
            S: Any, Sout: Any, gap: float) -> dict[str, Any]:
    Vn = n['V']; Vc = c['V']; lambdaN = float(n['q'][0]**2)
    qn = n['q']/math.sqrt(lambdaN); qc = c['q']/math.sqrt(lambdaN)
    En = -2*np.log(qn); Ec = -2*np.log(qc); Omega = Vn[:, 0]
    Bn = native.curl(neutral, 3); Bc = native.curl(charged, 3)
    rawph = Bn@Omega; projection = Vn.conj().T@rawph; ph = qn*projection
    Zph = float(np.vdot(ph, ph).real)
    if Zph <= 0: raise ArithmeticError('Actual photon preparation zero')
    ph /= math.sqrt(Zph); incoming = Vn@ph
    raw = (S@incoming).reshape(4, Vc.shape[0])
    projected = np.array([Vc.conj().T@row for row in raw]); Xi = qc[None, :]*projected
    weights = np.sum(abs(Xi)**2, axis=0); phweights = abs(ph)**2
    M0 = float(np.sum(weights))
    if M0 <= 0: raise ArithmeticError('Actual physical source zero')
    E1 = float(np.sum(weights*Ec)); R1 = float(np.sum(weights*qc**2))
    source_null = np.array([row-Vc@column for row, column in zip(raw, projected)])
    target = (Sout@Omega).reshape(4, Vc.shape[0])
    receiver = np.array([Vc.conj().T@(Bc@row) for row in target]); consumers = []
    for elapsed in (0., .5, 1., 2.):
        evolved = Xi*np.exp(-1j*elapsed*Ec)[None, :]/math.sqrt(M0)
        amplitude = np.vdot(receiver, evolved)
        consumers.append(dict(time=elapsed, coherent_Ref4_receiver_real=float(amplitude.real),
            coherent_Ref4_receiver_imag=float(amplitude.imag), coherent_Ref4_receiver_probability=float(abs(amplitude)**2),
            evolved_source_norm_squared=float(np.vdot(evolved, evolved).real)))
    return dict(actual_common_neutral_lambdaN=lambdaN, actual_positive_ranks=[len(qn), len(qc)],
        actual_normalized_min_positive_singular_values=[float(qn[-1]), float(qc[-1])],
        actual_photon_Zph=Zph, actual_photon_energy_mean=float(np.sum(phweights*En)),
        actual_photon_spectral_rows=[dict(energy=float(e), normalized_weight=float(w)) for e, w in zip(En, phweights)],
        actual_charged_source_spectral_rows=[dict(energy=float(e), absolute_weight=float(w), normalized_weight=float(w/M0)) for e, w in zip(Ec, weights)],
        actual_photon_mean_over_auxiliary_mean_PH_gap=float(np.sum(phweights*En)/gap),
        actual_source_mean_over_auxiliary_mean_PH_gap=E1/M0/gap,
        actual_photon_weight_below_auxiliary_mean_PH_gap=float(np.sum(phweights[En < gap])),
        actual_source_weight_below_auxiliary_mean_PH_gap=float(np.sum(weights[Ec < gap])/M0),
        actual_M0=M0, actual_E1=E1, actual_R1=R1, actual_energy_mean=E1/M0, actual_transfer_Rayleigh_mean=R1/M0,
        actual_raw_photon_Knull_norm_squared=float(np.vdot(rawph-Vn@projection, rawph-Vn@projection).real),
        actual_raw_source_Knull_norm_squared=float(np.vdot(source_null, source_null).real),
        actual_charged_source_absolute_band_rows=[dict(absolute_threshold=t, source_fraction=float(np.sum(weights[Ec <= t])/M0)) for t in (.25, 2., 4.)],
        actual_same_literal_source_consumer_rows=consumers,
        actual_neutral_phi_node_probabilities=np.sum(abs(Omega.reshape(3, -1))**2, axis=1).tolist())


def run(cutoff: int, grid: int, cases: list[tuple[float, float]]) -> dict[str, Any]:
    gamma4, metric = phi_frame(Parameters(kappa_power=400))
    ev, U = eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]); embedding = U[:, ev > 0]
    _, frame = eigh(embedding.conj().T@gamma4[0]@embedding); embedding = embedding@frame[:, ::-1]
    gamma = np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    y = np.array([-.09, 0., .09]); sigma = .03; radius = .12; dy = .09
    k = np.exp(-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y))); k /= np.linalg.norm(k)
    D = np.zeros((2, 3))
    for j in range(2):
        D[j, j] = math.sqrt(k[j+1]/k[j])/dy; D[j, j+1] = -math.sqrt(k[j]/k[j+1])/dy
    ev, U = eigh(2*sigma*sigma*D.T@D); heat = (U*np.exp(-ev/2))@U.T
    neutral = native.space(0, cutoff); charged = native.space(1, cutoff)
    packet = np.array([.1, .8, .4, .1]); packet /= np.linalg.norm(packet)
    outpacket = np.array([.5, .4, .1, .6]); outpacket /= np.linalg.norm(outpacket)
    S = native.source(neutral, charged, native.source_coefficients(gamma, packet), embedding, 3)
    Sout = native.source(neutral, charged, native.source_coefficients(gamma, outpacket), embedding, 3)
    results = []; means = []; endpoints = []
    for kaux in dict.fromkeys(k for k, _ in cases):
        start = time.monotonic()
        mean, pref = auxiliary_path_half(gamma, kaux, 1.+0j)
        onebody = np.sort(-2*np.log(eigh(mean, eigvals_only=True))); gap = float(onebody[2]-onebody[1])
        canonical = np.sort([sum(onebody[list(pair)])-2*math.log(pref) for pair in itertools.combinations(range(4), 2)])
        canonical -= canonical[0]
        means.append(dict(auxiliary_kappa=kaux, actual_full_onebody_SF_energies=onebody.tolist(), actual_full_onebody_SF_prefactor=pref,
            actual_auxiliary_mean_PH_gap=gap, actual_ALL6canonical_excitation_energies=canonical.tolist()))
        print(json.dumps(dict(stage='constructing_all_columns_PhiSF_without_E', auxiliary_kappa=kaux, gap=gap)), flush=True)
        n = phi_sf_factor(gamma, y, heat, neutral, grid, kaux); c = phi_sf_factor(gamma, y, heat, charged, grid, kaux)
        haar = observe(prepare_spectrum(n['F'][:, n['middle_columns']].copy()),
                       prepare_spectrum(c['F'][:, c['middle_columns']].copy()), neutral, charged, S, Sout, gap)
        endpoints.append(dict(auxiliary_kappa=kaux, actual_same_regulator_Haar_endpoint=haar))
        for case_k, UE in cases:
            if case_k != kaux: continue
            case_start = time.monotonic()
            wn = np.exp(-UE*n['squared_ALL4edge_electric']/2); wc = np.exp(-UE*c['squared_ALL4edge_electric']/2)
            if min(wn.min(), wc.min()) <= 0: raise ArithmeticError('Declared Eheat underflow; no null clipping')
            print(json.dumps(dict(stage='executing_full_positive_E_spectra', auxiliary_kappa=kaux, UE=UE)), flush=True)
            sn = prepare_spectrum(n['F']*wn[None, :]); sc = prepare_spectrum(c['F']*wc[None, :])
            obs = observe(sn, sc, neutral, charged, S, Sout, gap); del sn, sc
            results.append(dict(auxiliary_kappa=kaux, primitive_electric_input=UE, runtime_seconds=time.monotonic()-case_start,
                actual_factor_diagnostics={'neutral':n['diagnostics'], 'charged':c['diagnostics']},
                actual_Eheat_min=[float(wn.min()), float(wc.min())], actual_primitive_electric_heat_transfer=obs))
            print(json.dumps(dict(stage='case_complete', auxiliary_kappa=kaux, UE=UE,
                actual_photon_energy_mean=obs['actual_photon_energy_mean'], actual_source_energy_mean=obs['actual_energy_mean'], actual_M0=obs['actual_M0'])), flush=True)
        del n, c
        print(json.dumps(dict(stage='auxiliary_group_complete', auxiliary_kappa=kaux, runtime_seconds=time.monotonic()-start)), flush=True)
    descriptor = dict(plan_id=PLAN.stem, plan_sha256_at_IMPORT=PLAN_SHA_AT_IMPORT,
        pinned_general='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe', frozen_positive_primitive_SHA256=FROZEN_SHA,
        cases=[dict(auxiliary_kappa=k, primitive_electric_input=u) for k, u in cases],
        parameter_semantics='kaux is passed explicitly to EVERY sampled auxiliary half/pref and full onebody mean-gap; no monkeypatch/fallback. UE multiplies ALL4edge sum E² in the complete physical column space.',
        factor_reuse='For each kaux compute A=Phiheat_half SF_half once; F_U=A exp(-UE sum_ALL4edge E²/2). Phiheat remains LEFT. Haar is exactly the E0 columns of A at SAME declared regulator.',
        active_kappa=.09, light_homotopy=0., auxiliary_orbitals=4, auxiliary_canonical_number=2, ALL_auxiliary_CAR_states=6,
        active_modes=8, active_canonical_numbers=[4,5], active_CAR_states=[70,56], original_Ref=4, prepared_Ref=3, static_Ref_site=0,
        source_packet=packet.tolist(), receiver_packet=outpacket.tolist(), source_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        source_absolute_energy_thresholds=[.25,2.,4.], Phi_nodes=y.tolist(), Phi_sigma=sigma, Phi_radius=radius,
        Haar_quadrature_grid=grid, NEW_declared_electric_Galerkin_cutoff=cutoff, totalGauss='ONE active+auxiliary+staticRef total density; all legal integer states at cutoff',
        source_contract='Fixed active kappa .09 source mean-positive filter; physical spin embedding into ALL4 Refholes with exact CAR. Xi_ph=sqrt(KN/lambdaN) sin(theta) Omega/sqrtZph; Xi_ch=sqrt(KC/lambdaN) S_literal Xi_ph. M0=norm²Xi; E1=<Xi,-log(KC/lambdaN)Xi>; R1=<Xi,KC/lambdaN Xi>. ONE whole charged halfstep pervector; Ref4 receiver coherent before abs².',
        transfer_duration=1., spatial_spacing=1., no_speed_mass_quarter_fit=True, no_occupation_Phi_source_Pe_truncation=True,
        scope='Finite singleplaquette primitive-E compact rotor discriminator, not massless dispersion, commoncone, induced E/Maxwell or fullCartesianPhi/SU4/macro admission')
    sources = {Path(__file__).name: INITIAL_BYTES[Path(__file__).name]}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).parent == ROOT:
            name = Path(filename).name; sources[name] = INITIAL_BYTES[name]
    return dict(input_descriptor=descriptor, input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name, data in sources.items()},
        source_bytes_at_import={name:data.decode() for name, data in sources.items()},
        reference_capsule_hashes_at_IMPORT={name:hashlib.sha256(data).hexdigest() for name, data in BASELINE_BYTES.items()},
        runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        physical_mean_descriptor=metric, auxiliary_mean_rows=means, actual_Haar_endpoint_rows=endpoints, actual_case_rows=results,
        massless_commoncone_inducedE_Maxwell_macro_admitted=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--electric-cutoff', type=int, required=True); parser.add_argument('--haar-grid', type=int, required=True)
    parser.add_argument('--case', nargs=2, type=float, action='append', required=True, metavar=('KAUX','UE'))
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    start = time.monotonic(); result = run(args.electric_cutoff, args.haar_grid, [tuple(case) for case in args.case])
    result['runtime_seconds'] = time.monotonic()-start; args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), source_at_import=result['source_at_import'], input_descriptor_sha256=result['input_descriptor_sha256'], runtime_seconds=result['runtime_seconds'])), flush=True)

if __name__ == '__main__':
    main()
