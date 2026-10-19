#!/usr/bin/env python3
"""Actual static-Ref dipole site GF on frozen native primitive-electric UE1.

Four delta-site GF probes are distinct from the unchanged scattering packet.
Every charged level and every Ref4 complex amplitude is retained. Ring labels
are spectral packet projections, not translation eigenmomenta or cone data.
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

ROOT = Path(__file__).resolve().parent
INITIAL_BYTES = {p.name: p.read_bytes() for p in ROOT.glob('*.py')}
PLAN_PATH = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA_AT_IMPORT = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
FROZEN_NATIVE_SHA = '3a53c515002dda7c9ed7fb3667574dac94976ad717d0a5f14b681d107416187b'
if hashlib.sha256(INITIAL_BYTES['covariant_native_electric_heat_comparison.py']).hexdigest() != FROZEN_NATIVE_SHA:
    raise RuntimeError('Native primitive-electric source is not the assigned frozen source')
import numpy as np
import scipy
from scipy.linalg import eigh
import covariant_native_electric_heat_comparison as electric


def complex_array(value: np.ndarray) -> dict[str, Any]:
    return dict(real=np.real(value).tolist(), imag=np.imag(value).tolist())


def sources_at_import() -> dict[str, Any]:
    result = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().parent == ROOT:
            name = Path(filename).name
            data = INITIAL_BYTES[name]
            result[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=data.decode())
    return result


def raw_addition_gf(amplitudes: np.ndarray, energy: np.ndarray) -> dict[str, Any]:
    W = np.einsum('xrj,yrj->jxy', amplitudes.conj(), amplitudes)
    eigenvalues = np.linalg.eigvalsh(W)
    weights = np.sum(abs(amplitudes)**2, axis=1)
    norms = weights.sum(axis=1)
    integrated = W.sum(axis=0)
    integrated_eigenvalues = np.linalg.eigvalsh(integrated)
    tolerance = float(4*np.finfo(float).eps*max(abs(integrated_eigenvalues)))
    times = []
    for t in (0., .5, 1., 2.):
        G = np.einsum('j,jxy->xy', np.exp(-1j*t*energy), W)
        Gminus = np.einsum('j,jxy->xy', np.exp(1j*t*energy), W)
        times.append(dict(time=t, site_GF=complex_array(G),
            negative_time_adjoint_defect=float(np.linalg.norm(Gminus-G.conj().T))))
    ring = []
    for p in (0., math.pi/2, math.pi):
        packet = np.exp(1j*p*np.arange(4))/2
        a = np.einsum('x,xrj->rj', packet, amplitudes)
        w = np.sum(abs(a)**2, axis=0); total = float(w.sum())
        direct = np.einsum('x,jxy,y->j', packet.conj(), W, packet).real
        ring.append(dict(spectral_packet_label=p, packet=complex_array(packet),
            all_Ref_complex_amplitudes=complex_array(a), absolute_energy_weights=w.tolist(),
            normalized_energy_weights=(w/total).tolist(), raw_addition_norm_squared=total,
            raw_addition_energy_mean=float(w@energy/total),
            spectral_matrix_projection_max_absolute_defect=float(np.max(abs(direct-w)))))
    lowest = []
    for j in np.argsort(energy)[:12]:
        lowest.append(dict(level=int(j), energy=float(energy[j]),
            all_site_Ref_complex_amplitudes=complex_array(amplitudes[:, :, j]),
            site_spectral_matrix=complex_array(W[j]),
            all_site_absolute_residues=weights[:, j].tolist(),
            all_ring_packet_absolute_residues=[row['absolute_energy_weights'][j] for row in ring]))
    return dict(preparation='RAW ADDITION S_x Omega; no charged halfstep applied',
        all_complex_amplitudes=complex_array(amplitudes), all_site_spectral_matrices=complex_array(W),
        all_site_absolute_energy_weights=weights.tolist(),
        all_site_normalized_energy_weights=(weights/norms[:, None]).tolist(),
        site_raw_addition_norm_squared=norms.tolist(),
        site_raw_addition_energy_means=(weights@energy/norms).tolist(),
        real_time_site_GF=times, ring_spectral_packet_projections=ring,
        lowest_12_levels_with_all_residues=lowest,
        spectral_positivity=dict(all_level_eigenvalues=eigenvalues.tolist(),
            minimum_eigenvalue=float(eigenvalues.min()),
            maximum_Hermiticity_defect=float(np.linalg.norm(W-W.conj().transpose(0, 2, 1), axis=(1, 2)).max()),
            integrated_matrix=complex_array(integrated),
            integrated_eigenvalues=integrated_eigenvalues.tolist(),
            integrated_numerical_rank=int(np.sum(integrated_eigenvalues > tolerance)),
            integrated_rank_roundoff_tolerance=tolerance,
            integrated_diagonal_probe_norm_max_defect=float(np.max(abs(integrated.diagonal().real-norms))),
            positivity_roundoff_scale=float(4*np.finfo(float).eps*np.max(abs(eigenvalues))),
            no_level_dropped=True))


def run() -> dict[str, Any]:
    gamma4, metric = electric.phi_frame(electric.Parameters(kappa_power=400))
    values, frame = eigh(1j*gamma4[0]@gamma4[1]@gamma4[2])
    embedding = frame[:, values > 0]
    _, spinframe = eigh(embedding.conj().T@gamma4[0]@embedding)
    embedding = embedding@spinframe[:, ::-1]
    gamma = np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    y = np.array([-.09, 0., .09]); sigma = .03; radius = .12
    k = np.exp(-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y)))
    k /= np.linalg.norm(k)
    D = np.zeros((2, 3))
    for j in range(2):
        D[j, j] = math.sqrt(k[j+1]/k[j])/.09
        D[j, j+1] = -math.sqrt(k[j]/k[j+1])/.09
    values, frame = eigh(2*sigma*sigma*D.T@D)
    heat = (frame*np.exp(-values/2))@frame.T
    neutral = electric.native.space(0, 2); charged = electric.native.space(1, 2)
    dimensions = [3*len(neutral['physical']), 3*len(charged['physical'])]
    diagnostics = {}; spectra = {}
    for label, physical in (('neutral', neutral), ('charged', charged)):
        print(json.dumps(dict(stage='actual_full_positive_factor_and_SVD', sector=label, dimensions=dimensions)), flush=True)
        factor = electric.full_factor(gamma, y, heat, physical, 32, 1.)
        diagnostics[label] = {key: value for key, value in factor.items() if key not in ('F', 'Haar')}
        del factor['Haar']
        spectra[label] = electric.prepare_spectrum(factor.pop('F'))
        del factor
    n = spectra['neutral']; c = spectra['charged']
    Vn = n['V']; Vc = c['V']; Omega = Vn[:, 0]
    lambdaN = float(n['q'][0]**2)
    qn = n['q']/math.sqrt(lambdaN); qc = c['q']/math.sqrt(lambdaN)
    Ec = -2*np.log(qc)
    if len(qn) != dimensions[0] or len(qc) != dimensions[1]:
        raise ArithmeticError('Full positive factors unexpectedly lost physical columns')
    packets = np.eye(4)
    raw = []; coefficients = []
    charged3 = electric.native.space(1, 3)
    retained3 = np.array([charged3['index'][(state, n)] for state, n, _ in charged['physical']])
    omitted3 = np.array([(state, n) not in charged['index'] for state, n, _ in charged3['physical']])
    source_boundary = []
    for packet in packets:
        coeff = electric.native.source_coefficients(gamma, packet)
        coefficients.append(coeff)
        Sx = electric.native.source(neutral, charged, coeff, embedding, 3)
        raw.append((Sx@Omega).reshape(4, Vc.shape[0]))
        S3 = electric.native.source(neutral, charged3, coeff, embedding, 3)
        raw3 = (S3@Omega).reshape(4, 3, len(charged3['physical']))
        retained = raw3[:, :, retained3]
        leakage = raw3[:, :, omitted3]
        source_boundary.append(dict(site=len(source_boundary),
            full_literal_source_from_Omega_cut2_norm_squared=float(np.sum(abs(raw3)**2)),
            retained_charged_cut2_norm_squared=float(np.sum(abs(retained)**2)),
            omitted_charged_cut2_norm_squared=float(np.sum(abs(leakage)**2)),
            omitted_per_Ref_norm_squared=np.sum(abs(leakage)**2, axis=(1, 2)).tolist(),
            retained_mapping_amplitude_defect=float(np.linalg.norm(retained.reshape(4, -1)-raw[-1])),
            exact_support_bound='All initial |E|<=2 and Wilson shifts |delta E|<=1: chargedcut3 contains entire literal source image from THIS Omega_cut2; no cutoff3 transfer/SVD used'))
    raw = np.array(raw)
    projected = (Vc.conj().T@raw.reshape(16, -1).T).T.reshape(4, 4, -1)
    z = projected*qc[None, None, :]
    null = raw-(Vc@projected.reshape(16, -1).T).T.reshape(raw.shape)
    raw_gf = raw_addition_gf(projected, Ec)
    W = np.einsum('xrj,yrj->jxy', z.conj(), z)
    Wvalues = np.linalg.eigvalsh(W)
    hermitian_defects = np.linalg.norm(W-W.conj().transpose(0, 2, 1), axis=(1, 2))
    norms = []
    for x in range(4):
        norms.append(dict(site=x, raw_norm_squared=float(np.sum(abs(raw[x])**2)),
            raw_per_Ref_norm_squared=np.sum(abs(raw[x])**2, axis=1).tolist(),
            whole_charged_halfstep_norm_squared=float(np.sum(abs(z[x])**2)),
            whole_halfstep_per_Ref_norm_squared=np.sum(abs(z[x])**2, axis=1).tolist(),
            true_Knull_loss_norm_squared=float(np.sum(abs(null[x])**2)),
            Knull_scope='Null projection AFTER P_C(Ecut2) S_full P_N(Ecut2); not omitted Wilson-source boundary loss',
            true_Knull_per_Ref_loss_norm_squared=np.sum(abs(null[x])**2, axis=1).tolist()))
    ring = []; projected_packets = []
    for p in (0., math.pi/2, math.pi):
        packet = np.exp(1j*p*np.arange(4))/2
        amplitude = np.einsum('x,xrj->rj', packet, z)
        weight = np.sum(abs(amplitude)**2, axis=0)
        total = float(weight.sum())
        direct = np.einsum('x,jxy,y->j', packet.conj(), W, packet).real
        projected_packets.append(weight)
        ring.append(dict(spectral_packet_label=p, packet=complex_array(packet),
            absolute_energy_weights=weight.tolist(), normalized_energy_weights=(weight/total).tolist(),
            all_Ref_complex_amplitudes=complex_array(amplitude),
            whole_halfstep_norm_squared=total, actual_energy_mean=float(weight@Ec/total),
            full_positive_charged_rank=len(qc),
            spectral_matrix_projection_max_absolute_defect=float(np.max(abs(direct-weight))),
            translation_eigenmomentum_claim=False))
    siteweights = np.sum(abs(z)**2, axis=1)
    site_means = np.sum(siteweights*Ec[None, :], axis=1)/np.sum(siteweights, axis=1)
    integrated = W.sum(axis=0)
    direct_integrated = np.einsum('xrj,yrj->xy', z.conj(), z)
    eigenvalues = np.linalg.eigvalsh(integrated)
    rank_tolerance = float(4*np.finfo(float).eps*max(abs(eigenvalues)))
    times = []
    for t in (0., .5, 1., 2.):
        G = np.einsum('j,jxy->xy', np.exp(-1j*t*Ec), W)
        Gminus = np.einsum('j,jxy->xy', np.exp(1j*t*Ec), W)
        times.append(dict(time=t, site_GF=complex_array(G),
            negative_time_adjoint_defect=float(np.linalg.norm(Gminus-G.conj().T))))
    packet = np.array([.1, .8, .4, .1]); packet /= np.linalg.norm(packet)
    receiver = np.array([.5, .4, .1, .6]); receiver /= np.linalg.norm(receiver)
    S = electric.native.source(neutral, charged, electric.native.source_coefficients(gamma, packet), embedding, 3)
    Sout = electric.native.source(neutral, charged, electric.native.source_coefficients(gamma, receiver), embedding, 3)
    baseline = electric.observe(n, c, neutral, charged, S, Sout)
    lowest = []
    # Ordered by the actual energy, not by a residue/cheap-state threshold.
    for j in np.argsort(Ec)[:12]:
        lowest.append(dict(level=int(j), energy=float(Ec[j]),
            all_site_Ref_complex_amplitudes=complex_array(z[:, :, j]),
            site_spectral_matrix=complex_array(W[j]),
            all_site_absolute_residues=siteweights[:, j].tolist(),
            all_ring_packet_absolute_residues=[float(w[j]) for w in projected_packets]))
    descriptor = dict(plan_id=PLAN_PATH.stem, plan_sha256_at_IMPORT=PLAN_SHA_AT_IMPORT,
        frozen_native_source_sha256=FROZEN_NATIVE_SHA,
        primitive='Declared native primitive-electric INPUT, not induced E/Maxwell explanation',
        factor='F=Phiheat_half SF_half exp(-UE sum_ALL4edges E^2/2); K=F Fdag; Phi remains LEFT',
        UE=1., electric_cutoff=2, Haar_grid=32, active_kappa=.09, auxiliary_kappa=.04,
        active_modes=8, active_particle_numbers=[4, 5], active_CAR_counts=[70, 56],
        auxiliary_modes=4, auxiliary_particle_number=2, auxiliary_CAR_count=6,
        Ref_hole_components=4, static_Ref_site=0, Ref_occupation_transition=[4, 3],
        selected_Clifford_dimension=2, full_Dirac4_or_full_Cartesian_or_SU4_claim=False,
        Phi_nodes=y.tolist(), Phi_sigma=sigma, Phi_radius=radius,
        global_Gauss='ONE total active+aux+staticRef density, original native.space at Ecut2',
        original_scattering_source_packet=packet.tolist(), original_receiver_packet=receiver.tolist(),
        original_Pe_absolute=[.25, 2., 4.], original_scattering_source_receiver_Pe_unchanged=True,
        GF_probe_packets=packets.tolist(),
        GF_probe_definition='S_x=native.source(native.source_coefficients(gamma,delta_x),embedding,3); original normalized mean-positive active Dirac filter at kappa .09',
        delta_packet_scope='delta_x is input to the mean-positive filter, which mixes sites; NOT a perfectly local delta electron after filtering',
        raw_probe_regulator_scope='All GF amplitudes/norms use P_C(Ecut2) S_full P_N(Ecut2) Omega_cut2; Knull residual is distinct from source boundary projection',
        source_boundary_scope='Separate chargedcut3 image computes exact lost Wilson-source norm from fixed Omega_cut2; NOT an Omega_infinity error bound',
        source_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        GF_contract='z_xrefj=qc_j <Vc_j,S_x Omega>; qc=qC/sqrt(lambdaN); lambdaN=qN_max^2; W_jxy=sum_ref conj(z_xrefj)z_yrefj; G_xy(t)=sum_j exp(-itE_j)W_jxy; E_j=-2log(qc_j)',
        raw_addition_GF_contract='a_xrefj=<Vc_j,S_x Omega>; Wraw_jxy=sum_ref conj(a_xrefj)a_yrefj; Graw_xy(t)=sum_j exp(-itE_j)Wraw_jxy; same Ec as wholehalf, different residues',
        GF_sector='N4 ground to N5+staticRefhole ADDITION sector only; N3 removal absent; NOT full Dirac/retarded CAR anticommutator; no Ref addition on filled Ref4',
        GF_source_scope='Gauge-invariant added active electron plus STATIC Ref hole/dipole; NOT a free-electron propagator',
        spectral_packet_labels=[0., math.pi/2, math.pi],
        ring_projection='p_x=exp(i label*x)/2; sum_x p_x S_x after separately normalized site filters; NOT newly normalized native.source_coefficients(gamma,p)',
        translation_eigenmomentum_claim=False, commoncone_admitted=False,
        transfer_duration=1., site_spacing=1., no_speed_mass_quarter_fit=True,
        all_charged_levels_retained=True, no_residue_energy_threshold_selection=True)
    result: dict[str, Any] = dict(input_descriptor=descriptor,
        input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        sources_at_IMPORT=sources_at_import(),
        runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        physical_metric=metric, selected_gamma=complex_array(gamma), selected_embedding=complex_array(embedding),
        actual_Omega=complex_array(Omega), actual_site_filter_coefficients=complex_array(np.array(coefficients)),
        actual_factor_diagnostics=diagnostics, actual_positive_ranks=[len(qn), len(qc)], actual_common_neutral_lambdaN=lambdaN,
        actual_min_normalized_positive_singular_values=[float(qn[-1]), float(qc[-1])],
        charged_levels=dict(levels=list(range(len(qc))), energy=Ec.tolist(), normalized_singular_value=qc.tolist()),
        actual_raw_addition_GF=raw_gf,
        wholehalf_prepared_fields='All actual_all_*, actual_site_energy_means, actual_spectral_positivity, actual_real_time_site_GF and actual_ring_spectral_packet_projections below use z=qc*a; raw addition is separately actual_raw_addition_GF. Norm row fields explicitly distinguish both.',
        amplitude_axis_order=['site', 'Ref_hole', 'charged_level'], actual_all_complex_amplitudes=complex_array(z),
        spectral_matrix_axis_order=['charged_level', 'site_x', 'site_y'], actual_all_site_spectral_matrices=complex_array(W),
        actual_all_site_absolute_energy_weights=siteweights.tolist(),
        actual_all_site_normalized_energy_weights=(siteweights/siteweights.sum(axis=1)[:, None]).tolist(),
        actual_site_energy_means=site_means.tolist(), actual_site_norms_and_Knull=norms,
        actual_literal_source_boundary_projection_from_Omega_cut2=source_boundary,
        actual_spectral_positivity=dict(all_level_eigenvalues=Wvalues.tolist(), minimum_eigenvalue=float(Wvalues.min()),
            maximum_Hermiticity_defect=float(hermitian_defects.max()),
            integrated_eigenvalues=eigenvalues.tolist(), integrated_numerical_rank=int(np.sum(eigenvalues > rank_tolerance)),
            integrated_rank_roundoff_tolerance=rank_tolerance,
            integrated_matrix=complex_array(integrated),
            integrated_sum_Gram_defect=float(np.linalg.norm(integrated-direct_integrated)),
            integrated_diagonal_probe_norm_max_defect=float(np.max(abs(integrated.diagonal().real-np.sum(abs(z)**2, axis=(1, 2))))),
            positivity_roundoff_scale=float(4*np.finfo(float).eps*np.max(abs(Wvalues))),
            no_level_dropped=True),
        actual_real_time_site_GF=times, actual_ring_spectral_packet_projections=ring,
        actual_lowest_12_levels_with_all_residues=lowest,
        actual_unchanged_scattering_and_photon_baseline=baseline,
        interpretation='Compare photon energy mean with site/ring GF energy means only as different preparations/sectors; neither mean is a velocity, mass, dispersion, common cone, or free electron pole. Static Ref and fixed Wilson strings break translation: commoncone OPEN.')
    print(json.dumps(dict(stage='actual_GF_observed', ranks=result['actual_positive_ranks'],
        photon_energy=baseline['actual_photon_energy_mean'], site_energy_means=site_means.tolist(),
        raw_addition_site_energy_means=raw_gf['site_raw_addition_energy_means'],
        raw_addition_ring_energy_means=[row['raw_addition_energy_mean'] for row in raw_gf['ring_spectral_packet_projections']],
        ring_energy_means=[row['actual_energy_mean'] for row in ring],
        minimum_W_eigenvalue=float(Wvalues.min()), maximum_W_Hermiticity_defect=float(hermitian_defects.max()),
        integrated_site_rank=result['actual_spectral_positivity']['integrated_numerical_rank'])), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic(); result = run(); result['runtime_seconds'] = time.monotonic()-start
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), input_descriptor_sha256=result['input_descriptor_sha256'],
        own_source_sha256=result['sources_at_IMPORT'][Path(__file__).name]['sha256'], runtime_seconds=result['runtime_seconds'])), flush=True)


if __name__ == '__main__':
    main()
