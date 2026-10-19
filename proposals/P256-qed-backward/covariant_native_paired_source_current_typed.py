#!/usr/bin/env python3
"""Actual NEW Ref0->1 paired source, whole-half consumer and transfer currents.

Same frozen native primitive-E law, no new dynamic flavour.  Charge current
is -i[H_transfer,E_link], the external Peierls force convention, with exact
Gauss continuity.  It is not a fitted electron velocity or common cone.
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
from typing import Any, Callable, cast
ROOT = Path(__file__).resolve().parent
INITIAL_BYTES = {path.name: path.read_bytes() for path in ROOT.glob('*.py')}
PLAN = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT = PLAN.read_bytes()
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.linalg.blas import get_blas_funcs
zgemm = cast(Callable[..., np.ndarray], get_blas_funcs('gemm', dtype=np.complex128))
import covariant_native_electric_heat_comparison as electric
from covariant_native_charged_removal_gf import annihilator, complex_array, decode_complex, ADDITION_BYTES_AT_IMPORT, ADDITION_SHA
from covariant_native_paired_charge_conjugation import physical_conjugation
from covariant_shared_electric_left_svd import lapack_left

def capture_sources() -> dict[str, Any]:
    result = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if isinstance(filename, str):
            path = Path(filename)
            if path.suffix == '.py' and path.resolve().parent == ROOT:
                content = INITIAL_BYTES[path.name]
                result[path.name] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=content.decode())
    return result

def project(V: np.ndarray, wave: np.ndarray) -> np.ndarray:
    return zgemm(alpha=1.0, a=V, b=wave, trans_a=2)

def physical(V: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    return zgemm(alpha=1.0, a=V, b=amplitude)

def current_matrix(V: np.ndarray, energy: np.ndarray, amplitude: np.ndarray, space: dict[str, Any], Cbase: Any, particle: bool) -> dict[str, Any]:
    coefficients = amplitude.reshape(16, -1).T
    wave = physical(V, coefficients)
    Hwave = physical(V, energy[:, None] * coefficients)
    if particle:
        wave = Cbase @ wave
        Hwave = Cbase @ Hwave

    def pull(value: np.ndarray) -> np.ndarray:
        return project(V, Cbase.T @ value if particle else value)
    electric_flux = np.tile(np.array([flux for _, _, flux in space['physical']]), (3, 1))
    densities = np.tile(np.array([electric.native.density(space['states'][state][0], space['states'][state][1], space['charge']) for state, _, _ in space['physical']]), (3, 1))
    currents = []
    continuity = []
    spectral_vertices = []
    times = []
    for edge in range(4):
        Ewave = pull(electric_flux[:, edge, None] * wave)
        EHwave = pull(electric_flux[:, edge, None] * Hwave)
        current = -1j * (energy[:, None] * Ewave - EHwave)
        currents.append(current)
        vertex = np.einsum('xrj,yrj->jxy', amplitude.conj(), current.T.reshape(4, 4, -1))
        spectral_vertices.append(vertex)
    for site in range(4):
        rho_wave = pull(densities[:, site, None] * wave)
        rho_Hwave = pull(densities[:, site, None] * Hwave)
        rho_dot = 1j * (energy[:, None] * rho_wave - rho_Hwave)
        residual = currents[site] - currents[(site - 1) % 4] + rho_dot
        continuity.append(float(np.linalg.norm(residual)))
    spectral = np.array(spectral_vertices)
    for elapsed in (0.0, 0.5, 1.0, 2.0):
        vertex_GF = np.einsum('j,ejxy->exy', np.exp(-1j * elapsed * energy), spectral)
        times.append(dict(time=elapsed, edge_source_current_vertex_GF=complex_array(vertex_GF)))
    return dict(current_definition='j_e=-i[H_transfer,E_e], external Peierls force convention; ALLglobalGauss E, not projected/free electron velocity', amplitude_axis_order=['edge', 'site_probe', 'Ref_component', 'charged_level'], actual_all_current_amplitudes=complex_array(np.array([value.T.reshape(4, 4, -1) for value in currents])), vertex_axis_order=['edge', 'charged_level', 'source_site_x', 'source_site_y'], actual_all_source_current_spectral_matrices=complex_array(spectral), actual_all_time_source_current_vertex_GF=times, actual_Gauss_continuity_absolute_residuals=continuity, time_scope='One-sided GF with CURRENTinserted atsource: <Xi_x,exp(-itH)j_e Xi_y>. NOT mean-energy velocity or fullDirac anticommutator.')

def run() -> dict[str, Any]:
    prior = json.loads(ADDITION_BYTES_AT_IMPORT)
    gamma = decode_complex(prior['selected_gamma'])
    embedding = decode_complex(prior['selected_embedding'])
    Ebar = embedding.conj() @ np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([-0.09, 0.0, 0.09])
    sigma = 0.03
    radius = 0.12
    k = np.exp(-y * y / (4 * sigma * sigma) - sigma * sigma * y * y / (radius * radius * (radius * radius - y * y)))
    k /= np.linalg.norm(k)
    D = np.zeros((2, 3))
    for j in range(2):
        D[j, j] = math.sqrt(k[j + 1] / k[j]) / 0.09
        D[j, j + 1] = -math.sqrt(k[j] / k[j + 1]) / 0.09
    h, U = eigh(2 * sigma * sigma * D.T @ D)
    heat = U * np.exp(-h / 2) @ U.T
    neutral = electric.native.space(0, 2)
    removal = electric.native.space(-1, 2)
    particle_space = electric.native.space(1, 2)
    spectra = {}
    diagnostics = {}
    svd_rows = {}
    for name, space in (('neutral', neutral), ('removal', removal)):
        print(json.dumps(dict(stage='actual_fullquantum_paired_factor', sector=name, dimension=3 * len(space['physical']))), flush=True)
        factor = electric.full_factor(gamma, y, heat, space, 32, 1.0)
        diagnostics[name] = {key: value for key, value in factor.items() if key not in ('F', 'Haar')}
        del factor['Haar']
        matrix = np.asfortranarray(factor.pop('F'))
        del factor
        q, svd_rows[name] = lapack_left(matrix)
        spectra[name] = dict(V=matrix, q=q)
    n, c = (spectra['neutral'], spectra['removal'])
    V = c['V']
    lambdaN = float(n['q'][0] ** 2)
    q = c['q'] / math.sqrt(lambdaN)
    energy = -2 * np.log(q)
    Omega = n['V'][:, 0].copy()
    mean, _ = electric.native.square_jets(gamma, 0.09, 0.0)
    values, U = eigh(mean[0] @ mean[0])
    negative = U[:, values > 1]
    if negative.shape[1] != 4:
        raise ArithmeticError('Actual negative band is not complete rank4')

    def coefficient(packet: np.ndarray) -> np.ndarray:
        value = negative @ negative.conj().T @ np.kron(packet[:, None], np.eye(2))
        return value / np.linalg.norm(value, 'fro')
    packet = np.array([0.1, 0.8, 0.4, 0.1])
    packet /= np.linalg.norm(packet)
    receiver = np.array([0.5, 0.4, 0.1, 0.6])
    receiver /= np.linalg.norm(receiver)
    Sh = annihilator(neutral, removal, coefficient(packet), Ebar)
    Shout = annihilator(neutral, removal, coefficient(receiver), Ebar)
    source = electric.observe(n, c, neutral, removal, Sh, Shout)
    del spectra['neutral']
    del n
    Cbase = physical_conjugation(removal, particle_space, False)
    raw_h = []
    raw_p = []
    for site_packet in np.eye(4):
        raw_h.append((annihilator(neutral, removal, coefficient(site_packet), Ebar) @ Omega).reshape(4, -1))
        plus = electric.native.source(neutral, particle_space, electric.native.source_coefficients(gamma, site_packet), embedding, 3)
        raw_p.append((plus @ Omega).reshape(4, -1))
    Ah = project(V, np.array(raw_h).reshape(16, -1).T).T.reshape(4, 4, -1)
    Ap = project(V, Cbase.T @ np.array(raw_p).reshape(16, -1).T).T.reshape(4, 4, -1)
    Zh = q[None, None, :] * Ah
    Zp = q[None, None, :] * Ap
    removal_current = current_matrix(V, energy, Zh, removal, Cbase, False)
    particle_current = current_matrix(V, energy, Zp, particle_space, Cbase, True)
    Wh = np.einsum('xrj,yrj->jxy', Zh.conj(), Zh)
    Wp = np.einsum('xrj,yrj->jxy', Zp.conj(), Zp)
    Jh = decode_complex(removal_current['actual_all_source_current_spectral_matrices'])
    Jp = decode_complex(particle_current['actual_all_source_current_spectral_matrices'])
    descriptor = dict(plan_id=PLAN.stem, plan_sha256_at_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest(), law='SAME frozen3a native reflectedFULL SF with primitiveelectricUE1, Ecut2, Phi3 ONEcoordinate, all70/56active×aux6, ONEGauss', neutral='NEW SAME4RefCAR modes Ref0/background0, neutralKN equalsoldRef4/background4 byzeroRefrelativecharge', removal='N3 +Ref1, SAME4Refmodes staticCAR counterparticle positivecharge at0; NOT newdynamicflavour', particle='N5 +Ref3 frozen source ABI; eigenbasis generated by exactALLcolumn C kernel identity, NOT copieddata/mocked GF', source='r_dag W_inverse a_negative, Ebar=E*sigmax, canonicalCoherentRefsignbelow+5', source_packet=packet.tolist(), receiver_packet=receiver.tolist(), Pe=[0.25, 2.0, 4.0], time=[0.0, 0.5, 1.0, 2.0], auxiliary_modes=4, auxiliary_particles=2, auxiliary_states=6, active_modes=8, active_kappa=0.09, auxiliary_kappa=0.04, Phi_nodes=y.tolist(), selected_Clifford_dimension=2, old_addition_producer_capsule_sha256=ADDITION_SHA, common_neutral_lambdaN=lambdaN, no_fit=True, no_fullDiracCAR_commoncone_originalrealEM_fullSU4Z4_claim=True, current='FULLtransferH=-log(K/lambdaN); j=-i[H,E_phys] from externalPeierlsforce, divergence(j)+rho_dot=0')
    return dict(input_descriptor=descriptor, input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(), sources_at_IMPORT=capture_sources(), runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()), actual_factor_diagnostics=diagnostics, actual_native_LAPACK_ALLsingular_rows=svd_rows, actual_common_neutral_lambdaN=lambdaN, actual_all_charged_base_energies=energy.tolist(), actual_C_conjugate_photon_source_consumer=source, actual_removal_current_matrix=removal_current, actual_particle_current_matrix=particle_current, actual_all_removal_GF_complex_amplitudes=complex_array(Zh), actual_all_particle_GF_complex_amplitudes=complex_array(Zp), actual_all_removal_GF_spectral_site_matrices=complex_array(Wh), actual_all_particle_GF_spectral_site_matrices=complex_array(Wp), actual_C_GF_matrix_max_absolute_defect=float(np.max(abs(Wh - Wp))), actual_C_odd_current_matrix_max_absolute_defect=float(np.max(abs(Jh + Jp))), interpretation='ActualpairedCARprotocol/sourceconsumer and generated-currentmatrices, NOTGFfirstmoment electronmass/velocity/dispersion. FinitePhi3/Cliff2 conditionalmean; fundamentalquantum law/electricUE parameters remaininputs, massless/commoncone/fullQ stillOPEN.')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result = run()
    result['runtime_seconds'] = time.monotonic() - start
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(runtime_seconds=result['runtime_seconds'], source={key: result['actual_C_conjugate_photon_source_consumer'][key] for key in ('actual_M0', 'actual_E1', 'actual_R1', 'actual_energy_mean')}, current_C_defect=result['actual_C_odd_current_matrix_max_absolute_defect'], GF_C_defect=result['actual_C_GF_matrix_max_absolute_defect'])), flush=True)
if __name__ == '__main__':
    main()
