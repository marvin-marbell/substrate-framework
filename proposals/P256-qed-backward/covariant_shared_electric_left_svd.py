#!/usr/bin/env python3
"""Exact all-column shared UE1 successor: disk factor, overwrite LEFT-only SVD.

Preserves frozen predecessors. No Gram squaring, singular clipping, pruning,
source alteration, dispersion/pole/massless/common-cone promotion.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import mmap
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import scipy.linalg.cython_lapack as cython_lapack
from scipy.linalg import eigh, svd
from scipy.linalg.blas import get_blas_funcs
import covariant_shared_primitive_electric as core

zgemv = get_blas_funcs('gemv', dtype=np.complex128)

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT: dict[str, bytes] = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if isinstance(filename, str):
        path = Path(filename)
        if path.suffix == '.py' and path.resolve().parent == ROOT:
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()
ABI_EXPECTED = 'void (char *, char *, int *, int *, __pyx_t_double_complex *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, __pyx_t_double_complex *, int *, __pyx_t_double_complex *, int *, __pyx_t_double_complex *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *)'


def lapack_left(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if matrix.dtype != np.complex128 or not matrix.flags.f_contiguous or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Requires square complex128 Fortran factor, without conversion/copy')
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    capsule = cython_lapack.__pyx_capi__['zgesvd']
    name = get_name(capsule)
    if not isinstance(name, bytes) or name.decode() != ABI_EXPECTED:
        raise RuntimeError(f'Unrecognized actual capsule ABI: {name!r}')
    int_pointer = ctypes.POINTER(ctypes.c_int)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char),
        int_pointer, int_pointer, ctypes.c_void_p, int_pointer, double_pointer,
        ctypes.c_void_p, int_pointer, ctypes.c_void_p, int_pointer, ctypes.c_void_p,
        int_pointer, double_pointer, int_pointer)
    function = function_type(get_pointer(capsule, name))
    n = matrix.shape[0]
    size, lda, one = ctypes.c_int(n), ctypes.c_int(n), ctypes.c_int(1)
    jobu, jobvt = ctypes.c_char(b'O'), ctypes.c_char(b'N')
    singulars = np.empty(n, dtype=np.float64)
    rwork = np.empty(5 * n, dtype=np.float64)
    dummy = np.empty(1, dtype=np.complex128)
    work = np.empty(1, dtype=np.complex128)
    lwork, info = ctypes.c_int(-1), ctypes.c_int(0)
    def invoke() -> None:
        function(ctypes.byref(jobu), ctypes.byref(jobvt), ctypes.byref(size), ctypes.byref(size),
            matrix.ctypes.data, ctypes.byref(lda), singulars.ctypes.data_as(double_pointer),
            dummy.ctypes.data, ctypes.byref(one), dummy.ctypes.data, ctypes.byref(one),
            work.ctypes.data, ctypes.byref(lwork), rwork.ctypes.data_as(double_pointer), ctypes.byref(info))
    invoke()
    if info.value != 0:
        raise ArithmeticError(f'zgesvd workspace query INFO={info.value}')
    lwork.value = int(float(np.real(work[0])))
    work = np.empty(lwork.value, dtype=np.complex128)
    start = time.monotonic()
    invoke()
    if info.value != 0 or singulars[-1] <= 0:
        raise ArithmeticError(f'Actual zgesvd INFO={info.value}, min singular={singulars[-1]}; no clipping')
    return singulars, dict(actual_capsule_ABI=name.decode(), JOBU='O', JOBVT='N', ALL_singulars=n,
        matrix_overwritten_with_LEFT_U=True, no_right_vectors_allocated=True, INFO=info.value,
        workspace_complex_entries=lwork.value, runtime_seconds=time.monotonic() - start)


def project(vectors: np.ndarray, wave: np.ndarray) -> np.ndarray:
    return zgemv(alpha=1., a=vectors, x=wave, trans=2)


def physical_wave(vectors: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return zgemv(alpha=1., a=vectors, x=coefficients, trans=0)


def evict(matrix: np.memmap) -> None:
    matrix.flush()
    mapping = getattr(matrix, '_mmap', None)
    if mapping is not None:
        mapping.madvise(mmap.MADV_DONTNEED)


def frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    gamma4, metric = core.phi_frame(core.Parameters(kappa_power=400))
    values, U = eigh(1j * gamma4[0] @ gamma4[1] @ gamma4[2])
    embedding = U[:, values > 0]
    _, spinframe = eigh(embedding.conj().T @ gamma4[0] @ embedding)
    embedding = embedding @ spinframe[:, ::-1]
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
    return gamma, embedding, nodes, heat, Hphi, metric


def streamed_factor(gamma: np.ndarray, nodes: np.ndarray, heat: np.ndarray,
                    physical: dict[str, Any], grid: int, path: Path) -> tuple[np.memmap, np.ndarray, dict[str, Any]]:
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
    electric = np.array([sum(e * e for e in E) for _, _, _, E in physical['physical']])
    electric_half = np.exp(-electric / 2)
    nr = len(states)
    factor = np.memmap(path, mode='w+', dtype=np.complex128, shape=(3 * nr, 3 * nr), order='F')
    lower = []
    angles = 2 * math.pi * np.arange(grid) / grid
    for p, normal in enumerate(nodes):
        actual = gamma.copy()
        actual[1] *= 1 + float(normal)
        half, pref = core.matched_tree_half(actual, 0., 0.)
        mean = core.canonical(half, pref, occupied)
        samples = np.empty((grid, grid, 6, 6), complex)
        point_min = math.inf
        for i, thetaL in enumerate(angles):
            for j, thetaR in enumerate(angles):
                phaseB = complex((np.exp(1j * thetaL) + 1) / 2)
                phaseF = complex((np.exp(1j * thetaL) + np.exp(1j * (thetaL - thetaR))) / 2)
                half, pref = core.split_route_half(actual, .04, phaseB, phaseF)
                samples[i, j] = core.canonical(half, pref, auxiliary)
                point_min = min(point_min, float(eigh(samples[i, j], eigvals_only=True)[0]))
        lower.append(float(eigh(mean, eigvals_only=True)[0]) * point_min)
        coefficients = np.fft.fft2(samples, axes=(0, 1)) / grid ** 2
        for start in range(0, nr, 128):
            stop = min(nr, start + 128)
            deltaL = fluxL[:, None] - fluxL[None, start:stop] - n3[:, None] + n3[None, start:stop]
            deltaR = fluxR[:, None] - fluxR[None, start:stop]
            block = mean[lr[:, None], lr[None, start:stop]] * coefficients[
                deltaL % grid, deltaR % grid, ar[:, None], ar[None, start:stop]]
            block *= electric_half[None, start:stop]
            for out in range(3):
                factor[out * nr:(out + 1) * nr, p * nr + start:p * nr + stop] = heat[out, p] * block
        factor.flush()
    if min(lower) <= 0:
        raise ArithmeticError('Nonpositive point canonical factor')
    middle = np.array([j for j, (_, _, _, E) in enumerate(physical['physical']) if all(e == 0 for e in E)])
    columns = np.concatenate([middle + p * nr for p in range(3)])
    haar = np.asfortranarray(factor[:, columns])
    return factor, haar, dict(actual_factor_dimension=3 * nr, factor_disk_bytes=factor.nbytes,
        actual_complete_E0_columns=len(columns), actual_pointcanonical_positive_lower_bounds=lower,
        actual_electric_half_min=float(electric_half.min()), actual_ALL7edge_E2_max=int(electric.max()),
        disk_path=str(path), construction='128-column full-row streaming, no full SF duplicate')


def validate_native_zero(gamma: np.ndarray, nodes: np.ndarray, heat: np.ndarray) -> list[dict[str, Any]]:
    proofs = []
    for charge in (0, 1):
        physical = core.space(charge, 0)
        actual, _, _ = core.full_factor(gamma, nodes, heat, physical, 32)
        original = actual.copy(order='F')
        left, reference, _ = svd(original, full_matrices=False)
        singulars, abi = lapack_left(actual)
        probe = np.arange(1, len(singulars) + 1, dtype=float).astype(complex)
        probe /= np.linalg.norm(probe)
        direct = original @ (original.conj().T @ probe)
        represented = physical_wave(actual, singulars ** 2 * project(actual, probe))
        relative_q = float(np.linalg.norm(singulars - reference) / np.linalg.norm(reference))
        relative_operator = float(np.linalg.norm(direct - represented) / np.linalg.norm(direct))
        if max(relative_q, relative_operator) > 1e-11:
            raise ArithmeticError('Actual native Ecut0 left overwrite SVD validation failed')
        proofs.append(dict(charge=charge, actual_native_Ecut0_factor_dimension=len(singulars),
            actual_vs_full_gesdd_singular_relative_defect=relative_q,
            actual_positive_operator_action_relative_defect=relative_operator, ABI=abi))
    return proofs


def thin_spectrum(factor: np.ndarray) -> dict[str, Any]:
    vectors, singulars, _ = svd(factor, full_matrices=False, lapack_driver='gesdd')
    return dict(V=vectors, q=singulars)


def observe(n: dict[str, Any], c: dict[str, Any], neutral: dict[str, Any], charged: dict[str, Any],
            S: list[Any], Sout: list[Any], Hphi: np.ndarray) -> dict[str, Any]:
    Vn, Vc = n['V'], c['V']
    lambdaN = float(n['q'][0] ** 2)
    qn, qc = n['q'] / math.sqrt(lambdaN), c['q'] / math.sqrt(lambdaN)
    En, Ec = -2 * np.log(qn), -2 * np.log(qc)
    Omega = np.array(Vn[:, 0])
    BL, BR, CL, CR = core.curl(neutral, 0), core.curl(neutral, 1), core.curl(charged, 0), core.curl(charged, 1)
    rawL, rawR = core.apply_field(BL, Omega), core.apply_field(BR, Omega)
    bL, bR = project(Vn, rawL), project(Vn, rawR)
    target = core.apply_source(Sout, Omega)
    cases = []
    for q in (0., math.pi):
        mode = np.exp(1j * q)
        b = (bL + mode * bR) / math.sqrt(2)
        rawph = (rawL + mode * rawR) / math.sqrt(2)
        ph = qn * b
        Zph = float(np.vdot(ph, ph).real)
        if Zph <= 0:
            raise ArithmeticError('Zero actual face photon')
        ph /= math.sqrt(Zph)
        incoming = physical_wave(Vn, ph)
        photon_null = rawph - physical_wave(Vn, b)
        raw = core.apply_source(S, incoming)
        projected = np.array([project(Vc, w) for w in raw])
        Xi = qc[None, :] * projected
        M0 = float(np.vdot(Xi, Xi).real)
        if M0 <= 0:
            raise ArithmeticError('Zero actual literal source')
        E1 = float(np.sum(abs(Xi) ** 2 * Ec[None, :]))
        R1 = float(np.sum(abs(Xi) ** 2 * qc[None, :] ** 2))
        source_null = np.array([w - physical_wave(Vc, z) for w, z in zip(raw, projected)])
        outgoing = np.array([(core.apply_field(CL, w) + mode * core.apply_field(CR, w)) / math.sqrt(2) for w in target])
        receiver = np.array([project(Vc, w) for w in outgoing])
        consumer = []
        for elapsed in (0., .5, 1., 2.):
            evolved = Xi * np.exp(-1j * elapsed * Ec)[None, :] / math.sqrt(M0)
            amplitude = complex(np.vdot(receiver, evolved))
            consumer.append(dict(time=elapsed, coherent_Ref4_receiver_real=amplitude.real,
                coherent_Ref4_receiver_imag=amplitude.imag, coherent_Ref4_receiver_probability=abs(amplitude) ** 2,
                actual_evolved_source_norm_squared=float(np.vdot(evolved, evolved).real)))
        sourcewave = np.array([physical_wave(Vc, z) for z in Xi])
        cases.append(dict(actual_face_packet_q=q, actual_photon_Zph=Zph,
            actual_photon_energy_mean=float(np.sum(abs(ph) ** 2 * En)),
            actual_M0=M0, actual_E1=E1, actual_R1=R1, actual_energy_mean=E1 / M0, actual_transfer_Rayleigh_mean=R1 / M0,
            actual_raw_photon_Knull_norm_squared=float(np.vdot(photon_null, photon_null).real),
            actual_raw_source_Knull_norm_squared=float(np.vdot(source_null, source_null).real),
            actual_charged_source_band_rows=[dict(absolute_threshold=t, source_fraction=float(np.sum(abs(Xi[:, Ec <= t]) ** 2) / M0)) for t in (.25, 2., 4.)],
            actual_consumer=consumer, actual_quantum_photon_loads=core.loads(incoming, neutral, Hphi),
            actual_quantum_source_loads=core.loads(sourcewave, charged, Hphi)))
    spectral_rows = []
    for j, energy in enumerate(En):
        cross = complex(np.conjugate(bL[j]) * bR[j])
        matrix_real = [[float(abs(bL[j]) ** 2), cross.real], [cross.real, float(abs(bR[j]) ** 2)]]
        matrix_imag = [[0., cross.imag], [-cross.imag, 0.]]
        weights = [float(abs(qn[j] * (bL[j] + np.exp(1j * q) * bR[j]) / math.sqrt(2)) ** 2) for q in (0., math.pi)]
        spectral_rows.append(dict(neutral_singular_index=j, actual_neutral_energy=float(energy),
            actual_bL_real=float(np.real(bL[j])), actual_bL_imag=float(np.imag(bL[j])),
            actual_bR_real=float(np.real(bR[j])), actual_bR_imag=float(np.imag(bR[j])),
            actual_W_face_order=['L','R'], actual_W_real=matrix_real, actual_W_imag=matrix_imag,
            actual_prepared_q0_absolute_weight=weights[0], actual_prepared_qpi_absolute_weight=weights[1],
            actual_prepared_q0_normalized_weight=weights[0] / cases[0]['actual_photon_Zph'],
            actual_prepared_qpi_normalized_weight=weights[1] / cases[1]['actual_photon_Zph']))
    response = []
    for elapsed in (0., .5, 1., 2.):
        cross = complex(np.vdot(bL, bR * np.exp(-1j * elapsed * En)))
        response.append(dict(time=elapsed, actual_positive_sector_retarded_cross_face_response=2 * cross.imag,
            actual_positive_sector_cross_spectral_real=cross.real, actual_positive_sector_cross_spectral_imag=cross.imag))
    return dict(actual_common_neutral_lambdaN=lambdaN, actual_positive_ranks=[len(qn),len(qc)],
        actual_ALL_neutral_singular_values=n['q'].tolist(), actual_ALL_charged_singular_values=c['q'].tolist(),
        actual_normalized_min_positive_singular_values=[float(qn[-1]), float(qc[-1])],
        actual_neutral_ground_loads=core.loads(Omega, neutral, Hphi), actual_face_packet_cases=cases,
        actual_every_neutral_energy_full_2x2_face_spectral_matrix_rows=spectral_rows,
        actual_positive_sector_retarded_cross_face_rows=response)


def run(cutoff: int, scratch: Path) -> dict[str, Any]:
    gamma, embedding, nodes, heat, Hphi, metric = frame()
    validation = validate_native_zero(gamma, nodes, heat)
    identity = core.validate_light_identity(gamma, nodes)
    print(json.dumps(dict(stage='validated_actual_native_Ecut0_overwrite_SVD', rows=validation)), flush=True)
    neutral, charged = core.space(0, cutoff), core.space(1, cutoff)
    spectra, haars, diagnostics, mappings = [], [], {}, []
    for label, physical in (('neutral', neutral), ('charged', charged)):
        path = scratch / f'{label}_cut{cutoff}_leftU.complex128'
        print(json.dumps(dict(stage='streaming_fullcanonical_factor', sector=label, dimension=3 * len(physical['physical']))), flush=True)
        factor, haar, diag = streamed_factor(gamma, nodes, heat, physical, 32, path)
        haars.append(thin_spectrum(haar))
        del haar
        print(json.dumps(dict(stage='native_JOBUO_JOBVTN_ALLsingulars', sector=label)), flush=True)
        singulars, abi = lapack_left(factor)
        spectra.append(dict(V=factor, q=singulars))
        mappings.append(factor)
        diag['actual_overwrite_SVD'] = abi
        diagnostics[label] = diag
        evict(factor)
    packet = np.array([.1,.8,.4,.1]); packet /= np.linalg.norm(packet)
    receiver = np.array([.5,.4,.1,.6]); receiver /= np.linalg.norm(receiver)
    S = core.source(neutral, charged, core.native.source_coefficients(gamma, packet), embedding)
    Sout = core.source(neutral, charged, core.native.source_coefficients(gamma, receiver), embedding)
    full = observe(n=spectra[0], c=spectra[1], neutral=neutral, charged=charged, S=S, Sout=Sout, Hphi=Hphi)
    haar = observe(n=haars[0], c=haars[1], neutral=neutral, charged=charged, S=S, Sout=Sout, Hphi=Hphi)
    descriptor = dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',
        plan_sha256_at_IMPORT=hashlib.sha256(core.PLAN_BYTES_AT_IMPORT).hexdigest(),
        primitive_electric_input=1, NEW_declared_electric_cutoff_ALL7edges=cutoff, actual_field_grid_each_cycle=32,
        active_modes=8, active_particle_numbers=[4,5], active_CAR_states=[70,56], auxiliary_modes=4, auxiliary_particle_number=2, auxiliary_CAR_states=6,
        original_Ref=4, prepared_Ref=3, Phi_nodes=nodes.tolist(), Phi_sigma=.03, Phi_radius=.12,
        Phi3_is_three_levels_ONE_coordinate=True, active_kappa=.09, auxiliary_kappa=.04,
        F='PhiheatHalf SF6b0243_fullcanonical exp[-sum_ALL7edges E²/2], ONE globalGauss after full light and allaux matrices',
        E='(kL+rho0,kL+rho0+rho1,kL+rho0+rho1+rho2,kL+kR,-kR,-kR,-kR)',
        exact_memory_repair='Disk Fortran factor; zgesvd JOBUO JOBVTN overwrites LEFT U, ALL singulars, no Vh, no Gram/clipping/pruning; zgemv trans2 Udag without conjugate matrix copy',
        source_packet=packet.tolist(), receiver_packet=receiver.tolist(), Pe=[.25,2.,4.], times=[0.,.5,1.,2.],
        same_literal_source_paths=[[0]*7,[-1,0,0,0,0,0,0],[-1,-1,0,0,0,0,0],[0,0,0,1,0,0,0]],
        quantum_postulates_and_parameters='Declared inputs, not permission gates',
        operator_contract='SAME neutral lambdaN for KN/KC; actualOmega; Xi_ph=sqrt(KN/lambdaN) Bq Omega/sqrtZph; Xi_ch=sqrt(KC/lambdaN) S_literal Xi_ph; M0=norm²Xi; E1=<Xi,-log(KC/lambdaN)Xi>/tau1; R1=<Xi,KC/lambdaN Xi>; ONE charged whole halfstep pervector; coherent Ref4 before abs²',
        Haar_reference='Complete E=0 input columns of SAME F, exact UE→infinite endpoint',
        spectral_matrix='EVERY neutral energy, W_rs=conj(b_r)b_s; preparedq0/pi weights; qfacepacket NOT translationdispersion/pole',
        scope='Finite Gauss/electric/Pauli/Phi3; no bare magnetic action, inducedE, speed/mass/quarterfit, massless/commoncone/macro/fullCartesianPhi/SU4 promotion')
    result = dict(input_descriptor=descriptor, input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        plan_bytes_at_import=core.PLAN_BYTES_AT_IMPORT.decode(), physical_mean_descriptor=metric,
        runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=core.scipy.__version__, platform=core.platform.platform()),
        actual_native_Ecut0_overwrite_SVD_validation=validation, actual_nonzero_pointcanonical_identity_rows=identity,
        actual_fullcolumn_factor_diagnostics=diagnostics, actual_UE1_fullpositive=full, actual_same_factor_Haar_E0_endpoint=haar)
    for mapping in mappings:
        evict(mapping)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--electric-cutoff',type=int,required=True,choices=(1,2))
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    result: dict[str, Any]
    if args.validate_only:
        gamma, _, nodes, heat, _, _ = frame()
        result = dict(actual_native_Ecut0_overwrite_SVD_validation=validate_native_zero(gamma,nodes,heat),
            source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()})
    else:
        args.scratch.mkdir(parents=True,exist_ok=True)
        result = run(args.electric_cutoff,args.scratch)
    result['runtime_seconds'] = time.monotonic() - start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(stage='natural_result_written', output=str(args.output), runtime_seconds=result['runtime_seconds'])),flush=True)


if __name__ == '__main__':
    main()
