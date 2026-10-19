"""Actual N4 compensated electron/Refhole source in a partial Gaussian model.

The bare filled active1/aux63 sea and its 128-dimensional empty basis stay
fixed. All four Ref holes are retained with the unrescaled creator factor 1/2.
This is a positive-electron, zero/one-boson Hermitian star law, not full CAR
interacting dynamics: overlapping sea PH and added-electron Pauli corrections,
nonlinear Peierls orders, dynamical Phi/SU4 and the original canonical Y are
not computed. The frozen photon supplier already contains its positive RPA
PH/background response. Quantum law and primitive parameters are inputs.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import hashlib
import importlib.machinery
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
OWN = Path(__file__).read_bytes()
SOURCES: dict[str, bytes] = {str(Path(__file__).resolve()): OWN}
_GET_CODE = importlib.machinery.SourceFileLoader.get_code
CHARGED_SHA = '3a9ffc46f646f52fe9dcdb6a29793c69ea7f9ac93007682174cd27d2bde2fe08'


def capture_code(self: Any, fullname: str) -> Any:
    path = Path(self.path).resolve()
    if path.is_relative_to(ROOT) or (path.parent == Path('/tmp') and path.name.startswith('induced-')):
        data = path.read_bytes()
        if str(path) in SOURCES and SOURCES[str(path)] != data:
            raise RuntimeError('Scientific import epoch changed: ' + str(path))
        SOURCES[str(path)] = data
    return _GET_CODE(self, fullname)


importlib.machinery.SourceFileLoader.get_code = capture_code
import numpy as np
import induced_native_charged_green as charged
if hashlib.sha256(SOURCES[str(ROOT / 'induced_native_charged_green.py')]).hexdigest() != CHARGED_SHA:
    raise RuntimeError('Frozen charged primitive adapter changed')


def literal_compensated_coulomb(native: dict[str, Any], projectors: list[np.ndarray],
                                reference_projector: np.ndarray, ref: int) -> dict[str, Any]:
    """Full species-loaded Wick/CAR rho Lplus rho, with actual Ref occupancy.

    The Ref is a four-orbital local CAR factor, not a compensating c-number
    retag of a charged result. Its mean AND its number covariance enter.
    """
    V = native['V']
    background = np.full(V, 128.)
    background[ref] += 4
    mean = -background
    covariance = np.zeros((V, V))
    for profile, P in zip(native['Pprofiles'], projectors, strict=True):
        blocks = P.reshape(V, 4, V, 4).transpose(0, 2, 1, 3)
        density = np.trace(blocks[np.arange(V), np.arange(V)], axis1=1, axis2=2).real
        mean += profile['load'] * density
        covariance += profile['load'] * (np.diag(density) - np.sum(abs(blocks)**2, axis=(2, 3)))
    mean[ref] += np.trace(reference_projector).real
    covariance[ref, ref] += np.trace(reference_projector - reference_projector @ reference_projector).real
    u = native['UE'] / native['a']
    return {'energy': float(u * (mean @ native['Lplus'] @ mean + np.sum(native['Lplus'] * covariance))),
            'Hartree': float(u * mean @ native['Lplus'] @ mean), 'mean': mean,
            'covariance': covariance, 'background': background, 'total_charge': float(np.sum(mean))}


def projected_operators(native: dict[str, Any], profile: dict[str, Any],
                        Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All physical JA=a*dTheta and local electron densities in bare empty Z."""
    V = native['V']; n = Z.shape[1]
    JA = np.zeros((3*V, n, n), complex)
    rho = np.empty((V, n, n), complex)
    for x in range(V):
        ZX = Z[4*x:4*x+4]
        rho[x] = ZX.conj().T @ ZX
    for path in profile['paths']:
        x, y = path['source'], path['target']
        forward = Z[4*x:4*x+4].conj().T @ path['hop'] @ Z[4*y:4*y+4]
        contribution = 1j * native['a'] * (forward - forward.conj().T)
        for edge in path['edges']:
            JA[edge] += contribution
    return JA, rho


def stream_resolvents(supplier: Any, Z: np.ndarray, static: np.ndarray,
                       JA: np.ndarray, rho: np.ndarray, ref: int,
                       probes: tuple[complex, ...]) -> dict[str, Any]:
    """Consume actual modes, never inferred poles or inverse-G fits.

    Each virtual electron has the SAME full projected neutral static H as
    the zero-boson sector. Hbath_l=Hstatic+Omega_l, diagonalized once. B_l
    is the coherent JA*A+(rho-Ref)*phi contraction BEFORE forming a Gram.
    Only one qgroup's B_l is resident. Four spectator Refhole channels share
    this law; the RPA bath's PH particles carry zero total EM charge.
    """
    native = supplier.native; n = Z.shape[1]
    energies, U = np.linalg.eigh(static)
    sigma = [np.zeros((n, n), complex) for _ in probes]
    metric = [np.zeros((n, n), complex) for _ in probes]
    wrong_sigma = np.zeros((n, n), complex)
    gram = np.zeros((n, n), complex)
    mixed = np.zeros((n, n), complex)
    holonomy = np.zeros((n, n), complex)
    rows = []; modes = 0; framesA = 0; framesR = 0
    identity = np.eye(n)
    for site in native['sites']:
        q = tuple(int(v) for v in site)
        if q > tuple((-v) % native['N'] for v in q):
            continue
        block = supplier.block(q)
        edge = block.data['physical_edgeframe']; density = block.data['physical_densityframe']
        Oa = (edge.T @ JA.reshape(3*native['V'], -1)).reshape(block.M, n, n)
        Op = (density.T @ rho.reshape(native['V'], -1)).reshape(block.R, n, n)
        Op -= density[ref, :, None, None] * identity[None, :, :]
        amplitudes = block.amplitudes
        BA = np.einsum('mij,jk,ml->ikl', Oa, U, amplitudes[:block.M], optimize=True)
        BP = np.einsum('mij,jk,ml->ikl', Op, U, amplitudes[block.M:], optimize=True)
        B = (BA + BP).reshape(n, -1)
        d = (energies[:, None] + block.omega[None, :]).reshape(-1)
        group_gram = B @ B.conj().T
        gram += group_gram
        mixed += group_gram - BA.reshape(n, -1) @ BA.reshape(n, -1).conj().T - BP.reshape(n, -1) @ BP.reshape(n, -1).conj().T
        if q == (0, 0, 0):
            holonomy += group_gram
        for j, z in enumerate(probes):
            sigma[j] += (B / (z-d)[None, :]) @ B.conj().T
            metric[j] += (B / (abs(z-d)**2)[None, :]) @ B.conj().T
        # Deliberately omit ONLY the Ref scalar-potential vertex. Keep the
        # compensated static C, sea and all physical modes identical.
        phi_ref = density[ref] @ amplitudes[block.M:]
        Bwrong = B + (U[:, :, None] * phi_ref[None, None, :]).reshape(n, -1)
        wrong_sigma += (Bwrong / (probes[1]-d)[None, :]) @ Bwrong.conj().T
        modes += len(block.omega); framesA += block.M; framesR += block.R
        rows.append({'q': q, 'A_frames': block.M, 'phi_frames': block.R,
                     'positive_bosonic_modes': len(block.omega),
                     'omega': block.omega, 'virtual_bath_energy_range': [float(np.min(d)), float(np.max(d))],
                     'coherent_coupling_Gram_trace': float(np.trace(group_gram).real),
                     'ref_scalar_amplitude_squared': float(np.vdot(phi_ref, phi_ref).real)})
        print('Actual neutral Refhole qgroup', q, 'positive modes', len(block.omega), flush=True)
    if (framesA, framesR, len(rows), modes) != (129, 63, 36, 3865):
        raise ArithmeticError('Incomplete physical neutral star consumption')
    results = []
    for z, S, M in zip(probes, sigma, metric, strict=True):
        G = np.linalg.solve(z*identity-static-S, identity)
        spectral = -(G-G.conj().T)/(2j)
        control = z.imag * G.conj().T @ (identity+M) @ G
        results.append({'z': z, 'Sigma': S, 'G': G, 'minus_Im_G': spectral,
                        'minus_Im_G_eigenvalues': np.linalg.eigvalsh(spectral),
                        'independent_bath_spectral_metric': M,
                        'positive_star_identity_max_error': float(np.max(abs(spectral-control))),
                        'inverse_equation_max_error': float(np.max(abs((z*identity-static-S)@G-identity)))})
    wrongG = np.linalg.solve(probes[1]*identity-static-wrong_sigma, identity)
    return {'virtual_static_energies': energies, 'virtual_static_vectors': U,
            'star_auxiliary_dimension_per_Refhole': n*(1+modes),
            'zero_boson_dimension': n, 'one_boson_dimension': n*modes,
            'physical_frame_counts': [framesA, framesR], 'positive_bosonic_modes': modes,
            'qgroups': rows, 'retarded_probes': results,
            'coherent_bath_Gram': gram, 'coherent_A_phi_interference_Gram': mixed,
            'q0_holonomy_Gram': holonomy,
            'missing_Ref_scalar_counterfactual': {'z': probes[1], 'Sigma': wrong_sigma, 'G': wrongG,
                                                 'full_G_operator_defect': float(np.linalg.norm(wrongG-results[1]['G'], 2))}}


def run() -> dict[str, Any]:
    started = time.monotonic()
    photon_raw = charged.RECEIPT.read_bytes()
    if hashlib.sha256(photon_raw).hexdigest() != charged.RECEIPT_SHA:
        raise RuntimeError('Frozen actual photon receipt changed')
    supplier = charged.load_supplier(); native = supplier.native; s = supplier.s
    if (native['N'], native['V'], [p['load'] for p in native['Pprofiles']]) != (4, 64, [1, 63]):
        raise ArithmeticError('Native N4 active1/aux63 required')
    mean_path = Path(s.primitives.MEAN); mean_raw = mean_path.read_bytes()
    profile = native['Pprofiles'][0]; V = native['V']; zero = np.zeros(3*V)
    ref = int(np.flatnonzero(np.all(native['sites'] == [0, 0, 0], axis=1))[0])
    descriptor = {'plan_id': 'issue-222-induced-photon-delta13-continuation-20261008',
                  'execution_revision': 'd86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7',
                  'charged_source_sha256': CHARGED_SHA, 'photon_source_sha256': charged.PHOTON_SHA,
                  'photon_receipt_sha256': charged.RECEIPT_SHA, 'photon_epoch': supplier.full.ASSIGNED,
                  'N': 4, 'AP_seam': 'AP222', 'g0': native['g0'], 'gs': native['gs'],
                  'a': native['a'], 'UE': native['UE'], 'u': native['UE']/native['a'],
                  'Pprofile_loads': [p['load'] for p in native['Pprofiles']], 'active_model': asdict(profile['model']),
                  'source': 'E0=(1/2) sum_alpha creator_active(0,alpha) annihilator_Ref(alpha); no normalization',
                  'reference_site': native['sites'][ref], 'Ref_occupancies': [4, 3],
                  'background': '128 per site plus 4 delta_ref', 'spectral_bins': 256,
                  'virtual_electron_H': 'same Zdag(hbare+literal neutral DeltaC+full DW)Z as zero-boson H',
                  'probe_z': [complex(0, .5), complex(10, .25), complex(30, 1)],
                  'actual_mean_input': {'path': str(mean_path), 'sha256': hashlib.sha256(mean_raw).hexdigest(), 'bytes_utf8': mean_raw.decode()},
                  'source_hashes_at_import': {path: hashlib.sha256(data).hexdigest() for path, data in SOURCES.items()},
                  'binding_phase': 'after actual supplier import, before physical calculation'}
    descriptor = charged.packed(descriptor)
    input_sha = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    hfull = s.primitives.matter_hamiltonian(native, profile, zero)
    evals, vectors = np.linalg.eigh(hfull)
    occupied = vectors[:, evals < 0]; Z = vectors[:, evals > 0]
    if (occupied.shape[1], Z.shape[1]) != (128, 128):
        raise ArithmeticError('Actual bare sea/empty spectral ranks required')
    P = occupied @ occupied.conj().T
    projectors = [P]
    for aux in native['Pprofiles'][1:]:
        e, U = np.linalg.eigh(s.primitives.matter_hamiltonian(native, aux, zero))
        projectors.append(U[:, e < 0] @ U[:, e < 0].conj().T)
    selector = np.zeros((4*V, 4)); selector[4*ref:4*ref+4] = np.eye(4)
    source = Z.conj().T @ selector
    source_Gram = source.conj().T @ source
    raw_norm = float(np.trace(source_Gram).real/4)
    # Independent actual CAR determinant for each creator channel, including
    # the filled auxiliaries and the orthogonal three-particle Ref Slaters.
    aux_Gram = np.prod([float(np.linalg.det(Q.conj().T@Q).real)**p['load']
                       for p, Q in zip(native['Pprofiles'][1:],
                                       [np.linalg.eigh(s.primitives.matter_hamiltonian(native, p, zero))[1][:, :128]
                                        for p in native['Pprofiles'][1:]], strict=True)])
    CAR_norms = []
    for alpha in range(4):
        addition = Z @ source[:, alpha]
        slater = np.column_stack((occupied, addition))
        reference = np.eye(4)[:, [i for i in range(4) if i != alpha]]
        CAR_norms.append(float((np.linalg.det(slater.conj().T@slater)*np.linalg.det(reference.conj().T@reference)).real*aux_Gram/4))
    base = literal_compensated_coulomb(native, projectors, np.eye(4), ref)
    L = native['Lplus']; u = native['UE']/native['a']
    Cfull = 2*u*np.kron(np.diag(np.diag(L)), np.eye(4)) - 2*u*np.repeat(np.repeat(L, 4, axis=0), 4, axis=1)*P - 2*u*np.kron(np.diag(L[:, ref]), np.eye(4))
    checks = []
    orbitals = [Z[:, k] for k in range(128)]
    orbitals.extend((Z[:, k]+1j*Z[:, (k+17)%128])/np.sqrt(2) for k in range(0, 128, 16))
    for k, orbital in enumerate(orbitals):
        alpha = k % 4; Pref = np.eye(4); Pref[alpha, alpha] = 0
        added = literal_compensated_coulomb(native, [P+np.outer(orbital, orbital.conj()), projectors[1]], Pref, ref)
        predicted = float(np.vdot(orbital, Cfull@orbital).real)
        checks.append({'state': k, 'Refhole': alpha, 'literal_all64_CAR_energy_change': added['energy']-base['energy'],
                       'neutral_DeltaC_expectation': predicted,
                       'error': float(abs(added['energy']-base['energy']-predicted)),
                       'actual_total_added_charge': added['total_charge']-base['total_charge']})
    JA, rho = projected_operators(native, profile, Z)
    neutral_density = rho.copy(); neutral_density[ref] -= np.eye(128)
    total_density = np.sum(neutral_density, axis=0)
    covariance = np.zeros((3*V, 3*V)); counts = [0, 0, 0, 0]
    vertex_errors = []
    for site in native['sites']:
        q = tuple(int(v) for v in site)
        if q > tuple((-v)%native['N'] for v in q):
            continue
        block = supplier.block(q); edge = block.data['physical_edgeframe']; density = block.data['physical_densityframe']
        covariance += native['a']**2 * edge @ np.real(block.amplitudes[:block.M]@block.amplitudes[:block.M].conj().T) @ edge.T
        counts = [counts[0]+block.M, counts[1]+block.R, counts[2]+1, counts[3]+len(block.omega)]
        if q in ((0, 0, 0), (1, 0, 0), (1, 2, 0)):
            for m in range(block.M):
                direct = Z.conj().T@s.primitives.path_derivative(native, profile, zero, native['a']*edge[:, m])@Z
                projected = np.einsum('x,xij->ij', edge[:, m], JA)
                vertex_errors.append(float(np.max(abs(direct-projected))))
            for m in range(block.R):
                direct = Z.conj().T@np.diag(np.repeat(density[:, m], 4))@Z-density[ref, m]*np.eye(128)
                projected = np.einsum('x,xij->ij', density[:, m], neutral_density)
                vertex_errors.append(float(np.max(abs(direct-projected))))
    if counts != [129, 63, 36, 3865]:
        raise ArithmeticError('Incomplete physical covariance consumption')
    DWfull = np.zeros_like(hfull); variances = []
    for path in profile['paths']:
        edges = list(path['edges']); variance = float(np.sum(covariance[np.ix_(edges, edges)])); variances.append(variance)
        x, y = path['source'], path['target']; hop = path['hop']
        DWfull[4*x:4*x+4, 4*y:4*y+4] += -.5*variance*hop
        DWfull[4*y:4*y+4, 4*x:4*x+4] += -.5*variance*hop.conj().T
    cv, CU = np.linalg.eigh(covariance); DWcheck = np.zeros_like(hfull)
    for k in np.flatnonzero(cv > 1e-12):
        direction = CU[:, k]*np.sqrt(cv[k])
        DWcheck += .5*s.primitives.path_derivative(native, profile, zero, direction, 2, other=direction)
    static = Z.conj().T@(hfull+Cfull+DWfull)@Z
    star = stream_resolvents(supplier, Z, static, JA, rho, ref,
                             (complex(0, .5), complex(10, .25), complex(30, 1)))
    for probe in star['retarded_probes']:
        channel = source.conj().T@probe['G']@source
        # Refhole orthogonality makes the actual E source channel matrix
        # diagonal. Offdiagonal electron overlaps are NOT coherent Refholes.
        E_matrix = np.diag(np.diag(channel))/4
        E_spectral = -(E_matrix-E_matrix.conj().T)/(2j)
        probe.update({'electron_source_channel_G': channel, 'actual_Refhole_E_source_matrix_G': E_matrix,
                      'actual_unrescaled_E_source_scalar_G': np.trace(E_matrix),
                      'actual_E_source_minus_Im_matrix': E_spectral,
                      'actual_E_source_minus_Im_eigenvalues': np.linalg.eigvalsh(E_spectral),
                      'actual_E_source_minus_Im_scalar': float(np.trace(E_spectral).real),
                      'source_spectral_contraction_error': float(np.max(abs(E_spectral-np.diag(np.diag(source.conj().T@probe['minus_Im_G']@source))/4)))})
    wrong = star['missing_Ref_scalar_counterfactual']
    wrong['unrescaled_E_source_scalar_G'] = np.trace(source.conj().T@wrong['G']@source)/4
    wrong['E_source_scalar_defect'] = float(abs(wrong['unrescaled_E_source_scalar_G']-star['retarded_probes'][1]['actual_unrescaled_E_source_scalar_G']))
    errors = {'raw_E_norm_minus_half': abs(raw_norm-.5), 'literal_CAR_source_norm_join': abs(sum(CAR_norms)-raw_norm),
              'all64_compensated_C_join_max': max(c['error'] for c in checks),
              'total_added_charge_max': max(abs(c['actual_total_added_charge']) for c in checks),
              'zero_total_density_operator_max': float(np.max(abs(total_density))),
              'full_DW_literal_second_derivative_max': float(np.max(abs(DWfull-DWcheck))),
              'physical_projected_vertex_max': max(vertex_errors),
              'static_H_hermiticity_max': float(np.max(abs(static-static.conj().T))),
              'positive_star_identity_max': max(p['positive_star_identity_max_error'] for p in star['retarded_probes']),
              'minus_Im_G_min': min(float(np.min(p['minus_Im_G_eigenvalues'])) for p in star['retarded_probes']),
              'minus_Im_E_min': min(float(np.min(p['actual_E_source_minus_Im_eigenvalues'])) for p in star['retarded_probes'])}
    if max(value for key, value in errors.items() if key not in ('minus_Im_G_min', 'minus_Im_E_min')) > 1e-7 or min(errors['minus_Im_G_min'], errors['minus_Im_E_min']) < -1e-10:
        raise ArithmeticError('Physical neutral consumer identity failure: '+str(errors))
    if wrong['E_source_scalar_defect'] <= 1e-10 or np.trace(star['q0_holonomy_Gram']).real <= 1e-10:
        raise ArithmeticError('Reference coupling/holonomy adverse control failed')
    s.assert_frozen()
    return {'status': 'actual_native_EM_neutral_Refhole_positive_Gaussian_star_control',
            'input_descriptor': descriptor, 'input_sha256': input_sha, 'errors': errors,
            'source_at_import': {Path(path).name: {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes_utf8': data.decode()} for path, data in SOURCES.items()},
            'source_unchanged_since_import': {Path(path).name: Path(path).read_bytes() == data for path, data in SOURCES.items()},
            'input_receipt_unchanged': photon_raw == charged.RECEIPT.read_bytes() and mean_raw == mean_path.read_bytes(),
            'runtime': {'executable': sys.executable, 'python': sys.version, 'numpy': np.__version__},
            'bare_active_energies': evals, 'bare_filled_active_P': P, 'bare_empty_Z': Z,
            'source_all_four_v_alpha': source, 'raw_electron_source_Gram': source_Gram,
            'actual_Refhole_E_source_Gram': np.diag(np.diag(source_Gram))/4,
            'actual_unrescaled_Eraw_norm_squared': raw_norm, 'independent_all64_CAR_Refhole_source_norms': CAR_norms,
            'source_total_charge': float(np.trace(source.conj().T@total_density@source).real/4),
            'actual_added_density_operators': neutral_density, 'actual_zero_total_added_density_operator': total_density,
            'bath_PH_total_charge': 0, 'bath_charge_reason': 'Each physical RPA PH mode is built from number-preserving current/density bilinears; no q0 density fluctuation. Electron+Refhole charge operator is independently evaluated above.',
            'sea_reference_charge_before': base['total_charge'], 'actual_all64_neutral_C_energy': base['energy'],
            'actual_all64_background': base['background'], 'actual_all64_neutral_density_mean': base['mean'],
            'literal_all64_compensated_C_checks': checks, 'full_neutral_DeltaC': Cfull,
            'bare_h_projected': Z.conj().T@hfull@Z, 'neutral_DeltaC_projected': Z.conj().T@Cfull@Z,
            'full_DW_projected': Z.conj().T@DWfull@Z, 'static_neutral_H_projected': static,
            'phase_variance_range': [min(variances), max(variances)], 'neutral_star': star,
            'law': 'Four orthogonal Refhole channels, each Hstar=[[H0,B],[Bdag,direct_sum_l(H0+Omega_l I)]], H0=Zdag(hbare+DeltaCneutral+DW)Z. B_l=Zdag[sum JA_m bA_ml + sum(rho_m-phi_m(ref)I)bphi_ml]Z Ustatic. DeltaCneutral=2uLxx I-2uLxy Pactive_xy-2u diag(L_xref)I. G=(z-H0-Sigma)^-1; Sigma=sum_l B_l(z-H0-Omega_l)^-1 B_ldag in the static eigenbasis. Eraw scalar G=sum_alpha v_alpha_dag G v_alpha/4. Static instantaneous C occurs once; scalar bath is only correlation potential. Refhole has zero intrinsic on-site energy and no photon current.',
            'scope': ['NEW finite N4 AP222 positive-electron, zero/one-boson compressed linear Gaussian law. Actual fixed bare negative sea, all64 species loads, actual full129A+63phi frames and3865positive modes. Ref4->3 source total charge0, raw E norm1/2; no forced normalization or source whitening.',
                      'Hermitian star positivity is exact for this declared finite auxiliary law, not an interacting full-CAR spectral theorem. Pauli blocking is only the fixed bare positive projection: overlapping added-electron and sea-PH excitations are not antisymmetrized as a joint CAR Hilbert space; multipair/nonlinear Peierls orders and vacuum backreaction are absent.',
                      'No original canonical Y/source/probability replacement, no original-source admission, no complete SU4/Phi/Gauss vacuum, no real4096 transfer, no pole/mass/commoncone/eR/anomaly/fullQED promotion. Large actual phase variance does not supply a weak-field remainder. Different covariant laws are not composed.'],
            'elapsed_seconds': time.monotonic()-started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run()
    if not all(report['source_unchanged_since_import'].values()) or not report['input_receipt_unchanged']:
        raise RuntimeError('Scientific source/input changed during execution')
    args.output.write_text(json.dumps(charged.packed(report), indent=2, allow_nan=False)+'\n')
    print(json.dumps({'errors': report['errors'], 'raw_E_norm_squared': report['actual_unrescaled_Eraw_norm_squared'],
                      'source_probes': [{'z': charged.packed(p['z']), 'Eraw_G': charged.packed(p['actual_unrescaled_E_source_scalar_G']),
                                         'minus_Im_Eraw': p['actual_E_source_minus_Im_scalar']} for p in report['neutral_star']['retarded_probes']],
                      'missing_Ref_source_defect': report['neutral_star']['missing_Ref_scalar_counterfactual']['E_source_scalar_defect'],
                      'input_sha256': report['input_sha256'], 'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)
