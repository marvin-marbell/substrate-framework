"""NEW continuous-BZ regulated native Gaussian/GW charged Green, not SOURCE.

All positive cos/sin canonical PH/field modes are a finite spectral regulator,
NOT stable UV photons. Bare and static-internal candidates share one frozen
parameter set, literal one/two-link vertices, Coulomb and Gaussian contact.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any
ROOT = Path(__file__).resolve().parent
SOURCES: dict[str, bytes] = {str(Path(__file__).resolve()): Path(__file__).read_bytes()}
_GET_CODE = importlib.machinery.SourceFileLoader.get_code


def capture_code(self: Any, fullname: str) -> Any:
    path = Path(self.path).resolve()
    if path.suffix == '.py' and (path.is_relative_to(ROOT) or path.parent == Path('/tmp')):
        raw = path.read_bytes()
        previous = SOURCES.setdefault(str(path), raw)
        if previous != raw:
            raise RuntimeError(f'Import changed source {path}')
    return _GET_CODE(self, fullname)


importlib.machinery.SourceFileLoader.get_code = capture_code
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
PROVIDER = Path('/tmp/induced-native-bulk-photon-green.py')
PROVIDER_SHA = '3e905f9f4941b19658bf90383236546faafaa269ea80a65d2d828232b8fa2440'
PHOTON_RECEIPT = ROOT / 'induced_native_bulk_photon_green_receipt.json'
PHOTON_RECEIPT_SHA = '1849900b3611873863ede1e2dd788d75093af35a6dd7cffe4f94ccda125078a2'
ASSIGNED = 'd86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7'
CURRENT_OBSERVED = '452ad8053dd6efcbc7d30b55443c1da57c4e6b7e6866b58b11fc61284c6d0961'


def packed(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return packed(value.tolist())
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, np.generic):
        return packed(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): packed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [packed(v) for v in value]
    return value


def load_provider() -> Any:
    if hashlib.sha256(PROVIDER.read_bytes()).hexdigest() != PROVIDER_SHA:
        raise RuntimeError('Frozen full causal provider changed')
    if hashlib.sha256(PHOTON_RECEIPT.read_bytes()).hexdigest() != PHOTON_RECEIPT_SHA:
        raise RuntimeError('Frozen bulk photon receipt changed')
    spec = importlib.util.spec_from_file_location('native_bulk_charged_full_photon', PROVIDER)
    if spec is None or spec.loader is None:
        raise RuntimeError('Full causal photon provider inaccessible')
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quadrature(a: float, radial_order: int, angular_order: int, regulator: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Six disjoint max-coordinate pyramids tile the complete BZ.

    qtheta=t*(+/-1,u,v), Jacobian t^2. The q=0 integrable Coulomb
    singularity is integrated on [0,t1], never dropped or substituted.
    Radial shells resolve soft photon and active gap scales at fixed a.
    """
    physical = [0., .005, .02, .08, .32, 1.28, 5.12, 20.48, np.pi/a]
    if regulator == 'shifted-shell':
        physical = [0., .007, .028, .112, .448, 1.792, 7.168, 28.672, np.pi/a]
    edges = [float(x*a) for x in physical]
    rn, rw = leggauss(radial_order)
    an, aw = leggauss(angular_order)
    points = []
    weights = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for node, weight in zip(rn, rw):
            t = (lo+hi)/2 + (hi-lo)*node/2
            for axis in range(3):
                other = [j for j in range(3) if j != axis]
                for sign in (-1., 1.):
                    for i, u in enumerate(an):
                        for j, v in enumerate(an):
                            q = np.empty(3)
                            q[axis] = sign*t
                            q[other] = [t*u, t*v]
                            points.append(q)
                            weights.append(weight*(hi-lo)/2*t*t*aw[i]*aw[j]/(2*np.pi)**3)
    return np.asarray(points), np.asarray(weights), edges


def coulomb(model: Any, k: np.ndarray, q: np.ndarray, weight: np.ndarray, inertia: float) -> np.ndarray:
    """u int L(q)(I-2 Pminus(k-q)); u=UEphys/a, no auxiliary exchange.

    Equal filled density gives zero background Hartree. Both self and active
    exchange are retained; I-2Pminus=h/E is their exact algebraic sum.
    """
    kernel = inertia/model.config.a*weight/np.sum((2*np.sin(q/2))**2, axis=1)
    answer = np.zeros((len(k), 4, 4), complex)
    for momentum, coefficient in zip(q, kernel):
        _, lower, _ = model.projectors(k-momentum)
        answer += coefficient*(np.eye(4)-2*lower)
    return answer


def dw_h(model: Any, k: np.ndarray, variances: np.ndarray) -> np.ndarray:
    answer = np.zeros((len(k), 4, 4), complex)
    for variance, (axis, length, matrix) in zip(variances, model.paths):
        phase = np.exp(1j*length*k[:, axis])[:, None, None]
        answer -= .5*variance*(phase*matrix+phase.conj()*matrix.conj().T)
    return answer


def sigma(B: np.ndarray, d: np.ndarray, z: complex) -> np.ndarray:
    return (B/(z-d)[None, :])@B.conj().T


def pole_rows(h: np.ndarray, B: np.ndarray, d: np.ndarray) -> list[dict[str, Any]]:
    """Actual poles ONLY in the central bath-free finite-regulator interval."""
    lower = float(np.max(d[d < 0]))
    upper = float(np.min(d[d > 0]))
    margin = max(1e-9, (upper-lower)*1e-9)
    left, right = lower+margin, upper-margin
    def matrix(x: float) -> np.ndarray:
        value = x*np.eye(4)-h-sigma(B, d, complex(x))
        return (value+value.conj().T)/2
    el, er = np.linalg.eigvalsh(matrix(left)), np.linalg.eigvalsh(matrix(right))
    rows: list[dict[str, Any]] = []
    for band in range(4):
        if not el[band] < 0 < er[band]:
            continue
        solved, info = brentq(lambda x: float(np.linalg.eigvalsh(matrix(x))[band]), left, right, xtol=2e-11, full_output=True)
        if not info.converged:
            raise ArithmeticError('Charged Schur pole solver did not converge')
        energy = float(solved)
        if any(abs(energy-row['energy']) < 1e-7 for row in rows):
            continue
        values, vectors = np.linalg.eigh(matrix(energy))
        selected = abs(values) < 1e-7
        W = vectors[:, selected]
        metric = np.eye(4)+(B/((energy-d)**2)[None, :])@B.conj().T
        residue = W@np.linalg.solve(W.conj().T@metric@W, W.conj().T)
        rows.append({'energy': energy, 'multiplicity': int(np.sum(selected)), 'residue': residue,
                     'residue_trace': float(np.real(np.trace(residue))),
                     'root_residual': float(np.max(abs(matrix(energy)@W))),
                     'virtual_threshold_distance': float(np.min(abs(energy-d))),
                     'finite_regulator_central_interval': [lower, upper]})
    return rows


def dispersion(rows: list[dict[str, Any]], photon_speed: float) -> dict[str, Any]:
    positive = [[r for r in row['poles'] if r['energy'] > 0] for row in rows]
    if any(not r for r in positive):
        return {'isolated_pole_curvature_available': False, 'reason': 'At least one p has no isolated central finite-regulator positive pole; report actual G and virtual thresholds.'}
    energies = np.asarray([min(r['energy'] for r in poles) for poles in positive])
    p = np.asarray([row['p_physical'][0] for row in rows])
    coefficient = (energies[1:]**2-energies[0]**2)/p[1:]**2
    return {'isolated_pole_curvature_available': True, 'finite_regulator_dressed_mass': float(energies[0]),
            'energies': energies, 'actual_E_squared_secants_no_speed_fit': coefficient,
            'photon_cone_squared': photon_speed**2,
            'absolute_squared_cone_mismatch': abs(coefficient-photon_speed**2),
            'bare_squared_cone_mismatch': abs(.1601069274**2-photon_speed**2),
            'loop_improves_squared_cone_mismatch': abs(coefficient-photon_speed**2) < abs(.1601069274**2-photon_speed**2),
            'positive_rest_curvature': coefficient > 0,
            'real_velocity_admitted': False,
            'physical_continuum_pole_admitted': False}


def observe(h: np.ndarray, B: np.ndarray, d: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    poles = pole_rows(h, B, d)
    positive = [row['energy'] for row in poles if row['energy'] > 0]
    center = min(positive) if positive else float(np.linalg.eigvalsh(h)[2])
    probes = []
    minimum = float('inf')
    for eta in (.02, .005):
        for energy in (0., .5*center, center, 1.05*center, 1.5*center):
            z = complex(energy, eta)
            S = sigma(B, d, z)
            G = np.linalg.solve(z*np.eye(4)-h-S, np.eye(4))
            spectral = -(G-G.conj().T)/(2j)
            eigenvalues = np.linalg.eigvalsh(spectral)
            minimum = min(minimum, float(eigenvalues[0]))
            probes.append({'z': z, 'Sigma_R': S, 'G_R': G, 'minus_ImG_eigenvalues': eigenvalues,
                           'spectral_trace_over_pi': float(np.real(np.trace(spectral))/np.pi)})
    if minimum < -1e-8:
        raise ArithmeticError('Full charged retarded Green failed spectral positivity')
    return {'p_physical': p, 'external_hbare_C_DW': h, 'poles': poles,
            'actual_positive_virtual_threshold': float(np.min(d[d > 0])),
            'actual_negative_virtual_threshold': float(np.max(d[d < 0])),
            'minimum_minus_ImG_eigenvalue': minimum, 'actual_causal_G_probes': probes,
            'spectral_width_scope': 'eta is the declared resolvent broadening, NOT a measured physical width; no UV stable-photon assertion.'}


def run_configuration(provider: Any, radial: int, angular: int, bins_list: list[int], regulator: str, response_order: int) -> dict[str, Any]:
    supplier = provider.NativeBulkPhotonGreen(response_order)
    model, cfg = supplier.model, supplier.cfg
    a = cfg.a
    q, weight, edges = quadrature(a, radial, angular, regulator)
    pphysical = np.asarray([[x, 0., 0.] for x in (0., .02, .04, .08)])
    ptheta = a*pphysical
    Cext = coulomb(model, ptheta, q, weight, cfg.inertia)
    internal_k = (ptheta[:, None, :]-q[None, :, :]).reshape(-1, 3)
    Cinternal = coulomb(model, internal_k, q, weight, cfg.inertia).reshape(4, len(q), 4, 4)
    hbare_ext = model.h(ptheta)
    hbare_internal = model.h(internal_k).reshape(4, len(q), 4, 4)
    store = {bins: {'variances': np.zeros(len(model.paths)), 'q_modes': [], 'omega': [], 'plus': [], 'minus': [],
                    'canonical_errors': [], 'mixed_causal_norms': []} for bins in bins_list}
    probe_z = complex(.12, .03)
    qrows = []
    start = time.monotonic()
    for index, (qtheta, w) in enumerate(zip(q, weight)):
        block = supplier.block(qtheta/a)
        exact = block.retarded(probe_z)
        qrows.append({'qtheta': qtheta, 'weight': w, 'actual_uncompressed_PH_transitions': len(block.delta),
                      'microscopic_vertex_Ward': block.data['vertex_error'], 'spectral_Ward': block.data['spectral_ward'],
                      'mixed_Ward': block.data['mixed_ward'], 'bare_C': block.bare_C,
                      'uncompressed_mixed_retarded_norm': float(np.linalg.norm(exact[:2, 2]))})
        for bins in bins_list:
            canonical = block.canonical(bins)
            omega = canonical['omega']
            transform = canonical['complexq_transform']
            realq = canonical['amplitudes_realq']
            plus, minus = transform@realq, transform@realq.conj()
            data = store[bins]
            data['omega'].append(omega)
            data['plus'].append(plus)
            data['minus'].append(minus)
            data['q_modes'].append(len(omega))
            error = float(np.max(abs(block.canonical_retarded(probe_z, bins)-exact))/np.max(abs(exact)))
            data['canonical_errors'].append(error)
            data['mixed_causal_norms'].append(float(np.linalg.norm(exact[:2, 2])))
            covariance_A = np.real(block.transverse@(plus[:2]@plus[:2].conj().T)@block.transverse.T)
            for path_index, (axis, length, _) in enumerate(model.paths):
                path_sum = sum(np.exp(1j*qtheta[axis]*(ell-(length-1)/2)) for ell in range(length))
                data['variances'][path_index] += w*a*a*float(abs(path_sum)**2)*covariance_A[axis, axis]
        supplier.cache.clear()
        if index % 24 == 0:
            print('Full BZ quantum block', regulator, radial, index, '/', len(q), 'seconds', time.monotonic()-start, flush=True)
    binrows = []
    photon_speed = .1531681152
    for bins, data in store.items():
        DWext = dw_h(model, ptheta, data['variances'])
        DWinternal = dw_h(model, internal_k, data['variances']).reshape(4, len(q), 4, 4)
        candidate_rows = {}
        interference = {}
        for candidate in ('bare-internal-one-shot', 'NEW-static-internal-hbare-C-DW'):
            hint = hbare_internal if candidate == 'bare-internal-one-shot' else hbare_internal+Cinternal+DWinternal
            records = []
            mixed_rows = []
            for target, p in enumerate(ptheta):
                columns = []
                bath = []
                mixed_sigma = np.zeros((4, 4), complex)
                for index, (qtheta, w) in enumerate(zip(q, weight)):
                    block_frame = np.linalg.svd((2*np.sin(qtheta/2)/np.linalg.norm(2*np.sin(qtheta/2)))[None, :])[2][1:].T
                    j, _ = model.vertices((p-qtheta/2)[None, :], qtheta)
                    JT = a*np.einsum('ia,ijk->ajk', block_frame, j[0])
                    values, U = np.linalg.eigh(hint[target, index])
                    if not values[1] < 0 < values[2]:
                        raise ArithmeticError('Internal static candidate lost its filled/empty band split')
                    for band in range(4):
                        amplitudes = data['plus'][index] if band >= 2 else data['minus'][index]
                        BA = np.einsum('aij,al,j->il', JT, amplitudes[:2], U[:, band], optimize=True)
                        BP = U[:, band, None]*amplitudes[2][None, :]
                        B = np.sqrt(w)*(BA+BP)
                        d = values[band]+data['omega'][index] if band >= 2 else values[band]-data['omega'][index]
                        columns.append(B)
                        bath.append(d)
                        mixed_sigma += w*((BA/(probe_z-d)[None, :])@BP.conj().T+(BP/(probe_z-d)[None, :])@BA.conj().T)
                B = np.concatenate(columns, axis=1)
                d = np.concatenate(bath)
                records.append(observe(hbare_ext[target]+Cext[target]+DWext[target], B, d, pphysical[target]))
                mixed_rows.append({'p': pphysical[target], 'actual_coherent_minus_incoherent_Sigma_at_probe': mixed_sigma,
                                   'mixed_interference_norm': float(np.linalg.norm(mixed_sigma)), 'positive_bath_columns_both_bands': len(d)})
            candidate_rows[candidate] = {'charged_rows': records, 'dispersion': dispersion(records, photon_speed)}
            interference[candidate] = mixed_rows
        binrows.append({'bins': bins, 'all_q_positive_quantum_mode_counts': data['q_modes'],
                        'total_positive_bosonic_modes': sum(data['q_modes']),
                        'all_physical_A_path_variances': data['variances'], 'C_external': Cext, 'DW_external': DWext,
                        'max_full_uncompressed_canonical_relative_error': max(data['canonical_errors']),
                        'all_q_full_canonical_errors': data['canonical_errors'], 'candidates': candidate_rows,
                        'actual_mixed_A_phi_counterfactuals': interference})
    return {'radial_order': radial, 'angular_order': angular, 'regulator': regulator, 'response_order': response_order,
            'radial_theta_edges': edges, 'q_count': len(q), 'quadrature_weight_sum': float(np.sum(weight)),
            'smallest_physical_q': float(np.min(np.linalg.norm(q, axis=1))/a),
            'q0_IR_treatment': 'First pyramid cell starts at t=0; Gauss t>0 with t² Jacobian integrates Coulomb 1/q² and photon 1/q. No missing mode or q0 substitution. Soft resolution is finite and explicitly convergence-priced.',
            'actual_all_q_rows': qrows, 'bin_rows': binrows, 'elapsed_seconds': time.monotonic()-start}


def run(args: Any) -> dict[str, Any]:
    start = time.monotonic()
    provider = load_provider()
    supplier = provider.NativeBulkPhotonGreen(args.response_order)
    mean_path = Path(supplier.s.primitives.MEAN)
    mean_raw = mean_path.read_bytes()
    descriptor = packed({'plan_id': 'issue-222-induced-photon-delta13-continuation-20261008',
                         'assigned_plan_revision': ASSIGNED, 'current_observed_plan_revision': CURRENT_OBSERVED,
                         'actual_native_config': asdict(supplier.cfg), 'UEphysical': 1., 'u': 1./supplier.cfg.a,
                         'provider_source_sha256': PROVIDER_SHA, 'provider_receipt_sha256': PHOTON_RECEIPT_SHA,
                         'actual_mean_input': {'path': str(mean_path), 'sha256': hashlib.sha256(mean_raw).hexdigest(), 'bytes_utf8': mean_raw.decode()},
                         'regulators': args.regulators, 'radial_orders': args.radial_orders, 'angular_order': args.angular_order,
                         'bins': args.bins, 'response_order': args.response_order,
                         'source_hashes_after_import': {path: hashlib.sha256(raw).hexdigest() for path, raw in SOURCES.items()}})
    input_sha = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    rows = []
    for regulator, radial in zip(args.regulators, args.radial_orders):
        rows.append(run_configuration(provider, radial, args.angular_order, args.bins, regulator, args.response_order))
    convergence = []
    reference = rows[0]['bin_rows'][0]
    for cfgrow in rows:
        for binrow in cfgrow['bin_rows']:
            prices = {}
            for candidate in binrow['candidates']:
                actual = binrow['candidates'][candidate]['dispersion']
                previous = reference['candidates'][candidate]['dispersion']
                if actual['isolated_pole_curvature_available'] and previous['isolated_pole_curvature_available']:
                    prices[candidate] = {'mass_difference': actual['finite_regulator_dressed_mass']-previous['finite_regulator_dressed_mass'],
                                         'E_squared_secant_differences': np.asarray(actual['actual_E_squared_secants_no_speed_fit'])-np.asarray(previous['actual_E_squared_secants_no_speed_fit'])}
                else:
                    prices[candidate] = {'mass_curvature_comparison_unavailable': True}
                fixed_probe_errors = []
                for observed, baseline in zip(binrow['candidates'][candidate]['charged_rows'], reference['candidates'][candidate]['charged_rows']):
                    # First probe is z=0+.02i for every regulator, including pole absence.
                    current_G = observed['actual_causal_G_probes'][0]['G_R']
                    baseline_G = baseline['actual_causal_G_probes'][0]['G_R']
                    fixed_probe_errors.append(float(np.linalg.norm(current_G-baseline_G)/np.linalg.norm(baseline_G)))
                prices[candidate]['actual_fixed_upperplane_G_relative_differences'] = fixed_probe_errors
            convergence.append({'regulator': cfgrow['regulator'], 'radial_order': cfgrow['radial_order'], 'bins': binrow['bins'], 'differences_from_first_configuration': prices})
    source_record = {Path(path).name: {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes_utf8': raw.decode()} for path, raw in SOURCES.items()}
    unchanged = {Path(path).name: Path(path).read_bytes() == raw for path, raw in SOURCES.items()}
    if not all(unchanged.values()) or mean_path.read_bytes() != mean_raw:
        raise RuntimeError('Scientific source/input changed during execution')
    return {'status': 'actual_complete_regulated_bulk_Gaussian_GW_charged_consumer', 'input_descriptor': descriptor,
            'input_sha256': input_sha, 'source_at_import': source_record, 'source_unchanged_since_import': unchanged,
            'actual_configurations': rows, 'actual_regulator_bin_convergence_prices': convergence,
            'known_actual_photon_cone': .1531681152, 'bare_charged_coefficient': .1601069274**2,
            'scope': ['NEW finite continuous-BZ quadrature / canonical PH frequency-bin Gaussian/GW approximation; all positive modes, both fermion bands, coherent A+phi, literal complete path vertices, bare Coulomb self/exchange and full Gaussian DW.',
                      'No IR-pole-only kernel, UV stable-photon claim, missing-q fallback, imported bare curl, UE retuning, Gamma whitening, speed fit, microscopic conserving vertex, physical eR or QED admission.',
                      'Scalar boson commutator excludes bare C; C is included once as u integral L(I-2Pminus). NEW static internal candidate uses bare-projector C plus full DW, both internal bands; it is not self-consistent/conserving GW and not frozen N4/7b.',
                      'Central isolated poles and E² secants are actual finite-regulator observations only. Massless IR continuum and UV absorption remain physical-continuum debts; broadenings are not invented physical widths.',
                      'Next distinct physical design, not achieved here: resum the Gaussian Peierls cumulant by literal hopping factor exp(-Var(pathTheta)/2), derive its actual current/contact and recompute the shared absorbing photon and charged sectors. This changes the approximation materially without UE/speed calibration; negative E² curvature cannot be renamed a velocity.'],
            'elapsed_seconds': time.monotonic()-start}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--response-order', type=int, default=8)
    parser.add_argument('--angular-order', type=int, default=2)
    parser.add_argument('--radial-orders', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--regulators', nargs='+', choices=['geometric-shell', 'shifted-shell'], default=['geometric-shell', 'shifted-shell'])
    parser.add_argument('--bins', type=int, nargs='+', default=[24, 48])
    args = parser.parse_args()
    if len(args.regulators) != len(args.radial_orders):
        parser.error('One radial order per regulator is required')
    report = run(args)
    args.output.write_text(json.dumps(packed(report), indent=2, allow_nan=False)+'\n')
    print(json.dumps(packed({'input_sha256': report['input_sha256'], 'convergence': report['actual_regulator_bin_convergence_prices'],
                             'elapsed_seconds': report['elapsed_seconds']}), indent=2), flush=True)
    print('Natural exit', args.output, hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)
