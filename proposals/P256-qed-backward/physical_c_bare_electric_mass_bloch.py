#!/usr/bin/env python3
"""Exact EVEN full-cube Bloch SF/toron point controls, not quantum poles.

The same captured CartesianCWilson scalarpref+Gamma valence and local Psi are
used. Point fields have theta=e*q/sqrt(V), never fitted widths. There is NO
field heat in this explicitly labelled point control. Source spectral rows
use the exact maximal neutral canonical Gaussian AT THAT toron point, not
the native integrated/prefix packet. All canonical sector norms follow from
the complete onebody spectrum; no huge Fock matrices are allocated.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
import numpy as np
import scipy
from scipy.linalg import eigh
ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = Path(__file__).read_bytes()
MASS_BYTES = (ROOT/'physical_c_bare_electric_mass.py').read_bytes()
import physical_c_bare_electric_mass as mass
if mass.SOURCE_BYTES != MASS_BYTES:
    raise RuntimeError('Point control dependency changed during import')
core = mass.core; reference = mass.reference; alpha = mass.alpha


def onebody(p: np.ndarray, theta: np.ndarray, bare_mass: float) -> dict[str, Any]:
    law = alpha.CartesianCWilson(alpha.gamma_frame(), alpha.Parameters(mass=bare_mass, spacing=.01), np.zeros((1, 3), int))
    result = law.potential((p+theta)[None, :], np.zeros((1, 4, 4, 4)), np.broadcast_to(np.eye(4), (1, 3, 4, 4)).copy())
    values, vectors = eigh(result['h'])
    return dict(values=values, vectors=vectors, pref=result['pref'])


def point(lengths: tuple[int, int, int], q: np.ndarray, bare_mass: float, explicit_theta: np.ndarray | None = None) -> dict[str, Any]:
    volume = math.prod(lengths); e = alpha.charge_contract(alpha.Parameters())['bare_e']
    theta = e*q/math.sqrt(volume) if explicit_theta is None else explicit_theta
    rows = []; prefactor = 0.; valence = 0.; branch_data = {5: [], 3: []}
    local_law, _, _ = reference.native_law(core.reference_parameters(core.BareElectricParameters(bare_mass=bare_mass)))
    local_phase = np.broadcast_to(theta, (2, 3)).copy()
    local_psi = local_law.source_packet_jet(local_phase, np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy(),
                                          (np.zeros((2, 3)), np.zeros((2, 3))))['psi']
    for indexes in itertools.product(*(range(length) for length in lengths)):
        momentum = 2*math.pi*np.array(indexes)/np.array(lengths)
        result = onebody(momentum, theta, bare_mass)
        values, vectors = result['values'], result['vectors']
        if not (values[1] < 0 < values[2]):
            raise ArithmeticError('Point half-filled neutral sector no longer has exactly two positive bands')
        prefactor += result['pref']; valence += float(sum(values[2:]))
        packet = (local_psi[:4]+np.exp(-1j*momentum[0])*local_psi[4:])/math.sqrt(volume)
        weights = np.sum(abs(vectors.conj().T@packet)**2, axis=1)
        for number, selected, energies in [(5, slice(0, 2), -values[:2]), (3, slice(2, 4), values[2:])]:
            branch_data[number].append((energies, weights[selected]))
        rows.append(dict(momentum=momentum.tolist(), C_completed_h_eigenvalues=values.tolist(),
                         full_scalar_prefactor=float(result['pref'])))
    observations = []
    for number in (5, 3):
        energies = np.concatenate([row[0] for row in branch_data[number]])
        raw_weights = np.concatenate([row[1] for row in branch_data[number]])
        weights = raw_weights*np.exp(-energies)
        gram = float(sum(weights)); moment = float(weights@energies)
        gap = float(min(energies)); gap_select = abs(energies-gap) <= 1e-9
        observations.append(dict(branch='addition' if number == 5 else 'dual_removal',
            canonical_number=2*volume+(1 if number == 5 else -1), actual_top_gap_TRANSFER=gap,
            actual_top_gap_PHYSICAL=gap/.01, complete_point_source_energies_TRANSFER=energies.tolist(),
            complete_point_raw_source_weights=raw_weights.tolist(), complete_point_halfstep_source_weights=weights.tolist(),
            raw_source_Gram=float(sum(raw_weights)), ONE_halfstep_Gram=gram,
            fullsource_E1_PHYSICAL=moment/.01, fullsource_mean_PHYSICAL=moment/gram/.01,
            gap_branch_raw_source_weight=float(sum(raw_weights[gap_select])),
            gap_branch_halfstep_source_weight=float(sum(weights[gap_select])),
            literal_TRANSFER_Pe0_4_fraction=float(sum(weights[(energies >= 0) & (energies <= 4)])/gram),
            literal_PHYSICAL_Pe0_4_fraction=float(sum(weights[(energies >= 0) & (energies <= .04)])/gram),
            negative_energy_weight=float(sum(weights[energies < 0]))))
    return dict(lengths=list(lengths), volume=volume, harmonic_q=q.tolist(), physical_link_theta=theta.tolist(),
        bare_mass=bare_mass, spacing=.01, full_neutral_log_lambda_N=prefactor+valence,
        retained_total_scalar_prefactor=prefactor, retained_neutral_valence_log=valence,
        full_Bloch_onebody_rows=rows, original_Psi_point_source_observations=observations,
        exact_complete_canonical_operatornorms_from_full_onebody_spectrum=True,
        point_neutral_boundary='Exact maximal neutral canonical Gaussian at THIS fixed toron. Not native integrated/prefix Gaussian.',
        no_field_heat_point_control=True, full4_spin_components=True, source_support_sites=[0, 1],
        no_quantum_photon_electronpole_claim=True)


def actual_V2_calibration() -> dict[str, Any]:
    theta = np.array([.1, -.04, .03]); p = core.BareElectricParameters()
    law, _, _ = reference.native_law(core.reference_parameters(p))
    actual = law.potential(np.broadcast_to(theta, (2, 3)).copy(), np.zeros((2, 4, 4, 4)),
                           np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy())
    values = eigh(actual['h'], eigvals_only=True)
    bloch = point((2, 1, 1), np.zeros(3), p.bare_mass, theta)
    joined = np.sort(np.concatenate([row['C_completed_h_eigenvalues'] for row in bloch['full_Bloch_onebody_rows']]))
    error = float(np.max(abs(values-joined)))
    pref_error = abs(actual['pref']-bloch['retained_total_scalar_prefactor'])
    if max(error, pref_error) > 2e-11:
        raise ArithmeticError('Exact full Bloch C-completion does not match actual captured V2')
    return dict(actual_V2_full_onebody_spectrum_error=error, actual_V2_total_scalarpref_error=pref_error,
                calibration_physical_link_theta=theta.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--length', type=int, action='append')
    parser.add_argument('--bare-mass', type=float, default=mass.OLD_MASS)
    args = parser.parse_args(); start = time.time()
    calibration = actual_V2_calibration(); cases = []
    nodes = reference.hermite_rule(2)[0]
    qcases = [np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
              np.full(3, nodes[0]), np.full(3, nodes[1])]
    for length in args.length or [6, 10, 20]:
        if length % 2 or length < 2:
            raise ValueError('Point controls require EVEN periodic cubes')
        for q in qcases:
            row = point((length, length, length), q, args.bare_mass); cases.append(row)
            print(json.dumps(dict(stage='actual_even_cube_Bloch_point', length=length, harmonic_q=q.tolist(),
                full_neutral_log_lambda=row['full_neutral_log_lambda_N'],
                gap_PHYSICAL=row['original_Psi_point_source_observations'][0]['actual_top_gap_PHYSICAL'],
                source_PHYSICAL_Pe4_fraction=row['original_Psi_point_source_observations'][0]['literal_PHYSICAL_Pe0_4_fraction'])), flush=True)
    inputs = json.dumps(dict(lengths=args.length or [6, 10, 20], bare_mass=args.bare_mass,
        harmonic_q_cases=[q.tolist() for q in qcases], no_fieldheat_point_only=True,
        source='CapturedlocalPsi zeroextended to sites0,1; all4Ref columns coherent source before spectral weights.',
        source_boundary='Point maximal neutral Gaussian, NOT native original integrated/prefix packet.'), sort_keys=True)
    sources = {Path(__file__).name: SOURCE_BYTES, 'physical_c_bare_electric_mass.py': MASS_BYTES,
        'physical_c_bare_electric_kernel.py': mass.CORE_BYTES, 'captured_reference89.py': core.REFERENCE_BYTES,
        'captured_ledger.py': reference.LEDGER_BYTES,
        **{'captured_Alpha56/'+name: data for name, data in alpha.SOURCE_BYTES.items()}}
    unchanged = {Path(__file__).name: Path(__file__).read_bytes() == SOURCE_BYTES,
        'physical_c_bare_electric_mass.py': (ROOT/'physical_c_bare_electric_mass.py').read_bytes() == MASS_BYTES,
        'physical_c_bare_electric_kernel.py': (ROOT/'physical_c_bare_electric_kernel.py').read_bytes() == mass.CORE_BYTES,
        **{name: (ROOT/name).read_bytes() == data for name, data in mass.INPUT_BYTES.items()}}
    if not all(unchanged.values()):
        raise RuntimeError('Exact point control sources/inputs changed during execution')
    execution: dict[str, Any] = dict(source_at_IMPORT={name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
        source_bytes_at_IMPORT={name: data.decode() for name, data in sources.items()},
        input_json=inputs, input_sha256=hashlib.sha256(inputs.encode()).hexdigest(), startup_unchanged=unchanged,
        capsule_hashes_at_IMPORT={name: hashlib.sha256(data).hexdigest() for name, data in mass.INPUT_BYTES.items()},
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                     platform=platform.platform()), runtime_seconds=time.time()-start, governing_plan=mass.PLAN_ID)
    result = dict(actual_V2_Bloch_calibration=calibration, actual_even_cube_full_Bloch_point_cases=cases,
        execution=execution, no_quantum_fieldheat_sourcepole_or_macro_admission=True,
        scope='Exact full3D even-cube fixed-toron Gaussian point controls only. No huge canonical Fock allocation, no fitted widths/charge/UE.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stage='capsule_written', output=str(args.output), runtime_seconds=time.time()-start,
        source_sha256=execution['source_at_IMPORT'][Path(__file__).name])), flush=True)


if __name__ == '__main__':
    main()
