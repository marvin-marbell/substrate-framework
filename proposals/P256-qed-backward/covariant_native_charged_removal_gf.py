#!/usr/bin/env python3
"""NEW static positive-compensator removal diagnostic; frozen source unchanged.

N3 removal is not creation on the original filled Ref4.  This diagnostic
uses a distinct static positive countercharge at site0, with a Ref0->1
quartet tag and reversed integer Wilson paths.  No full CAR/Dirac claim.
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
SOURCE_NAMES = ('covariant_native_charged_removal_gf.py', 'covariant_native_charged_site_gf.py',
    'covariant_native_electric_heat_comparison.py', 'covariant_native_closed_auxiliary_loop.py',
    'covariant_native_source_gram_moments.py', 'covariant_native_haar_range_jets.py',
    'covariant_source_band_halfstep_control.py', 'covariant_induced_photon.py',
    'covariant_photon_energy_control.py', 'sea_selection.py', 'rotation_band.py')
SOURCE_BYTES_AT_IMPORT = {name: (ROOT/name).read_bytes() for name in SOURCE_NAMES}
PLAN_PATH = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT = PLAN_PATH.read_bytes()
ADDITION_PATH = ROOT/'covariant_native_charged_site_gf_cut2_UE1_capsule_20261008.json'
ADDITION_BYTES_AT_IMPORT = ADDITION_PATH.read_bytes()
ADDITION_SHA = '40792c2a85776ec550e4ddddff848f2cc877907a1f1c5f710b05a189729779ec'
if hashlib.sha256(ADDITION_BYTES_AT_IMPORT).hexdigest() != ADDITION_SHA:
    raise RuntimeError('Actual frozen addition-state capsule bytes changed')
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
import covariant_native_electric_heat_comparison as electric


def complex_array(value: np.ndarray) -> dict[str, Any]:
    return dict(real=np.real(value).tolist(), imag=np.imag(value).tolist())


def decode_complex(value: dict[str, Any]) -> np.ndarray:
    return np.array(value['real'])+1j*np.array(value['imag'])


def annihilator(neutral: dict[str, Any], removal: dict[str, Any], coefficients: np.ndarray,
                embedding: np.ndarray) -> Any:
    nn, nc = len(neutral['physical']), len(removal['physical'])
    paths = ((0,0,0,0), (-1,0,0,0), (-1,-1,0,0), (0,0,0,1))
    weights = (coefficients@embedding.conj().T).conj()
    rows, columns, entries = [], [], []
    for column, (state, n, electric_flux) in enumerate(neutral['physical']):
        occupied, auxiliary = neutral['states'][state]
        mask = sum(1 << mode for mode in occupied)
        for mode in occupied:
            target_occupation = tuple(m for m in occupied if m != mode)
            target = removal['state_index'][(target_occupation, auxiliary)]
            flux = tuple(a-b for a,b in zip(electric_flux, paths[mode//2]))
            row = removal['index'].get((target, flux[3]))
            if row is None:
                continue
            if removal['physical'][row][2] != flux:
                raise ArithmeticError('Actual reversed Wilson source flux mismatch')
            density = electric.native.density(target_occupation, auxiliary, -1)
            if any(flux[x]-flux[(x-1)%4] != density[x] for x in range(4)):
                raise ArithmeticError('Actual removal and static positive compensator violate Gauss')
            below = (mask & ((1 << mode)-1)).bit_count()
            # c_ref^dag c_active: active annihilation leaves N3+auxN2=5
            # occupied modes preceding EMPTY Ref; creation adds this odd sign.
            sign = (-1)**(below+len(target_occupation)+len(auxiliary))
            for ref in range(4):
                for node in range(3):
                    rows.append((ref*3+node)*nc+row)
                    columns.append(node*nn+column)
                    entries.append(sign*weights[mode,ref])
    return coo_matrix((entries,(rows,columns)), shape=(12*nc,3*nn)).tocsr()


def spectral_matrix(amplitudes: np.ndarray, energy: np.ndarray, preparation: str) -> dict[str, Any]:
    W = np.einsum('xrj,yrj->jxy', amplitudes.conj(), amplitudes)
    weights = np.sum(abs(amplitudes)**2, axis=1)
    norms = weights.sum(axis=1)
    if np.any(norms <= 0):
        raise ArithmeticError('Actual removal source site probe has zero support')
    eigenvalues = np.linalg.eigvalsh(W)
    times = []
    for elapsed in (0., .5, 1., 2.):
        forward = np.einsum('j,jxy->xy',np.exp(-1j*elapsed*energy),W)
        negative_frequency = np.einsum('j,jyx->xy',np.exp(1j*elapsed*energy),W)
        times.append(dict(time=elapsed, hole_forward_GF=complex_array(forward),
            electron_removal_negative_frequency_matrix=complex_array(negative_frequency)))
    return dict(preparation=preparation, all_complex_compensator_amplitudes=complex_array(amplitudes),
        all_site_spectral_matrices=complex_array(W), all_site_absolute_energy_weights=weights.tolist(),
        all_site_normalized_energy_weights=(weights/norms[:,None]).tolist(),
        site_norm_squared=norms.tolist(), site_energy_means=(weights@energy/norms).tolist(),
        all_real_time_matrices=times, minimum_W_eigenvalue=float(eigenvalues.min()),
        maximum_W_Hermiticity_defect=float(np.linalg.norm(W-W.conj().transpose(0,2,1),axis=(1,2)).max()),
        all_W_eigenvalues=eigenvalues.tolist(), no_residue_or_level_dropped=True)


def run() -> dict[str, Any]:
    addition = json.loads(ADDITION_BYTES_AT_IMPORT)
    descriptor = addition['input_descriptor']
    if (descriptor['UE'], descriptor['electric_cutoff'], descriptor['Haar_grid'],
        descriptor['active_kappa'], descriptor['auxiliary_kappa']) != (1.,2,32,.09,.04):
        raise RuntimeError('Frozen actual neutral-state input law does not match removal law')
    gamma, embedding = decode_complex(addition['selected_gamma']), decode_complex(addition['selected_embedding'])
    Omega = decode_complex(addition['actual_Omega'])
    lambdaN = addition['actual_common_neutral_lambdaN']
    neutral, removal, removal3 = electric.native.space(0,2), electric.native.space(-1,2), electric.native.space(-1,3)
    if Omega.shape != (3*len(neutral['physical']),):
        raise ArithmeticError('Frozen actual neutral ground vector has incorrect physical ordering')
    y = np.array([-.09,0.,.09]); sigma=.03; radius=.12
    k = np.exp(-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y)))
    k /= np.linalg.norm(k)
    D = np.zeros((2,3))
    for j in range(2):
        D[j,j]=math.sqrt(k[j+1]/k[j])/.09; D[j,j+1]=-math.sqrt(k[j]/k[j+1])/.09
    values, frame = eigh(2*sigma*sigma*D.T@D)
    heat = (frame*np.exp(-values/2))@frame.T
    print(json.dumps(dict(stage='actual_N3_fullcanonical_primitiveE_factor', dimension=3*len(removal['physical']))),flush=True)
    factor = electric.full_factor(gamma,y,heat,removal,32,1.)
    diagnostics = {key:value for key,value in factor.items() if key not in ('F','Haar')}
    del factor['Haar']
    spectrum = electric.prepare_spectrum(factor.pop('F')); del factor
    V, q = spectrum['V'], spectrum['q']/math.sqrt(lambdaN)
    energy = -2*np.log(q)
    mean_half,_ = electric.native.square_jets(gamma,.09,0.)
    mean_values, mean_vectors = eigh(mean_half[0]@mean_half[0])
    negative = mean_vectors[:,mean_values>1]
    if negative.shape[1] != 4:
        raise ArithmeticError('Actual negative mean band is not the specified four-dimensional sea')
    retained3 = np.array([removal3['index'][(state,n)] for state,n,_ in removal['physical']])
    omitted3 = np.array([(state,n) not in removal['index'] for state,n,_ in removal3['physical']])
    raw, coefficients, boundary = [], [], []
    for site, packet in enumerate(np.eye(4)):
        template = np.kron(packet[:,None],np.eye(2))
        coefficient = negative@negative.conj().T@template
        coefficient /= np.linalg.norm(coefficient,'fro')
        coefficients.append(coefficient)
        image = (annihilator(neutral,removal,coefficient,embedding)@Omega).reshape(4,3,-1)
        complete = (annihilator(neutral,removal3,coefficient,embedding)@Omega).reshape(4,3,-1)
        mapped = complete[:,:,retained3]
        boundary.append(dict(site=site, full_source_from_FIXED_Omega_cut2_norm_squared=float(np.sum(abs(complete)**2)),
            retained_cut2_norm_squared=float(np.sum(abs(mapped)**2)),
            omitted_cut2_norm_squared=float(np.sum(abs(complete[:,:,omitted3])**2)),
            actual_retained_mapping_defect=float(np.linalg.norm(mapped-image))))
        raw.append(image.reshape(4,-1))
    raw = np.array(raw)
    amplitudes = (V.conj().T@raw.reshape(16,-1).T).T.reshape(4,4,-1)
    prepared = q[None,None,:]*amplitudes
    reconstructed = (V@amplitudes.reshape(16,-1).T).T.reshape(raw.shape)
    null = np.sum(abs(raw-reconstructed)**2,axis=(1,2))
    raw_result = spectral_matrix(amplitudes,energy,'RAW N3 annihilation with distinct STATIC +1 compensator; no halfstep')
    prepared_result = spectral_matrix(prepared,energy,'ONE wholechargedhalfstep q times RAW removal amplitudes')
    removal_descriptor = dict(plan_id=PLAN_PATH.stem, plan_sha256_at_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest(),
        actual_frozen_addition_capsule_sha256=ADDITION_SHA, neutral_state_REUSED_from_actual_frozen_capsule=True,
        factor='SAME frozen3a F=Phiheat SF exp[-sum_ALL4edgesE²/2]; K=FFdag', UE=1., electric_cutoff=2, Haar_grid=32,
        active_modes=8, neutral_active_particles=4, removal_active_particles=3, active_CAR_states=[70,56],
        auxiliary_modes=4, auxiliary_particles=2, auxiliary_CAR_states=6, auxiliary_kappa=.04, active_kappa=.09,
        Phi3='threelevels of ONE coordinate', Phi_nodes=y.tolist(), selected_Clifford_dimension=2,
        compensator='NEW static positive charge+1 atsite0, Ref0->1 quartet tag, nondynamical; NOT creation on frozenfilledRef4',
        source='negative-mean-band annihilator c_refdag c_active, ALL4tagcomponents; opposite integerWilsonpaths',
        original_scattering_Ref4to3_source_unchanged=True, source_paths_reversed=[[0,0,0,0],[1,0,0,0],[1,1,0,0],[0,0,0,-1]],
        common_neutral_lambdaN=lambdaN, transfer_duration=1., spacing=1., quantum_law_and_parameters='declared inputs',
        scope='Paired addition/removal diagnostic on SAME active/aux/Phi/electric law, DIFFERENT staticcompensator/reference initialsector. NOT a fullsameoperatorCARsumrule/Dirac/commoncone/freeelectronGF orinducedE explanation')
    result: dict[str,Any] = dict(input_descriptor=removal_descriptor,
        input_descriptor_sha256=hashlib.sha256(json.dumps(removal_descriptor,sort_keys=True).encode()).hexdigest(),
        sources_at_IMPORT={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes=data.decode()) for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        runtime_packages=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        physical_metric=addition['physical_metric'], selected_gamma=complex_array(gamma), selected_embedding=complex_array(embedding),
        actual_factor_diagnostics=diagnostics, actual_base_removal_positive_rank=len(q), actual_min_positive_normalized_singular=float(q[-1]),
        actual_removal_energy=energy.tolist(), actual_negative_mean_band_values=mean_values[mean_values>1].tolist(),
        actual_site_negative_band_coefficients=complex_array(np.array(coefficients)), actual_raw_removal_GF=raw_result,
        actual_wholehalf_prepared_removal_GF=prepared_result, actual_source_Knull_AFTERcut2_projection=null.tolist(),
        actual_literal_removal_source_boundary_FROM_FIXED_Omega_cut2=boundary,
        actual_addition_comparison=dict(addition_lowest_energy=float(min(addition['charged_levels']['energy'])),
            removal_lowest_energy=float(energy.min()), raw_addition_site_means=addition['actual_raw_addition_GF']['site_raw_addition_energy_means'],
            raw_removal_site_means=raw_result['site_energy_means'], wholehalf_addition_site_means=addition['actual_site_energy_means'],
            wholehalf_removal_site_means=prepared_result['site_energy_means'],
            all_base_spectral_energy_absolute_differences=(np.sort(energy)-np.sort(addition['charged_levels']['energy'])).tolist()))
    print(json.dumps(dict(stage='actual_N3_source_and_spectral_GF', lowest_energy=float(energy.min()),
        raw_site_means=raw_result['site_energy_means'], prepared_site_means=prepared_result['site_energy_means'],
        boundary_losses=[row['omitted_cut2_norm_squared'] for row in boundary])),flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); started=time.monotonic(); result=run(); result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(source_sha256=result['sources_at_IMPORT'][Path(__file__).name]['sha256'],
        input_descriptor_sha256=result['input_descriptor_sha256'],runtime_seconds=result['runtime_seconds'])),flush=True)


if __name__ == '__main__':
    main()
