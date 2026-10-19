#!/usr/bin/env python3
"""Physical generator remainder outside every field-valued eight-word ray embedding."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from original_charged_principal import portable
from original_total_angular_transfer import scalar_orbital
from prepared_neutral_slater_fibre import slater_expectation
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_phi_word_channels import coefficient_channels
from prepared_neutral_shared_photon_state import load_shared_native_cache,loaded_source_epoch


def physical_remainder(model: Any,data: dict[str,Any],profile: str,q: np.ndarray,em: np.ndarray,h: float) -> dict[str,Any]:
    sample=model.basis_at(q,em);point=sample['point'];sector='charged'
    C,metadata=coefficient_channels(data,sector,profile);W=sample[sector]@C;psi=W[:,0]
    noether=model.noether_bank.evaluate(em[:52])
    generators=source_point(model,data)[2]
    guard_bytes=generators['guard_path'].read_bytes();guard=json.loads(guard_bytes)
    pull=physical_chart_pullback(model,sample,np.asarray(guard['alpha']),np.asarray(guard['velocity']))
    gauge=model.gauge_operators(pull['gauge'],sector)
    light=scalar_orbital(model.law,point['links'],sample['a'],np.asarray(guard['velocity']))+np.diag(np.repeat(noether['lambda_C'],4))
    heavy=scalar_orbital(model.heavy_law,point['links'],np.zeros((model.law.volume,3)),np.asarray(guard['velocity']))
    combined_light=gauge['body']+light;combined_heavy=gauge['body']+heavy;reference=gauge['reference']
    first=model.fibre.onebody(combined_light,sector)+model.fibre.reference(reference,sector)
    second=model.fibre.bilinear_product(combined_light,combined_light,sector,reference,reference)
    heavy_first=slater_expectation(model.heavy_projector,combined_heavy).real
    heavy_second=slater_expectation(model.heavy_projector,combined_heavy,combined_heavy).real
    identity=model.fibre.overlap(sector)
    total_first=first+heavy_first*identity
    outside_covariance=second-first@first+(heavy_second-heavy_first**2)*identity
    eigenvalues=np.linalg.eigvalsh(W.conj().T@W)
    # SVD is the physical orthogonal projector, not coordinate slicing or jitter.
    u,s,_=np.linalg.svd(W,full_matrices=False);tol=np.finfo(float).eps*max(W.shape)*s[0]
    physical_frame=u[:,s>tol]
    dem=np.r_[noether['flow'],noether['harmonic']]
    derivative=model.directional_basis(sample,pull['normal'],dem,h)['charged']@C[:,0]
    z=-1j*derivative+total_first@psi
    residual=z-physical_frame@(physical_frame.conj().T@z)
    residual2=float(np.vdot(residual,residual).real)
    lost2=float(np.vdot(psi,outside_covariance@psi).real)
    scalar=-2*float(np.sum(noether['lambda_C']))
    # The real sea identity and any half-density divergence are in the full W span.
    with_scalar=z+scalar*psi
    with_scalar_residual=with_scalar-physical_frame@(physical_frame.conj().T@with_scalar)
    baseline=model.fibre.onebody(light,sector)@psi
    baseline_residual=baseline-physical_frame@(physical_frame.conj().T@baseline)
    return dict(profile=profile,channel_metadata=metadata,physical_word_rank=physical_frame.shape[1],word_Gram_eigenvalues=eigenvalues,
        original_function_embedding_error=float(np.linalg.norm(psi-sample['charged']@data[profile+'_normalized_source_state'])),
        physical_light_modes=model.fibre.n,microscopic_light_local_components=model.law.generators.shape[-1],reference_anchor_dimension=4,
        complete_G_derivative_plus_all_body_projected_residual_squared=residual2,
        exact_ordinary_Fock_PH_covariance_squared=lost2,complete_full_word_point_outside_norm_squared=residual2+lost2,
        physical_interacting_spin_representation_UNPAID=True,
        outside_covariance_minimum_eigenvalue=float(np.linalg.eigvalsh((outside_covariance+outside_covariance.conj().T)/2)[0]),
        actual_light_orbital_body_only_outside_word_residual_squared=float(np.vdot(baseline_residual,baseline_residual).real),
        retained_sea_scalar=scalar,scalar_identity_projection_invariance_error=float(np.linalg.norm(with_scalar_residual-residual)),
        scalar_divergence_not_zeroed_in_physical_operator=True,
        normal_field_derivative_norm=float(np.linalg.norm(derivative)),body_action_norm=float(np.linalg.norm(total_first@psi)),
        original_wave_norm_squared_at_point=float(np.vdot(psi,psi).real),held_Ref_labels_NOT_physical_spinor_indices=True,
        full_word_projector=physical_frame,physical_compressed_G_excited_direction=residual,
        no_state_filter_SOURCEZ_norm_effect_retuning=True,no_fullH2_or_global_closure_claim=True,
        half_density_semantics='No scalar divergence is declared zero. A scalar times original Psi lies in W, so cancels from the orthogonal remainder exactly.',
        guard_sha256_at_read=hashlib.sha256(guard_bytes).hexdigest(),physical_Noether_reconstruction_error=pull['reconstruction_error'])


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-npz',type=Path,required=True);p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();start=time.monotonic()
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest);before=loaded_source_epoch();q,em,_=source_point(model,data)
    rows={name:[physical_remainder(model,data,name,q,em,h) for h in (2.**-18,2.**-19)] for name in ('source19','SOFT')}
    result=dict(scope='NEW physical embedding discriminator: full spatial G including field derivative + combined light/heavy/Ref Fock action outside ALL8 word rays, not only coherent tau',
        rows=rows,point_normal_norm=float(np.linalg.norm(q)),all55_original_EM_Gaussian=True,
        mathematical_implication='A positive point PH or projected residual excludes closure of the correlated Ref-word ray embedding even with arbitrary field coefficient functions. Zero does NOT prove global finite-mode closure.',
        selected_expansion='Explicit microscopic operator-family embedding or full108light-Fock interaction frame; preserve original SOURCE initial function',
        unpaid=['Encoded Root/fullauxGauss equality','Full H2/outside-Galerkin evolution','Actual Q/G/H/J simultaneous low-energy particle embedding and scattering','LEGAL/domain and original quantum-law parameters'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),loaded_sources=before,loaded_sources_stable=before==loaded_source_epoch(),program_wall_seconds=time.monotonic()-start)
    text=json.dumps(portable(result),sort_keys=True);args.output.write_text(text);os.chmod(args.output,0o600);print(text)

if __name__=='__main__':main()
