#!/usr/bin/env python3
"""Original SOURCE-relative physical Noether CAR image and explicit enlarged embedding."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from original_charged_principal import portable
from original_total_angular_transfer import scalar_orbital
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_phi_word_channels import coefficient_channels
from prepared_neutral_shared_photon_state import load_shared_native_cache,loaded_source_epoch
from prepared_neutral_quartet_functions import full_slater_relative_spin
from prepared_neutral_full_field import BETA


def mark_matrices(source: dict[str,Any]) -> tuple[np.ndarray,np.ndarray]:
    return np.asarray(source['full108_plus_coefficients']),np.asarray(source['full108_minus_coefficients'])


def relative_source(model: Any,sample: dict[str,Any],profile: str,word: np.ndarray,K: np.ndarray,R: np.ndarray,
                    normal: np.ndarray,dem: np.ndarray,h: float) -> dict[str,Any]:
    """Differentiate literal marks; common neutral/state factors cancel algebraically."""
    fibre=model.source_bank.seed_fibre;eta=model.source_bank.profiles[profile]
    def at(q: np.ndarray,y: np.ndarray) -> dict[str,Any]:
        point=model.chart.embedding(q);a,theta=model.electromagnetic_field(y)
        return fibre.source_vectors_at(point['phi'],point['links'],theta,a,eta,word)
    original=at(sample['q'],sample['em']);S,T=mark_matrices(original)
    step=h/max(1.,float(np.linalg.norm(normal)),float(np.linalg.norm(dem)))
    Sp,Tp=mark_matrices(at(sample['q']+step*normal,sample['em']+step*dem))
    Sm,Tm=mark_matrices(at(sample['q']-step*normal,sample['em']-step*dem))
    dS,dT=(Sp-Sm)/(2*step),(Tp-Tm)/(2*step)
    plus=-1j*dS+K@S-S@R;minus=-1j*dT+R@T-T@K
    marks={1:plus.reshape(model.law.volume,4,4),-1:minus.reshape(4,model.law.volume,4).transpose(1,0,2)}
    relative=fibre.source_vectors_from_marks(marks)['source_matrix']
    # Independent projected operator commutator WITH true neutral RIGHT-PH correction.
    control=full_slater_relative_spin(fibre,original,K)
    qC=fibre.onebody(K,'charged')+fibre.reference(R,'charged')
    qN=fibre.onebody(K,'neutral')+fibre.reference(R,'neutral')
    projected=qC@original['source_matrix']-original['source_matrix']@qN
    comm_marks={1:(K@S-S@R).reshape(model.law.volume,4,4),-1:(R@T-T@K).reshape(4,model.law.volume,4).transpose(1,0,2)}
    comm=fibre.source_vectors_from_marks(comm_marks)['source_matrix']
    right=np.zeros_like(original['source_matrix'])
    right[fibre.charged_slices[1],0]=(-fibre.empty.conj().T@K@fibre.projectors[1]@S).T.reshape(-1)
    right[fibre.charged_slices[-1],1]=(T@(np.eye(fibre.n)-fibre.projectors[-1])@K@fibre.occupied[-1]).reshape(-1)
    return dict(relative_matrix=relative,original_source_matrix=original['source_matrix'],
        derivative_marks=dict(plus=dS,minus=dT),CAR_commutator_marks=dict(plus=K@S-S@R,minus=R@T-T@K),
        actual_neutral_RIGHT_PH=right,projected_before_RIGHT_wrong_error=float(np.linalg.norm(projected-comm)),
        corrected_projected_commutator_error=float(np.linalg.norm(projected-right-comm)),
        independent_light_CAR_identity_error=control['exact_CAR_error'],
        no_heavy_common_sea_divergence_or_neutral_derivative_in_relative_operator=True,
        field_step=step,all55_field_directions=True)

def construct(model: Any,data: dict[str,Any],profile: str,h: float) -> dict[str,Any]:
    q,em,g=source_point(model,data);sample=model.basis_at(q,em);guard_bytes=g['guard_path'].read_bytes();guard=json.loads(guard_bytes)
    pull=physical_chart_pullback(model,sample,np.asarray(guard['alpha']),np.asarray(guard['velocity']))
    noether=model.noether_bank.evaluate(em[:52]);gauge=model.gauge_operators(pull['gauge'],'charged')
    K=gauge['body']+scalar_orbital(model.law,sample['point']['links'],sample['a'],np.asarray(guard['velocity']))+np.diag(np.repeat(noether['lambda_C'],4))
    R=gauge['reference'];dem=np.r_[noether['flow'],noether['harmonic']]
    C,metadata=coefficient_channels(data,'charged',profile);W=sample['charged']@C
    neutral=sample['neutral']@data['neutral_state'];Z=metadata['global_SOURCE_normalizer']
    coherent=relative_source(model,sample,profile,BETA,K,R,pull['normal'],dem,h)
    initial_error=float(np.linalg.norm(coherent['original_source_matrix']@neutral/Z-W[:,0]))
    u,s,_=np.linalg.svd(W,full_matrices=False);tol=np.finfo(float).eps*max(W.shape)*s[0];frame=u[:,s>tol]
    image=coherent['relative_matrix']@neutral/Z;outside=image-frame@(frame.conj().T@image)
    # All eight operator-generated images, using literal unit words rather than assigned spin matrices.
    generated=[];unit_rows=[]
    fibre=model.source_bank.seed_fibre
    for word in range(4):
        row=relative_source(model,sample,profile,np.eye(4,dtype=complex)[word],K,R,pull['normal'],dem,h)
        unit_rows.append(row)
        for charge in (1,-1):
            bank=0 if charge==1 else 1
            generated.append(row['relative_matrix'][:,bank]*neutral[bank]/Z)
    operators=np.column_stack(generated);extension=np.column_stack((W,operators))
    se=np.linalg.svd(extension,compute_uv=False);extension_rank=int(np.sum(se>np.finfo(float).eps*max(extension.shape)*se[0]))
    coefficients=np.zeros(extension.shape[1],complex);coefficients[0]=1
    return dict(profile=profile,channel_metadata=metadata,original_function_embedding_error=initial_error,
        physical_word_rank=frame.shape[1],relative_image_norm_squared=float(np.vdot(image,image).real),
        relative_image_outside_FULL8_word_rays_squared=float(np.vdot(outside,outside).real),
        actual_neutral_RIGHT_correction_norm=float(np.linalg.norm(coherent['actual_neutral_RIGHT_PH'])),
        projected_without_RIGHT_wrong_error=coherent['projected_before_RIGHT_wrong_error'],
        corrected_source_relative_CAR_error=coherent['corrected_projected_commutator_error'],
        operator_generated_extension=extension,extension_Gram=extension.conj().T@extension,extension_physical_rank=extension_rank,
        original_initial_coefficients_in_extension=coefficients,extension_initial_function_error=float(np.linalg.norm(extension@coefficients-W[:,0])),
        actual_SOURCE_relative_image=image,relative_image_outside_FULLword_rays=outside,
        generated_unit_word_images=operators,generated_order=['word%d charge%+d'%(word,charge) for word in range(4) for charge in (1,-1)],
        literal_coherent_row=coherent,unit_word_rows=unit_rows,
        common_heavy_neutral_sea_divergence_cancelled_algebraically_before_relative_norm=True,
        full_state_G_divergence_sea_heavy_NOT_removed=True,Refword_labels_NOT_physical_spin_indices=True,
        physical_light_modes=fibre.n,reference_modes=4,held_frozen_banks_original_globalZ_once=True,
        guard_sha256_at_actual_read=hashlib.sha256(guard_bytes).hexdigest(),point_normal_norm=float(np.linalg.norm(q)))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-npz',type=Path,required=True);p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();start=time.monotonic()
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest);before=loaded_source_epoch()
    rows={profile:[construct(model,data,profile,h) for h in (2.**-18,2.**-19)] for profile in ('source19','SOFT')}
    result=dict(scope='NEW SOURCE-relative full physicalG operator-image embedding onunchangedPsi0; notabsoluteGstateunderfit nor invariantH/Q/G/Jsector',rows=rows,
        identity='r_plus=-iDY55S+KactualS-SRactual; r_minus=-iDY55T+RactualT-TKactual; true neutral RIGHT before compression',
        debts=['Fullfield/globaloperator-generated Gram and H/G/connection/evolution','Originalcanonicalcurrent/contact and incomingphoton extension','EncodedRoot/fullauxGauss, fullH2/remainder, physicalspin and low-energy/scattering/legal','Quantumlaw andparameterderivation'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        loaded_sources=before,loaded_sources_stable=before==loaded_source_epoch(),program_wall_seconds=time.monotonic()-start)
    text=json.dumps(portable(result),sort_keys=True);args.output.write_text(text);os.chmod(args.output,0o600);print(text)

if __name__=='__main__':main()
