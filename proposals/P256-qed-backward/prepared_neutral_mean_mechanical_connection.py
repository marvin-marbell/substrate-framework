#!/usr/bin/env python3
"""Actual normal/Gauss mean blocks, mechanical Noether connection and SOURCE action."""
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
from scipy.linalg import solve
from original_charged_principal import portable
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_moving_matter_frame import mean_matter_generators
from prepared_neutral_shared_photon_state import load_shared_native_cache,loaded_source_epoch
from prepared_neutral_phi_word_channels import coefficient_channels
from prepared_neutral_full_field import BETA


def mechanical_mean(model: Any,data: dict[str,Any]) -> dict[str,Any]:
    n=model.chart.normal_dim;point=model.chart.evaluate(np.zeros(n));g=mean_matter_generators(model,data)
    guard_bytes=g['guard_path'].read_bytes();guard=json.loads(guard_bytes)
    pull=physical_chart_pullback(model,dict(point=point),np.asarray(guard['alpha']),np.asarray(guard['velocity']))
    inverse=point['cometric'];metric=point['metric'];normal=pull['normal'];vertical=pull['gauge']
    p_per_Omega=solve(model.chi*inverse[:n,:n],normal,assume_a='pos',check_finite=False)
    shifted_gauge=model.chi*(inverse[n:,:n]@p_per_Omega)
    mechanical=vertical-shifted_gauge
    alternative=solve(metric[n:,n:],metric[n:,:n]@normal+metric[n:,n:]@vertical,assume_a='pos',check_finite=False)
    gauge=model.gauge_operators(mechanical,'charged');source_rows={}
    sample=model.basis_at(np.zeros(n),np.zeros(55),point=point)
    for profile in ('source19','SOFT'):
        source=model.source_bank.seed_fibre.source_vectors_at(point['phi'],point['links'],sample['theta'],sample['a'],model.source_bank.profiles[profile],BETA)
        S=source['full108_plus_coefficients'];T=source['full108_minus_coefficients'];K=gauge['body'];R=gauge['reference']
        plus=K@S-S@R;minus=R@T-T@K
        relative=model.fibre.source_vectors_from_marks({1:plus.reshape(model.law.volume,4,4),-1:minus.reshape(4,model.law.volume,4).transpose(1,0,2)})['source_matrix']
        _,metadata=coefficient_channels(data,'charged',profile);Z=metadata['global_SOURCE_normalizer']
        source_rows[profile]=dict(plus=plus,minus=minus,relative_source_matrix=relative,original_source_matrix=source['source_matrix'],
            original_globalZ=Z,raw_relative_operator_norm=float(np.linalg.norm(relative)),relative_operator_norm_per_oldZ=float(np.linalg.norm(relative)/Z),
            no_SOURCE_derivative_or_packet_L_spin_term_claim=True)
    return dict(normal_dim=n,Gauss_dim=len(vertical),normal=normal,gauge=vertical,mechanical_gauge=mechanical,
        p_per_Omega=p_per_Omega,kinetic_boost_Gauss_correction=shifted_gauge,
        mechanical_by_metric=alternative,mechanical_metric_cometric_identity_error=float(np.linalg.norm(mechanical-alternative)),
        actual_cometric_NN=inverse[:n,:n],actual_cometric_GN=inverse[n:,:n],actual_cometric_GG=inverse[n:,n:],
        actual_metric_NN=metric[:n,:n],actual_metric_GN=metric[n:,:n],actual_metric_GG=metric[n:,n:],
        cometric_NN_identity_defect=float(np.linalg.norm(inverse[:n,:n]-np.eye(n))),cometric_GN_norm=float(np.linalg.norm(inverse[n:,:n])),
        normal_reference_Y0_error=float(np.linalg.norm(normal-g['Y0'])),normal_norm=float(np.linalg.norm(normal)),gauge_norm=float(np.linalg.norm(vertical)),
        mechanical_gauge_norm=float(np.linalg.norm(mechanical)),normal_boost_cancellation_error=float(np.linalg.norm(model.chi*inverse[:n,:n]@p_per_Omega-normal)),
        actual_boost_scalar_per_Omega_squared=-.5*float(normal@p_per_Omega),physical_mass_cometric_chi=model.chi,
        source_body_rows=source_rows,guard_sha256_at_actual_read=hashlib.sha256(guard_bytes).hexdigest(),
        formula='omega_mech=Yg−gInvGN solve(gInvNN,Yn)=solve(gGG,gGN Yn+gGG Yg); no identity or zero-cross assumption',
        IC_warning='Completion square uses formalbeta=Omega. ICpreservingVbeta usesbeta=Omega−Omega_at_start; leftover−Omega_at_start DY remains and heldoriginalstaticboost is counted once, not reset.',
        field_point='Literal original mean q0 and em0; no h(q) re-ground or SOURCE retuning',
        remaining=['qdependentmechanicalconnection/density/curvature','ActualpacketL/core-refscreening/intrinsicSO3spin','Fullfield/globalH/physicalROOT/LEGAL quantumlaw'])


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-npz',type=Path,required=True);p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arrays',type=Path,required=True);args=p.parse_args();start=time.monotonic()
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest);before=loaded_source_epoch();result=mechanical_mean(model,data)
    arrays={key:result.pop(key) for key in ('actual_cometric_NN','actual_cometric_GN','actual_cometric_GG','actual_metric_NN','actual_metric_GN','actual_metric_GG','p_per_Omega','normal','gauge','mechanical_gauge','mechanical_by_metric','kinetic_boost_Gauss_correction')}
    np.savez_compressed(args.arrays,**arrays);os.chmod(args.arrays,0o600)
    result.update(arrays=str(args.arrays),arrays_sha256=hashlib.sha256(args.arrays.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),loaded_sources=before,
        loaded_sources_stable=before==loaded_source_epoch(),program_wall_seconds=time.monotonic()-start)
    text=json.dumps(portable(result),sort_keys=True);args.output.write_text(text);os.chmod(args.output,0o600);print(text)

if __name__=='__main__':main()
