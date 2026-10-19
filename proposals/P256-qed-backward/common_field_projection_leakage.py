#!/usr/bin/env python3
"""Actual constant-Y0 normal-momentum leakage, not a full G/H time flow.

T_Y0=-i(Y0.grad+Y0.grad(log rho)/2) is a genuine SA normal component.
The common Gaussian and cutoff derivative survive; the Jhalf cancels.
Both leakage legs and the finite-Phi14 weak/strong adjoint defect are measured.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any,cast
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from scipy.linalg import solve
from native_closed_rotation_quantum import load_noether_export,physical_chart_pullback
from prepared_neutral_shared_photon_state import conditional_values,conditional_derivative,coupled_form,frozen_nodes,load_shared_native_cache
from raw_source_relative_noether_graph import source,matrix,derivative,raw_projection,norm,runtime_binding


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(base:np.ndarray,extra:np.ndarray,raw:np.ndarray)->np.ndarray:
    size=base.shape[1]*15
    out=np.zeros((base.shape[0],97),complex)
    out[:,:15]=base@raw[:size].reshape(base.shape[1],15)
    out[:,15:]=np.einsum('aen,e->an',extra,raw[size:],optimize=True)
    return out


def run(model:Any,data:dict[str,Any],exact:dict[str,Any],exact_arrays:dict[str,np.ndarray])->tuple[dict[str,Any],dict[str,np.ndarray]]:
    argv=data['manifest']['argv'];guard_path=Path(argv[argv.index('--noether-guard')+1])
    if sha(guard_path)!=data['manifest']['input_sha256']['noether_guard']:
        raise ValueError('Frozen physical guard changed')
    guard=json.loads(guard_path.read_bytes())
    mean=model.original.basis_at(np.zeros(2430),np.zeros(55))
    pull=physical_chart_pullback(model,mean,np.asarray(guard['alpha']),np.asarray(guard['velocity']))
    y0=pull['normal'];records=frozen_nodes(model);kernel=model.noether_bank.scalar_gram()
    coordinates={s:data[s+'_coordinates'] for s in ('neutral','charged')}
    grams={s:data[s+'_Gram'] for s in coordinates}
    initial=data['neutral_state'];neutral_raw=coordinates['neutral']@initial
    names=('source19','SOFT')
    charged_raw={n:coordinates['charged']@data[n+'_map']@initial for n in names}
    neutral_forward=np.zeros(214,complex);neutral_backward=np.zeros(214,complex)
    source_projection={n:np.zeros(1728,complex) for n in names}
    relative_projection={n:np.zeros(1728,complex) for n in names}
    charged_forward={n:np.zeros(1728,complex) for n in names}
    charged_backward={n:np.zeros(1728,complex) for n in names}
    strong_actual_source={n:np.zeros(1728,complex) for n in names}
    strong_charged_form=np.zeros_like(data['charged_raw_Gram'])
    charged_identity=model.fibre.overlap('charged')
    first_right={n:np.zeros(1728,complex) for n in names}
    cached=[]
    totals={n:dict(source_residual=0.,relative_full=0.,gaussian_adverse=0.,fd=0.) for n in names}
    neutral_full=0.;normal_gaussian_only=0.;maximum_source_embedding=0.
    for record in records:
        sample=record['sample'];weight=sample['importance_weight']/len(records)
        base,extra=conditional_values(model,sample)
        db,de=conditional_derivative(model,sample,base,extra,y0,np.zeros(3))
        density=float(model.chart.directional_jet(sample['q'],y0,sample['point'])['log_density_first'])
        # Restore +half divergence of the constant field in the actual rho dq.
        # This cancels the original -half log-density in full wavefunction jets.
        db={s:db[s]+.5*density*base[s] for s in base}
        de={s:de[s]+.5*density*extra[s] for s in extra}
        strong_charged_form+=-1j*weight*coupled_form(base['charged'],extra['charged'],db['charged'],de['charged'],charged_identity,kernel)
        psi=evaluate(base['neutral'],extra['neutral'],neutral_raw)
        dpsi=evaluate(db['neutral'],de['neutral'],neutral_raw)
        gpsi=-1j*dpsi
        gaussian=1j*float(sample['q']@y0)/model.sigma**2*psi
        normal_gaussian_only+=weight*norm(gaussian,kernel)
        neutral_full+=weight*norm(gpsi,kernel)
        neutral_forward+=weight*raw_projection(base['neutral'],extra['neutral'],dpsi,kernel)
        neutral_backward+=weight*raw_projection(db['neutral'],de['neutral'],psi,kernel)
        saved=dict(sample=sample,weight=weight,base_neutral=base['neutral'],extra_neutral=extra['neutral'],gpsi=gpsi,sources={})
        for name,profile in model.source_bank.profiles.items():
            marks=source(model,sample,profile);b=matrix(model,marks['plus'],marks['minus'])
            dm,error=derivative(model,sample,profile,dq=y0)
            db_source=matrix(model,dm['plus'],dm['minus'])
            actual=b@psi;relative=-1j*(db_source@psi)
            actual_derivative=db_source@psi+b@dpsi
            represented=evaluate(base['charged'],extra['charged'],charged_raw[name])
            represented_derivative=evaluate(db['charged'],de['charged'],charged_raw[name])
            q_source=actual-represented
            source_projection[name]+=weight*raw_projection(base['charged'],extra['charged'],actual,kernel)
            relative_projection[name]+=weight*raw_projection(base['charged'],extra['charged'],relative,kernel)
            charged_forward[name]+=weight*raw_projection(base['charged'],extra['charged'],represented_derivative,kernel)
            charged_backward[name]+=weight*raw_projection(db['charged'],de['charged'],represented,kernel)
            strong_actual_source[name]+=weight*raw_projection(base['charged'],extra['charged'],-1j*actual_derivative,kernel)
            first_right[name]+=weight*raw_projection(base['charged'],extra['charged'],b@gpsi,kernel)
            t=totals[name]
            t['relative_full']+=weight*norm(relative,kernel)
            t['gaussian_adverse']+=weight*norm(b@gaussian,kernel)
            fd=matrix(model,-1j*error['plus'],-1j*error['minus'])@psi
            t['fd']+=weight*norm(fd,kernel)
            maximum_source_embedding=max(maximum_source_embedding,float(np.linalg.norm(q_source)/(1e-100+np.linalg.norm(actual))))
            saved['sources'][name]=b
        cached.append(saved)
    def project(sector:str,raw:np.ndarray)->np.ndarray:
        return solve(grams[sector],coordinates[sector].conj().T@raw,assume_a='her')
    p_neutral=project('neutral',-1j*neutral_forward)
    weak_neutral=project('neutral',-.5j*(neutral_forward-neutral_backward))
    p_neutral_raw=coordinates['neutral']@p_neutral
    source_projected={n:project('charged',source_projection[n]) for n in names}
    source_projected_raw={n:coordinates['charged']@source_projected[n] for n in names}
    q_neutral_norm=0.;right_leg={n:np.zeros(1728,complex) for n in names}
    source_map_function_errors={n:0. for n in names}
    for saved in cached:
        base,extra=conditional_values(model,saved['sample'])
        projected=evaluate(saved['base_neutral'],saved['extra_neutral'],p_neutral_raw)
        outside=saved['gpsi']-projected
        q_neutral_norm+=saved['weight']*norm(outside,kernel)
        initial_here=evaluate(saved['base_neutral'],saved['extra_neutral'],neutral_raw)
        for name in names:
            b=saved['sources'][name]
            q_source=b@initial_here-evaluate(base['charged'],extra['charged'],source_projected_raw[name])
            totals[name]['source_residual']+=saved['weight']*norm(q_source,kernel)
            right_leg[name]+=saved['weight']*raw_projection(base['charged'],extra['charged'],b@outside,kernel)
            represented=evaluate(base['charged'],extra['charged'],coordinates['charged']@data[name+'_map']@p_neutral)
            source_map_function_errors[name]=max(source_map_function_errors[name],float(np.linalg.norm(represented-b@projected)/(1e-100+np.linalg.norm(b@projected))))
    arrays=dict(Y0=y0,neutral_T_strong_projection=p_neutral,neutral_T_weak_action=weak_neutral)
    result={}
    for name in names:
        z=exact['profiles'][name]['global_initial_SOURCE_Z'];gram=grams['charged'];s=data[name+'_map']
        def square(value:np.ndarray)->float:
            return float(np.vdot(value,gram@value).real)/z**2
        pc_relative=project('charged',relative_projection[name])
        weak_charged=project('charged',-.5j*(charged_forward[name]-charged_backward[name]))
        strong_projected_source=project('charged',strong_charged_form@source_projected_raw[name])
        a=project('charged',strong_actual_source[name])-strong_projected_source
        b=project('charged',right_leg[name])
        discrepancy=pc_relative-(weak_charged-s@weak_neutral)
        adjoint=strong_projected_source-weak_charged-s@(p_neutral-weak_neutral)
        coherent=a-b
        reconstruction=discrepancy-coherent-adjoint
        full_residual=exact_arrays[name+'_PC_full_minus_new_exactEM']*z
        remaining=full_residual-discrepancy
        rhs_direct=project('charged',first_right[name])-s@p_neutral
        actual_source_projection=source_projected[name]
        result[name]=dict(global_initial_SOURCE_Z=z,
            SOURCE_inclusion_full_function_residual_squared=totals[name]['source_residual']/z**2,
            SOURCE_inclusion_coefficient_projection_error_squared=square(actual_source_projection-s@initial),
            SOURCE_map_on_projected_neutral_function_relative_error=source_map_function_errors[name],
            first_PC_T_QC_B_PN_leg_squared=square(a),second_PC_B_QN_T_PN_leg_squared=square(b),
            first_minus_second_coherent_squared=square(coherent),
            first_second_coherent_cross=2*float(np.vdot(a,gram@b).real)/z**2,
            second_leg_independent_contraction_error_squared=square(b-rhs_direct),
            RAW_PC_relative_T_squared=square(pc_relative),RAW_full_relative_T_squared=totals[name]['relative_full']/z**2,
            separate_RAW_QC_relative_T_squared=totals[name]['relative_full']/z**2-square(pc_relative),
            PC_RAW_minus_SAweak_T_Galerkin_squared=square(discrepancy),
            conditional_Phi14_adjoint_IBP_correction_squared=square(adjoint),
            actual_leg_plus_adjoint_reconstruction_error_squared=square(reconstruction),
            frozen_FULL_exactEM_difference_squared=square(full_residual),
            frozen_FULL_difference_after_removing_constant_normal_component_squared=square(remaining),
            coherent_FULL_constant_normal_cross=2*float(np.vdot(full_residual,gram@discrepancy).real)/z**2,
            missing_common_Gaussian_RIGHT_adverse_norm=(totals[name]['gaussian_adverse']/z**2)**.5,
            SOURCE_FD_refinement_difference_norm=(totals[name]['fd']/z**2)**.5)
        if square(reconstruction)>1e-12*max(1.,square(discrepancy)):
            raise ValueError('Independent leakage plus finite-Phi adjoint reconstruction fails')
        for suffix,value in (('PC_relative_T',pc_relative),('first_leakage_leg_T',a),('second_leakage_leg_T',b),('weak_T_discrepancy',discrepancy),('Phi14_adjoint_correction_T',adjoint),('FULL_difference_after_T',remaining)):
            arrays[name+'_'+suffix]=value/z
    return dict(profiles=result,Y0_norm=float(np.linalg.norm(y0)),mean_RAW_reconstruction_error=pull['reconstruction_error'],
        full_neutral_T_norm_squared=neutral_full,projected_neutral_T_norm_squared=float(np.vdot(p_neutral,grams['neutral']@p_neutral).real),
        unrepresented_QN_T_norm_squared=q_neutral_norm,common_Gaussian_derivative_only_norm_squared=normal_gaussian_only,
        maximum_SOURCE_family_representation_relative_error=maximum_source_embedding,
        SOURCE_QC_projector='actual Gram-solved PC(B Xi) reconstructed as full97 coefficient functions; not assigned zero from family membership',
        physical_operator='constant normal component T_Y0=-i(Y0.grad+halfY0gradlogrho); actualY0 fromfullRAWmean primitive',
        density_semantics='fullGaussian andcutoff derivative retained; Jhalf canceled byactualrho divergence, not dropped',
        identity='PC R_T-Gal_Tweak=(PC T QC B PN-PC B QN T PN)+conditionalPhi14adjoint/IBPcorrection',
        matter_semantics='light/heavy/Ref fullheldstate factors arepassive forpurefield T, notassumedpassive forfullG; noPH-onlycause claim',
        norm_scope='nativeconditionalPhi14/harmonic exactscalar97EM norm withnormalFD; NOTwholefieldL2/legalSOFT/fullG/fullH-OmegaG trajectory',
        no_full_H_G_preparation_or_loop_recompile=True,no_SOURCE_state_normalization_reset=True,campaign_open=True),arrays


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cache-npz','cache-manifest','exact-G-npz','exact-G-receipt','output','output-npz','runtime-archive'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    exact_arrays,exact=load_noether_export(args.exact_G_npz,args.exact_G_receipt)
    if exact['input_sha256']['cache_npz_sha256']!=sha(args.cache_npz) or exact['input_sha256']['cache_manifest_sha256']!=sha(args.cache_manifest):
        raise ValueError('Common field graph requires the SAME immutable H/state/Gram cache')
    binding=runtime_binding(args.runtime_archive)
    science,arrays=run(model,data,exact,exact_arrays)
    np.savez_compressed(args.output_npz,**cast(Any,arrays))
    receipt=dict(plan='issue-222-fullfield-closedrotation-continuation-20261006',revision='668c7d8ea2ec9c71c964e9f16d3c75e8cd879e10433e72fbcad0c3b89c7da417',
        work_item='soft-spatial-closed-rotation',science=science,source_sha256=sha(Path(__file__)),runtime_binding=binding,
        cache_npz_sha256=sha(args.cache_npz),cache_manifest_sha256=sha(args.cache_manifest),
        exact_G_npz_sha256=sha(args.exact_G_npz),exact_G_receipt_sha256=sha(args.exact_G_receipt),
        export_npz_sha256=sha(args.output_npz),argv=sys.argv,wall_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':
    main()
