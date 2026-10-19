#!/usr/bin/env python3
"""NEW exact52EM weak Noether forms on the SAME frozen H/Xi/214x1728 basis.

Not a preparation/H rebuild, oldG or loop confirmation run. The full physical
normal/Gauss/Ref/Jhalf assembly is retained; only scalar EM moments change.
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
from exact_em_noether_kernels import ExactNoetherKernels
from native_closed_rotation_quantum import physical_chart_pullback,load_noether_export
from original_total_angular_transfer import scalar_orbital
from prepared_neutral_shared_photon_state import conditional_derivative,conditional_values,coupled_form,frozen_nodes,load_shared_native_cache
from raw_source_relative_noether_graph import runtime_binding
PLAN='issue-222-fullfield-closedrotation-continuation-20261006'
REVISION='668c7d8ea2ec9c71c964e9f16d3c75e8cd879e10433e72fbcad0c3b89c7da417'


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_exact_noether(model: Any, data: dict[str, Any],
                          em: ExactNoetherKernels) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Export WEAK Hermitian forms, never Gram^-1 operators under the same name."""
    manifest = data['manifest']
    guard_path = Path(manifest['argv'][manifest['argv'].index('--noether-guard')+1])
    guard_bytes = guard_path.read_bytes()
    if hashlib.sha256(guard_bytes).hexdigest()!=manifest['input_sha256']['noether_guard']:
        raise ValueError('Shared physical Noether guard differs from the frozen A input')
    guard = json.loads(guard_bytes)
    alpha,velocity = np.asarray(guard['alpha']),np.asarray(guard['velocity'])
    if float(model.c)!=.125 or float(model.ue)!=2.**-20:
        raise ValueError('The SAME immutable A17 native c/UE is required')
    kernels = em.noether_kernels()
    gramkernel = model.noether_bank.scalar_gram()
    records = frozen_nodes(model)
    sizes = {s:data[s+'_raw_Gram'].shape[0] for s in ('neutral','charged')}
    forms = {s:np.zeros((d,d),complex) for s,d in sizes.items()}
    errors = []
    for record in records:
        sample = record['sample']
        base,extra = conditional_values(model,sample)
        pull = physical_chart_pullback(model,sample,alpha,velocity)
        errors.append(pull['reconstruction_error'])
        db,de = conditional_derivative(model,sample,base,extra,pull['normal'],np.zeros(3))
        identity = {s:model.fibre.overlap(s) for s in sizes}
        node = {}
        heavy_orbit = scalar_orbital(model.heavy_law,sample['point']['links'],np.zeros((27,3)),velocity)
        heavy_scalar = float(np.trace(model.heavy_projector@heavy_orbit).real)
        for sector in sizes:
            gauge = model.gauge_operators(pull['gauge'],sector)['first']
            dbase = db[sector]+1j*gauge@base[sector]
            dextra = de[sector]+1j*np.einsum('ab,ben->aen',gauge,extra[sector],optimize=True)
            cross = coupled_form(base[sector],extra[sector],dbase,dextra,identity[sector],gramkernel)
            value = -.5j*(cross-cross.conj().T)
            value += heavy_scalar*coupled_form(base[sector],extra[sector],base[sector],extra[sector],identity[sector],gramkernel)
            value += coupled_form(base[sector],extra[sector],base[sector],extra[sector],identity[sector],kernels['coexact'])
            node[sector] = value
        # Harmonic velocity is a genuine operator, not meanY_EM(0)=0.
        for axis in range(3):
            db,de = conditional_derivative(model,sample,base,extra,np.zeros(2430),np.eye(3)[axis])
            for sector in sizes:
                cross = coupled_form(base[sector],extra[sector],db[sector],de[sector],identity[sector],kernels['harmonic'][axis])
                node[sector] += -.5j*(cross-cross.conj().T)
        # rho=lambda_C light density plus SAME Ref/background; lambda_R=0.
        for site in range(model.law.volume):
            local = np.zeros((108,108),complex)
            local[4*site:4*site+4,4*site:4*site+4] = np.eye(4)
            for sector in sizes:
                rho = model.fibre.onebody(local,sector)-2*identity[sector]
                node[sector] += coupled_form(base[sector],extra[sector],base[sector],extra[sector],rho,kernels['coulomb'][site])
        # Scalar orbital only, all original Peierls phases BEFORE projection.
        links,theta = sample['point']['links'],sample['theta']
        for x in range(model.law.volume):
            for axis in range(3):
                target = int(model.law.target[x,axis])
                scalar_velocity = (velocity[x,axis]+velocity[target,axis])/(4j*model.law.band.a)
                if not scalar_velocity:
                    continue
                body = np.zeros((108,108),complex)
                body[4*x:4*x+4,4*target:4*target+4] = scalar_velocity*links[x,axis]*np.exp(1j*theta[axis]/model.law.size)
                displacement = model.em['legs'][3*x+axis]
                kernel = em.weyl(displacement)
                for sector in sizes:
                    term = coupled_form(base[sector],extra[sector],base[sector],extra[sector],model.fibre.onebody(body,sector),kernel)
                    node[sector] += term+term.conj().T
        for sector in sizes:
            forms[sector] += sample['importance_weight']/len(records)*node[sector]
    out = {}
    hermiticity = {}
    for sector in sizes:
        coord = data[sector+'_coordinates']
        value = coord.conj().T@forms[sector]@coord
        hermiticity[sector] = float(np.linalg.norm(value-value.conj().T))
        if hermiticity[sector]>1e-7*(1+np.linalg.norm(value)):
            raise ValueError('Physical weak Noether form lost Hermiticity')
        out[sector+'_G'] = (value+value.conj().T)/2
    profiles = {}
    for name,profile in model.source_bank.profiles.items():
        weights = np.abs(np.asarray(profile))**2
        norm = float(np.sum(weights))
        profiles[name] = dict(raw_lattice_norm_squared=norm,
            raw_profile_weight_outside_alpha1=float(np.sum(weights[alpha!=1])/norm),
            scope='raw lattice profile geometry only; NOT coupled SOURCE or full physical L2 mass')
    return out,dict(generator_units='weak_form',Phi_nodes=len(records),
        max_RAW_full_J_reconstruction_error=max(errors),Hermiticity=hermiticity,
        gaussian_kernels={k:v for k,v in kernels.items() if k not in ('coexact','harmonic','coulomb')},
        native_raw_profiles=profiles,exact_neutral_RIGHT=True,derived_Gauss_body_Ref_once=True,Coulomb_count=1,
        no_assigned_intrinsic_body_spin=True,no_reference_position_motion=True,
        scope='actual physical RAW weak generator in SAME native coupled basis; exact all52 EM; finite Phi14/normalFD and fullSlaterPH/fullAux/fieldfunction projection debt remain')


def compare(data:dict[str,Any],forms:dict[str,np.ndarray],raw:dict[str,Any],vectors:Any)->tuple[dict[str,Any],dict[str,np.ndarray]]:
    observations,arrays={},{}
    for name in ('source19','SOFT'):
        source=data[name+'_map']; psi=data['neutral_state']
        z=raw['science']['profiles'][name]['global_initial_SOURCE_Z']
        new=(solve(data['charged_Gram'],forms['charged_G']@source@psi,assume_a='her')
             -source@solve(data['neutral_Gram'],forms['neutral_G']@psi,assume_a='her'))/z
        old=vectors[name+'_cached_Galerkin_relative']; pc=vectors[name+'_projected_full']
        boundary=vectors[name+'_projected_PH']; gram=data['charged_Gram']
        def norm(value:np.ndarray)->float:
            return float(np.vdot(value,gram@value).real)
        old_difference,new_difference=pc-old,pc-new
        delta=new-old
        cross=2*float(np.vdot(old_difference,gram@delta).real)
        expected=norm(old_difference)+norm(delta)-cross
        if abs(expected-norm(new_difference))>1e-10*max(1.,abs(expected)):
            raise ValueError('Coherent old/new conditional comparison identity failed')
        observations[name]=dict(global_initial_SOURCE_Z=z,
            raw_full_relative_norm_squared=raw['science']['profiles'][name]['raw_relative_graph']['full_norm_squared'],
            raw_PC_full_norm_squared=norm(pc),raw_QC_full_norm_squared=raw['science']['profiles'][name]['unprojected_QC_full_norm_squared'],
            old_Galerkin_relative_norm_squared=norm(old),new_exactEM_Galerkin_relative_norm_squared=norm(new),
            PC_full_minus_old_Galerkin_norm_squared=norm(old_difference),
            PC_full_minus_new_exactEM_Galerkin_norm_squared=norm(new_difference),
            exactEM_minus_old_Galerkin_relative_norm_squared=norm(delta),
            coherent_old_difference_exactEM_change_cross=cross,
            new_difference_after_SOURCE_on_neutral_QPH_boundary_squared=norm(new_difference-boundary),
            source_geometry_unchanged=raw['science']['profiles'][name]['SOURCE_geometry'],
            norm_scope='same conditioned charged_Gram/frozen14Phi/harmonic atoms; ONE initial SOURCE Z, not state/projector reset')
        arrays[name+'_new_exactEM_relative']=new
        arrays[name+'_PC_full_minus_new_exactEM']=new_difference
        arrays[name+'_exactEM_minus_old_relative']=delta
    return observations,arrays


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cache-npz','cache-manifest','generator-npz','generator-receipt','raw-graph','raw-graph-npz','output','output-npz','runtime-archive'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    old,old_meta=load_noether_export(args.generator_npz,args.generator_receipt)
    raw=json.loads(args.raw_graph.read_bytes())
    bindings=dict(cache_npz_sha256=sha(args.cache_npz),cache_manifest_sha256=sha(args.cache_manifest),
                  generator_npz_sha256=sha(args.generator_npz),generator_receipt_sha256=sha(args.generator_receipt))
    for key,value in bindings.items():
        if raw[key]!=value:
            raise ValueError('RAW graph and exactEM G require identical sealed inputs: '+key)
    if raw['output_npz_sha256']!=sha(args.raw_graph_npz):
        raise ValueError('Frozen RAW graph coefficient export changed')
    binding=runtime_binding(args.runtime_archive)
    kernel_start=time.perf_counter();em=ExactNoetherKernels(model)
    kernel_seconds=time.perf_counter()-kernel_start
    form_start=time.perf_counter();forms,metadata=compile_exact_noether(model,data,em)
    form_seconds=time.perf_counter()-form_start
    with np.load(args.raw_graph_npz,allow_pickle=False) as vectors:
        observations,arrays=compare(data,forms,raw,vectors)
    arrays.update(forms)
    arrays.update(scalar_coexact=em.coexact,scalar_harmonic=em.harmonic,scalar_coulomb=em.coulomb)
    arrays['orbital_Weyl_displacements']=np.asarray([np.frombuffer(k,dtype=float) for k in em.weyl_cache])
    arrays['orbital_Weyl_kernels']=np.asarray(list(em.weyl_cache.values()))
    np.savez_compressed(args.output_npz,**cast(Any,arrays))
    receipt=dict(plan=PLAN,revision=REVISION,work_item='soft-spatial-closed-rotation',
        input_sha256=bindings,raw_graph_sha256=sha(args.raw_graph),raw_graph_npz_sha256=sha(args.raw_graph_npz),
        source_sha256=sha(Path(__file__)),kernel_source_sha256=sha(Path(__file__).with_name('exact_em_noether_kernels.py')),
        runtime_binding=binding,new_generator=metadata,profiles=observations,
        new_weak_form_differences={s:float(np.linalg.norm(forms[s+'_G']-old[s+'_G'])) for s in ('neutral','charged')},
        exactEM_kernel_seconds=kernel_seconds,new_weak_form_seconds=form_seconds,
        orbital_exact_Weyl_kernel_count=len(em.weyl_cache),export_npz_sha256=sha(args.output_npz),
        no_H_preparation_refilter_or_old_loop_rerun=True,unchanged_H_state_Gram_basis=True,
        physical_operator_unchanged=True,no_Ward_commutator_enforcement_or_spinfit=True,
        fixed_normal_Gauss_Ref_halfmeasure_Phi14_FD_assembly=True,
        old_generated_G_MC_replaced='Noe seed222712 plus orbital generated-Weyl H-MC seed222710; frozen old weak forms retained separately',
        leakage_identity='P_C R_full-R_Gal=P_C G Q_C B P_N-P_C B Q_N G P_N; QC R_full separately retained',
        scope='NEW exactall52EM conditional G discriminator, NOT a fullchangingpulse, ground, legalSOFT or wholefield continuum proof',
        campaign_open=True,argv=sys.argv,wall_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
