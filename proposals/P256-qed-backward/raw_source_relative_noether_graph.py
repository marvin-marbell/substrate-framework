#!/usr/bin/env python3
"""SAME-state full108 RAW SOURCE-relative graph, not another H/G/loop run.

Exact conditional full52 Gaussian moments, frozen14 Phi/theta atoms, literal
Source marks and CAR. Finite field quadrature/central-FD debt is explicit.

The ``light_PH_relative`` fields measure SOURCE-on-neutral-QPH projection-boundary
tensors, not surviving multipair PH of the full CAR relative commutator.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import marshal
import math
import os
import sys
import time
import types
import zipfile
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import solve
from native_closed_rotation_quantum import load_noether_export, physical_chart_pullback
from original_total_angular_transfer import scalar_orbital
from prepared_neutral_shared_photon_state import conditional_values, frozen_nodes, load_shared_native_cache
from prepared_neutral_noether_gaussian_partners import FiniteGaussianFunction, function_kernel, high_precision_kernel
from prepared_neutral_weyl_hard_fock import GaussianFunction, ProjectionPolynomial
from radial_band_noether import BETA


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_binding(archive: Path) -> dict[str,Any]:
    """Archive source AND actual in-memory project function code objects."""
    root = Path(__file__).resolve().parent
    sources, functions = {}, {}
    def nested(code: types.CodeType) -> dict[str,types.CodeType]:
        out = {code.co_qualname:code}
        for value in code.co_consts:
            if isinstance(value,types.CodeType):
                out.update(nested(value))
        return out
    def equal_code(left: types.CodeType,right: types.CodeType) -> bool:
        attributes = ('co_argcount','co_posonlyargcount','co_kwonlyargcount',
            'co_nlocals','co_stacksize','co_flags','co_code','co_names',
            'co_varnames','co_filename','co_name','co_qualname','co_firstlineno',
            'co_linetable','co_exceptiontable','co_freevars','co_cellvars')
        if any(getattr(left,key)!=getattr(right,key) for key in attributes):
            return False
        if len(left.co_consts)!=len(right.co_consts):
            return False
        for a,b in zip(left.co_consts,right.co_consts):
            if isinstance(a,types.CodeType) or isinstance(b,types.CodeType):
                if not isinstance(a,types.CodeType) or not isinstance(b,types.CodeType) or not equal_code(a,b):
                    return False
            elif a!=b:
                return False
        return True
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as packed:
        for name,module in tuple(sys.modules.items()):
            value = getattr(module,'__file__',None)
            if not value:
                continue
            path = Path(value).resolve()
            if path.parent!=root or path.suffix!='.py':
                continue
            body = path.read_bytes()
            sources[str(path)] = hashlib.sha256(body).hexdigest()
            packed.writestr('sources/'+path.name,body)
            expected = nested(compile(body,str(path),'exec',dont_inherit=True))
            candidates = []
            for obj in vars(module).values():
                if inspect.isfunction(obj) and obj.__code__.co_filename==str(path):
                    candidates.append(obj)
                elif inspect.isclass(obj) and obj.__module__==name:
                    for member in vars(obj).values():
                        if isinstance(member,(staticmethod,classmethod)):
                            member = member.__func__
                        if inspect.isfunction(member) and member.__code__.co_filename==str(path):
                            candidates.append(member)
            for obj in candidates:
                code = obj.__code__
                raw = marshal.dumps(code)
                key = name+':'+code.co_qualname
                if key in functions:
                    continue
                original = expected.get(code.co_qualname)
                matches = original is not None and equal_code(code,original)
                functions[key] = dict(code_sha256=hashlib.sha256(raw).hexdigest(),
                    source_compiled_code_equal=matches,source=str(path))
                packed.writestr('code/'+hashlib.sha256(key.encode()).hexdigest()+'.marshal',raw)
    mismatches = [key for key,row in functions.items() if not row['source_compiled_code_equal']]
    if mismatches:
        raise ValueError('Actual imported code/source mismatch: '+repr(mismatches))
    return dict(sources=sources,actual_function_code=functions,archive=str(archive),
        archive_sha256=sha(archive),all_actual_callable_code_matches_archived_source=True,
        old_B_SCI_getter_import_NOT_retroactively_observed=True)


def library(model: Any, velocity: np.ndarray) -> tuple[np.ndarray,list[tuple[int,int,int,int]],int]:
    """97 original scalar waves plus literal active orbital Weyl waves."""
    bank = model.noether_bank
    base = bank.gaussian_functions()
    gram = bank.scalar_gram()
    bonds, phases = [], []
    for x in range(model.law.volume):
        for axis in range(3):
            target = int(model.law.target[x,axis])
            if velocity[x,axis]+velocity[target,axis]==0:
                continue
            start = len(base)+len(phases)
            bonds.append((x,axis,target,start))
            k = model.em['legs'][3*x+axis]
            phases.extend((k,-k))
    k = np.asarray(phases)
    size = len(base)+len(k)
    out = np.zeros((size,size),complex)
    out[:97,:97] = gram
    for index,phase in enumerate(k):
        wave = FiniteGaussianFunction((GaussianFunction(ProjectionPolynomial.constant(52,1),cast(np.ndarray,phase)),))
        for row,function in enumerate(base):
            out[row,97+index] = (high_precision_kernel(function,wave)
                if row==15 else function_kernel(function,wave))
    out[97:,:97] = out[:97,97:].conj().T
    delta = k[:,None,:]-k[None,:,:]
    out[97:,97:] = np.exp(-np.sum(delta*delta,axis=2)/4)
    return out,bonds,size


def source(model: Any,sample: dict[str,Any],profile: np.ndarray,
           dq: np.ndarray | None = None,dtheta: np.ndarray | None = None,
           sign: float = 0.,step: float = 0.) -> dict[str,Any]:
    point = sample['point']
    theta,a = sample['theta'],sample['a']
    if dq is not None and sign:
        point = model.chart.embedding(sample['q']+sign*step*dq)
    if dtheta is not None and sign:
        theta = theta+sign*step*dtheta
        a = a+sign*step*dtheta/model.law.size
    raw = model.fibre.source_vectors_at(point['phi'],point['links'],theta,a,profile,BETA)
    plus = raw['full108_plus_coefficients']@model.reference_frame
    minus = model.reference_frame.conj().T@raw['full108_minus_coefficients']
    return dict(plus=plus,minus=minus)


def matrix(model: Any,plus: np.ndarray,minus: np.ndarray) -> np.ndarray:
    return model.fibre.source_vectors_from_marks({1:plus.reshape(27,4,4),
        -1:minus.reshape(4,27,4).transpose(1,0,2)})['source_matrix']


def derivative(model: Any,sample: dict[str,Any],profile: np.ndarray,
               dq: np.ndarray | None=None,dtheta: np.ndarray | None=None) -> tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
    direction = dq if dq is not None else dtheta
    if direction is None:
        raise ValueError('A physical normal or harmonic SOURCE direction is required')
    scale = max(1.,float(np.linalg.norm(direction)))
    step = 2.**-19/scale
    def jet(h: float) -> dict[str,np.ndarray]:
        p = source(model,sample,profile,dq,dtheta,1.,h)
        m = source(model,sample,profile,dq,dtheta,-1.,h)
        return {key:(p[key]-m[key])/(2*h) for key in p}
    coarse,fine = jet(step),jet(step/2)
    return fine,{key:fine[key]-coarse[key] for key in fine}


def body_coefficients(model: Any,sample: dict[str,Any],pull: dict[str,Any],
                      velocity: np.ndarray,bonds: list[tuple[int,int,int,int]],size: int) -> tuple[np.ndarray,np.ndarray]:
    """FULL108 light Noe matrices; no Sigma, no Hamiltonian Lambda factor."""
    bodies = np.zeros((size,108,108),complex)
    gauge = model.gauge_operators(pull['gauge'],'neutral')
    bodies[0] = gauge['body']
    for index,row in enumerate(model.noether_bank.coefficients['lambda_C']):
        bodies[16+index] = np.diag(np.repeat(row,4))
    for x,axis,target,index in bonds:
        value = (velocity[x,axis]+velocity[target,axis])/(4j*model.law.band.a)
        block = value*sample['point']['links'][x,axis]*np.exp(1j*sample['theta'][axis]/3)
        bodies[index,4*x:4*x+4,4*target:4*target+4] = block
        bodies[index+1] = bodies[index].conj().T
    return bodies,gauge['reference']


def relative_coefficients(model: Any,sample: dict[str,Any],profile: np.ndarray,
                          pull: dict[str,Any],bodies: np.ndarray,reference: np.ndarray,
                          psi: np.ndarray,size: int) -> dict[str,np.ndarray]:
    """Return the full relative graph and the separate SOURCE-on-QPH boundary."""
    marks = source(model,sample,profile)
    dq,error = derivative(model,sample,profile,dq=pull['normal'])
    dtheta = [derivative(model,sample,profile,dtheta=np.eye(3)[axis])[0] for axis in range(3)]
    full,ph = np.zeros((448,size),complex),np.zeros((448,size),complex)
    for index,body in enumerate(bodies):
        plus,minus = body@marks['plus'],-marks['minus']@body
        if index==0:
            plus += -marks['plus']@reference-1j*dq['plus']
            minus += reference@marks['minus']-1j*dq['minus']
        elif 16<=index<97:
            row = model.noether_bank.coefficients['harmonic'][index-16]
            plus -= 1j*sum((row[a]*dtheta[a]['plus'] for a in range(3)),np.zeros_like(plus))
            minus -= 1j*sum((row[a]*dtheta[a]['minus'] for a in range(3)),np.zeros_like(minus))
        full[:,index] = matrix(model,plus,minus)@psi
        pplus,pminus = model.fibre.projectors[1],model.fibre.projectors[-1]
        lp = (np.eye(108)-pplus)@body@pplus@marks['plus']
        lm = -marks['minus']@(np.eye(108)-pminus)@body@pminus
        ph[:,index] = matrix(model,lp,lm)@psi
    value = matrix(model,marks['plus'],marks['minus'])@psi
    fd = matrix(model,-1j*error['plus'],-1j*error['minus'])@psi
    return dict(full=full,light_PH_relative=ph,source=value,normal_FD_difference=fd)


def norm(coeff: np.ndarray,gram: np.ndarray) -> float:
    return float(np.einsum('ai,ij,aj->',coeff.conj(),gram,coeff,optimize=True).real)


def overlap(left: np.ndarray,right: np.ndarray,gram: np.ndarray) -> complex:
    return complex(np.einsum('ai,ij,aj->',left.conj(),gram,right,optimize=True))


def raw_projection(base: np.ndarray,extra: np.ndarray,value: np.ndarray,kernel: np.ndarray) -> np.ndarray:
    hard = (base.conj().T@value)@kernel[:15].T
    generated = np.einsum('aek,al,kl->e',extra.conj(),value,kernel[15:97],optimize=True)
    return np.r_[hard.ravel(),generated]


def run(model: Any,data: dict[str,Any],generators: dict[str,np.ndarray],gmeta: dict[str,Any]) -> tuple[dict[str,Any],dict[str,np.ndarray]]:
    argv = data['manifest']['argv']
    guard_path = Path(argv[argv.index('--noether-guard')+1])
    if sha(guard_path)!=data['manifest']['input_sha256']['noether_guard']:
        raise ValueError('Original guard changed')
    guard = json.loads(guard_path.read_bytes())
    alpha,velocity = np.asarray(guard['alpha']),np.asarray(guard['velocity'])
    kernels,bonds,size = library(model,velocity)
    records = frozen_nodes(model)
    result,arrays = {},{}
    j = {name:{key:np.zeros(1728,complex) for key in ('full','core','complement','PH')} for name in ('source19','SOFT')}
    totals: dict[str,dict[str,Any]] = {name:dict(full=0.,core=0.,complement=0.,cross=0.,PH=0.,source=0.,source_core=0.,source_complement=0.,source_cross=0.,fd=0.,missingRIGHT_delta=0.,scalar_inner=0j) for name in j}
    maximum_linearity = 0.
    source_coexact_control = {}
    cached = {}
    for name in j:
        mapping = data[name+'_map']
        cached[name] = solve(data['charged_Gram'],generators['charged_G']@mapping@data['neutral_state'],assume_a='her')-mapping@solve(data['neutral_Gram'],generators['neutral_G']@data['neutral_state'],assume_a='her')
    for record in records:
        sample = record['sample']
        if not source_coexact_control:
            y = np.arange(1,53,dtype=float)/53
            altered = dict(sample,a=sample['a']+(model.em['legs']@y).reshape(27,3))
            for name,profile in model.source_bank.profiles.items():
                original,changed = source(model,sample,profile),source(model,altered,profile)
                error = max(float(np.linalg.norm(changed[key]-original[key]))/
                    (1e-100+float(np.linalg.norm(original[key]))) for key in original)
                source_coexact_control[name] = error
                if error>1e-10:
                    raise ValueError('Original SOURCE coexact dependence requires a larger exact scalar library')
        base,extra = conditional_values(model,sample)
        psi = base['neutral']@model.prepared14
        pull = physical_chart_pullback(model,sample,alpha,velocity)
        bodies,reference = body_coefficients(model,sample,pull,velocity,bonds,size)
        weight = float(sample['importance_weight'])/len(records)
        q,dq = sample['q'],pull['normal']
        logjet = model.chart.directional_jet(q,dq,sample['point'])['log_density_first']
        logder = -float(q@dq)/model.sigma**2-.5*float(logjet)+float(sample['cutoff_gradient']@dq)/sample['cutoff']
        dp = sample['polynomial_derivative'].T@dq
        dn = np.zeros_like(base['neutral']);dn[0,:7]=dp;dn[1,7:]=dp
        neutral_derivative = (dn+logder*base['neutral'])@model.prepared14
        for name,profile in model.source_bank.profiles.items():
            pieces = [profile,profile*(alpha==1),profile*(alpha!=1)]
            rows = [relative_coefficients(model,sample,p,pull,bodies,reference,psi,size) for p in pieces]
            whole,core,complement = rows
            maximum_linearity = max(maximum_linearity,float(np.linalg.norm(whole['full']-core['full']-complement['full'])/(1e-100+np.linalg.norm(whole['full']))))
            t = totals[name]
            for key,row in zip(('full','core','complement'),rows):
                t[key] += weight*norm(row['full'],kernels)
                j[name][key] += weight*raw_projection(base['charged'],extra['charged'],row['full'],kernels[:97])
            t['cross'] += weight*2*overlap(core['full'],complement['full'],kernels).real
            t['PH'] += weight*norm(whole['light_PH_relative'],kernels)
            j[name]['PH'] += weight*raw_projection(base['charged'],extra['charged'],whole['light_PH_relative'],kernels[:97])
            for key,row in zip(('source','source_core','source_complement'),rows):
                t[key] += weight*float(np.vdot(row['source'],row['source']).real)
            t['source_cross'] += weight*2*float(np.vdot(core['source'],complement['source']).real)
            t['fd'] += weight*float(np.vdot(whole['normal_FD_difference'],whole['normal_FD_difference']).real)
            marks = source(model,sample,profile)
            omitted = matrix(model,marks['plus'],marks['minus'])@(-1j*neutral_derivative)
            t['missingRIGHT_delta'] += weight*float(np.vdot(omitted,omitted).real)
            t['scalar_inner'] += weight*complex(whole['source'].conj()@whole['full']@kernels[0])
    for name,t in totals.items():
        z = gmeta['loop']['observations'][name]['native_SOURCE_norm']
        gram = data['charged_Gram']
        projection = {}
        for key,value in j[name].items():
            weak = data['charged_coordinates'].conj().T@value
            projection[key] = solve(gram,weak,assume_a='her')
            arrays[name+'_projected_'+key] = projection[key]/z
        p = projection['full']
        pnorm = float(np.vdot(p,gram@p).real)
        remainder = t['full']-pnorm
        if remainder < -1e-9*max(t['full'],pnorm,1e-100):
            raise ValueError('Exact conditional full graph/projection norm inconsistent')
        difference = p-cached[name]
        residual = difference-projection['PH']
        scalar = t['scalar_inner']/t['source']
        result[name] = dict(global_initial_SOURCE_Z=z,source_norm_squared=t['source']/z**2,
            SOURCE_geometry=dict(core_norm_squared=t['source_core']/z**2,complement_norm_squared=t['source_complement']/z**2,coherent_cross_term=t['source_cross']/z**2),
            raw_relative_graph=dict(full_norm_squared=t['full']/z**2,core_norm_squared=t['core']/z**2,complement_norm_squared=t['complement']/z**2,coherent_cross_term=t['cross']/z**2),
            left_projected_full_norm_squared=pnorm/z**2,unprojected_QC_full_norm_squared=remainder/z**2,
            cached_Galerkin_relative_norm_squared=float(np.vdot(cached[name],gram@cached[name]).real)/z**2,
            PC_full_minus_cached_Galerkin_norm_squared=float(np.vdot(difference,gram@difference).real)/z**2,
            full_light_PH_relative_norm_squared=t['PH']/z**2,
            projected_light_PH_relative_norm_squared=float(np.vdot(projection['PH'],gram@projection['PH']).real)/z**2,
            difference_after_removing_light_PH_channel_squared=float(np.vdot(residual,gram@residual).real)/z**2,
            normal_SOURCE_FD_refinement_difference_norm=math.sqrt(t['fd'])/z,
            positive_adverse=dict(missing_neutral_RIGHT_normal_derivative_delta_norm=math.sqrt(t['missingRIGHT_delta'])/z,
                dropping_complement_graph_norm=math.sqrt(t['complement'])/z,
                dropping_coherent_cross_changes_graph_squared=t['cross']/z**2,
                dropping_light_PH_left_projected_channel_norm=math.sqrt(float(np.vdot(projection['PH'],gram@projection['PH']).real))/z,
                scalar_best_fit_coefficient=[scalar.real,scalar.imag],scalar_fit_residual_norm_squared=(t['full']-abs(t['scalar_inner'])**2/t['source'])/z**2))
        arrays[name+'_cached_Galerkin_relative'] = cached[name]/z
    return dict(profiles=result,maximum_profile_split_linearity_relative_difference=maximum_linearity,
        Phi_harmonic_atoms=len(records),EM_coordinates=52,EM_integration='exact finite-polynomial/Weyl Gaussian moments; no G-matrix recompile',
        no_H_G_preparation_or_loop_rerun=True,no_profile_or_piece_renormalization=True,
        original_SOURCE_coexact_dependence_control=source_coexact_control,
        neutral_RIGHT='common field/sea/heavy derivatives cancel by product rule and full108 CAR BEFORE graph norms',
        heavy_common_PH_relative_commutator_exact_zero=True,
        leakage_identity='P_C R_full-R_Gal=P_C G Q_C B P_N-P_C B Q_N G P_N; separately retain Q_C R_full',
        cached_Galerkin_difference_NOT_sole_PH_cause=True,
        remaining_difference_scope='normal field-function closure plus cached G quadrature/FD and other projection channels, NOT a measured sole cause',
        scope='native conditional frozen14 Phi/harmonic norm, exact all52 EM; NOT rigorous whole-field continuum L2/legalSOFT bound, ground or physical time loop',
        legal_bridge_unchanged=True,native_H_light_coefficient=1,native_H_heavy_coefficient='-Lambda; no H action used by this graph',campaign_open=True),arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache-npz','cache-manifest','generator-npz','generator-receipt','output','output-npz','runtime-archive'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    model,data = load_shared_native_cache(args.cache_npz,args.cache_manifest)
    generators,gmeta = load_noether_export(args.generator_npz,args.generator_receipt)
    if gmeta['cache_npz_sha256']!=sha(args.cache_npz) or gmeta['cache_manifest_sha256']!=sha(args.cache_manifest):
        raise ValueError('Graphs require the SAME sealed H/state cache')
    binding = runtime_binding(args.runtime_archive)
    science,arrays = run(model,data,generators,gmeta)
    np.savez_compressed(args.output_npz,**cast(Any,arrays))
    receipt = dict(science=science,source_sha256=sha(Path(__file__)),runtime_binding=binding,
        cache_npz_sha256=sha(args.cache_npz),cache_manifest_sha256=sha(args.cache_manifest),
        generator_npz_sha256=sha(args.generator_npz),generator_receipt_sha256=sha(args.generator_receipt),
        output_npz_sha256=sha(args.output_npz),argv=sys.argv,wall_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
