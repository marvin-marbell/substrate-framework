#!/usr/bin/env python3
"""Exact finite-N RAW SU4 colour-Haar divergence of the original clover flow.

No finite difference enters colour_haar_divergence_exact. The CLI evaluates
NEW noncentral RAW inputs and independently differentiates the original
clover for validation. This is not the normal-chart rho/Gauss/Ref density,
a full quantum generator, a state construction or an H-OmegaG evolution.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from scipy.linalg import expm
from improved_phi_band import ImprovedPhiLaw,selected_band
from original_total_angular_transfer import oriented_step,clover,primitive_flow
from raw_source_relative_noether_graph import runtime_binding

IDENTITY=np.eye(4,dtype=complex)


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generator_normalization(generators:np.ndarray)->dict[str,float]:
    """Read and verify the actual 15 generators, not an assumed convention."""
    if generators.shape!=(15,4,4):
        raise ValueError('This producer is SU4 with all15 generators')
    gram=np.einsum('aij,bji->ab',generators,generators)
    casimir=np.einsum('aij,ajk->ik',generators,generators)
    result=dict(trace_Gram_half_identity_error=float(np.max(abs(gram-.5*np.eye(15)))),
        Hermiticity_error=float(np.max(abs(generators-generators.conj().swapaxes(-1,-2)))),
        traceless_error=float(np.max(abs(np.trace(generators,axis1=1,axis2=2)))),
        quadratic_Casimir=float(np.trace(casimir).real/4),
        Casimir_identity_error=float(np.max(abs(casimir-15/8*IDENTITY))))
    if max(result[key] for key in result if key.endswith('_error'))>1e-12:
        raise ValueError('Actual generators do not satisfy Tr(T_a T_b)=delta_ab/2')
    return result


def loop_edges(law:Any,links:np.ndarray,zeros:np.ndarray,anchor:int,
               pair:tuple[int,...],signs:tuple[int,...])->list[tuple[np.ndarray,tuple[int,int],int]]:
    """Actual oriented_step products plus the edge on which each factor lives."""
    first,second=pair;s,t=signs;point=anchor;result=[]
    for axis,direction in ((first,s),(second,t),(first,-s),(second,-t)):
        previous=point;point,unitary,_=oriented_step(law,point,axis,direction,links,zeros)
        result.append((unitary,(previous if direction==1 else point,axis),direction))
    if point!=anchor:
        raise ValueError('Original native oriented plaquette does not close')
    return result


def fierz_trace(left:np.ndarray,right:np.ndarray)->complex:
    """sum_a Tr(T_a left T_a right), for the verified actual convention."""
    return .5*(np.trace(left)*np.trace(right)-np.trace(left@right)/4)


def colour_haar_divergence_exact(law:Any,links:np.ndarray,velocity:np.ndarray)->dict[str,Any]:
    """Signed sum X_(edge,a) Y_(edge,a), X U=iT_a U, allN>=3.

    K_edge=-i dotU U^dagger=-a/2 sum_j vbar_j(C_x+U C_y U^dagger).
    A positive/negative edge occurrence gives deltaP=+i A T_a B or
    -i A T_a B. Its two-generator trace is contracted by Fierz. The
    derivative of the outer U C_y U^dagger is a commutator and its
    diagonal trace Tr(T_a [T_a,K]) vanishes, not the whole divergence.
    """
    normalization=generator_normalization(law.generators)
    if law.size<3 or links.shape!=(law.volume,3,4,4) or velocity.shape!=(law.volume,3):
        raise ValueError('Full native N>=3 links and actual physical velocity required')
    zeros=np.zeros((law.volume,3));per_edge=np.zeros((law.volume,3))
    # Keep all terms signed, including nonzero local values and their cancellation.
    terms=[]
    for x in range(law.volume):
        for axis in range(3):
            y=int(law.target[x,axis]);u=links[x,axis]
            for other in range(3):
                if other==axis:continue
                v=float((velocity[x,other]+velocity[y,other])/2)
                pair=tuple(sorted((other,axis)))
                orientation=1 if other<axis else -1
                for anchor,transport in ((x,IDENTITY),(y,u)):
                    for signs in itertools.product((-1,1),repeat=2):
                        edges=loop_edges(law,links,zeros,anchor,pair,signs)
                        prefix=[IDENTITY]
                        for edge,_,_ in edges:prefix.append(prefix[-1]@edge)
                        suffix=[IDENTITY for _ in range(5)]
                        for index in range(3,-1,-1):suffix[index]=edges[index][0]@suffix[index+1]
                        for index,(edge,wire,direction) in enumerate(edges):
                            if wire!=(x,axis):continue
                            if direction==1:
                                left,right=prefix[index],edge@suffix[index+1]
                            else:
                                left,right=prefix[index]@edge,suffix[index+1]
                            trace=fierz_trace(transport@left,right@transport.conj().T)
                            term=-v*orientation*signs[0]*signs[1]*direction*float(trace.real)/(4*law.band.a)
                            per_edge[x,axis]+=term
                            terms.append(term)
    return dict(total=float(np.sum(per_edge)),per_edge=per_edge,
        signed_occurrence_sum=float(sum(terms)),absolute_occurrence_sum=float(sum(abs(t) for t in terms)),
        occurrence_count=len(terms),generator_normalization=normalization)


def colour_haar_divergence_faces(law:Any,links:np.ndarray,velocity:np.ndarray)->dict[str,Any]:
    """Equivalent local-face formula, after both anchors are combined.

    D_(x,i)=C_F/(2a) sum_(j!=i) vbar_(x,i;j)
        [ReTr P_(x;ij)-ReTr P_(x-e_j;ij)].
    Each P is the literal positively oriented square; Real trace is
    orientation independent. Summing gives weighted face differences
    of the actual vbar, not an assumed zero or continuum div(xi).
    """
    generator_normalization(law.generators)
    zeros=np.zeros((law.volume,3));faces={};per_edge=np.zeros((law.volume,3))
    for pair in itertools.combinations(range(3),2):
        values=np.zeros(law.volume)
        for x in range(law.volume):
            product=IDENTITY
            for edge,_,_ in loop_edges(law,links,zeros,x,pair,(1,1)):product=product@edge
            values[x]=float(np.trace(product).real)
        faces[pair]=values
    for x in range(law.volume):
        for axis in range(3):
            y=int(law.target[x,axis])
            for other in range(3):
                if other==axis:continue
                site=law.sites[x].copy();site[other]=(site[other]-1)%law.size
                before=int(law.index[tuple(site)]);values=faces[tuple(sorted((axis,other)))]
                v=(velocity[x,other]+velocity[y,other])/2
                per_edge[x,axis]+=15/(16*law.band.a)*v*(values[x]-values[before])
    return dict(total=float(per_edge.sum()),per_edge=per_edge,face_real_traces=faces)


def original_clover_left_generator(law:Any,links:np.ndarray,velocity:np.ndarray,
                                   x:int,axis:int)->np.ndarray:
    """Independent FD provider: original clover, no analytic wire/Fierz code."""
    colour,_=clover(law,links,np.zeros((law.volume,3)))
    y=int(law.target[x,axis]);u=links[x,axis];result=np.zeros((4,4),complex)
    for other in range(3):
        v=(velocity[x,other]+velocity[y,other])/2
        result-=law.band.a*v*(colour[x,other,axis]+u@colour[y,other,axis]@u.conj().T)/2
    return result


def colour_haar_divergence_fd(law:Any,links:np.ndarray,velocity:np.ndarray,step:float)->np.ndarray:
    """All81*15 native directions atN3, independently using original clover."""
    rotations=[(expm(1j*step*t),expm(-1j*step*t)) for t in law.generators]
    result=np.zeros((law.volume,3))
    for x in range(law.volume):
        for axis in range(3):
            for generator,(positive,negative) in zip(law.generators,rotations):
                values=[]
                for rotation in (positive,negative):
                    changed=links.copy();changed[x,axis]=rotation@links[x,axis]
                    body=original_clover_left_generator(law,changed,velocity,x,axis)
                    values.append(float(2*np.trace(generator@body).real))
                result[x,axis]+=(values[0]-values[1])/(2*step)
    return result


def positive_control(generators:np.ndarray,u:np.ndarray,step:float)->dict[str,float]:
    """Known nonzero SU4 vector K(U)=traceless Im(U), separate from Noether."""
    direct=float(15/4*np.trace(u).real);fierz=float(2*fierz_trace(IDENTITY,u).real)
    fd=0.
    for t in generators:
        values=[]
        for sign in (1,-1):
            changed=expm(sign*1j*step*t)@u
            k=(changed-changed.conj().T)/(2j)
            k-=np.trace(k)/4*IDENTITY
            values.append(float(2*np.trace(t@k).real))
        fd+=(values[0]-values[1])/(2*step)
    if abs(direct)<1 or abs(fd-direct)>1e-6:
        raise AssertionError('Known nonzero independent FD/Fierz positive control fails')
    return dict(closed_exact=direct,Fierz=fierz,independent_FD=float(fd),absolute_FD_error=abs(fd-direct))


def point_binding(law:Any,fields:np.ndarray,links:np.ndarray,phases:np.ndarray,
                  velocity:np.ndarray,alpha:np.ndarray)->dict[str,float]:
    """Bind new analytic input to the actual original complete primitive flow."""
    phi,du,_=primitive_flow(law,fields,links,phases,velocity,alpha)
    actual=-1j*du@links.conj().swapaxes(-1,-2)
    colour,_=clover(law,links,phases);local=np.zeros_like(links)
    for x in range(law.volume):
        for axis in range(3):
            y=int(law.target[x,axis]);u=links[x,axis]
            for other in range(3):
                v=(velocity[x,other]+velocity[y,other])/2
                local[x,axis]-=law.band.a*v*(colour[x,other,axis]+u@colour[y,other,axis]@u.conj().T)/2
    return dict(original_complete_primitive_left_generator_error=float(np.max(abs(actual-local))),
        original_Phi_flow_norm=float(np.linalg.norm(phi)),
        link_unitarity_error=float(np.max(abs(links@links.conj().swapaxes(-1,-2)-IDENTITY))),
        link_determinant_error=float(np.max(abs(np.linalg.det(links)-1))),
        noncentral_commutator_norm=float(np.linalg.norm(links[0,0]@links[0,1]-links[0,1]@links[0,0])))


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-npz','cache-manifest','output','output-npz','runtime-archive'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    manifest=json.loads(args.cache_manifest.read_bytes())
    if sha(args.cache_npz)!=manifest['npz_sha256']:
        raise ValueError('Frozen cache changed')
    guard_argv=manifest['argv'];guard_path=Path(guard_argv[guard_argv.index('--noether-guard')+1])
    if sha(guard_path)!=manifest['input_sha256']['noether_guard']:
        raise ValueError('Actual physical Noether velocity supplier changed')
    guard=json.loads(guard_path.read_bytes());velocity=np.asarray(guard['velocity']);alpha=np.asarray(guard['alpha'])
    band,_,_=selected_band();law=ImprovedPhiLaw(3,band)
    seed=dict(manifest['seed_snapshot']['fibre']['dict'])
    with np.load(args.cache_npz) as cache:
        mean_fields=cache[seed['mean_fields']['array']];mean_links=cache[seed['mean_links']['array']]
    rng=np.random.default_rng(222341)
    def unitaries(shape:tuple[int,...],scale:float)->np.ndarray:
        coefficients=rng.normal(size=(*shape,15))*scale
        algebra=np.einsum('...a,aij->...ij',coefficients,law.generators)
        return np.array([expm(1j*k) for k in algebra.reshape(-1,4,4)]).reshape(*shape,4,4)
    fields=mean_fields+law.fields(.11*rng.normal(size=(law.volume,4,15)))
    links=unitaries((law.volume,3),.51)@mean_links
    phases=rng.uniform(-.8,.8,(law.volume,3))
    gauge=unitaries((law.volume,),.73)
    moved_fields=gauge[:,None]@fields@gauge[:,None].conj().swapaxes(-1,-2)
    moved_links=np.array([[gauge[x]@links[x,axis]@gauge[law.target[x,axis]].conj().T for axis in range(3)] for x in range(law.volume)])
    adverse_fields=mean_fields+law.fields(.23*rng.normal(size=(law.volume,4,15)))
    adverse_links=unitaries((law.volume,3),1.07)@mean_links
    cases={'noncentral_RAW':(fields,links,phases),'independent_local_gauge':(moved_fields,moved_links,phases),
           'independent_strong_noncentral_RAW':(adverse_fields,adverse_links,-phases)}
    results={};arrays:dict[str,Any]=dict(physical_velocity=velocity,physical_intrinsic_alpha=alpha,gauge_transform=gauge)
    coarse,fine=2.**-14,2.**-15
    for name,(phi,u,a) in cases.items():
        begin=time.perf_counter();exact=colour_haar_divergence_exact(law,u,velocity);faces=colour_haar_divergence_faces(law,u,velocity)
        binding=point_binding(law,phi,u,a,velocity,alpha)
        if binding['original_complete_primitive_left_generator_error']>1e-10 or binding['noncentral_commutator_norm']<.01:
            raise AssertionError('Original primitive/actual noncentral positive binding fails')
        analytic_error=float(np.max(abs(exact['per_edge']-faces['per_edge'])))
        if analytic_error>1e-11:
            raise AssertionError(f'Independent Fierz/face native formula fails: {name} {analytic_error}')
        fd=colour_haar_divergence_fd(law,u,velocity,fine)
        fd_coarse=colour_haar_divergence_fd(law,u,velocity,coarse) if name=='noncentral_RAW' else None
        fd_error=float(np.max(abs(fd-exact['per_edge'])))
        if fd_error>2e-7:
            raise AssertionError(f'Independent analytic/FD native comparison fails: {name} {analytic_error} {fd_error}')
        results[name]=dict(total_exact=exact['total'],total_face_formula=faces['total'],total_independent_FD=float(fd.sum()),
            signed_occurrence_sum=exact['signed_occurrence_sum'],absolute_occurrence_sum=exact['absolute_occurrence_sum'],
            occurrence_count=exact['occurrence_count'],maximum_absolute_local_divergence=float(np.max(abs(exact['per_edge']))),
            local_Fierz_vs_face_max_error=analytic_error,local_exact_vs_FD_max_error=fd_error,
            coarse_FD_total=float(fd_coarse.sum()) if fd_coarse is not None else None,
            coarse_to_fine_local_max_difference=float(np.max(abs(fd-fd_coarse))) if fd_coarse is not None else None,
            point_binding=binding,wall_seconds=time.perf_counter()-begin)
        arrays.update({name+'_fields':phi,name+'_links':u,name+'_phases':a,
            name+'_exact_local_divergence':exact['per_edge'],name+'_independent_FD_local_divergence':fd})
    gauge_error=float(np.max(abs(arrays['noncentral_RAW_exact_local_divergence']-arrays['independent_local_gauge_exact_local_divergence'])))
    if gauge_error>1e-11:
        raise AssertionError('Exact native colour-Haar divergence is not locally gauge invariant')
    control=positive_control(law.generators,links[0,0],fine)
    np.savez_compressed(args.output_npz,**arrays)
    result=dict(plan='issue-222-fullfield-closedrotation-continuation-20261006',revision='668c7d8ea2ec9c71c964e9f16d3c75e8cd879e10433e72fbcad0c3b89c7da417',work_item='soft-spatial-closed-rotation',
        source_sha256=sha(Path(__file__)),input_sha256=dict(cache_npz=sha(args.cache_npz),cache_manifest=sha(args.cache_manifest),actual_physical_Noether_guard=sha(guard_path)),
        science=dict(native_N=3,general_formula_N_at_least=3,all_native_links=81,all_SU4_generators=15,
            generator_normalization=generator_normalization(law.generators),actual_velocity_norm=float(np.linalg.norm(velocity)),
            original_primitive_sign='Y_U=-i a sum_j vbar_j Cbar_ji U; this is V, not the negative driven characteristic -Omega V',
            cases=results,independent_gauge_local_divergence_error=gauge_error,known_nonzero_positive_control=control,
            exact_producer_has_no_FD=True,no_zero_assignment_or_clipping=True,no_source_profile_state_filter_or_law_change=True,
            new_RAW_inputs_not_old_baseline_rerun=True,
            scope='RAW SU4 colour-Haar divergence only. Binary64 trace evaluation with displayed SU4 defects, not a directed-rounding enclosure.',
            unpaid=['qchart rho/Jhalf/Gauss/Ref density equivalence','EM/Phi full quantum generator and common H propagation','Omega_dot and full spatial2pi/legal state/L2/Born/quantum-law/parameter closure']),
        export_npz_sha256=sha(args.output_npz),runtime_binding=runtime_binding(args.runtime_archive),
        metadata_allowlist=['argv','source/input hashes','explicit native law geometry','numeric controls','source/callable archive','wall_seconds'],
        full_ENV_captured=False,argv=sys.argv,wall_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    main()
