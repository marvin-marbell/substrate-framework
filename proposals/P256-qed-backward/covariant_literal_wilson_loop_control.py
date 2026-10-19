#!/usr/bin/env python3
"""Energy-form prices for literal shortest body-to-Ref Wilson paths and loops."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_literal_wilson_loop_control.py','covariant_photon_energy_control.py',
    'covariant_reference_log_jet_words.py','covariant_induced_photon.py','sea_selection.py','rotation_band.py')}
from covariant_induced_photon import Parameters
from covariant_photon_energy_control import gaussian_energy_form_control
from covariant_reference_log_jet_words import sf_majorants,compositions


def source_path(site: tuple[int,int,int],L: int) -> list[dict[str,Any]]:
    current=list(site);result=[]
    for axis in (0,2,1):
        delta=(-current[axis])%L
        if delta>=L//2:delta-=L
        if delta:
            result.append(dict(start=list(current),axis=axis,orientation=1 if delta>0 else -1,length=abs(delta)))
            current[axis]=(current[axis]+delta)%L
    if current!=[0,0,0]:raise ArithmeticError('Literal x,z,y body-to-Ref path does not end at Ref')
    return result


def loop(site: tuple[int,int,int],axis: int,L: int) -> list[dict[str,Any]]:
    neighbor=list(site);neighbor[axis]=(neighbor[axis]+1)%L
    # U_direct / (W_x,Ref W_y,Ref^dagger).
    result=[dict(start=list(site),axis=axis,orientation=1,length=1,coefficient=1)]
    result.extend(dict(segment,coefficient=-1) for segment in source_path(site,L))
    result.extend(dict(segment,coefficient=1) for segment in source_path((neighbor[0],neighbor[1],neighbor[2]),L))
    return result


def winding(segments: list[dict[str,Any]],L: int) -> list[int]:
    displacement=[0,0,0]
    for segment in segments:
        displacement[segment['axis']]+=segment['coefficient']*segment['orientation']*segment['length']
    if any(x%L for x in displacement):raise ArithmeticError('Literal nearest-link/tree loop is not a closed periodic integer chain')
    return [x//L for x in displacement]


def native_loop_forms(L: int) -> list[dict[str,Any]]:
    n=np.arange(L);p=2*math.pi*((n+L//2)%L-L//2)/L
    q=np.stack(np.meshgrid(p,p,p,indexing='ij'),axis=-1)
    omega=2*np.arcsinh(np.linalg.norm(2*np.sin(q/2),axis=-1)/2)
    nonzero=omega>0;s=np.sinh(omega)
    qhat=2*np.sin(q/2);qhat2=np.sum(qhat*qhat,axis=-1)
    inverse=np.zeros_like(omega);inverse[nonzero]=1/(omega[nonzero]*s[nonzero])
    variance=np.zeros_like(omega);variance[nonzero]=1/s[nonzero]
    c=2*math.asinh(math.sqrt(3))/(math.pi*math.sqrt(3))
    harmonic=sum(1/r for r in range(1,L//2+1))
    coefficient=2*harmonic/(math.pi**2*c*c)+1/(12*c*c)
    rows=[]
    for site in ((L//4,0,0),(L//2-1,L//2-1,L//2-1),(L//2,L//2,L//2)):
        for axis in range(3):
            segments=loop(site,axis,L);J=np.zeros((*omega.shape,3),complex)
            for segment in segments:
                direction=segment['axis'];orientation=segment['orientation'];length=segment['length']
                phase=orientation*q[...,direction]
                # The removable zero is evaluated by the finite geometric sum,
                # rather than dividing two small sine functions.
                geometric=np.zeros_like(phase,dtype=complex)
                for k in range(length):geometric+=np.exp(1j*k*phase)
                midpoint=np.einsum('...j,j->...',q,np.array(segment['start']))+phase/2
                J[...,direction]+=segment['coefficient']*orientation*np.exp(1j*midpoint)*geometric
            longitudinal=np.zeros_like(omega,dtype=complex)
            longitudinal[nonzero]=np.sum(qhat*J,axis=-1)[nonzero]/qhat2[nonzero]
            transverse=J-qhat*longitudinal[...,None]
            norm=np.sum(np.abs(transverse)**2,axis=-1)
            K=float(np.sum(norm*inverse))/L**3
            V=float(np.sum(norm*variance))/L**3
            m=len(segments);perimeter=sum(segment['length'] for segment in segments)
            upper=m*perimeter*coefficient
            variance_upper=math.sqrt(upper*m*perimeter)
            if K>upper*(1+1e-12) or V>variance_upper*(1+1e-12):
                raise ArithmeticError('Actual literal loop exceeds the segment Parseval/inductance upper')
            divergence=float(np.max(np.abs(np.sum(qhat*J,axis=-1))))
            rows.append(dict(spatial_size=L,body_site=list(site),nearest_neighbor_axis=axis,
                literal_body_to_Ref_path=source_path(site,L),actual_loop_segments=segments,
                actual_integer_winding=winding(segments,L),segment_count=m,perimeter=perimeter,
                exact_nonzero_mode_count=L**3-1,actual_transverse_energy_form_K=K,
                actual_transverse_variance_form_V=V,proved_energy_form_K_upper=upper,
                proved_variance_form_V_upper=variance_upper,
                midpoint_Fourier_closed_chain_divergence=divergence))
    return rows


def conditional_words(p: Parameters,B: float,energy: dict[str,Any]) -> dict[str,Any]:
    sf=sf_majorants(p.spatial_size**3,5)['SF_relative_coefficients']
    d=math.log(8)+energy['photon_maximum_energy']/2
    relative=[1.]+[float(sf[n])*B**n*math.exp(n*d)*(1+2**-40) for n in range(1,6)]
    transfer=[1.]+[sum(relative[j]*relative[n-j] for j in range(n+1)) for n in range(1,6)]
    H=[0.]+[sum(math.prod(transfer[j] for j in word) for word in compositions(n)) for n in range(1,6)]
    words=[]
    for weight in range(5):
        for prefix in compositions(weight):
            for last in range(1,6):
                if weight+last<5:continue
                word=prefix+(last,);length=len(word)
                price=-p.beta_power/2*sum(word)+length*math.log10(p.time_slices)
                price+=sum(math.log10(H[j]) for j in word)-math.lgamma(length+1)/math.log(10)
                words.append(dict(jet_grades=list(word),conditional_log10_upper=price))
    peak=max(row['conditional_log10_upper'] for row in words)
    total=peak+math.log10(sum(10**(row['conditional_log10_upper']-peak) for row in words))
    return dict(active_reference_H_jet_log10_upper=[math.log10(x) for x in H[1:]],
        conditional_terminal_words=words,conditional_terminal_sum_log10_upper=total,
        meaning='Gaussian constant-Phi reference coefficient-price only. Actual Gauss-affineflux/commonrange/higher source and along-real-g estimates are NOT certified by replacing a field factor.')


def run() -> dict[str,Any]:
    p=Parameters(kappa_power=400);L=p.spatial_size
    energy=gaussian_energy_form_control(p,9,.25)
    c=energy['dispersion_lower_coefficient'];harmonic_upper=1+math.log(L/2)
    coefficient_upper=2*harmonic_upper/(math.pi**2*c*c)+1/(12*c*c)
    loop_segments=7;loop_perimeter=3*L+1
    loop_K=loop_segments*loop_perimeter*coefficient_upper
    loop_V=math.sqrt(loop_K*loop_segments*loop_perimeter)
    initial=energy['maximum_intermediate_energy']
    loop_A=math.sqrt(2*initial*loop_K+loop_V)
    path_segments=3;path_perimeter=3*L/2
    path_K=path_segments*path_perimeter*coefficient_upper
    path_V=math.sqrt(path_K*path_segments*path_perimeter)
    path_A=math.sqrt(2*initial*path_K+path_V)
    native_sizes=(4,8,16,32,64)
    inputs=dict(parameters=asdict(p),source_path='Shortest body-to-Ref x,z,y; negative direction on exact half-box ties',
        initial_reference_energy=.25,maximum_word_grade=9,maximum_loop_segments=loop_segments,
        maximum_loop_perimeter=loop_perimeter,maximum_source_path_segments=path_segments,
        maximum_source_path_perimeter=path_perimeter,nonzero_coexact_modes_only=True,
        harmonic_and_statistical_torons='Retained exactly, NEVER estimated by coexact energy-form bound',
        native_box_sizes=list(native_sizes))
    native=[row for box in native_sizes for row in native_loop_forms(box)]
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        Gaussian_word_energy_core=energy,macro_segment_inductance_coefficient_upper=coefficient_upper,
        macro_literal_loop_energy_form_K_upper=loop_K,macro_literal_loop_variance_form_V_upper=loop_V,
        macro_literal_loop_A_word_factor=loop_A,macro_literal_source_path_energy_form_K_upper=path_K,
        macro_literal_source_path_variance_form_V_upper=path_V,macro_literal_source_path_A_word_factor=path_A,
        actual_native_literal_loop_rows=native,conditional_reference_words=conditional_words(p,loop_A,energy),
        proof=[
            'A(J)=sum_e J_e A_e uses the actual coexact Gaussian covariance 1/(2sinh E_q). Annihilation energy estimates and creation CCR yield ||A(J)P_[Hph<=e]||²<=2e K_J+V_J, K_J=V^-1sum |J_T(q)|²/(E_q sinh E_q), V_J=V^-1sum |J_T(q)|²/sinh E_q.',
            'E_q>=c|p| and sinh E_q>=E_q. A straight segment of lengthd<=L has Fourier factor D_d(p_j). A perpendicular infinity shell has <=8r sites, hence its sum of1/(n_j²+|nperp|²) is <=8H_(L/2), plus1/n_j² for its zero perpendicular mode when n_j!=0.',
            'Exact1DParseval sum_n |D_d|²=Ld; the remaining sum |D_d|²/n²<=d² pi²/3. Therefore K_segment<=d[2H_(L/2)/(pi²c²)+1/(12c²)]. This grows with perimeter times logL, not guessed local64 or perimeter squared.',
            'For m oriented segments, FourierCauchy gives K_J<=m sum K_segment and ||J||²_l2<=m sumd. Since E_q/sinhE_q<=1, weightedCauchy yields V_J<=sqrt(K_J ||J||²_l2). Coexact projection is a contraction, including for open source paths.',
            'Literalnearestlink minusbodyx-toRef plusbodyy-toRef has<=7segments,totalperimeter<=3L+1. Each source path has<=3segments,perimeter<=3L/2. Exact half-box directions and x,z,y ordering are explicitly executed in native rows.',
            'The loop has exact integer windingw; its harmonic Wilson factor exp(i w dot theta) and statistical seams stay inside their exact transfer, not a small-field Taylor estimate. Only the nonzero coexact part is priced here.',
            'Exp(i g A_loop)-1 has ordinary field coefficients(i A_loop)^n/n!, not only its linear term. Existing activeSF inverse/sqrt/determinant/full exterior majorants retain these nonlinear contacts throughgrade5.',
            'This does not remove the rho-dependent transverse phase of the actual Gauss-affineflux kernel; nor does it identify the native treehomotopy t0 transfer with freeGaussian+CAR. No source/preparation gate or all-time coordinate confinement is introduced.'],
        microscopic_commonrange_or_Born_admission_proved=False,
        scope='Proved Gaussian reference energy-form price for actual literal long Wilson loops and source paths; actual microscopic state/commonrange and matching along-real-g remainder remain separate.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();result=run();result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256',
        'macro_literal_loop_A_word_factor','macro_literal_source_path_A_word_factor',
        'conditional_reference_words','runtime_seconds')}))


if __name__=='__main__':
    main()
