#!/usr/bin/env python3
"""NEW two-path auxiliary primitive; actual matched-tree light source/consumer.

The original single-edge auxiliary/source bytes are preserved. The new heavy
hopping is the equal superposition of TWO integer-charge paths on the same
plaquette, retaining a physical closed loop even at light-homotopy t0.
No bare gauge action, fitted speed, collapsed auxiliary or quartercut repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_native_closed_auxiliary_loop.py','covariant_native_source_gram_moments.py',
    'covariant_native_haar_range_jets.py','covariant_source_band_halfstep_control.py',
    'covariant_induced_photon.py','covariant_photon_energy_control.py','sea_selection.py','rotation_band.py')}
PLAN_PATH=Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA256_AT_IMPORT=hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
import covariant_native_haar_range_jets as native
from covariant_native_source_gram_moments import matched_tree_half
from covariant_induced_photon import Parameters,phi_frame


def auxiliary_path_half(gamma: np.ndarray,kappa: float,phase: complex) -> tuple[np.ndarray,float]:
    B=np.eye(2,dtype=complex)/(2*kappa)
    B[1,0]-=.5*phase;B[0,1]-=.5*np.conjugate(phase)
    F=np.zeros((2,2),complex);z=-gamma[2,0,1]/2
    F[1,0]+=z*phase;F[0,1]-=z*np.conjugate(phase)
    inverse=np.linalg.inv(B)
    T=np.block([[inverse,-inverse@F],[-F.conj().T@inverse,B+F.conj().T@inverse@F]])
    perm=[0,2,1,3];T=T[np.ix_(perm,perm)]
    values,U=eigh(T)
    if values[0]<=0:raise ArithmeticError('Full two-path auxiliary transfer is not positive')
    return (U*np.sqrt(values))@U.conj().T,float(np.sqrt(np.linalg.det(B).real))


def reflected_factor(gamma: np.ndarray,y: np.ndarray,heat_half: np.ndarray,
                     physical: dict[str,Any],grid: int) -> np.ndarray:
    middle=[physical['states'][i] for i in physical['middle']]
    ls=sorted(set(l for l,_ in middle));ars=sorted(set(a for _,a in middle))
    li={l:i for i,l in enumerate(ls)};ai={a:i for i,a in enumerate(ars)}
    lc=np.array([li[l] for l,_ in middle]);ac=np.array([ai[a] for _,a in middle])
    lr=np.array(physical['light']);la=np.array(ls)
    ar=np.array(physical['auxiliary']);aa=np.array(ars)
    rows=np.array([i for i,_,_ in physical['physical']]);flux=np.array([n%grid for _,n,_ in physical['physical']])
    blocks=[];outer_norm=0.
    for normal in y:
        actual=gamma.copy();actual[1]*=1+float(normal);samples=[]
        for theta in 2*math.pi*np.arange(grid)/grid:
            half,pref=matched_tree_half(actual,float(theta),0.)
            light=pref*np.linalg.det(half[lr[:,None,:,None],la[None,:,None,:]])
            # Long-tree gauge: direct 3->0 is exp(i theta); alternate path is1.
            heavy,hpref=auxiliary_path_half(actual,.04,complex((1+np.exp(1j*theta))/2))
            aux=hpref*np.linalg.det(heavy[ar[:,None,:,None],aa[None,:,None,:]])
            combined=light[:,lc][:,None,:]*aux[None,:,ac]
            samples.append(combined.reshape(len(physical['states']),len(middle)))
        coefficients=np.fft.fft(np.array(samples),axis=0)/grid
        blocks.append(coefficients[flux,rows,:])
        high=[j for j in range(grid) if min(j,grid-j)>12]
        outer_norm=max(outer_norm,float(np.linalg.norm(coefficients[high])))
    nr=len(rows);nc=len(middle);result=np.zeros((3*nr,3*nc),complex)
    for out in range(3):
        for inp in range(3):result[out*nr:(out+1)*nr,inp*nc:(inp+1)*nc]=heat_half[out,inp]*blocks[inp]
    physical['sampled_Fourier_norm_above12']=outer_norm
    return result


def run(grid: int) -> dict[str,Any]:
    gamma4,metric=phi_frame(Parameters(kappa_power=400))
    values,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,values>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    # Independent path/gauge controls, not a declaration that the new law=old.
    links=np.array([.17,-.23,.31,.47]);site_gauge=np.array([.13,-.37,.29,.41])
    incidence=np.zeros((4,4),int)
    for e in range(4):incidence[e,e]=1;incidence[(e+1)%4,e]=-1
    def combined_phase(A: np.ndarray) -> complex:
        return complex((np.exp(1j*A[3])+np.exp(-1j*np.sum(A[:3])))/2)
    phase=combined_phase(links);new_phase=combined_phase(links+incidence.T@site_gauge)
    expected=np.exp(1j*(site_gauge[3]-site_gauge[0]))*phase
    gauge_error=float(abs(new_phase-expected))
    old_mean,old_pref=native.auxiliary_half(gamma,.04,0.)
    new_mean,new_pref=auxiliary_path_half(gamma,.04,1.+0j)
    mean_error=float(np.linalg.norm(old_mean-new_mean,2)+abs(old_pref-new_pref))
    heavy,_=auxiliary_path_half(gamma,.04,phase);new_heavy,_=auxiliary_path_half(gamma,.04,new_phase)
    G=np.diag(np.repeat(np.exp(1j*site_gauge[[0,3]]),2))
    half_gauge_error=float(np.linalg.norm(new_heavy-G@heavy@G.conj().T,2))
    if max(gauge_error,mean_error,half_gauge_error)>1e-12:raise ArithmeticError('Actual two-path gauge/mean controls failed')
    y=np.array([-.09,0.,.09]);sigma=.03;radius=.12;dy=.09
    logk=-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y))
    k=np.exp(logk);k/=np.linalg.norm(k);D=np.zeros((2,3))
    for j in range(2):D[j,j]=math.sqrt(k[j+1]/k[j])/dy;D[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    Hphi=2*sigma*sigma*D.T@D;ev,evec=eigh(Hphi);heat_half=(evec*np.exp(-ev/2))@evec.T
    neutral=native.space(0,12);charged=native.space(1,12)
    Fn=reflected_factor(gamma,y,heat_half,neutral,grid);Fc=reflected_factor(gamma,y,heat_half,charged,grid)
    Vn,qn,_=np.linalg.svd(Fn,full_matrices=False);Vc,qc,_=np.linalg.svd(Fc,full_matrices=False)
    lambdaN=float(qn[0]**2);qn/=math.sqrt(lambdaN);qc/=math.sqrt(lambdaN)
    if min(qn[-1],qc[-1])<=0:raise ArithmeticError('Actual allcanonical positive range lost rank')
    En=-2*np.log(qn);Ec=-2*np.log(qc);Omega=Vn[:,0]
    Bn=native.curl(neutral,3);Bc=native.curl(charged,3)
    positive_curl=Vn.conj().T@(Bn@Vn)
    incoming_coefficients=qn*(Vn.conj().T@(Bn@Omega))
    Zph=float(np.vdot(incoming_coefficients,incoming_coefficients).real)
    if Zph<=0:raise ArithmeticError('New closed-loop primitive still has no physical sin(theta) prepared state')
    incoming_coefficients/=math.sqrt(Zph);incoming=Vn@incoming_coefficients
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet)
    outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    S=native.source(neutral,charged,native.source_coefficients(gamma,packet),embedding,3)
    Sout=native.source(neutral,charged,native.source_coefficients(gamma,outpacket),embedding,3)
    raw=(S@incoming).reshape(4,Vc.shape[0]);source=np.array([qc*(Vc.conj().T@row) for row in raw])
    M0=float(np.vdot(source,source).real)
    if M0<=0:raise ArithmeticError('Actual new primitive charged source has zero norm')
    E1=float(np.sum(abs(source)**2*Ec[None,:]));R1=float(np.sum(abs(source)**2*qc[None,:]**2))
    raw_target=(Sout@Omega).reshape(4,Vc.shape[0])
    receiver=np.array([Vc.conj().T@(Bc@row) for row in raw_target])
    consumers=[]
    for elapsed in (0.,.5,1.,2.):
        evolved=source*np.exp(-1j*elapsed*Ec)[None,:]/math.sqrt(M0)
        amplitude=np.vdot(receiver,evolved)
        consumers.append(dict(time=elapsed,coherent_Ref4_receiver_real=float(amplitude.real),
            coherent_Ref4_receiver_imag=float(amplitude.imag),coherent_Ref4_receiver_probability=float(abs(amplitude)**2),
            evolved_source_norm_squared=float(np.vdot(evolved,evolved).real)))
    Eavg=np.tile(np.array([sum(e)/4 for _,_,e in neutral['physical']]),3)
    static=np.tile(np.array([np.array(native.density(neutral['states'][i][0],neutral['states'][i][1],0))@np.array([0.,-.25,-.5,.25])
        for i,_,_ in neutral['physical']]),3)
    independent_E=Vn.conj().T@((Eavg-static)[:,None]*Vn)
    raw_null=np.array([row-Vc@(Vc.conj().T@row) for row in raw])
    bands=[]
    for threshold in (.25,2.,4.):
        selected=source[:,Ec<=threshold];weight=float(np.vdot(selected,selected).real)
        bands.append(dict(absolute_energy_threshold=threshold,rank=int(np.sum(Ec<=threshold)),source_fraction=weight/M0))
    descriptor=dict(governing_plan_id='issue-222-induced-photon-delta13-continuation-20261008',plan_sha256_at_IMPORT=PLAN_SHA256_AT_IMPORT,
        pinned_general='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        primitive_ABI='NEW-native-two-path-auxiliary-full6-matched-light-tree0-20261008',
        primitive_change='Auxiliary hopping=(U[3]+U[-2,-1,-0])/2 between SAME sites3 and0; mean unchanged, loop retained',
        alternative_path_weights=[.5,.5],integer_auxiliary_EM_charge=1,auxiliary_kappa=.04,auxiliary_sites=[0,3],
        all_auxiliary_orbitals=4,auxiliary_canonical_number=2,all_auxiliary_canonical_states=6,
        active_kappa=.09,light_homotopy=0.,active_canonical_numbers=[4,5],original_Ref=4,prepared_Ref=3,
        integer_active_source_charge=1,integer_Ref_hole_charge=-1,source_and_receiver_packets_unchanged=True,
        source_packet=packet.tolist(),receiver_packet=outpacket.tolist(),source_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        photon_operator='physical sin(theta), independent of light matter homotopy',
        spatial_spacing=1.,transfer_duration=1.,normal_nodes=y.tolist(),normal_sigma=sigma,normal_radius=radius,
        Haar_quadrature_grid=grid,integer_electric_Galerkin_cutoff=12,no_bare_gauge_action=True,
        absolute_source_band_thresholds=[.25,2.,4.],common_neutral_normalization=True,
        operator_contract='Xi_ph=sqrt(KN/lambdaN) sin(theta) Omega/sqrtZph; Xi_ch=sqrt(KC/lambdaN) S_literal Xi_ph. M0=norm²Xi_ch; E1=<Xi_ch,-log(KC/lambdaN) Xi_ch>; R1=<Xi_ch,KC/lambdaN Xi_ch>.')
    return dict(input_descriptor=descriptor,input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},physical_mean_descriptor=metric,
        actual_auxiliary_mean_equals_original_mean_defect=mean_error,actual_integer_path_gauge_covariance_defect=gauge_error,
        actual_positive_auxiliary_half_gauge_covariance_defect=half_gauge_error,
        actual_positive_range_physical_sin_theta_norm=float(np.linalg.norm(positive_curl,2)),
        actual_positive_range_Etheta_minus_matter_density_connection_norm=float(np.linalg.norm(independent_E,2)),
        actual_neutral_photon_Zph=Zph,actual_common_neutral_lambda_N=lambdaN,
        actual_source_M0_including_whole_charged_halfstep=M0,actual_energy_moment_Htransfer=E1,
        actual_transfer_moment_Kcharged_over_lambda_neutral=R1,actual_energy_mean=E1/M0,actual_transfer_Rayleigh_mean=R1/M0,
        actual_neutral_photon_energy_mean=float(np.sum(abs(incoming_coefficients)**2*En)),
        actual_raw_source_Knull_norm_squared=float(np.vdot(raw_null,raw_null).real),
        actual_charged_absolute_energy_band_rows=bands,actual_same_literal_source_consumer_rows=consumers,
        all_canonical_positive_ranks={'neutral':Vn.shape[1],'charged_per_Ref':Vc.shape[1]},
        sampled_Fourier_norm_above_integer12={'neutral':neutral['sampled_Fourier_norm_above12'],'charged':charged['sampled_Fourier_norm_above12']},
        source_or_quartercut_retrofitted=False,original_single_edge_auxiliary_or_direct_t1_capsule_retagged=False,
        massless_photon_or_common_cone_or_macro_source_admitted=False,
        scope='Actual distinct TWO-PATH positive quantum primitive with ALL6aux/Phi3/Ref4 and fixed Gauss, photon source and consumer. Finite Haar/Galerkin calculation; not original4D Wilson law, fullCartesianPhi/SU4 or macro full125/pole admission.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--haar-grid',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    start=time.monotonic();result=run(args.haar_grid);result['runtime_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256',
        'actual_auxiliary_mean_equals_original_mean_defect','actual_positive_range_physical_sin_theta_norm',
        'actual_positive_range_Etheta_minus_matter_density_connection_norm','actual_neutral_photon_Zph',
        'actual_source_M0_including_whole_charged_halfstep','actual_transfer_Rayleigh_mean',
        'actual_same_literal_source_consumer_rows','runtime_seconds')}))


if __name__=='__main__':
    main()
