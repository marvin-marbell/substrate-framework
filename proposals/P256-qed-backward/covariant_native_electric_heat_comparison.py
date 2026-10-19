#!/usr/bin/env python3
"""NEW physical electric heat vs its Haar-E0 endpoint, all canonical columns.

This is an explicit primitive-electric INPUT comparison, not an induced-
electric or supplied-Maxwell explanation. Same local c6e5 two-path SF,
mean, Phi3, full70/56 x all6 auxiliary, ONE Gauss, Ref4/source/receiver.
Electric regulator is declared anew, NEVER retagged as historical Ecut12.
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
import numpy as np
import scipy
from scipy.linalg import eigh,svd

ROOT=Path(__file__).resolve().parent
SOURCE_NAMES=('covariant_native_electric_heat_comparison.py','covariant_native_closed_auxiliary_loop.py',
 'covariant_native_source_gram_moments.py','covariant_native_haar_range_jets.py',
 'covariant_source_band_halfstep_control.py','covariant_induced_photon.py',
 'covariant_photon_energy_control.py','sea_selection.py','rotation_band.py')
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in SOURCE_NAMES}
PLAN_PATH=Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA256_AT_IMPORT=hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
from covariant_native_closed_auxiliary_loop import auxiliary_path_half
from covariant_native_source_gram_moments import matched_tree_half
from covariant_induced_photon import Parameters,phi_frame
import covariant_native_haar_range_jets as native


def full_factor(gamma: np.ndarray,y: np.ndarray,heat: np.ndarray,physical: dict[str,Any],grid: int,UE: float) -> dict[str,Any]:
    light=np.array(physical['light']);aux=np.array(physical['auxiliary'])
    rows=np.array([state for state,_,_ in physical['physical']]);n=np.array([flux for _,flux,_ in physical['physical']])
    energy=UE*np.array([sum(e*e for e in electric) for _,_,electric in physical['physical']])
    electric_half=np.exp(-energy/2)
    if np.any(electric_half<=0):raise ArithmeticError('Declared physical electric heat numerically underflowed; no null clipping allowed')
    nr=len(rows);F=np.zeros((3*nr,3*nr),complex);lower=[];hermitian=[]
    for p,normal in enumerate(y):
        actual=gamma.copy();actual[1]*=1+float(normal);samples=[];point_lower=[]
        for theta in 2*math.pi*np.arange(grid)/grid:
            half,pref=matched_tree_half(actual,float(theta),0.)
            L=pref*np.linalg.det(half[light[:,None,:,None],light[None,:,None,:]])
            half,pref=auxiliary_path_half(actual,.04,complex((1+np.exp(1j*theta))/2))
            A=pref*np.linalg.det(half[aux[:,None,:,None],aux[None,:,None,:]])
            point_lower.append(float(eigh(L,eigvals_only=True,subset_by_index=[0,0])[0]*eigh(A,eigvals_only=True,subset_by_index=[0,0])[0]))
            samples.append(np.kron(L,A))
        samples=np.array(samples);coeff=np.fft.fft(samples,axis=0)/grid
        SF=coeff[(n[:,None]-n[None,:])%grid,rows[:,None],rows[None,:]]
        hermitian.append(float(np.linalg.norm(SF-SF.conj().T)/np.linalg.norm(SF)))
        # Strict positivity follows from sample SFtheta>0 and orthonormal
        # field columns; retain full canonical means, not a Slater factor.
        lower.append(min(point_lower))
        for out in range(3):F[out*nr:(out+1)*nr,p*nr:(p+1)*nr]=heat[out,p]*SF*electric_half[None,:]
    middle=[j for j,(_,_,e) in enumerate(physical['physical']) if all(x==0 for x in e)]
    middle_columns=np.concatenate([np.array(middle)+p*nr for p in range(3)])
    # Eheat→P_E0 strongly. In ONE global Gauss, E0 forces TOTAL rho0;
    # extracting these same complete columns is the exact Haar endpoint.
    Haar=F[:,middle_columns].copy()
    return dict(F=F,Haar=Haar,actual_Eheat_min=float(electric_half.min()),
        actual_point_fullcanonical_SF_positive_lower_bounds=lower,
        actual_projected_SF_hermitian_relative_defects=hermitian,
        actual_complete_physical_columns=3*nr,actual_totalrho0_Haar_columns=len(middle_columns),
        physical_E_energy_max=float(energy.max()))


def prepare_spectrum(factor: np.ndarray) -> dict[str,Any]:
    V,q,_=svd(factor,full_matrices=False,lapack_driver='gesdd',overwrite_a=True)
    if q[-1]<=0:raise ArithmeticError('Actual positive factor lost a singular direction')
    return dict(V=V,q=q)


def observe(n: dict[str,Any],c: dict[str,Any],neutral: dict[str,Any],charged: dict[str,Any],S: Any,Sout: Any) -> dict[str,Any]:
    Vn=n['V'];Vc=c['V'];lambdaN=float(n['q'][0]**2);qn=n['q']/math.sqrt(lambdaN);qc=c['q']/math.sqrt(lambdaN)
    En=-2*np.log(qn);Ec=-2*np.log(qc);Omega=Vn[:,0]
    Bn=native.curl(neutral,3);Bc=native.curl(charged,3)
    rawph=Bn@Omega;projection=Vn.conj().T@rawph;ph=qn*projection;Zph=float(np.vdot(ph,ph).real)
    if Zph<=0:raise ArithmeticError('Actual physical photon preparation zero')
    ph/=math.sqrt(Zph);incoming=Vn@ph
    raw=(S@incoming).reshape(4,Vc.shape[0]);projected=np.array([Vc.conj().T@row for row in raw]);Xi=qc[None,:]*projected
    M0=float(np.vdot(Xi,Xi).real)
    if M0<=0:raise ArithmeticError('Actual physical electric comparison source zero')
    E1=float(np.sum(abs(Xi)**2*Ec[None,:]));R1=float(np.sum(abs(Xi)**2*qc[None,:]**2))
    source_null=np.array([row-Vc@column for row,column in zip(raw,projected)])
    target=(Sout@Omega).reshape(4,Vc.shape[0]);receiver=np.array([Vc.conj().T@(Bc@row) for row in target]);consumers=[]
    for elapsed in (0.,.5,1.,2.):
        evolved=Xi*np.exp(-1j*elapsed*Ec)[None,:]/math.sqrt(M0);amplitude=np.vdot(receiver,evolved)
        consumers.append(dict(time=elapsed,coherent_Ref4_receiver_real=float(amplitude.real),coherent_Ref4_receiver_imag=float(amplitude.imag),
            coherent_Ref4_receiver_probability=float(abs(amplitude)**2),evolved_source_norm_squared=float(np.vdot(evolved,evolved).real)))
    return dict(actual_common_neutral_lambdaN=lambdaN,actual_positive_ranks=[len(qn),len(qc)],
        actual_normalized_min_positive_singular_values=[float(qn[-1]),float(qc[-1])],
        actual_photon_Zph=Zph,actual_photon_energy_mean=float(np.sum(abs(ph)**2*En)),
        actual_photon_spectral_rows=[dict(energy=float(e),normalized_weight=float(w)) for e,w in zip(En,abs(ph)**2)],
        actual_M0=M0,actual_E1=E1,actual_R1=R1,actual_energy_mean=E1/M0,actual_transfer_Rayleigh_mean=R1/M0,
        actual_raw_photon_Knull_norm_squared=float(np.vdot(rawph-Vn@projection,rawph-Vn@projection).real),
        actual_raw_source_Knull_norm_squared=float(np.vdot(source_null,source_null).real),
        actual_charged_source_absolute_band_rows=[dict(absolute_threshold=t,source_fraction=float(np.sum(abs(Xi[:,Ec<=t])**2)/M0)) for t in (.25,2.,4.)],
        actual_same_literal_source_consumer_rows=consumers,
        actual_neutral_phi_node_probabilities=np.sum(abs(Omega.reshape(3,-1))**2,axis=1).tolist())


def run(cutoff: int,grid: int,UE: float) -> dict[str,Any]:
    gamma4,metric=phi_frame(Parameters(kappa_power=400));ev,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,ev>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    y=np.array([-.09,0.,.09]);sigma=.03;radius=.12;dy=.09
    logk=-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y));k=np.exp(logk);k/=np.linalg.norm(k)
    D=np.zeros((2,3))
    for j in range(2):D[j,j]=math.sqrt(k[j+1]/k[j])/dy;D[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    ev,U=eigh(2*sigma*sigma*D.T@D);heat=(U*np.exp(-ev/2))@U.T
    neutral=native.space(0,cutoff);charged=native.space(1,cutoff)
    print(json.dumps(dict(stage='constructing_FULL_CANONICAL_ELECTRIC_HEAT_columns',neutral_dimension=3*len(neutral['physical']),charged_dimension=3*len(charged['physical']),cutoff=cutoff,UE=UE)),flush=True)
    n=full_factor(gamma,y,heat,neutral,grid,UE);c=full_factor(gamma,y,heat,charged,grid,UE)
    diagnostics={label:{key:value for key,value in data.items() if key not in ('F','Haar')} for label,data in (('neutral',n),('charged',c))}
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet);outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    S=native.source(neutral,charged,native.source_coefficients(gamma,packet),embedding,3)
    Sout=native.source(neutral,charged,native.source_coefficients(gamma,outpacket),embedding,3)
    haar=observe(prepare_spectrum(n.pop('Haar')),prepare_spectrum(c.pop('Haar')),neutral,charged,S,Sout)
    print(json.dumps(dict(stage='executing_FULL_POSITIVE_physical_electric_spectrum',Haar_photon_energy=haar['actual_photon_energy_mean'])),flush=True)
    positiveN=prepare_spectrum(n.pop('F'));positiveC=prepare_spectrum(c.pop('F'))
    physical=observe(positiveN,positiveC,neutral,charged,S,Sout)
    descriptor=dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',plan_sha256_at_IMPORT=PLAN_SHA256_AT_IMPORT,
        primitive='NEW declared physical electric heat comparison, SAME c6e5 local two-path SF; not induced-electric explanation',
        primitive_electric_input=UE,primitive_electric_H='UE sum_ALL4edges Pi_e² with SAME total charged+aux+Ref Gauss; longitudinal/Coulomb load retained',
        positive_transfer_factor='F_U=normalheat_half SF_half exp(-UE sum Pi_e²/2) in ONE physical globalGauss space; K_U=F_U F_Udag',
        Haar_endpoint='UE→infinity gives P_E0, totalrho0 middle and exactly original Haar construction AT THIS declared regulator; no oldE12 retag',
        full_active_canonical_numbers=[4,5],full_active_CAR_states=[70,56],all_auxiliary_canonical_number=2,all_auxiliary_CAR_states=6,
        original_Ref=4,prepared_Ref=3,static_Ref_site=0,integer_active_aux_charges=[1,1],integer_Ref_hole_charge=-1,
        active_kappa=.09,auxiliary_kappa=.04,light_homotopy=0.,two_auxiliary_integer_path_weights=[.5,.5],
        source_packet=packet.tolist(),receiver_packet=outpacket.tolist(),source_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        source_absolute_energy_thresholds=[.25,2.,4.],source_operator_and_Pe_not_retrofitted=True,
        Phi_nodes=y.tolist(),Phi_sigma=sigma,Phi_radius=radius,Haar_quadrature_grid=grid,
        NEW_declared_electric_Galerkin_cutoff=cutoff,electric_regulator_is_NOT_historical12=True,
        transfer_duration=1.,spatial_spacing=1.,common_neutral_normalization=True,
        operator_contract='Xi_ph=sqrt(KN/lambdaN) sin(theta) Omega/sqrtZph; Xi_ch=sqrt(KC/lambdaN) S_literal Xi_ph. M0=norm²Xi_ch; E1=<Xi,-log(KC/lambdaN)Xi>; R1=<Xi,KC/lambdaN Xi>; ONE whole charged halfstep pervector.',
        fullrank_scope='ALL declared finite electric/Pauli/Phi columns; electric_heat and fullprojected pointSF andnormalheat strictlypositive. No untruncated EM/fullCartesianPhi/SU4/macro or massless admission',
        no_bare_magnetic_action=True,no_imported_Maxwell_kernel=True,no_speed_mass_quarter_fit=True)
    return dict(input_descriptor=descriptor,input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        runtime_packages=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        physical_mean_descriptor=metric,actual_fullcolumn_factor_diagnostics=diagnostics,
        actual_same_regulator_Haar_endpoint=haar,actual_primitive_electric_heat_transfer=physical,
        induced_electric_explanation_or_massless_commoncone_macro_source_admitted=False)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--electric-cutoff',type=int,required=True);parser.add_argument('--haar-grid',type=int,required=True)
    parser.add_argument('--primitive-electric-input',type=float,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    start=time.monotonic();result=run(args.electric_cutoff,args.haar_grid,args.primitive_electric_input);result['runtime_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256','runtime_seconds')}))
    print(json.dumps({label:{key:result[label][key] for key in ('actual_positive_ranks','actual_photon_Zph','actual_photon_energy_mean','actual_M0','actual_energy_mean','actual_transfer_Rayleigh_mean','actual_raw_source_Knull_norm_squared')} for label in ('actual_same_regulator_Haar_endpoint','actual_primitive_electric_heat_transfer')}))


if __name__=='__main__':
    main()
