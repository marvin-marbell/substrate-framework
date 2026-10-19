#!/usr/bin/env python3
"""NEW real shared-edge two-face quantum primitive, not a product control.

The SAME positive two-site B/F auxiliary halfstep now uses THREE equal
integer paths3->0: shared edge, left three-edge path, right three-edge path.
Their mean halfstep is unchanged. One total-density Gauss acts after the
full light canonical4/5 and auxiliary canonical2 matrices, all6 states.
Actual face q-packets, neutral transfer spectra, retarded cross-face kernel
and SAME literal source/Ref4/receiver are executed, with no imported speed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import scipy
from scipy.linalg import block_diag,eigh,svd
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parent
SOURCE_NAMES=('covariant_shared_plaquette_photon.py','covariant_native_collective_two_auxiliary.py',
    'covariant_native_closed_auxiliary_loop.py','covariant_native_source_gram_moments.py',
    'covariant_native_haar_range_jets.py','covariant_source_band_halfstep_control.py',
    'covariant_induced_photon.py','covariant_photon_energy_control.py','sea_selection.py','rotation_band.py')
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in SOURCE_NAMES}
PLAN_PATH=Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA256_AT_IMPORT=hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
from covariant_native_closed_auxiliary_loop import auxiliary_path_half
from covariant_native_source_gram_moments import matched_tree_half
from covariant_native_collective_two_auxiliary import positive_range,physical_wave,project
import covariant_native_haar_range_jets as native
from covariant_induced_photon import Parameters,phi_frame


def graph() -> tuple[np.ndarray,np.ndarray]:
    incidence=np.zeros((6,7),int)
    for edge,(x,y) in enumerate(((0,1),(1,2),(2,3),(3,0),(3,4),(4,5),(5,0))):
        incidence[x,edge]=1;incidence[y,edge]=-1
    cycles=np.array([[1,0],[1,0],[1,0],[1,1],[0,-1],[0,-1],[0,-1]],int)
    if np.any(incidence@cycles):raise ArithmeticError('Actual shared-face cycle incidence failed')
    return incidence,cycles


def phase_from_links(links: np.ndarray) -> complex:
    return complex((np.exp(1j*links[3])+np.exp(-1j*np.sum(links[:3]))+np.exp(1j*np.sum(links[4:])))/3)


def space(charge: int) -> dict[str,Any]:
    light=list(itertools.combinations(range(8),4+charge));aux=list(itertools.combinations(range(4),2))
    states=list(itertools.product(light,aux));rho=np.array([(*native.density(l,a,charge),0,0) for l,a in states])
    middle=np.flatnonzero(np.all(rho==0,axis=1));incidence,_=graph();physical=[]
    for state,density in enumerate(rho):
        for kL in range(-12,13):
            for kR in range(-12,13):
                electric=(kL+int(density[0]),kL+int(density[0]+density[1]),
                    kL+int(density[0]+density[1]+density[2]),kL+kR,-kR,-kR,-kR)
                if max(abs(e) for e in electric)>12:continue
                if np.any(incidence@np.array(electric)!=density):raise ArithmeticError('Actual six-node total Gauss failed')
                physical.append((state,kL,kR,electric))
    return dict(charge=charge,light=light,auxiliary=aux,states=states,rho=rho,middle=middle,
        physical=physical,index={(s,kL,kR):j for j,(s,kL,kR,_) in enumerate(physical)},
        state_index={s:j for j,s in enumerate(states)})


def canonical_caches(gamma: np.ndarray,physical: dict[str,Any],grid: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    L=np.array(physical['light']);A=np.array(physical['auxiliary']);angles=2*math.pi*np.arange(grid)/grid
    light=[];heavy=np.zeros((grid,grid,6,6),complex)
    for thetaL in angles:
        half,pref=matched_tree_half(gamma,float(thetaL),0.)
        light.append(pref*np.linalg.det(half[L[:,None,:,None],L[None,:,None,:]]))
    for i,thetaL in enumerate(angles):
        for j,thetaR in enumerate(angles):
            phase=complex((np.exp(1j*thetaL)+1+np.exp(1j*(thetaL-thetaR)))/3)
            half,pref=auxiliary_path_half(gamma,.04,phase)
            heavy[i,j]=pref*np.linalg.det(half[A[:,None,:,None],A[None,:,None,:]])
    lookupL={s:j for j,s in enumerate(physical['light'])};lookupA={s:j for j,s in enumerate(physical['auxiliary'])}
    trip=np.array([(lookupL[l],lookupA[a]) for l,a in physical['states']])
    return np.array(light),heavy,trip


def columns(gamma: np.ndarray,y: np.ndarray,physical: dict[str,Any],grid: int) -> dict[str,Any]:
    caches=[]
    for normal in y:
        actual=gamma.copy();actual[1]*=1+float(normal)
        caches.append(canonical_caches(actual,physical,grid))
    L,H,trip=canonical_caches(gamma,physical,1);c=trip[physical['middle']]
    mean=L[0,trip[:,0,None],c[None,:,0]]*H[0,0,trip[:,1,None],c[None,:,1]]
    _,s,Vh=svd(mean,full_matrices=False,lapack_driver='gesvd')
    if s[-1]<=0:raise ArithmeticError('Actual same-mean columns lost rank')
    pre=Vh.conj().T/s[None,:]
    rows=np.array([state for state,_,_,_ in physical['physical']])
    fluxL=np.array([kL%grid for _,kL,_,_ in physical['physical']]);fluxR=np.array([kR%grid for _,_,kR,_ in physical['physical']])
    A=np.zeros((3,len(rows),len(s)),complex);tail=0.
    for lo in range(0,len(physical['states']),48):
        hi=min(lo+48,len(physical['states']));selected=np.flatnonzero((rows>=lo)&(rows<hi));r=trip[lo:hi]
        for p,(light,heavy,_) in enumerate(caches):
            raw=light[:,None,r[:,0,None],c[None,:,0]]*heavy[:,:,r[:,1,None],c[None,:,1]]
            values=raw@pre;coeff=np.fft.fft2(values,axes=(0,1))/(grid*grid)
            A[p,selected]=coeff[fluxL[selected],fluxR[selected],rows[selected]-lo]
            high=np.array([min(j,grid-j)>12 for j in range(grid)])
            tail+=float(np.linalg.norm(coeff[high])**2+np.linalg.norm(coeff[:,high])**2)
    return dict(A=A,P=block_diag(*([s[:,None]*Vh]*3)),mean_singular_values=s,
        mean_preconditioned_Fourier_norm_outside_coordinate12=math.sqrt(tail))


def curl(physical: dict[str,Any],axis: int) -> Any:
    rows=[];columns=[];values=[];size=len(physical['physical'])
    for column,(state,kL,kR,_) in enumerate(physical['physical']):
        for shift,amplitude in ((1,1/(2j)),(-1,-1/(2j))):
            target=physical['index'].get((state,kL+shift*(axis==0),kR+shift*(axis==1)))
            if target is not None:rows.append(target);columns.append(column);values.append(amplitude)
    return coo_matrix((values,(rows,columns)),shape=(size,size)).tocsr()


def apply_field(operator: Any,wave: np.ndarray) -> np.ndarray:
    return (operator@wave.reshape(3,-1).T).T.reshape(-1)


def source(physicalN: dict[str,Any],physicalC: dict[str,Any],coefficients: np.ndarray,embedding: np.ndarray) -> list[Any]:
    nn=len(physicalN['physical']);nc=len(physicalC['physical']);weights=coefficients@embedding.conj().T
    paths=np.zeros((4,7),int);paths[1,0]=-1;paths[2,:2]=-1;paths[3,3]=1
    incidence,_=graph();expected=np.zeros((6,4),int)
    for x in range(4):expected[x,x]+=1;expected[0,x]-=1
    if np.any(incidence@paths.T!=expected):raise ArithmeticError('Actual global literal source path charge failed')
    rows=[];cols=[];signed_modes=[]
    for column,(state,kL,kR,electric) in enumerate(physicalN['physical']):
        occupied,aux=physicalN['states'][state];mask=sum(1<<m for m in occupied)
        for mode in range(8):
            if mode in occupied:continue
            target=physicalC['state_index'][(tuple(sorted((*occupied,mode))),aux)]
            row=physicalC['index'].get((target,kL+(mode//2==3),kR))
            if row is None:continue
            target_e=np.array(electric)+paths[mode//2]
            if np.any(np.array(physicalC['physical'][row][3])!=target_e):raise ArithmeticError('Actual global source flux differs from literal integer path')
            below=(mask&((1<<mode)-1)).bit_count();rows.append(row);cols.append(column);signed_modes.append((mode,(-1)**below))
    modes=np.array([m for m,_ in signed_modes]);signs=np.array([s for _,s in signed_modes])
    return [coo_matrix(((-1)**ref*signs*weights[modes,ref],(rows,cols)),shape=(nc,nn)).tocsr() for ref in range(4)]


def apply_source(operators: list[Any],wave: np.ndarray) -> np.ndarray:
    return np.array([(operator@wave.reshape(3,-1).T).T.reshape(-1) for operator in operators])


def quantum_loads(wave: np.ndarray,physical: dict[str,Any],Hphi: np.ndarray) -> dict[str,Any]:
    w=wave.reshape(-1,3,len(physical['physical']));norm=float(np.vdot(w,w).real)
    if norm<=0:raise ArithmeticError('Cannot normalize absent quantum loads')
    prob=np.sum(abs(w)**2,axis=0)/norm;row=prob.sum(axis=0);nodes=prob.sum(axis=1);loads=np.zeros(6)
    lookup={a:i for i,a in enumerate(physical['auxiliary'])}
    for weight,(state,_,_,_) in zip(row,physical['physical']):loads[lookup[physical['states'][state][1]]]+=weight
    return dict(norm_squared=norm,Phi_node_probabilities=nodes.tolist(),
        Phi_heat_energy=float(np.einsum('cnr,nm,cmr->',w.conj(),Hphi,w).real/norm),
        all6_auxiliary_occupations=[list(a) for a in physical['auxiliary']],all6_auxiliary_probabilities=loads.tolist(),
        actual_shared_edge_electric_square_mean=float(row@np.array([electric[3]**2 for _,_,_,electric in physical['physical']])))


def run(grid: int) -> dict[str,Any]:
    gamma4,metric=phi_frame(Parameters(kappa_power=400));ev,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,ev>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    incidence,cycles=graph();links=np.array([.17,-.23,.31,.47,-.19,.27,.41]);chi=np.array([.13,-.37,.29,.41,-.31,.23])
    phase=phase_from_links(links);changed=phase_from_links(links+incidence.T@chi)
    phase_defect=float(abs(changed-np.exp(1j*(chi[3]-chi[0]))*phase))
    mean,mp=auxiliary_path_half(gamma,.04,1+0j);old,op=native.auxiliary_half(gamma,.04,0.)
    mean_defect=float(np.linalg.norm(mean-old,2)+abs(mp-op))
    half,_=auxiliary_path_half(gamma,.04,phase);transformed,_=auxiliary_path_half(gamma,.04,changed)
    G=np.diag(np.repeat(np.exp(1j*chi[[0,3]]),2));half_gauge_defect=float(np.linalg.norm(transformed-G@half@G.conj().T,2))
    if max(mean_defect,phase_defect,half_gauge_defect)>1e-12:raise ArithmeticError('Actual threepath mean/gauge primitive failed')
    y=np.array([-.09,0.,.09]);sigma=.03;radius=.12;dy=.09
    logk=-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y));k=np.exp(logk);k/=np.linalg.norm(k)
    D=np.zeros((2,3))
    for j in range(2):D[j,j]=math.sqrt(k[j+1]/k[j])/dy;D[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    Hphi=2*sigma*sigma*D.T@D;ev,U=eigh(Hphi);heat=(U*np.exp(-ev/2))@U.T
    neutral=space(0);charged=space(1)
    print(json.dumps(dict(stage='constructing_actual_connected_fullcanonical_columns',grid=grid,
        neutral_rows=len(neutral['physical']),charged_rows=len(charged['physical']),neutral_middle=len(neutral['middle']),charged_middle=len(charged['middle']))),flush=True)
    n=columns(gamma,y,neutral,grid);c=columns(gamma,y,charged,grid)
    positive_range(n,heat);positive_range(c,heat)
    lambdaN=float(n['q'][0]**2);qn=n['q']/math.sqrt(lambdaN);qc=c['q']/math.sqrt(lambdaN)
    En=-2*np.log(qn);Ec=-2*np.log(qc);unit=np.zeros(len(qn),complex);unit[0]=1
    Omega=physical_wave(n,heat,unit)
    BL=curl(neutral,0);BR=curl(neutral,1);CL=curl(charged,0);CR=curl(charged,1)
    rawL=apply_field(BL,Omega);rawR=apply_field(BR,Omega);bL=project(n,heat,rawL);bR=project(n,heat,rawR)
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet);outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    S=source(neutral,charged,native.source_coefficients(gamma,packet),embedding)
    Sout=source(neutral,charged,native.source_coefficients(gamma,outpacket),embedding)
    raw_target=apply_source(Sout,Omega);cases=[]
    numerical_guard=64*np.finfo(float).eps*math.sqrt(n['A'].shape[1]*len(qn))
    for q in (0.,math.pi):
        mode=np.exp(1j*q);b=(bL+mode*bR)/math.sqrt(2);incoming_coeff=qn*b
        Zph=float(np.vdot(incoming_coeff,incoming_coeff).real)
        raw=(rawL+mode*rawR)/math.sqrt(2);projected=physical_wave(n,heat,b)
        photon_null=float(np.vdot(raw-projected,raw-projected).real)
        row:dict[str,Any]=dict(actual_face_packet_q=q,q_is_packet_label_not_translation_eigenvalue=True,
            actual_physical_photon_Zph=Zph,actual_raw_photon_Knull_norm_squared=photon_null,
            numerical_resolution_amplitude_guard=numerical_guard,guard_is_conservative_heuristic_not_certified_error_bound=True,
            actual_neutral_spectral_source_rows=[dict(energy=float(e),unnormalized_prepared_weight=float(w)) for e,w in zip(En,abs(incoming_coeff)**2)])
        if math.sqrt(Zph)<=numerical_guard:
            row.update(actual_normalized_photon_source_consumer_resolved=False,
                unresolved_reason='Prepared physical q-packet amplitude is below conservative numerical resolution; no cheap normalized photon or substitute state manufactured',
                actual_M0=None,actual_E1=None,actual_R1=None,actual_consumer=None)
            cases.append(row);continue
        incoming_coeff/=math.sqrt(Zph);incoming=physical_wave(n,heat,incoming_coeff)
        rawsource=apply_source(S,incoming);projected_source=np.array([project(c,heat,w) for w in rawsource]);Xi=qc[None,:]*projected_source
        M0=float(np.vdot(Xi,Xi).real)
        if M0<=0:raise ArithmeticError('Actual connected charged source absent')
        E1=float(np.sum(abs(Xi)**2*Ec[None,:]));R1=float(np.sum(abs(Xi)**2*qc[None,:]**2))
        source_null=np.array([w-physical_wave(c,heat,p) for w,p in zip(rawsource,projected_source)])
        outgoing=np.array([(apply_field(CL,w)+mode*apply_field(CR,w))/math.sqrt(2) for w in raw_target])
        receiver=np.array([project(c,heat,w) for w in outgoing]);consumer=[]
        for elapsed in (0.,.5,1.,2.):
            evolved=Xi*np.exp(-1j*elapsed*Ec)[None,:]/math.sqrt(M0);amplitude=np.vdot(receiver,evolved)
            consumer.append(dict(time=elapsed,coherent_Ref4_receiver_real=float(amplitude.real),coherent_Ref4_receiver_imag=float(amplitude.imag),
                coherent_Ref4_receiver_probability=float(abs(amplitude)**2),actual_evolved_source_norm_squared=float(np.vdot(evolved,evolved).real)))
        sourcewave=np.array([physical_wave(c,heat,z) for z in Xi])
        row.update(actual_normalized_photon_source_consumer_resolved=True,actual_M0=M0,actual_E1=E1,actual_R1=R1,
            actual_energy_mean=E1/M0,actual_transfer_Rayleigh_mean=R1/M0,
            actual_neutral_photon_energy_mean=float(np.sum(abs(incoming_coeff)**2*En)),
            actual_raw_source_Knull_norm_squared=float(np.vdot(source_null,source_null).real),actual_consumer=consumer,
            actual_charged_source_band_rows=[dict(absolute_threshold=t,source_fraction=float(np.sum(abs(Xi[:,Ec<=t])**2)/M0)) for t in (.25,2.,4.)],
            actual_quantum_photon_loads=quantum_loads(incoming,neutral,Hphi),actual_quantum_source_loads=quantum_loads(sourcewave,charged,Hphi))
        cases.append(row)
    cross=[]
    for elapsed in (0.,.5,1.,2.):
        spectral=np.vdot(bL,bR*np.exp(-1j*elapsed*En))
        cross.append(dict(time=elapsed,actual_positive_transfer_sector_retarded_cross_face_curl=2*float(spectral.imag),
            actual_positive_transfer_sector_spectral_cross_real=float(spectral.real),actual_positive_transfer_sector_spectral_cross_imag=float(spectral.imag)))
    descriptor=dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',plan_sha256_at_IMPORT=PLAN_SHA256_AT_IMPORT,
        pinned_general='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        primitive='NEW two neighboring plaquettes sharing oriented3->0 edge; same positive B/F aux half, three equal integer paths3->0',
        physical_graph_edges=[[0,1],[1,2],[2,3],[3,0],[3,4],[4,5],[5,0]],physical_cycle_columns=cycles.tolist(),
        integer_auxiliary_paths=[[0,0,0,1,0,0,0],[-1,-1,-1,0,0,0,0],[0,0,0,0,1,1,1]],path_weights=[1/3]*3,
        mean_halfstep_equals_frozen_c6e5_mean=True,no_bare_gauge_action=True,no_speed_mass_fit=True,
        gauge='long treeA0=A1=A2=A4=A5=0; A3=thetaL, A6=thetaL-thetaR; TWO independent physical holonomies',
        Fourier_electric_contract='kL conjugate thetaL,kR conjugate thetaR; E=(kL+rho0,kL+rho0+rho1,kL+rho0+rho1+rho2,kL+kR,-kR,-kR,-kR)',
        one_global_Gauss_after_full_active_and_auxiliary_canonical_matrices=True,auxiliary_all6_states=True,
        active_orbitals=8,active_canonical_numbers=[4,5],auxiliary_orbitals=4,auxiliary_canonical_number=2,
        active_sites=[0,1,2,3],auxiliary_sites=[0,3],remaining_geometric_nodes=[4,5],remaining_node_matter_charge=0,
        original_Ref=4,prepared_Ref=3,Ref_site=0,integer_active_aux_charges=[1,1],integer_Refhole_charge=-1,
        active_kappa=.09,auxiliary_kappa=.04,light_homotopy=0.,source_packet=packet.tolist(),receiver_packet=outpacket.tolist(),
        source_paths=[[0]*7,[-1,0,0,0,0,0,0],[-1,-1,0,0,0,0,0],[0,0,0,1,0,0,0]],
        spatial_face_packet_q=[0.,math.pi],q_not_translation_eigenvalue_due_tagged_left_geometry=True,
        source_absolute_energy_thresholds=[.25,2.,4.],Haar_grid_each_physical_cycle=grid,integer_electric_cutoff_ALL7edges=12,
        Phi_nodes=y.tolist(),Phi_sigma=sigma,Phi_radius=radius,transfer_duration=1.,spatial_spacing=1.,
        common_neutral_lambdaN=True,operator_contract='Xi_ph=sqrt(KN/lambdaN) Bq Omega/sqrtZph; Xi_ch=sqrt(KC/lambdaN) S_literal Xi_ph; M0=norm²Xi_ch; E1=<Xi,Htransfer Xi>; R1=<Xi,KC/lambdaN Xi>. ONE whole normalized charged halfstep pervector.',
        no_source_or_Pe_cut_or_packet_retrofit=True,scope='Finite Haar/electricGalerkin, native invariant2spin/Phi3 conditionalmean; real connected shared-edge mechanism, no continuum dispersion/commoncone/macro photon admission')
    return dict(input_descriptor=descriptor,input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        runtime_packages=dict(python=sys.version,python_executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        physical_mean_descriptor=metric,actual_mean_halfstep_defect=mean_defect,actual_integer_path_gauge_defect=phase_defect,actual_auxiliary_half_gauge_defect=half_gauge_defect,
        actual_common_neutral_lambdaN=lambdaN,actual_positive_ranks=[len(qn),len(qc)],actual_declared_middle_counts=[len(neutral['middle']),len(charged['middle'])],
        actual_positive_range_diagnostics={'neutral':n['diagnostics'],'charged':c['diagnostics']},
        actual_coordinate_Fourier_tail12={'neutral':n['mean_preconditioned_Fourier_norm_outside_coordinate12'],'charged':c['mean_preconditioned_Fourier_norm_outside_coordinate12']},
        actual_neutral_quantum_ground_loads=quantum_loads(Omega,neutral,Hphi),actual_shared_face_source_cases=cases,
        actual_retarded_cross_face_spectral_rows=cross,massless_photon_or_common_cone_or_macro_source_admitted=False,
        original_frozen_source_laws_retagged=False)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--haar-grid',type=int,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    start=time.monotonic();result=run(args.haar_grid);result['runtime_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('input_descriptor_sha256','source_at_import','actual_common_neutral_lambdaN',
        'actual_positive_ranks','actual_retarded_cross_face_spectral_rows','runtime_seconds')}))


if __name__=='__main__':
    main()
