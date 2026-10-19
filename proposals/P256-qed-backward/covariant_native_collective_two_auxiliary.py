#!/usr/bin/env python3
"""Exact finite-native two-copy inducing primitive; full canonical 6 x 6.

No occupation pruning. Stream matter blocks into Fourier mean-preconditioned
columns, factor their positive Gram by Cholesky, then SVD the UNSQUARED
positive square factor after undoing the actual mean-column SVD. No eigenvalue
clipping. All range operations use exact small pullbacks, not large left SVDs.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy.linalg import block_diag, cholesky, eigh, solve_triangular, svd
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parent
SOURCE_NAMES=('covariant_native_collective_two_auxiliary.py','covariant_native_closed_auxiliary_loop.py',
 'covariant_native_source_gram_moments.py','covariant_native_haar_range_jets.py',
 'covariant_source_band_halfstep_control.py','covariant_induced_photon.py',
 'covariant_photon_energy_control.py','sea_selection.py','rotation_band.py')
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in SOURCE_NAMES}
PLAN_PATH=Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA256_AT_IMPORT=hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
from covariant_native_closed_auxiliary_loop import auxiliary_path_half
from covariant_native_source_gram_moments import matched_tree_half
import covariant_native_haar_range_jets as native
from covariant_induced_photon import Parameters, phi_frame


def density(light: tuple[int,...],a: tuple[int,...],b: tuple[int,...],charge: int) -> tuple[int,...]:
    rho=[sum(m//2==x for m in light)-1 for x in range(4)]
    for aux in (a,b):
        for x in (0,3):rho[x]+=sum((0 if m<2 else 3)==x for m in aux)-1
    rho[0]-=charge
    if sum(rho)!=0:raise ArithmeticError('Total two-copy/active/static-Ref charge is not zero')
    return tuple(rho)


def space(charge: int) -> dict[str,Any]:
    light=list(itertools.combinations(range(8),4+charge))
    aux=list(itertools.combinations(range(4),2))
    states=list(itertools.product(light,aux,aux))
    middle=[i for i,(l,a,b) in enumerate(states) if density(l,a,b,charge)==(0,0,0,0)]
    physical=[]
    for i,(l,a,b) in enumerate(states):
        rho=density(l,a,b,charge)
        for n in range(-12,13):
            electric=tuple(n+sum(rho[:j+1]) for j in range(4))
            if max(abs(e) for e in electric)<=12:physical.append((i,n,electric))
    return dict(charge=charge,light=light,auxiliary=aux,states=states,middle=middle,physical=physical,
        index={(i,n):j for j,(i,n,_) in enumerate(physical)},state_index={s:i for i,s in enumerate(states)})


def samples(gamma: np.ndarray,physical: dict[str,Any],theta: float,ids: np.ndarray) -> np.ndarray:
    half,pref=matched_tree_half(gamma,theta,0.)
    heavy,hpref=auxiliary_path_half(gamma,.04,complex((1+np.exp(1j*theta))/2))
    L=np.array(physical['light']);A=np.array(physical['auxiliary'])
    light=pref*np.linalg.det(half[L[:,None,:,None],L[None,:,None,:]])
    aux=hpref*np.linalg.det(heavy[A[:,None,:,None],A[None,:,None,:]])
    lookupL={l:i for i,l in enumerate(physical['light'])};lookupA={a:i for i,a in enumerate(physical['auxiliary'])}
    trip=np.array([(lookupL[l],lookupA[a],lookupA[b]) for l,a,b in physical['states']])
    r=trip[ids];c=trip[physical['middle']]
    return light[r[:,0,None],c[None,:,0]]*aux[r[:,1,None],c[None,:,1]]*aux[r[:,2,None],c[None,:,2]]


def columns(gamma: np.ndarray,y: np.ndarray,physical: dict[str,Any],grid: int) -> dict[str,Any]:
    mean=samples(gamma,physical,0.,np.arange(len(physical['states'])))
    _,s,Vh=svd(mean,full_matrices=False,lapack_driver='gesvd')
    if s[-1]<=0:raise ArithmeticError('Actual mean columns lack full rank')
    pre=Vh.conj().T/s[None,:]
    rows=np.array([i for i,_,_ in physical['physical']]);flux=np.array([n%grid for _,n,_ in physical['physical']])
    A=np.zeros((3,len(rows),len(s)),complex);tail=0.
    for lo in range(0,len(physical['states']),64):
        hi=min(lo+64,len(physical['states']));ids=np.arange(lo,hi);selected=np.where((rows>=lo)&(rows<hi))[0]
        for p,normal in enumerate(y):
            actual=gamma.copy();actual[1]*=1+float(normal)
            values=np.array([samples(actual,physical,float(theta),ids)@pre for theta in 2*math.pi*np.arange(grid)/grid])
            coeff=np.fft.fft(values,axis=0)/grid
            A[p,selected,:]=coeff[flux[selected],rows[selected]-lo,:]
            high=[j for j in range(grid) if min(j,grid-j)>12]
            tail+=float(np.linalg.norm(coeff[high])**2)
    return dict(A=A,P=block_diag(*([s[:,None]*Vh]*3)),mean_singular_values=s,
        mean_preconditioned_Fourier_tail_above12_norm=math.sqrt(tail))


def pullback(left: dict[str,Any],right: dict[str,Any],heat: np.ndarray,operator: Any=None) -> np.ndarray:
    A=left['A'];B=right['A'];nl=A.shape[2];nr=B.shape[2]
    metric=heat.T@heat;result=np.zeros((3*nl,3*nr),complex)
    for p in range(3):
        for q in range(3):
            image=B[q] if operator is None else operator@B[q]
            result[p*nl:(p+1)*nl,q*nr:(q+1)*nr]=metric[p,q]*(A[p].conj().T@image)
    return result


def positive_range(data: dict[str,Any],heat: np.ndarray) -> None:
    gram=pullback(data,data,heat)
    asym=float(np.linalg.norm(gram-gram.conj().T)/np.linalg.norm(gram))
    R=cholesky(gram,lower=False)
    small=R@data['P'];u,q,_=svd(small,full_matrices=False,lapack_driver='gesvd')
    data['T']=solve_triangular(R,u,lower=False);data['q']=q
    data['diagnostics']=dict(mean_singular_min=float(data['mean_singular_values'][-1]),
        mean_singular_max=float(data['mean_singular_values'][0]),
        preconditioned_Gram_hermiticity_relative=asym,
        preconditioned_Gram_eigen_min=float(eigh(gram,eigvals_only=True,subset_by_index=[0,0])[0]),
        square_factor_singular_min=float(q[-1]),square_factor_singular_max=float(q[0]),
        orthonormal_range_defect=float(np.linalg.norm(data['T'].conj().T@gram@data['T']-np.eye(len(q)))))
    if q[-1]<=0:raise ArithmeticError('Positive square factor lost a direction')


def physical_wave(data: dict[str,Any],heat: np.ndarray,c: np.ndarray) -> np.ndarray:
    coefficients=(data['T']@c).reshape(3,data['A'].shape[2])
    middle=np.einsum('prm,pm->pr',data['A'],coefficients)
    return (heat@middle).reshape(-1)


def project(data: dict[str,Any],heat: np.ndarray,wave: np.ndarray) -> np.ndarray:
    w=heat.T@wave.reshape(3,data['A'].shape[1])
    raw=np.einsum('prm,pr->pm',data['A'].conj(),w).reshape(-1)
    return data['T'].conj().T@raw


def source(neutral: dict[str,Any],charged: dict[str,Any],coeff: np.ndarray,embedding: np.ndarray) -> Any:
    nn=len(neutral['physical']);nc=len(charged['physical']);paths=((0,0,0,0),(-1,0,0,0),(-1,-1,0,0),(0,0,0,1))
    weights=coeff@embedding.conj().T;rows=[];cols=[];values=[]
    for col,(state,n,electric) in enumerate(neutral['physical']):
        occupied,a,b=neutral['states'][state];mask=sum(1<<m for m in occupied)
        for mode in range(8):
            if mode in occupied:continue
            target=charged['state_index'][(tuple(sorted((*occupied,mode))),a,b)]
            flux=tuple(x+z for x,z in zip(electric,paths[mode//2]));row=charged['index'].get((target,flux[3]))
            if row is None:continue
            if charged['physical'][row][2]!=flux:raise ArithmeticError('Literal source integer Gauss path failed')
            rho=density(tuple(sorted((*occupied,mode))),a,b,1)
            if any(flux[x]-flux[(x-1)%4]!=rho[x] for x in range(4)):raise ArithmeticError('Local multicopy Gauss source failed')
            below=(mask&((1<<mode)-1)).bit_count()
            for ref in range(4):
                for node in range(3):
                    rows.append((ref*3+node)*nc+row);cols.append(node*nn+col);values.append((-1)**(below+ref)*weights[mode,ref])
    return coo_matrix((values,(rows,cols)),shape=(12*nc,3*nn)).tocsr()


def marginals(name: str,wave: np.ndarray,physical: dict[str,Any],y: np.ndarray,Hphi: np.ndarray) -> dict[str,Any]:
    w=wave.reshape(-1,3,len(physical['physical']));norm=float(np.vdot(w,w).real)
    prob=np.sum(abs(w)**2,axis=0)/norm;row=prob.sum(axis=0);nodes=prob.sum(axis=1)
    occupations=list(itertools.product(physical['auxiliary'],repeat=2));lookup={a:i for i,a in enumerate(occupations)};loads=np.zeros(36)
    for weight,(state,_,_) in zip(row,physical['physical']):loads[lookup[physical['states'][state][1:]]]+=weight
    return dict(state=name,norm_squared=norm,Phi_node_probabilities=nodes.tolist(),Phi_mean=float(nodes@y),
        Phi_variance=float(nodes@(y*y)-(nodes@y)**2),Phi_heat_energy=float(np.einsum('cnr,nm,cmr->',w.conj(),Hphi,w).real/norm),
        all36_joint_auxiliary_occupation_probabilities=loads.tolist(),all36_joint_auxiliary_occupations=[[list(a),list(b)] for a,b in occupations],
        copy1_occupation_loads=loads.reshape(6,6).sum(axis=1).tolist(),copy2_occupation_loads=loads.reshape(6,6).sum(axis=0).tolist(),
        integer_electric_squared_sum_mean=float(row@np.array([sum(e*e for e in electric) for _,_,electric in physical['physical']])))


def run(grid: int) -> dict[str,Any]:
    gamma4,metric=phi_frame(Parameters(kappa_power=400));values,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,values>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    y=np.array([-.09,0.,.09]);sigma=.03;radius=.12;dy=.09
    logk=-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y));k=np.exp(logk);k/=np.linalg.norm(k);D=np.zeros((2,3))
    for j in range(2):D[j,j]=math.sqrt(k[j+1]/k[j])/dy;D[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    Hphi=2*sigma*sigma*D.T@D;ev,evec=eigh(Hphi);heat=(evec*np.exp(-ev/2))@evec.T
    neutral=space(0);charged=space(1)
    print(json.dumps({'stage':'constructing_full36_columns','grid':grid,'neutral_rows':len(neutral['physical']),'neutral_middle':len(neutral['middle']),'charged_rows':len(charged['physical']),'charged_middle':len(charged['middle'])}),flush=True)
    N=columns(gamma,y,neutral,grid);C=columns(gamma,y,charged,grid)
    print(json.dumps({'stage':'factoring_actual_preconditioned_Grams'}),flush=True)
    positive_range(N,heat);positive_range(C,heat)
    lambdaN=float(N['q'][0]**2);qn=N['q']/math.sqrt(lambdaN);qc=C['q']/math.sqrt(lambdaN);En=-2*np.log(qn);Ec=-2*np.log(qc)
    vacuum_coeff=np.zeros(len(qn));vacuum_coeff[0]=1.;Omega=physical_wave(N,heat,vacuum_coeff)
    Bn=native.curl(neutral,3);Bc=native.curl(charged,3)
    raw_ph=Bn@Omega;incoming_coeff=qn*project(N,heat,raw_ph);Zph=float(np.vdot(incoming_coeff,incoming_coeff).real)
    if Zph<=0:raise ArithmeticError('Physical independent sin(theta) has no positive photon')
    incoming_coeff/=math.sqrt(Zph);incoming=physical_wave(N,heat,incoming_coeff)
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet);outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    S=source(neutral,charged,native.source_coefficients(gamma,packet),embedding);Sout=source(neutral,charged,native.source_coefficients(gamma,outpacket),embedding)
    raw=(S@incoming).reshape(4,-1);projected=np.array([project(C,heat,row) for row in raw]);prepared=projected*qc[None,:]
    M0=float(np.vdot(prepared,prepared).real);E1=float(np.sum(abs(prepared)**2*Ec[None,:]));R1=float(np.sum(abs(prepared)**2*qc[None,:]**2))
    raw_null=raw-np.array([physical_wave(C,heat,row) for row in projected]);phnull=raw_ph-physical_wave(N,heat,project(N,heat,raw_ph))
    target=(Sout@Omega).reshape(4,-1);receiver=np.array([project(C,heat,Bc@row) for row in target]);consumers=[]
    for t in (0.,.5,1.,2.):
        evolved=prepared*np.exp(-1j*t*Ec)[None,:]/math.sqrt(M0);amplitude=np.vdot(receiver,evolved)
        consumers.append(dict(time=t,coherent_Ref4_receiver_real=float(amplitude.real),coherent_Ref4_receiver_imag=float(amplitude.imag),coherent_Ref4_receiver_probability=float(abs(amplitude)**2),evolved_source_norm_squared=float(np.vdot(evolved,evolved).real)))
    print(json.dumps({'stage':'computing_independent_physical_operator_support'}),flush=True)
    B0=native.curl(neutral,1);curl_pull=pullback(N,N,heat,B0);curl_range=N['T'].conj().T@curl_pull@N['T']
    Eavg=np.array([sum(e)/4 for _,_,e in neutral['physical']]);static=np.array([np.array(density(neutral['states'][i][0],neutral['states'][i][1],neutral['states'][i][2],0))@np.array([0.,-.25,-.5,.25]) for i,_,_ in neutral['physical']])
    from scipy.sparse import diags
    electric_pull=pullback(N,N,heat,diags(Eavg-static));electric_range=N['T'].conj().T@electric_pull@N['T']
    quantum=[marginals('actual_neutral_top_eigenstate',Omega,neutral,y,Hphi),marginals('prepared_incoming_photon',incoming,neutral,y,Hphi),marginals('whole_halfstep_charged_source',np.array([physical_wave(C,heat,row) for row in prepared]),charged,y,Hphi)]
    bands=[dict(absolute_energy_threshold=t,rank=int(np.sum(Ec<=t)),source_fraction=float(np.sum(abs(prepared[:,Ec<=t])**2)/M0)) for t in (.25,2.,4.)]
    descriptor=dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',plan_sha256_at_IMPORT=PLAN_SHA256_AT_IMPORT,pinned_general='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        primitive='NEW two equal integer paths3->0 on same plaquette, two independent charged inducing species',N_total_charged_species=3,active_species=1,auxiliary_species=2,auxiliary_kappa=.04,active_kappa=.09,
        auxiliary_canonical_number_per_copy=2,auxiliary_orbitals_per_copy=4,all_auxiliary_tensor_occupations=36,active_canonical_numbers=[4,5],original_Ref=4,prepared_Ref=3,
        static_Ref_role='Original four static fermion modes, filled background minus4; all four coherent Refhole channels; no Ref hopping or heat supplied',
        source_packet=packet.tolist(),receiver_packet=outpacket.tolist(),source_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],integer_source_charge=1,integer_Refhole_charge=-1,
        bare_Haar_middle='All and only canonical states of TOTAL active+both auxiliary+staticRef density zero; constant integer-electric wave',
        normal_nodes=y.tolist(),normal_sigma=sigma,normal_radius=radius,Haar_grid=grid,electric_Galerkin_cutoff=12,light_homotopy=0.,spatial_spacing=1.,transfer_duration=1.,physical_photon_operator='sin(theta), independent of matter homotopy',thresholds=[.25,2.,4.],
        factorization='Actual mean-column SVD right preconditioning; full Fourier/Gauss columns; positive preconditioned Gram Cholesky; UNSQUARED square factor SVD after undoing preconditioner. No eigenvalue clipping or occupations truncated.',
        operator_contract='KN=Fn Fn†; KC=Fc Fc† tensorIRef4; lambdaN=norm(Fn)^2; Omega actual top. Xi_ph=sqrt(KN/lambdaN) sin(theta) Omega/sqrtZph; Xi_ch=sqrt(KC/lambdaN) S Xi_ph. M0=norm²Xi_ch; E1=<Xi_ch,-log(KC/lambdaN)Xi_ch>; R1=<Xi_ch,(KC/lambdaN)Xi_ch>.')
    return dict(input_descriptor=descriptor,input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},physical_mean_descriptor=metric,
        actual_neutral_photon_Zph=Zph,actual_common_neutral_lambda_N=lambdaN,actual_source_M0=M0,actual_energy_moment_E1=E1,actual_transfer_moment_R1=R1,actual_energy_mean=E1/M0,actual_transfer_Rayleigh_mean=R1/M0,actual_neutral_photon_energy_mean=float(np.sum(abs(incoming_coeff)**2*En)),
        actual_raw_source_Knull_norm_squared=float(np.vdot(raw_null,raw_null).real),actual_raw_source_norm_squared=float(np.vdot(raw,raw).real),actual_raw_photon_Knull_norm_squared=float(np.vdot(phnull,phnull).real),
        actual_positive_physical_sin_support_norm=float(np.linalg.norm(curl_range,2)),actual_positive_Etheta_minus_total_static_density_support_norm=float(np.linalg.norm(electric_range,2)),
        actual_charged_absolute_bands=bands,actual_coherent_consumer_times=consumers,actual_Phi_and_all36_occupation_loads=quantum,
        allcanonical_positive_ranks={'neutral':len(qn),'charged_per_Ref':len(qc)},integer_Gauss_basis_rows={'neutral':len(neutral['physical']),'charged_per_Ref':len(charged['physical'])},bare_Haar_middle_counts={'neutral':len(neutral['middle']),'charged':len(charged['middle'])},factor_diagnostics={'neutral':N['diagnostics'],'charged':C['diagnostics']},mean_preconditioned_Fourier_tail_above12={'neutral':N['mean_preconditioned_Fourier_tail_above12_norm'],'charged':C['mean_preconditioned_Fourier_tail_above12_norm']},
        massless_commoncone_macro_admitted=False,scope='Finite native integer-electric cutoff12 and grid quadrature, all36/Phi3/Ref4. Total static-density-subtracted Etheta and physical sin have independent loop support. No original4D/fullCartesianPhi/SU4, massless/commoncone or macro source claim; Fourier tails are observations, not infinite-lattice certificates.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--haar-grid',type=int,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    start=time.perf_counter();result=run(args.haar_grid);result['runtime_seconds']=time.perf_counter()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('input_descriptor_sha256','actual_neutral_photon_Zph','actual_common_neutral_lambda_N','actual_source_M0','actual_energy_moment_E1','actual_transfer_moment_R1','actual_energy_mean','actual_transfer_Rayleigh_mean','actual_neutral_photon_energy_mean','actual_raw_source_Knull_norm_squared','actual_positive_physical_sin_support_norm','actual_positive_Etheta_minus_total_static_density_support_norm','actual_charged_absolute_bands','actual_coherent_consumer_times','runtime_seconds')}),flush=True)

if __name__=='__main__':main()
