#!/usr/bin/env python3
"""Same-fixed-Gauss coherent sea/sea+four-positive reference-bank preparation.

The finite CAR control checks the actual state and unchanged four-column
bilinear SOURCE, not a second charge law or a second reference bank.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os,sys,time
from pathlib import Path
from typing import Any, cast
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import qr
from prepared_neutral_full_field import BETA
from prepared_neutral_canonical_source import LABELS,literal_coordinate_mark
from prepared_neutral_spatial_charge_transfer import GAMMA5
from sea_selection import SELECTED
from improved_phi_band import NSTAR
from prepared_neutral_full_onebody import full_light_field_factorization,full_coordinate_light_operators
from original_charged_principal import portable

IDENTITY='PNFF-RAW64-coherent-Ref4-sea-Ref0-seaPlus4-ONE-globalfilter-v2'


def coherent_bank_descriptor() -> dict:
    k=NSTAR//4
    return dict(identity=IDENTITY,reference_modes=4,fixed_reference_charge='Nref-4',
        raw_bank_amplitudes=np.ones(2,dtype=complex)/math.sqrt(2),
        banks=[dict(reference_occupation=4,light_occupation='2V',positive_spectators=[]),
               dict(reference_occupation=0,light_occupation='2V+4',positive_spectators=[dict(momentum=[k,0,0],positive_band_spins=[0,1]),dict(momentum=[-k,0,0],positive_band_spins=[0,1])])],
        momentum_convention='wrapped integer Bloch momenta; both complete positive spin pairs; distinct +/-N/4 outside narrow SOFT support',
        whole_field_transport='same actual full negative/positive cluster Kato transport; no isolated positive-mode gap assumed',
        raw_colour_amplitude='1/8 for each of64 m; normalized actual MinSet tie block; bank coefficient multiplies it by1/sqrt2',
        total_number='2V+4 in BOTH neutral banks AND both SOURCE outputs',
        charged_plus=dict(light_occupation='2V+1',reference_occupation=3),
        charged_minus=dict(light_occupation='2V+3',reference_occupation=1),
        absolute_Gauss='Nlight-2V+Nref-4=0; fixed operator, not only SOURCE commutator',
        preparation='exp(pi/4 (T4-T4dagger)) with existing even quartet transfer T4 between full reference wedge and four orthogonal positive Bloch orbitals; Gauss reconstruction uses existing dressed modes',
        source='UNCHANGED all4 raw columns; Bplus annihilates Ref0 bank and Bminus is Pauli blocked on Ref4 bank',
        source_Gram='anti occupied projector Pnegative+Pextra gives base Gram plus positive-semidefinite finite-rank term; plus unchanged; explicit1/sqrt2 raw bank factor',
        ordinary_bank_energy='D_ms,b=Lambda v_ms+Evac_ms+Enegative_ms+delta_b Eextra4_ms; actual O1 spectator energies/phases retained',
        photon_shadow_energy='same actual D_ms,b+omega_g before ONE normalization',
        reference_number_invariance='after quartet preparation is turned OFF, [Nref,H_D]=[Nref,Hactual]=[Nref,J]=[Nref,C]=[Nref,Gspatial]=0; no SOURCE/background change',
        receiver_templates='Y_j=a_gj_dagger (Bplus_j+Bminus_j) Un6 Xi; ONE GLOBAL Gram/rank1 complete binary effect, not separate charge effects',
        paired_SOURCE_norm='on neutral Ref0/4 direct sum, output Ref1/3 are orthogonal, so restricted norm max(CBplus,CBminus); unrestricted bound CBplus+CBminus',
        Coulomb_count_bound='4V+32',
        not_ground='four finite-energy positive spectators are genuine excitations, not a claimed light/heavy full ground',
        unpaid=['full quantum field/globalfilter normalization and SOURCE transfer','same-law static complete receiver/current structured finite-rank correction','neutral intrinsic C intertwining for successor, if required, needs independent proof'])


def canonical_bank_coordinate_marks(law,fields,links,theta,reference_site,
                                    profile,quartet,a):
    """Both original coefficient maps on ONE law and ONE fixed reference bank."""
    return {charge:literal_coordinate_mark(law,charge,fields,links,theta,
                reference_site,profile,quartet,a) for charge in (1,-1)}


def fourier_window_majorant(max_order:int=8) -> dict:
    """Absolute Fourier moment bound for the actual compact smooth taper."""
    bounds={};weighted={}
    for j in range(max_order+1):
        n=j+2
        An=2**(n+1)*(2*n)**(2*n)
        Dn=math.factorial(n)*(n+1)*An*(1+8*An)**(n+1)
        bounds[j]=3+6*Dn
        previous=2**n*(2*(n-1))**(2*(n-1))
        Dprevious=math.factorial(n-1)*n*previous*(1+8*previous)**n
        weighted[j]=6+6*(2*Dn+n*Dprevious) # actual (u F)^(n)=uF^(n)+nF^(n-1).
    if max((*bounds.values(),*weighted.values())).bit_length()>2048:
        raise ValueError('Requested Fourier order exceeds the proved2048-bit cap')
    return dict(orders=max_order,Cf_upper=2**2048,Cf_uF_upper=2**2048,
        individual_log2_ceil={j:(value-1).bit_length() for j,value in bounds.items()},
        weighted_uF_log2_ceil={j:(value-1).bit_length() for j,value in weighted.items()},
        proof='psi(s)=exp(-1/s): |psi^(n)|/n!<=2^n(2n)^(2n); denominator a+b>=exp(-2)>1/8; inverse Taylor recurrence bounded by(1+8An)^(n+1). Taper |F^(n)|<=Dn; support length3; split Fourier integral at|t|=1 and integrate by parts n=j+2 times =>Cf_j<=3+6Dn')


def positive_spectator_modes(onebody:np.ndarray,sites:np.ndarray,size:int,
                             momenta:tuple) -> dict:
    """Actual two complete positive Bloch pairs of a uniform centre operator."""
    blocks=[];energies=[];residuals=[]
    for k in momenta:
        plane=np.exp(2j*np.pi*np.asarray(sites)@np.asarray(k)/size)/math.sqrt(len(sites))
        block=np.kron(plane[:,None],np.eye(4))
        hk=block.conj().T@onebody@block
        values,vectors=np.linalg.eigh(hk);selected=np.flatnonzero(values>0)
        if len(selected)!=2:raise ValueError('Centre positive Bloch cluster must have rank2')
        modes=block@vectors[:,selected]
        blocks.append(modes);energies.extend(values[selected])
        residuals.append(float(np.linalg.norm(onebody@modes-modes*values[selected])))
    modes=np.column_stack(blocks)
    gram_error=float(np.linalg.norm(modes.conj().T@modes-np.eye(4)))
    if gram_error>1e-10 or max(residuals)>1e-9*max(1.,float(np.linalg.norm(onebody))):
        raise ValueError('Input is not the actual uniform Bloch centre at these momenta')
    return dict(modes=modes,energies=np.asarray(energies),extra_energy=float(sum(energies)),
        momentum_blocks=tuple(momenta),gram_error=gram_error,eigen_residuals=residuals)


def compressed_bank_basis(bulk_modes:int=12,base_particles:int=4) -> dict:
    """Exact conserved-number sectors, NOT deletion from physical Hilbert space."""
    masks={p:tuple(sum(1<<j for j in occupied) for occupied in itertools.combinations(range(bulk_modes),p))
           for p in (base_particles,base_particles+1,base_particles+3,base_particles+4)}
    ref4=15<<bulk_modes
    neutral=tuple(m|ref4 for m in masks[base_particles])+masks[base_particles+4]
    charged=(tuple(m|(ref4^(1<<(bulk_modes+r))) for r in range(4) for m in masks[base_particles+1])
             +tuple(m|(1<<(bulk_modes+r)) for r in range(4) for m in masks[base_particles+3]))
    index={mask:j for j,mask in enumerate(charged)};transfers={}
    for charge in (1,-1):
        for i,r in itertools.product(range(bulk_modes),range(4)):
            first,second=(bulk_modes+r,i) if charge==1 else (i,bulk_modes+r)
            rows=[];columns=[];signs=[]
            for column,mask in enumerate(neutral):
                if not(mask>>first&1):continue
                sign=(-1)**((mask&((1<<first)-1)).bit_count());mask^=1<<first
                if mask>>second&1:continue
                sign*=(-1)**((mask&((1<<second)-1)).bit_count());mask|=1<<second
                rows.append(index[mask]);columns.append(column);signs.append(sign)
            transfers[charge,i,r]=sparse.csr_matrix((signs,(rows,columns)),shape=(len(charged),len(neutral)))
    return dict(neutral_masks=neutral,charged_masks=charged,bulk_masks=masks,transfers=transfers,
                fixed_reference_charge='Nref-4',bulk_modes=bulk_modes,
                neutral_bank_dimensions=[len(masks[base_particles]),len(masks[base_particles+4])],
                charged_bank_dimensions=[4*len(masks[base_particles+1]),4*len(masks[base_particles+3])])


def annihilators(modes:int) -> list:
    dimension=1<<modes;out=[]
    for j in range(modes):
        columns=np.array([n for n in range(dimension) if n&(1<<j)],dtype=np.int64)
        rows=columns^(1<<j)
        signs=np.array([(-1)**((int(n)&((1<<j)-1)).bit_count()) for n in columns],float)
        out.append(sparse.csr_matrix((signs,(rows,columns)),shape=(dimension,dimension)))
    return out


def native_bank_light_view(model,profile=None,theta=None,a=None) -> dict:
    """Full108/actual coherent bank; explicit SOURCE profile and physical EM point.

    Sea/R4 are prepared from h(A=0). Coordinate Hamiltonian uses the supplied
    actual A, whose harmonic mean equals theta/N. No source-input whitening.
    """
    law=model.law;V=law.volume
    profile=np.asarray(model.orbitals[:,0] if profile is None else profile,complex)
    theta=np.asarray(np.zeros(3) if theta is None and a is None else
        (np.mean(cast(np.ndarray,a),axis=0)*law.size if theta is None else theta),float)
    a=np.tile(theta/law.size,(V,1)) if a is None else np.asarray(a,float)
    if profile.shape!=(V,) or theta.shape!=(3,) or a.shape!=(V,3):
        raise ValueError('Physical SOURCE profile/theta/A shapes mismatch')
    if np.linalg.norm(np.mean(a,axis=0)*law.size-theta)>1e-12:
        raise ValueError('Physical coordinate A harmonic mean must equal theta/N')
    fields=np.tile(model.frame,(V,1,1,1));links=np.tile(np.eye(4,dtype=complex),(V,3,1,1))
    h=law.onebody(fields,links,np.zeros((V,3)))
    values,vectors=np.linalg.eigh(h);negative=vectors[:,values<0]
    if negative.shape[1]!=2*V:raise ValueError('Entire actual negative cluster must have rank2V')
    P=negative@negative.conj().T
    ev,colour=np.linalg.eigh(SELECTED[0]);P0=np.kron(np.eye(V),colour[:,ev<0]@colour[:,ev<0].conj().T)
    I=np.eye(4*V);X=P@P0+(I-P)@(I-P0)
    d,U=np.linalg.eigh(X.conj().T@X)
    if d.min()<1e-12:raise ValueError('Full actual centre valence Kato chart is singular')
    K=X@((U/np.sqrt(d))@U.conj().T)
    old=np.kron(model.orbitals,np.eye(4))
    active_negative=K@old@np.kron(np.eye(2),colour[:,ev<0])
    k=max(1,law.size//4)
    extra=positive_spectator_modes(h,law.sites,law.size,((k,0,0),(-k,0,0)))
    positive_complement=I-P-extra['modes']@extra['modes'].conj().T
    Q,R,_=cast(Any,qr(positive_complement,pivoting=True,mode='economic'))
    if min(abs(np.diag(R)[:4]))<1e-10:
        raise ValueError('Actual positive complement must contain four empty retained orbitals')
    active_positive=Q[:,:4]
    basis=np.column_stack((active_negative,active_positive,extra['modes']))
    gram=float(np.linalg.norm(basis.conj().T@basis-np.eye(12)))
    if gram>1e-10:raise ValueError('Actual valence/positive complement/spectator basis is not orthonormal')
    outside=P-active_negative@active_negative.conj().T
    layout=compressed_bank_basis()
    seed=np.zeros(len(layout['neutral_masks']),complex)
    seed[layout['neutral_masks'].index(15|(15<<12))]=1/math.sqrt(2)
    seed[layout['neutral_masks'].index(15|(15<<8))]=1/math.sqrt(2)
    marks=canonical_bank_coordinate_marks(law,fields,links,theta,
        model.reference_site,profile,BETA,a)
    plus=marks[1].reshape(4*V,4)
    minus=marks[-1].transpose(1,0,2).reshape(4,4*V)
    source={}
    for charge,coefficients in ((1,basis.conj().T@plus),(-1,(minus@basis).T)):
        source[charge]=sum((coefficients[i,r]*layout['transfers'][charge,i,r]
            for i,r in itertools.product(range(12),range(4))),
            sparse.csr_matrix((len(layout['charged_masks']),len(layout['neutral_masks'])),dtype=complex))
    active_projector=basis@basis.conj().T
    outside_empty=I-active_projector-outside
    outside_source={1:outside_empty@plus,-1:minus@outside}
    occupied_errors={}
    for mask,amplitude in zip(layout['neutral_masks'],seed):
        if abs(amplitude)==0:continue
        occupied=basis[:,[j for j in range(12) if mask&(1<<j)]]
        physical=P if (mask>>12).bit_count()==4 else P+extra['modes']@extra['modes'].conj().T
        occupied_errors[(mask>>12).bit_count()]=float(np.linalg.norm(
            occupied@occupied.conj().T+outside-physical))
    if max(occupied_errors.values())>1e-10:
        raise AssertionError('Compressed actual seed does not reconstruct physical occupied projector')
    return dict(identity=IDENTITY,fields=fields,links=links,onebody=h,negative_projector=P,
        SOURCE_profile=profile,SOURCE_theta=theta,coordinate_EM=a,
        actual_coordinate_onebody=law.onebody(fields,links,a),
        SOURCE_input_binding='full raw beta/global2; explicit complex profile and theta=N meanA; h0 prepares sea/R4 while coordinateH uses actualA; no column normalization',
        positive_spectators=extra,active_basis=basis,active_basis_Gram_error=gram,
        frozen_outside_occupied_projector=outside,compressed_basis=layout,
        raw_neutral_light_seed=seed,projected_SOURCE=source,full_SOURCE_marks=marks,
        physical_occupied_projector_errors=occupied_errors,
        full_SOURCE_outside_active_coefficients=outside_source,
        full_SOURCE_current=full_coordinate_light_operators,
        full_coordinate_light_H=full_light_field_factorization(model),
        scalar_reference_charge='Nref-4 BOTH channels',native_light_counts=[2*V,2*V+4],
        charged_light_counts=[2*V+1,2*V+3],native_reference_counts=[4,0],
        scope='actual full108/shared9+ALL55EM coordinate operator and coherent active12/bank seed with TRUE valence50 complement; native finite embedding, not legal512 fullfieldfn or whole-Fock/nativephysicalNoe evolution')


def source_complete_fixed_bank_view(view:dict) -> dict:
    """SOURCE-adapted FIXED CAR12 representation of the SAME physical raw state.

    It captures BOTH literal maps at the specified anchor point, not h0/free
    dynamics, field-dependent SOURCE functions or a time-dependent recapture.
    """
    P=view['negative_projector'];R=view['positive_spectators']['modes']
    n=P.shape[0];extra=R@R.conj().T;empty=np.eye(n)-P-extra
    marks=view['full_SOURCE_marks'];plus=marks[1].reshape(n,4)
    minus=marks[-1].transpose(1,0,2).reshape(4,n)
    def complete(projector,columns):
        U,s,_=np.linalg.svd(projector@columns,full_matrices=False)
        rank=int(np.sum(s>1e-12*max(1.,float(s[0]))))
        occupied=U[:,:rank]
        if rank<4:
            Q,_,_=cast(Any,qr(projector-occupied@occupied.conj().T,pivoting=True,mode='economic'))
            occupied=np.column_stack((occupied,Q[:,:4-rank]))
        return occupied,rank
    negative,negative_rank=complete(P,minus.conj().T)
    positive,positive_rank=complete(empty,plus)
    basis=np.column_stack((negative,positive,R));active=basis@basis.conj().T
    outside=P-negative@negative.conj().T
    layout=view['compressed_basis'];source={}
    for charge,coefficients in ((1,basis.conj().T@plus),(-1,(minus@basis).T)):
        source[charge]=sum((coefficients[i,r]*layout['transfers'][charge,i,r]
            for i,r in itertools.product(range(12),range(4))),
            sparse.csr_matrix((len(layout['charged_masks']),len(layout['neutral_masks'])),dtype=complex))
    outside_coefficients={1:(np.eye(n)-active-outside)@plus,-1:minus@outside}
    errors={q:float(np.linalg.norm(c)) for q,c in outside_coefficients.items()}
    gram=float(np.linalg.norm(basis.conj().T@basis-np.eye(12)))
    occupied_errors={}
    for mask,amplitude in zip(layout['neutral_masks'],view['raw_neutral_light_seed']):
        if abs(amplitude)==0:continue
        occupied=basis[:,[j for j in range(12) if mask&(1<<j)]]
        ref=(mask>>12).bit_count();physical=P if ref==4 else P+extra
        occupied_errors[ref]=float(np.linalg.norm(outside+occupied@occupied.conj().T-physical))
    if gram>1e-10 or max(errors.values())>1e-10 or max(occupied_errors.values())>1e-10:
        raise AssertionError('Fixed SOURCE-complete basis changes state or misses literal SOURCE')
    result=dict(view)
    result.update(active_basis=basis,active_basis_Gram_error=gram,
        frozen_outside_occupied_projector=outside,projected_SOURCE=source,
        full_SOURCE_outside_active_coefficients=outside_coefficients,
        physical_occupied_projector_errors=occupied_errors,
        fixed_SOURCE_capture_errors=errors,source_adapted_ranks=[negative_rank,positive_rank],
        free_h0_outside_active_norm=float(np.linalg.norm((np.eye(n)-active)@view['onebody']@basis)),
        actual_coordinate_H_outside_active_norm=float(np.linalg.norm((np.eye(n)-active)@view['actual_coordinate_onebody']@basis)),
        representation='fixed SOURCE-adapted negative4/empty4/R4; same fullPnegative andphysicalRAW54/58 state; NO inputwhitening/time recapture',
        scope='both full original SOURCE maps EXACT on RAWneutral at FIXED profile/theta/field anchor; h0/current/physicalNoe dynamics remain Galerkin with unrepresented modes, not sourceFUNCTION(q) closure')
    return result


def native_bank_light_control(model) -> dict:
    view=native_bank_light_view(model);law=model.law;V=law.volume
    P=view['negative_projector'];R=view['positive_spectators']['modes']
    marks=view['full_SOURCE_marks'];plus=marks[1].reshape(4*V,4)
    minus=marks[-1].transpose(1,0,2).reshape(4,4*V)
    base=float(np.trace(minus@P@minus.conj().T).real)
    added=float(np.linalg.norm(minus@R)**2)
    plus_Z2=float(np.trace(plus.conj().T@(np.eye(4*V)-P)@plus).real/2)
    layout=view['compressed_basis'];seed=view['raw_neutral_light_seed']
    counts=np.array([mask.bit_count() for mask in layout['neutral_masks']])
    outside_Z2={q:float(np.linalg.norm(view['full_SOURCE_outside_active_coefficients'][q])**2/2)
        for q in (1,-1)}
    projected_Z2={q:float(np.linalg.norm(view['projected_SOURCE'][q]@seed)**2) for q in (1,-1)}
    split_errors={1:abs(plus_Z2-projected_Z2[1]-outside_Z2[1]),
        -1:abs((base+added)/2-projected_Z2[-1]-outside_Z2[-1])}
    if max(split_errors.values())>1e-10:
        raise AssertionError('Physical full SOURCE norm does not equal retained plus outside remainder')
    outcounts=np.array([mask.bit_count() for mask in layout['charged_masks']])
    gauss=float(np.linalg.norm((counts-8)*seed));source_G={}
    for charge in (1,-1):
        v=view['projected_SOURCE'][charge]@seed
        source_G[charge]=float(np.linalg.norm((outcounts-8)*v))
    if gauss>1e-12 or max(source_G.values())>1e-12 or plus_Z2<=0 or base<=0 or added<-1e-12:
        raise AssertionError('Actual coherent full108 state/SOURCE Gauss/norm failed')
    return dict(identity=IDENTITY,oneparticle_dimension=4*V,whole_negative_occupied=2*V,
        native_coherent_neutral_counts=[2*V,2*V+4],references=[4,0],charged_counts=[2*V+1,2*V+3],
        fixed_total_number=2*V+4,fixed_reference_charge='Nref-4',
        exact_compressed_global_Gauss_error=gauss,exact_compressed_SOURCE_Gauss_errors=source_G,
        active_basis_Gram_error=view['active_basis_Gram_error'],
        physical_occupied_projector_errors=view['physical_occupied_projector_errors'],
        projected_SOURCE_norm_squared=projected_Z2,outside_SOURCE_norm_squared=outside_Z2,
        SOURCE_retained_outside_norm_split_errors=split_errors,
        actual_positive_spectator_eigen_residuals=view['positive_spectators']['eigen_residuals'],
        actual_extra_four_energy=view['positive_spectators']['extra_energy'],
        actual_negative_cluster_energy=float(np.trace(P@view['onebody']).real),
        actual_neutral_field_basis_dimension=view['full_coordinate_light_H']['field_dimension'],
        native_coexact_modes=52,harmonic_modes=3,neutral_bank_dimensions=layout['neutral_bank_dimensions'],
        charged_bank_dimensions=layout['charged_bank_dimensions'],
        frozen_complement_rank=float(np.trace(view['frozen_outside_occupied_projector']).real),
        raw_full108_plus_SOURCE_norm_squared=plus_Z2,raw_full108_minus_SOURCE_norm_squared=(base+added)/2,
        minus_base_before_bank_norm_squared=base,positive_rank4_added_norm_squared=added,
        source_coefficients='actual original pure literal_coordinate_mark on SAME law for BOTH maps; not automatically adjoints or charge=-1H',
        cached_supplier_state_reprepared=False,new_coherent_light_state_constructed=True,scope=view['scope'])


def native_same_bank_control() -> dict:
    started=time.perf_counter();a=annihilators(12);dimension=4096
    energy,basis=np.linalg.eigh(SELECTED[0])
    negative=tuple(k*4+j for k in range(2) for j in np.flatnonzero(energy<0))
    positive=tuple(k*4+j for k in range(2) for j in np.flatnonzero(energy>0))
    reference=tuple(range(8,12));n0=sum(1<<j for j in (*negative,*reference))
    sea_ref4=np.zeros(dimension,complex);sea_ref4[n0]=1
    transfer=cast(Any,sparse.eye)(dimension,format='csr',dtype=complex)
    for j in positive:transfer=a[j].conj().T@transfer
    for j in reference:transfer=a[j]@transfer
    sea_plus4_ref0=transfer@sea_ref4
    target=(sea_ref4+sea_plus4_ref0)/math.sqrt(2)
    prepared=expm_multiply(math.pi/4*(transfer-transfer.conj().T),sea_ref4)
    psi=[sum((basis[i,j]*(a[j]+a[4+j])/math.sqrt(2) for j in range(4)),sparse.csr_matrix((dimension,dimension),dtype=complex)) for i in range(4)]
    mark=np.einsum('a,aij->ij',BETA,LABELS)
    anti=-GAMMA5@mark.conj().T@GAMMA5
    plus=sum((mark[i,r]*psi[i].conj().T@a[8+r] for i in range(4) for r in range(4)),sparse.csr_matrix((dimension,dimension),dtype=complex))
    minus=sum((anti[r,i]*a[8+r].conj().T@psi[i] for i in range(4) for r in range(4)),sparse.csr_matrix((dimension,dimension),dtype=complex))
    counts=np.array([n.bit_count() for n in range(dimension)])
    bulk=np.array([(n&255).bit_count() for n in range(dimension)])
    ref=counts-bulk;G=counts-8
    pp=plus@prepared;pm=minus@prepared
    old=np.zeros(dimension,complex);old[sum(1<<j for j in negative)]=1
    errors=dict(preparation=float(np.linalg.norm(prepared-target)),neutral_G=float(np.linalg.norm(G*prepared)),plus_G=float(np.linalg.norm(G*pp)),minus_G=float(np.linalg.norm(G*pm)),
        plus_wrong_bank=float(np.linalg.norm(plus@sea_plus4_ref0)),minus_wrong_bank=float(np.linalg.norm(minus@sea_ref4)),
        plus_charged_count=float(np.linalg.norm((bulk-5)*pp)+np.linalg.norm((ref-3)*pp)),minus_charged_count=float(np.linalg.norm((bulk-7)*pm)+np.linalg.norm((ref-1)*pm)))
    old_minus=minus@old;new_minus=minus@sea_plus4_ref0
    result:dict[str,Any]=dict(identity=IDENTITY,dimension=dimension,bulk_modes=8,reference_modes=4,total_neutral_number=8,
        errors=errors,plus_source_norm=float(np.linalg.norm(pp)),minus_source_norm=float(np.linalg.norm(pm)),
        old_bare_minus_absolute_Gauss_defect=float(np.linalg.norm(G*old)),
        old_bare_minus_SOURCE_absolute_Gauss_defect=float(np.linalg.norm(G*old_minus)/np.linalg.norm(old_minus)),
        base_anti_norm_squared=float(np.vdot(old_minus,old_minus).real),new_anti_norm_squared_before_bank_factor=float(np.vdot(new_minus,new_minus).real),
        added_positive_Gram_trace=float(cast(Any,np.vdot(new_minus,new_minus)).real-cast(Any,np.vdot(old_minus,old_minus)).real),
        extra_positive_count=len(positive),raw_quartet=BETA,bank_factor=1/math.sqrt(2),
        old_state_adverse='Ref0 plus ONLY sea violates SAME fixed Gauss by -4 despite number-conserving SOURCE',
        scope='actual CAR12 finite state/quartet unitary/unchanged full4-column raw Clifford SOURCE and absolute global Gauss. Not full local Gauss, legal512 field/filter, SOFT norm, or spatial closed-loop proof',wall_seconds=time.perf_counter()-started)
    if max(errors.values())>2e-12 or min(result['plus_source_norm'],result['minus_source_norm'])<=0 or result['added_positive_Gram_trace']<-2e-12:raise AssertionError(result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--state-capsule',type=Path,required=True);args=parser.parse_args()
    from prepared_neutral_spatial_gauss_well import PreparedSpatialGaussQuantumSlice
    root=Path(__file__).resolve().parent
    loaded={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest()
            for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
            and Path(cast(str,m.__file__)).parent==root and Path(cast(str,m.__file__)).suffix=='.py'}
    model=PreparedSpatialGaussQuantumSlice(state_capsule=args.state_capsule)
    result=dict(descriptor=coherent_bank_descriptor(),native=native_same_bank_control(),
        actual_full108=native_bank_light_control(model),Fourier_window=fourier_window_majorant(),
        input_state_sha256=hashlib.sha256(args.state_capsule.read_bytes()).hexdigest(),
        input_operator_signature=model.operator_signature(),loaded_source_sha256=loaded,
        loaded_sources_unchanged=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in loaded.items()))
    print(json.dumps(portable(result),indent=2,sort_keys=True))
