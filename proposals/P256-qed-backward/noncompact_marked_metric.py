#!/usr/bin/env python3
"""Complete-reference coherent source metric from microscopic CT, not P2.

Actual whole CAR, noncentral Wilson/normal fields and coupled photons.
This does not identify total charged spin or an asymptotic photon.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os
from pathlib import Path
from typing import Any, Callable, cast
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from bulk_quantum_selection import BulkQuantumLaw
from noncompact_native_current import accept, adverse
from noncompact_neutral_supplier import particle_hole
from noncompact_finite_packet_spin import GraphQuantum, creator_maps, prepare, lift, D, REFERENCE
from original_charged_principal import original_frame
from original_physical_rotation_lift import recover_gamma
from rotation_band import RotationBand

ROOT = Path(__file__).resolve().parent

def spin_fields(fields):
    grams = np.einsum('xmij,xnji->xmn', fields, fields).real/4
    gammas=[]
    for gram,phi in zip(grams,fields):
        values,vectors=np.linalg.eigh(gram)
        if values[0]<=0: raise ValueError('Marked bivector source requires full-rank scalar Gram')
        gammas.append(np.einsum('mn,nij->mij',(vectors/np.sqrt(values))@vectors.T,phi))
    gammas=np.asarray(gammas)
    spins = np.asarray([[-1j*(g[j]@g[k]-g[k]@g[j])/4 for j,k in ((2,3),(3,1),(1,2))] for g in gammas])
    labels = np.concatenate((np.tile(np.eye(4,dtype=complex),(len(fields),1,1,1)),2j*spins),axis=1)
    return grams,gammas,spins,labels

def marks(fields, links, A, eta):
    _,_,spins,labels = spin_fields(fields)
    # Genuine colour Wilson transport from the same outside R2 to sites0/1.
    wilson = np.asarray([links[0]@links[2],links[2],np.eye(4)])
    green = np.linalg.pinv(D@D.T,hermitian=True)
    phase = np.asarray([np.exp(1j*(D.T@green@(np.eye(3)[x]-np.eye(3)[REFERENCE]))@A) for x in range(3)])
    M = np.asarray([np.concatenate([eta[x]*phase[x]*(labels[x,a]@wilson[x]) for x in range(3)]) for a in range(4)])
    return M,spins

def maps_from_marks(M,neutral_rows,charged_rows):
    lookup = {r:i for i,r in enumerate(charged_rows)}
    creators=[]
    for mode in range(12):
        rr=[];cc=[];vv=[]
        for col,row in enumerate(neutral_rows):
            if mode in row: continue
            rr.append(lookup[tuple(sorted((*row,mode)))]);cc.append(col)
            vv.append((-1)**sum(j<mode for j in row))
        creators.append(sparse.coo_matrix((vv,(rr,cc)),shape=(len(charged_rows),len(neutral_rows))).tocsr())
    return [sparse.vstack([sum((M[a,i,b]*creators[i] for i in range(12)),sparse.csr_matrix(creators[0].shape,dtype=complex)) for b in range(4)],format='csr') for a in range(4)]

def source_gram(B,state):
    columns=np.column_stack([t@state for t in B])
    return columns.conj().T@columns

def encoded(a):
    z=np.asarray(a)
    return dict(real=z.real.tolist(),imag=z.imag.tolist())

def fixed_coordinate_case(frame,noise,rng,PH,neutral_rows,charged_rows):
    law=BulkQuantumLaw(3)
    local_gauge=np.asarray([expm(.3j*np.einsum('a,aij->ij',rng.normal(size=15),law.generators)) for _ in range(3)])
    fields=local_gauge[:,None]@np.tile(frame,(3,1,1,1))@local_gauge[:,None].conj().swapaxes(-1,-2)
    fields += noise*law.fields(rng.normal(size=(law.volume,4,15)))[:3]
    links=np.asarray([expm(.2j*np.einsum('a,aij->ij',rng.normal(size=15),law.generators)) for _ in range(4)])
    A=rng.normal(size=4)*3
    eta=np.asarray([math.sqrt(.6),math.sqrt(.4)*np.exp(.7j),0.])
    M,spins=marks(fields,links,A,eta)
    negative,_=marks(-fields,links,A,eta)
    accept(np.linalg.norm(M-negative),'actual entire coherent mark even under microscopic Phi sign CT')
    B=maps_from_marks(M,neutral_rows,charged_rows)
    psi=rng.normal(size=len(neutral_rows))+1j*rng.normal(size=len(neutral_rows));psi/=np.linalg.norm(psi)
    partner=PH@psi.conj()
    G=(source_gram(B,psi)+source_gram(B,partner))/2
    bare=np.einsum('air,bir->ab',M.conj(),M)/2
    metric_error=accept(np.linalg.norm(G-bare),'actual whole-CAR CT metric versus coordinate HS half',2e-8)
    # Whole neutral-sector CAR identity, not only this state's expectation.
    raw=B[0].conj().T@B[0]
    expected=sparse.eye(len(neutral_rows),dtype=complex)*np.vdot(M[0],M[0])-lift(M[0]@M[0].conj().T,neutral_rows)
    car_error=accept(sparse.linalg.norm(raw-expected),'complete neutral CAR operator identity',2e-8)
    gauge=np.asarray([expm(.3j*np.einsum('a,aij->ij',rng.normal(size=15),law.generators)) for _ in range(3)])
    moved_fields=gauge[:,None]@fields@gauge[:,None].conj().swapaxes(-1,-2)
    ends=((0,1),(0,1),(1,2),(2,0))
    moved_links=np.asarray([gauge[x]@u@gauge[y].conj().T for u,(x,y) in zip(links,ends)])
    moved,_=marks(moved_fields,moved_links,A,eta)
    target=np.asarray([np.concatenate([gauge[x]@M[a,4*x:4*x+4]@gauge[REFERENCE].conj().T for x in range(3)]) for a in range(4)])
    covariance=accept(np.linalg.norm(moved-target),'actual full marked colour Wilson/Gamma gauge covariance',2e-8)
    missing=[b[:len(charged_rows)] for b in B]
    missingG=(source_gram(missing,psi)+source_gram(missing,partner))/2
    wrong=adverse(np.linalg.norm(missingG-bare),'actual whole-CAR CT metric versus coordinate HS half',2e-8)
    casimir=max(float(np.linalg.norm(sum(s@s for s in site)-.75*np.eye(4))) for site in spins)
    if noise==0: accept(np.linalg.norm(G-2*np.eye(4)),'actual complete quartet Gram2 without point/P2 assumption',2e-8)
    return dict(normal_noise=noise,marked_metric=encoded(G),metric_eigenvalues=np.linalg.eigvalsh(G).tolist(),whole_CAR_error=car_error,CT_metric_error=metric_error,colour_covariance_error=covariance,actual_bivector_Casimir_defect=casimir,Gram2_not_assumed_for_nonClifford_normal_fields=noise!=0,missing_reference=wrong)

def occupied_adversary(frame,PH,neutral_rows,charged_rows):
    eta=np.asarray([math.sqrt(.6),math.sqrt(.4)*np.exp(.7j),0.])
    M,_=marks(np.tile(frame,(3,1,1,1)),np.tile(np.eye(4),(4,1,1)),np.zeros(4),eta)
    # All four actual coherent creator modes occupied; plus two orthogonal modes.
    U=np.linalg.qr(M[0],mode='complete')[0]
    psi=np.asarray([np.linalg.det(U[np.ix_(row,range(6))]) for row in neutral_rows])
    accept(abs(np.linalg.norm(psi)-1),'actual complete six-particle coherent Slater normalization')
    B=maps_from_marks(M,neutral_rows,charged_rows)
    unaveraged=source_gram(B,psi)
    averaged=(unaveraged+source_gram(B,PH@psi.conj()))/2
    accept(np.linalg.norm(averaged-2*np.eye(4)),'actual complete quartet Gram2 without point/P2 assumption')
    rejection=adverse(np.linalg.norm(unaveraged-2*np.eye(4)),'actual complete quartet Gram2 without point/P2 assumption')
    return dict(unaveraged_unlabelled_norm=float(unaveraged[0,0].real),CT_averaged_metric=encoded(averaged),missing_CT_same_assertion=rejection)

def quantum_case(args,frame,PH):
    neutral=GraphQuantum(args.levels,args.harmonics,0.,6,.37,.9,frame)
    charged=GraphQuantum(args.levels,args.harmonics,0.,7,.37,.9,frame)
    E0,state,cluster=neutral.solve(6)
    gammas=recover_gamma(np.einsum('mij,nji->mn',frame,frame).real/4,frame)
    chirality=gammas[0]@gammas[1]@gammas[2]@gammas[3]
    generator=lift(np.kron(np.eye(3),math.pi/2*(np.eye(4)-chirality)),neutral.rows)
    solver=cast(Callable[...,Any],expm_multiply)
    def CT(value):
        wave=neutral.wave(value).conj().reshape(neutral.n,neutral.l,neutral.d,-1)[:,::-1].reshape(neutral.phdim,neutral.d,-1)
        transformed=(PH@wave.transpose(1,0,2).reshape(neutral.d,-1))
        transformed=solver(-1j*generator,transformed,traceA=-1j*generator.diagonal().sum())
        return transformed.reshape(neutral.d,neutral.phdim,-1).transpose(1,0,2).reshape(np.asarray(value).shape)
    rng=np.random.default_rng(2221012)
    trial=rng.normal(size=neutral.size)+1j*rng.normal(size=neutral.size);trial/=np.linalg.norm(trial)
    symmetry=accept(np.linalg.norm(CT(neutral.action(trial))-neutral.action(CT(trial))),'actual coupled quantum neutral CT antiunitary commutator',2e-8)
    square=accept(np.linalg.norm(CT(CT(trial))-trial),'actual half-filled neutral CT square plus identity',2e-8)
    _,_,spins,labels=spin_fields(np.tile(frame,(3,1,1,1)))
    eta=np.asarray([math.sqrt(.6),math.sqrt(.4)*np.exp(.7j),0.])
    maps=creator_maps(neutral,charged,eta,list(labels[0]))
    def G(v):
        # prepare columns are label then reference then purification.
        columns=np.column_stack([prepare(neutral,charged,v,maps[a]).ravel() for a in range(4)])
        return columns.conj().T@columns
    gram=(G(state)+G(CT(state)))/2
    accept(np.linalg.norm(gram-2*np.eye(4)),'actual quantum photon/full-reference finite-mode quartet Gram2',2e-8)
    return dict(E0=E0,actual_cluster=cluster,whole_neutral_dimension=neutral.size,full_reference_charged_dimension=4*charged.size,CT_commutator=symmetry,CT_square=square,Gram=encoded(gram),full_neutral_comparison_residual=neutral.enlarged_residual(state,E0,frame),ground_only_scope='actual finite Galerkin coupled neutral; no full quantum normal or thermodynamic ground')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels',type=int,default=3);parser.add_argument('--harmonics',type=int,default=2)
    args=parser.parse_args()
    band=RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    selected,frame=original_frame(band,5,np.zeros(3),True)
    rows=tuple(itertools.combinations(range(12),6));charged_rows=tuple(itertools.combinations(range(12),7))
    PH=sparse.csr_matrix(particle_hole(rows,12))
    rng=np.random.default_rng(2221011)
    cases=[fixed_coordinate_case(frame,noise,rng,PH,rows,charged_rows) for noise in (0.,.01,.03)]
    all64=[]
    for theta in itertools.product(range(4),repeat=3):
        native,phi=original_frame(band,5,np.asarray(theta)*math.pi/2,True)
        _,_,spins,labels=spin_fields(np.tile(phi,(3,1,1,1)))
        hs=np.einsum('aij,bij->ab',labels[0].conj(),labels[0])/2
        error=accept(np.linalg.norm(hs-2*np.eye(4)),'actual all64 full native Gram marked HS metric2',2e-8)
        all64.append(dict(centre=list(theta),metric_error=error))
    # Arbitrary full microscopic endpoints, not only selected uniform fields.
    law=BulkQuantumLaw(3);fields=law.fields(rng.normal(size=(law.volume,4,15)))
    links=np.asarray([[expm(.2j*np.einsum('a,aij->ij',rng.normal(size=15),law.generators)) for _ in range(3)] for _ in range(law.volume)])
    A=rng.normal(size=(law.volume,3))
    native_CT=accept(np.linalg.norm(law.onebody(-fields,links,A)+law.onebody(fields,links,A)),'actual arbitrary full native onebody odd under Phi sign CT')
    print(json.dumps(dict(law='different noncompact real-link quadratic curl plus native full fields/CAR',native_CT_endpoint_error=native_CT,native_frame_Gram=selected['G'].tolist(),all64_HS=all64,full_sector_cases=cases,occupied_mode_countercase=occupied_adversary(frame,PH,rows,charged_rows),actual_quantum=quantum_case(args,frame,PH),metric_theorem='CT-invariant neutral and CT-even mark: Gram=half expected coordinate HS metric; Clifford quaternion gives2I',normal_Casimir_or_total_spin_proved=False,phase_or_Compton_proved=False,source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('noncompact_marked_metric.py','noncompact_marked_metric.md','noncompact_finite_packet_spin.py','noncompact_native_current.py')}),indent=2,allow_nan=False))

if __name__=='__main__':main()
