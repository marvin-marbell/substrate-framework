#!/usr/bin/env python3
"""Charged Lehmann response on the rich-photon Hubbard-loop law.

One uniform loop-flux coordinate has primitive electric inertia; its potential
and charged vertices arise from the same exact Peierls Hubbard Hamiltonian in
N=1,2,3 sectors. Units t=hbar=|e|=a=1.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, kron, eye
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "rich_photon_20261014_loop.py"
sys.path.insert(0, str(ROOT))
from rich_photon_20261014_loop import Parameters

LAW = "rich-photon-hubbard-plaquette-links-v1-20261014"
NGRID = 32
NLOW = 20


def bits_basis(n):
    return [sum(1 << i for i in occ) for occ in itertools.combinations(range(8), n)]


def hop(bits, target, source):
    if not bits & (1 << source) or bits & (1 << target):
        return None
    sign = (-1) ** ((bits & ((1 << source)-1)).bit_count())
    rem = bits ^ (1 << source)
    sign *= (-1) ** ((rem & ((1 << target)-1)).bit_count())
    return rem | (1 << target), sign


def matter_matrices(n, p):
    basis = bits_basis(n); index = {b:i for i,b in enumerate(basis)}
    interaction = np.zeros((len(basis),len(basis)), complex)
    transfers = np.zeros((4,len(basis),len(basis)), complex)
    for col,b in enumerate(basis):
        for site in range(4):
            interaction[col,col] += p.repulsion*((b>>(2*site))&1)*((b>>(2*site+1))&1)
        for edge in range(4):
            for spin in range(2):
                h=hop(b,2*((edge+1)%4)+spin,2*edge+spin)
                if h is not None:
                    nb,sgn=h; transfers[edge,index[nb],col] += -p.hopping*sgn
    return basis,index,interaction,transfers


def coupled_matrix(n, p, ng=NGRID):
    basis,index,interaction,T=matter_matrices(n,p); d=len(basis)
    q=2*np.pi*np.arange(ng)/ng; dx=2*np.pi/ng
    H=lil_matrix((ng*d,ng*d), dtype=complex)
    for g,x in enumerate(q):
        h=interaction.copy()
        for e in range(4):
            z=np.exp(1j*p.link_charge*x/4)
            h += z*T[e] + z.conjugate()*T[e].conj().T
        H[g*d:(g+1)*d,g*d:(g+1)*d]=h
    kin=1/(2*p.electric_inertia*dx*dx)
    gauge=np.asarray([np.exp(1j*np.pi/2*sum(site*(((b>>(2*site))&1)+((b>>(2*site+1))&1)) for site in range(4))) for b in basis])
    for g in range(ng):
        for i in range(d):
            H[g*d+i,g*d+i]+=2*kin
            if g+1<ng:
                H[g*d+i,(g+1)*d+i]-=kin
                H[(g+1)*d+i,g*d+i]-=kin
            else:
                H[g*d+i,i]-=kin*gauge[i]
                H[i,g*d+i]-=kin*gauge[i].conjugate()
    return H,basis,index


def eigensystem(n,p,ng=NGRID,k=NLOW):
    H,basis,index=coupled_matrix(n,p,ng)
    vals,vec=eigsh(H,k=min(k,H.shape[0]-2),which='SA',tol=1e-11,maxiter=20000)
    order=np.argsort(vals)
    return vals[order],vec[:,order],basis,index




def run():
    t0=time.time(); p=Parameters(); ng=NGRID
    neutral,psi2,b2,_=eigensystem(2,p,ng)
    sectors={n:eigensystem(n,p,ng) for n in (1,3)}
    basis1,i1=sectors[1][2],sectors[1][3]
    basis3,i3=sectors[3][2],sectors[3][3]
    e0=float(neutral[0]); psi=psi2[:,0].reshape(ng,len(b2))
    rows=[]
    for spin in (0,1):
      for k in (0.,np.pi/2):
        phase=np.exp(-1j*k*np.arange(4))/2
        add=np.zeros((len(basis3),len(b2)),complex)
        rem=np.zeros((len(basis1),len(b2)),complex)
        for site in range(4):
            mode=2*site+spin
            for j,b in enumerate(b2):
                sign=(-1)**((b&((1<<mode)-1)).bit_count())
                if not (b&(1<<mode)):
                    add[i3[b|(1<<mode)],j]+=phase[site]*sign
                else:
                    rem[i1[b^(1<<mode)],j]+=phase[site].conjugate()*sign
        vadd=np.concatenate([add@psi[g] for g in range(ng)])
        vrem=np.concatenate([rem@psi[g] for g in range(ng)])
        ea,va,_,_=sectors[3]; er,vr,_,_=sectors[1]
        addw=np.abs(va.conj().T@vadd)**2
        remw=np.abs(vr.conj().T@vrem)**2
        rows.append({"spin":"up" if spin==0 else "down","k_rad_per_cell":float(k),
          "addition":[{"omega":float(e-e0),"weight":float(w)} for e,w in zip(ea,addw) if w>1e-10],
          "removal":[{"omega":float(e0-e),"weight":float(w)} for e,w in zip(er,remw) if w>1e-10],
          "captured_addition_weight":float(addw.sum()),"captured_removal_weight":float(remw.sum()),
          "addition_threshold":float(ea[0]-e0),"removal_threshold":float(e0-er[0])})
    frozen={}
    for n in (1,2,3):
        _,_,inter,T=matter_matrices(n,p); H=inter.copy()
        for e in range(4): H+=T[e]+T[e].conj().T
        frozen[n]=np.linalg.eigvalsh(H)
    result={"law":LAW,"candidate":"finite dynamical uniform loop-flux coordinate, exact Peierls Hubbard matter, N=1/2/3",
      "parameters":p.__dict__,"state":"coupled N=2 exact lowest state; N=1 removal and N=3 addition Lehmann sectors; all spinful CAR configurations retained; finite flux-coordinate grid",
      "approximations":["one global electromagnetic loop-flux coordinate, not local Gauss-law-complete photon field","second-order finite-difference electric kinetic operator on Q in [0,2pi), with gauge-covariant large-gauge twisted boundary","20 lowest states per sector; omitted high-energy poles limit total spectral completeness","no relativistic negative-energy Dirac sea; N=1 is Hubbard removal, not positron"],
      "grid":{"N":ng,"Q_domain":"[0,2pi), gauge-covariant boundary","delta_Q":2*np.pi/ng,"electric_inertia":p.electric_inertia},"source_normalization":"c†_k=1/2 sum_l exp(-ik l)c†_l; k=0,pi/2; spin resolved",
      "coupled_neutral_energy":e0,"frozen_thresholds":{"addition":float(frozen[3][0]-frozen[2][0]),"removal":float(frozen[2][0]-frozen[1][0])},"responses":rows,
      "retarded_convention":"G^R(omega)=sum_add w/(omega-(E3-E2)+i eta)+sum_rem w/(omega+(E2-E1)+i eta), Fourier exp(-i omega t); energies t=1",
      "runtime":{"seconds":time.time()-t0,"python":sys.version,"numpy":np.__version__},"checks":{
        "spin_degeneracy_gap_max":max(abs(rows[i]["addition_threshold"]-rows[i+2]["addition_threshold"]) for i in (0,1)),
        "coupled_vs_frozen_addition_gap_shift":float(rows[0]["addition_threshold"]-(frozen[3][0]-frozen[2][0]))}}
    return result

if __name__=='__main__':
    out=run(); out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); out['input_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    target=ROOT/'rich_electron_20261014_hubbard_charged_result.json'
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({"result":str(target),"source_sha256":out['source_sha256'],"input_sha256":out['input_sha256'],"natural_exit":0,"checks":out['checks'],"thresholds":out['responses'][0]},indent=2))
