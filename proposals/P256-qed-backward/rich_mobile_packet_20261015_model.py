"""Physical source-generated CI of the 180-CAR spatial Peierls law.
No Gram sandwich, frozen sea, prescribed magnetic term, or external poles.
"""
from __future__ import annotations
from typing import Any
from itertools import product
from math import factorial
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, svd, expm
from scipy.special import eval_genlaguerre
from scipy.sparse.linalg import LinearOperator
from rich_common_colour_lepton_20261015_model import common_root


def car(mask: int, index: int, create: bool) -> Any:
    if bool(mask & (1 << index)) == create:
        return None
    return mask ^ (1 << index), (-1 if (mask & ((1 << index)-1)).bit_count()%2 else 1)


def geometry(c: dict) -> dict:
    cliff=common_root()['cliff']; cells=np.array(list(product(range(3),range(3),range(5))))
    lookup={tuple(x):i for i,x in enumerate(cells)}; nc=180
    mass=np.kron(np.eye(45),cliff[0]); components=[3*c['wilson']*mass,mass]; labels=['onsite','mass']; profiles=[np.zeros(8),np.zeros(8)]
    links=[]
    for axis in range(3):
        for level in (range(5) if axis<2 else [0]):
            forward=np.zeros((nc,nc),complex)
            for site,x in enumerate(cells):
                if axis<2 and x[2]!=level:continue
                y=x.copy();y[axis]=(y[axis]+1)%([3,3,5][axis]);target=lookup[tuple(y)]
                block=-.5j*c['velocity']*cliff[axis+1]-.5*c['wilson']*cliff[0]
                forward[4*site:4*site+4,4*target:4*target+4]+=block
                links.append((site,target,axis,level,block))
            profile=np.zeros(8)
            if axis<2:
                for band in range(2):
                    q=2*np.pi*(band+1)/5
                    profile[4*band+2*axis:4*band+2*axis+2]=c['charge']*np.array([np.cos(q*level),np.sin(q*level)])
            components.extend([forward+forward.conj().T,1j*(forward-forward.conj().T)])
            profiles.extend([profile,profile]);labels.extend([f'hop{axis}_{level}_cos',f'hop{axis}_{level}_sin'])
    reference=components[0]+c['coupling']*c['phi_center']*mass+sum(components[2::2])
    # Simultaneous kx,ky,kz and band eigenbasis: translation is an actual diagonal operator.
    orbitals=[];energies=[];momenta=[];blocks=[]
    for kx,ky,kz in product(range(3),range(3),range(5)):
        w=np.vstack([np.eye(4)*np.exp(2j*np.pi*(kx*x/3+ky*y/3+kz*z/5))/np.sqrt(45) for x,y,z in cells])
        e,u=eigh(w.conj().T@reference@w)
        for j in range(4):orbitals.append(w@u[:,j]);energies.append(e[j]);momenta.append((kx,ky,kz));blocks.append((kx,ky))
    order=np.argsort(np.array(energies)>=0,kind='stable');u=np.array(orbitals).T[:,order]
    e=np.array(energies)[order];mom=np.array(momenta)[order];bl=np.array(blocks)[order]
    if sum(e<0)!=90:raise ValueError('Reference must fill 90 negative modes')
    transformed=np.array([u.conj().T@h@u for h in components]);transformed[abs(transformed)<2e-13]=0
    return dict(cells=cells,cliff=cliff,components=np.array(components),profiles=np.array(profiles),labels=labels,reference=reference,
                orbitals=u,energies=e,momenta=mom,blocks=bl,transformed=transformed,sea=(1<<90)-1,offset=float(e[:90].sum()),links=links)


def shell_states(cut: int, axes: int=8) -> list:
    # Enumerate weak compositions, not a 3**8 Cartesian allocation.
    def comp(total: int,n: int) -> list:
        if n==1:return [(total,)]
        return [(i,)+tail for i in range(total+1) for tail in comp(total-i,n-1)]
    return [x for total in range(cut+1) for x in comp(total,axes)]


def field(c: dict,geo: dict) -> dict:
    states=shell_states(2);ns=len(states);a=[];look={s:i for i,s in enumerate(states)};width=c['shell_width'];inertia=c['volume']/2
    for axis in range(8):
        op=np.zeros((ns,ns),complex)
        for j,s in enumerate(states):
            if s[axis]:
                t=list(s);t[axis]-=1;op[look[tuple(t)],j]=np.sqrt(s[axis])
        a.append(op)
    q=np.array([width*(v+v.conj().T)/np.sqrt(2) for v in a]);p=np.array([-1j*(v-v.conj().T)/(np.sqrt(2)*width) for v in a])
    kinetic=np.zeros((ns,ns),complex);p2=[]
    for axis,v in enumerate(a):
        full=(np.diag([2*s[axis]+1 for s in states])-v@v-v.conj().T@v.conj().T)/(2*width**2)
        p2.append(full);kinetic+=full/(2*inertia)
    phi=c['phi_center']+2*c['phi_width']*(np.arange(5)-2)/5;pk=2*np.pi*np.fft.fftfreq(5,d=2*c['phi_width']/5)
    ft=np.exp(2j*np.pi*np.outer(np.arange(5),np.arange(5))/5)/np.sqrt(5);pp=ft@np.diag(pk)@ft.conj().T
    phi_h=pp@pp/(2*c['volume']*c['scalar_inertia'])+np.diag(c['volume']*c['scalar_stiffness']*(phi-c['phi_center'])**2/2)
    def disp(profile: np.ndarray) -> np.ndarray:
        result=np.ones((ns,ns),complex)
        for axis,coefficient in enumerate(profile):
            x=coefficient*width/np.sqrt(2);element=np.zeros((3,3),complex)
            for n,m in product(range(3),repeat=2):
                lo,hi=min(n,m),max(n,m)
                element[n,m]=np.exp(-x*x/2)*np.sqrt(factorial(lo)/factorial(hi))*(1j*x)**(hi-lo)*eval_genlaguerre(lo,hi-lo,x*x)
            idx=np.array(states)[:,axis];result*=element[idx[:,None],idx[None,:]]
        return result
    eye=np.eye(ns);terms=[np.eye(225),np.kron(np.diag(c['coupling']*phi),eye)];off=[np.eye(225),terms[1]];derivatives=np.zeros((8,len(geo['components']),225,225),complex)
    for k in range(2,len(geo['components']),2):
        d=disp(geo['profiles'][k]);co=(d+d.conj().T)/2;si=(d-d.conj().T)/(2j)
        terms.extend([np.kron(np.eye(5),co),np.kron(np.eye(5),si)]);off.extend([np.eye(225),np.zeros((225,225))])
        for axis in range(8):
            derivatives[axis,k]=-geo['profiles'][k,axis]*terms[k+1]
            derivatives[axis,k+1]=geo['profiles'][k,axis]*terms[k]
    lz=np.zeros((ns,ns),complex);pol=np.zeros_like(lz)
    for band in range(2):
        for polar in range(2):
            cc,ss=4*band+2*polar,4*band+2*polar+1
            lz+=(band+1)*1j*(a[ss].conj().T@a[cc]-a[cc].conj().T@a[ss])
        for phase in range(2):
            x,y=4*band+phase,4*band+2+phase
            pol+=1j*(a[y].conj().T@a[x]-a[x].conj().T@a[y])
    translation=expm(-2j*np.pi*lz/5)
    lvalues,lvectors=eigh(lz)
    large=shell_states(3);ll={s:i for i,s in enumerate(large)};qrect=[];prect=[]
    for axis in range(8):
        qr=np.zeros((len(large),ns),complex);pr=np.zeros_like(qr)
        for j,s in enumerate(states):
            t=list(s);t[axis]+=1;qr[ll[tuple(t)],j]=width*np.sqrt(t[axis]/2);pr[ll[tuple(t)],j]=1j*np.sqrt(t[axis]/2)/width
            if s[axis]:
                t=list(s);t[axis]-=1;qr[ll[tuple(t)],j]=width*np.sqrt(s[axis]/2);pr[ll[tuple(t)],j]=-1j*np.sqrt(s[axis]/2)/width
        qrect.append(qr);prect.append(pr)
    return dict(states=states,q=q,p=p,qrect=np.array(qrect),prect=np.array(prect),kinetic=kinetic,p2=np.array(p2),phi=phi,phi_h=phi_h,
                boson=np.kron(phi_h,eye)+np.kron(np.eye(5),kinetic),terms=np.array(terms),off=np.array(off),derivatives=derivatives,
                translation=translation,translation_values=np.rint(lvalues).astype(int)%5,translation_vectors=lvectors,pol=pol,inertia=inertia)


def tangent(geo: dict) -> dict:
    sea=geo['sea'];masks={sea};blocks=[np.flatnonzero(np.all(geo['blocks']==b,axis=1)) for b in product(range(3),repeat=2)]
    local_pairs=[];selection=[]
    for indices in blocks:
        occupied=[int(j) for j in indices if j<90];empty=[int(i) for i in indices if i>=90]
        for i,j in product(empty,occupied):masks.add(sea^(1<<j)^(1<<i))
        chosen=[]
        for band,polar in product(range(2),repeat=2):
            weights=np.zeros((180,180),complex)
            for k in range(2,len(geo['components']),2):
                weights+=geo['profiles'][k,4*band+2*polar]*geo['transformed'][k+1]
            candidates=sorted([(float(abs(weights[i,j])),i,j) for i,j in product(empty,occupied)],reverse=True)
            first=candidates[0];second=next(x for x in candidates if x[1]!=first[1] and x[2]!=first[2])
            if first[0]<1e-12 or second[0]<1e-12:raise ValueError('Nonzero two-current physical product required')
            mask=sea^(1<<first[1])^(1<<first[2])^(1<<second[1])^(1<<second[2]);masks.add(mask)
            chosen.append((first[1],first[2]));selection.append(dict(kind='local_two_current',band=band+1,polar=polar,pairs=[first[1:],second[1:]],weights=[first[0],second[0]]))
        local_pairs.append(chosen[0])
    for a in range(9):
        for b in range(a+1,9):
            i,j=local_pairs[a];k,l=local_pairs[b];masks.add(sea^(1<<i)^(1<<j)^(1<<k)^(1<<l))
            selection.append(dict(kind='cross_block_two_current',blocks=[a,b],pairs=[[i,j],[k,l]]))
    masks=sorted(masks)
    return dict(masks=masks,selection=selection,isometry=0.0,discarded=0.0,singular=np.ones(len(masks)))


def source_sector(neutral: dict, indices: np.ndarray, create: bool) -> tuple:
    masks=set();actions=[]
    for j in indices:
        act=[]
        for k,m in enumerate(neutral['masks']):
            hit=car(m,int(j),create)
            if hit is not None:masks.add(hit[0]);act.append((hit[0],k,hit[1]))
        actions.append(act)
    masks=sorted(masks);look={m:i for i,m in enumerate(masks)};sources=[]
    for act in actions:
        rows=[look[m] for m,k,s in act];cols=[k for m,k,s in act];values=[s for m,k,s in act]
        sources.append(sparse.csr_matrix((values,(rows,cols)),shape=(len(masks),len(neutral['masks']))))
    return dict(masks=masks,isometry=0.0,discarded=0.0,singular=np.ones(len(masks))),sources


def onebody(masks: list, h: np.ndarray) -> sparse.csr_matrix:
    look={m:i for i,m in enumerate(masks)};rows=[];cols=[];vals=[]
    targets=[[(int(i),h[i,j]) for i in np.flatnonzero(abs(h[:,j])>1e-13) if i!=j] for j in range(180)]
    diag=np.diag(h)
    for k,mask in enumerate(masks):
        occupied=[j for j in range(180) if mask&(1<<j)]
        value=sum(diag[j] for j in occupied)
        if abs(value)>1e-14:rows.append(k);cols.append(k);vals.append(value)
        for j in occupied:
            first=mask^(1<<j);sgn=-1 if (mask&((1<<j)-1)).bit_count()%2 else 1
            for i,v in targets[j]:
                if first&(1<<i):continue
                dest=look.get(first|(1<<i))
                if dest is not None:
                    parity=-1 if (first&((1<<i)-1)).bit_count()%2 else 1
                    rows.append(dest);cols.append(k);vals.append(v*sgn*parity)
    return sparse.csr_matrix((vals,(rows,cols)),shape=(len(masks),len(masks)))


def project(sec: dict, h: np.ndarray) -> sparse.csr_matrix:
    return onebody(sec['masks'],h)


def install(sec: dict, f: dict, offset: float, off: bool=False) -> LinearOperator:
    matrices=sec['matrices'];terms=[sparse.csr_matrix(k) for k in f['off' if off else 'terms']];boson=sparse.csr_matrix(f['boson']);n=matrices[0].shape[0];nb=225
    def action(vector: np.ndarray,adjoint: bool=False) -> np.ndarray:
        x=np.asarray(vector).reshape(n,nb);b=boson.conj().T if adjoint else boson
        result=(b@x.T).T-offset*x
        for m,k in zip(matrices,terms):
            mm=m.conj().T if adjoint else m;kk=k.conj().T if adjoint else k
            result+=(kk@(mm@x).T).T
        return result.ravel()
    def mm(x: np.ndarray) -> np.ndarray:return np.column_stack([action(v) for v in x.T])
    def rmm(x: np.ndarray) -> np.ndarray:return np.column_stack([action(v,True) for v in x.T])
    op=LinearOperator((n*nb,n*nb),matvec=action,rmatvec=lambda x:action(x,True),matmat=mm,rmatmat=rmm,dtype=complex)
    sec['trace']=n*boson.diagonal().sum()-offset*n*nb+sum(m.diagonal().sum()*k.diagonal().sum() for m,k in zip(matrices,terms))
    return op


def apply_field(state: np.ndarray, op: np.ndarray) -> np.ndarray:
    x=state.reshape(-1,5,45);return np.einsum('ab,mpb->mpa',op,x,optimize=True).ravel()


def apply_terms(sec: dict, terms: np.ndarray, state: np.ndarray) -> np.ndarray:
    x=state.reshape(-1,225);out=np.zeros_like(x)
    for m,k in zip(sec['matrices'],terms):
        if np.any(k):out+=(sparse.csr_matrix(k)@(m@x).T).T
    return out.ravel()


def expectation(state: np.ndarray, op: np.ndarray) -> float:
    x=state.reshape(-1,225);return float(np.vdot(x,op@x).real)


def translation(sec: dict, geo: dict) -> np.ndarray:
    phases=[]
    for mask in sec['masks']:
        kz=sum(int(geo['momenta'][j,2]) for j in range(180) if mask&(1<<j))
        phases.append(np.exp(-2j*np.pi*kz/5))
    return np.array(phases)


def momentum(sec: dict, f: dict, state: np.ndarray) -> dict:
    tm=sec['translation'];x=state.reshape(-1,5,45)
    labels=np.rint(-np.angle(tm)*5/(2*np.pi)).astype(int)%5
    amplitudes=(x.reshape(-1,45)@f['translation_vectors'].conj()).reshape(x.shape)
    weights=np.sum(abs(amplitudes)**2,axis=1);joint=np.zeros((5,5))
    for m,n in product(range(5),repeat=2):
        joint[m,n]=float(weights[labels==m][:,f['translation_values']==n].sum())
    return dict(joint=joint.tolist(),matter=joint.sum(axis=1).tolist(),field=joint.sum(axis=0).tolist(),total=[float(sum(joint[m,n] for m,n in product(range(5),repeat=2) if (m+n)%5==k)) for k in range(5)])
