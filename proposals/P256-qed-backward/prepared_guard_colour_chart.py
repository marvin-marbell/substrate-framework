#!/usr/bin/env python3
"""All-order guarded field-only SU4 mode on the owner's neutral chart.

No state is prepared here. Coordinate arrays and Hamiltonians are supplied by A;
physical vector-derived generators are supplied by B or derived by its API.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from fractions import Fraction
from typing import Any, Callable
import numpy as np
from scipy.linalg import expm, expm_frechet, solve_sylvester
from scipy.integrate import solve_ivp
from noncompact_screened_gauss_dressing import frame_chart, source_paths, screened_marks
from sea_selection import SELECTED


@dataclass
class Jet2:
    """Actual derivatives, not Taylor coefficients, along one physical path."""
    value: np.ndarray
    first: np.ndarray
    second: np.ndarray

    def __matmul__(self, other: Jet2) -> Jet2:
        return Jet2(self.value@other.value,
                    self.first@other.value+self.value@other.first,
                    self.second@other.value+2*self.first@other.first+self.value@other.second)

    def adjoint(self) -> Jet2:
        return Jet2(*(a.conj().swapaxes(-1,-2) for a in (self.value,self.first,self.second)))


def constant(value: np.ndarray) -> Jet2:
    value=np.asarray(value,complex)
    return Jet2(value,np.zeros_like(value),np.zeros_like(value))


def inverse_square_root_jet(matrix: Jet2) -> Jet2:
    """Positive Hermitian inverse square root, including noncommuting q2."""
    eigen,modes=np.linalg.eigh(matrix.value)
    if eigen[0]<=1e-12:raise ValueError('inverse-square-root positivity guard failed')
    root=(modes*np.sqrt(eigen))@modes.conj().T
    first=solve_sylvester(root,root,matrix.first)
    second=solve_sylvester(root,root,matrix.second-2*first@first)
    inverse=np.linalg.inv(root)
    return Jet2(inverse,-inverse@first@inverse,
                2*inverse@first@inverse@first@inverse-inverse@second@inverse)


def guarded_frame_jet(phi: Jet2) -> tuple[Jet2,dict]:
    """Exact guarded least-intertwiner/polar SU4 frame q2 on ALL60 Phi.

    Eigenvector parallel phase is removed by determinant normalization;
    the residual Z4 choice is a local chart, cancelled in peredge SOURCE.
    """
    _,guard=frame_chart(phi.value)
    p,dp,ddp=phi.value,phi.first,phi.second
    gram=np.einsum('mij,nji->mn',p,p).real/4
    dg=(np.einsum('mij,nji->mn',dp,p)+np.einsum('mij,nji->mn',p,dp)).real/4
    ddg=(np.einsum('mij,nji->mn',ddp,p)+2*np.einsum('mij,nji->mn',dp,dp)+np.einsum('mij,nji->mn',p,ddp)).real/4
    white=inverse_square_root_jet(Jet2(gram,dg,ddg))
    gamma=np.einsum('mn,nij->mij',white.value,p)
    dgamma=np.einsum('mn,nij->mij',white.first,p)+np.einsum('mn,nij->mij',white.value,dp)
    ddgamma=np.einsum('mn,nij->mij',white.second,p)+2*np.einsum('mn,nij->mij',white.first,dp)+np.einsum('mn,nij->mij',white.value,ddp)
    L=np.concatenate([np.kron(np.eye(4),g)-np.kron(s.T,np.eye(4)) for g,s in zip(gamma,SELECTED)])
    dL=np.concatenate([np.kron(np.eye(4),g) for g in dgamma])
    ddL=np.concatenate([np.kron(np.eye(4),g) for g in ddgamma])
    K=L.conj().T@L;dK=dL.conj().T@L+L.conj().T@dL
    ddK=ddL.conj().T@L+2*dL.conj().T@dL+L.conj().T@ddL
    eigen,modes=np.linalg.eigh(K);v=modes[:,0]
    if eigen[1]-eigen[0]<=1e-8:raise ValueError('least-intertwiner eigenvalue gap failed')
    inverse=(modes[:,1:]/(eigen[1:]-eigen[0]))@modes[:,1:].conj().T
    de=np.vdot(v,dK@v).real
    dv=-inverse@dK@v
    ddv=-inverse@(ddK@v+2*(dK-de*np.eye(16))@dv)-float(np.vdot(dv,dv).real)*v
    candidate=Jet2(*(x.reshape(4,4,order='F') for x in (v,dv,ddv)))
    polar=candidate@inverse_square_root_jet(candidate.adjoint()@candidate)
    U,dU,ddU=polar.value,polar.first,polar.second
    inverseU=np.linalg.inv(U)
    logfirst=np.trace(inverseU@dU)
    logsecond=np.trace(inverseU@ddU-inverseU@dU@inverseU@dU)
    phase=np.exp(-.25j*np.angle(np.linalg.det(U)))
    first_phase=-logfirst/4;second_phase=-logsecond/4+logfirst*logfirst/16
    frame=Jet2(phase*U,phase*(dU+first_phase*U),
               phase*(ddU+2*first_phase*dU+second_phase*U))
    return frame,guard


def full_phi_frame_jets(phi: Jet2) -> tuple[Jet2,list]:
    """Preserve independent stochastic/base-coordinate jets at every endpoint."""
    pairs=[guarded_frame_jet(Jet2(*(v[x] for v in (phi.value,phi.first,phi.second)))) for x in range(len(phi.value))]
    return Jet2(*(np.asarray([getattr(pair[0],key) for pair in pairs]) for key in ('value','first','second'))),[pair[1] for pair in pairs]


def vector_derived_generators(law: Any, phi: np.ndarray, axis: int=2) -> tuple[np.ndarray,list]:
    """Full15 mass-orthogonal projection of the ACTUAL Cartesian vector flow.

    No pole replacement: Y-Dlambda is retained explicitly away from the core.
    Scalar mass I/chi cancels in this local connection solve.
    """
    spins=[];guards=[]
    j,k=((2,3),(3,1),(1,2))[axis]
    for field in phi:
        _,guard=frame_chart(field)
        tangents=1j*(law.generators[:,None]@field-field@law.generators[:,None])
        D=np.einsum('aij,bmji->bma',law.basis,tangents).real.reshape(15,60).T/4
        vector=np.zeros_like(field);vector[j]=-field[k];vector[k]=field[j]
        Y=np.einsum('aij,mji->ma',law.basis,vector).real.ravel()/4
        inertia=D.T@D
        eigen=np.linalg.eigvalsh(inertia)
        if eigen[0]<=1e-10:raise ValueError('full15 Gauss inertia guard failed')
        coefficient=np.linalg.solve(inertia,D.T@Y)
        normal=Y-D@coefficient
        guard.update(Gauss_inertia_min=float(eigen[0]),vector_normal_norm=float(np.linalg.norm(normal)),
                     normal_orthogonality_defect=float(np.linalg.norm(D.T@normal)))
        spins.append(np.einsum('a,aij->ij',coefficient,law.generators));guards.append(guard)
    return np.asarray(spins),guards


def physical_profile(law: Any, centre: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """C-infinity compact radial guard, not the historical C6 collar."""
    if not 0<=inner<outer:raise ValueError('invalid physical guard radii')
    # Wrap exact integers BEFORE binary64 conversion: legal N~4e29 has an
    # ulp larger than the entire physical guard. Floating N/2 erases it.
    origin=np.asarray(centre,dtype=object)
    if any(int(v)!=v for v in origin):raise ValueError('physical guard centre must be an exact lattice site')
    size=int(law.size);half=size//2
    displacement=np.asarray([[(int(site[a])-int(origin[a])+half)%size-half for a in range(3)]
                             for site in law.sites],dtype=float)
    radius=np.linalg.norm(displacement,axis=1)*law.band.a
    u=np.clip((radius-inner)/(outer-inner),0.,1.)
    alpha=np.ones_like(u);alpha[u>=1]=0.
    interior=(u>0)&(u<1);v=u[interior]
    left=np.exp(-1/v);right=np.exp(-1/(1-v))
    alpha[interior]=right/(left+right)
    return alpha


def field_only_flow(law: Any, phi: np.ndarray, links: np.ndarray,
                    generators: np.ndarray, alpha: np.ndarray, angle: float) -> tuple[Jet2,Jet2,np.ndarray]:
    """Exact autonomous flow; adjoint covariance keeps its own S(Phi) fixed.

    Gfr=GPhi+GU ONLY. No matter action or reduced Root action is added.
    """
    lam=np.asarray(alpha)[:,None,None]*generators
    gauge=np.asarray([expm(1j*angle*s) for s in lam])
    moved_phi=gauge[:,None]@phi@gauge[:,None].conj().swapaxes(-1,-2)
    moved_links=gauge[:,None]@links@gauge[law.target].conj().swapaxes(-1,-2)
    def comm(field):
        return 1j*(lam[:,None]@field-field@lam[:,None])
    p1=comm(moved_phi)
    left,right=lam[:,None],lam[law.target]
    u1=1j*(left@moved_links-moved_links@right)
    u2=-(left@left@moved_links-2*left@moved_links@right+moved_links@right@right)
    return Jet2(moved_phi,p1,comm(p1)),Jet2(moved_links,u1,u2),gauge


def time_ordered_gauge(generator: Callable[[float],np.ndarray], start: float, end: float,
                       tolerance: float=2e-11) -> tuple[np.ndarray,dict]:
    """Integrate dg/dt=i lambda(t)g, without replacing it by exp(i integral lambda).

    Mathematical solution is the exact time-ordered exponential; numerical
    DOP853 evaluation has explicit tolerance, not an exact-arithmetic claim.
    """
    shape=np.asarray(generator(start)).shape
    initial=np.broadcast_to(np.eye(4,dtype=complex),shape).copy()
    result=solve_ivp(lambda t,y:(1j*np.asarray(generator(t))@y.reshape(shape)).ravel(),
                     (start,end),initial.ravel(),method='DOP853',rtol=tolerance,atol=tolerance/10)
    if not result.success:raise RuntimeError(result.message)
    gauge=result.y[:,-1].reshape(shape)
    defect=float(np.linalg.norm(gauge.conj().swapaxes(-1,-2)@gauge-np.eye(4)))
    return gauge,dict(method='DOP853 timeorderedODE',rtol=tolerance,unitarity_defect=defect,nfev=result.nfev)


def exponential_jet(algebra: np.ndarray, first: np.ndarray, second: np.ndarray) -> Jet2:
    """Exact Frechet/block-exponential q2 for a noncommuting SU4 chart."""
    value,d1=expm_frechet(algebra,first)
    block=np.zeros((12,12),complex)
    for k in range(3):block[4*k:4*k+4,4*k:4*k+4]=algebra
    block[:4,4:8]=first;block[4:8,8:]=first;block[:4,8:]=second/2
    return Jet2(value,d1,2*expm(block)[:4,8:])


def target_tree(law: Any, links: Jet2, birth: int, reference: int,
                path: list[tuple[int,int,int,int]]|None=None) -> Jet2:
    """U_birth→Ref, multiplying in path order; negative edges use U†."""
    if path is None:path=source_paths(law,reference)[birth]
    tree=constant(np.eye(4,dtype=complex));current=birth
    for origin,target,edge,encoded in path:
        if origin!=current:raise ValueError('disconnected target-directed tree')
        axis=abs(encoded)-1
        if encoded>0 and (edge!=origin or int(law.target[edge,axis])!=target):
            raise ValueError('wrong positive link endpoint')
        if encoded<0 and (edge!=target or int(law.target[edge,axis])!=origin):
            raise ValueError('wrong negative link endpoint')
        factor=Jet2(*(a[edge,axis] for a in (links.value,links.first,links.second)))
        tree=tree@(factor if encoded>0 else factor.adjoint());current=target
    if current!=reference:raise ValueError('tree does not reach fixed reference')
    return tree


def uniform_target_tree(link: Jet2, born_z: int, reference_z: int, size: int) -> Jet2:
    """Historical native positive z-word correction: +(Ref−Born) modulo N."""
    tree=constant(np.eye(4,dtype=complex))
    for _ in range((reference_z-born_z)%size):tree=tree@link
    return tree


def centre_word_jet(word: Jet2, eta: float=18.) -> Jet2:
    """All four Z4 characters; exact quotient first/second chain rule."""
    if not 0<eta<=80:raise ValueError('eta outside original finite packet guard')
    centres=np.array([1,1j,-1,-1j])
    scores=eta*np.real(centres.conj()*np.trace(word.value))
    d=eta*np.real(centres.conj()*np.trace(word.first))
    dd=eta*np.real(centres.conj()*np.trace(word.second))
    weights=np.exp(scores-scores.max())
    denominator=weights.sum();numerator=np.dot(centres,weights)
    d0=np.dot(weights,d);d1=np.dot(centres*weights,d)
    dd0=np.dot(weights,dd+d*d);dd1=np.dot(centres*weights,dd+d*d)
    value=numerator/denominator
    first=(d1-value*d0)/denominator
    second=(dd1-value*dd0-2*first*d0)/denominator
    return Jet2(np.asarray(value),np.asarray(first),np.asarray(second))


def endpoint_source_jets(frame_birth: Jet2, tree: Jet2, frame_reference: Jet2,
                         theta: Jet2, displacement: np.ndarray, size: int,
                         quartet: np.ndarray|None=None, eta: float=18.,
                         scalar_jet: Jet2|None=None,labels: np.ndarray|None=None) -> Jet2:
    """A's actual whole-word scalar/source; all4Ref/raw complex quartet retained.

    This is NOT the different product-of-edge-scalars screened source.
    Both endpoint frames, harmonic jets and mixed product terms are explicit.
    """
    if quartet is None:quartet=np.array([1,.3j,-.2+.4j,.6])
    if labels is None:labels=np.array([np.eye(4),*(SELECTED[0]@SELECTED[j] for j in range(1,4))])
    Q=np.einsum('a,aij->ij',quartet,labels)/2
    scalar=centre_word_jet(frame_birth.adjoint()@tree@frame_reference,eta) if scalar_jet is None else scalar_jet
    word=frame_birth@constant(Q)@frame_reference.adjoint()
    h=-np.dot(theta.value,displacement)/size
    dh=-np.dot(theta.first,displacement)/size;ddh=-np.dot(theta.second,displacement)/size
    phase=np.exp(1j*h);p1=1j*dh*phase;p2=(1j*ddh-dh*dh)*phase
    s=scalar.value*phase
    ds=scalar.first*phase+scalar.value*p1
    dds=scalar.second*phase+2*scalar.first*p1+scalar.value*p2
    return Jet2(s*word.value,ds*word.value+s*word.first,
                dds*word.value+2*ds*word.first+s*word.second)


def target_source_jets(law: Any, frames: Jet2, links: Jet2, birth: int,reference: int,
                       theta: Jet2,quartet: np.ndarray|None=None,eta: float=18.,
                       labels: np.ndarray|None=None,
                       path: list[tuple[int,int,int,int]]|None=None) -> Jet2:
    """ORIGINAL product-peredge SOURCE with BOTH endpoints and interior frames."""
    if path is None:path=source_paths(law,reference)[birth]
    target_tree(law,links,birth,reference,path)
    scalar=Jet2(np.asarray(1.+0j),np.asarray(0j),np.asarray(0j))
    def frame(x):return Jet2(*(v[x] for v in (frames.value,frames.first,frames.second)))
    for x,y,edge,encoded in path:
        link=Jet2(*(v[edge,abs(encoded)-1] for v in (links.value,links.first,links.second)))
        if encoded<0:link=link.adjoint()
        s=centre_word_jet(frame(x).adjoint()@link@frame(y),eta)
        scalar=Jet2(scalar.value*s.value,scalar.first*s.value+scalar.value*s.first,
                    scalar.second*s.value+2*scalar.first*s.first+scalar.value*s.second)
    if labels is None:labels=np.array([np.eye(4),*(SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2)))])
    return endpoint_source_jets(frame(birth),constant(np.eye(4)),frame(reference),theta,
          law.sites[birth]-law.sites[reference],law.size,quartet,eta,scalar,labels)


def gauss_frame_jet(frame: np.ndarray, lam: np.ndarray, angle: float) -> Jet2:
    moved=expm(1j*angle*lam)@frame
    return Jet2(moved,1j*lam@moved,-lam@lam@moved)


def pullback_metric(law: Any, phi_tangents: np.ndarray, link_tangents: np.ndarray,
                    links: np.ndarray, chi: float) -> dict:
    """Induced MASS metric; native colour cometric .6chi, NOT .3chi.

    Phi basis Tr(Ba Bb)/4=delta; SU4 basis Tr(Ta Tb)=delta/2.
    Unrepresented partners are reported, not replaced by flat-Haar claims.
    """
    dp=np.einsum('aij,qxmji->qxma',law.basis,phi_tangents).real/4
    algebra=-1j*link_tangents@links.conj().swapaxes(-1,-2)
    du=2*np.einsum('aij,qxeji->qxea',law.generators,algebra).real
    P=dp.reshape(len(dp),-1);U=du.reshape(len(du),-1)
    mass=P@P.T/chi+U@U.T/(.6*chi)
    eigen=np.linalg.eigvalsh(mass)
    if eigen[0]<=1e-12*max(1.,float(eigen[-1])):raise ValueError('dependent coordinate modes; quotient redundancy before inversion')
    return dict(mass=mass,cometric=np.linalg.inv(mass),density=np.sqrt(np.linalg.det(mass)),
                full_Phi_coordinates=P.shape[1],full_link_coordinates=U.shape[1],
                represented_modes=len(dp),omitted_tangent_dimension=P.shape[1]+U.shape[1]-len(dp),
                Haar_measure_restricted_to_induced_submanifold=True,full_Haar_or_full_law_claim=False)


def metric_first_jet(law: Any, phi_tangents: np.ndarray, link_tangents: np.ndarray,
                     dphi_tangents: np.ndarray, dlink_tangents: np.ndarray,
                     links: np.ndarray, dlinks: np.ndarray, chi: float) -> dict:
    metric=pullback_metric(law,phi_tangents,link_tangents,links,chi)
    P=np.einsum('aij,qxmji->qxma',law.basis,phi_tangents).real.reshape(len(phi_tangents),-1)/4
    dP=np.einsum('aij,qxmji->qxma',law.basis,dphi_tangents).real.reshape(len(phi_tangents),-1)/4
    A=-1j*link_tangents@links.conj().swapaxes(-1,-2)
    dA=-1j*(dlink_tangents@links.conj().swapaxes(-1,-2)+link_tangents@dlinks.conj().swapaxes(-1,-2))
    U=2*np.einsum('aij,qxeji->qxea',law.generators,A).real.reshape(len(phi_tangents),-1)
    dU=2*np.einsum('aij,qxeji->qxea',law.generators,dA).real.reshape(len(phi_tangents),-1)
    dmass=(dP@P.T+P@dP.T)/chi+(dU@U.T+U@dU.T)/(.6*chi)
    inverse=metric['cometric']
    metric.update(mass_first=dmass,cometric_first=-inverse@dmass@inverse,
                  log_density_first=.5*np.trace(inverse@dmass))
    return metric


def kinetic_action(metric: dict,metric_partials: list[dict],gradient: np.ndarray,
                   hessian: np.ndarray) -> complex:
    """Induced chart Laplace-Beltrami, including ALL supplied cross/drift jets."""
    inverse=metric['cometric']
    if len(metric_partials)!=len(gradient):raise ValueError('all coordinate metric derivatives required')
    drift=np.zeros(len(gradient))
    for a,partial in enumerate(metric_partials):
        drift+=partial['cometric_first'][a,:]+partial['log_density_first']*inverse[a,:]
    return complex(-.5*(np.sum(inverse*hessian)+np.dot(drift,gradient)))


def colour_defects(law: Any,phi: np.ndarray,links: np.ndarray) -> tuple[float,float]:
    W=0.;P=0.
    for x in range(law.volume):
        for a in range(3):
            u=links[x,a];y=int(law.target[x,a])
            W+=float(np.linalg.norm(phi[x]-u@phi[y]@u.conj().T)**2)
        for a,b in ((0,1),(0,2),(1,2)):
            loop=links[x,a]@links[law.target[x,a],b]@links[law.target[x,b],a].conj().T@links[x,b].conj().T
            P+=float(np.linalg.norm(loop-np.eye(4))**2)/2
    return W,P


def onebody_coordinate_jets(law: Any,phi: Jet2,links: Jet2,em: Jet2) -> Jet2:
    """Actual original/improved path endpoint average, before any CAR projection."""
    matrices=[np.zeros((4*law.volume,4*law.volume),complex) for _ in range(3)]
    def field(x,scalar):
        return Jet2(*(np.einsum('m,mij->ij',scalar,v[x]) for v in (phi.value,phi.first,phi.second)))
    for x in range(law.volume):
        sx=slice(4*x,4*x+4)
        for matrix,v in zip(matrices,(phi.value,phi.first,phi.second)):matrix[sx,sx]+=law.onsite*v[x,0]
        for displacement,scalar in law.paths:
            axis=int(np.flatnonzero(displacement)[0]);length=int(displacement[axis])
            tree=constant(np.eye(4,dtype=complex));angles=np.zeros(3);y=x
            for _ in range(length):
                tree=tree@Jet2(*(v[y,axis] for v in (links.value,links.first,links.second)))
                angles+=np.asarray([v[y,axis] for v in (em.value,em.first,em.second)]).real
                y=int(law.target[y,axis])
            left=field(x,scalar)@tree;right=tree@field(y,scalar)
            coefficient=Jet2(*((getattr(left,key)+getattr(right,key))/2 for key in ('value','first','second')))
            phase=np.exp(1j*angles[0]);p1=1j*angles[1]*phase;p2=(1j*angles[2]-angles[1]**2)*phase
            jet=Jet2(phase*coefficient.value,p1*coefficient.value+phase*coefficient.first,
                     p2*coefficient.value+2*p1*coefficient.first+phase*coefficient.second)
            sy=slice(4*y,4*y+4)
            for matrix,v in zip(matrices,(jet.value,jet.first,jet.second)):
                matrix[sx,sy]+=v;matrix[sy,sx]+=v.conj().T
    return Jet2(*matrices)


def law_pullback(law_light: Any,law_heavy: Any,phi: np.ndarray,links: np.ndarray,
                 em: np.ndarray,Lambda: float,chi: float,J: float,K4: float,
                 kappa: float) -> dict:
    """Canonical coordinate terms BEFORE CAR/field compression; no selector species."""
    W,P=colour_defects(law_heavy,phi,links)
    gram=np.einsum('xmij,xnji->xmn',phi,phi).real/4
    return dict(quartic=Lambda*kappa*np.sum(gram*gram)/2,
                alignment=J*W/8,colour_magnetic=K4*P,
                heavy_onebody=-Lambda*law_heavy.onebody(phi,links,np.zeros((law_heavy.volume,3))),
                light_onebody=law_light.onebody(phi,links,em),
                Phi_cometric=chi,colour_cometric=.6*chi,
                kinetic_form='(1/2) integral density*g^ab*(partial_a psi)^*partial_b psi; metric_first_jet gives cross/drift terms',
                omitted_partners_not_integrated=True)


def guarded_generator_jet(law: Any, phi: np.ndarray, direction: np.ndarray,
                          axis: int=2,second_direction: np.ndarray|None=None) -> dict:
    """Exact implicit q2 of full15 VECTOR projection, including crossjets."""
    if second_direction is None:second_direction=np.zeros_like(direction)
    base,guards=vector_derived_generators(law,phi,axis)
    j,k=((2,3),(3,1),(1,2))[axis]
    jets=[]
    for field,dfield,ddfield in zip(phi,direction,second_direction):
        Ds=[];Ys=[]
        for f in (field,dfield,ddfield):
            tangent=1j*(law.generators[:,None]@f-f@law.generators[:,None])
            Ds.append(np.einsum('aij,bmji->bma',law.basis,tangent).real.reshape(15,60).T/4)
            y=np.zeros_like(f);y[j]=-f[k];y[k]=f[j]
            Ys.append(np.einsum('aij,mji->ma',law.basis,y).real.ravel()/4)
        D,dD,ddD=Ds;Y,dY,ddY=Ys
        M=D.T@D;dM=dD.T@D+D.T@dD;ddM=ddD.T@D+2*dD.T@dD+D.T@ddD
        b=D.T@Y;db=dD.T@Y+D.T@dY;ddb=ddD.T@Y+2*dD.T@dY+D.T@ddY
        c=np.linalg.solve(M,b);dc=np.linalg.solve(M,db-dM@c)
        ddc=np.linalg.solve(M,ddb-ddM@c-2*dM@dc)
        jets.append([np.einsum('a,aij->ij',v,law.generators) for v in (c,dc,ddc)])
    jets=np.asarray(jets)
    return dict(value=base,first=jets[:,1],second=jets[:,2],guards=guards)


def guarded_chart_jet(law: Any, phi: Jet2, links: Jet2, alpha: Jet2, angle: Jet2,
                      axis: int=2) -> tuple[Jet2,Jet2,Jet2]:
    """Cross-aware pullback of arbitrary base-coordinate/angle q2 paths."""
    s=guarded_generator_jet(law,phi.value,phi.first,axis,phi.second)
    a=alpha.value[:,None,None];da=alpha.first[:,None,None];dda=alpha.second[:,None,None]
    lam=a*s['value'];dl=da*s['value']+a*s['first']
    ddl=dda*s['value']+2*da*s['first']+a*s['second']
    A=1j*angle.value*lam
    dA=1j*(angle.first*lam+angle.value*dl)
    ddA=1j*(angle.second*lam+2*angle.first*dl+angle.value*ddl)
    gs=[exponential_jet(x,dx,ddx) for x,dx,ddx in zip(A,dA,ddA)]
    gauge=Jet2(*(np.asarray([getattr(g,key) for g in gs]) for key in ('value','first','second')))
    gp=Jet2(*(x[:,None] for x in (gauge.value,gauge.first,gauge.second)))
    gr=Jet2(*(x[law.target] for x in (gauge.value,gauge.first,gauge.second)))
    return gp@phi@gp.adjoint(),gp@links@gr.adjoint(),gauge


def native_control(model: Any) -> dict:
    """Use A's once-prepared object, not an independent neutral/model."""
    law=model.law;V=law.volume
    ip=int(np.argmin(np.linalg.norm(model.points,axis=1)))
    ic=int(np.argmax(np.linalg.norm(model.points,axis=1)))
    p=model.points[ip]*model.scales[0];c=model.points[ic]*model.scales[1]
    phi=np.tile(model.frame,(V,1,1,1));phi[:,0]+=p[0]*SELECTED[1]+p[1]*SELECTED[2]
    links=np.tile(np.eye(4,dtype=complex),(V,3,1,1))
    links[:,2]=expm(1j*(c[0]*SELECTED[1]+c[1]*SELECTED[2])/math.sqrt(8))
    spins,guards=vector_derived_generators(law,phi)
    birth=law.index[(1,1,2)];ref=model.reference_site
    alpha=np.zeros(V);alpha[birth]=1.
    base_tree=target_tree(law,constant(links),birth,ref)
    positive_word=uniform_target_tree(constant(links[birth,2]),2,1,law.size)
    winding_defect=float(np.linalg.norm(positive_word.value-base_tree.value))
    if winding_defect<1e-3:raise AssertionError('signed vs wrapped winding control lost ordinary holonomy')
    flowed,uj,g=field_only_flow(law,phi,links,spins,alpha,2*math.pi)
    tree=target_tree(law,uj,birth,ref)
    expected=g[birth]@base_tree.value@g[ref].conj().T
    holonomy=float(np.linalg.norm(tree.value-expected))
    omitted=float(np.linalg.norm(base_tree.value-expected))
    frames=np.asarray([frame_chart(f)[0] for f in phi])
    def framed(angle):
        jets=[gauss_frame_jet(f,a*s,angle) for f,a,s in zip(frames,alpha,spins)]
        return Jet2(*(np.asarray([getattr(j,key) for j in jets]) for key in ('value','first','second')))
    source=target_source_jets(law,framed(2*math.pi),uj,birth,ref,constant(np.zeros(3)))
    initial=target_source_jets(law,constant(frames),constant(links),birth,ref,constant(np.zeros(3)))
    eta=np.zeros(V,complex);eta[birth]=1
    producer=screened_marks(law,phi,links,np.zeros((V,3)),eta,ref,18.,harmonic=np.zeros(3))
    raw=np.array([1,.3j,-.2+.4j,.6])
    actual_producer=np.einsum('a,aij->ij',raw,producer[:,4*birth:4*birth+4])/2
    producer_defect=float(np.linalg.norm(actual_producer-initial.value))
    if producer_defect>1e-10:raise AssertionError('immutable ORIGINAL source producer mismatch')
    body=expm(-2j*math.pi*spins[birth])
    adverse_plus=float(np.linalg.norm(body@source.value-initial.value))
    adverse_minus=float(np.linalg.norm(body@source.value+initial.value))
    nearp,nearu,_=field_only_flow(law,phi,links,spins,alpha,math.pi-.13)
    near=target_source_jets(law,framed(math.pi-.13),nearu,birth,ref,constant(np.zeros(3)))
    second_wrong=float(np.linalg.norm(near.second-near.first@near.first))
    metric=pullback_metric(law,nearp.first[None],nearu.first[None],nearu.value,model.chi)
    if holonomy>1e-9 or omitted<1 or adverse_plus>1e-8 or adverse_minus<1 or second_wrong<1e-4:
        raise AssertionError('undiscriminating physical endpoint/omittedlink/plus/secondjet control')
    Bstate=np.asarray(model.B@model.state)
    # Material off-core/crossjet control: full Cartesian normal is NOT erased.
    off=phi.copy();off[birth,1]+=.015*law.basis[0]
    direction=np.zeros_like(phi);direction[:,0]=SELECTED[1]+.3*SELECTED[2]
    sj=guarded_generator_jet(law,off,direction)
    h=2e-4
    sp,_=vector_derived_generators(law,off+h*direction)
    sm,_=vector_derived_generators(law,off-h*direction)
    spin_first=float(np.linalg.norm((sp-sm)/(2*h)-sj['first']))
    spin_second=float(np.linalg.norm((sp-2*sj['value']+sm)/(h*h)-sj['second']))
    rng=np.random.default_rng(222161)
    gauge=np.asarray([expm(.07j*np.einsum('a,aij->ij',rng.normal(size=15),law.generators)) for _ in range(V)])
    cov,_=vector_derived_generators(law,gauge[:,None]@off@gauge[:,None].conj().swapaxes(-1,-2))
    covariance=float(np.linalg.norm(cov-gauge@sj['value']@gauge.conj().swapaxes(-1,-2)))
    divergence=0.
    for a,t in enumerate(law.generators):
        df=1j*(t@off[birth]-off[birth]@t)
        ds=guarded_generator_jet(law,off[birth:birth+1],df[None])['first'][0]
        divergence+=2*np.trace(t@ds).real
    pj,lj,_=guarded_chart_jet(law,Jet2(off,direction,np.zeros_like(off)),constant(links),
                            constant(alpha),Jet2(np.asarray(.71),np.asarray(.23),np.asarray(0.)))
    em=constant(np.zeros((V,3)))
    hj=onebody_coordinate_jets(law,pj,lj,em)
    def chart_at(q):
        s,_=vector_derived_generators(law,off+q*direction)
        pp,uu,_=field_only_flow(law,off+q*direction,links,s,alpha,.71+.23*q)
        return pp.value,uu.value
    pp,up=chart_at(h);pm,um=chart_at(-h)
    hp=law.onebody(pp,up,np.real(em.value));hm=law.onebody(pm,um,np.real(em.value))
    H0=law.onebody(pj.value,lj.value,np.real(em.value))
    first_error=float(np.linalg.norm((hp-hm)/(2*h)-hj.first))
    second_error=float(np.linalg.norm((hp-2*H0+hm)/(h*h)-hj.second))
    fullframes,_=full_phi_frame_jets(pj)
    theta=Jet2(np.array([.13,-.07,.11]),np.array([.02,.01,-.03]),np.zeros(3))
    fullsource=target_source_jets(law,fullframes,lj,birth,ref,theta)
    def actual_source(f,u,t):
        marks=screened_marks(law,f,u,np.zeros((V,3)),eta,ref,18.,harmonic=t/law.size)
        return np.einsum('a,aij->ij',raw,marks[:,4*birth:4*birth+4])/2
    bp=actual_source(pp,up,theta.value+h*theta.first)
    bm=actual_source(pm,um,theta.value-h*theta.first)
    b0=actual_source(pj.value,lj.value,theta.value)
    source_value_error=float(np.linalg.norm(fullsource.value-b0))
    source_first_error=float(np.linalg.norm((bp-bm)/(2*h)-fullsource.first))
    source_second_error=float(np.linalg.norm((bp-2*b0+bm)/(h*h)-fullsource.second))
    ref_base_jet=float(np.linalg.norm(fullframes.first[ref]))
    if source_value_error>1e-8 or source_first_error>2e-5 or source_second_error>2e-4 or ref_base_jet<1e-3:
        raise AssertionError('actual BOTHendpoint Phi/cross SOURCE q2 mismatch')
    pulled=law_pullback(law,model.heavy,pj.value,lj.value,np.real(em.value),model.Lambda,model.chi,
                        .2*model.Lambda**1.25,.7*model.Lambda**1.25,2/model.band.a)
    value_error=float(np.linalg.norm(pulled['light_onebody']-hj.value))
    partial=metric_first_jet(law,pj.first[None],lj.first[None],pj.second[None],lj.second[None],
                             lj.value,lj.first,model.chi)
    kinetic=kinetic_action(partial,[partial],np.array([.3j]),np.array([[.4]]))
    if max(spin_first,covariance,abs(divergence),first_error,value_error)>2e-6 or max(spin_second,second_error)>2e-4:
        raise AssertionError('full15/crossjet actual beforecompression Hamiltonian mismatch')
    return dict(holonomy_endpoint_defect=holonomy,omitted_link_partner_defect=omitted,
                tree_minus_body_minus_PLUS_defect=adverse_plus,wrong_fitted_MINUS_defect=adverse_minus,
                physical_second_not_first_squared_defect=second_wrong,
                full15_adjoint_covariance_defect=covariance,fielddependent_Gauss_coefficient_divergence=float(divergence),
                generator_first_jet_defect=spin_first,generator_second_jet_defect=spin_second,
                crossjet_H_first_defect=first_error,crossjet_H_second_defect=second_error,
                crossjet_SOURCE_value_defect=source_value_error,crossjet_SOURCE_first_defect=source_first_error,
                crossjet_SOURCE_second_defect=source_second_error,Ref_stochastic_frame_first_norm=ref_base_jet,
                actual_H_pullback_value_defect=value_error,actual_normal_vector_residual=sj['guards'][birth]['vector_normal_norm'],
                induced_kinetic_action=[kinetic.real,kinetic.imag],
                base_coordinate_changes_BOTH_Phi_endpoints=True,original_productperedge_source_primary=True,
                immutable_original_producer_defect=producer_defect,
                signed_path_not_wrapped_winding_defect=winding_defect,
                historical_A_source_raw_norm=float(np.linalg.norm(Bstate)),neutral_norm=float(np.linalg.norm(model.state)),
                actual_operator_signature=model.operator_signature(),Phi_ref_varies_in_uniform_base_chart=True,
                fixed_Ref_outside_profile=float(alpha[ref]),represented_guard_mode_mass=float(metric['mass'][0,0]),
                omitted_tangent_dimension=metric['omitted_tangent_dimension'],guards=guards,
                physical_SO3_equivalence_proved=False,full_Haar_or_full_law_claim=False,
                state_reprepared=False,sampled_native_coordinate=[int(ip),int(ic)])


@dataclass
class JetSeries:
    """Taylor coefficients, unlike Jet2's actual first/second derivatives."""
    coefficients: list[np.ndarray]

    @property
    def order(self) -> int:
        return len(self.coefficients)-1

    def __matmul__(self,other: JetSeries) -> JetSeries:
        if self.order!=other.order:raise ValueError('unequal series orders')
        return JetSeries([sum((self.coefficients[k]@other.coefficients[n-k] for k in range(n+1)),
                             np.zeros_like(self.coefficients[0]@other.coefficients[0]))
                          for n in range(self.order+1)])

    def adjoint(self) -> JetSeries:
        return JetSeries([c.conj().swapaxes(-1,-2) for c in self.coefficients])

    def derivative(self,q: int) -> np.ndarray:
        return math.factorial(q)*self.coefficients[q]


def constant_series(value: np.ndarray,order: int=6) -> JetSeries:
    value=np.asarray(value,complex)
    return JetSeries([value,*[np.zeros_like(value) for _ in range(order)]])


def scalar_product_series(left: list,right: list,order: int) -> list:
    return [sum(left[k]*right[n-k] for k in range(n+1)) for n in range(order+1)]


def exponential_series(coefficients: list) -> list:
    """Scalar exponential recurrence with the actual nonzero base value."""
    result=[np.exp(coefficients[0])]
    for n in range(1,len(coefficients)):
        result.append(sum(k*coefficients[k]*result[n-k] for k in range(1,n+1))/n)
    return result


def inverse_sqrt_series(matrix: JetSeries) -> JetSeries:
    eigen,modes=np.linalg.eigh(matrix.coefficients[0])
    if eigen[0]<=1e-12:raise ValueError('higher inverse-root positivity guard failed')
    root=(modes*np.sqrt(eigen))@modes.conj().T
    roots=[root];inverse=[np.linalg.inv(root)]
    for n in range(1,matrix.order+1):
        rhs=matrix.coefficients[n]-sum((roots[k]@roots[n-k] for k in range(1,n)),np.zeros_like(root))
        roots.append(solve_sylvester(root,root,rhs))
        inverse.append(-inverse[0]@sum((roots[k]@inverse[n-k] for k in range(1,n+1)),np.zeros_like(root)))
    return JetSeries(inverse)


def guarded_frame_series(phi: JetSeries) -> tuple[JetSeries,dict]:
    """Actual nonlinear whitening/eigenvector/polar/determinant jets through q6."""
    _,guard=frame_chart(phi.coefficients[0]);order=phi.order
    gram=JetSeries([sum((np.einsum('mij,nji->mn',phi.coefficients[k],phi.coefficients[n-k])
                         for k in range(n+1)),np.zeros((4,4),complex))/4 for n in range(order+1)])
    white=inverse_sqrt_series(gram)
    gamma=[sum((np.einsum('mn,nij->mij',white.coefficients[k],phi.coefficients[n-k])
                for k in range(n+1)),np.zeros((4,4,4),complex)) for n in range(order+1)]
    systems=[np.concatenate([np.kron(np.eye(4),g)-(np.kron(s.T,np.eye(4)) if n==0 else 0)
                             for g,s in zip(gs,SELECTED)]) for n,gs in enumerate(gamma)]
    K=[sum((systems[k].conj().T@systems[n-k] for k in range(n+1)),
           np.zeros((16,16),complex)) for n in range(order+1)]
    eigen,modes=np.linalg.eigh(K[0]);v=[modes[:,0]];e=[float(eigen[0])]
    gap=float(eigen[1]-eigen[0])
    if gap<=1e-8:raise ValueError('higher least-intertwiner eigen gap failed')
    reduced=(modes[:,1:]/(eigen[1:]-eigen[0]))@modes[:,1:].conj().T
    for n in range(1,order+1):
        forcing=sum((K[k]@v[n-k] for k in range(1,n+1)),np.zeros(16,complex))
        reaction=sum((e[k]*v[n-k] for k in range(1,n)),np.zeros(16,complex))
        en=np.vdot(v[0],forcing-reaction)
        if abs(en.imag)>1e-7*max(1.,abs(en.real)):raise ValueError('non-Hermitian higher eigenvalue jet')
        e.append(float(en.real))
        parallel=-.5*sum(np.vdot(v[k],v[n-k]).real for k in range(1,n))
        v.append(-reduced@(forcing-reaction)+parallel*v[0])
    candidate=JetSeries([x.reshape(4,4,order='F') for x in v])
    polar=candidate@inverse_sqrt_series(candidate.adjoint()@candidate)
    u=polar.coefficients
    logarithm=[-.25j*np.angle(np.linalg.det(u[0]))]
    for n in range(1,order+1):
        logarithm.append(-sum((n-k)*np.trace(u[k].conj().T@u[n-k]) for k in range(n))/(4*n))
    phase=exponential_series(logarithm)
    frame=JetSeries([sum((phase[k]*u[n-k] for k in range(n+1)),np.zeros((4,4),complex))
                     for n in range(order+1)])
    guard.update(actual_eigen_gap=gap)
    return frame,guard


def centre_scalar_series(word: JetSeries,eta: float=18.) -> list:
    """Exact original psi1/psi0 quotient, retaining denominator jets at every q."""
    if not 0<eta<=80:raise ValueError('eta outside original packet range')
    centres=np.array([1,1j,-1,-1j])
    scores=[eta*np.real(centres.conj()*np.trace(v)) for v in word.coefficients]
    scores[0]=scores[0]-max(scores[0])
    seeds=exponential_series(scores)
    denominator=[np.sum(w) for w in seeds];numerator=[np.dot(centres,w) for w in seeds]
    result=[]
    for n in range(word.order+1):
        result.append((numerator[n]-sum(denominator[k]*result[n-k] for k in range(1,n+1)))/denominator[0])
    return result


def primary_source_series(law: Any,frames: dict[int,JetSeries],links: dict[tuple[int,int],JetSeries],
                          birth: int,reference: int,theta: JetSeries,
                          quartet: np.ndarray|None=None,eta: float=18.) -> JetSeries:
    """All q0..q6 of ORIGINAL signed/peredge/fullRef SOURCE, before compression."""
    if quartet is None:quartet=np.array([1,.3j,-.2+.4j,.6])
    labels=np.array([np.eye(4),*(SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2)))])
    Q=np.einsum('a,aij->ij',quartet,labels)/2;order=theta.order
    scalar=[1.+0j,*[0j for _ in range(order)]]
    path=source_paths(law,reference)[birth]
    for x,y,edge,encoded in path:
        factor=links[edge,abs(encoded)-1]
        if encoded<0:factor=factor.adjoint()
        word=frames[x].adjoint()@factor@frames[y]
        scalar=scalar_product_series(scalar,centre_scalar_series(word,eta),order)
    displacement=law.sites[birth]-law.sites[reference]
    phase=exponential_series([-1j*np.dot(t,displacement)/law.size for t in theta.coefficients])
    scalar=scalar_product_series(scalar,phase,order)
    matrix=frames[birth]@constant_series(Q,order)@frames[reference].adjoint()
    return JetSeries([sum((scalar[k]*matrix.coefficients[n-k] for k in range(n+1)),np.zeros((4,4),complex))
                      for n in range(order+1)])


def rational_sqrt_lower(value: Fraction) -> Fraction:
    if value<=0:raise ValueError('positive rational square root required')
    return Fraction(math.isqrt(value.numerator*value.denominator),value.denominator)


def rational_inverse_root_majorants(g: list[Fraction],minimum: Fraction) -> list[Fraction]:
    """Exact outward recurrence for ||(sqrt G)^-1|| Taylor coefficients."""
    root_min=rational_sqrt_lower(minimum)
    roots=[Fraction(0)];inverse=[1/root_min]
    for n in range(1,len(g)):
        roots.append((g[n]+sum(roots[k]*roots[n-k] for k in range(1,n)))/(2*root_min))
        inverse.append(inverse[0]*sum(roots[k]*inverse[n-k] for k in range(1,n+1)))
    return inverse


def primary_source_derivative_bounds(side: int,profile_l1: Fraction,order: int=6,
                                     gram_min: Fraction=Fraction(1,40),
                                     eigen_gap: Fraction=Fraction(2),
                                     polar_min: Fraction=Fraction(1,4),
                                     Phi_Frobenius_max: Fraction=Fraction(8),
                                     eta: Fraction=Fraction(18),
                                     harmonic_word_norm: Fraction=Fraction(4),
                                     path_length_upper: int|None=None) -> dict:
    """Proved local ambient mixed q0..q6 bounds; no global chart or creator Z.

    Coordinates: all60V Cartesian Phi, all45V left SU4 Lie coordinates,
    theta=N Ah. Supplied guards must hold on the actual regular tube.
    """
    if side<3 or order<2 or order>6 or profile_l1<0:raise ValueError('invalid source graph domain')
    if min(gram_min,eigen_gap,polar_min,Phi_Frobenius_max,eta)<=0:raise ValueError('positive tube guards required')
    F=Fraction;zero=F(0);one=F(1)
    g=[zero,Phi_Frobenius_max,one,*[zero for _ in range(order-2)]]
    c=rational_inverse_root_majorants(g,gram_min)
    p=[Phi_Frobenius_max,F(2),*[zero for _ in range(order-1)]]
    gamma=scalar_product_series(c,p,order)
    L=[2*gamma[0]+8,*[2*x for x in gamma[1:]]]
    K=scalar_product_series(L,L,order)
    vector=[one];energy=[zero]
    for n in range(1,order+1):
        forcing=sum((K[k]*vector[n-k] for k in range(1,n+1)),zero)
        reaction=sum((energy[k]*vector[n-k] for k in range(1,n)),zero)
        energy.append(forcing+reaction)
        parallel=sum((vector[k]*vector[n-k] for k in range(1,n)),zero)/2
        vector.append((forcing+reaction)/eigen_gap+parallel)
    polar_gram=scalar_product_series(vector,vector,order)
    pc=rational_inverse_root_majorants(polar_gram,polar_min*polar_min)
    U=scalar_product_series(vector,pc,order);U[0]=F(2)
    h=[zero]
    for n in range(1,order+1):
        h.append(sum(((n-k)*U[k]*U[n-k] for k in range(n)),zero)/(4*n))
    phase=[one]
    for n in range(1,order+1):
        phase.append(sum((k*h[k]*phase[n-k] for k in range(1,n+1)),zero)/n)
    frame=scalar_product_series(phase,U,order);frame[0]=one
    link=[F(1,math.factorial(n)) for n in range(order+1)]
    word=scalar_product_series(scalar_product_series(frame,link,order),frame,order)
    score=[zero,*[4*eta*v for v in word[1:]]]
    weights=[one]
    for n in range(1,order+1):
        weights.append(sum((k*score[k]*weights[n-k] for k in range(1,n+1)),zero)/n)
    ratio=[one]
    for n in range(1,order+1):
        ratio.append(weights[n]+sum(weights[k]*ratio[n-k] for k in range(1,n+1)))
    # Binary power of a truncated polynomial: never enumerate an astronomical word.
    length=3*side+2 if path_length_upper is None else path_length_upper
    if length<3*(side-1):raise ValueError('bound omits an original signed path')
    power=[one,*[zero for _ in range(order)]];factor=ratio.copy();exponent=length
    while exponent:
        if exponent&1:power=scalar_product_series(power,factor,order)
        exponent>>=1
        if exponent:factor=scalar_product_series(factor,factor,order)
    harmonic=[harmonic_word_norm**n/F(math.factorial(n)) for n in range(order+1)]
    source=scalar_product_series(scalar_product_series(scalar_product_series(frame,power,order),frame,order),harmonic,order)
    # Qraw†Qraw has twofold paired eigenvalues and Tr=1.65, hence ||Qraw||²<=.825<1.
    point_mixed=[(n**n if n else 1)*source[n] for n in range(order+1)]
    CB=[16*profile_l1*c for c in point_mixed]
    V=side**3
    return dict(order=order,guards=dict(gram_min=gram_min,eigen_gap=eigen_gap,polar_min=polar_min,
                Phi_Frobenius_max=Phi_Frobenius_max),coordinate_convention='ambient60Phi HS/2 +45colour leftLie T +theta=NAh',
                volume=V,signed_path_length_upper=3*(side-1),source_word_length_upper=length,
                harmonic_word_norm_upper=harmonic_word_norm,
                frame_Taylor_upper=frame,frame_mixed_derivative_upper=[(n**n if n else 1)*frame[n] for n in range(order+1)],
                edge_ratio_Taylor_upper=ratio,
                point_Taylor_upper=source,point_mixed_derivative_upper=point_mixed,
                source_operator_mixed_derivative_upper=CB,CB=CB[0],
                ambient_gradient_squared_upper=(60*V+45*V)*CB[1]**2,
                canonical_M1_squared_upper=(F(1,2)*60*V+F(3,10)*45*V)*CB[1]**2,
                harmonic_gradient_squared_upper=harmonic_word_norm**2*CB[0]**2,
                creator_norm_lower=None,source_normalization_not_invented=True,
                regular_tube_only=True,wholeTree_replacement=False)


def quotient_source_jet_log_bounds(bounds: dict,Jchart_log2_upper: int) -> dict:
    """FaadiBruno using actual D^j map<=j! J^(j+1), not a linear-map fiction."""
    raw=bounds['source_operator_mixed_derivative_upper'];coefficients=[];logs=[rational_ceil_log2(raw[0])]
    for n in range(1,len(raw)):
        terms=[math.factorial(n)*math.comb(n-1,k-1)*raw[k]/math.factorial(k) for k in range(1,n+1)]
        coefficients.append(dict(order=n,J_powers=list(range(n+1,2*n+1)),positive_coefficients=terms))
        logs.append(rational_ceil_log2(sum(terms,Fraction(0)))+2*n*Jchart_log2_upper)
    return dict(map_contract='D^j map<=j! Jchart^(j+1), j1..6',positive_polynomial_coefficients=coefficients,
                log2_mixed_derivative_upper=logs,regular_tube_only=True)


def rational_ceil_log2(value: Fraction) -> int:
    if value<=0:raise ValueError('positive logarithmic bound required')
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    if exponent>=0:
        return exponent if value.numerator<=value.denominator<<exponent else exponent+1
    return exponent if value.numerator<<(-exponent)<=value.denominator else exponent+1


def source_physical_Lie_graph(bounds: dict,Y_norm: Fraction,DY_Y_norm: Fraction) -> dict:
    """True coordinate Lie q2: DB[D_Y Y]+D²B[Y,Y], not matrix firstjet squared."""
    C=bounds['source_operator_mixed_derivative_upper']
    return dict(first=C[1]*Y_norm,second=C[1]*DY_Y_norm+C[2]*Y_norm*Y_norm,
                acceleration_term=C[1]*DY_Y_norm,Hessian_term=C[2]*Y_norm*Y_norm,
                full_physical_vector_inputs_must_come_from_B=True)


def kinetic_source_commutator_coefficients(bounds: dict) -> dict:
    """Ambient full-kinetic Leibniz coefficients; owner supplies actual state graphs.

    [H,B]psi terms: chi*(Phi_gradient*||gradPhi psi||+colour_gradient*
    ||gradColour psi||+Phi_laplacian+colour_laplacian) plus UE/N harmonic.
    Coexact SOURCE derivatives are exactly zero; CAR/Coulomb/beta separate.
    """
    C=bounds['source_operator_mixed_derivative_upper']
    V=bounds['volume'];harmonic=bounds['harmonic_word_norm_upper']
    return dict(Phi_gradient_squared_coefficient=60*V*C[1]**2,
                colour_gradient_squared_coefficient=Fraction(9,25)*45*V*C[1]**2,
                Phi_laplacian_coefficient=Fraction(1,2)*60*V*C[2],
                colour_laplacian_coefficient=Fraction(3,10)*45*V*C[2],
                harmonic_gradient_squared_coefficient=4*harmonic**2*C[0]**2,
                harmonic_laplacian_coefficient=harmonic**2*C[0],coexact_gradient=0,
                beta_commutator_coefficient=2*C[0],beta_is_TOTAL_not_per_edge=True,
                owner_state_gradient_and_CAR_Coulomb_graphs_required=True,creator_Z_not_supplied=True)


def positive_polynomial_add(left: list[Fraction],right: list[Fraction]) -> list[Fraction]:
    return [(left[k] if k<len(left) else Fraction(0))+(right[k] if k<len(right) else Fraction(0))
            for k in range(max(len(left),len(right)))]


def positive_polynomial_product(left: list[Fraction],right: list[Fraction]) -> list[Fraction]:
    result=[Fraction(0)]*(len(left)+len(right)-1)
    for i,a in enumerate(left):
        for j,b in enumerate(right):result[i+j]+=a*b
    return result


def normal_coordinate_polynomials(raw_mixed: list[Fraction]) -> list[list[Fraction]]:
    """Actual positive FaadiBruno polynomial in J, no huge tower evaluation."""
    polynomials=[[raw_mixed[0]]]
    for n in range(1,len(raw_mixed)):
        p=[Fraction(0)]*(2*n+1)
        for k in range(1,n+1):
            p[n+k]=math.factorial(n)*math.comb(n-1,k-1)*raw_mixed[k]/math.factorial(k)
        polynomials.append(p)
    return polynomials


def polynomial_log2_upper(polynomial: list[Fraction],J_log2_upper: int) -> int:
    degree=max(k for k,value in enumerate(polynomial) if value)
    return rational_ceil_log2(sum(polynomial,Fraction(0)))+degree*J_log2_upper


def covariant_normal_source_bounds(bounds: dict,Jchart_log2_upper: int) -> dict:
    """Full light/Ref module-map connection, not45Phi partial derivatives.

    The ambient derivative bound is independently exported. Here the full
    comparison frame connection uses both module ends with representation
    norm<=4V+4 and actual chart jets j!J^(j+1). A supplies extra connections
    not belonging to this specified frame comparison; no unknown Berry
    connection is silently declared equal to this one.
    """
    order=bounds['order'];V=bounds['volume'];representation=4*V+4
    B=normal_coordinate_polynomials(bounds['source_operator_mixed_derivative_upper'])
    frames=normal_coordinate_polynomials(bounds['frame_mixed_derivative_upper'])
    A=[]
    for j in range(order):
        p=[Fraction(0)]
        for k in range(j+1):
            term=positive_polynomial_product(frames[k],frames[j-k+1])
            p=positive_polynomial_add(p,[2*representation*math.comb(j,k)*c for c in term])
        A.append(p)
    # Ordered covariant derivatives: product-rule counts majorize every
    # ordered direction sequence, not just the symmetric/diagonal part.
    parallel=[[Fraction(1)]]
    for n in range(1,order+1):
        p=[Fraction(0)]
        for k in range(n):
            term=positive_polynomial_product(parallel[n-1-k],A[k])
            p=positive_polynomial_add(p,[c/(n*math.factorial(k)) for c in term])
        parallel.append(p)
    covariant=[]
    for n in range(order+1):
        p=[Fraction(0)]
        for k in range(n+1):
            term=positive_polynomial_product(parallel[k],B[n-k])
            p=positive_polynomial_add(p,[c*math.factorial(n)/math.factorial(n-k) for c in term])
        covariant.append([2**n*c for c in p]) # cometric<=4, per contracted index.
    M1=positive_polynomial_product(covariant[1],covariant[1])
    M1=[45*V*c for c in M1] # (chi/2)*90V, metric already in contracted jets.
    return dict(frame_normal_polynomials=frames,raw_source_normal_polynomials=B,
                ordered_covariant_source_polynomials=covariant,
                covariant_source_log2_upper=[polynomial_log2_upper(p,Jchart_log2_upper) for p in covariant],
                frame_normal_log2_upper=[polynomial_log2_upper(p,Jchart_log2_upper) for p in frames],
                canonical_covariant_M1_squared_polynomial=M1,
                canonical_covariant_M1_squared_log2_upper=polynomial_log2_upper(M1,Jchart_log2_upper),
                frame_connection_derivative_polynomials=A,representation_bound=representation,
                chart_map_contract='D^j map<=j! J^(j+1), cometric<=4, regular MinSet tube only',
                additional_owner_Berry_connections_not_substituted=True,creator_Z_not_invented=True)


def matrix_exponential_series(base: np.ndarray,direction: np.ndarray,order: int=6) -> JetSeries:
    """All noncommuting Taylor coefficients from one triangular exponential."""
    block=np.zeros((4*(order+1),4*(order+1)),complex)
    for k in range(order+1):
        block[4*k:4*k+4,4*k:4*k+4]=base
        if k<order:block[4*k:4*k+4,4*k+4:4*k+8]=direction
    result=expm(block)
    return JetSeries([result[:4,4*k:4*k+4] for k in range(order+1)])


def exact_bound_capsule(value: Any) -> Any:
    if isinstance(value,Fraction):
        return dict(numerator_hex=hex(value.numerator),denominator_hex=hex(value.denominator))
    if isinstance(value,dict):return {str(k):exact_bound_capsule(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [exact_bound_capsule(v) for v in value]
    return value


def higher_source_native_control(state_capsule: Path,normal_capsule: Path) -> dict:
    """NEW q6 consumer of actual cached amplitudes and owner's quadrature API.

    No H constructor, previous30s discriminator, filter or preparation is run.
    Uncompressed fullRef coefficient graphs are evaluated at all81 actual
    coordinate nodes with weights from the SAME cached neutral wavefunction.
    """
    from prepared_neutral_spatial_well import pair_basis
    from improved_phi_band import selected_band,ImprovedPhiLaw,NSTAR
    directory=Path(__file__).resolve().parent
    source_paths_loaded={Path(filename).resolve() for module in sys.modules.values()
                         if isinstance(filename:=getattr(module,'__file__',None),str)
                         and Path(filename).suffix=='.py' and Path(filename).resolve().parent==directory}
    source_closure={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths_loaded}
    with np.load(state_capsule,allow_pickle=False) as data:
        state=np.asarray(data['state']);signature=str(data['operator_signature'])
    if signature!='ebb7be04cda391f56f02e25eacd30bc6031a9bcff89713151e54eb9ec39c4d9f':
        raise ValueError('not the owner cached actual H/state')
    if state.shape!=(9*56*6*2*70,) or abs(np.linalg.norm(state)-1)>1e-10:
        raise ValueError('cached neutral tensor/normalization mismatch')
    tube=json.loads(normal_capsule.read_bytes())
    constants=tube['legal_constants_log2_upper']
    Jlog=int(constants['quotient_chart_jet_upper']);rlog=int(constants['quotient_radius'])
    raw_delta_log=2+2*Jlog+rlog
    if raw_delta_log>=-20:raise ValueError('normal capsule does not imply the required regular SOURCE tube')
    band,base_phi,_=selected_band();law=ImprovedPhiLaw(3,band);V=law.volume;ref=law.index[(0,1,1)]
    points,Q,_,_=pair_basis(3);Q2=np.kron(Q,Q)
    if np.linalg.norm(Q2.T@Q2-np.eye(9))>1e-10:raise ValueError('owner quadrature embedding fails isometry')
    coordinate_wave=Q2@state.reshape(9,-1)
    weights=np.sum(abs(coordinate_wave)**2,axis=1)
    if abs(weights.sum()-1)>1e-10:raise ValueError('actual cached coordinate weights not normalized')
    xyz=law.sites.astype(float)-1
    profile=xyz[:,0]*np.exp(-np.sum(xyz*xyz,axis=1)/(2*.65**2))
    profile/=np.linalg.norm(profile)
    l1=Fraction.from_float(float(np.sum(abs(profile)))).limit_denominator(2**30)+Fraction(1,2**28)
    bounds=primary_source_derivative_bounds(3,l1)
    # Existing owner's Phi/colour planes, before any compression. The full
    # ambient direction has norm<1; both stochastic endpoint jets are present.
    dphi=np.zeros_like(base_phi);dphi[0]=.18*(SELECTED[1]+.3*SELECTED[2])
    colour_direction=1j*(.01*SELECTED[1]-.004*SELECTED[2])/math.sqrt(8)
    theta=constant_series(np.zeros(3));theta.coefficients[1]=np.array([.015,-.007,.011])
    coordinate_direction_squared=V*np.einsum('mij,mji->',dphi,dphi).real/4
    generator=-1j*colour_direction
    coordinate_direction_squared+=3*V*2*np.trace(generator@generator).real+np.dot(theta.coefficients[1],theta.coefficients[1]).real
    if coordinate_direction_squared>=1:raise ValueError('q6 consumer direction exceeds mixed-jet unit ball')
    observed=[];frame_observed=[];producer_errors=[]
    minima=dict(Gram_min=math.inf,eigen_gap=math.inf,polar_min=math.inf)
    identity=constant_series(np.eye(4));raw=np.array([1,.3j,-.2+.4j,.6])
    samples=[]
    for ip in range(len(points)):
        for ic in range(len(points)):
            p=points[ip]*.10;c=points[ic]*.14
            phi=base_phi.copy();phi[0]+=p[0]*SELECTED[1]+p[1]*SELECTED[2]
            phiseries=constant_series(phi);phiseries.coefficients[1]=dphi
            frame,guard=guarded_frame_series(phiseries)
            minima['Gram_min']=min(minima['Gram_min'],guard['Gram_min'])
            minima['eigen_gap']=min(minima['eigen_gap'],guard['actual_eigen_gap'])
            minima['polar_min']=min(minima['polar_min'],guard['polar_min'])
            if guard['Gram_min']<1/40 or guard['actual_eigen_gap']<2 or guard['polar_min']<1/4 or np.linalg.norm(phi)>8:
                raise ValueError('actual native sample outside the proven derivative guards')
            colour=matrix_exponential_series(1j*(c[0]*SELECTED[1]+c[1]*SELECTED[2])/math.sqrt(8),colour_direction)
            frames={x:frame for x in range(V)}
            links={(x,a):colour if a==2 else identity for x in range(V) for a in range(3)}
            jets=[primary_source_series(law,frames,links,x,ref,theta) for x in range(V)]
            graph=[4*sum(abs(profile[x])*float(np.linalg.norm(jets[x].derivative(q))) for x in range(V)) for q in range(7)]
            observed.append(graph)
            frame_observed.append([float(np.linalg.norm(frame.derivative(q))) for q in range(7)])
            fields=np.tile(phi,(V,1,1,1));actual_links=np.tile(np.eye(4,dtype=complex),(V,3,1,1));actual_links[:,2]=colour.coefficients[0]
            original=screened_marks(law,fields,actual_links,np.zeros((V,3)),np.ones(V),ref,18.,harmonic=np.zeros(3))
            original=np.einsum('a,axir->xir',raw,original.reshape(4,V,4,4))/2
            producer_errors.append(max(float(np.linalg.norm(jets[x].coefficients[0]-original[x])) for x in range(V)))
            samples.append((phi,colour,frame,jets))
    observed=np.asarray(observed)
    weighted=np.sqrt(weights@observed**2)
    upper=np.array([float(c) for c in bounds['source_operator_mixed_derivative_upper']])
    if np.any(weighted>upper*(1+1e-9)) or max(producer_errors)>1e-9:
        raise AssertionError('actual fullRef source q6/immutable producer exceeds proved local graph')
    # Same cached-coordinate node, genuine adverse changed SOURCE contracts.
    phi,colour,frame,jets=samples[0]
    birth=law.index[(1,1,2)];path=source_paths(law,ref)[birth]
    alpha=np.zeros(V);alpha[birth]=1.
    for x,y,_,_ in path:
        if x!=birth and x!=ref:alpha[x]=.5
        if y!=birth and y!=ref:alpha[y]=.5
    angle=math.pi-.13
    fields=np.tile(phi,(V,1,1,1));physical=fields.copy()
    for x,a in enumerate(alpha):
        cc=math.cos(a*angle);ss=math.sin(a*angle)
        physical[x,1]=cc*fields[x,1]-ss*fields[x,2]
        physical[x,2]=ss*fields[x,1]+cc*fields[x,2]
    Y=np.zeros_like(physical);acceleration=np.zeros_like(physical)
    Y[:,1]=-alpha[:,None,None]*physical[:,2];Y[:,2]=alpha[:,None,None]*physical[:,1]
    acceleration[:,1]=-alpha[:,None,None]**2*physical[:,1]
    acceleration[:,2]=-alpha[:,None,None]**2*physical[:,2]
    framejets,_=full_phi_frame_jets(Jet2(physical,Y,acceleration))
    links=np.tile(np.eye(4,dtype=complex),(V,3,1,1))
    true=target_source_jets(law,framejets,constant(links),birth,ref,constant(np.zeros(3)))
    whole=endpoint_source_jets(Jet2(*(v[birth] for v in (framejets.value,framejets.first,framejets.second))),
            constant(np.eye(4)),Jet2(*(v[ref] for v in (framejets.value,framejets.first,framejets.second))),
            constant(np.zeros(3)),law.sites[birth]-law.sites[ref],law.size,
            labels=np.array([np.eye(4),*(SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2)))]))
    boost=endpoint_source_jets(constant(framejets.value[birth]),constant(np.eye(4)),constant(framejets.value[ref]),
                              constant(np.zeros(3)),law.sites[birth]-law.sites[ref],law.size)
    noacc_frames,_=full_phi_frame_jets(Jet2(physical,Y,np.zeros_like(Y)))
    noacc=target_source_jets(law,noacc_frames,constant(links),birth,ref,constant(np.zeros(3)))
    fitted=float(np.linalg.norm(true.second-true.first@true.first))
    acceleration_defect=float(np.linalg.norm(true.second-noacc.second))
    whole_defect=float(np.linalg.norm(true.value-whole.value))
    boost_defect=float(np.linalg.norm(true.value-boost.value))
    # Independent stochastic Ref jet adverse, using the same actual uniform sample.
    uniform_frames=Jet2(np.tile(frame.coefficients[0],(V,1,1)),
                        np.tile(frame.derivative(1),(V,1,1)),np.tile(frame.derivative(2),(V,1,1)))
    colourjets=Jet2(np.tile(np.eye(4,dtype=complex),(V,3,1,1)),np.zeros((V,3,4,4),complex),np.zeros((V,3,4,4),complex))
    correct=target_source_jets(law,uniform_frames,colourjets,birth,ref,constant(np.zeros(3)))
    wrongframes=Jet2(uniform_frames.value.copy(),uniform_frames.first.copy(),uniform_frames.second.copy())
    wrongframes.first[ref]=0;wrongframes.second[ref]=0
    zero_ref=target_source_jets(law,wrongframes,colourjets,birth,ref,constant(np.zeros(3)))
    zero_ref_defect=float(np.linalg.norm(correct.first-zero_ref.first))
    if min(fitted,acceleration_defect,whole_defect,boost_defect,zero_ref_defect)<1e-5:
        raise AssertionError('q6 consumer fails to discriminate source/Ref/true physical acceleration errors')
    # Selected legal N: exact rational raw ambient graphs plus symbolic-J normal/covariant graphs.
    eta_l1=Fraction(math.isqrt(NSTAR**3)+1)
    legal=primary_source_derivative_bounds(NSTAR,eta_l1)
    covariant=covariant_normal_source_bounds(legal,Jlog)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in source_closure.items()):
        raise ValueError('scientific source changes during q6 consumer execution')
    return dict(state_signature=signature,neutral_norm=float(np.linalg.norm(state)),state_reprepared=False,
                owner_quadrature_nodes=len(weights),actual_coordinate_weights=weights.tolist(),
                source_graph_weighted_fullRef_upper=weighted.tolist(),source_graph_bound=[float(c) for c in bounds['source_operator_mixed_derivative_upper']],
                actual_frame_derivative_max=np.max(np.asarray(frame_observed),axis=0).tolist(),actual_guard_minima=minima,
                immutable_producer_max_defect=max(producer_errors),ambient_direction_norm=math.sqrt(coordinate_direction_squared),
                wrong_wholeTree_projection_defect=whole_defect,wrong_BOOST_basis_defect=boost_defect,
                zero_stochastic_Ref_derivative_defect=zero_ref_defect,wrong_fitted_second_jet_defect=fitted,
                omitted_physical_DY_Y_term_defect=acceleration_defect,
                raw_delta_log2_upper=raw_delta_log,legal_raw_bounds=exact_bound_capsule(legal),
                legal_covariant_bounds=exact_bound_capsule(covariant),
                legal_kinetic_commutator_coefficients=exact_bound_capsule(kinetic_source_commutator_coefficients(legal)),
                graph_scope='q0..6 fullRef beforecompression coefficient/CAR operator-majorant graphs on actual cached coordinate amplitudes; not new creator Z or full MinSet state',
                physical_Lie_scope='actual intrinsic Cartesian VECTOR component with DY_Y; full orbital/clover/Gauss/Noether comparison stays B-owned',
                inputs={str(state_capsule):hashlib.sha256(state_capsule.read_bytes()).hexdigest(),
                        str(normal_capsule):hashlib.sha256(normal_capsule.read_bytes()).hexdigest()},
                scientific_source_hashes=source_closure,scientific_sources_stable_during_execution=True,
                source_sha256=source_closure[str(Path(__file__).resolve())])


def raw_source19_centre_calibration(descriptor: dict,precision: int=256) -> dict:
    """Actual raw quartet/allRef/SIGNED Source19 calibration, without whitening."""
    from flint import arb,ctx
    from improved_phi_band import NSTAR
    from prepared_neutral_full_field import BETA
    N=int(descriptor['N']);D=N//2
    if N!=NSTAR or tuple(map(int,descriptor['origin']))!=(D,D,D) or tuple(map(int,descriptor['reference']))!=(0,D,D):
        raise ValueError('not the selected legal Source19 Born/FAR geometry')
    offsets=np.asarray(descriptor['compact_displacement'],dtype=int)
    weights=np.asarray(descriptor['source_eta'],dtype=complex)
    if offsets.shape!=(19,3) or weights.shape!=(19,) or len({tuple(d) for d in offsets})!=19:
        raise ValueError('actual Source19 support/weights required')
    if any(not 0<=D+int(x)<N for d in offsets for x in d):
        raise ValueError('Source19 support crosses the fixed coordinate cut')
    lengths=[D+int(d[0])+abs(int(d[1]))+abs(int(d[2])) for d in offsets]
    if min(lengths)<1:raise ValueError('coincident source/reference is not this protocol')
    square=lambda z:Fraction.from_float(float(z.real))**2+Fraction.from_float(float(z.imag))**2
    profile_square=sum((square(z) for z in weights),Fraction(0))
    raw_prefactor=sum((square(z) for z in BETA),Fraction(0))/2
    previous=ctx.prec;ctx.prec=precision
    try:
        def A(value: Fraction):return arb(value.numerator)/value.denominator
        c18=arb(36).tanh()
        gram_scalar=sum((A(square(z))*c18**(2*length) for z,length in zip(weights,lengths)),arb(0))/2
        squared=2*A(raw_prefactor)*gram_scalar
        wholeTree=A(raw_prefactor)*A(profile_square)*c18**2
        lower=Fraction(113,128)
        if not squared>A(lower*lower):raise AssertionError('actual raw Source19 norm lower failed')
        if not abs(wholeTree-squared)>arb('.03'):raise AssertionError('wholeTree calibration is not discriminating')
        return dict(N=N,source_offsets=offsets.tolist(),source_eta=[dict(real=float(z.real),imag=float(z.imag)) for z in weights],
                    signed_lengths=lengths,signed_small_lengths=[n-D for n in lengths],
                    raw_quartet=[dict(real=float(z.real),imag=float(z.imag)) for z in BETA],
                    raw_quartet_prefactor=exact_bound_capsule(raw_prefactor),formal_decimal_prefactor='33/40',
                    profile_norm_squared_exact=exact_bound_capsule(profile_square),profile_renormalized=False,
                    c18_interval=str(c18),quartet_Gram_scalar_interval=str(gram_scalar),
                    raw_source_norm_squared_interval=str(squared),raw_source_norm_interval=str(squared.sqrt()),
                    raw_source_norm_rational_lower=exact_bound_capsule(lower),
                    wrong_wholeTree_norm_squared_interval=str(wholeTree),wrong_wholeTree_norm_squared_defect_interval=str(abs(wholeTree-squared)),
                    allRef=True,input_whitened=False,all64_m_all8_s=True,
                    centre_scope='every actual positive finiteN Gram G_N(m,s); scalar centre/root seam holonomies; not G_infinity or a 512argmin native solve',
                    normalization_scope='B_SOURCE tau raw norm; not filtered norm, photon input /sqrt2, receiver Gram or Born probability')
    finally:ctx.prec=previous


def raw_source19_trial_norm_bound(descriptor: dict,normal: dict) -> dict:
    """Actual selected normalized Gaussian/cutoff tau lower, NOT filtered Z."""
    from improved_phi_band import NSTAR
    def F(value):
        if isinstance(value,dict):return Fraction(int(value['numerator_hex'],16),int(value['denominator_hex'],16))
        return Fraction(value)
    legal=normal['legal_N'];V=NSTAR**3
    if int(legal['size'])!=NSTAR or int(legal['physical_normal_dimension'])!=90*V:
        raise ValueError('actual selected full90V normal proof required')
    M=F(legal['gaussian_covariance_majorant']);J=F(legal['quotient_chart_jet_upper']);r=F(legal['quotient_radius'])
    Mlog=rational_ceil_log2(M);Jlog=rational_ceil_log2(J);rlog=rational_ceil_log2(r)
    Llower=int(legal['selected_log2_lambda_conservative_lower'])
    # The actual 19 entries each have modulus<1. Thus ||eta19||1<19<ceil(sqrt V),
    # despite the stored binary64 ||eta19||2 being slightly greater than one.
    weights=np.asarray(descriptor['source_eta'],complex)
    if not np.all(abs(weights)<1) or math.isqrt(V)+1<=19:raise ValueError('Source19 profile does not fit frozen source majorant')
    bounds=primary_source_derivative_bounds(NSTAR,Fraction(math.isqrt(V)+1))
    covariant=covariant_normal_source_bounds(bounds,Jlog)
    ambient_log=rational_ceil_log2(bounds['canonical_M1_squared_upper'])
    covariant_log=covariant['canonical_covariant_M1_squared_log2_upper']
    C1log=rational_ceil_log2(bounds['source_operator_mixed_derivative_upper'][1])
    # E_tau||y||² <=4 M Lambda^-3/4 is the conservative A/Normal trial bound.
    # Fphi on every light/ref site cancels SOURCE's outside frames in Btilde.
    # Its remaining ratio derivative majorant is a positive subseries of rawMq.
    ambient_eight_log=24+16*Jlog+4*ambient_log+4*Mlog-3*Llower
    covariant_eight_log=8+4*covariant_log+4*Mlog-3*Llower
    uniform_log=2+2*Jlog+C1log+rlog
    holonomy_log=2+2*Jlog+rlog+rational_ceil_log2(Fraction(NSTAR))
    if max(ambient_eight_log,covariant_eight_log)>-56 or uniform_log>-7:
        raise AssertionError('unchanged selected tower/tube does not pay raw SOURCE norm perturbation')
    if holonomy_log>-4:raise AssertionError('m-tube disjointness not established from actual holonomies')
    calibration=raw_source19_centre_calibration(descriptor)
    return dict(calibration=calibration,gaussian_covariance_log2_upper=Mlog,chart_log2_upper=Jlog,radius_log2_upper=rlog,
                selected_log2_Lambda_lower=Llower,normal_moment='E_tau||y||²<=4 M Lambda^-3/4; normalized half-density Riccati Gaussian with radial cutoff, no window',
                ambient_canonical_M1_squared_log2_upper=ambient_log,covariant_canonical_M1_squared_log2_upper=covariant_log,
                ambient_delta_formula='8 J² M1ambient sqrt(M) Lambda^-3/8',ambient_eight_times_delta_log2_upper=ambient_eight_log,
                covariant_delta_formula='2 M1covariant sqrt(M) Lambda^-3/8',covariant_eight_times_delta_log2_upper=covariant_eight_log,
                uniform_co_moving_delta_log2_upper=uniform_log,holonomy_tube_radius_log2_upper=holonomy_log,
                centre_norm_lower=exact_bound_capsule(Fraction(113,128)),delta_rational_upper=exact_bound_capsule(Fraction(1,128)),
                raw_tau_source_norm_lower=exact_bound_capsule(Fraction(7,8)),raw_tau_source_norm_squared_lower=exact_bound_capsule(Fraction(49,64)),
                all64_m_amplitudes='1/8; disjoint supports proved by holonomy distance, not exact Hamiltonian sectors',
                s_weights='arbitrary normalized weights on actual finiteN minimizer SET, including normalized ties; all8 proof valid; SOURCE preserves cycles',
                raw_Gram='pointwise .5 sum_x |w_x scalar_path(x)|² I4 for fixed centre light Slater transported by EXACTSOURCE Fx and fullRef; not generic EndMark or filtered Gram',
                harmonic_and_Coulomb_phase_cost='zero in centre/trial Gram calibration for all real theta/A because chiral offsite trace cancellation; phases retained in SOURCE',
                heavy_projector='actual normalized whole-heavy Kato fibre; heavy-even commutation means it supplies no SOURCE norm weight',
                filtered_norm_claim=False,whole_H_window_or_Born_claim=False)


def native_source19_norm_control(descriptor_path: Path,normal_receipt_path: Path,size: int=8) -> dict:
    """New all512 phase/chiral/SU4seam/source primitive probe, no oldCAR/q128."""
    import itertools
    from improved_physical_compton import decode
    from improved_phi_band import ImprovedPhiLaw
    from prepared_neutral_centred_current import PreparedCentredCurrent
    from prepared_neutral_full_field import BETA
    from noncompact_screened_gauss_dressing import centre_scalar
    descriptor=decode(json.loads(descriptor_path.read_bytes()))
    normal=json.loads(normal_receipt_path.read_bytes())
    if size<8 or size%4:raise ValueError('native size multiple4 is required to expose false bulk i^m convention')
    source_directory=Path(primary_source_derivative_bounds.__code__.co_filename).resolve().parent
    paths_loaded={Path(filename).resolve() for module in sys.modules.values()
                  if isinstance(filename:=getattr(module,'__file__',None),str)
                  and Path(filename).suffix=='.py' and Path(filename).resolve().parent==source_directory}
    source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths_loaded}
    gram=np.asarray(normal['centre_probe'][0]['actual_full10_gram']['G'],float)
    eigen,vectors=np.linalg.eigh(gram);root=(vectors*np.sqrt(eigen))@vectors.T
    phi=np.einsum('mn,nij->mij',root,SELECTED)
    current=PreparedCentredCurrent(frame=phi);law=ImprovedPhiLaw(size,current.band);V=law.volume;D=size//2
    frame=frame_chart(phi)[0];chi=np.linalg.multi_dot(list(SELECTED))
    labels=np.array([np.eye(4),*(SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2)))])/2
    Q=np.einsum('a,aij->ij',BETA,labels)
    offsets=np.asarray(descriptor['compact_displacement'],int);weights=np.asarray(descriptor['source_eta'],complex)
    origin=np.array([D,D,D]);reference=law.index[(0,D,D)]
    births=np.asarray([law.index[tuple(origin+d)] for d in offsets],int)
    paths=source_paths(law,reference)
    lengths=np.asarray([len(paths[x]) for x in births],int)
    c18=float(math.tanh(36));amplitude=weights*c18**lengths
    # Arbitrary retained harmonic and longitudinal Coulomb endpoint phases:
    phase=np.exp(-1j*((origin+offsets-law.sites[reference])@np.array([.37,-.19,.11])/size))
    phase*=np.exp(1j*(.13*offsets[:,0]+.07*offsets[:,1]**2-.11*offsets[:,2]))
    amplitude*=phase
    modes=np.asarray(list(itertools.product(range(size),repeat=3)),int)
    points=2*math.pi*modes/size
    expected=.5*np.vdot(amplitude,amplitude).real*np.eye(4)
    largest_chiral=0.;largest_Gram=0.;smallest_gap=math.inf;singleRef=0.;occupied_complement=0.
    raw_norms=[]
    for m in itertools.product(range(4),repeat=3):
        for s in itertools.product(range(2),repeat=3):
            shift=math.pi*(np.asarray(m)/2+np.asarray(s))/size
            momenta=points+shift
            h=np.broadcast_to(current.onsite*phi[0],(V,4,4)).copy()
            for displacement,Fcoef in current.coefficients:
                e=np.exp(1j*(momenta@displacement))[:,None,None]
                h+=e*Fcoef+e.conj()*Fcoef.conj().T
            energies,basis=np.linalg.eigh(h)
            if np.any(energies[:,1]>=0) or np.any(energies[:,2]<=0):raise AssertionError('actual light centre occupied/empty split failed')
            empty=basis[:,:,2:]@basis[:,:,2:].conj().swapaxes(-1,-2)
            occupied=basis[:,:,:2]@basis[:,:,:2].conj().swapaxes(-1,-2)
            occupied_complement=max(occupied_complement,float(np.max(np.linalg.norm(empty+occupied-np.eye(4),axis=(-2,-1)))))
            largest_chiral=max(largest_chiral,float(np.max(np.linalg.norm(chi@empty@chi-occupied,axis=(-2,-1)))))
            smallest_gap=min(smallest_gap,float(np.min(energies[:,2])))
            fourier=np.exp(-1j*(momenta@(origin+offsets).T))@amplitude/math.sqrt(V)
            nu=fourier[:,None,None,None]*labels[None]
            G=np.einsum('paij,pik,pbkj->ab',nu.conj(),empty,nu)
            largest_Gram=max(largest_Gram,float(np.linalg.norm(G-expected)))
            raw_norms.append(float(np.vdot(BETA,G@BETA).real))
            if m==(1,2,3) and s==(1,0,1):
                G_one=np.einsum('pai,pik,pbk->ab',nu[:,:,:,0].conj(),empty,nu[:,:,:,0])
                singleRef=float(np.linalg.norm(G_one-expected))
    # All64 actual colour seam classes: SU4 Cartan Nth root plus twisted Phi,
    # never a scalar bulk exp(i pi m/(2N))I4 or a repeated i^m edge.
    maximum_word=0.;maximum_SU4=0.;wrong_uniform_holonomy=0.
    representative=None
    for m in itertools.product(range(4),repeat=3):
        angles=math.pi*np.asarray(m)/(2*size)
        roots=np.asarray([np.diag(np.exp(1j*a*np.array([-3.,1.,1.,1.]))) for a in angles])
        g=np.asarray([np.diag(np.exp(-1j*np.dot(site,angles)*np.array([-3.,1.,1.,1.]))) for site in law.sites])
        frames=g@frame
        seam=np.tile(np.eye(4,dtype=complex),(V,3,1,1))
        for axis in range(3):seam[law.sites[:,axis]==size-1,axis]=(1j**m[axis])*np.eye(4)
        uniform=np.broadcast_to(roots,(V,3,4,4)).copy()
        actual_word=frames[:,None].conj().swapaxes(-1,-2)@uniform@frames[law.target]
        expected_word=np.asarray([[frame.conj().T@seam[x,j]@frame for j in range(3)] for x in range(V)])
        maximum_word=max(maximum_word,float(np.max(np.linalg.norm(actual_word-expected_word,axis=(-2,-1)))))
        maximum_SU4=max(maximum_SU4,float(np.max(abs(np.linalg.det(roots)-1))))
        if m==(1,2,3):representative=(g,seam,uniform,frames)
        wrong_uniform_holonomy=max(wrong_uniform_holonomy,abs((1j**m[0])**size-1j**m[0]))
    if representative is None:raise AssertionError('representative seam not produced')
    g,seam,uniform,frames=representative
    profile=np.zeros(V,complex);profile[births]=weights
    fields=np.tile(phi,(V,1,1,1));gauged=g[:,None]@fields@g[:,None].conj().swapaxes(-1,-2)
    harmonic=np.array([.37,-.19,.11])/size
    marks=screened_marks(law,fields,seam,np.zeros((V,3)),profile,reference,18.,harmonic=harmonic)
    gauge_marks=screened_marks(law,gauged,uniform,np.zeros((V,3)),profile,reference,18.,harmonic=harmonic)
    seam_matrix=np.einsum('a,axr->xr',BETA,marks).reshape(V,4,4)/2
    gauge_matrix=np.einsum('a,axr->xr',BETA,gauge_marks).reshape(V,4,4)/2
    covariance=float(np.max(np.linalg.norm(gauge_matrix-g@seam_matrix@g[reference].conj().T,axis=(-2,-1))))
    immutable_matrix=float(np.max(np.linalg.norm(seam_matrix[births]-weights[:,None,None]*c18**lengths[:,None,None]*np.exp(-1j*((origin+offsets-law.sites[reference])@harmonic))[:,None,None]*(frame@Q@frame.conj().T),axis=(-2,-1))))
    # Wrapped-positive x path crosses the colour seam; canonical signed path does not.
    x=law.index[(D,D,D)];W=np.eye(4,dtype=complex);y=x
    for _ in range(size-D):W=W@seam[y,0];y=int(law.target[y,0])
    wrapped=frame@Q@frame.conj().T
    wrapped*=centre_scalar(frame.conj().T@W@frame,18.,law.generators)[0]
    signed=(c18**D)*frame@Q@frame.conj().T
    signed_word=float(np.linalg.norm(wrapped-signed))
    boost=np.array([np.eye(4),*(SELECTED[0]@SELECTED[j] for j in range(1,4))])/2
    boost_defect=float(np.linalg.norm(Q-np.einsum('a,aij->ij',BETA,boost)))
    # Large off-tube colour angle is an explicit family discriminator only.
    U=expm(.7j*SELECTED[1]);edge_value=centre_scalar(frame.conj().T@U.conj().T@frame,18.,law.generators)[0]
    whole_value=centre_scalar(frame.conj().T@np.linalg.matrix_power(U.conj().T,D)@frame,18.,law.generators)[0]
    wholeTree_defect=float(abs(edge_value**D-whole_value))
    if largest_Gram>2e-11 or largest_chiral>2e-11 or covariance>2e-11 or immutable_matrix>2e-11 or maximum_word>2e-11:
        raise AssertionError('actual all512 source/sea or SU4 centre gauge identity failed')
    if min(singleRef,signed_word,boost_defect,wholeTree_defect,wrong_uniform_holonomy)<.05:
        raise AssertionError('Source19 hardware/convention adverse branch did not discriminate')
    trial=raw_source19_trial_norm_bound(descriptor,normal)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in source_hashes.items()):
        raise ValueError('scientific source changed during raw Source19 probe')
    return dict(trial_proof=trial,native_size=size,native_history_count=512,native_momentum_modes=V,
                frozen_actual_N5_Gram=gram.tolist(),frozen_Gram_case=normal['centre_probe'][0]['case'],
                native_Gram_scope='ONE frozen actualN5G checked acrossALL512m/s phases atN8; not512argmins or astronomicaltau simulation',
                fullRef_G_scalar_max_defect=largest_Gram,occupied_empty_complement_defect=occupied_complement,
                true_light_projector_Gamma5_complement_max_defect=largest_chiral,minimum_light_gap=smallest_gap,
                raw_native_source_norm_squared_range=[min(raw_norms),max(raw_norms)],
                all64_SU4_Cartan_seam_word_max_defect=maximum_word,SU4_root_determinant_defect=maximum_SU4,
                original_producer_matrix_defect=immutable_matrix,original_producer_SU4_gauge_covariance_defect=covariance,
                wrong_singleRef_Gram_defect=singleRef,wrong_BOOST_source_matrix_defect=boost_defect,
                BOOST_scalar_Gram_alone_does_not_discriminate=True,wrong_positive_wrapped_source_word_defect=signed_word,
                wrong_peredge_i_to_m_holonomy_defect=wrong_uniform_holonomy,wrong_wholeTree_scalar_defect=wholeTree_defect,
                wrong_wholeTree_probe_scope='large off-tube angle tests SOURCEfamily only; separateMP legalcentre calibration retainsfullc18D',
                inputs={str(descriptor_path):hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),str(normal_receipt_path):hashlib.sha256(normal_receipt_path.read_bytes()).hexdigest()},
                scientific_source_hashes=source_hashes,scientific_sources_stable_during_execution=True,
                filtered_SOURCE_norm_claim=False,old_fullCAR_or_q128_rerun=False)


def narrow_soft_profile_norm_certificate(packet: dict,N: int,panels: int=8192,precision: int=192) -> dict:
    """Exact finite-grid bounds for the supplied B cubature-normalized bump.

    Cubature constants are literal stored IEEE bits, not exact integrals.
    No profile, column, CAR amplitude or preparation hardware is retuned.
    """
    from flint import arb,ctx
    if panels<1024:raise ValueError('radial enclosure needs at least1024 panels')
    row=packet['packet_family']['records'][-1]
    w=Fraction.from_float(float(row['width']));p0=[Fraction.from_float(float(x)) for x in row['centre']]
    C0=Fraction.from_float(float(row['scalar_norm']))
    if N<1 or w<=0 or C0<=0 or len(p0)!=3:
        raise ValueError('positive actual packet/grid constants required')
    previous=ctx.prec;ctx.prec=precision
    try:
        def A(value):
            value=Fraction(value);return arb(value.numerator)/value.denominator
        # g(r)=exp(-2/(1-r²)), r<1, is decreasing. Bound r² separately
        # on each rational panel: a²g(b)<=r²g(r)<=b²g(a).
        density=[(arb(-2)/(1-(arb(i)/panels)**2)).exp() for i in range(panels)]+[arb(0)]
        lower=sum((A(Fraction(i,panels)**2)*density[i+1]/panels for i in range(panels)),arb(0))
        upper=sum((A(Fraction(i+1,panels)**2)*density[i]/panels for i in range(panels)),arb(0))
        I_lower=4*arb.pi()*A(w)**3*lower;I_upper=4*arb.pi()*A(w)**3*upper
        cell_radius=arb(3).sqrt()*arb.pi()/N
        if not cell_radius<A(w) or not all(A(abs(x)+w)+cell_radius<arb.pi() for x in p0):
            raise ValueError('finite grid too coarse or packet crosses the specified BZ cut')
        # Nearest-cell quadrature, for EVERY real mesh/Bloch shift.
        # ||grad g(|p-p0|/w)||<=4/w. A nonzero difference has either
        # its point or cell centre in the w-ball, so lies in w+delta.
        alias=4/A(w)*cell_radius*(4*arb.pi()/3)*(A(w)+cell_radius)**3
        discrete_lower=I_lower-alias;discrete_upper=I_upper+alias
        if not discrete_lower>0:raise AssertionError('actual finiteN profile lower is unpaid')
        lo,hi=Fraction(31,32),Fraction(33,32)
        low=discrete_lower/A(C0)**2;high=discrete_upper/A(C0)**2
        if not low>A(lo*lo) or not high<A(hi*hi):
            raise AssertionError('supplied actual cubature constant escapes finiteN profile bounds')
        conventions={'bare_scalar':dict(profile_l2_rational_lower=exact_bound_capsule(lo),profile_l2_rational_upper=exact_bound_capsule(hi),
            finiteN_profile_l2_squared_lower_interval=str(low),finiteN_profile_l2_squared_upper_interval=str(high),
            profile_l1_upper=exact_bound_capsule(hi*(math.isqrt(N**3)+1)),stored_global_denominator=exact_bound_capsule(C0),
            Fourier_convention='eta_hat(k)=(2pi/N)^(3/2)*f(p_k)/stored_scalarNorm; eta=unitary inverseDFT; every spatial coefficient retained')}
        return dict(N=N,panels=panels,precision_bits=precision,actual_row=row,
            exact_width=exact_bound_capsule(w),exact_centre=[exact_bound_capsule(x) for x in p0],
            stored_cubature_scalar_norm=exact_bound_capsule(C0),
            radial_integral_lower_interval=str(I_lower),radial_integral_upper_interval=str(I_upper),
            finite_grid_alias_upper_interval=str(alias),relative_alias_to_stored_scalar_norm_squared_interval=str(alias/A(C0*C0)),
            cell_radius_interval=str(cell_radius),profile_conventions=conventions,
            all_real_mesh_shifts=True,cubature_equals_exact_integral_claim=False,cubature_equals_finiteN_sum_claim=False,
            percolumn_normalization=False,spatial_tail_cut=False,
            current_B_binding='BARE scalar f/scalarNorm BEFORE new actual original peredge creator Z; no old native sqrt2, cOne or CAR normalizer in eta. The old P+(p) is Pauli output, not p-dependent raw input.')
    finally:ctx.prec=previous


def generic_scalar_profile_raw_norm_bound(profile: dict,normal: dict,convention: str='bare_scalar') -> dict:
    """RAW tau bound for the exact supplied scalar profile, not filtered Z."""
    from improved_phi_band import NSTAR
    from prepared_neutral_full_field import BETA
    def F(value):
        if isinstance(value,dict):return Fraction(int(value['numerator_hex'],16),int(value['denominator_hex'],16))
        return Fraction(value)
    if int(profile['N'])!=NSTAR:raise ValueError('selected legal finiteN profile certificate required')
    p=profile['profile_conventions'][convention];eta_lo=F(p['profile_l2_rational_lower']);eta_hi=F(p['profile_l2_rational_upper'])
    # A finite positive Taylor sum proves exp72>2^103, so
    # 1-c18=2/(exp72+1)<2^-102. Bernoulli applies to every original
    # SIGNED source path, including seams: central seam factors are unitary.
    exponential_lower=sum((Fraction(72)**k/math.factorial(k) for k in range(145)),Fraction(0))
    length=3*NSTAR+2;one_minus_c=Fraction(1,2**102);word_lower=1-length*one_minus_c
    if exponential_lower<=2**103 or word_lower<=Fraction(3,4):
        raise AssertionError('actual legalN fullword Bernoulli lower failed')
    square=lambda z:Fraction.from_float(float(z.real))**2+Fraction.from_float(float(z.imag))**2
    prefactor=sum((square(z) for z in BETA),Fraction(0))/2
    if prefactor<Fraction(4,5):raise AssertionError('literal fixed raw quartet norm budget failed')
    # .5Tr(Qraw†Qraw)>=4/5; cword>=3/4 therefore centreZ>=.6||eta||2.
    if Fraction(4,5)*Fraction(3,4)**2<Fraction(3,5)**2:raise AssertionError('centre rational norm estimate failed')
    legal=normal['legal_N'];V=NSTAR**3
    if int(legal['size'])!=NSTAR or int(legal['physical_normal_dimension'])!=90*V:
        raise ValueError('actual complete normal proof required')
    Mlog=rational_ceil_log2(F(legal['gaussian_covariance_majorant']));Jlog=rational_ceil_log2(F(legal['quotient_chart_jet_upper']))
    rlog=rational_ceil_log2(F(legal['quotient_radius']));Llower=int(legal['selected_log2_lambda_conservative_lower'])
    # Pure homogeneity of B, not an input renormalization: CS gives
    # ||eta||1<=sqrtV||eta||2, and every Cq scales by ||eta||2.
    unit=primary_source_derivative_bounds(NSTAR,Fraction(math.isqrt(V)+1))
    cov=covariant_normal_source_bounds(unit,Jlog)
    # C1 already bounds every ambient mixed-unit direction. M1²=43.5V*C1²
    # is a looser directional majorant too; do not invert .5/.3 kinetic
    # weights to infer an unweighted HS norm from this canonical quantity.
    if unit['canonical_M1_squared_upper']<unit['source_operator_mixed_derivative_upper'][1]**2:
        raise AssertionError('canonical M1 no longer dominates the mixed-direction bound')
    ambient_log=rational_ceil_log2(unit['canonical_M1_squared_upper']);cov_log=cov['canonical_covariant_M1_squared_log2_upper']
    C1log=rational_ceil_log2(unit['source_operator_mixed_derivative_upper'][1])
    ambient_eight=24+16*Jlog+4*ambient_log+4*Mlog-3*Llower
    cov_eight=8+4*cov_log+4*Mlog-3*Llower
    uniform=2+2*Jlog+C1log+rlog
    if max(ambient_eight,cov_eight)>-56 or uniform>-7:
        raise AssertionError('unchanged actual tower/tube fails homogeneous relative norm perturbation')
    relative_lower=Fraction(3,5)-Fraction(1,128);raw_lower=relative_lower*eta_lo
    globally_normalized_majorant_factor=eta_hi/raw_lower
    if raw_lower<Fraction(1,2) or globally_normalized_majorant_factor>2:
        raise AssertionError('actual SOFT source fails common preparation norm/graph budget')
    return dict(profile_convention=convention,profile_certificate=profile,
        source_seed='constant ORIGINAL spatial-bivector quartet /global2/fullRef; NO p-dependent input spinmatrix; allowed P+(p) occurs on the filled-sea CAR output',
        path_length_upper=length,c18_one_minus_rational_upper=exact_bound_capsule(one_minus_c),
        Bernoulli_fullword_lower=exact_bound_capsule(word_lower),exponential_Taylor_terms=145,exp72_greater_than_2pow103=True,
        raw_quartet_prefactor=exact_bound_capsule(prefactor),centre_norm_lower='(3/5)||eta||2',
        delta_bound='(1/128)||eta||2; homogeneity only, no input rescaling',
        raw_tau_norm_relative_lower=exact_bound_capsule(relative_lower),raw_tau_norm_rational_lower=exact_bound_capsule(raw_lower),
        raw_tau_norm_advertised_lower='1/2',raw_tau_norm_squared_advertised_lower='1/4',
        normalized_SOURCE_q0_q6_majorant_factor=exact_bound_capsule(globally_normalized_majorant_factor),
        common_existing_factor2_budget_retained=True,unit_profile_q6_bounds=exact_bound_capsule(unit),
        gaussian_M_log2_upper=Mlog,chart_J_log2_upper=Jlog,radius_log2_upper=rlog,selected_log2_Lambda_lower=Llower,
        ambient_M1_squared_log2_upper=ambient_log,covariant_M1_squared_log2_upper=cov_log,
        ambient_eight_times_relative_delta_log2_upper=ambient_eight,covariant_eight_times_relative_delta_log2_upper=cov_eight,
        uniform_relative_co_moving_delta_log2_upper=uniform,
        raw_Gram='pointwise .5 sum_x |eta_x S_path(x)|² I4 for exactFphi-transported centre negative Slater/fullRef; all64m/all8s/allrealharmonic/Coulomb phases',
        heavy='actual whole-heavy occupied Riesz/Kato fibre, even commutation; no source norm weight',
        coherence='m cutoff supports disjoint by previously proved holonomy bound; SOURCE preserves s cycles, beta neednotcommute',
        actual_filtered_norm_claim=False,physical_photon_sqrt2_or_Born_claim=False)


def native_narrow_soft_norm_control(packet_path: Path,normal_path: Path) -> dict:
    """New scalar norm/mesh/SOURCE probe only: no500/512/fullCAR/q128 rerun."""
    import inspect
    from improved_phi_band import NSTAR,ImprovedPhiLaw
    from prepared_neutral_centred_current import PreparedCentredCurrent
    from prepared_neutral_full_field import BETA
    from radial_band_noether import source_seed
    from noncompact_screened_gauss_dressing import centre_scalar
    packet=json.loads(packet_path.read_bytes());normal=json.loads(normal_path.read_bytes())
    if hashlib.sha256(inspect.getsource(source_seed).encode()).hexdigest()!=packet['consumed_api_sha256']['radial_band_noether.source_seed']:
        raise ValueError('actual B constant source_seed API changed')
    directory=Path(__file__).resolve().parent
    sources={Path(fn).resolve() for module in sys.modules.values() if isinstance(fn:=getattr(module,'__file__',None),str)
             and Path(fn).suffix=='.py' and Path(fn).resolve().parent==directory}
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    profile=narrow_soft_profile_norm_certificate(packet,NSTAR)
    proof={name:generic_scalar_profile_raw_norm_bound(profile,normal,name) for name in profile['profile_conventions']}
    coarse_rejected=False
    try:narrow_soft_profile_norm_certificate(packet,31,panels=1024)
    except ValueError:coarse_rejected=True
    coarse_points=2*math.pi*np.indices((31,)*3).reshape(3,-1).T/31
    coarse_points=(coarse_points+math.pi)%(2*math.pi)-math.pi
    row=profile['actual_row'];coarse_inside=np.sum((coarse_points-np.asarray(row['centre']))**2,axis=1)<row['width']**2
    coarse_count=int(np.count_nonzero(coarse_inside))
    if coarse_count:raise AssertionError('native coarse-grid adverse unexpectedly contains packet modes')
    if not coarse_rejected:raise AssertionError('cubature-to-coarse-finiteN false normalization not rejected')
    size=8;V=size**3
    gram=np.asarray(normal['centre_probe'][0]['actual_full10_gram']['G'],float)
    eigen,modes=np.linalg.eigh(gram);root=(modes*np.sqrt(eigen))@modes.T;phi=np.einsum('mn,nij->mij',root,SELECTED)
    current=PreparedCentredCurrent(frame=phi);law=ImprovedPhiLaw(size,current.band);reference=law.index[(0,size//2,size//2)]
    Fphi=frame_chart(phi)[0];chi=np.linalg.multi_dot(list(SELECTED))
    labels=np.array([np.eye(4),*(SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2)))])/2
    Q=np.einsum('a,aij->ij',BETA,labels)
    seed=source_seed(phi,current.band);expected_seed=float(centre_scalar(np.eye(4),18,law.generators)[0].real)*Fphi@Q@Fphi.conj().T/math.sqrt(2)
    seed_error=float(np.linalg.norm(seed-expected_seed))
    # Native broad bump ONLY checks exact unitary finiteDFT and CAR mechanics.
    # The normative selected narrow width/centre remain the frozen B inputs.
    points=2*math.pi*np.indices((size,)*3).reshape(3,-1).T/size
    signed=(points+math.pi)%(2*math.pi)-math.pi;distance=np.sum((signed-np.array([.2,0,0]))**2,axis=1)
    f=np.zeros(V);inside=distance<1;f[inside]=np.exp(-1/(1-distance[inside]))
    eta_hat=(2*math.pi/size)**1.5*f
    eta=np.fft.ifftn(eta_hat.reshape((size,)*3),norm='ortho').reshape(-1)
    parseval=abs(float(np.vdot(eta,eta).real-np.vdot(eta_hat,eta_hat).real))
    fields=np.tile(phi,(V,1,1,1));links=np.tile(expm(.003j*SELECTED[1]),(V,3,1,1));harmonic=np.array([.37,-.19,.11])/size
    primitive=screened_marks(law,fields,links,np.zeros((V,3)),eta,reference,18.,harmonic=harmonic).reshape(4,V,4,4)
    coefficient=np.einsum('a,axij->xij',BETA,primitive)/2
    body=Fphi.conj().T@coefficient@Fphi
    transformed=np.fft.fftn(body.reshape(size,size,size,4,4),axes=(0,1,2),norm='ortho').reshape(V,4,4)
    h=np.broadcast_to(current.onsite*phi[0],(V,4,4)).copy()
    for displacement,coef in current.coefficients:
        e=np.exp(1j*(points@displacement))[:,None,None];h+=e*coef+e.conj()*coef.conj().T
    energy,basis=np.linalg.eigh(h);empty=basis[:,:,2:]@basis[:,:,2:].conj().swapaxes(-1,-2)
    empty=Fphi.conj().T@empty@Fphi;occupied=np.eye(4)-empty
    norm2=float(np.einsum('pij,pik,pkj->',transformed.conj(),empty,transformed).real)
    trace2=.5*float(np.linalg.norm(body)**2)
    chiral=float(np.max(np.linalg.norm(chi@empty@chi-occupied,axis=(-2,-1))))
    negative_input=occupied@transformed
    wrong_norm2=float(np.einsum('pij,pik,pkj->',negative_input.conj(),empty,negative_input).real)
    one_ref=float(np.einsum('pi,pik,pk->',transformed[:,:,0].conj(),empty,transformed[:,:,0]).real)
    column_fitted=Q/np.linalg.norm(Q,axis=0)[None,:]
    fitted_defect=float(np.linalg.norm(column_fitted-Q))
    if parseval>1e-12 or seed_error>1e-10 or abs(norm2-trace2)>1e-10 or chiral>1e-10 or norm2<.01:
        raise AssertionError('new actual constant-seed/fullRef/finiteDFT control failed')
    if abs(wrong_norm2)>1e-10 or fitted_defect<.1 or abs(one_ref-norm2)<.01:
        raise AssertionError('new wrong p-dependent occupied-input/singleRef/percolumn controls failed')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in hashes.items()):
        raise ValueError('scientific source changed during new SOFT norm consumer')
    return dict(legal_profile_certificate=profile,raw_tau_proofs=proof,native_size=size,
        native_scope='one frozen actualN5 full10G; ONE512-mode N8 projector; native broad bump only finiteDFT/CAR control, NOT the selected narrow packet or512histories/argmins',
        native_parseval_defect=parseval,actual_B_constant_seed_matrix_defect=seed_error,native_fullRef_SOURCE_norm_squared=norm2,
        native_chiral_trace_expected_squared=trace2,native_chiral_complement_defect=chiral,
        wrong_pdependent_occupied_input_SOURCE_norm_squared=wrong_norm2,wrong_singleRef_norm_squared=one_ref,
        percolumn_normalization_matrix_defect=fitted_defect,wrong_coarse_grid_cubature_normalization_rejected=coarse_rejected,
        wrong_coarse_grid_actual_supported_modes=coarse_count,wrong_coarse_grid_actual_profile_norm_squared=0,
        normative_selected_narrow_profile_retuned=False,percolumn_normalization=False,
        inputs={str(packet_path):hashlib.sha256(packet_path.read_bytes()).hexdigest(),str(normal_path):hashlib.sha256(normal_path.read_bytes()).hexdigest()},
        scientific_source_hashes=hashes,scientific_sources_stable_during_execution=True,filtered_norm_claim=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-capsule',type=Path)
    parser.add_argument('--higher-source-bounds',action='store_true')
    parser.add_argument('--normal-capsule',type=Path)
    parser.add_argument('--source19-rawtrial-norm',action='store_true')
    parser.add_argument('--descriptor',type=Path)
    parser.add_argument('--normal-proof-receipt',type=Path)
    parser.add_argument('--soft-rawtrial-norm',action='store_true')
    parser.add_argument('--soft-packet-receipt',type=Path)
    args=parser.parse_args()
    if args.soft_rawtrial_norm:
        if args.soft_packet_receipt is None or args.normal_proof_receipt is None:
            parser.error('--soft-rawtrial-norm requires --soft-packet-receipt and --normal-proof-receipt')
        print(json.dumps(native_narrow_soft_norm_control(args.soft_packet_receipt,args.normal_proof_receipt),indent=2,sort_keys=True))
        return
    if args.source19_rawtrial_norm:
        if args.descriptor is None or args.normal_proof_receipt is None:
            parser.error('--source19-rawtrial-norm requires --descriptor and --normal-proof-receipt')
        print(json.dumps(native_source19_norm_control(args.descriptor,args.normal_proof_receipt),indent=2,sort_keys=True))
        return
    if args.state_capsule is None:parser.error('--state-capsule is required for native state consumers')
    if args.higher_source_bounds:
        if args.normal_capsule is None:parser.error('--higher-source-bounds requires --normal-capsule')
        print(json.dumps(higher_source_native_control(args.state_capsule,args.normal_capsule),indent=2,sort_keys=True))
        return
    from prepared_neutral_spatial_gauss_well import PreparedSpatialGaussQuantumSlice
    model=PreparedSpatialGaussQuantumSlice(state_capsule=args.state_capsule)
    result=native_control(model)
    result.update(inputs={str(args.state_capsule):hashlib.sha256(args.state_capsule.read_bytes()).hexdigest()},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='actual A native sameNPZ coordinate/state SOURCE sample; no reprepare; not MinSet/fullHaar/physicalSO3 proof')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    main()
