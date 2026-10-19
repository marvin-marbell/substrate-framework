"""Initial full129 canonical cubic mixed jet, not evolved Wick or evolution.

Jacobi: M3_ij=128 <3 adH² Cij+2 adH[Jj,Ji]+[Jj,adH Ji]>.
Right-normal electric words use F_w=128s_w+64|w|². The implementation
aggregates connected same-copy CAR chains before initial contraction. All
numbers are CONDITIONAL binary64; no outward rounding or sign admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, cast
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as current
import joined_phi_u1_finite_current_kernel as prices
import joined_phi_u1_electric_hop_transport as history
import joined_phi_u1_initial_car_word_algebra as car

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_initial_photon_cubic_jet'
PROOF={
 'Jacobi':'delta_j A=i[A,Theta_j], delta_j adH A=adH delta_j A+[128Pi_j,A]. J=-partial M, C=partial²M, [Pi_j,J_i]=iCij, [Pi_j,M]=iJj. Hence M3=128(3adH²C+2adH[Jj,Ji]+[Jj,adH Ji]). No state stationarity used.',
 'electric':'adE O_w=O_w F_w, F_w=128s_w+64|w|². [s_w,O_r]=w.r, including FULL Gauss density plus physical Gram. Thus adH²O=O F²+[M,O]F+adE[M,O]-128 sum_r(w.r)O O_r+adM²O. Onsite r has zero wind, but is retained in both connected CAR chains.',
 'contractions':'Each connected same-role commutator is dGamma[A,B], including auxiliary aggregate without extra63. Ordered collective twohop expectation has falling-factorial63*62 disconnected blocks plus63 ordered samecopy moment. O rho² has at most3 initial CAR factors. Gaussian right-normal mean=-q_w.q_total/2, covariance=q_w.q_v/8 and character exp(-|q_total|²). Ref hole subtracts v(0) per actual four raw channels; background constant -128sum(v).',
 'matter':'Pure matter part of Jacobi expression reduces to same-species triple onebody chains. No products of different-species dGamma remain. The full one/two/AP stencil and onsite blocks are included.',
 'ABI':'M3 is the third derivative of i<[J(t),Theta]> at0, not coefficient divided by factorial. D fifth coefficient128M3/120, Pi response fourth M3/24; contact C0 uses existing physical129 ABI. All statements concern initial derivatives only.',
 'error':'Conditional binary64 coefficients only. Guard radii are diagnostic conditioning estimates, NOT outward errorballs; interval admission prohibited. Root/seas/Clifford numerical preparation uncertainty not priced here. Higher future remainder and physical transfer belong to parent.'}


def tr(a:np.ndarray,b:np.ndarray)->complex:
    return complex(np.einsum('ij,ji->',a,b))


class Engine:
    """Block-sparse connected Weyl/CAR contraction, arbitrary small boundary."""
    def __init__(self,k:np.ndarray,delta:np.ndarray,gram:np.ndarray,lp:np.ndarray,x:np.ndarray,y:np.ndarray,blocks:list[np.ndarray],onsite:list[np.ndarray],copies:int=63):
        self.k=k.astype(float);self.delta=delta;self.gram=gram;self.lp=lp
        self.x=x.astype(int);self.y=y.astype(int);self.b=blocks;self.onsite=onsite;self.copies=copies
        self.n=lp.shape[0];self.spin=blocks[0].shape[-1];self.h=len(x);self.edge=k.shape[1]
        self.v=delta@lp;self.overlap=k@k.T;self.q2=np.diag(gram)
        self.nz=[np.flatnonzero(r) for r in k]
        self.outgoing=[np.flatnonzero(x==s) for s in range(self.n)]
        self.links=[np.flatnonzero((y[a]==x)|(y==x[a])) for a in range(self.h)]
        self.counts={}

    def dense(self,a:int,role:int)->np.ndarray:
        out=np.zeros((self.n*self.spin,)*2,complex)
        out[self.sl(self.x[a]),self.sl(self.y[a])]=self.b[role][a]
        return out

    def sl(self,s:int)->slice:
        return slice(int(s)*self.spin,(int(s)+1)*self.spin)

    def density(self,p:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        d=np.real(np.diag(p))
        cov=np.diag(d)-abs(p)**2
        return d,cov

    def linear(self,a:np.ndarray,x:int,y:int,p:np.ndarray,v:np.ndarray)->tuple[complex,complex]:
        xs,ys=self.sl(x),self.sl(y);vv=np.repeat(v,self.spin)
        mean=tr(a,p[ys,xs])
        cov=tr(a,v[y]*p[ys,xs]-p[ys,:]@(vv[:,None]*p[:,xs]))
        return mean,cov

    def square(self,a:np.ndarray,x:int,y:int,p:np.ndarray,v:np.ndarray)->tuple[complex,complex,complex]:
        mean,cov=self.linear(a,x,y,p,v)
        xs,ys=self.sl(x),self.sl(y);vv=np.repeat(v,self.spin)
        qrow=-p[ys,:].copy();qrow[:,ys]+=np.eye(self.spin)
        vp=vv[:,None]*p[:,xs]
        third=tr(a,(qrow*vv[None,:])@(vp-2*p@vp))
        return mean,cov,third

    def add(self,out:np.ndarray,a:int|np.integer,b:int|np.integer,value:complex)->None:
        for i in self.nz[a]:
            for j in self.nz[b]:out[i,j]+=value*self.k[a,i]*self.k[b,j]

    def state(self,active:np.ndarray,aux:np.ndarray,hole:bool)->dict[str,np.ndarray]:
        ps=[active,aux];loads=[1,self.copies];ds=[self.density(p) for p in ps]
        rho=sum((l*d[0] for l,d in zip(loads,ds)),np.zeros(active.shape[0]))-2*(1+self.copies)/self.spin
        if hole:rho[:self.spin]-=1/self.spin
        vr=np.repeat(self.v,self.spin,axis=1)
        rmean=vr@rho
        var=sum((l*np.einsum('ri,ij,rj->r',vr,d[1],vr,optimize=True) for l,d in zip(loads,ds)),np.zeros(self.h))
        mean=[];cov=[]
        for p,b in zip(ps,self.b):
            pb=p.reshape(self.n,self.spin,self.n,self.spin).transpose(0,2,1,3)
            mm=np.einsum('rij,rji->r',b,pb[self.y,self.x]);cc=np.empty((self.h,self.h),complex)
            for a in range(self.h):
                subtract=np.einsum('ij,sjk,skl,sli->s',b[a],pb[self.y[a],self.x],b,pb[self.y,self.x[a]],optimize=True)
                direct=np.einsum('ij,sjk,ski->s',b[a],b,pb[self.y,self.x[a]],optimize=True)
                cc[a]=(self.y[a]==self.x)*direct-subtract
            mean.append(mm);cov.append(cc)
        mm=mean[0]+self.copies*mean[1]
        second=np.outer(mm,mm)+cov[0]+self.copies*cov[1]
        chars=np.exp(-(self.q2[:,None]+self.q2[None,:]+2*self.gram))
        electric_square=np.zeros((self.edge,self.edge),complex)
        linked=np.zeros_like(electric_square);collective=np.zeros_like(electric_square)
        for role,(p,load) in enumerate(zip(ps,loads)):
            for a in range(self.h):
                ma,ca,ta=self.square(self.b[role][a],self.x[a],self.y[a],p,self.v[a])
                fm=128*(rmean[a]-self.q2[a]/2)+64*self.overlap[a,a]
                value=np.exp(-self.q2[a])*load*(ma*(fm*fm+128**2*(var[a]+self.q2[a]/8))+256*ca*fm+128**2*ta)
                self.add(electric_square,a,a,-3*value)
                for b in self.links[a]:
                    # [O_b,O_a], total phase a+b. Same role only.
                    parts=[]
                    if self.y[b]==self.x[a]:parts.append((self.b[role][b]@self.b[role][a],self.x[b],self.y[a],1))
                    if self.y[a]==self.x[b]:parts.append((self.b[role][a]@self.b[role][b],self.x[a],self.y[b],-1))
                    damp=chars[a,b];qt2=self.q2[a]+self.q2[b]+2*self.gram[a,b]
                    ft2=self.overlap[a,a]+self.overlap[b,b]+2*self.overlap[a,b]
                    for block,x,y,sign in parts:
                        mu,cv=self.linear(block,x,y,p,self.v[a]);_,cvb=self.linear(block,x,y,p,self.v[b])
                        fa=128*(rmean[a]-(self.q2[a]+self.gram[a,b])/2)+64*self.overlap[a,a]
                        ft=128*(rmean[a]+rmean[b]-qt2/2)+64*ft2
                        va=load*sign*damp*(mu*fa+128*cv)
                        vt=load*sign*damp*(mu*ft+128*(cv+cvb))
                        self.add(linked,a,a,-3*(va+vt))
                        # 2 E[Jj,Ji]: comm[Ob,Oa] coefficients -2 ka_i kb_j+2 kb_i ka_j.
                        self.add(linked,a,b,-2*vt);self.add(linked,b,a,2*vt)
                        # [Jj,E Ji], comm[Ob,Oa] Fa.
                        self.add(linked,a,b,-va)
                # Onsite comm[Mon,Oa] has wind a; both electric placements.
                block=self.onsite[role][self.x[a]]@self.b[role][a]-self.b[role][a]@self.onsite[role][self.y[a]]
                mu,cv=self.linear(block,self.x[a],self.y[a],p,self.v[a])
                fa=128*(rmean[a]-self.q2[a]/2)+64*self.overlap[a,a]
                self.add(linked,a,a,-6*load*np.exp(-self.q2[a])*(mu*fa+128*cv))
        # Collective blocks: samecopy covariance carries63, disconnected auxiliary blocks63*62.
        weighted=chars*self.overlap*second
        for a in range(self.h):
            self.add(collective,a,a,384*sum(weighted[a]))
            for b in np.flatnonzero(self.overlap[a]):self.add(collective,a,b,128*weighted[a,b])
        return {'electric_square':electric_square,'electric_matter_connected':linked,'electric_matter_collective':collective}

    def matter(self,effective:list[np.ndarray])->np.ndarray:
        out=np.zeros((self.edge,self.edge),complex);chains=0
        k=np.r_[self.k,np.zeros((self.n,self.edge))]
        gram=np.zeros((self.h+self.n,)*2);gram[:self.h,:self.h]=self.gram
        x=np.r_[self.x,np.arange(self.n)];y=np.r_[self.y,np.arange(self.n)]
        nz=self.nz+[np.array([],int) for _ in range(self.n)]
        outgoing=[np.flatnonzero(x==s) for s in range(self.n)]
        def add(a:int|np.integer,b:int|np.integer,z:complex):
            for i in nz[a]:
                for j in nz[b]:out[i,j]+=z*k[a,i]*k[b,j]
        for role,p in enumerate(effective):
            blocks=np.r_[self.b[role],self.onsite[role]]
            pb=p.reshape(self.n,self.spin,self.n,self.spin).transpose(0,2,1,3)
            for a in range(len(x)):
                for b in outgoing[y[a]]:
                    ab=blocks[a]@blocks[b]
                    for c in outgoing[y[b]]:
                        q2=gram[a,a]+gram[b,b]+gram[c,c]+2*(gram[a,b]+gram[a,c]+gram[b,c])
                        z=tr(ab@blocks[c],pb[y[c],x[a]])*math.exp(-q2)
                        # 3 adM²C + 2 adM[Jj,Ji] + [Jj,adM Ji].
                        for r,s,f in ((c,c,-3),(b,b,6),(a,a,-3),(c,b,-2),(b,c,2),(b,a,3),(a,b,-2),(c,a,-1),(b,c,1),(a,c,-1)):
                            add(r,s,f*z)
                        chains+=1
        self.counts['matter_ordered_chains']=chains
        return out

    def contact(self,effective:list[np.ndarray])->np.ndarray:
        out=np.zeros((self.edge,self.edge),complex)
        for b,p in zip(self.b,effective):
            for a in range(self.h):self.add(out,a,a,-math.exp(-self.q2[a])*tr(b[a],p[self.sl(self.y[a]),self.sl(self.x[a])]))
        return out

    def evaluate(self,states:list[tuple[float,np.ndarray,bool]],aux:np.ndarray,mass:float)->dict[str,np.ndarray]:
        terms={key:np.zeros((self.edge,self.edge),complex) for key in ('electric_square','electric_matter_connected','electric_matter_collective')}
        for weight,p,hole in states:
            for key,value in self.state(p,aux,hole).items():terms[key]+=weight*value
        effective=[sum((w*p for w,p,_ in states),np.zeros_like(aux)),mass*self.copies*aux]
        terms['matter']=self.matter(effective)
        terms['M3']=128*sum(terms.values(),np.zeros((self.edge,self.edge),complex))
        terms['C0']=self.contact(effective)
        return terms


def finite_boundary_controls()->dict[str,Any]:
    """Independent finiteCAR/Hermite canonical jet with a converged interior."""
    k=np.array([[1,0],[0,1],[-1,0],[0,-1]],float)
    delta=np.array([[1,-1],[1,-1],[-1,1],[-1,1]],float)
    lp=np.array([[1,-1],[-1,1]],float)/8
    q=np.array([1,-1,-1,1],float)/math.sqrt(2)
    gram=np.outer(q,q)
    x=np.array([0,0,1,1]);y=1-x
    bs=[np.array([.23+.11j,-.17+.09j,.23-.11j,-.17-.09j])[:,None,None],np.array([.13+.07j,.19-.05j,.13-.07j,.19+.05j])[:,None,None]]
    on=[np.array([.31,-.22])[:,None,None],np.array([-.12,.18])[:,None,None]]
    p=np.outer(np.array([math.sqrt(.7),1j*math.sqrt(.3)]),np.array([math.sqrt(.7),1j*math.sqrt(.3)]).conj())
    aux=np.diag([1.,0.]).astype(complex)
    engine=Engine(k,delta,gram,lp,x,y,bs,on,copies=1)
    # Physical tiny boundary baseline is one particle per site across both roles.
    # Engine's physical64site baseline is128; for spin1,copies1 it equals4/site.
    # Both constants sum tozero against exact v, so no change to rhoLrho.
    analytic=engine.evaluate([(1.,p,False)],aux,1.)['M3']
    analytic=float(((np.array([1,-1])/math.sqrt(2))@analytic@(np.array([1,-1])/math.sqrt(2))).real)
    local=[engine.dense(a,0) for a in range(4)];other=[engine.dense(a,1) for a in range(4)]
    dg=[np.kron(car.finite_car(a),np.eye(4)) for a in local]
    dgaux=[np.kron(np.eye(4),car.finite_car(a)) for a in other]
    n0=car.finite_car(np.diag([1.,0.]));n1=car.finite_car(np.diag([0.,1.]))
    rho=np.kron(n0-n1,np.eye(4))+np.kron(np.eye(4),n0-n1)
    hc=64*rho@rho/8
    onsite=np.kron(car.finite_car(np.diag(on[0][:,0,0])),np.eye(4))+np.kron(np.eye(4),car.finite_car(np.diag(on[1][:,0,0])))
    orbital=np.array([math.sqrt(.7),1j*math.sqrt(.3)]);state=np.kron(np.array([0,orbital[0],orbital[1],0]),np.array([0,1,0,0]))
    results=[];wrong=[]
    for levels in (22,34):
        lowering=np.diag(np.sqrt(np.arange(1,levels)),1);theta=math.sqrt(2)*(lowering+lowering.T)
        pi=1j*(lowering.T-lowering)/(2*math.sqrt(2));ident=np.eye(levels)
        h=np.kron(64*pi@pi,np.eye(16))+np.kron(ident,hc+onsite)
        j=np.zeros_like(h);c=np.zeros_like(h)
        for a in range(4):
            term=np.kron(expm(1j*q[a]*theta),dg[a]+dgaux[a]);h+=term;j+=-1j*q[a]*term;c+=-q[a]**2*term
        th=np.kron(theta,np.eye(16));psi=np.kron(np.eye(levels)[:,0],state)
        ad=lambda a:h@a-a@h
        third=-1j*ad(ad(ad(j)))
        direct=1j*np.vdot(psi,(third@th-th@third)@psi)
        # Independently the Jacobi identity before contractions must match.
        jac=128*(3*ad(ad(c))+ad(j@j-j@j)*2+j@ad(j)-ad(j)@j)
        jacvalue=np.vdot(psi,jac@psi)
        wrongh=h-np.kron(32*pi@pi,np.eye(16))
        aw=lambda a:wrongh@a-a@wrongh
        wrongthird=-1j*aw(aw(aw(j)));wrongvalue=1j*np.vdot(psi,(wrongthird@th-th@wrongthird)@psi)
        results.append({'levels':levels,'canonical_M3':float(direct.real),'jacobi_M3':float(jacvalue.real),'imaginary_defect':float(abs(direct.imag)),'canonical_jacobi_defect':float(abs(direct-jacvalue)),'analytic_defect':float(abs(direct-analytic))})
        wrong.append(float(abs(wrongvalue-direct)))
    if abs(results[-1]['canonical_M3']-analytic)>2e-6*max(1,abs(analytic)):raise ArithmeticError({'initial_boundary_failure':results,'analytic':analytic})
    if results[-1]['canonical_jacobi_defect']>2e-6*max(1,abs(analytic)):raise ArithmeticError('Canonical sign/order failure')
    if wrong[-1]<1:raise ArithmeticError('Wrong HE half not discriminated')
    # Ref4->3 control adds an active particle and removes the matching Ref.
    rawmatrix=engine.evaluate([(1.,np.eye(2,dtype=complex),True)],aux,1.)['M3']
    direction=np.array([1,-1])/math.sqrt(2)
    rawanalytic=float((direction@rawmatrix@direction).real)
    rawpsi=np.kron(np.eye(levels)[:,0],np.kron(np.array([0,0,0,1]),np.array([0,1,0,0])))
    refhc=64*(rho-np.eye(16))@(rho-np.eye(16))/8
    rawh=h+np.kron(ident,refhc-hc)
    ar=lambda a:rawh@a-a@rawh
    rawthird=-1j*ar(ar(ar(j)))
    rawdirect=1j*np.vdot(rawpsi,(rawthird@th-th@rawthird)@rawpsi)
    missingref=1j*np.vdot(rawpsi,(third@th-th@third)@rawpsi)
    if abs(rawdirect-rawanalytic)>2e-6*max(1,abs(rawanalytic)):raise ArithmeticError('HC Ref hole boundary failure')
    if abs(missingref-rawdirect)<1:raise ArithmeticError('Missing Ref hole not discriminated')
    nohc=h-np.kron(ident,hc)
    ah=lambda a:nohc@a-a@nohc
    nohcjet=-1j*ah(ah(ah(j)))
    nohcvalue=1j*np.vdot(psi,(nohcjet@th-th@nohcjet)@psi)
    if abs(nohcvalue-direct)<1:raise ArithmeticError('Missing full HC not discriminated')
    # Wrong independent auxiliary field cannot contribute active Theta comm.
    independent_h=np.kron(64*pi@pi,np.eye(16))+np.kron(ident,hc+onsite)
    independent_j=np.zeros_like(h)
    for a in range(4):
        term=np.kron(expm(1j*q[a]*theta),dg[a])
        independent_h+=term+np.kron(ident,dgaux[a]);independent_j+=-1j*q[a]*term
    ai=lambda a:independent_h@a-a@independent_h
    independent_third=-1j*ai(ai(ai(independent_j)))
    independent_value=1j*np.vdot(psi,(independent_third@th-th@independent_third)@psi)
    if abs(independent_value-direct)<1:raise ArithmeticError('Wrong independent auxiliary Theta not discriminated')
    # Ordered role/collective controls use parent helper on actual small matrices.
    helper=car.role_word_moment(p,aux,[('auxiliary',other[0]),('auxiliary',other[2])],copies=2)
    literal=2*car.ordered_car_moment(aux,[other[0],other[2]])+2*car.ordered_car_moment(aux,[other[0]])*car.ordered_car_moment(aux,[other[2]])
    if abs(helper-literal)>1e-12:raise ArithmeticError('Auxiliary falling-factorial failure')
    twofield=two_field_order_control()
    return {'analytic_M3':analytic,'finite_CAR_Hermite':results,'wrong_HE32_defects':wrong,'ordered_auxiliary_helper_defect':float(abs(helper-literal)),'HC_Ref_raw_analytic':rawanalytic,'HC_Ref_raw_direct':float(rawdirect.real),'missing_Ref_defect':float(abs(missingref-rawdirect)),'missing_HC_defect':float(abs(nohcvalue-direct)),'wrong_independent_auxiliary_Theta_defect':float(abs(independent_value-direct)),'two_field_order':twofield,'scope':'Finite boundary initialjet controls only; Hermite convergence not fullBODY evolution.'}

def two_field_order_control()->dict[str,Any]:
    """Two independent Theta directions reject hidden M3-transpose assumptions."""
    from scipy.sparse import csr_matrix, kron, eye
    levels=20
    k=np.r_[np.eye(3),-np.eye(3)]
    x=np.array([0,0,0,1,1,1]);y=1-x
    delta=np.r_[np.tile([[1.,-1.]],(3,1)),np.tile([[-1.,1.]],(3,1))]
    lp=np.array([[1.,-1.],[-1.,1.]])/12
    basis=np.array([[1/math.sqrt(2),1/math.sqrt(6)],[-1/math.sqrt(2),1/math.sqrt(6)],[0,-2/math.sqrt(6)]])
    q=k@basis
    bs=np.array([.23+.11j,-.17+.09j,.08-.06j,.23-.11j,-.17-.09j,.08+.06j])[:,None,None]
    on=np.array([.31,-.22])[:,None,None]
    orbital=np.array([math.sqrt(.7),1j*math.sqrt(.3)])
    p=np.outer(orbital,orbital.conj());zero=np.zeros((2,2),complex)
    e=Engine(k,delta,q@q.T,lp,x,y,[bs,bs],[on,on],copies=0)
    analytic=basis.T@e.evaluate([(1.,p,False)],zero,1.)['M3']@basis
    lowering=np.diag(np.sqrt(np.arange(1,levels)),1)
    theta=math.sqrt(2)*(lowering+lowering.T)
    pi=1j*(lowering.T-lowering)/(2*math.sqrt(2))
    ftheta=[kron(csr_matrix(theta),eye(levels)),kron(eye(levels),csr_matrix(theta))]
    kinetic=kron(csr_matrix(pi@pi),eye(levels))+kron(eye(levels),csr_matrix(pi@pi))
    rhod=car.finite_car(np.diag([1.,-1.]))
    matter=64*rhod@rhod/12+car.finite_car(np.diag(on[:,0,0]))
    h=kron(64*kinetic,eye(4))+kron(eye(levels**2),csr_matrix(matter))
    js=[csr_matrix(h.shape,dtype=complex) for _ in range(2)]
    for a in range(6):
        phase=kron(csr_matrix(expm(1j*q[a,0]*theta)),csr_matrix(expm(1j*q[a,1]*theta)))
        term=kron(phase,csr_matrix(car.finite_car(e.dense(a,0))))
        h+=term
        for i in range(2):js[i]+=-1j*q[a,i]*term
    thetas=[kron(t,eye(4)) for t in ftheta]
    psi=np.kron(np.eye(levels**2)[:,0],np.array([0,orbital[0],orbital[1],0]))
    def powers(v):
        result=[v]
        for _ in range(3):result.append(h@result[-1])
        return result
    pv=powers(psi);tv=[powers(t@psi) for t in thetas]
    direct=np.zeros((2,2),complex)
    for i in range(2):
        for j in range(2):
            for a in range(4):
                # M3=[adH³J,Theta], since i*i³=1.
                direct[i,j]+=(-1)**a*math.comb(3,a)*(np.vdot(pv[3-a],js[i]@tv[j][a])-np.vdot(tv[j][3-a],js[i]@pv[a]))
    defect=float(np.max(abs(direct-analytic)))
    transpose=float(np.max(abs(direct-direct.T)))
    if defect>2e-7*max(1,float(np.max(abs(direct)))):raise ArithmeticError({'two_field_defect':defect,'analytic':analytic.tolist(),'direct':direct.tolist()})
    if transpose<1:raise ArithmeticError('Nonstationary antisymmetric channel control blind')
    if float(np.max(abs(-direct-analytic)))<1:raise ArithmeticError('Wrong canonical sign control blind')
    return {'levels_per_field':levels,'direct_real':np.real(direct).tolist(),'analytic_real':np.real(analytic).tolist(),'max_defect':defect,'nonstationary_transpose_defect':transpose,'wrong_canonical_sign_defect':float(np.max(abs(-direct-analytic)))}


def run(output:Path)->dict[str,Any]:
    started=time.monotonic();controls=finite_boundary_controls()
    data,sp,inputs,_=body.full_inputs();geo=current.geometry(data,sp[0]);geom=prices.exact_geometry(data['native'],geo)
    engine=Engine(geo['k'],geo['delta'],geom['gram'],geom['L'],geo['x'],geo['y'],[current.blocks(s) for s in sp],[np.broadcast_to(s.onsite,(64,4,4)) for s in sp])
    channels=history.initial_channels(sp[0])
    controls['actual_native_Gauss_AP_boundary']=history.controls(data['native'],geo,geom)
    # Full256 actual raw-channel ordered moment, not an averaged Slater.
    a=int(np.flatnonzero(geo['x']==0)[0])
    amat=engine.dense(a,0)
    vmat=np.diag(np.repeat(engine.v[a],4))
    average=sum((c['weight']*c['p'] for c in channels),np.zeros_like(sp[0].p))/.5
    actual=0j;special=0j
    for channel in channels:
        p=channel['p'];weight=channel['weight']
        actual+=weight*car.ordered_car_moment(p,[amat,vmat,vmat])
        ma,ca,ta=engine.square(engine.b[0][a],engine.x[a],engine.y[a],p,engine.v[a])
        vv=np.diag(vmat);dm,dc=engine.density(p)
        mv=vv@dm;variance=vv@dc@vv
        special+=weight*(ma*(mv*mv+variance)+2*ca*mv+ta)
    wrong=.5*car.ordered_car_moment(average,[amat,vmat,vmat])
    if abs(actual-special)>1e-10:raise ArithmeticError('Actual raw4 ordered rho² block contraction fails')
    if abs(actual-wrong)<1e-12:raise ArithmeticError('Wrong averaged raw Slater control blind')
    controls['actual_raw4_ordered_three_CAR']={'path':a,'actual_weighted':body.packed(actual),'block_specialized':body.packed(special),'helper_specialized_defect':float(abs(actual-special)),'wrong_averaged_Slater_defect':float(abs(actual-wrong)),'claim':'Initial threefactor subcomponent of electric_square, not future Wick.'}
    states={'background':([(1.,sp[0].p,False)],1.),'raw_source':([(c['weight'],c['p'],True) for c in channels],.5)}
    arrays={};report={}
    for label,(rows,mass) in states.items():
        terms=engine.evaluate(rows,sp[1].p,mass)
        for key,value in terms.items():arrays[label+'_'+key]=data['native']['Q'].T@value@data['native']['Q']
        m3=arrays[label+'_M3'];arrays[label+'_D5']=128*m3/120;arrays[label+'_Pi4']=m3/24
        imag=float(np.max(abs(m3.imag)));guard=1e-9*sum(float(np.linalg.norm(arrays[label+'_'+key],ord='fro')) for key in ('electric_square','electric_matter_connected','electric_matter_collective','matter'))*128
        arrays[label+'_M3_guard_NOT_outward']=np.full((129,129),guard)
        arrays[label+'_D5_guard_NOT_outward']=np.full((129,129),128*guard/120)
        arrays[label+'_Pi4_guard_NOT_outward']=np.full((129,129),guard/24)
        if not np.all(np.isfinite(m3)) or imag>max(1e-4,guard):raise ArithmeticError('Nonfinite or nonreal mixedjet')
        report[label]={'mass':mass,'shape':list(m3.shape),'M3_frobenius':float(np.linalg.norm(m3)),'M3_max_abs':float(np.max(abs(m3))),'imaginary_defect':imag,'diagnostic_guard_radius_NOT_outward':guard,'arithmetic_status':'CONDITIONAL_BINARY64_NO_ADMISSION','components_frobenius':{key:float(np.linalg.norm(arrays[label+'_'+key])) for key in ('electric_square','electric_matter_connected','electric_matter_collective','matter')},'contact_eigenvalue_range_conditional':[float(np.linalg.eigvalsh(arrays[label+'_C0'].real)[0]),float(np.linalg.eigvalsh(arrays[label+'_C0'].real)[-1])]}
    arrays['physical_Q']=data['native']['Q'];arrays['native_integer_winds']=geo['k'].astype(np.int64);arrays['exact_L_integer']=geom['L_integer'];arrays['exact_gram_numerator']=np.rint(geom['gram']*geom['denominator']).astype(np.int64)
    npz=output.with_suffix('.npz');np.savez_compressed(npz,**arrays)
    captured={**body.CAPTURED,**inputs}
    for module in (current,prices,history,car):
        path=Path(cast(str,module.__file__))
        captured[path.name]=path.read_bytes()
    source=Path(__file__).read_bytes();captured[Path(__file__).name]=source
    result={'status':'ALL129_INITIAL_M3_CONDITIONAL_COMPLETE_NO_FINITE_KERNEL_ADMISSION','proof':PROOF,'controls':controls,'states':report,'raw_channels':[{'weight':float(c['weight']),'Ref_hole':0} for c in channels],'all129':True,'full63_Gauss':True,'charged_copies':64,'exact_Gauss_denominator':geom['denominator'],'counts':engine.counts,'source_sha256':hashlib.sha256(source).hexdigest(),'input_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()},'npz_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),'runtime':{'executable':sys.executable,'python':sys.version},'elapsed_seconds':time.monotonic()-started,'measured_time_evolution':False,'outward':False,'admission':False,'probe_clock':'1/65536 (not adopted)','higher_remainder':'parent-owned; not priced by initial coefficients'}
    if any((ROOT/name).read_bytes()!=raw for name,raw in captured.items()):raise ArithmeticError('Input bytes changed during run')
    import zipfile
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():archive.writestr(name,raw)
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'_receipt.json'));args=parser.parse_args()
    with threadpool_limits(limits=1):receipt=run(args.output)
    print(json.dumps({'status':receipt['status'],'states':receipt['states'],'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
