"""NEW N4 local-SU4 reduced nonuniform Phi/colour quantum supplier.

No promotion to the original law or quantum ground.  Scientific wave is the
matrix functional at the implicit all-q feedback root, not printed matrices.
All Fourier fields use exp(i q.x)/8; reality pairs retain 5760 real degrees.
The adapted horizontal frame is polar(P(Phi)Q0), P=I-O(O*O)^-1O*.
Its derivative and the kinetic part of fast Hermite forcing are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
import time
from collections import Counter
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES:dict[str,bytes]={Path(__file__).name:Path(__file__).read_bytes()}

class CapturedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname:str)->Any:
        raw=Path(self.path).read_bytes(); SOURCE_BYTES[Path(self.path).name]=raw
        return self.source_to_code(raw,self.path)

class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname:str,path:Any=None,target:Any=None)->Any:
        candidate=ROOT/(fullname+'.py')
        if candidate.is_file():
            return importlib.util.spec_from_file_location(fullname,candidate,loader=CapturedLoader(fullname,str(candidate)))
        return None

finder=CapturedFinder(); sys.meta_path.insert(0,finder)
import induced_phi_uniform_selfconsistent_full_normal_quantum as stationary
sys.meta_path.remove(finder)
SOURCE_BYTES.update(stationary.CAPTURED)
import mpmath as mp
import numpy as np

iv:Any=mp.iv
exact,endpoint,enclosure,bounds=stationary.exact,stationary.endpoint,stationary.enclosure,stationary.bounds
F63='f63c1590af4fa9922757615b1bfccb788f430760d401d7c66372e8656163251c'
QINTEGER=list(product(range(4),repeat=3))
QCOUNTS=Counter(sum(0 if n==0 else 4 if n==2 else 2 for n in q) for q in QINTEGER)

def midpoint(x:Any)->float:
    return float((endpoint(x,0)+endpoint(x,1))/2)

def encode(x:np.ndarray)->dict[str,Any]:
    return dict(real=np.real(x).tolist(),imag=np.imag(x).tolist())

def masks(M:np.ndarray)->np.ndarray:
    gamma=stationary.full.supplier.primitive.SELECTED
    return np.asarray([[int(np.linalg.norm(m@g+g@m)<1e-12) for g in gamma] for m in M])

def fast_zpe_derivatives(g0:Any,gs:Any,mask:np.ndarray,ctx:Any)->tuple[Any,list[Any],list[list[Any]]]:
    z=sum(ctx.mpf(45)*count*ctx.sqrt(ctx.mpf(q2)/5)/2 for q2,count in QCOUNTS.items() if q2); gradient=[ctx.mpf(0),ctx.mpf(0)]; hessian=[[ctx.mpf(0) for _ in range(2)] for _ in range(2)]
    for row in mask:
        dh=[ctx.mpf(int(row[0]))/2,ctx.mpf(int(sum(row[1:])))/2]; H=dh[0]*g0+dh[1]*gs
        for q2,count in QCOUNTS.items():
            for c,multiplicity in ((F(1,5),1),(F(21,200),2)):
                w=ctx.sqrt(ctx.mpf(3)/50*H+stationary.number(c,ctx)*q2)
                z+=count*multiplicity*w/2
                for i in range(2):
                    gradient[i]+=count*multiplicity*ctx.mpf(3)/200*dh[i]/w
                    for j in range(2):hessian[i][j]-=count*multiplicity*ctx.mpf(9)/20000*dh[i]*dh[j]/w**3
    return z,gradient,hessian

def solve_mean(root_path:Path,full_path:Path)->dict[str,Any]:
    raw=root_path.read_bytes(); fraw=full_path.read_bytes(); root=json.loads(raw); old=json.loads(fraw)
    if hashlib.sha256(SOURCE_BYTES[Path(stationary.__file__).name]).hexdigest()!=F63:raise ArithmeticError('Frozen f63 source changed')
    parameters=root['actual_stationary_root']['parameters']
    initial=[F(parameters[k]['lower']) for k in ('g0','gs')]
    mean=stationary.input_mean(dict(microscopic_config=old['input_descriptor']['original_micro_config_UNCHANGED']),initial)
    profiles=stationary.full.scalar.fast.primitive_profiles(mean); rows=stationary.radial_data(profiles)
    iv.dps=80; native=stationary.full.supplier.quantum.build_native(mean,4)
    geom=stationary.full.normal_geometry(native,[exact(initial[0])]+[exact(initial[1])]*3); mask=masks(geom['M'])
    def law(x:Any,y:Any,ctx:Any)->tuple[list[Any],list[list[Any]],Any]:
        force,J,E=stationary.radial_force_jacobian(x,y,rows,ctx)
        z,dz,ddz=fast_zpe_derivatives(x,y,mask,ctx); eps=ctx.power(2,-5120)/64
        for i in range(2):
            divisor=1 if i==0 else 3
            force[i]+=eps*dz[i]/divisor
            for j in range(2):J[i][j]+=eps*ddz[i][j]/divisor
        return force,J,E+eps*z
    mp.mp.dps=2600
    start=[stationary.number(x,mp.mp) for x in initial]
    r=mp.findroot(lambda x,y:tuple(law(x,y,mp.mp)[0]),start,J=lambda x,y:mp.matrix(law(x,y,mp.mp)[1]),tol=mp.mpf(10)**-2500,maxsteps=40,solver='mdnewton')
    centres=[F(str(mp.nstr(x,2250))) for x in r]; radius=F(1,10**2150); iv.dps=2300
    point=[exact(x) for x in centres]; box=[enclosure(x-radius,x+radius) for x in centres]
    f0,J0,_=law(point[0],point[1],iv); _,JB,E=law(box[0],box[1],iv)
    rational=[[stationary.midpoint(v) for v in row] for row in J0]; det=rational[0][0]*rational[1][1]-rational[0][1]*rational[1][0]
    C=[[rational[1][1]/det,-rational[0][1]/det],[-rational[1][0]/det,rational[0][0]/det]]
    contractions=[]; images=[]
    for i in range(2):
        lip=sum(endpoint(abs(exact(int(i==j))-sum(exact(C[i][k])*JB[k][j] for k in range(2))),1) for j in range(2))
        contractions.append(lip); images.append(endpoint(abs(sum(exact(C[i][j])*f0[j] for j in range(2))),1)+lip*radius)
    if max(contractions)>=1 or max(images)>=radius:raise ArithmeticError('ALLq fast feedback root inclusion failed')
    z,dz,_=fast_zpe_derivatives(point[0],point[1],mask,iv)
    oldg=[exact(x) for x in initial]; _,old_dz,_=fast_zpe_derivatives(oldg[0],oldg[1],mask,iv)
    # Uniform Cartesian diagonal force: dg/deta=2 sqrt(g)/(8 sqrt(8)).
    force_sigma=iv.sqrt(sum((old_dz[i]*iv.sqrt(oldg[i])/iv.sqrt(exact(8))/4)**2/(1 if i==0 else 3) for i in range(2)))*iv.sqrt(iv.sqrt(exact(8))/2)
    output:dict[str,Any]=dict(parameters={k:bounds(x) for k,x in zip(('g0','gs'),box)},centre_exact_rational=[str(x) for x in centres],root_radius=bounds(exact(radius)),contraction=bounds(exact(max(contractions))),image_radii=[bounds(exact(x)) for x in images],energy_divided_64Lambda=bounds(E),fast_ZPE_divided_Lambda3over8=bounds(z),old_f63_fast_force_times_reference_slow_Phi_std=bounds(force_sigma),mean_shift=[bounds(point[i]-oldg[i]) for i in range(2)],equation='64Lambda Vclass(g)+Lambda^(3/8) Zall64q(g); radial spatial force divided3; all other normal forces vanish by Carttrace/cubic symmetry, not subtraction',not_quantum_ground=True)
    iv.dps=80; mp.mp.dps=80
    mean=stationary.input_mean(dict(microscopic_config=old['input_descriptor']['original_micro_config_UNCHANGED']),centres)
    native=stationary.full.supplier.quantum.build_native(mean,4); g=[enclosure(F(output['parameters']['g0']['lower']),F(output['parameters']['g0']['upper']))]+[enclosure(F(output['parameters']['gs']['lower']),F(output['parameters']['gs']['upper']))]*3
    geom=stationary.full.normal_geometry(native,g)
    return dict(root=output,centres=centres,g=g,mean=mean,native=native,geom=geom,profiles=profiles,frozen_slow_lower=F(old['actual_full45_coupled_quantum_consumer']['certified_eigenvalue_lower_divided_Lambda']['lower']),input_bytes={str(root_path):raw,str(full_path):fraw})

def orbit(frame:np.ndarray,T:np.ndarray)->np.ndarray:
    comm=1j*(T[:,None]@frame[None]-frame[None]@T[:,None])
    return (2*np.einsum('bij,amji->mba',T,comm).real).reshape(60,15)

def adapted_frame(frame:np.ndarray,T:np.ndarray,Q0:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    O=orbit(frame,T); H=O.T@O/8
    P=np.eye(60)-O@np.linalg.solve(O.T@O,O.T)
    N=P@Q0; w,U=np.linalg.eigh(N.T@N)
    if w[0]<=0:raise ArithmeticError('Adapted horizontal frame lost rank')
    Q=N@(U@np.diag(w**-.5)@U.T)
    return Q,O,H

def primitive_controls(data:dict[str,Any])->dict[str,Any]:
    native,geom=data['native'],data['geom']; T=geom['M']/np.sqrt(8); Q0=geom['Q']; frame=native['frame']
    Q,O,H=adapted_frame(frame,T,Q0); expected=.5*masks(geom['M'])@np.asarray([midpoint(x) for x in data['g']])
    casimir=np.max(abs(sum(t@t for t in T)-15/8*np.eye(4))); norm=np.max(abs(2*np.einsum('aij,bji->ab',T,T)-np.eye(15)))
    orbit_defect=np.max(abs(H-np.diag(expected))); qrows=[]; maxschur=0.; maxfreq=0.
    for q in QINTEGER:
        d=1-np.exp(2j*np.pi*np.asarray(q)/4); q2=float(np.vdot(d,d).real); dd=np.outer(d,d.conj())
        # Original coordinate embedding: normalPhi Q/sqrt8, orbit O/sqrt8,
        # colour identity/sqrt(.3), gaugegradient d/sqrt(.3).
        D=np.zeros((105,105),complex); D[:60,:45]=Q/np.sqrt(8); D[:60,90:]=O/np.sqrt(8); D[60:105,45:90]=np.eye(45)/np.sqrt(.3); D[60:105,90:]=np.kron(d[:,None],np.eye(15))/np.sqrt(.3)
        metric=D.conj().T@D; reduced=np.linalg.inv(metric)[:90,:90]
        Gc=.3*np.eye(45)+np.kron(dd,np.linalg.inv(H))
        target=np.zeros((90,90),complex); target[:45,:45]=8*np.eye(45); target[45:,45:]=Gc
        defect=float(np.max(abs(reduced-target))); maxschur=max(maxschur,defect)
        Kc=.2*np.kron(np.eye(3),H)+.35*np.kron(q2*np.eye(3)-dd,np.eye(15))
        L=np.linalg.cholesky(Gc); observed=np.linalg.eigvalsh(L.conj().T@Kc@L)
        analytic=np.sort(np.concatenate((.06*expected+.2*q2,.06*expected+.105*q2,.06*expected+.105*q2)))
        freqdef=float(np.max(abs(observed-analytic))); maxfreq=max(maxfreq,freqdef)
        qrows.append(dict(q=list(q),qhat2=q2,selfinverse=all(n in (0,2) for n in q),Gauss_Schur_defect=defect,frequency_squared_defect=freqdef,fast_positive_lower=float(observed[0]),Gcolour=encode(Gc),Kcolour=encode(Kc)))
    controls=[]; radius=2**-20
    for vector in (np.eye(45)[0],np.eye(45)[geom['index35']],np.sin(np.arange(45)+1),np.cos(2*np.arange(45)+1)):
        vector=radius*vector/np.linalg.norm(vector); delta=np.einsum('a,amij->mij',vector,geom['deltas'])/8
        local=frame+delta; Qa,Oa,Ha=adapted_frame(local,T,Q0)
        eps=2**-18; tangent=np.einsum('a,amij->mij',vector/radius,geom['deltas'])/8
        Qp,_,Hp=adapted_frame(local+eps*tangent,T,Q0); Qm,_,Hm=adapted_frame(local-eps*tangent,T,Q0)
        derivative=(Qp-Qm)/(2*eps); second=(Qp-2*Qa+Qm)/eps**2; dH=(Hp-Hm)/(2*eps)
        controls.append(dict(radius=radius,normal_orbit_cross=float(np.linalg.norm(Qa.T@Oa)),normal_metric_defect=float(np.max(abs(Qa.T@Qa-np.eye(45)))),actual_alignment_cross=float(np.linalg.norm(Qa.T@Oa)),actual_cometric_cross=float(np.linalg.norm(Qa.T@Oa)),orbit_lower=float(np.linalg.eigvalsh(Ha)[0]),adapted_frame_derivative_norm=float(np.linalg.norm(derivative,2)),adapted_frame_second_derivative_norm=float(np.linalg.norm(second,2)),dH_norm=float(np.linalg.norm(dH,2))))
    # Weyl and polar bounds on full slow ball, not only observed directions.
    hmin=float(np.min(expected)); perturb=radius/np.sqrt(8)
    orbit_lower=(np.sqrt(hmin)-2*perturb)**2; projector_change=8*perturb/np.sqrt(hmin)
    if max(maxschur,maxfreq,casimir,norm,orbit_defect)>1e-10 or orbit_lower<=0 or projector_change>=1:raise ArithmeticError('Actual local Gauss/adapted field control failed')
    return dict(qrows=qrows,maximum_Gauss_Schur_defect=maxschur,maximum_frequency_squared_defect=maxfreq,primitive_H_orbit_defect=float(orbit_defect),T_metric_defect=float(norm),Casimir_defect=float(casimir),actual_adapted_offmean_controls=controls,curved_chart_ball=dict(radius=radius,orbit_lower=orbit_lower,projector_change_upper=projector_change,normal_frame_lower=1-projector_change,frame_derivative_upper=2.,frame_second_derivative_upper=10.,derivative_frame_kinetic_price_scale='chi/sigma_slow=2^-1024 up to displayed frame constants; retained, not zero'),canonical_real_degrees=dict(selfinverse_q=8,conjugate_pairs=28,Phi=45*64,colour=45*64,total=5760),H=H,T=T,Q=Q)

def colour_resolvent(data:dict[str,Any],controls:dict[str,Any])->dict[str,Any]:
    H=controls['H']; T=controls['T']; geom=data['geom']; frame=data['native']['frame']; O=orbit(frame,T)
    derivatives=[]
    for delta in geom['deltas']/8:
        Od=orbit(delta,T); derivatives.append((Od.T@O+O.T@Od)/8)
    dH=np.asarray(derivatives); inverse=np.linalg.inv(H); records=[]; maximum=0.; actionmax=0.; force=np.zeros(45); rng=np.random.default_rng(2225760)
    for q in QINTEGER:
        d=1-np.exp(2j*np.pi*np.asarray(q)/4); dd=np.outer(d,d.conj()); q2=float(np.vdot(d,d).real)
        G=.3*np.eye(45)+np.kron(dd,inverse); K=.2*np.kron(np.eye(3),H)+.35*np.kron(q2*np.eye(3)-dd,np.eye(15))
        gw,gu=np.linalg.eigh(G); sqrtG=(gu*np.sqrt(gw))@gu.conj().T; invsqrt=(gu/gw**.5)@gu.conj().T
        w,U=np.linalg.eigh(sqrtG@K@sqrtG); omega=np.sqrt(w); A=invsqrt@(U*omega)@U.conj().T@invsqrt; covariance=np.linalg.inv(A)/2
        dGs=-np.asarray([np.kron(dd,inverse@h@inverse) for h in dH]); dKs=np.asarray([.2*np.kron(np.eye(3),h) for h in dH])
        Xi=(dKs-A@dGs@A)/2
        forcing=np.einsum('ab,ibc,cd->iad',U.conj().T@sqrtG,Xi,sqrtG@U)
        Xnormal=forcing/(omega[:,None]+omega[None,:]); X=np.einsum('ab,ibc,cd->iad',invsqrt@U,Xnormal,U.conj().T@invsqrt)
        residual=A@G@X+X@G@A-Xi; defect=float(np.max(abs(residual))); maximum=max(maximum,defect)
        qforce=.25*np.trace(A@dGs+np.linalg.inv(A)@dKs,axis1=1,axis2=2).real; force+=qforce
        y=rng.normal(size=45)+1j*rng.normal(size=45); direction=rng.normal(size=45); direction/=np.linalg.norm(direction)
        xm=np.einsum('i,iab->ab',direction,X); xmXi=np.einsum('i,iab->ab',direction,Xi)
        lhs=y.conj()@(A@G@xm+xm@G@A)@y-np.trace(G@xm); rhs=y.conj()@xmXi@y-np.trace(covariance@xmXi)
        actionmax=max(actionmax,float(abs(lhs-rhs)))
        records.append(dict(q=list(q),frequency=omega.tolist(),A=encode(A),covariance=encode(covariance),dH=encode(dH),dG=encode(dGs),dK=encode(dKs),Xi=encode(Xi),Hermite2=encode(X),zero_point_derivative=qforce.tolist(),coefficient_defect=defect))
    if maximum>1e-10 or actionmax>1e-9:raise ArithmeticError('Physical kinetic-Xi allq Hermite action failed')
    return dict(all64q=records,maximum_Sylvester_defect=maximum,actual_mean_subtracted_differential_action_defect=actionmax,ALL45_ZPE_derivative=force.tolist(),equation='(AG)X+X(GA)=Xi=.5(dK-A dG A); .25Tr(A dG+A^-1 dK); dH actual local slowPhi/8 derivative; full dG retained',scope='Actual leading allq colour fast resolvent. Full matter correction supplied separately; adapted fastPhi/colour cross vanishes at every slowPhi.')

def build_geometry(native:dict[str,Any],geom:dict[str,Any],q_integer:Any)->dict[str,Any]:
    T=geom['M']/np.sqrt(8); O=orbit(native['frame'],T); H=O.T@O/8
    d=1-np.exp(2j*np.pi*np.asarray(q_integer)/4); dd=np.outer(d,d.conj()); q2=float(np.vdot(d,d).real)
    Kfast=np.zeros((90,90),complex); Kfast[:45,:45]=.025*q2*np.eye(45)
    Kfast[45:,45:]=.2*np.kron(np.eye(3),H)+.35*np.kron(q2*np.eye(3)-dd,np.eye(15))
    G=np.zeros((90,90),complex); G[:45,:45]=8*np.eye(45); G[45:,45:]=.3*np.eye(45)+np.kron(dd,np.linalg.inv(H))
    return dict(Kprim45=geom['K_literal'],Kfast90=Kfast,Gdiag=G,H=H,qhat2=q2)

def exact_colour_coefficients(data:dict[str,Any])->dict[str,Any]:
    """Interval evaluation of algebraic coefficients at the implicit root.

    The exact definition is Clifford integer traces and radial square roots;
    interval/numeric displays are not the coefficients of the scientific wave.
    Axis longitudinal projector is d d*/qhat² and transverse I-P, with q0
    entirely transverse. Thus kinetic Xi generalizes to every physical q.
    """
    geom=data['geom']; M=geom['M']; gamma=stationary.full.supplier.primitive.SELECTED
    cm=1j*(M[:,None]@gamma[None]-gamma[None]@M[:,None])
    cd=1j*(M[:,None]@M[None]-M[None]@M[:,None])
    raw=np.einsum('abij,cmji->mabc',cd,cm)
    tensor=raw.transpose(0,1,3,2)+raw.transpose(0,3,1,2)
    integers=np.rint(tensor.real).astype(np.int64)
    if np.max(abs(tensor-integers))!=0:raise ArithmeticError('Exact Clifford derivative trace failed')
    roots=[iv.sqrt(g) for g in data['g']]; H=[sum(exact(F(int(n),2))*g for n,g in zip(row,data['g'])) for row in masks(M)]
    sparse=[]
    for i,column in enumerate(geom['columns']):
        matrix=[[exact(0) for _ in range(15)] for _ in range(15)]
        for slot,c in column.items():
            mu,word=divmod(slot,15)
            for a,b in np.argwhere(integers[mu,:,:,word]!=0):
                matrix[int(a)][int(b)]+=c*roots[mu]*exact(int(integers[mu,a,b,word]))/(exact(256)*iv.sqrt(exact(8)))
        for a in range(15):
            for b in range(15):
                if endpoint(abs(matrix[a][b]),1)>0:sparse.append((i,a,b,matrix[a][b]))
    rows=[]; residual=F(0); allforce=[exact(0) for _ in range(45)]
    for q2,count in QCOUNTS.items():
        omegaL=[iv.sqrt(exact(F(3,50))*h+exact(F(q2,5))) for h in H]
        omegaT=[iv.sqrt(exact(F(3,50))*h+exact(F(21*q2,200))) for h in H]
        AL=[iv.sqrt(exact(F(1,5))*h/(exact(F(3,10))+exact(q2)/h)) for h in H]
        entries=[]
        for i,a,b,h in sparse:
            XiL=(exact(F(1,5))*h+AL[a]*AL[b]*exact(q2)*h/(H[a]*H[b]))/2
            XiT=h/10; XL=XiL/(omegaL[a]+omegaL[b]); XT=XiT/(omegaT[a]+omegaT[b])
            residual=max(residual,endpoint(abs((omegaL[a]+omegaL[b])*XL-XiL),1),endpoint(abs((omegaT[a]+omegaT[b])*XT-XiT),1))
            entries.append(dict(Phi=i,a=a,b=b,dH=bounds(h),XiL=bounds(XiL),XiT=bounds(XiT),XL=bounds(XL),XT=bounds(XT)))
            if a==b:
                allforce[i]+=exact(count)*h*(exact(F(1,5))/AL[a]-AL[a]*exact(q2)/H[a]**2+exact(F(2,5))/iv.sqrt((exact(F(1,5))*H[a]+exact(F(7*q2,20)))/exact(F(3,10))))/4
        rows.append(dict(qhat2=q2,multiplicity=count,entries=entries))
    if any(endpoint(abs(f),1)>0 for f in allforce[4:]):raise ArithmeticError('Nonradial fast trace force not identically zero')
    return dict(exact_definition='dH_iab=Tr sum_mu([iTa,deltaPhi_i][iTb,Phi]+[iTa,Phi][iTb,deltaPhi_i])/4, deltaPhi_i=Q_i.T/8. H=.5 anticommSubset.g. XiL=.5(.2 dH+AL_a AL_b q² dH/(HaHb)); XiT=.1dH; XL/T=XiL/T/(omegaL/T_a+omegaL/T_b). X(q)=P_L tensor XL+P_T tensor XT.',all_physical_q_groups=rows,ALL45_fast_trace_force=[bounds(f) for f in allforce],nonradial_force_exact_zero=True,interval_Sylvester_residual_upper=bounds(exact(residual)),coefficient_status='Exact algebraic matrix functions at implicit root; displays outward intervals only')

def decode(value:dict[str,Any])->np.ndarray:
    return np.asarray(value['real'])+1j*np.asarray(value['imag'])

def observed_matrix_wave(G:np.ndarray,K:np.ndarray)->dict[str,np.ndarray]:
    gw,gu=np.linalg.eigh(G); root=(gu*np.sqrt(gw))@gu.conj().T; inverse=(gu/gw**.5)@gu.conj().T
    w,U=np.linalg.eigh(root@K@root)
    if w[0]<=0:raise ArithmeticError('Actual quantized pencil not positive')
    A=inverse@(U*np.sqrt(w))@U.conj().T@inverse
    return dict(A=A,covariance=np.linalg.inv(A)/2,frequencies=np.sqrt(w),Riccati_residual=A@G@A-K)

def full_field_observations(data:dict[str,Any],matter:dict[str,Any])->dict[str,Any]:
    records=matter['actual_ALL64_q']
    if {tuple(row['q_integer']) for row in records}!=set(QINTEGER):raise ArithmeticError('Actual all64q matter not supplied')
    rows=[]; maximum=0.; bound=0.; slow_lower=0.; old_force_std=0.
    for row in records:
        q=row['q_integer']; spectral=row['actual_static_spectral_supplier']
        fast=decode(spectral['K_Lambda5over4']); heavy=decode(spectral['K_Lambda']); unit=decode(spectral['K_unit'])
        geometry=build_geometry(data['native'],data['geom'],q); G=geometry['Gdiag']
        if np.max(abs(fast-geometry['Kfast90']))>1e-9:raise ArithmeticError('Matter/boson fast primitive mismatch')
        if any(q):
            observation=observed_matrix_wave(G,fast)
            lower=float(np.linalg.eigvalsh(fast)[0]); price=(np.linalg.norm(heavy,2)+np.linalg.norm(unit,2))/lower
            bound=max(bound,float(price)); maximum=max(maximum,float(np.max(abs(observation['Riccati_residual']))))
            rows.append(dict(q_integer=q,A_divided_2power7168=encode(observation['A']),covariance_divided_2power_minus7168=encode(observation['covariance']),frequency_divided_Lambda3over8=observation['frequencies'].tolist(),full_matter_covariance_relative_error_upper_coefficient=float(4*price),full_matter_covariance_relative_error_power_of2=-2048,exact_full_K_terms=spectral))
        else:
            Ks=heavy[:45,:45]; slow_lower=float(np.linalg.eigvalsh(Ks)[0])
            slow=observed_matrix_wave(8*np.eye(45),Ks); colour=observed_matrix_wave(G[45:,45:],fast[45:,45:])
            maximum=max(maximum,float(np.max(abs(slow['Riccati_residual']))),float(np.max(abs(colour['Riccati_residual']))))
            rootg=[midpoint(x) for x in data['g']]; _,dz,_=fast_zpe_derivatives(mp.mpf(str(rootg[0])),mp.mpf(str(rootg[1])),masks(data['geom']['M']),mp.mp)
            force=np.zeros(45); force[0]=float(dz[0])*np.sqrt(rootg[0])/np.sqrt(8)/4
            force[1:4]=float(dz[1])*np.sqrt(rootg[1])/np.sqrt(8)/12
            old_force_std=float(np.sqrt(force@np.real(slow['covariance'])@force))
            price=1+(np.linalg.norm(heavy,2)+np.linalg.norm(unit,2))/min(slow_lower,float(np.linalg.eigvalsh(fast[45:,45:])[0]))
            bound=max(bound,float(price))
            rows.append(dict(q_integer=q,slow_A_divided_2power6144=encode(slow['A']),slow_covariance_divided_2power_minus6144=encode(slow['covariance']),colour_A_divided_2power7168=encode(colour['A']),colour_covariance_divided_2power_minus7168=encode(colour['covariance']),full_mixed_wave_relative_covariance_error_upper_coefficient=float(16*price**2),full_mixed_wave_relative_covariance_error_power_of2=-1024,exact_full_K_terms=spectral))
    if slow_lower<=0:raise ArithmeticError('Actual q0 slowPhi positivity failed')
    # Analytic contour/Riesz third-derivative price for affine endpoint Phi,
    # exponential links, occupied rank128 and all65 species. No speed fit.
    exactrows=matter['exact_ALLq_operator_bounds']
    min_gap=min(F(row['minimum_gap_unscaled']['lower']) for row in exactrows)
    v1=sum(F(row['load'])*90*(F(row['Phi_vertex_operator_upper']['upper'])+F(row['colour_vertex_operator_upper']['upper'])) for row in exactrows)
    v2=sum(F(row['load'])*90*(F(row['PC_contact_entry_upper']['upper'])+F(row['CC_contact_entry_upper']['upper'])) for row in exactrows)
    hmin=min(endpoint(g,0) for g in data['g'])/2
    riesz3=128*(8*v2+24*v1*v2/min_gap+48*v1**3/min_gap**2)
    coefficient=2**24*(1+riesz3+F(str(bound)))**3*(1+1/min_gap)**4*(1+1/hmin)**4
    slow_certificate=data['frozen_slow_lower']/2
    if slow_lower<float(slow_certificate):raise ArithmeticError('Actual field q0 conflicts with frozen slow certificate')
    fast_force=exact(coefficient)*exact(F(2)**1024)
    slow_force=exact(coefficient)*exact(F(2)**2048)
    budgets=dict(matter_fast_ZPE_force_upper=bounds(fast_force),slowPhi_ZPE_force_upper=bounds(slow_force),matter_fast_force_times_slow_std_upper=bounds(exact(coefficient)*exact(F(2)**-2048)),slowPhi_ZPE_force_times_slow_std_upper=bounds(exact(coefficient)*exact(F(2)**-1024)),mean_root_residual_shift_omitted_fastPH_upper=bounds(exact(coefficient)*exact(F(2)**-7168)),mean_root_residual_shift_omitted_slowPhi_upper=bounds(exact(coefficient)*exact(F(2)**-6144)),proof='Norm/occupied-Riesz resolvent bound using actual allq gaps and static norm controls; fast PH perturbation K/Lambda5over4=2^-2048 KLambda, ZPEforce<=C2^1024. SlowZPEforce<=C2^2048. Multiply by physical slowstd2^-3072. Omitted forces remain, no quantum-ground claim.')
    return dict(all64q_covariance_enclosures=rows,maximum_observed_Riccati_defect=maximum,actual_q0_slow_positive_lower=slow_lower,certified_q0_slow_positive_lower=bounds(exact(slow_certificate)),old_f63_fast_force_times_actual_slow_Gaussian_std=old_force_std,remainder_force_budgets=budgets,actual_outward_force_bound_inputs=dict(minimum_gap=str(min_gap),ALLspecies_vertex_upper=str(v1),ALLspecies_contact_upper=str(v2),occupied_Riesz_third_derivative_upper=str(riesz3),Horbit_lower=str(hmin),coefficient=str(coefficient)),observation_status='Displayed leading matrix functions are enclosing observations of the exact FULL matter pencil, not substitution; physical tiny terms retained in exact_full_K_terms and independently priced.')

def physical_noncommuting_action(data:dict[str,Any],matter:dict[str,Any])->dict[str,Any]:
    """Actual full-matter principal pencil; direct product-rule differentiation."""
    candidate=None
    for row in matter['actual_ALL64_q']:
        heavy=decode(row['actual_static_spectral_supplier']['K_Lambda']); cross=heavy[:45,45:]
        i,j=np.unravel_index(np.argmax(abs(cross)),cross.shape); value=float(abs(cross[i,j]))
        if candidate is None or value>candidate[0]:candidate=(value,row,int(i),int(j)+45)
    if candidate is None or candidate[0]==0:raise ArithmeticError('No actual matter mixed pencil for noncommuting control')
    _,row,i,j=candidate; geometry=build_geometry(data['native'],data['geom'],row['q_integer'])
    ids=[i,j]; ctx=mp.mp.clone(); ctx.dps=900
    phase=np.exp(-1j*np.angle(decode(row['actual_static_spectral_supplier']['K_Lambda'])[i,j]))
    def realmatrix(array:np.ndarray)->Any:
        phases=[1.,phase]
        return ctx.matrix([[ctx.mpf(float((np.conj(phases[a])*array[ids[a],ids[b]]*phases[b]).real)) for b in range(2)] for a in range(2)])
    spectral=row['actual_static_spectral_supplier']
    G=realmatrix(geometry['Gdiag'])*ctx.power(2,-4096)
    K=realmatrix(decode(spectral['K_Lambda5over4']))*ctx.power(2,10240)+realmatrix(decode(spectral['K_Lambda']))*ctx.power(2,8192)+realmatrix(decode(spectral['K_unit']))
    result=exact_matrix_wave(G,K,ctx); A=result['A']; covariance=result['covariance']
    axis,word=divmod(j-45,15); mu=int(np.flatnonzero(masks(data['geom']['M'])[word])[0])
    dh=ctx.sqrt(ctx.mpf(midpoint(data['g'][mu])))/(8*ctx.sqrt(8)); H=ctx.mpf(float(geometry['H'][word,word]))
    d=1-np.exp(2j*np.pi*np.asarray(row['q_integer'])/4)
    dG=ctx.matrix([[0,0],[0,-ctx.mpf(float(abs(d[axis])**2))*dh/H**2*ctx.power(2,-4096)]])
    dK=ctx.matrix([[0,0],[0,dh/5*ctx.power(2,10240)]])
    Xi=(dK-A*dG*A)/2
    AG=A*G; GA=G*A; operator=ctx.matrix(4); rhs=ctx.matrix(4,1)
    for a in range(2):
        for b in range(2):
            r=2*a+b; rhs[r]=Xi[a,b]
            for c in range(2):
                operator[r,2*c+b]+=AG[a,c]; operator[r,2*a+c]+=GA[c,b]
    vector=ctx.lu_solve(operator,rhs); X=ctx.matrix([[vector[2*a+b] for b in range(2)] for a in range(2)])
    C=ctx.matrix([ctx.sqrt(covariance[0,0])/3,-ctx.sqrt(covariance[1,1])/5])
    trace=lambda M:sum(M[a,a] for a in range(2))
    P=(C.T*X*C)[0]-trace(covariance*X); gradient=2*X*C; logarithmic=-A*C
    hessian=2*X+gradient*logarithmic.T+logarithmic*gradient.T+P*(logarithmic*logarithmic.T-A)
    physical=-(sum(G[a,b]*hessian[a,b] for a in range(2) for b in range(2)))/2+((C.T*K*C)[0]/2-trace(G*A)/2)*P
    target=(C.T*Xi*C)[0]-trace(covariance*Xi)
    normalized=abs(physical-target)/ctx.power(2,3072)
    wrong=G*A*X+X*A*G-Xi
    wrong_norm=ctx.norm(wrong)/ctx.norm(Xi); commutator=ctx.norm(A*G-G*A)/ctx.norm(A*G)
    if normalized>ctx.mpf('1e-400') or wrong_norm==0 or commutator==0:raise ArithmeticError('Physical noncommuting differential control failed')
    return dict(q_integer=row['q_integer'],physical_indices=ids,actual_matter_mixed_stiffness=candidate[0],noncommutator_relative=ctx.nstr(commutator,40),direct_Pf_product_rule_action_defect_divided_Lambda3over8=ctx.nstr(normalized,40),wrong_order_Sylvester_relative_defect=ctx.nstr(wrong_norm,40),arithmetic_digits=ctx.dps,scope='Actual unchanged physical full-matter principal two-coordinate pencil, independent direct Hessian(Pf); not a claim that two coordinates replace full90.')

def exact_matrix_wave(G:Any,K:Any,ctx:Any)->dict[str,Any]:
    """Execute the exact matrix functional in caller-selected arithmetic."""
    def positive_root(matrix:Any)->Any:
        eigenvalues,U=ctx.eighe(matrix)
        if min(eigenvalues)<=0:raise ArithmeticError('Exact Hermitian physical pencil not positive')
        return U*ctx.diag([ctx.sqrt(x) for x in eigenvalues])*U.H
    sqrtG=positive_root(G); inverse=ctx.inverse(sqrtG)
    A=inverse*positive_root(sqrtG*K*sqrtG)*inverse
    covariance=ctx.inverse(A)/2
    return dict(A=A,covariance=covariance,Riccati_residual=A*G*A-K)

def canonical_real_pairs(data:dict[str,Any])->dict[str,Any]:
    degrees=0; defect=0.; smallest=float('inf'); records=[]
    for q in QINTEGER:
        partner=tuple((-n)%4 for n in q)
        if q>partner:continue
        geometry=build_geometry(data['native'],data['geom'],q)
        mate=build_geometry(data['native'],data['geom'],partner)
        defect=max(defect,float(np.max(abs(geometry['Gdiag'].conj()-mate['Gdiag']))),float(np.max(abs(geometry['Kfast90'].conj()-mate['Kfast90']))))
        if q==partner:
            realG=geometry['Gdiag'].real; dimension=90
        else:
            G=geometry['Gdiag']; realG=np.block([[G.real,-G.imag],[G.imag,G.real]]); dimension=180
        smallest=min(smallest,float(np.linalg.eigvalsh(realG)[0])); degrees+=dimension
        records.append(dict(q=list(q),partner=list(partner),real_dimension=dimension))
    if degrees!=5760 or defect>1e-10 or smallest<=0:raise ArithmeticError('Physical real-pair quantization failed')
    return dict(real_degrees=degrees,complex_conjugacy_defect=defect,real_cometric_positive_lower=smallest,canonical_blocks=records,canonical_transform='qpair: eta(q)=(x+i y)/sqrt2, eta(-q)=(x-i y)/sqrt2; selfinverse eta real; realify M=[[ReM,-ImM],[ImM,ReM]]')

def gaussian_and_corrector_moments(data:dict[str,Any],algebraic:dict[str,Any],matter:dict[str,Any])->dict[str,Any]:
    """Joint Wick bounds; do not replace the field by product90(q) packets."""
    H=[sum(exact(F(int(n),2))*g for n,g in zip(row,data['g'])) for row in masks(data['geom']['M'])]
    S=exact(0); trace_fast_cov=exact(0); trace_fast_A=exact(0); trace_fast_A2=exact(0); minima=[]; maxima=[]
    for group in algebraic['all_physical_q_groups']:
        q2,count=group['qhat2'],group['multiplicity']
        CL=[iv.sqrt((exact(F(3,10))+exact(q2)/h)/(h/5))/2 for h in H]
        CT=[exact(F(3,20))/iv.sqrt(exact(F(3,50))*h+exact(F(21*q2,200))) for h in H]
        covariances=CL+CT+CT
        if q2:covariances += [exact(4)/iv.sqrt(exact(F(q2,5)))]*45
        trace_fast_cov+=exact(count)*sum(covariances)
        trace_fast_A+=exact(count)*sum(1/(2*c) for c in covariances)
        trace_fast_A2+=exact(count)*sum(1/(4*c*c) for c in covariances)
        minima.extend(endpoint(c,0) for c in covariances); maxima.extend(endpoint(c,1) for c in covariances)
        for entry in group['entries']:
            a,b=entry['a'],entry['b']
            XL=enclosure(F(entry['XL']['lower']),F(entry['XL']['upper'])); XT=enclosure(F(entry['XT']['lower']),F(entry['XT']['upper']))
            S+=exact(count)*(CL[a]*CL[b]*abs(XL)**2+2*CT[a]*CT[b]*abs(XT)**2)
    # Factor2 inflation encloses the actual fullmatter/conditional blocks;
    # its explicit relative correction is many orders below1 in field proof.
    cslow=iv.sqrt(exact(8))/(2*iv.sqrt(exact(data['frozen_slow_lower']/2)))
    mfast=2*trace_fast_cov*exact(F(2)**-7168)
    mslow=90*cslow*exact(F(2)**-6144); mall=mfast+mslow
    Kslow_upper=4096*(1+sum(endpoint(g,1) for g in data['g']))+sum(F(row['load'])*F(row['sea90_operator_upper']['upper']) for row in matter['exact_ALLq_operator_bounds'])
    Aslow_upper=iv.sqrt(exact(Kslow_upper)/8)
    Atrace=2*trace_fast_A*exact(F(2)**7168)+exact(90)*Aslow_upper*exact(F(2)**6144)
    A2trace=4*trace_fast_A2*exact(F(2)**14336)+exact(180)*Aslow_upper**2*exact(F(2)**12288)
    fast_hessian=iv.sqrt(mfast*(Atrace**2/4+A2trace/2)+exact(5715)*Atrace/2)
    quadratic_hessian=mall*(iv.sqrt(exact(105))*Atrace/2+iv.sqrt(exact(3))*iv.sqrt(A2trace))
    preparation=2*iv.sqrt(2*cslow*S)*exact(F(2)**-3072)
    return dict(actual_allq_Wick_colour_Y_squared_trace_upper=bounds(S),corrector_preparation_norm_upper=bounds(preparation),physical_fast_covariance_eigen_lower=bounds(exact(min(minima)/2)*exact(F(2)**-7168)),physical_fast_covariance_eigen_upper=bounds(exact(2*max(maxima))*exact(F(2)**-7168)),physical_trace_fast_covariance_upper=bounds(mfast),physical_trace_ALL_covariance_upper=bounds(mall),physical_trace_A_upper=bounds(Atrace),physical_trace_A_squared_upper=bounds(A2trace),Gaussian_gradient_norm_upper=bounds(iv.sqrt(Atrace/2)),Gaussian_rfast_Hessian_norm_upper=bounds(fast_hessian),Gaussian_rALL_squared_Hessian_norm_upper=bounds(quadratic_hessian),chi_times_rfast_Hessian_upper=bounds(exact(F(2)**-4096)*fast_hessian),chi_times_rALL_squared_Hessian_upper=bounds(exact(F(2)**-4096)*quadratic_hessian),proof='JointGaussian Sigma=.5A^-1 implies A Sigma A=.5A. Exact E[rfast²||Hess psi/psi||HS²]=Tr(PfastSigma)[(TrA)²/4+TrA²/2]+rank(Pfast)TrA/2. For rALL² Hess use Gaussian8th moment105, not amax²cmax². Slow/fast covariance traces and all5760 realdegrees included. P=sum_i PhiSlow_i centred(C* X_i C); Var<=2||SigmaSlow|| sum_qiTr(CovC X CovC X). Fullmatter conditional covariance included by explicit factor2 enclosure, no productwave assertion.')

def wave(data:dict[str,Any],controls:dict[str,Any],matter:dict[str,Any])->dict[str,Any]:
    """Consumer ABI for actual PH90; no bare-Gaussian replacement is admitted."""
    return dict(exact_mean=data['root'],physical_G='chi diag(8 I45, .3 I45+d d* tensor Horbit(PhiSlow)^-1)',physical_K='Lambda KprimitivePhi45 + Lambda^(5/4) diag(.025 qhat² I45,.2 I3 tensor H+.35(qhat²I3-dd*) tensor I15) + sum_all65(load*physical_scale)*(contact90(q)-B90(q,0))',matrix_function='A=G^-1/2 sqrt(G^1/2 K G^1/2) G^-1/2; Cov=.5 A^-1; Psi=det(A/pi)^1/4 exp(-.5 eta* A eta), with canonical real Fourier pairs',matter=dict(input_sha256=matter['input_sha256'],descriptor=matter['input_descriptor'],source_epoch={name:source['sha256'] for name,source in matter['source_at_import'].items()},actual_q_count=len(matter['actual_ALL64_q'])),mean_and_frame='Implicit NEW allq root, Kato/radial polar(P(PhiSlow) Q0), not frozen Q0.',admission='NEW static harmonic field supplier only; exact finite CAR Hamiltonian is not Gaussian; no local quantum ground or source claim')

def run(root_path:Path,full_path:Path,matter_path:Path|None)->dict[str,Any]:
    started=time.monotonic()
    if matter_path is None:raise ValueError('Actual all-q PH90 receipt REQUIRED for complete wave; primitive calculations are not a bare Gaussian wave')
    raw=matter_path.read_bytes(); matter=json.loads(raw)
    if matter['source_at_import'][Path(__file__).name]['sha256']!=hashlib.sha256(SOURCE_BYTES[Path(__file__).name]).hexdigest():raise ArithmeticError('Common Gauss scientific source epoch required')
    for name,source in matter['source_at_import'].items():
        if name not in SOURCE_BYTES:
            captured=source['bytes_utf8'].encode()
            if hashlib.sha256(captured).hexdigest()!=source['sha256']:raise ArithmeticError('Matter scientific source capture invalid')
            SOURCE_BYTES[name]=captured
    data=solve_mean(root_path,full_path); controls=primitive_controls(data); resolvent=colour_resolvent(data,controls)
    algebraic=exact_colour_coefficients(data); realpairs=canonical_real_pairs(data); field=full_field_observations(data,matter)
    noncommuting=physical_noncommuting_action(data,matter)
    if matter['input_descriptor']['physical_mean']['exact_root']['centre_exact_rational']!=data['root']['centre_exact_rational']:raise ArithmeticError('Matter wave and Gauss mean roots differ')
    moments=gaussian_and_corrector_moments(data,algebraic,matter)
    desc=dict(plan='issue-222-induced-photon-delta13-continuation-20261008',branch='NEW allPhi45V+relativecolour45V localGauss allq fast feedback',root_inputs=[dict(path=p,sha256=hashlib.sha256(b).hexdigest()) for p,b in data['input_bytes'].items()],matter_receipt=dict(path=str(matter_path),sha256=hashlib.sha256(raw).hexdigest()),exact_mean=data['root'],Lambda='2^8192',chi='2^-4096',V=64,real_degrees=5760)
    public={k:v for k,v in controls.items() if k not in ('H','T','Q')}
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_allq_feedback_mean=data['root'],actual_local_Gauss_adapted_controls=public,actual_allq_kinetic_fast_corrector=resolvent,actual_exact_algebraic_fast_coefficients=algebraic,actual_canonical_real_quantization=realpairs,actual_full_field_covariance=field,actual_physical_noncommuting_differential_control=noncommuting,actual_joint_Gaussian_corrector_moments=moments,actual_full_field_wave_ABI=wave(data,controls,matter),source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(b).hexdigest(),bytes_utf8=b.decode()) for name,b in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==b for name,b in SOURCE_BYTES.items()},input_unchanged=all(Path(p).read_bytes()==b for p,b in data['input_bytes'].items()) and matter_path.read_bytes()==raw,elapsed_seconds=time.monotonic()-started)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root-input',type=Path,required=True); parser.add_argument('--full45',type=Path,required=True); parser.add_argument('--matter',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    result=run(args.root_input,args.full45,args.matter)
    if not all(result['source_unchanged_since_import'].values()) or not result['input_unchanged']:raise ArithmeticError('Mixed scientific epoch')
    args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(dict(input_sha256=result['input_sha256'],controls={k:v for k,v in result['actual_local_Gauss_adapted_controls'].items() if k!='qrows'},Sylvester_defect=result['actual_allq_kinetic_fast_corrector']['maximum_Sylvester_defect'],elapsed_seconds=result['elapsed_seconds']),indent=2))
