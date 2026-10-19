"""Canonical full-law heat-functional and current/contact consumer; no ground identified."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import zipfile
import numpy as np
from scipy.linalg import expm
from scipy.special import roots_hermitenorm
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import joined_phi_u1_stationary_full_phi_fiber as fiber

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_stationary_canonical_photon'
PROOF = {
 'law': 'H=4chi p_x^2+(3chi/20)E_SU4^2+64Pi^2+V4+Va+Vm+64rho Lplus rho+dGamma(-Lambda hneutral)+dGamma(hactive)+sum_63 dGamma(haux). x is3840 Cartesian TrTT=1/2 coordinates,192 fullHaar links and129 cycle coordinates. Lambda=2^8192, chi=2^-4096, V4=Lambda/a sum_x||Gram||F^2, a=1/64, Va=2^10240/40 sum||Phi_x-U Phi_y Udag||HS^2, Vm=(7/10)2^10240 sum_faces(4-ReTr orderedUface). The bosonic2^10240 is NOT charged energy. No bare EM potential.',
 'canonical': 'Background Nneutral=128, Nactive=128, each of63 auxiliaries Naux=128, Ref4. Source Nactive=129, Ref3 is a DISTINCT sector. rho_x=sum_64charged(n_s,x-2)+(NRef-4)delta_x0. Neutral is U1 neutral and has NEGATIVE physical Hamiltonian -Lambda hneutral, not a positive fictitious gap. Ref remains wedge^NRef(C4) with its Gauss representation, not only its scalar charge.',
 'heat_form': 'For arbitrary thermal inverse temperature b>0 and each configuration Bloch beta, use Friedrichs form H_beta from full_phi_fiber. On its smooth compact Cartesian/smooth Haar/Bloch canonical physical core, Vmatter is matrix multiplication linear in Phi; quartic confines it and KLMN closes the sum. Form-Trotter products converge strongly to exp(-b H_beta), with kinetic and full matrix-potential form domains. Trace convergence requires trace-norm domination/heat-kernel estimates, not strong convergence alone; polynomial confinement and ellipticity on the compact factors supply a trace-class semigroup and can be used with confining lower bounds to prove that extra step. Differentiation under a limiting trace additionally needs insertion-weight domination. No finite-path floating control below establishes these analytic convergence estimates.',
 'full_measure': 'At M steps tau=b/M, integrate all CartesianPhi path points against heat kernel with covariance8chi tau I3840, allSU4 links against Haar/Casimir heat kernel generator(3chi/20)Delta, and Theta cycle paths against heat kernel generator64Delta_Theta. Include exp(-tau(V4+Va+Vm)) perpoint. Do not freeze Phi or colour or discard their kinetic measure. Put compact SU4 physical Gauss projector P_G=int dg Gamma(g) into the canonical trace; equivalently close the paths with gauge-twisted endpoints and fermion monodromy R(g) at the trace boundary. U1 longitudinal Gauss is solved by rho and HC; totalrho=0 is mandatory. The numerical native paths are interior integrand controls at identity Gauss boundary, not the integrated Gauss-projected partition function.',
 'theta_kernel': 'Let saturated C192x129 satisfy DC=0, A=C^TQ, Theta=A^T t, t mod2pi. HTheta=64(p_t+beta)^T(A A^T)^-1(p_t+beta)=64(p_t+beta)^T(C^TC)^-1(p_t+beta). On quasi-periodic sections K_tau(Theta,ThetaPrime)=sum_l exp(2pi i beta.l) (256pi tau)^(-129/2) exp(-|Theta-ThetaPrime-2pi A^T l|^2/(256tau)), with Euclidean fundamental-domain measure. Fourier version has eigenvalues64|A^-1(m+beta)|^2 and normalized characters. Theta diffusion covariance128tau, NOT64tau. beta is configuration quasimomentum, not a photon momentum or thermal b. No real-time2V monodromy sign dichotomy is asserted.',
 'HS': 'Exactly exp(-64tau rho Lplus rho)=E_X exp(i X.rho), X meanzero Gaussian on1perp with Cov(X)=128tau Lplus. The SAME scalar X_j at each step drives active and ALL63aux. Split HC from dGamma(h) by form-Trotter, retaining their noncommutation; conditioned onebody T_s,j=diag(exp(i X_j,x)) exp(-tau h_s(Phi_j,U_j,Theta_j)). Scalar background/Ref phase is exp(i sum_j X_j.(-128*1+(NRef-4)delta0)); uniform baseline cancels because X zero-mean, Ref source does not. Physical neutral T_n,j=exp(+tau Lambda hneutral_unscaled), with no X. Neutral128 coefficient stays exact symbolic at the enormous physical Lambda; it is not numerically replaced by exp(-tau hneutral).',
 'functional': 'With monodromy M_s=T_s,M...T_s,1 and Gauss boundary R(g), F=Ref_Gauss_character(g) exp(i sum X.rho_background) [z_n^128]det(I+z_n R(g)M_n) times [z_a^Nactive]det(I+z_a R(g)M_a) times ([z_u^128]det(I+z_u R(g)M_u))^63. Coefficient extraction is PER COPY before power63; [z^8064]det(I+zM_u)^63 sums forbidden transfers of particle number between independent species. Integrate this complex/oscillatory F with the above bosonic measure. No positive HS probability weight for F or determinant sign claim. The Gaussian sampling measure alone is positive.',
 'finite_hessian': 'For external real probe a inserted in ALL literal native U1 ordered phases, differentiate EACH h_s(a) and exponential BEFORE forming monodromy and canonical projection. B_j=-tau h_j, E_jPrime=int_0^1 exp((1-v)B_j) B_jPrime exp(vB_j)dv. E_jSecond=int exp B_j B_jSecond exp B_j +2 int_{0<u<v<1} exp((1-v)B_j)B_jPrime exp((v-u)B_j)B_jPrime exp(uB_j)du dv. First term is contact; second is ordered two-current Duhamel. Leibniz differentiates the finite product including interstep pairs. For C_N=[z^N]det(I+zM), CPrime=[z^N]det R tr(R^-1 zMPrime), CSecond=[z^N]det R[(tr R^-1 zMPrime)^2-tr((R^-1 zMPrime)^2)+tr(R^-1 zMSecond)]. Log Hessian is CSecond/C-(CPrime/C)^2 where C nonzero, then add active+63aux for CONDITIONED logarithm only. After HS/boson/Gauss integration use ZSecond/Z-(ZPrime/Z)^2, including interspecies covariance; E(log Hessian) is NOT log of E(F).',
 'kinetic_boundary': 'If external a modifies the heat/pathkinetic or Gauss boundary as well, their first/second derivatives must join the same product BEFORE coefficient/log. For an edge-flow momentum source H(a)=64|Pi+a v|^2+HC+matter(a), Jkin=128 v.Pi and Kkin=128|v|^2. Source-independent fixed-background path derivative used numerically differentiates matter only, and is labelled so. FullE=QPi+D^T Lplus rho; hence HE+HC=64|E|^2. Resetting incomingPi loses128Pi.Q^Tk; treating copies as independently averaged loses collective cross terms. Gauss projector remains fixed only for gauge-invariant probes/closed conserved-current insertions.',
 'thermal_spectral': 'For gauge-invariant Hermitian J=dH/da and K=d2H/da2 on the physical sector, d2(-log Z)/da2=b <K>_b-int_0^b dt ds <T J(t)J(s)>_b,connected. Z=Tr_phys exp(-bH_beta). Current correlator spectral C_b(t)=Z^-1 sum_nm exp(-bEn) exp(t(En-Em)) J_nm J_mn minus <J>^2. At a minimizing betaStar and compact-resolvent fiber, exp(-b(H-E0))/Tr exp(-b(H-E0)) converges to P0/dimP0. It does NOT select a unique vector within degenerate P0; a chosen vector requires boundary preparation. For 0<t and a nondegenerate ground, C0(t)=sum_m>0 exp(-t(Em-E0)) |J_0m|^2; frequency C0(iw)=sum_m>0 2(Em-E0)|J_0m|^2/((Em-E0)^2+w^2). Degenerate intra-P0 covariance produces a static term and must be separated. These are SAME interacting ground matrix elements, not onebody monodromy eigenvalues.',
 'next_consumer': 'Use this very H_beta/functionally equivalent heat semigroup to extract stationary physical E0,P0 and the projected resolvent kernel P0 J_q (1-P0)(H-E0+z)^-1 J_-q P0, plus P0 K_q,-q P0, retaining Pi/HC, fullPhi/fullHaar, allcanonical copies and every native conserved-alpha coset. Select/characterize betaStar (not assumed0) and ground degeneracy; establish limiting insertion bounds, then locate photon spectral support/poles/residue/dispersion in that SAME state. Finite conditioned transfer Hessians below do not compute any photon pole or characterize the full ground.',
 'boundaries': 'Exact exterior-algebra canonical identity and Gaussian HS identity hold finite matrices; finite floating residuals are not outward certificates. Three sampled fullnative paths are NOT a converged Trotter path integral, stationary ground, positive measure, allPhi limit or photon dispersion. No root solve, oldSOURCE adoption, new clock or Maxwell law. Prior sectorwise-wilson-legendre.md was not found in bounded project/memory search; no theorem from it is presumed.',
 'regularity_and_meanPi': 'The established physical C* state is beta-diagonal periodicTheta/Wilson and compatibleelectric, not automatic regularity of raw noncompactTheta. Sharp original-cover beta packets can fail regularity for position-Weyl shifts; neither rawTheta covariance nor a normalizable real-coverL2 vacuum is inferred. At global minimizing betaStar, for every ground vector Rayleigh comparison under +/-d in H_beta-H_betaStar=128d.Pi+64d^2 gives <Pi>=0; polarization gives P0 Pi_i P0=0. This is a ground mean constraint, NOT Pi=0 operator or vanishing electric fluctuations; HC and Pi must remain in the functional.',
 'bounded_spectral_consumer': 'For a bounded gauge-invariant periodic Wilson-flux Fourier observable F_q, compute from the SAME physical H_betaStar and ground P0 the matrix P0 F_q^dag(1-P0)(H-E0+z)^-1 F_q P0. ElectricPi insertions require their graph-domain bounds. This bounded spectral consumer can establish an emergent lowenergy photon cluster without compactifying the original law, assuming rawTheta regularity or identifying transfer/Ritz/Taylor nodes with a pole. Current/contact kernel additionally needs the unbounded insertion domination described above.',
 'split_domain': 'For fullPhi, never use exp(-tau dGamma(h)) as a bounded free-standing semigroup: h is unbounded linear in Phi. Group V4+Va+Vm+dGamma(matter) into the semibounded matrix-multiplication form W; pointwise scalar exp(-tau Vbosonic) times the fermion transfer is exp(-tau W). Form-Trotter K, HC and W on their common form domain gives the stated strong representation. Oscillatory HS is exact at each finite HC factor, not a justification to interchange fullPhi/HS integration or derivatives absent domination.',
 'trace_class_and_derivatives': 'Set X=||Phi||HS=(sum_x,mu Tr Phi_x,mu^2)^1/2=||x_Cart||/sqrt2 with Tr(TaTb)=delta_ab/2. Gram PSD/SOS and64-site Cauchy give V4>=Lambda X^4/64. Absorbing the finite-CAR linear matter bound B X by Young gives H+C>=K+Lambda X^4/128. Min-max bounds full eigenvalues by this scalar confining elliptic comparison, tensor a FINITE canonical CAR dimension. Its quartic heat trace is finite by Golden-Thompson/free Cartesian heat diagonal integrated against exp(-b Lambda X^4/128), times compact Haar/Theta heat traces. Compact Gauss restriction cannot increase eigenvalue counts. Fixed gauge-invariant periodic phase probes J,K have matrix-multiplication norms<=const X; quartic form control and semigroup smoothing bound their finite-b Duhamel traces without globally bounded Phi/current assumptions. A traced kinetic/W form-Trotter limit or interchanging HS/Gauss/fullPhi limits must still supply trace-norm/insertion domination rather than infer it from strong convergence. The exact spectral identity is defined by the trace-class H itself, independent of that numerical limiting implementation.',
 'polarization': 'The executable arbitrary-direction quadratic form supplies every mixed current/contact Hessian by H(v,w)=(H(v+w,v+w)-H(v,v)-H(w,w))/2. Numerical native controls below exercise one nonzero transverse direction, not the full192x192 Hessian array. No isotropy or polarization degeneracy is presumed.',
 'coefficient_without_resolvent': 'For monodromy jets M,MPrime,MSecond compute normalized exterior coefficients c_k=e_k(M)/binom(n,k) by division-free-in-M normalized Faddeev-LeVerrier: B_0=I, c_k=Tr(M B_(k-1))/n, B_k=(n c_k I-k M B_(k-1))/(n-k). B_k is the kth Newton transform divided by binom(n-1,k). Apply product-jet Leibniz before each trace and update; stop before the unused B_n. No inverse of I+zM, determinant cutoff or skipped contour node occurs. A determinant zero may have nonzero second jet: shared zero-mean HS X=(pi,-pi), h(a)=sin(a)sigma_x gives M(a)=-exp(-sin(a)sigma_x), c_1=-cosh(sin(a)) and c_1Second(0)=-1. Floating recurrence values remain finite controls, not outward interval enclosures.',
}

def enc(z: complex) -> list[float]:
    return [float(np.real(z)),float(np.imag(z))]

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def product_jets(a: tuple[np.ndarray,np.ndarray,np.ndarray], b: tuple[np.ndarray,np.ndarray,np.ndarray]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    return a[0]@b[0],a[1]@b[0]+a[0]@b[1],a[2]@b[0]+2*a[1]@b[1]+a[0]@b[2]

def exp_jets(b: np.ndarray, db: np.ndarray, ddb: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    # Block exponential yields ordered Duhamel, with second derivative /2.
    n=len(b); block=np.zeros((3*n,3*n),complex)
    for k in range(3):block[k*n:(k+1)*n,k*n:(k+1)*n]=b
    block[:n,n:2*n]=db;block[n:2*n,2*n:]=db;block[:n,2*n:]=ddb/2
    e=expm(block)
    return e[:n,:n],e[:n,n:2*n],2*e[:n,2*n:]

def canonical(jets: tuple[np.ndarray,np.ndarray,np.ndarray], counts: list[int]) -> dict[int,np.ndarray]:
    n=len(jets[0]);eye=np.eye(n,dtype=complex)
    b=(eye,np.zeros_like(eye),np.zeros_like(eye))
    values={0:np.array([1.,0.,0.],dtype=complex)}
    limit=max(counts,default=0)
    for k in range(1,limit+1):
        p=product_jets(jets,b)
        coefficient=np.array([np.trace(v)/n for v in p])
        if k in counts:values[k]=coefficient
        if k<limit:
            denominator=n-k
            for derivative,v in enumerate(p):
                v*=-k/denominator
                v.reshape(-1)[::n+1]+=n*coefficient[derivative]/denominator
            b=p
    return {k:values[k] for k in counts}

def coefficient_eigen(m: np.ndarray,k: int) -> complex:
    c=np.zeros(k+1,complex);c[0]=1
    for lam in np.linalg.eigvals(m):c[1:]+=lam*c[:-1].copy()
    return c[k]/math.comb(len(m),k)

def logarithmic(v: np.ndarray) -> np.ndarray:
    return np.array([np.log(v[0]),v[1]/v[0],v[2]/v[0]-(v[1]/v[0])**2])

def native_h(profile: dict[str,Any],native: dict[str,Any],fields: np.ndarray,links: np.ndarray,theta: np.ndarray,direction: np.ndarray,probe: float=0.,omit: str='') -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    h=np.zeros((256,256),complex);dh=h.copy();ddh=h.copy()
    for x in range(64):h[4*x:4*x+4,4*x:4*x+4]=float(profile['onsite'])*fields[x,0]
    for p in field.native_paths(profile,native):
        if omit=='two' and len(p['edges'])==2:continue
        x,y=int(p['source']),int(p['target']);u=np.eye(4,dtype=complex)
        for edge in p['edges']:u=u@links[edge]
        fx=np.einsum('m,mij->ij',p['coefficients'],fields[x]);fy=np.einsum('m,mij->ij',p['coefficients'],fields[y])
        edges=list(p['edges']);w=float(np.sum(direction[edges]));phase=np.exp(1j*(np.sum(theta[edges])+probe*w))
        hop=p['seam']*(fx@u+u@fy)*phase/2
        for out,v in ((h,hop),(dh,1j*w*hop),(ddh,-w*w*hop)):
            out[4*x:4*x+4,4*y:4*y+4]+=v;out[4*y:4*y+4,4*x:4*x+4]+=v.conj().T
    return h,dh,ddh

def transfer(profile: dict[str,Any],native: dict[str,Any],paths: list[dict[str,np.ndarray]],tau: float,direction: np.ndarray,probe: float=0.,omit: str='') -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    m=(np.eye(256,dtype=complex),np.zeros((256,256),complex),np.zeros((256,256),complex))
    for path in paths:
        h,dh,ddh=native_h(profile,native,path['fields'],path['links'],path['theta'],direction,probe,omit)
        if omit=='contact':ddh*=0
        e=exp_jets(-tau*h,-tau*dh,-tau*ddh)
        phase=np.exp(1j*path['X']) if omit!='HC' else np.ones(64)
        d=np.repeat(phase,4)[:,None]
        e=(d*e[0],d*e[1],d*e[2])
        m=product_jets(e,m)
    return m

def car_lift(h: np.ndarray) -> np.ndarray:
    n=len(h);out=np.zeros((2**n,2**n),complex)
    for mask in range(2**n):
        for x in range(n):
            for y in range(n):
                if not (mask>>y)&1:continue
                sign=(-1)**((mask&((1<<y)-1)).bit_count());new=mask^(1<<y)
                if (new>>x)&1:continue
                sign*=(-1)**((new&((1<<x)-1)).bit_count());out[new|(1<<x),mask]+=sign*h[x,y]
    return out
def canonical_singular_controls() -> dict[str,Any]:
    eye=np.eye(2,dtype=complex);sx=np.array([[0.,1.],[1.,0.]],dtype=complex)
    jets=(-eye,sx,-eye)
    coefficients=canonical(jets,[0,1,2])
    expected={0:np.array([1.,0.,0.]),1:np.array([-1.,0.,-1.]),2:np.array([1.,0.,0.])}
    error=max(float(np.max(abs(coefficients[k]-v))) for k,v in expected.items())
    assert error<1e-13
    # Independent exact-sector CAR heat jets with the same zero-mean HS phase.
    zero=np.zeros((4,4),dtype=complex)
    lifted=exp_jets(zero,-car_lift(sx),zero)
    phase=np.diag([1.,-1.,-1.,1.])
    physical=np.array([np.trace((phase@v)[np.ix_([1,2],[1,2])])/2 for v in lifted])
    car_error=float(np.max(abs(physical-coefficients[1])))
    assert car_error<1e-13
    return dict(scope='Singular I+zM legitimate shared-HS canonical heat/contact boundary, not a pole',monodromy='-exp(-sin(a)sigma_x)',normalized_wedge1_jets=[enc(z) for z in coefficients[1]],fullCAR_normalized_jets=[enc(z) for z in physical],coefficient_error=error,independent_CAR_error=car_error)


def small_controls() -> dict[str,Any]:
    tau=.004;lp=.25
    n0=car_lift(np.diag([1,0,1,0]));n1=car_lift(np.diag([0,1,0,1]))
    paths=[]
    for phase in (.21,-.17):
        h=np.array([[.3,.7*np.exp(1j*phase)],[.7*np.exp(-1j*phase),-.2]])
        dh=np.array([[0,1j*h[0,1]],[-1j*h[1,0],0]]);ddh=np.array([[0,-h[0,1]],[-h[1,0],0]])
        paths.append((h,dh,ddh))
    nodes,weights=roots_hermitenorm(24);weights/=math.sqrt(2*np.pi);sigma=math.sqrt(128*tau*lp)
    cases=[]
    for active_count,ref in ((1,0),(2,-1)):
        masks=[m for m in range(16) if (m&3).bit_count()==active_count and (m>>2).bit_count()==1]
        # Physical source adds active particle and removes Ref together.
        totalcharge=n0+n1-2*np.eye(16)+ref*np.eye(16)
        assert np.max(abs(totalcharge[np.ix_(masks,masks)]))==0
        difference=n0-n1+ref*np.eye(16);hc=64*lp*difference@difference
        def full_step(row: tuple[np.ndarray,np.ndarray,np.ndarray],x: float,omit: str='') -> tuple[np.ndarray,np.ndarray,np.ndarray]:
            h,dh,ddh=row;h4=np.zeros((4,4),complex);dh4=h4.copy();ddh4=h4.copy()
            for k,scale in ((0,1.),(2,.6)):
                h4[k:k+2,k:k+2]=scale*h;dh4[k:k+2,k:k+2]=scale*dh;ddh4[k:k+2,k:k+2]=scale*ddh
            if omit=='contact':ddh4*=0
            e=exp_jets(-tau*car_lift(h4),-tau*car_lift(dh4),-tau*car_lift(ddh4))
            d=expm(1j*x*(difference if omit!='Ref' else n0-n1)) if omit!='HC' else np.eye(16)
            return d@e[0],d@e[1],d@e[2]
        integrated=np.zeros(3,complex);detintegrated=integrated.copy();conditional=0.
        wrong={k:np.zeros(3,complex) for k in (('HC','contact') if ref==0 else ('HC','contact','Ref'))}
        for i,x0 in enumerate(nodes*sigma):
            for j,x1 in enumerate(nodes*sigma):
                weight=weights[i]*weights[j];xs=(x0,x1)
                full=product_jets(full_step(paths[1],x1),full_step(paths[0],x0))
                fv=np.array([np.trace(v[np.ix_(masks,masks)]) for v in full]);integrated+=weight*fv
                logs=np.zeros(3,complex)
                for scale,count in ((1.,active_count),(.6,1)):
                    mon=(np.eye(2,dtype=complex),np.zeros((2,2),complex),np.zeros((2,2),complex))
                    for row,x in zip(paths,xs):
                        e=exp_jets(-tau*scale*row[0],-tau*scale*row[1],-tau*scale*row[2]);d=np.diag(np.exp(1j*x*np.array([1.,-1.])))
                        mon=product_jets((d@e[0],d@e[1],d@e[2]),mon)
                    value=canonical(mon,[count])[count]*math.comb(2,count);logs+=logarithmic(value)
                dv=np.exp(logs[0])*np.array([1,logs[1],logs[2]+logs[1]**2])*np.exp(1j*ref*(x0+x1))
                detintegrated+=weight*dv;conditional=max(conditional,float(np.max(abs(fv-dv))))
                for key in wrong:
                    bad=product_jets(full_step(paths[1],x1,key),full_step(paths[0],x0,key))
                    wrong[key]+=weight*np.array([np.trace(v[np.ix_(masks,masks)]) for v in bad])
        direct=(np.eye(16,dtype=complex),np.zeros((16,16),complex),np.zeros((16,16),complex))
        for row in paths:
            e=full_step(row,0);d=expm(-tau*hc);direct=product_jets((d@e[0],d@e[1],d@e[2]),direct)
        exact_split=np.array([np.trace(v[np.ix_(masks,masks)]) for v in direct])
        error=float(np.max(abs(integrated-exact_split)));deter=float(np.max(abs(integrated-detintegrated)))
        assert error<1e-11 and deter<1e-11 and conditional<1e-11
        bads={key:float(np.max(abs(integrated-v))) for key,v in wrong.items()};assert min(bads.values())>1e-6
        cases.append(dict(active=active_count,aux=1,Ref_shift=ref,totalcharge_exact_zero=True,canonical_masks=masks,conditional_fullCAR_vs_determinant_max=conditional,Hermite_shared_HS_vs_exact_HC_split_max=error,integrated_CAR_vs_determinant_current_contact_max=deter,canonical_trace_jets=[enc(v) for v in integrated],adverse_omit=bads))
    good=2**63;bad=math.comb(126,63);assert bad>good
    hn=np.diag([.4,-.7]);neg=float(np.trace(expm(tau*hn)).real);pos=float(np.trace(expm(-tau*hn)).real);assert abs(neg-pos)>1e-4
    return dict(fullCAR_dimension=16,Hermite_nodes_per_step=24,steps=2,physical_background_and_source=cases,integrated_CAR_vs_determinant_current_contact_max=max(row['integrated_CAR_vs_determinant_current_contact_max'] for row in cases),singular_canonical_boundary=canonical_singular_controls(),coefficient_before63=str(good),wrong_coefficient_after63=str(bad),wrong_after63_ratio=str(bad/good),neutral_negative_sheet_control=dict(correct=neg,wrong_positive=pos),scope='Independent physical background/source fullCAR4mode/Hermite current/contact/Ref/sharedHS control, not N4 ground')

def kinetic_controls(native: dict[str,Any]) -> dict[str,Any]:
    d=native['D'];lp=native['Lplus'];pt=np.eye(192)-d.T@lp@d;k=np.zeros(192);k[0]=1;p=.37*pt@k
    he=64*(np.dot(p+pt@k,p+pt@k)-np.dot(p,p));hc=64*(d@k)@lp@(d@k);total=he+hc
    assert abs(total-64*(1+2*p@k))<1e-10 and abs(total-64)>1
    tau=.07;beta=.23;delta=.41
    image=sum(np.exp(-(delta-2*np.pi*l)**2/(256*tau)+2j*np.pi*beta*l) for l in range(-12,13))/math.sqrt(256*np.pi*tau)
    fourier=sum(np.exp(-64*tau*(n+beta)**2+1j*(n+beta)*delta) for n in range(-20,21))/(2*np.pi)
    wrong=sum(np.exp(-32*tau*(n+beta)**2+1j*(n+beta)*delta) for n in range(-20,21))/(2*np.pi)
    assert abs(image-fourier)<1e-13 and abs(wrong-fourier)>.01
    collective=64*4.; independent=64*2.;assert collective!=independent
    return dict(incoming_transverse_Pi_squared=float(p@p),correct_HE_delta=float(he),HC_delta=float(hc),full_delta=float(total),wrong_zero_incoming=64.,kinetic_first=128*float(p@pt@k),kinetic_contact=128*float(k@pt@k),heatkernel_images=enc(image),heatkernel_Fourier=enc(fourier),wrong_diffusion64=enc(wrong),heatkernel_image_error=float(abs(image-fourier)),collective_two_identical_flows=collective,wrong_independent_copies=independent,scope='ActualN4 fullE HE/HC identity and incomingPi; exact-shape1D cycle kernel numerical phase/diffusion control, not129D integration')

def run(output: Path) -> dict[str,Any]:
    start=time.monotonic(); captured={Path(__file__).name:Path(__file__).read_bytes(),Path(fiber.__file__).name:Path(fiber.__file__).read_bytes(),**field.CAPTURED}
    names=['induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json']
    inputs={name:(ROOT/name).read_bytes() for name in names}
    for name,raw in inputs.items():assert sha(raw)==fiber.PINNED[name]
    record=json.loads(inputs[names[0]]);pars=record['actual_stationary_root']['parameters'];g=[float((Fraction(pars[k]['lower'])+Fraction(pars[k]['upper']))/2) for k in ('g0','gs')]
    data=field.load_mean(g_override=[g[0],g[1],g[1],g[1]],mean_descriptor=json.loads(inputs[names[2]])['input_descriptor'])
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.parent==ROOT:captured.setdefault(path.name,path.read_bytes())
    native=data['native']; rng=np.random.default_rng(2221010);n=64;paths=[];tau=.0005
    # Unfrozen, non-Clifford Cartesian endpoints and noncommuting ordered SU4.
    generators=data['geom']['M']/np.sqrt(8);c,chords=fiber.cycles(np.rint(native['D']).astype(int));q=np.linalg.qr(c.astype(float))[0]
    vals,vec=np.linalg.eigh(native['Lplus']);sqrtlp=(vec*np.sqrt(np.maximum(vals,0)))@vec.T
    direction=q[:,7]; path_meta=[]
    for j in range(3):
        fs=native['frame']
        fields=np.broadcast_to(fs,(64,4,4,4)).copy()
        coeff=rng.normal(size=(64,4,15))*.007*(j+1);fields+=np.einsum('xma,aij->xmij',coeff,generators)
        colour=rng.normal(size=(192,15))*.013*(j+1);links=np.array([expm(1j*np.einsum('a,aij->ij',v,generators)) for v in colour])
        theta=q@(rng.normal(size=129)*.09*(j+1));x=math.sqrt(128*tau)*sqrtlp@rng.normal(size=64)
        # Project numerical sqrt roundoff out of the forbidden constant HS mode.
        x-=x.mean()
        assert abs(x.sum())<1e-14
        paths.append(dict(fields=fields,links=links,theta=theta,X=x));path_meta.append(dict(X_sum=float(x.sum()),Theta_norm=float(np.linalg.norm(theta)),Phi_nonuniform=float(np.linalg.norm(fields-fields[0])),colour_noncommuting=float(np.linalg.norm(links[0]@links[1]-links[1]@links[0]))))
    np.savez_compressed(output.with_suffix('.npz'),direction=direction,C=c,**{f'{j}_{k}':v for j,p in enumerate(paths) for k,v in p.items()})
    pathbytes=output.with_suffix('.npz').read_bytes();rows=[];alllogs=[];roles=data['profiles'][1:]
    for profile in roles:
        m=transfer(profile,native,paths,tau,direction);cv=canonical(m,[128,129]);v=cv[128];log=logarithmic(v);alllogs.append(log)
        eigen=coefficient_eigen(m[0],128);err=abs(eigen-v[0]);assert err<1e-9
        eps=.002;vs=[]
        for a in (-eps,eps):
            mm=transfer(profile,native,paths,tau,direction,a);vs.append(coefficient_eigen(mm[0],128))
        fd1=(vs[1]-vs[0])/(2*eps);fd2=(vs[1]+vs[0]-2*v[0])/eps**2
        assert abs(fd1-v[1])<1e-7 and abs(fd2-v[2])<1e-7
        bads={}
        for omit in ('contact','HC','two'):
            bm=transfer(profile,native,paths,tau,direction,omit=omit);bv=canonical(bm,[128])[128]
            bads[omit]=dict(coefficient_difference=float(abs(bv[0]-v[0])),second_difference=float(abs(bv[2]-v[2])),log_Hessian_difference=float(abs(logarithmic(bv)[2]-log[2])))
        assert bads['contact']['second_difference']>1e-6 and bads['HC']['coefficient_difference']>1e-3 and bads['two']['coefficient_difference']>1e-6
        # independent primitive endpoint assembly at zeroU1 phase
        p=paths[0];h0=native_h(profile,native,p['fields'],p['links'],np.zeros(192),direction)[0]
        assembly=float(np.max(abs(h0-field.assemble(profile,native,p['fields'],p['links']))));assert assembly<1e-11
        pathcount=field.native_paths(profile,native)
        rows.append(dict(role=profile['role'],copies=profile['load'],onebody_dimension=256,canonical128_normalized_jets=[enc(z) for z in v],canonical129_normalized_jets=[enc(z) for z in cv[129]],log128_jets=[enc(z) for z in log],coefficient_recurrence_vs_eigen=err,finite_difference_first_error=float(abs(fd1-v[1])),finite_difference_second_error=float(abs(fd2-v[2])),primitive_endpoint_assembly_error=assembly,ordered_paths=len(pathcount),two_link_paths=sum(len(p['edges'])==2 for p in pathcount),AP_seam_paths=sum(p['seam']<0 for p in pathcount),adverse=bads))
    combined=alllogs[0]+63*alllogs[1];refphase=np.exp(-1j*sum(p['X'][0] for p in paths));assert abs(refphase-1)>1e-3
    result=dict(kind='EXACT_CANONICAL_FULLLAW_DYNAMIC_FUNCTIONAL_WITH_FINITE_NATIVE_INTEGRAND_CONTROLS',proof=PROOF,law=dict(N=4,V=64,charged_copies=64,aux_copies=63,canonical_each=128,neutral_scale='-2^8192',bosonic_stiffness='2^10240',chi='2^-4096',HE=64,HC=64),finite_path=dict(steps=3,tau=tau,total_thermal_length=3*tau,seed=2221010,shared_HS_covariance='128tau Lplus',realized_paths=path_meta,canonical_species=rows,active_plus63aux_conditioned_log_jets=[enc(z) for z in combined],source_Ref3_phase=enc(refphase),wrong_Ref_omitted_phase=[1,0],source_phase_difference=float(abs(refphase-1)),neutral128='[z^128]det(I+z ordered_product exp(+tau 2^8192 hneutral(Phi,U))) retained exact symbolic; never evaluated with charged tau',not_ground=True,not_integrated_Gauss_or_boson=True),independent_small_CAR_Hermite=small_controls(),Pi_HC_cycle_controls=kinetic_controls(native),input_sha256={k:sha(v) for k,v in inputs.items()},path_bytes_sha256=sha(pathbytes),source_sha256={k:sha(v) for k,v in captured.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items());assert all((ROOT/name).read_bytes()==raw for name,raw in inputs.items())
    assert output.with_suffix('.npz').read_bytes()==pathbytes
    result['source_inputs_unchanged']=True
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in captured.items():z.writestr('executed_source/'+name,raw)
        for name,raw in inputs.items():z.writestr('input/'+name,raw)
        z.writestr('input/nonconstant_paths.npz',pathbytes)
    output.write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):r=run(args.output)
    print(json.dumps(dict(kind=r['kind'],elapsed_seconds=r['elapsed_seconds'],native_roles=len(r['finite_path']['canonical_species']),small_CAR_error=r['independent_small_CAR_Hermite']['integrated_CAR_vs_determinant_current_contact_max'],source_inputs_unchanged=r['source_inputs_unchanged']),indent=2))
