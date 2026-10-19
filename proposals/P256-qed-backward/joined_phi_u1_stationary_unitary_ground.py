"""Exactly normalized all-native Slater-unitary trial; no ground/source adoption."""
from __future__ import annotations
import argparse
import ast
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile
import numpy as np
import sympy as sy
from scipy.linalg import eigh, expm, expm_frechet
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parent
SOURCE_AT_IMPORT:dict[str,bytes]={Path(__file__).name:Path(__file__).read_bytes()}

class CapturedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname:str)->Any:
        raw=Path(self.path).read_bytes();SOURCE_AT_IMPORT[Path(self.path).name]=raw
        return self.source_to_code(raw,self.path)

class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname:str,path:Any=None,target:Any=None)->Any:
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.origin is not None:
            source=Path(spec.origin)
            if source.parent==ROOT and source.suffix=='.py':spec.loader=CapturedLoader(fullname,str(source))
        return spec

finder=CapturedFinder();sys.meta_path.insert(0,finder)
import induced_phi_nonuniform_field_car as field
import joined_phi_u1_stationary_full_phi_fiber as fiber
sys.meta_path.remove(finder)

PREFIX = 'joined_phi_u1_stationary_unitary_ground'
ETA = F(1,128)
sys.set_int_max_str_digits(0)
PROOF = {
 'law': 'Uniform Clifford Phi_mu=sqrt(g_mu)Gamma_mu, SU4 links I; actual endpoint averages, AP seams, all one/two paths. Charged h=h_ci+Gamma0 A. Primitive ci is imaginary one-link only; reverse is its adjoint. Neutral retains physical -Lambda sign but its reference is unchanged. One shared origin Ref4 spectator, rank128 bulk in EACH of64 charged copies. Total neutral128+charged8192+Ref4=8324. HE64+HC64 retained, no Maxwell term.',
 'state': 'S=Gamma0 h_ci satisfies Sdag=-S and {S,Gamma0}=0. u=exp(-eta S), W=u Wminus, P=u Pminus udag=.5(I-exp(-2eta S)Gamma0). The wedge of the 128 orthonormal W columns has norm exactly1 at every Theta. Product of active1 and auxiliary63 copies remains exactly1 with constant normalized Haar Theta amplitude. Analytic periodic native integer characters give untruncated Fourier and all allowed PH orders, not a finite PH/Fourier approximation.',
 'density': 'S is Clifford-even. Any block of exp(-2eta S) is Clifford-even; trace_spin(even Gamma0)=0 in the faithful Cl4 representation. Thus Tr_spin P_xx=2 pointwise for every x and every Theta, not a statement about generic Slaters. Whole Ref retained: rho_x=sum64 n_s,x-128+(nRef-4)delta_x0. Shared Ref4 is fixed, so centered total rho mean0. Exact neutral occupancy is unchanged.',
 'berry': 'For derivative or full-E commutator, R=u^-1[E,u]=-eta integral_0^1 exp(t eta S)[E,S]exp(-t eta S)dt. R is Clifford-even. Tr(Gamma0 R)=0. Tr R=-eta Tr[E,S]=0: derivatives of Tr S=0 and finite matrix commutators have zero trace. Hence Tr(Pminus R)=0 exactly. Ordinary Berry uses E=-i derivative and has the same proof. No pointwise matrix sparsity assumption is used.',
 'electric': 'E=Q Pi+D^T Lplus rho, Q Q^T=I-D^T Lplus D, HE+HC=64 sum_192 E_e^2 on compatible Gauss sector. [E_e,S]_xy=(native k_e) S_xy because -iQ derivative plus longitudinal density commutator restores the FULL integer path incidence. E annihilates the onsite Ref4 reference. Duhamel and unitary HS invariance give sum_e ||R_e||HS^2<=eta^2 sum_e ||[E_e,S]||HS^2=eta^2 Tr h_ci^2, since actual ci paths have length1 and distinct unordered endpoints at N4. Slater variance is ||(1-Pminus)R Pminus||HS^2<=.5||R||HS^2 for Hermitian R. Exact zero mean implies product-copy cross terms vanish; therefore delta(HE+HC)<=64 eta^2 sigma_total^2, sigma_s^2=.5Tr h_ci,s^2. This is a trial-state Wick identity only.',
 'scalar': 'Let B_i=Gamma0 Gamma_i. The spin trace of every odd word in the three B_i vanishes (reduces to one B_i or B1B2B3, both traceless); even words need not be traceless. Site parity (-1)^(x0+x1+x2) reverses S; nearest A reverses site parity. In Tr A exp(-2eta S), even powers vanish by site parity and odd powers by spin trace, proving nearest c0 contribution zero without deleting it. Onsite and two-link A_even remain priced: ||A_even||<=C0=sqrt(g0)(|onsite|+2sum_length2|c0|). Their delta is .5Tr A_even(I-cos(2eta h_ci)) <= C0 eta^2 Tr h_ci^2=2C0 eta^2 sigma^2.',
 'odd': 'h_ci contribution equals -.5Tr[h_ci sin(2eta h_ci)]. For every real eigenvalue x, x sin(2eta x)>=2eta x^2-(4/3)eta^3 x^4; remainder follows |sin y-y|<=|y|^3/6. With ||h_ci||<=L=2sqrt(gs)sum_length1|ci|, delta odd<=-2eta sigma^2+(4/3)eta^3 L^2 sigma^2. No float eigenvalue enters the bound.',
 'scope': 'Root-independent normalized smooth beta0/alpha0 native trial exists at any positive Clifford g. Native integer charged characters preserve alpha0 and exact cycle-lattice periodicity gives beta0. NOT a beta minimizer, stationary eigenvector or ground characterization. Conditional FullPhi composition below uses actual uniform-image T, not a globally bounded R on arbitrary fullfield vectors. Ground photon pole/common cone remain open; no source or physical eta/clock adoption.',
 'conditional_T': 'Actual compact-source action PROOF.T/uniformity maps ANY unit bodyTheta/fixed-particle fibre and bounds R=Hfull T-T(Hbody+Ephi) uniformly EVERY realTheta. Its N8324/sourceactive129 envelope covers this background neutral128+active128+aux8064+sharedRef4=8324; commonZ4 acts trivially. Its Theta-independent real Kato/Haar/neutral lift commutes HE; charged nonlinear and HC commutators are already priced uniformly. New body wave is analytic on compact beta0 torus, hence all finite Theta derivatives L2 and D(Hbody^2). Actual b6 normal packet has zero boundary jets0..5 and Kato/Haar coefficients smooth on collar. The named global Friedrichs H4/zero-gluing proof therefore applies with compact periodic Theta in place of Schwartz Theta (no endpoint boundary terms). Ttrial and Treference are normalized and in D(Hfull^2). For each, |<THfullT>-<Hbody>-Ephi|<=epsilon, giving DeltaEfull<=DeltaEbody+2epsilon. Existing root/gap/SPD/chart/normalization-law suppliers remain conditional; existence of a body trial itself does not require them.'
}

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_clifford(data: dict[str,Any]) -> dict[str,Any]:
    gam=[]
    for f in data['native']['frame']:
        z=f/np.sqrt(np.trace(f@f).real/4)
        r=np.rint(z.real)+1j*np.rint(z.imag)
        assert np.max(abs(r-z))<1e-12
        gam.append(sy.Matrix([[sy.Integer(int(v.real))+sy.I*int(v.imag) for v in row] for row in r]))
    I=sy.eye(4)
    for i in range(4):
        for j in range(4):assert gam[i]*gam[j]+gam[j]*gam[i]==2*int(i==j)*I
    traces=[]
    for mask in range(16):
        m=I
        for j in range(4):
            if mask>>j&1:m=m*gam[j]
        if mask.bit_count()%2==0:
            assert sy.trace(m*gam[0])==0
            traces.append([mask,str(sy.trace(m*gam[0]))])
    B=[gam[0]*g for g in gam[1:]]
    odd=[str(sy.trace(b)) for b in B]+[str(sy.trace(B[0]*B[1]*B[2]))]
    assert odd==['0']*4
    return dict(exact_Cl4_anticommutators=True,even_times_Gamma0_trace=traces,odd_B_reduced_word_traces=odd)


def interval_energy(data:dict[str,Any],pars:dict[str,Any]) -> dict[str,Any]:
    ar=field.full.scalar.fast;ar.iv.dps=110;ex=ar.exact
    g0,gs=[ar.enclosure(F(pars[k]['lower']),F(pars[k]['upper'])) for k in ('g0','gs')]
    total=ex(0);sigmaall=ex(0);rows=[]
    for p in data['profiles'][1:]:
        one=[v for v in p['paths'] if v['length']==1]
        assert all(F(v['ci'])==0 for v in p['paths'] if v['length']==2)
        paths=field.native_paths(p,data['native']);seen=set();sigcoef=F(0)
        for path in paths:
            row=next(v for v in p['paths'] if v['axis']==path['axis'] and v['length']==path['length'])
            if F(row['ci']):
                key=tuple(sorted((int(path['source']),int(path['target']))))
                assert key not in seen;seen.add(key)
                assert len(path['edges'])==1
                sigcoef+=4*F(row['ci'])**2
        assert len(seen)==192
        sigma=ex(sigcoef)*gs
        C0=ar.iv.sqrt(g0)*ex(abs(F(p['onsite']))+2*sum((abs(F(v['c0'])) for v in p['paths'] if v['length']==2),F(0)))
        L2=gs*ex((2*sum((abs(F(v['ci'])) for v in one),F(0)))**2)
        bracket=-2*ex(ETA)+(64+2*C0)*ex(ETA**2)+ex(F(4,3)*ETA**3)*L2
        delta=sigma*bracket;total+=p['load']*delta;sigmaall+=p['load']*sigma
        rows.append(dict(role=p['role'],load=p['load'],sigma2_coefficient_times_gs=str(sigcoef),sigma2=ar.bounds(sigma),C0=ar.bounds(C0),L2=ar.bounds(L2),delta_upper=ar.bounds(delta),all384_forward_one_two_paths=len(paths),ci_distinct_pairs=len(seen)))
    assert ar.endpoint(total,1)<0
    return dict(arithmetic='110 decimal-digit directed interval arithmetic; exact rational binary primitive coefficients and saved rational root endpoints',eta_exact=str(ETA),species=rows,sigma2_total=ar.bounds(sigmaall),delta_energy=ar.bounds(total),strict_negative_upper=True)


def matrices(data:dict[str,Any],p:dict[str,Any],theta:np.ndarray,wind:np.ndarray|None=None) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    h=np.zeros((256,256),complex);ci=np.zeros_like(h);near=np.zeros_like(h);two=np.zeros_like(h);dh=np.zeros_like(h)
    frame=data['native']['frame'];G0=frame[0]/np.sqrt(np.trace(frame[0]@frame[0]).real/4)
    for x in range(64):h[4*x:4*x+4,4*x:4*x+4]=float(p['onsite'])*frame[0]
    for path in field.native_paths(p,data['native']):
        x,y=path['source'],path['target'];a,b=slice(4*x,4*x+4),slice(4*y,4*y+4)
        row=next(v for v in p['paths'] if v['axis']==path['axis'] and v['length']==path['length'])
        phase=path['seam']*np.exp(1j*sum(theta[e] for e in path['edges']))
        block=phase*np.einsum('m,mij->ij',path['coefficients'],frame)
        odd=phase*1j*float(row['ci'])*frame[path['axis']+1]
        scalar=phase*float(row['c0'])*frame[0]
        h[a,b]+=block;h[b,a]+=block.conj().T;ci[a,b]+=odd;ci[b,a]+=odd.conj().T
        target=near if path['length']==1 else two
        target[a,b]+=scalar;target[b,a]+=scalar.conj().T
        if wind is not None:
            d=1j*sum(wind[e] for e in path['edges'])*block
            dh[a,b]+=d;dh[b,a]+=d.conj().T
    return h,ci,near,two,dh


def finite_case(data:dict[str,Any],theta:np.ndarray) -> dict[str,Any]:
    frame=data['native']['frame'];g0=np.kron(np.eye(64),frame[0]/np.sqrt(np.trace(frame[0]@frame[0]).real/4))
    P0=(np.eye(256)-g0)/2;val,W=eigh(P0);W=W[:,val>.5]
    rng=np.random.default_rng(222101064);wind=data['native']['Q']@rng.normal(size=129);wind/=np.linalg.norm(wind)
    rows=[];electric=None
    for p in data['profiles'][1:]:
        h,ci,near,two,dh=matrices(data,p,theta,wind)
        S=g0@ci;u=expm(-float(ETA)*S);P=u@P0@u.conj().T
        density=np.trace(P.reshape(64,4,64,4)[np.arange(64),:,np.arange(64),:],axis1=1,axis2=2).real
        _,cip,_,_,_=matrices(data,p,theta+1e-5*wind)
        _,cim,_,_,_=matrices(data,p,theta-1e-5*wind)
        Sd=g0@(cip-cim)/2e-5
        ud:Any=expm_frechet(-float(ETA)*S,-float(ETA)*Sd,compute_expm=False)
        berry=complex(-1j*np.trace(P0@u.conj().T@ud))
        numerical_ud=(expm(-float(ETA)*g0@cip)-expm(-float(ETA)*g0@cim))/2e-5
        current=float(np.trace(P@dh).real)
        hplus=matrices(data,p,theta+1e-5*wind)[0];hminus=matrices(data,p,theta-1e-5*wind)[0]
        currentfd=float(np.trace(P@(hplus-hminus)).real/2e-5)
        if electric is None:
            kval,V=eigh(-1j*S);difference=kval[:,None]-kval[None,:]
            factor=np.ones_like(difference,dtype=complex);nz=abs(difference)>1e-11
            factor[nz]=np.expm1(1j*float(ETA)*difference[nz])/(1j*float(ETA)*difference[nz])
            fullvar=0.;maxmean=0.;allr=np.zeros((192,256,256),complex)
            for path in field.native_paths(p,data['native']):
                if path['length']!=1:continue
                x,y=path['source'],path['target'];a,b=slice(4*x,4*x+4),slice(4*y,4*y+4)
                B=np.zeros_like(S);B[a,b]=S[a,b];B[b,a]=-S[b,a]
                R=-float(ETA)*V@((V.conj().T@B@V)*factor)@V.conj().T
                mean=np.trace(P0@R);maxmean=max(maxmean,abs(mean))
                fullvar+=float(np.trace(P0@R@(np.eye(256)-P0)@R).real)
                allr[path['edges'][0]]=R
            arr=allr;Q=data['native']['Q'];rt=np.einsum('ea,eij->aij',Q,arr,optimize=True)
            hevar=sum(float(np.trace(P0@r@(np.eye(256)-P0)@r).real) for r in rt)
            blocks=P.reshape(64,4,64,4).transpose(0,2,1,3)
            cov=np.diag(density)-np.einsum('xyij,yxji->xy',blocks,blocks).real
            hcvar=float(np.sum(data['native']['Lplus']*cov))
            sigma=float(np.trace(ci@ci).real/2)
            electric=dict(full_native_variance_per_copy=fullvar,HE_variance_per_copy=hevar,HC_variance_per_copy=hcvar,Gauss_fullE_identity_defect=abs(fullvar-hevar-hcvar),Berry_max_all192= maxmean,analytic_variance_cap=float(ETA)**2*sigma,all129_transverse_directions=len(rt),all192_fullE_directions=len(arr),allcopy_HEHC_energy=64*64*fullvar,omitted_HC_energy_defect=64*64*hcvar,omitted63copies_energy_defect=63*64*fullvar)
            assert fullvar<=float(ETA)**2*sigma+1e-8 and abs(fullvar-hevar-hcvar)<1e-8 and maxmean<1e-10
            transverse=Q@Q[0]
            plus=matrices(data,p,theta+1e-5*transverse)[1]
            minus=matrices(data,p,theta-1e-5*transverse)[1]
            derivative=expm_frechet(-float(ETA)*S,-float(ETA)*g0@(plus-minus)/2e-5,compute_expm=False)
            density_generator=np.diag(np.repeat((data['native']['D'].T@data['native']['Lplus'])[0],4))
            direct=-1j*u.conj().T@derivative+u.conj().T@density_generator@u-density_generator
            electric['independent_fullE_Duhamel_defect']=float(np.max(abs(direct-arr[0])))
            electric['omitted_longitudinal_fullE_operator_defect']=float(np.linalg.norm(u.conj().T@density_generator@u-density_generator))
            assert electric['independent_fullE_Duhamel_defect']<1e-8
            assert electric['omitted_longitudinal_fullE_operator_defect']>.01
        delta=float(np.trace(h@(P-P0)).real)
        row=dict(role=p['role'],load=p['load'],unitarity_defect=float(np.max(abs(u.conj().T@u-np.eye(256)))),projector_defect=float(np.max(abs(P@P-P))),norm_Gram_determinant=float(np.linalg.det((u@W).conj().T@(u@W)).real),density2_defect=float(max(abs(density-2))),Berry_direction_abs=abs(berry),Frechet_derivative_defect=float(np.max(abs(ud-numerical_ud))),physical_current_J_minus_dH=-current,current_fd_defect=abs(current-currentfd),odd_energy=float(np.trace(ci@(P-P0)).real),nearest_c0_energy=float(np.trace(near@(P-P0)).real),two_link_delta_energy=float(np.trace(two@(P-P0)).real),matter_delta_energy=delta)
        two_derivative=(matrices(data,p,theta+1e-5*wind)[3]-matrices(data,p,theta-1e-5*wind)[3])/2e-5
        row['two_link_physical_current_J_minus_dH']=-float(np.trace(P@two_derivative).real)
        # Wrong incoming derivative uses the same sign as forward instead of its adjoint.
        wrongdh=np.zeros_like(h)
        for path in field.native_paths(p,data['native']):
            x,y=path['source'],path['target'];a,b=slice(4*x,4*x+4),slice(4*y,4*y+4)
            block=path['seam']*np.exp(1j*sum(theta[e] for e in path['edges']))*np.einsum('m,mij->ij',path['coefficients'],frame)
            dv=1j*sum(wind[e] for e in path['edges'])*block
            wrongdh[a,b]+=dv;wrongdh[b,a]-=dv.conj().T
        row['wrong_incoming_normal_law_defect']=float(np.linalg.norm(wrongdh-dh))
        assert row['wrong_incoming_normal_law_defect']>1
        # Independent finite full literal endpoint/AP implementation.
        literal=fiber.literal_operator(data,p,np.broadcast_to(frame,(64,4,4,4)),np.broadcast_to(np.eye(4),(192,4,4)),theta)
        row['actual_endpoint_AP_defect']=float(np.max(abs(h-literal)))
        row['two_link_operator_omission_defect']=float(np.linalg.norm(two))
        for k in ('unitarity_defect','projector_defect','density2_defect','Berry_direction_abs','current_fd_defect','actual_endpoint_AP_defect'):
            assert row[k]<1e-8,(k,row[k])
        assert abs(row['nearest_c0_energy'])<1e-8
        if np.linalg.norm(theta)>1e-10:
            assert row['two_link_operator_omission_defect']>1
        else:
            assert row['two_link_operator_omission_defect']<1e-10
            row['two_link_zero_Theta_cancellation']='N4 AP reverse length2 pair cancellation; not permission to omit the law at nonconstant Theta'
        rows.append(row)
    assert electric is not None
    # Same Slater P but Theta-dependent many-body phase: cross terms DO NOT vanish.
    phase_gradient=float(wind[0]);n=64
    wrongud=ud+1j*phase_gradient*u/128
    wrongberry=complex(-1j*np.trace(P0@u.conj().T@wrongud))
    assert abs(wrongberry.real-phase_gradient)<1e-10 and abs(phase_gradient)>1e-5
    meanvec=data['native']['Q']@data['native']['Q'][0]
    square=float(meanvec@meanvec)
    wrong=dict(phase='exp(i theta_edge0) per copy, an integer native character periodic on cycle lattice; orbital lift exp(i theta_edge0/128) represents exactly this wedge phase locally',native_edge=0,observed_directional_Berry=wrongberry.real,predicted_directional_Berry=phase_gradient,correct_64copy_HE_phase_energy=64*n*n*square,wrong_independent_HE_phase_energy=64*n*square,missing_cross_HE_energy=64*n*(n-1)*square,generic_independence_rejected=True,unchanged_density_and_Slater_projector=True)
    green=fiber.rational_green(data['native']['sites']);delta=np.rint(data['native']['D'][:,0]).astype(int)
    exact_square=1-sum((F(int(delta[x])*int(delta[y]))*green[x][y] for x in np.flatnonzero(delta) for y in np.flatnonzero(delta)),F(0))
    assert abs(float(exact_square)-square)<1e-12
    wrong['native_phase_mean_norm_squared_exact']=str(exact_square)
    wrong['missing_cross_HE_energy_exact']=str(64*n*(n-1)*exact_square)
    neutral=data['profiles'][0]
    uniform_fields=np.broadcast_to(frame,(64,4,4,4));identity_links=np.broadcast_to(np.eye(4),(192,4,4))
    hn=fiber.literal_operator(data,neutral,uniform_fields,identity_links,theta)
    hn2=fiber.literal_operator(data,neutral,uniform_fields,identity_links,theta+.31)
    neutralcontrol=dict(role=neutral['role'],literal_native_endpoint_AP_defect=float(np.max(abs(hn-field.assemble(neutral,data['native'],uniform_fields,identity_links)))),neutral_U1_independence_defect=float(np.max(abs(hn-hn2))),physical_scale='-2^8192 hneutral; dimensionless matrix compared before exact huge scaling',neutral_reference_unchanged=True,delta_energy_exact='0')
    assert neutralcontrol['literal_native_endpoint_AP_defect']<1e-10 and neutralcontrol['neutral_U1_independence_defect']==0
    return dict(scope='Finite float64 actual N4 controls, not interval proof or Theta quadrature',theta_native_norm=float(np.linalg.norm(theta)),rows=rows,neutral=neutralcontrol,electric=electric,wrong_nonzero_Berry=wrong,allcopy_matter_delta=sum(r['load']*r['matter_delta_energy'] for r in rows),allcopy_pointwise_energy_delta=sum(r['load']*r['matter_delta_energy'] for r in rows)+electric['allcopy_HEHC_energy'])


def run(output:Path) -> dict[str,Any]:
    started=time.monotonic();captured={}
    for name in ('induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'):
        raw=(ROOT/name).read_bytes();assert sha(raw)==fiber.PINNED[name];captured[name]=raw
    root=json.loads(captured['induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json'])['actual_stationary_root']
    saved=json.loads(captured['induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'])['input_descriptor']['exact_NEW_allq_BO_mean']
    centres=[float(F(v)) for v in saved['centre_exact_rational']]
    data=field.load_mean(g_override=[centres[0],centres[1],centres[1],centres[1]],mean_descriptor=saved)
    assert [p['load'] for p in data['profiles']]==[1,1,63]
    assert data['profiles'][1]['paths']==data['profiles'][2]['paths']
    clifford=exact_clifford(data)
    intervals=dict(actual_saved_stationary_root=interval_energy(data,root['parameters']),saved_NEW_allq_BO_descriptor=interval_energy(data,saved['parameters']))
    def take(name:str,pin:str|None=None)->dict[str,Any]:
        raw=(ROOT/name).read_bytes()
        if pin is not None:assert sha(raw)==pin
        captured[name]=raw
        return json.loads(raw)
    actionname='joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json'
    action=take(actionname,fiber.PINNED[actionname])
    assert action['action_input']['uniform_body_Hilbert_map']
    assert action['source_weights']['fixed_total_particle_number']==8324
    price=fiber.uniform_price(action,take)
    domainname='induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json'
    domain=take(domainname,fiber.PINNED[domainname])
    assert action['domain_receipt_sha256']==sha(captured[domainname])
    ar=field.full.scalar.fast
    body=intervals['saved_NEW_allq_BO_descriptor']['delta_energy']
    eps=ar.exact(F(price['epsilon']['upper']))
    full_delta=ar.enclosure(F(body['lower']),F(body['upper']))+2*eps
    assert ar.endpoint(full_delta,1)<-10000
    conditional=dict(scope='CONDITIONAL actual FullPhi T image, NOT global fullfield R or adopted source',operator_domain_premise=PROOF['conditional_T'],epsilon=price['epsilon'],matching_particle_number=8324,domain_receipt_sha256=sha(captured[domainname]),DeltaEfull_upper=ar.bounds(full_delta),price_components=price['components'])
    D=data['native']['D'];Q=data['native']['Q'];lp=data['native']['Lplus']
    assert np.max(abs(Q@Q.T+D.T@lp@D-np.eye(192)))<1e-12
    C,chords=fiber.cycles(np.rint(D).astype(np.int64))
    rng=np.random.default_rng(2221010128);theta=C@(.13*rng.normal(size=129))
    with threadpool_limits(limits=1):controls=[finite_case(data,np.zeros(192)),finite_case(data,theta)]
    sources={}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).parent==ROOT and Path(name).suffix=='.py':sources[Path(name).name]=Path(name).read_bytes()
    for name,raw in field.CAPTURED.items():
        if name in sources:assert sources[name]==raw
        sources[name]=raw
    for name,raw in SOURCE_AT_IMPORT.items():
        if name in sources:assert sources[name]==raw
        sources[name]=raw
    # Retain complete imported science bytes including modules captured by prior owners.
    for table in (fiber.preparation.SOURCE_BYTES,fiber.gauss.SOURCE_BYTES):
        for name,raw in table.items():
            if name in sources:assert sources[name]==raw
            sources[name]=raw
    sources[Path(__file__).name]=Path(__file__).read_bytes()
    for name in ('joined_phi_u1_actual_compact_source_action_certificate.py','induced_phi_compact_source_global_domain_control.py'):
        sources[name]=(ROOT/name).read_bytes()
    audits={}
    for name,keys in {'induced_phi_nonuniform_field_car.py':['native_paths','assemble'],'induced_phi_quantum_fast_transfer.py':['primitive_profiles'],'joined_phi_u1_stationary_unitary_ground.py':['interval_energy','finite_case']}.items():
        tree=ast.parse(sources[name]);audits[name]=[]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in keys:
                segment=ast.get_source_segment(sources[name].decode(),node);assert segment is not None
                audits[name].append(dict(symbol=node.name,sha256=sha(segment.encode()),line=node.lineno))
        assert len(audits[name])==len(keys)
    result:dict[str,Any]
    result=dict(kind='NEW_EXACT_UNTRUNCATED_ALLNATIVE_UNITARY_SLATER_BODY_AND_CONDITIONAL_FULLPHI_VARIATIONAL_TRIAL',proof=PROOF,exact_Clifford=clifford,interval_energy=intervals,conditional_FullPhi=conditional,finite_controls=controls,root_schema='.actual_stationary_root.parameters',finite_frame_schema='saved descriptor exact_NEW_allq_BO_mean centre_exact_rational',root_matching='Separate interval results for original saved stationary root and NEW allq BO saved descriptor. Finite controls use ONLY latter; no root solver or mixed-box arithmetic.',native_geometry=dict(V=64,edges=192,Theta=129,charged_species=64,occupancy_per_copy=128,Ref='one shared origin Ref4 spectator, whole Ref charge retained',fixed_total_particles=8324,conserved_beta='0',conserved_alpha='0',integer_cycle_basis_rank=129),source_sha256={n:sha(b) for n,b in sources.items()},input_sha256={n:sha(b) for n,b in captured.items()},AST_audit=audits,unpaid=['Identification of stationary fullPhi ground, beta minimizing coset and same-ground photon dispersion/pole/common cone','Root/gap/SPD/chart and exact source-law enclosures are inherited conditional inputs, not re-solved or proved here'])
    assert all((ROOT/n).read_bytes()==b for n,b in {**sources,**captured}.items())
    archive=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n,b in {**sources,**captured}.items():z.writestr(n,b)
    result['immutable_source_input_capture']=dict(path=archive.name,sha256=sha(archive.read_bytes()));result['source_inputs_unchanged']=True;result['elapsed_seconds']=time.monotonic()-started
    output.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();r=run(args.output)
    print(json.dumps(dict(receipt=args.output.name,receipt_sha256=sha(args.output.read_bytes()),interval_energy={k:v['delta_energy']['upper_decimal'] for k,v in r['interval_energy'].items()},finite_energy=[v['allcopy_pointwise_energy_delta'] for v in r['finite_controls']],elapsed_seconds=r['elapsed_seconds']),indent=2))
