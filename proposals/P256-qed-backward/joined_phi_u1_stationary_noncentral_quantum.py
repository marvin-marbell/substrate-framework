"""Full physical form-domain noncentral competitor, not a characterized vacuum.

The normalized variational section is defined by an actual implicit integral.
Exact energy budgets and finite projector controls are deliberately separate.
"""
from __future__ import annotations
import argparse
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
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
CAPTURED:dict[str,bytes]={Path(__file__).name:Path(__file__).read_bytes()}

class CaptureLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname:str)->Any:
        raw=Path(self.path).read_bytes();CAPTURED[Path(self.path).name]=raw
        return self.source_to_code(raw,self.path)

class CaptureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname:str,path:Any=None,target:Any=None)->Any:
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.origin is not None:
            source=Path(spec.origin)
            if source.parent==ROOT and source.suffix=='.py':
                spec.loader=CaptureLoader(fullname,str(source))
        return spec

finder=CaptureFinder();sys.meta_path.insert(0,finder)
import induced_phi_nonuniform_field_car as field
import joined_phi_u1_stationary_full_phi_fiber as fiber
sys.meta_path.remove(finder)

PREFIX='joined_phi_u1_stationary_noncentral_quantum'
PINS={
 'joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json':'1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94',
 'joined_phi_u1_stationary_full_phi_fiber_borel_checked_receipt_20261010.json':'4a024dde3c50046596518a65d8150e30cdf3253fd7b120269bf8f6cad6fff142',
}
PLAN=dict(id='issue-222-induced-photon-delta13-continuation-20261008',revision='6079658737e4b1fcb93e8ac99b7e10e01ae0bb6ea7929cf5102d2e3b51e3e786',template='general',template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='induced-photon',requirements=['verification','outcome'])
sys.set_int_max_str_digits(0)
PROOF={
 'law':'Same literal fullPhi form: Cartesian3840, Haar192*15, Theta129, K=4chi p_x^2+(3chi/20)Ecolour^2+64Pi^2; Lambda2^8192, chi2^-4096, original V4=Lambda/a sumTrGram^2, alignmentLambda54/40, plaquette7Lambda54/10, HEHC64. No law/source/normalizer adoption or bareB2.',
 'neutral_section':'At the exact flat2+2 witness let h be PLUS primitive neutral matrix, whose physical neutral Hamiltonian is -Lambda dGamma(h). On a small gauge-orbit tube the positive spectral cluster has rank128 and gap from zero. Let Pplus(F) be that spectral projector and v(F)=wedge^128(Pplus(F)) Omega_even, where Omega_even fills all4 colours at each of the32 checkerboard-even sites. This reference is an EXACT local SU4 singlet. Pplus covariance makes v gauge-equivariant with no local-slice or free-orbit assumption. Whenever v is nonzero it occupies the full upper cluster and its neutral energy is exactly -Lambda S128(h(F)).',
 'nonzero_projection':'Wrap-only witness is flat, SU4, and AP separate. Each Pauli block has h(p)=(M+R) sigma3 + transverse Pauli terms. Pair p with p+pi(1,1,1) and compress Pplus to checkerboard-even sites: PAA=(Pplus(p)+Pplus(p+pi))/2. With M=A*a and K2=2ci^2(b^2+c^2)>0, detPAA=(1-dot(nplus,nminus))/8. dot=(M2-R2-K2)/sqrt((M2-R2-K2)^2+4M2K2), decreasing with R2, hence detPAA>=eta=K2/[4(M2+K2)]. Its eigenvalues lie in[0,1], so EACH eigenvalue>=eta. All128 compression eigenvalues therefore give ||v(Fstar)||^2=detPAA>=eta^128. The executable proves eta>=2^-8 by exact rationals, hence c2>=2^-1024. This deliberately weaker eigenvalue bound avoids mistaking a finite determinant for an interval enclosure.',
 'tube_section':'M=R3840_Cart times SU4^192 with its canonical product metric has dimensionD6720 and nonnegative Ricci curvature. The compact gauge orbit O=SU4^64.Fstar is a smooth embedded homogeneous orbit even with a continuous stabilizer. d(F,O) is gauge-invariant and1-Lipschitz. Take f(F)=(1-d(F,O)/delta)_+, delta2^-2048; it is compact in Phi and H1, with no hard-wall Hamiltonian. Psi=Z^-1/2 f(F) v(F) tensor charged_even_singlets64copies tensor Ref4 tensor constantTheta. Z is its ACTUAL positive scalar integral; its lower bound is not a chosen normalization. This is a global form-domain Rayleigh state, not a D(H^2) assertion. Smooth invariant approximations are possible in form norm.',
 'parallel_volume':'Use the complete product metric in which Cart coordinates and the left SU4 generators T_A are orthonormal; Ric>=0. The compact embedded orbit has constant dimension959, codimension5761. For each unit normal (p,nu), before its first minimizing cut time tau (no later than first focal time), the normal-exponential Jacobian j>0 has j(t)=t^5760(1+O(t)); the symmetric distance Hessian S on all6719 orthogonal directions obeys trSprime=-trS2-Ric. Consequently u=j^(1/6719) satisfies u_doubleprime<=0 and u(0)=0; concavity between epsilon,t,2t and epsilon->0 gives j(2t)<=2^6719 j(t) whenever2t<tau. Put j=0 at/after tau only for integration/comparison, NOT to claim concavity of that discontinuous extension. If t<tau<=2t the comparison is trivial, and if tau<=t both are0. The normal cut locus has ambient measure0; outside it the minimizing footpoint and normal are unique, so the normal-bundle volume formula has multiplicity1. Integrating j(2t)*2 gives Vol(Tube(2r))<=2^6720 Vol(Tube(r)). Compact smooth embeddedness supplies a local tubular neighbourhood but no numerical reach bound enters. Thus ||grad f||2^2/||f||2^2<=4*2^6720/delta2. Stabilizers require the orbit SU4^64/H and its normal bundle, never a free-orbit parametrization.',
 'projector_gradient':'On the unit tube, native Schur/telescoping endpoint bounds give unscaled ||Dh||OP<=CH with exact conservative CH<2^20. Positive gap at the center is>=sqrtK2>2; CH*delta<1, so the tube stays separated by>1 from zero with the same128 bands. A real spectral-projector Sylvester/resolvent bound gives ||DP||OP<=2CH per unit tangent. Exterior128 differentiation and the6720 coordinate sum give ||grad v||<=2^40. Also ||v(F)-v(nearestorbit)||<=128*2CH*delta<2^-513, while center norm>=2^-512. Hence ||v||^2>=c2/4 throughout support. These are OP bounds on this compact tube, not global bounds on Phi/current.',
 'kinetic':'Using ||grad(fv)||2^2<=2||gradf||2^2+2*2^80||f||2^2 and Z>=c2||f||2^2/4 gives Kphi_colour<=32chi/c2*(4*2^D/delta2+2^80)<=2^7752. Charged references and Ref are localSU4singlets; there is no unpriced gauge carrier derivative. This huge absolute upper is still negligible relative toLambda; no chi=0 substitution.',
 'classical_variation':'On this tube the original neutral+quartic variation/Lambda is bounded by2^60 delta. This follows from128-band KyFan Lipschitz<=128CH delta and quartic telescoping4*((Rstar+delta)^4-Rstar^4), Rstar<16. Original zeroalignment/zeroplaquette center has geometry/Lambda<=2^20*Lambda^(1/4)*delta2: endpoint differences and four-link telescoping bound the complete original HS sums, not their floating residuals. Both coefficients are deliberately conservative and checked against primitive finite constants.',
 'charged_and_Gauss':'Each charged species fills all4 colours at the32 even sites, so every onsite expectation is tracePhi0=0 and every offsite one/two/AP expectation is0 for ALL fields/Theta. Charged matter expectation is exactly0, not its ground. All64 species and the shared origin Ref4 give rho_x=128*(-1)^(x0+x1+x2), totalrho0. Native Lrho=12rho, so HC=64rho.Lplus.rho=2^24/3 EXACT; HE0 in beta0. Longitudinal Gauss is solved as in the actual reduced law. Canonical neutral128+charged8192+Ref4=N8324; this is a legitimate physical sector. Its conserved electric coset is not assumedalpha0 or minimizing.',
 'original_cover':'This normalized beta0 trial is also a normalized original-Hilbert packet: use common periodic representatives and an L2 beta envelope supported in |A^-1 beta|<=1. The field/fermion section is Theta-independent, so the extra HE expectation<=64. No beta0 ground, real-cover L2 vacuum or pointwise Borel extension follows.',
 'quantum_comparison':'Combining the exact static noncentral upper with the positive kinetic, classical-variation, HC and optional original-cover HE budgets gives Etrial/Lambda<=enoncentral+2^-400. Therefore the central uniform-static floor remains beaten after paying fullPhi/SU4/Gauss/charged/Pi/Coulomb quantum costs. This is a variational competitor, NOT identification of the actual global well or full ground.',
 'conditional_old_image':'The old Clifford-root T image remains a CONDITIONAL LOCALWELL construction, not a global root theorem. Its actual field_ray has real orthonormal polar Q, uniform slow injection/8 and zero-mean fast injection, giving field distance to the central gauge orbit<=2|r| on the ACTUAL b6 support |r|<2^-20. Prepared_radius<2^-3000 is an L2/RMS bound, NOT a support radius: ||r f_A||2=sqrtTrCov and degree3 hypercontractivity gives ||r P f_A||2<=3^(7/4)||P f_A||2 sqrtTrCov<9||P f_A||2 sqrtTrCov. With b6<=1 and the conditional normalization upper2, ||r scalar||2<=prepared_radius, uniformly any unit body vector. Cauchy gives mean field distance<=2prepared_radius. Uniform Lipschitz bounds on the true compact support yield neutral+quartic expectation>=Lambda*(ecentral-2^60*2prepared_radius); arbitrary charged lower>=-2^30 there, and kinetic/alignment/plaquette/HE/HC are nonnegative. Therefore the competitor beats the whole old T image, conditional on its frozen root/SPD/gap/chart/normalizer suppliers. It does NOT prove pointwise support2^-3000, reject prepared-density/current statements, or rebind an old root/producer epoch.',
 'scope':'Exact geometric/form-domain variational reasoning, exact rational budgets, and finite floating projector controls are separate. No implicit6720-dimensional wave integral is numerically evaluated, no exact ground/pole/commoncone/phase uniqueness is inferred. No Root/mean solver rerun, oldSOURCE/CURRENT427/adoption or layout/process changes.'
}

def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def exact_budgets(flat:dict[str,Any],old:dict[str,Any])->dict[str,Any]:
    cert=flat['exact_outward_certificate'];p=flat['primitive_profiles'][0]
    a,b,c=[F(cert['trial_parameters'][k]) for k in ('a','b','c')]
    A=F(p['onsite']);rows=p['paths']
    assert len(rows)==3 and all(r['length']==1 for r in rows)
    assert len({r['ci'] for r in rows})==1
    ci=F(rows[0]['ci']);M2=(A*a)**2;K2=2*ci*ci*(b*b+c*c)
    eta=K2/(4*(M2+K2));assert eta>=F(1,256) and K2>4
    R2=256*(a*a+b*b+c*c);assert R2<256
    k=abs(A)+2*sum((abs(F(r['c0']))+abs(F(r['ci'])) for r in rows),F(0))
    CH=256*k*17;assert CH<F(2)**20
    delta=F(2)**-2048;chi=F(2)**-4096;lam=F(2)**8192;D=6720;c2=F(2)**-1024
    assert 128*2*CH*delta<F(2)**-513 and CH*delta<1
    kinetic=32*chi/c2*(4*F(2)**D/delta**2+F(2)**80)
    assert kinetic<F(2)**7752
    classical=F(2)**60*delta+F(2)**20*F(2)**2048*delta**2
    HC=F(2)**24/3;original_HE=F(64)
    extra=classical+(kinetic+HC+original_HE)/lam
    assert extra<F(2)**-400
    lower=F(cert['central_all64_lower']);upper=F(cert['trial_static_energy_divided_Lambda']['upper'])
    quantum_upper=upper+extra;separation=lower-quantum_upper;assert separation>F(3,10)
    oldradius=F(old['uniform_intertwiner_price']['prepared_radius']['upper']);assert oldradius<F(2)**-3000
    oldloss=F(2)**60*2*oldradius+F(2)**30/lam
    old_image_lower=lower-oldloss;old_image_gap=old_image_lower-quantum_upper;assert old_image_gap>F(3,10)
    return dict(eta_exact=str(eta),compression_eigenvalue_lower_coarse='2^-8',projection_norm_squared_lower='2^-1024',center_gap_squared_lower_exact=str(K2),native_CH_exact=str(CH),field_HS_radius_squared_exact=str(R2),ambient_dimension=D,gauge_stabilizer_dimension=1,gauge_orbit_dimension=959,delta_exact='2^-2048',chi_exact='2^-4096',Lambda_exact='2^8192',kinetic_absolute_upper_coarse='2^7752',classical_variation_divided_Lambda_exact=str(classical),HC_exact=str(HC),original_packet_HE_upper_exact='64',quantum_extra_divided_Lambda_upper_coarse='2^-400',quantum_energy_divided_Lambda_upper=str(quantum_upper),central_uniform_static_lower=str(lower),quantum_separation_lower=str(separation),quantum_separation_display=float(separation),conditional_old_image_lower=str(old_image_lower),conditional_old_image_separation=str(old_image_gap),conditional_old_image_separation_display=float(old_image_gap),actual_implicit_normalizer_not_selected=True,scope='Exact rational costs for the described physical form-domain state; oldimage comparison conditional on its frozen suppliers.')


def finite_controls(data:dict[str,Any],eta:F)->dict[str,Any]:
    n=data['native'];p=data['profiles'][0]
    a,b,c=map(float,(F(537,1000),F(229,500),F(81,250)))
    sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]],complex);sz=np.diag([1.,-1.])
    frame=np.zeros((4,4,4),complex)
    frame[0,:2,:2]=a*sz;frame[0,2:,2:]=a*sz
    frame[1,:2,:2]=b*sx;frame[2,:2,:2]=c*sy;frame[2,2:,2:]=c*sy;frame[3,2:,2:]=b*sx
    fields=np.broadcast_to(frame,(64,4,4,4)).copy();links=np.broadcast_to(np.eye(4,dtype=complex),(192,4,4)).copy()
    W=[np.diag([1,1,-1,-1]),np.eye(4),np.diag([-1,-1,1,1])]
    for s,site in enumerate(n['sites']):
        for j in range(3):
            if int(site[j])==3:links[3*s+j]=W[j]
    h=-field.assemble(p,n,fields,links);val,vec=np.linalg.eigh(h)
    assert sum(val>0)==128 and val[128]>2 and val[127]<-2
    P=vec[:,128:]@vec[:,128:].conj().T
    even=np.sum(n['sites'],axis=1)%2==0;ix=np.flatnonzero(np.repeat(even,4))
    PA=P[np.ix_(ix,ix)];pv=np.linalg.eigvalsh(PA)
    assert len(ix)==128 and min(pv)>float(eta)
    lognorm=float(np.log(pv).sum());assert lognorm>128*np.log(float(eta))
    rho=np.where(even,128,-128);D=np.rint(n['D']).astype(int)
    assert np.array_equal((D@D.T)@rho,12*rho) and sum(rho)==0
    green=np.array(fiber.rational_green(n['sites']),float);hc=float(64*rho@green@rho)
    assert abs(hc-float(F(2)**24/3))<1e-7
    rng=np.random.default_rng(222101030);T=data['geom']['M']/np.sqrt(8)
    gs=np.array([expm(1j*np.einsum('a,aij->ij',v,T)) for v in rng.normal(size=(64,15))*.17])
    f2=gs[:,None]@fields@gs.conj().swapaxes(-1,-2)[:,None]
    lookup={tuple(map(int,s)):i for i,s in enumerate(n['sites'])};l2=links.copy()
    for s,site in enumerate(n['sites']):
        for j in range(3):
            target=lookup[tuple(map(int,(site+np.eye(3,dtype=int)[j])%4))]
            l2[3*s+j]=gs[s]@links[3*s+j]@gs[target].conj().T
    G=np.zeros((256,256),complex)
    for s,g in enumerate(gs):G[4*s:4*s+4,4*s:4*s+4]=g
    hg=-field.assemble(p,n,f2,l2);cov=float(np.linalg.norm(hg-G@h@G.conj().T));assert cov<1e-10
    Pg=G@P@G.conj().T;lg=float(np.linalg.slogdet(Pg[np.ix_(ix,ix)])[1]);assert abs(lg-lognorm)<1e-10
    wrongsign=float(-val[:128].sum()+val[128:].sum());assert wrongsign>3000
    # Actual small field point lies OUTSIDE delta2^-2048; not an interval proof.
    tangent=rng.normal(size=(64,4,15));tangent/=np.linalg.norm(tangent)
    dphi=np.einsum('xma,aij->xmij',tangent,T)
    eps=2.**-14;hp=-field.assemble(p,n,fields+eps*dphi,links);vp,up=np.linalg.eigh(hp)
    assert sum(vp>0)==128
    PP=up[:,128:]@up[:,128:].conj().T;norm_difference=float(np.linalg.norm(PP-P,2))
    assert norm_difference>1e-7 and norm_difference/eps<2.**20
    return dict(scope='Finite float64 actualN4 center/gauge/projector controls. Perturbed point OUTSIDE quantum tube; not interval/ground/implicit integral evaluation.',neutral_positive_rank=128,neutral_spectral_min_abs=float(min(abs(val))),projection_compression_min_eigenvalue=float(min(pv)),projection_log_norm_squared=lognorm,exact_eta_lower=float(eta),neutral_unscaled_global128=float(val[128:].sum()),gauge_covariance_HS_defect=cov,invariant_singlet_projection_log_norm_defect=abs(lg-lognorm),charged_checkerboard_rho_squared=int(rho@rho),HC_actual_float=hc,HC_exact=str(F(2)**24/3),wrong_neutral_sheet_energy_defect_divided_Lambda=wrongsign,wrong_omit_HC_energy_defect=hc,external_perturbation_epsilon=eps,projector_changed_operator_norm=norm_difference,all64_all192_native=True)


def run(output:Path)->dict[str,Any]:
    started=time.monotonic();raws={name:(ROOT/name).read_bytes() for name in PINS}
    assert all(sha(raw)==PINS[name] for name,raw in raws.items())
    flat,old=[json.loads(raws[name]) for name in PINS]
    budget=exact_budgets(flat,old)
    data=field.load_mean(g_override=[1,1,1,1],mean_descriptor={'scope':'Fresh frame only; exact microconfig captured; no root solve.'})
    for actual,declared in zip(data['profiles'],flat['primitive_profiles']):
        assert str(F(actual['onsite']))==declared['onsite'] and actual['load']==declared['load']
        assert [str(F(r['ci'])) for r in actual['paths']]==[r['ci'] for r in declared['paths']]
    control=finite_controls(data,F(budget['eta_exact']))
    assert all((ROOT/name).read_bytes()==raw for name,raw in {**CAPTURED,**raws}.items())
    archive=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in {**CAPTURED,**raws}.items():z.writestr(name,raw)
    result=dict(kind='FULL_QUANTUM_NONCENTRAL_PHYSICAL_FORM_TRIAL_NOT_GROUND_CHARACTERIZATION',plan=PLAN,proof=PROOF,exact_budgets=budget,finite_controls=control,own_source_sha256=sha(CAPTURED[Path(__file__).name]),input_sha256=PINS,source_sha256={k:sha(v) for k,v in CAPTURED.items()},sources_inputs_unchanged=True,immutable_input_archive=dict(path=archive.name,sha256=sha(archive.read_bytes())),runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started,not_admitted=['actual globalground/unique phase','sameground photon','oldSOURCE/CURRENT427 adoption','new physical parameter/normalizer/clock','real-cover L2 vacuum','D(H2) of Lipschitz trial'])
    output.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.output)
    print(json.dumps(dict(exact_budgets=result['exact_budgets'],finite_controls=result['finite_controls'],elapsed_seconds=result['elapsed_seconds']),indent=2))
