"""Conditional global polar-normal SU4 chart certificate, not a source admission.

The algebraic derivation is part of the receipt. Floating controls do not certify
an implicit root. No occupied-fermion Kato frame is used here.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.linalg import expm, logm
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_gauss_quantum as gauss
import induced_phi_nonuniform_fermionic_source_unitary as source
import induced_phi_local_gauss_chart_jet_control as jets

ROOT = Path(__file__).resolve().parent
INPUTS = {
    'induced_phi_local_gauss_orbit_projector_control_receipt.json': 'e18a9ce4ece8dea57e8f739eb22ea4aebd21c0b4c033eba665a48b9cd108a376',
    'induced_phi_local_gauss_chart_jet_control_haar_aligned_receipt.json': '611aa4d15d58b46a150d0709966ff8a1f7b2eb72f4307364090f5fab5e66cfa9',
    'induced_phi_global_normal_h_jet_control_receipt.json': 'e48a53776184f9eda572d5f06ef061f880cfb13935b41f3a707b25d1994e58b8',
}
DERIVATION = {
    'domain': 'Conditional exact Clifford root Phi_mu=sqrt(g_mu)Gamma_mu, all g>0, h=min(g)/2>1/20, w=sum(g)/2<1/3. V=64 periodic connected N4 graph; root exact normal recipe Q0 from normal_geometry, not rounded cached columns. r=(s45,f2835,C2880), zero-site-mean f, ||r||<=rho, rho<=3*2^-20. Haar U in SU4^64 unrestricted. Q(s) is actual polar normal projector of Phi0+Q0*s/8. Admit the SHA-bound exact Q0 algebraic recipe and Q1<=512,Q2<=2^20,Q3<=2^33; these are normal-frame, NOT occupied Kato, hypotheses. Physical metric is Phi-cart/8 and link-cart*10/3; injection proof uses UNWEIGHTED cart coordinates.',
    'polar_collar_extension': 'The projector receipt constructs actual Qpolar holomorphically on complex slow ball radius1/256 with norm<9/8, using the inverse-square-root binomial series on N^T N (no occupied Kato). At ||s||<=rho<=3R, n-direction complex polydiscs of radius(1/256-rho)/n stay inside that ball. Multivariate Cauchy therefore gives D^n Q <= n^n*(9/8)/(1/256-rho)^n for n=1,2,3. The exact rational comparisons in this receipt extend512,2^20,2^33 to the3R collar, rather than assuming the older R-only derivative statement applies unchanged.',
    'reach_identity': 'For unitary g=tI+Y, TrY=0, Delta=gPhi gdag-Phi equals [conj(t)Y-tYdag,Phi]/2 + YPhiYdag-{YYdag,Phi}/2, since gdag g=ggdag=I. First term tangent, remainder tuple Frobenius norm<=2||Phi||F||Y||F^2. Clifford monomial decomposition on complex traceless matrices gives ||[g,Phi]||F^2>=gamma||Y||F^2 with gamma=4 min(g)>2/5. Thus ||PiN Delta||<=2||Phi||F/gamma*||Delta||F^2. If two orbit points p,q are nearest to p+n, distance equality implies ||q-p||^2=2<n,q-p><=4||n||||Phi||F/gamma*||q-p||^2. Hence uniqueness for ||n||<gamma/(4||Phi||F). Compactness gives nearest point; nearest residual is normal. The same inequality limits focal radius by taking q->p, so normal exponential is nonsingular on this tube. This is an explicit chord proof, not a theorem-name analogy.',
    'nearby_orbit': 'Root tuple ||Phi||F=2sqrt(sum g)<2. For slow delta tuple ||delta||F<=rho/(8sqrt2); perturbation of complex commutator map has norm<=2||delta||F<rho/4. Its least singular value on traceless matrices exceeds 3/5-rho/4; square>1/3. Perturbed tuple norm<2; reach>1/24 Frobenius. Root reach>1/20. Retain conservative common radius1/32 Frobenius, also1/32 Cartesian (Cartesian norm=sqrt2*Frobenius). Stabilizer exactly Z4 because positive Clifford tuple generates M4(C); nearby positive commutator gap also leaves only scalar commutants. Every field perturbation has Cartesian norm<=2rho, well inside1/32, including3R collar. Product-site uniqueness is applied sitewise, not by falsely multiplying reach by sqrt64.',
    'root_gauge_equation': 'Let O0=orbit(Phi0,T), P0=O0 O0^+, ||O0||<5, sigma_min(O0)>3/5, ||O0^+||<5/3. Set A_x=Phi0+Q0*s/8+Q(s)f_x. ||A-Phi0||site<=2rho; ||dA||<=4. P0(A-Phi0)=(Q(s)-Q0)f, norm<=512rho^2, and ||P0 dA||<=1024rho, using site Holder without sqrt64. Root-normal gauge solves G(alpha,A)=O0^+ cart(Ad_exp(-iTalpha) A-Phi0)=0. In ||alpha||<=1024rho^2, ||G_alpha+I||<=(5/3)(4rho+15||alpha||)<=16rho: commutator operator<=2||cart field||, root orbit<=5, real exp Maurer J-I<=||alpha||. ||G(0,A)||<=(2560/3)rho^2. The map alpha->alpha+G is contraction and maps this ball to itself since (2560/3)+16rho*1024<1024. It therefore supplies the actual principal root branch, not an assumed radial Kato gauge. Its residual is root normal, hence the unique tube nearest orbit representative. ||alpha||<=1024rho^2. Implicit differentiation gives ||Dalpha||<=(5/3)(1024rho+8*1024rho^2)/(1-16rho)<=2048rho. All bounds hold on the product-site alpha l2 ball by site Holder. The root branch is modulo the exact common scalar stabilizer.',
    'global_injection': 'Define F(r) as Q0^T cart(Ad_exp(-iTalpha_x) A_x-Phi0), plus Cart principal logs of exp(-iTalpha_x) exp(iTC_e) exp(iTalpha_y). F0 is the orthogonal mean/zero-mean normal injection and identity link injection. For Phi projection, Pnormal kills the root orbit angular column: ||DF_Phi-DF0_Phi||<=1024rho+8*1024rho^2+2(2rho+5*1024rho^2)*2048rho<2048rho. Principal log link coordinates have norm<=2rho; J inverse<=1/(1-2rho). Endpoint incidence lifts each have norm sqrt3<2. Link differential discrepancy <=[3rho+2*1024rho^2+4*2048rho]/(1-2rho)<16384rho: the three rho terms bound J(C)-I and J(Cprime)-I, while2alpha bounds output adjoint rotation. Thus ||DF-DF0||<=18432rho, giving support margin9/512 and collar27/512, both<1. Integrating on convex r ball gives ||F(r)-F(rprime)||>=(1-18432rho)||r-rprime||. These constants are derived, not the proposed65536 assertion. If two full Haar embeddings agree, unique root-nearest orbit fixes their transformed Phi and hence their gauge representatives up to site Z4. Link principal branches cannot differ by nontrivial Z4: their combined logarithmic path length<4rho, whereas nonidentity centre Frobenius distance from I>=sqrt8. Connectivity forces all site centres equal. Therefore embedding is injective modulo common Z4 over the entire product Haar group; no small-angle restriction. Common-centre representation descends only for total centre exponent n divisible by4 (8324 is such a case). This proves chart geometry, not the supplier source exponent itself.',
    'mean_inverse': 'In physical body framing, mean D sends normal Phi to /sqrt8, angular a to O0 a/sqrt8, links to sqrt(10/3)(C+d a). Orthogonality gives ||a||<=||vPhi||/sqrt(h), ||rPhi||<=sqrt8||vPhi||. ||d||<=sqrt12<4 and ||C||<=sqrt(3/10)||vlink||+4||a||. Triangle yields ||D0^-1||<4+5/sqrt(h). Thus the old u=4+10/sqrt_upper(h) really dominates this actual bound whenever sqrt(h)<=sqrt_upper(h)<=2sqrt(h); it is NOT asserted to dominate4+10/sqrt(h). A rational universal mean bound u0=54 follows h>1/20 and sqrt20<5. Recover all6720 input coordinates, not only reduced normal Schur complement.',
    'body_jets_and_inverse': 'Use actual intrinsic Haar/body D, angle Maurer equal I, not laboratory D derivatives or local exponential J(theta). Exact field chain at radiusrho: F1=17/8+512rho<4, F2=1024+2^20 rho, F3=3*2^20+2^33 rho. From actual D_Phi=(dA+[iTa,A])/sqrt8 and D_link=sqrt(10/3)(J(C)dC+a_x-Ad_E a_y), ||D1||<(F2+8)/2+10 and ||D2||<(F3+2F2)/2+20. Support bounds1053/2,1578005; collar1055/2,1586199. Fundamental theorem along body r implies ||D-D0||<=aD rho independent of Haar U. Since54*aD*rho<1/4 even on collar, Neumann inverse<=u/(1-u*aD*rho)<2u<=108. Old u<=54, and even54*4096*R=27/128<1/4 on support, justifying its actual inv=2u, not merely the larger universal108. The old4096 and2^24 D prices dominate both actual derived bounds, but sharper actual bounds are needed for collar inverse. The old reach=h^2/[100(1+2w)]<1/900<1/32 and old transition=2048(1+10/sqrt_upper(h))*R>2048R dominate our root-gauge derivative price on support. No square-root-upper reciprocal is silently treated as an exact lower.',
    'cometric_and_Haar': 'E=D^-1, A=E E^T, v=108, a=aD,b=bD. Eprime=-E Dprime E; Esecond has two ordered E Dprime E Dprime E terms minus E Dsecond E. Hence ||A||<=v^2, ||A1||<=2v^3 a, ||A2||<=6v^4 a^2+2v^3 b. For j=log|det D| in full6720 body coordinates, jprime=Tr(E Dprime), jsecond=Tr(E Dsecond-E Dprime E Dprime), giving ||grad j||<=6720 v a and ||Hess j||<=6720(v b+v^2 a^2). Relative density is |det D| dr dHaar, NOT a compact-angle Lebesgue determinant. Body angular fields have zero Haar divergence by unimodularity and j is angle-independent. Flat-measure conjugation of the normal divergence-form kinetic operator gives geometric potential magnitude<=(5760 A1*jg+5760 Amax*jh)/4+Amax*jg^2/8. This is exactly the old local_gauss_prices geometric expression; it does not validate its action/Hermite/Noether supplier estimates. CommonZ4 quotient multiplies Haar density by a constant, so log derivatives unchanged.',
    'scope': 'Conditional global geometry proven under stated exact root/Q hypotheses, plus finite structural controls. Root existence/enclosures and the input suppliers remain separate. No original SOURCE, action residual, occupied Kato, ground state, pole, current or interacting Phi/U1 body propagation promotion. Nonuniform numerical points are NOT a global proof. Old local_gauss_prices geometric ingredients are justified on this stronger root box, not automatically across its entire h>1/64 accepted domain.',
}


def exact_bounds() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, rho in [('support', F(1, 1 << 20)), ('collar', F(3, 1 << 20))]:
        f2 = 1024+(1 << 20)*rho
        f3 = 3*(1 << 20)+(1 << 33)*rho
        a, b = (f2+8)/2+10, (f3+2*f2)/2+20
        v = F(108)
        am, a1, a2 = v*v, 2*v**3*a, 6*v**4*a*a+2*v**3*b
        jg, jh = 6720*v*a, 6720*(v*b+v*v*a*a)
        q_derivatives = [F(n**n)*F(9,8)/(F(1,256)-rho)**n for n in (1,2,3)]
        assert all(bound < price for bound, price in zip(q_derivatives,(512,1 << 20,1 << 33),strict=True))
        assert 54*a*rho < F(1,4)
        assert F(2560,3)+16*rho*1024 < 1024
        assert F(5,3)*(1024+8192*rho)/(1-16*rho) < 2048
        assert 2*rho < F(1,32)
        assert 18432*rho < 1
        assert a < 4096 and b < 1 << 24
        result[label] = {k: str(x) for k,x in dict(radius=rho,root_gauge_size_upper=1024*rho*rho,root_gauge_derivative_upper=2048*rho,injection_D_difference_upper=18432*rho,mean_inverse_upper=54,Neumann_margin=54*a*rho,inverse_upper=v,D1_upper=a,D2_upper=b,cometric_upper=am,cometric_D1_upper=a1,cometric_D2_upper=a2,log_density_gradient_upper=jg,log_density_Hessian_upper=jh,Haar_potential_divided_chi_upper=(5760*a1*jg+5760*am*jh)/4+am*jg*jg/8).items()}
        result[label]['polar_Cauchy_derivative_upper'] = [str(x) for x in q_derivatives]
    return result


def finite_controls() -> dict[str, Any]:
    full = gauss.stationary.full
    g = [F(1,4), F(1,8), F(1,8), F(1,8)]
    phi0 = np.sqrt(np.asarray(g,dtype=float))[:,None,None]*full.supplier.primitive.SELECTED
    geom = full.normal_geometry({'frame':phi0,'a':1.,'g0':float(g[0]),'gs':float(g[1])},[full.exact(x) for x in g])
    T, q0 = geom['M']/np.sqrt(8), geom['Q']
    o0 = gauss.orbit(phi0,T)
    pinv = np.linalg.solve(o0.T@o0,o0.T)
    rng = np.random.default_rng(2221010)
    dag = lambda x: x.conj().swapaxes(-1,-2)
    def unit(shape: tuple[int,...]) -> np.ndarray:
        x = rng.normal(size=shape)
        return x/np.linalg.norm(x)
    def fields(s: np.ndarray, f: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        slow = np.einsum('a,amij->mij',s,geom['deltas'])/8
        q, _, _ = gauss.adapted_frame(phi0+slow,T,q0)
        return phi0[None]+slow[None]+np.einsum('xma,aij->xmij',(f@q.T).reshape(64,4,15),T), q
    def rootfix(a: np.ndarray) -> tuple[np.ndarray,np.ndarray,float]:
        al = np.zeros((64,15))
        fixed, tangent = a, np.zeros_like(al)
        for _ in range(4):
            us = np.asarray([expm(-1j*np.einsum('a,aij->ij',x,T)) for x in al])
            fixed = us[:,None]@a@dag(us)[:,None]
            tangent = jets.cart(fixed-phi0,T).reshape(64,60)@pinv.T
            al += tangent
        return al, fixed, float(np.linalg.norm(tangent))
    sites = np.asarray([(i,j,k) for i in range(4) for j in range(4) for k in range(4)])
    index = {tuple(x):i for i,x in enumerate(sites)}
    endpoints = [(x,index[tuple((site+np.eye(3,dtype=int)[axis])%4)]) for x,site in enumerate(sites) for axis in range(3)]
    base = np.asarray([expm(1j*np.einsum('a,aij->ij',2*unit((15,)),T)) for _ in range(64)])
    rows = []
    for rho in (2.**-20,3*2.**-20):
        s, f, c = [rho/2*unit(shape) for shape in ((45,),(64,45),(192,15))]
        f -= f.mean(axis=0)
        a,q = fields(s,f)
        al,fixed,residual = rootfix(a)
        us = np.asarray([expm(1j*np.einsum('a,aij->ij',x,T)) for x in al])
        es = np.asarray([expm(1j*np.einsum('a,aij->ij',x,T)) for x in c])
        logs = np.asarray([jets.cart(-1j*logm(dag(us[x])@e@us[y]),T) for e,(x,y) in zip(es,endpoints,strict=True)])
        laboratory = base[:,None]@a@dag(base)[:,None]
        representatives = base@us
        reconstructed = representatives[:,None]@fixed@dag(representatives)[:,None]
        links = np.asarray([base[x]@e@dag(base[y]) for e,(x,y) in zip(es,endpoints,strict=True)])
        rebuilt = np.asarray([representatives[x]@expm(1j*np.einsum('a,aij->ij',z,T))@dag(representatives[y]) for z,(x,y) in zip(logs,endpoints,strict=True)])
        ds,df = unit((45,)),unit((64,45));df-=df.mean(axis=0)
        slow=np.einsum('a,amij->mij',s,geom['deltas'])/8
        sd=np.einsum('a,amij->mij',ds,geom['deltas'])/8
        qj,qd=source.adapted_jet(phi0+slow,sd,T,q0)
        wrong_frozen=float(np.linalg.norm(f@qd.T))
        dt=unit((64,15))
        ang=1j*np.einsum('xa,aij->xij',dt,T)
        fd=np.einsum('xma,aij->xmij',(df@qj.T+f@qd.T).reshape(64,4,15),T)+sd[None]
        body=jets.cart(fd+ang[:,None]@a-a@ang[:,None],T)
        labdot=base[:,None]@(fd+ang[:,None]@a-a@ang[:,None])@dag(base)[:,None]
        physical=jets.cart(dag(base)[:,None]@labdot@base[:,None],T)/np.sqrt(8)
        wrongframe=float(np.linalg.norm(physical-jets.cart(fd,T)/np.sqrt(8)))
        wrongmetric=float(np.linalg.norm(physical-body))
        # A site-dependent central gauge is invisible to Phi but changes its adjacent links.
        centres=np.ones(64,dtype=complex);centres[0]=1j
        wrongcentre=np.asarray([centres[x]*e*centres[y].conjugate() for e,(x,y) in zip(es,endpoints,strict=True)])
        row=dict(radius=rho,actual_nonuniform_fast_norm=float(np.linalg.norm(f)),polar_adapter_alignment_error=float(np.linalg.norm(q-qj)),root_normal_residual=residual,root_gauge_norm=float(np.linalg.norm(al)),root_gauge_size_price=1024*rho*rho,Phi_reconstruction_error=float(np.linalg.norm(laboratory-reconstructed)),links_reconstruction_error=float(np.linalg.norm(links-rebuilt)),physical_body_frame_error=float(np.linalg.norm(physical-body/np.sqrt(8))),wrong_nonuniform_Z4_link_error=float(np.linalg.norm(wrongcentre-es)),wrong_frozen_polar_derivative_error=wrong_frozen,wrong_frozen_polar_derivative_error_over_radius=wrong_frozen/rho,wrong_omitted_local_frame_error=wrongframe,wrong_unweighted_physical_metric_error=wrongmetric,local_Haar_base_angle_norm=2.)
        assert residual < 1e-13 and row['Phi_reconstruction_error'] < 1e-12 and row['links_reconstruction_error'] < 1e-12
        assert row['root_gauge_norm'] < row['root_gauge_size_price']
        assert row['physical_body_frame_error'] < 1e-12
        assert wrong_frozen/rho > 1e-3 and wrongframe > .1 and wrongmetric > .1 and row['wrong_nonuniform_Z4_link_error'] > 1
        rows.append(row)
    return dict(couplings_TEST_only=[str(x) for x in g],exact_recipe_floating_Q_defect=float(np.linalg.norm(q0.T@q0-np.eye(45))),all64_sites_all192_links=True,full_Haar_unrestricted=True,rows=rows,status='Floating actual-adapter falsification controls, not interval root certification')


def run() -> dict[str, Any]:
    start=time.monotonic()
    raws={name:(ROOT/name).read_bytes() for name in INPUTS}
    for name,pinned in INPUTS.items():
        if pinned is not None and hashlib.sha256(raws[name]).hexdigest()!=pinned:
            raise ArithmeticError(f'Input epoch mismatch: {name}')
    # Capture actual imported dependencies and our two direct jet adapters.
    captures=dict(gauss.SOURCE_BYTES)
    captures.update(source.SOURCE_BYTES)
    for module in (source,jets):
        path=Path(module.__file__ or '')
        captures[path.name]=path.read_bytes()
    captures[Path(__file__).name]=Path(__file__).read_bytes()
    with threadpool_limits(limits=1):
        controls=finite_controls()
    inputs={name:hashlib.sha256(raw).hexdigest() for name,raw in raws.items()}
    assert all((ROOT/name).read_bytes()==raw for name,raw in raws.items())
    assert all((ROOT/name).read_bytes()==raw for name,raw in captures.items())
    return dict(kind='conditional_global_normal_gauss_chart_certificate',derivation=DERIVATION,exact_bounds=exact_bounds(),finite_controls=controls,input_sha256=inputs,input_set_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),source_at_import={name:dict(sha256=hashlib.sha256(raw).hexdigest(),bytes_utf8=raw.decode()) for name,raw in captures.items()},input_and_source_unchanged=True,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    receipt=run()
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),input_set_sha256=receipt['input_set_sha256'],exact_bounds=receipt['exact_bounds'],finite_controls=receipt['finite_controls'],elapsed_seconds=receipt['elapsed_seconds']),indent=2))
