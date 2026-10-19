"""Native N4 neutral128 static sea: uniform exact envelope and adverse controls.

This executable performs NEW finite controls, not a root solve or a full ground
computation. PROOF separates exact statements from floating demonstrations.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_stationary_phi_sea_envelope'
INPUTS = {
 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json':'4691ffaa9cd81902942cbc228fc47fafff14518be6b5df2e84d08c29c7da4756',
 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json':'5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354',
}
PROOF = {
 'native_fourier': 'Let spatial central holonomy z_j=exp(i pi ell_j/2), ell in{0,1,2,3}^3. Put z_j only on positive wrap edges (AP seam is separately native -1). With v_p(x)=exp(i p.x)/8, p_j=(2pi k_j+pi+pi ell_j/2)/4, native forward h_xy v_p(y) has factor exp(i p.d). Define h as PLUS primitive neutral h, so physical neutral operator is -Lambda dGamma(h). For every literal raw path c_d and onsite A, x_mu(p)=A delta_mu0+2 Re(sum_d exp(i p.d)c_d,mu). Hence h(p)=sum_mu x_mu Phi_mu. Ordered two-edge paths multiply actual links BEFORE endpoint averaging; no continuum sea formula or old lattice is imported. Neutral has native nearest paths only; charged actual two-link paths are exercised as endpoint/AP controls, not inserted in neutral pressure.',
 'trace_gram': 'For any four traceless Hermitian4 matrices, G_mu,nu=Tr(Phi_mu Phi_nu)/4 is real PSD and Tr h(p)^2=4 x(p)^T G x(p). Every real PSD4 G is realizable by traceless Hermitian4 matrices: choose four anticommuting Hermitian Gamma_a with Tr(Gamma_a Gamma_b)=4delta_ab, choose real R with RR^T=G, and Phi_mu=sum_a R_mu,a Gamma_a. This is a same-Gram realization, not an assertion that arbitrary Phi anticommutes.',
 'global_kyfan': 'For the FULL256 matrix H=directsum_p h(p), TrH=0. S128(H)=sum largest128 eigenvalues <=sum positive eigenvalues=Tr|H|/2. For each4 block, Tr|h|<=sqrt(4 Tr h^2)=4sqrt(xGx), so S128(H)<=2sum_p sqrt(xGx). No eigenpairing or two occupied states per momentum is assumed at either inequality. For the Clifford realization h^2=(xGx)I4 and tracelessness gives +/-sqrt(xGx) each twice (zero blocks harmless), so BOTH bounds saturate at total128 filling. Thus the minimum over all UNIFORM arbitrary Phi at fixed G of the neutral static energy is exactly -2Lambda sum_p sqrt(xGx).',
 'pressure_existence': 'F_ell(G)=c TrG^2-2sum_p sqrt(x_p G x_p), c=V/a=4096, on real PSD4. The square-root terms are concave in G; c||G||F^2 is strictly convex with modulus2c, so F is strongly strictly convex. With T=||G||F and B=2sum_p||x_p||, F>=c T^2-B sqrt(T), hence coercive and continuous and attains exactly one Gram minimum. There is no root solve in this argument. At a singular candidate choose unit v in kerG and G+t vv^T. The quartic change is c t^2, with ZERO linear term because v^T Gv=0. Sea derivative is -sum_p (v.x_p)^2/sqrt(x_p G x_p), with -infinity if an affected denominator vanishes. If span{x_p}=R4, at least one affected numerator is nonzero, so the directional derivative is strictly negative and boundary cannot minimize. The unique minimum is SPD. Interior equation is 2cG=sum_p x_p x_p^T/sqrt(x_p Gx_p); this characterizes existence, not a measured coupled-root result.',
 'exact_span_all_central': 'For the actual neutral one-link profile A onsite, real c0 and imaginary ci on each axis, x0=A+2sum_j c0_j cos p_j, xj=-2ci_j sin p_j. Four equally spaced samples at ANY shift have mean sin=cos=sin*cos=0 and mean sin^2=mean cos^2=1/2. Independent axes give (1/V)sum xx^T=diag(A^2+2sum c0_j^2,2ci_1^2,2ci_2^2,2ci_3^2), exactly. Native ci_j are nonzero, A>0, so span holds for ALL64 central sectors. Reflection/cubic symmetry is NOT assumed for quarter-central twists. The finite64 set has an attained sector minimum, but its identity and relation to configuration Bloch beta are not determined.',
 'noncliff_exact_adverse': 'At ell=0 choose Phi0=diag(3,-1,-1,-1), Phi1=Phi2=Phi3=0. Actual x0=A+sqrt2 c0 z for z=3,1,-1,-3, with multiplicities8,24,24,8. Native A+3sqrt2 c0>0 and c0<0. There are only64 positive eigenvalues (3x0), and192 negative eigenvalues (-x0). Fullhalf must occupy all64 positive and the64 least-negative, comprising24 copies of the lowest-x0 group and40 of the next group. Therefore S128=128A-112sqrt2 c0, whereas sum_p top2(h_p)=128A. The exact shortcut deficit is -112sqrt2 c0>0. Degenerate cutoff occupations are not unique; integer local counts in one eigenbasis are floating witnesses, not uniquely determined occupancies. SameGram Clifford Phi0=sqrt3 Gamma0 has S128=2sqrt3 sum_p x0 and saturates the envelope.',
 'nonuniform_global_coercivity': 'Define R^2=sum_x,mu||Phi_x,mu||HS^2 and Cphi=|A|+2sum_d||c_d||2. Minkowski on forward/reverse literal blocks, unitary multiplication and endpoint averaging give ||h(Phi,U)||F<=Cphi R (all SU4 links, including AP). Trh=0 remains true (traceless onsite and no zero-length paths). Hence S128<=8||h||F<=8Cphi R. At each site TrG_x^2 >=(TrG_x)^2/4=R_x^4/64, so sum_x TrG_x^2>=R^4/(64V). Exact static lower: E/ Lambda >= R^4/(64aV)-8Cphi R+2^2048(Aalign/40+7P/10). With c=1/(64aV), b=8Cphi, minimizing cR^4-bR analytically gives -3 b^(4/3)/(4^(4/3)c^(1/3)). This is a true global coercive lower, NOT equality at the uniform minimum and NOT an eigenstate.',
 'perturbation_cost': 'Fix a gauge and a central flat reference U0, put barPhi=site average, d^2=sum||Phi_x-barPhi||HS^2, r^2=sum_mu||barPhi_mu||HS^2, u^2=sum_e||U_e-U0_e||HS^2, Cu=2sum_d L_d||c_d||2. Telescoping ordered Wilson products and Minkowski give ||h(Phi,U)-h(barPhi,U0)||F<=Cphi d+Cu r u. The trace-zero difference obeys |S128(H)-S128(H0)|<=Tr|H-H0|/2<=8||H-H0||F. Flattening each site Phi to a real HS4-column matrix shows TrG^2=||Phi||Schatten4^4/16 is convex, so Jensen gives sum TrG_x^2>=V TrG_bar^2. Thus E/Lambda>=F_ell(Gbar)-8(Cphi d+Cu r u)+2^2048(Aalign/40+7P/10). Each term uses original alignment Aalign=sum_e,mu||Phi_x,mu-U_e Phi_y,mu U_e^dag||HS^2 and P=sum_faces(4-ReTr orderedUface). Young may absorb a linear controlled defect but leaves a negative constant; it does NOT establish the exact uniform ground.',
 'geometry_obstruction_and_open_inequality': 'Aalign=P=0 does NOT imply a central flat connection. Uniform diagonal Phi and constant noncentral diagonal SU4 links commute; all plaquettes are I and alignment zero, while Wilson loops have noncentral spectra invariant under gauge conjugacy. Such configurations have colour-dependent spatial momenta and are outside the scalar-x central theorem. The finite noncentral toron control exercises the literal sea rather than replacing its law. EXACT remaining requested global inequality, with ecentral=min_{ell in Z4^3,G PSD4} F_ell(G), is: for EVERY arbitrary sitewise traceless Hermitian Phi and EVERY SU4^192 U, (1/a)sum_x TrG_x^2 - S128(h(Phi,U)) + 2^2048[Aalign(Phi,U)/40+7P(U)/10] >= ecentral. This work proves coercivity and the explicit perturbation cost, not that inequality. In particular A/P cannot alone control distance u to central torons. A next discriminating representation must resolve all flat noncentral commuting holonomies and their covariantly constant Phi before claiming central/uniform dominance.',
 'scope': 'Exact finite algebra/proof and floating literal eigensystem controls are separate. Static neutral128 sea only; positive Phi and colour kinetic energies, the64 charged species plus neutral species/Gauss/Ref and configuration beta are NOT solved by a static neutral inequality. No fullPhi quantum ground, stationary photon, pole, macrobox, source normalization or root admission follows. Frozen input is used only for microscopic_config; unit g_override selects a fresh control frame, never a root solver.'
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def gram(phi: np.ndarray) -> np.ndarray:
    return np.einsum('mij,nji->mn',phi,phi).real/4


def coefficients(profile: dict[str, Any], ell: tuple[int,int,int]) -> tuple[np.ndarray,np.ndarray]:
    p=(2*np.pi*np.array(list(product(range(4),repeat=3)))+np.pi+np.pi*np.array(ell)/2)/4
    x=np.zeros((64,4));x[:,0]=float(profile['onsite'])
    for displacement,c in profile['raw_paths']:
        x+=2*np.real(np.exp(1j*p@displacement)[:,None]*np.asarray(c)[None])
    return p,x


def central_links(native: dict[str, Any], ell: tuple[int,int,int]) -> np.ndarray:
    links=np.broadcast_to(np.eye(4,dtype=complex),(192,4,4)).copy()
    for s,site in enumerate(native['sites']):
        for j in range(3):
            if int(site[j])==3:links[3*s+j]*=np.exp(1j*np.pi*ell[j]/2)
    return links


def fourier_matrix(native: dict[str, Any], p: np.ndarray) -> np.ndarray:
    return np.kron(np.exp(1j*native['sites']@p.T)/8,np.eye(4))


def sea(h: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(h)[-128:].sum())


def clifford_same_gram(phi: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    g=gram(phi);val,vec=np.linalg.eigh(g)
    root=(vec*np.sqrt(np.maximum(val,0)))@vec.T
    return np.einsum('ma,aij->mij',root,gamma)


def uniform_control(profile: dict[str, Any], native: dict[str, Any], phi: np.ndarray, ell: tuple[int,int,int]) -> dict[str, Any]:
    p,x=coefficients(profile,ell);blocks=np.einsum('pm,mij->pij',x,phi)
    # field.assemble returns physical heavy -h: reverse ONLY for neutral.
    h=field.assemble(profile,native,np.broadcast_to(phi,(64,4,4,4)),central_links(native,ell))
    if profile['scale']=='heavy':h=-h
    v=fourier_matrix(native,p);direct=np.zeros((256,256),complex)
    for k,b in enumerate(blocks):direct[4*k:4*k+4,4*k:4*k+4]=b
    eig=np.linalg.eigvalsh(blocks).reshape(-1);g=gram(phi)
    envelope=float(2*np.sqrt(np.maximum(np.einsum('pm,mn,pn->p',x,g,x),0)).sum())
    return dict(role=profile['role'],ell=list(ell),fourier_literal_Frobenius_error=float(np.linalg.norm(v.conj().T@h@v-direct)),trace_gram_max_error=float(np.max(np.abs(np.einsum('pij,pji->p',blocks,blocks).real-4*np.einsum('pm,mn,pn->p',x,g,x)))),global_KyFan128=float(np.sort(eig)[-128:].sum()),literal_KyFan128=sea(h),fixed_top2_per_momentum=float(np.linalg.eigvalsh(blocks)[:,2:].sum()),half_trace_absolute=float(np.abs(eig).sum()/2),gram_envelope=envelope,Gram=g.tolist())


def geometry_cost(native: dict[str, Any], phi: np.ndarray, links: np.ndarray) -> tuple[float,float]:
    sites=native['sites'];lookup={tuple(map(int,s)):i for i,s in enumerate(sites)}
    align=0.;plaquette=0.
    for s,site in enumerate(sites):
        for j in range(3):
            t=lookup[tuple(map(int,(site+np.eye(3,dtype=int)[j])%4))]
            delta=phi[s]-links[3*s+j]@phi[t]@links[3*s+j].conj().T
            align+=float(np.linalg.norm(delta)**2)
            for k in range(j+1,3):
                tk=lookup[tuple(map(int,(site+np.eye(3,dtype=int)[k])%4))]
                w=links[3*s+j]@links[3*t+k]@links[3*tk+j].conj().T@links[3*s+k].conj().T
                plaquette+=float(4-np.trace(w).real)
    return align,plaquette


def extension_control(profile: dict[str, Any], native: dict[str, Any], phi: np.ndarray, links: np.ndarray) -> dict[str, Any]:
    bar=phi.mean(axis=0);identity=np.broadcast_to(np.eye(4),(192,4,4));ref=np.broadcast_to(bar,phi.shape)
    h=-field.assemble(profile,native,phi,links);h0=-field.assemble(profile,native,ref,identity)
    cphi=abs(float(profile['onsite']))+2*sum(float(np.linalg.norm(c)) for _,c in profile['raw_paths'])
    cu=2*sum(int(np.sum(d))*float(np.linalg.norm(c)) for d,c in profile['raw_paths'])
    d=float(np.linalg.norm(phi-ref));r=float(np.linalg.norm(bar));u=float(np.linalg.norm(links-identity))
    a,p=geometry_cost(native,phi,links);q=sum(float(np.linalg.norm(gram(f))**2) for f in phi)
    qbar=64*float(np.linalg.norm(gram(bar))**2);s=sea(h);s0=sea(h0)
    return dict(Cphi=cphi,Cu=cu,d=d,r=r,u=u,alignment_original_HS_squared=a,plaquette_original_defect=p,geometry_scaled_by_Lambda='2^2048*(alignment/40+7*plaquette/10)',actual_h_difference_HS=float(np.linalg.norm(h-h0)),h_difference_HS_upper=cphi*d+cu*r*u,actual_KyFan_difference=abs(s-s0),KyFan_difference_upper=8*(cphi*d+cu*r*u),quartic_sum=q,quartic_Jensen_lower=qbar,sea=s,central_reference_sea=s0,static_without_positive_geometry_divided_Lambda=64*q-s,coercivity_lower_without_geometry=float(np.linalg.norm(phi)**4/64-8*cphi*np.linalg.norm(phi)))


def run(output: Path) -> dict[str, Any]:
    started=time.monotonic()
    snapshots={name:(ROOT/name).read_bytes() for name in INPUTS}
    sources=dict(field.CAPTURED)
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).suffix=='.py' and Path(path).parent==ROOT:
            sources.setdefault(Path(path).name,Path(path).read_bytes())
    sources[Path(__file__).name]=Path(__file__).read_bytes()
    snapshots.update(sources)
    for name,expected in INPUTS.items():
        if sha(snapshots[name])!=expected:raise ArithmeticError('Frozen input changed: '+name)
    data=field.load_mean(g_override=[1,1,1,1],mean_descriptor={'scope':'unit frame finite controls; no coupled root admission'})
    native=data['native'];profiles=data['profiles'];heavy=profiles[0];gamma=native['frame']
    rng=np.random.default_rng(22210);phi=[]
    for _ in range(4):
        m=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));m=(m+m.conj().T)/2;m-=np.trace(m)*np.eye(4)/4;phi.append(m/5)
    phi=np.array(phi);cliff=clifford_same_gram(phi,gamma)
    arbitrary=[uniform_control(heavy,native,phi,ell) for ell in ((0,0,0),(1,2,3),(3,3,3))]
    saturation=[uniform_control(heavy,native,cliff,ell) for ell in ((0,0,0),(1,2,3),(3,3,3))]
    noncliff=np.zeros((4,4,4),complex);noncliff[0]=np.diag([3,-1,-1,-1])
    adverse=uniform_control(heavy,native,noncliff,(0,0,0))
    positive=uniform_control(heavy,native,clifford_same_gram(noncliff,gamma),(0,0,0))
    _,x=coefficients(heavy,(0,0,0));bands=np.linalg.eigvalsh(x[:,0,None,None]*noncliff[0]);order=np.argsort(bands.reshape(-1));counts=np.bincount(order[-128:]//4,minlength=64)
    adverse['one_selected_eigenbasis_occupation_count_histogram']={str(k):int(np.sum(counts==k)) for k in range(5)}
    c0=heavy['paths'][0]['c0'];A=heavy['onsite'];adverse['exact_KyFan128']={'rational':str(128*A),'sqrt2_coefficient':str(-112*c0)}
    adverse['exact_fixed_top2']=str(128*A);adverse['exact_shortcut_deficit_sqrt2_coefficient']=str(-112*c0)
    adverse['actual_minimum_x0']=float(x[:,0].min());adverse['exact_positive_condition']='A^2>18c0^2 and A>0,c0<0';adverse['exact_positive_condition_holds']=bool(A>0 and c0<0 and A*A>18*c0*c0)
    span=[]
    diag_exact=[A*A+2*sum(p['c0']**2 for p in heavy['paths']),*[2*next(p['ci'] for p in heavy['paths'] if p['axis']==j)**2 for j in range(3)]]
    for ell0,ell1,ell2 in product(range(4),repeat=3):
        ell=(ell0,ell1,ell2)
        _,xx=coefficients(heavy,ell);span.append(dict(ell=list(ell),exact_covariance_diagonal=[str(z) for z in diag_exact],floating_covariance_error=float(np.linalg.norm(xx.T@xx/64-np.diag([float(z) for z in diag_exact])))))
    # Actual ordered two/AP endpoint control, kept distinct from neutral theorem.
    charged=[uniform_control(profiles[1],native,phi,(1,2,3))]
    nonuniform=np.broadcast_to(phi,(64,4,4,4)).copy();nonuniform+=np.sin(2*np.pi*native['sites'][:,0]/4)[:,None,None,None]*cliff[None]/11
    links=np.broadcast_to(np.eye(4,dtype=complex),(192,4,4)).copy()
    generator=np.diag([3,-1,-1,-1])/20
    for s,site in enumerate(native['sites']):links[3*s]=expm(1j*generator*np.cos(2*np.pi*site[1]/4))
    perturbed=extension_control(heavy,native,nonuniform,links)
    toron_fields=np.broadcast_to(noncliff/5,(64,4,4,4)).copy();toron_links=np.broadcast_to(np.eye(4,dtype=complex),(192,4,4)).copy()
    toron_link=np.diag(np.exp(1j*np.array([3,-1,-1,-1])/20))
    toron_links[::3]=toron_link
    toron=extension_control(heavy,native,toron_fields,toron_links)
    toron['exact_ansatz']={'Phi0':'diag(3,-1,-1,-1)/5','Phi1_Phi2_Phi3':'0','Ux':'diag(exp(3i/20),exp(-i/20),exp(-i/20),exp(-i/20))','Uy_Uz':'I4','alignment_exact':'0','plaquette_exact':'0','reason':'Diagonal commutation and constant commuting unitary links. Tiny floating HS residuals are roundoff, NOT a positive 2^2048 cost.'}
    toron['Wilson_x_spectrum']=[[float(z.real),float(z.imag)] for z in np.diag(np.linalg.matrix_power(toron_link,4))]
    toron['Wilson_noncentral_HS_distance_to_nearest_Z4']=float(min(np.linalg.norm(np.linalg.matrix_power(toron_link,4)-np.exp(1j*np.pi*j/2)*np.eye(4)) for j in range(4)))
    archive=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(snapshots.items()):z.writestr(name,raw)
    stable={name:(ROOT/name).read_bytes()==raw for name,raw in snapshots.items()}
    if not all(stable.values()):raise ArithmeticError('Input/source byte drift')
    cphi=perturbed['Cphi'];global_lower=-3*(8*cphi)**(4/3)/(4**(4/3)*(1/64)**(1/3))
    result=dict(schema=1,scope='uniform-central neutral static sea envelope EXACT; controls FLOATING; global geometry inequality OPEN',proof=PROOF,actual_profiles=[dict(role=p['role'],scale=p['scale'],load=p['load'],onsite=str(p['onsite']),paths=[{k:str(v) if isinstance(v,F) else v for k,v in row.items()} for row in p['paths']]) for p in profiles],arbitrary_uniform_controls=arbitrary,sameGram_clifford_controls=saturation,sameGram_max_error=float(np.linalg.norm(gram(phi)-gram(cliff))),noncliff_fixed2_adverse=adverse,noncliff_sameGram_positive=positive,all64_central_exact_span_controls=span,charged_native_two_link_AP_controls=charged,actual_nonuniform_SU4_extension_control=perturbed,actual_flat_noncentral_SU4_toron_control=toron,global_neutral_static_coercive_lower=dict(exact_expression='-3*(8*Cphi)^(4/3)/(4^(4/3)*(1/64)^(1/3))',floating_value=global_lower),input_source_manifest={name:dict(sha256=sha(raw),bytes=len(raw)) for name,raw in sorted(snapshots.items())},input_source_bytes_archive=dict(path=str(archive),sha256=sha(archive.read_bytes())),input_source_unchanged=stable,execution_seconds=time.monotonic()-started,next_decision='Resolve ALL noncentral flat SU4 commuting holonomies and covariantly constant Phi; then prove or falsify the quantified global inequality before any fullPhi ground admission.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.output)
    a=result['noncliff_fixed2_adverse'];p=result['actual_flat_noncentral_SU4_toron_control']
    print(json.dumps(dict(receipt=str(args.output),fixed2_shortcut_deficit=a['global_KyFan128']-a['fixed_top2_per_momentum'],sameGram_saturation_error=result['noncliff_sameGram_positive']['gram_envelope']-result['noncliff_sameGram_positive']['global_KyFan128'],noncentral_flat_toron=dict(alignment=p['alignment_original_HS_squared'],plaquette=p['plaquette_original_defect'],Wilson_distance=p['Wilson_noncentral_HS_distance_to_nearest_Z4']),global_geometry='OPEN'),indent=2))

if __name__=='__main__':main()
