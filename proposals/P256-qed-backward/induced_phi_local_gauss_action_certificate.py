"""Actual CartPhi3840/SU4^192 kinetic pullback on normal5760 x Haar960.

Conditional operator inequality, with independent finite controls. Neither the
finite controls nor this chart calculation admit a global root or D(H^2).
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

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).read_bytes()
CAPTURED: dict[str, bytes] = {Path(__file__).name: SELF}
class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes()
        CAPTURED[Path(self.path).name] = raw
        return self.source_to_code(raw, self.path)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.origin and Path(spec.origin).parent == ROOT and spec.origin.endswith('.py'):
            spec.loader = Loader(fullname, spec.origin)
        return spec
finder = Finder()
sys.meta_path.insert(0, finder)
import induced_phi_nonuniform_gauss_quantum as gauss
import induced_phi_nonuniform_fermionic_source_unitary as source
sys.meta_path.remove(finder)
CAPTURED.update(gauss.SOURCE_BYTES)
CAPTURED.update(source.SOURCE_BYTES)
import numpy as np
from scipy.linalg import expm, expm_frechet
from threadpoolctl import threadpool_limits
import mpmath as mp

sys.set_int_max_str_digits(0)
R = F(1, 1 << 20)
D, Z, N = 5760, 6720, 8324
PINNED = {
    'induced_phi_gaussian_action_certificate_receipt.json': '6abd28b7bae5d3e8370ec66555778c0ce6768c21e8ce90674e42c5e214147f8b',
    'induced_phi_global_normal_gauss_chart_certificate_receipt.json': '9152079e9ba0334a49aff3387e60fc497478453aee252a4c77468060502488e0',
    'induced_phi_global_occupied_kato_derivative_control_receipt.json': '270cdefd1b7bd9d68fcf94388ac60aee3c4a8c7d8ad75de099646c72bbb4714f',
    'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json': 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6',
}
DERIVATION = {
    'primitive_owner': 'OriginalBosonicLaw.full_form_contract in bosonic_original_law_join.py fixes chi sum p_Phi^2/2 and .3chi sum E4^2/2. Coordinate owners: sea_selection.matrix_unit_basis explicitly Tr(BaBb)=4delta; unrestricted_field_minimum.coordinates=Tr(Phi Ba)/4, matrices(u)=sumuBa. induced_phi_uniform_full_normal_quantum.normal_geometry lines140-151 constructs Tgrade=M/sqrt8, S_ab=Tr(Ba Tcart_b)/4, E=S Qcart and verifies E^T E=I45/8. Thus canonical u=S x with S^T S=I60/8 (including orthogonal grade/Cart rotations), NOT HS metricI/2. Changing from original matrix-unit B to Clifford M is orthogonal since both have Tr4. In Clifford coordinates x=sqrt8 u and p_u=sqrt8 p_x. Lebesgue du=8^-1920 dx across CartPhi3840; this constant cannot alter logJ derivatives or normalization. Therefore Tprimitive=chi/2[8 sum3840 p_x^2+(3/10) sum192,15 E_e,a^2]. E=-i invariant SU4 derivative with generators T; product normalized Haar. chi=Lambda^-1/2=2^-4096, Lambda=2^8192; a=1/64 is in existing microscopic coefficients, not another kinetic factor. No HS-orthonormal field law is inferred.',
    'literal_chart': 'r=(s45,f2835,C2880); sum_x f_x=0, slow injection Q0*s/8. F_x=Phi0+Q0*s/8+Qpolar(s)f_x. Qpolar=P(s)Q0[Q0^T P(s)Q0]^-1/2, P=I-O(O^T O)^-1O^T; O is the actual Cart orbit of Phi0+Q0*s/8. Phi_x=Ux F_x Uxdag; Lxy=Ux exp(iTa Ce,a) Uydag, all64 U unrestricted. Qpolar is NOT occupied W256. Whitened body differential Dphi=(dF+[iTa a,F])/sqrt8; Dlink=sqrt(10/3)[J(C)dC+a_x-Ad_exp(iTC)a_y]. Both endpoints and all192 links survive. Left/body angular velocities a=Uxdag dUx/i use invariant Haar, not exp-angle Lebesgue.',
    'complete_metric': 'D is square6720; E=D^-1; M=E E^T partitioned [[A5760,K5760x960],[K^T,B960]]. The physical tangent metric is diag(1/8,10/3). Jdensity=|det D| relative to dr dHaar; any constant from normalized Haar/globalZ4 cancels in derivatives. All derivatives here are in actual normal Cartesian r. Invariant angular fields have zero Haar divergence (SU4 unimodular) and coefficients A,K,B,j=logJ are independent of U, not angular Dirichlet-restricted.',
    'literal_operator': 'On equivariant Psi_phys=Gamma(U) J^-1/2 psi(r), flat-measure primitive kinetic is chi/2[p_i Aij p_j+p_i Ki,alpha Qalpha+Qalpha Kj,alpha p_j+Qalpha Balpha,beta Qbeta]+VJ. Here p=-i partial_r, Q=dGamma(all65 local fundamental Ta)+Ref, with the sign fixed by Gamma and the actual K block (mean K=-d H^-1). VJ=chi/4 partial_i(Aij partial_j j)+chi/8 partial_i j Aij partial_j j. This is an exact differential operator identity, NOT a named price or form-only estimate. Expanding the cross term gives -i chi[K Q partial+(partial K)Q/2]. Noncommuting Qalpha Qbeta remains ordered; symmetric B contracts the anticommutator. No angular derivatives are deleted: Haar-equivariance converts them to the full finite-CAR Noether operator.',
    'sqrtJ_product_rule': 'For L=-chi/(2J) partial_i(J Aij partial_j), partial(J^-1/2 psi)=J^-1/2(partial psi-j_i psi/2). Expanding twice cancels first-order j terms and leaves VJ with PLUS both displayed terms. In the normal/angular cross block the +j_i K Q from J divergence cancels the -j_i K Q from the two square-root derivatives. Therefore no extra j*K*Q scalar is added or omitted. det derivative j_i=Tr6720(E D_i); j_ij=Tr6720(E D_ij-E D_j E D_i), preserving order, trace dimension6720 and all cleared/nonzero branches. E_i=-E D_i E; E_ij=E D_j E D_i E+E D_i E D_j E-E D_ij E. M_i=E_i E^T+E E_i^T and M_ij includes all four second/product terms. sqrtJ_i/sqrtJ=j_i/2; sqrtJ_ij/sqrtJ=j_ij/2+j_i j_j/4.',
    'actual_linear_subtraction': 'At f=C=0, nonzeroq fastPhi Qpolar(s) is orthonormal/orthogonal to O(s), so its cometric is exactly8 and it has zero Phi/colour cross. The q0 slow derivative is Q0, NOT Qpolar(s). Its orbit cross Bslow=Q0^T O(s)/8 vanishes at s0, but is retained offmean: Sslow=I/8-Bslow H^-1 Bslow^T (uniform angular orbit scaling cancels), Aslow=Sslow^-1=8I+O(s^2), with derivative0 at s0. q0 graph d=0 gives no slow/colour cross; Fourier orthogonality gives no slow/fastPhi cross. Therefore A0=diag(8I2880,.3I2880+d H0^-1 ddag), and the FULL derivative A_s at root is supported ONLY colour2880. Define L(s)=sum45 s_l G_l(0), H=O(s)^T O(s)/8, G_l=-d H^-1 H_l H^-1 ddag; G_lm=d[H^-1 H_m H^-1 H_l H^-1+H^-1 H_l H^-1 H_m H^-1-H^-1 H_lm H^-1]ddag. div L=0 since rows/columns are C and coefficients depend only s. The quadratic q0 Schur branch is NEVER cleared: it lies in full A2 remainder. Tref=chi/2 p(A0+L)p; actual Xi=.5(dK-A dG A), ALL64q/canonical36 real groups, fixed root A/Xi must pay its nonzero linear forcing. This artifact does not independently pay that retained operator.',
    'weighted_remainder_proof': 'Delta=A-A0-L. Integrate first from (s,0) to(s,f,C), then Taylor the ACTUAL full A(s,0), including q0 slow Schur branch: ||Delta||op<=a1 rfast+a2 r^2/2. Each partial_i Delta has norm<=a1+a2 R. Frobenius contraction |Delta:Hess psi|<=sqrt5760 ||Delta||op ||Hess psi||F; triangle over5760 derivative rows gives ||divDelta||<=5760(a1+a2 R). Hence normal remainder/chi <=[sqrt5760(a1 Hfast+a2 H2/2)+5760(a1+a2 R) gradient]/2. Crucially Hfast=||rfast Hess psi||2 and H2=||rALL^2 Hess psi||2, NOT R times a huge unweighted Hessian.',
    'full_Noether_bound': 'On fixed totalnumber n8324, each particle occupies one site and its fundamental generators have sum15 Ta^2=15/8 I. Including Ref, Cauchy over n gives sum960 Qalpha^2<=n^2(15/8)I, ||Q-vector||<=n sqrt(15/8)<2n. This covers arbitrary occupations/all species, both Ref4/active128 and Ref3/active129, since 128*65+4=8324=128+129+8064+3. Common globalZ4 acts i^n=1; independent site centres alter adjacent links and are NOT a quotient. E0=.5 sum c_active0,a^dag R_a is a local singlet and conserves n; [Qalpha,E0]=0 exactly. Charge neutrality is not Q=0 and does not remove the local angular action.',
    'cross_and_angular_norm': '||K||<=m0, ||partial K||<=a1 from full M; cross/chi <=(2n)[2m0 gradient+5760 a1]/2 for a normalized scalar packet times any fixed fibre vector. Since B is positive with ||B||<=m0, ||Q^T B Q||<=m0 n^2 CF, angular/chi<=m0 n^2(15/8)/2. VJ/chi<=[5760 a1 jg+5760 m0 jh]/4+m0 jg^2/8. The factors5760 here are normal divergences; the factors6720 in jg/jh are full determinant traces. These are operator-norm bounds, not expectation estimates.',
    'Gaussian_products': 'Use ONLY outward derived Gaussian receipt bounds. v=(1-P)f_A with fixed root A,Xi, exact full M=A_CC^-1 A_CP, canonical real joint covariance C=.5A^-1. Gaussian g=sqrt(.5TrA); gp/hp are full covariance unwhitening bounds for degree3 P. Weighted corrected Hessians use actual joint Wick tensors and degree3 hypercontractivity from supplier. For any polynomial tensor Q of degree k, ||P Q||2<=3^((3+k)/2) p ||Q||2. Frobenius tensor xF tensor Hess(v) and x tensor x tensor Hess(v) give radial weights without loss of physical powers. Moment ||r||2k<=((2k-1)!!)^(1/(2k)) sqrt(Tr C) follows eigenbasis Minkowski, with trace on ALL5760. No adiabatic f_A(s) is substituted.',
    'compact_b6': 'b6=(1-r^2/R^2)^6_+; Db=-12 x/R^2 (1-r^2/R^2)^5; D2b=-12 I/R^2(1-r^2/R^2)^5+120 xx^T/R^4(1-r^2/R^2)^4. Exact Delta b=-12*5760/R^2 t^5+120 r^2/R^4 t^4. Corrected weighted gradient/Hessians below apply full Leibniz to b*v using these expressions. ||(1-b)f||2<=sqrt(12 TrC/R^2) because (1-b)^2<=1-b^2<=12r^2/R^2 globally. Thus ||b v||>=1-p-sqrt(12TrC/R^2)>1/2, Z>=1/4, normalization norm<=2. This is an upper bound on the actual normalization, not replacing it by2. Boundary vanishing through derivative5 implies compact C5 zero extension and local H4 in regular chart; support/collar is proven by chart receipt. Global operator D(H^2) remains parent-owned.',
    'occupied_transport_separation': 'For actual real occupied W on3R collar, define B_i=-i Wdag partial_i W and Qtilde=Wdag Q W. Conjugate the literal operator by W: replace p by p+B, Q by Qtilde, preserve ordering and derivatives of Qtilde; scalar VJ unchanged. This is the exact connection addition, NOT an artificial complex W. Equivalently apply the unconjugated operator to W(r) v(r); all DU/D2U Leibniz terms can be priced from real Fock full5760 gradient/Hessian receipt. Parent owns that separate addition and no Kato amount is added to the geometric total here. No assumption Qtilde is constant/commutes with p is made.',
    'support_collar': 'Use derived global normal chart input on R and3R: h>1/20,w<1/3, reachCartesian>=1/32, connected N4 commonZ4 only, injection discrepancy<=18432rho<1. Exact polar complex ball1/256 proves actual Q1/Q2/Q3 on3R, intrinsic body D1/D2 uniform over allHaar. D0 inverse<=54, 54 D1 rho<1/4, D^-1<=108. E/M/determinant formulas above give all constants anew from actual D, not local_gauss_prices. No hard-wall operator or angular support reduction is introduced.',
    'limits': 'Conditional on actual captured N4 root rectangle, exact Clifford normal recipe, static positivity/sea bounds and outward Gaussian covariance/corrector proof. No root solver or old producer is run. Potential, full SOURCE residual, original SOURCE, ground, photon pole, current, cone and global domain adoption are not conclusions.',
}

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def prices(gaussian: dict[str, Any], chart: dict[str, Any]) -> dict[str, Any]:
    iv = gauss.iv
    iv.dps = 90
    ex, ep = gauss.exact, gauss.endpoint
    pack = source.pack
    def up(name: str) -> Any:
        return ex(F(gaussian['outward_derived_certificate']['bounds'][name]['upper']))
    p, tc, ta = up('corrector_L2'), up('full_trace_covariance'), up('full_trace_A')
    gp = up('corrector_gradient')
    g = iv.sqrt(ta/2)
    radius = ex(R)
    # Proven lower1/2 norm; do not discard tiny terms by rounding Z to1.
    defect = p+iv.sqrt(12*tc/radius**2)
    if ep(defect,1) >= F(1,2):
        raise ArithmeticError('Actual compact corrector normalizer lower fails')
    nv = 1+p
    gradv = (1+9*p)*g+gp
    # Full weighted products dominate their fast-weighted counterparts.
    r2v = iv.sqrt(ex(3))*tc*(1+27*p)
    r2gradv = 9*tc*g*(1+81*p)+9*gp*iv.sqrt(ex(3))*tc
    r3gradv = 256*tc**ex(F(3,2))*((1+81*p)*g+gp)
    # Use ALL trace upper for rf bump terms only; the large leading Hfast uses
    # its actual fast Wick trace, preserving2^-3584 rather than2^-3072.
    rfv = iv.sqrt(tc)*(1+9*p)
    rfr2v = iv.sqrt(ex(15))*tc**ex(F(3,2))*(1+27*p)
    r4v = iv.sqrt(ex(105))*tc**2*(1+243*p)
    grad = 2*(gradv+12/radius**2*iv.sqrt(tc)*(1+9*p))
    hf = 2*(up('corrected_rfast_Hessian')+24/radius**2*r2gradv
            +12*iv.sqrt(ex(D))/radius**2*rfv+120/radius**4*rfr2v)
    h2 = 2*(up('corrected_rALL_squared_Hessian')+24/radius**2*r3gradv
            +12*iv.sqrt(ex(D))/radius**2*r2v+120/radius**4*r4v)
    result: dict[str, Any] = {}
    for label in ('support', 'collar'):
        c = chart['exact_bounds'][label]
        rho, v, a, b = (F(c[k]) for k in ('radius','inverse_upper','D1_upper','D2_upper'))
        if 54*a*rho >= F(1,4) or 18432*rho >= 1:
            raise ArithmeticError('Actual chart collar loses its margin')
        m0, a1, a2 = v*v, 2*v**3*a, 6*v**4*a*a+2*v**3*b
        jg, jh = Z*v*a, Z*(v*b+v*v*a*a)
        normal = (iv.sqrt(ex(D))*(ex(a1)*hf+ex(a2)*h2/2)+ex(D*(a1+a2*R))*grad)/2
        cross = ex(2*N)*(2*ex(m0)*grad+ex(D*a1))/2
        angular = ex(m0*N*N*F(15,16))
        volume = ex(F(D*a1*jg+D*m0*jh,4)+F(m0*jg*jg,8))
        components = dict(normal_metric_remainder=normal,full_Noether_cross=cross,
                          full_fixednumber_angular=angular,sqrtJ_potential=volume)
        total = sum(components.values())
        result[label] = dict(operator_constants_exact={k:str(x) for k,x in dict(
            radius=rho,inverse=v,D1=a,D2=b,cometric=m0,cometric_D1=a1,
            cometric_D2=a2,logJ_gradient=jg,logJ_Hessian=jh).items()},
            components_divided_chi={k:pack(x) for k,x in components.items()},
            geometric_remainder_norm_upper=pack(ex(F(2)**-4096)*total),
            geometric_remainder_divided_chi_upper=pack(total))
    # Optional flat-reference cutoff commutator, kept OUTSIDE geometric total.
    c = result['support']['operator_constants_exact']
    ref = ex(F(c['cometric'])+F(c['cometric_D1'])*R)
    rgradv = (1+27*p)*iv.sqrt(tc*ta/2+ex(F(D,2)))+3*gp*iv.sqrt(tc)
    bump = 2*ex(F(2)**-4096)*ref*(6*ex(D)/radius**2*nv+60/radius**4*r2v+12/radius**2*rgradv)
    return dict(cases=result,weighted_compact_scalar_packet=dict(
        gradient_upper=pack(grad),rfast_Hessian_Frobenius_upper=pack(hf),
        rALL_squared_Hessian_Frobenius_upper=pack(h2),normalizer_squared_lower='1/4',
        normalization_operator_upper='2',normalizer_defect_upper=pack(defect)),
        separate_reference_b6_commutator_norm_upper=pack(bump),
        separate_reference_b6_not_in_geometric_sum=True,
        retained_reference_operator='chi/2 p [G(0)+sum45 s_l dG_l(0)] p',
        occupied_connection_added_to_total=False,
        dimensions=dict(normal=D,slow=45,fast=5715,CartPhi=3840,links=192,Haar=960,physical=Z))

class PhysicalChart:
    """Full6720 body operator; block determinant plus exact rank45 update.

    Orthogonal site coordinates t_x=f_x+s/8 replace canonical real Fourier
    coordinates only computationally; trace/norm/determinant magnitudes agree.
    """
    def __init__(self, g: list[float]):
        full = gauss.stationary.full
        self.phi0 = np.sqrt(np.asarray(g))[:,None,None]*full.supplier.primitive.SELECTED
        self.geom = full.normal_geometry({'frame':self.phi0,'a':1.,'g0':g[0],'gs':g[1]},[full.exact(F(str(x))) for x in g])
        self.T = self.geom['M']/np.sqrt(8)
        self.q0 = self.geom['Q']
        sites = [(i,j,k) for i in range(4) for j in range(4) for k in range(4)]
        index = {site:x for x,site in enumerate(sites)}
        self.ends = [(x,index[tuple((np.asarray(site)+np.eye(3,dtype=int)[axis])%4)]) for x,site in enumerate(sites) for axis in range(3)]
        self.ad = np.asarray([self.adjoint(1j*t) for t in self.T])
    def adjoint(self, x: np.ndarray) -> np.ndarray:
        return 2*np.einsum('aij,bji->ab',self.T,x@self.T-self.T@x).real
    def coefficients(self, r: np.ndarray, frozen: bool = False) -> dict[str, Any]:
        t, c = r[:2880].reshape(64,45), r[2880:].reshape(192,15)
        s = t.sum(axis=0)/8
        f = t-s/8
        slow = np.einsum('a,amij->mij',s,self.geom['deltas'])/8
        frame = self.phi0+slow
        q, _ = source.adapted_jet(frame,np.zeros_like(frame),self.T,self.q0)
        qd = np.asarray([source.adapted_jet(frame,d/8,self.T,self.q0)[1] for d in self.geom['deltas']])
        fields = frame[None]+np.einsum('xma,aij->xmij',(f@q.T).reshape(64,4,15),self.T)
        orbit = np.asarray([gauss.orbit(x,self.T) for x in fields])
        base = np.concatenate((np.broadcast_to(q,(64,60,45)),orbit),axis=2)/np.sqrt(8)
        bi = np.linalg.inv(base)
        update = np.broadcast_to((self.q0-q)/8,(64,60,45)).copy()
        if not frozen:
            update += np.einsum('jab,xb->xaj',qd,f)
        update /= np.sqrt(8)
        bu = bi@update
        small = np.eye(45)+bu[:,:45].sum(axis=0)/8
        si = np.linalg.inv(small)
        jlink, adjlink = [], []
        for z in c:
            az = np.einsum('a,aij->ij',z,self.ad)
            # J(C)=integral exp(t ad_C)dt, evaluated by an entire series.
            term = np.eye(15); j = term.copy()
            for k in range(1,18):
                term = term@az/(k+1)
                j += term
            jlink.append(j)
            adjlink.append(expm(az))
        jl, al = np.asarray(jlink), np.asarray(adjlink)
        logj = sum(np.linalg.slogdet(x)[1] for x in base)+np.linalg.slogdet(small)[1]
        logj += sum(np.linalg.slogdet(x)[1]+15*np.log(10/3)/2 for x in jl)
        return dict(q=q,qd=qd,fields=fields,base=base,bi=bi,update=update,bu=bu,si=si,
                    small=small,jl=jl,ji=np.linalg.inv(jl),al=al,logj=float(logj))
    def forward(self, z: np.ndarray, c: dict[str, Any]) -> np.ndarray:
        t, links, a = z[:2880].reshape(64,45),z[2880:5760].reshape(192,15),z[5760:].reshape(64,15)
        phi = np.einsum('xij,xj->xi',c['base'],np.concatenate((t,a),axis=1))
        phi += np.einsum('xij,j->xi',c['update'],t.sum(axis=0)/8)
        link = np.einsum('eij,ej->ei',c['jl'],links)
        for e,(x,y) in enumerate(self.ends):
            link[e] += a[x]-c['al'][e]@a[y]
        return np.concatenate((phi.ravel(),(link*np.sqrt(10/3)).ravel()))
    def inverse(self, z: np.ndarray, c: dict[str, Any]) -> np.ndarray:
        phi, link = z[:3840].reshape(64,60),z[3840:].reshape(192,15)
        v = np.einsum('xij,xj->xi',c['bi'],phi)
        slow = c['si']@(v[:,:45].sum(axis=0)/8)
        v -= np.einsum('xij,j->xi',c['bu'],slow)
        a = v[:,45:]
        rhs = link/np.sqrt(10/3)
        for e,(x,y) in enumerate(self.ends):
            rhs[e] -= a[x]-c['al'][e]@a[y]
        links = np.einsum('eij,ej->ei',c['ji'],rhs)
        return np.concatenate((v[:,:45].ravel(),links.ravel(),a.ravel()))
    def inverse_transpose(self, p: np.ndarray, c: dict[str, Any]) -> np.ndarray:
        pt, pc, pa = p[:2880].reshape(64,45),p[2880:5760].reshape(192,15),p[5760:].reshape(64,15).copy()
        pl = np.einsum('eji,ej->ei',c['ji'],pc)
        for e,(x,y) in enumerate(self.ends):
            pa[x] -= pl[e]
            pa[y] += c['al'][e].T@pl[e]
        y = np.concatenate((pt,pa),axis=1).copy()
        correction = c['si'].T@np.einsum('xij,xi->j',c['bu'],y)
        y[:,:45] -= correction/8
        pp = np.einsum('xji,xj->xi',c['bi'],y)
        return np.concatenate((pp.ravel(),(pl/np.sqrt(10/3)).ravel()))

def finite_controls(g: list[float]) -> dict[str, Any]:
    rng = np.random.default_rng(2221011)
    model = PhysicalChart(g)
    def unit(n: int) -> np.ndarray:
        x = rng.normal(size=n)
        return x/np.linalg.norm(x)
    basis = gauss.stationary.full.supplier.primitive.matrix_unit_basis()
    canonical_map = np.einsum('bij,aji->ba',basis,model.T).real/4
    canonical_metric_error = float(np.linalg.norm(canonical_map.T@canonical_map-np.eye(15)/8))
    assert canonical_metric_error<1e-13
    assert np.linalg.norm(canonical_map.T@canonical_map-np.eye(15)/2)>.1
    r = float(R)/2*unit(D)
    c = model.coefficients(r)
    vel, mom = unit(Z), unit(Z)
    out = model.forward(vel,c)
    recovered = model.inverse(out,c)
    dual = model.inverse_transpose(mom,c)
    inverse_error = float(np.linalg.norm(recovered-vel))
    transpose_error = float(abs(mom@vel-dual@out))
    assert inverse_error<2e-12 and transpose_error<2e-12
    # Independent laboratory-frame primitive field/link differential, with
    # arbitrary64 Haar bases and noncommuting192 link tangent directions.
    dag = lambda x: x.conj().swapaxes(-1,-2)
    cart = lambda x: 2*np.einsum('aij,...ji->...a',model.T,x).real
    t, dt = r[:2880].reshape(64,45), vel[:2880].reshape(64,45)
    s, ds = t.sum(axis=0)/8, dt.sum(axis=0)/8
    f, df = t-s/8, dt-ds/8
    frame = model.phi0+np.einsum('a,amij->mij',s,model.geom['deltas'])/8
    sd = np.einsum('a,amij->mij',ds,model.geom['deltas'])/8
    q, dq = source.adapted_jet(frame,sd,model.T,model.q0)
    fd = sd[None]+np.einsum('xma,aij->xmij',(df@q.T+f@dq.T).reshape(64,4,15),model.T)
    angles = vel[5760:].reshape(64,15)
    us = np.asarray([expm(1j*np.einsum('a,aij->ij',2*unit(15),model.T)) for _ in range(64)])
    uds = us@(1j*np.einsum('xa,aij->xij',angles,model.T))
    fields = c['fields']
    labphi = uds[:,None]@fields@dag(us)[:,None]+us[:,None]@fd@dag(us)[:,None]+us[:,None]@fields@dag(uds)[:,None]
    bodyphi = cart(dag(us)[:,None]@labphi@us[:,None])/np.sqrt(8)
    bodylinks = []
    for e,(x,y) in enumerate(model.ends):
        ce = r[2880:].reshape(192,15)[e]
        dce = vel[2880:5760].reshape(192,15)[e]
        el, eld = expm_frechet(1j*np.einsum('a,aij->ij',ce,model.T),1j*np.einsum('a,aij->ij',dce,model.T))
        link = us[x]@el@dag(us[y])
        ld = uds[x]@el@dag(us[y])+us[x]@eld@dag(us[y])+us[x]@el@dag(uds[y])
        bodylinks.append(cart(-1j*dag(us[x])@ld@dag(link)@us[x])*np.sqrt(10/3))
    laboratory_error = float(np.linalg.norm(np.concatenate((bodyphi.ravel(),np.asarray(bodylinks).ravel()))-out))
    assert laboratory_error<2e-12
    # This is the full6720 dual kinetic form, not a normal/angular subset.
    primitive = float(dual@dual/2)
    wrong_metric = float((8*dual[:3840]@dual[:3840]+.3*dual[3840:]@dual[3840:])/2)
    frozen = model.coefficients(r,True)
    frozen_error = float(np.linalg.norm(model.forward(vel,c)-model.forward(vel,frozen)))
    # Slow derivatives of actual normal frame must not be frozen away.
    polar_term = float(np.linalg.norm(c['update']-frozen['update']))
    assert abs(primitive-wrong_metric)>1e-3 and polar_term/float(R)>1e-3
    # The fixed Q0 slow branch has a genuinely nonzero offmean Schur remainder.
    # Magnification is a finite structural control outside the proof collar.
    magnified = np.zeros(D)
    magnified[:2880] = np.tile(1e-3*unit(45)/8,64)
    slow_covector = np.zeros(Z)
    slow_covector[:2880] = np.tile(unit(45)/8,64)
    magnified_coefficients = model.coefficients(magnified)
    slow_dual = model.inverse_transpose(slow_covector,magnified_coefficients)
    q0_schur_error = float(abs(slow_dual@slow_dual-8))
    assert q0_schur_error>1e-9
    # Full mean operator independently reconstructed in every Fourier symbol.
    o = gauss.orbit(model.phi0,model.T)
    h = o.T@o/8
    hi = np.linalg.inv(h)
    qrows = []
    for q in gauss.QINTEGER:
        d = 1-np.exp(2j*np.pi*np.asarray(q)/4)
        incidence = np.kron(d[:,None],np.eye(15))
        dm = np.zeros((105,105),complex)
        dm[:60,:45]=model.q0/np.sqrt(8)
        dm[:60,90:]=o/np.sqrt(8)
        dm[60:,45:90]=np.sqrt(10/3)*np.eye(45)
        dm[60:,90:]=np.sqrt(10/3)*incidence
        inv = np.linalg.inv(dm.conj().T@dm)
        expected = .3*np.eye(45)+incidence@hi@incidence.conj().T
        err = max(float(np.linalg.norm(inv[:45,:45]-8*np.eye(45))),
                  float(np.linalg.norm(inv[45:90,45:90]-expected)),
                  float(np.linalg.norm(inv[45:90,90:]+incidence@hi)),
                  float(np.linalg.norm(inv[90:,90:]-hi)))
        assert err<2e-10
        qrows.append(dict(q=list(q),dimension=105,full_mean_cometric_defect=err))
    # Independent measure/sqrtJ control along an actual uniform normal direction.
    # Five-point derivative is a finite control only; endpoints0.0002 are outside
    # the proof collar, expressly NOT used to justify any interval constant.
    direction = np.zeros(D)
    direction[:2880] = np.tile(np.eye(45)[0]/8,64)
    step = 1e-4
    logs = [model.coefficients(k*step*direction)['logj'] for k in (-2,-1,0,1,2)]
    jm2,jm,j0,jp,jp2 = logs
    j1 = (jm2-8*jm+8*jp-jp2)/(12*step)
    j2 = (-jp2+16*jp-30*j0+16*jm-jm2)/(12*step**2)
    # At r0 the exact coefficient contraction in this direction is8 and its
    # directional derivative is0. This exercises the directional product rule,
    # NOT all mixed divergences or an action on a one-dimensional restriction.
    beta = 1.3
    w = np.asarray([np.exp(-beta*(k*step)**2/2-(j-j0)/2) for k,j in zip((-2,-1,0,1,2),logs,strict=True)])
    wp = (w[0]-8*w[1]+8*w[3]-w[4])/(12*step)
    wpp = (-w[4]+16*w[3]-30*w[2]+16*w[1]-w[0])/(12*step**2)
    direct = -4*(wpp+j1*wp)
    potential = 2*j2+j1*j1
    expected = 4*beta+potential
    measure_defect = abs(direct-expected)
    assert measure_defect<1e-4 and abs(potential)>1e-3
    # Actual fundamental local singlet commutator on8 modes in fixednumber4.
    masks = [m for m in range(256) if m.bit_count()==4]
    idx = {m:i for i,m in enumerate(masks)}
    def bilinear(a: int,b: int) -> np.ndarray:
        z = np.zeros((70,70),complex)
        for col,m in enumerate(masks):
            if not(m>>b&1): continue
            n=m^(1<<b)
            if n>>a&1: continue
            sign=(-1)**((m&((1<<b)-1)).bit_count()+(n&((1<<a)-1)).bit_count())
            z[idx[n|(1<<a)],col]=sign
        return z
    zero = np.zeros((70,70),complex)
    e0 = sum((bilinear(a,a+4) for a in range(4)),zero)/2
    qs = [sum((t[a,b]*(bilinear(a,b)+bilinear(a+4,b+4)) for a in range(4) for b in range(4)),zero) for t in model.T]
    singlet = max(float(np.linalg.norm(q@e0-e0@q)) for q in qs)
    bulk_q = [sum((t[a,b]*bilinear(a,b) for a in range(4) for b in range(4)),zero) for t in model.T]
    wrongcharge = max(float(np.linalg.norm(q@e0-e0@q)) for q in bulk_q)
    assert singlet<1e-12 and wrongcharge>.1
    # Common centre identity versus site-specific centre altering all incident links.
    links = np.asarray([expm(1j*np.einsum('a,aij->ij',z,model.T)) for z in r[2880:].reshape(192,15)])
    centres = np.ones(64,complex)
    centres[0] = 1j
    bad_links = np.asarray([centres[x]*e*centres[y].conjugate() for e,(x,y) in zip(links,model.ends,strict=True)])
    centre_error = np.linalg.norm(bad_links-links)
    assert centre_error>1
    # Integer matrix powering stays in the exact Gaussian-integer centre;
    # scalar complex ** uses a trigonometric float path for large exponents.
    central = 1j*np.eye(4)
    assert np.array_equal(np.linalg.matrix_power(central,N),np.eye(4))
    assert not np.array_equal(np.linalg.matrix_power(central,N+1),np.eye(4))
    return dict(root_centre_display=g,normal_dimensions=D,physical_dimensions=Z,
        physical_inverse_error=inverse_error,full_dual_transpose_identity_error=transpose_error,
        full_laboratory_body_primitive_differential_error=laboratory_error,
        unrestricted64_Haar_bases_exercised=True,all192_link_endpoints_exercised=True,
        original_canonical_basis_metric_error=canonical_metric_error,
        wrong_HS_normalized_Phi_metric_rejected=True,
        retained_q0_slow_Schur_finite_error=q0_schur_error,
        q0_schur_control_radius=1e-3,q0_schur_control_outside_proof_collar=True,
        full_kinetic_dual_form=primitive,wrong_doubleweighted_metric_form_error=abs(primitive-wrong_metric),
        polar_derivative_term_norm=polar_term,wrong_frozen_polar_full_D_error=frozen_error,
        full64_mean_symbols=qrows,full_log_density_at_supported_point=c['logj'],
        full_density_trace_dimensions=dict(Phi_block=3840,link_block=2880,rank_update=45,total=Z),
        directional_sqrtJ_product_rule_defect=measure_defect,wrong_omitted_directional_measure_term_error=abs(potential),
        actual_slow_logJ_gradient_finite=j1,actual_slow_logJ_Hessian_finite=j2,
        measure_control_scope='Full6720 determinant, one directional product-rule contraction only; full mixed divergence is proved algebraically, not finite-certified',
        measure_FD_endpoints_outside_proof_collar=True,measure_control_step=step,
        exact_local_source_singlet_commutator_defect=singlet,wrong_omitted_Ref_charge_commutator_error=wrongcharge,
        commonZ4_n8324_identity=True,wrong_n8325_commonZ4_rejected=True,
        independent_siteZ4_incident_link_Frobenius_error=float(centre_error),
        status='Finite controls on actual full operator; NOT root/interval/global certification')

def run() -> dict[str, Any]:
    start = time.monotonic()
    inputs = {name:(ROOT/name).read_bytes() for name in PINNED}
    for name,raw in inputs.items():
        if sha(raw)!=PINNED[name]: raise ArithmeticError('Pinned input changed: '+name)
    loaded = {name:json.loads(raw) for name,raw in inputs.items()}
    gaussian = loaded['induced_phi_gaussian_action_certificate_receipt.json']
    chart = loaded['induced_phi_global_normal_gauss_chart_certificate_receipt.json']
    supplier = loaded['induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json']
    pars = supplier['input_descriptor']['exact_NEW_allq_BO_mean']['parameters']
    g0,gs = (F(pars[k]['lower']) for k in ('g0','gs'))
    if min(g0,gs)/2<=F(1,20) or (F(pars['g0']['upper'])+3*F(pars['gs']['upper']))/2>=F(1,3):
        raise ArithmeticError('Actual N4 rectangle violates root/chart hypothesis')
    with threadpool_limits(limits=1):
        controls = finite_controls([float(g0),float(gs),float(gs),float(gs)])
    priced = prices(gaussian,chart)
    kato = loaded['induced_phi_global_occupied_kato_derivative_control_receipt.json']
    connection = {r['domain']:{k:r[k] for k in ('radius','full5760_gradient_upper','full5760_Hessian_Frobenius_upper')} for r in kato['conditional_cases']}
    if not all((ROOT/name).read_bytes()==raw for name,raw in inputs.items()):
        raise ArithmeticError('Input epoch changed during execution')
    if not all((ROOT/name).read_bytes()==raw for name,raw in CAPTURED.items()):
        raise ArithmeticError('Source epoch changed during execution')
    bundle = ROOT/('induced_phi_local_gauss_action_certificate_'+sha(SELF)[:16]+'_input_bytes.zip')
    captured_bytes = {**inputs,**CAPTURED}
    if bundle.exists():
        with zipfile.ZipFile(bundle,'r') as archive:
            if set(archive.namelist())!=set(captured_bytes) or any(archive.read(name)!=raw for name,raw in captured_bytes.items()):
                raise ArithmeticError('Immutable capsule content conflicts with this epoch')
    else:
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as archive:
            for name,raw in captured_bytes.items():
                info=zipfile.ZipInfo(name,date_time=(2026,10,10,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,raw)
    return dict(status='DERIVED_ACTUAL_PRIMITIVE_KINETIC_PULLBACK_CONDITIONAL_NORM_NOT_SOURCE_ADMISSION',
        governing_plan='issue-222-induced-photon-delta13-continuation-20261008',
        captured_assignment_revision='92c5eb8e5cd492763b25ab20238d14440c757d50f9ae5207b29f3342362bb8da',
        governing_revision='ee8f864c6f0d1b368c3ae52a8573855c5fe0001d19c89e828248aaf74e4c9fc7',
        template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        literal_operator_and_proofs=DERIVATION,actual_derived_prices=priced,finite_controls=controls,
        root_parameters_exact=pars,real_occupied_connection_inputs_NOT_ADDED=connection,
        input_sha256={name:sha(raw) for name,raw in inputs.items()},
        source_at_execution={name:dict(sha256=sha(raw),bytes_utf8=raw.decode()) for name,raw in CAPTURED.items()},
        immutable_input_source_bundle=dict(path=bundle.name,sha256=sha(bundle.read_bytes())),
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__),
        elapsed_seconds=time.monotonic()-start)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),status=result['status'],
        own_source_sha256=sha(SELF),geometric_support=result['actual_derived_prices']['cases']['support']['geometric_remainder_norm_upper'],
        finite_controls={k:v for k,v in result['finite_controls'].items() if k!='full64_mean_symbols'},
        elapsed_seconds=result['elapsed_seconds']),indent=2))
