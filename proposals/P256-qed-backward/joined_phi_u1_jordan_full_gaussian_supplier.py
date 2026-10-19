"""NEW full64/PH90 Jordan Gaussian and compact neutral-Pplus preparation.

Exact source-defined coefficients and conditional positive-pencil statements are
separate from exponent-graded Hermitian-part diagnostics of archived binary data.
No root, original SOURCE, old corrector, or Hamiltonian parameter is adopted.
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
import zipfile

import mpmath as mp
import numpy as np
from scipy.linalg import solve_sylvester
from threadpoolctl import threadpool_limits
import joined_phi_u1_matrix_jordan_supplier as jordan

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_jordan_full_gaussian_'
SELF = Path(__file__).read_bytes()
sys.set_int_max_str_digits(0)
PINNED = {
    'joined_phi_u1_neutral_ground_graph_epoch2_20261010.npz': '2b6fbe41a6c60f673c4d69374e87ec68f800e398037b2fc1e3ec9e277a75c509',
    'joined_phi_u1_neutral_ground_graph_epoch2_20261010.json': 'b404da2a7d99eb920905c947d37550e236d3fb54a0503870058bcf369d6e36fe',
    'joined_phi_u1_matrix_jordan_supplier.py': 'a4e5e4eee0c84c8c72b9ed3fae0be2042bc8031336cba73c94dfc8d9d8b96420',
    'joined_phi_u1_matrix_jordan_supplier_receipt_20261010.npz': 'e59b152a390f413b84bb5ea6d3d9578478b997e2356c5bae4b513de2678ce94e',
}
INPUTS = (
    'joined_phi_u1_matrix_jordan_supplier_receipt_20261010.json',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json',
    'induced_phi_global_normal_gauss_chart_certificate_receipt.json',
    'induced_phi_global_normal_h_jet_control_receipt.json',
    'induced_phi_global_occupied_kato_derivative_control_receipt.json',
    'induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json',
)
PROOF = {
    'source_coefficients': 'Exact input means the algebraic native source recipe, NOT a binary eigensolver matrix: T=M/sqrt8, Q the Clifford normal recipe, g the specified matching descriptor, AP phases exp(i*pi*(2k+1)/4), periodic q phases exp(i*pi*q/2), and exact profile coefficients. h_s(k)=sum_mu x_s,mu(k)*sqrt(g_mu)*Gamma_mu and epsilon_s(k)^2=sum_mu g_mu*x_s,mu(k)^2. Thus Pminus=(I-h/epsilon)/2, Pplus=(I+h/epsilon)/2. The neutral physical -Lambda h occupies Pplus; charged +h occupies Pminus. All128 occupied and128 empty states and all64 independent charged species remain.',
    'exact_PH90': 'For the source-ordered first vertices V_a(k+q,k) and tied second vertices D_ab[-q,+q], sum Tr(Pocc D_ab) minus BOTH q and -q PH Gram matrices divided by epsilon(k)+epsilon(k+q), with profile sign/load/physical scale retained. Equivalently the second derivative of the algebraic finite rank128 occupied energy. This derivation makes K1 and Kunit Hermitian and q-conjugate exactly; no Hermitian projection is part of the scientific law. D_ab retains both endpoints and each ordered one/two-link path. Arrays of the old receipt are observations of these formulas, not certified enclosures of them.',
    'structural_fast_zeros': 'At q0, sum_x f_x=0 and horizontal Q*O=0; fastPhi alignment qhat^2/40 is exactly0 and its colour cross is exactly0. Ordered plaquette has curl(q0)=0. Hence Kfast(q0)=diag(0_45,I3 tensor H/5), with H=O*O/8. G0(q0)=diag(8I45,3I45/10). These are algebraic identities; rounded O(1e-15) entries cannot replace either zero.',
    'exact_Riccati': 'KJ=2^10240 Kfast+2^8192(Kprimitive+Ksea_neutral+D35)+Ksea_active+63Ksea_aux, G=2^-4096 G0. A=G^-1/2 (G^1/2 KJ G^1/2)^1/2 G^-1/2, C=(1/2) A^-1, NOT the inverse square root. A G A=KJ. Every q uses FULL90, every PC/CP coefficient remains. The real direct sum consists of8 selfinverse90 and28 paired180 sectors, dimension5760,36 real-pair sectors.',
    'scales': 'For q!=0 write t=2^-1024, B=sqrtG0(Kfast+t^2 K1J+t^10 Kunit)sqrtG0; A=2^7168 invsqrtG0 sqrt(B) invsqrtG0 and C=2^-7168 sqrtG0 B^-1/2 sqrtG0/2. Here B^-1/2 DOES denote inverse matrix square root. At q0 sqrt(B) starts diag(t sqrt(B1PP),sqrt(BfastCC)); physical APP~2^6144, ACC~2^7168, APC~2^5120, M=ACC^-1 ACP~2^-2048 and CCP~2^-8192. No physical parameter is rescaled or adopted; exponent separation is only an evaluation representation.',
    'graded_evaluation': 'All coefficients of R(t)^2=B(t) through degree12 are solved with full90 square-root Sylvester equations. q0 uses its exact45-dimensional kernel ONLY to initialize a FULL90 coupled series, then every PC/CP and Kunit coefficient enters. A full block Schur inversion supplies Laurent coefficients of C and M, not a principal45 replacement. A(q) and C(q) above are the exact analytic functions; saved finite series are labelled approximate diagnostics with coefficientwise residuals, not exact physical covariances or an interval proof.',
    'order': 'For the SAME exact positive G and same exact SPD KGram, D35>=0 gives AJ>=Aold and 0<CJ<=Cold. Therefore every fullPSD B weight Tr(B CJ)<=Tr(B Cold). This does not order signed PC entries or M. Numerical covariance/PSD-weight displays below exercise only the explicit Hermitian-part diagnostic law and have their own rounding tolerance; no signed-M bound or old action/Hermite coefficient is transferred.',
    'new_preparation': 'For exact conditional full A define f_A=det(A/pi)^1/4 exp(-r^T A r/2) on ALL5760 real coordinates. Choose b6=(1-|r|^2/R^2)^6_+, R=2^-20, Z=int b6^2 |f_A|^2 dr, and psi=Z^-1/2 b6 f_A. Z is the actual nonzero integral; 2 is ONLY a normalization bound. In the injective full normal5760 x unrestricted normalized Haar960 chart use J^-1/2 Gamma(U) [psi(r) tensor neutral wedge(Pplus(r)) tensor v_body(Theta)]. The charged v_body is arbitrary in fixed canonical128 per species/Ref4 (source129/Ref3 separately); no charged Theta-following Slater is introduced. Direct unitary lift W=[PP0+(1-P)(1-P0)] [I-(P-P0)^2]^-1/2 sends P0=Pplus(0) to Pplus(r). This is a genuinely NEW normalized compact Gaussian preparation with no inherited corrector. HE commutes with T because neutral charge0 and W/psi are Theta-independent. HC commutes with T because only neutral occupied transport is used and local Gamma(U) preserves every charged rho_x.',
    'domain': 'The fullHaar/commonZ4 quotient has totalN8324 includingRef. b6 has zero traces through order5; the neutral real-gapped lift and chart are smooth on3R. Zero gluing gives localH4, Hpsi has zero traces through3, and global positive-form approximation proves psi in D(H_F^2), exactly as the source-independent boundary argument, not a hard-wall operator. The Jordan quartic shares the coercive global form domain by VGram<=VJordan<=16VGram. The boundary proof uses no old Gaussian/corrector coefficients.',
    'fresh_action': 'Covariance bounds below are recomputed from the NEW full diagonal reference and a full relative form perturbation, using actual variance scales2^-6144/2^-7168. Radius moments use Wick/Minkowski: ||r^k||_p <=[(pk-1)!!]^(1/p) (Tr C)^(k/2); tensor gradient/Hessian use Ar covariance A/2. No unit variance is assumed. Linear slow-colour potential and kinetic forcing are priced WITHOUT cancellation. Scalar reference/body split is primitive+neutral occupied energy versus new Gaussian; the charged static Kunit subtraction and arbitrary-body charged delta h are separately priced. Root/static/contour/CPP hypotheses remain conditional; this is not a stationary residual certificate.',
    'uncancelled_channels': 'These are OUTSIDE the fixed quadratic Hamiltonian, not a principal/truncated-A mismatch. At the uniform displaced centre O(s)=O0+O1[s], H(s)=O(s)*O(s)/8. Exact colour alignment Kfast(s)=I3 tensor H(s)/5 plus the s-independent plaquette. Thus cubic potential is (2^10240/2)sum_i s_i c^T[I3 tensor H_i/5]c. Exact Gauss cometric derivative is G_i(q)=-d H0^-1 H_i H0^-1 ddag, colour-only, and its nonlinear kinetic term is -chi/2 partial^T(sum_i s_i G_i)partial. Acting on fixed FULL coupled f_A yields sum_i s_i[(chi/2)Tr(G_i A)-(chi/2)r^T A G_i A r+(2^10240/2)c^T Kfast_i c]. AGA=K cancels the ENTIRE fixed quadratic including all PC/M; it cannot cancel this variable-metric/cubic-potential score. The two triangle bounds below deliberately claim neither mean/root subtraction nor kinetic-potential nor CPP cancellation. No old Xi/Hermite coefficient is transferred.',
    'Jordan_remainder': 'For arbitrary traceless Hermitian fields let B={Phi,Phi}, L={Phi,X}+{X,Phi}, Q={X,X}. Tuple Hilbert-Schmidt bounds ||L||<=4||Phi|| ||X|| and ||Q||<=2||X||^2, with centre tuple norm<2. The cubic/quartic terms in Lambda/(16a)||B+L+Q||^2 after order2 are bounded by128Lambda |r|^3+16Lambda |r|^4 in affine normal coordinates. Polar Q(s)-Q0 costs at most512|s||f|; its quadratic-Hessian change is separately bounded by2^20 Lambda |s||f||r|+2^30 Lambda |s|^2|f|^2. First-order fast terms vanish by sum f=0, not by clipping. This deliberately loose fresh bound contains Gram AND all35 Jordan modes.',
    'limits': 'This supplies a conditional exact full90 Gaussian functional and NEW local compact neutral-directed preparation with quantified conditional action components, plus full64 coupled floating diagnostics. It does NOT supply a new exact root, certified rounded-to-algebraic coefficient error, neutral contour/gap certificate, CPP corrector, global phase/ground, same-ground photon/pole, commoncone or originalSOURCE admission. A small nonlinear component is not a small total residual; root force and uncancelled linear forcing are retained as debts.',
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def structural_fast_zero(matrix: np.ndarray) -> None:
    if matrix.shape!=(90,90) or np.count_nonzero(matrix[:45,:]) or np.count_nonzero(matrix[:,:45]):
        raise ValueError('Structural q0 fast PP/PC zero was corrupted')


def checked_part(matrix: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    if matrix.shape != (90,90) or not np.all(np.isfinite(matrix)):
        raise ValueError('Malformed full90 '+name)
    defect = float(np.linalg.norm(matrix-matrix.conj().T,2))
    if defect > 2.0**-36*max(1.,float(np.linalg.norm(matrix,2))):
        raise ValueError('Non-Hermitian kernel exceeds explicit floating diagnostic cap: '+name)
    return (matrix+matrix.conj().T)/2, defect


def positive_root(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(matrix)
    if values[0] <= 0:
        raise ValueError('Nonpositive kernel; no clipping or fallback')
    root = (vectors*np.sqrt(values))@vectors.conj().T
    inverse = (vectors/np.sqrt(values))@vectors.conj().T
    return root,inverse,values


def series_product(left: list[np.ndarray], right: list[np.ndarray], n: int) -> list[np.ndarray]:
    shape=(left[0].shape[0],right[0].shape[1])
    return [sum((left[j]@right[k-j] for j in range(max(0,k-len(right)+1),min(k,len(left)-1)+1)),np.zeros(shape,complex)) for k in range(n+1)]


def inverse_series(values: list[np.ndarray], n: int) -> list[np.ndarray]:
    inverse=[np.linalg.inv(values[0])]
    for k in range(1,n+1):
        inverse.append(-inverse[0]@sum((values[j]@inverse[k-j] for j in range(1,min(k,len(values)-1)+1)),np.zeros_like(values[0])))
    return inverse


def root_series(G0: np.ndarray, fast: np.ndarray, one: np.ndarray, unit: np.ndarray, zero: bool, n: int = 12) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    sqrtG,invG,_=positive_root(G0)
    B={0:sqrtG@fast@sqrtG,2:sqrtG@one@sqrtG,10:sqrtG@unit@sqrtG}
    R:list[np.ndarray]=[np.zeros((90,90),complex) for _ in range(n+2)]
    if zero:
        structural_fast_zero(fast)
        S,_,sv=positive_root(B[2][:45,:45]); T,_,tv=positive_root(B[0][45:,45:])
        R[0][45:,45:]=T;R[1][:45,:45]=S
        for k in range(2,n+2):
            if k>=3:
                pp=-sum((R[j][:45,:]@R[k-j][:,:45] for j in range(2,k-1)),np.zeros((45,45),complex))
                if k in B:pp+=B[k][:45,:45]
                R[k-1][:45,:45]=solve_sylvester(S,S,pp)
            rhs=B.get(k,np.zeros((90,90),complex))-sum((R[j]@R[k-j] for j in range(1,k)),np.zeros((90,90),complex))
            R[k][:45,45:]=np.linalg.solve(T.T,rhs[:45,45:].T).T
            R[k][45:,:45]=R[k][:45,45:].conj().T
            R[k][45:,45:]=solve_sylvester(T,T,rhs[45:,45:])
        exact_min_reference=float(min(sv[0],tv[0]))
    else:
        R[0],_,values=positive_root(B[0]);exact_min_reference=float(values[0])
        for k in range(1,n+1):
            rhs=B.get(k,np.zeros((90,90),complex))-sum((R[j]@R[k-j] for j in range(1,k)),np.zeros((90,90),complex))
            R[k]=solve_sylvester(R[0],R[0],rhs)
    R=R[:n+1]
    precision=[invG@v@invG for v in R]
    if zero:
        cc=[v[45:,45:] for v in R];pc=[v[:45,45:] for v in R];cp=[v[45:,:45] for v in R]
        ci=inverse_series(cc,n)
        schur=[R[k][:45,:45]-v for k,v in enumerate(series_product(series_product(pc,ci,n+1),cp,n+1)) if k<=n]
        if np.linalg.norm(schur[0])!=0:raise ArithmeticError('Lost slow Laurent order')
        si=inverse_series(schur[1:],n-1)
        pcci=series_product(pc,ci,n);cicp=series_product(ci,cp,n)
        ipc=series_product(si,pcci,n-1)
        icp=series_product(cicp,si,n-1)
        correction=series_product(icp,pcci,n-1)
        # Index k means t^(k-1), so ALL90 C includes its PP Laurent pole.
        inverse=[np.zeros((90,90),complex) for _ in range(n)]
        for k in range(n):
            inverse[k][:45,:45]=si[k]
            inverse[k][:45,45:]=-ipc[k]
            inverse[k][45:,:45]=-icp[k]
            inverse[k][45:,45:]=correction[k]+(ci[k-1] if k else 0)
        covariance=[sqrtG@v@sqrtG/2 for v in inverse]
        # Precision M, not covariance regression with its opposite sign.
        mixed=[sqrtG[45:,45:]@v@invG[:45,:45] for v in cicp]
    else:
        inverse=inverse_series(R,n)
        covariance=[sqrtG@v@sqrtG/2 for v in inverse]
        cc=[v[45:,45:] for v in precision];cp=[v[45:,:45] for v in precision]
        mixed=series_product(inverse_series(cc,n),cp,n)
    residual=series_product(R,R,2*n)
    residuals=[];block_residuals={'PP':[], 'PC':[], 'CC':[]}
    for k,v in enumerate(residual):
        error=v-B.get(k,np.zeros((90,90),complex))
        residuals.append(float(np.linalg.norm(error,2)))
        for label,rows,cols in (('PP',slice(0,45),slice(0,45)),('PC',slice(0,45),slice(45,90)),('CC',slice(45,90),slice(45,90))):
            block_residuals[label].append(float(np.linalg.norm(error[rows,cols],2)))
    # Evaluate at the physical dyadic t without converting t to binary64.
    # Block normalization preserves the genuine slow/cross scale.
    ctx=mp.mp.clone();ctx.dps=80;t=ctx.power(2,-1024)
    physical_residual={}
    for label,values in block_residuals.items():
        shift=2 if zero and label!='CC' else 0
        value=sum(ctx.mpf(str(x))*t**(k-shift) for k,x in enumerate(values))
        denominator=float(np.linalg.norm(B[2][:45,:45],2)) if zero and label=='PP' else max(1.,float(np.linalg.norm(B[0],2)))
        physical_residual[label]=ctx.nstr(value/denominator,35)
    ac=series_product(precision,covariance,2*n)
    ac_residual=[]
    for k,v in enumerate(ac):
        degree=k-1 if zero else k
        error=v-(np.eye(90)/2 if degree==0 else 0)
        ac_residual.append(float(np.linalg.norm(error,2)))
    ac_physical=sum(ctx.mpf(str(x))*t**(k-1 if zero else k) for k,x in enumerate(ac_residual))
    if ac_physical>=ctx.mpf('1e-10') or max(ctx.mpf(v) for v in physical_residual.values())>=ctx.mpf('1e-10'):
        raise ArithmeticError('Full physical graded Riccati/inverse diagnostic failed')
    active=max(k for k in range(n+1) if np.linalg.norm(R[k])!=0)
    return precision,covariance,mixed,{'coefficientwise_R_square_residual_opnorm':residuals,'physical_t_Riccati_block_relative_residual_OBSERVATION':physical_residual,'physical_t_A_C_minus_halfI_residual_OBSERVATION':ctx.nstr(ac_physical,35),'coefficientwise_A_C_minus_halfI_residual':ac_residual,'series_degree':n,'last_nonzero_root_coefficient':active,'normalized_reference_min_observation':exact_min_reference,'Kunit_enters_at_t_degree':10,'C_Laurent_index_shift':-1 if zero else 0,'physical_A_exponent':7168,'physical_C_exponent':-7168}


def neutral_covariance_consumer(arrays: dict[str,np.ndarray], raws: dict[str,bytes]) -> dict[str,Any]:
    """Compose actual full90 neutral PH Gram with every covariance coefficient."""
    peer=json.loads(raws['joined_phi_u1_neutral_ground_graph_epoch2_20261010.json'])
    if [tuple(row['q_integer']) for row in peer['derivatives']['all64_PH90']]!=jordan.QSET:
        raise ValueError('Neutral PH Gram q ordering differs')
    with np.load(ROOT/'joined_phi_u1_neutral_ground_graph_epoch2_20261010.npz',allow_pickle=False) as z:
        grams=z['neutral_derivative_Gram_all64']
        projector=z['Pplus']
        if grams.shape!=(64,90,90):
            raise ValueError('Neutral PH90 Gram shape differs')
        if np.linalg.norm(projector-arrays['neutral_Pplus'],2)>1e-10:
            raise ValueError('Neutral owner and Gaussian centre Pplus disagree')
        records=[]
        for index,q in enumerate(jordan.QSET):
            gram=grams[index]
            if np.linalg.norm(gram-gram.conj().T,2)>1e-10 or np.linalg.eigvalsh(gram)[0]<-1e-10:
                raise ValueError('Malformed neutral Gram; no projection')
            key='_'.join(map(str,q));coefficients=[]
            for k in range(12 if q==(0,0,0) else 13):
                degree=k-1 if q==(0,0,0) else k
                covariance=arrays[key+f'_C_t{degree}_divided_2powminus7168']
                full=weight(gram,covariance)
                cross=weight(gram[:45,45:],covariance[45:,:45])+weight(gram[45:,:45],covariance[:45,45:])
                coefficients.append({'t_degree':degree,'physical_power_of_two':-7168-1024*degree,'full90_Tr_derivativeGram_C':full,'PC_CP_contribution_RETAINED':cross})
            records.append({'q_integer':list(q),'full90_covariance_derivative_consumer':coefficients})
    return {'kind':'ACTUAL_CENTRE_FULL64_NEUTRAL_PH90_GRAM_TIMES_NEW_COVARIANCE_SERIES','peer_npz_sha256':PINNED['joined_phi_u1_neutral_ground_graph_epoch2_20261010.npz'],'peer_receipt_sha256':PINNED['joined_phi_u1_neutral_ground_graph_epoch2_20261010.json'],'all64':records,'signed_cross_is_not_Loewner_bounded':True,'scope':'Centre first-jet conditional Gaussian variation diagnostic, not nonlinear/global ground or total Kato residual'}

def weight(matrix: np.ndarray, covariance: np.ndarray) -> float:
    return float(np.trace(matrix@covariance).real)


def mp_exact(value: Any) -> Any:
    fraction=F(value)
    return mp.mpf(fraction.numerator)/fraction.denominator


def number(value: Any) -> dict[str, str]:
    return {'value':str(mp.nstr(value,45)),'log2':str(mp.nstr(mp.log(abs(value),2),35)) if value else '-inf','semantics':'Conditional analytic upper evaluated at80decimal digits; not an interval certificate'}


def fresh_prices(data: dict[str, Any], old0: dict[str, Any], new0: dict[str, Any], totals: dict[str, float], metric: dict[str, Any]) -> dict[str, Any]:
    mp.mp.dps=80
    power=lambda k:mp.power(2,k)
    # Safe margins on floating reference displays remain explicit premises.
    cs=mp.mpf(str(new0['Cslow_norm']))*mp.mpf('1.001')*power(-6144)
    cf=mp.mpf(str(totals['fast_C_norm']))*mp.mpf('1.001')*power(-7168)
    ts=mp.mpf(str(new0['Cslow_trace']))*mp.mpf('1.001')*power(-6144)
    tf=mp.mpf(str(totals['fast_C_trace']))*mp.mpf('1.001')*power(-7168)
    ta=mp.mpf(str(new0['Aslow_trace']))*mp.mpf('1.001')*power(6144)+mp.mpf(str(totals['fast_A_trace']))*mp.mpf('1.001')*power(7168)
    M=mp.mpf(str(new0['M2_norm']))*power(-2048)
    # Relative K form is bounded in the whole slow/fast reference, so C cannot
    # be principal-replaced. The coefficient is an explicit exact-norm premise.
    eps=mp.mpf(str(totals['relative_K_coefficient_upper']))*power(-1024)
    inflation=1/mp.sqrt(1-eps)
    cs*=inflation;cf*=inflation;ts*=inflation;tf*=inflation;ta*=mp.sqrt(1+eps)
    tc=ts+tf;R=power(-20);chi=power(-4096);L=power(8192);Lf=power(10240)
    def radial(k: int, p: int, trace: Any) -> Any:
        product=1
        for j in range(1,p*k,2):product*=j
        return mp.mpf(product)**(mp.mpf(1)/p)*trace**(mp.mpf(k)/2)
    defect=mp.sqrt(12*tc/R**2)
    if defect>=mp.mpf('.5'):raise ArithmeticError('New compact normalizer bound fails')
    # Pure Gaussian, with normalization2 ONCE on each PRE norm.
    grad=mp.sqrt(ta/2)
    hess4=(mp.sqrt(105)/2+1)*ta
    grad_pre=grad+12/R**2*mp.sqrt(tc)
    hf_pre=radial(1,4,tf)*hess4+24/R**2*radial(2,4,tc)*mp.sqrt(3*ta/2)+12*mp.sqrt(5760)/R**2*mp.sqrt(tf)+120/R**4*radial(3,2,tc)
    h2_pre=radial(2,4,tc)*hess4+24/R**2*radial(3,4,tc)*mp.sqrt(3*ta/2)+12*mp.sqrt(5760)/R**2*radial(2,2,tc)+120/R**4*radial(4,2,tc)
    c=metric['exact_bounds']['support']
    m0,a1,a2,jg,jh=[mp_exact(F(c[key])) for key in ('cometric_upper','cometric_D1_upper','cometric_D2_upper','log_density_gradient_upper','log_density_Hessian_upper')]
    normal_pre=chi*(76*(a1*hf_pre+a2*h2_pre/2)+5760*(a1+a2*R)*grad_pre)/2
    haar_pre=chi*((5760*a1*jg+5760*m0*jh)/4+m0*jg*jg/8)
    n=8324
    noether_pre=chi*(2*n)*(2*m0*grad_pre+5760*a1)/2+chi*m0*n*n*mp.mpf(15)/16
    bump_pre=chi*(m0+a1*R)*(6*5760/R**2+60/R**4*radial(2,2,tc)+12/R**2*mp.sqrt(tc*ta/2+mp.mpf(5760)/2))
    jordan_pre=L*(128*radial(3,2,tc)+16*radial(4,2,tc)+power(20)*radial(1,6,ts)*radial(1,6,tf)*radial(1,6,tc)+power(30)*radial(2,4,ts)*radial(2,4,tf))
    alignment_pre=Lf*(mp.mpf(3)/5*radial(3,2,tf)+mp.mpf(9)/40*radial(4,2,tf)+radial(2,4,ts)*radial(2,4,tf)/80)
    plaquette_pre=Lf*(mp.mpf(21)/5*radial(3,2,tf)+mp.mpf(7)/5*radial(4,2,tf))
    # No CPP/old Xi cancellation. Potential and kinetic are two actual terms.
    hmin=mp.mpf(str(np.linalg.eigvalsh(data['H'])[0]))*mp.mpf('.999')
    h1=mp.mpf(1) # O1<=1, ||O0||<2, H1<=1/2;1 is conservative.
    g1=12*h1/hmin**2
    linear_potential_pre=Lf*h1/10*radial(1,4,ts)*radial(2,4,tf)
    linear_kinetic_pre=chi*g1*radial(1,4,ts)*ta*5
    # Direct neutral Pplus unitary: Riesz circle/split-band derivative estimates
    # remain explicit isolated-band/real3R hypotheses, never Loewner transfers.
    hn=mp.mpf(str(data['neutral_h_derivative_bound']))
    gap=mp.mpf(str(data['neutral_gap_lower_premise']))
    p1=8*hn/gap;p2=128*hn**2/gap**2+8*mp.mpf(str(data['neutral_h_second_bound']))/gap
    c1=128*mp.sqrt(5760)*4*p1
    c2=128*5760*(32*p2+128*p1*p1)
    kato_pre=chi*(m0*c1*grad_pre+(5760*a1*c1+76*m0*(c2+c1*c1))/2+m0*(2*n)*c1)
    # Finite-dimensional real Riesz-energy Cauchy; explicit NEW contour premise.
    contour_radius=power(-16);Emax=256*mp.mpf(str(data['neutral_h_norm_upper']))*2
    e3=27*Emax/(contour_radius-3*R)**3
    neutral_sea_pre=L*e3/6*radial(3,2,tc)
    charged_pre=mp.mpf(str(data['charged_uniform_h_bound']))*radial(1,2,tc)
    charged_static_pre=mp.mpf(str(totals['Kunit_norm_upper']))/2*radial(2,2,tc)
    prepared={key:number(2*value) for key,value in {
        'NEW_Jordan_cubic_quartic_and_polar':jordan_pre,
        'alignment_after_quadratic_and_linear_s_colour':alignment_pre,
        'ordered_plaquette_after_quadratic':plaquette_pre,
        'neutral_filled_Pplus_sea_after_quadratic':neutral_sea_pre,
        'nonlinear_metric_remainder':normal_pre,
        'full_Haar_sqrtJ_potential':haar_pre,
        'full_Noether_cross_and_angular':noether_pre,
        'flat_reference_b6_commutator':bump_pre,
        'neutral_Pplus_Kato_connection':kato_pre,
        'arbitrary_body_all64charged_delta_h':charged_pre,
        'charged_static_Kunit_subtraction':charged_static_pre,
        'UNCANCELLED_linear_s_colour_potential':linear_potential_pre,
        'UNCANCELLED_linear_s_colour_kinetic':linear_kinetic_pre,
    }.items()}
    controlled=sum(2*v for v in (jordan_pre,alignment_pre,plaquette_pre,neutral_sea_pre,normal_pre,haar_pre,noether_pre,bump_pre,kato_pre,charged_pre,charged_static_pre))
    return {'conditional_covariance':{'slow_variance_op':number(cs),'fast_marginal_and_conditional_variance_op':number(cf),'full_trace_C':number(tc),'fast_trace_C':number(tf),'full_trace_A':number(ta),'M_leading_physical_observation_NOT_a_bound':{'value':mp.nstr(M,45),'log2':mp.nstr(mp.log(M,2),35),'semantics':'Leading t^2 coefficient of actual full90 M series, not a signed-M order or certified upper; action prices do not use it.'},'relative_form_error_upper':number(eps),'STD_not_VARIANCE':number(mp.sqrt(cs)),'STD_divided_2minus3072':number(mp.sqrt(cs)*power(3072)), 'old_slow_trace_display':old0['Cslow_trace'],'new_slow_trace_display':new0['Cslow_trace']},'compact_normalization':{'definition':'Z=integral b6^2 |f_A|^2 over full5760; never replaced by1 or1/4','L2_defect_upper':number(defect),'Z_lower':'1/4','normalization_bound':'2','normalization_applied_once':True,'corrector':'NONE: materially NEW pure compact Gaussian'},'PREPARED_action_components':prepared,'paid_components_sum_EXCLUDING_root_and_uncancelled_forcing':number(controlled),'unpaid_root_force':'2||F_exact_matching_centre|| sqrt(Tr Cslow); existence/selection and cancellation are NOT asserted. Floating display is not substituted for exact root.','uncancelled_forcing_sum':number(2*(linear_potential_pre+linear_kinetic_pre)),'premises':{'SPD_exact_fullpencil':True,'exact_reference_lower_and_norm_bounds':'Floating displayed coefficients with explicit1.001/.999 margins require a separate exact enclosure; NOT certified by this receipt.','neutral_isolated_band_gap_lower':str(gap),'neutral_Riesz_contour_radius':'2^-16 with resolvent/Pplus norm<=2, all-normal real3R collar; conditional NEW contour premise','normal_chart':'Pinned source-independent normal/Haar geometry and Clifford Q bounds, no old action price imported'},'stationary_residual_certified':False,'unit_variance_adverse_log2_ratio':mp.nstr(mp.log(1/mp.sqrt(cs),2),35)}


def neutral_probe(data: dict[str, Any], arrays: dict[str,np.ndarray]) -> dict[str, Any]:
    old=jordan.old;native=data['native'];geom=data['geom']
    profile=next(p for p in data['profiles'] if p['scale']=='heavy')
    fields=np.broadcast_to(native['frame'],(64,4,4,4)).copy()
    links=np.broadcast_to(np.eye(4),(192,4,4)).copy().astype(complex)
    h=-old.assemble(profile,native,fields,links)
    ev,U=np.linalg.eigh(h);minus=U[:,:128];plus=U[:,128:]
    P=plus@plus.conj().T;gap=float(ev[128]-ev[127])
    direction=np.zeros((45,));direction[12]=1
    # Normal Q is60x45; canonical T order is axis-major.
    T=geom['M']/np.sqrt(8)
    delta=np.einsum('ma,aij->mij',(geom['Q']@direction).reshape(4,15),T)/8
    epsilon=2.0**-15
    def projector(t: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        hp=-old.assemble(profile,native,fields+t*delta[None],links)
        vals,vecs=np.linalg.eigh(hp)
        return vecs[:,128:]@vecs[:,128:].conj().T,hp,vals
    Pp,hp,vp=projector(epsilon);Pm,hm,vm=projector(-epsilon)
    dh=(hp-hm)/(2*epsilon)
    X=(minus.conj().T@dh@plus)/(ev[None,128:]-ev[:128,None])
    dP=minus@X@plus.conj().T+plus@X.conj().T@minus.conj().T
    derivative_error=float(np.linalg.norm((Pp-Pm)/(2*epsilon)-dP,2))
    d=Pp-P;B=Pp@P+(np.eye(256)-Pp)@(np.eye(256)-P)
    root,inverse,_=positive_root(np.eye(256)-d@d)
    W=B@inverse
    unitary=float(np.linalg.norm(W.conj().T@W-np.eye(256),2))
    lift=float(np.linalg.norm(W@P@W.conj().T-Pp,2))
    neutral_E=-float(np.sum(ev[128:]));wrong_E=-float(np.sum(ev[:128]))
    assert gap>0 and derivative_error<1e-6 and unitary<1e-10 and lift<1e-10 and wrong_E>neutral_E
    arrays.update(neutral_h_unscaled=h,neutral_Pplus=P,neutral_Wplus=plus,neutral_dP_direction=dP,neutral_direct_unitary_probe=W)
    # Raw one/two-link Schur bounds, no inherited occupied derivative constants.
    paths=old.path_terms(profile,native,geom)
    pathsize=lambda p: float(np.sum(abs(np.asarray(p['coefficients']))))
    coefficient=abs(float(profile['onsite']))+2*sum(pathsize(p) for p in paths)
    hbound=coefficient*2
    derivative=coefficient*4+sum(8*pathsize(p) for p in paths)
    second=coefficient*(2**20)+sum(64*pathsize(p) for p in paths)
    charged=0.
    for p in data['profiles']:
        if p['scale']=='heavy':continue
        rows=old.path_terms(p,native,geom)
        charged+=129*float(p['load'])*(abs(float(p['onsite']))+2*sum(pathsize(x) for x in rows))*4
    return {'gap_observation':gap,'gap_lower_premise':gap*.99,'Pplus_rank':128,'Hneutral_sign':'-Lambda h, occupied top128 of h','direct_unitary_lift_defect':lift,'unitarity_defect':unitary,'offband_Sylvester_derivative_FD_defect':derivative_error,'wrong_Pminus_energy_minus_Pplus_divided_Lambda':wrong_E-neutral_E,'derivative_bound':derivative,'second_bound':second,'h_norm_upper':hbound,'charged_uniform_h_bound':charged,'normal_direction':12,'epsilon_TEST_only':epsilon,'full256_native_paths':True,'no_mean_solve':True}


def run(output: Path) -> dict[str, Any]:
    started=time.monotonic()
    raws={name:(ROOT/name).read_bytes() for name in (*PINNED,*INPUTS)}
    if any(sha(raws[name])!=p for name,p in PINNED.items()):raise ArithmeticError('Pinned epoch changed')
    receipt=json.loads(raws[INPUTS[0]])
    if receipt['source_sha256']['joined_phi_u1_matrix_jordan_supplier.py']!=PINNED['joined_phi_u1_matrix_jordan_supplier.py']:raise ArithmeticError('Supplier receipt source epoch mismatch')
    data=jordan.old.load_mean(g_override=jordan.G,mean_descriptor={'kind':'FROZEN_MATCHING_CENTRE_ONLY_NOT_NEW_ROOT'})
    z=np.load(ROOT/'joined_phi_u1_matrix_jordan_supplier_receipt_20261010.npz',allow_pickle=False)
    D=z['Jordan_D90_divided_Lambda'];H=z['H_orbit'];arrays:dict[str,np.ndarray]={}
    if np.linalg.norm(D-D.T)>1e-10 or np.count_nonzero(np.linalg.eigvalsh(D)>1e-8)!=35:raise ArithmeticError('Jordan full35 addition corrupt')
    D,Jordan_defect=checked_part(D.astype(complex),'Jordan_D90_DIAGNOSTIC_ONLY')
    records=[];counts=[0,0];conjugacy=0.;old0={};new0={}
    totals={'fast_C_trace':0.,'fast_A_trace':0.,'fast_C_norm':0.,'relative_K_coefficient_upper':0.,'Kunit_norm_upper':0.}
    rng=np.random.default_rng(2221010);Z=rng.normal(size=(90,90))+1j*rng.normal(size=(90,90));dense=Z.conj().T@Z
    weights={'I90':np.eye(90),'Phi45_full90':np.diag([1]*45+[0]*45),'colour45_full90':np.diag([0]*45+[1]*45),'densePSD90':dense}
    with threadpool_limits(limits=1):
        for q in jordan.QSET:
            key='_'.join(map(str,q));partner=tuple((-x)%4 for x in q);zero=q==(0,0,0)
            raw={name:z[key+'_'+name] for name in ('G_divided_chi','K_Lambda5over4','K_Lambda','K_unit')}
            if zero:structural_fast_zero(raw['K_Lambda5over4'])
            if np.linalg.norm(raw['G_divided_chi']-raw['G_divided_chi'].conj().T)!=0:raise ValueError('Cometric is not exactly Hermitian in archived diagnostic')
            G0=raw['G_divided_chi'];parts={};defects={}
            for name in ('K_Lambda5over4','K_Lambda','K_unit'):
                parts[name],defects[name]=checked_part(raw[name],name)
            fast=parts['K_Lambda5over4'];one=parts['K_Lambda'];unit=parts['K_unit']
            oldA,oldC,oldM,oldinfo=root_series(G0,fast,one,unit,zero)
            newA,newC,newM,info=root_series(G0,fast,one+D,unit,zero)
            for k in range(len(newA)):arrays[key+f'_A_t{k}_divided_2pow7168']=newA[k]
            for k in range(len(newC)):arrays[key+f'_C_t{k-1 if zero else k}_divided_2powminus7168']=newC[k]
            for k in range(len(newM)):arrays[key+f'_M_t{k}']=newM[k]
            if q<=partner:counts[0 if q==partner else 1]+=1
            for name,matrix in raw.items():conjugacy=max(conjugacy,float(np.max(abs(matrix.conj()-z['_'.join(map(str,partner))+'_'+name]))))
            if zero:
                for target,A,C,M in ((old0,oldA,oldC,oldM),(new0,newA,newC,newM)):
                    target.update(Aslow_trace=float(np.trace(A[1][:45,:45]).real),Cslow_trace=float(np.trace(C[0][:45,:45]).real),Cslow_norm=float(np.linalg.norm(C[0][:45,:45],2)),M2_norm=float(np.linalg.norm(M[2],2)))
                referenceC=np.zeros((90,90),complex);referenceC[:45,:45]=newC[0][:45,:45];referenceC[45:,45:]=newC[1][45:,45:]
                totals['fast_C_trace']+=float(np.trace(newC[1][45:,45:]).real)
                totals['fast_A_trace']+=float(np.trace(newA[0][45:,45:]).real)
                totals['fast_C_norm']=max(totals['fast_C_norm'],float(np.linalg.norm(newC[1][45:,45:],2)))
                diff=oldC[0]-newC[0];comparison_scale='Phi slow2^-6144; full90 coefficient, colour leading difference0 and higher coefficients retained'
                pc_coefficient=float(np.linalg.norm(newA[2][:45,45:],2))
                Mcoefficient=float(np.linalg.norm(newM[2],2))
                wrongpc=np.zeros((90,90),complex);wrongpc[:45,45:]=newA[2][:45,45:];wrongpc[45:,:45]=newA[2][45:,:45]
                pcdefect=float(np.linalg.norm(wrongpc,2))
            else:
                referenceC=newC[0]
                totals['fast_C_trace']+=float(np.trace(referenceC).real)
                totals['fast_A_trace']+=float(np.trace(newA[0]).real)
                totals['fast_C_norm']=max(totals['fast_C_norm'],float(np.linalg.norm(referenceC,2)))
                diff=oldC[2]-newC[2];comparison_scale='first full90 covariance change2^-9216; leading fast2^-7168 identical'
                pc_coefficient=float(np.linalg.norm(newA[2][:45,45:],2));Mcoefficient=float(np.linalg.norm(newM[2],2));pcdefect=pc_coefficient
            totals['relative_K_coefficient_upper']=max(totals['relative_K_coefficient_upper'],1e5*max(1.,float(np.linalg.norm(one+D,2)),float(np.linalg.norm(unit,2))))
            totals['Kunit_norm_upper']=max(totals['Kunit_norm_upper'],float(np.linalg.norm(unit,2))*1.001)
            minchange=float(np.linalg.eigvalsh((diff+diff.conj().T)/2)[0])
            fullweights={name:weight(B,diff) for name,B in weights.items()}
            if minchange < -1e-8 or min(fullweights.values()) < -1e-7:raise ArithmeticError('FullPSD comparison diagnostic failed')
            # Wrong inverse square root, on same full leading90 kernel. At q0
            # initialize the full mixed hierarchy; no toy/principal control.
            if not zero:
                sqrtG,invG,_=positive_root(G0);B=sqrtG@fast@sqrtG;_,wrong,_=positive_root(B)
                wrongA=invG@wrong@invG
                wrongdefect=float(np.linalg.norm(wrongA@G0@wrongA-fast,2)/np.linalg.norm(fast,2))
            else:wrongdefect=None
            records.append({'q_integer':list(q),'full90':True,'structural_q0_zero':zero,'archived_antiHermitian_defects_NOT_suppressed':defects,'scientific_projection_adopted':False,'diagnostic_definition':'explicit Hermitian part of archived rounded matrices ONLY','new_Riccati_series':info,'old_Riccati_series':oldinfo,'fullPSD_Cold_minus_Cnew_weights':fullweights,'fullPSD_difference_min_observation':minchange,'covariance_change_scale':comparison_scale,'PC_precision_coefficient_RETAINED':pc_coefficient,'M_coefficient_RETAINED':Mcoefficient,'M_physical_leading_power':-2048,'wrong_reset_PC_coefficient_defect':pcdefect,'wrong_inverse_square_root_relative_Riccati_defect':wrongdefect})
        if counts!=[8,28] or conjugacy>1e-10:raise ArithmeticError('Full64 reality boundary failed')
        neutral=neutral_probe(data,arrays)
        coupled_neutral=neutral_covariance_consumer(arrays,raws)
    metric=json.loads(raws['induced_phi_global_normal_gauss_chart_certificate_receipt.json'])
    if 'exact_bounds' not in metric:raise ValueError('Missing chart exact constants')
    prices=fresh_prices({'H':H,'neutral_h_derivative_bound':neutral['derivative_bound'],'neutral_h_second_bound':neutral['second_bound'],'neutral_gap_lower_premise':neutral['gap_lower_premise'],'neutral_h_norm_upper':neutral['h_norm_upper'],'charged_uniform_h_bound':neutral['charged_uniform_h_bound']},old0,new0,totals,metric)
    malformed=raw['K_Lambda'].copy();malformed[0,1]+=1j
    try:checked_part(malformed,'ADVERSE_corrupt_PC')
    except ValueError:malformed_rejected=True
    else:raise AssertionError('Malformed kernel was silently projected')
    corruptfast=z['0_0_0_K_Lambda5over4'].copy();corruptfast[0,0]=1e-15
    try:root_series(z['0_0_0_G_divided_chi'],corruptfast,z['0_0_0_K_Lambda']+D,z['0_0_0_K_unit'],True)
    except ValueError:structural_rejected=True
    else:raise AssertionError('Structural zero contamination accepted')
    corruptPC=z['0_0_0_K_Lambda5over4'].copy()
    corruptPC[0,45]=1e-15j;corruptPC[45,0]=1e-15j
    # Its Hermitian part erases this adverse input: check STRUCTURE before it.
    try:structural_fast_zero(corruptPC)
    except ValueError:structural_PC_rejected=True
    else:raise AssertionError('Tiny anti-Hermitian structural PC was erased')
    arrays['Jordan_D90_divided_Lambda']=D;arrays['normal_Q']=z['normal_Q'];arrays['H_orbit']=H
    npz=output.with_suffix('.npz');np.savez_compressed(npz,allow_pickle=False,**arrays)
    captures={**jordan.old.CAPTURED,Path(jordan.__file__).name:Path(jordan.__file__).read_bytes(),Path(__file__).name:SELF}
    stable={name:(ROOT/name).read_bytes()==raw for name,raw in {**raws,**captures}.items()}
    if not all(stable.values()):raise ArithmeticError('Input/source mutation')
    archive=output.with_name(output.stem+'_input_source.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as out:
        for name,raw in {**raws,**captures}.items():out.writestr(name,raw)
    result={'kind':'NEW_CONDITIONAL_FULL64_PH90_JORDAN_GAUSSIAN_AND_COMPACT_NEUTRAL_PPLUS_PREPARATION','plan':{'id':'issue-222-induced-photon-delta13-continuation-20261008','revision':'55352a0aabbcc1da539641239f48f2807ec1e2a609b5d533ca0f186d8df27b6e','template':'general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','work_item':'induced-photon'},'proof':PROOF,'scope':{'momenta':64,'complex_fields_per_q':90,'real_fields':5760,'real_pair_sectors':36,'selfinverse':8,'conjugate_pairs':28,'Jordan_rank_each_q':35,'originalSOURCE_adopted':False,'new_root_adopted':False,'old_corrector_adopted':False,'fixed_physical_parameters':{'Lambda':'2^8192','chi':'2^-4096','fast':'2^10240','a':'1/64'},'all64_canonical_species_fullPi129_fullrhoHC64_body':'retained symbolically; diagnostic Gaussian does not compute its ground'},'old0':old0,'new0':new0,'full64_PH90_coupled_diagnostics':records,'q_conjugacy_defect':conjugacy,'new_preparation_action':prices,'neutral_actual_full256_consumer':neutral,'adverse_controls':{'malformed_PC_kernel_rejected':malformed_rejected,'nonzero_structural_q0_fastPP_1e_minus15_rejected':structural_rejected,'wrong_inverse_square_root_min_relative_defect':min(row['wrong_inverse_square_root_relative_Riccati_defect'] for row in records if row['wrong_inverse_square_root_relative_Riccati_defect'] is not None),'wrong_PC_reset_q0_coefficient_defect':records[0]['wrong_reset_PC_coefficient_defect'],'wrong_neutral_Pminus_sign_energy_defect_divided_Lambda':neutral['wrong_Pminus_energy_minus_Pplus_divided_Lambda'],'wrong_unit_variance_STD_ratio_log2':prices['unit_variance_adverse_log2_ratio']},'input_sources_unchanged':stable,'source_sha256':{name:sha(raw) for name,raw in captures.items()},'input_sha256':{name:sha(raw) for name,raw in raws.items()},'arrays':{'path':npz.name,'sha256':sha(npz.read_bytes()),'scope':'exponent-graded full90 numerical coefficients and actual neutral256 frames; NOT exact/interval covariance adoption'},'immutable_archive':{'path':archive.name,'sha256':sha(archive.read_bytes())},'runtime':{'python':sys.executable,'version':sys.version,'numpy':np.__version__,'mpmath':mp.__version__},'elapsed_seconds':time.monotonic()-started,'OPEN':['exact new stationary/root and fullPH coefficient enclosures','new corrector/CPP and mean-force cancellation','global Jordan phase/fullPhi ground identification','same-ground spatial photon pole/dispersion','commoncone/originalSOURCE/Compton/campaign']}
    result['actual_full64_neutral_covariance_consumer']=coupled_neutral
    result['Jordan_Hessian_antiHermitian_defect_DIAGNOSTIC_ONLY']=Jordan_defect
    result['adverse_controls']['raw_q0_fastPC_antiHermitian_1e_minus15_rejected_before_diagnostic_projection']=structural_PC_rejected
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'actual_20261010.json'))
    args=parser.parse_args()
    result=run(args.output)
    print(json.dumps({'kind':result['kind'],'scope':result['scope'],'new0':result['new0'],'controls':result['adverse_controls'],'paid_sum':result['new_preparation_action']['paid_components_sum_EXCLUDING_root_and_uncancelled_forcing'],'uncancelled':result['new_preparation_action']['uncancelled_forcing_sum'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
