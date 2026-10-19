"""NEW Jordan law: exact local Hessian and full-pencilled comparison supplier.

No mean/root solver, ground, SOURCE, old covariance or action is adopted.
The saved PH90 matrices are observations at an OLD comparison centre only.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import ijson
import mpmath as mp
import numpy as np
import sympy as sp
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as old

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_matrix_jordan_supplier'
SELF = Path(__file__).read_bytes()
G = [F('0.28917434559291342'), *[F('0.10475219290309079')]*3]
COMPARISON_NAME = 'induced_phi_nonuniform_field_car_final_common_epoch_actual_receipt.json'
COMPARISON_SHA = 'b20c715c5127d24e29d9d5db0ac3f6e2b8d282605fb01bbee89885e4016849f8'
QSET = list(product(range(4), repeat=3))
PROOF = {
 'stage': 'Independent NEW microscopic Jordan law; exact algebra/local physical Hessian, conditional full-pencilled Loewner comparison. NOT an actual new-root Gaussian action/source supplier.',
 'primitive': 'Only VGram is replaced: VJ=Lambda/(16a) sum_x,mu,nu Tr({Phi_mu,Phi_nu}^2). All four fields remain arbitrary traceless Hermitian4x4. No Clifford constraint, receiver, clock, covariance, EM B2 or speed law is added.',
 'identity': 'Gmunu=Tr(Phi_mu Phi_nu)/4, Fmunu={Phi_mu,Phi_nu}-2Gmunu I. TrF=0. Expanding Tr(F+2G I)^2 gives TrF^2+16G^2, hence VJ=Lambda/a sum TrG^2+Lambda/(16a) sum ||F||HS^2 exactly. F is Hermitian, so the difference is nonnegative.',
 'zero_set_SPD': 'This is the zero set of VJ-VGram, NOT of VJ. For SPD G choose R=G^1/2 and Gamma=R^-1 Phi. F=0 iff {Gamma_a,Gamma_b}=2delta_ab I. The universal complex Cl4 algebra is M4(C) (its16 monomials are an independent basis, verified here). A unital4-dimensional star representation is faithful and the unique irreducible representation, so Gamma=U GammaCanonical Udag. Every star automorphism of M4 is unitary-inner. Multiplying U by a scalar fixes detU=1 without changing conjugation. The stabilizer in SU4 is Z4. Thus at fixed SPD Gram the complete zero set is one SU4/Z4 orbit, not a separately imposed constraint. Singular Gram is outside this classification.',
 'linearization': 'At Phi=R Gamma, Lmunu(X)={X_mu,Phi_nu}+{Phi_mu,X_nu}-(Tr(X_mu Phi_nu+Phi_mu X_nu)/2)I. F(Phi+tX)=tL+t^2 F(X). Difference value and first derivative vanish. Hessian D2(VJ-VGram)[X,Y]=Lambda/(8a) sum ReTr(L(X)L(Y)). All60 Cart directions are retained.',
 'kernel_all_SPD': 'Let deltaGmunu=Tr(X_mu Phi_nu+Phi_mu X_nu)/4. The Gram lift Y_mu=(deltaG G^-1)munu Phi_nu/2 has this deltaG and L(Y)=0. After X-Y, and the invertible row transform R^-1, the relations linearize to {Z_a,Gamma_b}+{Gamma_a,Z_b}=0. Sending Gamma_a to Z_a defines a derivation of Cl4=M4 because it annihilates its defining relations. Every derivation of M4 is inner; star preservation permits an anti-Hermitian generator, unique modulo iI. Hence X-Y=i[H,Phi] with H traceless Hermitian. The gauge map has rank15 because the commutant of the four Gamma is scalar. Gram lifts have rank10 and disjoint gauge image since gauge deltaG=0. Therefore kernel dimension25 and rank35 for EVERY SPD Gram, not merely this floating centre.',
 'grade_boundary': 'At the identity Gram: vector-field sector16 has kernel16=10Gram+6gauge; bivector-field sector24 has kernel4 (vector generators), rank20; trivector-field sector16 has kernel1 (pseudoscalar generator), rank15; pseudoscalar-field sector4 is all gauge (trivector generators). The normal45 keeps10Gram zero additions and ALL35 positive orientation additions. The OLD total45 spectrum already had conditional35 orientation curvature from neutral-slack/sea curvature, not traceGram alone. Jordan adds curvature to those same retained modes; it does not repair a supposed omission in the old full45 supplier.',
 'canonical_units': 'Gamma0=X tensor I, Gamma1=Z tensor X, Gamma2=Z tensor Y, Gamma3=Z tensor Z. M words: vectors, increasing bivectors times+i, increasing trivectors times+i, ordered pseudoscalar with phase+1, matching existing normal_geometry exactly. TrM_aM_b=4delta. Cart T=M/sqrt8, TrT_aT_b=delta/2. Phi_mu=sqrt(g_mu)Gamma_mu, NOT2sqrt(g_mu). At a=1/64 the Hessian/Lambda in Cart is sum Tr(Lraw_i Lraw_j), where raw variations use M (the factor1/8 from T cancels1/(8a)).',
 'physical_Gauss': 'The existing grade normal Q is60x45, QdagQ=I, orbit O=i[T,Phi] in Cart, QdagO=0. Difference Hessian annihilates O. New normal addition D45=Qdag D60 Q has rank35 and exact Gram10 kernel. Fourier Phi_x=frame+sum_q exp(iq.x) Q T eta_q/8; Parseval cancels64 sites against1/64. Thus every one of64 q receives Lambda diag(D45,0colour45), no extraV, no colour or Gauss-cometric change at this centre. Gphysical(q)=chi diag(8I45,.3I45+(d ddag) tensor H^-1), H=OdagO/8. The law at nonzero displacement still has curved Gauss geometry; this is its Hessian at the stated centre.',
 'full_pencil': 'KGram(q)=2^10240 Kfast(q)+2^8192 K1(q)+Kunit(q), all three FULL90 Hermitian coefficient matrices, both PH directions/all65 loads and all PC blocks retained. KJ(q)=KGram(q)+2^8192 diag(D45,0). A=Gphysical^-1/2 sqrt(Gphysical^1/2 KJ Gphysical^1/2) Gphysical^-1/2, A Gphysical A=KJ, C=.5 A^-1; f=det(A/pi)^1/4 exp(-eta* A eta/2) in the5760-dimensional canonical real Fourier space. This exact functional is conditional on the actual full pencil being SPD; saved binary matrices do NOT establish actual new-root coefficients.',
 'Loewner': 'For any SAME positive Gphysical and SAME SPD exact full old-centre KGram, D>=0 implies KJ>=KGram. Operator monotonicity of sqrt and congruence give AJ>=AGram, and inversion gives0<CJ<=CGram. Consequently Tr(B CJ)<=Tr(B CGram) for every B>=0, including total/Phi/colour/projected variances. NonzeroD makes TrCJ<TrCGram (otherwise PSD covariance difference would vanish, forcingD=0). This does NOT bound general signed cross entries, M, derivative scores, Hermite resolvents, nonlinear action residuals, occupied-frame jets, new-root covariance, or interacting ground covariance. It is not an interval certification of the rounded saved matrices.',
 'mixing': 'Precision completion always uses M=ACC^-1 ACP and conditional colour z=c+M s; Cslow=.5(APP-APC ACC^-1 ACP)^-1, Ccolour_given_s=.5 ACC^-1. NoM=0 step is used. Full saved PC matrices are preserved; the actual selected physical2x2 principal comparison exercises a nonzero M. That block is NOT a replacement for full90 or its covariance.',
 'Friedrich': 'Put s_x=sum_mu TrPhi_mu^2. TrG_x^2>=(TrG_x)^2/4=s_x^2/64. Thus VJ>=VGram>=Lambda/(64a)sum s_x^2>=Lambda/(256aV)||PhiCart||^4. Also ||{A,B}||HS<=2||A||HS||B||HS gives VJ<=Lambda/(4a)sum s_x^2<=16VGram, hence0<=difference<=15VGram. Finite-N/finite-Fock matter is at most linear in ||Phi|| with a uniform finite coefficient; Young absorbs it in the quartic. Alignment, plaquette and HC are nonnegative; gauge links are compact and Theta-dependent finite hopping remains uniformly bounded inTheta. The original positive kinetic form plus this semibounded potential on smooth compact support has a Friedrich extension. The new and old forms share a form domain by these two-sided quartic comparisons. Gauge invariance permits the invariant closed-form restriction. Theta noncompactness prevents a compact-resolvent/globalground inference.',
 'Kato_holomorphy': 'For H(eta)=Hold+eta(VJ-VGram), common form domain and0<=difference<=15VGram make a holomorphic sectorial-form family for Reeta>0, locally sectorial with a common lower shift; its associated operators are Kato type(B). This is holomorphy in the Jordan interpolation parameter, not a reused occupied Slater W chart. Separately the finite-dimensional SPD pencil square root/inverse is locally holomorphic on a complex neighbourhood avoiding the negative-real spectral cut. Neither statement supplies real occupied-frame derivative norms or automatic oldaction reuse.',
 'debts': ['NEW stationary/root equation including changed35 orientation zero-point modes and possible non-Clifford minima', 'Actual full new-root PH90 coefficients and Gaussian covariance, not saved-old-centre adoption', 'Full joint Gaussian derivative/Hermite corrector, retained M and new35 forcing repricing', 'Jordan cubic/quartic Taylor remainder, nonlinear Gauss/chart and occupied real Kato/source residual repricing', 'Actual fullPhi quantum ground characterization, sameground photon dispersion/pole, commoncone and joined source remain OPEN']
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode(value: dict[str, Any]) -> np.ndarray:
    return np.asarray(value['real'])+1j*np.asarray(value['imag'])


def exact_basis() -> tuple[list[Any], list[Any], list[tuple[int, ...]]]:
    x = sp.Matrix([[0,1],[1,0]])
    y = sp.Matrix([[0,-sp.I],[sp.I,0]])
    z = sp.diag(1,-1)
    gamma = [sp.kronecker_product(x,sp.eye(2)), *[sp.kronecker_product(z,p) for p in (x,y,z)]]
    words = [(i,) for i in range(4)]+list(combinations(range(4),2))+list(combinations(range(4),3))+[(0,1,2,3)]
    matrices = []
    for word in words:
        matrix = sp.eye(4)
        for i in word:
            matrix = matrix*gamma[i]
        matrices.append((sp.I if len(word) in (2,3) else 1)*matrix)
    return gamma, matrices, words


def linear_design(frame: list[Any], matrices: list[Any], field_count: int = 4) -> Any:
    """Exact traceless anticommutator map in unscaled signed grade coefficients."""
    rows = []
    for mu in range(field_count):
        for nu in range(field_count):
            block = []
            for slot in range(60):
                axis, basis = divmod(slot,15)
                first = matrices[basis] if axis == mu else sp.zeros(4)
                second = matrices[basis] if axis == nu else sp.zeros(4)
                raw = first*frame[nu]+frame[nu]*first+frame[mu]*second+second*frame[mu]
                raw -= sp.trace(raw)*sp.eye(4)/4
                block.append([sp.expand(sp.trace(m*raw)/4) for m in matrices])
            rows.extend(zip(*block, strict=True))
    return sp.Matrix(rows)


def exact_algebra() -> tuple[dict[str, Any], Any, list[Any], list[Any]]:
    gamma, matrices, words = exact_basis()
    r = sp.symbols('r0:4', positive=True)
    frame = [r[i]*gamma[i] for i in range(4)]
    J = linear_design(frame, matrices)
    D = 4*J.T*J
    Ji = J.subs(dict.fromkeys(r,1))
    Di = 4*Ji.T*Ji
    gram = []
    for mu,nu in combinations(range(4),2):
        column = sp.zeros(60,1)
        column[15*mu+nu] = 1
        column[15*nu+mu] = 1
        gram.append(column)
    for mu in range(4):
        column = sp.zeros(60,1)
        column[15*mu+mu] = 1
        gram.append(column)
    gauge = []
    for h in matrices:
        column = []
        for p in gamma:
            variation = sp.I*(h*p-p*h)
            column.extend(sp.trace(b*variation)/4 for b in matrices)
        gauge.append(sp.Matrix(column))
    kernel = sp.Matrix.hstack(*gram,*gauge)
    allwords = sp.Matrix.hstack(*[sp.Matrix(m).reshape(16,1) for m in [sp.eye(4),*matrices]])
    metrics = sp.Matrix([[sp.trace(a*b) for b in matrices] for a in matrices])
    if metrics != 4*sp.eye(15) or allwords.rank() != 16 or Ji.rank() != 35 or kernel.rank() != 25 or Di*kernel != sp.zeros(60,25):
        raise ArithmeticError('Exact full60 Clifford/kernel theorem consumer failed')
    ranks = []
    for start,stop in ((0,4),(4,10),(10,14),(14,15)):
        ids = [15*mu+a for mu in range(4) for a in range(start,stop)]
        ranks.append(Ji[:,ids].rank())
    omitted = linear_design(gamma,matrices,3)
    witness = 15*3+4
    if omitted[:,witness] != sp.zeros(135,1) or Di[witness,witness] <= 0:
        raise ArithmeticError('Missing fourth-field control did not reach lost mode')
    t = sp.Symbol('t', real=True)
    perturbation = [sp.zeros(4),sp.zeros(4),sp.zeros(4),matrices[4]/sp.sqrt(8)]
    moving = [gamma[i]+t*perturbation[i] for i in range(4)]
    def variance(fields: list[Any]) -> Any:
        total = 0
        for a in fields:
            for b in fields:
                f = a*b+b*a-sp.trace(a*b)*sp.eye(4)/2
                total += sp.trace(f*f)*4  # Lambda/(16a)=4 Lambda
        return sp.expand(total)
    polynomial = variance(moving)
    if sp.expand(polynomial).coeff(t,2) != Di[witness,witness]/2:
        raise ArithmeticError('Exact Hessian versus full quartic unit control failed')
    tangent_fields = [gamma[0]*(1+t),*gamma[1:]]
    tangent_variance = variance(tangent_fields)
    wrong_gram_mass = sp.diff(((1+t)**2-1)**2,t,2).subs(t,0)
    if tangent_variance != 0 or wrong_gram_mass <= 0:
        raise ArithmeticError('Gram tangent wrongly counted as new mass')
    result = {'exact_rank':35,'exact_kernel_dimension':25,'Gram_lifts':10,'SU4_gauge':15,
              'identity_grade_ranks':ranks,'Cl4_matrix_algebra_dimension':16,
              'Hessian_entries':[[str(sp.expand(e)) for e in row] for row in D.tolist()],
              'root_symbols':[str(e) for e in r], 'word_order':[list(w) for w in words],
              'wrong_controls':{'missing_fourth_field_rank':int(omitted.rank()),'fourth_field_witness_Cart_slot':witness,
                  'correct_witness_Hessian_divided_Lambda':str(Di[witness,witness]),'wrong_omitted_Hessian':'0',
                  'full_quartic_witness_path':str(polynomial),'Gram_tangent_difference':'0',
                  'wrong_tangent_Gram_mass_second_derivative':str(wrong_gram_mass),
                  'wrong_Cart_M_instead_of_T_factor':'8', 'wrong_Phi_2sqrtg_Hessian_factor':'4'}}
    return result, D, gamma, matrices


def physical_addition(data: dict[str, Any], symbolic: Any, gamma: list[Any], matrices: list[Any]) -> dict[str, Any]:
    geom, frame = data['geom'], data['native']['frame']
    B = np.asarray([np.asarray(m,dtype=complex) for m in matrices])
    canonical = np.asarray([np.asarray(m,dtype=complex) for m in gamma])
    roots = np.sqrt(np.asarray([float(g) for g in G]))
    if np.max(abs(geom['M']-B)) != 0 or np.max(abs(frame-roots[:,None,None]*canonical)) > 1e-15:
        raise ArithmeticError('Signed basis or sqrtg centre was silently changed')
    r = sorted(symbolic.free_symbols,key=str)
    D60 = np.asarray(symbolic.subs(dict(zip(r,roots,strict=True))),dtype=float)
    Q = geom['Q']
    D45 = Q.T@D60@Q
    T = B/np.sqrt(8)
    variations = np.zeros((60,4,4,4),complex)
    for slot in range(60):
        axis,basis = divmod(slot,15)
        variations[slot,axis] = T[basis]
    literal_L = np.zeros((60,4,4,4,4),complex)
    for mu in range(4):
        for nu in range(4):
            raw = variations[:,mu]@frame[nu]+frame[nu]@variations[:,mu]+frame[mu]@variations[:,nu]+variations[:,nu]@frame[mu]
            literal_L[:,mu,nu] = raw-np.trace(raw,axis1=-2,axis2=-1)[:,None,None]*np.eye(4)/4
    literal_D60 = 8*np.einsum('kmnij,lmnji->kl',literal_L,literal_L).real
    sites = np.asarray(QSET)
    fourier = np.exp(2j*np.pi*sites@sites.T/4)/8
    fourier_defect = float(np.max(abs(fourier.conj().T@fourier-np.eye(64))))
    if F(64)*F(1,8)**2 != 1:
        raise ArithmeticError('Exact Fourier site factor failed')
    orbit = old.gauss.orbit(frame,T)
    H = orbit.T@orbit/8
    eigen60 = np.linalg.eigvalsh(D60)
    eigen45 = np.linalg.eigvalsh(D45)
    controls = {'Cart_metric':float(np.max(abs(2*np.einsum('aij,bji->ab',T,T)-np.eye(15)))),
        'literal_full60_Hessian':float(np.max(abs(literal_D60-D60))),
        'Fourier_Parseval':fourier_defect,
        'Q_metric':float(np.max(abs(Q.T@Q-np.eye(45)))),
        'orbit_normal':float(np.max(abs(Q.T@orbit))),
        'orbit_Hessian_kernel':float(np.max(abs(D60@orbit))),
        'Gram10_Hessian_kernel':float(np.max(abs(D45[:10]))),
        'wrong_Cart_factor_defect':float(np.linalg.norm(8*D60-D60,2)),
        'wrong_double_frame_defect':float(np.linalg.norm(4*D60-D60,2))}
    if max(controls[k] for k in ('Cart_metric','Q_metric','orbit_normal','orbit_Hessian_kernel','Gram10_Hessian_kernel','literal_full60_Hessian','Fourier_Parseval')) > 1e-10:
        raise ArithmeticError('Physical60/normal45/gauge/Gram unit boundary failed')
    if np.count_nonzero(eigen60>1e-8) != 35 or np.count_nonzero(eigen45>1e-8) != 35:
        raise ArithmeticError('Lost one of35 physical orientation modes')
    D90 = np.zeros((90,90)); D90[:45,:45] = D45
    return {'D60':D60,'D45':D45,'D90':D90,'Q':Q,'H':H,'eigen60':eigen60,'eigen45':eigen45,'controls':controls}


def comparison_snapshot(path: Path) -> tuple[dict[str, Any], bytes]:
    """Stream ONLY archived descriptor and all64 full spectral coefficient blocks."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    if digest.hexdigest() != COMPARISON_SHA:
        raise ArithmeticError('Immutable OLD comparison receipt SHA mismatch')
    with path.open('rb') as stream:
        descriptor = next(ijson.items(stream,'input_descriptor',use_float=True))
    rows = []
    with path.open('rb') as stream:
        for row in ijson.items(stream,'actual_ALL64_q.item',use_float=True):
            rows.append({'q_integer':row['q_integer'],'spectral':row['actual_static_spectral_supplier']})
    if [tuple(row['q_integer']) for row in rows] != QSET:
        raise ArithmeticError('Saved full64q comparison missing or reordered')
    snapshot = {'stage':'OLD_BINARY_MATRIX_COMPARISON_ONLY_NOT_NEW_ROOT_OR_SOURCE',
        'parent_receipt':{'path':path.name,'sha256':digest.hexdigest(),'bytes':path.stat().st_size},
        'extraction':'input_descriptor plus actual_ALL64_q[*].q_integer/actual_static_spectral_supplier; numerical JSON values preserved as binary64 observations',
        'descriptor':descriptor,'all64q':rows}
    raw = (json.dumps(snapshot,indent=2)+'\n').encode()
    return snapshot,raw


def matrix_wave(Gmatrix: np.ndarray, Kmatrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive matrix Riccati functional; covariance is HALF the inverse precision."""
    gv,gu = np.linalg.eigh(Gmatrix)
    if gv[0] <= 0:
        raise ValueError('Positive cometric required')
    sqrtG = (gu*np.sqrt(gv))@gu.conj().T
    inverse = (gu/np.sqrt(gv))@gu.conj().T
    w,U = np.linalg.eigh(sqrtG@Kmatrix@sqrtG)
    if w[0] <= 0:
        raise ValueError('Positive full stiffness required')
    A = inverse@(U*np.sqrt(w))@U.conj().T@inverse
    return A,np.linalg.inv(A)/2


def exact_action_control() -> dict[str, Any]:
    x,y = sp.symbols('x y',real=True)
    vector = sp.Matrix([x,y])
    A = sp.Matrix([[2,1],[1,3]])
    Gm = sp.Matrix([[3,1],[1,2]])
    K = A*Gm*A
    addition = sp.Matrix([[2,0],[0,0]])
    C = A.inv()/2
    fixed_old_residual = A*Gm*A-(K+addition)
    f = sp.exp(-(vector.T*A*vector)[0]/2)
    hess = sp.hessian(f,(x,y))
    energy = sp.trace(Gm*A)/2
    action = sp.simplify((-sp.trace(Gm*hess)/2+((vector.T*(K+addition)*vector)[0]/2-energy)*f)/f)
    if fixed_old_residual != -addition or sp.expand(action) != x*x or A*Gm == Gm*A or 2*A*C != sp.eye(2):
        raise ArithmeticError('Exact fixed-oldCov/noncommuting action boundary failed')
    Gf,Kf,Df = [np.asarray(m,dtype=float) for m in (Gm,K,addition)]
    newA,newC = matrix_wave(Gf,Kf+Df)
    defect = float(np.max(abs(newA@Gf@newA-(Kf+Df))))
    wrong = np.linalg.eigh(newA)
    wrongC = (wrong[1]/np.sqrt(wrong[0]))@wrong[1].T/2
    wrong_defect = float(np.linalg.norm(2*newA@wrongC-np.eye(2)))
    loewner = float(np.linalg.eigvalsh(np.asarray(C,dtype=float)-newC)[0])
    if defect>1e-10 or wrong_defect<=.1 or loewner < -1e-12:
        raise ArithmeticError('New AGA/C=.5inverse floating consumer failed')
    return {'exact_fixed_old_precision_Riccati_residual':str(fixed_old_residual),
        'exact_old_Gaussian_new_action_divided_wave':str(action),'exact_half_inverse_identity':str(2*A*C),
        'new_Riccati_defect':defect,'wrong_A_minus_half_covariance_identity_defect':wrong_defect,
        'Loewner_covariance_difference_min':loewner,'new_A':newA.tolist(),'new_C':newC.tolist()}


def mpfloat(ctx: Any, value: float) -> Any:
    numerator,denominator = float(value).as_integer_ratio()
    return ctx.mpf(numerator)/denominator


def principal_comparison(row: dict[str, Any], D: np.ndarray) -> dict[str, Any]:
    """Full physical scales on a principal2 diagnostic, never a90-mode substitute."""
    ctx: Any = mp.mp.clone(); ctx.dps = 900
    spectral = row['spectral']
    kf,k1,k0 = [decode(spectral[name]) for name in ('K_Lambda5over4','K_Lambda','K_unit')]
    scores = abs(k1[:45,45:]).copy()
    scores[np.diag(D[:45,:45])<1e-8] = 0
    i,jc = np.unravel_index(np.argmax(scores),scores.shape); j = int(jc)+45; i = int(i)
    if scores[i,jc] <= 1e-12:
        raise ArithmeticError('No genuine archived PC mixing on a Jordan-mass mode')
    ids = [i,j]
    def coefficient(matrix: np.ndarray) -> Any:
        observed = ctx.matrix([[ctx.mpc(mpfloat(ctx,matrix[a,b].real),mpfloat(ctx,matrix[a,b].imag)) for b in ids] for a in ids])
        # This principal diagnostic has an explicitly Hermitian rounded input.
        # The full archived90 coefficients below remain byte-bound and unchanged.
        return (observed+observed.H)/2
    K = coefficient(kf)*ctx.power(2,10240)+coefficient(k1)*ctx.power(2,8192)+coefficient(k0)
    delta = coefficient(D.astype(complex))*ctx.power(2,8192)
    Gp = ctx.diag([8*ctx.power(2,-4096),ctx.mpf(3)/10*ctx.power(2,-4096)])
    sqrtG = ctx.diag([ctx.sqrt(Gp[a,a]) for a in range(2)])
    inverse = ctx.inverse(sqrtG)
    def wave(stiffness: Any) -> tuple[Any,Any]:
        p = sqrtG*stiffness*sqrtG
        determinant = ctx.re(ctx.det(p)); trace = ctx.re(p[0,0]+p[1,1])
        if determinant<=0 or trace<=0:
            raise ArithmeticError('Actual scaled principal comparison not positive')
        root = (p+ctx.sqrt(determinant)*ctx.eye(2))/ctx.sqrt(trace+2*ctx.sqrt(determinant))
        precision = inverse*root*inverse
        return precision,ctx.inverse(precision)/2
    Aold,Cold = wave(K); Anew,Cnew = wave(K+delta)
    residual = ctx.norm(Anew*Gp*Anew-K-delta)/ctx.norm(K+delta)
    mix = Anew[1,0]/Anew[1,1]
    wrongmix = ctx.norm(Anew-ctx.diag([Anew[0,0],Anew[1,1]]))/ctx.norm(Anew)
    covariance_order = Cold-Cnew
    if residual>=ctx.mpf('1e-850') or mix==0 or wrongmix==0 or ctx.re(covariance_order[0,0])<=0:
        raise ArithmeticError('Physical unit/mixing/newCov principal control failed')
    return {'scope':'Explicit Hermitian part of OLD rounded principal coefficients + NEW Hessian observation; principal2 diagnostic ONLY, not full90 covariance/adoption',
        'principal_input_Hermitian_projection_defects':{name:float(np.max(abs(matrix[np.ix_(ids,ids)]-matrix[np.ix_(ids,ids)].conj().T))) for name,matrix in zip(('Kfast','K1','Kunit'),(kf,k1,k0),strict=True)},
        'arithmetic_digits':ctx.dps,'q_integer':row['q_integer'],'physical_indices':ids,
        'actual_saved_PC_coefficient_abs':float(scores[i,jc]),'Jordan_diagonal_addition_divided_Lambda':float(D[i,i]),
        'new_AGA_relative_defect':ctx.nstr(residual,35),'new_conditional_M_nonzero':ctx.nstr(mix,35),
        'wrong_reset_M_precision_relative_defect':ctx.nstr(wrongmix,35),
        'old_covariance_reuse_Riccati_relative_defect':ctx.nstr(ctx.norm(Aold*Gp*Aold-K-delta)/ctx.norm(K+delta),35),
        'old_Phi_C_divided_2minus6144':ctx.nstr(Cold[0,0]*ctx.power(2,6144),35),
        'new_Phi_C_divided_2minus6144':ctx.nstr(Cnew[0,0]*ctx.power(2,6144),35),
        'new_colour_C_divided_2minus7168':ctx.nstr(Cnew[1,1]*ctx.power(2,7168),35)}


def full_comparison(snapshot: dict[str, Any], physical: dict[str, Any], data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    descriptor = snapshot['descriptor']
    if descriptor['coordinate_order']['Phi'] != data['geom']['labels'] or max(abs(np.asarray(descriptor['numerical_display_g'])-np.asarray([float(g) for g in G])))>3e-16:
        raise ArithmeticError('Saved comparison coordinate/centre does not bind current display')
    arrays: dict[str,np.ndarray] = {'Jordan_D60':physical['D60'],'Jordan_D45':physical['D45'],
        'Jordan_D90_divided_Lambda':physical['D90'],'normal_Q':physical['Q'],'H_orbit':physical['H']}
    records = []; conjugacy = 0.; real_degrees = 0; selfinverse = 0; pairs = 0
    byq = {tuple(row['q_integer']):row for row in snapshot['all64q']}
    for row in snapshot['all64q']:
        q = tuple(row['q_integer']); partner = tuple((-n)%4 for n in q)
        spectral = row['spectral']; mate = byq[partner]['spectral']
        blocks = {}
        for name in ('G_divided_chi','K_Lambda5over4','K_Lambda','K_unit'):
            matrix = decode(spectral[name]); blocks[name] = matrix
            if matrix.shape != (90,90) or np.max(abs(matrix-matrix.conj().T))>1e-10:
                raise ArithmeticError('Full PH90 coefficient shape/Hermitian failure')
            conjugacy = max(conjugacy,float(np.max(abs(matrix.conj()-decode(mate[name])))))
            arrays['_'.join(map(str,q))+'_'+name] = matrix
        G0 = blocks['G_divided_chi']
        d = 1-np.exp(2j*np.pi*np.asarray(q)/4)
        expectedG = np.zeros((90,90),complex); expectedG[:45,:45] = 8*np.eye(45)
        expectedG[45:,45:] = .3*np.eye(45)+np.kron(np.outer(d,d.conj()),np.linalg.inv(physical['H']))
        metricdefect = float(np.max(abs(G0-expectedG)))
        if metricdefect>1e-10:
            raise ArithmeticError('Actual existing Gauss cometric convention mismatch')
        if q <= partner:
            if q == partner:
                selfinverse += 1; real_degrees += 90
            else:
                pairs += 1; real_degrees += 180
        rowrecord:dict[str,Any] = {'q_integer':list(q),'physical_complex_modes':90,
            'Jordan_addition_rank':35,'Gauss_cometric_defect':metricdefect,
            'PC_Lambda_matrix_norm_RETAINED':float(np.linalg.norm(blocks['K_Lambda'][:45,45:],2)),
            'PC_unit_matrix_norm_RETAINED':float(np.linalg.norm(blocks['K_unit'][:45,45:],2)),
            'common_G_Loewner':'conditional exact0<CJ<=CGram; comparison arrays do not define actual science coefficients'}
        if q == (0,0,0):
            oldslow = blocks['K_Lambda'][:45,:45]
            newslow = oldslow+physical['D45']
            _,Co = matrix_wave(8*np.eye(45),oldslow)
            _,Cn = matrix_wave(8*np.eye(45),newslow)
            rowrecord['slow_principal_ONLY_scaled_covariance'] = {'physical_scale':'2^-6144',
                'old_trace_observation':float(np.trace(Co).real),'new_trace_observation':float(np.trace(Cn).real),
                'covariance_order_min_observation':float(np.linalg.eigvalsh(Co-Cn)[0]),
                'NOT_full_joint_covariance':True,'new_slow_stiffness_min_observation':float(np.linalg.eigvalsh(newslow)[0])}
        records.append(rowrecord)
    if conjugacy>1e-10 or (real_degrees,selfinverse,pairs)!=(5760,8,28):
        raise ArithmeticError('All64 canonical reality/35 mode boundary failed')
    return {'all64q':records,'conjugacy_max_defect':conjugacy,'real_degrees':real_degrees,
        'selfinverse_q':selfinverse,'conjugate_pairs':pairs,'full_PH90_PC_matrices_preserved':True,
        'new_full_joint_Cov_computed':False,'new_full_conditional_M_computed':False,
        'principal_physical_comparison':principal_comparison(byq[(0,0,0)],physical['D90'])},arrays


def run(comparison: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    algebra,symbolic,gamma,matrices = exact_algebra()
    # Explicit display override ONLY: this path does not call either mean solver.
    data = old.load_mean(g_override=G,mean_descriptor={'law':'NEW Jordan branch COMPARISON CENTRE ONLY; no stationary/root/old-law/source adoption'})
    physical = physical_addition(data,symbolic,gamma,matrices)
    snapshot,snapshot_bytes = comparison_snapshot(comparison)
    full,arrays = full_comparison(snapshot,physical,data)
    controls = exact_action_control()
    sources = {**old.CAPTURED,Path(__file__).name:SELF}
    snapshots = {COMPARISON_NAME: {'sha256':COMPARISON_SHA,'archived_in_full':False,'reason':'Only selected scientific coefficient inputs are used; full2GB contains unconsumed PH arrays.'}}
    for info in (data['descriptor']['root_receipt'],data['descriptor']['full_receipt']):
        path = Path(info['path']); raw = path.read_bytes()
        if sha(raw) != info['sha256']:
            raise ArithmeticError('Display override prerequisite bytes changed')
        snapshots[path.name] = {'sha256':sha(raw),'bytes':len(raw),'scope':'old descriptor/native geometry ONLY; not new root adoption'}
    unchanged = {name:(ROOT/name).read_bytes()==raw for name,raw in sources.items()}
    if not all(unchanged.values()):
        raise ArithmeticError('Imported source changed during supplier execution')
    snapshot_path = output.with_name(PREFIX+'_comparison_inputs_20261010.json')
    snapshot_path.write_bytes(snapshot_bytes)
    npz = output.with_suffix('.npz'); np.savez_compressed(npz,allow_pickle=False,**arrays)
    archive_path = output.with_name(PREFIX+'_immutable_input_source_'+sha(snapshot_bytes)[:16]+'.zip')
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(snapshot_path.name,snapshot_bytes)
        for name,raw in sources.items():
            archive.writestr('source/'+name,raw)
        for info in (data['descriptor']['root_receipt'],data['descriptor']['full_receipt']):
            path = Path(info['path']); archive.write(path,'old_geometry_input/'+path.name)
        archive.writestr('binding.json',json.dumps({'selected_science_input_sha256':sha(snapshot_bytes),'old_full_receipt_sha256':COMPARISON_SHA,'sources':{name:sha(raw) for name,raw in sources.items()}},indent=2))
    return {'governing_plan':'issue-222-induced-photon-delta13-continuation-20261008',
        'governing_revision':'6079658737e4b1fcb93e8ac99b7e10e01ae0bb6ea7929cf5102d2e3b51e3e786',
        'stage':PROOF['stage'],'proof':PROOF,'exact_algebra':algebra,
        'comparison_center':{'g_exact':[str(g) for g in G],'g_float':[float(g) for g in G],'not_stationary':True},
        'physical_normal_addition':{'eigenvalues_D60_divided_Lambda':physical['eigen60'].tolist(),
            'eigenvalues_D45_divided_Lambda':physical['eigen45'].tolist(),'rank':35,'Gram_kernel':10,
            'controls':physical['controls'],'normal_labels':data['geom']['labels'],'matrices_npz':npz.name},
        'full64q_comparison':full,'action_controls':controls,
        'source_sha256':{name:sha(raw) for name,raw in sources.items()},'sources_unchanged':unchanged,
        'selected_science_input':{'path':snapshot_path.name,'sha256':sha(snapshot_bytes),'bytes':len(snapshot_bytes)},
        'old_input_scope':snapshots,'immutable_archive':{'path':archive_path.name,'sha256':sha(archive_path.read_bytes())},
        'arrays_sha256':sha(npz.read_bytes()),'runtime':{'python':sys.executable,'version':sys.version,
            'numpy':np.__version__,'sympy':sp.__version__,'ijson':ijson.__version__},'elapsed_seconds':time.monotonic()-started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,default=ROOT/COMPARISON_NAME)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.comparison,args.output)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'stage':result['stage'],'rank':result['exact_algebra']['exact_rank'],
        'normal_rank':result['physical_normal_addition']['rank'],'all_q':len(result['full64q_comparison']['all64q']),
        'real_degrees':result['full64q_comparison']['real_degrees'],'new_full_Cov_computed':False,
        'input_sha256':result['selected_science_input']['sha256'],'elapsed_seconds':result['elapsed_seconds']}))
