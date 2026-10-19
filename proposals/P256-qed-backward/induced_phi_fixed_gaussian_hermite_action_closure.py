"""Fixed-root full5760 scalar-packet Hermite action closure, not SOURCE adoption.

Only composed archived receipts are consumed. No root/mean/producer is run.
The scalar differential estimate holds with any unit fixed-N fibre vector;
occupied W and charged couplings are deliberately outside this certificate.
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
import mpmath as mp
import sympy as sp
import numpy as np
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_gauss_quantum as owner

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).read_bytes()
PREFIX = 'induced_phi_fixed_gaussian_hermite_action_closure'
INPUT_SHA = '919cb3c95ec4817eb0605111397d41e31ddf0f8be0a0b0a20a3312d8ba6f1091'
D, NS, NC, NF = 5760, 45, 2880, 5715
R = F(2)**-20
sys.set_int_max_str_digits(0)
PROOF = {
 'operator': 'G0 is the dimensionless actual root normal cometric; Gphysical=chi G0. Kphysical is the FULL all65 static root stiffness. H0=-chi/2 partial^T G0 partial+r^T Kphysical r/2, A Gphysical A=Kphysical, C=.5 A^-1, E0=.5Tr(Gphysical A). A, Xi, C0, M are frozen at the captured root. The reference is Href=H0+T, T=sum45 s_i[-chi/2 partial^T dG_i partial+c^T dKphysical_i c/2-Elead_i_prime]. Parent supplies the scalar residual of the actual bare force plus the exact leading fast zero-point force. Gphysical is NOT multiplied by chi twice.',
 'product_rule': '(H0-E0)(P f_A)/f_A=chi[(grad P)^T G0 A r-.5 G0:Hess P]. This identity keeps every slow/fast derivative and all mixed precision entries. Actual v=(1-P)f_A gives MINUS this commutator. T(v)=T(f_A)-P T(f_A)-[T,P]f_A. Therefore (Href-E0)v=T(f_A)-(H0-E0)(P f_A)-P T(f_A)-[T,P]f_A. A plus-P source doubles rather than cancels the retained forcing.',
 'noncommuting_colour': 'Leading physical Acolour=2^7168 A0colour, dKphysical=2^10240 dK0, dGphysical=2^-4096 dG0. XiPhysical=2^7168 X where A0 G0 X+X G0 A0=.5(dK0-A0 dG0 A0). The physical commutator coefficient is 2^3072 times the scaled forcing. P=sum_i s_i[c^T XiPhysical_i c-Tr(C0 XiPhysical_i)] has dimensionless physical size 2^-3072; Hfast times P has scale2^3072*2^-3072=1, not an arbitrary comparison*2^-1024. The constant identity Tr(C0*.5(dK-A dG A))=chi Tr(G0 XiPhysical) follows cyclic trace. It supplies the exact mean-subtracted retained colour kinetic AND potential forcing.',
 'physical_source': 'Actual P uses z=c+M s, M=A_CC^-1 A_CP in q0, centering by the LEADING C0, not by the full conditional covariance. Let P0 use c and Pm=P-P0. Pm=2 sum_i s_i c^T Xi_i M s+sum_i s_i(Ms)^T Xi_i Ms. With c=z-Ms, triangle bounds it by J[2M |s|^2 |z|+M^2 |s|^3], J=(sum_i||Xi_i||F^2)^.5 over FULL2880 real colour. Exact joint conditioning gives independent s,z; retained M and full5760 covariance are never set to zero.',
 'fastPhi': 'The actual normal injection Q(s)=P(s)Q0[Q0^T P(s)Q0]^-1/2 satisfies Q(s)^T Q(s)=I and O(s)^T Q(s)=0. At f=C=0 these identities make the nonzero-q fastPhi cometric8I and its colour cross zero at EVERY uniform slow s. Alignment has Hessian(qhat^2/40)I in that orthonormal fastPhi basis, independent of the uniform frame: sum_x f_x=0 removes its uniform linear coupling, and orbit orthogonality removes the fastPhi/colour Hessian cross. Thus its leading precision is sqrt(qhat^2/320)*2^7168 I independent of g, derived from both G and K, not assumed. This is a leading primitive statement; all matter PH90 mixing remains in fullA and epsilon.',
 'q0_slow': 'The uniform slow injection is Q0/8, NOT the off-mean Q(s). Its orbit cross B(s)=Q0^T O(s)/8 vanishes at root. Its Schur metric I/8-B(s) H(s)^-1 B(s)^T has derivative0 at root and inverse8I+O(s^2), not globally constant8I. The local geometry owner prices this O(s^2) branch separately. Our retained linear dG at root is colour-only, hence div(sum s_i dG_i)=0 exactly; no unpriced first-order divergence appears in T.',
 'perturbation': 'Define the LEADING reference precision Abar=diag(ArootSlow, AfastPhiPrimitive, AcolourPrimitive) with full static slow stiffness and primitive fast blocks. Reviewed relative K form epsilon gives sqrt(1-epsilon) Abar<=A<=sqrt(1+epsilon) Abar, so E=Abar^-1/2(A-Abar)Abar^-1/2 obeys ||E||<=delta=epsilon/(1+sqrt(1-epsilon)). In y=Abar^1/2 r the FULL covariance satisfies 1/(2sqrt(1+epsilon)) I<=Cy<=rho/2 I, rho=(1-epsilon)^-1/2. This is full5760 Loewner control with all65 PH matrices, NOT primitive OP jets relabelled as a fullHS bound.',
 'polynomial_estimates': 'For degree<=3 Q in the true joint Gaussian, ||D_y Q||2<=sqrt(6sqrt(1+epsilon))||Q||2 and ||D_y^2 Q||2<=2sqrt6 sqrt(1+epsilon)||Q||2. Tensor hypercontractivity ||Q||p<=(p-1)^(degree/2)||Q||2 and Holder are used below. ||y||2<=sqrt(rho*5760/2); ||y^2||4<=sqrt105 rho*5760/2 by eigenbasis Minkowski/Wick. ||s||4<=3^.25 sqrt(45 cs). These covariance inequalities do not assume product q packets.',
 'commutator_remainder': 'For P0 (linear in s), the leading slow commutator is bounded by OmegaSlow*3sqrt3*sqrt(6sqrt(1+epsilon))*||P0||2*sqrt(45rho/2). The fullA-minus-Abar commutator is bounded by Omega*delta*3sqrt3*sqrt(6sqrt(1+epsilon))*||P0||2*sqrt(5760rho/2). The EXACT fullH0 Gaussian number operator sends degree3 polynomials to energy<=3OmegaFull, hence ||(H0-E0)(Pm f_A)||2<=3OmegaFull||Pm||2. No fast Hessian is estimated by slow radius times Amax.',
 'linear_forcing_remainder': 'Only the kinetic score changes when f_Abar is replaced by fixed full f_A in T: trace dG(A-Abar) and r^T(A dG A-Abar dG Abar)r. In y their operator bounds are OmegaG*delta*2880 and OmegaG*delta(2+delta)|y|^2. Multiplication by s and Holder yield the displayed precision_score_difference. The dK term is unchanged. P*T(f_A) has norm<=27p*(T0norm+precision_difference) because T(f_A)/f_A has degree<=3. For [T,P] use chi s[(gradP)^T dG A r-.5 dG:HessP]. L6 Holder on (s,D_yP,y) gives25; L4 on (s,D_y^2P) gives3. Thus mixed_action below is a derived differential norm, not a descriptor-asserted constant.',
 'normalization': 'b6<=1 does not enlarge the derived norm. ||b6(1-P)f_A||>=1-p-sqrt(12TrC/R^2)>1/2 gives normalization upper2. PREnormalization and PREPARED=2*PRE are separate stored endpoints. [Href,b6] is NOT included; parent adds the local owner referenceb6 price once. There is no second Rayleigh-factor2 here: this is an action norm, not a scalar expectation perturbation.',
 'scope': 'For fixed scalar b6(1-P)f_A tensor ANY unit vector in fixedN8324 body Fock, scalar Href acts as identity on that fibre, so these norms are uniform in the fibre (no Slater restriction). They are NOT an operator bound on arbitrary body normal wavefunctions and cannot by themselves establish unbounded-current graph transfer. Actual occupied W(r) independent of Theta, its realKato derivatives, Noether/localgeometry and charged uniform operator coupling are separately parent-owned. No claim HE commutes with a field-dependent frame is made.',
 'conditional': 'Captured root existence/admission, positive static full65 sea and slow q0 lower, exact scientific rational1/5 and7/10 law, canonical normal chart and reviewed covariance/Loewner hypotheses remain conditional. No root solve or adiabatic f_A(s) or implicitRoot rerun occurs. No whole SOURCE, globalground, photonpole, current, commoncone, or originalSOURCE conclusion.'
}

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def price(packet: dict[str, Any], H: list[Any]) -> dict[str, Any]:
    iv = owner.iv
    iv.dps = 110
    ex, ep = owner.exact, owner.endpoint
    b = packet['gaussian']['bounds']
    up = lambda key: ex(F(b[key]['upper']))
    eps, p, M = up('relative_full_form_error'), up('corrector_L2'), up('fullmatrix_M')
    rho = 1/iv.sqrt(1-eps)
    delta = eps/(1+iv.sqrt(1-eps))
    sq = iv.sqrt(1+eps)
    hmin = min(ep(h,0) for h in H)
    hmax = max(ep(h,1) for h in H)
    Gmax = ex(F(3,10)+F(12)/hmin)
    acmin = iv.sqrt(ex(hmin/5)/Gmax)
    acmax = iv.sqrt(ex((hmax/5+F(21,5))/F(3,10)))
    if ep(acmin,1)>ep(iv.sqrt(ex(F(2,320))),0) or ep(acmax,0)<ep(iv.sqrt(ex(F(12,320))),1):
        raise ArithmeticError('Conservative fast precision bounds omit actual fastPhi')
    # FULL65 K0 operator norm, not primitive colour derivativeHS.
    sea = sum(ex(F(r['load']))*ex(F(r['matter_K0_to_K3_unscaled'][0]['upper'])) for r in packet['jets']['profiles'])
    root = packet['source']['input_descriptor']['exact_NEW_allq_BO_mean']
    gmax = F(root['parameters']['g0']['upper'])+3*F(root['parameters']['gs']['upper'])
    Kslow = ex(4096*(1+gmax))+sea
    apmax = iv.sqrt(Kslow/8)
    # Force bs=.3*slow_lower/2 is smaller than the slow K lower.
    # It is a legitimate conservative K lower; its LOWER endpoint is used.
    slowlo = ex(F(packet['force']['outward_constants']['bs']['lower']))
    cs = rho*iv.sqrt(ex(8))/(2*iv.sqrt(slowlo))*ex(F(2)**-6144)
    cf = rho/(2*acmin)*ex(F(2)**-7168)
    J2 = ex(0)
    for group in packet['gaussian']['coefficients']['all_physical_q_groups']:
        for e in group['entries']:
            x = [owner.enclosure(F(e[k]['lower']), F(e[k]['upper'])) for k in ('XL','XT')]
            J2 += ex(group['multiplicity'])*(abs(x[0])**2+2*abs(x[1])**2)
    J = iv.sqrt(J2)*ex(F(2)**7168)
    def moment(n: int, variance: Any, degree: int) -> Any:
        factor = 1
        for j in range(degree):
            factor *= n+2*j
        return iv.sqrt(ex(factor)*variance**degree)
    pm = J*(2*M*moment(NS,cs,2)*moment(NF,cf,1)+M**2*moment(NS,cs,3))
    p0 = p+pm
    omegaS = ex(8)*apmax*ex(F(2)**2048)
    omegaF = Gmax*acmax*ex(F(2)**3072)
    omega = ex(max(ep(omegaS,1),ep(omegaF,1)))
    omegaFull = sq*omega
    omegaG = up('uniform_primitive_G1_upper')*acmax*ex(F(2)**3072)
    # ||dKcolour||op<=H1/5 from actual affine orbit, not fullHS54.
    H1 = iv.sqrt(ex(8*hmax))/4
    omegaK = H1/(5*acmin)*ex(F(2)**3072)
    s2 = iv.sqrt(NS*cs)
    y2 = iv.sqrt(rho*D/2)
    dyp = iv.sqrt(6*sq)*p
    hyp = 2*iv.sqrt(ex(6))*sq*p
    gradfactor = iv.sqrt(6*sq)
    slow = omegaS*3*iv.sqrt(ex(3))*gradfactor*p0*iv.sqrt(NS*rho/2)
    drift = omega*delta*3*iv.sqrt(ex(3))*gradfactor*p0*y2
    mixed = 3*omegaFull*pm
    diff = omegaG*delta/2*ex(3)**ex(F(1,4))*s2*(NC+(2+delta)*iv.sqrt(ex(105))*rho*D/2)
    T0 = (omegaG+omegaK)*ex(3)**ex(F(1,4))*s2*(iv.sqrt(ex(105))*rho*D/4+ex(F(NC,4)))
    linearP = 27*p*(T0+diff)
    commTP = omegaG*s2*(25*(1+delta)*dyp*y2+ex(F(3,2))*iv.sqrt(ex(NC))*hyp)
    pieces = dict(slow_Hermite_action=slow,full_precision_drift_difference=drift,
                  exact_conditional_shift_action=mixed,full_precision_linear_score_difference=diff,
                  retained_linear_operator_times_P=linearP,retained_linear_commutator_with_P=commTP)
    total = sum(pieces.values())
    defect = p+iv.sqrt(12*up('full_trace_covariance')/ex(R)**2)
    if ep(defect,1) >= F(1,2):
        raise ArithmeticError('Compact actual source normalization lower fails')
    if ep(eps,1) >= F(1,4) or ep(slowlo,0) <= 0:
        raise ArithmeticError('Frozen full pencil hypotheses invalid')
    pack = owner.bounds
    return dict(arithmetic='mpmath interval110 digits, exact Fraction inputs and endpoints; display decimals NOT upper evidence',
        constants={k:pack(v) for k,v in dict(epsilon=eps,delta=delta,rho=rho,slow_covariance_operator_upper=cs,
             conditional_fast_covariance_operator_upper=cf,full_colour_Xi_HS=J,conditional_shift_Pm_L2=pm,
             leading_P0_L2=p0,full_slow_K_operator_upper=Kslow,leading_colour_precision_lower=acmin,
             leading_colour_precision_upper=acmax,leading_maximum_frequency=omega,full_maximum_frequency=omegaFull,
             slow_maximum_frequency=omegaS,kinetic_linear_frequency_coefficient=omegaG,
             potential_linear_frequency_coefficient=omegaK,leading_centered_linear_action_norm=T0,normalizer_defect=defect).items()},
        component_PREnormalization={k:pack(v) for k,v in pieces.items()},
        action_PREnormalization=pack(total),action_PREPARED=pack(2*total),normalization_upper='2',
        force_difference_INCLUDED=False,force_difference_PARENT_ONCE=packet['force']['actual_action_input_budgets']['prepared_total_force_action_upper'],
        root_mean_shift_PARENT_SEPARATE_PURE_GRADIENT=packet['force']['actual_action_input_budgets']['mean_shift_omitted_coefficients'],
        excluded=['Primitive nonlinear prepared1.971525295375008e-145','Local geometry prepared3.757996394599192e-138 including q0 slow O(s^2)','Real occupied Kato W and Etilde derivatives','Charged uniform coupling and actual non-scalar charged sea','Reference b6 commutator local-owner prepared2.651879433589567e-1059','Scalarforce difference forcevariance prepared2.440645450774814e-279 (parent once)','Archived root mean-shift pure-gradient7.652083415466412e-1824'],
        fibre_uniform=True,arbitrary_normal_wavefunction_operator_bound=False)

def symbolic_control() -> dict[str, Any]:
    x,y,s = sp.symbols('x y s', real=True)
    r = sp.Matrix([x,y])
    A = sp.Matrix([[2,1],[1,3]])
    G = sp.Matrix([[3,1],[1,2]])
    X = sp.Matrix([[1,2],[2,-1]])
    dG = sp.Matrix([[1,-1],[-1,2]])
    dK = 2*(A*G*X+X*G*A)+A*dG*A
    Xi = (dK-A*dG*A)/2
    C = A.inv()/2
    P = (r.T*X*r)[0]-sp.trace(C*X)
    grad = sp.Matrix([sp.diff(P,x),sp.diff(P,y)])
    hess = sp.hessian(P,(x,y))
    comm = (grad.T*G*A*r)[0]-sp.trace(G*hess)/2
    target = (r.T*Xi*r)[0]-sp.trace(C*Xi)
    if sp.expand(comm-target) != 0:
        raise ArithmeticError('Exact noncommuting action identity failed')
    wrong = Xi/(2*sp.trace(A*G)/2)
    wrong_order = A*G*wrong+wrong*G*A-Xi
    wrong_sign = sp.expand(target+comm)
    if wrong_order == sp.zeros(2) or wrong_sign == 0 or A*G == G*A:
        raise ArithmeticError('Exact symbolic adverse control blind')
    # Same action formula on a coupled physical slow/fast precision.
    mixedA = sp.Matrix([[3,1],[1,2]])
    mixedP = s*(y+x/sp.Integer(2))**2
    physical_comm = (sp.Matrix([sp.diff(mixedP,x),sp.diff(mixedP,y)]).T*G*mixedA*r)[0]-sp.trace(G*sp.hessian(mixedP,(x,y)))/2
    dropped_mixed = sp.expand(physical_comm.subs(x,0)-physical_comm)
    if dropped_mixed == 0:
        raise ArithmeticError('Mixed differential adverse control blind')
    return dict(exact_identity=str(sp.expand(comm-target)),noncommutator=str(A*G-G*A),
         wrong_commuting_order=str(wrong_order),wrong_source_sign=str(wrong_sign),
         wrong_dropped_mixed=str(dropped_mixed),covariance_factor_two_rejected=bool(sp.trace(A.inv()*Xi)!=sp.trace(G*X)))

def physical_action_control(packet: dict[str, Any], H: list[Any]) -> dict[str, Any]:
    """Independent direct product rule on every actual q and all45 Xi."""
    h = np.array([owner.midpoint(v) for v in H])
    groups = {r['qhat2']:r for r in packet['gaussian']['coefficients']['all_physical_q_groups']}
    rng = np.random.default_rng(222576010)
    slow = rng.normal(size=45)
    slow /= np.linalg.norm(slow)
    total = target = doubled = 0.
    maxdefect = wrongorder = wrongcov = 0.
    qrows = []
    Xzero: Any = None
    def realify(v: np.ndarray) -> np.ndarray:
        return np.block([[np.real(v),-np.imag(v)],[np.imag(v),np.real(v)]])
    for q in owner.QINTEGER:
        d = 1-np.exp(2j*np.pi*np.asarray(q)/4)
        q2 = int(round(float(np.vdot(d,d).real)))
        PL = np.outer(d,d.conj())/q2 if q2 else np.zeros((3,3))
        PT = np.eye(3)-PL
        AL = np.sqrt((h/5)/(.3+q2/h))
        AT = np.sqrt((h/5+.35*q2)/.3)
        A = np.kron(PL,np.diag(AL))+np.kron(PT,np.diag(AT))
        G = .3*np.eye(45)+np.kron(np.outer(d,d.conj()),np.diag(1/h))
        C = np.linalg.inv(A)/2
        tensors = {k:np.zeros((45,15,15)) for k in ('XL','XT','XiL','XiT')}
        for entry in groups[q2]['entries']:
            for key in tensors:
                interval = entry[key]
                tensors[key][entry['Phi'],entry['a'],entry['b']] = float((F(interval['lower'])+F(interval['upper']))/2)
        X = np.asarray([np.kron(PL,x)+np.kron(PT,t) for x,t in zip(tensors['XL'],tensors['XT'])])
        Xi = np.asarray([np.kron(PL,x)+np.kron(PT,t) for x,t in zip(tensors['XiL'],tensors['XiT'])])
        defect = float(np.max(abs(A@G@X+X@G@A-Xi)))
        maxdefect = max(maxdefect,defect)
        scalarX = Xi/(2*float(np.trace(G@A).real/45))
        wrongorder = max(wrongorder,float(np.linalg.norm(A@G@scalarX+scalarX@G@A-Xi)))
        qrows.append(dict(q=list(q),all45_Sylvester_defect=defect))
        if not any(q):
            Xzero = X.real
        partner = tuple((-n)%4 for n in q)
        if q>partner:
            continue
        selfinverse = q==partner
        xr = X.real if selfinverse else np.asarray([realify(v) for v in X])
        xir = Xi.real if selfinverse else np.asarray([realify(v) for v in Xi])
        ar,gr,cr = (v.real if selfinverse else realify(v) for v in (A,G,C))
        z = rng.normal(size=ar.shape[0])
        xx = np.einsum('i,iab->ab',slow,xr)
        force = np.einsum('i,iab->ab',slow,xir)
        # Literal Hess(Pf) cross/trace; no supplier action forwarding.
        direct = float((2*(ar@z)@(gr@(xx@z))-np.trace(gr@xx)).real)
        expected = float((z@force@z-np.trace(cr@force)).real)
        wrongcov = max(wrongcov,abs(float(np.trace(cr@force).real)))
        total += direct
        target += expected
        doubled += (1 if selfinverse else 2)*direct
    mixed = np.real(owner.decode(packet['gaussian']['full_mixed_coefficient']['coefficient']))
    s,z = rng.normal(size=45),rng.normal(size=45)
    Xs = np.einsum('i,iab->ab',s,Xzero)
    omittedMixed = float(np.linalg.norm(2*mixed.T@Xs@z))
    if maxdefect>1e-10 or abs(total-target)>1e-9:
        raise ArithmeticError('Actual full64/all45 direct differential cancellation failed')
    if min(wrongorder,wrongcov,abs(doubled-total),omittedMixed,abs(2*target))<=1e-8:
        raise ArithmeticError('Actual source order/covariance/pair/mixed/sign controls blind')
    return dict(all64_all45_coefficient_actions=qrows,canonical_once_action=total,canonical_once_target=target,
       direct_product_rule_defect=abs(total-target),wrong_plus_P_residual=abs(2*target),
       wrong_scalar_commuting_order_defect=wrongorder,wrong_covariance_no_half_centering_defect=wrongcov,
       wrong_qpair_second_factor_defect=abs(doubled-total),wrong_M_zero_actual_source_gradient_coefficient=omittedMixed,
       numerical_status='Finite actual coefficient observations only; interval/global sourceweighted inequalities are separate')

def actual_controls(packet: dict[str, Any], H: list[Any]) -> dict[str, Any]:
    iv, ex, ep = owner.iv, owner.exact, owner.endpoint
    coeff = packet['gaussian']['coefficients']
    actual = packet['gaussian']['all64_records']
    blocks = packet['gaussian']['real_blocks']
    if len(actual)!=64 or len(blocks)!=36 or sum(r['real_dimension'] for r in blocks)!=D:
        raise ArithmeticError('Actual full5760/all64 real pairing incomplete')
    if sorted(tuple(r['q']) for r in actual)!=sorted(owner.QINTEGER):
        raise ArithmeticError('Actual q coefficients changed')
    if sum(g['multiplicity'] for g in coeff['all_physical_q_groups'])!=64:
        raise ArithmeticError('Coefficient multiplicity is not full64')
    seen = set()
    residual = F(0)
    wrongDG = F(0)
    wrongSign = F(0)
    positive = 0
    coefficient_count = 0
    for g in coeff['all_physical_q_groups']:
        q2 = g['qhat2']
        wL = [iv.sqrt(ex(F(3,50))*h+ex(F(q2,5))) for h in H]
        wT = [iv.sqrt(ex(F(3,50))*h+ex(F(21*q2,200))) for h in H]
        for e in g['entries']:
            i,a,b = e['Phi'],e['a'],e['b']
            seen.add(i)
            vals = {k:owner.enclosure(F(e[k]['lower']),F(e[k]['upper'])) for k in ('dH','XiL','XiT','XL','XT')}
            for kind,w in (('L',wL),('T',wT)):
                difference=(w[a]+w[b])*vals['X'+kind]-vals['Xi'+kind]
                residual=max(residual,ep(abs(difference),1))
                if not ep(difference,0)<=0<=ep(difference,1):
                    raise ArithmeticError('Actual all45 Sylvester interval excludes zero')
                # Actual plus-P action is doubled, not an unrelated toy supplier.
                sign=2*vals['Xi'+kind]
                lower=ep(abs(sign),0)
                wrongSign=max(wrongSign,lower)
                positive+=int(lower>0)
            if q2:
                dropped=vals['XiL']-vals['dH']/10
                wrongDG=max(wrongDG,ep(abs(dropped),0))
            coefficient_count+=g['multiplicity']
    # Hermite operator intentionally has sparse zeros; ALL45 index presence,
    # full source ABI and actual canonical groups prevent a chosen radial fit.
    if seen!=set(range(45)) or wrongDG<=0 or wrongSign<=0 or positive==0:
        raise ArithmeticError(f'Actual adverse controls blind: seen={seen}, dG={wrongDG}, sign={wrongSign}, positive={positive}')
    corrector=packet['source']['exact_corrector']
    if 'C(0)+M PhiSlow' not in corrector or '(1-P)' not in corrector:
        raise ArithmeticError('Actual source ABI mismatch: this proof requires frozen minus-P and plus conditional M')
    # Same physical dimensions: multiply Fourier-pair factor twice changes D.
    double_dim=sum(r['real_dimension']*(2 if r['q']!=r['partner'] else 1) for r in blocks)
    if double_dim==D or F(2)**3072*F(2)**-3072!=1:
        raise ArithmeticError('Physical pairing/unit adverse control blind')
    return dict(all64_bound_to_source=True,all45_indices=sorted(seen),sparse_coefficient_multiplicity_count=coefficient_count,
       exact_Sylvester_interval_residual_upper=str(residual),wrong_plus_P_retained_forcing_coefficient_lower=str(wrongSign),
       wrong_frozen_dG_actual_nonzeroq_coefficient_lower=str(wrongDG),wrong_real_pair_doublecount_dimension=double_dim,
       correct_dimension=D,wrong_covariance_C_equals_Ainverse_rejected_by_symbolic_control=True,
       physical_scale_identity='2^3072*2^-3072=1; omitting chi multiplies forcing by2^4096',
       full_static_mixed_M_retained=packet['gaussian']['full_mixed_coefficient']['norm_observation'],
       scope='Exact coefficient interval inclusion and algebraic adverse controls on actual full64/all45/root source; not old producer or root rerun')

def run(inputs: Path) -> dict[str, Any]:
    start = time.monotonic()
    raw = inputs.read_bytes()
    if sha(raw)!=INPUT_SHA:
        raise ArithmeticError('Exact selected scientific inputs changed')
    packet = json.loads(raw)
    for name, captured in owner.SOURCE_BYTES.items():
        if packet['gaussian']['source_epoch'].get(name)!=sha(captured):
            raise ArithmeticError(f'Supplier source epoch mismatch: {name}')
    owner.iv.dps=110
    root = packet['source']['input_descriptor']['exact_NEW_allq_BO_mean']
    # Construct the native frame at the saved box; do NOT call solve_mean.
    centres = [F(v) for v in root['centre_exact_rational']]
    descriptor={'microscopic_config':packet['source']['input_descriptor']['original_micro_config_UNCHANGED']}
    mean=owner.stationary.input_mean(descriptor,centres)
    native=owner.stationary.full.supplier.quantum.build_native(mean,4)
    g=[owner.enclosure(F(root['parameters'][k]['lower']),F(root['parameters'][k]['upper'])) for k in ('g0','gs')]
    geom=owner.stationary.full.normal_geometry(native,[g[0],g[1],g[1],g[1]])
    H=[sum(owner.exact(F(int(n),2))*v for n,v in zip(row,[g[0],g[1],g[1],g[1]])) for row in owner.masks(geom['M'])]
    with threadpool_limits(limits=1):
        finite=dict(symbolic=symbolic_control(),actual=actual_controls(packet,H),
                    actual_differential_action=physical_action_control(packet,H))
    prices=price(packet,H)
    captured={**owner.SOURCE_BYTES,Path(__file__).name:SELF}
    return dict(status='DERIVED_FIXED_ROOT_FULL5760_SOURCEWEIGHTED_HERMITE_ACTION_NOT_WHOLE_SOURCE_ADMISSION',
       proof=PROOF,dimensions=dict(normal=D,slow=NS,colour=NC,fast=NF,physical_q=64,real_groups=36),
       fixed_inputs_provenance=packet['provenance'],selected_inputs_sha256=sha(raw),
       selected_input_bytes_utf8=raw.decode(),outward_action=prices,controls=finite,
       source_at_execution={n:dict(sha256=sha(b),bytes_utf8=b.decode()) for n,b in captured.items()},
       input_unchanged=inputs.read_bytes()==raw,
       source_unchanged={n:(ROOT/n).read_bytes()==b for n,b in captured.items()},
       runtime=dict(executable=sys.executable,python=sys.version,mpmath=mp.__version__,sympy=sp.__version__),
       elapsed_seconds=time.monotonic()-start)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/(PREFIX+'_inputs.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.inputs)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],outward_action=result['outward_action']['action_PREPARED'],
         controls=result['controls'],elapsed_seconds=result['elapsed_seconds']),indent=2))
