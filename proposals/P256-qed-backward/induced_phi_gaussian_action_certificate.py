"""Derived NEW full5760 Gaussian/cubic supplier-action certificate.

This entrypoint removes algebraic Gaussian/Hermite premises, not the full
interacting Hamiltonian/source debt. Exact input terms are retained separately
from finite floating observations. No root, ground, pole or SOURCE adoption.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import mpmath as mp
import numpy as np
from scipy.linalg import solve_sylvester
from threadpoolctl import threadpool_limits

import induced_phi_nonuniform_gauss_quantum as supplier

ROOT = Path(__file__).resolve().parent
SELF_BYTES = Path(__file__).read_bytes()
PROJECT_PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def upper(value: dict[str, Any]) -> Any:
    return mp.mpf(str(Fraction(value['upper']).numerator)) / Fraction(value['upper']).denominator


def realify(matrix: np.ndarray) -> np.ndarray:
    return np.block([[np.real(matrix), -np.imag(matrix)], [np.imag(matrix), np.real(matrix)]])


def real_trace(matrix: np.ndarray) -> float:
    value = np.trace(matrix)
    if abs(float(np.imag(value))) > 1e-11*(1+abs(float(np.real(value)))):
        raise ArithmeticError('Expected Hermitian real trace')
    return float(np.real(value))


def solve_jet(A: np.ndarray, G: np.ndarray, K1: np.ndarray,
              G1: np.ndarray, K2: np.ndarray, G2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate AGA=K without commuting any factors."""
    A1 = solve_sylvester(A @ G, G @ A, K1 - A @ G1 @ A)
    remainder = (2 * A1 @ G @ A1 + 2 * A1 @ G1 @ A
                 + 2 * A @ G1 @ A1 + A @ G2 @ A)
    A2 = solve_sylvester(A @ G, G @ A, K2 - remainder)
    return A1, A2


def outward_bounds(data: dict[str, Any], matter: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct interval constants from Clifford coefficients, not price labels."""
    supplier.iv.dps = 90
    iv = supplier.iv
    ex = supplier.exact
    ep = supplier.endpoint
    algebraic = supplier.exact_colour_coefficients(data)
    H = [sum(ex(Fraction(int(n),2))*g for n,g in zip(row,data['g']))
         for row in supplier.masks(data['geom']['M'])]
    hlo = min(ep(h,0) for h in H)
    hhi = max(ep(h,1) for h in H)
    slowlo = data['frozen_slow_lower']/2
    sea = sum(Fraction(row['load'])*Fraction(row['sea90_operator_upper']['upper'])
              for row in matter['exact_ALLq_operator_bounds'])
    Kslow = 4096*(1+sum(ep(g,1) for g in data['g']))+sea
    cslow = iv.sqrt(ex(8))/(2*iv.sqrt(ex(slowlo)))
    fastC = ex(0)
    trA0 = ex(0)
    trA20 = ex(0)
    Wick = ex(0)
    colourmax = Fraction(0)
    fastmin: list[Fraction] = []
    for group in algebraic['all_physical_q_groups']:
        q2,count = group['qhat2'],group['multiplicity']
        CL = [iv.sqrt((ex(Fraction(3,10))+ex(q2)/h)/(h/5))/2 for h in H]
        CT = [ex(Fraction(3,20))/iv.sqrt(ex(Fraction(3,50))*h+ex(Fraction(21*q2,200))) for h in H]
        covs = CL+CT+CT
        colourmax = max(colourmax,max(ep(c,1) for c in covs))
        if q2:
            covs += [ex(4)/iv.sqrt(ex(Fraction(q2,5)))]*45
        fastmin.extend(ep(c,0) for c in covs)
        fastC += ex(count)*sum(covs)
        trA0 += ex(count)*sum(1/(2*c) for c in covs)
        trA20 += ex(count)*sum(1/(4*c*c) for c in covs)
        for entry in group['entries']:
            a,b = entry['a'],entry['b']
            XL = supplier.enclosure(Fraction(entry['XL']['lower']),Fraction(entry['XL']['upper']))
            XT = supplier.enclosure(Fraction(entry['XT']['lower']),Fraction(entry['XT']['upper']))
            Wick += ex(count)*(CL[a]*CL[b]*abs(XL)**2+2*CT[a]*CT[b]*abs(XT)**2)
    # Exact relative-form error bounds, all64q, all matter coefficients retained.
    eps0 = ex(2*sea)*ex(Fraction(2)**-1024)/iv.sqrt(ex(slowlo*hlo/5))
    eps0 += ex(5*sea/hlo)*ex(Fraction(2)**-2048)
    eps0 += ex(sea/slowlo)*ex(Fraction(2)**-8192)
    epsnonzero = ex(sea/min(Fraction(1,20),hlo/5))*ex(Fraction(2)**-2048)
    eps = ex(max(ep(eps0,1),ep(epsnonzero,1)))
    if ep(eps,1) >= Fraction(1,4):
        raise ArithmeticError('Exact fullmatrix comparison fails')
    r = 1/iv.sqrt(1-eps)
    rminus = eps/(iv.sqrt(1-eps)*(1+iv.sqrt(1-eps)))
    delta = eps/(1+iv.sqrt(1-eps))
    # Norm of slow covariance bounded by cslow, independent of slow eigenbasis.
    p = iv.sqrt(ex(2)*cslow*Wick*(r**3+ex(1440)*r*rminus**2))*ex(Fraction(2)**-3072)
    acmin = ex(1/(2*colourmax))*ex(Fraction(2)**7168)
    M = ex(45)**ex(Fraction(1,2))*ex(sea)*(ex(Fraction(2)**8192)+1)
    M /= ex(Fraction(2)**-4096)*ex(Fraction(3,10))*(acmin*(1-delta))**2
    slowcmin = iv.sqrt(ex(8))/(2*iv.sqrt(ex(Kslow)))*ex(Fraction(2)**-6144)
    cmin = ex(min(ep(slowcmin,0),min(fastmin)*Fraction(2)**-7168))/iv.sqrt(1+eps)
    trC = r*(fastC*ex(Fraction(2)**-7168)+45*cslow*ex(Fraction(2)**-6144))
    ta = iv.sqrt(1+eps)*(trA0*ex(Fraction(2)**7168)+45*iv.sqrt(ex(Kslow)/8)*ex(Fraction(2)**6144))
    ta2 = (1+eps)*(trA20*ex(Fraction(2)**14336)+ex(Fraction(45,8)*Kslow)*ex(Fraction(2)**12288))
    gp = iv.sqrt(3/cmin)*p
    hp = iv.sqrt(ex(6))*p/cmin
    unweighted = iv.sqrt(ta**2/4+ta2/2)
    rfast = iv.sqrt(r*fastC*ex(Fraction(2)**-7168)*(ta**2/4+ta2/2)+ex(Fraction(5715,2))*ta)
    corrected = (1+27*p)*unweighted+18*gp*iv.sqrt(ta/2)+hp
    r2norm = iv.sqrt(ex(3))*trC
    # Q=x^T B x: ||Q||4<=105^.25 Tr(BC) by eigenbasis Minkowski.
    # Holder for Q_I Q_A², then triangle with A, derives the archived
    # sqrt105/sqrt3 weighted law on the FULL noncommuting Gaussian.
    r2hessian = trC*(iv.sqrt(ex(105))*ta/2+iv.sqrt(ex(3))*iv.sqrt(ta2))
    rfgrad = iv.sqrt(r*fastC*ex(Fraction(2)**-7168)*ta/2+ex(Fraction(5715,2)))
    r2grad = 9*trC*iv.sqrt(ta/2)
    corrected_rfast = (1+27*p)*rfast+18*gp*rfgrad+3*hp*iv.sqrt(r*fastC*ex(Fraction(2)**-7168))
    corrected_r2 = (1+ex(3)**ex(Fraction(7,2))*p)*r2hessian
    corrected_r2 += 2*ex(3)**ex(Fraction(5,2))*gp*r2grad+ex(3)**ex(Fraction(3,2))*hp*r2norm
    # Uniform primitive colour jets throughout the ACTUAL affine slow ball.
    # ||O1||F<=1 follows normalized Pauli T and horizontal Q:
    # sum_a 2||[Ta,deltaPhi]||HS^2<=15||deltaPhi||HS^2,
    # ||deltaPhi||HS^2=||s||^2/16, giving sqrt15/4<1.
    R = ex(Fraction(2)**-20)
    Omax = iv.sqrt(ex(8*hhi))
    tube_hlo = ex(hlo)-(2*Omax*R+R**2)/8
    tube_hhi = ex(hhi)+(2*Omax*R+R**2)/8
    H1 = (Omax+R)/4
    H2 = ex(Fraction(1,4))
    Gmax = ex(Fraction(3,10))+12/tube_hlo
    G1 = 12*H1/tube_hlo**2
    G2 = 12*(2*H1**2/tube_hlo**3+H2/tube_hlo**2)
    Kmin = tube_hlo/5
    Kmax = tube_hhi/5+ex(Fraction(21,5))
    a = iv.sqrt(Kmax/ex(Fraction(3,10)))
    ell = (Gmax/ex(Fraction(3,10)))/(2*iv.sqrt(ex(Fraction(3,10))*Kmin))
    a1 = ell*(H1/5+a**2*G1)
    a2 = ell*(H2/5+2*a1**2*Gmax+4*a*a1*G1+a**2*G2)
    if ep(tube_hlo,0)<=0:
        raise ArithmeticError('Primitive global slow-ball pencil loses positivity')
    values = {'relative_full_form_error':eps,'covariance_ratio_minus_one':rminus,
              'fullmatrix_M':M,'corrector_L2':p,'full_covariance_eigen_lower':cmin,
              'full_trace_covariance':trC,'full_trace_A':ta,'full_trace_A_squared':ta2,
              'corrector_gradient':gp,'corrector_Hessian':hp,'Gaussian_Hessian':unweighted,
              'Gaussian_rfast_Hessian':rfast,'corrected_unweighted_Hessian':corrected,
              'Gaussian_rALL_squared_Hessian':r2hessian,'corrected_rfast_Hessian':corrected_rfast,
              'corrected_rALL_squared_Hessian':corrected_r2,
              'uniform_primitive_H_lower':tube_hlo,'uniform_primitive_G1_upper':G1,
              'uniform_primitive_G2_upper':G2,'uniform_primitive_A1_upper':a1,
              'uniform_primitive_A2_upper':a2,'actual_allq_Wick_X_trace':Wick}
    return {'arithmetic':'mpmath interval 90 digits; exact rational endpoints',
            'bounds':{key:supplier.bounds(value) for key,value in values.items()},
            'independently_reconstructed_exact_coefficients':algebraic,
            'M_proof':'q0 mixed block: (8chi A_PP)A_PC+A_PC(.3chi A_CC)=K_PC. Both sides positive selfadjoint, inverse <=1/(.3chi accmin). ||K_PC||F<=sqrt45*(Lambda+1)*ALLsea. Therefore ||M||<=||K_PC||F/(.3chi accmin^2), WITH fullmatrix cross and 2^-2048 physical scale.',
            'conditional_inputs':'Exact Clifford/horizontal normalization, archived root box, q0 slow lower and all65 static sea90 norm certificates. No matter derivative/complex projector extension is assumed proven.',
            'primitive_bound_proof':'Affine O with ||O1||F<=sqrt15/4<1, H=O*O/8; use Weyl perturbation and exact inverse-jet formulas. K/G bounds run over all q2<=12 and ALL45 slow directions in ||s||<=2^-20. This is a primitive colour matrix-jet certificate, NOT radial-only.'}


def matrix_proof() -> dict[str, Any]:
    """Finite-dimensional identities and inequalities, valid without commutation."""
    return {
        'functional': 'A=G^-1/2 (G^1/2 K G^1/2)^1/2 G^-1/2; AGA=K; C=A^-1/2 (inverse divided by TWO, not inverse square root)',
        'jet_equations': [
            'L(A1)=K1-A G1 A, L(Y)=AGY+YGA',
            'L(A2)=K2-2 A1 G A1-2 A1 G1 A-2 A G1 A1-A G2 A',
            'C1=-.5 A^-1 A1 A^-1; C2=A^-1 A1 A^-1 A1 A^-1-.5 A^-1 A2 A^-1',
            'M=A_CC^-1 A_CP; M1=A_CC^-1(A_CP,1-A_CC,1 M)',
            'M2=A_CC^-1(A_CP,2-A_CC,2 M-2 A_CC,1 M1)'],
        'inverse_lemma': 'Set B=G^1/2 A G^1/2=(G^1/2 K G^1/2)^1/2. L(Y)=R becomes B Yhat+Yhat B=Rhat. Yhat=int_0^infty exp(-tB) Rhat exp(-tB)dt; ||Y||F <= (gmax/gmin)||R||F/(2 sqrt(gmin kmin)). This proves a UNIFORM jet bound whenever stated G/K/jets inequalities hold, not from a numerical residual.',
        'explicit_bound_recurrence': {
            'a': 'sqrt(kmax/gmin)', 'ell': '(gmax/gmin)/(2 sqrt(gmin kmin))',
            'a1': 'ell*(k1+a^2*g1)',
            'a2': 'ell*(k2+2*a1^2*gmax+4*a*a1*g1+a^2*g2)',
            'c1': 'a1/(2*amin^2)', 'c2': 'a1^2/amin^3+a2/(2*amin^2)',
            'm1': '(acp1+acc1*m)/accmin',
            'm2': '(acp2+acc2*m+2*acc1*m1)/accmin'},
        'full_matrix_comparison': 'For common G, (1-e)K0<=K<= (1+e)K0 gives sqrt(1-e)A0<=A<=sqrt(1+e)A0 by operator-monotonic sqrt, hence inverse covariance bounds. Define delta=max(1-sqrt(1-e),sqrt(1+e)-1). In q0 block-diagonal A0, ||M|| <= delta/(1-delta)*sqrt(aPmax/aCmin). This keeps A_CP, not M=0. Improved mixed-block exponents require a separate off-diagonal estimate.',
        'Hermite_resolvent': 'Xi=.5(K1-A G1 A); L(X)=Xi so A1=2X. H=-.5 grad.G.grad+.5 z.K.z, E=.5Tr(GA). (H-E)[(z^T X z-Tr(C X)) f_A]/f_A=z^T Xi z-Tr(GX)=z^T Xi z-Tr(C Xi). Tr(C Xi)=Tr(GX) follows AGA=K and cyclic trace, no commutativity.',
        'primitive_H_jets': 'O(Phi)=orbit(Phi,T) is LINEAR in actual Clifford frame, H=O^*O/8. H1=(O1^*O+O^*O1)/8; H2=O1^*O1/4 on affine rays. (H^-1)1=-H^-1 H1 H^-1; (H^-1)2=2H^-1 H1 H^-1 H1 H^-1-H^-1 H2 H^-1. Thus dG=-dd* tensor H^-1 H1 H^-1 is mandatory at nonzero q.',
        'scope': 'These identities certify the exact full PH90 functional and its 5760-dimensional real direct sum GIVEN positive pencils and derivative bounds. Primitive colour jets are derived/exercised below. Full matter K_j/occupied complex jets remain explicit premises, not inferred from labels.'}


def gaussian_proof() -> dict[str, Any]:
    return {
        'joint_conditioning': 'Write s=PhiSlow and z_C=Ccoord+M s; Gaussian precision completion gives independent s,z with S=.5(A_PP-A_PC A_CC^-1 A_CP)^-1 and Cf=.5 A_CC^-1. Conditional shift sign is PLUS. P=sum_i s_i(z^T X_i z-Tr(C0 X_i)).',
        'exact_L2': 'mu_i=Tr((Cf-C0)X_i). E[P^2]=sum_ij S_ij*(2 Tr(Cf X_i Cf X_j)+mu_i mu_j). All real-pair matrices once. Physical X=2^7168 Xscaled, slow covariance=2^-6144 Sscaled; P carries 2^-3072. FastPhi enters joint covariance and weighted derivatives even where X support is colour only.',
        'full_covariance_L2_bound': 'S<=s I, Cf<=r C0 implies ||P||2^2<=s*(2*r^2 sum_i ||C0^1/2 X_i C0^1/2||F^2+sum_i mu_i^2). |mu_i|<=||C0^-1/2(Cf-C0)C0^-1/2||op*sqrt(nC)*||C0^1/2 X_i C0^1/2||F. No unproved factor2 inflation needed; use explicit Riccati comparison r.',
        'hypercontractivity': 'For degree<=3 real polynomial in whitened independent standard Gaussians, ||P||p<=(p-1)^(3/2)||P||2, p>=2. Decompose into orthogonal Wiener chaoses: E||D_y P||^2=sum k||P_k||2^2<=3||P||2^2; E||D_y^2 P||F^2=sum k(k-1)||P_k||2^2<=6||P||2^2. Unwhitening gives ||grad P||2<=sqrt(3/cmin)||P||2, ||Hess P||2<=sqrt(6)/cmin||P||2 for the FULL covariance eigenvalue cmin.',
        'weighted_Gaussian': [
            'grad f=-A x f; Hess f=((Ax)(Ax)^T-A)f; E||grad f/f||^2=.5Tr(A)',
            'E||Hess f/f||F^2=.25Tr(A)^2+.5Tr(A^2), from Wick and C=.5A^-1. E[rB^2 ||Hess f/f||F^2]=Tr(BC)*(.25Tr(A)^2+.5Tr(A^2))+.5Tr(B)Tr(A), where rB^2=x^T B x.',
            'E[(x^T B x)(x^T D x)]=Tr(BC)Tr(DC)+2Tr(BCDC)',
            'E[(x^T B x)(x^T D x)^2]=Tr(BC)*(Tr(DC)^2+2Tr(DCDC))+4Tr(DC)*Tr(BCDC)+8Tr(BCDCDC)',
            'For any vector/tensor polynomial Q of degree d, ||Q||4<=(3)^(d/2)||Q||2 (component Minkowski). Holder ||P Q||2<=3^((3+d)/2)||P||2||Q||2. This derives conservative weighted-Hessian constants instead of relying on descriptor names.'],
        'derived_corrected_product_bound': {
            'unweighted_Hessian': '||Hess[(1-P)f]||2<= (1+27*p)*sqrt(.25TrA^2+.5Tr(A^2))+2*9*gp*sqrt(.5TrA)+hp; gp=sqrt(3/cmin)*p; hp=sqrt6*p/cmin',
            'rfast_Hessian': '||rF Hess[(1-P)f]||2 <= (1+3^3*p)*HF + 2*3^2*gp*||rF grad f||2 +3*hp*sqrt(Tr C_F); HF=||rF Hess f||2 via the displayed SIXTH Wick moment. Degrees: rF Hess-f tensor can be bounded componentwise with xF tensor Hess-f (degree3).',
            'rALL2_Hessian': 'Use tensor x tensor x tensor Hess-f (degree4), giving (1+3^3.5*p)*H2 +2*3^2.5*gp*||rALL^2 grad f||2 +3^1.5*hp*||rALL^2||2. All weighted norms are Wick traces, not Amax times radius.',
            'bump_and_source': 'Leibniz: D2(b U v)=b U D2v+2(Db)U Dv+(D2b)U v+2b(DU)Dv+2(Db)(DU)v+b(D2U)v. Source v=(1-P) f Etilde Psi. Retain D Etilde and D2 Etilde; ||D Etilde||<=2||E0||B1, ||D2 Etilde||<=2||E0||B2+4||E0||B1^2 when W jets are bounded by B1,B2.'},
        'P_coordinate_derivatives': [
            'At fixed physical Ccoord, z=Ccoord+M s: P_s,a=Y_a+2 sum_i s_i (M e_a)^T X_i z',
            'P_z=2 sum_i s_i X_i z; P_zz=2 sum_i s_i X_i',
            'P_s,a s,b=2(M e_b)^T X_a z+2(M e_a)^T X_b z+2 sum_i s_i(M e_a)^T X_i(M e_b)',
            'When A,X,M vary in a source parameter: P1=sum_i s_i(2 z1^T Xi z+z^T Xi1 z-Tr(C01 Xi+C0 Xi1)), z1=M1 s. P2 keeps z2=M2s, Xi2 and centering derivatives by the same Leibniz rule.'],
        'Gaussian_source_score': 'd f/f=.25Tr(A^-1 A1)-.5x^T A1 x; E(score)=0 and ||score f||2^2=.5Tr(C A1 C A1). d2f/f=(score)^2+.25Tr(A^-1 A2-A^-1 A1 A^-1 A1)-.5x^T A2 x. Source jets cannot be omitted even if f is normalized.',
        'no_admission': 'No nonlinear source residual, globalroot, U1 coupling, pole, common-cone or originalSOURCE promotion.'}


def run(quantum_path: Path, source_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    mp.mp.dps = 90
    sys.set_int_max_str_digits(0)
    inputs: dict[str, bytes] = {str(quantum_path): quantum_path.read_bytes(), str(source_path): source_path.read_bytes()}
    quantum = json.loads(inputs[str(quantum_path)])
    source = json.loads(inputs[str(source_path)])
    if digest(inputs[str(source_path)]) != 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6':
        raise ArithmeticError('Expected exact resumed source capsule')
    matter_info = quantum['input_descriptor']['matter_receipt']
    matter_path = ROOT / matter_info['path']
    inputs[str(matter_path)] = matter_path.read_bytes()
    if digest(inputs[str(matter_path)]) != matter_info['sha256']:
        raise ArithmeticError('Matter input changed')
    matter = json.loads(inputs[str(matter_path)])
    bindings = quantum['input_descriptor']['root_inputs']
    for binding in bindings:
        path = ROOT / binding['path']
        inputs[str(path)] = path.read_bytes()
        if digest(inputs[str(path)]) != binding['sha256']:
            raise ArithmeticError('Root input changed')
    captured = dict(supplier.SOURCE_BYTES)
    for packet in (quantum, source, matter):
        for name, entry in packet['source_at_import'].items():
            raw = entry['bytes_utf8'].encode()
            if digest(raw) != entry['sha256']:
                raise ArithmeticError('Invalid archived source capture')
            if name in captured and captured[name] != raw:
                raise ArithmeticError('Supplier source epoch mismatch')
            captured[name] = raw
    # Consume the archived implicit-root box; NEVER re-solve/retest the root.
    full_packet = json.loads(inputs[str(ROOT / bindings[1]['path'])])
    root = quantum['actual_allq_feedback_mean']
    centres = [Fraction(value) for value in root['centre_exact_rational']]
    mean = supplier.stationary.input_mean(
        {'microscopic_config': full_packet['input_descriptor']['original_micro_config_UNCHANGED']}, centres)
    native = supplier.stationary.full.supplier.quantum.build_native(mean, 4)
    g = [supplier.enclosure(Fraction(root['parameters'][key]['lower']),
                            Fraction(root['parameters'][key]['upper'])) for key in ('g0','gs')]
    g = [g[0],g[1],g[1],g[1]]
    geom = supplier.stationary.full.normal_geometry(native, g)
    data = {'geom':geom,'native':native,'g':g,
            'frozen_slow_lower':Fraction(full_packet['actual_full45_coupled_quantum_consumer']['certified_eigenvalue_lower_divided_Lambda']['lower'])}
    geom, native = data['geom'], data['native']
    T = geom['M'] / np.sqrt(8)
    O = supplier.orbit(native['frame'], T)
    H = O.T @ O / 8
    Hi = np.linalg.inv(H)
    dO = np.asarray([supplier.orbit(t, T) for t in geom['deltas'] / 8])
    dH = np.asarray([(od.T @ O + O.T @ od) / 8 for od in dO])
    rng = np.random.default_rng(222576045)
    direction = rng.normal(size=45)
    direction /= np.linalg.norm(direction)
    Od = np.einsum('i,iab->ab', direction, dO)
    H1 = (Od.T @ O + O.T @ Od) / 8
    H2 = Od.T @ Od / 4
    records: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    X0: Any = None
    gram = np.zeros((45, 45))
    full_degrees = 0
    joint_trC = joint_trA = joint_trA2 = 0.
    maximum = frozen = scalar = source_score = jet2_defect = 0.
    action = target = double_action = 0.
    sum_spectral_norm = mp.mpf(0)
    root_slow_lower = upper({'upper':str(data['frozen_slow_lower']/2)})
    for row in matter['exact_ALLq_operator_bounds']:
        sum_spectral_norm += row['load'] * upper(row['sea90_operator_upper'])
    perturbation_records = []
    for row in matter['actual_ALL64_q']:
        q = tuple(row['q_integer'])
        partner = tuple((-n) % 4 for n in q)
        spectral = row['actual_static_spectral_supplier']
        fast = supplier.decode(spectral['K_Lambda5over4'])
        # Heavy/unit matrices remain exact in spectral terms, not float-underflowed.
        geometry = supplier.build_geometry(native, geom, q)
        q2 = geometry['qhat2']
        d = 1 - np.exp(2j * np.pi * np.asarray(q) / 4)
        dd = np.outer(d, d.conj())
        G = geometry['Gdiag'][45:, 45:]
        K = fast[45:, 45:]
        wave = supplier.observed_matrix_wave(G, K)
        A, C = wave['A'], wave['covariance']
        Gs = -np.asarray([np.kron(dd, Hi @ dh @ Hi) for dh in dH])
        Ks = np.asarray([.2 * np.kron(np.eye(3), dh) for dh in dH])
        Xi = (Ks - A @ Gs @ A) / 2
        gw, gu = np.linalg.eigh(G)
        rootG = (gu * np.sqrt(gw)) @ gu.conj().T
        invG = (gu / np.sqrt(gw)) @ gu.conj().T
        w, U = np.linalg.eigh(rootG @ K @ rootG)
        omega = np.sqrt(w)
        forced = (U.conj().T @ rootG) @ Xi @ (rootG @ U)
        X = (invG @ U) @ (forced / (omega[:, None] + omega[None, :])) @ (U.conj().T @ invG)
        if not any(q):
            X0 = X
        residual = A @ G @ X + X @ G @ A - Xi
        defect = float(np.max(abs(residual)))
        maximum = max(maximum, defect)
        frozen_xi = Ks / 2
        frozen = max(frozen, float(np.linalg.norm(Xi - frozen_xi)))
        # A common wrong scalar division drops noncommuting ordering.
        wrong = Xi / (2 * float(np.mean(omega)))
        scalar = max(scalar, float(np.linalg.norm(A @ G @ wrong + wrong @ G @ A - Xi)))
        G1 = -np.kron(dd, Hi @ H1 @ Hi)
        G2 = np.kron(dd, 2 * Hi @ H1 @ Hi @ H1 @ Hi - Hi @ H2 @ Hi)
        K1 = .2 * np.kron(np.eye(3), H1)
        K2 = .2 * np.kron(np.eye(3), H2)
        A1, A2 = solve_jet(A, G, K1, G1, K2, G2)
        a2rhs = K2 - (2*A1@G@A1 + 2*A1@G1@A + 2*A@G1@A1 + A@G2@A)
        jet2_defect = max(jet2_defect, float(np.max(abs(A@G@A2+A2@G@A-a2rhs))))
        source_score += .5 * float(np.trace(C @ A1 @ C @ A1).real)
        Xdir = np.einsum('i,iab->ab', direction, X)
        z = rng.normal(size=45) + 1j * rng.normal(size=45)
        direct = float((z.conj() @ (A@G@Xdir+Xdir@G@A) @ z-np.trace(G@Xdir)).real)
        desired = float((z.conj() @ np.einsum('i,iab->ab', direction, Xi) @ z-np.trace(C @ np.einsum('i,iab->ab', direction, Xi))).real)
        records.append({'q': list(q), 'partner': list(partner), 'all45_coefficient_residual': defect,
                        'action_defect': abs(direct-desired), 'dG_norm': float(np.linalg.norm(Gs)),
                        'coefficient_tensor_shape':[45,45,45],
                        'coefficient_tensor_float_sha256':digest(X.tobytes()),
                        'exact_all45_coefficient_representation':{
                            'qhat2':int(round(q2)), 'longitudinal_projector':supplier.encode(dd/q2 if q2 else np.zeros((3,3))),
                            'formula':'X_i=P_L tensor XL_i+(I-P_L) tensor XT_i; all entries in independently_reconstructed_exact_coefficients. q0 P_L=0. No selected-direction replacement.'}})
        # Full physical pencil is retained as exact TERMS, not float underflow.
        if q2:
            baseline_lower = min(mp.mpf(str(.025*q2)), mp.mpf(str(.2*np.linalg.eigvalsh(H)[0])))
            eps = sum_spectral_norm * mp.power(2, -2048) / baseline_lower
        else:
            # q0 slow baseline Lambda*Kheavy_PP; fast colour Lambda54*Kfast_CC.
            # Congruence relative bounds include PC terms with geometric-mean scaling.
            cmin = mp.mpf(str(.2*np.linalg.eigvalsh(H)[0]))
            eps = (2*sum_spectral_norm/mp.sqrt(root_slow_lower*cmin)*mp.power(2,-1024)
                   + sum_spectral_norm/cmin*mp.power(2,-2048)
                   + sum_spectral_norm/root_slow_lower*mp.power(2,-8192))
        if eps >= mp.mpf('.25'):
            raise ArithmeticError('Full physical matrix perturbation not controlled')
        perturbation_records.append({'q': list(q), 'relative_form_error_upper': mp.nstr(eps, 70),
                                     'physical_stiffness_terms': spectral,
                                     'conditional_on': 'exact_ALLq sea norm certificates and q0 slow lower; not floating eigvalsh alone'})
        if q > partner:
            continue
        selfinverse = q == partner
        xr = np.real(X) if selfinverse else np.asarray([realify(x) for x in X])
        cr = np.real(C) if selfinverse else realify(C)
        ar = np.real(A) if selfinverse else realify(A)
        factor = 1 if selfinverse else 2
        gram += 2*np.einsum('iab,jba->ij', xr@cr, xr@cr)
        joint_trC += float(np.trace(cr))
        joint_trA += float(np.trace(ar))
        joint_trA2 += float(np.trace(ar@ar))
        # Retain all fastPhi degrees even though Hermite X has zero Phi block.
        if q2:
            phi = supplier.observed_matrix_wave(8*np.eye(45), fast[:45, :45])
            pc, pa = phi['covariance'], phi['A']
            joint_trC += factor*real_trace(pc)
            joint_trA += factor*real_trace(pa)
            joint_trA2 += factor*real_trace(pa@pa)
        full_degrees += 90*factor
        blocks.append({'q': list(q), 'partner': list(partner), 'real_dimension': 90*factor,
                       'colour_dimension': 45*factor, 'count_once': True})
        zr = rng.normal(size=cr.shape[0])
        xdirr = np.einsum('i,iab->ab', direction, xr)
        gr = G.real if selfinverse else realify(G)
        adirect = float(zr@(ar@gr@xdirr+xdirr@gr@ar)@zr-np.trace(gr@xdirr))
        action += adirect
        xir = np.einsum('i,iab->ab', direction, Xi.real if selfinverse else np.asarray([realify(x) for x in Xi]))
        target += float(zr@xir@zr-np.trace(cr@xir))
        double_action += factor*adirect
    slowK = supplier.decode(matter['actual_ALL64_q'][0]['actual_static_spectral_supplier']['K_Lambda'])[:45,:45]
    slow = supplier.observed_matrix_wave(8*np.eye(45), slowK)
    S = np.real(slow['covariance'])
    p2scaled = float(np.trace(S @ gram))
    pphysical = mp.sqrt(mp.mpf(str(p2scaled))) * mp.power(2,-3072)
    gram_eigen_min = float(np.linalg.eigvalsh(gram)[0])
    # Analytic Wick-weighted Hessian at full physical scales, including all 45 slow modes.
    trC = mp.mpf(str(joint_trC))*mp.power(2,-7168)+mp.mpf(str(np.trace(S)))*mp.power(2,-6144)
    trA = mp.mpf(str(joint_trA))*mp.power(2,7168)+mp.mpf(str(np.trace(slow['A']).real))*mp.power(2,6144)
    trA2 = mp.mpf(str(joint_trA2))*mp.power(2,14336)+mp.mpf(str(np.trace(slow['A']@slow['A']).real))*mp.power(2,12288)
    # Exact weighted moment formulas with B=full/fast coordinate projector, D=A^2.
    # C A^2 C=I/4 and C A^2 C A^2 C=A/8.
    unweighted_h2 = trA**2/4 + trA2/2
    fast_trC = mp.mpf(str(joint_trC))*mp.power(2,-7168)
    fast_h2 = fast_trC*unweighted_h2 + mp.mpf(5715)*trA/2
    full_h2 = trC*unweighted_h2 + mp.mpf(5760)*trA/2
    # Adverse control: source Gaussian normalization derivative must be nonzero.
    source_derivative_norm = float(np.sqrt(source_score))
    double_defect = abs(double_action-action)
    if full_degrees != 5760 or len(blocks) != 36 or len(records) != 64:
        raise ArithmeticError('Incomplete physical Fourier field')
    if maximum > 2e-10 or jet2_defect > 2e-10 or abs(action-target)>1e-8:
        raise ArithmeticError('Actual matrix/Hermite action identity failed')
    if min(frozen, scalar, double_defect, source_derivative_norm) <= 1e-8 or p2scaled <= 0:
        raise ArithmeticError('Adverse control did not discriminate')
    largest_eps = max(mp.mpf(p['relative_form_error_upper']) for p in perturbation_records)
    covariance_ratio = 1/mp.sqrt(1-largest_eps)
    # Stable small differences: no subtraction of 1 at 2^-1024.
    delta = largest_eps/(1+mp.sqrt(1-largest_eps))
    ratio_minus_one = largest_eps/(mp.sqrt(1-largest_eps)*(1+mp.sqrt(1-largest_eps)))
    apmax = mp.mpf(str(np.linalg.eigvalsh(slow['A']).max()))*mp.power(2,6144)
    colour0 = supplier.observed_matrix_wave(.3*np.eye(45),.2*np.kron(np.eye(3),H))
    acmin = mp.mpf(str(np.linalg.eigvalsh(colour0['A']).min()))*mp.power(2,7168)
    coarse_Mbound = delta/(1-delta)*mp.sqrt(apmax/acmin)
    # q0 Gp is scalar and Gc=.3 I, so the mixed Sylvester map is
    # selfadjoint positive. Its inverse <=1/(chi*.3*lambda_min(A_CC)).
    # K_PC has ONLY heavy/unit sea terms, Frobenius <=sqrt45 times op bound.
    KPCbound = mp.sqrt(45)*sum_spectral_norm*(mp.power(2,8192)+1)
    Mbound = KPCbound/(mp.power(2,-4096)*mp.mpf('.3')*(acmin*(1-delta))**2)
    covariance_inflation = covariance_ratio**3
    # Use S<=r*S0 in the weighted Gram metric, not a scalar-S replacement.
    pbound = pphysical*mp.sqrt(covariance_inflation+ratio_minus_one**2*2880*covariance_ratio/2)
    outward = outward_bounds(data, matter)
    # Nonzero first mixed coefficient of the ORIGINAL full q0 functional:
    # A_PC=2^5120 Y + higher orders, .3 Y Acolour=Kheavy_PC.
    # M=2^-2048 Acolour^-1 Y^T. Physical slow A adds relative 2^-1024;
    # its terms and all higher orders are retained by the rigorous mixed bound.
    qzero = next(row for row in matter['actual_ALL64_q'] if not any(row['q_integer']))
    qzero_heavy = supplier.decode(qzero['actual_static_spectral_supplier']['K_Lambda'])
    mixedY = solve_sylvester(np.zeros((45,45)), .3*colour0['A'], qzero_heavy[:45,45:])
    Mcoefficient = np.linalg.solve(colour0['A'], mixedY.conj().T)
    mixed_observation = float(np.linalg.norm(Mcoefficient))
    sprobe,zprobe = rng.normal(size=45),rng.normal(size=45)
    Xs = np.einsum('i,iab->ab',sprobe,X0)
    omitted_m_source_derivative = float(abs(2*zprobe@(Xs@(Mcoefficient@direction))).real)
    if mixed_observation<=1e-8 or omitted_m_source_derivative<=1e-8:
        raise ArithmeticError('Actual fullmatrix source derivative control blind')
    output = {
        'status': 'DERIVED_GAUSSIAN_HERMITE_ACTION_CONDITIONAL_PENCIL_CERTIFICATE_NOT_SOURCE_ADMISSION',
        'dimensions': {'normal':5760,'slow':45,'fast':5715,'colour':2880,'physical_q':64,'canonical_real_groups':36,'coefficients_per_q':45},
        'input_capsule': {path:{'sha256':digest(raw),'bytes':len(raw)} for path,raw in inputs.items()},
        'exact_original_wave_ABI': quantum['actual_full_field_wave_ABI'],
        'exact_all_original_algebraic_coefficients': quantum['actual_exact_algebraic_fast_coefficients'],
        'matrix_jet_derivation': matrix_proof(), 'Gaussian_Hermite_derivation': gaussian_proof(),
        'actual_all64q_all45_coefficients_and_actions': records,
        'canonical_once_blocks': blocks, 'full_physical_pencil_terms_and_comparison': perturbation_records,
        'outward_derived_certificate':outward,
        'fullmatrix_mixed_coefficient':{'physical_power_of_two':-2048,'coefficient':supplier.encode(Mcoefficient),
                                       'norm_observation':mixed_observation,
                                       'status':'Actual full90 q0 first mixed coefficient; NOT a float substitution for the tiny physical terms. Exact full M bound proved separately.'},
        'derived_conditional_bounds': {
            'relative_form_error_upper':mp.nstr(largest_eps,70),'covariance_inflation_upper':mp.nstr(covariance_ratio,70),
            'fullmatrix_M_operator_upper':mp.nstr(Mbound,70),'full_corrector_L2_upper':mp.nstr(pbound,70),
            'leading_exact_Wick_p2_scaled_observation':p2scaled,'leading_gram_eigen_min_observation':gram_eigen_min,
            'leading_physical_trace_covariance':mp.nstr(trC,70),'leading_physical_trace_A':mp.nstr(trA,70),
            'leading_physical_trace_A_squared':mp.nstr(trA2,70),
            'leading_unweighted_Hessian_norm':mp.nstr(mp.sqrt(unweighted_h2),70),
            'leading_rfast_Hessian_norm':mp.nstr(mp.sqrt(fast_h2),70),'leading_rALL_Hessian_norm':mp.nstr(mp.sqrt(full_h2),70),
            'numeric_status':'Displayed coefficient evaluations are finite observations, NOT outward certificates. Inequalities hold algebraically with exact root-box/Clifford/norm constants in formulas. The physical tiny terms are not discarded: all terms and relative form bounds retained.'},
        'controls': {'correct_full_noncommuting_Sylvester_max_defect':maximum,'second_matrix_jet_identity_max_defect':jet2_defect,
                     'canonical_product_rule_defect':abs(action-target),'wrong_frozen_dG_forcing_norm':frozen,
                     'wrong_scalar_noncommuting_action_norm':scalar,'wrong_qpair_doublecount_action_defect':double_defect,
                     'wrong_omitted_fullmatrix_M_source_derivative_coefficient':omitted_m_source_derivative,
                     'wrong_omitted_normalized_Gaussian_source_derivative_norm':source_derivative_norm,
                     'source_derivative_scope':'Actual all64q/full45 colour covariance score of normalized full joint supplier; CAR W/Etilde derivatives separately retained symbolically, no new Kato claim.'},
        'replaced_premises':['Noncommuting positive matrix functional and first/second jet identities/coercivity',
                            'Full joint Wick/cubic corrector L2 identity with nonzero fullmatrix conditional M',
                            'Degree3 hypercontractivity/Dirichlet and Gaussian weighted tensor Holder bounds',
                            'All64q Hermite kinetic product rule with dG and canonical real pairing',
                            'Gaussian/source Leibniz jets, including normalization and corrector/source derivatives'],
        'still_conditional':[
            'Existence/uniqueness/admission of the implicit global root is NOT established; root box and exact Clifford frame remain inputs.',
            'Static all65 contact-minus-bubble norm/lower-gap certificates remain inputs; numeric PH displays do not prove their global validity.',
            'Global occupied complex Riesz/Kato/matter K1/K2/K3 bounds are not supplied by the Gaussian identities; actual fullmatter matrix jets remain conditional on those bounds.',
            'Parent chart/Haar/polar/Kato derivative controls are separate; no rerun or automatic complex extension.',
            'Whole primitive nonlinear/bump/local-metric residual arithmetic is NOT certified by this result. Gaussian/Hermite/covariance pieces have new outward bounds; replacing archived budgets in the whole residual remains parent integration.',
            'The actual archived source residual must be recomposed with these outward formulas and independent complex matter/source jets. No residual/source admission follows from finite coefficient residuals.',
            'D(H^2) closure for the actual global operator, original source, U1/Theta/Pi/Coulomb, global ground, photon pole and common cone remain OPEN.'],
        'source_at_import':{name:{'sha256':digest(raw),'bytes_utf8':raw.decode()} for name,raw in {**captured,Path(__file__).name:SELF_BYTES}.items()},
        'input_unchanged':{path:Path(path).read_bytes()==raw for path,raw in inputs.items()},
        'source_unchanged':{name:(ROOT/name).read_bytes()==raw for name,raw in {**captured,Path(__file__).name:SELF_BYTES}.items()},
        'runtime':{'executable':sys.executable,'python':sys.version,'numpy':np.__version__,'mpmath':mp.__version__,'arithmetic_digits':mp.mp.dps},
        'elapsed_seconds':time.monotonic()-started}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quantum',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.quantum, args.source)
    if not all(result['input_unchanged'].values()) or not all(result['source_unchanged'].values()):
        raise ArithmeticError('Source/input epoch changed')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'dimensions':result['dimensions'],'controls':result['controls'],
                      'derived_conditional_bounds':result['derived_conditional_bounds'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
