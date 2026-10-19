"""Actual currentN4 full-Phi fiber existence and a new rational allnative trial.

Proof statements, outward moment prices and finite adverse controls are separate.
No root solve, field benchmark, normalizer choice or physical vacuum/pole claim.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
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
import sympy as sy
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import induced_phi_nonuniform_fermionic_source_unitary as preparation
import induced_phi_nonuniform_gauss_quantum as gauss

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_stationary_full_phi_fiber'
sys.set_int_max_str_digits(0)
PINNED = {
 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json':'4691ffaa9cd81902942cbc228fc47fafff14518be6b5df2e84d08c29c7da4756',
 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json':'5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354',
 'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json':'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6',
 'joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json':'2cc91cbe6b8b64f6d270ff04a9b946b22bd3a9a52ff9627e4bf15325f0c0adf4',
 'joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json':'2a1bbb46852707c82e477f28b5497e303b5ca66453db76d0655379f35cfb9a97',
 'induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json':'f6a6b6d79496e4e24d0a5df4762a9098b0db24debe16c2f82c32572696592d2f',
}
DIRECT = (
 'bosonic_original_law_join.py','induced_phi_backreaction.py',
 'induced_phi_uniform_relative_colour_quantum.py',
 'induced_phi_global_normal_gauss_chart_certificate.py',
 'induced_phi_nonuniform_local_gauss.py',
 'induced_phi_global_normal_source_reprice.py',
 'induced_phi_global_occupied_kato_derivative_control.py',
 'joined_phi_u1_actual_compact_source_action_certificate.py',
 'joined_phi_u1_full_current_graph_transfer.py',
 'induced_phi_compact_source_global_domain_control.py',
)
PROOF = {
 'law_owner_and_coordinates': 'This proof uses the current originalFullForm exact scientific alignment1/5 and plaquette7/10 ABI, not the old LAW05 twice-colour branch or the bare-EM original branch. Lambda=2^8192, chi=Lambda^(-1/2)=2^-4096, Lambda^(5/4)=2^10240; these finite hierarchy multipliers are retained exactly. Clifford Cartesian B_A have Tr(B_A B_B)=4 delta; SU4 T_A=B_A/sqrt8 have Tr(T_A T_B)=delta/2. Phi=sum y_A B_A=sum x_A T_A gives y=x/sqrt8 and p_y=sqrt8 p_x, hence chi/2 sum p_y^2=4chi sum p_x^2. This derives global Cart-x cometric8chi; it is not promotion of a local rescaled normal symbol. HS norm squared is |x|^2/2. The uniform slow injection x=s/8 has norm preserved after summing64 sites; no Lambda coordinate rescaling is made. Colour left generators use T_A and current originalFullForm kinetic3chi/20 sum E_A^2. The old LAW05 .3chi E^2 is explicitly excluded by uniform_relative_colour_quantum.py original colour normalization.',
 'operator': 'On R3840_x times SU4^192 times Theta quotient, tensor fixed species CAR and Ref, K=4chi sum p_x^2+(3chi/20)sum E_SU4^2+64 sum Pi_Theta^2. V4=(Lambda/a)sum||Gram_x||F^2, Gram_mn=Tr(Phi_m Phi_n)/4; Va=Lambda^(5/4)/40 sum||Phi_x-Uxy Phi_y Uxy^dag||HS^2; Vm=(7/10)Lambda^(5/4)sum_faces(4-ReTr orderedUface); HC=64rho_allcharged Lplus rho_allcharged. Matter is dGamma(-Lambda hneutral)+dGamma(hactive)+sum63 dGamma(haux), each actual onsite and ordered one/two/AP endpoint hxy=(Fx Uxy+Uxy Fy)/2 times its unit U1 character. Ref and fixed background charges are retained. No EM curl/B2 and no static filled-sea replacement of matter. Current coefficients originate in primitive_profiles/models and the exact current source ABI; binary .2/.7 alternative is not silently adopted.',
 'period_lattice': 'D is literal integer64x192 incidence; a connected tree gives saturated fundamental cycle C192x129 with DC=0 and identity chord minor. With native orthonormal Q, A=C^T Q is invertible and Theta=A^T t, t in[0,2pi)^129. Native edge incidence k has character exp(i (C^T k).t) with integer C^T k, including OPEN paths multiplying CAR; periodicity is not restricted to static Phi or U=I. Bloch beta boundary is exp(2pi i beta.m). On periodic representatives Pi=A^-1(-i grad_t+beta), with elliptic electric metric A^-T A^-1 and lower eigenvalue >=1/Tr(C^T C). beta is configuration Bloch quasimomentum, not spatial photon q.',
 'closed_semibounded_fiber': 'For each beta use smooth compact x, smooth fullHaar and smooth Bloch sections with finite CAR, projected to the compact SU4 Gauss sector after U1 longitudinal Gauss solution. Complete the positive derivative form and intersect its domain with weighted domains of V4,Va,Vm,HC. All positive potentials are smooth nonnegative; nonnegative potential addition gives a closed form. Native Schur bounds give ||M||<=B X with X=||Phi||HS and B=128Lambda Kneutral+129Kactive+8064Kaux; K=|onsite|+2sum(|c0|+|ci|), source count is a bound also for background128. Gram PSD/SOS and64-site Cauchy give V4>=Lambda X^4/64. Young absorbing half gives H_beta+C>=K+Lambda X^4/128. Rational C=Lambda C0, C0^3>=27*128*B0^4/256 with B<=Lambda B0. KLMN produces the semibounded closed matching fiber form and its Friedrichs operator, not an essential-selfadjointness claim. If an extra field-independent Ref scalar is adopted, its explicit absolute value must be added; the present selected Ref has none. Form core density uses cutoffs, manifold partitions/mollifiers and compact Gauss projection.',
 'nonempty_sector': 'Nonzero canonical sectors are constructed, not inferred merely from Nmod4. Fill all4 colours at32 selected sites in EACH of65 species (neutral128, active128, auxiliary63*128) and fullRef4 at0. Each determinant is a local SU4 singlet. Choose site0 outside those32 sites. Multiply by invariant compact smooth radial Phi scalar, constant fullHaar links and normalized Bloch exp(i p_beta.Theta). Longitudinal U1 dressing solves D ell=rho because rho has total0, leaving Coulomb in the reduced Hamiltonian; no unreduced constant gauge volume is normalized. This is a nonzero background physical section with N8324. Apply sum_alpha c_active,0,alpha^dag R_alpha/2: active129,Ref3, totalN8324 and totalU1rho unchanged. The explicit eight-mode CAR state has norm1 here and all traceless local SU4 charges vanish. This determinant existence witness is NOT the actual onsite sea or an adopted source normalization.',
 'compact_resolvent': 'At every finite positive chi, H+C controls full x H1, Haar H1, torus H1, and X^4. On bounded x regions Rellich holds on compact smooth SU4^192 times torus with finite Fock fiber. The exterior L2 mass is at most const/R^4 from quartic. A bounded form-norm sequence therefore has an L2-convergent subsequence by local Rellich plus the tail estimate. Compact Gauss-sector restriction preserves compact embedding. The nonempty sector has compact resolvent and a lowest eigenvalue E0(beta) attained by a normalized full-Phi physical eigenvector. The tiny kinetic coefficient is positive, not set0. This existence conclusion is independent of the root/SPD/gap/chart suppliers and does not numerically characterize that eigenvector.',
 'beta_continuity_minimum': 'Gauge all fibers to periodic representatives by exp(i beta.t), giving one common form domain and physical Gauss sector. The only beta dependence is64||A^-1(p+beta)psi||^2. Cauchy gives locally small form difference controlled by delta_beta times K+constant||psi||^2; K is controlled by H+C uniformly on a compact beta chart. Min-max gives continuous ground energy (also ordered eigenvalues), or norm-resolvent form continuity does. Integer beta shifts are unitary equivalent via integer lattice characters. Compact beta torus therefore has some global minimizer beta*, WITHOUT identifying it as0. Original real-cover bottom is min_beta E0(beta). A normalized real-cover L2 ground needs a positive-measure minimizing band set and an integrable measurable section; neither absence nor existence of a flat minimizing band is asserted.',
 'stationary_functional': 'At a minimizing beta* choose a normalized lowest physical eigenvector Omega. Specify the actual C* algebra generated by bounded gauge-invariant fullPhi/orderedSU4/periodicTheta matrix functions, gauge-dressed sector-preserving canonical CAR characters, compatible electric translations and bounded CONTINUOUS functional calculus/resolvents of physical field operators; close it in norm and under physical time evolution. Its bounded fiber fields and adjoints are strongly continuous in beta on common periodic representatives: multiplication/CAR generators are beta-independent, electric translations are continuous scalar shifts, continuous functional calculus follows strong-resolvent continuity, and finite-time propagators are strongly continuous. Norm closure preserves this property. Original-cover operator equality therefore implies equality in EVERY fiber, not merely almost every fiber; fiber evaluation is a contractive star representation because strong continuity makes the operator norm lower semicontinuous. Thus omega(A)=<Omega,A_beta* Omega> is normalized positive. Physical H_beta* implements dynamics on the cyclic subspace, giving stationarity and nonnegative ground generator H_beta*-E0(beta*). This is a chosen-sector periodic physical algebra, not arbitrary Borel functions of beta-dependent operators. Unbounded currents/contact remain affiliated forms. No chart restriction, static sea, ordinary real-cover L2 vacuum, rawTheta or nonlattice multiplication-Weyl ground extension is asserted. No energy ordering between different canonical sectors is proved.',
 'original_cover_ground_functional_limit': 'The functional on that SAME algebra is also an original-Hilbert limit, not a pointwise Borel substitution. In a local beta chart use the common periodic Gauss Hilbert space and Omega at beta*. Pi_beta=A^-1(p+beta). First-derivative Pi is infinitesimally operator-bounded relative to H_beta* by H+C>=64Pi^2, so beta shifts have the common operator domain. For d=A^-1(beta-beta*), (H_beta-E*)Omega=128 d.Pi_beta* Omega+64|d|^2 Omega. Choose normalized L2 envelopes eta_epsilon supported within |beta-beta*|<=epsilon and set Psi_epsilon(beta)=eta_epsilon(beta)Omega. These are normalized original-cover states under inverse Floquet transform, with residual <=128||A^-1||epsilon||Pi_beta*Omega||+64||A^-1||^2epsilon^2 ->0. Strong-star beta continuity of each chosen bounded physical A gives convergence of its packet expectation to the fiber functional, including physical time correlations; the concrete fiber Hamiltonian supplies continuous positive-energy dynamics. Every globally null operator stays null in each packet and its limit. For a beta-discontinuous arbitrary Borel effect, direct fiber evaluation is NOT authorized: the free real-cover 1_{0}(Pi) is globally0 but its beta0 fiber projection is nonzero. Its packet expectation is0, not1. Bounded continuous 1/(1+Pi^2) has packet limit1 at beta0, providing the positive control. No canonical extension to all unsmeared Borel observables, rawTheta moments, full-CCR regularity or an attained normalizable real-cover ground follows.',
 'actual_T_fiber_audit': 'T v=Z^-1/2 b6(r)(1-P(r))f_A(r) J(r)^-1/2 Gamma(Ulocal) [neutral upper-h rank128 Slater transported by Wneutral(r) tensor Gamma(Wcharged(r))v(Theta)]. r contains slow45+zeroMeanPhi2835+colour2880; normalized full localHaar960 is integrated with exactly one commonZ4. Z is the SAME implicit scalar norm, not set equal to its proof bound; nonzero follows Gaussian/normalizer suppliers. W solves actual occupied radial d_tW=[DP(tr)[r],P(tr)]W, not Qnormal polar transport, and is independent of Theta. Anti-Hermitian K gives unitarity, exterior lifts preserve inner products and neutral Slater has norm1, so T* T=I for arbitrary fixed-charge body vectors. J cancels the geometric density; scalar integral isZ. LocalHaar carriers make Gauss equivariance; commonZ4 acts trivially on totalN8324. Translation by2pi Q^TCm commutes with every T factor, so T acts in EVERY beta fiber. On trigonometric body core, b6 zero traces0..5, smoothHaar/Kato and compactPhi put Tv in D(H_beta^2). Literal HE commutesT; no Theta derivative ofW is omitted. The remainder is an order0 Theta-periodic operator-valued multiplier, from scalar and Phi/Haar differential actions on packet, charged endpoint variation, twice chargedsea subtraction and HC commutator. Its allTheta pointwise norm<=epsilon implies the SAME bound on each compact cell/fiber, not only almost-every-beta inferred from real-cover Fubini. Graph closure extends T:D(Hbody_beta)->D(H_beta). It is a bounded body-to-fullfield map, NEVER a bounded remainder/current on the fullfield Hilbert space.',
 'uniform_R_supplier_boundary': 'The consumer independently reconstructs epsilon from exact rational upper inputs and the literal action source formula: prepared Gaussian trace/corrector radius; primitive Taylor/Hermite/root-force/b6/Haar/Kato budgets; full fixedN crossparticle C1^2 andNoether8324; arbitrary charged h andHC prices. The exact underlying root, root-force cancellation, full5760 SPD/corrector, staticoccupied gaps, injective3R chart and scalar-action suppliers remain CONDITIONAL. Their numbers are not re-proved by controls. No claim that a receipt string alone constructs an executable normalized5760 vacuum. Fresh primitive nonuniformPhi/SU4/U1/Gauss controls and source AST audit test the relevant structural boundaries. No failed antecedent may be backfilled with assumed beta0 or substituted normalizer.',
 'new_trial': 'At beta0 in background active128/aux8064/Ref4 use ref=onsite negativeGamma0 occupancy2 at every site/copy, constantTheta. It is an exact onsite+HE+HC eigenvector with Eref=-128sqrt(g0)(onsiteActive+63onsiteAux) and rho=0. r=(Hbody-Eref)ref contains ALL native one/two/AP paths; for this particular primitive ci_two=0, two-link paths annihilate ref but remain in <r,H r> through1+1-2 interference. Distinct PH electron/hole/INTEGER-character labels are orthogonal. sigma2=||r||^2, mu3=<r,(Hbody-Eref)r>. Clifford traces Pminus Gamma_i Pplus Gamma_j=2delta_ij derive weights2ci^2 gs without float eigenvectors. Each nonzero residual one-link has HE=64(1-deltaLplusdelta) and HC=64deltaLplusdelta, EXACT common cancellation; onsite cost2onsite sqrtg0. Electron and hole CAR terms are both included in the cubic hopping coefficient, derived by literal native character lookup and Clifford projection, not a float eigenvalue cap. v_gamma=(ref-gamma r)/sqrt(1+gamma^2sigma2), gamma=1/1024 EXACT. DeltaE=(gamma^2mu3-2gamma sigma2)/(1+gamma^2sigma2). Outward intervals in the exact saved root enclosures price all moments. T v_gamma is a new same-law fullPhi variational vector; |<Tv,H Tv>-(Ephi+<v,Hbody v>)|<=epsilon. CommonEphi cancels in comparing Tv_gamma toTref; PAY2epsilon, one at each endpoint. A negative upper DeltaE+2epsilon proves fullPhi reference lowering, not that ref/Ttrial is the ground. E0(0)<=Ephi+Etrial+epsilon and min_betaE0<=E0(0); global lower is-C. Ephi-relative lower-C-|Ephi| is deliberately coarse, NOT a ground gap/pole characterization.',
 'open_goal': 'FullPhi ground characterization, beta* identity, real-cover normalizable ground/flat-band question, stationary expectation values, same-ground photon pole/residue/dispersion and QED campaign remain OPEN. Exact proof, conditional supplier arithmetic, finite controls and historical numerical leads are separately labelled.'
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def upper(record: dict[str, Any]) -> F:
    for key in ('upper','exact_upper','exact'):
        if key in record:return F(record[key])
    raise ValueError('Missing exact rational upper price')


def cycles(d: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Integer tree elimination; no floating inverse defines the lattice."""
    adjacency: dict[int, list[tuple[int,int]]] = defaultdict(list)
    for e in range(192):
        x,y = map(int,np.flatnonzero(d[:,e])); adjacency[x].append((y,e)); adjacency[y].append((x,e))
    parent={0:(-1,-1)}; order=[0]
    for x in order:
        for y,e in adjacency[x]:
            if y not in parent: parent[y]=(x,e); order.append(y)
    assert len(order)==64
    tree={parent[x][1] for x in order[1:]}; chords=[e for e in range(192) if e not in tree]
    c=np.zeros((192,129),dtype=np.int64)
    for j,e in enumerate(chords):
        c[e,j]=1; b=-d[:,e].copy()
        for x in reversed(order[1:]):
            y,f=parent[x]; a=int(b[x])*int(d[x,f]); c[f,j]=a; b[y]-=int(d[y,f])*a; b[x]=0
        assert b[0]==0
    assert not np.any(d@c) and np.array_equal(c[chords],np.eye(129,dtype=int))
    return c,chords


def rational_green(sites: np.ndarray) -> list[list[F]]:
    cos=(1,0,-1,0); lookup={}
    for delta in product(range(4),repeat=3):
        value=F(0)
        for q in product(range(4),repeat=3):
            lam=sum(2-2*cos[j] for j in q)
            if lam:value+=F(cos[sum(a*b for a,b in zip(q,delta))%4],64*lam)
        lookup[delta]=value
    return [[lookup[tuple(map(int,(x-y)%4))] for y in sites] for x in sites]


def exact_moments(data: dict[str,Any], c: np.ndarray, mean: dict[str,Any]) -> dict[str,Any]:
    iv=gauss.iv; ex=gauss.exact; pack=gauss.bounds
    g0,gs=[field.full.enclosure(F(mean['parameters'][k]['lower']),F(mean['parameters'][k]['upper'])) for k in ('g0','gs')]
    root0=iv.sqrt(g0); green=rational_green(data['native']['sites'])
    green_error=float(np.max(abs(np.asarray(green,dtype=float)-data['native']['Lplus'])))
    assert green_error<1e-12
    D=np.rint(data['native']['D']).astype(np.int64)
    L=D@D.T
    for x in range(64):
        assert sum(green[x],F(0))==0
        for y in range(64):
            assert sum((int(L[x,z])*green[z][y] for z in np.flatnonzero(L[x])),F(0))==F(int(x==y))-F(1,64)
    rows=[]; energy=ex(0); sigma=ex(0); mu3=ex(0)
    for profile in data['profiles'][1:]:
        signed=[]; amplitudes={}; outgoing=defaultdict(list); incoming=defaultdict(list)
        for path in field.native_paths(profile,data['native']):
            row=next(r for r in profile['paths'] if r['axis']==path['axis'] and r['length']==path['length'])
            k=np.zeros(192,dtype=np.int64); k[list(path['edges'])]=1; character=c.T@k
            for x,y,ch,kk,c0,ci in ((path['source'],path['target'],character,k,F(row['c0'])*path['seam'],F(row['ci'])*path['seam']), (path['target'],path['source'],-character,-k,F(row['c0'])*path['seam'],-F(row['ci'])*path['seam'])):
                x,y=int(x),int(y); v=[F(0)]*3; v[path['axis']]=ci
                item=(x,y,tuple(map(int,ch)),c0,tuple(v),kk)
                signed.append(item); outgoing[y].append(item); incoming[x].append(item)
                key=(y,x,item[2]); old=amplitudes.get(key,(F(0),)*3)
                amplitudes[key]=tuple(a+b for a,b in zip(old,v))
        nonzero={key:a for key,a in amplitudes.items() if any(a)}
        sigcoef=F(0); cubiccoef=F(0); diagonalcoef=F(0); cancellation=[]; triangles=0
        for (hole,electron,char),a in nonzero.items():
            weight=2*sum((v*v for v in a),F(0)); sigcoef+=weight
            match=next(item for item in signed if item[:3]==(electron,hole,char))
            k=match[-1]; delta=D@k
            rho=green[electron][electron]+green[hole][hole]-2*green[electron][hole]
            assert sum(abs(int(v)) for v in delta)==2
            electric=F(int(k@k))-sum((int(delta[x])*green[x][y]*int(delta[y]) for x in np.flatnonzero(delta) for y in np.flatnonzero(delta)),F(0))
            assert electric+rho==1 and electric>0 and rho>0
            cancellation.append((str(electric),str(rho)))
            diagonalcoef+=weight*64*(electric+rho)
            for x,y,q,c0,_,_ in outgoing[electron]:
                other=nonzero.get((hole,x,tuple(a+b for a,b in zip(char,q))))
                if other:
                    cubiccoef+=2*c0*sum((v*w for v,w in zip(other,a)),F(0)); triangles+=1
            for x,y,q,c0,_,_ in incoming[hole]:
                other=nonzero.get((y,electron,tuple(a+b for a,b in zip(char,q))))
                if other:
                    # Minus hole CAR sign times Pminus Gamma0 Pminus=-Pminus.
                    cubiccoef+=2*c0*sum((v*w for v,w in zip(other,a)),F(0)); triangles+=1
        s=ex(sigcoef)*gs; diag=gs*(ex(diagonalcoef)+ex(2*F(profile['onsite'])*sigcoef)*root0); cubic=ex(cubiccoef)*root0*gs
        load=profile['load']; e=-ex(128*F(profile['onsite'])*load)*root0
        energy+=e; sigma+=load*s; mu3+=load*(diag+cubic)
        rows.append(dict(role=profile['role'],load=load,signed_native_paths=len(signed),nonzero_residual_PH_characters=len(nonzero),exact_sigma2_coefficient_times_gs=str(sigcoef),exact_mu3_diagonal_electric_HC_coefficient_times_gs=str(diagonalcoef),exact_mu3_onsite_coefficient_times_sqrtg0_gs=str(2*F(profile['onsite'])*sigcoef),exact_mu3_hopping_coefficient_times_sqrtg0_gs=str(cubiccoef),surviving_electron_and_hole_character_matches=triangles,exact_HE_HC_pairs=sorted(set(cancellation)),all_path_character_sha256=sha(json.dumps([list(item[:5]) for item in signed],default=str,separators=(',',':')).encode()),sigma2_per_copy=pack(s),diagonal_mu3_per_copy=pack(diag),hopping_mu3_per_copy=pack(cubic),energy_all_copies=pack(e)))
    gamma=F(1,1024); shift=(ex(gamma**2)*mu3-2*ex(gamma)*sigma)/(1+ex(gamma**2)*sigma)
    return dict(arithmetic='110-digit outward mpmath.iv, exact rational primitive binary path coefficients and exact implicit saved root enclosures; NOT float eigenvalue enclosure',gamma_exact=str(gamma),Eref=pack(energy),sigma2=pack(sigma),mu3=pack(mu3),DeltaEbody=pack(shift),Etrialbody=pack(energy+shift),species=rows,exact_Green_Lplus_identity=True,native_float_Green_comparison_control=green_error,residual_definition='Literal ALL native path action, zero two-link ci is derived from primitive not deletion. Full129 electric and full64 HC; all64 charged flavours.',trial_norm_squared=pack(1+ex(gamma**2)*sigma))


def cuberoot_upper(value: F) -> F:
    lo,hi=0,1
    while F(hi)**3<value:hi*=2
    while hi-lo>1:
        mid=(hi+lo)//2
        if F(mid)**3>=value:hi=mid
        else:lo=mid
    return F(hi)


def uniform_price(action:dict[str,Any], take:Any) -> dict[str,Any]:
    inputs={name:take(name,pin) for name,pin in action['input_sha256'].items()}
    get=lambda prefix:next(value for name,value in inputs.items() if name.startswith(prefix))
    iv=gauss.iv; ex=gauss.exact; pack=gauss.bounds
    gb=get('induced_phi_gaussian_action_certificate')['outward_derived_certificate']['bounds']
    local=get('induced_phi_local_gauss_action_certificate')['actual_derived_prices']; reprice=get('induced_phi_global_normal_source_reprice')
    radius=2*iv.sqrt(ex(upper(gb['full_trace_covariance'])))*(1+9*ex(upper(gb['corrector_L2'])))
    constants=local['cases']['support']['operator_constants_exact']; m0,a1=[F(constants[k]) for k in ('cometric','cometric_D1')]
    gradient=ex(upper(local['weighted_compact_scalar_packet']['gradient_upper']))
    c1=F(reprice['corrected_inputs']['C1']['exact_upper']); c2=F(reprice['corrected_inputs']['C2_total_tensor']['exact_upper']); chi=F(2)**-4096
    kato=ex(chi)*(ex(m0*c1)*gradient+ex(5760*a1*c1+76*m0*(c2+c1*c1))/2+ex(m0*2*8324*c1))
    coupling=reprice['new_U1_extension']; hc=F(coupling['Coulomb_operator_norm_upper']['exact_upper']); hcoef=F(coupling['charged_h_coefficient']['exact_upper']); wcoef=F(coupling['charged_W_coefficient']['exact_upper'])
    force=get('induced_phi_complex_matter_force_source_variance')
    components={
     'primitive_nonlinear':ex(upper(get('induced_phi_primitive_nonlinear_action_certificate')['derived_outward_weighted_action']['actual_nonlinear_action_norm_with_proved_normalization_upper2'])),
     'Haar_metric_Noether':ex(upper(local['cases']['support']['geometric_remainder_norm_upper'])),
     'Gaussian_Hermite':ex(upper(get('induced_phi_fixed_gaussian_hermite_action_closure')['outward_action']['action_PREPARED'])),
     'root_force':ex(upper(force['actual_action_input_budgets']['prepared_total_force_action_upper'])),
     'real_Kato_crossparticle':kato,
     'b6_commutator':ex(upper(local['separate_reference_b6_commutator_norm_upper'])),
     'twice_charged_delta_plus_HC':(2*ex(hcoef)+2*ex(hc*wcoef))*radius,
     'saved_mean_shift':ex(upper(force['actual_action_input_budgets']['mean_shift_omitted_coefficients']))*radius,
    }
    epsilon=sum(components.values()); epsupper=gauss.endpoint(epsilon,1)
    assert epsupper<=upper(action['action_input']['epsilon']) and epsupper<F(2)**-400
    return dict(epsilon=pack(epsilon),components={k:pack(v) for k,v in components.items()},literal_formula_reconstructed=True,actual_Z_not_selected=True,normalizer_upper_only=2,prepared_radius=pack(radius),conditional_inputs=action['input_sha256'],scope='CONDITIONAL suppliers; independently assembled bound, not numerical evaluation of implicit5760 normalized wave or a globally bounded fullfield R')


def literal_operator(data:dict[str,Any],profile:dict[str,Any],phi:np.ndarray,links:np.ndarray,theta:np.ndarray) -> np.ndarray:
    h=np.zeros((256,256),complex)
    for x in range(64):h[4*x:4*x+4,4*x:4*x+4]=float(profile['onsite'])*phi[x,0]
    for p in field.native_paths(profile,data['native']):
        x,y=p['source'],p['target']; u=np.eye(4,dtype=complex)
        for e in p['edges']:u=u@links[e]
        fx,fy=np.einsum('m,mij->ij',p['coefficients'],phi[x]),np.einsum('m,mij->ij',p['coefficients'],phi[y])
        phase=1 if profile['scale']=='heavy' else np.exp(1j*sum(theta[e] for e in p['edges']))
        b=p['seam']*(fx@u+u@fy)/2*phase
        h[4*x:4*x+4,4*y:4*y+4]+=b;h[4*y:4*y+4,4*x:4*x+4]+=b.conj().T
    return (-1 if profile['scale']=='heavy' else 1)*h


def nonuniform_controls(data:dict[str,Any],c:np.ndarray) -> dict[str,Any]:
    rng=np.random.default_rng(2221010129); T=data['geom']['M']/np.sqrt(8)
    def unit(shape:tuple[int,...],norm:float)->np.ndarray:
        x=rng.normal(size=shape);return x*norm/np.linalg.norm(x)
    slow=unit((45,),2.**-23);fast=unit((64,45),2.**-22);fast-=fast.mean(axis=0); colour=unit((64,3,15),2.**-22)
    phi,_,links,_=preparation.field_ray(data,1.,slow,fast,colour)
    gauges=np.asarray([expm(1j*np.einsum('a,aij->ij',x,T)) for x in unit((64,15),5.)])
    phig=gauges[:,None]@phi@gauges.conj().swapaxes(-1,-2)[:,None]
    endpoints=[]
    for e in range(192):
        p=next(p for p in data['native']['Pprofiles'][0]['paths'] if p['length']==1 and p['edges'][0]==e)
        endpoints.append((p['source'],p['target']))
    linkg=np.asarray([gauges[x]@u@gauges[y].conj().T for u,(x,y) in zip(links,endpoints)])
    theta=unit((192,),1.3); m=np.zeros(129,dtype=np.int64);m[[0,7,44,128]]=[1,-2,1,3];shift=2*np.pi*(c@m)
    alpha=unit((64,),.7);gaugephase=np.asarray([alpha[x]-alpha[y] for x,y in endpoints])
    bigg=np.zeros((256,256),complex);bigphase=np.zeros_like(bigg)
    for x in range(64):bigg[4*x:4*x+4,4*x:4*x+4]=gauges[x];bigphase[4*x:4*x+4,4*x:4*x+4]=np.eye(4)*np.exp(1j*alpha[x])
    rows=[]
    for p in data['profiles']:
        h=literal_operator(data,p,phi,links,theta); shifted=literal_operator(data,p,phi,links,theta+shift)
        transformed=literal_operator(data,p,phig,linkg,theta)
        gauged=literal_operator(data,p,phi,links,theta+gaugephase)
        direct=literal_operator(data,p,phi,links,np.zeros(192)); owner=field.assemble(p,data['native'],phi,links)
        static=literal_operator(data,p,np.repeat(data['native']['frame'][None],64,axis=0),np.repeat(np.eye(4,dtype=complex)[None],192,axis=0),theta)
        wrong=literal_operator(data,p,phi,links,theta+np.linspace(0.,.9,192))
        u1target=h if p['scale']=='heavy' else bigphase@h@bigphase.conj().T
        errors=dict(periodicity=float(np.linalg.norm(shifted-h)),SU4_Gauss_covariance=float(np.linalg.norm(transformed-bigg@h@bigg.conj().T)),U1_Gauss_covariance=float(np.linalg.norm(gauged-u1target)),literal_owner_match=float(np.linalg.norm(direct-owner)),wrong_nonlattice_shift=float(np.linalg.norm(wrong-h)),wrong_static_surrogate=float(np.linalg.norm(static-h)))
        assert max(errors[k] for k in ('periodicity','SU4_Gauss_covariance','U1_Gauss_covariance','literal_owner_match'))<1e-9
        # Heavy charge0 is retained exactly, not inferred from a Ward display.
        if p['scale']=='charged':assert errors['wrong_nonlattice_shift']>1 and errors['wrong_static_surrogate']>1e-8
        rows.append(dict(role=p['role'],**errors))
    B=data['geom']['M']; gram=np.real(np.einsum('aij,bji->ab',B,B)); tg=np.real(np.einsum('aij,bji->ab',T,T)); S=gram/np.sqrt(8)/4
    assert np.max(abs(gram-4*np.eye(15)))<1e-12 and np.max(abs(tg-np.eye(15)/2))<1e-12
    assert np.max(abs(S.T@S-np.eye(15)/8))<1e-12
    return dict(scope='NEW finite float64 structural controls on actual currentN4 primitive nonuniformPhi/orderedSU4/U1, not root/global inequalities or vacuum certification',seed=2221010129,slow_norm=float(np.linalg.norm(slow)),zero_mean_fast_norm=float(np.linalg.norm(fast)),colour_norm=float(np.linalg.norm(colour)),local_Haar_total_angle_norm=5.,all64_sites_all192_links=True,rows=rows,Cart_y_to_x_metric_factor_exact='1/8',global_Cart_x_kinetic_exact='4*2^-4096 sum p_x^2',physical_colour_kinetic_exact='3*2^-4096/20 sum E^2')


def car_apply(state:dict[int,F],a:int,b:int) -> dict[int,F]:
    out=defaultdict(F)
    for mask,value in state.items():
        if not(mask>>b)&1:continue
        sign=(-1)**((mask&((1<<b)-1)).bit_count()); t=mask^(1<<b)
        if(t>>a)&1:continue
        sign*=(-1)**((t&((1<<a)-1)).bit_count());t|=1<<a;out[t]+=sign*value
    return {mask:value for mask,value in out.items() if value}


def exact_adverse_controls() -> dict[str,Any]:
    ref={sum(1<<i for i in range(4,8)):F(1)}; source=defaultdict(F)
    for i in range(4):
        for mask,value in car_apply(ref,i,4+i).items():source[mask]+=value/2
    assert sum((v*v for v in source.values()),F(0))==1
    for a in range(4):
        for b in range(4):
            result=defaultdict(F)
            for mask,value in car_apply(source,a,b).items():result[mask]+=value
            for mask,value in car_apply(source,a+4,b+4).items():result[mask]+=value
            if a==b:
                for mask,value in source.items():result[mask]-=value
            assert not any(result.values())
    x=sy.Symbol('x',real=True);poly=1+x+x*x
    good=(1-x*x)**6*poly;bad=(1-x*x)**2*poly
    H=lambda f:-sy.diff(f,x,2)+x**4*f
    gj=[sy.diff(good,x,k).subs(x,1) for k in range(6)];gh=[sy.diff(H(good),x,k).subs(x,1) for k in range(4)]
    assert not any(gj) and not any(gh) and H(bad).subs(x,1)!=0
    # One-particle two-mode CAR sector, constant periodic spin mixing, half flux.
    car=sy.Matrix([[0,1],[1,0]]);beta=sy.Symbol('beta',real=True)
    assert car.eigenvals()=={-1:1,1:1}
    e0=F(1,4)-1;ehalf=F(-1);assert ehalf<e0
    # Exact Clifford weights without chosen float eigenvectors.
    gamma0=sy.diag(-1,-1,1,1); pm=(sy.eye(4)-gamma0)/2;pp=(sy.eye(4)+gamma0)/2
    pauli=[sy.Matrix([[0,1],[1,0]]),sy.Matrix([[0,-sy.I],[sy.I,0]]),sy.diag(1,-1)]
    gammas=[sy.BlockMatrix([[sy.zeros(2),v],[v,sy.zeros(2)]]).as_explicit() for v in pauli]
    trace=[[sy.trace(pm*a*pp*b) for b in gammas] for a in gammas]
    assert trace==[[2,0,0],[0,2,0],[0,0,2]]
    # A global null Borel effect must not become a fiber-point observable.
    b=sy.Symbol('b',real=True);width=sy.Rational(1,8)
    point=sy.Piecewise((1,sy.Eq(b,0)),(0,True))
    global_point=sy.integrate(point,(b,-width,width))/(2*width)
    fiber_point=sy.Integer(1)
    continuous=sy.integrate(1/(1+b*b),(b,-width,width))/(2*width)
    z=sy.Symbol('z',positive=True)
    continuous_limit=sy.limit(sy.atan(z)/z,z,0,dir='+')
    assert global_point==0 and fiber_point==1 and continuous_limit==1
    borel_boundary=dict(global_zero_projection_packet_norm_squared=str(global_point),
                        beta0_fiber_zero_projection_norm_squared=str(fiber_point),
                        continuous_packet_positive_control=str(continuous),
                        continuous_packet_limit=str(continuous_limit),
                        normalized_free_packet_width_exact=str(width),
                        free_packet_energy_exact=str(width**2/3),
                        free_packet_H_residual_squared_exact=str(width**4/5),
                        scope='Exact free real-cover/Bloch representation boundary, NOT a currentN4 spectral result. Reject arbitrary Borel point evaluation; retain beta-continuous physical C* algebra and original-Hilbert packet limit.')
    return dict(Borel_fiber_boundary=borel_boundary,nonempty_source_CAR=dict(active1_Ref3_state={str(k):str(v) for k,v in source.items()},norm_squared_exact='1',all_local_traceless_SU4_charges_annihilate=True,scope='Nonempty-sector witness only; not physical prepared-source probability'),nonuniform_domain=dict(good_b6_Phi_factor='(1-x^2)^6(1+x+x^2)',good_boundary_jets_0to5=list(map(str,gj)),good_H_boundary_jets_0to3=list(map(str,gh)),wrong_b2_H_boundary=str(H(bad).subs(x,1)),wrong_zero_extension_H_squared_rejected=True,periodic_rotor_and_smooth_SU4_factors='Their smooth bounded derivatives preserve these boundary zero jets; compactPhi support and elliptic order2 give D(Hbeta^2).',scope='Exact local zero-gluing control, not a numerical currentN4 field spectral truncation'),Clifford_PH_trace_exact=[list(map(str,row)) for row in trace],abstract_CAR_rotor=dict(operator='Hbeta=(-i d_t+beta-1/2)^2 I + c0dag c1+c1dag c0 in fixedN1 CAR2, on periodic torus; finite-cell compact resolvent',exact_E0_beta0=str(e0),exact_E0_beta_half=str(ehalf),global_minimum_beta_mod1='1/2',beta0_minimum_assumption_rejected=True,scope='ABSTRACT independent periodic CAR/rotor adverse case; NOT currentN4 no-go or a currentN4 added holonomy'))


def run(output:Path) -> dict[str,Any]:
    start=time.monotonic();gauss.iv.dps=110
    captured={}; sources={}
    def take(name:str,pin:str|None=None)->dict[str,Any]:
        raw=(ROOT/name).read_bytes()
        if pin is not None and sha(raw)!=pin:raise ArithmeticError('Pinned input changed: '+name)
        captured[name]=raw;return json.loads(raw)
    records={name:take(name,pin) for name,pin in PINNED.items()}
    archive=records['induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'];mean=archive['input_descriptor']['exact_NEW_allq_BO_mean']; centres=[F(v) for v in mean['centre_exact_rational']]
    data=field.load_mean(g_override=[float(centres[0]),float(centres[1]),float(centres[1]),float(centres[1])],mean_descriptor=mean)
    d=np.rint(data['native']['D']).astype(np.int64);assert np.array_equal(d,data['native']['D']);c,chords=cycles(d)
    action=records['joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json'];price=uniform_price(action,take)
    moments=exact_moments(data,c,mean)
    shift=field.full.enclosure(F(moments['DeltaEbody']['lower']),F(moments['DeltaEbody']['upper']));eps=gauss.exact(upper(price['epsilon']))
    improvement=shift+2*eps;assert gauss.endpoint(improvement,1)<-1000
    constants=[]
    for p in data['profiles']:
        constants.append(abs(F(p['onsite']))+2*sum((abs(F(v['c0']))+abs(F(v['ci'])) for v in p['paths']),F(0)))
    b0=128*constants[0]+129*constants[1]+8064*constants[2];c0=cuberoot_upper(F(27*128,256)*b0**4);lam=F(2)**8192;C=lam*c0
    graph=records['joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json'];erefabs=upper(graph['global_constants']['Erefabs'])
    with threadpool_limits(limits=1):finite=nonuniform_controls(data,c)
    controls=exact_adverse_controls()
    for name in DIRECT:sources[name]=(ROOT/name).read_bytes()
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).parent==ROOT and Path(name).suffix=='.py':sources[Path(name).name]=Path(name).read_bytes()
    for table in (field.CAPTURED,preparation.SOURCE_BYTES,gauss.SOURCE_BYTES):
        for name,raw in table.items():
            if name in sources and sources[name]!=raw:raise ArithmeticError('Import source epoch changed: '+name)
            sources[name]=raw
    # Preserve OLD input-producer epochs, never relabel current sources as old.
    for name,record in archive['source_at_import'].items():
        if 'bytes_utf8' in record:
            raw=record['bytes_utf8'].encode();assert sha(raw)==record['sha256'];sources['archived_preparation/'+name]=raw
    source_audit={}
    for name,names in {'induced_phi_nonuniform_field_car.py':['assemble','native_paths'],'induced_phi_nonuniform_fermionic_source_unitary.py':['field_ray','h_jet','occupied'],'induced_phi_global_normal_source_reprice.py':['run'],'joined_phi_u1_actual_compact_source_action_certificate.py':['run']}.items():
        tree=ast.parse(sources[name]); source_audit[name]=[]
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
                text=ast.get_source_segment(sources[name].decode(),node)
                if text is None:raise ValueError('Missing literal source segment')
                source_audit[name].append(dict(symbol=node.name,line=node.lineno,end_line=node.end_lineno,sha256=sha(text.encode())))
        assert len(source_audit[name])==len(names)
    input_hash={name:sha(raw) for name,raw in captured.items()};source_hash={name:sha(raw) for name,raw in sources.items()}
    result: dict[str,Any]
    result=dict(kind='ACTUAL_FULLPHI_FIBER_EXISTENCE_AND_CONDITIONAL_NEW_ALLNATIVE_VARIATIONAL_TRIAL',proof=PROOF,claim_boundaries=dict(derived='Full primitive semibounded closed fiber, nonempty compatible sectors, compact resolvent and attained fiber grounds, continuous beta energy and some minimizing beta, stationary positive periodic physical functional.',exact='Integer cycle/Green/Clifford/native character arithmetic, local Gauss CAR witness and abstract rotor adverse case.',conditional='Implicit saved root/gap/SPD/chart and action suppliers needed ONLY for actual T trial and numerical lowering, not existence.',finite='Fresh nonuniformPhi/fullSU4/U1 structural controls; no ground eigensolve.',open='Vacuum characterization, beta* value, real-cover L2 ground/flat bands and photon pole remain open.'),plan=dict(id='issue-222-induced-photon-delta13-continuation-20261008',assignment_revision='444bae46f65f84b4ee667368135eaf25229de62d786588bc980fc8511e28ce49',contract_revision='10b3c5e3e75de73f67e3e6a20f2a91c75ed63c9e17762fd848f15b8697372ed2',template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='induced-photon',requirements=['verification','outcome']),fiber_geometry=dict(D_shape=list(d.shape),C_shape=list(c.shape),DC_exact_zero=True,chord_minor_exact_identity=True,chords=chords,C_sha256=sha(c.tobytes()),electric_metric_lower_exact=str(F(64,int(np.sum(c*c)))),ambient_CartPhi=3840,SU4_links=192,periodic_Theta=129),global_lower_bracket=dict(c_quartic_exact=str(lam/128),K_exact=[str(v) for v in constants],Young_C0_exact=str(c0),C_exact=str(C),ground_lower_absolute_exact=str(-C),Ephi_absolute_upper_exact=str(erefabs),ground_minus_Ephi_lower_exact=str(-C-erefabs),precision='Coarse trace/Young lower bound; no tiny ground-gap or normal-form precision claim'),uniform_intertwiner_price=price,new_ground_directed_trial=dict(**moments,fullPhi_trial_minus_fullPhi_reference_upper=gauss.bounds(improvement),epsilon_paid_at_BOTH_endpoints=True,fullPhi_ground_upper_minus_common_Ephi=gauss.bounds(field.full.enclosure(F(moments['Etrialbody']['lower']),F(moments['Etrialbody']['upper']))+eps),Ephi_semantics='Same static+bosonic reference scalar minus chargedroot sea; not chosen or separately fitted. Unknown absolute fullPhi ground is not numerically characterized.'),finite_nonuniform_physical_controls=finite,independent_exact_adverse_controls=controls,literal_source_AST_audit=source_audit,actual_uniform_T_status='No structural discrepancy found in the literal current originalFullForm recipes; numerical T/R prices remain supplier-conditional, not an unconditional measured isometry or vacuum.',input_sha256=input_hash,source_sha256=source_hash,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,sympy=sy.__version__,interval_digits=gauss.iv.dps),next_material_decision='Use the attained fullPhi physical fiber grounds, not prepared/root/static states, to characterize beta dependence and stationary full129 current/contact spectral measures; a same-ground pole/residue/dispersion still needs a new controlled construction.')
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    assert all((ROOT/name).read_bytes()==raw for name,raw in sources.items() if not name.startswith('archived_preparation/'))
    result['source_inputs_unchanged']=True;result['elapsed_seconds']=time.monotonic()-start
    archivepath=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archivepath,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in {**sources,**captured}.items():z.writestr(name,raw)
    result['immutable_source_input_capture']=dict(path=archivepath.name,sha256=sha(archivepath.read_bytes()))
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();receipt=run(args.output)
    print(json.dumps(dict(receipt=args.output.name,receipt_sha256=sha(args.output.read_bytes()),source_sha256=receipt['source_sha256'][Path(__file__).name],archive=receipt['immutable_source_input_capture'],trial=receipt['new_ground_directed_trial']['fullPhi_trial_minus_fullPhi_reference_upper'],existence='Derived fullPhi fiber existence, NOT characterized vacuum/pole',supplier_boundary='Conditional T/root/SPD/gap/chart/action prices',elapsed_seconds=receipt['elapsed_seconds']),indent=2))
