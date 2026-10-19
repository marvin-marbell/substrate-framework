"""Conditional GLOBAL Friedrichs-square domain proof for the actual NEW packet.

Exact boundary and coercivity identities, not an interior-grid domain test.
No root solver, source preparation, operator parameter, or shared file is changed.
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

import sympy as sy
import induced_phi_quantum_fast_transfer as fast

ROOT = Path(__file__).resolve().parent
PREFIX = 'induced_phi_compact_source_global_domain_control'
sys.set_int_max_str_digits(0)
PINNED = {
    'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json': 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6',
    'induced_phi_global_normal_gauss_chart_certificate_receipt.json': '9152079e9ba0334a49aff3387e60fc497478453aee252a4c77468060502488e0',
    'induced_phi_global_occupied_kato_derivative_control_receipt.json': '270cdefd1b7bd9d68fcf94388ac60aee3c4a8c7d8ad75de099646c72bbb4714f',
    'induced_phi_complex_matter_jet_certificate_parent_review_20261010.json': '04c90d9e061ef981f954754716104b31c2242e4fee865ee3e4b01224e5203485',
}
DIRECT = (
    'bosonic_original_law_join.py', 'bulk_quantum_selection.py',
    'induced_phi_nonuniform_fermionic_source_unitary.py',
    'induced_phi_nonuniform_gauss_quantum.py',
    'induced_phi_global_normal_gauss_chart_certificate.py',
    'induced_phi_global_occupied_kato_derivative_control.py',
    'induced_phi_complex_matter_jet_certificate.py',
    'joined_phi_u1_finite_current_kernel.py',
    'joined_phi_u1_nonlocal_receiver_electric_control.py',
    'joined_phi_u1_unrestricted_source_phase_control.py',
)
PROOF = {
    'hilbert': 'H=L2(R^3840_CartPhi x SU4^192 x R^129_Theta, dCartPhi dHaar dTheta) tensor the finite-dimensional fixed-species-particle Fock space including Ref3 or Ref4, restricted to the compatible Gauss/centre sector. This is global, not L2 on a truncated normal chart. U1 is Gauss-reduced unwrapped R129, not gauge-invariant normalizable L2 on unreduced R192. The source/background sectors are handled separately with the same primitive coefficients.',
    'positive_form': 'Start on smooth compact CartPhi/Theta functions and smooth Haar functions with finite Fock components (or their Gauss projection). qplus is chi/2[8||pCartPhi psi||²+(3/10)sum||E_SU4 psi||²]+64sum129||PiTheta psi||² plus Lambda/a sum_x||Gram_x||F², scientific-source (1/5)Lambda^(5/4)alignment/8, (7/10)Lambda^(5/4)sum_faces(4-ReTr Uface), and HC=64 rho_allcharged Lplus rho_allcharged. This NEW joined law has no added bare EM curl/B² potential. chi=2^-4096,Lambda=2^8192,a=1/64. Gram_mn=Tr(Phi_m Phi_n)/4. All positive potential matrices are smooth; HC is positive and bounded in each finite-particle sector. The derivative form is closed by completion, and adding nonnegative locally bounded potentials is a closed form on their weighted L2 intersection. The core is form dense by CartPhi/Theta cutoffs, smooth manifold partitions, mollification and finite Fock components; Gauss projection commutes with these invariant form limits.',
    'coercivity': 'Gram is PSD4 as a Gram matrix of Hermitian Phi_m/2 in the real Hilbert-Schmidt inner product. ||Gram_x||F² >= (Tr Gram_x)²/4 = (sum_m||Phi_xm||HS²)²/64. Spatial Cauchy gives sum_x t_x² >= (sum_x t_x)²/V. Hence Vquartic >= c||Phi||HS^4, c=Lambda/(64aV), V=64. The receipt checks both difference-of-squares identities exactly, not a sampled PSD test.',
    'matter_form_bound': 'Actual native profiles retain onsite and both Phi endpoints of every ordered one/two-link AP path. SU4 path products have norm1 and real U1 characters modulus1. For profile j, ||h_j(Phi,U,Theta)|| <= Kj X, X=||Phi||HS, Kj=|onsite_j|+2sum_paths(|c0|+|ci|); the row/block Schur bound follows from ||sum c_m Phi_m|| <= sum|c_m| X, with endpoint averaging and both orientations. Fixed-N ||dGamma(h_j)|| <= Nj||h_j||. Thus ||M|| <= B X+Cref with B=128 Lambda Kneutral+129 Kactive+8064 Kaux (background active128 is smaller); the neutral physical coefficient is -Lambda, all64 charged flavours have unit scale, and Ref3/4 is retained. Cref is finite if an additional field-independent reference term is present. For any epsilon>0, Young gives B X <= epsilon c X^4 + Cepsilon, Cepsilon^3=27 B^4/(256 epsilon c). Using Cepsilon<=1+Cepsilon^3 yields an explicit rational constant. Consequently |qM[psi]| <= epsilon qplus[psi]+(1+Cepsilon^3+Cref)||psi||². epsilon=1/2 is checked; KLMN gives a closed semibounded form and its UNIQUE Friedrichs selfadjoint operator H_F. No essential-selfadjointness assertion is used. B is an intentionally loose existence bound, not an action or interval price.',
    'actual_preparation': 'Use only the bound NEW preparation: fixed SPD full5760 Gaussian f_A(root), fixed Xi(root) degree3 corrector P, b6=(1-||r||²/R²)^6_+, R=2^-20, scalar nonzero normalization Z^-1/2, sqrt(J)^-1, and actual occupied REAL radial W_Kato solving d_t W=[DP(tr)[r],P(tr)]W,W(0)=I. Include full SU4^64 normalized Haar (960 directions), with no restricted-angle substitute. Phi45 uniform slow +2835 zero-mean fastPhi+2880 colour=5760; plus960Haar yields the actual ambient6720 chart. W remains the actual Phi/colour radial frame built at Theta0 and is independent of the evolving Theta; no Theta-following occupied projector is introduced. No f_A(slow), CartQpolar-as-occupied-W or new covariance is substituted. Multiply by the actual real Theta129 Gaussian (density variance2), a Schwartz factor.',
    'real_smoothness': 'The named conditional exact root, positive-g Gram/SPD Gaussian, fixed Xi, injective3R collar, positive smooth Haar Jacobian, polar-normal recipe and uniform real occupied/unoccupied gap remain premises. The bound complex-P producer uses a fixed contour and analytic actual h on a complex neighbourhood. Restriction gives P C-infinity on the real collar (rank128 occupied sheet; neutral physical -Lambda uses its complement). Then K(t,r)=[DP(tr)[r],P(tr)] is smooth in (t,r). On compact [0,1] times any real closed subcollar it is bounded with each fixed derivative. Linear finite-dimensional ODE existence/uniqueness and differentiation of its integral equation give W smooth in REAL r to every finite order. Anti-Hermiticity gives unitary W on real r. Finite exterior lifts and fullHaar representations are smooth. This does NOT assert quantitative complex-W bounds or prices.',
    'boundary': 'At the sphere ||r||=R the interior factor b6 has a zero of order6 in a smooth normal defining function; every Cartesian/mixed derivative of total order0..5 has zero trace. Leibniz preserves these zero traces after multiplying by the actual smooth Gaussian/corrector/Jacobian/Kato/Haar/Fock factors. Extend by zero inside the injective3R chart and then onto the GLOBAL manifold; the collar prevents artificial chart-edge jumps. Normal weak derivatives through order4 are the interior derivatives extended by zero (no boundary delta terms), hence locally H4. Smooth Haar and Schwartz Theta provide the remaining derivatives. Only H4 is needed; b6 is C5, not globally C-infinity.',
    'global_L2': 'The global primitive differential expression H has order2 with smooth coefficients on the packet support. CartPhi support is compact after including the compact fullHaar orbit, so polynomial Gram/alignment/matter coefficients and every finite-order CartPhi derivative are bounded there; SU4 is compact. Theta-dependent characters and all their derivatives are bounded for each actual finite path; HC is a finite matrix at fixedN. Schwartz Theta is preserved by the actual electric differentiation and finite characters. Therefore psi,Hexpr psi,Hexpr² psi are L2. More strongly psi and eta=Hexpr psi are in the global positive form domain: psi has zero boundary jets through5, eta through3; eta has H2 normal regularity, smooth Haar and Schwartz Theta. No finite-point interior integral alone establishes this.',
    'Friedrichs_identity_twice': 'Approximate psi and eta by the global form core using chart partitions, normal mollification of the zero-glued H4/H2 functions and Theta cutoffs. Zero traces eliminate all boundary Green terms; smooth Haar integration uses unimodular Haar and there is no compact-angle endpoint. For every core test v, q(v,psi)=<v,eta> and q(v,eta)=<v,zeta>, zeta=Hexpr²psi in L2. Continuity in the closed semibounded form norm and core density extend both identities to EVERY v in Q(q). The representation theorem first gives psi in D(H_F),H_F psi=eta, and then eta in D(H_F),H_F eta=zeta. Thus psi in D(H_F²) of the GLOBAL semibounded Friedrichs operator, not just a chart differential expression. Subtracting a finite reference scalar or normalizing the packet does not alter these domains.',
    'insertions': 'The same proof applies separately to E0psi, all129 physical current J_mu psi=-dH/dA_mu psi, every contact C_mu_nu psi=d²H/dA_mu dA_nu psi (129x129), and any fixed finite-Wilson receiver E_xpsi and its adjoint/product in their correct source/background Ref sectors. Each current/contact coefficient is a finite sum of native paths, linear in actual Phi, with ordered SU4 matrices and finite real exp(i q.Theta) characters; each derivative multiplies by fixed finite q components. Neutral charge0 contributes no U1 current, but remains in H. Each receiver is a smooth ordered finite SU4 path times finite U1 character times a finite CAR/Ref creation/hole map. Compact Phi polynomials, these finite matrix multipliers, and finite compositions preserve zero jets through5, H4 normal regularity, smooth Haar and Schwartz Theta. Therefore both form identities apply again to each inserted vector. This is domain preservation ON THIS packet class, not an assertion that arbitrary multipliers preserve every vector in D(H_F²). No receiver path length limit is inferred beyond each specified finite path.',
    'common_Z4': 'Simultaneous central iI at all64 Haar sites leaves Phi/link chart unchanged and acts on the full matter+Ref lift by i^N. Source N=128neutral+129active+63*128aux+3Ref=8324 is divisible4; background128+128+8064+4=8324 too. Omitting Ref gives source8321, which fails. Omitting Ref from background gives8320, divisible4 and BLIND; this blindness is retained as a limitation, not treated as a CAR test. Local centre chart transformations that alter adjacent ordered links are coordinate changes, not extra identifications; the chart kernel is the commonZ4 only.',
    'stage': 'Conditional mathematical global form/domain closure for NEW compactPhi/SU4/U1 packet and actual physical insertions. Exact algebra and symbolic adverse boundary control are independent of floating point displays. Root/static/gap/SPD/chart input suppliers remain conditional. PREnormalization and PREPARED share the same domain; no new quantitative norm/interval action prices are asserted. No original SOURCE/CURRENT427 adoption, global ground, pole, common cone, current transfer, or dynamical current-kernel admission.',
}


def exact_coercivity(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    d = sy.symbols('d0:4', real=True)
    off = sy.symbols('o0:6', real=True)
    remainder = sum(x*x for x in d)+2*sum(x*x for x in off)-sum(d)**2/4
    sos = sum((d[i]-d[j])**2 for i in range(4) for j in range(i))/4+2*sum(x*x for x in off)
    assert sy.expand(remainder-sos) == 0
    t = sy.symbols('t0:64', real=True)
    spatial = sum(x*x for x in t)-sum(t)**2/64
    spatial_sos = sum((t[i]-t[j])**2 for i in range(64) for j in range(i))/64
    assert sy.expand(spatial-spatial_sos) == 0
    rows = []
    for p in profiles:
        k = abs(F(p['onsite']))+2*sum(abs(F(v['c0']))+abs(F(v['ci'])) for v in p['paths'])
        rows.append({'role':p['role'], 'K_exact':str(k), 'load':p['load'],
                     'paths':[{'axis':v['axis'],'length':v['length'],'c0':str(v['c0']),'ci':str(v['ci'])} for v in p['paths']]})
    assert len(rows) == 3 and rows[1]['load'] == 1 and rows[2]['load'] == 63
    lam, a, volume = F(2)**8192, F(1,64), 64
    c = lam/(64*a*volume)
    b = 128*lam*F(rows[0]['K_exact'])+129*F(rows[1]['K_exact'])+8064*F(rows[2]['K_exact'])
    epsilon = F(1,2)
    young_cube = 27*b**4/(256*epsilon*c)
    X, B, e, C = sy.symbols('X B e c', positive=True)
    xmax = (B/(4*e*C))**sy.Rational(1,3)
    optimum = sy.simplify(B*xmax-e*C*xmax**4)
    assert sy.simplify(optimum**3-27*B**4/(256*e*C)) == 0
    assert c>0 and b>0 and epsilon<1 and c == lam/64
    return {'Gram_PSD_recipe':'Gram_mn=<Phi_m/2,Phi_n/2>HS',
            'Gram_SOS_identity_exact':True,'spatial_SOS_identity_exact':True,
            'quartic_coercivity_coefficient_exact':str(c),'coercivity_formula':'Lambda/(64*a*V)',
            'profiles':rows,'matter_linear_B_exact':str(b),
            'relative_form_bound_exact':str(epsilon), 'Young_constant_cubed_exact':str(young_cube),
            'rational_additive_form_constant_exact_excluding_optional_Cref':str(1+young_cube),
            'Young_stationary_identity_exact':True,'bound_semantics':'existence/closedness, NOT action price'}


def boundary_control() -> dict[str, Any]:
    x = sy.Symbol('x', real=True)
    gaussian = sy.exp(-x*x/2)
    def H(v: Any) -> Any:
        return sy.expand(-sy.diff(v,x,2)/2+x**4*v)
    records = {}
    for power in (6,2):
        psi = (1-x*x)**power*gaussian
        hpsi, h2psi = H(psi), H(H(psi))
        jets = [[sy.simplify(sy.diff(psi,x,k).subs(x,s)) for k in range(6)] for s in (-1,1)]
        hjets = [[sy.simplify(sy.diff(hpsi,x,k).subs(x,s)) for k in range(4)] for s in (-1,1)]
        # For zero exterior values: D²f includes [f]delta'+[f']delta.
        # The common boundary assertion, not the naive integral, rejects b2.
        accepted = all(v==0 for side in jets for v in side) and all(v==0 for side in hjets for v in side)
        poly = sy.Poly(sy.expand(h2psi/gaussian),x)
        squared = sy.Poly(poly.as_expr()**2,x)
        # Exact Gaussian moments on [-1,1], recurrence by integration by parts.
        moments: dict[int, Any] = {0:sy.sqrt(sy.pi)*sy.erf(1)}
        for k in range(2,len(squared.all_coeffs()),2):
            moments[k] = (k-1)*moments[k-2]/2-sy.exp(-1)
        integral = sy.expand(sum(coeff*moments[degree[0]] for degree,coeff in squared.terms()))
        records[str(power)] = {'endpoint_psi_jets_orders0through5':[[str(v) for v in side] for side in jets],
            'endpoint_Hpsi_jets_orders0through3':[[str(v) for v in side] for side in hjets],
            'accepted_by_boundary_assertion':accepted,'interior_H2_polynomial':str(poly.as_expr()),
            'naive_interior_H2_norm_squared_exact':str(integral),
            'naive_interior_H2_norm_squared_float_control':str(sy.N(integral,30)),
            'interior_integral_finite_by_polynomial_compactness':True}
    assert records['6']['accepted_by_boundary_assertion'] is True
    assert records['2']['accepted_by_boundary_assertion'] is False
    wrong = (1-x*x)**2*gaussian
    assert sy.simplify(sy.diff(wrong,x,2).subs(x,1)-8*sy.exp(-sy.Rational(1,2)))==0
    assert sy.simplify(H(wrong).subs(x,1)+4*sy.exp(-sy.Rational(1,2)))==0
    records['wrong_b2_distribution'] = {
        'psi_D2_at_1minus':'8*exp(-1/2)', 'Hpsi_at_1minus':'-4*exp(-1/2)',
        'Hpsi_jump_at_plus1':'4*exp(-1/2)',
        'H2_delta_prime_coefficient_at_plus1':'-2*exp(-1/2)',
        'reason':'b2 is H2 and in D(H_F), but Hpsi has a nonzero jump. -1/2 D²(Hpsi) contains nonzero delta-prime (and delta); x^4 Hpsi cannot cancel them. Thus b2 is NOT in D(H_F²), although naive interior H² norm is finite.'}
    return records


def centre_control() -> dict[str, Any]:
    source = 128+129+8064+3
    background = 128+128+8064+4
    wrong_source, wrong_background = source-3, background-4
    assert source == background == 8324 and source%4==0
    assert wrong_source==8321 and wrong_source%4!=0
    assert wrong_background==8320 and wrong_background%4==0
    return {'source_N':source,'background_N':background,'source_common_Z4_character':'1',
            'wrong_omitRef_source_N':wrong_source,'wrong_omitRef_source_character':'i',
            'wrong_omitRef_source_rejected':True,'wrong_omitRef_background_N':wrong_background,
            'background_omitRef_blind':True,'control_scope':'centre equivariance, NOT CAR test'}


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    raws = {name:(ROOT/name).read_bytes() for name in PINNED}
    for name,pin in PINNED.items():
        if hashlib.sha256(raws[name]).hexdigest()!=pin:
            raise ArithmeticError(f'Immutable input mismatch: {name}')
    old = json.loads(raws['induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'])
    desc = old['input_descriptor']
    profiles = fast.primitive_profiles({'microscopic_config':desc['original_micro_config_UNCHANGED']})
    own = Path(__file__).read_bytes()
    captures = dict(fast.SOURCE_BYTES)
    captures.update({name:(ROOT/name).read_bytes() for name in DIRECT})
    captures[Path(__file__).name] = own
    # Retain the actual source-preparation execution bytes rather than relabel
    # current bytes as the old source. This does not execute the old supplier.
    for name,record in old['source_at_import'].items():
        if 'bytes_utf8' in record:
            raw = record['bytes_utf8'].encode()
            if hashlib.sha256(raw).hexdigest()!=record['sha256']:
                raise ArithmeticError(f'Archived source bytes inconsistent: {name}')
            captures['archived_preparation/'+name] = raw
    result = {'kind':'conditional_GLOBAL_primitive_Friedrichs_square_compact_source_domain',
        'proof':PROOF,'exact_coercivity_and_relative_form_bound':exact_coercivity(profiles),
        'symbolic_boundary_adverse_control':boundary_control(),'common_Z4_control':centre_control(),
        'actual_preparation_binding':{'input_descriptor':desc,
            'radius_exact':'1/1048576','bump_power':6,'full_fixed_joint_Gaussian_dimensions':5760,
            'ambient_CartPhi_dimensions':3840,'ambient_SU4_link_count':192,
            'fullHaar_dimensions':960,'Theta_real_Gauss_reduced_dimensions':129,
            'normalizer_and_fixed_Xi_binding':'input descriptor exact_wave and archived actual preparation bytes',
            'source_definition':'.5 sum4 c_active0alpha^dag Ralpha, Ref4->3'},
        'input_sha256':PINNED,
        'source_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in captures.items()},
        'plan':{'id':'issue-222-induced-photon-delta13-continuation-20261008',
                'revision':'213ae802ebba544ced47e9cb553dde901d46dc57ca94beca9ce17256fa8e41a7',
                'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
                'work':'joined-explanatory-campaign/induced-photon; evidence verification'},
        'runtime':{'executable':sys.executable,'python':sys.version,'sympy':sy.__version__}}
    assert all((ROOT/name).read_bytes()==raw for name,raw in raws.items())
    assert all((ROOT/name).read_bytes()==raw for name,raw in captures.items() if not name.startswith('archived_preparation/'))
    result['input_and_source_unchanged'] = True
    result['elapsed_seconds'] = time.monotonic()-started
    archive = output.with_name(output.stem+'_immutable_inputs.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in {**raws,**captures}.items():
            z.writestr(name,raw)
    result['immutable_input_archive'] = {'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    receipt = run(args.output)
    print(json.dumps({'output':str(args.output),'receipt_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'boundary_b6_accepted':receipt['symbolic_boundary_adverse_control']['6']['accepted_by_boundary_assertion'],
        'boundary_wrong_b2_rejected':not receipt['symbolic_boundary_adverse_control']['2']['accepted_by_boundary_assertion'],
        'centre':receipt['common_Z4_control'],'stable':receipt['input_and_source_unchanged'],
        'stage':PROOF['stage'],'elapsed_seconds':receipt['elapsed_seconds']},indent=2))
