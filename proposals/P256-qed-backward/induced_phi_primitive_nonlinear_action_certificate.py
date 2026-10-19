"""Actual fixed-source primitive/filled-sea Taylor action, conditional N4 root.

This certificate does not admit the whole action, source, ground or pole.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import induced_phi_nonuniform_fermionic_source_unitary as source
import induced_phi_nonuniform_gauss_quantum as gauss
import induced_phi_primitive_nonlinear_action_certificate_controls as controls

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).read_bytes()
R = F(1, 1 << 20)
LAMBDA = F(2)**8192
LAMBDA54 = F(2)**10240
PLAN = 'issue-222-induced-photon-delta13-continuation-20261008'
ASSIGNMENT_REVISION = '92c5eb8e5cd492763b25ab20238d14440c757d50f9ae5207b29f3342362bb8da'
GOVERNING_REVISION = 'ee8f864c6f0d1b368c3ae52a8573855c5fe0001d19c89e828248aaf74e4c9fc7'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
sys.set_int_max_str_digits(0)

PROOF = {
 'operator': 'V4=(Lambda/a) sum_x ||G_x||F^2, G_x,mn=Tr(Phi_x,m Phi_x,n)/4. Va=(1/5)Lambda^(5/4) sum_{x,i,mu} ||Phi_x,mu-U_xi Phi_(x+i),mu U_xi^dag||HS^2/8. Vm=(7/10)Lambda^(5/4) sum_{x,i<j} ||U_xi U_(x+i),j U_(x+j),i^dag U_xj^dag-I4||HS^2/2. Matter is the actual all65 independent rank128 seas of signed h: negative heavy h has physical multiplier Lambda, active and63 auxiliary copies multiplier1. One AND two ordered paths, BOTH endpoint Phi, reverse adjoint and native AP seams are retained.',
 'law_owner': 'Pinned NEW source ABI and its exact algebraic producer are the coefficient owner, not a floating evaluator: induced_phi_nonuniform_gauss_quantum.py194-200 explicitly says interval/numeric displays are not scientific wave coefficients;222-224 and232 use exact F(1,5), F(3,50), F(q2,5), F(21*q2,200), F(7*q2,20). This fixes alignment1/5 and plaquette7/10. Native full_form_contract .2/.7 names the same functional and field_potential_at and link_defects supply its spatial/trace/order conventions. The compatibility of that exact declared full law with the frozen source is an explicit law-binding antecedent. Python .2/.7 finite evaluation is NOT promoted to an exact binary physical operator. The binary-literal alternative mismatch is separately priced, never dropped or made small by retuning.',
 'coordinates': 's is the45 uniform slow vector. f is the63q zero-mean fastPhi vector in an orthonormal REAL Fourier transform, ||f||^2=sum_x ||f_x||^2. C has192*15 edge coordinates. B(s)=B0+Q0 s.T/8, Phi_x=B(s)+Q(s) f_x.T. T_A=M_A/sqrt8, Tr(T_A T_B)=delta_AB/2. Q(s)=P(s)Q0(Q0^T P(s)Q0)^-1/2 is the actual normal-polar map; at real s, Q^T Q=I and Q^T O(s)=0. This is not occupied W_Kato. Local960 angular and all unrestricted CartPhi3840/SU4link192 directions are related by exact gauge invariance of the potential; no local-angle potential derivatives or charged source projection are omitted.',
 'collar': 'Every inequality below holds on ||(s,f,C)||<=3R, including supportR. Archived root totalg<2/3 gives ||B(s)||HS<2. The complex polar proof on radius1/4096 gives ||Q||<9/8; Cauchy with margin1/4096-3R yields ||DQ||<=8192. Real Q is exactly isometric. These are global support/collar inequalities, not finite-direction claims.',
 'quartic': 'For F_x=Q(s)f_x.T and G(B+F)=G(B)+J_B(F)+G(F), ||J_B(F)||F<=||B||HS||F||HS/2 and ||G(F)||F<=||F||HS^2/4. The sum of the linear fast term is EXACTLY0 since sum_x f_x=0, at EVERY s. Uniform slow Frobenius norm per site is ||s||/(8sqrt2); fast total Frobenius norm is ||f||/sqrt2. The uniform cubic/quartic remainder after degree2 is <=Lambda(2s^3+s^4/64). The fast cubic/quartic is <=Lambda(12f^3+f^4). The fast quadratic at B(s),Q(s) differs from its frozen root value by <=Lambda(5+64*8192)s f^2: B variation costs5, Q variation costs64||DQ||. No Lambda*s*f quadratic is discarded: its spatial sum is exactly0, and this antecedent is tested adversely.',
 'alignment_hessian': 'Let O(s)=O0+Od(s), ||Od(s)||op<=s (sqrt15/4<1 Casimir with slow /8). Linear covariant difference L_e(s)=F_x-F_y-i[X_e,B(s)], X_e=C_e.T. Horizontal Q(s)^T O(s)=0 kills its Phi/colour cross EXACTLY even before summing. Thus Va_linear=(Lambda54/80)sum||f_x-f_y||^2+(Lambda54/10)sum C_e^T H(s)C_e, H=O^T O/8. This is precisely frozen allq KfastPhi=qhat^2 I/40 and KfastC=I3 tensor H/5. Retain the actual linear-s colour term Lambda54/2 C^T(dKfast[s])C, not a retagged negligible cubic; it is the potential part of rootXi forcing. The leftover quadratic-H term is <=Lambda54*s^2*C^2/80. Kinetic -A dG A, mean/ZPE trace and action of P belong to parent/kinetic owner.',
 'alignment_remainder': 'Real unitary D_e-L_e=-(AdU-I-i adX)B-(AdU-I)F_y. Integral Taylor gives ||AdU-I||<=2||X||op, ||AdU-I-iadX||<=2||X||op^2, ||X_e||op<=||C_e||/sqrt2. Every site is endpoint of3 outgoing bonds; hence ||N||ell2HS<=2 C^2+sqrt3 C f and ||L||ell2HS<=sqrt6 f+2sqrt2 C. The SUM f+C is NOT the combined fast Euclidean norm t=sqrt(f^2+C^2). Cauchy gives ||L||<=sqrt14 t<4t. The quadratic form 2 C^2+sqrt3 Cf has top eigenvalue1+sqrt7/2<3, hence ||N||<=3t^2. Therefore |Va-Va_linear|<=(Lambda54/40)(2*4t*3t^2+9t^4)=Lambda54[(3/5)t^3+(9/40)t^4]. Includes ordered noncommuting colour and all nonuniform endpoints; no affine-link substitute. Previous sum-to-combined norm error was explicitly withdrawn; finite agreement alone had not certified it.',
 'plaquette_remainder': 'Use the literal ordered four-link word and positive squared norm, not 4-ReTr subtraction. Lface=i(X1+X2-X3-X4), ||L||ell2HS<=sqrt6 C<3C by curl spectrum qhat^2<=12. Along t, product second derivative normHS<= (sum||X_i||op)(sum||X_i||HS); integral Taylor gives ||Nface||HS<=sum_face ||C_e||^2. Every edge belongs to4 faces, so ||N||ell2HS<=2C^2. Thus |Vm-(7/20)Lambda54||L||^2|<=Lambda54[(21/5)C^3+(7/5)C^4]. All tensor/face factors are counted; no claim that nonabelian cubic vanishes.',
 'matter': 'NEW complex Riesz receipt supplies Emax=256(H+delta)Pmax on the same fixed contour and all-normal analytic h in the tube1/65536; heavy sign is already in h. At every real ||r||<=3R, multivariate Cauchy with n discs of radius margin/n gives ||D^n E||multilinear<=n^n Emax/margin^n. These scalar all-normal third/fourth/fifth bounds NUMERICALLY equal that receipt K1/K2/K3 formulas; this is a fresh semantic derivation, not relabeling K_j slow-to-fast Hessian tensors. Hence |E(r)-E0-E1[r]-E2[r,r]/2|<=E3*r^3/6. Physical E3 is Lambda*K1_heavy+K1_active+63K1_aux. K0..K3 are all recorded with physical scaling. No physical Lambda is added to projector jets. Frozen static Hessian equals tied-chart Hessian at mean: chart curvature is uniform derivative DQ[s]f and contracts a translation-invariant mean gradient to zero because sum f=0. Retain native contact-minus-BOTH PH quotients; do not reuse per-q D1, commonC or force consumer.',
 'Gaussian_price': 'Fixed f_A(root), Xi(root), C0(root), M=A_CC^-1 A_CP, P degree3 and b6 are unchanged. NEW outward Gaussian form comparison gives C<=r C0, r=(1-eps)^-1/2; completion has independent s,z=(f,C+M s) and ||physicalFast||<=||z||+||M||||s||. s covarianceop<=r sqrt8/(2sqrt(slowlo))*2^-6144, NEVER std=2^-3072. z covarianceop<=r max(CL,CT,4/sqrt(qhat^2/5))*2^-7168 over ALL64q. For covariance<=c I_d, ||radius^k||p<=[c^(pk/2) product_{j=0}^{pk/2-1}(d+2j)]^(1/p). Expand nonnegative polynomials in s,z and use independent marginal moments. Holder and degree3 hypercontractivity give ||F(1-P)f_A||2<=||F||4(1+9||P||2); b6<=1. Recorded result is before normalization plus a separate bound with proved normalization multiplier2. No actual normalization or preparation is changed.',
 'scope': 'Literal formulas, global polynomial inequalities, outward weighted arithmetic and finite observations are separate. Exact root existence/admission, static q0 positivity and source-law agreement remain explicit assumptions; this is not a full D(H^2), action, source, ground, pole, photon/Compton or originalSOURCE certificate.'
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def upper(record: dict[str, Any]) -> F:
    return F(record['upper'])


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, bytes]]:
    inputs: dict[str, bytes] = {}
    def take(name: str, expected: str | None = None) -> dict[str, Any]:
        raw = (ROOT/name).read_bytes()
        if expected is not None and digest(raw) != expected:
            raise ArithmeticError('Pinned input changed: '+name)
        inputs[name] = raw
        return json.loads(raw)
    packet = take('induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json', 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6')
    take(Path(packet['input_descriptor']['quantum']['path']).name, packet['input_descriptor']['quantum']['sha256'])
    matter = take(Path(packet['input_descriptor']['matter']['path']).name, packet['input_descriptor']['matter']['sha256'])
    gcert = take('induced_phi_gaussian_action_certificate_receipt.json', '6abd28b7bae5d3e8370ec66555778c0ce6768c21e8ce90674e42c5e214147f8b')
    jcert = take('induced_phi_complex_matter_jet_certificate_receipt.json', '613204e675fe443ca88e17dca1df850d828443d6d879868242ebc97635c62771')
    saved = packet['input_descriptor']['exact_NEW_allq_BO_mean']
    centres = [F(x) for x in saved['centre_exact_rational']]
    data = field.load_mean(g_override=[centres[0], *([centres[1]]*3)], mean_descriptor={'law':'Captured NEW N4 root; no solve or adoption'})
    g = [field.full.enclosure(F(saved['parameters'][k]['lower']), F(saved['parameters'][k]['upper'])) for k in ('g0', 'gs')]
    data['g'] = [g[0], *([g[1]]*3)]
    for k in ('root_receipt', 'full_receipt'):
        b = data['descriptor'][k]
        take(Path(b['path']).name, b['sha256'])
    data['root_descriptor'] = saved
    data['source_descriptor'] = packet['input_descriptor']
    if [p['load'] for p in data['profiles']] != [1, 1, 63]:
        raise ArithmeticError('Actual65 independent species changed')
    return data, matter, gcert, jcert, inputs


def prices(data: dict[str, Any], gcert: dict[str, Any], jcert: dict[str, Any]) -> dict[str, Any]:
    iv = field.full.iv
    iv.dps = 110
    ex, ep, cap = field.full.exact, field.full.endpoint, source.cap
    bounds = gcert['outward_derived_certificate']['bounds']
    eps = ex(upper(bounds['relative_full_form_error']))
    inflation = 1/iv.sqrt(1-eps)
    quantum = json.loads((ROOT/Path(data['source_descriptor']['quantum']['path']).name).read_bytes())
    slowlo = F(quantum['actual_full_field_covariance']['certified_q0_slow_positive_lower']['lower'])
    cs = cap(inflation*iv.sqrt(ex(8))/(2*iv.sqrt(ex(slowlo)))*ex(F(2)**-6144))
    H = [sum(ex(F(int(n),2))*g for n,g in zip(row,data['g'])) for row in gauss.masks(data['geom']['M'])]
    fastmax = F(0)
    for q2 in gauss.QCOUNTS:
        values = [iv.sqrt((ex(F(3,10))+ex(q2)/h)/(h/5))/2 for h in H]
        values += [ex(F(3,20))/iv.sqrt(ex(F(3,50))*h+ex(F(21*q2,200))) for h in H]
        if q2:
            values.append(ex(4)/iv.sqrt(ex(F(q2,5))))
        fastmax = max(fastmax, *(ep(x,1) for x in values))
    cf = cap(inflation*ex(fastmax)*ex(F(2)**-7168))
    M = ex(upper(bounds['fullmatrix_M']))
    p = ex(upper(bounds['corrector_L2']))
    moment = source.Moments(cs, cf)
    s = source.term(ex(1),1)
    z = source.term(ex(1),0,1)
    t = source.add(z, source.scale(s,M))
    r = source.add(s,t)
    bare = source.scale(source.add(source.scale(source.power(s,3),ex(2)), source.scale(source.power(s,4),ex(F(1,64))), source.scale(source.multiply(s,source.power(t,2)),ex(5+64*8192)), source.scale(source.power(t,3),ex(12)), source.power(t,4)), ex(LAMBDA))
    align = source.scale(source.add(source.scale(source.power(t,3),ex(F(3,5))),source.scale(source.power(t,4),ex(F(9,40))),source.scale(source.multiply(source.power(s,2),source.power(t,2)),ex(F(1,80)))),ex(LAMBDA54))
    magnetic = source.scale(source.add(source.scale(source.power(t,3),ex(F(21,5))),source.scale(source.power(t,4),ex(F(7,5)))),ex(LAMBDA54))
    rows = []
    physical_K = [F(0)]*4
    for row in jcert['profiles']:
        Ks = [upper(x) for x in row['matter_K0_to_K3_unscaled']]
        scale = LAMBDA if row['heavy_negative_sheet'] else F(1)
        for j in range(4):
            physical_K[j] += row['load']*scale*Ks[j]
        rows.append({'role':row['role'],'copies':row['load'],'physical_energy_multiplier':str(scale),'K0_to_K3_unscaled':row['matter_K0_to_K3_unscaled'],'physical_K0_to_K3': [field.full.bounds(ex(scale*row['load']*x)) for x in Ks]})
    matter = source.scale(source.power(r,3),ex(physical_K[1]/6))
    polynomials = {'bare_Phi_quartic':bare,'spatial_alignment_after_actual_fast_Hessian_and_linear_s_colour_forcing':align,'ordered_SU4_plaquette_after_actual_fast_Hessian':magnetic,'full65_physical_filled_sea_Taylor':matter}
    values = {name:cap(moment.norm(poly,4)*(1+9*p)) for name,poly in polynomials.items()}
    total = cap(sum(values.values()))
    binary_align = abs(F.from_float(.2)-F(1,5))
    binary_magnetic = abs(F.from_float(.7)-F(7,10))
    # If binary literals were exact physical constants, their quadratic mismatch
    # against the unchanged rational f_A must be paid, not called nonlinear.
    mismatch = source.scale(source.power(t,2),ex(LAMBDA54*(binary_align*F(9,8)+binary_magnetic*3)))
    mismatchnorm = cap(moment.norm(mismatch,4)*(1+9*p))
    law_indices = {'alignment_literal_binary_difference':str(binary_align),'plaquette_literal_binary_difference':str(binary_magnetic),'exact_rational_source_law_binding_required':True,'binary_physical_law_alternative_quadratic_mismatch_upper':field.full.bounds(mismatchnorm),'binary_literal_law_is_adopted':False,'binary_upper_is_not_a_lower_bound_or_failure_proof':True}
    if ep(total,1) >= F(1,1 << 400):
        raise ArithmeticError('Derived actual nonlinear action budget not useful')
    return {'marginal_covariance_operator_uppers':{'uniform45':field.full.bounds(cs),'conditional_fast5715':field.full.bounds(cf),'uniform_standard_deviation_scale_times_actual_coefficient':field.full.bounds(iv.sqrt(cs)*ex(F(2)**3072)),'conditional_shift_M':bounds['fullmatrix_M'],'corrector_L2':bounds['corrector_L2'],'source_marginal_mean_is_C_plus_Ms':True},'polynomial_bounds':{name:[{'slow_degree':a,'conditional_fast_degree':b,'coefficient':field.full.bounds(c)} for (a,b),c in sorted(poly.items())] for name,poly in polynomials.items()},'component_norms_before_normalization':{k:field.full.bounds(v) for k,v in values.items()},'actual_nonlinear_action_norm_before_normalization_upper':field.full.bounds(total),'actual_nonlinear_action_norm_with_proved_normalization_upper2':field.full.bounds(2*total),'normalization_multiplier_is_only_a_bound':True,'physical_full65_matter_K0_to_K3': [field.full.bounds(ex(x)) for x in physical_K],'matter_profiles':rows,'tensor_accounting':{'trace_dimension_paid':256,'occupied_rank_per_copy':128,'independent_charged_loads':[1,63],'normal':5760,'sqrt_fast_normal_HS_bound':76,'sqrt_slow_tensor_factor_bound':7,'Kj_block_HS_bound':'76 K_j','Kj_slow_derivative_tensor_HS_bound':'76*7^j K_j','fully_nonuniform_Hessian_derivative_HS_bound':'76^(j+1) K_j','scalar_radial_Taylor_uses_multilinear_opnorm_not_HS':True,'site_oriented_edges':192,'ordered_faces':192,'faces_per_edge':4},'law_binding_alternative':law_indices,'not_included':['actual full linear root/static-force residual','primitive linear_s fast_colour Hessian forcing and mean trace','kinetic -A dG A corrector action','localGauss/Haar/Noether/bump/Kato/endpoint terms','whole action/source admission']}


def run() -> dict[str, Any]:
    start = time.monotonic()
    mp.mp.dps = 110
    captured = {**field.CAPTURED,**source.SOURCE_BYTES}
    for module in tuple(sys.modules.values()):
        file = getattr(module,'__file__',None)
        if file is not None and Path(file).parent == ROOT and Path(file).suffix == '.py':
            captured[Path(file).name] = Path(file).read_bytes()
    captured[Path(__file__).name] = SELF
    runner = 'induced_phi_primitive_nonlinear_action_certificate_natural_exit.py'
    captured[runner] = (ROOT/runner).read_bytes()
    data, matter, gcert, jcert, inputs = load_inputs()
    if F(jcert['domain']['collar']) != 3*R:
        raise ArithmeticError('Riesz collar mismatch')
    if not sum(F(field.full.bounds(g)['upper']) for g in data['g']) < F(2,3):
        raise ArithmeticError('Actual root frame bound unavailable')
    with threadpool_limits(limits=1):
        finite = controls.run(data)
    weighted = prices(data,gcert,jcert)
    archive = BytesIO()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        for category,collection in (('source',captured),('input',inputs)):
            for name,raw in sorted(collection.items()):
                info = zipfile.ZipInfo(category+'/'+name, date_time=(2026,10,10,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info,raw)
    archive_bytes = archive.getvalue()
    archive_name = 'induced_phi_primitive_nonlinear_action_certificate_immutable_'+digest(archive_bytes)+'.zip'
    (ROOT/archive_name).write_bytes(archive_bytes)
    stable = {name:(ROOT/name).read_bytes()==raw for name,raw in {**captured,**inputs}.items()}
    if not all(stable.values()):
        raise ArithmeticError('Immutable source/input epoch changed')
    return {'status':'DERIVED_ACTUAL_PRIMITIVE_FULLSEA_NONLINEAR_ACTION_CONDITIONAL_ROOT_STATIC_LAW_NOT_SOURCE_ADMISSION','operator_formulas_and_global_derivation':PROOF,'support_radius':str(R),'global_collar_radius':str(3*R),'fixed_source_boundary':'Captured b6(1-P)f_A(root), Xi(root), C0(root), M fullmatrix; no solve_mean/load_quantum_mean/no old producers; no source/parameter/adoption change','derived_outward_weighted_action':weighted,'observed_finite_operator_controls':finite,'remaining_assumptions':['Captured NEW N4 root rectangle exists and is the selected implicit mean; no root solve or existence proof here.','Pinned exact scientific source-law coefficients agree with the declared full microscopic functional (rational1/5 and7/10), not binary-literal physical adoption.','Archived static all65 sea/gap bounds and certified q0 slow positive lower used by the NEW outward Gaussian form comparison.','Parent independently prices actual linear root/mean/static force, localGauss kinetic/source/domain and whole operator action; no numerical oldC is used.'],'inputs':{name:{'sha256':digest(raw),'bytes':len(raw)} for name,raw in inputs.items()},'sources':{name:{'sha256':digest(raw),'bytes':len(raw)} for name,raw in captured.items()},'all_source_and_input_bytes_unchanged':stable,'immutable_capsule':{'path':archive_name,'sha256':digest(archive_bytes),'bytes':len(archive_bytes),'source_and_input_entries':len(captured)+len(inputs)},'own_sha256':digest(SELF),'plan':{'id':PLAN,'assignment_revision':ASSIGNMENT_REVISION,'observed_governing_revision':GOVERNING_REVISION,'template_revision':TEMPLATE_REVISION,'work_items':['joined-explanatory-campaign','induced-photon','verification']},'runtime':{'executable':sys.executable,'python':sys.version,'numpy':np.__version__,'mpmath':mp.__version__,'interval_digits':field.full.iv.dps},'elapsed_seconds':time.monotonic()-start}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'action':result['derived_outward_weighted_action']['actual_nonlinear_action_norm_with_proved_normalization_upper2'],'component_norms':result['derived_outward_weighted_action']['component_norms_before_normalization'],'finite_controls':result['observed_finite_operator_controls']['summary'],'immutable_capsule':result['immutable_capsule'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
