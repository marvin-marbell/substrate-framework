"""Actual conditional fixed compact-Phi/body intertwiner action, not original SOURCE."""
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
import sympy as sp
import induced_phi_nonuniform_gauss_quantum as gauss

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).read_bytes()
sys.set_int_max_str_digits(0)
PINNED = {'induced_phi_gaussian_action_certificate_receipt.json': '6abd28b7bae5d3e8370ec66555778c0ce6768c21e8ce90674e42c5e214147f8b', 'induced_phi_primitive_nonlinear_action_certificate_receipt.json': 'a198a2290ef9347d119a833bb1f2f48d3bb37a48efc6dc88ae49dbfbfcf93e1d', 'induced_phi_local_gauss_action_certificate_receipt.json': '71834b782b1d0a73a5406179a186be49618a57b5dc5aeda03c2c5d63032fb3fa', 'induced_phi_fixed_gaussian_hermite_action_closure_receipt.json': '12ef587586c7e152d404ac82ad45818837459c20bdb7e83490f83ea2fbdb39fb', 'induced_phi_complex_matter_force_source_variance_repriced_20261010.json': '78dce47f73d422173060fd10d633996e81525569e407fc34c462acdcb51ff5d2', 'induced_phi_global_normal_source_reprice_receipt.json': '14f50709506112ee0f9957d59d1d0661e66133cbdf27665c324095ba6359581a', 'induced_phi_global_occupied_kato_derivative_control_receipt.json': '270cdefd1b7bd9d68fcf94388ac60aee3c4a8c7d8ad75de099646c72bbb4714f', 'induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json': 'f6a6b6d79496e4e24d0a5df4762a9098b0db24debe16c2f82c32572696592d2f', 'induced_phi_global_normal_h_jet_control_receipt.json': 'e48a53776184f9eda572d5f06ef061f880cfb13935b41f3a707b25d1994e58b8'}
PROOF = {
 'model': 'NEW finiteN4 CartPhi3840/SU4link192/GaussreducedTheta129, Lambda2^8192, chi2^-4096, exact scientific1/5alignment7/10plaquette. No EM B2. CurrentN4root/static/q0 suppliers remain conditional. OriginalSOURCE/CURRENT427 frozen.',
 'T': 'T maps ONLY bodyTheta/fixedN chargedFock to fullfields via actual normalizedscalar b6(1-P)f_A(root), fullHaarGamma(U), actual Theta-independent realGamma(W(r)), and neutralfilledupper-hSlater forphysical -Lambda h. Scalarweightnorm is1 by its actual implicit Z; normalization upper2 is only a proof bound. UnitaryGamma/Haar and neutralSlater make T an isometry for any bodyvector, not an operator on arbitrary normal-coordinate wavefunctions.',
 'reference_split': 'R=Hfull T-T(Hbody+Ephi), Ephi is scalar full65 static+bosonicGaussian reference MINUS chargedsea ROOT constant already inHbody. Auxiliary63 andsourceactive129/Ref3 included. Removing chargedPhi scalar variation and comparing actualcharged hPhiTheta to hrootTheta cost TWO chargedh prices. HCcommGammaW explicitly priced. HEcommutesT becauseW andscalarpacket do not dependTheta. LocalGamma(U) commutesrho and covariance removesU uniformly, no Wlocalrho preservation assumed.',
 'scalar_action': 'Primitive quartic/alignment/orderedplaquette/full65scalar Taylor remainder, fixed full5760 Gaussian/Hermite cancellation, fullprecisionLoewner/mixedM score residual, forcevariance/sourcecorrector/normalizer correction and flatb6commutator are added once as PREPARED norms. No generic comparison coefficient or unsupported oldD1 residual reused. Scalarrootforce cancellation uses the archived exact currentN4 mean equation; residual pressure differences and saved root-gradient shift are explicit inputs, not newly assertedrootexistence.',
 'Kato': 'Apply exact UNCONJUGATED kinetic to W(r)v(r), avoiding accidental omission/doublecount of derivatives of Wdag Q W. Local prices normal/cross/angular/VJ acting on scalarv with arbitraryunitfibre; ||Qvector||<2N forN8324 includingRef. Extra normal terms are chi A DW Dv, chi divA DW v/2, chi A:D2W v/2; extra cross term is chi K Q DW v. C1 is full5760 Fockgradient OPvectorbound; C2data is fulltensor sum singleparticle seconddata, and C1^2 prices all crossparticle products. ||A||F<=76m0, ||divA||<=5760a1. Therefore extra norm<=chi[m0 C1 preparedGrad+(5760a1 C1+76m0(C2data+C1^2))/2+m0(2N)C1]. LocalQterms already include Q acting on Wv; only QDW addition remains. RealWbounds valid on3R, no quantitativecomplexW needed.',
 'radius': 'ForfulljointC=.5A^-1, ||r f||2=sqrtTrC. Holder/hypercontractivitydegree3 give ||r P f||2<=3^(7/4)p sqrtTrC<9p sqrtTrC. With b6<=1 and normalization<=2, preparedradius<=2sqrtTrC(1+9p). No unitvariance or old unsupportedmoment inversion assumed. This is a bound, not an actual source normalization.',
 'uniformity': 'Allscalar/Kato/Noether bounds uniform anyunitfixedparticleFockfibre; allcharged delta h/HC bounds uniform EVERY realTheta. Fubini makes R bounded FROM bodyL2Theta/Fock by the statedepsilon. It is NOT a globallybounded fullfieldR_B, and no post-evolution tube support or operatornormJ follows. Higherdomain proof supplied separately onGLOBALFriedrichoperator, not a hardwallchart.',
 'limits': 'ExactN4root existence/selection, staticall65 gap/slowpositivefullpencil and declared exact rational source-law agreement remain assumptions. The action/supplier derivations and GLOBALdomain are conditional mathematical statements, finitecontrols/floatingdisplays distinct. No originalSOURCE/globalground/photonpole/commoncone/current-transfer admission here.'
}
def upper(record: dict[str, Any]) -> F:
    return F(record['upper'] if 'upper' in record else record['exact_upper'])

def crossparticle_control() -> dict[str, Any]:
    x = sp.Matrix([[0,1],[1,0]])
    i = sp.eye(2)
    gen = sp.kronecker_product(x,i)+sp.kronecker_product(i,x)
    actual = sp.Matrix(-gen*gen)
    wrong = -2*sp.eye(4)
    eigen = max(abs(e) for e in actual.eigenvals())
    defect = max(abs(e) for e in sp.Matrix(actual-wrong).eigenvals())
    assert eigen==4 and defect==2 and eigen>2
    return {'two_distinct_unit_particle_radial_Kato_rotations':'exp(i r sigma_x) tensor exp(i r sigma_x), projectorP0 alongsigma_z; generator offdiagonal soactualradialKato', 'actual_second_operator_norm_exact':'4','wrong_omit_crossparticle_upper':'2','missing_cross_operator_norm_exact':'2','wrong_bound_rejected':True,'scope':'Exact finite CAR/exterior-product derivative invariant, not N4 measuredsource or new law.'}

def run(output: Path) -> dict[str, Any]:
    start=time.monotonic()
    raws={name:(ROOT/name).read_bytes() for name in PINNED}
    assert all(hashlib.sha256(raws[name]).hexdigest()==p for name,p in PINNED.items())
    data={name:json.loads(raw) for name,raw in raws.items()}
    byprefix=lambda p: next(v for k,v in data.items() if k.startswith(p))
    g=byprefix('induced_phi_gaussian_action_certificate_receipt')
    primitive=byprefix('induced_phi_primitive_nonlinear_action_certificate_receipt')
    local=byprefix('induced_phi_local_gauss_action_certificate_receipt')['actual_derived_prices']
    hermite=byprefix('induced_phi_fixed_gaussian_hermite_action_closure_receipt')['outward_action']
    force=byprefix('induced_phi_complex_matter_force_source_variance_repriced_20261010')
    reprice=byprefix('induced_phi_global_normal_source_reprice_receipt')
    domain=byprefix('induced_phi_compact_source_global_domain_control_corrected_receipt')
    iv=gauss.iv; iv.dps=110
    ex,ep,pack=gauss.exact,gauss.endpoint,gauss.bounds
    gb=g['outward_derived_certificate']['bounds']
    radius=2*iv.sqrt(ex(upper(gb['full_trace_covariance'])))*(1+9*ex(upper(gb['corrector_L2'])))
    constants=local['cases']['support']['operator_constants_exact']
    m0,a1=(F(constants[k]) for k in ('cometric','cometric_D1'))
    grad=ex(upper(local['weighted_compact_scalar_packet']['gradient_upper']))
    c1=F(reprice['corrected_inputs']['C1']['exact_upper'])
    c2=F(reprice['corrected_inputs']['C2_total_tensor']['exact_upper'])
    chi=F(2)**-4096; n=8324
    kato=ex(chi)*(ex(m0*c1)*grad+ex(F(5760)*a1*c1+76*m0*(c2+c1*c1))/2+ex(m0*2*n*c1))
    coupling=reprice['new_U1_extension']
    hc=F(coupling['Coulomb_operator_norm_upper']['exact_upper'])
    hcoef=F(coupling['charged_h_coefficient']['exact_upper'])
    wcoef=F(coupling['charged_W_coefficient']['exact_upper'])
    uniform=(2*ex(hcoef)+2*ex(hc*wcoef))*radius
    mean=ex(upper(force['actual_action_input_budgets']['mean_shift_omitted_coefficients']))*radius
    comps={
      'actual_primitive_nonlinear':ex(upper(primitive['derived_outward_weighted_action']['actual_nonlinear_action_norm_with_proved_normalization_upper2'])),
      'actual_full_Haar_metric_Noether':ex(upper(local['cases']['support']['geometric_remainder_norm_upper'])),
      'fixed_full_Gaussian_Hermite':ex(upper(hermite['action_PREPARED'])),
      'root_fullPH_force_difference':ex(upper(force['actual_action_input_budgets']['prepared_total_force_action_upper'])),
      'actual_real_occupied_Kato':kato,
      'reference_b6_commutator':ex(upper(local['separate_reference_b6_commutator_norm_upper'])),
      'charged_sea_removal_allTheta_and_HC':uniform,
      'archived_root_gradient_shift_action':mean}
    epsilon=sum(comps.values())
    if ep(epsilon,1)>=F(2)**-400:
        raise ArithmeticError('Actualsource actionbound fails statedusefulclockguard; no parameter change allowed')
    assert domain['input_and_source_unchanged'] and domain['symbolic_boundary_adverse_control']['6']['accepted_by_boundary_assertion']
    captures=dict(gauss.SOURCE_BYTES); captures[Path(__file__).name]=SELF
    assert all((ROOT/name).read_bytes()==raw for name,raw in raws.items())
    assert all((ROOT/name).read_bytes()==raw for name,raw in captures.items())
    result={'kind':'CONDITIONAL_ACTUAL_COMPACT_PHI_BODY_UNIFORM_SOURCE_INTERTWINER_ACTION',
      'proof':PROOF,'action_input':{'epsilon':pack(epsilon),'uniform_body_Hilbert_map':True,'not_globally_bounded_fullfield_RB':True},
      'residual_components_PREPARED':{k:pack(v) for k,v in comps.items()},
      'reference_split':{'charged_root_sea_constant_removed':True,'W_Theta_independent':True,'source_maps_fixed_body_Hilbert_not_fullfield_RB':True,'isometry_normalization':'actualPhi scalar Z implicit; upper2 only','energy_reference':'full65static+bosonicGaussian reference minus chargedsea rootconstant','no_EM_B2_added':True},
      'source_weights':{'prepared_radius_L2':pack(radius),'prepared_gradient':pack(grad),'C1_full_Fock_gradient':str(c1),'C2_single_particle_tensor_data':str(c2),'C2_full_Fock_with_cross_upper':str(c2+c1*c1),'Qnorm_upper':str(2*n),'cometric_upper':str(m0),'cometric_D1_upper':str(a1),'normal_dimensions':5760,'Phi_cart_dimensions':3840,'Haar_dimensions':960,'body_Theta_dimensions':129,'fixed_total_particle_number':n},
      'crossparticle_adverse_control':crossparticle_control(),'input_sha256':PINNED,
      'domain_receipt_sha256':PINNED[next(k for k in PINNED if k.startswith('induced_phi_compact_source_global_domain_control_corrected_receipt'))],
      'all_inputs_and_sources_unchanged':True,'own_sha256':hashlib.sha256(SELF).hexdigest(),
      'source_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in captures.items()},
      'plan':{'id':'issue-222-induced-photon-delta13-continuation-20261008','execution_revision':'14e500c5d51931331025515a442791f07cc0af03c15e7c178384b7c133767175','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'},
      'runtime':{'executable':sys.executable,'python':sys.version,'interval_digits':iv.dps},'elapsed_seconds':time.monotonic()-start}
    archive=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in {**raws,**captures}.items():z.writestr(name,raw)
    result['immutable_inputs']={'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=run(args.output)
    print(json.dumps({'epsilon':result['action_input']['epsilon'],'components_PREPARED':result['residual_components_PREPARED'],'source_weights':{'radius':result['source_weights']['prepared_radius_L2']},'control':result['crossparticle_adverse_control'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
