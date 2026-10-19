"""Reprice archived conditional Phi source; price a NEW U1 coupling extension.

No frozen law/source/covariance is changed. No old solver or giant supplier
receipt is rerun/parsed. The new U1 comparison retains electric and complete
all-charged density/Coulomb terms; its interacting body propagator is NOT the
old free-band reference amplitude.
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
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import induced_phi_nonuniform_fermionic_source_unitary as source
import joined_phi_u1_finite_car as bridge

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'
OLD_SHA = 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6'
JETS = ROOT / 'induced_phi_global_normal_h_jet_control_receipt.json'
JETS_SHA = 'e48a53776184f9eda572d5f06ef061f880cfb13935b41f3a707b25d1994e58b8'
R = F(1, 1 << 20)
sys.set_int_max_str_digits(0)


def upper(x: dict[str, Any]) -> F:
    return F(x['upper'])


def lower(x: dict[str, Any]) -> F:
    return F(x['lower'])


def report(x: F) -> dict[str, Any]:
    value = mp.mpf(x.numerator)/x.denominator
    return {'exact_upper': str(x), 'decimal': mp.nstr(value, 30),
            'log10_upper': mp.nstr(mp.log10(value), 20) if x > 0 else None}


def report_lower(x: F) -> dict[str, Any]:
    record=report(x)
    record['exact_lower']=record.pop('exact_upper')
    record['log10_lower']=record.pop('log10_upper')
    return record


def run() -> dict[str, Any]:
    start = time.monotonic()
    raw, jraw = OLD.read_bytes(), JETS.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=OLD_SHA or hashlib.sha256(jraw).hexdigest()!=JETS_SHA:
        raise ArithmeticError('Required archived/new bound epochs changed')
    old, jets = json.loads(raw), json.loads(jraw)
    p = old['actual_full5760_local960_corrected_quantum_CAR_source_proof']
    mp.mp.dps = 100
    # Directional second-frame data needs a full tensor dimension factor.
    newC1 = upper(jets['safe_full_ambient_Fock_gradient_price'])
    newC2 = 5760*upper(jets['directional_Fock_second_data_NOT_Hessian_norm'])
    oldC1, oldC2 = upper(p['all_CAR_first_upper']), upper(p['all_CAR_second_upper'])
    if min(oldC1, oldC2) <= 0:
        raise ArithmeticError('Positive archived derivative prices required for monotone repricing')
    eta = max(F(1), newC1/oldC1, (newC1/oldC1)**2, newC2/oldC2)
    d = p['actual_corrected_derivatives']
    old_active,new_active = p['actual_full65_occupied_projector_Kato_prices'][1],jets['profiles'][1]
    moment4_shape = upper(p['source_positive_block_variation_upper'])/(4*lower(old_active['all_normal_first_h_derivative_upper'])*lower(d['normalization_norm_upper']))
    # Do not use an upward cap of 1-small as a LOWER normalizer.
    normalizer_lower = 1-24*moment4_shape**2/R**2-4*upper(p['actual_joint_corrector_L2_upper'])
    if normalizer_lower <= F(1,2):
        raise ArithmeticError('Direct rational corrected normalizer not proved positive')
    norm = F(2)  # 1/sqrt(normalizer_lower) < sqrt2 < 2.
    gradient, Hfast, H2 = (2*eta*upper(d[key]) for key in ('gradient', 'Hfast', 'H2'))
    body_radius_L2 = norm*moment4_shape
    prior_precision=source.iv.dps
    source.iv.dps=80
    chosen_z=1-F(2)**-400
    upward_z=source.cap(source.exact(chosen_z))
    wrong_norm=source.endpoint(source.cap(1/source.iv.sqrt(upward_z)),1)
    rejected=chosen_z*wrong_norm**2 < 1
    source.iv.dps=prior_precision
    if not rejected or chosen_z <= F(1,2):
        raise ArithmeticError('Normalizer-direction wrong control not discriminated')
    pars = old['input_descriptor']['exact_NEW_allq_BO_mean']['parameters']
    g0l,g0u,gsl,gsu = (F(pars[k][side]) for k,side in (('g0','lower'),('g0','upper'),('gs','lower'),('gs','upper')))
    metric = source.local.local_gauss_prices(g0l,gsl,g0u,gsu,64,8324,R,gradient,Hfast,H2)
    chi = F(2)**-4096
    Gmax = F(8)+F(3,10)+12/(min(g0l,gsl)/2)
    geometry = chi*metric['total_reference_local_Gauss_kinetic_remainder_divided_chi_upper']
    kato1 = 2*chi*Gmax*newC1*gradient
    kato2 = chi*Gmax*5760*(newC2+newC1**2)/2
    # Do not subtract rounded old components; retaining them is conservative.
    complete = 4*eta*upper(p['physical_reference_energy_residual_upper'])+norm*(geometry+kato1+kato2)
    source_h = 4*upper(new_active['repaired_global_D1'])*body_radius_L2
    endpoint = 16*upper(new_active['repaired_global_D1'])/lower(new_active['supported_distance_to_zero_lower'])*body_radius_L2
    repriced = []
    for row in p['raw_source_times']:
        t = abs(F(row['time']))
        error = F(0) if t==0 else 8*t*complete+4*t*source_h+8*endpoint
        # Saved old modulus-minus-positive-price is also a conservative lower
        # for the unchanged REFERENCE modulus, not the rejected physical bound.
        modulus_lower=lower(row['nonzero_modulus_lower'])-error
        if modulus_lower <= 0:
            raise ArithmeticError('Repriced no-U1 conditional raw modulus not positive')
        repriced.append({'time': row['time'], 'complex_error_upper': report(error),
                         'reference': 'UNCHANGED archived free-band reference, ONLY the no-U1 Phi law',
                         'no_U1_raw_modulus_lower':report_lower(modulus_lower),
                         'no_U1_raw_probability_lower':report_lower(modulus_lower**2),
                         'below_two_to_minus37': error < F(1,1<<37)})
    if not all(row['below_two_to_minus37'] for row in repriced):
        raise ArithmeticError('Corrected conditional source clock budget fails')
    centres = [F(x) for x in old['input_descriptor']['exact_NEW_allq_BO_mean']['centre_exact_rational']]
    field.full.iv.dps = 100
    data = field.load_mean(g_override=[centres[0],centres[1],centres[1],centres[1]],
                           mean_descriptor={'law':'NEW numerical coupling control at saved Phi mean; not adopted U1 root'})
    data['g'] = [field.full.enclosure(g0l,g0u), *[field.full.enclosure(gsl,gsu)]*3]
    ex, iv = field.full.exact, field.full.iv
    charged_h_coefficient = F(0)
    charged_W_coefficient = F(0)
    for profile, jet in zip(data['profiles'],jets['profiles'],strict=True):
        if profile['role'] != jet['role']:
            raise ArithmeticError('Coupling profile epoch mismatch')
        if profile['scale']=='heavy':
            continue
        Hmean = abs(ex(profile['onsite']))*iv.sqrt(data['g'][0])
        for path in profile['paths']:
            Hmean += 2*(abs(ex(path['c0']))*iv.sqrt(data['g'][0])+abs(ex(path['ci']))*iv.sqrt(data['g'][path['axis']+1]))
        Hupper = upper(field.full.bounds(Hmean))
        count = int(jet['fixed_bulk_particle_count'])
        b1 = upper(jet['directional_Kato_first'])
        charged_h_coefficient += count*(upper(jet['repaired_global_D1'])+2*Hupper*b1)
        charged_W_coefficient += count*b1
    # Fixed charge sector: 64 charged species, active129+aux63*128,
    # Ref3 relative to fullRef4 at site0. Total U1 charge zero, rho_x<=129.
    g, V, lapgap, rho_max = F(64),64,F(2),129
    Coulomb_operator = g*V*rho_max**2/lapgap
    coulomb_coefficient = 2*Coulomb_operator*charged_W_coefficient
    # Remove the old occupied charged-sea scalar cancellation before allowing
    # an arbitrary charged state, then price the new all-Theta charged delta.
    coupling_residual = (2*charged_h_coefficient+coulomb_coefficient)*body_radius_L2
    extension_times = [{'time':str(t),'raw_twoleg_comparison_error_upper':report(8*abs(t)*(complete+coupling_residual)+8*endpoint)}
                       for t in (F(0),F(1,4096),F(4096))]
    native,geom = data['native'],data['geom']
    # A legal fixed-N extreme Fock vector, not a model of the chosen source.
    rho = np.full(V,-128.,float)
    rho[:32] = 128.
    rho[32] += 1
    rho[0] -= 1
    if float(rho.sum()) != 0:
        raise ArithmeticError('U1 Gauss accept boundary requires complete total charge')
    energy = float(g)*float(rho@native['Lplus']@rho)
    wrong_one_flavour_bound = float(g*V*3**2/lapgap)
    if not wrong_one_flavour_bound < energy < float(Coulomb_operator):
        raise ArithmeticError('Allcharged Coulomb bound does not discriminate flavour omission')
    # Gaussian density variance s=2 in ALL129 real quotient modes.
    s,M = F(2),129
    electric_mean = g*M/(4*s)
    electric_squared_norm = g*g*M*(M+2)/(16*s*s)
    # Actual nonlinear added coupling at large, uncompactified theta angles.
    rng = np.random.default_rng(2226401)
    theta = rng.normal(0,2.,192)
    _, eigvec = np.linalg.eigh(native['frame'][0]); v=eigvec[:,-1]
    X=(np.outer(v,v.conj())-np.eye(4)/4)/np.sqrt(1.5)
    coeff=2*np.einsum('aij,ji->a',geom['deltas'][:,0],X).real
    delta=np.einsum('a,amij->mij',coeff,geom['deltas'])
    loc=-np.ones(64)/64;loc[0]+=1;loc/=np.linalg.norm(loc)
    fields=np.broadcast_to(native['frame'],(64,4,4,4)).copy()+float(R/2)*loc[:,None,None,None]*delta[None]
    C=rng.normal(size=(192,15));C*=float(R/2)/np.linalg.norm(C)
    T=geom['M']/np.sqrt(8)
    links=np.asarray([expm(1j*x) for x in np.einsum('ea,aij->eij',C,T)])
    identity=np.broadcast_to(np.eye(4),(192,4,4)).copy()
    mean_fields=np.broadcast_to(native['frame'],(64,4,4,4)).copy()
    radius=float(R)/np.sqrt(2)
    controls=[]
    for profile,jet in zip(data['profiles'][1:],jets['profiles'][1:],strict=True):
        actual=bridge.assemble(profile,native,fields,links,theta)
        frozen=bridge.assemble(profile,native,mean_fields,identity,theta)
        observed=float(np.max(abs(np.linalg.eigvalsh(actual-frozen))))
        bound=float(upper(jet['repaired_global_D1']))*radius
        if not 1e-7 < observed < bound:
            raise ArithmeticError('Actual Phi/U1 coupling or repaired uniform bound not exercised')
        controls.append({'role':profile['role'],'actual_nonlinear_difference_norm':observed,
                         'repaired_upper':bound,'wrong_frozen_Phi_colour_difference':observed})
    captured={**field.CAPTURED,**source.SOURCE_BYTES,**bridge.SOURCE_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
    epochs={name:{'sha256':hashlib.sha256(b).hexdigest(),'unchanged':b==(ROOT/name).read_bytes()} for name,b in captured.items()}
    if not all(row['unchanged'] for row in epochs.values()) or OLD.read_bytes()!=raw or JETS.read_bytes()!=jraw:
        raise ArithmeticError('Execution input/source changed')
    return {'governing_plan':'issue-222-induced-photon-delta13-continuation-20261008',
            'governing_revision':'f6e9aaf487b4ce8899afda2bc1d431b9aad7e560a416b97e0038bb42b6395d60',
            'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
            'inputs':{'archived_source':{'path':str(OLD),'sha256':OLD_SHA},'corrected_jets':{'path':str(JETS),'sha256':JETS_SHA}},
            'derivation':{'monotone_reprice':'All C1/C2 occurrences in gaussian_weighted_bounds and corrected_derivatives have nonnegative coefficients and degree<=2. Eta>=1,C1new/C1old,(C1new/C1old)^2,C2new/C2old. Direct normalizer_lower>1/2 permits norm<sqrt2<2; use2eta on old gradient/Hfast/H2 and4eta on old total core. C2new includes5760 tensor factor; trace retains5760 and first-square terms. Recompute localGauss, Kato1/Kato2 and endpoints; retain rather than subtract rounded old components.',
                          'moment_recovery':'Archived source_h is a positive upward scalar computation 4D1old*chosenRadiusL4*(1+9p)*computedNorm. Divide by positive recorded D1old_lower/computedNorm_lower to recover the chosen moment/corrector price; this is NOT inversion of an invalid physical inequality. RadiusL4 bounds sqrt(actual Gaussian trace); exact rational 1-24moment4_shape^2/R^2-4p>1/2 supplies safe normalization2 independently of upward-capped1. Gaussian/Holder/corrector and core descriptor assumptions remain conditional.',
                          'new_law':'Hjoin=H_Phi_SU4 + H_E,U1 + g rho_allcharged Lplus rho_allcharged + dGamma_charged[h(Phi,C,Theta)-h(Phi,C,0)]. Real unwrapped coexact+harmonic Theta, g=UE/a=64; no magnetic B2 imported, no source/parameter/covariance adopted.',
                          'body_comparator':'Same compact corrected Phi packet and localSU4 reconstruction; neutral negative mean band remains unchanged. ChargedTheta dynamics may be ANY fixed8193-particle state, with Ref3 at0 and zero totalU1charge. Hbody adds actual full h(meanPhi,Theta), transverse/harmonic Pi and complete Coulomb. Wphi is the occupied Kato frame atTheta0, independentTheta, hence [H_E,U1,Wphi]=0. This is NOT theta-following Slater Wick for an interacting state.',
                          'uniform_coupling':'Each charged onebody delta <=(D1+2Hmean b1)r, lifted by actual per-species fixed particle counts. Price TWICE: first remove the old occupied charged-sea scalar cancellation from the Phi core, then add the actual arbitrary-state/allTheta delta. ||GammaW-I||<=sum N b1 r. rho_x operatornorm<=129, Lplusnorm<=1/2, giving HC<=34080768. Conjugation error <=2HC sum N b1 r. Bounds uniform in all realTheta by unit-modulus scalar phases. Full corrected Phi/Noether/core descriptor hypotheses still required.'},
            'corrected_inputs':{'C1':report(newC1),'C2_total_tensor':report(newC2),'positive_monomial_multiplier_eta':report(eta),
                                'safe_gradient':report(gradient),'safe_Hfast':report(Hfast),'safe_H2':report(H2)},
            'normalizer_repair':{'exact_lower':str(normalizer_lower),'above_one_half':normalizer_lower>F(1,2),
                                 'normalization_norm_upper':str(norm),'wrong_control':{'exact_chosen_lower':str(chosen_z),
                                 'upward_capped_lower':str(source.endpoint(upward_z,1)),'wrong_norm_upper':str(wrong_norm),
                                 'exact_squared_inequality_rejects_wrong_upper':rejected,
                                 'limit':'Bound-direction control, not a claim about the exact physical packet norm or a rerun of old photon precision.'}},
            'repriced_no_U1_Phi_source':{'reference_residual_upper':report(complete),'prepared_Rayleigh_upper':report(2*complete),
                                       'source_h_upper':report(source_h),'endpoint_upper':report(endpoint),'clock_rows':repriced,
                                       'scope':'RESTORED conditional arithmetic with corrected global derivatives; NOT independent global source certification'},
            'new_U1_extension':{'weighted_body_radius_L2':report(body_radius_L2),'charged_h_coefficient':report(charged_h_coefficient),
                                'charged_W_coefficient':report(charged_W_coefficient),'Coulomb_operator_norm_upper':report(Coulomb_operator),
                                'coupling_residual_upper':report(coupling_residual),'clock_rows':extension_times,
                                'Coulomb_Fock_control':{'complete_zero_charge':float(rho.sum()),'actual_extreme_Fock_energy':energy,'wrong_one_flavour_upper':wrong_one_flavour_bound},
                                'Gaussian_U1_Pi_control':{'all_real_modes':M,'density_variance':str(s),'exact_mean_HE':str(electric_mean),'exact_norm_HEpsi_squared':str(electric_squared_norm)},
                                'actual_nonlinear_Phi_U1_controls':controls,'Theta_max_abs':float(np.max(abs(theta))),
                                'admission':'Conditional uniform-coupling comparison price ONLY. Full interacting U1 body propagator/raw probability/current must still be evaluated; old free-band amplitude/probability NOT transferred.'},
            'source_at_execution':epochs,'own_source_utf8':Path(__file__).read_text(),
            'runtime':{'executable':sys.executable,'python':sys.version},'elapsed_seconds':time.monotonic()-start}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1): result=run()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'eta':result['corrected_inputs']['positive_monomial_multiplier_eta']['decimal'],
                      'repriced_Rayleigh':result['repriced_no_U1_Phi_source']['prepared_Rayleigh_upper']['decimal'],
                      'repriced_clocks':[{k:(v['decimal'] if isinstance(v,dict) and 'decimal' in v else v) for k,v in row.items()} for row in result['repriced_no_U1_Phi_source']['clock_rows']],
                      'U1_coupling_residual':result['new_U1_extension']['coupling_residual_upper']['decimal'],
                      'new_U1_clocks':[{k:(v['decimal'] if isinstance(v,dict) and 'decimal' in v else v) for k,v in row.items()} for row in result['new_U1_extension']['clock_rows']],
                      'Coulomb_control':result['new_U1_extension']['Coulomb_Fock_control'],
                      'Pi_control':result['new_U1_extension']['Gaussian_U1_Pi_control'],
                      'nonlinear_controls':result['new_U1_extension']['actual_nonlinear_Phi_U1_controls'],
                      'elapsed_seconds':result['elapsed_seconds']},indent=2))
