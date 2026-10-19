"""Join frozen longer-clock kernels at physical endpoints and their own initials.

The actual raw source is E0_full T psi, not silently T E0_body psi.
No old numerical producer, root solver, source normalization or clock is changed.
"""
from __future__ import annotations
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
CLOCK = F(1, 65536)
PLAN = 'issue-222-induced-photon-delta13-continuation-20261008'
REVISION = '9ea88239ef1b96d1414978150fd9e2c919e29de24d91834808068dfd13a33340'
PINS = {
    'joined_phi_u1_current_retarded_higher_binding_receipt_20261010.json': '9197f5666f1c5bf2c1de601fc008a913486592467fc3a39c1db076b31f558b92',
    'joined_phi_u1_current_retarded_higher_binding_natural_exit_capsule_20261010.json': 'bc3ab46bd89690d895528c67464b915d0ea8f9d88f7069179d229ca545345367',
    'joined_phi_u1_nonlocal_paired_higher_final_composed_natural_exit_20261010_receipt.json': 'dfe9f6606311d493b7905d96183129ded75b1d4e5b246f357fe9b64b775a3c49',
    'joined_phi_u1_nonlocal_paired_higher_final_composed_natural_exit_20261010.json': '8097094018e5ab2cd643a431ad95cc042d66771b1e9820c0b4b785d80a2a630c',
    'joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json': '2cc91cbe6b8b64f6d270ff04a9b946b22bd3a9a52ff9627e4bf15325f0c0adf4',
    'joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json': '2a1bbb46852707c82e477f28b5497e303b5ca66453db76d0655379f35cfb9a97',
    'joined_phi_u1_nonlocal_body_transport_verified_source_transfer_receipt_20261010.json': '442299e80be00a54ce32cf9482b5c70040398871cddfecb06afe5adfea5fe894',
    'joined_phi_u1_nonlocal_signal_interval_certificate_typed_verified_receipt_20261010.json': '6faff3a4576d6de88617ce08c21321cf8f68761ce63355f85d188e2087ff266b',
    'joined_phi_u1_photon_higher_jet_source_composition_receipt_20261010.json': '968e25dd692c1d92a02a02bf0a3507aa78522c4082eb70118997b95549d79711',
    'joined_phi_u1_photon_higher_jet_source_composition_natural_exit_capsule_20261010.json': '6b0d4457a4557449213a60ea0aef244b8ba275159a64753ccd7a581bd96b513f',
}
PROOF = {
    'source': 'Background is T psi. Declared intermediate raw vector v=E0_body psi has squared norm1/2. Actual physical raw source xi=E0_full T psi=T v+zeta0, ||zeta0||<=eta0. No renormalization. |norm(xi)^2-1/2|<=2eta0+eta0^2. E0 is the local colour/U1 singlet sum c0dag R0/2 with actual Ref, so it commutes field kinetic, primitive electric, total local Gauss charges and full rho Lplus rho. Its commutator K0 is only actual active hPhi versus hroot, not a copied offsite remainder.',
    'initial_graph': 'On the preserved smooth compact-source core, Hs zeta0=Ephi zeta0+(E0_full T-T E0_body)Hb_bg psi+E0_full Rbg psi-Rs v+K0_full T psi-T K0_body psi. The active creation-column norm is at most its onebody operator norm. Tr Phi_mu^2/4 bounds ||Phi_mu||op^2/4, so K0_full T is bounded by2 matter Xsupport and K0_body by2 matter Xroot. Here matter is the inherited positive sum of absolute onebody coefficients over all charged particles, hence overprices the single active column. Thus zeta0 belongs to D(Hs) with the stated explicit graph cap; no global E0/J commutator bound is assumed.',
    'quartic': 'Inherited Hfull+C>=c rho^4 on the global Friedrich form. For z=UF(t)zeta0, ||Jfull z||directsum<=kappa [eta0^2(eta0 h0+C eta0^2)/c]^(1/4), because both Hilbert and H norms are conserved. This is a vector graph bound, not a bounded fullfield current. A rational power-of-two fourth-root upper overprices it.',
    'current': 'For R=t/told>=1, old d,h and quartic current prices scale by at most R. Each target BODY current directsum norm is bounded by the independently derived jbody times norm(v), NOT observed inserted current times1ppm. Add propagated J zeta0 on the left and initial support kappa Xsupport eta0 on the right. Twice the product-expansion bounds the full129 retarded HS discrepancy. Pay its actual own-initial R0 error separately; BODY timeparity never zeroes physical R0.',
    'photon_source_bridge': 'T and E0_full commute with Theta/Pi at the co-local source, and zeta0 is Theta-independent multiplication of the same initial Gaussian. Initial Theta/Pi zeta norms are at most eta0 sqrt258 and eta0 sqrt(129/8). Exact Noether equations dotTheta=128Pi,dotPi=J on the core, extended by graph approximation, give Pi zeta(t)<=initialPi+t Jzeta and Theta zeta(t)<=initialTheta+128t initialPi+64t^2 Jzeta. This avoids the unusable unshifted total-H electric bound. Add vector-product errors to already composed photon kernels. Physical D free baseline is128 norm(xi)^2 t I and Pi baseline norm(xi)^2 I; pay the mass discrepancy, do not label1/2 physical normalization.',
    'nonlocal': 'Use actual signed ROW kernel <Ex UBpsi,US E0psi>. Uniform actual Ex/E0 comparison price is2t epsilon sqrt(1/2)+eta_x sqrt(1/2)+eta0. Physical initial need not vanish; subtract the separate initial endpoint price for change from its own initial. No physical parity, autocorrelation or bare64/128 substitution.',
    'scope': 'Exploratory2^-16, conditional on inherited exact rational currentN4 law/static/source/domain suppliers. Actual unnormalized physical preparation is retained. No originalSOURCE/CURRENT427, clock adoption, interacting ground, stationary spectral pole, massless dispersion, commoncone or Compton admission.',
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def show(x: F) -> str:
    with localcontext() as context:
        context.prec = 25
        return str(Decimal(x.numerator)/Decimal(x.denominator))


def root_upper(x: F, n: int) -> F:
    if x < 0 or n <= 0:
        raise ArithmeticError('Invalid positive-root argument')
    if not x:
        return F(0)
    exponent = (x.numerator.bit_length()-x.denominator.bit_length()+n-1)//n
    result = F(2)**exponent
    while result**n < x:
        result *= 2
    while (result/2)**n >= x:
        result /= 2
    return result


def run(output: Path) -> dict[str, Any]:
    start = time.monotonic()
    sys.set_int_max_str_digits(0)
    own = Path(__file__).read_bytes()
    captured: dict[str, bytes] = {Path(__file__).name: own}
    loaded: dict[str, Any] = {}
    for name, pin in PINS.items():
        raw = (ROOT/name).read_bytes()
        if digest(raw) != pin:
            raise ArithmeticError('Pinned physical composition epoch changed: '+name)
        captured[name] = raw
        loaded[name] = json.loads(raw)
    current, current_caps, nonlocal_body, nonlocal_caps, graph, action, transfer, typed, photon, photon_caps = loaded.values()
    if current_caps['natural_numeric_exit'] != 0 or not current_caps['source_stable'] or not current_caps['receipt_source_matches']:
        raise ArithmeticError('Current canonical binding lacks stable natural0')
    if current_caps['receipt_sha256'] != PINS['joined_phi_u1_current_retarded_higher_binding_receipt_20261010.json']:
        raise ArithmeticError('Current receipt/capsule mismatch')
    if nonlocal_caps['returncode'] != 0 or not nonlocal_caps['natural_numeric_exit'] or not nonlocal_caps['source_unchanged']:
        raise ArithmeticError('Paired kernel lacks stable natural0')
    if nonlocal_caps['receipt']['sha256'] != PINS['joined_phi_u1_nonlocal_paired_higher_final_composed_natural_exit_20261010_receipt.json']:
        raise ArithmeticError('Paired receipt/capsule mismatch')
    if photon_caps['returncode'] != 0 or not photon_caps['natural_exit'] or not photon_caps['source_stable'] or not photon_caps['inputs_stable']:
        raise ArithmeticError('Photon composition lacks stable natural0')
    if photon_caps['receipt_sha256'] != PINS['joined_phi_u1_photon_higher_jet_source_composition_receipt_20261010.json']:
        raise ArithmeticError('Photon receipt/capsule mismatch')
    for capsule, pin in ((current_caps, current_caps['source_sha256_before']), (nonlocal_caps, nonlocal_caps['source_sha256']), (photon_caps, photon_caps['source_sha256'])):
        name = Path(capsule['command'][1]).name
        raw = (ROOT/name).read_bytes()
        if digest(raw) != pin:
            raise ArithmeticError('Frozen producer source binding changed: '+name)
        captured[name] = raw
    for receipt in (current, nonlocal_body, photon):
        if F(receipt['clock_exact']) != CLOCK or receipt['clock_adopted']:
            raise ArithmeticError('Longer clock scope mismatch')
    if not action['all_inputs_and_sources_unchanged'] or not transfer['all_inputs_and_source_unchanged']:
        raise ArithmeticError('Actual compact-source supplier is not stable')
    if not action['action_input']['uniform_body_Hilbert_map']:
        raise ArithmeticError('Uniform source map premise missing')
    epsilon = F(action['action_input']['epsilon']['upper'])
    radius = F(transfer['source_radius_L2_upper_exact'])
    if epsilon != F(transfer['source_action_bound_exact']):
        raise ArithmeticError('Source action epochs differ')
    gc = graph['global_constants']
    if radius != F(gc['prepared_full_radius']['upper']):
        raise ArithmeticError('Prepared radius epochs differ')
    b1 = F(transfer['fundamental_active_W1_upper_exact'])
    eta0 = b1*radius
    eta_mass = 2*eta0+eta0**2
    if not 0 < eta_mass < F(1,2):
        raise ArithmeticError('Physical raw-mass interval or norm-upper1 fails')
    kappa, jbody, eta_j = (F(gc[k]['upper']) for k in ('current_kappa','body_J_operator','endpoint_norm'))
    c = F(gc['graph_quartic']['lower'])
    C, energy = (F(gc[k]['upper']) for k in ('additive_C','Erefabs'))
    xroot, xsupport = F(14), F(14)+F(3,2**20)
    matter = F(typed['conserved_energy_rational_prices']['matter'])
    hb_bg = F(graph['states']['background']['prices']['Hbpsi']['upper'])
    h0 = energy*eta0+eta0*hb_bg+2*epsilon+2*matter*(xsupport+xroot)
    jz = kappa*root_upper(eta0**2*(eta0*h0+C*eta0**2)/c, 4)
    jz_initial = kappa*xsupport*eta0
    theta_z0, pi_z0 = eta0*root_upper(F(258),2), eta0*root_upper(F(129,8),2)
    pi_zt = pi_z0+CLOCK*jz
    theta_zt = theta_z0+128*CLOCK*pi_z0+64*CLOCK**2*jz
    theta0, pi0 = root_upper(F(258),2), root_upper(F(129,8),2)
    q = F(current['Dcar_connected_operator_upper_exact'])
    shadow, old_paid = F(current['Dcar_conservative_nuclear_shadow_upper_exact']), F(current['old_paid_unroot_Dcar_exact'])
    if shadow != 4*q or not 0 < shadow <= old_paid:
        raise ArithmeticError('Global endpoint Dcar rescue fails')
    current_states: dict[str, Any] = {}
    photon_states: dict[str, Any] = {}
    for label, mass in (('background',F(1)),('raw_source',F(1,2))):
        old = graph['states'][label]
        body = current['states'][label]
        if F(body['raw_mass_exact']) != mass or F(old['raw_norm_squared']) != mass:
            raise ArithmeticError('Original prepared-state weights changed')
        ratio = CLOCK/F(old['clock_exact'])
        if ratio < 1:
            raise ArithmeticError('Clock homogeneous comparison invalid')
        left = ratio*F(old['prices']['current_left_error']['upper'])
        right = ratio*F(old['prices']['current_right_error']['upper'])
        own_j = eta_j
        source_bridge = label == 'raw_source'
        if source_bridge:
            left += jz
            right += jz_initial
            own_j += jz_initial
        discrepancy = 2*(left*jbody+jbody*right+left*right)
        own_initial = 2*(2*jbody*own_j+own_j**2)
        lower = F(body['BODY_whole129_HS_lower_exact'])-discrepancy-own_initial
        if lower <= 0:
            raise ArithmeticError('Physical current own-initial signal failed: '+label)
        current_states[label] = {'physical_preparation': 'E0_full T psi' if source_bridge else 'T psi',
            'BODY_norm_squared_exact':str(mass), 'physical_norm_squared_exact':None if source_bridge else '1',
            'physical_norm_squared_interval':{'lower':str(mass-eta_mass),'upper':str(mass+eta_mass)} if source_bridge else {'lower':'1','upper':'1'},
            'whole129_physical_HS_lower_from_own_initial_exact':str(lower), 'lower_display':show(lower),
            'transfer_HS_upper_exact':str(discrepancy), 'own_initial_R0_HS_upper_exact':str(own_initial),
            'current_left_error_exact':str(left), 'current_right_error_exact':str(right),
            'physical_initial_zero_assumed':False}
        bridges = {'D':F(0),'Pi':F(0),'mixed':F(0)}
        if source_bridge:
            # Safe norm upper1 for the intermediate raw BODY vector: no whitening.
            j_phi = jbody+ratio*F(old['prices']['current_left_error']['upper'])
            theta_phi = theta0+128*CLOCK*pi0+64*CLOCK**2*j_phi
            pi_phi = pi0+CLOCK*j_phi
            bridges['D'] = 2*(theta_zt*theta0+theta_phi*theta_z0+theta_zt*theta_z0)+128*CLOCK*eta_mass
            bridges['Pi'] = 2*(pi_zt*theta0+pi_phi*theta_z0+pi_zt*theta_z0)+eta_mass
            bridges['mixed'] = 2*(jz*theta0+j_phi*theta_z0+jz*theta_z0)
        responses = {}
        for kernel, extra in bridges.items():
            old_lower = F(photon['all129_states'][label]['responses'][kernel]['directional_restoring_lower_exact'])
            lower = old_lower-extra
            if lower <= 0:
                raise ArithmeticError('Same physical source photon restoring lower failed: '+label+'/'+kernel)
            responses[kernel] = {'directional_restoring_lower_exact':str(lower),'lower_display':show(lower),
                'actual_source_and_free_mass_bridge_upper_exact':str(extra)}
        photon_states[label] = {'physical_preparation':current_states[label]['physical_preparation'],
            'free_baseline_mass':'norm(E0_full T psi)^2' if source_bridge else '1', 'responses':responses,
            'physical_timeparity_assumed':False,'own_initials':'Theta-Theta=0, Pi-Theta canonical mass, J-Theta=0 on the inherited core'}
    receiver_map = {int(row['path']): row for row in typed['receiver_rows']}
    body_rows = nonlocal_body['receiver_rows']
    if len(receiver_map) != 12 or len(body_rows) != 12 or len({row['path'] for row in body_rows}) != 12:
        raise ArithmeticError('Actual twelve receiver coverage changed')
    nonlocal_rows = []
    for row in body_rows:
        prior = receiver_map[int(row['path'])]
        if (row['ROW'],row['COLUMN'],row['length']) != (prior['row'],prior['col'],prior['length']):
            raise ArithmeticError('ROW/path/source binding changed')
        eta_x = F(prior['eta_receiver_uniform_exact'])
        if F(prior['eta_physical_source_vs_fixed_Tsrc_E0body_exact']) != eta0:
            raise ArithmeticError('Actual physical source endpoint epoch differs')
        initial = eta_x*root_upper(F(1,2),2)+eta0
        error = 2*CLOCK*epsilon*root_upper(F(1,2),2)+initial
        il, iu = (F(row['imaginary_BODY_interval'][side]) for side in ('lower','upper'))
        rl, ru = (F(row['real_BODY_interval'][side]) for side in ('lower','upper'))
        il, iu, rl, ru = il-error, iu+error, rl-error, ru+error
        sign = 1 if il > 0 else -1 if iu < 0 else 0
        signed = il if sign == 1 else -iu if sign == -1 else F(0)
        own_change = signed-initial
        if own_change <= 0:
            raise ArithmeticError('Actual physical paired receiver own-initial sign unresolved')
        nonlocal_rows.append({'path':row['path'],'ROW':row['ROW'],'COLUMN':row['COLUMN'],'length':row['length'],
            'physical_imaginary_interval':{'lower':str(il),'upper':str(iu)},
            'physical_real_interval':{'lower':str(rl),'upper':str(ru)},
            'signed_orientation':sign,'signed_change_from_own_initial_lower_exact':str(own_change),
            'signed_lower_display':show(own_change),'endpoint_both_time_upper_exact':str(error),
            'own_initial_upper_exact':str(initial),'physical_C0_zero_assumed':False})
    # Exact one-colour CAR control, states |00>,|10>,|01>,|11>.
    # E0 maps occupied Ref -> active; total local charge is unchanged.
    nactive, nref = (0,1,0,1), (0,0,1,1)
    correct_hc = tuple(64*(a+r-1)**2 for a,r in zip(nactive,nref,strict=True))
    wrong_hc = tuple(64*a*a for a in nactive)
    correct_comm, wrong_comm = correct_hc[1]-correct_hc[2], wrong_hc[1]-wrong_hc[2]
    if correct_comm != 0 or wrong_comm != 64:
        raise ArithmeticError('Exact CAR/Ref source-domain control failed')
    for name, raw in captured.items():
        if (ROOT/name).read_bytes() != raw:
            raise ArithmeticError('Physical composition source/input mutated: '+name)
    result = {'status':'ACTUAL_PHYSICAL_LONG_CLOCK_CURRENT_PHOTON_AND_SIGNED_PAIRED_KERNEL_COMPOSITION',
        'governing_plan':PLAN,'captured_revision':REVISION,'clock_exact':str(CLOCK),'clock_adopted':False,
        'proof':PROOF,'all129_current_states':current_states,'all129_photon_states':photon_states,
        'all12_nonlocal_receivers':nonlocal_rows,
        'actual_source_bridge':{'eta0_upper_exact':str(eta0),'mass_discrepancy_upper_exact':str(eta_mass),
            'H_zeta0_upper_exact':str(h0),'J_U_zeta0_global_upper_exact':str(jz),
            'initial_J_zeta0_upper_exact':str(jz_initial),'Theta_initial_upper_exact':str(theta_z0),
            'Pi_initial_upper_exact':str(pi_z0),'Theta_at_clock_upper_exact':str(theta_zt),'Pi_at_clock_upper_exact':str(pi_zt),
            'normalization_applied':False},
        'Dcar_global_rescue':{'connected_operator_upper_exact':str(q),'nuclear_shadow_upper_exact':str(shadow),
            'old_paid_upper_exact':str(old_paid),'positive_paid_gap_exact':str(old_paid-shadow),
            'native_endpoint_only_rationale_withdrawn':True,'old_numeric_value_changed':False},
        'exact_CAR_Ref_control':{'active_occupancy':nactive,'Ref_occupancy':nref,'correct_commutator_entry':correct_comm,
            'wrong_omit_Ref_commutator_entry':wrong_comm,'finite_control_scope':'one colour CAR; operator total-charge identity, not a fullN4 state expectation'},
        'input_sha256':PINS,'source_sha256':digest(own),'all_inputs_and_source_stable':True,
        'measured_time_evolution':False,'originalSOURCE_adopted':False,
        'still_open':['stationary interacting ground','photon pole/dispersion','charged dressing','commoncone','Compton'],
        'runtime':{'executable':sys.executable,'python':sys.version},'elapsed_seconds':time.monotonic()-start}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({'status':result['status'],'clock':result['clock_exact'],
        'current_lowers':{k:v['lower_display'] for k,v in result['all129_current_states'].items()},
        'photon_lowers':{k:{q:r['lower_display'] for q,r in v['responses'].items()} for k,v in result['all129_photon_states'].items()},
        'nonlocal_minimum':show(min(F(v['signed_change_from_own_initial_lower_exact']) for v in result['all12_nonlocal_receivers']))},indent=2))
