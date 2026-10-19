"""NEW2^-16 all129 mixed/photon moment closure, unchanged declared BODY.

A failed price is a measured construction failure, not a physical no-go.
No future Slater assumption, source adoption, or clock replacement.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
CLOCK = F(1, 2**16)
REVISION = '9a2fd8299a741b12cbc9effe2c6a264ce87e2deef1bf73787109d7604849ce49'
PINS = {
    'joined_phi_u1_local_electric_moments_receipt.json': '4c8d5412cf1cfed963756791655f3c90e3da90ce5e9799c0e5ffc100dfcd6bc6',
    'joined_phi_u1_parity_density_closure_selected_20261010_receipt.json': 'b34dba51c23f5e51554e133e2560e9983136bd4d7d237741d19b235bd31bf2a0',
    'joined_phi_u1_photon_causal_response_verified_20261010_receipt.json': 'c7c3ec787e34d5aa58c806650db7603e9f06c92d388cf590672d3644efbe262f',
    'joined_phi_u1_electric_locality_response_receipt.json': '7e27f3a5bc8dde8c1a02efc8ae34dbe80496771833868e21f396a302a6aa2bbd',
}
PROOF = {
    'boundary': 'Same fullN4 BODY, real129/alltorons, HE=HC64, AP, allsigned1/2links and64charged copies. Initial Slater-Gauss moments only; full-H future moment hierarchy. OriginalSOURCE/CURRENT427/oldclocks unchanged; NEW2^-16 probe is not adoption.',
    'parity': 'The supplied fullBODY and original source-mixture antiunitary theorem implies M(t)=i<[J(t),Theta]> odd, D odd and Pi response even. M(0)=0, Mprime(0)=-128<C0>, Msecond(0)=0, with no stationary premise.',
    'normal_order': 'A=i^3 ad_H^3 J is Hermitian. Every connected term is right-normal-ordered as a CAR product of m native signed hops times a polynomial of degree d in mutually commuting native s_r. Three commutators from J: electric derivative128m to degree d+1; electric self128m^2 and onsite2mK to degree d; same-species CAR mDcar to m+1; polynomial finite differences binom(d,l)L_l to m+1,d-l. Root-priced J initial coefficient aJ; Dcar deliberately uses the larger supplied unrooted valid envelope. Spectral Holder gives product degree d norm<=maximum native a_d on full-H evolved vectors.',
    'canonical_reduction': 'At future state psi_tau=U(tau)psi, rewrite [A(tau),Theta] as [A,Theta(-tau)]. Exact Theta(-tau)=Theta-128tauPi+128 integral_0^tau(tau-u)J(-u)du. For a right-normal word, [A,Theta] differentiates its polynomial: at most d*qmax and degree d-1. [A,Pi] differentiates the multiplier: at most m*qmax and degree d. The integral commutator is paid by 2||Apsi_tau||HS||J(-u)psi_tau||HS<=2Aprice*aJ*n; it is not discarded or called a local-current exact identity.',
    'third_price': 'B_M3=n sum c*aJ*d*qmax*a_(d-1)+128t*n sum c*aJ*m*qmax*a_d+128t^2*aJ*n sum c*aJ*a_d, uniformly |tau|<=t. Coefficient qmax127/100 dominates exact129/80 square root. This is a positive envelope, not a numerical estimate of a derivative.',
    'integration': 'Taylor with exact absent even coefficient gives M remainder B_M3*t^3/6. D=128t massI+128 integral(t-s)M(s)ds, hence cubic=-8192<C0>t^3/3 and remainder128B_M3*t^5/120. Pi=Dprime/128 has quadratic=-64<C0>t^2 and remainder B_M3*t^4/24. Retain earlier outward whole-contact entry radius and actual physicalTheta/current graph transfer at this SAME new probe. No observed contact eigenvalue is an admitted lower.',
    'limits': 'Initial contact arithmetic remains the supplied outward ball conditional on exact-root/Clifford/domain/action suppliers. Prices are exact rational compositions; matrices are analytic polynomials plus error balls, NOT measured evolution. A nonpositive margin is an unresolved norm closure. No pole, ground, original source, commoncone or Compton admission.',
}


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    inputs: dict[str, bytes] = {}
    data: dict[str, Any] = {}
    for name, pin in PINS.items():
        raw = (ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ArithmeticError('Input epoch changed '+name)
        inputs[name] = raw
        data[name] = json.loads(raw)
    moments = data['joined_phi_u1_local_electric_moments_receipt.json']
    parity = data['joined_phi_u1_parity_density_closure_selected_20261010_receipt.json']
    photon = data['joined_phi_u1_photon_causal_response_verified_20261010_receipt.json']
    locality = data['joined_phi_u1_electric_locality_response_receipt.json']
    if moments['clock_exact'] != str(CLOCK) or parity['clock_exact'] != str(CLOCK):
        raise ArithmeticError('Moment or symmetry epoch clock differs')
    if parity['clock_adopted']:
        raise ArithmeticError('Exploration cannot adopt a clock')
    force = [max(F(moments['price_catalogue'][i]['force_Lp_upper'][k])
                 for i in set(moments['native_signed_path_price_ids'])) for k in range(4)]
    dcar = F(parity['Dcar_exact'])
    onsite = F(643170321783849, 17592186044416)
    qmax = F(127,100)
    if qmax*qmax < F(129,80):
        raise ArithmeticError('Unsafe physical q bound')
    a_j = F(moments['paid_all129_current_contact_insertions']['states_by_insertion']['J']['background']['initial_Bpsi_a_upper'][0])
    terms: dict[tuple[int,int], F] = {(1,0): F(1)}
    for _ in range(3):
        out: dict[tuple[int,int], F] = defaultdict(F)
        for (m,d), coefficient in terms.items():
            out[m,d+1] += coefficient*128*m
            out[m,d] += coefficient*(128*m*m+2*m*onsite)
            out[m+1,d] += coefficient*m*dcar
            for degree in range(1,d+1):
                import math
                out[m+1,d-degree] += coefficient*math.comb(d,degree)*force[degree]
        terms = dict(out)
    states: dict[str, Any] = {}
    for label, mass in (('background',F(1)),('raw_source',F(1,2))):
        a = list(map(F,moments['maximum_native_one_two_moments'][label]['horizon_a_upper']))
        n = a[0]
        if n*n < mass:
            raise ArithmeticError('Moment norm upper is below actual raw norm')
        norm_a3 = sum((c*a_j*a[d] for (_,d),c in terms.items()),F(0))
        comm_theta = n*sum((c*a_j*d*qmax*a[d-1] for (_,d),c in terms.items() if d),F(0))
        comm_pi = n*sum((c*a_j*m*qmax*a[d] for (m,d),c in terms.items()),F(0))
        integral = 128*CLOCK**2*a_j*n*norm_a3
        third = comm_theta+128*CLOCK*comm_pi+integral
        remainders = {'mixed':third*CLOCK**3/6,'D':128*third*CLOCK**5/120,'Pi':third*CLOCK**4/24}
        lower = F(photon['all129_states'][label]['contact_minimum_directional_outward_lower']['lower'])
        radius = F(locality['all129_states'][label]['initial_projected192_entry_radius'])*192
        signals = {'mixed':128*CLOCK*lower,'D':F(8192,3)*CLOCK**3*lower,'Pi':64*CLOCK**2*lower}
        transfer = locality['all129_states'][label]['physical_transfer']
        errors = {'mixed':F(transfer['mixed_transfer']['upper'])+128*CLOCK*radius,
                  'D':F(transfer['D_transfer']['upper'])+F(8192,3)*CLOCK**3*radius,
                  'Pi':F(transfer['Pi_transfer']['upper'])+64*CLOCK**2*radius}
        margins = {name:signals[name]-remainders[name]-errors[name] for name in signals}
        states[label] = {'raw_mass_exact':str(mass),'moment_norm_upper_exact':str(n),
            'adH3_J_norm_upper_exact':str(norm_a3),'canonical_comm_Theta_upper_exact':str(comm_theta),
            'canonical_comm_Pi_upper_exact':str(comm_pi),'unequal_current_integral_upper_exact':str(integral),
            'mixed_third_derivative_upper_exact':str(third),
            'responses':{name:{'minimum_directional_signal_lower_exact':str(signals[name]),
                'body_remainder_upper_exact':str(remainders[name]),'coefficient_plus_physical_error_upper_exact':str(errors[name]),
                'signed_margin_lower_exact':str(margins[name]),'error_to_signal_display':float((remainders[name]+errors[name])/signals[name]),
                'resolved':margins[name]>0} for name in signals}}
    source = Path(__file__).read_bytes()
    if any((ROOT/name).read_bytes()!=raw for name,raw in inputs.items()):
        raise ArithmeticError('Inputs changed during composition')
    result = {'status':'ALL129_PARITY_LOCAL_MOMENT_SIGNED' if all(r['resolved'] for s in states.values() for r in s['responses'].values()) else 'ALL129_PARITY_LOCAL_MOMENT_ATTEMPT_NOT_CONTROLLED',
        'governing_plan':'issue-222-induced-photon-delta13-continuation-20261008','captured_revision':REVISION,
        'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        'clock_exact':str(CLOCK),'clock_adopted':False,'proof':PROOF,
        'right_normal_third_terms_exact':{f'{m},{d}':str(c) for (m,d),c in sorted(terms.items())},
        'J_price_upper_exact':str(a_j),'qmax_upper_exact':str(qmax),'Dcar_upper_exact':str(dcar),
        'onsite_upper_exact':str(onsite),'native_force_Lp_upper_exact':list(map(str,force)),
        'all129_states':states,'source_sha256':hashlib.sha256(source).hexdigest(),
        'input_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in inputs.items()},
        'runtime':{'executable':sys.executable,'python':sys.version},'elapsed_seconds':time.monotonic()-started}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    receipt = run(args.output)
    print(json.dumps({'status':receipt['status'],'ratios':{label:{name:row['error_to_signal_display'] for name,row in state['responses'].items()} for label,state in receipt['all129_states'].items()}}))
