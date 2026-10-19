"""Exact variable-Phi native CAR loop diagnostics and full-operator form controls.
No old consumer, source solver, numerical ground, or frozen frame is executed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys
from typing import Any, cast
import sympy as sp

ROOT = Path(__file__).resolve().parent
ZERO = (F(0), F(0))
ONE = (F(1), F(0))


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def mul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def scale(a, b):
    return a[0]*b, a[1]*b


def gaussian(z):
    z = sp.expand(z)
    return F(str(sp.re(z))), F(str(sp.im(z)))


def car(mask, create, annihilate, wrong=False):
    if not (mask >> annihilate) & 1:
        return None
    sign = (-1) ** ((mask & ((1 << annihilate)-1)).bit_count())
    mask ^= 1 << annihilate
    if (mask >> create) & 1:
        return None
    sign *= (-1) ** ((mask & ((1 << create)-1)).bit_count())
    return mask | (1 << create), 1 if wrong else sign


def norm_bound(matrix):
    return sum((abs(z[0])+abs(z[1]) for row in matrix for z in row), F(0))


def encoded(z):
    return {'real': str(z[0]), 'imag': str(z[1])}


def fields_and_links():
    p0 = sp.diag(-1, -1, 1, 1)
    a = sp.Matrix([[0,0,1,0],[0,0,0,2],[1,0,0,0],[0,2,0,0]])
    b = sp.Matrix([[0,0,-sp.I,0],[0,0,0,sp.I],[sp.I,0,0,0],[0,-sp.I,0,0]])
    d = sp.diag(1,-1,2,-2)
    fields = [[(1+sp.Rational(x,10))*p0,
               (1+sp.Rational(x,7))*a+sp.Rational(x,20)*d,
               b+sp.Rational(x,13)*a, d] for x in range(4)]
    links = [sp.diag(sp.I,-sp.I,1,1), sp.eye(4),
             sp.diag(1,1,sp.I,-sp.I), sp.eye(4)]
    for row in fields:
        for m in row:
            assert m == m.conjugate().T and sp.trace(m) == 0
    assert all(u*u.conjugate().T == sp.eye(4) and u.det() == 1 for u in links)
    return fields, links, a, b


def native_profile(cfg, mass):
    # Binary source arithmetic, exactly as RotationBand.couplings/scalar_paths.
    a, r, v = cfg['a'], cfg['light_r'], cfg['light_v']
    return {'onsite': F(mass + 4.5*r/a), 'c0': F(-r/a),
            'ci': F(-v/(2*a)), 'c2': F(r/(4*a))}


def hopping_matrices(profile, fields, links):
    result = []
    for edge, (axis, direction) in enumerate(((1,1),(2,1),(1,-1),(2,-1))):
        x, y = edge, (edge+1)%4
        if direction < 0:
            x, y = y, x
        fx = sp.Rational(profile['c0'])*fields[x][0]+sp.I*sp.Rational(profile['ci'])*fields[x][axis]
        fy = sp.Rational(profile['c0'])*fields[y][0]+sp.I*sp.Rational(profile['ci'])*fields[y][axis]
        u = links[edge]
        h = (fx*u+u*fy)/2
        if direction < 0:
            h = h.conjugate().T
        result.append([[gaussian(h[i,j]) for j in range(4)] for i in range(4)])
    return result


def primitive(profile, matrices, fields, mask, incoming=0, z=F(0), wrong_car=False, one_charge=False):
    """Complete 24 orders/16 local modes, with ALL intermediate occupations.
    Diagnostic diagonal onsite reference only; not full quantum H resolvents.
    Incoming flow is common to all species; gauss charge checked each hop.
    """
    ref = mask
    onsite = [[F(str(fields[x][0][i,i]))*profile['onsite'] for i in range(4)] for x in range(4)]
    ref_energy = sum(onsite[x][i] for x in range(4) for i in range(4) if (mask>>(4*x+i))&1)
    rho0 = [((mask>>(4*x))&15).bit_count()-2 for x in range(4)]
    # Incoming divergence zero for this ring; nonzero local reference charges
    # are paid by compatible exterior strings, not silently set to zero.
    if any(rho0):
        raise ValueError('This diagnostic reference needs specified exterior flow')
    total = ZERO
    orders = []
    charges = set()
    denominators = set()
    for order in permutations(range(4)):
        amplitudes = {ref: ONE}
        flow = [incoming]*4
        for depth, edge in enumerate(order):
            flow[edge] += 1
            updated = {}
            for state, amplitude in amplitudes.items():
                for i in range(4):
                    for j in range(4):
                        h = matrices[edge][i][j]
                        if h == ZERO:
                            continue
                        moved = car(state, 4*edge+i, 4*((edge+1)%4)+j, wrong_car)
                        if moved is None:
                            continue
                        dest, sign = moved
                        value = scale(mul(amplitude,h), F(sign))
                        updated[dest] = add(updated.get(dest,ZERO),value)
            if depth < 3:
                for state in list(updated):
                    if state == ref:
                        del updated[state]
                        continue
                    rho = [((state>>(4*x))&15).bit_count()-2 for x in range(4)]
                    assert rho == [flow[x]-flow[(x-1)%4] for x in range(4)]
                    qnorm = sum(abs(q) for q in rho)
                    charges.add(qnorm)
                    if one_charge and qnorm > 2:
                        del updated[state]
                        continue
                    matter = sum(onsite[x][i] for x in range(4) for i in range(4) if (state>>(4*x+i))&1)-ref_energy
                    denominator = matter + 64*sum(k*k for k in flow)-z
                    assert denominator > 0
                    denominators.add(denominator)
                    updated[state] = scale(updated[state], -1/denominator)
            amplitudes = updated
        value = amplitudes.get(ref,ZERO)
        total = add(total,value)
        orders.append({'order':list(order),'amplitude':encoded(value)})
    return {'coefficient':encoded(total),'orders':orders,'intermediate_l1_charge':sorted(charges),
            'min_denominator':str(min(denominators)) if denominators else None,
            'incoming_flow':[incoming]*4,'z':str(z)}


def raw_loop(mask, matrices, wrong=False):
    total = ZERO
    for order in permutations(range(4)):
        amps = {mask:ONE}
        for edge in order:
            updated = {}
            for state, amp in amps.items():
                for i in range(4):
                    for j in range(4):
                        h = matrices[edge][i][j]
                        if h == ZERO:
                            continue
                        moved = car(state,4*edge+i,4*((edge+1)%4)+j,wrong)
                        if moved:
                            dest,sign = moved
                            updated[dest] = add(updated.get(dest,ZERO),scale(mul(amp,h),F(sign)))
            amps = updated
        total = add(total,amps.get(mask,ZERO))
    return total


def run(request):
    pins = request['input_pins']
    captured = {name:(ROOT/name).read_bytes() for name in pins}
    for name, raw in captured.items():
        assert hashlib.sha256(raw).hexdigest() == pins[name], name
    cfg = json.loads(captured[request['micro_input']])['input_descriptor']['original_micro_config_UNCHANGED']
    assert cfg['auxiliary_count'] == 63 and cfg['a'] == 1/64
    fields,links,a,b = fields_and_links()
    b = b + a/3  # Nonzero actual Jordan defect, not a Cliff control.
    anti = a*b+b*a
    lhs = -sp.re(sp.trace(a*b*a*b))
    rhs = sp.trace(a*a*b*b)-sp.trace(anti*anti)/2
    assert sp.simplify(lhs-rhs) == 0
    spin = {'minus_ReTr_ABAB':str(lhs),'Tr_A2B2':str(sp.trace(a*a*b*b)),
            'half_anticommutator_square':str(sp.trace(anti*anti)/2),
            'identity_residual':'0','physics_lower':False}
    sea = sum(1 << (4*x+i) for x in range(4) for i in (0,1))
    rank2_other = sum(1 << (4*x+i) for x in range(4) for i in (0,2))
    rows = []
    for role,count,mass in [('active',1,cfg['active_m']),('auxiliary',63,cfg['auxiliary_m'])]:
        p = native_profile(cfg,mass)
        mats = hopping_matrices(p,fields,links)
        actual = primitive(p,mats,fields,sea)
        wrong = primitive(p,mats,fields,sea,wrong_car=True)
        higher = primitive(p,mats,fields,sea,one_charge=True)
        incoming = primitive(p,mats,fields,sea,incoming=1)
        energy = primitive(p,mats,fields,sea,z=F(-1))
        assert actual['coefficient'] != wrong['coefficient']
        assert actual['coefficient'] != higher['coefficient']
        assert actual['coefficient'] != incoming['coefficient']
        assert actual['coefficient'] != energy['coefficient']
        assert raw_loop(0,mats) == ZERO and raw_loop((1<<16)-1,mats) == ZERO
        assert raw_loop(sea,mats) != raw_loop(rank2_other,mats)
        flux_pi = [[scale(z,F(-1)) for z in row] for row in mats[0]]
        flux_loop = raw_loop(sea,[flux_pi,*mats[1:]])
        assert flux_loop == scale(raw_loop(sea,mats),F(-1))
        rows.append({'role':role,'count':count,'binary_native_coefficients':{k:str(v) for k,v in p.items()},
                     'complete_variable_phi_primitive':actual,'wrong_drop_CAR':wrong,
                     'wrong_drop_higher_charge':higher,'incoming_changed':incoming,'energy_changed':energy,
                     'rank2_carrier_A':encoded(raw_loop(sea,mats)),
                     'rank2_carrier_B':encoded(raw_loop(rank2_other,mats)),
                     'wrong_Jordan_implies_U1_flux_rejected':True,
                     'same_Phi_SU4_flux_pi_bare_word':encoded(flux_loop),
                     'empty_and_full_raw_loop':encoded(ZERO)})
    # Four sites, incoming/outgoing positive native paths, three axes,
    # both lengths and adjoints. All neighboring endpoints obey the stated
    # field cutoff in this conditioned gate, not an actual-ground assertion.
    fmax = max(sum(abs(gaussian(m[i,j])[0])+abs(gaussian(m[i,j])[1])
                   for i in range(4) for j in range(4)) for row in fields for m in row)
    local_b = F(0)
    for row in rows:
        c = {k:F(v) for k,v in row['binary_native_coefficients'].items()}
        local_b += row['count']*4*4*3*fmax*(abs(c['c0'])+abs(c['ci'])+abs(c['c2']))
    delta = 2*min(F(row['binary_native_coefficients']['onsite']) for row in rows)
    ratio = local_b/delta
    assert ratio > 1
    gate = {'support':'all native1/2 paths touching four face sites and their adjoints, all64 roles; conditioned neighboring-endpoint field cutoff',
            'field_endpoint_entry_l1_upper':str(fmax),'local_hopping_norm_upper':str(local_b),
            'conditioned_reference_gap':str(delta),'normalized_ratio_at_physical1':str(ratio),
            'physical1_neumann_gate':False,'full_variable_phi_uniform_gap': '0',
            'remainder_if_lambda_ratio_lt1':'lambda^5 b^5/[delta^4(1-lambda b/delta)]',
            'positive_paid_test_lambda':str(delta/(4*local_b)),
            'not_N_uniform_fullH_continuation':True}
    test_lambda = delta/(4*local_b)
    gate['test_remainder_exact'] = str(test_lambda**5*local_b**5/delta**4/(1-test_lambda*local_b/delta))
    assert F(gate['test_remainder_exact']) > 0
    # Exact resummed/Gram method probes, never called a physical backend.
    d = sp.Matrix([[3,1],[1,2]])
    c = sp.Matrix([1,2])
    exact = (c.T*d.inv()*c)[0]
    image_norm = (c.T*c)[0]
    image_energy = (c.T*d*c)[0]
    jensen = image_norm**2/image_energy
    assert exact >= jensen and exact != (c.T*(sp.eye(2)/3)*c)[0]
    # Positive matrix need not have positive off-diagonal/loop character.
    inverse = cast(Any, d.inv())
    assert inverse.det()>0 and inverse[0,1]<0
    resummed = {'label':'exact rational method-only full-resolvent/Gram discriminators; NOT fullH numerical backend',
                'D':[[str(v) for v in row] for row in d.tolist()],
                'C':[str(v) for v in c],'Cdag_Dinv_C':str(exact),
                'image_norm_squared':str(image_norm),'image_energy':str(image_energy),
                'paid_lower':str(jensen),'inverse_offdiagonal':str(inverse[0,1]),
                'wrong_positive_Gram_implies_positive_loop_rejected':True,
                'wrong_scalar_denominator_rejected':True,
                'full_operator_physical1_representation':'proof.resummed_actual_operator',
                'uniform_flux_lower_proved':False}
    result = {'kind':'VARIABLE_FULLPHI_CAR_LOOP_AND_PHYSICAL1_PAID_OPERATOR_FORM_NOT_PHASE',
              'spin_identity':spin,'roles':rows,'local_normalized_gate':gate,
              'resummed_attempt':resummed,'all_input_pins':pins,
              'inputs_unchanged_after':{name:(ROOT/name).read_bytes()==raw for name,raw in captured.items()},
              'canonical_empty_full_embedding':{'N':8,'V':512,'particles_per_species':1024,
               'empty_four_sites_compensation':'put all4 on any other4 sites, all remaining sites2',
               'full_four_sites_compensation':'put0 on any other4 sites, all remaining sites2',
               'Gauss':'exterior compatible integer strings pay shared64rho; not uncharged ground',
               'loop_zero_is_carrier_statement_not_ground_claim':True},
              'runtime':{'executable':sys.executable,'python':sys.version,'sympy':sp.__version__},
              'campaign_complete':False}
    assert all(result['inputs_unchanged_after'].values())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run(json.loads(args.request.read_text()))
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'kind':result['kind'],'roles':len(result['roles']),
                      'physical1_neumann_gate':False,'uniform_flux_lower':False}))
