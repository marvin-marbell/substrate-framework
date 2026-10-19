"""Exact charged/CAR weighted half-loop lower, a NEW resummed-form consumer.
Finite diagonal-reference calculation is labelled; fullH representation is analytic.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
from itertools import permutations
import json
from pathlib import Path
import jordan_loop_form_consumer_20261012 as native

ROOT = Path(__file__).resolve().parent


def adjoint(matrix):
    return [[(matrix[j][i][0],-matrix[j][i][1]) for j in range(4)] for i in range(4)]


def step(amplitudes, edge, matrix, reverse=False):
    result = {}
    for state, amp in amplitudes.items():
        for i in range(4):
            for j in range(4):
                h = matrix[i][j]
                if h == native.ZERO:
                    continue
                # Select sites explicitly; no unsigned occupation surrogate.
                if reverse:
                    x,y = (edge+1)%4,edge
                else:
                    x,y = edge,(edge+1)%4
                moved = native.car(state,4*x+i,4*y+j)
                if moved:
                    dest,sign = moved
                    result[dest] = native.add(result.get(dest,native.ZERO),native.scale(native.mul(amp,h),F(sign)))
    return {state:value for state,value in result.items() if value != native.ZERO}


def paid_form(profile, matrices, fields, sea):
    onsite = [[F(str(fields[x][0][i,i]))*profile['onsite'] for i in range(4)] for x in range(4)]
    ref = sum(onsite[x][i] for x in range(4) for i in (0,1))
    def energy(state,flow):
        rho = [((state>>(4*x))&15).bit_count()-2 for x in range(4)]
        assert rho == [flow[x]-flow[(x-1)%4] for x in range(4)]
        value = sum(onsite[x][i] for x in range(4) for i in range(4) if (state>>(4*x+i))&1)-ref+64*sum(k*k for k in flow)
        assert value > 0
        return value
    rows = []
    total = native.ZERO
    price = F(0)
    for order in permutations(range(4)):
        left = {sea:native.ONE}
        left_flow = [0]*4
        for depth,edge in enumerate(order[:2]):
            left_flow[edge] += 1
            left = step(left,edge,matrices[edge])
            left.pop(sea,None)  # Whole selected sea/source for every flow.
            if depth == 0:
                left = {state:native.scale(amp,1/energy(state,left_flow)) for state,amp in left.items()}
        right = {sea:native.ONE}
        right_flow = [1]*4
        for depth,edge in enumerate(reversed(order[2:])):
            right_flow[edge] -= 1
            right = step(right,edge,adjoint(matrices[edge]),True)
            right.pop(sea,None)
            if depth == 0:
                right = {state:native.scale(amp,1/energy(state,right_flow)) for state,amp in right.items()}
        assert left_flow == right_flow
        cross = native.ZERO
        normleft = normright = F(0)
        for state in set(left)|set(right):
            a,b = left.get(state,native.ZERO),right.get(state,native.ZERO)
            den = energy(state,left_flow)
            cross = native.add(cross,native.scale(native.mul((b[0],-b[1]),a),1/den))
            normleft += (a[0]**2+a[1]**2)/den
            normright += (b[0]**2+b[1]**2)/den
        paid = (normleft+normright)/2
        assert -cross[0] >= -paid
        exact_square = normleft+normright-2*cross[0]
        assert exact_square >= 0 and -cross[0] == exact_square/2-paid
        total = native.add(total,cross)
        price += paid
        rows.append({'order':list(order),'W':native.encoded(cross),
                     'X_squared':str(normleft),'Y_squared':str(normright),
                     'difference_squared':str(exact_square),'signed_lower':str(-paid)})
    return {'orders':rows,'positive_R_word':native.encoded(total),
            'signed_minus_ReW':str(-total[0]),'complete_paid_lower':str(-price),
            'halfloop_identity_exact':True,'carrier_and_intermediate_charge_paid':True}


def run(request):
    raw = {name:(ROOT/name).read_bytes() for name in request['input_pins']}
    assert all(hashlib.sha256(value).hexdigest()==request['input_pins'][name] for name,value in raw.items())
    cfg = json.loads(raw[request['micro_input']])['input_descriptor']['original_micro_config_UNCHANGED']
    previous = json.loads(raw['jordan_loop_form_actual1_20261012_result.json'])
    fields,links,_,_ = native.fields_and_links()
    sea = sum(1 << (4*x+i) for x in range(4) for i in (0,1))
    roles = []
    for role,count,mass in [('active',1,cfg['active_m']),('auxiliary',63,cfg['auxiliary_m'])]:
        profile = native.native_profile(cfg,mass)
        matrices = native.hopping_matrices(profile,fields,links)
        result = paid_form(profile,matrices,fields,sea)
        old = next(row for row in previous['roles'] if row['role']==role)['complete_variable_phi_primitive']['coefficient']
        assert result['positive_R_word'] == native.encoded((-F(old['real']),-F(old['imag'])))
        roles.append({'role':role,'count':count,'weighted_halfloop':result})
    lower = sum(row['count']*F(row['weighted_halfloop']['complete_paid_lower']) for row in roles)
    value = sum(row['count']*F(row['weighted_halfloop']['signed_minus_ReW']) for row in roles)
    assert value >= lower
    return {'kind':'COMPLETE_VARIABLE_PHI_NATIVE_CAR_CHARGED_HALFLOOP_PAID_LOWER_EXACT',
            'scope':'Finite diagonal onsite reference, source sea/all flows, all24 orders,16CAR, all canonical8 local states reached, all intermediate shared integer charge. NOT full quantum H backend.',
            'physical_hopping':1,'roles':roles,'all1plus63_signed_form':str(value),
            'all1plus63_paid_lower':str(lower),'outward_control':'EXACT GAUSSIAN RATIONAL',
            'fullH_material_resummation':'jordan_loop_form_proof_20261012.json resummed_actual_operator; cutoff R_eta exact paid lower valid at1, positive stiffness NOT proved',
            'not_positive_flux_lower':True,'input_pins':request['input_pins'],
            'source_inputs_unchanged':all((ROOT/name).read_bytes()==value for name,value in raw.items()),
            'campaign_complete':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(json.loads(args.request.read_text()))
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'kind':result['kind'],'all1plus63_paid_lower':result['all1plus63_paid_lower'],'positive_flux_lower':False}))
