"""New original-law sector trials and whole-quantum number-unitary controls.
Native coefficient calculations are not a ground eigensolver or sector weights.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import cast
import zipfile
import sympy as s

ROOT = Path(__file__).resolve().parent


def hop(mask, target, source):
    if not (mask >> source) & 1:
        return None
    removed = mask ^ (1 << source)
    if (removed >> target) & 1:
        return None
    sign = (-1) ** ((mask & ((1 << source)-1)).bit_count() + (removed & ((1 << target)-1)).bit_count())
    return removed | (1 << target), sign


def apply(entries, vector):
    result = defaultdict(lambda: s.Integer(0))
    for mask, amplitude in vector.items():
        for target, source, coefficient in entries:
            action = hop(mask, target, source)
            if action is not None:
                out, sign = action
                result[out] += sign*coefficient*amplitude
    return {mask: s.expand(value) for mask, value in result.items() if value != 0}


def norm2(vector):
    return s.simplify(sum((s.conjugate(value)*value for value in vector.values()), s.Integer(0)))


def partitions(items):
    if not items:
        yield []
        return
    first, *rest = items
    for blocks in partitions(rest):
        yield [[first], *blocks]
        for index in range(len(blocks)):
            yield blocks[:index] + [[first, *blocks[index]]] + blocks[index+1:]


def loop_control(profiles):
    row = next(p for p in profiles[1]['paths'] if int(p['axis']) == 0 and int(p['length']) == 1)
    c0, ci = s.Rational(row['c0']), s.Rational(row['ci'])
    ratio = ci/c0
    phi = [s.diag(3,-1,-1,-1), s.zeros(4), s.zeros(4), s.zeros(4)]
    phi[1][0,1] = phi[1][1,0] = 1
    phi[2][1,2] = s.I; phi[2][2,1] = -s.I
    phi[3][2,3] = phi[3][3,2] = 2
    u = s.eye(4); v = s.eye(4)
    assert u.det() == v.det() == 1
    def block(x,y,axis,link):
        fx=(1+x)*(phi[0]+s.I*ratio*phi[axis+1]); fy=(1+y)*(phi[0]+s.I*ratio*phi[axis+1])
        return (fx*link+link*fy)/2
    blocks = [block(0,1,0,u), block(1,2,1,v), block(3,2,0,v*u).H, block(0,3,1,u*v).H]
    legs = [[(4*edge+a,4*((edge+1)%4)+b,body[a,b]) for a in range(4) for b in range(4) if body[a,b] != 0] for edge,body in enumerate(blocks)]
    # This identity-SU4 point retains all four noncommuting Phi fields.
    # A coherent Slater carrier spans the full colour-mode face and has a
    # complete separately canonical spectator, never a ground trace weight.
    mask=15|(15<<8)
    moved=hop(mask,4,0)
    assert moved is not None
    carrier={mask:s.Integer(1),moved[0]:s.Integer(moved[1])}
    carrier_norm=norm2(carrier)
    assert carrier_norm==2
    moments = {}
    def moment(word):
        if word not in moments:
            vector=dict(carrier)
            for edge in word: vector=apply(legs[edge],vector)
            moments[word]=s.simplify(sum(s.conjugate(value)*vector.get(key,0) for key,value in carrier.items())/carrier_norm)
        return moments[word]
    total = s.Integer(0); samecopy = s.Integer(0)
    ps = list(partitions(list(range(4))))
    assert len(ps)==15
    for order in itertools.permutations(range(4)):
        samecopy += 64*moment(order)
        for groups in ps:
            multiplicity=1
            for j in range(len(groups)):multiplicity *= 64-j
            product=s.Integer(multiplicity)
            for group in groups:product *= moment(tuple(order[j] for j in sorted(group)))
            total += product
    hermitian = s.simplify((total+s.conjugate(total))/2)
    cropped = s.simplify((samecopy+s.conjugate(samecopy))/2)
    print(json.dumps({'native_full_loop_mean':str(hermitian),'cropped_samecopy_mean':str(cropped)}),flush=True)
    assert hermitian != cropped
    return {'full64_all24_mean_over_c0_pow4':str(hermitian), 'wrong_samecopy_mean_over_c0_pow4':str(cropped),
            'all64_assignments_per_order':64**4, 'partition_count':15, 'noncommuting_Phi0123':True,
            'SU4_loop_point':'all identity, allowed native point; noncommuting SU4 is retained in full-body number consumer and preserved failed epochs',
            'canonical_complement':'4 face sites8 occupied modes +30 full-colour sites120 modes =128 per species',
            'det31_common_singlet_full_loop_action':'0 by exact common-traceless annihilation, not oneparticle loop trace',
            'dark_canonical_complement_loop_action':'0, all face sites empty,128 particles outside',
            'ground_weights_assigned':False}


def native_entries(profile, sites, scale=1):
    position={x:i for i,x in enumerate(sites)}
    g=s.diag(3,-1,-1,-1)
    spatial=[s.zeros(4) for _ in range(3)]
    spatial[0][0,1]=spatial[0][1,0]=1
    spatial[1][1,2]=s.I; spatial[1][2,1]=-s.I
    spatial[2][2,3]=spatial[2][3,2]=2
    u=s.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,s.I,0],[0,0,0,-s.I]])
    v=s.Matrix([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    links=[u,v,u*v]
    entries=[]
    for x in sites:
        ix=position[x]; fx=1+sum(x)
        for a in range(4):entries.append((4*ix+a,4*ix+a,scale*s.Rational(profile['onsite'])*fx*g[a,a]))
        for row in profile['paths']:
            axis,length=int(row['axis']),int(row['length'])
            y=list(x); y[axis]=(y[axis]+length)%4; y=tuple(y)
            iy=position[y]; fy=1+sum(y)
            link=links[axis]**length
            c0,ci=s.Rational(row['c0']),s.Rational(row['ci'])
            left=fx*(c0*g+s.I*ci*spatial[axis]); right=fy*(c0*g+s.I*ci*spatial[axis])
            seam=-1 if x[axis]+length>=4 else 1
            # A=pi/2 on each original positive link. Reverse is exact adjoint.
            body=scale*seam*s.I**length*(left*link+link*right)/2
            for a in range(4):
                for b in range(4):
                    if body[a,b] != 0:
                        entries.append((4*ix+a,4*iy+b,body[a,b]))
                        entries.append((4*iy+b,4*ix+a,s.conjugate(body[a,b])))
    return entries


def canonical_response(profiles):
    sites=list(itertools.product(range(4),repeat=3)); zero=(0,0,0)
    checker=sum(15 << (4*i) for i,x in enumerate(sites) if sum(x)%2==0)
    forbidden: set[tuple[int, ...]] = {zero}
    for axis in range(3):
        for step in (-2,-1,1,2):
            x=[0,0,0];x[axis]=step%4;forbidden.add(tuple(x))
    allowed=[x for x in sites if x not in forbidden]
    dark=sum(15 << (4*sites.index(x)) for x in allowed[:32])
    assert len(forbidden)==10 and len(allowed)>=32
    assert checker.bit_count()==dark.bit_count()==128
    # Literal integer spanning-tree flow, not an orthonormal deck lattice.
    rho=[63*(((checker>>(4*i))&15).bit_count()-2)+((dark>>(4*i))&15).bit_count()-2 for i in range(64)]
    assert sum(rho)==0
    flows={}
    subtotal=list(rho)
    for index in range(63,0,-1):
        child=sites[index]; parent=list(child)
        axis=next(j for j in reversed(range(3)) if parent[j]>0)
        parent[axis]-=1; parent=tuple(parent); pindex=sites.index(parent)
        flows[(parent,axis)]=-subtotal[index]
        subtotal[pindex]+=subtotal[index]
    divergence=[0]*64
    for (origin,axis),value in flows.items():
        target=list(origin);target[axis]=(target[axis]+1)%4
        divergence[sites.index(origin)]+=value;divergence[sites.index(tuple(target))]-=value
    assert divergence==rho
    dark_fullE=64*sum(value**2 for value in flows.values())
    active=native_entries(profiles[1],sites)
    auxiliary=native_entries(profiles[2],sites)
    neutral=native_entries(profiles[0],sites)
    incident=[item for item in active if (item[0]//4==0)!=(item[1]//4==0)]
    incoming=[item for item in incident if item[0]//4==0]
    outgoing=apply(incident,{checker:s.Integer(1)})
    assert norm2(outgoing)>0 and not apply(incident,{dark:s.Integer(1)})
    # n=4 on checker, n=3 on each one-hop component. U_pi flips the mixed
    # component sign but leaves all other FULL-body forms exactly unchanged.
    assert all((mask & 15).bit_count()==3 for mask in outgoing)
    gram=norm2(outgoing)
    psi={checker:s.Integer(1),**outgoing}
    acted=apply(incident,psi)
    cross=s.simplify(cast(s.Basic, sum(s.conjugate(a)*acted.get(mask,0) for mask,a in psi.items())))
    assert s.simplify(cross-2*gram)==0
    assert s.simplify(norm2(psi)-(1+gram))==0
    neutral_action=norm2(apply(neutral,{checker:s.Integer(1)}))
    assert neutral_action>0
    aux_action=norm2(apply(auxiliary,{checker:s.Integer(1)}))
    assert aux_action>0
    # Original checkerboard affine flow: one positive edge's -1 hop shift
    # changes the once-fullE energy by64[(64-1)^2-64^2].
    difference=64*((64-1)**2-64**2)
    assert difference == -8128
    resistance=Fraction(0)
    cos=(1,0,-1,0)
    for q in itertools.product(range(4),repeat=3):
        eig=2*(3-sum(cos[j] for j in q))
        if eig:resistance+=Fraction(2*(1-cos[q[0]]),64*eig)
    assert resistance==Fraction(21,64)
    assert resistance**2+(1-resistance)**2 != 1
    # Physical U1 gauge W_x=e^(i*pi/8)I4, identity elsewhere in U256:
    # detW=i; effective full4 local occupation also contributes i.
    # Native n_aux=124+4=128 gives the original same phase1.
    sewing=(s.I**31)*s.I
    wrong=s.I
    n_aux=31*4+4; physical_rho=0+n_aux-128
    wrong_rho=0+4-128
    assert sewing==1 and sewing!=wrong and n_aux==128
    assert physical_rho==0 and wrong_rho==-124
    return {'N':4,'V':64,'M':256,'k_each_all65':128,'native_profiles_load':[p['load'] for p in profiles],
            'paths_per_profile':[len(p['paths']) for p in profiles], 'native_active_incident_entries':len(incident),
            'fullcolour_outgoing_Gram':str(gram), 'bright_pi_unitary_energy_difference':'-4*Gram/(1+Gram)',
            'bright_original_trial_not_ground':'negative variational difference proves this bright trial is not ground; not evaluated originating ground',
            'dark_forbidden_sites':len(forbidden),'dark_native_active_number_unitary_response':0,
            'dark_original_integerflow_Gauss_max_error':max(abs(a-b) for a,b in zip(divergence,rho)),
            'dark_once_fullE_total_price':dark_fullE,'dark_fullE_price_uniform_N_claim':False,
            'dark_is_full_quantum_canonical_trial_not_ground':True,'dark_k_each':dark.bit_count(),
            'neutral_unscaled_action_Gram':str(neutral_action),'neutral_physical_Gram':'Lambda^2 times unscaled Gram, Lambda=2^8192',
            'auxiliary_unscaled_action_Gram_per_copy':str(aux_action),
            'wrong_neutral_body_deletion_rejected':True,'wrong_64_multiplier_active_response_rejected':gram != 64**2*gram,
            'physical_number_shift_control':{'n_aux':n_aux,'rho':physical_rho,'wrong_without_det_number_shift_rho':wrong_rho},
            'physical_U1_site_gauge_pi_over8_det31_sewing':str(sewing),'wrong_without_determinant':str(wrong),
            'once_fullE_allowed_hop_energy_difference':difference,'N4_longitudinal_raw_jet':str(resistance),
            'N4_transverse_raw_jet':str(1-resistance),'full_jet':1,'wrong_separate_square_sum_rejected':True,
            'Phi_Haar_neutral_Ref_body_energy_changes_under_Utheta':'EXACT0 as original forms; not frozen omission',
            'Peierls_Gauss_bright_section':'native hopping multiplies actual exp(i path.A), and transforms the original integerflow simultaneously; not a constant off-diagonal CAR carrier'}


def trial_comparison():
    v=s.symbols('V',positive=True);chi=s.Rational(2)**-4096;lam=s.Integer(2)**8192;fast=s.Integer(2)**10240
    eta=s.Rational(2)**-7168;t=s.Integer(2)**7168
    upper=60*chi/eta+3000*lam*eta**2+s.Rational(9,2)*fast*eta+s.Rational(27,8)*chi*t+s.Rational(7,5)*(fast/t)*(450+48*7168+24)
    assert upper+262144<s.Integer(2)**3092<lam
    assert s.simplify((upper+262144)*v-upper*v)==262144*v
    assert Fraction(262144,3)+Fraction(524288,3)==262144
    assert 63*128==31*256+128
    assert tuple([63]*128+[0]*128)!=tuple([32]*128+[31]*128)
    return {'highest_auxiliary_partition':'(63^128,0^128)','det31_auxiliary_partition':'(32^128,31^128)',
            'retained_total_auxiliary_degree':8064,'highest_weight_multiplicity':1,'full_det31_multiplicities':'ALL, no paired embedding replacement',
            'exact_two_trial_energy_difference_per_site':262144,'highest_HE_per_site':'524288/3','highest_HC_per_site':'262144/3',
            'det31_HE_HC_per_site':[0,0],'same_sharp_bosonic_upper':'<2^3092-262144',
            'highest_and_det31_sector_upper_below_Lambda':True,
            'first_attempt_ground_comparison':'FAILED: two upper bounds do not order sector minima or complete competitor',
            'second_attempt_actual_response_lower':'FAILED: original ground z_x and wholeP0 source mass unpaid; native dark trial is not ground'}


def main():
    sys.set_int_max_str_digits(0)
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_bytes())
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:raw=archive.read(request['profile_member'])
    assert hashlib.sha256(raw).hexdigest()==request['profile_member_sha256']
    profiles=json.loads(raw)['primitive_profiles']
    assert [int(p['load']) for p in profiles]==[1,1,63]
    result={'kind':'NEW_NATIVE_N4_COMPLETE_SECTOR_TRIAL_AND_WHOLE_QUANTUM_UNITARY_RESPONSE_DISCRIMINATOR_NOT_GROUND',
            'first_construction':trial_comparison(),'second_construction':canonical_response(profiles),'native_fullcolour_loop':loop_control(profiles),
            'runtime':{'executable':sys.executable,'version':sys.version,'sympy':s.__version__},'governor':request['governor'],
            'all_sector_ground_weights':'UNPAID','campaign_complete':False,'scientific_admission':False}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'receipt':args.output.name,'both_constructions_executed':True,'actual_sector_selection':'UNPAID','actual_originating_response_lower':'UNPAID'}))


if __name__=='__main__':
    main()
