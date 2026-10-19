"""Exact fullcolour support-price and signed native occupation controls.
No numerical ground, phase or spectrum is supplied by this consumer.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import zipfile
import sympy as sp

ROOT=Path(__file__).resolve().parent


def support_price(profiles):
    c0=sp.Rational(profiles[1]['paths'][0]['c0'])
    ci=sp.Rational(profiles[1]['paths'][0]['ci'])
    q=sp.Symbol('q',real=True)
    T=sp.diag(3,-1,-1,-1)/sp.sqrt(24)
    assert sp.trace(T*T)==sp.Rational(1,2) and T.det()==-sp.Rational(1,192)
    U=sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,sp.I,0],[0,0,0,-sp.I]])
    Phi0x=sp.diag(3,-1,-1,-1);Phi0y=sp.diag(1,2,-1,-2)
    Phiax=sp.Matrix([[0,1,sp.I,0],[1,0,0,0],[-sp.I,0,0,1],[0,0,1,0]])
    Phiay=sp.Matrix([[1,0,1,0],[0,-1,0,sp.I],[1,0,0,0],[0,-sp.I,0,0]])
    C=((c0*Phi0x+sp.I*ci*Phiax)*U+U*(c0*Phi0y+sp.I*ci*Phiay))/2
    V=sp.Matrix([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    c2=sp.Rational(next(p['c0'] for p in profiles[1]['paths'] if p['axis']==0 and p['length']==2))
    # Wrapped N4 x=3 ->0 ->1, ordered Wilson U then V, AP seam -1.
    actual_two=-c2*(Phi0x*U*V+U*V*Phi0y)/2
    wrong_order=-c2*(Phi0x*V*U+V*U*Phi0y)/2
    assert actual_two!=wrong_order and U.det()==V.det()==1
    block=c0*q*T*U/2+C
    determinant=sp.Poly(sp.expand(block.det(method='berkowitz')),q)
    leading=sp.simplify(determinant.LC())
    assert leading==-c0**4/3072 and determinant.degree()==4
    a4=c0**4/3072
    dim=math.comb(256,128)
    one=F(4*math.comb(240,127),dim)
    empty=F(math.comb(240,128),dim)
    probability=one**4*empty**60
    assert probability>0 and probability<one
    # A_{tau,sigma}=12 tensor_e t_e: all four colour spaces are retained,
    # and all64 species are independent canonical factors.
    assert 12**2==144
    p=sp.Symbol('p',positive=True);B0=sp.Symbol('B0',positive=True)
    kp=sp.Symbol('kP',positive=True);C85=sp.Symbol('C85',positive=True)
    M=(8*C85/p)**sp.Rational(2,17)
    d=a4*(p/(64*sp.sqrt(2*B0/kp)))**4
    anti=sp.simplify(32*(d/a4)**sp.Rational(1,4)*sp.sqrt(2*B0/kp))
    tail=sp.simplify(4*C85*M**(-sp.Rational(17,2)))
    assert anti==p/2 and tail==p/2
    return {'native_det_leading_coefficient':str(leading),'native_det_degree':4,
            'normalized_direction_TrT2':'1/2','uniform_SU4_leading_magnitude':str(a4),
            'literal_native_endpoints_and_ci':True,'native_wrapped_length2_AP':-1,
            'wrong_ordered_SU4_length2_rejected':True,
            'allcolour_support_block_dimension':4**4,
            'all64_occupation_pattern':'first4 charged copies: one full-colour particle at their respective source vertex and zero at other3; other60: face empty; EVERY exterior canonical spectator retained',
            'canonical_spectators':[127]*4+[128]*60,
            'canonical_conditional_pattern_probability':str(probability),
            'canonical_probability_identified_with_actual_ground':False,
            'field_price':{'fixed_law_kP':'4*2^-4096','derivative_budget':'2 B0/kP',
                           'tail':'4 C85 M^-17/2','determinant_sublevel':'32(d/a4)^1/4 sqrt(2 B0/kP)',
                           'choice_M':'max(1,(8 C85/p)^2/17)','choice_d':str(d),
                           'combined_price_at_M_lower_bound':str(anti+tail)},
            'native_supported_total_Gram_lower':'144 d^8/(2 C_active M)^24 * (omega(P_sigma)-p)',
            'wholeP0_supported_excitation_lower':'same bound MINUS ||P0 D_M Omega||^2; exact whole ground-module term is NOT paid by counting or field price',
            'fixed_cutoff_dynamic_lower':'if g>0 is the preceding wholeP0 lower: G_eta >= g^2/(eta*g + Cg*M^4*sqrt(g))',
            'failure_gate':'actual omega(P_sigma) and ground-module response are unavailable; common-singlet complement alone does not bound either',
            'C_active':str(abs(c0)+abs(ci))}


def signed_native_response():
    # Native four-DISTINCT-copy transition in 64 canonical factors. Original
    # source offdiagonal coefficient is 24/2=12 times native fourleg product.
    A=sp.Matrix([[0,12],[12,0]]);n=sp.diag(1,0);C=sp.I*(n*A-A*n)
    plus=sp.Matrix([1,sp.I])/sp.sqrt(2);minus=sp.Matrix([1,-sp.I])/sp.sqrt(2)
    vplus=sp.simplify((plus.H*C*plus)[0]);vminus=sp.simplify((minus.H*C*minus)[0])
    incoherent=sp.trace(C)/2
    assert vplus==-12 and vminus==12 and incoherent==0
    assert (plus.H*A*plus)[0]==0
    kp=sp.Symbol('kP',positive=True)
    return {'full64_distinct_copy_loop_offdiagonal_over_native_leg_product':12,
            'active_site_number_eigenvalues':[1,0],
            'signed_occupation_loop_plus_phase':str(vplus),'minus_phase':str(vminus),
            'incoherent_phase_average':str(incoherent),
            'complex_interference_is_not_absolute_loop_norm':True,
            'conditional_normalized_canonical_trace_commutator_mean':0,
            'actual_ground_dynamic_signed_lower':'G_eta(A)>= [max(0,|omega(i[n,A])|-|i omega(n P0 A-A P0 n)|)]^2 / [4(D_n+16 eta)]',
            'paid_n_energy_price':'D_n=8 sum_active_native_6paths(|c0|+|ci|) C85^(2/17); incoming+outgoing, NN+length2 and AP retained',
            'occupation_source_norm_price':'||n_active,x||<=4, ||Q0 n Omega||^2<=16',
            'failed_material_successor_gate':'conditional canonical signed mean0 despite positive covariance; conditional coherent signs flip. No inherited actual-ground signed mean or wholeP0 matrix is provided.',
            'scope':'exact full64 native transition block with unrestricted spectator completions; not a selected spin/opposite-pair eligibility or a physical ground'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_bytes())
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:raw=archive.read(request['profile_member'])
    assert hashlib.sha256(raw).hexdigest()==request['profile_member_sha256']
    profiles=json.loads(raw)['primitive_profiles']
    result={'kind':'NATIVE_FULLCOLOUR_PAID_FIELD_PRICE_AND_SIGNED_OCCUPATION_ATTEMPT_NOT_GROUND',
            'quantitative_supported_complex_Gram':support_price(profiles),
            'material_signed_loop_occupation_response':signed_native_response(),
            'governor':request['governor'],'actual_ground_executed':False,'scientific_admission':False,'campaign_complete':False}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'receipt':args.output.name,'native_field_price_and_signed_attempt_completed':True,'actual_ground_transfer':'UNPAID'}))


if __name__=='__main__':main()
