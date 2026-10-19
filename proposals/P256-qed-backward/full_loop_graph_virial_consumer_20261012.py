"""Own exact full-law radial virial controls; no native ground/evolution backend."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import zipfile
from fractions import Fraction as F
from pathlib import Path
import sympy as sp

ROOT=Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def radial_and_quantum_identity() -> dict:
    a,y=sp.symbols('a y',nonnegative=True)
    t=1+y
    f=t**3/(1+a)
    # a=epsilon*t^(7/2) so a_y=(7/2)*a/t.
    fy=sp.diff(f,y)+sp.diff(f,a)*sp.Rational(7,2)*a/t
    radial=sp.factor(f+2*y*fy)
    target=t**2*(1+7*y+a)/(1+a)**2
    assert sp.simplify(radial-target)==0
    assert all(c>0 for c in sp.Poly(1+7*y+a,a,y).coeffs())
    p=sp.Integer(1);rows=[];constants=[]
    for j in range(3):
        p=sp.expand((3-j)*p*(1+a)+sp.Rational(7,2)*a*((1+a)*sp.diff(p,a)-(j+1)*p))
        bound=sum(abs(c) for c in sp.Poly(p,a).all_coeffs())
        constants.append(bound)
        rows.append({'jet_order':j+1,'polynomial':str(p),'coefficient_l1_bound':str(bound)})
    d=60
    cdiv=d*(d+2)*constants[0]+4*(d+2)*constants[1]+4*constants[2]
    assert cdiv>0
    # Actual operator identity with arbitrary smooth wavefunction, not a point derivative.
    x=sp.Symbol('x',real=True)
    v=sp.Function('v')(x);psi=sp.Function('psi')(x)
    def h(g):
        return -sp.diff(g,x,2)+x**4*g
    def dilation(g):
        return -sp.I*(v*sp.diff(g,x)+sp.diff(v,x)*g/2)
    comm=sp.expand(sp.I*(h(dilation(psi))-dilation(h(psi))))
    rhs=-2*sp.diff(sp.diff(v,x)*sp.diff(psi,x),x)-sp.diff(v,x,3)*psi/2-4*v*x**3*psi
    assert sp.simplify(comm-rhs)==0
    wrong=-2*sp.diff(sp.diff(v,x)*sp.diff(psi,x),x)+sp.diff(v,x,3)*psi/2+4*v*x**3*psi
    assert sp.simplify(comm-wrong)!=0
    # Epsilon=0 full60 radial formula, native radius|q|/sqrt2.
    delta0=3*d*(d+2)*t*t+24*(d+2)*y*t+24*y*y
    assert sp.Poly(delta0,y).all_coeffs()==[12672,23808,11160]
    return {'radial_eigenvalue':str(target),'positive_tangential_and_radial':True,
            'uniform_quantum_jet_polynomials':rows,'native_dimension':d,
            'Cdiv_uniform_epsilon':str(cdiv),'Delta_div_v_epsilon0':str(sp.expand(delta0)),
            'exact_differential_operator_residual':0,'wrong_potential_or_quantum_sign_rejected':True}


def gaussian_moment(power: int, center: int, extra_y: int=0) -> F:
    result=F(0)
    for j in range(power+1):
        k=j+extra_y
        if k%2==0:
            even=math.factorial(k)//math.factorial(k//2)
            result+=F(math.comb(power,j)*center**(power-j)*even,4**(k//2))
    return result


def quantum_adverse_packets() -> dict:
    rows=[]
    for r in (4,16,64):
        m4,m6,m10=[gaussian_moment(p,r) for p in (4,6,10)]
        weighted_y=gaussian_moment(6,r,2)
        p2=(105*m4+4*m10-14*weighted_y)/(14*m6)
        assert p2>0
        weighted_kinetic=14*(p2*m6+weighted_y)
        assert weighted_kinetic-105*m4-4*m10==0
        # Omitting the positive weighted kinetic gives a FALSE upper bound.
        assert 4*m10 > -105*m4
        assert m10>r**10 and p2<r**4
        rows.append({'R':r,'P_squared':str(p2),'radial10':str(m10),
                     'weighted_kinetic':str(weighted_kinetic),'virial_residual':'0',
                     'Phi_kinetic_unweighted':str(p2+F(1,2))})
    assert F(rows[-1]['radial10'])>1000000*F(rows[0]['radial10'])
    # Ground qualification: these Schwartz states are NOT quartic ground vectors.
    # (Hpsi)/psi=(P^2+1-Y^2+2i P Y)+x^4 varies in x.
    x,R,P=sp.symbols('x R P',real=True)
    local_energy=P**2+1-(x-R)**2+2*sp.I*P*(x-R)+x**4
    assert sp.diff(local_energy,x)!=0
    return {'Schwartz_quantum_packet':'pi^-1/4 exp(-(x-R)^2/2) exp(iPx)',
            'Hamiltonian':'p^2+x^4 method control, NOT native ground',
            'weighted_vector_field':'v=x^7',
            'exact_identity':'14<p x^6 p>-105<x^4>-4<x^10>=0',
            'packets':rows,'nonground_local_energy_derivative':str(sp.diff(local_energy,x)),
            'positive_weighted_kinetic_drop_rejected':True,
            'native_ground_falsified':False}


def native_matter_coefficients(request: dict) -> dict:
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        raw=archive.read(request['profile_member'])
    profiles=json.loads(raw)['primitive_profiles']
    assert [p['load'] for p in profiles]==[1,1,63]
    assert [len(p['paths']) for p in profiles]==[3,6,6]
    assert len({p['onsite'] for p in profiles})==3
    w=[abs(F(p['onsite']))+2*sum((abs(F(r['c0']))+abs(F(r['ci'])) for r in p['paths']),F(0)) for p in profiles]
    B=4*(F(2**8192)*w[0]+w[1]+63*w[2])
    assert B<512*2**8192
    return {'primitive_member_sha256':sha(raw),'all65_loads':[1,1,63],
            'native_path_counts':[3,6,6],'distinct_onsite':True,
            'B_sha256':sha(str(B).encode()),'matter_derivative_bound':'8B sum(1+r^7)',
            'source_fullH_changed':False}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text())
    captured={name:(ROOT/name).read_bytes() for name in request['pins']}
    assert all(sha(raw)==request['pins'][name] for name,raw in captured.items())
    result={'kind':'EXECUTED_MATERIALLY_DISTINCT_FULL_GROUND_VIRIAL_ATTEMPT_CONTROLS',
            'radial_and_quantum_identity':radial_and_quantum_identity(),
            'quantum_adverse_packets':quantum_adverse_packets(),
            'complete_native_matter':native_matter_coefficients(request),
            'proof':request['proof'],'inputs':request['pins'],
            'failed_gate':'No uniform K_epsilon weighted r6 pPhi2; its POSITIVE sign prevents proposed upper radial10 bound',
            'actual_full_ground_stationarity':'Analytic fixed-epsilon form-flow proof, not computed state',
            'retained_paid_successor':'SAME L source in D(L^s),0<s<1/17',
            'numerical_native_ground':False,'numerical_fullH_evolution':False,
            'scientific_admission':False,'campaign_complete':False}
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    result['inputs_unchanged']=True
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'result':str(args.output),'inputs_unchanged':True,'virial_sign_gate_reached':True}))


if __name__=='__main__':
    main()
