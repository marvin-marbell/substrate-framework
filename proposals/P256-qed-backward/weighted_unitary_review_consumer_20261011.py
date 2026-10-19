"""Independent exact native-coefficient and synthetic unitary-limit controls.
Never imports/runs a previous producer; synthetic matrices are not native fullH.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np
import scipy.linalg as la
import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)

def record(x: F) -> dict:
    return {'sha256': hashlib.sha256(str(x).encode()).hexdigest(),
            'numerator_bits': x.numerator.bit_length(), 'denominator_bits': x.denominator.bit_length()}

def coefficients(profiles: list) -> dict:
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    species = [p for p in profiles for _ in range(int(p['load']))]
    assert len(species) == 65 and sum(p['scale'] == 'charged' for p in species) == 64
    role_weights = [abs(F(p['onsite'])) + sum((2*(abs(F(t['c0']))+abs(F(t['ci']))) for t in p['paths']), F()) for p in profiles]
    # Expand all65 first; each2V canonical restriction cannot increase full-Fock bounds.
    b = sum((4*(lam if p['scale']=='heavy' else 1)*(abs(F(p['onsite'])) + sum((2*(abs(F(t['c0']))+abs(F(t['ci']))) for t in p['paths']), F())) for p in species), F())
    assert b == 4*(lam*role_weights[0]+role_weights[1]+63*role_weights[2]) < 512*lam
    kp, kh, ke, alpha, beta = 4*chi, F(3,20)*chi, F(64), fast/40, 7*fast/10
    z = 4096+2/lam
    terms = {'kinetic':F(54), 'Phi_matter':480*kp*b*b*4096,
             'Phi_alignment':184320*kp*alpha*alpha*z,
             'Haar_matter':480*kh*b*b*z, 'fullE_matter':32*ke*b*b*z,
             'matter_matter':32*b*b*z, 'Haar_alignment':46080*kh*alpha*alpha/lam,
             'Haar_plaquette':2880*kh*beta*beta*4096}
    ljet = 2**16*(1+fast+b)
    cstar = 2**32*65*(1+ljet*ljet)*z
    # Independent stronger termwise domination; no floating huge constants.
    assert all(8*v <= cstar for v in terms.values())
    cindex = sum(terms.values(), F())
    assert cindex <= cstar
    wrong_b = 4*(lam*role_weights[0]+role_weights[1]+role_weights[2])
    assert b-wrong_b == 4*62*role_weights[2] > 0
    return {'scope':'native pinned all65 coefficient arithmetic, not evolution',
            'role_weights_exact':[str(w) for w in role_weights], 'species':65,'charged':64,
            'normalization':'Tr(T_A T_B)=delta_AB/2; s=|q|^2/2; canonical each2V; Ref4 inert',
            'B':record(b),'B_over_lambda_upper':512,'allocations':{k:record(v) for k,v in terms.items()},
            'C_index':record(cindex),'Cstar':record(cstar),'each_allocation_le_Cstar_over_8':True,
            'deleted_62_auxiliary_coefficient_defect':str(b-wrong_b),
            'C_index_over_Cstar_float':float(cindex/cstar)}

def mechanism() -> dict:
    # New noncommuting positive matrix pair, unrelated to native truncation.
    h = sp.Matrix([[4,1,0],[1,6,2],[0,2,9]])
    q = sp.diag(2,5,11)
    s = sp.diag(1,1,0)
    c = sp.I*(h*q-q*h)
    assert h.det()>0 and h[:2,:2].det()>0 and h[0,0]>0
    assert c != sp.zeros(3)
    minors=[]
    for sign in (-1,1):
        bounded = 4*q+sign*c
        for size in (1,2,3):
            for inds in itertools.combinations(range(3),size):
                d=sp.simplify(bounded.extract(inds,inds).det());assert d>=0;minors.append(str(d))
    hp = s*h*s
    assert sp.I*(hp*q-q*hp) == s*c*s
    rotated_v=sp.Matrix([1,0,1])/sp.sqrt(2)
    wrong=rotated_v*rotated_v.H
    defect=sp.I*(wrong*h*wrong*q-q*wrong*h*wrong)-wrong*c*wrong
    assert defect != sp.zeros(3)
    # Exact positive resolvent keeps the outside identity; compression-only wrong inverse deletes it.
    good=(hp+sp.eye(3)).inv()
    bad=s*(hp+sp.eye(3)).inv()*s
    outside=sp.Matrix([0,0,1])
    assert good*outside==outside and bad*outside==sp.zeros(3,1)
    # Compression is not energy-monotone since h and Q do not commute.
    witness=sp.Matrix([0,1,-sp.Rational(1,10)])
    assert (witness.H*hp*witness)[0] > (witness.H*h*witness)[0]
    hn=np.array(h,dtype=complex);qn=np.array(q,dtype=complex);sn=np.array(s,dtype=complex)
    shift=3
    hr=sn@hn@sn-shift*np.eye(3)
    e3=np.array([0,0,1],dtype=complex)
    t=math.pi/3
    correct=la.expm(-1j*t*hr)
    deleted=sn@correct@sn
    wrong_extension=la.expm(-1j*t*(sn@(hn-shift*np.eye(3))@sn))
    assert np.linalg.norm(correct@e3+e3)<1e-12
    assert np.linalg.norm(deleted@e3)==0
    assert np.linalg.norm(wrong_extension@e3-correct@e3)>1.99
    psi=np.array([1,1j,2],dtype=complex);psi/=np.linalg.norm(psi)
    initial=float(np.vdot(psi,qn@psi).real)
    propagation=[]
    for tt in (-.17,0,.31):
        u=la.expm(-1j*tt*hr)@psi
        energy=float(np.vdot(u,qn@u).real)
        bound=math.exp(4*abs(tt))*initial
        assert energy<=bound+1e-12 and abs(np.linalg.norm(u)-1)<1e-12
        propagation.append({'t':tt,'Q':energy,'bound':bound})
    # A genuine infinite direct-sum mechanism: Q_n=n^2 Q, H'_n=n^2 H'.
    # psi_n=n^-2 e3 /sqrt(zeta(4)) belongs form domain but not graph domain.
    partial=[]
    for n in (2,8,32):
        tail2=sp.zeta(2)-sum((sp.Rational(1,j*j) for j in range(1,n+1)),sp.S.Zero)
        tail4=sp.zeta(4)-sum((sp.Rational(1,j**4) for j in range(1,n+1)),sp.S.Zero)
        partial.append({'cutoff':n,'Q_form_tail':float(11*tail2/sp.zeta(4)),
                        'unitary_error_squared_bound':float(4*tail4/sp.zeta(4)),
                        'deleted_extension_t0_norm_loss_squared':float(tail4/sp.zeta(4))})
    assert all(partial[i+1]['Q_form_tail']<partial[i]['Q_form_tail'] for i in (0,1))
    # First-energy excess->0 but graph norm diverges for L e_m=m^2 e_m.
    packet=[]
    for m in (2,16,128):
        eps=sp.Rational(1,m**3)
        first=eps*m*m;second=eps*m**4
        assert first==sp.Rational(1,m) and second==m
        packet.append({'m':m,'first_energy':str(first),'graph_norm_squared':str(second)})
    return {'scope':'synthetic exact noncommuting mechanism and infinite direct sum ONLY, never native fullH',
            'current_C':4,'positive_principal_minors':minors,'correct_Q_cutoff_current':True,
            'wrong_rotated_cutoff_defect_squared':str(sp.trace(defect.H*defect)),
            'outside_resolvent_identity_exact':True,'deleted_outside_resolvent_rejected':True,
            'incorrect_outside_unitary_distance':float(np.linalg.norm(wrong_extension@e3-correct@e3)),
            'deleted_outside_unitary_norm':float(np.linalg.norm(deleted@e3)),
            'energy_monotonicity_counterexample':{'original':str((witness.H*h*witness)[0]),'compressed':str((witness.H*hp*witness)[0])},
            'actual_REAL_expm_controls':propagation,'infinite_form_not_graph_vector':{'normalization':'zeta(4)^(-1/2)','Q_energy':'11*zeta(2)/zeta(4)','graph_divergence':'85/zeta(4) per block; sum diverges','tails':partial},
            'first_mean_not_graph_packets':packet}

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    before={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in request['pins']}
    assert before==request['pins']
    with zipfile.ZipFile(ROOT/request['epoch_archive']) as z:
        raw=z.read(request['profile']);profiles=json.loads(raw)['primitive_profiles']
    result={'scope':'INDEPENDENT_NATIVE_COEFFICIENT_AND_SEPARATE_SYNTHETIC_LIMIT_CONTROLS',
            'coefficients':coefficients(profiles),'mechanism':mechanism(),
            'profile_member_sha256':hashlib.sha256(raw).hexdigest(),'executable':sys.executable,
            'source_input_sha256_before':before,
            'source_input_sha256_after':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in request['pins']}}
    assert result['source_input_sha256_after']==before
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'all65_exact_constant':True,'outside_and_wrong_projection_rejected':True,'native_fullH_run':False}))

if __name__=='__main__':
    main()
