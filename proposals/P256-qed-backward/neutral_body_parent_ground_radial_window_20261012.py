"""Static ground-family radial weight from complete-H trial/coercivity, not Clifford order."""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
from typing import cast

import sympy as sp

ROOT=Path(__file__).resolve().parent


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    for name,h in request['pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    profiles=json.loads((ROOT/request['profiles']).read_text())['primitive_profiles']
    neutral=profiles[0]
    m,c0,ci=Q(neutral['onsite']),Q(neutral['paths'][0]['c0']),Q(neutral['paths'][0]['ci'])
    a0,ai=m*m+6*c0*c0,2*ci*ci
    amax=max(a0,ai)
    assert amax<Q(51,2)**2
    dch=Q(0)
    for profile in profiles[1:]:
        d=2*abs(Q(profile['onsite']))+4*sum(abs(Q(p['c0']))+abs(Q(p['ci'])) for p in profile['paths'])
        dch+=int(profile['load'])*d
    assert dch==Q(855444290449274477,35184372088832)
    assert dch/Q(2**8192)<Q(1,2)
    trial=json.loads((ROOT/request['source_interface']).read_text())
    assert Q(trial['energy']['total_upper_exact'])<-21
    # Trace0 half-canonical KyFan <=.5Tr|h|<=sqrt(V Tr h^2).
    # Native edge c0/ci imaginary cross is zero even at noncommuting fields.
    x=cast(sp.Matrix,sp.Matrix([[0,1],[1,0]]))
    y=cast(sp.Matrix,sp.Matrix([[0,-sp.I],[sp.I,0]]))
    z=cast(sp.Matrix,sp.diag(1,-1))
    left=[cast(sp.Matrix,sp.kronecker_product(z,sp.eye(2))/2),cast(sp.Matrix,sp.kronecker_product(x,y)/3)]
    right=[left[0]+sp.diag(3,-1,-1,-1)/16,left[1]+cast(sp.Matrix,sp.kronecker_product(y,z)/32)]
    t=sp.zeros(4);t[0,2]=t[2,0]=1
    u=cast(sp.Matrix,(sp.eye(4)+sp.I*t/3)*(sp.eye(4)-sp.I*t/3).inv())
    p,q=left[0]*u+u*right[0],left[1]*u+u*right[1]
    cross=sp.simplify(sp.I*sp.trace(p*q.H-q*p.H))
    assert cross==0 and p*q.H-q*p.H!=sp.zeros(4)
    for f,lo,hi in [(p,left[0],right[0]),(q,left[1],right[1])]:
        assert sp.simplify(2*sp.trace(lo**2+hi**2)-sp.trace(f*f.H))>=0
    # Complete-H lower E/V >= -(sqrt(Amax)Lambda+Dch)sqrt(<s>).
    # NEW quantum-trial premise + global minimality therefore imply <s>>441/676.
    mean_floor=Q(21,26)**2
    moment_upper=Q(2**14)
    threshold=Q(1,4)
    high_threshold=Q(2**16)
    high_mass=(mean_floor-threshold)**2/moment_upper
    window_mass=high_mass-moment_upper/high_threshold**2
    assert high_mass==Q(289,28561*1024)
    assert window_mass>0
    # Same moments need not give simultaneous patch occupancy or ordered matrices.
    assert moment_upper/high_threshold**2==Q(1,2**18)
    # Separately keep the diagonal Jordan penalty of the three spatial fields.
    # Their optimized component lower exceeds -14Lambda. Thus the NEW upper
    # also forces a nonzero Phi0 second moment, not only a total radial moment.
    assert 3*ai**2/128 < Q(28,3)**3
    assert 12*dch/Q(2**8192)<Q(1,2)
    phi0_mean_floor=Q(13,51)**2
    phi0_threshold=Q(1,64)
    phi0_high=Q(2**20)
    phi0_window=(phi0_mean_floor-phi0_threshold)**2/moment_upper-moment_upper/phi0_high**2
    assert phi0_window>0
    result={'kind':'CONDITIONAL_COMPLETE_H_ACTUAL_FAMILY_STATIC_RADIAL_WINDOW_NOT_CLIFFORD_OR_PHASE',
        'measure':'inherited normalized physical Gauss/sewn-cover Phi marginal; translation-averaged globally minimizing Bloch family',
        'lattice_scope':'every even N>=4',
        'neutral_trace_square_coefficients':{'A0':str(a0),'Ai':str(ai),'Amax':str(amax)},
        'all64_charged_Dch_exact':str(dch),
        'actual_family_mean_s_lower_exact':str(mean_floor),
        'actual_family_s2_upper':str(moment_upper),
        'per_fixed_site_static_window':'1/4<=s_x=sum_mu Tr(Phi_mux^2)<=2^16',
        'conditional_window_probability_lower_exact':str(window_mass),
        'native_complex_trace_cross_vanishes_without_commuting_fields':True,
        'source_trial_and_inherited_coercivity_must_be_reviewed':True,
        'Phi0_static_component':{
            'spatial_Jordan_and_neutral_lower_per_Lambda':'>-14',
            'actual_family_mean_TrPhi0_squared_lower_exact':str(phi0_mean_floor),
            'per_fixed_site_window':'1/64<=Tr(Phi0_x^2)<=2^20',
            'conditional_probability_lower_exact':str(phi0_window),
            'derivation':'J>=64 sum_mu x_mu^2; neutralK/V<=2sqrt(A0*x0)+2sqrt(Ai*sum_i xi); spatial minimization>-14, charged cost<.5Lambda, so 51sqrt(<x0>)>13/2 and <TrPhi0^2>>169/2601; Cauchy+Markov with M2^14'},
        'analytic_implication':['Trace0 K_2V(h)<=.5Tr|h|<=sqrt(V Trh^2)',
          'Trh^2<=sum_x[(m^2+6c0^2)s0x+sum_i2ci^2six]',
          'Eglob/V>=-(sqrt(Amax)Lambda+Dch)sqrt(<s>)',
          'Eglob/V<-21Lambda and coefficient/Lambda<26 imply <s>>441/676',
          '<s><=1/4+sqrt(<s^2>P(s>=1/4)); subtract Markov upper for s>2^16'],
        'simultaneous_all_patch_or_Clifford_rigidity':False,'ground_loop_Gram_or_wholeP0_excitation':False,
        'sharp_field_projector_form_preservation_claimed':False,'phase_stiffness_atom_commoncone_admitted':False,
        'actual_minimizing_vector_computed':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Conditional actual-family radial weight proved; no Clifford/order/wholeP0 excitation claim')


if __name__=='__main__':
    main()
