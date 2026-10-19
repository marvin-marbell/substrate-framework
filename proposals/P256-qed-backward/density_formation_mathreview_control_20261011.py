"""Independent finite exact/adverse consumer, never an old extractor/solver run."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sympy as s

ROOT = Path(__file__).resolve().parent
V = s.symbols('z a b c d')
Q = V[1]*V[4]-V[2]*V[3]

def labels(n):
    return [(l,p,q) for l in range(n+1) for p in range(l+1) for q in range(l+1)]

def norm(n,l):
    return s.Rational(math.factorial(n-l)*math.factorial(l+1)*math.factorial(2*n+2),math.factorial(n+1)*math.factorial(n+l+2))*s.Rational(2)**(l-n)

def basis(n,l,p,q):
    scale=s.sqrt(s.Rational(math.factorial(p)*math.factorial(q)*math.factorial(l-p)*math.factorial(l-q),math.factorial(l)))
    psi=0
    for t in range(l+1):
        ks=(l-p-q+t,q-t,p-t,t)
        if min(ks)>=0:
            psi+=scale*s.prod(x**k/math.factorial(k) for x,k in zip(V[1:],ks))
    m=n-l
    result=sum(s.Rational(math.factorial(m),2**r*math.factorial(r)*math.factorial(m-2*r))*V[0]**(m-2*r)*Q**r/s.rf(l+2,r)*psi for r in range(m//2+1))
    return s.expand(result/s.sqrt(norm(n,l)))

def inner(a,b):
    aa,bb=s.Poly(a,*V),s.Poly(b,*V)
    return s.simplify(sum((v*bb.coeff_monomial(k)*math.prod(math.factorial(t) for t in k) for k,v in aa.terms()), s.Integer(0)))

def coefficients(n,l,p,q,k):
    # Literal model independently transcribed from pinned extraction lines48..56.
    if k==0:
        return [] if l==n else [((l,p,q),s.sqrt(s.Rational((n-l)*(n+l+2),n*(2*n+1))))]
    a,b=divmod(k-1,2)
    sp=(l-p if a==0 else p);sq=(l-q if b==0 else q)
    result=[]
    if l and sp and sq:
        result.append(((l-1,p-a,q-b),s.sqrt(s.Rational((n+l+1)*(n+l+2)*sp*sq,2*n*l*(l+1)*(2*n+1)))))
    if l<n-1:
        tp=(p+1 if a==0 else l-p+1);tq=(q+1 if b==0 else l-q+1)
        result.append(((l+1,p+1-a,q+1-b),(-1)**(a+b)*s.sqrt(s.Rational((n-l)*(n-l-1)*tp*tq,2*n*(l+1)*(l+2)*(2*n+1)))))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out={'kind':'INDEPENDENT_EXACT_FINITE_AND_ADVERSE_CONTROLS_NOT_ALL_N_PROOF','degree_controls':[]}
    derivative_count=0
    for n in range(1,5):
        source={lab:basis(n,*lab) for lab in labels(n)}
        target={lab:basis(n-1,*lab) for lab in labels(n-1)}
        for lab,poly in source.items():
            assert inner(poly,poly)==1
            lap=s.diff(poly,V[0],2)-2*(s.diff(poly,V[1],V[4])-s.diff(poly,V[2],V[3]))
            assert s.expand(lap)==0
            for k in range(5):
                predicted=sum(c*target[t] for t,c in coefficients(n,*lab,k))
                assert s.simplify(s.expand(s.diff(poly,V[k])/s.sqrt(n)-predicted))==0
                derivative_count+=1
        if n<=3:
            assert all(inner(a,b)==(i==j) for i,a in enumerate(source.values()) for j,b in enumerate(source.values()))
        out['degree_controls'].append({'n':n,'dimension':len(source),'all_column_derivatives_and_boundaries':True,'full_gram':n<=3})
    # A stage-dependent slot sign is an isometry but not a particle tower.
    source=V[1]*V[4]
    ordered={(i,j):s.diff(source,V[i],V[j])/s.sqrt(2) for i in range(5) for j in range(5)}
    wrong={(i,j):(-1 if i==1 else 1)*z for (i,j),z in ordered.items()}
    assert sum(z*z for z in ordered.values())==sum(z*z for z in wrong.values())==1
    assert wrong[1,4]!=wrong[4,1]
    proper=sum((1 if i==j else 2)*z*z for (i,j),z in ordered.items() if i<=j)
    unfolded=sum(z*z for (i,j),z in ordered.items() if i<=j)
    assert proper==1 and unfolded==s.Rational(1,2)
    # Actual relative up sign is not optional: n2,l0 channel01 must be -c/(2sqrt5).
    p20=basis(2,0,0,0)
    assert s.simplify(s.diff(p20,V[2])/s.sqrt(2)+V[3]/(2*s.sqrt(5)))==0
    assert s.simplify(s.diff(p20,V[2])/s.sqrt(2)-V[3]/(2*s.sqrt(5)))!=0
    # Normalized y=(1,1)/sqrt2 permits entry errors<=kappa/2;
    # the symmetric signed error has nuclear norm sqrt2*kappa>kappa.
    gram_error=s.Matrix([[1,1],[1,-1]])*s.Rational(1,200)
    assert gram_error*gram_error.T==s.eye(2)*s.Rational(1,20000)
    assert s.sqrt(2)/100>s.Rational(1,100)
    out['adverse']={'mere_isometry_wrong_tower':{'ordered_norm_squared':'1','commutator_pair_1_4':str(wrong[1,4]-wrong[4,1])},'missing_sqrt2_norm_squared':str(unfolded),'wrong_up_sign_residual':str(s.simplify(s.diff(p20,V[2])/s.sqrt(2)-V[3]/(2*s.sqrt(5)))),'dimension_factor_witness_nuclear_norm':'sqrt2/100','dimension_factor_witness_entry_error':'1/200','dimension_factor_witness_normalized_input_norm_squared':'1','dimension_factor_witness_kappa':'1/100'}
    # Binary norm independently evaluated using float.as_integer_ratio, not exponent bins.
    vals=[0.0,-0.0,1.0,-1.5,float.fromhex('0x0.0000000000001p-1022'),float.fromhex('0x1.0000000000000p-1022')]
    expected=sum((Fraction(x)**2 for x in vals),Fraction(0))
    bins=[0]*2047
    import struct
    for x in vals:
        word=struct.unpack('Q',struct.pack('d',x))[0]
        e=(word>>52)&2047;m=word&((1<<52)-1)
        if e:m+=1<<52
        bins[e]+=m*m
    got=Fraction(bins[0]+sum(v<<(2*(e-1)) for e,v in enumerate(bins[1:],1)),1<<2148)
    assert got==expected
    out['binary_norm_adverse_domain']={'zero_signed_normal_subnormal_min_normal':True,'historical_no_underflow_not_inferred':True}
    interface=json.loads((ROOT/'three_cell_density_formation_interface_20261011.json').read_text())
    N=Fraction(interface['source_norm_squared_exact']);N0=Fraction(interface['common_N0_exact'])
    u=Fraction(1,2**53)
    gamma=lambda k:k*u/(1-k*u)
    stage=5*(3*u+gamma(4)*(1+3*u));two=2*stage+stage*stage;fold=2*two+3*u*(1+2*two)
    audited={}
    for role,d,L,e in [('AA',15,690880,fold),('UU',15,9224250,fold),('AU',25,3373755,two)]:
        B=(L+32767)//32768;kappa=gamma(65536)+(1+gamma(65536))*gamma(B)
        delta=N/N0*(e*(2+e)+d*kappa*(1+e)**2)+abs(N/N0-1)
        row=interface['errors'][role]
        assert delta==Fraction(row['delta_trace_norm_exact'])
        field0=Fraction(row['field0_trace_exact'])/N0;field1=Fraction(row['field1_trace_exact'])/N0
        assert abs(field0-1)<=delta and abs(field1-1)>delta
        audited[role]={'delta_exact_equal':True,'display':float(delta),'blocks':B,'field1_normalized_trace':float(field1),'field1_substitution_rejected':True}
    out['conditional_scalar_audit']=audited
    # Entire stored vector: an independent integer-ratio denominator aggregation.
    import numpy as np
    with np.load(ROOT/'local_so5_resum_actual04_20261011_state.npz',allow_pickle=False) as data:
        vector=data['state'].reshape(-1)
        assert vector.dtype==np.dtype('float64') and vector.size==9672320
        denominator_bins={}
        for x in vector:
            numerator,denominator=float(x).as_integer_ratio()
            denominator_bins[denominator]=denominator_bins.get(denominator,0)+numerator*numerator
        independent_N=sum((Fraction(v,d*d) for d,v in denominator_bins.items()),Fraction(0))
        assert independent_N==N
        out['entire_source_norm']={'entries':int(vector.size),'equals_interface_exact_rational':True,'method':'float.as_integer_ratio integer numerator-squares grouped by exact denominator','exact':str(independent_N)}
    out['derivative_identity_instances']=derivative_count
    out['conditional_scalar_audit_does_not_attest_historical_machine_operations']=True
    out['own_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':
    main()
