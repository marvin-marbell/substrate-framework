"""Exact normalization and wholeP0 implications; no original ground/backend."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import sympy as sp


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_bytes())
    with zipfile.ZipFile(request['preexecution_archive']) as archive:
        for name,pin in request['pins'].items():
            assert hashlib.sha256(archive.read(Path(name).name)).hexdigest()==pin
    m=256;k=128;r=4;c=sp.Rational(m,4*(m-1))
    mean=sp.Rational(r*k,m)
    factorial=sp.Rational(r*(r-1)*k*(k-1),m*(m-1))
    norm2=mean+factorial
    tracefree=sp.Rational(r)-sp.Rational(r*r,m)
    assert norm2==sp.Rational(424,85)==mean**2+c*tracefree
    assert c*tracefree==sp.Rational(84,85) and c*tracefree!=norm2
    unit_carrier=sp.sqrt(1/norm2)
    assert sp.simplify(unit_carrier**2*norm2)==1
    h=sp.diag(0,0,2,5);omega=sp.eye(4)[:,0]
    assert (omega.H*omega)[0]==1
    a=sp.zeros(4);a[2,0]=a[0,2]=1;a[1,0]=a[0,1]=3
    b=sp.zeros(4);b[2,0]=sp.I;b[0,2]=-sp.I;b[3,0]=b[0,3]=1;b[1,0]=b[0,1]=-3
    z=sp.Matrix([1,sp.I])/sp.sqrt(2)
    assert (z.H*z)[0]==1
    u=(z[0]*a+z[1]*b)*omega;q0=sp.diag(0,0,1,1)
    rankone=sp.eye(4)-omega*omega.H
    excited=sp.simplify((u.H*q0*u)[0]);total=sp.simplify((u.H*u)[0])
    first=sp.simplify((u.H*h*u)[0]);second=sp.simplify((u.H*h*h*u)[0])
    eta=sp.Rational(2,3);g=sp.simplify((u.H*q0*(h+eta*sp.eye(4)).inv()*q0*u)[0])
    assert excited==sp.Rational(1,2) and total==sp.Rational(19,2)
    assert sp.simplify((u.H*rankone*u)[0])==total!=excited
    assert first==sp.Rational(5,2) and second==sp.Rational(25,2)
    assert sp.simplify(g-first**3/(second*(second+eta*first)))==0 and g==sp.Rational(3,34)
    assert excited==first**2/second
    result={'kind':'EXACT_CANONICAL_CARRIER_AND_COMPLEX_UNIT_POLARIZATION_NORMALIZATION_ADVERSARIES',
            'canonical_all_common64_singlet_active_reduced_uniform_wedge_k':'SU(M) singlet invariance plus active wedge irreducibility, no actual originating ground weight',
            'M':m,'k':k,'site_orbitals':r,'mean_n':str(mean),'factorial_moment':str(factorial),
            'norm2_n_active_x_PS':str(norm2),'normalizing_factor':str(unit_carrier),
            'WRONG_tracefree_only_norm2':str(c*tracefree),'complex_coefficient_unit_norm':True,
            'wholeP0_excited_norm2':str(excited),'WRONG_rankone_removed_norm2':str(total),
            'first_energy':str(first),'second_energy':str(second),'G_eta':str(g),
            'mass_and_resolvent_Cauchy_bounds_saturated':True,
            'not_native_ground_or_spectrum':True,'source_tau_paid':False,'scientific_admission':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'receipt':args.output.name,'exact_normalization_and_wholeP0_adversaries':True}))


if __name__=='__main__':
    main()
