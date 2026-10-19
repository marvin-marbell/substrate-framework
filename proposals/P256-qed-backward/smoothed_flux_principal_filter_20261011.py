"""New normalized Abel/principal ground-block filter; prior full-law controls stay frozen."""
from __future__ import annotations
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any
import numpy as np
from magnetic_mechanism_scientist_20261011 import incidence
from smoothed_flux_scientist_20261011 import abel, require, reject

ROOT=Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_native(n: int) -> dict[str, Any]:
    D,C,S=incidence(n);v=n**3
    sites=np.array(list(np.ndindex(n,n,n)))
    f=np.zeros(3*v);f[::3]=np.sqrt(2/v)*np.cos(2*np.pi*sites[:,0]/n)
    kappa=2*np.sin(np.pi/n);rsoft=kappa/(4+kappa)
    theta=np.pi*np.random.default_rng(20261012+n).integers(-19,20,3*v)/11
    phi=C@theta
    g=C.T@(f*abel(phi,rsoft)[1])
    normalized_g=g/(2*rsoft)
    cap=48*(4+kappa)**2
    require(64*(normalized_g@normalized_g)<=cap+1e-10, 'normalized comparator contact cap')
    require(abs((192*kappa*kappa)/(4*rsoft*rsoft)-cap)<1e-10, 'normalization cancels apparent q2 improvement')
    require(np.max(np.abs(D@normalized_g))<1e-13, 'normalized flux full Gauss closure')
    fixed_r=Fraction(3,4)
    reference_m1=128/(1-fixed_r*fixed_r)
    # r->1 Abel values approach the periodic principal function, radial midpoint0 at seams.
    principal=(phi+np.pi)%(2*np.pi)-np.pi
    seam=np.abs(np.abs(principal)-np.pi)<1e-12
    principal[seam]=0
    errors=[]
    for rr in (0.9,0.99,0.999,0.9999):
        values=abel(phi,rr)[0]/(2*rr)
        errors.append(float(np.max(np.abs(values-principal/2))))
    require(errors[-1]<errors[0] and errors[-1]<0.002, 'normalized principal radial limit is exercised')
    require(np.max(np.abs(S@(principal/2)))>0.1, 'principal filter does not erase Bianchi defects')
    wrong=reject(cap<192*kappa*kappa, 'normalized source weakening is not q2 magnetic regularity')
    require(wrong['rejected'], 'normalization adverse case reaches assertion')
    return dict(N=n,kappa=float(kappa),rsoft=float(rsoft),first_harmonic_before=float(2*rsoft),
        first_harmonic_after=1, normalized_contact_configuration=64*float(normalized_g@normalized_g),
        normalized_contact_cap=float(cap), normalized_cap_limit_q0=768,
        normalized_source_weakening_wrongcase=wrong,
        fixed_r_normalized_reference_m1_exact=str(reference_m1),
        principal_radial_max_errors=errors, principal_first_harmonic=1,
        principal_Bianchi_defects_retained=True, ground_measured=False)


def principal_resolvent(n: int) -> dict[str, Any]:
    eps=Fraction(4,n);cutoff=64
    m0=Fraction(0);m1=Fraction(0)
    for j in range(1,cutoff+1):
        energy=256*j*j
        weight=Fraction(1,2*j*j) # principal/2 has fixed first harmonic1
        factor=eps/(energy+eps)
        m0+=weight*factor*factor;m1+=energy*weight*factor*factor
    tail0=eps*eps/Fraction(655360*cutoff**5)
    tail1=eps*eps/Fraction(1536*cutoff**3)
    require(m1>=256*m0, 'resolvent principal reference retains exact support gap')
    require(m1+tail1<=eps*eps/Fraction(512)*Fraction(2), 'uniform convergent filtered principal moment')
    fixed_r=Fraction(3,4);fixed0=Fraction(0);fixed1=Fraction(0)
    for j in range(1,cutoff+1):
        energy=256*j*j;weight=fixed_r**(2*j-2)/Fraction(2*j*j)
        factor=eps/(energy+eps)
        fixed0+=weight*factor*factor;fixed1+=energy*weight*factor*factor
    # An atomless continuum gives the SAME IR weight order and no atom.
    # mu_IR has density1 on [q,2q], epsilon=4q. Integral is16q(1/5-1/6)=8q/15.
    q=Fraction(1,n)
    continuum=Fraction(8,15)*q
    atom=Fraction(16,25)*q
    require(continuum>0 and atom>0, 'normalized filtered method accepts atomless IR too')
    return dict(N=n,epsilon_exact=str(eps),principal_normalization='principal/2, first harmonic1',
        unfiltered_m0='pi^2/12',unfiltered_m1='diverges; do NOT use unfiltered form insertion',
        domain='epsilon(L+epsilon)^-1 Q0 F1 P0 maps into D(L) even when F1 insertion is not D(form)',
        cutoff=cutoff,principal_filtered_m0_interval_exact=[str(m0),str(m0+tail0)],
        principal_filtered_m1_interval_exact=[str(m1),str(m1+tail1)],
        tail_m0_upper_exact=str(tail0),tail_m1_upper_exact=str(tail1),
        fixed_r_normalized_filtered_m0_partial_exact=str(fixed0),
        fixed_r_normalized_filtered_m1_partial_exact=str(fixed1),
        reference_gap_lower=256, reference_applicable=n>=8,
        reference_scope='conditional matching yz external-only closed flows and exact g; N4/6 electric probe only',
        actual_physical_ground=False,
        spectral_method_control=dict(not_physical_ground=True,atom_IR_filtered_mass_exact=str(atom),
            atomless_IR_filtered_mass_exact=str(continuum),
            continuum_density='1 on [q,2q]; UV delta_256 mass1 may be added',
            UV_total_m1_exact=256, UV_filtered_m1_exact=str(256*(eps/(256+eps))**2),
            total_UV_contact_is_not_photon_necessity=True,low_weight_is_not_atom=True))


def run(request_path: Path, output: Path) -> None:
    request=json.loads(request_path.read_text())
    for name,pin in request['input_sha256'].items():require(sha(ROOT/name)==pin,'input SHA '+name)
    prior=json.loads((ROOT/request['prior_receipt']).read_text())
    require(not prior['physical_ground_measured'] and not prior['atom'],'prior scoped controls are not ground')
    result=dict(kind='NORMALIZATION_AUDIT_AND_NEW_PRINCIPAL_RESOLVENT_DOMAIN_CONSTRUCTION_NOT_PHYSICAL_GROUND',
        source_sha256=sha(Path(__file__)),request_sha256=sha(request_path),input_sha256=request['input_sha256'],
        prior_frozen_full_law_control_sha256=sha(ROOT/request['prior_receipt']),
        normalized_native=[normalized_native(n) for n in (4,6,8)],
        principal_resolvent=[principal_resolvent(n) for n in (4,6,8)],
        source_weakening_rejected_as_physical_advance=True,
        fixed_r_and_principal_filtered_route_active=True,actual_ground_IR_estimate=False,atom=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(receipt=output.name,N=[4,6,8],normalization_gate='failed source-weakening route',new_operator='normalized principal ground-block resolvent',physical_ground=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.request,args.output)
