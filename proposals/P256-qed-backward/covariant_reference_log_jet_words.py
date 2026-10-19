#!/usr/bin/env python3
"""Finite active-SF/log jets at the Gaussian reference, with explicit word counts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_reference_log_jet_words.py','covariant_photon_energy_control.py',
    'covariant_induced_photon.py','sea_selection.py','rotation_band.py')}
from covariant_induced_photon import Parameters
from covariant_photon_energy_control import gaussian_energy_form_control, reference_linear_log_jet_control


def product(a: list[Fraction],b: list[Fraction]) -> list[Fraction]:
    return [sum((a[i]*b[n-i] for i in range(n+1)),Fraction(0)) for n in range(len(a))]


def compositions(n: int) -> list[tuple[int,...]]:
    if n==0:
        return [()]
    return [(j,)+tail for j in range(1,n+1) for tail in compositions(n-j)]


def sf_majorants(volume: int,maximum_grade: int=5) -> dict[str,Any]:
    """Exact rational formal-word majorants; field factor B^n is separate."""
    if volume<1 or maximum_grade<1:
        raise ValueError('Positive volume and grade required')
    nmax=maximum_grade
    perturb=[Fraction(0)]+[Fraction(3,math.factorial(n)) for n in range(1,nmax+1)]
    b=[Fraction(8)]+perturb[1:]
    f=[Fraction(3,math.factorial(n)) for n in range(nmax+1)]
    inverse=[Fraction(1)]+[Fraction(0)]*nmax
    for n in range(1,nmax+1):
        inverse[n]=sum((perturb[j]*inverse[n-j] for j in range(1,n+1)),Fraction(0))
    invf=product(inverse,f);finvf=product(f,invf)
    t=[inverse[n]+2*invf[n]+b[n]+finvf[n] for n in range(nmax+1)]
    square=[Fraction(0)]*(nmax+1)
    relative=[Fraction(1)]+[Fraction(0)]*nmax
    for n in range(1,nmax+1):
        square[n]=Fraction(3,2)*(t[n]+sum((square[j]*square[n-j] for j in range(1,n)),Fraction(0)))
        relative[n]=3*square[n]
    logdet=[Fraction(0)]*(nmax+1)
    power=[Fraction(1)]+[Fraction(0)]*nmax
    for r in range(1,nmax+1):
        power=product(power,perturb)
        for n in range(1,nmax+1):
            logdet[n]+=volume*power[n]/r
    determinant=[Fraction(1)]+[Fraction(0)]*nmax
    for n in range(1,nmax+1):
        determinant[n]=sum((j*logdet[j]*determinant[n-j] for j in range(1,n+1)),Fraction(0))/n
    exterior=[Fraction(1)]+[Fraction(0)]*nmax
    delta=[Fraction(0)]+relative[1:]
    power=[Fraction(1)]+[Fraction(0)]*nmax
    slots=4*volume
    for k in range(1,nmax+1):
        power=product(power,delta)
        choose=math.comb(slots,k) if k<=slots else 0
        for n in range(1,nmax+1):
            exterior[n]+=choose*power[n]
    boundary=product(determinant,exterior)
    return dict(onebody_T_coefficients=t,onebody_sqrt_coefficients=square,
                onebody_relative_sqrt_coefficients=relative,
                determinant_coefficients=determinant,full_exterior_coefficients=exterior,
                SF_relative_coefficients=boundary)


def run(p: Parameters) -> dict[str,Any]:
    volume=p.spatial_size**3;B=64.;maximum_grade=5
    sf=sf_majorants(volume,maximum_grade)
    boundary=sf['SF_relative_coefficients']
    energy=gaussian_energy_form_control(p,9,.25)
    d=math.log(8)+energy['photon_maximum_energy']/2
    guard=1+2**-40
    whitened=[1.]+[float(sum((boundary[i]*boundary[n-i] for i in range(n+1)),Fraction(0)))
                         *B**n*math.exp(n*d)*guard for n in range(1,maximum_grade+1)]
    h=[0.]+[sum(math.prod(whitened[j] for j in word) for word in compositions(n))*guard
                for n in range(1,maximum_grade+1)]
    sharp=reference_linear_log_jet_control(p)
    h[1]=min(h[1],10**sharp['norm_H1_on_reference_energy_core_log10_upper']*guard)
    jet_rows=[dict(grade=n,ordered_log_compositions=len(compositions(n)),
                   SF_relative_rational=str(boundary[n]),
                   whitened_transfer_coefficient_log10_upper=math.log10(whitened[n]),
                   active_log_jet_coefficient_log10_upper=math.log10(h[n])) for n in range(1,maximum_grade+1)]
    # This prices the actual finite Duhamel word geometry, but does not assert
    # the missing along-g fifth-remainder bound from jets at zero.
    terminal=[]
    for prefix_grade in range(5):
        for prefix in compositions(prefix_grade):
            for last in range(1,6):
                if prefix_grade+last<5:
                    continue
                word=prefix+(last,);weight=sum(word);length=len(word)
                price=(-p.beta_power/2*weight+length*math.log10(p.time_slices)
                       +sum(math.log10(h[j]) for j in word)-math.lgamma(length+1)/math.log(10))
                terminal.append(dict(ordered_jet_grades=list(word),total_grade=weight,
                                     timeword_length=length,conditional_norm_log10_upper=price))
    highest=max(row['conditional_norm_log10_upper'] for row in terminal)
    total=highest+math.log10(sum(10**(row['conditional_norm_log10_upper']-highest) for row in terminal))
    inputs=dict(parameters=asdict(p),reference_initial_energy=.25,maximum_zero_coupling_jet_grade=5,
                maximum_terminal_grade=9,active_exterior_slots=str(4*volume),field_word_factor=64,
                microscopic_auxiliary_treatment='Separate quantum auxiliary-to-Gaussian comparison; not omitted full NF lifting')
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={k:v.decode() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        exact_onebody_word_majorants={key:[str(x) for x in value] for key,value in sf.items() if key.startswith('onebody')},
        active_zero_coupling_log_jet_rows=jet_rows,reference_energy_form=energy,
        conditional_finite_Duhamel_terminal_words=terminal,
        conditional_finite_Duhamel_total_log10_upper=total,
        proof=[
            'Let each link-field word carry unit formal norm, its actual energy-core norm being B^n. For B(g) and F(g), coefficient bounds are 3/n! for n>=1, with B0<=8, F0<=3, B0^-1<=1. The inverse majorant is [1-3(exp z-1)]^-1. Sum the four canonical T1 block norms to obtain the displayed exact rational t_n; no repository-size polynomial count is added a second time.',
            'S0 S_n+S_n S0=T_n-sum_(j=1)^(n-1) S_j S_(n-j). At the actual free T1 spectrum[1/8,8], the Sylvester inverse norm<=sqrt8/2<=3/2; right multiplication by S0^-1 costs<=sqrt8<=3. These recurrences give every displayed relative-square-root word coefficient through grade5.',
            'The det(B)^1/2 relative factor has log-majorant V[-log(1-3(exp z-1))], since B has dimension2V. Its exponential coefficients are computed by n d_n=sum_j j ell_j d_(n-j). Exterior Gamma of the onebody relative factor has exactly binomial(4V,k) choices of k nonidentity slots. Antisymmetrization is contractive; fields at the same slice commute. The determinant/exterior convolution retains all active CAR sectors, not only one open line.',
            'Each SF_n word contains at most n bilinears and n photon fields. Whitening it with the free K0 costs exp[n(log8+Omega/2)]. K_n=sum_(i+j=n) R_i K0 R_j^dagger gives the displayed X_n=K0^-1/2 K_n K0^-1/2 bounds.',
            'The coefficient of -log K is the sum over ordered compositions of n. Each resolvent integral is V^dagger X_a1 B_s X_a2 ... B_s X_ar V, where V is the isometry of the unital first-log CP map and B_s=K0/(K0+s) is a contraction. Free-energy projections commute with these resolvents. Its norm is bounded by the product of X_aj energy-core norms; this is not entrywise divided-difference domination.',
            'For real-time dynamics, replacing the actual neutral vacuum normalization by the fixed free scalar changes only an overall phase in the fixed total charged-number sector. This does not replace the actual neutral lambda0 in the charged preparation energy projector, which remains an independent source-admission obligation.'],
        proved_scope='Zero-coupling active-SF/Gaussian-photon log jets through grade5 on finite reference energy-form cores, conditional constant Phi and static harmonic angles. All full active exterior word counts are explicit.',
        conditional_Duhamel_scope='The terminal-word prices assume an actual along-real-g fifth Hamiltonian remainder bounded by the displayed H5 coefficient on the indicated word cores. Zero-coupling jets alone do NOT prove that assumption. Exact physical U stays at the far left; no non-selfadjoint polynomial Hamiltonian is installed.',
        unproved_obligations=['Along-g fifth remainder, not just jets at zero',
                              'Compact electric/magnetic and Gauss/Coulomb joint contributions to H2..H5',
                              'Actual auxiliary/Root/Field common-range comparison and normalized source Grams',
                              'Linked grade3/4 inclusive hard receiver remainder and actual Born lower'],
        microscopic_Born_lower_proved=False)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();result=run(Parameters(kappa_power=400));result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256','active_zero_coupling_log_jet_rows',
        'conditional_finite_Duhamel_total_log10_upper','microscopic_Born_lower_proved','runtime_seconds')}))


if __name__=='__main__':
    main()
