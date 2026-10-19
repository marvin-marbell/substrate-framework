#!/usr/bin/env python3
"""Include the solved integer U(1) Gauss/Coulomb step in active reference log jets."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_reference_gauss_log_jets.py','covariant_reference_log_jet_words.py',
    'covariant_photon_energy_control.py','covariant_induced_photon.py','sea_selection.py','rotation_band.py')}
from covariant_induced_photon import Parameters
from covariant_photon_energy_control import gaussian_energy_form_control
from covariant_reference_log_jet_words import sf_majorants,compositions


def actual_periodic_gauss_controls() -> list[dict[str,Any]]:
    rows=[]
    for L in (4,8,16,32,64,128):
        n=np.arange(L);one=(2*np.sin(math.pi*n/L))**2
        lap=one[:,None,None]+one[None,:,None]+one[None,None,:]
        nonzero=lap>0
        inverse=np.zeros_like(lap);inverse[nonzero]=1/lap[nonzero]
        green=float(np.sum(inverse))/L**3
        r=L//4
        pair=float(np.sum(inverse*(1-np.cos(2*math.pi*n*r/L))[:,None,None]))/L**3
        if green>13/16 or pair>2*green+1e-14:
            raise ArithmeticError('Actual periodic Coulomb sum violates the shell/charge bound')
        rows.append(dict(spatial_size=L,actual_nonzero_modes=L**3-1,
                         actual_inverse_laplacian_mean=green,
                         actual_static_unit_pair_separation=r,
                         actual_static_unit_pair_Coulomb_coefficient=pair))
    return rows


def run(p: Parameters) -> dict[str,Any]:
    volume=p.spatial_size**3;B=64.;maximum=5
    energy=gaussian_energy_form_control(p,9,.25)
    sf=sf_majorants(volume,maximum)['SF_relative_coefficients']
    d=math.log(8)+energy['photon_maximum_energy']/2
    # This is a full active Fock operator bound, not a single charged open line.
    coulomb=Fraction(13,32)*(4*volume+1)**2
    guard=1+2**-40
    whitened=[1.]
    rows=[]
    for n in range(1,maximum+1):
        terms=[]
        for r in range(n//2+1):
            order=n-2*r
            boundary=sum((sf[i]*sf[order-i] for i in range(order+1)),Fraction(0))
            price=float(boundary*coulomb**r/math.factorial(r))*B**order*math.exp(order*d)*guard
            terms.append(dict(Coulomb_heat_insertions=r,active_SF_field_degree=order,
                              norm_coefficient_upper=price))
        whitened.append(sum(x['norm_coefficient_upper'] for x in terms)*guard)
        rows.append(dict(grade=n,actual_full_active_slot_count=str(4*volume),
                         middle_Gauss_step_terms=terms,
                         whitened_transfer_jet_log10_upper=math.log10(whitened[n])))
    logjets=[0.]+[sum(math.prod(whitened[j] for j in word) for word in compositions(n))*guard
                    for n in range(1,maximum+1)]
    for row in rows:
        row['active_Gauss_log_jet_log10_upper']=math.log10(logjets[row['grade']])
    terminals=[]
    for weight in range(5):
        for prefix in compositions(weight):
            for last in range(1,6):
                if weight+last<5:
                    continue
                word=prefix+(last,);length=len(word)
                price=-p.beta_power/2*sum(word)+length*math.log10(p.time_slices)
                price+=sum(math.log10(logjets[j]) for j in word)-math.lgamma(length+1)/math.log(10)
                terminals.append(dict(jet_grades=list(word),conditional_log10_upper=price))
    peak=max(x['conditional_log10_upper'] for x in terminals)
    total=peak+math.log10(sum(10**(x['conditional_log10_upper']-peak) for x in terminals))
    inputs=dict(parameters=asdict(p),reference_initial_energy=.25,maximum_zero_coupling_jet_grade=5,
        maximum_terminal_grade=9,integer_tagged_charge=1,integer_Ref_hole_charge=-1,
        original_Ref_background=-4,actual_Ref_occupation=3,active_orbitals_per_site=4,
        inverse_laplacian_shell_upper='13/16',field_word_factor=64,
        actual_periodic_control_sizes=[4,8,16,32,64,128])
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={k:v.decode() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        exact_Coulomb_operator_majorant=str(coulomb),actual_periodic_Gauss_controls=actual_periodic_gauss_controls(),
        active_Gauss_log_jet_rows=rows,conditional_terminal_words=terminals,
        conditional_terminal_total_log10_upper=total,reference_energy_form=energy,
        proof=[
            'For q!=0, qhat²>=4|p|²/pi². An infinity-norm shell r has at most26r² modes. Thus V^-1 sum qhat^-2<=13/16. The zero mode is omitted ONLY because the physical integer body+Ref charge is exactly zero; not by a photon mass or charge shrink.',
            'The longitudinal minimum is C=(1/(2V))sum_(q!=0) rho_q^dagger rho_q/qhat². Every full-active-Fock density has norm<=4V; the fixed Ref hole adds1. Thus ||C||<=13(4V+1)²/32. The uniform filled-sea background vanishes only at q!=0, not as a real-PH/occupied CAR subtraction.',
            'The solved Gaussian electric step is Kph exp(-g²C). Since C commutes with coexact field operators, the positive comparison transfer is SF(gA) Kph exp(-g²C) SF(gA)^dagger. At reference g=0, its whitened coefficient is sum_(i+j+2r=n) Y_i (-C)^r Y_j^dagger/r!, Y_i=K0^-1/2 R_i K0^1/2. There is no additional free-fermion whitening of the central C: both SF0 factors cancel exactly.',
            'The same ordered-composition resolvent CP/Stinespring contraction as the active-SF-only branch gives the displayed H_n bounds. C^r has at most2r density bilinears, so total fermion transitions<=2n log8 and field degree<=n. The terminal grade9 Gaussian energy core covers these words.'],
        proved_scope='Zero-coupling log jets of the full active-CAR/Gaussian coexact/solved-longitudinal reference transfer through grade5. Static harmonic angles and constant normalized Phi are comparison inputs. Quantum auxiliary and Root/Field source admission remain separate.',
        conditional_remainder_status='The terminal sum is conditional on a matching along-real-g fifth Hamiltonian remainder. Jets at zero alone are not that theorem. Compact electric and magnetic non-Gaussian contributions still require their own joint estimate.',
        microscopic_Born_lower_proved=False)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();result=run(Parameters(kappa_power=400));result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256','actual_periodic_Gauss_controls',
        'conditional_terminal_total_log10_upper','runtime_seconds')}))


if __name__=='__main__':
    main()
