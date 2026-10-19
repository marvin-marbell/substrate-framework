#!/usr/bin/env python3
"""Actual compact Wilson comparison transfer on a four-link Gauss cycle.

The beta here is the induced Wilson coefficient; no bare term is appended to
any campaign law. This finite comparison does not replace the macro auxiliary
transfer or its source. Both neutral and integer static-charge sectors run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.special import ive

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_compact_gauss_photon_native.py','covariant_photon_energy_control.py',
    'covariant_induced_photon.py','sea_selection.py','rotation_band.py')}


def native(beta: float,cutoff_factor: int) -> dict[str,Any]:
    cutoff=math.ceil(cutoff_factor*math.sqrt(beta))
    flux=np.arange(-cutoff,cutoff+1,dtype=int)
    difference=np.abs(flux[:,None]-flux[None,:])
    magnetic=ive(difference,beta/2)
    i0=float(ive(0,beta))
    one_link=ive(np.abs(flux),beta)/i0
    charged_link=ive(np.abs(flux+1),beta)/i0
    neutral=(magnetic*one_link[None,:]**4)@magnetic
    charged=(magnetic*(one_link**2*charged_link**2)[None,:])@magnetic
    dim=len(flux)
    values,vectors=eigh(neutral,subset_by_index=[dim-4,dim-1])
    if min(values)<=0:
        raise ArithmeticError('Observed physical neutral leading spectrum is not positive')
    lambda0=float(values[-1]);ground=vectors[:,-1]
    cvalues,cvec=eigh(charged,subset_by_index=[dim-4,dim-1])
    if min(cvalues)<=0:
        raise ArithmeticError('Observed physical charged leading spectrum is not positive')
    charged_lambda=float(cvalues[-1]);gap=-math.log(charged_lambda/lambda0)
    raw_gram=float(ground@charged@ground)/lambda0
    def retained(threshold: float) -> dict[str,float|int]:
        eigen,frames=eigh(charged,subset_by_value=[threshold,np.inf])
        overlaps=frames.T@ground
        numerator=float(np.sum(eigen*overlaps**2))/lambda0
        return dict(retained_eigenstates=len(eigen),prepared_retained_probability=numerator/raw_gram)
    physical=retained(lambda0*math.exp(-.25))
    wrong=retained(charged_lambda*math.exp(-.25))
    incidence=np.zeros((4,4),dtype=int)
    for edge in range(4):
        incidence[edge,edge]=1;incidence[(edge+1)%4,edge]=-1
    neutral_e=np.repeat(flux[:,None],4,axis=1)
    charged_e=neutral_e+np.array([1,1,0,0])
    neutral_gauss=neutral_e@incidence.T
    charged_gauss=charged_e@incidence.T-np.array([1,0,-1,0])
    if np.any(neutral_gauss) or np.any(charged_gauss):
        raise ArithmeticError('Actual integer electric basis violates Gauss')
    return dict(beta=beta,integer_flux_cutoff=cutoff,actual_Hilbert_dimension=dim,
        actual_neutral_lambda0=lambda0,actual_neutral_first_three_gaps=[-math.log(float(x)/lambda0) for x in values[-2::-1]],
        actual_charged_lambda0=charged_lambda,actual_static_charge_gap_same_neutral_zero=gap,
        actual_charged_halfstep_Gram_relative_neutral=raw_gram,
        quarter_cut_same_neutral_lambda0=physical,adverse_charged_own_lambda0_cut=wrong,
        actual_neutral_Gauss_maximum=int(np.max(np.abs(neutral_gauss))),
        actual_charged_Gauss_maximum=int(np.max(np.abs(charged_gauss))),
        factorization='Kneutral=V diag[(I_m(beta)/I_0(beta))^4] V; Kcharge=V diag[(I_(m+1)/I_0)^2 (I_m/I_0)^2] V; V_mn=exp(-beta/2) I_(m-n)(beta/2). All four oriented electric links and every retained integer flux enter.',
        positivity='Each exact Galerkin V is positive multiplication by exp[-beta(1-cos phi)/2], and every exact Bessel heat coefficient is positive. Leading eigenvalues above are actual numeric observations; no claim is made about numerically unresolved bottom eigenvalues.')


def run() -> dict[str,Any]:
    betas=(1,4,16,64,256,1024)
    rows=[]
    for beta in betas:
        first=native(float(beta),13);second=native(float(beta),17)
        second['smaller_cutoff_gap_difference_estimate']=abs(second['actual_neutral_first_three_gaps'][0]-first['actual_neutral_first_three_gaps'][0])
        second['smaller_cutoff_charged_gap_difference_estimate']=abs(second['actual_static_charge_gap_same_neutral_zero']-first['actual_static_charge_gap_same_neutral_zero'])
        second['gaussian_reference_gap']=2*math.asinh(1.)
        second['induced_auxiliary_count_kappa400']=str(beta*10**1600//16)
        rows.append(second)
    inputs=dict(native_graph='One contractible four-edge cycle, no short spatial winding',
        incidence='edge i points from node i to (i+1)%4; divergence is outgoing minus incoming',
        neutral_charge=[0,0,0,0],static_source_charge=[1,0,-1,0],integer_source_path=[1,1,0,0],
        induced_beta_values=list(betas),kappa='1e-400',auxiliary_rule='Naux=beta*10^1600/16 exactly',
        flux_cutoff_factors=[13,17],charged_energy_cut=.25,
        source='Neutral actual ground, followed by the integer Wilson source mapping to charge+1/-1 sector and charged positive halfstep',
        preparation_zero='Same neutral lambda0 in BOTH sector source halfstep and quarter energy projector')
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={k:v.decode() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        actual_compact_Gauss_transfer_rows=rows,
        reference_limit='Gauss leaves a single contractible flux. At large beta H_E=2m^2/beta, V_B=beta phi^2/2. A=sqrt(beta) phi/2 and Pi=2m/sqrt(beta) give K=e^-A² e^-Pi²/2 e^-A², whose positive oscillator gap is2asinh1, not an assigned classical frequency.',
        finite_cutoff_status='Two actual flux cutoffs give observed convergence estimates, not outward tail enclosures. The exact displayed transfer and Gauss basis retain all integer m before Galerkin evaluation.',
        microscopic_admission_proved=False,
        scope='Exact compact, Gauss-physical Wilson comparison photon/STATIC-charge transfer. Native beta sweep changes auxiliary species count explicitly and is not the macro beta150 epoch. The quantum auxiliary-to-Wilson/Root/Field comparison, mobile fermion source and macro inclusive receiver are not replaced or proved.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();result=run();result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'source':result['source_at_import'],'input':result['input_descriptor_sha256'],
        'rows':[{key:row[key] for key in ('beta','actual_neutral_first_three_gaps','actual_static_charge_gap_same_neutral_zero',
        'quarter_cut_same_neutral_lambda0','adverse_charged_own_lambda0_cut')} for row in result['actual_compact_Gauss_transfer_rows']],
        'runtime_seconds':result['runtime_seconds']}))


if __name__=='__main__':
    main()
