#!/usr/bin/env python3
"""Pure-Coulomb SOURCE first commutator, retaining true photon-creator moments.

The neutral Pi*(sinF-F) norm is NOT this SOURCE-relative commutator. Full
photon-leg commutators and driven occupation remain separate obligations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from fractions import Fraction as F
from pathlib import Path
from typing import Any
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from numpy.polynomial.hermite import hermgauss
from improved_phi_band import C,NSTAR,UE
from noncompact_quantitative_normal_budget import ceil_log2
from original_charged_principal import portable
from prepared_neutral_local_well import window_photon_moments,exact_capsule


def source_relative_clover_budget(normal:dict[str,Any],time:F,radius:F,
                                  source_squared_lower:F=F(1,16)) -> dict[str,Any]:
    moments=window_photon_moments(normal,time,radius)
    factorial=moments['evolved_factorial_moment_upper']
    # a_g (N+3)^3 a_g^dagger <= (N+4)^3(N+1), for any unit coexact g.
    # Exact falling-factorial conversion; no degree-one photon truncation.
    creator_fourth=64+186*factorial[1]+106*factorial[2]+19*factorial[3]+factorial[4]
    V=NSTAR**3;CB2=F(256*V)
    # Curl gives nu/omega=sqrt(nu)/c: unlike A-displacement, this has NO 1/omega_min.
    flux_variance=16*UE/C
    squared=(7*2**20*V*radius*NSTAR)**2*(CB2/source_squared_lower)*flux_variance**3*creator_fourth
    target=UE/(V*2**40)
    log=(ceil_log2(squared)+1)//2
    return dict(identity='original-PURECOULOMB-source-relative-clover-firstcomm',
        regimes='analytic actual first SOURCE commutator; conditional actual MinSet moment/norm inputs',
        Pi_cancellation='[.5{Pi_e,sinF-F},B]=-i(sinF-F)*partial_Ae B; NO Pi remains, including longitudinal/harmonic components',
        actual_source='original signed-path per-edge spatial-quaternion raw quartet/2; coexact derivative zero, Coulomb Green div A plus explicit harmonic',
        derivative_upper='||partial_Ae B||<=Nstar*CB; Green gap>=16/N² and harmonic |d|/V<=N/V',
        flux_variance_upper=flux_variance,flux_scope='all coexact curl modes, canonical UE Pi²+K curl²/2; curl-zero harmonic does not enter F',
        true_creator_fourth_moment_upper=creator_fourth,source_operator_squared_upper=CB2,
        source_squared_lower=source_squared_lower,angular_total_upper=7,
        source_commutator_squared_upper=squared,source_commutator_log2_upper=log,
        target=target,target_log2=ceil_log2(target),margin_bits_lower=-(log+ceil_log2(1/target)),
        below_target_if_inputs_hold=squared<target*target,
        normalization_and_ordinary_moment_inputs_independently_paid=False,
        not_whole_neutral_Pi_bound=True,not_total_photon_creator_commutator=True,
        NOT_paid=['[Gerr,a_gdagger] photon-leg terms','driven occupation/whole harmonic Pi mixed graph','full physical Noether minus/scattering probability','actual CURRENT MinSet filtered Z and ordinary moment admission'])


def native_first_commutator() -> dict[str,Any]:
    """Actual continuum oscillator/dressing algebra; not a field-state mock."""
    points,weights=hermgauss(100)
    scale=.17;dipole=.73;flux=scale*points
    delta=np.sin(flux)-flux
    photon_density=2*points**2
    measure=weights/math.sqrt(math.pi)
    actual=float(np.sum(measure*photon_density*abs(dipole*delta)**2))
    cubic=float(np.sum(measure*photon_density*(dipole*abs(flux)**3/6)**2))
    records=[]
    for width in (.01,100.):
        y=points/math.sqrt(width);B=np.exp(1j*dipole*y)
        dB=1j*dipole*B
        # p_y(B psi)-B p_y psi is evaluated, not replaced by the answer.
        pBpsi=-1j*dB+1j*width*y*B
        Bppsi=B*(1j*width*y)
        comm=delta[:,None]*(pBpsi-Bppsi)[None,:]
        expected=-1j*delta[:,None]*dB[None,:]
        weighted=float(np.sum(measure[:,None]*photon_density[:,None]*measure[None,:]*abs(comm)**2))
        defect=float(np.max(abs(comm-expected)))
        if defect>2e-12 or abs(weighted-actual)>2e-12 or actual>cubic:
            raise AssertionError('physical momentum-cancellation/cubic bound failed')
        records.append(dict(neutral_momentum_variance=width/2,actual_firstcomm_squared=weighted,operator_defect=defect))
    return dict(canonical_dressing='B=exp(i dipole*y_long), coexact F=.17*x; input actual onephoton Hermite state',
        actual_firstcomm_squared=actual,cubic_upper_squared=cubic,
        neutral_Pi_variance_changes_by=10000,records=records,
        scope='isolated untruncated oscillator SOURCE commutator; not MinSet/Gauss/physicalNoether closure')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--normal-receipt',type=Path,required=True)
    p.add_argument('--time',default='1008124871316563/2');p.add_argument('--radius',default='1342177280000')
    a=p.parse_args();data=a.normal_receipt.read_bytes();root=Path(__file__).resolve().parent
    names=('prepared_neutral_source_moments.py','prepared_neutral_local_well.py','prepared_neutral_canonical_source.py')
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    result=dict(inputs={str(a.normal_receipt):hashlib.sha256(data).hexdigest()},source_hashes=hashes,
        legal=source_relative_clover_budget(json.loads(data),F(a.time),F(a.radius)),native=native_first_commutator(),
        source_unchanged=all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in hashes.items()))
    print(json.dumps(exact_capsule(result),indent=2,sort_keys=True))

if __name__=='__main__':main()
