#!/usr/bin/env python3
"""Actual unchanged Source2a native source/Pe, with interior proof witnesses.

The witness changes neither source nor preparation nor the existing Pe band.
Macro rows are reference/envelope ledgers, NOT substituted physical inputs.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy.linalg import eigh
import covariant_prepared_gauss_state as source
from covariant_induced_photon import Parameters, WilsonConsumer, phi_frame

ROOT=Path(__file__).resolve().parent
OWN_BYTES_AT_IMPORT=Path(__file__).read_bytes()
if sys._getframe().f_code!=compile(OWN_BYTES_AT_IMPORT,__file__,'exec',dont_inherit=True,optimize=sys.flags.optimize):
    raise ArithmeticError('Loaded source consumer code differs from import-time snapshot')
PLAN_REVISION='1124ce49865f8f7cc5f912ef9569edc52ad3227fb532fd78d5637658c923b990'
SELECTED='covariant_prepared_gauss_state_conditioned_source_physical_Dminus_verified_20261008.json'
EXPECTED_SOURCE='2a0ecfb2341fceb7866d5062cf33d1da71b2fafa4b148829d21d14f0650fe307'
EXPECTED_ROOT='6911f0283dc0f84f4cc05202deb67dcd3266d5c64ca1a05e149f1567451e0f9c'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm2(a: np.ndarray) -> float:
    return float(np.linalg.norm(a,2))


def band_witness(A1: np.ndarray,A0: np.ndarray,raw: np.ndarray,
                 cut: float,clocks: list[float],interior_energy: tuple[float,float]) -> dict[str,Any]:
    """Exact finite matrices, including null; full-source outliers stay explicit."""
    v1,u1=eigh((A1+A1.conj().T)/2)
    v0,u0=eigh((A0+A0.conj().T)/2)
    lower=math.exp(-cut);upper=1.
    e_low,e_high=interior_energy
    l0,u0value=math.exp(-e_high),math.exp(-e_low)
    if not lower<l0<u0value<upper:
        raise ValueError('Proof witness must lie strictly inside unchanged Pe')
    selected1=(v1>=lower)&(v1<=upper)
    selected0=(v0>=l0)&(v0<=u0value)
    b1=v1[selected1];b0=v0[selected0]
    q1=u1[:,selected1];q0=u0[:,selected0]
    if not len(b1) or not len(b0):
        raise ValueError('Actual source comparison has an empty proof band')
    X=q1.conj().T@q0
    E=b1[:,None]*X-X*b0[None,:]
    delta=norm2(A1-A0);delta_band=norm2(E)
    leakage=norm2(q0-q1@X)
    dlo=l0-lower;dup=upper-u0value
    leakage_bound=delta*math.sqrt(dlo**-2+dup**-2)
    log_defect=np.log(b1)[:,None]*X-X*np.log(b0)[None,:]
    sqrt_defect=np.sqrt(b1)[:,None]*X-X*np.sqrt(b0)[None,:]
    log_bound=delta_band/lower
    sqrt_bound=delta_band/(math.sqrt(lower)+math.sqrt(l0))
    raw0=u0.conj().T@raw
    raw1=u1.conj().T@raw
    c0=raw0[selected0]
    weighted_reference=np.sqrt(np.maximum(v0,0))[:,None]*raw0
    outlier=weighted_reference.copy();outlier[selected0]=0
    null=v0<=max(1e-12,float(v0[-1])*1e-12)
    actual_source=np.sqrt(b1)[:,None]*raw1[selected1]
    actual_source_gram=float(np.linalg.norm(actual_source,'fro')**2)
    records=[]
    for t in clocks:
        f1=np.sqrt(b1)*np.exp(1j*t*np.log(b1))
        f0=np.sqrt(b0)*np.exp(1j*t*np.log(b0))
        defect=f1[:,None]*X-X*f0[None,:]
        bound=math.sqrt(upper)*abs(t)*delta_band/lower+sqrt_bound
        observed=norm2(defect)
        if observed>bound*(1+1e-8)+1e-10:
            raise ValueError('Actual filtered consumer violates intertwining bound')
        records.append(dict(lattice_clock=t,observed_operator_error=observed,proved_band_operator_bound=bound,
            actual_reference_source_defect_HS=float(np.linalg.norm(defect@c0,'fro')),
            reference_source_defect_HS_bound=bound*float(np.linalg.norm(c0,'fro')),
            actual_original_filtered_source_norm=float(np.linalg.norm(f1[:,None]*raw1[selected1],'fro'))))
    if norm2(log_defect)>log_bound*(1+1e-8)+1e-10 or norm2(sqrt_defect)>sqrt_bound*(1+1e-8)+1e-10:
        raise ValueError('Actual block functional-calculus bound failed')
    return dict(existing_energy_cut=cut,existing_band=[lower,upper],proof_interior_energy=[e_low,e_high],
        proof_interior_spectrum=[l0,u0value],actual_band_rank=len(b1),reference_witness_rank=len(b0),
        actual_full_kernel_difference=delta,actual_intertwining_defect=delta_band,
        actual_log_intertwining_error=norm2(log_defect),log_intertwining_bound=log_bound,
        actual_sqrt_intertwining_error=norm2(sqrt_defect),sqrt_intertwining_bound=sqrt_bound,
        actual_witness_leakage=leakage,source_buffer_leakage_bound=leakage_bound,
        actual_reference_weighted_outlier_HS=float(np.linalg.norm(outlier,'fro')),
        actual_reference_raw_outlier_HS=float(np.linalg.norm(raw0[~selected0],'fro')),
        actual_reference_raw_null_HS=float(np.linalg.norm(raw0[null],'fro')),
        actual_original_energy_prepared_source_Gram=actual_source_gram,
        actual_reference_weighted_source_Gram=float(np.linalg.norm(weighted_reference,'fro')**2),
        clocks=records,witness_is_not_preparation=True,macro_admission=False)




def gaussian_packet_heat_bound(macro: dict[str,Any],gamma: np.ndarray) -> dict[str,Any]:
    photon=macro['photon'];q=np.asarray(photon['momenta'])
    consumer=WilsonConsumer(gamma,.01,1.)
    energies=np.array([consumer.photon_energy(k) for k in q])
    emin,emax=float(energies.min()),float(energies.max())
    coefficient=16*math.exp(2*emax+emin)/(math.e*emin)
    variance=photon['circular_complex_reference_variance']
    return dict(actual_packet_modes=len(q),both_Fourier_orientations=True,Emin=emin,Emax=emax,
        bound_B_Ghalf_squared_over_lambda_G=coefficient,
        actual_reference_complex_second_moment=variance,actual_reference_complex_fourth_moment=2*variance**2,
        formula='16 exp(2Emax+Emin)/(e Emin)',
        proof='Reference Mehler Hermite multiplication: sup n exp(-Emin n)<=1/(e Emin), sup(n+1)exp(-Emin n)<=exp(Emin)/(e Emin); coherent normalized packet coefficient l2 norms and BOTH Fourier orientations cost at most16 exp(2Emax). Bclip=rho(B)B with ||rho||<=1 preserves ||Bclip Ghalf||<=||B Ghalf||. These are reference Gaussian modes, NOT global microscopic CCR.',
        lift_requirement='An actual field-kernel CAR norm majorant by the SAME positive G is required before applying this weighted heat bound to microscopic K. Scalar action/determinant ratios alone do not supply it.',
        no_global_B_operator_cap_used=True)






