#!/usr/bin/env python3
"""Exact uncut global neutral twelve-mode Gaussian Gram, with cutoff bound."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import eigh
from scipy.special import gammaincc,roots_hermitenorm
from original_charged_principal import portable
from prepared_neutral_shared_photon_state import load_shared_native_cache
from prepared_neutral_moving_matter_frame import mean_matter_generators


def neutral_global_gram(model: Any,data: dict[str,Any],direction: np.ndarray,order: int=4) -> dict[str,Any]:
    raw=model.neutral_coordinates@data['neutral_state']
    hard=raw[:210].reshape(2,7,15)
    if np.linalg.norm(raw[210:])!=0 or np.linalg.norm(hard[:,:,1:])!=0:
        raise ValueError('Frozen original state is not the original14 normal-polynomial/hardvac embedding')
    coefficients=hard[:,:,0];n=model.chart.normal_dim
    d=direction/np.linalg.norm(direction);axes=model.original.directions
    if np.linalg.norm(axes.T@axes-np.eye(6))>1e-12:raise ValueError('Original Gaussian polynomial directions not orthonormal')
    projections=axes.T@d;remaining=d-axes@projections;residual=float(np.linalg.norm(remaining))
    active=7 if residual>1e-14 else 6;rest=n-active
    abscissa,weight=roots_hermitenorm(order);weight=weight/math.sqrt(2*math.pi)
    forms=np.zeros((2,6,6),complex)
    for index in itertools.product(range(order),repeat=active):
        z=abscissa[list(index)];w=float(np.prod(weight[list(index)]))
        along=float(projections@z[:6])+(residual*z[-1] if active==7 else 0.)
        radius=float(z@z)
        phi=np.array([1.,along,(along*along-1)/math.sqrt(2),(radius-active)/math.sqrt(2*n)])
        products=np.outer(phi,phi)
        # Conditional complement chi-square variance restores ALL n directions.
        products[3,3]+=rest/n
        original=np.column_stack((np.ones(1),z[:6].reshape(1,6)))@coefficients.T
        densities=np.abs(original.ravel())**2
        for bank in range(2):
            forms[bank,:4,:4]+=w*densities[bank]*products
            forms[bank,4,4]+=w*densities[bank]
            forms[bank,5,5]+=w*densities[bank]
    total=forms[0]+forms[1];difference=forms[0]-forms[1]
    gram=np.block([[total,difference],[difference,total]])
    A=float(np.sum(np.abs(coefficients[:,0])**2));B=float(np.sum(np.abs(coefficients[:,1:])**2))
    flat=model.radius/2;x=(flat/model.sigma)**2;a=n/2
    # Sum squared six multipliers averaged over original55 EM Gaussian is
    # <=3.5+n/2+3S+(.5+.5/n)S², S=2||q||²/sigma².
    polynomial=np.polynomial.polynomial.polymul([2*A,2*B],[3.5+n/2,3.,.5+.5/n])
    tail=0.;rising=1.
    for k,coefficient in enumerate(polynomial):
        if k:rising*=a+k-1
        tail+=2*coefficient*(2.**k)*rising*gammaincc(a+k,x)
    return dict(uncut_global_Gram=gram,bank_forms=forms,uncut_original_norm_squared=float(gram[0,0].real),
        Gram_eigenvalues=eigh(gram,eigvals_only=True),active_normal_directions=active,
        analytically_integrated_complement_dimensions=rest,normal_dim=n,GH_order=order,
        cutoff_Gram_operator_loss_bound_numeric=float(tail),
        cutoff_loss_bound_exact='2*sum_{k=0..3} polynomial_k*2^k*(n/2)_k*Q(n/2+k,(radius/(2sigma))^2)',
        floating_bound_evaluation_NOT_certified_interval=True,
        source_state_and_GLOBALZ_NOT_renormalized=True,
        physical_new_H_operator_NOT_changed=True,
        scope='Exact uncut neutral Gaussian polynomial moments, all2430normal/all55EM originalhardvac; compactcutoff analyticbound; NOTSOURCEGram/Hamiltonian/domain/legal/ground proof')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-npz',type=Path,required=True)
    p.add_argument('--cache-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();start=time.monotonic();model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    direction=mean_matter_generators(model,data)['Y0'];four=neutral_global_gram(model,data,direction,4)
    five=neutral_global_gram(model,data,direction,5)
    difference=float(np.linalg.norm(four['uncut_global_Gram']-five['uncut_global_Gram']))
    if difference>1e-11:raise AssertionError('Two exact polynomial Gaussian rules disagree')
    if four['Gram_eigenvalues'][0]<=0:raise AssertionError('Actual neutral enriched functions not independent')
    result=dict(result=four,GH4_vs_GH5_matrix_difference=difference,
        frozen_old_conditional_norm_squared=float(np.vdot(data['neutral_state'],data['neutral_Gram']@data['neutral_state']).real),
        original_physical_state_function_held=True,no_preparation_H_G_loop_rerun=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        program_wall_seconds=time.monotonic()-start)
    text=json.dumps(portable(result),sort_keys=True);args.output.write_text(text);os.chmod(args.output,0o600);print(text)

if __name__=='__main__':main()
