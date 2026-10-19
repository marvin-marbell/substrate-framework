#!/usr/bin/env python3
"""Unchanged original signed Source2a/Pe consumer in its joint supported range.

The full raw source and exact K-null component are retained. Range compression
is a numerical evaluation optimization with residual prices, NOT a source gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import eigh
import covariant_actual_source_band_consumer as consumer
from covariant_induced_photon import Parameters,phi_frame
source=consumer.source
ROOT=Path(__file__).resolve().parent
OWN_BYTES_AT_IMPORT=Path(__file__).read_bytes()
if sys._getframe().f_code!=compile(OWN_BYTES_AT_IMPORT,__file__,'exec',dont_inherit=True,optimize=sys.flags.optimize):
    raise ArithmeticError('Loaded range source code differs from import-time snapshot')


def run_signed(gamma: np.ndarray,sign: int,nodes: int,steps: int,*,return_runtime: bool=False) -> dict[str,Any]:
    original=source.native_normal_preparation(gamma,steps,nodes,True,body_charge=sign)
    r=original['runtime'];prepared=source.energy_cut_source(r,6.)
    zero=source.native_range(source.two_spin_frame(gamma),.09,32,2,sign)
    physical=source.physical_basis(zero['states'],3,sign)
    index={(i,n):j for j,(i,n,_) in enumerate(physical)}
    embed=np.zeros((len(physical),len(zero['physical'])))
    for j,(i,n,_) in enumerate(zero['physical']):embed[index[i,n],j]=1
    embedding=np.kron(np.eye(2),np.kron(np.eye(nodes),embed))
    q1=embedding@r['charged_range_isometry']
    q0=embedding@np.kron(np.eye(2),np.kron(np.eye(nodes),zero['u']))
    union,s,_=np.linalg.svd(np.column_stack((q1,q0)),full_matrices=False)
    union=union[:,s>1e-12]
    frame1=union.conj().T@q1;frame0=union.conj().T@q0
    residual1=float(np.linalg.norm(q1-union@frame1,2));residual0=float(np.linalg.norm(q0-union@frame0,2))
    if max(residual1,residual0)>1e-11:raise ValueError('Joint support compression failed its numerical residual price')
    A1=(frame1*r['charged_eigenvalues'])@frame1.conj().T
    levels,u=eigh(r['normal_hamiltonian'])
    nhalf=(u*np.exp(-levels/2))@u.conj().T
    factor=np.kron(np.eye(2),np.kron(nhalf,np.diag(np.sqrt(zero['values']))))
    F0=frame0@factor
    A0=F0@F0.conj().T/r['neutral_lambda0']
    raw=r['raw_source_purification'];projected=union.conj().T@raw
    witness=consumer.band_witness(A1,A0,projected,6.,[0.,float(steps),4.*steps],(.5,5.))
    numerical_null=witness['actual_reference_raw_null_HS']
    witness['actual_reference_raw_null_HS']=float(np.linalg.norm(raw-q0@(q0.conj().T@raw),'fro'))
    witness['numerical_tiny_eigenvalue_null_proxy_HS']=numerical_null
    witness['actual_original_K_null_source_HS']=float(np.linalg.norm(raw-q1@(q1.conj().T@raw),'fro'))
    expected=prepared['summary']['total_source_retained_Gram']
    observed=witness['actual_original_energy_prepared_source_Gram']
    if abs(expected-observed)>1e-8*(1+expected):raise ValueError('Range evaluation changed unchanged source/Pe Gram')
    result=dict(body_charge=sign,original_native_steps=steps,original_native_nodes=nodes,original_native_Pe_energy_cut=6.,
        physical_raw_source_codomain=raw.shape[0],joint_supported_dimension=union.shape[1],
        exact_physical_raw_K_null_retained=True,not_a_new_source_or_preparation=True,
        compression_residual_operator_norms=[residual1,residual0],
        exact_original_boundary_minors=original['summary']['microscopic_boundary_minors'],
        actual_original_energy_preparation=prepared['summary'],actual_original_source_band_consumer=witness)
    if return_runtime:
        result['runtime']=dict(actual_charged_kernel=A1,reference_charged_kernel=A0,
            original_raw_source_purification=projected,
            physical_raw_source_outside_joint_support_HS=float(np.linalg.norm(raw-union@projected,'fro')),
            original_actual_Pe_Gram=expected)
    return result


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    cfg=json.loads((ROOT/'covariant_prepared_gauss_state_input_kappa400_sigma2000_r2100.json').read_text())
    selected=json.loads((ROOT/consumer.SELECTED).read_text());p=Parameters(**cfg['photon_parameters']);gamma,_=phi_frame(p)
    if consumer.digest(ROOT/'covariant_prepared_gauss_state.py')!=consumer.EXPECTED_SOURCE or consumer.digest(ROOT/'covariant_root_transfer.py')!=consumer.EXPECTED_ROOT:raise ValueError('Frozen source epoch changed')
    rows=[run_signed(gamma,sign,max(selected['inputs']['normal_nodes']),selected['inputs']['native_steps']) for sign in (1,-1)]
    result=dict(governing_plan_revision=consumer.PLAN_REVISION,source_sha256=hashlib.sha256(OWN_BYTES_AT_IMPORT).hexdigest(),
        source_consumer_import_sha256=hashlib.sha256(consumer.OWN_BYTES_AT_IMPORT).hexdigest(),
        frozen_compiled_source_hashes=source.COMPILED,
        Root_import_source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in source.root_transfer.SOURCE_BYTES_AT_IMPORT.items()},
        exact_Source2a_sha256=consumer.EXPECTED_SOURCE,exact_Root6911_sha256=consumer.EXPECTED_ROOT,
        selected_original_capsule_sha256=consumer.digest(ROOT/consumer.SELECTED),actual_signed_source_consumers=rows,
        original_all_sources_preparation_Pe_clocks_unchanged=True,
        scope='Actual unchanged Source2a selectedCliff2 signed quantum normal/integerGauss/Ref native epochs; NOT the independent complete4Dirac finiteNF source or macro family.',
        actual_macro_M0=None,actual_macro_M1=None,actual_macro_Pe_Gram=None,macro_source_admitted=False,elapsed_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(source.portable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),elapsed_seconds=result['elapsed_seconds'],
        actual_original_signed_Pe_Grams=[r['actual_original_energy_preparation']['total_source_retained_Gram'] for r in rows],
        supported_dimensions=[r['joint_supported_dimension'] for r in rows],macro_source_admitted=False)))


if __name__=='__main__':main()
