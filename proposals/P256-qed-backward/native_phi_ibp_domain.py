#!/usr/bin/env python3
"""Audit the frozen Phi14 Gaussian IBP rule without changing H/state/Grams.

A smooth compact wave has zero true boundary term.  An empirical Phi14
rule need not integrate its divergence, even for a linear scalar witness.
This audit is not a full-field integral or a repaired quantum generator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from scipy.special import gammaincc
from prepared_neutral_ambient_weak_state import compact_radial
from raw_source_relative_noether_graph import runtime_binding


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cache-npz','cache-manifest','field-npz','field-receipt','output','runtime-archive'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    manifest=json.loads(args.cache_manifest.read_bytes())
    field=json.loads(args.field_receipt.read_bytes())
    if sha(args.cache_npz)!=manifest['npz_sha256'] or field['cache_npz_sha256']!=sha(args.cache_npz) or field['cache_manifest_sha256']!=sha(args.cache_manifest):
        raise ValueError('IBP witness requires the SAME frozen state cache')
    if sha(args.field_npz)!=field['export_npz_sha256']:
        raise ValueError('Actual Y0 component export changed')
    config=dict(manifest['seed_snapshot']['original']['dict'])
    sigma,radius=float(config['sigma']),float(config['radius'])
    with np.load(args.field_npz) as exported:
        y0=exported['Y0']
    with np.load(args.cache_npz) as cache:
        directions=cache[config['directions']['array']]
    dimension=len(y0)
    # Exact q-only subexpression of frozen_nodes; no basis/H/preparation rerun.
    q7=np.random.default_rng(222).standard_normal((7,dimension))*sigma/np.sqrt(2.)
    q=np.stack((-q7,q7),axis=1).reshape(14,dimension)
    cutoff_and_gradient=[compact_radial(node,radius) for node in q]
    cut=np.array([value for value,_ in cutoff_and_gradient])
    grad=np.array([value for _,value in cutoff_and_gradient])
    weight=cut**2/14
    projection=q@y0
    norm2=float(y0@y0)
    gaussian_second=float(weight@(projection**2))
    empirical_witness=float(weight@(norm2-2*projection**2/sigma**2)+np.sum(2*cut*(grad@y0)*projection)/14)
    full_gaussian_second=sigma**2*norm2/2
    # Independent Gauss-Hermite positive control for the same scalar identity.
    nodes,weights=np.polynomial.hermite.hermgauss(3)
    control_projection=sigma*np.sqrt(norm2)*nodes
    control=float(weights@(norm2-2*control_projection**2/sigma**2)/np.sqrt(np.pi))
    if abs(control)>1e-12*max(1.,norm2):
        raise AssertionError('Gaussian IBP positive control fails')
    if np.any(cut!=1.) or np.any(grad!=0.):
        raise AssertionError('Claim that all frozen Phi14 nodes lie in flat-cutoff region fails')
    scalar_matrix=2*(q@directions).T@(weight[:,None]*(q@directions))/sigma**2
    binding=runtime_binding(args.runtime_archive)
    result=dict(plan=field['plan'],revision=field['revision'],work_item='soft-spatial-closed-rotation',
        source_sha256=sha(Path(__file__)),runtime_binding=binding,
        inputs=dict(cache_npz=sha(args.cache_npz),cache_manifest=sha(args.cache_manifest),field_npz=sha(args.field_npz),field_receipt=sha(args.field_receipt)),
        science=dict(normal_dimension=dimension,sigma=sigma,radius=radius,mean_Y0_norm_squared=norm2,
            q_only_sampling='literal frozen_nodes first RNG subexpression seed222/seven full-normal samples with antipodes; not a supplier execution',
            node_norm_min=float(np.min(np.linalg.norm(q,axis=1))),node_norm_max=float(np.max(np.linalg.norm(q,axis=1))),
            flat_cutoff_radius=radius/2,all14_cutoffs_exact_one=True,all14_cutoff_gradients_exact_zero=True,
            observed_Y0_Gaussian_second_moment=gaussian_second,continuous_full_Gaussian_second_moment=full_gaussian_second,
            observed_over_continuous_second_moment=gaussian_second/full_gaussian_second,
            actual_Phi14_linear_scalar_IBP_defect=empirical_witness,independent_GaussHermite_IBP_control=control,
            witness='f(q)=q.Y0; integral of Y0.grad[f eta^2 exp(-q^2/sigma^2)] is zero for smooth compact eta',
            cutoff_boundary_semantics='eta and all derivatives vanish at radius; no true surface term on smooth native chart support. Phi14 samples are entirely inside flat eta=1 region.',
            density_semantics='rho from measure cancels wave Jhalf squared; diagnostic uses the unchanged scalar Gaussian sampling measure, not a new state norm',
            underlying_Gaussian_probability_outside_flat_cutoff=float(gammaincc(dimension/2,(radius/(2*sigma))**2)),
            tail_probability_semantics='exact radial probability for uncut isotropic native Gaussian only; not a SOURCE/full-L2 error bound',
            selected_six_polynomial_empirical_second_moment=scalar_matrix.tolist(),
            selected_six_second_moment_minus_identity_Frobenius=float(np.linalg.norm(scalar_matrix-np.eye(6))),
            sample_covariance_rank_upper_bound=7,normal_dimension_not_reduced=True,
            conclusion='Finite Phi14 rule violates scalar IBP despite an identically zero smooth compact support boundary. This discriminates empirical quadrature/domain assumptions, not an operator correction or a sole historical-loop cause.',
            no_whitening_or_covariance_repair=True,no_H_G_preparation_loop_rerun=True,no_state_or_SOURCE_normalization_reset=True,
            no_full_native_L2_or_legal_SOFT_claim=True,campaign_open=True),
        argv=sys.argv,wall_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    main()
