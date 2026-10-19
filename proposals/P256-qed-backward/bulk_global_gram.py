#!/usr/bin/env python3
"""Existing convex Gram theorem on the actual whole-heavy periodic bulk law.

Global uniform all60-field minimum only; not a nonuniform/quantum phase proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__=='__main__':
    for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
        os.environ[variable]='1'

import numpy as np

from periodic_bulk_frame import PeriodicFunctional, source_hashes as inherited_source_hashes
from rotation_band import RotationBand
from sea_selection import SELECTED, gram, matrix_unit_basis, rank_report
from soft_budget import elastic, select_gram
from unrestricted_field_minimum import ALIGNMENT, BARE, STIFFNESS, check, coordinates, matrices

ROOT=Path(__file__).resolve().parent


def source_hashes():
    return {**inherited_source_hashes(),**{
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('soft_budget.md','bulk_global_gram.py','bulk_global_gram.md')}}


def assess(data, fields, minimum) -> dict[str, Any]:
    value=coordinates(fields,data.basis).reshape(-1)
    energy,gradient=data.evaluate(value)
    G=gram(fields)
    radii=np.sqrt(np.einsum('pm,mn,pn->p',data.x,G,data.x))
    lower=elastic(G,STIFFNESS,0.)-2*float(radii.mean())
    spectral_slack=energy-lower
    gram_distance=float(np.linalg.norm(G-minimum['G']))
    strong_convexity=STIFFNESS/2*gram_distance**2
    if spectral_slack< -2e-10 or lower-minimum['value']-strong_convexity < -2e-10:
        raise AssertionError('Actual global-half energy/strong Gram bound failed')
    eig,_,occupation,gap=data.spectral(value)
    counts=occupation.sum(axis=1)
    return dict(actual_global_half_energy=energy,Gram_lower_functional=lower,
        reference_uniform_global_minimum_numerical=minimum['value'],
        spectral_slack=spectral_slack,Gram_distance_to_optimum=gram_distance,
        strong_convexity_remainder=strong_convexity,
        energy_excess=energy-minimum['value'],actual_full60_gradient_norm=float(np.linalg.norm(gradient)),
        block_occupation_histogram={str(k):int(np.sum(counts==k)) for k in range(5)},
        global_occupied_empty_gap=gap,per_block_positive_histogram={str(k):int(np.sum((eig>0).sum(axis=1)==k)) for k in range(5)})


def run(input_receipt:Path,*,grids):
    identities=source_hashes()
    payload=input_receipt.read_bytes()
    receipt=json.loads(payload)
    if receipt['bare_coefficients']!={**BARE,'stiffness':STIFFNESS,'g0':0.,'Jpref':ALIGNMENT}:
        raise ValueError('Actual shared periodic bare law required')
    if receipt['source_sha256']['sea_selection.py']!=identities['sea_selection.py']:
        raise ValueError('Input actual coordinate basis changed')
    band=RotationBand(**BARE);basis=matrix_unit_basis()
    rng=np.random.default_rng(222060)
    rows=[]
    for size in grids:
        data=PeriodicFunctional(band,size,basis)
        solution=select_gram(data.x,STIFFNESS,0.)
        fields=np.sqrt(np.diag(solution['G']))[:,None,None]*SELECTED
        attainment=assess(data,fields,solution)
        check(abs(attainment['spectral_slack']),'actual Clifford global-half Gram saturation',3e-10)
        check(abs(attainment['energy_excess']),'actual global-uniform Gram attainment',3e-10)
        radii=np.sqrt(np.einsum('pm,mn,pn->p',data.x,solution['G'],data.x))
        matrix_gradient=STIFFNESS*solution['G']-np.einsum('pm,pn,p->mn',data.x,data.x,1/radii)/data.count
        residual=float(np.linalg.norm(matrix_gradient))
        check(residual,'full PSD Gram stationarity, including offdiagonal coordinates',3e-10)
        comparisons=[]
        for branch in receipt['same_law_seed_branches']:
            candidate=branch['validation_stationary_candidate']
            actual=matrices(np.asarray(candidate['coordinate_values']),basis)
            comparisons.append(dict(branch=branch['name'],input_stationary_grid=candidate['grid'],
                result=assess(data,actual,solution)))
        probes=[]
        for _ in range(12):
            actual=matrices(rng.normal(size=(4,15))*.35,basis)
            probes.append(assess(data,actual,solution))
        rank=rank_report(data.x)
        if rank['quadratic_rank']!=10:
            raise AssertionError('Actual input lacks rank10 needed to infer Clifford equality')
        # One large positive temporal eigenvalue is occupied in every block;
        # the next two bands transfer particles to higher-x0 blocks. The fixed
        # TOTAL half filling therefore does not impose two in each block.
        wrong_block_rule=np.zeros((4,4,4),dtype=complex)
        wrong_block_rule[0]=np.diag([5.,1.,1.,-7.])
        adverse=assess(data,wrong_block_rule,solution)
        value=coordinates(wrong_block_rule,basis).reshape(-1)
        eigenvalues,_,_,_=data.spectral(value)
        forced=elastic(gram(wrong_block_rule),STIFFNESS,0.)-float(eigenvalues[:,2:].sum())/data.count
        adverse['incorrect_forced_two_block_energy']=forced
        adverse['wrong_block_rule_energy_defect']=forced-adverse['actual_global_half_energy']
        if (adverse['block_occupation_histogram']['2']==data.count
                or adverse['wrong_block_rule_energy_defect']<=1e-5):
            raise AssertionError('Actual global-filling adverse block-transfer case not reached')
        rows.append(dict(grid=size,momenta=data.count,Gram=solution['G'].tolist(),
            minimum_energy_numerical=solution['value'],full_Gram_gradient_norm=residual,
            analytic_energy_error_upper_from_exact_residual=residual**2/(2*STIFFNESS),
            residual_error_formula_status='floating evaluation, not interval certification',
            input_richness=rank,actual_attainment=attainment,input_branch_comparisons=comparisons,
            unrestricted_all60_probes=probes,actual_non_two_block_global_occupation=adverse,
            coordinate_values=coordinates(fields,basis).tolist()))
    if source_hashes()!=identities or input_receipt.read_bytes()!=payload:
        raise RuntimeError('Gram supplier/proof/input changed during execution')
    return dict(source_sha256=identities,source_bindings_unchanged=True,
        input_binding=dict(path=str(input_receipt),sha256=hashlib.sha256(payload).hexdigest()),
        theorem_owner='soft_budget.select_gram and its existing convex Gram proof; actual source-owned periodic whole-heavy global-half consumer',
        bare_coefficients=receipt['bare_coefficients'],rows=rows,all_checks_pass=True,
        scope='Exact global UNIFORM all60 field reduction at identity links/selected periodic meshes; no nonuniform link or quantum phase',
        limits=['Full PSD Gram minimization is a proved convex reduction, not a Hessian-only global claim.',
            'Actual source basis and fixed TOTAL half filling; no absolute-value energy replacement or forced two-per-block filling.',
            'Each finite quadrature has its own unique optimum; no continuum interval enclosure supplied.',
            'No nonuniform global minimum, normal quantum confinement, charged gap, compact phase or campaign completion.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-receipt',type=Path,default=ROOT/'periodic_bulk_frame_receipt.json')
    parser.add_argument('--grids',type=int,nargs='+',default=[8,16,24,32])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.grids or any(size<3 for size in args.grids):
        raise ValueError('Rich cubic midpoint grids>=3 required')
    result=run(args.input_receipt,grids=args.grids)
    text=json.dumps(result,indent=2,allow_nan=False)
    args.output.write_text(text+'\n');print(text)


if __name__=='__main__':
    main()
