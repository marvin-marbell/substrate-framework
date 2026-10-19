#!/usr/bin/env python3
"""Full ordinary-CAR Coulomb bound; no projected-only or single-PH closure."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any,cast
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from prepared_neutral_shared_photon_state import load_shared_native_cache


def density_square_max(volume: int,light_count: int,reference_count: int) -> dict[str,Any]:
    """Exact occupation DP; rho=n_light-2+(Nref-4) at the reference site.

    Every four-colour local number is 0..4 by ordinary CAR. The heavy sector
    is EM neutral. This maximizes over the ENTIRE fixed-number Fock sector.
    """
    table={0:(0,())}
    for _ in range(volume-1):
        next_table={}
        for count,(value,witness) in table.items():
            for occupation in range(5):
                new_count=count+occupation;score=value+(occupation-2)**2
                if new_count not in next_table or score>next_table[new_count][0]:
                    next_table[new_count]=(score,(*witness,occupation))
        table=next_table
    candidates=[]
    for reference_light in range(5):
        count=light_count-reference_light
        if count in table:
            value,witness=table[count]
            candidates.append((value+(reference_light-2+reference_count-4)**2,
                (*witness,reference_light)))
    if not candidates:raise ValueError('Impossible ordinary-CAR particle-number sector')
    maximum,witness=max(candidates)
    rho=np.asarray(witness,dtype=int)-2
    rho[-1]+=reference_count-4
    if int(rho@rho)!=maximum or sum(witness)!=light_count or int(rho.sum())!=light_count+reference_count-2*volume-4:
        raise AssertionError('Exact density witness inconsistent')
    return dict(max_density_square=maximum,light_count=light_count,reference_count=reference_count,
        witness_occupations=list(witness),witness_density=rho.tolist(),total_charge=int(rho.sum()))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-npz',type=Path,required=True);p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--duration',type=float,default=.031)
    args=p.parse_args();model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    if model.law.size!=3:raise ValueError('Exact native Green eigenvalue1/3 proof is N3-only')
    ue=Fraction.from_float(float(model.ue));green_bound=Fraction(1,3)
    neutral_counts={q:len(model.fibre.occupied_indices[q]) for q in (1,-1)}
    sectors={'neutral_plus':(neutral_counts[1],4),'neutral_minus':(neutral_counts[-1],0),
             'charged_plus':(neutral_counts[1]+1,3),'charged_minus':(neutral_counts[-1]-1,1)}
    clock=566;t0=.37*clock;actual_duration=Fraction.from_float((t0+args.duration)-t0)
    rows={}
    for name,(light,reference) in sectors.items():
        row=density_square_max(model.law.volume,light,reference)
        bound=ue*green_bound*row['max_density_square']
        # Witness is in site occupation basis. Move its final entry to FAR.
        rho=np.asarray(row['witness_density'],float)
        rho[[model.reference_site,-1]]=rho[[-1,model.reference_site]]
        energy=float(model.ue*(rho@model.green@rho))
        wrong=rho.copy();wrong[model.reference_site]+=4
        if energy< -1e-14 or energy>float(bound)+1e-14 or row['total_charge']!=0:
            raise AssertionError('Full number-sector Coulomb witness violated analytic bound')
        row.update(operator_norm_bound_exact=str(bound),operator_norm_bound=float(bound),
            short_interval_Duhamel_norm_bound_exact=str(actual_duration*bound),
            short_interval_Duhamel_norm_bound=float(actual_duration*bound),
            native_clock566_Duhamel_norm_bound=float(clock*bound),
            native_witness_HC_energy=energy,wrong_missing_Ref_background_total_charge=float(wrong.sum()))
        rows[name]=row
    measured=np.linalg.eigvalsh(model.green)
    if abs(float(measured[-1])-1/3)>1e-12 or np.linalg.norm(model.green@np.ones(model.law.volume))>1e-12:
        raise AssertionError('Actual original Green differs from N3 lattice Lplus')
    sources={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest() for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).parent==Path(__file__).parent}
    result=dict(HC_law='UE rho^T Lplus rho; NO factor1/2',density_law='n_x-2+(Nref-4)delta_xR',
        ordinary_CAR_modes=4*model.law.volume+4,heavy_EM_neutral=True,
        sectors=rows,UE_exact=str(ue),UE=float(model.ue),Green_operator_bound_exact='1/3',
        Green_numeric_eigenvalues_retained=[float(measured[0]),float(measured[-1])],
        actual_float_interval_duration_exact=str(actual_duration),clock=clock,
        proof='Commuting ordinary-CAR densities; Lplus PSD and norm1/3; exact fixed-number rho-square DP. Unitary Duhamel bound uniform in all photon/Phi wavefunctions and all Coulomb multipair channels, conditional on common selfadjoint evolution.',
        native_full_clock_bound_above_1e_minus4_target_NOT_actual_failure=any(r['native_clock566_Duhamel_norm_bound']>1e-4 for r in rows.values()),
        no_Coulomb_weak_matrix_compile_or_photon_Fock_crop=True,no_H_G_preparation_or_loop_rerun=True,
        legal_N_general_upper='UE * N^2/16 * (4N^3+32), ordinaryCAR only; legal parameters NOT repriced here',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),loaded_sources=sources,
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        scope='Uniform fullordinary112-CAR nativeN3 Coulomb operator/Duhamel bound, not full quantumH propagation/legal-N/encodedRootCAR/Born')
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True));print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
