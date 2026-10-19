#!/usr/bin/env python3
"""Independent a-posteriori bound avoiding floating Galerkin orthogonality claims.

For approximate z,zx,zy,zxy with coupled-equation residuals r_j, errors satisfy
 B e=r, B ex=rx-Ax e, B ey=ry-Ay e,
 B exy=rxy-Axy e-Ax ey-Ay ex.
At t>0 these give ||e||<=||r||/t and the recursively propagated bounds below.
They remain valid without EXACT numerical Qdag r=0, unlike the quadratic
Galerkin energy identity. Float residual evaluation is not interval-certified.
This review NEVER promotes a tiny quadratic estimate into machine accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(__file__).parent;own=Path(__file__).read_bytes()
    receipt=root/'physical_alpha0226_weighted_physical_jet_bounds_capsule.json'
    data=json.loads(receipt.read_text());vectors=np.load(data['vectors']['path'])
    point=root/'physical_alpha0226_inverse_adapted_source_point_capsule.npz';source=np.load(point)
    mn={k:float(np.linalg.norm(source['source'+k])) for k in ('','x','y','xy')}
    upper=data['actual_complete_operator_jet_norm_upper'];bx,by,bxy=(upper[k] for k in ('x','y','xy'))
    cases=[]
    for case in data['actual_weighted_jet_cases']:
        t=case['shift'];prefix='shift_'+str(t).replace('.','p')+'_'
        q=vectors[prefix+'Q'];r={k:vectors[prefix+'residual'+k] for k in ('','x','y','xy')}
        norms={k:float(np.linalg.norm(v)) for k,v in r.items()}
        errors={'':norms['']/t}
        errors['x']=(norms['x']+bx*errors[''])/t;errors['y']=(norms['y']+by*errors[''])/t
        errors['xy']=(norms['xy']+bxy*errors['']+bx*errors['y']+by*errors['x'])/t
        response={'':mn['']*errors[''],
                  'x':mn['x']*errors['']+mn['']*errors['x'],
                  'y':mn['y']*errors['']+mn['']*errors['y'],
                  'xy':mn['xy']*errors['']+mn['x']*errors['y']+mn['y']*errors['x']+mn['']*errors['xy']}
        orth={k:float(np.linalg.norm(q.conj().T@v)) for k,v in r.items()}
        # Bound projected solution jets using actual same-family source norms.
        zb={'':mn['']/t};zb['x']=(mn['x']+bx*zb[''])/t;zb['y']=(mn['y']+by*zb[''])/t
        zb['xy']=(mn['xy']+bxy*zb['']+bx*zb['y']+by*zb['x'])/t
        guards={'':zb['']*orth[''],
                'x':zb['x']*orth['']+zb['']*orth['x'],
                'y':zb['y']*orth['']+zb['']*orth['y'],
                'xy':zb['xy']*orth['']+zb['x']*orth['y']+zb['y']*orth['x']+zb['']*orth['xy']}
        energy=case['repaired_fullcallback_energy_jet_bound']
        guarded={'':energy['energy_upper']+guards[''],
                 'x':energy['first_x_upper']+guards['x'],
                 'y':energy['first_y_upper']+guards['y'],
                 'xy':energy['mixed_absolute_upper']+guards['xy']}
        cases.append({'shift':t,'whole_residual_norms':norms,'observed_Qdag_residual_jet_norms':orth,
          'original_source_jet_complement_norms':{k:float(np.linalg.norm(source['source'+k]-q@(q.conj().T@source['source'+k]))) for k in ('','x','y','xy')},
          'log_reference_source_complement_debt':'Finite resolvent energy identity requires no source-capture assumption. A compressed log reference term additionally needs these original source jet complements priced; it is not certified here.',
          'solution_error_norm_upper':errors,'source_response_absolute_error_upper_without_exact_orthogonality':response,
          'Galerkin_identity_correction_upper':guards,'guarded_source_energy_jet_error_upper':guarded,
          'recommended_reported_error_upper':{k:min(response[k],guarded[k]) for k in errors},
          'scope':'Conditional exact-arithmetic residual/norm estimates at the named finite shift; no directed-rounding, t=0/infinity or fullphysical transport certificate.'})
    files=[receipt,Path(data['vectors']['path']),point]
    report={'derivation':__doc__,'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'source_jet_norms':mn,'cases':cases,
            'execution':{'source_sha256':hashlib.sha256(own).hexdigest(),'source_bytes':own.decode(),'executable':sys.executable,
                         'python':sys.version,'numpy':np.__version__,'runtime':platform.platform(),'startup_unchanged':Path(__file__).read_bytes()==own,
                         'natural_exit':'External receipt must bind observed returncode.'}}
    output=Path(args.output);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'capsule':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'recommended_error_uppers':[c['recommended_reported_error_upper'] for c in cases]}))

if __name__=='__main__':main()
