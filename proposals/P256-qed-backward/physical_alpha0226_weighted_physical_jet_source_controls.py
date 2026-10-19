#!/usr/bin/env python3
"""Individual actual-chart physical transport and pulled-source jet controls.

This is a NEW omission discrimination at persisted ORIGINAL M64 chart inputs,
not another validation of completed integer/FD runs. No fulloperator retag.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import scipy
import scipy.linalg
import physical_alpha0226_physical_log_jets as jets
import physical_alpha0226_source_receiver as receiver


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(__file__).parent;own=Path(__file__).read_bytes()
    input_path=root/'physical_alpha0226_inverse_adapted_source_point_capsule.npz'
    z=np.load(input_path);a,ax,ay,axy=(z[k] for k in ('A','Ax','Ay','Axy'))
    physical=jets.physical_range_jets(a,z['Cx'],z['Cy'],z['Cxy'],z['Dxy'],ax,ay,axy)
    source={k:z['Q'].conj().T@z['source'+k] for k in ('','x','y','xy')}
    values,frame=np.linalg.eigh(jets.hermitian(a));s=(frame*np.sqrt(values))@frame.conj().T;inv=(frame/np.sqrt(values))@frame.conj().T
    sx=scipy.linalg.solve_sylvester(s,s,ax);sy=scipy.linalg.solve_sylvester(s,s,ay)
    sxy=scipy.linalg.solve_sylvester(s,s,axy-sx@sy-sy@sx)
    ix=-inv@sx@inv;iy=-inv@sy@inv
    ixy=inv@sx@inv@sy@inv+inv@sy@inv@sx@inv-inv@sxy@inv
    m,mx,my,mxy=(source[k] for k in ('','x','y','xy'))
    n=inv@m;nx=ix@m+inv@mx;ny=iy@m+inv@my
    ox,oy,qxy=(physical[k] for k in ('Omega_x','Omega_y','Pancharatnam_Qxy'))
    sigma=qxy-(ox@oy+oy@ox)/2
    source_x={'inverse_first':ix@m,'original_source_first':inv@mx,'Omega_x_source':ox@n}
    source_y={'inverse_first':iy@m,'original_source_first':inv@my,'Omega_y_source':oy@n}
    source_xy={'inverse_second':ixy@m,'inverse_x_source_y':ix@my,'inverse_y_source_x':iy@mx,'original_source_second':inv@mxy,
               'Omega_x_inverse_and_source_y':ox@ny,'Omega_y_inverse_and_source_x':oy@nx,
               'sigma_xy_source':sigma@n,'BOTH_Omega_products_source':(ox@oy+oy@ox)@n/2}
    pulled={'':n,'x':sum(source_x.values()),'y':sum(source_y.values()),'xy':sum(source_xy.values())}
    supplier=receiver.pulled_original_source_jets(a,ax,ay,axy,source,physical)
    sigmafree={**physical,'physical_xy':physical['physical_xy']-(sigma@physical['H']-physical['H']@sigma)}
    sourcefrozen=receiver.pulled_original_source_jets(a,ax,ay,axy,{'':m,'x':np.zeros_like(mx),'y':np.zeros_like(my),'xy':np.zeros_like(mxy)},physical)
    allpulledfrozen={k:np.zeros_like(v) if k else v for k,v in pulled.items()}
    times=[]
    for t in (.015,.025):
        correct=receiver.connected_survival(physical,pulled,t)
        omitted_source=receiver.connected_survival(physical,sourcefrozen,t)
        omitted_pull=receiver.connected_survival(physical,allpulledfrozen,t)
        omitted_sigma=receiver.connected_survival(sigmafree,pulled,t)
        times.append({'observation_time':t,'complete_chart':correct,'wrong_freeze_original_midpoint_source':omitted_source,
                      'wrong_freeze_entire_pulled_source':omitted_pull,'wrong_omit_sigma_contact_only':omitted_sigma,
                      'discriminated_connected_mixed_differences':{'freeze_original_source':correct['connected_log_mixed']-omitted_source['connected_log_mixed'],
                          'freeze_pulled_source':correct['connected_log_mixed']-omitted_pull['connected_log_mixed'],
                          'omit_sigma':correct['connected_log_mixed']-omitted_sigma['connected_log_mixed']}})
    files=[input_path,Path(jets.__file__),Path(receiver.__file__)]
    report={'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'source_channel_norms':{'x':{k:float(np.linalg.norm(v)) for k,v in source_x.items()},
                                    'y':{k:float(np.linalg.norm(v)) for k,v in source_y.items()},
                                    'xy':{k:float(np.linalg.norm(v)) for k,v in source_xy.items()}},
            'source_channel_sum_defects':{k:float(np.linalg.norm(v-supplier[k])) for k,v in pulled.items()},
            'horizontal_range_contact_channel':{
                'crossmetric_norm':float(np.linalg.norm(physical['horizontal_range_crossmetric'],2)),
                'ONE_kernel_energy_unit_mixed_contact_price_norm':float(np.linalg.norm(physical['ONE_kernel_energy_unit_mixed_contact_price'],2)),
                'scope':'Original positive-range horizontal channel retained as a price. No arbitrary null/kernel-energy extension is adopted; full horizontal complement is unpaid.'},
            'noninteger_observed_omission_controls':times,
            'scope':'NEW physical chart and differentiated original pulled-source control only. Not whole44336 physical operators, and chart noninteger agreement/controls do not remove weighted full-complement or range-transport debts.',
            'execution':{'source_sha256':hashlib.sha256(own).hexdigest(),'source_bytes':own.decode(),'executable':sys.executable,'python':sys.version,
                         'numpy':np.__version__,'scipy':scipy.__version__,'runtime':platform.platform(),'startup_unchanged':own==Path(__file__).read_bytes(),
                         'natural_exit':'Parent wrapper must bind observed returncode.'}}
    output=Path(args.output);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'capsule':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'controls':[x['discriminated_connected_mixed_differences'] for x in times]}))

if __name__=='__main__':main()
