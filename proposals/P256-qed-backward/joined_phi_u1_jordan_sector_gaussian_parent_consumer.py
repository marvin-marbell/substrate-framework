"""Independent parent consumption: static exclusions, neutral band and mixed C.

Numerical band/covariance consumers are diagnostics, never Lambda residuals or
stationary poles. Fraction inequalities certify the static pressure endpoints.
"""
from __future__ import annotations
import argparse
import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
MANIFEST='joined_phi_u1_jordan_sector_gaussian_parent_manifest_20261010.json'


def audit211(data: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    a=F(profile['onsite']); rows=profile['paths']
    c0=a*a+2*sum(F(z['c0'])**2 for z in rows)
    sp=[2*F(z['ci'])**2 for z in rows]
    cert=data['exact_outward_certificate']; r=F(cert['r']); M=F(cert['support_cap'])
    c0sq=sum(F(z['c0'])**2 for z in rows); cisq=sum(F(z['ci'])**2 for z in rows)
    remainder=M-a-2*sum(abs(F(z['c0'])) for z in rows)
    assert remainder>=0 and remainder**2>=4*cisq*(1-r)
    defect=a*a*c0sq*r*r/M**3
    alpha=F(cert['rational_ALLangle_scalar_pressure_upper'])
    assert alpha*alpha>=c0*r+max(sp)*(1-r)
    assert (alpha+defect)**2>=c0
    d3=sum(z*z for z in sorted([c0,*sp],reverse=True)[:3])
    block=F(cert['two_block_lower']['lower']); scalar=F(cert['each_scalar_lower']['lower'])
    assert (-F(2,3)*block)**3>=256*d3
    assert (-F(4,3)*scalar)**3>=256*alpha**4
    lower=F(cert['ALLangle_sector_lower']['lower']); upper=F(cert['central_trial_upper'])
    assert lower<=block+2*scalar and lower>upper
    return {'static211_separation_exact':str(lower-upper),'separation_display':float(lower-upper),'support_and_radial_squared_cubed_checks':True}


def audit22(data: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    cert=data['certificate']; c0,cs,*other=map(F,cert['C_exact'])
    assert other==[cs,cs]
    d3=F(cert['D3_exact']); delta=F(cert['defect']['lower'])
    assert d3==c0*c0+2*cs*cs and delta<=F(old['routes'][-1]['defect']['lower'])
    n=int(cert['closed_cells']); assert n>=16384 and F(3,5)**2>F(1,3)
    maximum=F(0); previous=F(0)
    for i,row in enumerate(cert['cell_bounds']):
        lo,hi,p=F(row['lo']),F(row['hi']),F(row['pressure_upper'])
        assert row['index']==i and lo==previous==F(i,n) and hi==F(i+1,n)
        q=(p+delta*lo*lo)**2
        if lo<F(3,5):
            linear=(c0-cs)*hi; radical_squared=3*cs*cs
        else:
            linear=c0*hi; radical_squared=2*cs*cs*(1-lo*lo)
        assert q*q>=d3 or (q>=linear and (q-linear)**2>=radical_squared)
        maximum=max(maximum,p); previous=hi
    assert previous==1 and maximum==F(cert['alpha_upper'])
    lower=F(cert['lower']['lower']); upper=F(data['central_trial']['upper'])
    assert (-lower/3)**3>=256*maximum**4 and lower>upper
    return {'static22_separation_exact':str(lower-upper),'separation_display':float(lower-upper),'closed_cell_count':n,'independent_squared_cell_certificate':True,'radial_cube_checked':True}


def run(output: Path) -> dict[str, Any]:
    sys.set_int_max_str_digits(0); start=time.monotonic()
    raws={MANIFEST:(ROOT/MANIFEST).read_bytes(),Path(__file__).name:Path(__file__).read_bytes()}
    manifest=json.loads(raws[MANIFEST]); primary={}
    for name,ref in manifest['inputs'].items():
        source=(ROOT/ref['source']).read_bytes(); raw=(ROOT/ref['receipt']).read_bytes()
        assert hashlib.sha256(source).hexdigest()==ref['source_sha256']
        assert hashlib.sha256(raw).hexdigest()==ref['receipt_sha256']
        raws[ref['source']]=source; raws[ref['receipt']]=raw; primary[name]=json.loads(raw)
    for ref in manifest['supplemental']:
        path=Path(ref['path']); content=path.read_bytes()
        assert hashlib.sha256(content).hexdigest()==ref['sha256']
        raws[path.name]=content
    profile=json.loads(raws['joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json'])['primitive_profiles'][0]
    scalar=audit211(primary['211'],profile); joint=audit22(primary['22joint'],primary['22old'])
    bad=copy.deepcopy(primary['22joint'])
    row=max(bad['certificate']['cell_bounds'],key=lambda z:F(z['pressure_upper']))
    row['pressure_upper']=str(F(row['pressure_upper'])-F(1,1000))
    try:audit22(bad,primary['22old'])
    except AssertionError:adverse_joint=True
    else:raise AssertionError('Optimistic joint pressure accepted')
    with np.load(ROOT/'joined_phi_u1_neutral_ground_graph_final_20261010.npz',allow_pickle=False) as z:
        h,p=z['h_unscaled_plus'],z['Pplus']; w=z['Wplus']; wm=z['Wminus']
        assert h.shape==(256,256) and w.shape==(256,128)
        projector_defect=float(np.linalg.norm(p@p-p)); band_defect=float(np.linalg.norm(h@p-p@h))
        energy=-float(np.trace(p@h).real); wrong=-float(np.trace((np.eye(256)-p)@h).real)
        assert max(projector_defect,band_defect)<1e-9 and energy<0<wrong
        off=float(np.linalg.norm(wm.conj().T@h@w)**2)
        assert abs(energy-primary['neutral']['neutral']['energy_over_Lambda'])<1e-8
    gaussian=primary['gaussian']; peername='joined_phi_u1_neutral_ground_graph_epoch2_20261010.npz'
    assert hashlib.sha256(raws[peername]).hexdigest()==gaussian['input_sha256'][peername]
    with np.load(ROOT/peername,allow_pickle=False) as peer, np.load(ROOT/'joined_phi_u1_jordan_full_gaussian_delivered_20261010.npz',allow_pickle=False) as cov:
        grams=peer['neutral_derivative_Gram_all64']; assert grams.shape==(64,90,90)
        records=gaussian['actual_full64_neutral_covariance_consumer']['all64']; q0_cross=None; error=0.
        for index,row in enumerate(records):
            gram=grams[index]; assert np.linalg.eigvalsh(gram)[0]>-1e-9
            key='_'.join(map(str,row['q_integer']))
            for entry in row['full90_covariance_derivative_consumer']:
                c=cov[key+f"_C_t{entry['t_degree']}_divided_2powminus7168"]
                value=float(np.trace(gram@c).real)
                cross=float((np.trace(gram[:45,45:]@c[45:,:45])+np.trace(gram[45:,:45]@c[:45,45:])).real)
                scale=max(1.,abs(value)); error=max(error,abs(value-entry['full90_Tr_derivativeGram_C'])/scale)
                assert abs(value-entry['full90_Tr_derivativeGram_C'])<1e-8*scale
                if index==0 and entry['t_degree']==1:q0_cross=cross
        assert q0_cross is not None and abs(q0_cross)>.1
    action=gaussian['new_preparation_action']; assert not action['stationary_residual_certified']
    assert float(action['uncancelled_forcing_sum']['value'])>1e9
    frob=primary['frobenius']; global_data=primary['global']
    assert frob['source_inputs_unchanged'] and global_data['source_inputs_unchanged']
    bounds=frob['global_ground_budgets']
    archive=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(raws.items()):z.writestr(name,raw)
    assert all((ROOT/name).read_bytes()==raw for name,raw in raws.items())
    result={'stage':'SCOPED_STATIC_EXCLUSIONS_AND_GLOBAL_FORM_PLUS_CONDITIONAL_GAUSSIAN_BAND_CONSUMER_NOT_CAMPAIGN_CLOSE','captured_plan':manifest['plan'],'static211':scalar,'static22':joint,'optimistic_joint_cell_rejected':adverse_joint,'neutral_FLOAT_diagnostic':{'energy_over_Lambda':energy,'wrong_negative_sheet_energy_over_Lambda':wrong,'binary_offband_residual_over_Lambda2':off,'projector_defect':projector_defect,'commutator_defect':band_defect,'no_Lambda_amplified_residual_admitted':True},'Gaussian_FLOAT_series_consumer':{'peer_epoch':'epoch2 retained, NOT rebound to final neutral epoch','all64_full90_PC_CP_retained':True,'relative_trace_difference':error,'q0_degree1_PC_CP':q0_cross,'wrong_drop_PC_CP_loses_nonzero_component':True,'no_interval_or_total_action_admission':True},'global_form_budgets':bounds,'scope_limits':['31 gap22.66809136 and off-flat/globalground characterization remain','neutral fixedsection Pplus exactfunctional, stored floatingmatrices onlydiagnostics','new fullGaussian exactconditional AGA=K; no coefficient enclosures/rootforce/Hermitefullaction','photon firstgraph highfrequency/nonstationary only','stationary charged/PiHC/fullPhi ground and sameground spatial photon/pole/commoncone/originalSOURCE/campaign OPEN'],'source_input_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in raws.items()},'immutable_input_archive':{'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},'source_inputs_unchanged':True,'elapsed_seconds':time.monotonic()-start}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    with threadpool_limits(limits=1):result=run(parser.parse_args().output)
    print(json.dumps({'stage':result['stage'],'static211':result['static211'],'static22':result['static22'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
