"""Actual same-source higher density-word boundary controls, not a jet rerun."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_initial_density_higher_jets as jets

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_initial_density_higher_jets_boundary_controls'


def run()->dict[str,Any]:
    start=time.monotonic()
    receipt_path=ROOT/'joined_phi_u1_initial_density_higher_jets_actual_receipt_20261010.json'
    receipt_raw=receipt_path.read_bytes();receipt=json.loads(receipt_raw)
    if receipt['source_sha256']!=jets.digest((ROOT/'joined_phi_u1_initial_density_higher_jets.py').read_bytes()):raise ArithmeticError('Coefficient producer changed')
    data,sp,inputs,_=jets.body.full_inputs();geo=jets.current.geometry(data,sp[0]);geom=jets.prior.prices.exact_geometry(data['native'],geo)
    engine=jets.Engine(data,sp,geo,geom,1000000)
    rows=[];cache={}
    for row in receipt['receiver_rows']:
        x=row['receiver']
        if x not in cache:
            terms=engine.initial(x)
            # Isolate the literal E^3(inner matter density commutator) component
            # of n4. This contains O*s^3 and is sensitive to auxiliary variance;
            # full second is necessarily blind to an auxiliary centered mean.
            for _ in range(3):
                out=defaultdict(complex)
                for (w,fs,ss),c in terms.items():
                    out[w,fs,ss]+=c*64*engine.winds[w]['w2']
                    out[w,fs,tuple(sorted(ss+(w,)))]+=c*128
                terms={k:v for k,v in out.items() if v}
            correct=engine.evaluate(terms,4)['real']
            variants={name:engine.evaluate(terms,4,**parameters)['real'] for name,parameters in {
                'wrong_auxiliary_load0':dict(copies=0),'wrong_Ref4_raw':dict(Ref_hole=False),
                'wrong_field_density_variance4':dict(field_scale=F(2))}.items()}
            controls={name:dict(value=value,defect=abs(value-correct)) for name,value in variants.items()}
            for name,control in controls.items():
                if control['defect']<1e-5:raise ArithmeticError('Actual higher density-word boundary control blind: '+name)
            cache[x]=dict(correct_E3_hop_fourth_component=correct,wrong_controls=controls,ordered_terms=len(terms))
        onsite=np.kron(np.eye(64),sp[0].onsite)
        number=np.zeros((256,256));number[4*x:4*x+4,4*x:4*x+4]=np.eye(4)
        defect=float(np.max(abs(onsite@number-number@onsite)))
        if defect!=0:raise ArithmeticError('Actual nohop onsite density commutator not zero')
        rows.append(dict(path=row['path'],receiver=x,length=row['length'],**cache[x],nohop_onsite_commutator_exact_defect=defect,
                         HE_HC_nohop_commutator='exact0: Pi acts in field factor and all site number operators commute'))
    parity=receipt['controls']['parity']
    if parity['wrong_single_channel_state_defect']<.01:raise ArithmeticError('Actual wrong single channel parity control blind')
    captures={**jets.body.CAPTURED,**inputs,receipt_path.name:receipt_raw}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captures.setdefault(Path(file).name,Path(file).read_bytes())
    for name,raw in captures.items():
        if (ROOT/name).read_bytes()!=raw:raise ArithmeticError('Boundary input/source changed '+name)
    archive=ROOT/(PREFIX+'_input_bytes_20261010.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captures.items()):z.writestr(name,raw)
    result: dict[str,Any]=dict(status='ACTUAL_HIGHER_DENSITY_WORD_BOUNDARY_CONTROLS_COMPLETE',arithmetic='conditional floating, not whole-kernel outward admission',actual12=rows,
                full_jet_receipt_sha256=jets.digest(receipt_raw),source_sha256=jets.digest(Path(__file__).read_bytes()),
                input_archive=dict(path=archive.name,sha256=jets.digest(archive.read_bytes()),files={n:jets.digest(v) for n,v in captures.items()}),
                wrong_single_channel_state_parity_defect=parity['wrong_single_channel_state_defect'],
                correct_n2_load_mean_blind='EXPECTED: auxiliary mean is centered0; the n4 E^3 word control tests its nonzero variance instead',
                wrong_HC32_actual_second_defects=[r['initial_coefficients']['2']['actual_wrong_controls']['wrong_HC32']['defect'] for r in receipt['receiver_rows']],
                wrong_Wick_plus_independent_defect=receipt['controls']['wrong_Wick_plus_defect'],
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)
    output=ROOT/(PREFIX+'_receipt_20261010.json');output.write_text(json.dumps(result,indent=2)+'\n')
    capsule=dict(numeric_exit=0,natural_exit=True,source_stable=True,source_sha256=result['source_sha256'],receipt=dict(path=output.name,sha256=jets.digest(output.read_bytes())),
                 elapsed_seconds=result['elapsed_seconds'],input_archive=result['input_archive'])
    (ROOT/(PREFIX+'_natural_exit_capsule_20261010.json')).write_text(json.dumps(capsule,indent=2)+'\n')
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):result=run()
    print(json.dumps(dict(status=result['status'],actual12=len(result['actual12']),elapsed_seconds=result['elapsed_seconds'])),flush=True)
