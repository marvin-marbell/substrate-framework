"""Actual source-sensitive wrong sharedTheta/averaged-Slater adversaries."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction as F
import json
import math
from pathlib import Path
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_initial_density_higher_jets as jets

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_initial_density_higher_jets_shared_field_raw_mixture_controls'


def electric_word(engine,x):
    terms=engine.initial(x)
    for _ in range(3):
        out=defaultdict(complex)
        for (w,fs,ss),c in terms.items():
            out[w,fs,ss]+=c*64*engine.winds[w]['w2']
            out[w,fs,tuple(sorted(ss+(w,)))]+=c*128
        terms={k:v for k,v in out.items() if v}
    return terms


def run()->dict[str,Any]:
    start=time.monotonic()
    oldpath=ROOT/'joined_phi_u1_initial_density_higher_jets_boundary_controls_receipt_20261010.json'
    oldraw=oldpath.read_bytes();old=json.loads(oldraw)
    if old['full_jet_receipt_sha256']!=jets.digest((ROOT/'joined_phi_u1_initial_density_higher_jets_actual_receipt_20261010.json').read_bytes()):raise ArithmeticError('Conditional epoch changed')
    data,sp,inputs,_=jets.body.full_inputs();geo=jets.current.geometry(data,sp[0]);geom=jets.prior.prices.exact_geometry(data['native'],geo)
    correct=jets.Engine(data,sp,geo,geom,1000000);wrong=jets.Engine(data,sp,geo,geom,1000000)
    average=2*sum((weight*p for weight,p,_ in correct.states[1:]),np.zeros_like(sp[0].p))
    wrong.states=[correct.states[0]]+[(.5 if a==0 else 0.,average,True) for a in range(4)]
    wrong.complements=[np.eye(256)-p for _,p,_ in wrong.states]+[np.eye(256)-wrong.aux]
    rows=[];cache={}
    for row in old['actual12']:
        x=row['receiver']
        if x not in cache:
            terms=electric_word(wrong,x)
            value=wrong.evaluate(terms,4)['real'];expected=row['correct_E3_hop_fourth_component'];defect=abs(value-expected)
            if defect<1e-5:raise ArithmeticError('Actual averaged-Slater n4 control blind')
            phase_control=None
            for j in correct.incident[x]:
                w=correct.native_wind[j]
                for k in range(len(correct.k)):
                    active_wind=correct.add_wind(w,correct.native_wind[k])
                    ww=correct.add_wind(active_wind,w)
                    independent_q2=correct.winds[active_wind]['q2']+correct.winds[w]['q2']
                    if correct.winds[ww]['q2']==independent_q2:continue
                    factors=(correct.native[0][j],correct.native[0][k],correct.native[1][j])
                    car=sum(weight*correct.role_moment(state,factors) for state,(weight,_,_) in enumerate(correct.states))
                    joint=math.exp(-float(correct.winds[ww]['q2']))*car
                    independent=math.exp(-float(independent_q2))*car
                    if abs(joint-independent)>1e-12:
                        phase_control=dict(active_native_paths=[j,k],auxiliary_native_path=j,
                                           shared_character_q_squared_exact=str(correct.winds[ww]['q2']),
                                           independent_character_sum_q_squared_exact=str(independent_q2),
                                           correct_joint_real=joint.real,correct_joint_imaginary=joint.imag,
                                           wrong_independent_real=independent.real,wrong_independent_imaginary=independent.imag,
                                           defect=abs(joint-independent),auxiliary_load_exact=63)
                        break
                if phase_control is not None:break
            if phase_control is None:raise ArithmeticError('Actual shared field source word control blind')
            cache[x]=dict(correct_E3_hop_fourth_component=expected,wrong_averaged_Slater_fourth_component=value,
                          wrong_averaged_Slater_defect=defect,shared_field_control=phase_control)
        rows.append(dict(path=row['path'],receiver=x,length=row['length'],**cache[x]))
    captures={**jets.body.CAPTURED,**inputs,oldpath.name:oldraw}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captures.setdefault(Path(file).name,Path(file).read_bytes())
    for name,raw in captures.items():
        if (ROOT/name).read_bytes()!=raw:raise ArithmeticError('Counterfactual input/source changed '+name)
    archive=ROOT/(PREFIX+'_input_bytes_20261010.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captures.items()):z.writestr(name,raw)
    result: dict[str,Any]=dict(status='ACTUAL_RAW4_AND_SHARED_FIELD_WORD_CONTROLS_COMPLETE',arithmetic='conditional numerical controls, not outward kernel proof',
                actual12=rows,raw_mass_exact='1/2',raw_normalization_changed=False,
                wrong_averaged_Slater_mean='Pavg=2 sum actual_weight P_alpha; identical first moment but wrong higher Wick law',
                wrong_independent_field='active and auxiliary hopping words incorrectly given independent Theta; actual shared phase sum precedes character',
                source_sha256=jets.digest(Path(__file__).read_bytes()),boundary_receipt_sha256=jets.digest(oldraw),
                input_archive=dict(path=archive.name,sha256=jets.digest(archive.read_bytes()),files={n:jets.digest(v) for n,v in captures.items()}),
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)
    output=ROOT/(PREFIX+'_receipt_20261010.json');output.write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/(PREFIX+'_natural_exit_capsule_20261010.json')).write_text(json.dumps(dict(numeric_exit=0,natural_exit=True,source_stable=True,
        source_sha256=result['source_sha256'],receipt=dict(path=output.name,sha256=jets.digest(output.read_bytes())),
        input_archive=result['input_archive'],elapsed_seconds=result['elapsed_seconds']),indent=2)+'\n')
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):result=run()
    print(json.dumps(dict(status=result['status'],actual12=len(result['actual12']),elapsed_seconds=result['elapsed_seconds'])),flush=True)
