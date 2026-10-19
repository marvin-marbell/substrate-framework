"""Independent finite-Fock acceptance of the standalone typed Ball evaluator.

The already-completed full-source n4 guard remains immutable; this exercises
its typed precision separation without repeating that full-source producer.
"""
from __future__ import annotations
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
import joined_phi_u1_initial_density_higher_jets_interval as interval

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_initial_density_higher_jets_interval_control'


def run()->dict[str,Any]:
    start=time.monotonic();ar=interval.ar;ar.iv.dps=75
    p=np.block([[.5*np.eye(4),.5*np.eye(4)],[.5*np.eye(4),.5*np.eye(4)]]).astype(complex)
    block=np.diag([.125,.25,.375,.5]).astype(complex)
    onsite=np.diag([.03,-.02,.04,-.01]).astype(complex)
    geo=dict(k=np.array([[1],[-1]]),x=np.array([0,1]),y=np.array([1,0]))
    species=[(ar.Ball(np.asarray([block,block])),ar.Ball(onsite),ar.Ball(p)),
             (ar.Ball(np.zeros((2,4,4))),ar.Ball(np.zeros((4,4))),ar.Ball(p))]
    engine=interval.OutwardEngine(dict(native=dict(D=np.array([[1],[-1]]))),geo,
        dict(denominator=4,L_integer=np.array([[1,-1],[-1,1]])),species)
    number=jets.wick.finite_car(np.diag([1.]*4+[0.]*4))
    difference=jets.wick.finite_car(np.diag([1.]*4+[-1.]*4))
    matter=jets.wick.finite_car(np.block([[onsite,block],[block,onsite]]))
    w=np.vstack((np.eye(4),np.eye(4)))/math.sqrt(2);q=(np.eye(8)-p)[:,:4]
    def slater(occupied):
        vector=np.zeros(256,complex)
        for sites in jets.itertools.combinations(range(8),occupied.shape[1]):vector[sum(1<<i for i in sites)]=np.linalg.det(occupied[list(sites)])
        return vector
    bg=slater(w);source=[slater(np.column_stack((w,q[:,a]/np.linalg.norm(q[:,a])))) for a in range(4)]
    hbg=16*difference@difference+matter;shifted=difference-np.eye(256);hraw=16*shifted@shifted+matter
    cbg=number.copy();craw=number.copy();symbols=engine.initial(0);rows=[]
    for order in range(1,5):
        cbg=hbg@cbg-cbg@hbg;craw=hraw@craw-craw@hraw
        if order>1:symbols=engine.derivative(symbols,order)
        if order%2:continue
        direct=(1j**order)*(-.5*np.vdot(bg,cbg@bg)+sum(.125*np.vdot(v,craw@v) for v in source))
        computed=engine.evaluate(symbols,order,copies=0);bounds=ar.interval(computed.real_interval());point=F.from_float(float(direct.real))
        if not F(bounds['lower'])<=point<=F(bounds['upper']):raise ArithmeticError('Standalone Ball fails independent finite-Fock inclusion')
        if abs(direct.imag)>1e-7:raise ArithmeticError('Finite-Fock Hermitian control has excessive imaginary defect')
        rows.append(dict(order=order,direct=float(direct.real),Ball_enclosure=bounds,Ball_centre=float(np.real(computed.v)),Ball_radius=computed.r,
                         literal_finite_Fock_inclusion=True))
    recurrence=[F(1),F(-1,4)]
    for n in range(2,7):recurrence.append(F(-1,4)*recurrence[-1]+F(n-1,32)*recurrence[-2])
    gaussian=jets.wick.gaussian_linear_moment([F(-1,4)]*6,[[F(1,32)]*6 for _ in range(6)])
    if gaussian!=recurrence[6]:raise ArithmeticError('Standalone initial Gaussian sixth differs from Hermite recurrence')
    captures={'joined_phi_u1_initial_density_higher_jets_interval.py':(ROOT/'joined_phi_u1_initial_density_higher_jets_interval.py').read_bytes(),
              Path(__file__).name:Path(__file__).read_bytes(),
              'joined_phi_u1_initial_density_higher_jets.py':(ROOT/'joined_phi_u1_initial_density_higher_jets.py').read_bytes()}
    for name,raw in captures.items():
        if (ROOT/name).read_bytes()!=raw:raise ArithmeticError('Typed Ball control source changed '+name)
    archive=ROOT/(PREFIX+'_source_bytes_20261010.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in captures.items():z.writestr(name,raw)
    result: dict[str,Any]=dict(status='STANDALONE_BALL_FINITE_CAR_AND_GAUSSIAN_CONTROL_COMPLETE',rows=rows,Gaussian_sixth_exact=str(gaussian),
          scope='Independent two-site/four-colour finite-Fock HC64/onsite and initial Gaussian control; NOT a replacement for all129/all63 actual source guard',
          source_sha256=jets.digest(Path(__file__).read_bytes()),typed_interval_source_sha256=jets.digest((ROOT/'joined_phi_u1_initial_density_higher_jets_interval.py').read_bytes()),
          source_archive=dict(path=archive.name,sha256=jets.digest(archive.read_bytes())),runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)
    output=ROOT/(PREFIX+'_receipt_20261010.json');output.write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/(PREFIX+'_natural_exit_capsule_20261010.json')).write_text(json.dumps(dict(numeric_exit=0,natural_exit=True,source_stable=True,
         source_sha256=result['source_sha256'],receipt=dict(path=output.name,sha256=jets.digest(output.read_bytes())),source_archive=result['source_archive'],elapsed_seconds=result['elapsed_seconds']),indent=2)+'\n')
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):result=run()
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
