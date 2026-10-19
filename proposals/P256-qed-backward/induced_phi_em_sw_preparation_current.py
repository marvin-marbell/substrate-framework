"""Actual full-Href causal current on the charged-alignment momentum source."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from rotation_band import CHARGE_CONJUGATION
from induced_phi_em_sw_preparation import SOURCE_BYTES as PREP_BYTES, build_preparation
from induced_phi_em_sw_current import SOURCE_BYTES as CURRENT_BYTES, measure, operators, serialize
from induced_phi_em_sw_momentum import section
from induced_phi_em_gaussian_variational import build_native

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**PREP_BYTES,**CURRENT_BYTES,Path(__file__).name:Path(__file__).read_bytes()}


def run(mean_path: Path,source_path: Path,samples: int) -> dict[str,Any]:
    started=time.monotonic();meanraw=mean_path.read_bytes();raw=source_path.read_bytes();source=json.loads(raw)
    state=source['selected'];eta=float(state['eta']);variance=float(state['variance']);kappa=float(state['kappa'])
    if source['input_descriptor']['N']!=2 or eta!=source['input_descriptor']['preparation_eta']:
        raise ValueError('Actual source construction/size descriptor is inconsistent')
    native=build_native(json.loads(meanraw),2);native['variance']=variance;model=build_preparation(native,eta)
    direction=np.zeros(3*native['V']);direction[1::3]=np.cos(np.pi*native['sites'][:,0]);direction/=np.linalg.norm(direction)
    ops=operators(native,direction);rng=np.random.default_rng(2221022);points=rng.normal(size=(samples,native['M']))/np.sqrt(4*variance)
    rows=[]
    for z in points:
        plus=measure(model,ops,z,kappa,variance);minus=measure(model,ops,-z,kappa,variance)
        rows.append({key:(plus[key]+minus[key])/2 for key in plus})
    means={key:sum(row[key] for row in rows)/samples for key in rows[0]}
    errors={key:float(np.std([row[key].real for row in rows],ddof=1)/np.sqrt(samples)) for key in rows[0]}
    a=measure(model,ops,points[0],0.,variance);b=measure(model,ops,-points[0],0.,variance);seed={key:(a[key]+b[key])/2 for key in a}
    C=np.kron(np.eye(native['V']),CHARGE_CONJUGATION);cp=[]
    for profile in model['profiles']:
        left=section(profile,points[0],kappa);right=section(profile,-points[0],kappa);P=left@left.conj().T
        cp.append(float(np.linalg.norm(right@right.conj().T-(np.eye(len(P))-C@P.conj()@C.conj().T))))
    cap=ops['remainder']['retarded_kernel_second_derivative_upper'];t=.02*abs(means['retarded_initial_tangent'].real)/cap
    descriptor=dict(mean_sha256=hashlib.sha256(meanraw).hexdigest(),source_sha256=hashlib.sha256(raw).hexdigest(),N=2,V=8,M=17,
        a=native['a'],UE=native['UE'],charged_counts=[1,63],eta_preparation_ONLY=eta,variance=variance,kappa=kappa,
        independent_antithetic_pairs=samples,probe='LiteralcoexactJy withq=(pi,0,0), physicalHref NOTpreparationh',
        plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        actual_prepared=means,standard_errors=errors,same_eta_Gaussian_seed=seed,
        momentum_dressing_tangent_shift=means['retarded_initial_tangent']-seed['retarded_initial_tangent'],
        CP_density_intertwining_errors=cp,priced_short_time=dict(t=t,retarded_linear_value=t*means['retarded_initial_tangent'],
            absolute_remainder_upper=t*t*cap/2,purpose='Formula-baseddiagnosticwindow, notclock/speedfit orobservedfiniteTpropagator'),
        remainder=ops['remainder'],
        exact_functional='chi_R(t)=-i theta(t) Tr(rho_eta,kappa [J_f(t),J_f(0)]), physicalJ_f andHref fixed. mu=<[[Href,J_f],J_f]>. All64CAR, primitivePi andfullrealGaussCoulomb computedin actualjointmomentumsource withcofactors, neverbare/ThetaSlater.',
        limitations='NativeN2midpointMC/SE, notintervalproof. No stationaryphotonpole/residue, finiteclock4096 evolution, macroground/commoncone/Compton/QEDpromotion. Eta modifiesONLYthepreparedchargedstate. Genericwordbackend independentlycontrolledagainstactual4096CAR/216wedge, distinctspeciesnotdropped.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','source','output'):parser.add_argument('--'+role,type=Path,required=True)
    parser.add_argument('--samples',type=int,required=True);args=parser.parse_args()
    if args.samples<2:raise ValueError('ActualSEM requires two independent antithetic pairs')
    report=run(args.mean,args.source,args.samples);args.output.write_text(json.dumps(report,indent=2,default=serialize)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','actual_prepared','standard_errors','same_eta_Gaussian_seed','momentum_dressing_tangent_shift','CP_density_intertwining_errors','priced_short_time','elapsed_seconds')},indent=2,default=serialize))
