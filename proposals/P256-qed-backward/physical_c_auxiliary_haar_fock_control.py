#!/usr/bin/env python3
"""Positive full-Fock charge-Haar projector and original RAW source control.

All70/4900 reachable occupation amplitudes; literal shifted integer projector,
not Coulomb exp(-.02rhoLrho).  Fourier cancellation is reported rather than
clipped, and positive charge-sector evaluation resolves the actual operator.
Two-site native algebra control, not global Root931 or macrovolume admission.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import physical_c_auxiliary_coulomb_source_fock_control as fock
source=fock.source
SOURCE_BYTES=Path(__file__).read_bytes();FOCK_BYTES=fock.SOURCE_BYTES


def positive_case(context: dict,direct: dict,Q: int) -> dict:
    nf=context['NF'];SF=direct['SF'];prime=direct['SFprime'];rho=direct['rho']+np.array([Q,-Q])
    projector=np.all(rho==0,axis=1);first=fock.tensor_apply(SF,direct['body'],nf)
    prepared=fock.tensor_apply(SF,projector*first,nf)
    multiplication=math.sqrt(context['width']/2)*direct['q']*prepared
    derivative=-fock.tensor_prime(SF,prime,projector*first,nf)/math.sqrt(2*context['width'])
    raw=multiplication+derivative
    final=fock.tensor_apply(SF,raw,nf);mfinal=fock.tensor_apply(SF,multiplication,nf);dfinal=fock.tensor_apply(SF,derivative,nf)
    full=float(np.vdot(final,projector*final).real)
    scalar=float(np.vdot(mfinal,projector*mfinal).real);gradient=float(np.vdot(dfinal,projector*dfinal).real)
    interference=float(2*np.vdot(mfinal,projector*dfinal).real)
    return dict(projector_rank=int(np.sum(projector)),
        preparation_Gram=float(np.vdot(prepared,prepared).real),raw_creator_Gram=float(np.vdot(raw,raw).real),wholehalf_creator_Gram=full,
        multiplication_Gram=scalar,derivative_Gram=gradient,multiplication_derivative_interference=interference,
        exact_zero_by_charge_sector=bool(not np.any(projector)),fullFock_dimension=len(prepared),
        bareGamma_global_Haar_falsifier='Without occupied2/site centering, total number4NF is nonzero and the GLOBAL U1 Haar average kills the entire reachable sector. Anchoring eta0 would then be invalid.')


def Fourier_case(context: dict,Q: int,order: int) -> dict:
    angles=2*math.pi*np.arange(order)/order;etas=[np.array([0.,x]) for x in angles];zero=np.zeros(2);background=np.array([Q,-Q])
    histories=[source.histories(context,eta) for eta in etas];prep=0j;raw=0j;final=np.zeros(3,np.complex128)
    for ia,iz in itertools.product(range(order),repeat=2):
        eta=etas[ia];zeta=etas[iz];left,leftsf,cl,csl=histories[ia];right,rightsf,cr,csr=histories[iz]
        coefficient=np.exp(1j*((zeta-eta)@background))/(order**2)
        part=source.quadratic(context,left,right,cl,cr,zero,False)
        prep+=coefficient*part[0];raw+=coefficient*np.sum(part[1:])
        for xi in etas:
            part=source.quadratic(context,leftsf,rightsf,csl,csr,xi,True)
            final+=coefficient*np.exp(1j*(xi@background))*part[1:]/order
    return dict(order=order,preparation_Gram=float(prep.real),raw_creator_Gram=float(raw.real),wholehalf_creator_Gram=float(np.sum(final).real),
        scalar_Gram=float(final[0].real),horizontal_tangent_Gram=float(final[1].real),scalar_tangent_interference=float(final[2].real),
        wholehalf_imaginary=float(np.sum(final).imag),contract='M>2(2NF+Q) integrates the exact finite charge polynomial without aliasing; float cancellation errors are measured, never floored or retagged as physical zeros.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);args=parser.parse_args();start=time.time();rows=[]
    for nf in (1,2):
        context,direct=fock.build(nf);context['heat']=np.ones((1,1));context['UA']=math.inf
        for Q in (0,1,3):
            exact=positive_case(context,direct,Q);approx=Fourier_case(context,Q,17)
            absolute={key:abs(exact[key]-approx[key]) for key in ('preparation_Gram','raw_creator_Gram','wholehalf_creator_Gram')}
            relative={key:abs(exact[key]-approx[key])/exact[key] if exact[key] else None for key in absolute}
            rows.append(dict(NF=nf,Q=Q,positive_fullFock_projector_source=exact,centred_threeHaar_Fourier_control=approx,
                absolute_errors=absolute,relative_errors_when_nonzero=relative,
                resolved_primitive='Positive exact charge-sector projector is used for actual rare/zero sector evaluation; the Fourier residual is disclosed, not promoted.'))
    execution=dict(source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),Fock_source_sha256=hashlib.sha256(FOCK_BYTES).hexdigest(),Fock_source_bytes_at_IMPORT=FOCK_BYTES.decode(),
        coupled_source_bytes_at_IMPORT=source.SOURCE_BYTES.decode(),prepared_source_bytes_at_IMPORT=source.PREPARED_BYTES.decode(),SF_source_bytes_at_IMPORT=source.SF_BYTES.decode(),core_source_bytes_at_IMPORT=source.coul.loops.CORE_BYTES.decode(),frozen_dependencies=source.coul.loops.IMPORT_EPOCH,
        argv=sys.argv,input_parameters=dict(shape=[2,1,1],NF=[1,2],Q=[0,1,3],Fourier_order=17,Gaussian_width=3.5,k=.01,q=.3,field_middle='bounded single-node rank-one chart',gauge_middle='normalized centered charge Haar projector'),
        python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,elapsed_seconds=time.time()-start,startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES)
    report=dict(rows=rows,scope='Native pureAbelian positive charge-Haar/SF/originalRAWB algebra. FiniteNF1 Q3 is genuinely impossible by local Pauli charge capacity, not a tinycharge fit or source gate. NF2 Q3 is a rare but allowed sector. Does NOT establish macrocharged range/Gram, allcompactcoexact+toron Haar, Root931/Ref4 source equivalence, or physicalpole.')
    Path(args.capsule).write_text(json.dumps(dict(report=report,execution=execution),indent=2)+'\n');print(json.dumps(dict(capsule=args.capsule,report=report,elapsed_seconds=execution['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
