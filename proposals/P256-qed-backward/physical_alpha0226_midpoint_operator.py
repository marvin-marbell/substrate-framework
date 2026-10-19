#!/usr/bin/env python3
"""Exact two-site Alpha0226 midpoint/operator jets and original source.

This is a matrix-free representation of the COMPLETE Gauss midpoint, not an
occupied-sea or charged-band compression. SF jets carry both scalar and
noncommuting exponential contacts. Abar jets alone are NOT physical logK jets:
the ORIGINAL field-space immersion and its moving positive range are exposed
through cross_gram(left,right). The frozen GLOBAL neutral norm is constant.
No kernel-energy/log extension or positive-range contact convention is chosen
by this module. Its callers must declare those and price source complements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, cast

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import scipy
import physical_c_compact_haar_field_current as field

SOURCE_BYTES=Path(__file__).read_bytes()
FIELD_BYTES=Path(cast(str,field.__file__)).read_bytes()
if hashlib.sha256(FIELD_BYTES).hexdigest()!='8dda133b022a54c8f8b716d3b1a481ff56a7ed638887f2397fd47651fc70aab4':
    raise ArithmeticError('Exact midpoint bridge requires frozen8dda field jets')
primitive=field.primitive
consumer=field.consumer


class Alpha0226Midpoint:
    """Actual fullcanonical K=VVdag, Abar=VdagV with ALLrho0 states."""
    def __init__(self, *, order: int=64, jets: bool=True, x: float=0., y: float=0.):
        self.order=order;self.jets=jets;self.x=x;self.y=y
        self.parameters=consumer.reference.NativeParameters()
        self.active,_,_=consumer.reference.native_law(self.parameters)
        self.auxiliary,_,_=consumer.reference.native_law(self.parameters)
        self.auxiliary.kappa=.08
        self.va,self.la=consumer.meanbody(self.active)
        self.vu,self.lu=consumer.meanbody(self.auxiliary)
        self.body=np.einsum('a,b,c->abc',self.va,self.vu,self.vu)
        self.nodes,self.weights,self.boundary=consumer.compact_nodes(order,20261009,3.5,100.)
        self.boundary/=np.sqrt(self.weights@self.boundary**2)
        self.pn=consumer.physical_density_projector(4,3)
        self.pc=consumer.physical_density_projector(5,3)
        self.dimension=int(self.pc.sum());self.shape=self.pc.shape
        self.creators=consumer.source_creators()
        self.dx=np.zeros((2,3));self.dy=np.zeros((2,3))
        self.dx[:,1]=[1.,-1.];self.dy[:,2]=[1.,-1.]
        self.dx/=np.sqrt(200.);self.dy/=np.sqrt(200.)
        self.pin_path=Path(__file__).parent/'physical_c_compact_haar_charged_source_final_capsule.json'
        self.pin_bytes=self.pin_path.read_bytes()
        if hashlib.sha256(self.pin_bytes).hexdigest()!='70b03620f5e0eadcb13e937e2aea802fdc2ca2448e74796c9d6071b6611d9e45':
            raise ArithmeticError('Frozen Alpha0226 normalizer/source capsule changed')
        self.pin=json.loads(self.pin_bytes)
        self.lambdaN=float(self.pin['cases'][0]['neutral_trial_Rayleigh'])
        self.normalizer_scale=np.sqrt(self.lambdaN)
        self.factors=[];self.neutral=[]
        remainder=np.zeros((2,4,4,4),complex)
        colour=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex)
        for coordinate in self.nodes:
            phases=np.array([[0.,coordinate[1],coordinate[2]],[coordinate[0],coordinate[3],coordinate[4]]])+x*self.dx+y*self.dy
            if jets:
                a=primitive.canonical_jet(self.active,3*phases,remainder,colour,5,self.la,(3*self.dx,3*self.dy))
                n=primitive.canonical_jet(self.active,3*phases,remainder,colour,4,self.la,(3*self.dx,3*self.dy))
                u=primitive.canonical_jet(self.auxiliary,phases,remainder,colour,4,self.lu,(self.dx,self.dy))
            else:
                a=dict(SF=consumer.canonical_SF(self.active.potential(3*phases,remainder,colour),5,self.la))
                n=dict(SF=consumer.canonical_SF(self.active.potential(3*phases,remainder,colour),4,self.la))
                u=dict(SF=consumer.canonical_SF(self.auxiliary.potential(phases,remainder,colour),4,self.lu))
            self.neutral.append((n,u,u))
            # Only the active factor carries the ONE common scalar norm.
            scaled={key:(value/self.normalizer_scale if key.startswith('SF') else value) for key,value in a.items()}
            self.factors.append((scaled,u,u))
        self.source_mid,self.prep_Gram,self.raw_Gram=self.original_source()
        self.half_Gram=float(np.vdot(self.source_mid,self.source_mid).real)

    def embed(self, v: np.ndarray) -> np.ndarray:
        if v.shape[0]!=self.dimension:raise ValueError('Complete midpoint dimension mismatch')
        full=np.zeros(self.shape+v.shape[1:],complex);full[self.pc]=v
        return full

    def pack(self, v: np.ndarray) -> np.ndarray:
        return v[self.pc]

    def cross_gram(self, v: np.ndarray, left: str='', right: str='') -> np.ndarray:
        """V_left^dag V_right v in exact COMPLETE original midpoint coordinates."""
        if (left or right) and not self.jets:raise ValueError('Actual SF jets were not constructed')
        full=self.embed(v);total=np.zeros_like(full)
        for parts,w in zip(self.factors,self.weights,strict=True):
            image=field.tensor_action(parts,full,right)
            total+=w*field.tensor_action(parts,image,left)
        return self.pack(total)

    def apply(self, v: np.ndarray, suffix: str='') -> np.ndarray:
        """Complete Abar derivative; NOT physical -logK derivative."""
        if not suffix:return self.cross_gram(v)
        if suffix in ('x','y'):return self.cross_gram(v,suffix,'')+self.cross_gram(v,'',suffix)
        if suffix=='xy':
            return self.cross_gram(v,'xy','')+self.cross_gram(v,'x','y')+self.cross_gram(v,'y','x')+self.cross_gram(v,'','xy')
        raise ValueError('Unknown actual midpoint derivative')

    def original_source(self) -> tuple[np.ndarray,float,float]:
        """Original unchanged boundary -> fullK first -> raw4Ref -> V_Cdag."""
        middle=np.zeros_like(self.body)
        for parts,w,f in zip(self.neutral,self.weights,self.boundary,strict=True):
            middle+=w*f*field.tensor_action(parts,self.body)
        middle*=self.pn;prep=0.;raw_norm=0.
        charged=np.zeros(self.shape+(4,),complex)
        for nparts,cparts,w in zip(self.neutral,self.factors,self.weights,strict=True):
            state=field.tensor_action(nparts,middle);prep+=w*float(np.vdot(state,state).real)
            raw=np.stack([(c@state.reshape(70,-1)).reshape(self.shape) for c in self.creators],axis=-1)
            raw_norm+=w*float(np.vdot(raw,raw).real)
            charged+=w*field.tensor_action(cparts,raw)
        charged*=self.pc[...,None]
        if abs(raw_norm/prep-1)>3e-10:raise ArithmeticError('Original fullK raw4Ref CAR changed')
        return self.pack(charged)/np.sqrt(prep),float(prep),float(raw_norm/prep)

    def descriptor(self) -> dict[str,Any]:
        return dict(evaluated_kernel_source=hashlib.sha256(primitive.CONSUMER_BYTES).hexdigest(),
            frozen_preparation_capsule_sha256=hashlib.sha256(self.pin_bytes).hexdigest(),
            order=self.order,seed=20261009,weight_sum=float(sum(self.weights)),
            coordinates=self.nodes.tolist(),weights=self.weights.tolist(),boundary=self.boundary.tolist(),
            geometry=[2,1,1],Q=3,Naux=2,kaux=.08,Phi='0',colour='I',statistical_sector=[0,0,0],
            actual_COMPLETEcharged_midpoint_dimension=self.dimension,active_fullcanonical=56,aux_each_fullcanonical=70,
            Refprimitive_charge=0,external_compensator=[1,-3],source='ORIGINALfixedmeanbody andperiodicGaussian3.5; actualFULLK first, raw4Ref then COMPLETEpositivehalf. No adoptedsea/sourcegate.',
            positive_source_mid_shape=list(self.source_mid.shape),original_preparation_Gram=self.prep_Gram,
            raw_fourRef_CAR_Gram=self.raw_Gram,wholehalf_source_Gram=self.half_Gram,
            common_GLOBALneutral_lambdaN=self.lambdaN,lambdaN_parameter_family='FROZENconstant commonGLOBALneutral norm. No lambdaNj/lambdaNij terms.',
            units='Abar=VdagV is K/lambdaN nonzero spectrum; H=-log(Abar)/a, a=.01. Abar derivative alone omits ORIGINALfield polarconnection.',
            spacing=self.parameters.spacing,photon_coordinate_calibration_beta=100.,background_parameters=[self.x,self.y],
            transverse_directions=[self.dx.tolist(),self.dy.tolist()],
            range='Originalfield immersion V_i=SF_i P/sqrt(lambdaN). cross_gram gives all V_left†V_right, including movingrange. No kernel extension/transport convention chosen.',
            finite_scope='Finite positive complete5cycleHaar quadrature, no continuousHaar/wholePhi/SU4/statistical/macrolaw certificate. Order64 uses EXACTfrozen source/weights; anyotherorder is separatelylabelled quadrature.',
            interpretation='Callback is completeGauss Abar, NOT compressedoccupied/band/logK operator. NumericalKrylov consumers must price residual/complement and not retag lowrankprojected log as fullphysicallogK.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--order',type=int,default=64)
    args=parser.parse_args();started=time.time();model=Alpha0226Midpoint(order=args.order,jets=False)
    m=model.source_mid;am=model.apply(m)
    report=dict(descriptor=model.descriptor(),actual_midpoint_source_K_moment1=float(np.vdot(m,am).real/np.vdot(m,m).real),
        actual_Hermitian_midpoint_source_defect=float(abs(np.vdot(m,am).imag)),
        execution=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
            field_at_IMPORT=hashlib.sha256(FIELD_BYTES).hexdigest(),consumer_at_IMPORT=hashlib.sha256(primitive.CONSUMER_BYTES).hexdigest(),
            input_json=model.descriptor(),input_sha256=hashlib.sha256(json.dumps(model.descriptor(),sort_keys=True).encode()).hexdigest(),
            python=sys.version,numpy=np.__version__,scipy=scipy.__version__,runtime=platform.platform(),
            startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(cast(str,field.__file__)).read_bytes()==FIELD_BYTES and model.pin_path.read_bytes()==model.pin_bytes,
            elapsed_seconds=time.time()-started))
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('descriptor','execution')}))


if __name__=='__main__':main()
