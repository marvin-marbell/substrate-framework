#!/usr/bin/env python3
"""FIRST/MIXED ORIGINAL Alpha0226 source and finite source-range geometry.

The original meanbody/Gaussian boundary and Gauss masks stay fixed; the full
neutral preparation, primitive path creator, charged SF and prep normalization
all move. This is NOT the external-receiver family holding a postprepared
source fixed, nor a physical logK current, full-range Omega/contact or Compton.
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
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import scipy
import physical_alpha0226_midpoint_operator as core

IMPORT_EPOCH=time.time()
SOURCE_BYTES=Path(__file__).read_bytes()
MODULES=(core,core.field,core.primitive,core.consumer,core.consumer.reference,core.consumer.ALPHA)
IMPORT_BYTES={str(module.__name__):Path(cast(str,module.__file__)).read_bytes() for module in MODULES}
SUFFIXES=('','x','y','xy')
SPLITS={'':(('', ''),),'x':(('x',''),('','x')),'y':(('y',''),('','y')),
        'xy':(('xy',''),('x','y'),('y','x'),('','xy'))}


def gram_jets(jets: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    """Scalar for a neutral vector; coherent fourRef matrix for a packet."""
    def inner(a: np.ndarray,b: np.ndarray) -> np.ndarray:
        if a.ndim==3:return np.asarray(np.vdot(a,b))
        aa=a.reshape(-1,a.shape[-1]);bb=b.reshape(-1,b.shape[-1])
        return aa.conj().T@bb
    return {s:sum((inner(jets[a],jets[b]) for a,b in SPLITS[s]),np.asarray(0j)) for s in SUFFIXES}


def scalar_product(a: dict[str,Any],b: dict[str,Any]) -> dict[str,Any]:
    return {s:sum(a[p]*b[q] for p,q in SPLITS[s]) for s in SUFFIXES}


def inverse_power(a: dict[str,float],power: float) -> dict[str,float]:
    value=a['']
    return {'':value**power,'x':power*value**(power-1)*a['x'],
        'y':power*value**(power-1)*a['y'],
        'xy':power*value**(power-1)*a['xy']+power*(power-1)*value**(power-2)*a['x']*a['y']}


def act(parts: Any,jets: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    return {s:sum((core.field.tensor_action(parts,jets[b],a) for a,b in SPLITS[s]),np.zeros_like(jets[''])) for s in SUFFIXES}


def create(source: dict[str,np.ndarray],jets: dict[str,np.ndarray],shape: tuple[int,...]) -> dict[str,np.ndarray]:
    return {s:sum((np.stack([(op@jets[b].reshape(70,-1)).reshape(shape) for op in source[a]],axis=-1)
        for a,b in SPLITS[s]),np.zeros(shape+(4,),complex)) for s in SUFFIXES}


class Alpha0226SourceGeometry:
    """Packed ORIGINAL source jets; no field-node-by-complete-basis matrices."""
    def __init__(self, midpoint: core.Alpha0226Midpoint | None=None):
        self.midpoint=midpoint if midpoint is not None else core.Alpha0226Midpoint(order=64,jets=True)
        model=self.midpoint
        if not model.jets:raise ValueError('Original source derivatives require actual SF jets')
        self.path_jet_max={s:0. for s in SUFFIXES}
        middle={s:np.zeros_like(model.body) for s in SUFFIXES}
        for parts,w,f in zip(model.neutral,model.weights,model.boundary,strict=True):
            for s in SUFFIXES:middle[s]+=w*f*core.field.tensor_action(parts,model.body,s)
        for value in middle.values():value*=model.pn
        prep={s:0. for s in SUFFIXES}
        raw={s:np.zeros((4,4),complex) for s in SUFFIXES}
        charged={s:np.zeros(model.shape+(4,),complex) for s in SUFFIXES}
        links=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex)
        creator_value_defect=0.
        for coordinate,nparts,cparts,w in zip(model.nodes,model.neutral,model.factors,model.weights,strict=True):
            phases=np.array([[0.,coordinate[1],coordinate[2]],[coordinate[0],coordinate[3],coordinate[4]]])+model.x*model.dx+model.y*model.dy
            packet=core.primitive.source_jet(model.active,3*phases,links,(3*model.dx,3*model.dy))
            source={s:packet['_'+s if s else ''] for s in SUFFIXES}
            for s in SUFFIXES:self.path_jet_max[s]=max(self.path_jet_max[s],float(np.linalg.norm(source[s])))
            creator_value_defect=max(creator_value_defect,float(np.linalg.norm(source['']-np.asarray(model.creators))))
            state=act(nparts,middle)
            for s,g in gram_jets(state).items():prep[s]+=w*float(np.real(g))
            packet_jets=create(source,state,model.shape)
            for s,g in gram_jets(packet_jets).items():raw[s]+=w*g
            half=act(cparts,packet_jets)
            for s in SUFFIXES:charged[s]+=w*half[s]
        self.prep_Gram_jets=prep
        self.raw_unnormalized_Gram_jets=raw
        self.rawCAR_Gram_jets=scalar_product(raw,inverse_power(prep,-1.))
        self.jets=scalar_product({s:model.pack(value*model.pc[...,None]) for s,value in charged.items()},inverse_power(prep,-.5))
        self.m=self.jets[''];self.mx=self.jets['x'];self.my=self.jets['y'];self.mxy=self.jets['xy']
        self.half_Gram_jets=gram_jets(self.jets)
        self.creator_value_defect=creator_value_defect
        self.core_value_relative_defect=float(np.linalg.norm(self.m-model.source_mid)/np.linalg.norm(model.source_mid))
        if self.core_value_relative_defect>2e-11:raise ArithmeticError('Original primitive source and core source disagree')

    def immersion_metrics(self) -> dict[str,Any]:
        """Moving fourRef source-generated range ONLY, not complete field range.

        F=V m, F_j=V_j m+V m_j. Projected normal metric removes projection
        onto span(F), not onto range(V). No omitted complete-range component
        is declared zero, and this finite metric cannot supply full Omega.
        """
        model=self.midpoint;packed={s:model.embed(v) for s,v in self.jets.items()}
        all_cross={a+b:np.zeros((4,4),complex) for a in ('0','x','y','xy') for b in ('0','x','y','xy')}
        source_cross={key:np.zeros((4,4),complex) for key in all_cross}
        labels={'':'0','x':'x','y':'y','xy':'xy'}
        for parts,w in zip(model.factors,model.weights,strict=True):
            pure={s:core.field.tensor_action(parts,packed[''],s).reshape(-1,4) for s in SUFFIXES}
            moving={s:value.reshape(-1,4) for s,value in act(parts,packed).items()}
            for a in SUFFIXES:
                for b in SUFFIXES:
                    key=labels[a]+labels[b]
                    all_cross[key]+=w*(pure[a].conj().T@pure[b])
                    source_cross[key]+=w*(moving[a].conj().T@moving[b])
        g=source_cross['00'];eigenvalues=np.linalg.eigvalsh(g)
        if eigenvalues[0]<=0:raise ArithmeticError('FourRef projected immersion lost rank')
        normal={a+b:source_cross[a+b]-source_cross['0'+a].conj().T@np.linalg.solve(g,source_cross['0'+b]) for a in ('x','y') for b in ('x','y')}
        core_g=self.m.conj().T@model.cross_gram(self.m)
        return dict(pure_V_source_cross_Grams=all_cross,moving_Vm_source_cross_Grams=source_cross,
            projected_fourRef_normal_tangent_Grams=normal,projected_fourRef_field_Gram_eigenvalues=eigenvalues,
            core_cross_gram_contraction_relative_defect=float(np.linalg.norm(g-core_g)/np.linalg.norm(g)),
            scope='F(s)=V_C(s)m_C(s). Finite fourRef source-generated projected field range; NOT complete positive-range Omega/contact. The complete field-range connection is still owed separately.')


def nonlinear_source(base: core.Alpha0226Midpoint,x: float,y: float) -> dict[str,Any]:
    """Independent nonlinear reconstruction: eigh SF and actual path exp."""
    n=core.field.make_value_factors(base.active,base.auxiliary,base.nodes,4,base.la,base.lu,x,y,(base.dx,base.dy))
    c=core.field.make_value_factors(base.active,base.auxiliary,base.nodes,5,base.la,base.lu,x,y,(base.dx,base.dy))
    middle=np.zeros_like(base.body)
    for parts,w,f in zip(n,base.weights,base.boundary,strict=True):middle+=w*f*core.field.tensor_action(parts,base.body)
    middle*=base.pn;prep=0.;raw=np.zeros((4,4),complex);charged=np.zeros(base.shape+(4,),complex)
    links=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex)
    for coordinate,np_,cp,w in zip(base.nodes,n,c,base.weights,strict=True):
        state=core.field.tensor_action(np_,middle);prep+=w*float(np.vdot(state,state).real)
        phases=np.array([[0.,coordinate[1],coordinate[2]],[coordinate[0],coordinate[3],coordinate[4]]])+x*base.dx+y*base.dy
        # Actual nonlinear path, independent of derivative contractions.
        path=np.vstack((np.exp(3j*phases[0,0])*links[0,0],np.eye(4)))/2
        source=core.primitive.packet_creation(path)
        packet=np.stack([(op@state.reshape(70,-1)).reshape(base.shape) for op in source],axis=-1)
        flat=packet.reshape(-1,4);raw+=w*(flat.conj().T@flat)
        charged+=w*core.field.tensor_action(cp,packet)/base.normalizer_scale
    m=base.pack(charged*base.pc[...,None])/np.sqrt(prep)
    return dict(m=m,prep=prep,rawCAR=float(np.trace(raw).real/prep),half=float(np.vdot(m,m).real),
        rawCAR_matrix=raw/prep,half_matrix=m.conj().T@m)


def finite_differences(geometry: Alpha0226SourceGeometry) -> list[dict[str,Any]]:
    rows=[];model=geometry.midpoint
    for h in (.004,.002):
        values={(x,y):nonlinear_source(model,x*h,y*h) for x,y in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))}
        row: dict[str,Any]=dict(step=h,rawCAR_range=[min(v['rawCAR'] for v in values.values()),max(v['rawCAR'] for v in values.values())],jets={})
        for s in ('x','y','xy'):
            def diff(key: str):
                if s=='x':return (values[1,0][key]-values[-1,0][key])/(2*h)
                if s=='y':return (values[0,1][key]-values[0,-1][key])/(2*h)
                return (values[1,1][key]-values[1,-1][key]-values[-1,1][key]+values[-1,-1][key])/(4*h*h)
            delta=diff('m')-geometry.jets[s];norm=float(np.linalg.norm(geometry.jets[s]))
            row['jets'][s]=dict(vector_absolute_residual=float(np.linalg.norm(delta)),vector_relative_residual=float(np.linalg.norm(delta)/norm) if norm else None,
                prep_Gram_absolute_residual=float(abs(diff('prep')-geometry.prep_Gram_jets[s])),
                rawCAR_derivative_absolute_residual=float(abs(diff('rawCAR')-np.trace(geometry.rawCAR_Gram_jets[s]).real)),
                half_Gram_absolute_residual=float(abs(diff('half')-np.trace(geometry.half_Gram_jets[s]).real)),
                rawCAR_fourRef_matrix_absolute_residual=float(np.linalg.norm(diff('rawCAR_matrix')-geometry.rawCAR_Gram_jets[s])),
                half_fourRef_matrix_absolute_residual=float(np.linalg.norm(diff('half_matrix')-geometry.half_Gram_jets[s])))
        rows.append(row)
    if max(v['vector_relative_residual'] or 0 for v in rows[-1]['jets'].values())>2e-4:
        raise ArithmeticError('Original source jets fail independent nonlinear reconstruction')
    return rows


def serial(value: Any) -> Any:
    if isinstance(value,np.ndarray):
        return {'real':value.real.tolist(),'imag':value.imag.tolist()} if np.iscomplexobj(value) else value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--vectors',required=True)
    args=parser.parse_args();started=time.time();geometry=Alpha0226SourceGeometry();model=geometry.midpoint
    fd=finite_differences(geometry);metrics=geometry.immersion_metrics()
    np.savez(args.vectors,**{('m'+s):v for s,v in geometry.jets.items()})
    input_json=dict(descriptor=model.descriptor(),boundary_family='ORIGINAL boundary fixed; FULLneutralK, path source, charged SF and prep norm differentiated. Postprepared source NOT held fixed.',
        steps=[.004,.002],normalizer='Frozen common GLOBAL lambdaN; no lambda derivatives',positive_range='Gauss density mask fixed; original field immersion moves; no kernel log extension or Omega/contact convention supplied')
    report=dict(input=input_json,prep_Gram_jets=geometry.prep_Gram_jets,rawCAR_fourRef_Gram_jets=geometry.rawCAR_Gram_jets,
        rawCAR_total_jets={s:float(np.trace(v).real) for s,v in geometry.rawCAR_Gram_jets.items()},
        half_fourRef_Gram_jets=geometry.half_Gram_jets,actual_packed_jet_norms={s:float(np.linalg.norm(v)) for s,v in geometry.jets.items()},
        actual_path_source_jet_maxnorms=geometry.path_jet_max,creator_value_defect=geometry.creator_value_defect,
        core_value_relative_defect=geometry.core_value_relative_defect,independent_nonlinear_finite_differences=fd,immersion_metrics=metrics,
        vectors=dict(path=args.vectors,sha256=hashlib.sha256(Path(args.vectors).read_bytes()).hexdigest(),shape=list(geometry.m.shape)),
        execution=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),import_epoch=IMPORT_EPOCH,
            module_sha256_at_IMPORT={name:hashlib.sha256(data).hexdigest() for name,data in IMPORT_BYTES.items()},module_bytes_at_IMPORT={name:data.decode() for name,data in IMPORT_BYTES.items()},
            input_sha256=hashlib.sha256(json.dumps(input_json,sort_keys=True).encode()).hexdigest(),input_json=input_json,
            python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,runtime=platform.platform(),
            startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and all(Path(cast(str,module.__file__)).read_bytes()==IMPORT_BYTES[str(module.__name__)] for module in MODULES),elapsed_seconds=time.time()-started),
        interpretation='Exact M64 finite frozen state numerical source derivatives and projected fourRef immersion only. No Hcurrent/Compton/continuousHaar/fullpositive-range contact promotion.')
    Path(args.output).write_text(json.dumps(serial(report),indent=2)+'\n')
    print(json.dumps({key:serial(report[key]) for key in ('actual_packed_jet_norms','actual_path_source_jet_maxnorms','rawCAR_total_jets','independent_nonlinear_finite_differences')}))


if __name__=='__main__':main()
