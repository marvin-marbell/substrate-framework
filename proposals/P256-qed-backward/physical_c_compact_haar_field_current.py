#!/usr/bin/env python3
"""Full field/full4Dirac Fock literal-Haar K current and mixed contact.

K=V Vdag is differentiated in the ORIGINAL compact field representation,
not by differentiating Abar in a moving polar/occupied basis.  All canonical
active/auxiliary states, exact density0 midpoint and coherent four Ref holes
are retained.  Primitive scalar and nonlinear contact come from frozen7aa8.
This is a finite NFaux2, eight-node, Phi0/colourI/stat000 native construction.
It supplies K/Kx/Ky/Kxy, NOT the physical -log(K/lambdaN)/a current, a full
Cartesian/SU4/generalNF factory, mass1, or a measured Compton probability.
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
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import scipy
import physical_c_compact_haar_current_contact as primitive

SOURCE_BYTES=Path(__file__).read_bytes();PRIMITIVE_BYTES=Path(cast(str,primitive.__file__)).read_bytes()
if hashlib.sha256(PRIMITIVE_BYTES).hexdigest()!='7aa8891a7683e8a9526734c4b0ac0c116fe44852208f321aa38b64f47944fa76':raise ArithmeticError('Field current requires frozen7aa8 canonical jets')
consumer=primitive.consumer


def product(a: np.ndarray,b: np.ndarray,c: np.ndarray,v: np.ndarray) -> np.ndarray:
    """Whole canonical tensor; optional last axis carries coherent Ref holes."""
    if v.ndim==3:return consumer.product_apply(a,b,c,v)
    return np.stack([consumer.product_apply(a,b,c,v[...,r]) for r in range(v.shape[-1])],axis=-1)


def tensor_action(parts: tuple[dict[str,Any],dict[str,Any],dict[str,Any]],
                  v: np.ndarray,suffix: str='') -> np.ndarray:
    if suffix=='':return product(parts[0]['SF'],parts[1]['SF'],parts[2]['SF'],v)
    result=np.zeros_like(v)
    for i in range(3):
        matrices=[part['SF'] for part in parts];matrices[i]=parts[i]['SF'+suffix]
        result+=product(matrices[0],matrices[1],matrices[2],v)
    if suffix=='xy':
        for i in range(3):
            for j in range(3):
                if i==j:continue
                matrices=[part['SF'] for part in parts];matrices[i]=parts[i]['SFx'];matrices[j]=parts[j]['SFy']
                result+=product(matrices[0],matrices[1],matrices[2],v)
    return result


def mask_for(project: np.ndarray,node: np.ndarray) -> np.ndarray:
    return project if node.ndim==3 else project[...,None]


def field_value(factors: list[tuple[dict[str,Any],dict[str,Any],dict[str,Any]]],
                weights: np.ndarray,project: np.ndarray,vector: np.ndarray) -> np.ndarray:
    middle=np.zeros_like(vector[0])
    for parts,w,v in zip(factors,weights,vector,strict=True):middle+=w*tensor_action(parts,v)
    middle*=mask_for(project,middle)
    return np.stack([tensor_action(parts,middle) for parts in factors])


def field_jet(factors: list[tuple[dict[str,Any],dict[str,Any],dict[str,Any]]],
              weights: np.ndarray,project: np.ndarray,vector: np.ndarray) -> dict[str,np.ndarray]:
    """Exact physical K jets: BOTH SF sides and every tensor cross contact."""
    middle={s:np.zeros_like(vector[0]) for s in ('','x','y','xy')}
    for parts,w,v in zip(factors,weights,vector,strict=True):
        for suffix in middle:middle[suffix]+=w*tensor_action(parts,v,suffix)
    for v in middle.values():v*=mask_for(project,v)
    rows={s:[] for s in ('','x','y','xy','cross_sides')}
    for parts in factors:
        rows[''].append(tensor_action(parts,middle['']))
        rows['x'].append(tensor_action(parts,middle[''],'x')+tensor_action(parts,middle['x']))
        rows['y'].append(tensor_action(parts,middle[''],'y')+tensor_action(parts,middle['y']))
        cross=tensor_action(parts,middle['y'],'x')+tensor_action(parts,middle['x'],'y')
        rows['cross_sides'].append(cross)
        rows['xy'].append(tensor_action(parts,middle[''],'xy')+cross+tensor_action(parts,middle['xy']))
    return {s:np.stack(v) for s,v in rows.items()}


def weighted_norm(v: np.ndarray,weights: np.ndarray) -> float:
    return float(np.sqrt(sum(w*np.vdot(x,x).real for w,x in zip(weights,v,strict=True))))


def gauge_values(number: int,q: int,chi: np.ndarray) -> np.ndarray:
    a=np.diag(primitive.gauge_generator(number,q,chi,number==5));u=np.diag(primitive.gauge_generator(4,1,chi))
    return a[:,None,None]+u[None,:,None]+u[None,None,:]


def gauge_action(values: np.ndarray,vector: np.ndarray) -> np.ndarray:
    return vector*(values[None,...] if vector.ndim==4 else values[None,...,None])


def make_value_factors(active: Any,auxiliary: Any,nodes: np.ndarray,number: int,la: float,lu: float,
                       x: float,y: float,directions: tuple[np.ndarray,np.ndarray]) -> list[tuple[dict[str,Any],dict[str,Any],dict[str,Any]]]:
    remainder=np.zeros((2,4,4,4),complex);links=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex);factors=[]
    for coordinate in nodes:
        phases=np.array([[0.,coordinate[1],coordinate[2]],[coordinate[0],coordinate[3],coordinate[4]]])+x*directions[0]+y*directions[1]
        sa=consumer.canonical_SF(active.potential(3*phases,remainder,links),number,la)
        su=consumer.canonical_SF(auxiliary.potential(phases,remainder,links),4,lu)
        factors.append((dict(SF=sa),dict(SF=su),dict(SF=su)))
    return factors


def probe(active: Any,auxiliary: Any,nodes: np.ndarray,weights: np.ndarray,vector: np.ndarray,
          number: int,la: float,lu: float,directions: tuple[np.ndarray,np.ndarray],mode: str) -> dict[str,Any]:
    remainder=np.zeros((2,4,4,4),complex);links=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex);factors=[]
    for coordinate in nodes:
        phases=np.array([[0.,coordinate[1],coordinate[2]],[coordinate[0],coordinate[3],coordinate[4]]])
        a=primitive.canonical_jet(active,3*phases,remainder,links,number,la,(3*directions[0],3*directions[1]))
        u=primitive.canonical_jet(auxiliary,phases,remainder,links,4,lu,directions)
        factors.append((a,u,u))
    project=consumer.physical_density_projector(number,3);jet=field_jet(factors,weights,project,vector)
    report: dict[str,Any]=dict(number=number,mode=mode,field_nodes=len(nodes),complete_Fock_per_node=int(np.prod(project.shape)),coherent_Ref_holes=1 if vector.ndim==4 else vector.shape[-1],
        actual_K_source_norm=weighted_norm(jet[''],weights),actual_Kx_source_norm=weighted_norm(jet['x'],weights),actual_Kxy_source_norm=weighted_norm(jet['xy'],weights),
        OMITcross_SF_sides_contact_source_defect=weighted_norm(jet['cross_sides'],weights))
    if mode=='gauge':
        chi=np.array([.31,-.17]);g=gauge_values(number,3,chi)
        if np.max(abs(g[project]))>3e-14:raise ArithmeticError('Exact full Gauss midpoint does not annihilate centred gauge generator')
        gu=gauge_action(g,vector);g2u=gauge_action(g,gu);kg=field_value(factors,weights,project,gu);kg2=field_value(factors,weights,project,g2u)
        expected=1j*(gauge_action(g,jet[''])-kg)
        mixed=-gauge_action(g,gauge_action(g,jet['']))+2*gauge_action(g,kg)-kg2
        first=weighted_norm(jet['x']-expected,weights);second=weighted_norm(jet['xy']-mixed,weights)
        report.update(first_fullK_Ward_defect=first,mixed_fullK_Ward_defect=second)
        if max(first,second)>3e-8:raise ArithmeticError('Original-field FULLK current/contact violates U1 gauge Ward')
    else:
        def value(x: float,y: float) -> np.ndarray:
            return field_value(make_value_factors(active,auxiliary,nodes,number,la,lu,x,y,directions),weights,project,vector)
        rows=[]
        for step in (.004,.002):
            first=(value(step,0)-value(-step,0))/(2*step)
            mixed=(value(step,step)-value(step,-step)-value(-step,step)+value(-step,-step))/(4*step*step)
            rows.append(dict(step=step,first_absolute_defect=weighted_norm(first-jet['x'],weights),mixed_absolute_defect=weighted_norm(mixed-jet['xy'],weights)))
        report['actual_nonlinear_FULLK_finite_differences']=rows
        if rows[-1]['first_absolute_defect']>2e-4 or rows[-1]['mixed_absolute_defect']>2e-4:raise ArithmeticError('Physical field K jets differ from actual full nonlinear transfer')
    return report


def run() -> dict[str,Any]:
    parameters=consumer.reference.NativeParameters();active,_,_=consumer.reference.native_law(parameters);auxiliary,_,_=consumer.reference.native_law(parameters);auxiliary.kappa=.08
    va,la=consumer.meanbody(active);vu,lu=consumer.meanbody(auxiliary);nodes,weights,boundary=consumer.compact_nodes(8,20261009,3.5,100.)
    boundary/=np.sqrt(weights@boundary**2);body=np.einsum('a,b,c->abc',va,vu,vu);neutral_input=np.stack([f*body for f in boundary])
    zero=np.zeros((2,3));base=make_value_factors(active,auxiliary,nodes,4,la,lu,0,0,(zero,zero));prepared=field_value(base,weights,consumer.physical_density_projector(4,3),neutral_input)
    creators=consumer.source_creators();charged_input=np.stack([np.stack([(c@v.reshape(70,-1)).reshape(56,70,70) for c in creators],axis=-1) for v in prepared])
    raw=weighted_norm(charged_input,weights)**2/weighted_norm(prepared,weights)**2
    if abs(raw-1)>3e-11:raise ArithmeticError('Actual original FULLK fourRef source lost raw CAR/C normalization')
    charged_input/=weighted_norm(charged_input,weights)
    chi=np.array([.31,-.17]);gradient=np.zeros((2,3));gradient[:,0]=[chi[0]-chi[1],chi[1]-chi[0]]
    dy=np.zeros((2,3));dz=np.zeros((2,3));dy[:,1]=[1.,-1.];dz[:,2]=[1.,-1.];dy/=np.sqrt(200.);dz/=np.sqrt(200.)
    rows=[]
    for number,vector in ((4,neutral_input),(5,charged_input)):
        rows.append(probe(active,auxiliary,nodes,weights,vector,number,la,lu,(gradient,gradient),'gauge'))
        rows.append(probe(active,auxiliary,nodes,weights,vector,number,la,lu,(dy,dz),'transverse'))
    inputs=dict(Naux=2,kaux=.08,Q=3,geometry=[2,1,1],all_compact_cycles=5,statistical_sector=[0,0,0],Phi='0',colour='I',order=8,seed=20261009,bare_mass=parameters.critical_bare_mass,spacing=parameters.spacing,
        scalar_origins=[la,lu],beta_CALIBRATION=100.,neutral_input='ORIGINALfixedmeanbody times periodizedGaussian3.5; no source/sea gate',charged_input='ALLfour coherentRef holes from raw S on actual originalFULLK neutral preparation, normalized only for the probe; same operator K on every Refhole',
        units='Kjets dimensionless; canonical photon probe directions theta=A/sqrt(beta_CALIBRATION), activeQ/aux1. Physical H=-log(K/lambdaN)/a current and lambdaN vacuum derivative NOT implemented/inferred.',
        construction='Original field-space K=VVdag; complete rho0 midpoint and ALLcanonical Fock; scalar+matrix-log+exponential+tensor-cross+both-SF-side contacts retained. No differentiated Abar/polar shortcut, no bareE/RPA, no fullPhi/SU4/generalNF/macro/pole/current-amputation/Compton claim.',
        gauge_chi=chi.tolist(),transverse_directions=[dy.tolist(),dz.tolist()])
    return dict(original_fullK_RAW_fourRef_CAR_Gram=raw,rows=rows,input=inputs)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();started=time.time();report=run();inputs=report['input']
    report['execution']=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),primitive_at_IMPORT=hashlib.sha256(PRIMITIVE_BYTES).hexdigest(),primitive_bytes_at_IMPORT=PRIMITIVE_BYTES.decode(),consumer_at_IMPORT=hashlib.sha256(primitive.CONSUMER_BYTES).hexdigest(),reference_at_IMPORT=hashlib.sha256(consumer.REFERENCE_BYTES).hexdigest(),ledger_at_IMPORT=hashlib.sha256(consumer.ALPHA_BYTES).hexdigest(),input_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),input_json=inputs,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,runtime=platform.platform(),startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(cast(str,primitive.__file__)).read_bytes()==PRIMITIVE_BYTES and Path(consumer.__file__).read_bytes()==primitive.CONSUMER_BYTES,elapsed_seconds=time.time()-started)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('input','execution')}))


if __name__=='__main__':main()
