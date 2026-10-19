"""Independent finite-CAR/Hermite checks of actual paired ordering and C3.

This is an algebra boundary, never a fullN4 receiver result. Frozen all12
coefficients and physical source remain the separate actual consumers.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, kron, eye, block_diag
from threadpoolctl import threadpool_limits
import joined_phi_u1_nonlocal_paired_higher as paired
import joined_phi_u1_current_signal_interval_certificate as ar
ROOT=Path(__file__).resolve().parent

def sector_one(matrix:np.ndarray,count:int)->tuple[Any,list[int]]:
    states=[s for s in range(256) if s.bit_count()==count];lookup={s:i for i,s in enumerate(states)};rows=[];cols=[];vals=[]
    for j,s in enumerate(states):
        for b in range(8):
            if not s>>b&1:continue
            t=s^(1<<b);sgn=(-1)**((s&((1<<b)-1)).bit_count())
            for a in range(8):
                if t>>a&1 or matrix[a,b]==0:continue
                u=t|(1<<a);rows.append(lookup[u]);cols.append(j);vals.append(sgn*(-1)**((t&((1<<a)-1)).bit_count())*matrix[a,b])
    return csr_matrix((vals,(rows,cols)),shape=(len(states),len(states))),states

def creation(site:int,colour:int,bg:list[int],src:list[int])->Any:
    lookup={s:i for i,s in enumerate(src)};rows=[];cols=[];vals=[];a=4*site+colour
    for j,s in enumerate(bg):
        if s>>a&1:continue
        rows.append(lookup[s|(1<<a)]);cols.append(j);vals.append(.5*(-1)**((s&((1<<a)-1)).bit_count()))
    return csr_matrix((vals,(rows,cols)),shape=(len(src),len(bg)))

def run(output:Path)->dict[str,Any]:
    started=time.monotonic();gamma=np.asarray(paired.body.joined.old.full.supplier.primitive.SELECTED)
    captured={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':
            captured[Path(file).name]=Path(file).read_bytes()
    hop=.23*gamma[0]+.11j*gamma[1];on=.31*gamma[0]
    wind=np.zeros((2,192),np.int64);wind[0,0]=1;wind[1,0]=-1
    d=np.zeros((64,192),np.int64);d[0,0]=1;d[1,0]=-1
    delta=np.zeros((2,64),np.int64);delta[0,:2]=[1,-1];delta[1,:2]=[-1,1]
    lint=np.zeros((64,64),np.int64);lint[:2,:2]=[[1,-1],[-1,1]]
    geo={'k':wind,'D':d,'x':np.array([0,1]),'y':np.array([1,0]),'delta':delta,'lint':lint,'den':8}
    p:Any=np.zeros((256,256),complex);p[:8,:8]=np.kron(np.ones((2,2))*.5,np.eye(4))
    for s in range(2,64):p[4*s:4*s+4,4*s:4*s+4]=np.diag([1,1,0,0])
    aux=np.kron(np.eye(64),np.diag([1,1,0,0]))
    ps=[ar.Ball(p),ar.Ball(aux)];bs=[ar.Ball(np.array([hop,hop.conj().T])),ar.Ball(np.zeros((2,4,4)))];ons=[ar.Ball(on),ar.Ball(np.zeros((4,4)))]
    engine=paired.Engine(geo,bs,ons,ps);terms=[paired.Term(F(1),1,ar.Ball(np.eye(4)),(),tuple(map(int,wind[1])),(),1)];initial={}
    for n in range(4):
        if n in (0,1,3):initial[str(n)]=paired.enclosed(engine.evaluate(terms))
        if n<3:terms=engine.expand(terms)
    h=np.zeros((8,8),complex);h[:4,:4]=on;h[4:,4:]=on;h[:4,4:]=hop;h[4:,:4]=hop.conj().T
    hg,bg=sector_one(h,4);hs,src=sector_one(h,5)
    hopone=np.zeros((8,8),complex);hopone[:4,4:]=hop
    bgforward,_=sector_one(hopone,4);srcforward,_=sector_one(hopone,5)
    onsite=np.zeros((8,8),complex);onsite[:4,:4]=on;onsite[4:,4:]=on
    bgon,_=sector_one(onsite,4);srcon,_=sector_one(onsite,5)
    rb=np.array([sum((s>>a)&1 for a in range(4))-sum((s>>a)&1 for a in range(4,8)) for s in bg],float)
    rs=np.array([sum((s>>a)&1 for a in range(4))-sum((s>>a)&1 for a in range(4,8))-1 for s in src],float)
    bgHC=csr_matrix(np.diag(8*rb**2));srcHC=csr_matrix(np.diag(8*rs**2))
    occupied=np.vstack((np.eye(4),np.eye(4)))/math.sqrt(2)
    psi=np.array([np.linalg.det(occupied[[a for a in range(8) if s>>a&1]]) for s in bg],complex)
    c0=[creation(0,a,bg,src) for a in range(4)];cx=[creation(1,a,bg,src) for a in range(4)]
    observations=[];wrong=[]
    for levels in (24,36):
        lowering=np.diag(np.sqrt(np.arange(1,levels)),1);theta=math.sqrt(2)*(lowering+lowering.T);pi=1j*(lowering.T-lowering)/(2*math.sqrt(2));phase=csr_matrix(expm(1j*theta/math.sqrt(2)));ident=eye(levels,format='csr')
        HB=kron(csr_matrix(64*pi@pi),eye(len(bg)))+kron(ident,bgHC+bgon)+kron(phase,bgforward)+kron(phase.conj().T,bgforward.conj().T)
        HS=kron(csr_matrix(64*pi@pi),eye(len(src)))+kron(ident,srcHC+srcon)+kron(phase,srcforward)+kron(phase.conj().T,srcforward.conj().T)
        bgpsi=np.kron(np.eye(levels)[:,0],psi);source=[kron(ident,c)@bgpsi for c in c0];ex=[kron(phase.conj().T,c) for c in cx]
        norm=sum(float(np.vdot(v,v).real) for v in source)
        def derivatives(hb:Any,hs:Any)->dict[str,complex]:
            b=[bgpsi];s=[[v] for v in source]
            for _ in range(3):
                b.append(hb@b[-1])
                for v in s:v.append(hs@v[-1])
            result={}
            for n in (0,1,3):
                result[str(n)]=sum((-1)**a*math.comb(n,a)*sum(np.vdot(ex[c]@b[a],s[c][n-a]) for c in range(4)) for a in range(n+1))
            return result
        actual=derivatives(HB,HS)
        noHC=derivatives(HB-kron(ident,bgHC),HS-kron(ident,srcHC))
        half=derivatives(HB-kron(csr_matrix(32*pi@pi),eye(len(bg))),HS-kron(csr_matrix(32*pi@pi),eye(len(src))))
        noRefHC=csr_matrix(np.diag(8*(rs+1)**2));noRef=derivatives(HB,HS+kron(ident,noRefHC-srcHC))
        observations.append({'levels':levels,'raw_source_norm_squared':norm,'m':{k:{'real':complex(v).real,'imaginary':complex(v).imag} for k,v in actual.items()},'analytic_m3_defect':abs(actual['3']-initial['3']['centre']['real']-1j*initial['3']['centre']['imaginary'])})
        wrong.append({'levels':levels,'HC_m3_contrast':abs(noHC['3']-actual['3']),'HEhalf_m3_contrast':abs(half['3']-actual['3']),'missing_source_Refhole_m3_contrast':abs(noRef['3']-actual['3']),'wrong_scalar_normalized_m3_contrast':abs(actual['3']),'wrong_source_autocorrelation_C0_contrast':abs(norm-actual['0'])})
    defect=observations[-1]['analytic_m3_defect'];scale=max(1,abs(initial['3']['centre']['real']))
    agrees=defect<2e-6*scale and abs(observations[-1]['raw_source_norm_squared']-.5)<1e-12
    if paired.__file__ is None:
        raise ArithmeticError('Paired producer has no source file binding')
    result={'status':'INDEPENDENT_PAIRED_CAR_HERMITE_AGREES' if agrees else 'INDEPENDENT_PAIRED_CAR_HERMITE_FAILED_REGIME','scope':'Finite algebra boundary only, no all12/fullN4 evolution claim','initial_ordered_engine':initial,'direct_same_H_paired':observations,'adverse_controls':wrong,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'producer_sha256':hashlib.sha256(Path(paired.__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-started,'runtime':{'python':sys.version,'executable':sys.executable}}
    if any((ROOT/name).read_bytes()!=raw for name,raw in captured.items()):
        raise ArithmeticError('Independent control source bytes changed during execution')
    archive=output.with_suffix('.zip')
    with paired.zipfile.ZipFile(archive,'w',paired.zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captured.items()):z.writestr(name,raw)
        z.writestr('boundary_exact_preparation.json',json.dumps({'projector8_real':p[:8,:8].real.tolist(),'projector8_imaginary':p[:8,:8].imag.tolist(),'Ref4to3':True,'raw_E0_factor':'1/2','density_Lplus':[['1/8','-1/8'],['-1/8','1/8']],'HE':'64','HC':'64','Theta_density_variance':'2'}))
    result['input_sha256']={name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()}
    result['immutable_inputs']={'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.output)
