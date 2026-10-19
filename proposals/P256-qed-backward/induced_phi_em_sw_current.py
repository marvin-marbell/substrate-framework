"""Full-Href causal initial moment on the noncommuting PH/momentum source."""
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
from induced_phi_em_sw_momentum import SOURCE_BYTES as SW_BYTES, build_source, section
from induced_phi_em_gaussian_variational import build_native
from induced_phi_em_causal_current import make_operators, commutator, remainder_bound

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**SW_BYTES,'induced_phi_em_causal_current.py':(ROOT/'induced_phi_em_causal_current.py').read_bytes(),Path(__file__).name:Path(__file__).read_bytes()}


def group(events: dict,dimension: int) -> dict[tuple[int,...],np.ndarray]:
    result={}
    for (i,j,phase),coeff in events.items():
        if phase not in result:result[phase]=np.zeros((2,dimension,dimension),complex)
        result[phase][:,i,j]+=coeff
    return result


class Words:
    def __init__(self,model: dict[str,Any],z: np.ndarray,kappa: float,variance: float):
        self.model=model;self.z=z;self.kappa=kappa;self.variance=variance;self.cache={}

    def prepare(self,phase: tuple[int,...]) -> tuple[float,list[dict[str,Any]]]:
        if phase in self.cache:return self.cache[phase]
        b=self.model['native']['Q'].T@np.asarray(phase);prepared=[]
        for profile in self.model['profiles']:
            left=section(profile,self.z+b/2,self.kappa);right=section(profile,self.z-b/2,self.kappa)
            U,s,Vh=np.linalg.svd(left.conj().T@right);V=Vh.conj().T
            ph=complex(np.linalg.det(U)*np.linalg.det(Vh));prefix=np.r_[1.,np.cumprod(s)];suffix=np.r_[np.cumprod(s[::-1])[::-1],1.]
            weights=prefix[:-1]*suffix[1:];second=np.zeros((len(s),len(s)))
            for i in range(len(s)):
                middle=1.
                for j in range(i+1,len(s)):
                    second[i,j]=second[j,i]=prefix[i]*middle*suffix[j+1];middle*=s[j]
            L=U.conj().T@left.conj().T;R=right@V
            prepared.append(dict(chi=ph*prefix[-1],N=ph*(R*weights)@L,L=L,R=R,phase=ph,second=second,load=profile['load']))
        result=(float(np.exp(-self.variance*float(b@b)/2)),prepared);self.cache[phase]=result
        return result

    @staticmethod
    def spectators(prepared: list[dict[str,Any]],roles: tuple[int,...]) -> complex:
        result=1+0j
        for r,item in enumerate(prepared):result*=item['chi']**(item['load']-roles.count(r))
        return result

    def one(self,phase: tuple[int,...],A: np.ndarray) -> complex:
        factor,prepared=self.prepare(phase);result=0j
        for r,item in enumerate(prepared):
            result+=item['load']*self.spectators(prepared,(r,))*np.einsum('ij,ji->',A[r],item['N'])
        return complex(factor*result)

    def pair(self,phase: tuple[int,...],A: np.ndarray,B: np.ndarray) -> complex:
        factor,prepared=self.prepare(phase);result=0j;firstA=[];firstB=[]
        for r,item in enumerate(prepared):
            firstA.append(np.einsum('ij,ji->',A[r],item['N']));firstB.append(np.einsum('ij,ji->',B[r],item['N']))
            a=item['L']@A[r]@item['R'];b=item['L']@B[r]@item['R']
            word=np.einsum('ij,jk,ki->',A[r],B[r],item['N'])+item['phase']*np.sum(item['second']*(np.outer(np.diag(a),np.diag(b))-a*b.T))
            result+=item['load']*self.spectators(prepared,(r,))*word
        for r,item in enumerate(prepared):
            if item['load']>=2:
                result+=item['load']*(item['load']-1)*self.spectators(prepared,(r,r))*firstA[r]*firstB[r]
            for t,other in enumerate(prepared):
                if r!=t:result+=item['load']*other['load']*self.spectators(prepared,(r,t))*firstA[r]*firstB[t]
        return complex(factor*result)


def operators(native: dict[str,Any],direction: np.ndarray) -> dict[str,Any]:
    identity=dict(native=native,W=np.eye(4),profiles=[dict(load=p['load']) for p in native['Pprofiles']])
    H,J=make_operators(identity,direction);double=commutator(commutator(H,J),J);dimension=4*native['V']
    currents=group(J,dimension);charges={}
    for (i,j,phase),value in J.items():
        d=np.zeros(native['V']);d[i//4]+=1;d[j//4]-=1
        if phase in charges and not np.array_equal(charges[phase],d):raise ArithmeticError('Current phase group has inconsistent physical charge displacement')
        charges[phase]=d
    contacts={}
    for phase,A in group(H,dimension).items():contacts[phase]=-(np.asarray(phase)@direction)**2*A
    return dict(J=currents,matter=group(double,dimension),contact=contacts,charges=charges,
                remainder=remainder_bound(identity,float(native['variance']),direction))


def measure(model: dict[str,Any],ops: dict[str,Any],z: np.ndarray,kappa: float,variance: float) -> dict[str,complex]:
    native=model['native'];words=Words(model,z,kappa,variance);g=native['UE']/native['a']
    current=sum(words.one(q,A) for q,A in ops['J'].items());contact=sum(words.one(q,A) for q,A in ops['contact'].items())
    matter=sum(words.one(q,A) for q,A in ops['matter'].items());electric=0j;coulomb=0j;second=0j
    rows=list(ops['J'].items())
    for phase,A in rows:
        bi=native['Q'].T@np.asarray(phase);di=ops['charges'][phase];qi=di@native['Lplus'];ci=float(di@native['Lplus']@di)
        Q=np.stack([np.diag(np.repeat(qi,4))]*2);background=128*float(qi.sum())
        for other,B in rows:
            bj=native['Q'].T@np.asarray(other);dj=ops['charges'][other];q=tuple(a+b for a,b in zip(phase,other))
            comm=A@B-B@A;AB=words.pair(q,A,B);BA=words.pair(q,B,A);one=words.one(q,comm)
            second+=AB
            electric+=g*(2*float(bi@z)*one+float(bi@bj)*(AB+BA))
            coulomb+=g*(2*(words.pair(q,comm,Q)-background*one)+ci*one+2*float(di@native['Lplus']@dj)*AB)
    return dict(current=current,contact=contact,second=second,Peierls_CAR=matter,
                primitive_electric=electric,full_real_Gauss_Coulomb=coulomb,retarded_initial_tangent=matter+electric+coulomb)


def run(mean_path: Path,source_path: Path,samples: int) -> dict[str,Any]:
    started=time.monotonic();meanraw=mean_path.read_bytes();raw=source_path.read_bytes();source=json.loads(raw)
    if source['input_descriptor']['N']!=2:raise ValueError('No silent macro orN6 source reduction')
    native=build_native(json.loads(meanraw),2);variance=float(source['input_descriptor']['variance']);native['variance']=variance
    model=build_source(native);kappa=float(source['selected']['kappa']);direction=np.zeros(3*native['V'])
    direction[1::3]=np.cos(np.pi*native['sites'][:,0]);direction/=np.linalg.norm(direction)
    ops=operators(native,direction);rng=np.random.default_rng(2221019);points=rng.normal(size=(samples,native['M']))/np.sqrt(4*variance)
    rows=[]
    for z in points:
        plus=measure(model,ops,z,kappa,variance);minus=measure(model,ops,-z,kappa,variance)
        rows.append({key:(plus[key]+minus[key])/2 for key in plus})
    means={key:sum(row[key] for row in rows)/samples for key in rows[0]}
    errors={key:float(np.std([row[key].real for row in rows],ddof=1)/np.sqrt(samples)) for key in rows[0]}
    seed=measure(model,ops,points[0],0.,variance);seed2=measure(model,ops,-points[0],0.,variance)
    seed={key:(seed[key]+seed2[key])/2 for key in seed}
    probe=points[0];C=np.kron(np.eye(native['V']),CHARGE_CONJUGATION);cp=[];noncomm=[]
    for profile in model['profiles']:
        left=section(profile,probe,kappa);right=section(profile,-probe,kappa);P=left@left.conj().T
        cp.append(float(np.linalg.norm(right@right.conj().T-(np.eye(len(P))-C@P.conj()@C.conj().T))))
        noncomm.append(max(float(np.linalg.norm(profile['B'][i]@profile['B'][j]-profile['B'][j]@profile['B'][i])) for i in range(native['M']) for j in range(i)))
    cap=ops['remainder']['retarded_kernel_second_derivative_upper'];t=.02*abs(means['retarded_initial_tangent'].real)/cap
    descriptor=dict(mean_sha256=hashlib.sha256(meanraw).hexdigest(),source_sha256=hashlib.sha256(raw).hexdigest(),N=2,V=8,M=17,
        counts=[1,63],kappa=kappa,variance=variance,independent_antithetic_pairs=samples,
        probe='LiteralcoexactmatterJy q=(pi,0,0), Df0; UVnative, notmacroIR',plan='issue-222-induced-photon-delta13-continuation-20261008',
        plan_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        actual_prepared=means,standard_errors=errors,Gaussian_seed=seed,prepared_minus_seed_tangent=means['retarded_initial_tangent']-seed['retarded_initial_tangent'],
        CP_density_intertwining_errors=cp,actual_noncommuting_generator_norms=noncomm,
        priced_short_time=dict(t=t,retarded_linear_value=t*means['retarded_initial_tangent'],absolute_remainder_upper=t*t*cap/2,
            purpose='Diagnosticpricechosenfromfullgraph, NOTphysicalclock/speed fit'),remainder=ops['remainder'],
        exact_kernel='chi_R(t)=-i theta(t)<[J_f(t),J_f(0)]> inactualfixedjointsource; chi_R(0)=0, mu=<[[Href,J_f],J_f]>. Piweightedcrossmomentum words, originalall64CAR, completeoperatorGaussCoulomb includingbetweenflavour/current correlations. No conditionalThetaSlater assumption or missingBornHuang.',
        limitations='FiniteN2meanmidpoint GaussianMC, SE notintervalcertification. Native initialmoment and conservativefullHrefshorttimeformula, NOTfiniteclock4096 propagation, stationarypole/residue, photonvacuum, commoncone ormacro/QEDadmission.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},elapsed_seconds=time.monotonic()-started)


def serialize(value: Any) -> Any:
    if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
    raise TypeError(type(value).__name__)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','source','output'):parser.add_argument('--'+role,type=Path,required=True)
    parser.add_argument('--samples',type=int,required=True);args=parser.parse_args()
    if args.samples<2:raise ValueError('Actual SEM requires two independent antithetic pairs')
    report=run(args.mean,args.source,args.samples);args.output.write_text(json.dumps(report,indent=2,default=serialize)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','actual_prepared','standard_errors','Gaussian_seed','prepared_minus_seed_tangent','CP_density_intertwining_errors','actual_noncommuting_generator_norms','priced_short_time','elapsed_seconds')},indent=2,default=serialize))
