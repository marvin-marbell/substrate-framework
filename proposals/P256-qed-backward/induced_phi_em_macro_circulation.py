"""Actual macro two-link transverse norm via exact physical network bracketing."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import base64
import zlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import cg
from induced_phi_em_macro_squeeze import SOURCE_BYTES as SQUEEZE_BYTES
from induced_phi_em_gain_certificate import encoded

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**SQUEEZE_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
PLAN='issue-222-induced-photon-delta13-continuation-20261008'
REVISION='8d37944a271479ff0ea3f97aa4ad022c3a0aaa4779384dfc95deb089890f6547'


def grid_laplacian(n: int,wired: bool) -> Any:
    diagonal=np.full(n,2.)
    if not wired:diagonal[[0,-1]]=1.
    one=diags(diagonal,0,shape=(n,n),format='csr')+diags(-np.ones(n-1),-1,shape=(n,n),format='csr')+diags(-np.ones(n-1),1,shape=(n,n),format='csr');I=eye(n,format='csr')
    return kron(kron(one,I),I,format='csr')+kron(kron(I,one),I,format='csr')+kron(kron(I,I),one,format='csr')


def exact_square_sum(arrays: list[Any]) -> int:
    return sum(int(x)*int(x) for array in arrays for x in array.flat)


def packed(array: Any) -> dict[str,Any]:
    raw=np.asarray(array,dtype='<i8').tobytes()
    return dict(shape=list(array.shape),dtype='little-endian signed int64',sha256=hashlib.sha256(raw).hexdigest(),zlib_base64=base64.b64encode(zlib.compress(raw)).decode())


def solve_voltage(n: int,source: tuple[int,int,int],target: tuple[int,int,int],wired: bool) -> Any:
    b=np.zeros((n,n,n));b[source]=1.;b[target]=-1.
    # Solver is ONLY a proposal for a rational primal/dual witness. Its status/residual is not the certificate.
    v,info=cg(grid_laplacian(n,wired),b.ravel(),rtol=1e-12,atol=0.,maxiter=10000)
    if info!=0:raise ArithmeticError(('No voltage proposal',info))
    return v.reshape(n,n,n)


def run(source_path: Path,radius: int) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();source=json.loads(raw);state=source['input_descriptor'];N=int(state['N']);V=int(state['V'])
    if radius<2 or 2*radius+3>=N:raise ValueError('Exact embedded patch must avoid physicaltorus wraps')
    scale=2**50;ni=2*radius-1;nf=2*radius+1
    si=(radius-2,radius-1,radius-1);ti=(radius,radius-1,radius-1)
    sf=(radius-1,radius,radius);tf=(radius+1,radius,radius)
    voltage=solve_voltage(ni,si,ti,True);w=np.rint(voltage*scale).astype(np.int64)
    jumps=[np.diff(w,axis=axis) for axis in range(3)]
    # Wired outside is atzero; each boundary face contributes original outward edge energy.
    faces=[np.take(w,index,axis=axis) for axis in range(3) for index in (0,-1)]
    wired_energy=exact_square_sum(jumps+faces)
    lower=F(2*scale*(int(w[si])-int(w[ti]))-wired_energy,scale*scale)
    wired_witness=packed(w)
    voltage=solve_voltage(nf,sf,tf,False);w=np.rint(voltage*scale).astype(np.int64)
    if int(np.abs(w).max())*12>=2**63:raise ArithmeticError('Integer witness exceeds exact int64 accumulation range')
    flow=[-np.diff(w,axis=axis) for axis in range(3)];div=np.zeros_like(w)
    for axis,j in enumerate(flow):
        left=[slice(None)]*3;right=[slice(None)]*3;left[axis]=slice(None,-1);right[axis]=slice(1,None)
        div[tuple(left)]+=j;div[tuple(right)]-=j
    residual=-div;residual[sf]+=scale;residual[tf]-=scale
    if sum(abs(int(x)) for x in residual.flat)>=2**62:raise ArithmeticError('Tree repair accumulation could overflow')
    for x in range(nf-1,-1,-1):
        for y in range(nf-1,-1,-1):
            for z in range(nf-1,-1,-1):
                value=int(residual[x,y,z])
                if x:flow[0][x-1,y,z]-=value;residual[x-1,y,z]+=value
                elif y:flow[1][0,y-1,z]-=value;residual[0,y-1,z]+=value
                elif z:flow[2][0,0,z-1]-=value;residual[0,0,z-1]+=value
    if int(residual[0,0,0])!=0:raise ArithmeticError('Unit-flow residual is not neutral')
    # Independently evaluate repaired incidence exactly; no approximate-current promotion.
    div=np.zeros_like(w)
    for axis,j in enumerate(flow):
        left=[slice(None)]*3;right=[slice(None)]*3;left[axis]=slice(None,-1);right[axis]=slice(1,None)
        div[tuple(left)]+=j;div[tuple(right)]-=j
    wanted=np.zeros_like(w);wanted[sf]=scale;wanted[tf]=-scale
    if not np.array_equal(div,wanted):raise ArithmeticError('Rational witness does not obey exact Gauss')
    upper=F(exact_square_sum(flow),scale*scale)
    u2=(2-upper,2-lower)
    if lower<=0 or lower>upper or u2[0]<=0 or u2[1]>=2:raise ArithmeticError('Invalid physical network bracket')
    descriptor=dict(source_sha256=hashlib.sha256(raw).hexdigest(),source_input_sha256=source['input_sha256'],N=N,V=V,M=int(state['M']),radius=radius,dyadic_scale=str(scale),
        a=state['a'],UE=state['UE'],charged_counts=state['charged_counts'],plan=PLAN,plan_revision=REVISION,inherited_source_plan_revision=state['plan_revision'])
    witness=dict(wired_dual=dict(n=ni,source=si,target=ti,voltage_integer_witness=wired_witness,
        exact_edge_plus_boundary_energy_integer=str(wired_energy),lower=encoded(lower)),free_primal=dict(n=nf,source=sf,target=tf,
        flow_integer_witness=[packed(j) for j in flow],exact_incidence_identity=True,exact_flow_energy_integer=str(upper.numerator*(scale*scale//upper.denominator)),upper=encoded(upper)))
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        actual_macro_two_link_Q_norm_squared_interval=[encoded(x) for x in u2],actual_macro_dipole_Coulomb_resistance_interval=[encoded(lower),encoded(upper)],
        one_link_Q_norm_squared_exact=encoded(F(2,3)+F(1,3*V)),rational_witness_summary=witness,
        proof='Dincidence isliteralunitcubic torus; PT=I-D^T LplusD includesALLharmonics. Straight2link ell hasnorm²2 andDell=delta_target-delta_source, hence||Q^Tell||²=2-Reffective. Freeembeddedpatch deletesoutsideedges: resistanceUPPER. Wiredpatch contractsALLoutside/boundarynodes: resistanceLOWER. Rayleighmonotonicity thereforeboundsACTUALN torus, neverpromotespatchasphysicalsize. Rationaldual 2b.v-v.Lv isLOWER foranydyadicv. Rationalrepairedunitflow withEXACTdivergenceb hasenergyUPPER byThomson. Everycertificatearithmetic integer/Fraction; scipyCG onlyconstructswitness.',
        scope='OriginalrealGaussoperator/localnetworkcertificate, notnewlaw or photonpole. NoCAR/source/Hamiltonianmodification. Nextconsumer maytightenactualquantumGaussianphasefactors andvariationalwidth usingthisexactoperatorbound.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('source','output'):parser.add_argument('--'+role,type=Path,required=True)
    parser.add_argument('--radius',type=int,required=True);args=parser.parse_args();report=run(args.source,args.radius);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','actual_macro_two_link_Q_norm_squared_interval','actual_macro_dipole_Coulomb_resistance_interval','elapsed_seconds')},indent=2))
