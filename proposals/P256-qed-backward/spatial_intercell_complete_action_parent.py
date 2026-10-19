"""Complete Gaussian-frame first-H action at the actual N4 flat-field point.

Pointwise action components are NOT Haar-integrated physical source variance.
The whole canonical exterior is retained; no PHP-squared replacement.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile
import numpy as np
from scipy.linalg import expm, expm_frechet
from threadpoolctl import threadpool_limits
import spatial_intercell_source as source

ROOT=Path(__file__).resolve().parent
N=4
G=64
COPIES=64
ORDER=8

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def archive(path: Path, captured: dict[str,bytes]) -> dict:
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captured.items()):z.writestr(name,raw)
    return {'path':str(path),'sha256':sha(path.read_bytes()),'members':{n:sha(b) for n,b in captured.items()}}

def cell_second(theta: np.ndarray,gam: np.ndarray) -> np.ndarray:
    U=np.eye(16,dtype=complex);second=np.zeros((4,16,16),complex)
    for k,(x,y,axis) in enumerate(((0,1,1),(1,3,2),(2,3,1),(0,2,2))):
        a,b=slice(4*x,4*x+4),slice(4*y,4*y+4)
        S=np.zeros((16,16),complex);v=np.exp(1j*theta[k])*1j*gam[0]@gam[axis]
        S[a,b]=v;S[b,a]=-v.conj().T
        J=np.zeros_like(S);J[a,a]=np.eye(4);J[b,b]=np.eye(4)
        gate=np.eye(16)+(np.cos(source.seed.T)-1)*J-np.sin(source.seed.T)*S
        second=np.einsum('ij,ejk->eik',gate,second)
        second[k]+=np.sin(source.seed.T)*S@U
        U=gate@U
    return second

def path_second(native:dict,profile:dict,theta:np.ndarray,edge:int) -> np.ndarray:
    h=np.zeros((4*native['V'],4*native['V']),complex)
    for displacement,coef in profile['raw_paths']:
        for p in native['Pprofiles'][0]['paths']:
            if not np.array_equal(displacement,p['displacement']) or edge not in p['edges']:continue
            count=sum(int(e==edge) for e in p['edges'])
            a,b=slice(4*p['source'],4*p['source']+4),slice(4*p['target'],4*p['target']+4)
            block=-count**2*p['seam']*np.exp(1j*sum(theta[e] for e in p['edges']))*np.einsum('m,mij->ij',coef,native['frame'])
            h[a,b]+=block;h[b,a]+=block.conj().T
    return h

def exponential_derivatives(A:np.ndarray,E:np.ndarray,F:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    term=np.eye(len(A),dtype=complex);first=np.zeros_like(A);second=np.zeros_like(A)
    total=term.copy();dt=first.copy();ddt=second.copy()
    for m in range(1,ORDER+1):
        term,first,second=term@A/m,(first@A+term@E)/m,(second@A+2*first@E+term@F)/m
        total+=term;dt+=first;ddt+=second
    return total,dt,ddt

def translations(native:dict,dimension:int) -> list[tuple[np.ndarray,np.ndarray]]:
    sites=native['sites'];idx={tuple(s):i for i,s in enumerate(sites)};out=[]
    for x in range(0,N,2):
        for y in range(0,N,2):
            for z in range(N):
                shift=np.array([x,y,z]);target=np.array([idx[tuple((s+shift)%N)] for s in sites])
                sign=np.array([(-1)**int(np.sum((s+shift)//N)) for s in sites])
                permutation=(dimension*target[:,None]+np.arange(dimension)).reshape(-1)
                inverse=np.argsort(permutation);sign=np.repeat(sign,dimension)[inverse]
                out.append((inverse,sign))
    return out

def move(matrix:np.ndarray,translation:tuple[np.ndarray,np.ndarray]) -> np.ndarray:
    inverse,sign=translation
    return matrix[np.ix_(inverse,inverse)]*sign[:,None]*sign[None,:]

def action_case(native:dict,data:dict,tau:float) -> dict:
    V=native['V'];d=4*V;theta=np.zeros(3*V)
    c=source.construction(native,theta);P=c['P'];U0=c['U0'];p0=c['p0'];mask=c['mask']
    roles=data['profiles'][1:];hs=[source.path_matrix(native,r,theta) for r in roles]
    hic=np.where(mask,0,hs[0]);S=hic@P-P@hic;A=-tau*S;U=expm(A);B=U@U0
    a=float(np.linalg.norm(A,2))
    gam=native['frame']/np.sqrt(np.trace(native['frame']@native['frame'],axis1=1,axis2=2).real/4)[:,None,None]
    _,frame=np.linalg.eigh(gam[0]);W0=np.kron(np.eye(V),frame[:,:2]);Y0=np.kron(np.eye(V),frame[:,2:])
    full_trans=translations(native,4);half_trans=translations(native,2)
    assert max(np.max(abs(move(hs[0],tr)-hs[0])) for tr in full_trans)<1e-10
    assert max(np.max(abs(move(U0,tr)-U0)) for tr in full_trans)<1e-12
    sites=source.seed.tiles(native)[0];representatives=[3*int(s)+axis for s in sites for axis in range(3)]
    W=B@W0;Ms=[Y0.conj().T@B.conj().T@h@W for h in hs]
    zero=sum(int(r['load'])*np.trace(W.conj().T@h@W) for r,h in zip(roles,hs))
    electric=np.zeros_like(Ms[0]);Ks=[];means=[];tail=[];frechet=[]
    for edge in representatives:
        up=np.zeros_like(P);upp=np.zeros_like(P);pp=np.zeros_like(P)
        if edge in c['derivatives']:
            inds,du,_=c['derivatives'][edge];up[np.ix_(inds,inds)]=du
            cellsites=next(s for s in source.seed.tiles(native) if int(edge//3) in s)
            edges=[3*cellsites[0],3*cellsites[1]+1,3*cellsites[2],3*cellsites[0]+1]
            local=cell_second(theta[edges],gam)[edges.index(edge)];upp[np.ix_(inds,inds)]=local
        pp=upp@p0@U0.conj().T+2*up@p0@up.conj().T+U0@p0@upp.conj().T
        dp=up@p0@U0.conj().T+U0@p0@up.conj().T
        hp=np.where(mask,0,source.path_matrix(native,roles[0],theta,edge=edge))
        hpp=np.where(mask,0,path_second(native,roles[0],theta,edge))
        sp=hp@P-P@hp+hic@dp-dp@hic
        spp=hpp@P-P@hpp+2*(hp@dp-dp@hp)+hic@pp-pp@hic
        poly,udp,udd=exponential_derivatives(A,-tau*sp,-tau*spp)
        check=expm_frechet(A,-tau*sp,compute_expm=False)
        frechet.append(float(np.max(abs(udp-check))));assert np.max(abs(poly-U))<1e-12
        bp=udp@U0+U@up;bpp=udd@U0+2*udp@up+U@upp
        R=-1j*B.conj().T@bp;T=-1j*(bp.conj().T@bp+B.conj().T@bpp)
        assert np.max(abs(R-R.conj().T))<1e-11 and np.max(abs(T-T.conj().T))<1e-11
        K=Y0.conj().T@R@W0;roo=W0.conj().T@R@W0;ree=Y0.conj().T@R@Y0
        mean=COPIES*np.trace(roo);means.append(abs(mean));Ks.append(K)
        zero+=len(half_trans)*G*(mean**2+COPIES*np.vdot(K,K).real-1j*COPIES*np.trace(W0.conj().T@T@W0))
        delta=G*(2*mean*K+ree@K-K@roo-1j*Y0.conj().T@T@W0)
        for tr in half_trans:electric+=move(delta,tr)
        e=float(np.linalg.norm(-tau*sp,2));f=float(np.linalg.norm(-tau*spp,2))
        tail.append(dict(exponential=math.exp(a)*a**(ORDER+1)/math.factorial(ORDER+1),first=math.exp(a)*e*a**ORDER/math.factorial(ORDER),second=math.exp(a)*(f*a**ORDER/math.factorial(ORDER)+e*e*a**(ORDER-1)/math.factorial(ORDER-1))))
    one=sum(int(r['load'])*np.vdot(m+electric,m+electric).real for r,m in zip(roles,Ms))
    pair=0j;wrong_independent=0j
    for k in Ks:
        for l in Ks:
            for tr in half_trans:
                Z=k.conj().T@move(l,tr);trace=np.trace(Z);exchange=np.sum(Z*Z.T)
                pair+=COPIES**2*trace**2-COPIES*exchange
                wrong_independent+=COPIES*(trace**2-exchange)
    pair*=2*G*G*len(half_trans);wrong_independent*=2*G*G*len(half_trans)
    assert abs(pair.imag)<1e-7*max(1,abs(pair.real)) and pair.real>0
    average=sum(int(r['load'])*m for r,m in zip(roles,Ms))/COPIES
    wrong_average=COPIES*np.vdot(average+electric,average+electric).real
    assert one-wrong_average>0
    assert pair.real-wrong_independent.real>1
    # The computed scalar varies with the field; this subtraction is only a
    # pointwise action component, never the integrated quantum variance.
    return dict(tau=tau,field='all192 raw edge phases zero, AP retained',energy_pointwise_real=float(zero.real),energy_pointwise_imag=float(zero.imag),onePH_squared=float(one),twoPH_squared=float(pair.real),complete_firstH_norm_squared=float(abs(zero)**2+one+pair.real),pointwise_non_scalar_action_squared=float(one+pair.real),wrong_independent_twoPH_squared=float(wrong_independent.real),wrong_role_averaged_onePH_squared=float(wrong_average),mean_Berry_float_max=float(max(means)),frechet_first_max_defect=max(frechet),Taylor_order=ORDER,Taylor_analytic_tail_max={key:max(t[key] for t in tail) for key in tail[0]},roundoff_interval_enclosed=False,physical_source_variance_computed=False)

def run(output:Path) -> dict:
    started=time.monotonic()
    names=['induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json','spatial_intercell_typed_bound_20261010_receipt.json']
    inputs={n:(ROOT/n).read_bytes() for n in names};saved=json.loads(inputs[names[0]])['input_descriptor']['exact_NEW_allq_BO_mean']
    centres=[Fraction(x) for x in saved['centre_exact_rational']]
    data=source.seed.field.load_mean(g_override=[float(centres[0]),float(centres[1]),float(centres[1]),float(centres[1])],mean_descriptor=saved)
    captured={**source.seed.SOURCE_BYTES,**source.seed.field.CAPTURED,**inputs,Path(__file__).name:Path(__file__).read_bytes(),'spatial_intercell_source.py':(ROOT/'spatial_intercell_source.py').read_bytes()}
    pre=archive(output.with_name(output.stem+'_preexecution.zip'),captured)
    tau=float(Fraction(json.loads(inputs[names[1]])['uniform_bound']['tau']['exact']))
    with threadpool_limits(limits=1):
        native=source.seed.field.full.supplier.quantum.build_native(data['mean'],N)
        cases=[action_case(native,data,t) for t in [0.,tau]]
    executed=archive(output.with_name(output.stem+'_executed.zip'),captured)
    result=dict(kind='COMPLETE_WHOLE_GAUSSIAN_N4_POINTWISE_FIRST_H_ACTION_NOT_INTEGRATED_VARIANCE',cases=cases,preexecution=pre,executed=executed,source_inputs_unchanged=all((ROOT/n).read_bytes()==b for n,b in captured.items()),elapsed_seconds=time.monotonic()-started,scope='Frozen CliffordPhi/identitySU4/beta0 body; actual active1+aux63/fullE. Exact Gaussian012PH first-action formula, small exponential derivative series with analytic truncation tail and finite float evaluation. Not a truncation of a nonGaussian source, Haar integration, stationarity or photon pole.',unpaid=['physicalPhi/SU4/Gauss lift','Haar-integrated complete variance','actual stationary/minimizing ground','phase/IRatom/commoncone/originalSOURCE'],campaign_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(args.output);print(json.dumps({'output':str(args.output),'cases':result['cases'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
