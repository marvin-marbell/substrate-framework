"""Revision-invalidated estimator comparison, own Trotter state and PH/Gauss trace."""
import json
from pathlib import Path
import itertools
import numpy as np
from scipy.linalg import expm
import rich_unitary_rotor_photon_20261016_thermal as r
import rich_collective_photon_response_20261016_model as model
from rich_review_wave45_20261016_collective import fresh

ROOT=Path(__file__).resolve().parent;P='rich_review_wave45_20261016_'


def trace_probe():
    beta=.63;I=3.7;Q=64;basis=[n for n in range(16) if n.bit_count()==2];lookup={n:i for i,n in enumerate(basis)}
    hop=np.zeros((6,6));rho=np.array([((n>>0)&1)+((n>>1)&1)-1 for n in basis])
    for col,n in enumerate(basis):
        for a,b in [(0,2),(2,0),(1,3),(3,1)]:
            if (n>>b)&1 and not (n>>a)&1:
                mid=n^(1<<b);sign=(-1)**((n&((1<<b)-1)).bit_count()+(mid&((1<<a)-1)).bit_count())
                hop[lookup[mid^(1<<a)],col]-=sign
    exact=np.trace(expm(-beta*hop)@np.diag(np.exp(-beta*(4*rho*rho+rho*rho/(2*I)))))
    angles=2*np.pi*np.arange(Q)/Q;alpha=np.array(list(itertools.product(angles,repeat=2)))
    logK,_,_=r.heat(alpha[:,0]-alpha[:,1],beta,I);K=np.exp(logK)
    T=expm(beta*np.array([[0.,1.],[1.,0.]]));lam=np.arccosh(np.exp(beta*2));total=0.;wrong=0.;unprojected=0.
    for signs in itertools.product((-1,1),repeat=2):
        B=T*np.exp(lam*np.array(signs))[None,:];C=np.exp(-1j*alpha)[:,:,None]*B
        det=np.linalg.det(np.eye(2)[None,:,:]+C);scalar=np.exp(-beta*4-lam*sum(signs))/4
        total+=np.mean(K*abs(det)**2)*scalar;wrong+=np.mean(K*det*det).real*scalar
        unprojected+=abs(np.linalg.det(np.eye(2)+B))**2*scalar
    return dict(beta=beta,I=I,Q=Q,dimension=6,exact_physical_transfer_trace=float(exact),positive_PH_Gauss_trace=float(total),relative_error=float(abs(total-exact)/exact),
        wrong_nonconjugate_relative_error=float(abs(wrong-exact)/exact),wrong_unprojected_relative_error=float(abs(unprojected-exact)/exact),
        scope='Actual original same-charge spinful two-site physical Gauss sector vs compact temporal holonomy plus exact HS sum; not a new opposite-charge law')


def revision_probe():
    raw=np.load(ROOT/(P+'collective_b1_M4_s45101_raw.npz'));pos,arcs,D=r.geometry(4)
    theta,alpha,s=raw['theta'][0],raw['alpha'][0],raw['HS'][0]
    data=r.path_data(theta,alpha,s,arcs,D,1.);obs=model.interval_response(theta,alpha,s,data,pos,arcs,D,1.)
    return dict(source_hessian_difference=float(np.max(abs(obs['source_hessian']-raw['source_hessian'][0]))),EE_matsubara_difference=float(np.max(abs(obs['EE_matsubara']-raw['EE_matsubara'][0]))),
        scope='Only source estimator changed: stable HS addition unused; spectral cache and masked near-degenerate series evaluated on an own executed physical sample. No ancestor producer output.')


def main():
    result=dict(trace=trace_probe(),revision=revision_probe(),Trotter_case=fresh(1.,8,45103))
    (ROOT/(P+'collective_addendum_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(trace=result['trace'],revision=result['revision'],newcase=result['Trotter_case']['name'])),flush=True)


if __name__=='__main__':main()
