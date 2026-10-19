"""Changed-layer independent physical response consumer of Rotor41 free_bridge."""
import json
from pathlib import Path
import numpy as np
from rich_unitary_rotor_photon_20261016_bridge import free_bridge
from rich_unitary_rotor_photon_20261016_thermal import heat,path_data

ROOT=Path(__file__).resolve().parent
P='rich_review_wave43_20261016_'


def main():
    cfg=json.loads((ROOT/(P+'bridge_owner_input.json')).read_text());rng=np.random.default_rng(cfg['seed'])
    beta=cfg['beta'];I=cfg['I'];M=cfg['slices'];dt=beta/M
    paths=np.array([free_bridge(.37,0.,M,beta,I,rng) for _ in range(cfg['samples'])])
    delta=np.roll(paths,-1,axis=1)-paths
    _,score,second=heat(delta,dt,I)
    EE=np.array([-np.mean(score*np.roll(score,lag,axis=1),axis=1) for lag in range(M)]).T
    EE[:,0]+=np.mean(-second+score**2,axis=1)
    blocks=np.array([a.mean(axis=0) for a in np.array_split(EE,20)])
    mean=EE.mean(axis=0);sem=blocks.std(axis=0,ddof=1)/np.sqrt(len(blocks))
    m=np.arange(-80,81);w=np.exp(-beta*m*m/(2*I));truth=float(np.sum(w*m*m)/w.sum())
    V=4;arcs=np.array([[0,1,1],[1,2,2],[2,3,1],[3,0,2]]);D=np.zeros((V,4))
    for e,(a,b,_) in enumerate(arcs):D[a,e]=1;D[b,e]=-1
    alpha=rng.normal(0,.1,V);theta=np.array([free_bridge(.2,(D.T@alpha)[e],M,beta,I,rng) for e in range(4)]).T
    s=rng.choice([-1,1],(M,V));before=path_data(theta,alpha,s,arcs,D,beta,I)
    changed=theta.copy();changed[:,1]=free_bridge(theta[0,1],(D.T@alpha)[1],M,beta,I,rng)
    after=path_data(changed,alpha,s,arcs,D,beta,I)
    old=heat(before['delta'][:,1],dt,I)[0].sum();new=heat(after['delta'][:,1],dt,I)[0].sum()
    corrected=after['logweight']-before['logweight']+old-new
    def determinant_log(data):return 2*np.linalg.slogdet(np.eye(V)+data['twist'][:,None]*data['P'])[1]
    determinant_only=determinant_log(after)-determinant_log(before)
    result=dict(kind='OWN_PHYSICAL_CONSUMER_OF_CORRECTED_BRIDGE',input=cfg,exact_free_rotor_EE=truth,
        measured_EE=mean.tolist(),block_standard_error=sem.tolist(),standardized_error=((mean-truth)/sem).tolist(),
        full_target_Hastings_logratio=float(corrected),independent_determinant_logratio=float(determinant_only),
        Hastings_cancellation_error=float(abs(corrected-determinant_only)),
        scope='actual full-link proposal helper; exact free quantum rotor unequal-imaginary-time response control and compact weighted interacting-path detailed-balance identity, NOT converged V64 MC')
    np.savez_compressed(ROOT/(P+'bridge_owner_raw.npz'),blocks=blocks,mean=mean,sem=sem)
    (ROOT/(P+'bridge_owner_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    assert max(abs((mean-truth)/sem))<5
    assert result['Hastings_cancellation_error']<1e-10


if __name__=='__main__':main()
