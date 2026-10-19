"""Fresh same-H V64 finite-transfer ensembles and colder stabilized method fork."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import rich_unitary_rotor_photon_20261016_thermal as rotor
from rich_unitary_rotor_photon_20261016_bridge import free_bridge
import rich_collective_photon_response_20261016_model as model
from rich_collective_photon_response_20261016_probe import physical_probe

ROOT=Path(__file__).resolve().parent
P='rich_collective_photon_response_20261016_'


def summarize(x):
    x=np.asarray(x);blocks=np.array([b.mean(axis=0) for b in np.array_split(x,8) if len(b)])
    return dict(real=x.mean(axis=0).real.tolist(),imag=x.mean(axis=0).imag.tolist(),
                block_SEM_real=(blocks.real.std(axis=0,ddof=1)/np.sqrt(len(blocks))).tolist(),
                block_SEM_imag=(blocks.imag.std(axis=0,ddof=1)/np.sqrt(len(blocks))).tolist(),
                first_half_real=x[:len(x)//2].mean(axis=0).real.tolist(),
                second_half_real=x[len(x)//2:].mean(axis=0).real.tolist())


def sample(beta,M,seed,steps=3600,burn=1200,stride=40,I=16.,stabilized=False):
    start=time.monotonic();rng=np.random.default_rng(seed);pos,arcs,D=rotor.geometry(4);V=len(pos);ne=len(arcs)
    vals,vecs=np.linalg.eigh(D@D.T);keep=vals>1e-8
    alpha=vecs[:,keep]@(rng.normal(size=keep.sum())*np.sqrt(beta/I/vals[keep]))+rng.uniform(-np.pi,np.pi)
    theta0=rng.uniform(-np.pi,np.pi,ne);g=D.T@alpha
    theta=np.array([free_bridge(theta0[e],g[e],M,beta,I,rng) for e in range(ne)]).T
    s=rng.choice(np.array([-1,1]),size=(M,V))
    def pd(th,al,hs):return model.path_data(th,al,hs,arcs,D,beta,I,stabilized=stabilized)
    data=pd(theta,alpha,s);attempts=np.zeros(9,int);accepts=attempts.copy();samples=[];paths=[];history=[];HSaccepted=0
    modes,_=model.spatial_modes(pos,arcs)
    for step in range(steps):
        kind=step%9;attempts[kind]+=1;tt=theta.copy();aa=alpha.copy();ss=s.copy();hastings=0.
        if kind==0:
            e=rng.integers(ne);tt[:,e]+=rng.normal(0,.85)
        elif kind==1:
            e=rng.integers(ne);tt[:,e]=free_bridge(theta[0,e],(D.T@alpha)[e],M,beta,I,rng)
            old=rotor.heat(data['delta'][:,e],beta/M,I)[0].sum()
            delta=np.roll(tt[:,e],-1)-tt[:,e];delta[-1]+=(D.T@alpha)[e]
            hastings=old-rotor.heat(delta,beta/M,I)[0].sum()
        elif kind==2:
            v=rng.integers(V);aa[v]+=rng.normal(0,.65*np.sqrt(beta/I))
            tt+=np.arange(M)[:,None]/M*(D[v]*(aa[v]-alpha[v]))[None,:]
        elif kind==3:aa+=rng.uniform(-np.pi,np.pi)
        elif kind in (4,8):
            l=int(rng.integers(M))
            if stabilized:ss,count=model.stable_hs_slice(s,data,l,rng)
            else:ss,count=rotor.hs_slice(s,data,l,rng)
            HSaccepted+=count
        elif kind==5:
            f=modes[int(rng.integers(len(modes)))];n=int(rng.integers(1,M))
            profile=np.real(np.exp(2j*np.pi*n*np.arange(M)/M+1j*rng.uniform(0,2*np.pi))[:,None]*f[None,:])
            tt+=rng.normal(0,.7*np.sqrt(beta/I))*profile
        elif kind==6:
            # Independent ALL-link compact free bridge. Hastings removes its entire
            # free kinetic action; only the full interacting determinant remains.
            tt=np.array([free_bridge(theta[0,e],(D.T@alpha)[e],M,beta,I,rng) for e in range(ne)]).T
            delta=np.roll(tt,-1,axis=0)-tt;delta[-1]+=D.T@alpha
            hastings=rotor.heat(data['delta'],beta/M,I)[0].sum()-rotor.heat(delta,beta/M,I)[0].sum()
        else:tt+=rng.normal(0,.08,ne)[None,:]
        trial=pd(tt,aa,ss)
        if kind in (4,8) or np.log(rng.random())<trial['logweight']-data['logweight']+hastings:
            theta,alpha,s,data=tt,aa,ss,trial;accepts[kind]+=1
        if step%20==0:history.append((step,data['logweight'],np.mean(s),np.mean(np.cos(theta[0]))))
        if step>=burn and (step-burn)%stride==0:
            compare=stabilized and len(samples) in (0,(steps-burn-1)//stride)
            obs=model.interval_response(theta,alpha,s,data,pos,arcs,D,beta,I=I,stabilized=stabilized,compare_step=compare)
            obs['step_comparison_exercised']=np.array(compare)
            samples.append(obs);paths.append((theta.copy(),alpha.copy(),s.copy(),data['logweight']))
    arrays={k:np.array([x[k] for x in samples]) for k in samples[0]}
    arrays.update(theta=np.array([p[0] for p in paths]),alpha=np.array([p[1] for p in paths]),HS=np.array([p[2] for p in paths]),
                  logweight=np.array([p[3] for p in paths]),chain_history=np.array(history))
    meta=dict(beta=beta,M=M,dt=beta/M,seed=seed,steps=steps,burn=burn,stride=stride,samples=len(samples),I=I,U=4,t=1,
              V=V,edges=ne,stabilized=stabilized,attempts=attempts.tolist(),accepts=accepts.tolist(),
              HS_single_site_accepts=HSaccepted,elapsed_seconds=time.monotonic()-start,
              moves=['static link','compact link bridge','local holonomy/linear path','global holonomy','HS slice',
                     'collective transverse temporal mode','ALL-link bridge','all-static-link Gaussian','second HS slice'],
              ensemble='same original two same-charge species; exact down PH, local compact Gauss holonomies; original totalN64 spin mixture',
              cold_method='pivoted QDT product and stabilized cyclic HS Green; Frechet local jets with centered stable determinant derivatives h=.008 and selected step-halving' if stabilized else 'analytic full determinant interval-source jets',
              mixing_scope='Fresh chains, all-link free-bridge and collective moves; finite burn/8-block SEM/half-chain comparisons are diagnostics, not equilibration certification',
              winding_endpoint_image_log_bound=float(-(7*np.pi)**2/(2*beta/I)),
              heat_image_log_bound=float(-(5*np.pi)**2/(2*(beta/M/I))))
    return arrays,meta


def main():
    stage=sys.argv[1]
    if stage=='probe':
        result,raw=physical_probe();np.savez_compressed(ROOT/(P+'probe_v2_raw.npz'),**raw)
        (ROOT/(P+'probe_v2_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return
    if stage=='hot':cases=[(.5,4,46101),(1.,4,46102),(1.,8,46103),(.5,4,46104),(1.,8,46105)]
    elif stage=='cold':cases=[(4.,16,46201),(4.,16,46202),(8.,32,46203)]
    else:raise ValueError(stage)
    result=dict(stage=stage,cases=[],scope='Finite scientific chains, not photon pole/equilibration/continuum certification')
    for beta,M,seed in cases:
        arrays,meta=sample(beta,M,seed,steps=3600 if beta<=4 else 2700,burn=1200 if beta<=4 else 900,
                           stride=40 if beta<=1 else 60,stabilized=beta>1)
        name=f'b{beta:g}_M{M}_s{seed}';np.savez_compressed(ROOT/(P+name+'_raw.npz'),**arrays)
        keys=('source_hessian','logdet_hessian','source_one','interval_contact','interval_current_matsubara','EE_matsubara','EE_tau',
              'qminus_EE_matsubara','qminus_EE_tau','direct_EE_matsubara','direct_EE_tau','derivative_step_difference')
        entry=dict(name=name,metadata=meta,observables={k:summarize(arrays[k]) for k in keys},
                   max_source_matrix_condition=float(arrays['max_source_matrix_condition'].max()),
                   max_transverse_divergence=float(arrays['transverse_divergence'].max()),
                   maximum_temporal_winding=float(abs(arrays['compact_temporal_winding']).max()),
                   selected_step_halving_exercised=int(arrays['step_comparison_exercised'].sum()))
        (ROOT/(P+name+'_result.json')).write_text(json.dumps(entry,indent=2)+'\n');result['cases'].append(entry)
        (ROOT/(P+stage+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(name=name,samples=meta['samples'],elapsed=meta['elapsed_seconds'],accepts=meta['accepts'],
                             EE_n1=arrays['EE_matsubara'][:,:,1].mean(axis=0).tolist(),
                             independent_EE_n0=arrays['EE_matsubara'][:,:,0].mean(axis=0).tolist())),flush=True)
    print('Fresh same-H finite-transfer physical consumers finished.',flush=True)


if __name__=='__main__':main()
