"""Fresh V64 Gauss/PH compact paths consumed by new46 finite-transfer vertices."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import rich_unitary_rotor_photon_20261016_thermal as r
from rich_unitary_rotor_photon_20261016_bridge import free_bridge
import rich_collective_photon_response_20261016_model as model

ROOT=Path(__file__).resolve().parent;P='rich_review_wave45_20261016_'


def vertices_probe():
    rng=np.random.default_rng(45171);pos,arcs,D=r.geometry(2);V=len(pos)
    theta=rng.normal(size=len(arcs));f=np.array([.21,-.13,.08]*V);dt=.23;eps=8e-5
    A,C,AC,_,_=model.interval_vertices(theta,arcs,V,dt,f)
    t=expm(-dt*r.hopping(theta,arcs,V));tp=expm(-dt*r.hopping(theta+eps*f,arcs,V));tm=expm(-dt*r.hopping(theta-eps*f,arcs,V))
    point=dt*r.current_matrix(theta,arcs,f,V)@t
    return dict(first_Frechet_error=float(np.linalg.norm((tp-tm)/(2*eps)-A)),
        second_contact_Frechet_error=float(np.linalg.norm((tp+tm-2*t)/eps**2-AC)),
        wrong_point_vertex_error=float(np.linalg.norm(A-point)),
        transverse_probe_geometry='L2 has opposite parallel primitive arcs; all physical edges retained')


def fresh(beta,M,seed,steps=1200,burn=400,stride=80):
    rng=np.random.default_rng(seed);pos,arcs,D=r.geometry(4);V=len(pos);ne=len(arcs);I=16.
    vals,vecs=np.linalg.eigh(D@D.T);keep=vals>1e-8
    alpha=vecs[:,keep]@(rng.normal(size=keep.sum())*np.sqrt(beta/I/vals[keep]))+rng.uniform(-np.pi,np.pi)
    theta=np.array([free_bridge(rng.uniform(-np.pi,np.pi),(D.T@alpha)[e],M,beta,I,rng) for e in range(ne)]).T
    s=rng.choice([-1,1],size=(M,V));data=r.path_data(theta,alpha,s,arcs,D,beta,I)
    attempts=np.zeros(5,int);accepts=attempts.copy();rows=[];hsaccepted=0
    for step in range(steps):
        kind=step%5;attempts[kind]+=1;tt=theta.copy();aa=alpha.copy();ss=s.copy();hastings=0.
        if kind==0:
            e=rng.integers(ne);tt[:,e]+=rng.normal(0,.9)
        elif kind==1:
            e=rng.integers(ne);tt[:,e]=free_bridge(theta[0,e],(D.T@alpha)[e],M,beta,I,rng)
            old=r.heat(data['delta'][:,e],beta/M,I)[0].sum()
            delta=np.roll(tt[:,e],-1)-tt[:,e];delta[-1]+=(D.T@alpha)[e]
            hastings=old-r.heat(delta,beta/M,I)[0].sum()
        elif kind==2:
            v=rng.integers(V);aa[v]+=rng.normal(0,.65*np.sqrt(beta/I))
            tt+=np.arange(M)[:,None]/M*(D[v]*(aa[v]-alpha[v]))[None,:]
        elif kind==3:aa+=rng.uniform(-np.pi,np.pi)
        else:ss,count=r.hs_slice(s,data,int(rng.integers(M)),rng);hsaccepted+=count
        trial=r.path_data(tt,aa,ss,arcs,D,beta,I)
        if kind==4 or np.log(rng.random())<trial['logweight']-data['logweight']+hastings:
            theta,alpha,s,data=tt,aa,ss,trial;accepts[kind]+=1
        if step>=burn and (step-burn)%stride==0:
            obs=model.interval_response(theta,alpha,s,data,pos,arcs,D,beta)
            boundary=r.observe(theta,alpha,s,data,pos,arcs,D,beta)
            rows.append(dict(**obs,theta=theta.copy(),alpha=alpha.copy(),HS=s.copy(),JJ=boundary['JJ'],contact=boundary['contact'],logweight=np.array(data['logweight'])))
    arrays={k:np.array([z[k] for z in rows]) for k in rows[0]}
    name=f'collective_b{beta:g}_M{M}_s{seed}';np.savez_compressed(ROOT/(P+name+'_raw.npz'),**arrays)
    def summary(key):
        x=arrays[key];blocks=np.array_split(x,5);means=np.array([z.mean(axis=0) for z in blocks]);mu=x.mean(axis=0)
        return dict(mean_real=mu.real.tolist(),mean_imag=mu.imag.tolist(),five_block_SE=(means.std(axis=0,ddof=1)/np.sqrt(5)).tolist())
    return dict(name=name,beta=beta,M=M,dt=beta/M,seed=seed,V=V,steps=steps,burn=burn,stride=stride,samples=len(rows),
        attempts=attempts.tolist(),accepts=accepts.tolist(),HS_site_accepts=hsaccepted,
        EE_matsubara=summary('EE_matsubara'),direct_EE_matsubara=summary('direct_EE_matsubara'),source_hessian=summary('source_hessian'),
        logdet_hessian=summary('logdet_hessian'),JJ=summary('JJ'),contact=summary('contact'),
        max_source_matrix_condition=float(arrays['max_source_matrix_condition'].max()),
        max_transverse_divergence=float(arrays['transverse_divergence'].max()),
        limits='Independent fresh short physical paths. Five-block errors are descriptive, not mixing proof. No ancestor solved state or response array used; beta2 is colder but not ground. Current full response is not irreducible stiffness.')


def free_control():
    # A single full compact rotor: frequency-nonzero EE vanishes and n0=M<E²>.
    m=np.arange(-40,41);beta=1.;I=16.;M=4;p=np.exp(-beta*m*m/(2*I));p/=p.sum()
    exact=M*np.sum(p*m*m)
    rng=np.random.default_rng(45202);N=1000;direct=[]
    for _ in range(N):
        theta=free_bridge(rng.uniform(-np.pi,np.pi),0.,M,beta,I,rng)
        delta=np.roll(theta,-1)-theta;_,score,second=r.heat(delta,beta/M,I)
        E=-1j*score;ct=np.array([np.mean(E*np.roll(E,l)) for l in range(M)]);ct[0]+=np.mean(-second+score**2)
        direct.append(np.fft.fft(ct).real)
    arr=np.array(direct);np.savez_compressed(ROOT/(P+'collective_free_control_raw.npz'),direct=arr,exact=np.r_[exact,np.zeros(M-1)])
    return dict(samples=N,exact_zero_mode=float(exact),direct_mean=arr.mean(axis=0).tolist(),standard_error=(arr.std(axis=0,ddof=1)/np.sqrt(N)).tolist(),nonzero_current_linked_exact=0.,scope='Free bridge only; cancellation-noise diagnosis, not interacting-state replacement')


def main():
    result=dict(kind='OWN_NEW46_PHYSICAL_TRANSFER_AND_FRESH_V64_REVIEW',vertex_probe=vertices_probe(),free_control=free_control(),cases=[])
    for beta,M,seed in [(1.,4,45101),(2.,8,45102)]:
        entry=fresh(beta,M,seed);result['cases'].append(entry)
        (ROOT/(P+'collective_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(case=entry['name'],samples=entry['samples'],accepts=entry['accepts'],maxcond=entry['max_source_matrix_condition'])),flush=True)
    print(json.dumps(dict(vertex_probe=result['vertex_probe'],free_control=result['free_control'])),flush=True)


if __name__=='__main__':main()
