"""Material sampler correction: compact free-rotor entire-link heatbath bridge.
The target remains the same full thermal law and exact determinant/HS/Gauss weight.
"""
import numpy as np
import rich_unitary_rotor_photon_20261016_thermal as r


def free_bridge(theta0,holonomy,M,beta,I,rng):
    """Wrapped Brownian bridge; endpoint windings included, no electric cap.
    Winding-image estimator -4..4 omits >=7pi endpoint distances; beta<=1,I16.
    This numerical image tail is separate from the full law's integer E spectrum.
    """
    g=(holonomy+np.pi)%(2*np.pi)-np.pi
    ws=np.arange(-4,5);logs=-(g+2*np.pi*ws)**2/(2*beta/I)
    p=np.exp(logs-logs.max());p/=p.sum();w=rng.choice(ws,p=p)
    increments=rng.normal(0,np.sqrt(beta/M/I),M)
    increments-=increments.mean();increments+=(g+2*np.pi*w)/M
    path=np.r_[0,np.cumsum(increments)[:-1]]
    return theta0+(path+np.pi)%(2*np.pi)-np.pi


def sample(beta,M,seed,steps=4800,burn=1600,stride=16,I=16.,L=4):
    rng=np.random.default_rng(seed);pos,arcs,D=r.geometry(L);V=len(pos);ne=len(arcs)
    # Initialize all temporal modes, rather than waiting for one-coordinate mixing.
    lap=D@D.T;vals,vecs=np.linalg.eigh(lap);keep=vals>1e-8
    alpha=vecs[:,keep]@(rng.normal(size=keep.sum())*np.sqrt(beta/I/vals[keep]))+rng.uniform(-np.pi,np.pi)
    g=D.T@alpha;theta0=rng.uniform(-np.pi,np.pi,ne)
    theta=np.array([free_bridge(theta0[e],g[e],M,beta,I,rng) for e in range(ne)]).T
    s=rng.choice(np.array([-1,1]),size=(M,V));data=r.path_data(theta,alpha,s,arcs,D,beta,I)
    attempts=np.zeros(6,int);accepts=attempts.copy();samples=[];weights=[];raw=[];HSaccepted=0
    for step in range(steps):
        kind=step%6;attempts[kind]+=1;tt=theta.copy();aa=alpha.copy();ss=s.copy();hastings=0.
        if kind==0:
            e=rng.integers(ne);tt[:,e]+=rng.normal(0,.85)
        elif kind==1:
            e=rng.integers(ne);tt[:,e]=free_bridge(theta[0,e],(D.T@alpha)[e],M,beta,I,rng)
            old=r.heat(data['delta'][:,e],beta/M,I)[0].sum()
            delta=np.roll(tt[:,e],-1)-tt[:,e];delta[-1]+=(D.T@alpha)[e]
            new=r.heat(delta,beta/M,I)[0].sum();hastings=old-new
        elif kind==2:
            v=rng.integers(V);aa[v]+=rng.normal(0,.65*np.sqrt(beta/I))
            tt+=np.arange(M)[:,None]/M*(D[v]*(aa[v]-alpha[v]))[None,:]
        elif kind==3:aa+=rng.uniform(-np.pi,np.pi)
        elif kind==4:
            ss,count=r.hs_slice(s,data,int(rng.integers(M)),rng);HSaccepted+=count
        else:
            # Refresh another complete temporal link; do not leave modes pinned.
            e=rng.integers(ne);tt[:,e]=free_bridge(theta[0,e],(D.T@alpha)[e],M,beta,I,rng)
            old=r.heat(data['delta'][:,e],beta/M,I)[0].sum()
            delta=np.roll(tt[:,e],-1)-tt[:,e];delta[-1]+=(D.T@alpha)[e]
            new=r.heat(delta,beta/M,I)[0].sum();hastings=old-new
        trial=r.path_data(tt,aa,ss,arcs,D,beta,I)
        if kind==4 or np.log(rng.random())<trial['logweight']-data['logweight']+hastings:
            theta,alpha,s,data=tt,aa,ss,trial;accepts[kind]+=1
        if step>=burn and (step-burn)%stride==0:
            samples.append(r.observe(theta,alpha,s,data,pos,arcs,D,beta))
            weights.append(data['logweight']);raw.append((theta.copy(),alpha.copy(),s.copy()))
    arrays={k:np.array([x[k] for x in samples]) for k in samples[0]}
    arrays.update(theta=np.array([x[0] for x in raw]),alpha=np.array([x[1] for x in raw]),HS=np.array([x[2] for x in raw]),logweight=np.array(weights))
    return arrays,dict(beta=beta,M=M,dt=beta/M,seed=seed,steps=steps,burn=burn,stride=stride,samples=len(samples),
                       attempts=attempts.tolist(),accepts=accepts.tolist(),HS_single_site_accepts=HSaccepted,
                       initialization='fresh Haar static links, free compact Gaussian temporal bridges, Gaussian local holonomy plus uniform global holonomy, random HS',
                       moves=['static link','full temporal link heatbath bridge','joint local holonomy/linear path','uniform global holonomy','exact HS slice','second full temporal bridge'],
                       bridge_Hastings='log product K(old)-log product K(new), exact compact conditional bridge normalization cancels',
                       bridge_winding_image_log_tail_bound=float(-(7*np.pi)**2/(2*beta/I)),
                       ensemble='Gauss-only original totalN64 spin mixture',I=I,U=4,t=1,V=V,arcs=ne)
