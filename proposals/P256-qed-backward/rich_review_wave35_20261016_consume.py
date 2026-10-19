"""Independent fresh-chain canonical-CAR contraction of a declared V64 amplitude input.
Imports ONLY kernel/physical amplitude for provenance/comparison, never producer samples.
"""
from pathlib import Path
import json
import time
import numpy as np
from typing import Any
from rich_review_wave35_20261016_geometry import geometry,sources,car
import rich_bulk_photon_20261016_state as declared
ROOT=Path(__file__).resolve().parent
P='rich_review_wave35_20261016_'
POS,ARCS,B=geometry(4)
NEXT=np.array([3*b+mu for _,b,mu in ARCS]);PREV=np.argsort(NEXT)


def parity(up,down):
    return (-1.)**np.sum(down[None,:]<up[:,None])


def occupations(up,down):
    occ=np.zeros((2,64),dtype=int);occ[0,up]=1;occ[1,down]=1
    return occ


def fieldlog(up,down,E,params):
    return -params['g']*len(np.intersect1d(up,down))-params['a']*(E@E)-params['b']*np.sum((E-E[NEXT])**2)


def amp(up,down,E,F,params):
    return parity(up,down)*np.linalg.det(F[np.ix_(up,down)])*np.exp(fieldlog(up,down,E,params))


def logguide(up,down,E,F,G,params,epsilon,floor):
    # Explicit constant floor retains every legal configuration, including common nodes.
    d=np.linalg.det(F[np.ix_(up,down)]);dg=np.linalg.det(G[np.ix_(up,down)])
    return 2*fieldlog(up,down,E,params)+np.log(d*d+epsilon**2*dg*dg+floor**2)


def loopset():
    loops=[]
    for a in range(64):
        for mu in range(3):
            for nu in range(mu+1,3):
                z=np.zeros(192,dtype=int);e=3*a+mu;f=3*a+nu
                z[e]=1;z[3*ARCS[e,1]+nu]=1;z[3*ARCS[f,1]+mu]=-1;z[f]=-1;loops.append(z)
    for mu in range(3):
        for a in range(64):
            if POS[a,mu]:continue
            z=np.zeros(192,dtype=int);v=a
            for _ in range(4):e=3*v+mu;z[e]=1;v=ARCS[e,1]
            loops.append(z)
    return np.array(loops)


def sample(seed,n,burn,thin,F,G,params,epsilon,floor):
    rng=np.random.default_rng(seed);up=np.nonzero(POS.sum(1)%2==0)[0];down=np.setdiff1d(np.arange(64),up);E=np.zeros(192,dtype=int)
    loops=loopset();logw=logguide(up,down,E,F,G,params,epsilon,floor)
    us=[];ds=[];fields=[];accepted=np.zeros(3,int);proposed=np.zeros(3,int)
    for step in range(burn+n*thin):
        u=up.copy();d=down.copy();field=E.copy();r=rng.random()
        if r<.55:
            kind=0;e=rng.integers(192);direction=rng.choice((-1,1));spin=rng.integers(2)
            a,b,_=ARCS[e];a,b=(a,b) if direction==1 else (b,a);occ=u if spin==0 else d
            if a in occ and b not in occ and abs(field[e]+direction)<=1:
                occ[np.nonzero(occ==a)[0][0]]=b;occ.sort();field[e]+=direction
        elif r<.9:
            kind=1;candidate=field+rng.choice((-1,1))*loops[rng.integers(len(loops))]
            if np.max(abs(candidate))<=1:field=candidate
        else:
            kind=2;a=rng.choice(u);b=rng.choice(d)
            if a not in d and b not in u:u[u==a]=b;d[d==b]=a;u.sort();d.sort()
        proposed[kind]+=1
        lw=logguide(u,d,field,F,G,params,epsilon,floor)
        if np.log(rng.random())<lw-logw:
            if np.any(u!=up) or np.any(d!=down) or np.any(field!=E):accepted[kind]+=1
            up,down,E,logw=u,d,field,lw
        if step>=burn and (step-burn)%thin==0:us.append(up.copy());ds.append(down.copy());fields.append(E.copy())
    return np.array(us),np.array(ds),np.array(fields),accepted,proposed


def contract(up,down,E,F,params,coeff):
    """Canonical sorted pair det + actual interleaved CAR. Direct determinants at nodes."""
    occ=occupations(up,down);bits=sum(1<<(2*int(v)+s) for s in range(2) for v in np.nonzero(occ[s])[0])
    M=F[np.ix_(up,down)];det=np.linalg.det(M);sign=parity(up,down)
    env=np.exp(fieldlog(up,down,E,params));psi=sign*det*env
    stable=np.linalg.cond(M)<1e10;inv=np.linalg.inv(M) if stable else None
    rho=occ.sum(0)-1;diag=2*(rho@rho)+(E@E)/32
    Hpsi=diag*psi;J=np.zeros((len(coeff),2),complex);contact=np.zeros((len(coeff),2,2),complex)
    arcJ=np.zeros(192,complex);rho_comm=np.zeros(64,complex);HF_direct=np.array([c@E*Hpsi/16 for c in coeff],complex)
    neighbors=0
    for e,(aa,bb,_) in enumerate(ARCS):
        for direction in (-1,1):
            if abs(E[e]+direction)>1:continue
            a,b=(aa,bb) if direction==1 else (bb,aa)
            for spin in (0,1):
                target,h=car(bits,2*a+spin,2*b+spin)
                if not h:continue
                old=up if spin==0 else down;row=int(np.searchsorted(old,a));new=old.copy();new[row]=b;new.sort()
                newidx=int(np.searchsorted(new,b));reorder=(-1.)**abs(row-newidx)
                if stable:
                    assert inv is not None
                    ratio=(F[b,down]@inv[:,row]) if spin==0 else (inv[row,:]@F[up,b])
                    ndet=det*ratio*reorder
                else:ndet=np.linalg.det(F[np.ix_(new,down)] if spin==0 else F[np.ix_(up,new)])
                nu,nd=(new,down) if spin==0 else (up,new)
                deltaD=int(occ[1-spin,b])-int(occ[1-spin,a])
                df=-params['g']*deltaD-params['a']*(2*direction*E[e]+1)-params['b']*(2*direction*(2*E[e]-E[NEXT[e]]-E[PREV[e]])+2)
                other=parity(nu,nd)*ndet*env*np.exp(df);hv=h*other
                Hpsi+=hv;neighbors+=1
                arcJ[e]+=-1j*direction*hv
                rho_comm[a]+=-1j*hv;rho_comm[b]+=1j*hv
                for k,c in enumerate(coeff):
                    J[k]+=-1j*direction*c[:,e]*hv
                    contact[k]+=-np.outer(c[:,e].conj(),c[:,e])*hv*psi
                    HF_direct[k]+=(c@E+direction*c[:,e])*hv/16
    Fv=np.array([c@E*psi/16 for c in coeff])
    HF=np.array([c@E*Hpsi/16 for c in coeff])+1j*J/16
    return dict(psi=psi,Hpsi=Hpsi,F=Fv,J=J,contact=contact,HF=HF,HF_direct=HF_direct,continuity=np.max(abs(B@arcJ-rho_comm)),wrong_sign_continuity=np.max(abs(-B@arcJ-rho_comm)),gauss=np.max(abs(B@E+rho)),neighbors=neighbors,stable=stable)


def statistics(raw,normalizer):
    n=len(raw);z=np.sum(normalizer);mean=np.sum(raw,axis=0)/z
    blocks=np.array_split(np.arange(n),min(16,n//8));est=np.array([np.sum(raw[b],axis=0)/np.sum(normalizer[b]) for b in blocks if np.sum(normalizer[b])>0])
    se=np.std(est,axis=0,ddof=1)/np.sqrt(len(est))
    return mean,se


def run_chain(seed,config,F,G,params):
    epsilon=config['epsilon'];floor=config['floor'];n=config['samples'];qs=config['q'];coeff=np.array([sources(4,q)[0] for q in qs])
    up,down,fields,accept,propose=sample(seed,n,config['burn'],config['thin'],F,G,params,epsilon,floor)
    scalar=[];vec=[];hvec=[];contacts=[];checks=[];amp_checks=[];normalizer=[];psiv=[];hpsiv=[]
    for i,(u,d,E) in enumerate(zip(up,down,fields)):
        out=contract(u,d,E,F,params,coeff);w=np.exp(logguide(u,d,E,F,G,params,epsilon,floor));psi=out['psi'];h=out['Hpsi']
        normalizer.append(psi*psi/w);scalar.append([psi*h/w,h*h/w,len(np.intersect1d(u,d))*psi*psi/w,(E@E)*psi*psi/w])
        vv=np.concatenate((out['F'],out['J']),axis=1)
        vec.append(vv/np.sqrt(w));hvec.append(out['HF']/np.sqrt(w));contacts.append(out['contact']/w)
        psiv.append(psi/np.sqrt(w));hpsiv.append(h/np.sqrt(w))
        checks.append([out['gauss'],out['continuity'],out['wrong_sign_continuity'],np.max(abs(out['HF']-out['HF_direct'])),int(psi==0),out['neighbors'],int(out['stable'])])
        if i<12:
            occ=occupations(u,d);decl=declared.physical_amplitude(occ,E,F,**params)
            badE=E.copy();badE[0]+=1 if E[0]<1 else -1
            bad=declared.physical_amplitude(occ,badE,F,**params)
            amp_checks.append([psi,decl,bad])
    normalizer=np.array(normalizer);scalar=np.array(scalar);vec=np.array(vec);hvec=np.array(hvec);contacts=np.array(contacts);checks=np.array(checks);amp_checks=np.array(amp_checks)
    mean,se=statistics(scalar,normalizer);norm=normalizer.sum();psiv=np.array(psiv);hpsiv=np.array(hpsiv)
    source_means=np.einsum('n,nqi->qi',psiv.conj(),vec)/norm
    vec-=psiv[:,None,None]*source_means[None,:,:]
    hvec-=hpsiv[:,None,None]*source_means[None,:,:2]
    grams=np.einsum('nqi,nqj->qij',vec.conj(),vec)/norm
    moments=np.einsum('nqi,nqj->qij',vec[:,:,:2].conj(),hvec)/norm
    result: dict[str,Any]
    result=dict(seed=seed,samples=n,energy=float(mean[0]),energy_block_SE=float(se[0]),energy_per_cell=float(mean[0]/64),full_H2=float(mean[1]),full_H_variance=float(mean[1]-mean[0]**2),double_occupancy=float(mean[2]),field_E2=float(mean[3]),importance_ESS=float(norm**2/(normalizer@normalizer)),accepted=accept.tolist(),proposed=propose.tolist(),gauss_max=float(np.max(checks[:,0])),continuity_max=float(np.max(checks[:,1])),wrong_sign_continuity_max=float(np.max(checks[:,2])),HF_commutator_max=float(np.max(checks[:,3])),guide_nodes=int(np.sum(checks[:,4])),node_H2=float(np.sum(scalar[checks[:,4]>0,1])/norm),unstable_determinants=int(np.sum(checks[:,6]==0)),amplitude_relative_error=float(np.max(abs(amp_checks[:,0]-amp_checks[:,1])/(abs(amp_checks[:,0])+1e-100))),invalid_Gauss_amplitude_max=float(np.max(abs(amp_checks[:,2]))),q=[])
    for k,q in enumerate(qs):
        gram=grams[k];fg=gram[:2,:2];mix=gram[:2,2:];jg=gram[2:,2:];leak=jg-mix.conj().T@np.linalg.pinv(fg,rcond=1e-10)@mix
        centered=moments[k]-mean[0]*fg;herm=(centered+centered.conj().T)/2
        vals,U=np.linalg.eigh(fg);W=U[:,vals>1e-10]/np.sqrt(vals[vals>1e-10]);ritz=np.linalg.eigvalsh(W.conj().T@herm@W)
        result['q'].append(dict(q=q,field_norm2=fg.diagonal().real.tolist(),current_norm2=jg.diagonal().real.tolist(),current_outside_field_norm2=leak.diagonal().real.tolist(),current_outside_field_fraction=(leak.diagonal().real/jg.diagonal().real).tolist(),field_source_Rayleigh_ritz=ritz.tolist(),moment_antihermitian_norm=float(np.linalg.norm(centered-centered.conj().T)),contact=(contacts[:,k].sum(0)/norm).diagonal().real.tolist()))
    np.savez_compressed(ROOT/(P+f'chain{seed}_raw.npz'),up=up,down=down,E=fields,normalizer=normalizer,scalar=scalar,vectors=vec,HF=hvec,contacts=contacts,checks=checks,amplitude_checks=amp_checks,grams=grams,field_moments=moments,source_means=source_means,psi=psiv,Hpsi=hpsiv)
    return result


def main():
    config=json.loads((ROOT/(P+'input.json')).read_text());params=config['params']
    F,_,meta=declared.kernel(**config['kernel']);rng=np.random.default_rng(config['guide_seed']);G=F+.02*rng.normal(size=F.shape)
    np.savez_compressed(ROOT/(P+'declared_input_arrays.npz'),F=F,G=G)
    result: dict[str,Any]
    started=time.time();result=dict(kind='INDEPENDENT_NEW_V64_FINITE_STATE_NOT_SOLVED_VACUUM_OR_FULL_H_DYNAMICS',input=config,kernel_metadata=meta,chains=[])
    for seed in config['seeds']:
        r=run_chain(seed,config,F,G,params);result['chains'].append(r)
        (ROOT/(P+'findings.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(r),flush=True)
    result['seconds']=time.time()-started;(ROOT/(P+'findings.json')).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
