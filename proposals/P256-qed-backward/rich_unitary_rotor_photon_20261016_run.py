"""Actual V64 correlated full-rotor thermal consumers and discriminating probes."""
import itertools
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.linalg import expm
import rich_unitary_rotor_photon_20261016_thermal as r
import rich_unitary_rotor_photon_20261016_bridge as bridge

ROOT=Path(__file__).resolve().parent
P='rich_unitary_rotor_photon_20261016_'


def small_probe():
    # Two original sites, same-charge spinful, one full bilateral rotor link.
    # Exact physical Gauss sector has total N=2 and E=n0-1; no electric cutoff.
    basis=[n for n in range(16) if n.bit_count()==2];index={n:i for i,n in enumerate(basis)}
    hop=np.zeros((6,6));ons=np.zeros(6);ee=np.zeros(6)
    for k,n in enumerate(basis):
        n0=((n>>0)&1)+((n>>1)&1);n1=2-n0
        ons[k]=2*((n0-1)**2+(n1-1)**2);ee[k]=(n0-1)**2/32
        for a,b in ((0,2),(2,0),(1,3),(3,1)):
            if n>>b&1 and not n>>a&1:
                mid=n^(1<<b);sign=(-1)**((n&((1<<b)-1)).bit_count()+(mid&((1<<a)-1)).bit_count())
                hop[index[mid|(1<<a)],k]-=sign
    beta=.8;exact=np.trace(expm(-beta*hop)@np.diag(np.exp(-beta*(ons+ee))))
    # Temporal-gauge Haar quadrature Q=96; all HS configurations summed.
    Q=96;angles=2*np.pi*np.arange(Q)/Q;grid=np.array(list(itertools.product(angles,repeat=2)))
    K=np.exp(r.heat(grid[:,0]-grid[:,1],beta)[0]);lam=np.arccosh(np.exp(beta*2))
    T=expm(np.array([[0,beta],[beta,0]]));total=0.;wrong=0.;minimum=1.
    for signs in itertools.product((-1,1),repeat=2):
        B=T*np.exp(lam*np.array(signs))[None,:]
        C=np.exp(-1j*grid)[:,:,None]*B[None,:,:]
        determinants=np.linalg.det(np.eye(2)[None,:,:]+C)
        scalar=np.exp(-beta*4-lam*sum(signs))/4
        weights=K*abs(determinants)**2*scalar
        total+=weights.mean();minimum=min(minimum,float(weights.min()))
        wrong+=float((K*determinants*determinants*scalar).mean().real)
    # A wrong PH sign is not abs-square pairing at nonzero holonomy.
    _,arcs,D=r.geometry(2);rng=np.random.default_rng(413);theta=rng.normal(size=(2,len(arcs)))
    alpha=rng.normal(size=8);s=rng.choice([-1,1],size=(2,8));data=r.path_data(theta,alpha,s,arcs,D,.5)
    l=1;v=3;C=data['B'][0]@np.diag(data['twist'])@data['B'][1]
    G=np.linalg.inv(np.eye(8)+C);delta=np.expm1(-2*data['lam']*s[l,v]);ratio=1+delta*(1-G[v,v])
    ss=s.copy();ss[l,v]*=-1;trial=r.path_data(theta,alpha,ss,arcs,D,.5)
    analytic=2*np.log(abs(ratio))+2*data['lam']*s[l,v]
    # Direct primitive Peierls derivatives, not compressed source response.
    coeff=np.ones(len(arcs));eps=1e-5;h=r.hopping(theta[0],arcs,8)
    hp=r.hopping(theta[0]+eps,arcs,8);hm=r.hopping(theta[0]-eps,arcs,8)
    j=r.current_matrix(theta[0],arcs,coeff,8);k=r.current_matrix(theta[0],arcs,coeff,8,True)
    result=dict(physical_basis_dimension=6,Q=Q,beta=beta,M=1,I=16,
                exact_projected_Trotter_trace=float(exact),positive_path_trace=float(total),
                relative_trace_error=float(abs(total-exact)/exact),wrong_nonconjugate_trace=float(wrong),
                wrong_phase_relative_error=float(abs(wrong-exact)/exact),minimum_positive_weight=minimum,
                HS_rank_one_logratio_error=float(abs(analytic-trial['logweight']+data['logweight'])),
                primitive_current_derivative_error=float(np.max(abs(j+(hp-hm)/(2*eps)))),
                primitive_contact_derivative_error=float(np.max(abs(k-(hp+hm-2*h)/eps**2))),
                heat_omitted_image_log_bound=float(-(5*np.pi)**2/(2*(beta/16))))
    assert result['relative_trace_error']<1e-8,result
    assert result['wrong_phase_relative_error']>.01,result
    assert result['HS_rank_one_logratio_error']<1e-8,result
    assert result['primitive_current_derivative_error']<1e-8,result
    return result


def summary(arr)->dict[str,Any]:
    out:dict[str,Any]={}
    for key in ('EE','JJ','EJ','E','J','contact','plaquette','doublon','rho'):
        x=arr[key];blocks=np.array_split(x,8);means=np.array([b.mean(axis=0) for b in blocks if len(b)])
        mu=x.mean(axis=0);err=means.std(axis=0,ddof=1)/np.sqrt(len(means))
        out[key]=dict(real=mu.real.tolist(),imag=mu.imag.tolist(),block_standard_error=err.tolist())
    out['max_propagator_condition']=float(arr['max_propagator_condition'].max())
    # Covariance cross response is not divided by a putative pole or fitted c.
    out['lag_contact_identity_scope']='Boundary-inserted finite-Trotter imaginary-time operators; no claimed exact derivative Ward identity at finite dt.'
    out['bare_same_H_defeat']='At t=1 the primitive commutator [H,E_e] is nonzero J_e; contact is measured on this coupled state. Electric-only conserved-E evolution drops actual Hhop and is not this Hamiltonian.'
    return out


def main():
    probe=small_probe();(ROOT/(P+'small_result.json')).write_text(json.dumps(probe,indent=2)+'\n')
    print(json.dumps(dict(small_probe=probe)),flush=True)
    cases=[(.5,4,41101),(1.,4,41102),(1.,8,41103),(.5,4,41104),(1.,8,41105)]
    result:dict[str,Any]=dict(small=probe,cases=[],classification='Actual positive Gauss-projected full-rotor finite-beta finite-Trotter Monte Carlo; finite-chain exploratory estimates, not certified equilibration or poles')
    for beta,M,seed in cases:
        arrays,meta=bridge.sample(beta,M,seed,steps=4800,burn=1600,stride=16)
        payload:dict[str,Any]=arrays
        name=f'b{beta:g}_M{M}_s{seed}';np.savez_compressed(ROOT/(P+name+'_raw.npz'),**payload)
        entry:dict[str,Any]=dict(name=name,metadata=meta,observables=summary(arrays))
        (ROOT/(P+name+'_result.json')).write_text(json.dumps(entry,indent=2)+'\n');result['cases'].append(entry)
        (ROOT/(P+'result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(case=name,accepts=meta['accepts'],samples=meta['samples'],max_condition=entry['observables']['max_propagator_condition'])),flush=True)
    print('actual V64 full-rotor thermal consumers completed',flush=True)


if __name__=='__main__':main()
