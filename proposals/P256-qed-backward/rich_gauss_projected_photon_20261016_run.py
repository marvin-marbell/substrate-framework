"""Fresh finite-state experiments; raw arrays retain reweighting and nodal support."""
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import eigh
import rich_gauss_projected_photon_20261016_state as s
from rich_periodic_photon_20261015_model import source_geometry
ROOT=Path(__file__).resolve().parent
P='rich_gauss_projected_photon_20261016_'


def save(name,value):
    (ROOT/(P+name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def estimate(raw,chains=4) -> dict[str,Any]:
    z=raw[:,0].mean();E=raw[:,1].mean()/z
    residual2=raw[:,2].mean()/z-E*E
    pieces=np.array_split(raw,chains*10)
    energies=np.array([x[:,1].mean()/x[:,0].mean() for x in pieces])
    variances=np.array([x[:,2].mean()/x[:,0].mean()-(x[:,1].mean()/x[:,0].mean())**2 for x in pieces])
    return dict(norm_reweight=z,energy=E,energy_block_SE=float(energies.std(ddof=1)/np.sqrt(len(energies))),
                full_H_residual_squared_estimator=residual2,full_H_residual_estimator=float(np.sqrt(max(0,residual2))),
                residual_squared_block_SE=float(variances.std(ddof=1)/np.sqrt(len(variances))),
                doublons=float(raw[:,3].mean()/z),electric_squared=float(raw[:,4].mean()/z),
                total_flux_mean=float(raw[:,5].mean()/z),total_flux_squared=float(raw[:,6].mean()/z),
                four_pair_weight=float(raw[:,9].mean()/z),guide_node_fraction=float(raw[:,8].mean()),
                chains=chains,blocks=len(pieces),samples=len(raw),
                error_limit='Block SE not a proved autocorrelation bound; compare chain means and exact small benchmark; full H2 includes nonzero Hpsi at psi=0 through positive guide.')


def small_benchmark():
    # Independent full tensor orbital matrices, two distinct opposite physical arcs.
    annih=[]
    for p in range(4):
        c=np.zeros((16,16))
        for bits in range(16):
            if (bits>>p)&1:c[bits^(1<<p),bits]=(-1)**((bits&((1<<p)-1)).bit_count())
        annih.append(c)
    raiseE=np.diag(np.ones(2),-1);eye=np.eye(3)
    full=np.zeros((144,144))
    fields=[(a,b) for a in (-1,0,1) for b in (-1,0,1)]
    for bits in range(16):
        rho=np.array([(bits&1)+((bits>>1)&1)-1,((bits>>2)&1)+((bits>>3)&1)-1])
        for k,(a,b) in enumerate(fields):full[9*bits+k,9*bits+k]=(a*a+b*b)/32+2*np.dot(rho,rho)
    for arc,(a,b) in enumerate(((0,1),(1,0))):
        U=np.kron(raiseE,eye) if arc==0 else np.kron(eye,raiseE)
        for spin in range(2):
            hop=annih[2*b+spin].T@annih[2*a+spin]
            T=np.kron(hop,U);full-=T+T.T
    states=[];indices=[]
    for bits in range(16):
        if sum((bits>>(2*v))&1 for v in range(2))!=1 or sum((bits>>(2*v+1))&1 for v in range(2))!=1:continue
        rho0=(bits&1)+((bits>>1)&1)-1
        for k,(a,b) in enumerate(fields):
            if a-b+rho0==0:states.append((bits,a,b));indices.append(9*bits+k)
    H=full[np.ix_(indices,indices)]
    direct=np.zeros_like(H);wrong=np.zeros_like(H)
    lookup={x:i for i,x in enumerate(states)}
    for col,(bits,a,b) in enumerate(states):
        direct[col,col]=H[col,col];wrong[col,col]=H[col,col]
        for e,(src,dst) in enumerate(((0,1),(1,0))):
            for d in (-1,1):
                A,B=(src,dst) if d==1 else (dst,src)
                E=[a,b];E[e]+=d
                if abs(E[e])>1:continue
                for spin in range(2):
                    p,q=2*A+spin,2*B+spin
                    if ((bits>>p)&1) and not ((bits>>q)&1):
                        after=bits^(1<<p);target=after|(1<<q)
                        parity=(bits&((1<<p)-1)).bit_count()+(after&((1<<q)-1)).bit_count()
                        row=lookup[(target,*E)]
                        direct[row,col]-=(-1)**parity;wrong[row,col]-=1
    psi=[]
    for bits,a,b in states:
        up=next(v for v in range(2) if (bits>>(2*v))&1)
        down=next(v for v in range(2) if (bits>>(2*v+1))&1)
        psi.append((-1)**int(down<up)*np.exp(-.45*int(up==down)-.15*(a*a+b*b)))
    psi=np.array(psi);guide=psi*psi+.04
    norm=np.dot(psi,psi);image=H@psi;E=np.dot(psi,image)/norm
    exact_var=np.dot(image,image)/norm-E*E
    node=np.zeros(len(states));node[0]=1;image_node=H@node
    # Fixed primitive proposals, invalid=self, including the noncontractible winding.
    proposal=np.zeros_like(H)
    for row,(bits,a,b) in enumerate(states):
        for col,(target,c,d) in enumerate(states):
            if row!=col and H[row,col]!=0:proposal[row,col]+=.4/8
        for sign in (-1,1):
            target=lookup.get((bits,a+sign,b+sign),row)
            proposal[row,target]+=.5/2
        n0=bits&3;n1=(bits>>2)&3
        target=lookup.get((bits^15,a,b),row) if (n0,n1) in ((1,2),(2,1)) else row
        proposal[row,target]+=.1
        proposal[row,row]+=1-proposal[row].sum()
    Q=proposal*np.minimum(1,guide[None,:]/guide[:,None])
    np.fill_diagonal(Q,0);np.fill_diagonal(Q,1-Q.sum(axis=1))
    stationary=guide/guide.sum()
    return dict(dimension=len(states),full_tensor_dimension=144,full_tensor_vs_CAR_entry_error=float(np.max(abs(H-direct))),
                wrong_unsigned_CAR_error=float(np.max(abs(H-wrong))),energy=float(E),full_residual=float(np.sqrt(exact_var)),
                reweighted_H2_error=float(abs(np.sum(guide*(image*image/guide))/norm-np.dot(image,image)/norm)),
                proposal_symmetry_error=float(np.max(abs(proposal-proposal.T))),
                detailed_balance_error=float(np.max(abs(stationary[:,None]*Q-stationary[None,:]*Q.T))),
                nodal_counterexample=dict(localenergy_variance=0.0,full_H_residual_squared=float(np.dot(image_node,image_node)-image_node[0]**2)),
                exact_ground_energy=float(np.linalg.eigvalsh(H)[0]))


def run_candidate(name,params,n=4000,response=False):
    table=s.pair_table(params['g'],params['eta']);parts=[];Gs=[];Ks=[];Cs=[];allbs=[];allEs=[];accept=[]
    coeff=source_geometry((1,0,0))['coefficients'][:2]
    for c in range(4):
        bs,Es,ac,pr=s.sample(713+c*1021,n,10000,15,table,params['a'],params['b'],params['tau'],params['shift'],params['epsilon'],params['g'])
        raw,G,K,C=s.observe(bs,Es,table,params['a'],params['b'],params['tau'],params['shift'],params['epsilon'],params['g'],coeff,response)
        parts.append(raw);Gs.append(G);Ks.append(K);Cs.append(C);allbs.append(bs);allEs.append(Es)
        accept.append(dict(accepted=ac.tolist(),proposed=pr.tolist(),changed_fraction=(ac/pr).tolist()))
        print(json.dumps(dict(candidate=name,chain=c,quality=estimate(raw,1))),flush=True)
    raw=np.concatenate(parts);bs=np.concatenate(allbs);Es=np.concatenate(allEs)
    result:dict[str,Any]=dict(name=name,parameters=params,quality=estimate(raw),chain_quality=[estimate(x,1) for x in parts],acceptance=accept,
                sample_max_Gauss=int(max(np.max(np.abs(s.INCIDENCE@E+s.charge(int(bits)))) for bits,E in zip(bs,Es))),
                matter_configurations_seen=len(set(map(int,bs))),fields_seen=len({tuple(x) for x in Es}),
                pair_histogram=np.bincount([sum(((int(bits)>>(2*v))&3)==3 for v in range(8)) for bits in bs],minlength=5).tolist())
    np.savez_compressed(ROOT/(P+name+'_raw.npz'),bits=bs,E=Es,raw=raw,gram=np.concatenate(Gs),moment=np.concatenate(Ks),contact=np.concatenate(Cs),table=table)
    save(name+'_result',result)
    return result,bs,Es,table


def response(name,params,bs,Es,table,qbits):
    coeff=source_geometry(qbits)['coefficients'][:2]
    raw,G,K,C=s.observe(bs,Es,table,params['a'],params['b'],params['tau'],params['shift'],params['epsilon'],params['g'],coeff,True)
    z=raw[:,0].mean();gram=G.mean(axis=0)/z;moment=K.mean(axis=0)/z;contact=C.mean(axis=0)/z
    energy=moment[0,0].real
    # Connected sources by exact finite linear algebra of sampled moments.
    T=np.zeros((5,4),complex);T[1:]=np.eye(4);T[0]=-gram[0,1:]
    S=T.conj().T@gram@T;M=T.conj().T@(moment-energy*gram)@T
    herm=float(np.max(abs(M-M.conj().T)));M=(M+M.conj().T)/2
    w,U=np.linalg.eigh(S);keep=w>1e-7
    whitening=U[:,keep]/np.sqrt(w[keep]);omega,V=eigh(whitening.conj().T@M@whitening)
    eigenvectors=whitening@V
    overlaps=eigenvectors.conj().T@S
    frequencies=np.array([.0,.2,.5,1.,2.])
    chi=[]
    for f in frequencies:
        zeta=f+.1j
        chi.append(sum(np.outer(o.conj(),o)/(om-zeta)+np.outer(o,o.conj())/(om+zeta) for om,o in zip(omega,overlaps)))
    chi=np.asarray(chi)
    tag='q'+''.join(map(str,qbits))
    np.savez_compressed(ROOT/(P+name+'_'+tag+'_response.npz'),coeff=coeff,gram=S,shifted_H=M,raw_gram=gram,raw_H=moment,
                        omega=omega,source_overlaps=overlaps,contact=contact,frequencies=frequencies,retarded_projected_chi=chi,
                        twist_response=contact-chi[:,2:,2:],raw=raw)
    result=dict(qbits=list(qbits),incidence_transverse_error=float(np.max(abs(s.INCIDENCE@coeff.T))),
                samples=len(bs),energy=float(energy),gram_eigenvalues=w.tolist(),ritz_shifted_energies=omega.tolist(),
                field_gram=np.real(S[:2,:2]).tolist(),current_gram=np.real(S[2:,2:]).tolist(),
                field_current_imag=np.imag(S[:2,2:]).tolist(),contact=contact.tolist(),
                unhermitized_shifted_H_error=herm,
                interpretation='Sampled connected four-source Ritz response, not stationary Lehmann poles. Nonzero residual and finite-chain errors prohibit photon or massless claims. Complex-frequency resolvent is projected; contact retained separately.')
    save(name+'_'+tag+'_response',result);print(json.dumps(result),flush=True)
    return result


def main():
    start=time.time();benchmark=small_benchmark();save('small_benchmark',benchmark);print(json.dumps(benchmark),flush=True)
    params=dict(g=.45,eta=0.,a=.15,b=0.,tau=0.,shift=-10.,epsilon=.2)
    first,_,_,_=run_candidate('slater',params)
    save('first_physics_checkpoint',dict(benchmark=benchmark,candidate=first,actual_elapsed_seconds=time.time()-start))
    # Material change: genuine pair correlations PLUS nonlocal current/H-image dressing.
    dressed=dict(g=.45,eta=.08,a=.15,b=.08,tau=.10,shift=-10.,epsilon=.2)
    second,bs,Es,table=run_candidate('pair_Himage',dressed)
    selected=second if second['quality']['energy']<first['quality']['energy'] else first
    if selected is first:
        params=first['parameters'];data=np.load(ROOT/(P+'slater_raw.npz'));bs=data['bits'];Es=data['E'];table=data['table']
    else:params=dressed
    # Separately thinned response from all chains, no evolved-output import.
    bs=bs[::5];Es=Es[::5]
    responses=[response(selected['name'],params,bs,Es,table,q) for q in ((1,0,0),(1,1,0))]
    save('result',dict(first=first,second=second,selected=selected['name'],responses=responses,small_benchmark=benchmark,
                       actual_elapsed_seconds=time.time()-start,scope='finite 8-site Ecap1 regulator; sampled energy and full H2 estimators, not exact norm/continuum/photon proof'))
    save('selected_state',dict(name=selected['name'],parameters=params,table_file=P+selected['name']+'_raw.npz',
                               representation='amplitude function + 65536 matter lookup only, never a field configuration list'))


if __name__=='__main__':main()
