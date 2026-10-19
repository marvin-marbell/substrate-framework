"""Fresh actual V64 joint states and raw nonstationary transverse response."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
import rich_bulk_photon_20261016_state as base
from rich_bulk_photon_20261016_run import estimate
import rich_flux_phase_photon_20261016_state as s
ROOT=Path(__file__).resolve().parent
P='rich_flux_phase_photon_20261016_'


def save(name,data):
    (ROOT/(P+name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def ring():
    basis=[sum(1<<i for i in u)+sum(1<<(i+4) for i in d) for u in itertools.combinations(range(4),2) for d in itertools.combinations(range(4),2)]
    index={m:i for i,m in enumerate(basis)};out={}
    for flux in (0.,np.pi):
        H=np.zeros((36,36),complex)
        for col,mask in enumerate(basis):
            H[col,col]=4*sum(bool(mask&(1<<i)) and bool(mask&(1<<(4+i))) for i in range(4))
            for i in range(4):
                j=(i+1)%4
                for direction in (-1,1):
                    a,b=(i,j) if direction==1 else (j,i)
                    for spin in range(2):
                        aa=a+4*spin;bb=b+4*spin
                        if not mask&(1<<aa) or mask&(1<<bb):continue
                        intermediate=mask^(1<<aa)
                        sign=(-1)**((mask&((1<<aa)-1)).bit_count()+(intermediate&((1<<bb)-1)).bit_count())
                        row=index[intermediate|(1<<bb)]
                        H[row,col]-=sign*np.exp(1j*direction*flux/4)
        eig=np.linalg.eigvalsh(H)
        out[str(flux)]=dict(ground=float(eig[0]),spectrum=eig.tolist(),hermiticity=float(np.max(abs(H-H.conj().T))))
    out['meaning']='Exact full36-dimensional Nup=Ndown2 Hubbard U4 clamped-cnumber ring only; NOT the nonunitary qutrit-link vacuum.'
    save('ring',out)
    return out


def geometry():
    codes=s.flux_codes(True);plaquette=s.LOOPS[:192]@codes;winding=s.LOOPS[192:]@codes
    result=dict(sites=64,arcs=len(s.ARCS),distinct_positive_arcs=len(set(map(tuple,s.ARCS.tolist()))),incidence_column_sum_max=int(abs(s.INCIDENCE.sum(axis=0)).max()),loop_divergence_max=int(abs(s.INCIDENCE@s.LOOPS.T).max()),plaquettes=len(plaquette),pi_plaquettes=int(np.sum(plaquette%2==1)),windings=len(winding),trivial_windings=int(np.sum(winding%2==0)),plaquette_integer_flux=plaquette.tolist(),winding_integer_flux=winding.tolist(),external_H_phase=0.,fixed_law=dict(L=4,Nup=32,Ndown=32,ions=1,t=1,U=4,I=16,Ecap=1,local_Fock=4,link_dimension=3,magnetic_term=0))
    assert result['distinct_positive_arcs']==192 and result['pi_plaquettes']==192 and result['trivial_windings']==48 and result['loop_divergence_max']==0
    save('geometry',result);return result


def influence_tau(raw):
    w=raw[:,0];energy=raw[:,1].sum()/w.sum();x=raw[:,1]-energy*w;x-=x.mean()
    variance=x@x
    if variance==0:return dict(tau=1.,autocorrelation_ESS=float(len(x)),max_lag=0)
    correlations=np.array([x[:-k]@x[k:]/variance for k in range(1,min(len(x)//3,200))])
    total=0.;used=0
    for k in range(0,len(correlations)-1,2):
        pair=correlations[k]+correlations[k+1]
        if pair<=0:break
        total+=pair;used=k+2
    tau=max(1.,1+2*total)
    return dict(tau=float(tau),autocorrelation_ESS=float(len(x)/tau),max_lag=used,meaning='Initial-positive paired autocorrelation estimate of weighted-energy influence, exploratory not mixing proof')


def diagnostics(u,d,E,F,G,codes,cfg,coeff,density):
    args=(F,G,codes,*cfg['correlation'],*cfg['guide'])
    realcoeff=coeff.real.astype(complex)
    v,h,c,w,stable,hb=s.evaluate(u.copy(),d.copy(),E.copy(),*args,realcoeff,density)
    step=1e-4
    plus=s.evaluate(u.copy(),d.copy(),E.copy(),*args,realcoeff,density,step)[1]
    minus=s.evaluate(u.copy(),d.copy(),E.copy(),*args,realcoeff,density,-step)[1]
    fd1=(plus-minus)/(2*step);fd2=(plus+minus-2*h)/(step*step)
    car=[];px=base.parity(u,d);mask=sum(1<<int(2*i) for i in u)+sum(1<<int(2*i+1) for i in d)
    for a,b,mu in s.ARCS:
        for spin in range(2):
            occ=u if spin==0 else d
            if a not in occ or b in occ:continue
            aa=2*int(a)+spin;bb=2*int(b)+spin;intermediate=mask^(1<<aa)
            sign=(-1)**((mask&((1<<aa)-1)).bit_count()+(intermediate&((1<<bb)-1)).bit_count())
            nu=u.copy();nd=d.copy();o=nu if spin==0 else nd;o[np.nonzero(o==a)[0][0]]=b
            car.append(abs(sign*base.parity(nu,nd)-px))
    neel=np.nonzero(s.POSITIONS.sum(axis=1)%2==0)[0];other=np.setdiff1d(np.arange(64),neel)
    rng=np.random.default_rng(8231);nodalF=rng.normal(size=(64,64));nodalF[0]=0.;nodalG=rng.normal(size=(64,64))
    nv,nh,nc,nw,ns,nb=s.evaluate(neel.copy(),other.copy(),np.zeros(192,np.int64),nodalF,nodalG,np.zeros(192,np.int64),*cfg['correlation'],*cfg['guide'],realcoeff,density)
    return dict(derivative_step=step,current_first_relative=float(abs(fd1-v[3])/max(abs(v[3]),1e-200)),contact_second_relative=float(abs(fd2*v[0]-c[0,0])/max(abs(c[0,0]),1e-200)),CAR_128mode_Python_integer_edges=len(car),CAR_max_error=max(car),nodal_psi_squared=float(abs(nv[0])**2),nodal_Hpsi_squared=float(abs(nh)**2),nodal_positive_guide=float(nw),nodal_H2_over_guide=float(abs(nh)**2/nw),nodal_stable_inverse=bool(ns),nodal_meaning='Constructed exact determinant row node, full neighbor determinant fallback; separate estimator adverse diagnostic, not production state.')


def response(v,hb,hp,contacts,diagonal_source,q,quality,label,frequency):
    norm=float(np.mean(abs(v[:,0])**2));psi=v[:,0]
    B=np.column_stack([v[:,1:3],v[:,5],v[:,1:3].conj(),v[:,5].conj()]);HB=np.column_stack([hb,hb.conj()])
    # Physical source EOM is [H,B]Omega, NOT (H-<H>)B Omega.
    # Form B Hpsi without dividing by psi, including exact determinant nodes.
    eom_plus=hb-diagonal_source*hp[:,None]
    EOM=np.column_stack([eom_plus,eom_plus.conj()])
    means=np.mean(psi.conj()[:,None]*B,axis=0)/norm
    B-=psi[:,None]*means;HB-=hp[:,None]*means
    G=B.conj().T@B/len(B)/norm
    Kfull=B.conj().T@HB/len(B)/norm-quality['energy']*G
    K=B.conj().T@EOM/len(B)/norm
    HGram=HB.conj().T@HB/len(B)/norm
    J=np.column_stack([v[:,3:5],-v[:,3:5].conj()]);jm=np.mean(psi.conj()[:,None]*J,axis=0)/norm;J-=psi[:,None]*jm
    JJ=J.conj().T@J/len(B)/norm;BJ=B.conj().T@J/len(B)/norm
    eg,ug=np.linalg.eigh(G);keep=eg>1e-10*max(eg);W=ug[:,keep]/np.sqrt(eg[keep])[None,:]
    generator=W.conj().T@K@W;amplitude=W.conj().T@BJ
    outside=JJ-amplitude.conj().T@amplitude
    contact=contacts.mean(axis=0)/norm
    chi=[]
    for omega in frequency:
        z=omega+.15j;eye=np.eye(generator.shape[0])
        plus=amplitude[:,2:];minus=amplitude[:,:2]
        chi.append(plus.conj().T@np.linalg.solve(generator-z*eye,plus)+(minus.conj().T@np.linalg.solve(generator+z*eye,minus)).T)
    chi=np.asarray(chi);twist=contact[None]-chi
    Gdot=1j*(K-K.conj().T)
    # Sdot is the unique Hermitian Sylvester solution on the observed positive Gram span.
    S=np.diag(np.sqrt(eg[keep]));U=ug[:,keep];gd=U.conj().T@Gdot@U
    Sdot=gd/(np.sqrt(eg[keep])[:,None]+np.sqrt(eg[keep])[None,:])
    connection=1j*Sdot@np.linalg.inv(S)
    # At this real t0 and for commuting diagonal field/density observables,
    # exact Gdot=0 by real-H time reversal. Sample anti-Hermitian drift is MC noise.
    exact_initial_Gdot=np.zeros_like(Gdot)
    np.savez_compressed(ROOT/(P+label+'_response.npz'),q=q,G=G,K=K,Kfull=Kfull,EOM_vectors=EOM,HGram=HGram,Gdot_sample=Gdot,Gdot_exact_initial=exact_initial_Gdot,gram_eigenvalues=eg,whitening=W,frozen_generator=generator,moving_connection_sample=connection,moving_connection_exact_initial=np.zeros_like(connection),currentGram=JJ,currentOverlap=BJ,currentAmplitudes=amplitude,currentOutside=outside,contact=contact,frequency=frequency,chi_frozen=chi,twist_frozen=twist,source_means=means,current_means=jm)
    return dict(q=np.asarray(q).tolist(),rank=int(keep.sum()),gram_eigenvalues=eg.tolist(),K_antihermitian_relative=float(np.linalg.norm(K-K.conj().T)/max(np.linalg.norm(K),1e-300)),metric_derivative_norm=float(np.linalg.norm(Gdot)),moving_connection_norm=float(np.linalg.norm(connection)),current_norm=JJ.diagonal().real.tolist(),current_outside_fraction=(outside.diagonal().real/JJ.diagonal().real).tolist(),contact_real=contact.real.tolist(),contact_imag=contact.imag.tolist(),frozen_static_twist_real=twist[0].real.tolist(),frozen_generator_eigenvalues_real=np.linalg.eigvals(generator).real.tolist(),frozen_generator_eigenvalues_imag=np.linalg.eigvals(generator).imag.tolist(),interpretation='Raw exact-all-hop two transverse F and density +/-q source response. K NOT Hermitized. Frozen-time projected resolvent with +/-q current and contact is NOT stationary Kubo/Lehmann or photon poles. Metric derivative and moving-basis connection retained separately; no reducible kernel Dyson iteration, no forced current closure.')


def main():
    start=time.time();cfg=json.loads((ROOT/(P+'input.json')).read_text());ring_result=ring();geo=geometry();results={}
    for order in ('pair','af_dimer'):
        for pi in (False,True):
            label=('pi_' if pi else 'zero_')+order
            F,G,codes,metadata=s.kernel(pi,order,cfg['delta'])
            np.savez_compressed(ROOT/(P+label+'_state.npz'),F=F,G=G,codes=codes,positions=s.POSITIONS,arcs=s.ARCS,incidence=s.INCIDENCE,loops=s.LOOPS)
            coeff,density,polarizations=base.sources(cfg['momenta'][0]);us=[];ds=[];Es=[];chains=[];allraw=[]
            for seed in cfg['sampling']['seeds']:
                mc=cfg['sampling'];u,d,E,accepted,proposed=base.sample(seed,mc['n'],mc['burn'],mc['thin'],F,G,*cfg['correlation'],*cfg['guide'])
                raw,v,hb,c,hp=s.measure(u,d,E,F,G,codes,*cfg['correlation'],*cfg['guide'],coeff,density)
                quality=estimate(raw,mc['blocks']);quality.update(influence_tau(raw));quality.update(seed=seed,accepted=accepted.tolist(),proposed=proposed.tolist())
                chains.append(quality);us.append(u);ds.append(d);Es.append(E);allraw.append(raw)
                print(json.dumps(dict(label=label,chain=quality)),flush=True)
            u=np.concatenate(us);d=np.concatenate(ds);E=np.concatenate(Es);raw=np.concatenate(allraw);quality=estimate(raw,cfg['sampling']['blocks']*len(chains))
            maxgauss=max(int(abs(s.INCIDENCE@ee+np.bincount(np.r_[uu,dd],minlength=64)-1).max()) for uu,dd,ee in zip(u,d,E))
            diag=diagnostics(u[0],d[0],E[0],F,G,codes,cfg,coeff,density);responses=[]
            for iq,q in enumerate(cfg['momenta']):
                coeff,density,polarizations=base.sources(q)
                raw,v,hb,c,hp=s.measure(u,d,E,F,G,codes,*cfg['correlation'],*cfg['guide'],coeff,density)
                qlabel=label+'_q'+str(iq)
                np.savez_compressed(ROOT/(P+qlabel+'_raw.npz'),up=u,down=d,E=E,raw=raw,vectors=v,Hbasis=hb,Hpsi=hp,contact=c,coeff=coeff,density=density,polarizations=polarizations)
                occupations=np.zeros((len(u),64))
                for row,(uu,dd) in enumerate(zip(u,d)):occupations[row]=np.bincount(np.r_[uu,dd],minlength=64)
                diagonal_source=np.column_stack([E@coeff.T/16.,(occupations-1)@density])
                responses.append(response(v,hb,hp,c,diagonal_source,q,quality,qlabel,np.linspace(0,8,65)))
            phase_control=None
            if pi:
                control=s.measure(u,d,E,F,G,np.zeros(192,np.int64),*cfg['correlation'],*cfg['guide'],coeff,density)[0]
                np.savez_compressed(ROOT/(P+label+'_phase_control_raw.npz'),raw=control)
                phase_control=estimate(control,cfg['sampling']['blocks']*len(chains))
            results[label]=dict(metadata=metadata,quality=quality,chains=chains,maxGauss=maxgauss,diagnostics=diag,responses=responses,zero_phase_same_pi_orbitals=phase_control)
            save(label+'_result',results[label]);save('result',dict(results=results,ring=ring_result,geometry=geo,elapsed_seconds=time.time()-start))
            print(json.dumps(dict(completed=label,quality=quality,maxGauss=maxgauss,diagnostics=diag)),flush=True)
    save('completion',dict(natural_scientific_completion=True,states=list(results),elapsed_seconds=time.time()-start,interpretation='Both pair and material AF-correlated oriented dimer alternative actually run, no parameter fitting or changed Hamiltonian. Finite nonstationary data only.'))
    print(json.dumps(dict(natural_scientific_completion=True,elapsed=time.time()-start)),flush=True)


if __name__=='__main__':main()
