"""Independent exact full-H response derivatives on V64 unbounded integer rotors.
A finite-support declared state is not an electric Hilbert-space cutoff.
No projected trace propagation, thermal claim, producer states or outputs.
"""
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
P='rich_review_wave43_20261016_'


def add(out,key,value):
    out[key]=out.get(key,0j)+value


def flux_step(flux,e,delta):
    d=dict(flux);d[e]=d.get(e,0)+delta
    if d[e]==0:del d[e]
    return tuple(sorted(d.items()))


def hop(mask,source,target):
    if not (mask>>source&1) or mask>>target&1:return None
    sign=(-1)**((mask&((1<<source)-1)).bit_count())
    mask^=1<<source
    sign*=(-1)**((mask&((1<<target)-1)).bit_count())
    return mask^(1<<target),sign


def inner(a,b):return sum(np.conj(value)*b.get(key,0j) for key,value in a.items())


def scale(a,x):return {k:v*x for k,v in a.items() if abs(v*x)>1e-15}


def plus(a,b,x=1):
    out=a.copy()
    for k,v in b.items():add(out,k,x*v)
    return {k:v for k,v in out.items() if abs(v)>1e-14}


def main():
    cfg=json.loads((ROOT/(P+'rotor64_input.json')).read_text())
    L=cfg['L'];sites=list(itertools.product(range(L),repeat=3));lookup={s:i for i,s in enumerate(sites)}
    arcs=[]
    for a,s in enumerate(sites):
        for axis in range(3):
            dest=list(s);dest[axis]=(dest[axis]+1)%L
            arcs.append((a,lookup[tuple(dest)],axis,np.array(s)+np.eye(3)[axis]/2))
    V=len(sites)
    mask=sum(1<<(2*i+(sum(s)%2)) for i,s in enumerate(sites))
    omega={(mask,()):1.+0j}
    def diagonal(key):
        occ,flux=key
        return sum(m*m for _,m in flux)/(2*cfg['I'])+cfg['U']/2*sum((((occ>>(2*v))&1)+((occ>>(2*v+1))&1)-1)**2 for v in range(V))
    def H(state,kinetic=True):
        out={}
        for (occ,flux),value in state.items():
            add(out,(occ,flux),diagonal((occ,flux))*value)
            if not kinetic:continue
            for e,(a,b,_,_) in enumerate(arcs):
                for species in (0,1):
                    for source,target,delta in ((b,a,1),(a,b,-1)):
                        moved=hop(occ,2*source+species,2*target+species)
                        if moved is not None:add(out,(moved[0],flux_step(flux,e,delta)),-cfg['t']*moved[1]*value)
        return {k:v for k,v in out.items() if abs(v)>1e-14}
    # NEW declared nonstationary interacting quantum state, no solved vacuum claim.
    psi=plus(omega,H(omega),-cfg['correlation'])
    psi=scale(psi,1/np.sqrt(inner(psi,psi).real))
    hp=H(psi);energy=inner(psi,hp).real
    q=np.array(cfg['q']);responses={};arrays={}
    def gauge_error(state):
        error=0
        for (occ,flux) in state:
            div=np.zeros(V,int)
            for e,m in flux:
                a,b,_,_=arcs[e];div[a]+=m;div[b]-=m
            rho=np.array([((occ>>(2*v))&1)+((occ>>(2*v+1))&1)-1 for v in range(V)])
            error=max(error,int(np.max(abs(div-rho))))
        return error
    gauss=[gauge_error(psi),gauge_error(hp)]
    for axis in (1,2):
        for sign in (-1,1):
            weights=np.array([np.exp(1j*sign*q@mid)*(a==axis)/np.sqrt(V) for _,_,a,mid in arcs])
            def field(state):return {key:value*sum(weights[e]*m for e,m in key[1]) for key,value in state.items() if key[1]}
            f=field(psi);mean=inner(psi,f);f=plus(f,psi,-mean)
            hf=H(f);centered=plus(hf,f,-energy)
            # i[H,E] exact full-H vector, INCLUDING out-of-state-support virtual configurations.
            jp=scale(plus(H(field(psi)),field(hp),-1),1j)
            c0=inner(f,f).real;m1=inner(f,centered);m2=inner(centered,centered).real
            jweight=inner(jp,jp).real
            times=np.array(cfg['derivative_sample_times'])
            expansion=c0-1j*times*m1-.5*times**2*m2
            label=f'T{axis}_q{sign}'
            responses[label]=dict(E_weight=c0,first_spectral_moment=[float(m1.real),float(m1.imag)],second_spectral_moment=m2,
                current_weight=jweight,field_Himage_support=len(hf),current_support=len(jp),
                response_class='exact full-H C(0), Cprime(0), Cdoubleprime(0); displayed quadratic is NOT finite-time controlled propagation',
                short_time_polynomial_real=expansion.real.tolist(),short_time_polynomial_imag=expansion.imag.tolist())
            arrays[label+'_coefficients']=np.array([c0,m1,m2,jweight],complex)
            gauss.append(gauge_error(hf));gauss.append(gauge_error(jp))
    # A physical source phase exp(i eta E_T,cos) leaves all diagonal fields/densities initially fixed.
    w=np.array([np.cos(q@mid)*(a==1)/np.sqrt(V) for _,_,a,mid in arcs])
    def evalue(key):return sum(w[e]*m for e,m in key[1])
    pulse={key:value*np.exp(1j*cfg['pulse']*evalue(key)) for key,value in psi.items()}
    def E(state):return {k:v*evalue(k) for k,v in state.items()}
    def flow(state,bare=False):return 1j*inner(state,plus(H(E(state),not bare),E(H(state,not bare)),-1))
    flow0=flow(psi);flow1=flow(pulse);bare=flow(pulse,True)
    # Exact bilateral U acts outside physical Gauss space, as it must without charged matter shift.
    check=(mask,((0,3),));roundtrip=(check[0],flux_step(flux_step(check[1],0,1),0,-1))
    result=dict(kind='OWN_V64_FULL_ROTOR_QUANTUM_STATE_FULL_H_RESPONSE_DERIVATIVES',input=cfg,V=V,arcs=len(arcs),
        state='normalized (1-a Hhop)|Neel one fermion/site,E0>; finite support in unbounded Hilbert space',
        state_support=len(psi),Hstate_support=len(hp),source_energy=float(energy),
        full_H_residual_squared=float(inner(plus(hp,psi,-energy),plus(hp,psi,-energy)).real),
        Gauss_max=max(gauss),U_roundtrip_exact=(roundtrip==check),
        source_phase_initial_diagonal_probability_change=max(abs(abs(psi[k])**2-abs(pulse[k])**2) for k in psi),
        baseline_Edot=[float(flow0.real),float(flow0.imag)],driven_Edot=[float(flow1.real),float(flow1.imag)],
        bare_HE_plus_V_Edot=[float(bare.real),float(bare.imag)],responses=responses,
        retained='all full-H outgoing hopping/onsite/electric configurations for every displayed derivative; both original same-charge species',
        limits=['nonstationary OWN declared state, not author thermal state or ground',
            'finite exact derivative order, not projected time propagation, Maxwell pole, c, stability or real-time photon',
            'no claim branch joined to Band40 tree model; common rotor algebra only'])
    np.savez_compressed(ROOT/(P+'rotor64_raw.npz'),times=np.array(cfg['derivative_sample_times']),**arrays)
    (ROOT/(P+'rotor64_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    assert max(gauss)==0 and roundtrip==check
    assert abs(flow1-flow0)>.001 and abs(bare)<1e-12
    assert all(v['E_weight']>0 and v['current_weight']>0 for v in responses.values())


if __name__=='__main__':main()
