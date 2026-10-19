"""Independent local LEFT/sewing/source/transport discriminator, not a CAR32 claim."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, expm_multiply
import rich_gauge_band_frame_20261016_model as b

ROOT=Path(__file__).resolve().parent
P='rich_review_wave49_20261016_'


def run():
    even=b.angular((0,1)); odd=b.angular((.5,1.5)); geo=b.geometry()
    s0=b.Sector(4);s1=b.Sector(5); spin=geo['spin']
    # Fresh Haar-neutral staggered-ion supplier. c remains scalar; D(g)c is sewn.
    ionmask=sum(int(n)<<i for i,n in enumerate(geo['ions']))
    neutral=np.zeros((s0.dimension,len(even['labels'])),complex)
    neutral[s0.lookup[ionmask],0]=1
    neutral=neutral.ravel(); B=[]
    for mu in range(8):
        source=sp.csr_matrix((s1.dimension*20,s0.dimension*10),dtype=complex)
        for mode in range(8):
            def values(points,mu=mu,mode=mode):
                return np.array([expm(-1j*a*spin[2])@expm(-1j*c*spin[1])@expm(-1j*d*spin[2]) for a,c,d in points])[:,mu,mode]
            source+=sp.kron(s1.source(s0,mode,True),b.multiplication(odd,even,values),format='csr')
        B.append(b.clean(source))
    Le=[sp.kron(sp.eye(s0.dimension),L,format='csr') for L in even['L']]
    Lo=[sp.kron(sp.eye(s1.dimension),L,format='csr') for L in odd['L']]
    # Check only fresh Haar input, where angular source image is exact.
    comm=[]
    for a in range(3):
        for mu in range(8):
            actual=Lo[a]@(B[mu]@neutral)-B[mu]@(Le[a]@neutral)
            expected=-sum(spin[a][mu,nu]*(B[nu]@neutral) for nu in range(8))
            comm.append(np.linalg.norm(actual-expected))
    strengths=np.array([np.linalg.norm(q@neutral)**2 for q in B])
    eigenvalues,eigenvectors=np.linalg.eigh(-spin[2].T)
    mu=max([i for i,x in enumerate(eigenvalues) if x>.49],key=lambda i:np.dot(abs(eigenvectors[:,i])**2,strengths))
    polarization=eigenvectors[:,mu]
    charged=sum(c*(q@neutral) for c,q in zip(polarization,B));charged/=np.linalg.norm(charged)
    dl=len(charged);de=len(neutral);size=dl*de
    def transfer(x):
        return sum(np.asarray(q.conj().T@x)@q.T for q in B)
    def adjoint(y):
        return sum(np.asarray(q@y)@q.conj() for q in B)
    def action(v):
        v=np.asarray(v).reshape(-1);x=v[:size].reshape(dl,de);y=v[size:].reshape(de,dl)
        return np.concatenate([adjoint(y).ravel(),transfer(x).ravel()])
    H=LinearOperator((2*size,2*size),matvec=action,rmatvec=action,dtype=complex)
    def generator(v):return -1j*action(v)
    A=LinearOperator(H.shape,matvec=generator,rmatvec=lambda v:1j*action(v),dtype=complex)
    start=np.concatenate([np.outer(charged,neutral).ravel(),np.zeros(size,complex)])
    final=expm_multiply(.51*A,start,traceA=0)
    def density(state):
        x=state[:size].reshape(dl,de);y=state[size:].reshape(de,dl)
        L2=[sum(np.linalg.norm(L@x)**2 for L in Lo)+sum(np.linalg.norm(L@y)**2 for L in Le),sum(np.linalg.norm(x@L.T)**2 for L in Le)+sum(np.linalg.norm(y@L.T)**2 for L in Lo)]
        excess=[np.linalg.norm(x)**2,np.linalg.norm(y)**2]
        vectors=[[float((np.vdot(x,Lo[a]@x)+np.vdot(y,Le[a]@y)).real) for a in range(3)],[float((np.vdot(x,x@Le[a].T)+np.vdot(y,y@Lo[a].T)).real) for a in range(3)]]
        return dict(physical_local_LEFT_Casimir=L2,physical_local_Noether_vector=vectors,charge_excess=excess,total_N_fraction=[(4+excess[0])/9,(4+excess[1])/9],norm=float(np.linalg.norm(state)**2))
    initial=density(start);out=density(final)
    # CENTRAL sewing is pointwise: D(-g)=-D(g), scalar c creation toggles N parity.
    seam=max(np.linalg.norm(expm(-2j*np.pi*a)+np.eye(8)) for a in spin)
    delta=np.array(out['physical_local_LEFT_Casimir'])-initial['physical_local_LEFT_Casimir']
    assert max(comm)<1e-10,comm
    assert out['charge_excess'][1]>.02 and delta[1]>.02,(out,delta)
    assert np.linalg.norm(delta)>.02 # A pinned/global-only spin-density countercase reaches assertion.
    vector_delta=np.array(out['physical_local_Noether_vector'])-initial['physical_local_Noether_vector']
    assert vector_delta[1,2]>.02,vector_delta
    np.savez_compressed(ROOT/(P+'local_benchmark_raw.npz'),initial_state=start,final_state=final,source_strengths=strengths,source_component=mu,local_source_state=charged,neutral_state=neutral,commutator_errors=comm)
    result=dict(kind='OWN_FRESH_TWO_CELL_LOCAL_LEFT_COUNTERCASE',initial=initial,final=out,local_spin_density_change=delta.tolist(),wrong_pinned_global_spin_density_error=float(np.linalg.norm(delta)),central_seam_error=float(seam),LEFT_source_commutator_error=float(max(comm)),all_eight_source_strengths=strengths.tolist(),Gauss_errors=[s0.gauss_error,s1.gauss_error],Hilbert_dimension=2*size,limits='All CAR8 bands per cell in N4/N5 blocks, all g multiplets 0,1 and .5,1.5. Two-cell charge-transfer image law only: excludes N3/N6 intermediate sectors, r/R and electromagnetic dynamics; NOT full CAR32 acceptance or localization theorem. n/N deliberately reported as occupied-background density, not excitation PVM.')
    result['physical_Noether_vector_density_change']=vector_delta.tolist()
    result['wrong_global_vector_density_error']=float(np.linalg.norm(vector_delta))
    (ROOT/(P+'local_benchmark_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
