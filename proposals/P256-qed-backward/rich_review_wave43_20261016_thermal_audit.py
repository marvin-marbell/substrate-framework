"""Independent finite-path Fock consumer of Rotor41 declared thermal helpers."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from rich_unitary_rotor_photon_20261016_thermal import path_data, hs_slice, hopping, current_matrix, heat
from rich_review_wave43_20261016_rotor import annihilator

ROOT=Path(__file__).resolve().parent
P='rich_review_wave43_20261016_'


def lift(matrix):
    """Exact CAR Fock representation by exterior minors, never log of a link."""
    V=len(matrix);out=np.zeros((1<<V,1<<V),complex)
    occup=[[i for i in range(V) if b>>i&1] for b in range(1<<V)]
    for i,I in enumerate(occup):
        for j,J in enumerate(occup):
            if len(I)==len(J):out[i,j]=np.linalg.det(matrix[np.ix_(I,J)]) if I else 1
    return out


def main():
    cfg=json.loads((ROOT/(P+'thermal_audit_input.json')).read_text())
    rng=np.random.default_rng(cfg['seed']);V=4;M=3;beta=cfg['beta'];dt=beta/M
    arcs=np.array([[0,1,1],[1,2,2],[2,3,1],[3,0,2]])
    D=np.zeros((V,4))
    for e,(a,b,_) in enumerate(arcs):D[a,e]=1;D[b,e]=-1
    theta=rng.normal(0,.24,(M,4));alpha=np.array([.13,-.17,.21,.07]);s=rng.choice([-1,1],(M,V))
    data=path_data(theta,alpha,s,arcs,D,beta,I=cfg['I'])
    # Independently exact sequential Metropolis for identical HS random draws.
    seed=cfg['sweep_seed'];ra=np.random.default_rng(seed);rb=np.random.default_rng(seed)
    actual,accepted=hs_slice(s,data,1,ra);expected=s.copy();own_accepts=0
    for v in rb.permutation(V):
        proposal=expected.copy();proposal[1,v]*=-1
        before=path_data(theta,alpha,expected,arcs,D,beta,I=cfg['I'])['logweight']
        after=path_data(theta,alpha,proposal,arcs,D,beta,I=cfg['I'])['logweight']
        if np.log(rb.random())<after-before:expected=proposal;own_accepts+=1
    B=data['B'];Pmat=data['P'];Z=np.diag(np.exp(-1j*alpha));C=Z@Pmat
    n0=np.linalg.solve(np.eye(V)+C,C);FC=lift(C);partition=np.trace(FC)
    c=[annihilator(i,V) for i in range(V)]
    def bilinear(A):return sum((A[i,j]*c[i].conj().T@c[j] for i,j in itertools.product(range(V),repeat=2)),np.zeros((1<<V,1<<V),complex))
    coeff=np.array([1,.3j,-.4,.2j])
    A0=current_matrix(theta[0],arcs,coeff.conjugate(),V)
    Q=np.eye(V,dtype=complex);errors=[];values=[]
    for lag in range(M):
        if lag:Q=B[lag-1]@Q
        invQ=np.linalg.inv(Q);n=Q@n0@invQ
        A=current_matrix(theta[lag],arcs,coeff,V)
        wick=np.trace(A@n)*np.trace(A0@n0)+np.trace(A@Q@(np.eye(V)-n0)@A0@n0@invQ)
        FQ=lift(Q)
        exact=np.trace(np.linalg.solve(FQ,bilinear(A)@FQ)@bilinear(A0)@FC)/partition
        errors.append(float(abs(wick-exact)));values.append([float(exact.real),float(exact.imag)])
    # Physical current/contact source derivative at actual complex link angle.
    realcoef=coeff.real;epsilon=cfg['difference_step']
    hp=hopping(theta[1]+epsilon*realcoef,arcs,V);hm=hopping(theta[1]-epsilon*realcoef,arcs,V);h0=hopping(theta[1],arcs,V)
    j=current_matrix(theta[1],arcs,realcoef,V)
    contact=current_matrix(theta[1],arcs,realcoef**2,V,True)
    current_error=np.linalg.norm((hp-hm)/(2*epsilon)+j)
    contact_error=np.linalg.norm((hp-2*h0+hm)/epsilon**2-contact)
    # Poisson/full integer Fourier electric insertions on a physically retained kernel.
    x=np.array([.11,-.23]);logk,score,second=heat(x,dt,cfg['I']);m=np.arange(-40,41)
    km=np.exp(-dt*m*m/(2*cfg['I']))
    fk=np.sum(km*np.exp(1j*x[:,None]*m),axis=1)
    first=np.sum(m*km*np.exp(1j*x[:,None]*m),axis=1)/fk
    second_exact=np.sum(m*m*km*np.exp(1j*x[:,None]*m),axis=1)/fk
    kernel_error=float(np.max(abs(np.exp(logk)-fk)))
    E_error=float(np.max(abs(-1j*score-first)))
    E2_error=float(np.max(abs(-second-second_exact)))
    result=dict(kind='OWN_CONDITIONAL_PATH_EXACT_FOCK_THERMAL_HELPER_AUDIT',input=cfg,
        HS_sweep_configuration_equal=bool(np.array_equal(actual,expected)),HS_accepts=[accepted,own_accepts],
        full_Fock_unequal_time_Wick_errors=errors,conditional_current_correlators=values,
        current_derivative_error=float(current_error),contact_derivative_error=float(contact_error),
        Poisson_vs_integer_kernel_error=kernel_error,E_insertion_error=E_error,E2_insertion_error=E2_error,
        scope='Declared fixed finite path, exact full 16-Fock all particle sectors for one PH species; NOT a thermal MC state or statistical convergence claim',
        retained_fields='all theta/alpha/HS; physical same-charge mapping checked separately on original spinful Fock',
        discretization='point-operator current insertions between Trotter factors, not exact finite-dt derivative of exp(-dt h) when [h,J]!=0')
    (ROOT/(P+'thermal_audit_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(P+'thermal_audit_raw.npz'),theta=theta,alpha=alpha,HS=s,unequal_time=np.array(values))
    print(json.dumps(result,indent=2),flush=True)
    assert np.array_equal(actual,expected) and accepted==own_accepts
    assert max(errors)<1e-9 and current_error<1e-6 and contact_error<1e-5
    assert max(kernel_error,E_error,E2_error)<1e-9


if __name__=='__main__':main()
