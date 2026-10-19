"""Exact full-H two-site Gauss-sector thermal spectral benchmark; no E cap."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parent
P='rich_unitary_rotor_photon_20261016_'


def main():
    basis=[n for n in range(16) if n.bit_count()==2];index={n:i for i,n in enumerate(basis)}
    hop=np.zeros((6,6));J=np.zeros((6,6),complex);E=np.zeros(6);ons=np.zeros(6)
    for k,n in enumerate(basis):
        n0=((n>>0)&1)+((n>>1)&1);E[k]=n0-1;ons[k]=4*(n0-1)**2
        for a,b in ((0,2),(2,0),(1,3),(3,1)):
            if n>>b&1 and not n>>a&1:
                mid=n^(1<<b);sign=(-1)**((n&((1<<b)-1)).bit_count()+(mid&((1<<a)-1)).bit_count())
                row=index[mid|(1<<a)];hop[row,k]-=sign;J[row,k]+=1j*sign*(1 if a<2 else -1)
    electric=np.diag(E);diag=np.diag(E**2/32+ons);H=hop+diag
    w,v=np.linalg.eigh(H);beta=.8;weight=np.exp(-beta*w);Z=weight.sum()
    ops=np.array([v.conj().T@op@v for op in (electric,J,-hop)])
    tau=np.linspace(0,beta,9);C=np.zeros((2,2,len(tau)),complex)
    for a in range(2):
        for b in range(2):
            for k,t in enumerate(tau):
                C[a,b,k]=np.sum(np.exp(-(beta-t)*w[:,None]-t*w[None,:])*ops[a]*ops[b].T)/Z
    means=np.array([np.dot(weight,np.diag(a))/Z for a in ops])
    approximations=[]
    for M in (1,2,4,8,16):
        transfer=expm(-beta/M*hop)@expm(-beta/M*diag);ZM=np.trace(np.linalg.matrix_power(transfer,M))
        approximations.append(dict(M=M,dt=beta/M,trace=float(ZM),relative_trace_error=float(abs(ZM-Z)/Z)))
    dot=1j*(H@electric-electric@H);commutator_error=float(np.max(abs(dot-J)))
    output=dict(kind='OWN_EXACT_FULL_H_PHYSICAL_THERMAL_RESPONSE_NOT_V64_EXTRAPOLATION',beta=beta,I=16,U=4,t=1,physical_basis_dimension=6,
                exact_trace=float(Z),Trotter_trace_comparison=approximations,
                EE_tau0=float(C[0,0,0].real),EE_tauhalf=float(C[0,0,4].real),
                JJ_tau0=float(C[1,1,0].real),JJ_tauhalf=float(C[1,1,4].real),
                contact=float(means[2].real),electric_Heisenberg_dot_equals_J_error=commutator_error,
                scope='Tree Gauss fixes E=n0-1 exactly, not an electric cutoff. Physical full-H spectral response and actual Trotter error, no analytic continuation. Same-H Hhop produces nonconserved E; electric-only reference would have constant EE and zero J.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(ROOT/(P+'small_spectrum_raw.npz'),energies=w,tau=tau,correlator=C,means=means,H=H,E=electric,J=J)
    (ROOT/(P+'small_spectrum_result.json')).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)
    assert commutator_error<1e-12
    assert abs(C[0,0,0]-C[0,0,4])>.001


if __name__=='__main__':main()
