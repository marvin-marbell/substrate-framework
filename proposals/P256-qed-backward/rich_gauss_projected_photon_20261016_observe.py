"""Actual selected amplitude symmetry, occupation/field/current and loop coherence."""
import json
from pathlib import Path
import numpy as np
from numba import njit
from scipy.linalg import null_space
import rich_gauss_projected_photon_20261016_state as s
ROOT=Path(__file__).resolve().parent
P='rich_gauss_projected_photon_20261016_'


@njit(cache=True)
def joint(bs,Es,table,a,b,tau,shift,epsilon,g):
    means=np.zeros(56,np.complex128);gram=np.zeros((56,56),np.complex128)
    loopmean=np.zeros(len(s.LOOPS));z=0.0;Dcross=0.0;Dmean=0.0;E2mean=0.0
    for k in range(len(bs)):
        bits=bs[k];E=Es[k].copy();psi=s.amplitude(bits,E,table,a,b,tau,shift)
        guide=s.guide(bits,E,table,a,b,tau,shift,epsilon,g)
        v=np.zeros(56,np.complex128)
        for u in range(8):v[u]=s.charge(bits)[u]*psi
        for e in range(24):v[8+e]=E[e]*psi/16.
        for e in range(24):
            for d in (-1,1):
                if abs(E[e]+d)>1:continue
                for spin in range(2):
                    target,h=s.hop(bits,e,d,spin)
                    if h==0:continue
                    E[e]+=d;other=s.amplitude(target,E,table,a,b,tau,shift);E[e]-=d
                    v[32+e]-=1j*h*d*other
        means+=psi*v/guide
        gram+=np.outer(v.conjugate(),v)/guide
        z+=psi*psi/guide
        D=0
        for u in range(8):D+=((bits>>(2*u))&1)*((bits>>(2*u+1))&1)
        E2=np.sum(E*E);Dmean+=D*psi*psi/guide;E2mean+=E2*psi*psi/guide
        Dcross+=D*E2*psi*psi/guide
        for l in range(len(s.LOOPS)):
            for d in (-1,1):
                newE=E+d*s.LOOPS[l]
                if np.max(np.abs(newE))<=1:loopmean[l]+=.5*psi*s.amplitude(bits,newE,table,a,b,tau,shift)/guide
    return means/z,gram/z,loopmean/z,Dcross/z-Dmean*E2mean/(z*z)


def transform(bits,E,mapping,axis):
    occupied=[]
    for p in range(16):
        if (bits>>p)&1:occupied.append(2*mapping[p//2]+p%2)
    parity=sum(occupied[i]>occupied[j] for i in range(len(occupied)) for j in range(i+1,len(occupied)))
    newbits=sum(1<<p for p in occupied);newE=np.empty(24,np.int64)
    edgeindex={(a,mu):e for e,(a,_,mu) in enumerate(s.EDGES)}
    for e,(a,_,mu) in enumerate(s.EDGES):newE[edgeindex[mapping[a],axis[mu]]]=E[e]
    return newbits,newE,(-1)**parity


def main():
    selected=json.loads((ROOT/(P+'selected_state.json')).read_text());p=selected['parameters']
    data=np.load(ROOT/selected['table_file']);bs=data['bits'];Es=data['E'];table=data['table']
    means,gram,coherence,Dcross=joint(bs,Es,table,p['a'],p['b'],p['tau'],p['shift'],p['epsilon'],p['g'])
    covariance=gram-np.outer(means.conjugate(),means)
    cycles=null_space(s.INCIDENCE.astype(float));projected=cycles.T@covariance[8:32,8:32]@cycles
    positionlookup={tuple(r):i for i,r in enumerate(s.POSITIONS)}
    transforms=[]
    for mu in range(3):
        translation=np.eye(3,dtype=int)[mu]
        transforms.append(('translation'+str(mu),[positionlookup[tuple((r+translation)%2)] for r in s.POSITIONS],[0,1,2]))
    for axis in ([1,0,2],[2,1,0]):
        transforms.append(('axes'+str(axis),[positionlookup[tuple(r[axis])] for r in s.POSITIONS],axis))
    errors={}
    for name,mapping,axis in transforms:
        error=0.;scale=0.
        for bits,E in zip(bs[::157],Es[::157]):
            bits=int(bits);newbits,newE,sign=transform(bits,E,mapping,axis)
            old=s.physical_amplitude(bits,E,table,p['a'],p['b'],p['tau'],p['shift'])
            new=sign*s.physical_amplitude(newbits,newE,table,p['a'],p['b'],p['tau'],p['shift'])
            error=max(error,abs(old-new));scale=max(scale,abs(old))
        errors[name]=dict(max_absolute=error,relative_to_max_amplitude=error/max(1e-30,scale))
    spinerror=0.;spinscale=0.
    for bits,E in zip(bs[::157],Es[::157]):
        bits=int(bits)
        singles=[v for v in range(8) if (bits>>(2*v))&3==2]
        if not singles:continue
        target=bits^(3<<(2*singles[0]));value=0.
        for v in range(8):
            if (target>>(2*v))&3==1:
                precursor=target^(3<<(2*v))
                amp=s.physical_amplitude(precursor,E,table,p['a'],p['b'],p['tau'],p['shift'])
                value+=amp;spinscale=max(spinscale,abs(amp))
        spinerror=max(spinerror,abs(value))
    np.savez_compressed(ROOT/(P+'joint_correlations.npz'),means=means,covariance=covariance,cycles=cycles,
                        cycle_field_covariance=projected,loop_coherence=coherence,loop_vectors=s.LOOPS)
    result=dict(samples=len(bs),symmetry=errors,spin_raising_annihilation_max=spinerror,
                spin_raising_relative=spinerror/max(1e-30,spinscale),cycle_dimension=cycles.shape[1],
                cycle_field_eigenvalues=np.linalg.eigvalsh(projected).real.tolist(),
                charge_field_covariance_norm=float(np.linalg.norm(covariance[:8,8:32])),
                charge_current_covariance_norm=float(np.linalg.norm(covariance[:8,32:])),
                doublon_E2_connected=float(Dcross),
                field_current_covariance_norm=float(np.linalg.norm(covariance[8:32,32:])),
                loop_coherence=[dict(length=int(np.sum(abs(loop))),value=float(value)) for loop,value in zip(s.LOOPS,coherence)],
                limits='Finite-chain connected covariance and coherent Wilson insertions, not time evolution or17 photons. Cube symmetries checked at102 actual sampled raw states including CAR permutation signs; spin-raising annihilation is a state-amplitude singlet check.')
    (ROOT/(P+'correlation_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
