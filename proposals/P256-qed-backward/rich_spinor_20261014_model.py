"""NEW octahedral charged molecule with a dynamic neutral s/p orientor.
No Dirac matrices, direct electron L.S, spin Zeeman term, or photon Hamiltonian.
Units hbar=site radius=elementary charge=1. Electron charge is -N.
"""
from dataclasses import dataclass, asdict
from itertools import combinations
import numpy as np
from scipy.linalg import eigh

LAW = 'octahedron-neutral-orientor-spin-transfer-v1'

@dataclass(frozen=True)
class Parameters:
    p_gap: float = 1.4
    d_gap: float = 3.2
    rotor_gap: float = 2.0
    orientation: float = 0.85
    rotor_spin: float = 1.1
    charging: float = 2.3


def geometry():
    r = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
    s = np.ones((6,1))/np.sqrt(6)
    p = r/np.sqrt(2)
    ps, pp = s@s.T, p@p.T
    pauli = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]], complex)/2
    angular = np.zeros((3,3,3), complex)
    for i,j,k,sign in [(0,1,2,1),(1,2,0,1),(2,0,1,1),(0,2,1,-1),(2,1,0,-1),(1,0,2,-1)]:
        angular[i,j,k] = -1j*sign
    neutral_l = np.zeros((3,4,4), complex)
    neutral_n = np.zeros_like(neutral_l)
    for i in range(3):
        neutral_l[i,1:,1:] = angular[i]
        neutral_n[i,0,i+1] = neutral_n[i,i+1,0] = 1/np.sqrt(3)
    inversion = np.zeros((6,6))
    for a in range(6): inversion[a^1,a] = 1
    return {'positions':r, 's':s, 'p':p, 'Ps':ps, 'Pp':pp,
            'Pd':np.eye(6)-ps-pp, 'S':pauli, 'Lp':angular,
            'Ln':neutral_l, 'n':neutral_n, 'inversion':inversion}


def occupation(number):
    return np.array([sum(1<<a for a in c) for c in combinations(range(12),number)],dtype=np.int64)


def operation(bits,mode,creation):
    occupied=(bits>>mode)&1
    if occupied==creation: return None
    return bits^(1<<mode), (-1)**((bits&((1<<mode)-1)).bit_count())


def lift(one, states):
    lookup={int(b):a for a,b in enumerate(states)}
    matrix=np.zeros((len(states),len(states)),complex)
    for col,b in enumerate(states):
        for origin in range(12):
            removed=operation(int(b),origin,False)
            if removed is None: continue
            for destination in range(12):
                if abs(one[destination,origin])<1e-15: continue
                added=operation(removed[0],destination,True)
                if added is not None:
                    matrix[lookup[added[0]],col]+=removed[1]*added[1]*one[destination,origin]
    return matrix


def parity_fock(states, inversion):
    matrix=np.zeros((len(states),len(states)))
    lookup={int(b):a for a,b in enumerate(states)}
    for col,b in enumerate(states):
        occupied=[a for a in range(12) if (int(b)>>a)&1]
        mapped=[2*((a//2)^1)+a%2 for a in occupied]
        sign=(-1)**sum(mapped[i]>mapped[j] for i in range(len(mapped)) for j in range(i+1,len(mapped)))
        matrix[lookup[sum(1<<a for a in mapped)],col]=sign
    return matrix


def build_sector(params=Parameters(), number=1):
    g=geometry(); states=occupation(number); dim=len(states); eye=np.eye(dim)
    orbital=params.p_gap*g['Pp']+params.d_gap*g['Pd']
    h0=np.kron(orbital,np.eye(2))
    h=np.kron(lift(h0,states),np.eye(4))
    h+=np.kron(eye,np.diag([0,params.rotor_gap,params.rotor_gap,params.rotor_gap]))
    h+=params.charging*number*(number-1)/2*np.eye(dim*4)
    spin=[]; position=[]; angular=[]
    for i in range(3):
        sf=lift(np.kron(np.eye(6),g['S'][i]),states)
        rf=lift(np.kron(np.diag(g['positions'][:,i]),np.eye(2)),states)
        lf=lift(np.kron(g['p']@g['Lp'][i]@g['p'].T,np.eye(2)),states)
        h+=params.orientation*np.kron(rf,g['n'][i])
        h+=params.rotor_spin*np.kron(sf,g['Ln'][i])
        spin.append(np.kron(sf,np.eye(4)))
        position.append(np.kron(rf,np.eye(4)))
        angular.append(np.kron(lf+sf,np.eye(4))+np.kron(eye,g['Ln'][i]))
    parity=np.kron(parity_fock(states,g['inversion']),np.diag([1,-1,-1,-1]))
    return {'H':h,'states':states,'number':number,'charge':-number,'geometry':g,
            'orbital':orbital,'spin':np.array(spin),'position':np.array(position),
            'angular':np.array(angular),'parity':parity,'parameters':asdict(params),'law':LAW}


def line_profile(q,epsilon,r1,r2):
    """Exact straight-link integral of epsilon.exp(i q.r); no midpoint approximation."""
    delta=r1-r2
    return (epsilon@delta)*np.exp(1j*q@((r1+r2)/2))*np.sinc((q@delta)/(2*np.pi))


def current(sector,q,epsilon):
    """J(q,eps)=-dH/dA, gauging EVERY offsite electronic hop, rotor unchanged.
    Complex Fourier vertex has J(q,eps)^dagger=J(-q,conj(eps)).
    """
    r=sector['geometry']['positions']; orbital=sector['orbital']
    one=np.zeros((12,12),complex)
    for a in range(6):
        for b in range(6):
            f=line_profile(np.asarray(q),np.asarray(epsilon),r[a],r[b])
            one[2*a:2*a+2,2*b:2*b+2]=-1j*orbital[a,b]*f*np.eye(2)
    return np.kron(lift(one,sector['states']),np.eye(4))


def contact(sector,qout,epsout,qin,epsin):
    """Actual mixed second derivative for outgoing conjugate/incoming Fourier probes."""
    r=sector['geometry']['positions']; orbital=sector['orbital']
    one=np.zeros((12,12),complex)
    for a in range(6):
        for b in range(6):
            fout=line_profile(np.asarray(qout),np.asarray(epsout),r[a],r[b]).conjugate()
            fin=line_profile(np.asarray(qin),np.asarray(epsin),r[a],r[b])
            one[2*a:2*a+2,2*b:2*b+2]=-orbital[a,b]*fout*fin*np.eye(2)
    return np.kron(lift(one,sector['states']),np.eye(4))


def spin_current(sector,q,epsilon,axis):
    j=current(sector,q,epsilon); s=sector['spin'][axis]
    return (s@j+j@s)/2


def solve(sector):
    """Complete Hermitian spectrum with parity/Jz resolution inside exact degeneracies."""
    e,v=eigh(sector['H'])
    start=0
    while start<len(e):
        end=start+1
        while end<len(e) and abs(e[end]-e[start])<1e-9: end+=1
        block=v[:,start:end]
        resolver=block.conj().T@(10*sector['parity']+sector['angular'][2])@block
        _,u=eigh(resolver); v[:,start:end]=block@u
        start=end
    sector['energies'],sector['vectors']=e,v
    return sector


def scattering(sector,initial,final,qin,epsin,qout,epsout,omega,eta=0.025,virtual=None):
    """Complete Kramers-Heisenberg current/contact; elastic same charged molecule.
    Photons are external normalized probes, not a claimed propagating photon phase.
    """
    e=sector['energies']; v=sector['vectors']; a=v[:,initial]; b=v[:,final]
    jin=current(sector,qin,epsin); jout=current(sector,qout,epsout).conj().T
    c=contact(sector,qout,epsout,qin,epsin)
    indexes=np.arange(len(e)) if virtual is None else np.asarray(virtual)
    u=v[:,indexes]; ei=e[initial]; omegaout=omega+ei-e[final]
    term1=np.sum((b.conj()@jout@u)*(u.conj().T@jin@a)/(ei+omega-e[indexes]+1j*eta))
    term2=np.sum((b.conj()@jin@u)*(u.conj().T@jout@a)/(ei-omegaout-e[indexes]+1j*eta))
    return complex(b.conj()@c@a+term1+term2)


def car_source(origin,destination,mode,creation):
    lookup={int(b):a for a,b in enumerate(destination['states'])}
    matrix=np.zeros((len(destination['states']),len(origin['states'])),complex)
    for col,b in enumerate(origin['states']):
        result=operation(int(b),mode,creation)
        if result is not None: matrix[lookup[result[0]],col]=result[1]
    return np.kron(matrix,np.eye(4))


def charged_spectral(sectors,ground_indices=(0,1)):
    """Spin-resolved 12x12 CAR G about an incoherent occupied N=1 doublet.
    N=0 removal and ALL N=2 addition states retained, not positrons.
    """
    reference=sectors[1]; v=reference['vectors']; e0=reference['energies']
    result={}
    for number,tag,creation in [(2,'addition',True),(0,'removal',False)]:
        dest=sectors[number]; amplitudes=[]
        for ground in ground_indices:
            source=np.column_stack([car_source(reference,dest,a,creation)@v[:,ground] for a in range(12)])
            amplitudes.append(dest['vectors'].conj().T@source)
        result[tag+'_amplitudes']=np.array(amplitudes)
        result[tag+'_energies']=np.array([dest['energies']-e0[a] for a in ground_indices])
    return result


def charged_green(data,omega,eta=0.025):
    """Retarded G_ab: <{c_a(t),c_b^dagger}> with z=omega+i eta."""
    z=omega+1j*eta
    a=data['addition_amplitudes']; b=data['removal_amplitudes']
    plus=np.einsum('gna,gnb,gn->ab',a.conj(),a,1/(z-data['addition_energies']))/a.shape[0]
    minus=np.einsum('gnb,gna,gn->ab',b.conj(),b,1/(z+data['removal_energies']))/b.shape[0]
    return plus+minus
