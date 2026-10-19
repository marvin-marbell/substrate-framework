"""Exact-Gauss projected singlet pair state and actual full-H image dressing.
No configuration-list or tensor truncation defines this amplitude. Imported geometry
and primitive CAR hops are pure frozen definitions; arrays here are fresh parameters.
"""
from itertools import combinations
import numpy as np
from numba import njit
from rich_periodic_photon_20261015_model import POSITIONS, EDGES, INCIDENCE, loops

ARCS = np.asarray(EDGES, dtype=np.int64)
LOOPS = np.asarray(loops(), dtype=np.int64)
PAIRS = np.asarray([(e, next(f for f,(c,d,n) in enumerate(EDGES) if c==b and d==a and n==mu))
                    for e,(a,b,mu) in enumerate(EDGES) if a<b], dtype=np.int64)


def pair_table(g=0.45, eta=0.0):
    """Spin singlet det F(up,down), interleaved CAR order, all 4900 matter states.
    eta=0 is the closed-shell Slater determinant; eta>0 adds unoccupied pairs.
    Four negative-energy cube orbitals are a unique cubic-symmetric filled shell.
    """
    momenta=np.array([(x,y,z) for x in range(2) for y in range(2) for z in range(2)])
    phi=(-1.0)**(POSITIONS@momenta.T)/np.sqrt(8)
    occupied=np.sum(momenta,axis=1)<=1
    F=(phi*np.where(occupied,1.0,eta))@phi.T
    table=np.zeros(65536)
    for up in combinations(range(8),4):
        for down in combinations(range(8),4):
            bits=sum(1<<(2*v) for v in up)+sum(1<<(2*v+1) for v in down)
            parity=sum(d<u for u in up for d in down)
            D=len(set(up)&set(down))
            value=np.linalg.det(F[np.ix_(up,down)])
            if abs(value)<1e-13:value=0.0
            table[bits]=(-1)**parity*value*np.exp(-g*D)
    table/=np.max(np.abs(table))
    return table


@njit(cache=True)
def charge(bits):
    rho=np.empty(8,np.int64)
    for v in range(8):rho[v]=((bits>>(2*v))&1)+((bits>>(2*v+1))&1)-1
    return rho


@njit(cache=True)
def field_weight(E,a,b):
    total=0.0
    for e in range(24):total+=a*E[e]*E[e]
    for k in range(12):
        i,j=PAIRS[k]
        total+=b*(E[i]+E[j])**2
    return np.exp(-total)


@njit(cache=True)
def base(bits,E,table,a,b):
    return table[bits]*field_weight(E,a,b)


@njit(cache=True)
def diagonal(bits,E):
    value=0.0
    for e in range(24):value+=E[e]*E[e]/32.0
    for r in charge(bits):value+=2*r*r
    return value


@njit(cache=True)
def hop(bits,e,direction,spin):
    a,b,_=ARCS[e]
    if direction<0:a,b=b,a
    p,q=2*a+spin,2*b+spin
    if not ((bits>>p)&1) or ((bits>>q)&1):return bits,0.0
    after=bits^(1<<p)
    parity=0
    for k in range(p):parity+=(bits>>k)&1
    for k in range(q):parity+=(after>>k)&1
    return after|(1<<q),(-1.0 if parity%2==0 else 1.0)


@njit(cache=True)
def hbase(bits,E,table,a,b):
    value=diagonal(bits,E)*base(bits,E,table,a,b)
    for e in range(24):
        for direction in (-1,1):
            if abs(E[e]+direction)>1:continue
            for spin in range(2):
                target,h=hop(bits,e,direction,spin)
                if h==0:continue
                E[e]+=direction
                value+=h*base(target,E,table,a,b)
                E[e]-=direction
    return value


@njit(cache=True)
def amplitude(bits,E,table,a,b,tau=0.0,shift=-10.0):
    value=base(bits,E,table,a,b)
    if tau!=0:value-=tau*(hbase(bits,E,table,a,b)-shift*value)
    return value


def physical_amplitude(bits,E,table,a,b,tau=0.0,shift=-10.0):
    E=np.asarray(E,dtype=np.int64)
    up=sum((bits>>(2*v))&1 for v in range(8))
    down=sum((bits>>(2*v+1))&1 for v in range(8))
    if up!=4 or down!=4 or np.any(abs(E)>1) or np.any(INCIDENCE@E+charge(bits)):
        return 0.0
    return float(amplitude(bits,E.copy(),table,a,b,tau,shift))


@njit(cache=True)
def guide(bits,E,table,a,b,tau,shift,epsilon,g):
    psi=amplitude(bits,E,table,a,b,tau,shift)
    D=0
    for v in range(8):D+=((bits>>(2*v))&1)*((bits>>(2*v+1))&1)
    envelope=np.exp(-g*D)*field_weight(E,a,b)
    return psi*psi+epsilon*epsilon*envelope*envelope


@njit(cache=True)
def sample(seed,n,burn,thin,table,a,b,tau,shift,epsilon,g):
    """Fixed symmetric proposals with rejection; guiding density strictly positive.
    Primitive CAR hops, opposite-spin swaps, plaquettes AND torus windings.
    Invalid moves are self transitions, so no degree correction is missing.
    """
    np.random.seed(seed)
    bits=0
    for v in range(8):bits+=1<<(2*v+(v%2))
    E=np.zeros(24,np.int64)
    weight=guide(bits,E,table,a,b,tau,shift,epsilon,g)
    bs=np.empty(n,np.int64);Es=np.empty((n,24),np.int64)
    accepted=np.zeros(3,np.int64);proposed=np.zeros(3,np.int64)
    for step in range(burn+n*thin):
        newbits=bits;newE=E.copy();kind=np.random.randint(10)
        if kind<4:
            kind=0;e=np.random.randint(24);d=2*np.random.randint(2)-1;s=np.random.randint(2)
            if abs(E[e]+d)<=1:
                newbits,h=hop(bits,e,d,s)
                if h!=0:newE[e]+=d
        elif kind<9:
            kind=1;k=np.random.randint(len(LOOPS));d=2*np.random.randint(2)-1
            valid=True
            for e in range(24):
                newE[e]+=d*LOOPS[k,e]
                if abs(newE[e])>1:valid=False
            if not valid:newE=E.copy()
        else:
            kind=2;v=np.random.randint(8);w=np.random.randint(8)
            nv=(bits>>(2*v))&3;nw=(bits>>(2*w))&3
            if (nv==1 and nw==2) or (nv==2 and nw==1):newbits=bits^(3<<(2*v))^(3<<(2*w))
        proposed[kind]+=1
        newweight=guide(newbits,newE,table,a,b,tau,shift,epsilon,g)
        if np.random.random()*weight<newweight:
            if newbits!=bits or np.any(newE!=E):accepted[kind]+=1
            bits=newbits;E=newE;weight=newweight
        if step>=burn and (step-burn)%thin==0:
            k=(step-burn)//thin;bs[k]=bits;Es[k]=E
    return bs,Es,accepted,proposed


@njit(cache=True)
def vectors(bits,E,table,a,b,tau,shift,coeff):
    """[psi,F1 psi,F2 psi,J1 psi,J2 psi], F=cE/16, J=-i[H,cE]."""
    psi=amplitude(bits,E,table,a,b,tau,shift)
    out=np.zeros(5,np.complex128);out[0]=psi
    for p in range(2):
        for e in range(24):out[1+p]+=coeff[p,e]*E[e]*psi/16.0
    for e in range(24):
        for d in (-1,1):
            if abs(E[e]+d)>1:continue
            for s in range(2):
                target,h=hop(bits,e,d,s)
                if h==0:continue
                E[e]+=d;other=amplitude(target,E,table,a,b,tau,shift);E[e]-=d
                for p in range(2):out[3+p]-=1j*h*d*coeff[p,e]*other
    return out

@njit(cache=True)
def observe(bs,Es,table,a,b,tau,shift,epsilon,g,coeff,fullresponse):
    n=len(bs);raw=np.zeros((n,10),np.float64)
    grams=np.zeros((n,5,5),np.complex128)
    moments=np.zeros((n,5,5),np.complex128)
    contacts=np.zeros((n,2,2),np.float64)
    for k in range(n):
        bits=bs[k];E=Es[k].copy()
        weight=guide(bits,E,table,a,b,tau,shift,epsilon,g)
        v=vectors(bits,E,table,a,b,tau,shift,coeff)
        Hv=diagonal(bits,E)*v
        contact=np.zeros((2,2))
        for e in range(24):
            for d in (-1,1):
                if abs(E[e]+d)>1:continue
                for s in range(2):
                    target,h=hop(bits,e,d,s)
                    if h==0:continue
                    E[e]+=d
                    if fullresponse:other=vectors(target,E,table,a,b,tau,shift,coeff)
                    else:
                        other=np.zeros(5,np.complex128)
                        other[0]=amplitude(target,E,table,a,b,tau,shift)
                    E[e]-=d
                    Hv+=h*other
                    for p in range(2):
                        for q in range(2):contact[p,q]-=h*coeff[p,e]*coeff[q,e]*other[0].real*v[0].real/weight
        raw[k,0]=abs(v[0])**2/weight
        raw[k,1]=(v[0].conjugate()*Hv[0]).real/weight
        raw[k,2]=abs(Hv[0])**2/weight
        D=0
        for z in range(8):D+=((bits>>(2*z))&1)*((bits>>(2*z+1))&1)
        raw[k,3]=D*raw[k,0]
        raw[k,4]=np.sum(E*E)*raw[k,0]
        raw[k,5]=np.sum(E)*raw[k,0]
        raw[k,6]=np.sum(E)**2*raw[k,0]
        raw[k,7]=float(np.max(np.abs(E)))
        raw[k,8]=1.0 if v[0]==0 else 0.0
        raw[k,9]=raw[k,0] if D==4 else 0.0
        grams[k]=np.outer(v.conjugate(),v)/weight
        if fullresponse:moments[k]=np.outer(v.conjugate(),Hv)/weight
        contacts[k]=contact
    return raw,grams,moments,contacts
