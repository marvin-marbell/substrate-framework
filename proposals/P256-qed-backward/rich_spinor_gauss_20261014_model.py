"""Literal integer-electric K6 octahedral molecule, CAR and native 4-state orientor.
Energy truncation P(sum E_edge^2<=cutoff) precedes every operator, not spin projection.
"""
from dataclasses import dataclass, asdict
from itertools import combinations, product
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu
from rich_spinor_20261014_model import geometry, occupation, operation, line_profile

LAW='octahedral-K6-integer-electric-native-orientor-v1-20261014'
@dataclass(frozen=True)
class Parameters:
    cutoff:int=4
    p_gap:float=1.4
    d_gap:float=3.2
    rotor_gap:float=2.0
    orientation:float=.85
    rotor_spin:float=1.1
    electric_inertia:float=3.0
    charging:float=2.3
    anchor:int=0
    flux_reference:float=0.0

def electric_ball(cutoff,links):
    """All integer vectors of literal Euclidean electric energy cutoff."""
    vectors=[np.zeros(links,dtype=np.int16)]
    for count in range(1,min(cutoff,links)+1):
        for support in combinations(range(links),count):
            for magnitudes in product(range(1,int(np.sqrt(cutoff))+1),repeat=count):
                if sum(x*x for x in magnitudes)>cutoff: continue
                for signs in product((-1,1),repeat=count):
                    e=np.zeros(links,dtype=np.int16)
                    e[list(support)]=np.array(magnitudes)*np.array(signs)
                    vectors.append(e)
    return np.array(vectors)

class Sector:
    def __init__(self,p,number,electric=None):
        self.p,self.number=p,number; self.g=geometry(); self.r=self.g['positions']
        self.edges=list(combinations(range(6),2)); self.B=np.zeros((6,15),int)
        for l,(a,b) in enumerate(self.edges): self.B[a,l]=1;self.B[b,l]=-1
        self.all_e=electric_ball(p.cutoff,15) if electric is None else electric
        background=np.zeros(6,int); background[p.anchor]=number; self.background=background
        divergences=self.all_e@self.B.T
        by_density={}
        for ei,div in enumerate(divergences):
            n=background-div
            if np.min(n)>=0 and np.max(n)<=2 and np.sum(n)==number:
                by_density.setdefault(tuple(n),[]).append(ei)
        self.bits=[]; self.eindex=[]
        for bits in occupation(number):
            n=tuple(((int(bits)>>(2*i))&1)+((int(bits)>>(2*i+1))&1) for i in range(6))
            for ei in by_density.get(n,[]): self.bits.append(int(bits)); self.eindex.append(ei)
        self.bits=np.array(self.bits,dtype=np.int64); self.eindex=np.array(self.eindex,int)
        self.e=self.all_e[self.eindex]; self.d0=len(self.bits); self.dimension=self.d0*4
        self.lookup={(int(b),self.e[i].tobytes()):i for i,b in enumerate(self.bits)}
        self.orbital=p.p_gap*self.g['Pp']+p.d_gap*self.g['Pd']
        diag=np.array([sum(self.orbital[a,a]*(((int(b)>>(2*a))&1)+((int(b)>>(2*a+1))&1)) for a in range(6)) for b in self.bits])
        diag+=np.sum(self.e**2,axis=1)/(2*p.electric_inertia)+p.charging*number*(number-1)/2
        h=sparse.kron(sparse.diags(diag.astype(complex)),sparse.eye(4),format='csr')
        h+=sparse.kron(sparse.eye(self.d0),sparse.diags([0,p.rotor_gap,p.rotor_gap,p.rotor_gap]),format='csr')
        self.transfers=[]
        for l,(a,b) in enumerate(self.edges):
            rows=[];cols=[];data=[]
            for col,bits in enumerate(self.bits):
                newe=self.e[col].copy(); newe[l]-=1
                for spin in (0,1):
                    rem=operation(int(bits),2*b+spin,False)
                    add=None if rem is None else operation(rem[0],2*a+spin,True)
                    if add is None: continue
                    row=self.lookup.get((add[0],newe.tobytes()))
                    if row is not None: rows.append(row);cols.append(col);data.append(self.orbital[a,b]*rem[1]*add[1])
            t=sparse.kron(sparse.csr_matrix((data,(rows,cols)),shape=(self.d0,self.d0)),sparse.eye(4),format='csr')
            self.transfers.append(t); h+=t+t.getH()
        self.spin=[]; self.position=[]
        for axis in range(3):
            rows=[];cols=[];data=[]
            for col,bits in enumerate(self.bits):
                for site in range(6):
                    for s1 in (0,1):
                        rem=operation(int(bits),2*site+s1,False)
                        if rem is None: continue
                        for s2 in (0,1):
                            val=self.g['S'][axis,s2,s1]
                            if not abs(val):continue
                            add=operation(rem[0],2*site+s2,True)
                            if add is not None:
                                row=self.lookup[(add[0],self.e[col].tobytes())]
                                rows.append(row);cols.append(col);data.append(rem[1]*add[1]*val)
            sf=sparse.csr_matrix((data,(rows,cols)),shape=(self.d0,self.d0))
            pos=np.array([sum(self.r[a,axis]*(((int(bits)>>(2*a))&1)+((int(bits)>>(2*a+1))&1)) for a in range(6)) for bits in self.bits])
            self.spin.append(sparse.kron(sf,sparse.eye(4),format='csr'))
            self.position.append(sparse.kron(sparse.diags(pos),sparse.eye(4),format='csr'))
            h+=p.rotor_spin*sparse.kron(sf,self.g['Ln'][axis],format='csr')
            h+=p.orientation*sparse.kron(sparse.diags(pos),self.g['n'][axis],format='csr')
        self.H=h.tocsr(); self.currents=[-1j*(t-t.getH()) for t in self.transfers]
        self.contacts=[-(t+t.getH()) for t in self.transfers]
        self.rho=np.array([[((int(bits)>>(2*a))&1)+((int(bits)>>(2*a+1))&1)-background[a] for a in range(6)] for bits in self.bits])
        # EXTERNAL loop-restoring reference, never an induced magnetic mechanism.
        if p.flux_reference:
            for triangle in combinations(range(6),3):
                u=self.wilson(list(triangle),raw=True)
                self.H+=p.flux_reference*(sparse.eye(self.dimension)-(u+u.getH())/2)

    def solve(self,count=6):
        e,v=eigsh(self.H,k=min(count,self.dimension-2),which='SA',tol=1e-11,v0=np.linspace(.5,1.5,self.dimension).astype(complex))
        order=np.argsort(e);e=e[order];v=v[:,order]
        start=0
        while start<len(e):
            end=start+1
            while end<len(e) and abs(e[end]-e[start])<1e-8:end+=1
            block,_=np.linalg.qr(v[:,start:end]); _,u=np.linalg.eigh(block.conj().T@(self.spin[2]@block));v[:,start:end]=block@u
            start=end
        self.energies,self.vectors=e,v
        return e,v

    def profiles(self,q,eps):
        return np.array([line_profile(np.array(q),np.array(eps),self.r[a],self.r[b]) for a,b in self.edges])

    def current(self,profile):
        return sum((f*j for f,j in zip(profile,self.currents)),sparse.csr_matrix(self.H.shape,dtype=complex))

    def contact(self,outgoing,incoming):
        return sum((a.conjugate()*b*c for a,b,c in zip(outgoing,incoming,self.contacts)),sparse.csr_matrix(self.H.shape,dtype=complex))

    def magnetic(self,axis):
        area=np.array([np.cross(self.r[b],self.r[a])[axis]/2 for a,b in self.edges])
        return self.current(area)

    def wilson(self,vertices,raw=False):
        shift=np.zeros(15,np.int16)
        for a,b in zip(vertices,vertices[1:]+vertices[:1]):
            l=self.edges.index((min(a,b),max(a,b)));shift[l]+=1 if a<b else -1
        rows=[];cols=[]
        for col,bits in enumerate(self.bits):
            row=self.lookup.get((int(bits),(self.e[col]+shift).tobytes()))
            if row is not None:rows.append(row);cols.append(col)
        u=sparse.kron(sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(self.d0,self.d0)),sparse.eye(4),format='csr')
        return u if raw else (u-u.getH())/(2j)

    def constraint(self):
        gauss=np.max(np.abs(self.e@self.B.T+self.rho));continuity=0.
        for a in range(6):
            n=sparse.diags(np.repeat(self.rho[:,a],4));err=1j*(self.H@n-n@self.H)
            for l,j in enumerate(self.currents):err-=self.B[a,l]*j
            if err.nnz:continuity=max(continuity,float(np.max(np.abs(err.data))))
        return {'gauss_error':float(gauss),'continuity_error':continuity,'integer_electric_error':float(np.max(np.abs(self.e-np.rint(self.e))))}

    def state_data(self,g):
        prob=np.sum(np.abs(g.reshape(self.d0,4))**2,axis=1)
        wave=np.zeros((48,len(self.all_e)),complex)
        for i,b in enumerate(self.bits):
            site=int(b).bit_length()-1
            if self.number==1: wave[site*4:site*4+4,self.eindex[i]]=g[4*i:4*i+4]
        gram=wave@wave.conj().T;val=np.linalg.eigvalsh(gram);val=val[val>1e-14]
        return {'electric_energy':float(prob@np.sum(self.e**2,axis=1)/(2*self.p.electric_inertia)),
                'cutoff_shell_weight':float(prob@(np.sum(self.e**2,axis=1)==self.p.cutoff)),
                'matter_field_entropy':float(-np.sum(val*np.log(val))),
                'density':(prob@(self.rho+self.background)).tolist(),
                'spin':[float(np.vdot(g,s@g).real) for s in self.spin],
                'orbital_source_moment':[float(np.vdot(g,self.magnetic(a)@g).real) for a in range(3)]}

def charged_source(origin,destination,mode,creation):
    """CAR with physical anchor countercharge and direct anchor/site electric string."""
    site=mode//2;shift=np.zeros(15,np.int16);amount=1 if creation else -1
    if site!=origin.p.anchor:
        a,b=origin.p.anchor,site;l=origin.edges.index((min(a,b),max(a,b)))
        shift[l]=amount*(1 if a<b else -1)
    rows=[];cols=[];data=[];outside=0
    for col,bits in enumerate(origin.bits):
        result=operation(int(bits),mode,creation)
        if result is None:continue
        row=destination.lookup.get((result[0],(origin.e[col]+shift).tobytes()))
        if row is None:outside+=1;continue
        rows.append(row);cols.append(col);data.append(result[1])
    op=sparse.kron(sparse.csr_matrix((data,(rows,cols)),shape=(destination.d0,origin.d0)),sparse.eye(4),format='csr')
    return op,{'string_shift':shift.tolist(),'cutoff_excluded':outside,'nnz':op.nnz}

def full_resolvent(h,z,rhs):
    return splu((z*sparse.eye(h.shape[0],format='csc')-h).tocsc()).solve(np.asarray(rhs,dtype=complex))

def retarded(sector,g,e,ops,z):
    sources=np.column_stack([o@g-np.vdot(g,o@g)*g for o in ops]);dh=sector.H-e*sparse.eye(sector.dimension)
    return sources.conj().T@full_resolvent(dh,z,sources)-(sources.conj().T@full_resolvent(-dh,z,sources)).T

def scattering(sector,g,e,qin,qout,omega,eta):
    """Both helicities at both ends, full-H virtual states and actual contact."""
    def polarizations(q):
        q=np.array(q,float);n=q/np.linalg.norm(q);a=np.cross(n,[0,1,0]);a/=np.linalg.norm(a);b=np.cross(n,a)
        return [(a+1j*b)/np.sqrt(2),(a-1j*b)/np.sqrt(2)]
    pin=[sector.profiles(qin,ep) for ep in polarizations(qin)];pout=[sector.profiles(qout,ep) for ep in polarizations(qout)]
    jin=[sector.current(p) for p in pin];jout=[sector.current(p).getH() for p in pout]
    incoming=np.column_stack([j@g for j in jin]);outgoing=np.column_stack([j@g for j in jout])
    plus=full_resolvent(sector.H,e+omega+1j*eta,incoming);minus=full_resolvent(sector.H,e-omega+1j*eta,outgoing)
    amp=np.zeros((2,2),complex);c=np.zeros_like(amp)
    for a in range(2):
        for b in range(2):
            c[a,b]=np.vdot(g,sector.contact(pout[a],pin[b])@g)
            amp[a,b]=c[a,b]+np.vdot(g,jout[a]@plus[:,b])+np.vdot(g,jin[b]@minus[:,a])
    return amp,c
