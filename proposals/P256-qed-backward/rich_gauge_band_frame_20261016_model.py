"""NEW central-sewn scalar-CAR band frame with exact integer tree-Wilson Gauss.

No old Hamiltonian or precomputed evolved output is imported.  Complete angular
multiplets are Galerkin regulators, NOT full RIGHT constraints or spin labels.
"""
from functools import lru_cache
from itertools import product
from typing import Any
import numpy as np
from scipy import sparse as sp
from scipy.linalg import expm

DEFAULTS: dict[str, Any] = dict(k=[.19,.31,.43], radial_nodes=[.45,.9,1.35], radial_mass=2.,
                radial_stiffness=3., radial_centre=.9, material_inertia=3.,
                electric_inertia=4., root_coupling=.7, angular_level=1)


def clean(a):
    a = sp.csr_matrix(a); a.data[abs(a.data)<2e-12] = 0
    a.eliminate_zeros(); return a


def tensor(*values):
    out = sp.csr_matrix([[1.]])
    for a in values: out = sp.kron(out,a,format='csr')
    return clean(out)


def car(mask, mode, create):
    if bool(mask & (1 << mode)) == create: return None
    return mask ^ (1 << mode), (-1)**((mask & ((1 << mode)-1)).bit_count())


def geometry() -> dict[str, Any]:
    sites = np.array(list(product(range(2),repeat=3)))
    lookup = {tuple(s):i for i,s in enumerate(sites)}
    gamma=[]
    for axis in range(3):
        a=np.zeros((8,8),complex)
        for col,s in enumerate(sites):
            d=s.copy();d[axis]^=1
            a[lookup[tuple(d)],col]=(-1)**sum(s[:axis])
        gamma.append(a)
    beta=np.diag((-1.)**sites.sum(axis=1))
    spin=[-1j*gamma[(a+1)%3]@gamma[(a+2)%3]/2 for a in range(3)]
    # Fixed rooted spanning cube tree; each vertex removes its highest set bit.
    parents=[-1]+[i^(1<<(i.bit_length()-1)) for i in range(1,8)]
    paths=np.zeros((8,8),int);paths[:,7]=1
    for s in range(1,8):
        v=s
        while v: paths[s,v-1]=1;v=parents[v]
    # E oriented child -> parent: divergence at child includes +E_child.
    incidence=np.zeros((8,8),int)
    for child in range(1,8):
        incidence[child,child-1]=1;incidence[parents[child],child-1]=-1
    incidence[0,7]=1
    return dict(sites=sites,gamma=gamma,beta=beta,spin=spin,parents=parents,
                paths=paths,incidence=incidence,ions=((1-np.diag(beta))/2).astype(int))


def jmat(j):
    m=np.arange(-j,j+1);z=np.diag(m);plus=np.zeros((len(m),len(m)))
    for n,x in enumerate(m[:-1]):plus[n+1,n]=np.sqrt(j*(j+1)-x*(x+1))
    return [(plus+plus.T)/2,(plus-plus.T)/(2j),z]


@lru_cache(maxsize=8)
def angular(js) -> dict[str, Any]:
    """Actual normalized Peter-Weyl wavefunctions and LEFT differential action."""
    js=tuple(js);maxj=max(js)
    # SU2 alpha length 4pi, gamma length 2pi, cos(beta) Haar; R integer same grid.
    na=max(8,int(4*maxj+6));ng=na;nb=max(5,int(2*maxj+4))
    x,w=np.polynomial.legendre.leggauss(nb)
    points=np.array(list(product(np.arange(na)*4*np.pi/na,np.arccos(x),np.arange(ng)*2*np.pi/ng)))
    weights=np.tile(np.repeat(w/(2*na*ng),ng),na)
    blocks=[];labels=[];L=[[] for _ in range(3)];cas=[]
    for j in js:
        J=jmat(j);d=len(J[0]);D=np.array([expm(-1j*a*J[2])@expm(-1j*b*J[1])@expm(-1j*c*J[2]) for a,b,c in points])
        blocks.append(np.sqrt(d)*D.reshape(len(points),d*d))
        labels.extend((j,m,n) for m in np.arange(-j,j+1) for n in np.arange(-j,j+1))
        for axis in range(3):L[axis].append(sp.kron(-J[axis].T,np.eye(d)))
        cas.extend([j*(j+1)]*(d*d))
    F=np.concatenate(blocks,axis=1)
    # Ordinary active rotation matrices computed from actual fundamental SU2.
    pauli=[2*a for a in jmat(.5)]
    rotations=[]
    for a,b,c in points:
        u=expm(-.5j*a*pauli[2])@expm(-.5j*b*pauli[1])@expm(-.5j*c*pauli[2])
        rotations.append([[np.trace(pauli[i]@u@pauli[j]@u.conj().T).real/2 for j in range(3)] for i in range(3)])
    rotations=np.array(rotations)
    O=[[clean(F.conj().T@(weights[:,None]*rotations[:,i,j,None]*F)) for j in range(3)] for i in range(3)]
    return dict(F=F,weights=weights,points=points,O=O,L=[clean(sp.block_diag(a)) for a in L],
                casimir=sp.diags(cas),labels=labels,gram_error=float(np.linalg.norm(F.conj().T@(weights[:,None]*F)-np.eye(F.shape[1]))))


def multiplication(target, reference, values):
    """Cross-parity source multiplication; quadratures chosen independently."""
    # Evaluate reference and target on a shared sufficiently exact grid.
    js=tuple(sorted(set(x[0] for x in target['labels']+reference['labels'])))
    common=angular(js);F=common['F'];labs=common['labels']
    ti=[labs.index(x) for x in target['labels']];ri=[labs.index(x) for x in reference['labels']]
    return clean(F[:,ti].conj().T@(common['weights'][:,None]*values(common['points'])[:,None]*F[:,ri]))


class Sector:
    def __init__(self,number):
        self.number=number;self.masks=[m for m in range(256) if m.bit_count()==number]
        self.lookup={m:i for i,m in enumerate(self.masks)};self.dimension=len(self.masks)
        self.n=np.array([[(m>>s)&1 for s in range(8)] for m in self.masks])
        geo=geometry();self.q=self.n-geo['ions'];self.electric=self.q@geo['paths']
        self.gauss_error=float(np.max(abs(self.electric@geo['incidence'].T-self.q)))

    def bilinear(self,a):
        rows=[];cols=[];data=[]
        for col,m in enumerate(self.masks):
            for s,t in zip(*np.nonzero(abs(a)>1e-13)):
                first=car(m,int(t),False)
                last=None if first is None else car(first[0],int(s),True)
                if first is not None and last is not None:
                    rows.append(self.lookup[last[0]]);cols.append(col);data.append(a[s,t]*first[1]*last[1])
        return clean(sp.csr_matrix((data,(rows,cols)),shape=(self.dimension,)*2))

    def source(self,reference,mode,create):
        rows=[];cols=[];data=[]
        for col,m in enumerate(reference.masks):
            step=car(m,mode,create)
            if step is not None:rows.append(self.lookup[step[0]]);cols.append(col);data.append(step[1])
        return sp.csr_matrix((data,(rows,cols)),shape=(self.dimension,reference.dimension))


class Model:
    def __init__(self,number,parameters=None):
        self.parameters=p=dict(DEFAULTS if parameters is None else parameters)
        self.sector=s=Sector(number);self.geo=geo=geometry()
        level=p['angular_level'];self.g=angular(tuple(np.arange(number%2/2,level+number%2/2+.1,1)))
        self.material=angular(tuple(range(level+1)))
        self.rnodes=r=np.array(p['radial_nodes']);ng=len(self.g['labels']);nm=len(self.material['labels']);nr=len(r)
        self.shape=(s.dimension,ng,nm,nr);self.dimension=int(np.prod(self.shape))
        fi,gi,mi,ri=[sp.eye(n,format='csr') for n in self.shape]
        self.lift_f=lambda a:tensor(a,gi,mi,ri)
        self.J=[tensor(fi,self.g['L'][a],mi,ri)+tensor(fi,gi,self.material['L'][a],ri) for a in range(3)]
        self.Jg=[tensor(fi,self.g['L'][a],mi,ri) for a in range(3)]
        self.JR=[tensor(fi,gi,self.material['L'][a],ri) for a in range(3)]
        self.r=tensor(fi,gi,mi,sp.diags(r))
        self.E=[self.lift_f(sp.diags(s.electric[:,e])) for e in range(8)]
        self.n=[self.lift_f(sp.diags(s.n[:,v])) for v in range(8)]
        self.HE=self.lift_f(sp.diags(np.sum(s.electric**2,axis=1)/(2*p['electric_inertia'])))
        dx=r[1]-r[0]
        if min(r)<=0 or not np.allclose(np.diff(r),dx):raise ValueError('uniform positive Friedrichs interior nodes')
        kinetic=(2*sp.eye(nr)-sp.eye(nr,k=1)-sp.eye(nr,k=-1))/(2*p['radial_mass']*dx**2)
        potential=sp.diags(p['radial_stiffness']*(r-p['radial_centre'])**2/2+3/(8*p['radial_mass']*r**2))
        self.Hradial=tensor(fi,gi,mi,kinetic+potential)
        self.Hg=tensor(fi,self.g['casimir'],mi,sp.diags(1/(2*p['radial_mass']*r**2)))
        self.HR=tensor(fi,gi,self.material['casimir']/(2*p['material_inertia']),ri)
        self.beta=self.lift_f(s.bilinear(geo['beta']))
        self.Hmass=p['root_coupling']*self.r@self.beta
        self.Hrelative=sp.csr_matrix((self.dimension,)*2,dtype=complex)
        self.relative_terms=[]
        v=-2*np.cos(p['k'])
        for i,a,ell in product(range(3),repeat=3):
            term=v[a]*tensor(s.bilinear(geo['gamma'][i]),self.g['O'][ell][i],self.material['O'][ell][a],ri)
            self.Hrelative+=term;self.relative_terms.append(term)
        self.H=clean(self.HE+self.Hradial+self.Hg+self.HR+self.Hmass+self.Hrelative)
        # ALL r derivatives of potential, cone/frame kinetic, and beta*r.
        radial_force=-p['radial_stiffness']*(r-p['radial_centre'])+3/(4*p['radial_mass']*r**3)
        self.force=tensor(fi,gi,mi,sp.diags(radial_force))+tensor(fi,self.g['casimir'],mi,sp.diags(1/(p['radial_mass']*r**3)))-p['root_coupling']*self.beta
        self.currents=[clean(1j*(self.H@e-e@self.H)) for e in self.E]
        self.torque_g=[clean(1j*(self.H@a-a@self.H)) for a in self.Jg]
        self.torque_R=[clean(1j*(self.H@a-a@self.H)) for a in self.JR]

    def source_from(self,reference,component=0,create=True):
        """B_mu(g)=Dband(g)_mu,s W_s cdag_s; removal complex-conjugate."""
        spin=self.geo['spin']
        def dvalues(points,mode):
            vals=np.array([expm(-1j*a*spin[2])@expm(-1j*b*spin[1])@expm(-1j*c*spin[2]) for a,b,c in points])[:,component,mode]
            return vals if create else vals.conj()
        out=sp.csr_matrix((self.dimension,reference.dimension),dtype=complex)
        for mode in range(8):
            mult=multiplication(self.g,reference.g,lambda points:dvalues(points,mode))
            out+=tensor(self.sector.source(reference.sector,mode,create),mult,sp.eye(len(self.material['labels'])),sp.eye(len(self.rnodes)))
        return clean(out)

    def preparation(self):
        """Declared nonstationary neutral input: staggered ions, Haar g,R, cone wave."""
        if self.sector.number!=4:raise ValueError('preparation requires neutral N4')
        mask=sum(int(n)<<i for i,n in enumerate(self.geo['ions']))
        f=np.eye(self.sector.dimension)[:,self.sector.lookup[mask]]
        g=np.eye(self.shape[1])[:,0];material=np.eye(self.shape[2])[:,0]
        radial=np.exp(-2*(self.rnodes-.9)**2);radial/=np.linalg.norm(radial)
        return np.kron(np.kron(np.kron(f,g),material),radial)
