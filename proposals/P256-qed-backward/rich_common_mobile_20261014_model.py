"""Entire mobile scalar-colour/Gauss-frame, Gaussian field and full supplier law.

Two fixed translation cells are a finite CM regulator, not free-space motion.
No old producer is executed. Retained pure Gaussian and frame constructors only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Any
import numpy as np
from scipy import sparse as sp
from rich_frame_photon_20261014_model import Parameters, cube, coulomb_kernel, mode_values, physical_frame


@dataclass(frozen=True)
class MobileParameters:
    base: Parameters = field(default_factory=lambda: Parameters(grid=11, wall=2.55))
    centres: tuple = ((-0.65, -0.20, 0.12), (0.60, 0.28, -0.16))
    cm_hopping: float = 0.48
    core_mass: float = 1.0


def local_geometry(p: MobileParameters) -> dict[str, Any]:
    f = physical_frame()
    n = np.diag(f['N'])
    parity = np.diag((-1.) ** np.rint(f['N']))
    sources = [np.kron(c, np.eye(4)) for c in f['source']]
    sources += [np.kron(parity, c) for c in f['source']]
    numbers = [np.kron(n, np.eye(4)), np.kron(np.eye(4), n)]
    spins = [[np.kron(s, np.eye(4)) for s in f['S']],
             [np.kron(np.eye(4), s) for s in f['S']]]
    return dict(frame=f, sources=sources, numbers=numbers, spins=spins,
                N=np.real(np.diag(numbers[0] + numbers[1])))


def angular_tensor(centre, p: Parameters):
    """q integral Gaussian(y) y cross u_a(R+y) dy, q=-1."""
    r = np.asarray(centre)
    s2, l2 = p.core_width ** 2, p.ell ** 2
    mean = -s2 * r / (l2 + s2)
    v = s2 * l2 / (l2 + s2)
    factor = (l2 / (l2 + s2)) ** 1.5 * np.exp(-np.dot(r, r) / (2 * (l2 + s2)))
    return np.column_stack([-factor * ((2*v + np.dot(mean, r + mean))*a
                            - (r + mean)*np.dot(mean, a))
                            for a in (np.array([0., 0., 1.]), np.array([0., 1., 0.]))])


def line_form(start, end, wavevector, width):
    nodes, weights = np.polynomial.legendre.leggauss(40)
    points = np.asarray(start) + (nodes[:, None] + 1) * (np.asarray(end)-start) / 2
    return np.dot(weights / 2, np.exp(-1j * points @ wavevector)) * np.exp(-width**2*np.dot(wavevector,wavevector)/2)


def line_modes(start, end, ell, width):
    nodes, weights = np.polynomial.legendre.leggauss(40)
    delta = np.asarray(end)-start
    points = np.asarray(start)+(nodes[:,None]+1)*delta/2
    return np.einsum('aik,k,i->a', mode_values(points, ell, width), delta, weights/2)


class Mobile:
    """Order: field pair, complete physical mobile N sector, all64 supplier."""
    def __init__(self, p: MobileParameters, occupation=1, links=True, metric=True, coupling=True):
        self.p, self.occupation, self.links, self.metric = p, occupation, links, metric
        self.coupling = coupling
        b = p.base
        self.geom = local_geometry(p)
        self.ix = np.flatnonzero(abs(self.geom['N']-occupation)<1e-8)
        self.ns = len(self.ix)
        sub = lambda x: np.asarray(x)[np.ix_(self.ix,self.ix)]
        self.number = [sub(x) for x in self.geom['numbers']]
        self.spin = [[sub(x) for x in s] for s in self.geom['spins']]
        self.cube = cube(b)
        n = b.grid
        self.dx = 2*b.wall/(n+1)
        axis = np.linspace(-b.wall+self.dx,b.wall-self.dx,n)
        aa,bb=np.meshgrid(axis,axis,indexing='ij')
        self.coordinates=np.stack([aa.ravel(),bb.ravel()])
        self.nfield=n*n
        self.internal=self.ns*64
        self.size=self.nfield*self.internal
        self.I=sp.eye(self.size,format='csr',dtype=complex)
        self.If=sp.eye(self.nfield,format='csr')
        self.Im=sp.eye(self.ns,format='csr')
        self.I64=sp.eye(64,format='csr')
        ident=sp.eye(n,format='csr')
        lap=sp.diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='csr')/self.dx**2
        mom=sp.diags([1j*np.ones(n-1),-1j*np.ones(n-1)],[-1,1],format='csr')/(2*self.dx)
        self.mode_norm=np.pi**1.5*b.ell**5
        self.mass=b.epsilon*self.mode_norm
        self.E=sp.kron((sp.kron(lap,ident)+sp.kron(ident,lap))/(2*self.mass),sp.eye(self.internal),format='csr')
        self.A=[self.fop(sp.diags(x)) for x in self.coordinates]
        self.P=[self.fop(x) for x in (sp.kron(mom,ident),sp.kron(ident,mom))]
        self.N=[self.mop(x) for x in self.number]
        self.S=[self.mop(sum(s[a] for s in self.spin)) for a in range(3)]
        self.X=[sum(p.centres[c][a]*self.N[c] for c in range(2)) for a in range(3)]
        self.tensor=[angular_tensor(r,b) * int(coupling) for r in p.centres]
        self.ic=1/b.common_inertia if metric else 0.
        self.ir=1/b.relative_inertia
        self.cmphase=-line_modes(np.asarray(p.centres[0]),np.asarray(p.centres[1]),b.ell,b.core_width)
        rawhop=sum(self.geom['sources'][2+a]@self.geom['sources'][a].conj().T for a in range(2))
        self.cmhop=sp.csr_matrix(sub(rawhop))
        self.HL_values,self.charge_data=self.longitudinal()
        self.HL=sp.kron(self.If,sp.diags(self.HL_values.ravel()),format='csr')
        self.Hbody=self.body()
        doublon=np.sum(self.cube['density']==2,axis=1)
        self.Hdiag=self.mop(p.core_mass*occupation*np.eye(self.ns))+sp.kron(self.If,sp.kron(self.Im,sp.diags(b.hubbard*doublon)),format='csr')
        self.H=(self.E+self.HL+self.Hdiag+self.Hbody+self.link_derivative(())).tocsr()
        self.J=[(-self.link_derivative((a,))+self.body_current(a)).tocsr() for a in range(2)]
        self.K=[[ (self.link_derivative((a,c))+self.body_contact(a,c)).tocsr() for c in range(2)]for a in range(2)]

    def mop(self,x):
        return sp.kron(self.If,sp.kron(sp.csr_matrix(x),self.I64),format='csr')

    def fop(self,x):
        return sp.kron(x,sp.eye(self.internal),format='csr')

    def longitudinal(self):
        b=self.p.base
        pos=np.vstack([self.cube['sites'],self.cube['sites'],np.asarray(self.p.centres)])
        widths=np.array([b.supplier_width]*8+[b.ion_width]*8+[b.core_width]*2)
        charge=np.empty((self.ns,64,18))
        for m in range(self.ns):
            charge[m]=np.hstack([self.cube['density'],np.full((64,8),-.25),np.tile([-self.number[c][m,m].real for c in range(2)],(64,1))])
        kernel=coulomb_kernel(pos,widths)/(4*np.pi*b.epsilon)
        return np.einsum('mbi,ij,mbj->mb',charge,kernel,charge)/2,dict(positions=pos,widths=widths,charges=charge,kernel=kernel)

    def body(self,shift=(0.,0.)):
        b=self.p.base
        result=sp.csr_matrix((self.size,self.size),dtype=complex)
        for c in range(2):
            s2=sum(s@s for s in self.spin[c])
            result+=self.mop((self.ic+1/b.frame_inertia)*s2/2)
            d=self.tensor[c]@(self.coordinates+np.asarray(shift)[:,None])
            for a in range(3):
                result-=self.ic*sp.kron(sp.diags(d[a]),sp.kron(sp.csr_matrix(self.number[c]@self.spin[c][a]),self.I64),format='csr')
            result+=(self.ic+self.ir)/2*sp.kron(sp.diags(np.sum(d*d,axis=0)),sp.kron(sp.csr_matrix(self.number[c]@self.number[c]),self.I64),format='csr')
        return result.tocsr()

    def body_current(self,a):
        result=sp.csr_matrix((self.size,self.size),dtype=complex)
        for c in range(2):
            d=self.tensor[c]
            result+=self.ic*self.mop(sum(d[i,a]*self.number[c]@self.spin[c][i] for i in range(3)))
            result-=(self.ic+self.ir)*sp.kron(sp.diags((d[:,a]@d)@self.coordinates),sp.kron(sp.csr_matrix(self.number[c]@self.number[c]),self.I64),format='csr')
        return result

    def body_contact(self,a,b):
        return sum((self.ic+self.ir)*np.dot(d[:,a],d[:,b])*self.mop(n@n) for d,n in zip(self.tensor,self.number))

    def link_terms(self):
        yield self.cmphase*int(self.coupling),self.cmhop,True,self.p.cm_hopping
        for w,h in zip(self.cube['phase'],self.cube['hops']):
            yield w*int(self.coupling),h,False,self.p.base.hopping

    def link_derivative(self,axes,shift=(0.,0.),uniform=(0.,0.,0.)):
        result=sp.csr_matrix((self.size,self.size),dtype=complex)
        if not self.links:return result
        for index,(w,hop,iscm,t) in enumerate(self.link_terms()):
            if iscm:
                delta=-(np.asarray(self.p.centres[1])-self.p.centres[0])
            else:
                i,j=self.cube['edges'][index-1]
                delta=self.cube['sites'][j]-self.cube['sites'][i]
            coeff=-t*np.exp(1j*(w@(self.coordinates+np.asarray(shift)[:,None])+np.dot(delta,uniform)))
            for a in axes:coeff*=1j*w[a]
            matter=sp.kron(hop,self.I64) if iscm else sp.kron(self.Im,hop)
            forward=sp.kron(sp.diags(coeff),matter,format='csr')
            result+=forward+forward.getH()
        return result

    def shifted(self,shift=(0.,0.),uniform=(0.,0.,0.)):
        return self.E+self.HL+self.Hdiag+self.body(shift)+self.link_derivative((),shift,uniform)

    def source_to(self,target,index):
        c=self.geom['sources'][index][np.ix_(target.ix,self.ix)]
        return sp.kron(self.If,sp.kron(sp.csr_matrix(c),self.I64),format='csr')

    def density_fourier(self,k):
        k=np.asarray(k)
        d=self.charge_data
        form=np.exp(-1j*d['positions']@k-d['widths']**2*np.dot(k,k)/2)
        values=d['charges']@form
        return sp.kron(self.If,sp.diags(values.ravel()),format='csr')

    def cm_current(self):
        """Uniform physical A derivative of the core link alone, charge=-1."""
        if not self.links:
            return [sp.csr_matrix((self.size,self.size),dtype=complex) for _ in range(3)]
        phase=self.cmphase*int(self.coupling)
        forward=sp.kron(sp.diags(-self.p.cm_hopping*np.exp(1j*(phase@self.coordinates))),
                        sp.kron(self.cmhop,self.I64),format='csr')
        flow=-1j*(forward-forward.getH())
        delta=np.asarray(self.p.centres[1])-self.p.centres[0]
        return [-delta[a]*flow for a in range(3)]

    def current_fourier(self,k):
        k=np.asarray(k)
        b=self.p.base
        result=[sp.csr_matrix((self.size,self.size),dtype=complex) for _ in range(3)]
        if self.links:
            terms=list(self.link_terms())
            for index,(w,hop,iscm,t) in enumerate(terms):
                if iscm:
                    start,end=np.asarray(self.p.centres); width=b.core_width; charge=-1.
                else:
                    i,j=self.cube['edges'][index-1];start,end=self.cube['sites'][[i,j]];width=b.supplier_width;charge=1.
                delta=end-start
                form=line_form(start,end,k,width)
                matter=sp.kron(hop,self.I64) if iscm else sp.kron(self.Im,hop)
                forward=sp.kron(sp.diags(-t*np.exp(1j*(w@self.coordinates))),matter,format='csr')
                flow=-1j*(forward-forward.getH())
                for a in range(3):result[a]+=charge*delta[a]*form*flow
        for c in range(2):
            d=self.tensor[c]@self.coordinates
            omega=[self.ic*self.mop(self.number[c]@self.spin[c][a])-(self.ic+self.ir)*sp.kron(sp.diags(d[a]),sp.kron(sp.csr_matrix(self.number[c]@self.number[c]),self.I64),format='csr') for a in range(3)]
            coeff=1j*b.core_width**2*np.exp(-1j*np.dot(self.p.centres[c],k)-b.core_width**2*np.dot(k,k)/2)
            for a in range(3):
                j,l=(a+1)%3,(a+2)%3
                result[a]+=coeff*(omega[j]*k[l]-omega[l]*k[j])
        return [x.tocsr() for x in result]
