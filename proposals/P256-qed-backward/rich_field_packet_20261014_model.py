"""Translated transverse geometry and one projected field/mobile/full64 law.

The finite plane-wave-in-Q projection is explicit. No oscillator frequency,
magnetic stiffness or inherited response is installed. Bare electric regional
energies commute exactly; changes require the actual shared quantum matter.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
import numpy as np
from scipy import sparse as sp
from rich_common_mobile_20261014_model import Mobile, MobileParameters, local_geometry, angular_tensor, line_form
from rich_frame_photon_20261014_model import Parameters, cube, mode_values


@dataclass(frozen=True)
class PacketParameters:
    base: Parameters = field(default_factory=lambda: Parameters(ell=.85, epsilon=.35, core_width=.38))
    centres: tuple = ((-.35,-.12,.08),(.45,.16,-.10))
    field_centres: tuple = ((-2.,0.,0.),(0.,0.,0.),(2.,0.,0.))
    cm_hopping: float = .48
    core_mass: float = 1.
    cutoff: int = 2
    qwall: float = 2.8
    supplier_spacing: float = 2.


def profiles(points, p, smear=0.):
    return np.concatenate([mode_values(np.asarray(points)-r,p.base.ell,smear) for r in p.field_centres])


def gram(p):
    centres=np.repeat(np.asarray(p.field_centres),2,axis=0)
    axes=np.tile([[0.,0.,1.],[0.,1.,0.]],(len(p.field_centres),1))
    d=centres[:,None]-centres[None,:]
    ell=p.base.ell
    pref=np.pi**1.5*ell**3*np.exp(-np.sum(d*d,axis=-1)/(4*ell**2))
    return pref*((ell**2-np.sum(d*d,axis=-1)/4)*(axes@axes.T)+np.einsum('abk,ak->ab',d,axes)*np.einsum('abk,bk->ab',d,axes)/4)

def line_profile(start,end,p,width):
    nodes,weights=np.polynomial.legendre.leggauss(40)
    delta=np.asarray(end)-start
    points=np.asarray(start)+(nodes[:,None]+1)*delta/2
    return np.einsum('aik,k,i->a',profiles(points,p,width),delta,weights/2)


def supplier(p):
    c=cube(p.base)
    sites=np.array([[x,y,0.] for x in np.arange(4)*p.supplier_spacing-1.5*p.supplier_spacing for y in (-.5,.5)])
    edges=[(i,i+1) for i in range(0,8,2)]+[(i,i+2) for i in range(6)]
    bits=c['bits']; lookup={b:i for i,b in enumerate(bits)}
    hops=[]
    for i,j in edges:
        rows=[];cols=[];vals=[]
        for col,state in enumerate(bits):
            for flavour in range(2):
                s,t=i+8*flavour,j+8*flavour
                if state&(1<<s) and not state&(1<<t):
                    intermediate=state^(1<<s)
                    sign=(-1)**((state&((1<<s)-1)).bit_count()+(intermediate&((1<<t)-1)).bit_count())
                    rows.append(lookup[intermediate|(1<<t)]);cols.append(col);vals.append(sign)
        hops.append(sp.csr_matrix((vals,(rows,cols)),shape=(64,64),dtype=complex))
    c.update(sites=sites,edges=edges,hops=hops,phase=np.array([line_profile(sites[i],sites[j],p,p.base.supplier_width) for i,j in edges]))
    return c


def momentum_indices(modes,cutoff):
    # Enumerate the joint L1 ball without allocating its tensor-product exterior.
    result=[]
    def visit(prefix,left):
        if len(prefix)==modes:
            result.append(prefix);return
        for k in range(-left,left+1):visit(prefix+[k],left-abs(k))
    visit([],cutoff)
    return np.array(result,dtype=int)


class Packet(Mobile):
    """Inherited physical sources, longitudinal Gauss and densities; new geometry."""
    def __init__(self,p,occupation=1,induction=True,coupling=True,metric=True,bridge=True):
        self.p=p;self.occupation=occupation;self.induction=induction;self.coupling=coupling;self.metric=metric;self.links=True;self.bridge=bridge
        b=p.base;self.geom=local_geometry(MobileParameters(base=b))
        self.ix=np.flatnonzero(abs(self.geom['N']-occupation)<1e-8);self.ns=len(self.ix)
        sub=lambda x:np.asarray(x)[np.ix_(self.ix,self.ix)]
        self.number=[sub(x) for x in self.geom['numbers']]
        self.spin=[[sub(x) for x in s] for s in self.geom['spins']]
        self.cube=supplier(p);self.nmodes=2*len(p.field_centres)
        self.indices=momentum_indices(self.nmodes,p.cutoff);self.nfield=len(self.indices)
        self.internal=self.ns*64;self.size=self.nfield*self.internal
        self.If=sp.eye(self.nfield,format='csr');self.Im=sp.eye(self.ns,format='csr');self.I64=sp.eye(64,format='csr');self.I=sp.eye(self.size,format='csr')
        self.G=gram(p);self.M=b.epsilon*self.G;self.Minv=np.linalg.inv(self.M)
        self.momenta=self.indices*np.pi/p.qwall
        self.velocities=self.momenta@self.Minv
        self.Efield=sp.diags(np.einsum('fa,ab,fb->f',self.momenta,self.Minv,self.momenta)/2,format='csr')
        self.E=self.fop(self.Efield)
        self.Q=[self.qproduct((a,)) for a in range(self.nmodes)]
        self.A=[self.fop(q) for q in self.Q];self.P=[self.fop(sp.diags(self.momenta[:,a])) for a in range(self.nmodes)]
        self.N=[self.mop(x) for x in self.number]
        self.S=[self.mop(sum(s[a] for s in self.spin)) for a in range(3)]
        self.X=[sum(p.centres[c][a]*self.N[c] for c in range(2)) for a in range(3)]
        self.ic=1/b.common_inertia if metric else 0.;self.ir=1/b.relative_inertia
        self.tensor=[np.concatenate([angular_tensor(np.asarray(r)-f,b) for f in p.field_centres],axis=1)*int(coupling) for r in p.centres]
        self.cmphase=-line_profile(np.asarray(p.centres[0]),np.asarray(p.centres[1]),p,b.core_width)
        rawhop=sum(self.geom['sources'][2+a]@self.geom['sources'][a].conj().T for a in range(2));self.cmhop=sp.csr_matrix(sub(rawhop))
        self.HL_values,self.charge_data=self.longitudinal();self.HL=sp.kron(self.If,sp.diags(self.HL_values.ravel()),format='csr')
        doublon=np.sum(self.cube['density']==2,axis=1)
        self.Hdiag=self.mop(p.core_mass*occupation*np.eye(self.ns))+sp.kron(self.If,sp.kron(self.Im,sp.diags(b.hubbard*doublon)),format='csr')
        self.Hbody=self.body();self.H=(self.E+self.HL+self.Hdiag+self.Hbody+self.link_derivative(())).tocsr()
        self.J=[(-self.link_derivative((a,))+self.body_current(a)).tocsr() for a in range(self.nmodes)]

    def qproduct(self,axes):
        d=self.indices[None,:,:]-self.indices[:,None,:]
        result=np.ones((self.nfield,self.nfield),dtype=complex)
        for a in range(self.nmodes):
            power=axes.count(a);z=d[:,:,a];nonzero=z!=0
            value=(z==0).astype(complex)
            if power==1:
                value[:]=0;value[nonzero]=-1j*self.p.qwall*(-1.)**z[nonzero]/(np.pi*z[nonzero])
            elif power==2:
                value[:]=self.p.qwall**2/3;value[nonzero]=2*self.p.qwall**2*(-1.)**z[nonzero]/(np.pi*z[nonzero])**2
            result*=value
        return sp.csr_matrix(result)

    def phase_matrix(self,w):
        d=self.indices[None,:,:]-self.indices[:,None,:]
        return sp.csr_matrix(np.prod(np.sinc(d+w[None,None,:]*self.p.qwall/np.pi),axis=2))

    def body(self,shift=None):
        shift=np.zeros(self.nmodes) if shift is None else np.asarray(shift)
        result=sp.csr_matrix((self.size,self.size),dtype=complex)
        for c in range(2):
            n=self.number[c];d=self.tensor[c]
            result+=self.mop((self.ic+1/self.p.base.frame_inertia)*sum(s@s for s in self.spin[c])/2)
            for j in range(3):
                linear=sum(d[j,a]*self.Q[a] for a in range(self.nmodes))+sp.eye(self.nfield)*np.dot(d[j],shift)
                result-=self.ic*sp.kron(linear,sp.kron(sp.csr_matrix(n@self.spin[c][j]),self.I64),format='csr')
            quadratic=sp.csr_matrix((self.nfield,self.nfield),dtype=complex)
            dd=d.T@d
            for a in range(self.nmodes):
                for b in range(self.nmodes):
                    quadratic+=dd[a,b]*(self.qproduct((a,b))+shift[a]*self.Q[b]+shift[b]*self.Q[a]+sp.eye(self.nfield)*shift[a]*shift[b])
            result+=(self.ic+self.ir)/2*sp.kron(quadratic,sp.kron(sp.csr_matrix(n@n),self.I64),format='csr')
        return result.tocsr()

    def body_current(self,a):
        result=sp.csr_matrix((self.size,self.size),dtype=complex)
        for c in range(2):
            d=self.tensor[c];n=self.number[c]
            result+=self.ic*self.mop(sum(d[j,a]*n@self.spin[c][j] for j in range(3)))
            linear=sum((d[:,a]@d[:,b])*self.Q[b] for b in range(self.nmodes))
            result-=(self.ic+self.ir)*sp.kron(linear,sp.kron(sp.csr_matrix(n@n),self.I64),format='csr')
        return result.tocsr()

    def link_terms(self):
        yield self.cmphase*int(self.coupling),self.cmhop,True,self.p.cm_hopping
        for index,(w,h) in enumerate(zip(self.cube['phase'],self.cube['hops'])):
            i,j=self.cube['edges'][index]
            cut=not self.bridge and self.cube['sites'][i,0]<0<self.cube['sites'][j,0]
            yield w*int(self.coupling),h,False,self.p.base.hopping*int(self.induction and not cut)

    def link_derivative(self,axes,shift=None,uniform=(0.,0.,0.)):
        shift=np.zeros(self.nmodes) if shift is None else np.asarray(shift)
        result=sp.csr_matrix((self.size,self.size),dtype=complex)
        for index,(w,hop,iscm,t) in enumerate(self.link_terms()):
            if iscm:delta=-(np.asarray(self.p.centres[1])-self.p.centres[0])
            else:
                i,j=self.cube['edges'][index-1];delta=self.cube['sites'][j]-self.cube['sites'][i]
            coefficient=-t*np.exp(1j*(w@shift+np.dot(delta,uniform)))
            for a in axes:coefficient*=1j*w[a]
            matter=sp.kron(hop,self.I64) if iscm else sp.kron(self.Im,hop)
            forward=sp.kron(coefficient*self.phase_matrix(w),matter,format='csr')
            result+=forward+forward.getH()
        return result.tocsr()

    def shifted(self,shift=None,uniform=(0.,0.,0.)):
        return self.E+self.HL+self.Hdiag+self.body(shift)+self.link_derivative((),shift,uniform)

    def current_fourier(self,k):
        k=np.asarray(k);b=self.p.base
        result=[sp.csr_matrix((self.size,self.size),dtype=complex) for _ in range(3)]
        for index,(w,hop,iscm,t) in enumerate(self.link_terms()):
            if iscm:start,end=np.asarray(self.p.centres);width=b.core_width;charge=-1.
            else:
                i,j=self.cube['edges'][index-1];start,end=self.cube['sites'][[i,j]];width=b.supplier_width;charge=1.
            matter=sp.kron(hop,self.I64) if iscm else sp.kron(self.Im,hop)
            forward=sp.kron(-t*self.phase_matrix(w),matter,format='csr');flow=-1j*(forward-forward.getH())
            for a in range(3):result[a]+=charge*(end-start)[a]*line_form(start,end,k,width)*flow
        for c in range(2):
            n=self.number[c];d=self.tensor[c]
            omega=[self.ic*self.mop(n@self.spin[c][a])-(self.ic+self.ir)*sp.kron(sum(d[a,j]*self.Q[j] for j in range(self.nmodes)),sp.kron(sp.csr_matrix(n@n),self.I64),format='csr') for a in range(3)]
            coefficient=1j*b.core_width**2*np.exp(-1j*np.dot(self.p.centres[c],k)-b.core_width**2*np.dot(k,k)/2)
            for a in range(3):
                j,l=(a+1)%3,(a+2)%3;result[a]+=coefficient*(omega[j]*k[l]-omega[l]*k[j])
        return [x.tocsr() for x in result]

    def cm_current(self):
        forward=sp.kron(-self.p.cm_hopping*self.phase_matrix(self.cmphase*int(self.coupling)),sp.kron(self.cmhop,self.I64),format='csr')
        flow=-1j*(forward-forward.getH());delta=np.asarray(self.p.centres[1])-self.p.centres[0]
        return [-delta[a]*flow for a in range(3)]

    def electric_window(self,g):
        values=self.p.base.epsilon*np.einsum('fa,ab,fb->f',self.velocities,g,self.velocities)/2
        return sp.diags(values,format='csr')


def spatial_windows(p):
    """Actual 3D quadrature, including tails, not mode/site projectors."""
    regions=[(-7.,-1.),(-1.,1.),(1.,7.)]
    grams=[];first=[];tail=[]
    yz,wyz=np.polynomial.legendre.leggauss(30)
    for low,high in regions:
        gx,wx=np.polynomial.legendre.leggauss(44)
        x=(low+high)/2+(high-low)*gx/2
        xx,yy,zz=np.meshgrid(x,5*yz,5*yz,indexing='ij')
        weights=(wx[:,None,None]*(high-low)/2*(5*wyz)[None,:,None]*(5*wyz)[None,None,:]).ravel()
        points=np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()]);u=profiles(points,p)
        grams.append(np.einsum('aik,bik,i->ab',u,u,weights))
        first.append(np.einsum('aik,bik,i->ab',u,u,weights*points[:,0]))
    return np.array(grams),np.array(first)
