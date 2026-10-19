"""Exact unbounded electric interaction-picture primitives, hbar=1.
U0=exp[-it Pi²/(2I)], W=exp(i impulse.a), basis U0 W|n>.
Finite labels restrict INITIAL coefficients, never the physical free evolution.
"""
from math import factorial
import numpy as np
from scipy.special import eval_genlaguerre,eval_hermite
import rich_spatial_frame_photon_20261016_fields as frozen


def displacement(target,reference,coefficients,time,inertia=2.,impulse=None):
    """P exp[i c.(a+t(Pi+impulse)/I)] P; exact full Weyl/BCH matrix."""
    c=np.asarray(coefficients);tau=time/inertia
    impulse=np.zeros(c.shape[-1]) if impulse is None else np.asarray(impulse)
    out=np.ones(c.shape[:-1]+(len(target),len(reference)),dtype=np.complex128)
    for axis in range(c.shape[-1]):
        alpha=(-tau+1j)*c[...,axis,None,None]/np.sqrt(2)
        n=np.array(target)[:,axis,None];m=np.array(reference)[None,:,axis]
        lo=np.minimum(n,m);hi=np.maximum(n,m);delta=hi-lo
        power=np.where(n>=m,alpha,-alpha.conj())**delta
        out*=np.exp(-abs(alpha)**2/2)*power*np.sqrt(np.vectorize(factorial)(lo)/np.vectorize(factorial)(hi))*eval_genlaguerre(lo,delta,abs(alpha)**2)
    return out*np.exp(1j*tau*(c@impulse))[...,None,None]


def primitive_images(target,reference,time,inertia=2.,impulse=None):
    """Full b_l=a_l+t(Pi_l+impulse_l)/I images, then their ordered products."""
    nm=len(reference[0]);eps=np.zeros(nm) if impulse is None else np.asarray(impulse)
    lookup={v:i for i,v in enumerate(target)}
    embedding=np.zeros((len(target),len(reference)))
    for j,v in enumerate(reference):
        if v in lookup:embedding[lookup[v],j]=1
    q=np.array([frozen.primitive(target,reference,a,'q').toarray() for a in range(nm)])
    p=np.array([frozen.primitive(target,reference,a,'p').toarray() for a in range(nm)])
    b=q+time/inertia*(p+eps[:,None,None]*embedding)
    return dict(q=q,p=p,b=b,embedding=embedding,bb=np.einsum('lei,mej->lmij',b.conj(),b,optimize=True))


def coefficient_derivative(target,reference,coefficients,axes,time,inertia=2.,impulse=None):
    """Coefficient-c derivative, NOT external classical-A source variation."""
    current=reference;op=np.eye(len(reference),dtype=np.complex128)
    extended=frozen.basis(max(map(sum,reference))+len(axes),len(reference[0]))
    for axis in reversed(axes):
        image=primitive_images(extended,current,time,inertia,impulse)['b'][axis]
        op=image@op;current=extended
    return (1j**len(axes))*np.einsum('...ij,jk->...ik',displacement(target,current,coefficients,time,inertia,impulse),op,optimize=True)


def wavefunctions(x,maximum,time,inertia=2.,impulse=0.):
    """Exact 1D U0(t)exp(i impulse*a) Hermite functions and x derivatives."""
    x=np.asarray(x);tau=time/inertia;width=np.sqrt(1+tau*tau);y=x-tau*impulse;u=y/width
    phase=np.exp(1j*(tau*y*y/(2*width**2)+impulse*x-tau*impulse**2/2))
    oscillator=np.array([np.pi**(-.25)*eval_hermite(n,u)*np.exp(-u*u/2)/np.sqrt(2.**n*factorial(n)) for n in range(maximum+2)]).T
    waves=[];derivatives=[]
    for n in range(maximum+1):
        factor=phase*np.exp(-1j*(n+.5)*np.arctan(tau))/np.sqrt(width)
        derivative=(np.sqrt(n/2)*oscillator[:,n-1] if n else 0)-np.sqrt((n+1)/2)*oscillator[:,n+1]
        waves.append(factor*oscillator[:,n])
        derivatives.append(factor*(derivative/width+1j*(impulse+tau*y/width**2)*oscillator[:,n]))
    return np.array(waves).T,np.array(derivatives).T


def protocol_overlap(target,reference,left_impulse,right_impulse):
    """Same absolute time overlap; U0 cancels, Wleft† Wright is full Weyl."""
    return frozen.displacement(target,reference,np.asarray(right_impulse)-np.asarray(left_impulse))
