"""Unbounded oscillator primitives projected only after their products/exponentials.
Oscillator functions are a numerical basis, NOT a supplied oscillator energy.
"""
from itertools import product
from math import factorial
import numpy as np
from scipy import sparse as sp
from scipy.special import eval_genlaguerre


def basis(cut,modes=8):
    if modes==1:return [(i,) for i in range(cut+1)]
    return [(i,)+v for i in range(cut+1) for v in basis(cut-i,modes-1)]


def primitive(target,reference,axis,kind):
    lookup={v:i for i,v in enumerate(target)};r=[];c=[];d=[]
    for j,v in enumerate(reference):
        for sign in (-1,1):
            w=list(v);w[axis]+=sign;k=lookup.get(tuple(w))
            if k is None:continue
            z=np.sqrt((v[axis]+(sign>0))/2)
            r.append(k);c.append(j);d.append(z*(1j*sign if kind=='p' else 1))
    return sp.csr_matrix((d,(r,c)),shape=(len(target),len(reference)))


def displacement(target,reference,coefficients):
    """Exact <target|exp(i sum c_l a_l)|reference>; batched c (...,modes)."""
    c=np.asarray(coefficients);out=np.ones(c.shape[:-1]+(len(target),len(reference)),complex)
    for a in range(c.shape[-1]):
        z=c[...,a,None,None]/np.sqrt(2)
        n=np.array(target)[:,a,None];m=np.array(reference)[None,:,a]
        lo=np.minimum(n,m);hi=np.maximum(n,m);delta=hi-lo
        out*=np.exp(-z*z/2)*(1j*z)**delta*np.sqrt(np.vectorize(factorial)(lo)/np.vectorize(factorial)(hi))*eval_genlaguerre(lo,delta,z*z)
    return out


def displacement_derivative(target,reference,coefficients,axes):
    """Coefficient-c derivatives (i*a insertions), NOT external-A source variation."""
    c=np.asarray(coefficients);extended=basis(max(map(sum,reference))+len(axes),len(reference[0]))
    op=sp.eye(len(reference),format='csr');current=reference
    for axis in reversed(axes):
        op=primitive(extended,current,axis,'q')@op;current=extended
    return (1j**len(axes))*np.einsum('...ij,jk->...ik',displacement(target,current,c),sp.csr_matrix(op).toarray())
