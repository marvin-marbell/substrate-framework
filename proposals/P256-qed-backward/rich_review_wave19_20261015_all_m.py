"""Exact ALL-total-M factor representation, not a fixed-body/selected-M cut.
Every component is reconstructed into the full body x phi x R x field state.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,expm_multiply
from rich_spatial_colour_em_20261015_model import FactorOperator
from rich_spatial_colour_em_20261015_spectrum import _total_M_operator


class AllM:
    def __init__(self,model,name,h):
        base=model['base'];field=model['field'];self.h=h
        labels=np.rint(2*base['operators'][name]['S'][2].diagonal().real).astype(int)
        r,ur=np.linalg.eigh(base['rotor']['L'][2].toarray());f,uf=np.linalg.eigh(field['F'][2]);self.u=np.kron(ur,uf)
        rq=(np.rint(2*r).astype(int)[:,None]+np.rint(2*f).astype(int)[None,:]).ravel()
        transformed=FactorOperator(h.nm,h.scalar,h.nrq,[(a,self.u.conj().T@b.toarray()@self.u,phi) for a,b,phi in h.terms],h.include_scalar)
        diagonal=transformed.diagonal();self.parts=[];coverage=np.zeros(h.shape[0],int)
        for total in sorted(set((labels[:,None]+rq[None,:]).ravel())):
            operator,selected=_total_M_operator(transformed,labels,rq,int(total));coverage[selected]+=1;self.parts.append((int(total),operator,selected,diagonal[selected].sum()))
        if not np.all(coverage==1):raise ValueError('All-M cover must contain every retained coordinate exactly once')
        rng=np.random.default_rng(190193);x=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]);x/=np.linalg.norm(x);y=self.to_basis(x);image=np.zeros_like(y)
        for label,operator,indices,trace in self.parts:image[indices]=operator@y[indices]
        hx=h@x;self.evidence=dict(full_dimension=h.shape[0],sum_component_dimensions=sum(len(v) for _,_,v,_ in self.parts),twice_total_M=[m for m,_,_,_ in self.parts],component_dimensions=[len(v) for _,_,v,_ in self.parts],all_M_cover_exact=True,full_H_action_relative_error=float(np.linalg.norm(self.from_basis(image)-hx)/max(1.,np.linalg.norm(hx))),meaning='All physical total M blocks retained and reconstructed; this is an operator-factor optimization only. No fixed body M, Jhalf filter, smaller field/body/scalar cut or source/state deletion.')
        if self.evidence['full_H_action_relative_error']>1e-10:raise ValueError('All-M representation is not the owning H')

    def to_basis(self,state):
        return np.einsum('ab,mfb->mfa',self.u.conj().T,state.reshape(self.h.nm,self.h.nf,self.h.nrq),optimize=True).ravel()

    def from_basis(self,state):
        return np.einsum('ab,mfb->mfa',self.u,state.reshape(self.h.nm,self.h.nf,self.h.nrq),optimize=True).ravel()

    def evolve(self,state,t,energy):
        if not t:return state.copy()
        x=self.to_basis(state);out=np.zeros_like(x)
        for label,h,indices,trace in self.parts:
            value=x[indices]
            if not np.any(value):continue
            shifted=LinearOperator(h.shape,matvec=lambda z,a=h:a@z-energy*z,rmatvec=lambda z,a=h:a.H@z-energy*z,matmat=lambda z,a=h:a@z-energy*z,dtype=complex)
            out[indices]=expm_multiply(-1j*t*shifted,value,traceA=-1j*t*(trace-energy*len(indices)))
        return self.from_basis(out)
