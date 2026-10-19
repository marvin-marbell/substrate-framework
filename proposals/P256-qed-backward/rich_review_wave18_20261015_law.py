"""Matrix-free ORIGINAL native body x SO3(10) x scalar source family.
Original forward+h.c. hopping, source derivatives and before-cut rotor moments;
no changed coupling, H averaging, current patch, angular cut or author mutation.
"""
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


class PhysicalLaw(LinearOperator):
    def __init__(self,op: Any,rotor: Any,scalar: Any,p: Any,g: Any,fields: Any=None,derivative: Any=(),path_scale: Any=(1.,1.)):
        self.n=op['Q'].shape[0];self.nf=len(scalar['grid']);self.scalar=scalar;self.order=len(derivative);self.terms=[];self.linear=[];self._factor_adjoints={};eye=sparse.eye(self.n,format='csr');ir=np.eye(10);fields=np.zeros(6) if fields is None else np.asarray(fields,float);conn=op['connection'];sign=-1 if self.order==1 else 1
        def add(coefficient: Any,f: Any,r: Any,phi: bool=False)->None:
            if coefficient!=0:self.terms.append((complex(sign*coefficient),f,np.asarray(r.toarray() if sparse.issparse(r) else r),phi))
        def adjoint(f: Any)->Any:
            key=id(f)
            if key not in self._factor_adjoints:
                ft=f.conj().T.tocsr();self._factor_adjoints[key]=ft;self._factor_adjoints[id(ft)]=f
            return self._factor_adjoints[key]
        def quadratic(k: int,l: int,coefficient: Any)->None:
            product_type=int(k>=3)+int(l>=3);r=rotor['integrate'](np.sum(conn['values'][k]*conn['values'][l],axis=0));add(coefficient/(2*p['orientation_inertia']),conn['products'][product_type],r)
        if self.order==0:
            add(1.,op['electric']-p['plaquette']*op['B'],ir);add(1/(2*p['orientation_inertia']),eye,rotor['L2'])
            for k in range(6):
                if fields[k]:self.linear.append((complex(fields[k]),conn['linear'][k]))
            for k in range(6):
                for l in range(6):
                    if fields[k] and fields[l]:quadratic(k,l,fields[k]*fields[l])
        elif self.order==1:
            k=derivative[0];self.linear.append((-1.+0j,conn['linear'][k]))
            for l in range(6):
                if fields[l]:quadratic(k,l,2*fields[l])
        else:quadratic(derivative[0],derivative[1],2.)
        for edge in range(2):
            if not path_scale[edge]:continue
            weight=g['weights'][edge]*path_scale[edge];coefficient=g['coefficients'][:,edge]
            for charge,matrices in op['hops'][edge]:
                values=np.exp(1j*charge*(coefficient@fields))
                for k in derivative:values*=1j*charge*coefficient[:,k]
                angle=rotor['integrate'](values);f=matrices[0];ft=adjoint(f);add(p['scalar_coupling']*weight,f,angle,True);add(p['scalar_coupling']*weight,ft,angle.conj().T,True)
                for axis in range(3):
                    angle=rotor['integrate'](values*g['tangents'][:,edge,axis]);f=matrices[axis+1];ft=adjoint(f);add(1j*p['vector_velocity']*weight,f,angle);add(-1j*p['vector_velocity']*weight,ft,angle.conj().T)
        self.adjoint_terms=[(np.conj(c),adjoint(f),r.conj().T,phi) for c,f,r,phi in self.terms];self.adjoint_linear=[(np.conj(c),m.conj().T.tocsr()) for c,m in self.linear];self._trace=sum(c*f.diagonal().sum()*np.trace(r)*(sum(scalar['grid']) if phi else self.nf) for c,f,r,phi in self.terms)+sum(c*m.diagonal().sum()*self.nf for c,m in self.linear)
        if self.order==0:self._trace+=self.n*10*np.trace(scalar['H'])
        super().__init__(dtype=np.dtype(complex),shape=(self.n*10*self.nf,)*2)

    def action(self,value: Any,adjoint: bool=False)->Any:
        x=np.asarray(value).reshape(self.n,10,self.nf);out=np.zeros_like(x,dtype=complex)
        for c,f,r,phi in self.adjoint_terms if adjoint else self.terms:
            weighted=(f@x.reshape(self.n,-1)).reshape(x.shape);part=np.einsum('ab,mbf->maf',r,weighted,optimize=True)
            if phi:part*=self.scalar['grid'][None,None,:]
            out+=c*part
        for c,m in self.adjoint_linear if adjoint else self.linear:out+=(c*(m@x.reshape(self.n*10,self.nf))).reshape(x.shape)
        if self.order==0:out+=x@(self.scalar['H'].conj() if adjoint else self.scalar['H'].T)
        return out.ravel()

    def _matvec(self,value: Any)->Any:return self.action(value)
    def _rmatvec(self,value: Any)->Any:return self.action(value,True)
    def _matmat(self,value: Any)->Any:return np.column_stack([self.action(v) for v in value.T])
    def _rmatmat(self,value: Any)->Any:return np.column_stack([self.action(v,True) for v in value.T])
    def trace(self)->Any:return self._trace


def physical_law(op: Any,rotor: Any,scalar: Any,p: Any,g: Any,fields: Any=None,derivative: Any=(),path_scale: Any=(1.,1.))->Any:
    return PhysicalLaw(op,rotor,scalar,p,g,fields,derivative,path_scale)
