"""Compact SAME physical law, retaining body x R x phi snapshot ordering.
Reuses the actual peer's generic FactorOperator only, NEVER its H/state/energies.
No field-space reduction, current replacement, new coupling or spin term.
"""
import numpy as np
from scipy import sparse
from rich_spatial_colour_em_20261015_model import FactorOperator
from rich_spatial_colour_root_20261015_model import phase_matrix,clean,TensorOperator
from rich_spatial_colour_root_20261015_calculate import lowest


class NativeFactor(FactorOperator):
    def _matmat(self,vectors):
        vectors=np.asarray(vectors);columns=vectors.shape[1]
        x=vectors.reshape(self.nm,self.nrq,self.nf,columns);out=np.zeros_like(x,dtype=complex)
        if self.include_scalar:out+=np.einsum('fg,mkgc->mkfc',self.scalar['H'],x,optimize=True)
        for matter,rotation,phi in self.terms:
            left=(matter@x.reshape(self.nm,-1)).reshape(x.shape)
            right=(rotation@left.transpose(1,0,2,3).reshape(self.nrq,-1)).reshape(self.nrq,self.nm,self.nf,columns).transpose(1,0,2,3)
            if phi:right=right*self.scalar['grid'][None,None,:,None]
            out+=right
        return out.reshape(self.shape[0],columns)
    def diagonal(self):
        diagonal=np.zeros((self.nm,self.nrq,self.nf),complex)
        if self.include_scalar:diagonal+=np.diag(self.scalar['H'])[None,None,:]
        for matter,rotation,phi in self.terms:
            scalar=self.scalar['grid'] if phi else np.ones(self.nf)
            diagonal+=matter.diagonal()[:,None,None]*rotation.diagonal()[None,:,None]*scalar[None,None,:]
        return diagonal.ravel()
    def _rmatvec(self,vector):
        if self.hermitian:return self._matvec(vector)
        return NativeFactor(self.nm,self.scalar,self.nrq,[(a.conj().T,b.conj().T,p) for a,b,p in self.terms],self.include_scalar)@vector


_ADJOINTS={}
def adjoint(matrix):
    key=id(matrix)
    if key not in _ADJOINTS:_ADJOINTS[key]=(matrix,matrix.conj().T.tocsr())
    return _ADJOINTS[key][1]


def law(op,rotor,scalar,p,g,fields=None,derivative=(),path_scale=(1.,1.)):
    fields=np.zeros(6) if fields is None else np.asarray(fields,float);order=len(derivative);n=op['Q'].shape[0]
    terms=[];identity=sparse.eye(n,format='csr');ir=sparse.eye(10,format='csr');sign=-1 if order==1 else 1
    def add(a,b,phi=False,coefficient=1.):
        if coefficient:terms.append((a,clean(b*(sign*coefficient)),phi))
    def linear(k,weight):
        matter=op['density_delta'] if k<3 else op['Q']
        for a in range(3):
            alpha=rotor['integrate'](op['connection']['values'][k,a]);angular=rotor['L'][a]@alpha+alpha@rotor['L'][a]
            add(matter,angular,False,-weight/(2*p['orientation_inertia']))
    def quadratic(k,l,weight):
        kind=int(k>=3)+int(l>=3);moment=rotor['integrate'](np.sum(op['connection']['values'][k]*op['connection']['values'][l],axis=0))
        add(op['connection']['products'][kind],moment,False,weight/(2*p['orientation_inertia']))
    if order==0:
        add(op['electric']-p['plaquette']*op['B'],ir);add(identity,rotor['L2'],False,1/(2*p['orientation_inertia']))
        for k in range(6):
            if fields[k]:
                linear(k,fields[k])
                for l in range(6):
                    if fields[l]:quadratic(k,l,fields[k]*fields[l])
    elif order==1:
        linear(derivative[0],1.)
        for l in range(6):
            if fields[l]:quadratic(derivative[0],l,2*fields[l])
    else:quadratic(derivative[0],derivative[1],2.)
    if order==0 and not np.any(fields):
        # Exact algebraic collection at H0, not a low-mode/product approximation.
        phi=sparse.csr_matrix((n,n),dtype=complex)
        for e in range(2):
            for _,matrices in op['hops'][e]:
                phi+=g['weights'][e]*path_scale[e]*p['scalar_coupling']*(matrices[0]+adjoint(matrices[0]))
        add(phi,ir,True)
        for i in range(3):
            for a in range(3):
                body=sparse.csr_matrix((n,n),dtype=complex)
                for e in range(2):
                    coefficient=1j*p['vector_velocity']*g['weights'][e]*path_scale[e]*g['body_tangent'][e,a]
                    if coefficient:
                        for _,matrices in op['hops'][e]:body+=coefficient*(matrices[i+1]-adjoint(matrices[i+1]))
                add(body,rotor['coordinate'][i][a])
    else:
        for e in range(2):
            weight=g['weights'][e]*path_scale[e]
            if not weight:continue
            for q,matrices in op['hops'][e]:
                angle=phase_matrix(rotor,g,e,q,fields,derivative)
                add(matrices[0],angle,True,p['scalar_coupling']*weight);add(adjoint(matrices[0]),angle.conj().T,True,p['scalar_coupling']*weight)
                for i in range(3):
                    angle=phase_matrix(rotor,g,e,q,fields,derivative,i)
                    add(matrices[i+1],angle,False,1j*p['vector_velocity']*weight);add(adjoint(matrices[i+1]),angle.conj().T,False,-1j*p['vector_velocity']*weight)
    return NativeFactor(n,scalar,10,terms,include_scalar=(order==0))


def route_current(op,rotor,scalar,p,g,e):
    n=op['Q'].shape[0];terms=[];phi=sparse.csr_matrix((n,n),dtype=complex)
    for q,matrices in op['hops'][e]:phi+=-1j*q*p['scalar_coupling']*g['weights'][e]*(matrices[0]-adjoint(matrices[0]))
    terms.append((phi,sparse.eye(10),True))
    for i in range(3):
        body=sparse.csr_matrix((n,n),dtype=complex)
        for q,matrices in op['hops'][e]:body+=q*p['vector_velocity']*g['weights'][e]*(matrices[i+1]+adjoint(matrices[i+1]))
        terms.append((body,rotor['integrate'](g['tangents'][:,e,i]),False))
    return NativeFactor(n,scalar,10,terms)


def matrix(arrays,key):
    return sparse.csr_matrix((arrays[key+'_data'],arrays[key+'_indices'],arrays[key+'_indptr']),shape=tuple(arrays[key+'_shape']))


def stationary(arrays,name,h,op,rotor,scalar,M,count,p):
    mr,u=np.linalg.eigh(rotor['L'][2].toarray());labels=(op['S_z_labels'][:,None]+np.round(mr,8)[None,:]).ravel();ids=np.flatnonzero(abs(labels-M)<1e-7)
    transform=sparse.kron(sparse.eye(op['Q'].shape[0]),sparse.csr_matrix(u),format='csr')[:,ids]
    full=matrix(arrays,name+'_H0_one');one=clean(transform.conj().T@full@transform);del full
    full=matrix(arrays,name+'_H0_phi');phi=clean(transform.conj().T@full@transform);del full
    restricted=TensorOperator(one,phi,scalar,True);energies,vectors=lowest(restricted,count,p)
    states=np.column_stack([(transform@v.reshape(len(ids),len(scalar['grid']))).ravel() for v in vectors.T])
    residuals=[float(np.linalg.norm(h@states[:,j]-energies[j]*states[:,j])) for j in range(len(energies))]
    return energies,states,dict(total_M=M,stationary_dimension=restricted.shape[0],full_dimension=h.shape[0],residuals=residuals,orthogonality=float(np.linalg.norm(states.conj().T@states-np.eye(len(energies)))),scope='Exact total-M eigen block of OWN captured H0, compact full-H action residual, all-M propagation/resolvents unchanged.')
