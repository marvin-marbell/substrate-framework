"""Exact C0 tensor I + Cphi tensor Phi + I tensor Hphi, ALL scalar modes.
Representation-only resource boundary: no scalar mean/state or virtual cutoff.
"""
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from rich_composite_geometry_20261015_angular import project_product,right_angular


class PhiHamiltonian(LinearOperator):
    def __init__(self,c0,cphi,hphi,grid):
        self.c0=sparse.csr_matrix(c0);self.cphi=sparse.csr_matrix(cphi)
        self.hphi=np.asarray(hphi,complex);self.grid=np.asarray(grid,float)
        self.n=self.c0.shape[0];self.nphi=len(self.grid)
        self.adj0=self.c0.conj().T;self.adjphi=self.cphi.conj().T
        super().__init__(dtype=np.dtype(complex),shape=(self.n*self.nphi,)*2)
    def action(self,value,adjoint=False):
        vector=np.asarray(value);columns=1 if vector.ndim==1 else vector.shape[1]
        tensor=vector.reshape(self.n,self.nphi,columns)
        c0=self.adj0 if adjoint else self.c0;cphi=self.adjphi if adjoint else self.cphi
        result=(c0@tensor.reshape(self.n,-1)).reshape(tensor.shape)
        result+=(cphi@(tensor*self.grid[None,:,None]).reshape(self.n,-1)).reshape(tensor.shape)
        result+=np.einsum('ab,nbk->nak',self.hphi.conj().T if adjoint else self.hphi,tensor)
        return result.reshape(vector.shape)
    def _matvec(self,value):return self.action(value)
    def _matmat(self,value):return self.action(value)
    def _rmatvec(self,value):return self.action(value,True)
    def _rmatmat(self,value):return self.action(value,True)
    def diagonal(self):return (self.c0.diagonal()[:,None]+self.cphi.diagonal()[:,None]*self.grid[None,:]+self.hphi.diagonal()[None,:]).reshape(-1)
    def toarray(self):
        if self.shape[0]>16:raise ValueError('No giant dense materialization')
        return self@np.eye(self.shape[0])
    def restrict_body(self,indices):return PhiHamiltonian(self.c0[indices][:,indices],self.cphi[indices][:,indices],self.hphi,self.grid)
    def remove_static(self,operator):return PhiHamiltonian(self.c0-operator,self.cphi,self.hphi,self.grid)
    def hermiticity_error(self):
        a=[self.c0-self.c0.conj().T,self.cphi-self.cphi.conj().T,sparse.eye(self.n,format='csr')]
        b=[np.eye(self.nphi),np.diag(self.grid),self.hphi-self.hphi.conj().T]
        value=0j
        for i in range(3):
            for j in range(3):value+=a[i].conjugate().multiply(a[j]).sum()*np.vdot(b[i],b[j])
        return float(np.sqrt(value.real))
    def save(self,path):
        d={'factor_kind':np.array('C0_I_Cphi_Phi_I_Hphi_v1'),'hphi':self.hphi,'grid':self.grid}
        for name,m in [('c0',self.c0),('cphi',self.cphi)]:
            d.update({name+'_data':m.data,name+'_indices':m.indices,name+'_indptr':m.indptr,name+'_shape':m.shape})
        np.savez_compressed(path,**d)


def load_h(path):
    with np.load(path,allow_pickle=False) as d:
        if 'factor_kind' not in d:return sparse.load_npz(path)
        if str(d['factor_kind'])!='C0_I_Cphi_Phi_I_Hphi_v1':raise ValueError('Unknown factor schema')
        matrices={name:sparse.csr_matrix((d[name+'_data'],d[name+'_indices'],d[name+'_indptr']),shape=tuple(d[name+'_shape'])) for name in ('c0','cphi')}
        return PhiHamiltonian(matrices['c0'],matrices['cphi'],d['hphi'],d['grid'])


def reduce_factors(ops,coupling,rotor,scalar,p,graph):
    eye=sparse.eye(ops['B'].shape[0],format='csr');rotation_eye=sparse.eye(10,format='csr')
    c0=project_product(coupling,ops['electric']-p['plaquette']*ops['B'],rotation_eye)
    k=right_angular(rotor);kinetic=sparse.csr_matrix((10,10),dtype=complex)
    for a,b in np.ndindex(3,3):kinetic+=graph['inertia_inverse'][a,b]*(k[a]@k[b])/2
    c0+=project_product(coupling,eye,kinetic)
    for i,a in np.ndindex(3,3):c0+=p['vector_velocity']*project_product(coupling,ops[f'vector_{i}_{a}'],rotor['coordinate'][i][a])
    cphi=p['scalar_coupling']*project_product(coupling,ops['scalar'],rotation_eye)
    return PhiHamiltonian(c0,cphi,scalar['H'],scalar['grid'])


def finite_matrix_control(h):
    """Positive formula control on a small actual selected row/column subblock.
    No stationary source state/eigenvalue comes from this block.
    """
    indices=np.unique(np.linspace(0,h.n-1,min(5,h.n),dtype=int));small=h.restrict_body(indices)
    direct=sparse.kron(small.c0,sparse.eye(h.nphi))+sparse.kron(small.cphi,np.diag(h.grid))+sparse.kron(sparse.eye(len(indices)),h.hphi)
    rng=np.random.default_rng(422);v=rng.normal(size=small.shape[0])+1j*rng.normal(size=small.shape[0])
    return {'dimension':small.shape[0],'full_scalar_modes':h.nphi,'action_error':float(np.linalg.norm(small@v-direct@v)),
        'adjoint_action_error':float(np.linalg.norm(small.rmatvec(v)-direct.conj().T@v)),
        'scope':'Small arbitrary actual coefficient subblock Kronecker formula control, NOT a physics/virtual/state truncation.'}
