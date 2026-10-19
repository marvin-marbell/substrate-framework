"""NEW extended spatial CAR32/Gauss/SO3/phi/3D quantum EM Hamiltonian.
Every Peierls/spin/rotation/field product is projected as one physical product.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from rich_spatial_colour_root_20261015_model import build as spatial_build, clean, site_source, raw_C
from rich_spatial_colour_em_20261015_fields import (
    field_shell,route_coefficients,endpoint_connections,phase_kernel,mechanical_kernels,gaussian_green)


class FactorOperator(LinearOperator):
    """State order physical CAR/PW, scalar, material-SO3, complete field shell."""
    def __init__(self,matter_dimension,scalar,rq_dimension,terms,include_scalar=False,hermitian=True):
        self.nm=matter_dimension;self.scalar=scalar;self.nf=len(scalar['grid']);self.nrq=rq_dimension
        self.terms=[(clean(a),clean(b),bool(phi)) for a,b,phi in terms if sparse.csr_matrix(a).nnz and sparse.csr_matrix(b).nnz]
        self.include_scalar=include_scalar;self.hermitian=hermitian
        dimension=self.nm*self.nf*self.nrq
        super().__init__(dtype=np.dtype(complex),shape=(dimension,dimension))
    def _matmat(self,vectors):
        vectors=np.asarray(vectors);columns=vectors.shape[1]
        x=vectors.reshape(self.nm,self.nf,self.nrq,columns);out=np.zeros_like(x,dtype=complex)
        if self.include_scalar:out+=np.einsum('fg,mgkc->mfkc',self.scalar['H'],x,optimize=True)
        for matter,rq,phi in self.terms:
            left=(matter@x.reshape(self.nm,-1)).reshape(x.shape)
            right=(rq@left.transpose(2,0,1,3).reshape(self.nrq,-1)).reshape(self.nrq,self.nm,self.nf,columns).transpose(1,2,0,3)
            if phi:right=right*self.scalar['grid'][None,:,None,None]
            out+=right
        return out.reshape(self.shape[0],columns)
    def _matvec(self,vector):return self._matmat(np.asarray(vector).reshape(-1,1))[:,0]
    def _rmatvec(self,vector):
        if self.hermitian:return self._matvec(vector)
        return FactorOperator(self.nm,self.scalar,self.nrq,[(a.conj().T,b.conj().T,phi) for a,b,phi in self.terms],self.include_scalar)@vector
    def diagonal(self):
        diagonal=np.zeros((self.nm,self.nf,self.nrq),complex)
        if self.include_scalar:diagonal+=np.diag(self.scalar['H'])[None,:,None]
        for matter,rq,phi in self.terms:
            scalar=self.scalar['grid'] if phi else np.ones(self.nf)
            diagonal+=matter.diagonal()[:,None,None]*scalar[None,:,None]*rq.diagonal()[None,None,:]
        return diagonal.ravel()
    def trace(self):return self.diagonal().sum()


class MatterMap(LinearOperator):
    def __init__(self,matrix,nphi,nrotor,nfield,field_matrix=None):
        self.matrix=clean(matrix);self.nphi=nphi;self.nrotor=nrotor;self.nfield=nfield
        self.field_matrix=np.eye(nfield) if field_matrix is None else field_matrix
        scale=nphi*nrotor*nfield
        super().__init__(dtype=np.dtype(complex),shape=(matrix.shape[0]*scale,matrix.shape[1]*scale))
    def _matmat(self,vectors):
        columns=vectors.shape[1];x=(self.matrix@vectors.reshape(self.matrix.shape[1],-1)).reshape(self.matrix.shape[0],self.nphi,self.nrotor,self.nfield,columns)
        y=np.einsum('ab,mfrbc->mfrac',self.field_matrix,x,optimize=True)
        return y.reshape(self.shape[0],columns)
    def _matvec(self,vector):return self._matmat(np.asarray(vector).reshape(-1,1))[:,0]
    def _rmatvec(self,vector):return MatterMap(self.matrix.conj().T,self.nphi,self.nrotor,self.nfield,self.field_matrix.conj().T)@vector


def build(p):
    base=spatial_build(p,with_povm=False);field=field_shell(p['em_cut'],p['em_basis_scale'])
    coefficients=route_coefficients(base['geometry'],base['rotor'],p)
    connections=endpoint_connections(base['geometry'],base['rotor'],p)
    self_kernel=float(gaussian_green(0.,p['charge_sigma'],p['epsilon']))
    cross_kernel=float(gaussian_green(p['vertex_separation'],p['charge_sigma'],p['epsilon']))
    model=dict(base=base,field=field,coefficients=coefficients,connections=connections,parameters=p,
        V_self=self_kernel,V_cross=cross_kernel,phase_cache={},mechanical_cache={})
    return model


def assemble(model,name,derivative=(),external=None,coupled=True,route_derivative=None):
    base=model['base'];field=model['field'];p=model['parameters'];rotor=base['rotor'];geometry=base['geometry'];scalar=base['scalar'];op=base['operators'][name]
    nm=op['Q'].shape[0];nr=rotor['dimension'];ne=field['dimension'];nrq=nr*ne;identity=sparse.eye(nm,format='csr');eye_rq=sparse.eye(nrq,format='csr')
    order=len(derivative)+(route_derivative is not None);terms=[]
    if order==0:
        # Only PHYSICAL EM charge density enters this open-space Coulomb kernel.
        longitudinal=((model['V_self']+model['V_cross'])*(op['Q']@op['Q'])+(model['V_self']-model['V_cross'])*op['density_delta2'])/4
        terms.extend([(op['electric']-p['plaquette']*op['B']+longitudinal,eye_rq,False),
            (identity,sparse.kron(rotor['L2']/(2*p['orientation_inertia']),sparse.eye(ne)),False),
            (identity,sparse.kron(sparse.eye(nr),field['H_E']),False)])
    if coupled and route_derivative is None:
        key=(tuple(derivative),None if external is None else tuple(external))
        if key not in model['mechanical_cache']:
            model['mechanical_cache'][key]=mechanical_kernels(rotor,field,model['connections'],p['orientation_inertia'],external,derivative)
        mechanical=model['mechanical_cache'][key]
        charges=[op['Q'],op['density_delta']]
        for charge,kernel in zip(charges,mechanical['linear']):terms.append((charge,kernel,False))
        squares=[op['Q']@op['Q'],op['Q']@op['density_delta'],op['density_delta2']]
        for charge,kernel in zip(squares,mechanical['quadratic']):terms.append((charge,kernel,False))
    for edge in range(2):
        if route_derivative is not None and edge!=route_derivative:continue
        weight=geometry['weights'][edge]
        for physical_charge,matrices in op['hops'][edge]:
            effective_charge=physical_charge if coupled else 0.
            for spin_axis in [None,0,1,2]:
                key=(edge,effective_charge,tuple(derivative),spin_axis,None if external is None else tuple(external))
                if key not in model['phase_cache']:
                    model['phase_cache'][key]=phase_kernel(rotor,geometry,model['coefficients'],field,edge,effective_charge,derivative,spin_axis,external)
                kernel=model['phase_cache'][key]
                if route_derivative is not None:kernel=1j*effective_charge*kernel
                if spin_axis is None:
                    matter=p['scalar_coupling']*weight*matrices[0];phi=True
                else:
                    matter=1j*p['vector_velocity']*weight*matrices[spin_axis+1];phi=False
                terms.append((matter,kernel,phi));terms.append((matter.conj().T,kernel.conj().T,phi))
    if order==1:terms=[(-a,b,phi) for a,b,phi in terms]
    return FactorOperator(nm,scalar,nrq,terms,include_scalar=(order==0))


def observable(model,name,matter=None,rq=None,scalar_phi=False):
    base=model['base'];op=base['operators'][name];nrq=base['rotor']['dimension']*model['field']['dimension']
    matter=sparse.eye(op['Q'].shape[0]) if matter is None else matter
    rq=sparse.eye(nrq) if rq is None else rq
    return FactorOperator(op['Q'].shape[0],base['scalar'],nrq,[(matter,rq,scalar_phi)])


def angular(model,name,axis):
    base=model['base'];op=base['operators'][name];field=model['field'];nr=base['rotor']['dimension'];ne=field['dimension']
    coordinate=sparse.kron(base['rotor']['L'][axis],sparse.eye(ne))+sparse.kron(sparse.eye(nr),field['F'][axis])
    return FactorOperator(op['Q'].shape[0],base['scalar'],nr*ne,[(op['S'][axis],sparse.eye(nr*ne),False),(sparse.eye(op['Q'].shape[0]),coordinate,False)])


def source(model,name,site=0,spin=.5):
    base=model['base'];kind='baryon' if 'baryon' in name else 'lepton';removal=name.startswith('anti')
    matrix=site_source(base['sectors']['neutral'],base['sectors'][name],site,kind,spin,removal)
    return MatterMap(matrix,len(base['scalar']['grid']),base['rotor']['dimension'],model['field']['dimension'])


def conjugation(model,name):
    base=model['base'];anti='neutral' if name=='neutral' else 'anti'+name
    positive=base['sectors'][name];negative=base['sectors'][anti]
    matrix=negative['P'].conj().T@raw_C(positive['masks'],negative['masks'])@positive['P']
    return MatterMap(matrix,len(base['scalar']['grid']),base['rotor']['dimension'],model['field']['dimension'],model['field']['C'])
