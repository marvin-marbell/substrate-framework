"""Declared spatial source completion; coordinate products precede Haar projection.

L_rot=I omega^2/2+omega.Lambda, Lambda=sum Q x cross A(x),
H_rot=(L-Lambda)^2/(2I). This is an added source-family hypothesis,
not a modification or unique consequence of the frozen zero-field law.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import LinearOperator
from rich_local_frame_carrier_20261015_model import VERTICES, EDGES


@lru_cache(maxsize=4)
def quadrature(order=8):
    """Normalized SO(3) Haar Euler quadrature, actual old material basis."""
    z, w = np.polynomial.legendre.leggauss(order)
    rotations, weights = [], []
    for a in np.arange(order)*2*np.pi/order:
        ca, sa = np.cos(a), np.sin(a)
        rz = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1.]])
        for c, wc in zip(z,w):
            s = np.sqrt(1-c*c)
            ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
            for g in np.arange(order)*2*np.pi/order:
                cg, sg = np.cos(g), np.sin(g)
                rotations.append(rz@ry@np.array([[cg,-sg,0],[sg,cg,0],[0,0,1.]]))
                weights.append(wc/(2*order**2))
    r = np.asarray(rotations); weights = np.asarray(weights)
    basis = np.column_stack([np.ones(len(r)), np.sqrt(3)*r.reshape(-1,9)])
    return r, weights, basis


def project(values, order=8):
    _, weights, basis = quadrature(order)
    result = np.einsum('qi,q,qj->ij', basis, weights*values, basis, optimize=True)
    result[abs(result)<2e-13] = 0
    return result


class Terms(LinearOperator):
    """Tensor sum retaining every gauge, frame and radial amplitude."""
    def __init__(self, model, terms, base=None):
        self.model, self.terms, self.base = model, terms, base
        super().__init__(dtype=np.dtype(complex), shape=(model.dimension,)*2)

    def _matvec(self, vector):
        m = self.model; n = m.sector.dimension; nr = m.radial_dimension
        v = np.asarray(vector).reshape(n,10,nr)
        out = np.zeros_like(v, dtype=complex)
        for gauge, material, radial in self.terms:
            if sp.issparse(gauge):
                work = (gauge@v.reshape(n,-1)).reshape(n,10,nr)
            else:
                work = gauge[:,None,None]*v
            work = np.einsum('ab,nbk->nak', material, work, optimize=True)
            out += work*radial
        result = out.ravel()
        if self.base is not None:
            result += self.base@np.asarray(vector).reshape(-1)
        return result

    def _rmatvec(self, vector):
        return self._matvec(vector)


class SpatialFamily:
    def __init__(self, model, direction, profile='B', quadrature_order=8):
        self.model = model; self.direction = np.asarray(direction,float)
        self.profile = profile; self.order = quadrature_order
        self.rotations, self.weights, self.basis = quadrature(self.order)
        self.x = np.einsum('qab,vb->qva',self.rotations,VERTICES)
        if profile == 'B':
            self.A = np.cross(self.direction,self.x)/2
        elif profile == 'constant':
            self.A = np.broadcast_to(self.direction,self.x.shape)
        else:
            raise ValueError(profile)
        self.connection = np.cross(self.x,self.A)
        self.phases = []
        for i,j in EDGES:
            # Both allowed potentials are affine, so midpoint integration exact.
            self.phases.append(np.sum((self.A[:,i]+self.A[:,j])/2*(self.x[:,j]-self.x[:,i]),axis=1))
        self.h0 = model.law()[0]

    def project(self, values):
        return project(values,self.order)

    def _hopping(self, amplitude, derivative=0):
        m = self.model; terms = []
        for e,(i,j) in enumerate(EDGES):
            phase = self.phases[e]
            factor = np.exp(-1j*amplitude*phase)*(-1j*phase)**derivative
            if derivative == 0:
                factor -= 1
            radial = ((m.radial_coordinates[i]+m.radial_coordinates[j])/2).diagonal()[None,None,:]
            direction = VERTICES[j]-VERTICES[i]; direction /= np.linalg.norm(direction)
            components = [(m.parameters['hopping']*m.sector.hop(e,np.eye(4)),np.ones(len(phase)))]
            for axis in range(3):
                tangent = self.rotations[:,axis,:]@direction
                components.append((1j*m.parameters['vector_hopping']*m.sector.hop(e,m.architecture['gamma'][axis]),tangent))
            for gauge, coordinate in components:
                mat = self.project(factor*coordinate)
                terms += [(-gauge,mat,radial),(-gauge.conj().T,mat.conj().T,radial)]
        return terms

    def _mechanical(self, amplitude, derivative=0, compressed_square=False):
        m = self.model; terms = []; inertia = m.parameters['material_inertia']
        linear = amplitude if derivative == 0 else (1 if derivative == 1 else 0)
        quadratic = amplitude**2 if derivative == 0 else (2*amplitude if derivative == 1 else 2)
        for v in range(3):
            q = m.sector.charge[:,v]
            for a in range(3):
                lam = self.project(self.connection[:,v,a]); angular = m.L[a].toarray()
                if linear:
                    terms.append((q,-linear*(angular@lam+lam@angular)/(2*inertia),1))
            for u in range(3):
                if not quadratic:
                    continue
                if compressed_square:
                    product = sum(self.project(self.connection[:,v,a])@self.project(self.connection[:,u,a]) for a in range(3))
                else:
                    product = self.project(np.sum(self.connection[:,v]*self.connection[:,u],axis=1))
                terms.append((q*m.sector.charge[:,u],quadratic*product/(2*inertia),1))
        return terms

    def law(self, amplitude=0.):
        return Terms(self.model,self._hopping(amplitude)+self._mechanical(amplitude),self.h0)

    def derivatives(self):
        hop = Terms(self.model,self._hopping(0,1))
        mechanical = Terms(self.model,self._mechanical(0,1))
        first = Terms(self.model,self._hopping(0,1)+self._mechanical(0,1))
        contact = Terms(self.model,self._hopping(0,2)+self._mechanical(0,2))
        return first,contact,hop,mechanical

    def wrong_contact(self):
        return Terms(self.model,self._hopping(0,2)+self._mechanical(0,2,True))

    def dipole(self):
        values = self.x@self.direction
        return Terms(self.model,[(self.model.sector.charge[:,v],self.project(values[:,v]),1) for v in range(3)])
