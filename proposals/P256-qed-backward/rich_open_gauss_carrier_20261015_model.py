"""Owning finite outward-boundary quantum-link model; no historical state import.

All source and hopping products are formed before physical projection. The
primitive two CAR components are NOT a derived spatial Noether spin embedding.
"""
from __future__ import annotations
import itertools
import hashlib
import numpy as np
from scipy import sparse
from rich_photon_fullq_20261014_model import geometry, fock_basis
from rich_photon_20261014_loop import hop

LAW_ID = 'open-gauss-cube-qlink3-I16-t1-U4-20261015'

def maximum(a):
    return float(np.max(np.abs(a.data))) if a.nnz else 0.0

def commutator(a, b):
    return a @ b - b @ a

class NeutralCube:
    """Complete block, neutral by default; boundary link at site zero."""
    def __init__(self, up=4, down=4):
        self.up, self.down = up, down
        self.N = up + down
        if self.N not in (7, 8, 9):
            raise ValueError('Finite boundary admits N7/8/9 only')
        self.positions, self.edges, self.tree, self.chords, self.incidence, self.faces = geometry('cube')
        self.boundary_flux = 8-self.N
        self.fock = np.asarray(fock_basis(8, up, down), dtype=np.int64)
        self.fi = {int(b): i for i, b in enumerate(self.fock)}
        self.flux = np.asarray(list(itertools.product((-1, 0, 1), repeat=5)), dtype=np.int8)
        self.nflux = len(self.flux)
        self.mi = {tuple(m): i for i, m in enumerate(self.flux)}
        occupation = np.asarray([[(int(b)>>(2*i)&1)+(int(b)>>(2*i+1)&1) for i in range(8)] for b in self.fock], dtype=np.int8)
        inverse = np.linalg.inv(self.incidence[np.ix_(range(1,8), self.tree)])
        base = -(occupation[:,1:]-1) @ inverse.T
        cycle = -self.incidence[np.ix_(range(1,8), self.chords)].T @ inverse.T
        solved = np.rint(base[:,None,:]+(self.flux@cycle)[None,:,:]).astype(np.int8)
        self.old_indices = np.flatnonzero(np.all(np.abs(solved)<=1, axis=2).ravel())
        self.dimension = len(self.old_indices)
        self.lookup = np.full(len(self.fock)*self.nflux, -1, dtype=np.int32)
        self.lookup[self.old_indices] = np.arange(self.dimension)
        self.fock_id, self.flux_id = np.divmod(self.old_indices,self.nflux)
        self.electric = np.empty((self.dimension,13), dtype=np.int8)
        self.electric[:,self.tree] = solved.reshape(-1,7)[self.old_indices]
        self.electric[:,self.chords] = self.flux[self.flux_id]
        self.electric[:,12] = self.boundary_flux
        self.full_incidence = np.column_stack((self.incidence, np.eye(8)[:,0]))
        self.density = occupation[self.fock_id]
        self.rho = self.density-1
        self.doubles = np.asarray([[(int(b)>>(2*i)&1)*(int(b)>>(2*i+1)&1) for i in range(8)] for b in self.fock], dtype=np.int8)[self.fock_id]
        self.holons = (self.density==0).astype(np.int8)
        self.left_sites = np.flatnonzero(self.positions[:,0]==0)
        self.right_sites = np.flatnonzero(self.positions[:,0]==1)
        self.left_cycle = next(c for c in self.faces if np.all(self.positions[np.flatnonzero(np.any(self.incidence[:,np.flatnonzero(c)]!=0,axis=1)),0]==0))
        self.left_edges = np.flatnonzero(self.left_cycle)
        self.right_edges = np.asarray([e for e,(a,b) in enumerate(self.edges) if self.positions[a,0]==self.positions[b,0]==1], dtype=int)
        self.bridges = np.asarray([e for e,(a,b) in enumerate(self.edges) if self.positions[a,0]!=self.positions[b,0]], dtype=int)
        self.weights = np.asarray([(self.positions[a,0]+self.positions[b,0])/2 for a,b in self.edges]+[0.0])
        self.transfers = []
        for edge,(a,b) in enumerate(self.edges):
            shift = np.zeros(5,dtype=int)
            if edge in self.chords:
                shift[self.chords.index(edge)] = 1
            fm = np.asarray([self.mi.get(tuple(m+shift),-1) for m in self.flux],dtype=int)
            rows, cols, values = [], [], []
            for spin in (0,1):
                targets = np.full(len(self.fock),-1,dtype=int)
                signs = np.zeros(len(self.fock),dtype=int)
                for k,bits in enumerate(self.fock):
                    result = hop(int(bits),2*b+spin,2*a+spin)
                    if result is not None:
                        targets[k],signs[k] = self.fi[result[0]],result[1]
                source = np.flatnonzero((targets[self.fock_id]>=0)&(fm[self.flux_id]>=0))
                target = self.lookup[targets[self.fock_id[source]]*self.nflux+fm[self.flux_id[source]]]
                good = target>=0
                rows.extend(target[good]); cols.extend(source[good]); values.extend(-signs[self.fock_id[source[good]]])
            self.transfers.append(sparse.csr_matrix((values,(rows,cols)),shape=(self.dimension,self.dimension),dtype=float))
        self.currents = [-1j*(t-t.T) for t in self.transfers]
        self.contacts = [-(t+t.T) for t in self.transfers]
        self.HE = np.sum(self.electric.astype(float)**2,axis=1)/32
        self.HU = 2*np.sum(self.rho.astype(float)**2,axis=1)
        self.Hoff = sparse.diags(self.HE+self.HU,format='csr')
        zero = sparse.csr_matrix(self.Hoff.shape)
        self.H = self.Hoff+sum((t+t.T for t in self.transfers),zero)
        self.ER = np.sum(self.electric[:,self.right_edges].astype(float)**2,axis=1)/32
        self.HRoff = sparse.diags(np.sum(self.electric.astype(float)**2*self.weights,axis=1)/32+2*np.sum(self.rho[:,self.right_sites].astype(float)**2,axis=1),format='csr')
        self.HR = self.HRoff+sum((w*(t+t.T) for w,t in zip(self.weights,self.transfers)),zero)
        self.Hcut = self.H-sum((self.transfers[e]+self.transfers[e].T for e in self.bridges),zero)
        self.HRcut = self.HRoff+sum((self.transfers[e]+self.transfers[e].T for e in self.right_edges),zero)
        self.W = self.loop(self.left_cycle)
        self.source = (self.W+self.W.T)/2
        self.basis_hash = hashlib.sha256(self.fock.tobytes()+self.old_indices.tobytes()+self.electric.tobytes()).hexdigest()

    def loop(self, cycle):
        mapping = np.asarray([self.mi.get(tuple(m+cycle[self.chords]),-1) for m in self.flux],dtype=int)
        source = np.flatnonzero(mapping[self.flux_id]>=0)
        target = self.lookup[self.fock_id[source]*self.nflux+mapping[self.flux_id[source]]]
        good = target>=0
        return sparse.csr_matrix((np.ones(np.sum(good)),(target[good],source[good])),shape=self.H.shape)

    def seed(self):
        if (self.up,self.down)!=(4,4):
            raise ValueError('Singlet seed is neutral only')
        state = np.zeros(self.dimension)
        for choices in itertools.product((0,1),repeat=4):
            bits,sign = 0,1
            for k,choice in enumerate(choices):
                a,b = 2*k,2*k+1
                bits |= (1<<(2*a+choice))|(1<<(2*b+1-choice))
                sign *= (-1)**choice
            state[self.lookup[self.fi[bits]*self.nflux+self.mi[(0,)*5]]] = sign/4
        return state

    def path(self, site):
        """Deterministic tree path from boundary anchor to physical site."""
        stack = [(0, np.zeros(12,dtype=int))]
        visited = set()
        while stack:
            vertex,shift = stack.pop()
            if vertex==site:
                return shift
            visited.add(vertex)
            for e in self.tree:
                a,b = self.edges[e]
                if vertex in (a,b):
                    other = b if vertex==a else a
                    if other not in visited:
                        new = shift.copy(); new[e] += 1 if vertex==a else -1
                        stack.append((other,new))
        raise ValueError(site)

    def charged_source(self, target, site, spin=0, creation=True):
        """Rectangular raw Wilson-CAR product; finite ladder is not unitary."""
        delta = 1 if creation else -1
        expected = (self.up+delta*(spin==0),self.down+delta*(spin==1))
        if (target.up,target.down)!=expected:
            raise ValueError('Wrong CAR target block')
        shift = delta*self.path(site)
        rows,cols,values = [],[],[]
        orbital = 2*site+spin
        fm = np.asarray([target.mi.get(tuple(m+shift[self.chords]),-1) for m in self.flux],dtype=int)
        for f,bits0 in enumerate(self.fock):
            bits = int(bits0)
            if bool(bits>>orbital&1)==creation:
                continue
            changed = bits^(1<<orbital)
            sign = (-1)**((bits&((1<<orbital)-1)).bit_count())
            source = np.flatnonzero((self.fock_id==f)&(fm[self.flux_id]>=0))
            destination = target.lookup[target.fi[changed]*self.nflux+fm[self.flux_id[source]]]
            good = destination>=0
            # Explicit full physical product capacity, including boundary link.
            good &= np.all(np.abs(self.electric[source,:12]+shift)<=1,axis=1)
            good &= abs(self.boundary_flux-delta)<=1
            rows.extend(destination[good]);cols.extend(source[good]);values.extend([sign]*int(np.sum(good)))
        return sparse.csr_matrix((values,(rows,cols)),shape=(target.dimension,self.dimension),dtype=float)

    def measure(self, state):
        p = np.abs(state)**2
        return dict(norm=float(p.sum()),energy=float(np.vdot(state,self.H@state).real),rho=(p@self.rho).tolist(),electric_flux=(p@self.electric).tolist(),physical_field=(p@self.electric/16).tolist(),doublon=(p@self.doubles).tolist(),holon=(p@self.holons).tolist(),current=[float(np.vdot(state,j@state).real) for j in self.currents],right_charge=float(p@np.sum(self.rho[:,self.right_sites],axis=1)),right_electric_energy=float(p@self.ER),right_owned_energy=float(np.vdot(state,self.HR@state).real))

    def algebra(self):
        residual = dict(gauss=float(np.max(np.abs(self.electric@self.full_incidence.T+self.rho))),hermiticity=maximum(self.H-self.H.T),flux_current=0.0,continuity=0.0)
        for e,j in enumerate(self.currents):
            E = sparse.diags(self.electric[:,e])
            residual['flux_current'] = max(residual['flux_current'],maximum(1j*commutator(self.H,E)-j))
        for site in range(8):
            n = sparse.diags(self.rho[:,site])
            div = sum((self.incidence[site,e]*j for e,j in enumerate(self.currents)),sparse.csr_matrix(self.H.shape))
            residual['continuity'] = max(residual['continuity'],maximum(1j*commutator(self.H,n)+div))
        return residual
