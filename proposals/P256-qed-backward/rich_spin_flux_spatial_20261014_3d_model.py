"""NEW three-dimensional two-plane ladder, literal common Gauss material.
Reuses only CAR routines and full-state operator methods, not any prior state/D.
"""
from itertools import product
import numpy as np
from scipy import sparse
from rich_spin_flux_spatial_20261014_model import Parameters, Sector as OperatorMethods, fock_states, hop

LAW = '3D-two-plane-two-cell-CAR-orientor-supplier-common-Gauss-electric-v3-20261014'


def geometry():
    positions = np.array([(x, int(r == 1), int(r == 2)) for x in range(3) for r in range(3)], float)
    edges = [(3*x+r, 3*(x+1)+r) for x in range(2) for r in range(3)]
    edges += [(3*x, 3*x+r) for x in range(3) for r in (1, 2)]
    B = np.zeros((9, 12), int)
    for l, (a, b) in enumerate(edges):
        B[a, l], B[b, l] = 1, -1
    faces = []
    for x in range(2):
        for r in (1, 2):
            face = np.zeros(12, int)
            face[3*x] = 1
            face[3*x+r] = -1
            face[6+2*(x+1)+r-1] = 1
            face[6+2*x+r-1] = -1
            faces.append(face)
    tree = [0, 3, 6, 7, 8, 9, 10, 11]
    chords = [1, 2, 4, 5]
    return positions, edges, B, np.array(faces), tree, chords


class Sector(OperatorMethods):
    """Inherited operations are pure common-basis algebra; init builds NEW joint H."""
    def __init__(self, p=Parameters(), up=1, down=0, supplier=1):
        self.p, self.up, self.down, self.supplier = p, up, down, supplier
        self.positions, self.edges, self.B, self.faces, self.tree, self.chords = geometry()
        vertices, links = len(self.positions), len(self.edges)
        self.vertices, self.links = vertices, links
        self.background = np.zeros(vertices, int)
        self.background[0] = up+down+supplier
        chord_values = np.array(list(product(range(-p.bound, p.bound+1), repeat=len(self.chords))), int)
        inverse = np.linalg.inv(self.B[np.ix_(range(1, vertices), self.tree)])
        cycle_tree = -self.B[np.ix_(range(1, vertices), self.chords)].T@inverse.T
        basis, electric = [], []
        for bits in fock_states(up, down, supplier, vertices):
            rho = np.array([((bits >> (2*x)) & 1)+((bits >> (2*x+1)) & 1)+((bits >> (2*vertices+x)) & 1) for x in range(vertices)])-self.background
            fields = np.zeros((len(chord_values), links), int)
            fields[:, self.tree] = np.rint((-rho[1:])@inverse.T+chord_values@cycle_tree).astype(int)
            fields[:, self.chords] = chord_values
            for flux, field in zip(chord_values, fields):
                if np.max(np.abs(field)) <= p.bound:
                    for r in range(3):
                        basis.append((bits, r, *flux)); electric.append(field)
        self.basis = basis
        self.lookup = {b: i for i,b in enumerate(basis)}
        self.dimension = len(basis)
        self.bits = np.array([b[0] for b in basis], np.int64)
        self.r = np.array([b[1] for b in basis])
        self.electric = np.array(electric)
        self.active_density = np.array([[((int(bits) >> (2*x)) & 1)+((int(bits) >> (2*x+1)) & 1) for x in range(vertices)] for bits in self.bits])
        self.supplier_density = np.array([[(int(bits) >> (2*vertices+x)) & 1 for x in range(vertices)] for bits in self.bits])
        self.rho = self.active_density+self.supplier_density-self.background
        self.electric_energy = np.sum(self.electric**2, axis=1)/(2*p.inertia)
        self.alignment_energy = -p.alignment*sum(self.active_density[:, site]*np.cos(2*np.pi*(angle-self.r)/3) for angle,site in enumerate([0, 1, 2]))
        self.H = sparse.diags(self.electric_energy+self.alignment_energy, format='csr', dtype=complex)
        self.transfers = {'active': [], 'supplier': []}
        self.cut_edges = [3, 4, 5]
        for species, strength, spins in [('active', p.active_hop, (0, 1)), ('supplier', p.supplier_hop, (0,))]:
            for l,(a,b) in enumerate(self.edges):
                rows, cols, values = [], [], []
                if not (p.cut and l in self.cut_edges):
                    for col,state in enumerate(basis):
                        for spin in spins:
                            origin = 2*a+spin if species == 'active' else 2*vertices+a
                            destination = 2*b+spin if species == 'active' else 2*vertices+b
                            moved = hop(state[0], destination, origin)
                            if moved is None:
                                continue
                            flux = list(state[2:])
                            if l in self.chords:
                                flux[self.chords.index(l)] += 1
                            row = self.lookup.get((moved[0], state[1], *flux))
                            if row is not None:
                                rows.append(row); cols.append(col); values.append(-strength*moved[1])
                t = sparse.csr_matrix((values,(rows,cols)), shape=self.H.shape, dtype=complex)
                self.transfers[species].append(t)
                self.H += t+t.getH()
        rows = [self.lookup[(b[0], (b[1]+1) % 3, *b[2:])] for b in basis]
        orientor = sparse.csr_matrix((np.ones(self.dimension),(rows,range(self.dimension))), shape=self.H.shape)
        self.Ln = 1j*(orientor-orientor.getH())/np.sqrt(3)
        self.H += -p.neutral_hop*(orientor+orientor.getH())+p.torque*((up-down)/2)*self.Ln
        self.H = self.H.tocsr()
        self.currents = {s: [-1j*(t-t.getH()) for t in ts] for s,ts in self.transfers.items()}
        self.contacts = {s: [-(t+t.getH()) for t in ts] for s,ts in self.transfers.items()}
        self.total_currents = [self.currents['active'][l]+self.currents['supplier'][l] for l in range(links)]
        self.W = [self.wilson(f) for f in self.faces]
        self.F = [(w-w.getH())/(2j) for w in self.W]
        self.C = [(w+w.getH())/2 for w in self.W]
        self.loop_electric = [sparse.diags(self.electric@f/4,format='csr') for f in self.faces]
        self.body_x = sparse.diags(self.active_density@self.positions[:,0],format='csr')
        self.body_velocity = 1j*(self.H@self.body_x-self.body_x@self.H)
        self.neutral_torque = 1j*(self.H@self.Ln-self.Ln@self.H)
        self.vector_current = [sum((self.positions[b,axis]-self.positions[a,axis])*self.total_currents[l] for l,(a,b) in enumerate(self.edges)) for axis in range(3)]
