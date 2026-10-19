"""Owning finite physical matrix-frame / rotating-material / literal Gauss law.

No ancestor science imports. Reduced spin matrices are the pullback of LEFT
frame derivatives on I psi(g)=U_F(g^-1)psi, not a second primitive physical spin.
"""
from __future__ import annotations
from itertools import product
import numpy as np
from scipy import sparse as sp

SIGMA = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]], complex), np.diag([1., -1.])]
VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [.35, .8, 0.]])
EDGES = [(0, 1), (1, 2), (2, 0)]
BACKGROUND = np.array([1, 1, 0])


def epsilon(a: int, b: int, c: int) -> int:
    if len({a, b, c}) != 3:
        return 0
    return 1 if (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else -1


def material() -> tuple[list, list]:
    """Exact Haar compression in basis {1,sqrt(3) R_ab}, l=0 plus ALL l=1."""
    position = []
    for a, b in product(range(3), repeat=2):
        r = np.zeros((10, 10))
        r[0, 1+3*a+b] = r[1+3*a+b, 0] = 1/np.sqrt(3)
        for c, d, e, f in product(range(3), repeat=4):
            r[1+3*c+d, 1+3*e+f] = epsilon(c, a, e)*epsilon(d, b, f)/2
        position.append(sp.csr_matrix(r))
    angular = []
    for axis in range(3):
        j = np.zeros((10, 10), complex)
        for a, c, b in product(range(3), repeat=3):
            j[1+3*a+b, 1+3*c+b] = -1j*epsilon(axis, a, c)
        angular.append(sp.csr_matrix(j))
    return position, angular


def car(mask: int, mode: int, create: bool) -> tuple[int, int] | None:
    present = bool(mask & (1 << mode))
    if present == create:
        return None
    sign = (-1)**((mask & ((1 << mode)-1)).bit_count())
    return mask ^ (1 << mode), sign


def transfer(mask: int, outmode: int, inmode: int) -> tuple[int, int] | None:
    first = car(mask, inmode, False)
    if first is None:
        return None
    second = car(first[0], outmode, True)
    return None if second is None else (second[0], first[1]*second[1])


def tensor(*factors):
    answer = sp.csr_matrix([[1.]])
    for factor in factors:
        answer = sp.kron(answer, factor, format='csr')
    return answer


def clean(matrix):
    result = sp.csr_matrix(matrix)
    result.eliminate_zeros()
    return result


class Sector:
    """Complete fixed-N Fock block subject only to literal finite-link Gauss."""
    def __init__(self, number: int, cap: int):
        self.number = number
        self.cap = cap
        self.boundary = 2-number
        self.incidence = np.array([[1, 0, -1], [-1, 1, 0], [0, -1, 1]])
        self.states = []
        for mask in range(64):
            if mask.bit_count() != number:
                continue
            density = np.array([sum(bool(mask & (1 << (2*i+s))) for s in range(2)) for i in range(3)])
            charge = BACKGROUND-density
            for electric in product(range(-cap, cap+1), repeat=3):
                if np.array_equal(self.incidence @ electric + [self.boundary, 0, 0], charge):
                    self.states.append((mask, *electric))
        self.lookup = {state: i for i, state in enumerate(self.states)}
        self.dimension = len(self.states)
        self.electric = np.array([s[1:] for s in self.states], float)
        self.density = np.array([[sum(bool(s[0] & (1 << (2*i+r))) for r in range(2)) for i in range(3)] for s in self.states], float)
        self.charge = BACKGROUND-self.density
        self.doubles = (self.density == 2).sum(axis=1)
        self.holons = ((self.density == 0) & (BACKGROUND[None, :] == 1)).sum(axis=1)
        self.hops = []
        for edge, (i, j) in enumerate(EDGES):
            matrices = []
            for sigma in [np.eye(2)] + SIGMA:
                rows, cols, vals = [], [], []
                for column, state in enumerate(self.states):
                    for s, r in product(range(2), repeat=2):
                        value = sigma[s, r]
                        if value == 0:
                            continue
                        moved = transfer(state[0], 2*j+s, 2*i+r)
                        if moved is None:
                            continue
                        electric = list(state[1:]); electric[edge] += 1
                        row = self.lookup.get((moved[0], *electric))
                        if row is not None:
                            rows.append(row); cols.append(column); vals.append(value*moved[1])
                matrices.append(sp.csr_matrix((vals, (rows, cols)), shape=(self.dimension, self.dimension)))
            self.hops.append(matrices)
        self.spin = []
        for sigma in SIGMA:
            rows, cols, vals = [], [], []
            for column, state in enumerate(self.states):
                for site, s, r in product(range(3), range(2), range(2)):
                    if sigma[s, r] == 0:
                        continue
                    moved = transfer(state[0], 2*site+s, 2*site+r)
                    if moved is not None:
                        rows.append(self.lookup[(moved[0], *state[1:])]); cols.append(column); vals.append(sigma[s, r]*moved[1]/2)
            self.spin.append(sp.csr_matrix((vals, (rows, cols)), shape=(self.dimension, self.dimension)))
        rows, cols = [], []
        for column, state in enumerate(self.states):
            row = self.lookup.get((state[0], *(e+1 for e in state[1:])))
            if row is not None:
                rows.append(row); cols.append(column)
        self.wilson = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.dimension, self.dimension))


def source(target: Sector, reference: Sector, site: int, branch: str, component: int = 0):
    """Boundary-dressed full source; remote insertions carry actual tree strings."""
    create = branch == 'addition'
    mode = 2*site + (component if create else 1-component)
    epsilon_sign = 1 if create or component == 0 else -1
    rows, cols, vals = [], [], []
    for column, state in enumerate(reference.states):
        moved = car(state[0], mode, create)
        if moved is None:
            continue
        electric = list(state[1:])
        for edge in range(site):
            electric[edge] += 1 if create else -1
        row = target.lookup.get((moved[0], *electric))
        if row is not None:
            rows.append(row); cols.append(column); vals.append(epsilon_sign*moved[1])
    return sp.csr_matrix((vals, (rows, cols)), shape=(target.dimension, reference.dimension))


class Model:
    def __init__(self, number: int, parameters: dict):
        self.parameters = parameters
        self.sector = Sector(number, parameters['electric_cap'])
        sector = self.sector
        self.R, self.L = material()
        self.r = np.array(parameters['radial_nodes'], float)
        nr = len(self.r)
        step = self.r[1]-self.r[0]
        if not np.allclose(np.diff(self.r), step):
            raise ValueError('Uniform positive radial u=r psi grid required')
        self.colour_X = sp.diags([np.ones(2), np.ones(2)], [-1, 1], shape=(3, 3), format='csr')
        self.colour_C2 = sp.diags([0., .75, 2.])
        ig, ir, ic = sp.eye(sector.dimension), sp.eye(10), sp.eye(3)
        self.identity = sp.eye(sector.dimension*10*nr*3, format='csr')
        self.shape = (sector.dimension, 10, nr, 3)
        self.dimension = self.identity.shape[0]
        radid = sp.eye(nr)
        self.lift_gauge = lambda a: tensor(a, ir, radid, ic)
        self.lift_material = lambda a: tensor(ig, a, radid, ic)
        self.S = [self.lift_gauge(a) for a in sector.spin]
        self.material_L = [self.lift_material(a) for a in self.L]
        self.J = [clean(a+b) for a, b in zip(self.S, self.material_L)]
        self.S2 = clean(sum(a@a for a in self.S))
        self.J2 = clean(sum(a@a for a in self.J))
        self.E = [self.lift_gauge(sp.diags(sector.electric[:, e])) for e in range(3)]
        self.Q = [self.lift_gauge(sp.diags(sector.charge[:, i])) for i in range(3)]
        self.D = self.lift_gauge(sp.diags(sector.doubles))
        self.holon = self.lift_gauge(sp.diags(sector.holons))
        self.r2 = tensor(ig, ir, sp.diags(self.r**2), ic)
        self.colour_loop = tensor(ig, ir, radid, self.colour_X)
        self.colour_electric = tensor(ig, ir, radid, self.colour_C2)
        self.HE = self.lift_gauge(sp.diags((np.sum(sector.electric**2, axis=1)+sector.boundary**2)/(2*parameters['electric_inertia'])))
        charge = self.lift_gauge(sp.diags(parameters['charge_cost']*np.sum(sector.charge**2, axis=1)/2))
        radial_kinetic = sp.diags([-np.ones(nr-1), 2*np.ones(nr), -np.ones(nr-1)], [-1, 0, 1], format='csr')/(2*parameters['radial_mass']*step**2)
        radial_h = radial_kinetic + sp.diags(parameters['radial_stiffness']*(self.r-parameters['radial_centre'])**2/2)
        material_l2 = sum(a@a for a in self.L)
        spin2 = sum(a@a for a in sector.spin)
        frame = tensor(spin2, ir, sp.diags(1/self.r**2), ic)/(2*parameters['frame_inertia'])
        material_h = tensor(ig, material_l2, radid, ic)/(2*parameters['material_inertia'])
        self.Hcolour = clean(tensor(ig, ir, sp.diags(1/self.r**2), self.colour_C2)/(2*parameters['colour_inertia']) + parameters['colour_potential']*tensor(ig, ir, sp.diags(self.r**2), 2*ic-self.colour_X))
        self.Hstatic = clean(self.HE + charge + tensor(ig, ir, radial_h, ic) + frame + material_h + self.Hcolour)
        colour_transport = ic + parameters['colour_transport']*self.colour_X
        self.scalar_transport, self.vector_transport = [], []
        for edge, (i, j) in enumerate(EDGES):
            direction = VERTICES[j]-VERTICES[i]
            direction /= np.linalg.norm(direction)
            self.scalar_transport.append(parameters['hopping']*tensor(sector.hops[edge][0], ir, sp.diags(self.r), colour_transport))
            vector = sp.csr_matrix(self.identity.shape, dtype=complex)
            for axis in range(3):
                tangent = sum(direction[b]*self.R[3*axis+b] for b in range(3))
                vector += 1j*parameters['vector_hopping']*tensor(sector.hops[edge][axis+1], tangent, sp.diags(self.r), colour_transport)
            self.vector_transport.append(clean(vector))
        self.F = self.lift_gauge((sector.wilson-sector.wilson.conj().T)/(2j))

    def law(self, variant: str = 'full', phase: float = 0., phase_edge: int = 0):
        active = [0, 1, 2] if variant != 'path_cut' else [1]
        transports = []
        hamiltonian = self.Hstatic.copy().astype(complex)
        for edge in range(3):
            a = self.scalar_transport[edge].copy().astype(complex)
            if variant != 'vector_off':
                a += self.vector_transport[edge]
            if edge not in active:
                a = sp.csr_matrix(a.shape, dtype=complex)
            if edge == phase_edge:
                a *= np.exp(1j*phase)
            transports.append(clean(a))
            hamiltonian -= a+a.conj().T
        hamiltonian = clean(hamiltonian)
        currents = [clean(1j*(a-a.conj().T)) for a in transports]
        contacts = [clean(a+a.conj().T) for a in transports]
        return hamiltonian, currents, contacts

    def source_from(self, reference: Model, site: int, branch: str, component: int = 0):
        return tensor(source(self.sector, reference.sector, site, branch, component), sp.eye(10), sp.eye(len(self.r)), sp.eye(3))
