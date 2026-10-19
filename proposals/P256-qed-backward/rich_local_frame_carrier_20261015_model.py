"""Product physical-frame, full CAR12, local cone fields and literal U(1) law.

Pure angular/frame helper imports supply fresh operators, never old states/laws.
The displayed Pauli-equivalent matrices are measured LEFT-derivative pullbacks.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
import hashlib
import json
import numpy as np
from scipy import sparse as sp
from rich_frame_holonomy_20261014_model import raw_frame
from rich_frame_gauss_join_20261015_model import material

VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [.35, .8, 0.]])
EDGES = [(0, 1), (1, 2), (2, 0)]
INCIDENCE = np.array([[1, 0, -1], [-1, 1, 0], [0, -1, 1]])
DEFAULTS = dict(electric_cap=2, electric_inertia=16., charge_cost=.35,
                radial_nodes=[.55, 1., 1.45], radial_mass=2.,
                radial_stiffness=3., radial_centre=1., material_inertia=4.,
                root_coupling=1., hopping=1., vector_hopping=.65)


def clean(value):
    value = sp.csr_matrix(value)
    value.data[abs(value.data) < 2e-13] = 0
    value.eliminate_zeros()
    return value


def tensor(*values):
    result = sp.csr_matrix([[1.]])
    for value in values:
        result = sp.kron(result, value, format='csr')
    return result


def car(mask, mode, create):
    if bool(mask & (1 << mode)) == create:
        return None
    return mask ^ (1 << mode), (-1)**((mask & ((1 << mode)-1)).bit_count())


def transfer(mask, outmode, inmode):
    first = car(mask, inmode, False)
    if first is None:
        return None
    last = car(first[0], outmode, True)
    return None if last is None else (last[0], first[1]*last[1])


@lru_cache(maxsize=1)
def architecture():
    """Fresh raw D(g)c contraction, RIGHT Gauss and actual LEFT generators."""
    frame = raw_frame()
    creators = [clean(a).toarray() for a in frame['source']]
    spin = [clean(a).toarray() for a in frame['S']]
    one = [a[np.ix_([1, 2, 4, 8], [1, 2, 4, 8])] for a in spin]
    # Constitutive ordered orbit: Gamma0 = orbital X; Gamma_i = orbital Z
    # times twice the measured one-root LEFT derivative, NOT supplied free H.
    gamma0 = np.kron(np.array([[0., 1.], [1., 0.]]), np.eye(2))
    gamma = [np.kron(np.diag([1., -1.]), 2*a[:2, :2]) for a in one]
    mix = sum(gamma0[a, b]*creators[a]@creators[b].conj().T
              for a, b in product(range(4), repeat=2))
    p = frame['P']
    return dict(frame=frame, create=creators, spin=spin, one_spin=one,
                gamma0=gamma0, gamma=gamma, mix=clean(mix),
                S2=clean(sum(a@a for a in spin)),
                provenance=dict(raw_dimension=224, physical_dimension=16,
                    haar_gram_error=float(np.linalg.norm(p.conj().T@p-np.eye(16))),
                    right_gauss_error=max(float(np.linalg.norm(g@p)) for g in frame['G']),
                    CAR_error=max(float(np.linalg.norm(a.conj().T@b+b@a.conj().T-(np.eye(16) if i == j else 0)))
                                  for i, a in enumerate(creators) for j, b in enumerate(creators)),
                    clifford_error=max(float(np.linalg.norm(a@b+b@a-(2*np.eye(4) if i == j else 0)))
                                        for i, a in enumerate([gamma0]+gamma) for j, b in enumerate([gamma0]+gamma))))


class Sector:
    """ALL fixed-N CAR12 configurations and ALL finite-link Gauss solutions."""
    def __init__(self, number, cap=2):
        self.number, self.cap, self.boundary = number, cap, 6-number
        self.states = []
        for mask in range(1 << 12):
            if mask.bit_count() != number:
                continue
            density = np.array([((mask >> (4*v)) & 15).bit_count() for v in range(3)])
            q = 2-density
            # One independent cycle electric integer; solve tree exactly.
            for e2 in range(-cap, cap+1):
                e0 = q[0]-self.boundary+e2
                e1 = q[1]+e0
                if abs(e0) <= cap and abs(e1) <= cap:
                    self.states.append((mask, int(e0), int(e1), e2))
        self.lookup = {state: i for i, state in enumerate(self.states)}
        self.dimension = len(self.states)
        self.electric = np.array([s[1:] for s in self.states], float)
        self.density = np.array([[((s[0] >> (4*v)) & 15).bit_count() for v in range(3)] for s in self.states], float)
        self.charge = 2-self.density
        self.basis_hash = hashlib.sha256(np.array(self.states, np.int16).tobytes()).hexdigest()
        rows, cols = [], []
        for col, state in enumerate(self.states):
            row = self.lookup.get((state[0], *(e+1 for e in state[1:])))
            if row is not None:
                rows.append(row); cols.append(col)
        self.wilson = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.dimension,)*2)

    def local(self, site, matrix):
        matrix = sp.csc_matrix(matrix)
        rows, cols, values = [], [], []
        for col, state in enumerate(self.states):
            local = (state[0] >> (4*site)) & 15
            for index in range(matrix.indptr[local], matrix.indptr[local+1]):
                target = int(matrix.indices[index])
                mask = (state[0] & ~(15 << (4*site))) | (target << (4*site))
                row = self.lookup.get((mask, *state[1:]))
                if row is not None:
                    rows.append(row); cols.append(col); values.append(matrix.data[index])
        return clean(sp.csr_matrix((values, (rows, cols)), shape=(self.dimension,)*2))

    def hop(self, edge, matrix):
        origin, target = EDGES[edge]
        entries = [(a, b, matrix[a, b]) for a, b in product(range(4), repeat=2) if abs(matrix[a, b]) > 2e-13]
        rows, cols, values = [], [], []
        for col, state in enumerate(self.states):
            electric = list(state[1:]); electric[edge] += 1
            for a, b, value in entries:
                step = transfer(state[0], 4*target+a, 4*origin+b)
                if step is None:
                    continue
                row = self.lookup.get((step[0], *electric))
                if row is not None:
                    rows.append(row); cols.append(col); values.append(value*step[1])
        return clean(sp.csr_matrix((values, (rows, cols)), shape=(self.dimension,)*2))


def source(target, reference, site, branch, component=0, orbital=0):
    """Dressed physical B_v, literal boundary Wilson tree and exact CAR sign.

    Removal spinor uses epsilon contraction: component0 removes down,
    component1 minus up. It is a hole source, not a measured positron.
    """
    create = branch == 'addition'
    mode = 4*site+2*orbital+(component if create else 1-component)
    epsilon_sign = 1 if create or component == 0 else -1
    rows, cols, values = [], [], []
    for col, state in enumerate(reference.states):
        step = car(state[0], mode, create)
        if step is None:
            continue
        electric = list(state[1:])
        for edge in range(site):
            electric[edge] += 1 if create else -1
        row = target.lookup.get((step[0], *electric))
        if row is not None:
            rows.append(row); cols.append(col); values.append(step[1]*epsilon_sign)
    return sp.csr_matrix((values, (rows, cols)), shape=(target.dimension, reference.dimension))


class Model:
    def __init__(self, number, parameters=None):
        self.parameters = dict(DEFAULTS if parameters is None else parameters)
        p = self.parameters
        self.architecture = architecture()
        self.sector = s = Sector(number, p['electric_cap'])
        self.R, self.L = material()
        self.r = np.array(p['radial_nodes'], float)
        nr = len(self.r); self.radial_dimension = nr**3
        self.shape = (s.dimension, 10, nr, nr, nr)
        self.dimension = int(np.prod(self.shape))
        self.identity = sp.eye(self.dimension, format='csr')
        ig, im, ir = sp.eye(s.dimension), sp.eye(10), sp.eye(self.radial_dimension)
        self.lift_gauge = lambda a: tensor(a, im, ir)
        self.lift_material = lambda a: tensor(ig, a, ir)
        self.lift_radial = lambda a: tensor(ig, im, a)
        eye = sp.eye(nr)
        radial = lambda site, a: tensor(*[a if v == site else eye for v in range(3)])
        self.radial_coordinates = [radial(v, sp.diags(self.r)) for v in range(3)]
        self.radial_squared = [radial(v, sp.diags(self.r**2)) for v in range(3)]
        self.r2_local = [self.lift_radial(a) for a in self.radial_squared]
        self.n = [self.lift_gauge(sp.diags(s.density[:, v])) for v in range(3)]
        self.Q = [self.lift_gauge(sp.diags(s.charge[:, v])) for v in range(3)]
        self.E = [self.lift_gauge(sp.diags(s.electric[:, e])) for e in range(3)]
        self.spin_gauge = [[s.local(v, a) for a in self.architecture['spin']] for v in range(3)]
        self.spin2_gauge = [s.local(v, self.architecture['S2']) for v in range(3)]
        self.S_local = [[self.lift_gauge(a) for a in site] for site in self.spin_gauge]
        self.S2_local = [self.lift_gauge(a) for a in self.spin2_gauge]
        self.S = [clean(sum(site[a] for site in self.S_local)) for a in range(3)]
        self.material_L = [self.lift_material(a) for a in self.L]
        self.J = [clean(a+b) for a, b in zip(self.S, self.material_L)]
        self.F = self.lift_gauge((s.wilson-s.wilson.conj().T)/(2j))
        self.Wcos = self.lift_gauge((s.wilson+s.wilson.conj().T)/2)
        self.HE = self.lift_gauge(sp.diags((np.sum(s.electric**2, axis=1)+s.boundary**2)/(2*p['electric_inertia'])))
        charge = self.lift_gauge(sp.diags(p['charge_cost']*np.sum(s.charge**2, axis=1)/2))
        step = self.r[1]-self.r[0]
        if min(self.r) <= 0 or not np.allclose(np.diff(self.r), step):
            raise ValueError('Positive uniform interior u=r^(3/2)f radial grid required')
        kinetic = sp.diags([-np.ones(nr-1), 2*np.ones(nr), -np.ones(nr-1)], [-1, 0, 1], format='csr')/(2*p['radial_mass']*step**2)
        potential = sp.diags(p['radial_stiffness']*(self.r-p['radial_centre'])**2/2+3/(8*p['radial_mass']*self.r**2))
        self.Hradial = self.lift_radial(sum(radial(v, kinetic+potential) for v in range(3)))
        self.Hframe = clean(sum(tensor(self.spin2_gauge[v], im, radial(v, sp.diags(1/self.r**2)))/p['radial_mass'] for v in range(3)))
        self.Hroot = clean(sum(p['root_coupling']*tensor(s.local(v, self.architecture['mix']), im, self.radial_coordinates[v]) for v in range(3)))
        self.Hmaterial = self.lift_material(sum(a@a for a in self.L)/(2*p['material_inertia']))
        self.Hstatic = clean(self.HE+charge+self.Hradial+self.Hframe+self.Hroot+self.Hmaterial)
        self.scalar_transport, self.vector_transport = [], []
        for edge, (i, j) in enumerate(EDGES):
            amplitude = (self.radial_coordinates[i]+self.radial_coordinates[j])/2
            self.scalar_transport.append(p['hopping']*tensor(s.hop(edge, np.eye(4)), im, amplitude))
            direction = VERTICES[j]-VERTICES[i]; direction /= np.linalg.norm(direction)
            vector = sp.csr_matrix((self.dimension,)*2, dtype=complex)
            for axis in range(3):
                tangent = sum(direction[b]*self.R[3*axis+b] for b in range(3))
                vector += 1j*p['vector_hopping']*tensor(s.hop(edge, self.architecture['gamma'][axis]), tangent, amplitude)
            self.vector_transport.append(clean(vector))
        self.basis_hash = hashlib.sha256(json.dumps(dict(gauge=s.basis_hash, shape=self.shape, parameters=p), sort_keys=True).encode()).hexdigest()

    def law(self, variant='full', phase=0., phase_edge=0):
        if variant not in ('full', 'path_cut', 'vector_off', 'all_hop_off'):
            raise ValueError(variant)
        h = self.Hstatic.copy().astype(complex)
        forwards = []
        for e in range(3):
            a = self.scalar_transport[e].copy().astype(complex)
            if variant != 'vector_off':
                a += self.vector_transport[e]
            if variant == 'all_hop_off' or (variant == 'path_cut' and e != 1):
                a = sp.csr_matrix(a.shape, dtype=complex)
            if e == phase_edge:
                a *= np.exp(-1j*phase)
            a = clean(a); forwards.append(a); h -= a+a.conj().T
        return clean(h), [clean(-1j*(a-a.conj().T)) for a in forwards], [clean(a+a.conj().T) for a in forwards]

    def source_from(self, reference, site, branch, component=0, orbital=0):
        return tensor(source(self.sector, reference.sector, site, branch, component, orbital), sp.eye(10), sp.eye(self.radial_dimension))

    def count_projector(self, site, count):
        return self.lift_gauge(sp.diags((self.sector.density[:, site] == count).astype(float)))

    def local_spin_projector(self, site, j):
        """Exact local S_v² spectral PVM (j=0,1/2,1), all orbital counts."""
        casimir = self.architecture['S2'].toarray()
        values, vectors = np.linalg.eigh(casimir)
        selected = vectors[:, abs(values-j*(j+1)) < 1e-9]
        return self.lift_gauge(self.sector.local(site, selected@selected.conj().T))

    def root_band_count(self, site, signed):
        gamma = self.architecture['gamma0']
        projector = (np.eye(4)+signed*gamma)/2
        creators = self.architecture['create']
        op = sum(projector[a, b]*creators[a]@creators[b].conj().T for a, b in product(range(4), repeat=2))
        return self.lift_gauge(self.sector.local(site, op))


def prepared_reference(model):
    """NEW owning INPUT, not stationary or global-vacuum theorem.

    Two negative root orbitals occupied at every site; all loop electric0,±1
    coherent, local radial ground of the actual isolated negative-filled root.
    Full interacting H, not this preparation, owns subsequent propagation.
    """
    local = np.zeros(16, complex); local[0] = 1
    creators = model.architecture['create']
    for spin in [1, 0]:
        local = (creators[spin]-creators[2+spin])@local/np.sqrt(2)
    gauge = np.zeros(model.sector.dimension, complex)
    loop = {-1: .5, 0: 1., 1: .5}
    for index, state in enumerate(model.sector.states):
        electric = state[1:]
        if electric[0] == electric[1] == electric[2] and electric[0] in loop:
            gauge[index] = loop[electric[0]]*np.prod([local[(state[0] >> (4*v)) & 15] for v in range(3)])
    gauge /= np.linalg.norm(gauge)
    r = model.r; p = model.parameters; dx = r[1]-r[0]
    h = (2*np.eye(len(r))-np.eye(len(r), k=1)-np.eye(len(r), k=-1))/(2*p['radial_mass']*dx**2)
    h += np.diag(p['radial_stiffness']*(r-p['radial_centre'])**2/2+3/(8*p['radial_mass']*r**2)-2*p['root_coupling']*r)
    _, radial = np.linalg.eigh(h)
    material_state = np.eye(10)[:, 0]
    answer = np.kron(np.kron(gauge, material_state), np.kron(np.kron(radial[:, 0], radial[:, 0]), radial[:, 0]))
    return answer/np.linalg.norm(answer)
