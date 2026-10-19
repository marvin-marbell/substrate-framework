#!/usr/bin/env python3
"""Open two-plane electric/CAR ladder: actual spatial response, never Bloch q."""
from __future__ import annotations
import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply
from rich_photon_fullq_20261014_model import fock_basis
from rich_photon_20261014_loop import hop

LAW_ID = 'rich-spatial-open-integer-gauss-two-plane-ladder-20261014'
ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Parameters:
    cells: int = 2
    electric_bound: int = 1
    hopping: float = 1.0
    repulsion: float = 4.0
    inertia: float = 1.0
    rail_hopping: float = 1.0


def geometry(cells):
    positions = np.array([(x, int(rail == 1), int(rail == 2)) for x in range(cells + 1) for rail in range(3)], float)
    edges = [(3*x+r, 3*(x+1)+r) for x in range(cells) for r in range(3)]
    edges += [(3*x, 3*x+r) for x in range(cells+1) for r in (1, 2)]
    tree = [3*x for x in range(cells)] + list(range(3*cells, len(edges)))
    chords = [i for i in range(len(edges)) if i not in tree]
    incidence = np.zeros((len(positions), len(edges)), int)
    for i, (a, b) in enumerate(edges):
        incidence[a, i], incidence[b, i] = 1, -1
    faces = []
    for x in range(cells):
        for rail in (1, 2):
            face = np.zeros(len(edges), int)
            face[3*x] = 1
            face[3*x+rail] = -1
            face[3*cells+2*(x+1)+rail-1] = 1
            face[3*cells+2*x+rail-1] = -1
            faces.append(face)
    return positions, edges, tree, chords, incidence, np.array(faces)


class Sector:
    """All integer links bounded symmetrically; solve Gauss before projection."""
    def __init__(self, p: Parameters, up=1, down=1):
        self.p, self.up, self.down = p, up, down
        self.positions, self.edges, self.tree, self.chords, self.incidence, self.faces = geometry(p.cells)
        vertices = len(self.positions)
        self.fock = fock_basis(vertices, up, down)
        self.background = np.zeros(vertices, int)
        if up+down >= 2:
            self.background[0], self.background[3] = up+down-1, 1
        else:
            self.background[0] = up+down
        rows = list(range(1, vertices))
        inv = np.linalg.inv(self.incidence[np.ix_(rows, self.tree)])
        chord_values = np.array(list(itertools.product(range(-p.electric_bound, p.electric_bound+1), repeat=len(self.chords))), int)
        cycle_tree = -self.incidence[np.ix_(rows, self.chords)].T @ inv.T
        bits_list, flux_list, field_list = [], [], []
        for bits in self.fock:
            occupation = np.array([((bits >> (2*i)) & 1)+((bits >> (2*i+1)) & 1) for i in range(vertices)])
            rho = occupation-self.background
            electric = np.zeros((len(chord_values), len(self.edges)), int)
            electric[:, self.tree] = np.rint((-rho[rows]) @ inv.T + chord_values @ cycle_tree).astype(int)
            electric[:, self.chords] = chord_values
            valid = np.max(np.abs(electric), axis=1) <= p.electric_bound
            bits_list.extend([bits]*int(valid.sum()))
            flux_list.extend(chord_values[valid].tolist())
            field_list.extend(electric[valid].tolist())
        self.bits = np.array(bits_list, dtype=np.int64)
        self.flux = np.array(flux_list, int)
        self.electric = np.array(field_list, int)
        self.dimension = len(self.bits)
        self.index = {(int(bits), tuple(flux)): i for i, (bits, flux) in enumerate(zip(self.bits, self.flux))}
        self.occupation = np.array([[((int(bits) >> (2*i)) & 1)+((int(bits) >> (2*i+1)) & 1) for i in range(vertices)] for bits in self.bits])
        self.rho = self.occupation-self.background
        self.gauss_error = float(np.max(np.abs(self.electric @ self.incidence.T+self.rho)))
        self.transfers = []
        for edge, (a, b) in enumerate(self.edges):
            r, c, data = [], [], []
            shift = np.zeros(len(self.chords), int)
            if edge in self.chords:
                shift[self.chords.index(edge)] = 1
            coefficient = p.hopping * (p.rail_hopping if edge < 3*p.cells and edge % 3 else 1)
            for col, (bits, flux) in enumerate(zip(self.bits, self.flux)):
                for spin in (0, 1):
                    result = hop(int(bits), 2*b+spin, 2*a+spin)
                    if result is None:
                        continue
                    newbits, sign = result
                    target = self.index.get((newbits, tuple(flux+shift)))
                    if target is not None:
                        r.append(target); c.append(col); data.append(-coefficient*sign)
            self.transfers.append(sparse.csr_matrix((data, (r, c)), shape=(self.dimension, self.dimension)))
        self.electric_energy = np.sum(self.electric**2, axis=1)/(2*p.inertia)
        doubles = np.array([sum(((int(bits) >> (2*i)) & 1)*((int(bits) >> (2*i+1)) & 1) for i in range(vertices)) for bits in self.bits])
        self.H = sparse.diags(self.electric_energy+p.repulsion*doubles, format='csr')
        self.H += sum((t+t.T for t in self.transfers), sparse.csr_matrix(self.H.shape))
        self.currents = [-1j*(t-t.T) for t in self.transfers]
        self.contacts = [-(t+t.T) for t in self.transfers]

    def wilson(self, face):
        shift = np.asarray(face)[self.chords]
        r, c = [], []
        for col, (bits, flux) in enumerate(zip(self.bits, self.flux)):
            target = self.index.get((int(bits), tuple(flux+shift)))
            if target is not None:
                r.append(target); c.append(col)
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=self.H.shape)

    def solve(self, count=16):
        values, vectors = eigsh(self.H, k=min(count, self.dimension-2), which='SA', tol=2e-10, v0=np.ones(self.dimension)/np.sqrt(self.dimension))
        order = np.argsort(values)
        return values[order], vectors[:, order]

    def continuity_error(self):
        maximum = 0.0
        for site in range(len(self.positions)):
            density = sparse.diags(self.occupation[:, site])
            residual = 1j*(self.H@density-density@self.H)
            residual += sum((self.incidence[site, edge]*j for edge, j in enumerate(self.currents)), sparse.csr_matrix(self.H.shape, dtype=complex))
            if residual.nnz:
                maximum = max(maximum, float(np.max(np.abs(residual.data))))
        return maximum


def transverse_current(sector, x, rail):
    return sector.currents[3*sector.p.cells+2*x+rail-1]


def time_response(sector, ground, energy, operators, times, source_indices):
    """chi_AB(t)=-i<[A(t),B]>; full-H exponential, no pole truncation."""
    sources = np.column_stack([operator@ground for operator in operators])
    centered = sources-ground[:, None]*(ground.conj()@sources)[None, :]
    evolved = expm_multiply(-1j*(sector.H-energy*sparse.eye(sector.dimension)), centered[:, source_indices], start=float(times[0]), stop=float(times[-1]), num=len(times), traceA=-1j*float(sector.H.diagonal().sum()-energy*sector.dimension))
    response = np.array([2*np.imag(centered.conj().T@wave) for wave in evolved])
    return response, centered


def physical_charge_source(target, site):
    """CAR addition to empty E=0 state; countercharge0 and literal tree Wilson."""
    previous = {0: None}
    queue = [0]
    for vertex in queue:
        for edge in target.tree:
            a, b = target.edges[edge]
            other = b if a == vertex else a if b == vertex else None
            if other is not None and other not in previous:
                previous[other] = (vertex, edge)
                queue.append(other)
    electric = np.zeros(len(target.edges), int)
    vertex = site
    while vertex != 0:
        old, edge = previous[vertex]
        electric[edge] += 1 if target.edges[edge] == (old, vertex) else -1
        vertex = old
    bits = 1 << (2*site)
    index = target.index.get((bits, tuple(electric[target.chords])))
    if index is None:
        raise ValueError('Source dressing lies outside finite field domain')
    if not np.array_equal(target.electric[index], electric):
        raise ValueError('Source Gauss/tree mismatch')
    state = np.zeros(target.dimension, complex)
    state[index] = 1
    return state, {'site': site, 'static_countercharge': 0, 'electric_string': electric.tolist(), 'Gauss_error': float(np.max(np.abs(target.incidence@electric+target.rho[index]))), 'norm': 1.0, 'bare_source_admissible': bool(np.all(target.rho[index] == 0))}


def charge_packet(p, times):
    sector = Sector(p, 1, 0)
    initial, source = physical_charge_source(sector, 1)
    waves = expm_multiply(-1j*sector.H, initial, start=0, stop=float(times[-1]), num=len(times), traceA=-1j*float(sector.H.diagonal().sum()))
    probability = np.abs(waves)**2
    density = probability @ sector.occupation
    cell_density = density.reshape(len(times), p.cells+1, 3).sum(axis=2)
    electric = probability @ sector.electric_energy
    current = np.array([[float(np.vdot(wave, j@wave).real) for j in sector.currents] for wave in waves])
    total_energy = np.array([float(np.vdot(wave, sector.H@wave).real) for wave in waves])
    # Frozen links are a non-Gauss comparator, never the delivered state.
    frozen = np.zeros((len(sector.positions), len(sector.positions)))
    for edge, (a, b) in enumerate(sector.edges):
        coefficient = p.hopping*(p.rail_hopping if edge < 3*p.cells and edge % 3 else 1)
        frozen[a, b] = frozen[b, a] = -coefficient
    seed = np.zeros(len(sector.positions), complex); seed[1] = 1
    free = expm_multiply(-1j*frozen, seed, start=0, stop=float(times[-1]), num=len(times))
    free_density = (np.abs(free)**2).reshape(len(times), p.cells+1, 3).sum(axis=2)
    return {'dimension': sector.dimension, 'source': source, 'times': times.tolist(), 'cell_density': cell_density.tolist(), 'link_currents': current.tolist(), 'electric_energy': electric.tolist(), 'total_energy_drift': float(np.ptp(total_energy)), 'norm_error': float(np.max(np.abs(probability.sum(axis=1)-1))), 'frozen_field_cell_density': free_density.tolist(), 'frozen_field_density_max_difference': float(np.max(np.abs(cell_density-free_density))), 'electric_exchange_range': float(np.ptp(electric)), 'max_receiver_density': cell_density.max(axis=0).tolist(), 'Gauss_error': sector.gauss_error}


def run(p, name):
    started = time.time()
    sector = Sector(p)
    energies, states = sector.solve()
    ground, energy = states[:, 0], float(energies[0])
    loops = [sector.wilson(face) for face in sector.faces]
    flux = [(w-w.T)/(2j) for w in loops]
    operators = flux+[transverse_current(sector, x, rail) for x in range(p.cells+1) for rail in (1, 2)]
    labels = [f'F_{x}_{rail}' for x in range(p.cells) for rail in ('y', 'z')]+[f'J_{x}_{rail}' for x in range(p.cells+1) for rail in ('y', 'z')]
    times = np.linspace(0, 10, 81)
    sources = [0, 1, 2*p.cells, 2*p.cells+1]
    response, vectors = time_response(sector, ground, energy, operators, times, sources)
    overlap = states.conj().T@vectors
    residues = np.abs(overlap)**2
    boundary = np.any(np.abs(sector.electric) == p.electric_bound, axis=1)
    impulse = expm_multiply(-0.4j*flux[0], ground)
    packet = expm_multiply(-1j*sector.H, impulse, start=0, stop=10, num=len(times), traceA=-1j*float(sector.H.diagonal().sum()))
    cell_field = []
    for x in range(p.cells):
        edges = np.flatnonzero(np.any(np.abs(sector.faces[2*x:2*x+2]) > 0, axis=0))
        cell_field.append(np.sum(sector.electric[:, edges]**2, axis=1)/(2*p.inertia))
    electric_wave = np.abs(packet)**2@np.array(cell_field).T
    electric_baseline = np.abs(ground)**2@np.array(cell_field).T
    no_matter = Sector(p, 0, 0)
    empty_ground = np.zeros(no_matter.dimension); empty_ground[no_matter.index[(0, tuple(np.zeros(len(no_matter.chords), int)))]] = 1
    empty_flux = [(no_matter.wilson(face)-no_matter.wilson(face).T)/(2j) for face in no_matter.faces]
    empty_response, _ = time_response(no_matter, empty_ground, 0., empty_flux, times, [0, 1])
    result = {'candidate_id': LAW_ID, 'name': name, 'parameters': asdict(p), 'geometry': {'positions': sector.positions.tolist(), 'edges': sector.edges, 'tree': sector.tree, 'chords': sector.chords, 'faces': sector.faces.tolist(), 'background': sector.background.tolist(), 'boundary': 'open; no translation or Bloch momentum claimed'}, 'state': {'kind': 'joint lowest eigenstate in Nup=Ndown=1', 'dimension': sector.dimension, 'ground_energy': energy, 'integer_Gauss_error': sector.gauss_error, 'operator_continuity_error': sector.continuity_error(), 'boundary_link_probability': float(np.sum(np.abs(ground[boundary])**2)), 'loop_cosines': [float(np.vdot(ground, (w+w.T)@ground/2).real) for w in loops]}, 'response': {'times': times.tolist(), 'operator_labels': labels, 'source_labels': [labels[i] for i in sources], 'retarded_tensor_time': response.tolist(), 'max_abs_receiver_by_source': np.max(np.abs(response), axis=0).tolist(), 'source_norms_squared': np.sum(np.abs(vectors)**2, axis=0).tolist(), 'low_energy_gaps': (energies-energy).tolist(), 'selected_partial_spectral_residues': residues.tolist(), 'no_matter_flux_response': empty_response.tolist(), 'no_matter_max_abs_receiver_by_source': np.max(np.abs(empty_response), axis=0).tolist(), 'scope': 'Real site/cell separation and full-H finite-time impulse response. Low eigenstate residues are partial, not complete spectrum. No q called momentum.'}, 'prepared_flux_pulse': {'state': 'exp(-0.4i F_cell0_y)|joint ground>', 'cell_electric_energy_excess': (electric_wave-electric_baseline).tolist(), 'electric_energy_exchange_range': float(np.ptp(np.abs(packet)**2@sector.electric_energy)), 'boundary_probability_max': float(np.max(np.sum(np.abs(packet[:, boundary])**2, axis=1)))}, 'charged_packet': charge_packet(p, times), 'seconds': time.time()-started}
    output = ROOT/f'{name}_result.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(ROOT/f'{name}_states.npz', ground=ground, energies=energies, eigenstates=states, bits=sector.bits, electric=sector.electric, retarded_response=response)
    print(json.dumps({'result': str(output), 'dimension': sector.dimension, 'energy': energy, 'loop_cosines': result['state']['loop_cosines'], 'maximum_response': result['response']['max_abs_receiver_by_source'], 'empty_maximum_response': result['response']['no_matter_max_abs_receiver_by_source'], 'charge_max_receiver': result['charged_packet']['max_receiver_density'], 'charge_field_exchange': result['charged_packet']['electric_exchange_range'], 'seconds': result['seconds']}, indent=2))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', type=int, default=2)
    parser.add_argument('--hopping', type=float, default=1.)
    parser.add_argument('--rail-hopping', type=float, default=1.)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    raise SystemExit(run(Parameters(cells=args.cells, hopping=args.hopping, rail_hopping=args.rail_hopping), args.name))
