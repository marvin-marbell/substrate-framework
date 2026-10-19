#!/usr/bin/env python3
"""Compact single-plaquette descendant of the published Hubbard/link law.

Finite electric-cutoff ED; exact Gauss constraint, N=2, no bare magnetic term.
External two-leg probes are response/Raman tensors, not an asymptotic QED S matrix.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from rich_photon_20261014_loop import Parameters, LoopMatter, hop, plaquette_embedding, transverse_frame

LAW_ID = 'rich-interaction-compact-one-plaquette-v1-20261014'


def complex_data(value):
    a = np.asarray(value)
    return {'real': a.real.tolist(), 'imag': a.imag.tolist()}


class CoupledMolecule:
    """Basis (fermion bits, twice electric fields), div E=-rho.

    E_3 is integer+1/2 and all |E_e|<=cutoff+1/2. This fixes a theta-sector
    compatible with static half-charge ions. Edge0/2 E are integer, edge1/3
    half-integer. This extra electric topological-sector choice is explicit.
    """
    def __init__(self, cutoff=4, parameters=Parameters(), frozen=False):
        self.parameters, self.cutoff, self.frozen = parameters, cutoff, frozen
        if parameters.link_charge != 1 or parameters.particles_per_loop != 2 or parameters.sites_per_loop != 4:
            raise ValueError('This physical Gauss basis implements only e=1, N=2, four sites')
        self.matter = LoopMatter(parameters)
        self.basis = []
        for bits in self.matter.basis:
            if frozen:
                self.basis.append((bits, (0, 0, 0, 0)))
                continue
            rho2 = [2*(((bits >> (2*s)) & 1)+((bits >> (2*s+1)) & 1))-1 for s in range(4)]
            for circulation in range(-cutoff-2, cutoff+2):
                previous = 2*circulation+1
                fields = []
                for site in range(4):
                    previous -= rho2[site]
                    fields.append(previous)
                if max(abs(e) for e in fields) <= 2*cutoff+1:
                    self.basis.append((bits, tuple(fields)))
        self.index = {key: i for i, key in enumerate(self.basis)}
        dimension = len(self.basis)
        self.transfer = np.zeros((4, 2, dimension, dimension), complex)
        self.electric = np.array([np.asarray(fields)/2 for _, fields in self.basis])
        self.double = np.array([sum(((b >> (2*s)) & 1)*((b >> (2*s+1)) & 1) for s in range(4)) for b, _ in self.basis])
        self.electric_energy = np.sum(self.electric**2, axis=1)/(2*parameters.electric_inertia)
        for column, (bits, fields) in enumerate(self.basis):
            for edge in range(4):
                for spin in range(2):
                    result = hop(bits, 2*((edge+1) % 4)+spin, 2*edge+spin)
                    if result is None:
                        continue
                    newbits, sign = result
                    shifted = list(fields)
                    if not frozen:
                        shifted[edge] += 2
                    row = self.index.get((newbits, tuple(shifted)))
                    if row is not None:
                        self.transfer[edge, spin, row, column] = -parameters.hopping*sign
        adjoint = self.transfer.conj().transpose(0, 1, 3, 2)
        self.current = -1j*(self.transfer-adjoint)
        self.contact = -(self.transfer+adjoint)
        self.H = np.diag(parameters.repulsion*self.double+self.electric_energy)+np.sum(self.transfer+adjoint, axis=(0, 1))
        self.energies, self.states = np.linalg.eigh(self.H)
        self.ground = self.states[:, 0]
        self.gaps = self.energies-self.energies[0]
        self.state_id = hashlib.sha256(self.H.tobytes()+self.ground.tobytes()).hexdigest()

    def vertex(self, q):
        """J_(spin,pol)(q); q 1/a, transverse lattice frame, xy molecule."""
        profiles = plaquette_embedding(q, 0, 1)@transverse_frame(q)
        return np.einsum('ep,esij->spij', profiles, self.current).reshape(4, len(self.basis), len(self.basis))

    def two_leg_contact(self, qout, qin):
        """H'' contracted with outgoing conjugate and incoming profiles."""
        left = plaquette_embedding(qout, 0, 1)@transverse_frame(qout)
        right = plaquette_embedding(qin, 0, 1)@transverse_frame(qin)
        result = np.zeros((4, 4, len(self.basis), len(self.basis)), complex)
        for spin in range(2):
            result[2*spin:2*spin+2, 2*spin:2*spin+2] = np.einsum('ep,eq,eij->pqij', left.conj(), right, self.contact[:, spin])
        return result

    def response(self, qout, qin, z):
        """C+chi; receiver J(qout)^dagger, source J(qin), retarded both poles."""
        incoming, outgoing = self.vertex(qin), self.vertex(qout).conj().transpose(0, 2, 1)
        g, u = self.ground, self.states
        a = np.einsum('i,aij,jn->an', g.conj(), outgoing, u)
        b = np.einsum('ni,bij,j->bn', u.conj().T, incoming, g)
        c = np.einsum('i,bij,jn->bn', g.conj(), incoming, u)
        d = np.einsum('ni,aij,j->an', u.conj().T, outgoing, g)
        susceptibility = np.einsum('an,bn,n->ab', a[:, 1:], b[:, 1:], 1/(z-self.gaps[1:]))-np.einsum('bn,an,n->ab', c[:, 1:], d[:, 1:], 1/(z+self.gaps[1:]))
        contact = np.einsum('i,abij,j->ab', g.conj(), self.two_leg_contact(qout, qin), g)
        return contact+susceptibility

    def raman(self, qout, qin, omega, eta, final, ground_only=False):
        """Kramers-Heisenberg two-leg tensor, exact finite joint intermediates.

        Outgoing energy omega-(E_final-E_ground) includes actual molecular/flux
        recoil. q are independent source profiles: anchored molecule can absorb
        momentum; no translational COM recoil or on-shell free photon is assumed.
        """
        incoming = self.vertex(qin)
        outgoing = self.vertex(qout).conj().transpose(0, 2, 1)
        u, g, f = self.states, self.ground, self.states[:, final]
        indices = np.array([0]) if ground_only else np.arange(len(self.gaps))
        intermediate = u[:, indices]
        outgoing_energy = omega-self.gaps[final]
        left = np.einsum('i,aij,jn->an', f.conj(), outgoing, intermediate)
        right = np.einsum('ni,bij,j->bn', intermediate.conj().T, incoming, g)
        crossed_left = np.einsum('i,bij,jn->bn', f.conj(), incoming, intermediate)
        crossed_right = np.einsum('ni,aij,j->an', intermediate.conj().T, outgoing, g)
        contact = np.einsum('i,abij,j->ab', f.conj(), self.two_leg_contact(qout, qin), g)
        return contact+np.einsum('an,bn,n->ab', left, right, 1/(omega-self.gaps[indices]+1j*eta))+np.einsum('bn,an,n->ab', crossed_left, crossed_right, 1/(-outgoing_energy-self.gaps[indices]+1j*eta))

    def time_motion(self, q):
        # A real impulsive current perturbation, then autonomous joint evolution.
        source = self.vertex(q)[0]+self.vertex(q)[2]
        source = (source+source.conj().T)/2
        initial = self.ground+.2*source@self.ground
        initial /= np.linalg.norm(initial)
        coefficients = self.states.conj().T@initial
        rows = []
        for t in np.linspace(0, 20, 81):
            psi = self.states@(np.exp(-1j*self.gaps*t)*coefficients)
            prob = np.abs(psi)**2
            rows.append([float(t), float(prob@self.electric_energy), float(prob@self.double), float(np.real(psi.conj()@self.H@psi))])
        return np.asarray(rows)

    def spectrum(self, q):
        """Complete finite Lehmann data, all 4 spin/polarization source columns."""
        return np.einsum('ni,aij,j->na', self.states.conj().T, self.vertex(q), self.ground)


def run(output):
    started = time.perf_counter()
    qin, qout = np.array([.7, .2, .45]), np.array([-.25, .65, .3])
    omega, eta = 4.3, .04
    models = [CoupledMolecule(cutoff=c) for c in (2, 3, 4)]
    full = models[-1]
    frozen = CoupledMolecule(frozen=True)
    rows = []
    for model in models+[frozen]:
        response = model.response(qout, qin, 1.1+.04j)
        motion = model.time_motion(qin)
        elastic = model.raman(qout, qin, omega, eta, 0)
        wrong = model.raman(qout, qin, omega, eta, 0, ground_only=True)
        rows.append({'cutoff': model.cutoff, 'frozen_link': model.frozen, 'dimension': len(model.basis), 'ground_energy': float(model.energies[0]), 'gap': float(model.gaps[1]), 'ground_double_occupancy': float(np.abs(model.ground)**2@model.double), 'ground_electric_energy': float(np.abs(model.ground)**2@model.electric_energy), 'nonforward_response': complex_data(response), 'elastic_raman': complex_data(elastic), 'ground_intermediate_raman': complex_data(wrong), 'elastic_full_minus_ground_intermediate_norm': float(np.linalg.norm(elastic-wrong)), 'time_electric_energy_range': float(np.ptp(motion[:, 1])), 'time_double_occupancy_range': float(np.ptp(motion[:, 2])), 'time_total_energy_range': float(np.ptp(motion[:, 3])), 'time_columns': ['time', 'electric_energy', 'double_occupancy', 'total_energy'], 'time_samples': motion.tolist(), 'state_id': model.state_id})
    finals = np.flatnonzero((full.gaps > 1e-9) & (full.gaps < omega))
    raman = []
    for final in finals:
        tensor = full.raman(qout, qin, omega, eta, int(final))
        raman.append({'final': int(final), 'energy_transfer': float(full.gaps[final]), 'outgoing_frequency': float(omega-full.gaps[final]), 'tensor': complex_data(tensor), 'norm_squared': float(np.linalg.norm(tensor)**2)})
    # Complete spectrum, not a selected-pole effective fit.
    frequencies = np.linspace(0, 6, 121)
    response_grid = np.asarray([full.response(qout, qin, w+1j*eta) for w in frequencies])
    spectral_path = output.with_suffix('.npz')
    np.savez_compressed(spectral_path, energies=full.energies, eigenvectors=full.states, basis_bits=np.array([b for b, _ in full.basis]), basis_electric=full.electric, qin=qin, qout=qout, incoming_transitions=full.spectrum(qin), outgoing_transitions=full.spectrum(qout), frequencies=frequencies, nonforward_response=response_grid, current=full.current, contact=full.contact, hamiltonian=full.H)
    result = {'law_id': LAW_ID, 'parent_law_id': 'rich-photon-hubbard-plaquette-links-v1-20261014', 'parameters': asdict(full.parameters), 'qin': qin.tolist(), 'qout': qout.tolist(), 'incoming_frequency': omega, 'eta': eta, 'models': rows, 'inelastic_channels': raman, 'spectral_artifact': spectral_path.name, 'coupled_vs_frozen_response_norm': float(np.linalg.norm(full.response(qout, qin, 1.1+.04j)-frozen.response(qout, qin, 1.1+.04j))), 'elapsed_seconds': time.perf_counter()-started, 'runtime': {'python': platform.python_version(), 'numpy': np.__version__}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'supplier_sha256': hashlib.sha256(Path(__file__).with_name('rich_photon_20261014_loop.py').read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'models': [{k: r[k] for k in ('cutoff', 'frozen_link', 'dimension', 'ground_energy', 'gap', 'time_electric_energy_range', 'elastic_full_minus_ground_intermediate_norm')} for r in rows], 'inelastic_channel_count': len(raman), 'coupled_vs_frozen_response_norm': result['coupled_vs_frozen_response_norm'], 'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
