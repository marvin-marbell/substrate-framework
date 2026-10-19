#!/usr/bin/env python3
"""Induced-link photon candidate: exact Hubbard loop response, quadratic feedback.

Units a=hbar=charge=t=1 unless configured. No bare magnetic term exists.
The finite-ring matter calculation is exact; the quantized-link dynamics is the
small-amplitude quadratic influence action, not exact full nonlinear many-body QED.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

LAW_ID = "rich-photon-hubbard-plaquette-links-v1-20261014"

@dataclass(frozen=True)
class Parameters:
    hopping: float = 1.0
    repulsion: float = 4.0
    electric_inertia: float = 1.0
    link_charge: float = 1.0
    particles_per_loop: int = 2
    sites_per_loop: int = 4


def hop(bits: int, target: int, source: int):
    if not bits & (1 << source) or bits & (1 << target):
        return None
    sign = (-1) ** ((bits & ((1 << source)-1)).bit_count())
    removed = bits ^ (1 << source)
    sign *= (-1) ** ((removed & ((1 << target)-1)).bit_count())
    return removed | (1 << target), sign


class LoopMatter:
    """One physical spinful four-site molecule, fixed two-fermion neutral sector.

    Each cubic plaquette hosts one such molecule, with four distinct orbitals;
    molecules on different plaquettes share electromagnetic links, NOT orbitals.
    Ionic background -1/2 per site neutralizes the two electrons. Quantum spin and
    all intraring particle-hole/double-occupancy states are retained exactly.
    """
    def __init__(self, parameters: Parameters = Parameters()):
        self.parameters = parameters
        self.basis = [sum(1 << i for i in occupied)
                      for occupied in itertools.combinations(range(8), parameters.particles_per_loop)]
        self.index = {bits: i for i, bits in enumerate(self.basis)}
        self.transfers = np.zeros((4, len(self.basis), len(self.basis)), complex)
        self.density = np.zeros_like(self.transfers)
        interaction = np.zeros(len(self.basis))
        for column, bits in enumerate(self.basis):
            for site in range(4):
                nup, ndown = (bits >> (2*site)) & 1, (bits >> (2*site+1)) & 1
                self.density[site, column, column] = parameters.link_charge*(nup + ndown - .5)
                interaction[column] += parameters.repulsion * nup * ndown
            for edge in range(4):
                for spin in range(2):
                    result = hop(bits, 2*((edge+1) % 4)+spin, 2*edge+spin)
                    if result is not None:
                        newbits, sign = result
                        self.transfers[edge, self.index[newbits], column] += -parameters.hopping * sign
        self.interaction = np.diag(interaction)
        self.H = self.hamiltonian(np.zeros(4))
        self.energies, self.states = np.linalg.eigh(self.H)
        self.ground = self.states[:, 0]
        self.gaps = self.energies[1:] - self.energies[0]
        # H(theta)=T exp(i e theta)+hc; J=-H', contact=H''.
        self.current = -1j*parameters.link_charge*(self.transfers-self.transfers.conj().transpose(0, 2, 1))
        self.contact_operators = -parameters.link_charge**2*(self.transfers+self.transfers.conj().transpose(0, 2, 1))
        self.contact = np.einsum('i,aij,j->a', self.ground.conj(), self.contact_operators, self.ground).real
        self.operators = np.concatenate((self.density, self.current))
        self.transitions = np.einsum('i,aij,jn->na', self.ground.conj(), self.operators, self.states[:, 1:])

    def hamiltonian(self, phases):
        weighted = np.exp(1j*self.parameters.link_charge*np.asarray(phases))[:, None, None]*self.transfers
        return self.interaction + np.sum(weighted+weighted.conj().transpose(0, 2, 1), axis=0)

    def susceptibility(self, z: complex):
        """Retarded chi_ab=-i theta(t)<[O_a(t),O_b]>, O=(rho_0..3,J_0..3).

        z=omega+i eta, eta>0; both +/- exact matter poles. Fourier e^{-i omega t}.
        Equal-time current contact is NOT included here.
        """
        v = self.transitions
        return (np.einsum('na,nb,n->ab', v, v.conj(), 1/(z-self.gaps))
                - np.einsum('na,nb,n->ab', v.conj(), v, 1/(z+self.gaps)))

    def edge_kernel(self, z: complex, frozen: bool = False):
        """Exact quadratic matter Hessian C+chi_JJ (four directed phases)."""
        return np.diag(self.contact).astype(complex) + (0 if frozen else self.susceptibility(z)[4:, 4:])

    def ward_data(self):
        # dot rho_s = J_in-J_out, physical J=-dH/dtheta.
        continuity = []
        for site in range(4):
            derivative = 1j*(self.H@self.density[site]-self.density[site]@self.H)
            continuity.append(np.max(np.abs(derivative-(self.current[(site-1) % 4]-self.current[site]))))
        incidence = np.zeros((4, 4))
        for edge in range(4):
            incidence[edge, (edge+1) % 4] = 1
            incidence[edge, edge] = -1
        k0 = self.edge_kernel(0j)
        return {'operator_continuity_max': float(max(continuity)),
                'static_gradient_kernel_max': float(np.max(np.abs(k0@incidence))),
                'ground_gap': float(self.gaps.min()),
                'flux_stiffness': float(np.real(np.ones(4)@k0@np.ones(4)/16)),
                'contact_edges': self.contact.tolist(),
                'pair_current_spectral_weight': float(np.sum(np.abs(self.transitions[:, 4:])**2))}


def plaquette_embedding(q, i: int, j: int):
    """Directed ring phases from midpoint-link vector A, plaquette-centered Fourier.

    edges +i,+j,-i,-j. q in [-pi,pi]^3; finite lattice curl qhat=2 sin(q/2).
    """
    q = np.asarray(q)
    r = np.zeros((4, 3), complex)
    r[0, i], r[1, j] = np.exp(-.5j*q[j]), np.exp(.5j*q[i])
    r[2, i], r[3, j] = -np.exp(.5j*q[j]), -np.exp(-.5j*q[i])
    return r


def density_current_response(matter: LoopMatter, q, z: complex):
    """Four-current chi order (rho,Jx,Jy,Jz), plus spatial contact, per cell.

    rho Fourier phases follow the actual four molecular sites. No contact is
    added to density-density; spatial Pi equals contact+chi_JJ.
    """
    response = np.zeros((4, 4), complex)
    contact = np.zeros((3, 3), complex)
    spectral_ward = 0.
    qhat = 2*np.sin(np.asarray(q)/2)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        positions = np.zeros((4, 3))
        positions[:, i], positions[:, j] = [-.5, .5, .5, -.5], [-.5, -.5, .5, .5]
        source = np.zeros((8, 4), complex)
        source[:4, 0] = np.exp(1j*positions@np.asarray(q))
        r = plaquette_embedding(q, i, j)
        source[4:, 1:] = r
        response += source.conj().T@matter.susceptibility(z)@source
        contact += r.conj().T@np.diag(matter.contact)@r
        amplitudes = matter.transitions@source
        spectral_ward = max(spectral_ward, float(np.max(np.abs(
            amplitudes[:, 1:]@qhat + matter.gaps*amplitudes[:, 0]))))
    return response, contact, spectral_ward


def polarization(matter: LoopMatter, q, z: complex, frozen: bool = False):
    """Pi_ij(q,z), per unit cell, all three oriented molecular plaquettes."""
    kernel = matter.edge_kernel(z, frozen)
    return sum(r.conj().T@kernel@r for r in
               (plaquette_embedding(q, i, j) for i, j in ((0, 1), (0, 2), (1, 2))))


def inverse_propagator(matter: LoopMatter, q, z: complex, frozen: bool = False):
    """D^{-1}=Pi-z^2 epsilon_b I. epsilon_b is primitive electric inertia."""
    return polarization(matter, q, z, frozen)-matter.parameters.electric_inertia*z*z*np.eye(3)


def transverse_frame(q):
    qhat = 2*np.sin(np.asarray(q)/2)
    unit = qhat/np.linalg.norm(qhat)
    helper = np.eye(3)[np.argmin(np.abs(unit))]
    first = np.cross(unit, helper); first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(unit, first)))


def propagator(matter: LoopMatter, q, z: complex):
    """Full 3x3 temporal-gauge D. Gauss density sector must accompany J.

    Use full inverse (not a scalar polarization fit); for transverse physical
    response project AFTER inversion if longitudinal mixing is present.
    """
    return np.linalg.inv(inverse_propagator(matter, q, z))


def transverse_propagator(matter: LoopMatter, q, z: complex):
    """Two-polarization tensor; temporal-gauge inverse then physical projection."""
    frame = transverse_frame(q)
    return frame.conj().T@propagator(matter, q, z)@frame


def collective_matrix(matter: LoopMatter, q, frozen: bool = False):
    """Mass-normalized quadratic Hamiltonian matrix from exact ring poles."""
    epsilon = matter.parameters.electric_inertia
    contact = np.zeros((3, 3), complex)
    columns, gaps = [], []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        r = plaquette_embedding(q, i, j)
        contact += r.conj().T@np.diag(matter.contact)@r
        amplitudes = matter.transitions[:, 4:]@r
        for delta, v in zip(matter.gaps, amplitudes):
            if np.linalg.norm(v) > 1e-12:
                columns.append(np.sqrt(2*delta)*v.conj())
                gaps.append(delta)
    # Ring real Hamiltonian makes all J transitions imaginary: residue matrices
    # are real at ring level. Fourier embedding complex requires Hermitian K.
    b = np.column_stack(columns) if columns else np.zeros((3, 0), complex)
    if frozen:
        b = np.zeros_like(b)
    dynamic = np.block([[contact/epsilon, -b/np.sqrt(epsilon)],
                        [-b.conj().T/np.sqrt(epsilon), np.diag(np.asarray(gaps)**2)]])
    return dynamic, np.asarray(gaps)


def collective_gauss(matter: LoopMatter, q):
    """Linearized Gauss generator G=n†P; M n=0 makes G conserved.

    Physical quadratic sector selects G=0 and quotients coordinate shifts Q+nλ.
    This is NOT a full compact nonlinear Hubbard/link ground-state certificate.
    """
    dynamic, gaps = collective_matrix(matter, q)
    field = np.sqrt(matter.parameters.electric_inertia)*2*np.sin(np.asarray(q)/2)
    pair = -dynamic[3:, :3]@field/gaps**2
    null = np.concatenate((field, pair))
    return null, float(np.linalg.norm(dynamic@null))


def collective_modes(matter: LoopMatter, q, frozen: bool = False):
    """Full exact finite collective realization; zero gauge mode is not photon."""
    dynamic, gaps = collective_matrix(matter, q, frozen)
    values, vectors = np.linalg.eigh(dynamic)
    weights = np.sum(np.abs(vectors[:3])**2, axis=0)/matter.parameters.electric_inertia
    return values, weights, vectors[:3], gaps


def feedback_motion(matter: LoopMatter, q):
    """New finite-time actual collective field/matter exchange, not fixed sea."""
    dynamic, _ = collective_matrix(matter, q)
    values, vectors = np.linalg.eigh(dynamic)
    frequencies = np.sqrt(np.maximum(values, 0))
    initial = np.zeros(len(values), complex)
    initial[:3] = .02*transverse_frame(q)[:, 0]
    coefficients = vectors.conj().T@initial
    frozen, _ = collective_matrix(matter, q, frozen=True)
    fvalues, fvectors = np.linalg.eigh(frozen)
    fcoefficients = fvectors.conj().T@initial
    initial_energy = float(np.real(initial.conj()@dynamic@initial)/2)
    rows = []
    gauss_null, _ = collective_gauss(matter, q)
    for t in np.linspace(0., 20., 81):
        coordinate = vectors@(np.cos(frequencies*t)*coefficients)
        velocity = -vectors@(frequencies*np.sin(frequencies*t)*coefficients)
        fcoordinate = fvectors@(np.cos(np.sqrt(np.maximum(fvalues, 0))*t)*fcoefficients)
        energy = float(np.real(velocity.conj()@velocity+coordinate.conj()@dynamic@coordinate)/2)
        rows.append({'time': float(t), 'field_coordinate_norm': float(np.linalg.norm(coordinate[:3])),
                     'matter_coordinate_norm': float(np.linalg.norm(coordinate[3:])),
                     'field_frozen_difference': float(np.linalg.norm(coordinate[:3]-fcoordinate[:3])),
                     'gauss_generator_abs': float(abs(gauss_null.conj()@velocity)),
                     'energy': energy})
    return {'q': np.asarray(q).tolist(), 'initial_field_amplitude': .02,
            'initial_energy': initial_energy,
            'energy_max_drift': max(abs(row['energy']-initial_energy) for row in rows),
            'max_matter_displacement': max(row['matter_coordinate_norm'] for row in rows),
            'max_frozen_field_difference': max(row['field_frozen_difference'] for row in rows),
            'max_gauss_generator_abs': max(row['gauss_generator_abs'] for row in rows),
            'samples': rows}


def serialize_complex(array):
    array = np.asarray(array)
    return {'real': array.real.tolist(), 'imag': array.imag.tolist()}


def run():
    started = time.time()
    matter = LoopMatter()
    free = LoopMatter(Parameters(repulsion=0.0))
    rows = []
    for direction in ((0., 0., 1.), (1., 1., 1.), (1., 2., 3.)):
        direction = np.asarray(direction)/np.linalg.norm(direction)
        for scale in (.08, .2, .5, 1.0):
            q = scale*direction
            values, weights, vectors, _ = collective_modes(matter, q)
            frame = transverse_frame(q)
            transverse_weights = np.sum(np.abs(frame.conj().T@vectors)**2, axis=0)
            candidates = np.flatnonzero((values > 1e-10) & (transverse_weights > .05))
            low = candidates[:2]
            frequencies = np.sqrt(values[low])
            responses = []
            for index, omega in zip(low, frequencies):
                matrix = inverse_propagator(matter, q, complex(omega))
                responses.append(float(np.linalg.norm(matrix@vectors[:, index])/np.linalg.norm(vectors[:, index])))
            fv, fw, fvec, _ = collective_modes(matter, q, frozen=True)
            ftrans = np.sum(np.abs(frame.conj().T@fvec)**2, axis=0)
            frozen_ids = np.flatnonzero((fv > 1e-10) & (ftrans > .05))[:2]
            uv, uw, uvec, _ = collective_modes(free, q)
            utrans = np.sum(np.abs(frame.conj().T@uvec)**2, axis=0)
            free_ids = np.flatnonzero((uv > 1e-10) & (utrans > .05))[:2]
            rows.append({'q': q.tolist(), 'qhat_norm': float(np.linalg.norm(2*np.sin(q/2))),
                         'frequencies': frequencies.tolist(),
                         'phase_velocities_qhat': (frequencies/np.linalg.norm(2*np.sin(q/2))).tolist(),
                         'field_residues': weights[low].tolist(),
                         'matter_fraction': (1-weights[low]*matter.parameters.electric_inertia).tolist(),
                         'pole_inverse_residuals': responses,
                         'min_squared_frequency': float(values.min()),
                         'frozen_frequencies': np.sqrt(fv[frozen_ids]).tolist(),
                         'free_U0_frequencies': np.sqrt(uv[free_ids]).tolist()})
    q = np.array([0.12, 0.17, 0.23])
    grid = []
    for omega in (0., .1, .5, 1., 2., 3., 4., 6., 10.):
        z = complex(omega, .03)
        four_current, contact, spectral_ward = density_current_response(matter, q, z)
        grid.append({'omega': omega, 'eta': .03,
                     'Pi': serialize_complex(polarization(matter, q, z)),
                     'D': serialize_complex(propagator(matter, q, z)),
                     'D_transverse': serialize_complex(transverse_propagator(matter, q, z)),
                     'chi_four_current': serialize_complex(four_current),
                     'spatial_contact': serialize_complex(contact),
                     'spectral_continuity_error': spectral_ward,
                     'rho_J_chi_ring': serialize_complex(matter.susceptibility(z))})
    values, weights, vectors, gaps = collective_modes(matter, q)
    frozen_static = polarization(matter, np.zeros(3), 0j, frozen=True)
    full_static = polarization(matter, np.zeros(3), 0j)
    decoupled = LoopMatter(Parameters(link_charge=0.0))
    removed_matter_squared_modes, _, _, _ = collective_modes(decoupled, q)
    decoupled_pi = polarization(decoupled, q, complex(.5, .03))
    receipt = {'law': LAW_ID, 'parameters': asdict(matter.parameters),
               'state': 'Exact unique two-fermion Hubbard-ring ground; product over three plaquette molecules per cell, quadratic quantized link/matter tangent',
               'approximations': ['Exact finite ring diagonalization and source derivatives.',
                                  'Quadratic amplitude influence action; nonlinear interloop/link corrections uncomputed.',
                                  'Primitive electric inertia retained; all magnetic stiffness induced.',
                                  'Intraring virtual particle-hole/double-occupancy pairs, not a derived relativistic positron sea.'],
               'ward_and_pair_data': matter.ward_data(), 'free_U0_data': free.ward_data(),
               'dispersion': rows, 'response_q': q.tolist(), 'response_grid': grid,
               'feedback_motion': feedback_motion(matter, np.array([0., 0., .5])),
               'linearized_gauss_null_residual': collective_gauss(matter, q)[1],
               'spectral': {'squared_modes': values.tolist(), 'field_residues': weights.tolist(),
                            'field_residue_sum': float(weights.sum()),
                            'exact_matter_gaps_with_multiplicity': matter.gaps.tolist(),
                            'active_pair_gaps_per_three_loops': gaps.tolist()},
               'controls': {'frozen_zero_q_mass_eigenvalues': np.linalg.eigvalsh(frozen_static).real.tolist(),
                            'full_zero_q_static_max': float(np.max(np.abs(full_static))),
                            'removed_all_matter_squared_modes': removed_matter_squared_modes.tolist(),
                            'removed_matter_law': 'Same Hubbard matter H(theta) at link_charge=0; primitive electric unchanged',
                            'removed_matter_Pi_max': float(np.max(np.abs(decoupled_pi))),
                            'frozen_rejected': bool(np.linalg.norm(frozen_static-full_static) > .1),
                            'feedback_nonzero': bool(max(max(r['matter_fraction']) for r in rows) > .01)},
               'runtime': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                           'elapsed_seconds': time.time()-started},
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = Path(__file__).with_name('rich_photon_20261014_result.json')
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'law': LAW_ID, 'ward': receipt['ward_and_pair_data'],
                      'dispersion': rows, 'controls': receipt['controls'],
                      'output': str(path)}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(run())
