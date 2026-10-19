"""NEW correlated tetrahedral Hubbard matter and electric-only spatial field.
No earlier physical producer is imported. The finite whole H and explicitly
approximate extensive continuation share charges, geometry, Gauss and Peierls.
"""
from __future__ import annotations
from itertools import combinations, product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh


def hop(mask, dst, src):
    if not (mask >> src) & 1 or (mask >> dst) & 1:
        return None
    sign = (-1) ** ((mask & ((1 << src) - 1)).bit_count())
    mask ^= 1 << src
    sign *= (-1) ** ((mask & ((1 << dst) - 1)).bit_count())
    return mask | (1 << dst), sign


def matter(cfg):
    """Entire Nup=Ndown=1 tetrahedron, 16 states; three independent loops.
    Repeating this cell at fixed volume defines one correlated molecule per V.
    """
    lengths = np.asarray(cfg['box_lengths'], float)
    positions = cfg['tetrahedron_edge'] / (2 * np.sqrt(2)) * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    masks = sorted((1 << up) | (1 << (4 + down)) for up, down in product(range(4), repeat=2))
    lookup = {mask: i for i, mask in enumerate(masks)}
    densities = np.array([[(mask >> i & 1) + (mask >> (i + 4) & 1) for i in range(4)] for mask in masks])
    double = np.array([sum((mask >> i & 1) * (mask >> (i + 4) & 1) for i in range(4)) for mask in masks])
    diagonal = cfg['hubbard_U'] * double + densities @ np.asarray(cfg['site_potentials'])
    # Periodic longitudinal Gauss energy is separate from the transverse field.
    # Literal +1/2 ions at all four sites remove the charge zero mode exactly.
    gauss = np.zeros(16)
    ionic = np.full(4, 0.5)
    for integers in product(range(-cfg['gauss_cut'], cfg['gauss_cut'] + 1), repeat=3):
        if integers == (0, 0, 0):
            continue
        k = 2 * np.pi * np.array(integers) / lengths
        k2 = k @ k
        rho = cfg['charge'] * (densities - ionic) @ np.exp(1j * positions @ k)
        gauss += np.exp(-cfg['charge_radius'] ** 2 * k2) * abs(rho) ** 2 / (2 * cfg['epsilon'] * np.prod(lengths) * k2)
    diagonal += gauss
    bonds, operators = list(combinations(range(4), 2)), []
    for src, dst in bonds:
        operator = np.zeros((16, 16), complex)
        for spin in range(2):
            for col, mask in enumerate(masks):
                step = hop(mask, dst + 4 * spin, src + 4 * spin)
                if step is not None:
                    target, sign = step
                    operator[lookup[target], col] += sign
        operators.append(operator)
    return dict(positions=positions, lengths=lengths, volume=float(np.prod(lengths)),
                source_density=1 / float(np.prod(lengths)), independent_loop_density=3 / float(np.prod(lengths)),
                diagonal=diagonal, gauss_diagonal=gauss, densities=densities, double=double,
                bonds=bonds, hops=np.array(operators), masks=masks, dimension=16)


def polarizations(q):
    q = np.asarray(q, float)
    direction = q / np.linalg.norm(q) if np.linalg.norm(q) else np.array([0., 0., 1.])
    reference = np.eye(3)[np.argmin(abs(direction))]
    first = np.cross(direction, reference); first /= np.linalg.norm(first)
    return np.array([first, np.cross(direction, first)])


def line_integrals(mat, q, phase=0., normalized_standing=True):
    """EXACT edge integral of e_lambda sqrt(2) cos(q.r+phase), no fitted q."""
    q = np.asarray(q, float); pols = polarizations(q)
    values = []
    for src, dst in mat['bonds']:
        delta = mat['positions'][dst] - mat['positions'][src]
        midpoint = (mat['positions'][dst] + mat['positions'][src]) / 2
        scalar = np.sinc((q @ delta) / (2 * np.pi)) * np.cos(q @ midpoint + phase)
        values.append((pols @ delta) * scalar * (np.sqrt(2.) if normalized_standing else 1.))
    return np.array(values)


def molecular_h(mat, cfg, coordinates=None, coefficients=None, dw=None):
    amplitudes = np.ones(6) if dw is None else np.asarray(dw)
    phases = np.zeros(6) if coordinates is None else coefficients @ coordinates
    h = np.diag(mat['diagonal']).astype(complex)
    for t, weight, phase in zip(mat['hops'], amplitudes, phases):
        link = np.exp(1j * phase) * t
        h -= cfg['hopping'] * weight * (link + link.conj().T)
    return h


def vertices(mat, cfg, coefficients, coordinates=None, dw=None):
    amplitudes = np.ones(6) if dw is None else np.asarray(dw)
    phases = np.zeros(6) if coordinates is None else coefficients @ coordinates
    current = np.zeros((2, 16, 16), complex); contact = np.zeros((2, 2, 16, 16), complex)
    for t, weight, phase, b in zip(mat['hops'], amplitudes, phases, coefficients):
        link = np.exp(1j * phase) * t
        for a in range(2):
            current[a] += 1j * cfg['hopping'] * weight * b[a] * (link - link.conj().T)
            for c in range(2):
                contact[a, c] += cfg['hopping'] * weight * b[a] * b[c] * (link + link.conj().T)
    return current, contact


def molecular_response(mat, cfg, coefficients, dw=None):
    energy, vectors = eigh(molecular_h(mat, cfg, dw=dw))
    ground = vectors[:, 0]; current, contact = vertices(mat, cfg, coefficients, dw=dw)
    transition = np.array([vectors[:, 1:].conj().T @ j @ ground for j in current])
    diamagnetic = np.einsum('i,abij,j->ab', ground.conj(), contact, ground).real
    return dict(energies=energy, vectors=vectors, gaps=energy[1:] - energy[0], transition=transition,
                contact=diamagnetic, double_occupancy=float(np.vdot(ground, mat['double'] * ground).real),
                density=np.einsum('i,ia,i->a', ground.conj(), mat['densities'], ground).real,
                connected_density=np.einsum('i,ia,ib,i->ab', ground.conj(), mat['densities'], mat['densities'], ground).real
                                  - np.outer(np.einsum('i,ia,i->a', ground.conj(), mat['densities'], ground).real,
                                             np.einsum('i,ia,i->a', ground.conj(), mat['densities'], ground).real))


def collective(mat, cfg, q, dw, bulk=False, contact_scale=1., normalization_scale=1.):
    """Full SIGNED correlated cluster Lehmann, represented by transition pairs.
    Gaussian/RPA APPROXIMATION only. Auxiliary oscillators are existing matter
    transitions: the microscopic field has NO A^2, B^2 or Maxwell restoring term.
    """
    phases = [0., np.pi / 2, np.pi, 3 * np.pi / 2] if bulk else [cfg['standing_phase']]
    canonical = cfg['charge'] * np.sqrt(normalization_scale / (cfg['epsilon'] * mat['volume']))
    contact = np.zeros((2, 2)); couplings = []; gaps = []
    for phase in phases:
        coefficients = line_integrals(mat, q, phase) * canonical
        response = molecular_response(mat, cfg, coefficients, dw)
        # H(0) real: all physical current transition amplitudes are imaginary.
        j = response['transition'].imag / np.sqrt(len(phases))
        contact += response['contact'] / len(phases)
        couplings.append(j * np.sqrt(2 * response['gaps'])[None, :]); gaps.extend(response['gaps'])
    coupling = np.concatenate(couplings, axis=1); gaps = np.array(gaps)
    k = np.diag(np.r_[np.zeros(2), gaps ** 2]); k[:2, :2] = contact_scale * contact
    k[:2, 2:] = coupling; k[2:, :2] = coupling.T
    eigenvalues, eigenvectors = eigh(k); stable = bool(eigenvalues[0] > 1.e-12)
    frequencies = np.sqrt(np.maximum(eigenvalues, 0.)); field_vectors = eigenvectors[:2]
    residues = field_vectors ** 2 / (2 * np.maximum(frequencies, 1.e-30))[None, :]
    covariance = (field_vectors / (2 * np.maximum(frequencies, 1.e-30))[None, :]) @ field_vectors.T if stable else None
    static = contact_scale * contact - (coupling / gaps[None, :] ** 2) @ coupling.T
    return dict(stable=stable, eigenvalues=eigenvalues, frequencies=frequencies, field_vectors=field_vectors,
                residues=residues, covariance=covariance, contact=contact, static=static,
                transition_count=len(gaps), response=response,
                imaginary_phase_defect=float(np.max(abs(response['transition'].real))))


def selfconsistent(mat, cfg, q=None, bulk=False):
    """Own D energies/residues -> DW bonds -> correlated current/reaction -> D.
    Explicit UV/IR-regulated Gaussian DW/RPA closure, NOT exact thermodynamics.
    No supplied bath, free-Dirac polarization, fit of c, or one-pass frozen chi.
    """
    dw = np.ones(6); history = []; mesh = cfg['bulk_mesh_spacing']
    qs = [mesh * np.array(k, float) for k in product([-1, 0, 1], repeat=3) if k != (0, 0, 0)] if bulk else [np.asarray(q)]
    for iteration in range(cfg['feedback_iterations']):
        variance = np.zeros(6); stable = True
        for momentum in qs:
            mode = collective(mat, cfg, momentum, dw, bulk=bulk)
            if not mode['stable']:
                stable = False; break
            if bulk:
                # x=sqrt(epsilon V) A_q, and sum_q/V -> integral d^3q/(2pi)^3.
                factor = cfg['charge'] ** 2 * mesh ** 3 / ((2 * np.pi) ** 3 * cfg['epsilon'])
                phases = [0., np.pi/2, np.pi, 3*np.pi/2]
            else:
                factor = cfg['charge'] ** 2 / (cfg['epsilon'] * mat['volume'])
                phases = [cfg['standing_phase']]
            for phase in phases:
                b = line_integrals(mat, momentum, phase)
                variance += factor * np.einsum('ba,ac,bc->b', b, mode['covariance'], b) / len(phases)
        if not stable:
            return dict(stable=False, converged=False, dw=dw, history=history,
                        unstable_q=momentum, smallest_eigenvalue=float(mode['eigenvalues'][0]))
        target = np.exp(-variance / 2); error = float(np.max(abs(target - dw)))
        history.append(dict(iteration=iteration, error=error, dw=dw.tolist(), target=target.tolist()))
        dw = cfg['feedback_mixing'] * target + (1 - cfg['feedback_mixing']) * dw
        if error < cfg['feedback_tolerance']:
            break
    return dict(stable=True, converged=error < cfg['feedback_tolerance'], dw=dw, history=history,
                momentum_count=len(qs), fluctuation_variance=variance)


def field_grid(cfg, box=None):
    size = cfg['field_points']; wall = cfg['field_wall'] if box is None else box
    step = 2 * wall / (size + 1); coordinate = np.linspace(-wall + step, wall - step, size)
    kinetic = sparse.diags([-np.ones(size-1), 2*np.ones(size), -np.ones(size-1)], [-1, 0, 1]) / (2 * step ** 2)
    electric = sparse.kronsum(kinetic, kinetic, format='csr')
    return np.array(list(product(coordinate, repeat=2))), electric, step


def whole_h(mat, cfg, q, box=None):
    coordinates, electric, step = field_grid(cfg, box)
    coefficients = line_integrals(mat, q, cfg['standing_phase']) * cfg['charge'] / np.sqrt(cfg['epsilon'] * mat['volume'])
    hs = [molecular_h(mat, cfg, x, coefficients) for x in coordinates]
    h = sparse.kron(electric, sparse.eye(16), format='csr') + sparse.block_diag(hs, format='csr')
    vertex = [vertices(mat, cfg, coefficients, x) for x in coordinates]
    currents = [sparse.block_diag([sparse.csr_matrix(v[0][a]) for v in vertex], format='csr') for a in range(2)]
    contacts = [[sparse.block_diag([sparse.csr_matrix(v[1][a, b]) for v in vertex], format='csr') for b in range(2)] for a in range(2)]
    observables = [sparse.diags(np.repeat(coordinates[:, a], 16), format='csr') for a in range(2)]
    return dict(h=h, coordinates=coordinates, electric=electric, step=step, coefficients=coefficients,
                currents=currents, contacts=contacts, Q=observables)


def signed_lehmann(gaps, amplitudes, z):
    """Retarded commutator, both signs and ALL retained excited states."""
    plus = (amplitudes.conj() / (z - gaps)[None, :]) @ amplitudes.T
    minus = (amplitudes / (z + gaps)[None, :]) @ amplitudes.conj().T
    return plus - minus


def exact_consumer(mat, cfg, q, box=None):
    law = whole_h(mat, cfg, q, box); h = law['h']; dimension = h.shape[0]
    energy, vectors = eigh(h.toarray(), driver='evd')
    ground = vectors[:, 0]; gaps = energy[1:] - energy[0]
    coordinate_amp = np.array([vectors[:, 1:].conj().T @ operator @ ground for operator in law['Q']])
    current_amp = np.array([vectors[:, 1:].conj().T @ operator @ ground for operator in law['currents']])
    coordinate_weights = abs(coordinate_amp) ** 2; current_weights = abs(current_amp) ** 2
    contact = np.array([[np.vdot(ground, law['contacts'][a][b] @ ground).real for b in range(2)] for a in range(2)])
    reduced = ground.reshape(-1, 16); matter_rho = reduced.conj().T @ reduced
    probability = np.sum(abs(reduced) ** 2, axis=1).reshape(cfg['field_points'], cfg['field_points'])
    boundary = float(probability[0].sum() + probability[-1].sum() + probability[1:-1, 0].sum() + probability[1:-1, -1].sum())
    resolvents = []
    for omega in cfg['response_frequencies']:
        z = complex(omega, cfg['response_eta']); d = signed_lehmann(gaps, coordinate_amp, z); j = signed_lehmann(gaps, current_amp, z)
        resolvents.append(dict(omega=omega, D_real=d.real.tolist(), D_imag=d.imag.tolist(), current_real=j.real.tolist(), current_imag=j.imag.tolist()))
    # Physical field-only unitary, then unconditioned exact ONE-H reaction.
    kick = np.exp(1j * cfg['field_kick'] * np.repeat(law['coordinates'][:, 0], 16)) * ground
    times = np.linspace(0., cfg['reaction_time'], cfg['reaction_steps']); initial = vectors.conj().T @ kick
    electric_full = sparse.kron(law['electric'], sparse.eye(16), format='csr')
    double_operator = sparse.kron(sparse.eye(len(law['coordinates'])), sparse.diags(mat['double']))
    baseline_e = float(np.vdot(ground, electric_full @ ground).real); baseline_double = float(np.vdot(ground, double_operator @ ground).real)
    reaction = []
    for time in times:
        state = vectors @ (np.exp(-1j * (energy - energy[0]) * time) * initial)
        reaction.append(dict(time=float(time), norm=float(np.vdot(state, state).real), electric_excess=float(np.vdot(state, electric_full @ state).real - baseline_e),
                             double_occupancy_change=float(np.vdot(state, double_operator @ state).real - baseline_double),
                             current=[float(np.vdot(state, j @ state).real) for j in law['currents']]))
    # Genuinely frozen matter: project microscopic H_m(Q), but not its response,
    # onto one unchanging bare ground. It has no virtual current relaxation.
    bare_molecular = molecular_response(mat, cfg, law['coefficients']); mg = bare_molecular['vectors'][:, 0]
    frozen_potential = np.array([np.vdot(mg, molecular_h(mat, cfg, x, law['coefficients']) @ mg).real for x in law['coordinates']])
    frozen_energy, frozen_vectors = eigh(law['electric'].toarray() + np.diag(frozen_potential))
    frozen_weights = np.array([abs(frozen_vectors[:, 1:].conj().T @ (law['coordinates'][:, a] * frozen_vectors[:, 0])) ** 2 for a in range(2)])
    bare_energy, bare_vectors = eigh(law['electric'].toarray())
    bare_weights = np.array([abs(bare_vectors[:, 1:].T @ (law['coordinates'][:, a] * bare_vectors[:, 0])) ** 2 for a in range(2)])
    # Exact bond-density continuity at a generic nonzero field point.
    position = np.array([0.47, -0.31]); hm = molecular_h(mat, cfg, position, law['coefficients']); continuity = []
    for site in range(4):
        incoming = np.zeros((16, 16), complex)
        for (src, dst), op, b in zip(mat['bonds'], mat['hops'], law['coefficients']):
            link = np.exp(1j * b @ position) * op
            incoming += ((1 if site == dst else 0) - (1 if site == src else 0)) * 1j * cfg['hopping'] * (link - link.conj().T)
        density = np.diag(mat['densities'][:, site]); continuity.append(float(np.linalg.norm(1j * (hm @ density - density @ hm) - incoming)))
    residuals = np.linalg.norm(h @ vectors - vectors * energy[None, :], axis=0)
    return dict(q=np.asarray(q).tolist(), dimension=dimension, ground_energy=float(energy[0]), gaps=gaps.tolist(),
                coordinate_weights=coordinate_weights.tolist(), current_weights=current_weights.tolist(), matter_purity=float(np.trace(matter_rho @ matter_rho).real),
                field_boundary_probability=boundary, contact=contact.tolist(), full_retarded=resolvents, reaction=reaction,
                max_eigen_residual=float(max(residuals)), continuity_defect=max(continuity),
                frozen_gaps=(frozen_energy[1:] - frozen_energy[0]).tolist(), frozen_coordinate_weights=frozen_weights.tolist(),
                bare_electric_gaps=(bare_energy[1:] - bare_energy[0]).tolist(), bare_coordinate_weights=bare_weights.tolist(),
                covariance=np.array([[np.vdot(ground, a @ b @ ground).real for b in law['Q']] for a in law['Q']]).tolist(),
                baseline_double_occupancy=baseline_double), dict(energies=energy, vectors=vectors, field_probability=probability)
