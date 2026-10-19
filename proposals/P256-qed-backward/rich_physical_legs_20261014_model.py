"""Full finite-H mixed spectral legs; no coordinate-index photon declaration.

All charged/neutral responses own their reference. Canonical whitening uses
actual M=epsilon G. The projected Fourier-Q law does not obey an assumed CCR.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from scipy import sparse as sp
from rich_field_packet_20261014_connected import ConnectedPacket


def encode(a):
    a = np.asarray(a)
    return dict(real=a.real.tolist(), imag=a.imag.tolist())


def expectation(op, state):
    return float(np.vdot(state, op @ state).real)


def metric_roots(m):
    values, vectors = eigh(m.M)
    return (vectors * np.sqrt(values)) @ vectors.T, (vectors / np.sqrt(values)) @ vectors.T


def stationary_spectrum(m):
    """Entire finite spectrum, with all degeneracy partners and no Krylov cutoff."""
    energies, vectors = eigh(m.H.toarray(), driver='evd', overwrite_a=True, check_finite=False)
    ground_indices = np.flatnonzero(energies - energies[0] < 2e-8)
    ground_basis = vectors[:, ground_indices]
    if m.occupation:
        axis = np.array([.31, -.49, .815]); axis /= np.linalg.norm(axis)
        spin = sum(axis[a] * m.S[a] for a in range(3))
        _, rotation = eigh(ground_basis.conj().T @ (spin @ ground_basis))
        reference = ground_basis @ rotation[:, -1]
    else:
        reference = ground_basis[:, 0]
    return energies, vectors, reference, ground_indices


def insertions(m, reference):
    sqrt_m, inverse_sqrt = metric_roots(m)
    field_q = np.column_stack([a @ reference for a in m.A])
    field_p = np.column_stack([a @ reference for a in m.P])
    current = np.column_stack([j @ reference for j in m.J])
    spin = np.column_stack([s @ reference for s in m.S]) if m.occupation else np.zeros((m.size, 3), complex)
    cm = np.column_stack([j @ reference for j in m.cm_current()]) if m.occupation else np.zeros((m.size, 3), complex)
    columns = np.column_stack([field_q @ sqrt_m, field_p @ inverse_sqrt, current @ inverse_sqrt, spin, cm])
    means = reference.conj() @ columns
    columns -= reference[:, None] * means
    return columns, field_q - reference[:, None] * (reference.conj() @ field_q), dict(sqrt_M=sqrt_m, inverse_sqrt_M=inverse_sqrt, means=means)


def blocks(energies, tolerance=2e-8):
    start = 0
    while start < len(energies):
        end = start + 1
        while end < len(energies) and energies[end] - energies[start] < tolerance:
            end += 1
        yield start, end
        start = end


def spectral_response(m, energies, vectors, reference, frequency, broadening):
    columns, raw_q, metric = insertions(m, reference)
    amplitudes = vectors.conj().T @ columns
    gaps = energies - energies[0]
    positive = 1 / (frequency + 1j * broadening - gaps)
    negative = 1 / (frequency + 1j * broadening + gaps)
    retarded = amplitudes.conj().T @ (positive[:, None] * amplitudes) - (amplitudes.conj().T @ (negative[:, None] * amplitudes)).T
    records = []
    n = m.nmodes
    for start, end in blocks(energies):
        t = amplitudes[start:end]
        fq, fp, j = t[:, :n], t[:, n:2*n], t[:, 2*n:3*n]
        cross = fq.conj().T @ j
        singular = np.linalg.svd(cross, compute_uv=False)
        # Rank threshold is explicit and absolute in canonical model units.
        rank = int(np.count_nonzero(singular > 1e-9)) if gaps[start] > 2e-8 else 0
        records.append(dict(start=start, end=end, degeneracy=end-start, gap=float(gaps[start]),
                            field_Q_residue=encode(fq.conj().T @ fq), canonical_P_residue=encode(fp.conj().T @ fp),
                            current_residue=encode(j.conj().T @ j), mixed_Q_current=encode(cross),
                            mixed_singular_values=singular.tolist(), bright_rank=rank,
                            field_trace=float(np.linalg.norm(fq)**2), current_trace=float(np.linalg.norm(j)**2)))
    contact = np.empty((n, n), complex)
    for a in range(n):
        for b in range(n):
            k = m.link_derivative((a,b)) + m.body_contact(a,b)
            contact[a,b] = np.vdot(reference, k @ reference)
    contact_canonical = metric['inverse_sqrt_M'] @ contact @ metric['inverse_sqrt_M']
    return dict(retarded=retarded, amplitudes=amplitudes, source_columns=columns, raw_q_columns=raw_q,
                metric=metric, records=records, contact=contact, canonical_contact=contact_canonical,
                source_Gram=columns.conj().T @ columns,
                spectral_Gram_residual=float(np.linalg.norm(amplitudes.conj().T @ amplitudes - columns.conj().T @ columns)))


def normalized_leg(m, energies, vectors, reference, coefficients, window):
    """A specified physical field source restricted to its own finite energy window.

    This is a global, generally matter-entangled prepared leg. The window is
    not a physical detector, spin measurement or evidence of signal transport.
    """
    coefficients = np.asarray(coefficients, complex)
    q = sum(coefficients[a] * m.A[a] for a in range(m.nmodes))
    source = q @ reference
    source -= reference * np.vdot(reference, source)
    amplitudes = vectors.conj().T @ source
    gaps = energies - energies[0]
    selected = (gaps >= window[0]) & (gaps <= window[1])
    selected &= gaps > 2e-8
    filtered = amplitudes * selected
    residue = float(np.vdot(filtered, filtered).real)
    if residue <= 1e-14:
        raise ValueError('Specified physical field source has no resolved energy-window residue')
    state = vectors @ filtered / np.sqrt(residue)
    omega = float(np.dot(abs(filtered)**2, gaps) / residue)
    variance = float(np.dot(abs(filtered)**2, (gaps-omega)**2) / residue)
    field_amplitude = np.array([np.vdot(reference, a @ state) for a in m.A])
    current_amplitude = np.array([np.vdot(reference, j @ state) for j in m.J])
    return dict(state=state, energy=omega, energy_variance=variance, residue=residue,
                field_amplitude=field_amplitude, current_amplitude=current_amplitude,
                coefficients=coefficients, window=window, indices=np.flatnonzero(selected),
                spectral_amplitudes=filtered / np.sqrt(residue), norm=float(np.vdot(state,state).real),
                field_current_overlap=float(np.linalg.norm(current_amplitude)))


def current_reaction(m, energies, vectors, reference, incoming, outgoing, broadening, wrong_noether=False):
    """Finite two-field current/contact response, not a cross section or S matrix.

    Its external vectors ARE actual <g|A|normalized leg> amplitudes. Every
    virtual eigenstate contributes, including dark/high-energy matter states.
    """
    ai = incoming['field_amplitude']; ao = outgoing['field_amplitude'].conj()
    if wrong_noether:
        # Deliberately wrong identification, same units not claimed.
        currents = [m.S[a % 3] for a in range(m.nmodes)]
    else:
        currents = m.J
    ji = sum(ai[a] * currents[a] for a in range(m.nmodes))
    jo = sum(ao[a] * currents[a] for a in range(m.nmodes))
    contact_state = np.zeros(m.size, complex)
    for a in range(m.nmodes):
        for b in range(m.nmodes):
            contact_state += ao[a]*ai[b]*((m.link_derivative((a,b))+m.body_contact(a,b)) @ reference)
    def resolvent(state, shift):
        return vectors @ ((vectors.conj().T @ state) / (energies[0] + shift + 1j*broadening - energies))
    positive = jo @ resolvent(ji @ reference, incoming['energy'])
    negative = ji @ resolvent(jo @ reference, -outgoing['energy'])
    reaction = contact_state + positive + negative
    return dict(state=reaction, norm=float(np.vdot(reaction,reaction).real),
                contact_norm=float(np.linalg.norm(contact_state)),
                no_contact_norm=float(np.linalg.norm(positive+negative)),
                incoming_energy=incoming['energy'], outgoing_energy=outgoing['energy'],
                incoming_residue=incoming['residue'], outgoing_residue=outgoing['residue'],
                incoming_vertex=ai, outgoing_vertex=ao, virtual_states=len(energies))


def frozen_supplier_spectrum(m, reference):
    """Explicit comparator: freeze supplier to its own reference reduced state.

    The dominant eigenvector is selected from the ACTUAL full-H supplier RDM;
    all virtual supplier directions are then absent, not secretly replayed.
    """
    tensor = reference.reshape(m.nfield*m.ns, 64)
    rdm = tensor.conj().T @ tensor
    values, vectors = eigh(rdm)
    supplier_state = vectors[:, -1].conj()
    embedding = sp.kron(sp.eye(m.nfield*m.ns), sp.csr_matrix(supplier_state[:,None]), format='csr')
    h = (embedding.getH() @ m.H @ embedding).toarray()
    energies, states = eigh(h)
    g = states[:,0]
    sqrt_m, inverse_sqrt = metric_roots(m)
    q = np.column_stack([(embedding.getH() @ a @ embedding) @ g for a in m.A]) @ sqrt_m
    j = np.column_stack([(embedding.getH() @ a @ embedding) @ g for a in m.J]) @ inverse_sqrt
    q -= g[:,None]*(g.conj()@q); j -= g[:,None]*(g.conj()@j)
    t = states.conj().T @ np.column_stack([q,j])
    result=[]
    for start,end in blocks(energies):
        cross=t[start:end,:m.nmodes].conj().T@t[start:end,m.nmodes:]
        singular=np.linalg.svd(cross,compute_uv=False)
        result.append(dict(gap=float(energies[start]-energies[0]),degeneracy=end-start,mixed_singular_values=singular.tolist(),bright_rank=int(np.count_nonzero(singular>1e-9)) if start else 0))
    return dict(supplier_RDM_largest_eigenvalue=float(values[-1]), dimension=len(energies), poles=result,
                missing_virtual_supplier=True, ground_energy=float(energies[0]))
