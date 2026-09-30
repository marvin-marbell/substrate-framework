"""Conditional quantum-rotor supplier; run directly for a JSON calculation.

This imports a quantum Hilbert space, rotor algebra and matter CAR. It does
not derive electron statistics, spin, Dirac propagation or a Compton amplitude.
The finite star-graph charge sector is EXACT for the stated hopping Hamiltonian;
the auxiliary rotor cutoff is not an approximation to its physical dynamics.
The separate cubic photon calculation is a harmonic, noncompact approximation.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm

EDGES = ((0, 1), (0, 2), (0, 3))
BACKGROUND = np.array([0, 1, 1, 0], dtype=int)


def matter_action(mask: int, site: int, create: bool, fermionic: bool):
    """Occupation-basis CAR, or commuting-site hard-core bosons."""
    occupied = bool(mask & (1 << site))
    if occupied == create:
        return None
    sign = -1 if fermionic and (mask & ((1 << site) - 1)).bit_count() % 2 else 1
    return mask ^ (1 << site), sign


def occupation(mask: int) -> np.ndarray:
    return np.array([(mask >> i) & 1 for i in range(4)], dtype=int)


def gauss(state) -> np.ndarray:
    mask, *electric = state
    divergence = np.array([sum(electric), *(-np.array(electric))])
    return divergence - occupation(mask) + BACKGROUND


def hopping(states, fermionic: bool, covariant: bool):
    """T_l=c_y^dagger U_l^dagger c_x on x->y; no cyclic cutoff wrap."""
    lookup = {state: i for i, state in enumerate(states)}
    matrices = []
    for link, (source, target) in enumerate(EDGES):
        rows, cols, data = [], [], []
        for column, state in enumerate(states):
            first = matter_action(state[0], source, False, fermionic)
            if first is None:
                continue
            second = matter_action(first[0], target, True, fermionic)
            if second is None:
                continue
            electric = list(state[1:])
            if covariant:
                electric[link] -= 1
            output = (second[0], *electric)
            if output in lookup:
                rows.append(lookup[output])
                cols.append(column)
                data.append(first[1] * second[1])
        matrices.append(sparse.csr_matrix((data, (rows, cols)),
                                         shape=(len(states), len(states)), dtype=float))
    return matrices


def commutator_diagonal(matrix, diagonal):
    """[diag(diagonal), matrix], preserving sparse storage."""
    return matrix.multiply(diagonal[:, None]) - matrix.multiply(diagonal[None, :])


def evolve(hamiltonian, initial, time):
    energies, vectors = np.linalg.eigh(hamiltonian)
    state = vectors @ (np.exp(-1j * time * energies) * (vectors.conj().T @ initial))
    return energies, state


def algebra_report():
    size = 16
    identity = np.eye(size)
    operators = []
    for site in range(4):
        annihilator = np.zeros((size, size))
        for mask in range(size):
            output = matter_action(mask, site, False, True)
            if output is not None:
                annihilator[output[0], mask] = output[1]
        operators.append(annihilator)
    car_residual = max(float(np.linalg.norm(
        a @ b.T + b.T @ a - (identity if i == j else 0)))
        for i, a in enumerate(operators) for j, b in enumerate(operators))
    annihilator_residual = max(float(np.linalg.norm(a @ b + b @ a))
                              for a in operators for b in operators)
    electric = np.diag(np.arange(-2, 3, dtype=float))
    shift = np.diag(np.ones(4), k=-1)
    return {
        "matter_hilbert_dimension": size,
        "CAR_imported_not_emergent": True,
        "CAR_residual": car_residual,
        "annihilator_anticommutator_residual": annihilator_residual,
        "rotor_auxiliary_bounds": [-2, 2],
        "rotor_commutator_residual": float(np.linalg.norm(electric @ shift - shift @ electric - shift)),
        "rotor_upper_unitarity_defect": float(np.linalg.norm(shift.T @ shift - np.eye(5))),
        "rotor_lower_unitarity_defect": float(np.linalg.norm(shift @ shift.T - np.eye(5))),
        "cutoff_wrap_used": False,
    }


def many_body_report(fermionic: bool, hopping_strength=1.0, electric_stiffness=1.0, time=0.7):
    masks = [mask for mask in range(16) if mask.bit_count() == 2]
    states = [(mask, *electric) for mask in masks
              for electric in itertools.product(range(-2, 3), repeat=3)]
    physical_indices = [i for i, state in enumerate(states) if not np.any(gauss(state))]
    physical = [states[i] for i in physical_indices]
    constraint = np.array([gauss(state) for state in states])
    legal_links = hopping(states, fermionic, True)
    bare_links = hopping(states, fermionic, False)
    legal = -hopping_strength * sum((link + link.T for link in legal_links),
                                   sparse.csr_matrix((len(states), len(states))))
    bare = -hopping_strength * sum((link + link.T for link in bare_links),
                                  sparse.csr_matrix((len(states), len(states))))
    electric_energy = electric_stiffness / 2 * np.array([
        sum(field * field for field in state[1:]) for state in states])
    hamiltonian = legal + sparse.diags(electric_energy)
    gauss_commutator = max(float(sparse_norm(commutator_diagonal(legal, constraint[:, i])))
                          for i in range(4))
    bare_commutator = max(float(sparse_norm(commutator_diagonal(bare, constraint[:, i])))
                         for i in range(4))
    physical_index_set = set(physical_indices)
    outside = [i for i in range(len(states)) if i not in physical_index_set]
    leakage = float(sparse_norm(legal[outside, :][:, physical_indices]))
    clipped_transitions = 0
    for mask, *electric in physical:
        for link, (source, target) in enumerate(EDGES):
            for start, end, delta in ((source, target, -1), (target, source, 1)):
                if (mask >> start) & 1 and not ((mask >> end) & 1):
                    clipped_transitions += int(not -2 <= electric[link] + delta <= 2)
    reduced = hamiltonian[physical_indices, :][:, physical_indices].toarray()
    initial_index = physical.index((6, 0, 0, 0))
    initial = np.eye(len(physical), dtype=complex)[:, initial_index]
    energies, evolved = evolve(reduced, initial, time)
    # Bare hopping evolves the six occupations with every link fixed at E=0.
    bare_indices = [states.index((mask, 0, 0, 0)) for mask in masks]
    bare_initial = np.eye(len(masks), dtype=complex)[:, masks.index(6)]
    _, forbidden = evolve(bare[bare_indices, :][:, bare_indices].toarray(), bare_initial, time)
    bare_gauss_squared = np.array([sum(gauss((mask, 0, 0, 0)) ** 2) for mask in masks])
    # Six legal moves exchange the two initial particles, returning all E's to zero.
    exchange_path = ((1, 0), (0, 3), (2, 0), (0, 1), (3, 0), (0, 2))
    current = initial_index
    exchange_amplitude = 1.0
    path_states = [physical[current]]
    for start, end in exchange_path:
        link_number = next(i for i, edge in enumerate(EDGES) if set(edge) == {start, end})
        operator = legal_links[link_number]
        if (start, end) != EDGES[link_number]:
            operator = operator.T.tocsr()
        column = operator[physical_indices, :][:, physical_indices][:, [current]].tocoo()
        if len(column.data) != 1:
            raise RuntimeError("Exchange path does not have exactly one allowed transition")
        exchange_amplitude *= float(column.data[0])
        current = int(column.row[0])
        path_states.append(physical[current])
    number = np.array([occupation(state[0]) for state in states])
    continuity_residual = 0.0
    for site in range(4):
        # i[H,n_x] equals lattice divergence of i[H,E_l].
        n_dot = -1j * commutator_diagonal(legal, number[:, site])
        divergence_dot = sparse.csr_matrix(legal.shape, dtype=complex)
        for link, (source, target) in enumerate(EDGES):
            orientation = int(site == source) - int(site == target)
            if orientation:
                e_dot = -1j * commutator_diagonal(legal, np.array([s[link + 1] for s in states]))
                divergence_dot += orientation * e_dot
        continuity_residual = max(continuity_residual, float(sparse_norm(n_dot - divergence_dot)))
    return {
        "statistics": "imported CAR fermions" if fermionic else "commuting-site hard-core bosons",
        "graph": "four-site star, center 0, oriented links 0->1,0->2,0->3",
        "hopping_strength": hopping_strength, "electric_stiffness": electric_stiffness,
        "time": time, "background_number": BACKGROUND.tolist(),
        "auxiliary_dimension": len(states), "exact_physical_dimension": len(physical),
        "physical_states_mask_E": [list(state) for state in physical],
        "physical_max_abs_E": max(abs(field) for state in physical for field in state[1:]),
        "physical_clipped_transition_count": clipped_transitions,
        "legal_Gauss_commutator_norm": gauss_commutator,
        "bare_Gauss_commutator_norm": bare_commutator,
        "legal_physical_leakage_norm": leakage,
        "continuity_residual": continuity_residual,
        "legal_energies": energies.tolist(),
        "legal_evolved_probabilities": (abs(evolved) ** 2).tolist(),
        "legal_departure_probability": float(1 - abs(evolved[initial_index]) ** 2),
        "bare_evolved_Gauss_squared": float(np.dot(abs(forbidden) ** 2, bare_gauss_squared)),
        "exchange_path_states_mask_E": [list(state) for state in path_states],
        "exchange_returns_initial_state": current == initial_index,
        "exchange_ordered_matrix_element_product": exchange_amplitude,
        "physical_evolved_Gauss_squared": float(np.dot(abs(evolved) ** 2,
            [sum(gauss(state) ** 2) for state in physical])),
    }


def photon_modes(momentum, a=1.0, electric_stiffness=1.0, magnetic_stiffness=1.0):
    """Midpoint Fourier convention: curl^dagger curl = d^2 I - d d^T."""
    momentum = np.asarray(momentum, dtype=float)
    d = 2 * np.sin(a * momentum / 2)
    d_squared = float(d @ d)
    curl_squared = d_squared * np.eye(3) - np.outer(d, d)
    eigenvalues, eigenvectors = np.linalg.eigh(curl_squared)
    omega_squared = electric_stiffness * magnetic_stiffness * eigenvalues
    # The gap is proportional to |d|^2, so an absolute floor would erase soft photons.
    # At exactly d=0 all eigenvalues vanish: these are global modes, not this pair.
    transverse = eigenvalues > d_squared * 1e-10
    omega = np.sqrt(np.maximum(omega_squared[transverse], 0))
    speed = a * np.sqrt(electric_stiffness * magnetic_stiffness)
    continuum = float(speed * np.linalg.norm(momentum))
    return {
        "momentum": momentum.tolist(), "lattice_d": d.tolist(),
        "curl_squared_eigenvalues": eigenvalues.tolist(),
        "physical_rank": int(np.count_nonzero(transverse)),
        "physical_omega": omega.tolist(), "continuum_omega": continuum,
        "omega_over_continuum": (omega / continuum).tolist() if continuum else [],
        "longitudinal_null_residual": float(np.linalg.norm(curl_squared @ d)),
        "transverse_vectors": eigenvectors[:, transverse].T.tolist(),
        "transverse_dot_d_residual": float(np.linalg.norm(eigenvectors[:, transverse].T @ d)),
        "ground_state_theta_variance_per_mode": (electric_stiffness / (2 * omega)).tolist(),
    }


def coulomb_report(size=8, electric_stiffness=1.0):
    """Neutral periodic source response; zero mode explicitly excluded."""
    charge = np.zeros((size, size, size))
    charge[0, 0, 0] = 1
    charge[size // 2, 0, 0] = -1
    frequency = 2 * np.pi * np.fft.fftfreq(size)
    px, py, pz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    laplacian = 4 * (np.sin(px / 2) ** 2 + np.sin(py / 2) ** 2 + np.sin(pz / 2) ** 2)
    inverse = np.zeros_like(laplacian)
    np.divide(1, laplacian, out=inverse, where=laplacian > 0)
    potential = np.fft.ifftn(np.fft.fftn(charge) * inverse).real
    electric = [potential - np.roll(potential, -1, axis=i) for i in range(3)]
    divergence = sum(field - np.roll(field, 1, axis=i) for i, field in enumerate(electric))
    energy = electric_stiffness / 2 * sum(float(np.sum(field * field)) for field in electric)
    response_energy = electric_stiffness / 2 * float(np.sum(charge * potential))
    return {
        "periodic_size": size, "number_charge_sum": float(np.sum(charge)),
        "zero_mode_removed": True, "Gauss_response_residual": float(np.max(abs(divergence - charge))),
        "electric_energy": energy, "inverse_laplacian_energy": response_energy,
        "energy_identity_residual": abs(energy - response_energy),
        "source_potential_difference": float(potential[0, 0, 0] - potential[size // 2, 0, 0]),
        "scope": "harmonic real E minimizer, not an exact compact integer-flux eigenstate",
    }


def report():
    fermions = many_body_report(True)
    bosons = many_body_report(False)
    directions = (np.array([1.0, 0, 0]), np.ones(3) / np.sqrt(3),
                  np.array([1.0, 2, 3]) / np.sqrt(14))
    photons = [photon_modes(radius * direction) for radius in (0.05, 0.5, 1.5)
               for direction in directions]
    coulomb = coulomb_report()
    algebra = algebra_report()
    soft_photons = [photon_modes(radius * direction) for radius in (1e-8, 1e-10)
                    for direction in directions]
    zero_momentum = photon_modes(np.zeros(3))
    checks = {
        "exact_CAR": algebra["CAR_residual"] < 1e-12 and algebra["annihilator_anticommutator_residual"] < 1e-12,
        "legal_constraint_and_nontrivial_dynamics": all(
            result["legal_Gauss_commutator_norm"] < 1e-12
            and result["legal_physical_leakage_norm"] < 1e-12
            and result["physical_clipped_transition_count"] == 0
            and result["continuity_residual"] < 1e-12
            and result["legal_departure_probability"] > 1e-3 for result in (fermions, bosons)),
        "bare_hopping_rejected": all(result["bare_Gauss_commutator_norm"] > 1
                                     and result["bare_evolved_Gauss_squared"] > 1e-3
                                     for result in (fermions, bosons)),
        "exchange_discriminates_statistics": fermions["exchange_returns_initial_state"]
            and bosons["exchange_returns_initial_state"]
            and fermions["exchange_ordered_matrix_element_product"] == -1
            and bosons["exchange_ordered_matrix_element_product"] == 1,
        "two_transverse_modes": all(mode["physical_rank"] == 2
            and mode["longitudinal_null_residual"] < 1e-12
            and mode["transverse_dot_d_residual"] < 1e-12 for mode in photons),
        "soft_modes_not_erased": all(mode["physical_rank"] == 2
            and np.allclose(mode["omega_over_continuum"], [1., 1.], rtol=1e-12, atol=0.)
            for mode in soft_photons) and zero_momentum["physical_rank"] == 0,
        "finite_lattice_anisotropy_detected": abs(photons[6]["physical_omega"][0]
                                                  - photons[7]["physical_omega"][0]) > 1e-3,
        "neutral_Coulomb_response": coulomb["Gauss_response_residual"] < 1e-12
            and coulomb["energy_identity_residual"] < 1e-12,
    }
    return {
        "status": "conditional supplier attempt, not QED or electron explanation",
        "units": "hbar=1, default a=U=K=t=1; no electron mass identified",
        "algebra": algebra, "fermionic_matter": fermions, "bosonic_false_positive": bosons,
        "photon_modes": photons, "Coulomb_response": coulomb, "checks": checks,
        "soft_photon_modes": soft_photons, "zero_momentum_global_mode": zero_momentum,
        "all_checks_passed": all(checks.values()),
    }


def main():
    result = report()
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
