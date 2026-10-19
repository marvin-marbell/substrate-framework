#!/usr/bin/env python3
"""Finite-volume compact U(1) supplier probe, not a thermodynamic phase proof."""
from __future__ import annotations

import argparse
import itertools
import json
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

PLANES = tuple(itertools.combinations(range(4), 2))
CUBES = tuple(itertools.combinations(range(4), 3))


def forward(field: np.ndarray, axis: int) -> np.ndarray:
    return np.roll(field, -1, axis=axis)


def difference(field: np.ndarray, axis: int) -> np.ndarray:
    return forward(field, axis) - field


def principal(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def curl(links: np.ndarray, mu: int, nu: int) -> np.ndarray:
    return difference(links[nu], mu) - difference(links[mu], nu)


def action(links: np.ndarray, beta: float) -> float:
    return float(-beta * sum(np.cos(curl(links, mu, nu)).sum() for mu, nu in PLANES))


def staple(links: np.ndarray, mu: int) -> np.ndarray:
    """Conditional weight exp(beta Re(exp(i theta_mu) staple))."""
    result = np.zeros(links.shape[1:], dtype=complex)
    for nu in range(4):
        if nu == mu:
            continue
        positive = forward(links[nu], mu) - forward(links[mu], nu) - links[nu]
        shifted_mu = np.roll(links[mu], 1, axis=nu)
        shifted_nu = np.roll(links[nu], 1, axis=nu)
        negative = -shifted_mu - forward(shifted_nu, mu) + shifted_nu
        result += np.exp(1j * positive) + np.exp(1j * negative)
    return result


@dataclass
class CompactGauge:
    size: int
    beta: float
    seed: int
    initial: str = "ordered"

    def __post_init__(self) -> None:
        if self.size < 4 or self.size % 2 or self.beta <= 0:
            raise ValueError("Periodic checkerboard heat bath requires even size>=4 and beta>0")
        self.rng = np.random.default_rng(self.seed)
        self.links = np.zeros((4,) + (self.size,) * 4)
        if self.initial == "random":
            self.links = self.rng.uniform(-np.pi, np.pi, self.links.shape)
        elif self.initial != "ordered":
            raise ValueError("Initial state must be ordered or random")
        self.coordinates = np.indices((self.size,) * 4)
        self.parity = np.indices((self.size,) * 4).sum(axis=0) % 2

    def sweep(self) -> None:
        for mu in self.rng.permutation(4):
            for color in self.rng.permutation(2):
                mask = self.parity == color
                field = staple(self.links, int(mu))[mask]
                self.links[mu][mask] = self.rng.vonmises(-np.angle(field), self.beta * np.abs(field))
        self.flux_move()

    def flux_move(self) -> None:
        """Mix integer global flux sectors; monopole equilibration is not guaranteed."""
        mu, nu = PLANES[int(self.rng.integers(len(PLANES)))]
        delta = (1 if self.rng.integers(2) else -1) * 2 * np.pi / self.size ** 2
        plaquette = curl(self.links, mu, nu)
        change = -self.beta * float((np.cos(plaquette + delta) - np.cos(plaquette)).sum())
        if np.log(self.rng.random()) < -change:
            self.links[nu] = principal(self.links[nu] + delta * self.coordinates[mu])
            boundary = self.coordinates[mu] == self.size - 1
            self.links[mu] = principal(self.links[mu]
                                      - delta * self.size * self.coordinates[nu] * boundary)

    def observables(self) -> dict:
        plaquettes = {plane: curl(self.links, *plane) for plane in PLANES}
        wrapped = {plane: principal(value) for plane, value in plaquettes.items()}
        volume = self.size ** 4
        cosines = np.array([np.cos(plaquettes[plane]).mean() for plane in PLANES])
        sums = np.array([np.sin(plaquettes[plane]).sum() for plane in PLANES])
        cubes = {}
        for mu, nu, rho in CUBES:
            boundary = (difference(wrapped[nu, rho], mu)
                        - difference(wrapped[mu, rho], nu)
                        + difference(wrapped[mu, nu], rho))
            cubes[mu, nu, rho] = np.rint(boundary / (2 * np.pi))
        closure = (difference(cubes[1, 2, 3], 0) - difference(cubes[0, 2, 3], 1)
                   + difference(cubes[0, 1, 3], 2) - difference(cubes[0, 1, 2], 3))
        assert np.max(np.abs(closure)) == 0, "Monopole current endpoints would violate d squared=0"
        # Six-plane average. This is an estimator, not clipped to force positivity.
        helicity = self.beta * cosines.mean() - self.beta ** 2 * np.mean(sums ** 2) / volume
        cosine = float(cosines.mean())
        density = float(np.mean([np.mean(np.abs(value)) for value in cubes.values()]))
        # Spatial momentum in direction 1. F12/F13 are the two magnetic transverse
        # channels. F23 is an adverse longitudinal channel, not a third photon.
        phase = np.exp(-2j * np.pi * np.arange(self.size) / self.size)[None, :, None, None]
        correlations = []
        for plane in ((1, 2), (1, 3), (2, 3)):
            mode = (np.sin(plaquettes[plane]) * phase).sum(axis=(1, 2, 3)) / self.size ** 1.5
            correlations.append([float(np.real(np.mean(np.roll(mode, -dt) * mode.conj())))
                                 for dt in range(self.size // 2 + 1)])
        loops = []
        for distance in (1, 2):
            for time_extent in (1, 2, 3):
                values = []
                for spatial in (1, 2, 3):
                    angle = sum(np.roll(np.roll(plaquettes[0, spatial], -r, axis=spatial),
                                        -t, axis=0)
                                for r in range(distance) for t in range(time_extent))
                    values.append(np.cos(angle).mean())
                loops.append(float(np.mean(values)))
        return {"plaquette": cosine, "helicity": float(helicity), "monopole_density": density,
                "cosines": cosines.tolist(), "sine_sums": sums.tolist(),
                "wilson_loops": loops, "correlations": correlations}


def mean_and_error(values: np.ndarray, block: int = 20) -> tuple[np.ndarray, np.ndarray]:
    blocks = len(values) // block
    if blocks < 2:
        raise ValueError("At least two full statistical blocks are required")
    reduced = values[:blocks * block].reshape((blocks, block) + values.shape[1:]).mean(axis=1)
    return reduced.mean(axis=0), reduced.std(axis=0, ddof=1) / np.sqrt(blocks)


def spectral_energy(triple: np.ndarray) -> float | None:
    """A positive-transfer-matrix correlator cannot have a negative spectral weight."""
    if np.any(triple <= 0):
        return None
    ratio = float((triple[0] + triple[2]) / (2 * triple[1]))
    return float(np.arccosh(ratio)) if ratio >= 1 else None


def gaussian_loop(size: int, distance: int, time_extent: int, beta: float) -> float:
    """Independent exact conserved-loop expectation in the finite Gaussian theory."""
    current = np.zeros((4,) + (size,) * 4)
    for t in range(time_extent):
        current[0, t, 0, 0, 0] += 1
        current[0, t, distance, 0, 0] -= 1
    for r in range(distance):
        current[1, time_extent, r, 0, 0] += 1
        current[1, 0, r, 0, 0] -= 1
    divergence = sum(current[mu] - np.roll(current[mu], 1, axis=mu) for mu in range(4))
    assert np.max(np.abs(divergence)) == 0
    momenta = np.indices((size,) * 4)
    laplacian = 4 * np.sin(np.pi * momenta / size) ** 2
    laplacian = laplacian.sum(axis=0)
    kernel = np.zeros_like(laplacian)
    np.divide(1, laplacian, out=kernel, where=laplacian != 0)
    transformed = np.fft.fftn(current, axes=(1, 2, 3, 4))
    variance = float((np.sum(np.abs(transformed) ** 2, axis=0) * kernel).sum()
                     / (beta * size ** 4))
    return float(np.exp(-variance / 2))


def conditional_control() -> dict:
    rng = np.random.default_rng(111)
    links = rng.uniform(-np.pi, np.pi, (4, 4, 4, 4, 4))
    beta = 0.7
    index = (1, 2, 0, 3)
    errors = []
    wrong_sign_errors = []
    for mu in range(4):
        value = staple(links, mu)[index]
        old = float(links[(mu,) + index])
        new = old + 0.43
        before = action(links, beta)
        modified = links.copy()
        modified[(mu,) + index] = new
        actual = action(modified, beta) - before
        predicted = -beta * np.real((np.exp(1j * new) - np.exp(1j * old)) * value)
        wrong = -beta * np.real((np.exp(-1j * new) - np.exp(-1j * old)) * value)
        errors.append(float(abs(actual - predicted)))
        wrong_sign_errors.append(float(abs(actual - wrong)))
    assert max(errors) < 1e-11
    assert max(wrong_sign_errors) > 0.05
    return {"max_action_change_error": max(errors),
            "wrong_staple_phase_max_error": max(wrong_sign_errors)}


def new_controls() -> dict:
    correlator = np.cosh(0.75 * (np.arange(5) - 4))
    energies = [spectral_energy(correlator[t - 1:t + 2]) for t in range(1, 4)]
    assert all(value is not None and abs(value - 0.75) < 1e-12 for value in energies)
    adverse = spectral_energy(np.array([-1., -0.25, -1.]))
    assert adverse is None
    rng = np.random.default_rng(919)
    links = rng.uniform(-np.pi, np.pi, (4, 4, 4, 4, 4))
    coordinates = np.indices((4,) * 4)
    delta = 2 * np.pi / 16
    modified = links.copy()
    modified[1] += delta * coordinates[0]
    modified[0] -= delta * 4 * coordinates[1] * (coordinates[0] == 3)
    actual = action(modified, 0.7) - action(links, 0.7)
    predicted = -0.7 * float((np.cos(curl(links, 0, 1) + delta)
                              - np.cos(curl(links, 0, 1))).sum())
    assert abs(actual - predicted) < 1e-11
    return {"negative_spectral_weights_rejected": adverse is None,
            "known_free_energy_max_error": max(abs(value - 0.75) for value in energies if value is not None),
            "global_flux_action_error": abs(actual - predicted)}


def sample(model: CompactGauge, burn: int, samples: int, thinning: int) -> dict:
    size, beta, seed, initial = model.size, model.beta, model.seed, model.initial
    for _ in range(burn):
        model.sweep()
    records = []
    for _ in range(samples):
        for _ in range(thinning):
            model.sweep()
        records.append(model.observables())
    report: dict = {"size": size, "beta": beta, "bare_charge": float(1 / np.sqrt(beta)),
                    "seed": seed, "burn_sweeps": burn, "samples": samples,
                    "thinning": thinning, "initial_links": initial, "block_samples": 20,
                    "integer_flux_updates": "one random plane and sign proposal per sweep"}
    for key in ("plaquette", "helicity", "monopole_density"):
        values = np.array([record[key] for record in records])
        mean, error = mean_and_error(values)
        first = float(values[:len(values) // 2].mean())
        last = float(values[len(values) // 2:].mean())
        report[key] = {"mean": float(mean), "blocked_standard_error": float(error),
                       "first_half_mean": first, "second_half_mean": last}
    cosines = np.array([record["cosines"] for record in records])
    sine_sums = np.array([record["sine_sums"] for record in records])
    count = len(records) // 20
    cosine_blocks = cosines[:count * 20].reshape(count, 20, 6).mean(axis=1)
    sine_blocks = sine_sums[:count * 20].reshape(count, 20, 6).mean(axis=1)
    square_blocks = (sine_sums[:count * 20] ** 2).reshape(count, 20, 6).mean(axis=1)
    def stiffness(cosine_mean, sine_mean, square_mean):
        return float(np.mean(beta * cosine_mean
                             - beta ** 2 * (square_mean - sine_mean ** 2) / size ** 4))
    helicity = stiffness(cosine_blocks.mean(axis=0), sine_blocks.mean(axis=0),
                        square_blocks.mean(axis=0))
    replicas = [stiffness((cosine_blocks.sum(axis=0) - cosine_blocks[b]) / (count - 1),
                         (sine_blocks.sum(axis=0) - sine_blocks[b]) / (count - 1),
                         (square_blocks.sum(axis=0) - square_blocks[b]) / (count - 1))
                for b in range(count)]
    report["connected_helicity"] = {
        "mean": helicity, "blocked_jackknife_error": float(np.std(replicas) * np.sqrt(count - 1)),
        "definition": "beta mean(cos) - beta squared/volume connected variance(sum sin)"}
    correlations = np.array([record["correlations"] for record in records])
    mean, error = mean_and_error(correlations)
    correlation_blocks = correlations[:count * 20].reshape((count, 20) + correlations.shape[1:]).mean(axis=1)
    transverse = mean[:2].mean(axis=0)
    report["magnetic_channels"] = {"transverse_correlations": mean[:2].tolist(),
        "longitudinal_correlation": mean[2].tolist(), "blocked_standard_errors": error.tolist(),
        "longitudinal_to_transverse_equal_time": float(mean[2, 0] / transverse[0])}
    predicted = float(2 * np.arcsinh(np.sin(np.pi / size)))
    effective = []
    for dt in range(1, len(transverse) - 1):
        energy = spectral_energy(transverse[dt - 1:dt + 2])
        leave_one = ((correlation_blocks.sum(axis=0) - correlation_blocks) / (count - 1)).mean(axis=1)[:, dt - 1:dt + 2]
        fits = [spectral_energy(triple) for triple in leave_one]
        valid_fits = [value for value in fits if value is not None]
        fit_error = (float(np.std(valid_fits) * np.sqrt(count - 1))
                     if energy is not None and len(valid_fits) == len(fits) else None)
        effective.append({"time_separation": dt, "energy": energy,
                          "blocked_jackknife_error": fit_error,
                          "interpretation": "correlator estimate; invalid/unstable positive spectral fit is null"})
    report["magnetic_channels"]["effective_energies"] = effective
    loop_means, loop_errors = mean_and_error(np.array([record["wilson_loops"] for record in records]))
    report["wilson_loops"] = [
        {"distance": distance, "time_extent": time_extent, "measured": float(loop_means[index]),
         "blocked_standard_error": float(loop_errors[index]),
         "bare_gaussian_prediction": gaussian_loop(size, distance, time_extent, beta),
         "independent_stiffness_gaussian_prediction":
             gaussian_loop(size, distance, time_extent, helicity) if helicity > 1 else None}
        for index, (distance, time_extent) in enumerate(itertools.product((1, 2), (1, 2, 3)))]
    report["canonical_charge_from_stiffness"] = (1 / np.sqrt(helicity) if helicity > 1 else None)
    report["bare_charge_rate_normalization_ratio"] = (beta / helicity) ** 2 if helicity > 1 else None
    report["free_massless_lattice_energy"] = predicted
    report["scope"] = ("Finite 4D pure-gauge Euclidean Wilson ensemble; blocked estimates. "
                       "No thermodynamic phase proof, no charged-matter vacuum feedback included, "
                       "no continuum-time Hamiltonian equivalence beyond its regulator limit, "
                       "no scattering calculation. Measured stiffness is not the bare beta.")
    return report


def run(burn: int = 300, samples: int = 400, thinning: int = 2) -> dict:
    control = conditional_control()
    controls = new_controls()
    rows = [sample(CompactGauge(size, beta, 9000 + size + int(beta * 100) + offset, initial),
                   burn, samples, thinning)
            for size in (4, 6, 8) for beta in (1 / 0.3 ** 2, 0.35)
            for initial, offset in (("ordered", 0), ("random", 1234))]
    return {"runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "conditional_heat_bath_control": control, "spectral_and_flux_controls": controls,
            "finite_volume_ensembles": rows,
            "matching": {"bare_charge_squared": "1/beta at bare isotropic Maxwell normalization",
                         "renormalized_charge_squared": "1/connected helicity, conditional long-wavelength pure-gauge map",
                         "selected_bare_charge": 0.3,
                         "same_charge_must_enter": "source, photon residue and Peierls vertices"},
            "status": "Constructive phase discriminant; statistical estimates, not a proved emergent electron"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burn", type=int, default=300)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--thinning", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.burn, args.samples, args.thinning)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
        print(json.dumps({"output": str(args.output), "runtime": result["runtime"],
                          "ensembles": len(result["finite_volume_ensembles"]),
                          "status": result["status"]}))
    else:
        print(payload)
