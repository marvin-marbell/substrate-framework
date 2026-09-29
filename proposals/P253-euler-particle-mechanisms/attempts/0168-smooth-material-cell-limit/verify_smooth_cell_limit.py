"""Replay two physical 3-D Euler checks of the smooth material-cell limit.

The shear is an exact global Euler trajectory; the remote compact swirl has
smooth local Euler existence and its analytic exterior pressure is evaluated
at the initial slice. Proof and scope are in derivation.md.
"""

from __future__ import annotations

import math


def swirl_pressure(x: float, y: float, z: float, distance: float) -> float:
    """Exterior physical pressure for rho=J=1 and axis/center distance e_z."""
    dz = z - distance
    radius_squared = x * x + y * y + dz * dz
    return -(3 * dz * dz - radius_squared) / (3 * radius_squared ** 2.5)


def check_shear() -> None:
    """The global solution u=(sin y,0,0) preserves its material-cell variance."""
    for width in (0.4, 0.2, 0.1, 0.05):
        # A cell centered at y=0 has zero center speed at every time.
        analytic = 0.5 - math.sin(width) / (2 * width)
        steps = 10000
        sampled = math.fsum(
            math.sin(width * (index + 0.5) / steps - width / 2) ** 2
            for index in range(steps)
        ) / steps
        assert abs(sampled - analytic) < 4e-10
        assert analytic > 0
        assert abs(analytic / (width * width) - 1 / 12) < width * width / 240
        position = 0.3
        internal_scaled = math.sin(width * position) / width
        assert abs(internal_scaled - position) <= width * width * position**3 / 6
        print(f"exact shear material cell width={width:.2f}: variance={analytic:.10f}, variance/h^2={analytic / width**2:.8f}")


def check_high_gradient_survivor() -> None:
    """Actual invariant fast shear retains energy, but has zero stress divergence."""
    amplitude = 1.0
    for periods in (8, 16, 32, 64):
        width = 2 * math.pi / periods
        samples = 1024
        cell_variances = []
        for cell in range(periods):
            values = [
                math.sin(width * (cell + (index + 0.5) / samples))
                + amplitude * math.sin(2 * math.pi * (index + 0.5) / samples)
                for index in range(samples)
            ]
            mean = math.fsum(values) / samples
            cell_variances.append(math.fsum((value - mean) ** 2 for value in values) / samples)
        energy_density = math.fsum(cell_variances) / (2 * periods)
        bound = amplitude * width / math.sqrt(2) + width**2 / 2
        assert abs(energy_density - amplitude**2 / 4) < bound
        assert energy_density > 0.2
        print(f"exact high-gradient Euler shear N={periods}: internal energy/(rho volume)={energy_density:.9f}")


def check_transverse_perturbation() -> None:
    """Exact 3-D triangular Euler perturbations lose uniform H1 control."""
    time = 1.0
    samples = 4096
    for periods in (8, 16, 32, 64):
        mean_shear_gradient_squared = math.fsum(
            (
                math.cos(2 * math.pi * (index + 0.5) / samples)
                + periods * math.cos(2 * math.pi * periods * (index + 0.5) / samples)
            ) ** 2
            for index in range(samples)
        ) / samples
        transverse_gradient = time * math.sqrt(mean_shear_gradient_squared / 2) / periods
        analytic = time * math.sqrt(1 + periods**2) / (2 * periods)
        initial_h1 = 1 / periods
        assert abs(transverse_gradient - analytic) < 1e-12
        assert transverse_gradient > 0.5 and initial_h1 < 0.13
        print(
            f"exact transverse Euler N={periods}: initial H1={initial_h1:.6f}, "
            f"time-1 transverse-gradient L2={transverse_gradient:.6f}"
        )


def check_remote_pressure() -> None:
    """An actual admissible remote swirl changes resting-cell acceleration."""
    distance = 4.0
    step = 0.001
    origin = swirl_pressure(0, 0, 0, distance)
    gradient_z = (
        swirl_pressure(0, 0, step, distance)
        - swirl_pressure(0, 0, -step, distance)
    ) / (2 * step)
    diagonal = tuple(
        (
            swirl_pressure(step * (axis == 0), step * (axis == 1), step * (axis == 2), distance)
            - 2 * origin
            + swirl_pressure(-step * (axis == 0), -step * (axis == 1), -step * (axis == 2), distance)
        ) / step**2
        for axis in range(3)
    )
    target = (4 / distance**5, 4 / distance**5, -8 / distance**5)
    assert abs(gradient_z + 2 / distance**4) < 1e-8
    assert all(abs(observed - expected) < 1e-8 for observed, expected in zip(diagonal, target))
    assert abs(sum(diagonal)) < 2e-8  # Pressure is harmonic near the resting cell.
    assert diagonal[0] > 0 and diagonal[2] < 0
    print(f"remote-swirl resting-cell acceleration={-gradient_z:.10f}, pressure Hessian={diagonal}")
    print("zero-velocity Euler control: center and internal accelerations both zero")


if __name__ == "__main__":
    check_shear()
    check_high_gradient_survivor()
    check_transverse_perturbation()
    check_remote_pressure()
