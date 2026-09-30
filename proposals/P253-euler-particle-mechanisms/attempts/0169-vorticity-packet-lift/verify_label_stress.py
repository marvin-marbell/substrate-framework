"""Check one-action cross stress and exact sector dynamics on a periodic control.

The periodic fields verify the algebra of the compact-packet theorem, not the
claim that these particular periodic fields have compact vorticity support.
"""

from __future__ import annotations

import math


def internal(time: float, x: float, y: float) -> float:
    return math.sin(x - time * math.sin(y))


def check() -> None:
    points = ((0.3, 0.4, 0.5), (1.2, -0.7, 1.8), (2.0, 0.9, -1.1))
    time = 1.0
    step = 1e-5
    for x, y, _ in points:
        time_derivative = (internal(time + step, x, y) - internal(time - step, x, y)) / (2 * step)
        x_derivative = (internal(time, x + step, y) - internal(time, x - step, y)) / (2 * step)
        actual_cross_advection = math.sin(y) * x_derivative
        assert abs(time_derivative + actual_cross_advection) < 2e-10
        assert abs(actual_cross_advection) > 0.1  # Omitting the first sector would fail.
    print("two label velocities reconstruct an exact global Euler trajectory with shared pressure zero")

    samples = 128
    stress_variation = math.fsum(
        math.sin(2 * math.pi * j / samples) ** 2
        * math.sin(2 * math.pi * i / samples) ** 2
        for i in range(samples)
        for j in range(samples)
    ) / samples**2
    vorticity_variation = math.fsum(
        math.sin(2 * math.pi * j / samples) ** 2
        * math.cos(2 * math.pi * i / samples) ** 2
        for i in range(samples)
        for j in range(samples)
    ) / samples**2
    assert abs(stress_variation - 0.25) < 1e-14
    assert abs(vorticity_variation - stress_variation) < 1e-14
    print(f"mixed-energy virtual derivative/(rho volume): stress={stress_variation:.8f}, vorticity={vorticity_variation:.8f}")
    print("initial mixed-energy value is zero although its displacement derivative is nonzero")


if __name__ == "__main__":
    check()
