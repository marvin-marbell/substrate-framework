"""Check exact two-sheet Euler pressure and the full 3-D escape coefficient.

Analytic proof, material-slab boundary, and limitations are in derivation.md.
"""

from __future__ import annotations

import math


ALPHA = 0.7
BETA = 1.2


def velocity(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (
        ALPHA * math.sin(x) * math.cos(y),
        -ALPHA * math.cos(x) * math.sin(y) + BETA * math.sin(y) * math.cos(z),
        -BETA * math.cos(y) * math.sin(z),
    )


def pressure(x: float, y: float, z: float, mixed: bool = True) -> float:
    self_pressure = -ALPHA**2 * (math.sin(x) ** 2 + math.sin(y) ** 2) / 2
    self_pressure -= BETA**2 * (math.sin(y) ** 2 + math.sin(z) ** 2) / 2
    if not mixed:
        return self_pressure
    return self_pressure - ALPHA * BETA * math.cos(x) * math.cos(z) * (
        0.5 + math.cos(2 * y) / 6
    )


def escape(x: float, y: float, z: float) -> tuple[float, float, float]:
    coefficient = 2 * ALPHA * BETA / 3
    return (
        -coefficient * math.sin(x) * math.cos(2 * y) * math.cos(z),
        coefficient * math.cos(x) * math.sin(2 * y) * math.cos(z),
        -coefficient * math.cos(x) * math.cos(2 * y) * math.sin(z),
    )


def shifted(point: tuple[float, float, float], axis: int, offset: float) -> tuple[float, float, float]:
    return (
        point[0] + offset * (axis == 0),
        point[1] + offset * (axis == 1),
        point[2] + offset * (axis == 2),
    )


def physical_acceleration(x: float, y: float, z: float, mixed: bool = True) -> tuple[float, float, float]:
    point = (x, y, z)
    flow = velocity(*point)
    step = 1e-5
    forward = [pressure(*shifted(point, i, step), mixed) for i in range(3)]
    backward = [pressure(*shifted(point, i, -step), mixed) for i in range(3)]

    def component(i: int) -> float:
        return -sum(
            flow[j]
            * (
                velocity(*shifted(point, j, step))[i]
                - velocity(*shifted(point, j, -step))[i]
            ) / (2 * step)
            for j in range(3)
        ) - (forward[i] - backward[i]) / (2 * step)

    return component(0), component(1), component(2)


def bump(z: float) -> float:
    return math.exp(-1 / (1 - z * z)) if abs(z) < 1 else 0.0


def check() -> None:
    for point in ((0.3, 0.4, 0.5), (1.1, -0.2, 1.7), (-0.7, 0.9, -1.2)):
        observed = physical_acceleration(*point)
        target = escape(*point)
        assert max(abs(a - b) for a, b in zip(observed, target)) < 2e-9
        without_common_pressure = physical_acceleration(*point, mixed=False)
        assert max(abs(a - b) for a, b in zip(without_common_pressure, target)) > 0.1
    samples = 16
    average_squared = math.fsum(
        sum(component**2 for component in escape(2 * math.pi * i / samples, 2 * math.pi * j / samples, 2 * math.pi * k / samples))
        for i in range(samples)
        for j in range(samples)
        for k in range(samples)
    ) / samples**3
    assert abs(average_squared - (ALPHA * BETA) ** 2 / 6) < 1e-13
    print(f"full 3-D mixed escape norm/sqrt(volume)={math.sqrt(average_squared):.9f}")
    print("three direct full-pressure Euler accelerations agree; dropping mixed pressure fails")

    steps = 4000
    spacing = 2 / steps
    i_plus = spacing * math.fsum(
        math.exp(2 * (-1 + (n + 0.5) * spacing)) * bump(-1 + (n + 0.5) * spacing) ** 2
        for n in range(steps)
    )
    z = 1.3
    delta = 1e-5
    # Outside the slab the nonzero Fourier pressure mode is exp(-2z)*I_plus.
    exterior = lambda level: math.exp(-2 * level) * i_plus / 2
    numerical_vertical_accel = -(exterior(z + delta) - exterior(z - delta)) / (2 * delta)
    expected = math.exp(-2 * z) * i_plus
    assert i_plus > 0 and abs(numerical_vertical_accel - expected) < 5e-11
    print(f"exterior vertical acceleration at (x,y,z)=(0,0,{z}): {expected:.12f}")


if __name__ == "__main__":
    check()
