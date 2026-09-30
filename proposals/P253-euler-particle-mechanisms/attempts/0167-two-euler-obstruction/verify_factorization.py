"""Replay symbolic Euler factorization obstruction and its exact survivors."""

from __future__ import annotations

import math

import sympy as sp


x, y, z, t, a, b = sp.symbols("x y z t a b", real=True)
coordinates = (x, y, z)


def curl(field: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(
        [
            sp.diff(field[2], y) - sp.diff(field[1], z),
            sp.diff(field[0], z) - sp.diff(field[2], x),
            sp.diff(field[1], x) - sp.diff(field[0], y),
        ]
    )


def convective(field: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(
        [sum(field[j] * sp.diff(field[i], coordinates[j]) for j in range(3)) for i in range(3)]
    )


def check() -> None:
    material_label = x - t * sp.sin(y)
    composed_acceleration = sp.Matrix(
        [
            2 * sp.sin(material_label) * sp.cos(y)
            - t * sp.sin(material_label) ** 2 * sp.sin(y),
            0,
            0,
        ]
    )
    observed_curl = sp.simplify(curl(composed_acceleration).subs({x: sp.pi / 2 + t, y: sp.pi / 2})[2])
    assert observed_curl == 2
    for time in (-2.0, 0.0, 1.0, 3.0):
        samples = 64
        # Physical velocity is v + eta_*w, not v+w at positive time.
        energy = math.fsum(
            (
                math.sin(2 * math.pi * j / samples)
                + time * math.cos(2 * math.pi * j / samples)
                * math.sin(2 * math.pi * i / samples - time * math.sin(2 * math.pi * j / samples))
            ) ** 2
            + math.sin(2 * math.pi * i / samples - time * math.sin(2 * math.pi * j / samples)) ** 2
            for i in range(samples)
            for j in range(samples)
        ) / (2 * samples * samples)
        assert abs(energy - (0.5 + time * time / 8)) < 1e-13
        print(f"factor-composition time={time:g}: curl witness=2, energy/(rho volume)={energy:.8f}")

    abc = sp.Matrix([sp.sin(z) + sp.cos(y), sp.sin(x) + sp.cos(z), sp.sin(y) + sp.cos(x)])
    assert curl(abc) == abc
    assert all(sp.simplify(value) == 0 for value in convective(abc) - sp.Matrix([sp.diff(abc.dot(abc) / 2, c) for c in coordinates]))
    first = sp.Matrix([sp.sin(z), sp.cos(z), 0])
    second = sp.Matrix([0, sp.sin(x), sp.cos(x)])
    combined = a * first + b * second
    assert curl(first) == first and curl(second) == second and curl(combined) == combined
    cross_pressure = -a * b * sp.sin(x) * sp.cos(z)
    assert all(
        sp.simplify(value) == 0
        for value in convective(combined) + sp.Matrix([sp.diff(cross_pressure, c) for c in coordinates])
    )
    assert sp.simplify(first.dot(second) - sp.sin(x) * sp.cos(z)) == 0
    print("locked ABC composition and distinct helical-mode cross pressure satisfy exact Euler")


if __name__ == "__main__":
    check()
