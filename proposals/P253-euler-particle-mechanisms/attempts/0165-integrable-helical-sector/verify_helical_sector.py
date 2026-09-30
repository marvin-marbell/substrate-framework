"""Replay scoped quadratures for the exact U_0 helical Euler calculations.

The analytic Euler and admissibility arguments are in derivation.md; these
quadratures check coefficients, not nonlinear stability or particle identity.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def simpson(function: Callable[[float], float], left: float, right: float, steps: int = 4096) -> float:
    assert steps > 0 and steps % 2 == 0
    spacing = (right - left) / steps
    total = function(left) + function(right)
    for index in range(1, steps):
        total += (4 if index % 2 else 2) * function(left + index * spacing)
    return total * spacing / 3


def bump(point: float) -> float:
    if abs(point) >= 1:
        return 0.0
    return math.exp(-1.0 / (1.0 - point * point))

def bump_first(point: float) -> float:
    if abs(point) >= 1:
        return 0.0
    return -2.0 * point * bump(point) / (1.0 - point * point) ** 2


def bump_second(point: float) -> float:
    if abs(point) >= 1:
        return 0.0
    one_minus_square = 1.0 - point * point
    first_log = -2.0 * point / one_minus_square**2
    second_log = -2.0 / one_minus_square**2 - 8.0 * point * point / one_minus_square**3
    return bump(point) * (first_log * first_log + second_log)


def receiver_overlap(time: float, vx: float = 0.0, vy: float = 0.0) -> float:
    """Inner product of the exact advected packet with a translated compact receiver."""

    def correlation(shift: float, differentiated: bool) -> float:
        if abs(shift) >= 2:
            return 0.0
        profile = bump_first if differentiated else bump
        return simpson(lambda x: profile(x) * profile(x - shift), -1, 1, steps=256)

    def vertical_integrand(z: float) -> float:
        dx = time * (math.sin(z) - vx)
        dy = time * (math.cos(z) - vy)
        return bump(z) ** 2 * (
            correlation(dx, False) * correlation(dy, True)
            + correlation(dx, True) * correlation(dy, False)
        )

    return simpson(vertical_integrand, -1, 1, steps=256)


def radial_bump_first(radius_squared: float) -> float:
    if radius_squared >= 1.0:
        return 0.0
    gap = 1.0 - radius_squared
    return -math.exp(-1.0 / gap) / (gap * gap)


def radial_pressure_virial() -> tuple[float, float]:
    """Check 2∫w_x² directly against the analytic transverse pressure moment."""
    g0 = math.exp(-1.0)
    analytic = 4.0 * math.pi * g0 * g0 * simpson(
        lambda s: s * radial_bump_first(s) ** 2, 0.0, 1.0
    )

    def radial_integrand(radius: float) -> float:
        slope = radial_bump_first(radius * radius)
        return 2.0 * radius * simpson(
            lambda angle: (2.0 * radius * math.sin(angle) * slope * g0) ** 2,
            0.0,
            2.0 * math.pi,
            steps=128,
        )

    direct = simpson(radial_integrand, 0.0, 1.0, steps=512)
    return analytic, direct


def main() -> None:
    integral_bump = simpson(bump, -1, 1)
    cos_bump = simpson(lambda z: math.cos(z) * bump(z), -1, 1)
    compensated_cos = simpson(
        lambda z: math.cos(z) * (bump_second(z) + bump(z)), -1, 1
    )
    column_derivative_direct = simpson(
        lambda z: 2.0 * math.exp(-2.0) * (bump_second(z) + bump(z)), -1, 1
    )
    column_derivative_integrated = 2.0 * math.exp(-2.0) * integral_bump
    weight = simpson(lambda z: bump(z) ** 2, -1, 1)
    mean_cos = simpson(lambda z: bump(z) ** 2 * math.cos(z), -1, 1) / weight
    broadening = 1.0 - mean_cos * mean_cos
    derivative_weight = -simpson(lambda z: bump(z) * bump_second(z), -1, 1)
    packet_norm_squared = 2.0 * weight * weight * derivative_weight
    stationary_cutoff = 2.0 * math.sqrt(2.0)
    centroid_cutoff = stationary_cutoff / (1.0 - mean_cos)
    receiver_initial = receiver_overlap(0.0)
    receiver_half = receiver_overlap(0.5)
    receiver_one = receiver_overlap(1.0)
    receiver_three = receiver_overlap(3.0)
    receiver_tracking = receiver_overlap(51.0, vy=mean_cos)
    angular = simpson(lambda z: math.sin(z) ** 4, 0, math.pi / 2)
    energies = [2 * math.pi * 0.4**2 * angular / (3 * scale) for scale in (1, 2, 4, 8, 16)]
    analytic_virial, direct_virial = radial_pressure_virial()

    print(f"compact bump integral: {integral_bump:.15f}")
    print(f"nonzero initial phase-column coefficient: {cos_bump:.15f}")
    print(f"prepared zero phase column: {compensated_cos:.3e}")
    print(f"direct Euler column derivative: {column_derivative_direct:.15f}")
    print(f"integrated-by-parts derivative: {column_derivative_integrated:.15f}")
    print(f"exact linear shear mean cosine: {mean_cos:.15f}")
    print(f"exact linear shear width coefficient: {broadening:.15f}")
    print(f"prepared packet initial L2 norm squared: {packet_norm_squared:.15f}")
    print(f"stationary receiver support cutoff: {stationary_cutoff:.15f}")
    print(f"centroid receiver support cutoff: {centroid_cutoff:.15f}")
    print(f"physical stationary receiver t=0,0.5,1,3: {[receiver_initial, receiver_half, receiver_one, receiver_three]}")
    print(f"physical centroid receiver t=51: {receiver_tracking:.15f}")
    print(f"fixed-Q leading energies, sigma 1,2,4,8,16: {energies}")
    print(f"quadratic Euler pressure transverse virial, analytic/direct: {analytic_virial:.15f}, {direct_virial:.15f}")

    assert abs(compensated_cos) < 1e-12
    assert abs(column_derivative_direct - column_derivative_integrated) < 1e-12
    assert column_derivative_direct > 0.12
    assert cos_bump > 0.4
    assert abs(broadening - 0.10930592442272413) < 1e-12
    assert packet_norm_squared > 0
    assert 3.0 > stationary_cutoff
    assert 51.0 > centroid_cutoff
    assert abs(receiver_initial - packet_norm_squared) < 1e-7
    assert receiver_half > 0
    assert receiver_one < 0
    assert receiver_three == 0
    assert receiver_tracking == 0
    assert abs(angular - 3 * math.pi / 16) < 1e-12
    for scale, energy in zip((1, 2, 4, 8, 16), energies, strict=True):
        assert abs(energy - math.pi**2 * 0.4**2 / (8 * scale)) < 1e-12
    assert analytic_virial > 0.11
    assert abs(analytic_virial - direct_virial) < 1e-10


if __name__ == "__main__":
    main()
