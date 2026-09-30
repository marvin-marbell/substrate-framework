"""Finite-core Plummer tail overlap; not a nonlinear Euler interaction/force test.

For theta=Q/(sigma**2+r**2)**.5, Fourier(theta)=4*pi*Q*sigma*K1(sigma*k)/k.
The full divergence-free dress adds lower-order cross terms at fixed core width.
"""

from __future__ import annotations

import math

from scipy.integrate import quad
from scipy.special import k1, spherical_jn


def tail_overlap(distance: float, width: float, axial: bool) -> float:
    """Return integral of two translated z-derivatives at Q1=Q2=1."""
    assert distance > 0 and width > 0
    angular_j2 = -2.0 if axial else 1.0

    def radial_integrand(wave_number: float) -> float:
        if wave_number == 0:
            return 8.0 / 3.0
        weight = width * wave_number * k1(width * wave_number)
        argument = distance * wave_number
        return 8.0 * weight * weight * (
            spherical_jn(0, argument) + angular_j2 * spherical_jn(2, argument)
        ) / 3.0

    value, error = quad(radial_integrand, 0, 40 / width, epsabs=1e-11, limit=1500)
    assert error < 1e-7
    return value


def main() -> None:
    for width in (1.0, 0.5):
        for distance in (8.0, 16.0, 32.0):
            transverse = tail_overlap(distance, width, axial=False)
            axial = tail_overlap(distance, width, axial=True)
            leading = 2 * math.pi / distance
            print(
                f"sigma={width:g} R={distance:g}: transverse={transverse:.12f} "
                f"axial={axial:.12f} transverse/(2pi/R)={transverse / leading:.12f}"
            )
            assert transverse > axial > 0
    assert tail_overlap(32.0, 0.5, axial=False) / (2 * math.pi / 32) > 0.997
    assert 32.0 * tail_overlap(32.0, 0.5, axial=True) < 0.024


if __name__ == "__main__":
    main()
