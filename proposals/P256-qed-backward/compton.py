#!/usr/bin/env python3
"""Spin-resolved tree Compton reference, not a microscopic explanation.

Metric (+---), hbar=c=1, q_e=-e, ubar*u=2m. All four-vectors are
contravariant in order (E, px, py, pz). Spin labels 0/1 are rotation-free
boosts of rest-frame +z/-z spin, not helicities. Photon transfer Q points
INTO each vertex; Q=k for absorption and Q=-kout for emission.

Gamma(eps,Q) = slash(eps) - kappa*[slash(eps),slash(Q)]/(4m)
             = eps_mu*(gamma^mu + i*kappa*sigma^{mu nu}*Q_nu/(2m)).
Thus F1(0)=1, F2(0)=kappa and g=2(1+kappa). Outgoing polarization is
conjugated internally. kappa!=0 is a conditional local Pauli-EFT control,
not a loop calculation or a claim that gauge invariance fixes g=2.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
METRIC = np.array([1.0, -1.0, -1.0, -1.0])
IDENTITY = np.eye(4, dtype=complex)
PAULI = np.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=complex,
)
GAMMA = np.empty((4, 4, 4), dtype=complex)
GAMMA[0] = np.diag([1, 1, -1, -1])
_ZERO2 = np.zeros((2, 2), dtype=complex)
for _i in range(3):
    GAMMA[_i + 1] = np.block([[_ZERO2, PAULI[_i]], [-PAULI[_i], _ZERO2]])


def minkowski(a: ArrayLike, b: ArrayLike) -> float:
    """Real four-momentum inner product; no polarization conjugation."""
    return float(np.dot(np.asarray(a, dtype=float) * METRIC, np.asarray(b, dtype=float)))


def slash(p: ArrayLike) -> ComplexArray:
    """gamma^mu*p_mu, also accepting complex polarizations."""
    return np.einsum("m,mij->ij", np.asarray(p, dtype=complex) * METRIC, GAMMA)


def spinor(p: ArrayLike, spin: int, m: float = 1.0) -> ComplexArray:
    """Positive-energy on-shell spinor, normalized by ubar*u=2m."""
    if spin not in (0, 1):
        raise ValueError("spin must be 0 (+z) or 1 (-z)")
    momentum = np.asarray(p, dtype=float)
    chi = np.eye(2, dtype=complex)[:, spin]
    sigma_p = np.einsum("i,ijk->jk", momentum[1:], PAULI)
    return np.sqrt(momentum[0] + m) * np.concatenate(
        (chi, sigma_p @ chi / (momentum[0] + m))
    )


def kinematics(omega: float, theta: float, m: float = 1.0) -> tuple[RealArray, ...]:
    """Lab target at rest; incoming photon +z, outgoing in the xz plane."""
    if m <= 0.0 or omega <= 0.0 or not 0.0 <= theta <= math.pi:
        raise ValueError("require m>0, omega>0 and theta in [0,pi]")
    omega_out = omega / (1.0 + omega / m * (1.0 - math.cos(theta)))
    p = np.array([m, 0.0, 0.0, 0.0])
    k = np.array([omega, 0.0, 0.0, omega])
    kout = omega_out * np.array([1.0, math.sin(theta), 0.0, math.cos(theta)])
    pout = p + k - kout
    return p, k, pout, kout


def polarizations(k: ArrayLike) -> tuple[RealArray, RealArray]:
    """Two real Coulomb-gauge unit vectors; least-aligned-axis convention."""
    momentum = np.asarray(k, dtype=float)
    direction = momentum[1:] / np.linalg.norm(momentum[1:])
    axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = axis - np.dot(axis, direction) * direction
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return np.concatenate(([0.0], first)), np.concatenate(([0.0], second))


def vertex(eps: ArrayLike, transfer: ArrayLike, m: float = 1.0,
           kappa: float = 0.0) -> ComplexArray:
    """Contracted Dirac-Pauli vertex, with photon momentum into the vertex."""
    polarization_slash = slash(eps)
    if kappa == 0.0:
        return polarization_slash
    transfer_slash = slash(transfer)
    return polarization_slash - kappa / (4.0 * m) * (
        polarization_slash @ transfer_slash - transfer_slash @ polarization_slash
    )


def _operator(p: ArrayLike, k: ArrayLike, kout: ArrayLike, eps_in: ArrayLike,
              eps_out: ArrayLike, m: float, kappa: float, channel: str) -> ComplexArray:
    if channel not in ("both", "s", "u"):
        raise ValueError("channel must be 'both', 's' or 'u'")
    p_array = np.asarray(p, dtype=float)
    k_array = np.asarray(k, dtype=float)
    kout_array = np.asarray(kout, dtype=float)
    incoming = vertex(eps_in, k_array, m, kappa)
    outgoing = vertex(np.asarray(eps_out, dtype=complex).conj(), -kout_array, m, kappa)
    operator = np.zeros((4, 4), dtype=complex)
    if channel in ("both", "s"):
        propagator_s = (slash(p_array + k_array) + m * IDENTITY) / (
            2.0 * minkowski(p_array, k_array)
        )
        operator += outgoing @ propagator_s @ incoming
    if channel in ("both", "u"):
        propagator_u = (slash(p_array - kout_array) + m * IDENTITY) / (
            -2.0 * minkowski(p_array, kout_array)
        )
        operator += incoming @ propagator_u @ outgoing
    return operator


def amplitude(p: ArrayLike, k: ArrayLike, pout: ArrayLike, kout: ArrayLike,
              eps_in: ArrayLike, eps_out: ArrayLike, spin_in: int, spin_out: int,
              m: float = 1.0, e: float = 1.0, kappa: float = 0.0,
              channel: str = "both") -> complex:
    """Dimensionless M in S_fi=i(2pi)^4 delta^4(Pf-Pi)*M.

    On-shell momenta and momentum conservation are the caller's contract.
    eps_in/out may be momentum replacements or complex helicity vectors.
    """
    initial = spinor(p, spin_in, m)
    final_bar = spinor(pout, spin_out, m).conj() @ GAMMA[0]
    operator = _operator(p, k, kout, eps_in, eps_out, m, kappa, channel)
    return complex(-e**2 * (final_bar @ operator @ initial))


def unpolarized_squared(omega: float, theta: float, m: float = 1.0,
                        e: float = 1.0, kappa: float = 0.0,
                        channel: str = "both") -> float:
    """Average initial 2 spins*2 polarizations, sum the 4 final states."""
    p, k, pout, kout = kinematics(omega, theta, m)
    total = 0.0
    for eps_in in polarizations(k):
        for eps_out in polarizations(kout):
            for spin_in in (0, 1):
                for spin_out in (0, 1):
                    total += abs(amplitude(
                        p, k, pout, kout, eps_in, eps_out, spin_in, spin_out,
                        m, e, kappa, channel,
                    ))**2
    return total / 4.0


def spin_trace_squared(omega: float, theta: float, m: float = 1.0,
                       e: float = 1.0, kappa: float = 0.0) -> float:
    """Independent spin-completeness trace with physical polarization sum."""
    p, k, pout, kout = kinematics(omega, theta, m)
    initial_projector = slash(p) + m * IDENTITY
    final_projector = slash(pout) + m * IDENTITY
    total = 0.0j
    for eps_in in polarizations(k):
        for eps_out in polarizations(kout):
            operator = _operator(p, k, kout, eps_in, eps_out, m, kappa, "both")
            adjoint = GAMMA[0] @ operator.conj().T @ GAMMA[0]
            total += np.trace(final_projector @ operator @ initial_projector @ adjoint)
    _require(abs(total.imag) < 1e-9 * max(1.0, abs(total.real)), "complex spin trace")
    return float(e**4 * total.real / 4.0)


def covariant_trace_squared(omega: float, theta: float, m: float = 1.0,
                            e: float = 1.0) -> float:
    """Minimal-QED trace using -eta sums; valid only for the coherent pair."""
    p, k, pout, kout = kinematics(omega, theta, m)
    initial_projector = slash(p) + m * IDENTITY
    final_projector = slash(pout) + m * IDENTITY
    basis = np.eye(4)
    total = 0.0j
    for mu in range(4):
        for nu in range(4):
            operator = _operator(p, k, kout, basis[mu], basis[nu], m, 0.0, "both")
            adjoint = GAMMA[0] @ operator.conj().T @ GAMMA[0]
            total += METRIC[mu] * METRIC[nu] * np.trace(
                final_projector @ operator @ initial_projector @ adjoint
            )
    _require(abs(total.imag) < 1e-9 * max(1.0, abs(total.real)), "complex covariant trace")
    return float(e**4 * total.real / 4.0)


def klein_nishina_squared(omega: float, theta: float, m: float = 1.0,
                         e: float = 1.0) -> float:
    """Analytic comparator only; never used to compute amplitude/spin sum."""
    ratio = 1.0 / (1.0 + omega / m * (1.0 - math.cos(theta)))
    return 2.0 * e**4 * (ratio + 1.0 / ratio - math.sin(theta)**2)


def differential_cross_section(omega: float, theta: float, m: float = 1.0,
                               e: float = 1.0, kappa: float = 0.0) -> float:
    """Lab d sigma/d Omega in inverse mass squared, from the evaluated M."""
    _, _, _, kout = kinematics(omega, theta, m)
    return (kout[0] / omega)**2 * unpolarized_squared(
        omega, theta, m, e, kappa
    ) / (64.0 * math.pi**2 * m**2)


def thomson_cross_section(theta: float, m: float = 1.0, e: float = 1.0) -> float:
    """r_e^2*(1+cos(theta)^2)/2, r_e=e^2/(4pi*m)."""
    return e**4 * (1.0 + math.cos(theta)**2) / (32.0 * math.pi**2 * m**2)


def ward_residuals(omega: float, theta: float, m: float = 1.0,
                   e: float = 1.0, kappa: float = 0.0,
                   channel: str = "both") -> dict[str, float]:
    """Max over opposite physical polarizations and all electron spins.

    Raw replacements have energy units. Dividing by e^2*photon energy
    gives dimensionless residuals; this prevents a soft photon hiding a
    missing-channel Ward failure.
    """
    p, k, pout, kout = kinematics(omega, theta, m)
    incoming_max = 0.0
    outgoing_max = 0.0
    for spin_in in (0, 1):
        for spin_out in (0, 1):
            for eps_out in polarizations(kout):
                incoming_max = max(incoming_max, abs(amplitude(
                    p, k, pout, kout, k, eps_out, spin_in, spin_out,
                    m, e, kappa, channel,
                )))
            for eps_in in polarizations(k):
                outgoing_max = max(outgoing_max, abs(amplitude(
                    p, k, pout, kout, eps_in, kout, spin_in, spin_out,
                    m, e, kappa, channel,
                )))
    return {
        "incoming_absolute": incoming_max,
        "outgoing_absolute": outgoing_max,
        "incoming_over_e2_omega": incoming_max / (e**2 * omega),
        "outgoing_over_e2_omega_out": outgoing_max / (e**2 * kout[0]),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(actual: float | complex, expected: float | complex, message: str,
           rtol: float = 2e-10, atol: float = 2e-11) -> None:
    _require(abs(actual - expected) <= atol + rtol * abs(expected),
             f"{message}: {actual!r} != {expected!r}")


def _check_algebra() -> None:
    for mu in range(4):
        for nu in range(4):
            expected = 2.0 * METRIC[mu] * IDENTITY if mu == nu else 0.0 * IDENTITY
            _require(np.allclose(GAMMA[mu] @ GAMMA[nu] + GAMMA[nu] @ GAMMA[mu],
                                 expected, atol=1e-14), "Clifford algebra")
    p, _, pout, _ = kinematics(0.4, math.pi / 3.0)
    for momentum in (p, pout):
        projector = np.zeros((4, 4), dtype=complex)
        for spin in (0, 1):
            u = spinor(momentum, spin)
            bar = u.conj() @ GAMMA[0]
            _close(complex(bar @ u), 2.0, "spinor normalization")
            _require(float(np.linalg.norm((slash(momentum) - IDENTITY) @ u)) < 1e-12,
                     "on-shell Dirac equation")
            projector += np.outer(u, bar)
        _require(np.allclose(projector, slash(momentum) + IDENTITY, atol=1e-12),
                 "spin completeness")


def _resolved_channels(omega: float, theta: float, m: float, e: float,
                       kappa: float) -> list[dict[str, Any]]:
    p, k, pout, kout = kinematics(omega, theta, m)
    rows = []
    for pol_in, eps_in in enumerate(polarizations(k)):
        for pol_out, eps_out in enumerate(polarizations(kout)):
            for spin_in in (0, 1):
                for spin_out in (0, 1):
                    value = amplitude(p, k, pout, kout, eps_in, eps_out,
                                      spin_in, spin_out, m, e, kappa)
                    rows.append({
                        "pol_in": pol_in, "pol_out": pol_out,
                        "spin_in": spin_in, "spin_out": spin_out,
                        "amplitude_re": value.real, "amplitude_im": value.imag,
                        "squared": abs(value)**2,
                    })
    return rows


def run_reference(energy_ratios: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0),
                  angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0),
                  m: float = 1.0, e: float = 1.0) -> dict[str, Any]:
    """Exercise the amplitude, independent traces, Ward controls and limits."""
    if m <= 0.0 or e <= 0.0:
        raise ValueError("reference run requires m>0 and e>0")
    _check_algebra()
    reference_rows = []
    for ratio in energy_ratios:
        omega = m * ratio
        for angle in angles_deg:
            theta = math.radians(angle)
            p, k, pout, kout = kinematics(omega, theta, m)
            _close(minkowski(pout, pout), m**2, "recoil mass shell")
            for momentum in (k, kout):
                eps1, eps2 = polarizations(momentum)
                _close(minkowski(momentum, momentum), 0.0, "photon mass shell")
                _close(minkowski(momentum, eps1), 0.0, "first transverse polarization")
                _close(minkowski(momentum, eps2), 0.0, "second transverse polarization")
                _close(minkowski(eps1, eps2), 0.0, "polarization orthogonality")
                _close(minkowski(eps1, eps1), -1.0, "polarization norm")
            numeric = unpolarized_squared(omega, theta, m, e)
            analytic = klein_nishina_squared(omega, theta, m, e)
            trace = spin_trace_squared(omega, theta, m, e)
            covariant = covariant_trace_squared(omega, theta, m, e)
            _close(numeric, analytic, "spin sum versus Klein-Nishina")
            _close(trace, numeric, "trace versus spin sum")
            _close(covariant, numeric, "covariant versus physical polarization sum")
            ward = ward_residuals(omega, theta, m, e)
            _require(ward["incoming_over_e2_omega"] < 1e-9, "incoming Ward identity")
            _require(ward["outgoing_over_e2_omega_out"] < 1e-9, "outgoing Ward identity")
            reference_rows.append({
                "omega_over_m": ratio, "theta_deg": angle,
                "omega_out_over_omega": float(kout[0] / omega),
                "spin_sum": numeric, "physical_spin_trace": trace,
                "covariant_spin_trace": covariant, "klein_nishina_squared": analytic,
                "relative_kn_error": abs(numeric - analytic) / analytic,
                "d_sigma_d_solid_angle": differential_cross_section(omega, theta, m, e),
                "ward": ward,
            })

    theta = math.pi / 3.0
    omega = 0.4 * m
    controls = {}
    for channel in ("s", "u"):
        control = ward_residuals(omega, theta, m, e, channel=channel)
        _require(control["incoming_over_e2_omega"] > 1e-3,
                 f"{channel}-only incoming Ward adverse control was insensitive")
        _require(control["outgoing_over_e2_omega_out"] > 1e-3,
                 f"{channel}-only outgoing Ward adverse control was insensitive")
        controls[channel + "_only"] = control

    # Complex polarization checks detect conjugation/sign errors invisible
    # to a purely real unpolarized comparator.
    p, k, pout, kout = kinematics(omega, theta, m)
    pin, pout_pol = polarizations(k), polarizations(kout)
    circular_in = (pin[0] + 1j * pin[1]) / math.sqrt(2.0)
    circular_out = (pout_pol[0] + 1j * pout_pol[1]) / math.sqrt(2.0)
    for spin_in in (0, 1):
        for spin_out in (0, 1):
            base = amplitude(p, k, pout, kout, circular_in, circular_out,
                             spin_in, spin_out, m, e)
            gauged = amplitude(p, k, pout, kout, circular_in + (0.3 + 0.2j) * k / m,
                               circular_out + (-0.2 + 0.1j) * kout / m,
                               spin_in, spin_out, m, e)
            _close(gauged, base, "complex polarization gauge shifts")
            expected = 0.0j
            for i in range(2):
                for j in range(2):
                    expected += (1j)**i * (-1j)**j * amplitude(
                        p, k, pout, kout, pin[i], pout_pol[j], spin_in, spin_out, m, e
                    ) / 2.0
            _close(base, expected, "outgoing polarization conjugation")

    # m and e are inputs, but the reference must carry their dimensions.
    baseline = unpolarized_squared(0.4, theta)
    _close(unpolarized_squared(0.8, theta, m=2.0, e=0.7),
           0.7**4 * baseline, "charge power and dimensionless amplitude")
    _close(differential_cross_section(0.8, theta, m=2.0, e=0.7),
           0.7**4 / 4.0 * differential_cross_section(0.4, theta),
           "cross-section inverse mass squared")

    low_energy_rows = []
    for kappa in (0.0, 0.3):
        previous_error = None
        for ratio in (1e-1, 1e-2, 1e-3, 1e-4):
            soft = ratio * m
            p, k, pout, kout = kinematics(soft, theta, m)
            amplitude_error = 0.0
            for eps_in in polarizations(k):
                for eps_out in polarizations(kout):
                    for spin_in in (0, 1):
                        for spin_out in (0, 1):
                            expected = -2.0 * e**2 * float(np.dot(eps_out[1:], eps_in[1:]))
                            if spin_in != spin_out:
                                expected = 0.0
                            value = amplitude(p, k, pout, kout, eps_in, eps_out,
                                              spin_in, spin_out, m, e, kappa)
                            amplitude_error = max(amplitude_error, abs(value - expected) / e**2)
            _require(amplitude_error < 50.0 * ratio, "spin-resolved Thomson limit")
            if previous_error is not None:
                _require(amplitude_error < 0.25 * previous_error, "soft convergence")
            previous_error = amplitude_error
            cross_section = differential_cross_section(soft, theta, m, e, kappa)
            thomson = thomson_cross_section(theta, m, e)
            ward = ward_residuals(soft, theta, m, e, kappa)
            _require(ward["incoming_over_e2_omega"] < 1e-8, "Pauli incoming Ward identity")
            _require(ward["outgoing_over_e2_omega_out"] < 1e-8, "Pauli outgoing Ward identity")
            _close(spin_trace_squared(soft, theta, m, e, kappa),
                   unpolarized_squared(soft, theta, m, e, kappa), "Pauli spin trace",
                   rtol=2e-8)
            low_energy_rows.append({
                "kappa": kappa, "g": 2.0 * (1.0 + kappa), "omega_over_m": ratio,
                "max_spin_resolved_thomson_error_over_e2": amplitude_error,
                "cross_section_over_thomson": cross_section / thomson,
                "ward": ward,
            })
        _require(abs(low_energy_rows[-1]["cross_section_over_thomson"] - 1.0) < 1e-3,
                 "Thomson cross-section normalization")

    minimal = _resolved_channels(omega, theta, m, e, 0.0)
    deformed = _resolved_channels(omega, theta, m, e, 0.3)
    max_difference = max(abs(complex(a["amplitude_re"], a["amplitude_im"])
                             - complex(b["amplitude_re"], b["amplitude_im"])) / e**2
                         for a, b in zip(minimal, deformed))
    _require(max_difference > 1e-3, "finite-energy spin channels did not discriminate Pauli term")
    pauli_ward = ward_residuals(omega, theta, m, e, kappa=0.3)
    _require(pauli_ward["incoming_over_e2_omega"] < 1e-9, "finite-energy Pauli incoming Ward")
    _require(pauli_ward["outgoing_over_e2_omega_out"] < 1e-9, "finite-energy Pauli outgoing Ward")
    return {
        "scope": "conditional tree QED reference, not an explanatory construction",
        "units": "hbar=c=1; cross sections in inverse mass squared",
        "m": m, "e": e, "q_e": -e, "alpha_for_selected_e": e**2 / (4.0 * math.pi),
        "initial_state_average": 4, "perturbative_order": "M: e^2; d_sigma: e^4 (alpha^2)",
        "reference": reference_rows,
        "missing_channel_controls_at_omega_over_m_0_4_theta_deg_60": controls,
        "low_energy_at_theta_deg_60": low_energy_rows,
        "resolved_at_omega_over_m_0_4_theta_deg_60": minimal,
        "pauli_control": {
            "kappa": 0.3, "g": 2.6, "ward": pauli_ward,
            "max_resolved_amplitude_difference_over_e2": max_difference,
            "minimal_unpolarized_squared": unpolarized_squared(omega, theta, m, e),
            "pauli_unpolarized_squared": unpolarized_squared(omega, theta, m, e, 0.3),
            "resolved": deformed,
        },
        "checks_passed": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energies", type=float, nargs="+", default=[0.01, 0.1, 0.5, 1.0],
                        help="incoming photon energies omega/m")
    parser.add_argument("--angles-deg", type=float, nargs="+", default=[0, 45, 90, 135, 180])
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--e", type=float, default=1.0,
                        help="positive charge magnitude; default e=1 is not physical alpha")
    args = parser.parse_args(argv)
    result = run_reference(tuple(args.energies), tuple(args.angles_deg), args.mass, args.e)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
