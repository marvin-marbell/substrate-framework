"""Enclose shortwave monodromy on two exact Euler-background streamlines.

The analytic midpoint-product error bounds are stated in floquet.md. This
script verifies parameter-specific constants and interval products; it does
not evolve an Euler wavepacket or prove all-mode stability.
"""
from mpmath import iv

iv.dps = 50


def certify(e_tenths, steps, m_lower, L, K1, K2, local_upper, exp_upper, global_upper):
    """Enclose the one-ray WKB ODE trace with rational outward-rounded bounds."""
    e = iv.mpf(e_tenths) / 10
    a = 1 / iv.sqrt(2)
    a_upper = iv.mpf(708) / 1000
    period_upper = iv.mpf(6284) / 1000
    assert a.b < a_upper
    assert (a - e).a > m_lower
    assert (2 * iv.pi).b < period_upper

    # Coordinates n=(delta_x+delta_y)/2, b=delta_z:
    # C(x)=[[-e*sin(x),a],[-2*e*cos(x),0]]/(a+e*cos(x)).
    assert ((a_upper + e) / m_lower).b < L
    first_row = e / m_lower + e**2 / m_lower**2 + a_upper * e / m_lower**2
    second_row = 2 * (e / m_lower + e**2 / m_lower**2)
    assert first_row.b < K1 and second_row.b < K1
    second_derivative_first = (
        e / m_lower
        + 3 * e**2 / m_lower**2
        + 2 * e**3 / m_lower**3
        + a_upper * e / m_lower**2
        + 2 * a_upper * e**2 / m_lower**3
    )
    second_derivative_second = 2 * (
        e / m_lower + 3 * e**2 / m_lower**2 + 2 * e**3 / m_lower**3
    )
    assert second_derivative_first.b < K2 and second_derivative_second.b < K2
    local_coefficient = (
        K2 / 24 + L * K1 / 2
        + iv.exp(L * period_upper / steps) * L**3 / 3 + L**3
    )
    assert local_coefficient.b < local_upper
    exp_bound = iv.exp((L + iv.mpf('0.01')) * period_upper)
    assert exp_bound.b < exp_upper
    global_error = local_upper * period_upper * (period_upper / steps) ** 2 * exp_bound
    assert global_error.b < global_upper

    step = -2 * iv.pi / steps
    Y00 = Y11 = iv.mpf(1)
    Y01 = Y10 = iv.mpf(0)
    for j in range(steps):
        x = (j + iv.mpf('0.5')) * step
        h = a + e * iv.cos(x)
        d = step * (-e * iv.sin(x) / h) / 2
        u = step * (a / h) / 2
        v = step * (-2 * e * iv.cos(x) / h) / 2
        D = 1 - d - u * v
        P00 = (1 + d + u * v) / D
        P01 = 2 * u / D
        P10 = 2 * v / D
        P11 = (1 - d + u * v) / D
        Y00, Y01, Y10, Y11 = (
            P00 * Y00 + P01 * Y10,
            P00 * Y01 + P01 * Y11,
            P10 * Y00 + P11 * Y10,
            P10 * Y01 + P11 * Y11,
        )

    trace = Y00 + Y11
    trace_lower = trace - 2 * global_error
    trace_upper = trace + 2 * global_error
    print(f'e={e_tenths}/10 interval_cayley_trace:', trace)
    print(f'e={e_tenths}/10 interval_cayley_det:', Y00 * Y11 - Y01 * Y10)
    print(f'e={e_tenths}/10 global_matrix_norm_error_lt:', global_error.b)
    print(f'e={e_tenths}/10 true_wkb_trace_interval:', trace_lower.a, trace_upper.b)
    return trace_lower.a, trace_upper.b


# The e=3/10 row preserves the separately reviewed adverse-ray certificate.
lower, upper = certify(
    3, 50_000, iv.mpf(407) / 1000, iv.mpf('2.48'), iv.mpf('2.57'),
    iv.mpf('6.4'), iv.mpf(24), iv.mpf(6_300_000), iv.mpf('15.01'),
)
assert upper < -2
print('e=3/10 certified_wkb_trace_below_minus_two: True')

# This one ray is elliptic at e=1/10; other rays and PDE modes remain open.
lower, upper = certify(
    1, 10_000, iv.mpf(607) / 1000, iv.mpf('1.332'), iv.mpf('0.39'),
    iv.mpf('0.52'), iv.mpf('3.5'), iv.mpf(5000), iv.mpf('0.044'),
)
assert lower > -2 and upper < 2
print('e=1/10 certified_one_ray_wkb_trace_between_minus_two_and_two: True')
