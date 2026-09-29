"""Certify the e=1/10 ABC material-ray instability on the full-space orbit.

This certifies a 2x2 material monodromy, not an Euler PDE by itself. The
pressure-aware BAS and whole-space packet transfer are derived in material.md.
"""

from mpmath import iv

iv.dps = 40

e = iv.mpf(1) / 10
a = 1 / iv.sqrt(2)
steps = 600_000
period_upper = iv.mpf(6284) / 1000
m_lower = iv.mpf(607) / 1000
a_upper = iv.mpf(708) / 1000
L = iv.mpf('1.332')
K1 = iv.mpf('0.39')
K2 = iv.mpf('0.52')
local_upper = iv.mpf('3.5')
exp_upper = iv.mpf(5000)

assert a.b < a_upper and (a - e).a > m_lower
assert (2 * iv.pi).b < period_upper
assert ((a_upper + e) / m_lower).b < L
first_row = e / m_lower + e**2 / m_lower**2 + a_upper * e / m_lower**2
second_row = 2 * (e / m_lower + e**2 / m_lower**2)
assert first_row.b < K1 and second_row.b < K1
second_derivative_first = (
    e / m_lower + 3 * e**2 / m_lower**2 + 2 * e**3 / m_lower**3
    + a_upper * e / m_lower**2 + 2 * a_upper * e**2 / m_lower**3
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
assert iv.exp((L + iv.mpf('0.01')) * period_upper).b < exp_upper
global_error = local_upper * period_upper * (period_upper / steps) ** 2 * exp_upper
assert global_error.b < iv.mpf('0.0000121')

# x runs from 0 to -2pi, xdot=-(a+e cos x). Material n=(dx+dy)/2,
# z=dz obey d/dx[n,z]^T=C(x)[n,z]^T, C=-B/(a+e cos x),
# B=[[-e sin x,a],[-2e cos x,0]]. The norm bounds above apply to
# C and its first two x derivatives (the same bounds as for +B/h).
step = -2 * iv.pi / steps
Y00 = Y11 = iv.mpf(1)
Y01 = Y10 = iv.mpf(0)
for j in range(steps):
    x = (j + iv.mpf('0.5')) * step
    h = a + e * iv.cos(x)
    d = step * (e * iv.sin(x) / h) / 2
    u = step * (-a / h) / 2
    v = step * (2 * e * iv.cos(x) / h) / 2
    denominator = 1 - d - u * v
    P00 = (1 + d + u * v) / denominator
    P01 = 2 * u / denominator
    P10 = 2 * v / denominator
    P11 = (1 - d + u * v) / denominator
    Y00, Y01, Y10, Y11 = (
        P00 * Y00 + P01 * Y10,
        P00 * Y01 + P01 * Y11,
        P10 * Y00 + P11 * Y10,
        P10 * Y01 + P11 * Y11,
    )

trace = Y00 + Y11
trace_lower = trace - 2 * global_error
trace_upper = trace + 2 * global_error
print('material Cayley trace:', trace)
print('material Cayley determinant:', Y00 * Y11 - Y01 * Y10)
print('material global matrix norm error <', global_error.b)
print('true material trace interval:', trace_lower.a, trace_upper.b)
assert trace_lower.a > 2
print('e=1/10 certified material trace above two: True')
