"""Exact algebraic oracle for P253/0164; not a nonlinear Euler stability test."""

import sympy as sp

x, y, z, lam, e = sp.symbols("x y z lam e", real=True, positive=True)
X = sp.Matrix([x, y, z])
coordinates = (x, y, z)


def curl(v):
    return sp.Matrix(
        [
            sp.diff(v[2], y) - sp.diff(v[1], z),
            sp.diff(v[0], z) - sp.diff(v[2], x),
            sp.diff(v[1], x) - sp.diff(v[0], y),
        ]
    )


background = sp.Matrix(
    [sp.sin(z) - e * sp.sin(y), sp.cos(z) + e * sp.cos(x), e * (sp.cos(y) - sp.sin(x))]
)
assert all(sp.simplify(v) == 0 for v in curl(background) - background)
assert sp.simplify(sum((sp.diff(background[i], coordinates[i]) for i in range(3)), sp.Integer(0))) == 0
assert all(
    sp.simplify(v) == 0
    for v in background.jacobian(coordinates) * background
    - sp.Matrix([sp.diff(background.dot(background) / 2, q) for q in coordinates])
)

v0 = sp.Matrix([-z / 2, 0, x / 2])
w = sp.Matrix([5 * y * z, -4 * x * z, -x * y]) / 3
S = sp.diag(1, 2, -3)
assert curl(v0) == sp.Matrix([0, -1, 0])
assert curl(w) == S * X
A0 = -X.cross(v0) / 3
Aw = -X.cross(w) / 4
assert curl(A0) == v0
assert curl(Aw) == w

# chi = exp(-lam |X|^2). Divide it out of both curls to keep the moment polynomial.
P0 = sp.simplify(v0 - 2 * lam * X.cross(A0))
Pw = sp.simplify(w - 2 * lam * X.cross(Aw))
delta = sp.symbols("delta", positive=True)
chi = sp.exp(-lam * X.dot(X))
jet_origin = {x: 0, y: 0, z: 0}
omega_plus = curl(background.subs(e, 0) + chi * (P0 + delta * Pw))
assert omega_plus.subs(jet_origin) == sp.zeros(3, 1)
assert omega_plus.jacobian(coordinates).subs(jet_origin) == sp.Matrix(
    [[delta, 0, 1], [0, 2 * delta, 0], [0, 0, -3 * delta]]
)
assert sp.factor(omega_plus.jacobian(coordinates).subs(jet_origin).det()) == -6 * delta**3
assert background.subs(jet_origin) + curl(chi * P0).subs(jet_origin) == sp.Matrix(
    [0, e, e]
)
H = y * x**2 - y**3 / 3
Hessian = sp.hessian(H, coordinates)
assert sp.simplify(sum((sp.diff(H, q, 2) for q in coordinates), sp.Integer(0))) == 0


def gaussian_moment(power, rate):
    if power % 2:
        return sp.Integer(0)
    half = power // 2
    return sp.sqrt(sp.pi) * sp.factorial2(power - 1) / (
        2**half * rate ** (half + sp.Rational(1, 2))
    )


cross_polynomial = sp.Poly(sp.expand(2 * (P0.T * Hessian * Pw)[0]), x, y, z)
q_cross = sp.simplify(
    sum(
        (
            coefficient
            * gaussian_moment(monomial[0], 2 * lam)
            * gaussian_moment(monomial[1], 2 * lam)
            * gaussian_moment(monomial[2], 2 * lam)
            for monomial, coefficient in cross_polynomial.terms()
        ),
        sp.Integer(0),
    )
)
# The U0-W moment reduces by curl integration to the displayed cos(z) polynomial.
z_cos = sp.sqrt(sp.pi / lam) * sp.exp(-1 / (4 * lam))
z2_cos = z_cos * (1 / (2 * lam) - 1 / (4 * lam**2))
q_background = sp.factor(-sp.pi * (z_cos + lam * z2_cos) / (6 * lam**3))
q_total = sp.factor(q_background + q_cross)
side_wave = sp.diff(background, e)
side_moment_density = sp.expand(2 * (side_wave.T * Hessian * Pw)[0])
# The Gaussian is even in z. Every side-wave stress zeroth-moment
# entry is odd in at least one coordinate; the cubic moment is odd in z.
assert sp.simplify(side_moment_density + side_moment_density.subs(z, -z)) == 0
assert all(
    any(
        sp.simplify(side_wave[i] * Pw[j] + (side_wave[i] * Pw[j]).subs(q, -q))
        == 0
        for q in coordinates
    )
    for i in range(3)
    for j in range(3)
)
assert sp.simplify(
    q_cross + 13 * sp.sqrt(2) * sp.pi ** sp.Rational(3, 2) / (4608 * lam ** sp.Rational(7, 2))
) == 0
assert sp.simplify(
    q_background
    + sp.pi ** sp.Rational(3, 2)
    * (6 * lam - 1)
    * sp.exp(-1 / (4 * lam))
    / (24 * lam ** sp.Rational(9, 2))
) == 0
assert sp.N(q_total.subs(lam, 1)).is_negative is True
print("curl background = background; steady Euler residual = 0")
print("curl v0 = -ey; curl w = diag(1,2,-3) X")
print("H:", H)
print("Q_background:", q_background)
print("Q_compact_cross:", sp.factor(q_cross))
print("Q_sidewave: 0 for every lambda and e by z parity")
print("Q_total:", q_total)
print("Q_total(lambda=25):", sp.N(q_total.subs(lam, 25), 15))
