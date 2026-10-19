"""Independent math-only contrary controls; never imports a scientific consumer."""
import json
import math
from fractions import Fraction
import sympy as s
from scipy.integrate import quad
from scipy.special import i0e

# Independent generalized Gell-Mann basis, Tr(T_A T_B)=delta_AB/2.
T = []
for i in range(4):
    for j in range(i + 1, 4):
        a, b = s.zeros(4), s.zeros(4)
        a[i, j] = a[j, i] = s.Rational(1, 2)
        b[i, j], b[j, i] = -s.I / 2, s.I / 2
        T.extend([a, b])
for k in range(1, 4):
    diagonal = [1] * k + [-k] + [0] * (3 - k)
    T.append(s.diag(*diagonal) / s.sqrt(2 * k * (k + 1)))
assert len(T) == 15
assert all(s.simplify(s.trace(a * b)) == (s.Rational(1, 2) if i == j else 0)
           for i, a in enumerate(T) for j, b in enumerate(T))
casimir = s.simplify(sum((a * a for a in T), s.zeros(4)))
assert casimir == s.Rational(15, 8) * s.eye(4)
crossed = s.simplify(sum((s.trace(a * b * a * b) for a in T for b in T), s.Integer(0)))
square = s.trace(casimir * casimir)
same = 4 * (2 * square + crossed)
different = 2 * (square + crossed)
ordered = 4 * same + 12 * different
assert ordered == 750 and 4 * same + 6 * different == s.Rational(1185, 2)
face_delta = 4 * s.trace(casimir * (-s.eye(4)))
assert face_delta == -30
assert 8 * casimir != casimir

# Previously unlisted even N=10: native integer flow, exact full/HC/HE split.
N = 10
p = {(x, y, z): (-1) ** (x + y + z) for x in range(N) for y in range(N) for z in range(N)}
k = {x: 64 * value for x, value in p.items()}
rho = {x: 128 * value for x, value in p.items()}
assert all(k[x] - k[((x[0] - 1) % N, x[1], x[2])] == rho[x] for x in p)
assert sum(rho.values()) == 0
assert sum(4 for value in p.values() if value == 1) == 2 * N**3
assert all(6 * p[x] - sum(p[y] for y in [((x[0]+d)%N,x[1],x[2]) for d in (-1,1)]
                                       + [(x[0],(x[1]+d)%N,x[2]) for d in (-1,1)]
                                       + [(x[0],x[1],(x[2]+d)%N) for d in (-1,1)]) == 12 * p[x] for x in p)
whole = Fraction(64 * sum(v*v for v in k.values()), N**3)
hc = Fraction(64 * sum(v*v for v in rho.values()), 12 * N**3)
assert whole == 262144 and hc == Fraction(262144, 3)
assert whole - hc == Fraction(524288, 3)
wrong_species_hc = Fraction(64 * 64 * 4, 12)
assert wrong_species_hc != hc

# Synthetic circle law, not SU4 Gibbs sampling: discriminates normalizer and sign.
t = 3.0
C = lambda theta: 1 - math.cos(theta)
Z = float(i0e(t))
avg = lambda fun: quad(lambda theta: fun(theta) * math.exp(-t*C(theta)) / (2*math.pi*Z), -math.pi, math.pi, epsabs=1e-12)[0]
normalization = avg(lambda _: 1)
mean_C = avg(C)
mean_delta_C = avg(math.cos)
mean_grad_squared = avg(lambda theta: math.sin(theta)**2)
correct_secant = (2/t) * (math.log(float(i0e(t/2))) - math.log(Z))
wrong_secant = -correct_secant
assert abs(normalization - 1) < 1e-12 and Z < 1
assert abs(t*mean_grad_squared - mean_delta_C) < 1e-12
assert mean_C <= correct_secant <= -2*math.log(Z)/t
assert not mean_C <= wrong_secant
assert mean_delta_C > 0 and math.cos(math.pi) < 0
coefficient = Fraction(60) + Fraction(9,2) + Fraction(27,8) + Fraction(7,5)*(450+48*7168+24)
assert coefficient == Fraction(19296843,40) and coefficient < 2**19
assert 2**28 * 16 * 2**15 == 2**47
print(json.dumps({
    'kind':'INDEPENDENT_NEW_MATH_CONTRARY_CONTROLS_NOT_GROUND_OR_COLOUR_GIBBS',
    'exact': {'independent_basis_Casimir': '15/8', 'wrong_B_factor': 8, 'crossed_trace': str(crossed),
              'ordered_Wick': str(ordered), 'wrong_unordered_Wick': str(4*same+6*different),
              'negative_face_Delta': str(face_delta), 'N':N, 'fullE_perV':str(whole), 'HC_perV':str(hc),
              'HE_perV':str(whole-hc), 'wrong_independent_species_HC':str(wrong_species_hc),
              'energy_coefficient_outward':str(coefficient), 'cutoff_coefficient':'2^47/M'},
    'finite_float_synthetic_circle': {'t':t, 'Z':Z, 'normalized_norm_squared':normalization,
                                    'missing_normalizer_norm_squared':Z, 'mean_C':mean_C,
                                    'correct_secant_upper':correct_secant, 'wrong_sign_secant':wrong_secant,
                                    'mean_Delta_C':mean_delta_C, 't_mean_grad_squared':t*mean_grad_squared},
    'limits':'Exact finite algebra and synthetic circle quadrature controls only; no interacting-ground calculation, no SU4 Gibbs sample or Z_N evaluation.'
}, indent=2))
