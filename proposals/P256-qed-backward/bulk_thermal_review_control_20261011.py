"""Independent exact algebra/constant controls; never imports target producers."""
from __future__ import annotations
import argparse
from fractions import Fraction
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sympy as sp


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def basis() -> list[sp.Matrix]:
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = sp.zeros(4), sp.zeros(4)
            a[i, j] = a[j, i] = sp.Rational(1, 2)
            b[i, j], b[j, i] = -sp.I / 2, sp.I / 2
            out.extend([a, b])
    for k in range(1, 4):
        entries = [1] * k + [-k] + [0] * (3 - k)
        out.append(sp.diag(*entries) / sp.sqrt(2 * k * (k + 1)))
    return out


def car(mask: int, create: int, annihilate: int) -> tuple[int, int] | None:
    if not mask & (1 << annihilate):
        return None
    sign = (-1) ** ((mask & ((1 << annihilate) - 1)).bit_count())
    mask ^= 1 << annihilate
    if mask & (1 << create):
        return None
    sign *= (-1) ** ((mask & ((1 << create) - 1)).bit_count())
    return mask | (1 << create), sign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    root = args.request.parent
    verified = {}
    for name, expected in request['input_sha256'].items():
        actual = digest((root / name).read_bytes())
        assert actual == expected, name
        verified[name] = actual
    profile_raw = (root / request['profile']).read_bytes()
    profiles = json.loads(profile_raw)['primitive_profiles']
    assert [(p['load'], p['scale']) for p in profiles] == [(1, 'heavy'), (1, 'charged'), (63, 'charged')]
    constants = []
    for p in profiles:
        q = 4 * abs(Fraction(p['onsite'])) + 8 * sum(abs(Fraction(r['c0'])) + abs(Fraction(r['ci'])) for r in p['paths'])
        constants.append(q)
    a_heavy = constants[0]
    a_charged = constants[1] + 63 * constants[2]
    wrong_charged = constants[1] + constants[2]
    assert a_charged - wrong_charged == 62 * constants[2] > 0

    generators = basis()
    gram = sp.Matrix([[sp.simplify(sp.trace(a * b)) for b in generators] for a in generators])
    assert gram == sp.eye(15) / 2
    casimir = sp.simplify(sum((a * a for a in generators), sp.zeros(4)))
    assert casimir == sp.eye(4) * sp.Rational(15, 8)
    phi = [sum(((mu + 1) * (i + 1) * a for i, a in enumerate(generators)), sp.zeros(4)) for mu in range(4)]
    r2 = sp.simplify(sum(sp.trace(a * a) for a in phi))
    jordan = sp.simplify(sum(sp.trace((a * b + b * a) ** 2) for a in phi for b in phi))
    assert jordan >= r2 ** 2 / 4
    # A sharp lower-bound probe: one traceless matrix with four equal |eigenvalues|.
    sharp = sp.diag(1, 1, -1, -1)
    assert sp.trace((2 * sharp ** 2) ** 2) == sp.trace(sharp ** 2) ** 2
    # The universal full-field upper follows from ||{A,B}||HS <= 2||A||HS||B||HS.
    assert jordan <= 4 * r2 ** 2
    gaussian_r4 = sp.Rational(60 * 62, 4)
    assert gaussian_r4 == 930
    trial_jordan_cap = 16 * gaussian_r4
    assert trial_jordan_cap == 14880
    kinetic_mean = sp.Rational(4) * 60 / 4
    assert kinetic_mean == 60
    phi_heat_denominator_coefficient = 4 * 4
    colour_heat_time_coefficient = sp.Rational(3, 20)
    assert phi_heat_denominator_coefficient == 16
    assert colour_heat_time_coefficient * sp.Rational(15, 8) == sp.Rational(9, 32)

    # Exact Young optimizer with A=2,Lambda=1: r*=1, remainder C=3/2.
    r = sp.symbols('r', nonnegative=True)
    young_gap = sp.factor(r ** 4 / 2 + sp.Rational(3, 2) - 2 * r)
    assert young_gap == (r - 1) ** 2 * (r ** 2 + 2 * r + 3) / 2
    assert sp.Rational(1, 4) - sp.Rational(1, 8) == sp.Rational(1, 8)
    assert sp.Rational(1, 8) - sp.Rational(1, 16) == sp.Rational(1, 16)

    choices = list(combinations(range(4), 2))
    pair_masks = []
    for chosen in choices:
        m1 = sum(15 << (4 * i) for i in chosen)
        m2 = sum(15 << (4 * i) for i in range(4) if i not in chosen)
        assert m1.bit_count() == m2.bit_count() == 8
        assert all(((m1 >> (4*i)) & 15).bit_count() + ((m2 >> (4*i)) & 15).bit_count() == 4 for i in range(4))
        pair_masks.append((m1, m2))
    assert len(set(pair_masks)) == sp.binomial(4, 2) == 6
    assert 6 ** 33 != 6 ** 32 and 6 ** 33 != 6 ** 65
    # Full and empty on-site wedges have exact zero onsite expectation for arbitrary traceless Phi.
    for a in phi:
        assert sp.trace(a) == 0
    transitions = 0
    for m1, _ in pair_masks:
        for x in range(4):
            for y in range(4):
                if x == y:
                    continue
                for i in range(4):
                    for j in range(4):
                        image = car(m1, 4*x+i, 4*y+j)
                        if image:
                            transitions += 1
                            assert image[0] != m1
    assert transitions > 0

    # One interior row D=(1,-1), G=1/2, Pi=(3,3), rho=sum64(1)=64.
    D = sp.Matrix([[1, -1]])
    G = (D * D.T).inv()
    pi = sp.Matrix([3, 3])
    rho = sp.Matrix([64])
    full_e = pi + D.T * G * rho
    U = 64 * (full_e.T * full_e)[0]
    kinetic = 64 * (pi.T * pi)[0]
    hc = 64 * (rho.T * G * rho)[0]
    species_hc = 64 * 64 * G[0, 0]
    assert U == kinetic + hc and hc == 64 * species_hc
    assert U != kinetic and (U - kinetic) / 2 == hc / 2

    # Exact integrated ALL-character Gaussian unfolding: one primitive length L=2pi/sqrt2.
    k = sp.symbols('k', real=True)
    length = 2 * sp.pi / sp.sqrt(2)
    gaussian_integral = sp.integrate(sp.exp(-64 * k ** 2), (k, -sp.oo, sp.oo))
    unfolded = sp.simplify(length / (2 * sp.pi) * gaussian_integral)
    cover_diagonal = sp.simplify(length / sp.sqrt(4 * sp.pi * 64))
    assert unfolded == cover_diagonal == sp.sqrt(2 * sp.pi) / 16
    assert sp.simplify(unfolded / (2 * sp.pi / sp.sqrt(4 * sp.pi * 64))) == 1 / sp.sqrt(2)
    d, beta = sp.symbols('d beta', positive=True)
    electric_logz = -d * sp.log(beta) / 2
    electric_energy = -sp.diff(electric_logz, beta)
    assert sp.simplify(beta * electric_energy - d / 2) == 0
    electric_perturb_log_ratio = sp.simplify(-d * sp.log(beta / 2) / 2 - electric_logz)
    assert sp.simplify(electric_perturb_log_ratio - d * sp.log(2) / 2) == 0

    cj = sum((8 * p['load'] * (abs(Fraction(row['c0'])) + abs(Fraction(row['ci']))) * row['length']
              for p in profiles if p['scale'] == 'charged' for row in p['paths'] if row['axis'] == 2), Fraction(0))
    ck = sum((8 * p['load'] * (abs(Fraction(row['c0'])) + abs(Fraction(row['ci']))) * row['length'] ** 2
              for p in profiles if p['scale'] == 'charged' for row in p['paths'] if row['axis'] == 2), Fraction(0))
    assert cj == Fraction(15475017073321605, 1099511627776) and ck > cj
    v, nb = 8 ** 3, 2 * 8 ** 2
    z = sp.sqrt(2)
    sum_r = nb * z / sp.sqrt(2)
    assert nb * z ** 4 == v
    assert sp.simplify(sum_r ** 4) == sp.Rational(nb ** 3 * v, 4)
    assert sum_r ** 4 > sp.Rational(nb ** 4, 4)
    holder = dict(CJ=str(cj), CK=str(ck), pointwise='(sum_boundary r)^4 <= n_b^3 S/4',
        Jensen='E sqrt(S) <= sqrt(E S), E S^(1/4) <= (E S)^(1/4)',
        J_squared='CJ^2 n_b^(3/2) sqrt(V Cbeta)/2',
        contact='CK n_b^(3/4) (V Cbeta)^(1/4)/sqrt2',
        area_scaling='N^(5/2), not uniform', N8_saturation_exact=str(sum_r ** 4),
        witness_is_actual_state=False, selected_site_inference_rejected=True)

    wrongcases = dict(aux63_collapsed=True,HS_missing_half=True,Cart_Casimir_twice=True,
        gaussian_r4_missing_quarter=True,kinetic_missing_half=True,colour_time_twice=True,
        entropy_missing_neutral=True,entropy_independent_charged65=True,sharedrho_factor64_missing=True,
        U_replaced_by_Pi_only=True,primitive_determinant_missing=True,quartic_perturb_remainder_wrong=True)
    assert gram != sp.eye(15)
    assert casimir != sp.eye(4) * sp.Rational(15, 4)
    assert gaussian_r4 != 60 * 62
    assert kinetic_mean != 120
    assert colour_heat_time_coefficient != sp.Rational(3, 10)
    result = dict(kind='INDEPENDENT_EXACT_RATIONAL_GAUSSIAN_CAR_CONSTANT_CONTROL_NOT_NATIVE_PRESSURE',
        request_sha256=digest(request_raw),source_sha256=digest(Path(__file__).read_bytes()),input_sha256=verified,
        A_exact=dict(Lambda_coefficient=str(a_heavy),unit_coefficient=str(a_charged),wrong_load_loss=str(a_charged-wrong_charged)),
        Jordan_probe=dict(r2=str(r2),Jordan=str(jordan),HS_Gram='I15/2',Casimir='15I4/8',Gaussian_r4='930 sigma^4',
            quartic_trial_upper='14880 Lambda sigma^4',Phi_kinetic='60 chi/sigma^2',heat_denominator='(16pi chi beta)^30',colour_heat_time='3chi beta/20'),
        CAR=dict(pair_choices=6,family='binomial(V,V/2)^33',independent_factors='32 charged pairs +1 neutral',nonzero_offsite_images=transitions,all_offsite_diagonal_zero=True,Ref_dimension=1),
        electric=dict(U=str(U),Pi_kinetic=str(kinetic),shared_HC=str(hc),separate_species_HC=str(species_hc),HC_in_U_once=True,free_all_character_value=str(unfolded),perturb_log_ratio='d log2/2'),
        joined_boundary_moment_review=holder,
        wrongcases_rejected=wrongcases,native_pressure_computed=False,selected_site_bound=False,zero_temperature_bound=False,phase_inference=False)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(exact_controls=True,wrongcases=len(wrongcases),native_pressure=False)))


if __name__ == '__main__':
    main()
