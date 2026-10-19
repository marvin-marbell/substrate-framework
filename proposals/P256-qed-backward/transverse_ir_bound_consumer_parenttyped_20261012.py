"""New exact native geometry/factor and separately nonnative spectral controls.
No native full-H/ground backend, old producer import, or numerical eigensolver.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import zipfile
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

import sympy as sp

ROOT = Path(__file__).resolve().parent
Poly = tuple[int, ...]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Cyclotomic:
    """Exact integer quotient ring Z[w]/Phi_(2N), w=exp(i*pi/N)."""

    def __init__(self, n: int):
        self.n = n
        x = sp.Symbol('w')
        polynomial = sp.Poly(sp.cyclotomic_poly(2*n, x), x)
        self.degree: int = cast(int, polynomial.degree())
        self.modulus = tuple(int(polynomial.nth(i)) for i in range(self.degree))
        self.zero = (0,)*self.degree
        self.one = self.scalar(1)
        w = (0, 1)+(0,)*(self.degree-2)
        powers = [self.one]
        for _ in range(1, 2*n):
            powers.append(self.mul(powers[-1], w))
        self.powers = powers
        assert self.mul(powers[-1], w) == self.one
        assert self.power(n) == self.scalar(-1)

    def scalar(self, value: int) -> Poly:
        return (value,)+(0,)*(self.degree-1)

    def add(self, a: Poly, b: Poly) -> Poly:
        return tuple(x+y for x, y in zip(a, b))

    def neg(self, a: Poly) -> Poly:
        return tuple(-x for x in a)

    def sub(self, a: Poly, b: Poly) -> Poly:
        return self.add(a, self.neg(b))

    def mul(self, a: Poly, b: Poly) -> Poly:
        coefficients = [0]*(2*self.degree-1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                coefficients[i+j] += ai*bj
        for k in range(2*self.degree-2, self.degree-1, -1):
            lead = coefficients[k]
            if lead:
                for i, value in enumerate(self.modulus):
                    coefficients[k-self.degree+i] -= lead*value
        return tuple(coefficients[:self.degree])

    def total(self, values: Any) -> Poly:
        result = self.zero
        for value in values:
            result = self.add(result, value)
        return result

    def power(self, exponent: int) -> Poly:
        return self.powers[exponent % (2*self.n)]

    def conjugate(self, a: Poly) -> Poly:
        return self.total(tuple(coefficient*x for x in self.power(-i))
                          for i, coefficient in enumerate(a))


def exact_fourier(n: int) -> dict[str, Any]:
    ring = Cyclotomic(n)
    mul, add, total = ring.mul, ring.add, ring.total
    # Tensor products of these exact root orthogonality identities prove the
    # normalized native three-dimensional transform is unitary for ALL modes.
    for m in range(n):
        assert total(ring.power(2*m*x) for x in range(n)) == ring.scalar(n if m == 0 else 0)
        for x in range(n):
            for length in (1, 2):
                y = (x+length) % n
                seam = (-1)**((x+length)//n)
                ap = ring.power((2*m+1)*y)
                assert tuple(seam*v for v in ap) == ring.power((2*m+1)*(x+length))
                periodic = ring.power(2*m*y)
                assert periodic == ring.power(2*m*(x+length))
    minimum = ring.sub(ring.scalar(2), add(ring.power(2), ring.power(-2)))
    nonzero, minimum_modes, cycle_rank = 0, [], Counter()
    for momentum in itertools.product(range(n), repeat=3):
        # t=i*hatp is the CENTERED incidence row. Native phases are exact roots.
        t = [ring.sub(ring.power(m), ring.power(-m)) for m in momentum]
        q2 = ring.neg(total(mul(v, v) for v in t))
        for i, m in enumerate(momentum):
            uncentered = ring.sub(ring.one, ring.power(-2*m))
            assert mul(ring.power(m), uncentered) == t[i]
            assert ring.conjugate(t[i]) == ring.neg(t[i])
        assert ring.conjugate(q2) == q2
        # H=q^2 P_T. Polynomial identities avoid division and floats entirely.
        h = [[add(q2 if i == j else ring.zero, mul(t[i], t[j]))
              for j in range(3)] for i in range(3)]
        assert all(total(mul(t[i], h[i][j]) for i in range(3)) == ring.zero for j in range(3))
        assert all(total(mul(h[i][k], h[k][j]) for k in range(3)) == mul(q2, h[i][j])
                   for i in range(3) for j in range(3))
        assert all(ring.conjugate(h[j][i]) == h[i][j] for i in range(3) for j in range(3))
        # Face-center stripped plaquette column ci=t_j, cj=-t_i is native curl.
        for i, j in ((0, 1), (0, 2), (1, 2)):
            c: list[Poly] = [ring.zero]*3
            c[i], c[j] = t[j], ring.neg(t[i])
            assert total(mul(t[k], c[k]) for k in range(3)) == ring.zero
            assert [total(mul(h[k][l], c[l]) for l in range(3)) for k in range(3)] == [mul(q2, v) for v in c]
        if momentum == (0, 0, 0):
            assert q2 == ring.zero and h == [[ring.zero]*3 for _ in range(3)]
            cycle_rank[0] += 1  # Local curls vanish; separate harmonic rank is3.
        else:
            assert q2 != ring.zero
            assert total(h[i][i] for i in range(3)) == mul(ring.scalar(2), q2)
            cycle_rank[2] += 1
            nonzero += 1
        if q2 == minimum:
            minimum_modes.append(momentum)
    assert nonzero == n**3-1 and len(minimum_modes) == 6
    # Explicit native adverse mode shows omitting center phase breaks closure.
    t0 = ring.sub(ring.power(1), ring.power(-1))
    t1 = ring.sub(ring.power(2), ring.power(-2))
    wrong = ring.sub(mul(ring.sub(ring.one, ring.power(-2)), t1),
                     mul(ring.sub(ring.one, ring.power(-4)), t0))
    assert wrong != ring.zero
    assert ring.sub(ring.one, ring.power(-1)) != ring.zero  # AP electric p=0 would not be a toron.
    return {'scope': 'EXACT_NATIVE_GRAPH_FOURIER_AP_NOT_GROUND_OR_FULLH', 'N': n,
            'cyclotomic_modulus_2N': list(ring.modulus)+[1], 'all_modes': n**3,
            'all_nonzero_modes': nonzero, 'native_centered_projector_identities_exact': True,
            'plaquette_span_rank_by_mode': dict(cycle_rank), 'separate_p0_harmonic_rank': 3,
            'minimum_modes': minimum_modes, 'q_N': str(2*sp.sin(sp.pi/n)),
            'q_N_squared_exact_ring': list(minimum), 'real_norm2_upper': '1/256',
            'complex_inverse_upper_ALL_modes': 'P_T/128',
            'positive_band_ALL_modes': 'omega P_T/128',
            'q_N_band': 'kappa q_N P_T/128',
            'wrong_centering_rejected': True, 'wrong_electric_AP_shift_rejected': True,
            'native_AP_one_two_hop_all_1D_modes_exact': True}


def native_factors(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    volume = n**3

    def move(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        return tuple((v+1) % n if i == axis else v for i, v in enumerate(x))

    def divergence(flow: dict[tuple[tuple[int, ...], int], F]) -> dict[tuple[int, ...], F]:
        output = dict.fromkeys(sites, F(0))
        for (x, axis), value in flow.items():
            output[x] += value
            output[move(x, axis)] -= value
        return output

    origin = (n-1, 0, 0)  # Empty checkerboard endpoint, source at its occupied neighbor.
    edge = (origin, 0)
    face = {edge: F(1), (move(origin, 0), 1): F(1),
            (move(origin, 1), 0): F(-1), (origin, 1): F(-1)}
    assert not any(divergence(face).values()) and sum(v*v for v in face.values()) == 4
    rho = {x: F(128*(-1)**sum(x)) for x in sites}
    longitudinal = {(x, i): F(64, 3)*(-1)**sum(x) for x in sites for i in range(3)}
    assert divergence(longitudinal) == rho
    assert sum(longitudinal[e]*v for e, v in face.items()) == 0
    transverse = {e: F(3, 8)*v for e, v in face.items()}
    electric = {e: value+transverse.get(e, F(0)) for e, value in longitudinal.items()}
    assert divergence(electric) == rho
    dot = lambda a, b: sum((value*b.get(e, F(0)) for e, value in a.items()), F(0))
    norm = lambda a: dot(a, a)
    he, hc = 64*norm(transverse), 64*norm(longitudinal)
    assert he == 36 and hc == F(262144*volume, 3)
    assert 64*norm(electric) == he+hc
    s = F(2, 7)
    shifted = {e: value+s*face.get(e, F(0)) for e, value in electric.items()}
    full_delta = 64*(norm(shifted)-norm(electric))
    assert full_delta == 128*s*dot(face, electric)+64*s*s*norm(face)
    assert dot(face, electric) == F(3, 2) and divergence(shifted) == rho
    # Effective resistance of one edge: cubic symmetry plus tr(P_L)=V-1.
    # Fourier control has verified the unique zero incidence eigenmode.
    plnorm2 = F(volume-1, 3*volume)
    delta_hc = 128*longitudinal[edge]+64*plnorm2
    delta_he = 128*transverse[edge]+64*(1-plnorm2)
    delta_full = 128*electric[edge]+64
    assert delta_hc+delta_he == delta_full and delta_hc != 0
    independent_copies_hc = F(4096*volume, 3)
    assert hc/independent_copies_hc == 64  #64 separately priced charged HC terms are still wrong.
    counts, apcounts, aliases = [0]*3, [0]*3, [0]*3
    coefficient_load = F(0)
    for role, profile in enumerate(profiles):
        for path in profile['paths']:
            coefficient_load += F(profile['load'])*path['length']*(abs(F(path['c0']))+abs(F(path['ci']))) if role else F(0)
            for x in sites:
                y, seam = x, 1
                pathflow: dict[tuple[tuple[int, ...], int], F] = {}
                for _ in range(path['length']):
                    e = (y, path['axis'])
                    pathflow[e] = pathflow.get(e, F(0))+1
                    seam *= -1 if y[path['axis']] == n-1 else 1
                    y = move(y, path['axis'])
                d = divergence(pathflow)
                assert all(d[z] == int(z == x)-int(z == y) for z in sites)
                assert seam == (-1)**((x[path['axis']]+path['length'])//n)
                counts[role] += 1
                apcounts[role] += int(seam == -1)
                aliases[role] += int(n == 4 and path['length'] == 2)
    assert counts == [3*volume, 6*volume, 6*volume]
    assert apcounts == [3*n*n, 9*n*n, 9*n*n]
    assert coefficient_load == F(46425051219964815, 8796093022208)
    width = 4*volume
    local = sum(15 << (4*j) for j, x in enumerate(sites) if sum(x) % 2 == 0)
    assert local.bit_count() == 2*volume
    mask = sum(local << (role*width) for role in range(65)) | (15 << (65*width))
    destination, source = sites.index(origin), sites.index(move(origin, 0))

    def transfer(bits: int, dest: int, src: int) -> tuple[int, int]:
        assert bits & (1 << src) and not bits & (1 << dest)
        sign = (-1)**((bits & ((1 << src)-1)).bit_count())
        bits ^= 1 << src
        sign *= (-1)**((bits & ((1 << dest)-1)).bit_count())
        return bits | (1 << dest), sign

    for role in range(1, 65):
        offset = role*width
        changed, sign = transfer(mask, offset+4*destination, offset+4*source)
        restored, backsign = transfer(changed, offset+4*source, offset+4*destination)
        assert restored == mask and sign*backsign == 1
        assert ((changed >> offset) & ((1 << width)-1)).bit_count() == 2*volume
        assert (changed & ((1 << width)-1)) == local and changed >> (65*width) == 15
        changed_rho = dict(rho)
        changed_rho[origin] += 1
        changed_rho[move(origin, 0)] -= 1
        hop_electric = dict(electric)
        hop_electric[edge] += 1
        assert divergence(hop_electric) == changed_rho
        # Transverse-only k falsely leaves the original divergence rho.
        assert changed_rho != rho
    gx = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    gy = sp.diag(sp.I, -sp.I, 1, 1)
    fx = sp.diag(2, -1, 1, -2)
    fy = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]])
    u = gx*gy
    block = (fx*u+u*fy)/2
    transformed = ((gx*fx*gx.H)*(gx*u*gy.H)+(gx*u*gy.H)*(gy*fy*gy.H))/2
    assert transformed == gx*block*gy.H and gx.det() == gy.det() == 1
    assert gx*gy != gy*gx and fx*fy != fy*fx
    jordan = fx*fy+fy*fx
    assert jordan != sp.zeros(4) and sp.trace(jordan**2) > 0
    assert (gx*fx*gx.H)*(gx*fy*gx.H)+(gx*fy*gx.H)*(gx*fx*gx.H) == gx*jordan*gx.H
    assert sp.trace((gx*jordan*gx.H)**2) == sp.trace(jordan**2)
    return {'scope': 'EXACT_NATIVE_FULL_AFFINE_GAUSS_AP_CAR_ROLE_MATRIX_FACTORS_NOT_GROUND',
            'N': n, 'literal_profile_counts': counts, 'AP222_seam_counts': apcounts,
            'N4_distinct_twohop_alias_literals': aliases, 'all64_coefficient_load_W': str(coefficient_load),
            'canonical_each65': 2*volume, 'charged_distinct_CAR_transfers': 64, 'Ref4_unchanged': True,
            'rho_total': '128 checkerboard parity, Ref4 net0', 'full_HE': str(he), 'collective_HC': str(hc),
            'fullE64_equals_HE_plus_HC': True, 'incoming_transverse_face_mean': '3/8',
            'full_closed_boost_delta': str(full_delta), 'Gauss_closed_boost_exact': True,
            'one_hop_PL_norm2': str(plnorm2), 'one_hop_delta_HE': str(delta_he),
            'one_hop_delta_HC': str(delta_hc), 'one_hop_full_delta': str(delta_full),
            'wrong_transverse_only_open_hop_rejected': True, 'wrong_independent_HC_ratio': '64',
            'ordered_full_matrix_SU4_covariance_exact': True, 'fullmatrix_Jordan_covariance_exact': True,
            'Phi_SU4_neutral_Ref_terms_retained_by_exact_commutation_not_deleted': True}


def spectral_controls(kappa: F, qvalues: list[F]) -> dict[str, Any]:
    # These are exact NONNATIVE positive spectral representations/method controls.
    c = F(1, 128)
    jmax = 4
    v, k, e, qs = sp.symbols('v kappa epsilon q', positive=True)
    continuum_density = v/(128*k)
    exact_integrals = [sp.integrate(v**j*continuum_density, (v, 0, k)) for j in range(jmax+1)]
    assert all(sp.simplify(value-k**(j+1)/(128*(j+2))) == 0 for j, value in enumerate(exact_integrals))
    assert sp.integrate(continuum_density/v, (v, 0, k)) == sp.Rational(1, 128)
    physical_density = e/(128*k*qs)
    assert sp.simplify(sp.integrate(physical_density/e, (e, 0, k*qs))) == sp.Rational(1, 128)
    assert sp.simplify(sp.integrate(physical_density, (e, 0, k*qs))/qs) == k/256
    assert (F(1, 128)+0)/2 == F(1, 256)
    singleton_measure = sp.FiniteSet(sp.Rational(1, 2)).measure
    assert singleton_measure == 0
    models = []
    for q in qvalues:
        velocity = kappa/2
        residue = velocity/256
        atomic_mass = residue*q
        atomic_inverse = atomic_mass/(velocity*q)
        assert atomic_inverse == F(1, 256)
        moments = [residue*velocity**j for j in range(jmax+1)]
        assert all(value <= kappa**(j+1)*c for j, value in enumerate(moments))
        continuum_moments = [kappa**(j+1)/(128*(j+2)) for j in range(jmax+1)]
        assert continuum_moments[0] == kappa/256
        assert all(value <= kappa**(j+1)*c for j, value in enumerate(continuum_moments))
        for eta in (kappa/8, kappa/2, kappa):
            continuum_lowmass = eta*eta/(256*kappa)
            assert continuum_lowmass <= eta*c
        collapsing_v = q*kappa/8
        collapsing_a = collapsing_v/256
        assert 0 < collapsing_v <= kappa and collapsing_a/collapsing_v == F(1, 256)
        massive_energy = 16*kappa
        massive_inverse = F(1, 256)
        assert massive_energy > kappa*q and massive_inverse <= c
        models.append({'q': str(q), 'atomic_energy': str(q*velocity), 'atomic_mu_mass': str(atomic_mass),
                       'atomic_sigma_mass': str(residue), 'atomic_inverse': str(atomic_inverse),
                       'atomic_sigma_moments_j0to4': [str(v) for v in moments],
                       'atomless_sigma_density': 'v/(128*kappa) on(0,kappa]',
                       'atomless_mu_density': 'epsilon/(128*kappa*q) on(0,kappa*q]',
                       'atomless_inverse': '1/128', 'atomless_sigma_moments_j0to4': [str(v) for v in continuum_moments],
                       'atomless_atoms': 0, 'collapsing_velocity': str(collapsing_v),
                       'collapsing_sigma_mass': str(collapsing_a), 'collapsing_inverse': '1/256',
                       'massive_energy': str(massive_energy), 'massive_mu_mass': str(massive_energy*massive_inverse),
                       'massive_inverse': str(massive_inverse), 'massive_positive_lowband_mass': '0',
                       'zero_model_all_mass': '0'})
    gram = sp.Matrix([[1, sp.I], [-sp.I, 1]])/256
    z = sp.Matrix([1, -sp.I])/sp.sqrt(2)
    assert sp.simplify((z.H*gram*z)[0]) == sp.Rational(1, 128)
    x, y = sp.symbols('x y', real=True)
    assert (sp.Matrix([[x, y]])*gram*sp.Matrix([x, y]))[0].expand() == ((x*x+y*y)/256).expand()
    assert (gram-sp.eye(2)/256).det() == -sp.Rational(1, 256**2)
    # An unrelated selected feature Q destroys half the actual unit-energy source.
    full_e = sp.Matrix([sp.Rational(1, 16), sp.Rational(1, 16)])
    selected = sp.diag(1, 0)*full_e
    assert (full_e.H*full_e)[0] == sp.Rational(1, 128)
    assert (selected.H*selected)[0] == sp.Rational(1, 256)
    oversized = cast(sp.MatrixBase, sp.sqrt(2)*full_e)
    assert sp.simplify((oversized.H*oversized)[0]) == sp.Rational(1, 64)
    assert sp.simplify((oversized.H*sp.diag(1, 0)*oversized)[0]) == sp.Rational(1, 128)
    # Physical plaquette curl amplitude at p=(p1,0,0) is q times unit polarization.
    q = sp.Symbol('q', positive=True)
    mu_physical = q/256
    mu_face = q*q*mu_physical
    assert mu_face/mu_physical == q*q and mu_face/q**2 == mu_physical
    return {'scope': 'EXACT_NONNATIVE_POSITIVE_MEASURE_AND_ADVERSE_METHOD_CONTROLS_NOT_NATIVE_SPECTRUM',
            'kappa': str(kappa), 'models': models,
            'atomless_exact_integrals_j0to4': [str(value) for value in exact_integrals],
            'real_conjugate_pair_compatible': 'inverse1/128 on half BZ S, zero on -S => real inverse1/256',
            'singleton_exact_Lebesgue_measure': str(singleton_measure),
            'complex_imaginary_cross_Gram': [[str(v) for v in gram.row(i)] for i in range(2)],
            'real_norm2_upper': '1/256', 'complex_norm2_saturation': '1/128',
            'false_complex_factor1_rejected': True,
            'selected_source_weakening_actual_inverse': '1/64', 'selected_false_pass_inverse': '1/128',
            'selected_source_weakening_rejected': True,
            'face_vs_physical_amplitude_factor2': 'q^2', 'source_normalization_weakening_rejected': True,
            'pointwise_countermodel': {'multiplier1': 'P_T/256 a.e.',
                                       'multiplier2': 'same a.e.; value P_T/64 at one chosen nonzero p',
                                       'difference_Haar_measure': '0', 'same_L2_operator': True,
                                       'arbitrary_pointwise_upgrade_rejected': True},
            'zero_velocity_mass_atom_not_forced': True, 'zero_velocity_mass_limit_excluded_by_bound': True,
            'positive_lower_velocity_not_implied': True,
            'atomic_atomless_massive_zero_ALL_allowed': True, 'phase_stiffness_atom_commoncone_inferred': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    before = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        profiles = json.loads(archive.read(request['profile_member']))['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    scalar_interface = json.loads((ROOT/request['scalar_interface']).read_bytes())
    assert scalar_interface['owner'] == 'ClosedBetaInverse'
    assert scalar_interface['finite']['real_inverse_upper'] == {'numerator': 1, 'denominator': 256, 'multiplier': '||h||^2'}
    assert scalar_interface['complexification']['imaginary_cross_retained']
    receipt = {'kind': 'NEW_TRANSVERSE_IR_OPERATOR_UPPER_EXACT_CONTROLS_NOT_NATIVE_SPECTRUM_OR_ADMISSION',
               'native_graph_Fourier': [exact_fourier(n) for n in request['N']],
               'native_full_affine_factors': [native_factors(n, profiles) for n in request['N']],
               'nonnative_positive_spectral_controls': spectral_controls(F(request['kappa']), [F(v) for v in request['q_values']]),
               'proof': request['proof'], 'interface': request['interface'], 'conditional_scalar_interface': request['scalar_interface'],
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: sha((ROOT/name).read_bytes()) for name in request['pins']},
               'native_ground_or_fullH_backend': False, 'old_producer_root_solver_checker_rerun': False,
               'scientific_admission': False, 'campaign_complete': False}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'all_finite_native_modes_checked': sum(n**3 for n in request['N']),
                      'completed_exact_controls': True, 'native_spectrum_computed': False}))


if __name__ == '__main__':
    main()
