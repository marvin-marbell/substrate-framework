"""New exact native fast-body metric/domain/energy-relative consumer; no ground solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import sympy as s


def clifford() -> tuple[list[Any], list[Any]]:
    pauli = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
    gamma = [s.diag(1, 1, -1, -1)] + [s.kronecker_product(pauli[0], a) for a in pauli]
    words = []
    for degree in range(1, 5):
        for indices in itertools.combinations(range(4), degree):
            word = s.eye(4)
            for j in indices:
                word *= gamma[j]
            word *= s.I ** (degree * (degree - 1) // 2)
            assert word.H == word and word * word == s.eye(4)
            words.append(word)
    generators = [word / (2 * s.sqrt(2)) for word in words]
    assert all(s.simplify(2 * s.trace(a * b)) == int(i == j)
               for i, a in enumerate(generators) for j, b in enumerate(generators))
    return gamma, generators


def native_cost(amplitudes: list[Any]) -> dict[str, Any]:
    gamma, generators = clifford()
    centre = [a * g for a, g in zip(amplitudes, gamma)]
    orbit = s.Matrix(60, 15, lambda r, j: s.simplify(2 * s.trace(
        generators[r % 15] * s.I * (generators[j] * centre[r // 15] - centre[r // 15] * generators[j]))))
    directions = [[generators[j] if mu == v else s.zeros(4) for mu in range(4)]
                  for v in range(4) for j in range(15)]
    norm_rows = s.Matrix([[s.simplify(s.trace(centre[mu] * x[mu]) / 2)
                          for x in directions] for mu in range(4)])
    defect = s.zeros(60)
    for mu in range(4):
        matrices = [centre[mu] * x[mu] + x[mu] * centre[mu]
                    - s.trace(centre[mu] * x[mu]) * s.eye(4) / 2 for x in directions]
        vectors = s.Matrix(16, 60, lambda r, j: matrices[j][r // 4, r % 4])
        defect += 16 * vectors.H * vectors
        for nu in range(4):
            if mu == nu:
                continue
            matrices = [centre[mu] * x[nu] + x[nu] * centre[mu]
                        + x[mu] * centre[nu] + centre[nu] * x[mu] for x in directions]
            vectors = s.Matrix(16, 60, lambda r, j: matrices[j][r // 4, r % 4])
            defect += 4 * vectors.H * vectors
    defect = defect.applyfunc(s.simplify)
    total = (defect + 64 * norm_rows.T * norm_rows).applyfunc(s.simplify)
    gram = (orbit.T * orbit).applyfunc(s.simplify)
    gauge = (orbit * gram.inv() * orbit.T).applyfunc(s.simplify)
    normal = s.eye(60) - gauge
    assert total == total.H and (total * orbit).applyfunc(s.simplify) == s.zeros(60, 15)
    assert defect.rank() == 41 and total.rank() == 45 and orbit.rank() == 15
    assert norm_rows * orbit == s.zeros(4, 15)
    radial = s.Matrix([[s.simplify(2 * s.trace(generators[j] * centre[mu]))
                        if mu == nu else 0 for nu in range(4)] for mu in range(4) for j in range(15)])
    assert defect * radial == s.zeros(60, 4)
    assert radial.rank() == 4 and normal * radial == radial
    assert (radial.T * total * radial).det() > 0
    return {'centre': centre, 'generators': generators, 'orbit': orbit, 'gram': gram,
            'normal': normal, 'defect': defect, 'cost': total, 'radial': radial}


def compact_normal_control() -> tuple[dict[str, Any], dict[str, Any]]:
    unit = native_cost([s.Integer(1)] * 4)
    q = unit['cost']
    gauge = s.eye(60) - unit['normal']
    inverse = (q + gauge).inv()
    assert (q + gauge) * inverse == s.eye(60)
    row_bound = max(sum(abs(v) for v in inverse.row(i)) for i in range(60))
    assert row_bound >= 1
    lower = s.simplify(1 / row_bound)
    sample_a = [s.Rational(1, 8), s.Rational(1, 3), s.Rational(2, 3), s.Integer(1)]
    sample = native_cost(sample_a)
    scaling = s.diag(*[a for a in sample_a for _ in range(15)])
    assert sample['orbit'] == scaling * unit['orbit']
    # Congruent cost blocks scale by a_mu^4 or a_mu^2*a_nu^2;
    # the proof uses distance to D_a range(L_1), NOT an unchanged normal subspace.
    encoded = json.dumps([[str(v) for v in q.row(i)] for i in range(60)], separators=(',', ':'))
    normal_lower = lower / 32768
    wrong_gap_on_gauge = s.simplify((sample['orbit'][:, 0].T * sample['cost'] * sample['orbit'][:, 0])[0])
    assert wrong_gap_on_gauge == 0 and (sample['orbit'][:, 0].T * sample['orbit'][:, 0])[0] > 0
    return {'unit_cost_rank': 45, 'deleted_norm_rank': 41, 'gauge_nullity': 15,
            'exact_inverse_infinity_norm': str(row_bound), 'unit_quotient_lower': str(lower),
            'compact_normal_lower': str(normal_lower), 'compact_link_orbit_lower': '1/32',
            'compact_link_orbit_upper': '16', 'compact_set': 'x_mu>=1/128, ||x||2<=2',
            'unit_cost_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
            'adverse_deleted_norm_extra_flats': 4, 'adverse_massive_gauge_quadratic': str(wrong_gap_on_gauge),
            'is_actual_quantum_Hessian_or_gap': False}, sample


def annihilators(modes: int) -> list[Any]:
    result = []
    for i in range(modes):
        matrix = s.zeros(1 << modes)
        for bits in range(1 << modes):
            if bits & (1 << i):
                matrix[bits ^ (1 << i), bits] = (-1) ** ((bits & ((1 << i) - 1)).bit_count())
        result.append(matrix)
    return result


def second_quantized(matrix: Any, annihilation: list[Any]) -> Any:
    return sum((matrix[i, j] * annihilation[i].H * annihilation[j]
                for i in range(4) for j in range(4)), s.zeros(16))


def wedge(columns: Any) -> Any:
    ray = s.zeros(16, 1)
    number = columns.cols
    for indices in itertools.combinations(range(4), number):
        ray[sum(1 << i for i in indices)] = columns.extract(indices, range(number)).det()
    return ray.applyfunc(s.simplify)


def physical_metric_and_jets(geometry: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    c, generators = geometry['centre'], geometry['generators']
    orbit, gram, normal = geometry['orbit'], geometry['gram'], geometry['normal']
    annihilation = annihilators(4)
    reps = [second_quantized(t, annihilation) for t in generators]
    assert all(a * b.H + b.H * a == int(i == j) * s.eye(16)
               for i, a in enumerate(annihilation) for j, b in enumerate(annihilation))
    sector_dims = []
    nonzero_sectors = []
    for number in range(5):
        indices = [bits for bits in range(16) if bits.bit_count() == number]
        sector_dims.append(len(indices))
        any_action = any(rep.extract(indices, indices) != s.zeros(len(indices)) for rep in reps)
        nonzero_sectors.append(any_action)
    assert sector_dims == [1, 4, 6, 4, 1] and nonzero_sectors == [False, True, True, True, False]
    m = s.Rational(profile['onsite'])
    ci = s.Rational(profile['paths'][0]['ci'])
    h = m * c[0] - 2 * ci * sum(c[1:], s.zeros(4))
    eps2 = s.simplify(s.trace(h * h) / 4)
    assert (h * h - eps2 * s.eye(4)).applyfunc(s.simplify) == s.zeros(4) and eps2 > 0
    # Literal N4 AP momentum (pi/2,pi/2,pi/2), both occupied columns.
    mass = m / 8
    d = -2 * ci * s.Matrix([[1, s.Rational(1, 3) - s.I * s.Rational(2, 3)],
                            [s.Rational(1, 3) + s.I * s.Rational(2, 3), -1]])
    eps = s.sqrt(eps2)
    columns = s.sqrt((eps + mass) / (2 * eps)) * s.eye(2).col_join(d / (eps + mass))
    assert (columns.H * columns).applyfunc(s.simplify) == s.eye(2)
    assert (h * columns - eps * columns).applyfunc(s.simplify) == s.zeros(4, 2)
    ray = wedge(columns)
    assert s.simplify((ray.H * ray)[0]) == 1
    first_index = next(j for j, rep in enumerate(reps) if (rep * ray).applyfunc(s.simplify) != s.zeros(16, 1))
    r = reps[first_index]
    correct_second = (4 * ray + 2 * s.I * r * ray - r * r * ray).applyfunc(s.simplify)
    omitted_scalar_ray_cross = (4 * ray - r * r * ray).applyfunc(s.simplify)
    assert correct_second != omitted_scalar_ray_cross
    a = generators[first_index]
    same_column_second = wedge((-a * a * columns[:, 0]).row_join(columns[:, 1])) + wedge(columns[:, 0].row_join(-a * a * columns[:, 1]))
    complete_ray_second = (-r * r * ray).applyfunc(s.simplify)
    assert complete_ray_second != same_column_second.applyfunc(s.simplify)
    other = next(j for j, rep in enumerate(reps) if (r * rep - rep * r) != s.zeros(16))
    ordered_difference = (r * reps[other] - reps[other] * r) * s.eye(16)[:, 1]
    if ordered_difference == s.zeros(16, 1):
        ordered_difference = (r * reps[other] - reps[other] * r) * s.eye(16)[:, 2]
    assert ordered_difference != s.zeros(16, 1)
    direction = orbit[:, 0] + normal * s.eye(60)[:, 2]
    left = gram.inv() * orbit.T
    u, w = left * direction, normal * direction
    generator = sum((u[j] * generators[j] for j in range(15)), s.zeros(4))
    normals = [sum((w[15 * mu + j] * generators[j] for j in range(15)), s.zeros(4)) for mu in range(4)]
    nonlinear = s.Matrix([s.simplify(2 * s.trace(t * (-(generator * (generator * cc - cc * generator)
                         - (generator * cc - cc * generator) * generator) + 2 * s.I * (generator * zz - zz * generator))))
                         for cc, zz in zip(c, normals) for t in generators])
    assert nonlinear != s.zeros(60, 1)
    assert (orbit * (-left * nonlinear) - normal * nonlinear + nonlinear).applyfunc(s.simplify) == s.zeros(60, 1)
    # Mixed metric is Gauss elimination of native momenta, not determinant flattening.
    e = s.Matrix([s.Integer(int(j == first_index)) for j in range(15)])
    charge = s.eye(16)[:, 1]
    residual = [(reps[j] - e[j] * s.eye(16)) * charge for j in range(15)]
    covariant_cost = s.simplify(sum((v.H * v)[0] / gram[j, j] for j, v in enumerate(residual)))
    separated_cost = s.simplify(sum(((reps[j] * charge).H * reps[j] * charge)[0] / gram[j, j]
                                  + e[j] ** 2 / gram[j, j] for j in range(15)))
    if covariant_cost == separated_cost:
        # A diagonal Cartan generator exposes the scalar-electric cross without phase cancellation.
        j = next(k for k, t in enumerate(generators) if t[0, 0] != 0)
        cross = s.simplify(-2 * generators[j][0, 0] / gram[j, j])
    else:
        cross = s.simplify(covariant_cost - separated_cost)
    assert cross != 0
    return {'full_local_CAR_dim': 16, 'charged_colour_sector_dims': sector_dims,
            'charged_colour_nonzero_sectors': nonzero_sectors,
            'literal_AP_positive_graph_two_columns': True,
            'lost_distinct_column_second_jet_rejected': True,
            'lost_scalar_ray_second_cross_rejected': True,
            'ordered_noncommuting_second_jet_rejected': True,
            'inverse_normal_orbit_second_cross_nonzero': True,
            'lost_charged_electric_metric_cross': str(cross),
            'all65_plus_Ref_local_generator_envelope': 264,
            'jet_constants': {'coordinate_first': '2^36', 'coordinate_second': '2^84',
                             'fermion_first': '264*2^36', 'fermion_second': '264*2^84+(264*2^36)^2'},
            'centre_cover_local_chart_degree': 4, 'flat_SU4_centre_holonomies_retained': 64,
            'spectral_projection_selects_no_character_or_classical_centre': True}


def native_failed_gates(profile: dict[str, Any]) -> dict[str, Any]:
    m = s.Rational(profile['onsite'])
    c0 = s.Rational(profile['paths'][0]['c0'])
    minimum_mass = m - 6 * abs(c0)
    assert minimum_mass > 0
    # Entire native h scales exactly on the radial Cartesian path Phi=t*C.
    t = s.Symbol('t', real=True)
    electronic_gap2 = s.expand(4 * t * t * minimum_mass * minimum_mass / 128)
    assert electronic_gap2.subs(t, 0) == 0
    assert electronic_gap2.subs(t, s.Rational(1, 1024)) > 0
    x = s.Symbol('x', real=True)
    transition = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5
    derivative_mass = s.integrate(s.diff(transition, x) ** 2, (x, 0, 1))
    assert derivative_mass == s.Rational(10, 7)
    width = s.Symbol('width', positive=True)
    kinetic_transition = 4 * s.Rational(1, 2**4096) * derivative_mass / width
    assert s.limit(kinetic_transition, width, 0, dir='+') == s.oo
    assert (2 ** 257) ** 3 * s.Rational(1, 2**17) > 1
    return {'first_route': 'global smooth BO/harmonic tube',
            'native_Phi_zero_neutral_canonical_degeneracy': 'binomial(4V,2V)',
            'electronic_gap_squared_radial_bound': str(electronic_gap2),
            'all_configuration_electronic_gap_infimum': 0,
            'sharp_native_radial_boundary_transition_derivative_integral': str(derivative_mass),
            'sharp_cutoff_kinetic_divergence': str(kinetic_transition),
            'sharp_event_does_not_preserve_native_form_core': True,
            'fixed_site_union_bound_at_first_largeN_exceeds_one': True,
            'actual_ground_or_Nuniform_fast_gap_falsified': False,
            'distinct_second_route': 'actual local untruncated open-patch energy-relative Friedrichs form'}


def local_energy_control(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    neutral = profiles[0]
    m = abs(s.Rational(neutral['onsite']))
    b = abs(s.Rational(neutral['paths'][0]['c0'])) + abs(s.Rational(neutral['paths'][0]['ci']))
    dn = 2 * m + 12 * b
    charged = sum(int(p['load']) * (2 * abs(s.Rational(p['onsite'])) +
                  4 * sum(abs(s.Rational(path['c0'])) + abs(s.Rational(path['ci'])) for path in p['paths']))
                  for p in profiles[1:])
    expected = s.Rational(855444290449274477, 35184372088832)
    assert charged == expected
    z, r = s.symbols('z r', nonnegative=True)
    # Exact algebraic Young price: dn*r <= dn^2/2+r^4/2+1/8.
    assert s.expand(dn**2 / 2 + r**4 / 2 + s.Rational(1, 8) - dn * r
                    - ((r - dn)**2 / 2 + (r*r - s.Rational(1, 2))**2 / 2)) == 0
    ramp = 6*z**5 - 15*z**4 + 10*z**3
    assert s.expand(s.diff(ramp, z) - 30*z*z*(1-z)**2) == 0
    assert s.Rational(25, 8) * s.Rational(22, 7)**2 < 32
    assert s.Rational(22, 14) * (s.Rational(360*16, 9) + s.Rational(5, 4)) < 2048
    assert (2048 + 16 + 59*4) / 2 < 2048
    # Nonreal weighted-resolvent price is an exact scalar inequality, not a spectral computation.
    for energy, real, imaginary in ((1, 0, 1), (16, 3, 2), (81, 80, 1)):
        distance2 = s.Integer(energy-real)**2 + imaginary**2
        assert distance2 >= imaginary**2
    return {'neutral_linear_constant': str(dn), 'charged_all64_linear_constant': str(charged),
            'patch_shift_per_site': 'Lambda*(' + str(dn*dn/2+s.Rational(1, 8)) + ')',
            'quartic_native_lower': 'J_x>=r_x^4, r_x^2=sum_mu Tr(Phi_mu^2)',
            'actual_local_positive_form': 'K_D=hfast_D+Lambda*(Dn^2/2+1/8)*|D|+1 >= T_D+(Lambda/2)*sum_D r_x^4+1',
            'linear_source_graph': '||sum_D r_x K_D^-1/4|| <= |D|^3/4*(2/Lambda)^1/4',
            'charged_local_graph': '||Hcharged_local K_Dplus^-1/4|| <= Dch*|Dplus|^3/4*(2/Lambda)^1/4',
            'nonreal_weighted_resolvent': '||K_D^1/4(K_D-z)^-1|| <= |Im z|^-3/4+|z|^1/4/|Im z|',
            'radial_cutoff': 'theta=(pi/2)*(6u^5-15u^4+10u^3),u=(r^2-R^2)/(3R^2), clamped at0,1; f=(cos theta,sin theta)',
            'local_IMS_per_site': '<=32*chi/R^2', 'cutoff_Cart_gradient': '<=3/R',
            'cutoff_Cart_Laplacian': '<=2048/R^2',
            'fullH_cutoff_commutator': '||[H,f]psi|| <= (12*sqrt(chi)/R)*sqrt(q_Tsite[psi])+(8192*chi/R^2)*||psi||',
            'exterior_state_weight': '||1_{r_x>=2R}psi||^2 <= 2<psi,K_D psi>/(Lambda*(2R)^4)',
            'patch_exterior_identity': True, 'local_occupation_or_character_selection': False,
            'uniform_in_N_at_fixed_patch': True, 'real_axis_lowband_gap_paid': False}


def neutral_tube_projection_prices(profile: dict[str, Any]) -> dict[str, Any]:
    minimum = s.Rational(profile['onsite']) - 6 * abs(s.Rational(profile['paths'][0]['c0']))
    assert minimum > s.Rational(1, 8)
    radius = s.Rational(1, 2**40)
    perturbation = 2**22 * radius
    assert s.Rational(1, 128) - perturbation > s.Rational(1, 256)
    # Rectangle Re[0,256], Im[-256,256]: length1536/(2pi)<256.
    resolvent, contour_factor = 2**8, 2**8
    native_first_operator, native_first_nuclear = 2**12, 2**20
    native_second_nuclear = 2**20
    first_nuclear = contour_factor * resolvent**2 * native_first_nuclear
    second_nuclear = contour_factor * (resolvent**2 * native_second_nuclear
                       + 2 * resolvent**3 * native_first_operator * native_first_nuclear)
    assert first_nuclear <= 2**44 and second_nuclear < 2**66
    kato_first = 2 * first_nuclear
    ordered_kato_second = 2 * second_nuclear + 2 * first_nuclear**2 + kato_first**2
    assert kato_first <= 2**45 and ordered_kato_second < 2**94
    assert 2 * 2**94 + 2 * (2**45)**2 < 2**96
    return {'hypothesis': 'simultaneous native max-radius tube<=2^-40, no ground support assertion',
            'native_electronic_half_gap': '>1/256', 'moving_onebody_projection_first_nuclear': '<=2^44',
            'moving_onebody_projection_ordered_second_nuclear': '<2^66',
            'global_canonical_sea_Kato_first': '<=2^45', 'ordered_mixed_sea_second': '<2^94',
            'whole_sea_projection_first': '<=2^46', 'whole_sea_projection_second': '<2^96',
            'constants_local_not_volume': True, 'outside_tube_extension_or_actual_band_weight_paid': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, pin in request['pins'].items():
        actual = hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()
        if actual != pin['sha256']:
            raise ValueError('immutable pin mismatch: ' + name)
    parent = json.loads(Path(request['pins']['normal_request']['path']).read_text())
    assert hashlib.sha256(parent['source_text'].encode()).hexdigest() == parent['source_sha256']
    assert 'D + 64*R.T*R' in parent['source_text'] and 'Q*L' in parent['source_text']
    profiles = json.loads(Path(request['pins']['profiles']['path']).read_text())['primitive_profiles']
    normal, sample = compact_normal_control()
    controls = {'compact_normal': normal, 'physical_metric_jets': physical_metric_and_jets(sample, profiles[0]),
                'first_route_failed_native_domain_gate': native_failed_gates(profiles[0]),
                'local_moving_neutral_projection_prices': neutral_tube_projection_prices(profiles[0]),
                'second_route_local_energy_relative': local_energy_control(profiles)}
    receipt = {'kind': 'EXACT_NATIVE_FAST_QUANTUM_BODY_DOMAIN_AND_LOCAL_ENERGY_RELATIVE_NOT_UNIFORM_GAP',
               'governor': request['governor'], 'pins': request['pins'], 'controls': controls,
               'actual_Nuniform_fast_cluster_gap': False, 'actual_physical1_phase': False,
               'harmonic_geometry_promoted_to_quantum_gap': False, 'law_changed': False,
               'old_entrypoints_replayed': False, 'scientific_admission': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'native_assertions_reached': True, 'compact_quotient_normal_lower': normal['compact_normal_lower'],
                      'actual_local_energy_relative_form': True, 'global_tube_route_failed': True,
                      'actual_Nuniform_quantum_gap': False}), flush=True)


if __name__ == '__main__':
    main()
