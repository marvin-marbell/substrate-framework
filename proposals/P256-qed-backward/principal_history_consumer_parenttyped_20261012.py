"""New all-word native principal-flux costs and same-history spectral controls.
No old producer imports, native ground numerics, or independent feature generator.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact(x: F) -> dict[str, Any]:
    return {'sha256': digest(str(x).encode()), 'numerator_bits': x.numerator.bit_length(),
            'denominator_bits': x.denominator.bit_length(),
            'value': str(x) if max(x.numerator.bit_length(), x.denominator.bit_length()) < 512 else None}


def face_geometry(n: int) -> dict[str, Any]:
    def shift(x: tuple[int, int, int], axis: int, steps: int = 1) -> tuple[int, int, int]:
        y = list(x); y[axis] = (y[axis] + steps) % n
        return tuple(y)  # type: ignore[return-value]
    centre = (0, 0, 0)
    rows = []
    for j, k in ((0, 1), (0, 2), (1, 2)):
        c = {(centre, j): 1, (shift(centre, j), k): 1,
             (shift(centre, k), j): -1, (centre, k): -1}
        divergence: dict[tuple[int, int, int], int] = {}
        for (x, axis), value in c.items():
            divergence[x] = divergence.get(x, 0) + value
            y = shift(x, axis); divergence[y] = divergence.get(y, 0) - value
        assert all(value == 0 for value in divergence.values())
        assert sum(value * value for value in c.values()) == 4
        # Rational upper bound to theta^-1=2^(distance/4), not a diameter bound.
        inv = sum((F(2**math.ceil(sum(min(z, n-z) for z in x)/4))
                   for x, _ in c), F(0))
        assert inv == 6
        # Full native length1/2 charged path phase, including cancellations.
        path_counts = []
        for length in (1, 2):
            counts: dict[int, int] = {}
            for x in itertools.product(range(n), repeat=3):
                for axis in range(3):
                    coefficient = sum(c.get((shift(cast(tuple[int, int, int], x), axis, t), axis), 0) for t in range(length))
                    counts[coefficient] = counts.get(coefficient, 0) + 1
            assert sum(counts.values()) == 3*n**3
            assert counts.get(1, 0) == counts.get(-1, 0)
            path_counts.append({'length': length, 'phase_derivative_histogram': counts,
                                'AP_sign_modulus': 1})
        rows.append({'axes': [j, k], 'Dc_zero': True, 'c_squared': 4,
                     'c_dot_c_over4': '1', 'face_weight_inverse_sum_upper': str(inv),
                     'p_B_second_moment_per_Q_upper': str(inv/1024),
                     'whole_native_path_phase_counts': path_counts})
    return {'N': n, 'faces': rows, 'geometry_only_not_ground': True}


def constants(profiles: list[dict[str, Any]]) -> tuple[F, F, F]:
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    assert [p['load'] for p in profiles] == [1, 1, 63]
    w = [abs(F(p['onsite'])) + 2*sum((abs(F(r['c0']))+abs(F(r['ci']))
                                     for r in p['paths']), F(0)) for p in profiles]
    b = 4*(lam*w[0]+w[1]+63*w[2])
    assert b < 512*lam
    jet = 2**16*(1+fast+b)
    cstar = 2**32*65*(1+jet**2)*(4096+2/lam)
    r = 60*chi+11192*lam+F(129, 10)*fast+262144
    b0 = 33*(1+10*r*4096)+16384*lam*4096
    return b, cstar, b0


def paid_words(profiles: list[dict[str, Any]], words: list[list[dict[str, Any]]],
               ells: list[int]) -> dict[str, Any]:
    b, cstar, b0 = constants(profiles)
    rows = []
    for word in words:
        norm, energy, graph2 = F(1), b0, F(0)
        costs = []
        for letter in word:
            # One actual finite propagation segment dt=scaled_dt/Cstar.
            dt = F(letter['scaled_dt'])/cstar
            power = math.ceil(2*abs(F(letter['scaled_dt'])))
            energy *= 2**power  # exp(z)<=2^ceil(2z), z>=0.
            k = int(letter['phase_power']); a = abs(F(letter.get('amplitude', '1')))
            assert a > 0 and letter['closed_face_boost']
            # M=a U^k, W_h closed face boost. Haar/Phi jets vanish.
            d = 256*a*a*k*k
            # Native support extends through actual length2 matter paths;
            # m<=108, radius<=4, tau^-1<=2. All65 B remains in propagation.
            c = a*(256*k*k+64*abs(k))+4*a*b*108**2
            old_norm = norm
            graph2 = 2*a*a*graph2+2*c*c*energy
            energy = 66*a*a*energy+(66*d+16384*F(2**8192)*4096*a*a)*old_norm**2
            norm *= a
            costs.append({'k': k, 'a': str(a), 'scaled_dt': letter['scaled_dt'],
                          'actual_dt': exact(dt), 'exp_Cstar_dt_upper': 2**power,
                          'fullE_weighted_jet_cost': exact(d), 'commutator_upper': exact(c),
                          'support_sites_upper': 108, 'centre_radius_upper': 4})
        energy *= 2  # Pay an explicit leading propagation with Cstar|dt|<=1/4.
        moment = energy*F(6, 1024)
        root_upper = F(math.isqrt(moment.numerator//moment.denominator)+1)
        assert root_upper*root_upper >= moment
        rho = norm*root_upper/F(2**4100)
        arc_rows = []
        previous = None
        for ell in ells:
            m = 2**ell
            a_arc = F(1, 2**(2*ell+4100)); delta = a_arc/2
            r = 1-delta/m
            arc = norm**2/m+rho*F(m-1, m*m)
            error2 = 64*norm**2/F(m*m)+16*arc
            assert F(1, 2) < r < 1 and m*a_arc < 1
            assert float(r) == 1.0  # Binary64 radius is an invalid consumer.
            if previous is not None:
                assert error2 < previous
            previous = error2
            arc_rows.append({'ell': ell, 'same_radius_one_minus_r': exact(1-r),
                             'unnormalized_arc_upper': exact(arc), 'all_word_L2_error2_upper': exact(error2),
                             'first_harmonic': '1', 'binary64_radius_rejected': True})
        rows.append({'word': word, 'norm_upper': exact(norm), 'Q_upper': exact(energy),
                     'graph2_upper': exact(graph2), 'p_B_moment_upper': exact(moment),
                     'rotation_difference_slope_upper': exact(root_upper), 'costs': costs,
                     'arc_and_principal_sequence': arc_rows})
    # Smooth bounded native general MW letters have the accepted recursion too;
    # these face/boost controls do not replace its arbitrary-word analytic proof.
    return {'all65_loads': [1, 1, 63], 'B': exact(b), 'B0': exact(b0), 'Cstar': exact(cstar),
            'words': rows, 'new_native_cost_controls_not_fullH_evolution': True}


def noncommuting_discriminator() -> dict[str, Any]:
    phi: Any = sp.Symbol('phi', real=True)
    f: Any = phi/2
    integrate = lambda value: sp.simplify(sp.integrate(value, (phi, -sp.pi, sp.pi))/(2*sp.pi))
    f2 = integrate(f*f); f4 = integrate(f**4)
    assert f2 == sp.pi**2/12 and f4 == sp.pi**4/80
    # Rank-two selfadjoint source-only surrogate B=|F1><1|+|1><F1|.
    # It has EXACTLY B1=F1, but fails on the native phase word U1.
    fu_mean = integrate(f*sp.exp(sp.I*phi))
    assert fu_mean == sp.I/2
    defect = sp.simplify(f2-abs(fu_mean)**2)
    assert defect == sp.pi**2/12-sp.Rational(1, 4)
    assert sp.N(defect) > 0
    product_defect = sp.simplify(f4-f2*f2)
    assert product_defect == sp.pi**4/180
    # R_pi is translation in phase. It and U anticommute; F is periodic,
    # and its commutator with R_pi on1 has norm² pi²/4, not zero.
    shifted: Any = sp.Piecewise(((phi+sp.pi)/2, phi < 0), ((phi-sp.pi)/2, True))
    commutator2 = integrate((f-shifted)**2)
    assert commutator2 == sp.pi**2/4
    r: Any = sp.Symbol('r', positive=True)
    first = (-1)**2*r**0/sp.Integer(1)
    assert first == 1
    return {'scope': 'exact regular circle Weyl representation METHOD discriminator, not native minimizing ground',
            'dense_word': 'R_pi U 1=-U 1; U R_pi 1=U 1',
            'source_vector_surrogate_matches_on_vacuum': True,
            'surrogate_whole_word_defect_squared': str(defect),
            'surrogate_product_defect_squared': str(product_defect),
            'principal_boost_commutator_squared': str(commutator2),
            'selected_feature_as_whole_operator_rejected': True,
            'raw_additive_principal_rule_rejected': True, 'fixed_first_harmonic': str(first)}


def spectral_controls(epsilon: F, volumes: list[F]) -> dict[str, Any]:
    assert epsilon > 0
    # Exact positive joint MATRIX atoms, only a calculus discriminator;
    # NOT a second generator used by the native proof.
    energies = [F(0), F(0), F(1), F(4)]
    vectors = [[F(1), F(2)], [F(2), F(-1)], [F(1), F(3)], [F(2), F(1)]]
    filtered = [[F(0), F(0)], [F(0), F(0)]]
    graph = [[F(0), F(0)], [F(0), F(0)]]
    cross = [[F(0), F(0)], [F(0), F(0)]]
    mass = [[F(0), F(0)], [F(0), F(0)]]
    for energy, v in zip(energies, vectors):
        if energy == 0:
            continue
        s = epsilon/(epsilon+energy)
        for a in range(2):
            for b in range(2):
                weight = v[a]*v[b]
                mass[a][b] += weight; filtered[a][b] += s*s*weight
                graph[a][b] += energy**2*s*s*weight
                cross[a][b] += energy*s*s*weight
    assert filtered[0][0]*filtered[1][1] >= filtered[0][1]**2
    for a in range(2):
        assert graph[a][a] <= epsilon**2*mass[a][a]
    # Mean subtraction kills only first zero vector, not the second.
    assert sum(vectors[1][a]**2 for a in range(2)) == 5
    q: Any = sp.Symbol('q', positive=True)
    energy_symbol: Any = sp.Symbol('E', positive=True)
    density: Any = (sp.Rational(epsilon)/(sp.Rational(epsilon)+energy_symbol))**2
    continuous_interval = sp.integrate(density, (energy_symbol, q, 2*q))
    assert sp.limit(continuous_interval, q, 0) == 0
    gates = []
    for volume in volumes:
        assert 0 < volume <= 1
        gates.append({'normalized_BZ_annulus_volume': str(volume),
                      'covariance_lower': '0', 'covariance_upper_conservative_pi_over2_lt2': str(4/volume),
                      'cone_weight_lower': '0', 'cone_weight_upper_conservative': str(4/volume),
                      'atom_residue_lower': '0', 'actual_native_covariance_measured': False})
    return {'scope': 'exact calculus METHOD matrix plus certified NATIVE budget envelopes, not native spectral data',
            'epsilon': str(epsilon), 'filtered_joint_matrix': [[str(x) for x in row] for row in filtered],
            'graph_joint_matrix': [[str(x) for x in row] for row in graph],
            'insertion_graph_cross_matrix': [[str(x) for x in row] for row in cross],
            'whole_zero_block_not_mean_subtraction_rejected': True,
            'collapse_example': 'L_n=diag(0,1/n,1): Q_n e2=e2, limit Q e2=0',
            'finite_Q_as_limit_Q_rejected': True,
            'continuous_filtered_interval_mass': str(continuous_interval),
            'continuous_measure_cannot_gain_atom_by_filter': True,
            'source_weakening': 'g(q)=q multiplies covariance and any residue by q²; first harmonic becomes q, not1',
            'source_weakening_rejected': True, 'annulus_gate_envelopes': gates,
            'uniform_positive_low_q_weight_or_atom_certified': False,
            'BZ_average_not_pointwise': 'constant spatial vector: one mode V, normalized BZ average1'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        profile_raw = archive.read(request['profile_member'])
        profiles = json.loads(profile_raw)['primitive_profiles']
    receipt = {'schema': 1, 'kind': 'NEW_ALL_WORD_PRINCIPAL_HISTORY_NATIVE_COST_AND_SPECTRAL_CONSUMER',
               'native_face_geometry': [face_geometry(n) for n in request['N']],
               'paid_native_words': paid_words(profiles, request['words'], request['ells']),
               'noncommuting_history_word_discriminator': noncommuting_discriminator(),
               'same_generator_spectral_calculus_and_IR_gate': spectral_controls(F(request['epsilon']),
                   [F(v) for v in request['annulus_volumes']]),
               'literal_analytic_proof': json.loads((ROOT/request['proof']).read_text()),
               'profile_member_sha256': digest(profile_raw),
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: digest((ROOT/name).read_bytes()) for name in request['pins']},
               'numerical_native_ground_or_fullH_evolution': False,
               'physical1_phase_stiffness_pole_admission': False, 'campaign_complete': False}
    assert before == receipt['source_input_sha256_after']
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'N': request['N'],
                      'native_words': len(request['words']), 'all_word_strong_operator_analytic_conditional': True,
                      'noncommuting_source_only_wrongcase_rejected': True,
                      'low_q_covariance_weight_atom_positive_gate': False}))


if __name__ == '__main__':
    main()
