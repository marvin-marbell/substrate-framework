"""Independent principal-history proof discriminators, not native fullH evolution.
Only reads frozen inputs; never imports or invokes their producers/checkers.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import zipfile
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact(value: F) -> dict[str, Any]:
    raw = str(value)
    return {'value': raw if len(raw) < 512 else None, 'sha256': sha(raw.encode()),
            'numerator_bits': value.numerator.bit_length(),
            'denominator_bits': value.denominator.bit_length()}


def frozen_order_inspection(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'paid_words')
    outer = next(node for node in function.body if isinstance(node, ast.For))
    inner = next(node for node in outer.body if isinstance(node, ast.For))
    assert ast.unparse(inner.target) == 'letter' and ast.unparse(inner.iter) == 'word'
    init = next(node for node in outer.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Tuple) for target in node.targets))
    assert ast.unparse(init.value) == '(F(1), b0, F(0))'
    inner_text = [ast.unparse(node) for node in inner.body]
    positions = {name: next(i for i, text in enumerate(inner_text) if text.startswith(name))
                 for name in ('energy *=', 'old_norm =', 'graph2 =', 'energy =', 'norm *=')}
    assert positions['energy *='] < positions['old_norm ='] < positions['graph2 =']
    assert positions['graph2 ='] < positions['energy ='] < positions['norm *=']
    leading = next(node for node in outer.body if isinstance(node, ast.AugAssign)
                   and isinstance(node.target, ast.Name) and node.target.id == 'energy')
    assert ast.unparse(leading) == 'energy *= 2'
    return {'scope': 'STATIC AST INSPECTION, no frozen function executed',
            'initial_assignment': ast.unparse(init), 'request_order_loop': ast.unparse(inner.target),
            'propagation_precedes_insertion': True, 'old_energy_used_for_graph': True,
            'separately_declared_leading_Q_event': ast.unparse(leading),
            'packet_eta_input_absent': not any(arg.arg == 'eta' for arg in function.args.args),
            'numerical_boost_execution': False,
            'interpretation': 'Processing order, not printed multiplication order; exact-ground scalar costs only.'}


def native_budget_and_universal_arcs(profiles: list[dict[str, Any]], request: dict[str, Any]) -> dict[str, Any]:
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    assert [p['load'] for p in profiles] == [1, 1, 63]
    allocations = [abs(F(p['onsite'])) + 2*sum((abs(F(path['c0'])) + abs(F(path['ci']))
                    for path in p['paths']), F(0)) for p in profiles]
    b = 4*(lam*allocations[0] + allocations[1] + 63*allocations[2])
    assert b < 512*lam
    jet = 2**16*(1+fast+b)
    cstar = 2**32*65*(1+jet**2)*(4096+2/lam)
    base = 33*(1+10*(60*chi+11192*lam+F(129, 10)*fast+262144)*4096)+16384*lam*4096
    rows = []
    radii: dict[int, F] = {}
    for word in request['ordered_cost_words']:
        times = [F(item['scaled_time']) for item in word]
        assert times
        # Unlike frozen paid_words, this is a complete absolute-time packet timeline.
        events: list[tuple[str, F | dict[str, Any]]] = [('propagator', times[0])]
        for index, item in enumerate(word):
            if index:
                events.append(('propagator', times[index]-times[index-1]))
            events.append(('insertion', item))
        events.append(('propagator', -times[-1]))
        n, energy, graph2 = F(1), base, F(request['packet_eta'])**2
        event_rows = []
        for kind, payload in events:
            if kind == 'propagator':
                assert isinstance(payload, F)
                factor = F(2**math.ceil(2*abs(payload)))
                energy *= factor
                event_rows.append({'event': kind, 'scaled_dt': str(payload),
                                   'actual_dt': exact(payload/cstar), 'Q_factor_upper': str(factor)})
            else:
                assert isinstance(payload, dict)
                amplitude = abs(F(payload['amplitude']))
                k = int(payload['phase_power'])
                d = 256*amplitude**2*k**2
                c = amplitude*(256*k*k+64*abs(k))+4*amplitude*b*108**2
                old_n, old_energy = n, energy
                graph2 = 2*amplitude**2*graph2+2*c**2*old_energy
                energy = 66*amplitude**2*old_energy+(66*d+16384*lam*4096*amplitude**2)*old_n**2
                n *= amplitude
                event_rows.append({'event': kind, 'amplitude': str(amplitude), 'phase_power': k,
                                   'old_Q_paid_before_insertion': True})
        moment = energy*F(6, 1024)
        root_upper = F(math.isqrt(moment.numerator//moment.denominator)+1)
        assert root_upper**2 >= moment
        caps = []
        previous: F | None = None
        for ell in request['ells']:
            m = 2**ell
            delta = F(1, 2**(4101+2*ell))
            r = 1-delta/m
            assert radii.setdefault(ell, r) == r
            arc = n*n/m+n*root_upper*F(m-1, 2**4100*m*m)
            error2 = 64*n*n/F(m*m)+16*arc
            assert F(1, 2) < r < 1 and m*2*delta < 1
            assert previous is None or error2 < previous
            previous = error2
            with localcontext() as context:
                context.prec = 80
                low = Decimal(r.numerator)/Decimal(r.denominator)
            precision = len(str(r.denominator))+10
            with localcontext() as context:
                context.prec = precision
                high = Decimal(r.numerator)/Decimal(r.denominator)
            assert float(r) == 1.0 and low == 1 and high < 1
            caps.append({'ell': ell, 'one_minus_r': exact(1-r), 'arc_mass_upper': exact(arc),
                         'principal_error_squared_upper': exact(error2),
                         'binary64_and_decimal80_round_to_one': True,
                         'decimal_precision_recovering_strict_interior_radius': precision,
                         'first_harmonic': '1'})
        rows.append({'word': word, 'events_in_actual_right_to_left_processing_order': event_rows,
                     'norm_upper': exact(n), 'Q_upper': exact(energy), 'graph_squared_upper': exact(graph2),
                     'face_p_B_moment_upper': exact(moment), 'universal_sequence_caps': caps})
    # Independent exact normalization adverse case in the regular Haar circle.
    # v=16*1 has mass256, derivative moment0, seam arc length1/8 and M=4.
    honest_cap, wrongly_normalized_cap = F(256, 4), F(1, 4)
    actual_mass: Any = sp.Rational(16)/sp.pi
    assert honest_cap == 64 and sp.simplify(actual_mass-wrongly_normalized_cap).is_positive
    scale, n0, root0, m0 = F(3), F(2), F(5), 16
    cap0 = n0**2/m0+n0*root0*F(m0-1, 2**4100*m0*m0)
    cap_scaled = (scale*n0)**2/m0+scale*n0*scale*root0*F(m0-1, 2**4100*m0*m0)
    assert cap_scaled == scale**2*cap0
    return {'scope': 'EXACT inherited all65 native coefficient budgets; new declared scalar packet timelines, NOT native evolution',
            'role_loads': [1, 1, 63], 'B': exact(b), 'B0': exact(base), 'Cstar': exact(cstar),
            'packet_eta_graph_initialization': request['packet_eta'], 'words': rows,
            'null_word_cap': '0', 'unnormalized_circle_arc': {'actual_mass': str(actual_mass),
                'valid_cap': str(honest_cap), 'wrong_mass1_cap': str(wrongly_normalized_cap),
                'wrong_cap_rejected': True, 'homogeneity_under_word_scaling': 'degree2'},
            'all_words_use_identical_radii': True, 'fixed_word_only': True}


def operator_and_seam_controls() -> dict[str, Any]:
    phi: Any = sp.Symbol('phi', real=True)
    f: Any = phi/2
    integrate = lambda expr: sp.simplify(sp.integrate(expr, (phi, -sp.pi, sp.pi))/(2*sp.pi))
    second = integrate(f*f)
    coefficient = integrate(f*sp.exp(2*sp.I*phi))
    assert second == sp.pi**2/12 and coefficient == -sp.I/4
    dense_word_defect = sp.simplify(second-abs(coefficient)**2)
    assert dense_word_defect == sp.pi**2/12-sp.Rational(1, 16)
    # Independent ordered F R_(2pi/3) F product; only polynomial integrals.
    cut: Any = sp.pi/3
    shift_left: Any = (phi+2*sp.pi/3)/2
    shift_right: Any = (phi-4*sp.pi/3)/2
    def shifted_integral(left: Any, right: Any) -> Any:
        return sp.simplify((sp.integrate(left, (phi, -sp.pi, cut))+
                            sp.integrate(right, (phi, cut, sp.pi)))/(2*sp.pi))
    correlation = shifted_integral(f*shift_left, f*shift_right)
    assert correlation == -sp.pi**2/36
    product_defect = shifted_integral((f*shift_left-correlation)**2,
                                     (f*shift_right-correlation)**2)
    assert sp.simplify(product_defect/sp.pi**4) > 0
    commutator_squared = shifted_integral((f-shift_left)**2, (f-shift_right)**2)
    assert commutator_squared == 2*sp.pi**2/9
    # Separate vacuum-only seam method: finite2 has no regular Weyl rotation.
    seam_projection = sp.diag(0, 1)
    vacuum = sp.Matrix([1, 0]); history_letter = sp.Matrix([[0, 1], [1, 0]])
    assert (seam_projection*vacuum).norm() == 0
    assert (seam_projection*history_letter*vacuum).norm() == 1
    return {'scope': 'EXACT regular-circle Weyl METHOD, not native physical ground',
            'source_only_surrogate': 'S=|f><1|+|1><f|; S1=f=F1',
            'dense_word': 'U^2 1', 'surrogate_dense_word_defect_squared': str(dense_word_defect),
            'ordered_product': 'F R_(2pi/3) F 1 versus S R_(2pi/3) S 1',
            'surrogate_product_defect_squared': str(product_defect),
            'principal_boost_commutator_squared': str(commutator_squared),
            'source_only_as_operator_products_rejected': True,
            'raw_additive_principal_rule_rejected': True,
            'vacuum_only_seam': {'scope': 'NONREGULAR finite2 adverse method; not a counterexample to all-word Weyl theorem',
                'vacuum_seam_mass': '0', 'history_word_seam_mass': '1',
                'principal_limit_error_on_history_word_squared': 'pi^2/4',
                'whole_word_seam_payment_required': True}}


def packet_order_control() -> dict[str, Any]:
    eta: Any = sp.Rational(1, 8)
    generator = sp.diag(0, eta, 1)
    qform = sp.Matrix([[2, -1, 0], [-1, 2, 0], [0, 0, 1]])
    packet = sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5), 0])
    ground = sp.Matrix([1, 0, 0])
    a1 = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    a2 = sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    u1 = sp.diag(1, sp.cos(sp.pi/8)-sp.I*sp.sin(sp.pi/8), -1)
    u2 = sp.diag(1, sp.cos(sp.pi/4)-sp.I*sp.sin(sp.pi/4), 1)
    # Canonical cyclotomic arithmetic avoids unnormalized nested-radical zeros.
    field = sp.QQ.algebraic_field(sp.exp(sp.I*sp.pi/8))
    direct = u2.conjugate().T*a2*u2*u1.conjugate().T*a1*u1*packet
    corrected = u2.conjugate().T*a2*u1*a1*u1*packet
    omitted = u2.conjugate().T*a2*u1*a1*packet
    assert all(field.from_sympy(x) == field.zero for x in direct-corrected)
    defect = sp.simplify(((corrected-omitted).conjugate().T*(corrected-omitted))[0])
    assert field.from_sympy(defect-sp.Rational(32, 25)*(1-sp.cos(sp.pi/8))) == field.zero
    assert defect.is_positive
    before = (packet.T*qform*packet)[0]
    after = sp.simplify(((u1*packet).conjugate().T*qform*(u1*packet))[0])
    assert before == sp.Rational(26, 25)
    assert sp.simplify(after-before).is_positive
    assert u1*ground == ground
    full_graph = generator*a1*packet
    comm_only = (generator*a1-a1*generator)*packet
    assert full_graph == comm_only+a1*generator*packet
    assert full_graph != comm_only
    return {'scope': 'EXACT non-native finite3 packet/ordered-matrix METHOD',
            'L': 'diag(0,1/8,1)', 'Q_form': '[[2,-1,0],[-1,2,0],[0,0,1]]',
            'packet': '(3/5,4/5,0)', 'absolute_times': ['pi', '2pi'],
            'corrected_factorization_equals_direct_alpha_word': True,
            'omitted_rightmost_propagator_defect_squared': str(defect),
            'initial_Q': str(before), 'Q_after_rightmost_U_t1': str(after),
            'initial_Q_increased_despite_small_support': True,
            'exact_ground_omission_valid': True, 'AL_packet_graph_term_required': True,
            'first_energy_only_adverse': {'L_excitation': '4096', 'probability': '1/64^3',
                'first_energy': '1/64', 'graph_squared': '64',
                'shrinking_total_support_not_first_energy_required': True}}


def mixed_filter_control(epsilon_a: F, epsilon_b: F) -> dict[str, Any]:
    ea: Any = sp.Rational(epsilon_a); eb: Any = sp.Rational(epsilon_b)
    energies: list[Any] = [sp.Integer(0), sp.Integer(0), sp.Rational(2, 3), sp.Rational(5, 2), sp.Integer(9)]
    vectors: list[list[Any]] = [[1, 2*sp.I, 1+sp.I, 2, -sp.I],
                               [2, -1, 2*sp.I, 1-sp.I, 3]]
    y: list[Any] = [0, 1, 1-2*sp.I, -2, 1+sp.I]
    # Joint T eigencharacters [1,-1,i,-i,-1], x=1 and t=6pi.
    phases: list[Any] = [1, -1, sp.I, sp.I, -1]
    filters_a: list[Any] = [0 if e == 0 else ea/(ea+e) for e in energies]
    filters_b: list[Any] = [0 if e == 0 else eb/(eb+e) for e in energies]
    xa = [s*v for s, v in zip(filters_a, vectors[0])]
    xb = [s*v for s, v in zip(filters_b, vectors[1])]
    qy = [0 if e == 0 else v for e, v in zip(energies, y)]
    atom_matrices = []
    for index in range(2, 5):
        column = sp.Matrix([sp.conjugate(vectors[0][index]), sp.conjugate(vectors[1][index])])
        atom = column*column.conjugate().T
        assert atom == atom.conjugate().T and atom.det() == 0
        z = sp.Matrix([1+sp.I, 2-sp.I])
        quadratic = sp.simplify((z.conjugate().T*atom*z)[0])
        assert quadratic.is_nonnegative
        atom_matrices.append({'E': str(energies[index]), 'matrix': [[str(value) for value in row] for row in atom.tolist()],
                              'positive_quadratic_control': str(quadratic)})
    results = {}
    for power in (0, 1, 2):
        direct = sp.simplify(cast(Any, sum(sp.conjugate(a)*phase*e**power*b
                                for a, phase, e, b in zip(xa, phases, energies, xb))))
        formula = sp.simplify(cast(Any, sum(phase*e**power*sa*sb*sp.conjugate(va)*vb
                      for phase, e, sa, sb, va, vb in zip(phases, energies, filters_a, filters_b, vectors[0], vectors[1]))))
        assert direct == formula
        results[str(power)] = str(direct)
    cross_results = {}
    for power in (0, 1):
        direct = sp.simplify(cast(Any, sum(sp.conjugate(a)*phase*e**power*b
                                for a, phase, e, b in zip(xa, phases, energies, qy))))
        one_filter = sp.simplify(cast(Any, sum(phase*e**power*sa*sp.conjugate(va)*vb
                      for phase, e, sa, va, vb in zip(phases, energies, filters_a, vectors[0], qy))))
        wrong_squared = sp.simplify(cast(Any, sum(phase*e**power*sa**2*sp.conjugate(va)*vb
                      for phase, e, sa, va, vb in zip(phases, energies, filters_a, vectors[0], qy))))
        assert direct == one_filter and sp.simplify(direct-wrong_squared) != 0
        cross_results[str(power)] = {'correct_one_filter': str(direct), 'wrong_squared_filter': str(wrong_squared)}
    norm_a = sum(abs(v)**2 for v in vectors[0]); norm_b = sum(abs(v)**2 for v in vectors[1])
    qnorm_a = sum(abs(vectors[0][i])**2 for i in range(2, 5))
    graph_a = sp.simplify(cast(Any, sum(e**2*abs(v)**2 for e, v in zip(energies, xa))))
    assert graph_a <= ea**2*qnorm_a
    rankone_residual = abs(vectors[0][1])**2
    assert rankone_residual == 4 and xa[0] == xa[1] == 0
    wrong_same_filter = sp.simplify(cast(Any, sum(phase*sa**2*sp.conjugate(va)*vb
                     for phase, sa, va, vb in zip(phases, filters_a, vectors[0], vectors[1]))))
    assert sp.simplify(sp.sympify(results['0'])-wrong_same_filter) != 0
    collapse = []
    for n in (32, 1024):
        collapsed_filter = ea/(ea+sp.Rational(1, n))
        collapse.append({'n': n, 'finite_Q_collapsed_source_filtered_mass': str(4*collapsed_filter**2),
                         'true_limiting_Q_collapsed_source_mass': '0'})
    return {'scope': 'EXACT complex positive joint finite5 MATRIX METHOD; not native spectral data',
            'epsilon_a': str(epsilon_a), 'epsilon_b': str(epsilon_b), 'whole_zero_dimension': 2,
            'positive_matrix_atoms': atom_matrices, 'x': 1, 't': '6pi',
            'mixed_two_filter_insertion_one_graph_two_graph_kernels': results,
            'filtered_principal_against_unfiltered_y_cross_kernels': cross_results,
            'wrong_one_filter_or_squared_filter_rejected': True,
            'different_epsilon_requires_s_a_s_b_not_s_a_squared': True,
            'mixed_total_variation_Cauchy_bound': str(sp.sqrt(norm_a*norm_b)),
            'principal_graph_squared': str(graph_a), 'graph_squared_bound': str(ea**2*qnorm_a),
            'mean_subtraction_residual_zero_energy_mass': str(rankone_residual),
            'whole_limiting_P0_removes_both_zero_directions': True,
            'finite_positive_gap_collapse': collapse,
            'finite_Q_N_as_limit_Q_rejected': True,
            'unfiltered_domain_adverse': 'l2 v_n=1/n, L_n=n: sum |v_n|^2=pi^2/6 but sum |L v_n|^2=infinity; filtered graph <=epsilon^2*pi^2/6',
            'positive_IR_atom_residue_or_static_promotion': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    before = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        profile_raw = archive.read(request['profile_member'])
    profiles = json.loads(profile_raw)['primitive_profiles']
    receipt = {'schema': 1, 'kind': 'NEW_INDEPENDENT_PRINCIPAL_HISTORY_LITERAL_REVIEW_AND_DISCRIMINATORS',
               'frozen_paid_words_AST': frozen_order_inspection((ROOT/request['frozen_consumer']).read_text()),
               'native_budget_and_all_word_arc_controls': native_budget_and_universal_arcs(profiles, request),
               'source_operator_ordered_products_and_seam': operator_and_seam_controls(),
               'retained_rightmost_packet_propagator': packet_order_control(),
               'same_L_whole_zero_and_complex_mixed_filters': mixed_filter_control(F(request['epsilon_a']), F(request['epsilon_b'])),
               'complete_literal_review': json.loads((ROOT/request['review_proof']).read_text()),
               'profile_member_sha256': sha(profile_raw), 'source_input_sha256_before': before,
               'source_input_sha256_after': {name: sha((ROOT/name).read_bytes()) for name in request['pins']},
               'request_sha256': sha(request_raw), 'old_producer_root_solver_checker_run': False,
               'native_ground_or_fullH_numerics': False, 'scientific_admission': False,
               'static_IR_atom_physical1_promotion': False, 'campaign_complete': False}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'independent_controls_completed': True,
                      'entire_history_theorem_scope': 'CONDITIONAL bounded strong operator; fixed-epsilon same-L graph',
                      'packet_initial_propagator_retained': True, 'physical1_IR_atom_residue_admission': False}))


if __name__ == '__main__':
    main()
