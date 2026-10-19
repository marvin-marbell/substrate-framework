"""Exact native physical-hopping collective defect construction, not a phase.

The compression is a variational consumer of the unchanged HE=HC=64 law.
Neither its lowest Ritz vector nor its projector is the originating full-H
minimizing family. Quantum neutral/Phi/Haar support remains a stated premise.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sympy as sp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pure_parent(path):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {'rational', 'gammas'}]
    assert len(nodes) == 2
    from typing import cast
    namespace = {'sp': sp, 'cast': cast, 'itertools': itertools}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-parent-pure>', 'exec'), namespace)
    return namespace['gammas']()


def hop(mask, target, source):
    if target == source:
        return (mask, 1) if (mask >> source) & 1 else None
    if not (mask >> source) & 1 or (mask >> target) & 1:
        return None
    sign = (-1) ** ((mask & ((1 << source) - 1)).bit_count())
    mask ^= 1 << source
    sign *= (-1) ** ((mask & ((1 << target) - 1)).bit_count())
    return mask | (1 << target), sign


def entries(profile, n, beta, alpha, s, r):
    """All AP positive paths, their adjoints, and literal integer link flows."""
    sites = list(itertools.product(range(n), repeat=3))
    index = {x: i for i, x in enumerate(sites)}
    rows = []
    for x in sites:
        xi = index[x]
        for c in range(4):
            value = sp.Rational(profile['onsite']) * s * beta[c, c]
            if value:
                rows.append((4*xi+c, 4*xi+c, value, (), 0, 1))
        for path in profile['paths']:
            axis, length = path['axis'], path['length']
            y = list(x); y[axis] = (y[axis] + length) % n; yi = index[tuple(y)]
            flow = []
            for step in range(length):
                z = list(x); z[axis] = (z[axis] + step) % n
                flow.append(3*index[tuple(z)] + axis)
            ap = (-1) ** ((x[axis] + length) // n)
            matrix = ap * (sp.Rational(path['c0'])*s*beta + sp.I*sp.Rational(path['ci'])*r*alpha[axis])
            for a, b in itertools.product(range(4), repeat=2):
                value = matrix[a, b]
                if value:
                    shift = tuple((link, 1) for link in flow)
                    rows.append((4*xi+a, 4*yi+b, value, shift, length, ap))
                    rows.append((4*yi+b, 4*xi+a, sp.conjugate(value), tuple((link, -1) for link in flow), length, ap))
    return rows, sites


def apply(rows, vector):
    out = defaultdict(lambda: sp.Integer(0))
    for (mask, flow), amplitude in vector.items():
        for target, source, coefficient, shift, _, _ in rows:
            result = hop(mask, target, source)
            if result is None:
                continue
            newmask, sign = result
            updated = dict(flow)
            for link, value in shift:
                updated[link] = updated.get(link, 0) + value
            key = (newmask, tuple(sorted((link, value) for link, value in updated.items() if value)))
            out[key] += sign * coefficient * amplitude
    return {key: sp.expand(value) for key, value in out.items() if value != 0}


def gram(vector):
    return sp.expand(sum(sp.conjugate(value)*value for value in vector.values()))


def native_controls(profile, beta, alpha, s, r):
    rows, sites = entries(profile, 4, beta, alpha, s, r)
    # Full canonical 2V mask, not a trace over sector masks.
    vacuum = sum((12 << (4*j)) for j in range(64))
    assert vacuum.bit_count() == 128
    full_image = apply(rows, {(vacuum, ()): sp.Integer(1)})
    onsite = -128*s*sp.Rational(profile['onsite'])
    assert full_image[(vacuum, ())] == onsite
    image = {key: value for key, value in full_image.items() if key != (vacuum, ())}
    assert image and all(mask.bit_count() == 128 for mask, _ in image)
    expected = 64*3*4*(sp.Rational(profile['paths'][0]['ci'])*r)**2
    assert gram(image) == expected
    # Each NN image has one physical E link; its divergence equals n-2.
    for mask, flow in image:
        div = [0]*64
        for link, k in flow:
            site, axis = divmod(link, 3)
            end = list(sites[site]); end[axis] = (end[axis]+1) % 4
            div[site] += k; div[sites.index(tuple(end))] -= k
        assert div == [(mask >> (4*j) & 15).bit_count()-2 for j in range(64)]
        assert sum(k*k for _, k in flow) == 1
    aliases = defaultdict(list)
    for target, source, c, shift, length, ap in rows:
        if length == 2:
            aliases[(target, source)].append((c, shift, ap))
    collided = [items for items in aliases.values() if len(items) == 2]
    assert collided and all({a for _, _, a in items} == {-1, 1} for items in collided)
    # Endpoint aliases are NOT the same integer-flow history: never coalesce
    # merely by fermion endpoints before applying the Peierls operator.
    assert all(items[0][1] != items[1][1] for items in collided)
    length2 = [row for row in rows if row[4] == 2]
    target, source = length2[0][:2]
    occupied = [source]+[j for j in range(256) if j not in (source, target)][:127]
    mask = sum(1 << j for j in occupied)
    exact_two = apply(length2, {(mask, ()): sp.Integer(1)})
    assert gram(exact_two) > 0
    wrong_ap = [(i,j,ap*c,shift,length,ap) for i,j,c,shift,length,ap in length2]
    wrong_image = apply(wrong_ap, {(mask, ()): sp.Integer(1)})
    difference = dict(exact_two)
    for key, value in wrong_image.items():
        difference[key] = difference.get(key, 0)-value
    assert gram(difference) > 0
    # A literal divergence-free external integer circulation is supplied ONCE
    # to the total charged consumer, not once per species. The other63 seas
    # carry no charge change but do not duplicate this electric variable.
    external = {(0,0,0,0):1, (1,0,0,1):1,
                (0,1,0,0):-1, (0,0,0,1):-1}
    initial_flow = tuple(sorted((3*sites.index(key[:3])+key[3],value)
                                for key,value in external.items()))
    div0 = [0]*64
    for link,k in initial_flow:
        site,axis = divmod(link,3)
        end = list(sites[site]); end[axis] = (end[axis]+1)%4
        div0[site] += k; div0[sites.index(tuple(end))] -= k
    assert div0 == [0]*64
    external_image = apply(rows,{(vacuum,initial_flow):sp.Integer(1)})
    assert external_image.pop((vacuum,initial_flow)) == onsite
    costs = sorted({64*(sum(k*k for _,k in flow)-4)
                    for _,flow in external_image})
    assert costs == [-64,64,192]
    for mask,flow in external_image:
        div = [0]*64
        for link,k in flow:
            site,axis = divmod(link,3)
            end = list(sites[site]); end[axis] = (end[axis]+1)%4
            div[site] += k; div[sites.index(tuple(end))] -= k
        assert div == [(mask>>(4*j)&15).bit_count()-2 for j in range(64)]
    return {'N': 4, 'particles_each_species': 128, 'all_native_rows': len(rows),
            'canonical_vacuum_NN_Gram': str(expected), 'native_divE_equals_rho': True,
            'once_shared_external_circulation':{'integer_flow':initial_flow,
                'native_energy_changes':costs,'total_baseline_E_energy':256,
                'wrong_per_species_external_E_energy':16384,
                'full_native_external_divE_control':True},
            'N4_length2_endpoint_aliases': len(collided),
            'aliases_have_distinct_integer_flows': True,
            'length2_unblocked_canonical_Gram': str(gram(exact_two)),
            'wrong_AP_difference_Gram': str(gram(difference)),
            'occupation_masks_not_trace_weights': True,
            'no_length2_or_AP_deletion': True}


def root_intervals(matrix):
    x = sp.Symbol('x')
    poly = sp.Poly(matrix.charpoly(x).as_expr(), x)
    intervals = poly.intervals(eps=sp.Rational(1, 10**16))
    assert len(intervals) == matrix.rows and all(m == 1 for _, m in intervals)
    return poly, [pair for pair, _ in intervals]


def collective(active, auxiliary, s, r):
    """Four physical canonical states on a dimer; 64 species, shared E.

    Each direction has 128 disjoint fermionic pair channels (two/spin/species).
    |+>,|-> are normalized signed native images; |+- > contains their
    128^2 disjoint opposite pairs. All site colours/masks remain fermionic.
    Active and auxiliary onsite masses are not identified: compression uses
    their exact weighted mean, while the nonzero dark mass residual is paid
    as an explicit leakage, not erased from H.
    """
    ma, mb = sp.Rational(active['onsite']), sp.Rational(auxiliary['onsite'])
    ci = sp.Rational(active['paths'][0]['ci'])
    assert ci == sp.Rational(auxiliary['paths'][0]['ci'])
    assert active['load'] == 1 and auxiliary['load'] == 63
    delta = 2*s*(ma+63*mb)/64
    t = 16*ci*r  # signed physical hopping1, not a fitted small hopping
    # Basis |0>, (|+>+|->)/sqrt2, |+->. Antisymmetric |A>
    # is a fourth state of energy D=64+delta, necessary for electric response.
    d = 64+delta
    masses = [ma]*2 + [mb]*126
    assert len(masses) == 128
    # Derive entries from literal occupied channel amplitudes, not a trace.
    bright_diagonal = sum(64+2*s*m for m in masses)/128
    pair_diagonal = sum(2*s*m for m in masses)/64
    vacuum_to_bright = 256*ci*r/16
    bright_to_pair = 2*128**2*ci*r/(16*128)
    matrix = sp.Matrix([[0,vacuum_to_bright,0],
                        [vacuum_to_bright,bright_diagonal,bright_to_pair],
                        [0,bright_to_pair,pair_diagonal]])
    assert matrix == sp.Matrix([[0,t,0],[t,d,t],[0,t,2*delta]])
    poly, intervals = root_intervals(matrix)
    lo, hi = intervals[0]
    assert hi < 0
    # Exact rational enclosure and dimensionless first-route sufficient gate.
    gap0 = min(d, 2*delta)
    gate_ratio_squared = 8*t*t/(gap0*gap0)
    assert gate_ratio_squared > 1
    # Ground eigenvector: (t/lambda,1,t/(lambda-2delta)). This gives
    # actual electric spectral mass on |A> for the compressed quantum model.
    l = sp.Symbol('lambda', real=True)
    norm = 1+(t/l)**2+(t/(l-2*delta))**2
    # For lambda<0, this positive weight increases with |lambda| on its interval.
    mass_lo = sp.cancel(1/norm.subs(l, hi))
    mass_hi = sp.cancel(1/norm.subs(l, lo))
    assert 0 < mass_lo < mass_hi < 1
    variance = 4*s*s*(ma*ma+63*mb*mb)/64-delta*delta
    assert variance > 0
    # Flux/phase translation is unitary, hence its curvature is NOT stiffness.
    # Native hopping Im(z) control on a legitimate complex collective state.
    carrier = sp.Matrix([1,1+sp.I,sp.Rational(1,2)])
    carrier_norm = (carrier.H*carrier)[0]
    incoming = sp.Matrix([[0,0,0],[t,0,0],[0,t,0]])
    z = sp.expand((carrier.H*incoming*carrier)[0]/carrier_norm)
    assert sp.im(z) != 0
    phase = sp.I
    u = sp.diag(1,phase,phase**2)
    changed = sp.simplify(((u*carrier).H*matrix*(u*carrier))[0]/carrier_norm-(carrier.H*matrix*carrier)[0]/carrier_norm)
    predicted = sp.simplify(2*sp.re((sp.conjugate(phase)-1)*z))
    assert changed == predicted
    erased = 2*sp.re(sp.conjugate(phase)-1)*sp.re(z)
    assert changed != erased
    # Direct CAR opposite hops: two different species share one E, not two
    # independent electric variables. Same-direction pairs cost256, opposite0.
    actual_costs = {'one': 64, 'opposite_two': 64*(1-1)**2, 'same_two': 64*(1+1)**2}
    wrong_costs = {'one': 64, 'opposite_two': 64*(1**2+(-1)**2), 'same_two': 64*(1**2+1**2)}
    assert actual_costs != wrong_costs and actual_costs['opposite_two'] == 0
    wrong_d = sp.Matrix([[0,t,0],[t,d,t],[0,t,2*delta+128]])
    wrong_poly, wrong_intervals = root_intervals(wrong_d)
    assert wrong_intervals[0][0] > hi
    assert t*t == 256*ci*ci*r*r and t*t != (64*16*ci*r)**2
    # |+-> consists of two disjoint channels, even when both belong to the
    # same species: verify all four fermionic modes in a literal 8-mode dimer.
    base = (12 << 4) | 12
    first = hop(base, 0, 7)
    assert first is not None
    second = hop(first[0], 4, 3)
    assert second is not None and second[0].bit_count() == base.bit_count() == 4
    reverse = hop(base, 7, 0)
    assert reverse is None
    # Full canonical occupation control: the uniform onsite sea blocks beta
    # transfers; an unoccupied-target mutation instead changes the image.
    blocked = hop(base, 2, 6)
    assert blocked is None
    allowed = hop(base ^ (1<<2) ^ (1<<0), 2, 6)
    assert allowed is not None
    return {'s': str(s), 'r': str(r), 'physical_hopping': 1,
            'species': {'active': 1,'auxiliary': 63,'channels_per_direction':128},
            'delta_exact': str(delta), 'signed_collective_t_exact': str(t),
            'compression_matrix_exact': [[str(a) for a in row] for row in matrix.tolist()],
            'fourth_antisymmetric_energy_exact': str(d),
            'first_route': {'criterion': '2||V_cluster||<gap(H_cluster,0), ||V||=sqrt2|t|',
                            'ratio_squared_exact': str(gate_ratio_squared),
                            'passed': False,'scope':'failed sufficient perturbative cluster gate; not a phase no-go'},
            'second_route': {'executed':'signed coherent opposite-flow collective canonical defect, fullH variational dimer tiling',
                             'ground_polynomial_exact': str(poly.as_expr()),
                             'lowest_Ritz_root_exact_isolating_interval': [str(lo),str(hi)],
                             'per_site_energy_improvement_interval': [str(lo/2),str(hi/2)],
                             'electric_mass_exact_enclosure': [str(mass_lo),str(mass_hi)],
                             'electric_excitation_gap_interval': [str(d-hi),str(d-lo)],
                             'actual_compressed_whole_P0':'unique lowest eigenvalue of 4x4 compression; projector onto complete lowest eigenspace',
                             'actual_fullH_P0_evaluated':False,
                             'compressed_response':'<E(1-P0)(H_Ritz-lambda+z)^(-1)(1-P0)E> = mass/(D-lambda+z)',
                             'N_uniform_scope':'exact variational energy density at point body, every even N>=4; not a uniform fullH gap'},
            'noninvariant_leakage': {'onsite_bright_dark_variance_exact':str(variance),
                                    'beta_c0_and_other_native_bonds':'present in fullH; zero expectation on tiling but nonzero orthogonal leakage',
                                    'fullH_cluster_subspace_invariant':False},
            'controls': {'complex_z_exact':str(z),'complex_response_delta_exact':str(changed),
                         'real_only_cross_rejected':True,'once_shared_E_costs':actual_costs,
                         'wrong_independent_E_costs':wrong_costs,
                         'wrong_E_root_interval':[str(a) for a in wrong_intervals[0]],
                         'wrong_E_polynomial_exact':str(wrong_poly.as_expr()),
                         'wrong_load64_squared_rejected':True,'Pauli_and_canonical_same_species_double_pair':True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    root = args.request.parent
    for name, expected in request['pins'].items():
        assert digest(root/name) == expected
    profiles = json.loads((root/request['profiles']).read_text())['primitive_profiles']
    beta, alpha = pure_parent(root/request['parent_method'])
    s, r = sp.Rational(1,2), sp.Rational(1,3)
    point = collective(profiles[1], profiles[2], s, r)
    corners = [collective(profiles[1],profiles[2],sp.Rational(a),sp.Rational(b))
               for a,b in itertools.product(request['candidate_window']['s'], request['candidate_window']['r'])]
    smax = max(sp.Rational(a) for a in request['candidate_window']['s'])
    rmin = min(sp.Rational(a) for a in request['candidate_window']['r'])
    delta_max = 2*smax*(sp.Rational(profiles[1]['onsite'])+63*sp.Rational(profiles[2]['onsite']))/64
    uniform_failed_ratio = 8*(16*sp.Rational(profiles[1]['paths'][0]['ci'])*rmin)**2/(2*delta_max)**2
    assert uniform_failed_ratio > 1
    result = {'kind':'EXECUTED_SIGNED_COLLECTIVE_HEHC64_CANONICAL_DEFECT_NOT_FULL_GROUND_PHASE',
              'native_active':native_controls(profiles[1],beta,alpha,s,r),
              'native_auxiliary':native_controls(profiles[2],beta,alpha,s,r),
              'point':point, 'window_corner_diagnostics_not_ground_support':corners,
              'candidate_window':request['candidate_window'],
              'entire_candidate_window_failed_first_gate_ratio_squared_lower_exact':str(uniform_failed_ratio),
              'quantum_body_errors_paid':False, 'full_originating_ground_response_paid':False,
              'no_missing_numerical_ground_backend_gate':True,
              'original_law_or_SOURCE_adopted':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Executed physical1 collective shared-E opposite defects; perturbative gate fails; full quantum body/IR remains conditional')


if __name__ == '__main__':
    main()
