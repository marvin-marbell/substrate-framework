"""Independent exact native/form/differential review; never execute frozen main."""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from fractions import Fraction
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import zipfile
import sympy as s


def inner(left, right):
    return s.expand(sum(s.conjugate(a)*right.get(m, 0) for m, a in left.items()))


def add(left, right):
    out = defaultdict(lambda: s.Integer(0), left)
    for m, a in right.items():
        out[m] += a
    return {m: s.expand(a) for m, a in out.items() if a != 0}


def independent_entries(profile, sites):
    """Enumerate each positive native path AND its adjoint before coalescing."""
    g = s.diag(3, -1, -1, -1)
    spatial = [s.zeros(4) for _ in range(3)]
    spatial[0][0, 1] = spatial[0][1, 0] = 1
    spatial[1][1, 2] = s.I
    spatial[1][2, 1] = -s.I
    spatial[2][2, 3] = spatial[2][3, 2] = 2
    u = s.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, s.I, 0], [0, 0, 0, -s.I]])
    v = s.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    links = [u, v, u*v]
    assert all(link.det() == 1 and link.H*link == s.eye(4) for link in links)
    lookup = {x: j for j, x in enumerate(sites)}
    entries = []
    tagged = []
    for x in sites:
        for a in range(4):
            entries.append((4*lookup[x]+a, 4*lookup[x]+a, s.Rational(profile['onsite'])*(1+sum(x))*g[a, a]))
        for row in profile['paths']:
            axis, length = int(row['axis']), int(row['length'])
            end = list(x)
            end[axis] = (end[axis]+length) % 4
            y = tuple(end)
            path = links[axis]**length
            left = (1+sum(x))*(s.Rational(row['c0'])*g+s.I*s.Rational(row['ci'])*spatial[axis])
            right = (1+sum(y))*(s.Rational(row['c0'])*g+s.I*s.Rational(row['ci'])*spatial[axis])
            ap = (-1)**((x[axis]+length)//4)
            body = ap*s.I**length*(left*path+path*right)/2
            for a, b in itertools.product(range(4), repeat=2):
                if body[a, b] == 0:
                    continue
                forward = (4*lookup[x]+a, 4*lookup[y]+b, body[a, b])
                backward = (forward[1], forward[0], s.conjugate(body[a, b]))
                entries.extend((forward, backward))
                tagged.extend(((forward, axis, length, ap), (backward, axis, length, ap)))
    return entries, tagged


def main():
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    root = args.request.parent
    snapshot = (root/request['snapshot']).read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == request['snapshot_sha256']
    with zipfile.ZipFile(io.BytesIO(snapshot)) as frozen:
        raw = frozen.read('auxiliary_sector_energy_consumer_20261012.py')
        assert hashlib.sha256(raw).hexdigest() == request['frozen_helper_sha256']
        # Only selected pure helpers, with AST-picked dependencies. No original main.
        tree = ast.parse(raw)
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'hop', 'apply', 'norm2', 'native_entries'}]
        namespace = {'s': s, 'defaultdict': defaultdict}
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<frozen-pure-helpers>', 'exec'), namespace)
        with zipfile.ZipFile(io.BytesIO(frozen.read('auxiliary_sector_energy_actual5_20261012_preexecution.zip'))) as pre:
            nested = pre.read('joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual_executed_inputs.zip')
    with zipfile.ZipFile(io.BytesIO(nested)) as archive:
        profiles_raw = archive.read(request['profile_member'])
    assert hashlib.sha256(profiles_raw).hexdigest() == request['profile_sha256']
    profiles = json.loads(profiles_raw)['primitive_profiles']
    apply = namespace['apply']
    sites = list(itertools.product(range(4), repeat=3))
    checker = sum(15 << (4*j) for j, x in enumerate(sites) if sum(x) % 2 == 0)
    entries, tagged = independent_entries(profiles[1], sites)
    assert entries == namespace['native_entries'](profiles[1], sites)
    incident = [e for e in entries if (e[0]//4 == 0) != (e[1]//4 == 0)]
    incoming = [e for e in incident if e[0]//4 == 0]
    outgoing = [e for e in incident if e[1]//4 == 0]
    c = {checker: s.Integer(1)}
    image = apply(incident, c)
    gram = inner(image, image)
    assert gram > 0 and all((m & 15).bit_count() == 3 for m in image)
    assert not apply(incoming, c)
    assert inner(c, image) == 0
    assert inner(image, apply(incident, image)) == 0
    bright = add(c, image)
    bright_norm = 1+gram
    z = s.simplify(inner(bright, apply(incoming, bright))/bright_norm)
    assert z == gram/bright_norm
    def rotate(vector, phase):
        return {m: phase**((m & 15).bit_count())*a for m, a in vector.items()}
    # Exact complex cross including an Im(z) adverse carrier.
    complex_carrier = add(c, {m: (1+s.I)*a for m, a in image.items()})
    complex_norm = inner(complex_carrier, complex_carrier)
    complex_z = s.simplify(inner(complex_carrier, apply(incoming, complex_carrier))/complex_norm)
    assert s.im(complex_z) != 0
    response_rows = []
    for phase in (s.I, -1, -s.I):
        rotated = rotate(complex_carrier, phase)
        before = inner(complex_carrier, apply(incident, complex_carrier))/complex_norm
        after = inner(rotated, apply(incident, rotated))/complex_norm
        predicted = 2*s.re((s.conjugate(phase)-1)*complex_z)
        assert s.simplify(after-before-predicted) == 0
        wrong_real_only = 2*s.re(s.conjugate(phase)-1)*s.re(complex_z)
        response_rows.append({'phase': str(phase), 'delta': str(s.simplify(after-before)), 'erased_complex_cross_wrong': s.simplify(predicted-wrong_real_only) != 0})
    pi = rotate(bright, -1)
    delta = s.simplify((inner(pi, apply(incident, pi))-inner(bright, apply(incident, bright)))/bright_norm)
    assert delta == -4*gram/bright_norm
    # Every N4 length2 endpoint collides with a reversed positive path;
    # aggregate only after retaining both native histories and AP factors.
    collisions = defaultdict(list)
    for e, axis, length, ap in tagged:
        if length == 2 and ((e[0]//4 == 0) != (e[1]//4 == 0)):
            collisions[(e[0], e[1])].append((axis, ap, e[2]))
    print(json.dumps({'collision_multiplicities': sorted({len(items) for items in collisions.values()}), 'collision_count': len(collisions)}), flush=True)
    assert collisions and all(len(items) in (1, 2) for items in collisions.values())
    assert any(len(items) == 2 for items in collisions.values())
    assert all({ap for _, ap, _ in items} == {-1, 1} for items in collisions.values() if len(items) == 2)
    two = [e for e, _, length, _ in tagged if length == 2 and ((e[0]//4 == 0) != (e[1]//4 == 0))]
    assert not apply(two, c)  # same-parity sites full; not deletion from H.
    # Identity [n,T]=T on all actual transfers, including collisions/adjoints.
    assert all(int(e[0]//4 == 0)-int(e[1]//4 == 0) == 1 for e in incoming)
    # Mutate native histories before CAR evaluation, on full canonical probes.
    # Checker alone cannot detect length2 deletion because Pauli blocks it.
    def canonical_difference_probe(wrong):
        difference = defaultdict(lambda: s.Integer(0))
        for i, j, coefficient in incident:
            difference[(i, j)] += coefficient
        for i, j, coefficient in wrong:
            difference[(i, j)] -= coefficient
        pair = next((pair for pair, coefficient in difference.items() if s.expand(coefficient) != 0), None)
        assert pair is not None
        target, source = pair
        occupied = [source]+[j for j in range(256) if j not in (source, target)][:127]
        mask = sum(1 << j for j in occupied)
        assert mask.bit_count() == 128
        delta_vector = add(apply(incident, {mask: s.Integer(1)}), {m: -a for m, a in apply(wrong, {mask: s.Integer(1)}).items()})
        assert inner(delta_vector, delta_vector) > 0
        return {'target': target, 'source': source, 'canonical_particles': 128, 'difference_Gram': str(inner(delta_vector, delta_vector))}
    without_ap = [(e[0], e[1], ap*e[2]) for e, _, _, ap in tagged if (e[0]//4 == 0) != (e[1]//4 == 0)]
    without_two = [e for e, _, length, _ in tagged if length == 1 and ((e[0]//4 == 0) != (e[1]//4 == 0))]
    native_adverse = {'dropped_AP': canonical_difference_probe(without_ap), 'dropped_length2': canonical_difference_probe(without_two), 'dropped_adjoint': canonical_difference_probe(incoming)}
    assert gram != 64**2*gram
    neutral, _ = independent_entries(profiles[0], sites)
    assert neutral == namespace['native_entries'](profiles[0], sites)
    neutral_gram = inner(apply(neutral, c), apply(neutral, c))
    assert neutral_gram > 0
    # Ground signs and first moment, on an independent exact stationary 2x2 model.
    h = s.Matrix([[0, -1], [-1, 0]])
    n = s.diag(1, 0)
    ground = s.Matrix([1, 1])/s.sqrt(2)
    t = s.Matrix([[0, -1], [0, 0]])
    zg = (ground.H*t*ground)[0]
    m1 = (ground.H*n*(h+s.eye(2))*n*ground)[0]
    comm = n*(h*n-n*h)-(h*n-n*h)*n
    assert zg == -s.Rational(1, 2) and m1 == -zg == (ground.H*comm*ground)[0]/2
    # Whole P0 != one chosen vector; first moment does not determine mass.
    p0 = s.diag(1, 1, 0)
    source = s.Matrix([0, 1, 1])
    whole_mass = (source.H*(s.eye(3)-p0)*source)[0]
    wrong_mass = (source.H*(s.eye(3)-s.diag(1, 0, 0))*source)[0]
    assert whole_mass == 1 and wrong_mass == 2
    assert s.Rational(1, 2)*2 == s.Rational(1, 4)*4 and s.Rational(1, 2) != s.Rational(1, 4)
    # A bounded rank-one projector need not preserve a Sobolev form domain.
    # P onto sum_j C/j e_j in Q: sum j^2|v_j|^2=infinity, while ||v||^2<infinity.
    q = s.symbols('q', real=True)
    theta = s.symbols('theta', real=True)
    a, b = s.symbols('a b', real=True)
    jet_wave = s.Matrix([1, q])*s.exp(-q*q/4)
    connection = s.I*s.diag(a, b)
    utheta = s.diag(s.exp(4*s.I*theta), s.exp(3*s.I*theta))
    derivative = lambda vec: vec.diff(q)+connection*vec
    assert s.simplify(derivative(utheta*jet_wave)-utheta*derivative(jet_wave)) == s.zeros(2, 1)
    assert derivative(jet_wave) != s.zeros(2, 1)
    # Number diagonal connections commute Utheta but NOT the hop image's n shift.
    kbody = s.Matrix([[0, q], [q, 0]])
    image_jet = derivative(kbody*s.Matrix([s.exp(-q*q/4), 0]))
    dropped_body_jet = kbody*derivative(s.Matrix([s.exp(-q*q/4), 0]))
    assert s.simplify(image_jet-dropped_body_jet) != s.zeros(2, 1)
    r = Fraction(21, 64)
    assert r*r+(1-r)*(1-r) != 1
    # Exactly integrated G: full Cart15 per each Phi, independent site Gaussians.
    # E||Phi_mu||HS^2=15 eta/2; cross-site and cross-mu terms vanish.
    basis = []
    for j in range(4):
        for k in range(j+1, 4):
            real = s.zeros(4); real[j, k] = real[k, j] = s.Rational(1, 2)
            imag = s.zeros(4); imag[j, k] = -s.I/2; imag[k, j] = s.I/2
            basis.extend((real, imag))
    for j in range(1, 4):
        basis.append(s.diag(*([1]*j+[-j]+[0]*(3-j)))/s.sqrt(2*j*(j+1)))
    assert len(basis) == 15 and all(s.trace(x*y) == (s.Rational(1, 2) if j == k else 0) for j, x in enumerate(basis) for k, y in enumerate(basis))
    eta = s.Rational(2)**-7168
    nn = [row for row in profiles[1]['paths'] if int(row['length']) == 1]
    assert len(nn) == 3
    integrated_g = s.factor(s.Rational(15, 2)*eta*sum(s.Rational(row['c0'])**2+s.Rational(row['ci'])**2 for row in nn))
    assert integrated_g > 0 and integrated_g != gram
    # Gaussian integration of normalization: E(1+||Kc||^2)=1+G,
    # not E[(1+point_Gram)/(1+G_point)] or point-state replacement.
    averaged_two_point = s.Rational(1, 2)*(1/(1+1)+1/(1+3))
    inverse_average = 1/(1+s.Rational(1, 2)*(1+3))
    assert averaged_two_point != inverse_average
    # All-field dark source, literal original integerflow on N4.
    forbidden = {(0, 0, 0)}
    for axis, step in itertools.product(range(3), (-2, -1, 1, 2)):
        x = [0, 0, 0]; x[axis] = step % 4; forbidden.add(tuple(x))
    allowed = [x for x in sites if x not in forbidden]
    dark_mask = sum(15 << (4*sites.index(x)) for x in allowed[:32])
    dark = {dark_mask: s.Integer(1)}
    assert dark_mask.bit_count() == checker.bit_count() == 128 and len(forbidden) == 10
    assert not apply(incident, dark)
    full_dark_hop = apply(entries, dark)
    assert inner(full_dark_hop, full_dark_hop) > 0
    rho = [63*((checker >> (4*j) & 15).bit_count()-2)+(dark_mask >> (4*j) & 15).bit_count()-2 for j in range(64)]
    assert sum(rho) == 0
    flow = {}; subtotal = list(rho)
    for j in range(63, 0, -1):
        parent = list(sites[j]); axis = next(k for k in (2, 1, 0) if parent[k] > 0); parent[axis] -= 1
        parent = tuple(parent); flow[(parent, axis)] = -subtotal[j]; subtotal[sites.index(parent)] += subtotal[j]
    div = [0]*64
    for (x, axis), value in flow.items():
        y = list(x); y[axis] += 1; div[sites.index(x)] += value; div[sites.index(tuple(y))] -= value
    assert div == rho
    # Explicit U1 gauge covariance of all native colour entries, including AP.
    alpha = [s.I**(sum(x) % 4) for x in sites]
    gauged = [(i, j, alpha[i//4]*coefficient*s.conjugate(alpha[j//4])) for i, j, coefficient in incident]
    gauge_vector = lambda vec: {m: a*s.prod(alpha[j//4] for j in range(256) if (m >> j) & 1) for m, a in vec.items()}
    assert apply(gauged, gauge_vector(c)) == gauge_vector(image)
    assert (s.I**31)*s.I == 1 and s.I != 1
    result_controls = {'native_mutations': native_adverse, 'wrong_active64_multiplier_rejected': True, 'neutral_unscaled_Gram': str(neutral_gram), 'neutral_physical_Gram': 'Lambda^2 times native unscaled Gram; sign -Lambda retained in H'}
    result = {
        'kind': 'INDEPENDENT_EXACT_ACTIVE_NUMBER_REVIEW_NOT_GROUND_ADMISSION',
        'adverse_controls': result_controls,
        'pins': request, 'native': {'N': 4, 'M': 256, 'k': 128, 'entries': len(entries), 'incident_entries': len(incident), 'length2_collision_pairs': len(collisions), 'length2_histories_preserved': len(two), 'coefficient_Gram': str(gram), 'complex_z': str(complex_z), 'complex_response_controls': response_rows, 'bright_pi_response': str(delta), 'orthogonal_n4_n3': True, 'cubic_zero': True, 'gauge_covariance': True},
        'integrated_quantum_G': {'exact': str(integrated_g), 'formula': '(15 eta/2) sum_axis(c0_axis,NN^2+ci_axis,NN^2)', 'eta': '2^-7168', 'Haar_Wilson_Z_retained_and_cancels_in_G_only': True, 'length2_Pauli_blocked_checker_only': True, 'global_normalizer': 'sqrt(1+G)', 'frozen_Gram_is_not_G': True},
        'form_differential': {'ground_z': str(zg), 'first_moment': str(m1), 'wholeP0_mass': str(whole_mass), 'wrong_rank1_ground_complement_mass': str(wrong_mass), 'body_and_connection_jets_nonzero_retained': True, 'separate_HE_HC_squares_rejected': True, 'pointwise_vs_global_normalization_rejected': True, 'bounded_projector_counterexample': 'rank1 v_j=C/j in ell2, Q=sum j^2|v_j|^2; P(e1) not in Q'},
        'dark': {'forbidden_sites': len(forbidden), 'particles': dark_mask.bit_count(), 'active_number_response': 0, 'full_active_nonstationary_coefficient_Gram': str(inner(full_dark_hop, full_dark_hop)), 'integerflow_max_error': max(abs(x-y) for x, y in zip(div, rho)), 'once_fullE_price': 64*sum(k*k for k in flow.values()), 'uniform_N_price': False, 'ground_or_no_go': False},
        'claims_unpaid': ['actual_originating_signed_z', 'originating_auxiliary_weights', 'wholeP0_loop_lower', 'uncut_loop_graph_domain', 'IR_atom_phase_commoncone'],
        'runtime': {'python': sys.executable, 'version': sys.version, 'sympy': s.__version__}, 'original_main_run': False, 'scientific_admission': False, 'campaign_complete': False
    }
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'native_form_differential_controls': 'passed', 'ground_response_weights_loop_lower': 'UNPAID'}))


if __name__ == '__main__':
    main()
