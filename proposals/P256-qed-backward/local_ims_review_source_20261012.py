"""Independent exact translated-cube IMS, canonical dual and native charged consumer.
No original CLI, eigensolver, optimizer, ground vector or trial state is invoked.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import zipfile

import sympy as sp


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact(a):
    return sp.simplify(a)


def hs2(a):
    return exact(sp.trace(a * a.conjugate().T))


def q(text):
    value = Fraction(text)
    return sp.Rational(value.numerator, value.denominator)


def cube(n, r, z):
    return {tuple((z[a] + t[a]) % n for a in range(3))
            for t in itertools.product(range(r), repeat=3)}


def translated_controls():
    rows = []
    for n in (4, 6):
        sites = list(itertools.product(range(n), repeat=3))
        for r in sorted({1, 2, n // 2, n - 1}):
            cubes = [cube(n, r, z) for z in sites]
            vertex = {x: 0 for x in sites}
            overlaps = {(x, a): 0 for x in sites for a in range(3)}
            for patch in cubes:
                assert len(patch) == r ** 3
                for x in patch:
                    vertex[x] += 1
                    for a in range(3):
                        y = list(x); y[a] = (y[a] + 1) % n
                        if tuple(y) in patch:
                            overlaps[x, a] += 1
            assert set(vertex.values()) == {r ** 3}
            assert set(overlaps.values()) == {(r - 1) * r ** 2}
            # Actual seam edges have AP=-1; multiplicity is unchanged, not sign erased.
            seam = [overlaps[x, a] for x in sites for a in range(3) if x[a] == n - 1]
            assert set(seam) == {(r - 1) * r ** 2}
            rows.append({'N': n, 'R': r, 'cubes': len(cubes),
                         'vertex_multiplicity': r ** 3,
                         'NN_overlap': str(sp.Rational(r - 1, r)),
                         'AP_seam_edges_counted': len(seam)})
    return rows


def native_fixture(profiles):
    n = 4
    sites = list(itertools.product(range(n), repeat=3))
    index = {x: i for i, x in enumerate(sites)}
    eye = sp.eye(4)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.diag(1, -1)
    beta = sp.kronecker_product(sz, sp.eye(2))
    alpha = [sp.kronecker_product(sx, a) for a in (sx, sy, sz)]
    u = (3 * eye + 4 * sp.I * beta) / 5
    v = (5 * eye + 12 * sp.I * alpha[0]) / 13
    assert exact(u.det()) == exact(v.det()) == 1
    assert (u * u.conjugate().T - eye).applyfunc(exact) == sp.zeros(4)
    assert (v * v.conjugate().T - eye).applyfunc(exact) == sp.zeros(4)
    face = u * v * u.conjugate().T * v.conjugate().T
    assert hs2(face - eye) > 0
    fields = []
    for x in sites:
        # Inhomogeneous and non-Clifford: Phi1 has a beta component.
        fields.append([beta / 2,
                       alpha[0] / 3 + (-1) ** x[0] * beta / 4,
                       (-1) ** x[1] * alpha[1] / 3,
                       (-1) ** x[2] * alpha[2] / 3])
    s = exact(sum(hs2(p) for p in fields[0]))
    assert s == sp.Rational(31, 12)
    assert all(exact(sum(hs2(p) for p in f)) == s for f in fields)
    links = {(i, a): (u if a == 0 else v if a == 1 else eye)
             for i in range(len(sites)) for a in range(3)}
    # A globally canonical rank2V projector, with nonlocal coherence on an AP seam.
    occupations = [[1, 1, 0, 0] for _ in sites]
    seam_x = index[(0, 0, 0)]; seam_y = index[(3, 0, 0)]
    occupations[seam_y] = [0, 0, 1, 1]
    gamma = {(i, i): sp.diag(*occupation) for i, occupation in enumerate(occupations)}
    c, t = sp.Rational(3, 5), sp.Rational(4, 5)
    gamma[seam_x, seam_x][0, 0] = c ** 2
    gamma[seam_y, seam_y][0, 0] = t ** 2
    coherence = sp.zeros(4); coherence[0, 0] = c * t
    gamma[seam_x, seam_y] = coherence
    gamma[seam_y, seam_x] = coherence
    assert sum(sp.trace(gamma[i, i]) for i in range(len(sites))) == 2 * len(sites)
    two = sp.Matrix([[c ** 2, c * t], [c * t, t ** 2]])
    assert two * two == two

    def build(profile, charged=False, erase_ap=False, drop_two=False):
        blocks = {(i, i): q(profile['onsite']) * f[0] for i, f in enumerate(fields)}
        path_rows = []
        for path in profile['paths']:
            axis, length = path['axis'], path['length']
            if drop_two and length == 2:
                continue
            c0, ci = q(path['c0']), q(path['ci'])
            for i, x in enumerate(sites):
                current = list(x); transport = eye; ap = 1
                for step in range(length):
                    ii = index[tuple(current)]
                    transport = transport * links[ii, axis]
                    if charged:
                        # An exact native U1 phase; length2 uses both ordered links.
                        transport *= (3 + 4 * sp.I) / 5 if (ii + axis) % 2 else (5 + 12 * sp.I) / 13
                    if current[axis] == n - 1:
                        ap *= -1
                    current[axis] = (current[axis] + 1) % n
                j = index[tuple(current)]
                if erase_ap:
                    ap = 1
                p, f = fields[i], fields[j]
                block = ap * (c0 * (p[0] * transport + transport * f[0])
                              + sp.I * ci * (p[axis + 1] * transport + transport * f[axis + 1])) / 2
                block = block.applyfunc(exact)
                blocks[i, j] = blocks.get((i, j), sp.zeros(4)) + block
                blocks[j, i] = blocks.get((j, i), sp.zeros(4)) + block.conjugate().T
                path_rows.append((i, j, axis, length, block))
        return blocks, path_rows

    def expectation(blocks):
        return exact(sum(sp.trace(gamma.get((j, i), sp.zeros(4)) * h)
                         for (i, j), h in blocks.items()))

    neutral, paths = build(profiles[0])
    value = expectation(neutral)
    without_ap, _ = build(profiles[0], erase_ap=True)
    assert value != expectation(without_ap)
    ims_rows = []
    b = abs(q(profiles[0]['paths'][0]['c0'])) + abs(q(profiles[0]['paths'][0]['ci']))
    for r in (1, 2, 3):
        traces = 0; patch_energy = 0
        for z in sites:
            ids = {index[x] for x in cube(n, r, z)}
            traces += sum(sp.trace(gamma[i, i]) for i in ids)
            # This is the actual restriction P_z Gamma P_z, not an independently filled patch.
            patch_energy += sum(sp.trace(gamma.get((j, i), sp.zeros(4)) * h)
                                for (i, j), h in neutral.items() if i in ids and j in ids)
        traces = exact(traces / sp.Integer(r ** 3))
        localized = exact(patch_energy / sp.Integer(r ** 3))
        onsite = exact(sum(sp.trace(gamma[i, i] * neutral[i, i]) for i in range(len(sites))))
        assert traces == 2 * len(sites)
        assert localized == onsite + sp.Rational(r - 1, r) * (value - onsite)
        error = exact(value - localized)
        squared_price = (12 * b * len(sites) / r) ** 2 * s
        assert exact(squared_price - error ** 2) > 0
        ims_rows.append({'R': r, 'weighted_patch_trace': str(traces),
                         'actual_TrGamma_h': str(value), 'localized': str(localized),
                         'signed_residual': str(error), 'boundary_bound_squared': str(squared_price)})
    charged_rows = []
    for profile in profiles[1:]:
        h, paths = build(profile, charged=True)
        energy = expectation(h)
        d = 2 * abs(q(profile['onsite'])) + 4 * sum(abs(q(t['c0'])) + abs(q(t['ci'])) for t in profile['paths'])
        assert exact((d * len(sites)) ** 2 * s - energy ** 2) > 0
        assert exact(sum(sp.trace(h[i, i]) for i in range(len(sites)))) == 0
        # Signed filled complement: Tr[(I-Gamma)h] = -Tr[Gamma h].
        complement = exact(sum(sp.trace(h[i, i]) for i in range(len(sites))) - energy)
        assert complement == -energy
        h1, _ = build(profile, charged=True, drop_two=True)
        assert h != h1
        # Individual edge trace-norm bound follows from rank<=4 and HS norm.
        for i, j, axis, length, block in paths:
            coeff = next(t for t in profile['paths'] if t['axis'] == axis and t['length'] == length)
            bb = abs(q(coeff['c0'])) + abs(q(coeff['ci']))
            assert exact(4 * bb ** 2 * s - 4 * hs2(block)) >= 0
        charged_rows.append({'role': profile['role'], 'load': profile['load'], 'D_s': str(d),
                             'expectation': str(energy), 'signed_complement': str(complement),
                             'native_paths': len(profile['paths']), 'length2_blocks_material': True})
    dch = exact(sum(row['load'] * q(row['D_s']) for row in charged_rows))
    assert sum(row['load'] for row in charged_rows) == 64
    assert dch == sp.Rational(855444290449274477, 35184372088832)
    return {'sites': len(sites), 'onebody_dimension': 4 * len(sites), 'rank': 2 * len(sites),
            's_per_site': str(s), 'AP_changes_actual_expectation': True,
            'curved_face_HS_squared': str(hs2(face - eye)), 'IMS': ims_rows,
            'charged': charged_rows, 'D_ch': str(dch)}, b


def dual_controls(m):
    levels = [3 * m / 4, -m / 4, -m / 4, -m / 4,
              3 * m / 8, -m / 8, -m / 8, -m / 8]
    chosen = [0, 4, 5, 6]
    canonical = exact(sum(levels[i] for i in chosen))
    assert canonical == 7 * m / 8
    local_rank = 3 * m / 4
    assert canonical > local_rank
    rows = []
    # Noncommuting contraction; diagonal entries in A's eigenbasis lie in [0,1].
    w = sp.eye(8)
    w[0, 0] = w[1, 1] = sp.Rational(3, 5)
    w[0, 1] = -sp.Rational(4, 5); w[1, 0] = sp.Rational(4, 5)
    contraction = sp.diag(sp.Rational(1, 5), sp.Rational(4, 5), 0, 1, 1, 0, 1, 0)
    assert sp.trace(contraction) == 4
    h = w * sp.diag(*levels) * w.T
    assert h * contraction != contraction * h
    expectation = exact(sp.trace(contraction * h))
    for mu in (-m, -m / 4, -m / 8, 0, m):
        positive = sum(max(sp.S.Zero, x - mu) for x in levels)
        price = exact(4 * mu + positive)
        assert price >= canonical >= expectation
        rows.append({'one_common_mu': str(mu), 'dual': str(price), 'canonical': str(canonical),
                     'noncommuting_contraction_energy': str(expectation)})
    assert rows[2]['dual'] == str(canonical)
    return {'controls': rows, 'wrong_independent_rank2_patch_duals': str(local_rank),
            'actual_patch_occupations': [1, 3], 'negative_levels_occupied': 2,
            'positive_only_not_canonical': str(9 * m / 8),
            'counterexample_consequence': 'Independent mu_z with fixed2|B_z| discards actual trace redistribution and is not a valid canonical upper.'}


def moment_controls(b):
    m = sp.Integer(2) ** 14
    a = sp.Rational(40, 2 ** 2035)
    c = sp.Rational(10, 7 * 2 ** 2035)
    l = sp.Integer(2) ** 32
    eps2 = sp.Rational(1, 2 ** 200)
    rows = []
    for n in (4, 6, 8, 1024, 8192, 16384):
        r = min(4096, n // 2)
        cap = r ** 3 * m / l ** 4
        field = 48 * r ** 4 * a / eps2
        curve = 1728 * r ** 5 * c / eps2
        boundary = 144 * b / r
        assert 48 * r ** 4 * a >= 9 * r ** 2 * a
        assert 1728 * r ** 5 * c >= 6 * (3 * r ** 2) ** 2 * c
        if r == 4096:
            assert cap == sp.Rational(1, 2 ** 78)
            assert field < sp.Rational(1, 2 ** 1700)
            assert curve < sp.Rational(1, 2 ** 1700)
            assert boundary < 1
        rows.append({'N': n, 'R': r, 'cap_failure_upper': str(cap),
                     'field_failure_upper': str(field), 'curvature_failure_upper': str(curve),
                     'joint_failure_upper': str(cap + field + curve),
                     'boundary_per_Lambda_site_outward': str(boundary),
                     'boundary_less_one': bool(boundary < 1)})
    # Translation-invariant marginal with a genuine rare violation of every hard cap.
    spike_s = 4 * l ** 2
    probability = m / (2 * spike_s ** 2)
    assert probability > 0 and probability * spike_s ** 2 == m / 2 < m
    assert spike_s > l ** 2
    # Constant commuting fields and zero fields are exactly flat, not Clifford/nonzero.
    p = sp.diag(1, 1, -1, -1) / 4
    cross = hs2(p * p + p * p)
    assert cross > 0
    # Shared Gauss density: opposite species cancel, aligned species have cross terms.
    rho = [1] * 32 + [-1] * 32
    assert sum(rho) ** 2 == 0 and sum(x ** 2 for x in rho) == 64
    aligned = [1] * 64
    assert sum(aligned) ** 2 == 4096 and sum(x ** 2 for x in aligned) == 64
    return {'N_R_events': rows, 'rare_spike_probability': str(probability),
            'rare_spike_moment': str(m / 2), 'rare_spike_exceeds_cap': True,
            'flat_commuting_ordered_cross_HS_squared': str(cross),
            'zero_field_also_satisfies_flatness': True,
            'shared_rho_cancellation_squared': 0, 'wrong_per_copy_rho_squared': 64,
            'aligned_totalrho_squared': 4096, 'M_fourthroot_exact': str(m ** sp.Rational(1, 4)),
            'outward_sqrt_s_bound': 12}


def run(request_path: Path, output: Path):
    root = request_path.parent
    request = json.loads(request_path.read_text())
    pins = {}
    for name, pin in request['pins'].items():
        data = Path(pin['path']).read_bytes()
        actual = digest(data)
        if actual != pin['sha256']:
            raise ValueError(f'Pinned bytes changed: {name}')
        pins[name] = actual
    archive = Path(request['input_archive'])
    with zipfile.ZipFile(archive) as frozen:
        profiles_bytes = frozen.read('neutral_body_parent_primitive_profiles_20261012.json')
        profiles = json.loads(profiles_bytes)['primitive_profiles']
        input_members = {name: digest(frozen.read(name)) for name in (
            'neutral_body_parent_primitive_profiles_20261012.json',
            'local_body_leading_proof_20261012.json',
            'native_neutral_body_proof_20261012.json',
            'physical_ground_lift_source_20261011.py')}
    multiplicities = translated_controls()
    native, b = native_fixture(profiles)
    dual = dual_controls(q(profiles[0]['onsite']))
    moments = moment_controls(b)
    result = {'kind': 'INDEPENDENT_EXACT_TRANSLATED_CUBE_NATIVE_IMS_CANONICAL_CHARGED_MOMENT_REVIEW_NOT_GROUND',
              'runtime': {'python': platform.python_version(), 'sympy': sp.__version__,
                          'executable': __import__('sys').executable},
              'input_pins': pins, 'frozen_input_members': input_members,
              'translated_cube_controls': multiplicities, 'native': native,
              'operator_positive_part_and_canonical_controls': dual,
              'actual_family_moment_antecedent_and_event_controls': moments,
              'scope': {'independent_new_consumer': True, 'original_CLI_invoked': False,
                        'eigensolver_or_optimizer_invoked': False, 'ground_vector_computed': False,
                        'actual_family_coercivity': 'Accepted antecedent, not evaluated trial identity',
                        'conditional_density_and_gauge_quotient': 'Analytic disintegration proof in independent proof JSON',
                        'patch_minimum_centre_rigidity_phase_paid': False,
                        'quantum_upper_substituted_for_lower': False},
              'all_exact_controls_passed': True, 'campaign_complete': False}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('Independent exact native translated-cube IMS/dual/charged/moment controls completed naturally.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.output)
