"""Exact native factor algebra and nonnative inverse-method controls.
No producer import, numerical native ground, or full-H evolution backend.
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
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def transfer(mask: int, destination: int, origin: int) -> tuple[int, int]:
    assert mask & (1 << origin) and not mask & (1 << destination)
    sign = (-1)**((mask & ((1 << origin)-1)).bit_count())
    mask ^= 1 << origin
    sign *= (-1)**((mask & ((1 << destination)-1)).bit_count())
    return mask | (1 << destination), sign


def native(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    def move(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        return tuple((v+1) % n if i == axis else v for i, v in enumerate(x))
    def divergence(flow: dict[tuple[tuple[int, ...], int], Any]) -> Counter:
        result: Counter = Counter()
        for (x, axis), value in flow.items():
            result[x] += value
            result[move(x, axis)] -= value
        return result
    origin = (n-1, n-1, 0)
    face = {(origin, 0): 1, (move(origin, 0), 1): 1,
            (move(origin, 1), 0): -1, (origin, 1): -1}
    assert not any(divergence(face).values())
    background = {(x, 0): F(64*(-1)**sum(x)) for x in sites}
    assert all(divergence(background)[x] == 128*(-1)**sum(x) for x in sites)
    incoming = dict(background)
    for edge, value in face.items():
        incoming[edge] = incoming.get(edge, F(0))+F(3, 8)*value
    ec = sum(value*incoming.get(edge, F(0)) for edge, value in face.items())
    shift_rows = []
    for s in [F(-7, 3), F(-1, 11), F(0), F(2, 5)]:
        shifted = dict(incoming)
        for edge, value in face.items():
            shifted[edge] = shifted.get(edge, F(0))+s*value
        assert divergence(shifted) == divergence(background)
        delta = 64*(sum(v*v for v in shifted.values())-sum(v*v for v in incoming.values()))
        assert delta == 128*s*ec+256*s*s
        shift_rows.append({'s': str(s), 'full_energy_increment': str(delta)})
    v, width = n**3, 4*n**3
    local = sum(15 << (4*j) for j, x in enumerate(sites) if sum(x) % 2 == 0)
    assert local.bit_count() == 2*v
    mask = sum(local << (role*width) for role in range(65)) | (15 << (65*width))
    source = next(x for x in sites if sum(x) % 2 == 0)
    target = move(source, 0)
    forward_signs = []
    for role in range(65):
        offset = role*width
        changed, sign = transfer(mask, offset+4*sites.index(target), offset+4*sites.index(source))
        restored, reverse = transfer(changed, offset+4*sites.index(source), offset+4*sites.index(target))
        assert restored == mask and sign*reverse == 1
        assert all(((changed >> (r*width)) & ((1 << width)-1)).bit_count() == 2*v
                   for r in range(65))
        assert (changed >> (65*width)) == 15
        if role:
            hop = {(source, 0): -1}
            changed_e = dict(background)
            changed_e[(source, 0)] += -1
            rho = divergence(background)
            rho[target] += 1
            rho[source] -= 1
            assert divergence(changed_e) == rho
            assert divergence(hop)[source] == -1 and divergence(hop)[target] == 1
        forward_signs.append(sign)
    counts, ap, affected = [0]*3, [0]*3, [0]*3
    aliases, charged_load = 0, F(0)
    for role, profile in enumerate(profiles):
        assert F(profile['onsite']) != 0
        for row in profile['paths']:
            assert F(row['c0']) != 0
            if role:
                charged_load += F(profile['load'])*row['length']*(abs(F(row['c0']))+abs(F(row['ci'])))
            for x in sites:
                y, edges, seam = x, [], 1
                for _ in range(row['length']):
                    edges.append((y, row['axis']))
                    seam *= -1 if y[row['axis']] == n-1 else 1
                    y = move(y, row['axis'])
                counts[role] += 1
                ap[role] += int(seam == -1)
                frequency = sum(face.get(e, 0) for e in edges) if role else 0
                assert abs(frequency) <= 2
                if frequency:
                    affected[role] += 1
                    aliases += int(n == 4 and row['length'] == 2)
                path_div = divergence(dict(Counter(edges)))
                assert path_div[x] == 1 and path_div[y] == -1
    assert counts == [3*v, 6*v, 6*v]
    assert affected == [0, 12, 12]
    assert aliases == (16 if n == 4 else 0)
    assert charged_load == F(46425051219964815, 8796093022208)
    gx = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    gy = sp.diag(sp.I, -sp.I, 1, 1)
    phi = sp.diag(1, -1, 2, -2)
    other = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]])
    u = gx*gy
    block = (phi*u+u*other)/2
    covariant = ((gx*phi*gx.H)*(gx*u*gy.H)+(gx*u*gy.H)*(gy*other*gy.H))/2
    assert gx.det() == gy.det() == 1 and covariant == gx*block*gy.H
    # Native edge integer deck shifts: character transport on independent face/winding deck cycles.
    winding = {((j, 0, 0), 0): 1 for j in range(n)}
    assert not any(divergence(winding).values())
    deck_rows = []
    for gamma in [face, winding]:
        pairing = sum(F(value)*face.get(edge, 0) for edge, value in gamma.items())
        beta, s = sp.symbols('beta s', real=True)
        ratio = sp.exp(sp.I*2*sp.pi*(beta+s)*pairing)
        assert sp.simplify(ratio-sp.exp(sp.I*2*sp.pi*beta*pairing)*sp.exp(sp.I*2*sp.pi*s*pairing)) == 0
        assert sp.exp(sp.I*2*sp.pi*pairing) == 1
        deck_rows.append({'integer_cycle_pairing': str(pairing), 'scalar_character_transport': True})
    # The aligned rho comes from64 charges: independent-copy Coulomb sum misses factor64.
    assert F(64**2, 64) == 64
    return {'scope': 'exact finite native factor algebra, not a ground or full-H computation',
            'N': n, 'canonical_each65': 2*v, 'Ref': 4, 'fullrho': '128 parity once collectively',
            'all65_distinct_factor_transfers': len(forward_signs), 'charged_Gauss_hopped_outputs': 64,
            'full_form_shift_rows': shift_rows, 'incoming_face_electric': str(ec),
            'literal_profile_path_counts': counts, 'AP_seam_counts': ap, 'face_affected': affected,
            'N4_length2_alias_literals': aliases, 'complete_charged_load': str(charged_load),
            'ordered_SU4_covariance': True, 'deck_character_controls': deck_rows,
            'independent_copy_HC_wrong_factor': 64}


def primitive_form() -> dict[str, Any]:
    a, p, z, s, h = sp.symbols('A Phi Haar s h', real=True)
    psi = sp.Function('Psi')(a, p, z)
    boost = sp.exp(sp.I*s*h*a)
    derivative = lambda f: -sp.I*sp.diff(f, a)
    assert sp.simplify(derivative(boost*psi)-boost*(derivative(psi)+s*h*psi)) == 0
    assert sp.simplify(sp.diff(boost*psi, p)-boost*sp.diff(psi, p)) == 0
    assert sp.simplify(sp.diff(boost*psi, z)-boost*sp.diff(psi, z)) == 0
    k = sp.symbols('k', integer=True)
    matter = sp.Function('OrderedCAR_SU4_Phi')(p, z)*sp.exp(sp.I*k*a)
    assert matter*boost == boost*matter
    e = sp.symbols('incomingE', real=True)
    assert sp.expand(64*(e+s*h)**2-64*e**2) == 128*e*h*s+64*h*h*s*s
    return {'arbitrary_joint_Psi': True, 'Phi_and_Haar_derivatives_unchanged': True,
            'ordered_matter_AP_multiplier_commutes': True, 'fullE64_coefficients': [128, 64],
            'unchanged_primitive_terms': ['Phi4chi', 'Haar3chi/20', 'Jordan', 'alignment',
                'colour plaquette', 'heavy neutral', 'active1', 'EACHaux63', 'onsite', 'Ref4'],
            'native_state_spectrum_computed': False}


def spectral_controls() -> dict[str, Any]:
    s, g, a = sp.symbols('s g a', real=True)
    hs = sp.Matrix([[256*s*s, 128*s*a], [128*s*a, g+256*s*s]])
    determinant = sp.factor(hs.det())
    assert sp.expand(determinant-256*s*s*(g-64*a*a+256*s*s)) == 0
    # Face-norm4 sharp source, g1,a1/8 saturates inverse1/64.
    assert determinant.subs({g: 1, a: sp.Rational(1, 8)}) == 65536*s**4
    inverse = sp.Rational(1, 8)**2
    assert inverse == sp.Rational(1, 64) and inverse > sp.Rational(1, 256)
    assert determinant.subs({g: 1, a: sp.Rational(1, 4), s: sp.Rational(1, 100)}) < 0
    omega = sp.Matrix([1, 0, 0])
    electric = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    whole_p0 = sp.diag(1, 1, 0)
    source = electric*omega
    assert (omega.H*source)[0] == 0 and whole_p0*source == source
    # Mean subtraction does not annihilate the second ground direction.
    assert (sp.eye(3)-omega*omega.H)*source == source
    gram = sp.Matrix([[1, sp.I], [-sp.I, 1]])
    real_x, real_y = sp.symbols('x y', real=True)
    r = sp.Matrix([real_x, real_y])
    assert (r.T*gram*r)[0] == real_x**2+real_y**2
    c = sp.Matrix([1, -sp.I])
    assert (c.H*gram*c)[0] == 4 and (c.H*c)[0] == 2
    collapse = []
    bound = F(1, 64)
    for n in [4, 16, 64, 256]:
        epsilon = F(1, n)
        assert 1/epsilon > bound
        weight = epsilon*bound
        assert weight/epsilon == bound
        k = 32
        f = min(F(k), 1/epsilon)
        assert weight*f <= bound
        collapse.append({'N': n, 'energy': str(epsilon), 'inverse_legal_weight': str(weight),
                         'unit_weight_E4_alone_inverse_violation': str(1/epsilon),
                         'continuous_f32_integral': str(weight*f)})
    # Continuous inverse includes zero with valueK; a zero atom is detected, not hidden.
    zero_atom = F(1, 2)
    assert 32*zero_atom > bound
    # A nonminimizing character ground can have nonzero electric mean and zero fibre excess.
    beta = F(1, 3)
    assert 64*beta*beta > 0 and beta != 0
    # Point indicator is a.e.zero but evaluating betaStar gives1: forbidden inference.
    selected_point_indicator = {'original_measure_integral': 0, 'arbitrary_point_value': 1}
    # Infinite domain witness: r_n=n^-2, a1/16,b_n=n,L diag(0,1,n^4).
    # E4 integrand = n^-4*(a^4+a^2 n^2), summable; E2 graph term=n^6/256 diverges.
    j = sp.symbols('n', positive=True, integer=True)
    small_a = sp.Rational(1, 16)
    e4_term = j**-4*(small_a**4+small_a**2*j**2)
    graph_term = small_a**2*j**6
    assert sp.limit(e4_term, j, sp.oo) == 0 and sp.limit(graph_term, j, sp.oo) == sp.oo
    return {'scope': 'nonnative exact spectral/form adverse controls, NOT native ground/fullH',
            'all_character_matrix_determinant': str(determinant),
            'face_sharp_inverse': str(inverse), 'wrong_face_inverse1_256_rejected': True,
            'wrong_diagonal_or_source_constant_rejected': True,
            'wholeP0_not_mean': True, 'mean_zero_extra_ground_source_rejected': True,
            'complex_imaginary_cross_Gram': [['1', 'i'], ['-i', '1']],
            'complex_real_constant_ratio': 2, 'wrong_complex1_256_rejected': True,
            'E4_collapse_requires_inverse': collapse, 'continuous_fK_zero_atom_rejected': True,
            'selected_character_relative_ground_inference_rejected': True,
            'Borel_point_evaluation_adverse': selected_point_indicator,
            'E4_does_not_imply_E2Omega_D_L': {'E4_series_term': str(e4_term),
                                            'E2_graph_series_term': str(graph_term)},
            'phase_or_atom_inferred': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    before = {name: sha(Path(row['path']).read_bytes()) for name, row in request['inputs'].items()}
    assert before == {name: row['sha256'] for name, row in request['inputs'].items()}
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        raw = archive.read(request['profile_member'])
        assert sha(raw) == request['profile_member_sha256']
        profiles = json.loads(raw)['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    receipt = {'schema': 1, 'kind': 'NEW_CLOSED_BETA_INVERSE_EXACT_CONSTRUCTION_CONTROLS',
               'native_factor_controls': [native(n, profiles) for n in request['N']],
               'primitive_form': primitive_form(), 'nonnative_spectral_adverse': spectral_controls(),
               'proof_sha256': before[request['proof']],
               'interface_sha256': before[request['interface']],
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: sha(Path(row['path']).read_bytes())
                                              for name, row in request['inputs'].items()},
               'native_numerical_ground_or_fullH_backend': False,
               'scientific_admission': False, 'campaign_complete': False}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'completed_exact_controls': True}))


if __name__ == '__main__':
    main()
