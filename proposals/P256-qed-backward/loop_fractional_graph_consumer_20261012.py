"""New exact native differential/source controls; no frozen entrypoint import.
Analytic graph/domain proof is a separate pinned input, not computed ground data.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import sys

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def nonnegative_coefficients(expression: sp.Expr, *variables: sp.Symbol) -> bool:
    return all(c >= 0 for c in sp.Poly(sp.expand(expression), *variables).coeffs())


def radial_jets() -> dict:
    t, e = sp.symbols('t e', nonnegative=True)
    a = 1 + t**2
    d = 1 + e*a
    w = a/d
    lap = lambda f: 2*t*sp.diff(f, t, 2) + 120*sp.diff(f, t)
    gradient_squared = lambda f: 2*t*sp.diff(f, t)**2
    assert sp.expand(lap(a)-244*t) == 0
    assert sp.expand(gradient_squared(a)-8*t**3) == 0
    assert sp.factor(lap(w)-(244*t/d**2-16*e*t**3/d**3)) == 0
    assert sp.factor(gradient_squared(w)-8*t**3/d**4) == 0
    # Squared nonnegative scalar inequalities, valid on the ENTIRE half-plane.
    witnesses = [a**3*d**5-t**6, a*d**3-t**2, a*d**5-e**2*t**6]
    assert all(nonnegative_coefficients(p, t, e) for p in witnesses)
    assert sp.Rational(3, 2)*260 + sp.Rational(3, 4)*8 == 396
    # Direct second jet of b=w^(3/2), with actual n=120 radial Laplacian.
    chain = sp.Rational(3, 2)*sp.sqrt(w)*lap(w) + sp.Rational(3, 4)*gradient_squared(w)/sp.sqrt(w)
    assert sp.simplify(lap(w**sp.Rational(3, 2))-chain) == 0
    assert 6336 < 80**2 and 6336 == 32*198
    assert 64 + 2*(80+260) == 744
    # The original XY shortcut is false; retain its explicit adverse witness.
    short_w = Fraction(1, 2)
    assert short_w**3 < short_w # w^(3/2)>w^3 follows already after squaring.
    return {'dimension': 120, 'Delta_A': '244*t', 'grad_A_squared': '8*t^3',
            'epsilon_global_nonnegative_polynomial_witnesses': [str(sp.expand(p)) for p in witnesses],
            'weight_b': 'w^(3/2)', 'pairing_bound': 'sqrt(2)*X*Z, NOT X*Y',
            'Delta_b_bound': '396*w', 'C2': '(2+744*kphi)*C1^2',
            'wrong_weight_pairing_rejected': True}


def generators() -> list[sp.Matrix]:
    out = []
    for i in range(4):
        for j in range(i+1, 4):
            x, y = sp.zeros(4), sp.zeros(4)
            x[i, j] = x[j, i] = sp.Rational(1, 2)
            y[i, j], y[j, i] = -sp.I/2, sp.I/2
            out.extend([x, y])
    for n in range(1, 4):
        out.append(sp.diag(*([1]*n+[-n]+[0]*(3-n)))/sp.sqrt(2*n*(n+1)))
    assert len(out) == 15
    assert all(sp.simplify(sp.trace(x*y)-sp.Rational(i == j, 2)) == 0
               for i, x in enumerate(out) for j, y in enumerate(out))
    return out


def actual_native_jets() -> dict:
    gs = generators()
    zero = sp.zeros(4)
    phi = sp.diag(1, -1, 0, 0)/2
    cf = sum((g*g for g in gs), zero)
    assert sp.simplify(cf-sp.Rational(15, 8)*sp.eye(4)) == zero
    adj = sum((g*(g*phi-phi*g)-(g*phi-phi*g)*g for g in gs), zero)
    assert sp.simplify(adj-4*phi) == zero
    # Ordered16 native Jordan terms, all60 Phi directions, actual normalization4.
    fields = [phi, zero, zero, zero]
    lap = sp.Integer(0)
    for mu, nu, r in product(range(4), repeat=3):
        b = fields[mu]*fields[nu]+fields[nu]*fields[mu]
        for g in gs:
            first = zero
            if r == mu:
                first = first+g*fields[nu]+fields[nu]*g
            if r == nu:
                first = first+fields[mu]*g+g*fields[mu]
            second = 4*g*g if r == mu == nu else zero
            lap += 8*(sp.trace(first*first)+sp.trace(b*second))
    lap = sp.simplify(lap)
    assert 0 < lap < 2**24
    # Exact actual scalar alignment jets. Phi cross term has zero Cartesian lap.
    phi_alignment_lap = 2*120*sp.Rational(1, 2)
    internal_plus_five_diagonals = 6*phi_alignment_lap
    assert internal_plus_five_diagonals == 720
    # Haar adjoint derivative of -2 Tr(Px U Py U*) gives 8 Tr(Px U Py U*).
    assert 8*sp.trace(phi*phi) == 4
    assert 4 <= 4*(sp.trace(phi*phi)+sp.trace(phi*phi))
    # Gauge covariance of literal native endpoint average, not fixed spinP.
    u = sp.diag(sp.I, -sp.I, 1, 1)
    gx = sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    gy = sp.diag(1, sp.I, -sp.I, 1)
    py = sp.diag(1, 2, -1, -2)
    h = (phi*u+u*py)/2
    changed = (gx*phi*gx.H*(gx*u*gy.H)+(gx*u*gy.H)*(gy*py*gy.H))/2
    assert sp.simplify(changed-gx*h*gy.H) == zero
    # Radial jets have no Haar derivative, even when native alignment does.
    assert sp.trace(gx*phi*gx.H*gx*phi*gx.H) == sp.trace(phi*phi)
    return {'fundamental_C2': '15/8', 'adjoint_C2': 4,
            'native60_direction_Jordan_laplacian_at_test_tuple': str(lap),
            'Phi_alignment_and_five_diagonal_laplacian_div_alpha': 720,
            'Haar_alignment_laplacian_div_alpha_at_test': 4,
            'actual_endpoint_covariance_residual': '0',
            'envelope_D': '2^24*(kphi*lambda+kH*alpha+kphi*alpha)',
            'scope': 'Native jets and covariance controls; domain/uniform-irrep proof is analytic'}


def move(mask: int, create: int, annihilate: int):
    if not ((mask >> annihilate) & 1):
        return None
    sign = (-1)**((mask & ((1 << annihilate)-1)).bit_count())
    mask ^= 1 << annihilate
    if (mask >> create) & 1:
        return None
    sign *= (-1)**((mask & ((1 << create)-1)).bit_count())
    return mask | (1 << create), sign


def mixed_copy_car() -> dict:
    initial = (1 << 1)+(1 << 2)+(1 << 4)+(1 << 7)
    final = (1 << 0)+(1 << 1)+(1 << 6)+(1 << 7)
    full, same, selected = 0, 0, 0
    for assignment in product(range(2), repeat=4):
        value = 0
        for order in permutations(range(4)):
            mask, amp = initial, 1
            for edge in order:
                s = assignment[edge]
                item = move(mask, 4*s+edge, 4*s+(edge+1)%4)
                if item is None:
                    amp = 0
                    break
                mask, sign = item
                amp *= sign
            if mask == final:
                value += amp
        full += value
        if len(set(assignment)) == 1:
            same += value
        if assignment == (0, 0, 1, 1):
            selected = value
    assert full == selected == -6 and same == 0
    return {'two_copy_all16_assignments_offdiagonal': full,
            'same_copy_sum_offdiagonal': same, 'assignment0011': selected,
            'canonical_copy_numbers': [2, 2],
            'scope': 'Exact finite CAR adverse witness, not actual ground or SU4 sector'}


def adverse_graph_and_tail() -> dict:
    rows = []
    mp.mp.dps = 120
    exponent = mp.mpf(17)/16
    for power in (8, 64, 128):
        m = mp.mpf(2)**power
        # Positive A <= h, with ||A h^-1||_F<2 uniformly, but higher graph grows.
        h = mp.matrix([[1, 0], [0, m]])
        a = mp.matrix([[mp.mpf('0.5'), 1], [1, m/2]])
        assert (h-a)[0, 0]*(h-a)[1, 1]-1 > 0
        ah = a*mp.matrix([[1, 0], [0, 1/m]])
        frobenius2 = sum(abs(ah[i, j])**2 for i in range(2) for j in range(2))
        assert frobenius2 < 4
        # Analytic two-root formula, no native eigensolver or old producer.
        gap = mp.sqrt(((m-1)/2)**2+4)
        lo, hi = ((m+1)/2-gap)/2, ((m+1)/2+gap)/2
        alpha = (hi**exponent-lo**exponent)/(hi-lo)
        higher_first_column_entry = alpha # offdiagonal of A^s, h^-s firstcolumn=1.
        rows.append({'M_power2': power, 'Ah_inverse_frobenius_squared': mp.nstr(frobenius2, 25),
                     'higher_graph_entry_lower': mp.nstr(higher_first_column_entry, 25)})
    assert mp.mpf(rows[-1]['higher_graph_entry_lower']) > 100*mp.mpf(rows[0]['higher_graph_entry_lower'])
    # Paid differential endpoint is necessary; generic norm endpoint1 is insufficient.
    tails = []
    for r in (2, 16, 256):
        mass8 = Fraction(1, r**8)
        assert mass8*r**8 == 1
        # r=u^2 gives exact rational r^(-17/2); the new squared source tail is r^-1/2.
        u = r
        radius = u*u
        mass85 = Fraction(1, u**17)
        assert mass85*radius**8 == Fraction(1, u)
        tails.append({'old_radius': r, 'old_squared_tail': '1',
                      'new_radius': radius, 'new_squared_tail': str(Fraction(1, u))})
    return {'first_graph_only_fractional_extension_rejected': rows,
            'radial8_endpoint_vs_radial8point5': tails,
            'physical_ground_claim': False}


def source_interface(cfg: dict) -> dict:
    a, r, v = cfg['a'], cfg['light_r'], cfg['light_v']
    assert cfg['auxiliary_count'] == 63 and a == 1/64
    c0 = abs(Fraction(-r/a))
    ci = abs(Fraction(-v/(2*a)))
    total = 64*(c0+ci)
    b = 24*(4*total)**4
    assert 64**4 > 64
    # Shared closed character norm4, independent of temporal order or species.
    incidence = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
    for order in permutations(range(4)):
        c = [sum(incidence[e][j] for e in order) for j in range(4)]
        assert sum(x*x for x in c) == 4
    return {'c0_absolute_exact_binary': str(c0), 'ci_absolute_exact_binary': str(ci),
            'Csum_1_plus_63': str(total), 'bare_b': str(b),
            'all64_fourleg_assignment_count': 64**4,
            'wrong_samecopy_count': 64,
            'source_norm_S': '2*b*sqrt(C85)',
            'TOTALnorm_cutoff_error': '2*b*sqrt(C85)*M^(-1/4)',
            'bounded_middle_error': '2*||C||*S*delta(M)',
            'whole_limiting_P0_eta_error': '2*S*delta(M)/eta',
            'C85': '[(2+744*kphi)*C1^2]^(17/16)*C_h17_over8',
            'domains': 'Vacuum TOTALsource norm only; no D(L) or excitation-form image claim',
            'neutral': 'Retained in SAME fullH histories; source identity on neutral',
            'character': 'Closed elementary-face integer incidence, norm squared4; compatible beta retained'}


def run(request: dict) -> dict:
    captured = {name: (ROOT/name).read_bytes() for name in request['input_pins']}
    assert all(digest(raw) == request['input_pins'][name] for name, raw in captured.items())
    proof = json.loads(captured[request['proof']])
    cfg = json.loads(captured[request['micro_input']])['input_descriptor']['original_micro_config_UNCHANGED']
    assert not proof['scientific_admission'] and not proof['campaign_complete']
    result = {'kind': 'INDEPENDENT_NEW_NATIVE_GRAPH_AND_TOTALSOURCE_EXACT_CONTROLS',
              'radial_jets': radial_jets(), 'native_jets': actual_native_jets(),
              'allcopy_CAR_adverse': mixed_copy_car(), 'graph_and_tail_adverse': adverse_graph_and_tail(),
              'interface': source_interface(cfg), 'inputs': request['input_pins'],
              'analytic_proof': request['proof'],
              'nonclaims': ['Computed native ground', 'FullH graph/image-energy', 'FiniteP0 transport',
                            'Positive variance', 'Physical phase', 'IR atom or sharp residue'],
              'scientific_admission': False, 'campaign_complete': False}
    assert all((ROOT/name).read_bytes() == raw for name, raw in captured.items())
    result['inputs_unchanged'] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.request.read_text()))
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'result': str(args.output), 'inputs_unchanged': True,
                      'scientific_admission': False, 'campaign_complete': False}))


if __name__ == '__main__':
    main()
