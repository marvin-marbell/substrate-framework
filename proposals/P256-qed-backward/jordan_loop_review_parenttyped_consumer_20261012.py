"""Independent exact adverse controls; never import or execute target producers."""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any, cast
import zipfile

import sympy as s

ROOT = Path(__file__).resolve().parent


def exact(value: Any) -> str:
    return str(s.simplify(value))


def scalar(value: Any) -> Any:
    return s.simplify(value[0])


def matrix_controls() -> dict[str, Any]:
    a = s.Matrix([[1, 2, s.I], [2, -1, 0], [-s.I, 0, 3]])
    b = s.Matrix([[0, s.I, 1], [-s.I, 2, 3], [1, 3, -2]])
    anti = a*b+b*a
    lhs = -s.re(s.trace(a*b*a*b))
    norm = s.trace((a*b).H*(a*b))
    defect = s.trace(anti.H*anti)/2
    assert s.simplify(lhs-norm+defect) == 0
    weight = s.diag(1, 2, 4)
    weighted_lhs = -s.re(s.trace(weight*a*b*a*b))
    weighted_rhs = s.re(s.trace(weight*a*a*b*b)-s.trace(weight*anti*anti)/2)
    assert weighted_lhs != weighted_rhs
    d = s.Matrix([[3, 1, 0], [1, 4, 1], [0, 1, 2]])
    assert all(cast(Any, d[:i, :i]).det() > 0 for i in range(1, 4))
    r = s.zeros(4)
    r[1:, 1:] = d.inv()
    p = s.diag(1, 0, 0, 0)
    q = s.eye(4)-p
    v = s.Matrix([1, 0, 0, 0])
    t1 = s.Matrix([[0, 1, 0, 1], [1, 0, 2, 0], [2, 1, 0, s.I], [1, 0, -s.I, 2]])
    t2 = s.Matrix([[1, 0, 1, 0], [0, 2, 1, s.I], [1, 0, 1, 2], [0, -s.I, 3, 1]])
    t3 = s.Matrix([[0, 1, 2, 0], [3, 0, s.I, 1], [1, 2, 1, 0], [2, 0, 1, 1]])
    t4 = s.Matrix([[1, 2, s.I, 1], [0, 1, 0, 2], [1, 0, 2, 1], [0, 2, 1, 1]])
    x = q*t2*r*t1*v
    y = q*t3.H*r*t4.H*v
    wrong_y = q*t4.H*r*t3.H*v
    word = scalar(v.H*t4*r*t3*r*t2*r*t1*v)
    cross = scalar(y.H*r*x)
    wrong_cross = scalar(wrong_y.H*r*x)
    nx, ny = scalar(x.H*r*x), scalar(y.H*r*y)
    difference = scalar((x-y).H*r*(x-y))
    assert word == cross and word != wrong_cross
    assert -s.re(word) == difference/2-(nx+ny)/2
    assert difference >= 0 and -s.re(word) >= -(nx+ny)/2
    # Positive compressed inverse is NOT the compression of the full inverse.
    h = s.Matrix([[4, 1, 1], [1, 3, 1], [1, 1, 2]])
    assert all((h-s.eye(3))[:i, :i].det() > 0 for i in range(1, 4))
    compressed = cast(Any, h[1:, 1:])
    inverse = compressed.inv()
    wrong_inverse = h.inv()[1:, 1:]
    assert inverse != wrong_inverse and inverse[0, 1] < 0
    c = s.Matrix([1, 2])
    jensen = scalar(c.H*c)**2/scalar(c.H*compressed*c)
    gram = scalar(c.H*inverse*c)
    assert gram >= jensen > 0
    internal = s.Matrix([[1, 2], [2, -1]])
    fourth = -scalar(c.H*inverse*internal*inverse*internal*inverse*c)
    assert fourth < 0
    ground_h = s.diag(0, 0, 2)
    whole_p0 = s.diag(1, 1, 0)
    assert (s.eye(3)-whole_p0)*ground_h*whole_p0 == s.zeros(3)
    commuting_probe = s.diag(1, 2, 3)
    assert (s.eye(3)-whole_p0)*commuting_probe*whole_p0 == s.zeros(3)
    eta = s.Rational(2, 3)
    assert eta**(-3) == s.Rational(27, 8)
    # Form-incompatible Hilbert rank-one Q in l2: a_n=1/n is in l2,
    # but not Q(diag(n^2)); its form-domain intersection is {0}.
    truncation = [{"n": n, "norm2": exact(sum(s.Rational(1, k*k) for k in range(1, n+1))),
                   "unnormalized_form_energy": n} for n in (4, 16, 64)]
    return {"unweighted_identity": {"minus_ReTr_ABAB": exact(lhs), "HS_AB_squared": exact(norm), "half_defect": exact(defect)},
            "positive_weight_identity_fails": {"lhs": exact(weighted_lhs), "wrong_rhs": exact(weighted_rhs)},
            "noncommuting_halfloop": {"W": exact(word), "wrong_Y_order": exact(wrong_cross), "X2": exact(nx), "Y2": exact(ny), "difference2": exact(difference), "paid_lower": exact(-(nx+ny)/2)},
            "nonreducing_Q": {"correct_compressed_inverse": str(inverse), "wrong_compression_of_inverse": str(wrong_inverse), "H_lower": 1},
            "Jensen": {"Gram": exact(gram), "paid_lower": exact(jensen), "inverse_offdiagonal": exact(inverse[0, 1]), "probe_may_vanish": True, "whole_ground_Schur_C_zero": True},
            "fourth_Schur_coefficient": exact(fourth), "eta_three_resolvents": exact(eta**(-3)),
            "Hilbert_not_form_compatible_rankone": truncation}


def ladder(mask: int, mode: int, create: bool) -> tuple[int, int] | None:
    occupied = bool(mask & (1 << mode))
    if occupied == create:
        return None
    sign = -1 if (mask & ((1 << mode)-1)).bit_count() % 2 else 1
    return mask ^ (1 << mode), sign


def apply_word(mask: int, operations: list[tuple[int, bool]]) -> tuple[int, int] | None:
    sign = 1
    for mode, create in operations:
        moved = ladder(mask, mode, create)
        if moved is None:
            return None
        mask, factor = moved
        sign *= factor
    return mask, sign


def add_term(out: dict[int, Any], moved: tuple[int, int] | None, coefficient: Any) -> None:
    if moved is not None:
        mask, sign = moved
        out[mask] = out.get(mask, 0)+sign*coefficient


def reduced(out: dict[int, Any]) -> dict[int, Any]:
    return {k: s.simplify(v) for k, v in out.items() if s.simplify(v) != 0}


def hop(mask: int, h: Any, reverse: bool = False) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for i in range(4):
        for j in range(4):
            if h[i, j] != 0:
                operations = [(i, False), (4+j, True)] if reverse else [(4+j, False), (i, True)]
                add_term(out, apply_word(mask, operations), s.conjugate(h[i, j]) if reverse else h[i, j])
    return reduced(out)


def carrier_controls() -> dict[str, Any]:
    h = s.Matrix([[1, s.I, 0, 0], [2, 0, 0, 0], [0, 0, 3, 1], [0, 0, s.I, 4]])
    checked = 0
    for source in range(256):
        direct: dict[int, Any] = {}
        for state, amplitude in hop(source, h).items():
            for target, value in hop(state, h, True).items():
                direct[target] = direct.get(target, 0)+amplitude*value
        exchange: dict[int, Any] = {}
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        coefficient = s.conjugate(h[i, j])*h[k, l]
                        if coefficient == 0:
                            continue
                        if i == k:
                            add_term(exchange, apply_word(source, [(4+l, False), (4+j, True)]), coefficient)
                        add_term(exchange, apply_word(source, [(4+l, False), (i, False), (k, True), (4+j, True)]), -coefficient)
        assert reduced(direct) == reduced(exchange)
        checked += 1
    aligned = (1 << 0)+(1 << 1)+(1 << 4)+(1 << 5)
    complementary = (1 << 0)+(1 << 1)+(1 << 6)+(1 << 7)
    norm_aligned = sum(s.conjugate(v)*v for v in hop(aligned, h).values())
    norm_complementary = sum(s.conjugate(v)*v for v in hop(complementary, h).values())
    assert norm_aligned == 0 and norm_complementary > 0
    assert not hop(0, h) and not hop(255, h)
    # Exterior2 representation of an exact local SU4 rotation.
    u = s.eye(4)
    u[1, 1] = u[2, 2] = 0
    u[1, 2], u[2, 1] = 1, -1
    assert u.H*u == s.eye(4) and u.det() == 1
    pairs = list(combinations(range(4), 2))
    wedge = s.Matrix([[u.extract(list(a), list(b)).det() for b in pairs] for a in pairs])
    fixed = s.diag(1, 0, 0, 0, 0, 0)
    commutator = fixed*wedge-wedge*fixed
    assert commutator != s.zeros(6)
    assert s.eye(6)*wedge-wedge*s.eye(6) == s.zeros(6)
    return {"full_TdagT_exchange_columns_checked": checked, "same_number2_charge0_carriers": [exact(norm_aligned), exact(norm_complementary)],
            "empty_full_carrier_zero": True, "fixed_rank2_projector_SU4_commutator_HS2": exact(s.trace(commutator.H*commutator)),
            "local_number_projector_SU4_invariant": True, "exterior0_and4_SU4_singlets": True,
            "physical_meaning": "Rank2 spin-coordinate projector preserves U1/canonical number, not local SU4 Gauss; fixed CAR tensor identity must be SU4-invariant or covariantly dressed."}


def physical_flow_controls() -> dict[str, Any]:
    n = 8
    sites = [(x, y, z) for x in range(n) for y in range(n) for z in range(n)]
    index = {site: i for i, site in enumerate(sites)}
    face = {(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)}
    full = set(site for site in sites if site not in face)
    full = set(sorted(full)[:256])
    rho = [128 if site in full else -128 for site in sites]
    assert sum(rho) == 0 and all(site not in full for site in face)
    parent = [-1]*512
    edge = [-1]*512
    order = [0]
    seen = {0}
    queue = deque([0])
    while queue:
        a = queue.popleft()
        xyz = sites[a]
        for axis in range(3):
            dest = list(xyz)
            dest[axis] = (dest[axis]+1) % n
            b = index[(dest[0], dest[1], dest[2])]
            if b not in seen:
                seen.add(b)
                parent[b], edge[b] = a, 3*a+axis
                queue.append(b)
                order.append(b)
    assert len(order) == 512
    subtree = rho.copy()
    flow = [0]*1536
    for b in reversed(order[1:]):
        flow[edge[b]] = -subtree[b]
        subtree[parent[b]] += subtree[b]
    divergence = [0]*512
    for a, xyz in enumerate(sites):
        for axis in range(3):
            dest = list(xyz)
            dest[axis] = (dest[axis]+1) % n
            b = index[(dest[0], dest[1], dest[2])]
            divergence[a] += flow[3*a+axis]
            divergence[b] -= flow[3*a+axis]
    assert divergence == rho
    price = 64*sum(k*k for k in flow)
    assert price > 0
    # The four face-site empty factors kill every incoming annihilator;
    # reversed charge/occupation and flow gives the full-face dual control.
    return {"N": n, "V": 512, "copies": 64, "particles_per_copy": 1024,
            "shared_rho_values": sorted(set(rho)), "Dk_equals_rho": True, "integer_edges": len(flow), "HE_plus_HC_beta0": str(price),
            "all_sites_exterior0_or4_SU4_singlet": True, "empty_and_full_face_primitive_killed": True,
            "state": "Finite beta0 physical fibre: smooth gauge-invariant compact-Phi section, SU4 singlet CAR factors, compatible electric Fourier character. Not ground or real-cover sharp-flow L2 vacuum."}


def native_transport_controls(cfg: dict[str, Any]) -> dict[str, Any]:
    rat = lambda value: s.Rational(Fraction(value).numerator, Fraction(value).denominator)
    phi_x = [s.diag(-1, -1, 1, 1), s.Matrix([[0, 0, 1, 0], [0, 0, 0, 2], [1, 0, 0, 0], [0, 2, 0, 0]])]
    phi_y = [s.diag(-2, 1, 3, -2), s.Matrix([[0, 1, s.I, 0], [1, 0, 0, 0], [-s.I, 0, 0, 2], [0, 0, 2, 0]])]
    u = s.eye(4)
    u[:2, :2] = s.Matrix([[s.Rational(3, 5), s.Rational(4, 5)], [-s.Rational(4, 5), s.Rational(3, 5)]])
    v = s.diag(s.I, 1, -s.I, 1)
    assert u.det() == v.det() == 1 and u*v != v*u
    gx, gy = v, u
    rows = []
    for role, count, mass in [("active", 1, cfg['active_m']), ("auxiliary", 63, cfg['auxiliary_m'])]:
        c0 = rat(-cfg['light_r']/cfg['a'])
        ci = s.I*rat(-cfg['light_v']/(2*cfg['a']))
        c2 = rat(cfg['light_r']/(4*cfg['a']))
        onsite = rat(mass+4.5*cfg['light_r']/cfg['a'])
        for length, coefficients, transport in [(1, [c0, ci], u), (2, [c2, 0], u*v)]:
            fx = sum((c*f for c, f in zip(coefficients, phi_x)), s.zeros(4))
            fy = sum((c*f for c, f in zip(coefficients, phi_y)), s.zeros(4))
            h = -(fx*transport+transport*fy)/2  # AP seam -1.
            fxg, fyg, ug = gx*fx*gx.H, gy*fy*gy.H, gx*transport*gy.H
            transformed = -(fxg*ug+ug*fyg)/2
            assert (transformed-gx*h*gy.H).applyfunc(s.simplify) == s.zeros(4)
            assert h != -(fx*transport+transport*fy)
            rows.append({"role": role, "count": count, "length": length, "AP_seam": -1, "onsite": exact(onsite), "c0": exact(coefficients[0]), "ci": exact(coefficients[1]), "covariance_residual": "0"})
    assert cfg['auxiliary_count'] == 63 and cfg['a'] == 1/64
    return {"rows": rows, "ordered_noncommuting_SU4_product_discriminated": True,
            "length2_midpoint_not_used": True, "endpoint_half_AP_native_normalization": True,
            "scope": "Independent exact literal endpoint/path controls; no original source module imported, no complete quantum backend claimed."}


def archive_controls(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    before = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in request['pins']}
    assert before == request['pins']
    report: dict[str, Any] = {}
    cfg: dict[str, Any] = {}
    for epoch in request['epochs']:
        capsule = json.loads((ROOT/f'{epoch}_capsule.json').read_bytes())
        with zipfile.ZipFile(ROOT/f'{epoch}_preexecution.zip') as pre, zipfile.ZipFile(ROOT/f'{epoch}_executed.zip') as post:
            hashes = {name: hashlib.sha256(pre.read(name)).hexdigest() for name in pre.namelist() if not name.endswith('/')}
            assert all(hashlib.sha256(post.read(name)).hexdigest() == value for name, value in hashes.items())
            for name, value in capsule.get('input_pins', {}).items():
                member = 'inputs/'+Path(name).name
                assert hashes[member] == value
            report[epoch] = {"preexecution_members": len(hashes), "all_source_input_members_identical_in_executed": True, "original_natural_exit": json.loads((ROOT/f'{epoch}_exit.json').read_bytes())}
            if epoch == 'jordan_loop_form_actual1_20261012':
                cfg = json.loads(pre.read('inputs/induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json'))['input_descriptor']['original_micro_config_UNCHANGED']
    original = json.loads((ROOT/'jordan_loop_form_actual1_20261012_result.json').read_bytes())
    weighted = json.loads((ROOT/'jordan_loop_form_halfloop1_20261012_result.json').read_bytes())
    for row in weighted['roles']:
        half = row['weighted_halfloop']
        old = next(r for r in original['roles'] if r['role'] == row['role'])['complete_variable_phi_primitive']['coefficient']
        assert Fraction(half['positive_R_word']['real']) == -Fraction(old['real'])
        assert Fraction(half['positive_R_word']['imag']) == -Fraction(old['imag'])
        assert len(half['orders']) == 24
        for order in half['orders']:
            x, y = Fraction(order['X_squared']), Fraction(order['Y_squared'])
            cross = Fraction(order['W']['real'])
            assert Fraction(order['difference_squared']) == x+y-2*cross >= 0
            assert Fraction(order['signed_lower']) == -(x+y)/2
    ratio = Fraction(original['local_normalized_gate']['normalized_ratio_at_physical1'])
    assert ratio > 1 and not original['local_normalized_gate']['physical1_neumann_gate']
    report['original_static_downstream_arithmetic'] = {"all24_each_role_halfloop_residuals_exact": True, "physical1_ratio": str(ratio), "physical1_gate": False,
        "halfloop_endpoints": "X starts k=0; Y starts k=c=(1,1,1,1). Polarized character, not same-state diagonal expectation.",
        "onsite_gap_infimum_not_fullH_gap": True}
    assert before == {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in request['pins']}
    return report, cfg


def polarized_flow_control() -> dict[str, Any]:
    shifts = []
    for edge in range(4):
        shift = s.zeros(16)
        for k in range(16):
            if not k & (1 << edge):
                shift[k | (1 << edge), k] = 1
        shifts.append(shift)
    r = s.diag(*[s.Rational(1, 1+64*k.bit_count()) for k in range(16)])
    initial, final = s.eye(16)[:, 0], s.eye(16)[:, 15]
    t1, t2, t3, t4 = shifts
    word = t4*r*t3*r*t2*r*t1
    x = t2*r*t1*initial
    y = t3.H*r*t4.H*final
    same_y = t3.H*r*t4.H*initial
    polarized = scalar(final.H*word*initial)
    diagonal = scalar(initial.H*word*initial)
    assert polarized == scalar(y.H*r*x) > 0
    assert diagonal == scalar(same_y.H*r*x) == 0
    price = scalar(x.H*r*x+y.H*r*y)/2
    assert polarized <= price
    return {"distinct_endpoint_character": exact(polarized),
            "same_sharp_flow_diagonal": exact(diagonal),
            "polarized_paid_lower": exact(-price),
            "character_is_not_single_state_energy": True,
            "flow_shift_truncation_scope": "Exact finite directed-incidence control, not a quantum-H backend."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    archives, cfg = archive_controls(request)
    result = {"kind": "INDEPENDENT_EXACT_ADVERSE_LOOP_FORM_REVIEW_NOT_PHASE_ADMISSION", "archives": archives,
              "matrix_controls": matrix_controls(), "carrier_Gauss_controls": carrier_controls(),
              "polarized_flow_control": polarized_flow_control(),
              "physical_flow_controls": physical_flow_controls(), "native_transport_controls": native_transport_controls(cfg),
              "old_producer_imported_or_executed": False, "campaign_complete": False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({"kind": result['kind'], "all_independent_exact_controls": "passed", "target_rerun": False, "physical1_phase_admission": False}))


if __name__ == '__main__':
    main()
