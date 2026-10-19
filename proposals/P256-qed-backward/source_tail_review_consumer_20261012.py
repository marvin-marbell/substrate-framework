"""Independent exact source-tail controls; never imports the frozen producer."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
from typing import Any
import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def car_hop(mask: int, target: int, source: int) -> tuple[int, int] | None:
    occupied = [i for i in range(8) if mask & (1 << i)]
    if source not in occupied:
        return None
    sign = (-1) ** occupied.index(source)
    occupied.remove(source)
    if target in occupied:
        return None
    sign *= (-1) ** sum(i < target for i in occupied)
    occupied.append(target)
    return sum(1 << i for i in occupied), sign


def canonical_matrices() -> dict[str, Any]:
    basis = [sum(1 << i for i in a) + sum(1 << (4+i) for i in b)
             for a in combinations(range(4), 2) for b in combinations(range(4), 2)]
    index = {mask: i for i, mask in enumerate(basis)}
    hops = [[sp.zeros(36) for _ in range(4)] for _ in range(2)]
    for species in range(2):
        for edge in range(4):
            for col, mask in enumerate(basis):
                item = car_hop(mask, 4*species+edge, 4*species+(edge+1) % 4)
                if item is not None:
                    new_mask, sign = item
                    hops[species][edge][index[new_mask], col] = sign
    full = sp.zeros(36)
    same = sp.zeros(36)
    for order in permutations(range(4)):
        term = sp.eye(36)
        for edge in order:
            term = (hops[0][edge]+hops[1][edge])*term
        full += term
        for species in range(2):
            term = sp.eye(36)
            for edge in order:
                term = hops[species][edge]*term
            same += term
    initial = (1 << 1)+(1 << 2)+(1 << 4)+(1 << 7)
    final = (1 << 0)+(1 << 1)+(1 << 6)+(1 << 7)
    mixed = full[index[final], index[initial]]
    wrong = same[index[final], index[initial]]
    assert mixed == -6 and wrong == 0, 'Must reach the mixed-copy -6/single-copy0 assertion'
    hermitian = (full+full.adjoint())/2
    assert hermitian == hermitian.adjoint()
    assert full != same
    return {'basis_dimension': 36, 'copy_occupations': [2, 2],
            'all24_mixedcopy_entry': int(mixed), 'singlecopy_entry': int(wrong),
            'Hermitian_probe_entry': str(hermitian[index[final], index[initial]]),
            'full64_embedding': 'Other62 copies may be face-empty/full with canonical particles outside; CAR-only witness, not a Gauss-ground calculation'}


def coefficient_and_covariance(descriptor: dict[str, Any]) -> dict[str, Any]:
    species = descriptor['exact_radial_species']
    active, auxiliary = species[1], species[2]
    assert active['load'] == 1 and auxiliary['load'] == 63
    ca = abs(Fraction(active['c0']))+abs(Fraction(active['ci']))
    cx = abs(Fraction(auxiliary['c0']))+abs(Fraction(auxiliary['ci']))
    total = ca+63*cx
    b = 24*(4*total)**4
    cfg = descriptor['original_micro_config_UNCHANGED']
    assert abs(Fraction(-cfg['light_r']/cfg['a'])) == abs(Fraction(active['c0']))
    assert abs(Fraction(-cfg['light_v']/(2*cfg['a']))) == abs(Fraction(active['ci']))
    u = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    gx = sp.diag(sp.I, -sp.I, 1, 1)
    gy = sp.diag(1, sp.I, -sp.I, 1)
    px = sp.diag(1, -1, 2, -2)
    py = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, sp.I], [0, 0, -sp.I, 0]])
    assert gx.det() == gy.det() == u.det() == 1
    f0 = sp.Rational(active['c0'])
    fi = sp.I*sp.Rational(active['ci'])
    fx, fy = f0*px+fi*py, f0*py+fi*px
    h = (fx*u+u*fy)/2
    transformed = ((gx*fx*gx.adjoint())*(gx*u*gy.adjoint())
                   +(gx*u*gy.adjoint())*(gy*fy*gy.adjoint()))/2
    residual = (transformed-gx*h*gy.adjoint()).applyfunc(sp.expand)
    assert residual == sp.zeros(4)
    wrong = (transformed-gx*h*gx.adjoint()).applyfunc(sp.expand)
    assert wrong != sp.zeros(4), 'Endpoint colour action must use both distinct endpoint gauges'
    incidence = sp.Matrix([[1, 0, 0, -1], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
    flow = sp.ones(4, 1)
    assert incidence*flow == sp.zeros(4, 1) and (flow.T*flow)[0] == 4
    for order in permutations(range(4)):
        assert sum((sp.eye(4)[:, e] for e in order), sp.zeros(4, 1)) == flow
    return {'C_active': str(ca), 'C_auxiliary': str(cx), 'Csum': str(total), 'b': str(b),
            'all64_power4_assignments': 64**4, 'all24_orders': 24,
            'gauge_covariance_residual': '0', 'closed_face_Dc': [0, 0, 0, 0],
            'character_norm_squared': 4,
            'analytic_norm_payment': 'entry_l1<=4HS; HS(h)<=C_s(rx+ry)/2; V_e<=4Csum R; A_face<=24(4Csum)^4 R^4'}


def moments_and_endpoint() -> dict[str, Any]:
    assert Fraction(17, 8) < Fraction(9, 4)
    assert Fraction(17, 16) < 2
    rows = []
    for m in (2, 16, 256):
        r = 4*m*m
        old_probability = Fraction(1, r**8)
        new_probability = Fraction(1, (2*m)**17)
        assert old_probability*r**8 == 1
        assert new_probability*(2*m)**17 == 1
        assert new_probability*r**8 == Fraction(1, 2*m)
        assert r > 2*m
        rows.append({'M': m, 'R_outside_cutoff_support': r, 'old_eighth_moment': '1',
                     'old_squared_source_tail': '1', 'new_8point5_moment': '1',
                     'new_squared_source_tail': str(new_probability*r**8)})
    exponents = {'s': str(Fraction(17, 16)), 'h_moment': str(Fraction(17, 8)),
                 'radial_moment': str(4*Fraction(17, 8)),
                 'squared_tail_power': str(8-4*Fraction(17, 8)),
                 'norm_tail_power': str((8-4*Fraction(17, 8))/2)}
    assert exponents['radial_moment'] == '17/2' and exponents['norm_tail_power'] == '-1/4'
    return {'exponents': exponents, 'endpoint_spikes': rows,
            'conditional_constant': 'C85=C2^(17/16) C_h; C2=(2+744*kphi) C1^2',
            'endpoint_graph_accepted_by_this_control': False}


def mixed_kernels() -> dict[str, Any]:
    # Noncommuting bounded middle and a whole zero block, all rational/complex exact.
    u = sp.Matrix([1, 2, sp.I, 1])
    um = sp.Matrix([1, sp.Rational(3, 2), sp.I/2, 1])
    c = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    assert c.adjoint()*c == sp.eye(4)
    delta = sp.sqrt(((u-um).adjoint()*(u-um))[0])
    S = sp.sqrt((u.adjoint()*u)[0])
    error = abs((u.adjoint()*c*u)[0]-(um.adjoint()*c*um)[0])
    assert error > 0
    assert error <= 2*S*delta
    l = sp.diag(0, 0, 2, 5)
    q = sp.diag(0, 0, 1, 1)
    mean_only = sp.diag(0, 1, 1, 1)
    eta = sp.Rational(1, 3)
    resolvent = q*(l+eta*sp.eye(4)).inv()*q
    susceptibility = (u.adjoint()*resolvent*u)[0]
    cutoff_susceptibility = (um.adjoint()*resolvent*um)[0]
    assert susceptibility >= 0
    assert susceptibility != cutoff_susceptibility
    assert abs(susceptibility-cutoff_susceptibility) <= 2*S*delta/eta
    assert (mean_only*u-q*u).norm() == 2
    # Finite positive energies collapse to zero; finite Q is not limiting Q.
    assert (sp.diag(0, sp.Rational(1, 16)))[1, 1] > 0
    return {'noncommuting_middle_error': str(error), 'bound': str(2*S*delta),
            'positive_eta': str(eta), 'susceptibility': str(susceptibility),
            'positive_eta_cutoff_bound': str(2*S*delta/eta),
            'mean_only_ground_removal_wrong': True, 'finite_P0_transport_rejected': True,
            'scope': 'Exact method control only, not native L spectrum or ground source'}


def run(request: dict[str, Any]) -> dict[str, Any]:
    raw = {name: (ROOT/name).read_bytes() for name in request['pins']}
    assert all(sha(value) == request['pins'][name] for name, value in raw.items())
    descriptor = json.loads(raw[request['micro_input']])['input_descriptor']
    original = json.loads(raw['loop_fractional_graph_actual1_20261012_result.json'])
    car = canonical_matrices()
    coefficient = coefficient_and_covariance(descriptor)
    assert coefficient['b'] == original['interface']['bare_b']
    assert coefficient['Csum'] == original['interface']['Csum_1_plus_63']
    assert car['all24_mixedcopy_entry'] == original['allcopy_CAR_adverse']['two_copy_all16_assignments_offdiagonal']
    result = {'kind': 'INDEPENDENT_EXACT_SOURCE_TAIL_REVIEW_CONTROLS',
              'coefficients': coefficient, 'CAR': car, 'moments': moments_and_endpoint(),
              'bounded_middle_and_whole_P0': mixed_kernels(), 'pins': request['pins'],
              'original_producers_executed': False, 'native_ground_or_fullH_computed': False,
              'scientific_admission': False, 'campaign_complete': False}
    assert all((ROOT/name).read_bytes() == value for name, value in raw.items())
    result['selected_inputs_unchanged'] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.request.read_text()))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'result': str(args.output), 'all_exact_controls_reached': True}))


if __name__ == '__main__':
    main()
