from __future__ import annotations

import argparse
import hashlib
import json
import math
import itertools
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = {
    'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json': '102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
    'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json': '2f019dd49873a5e13f130c65ef6739b9af78a16f85138dc6c6ab6e32e65c7a12',
    'charged_lower_review_attempt2_20261011_receipt.json': '7e2bed994a5404794fd888b4812d4bac7873c897f0edb615a3ddadfd7ae961f0',
    'charged_image_proof_control_20261011_receipt.json': 'c45fd28aee5c1cd50c2e7350fc3ba5a276029fd118073da80655d88007975590',
    'joined_phi_u1_physical_charged_canonical_neutral_scoped_acceptance_20261011.json': '46d482d19a3cd409979b3a74d9237887c5bdf13d7e241ad9ea9fcf30d597841c',
    'three_cell_native_proof_20261011.json': 'c67fa1780acd241d0f7869e34d4d3e7979f3199b57763392354817d28c7fb043',
    'three_cell_native_actual3_20261011_receipt.json': '1616f84b0a4c09f7ab1905d0b158601cd2fa7e082a4ee4e1312b35e1dcc7a741',
}
Key = tuple[int, int, int]
State = dict[Key, F]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def root_interval(value: F, bits: int = 192) -> tuple[F, F]:
    denominator = 1 << bits
    lower = math.isqrt(value.numerator * denominator**2 // value.denominator)
    left, right = F(lower, denominator), F(lower + 1, denominator)
    assert left * left <= value <= right * right
    return left, right


def combine(*terms: tuple[F, State]) -> State:
    result: State = {}
    for coefficient, state in terms:
        for key, value in state.items():
            result[key] = result.get(key, F(0)) + coefficient * value
    return {key: value for key, value in result.items() if value}


def inner(left: State, right: State) -> F:
    return sum((value * right.get(key, F(0)) for key, value in left.items()), F(0))


def source() -> State:
    result: State = {(0, 0, 0): F(1)}
    for spin in range(6):
        result = {
            (ke, kf, mask | (1 << (2 * spin + endpoint))): value * amplitude
            for (ke, kf, mask), value in result.items()
            for endpoint, amplitude in ((0, F(3, 5)), (1, F(4, 5)))
        }
    assert inner(result, result) == 1
    return result


def hop(state: State, row_cell: int, row_endpoint: int,
        column_cell: int, column_endpoint: int, flow: int) -> State:
    result: State = {}
    for (ke, kf, mask), value in state.items():
        for spin in range(2):
            row = 4 * row_cell + 2 * spin + row_endpoint
            column = 4 * column_cell + 2 * spin + column_endpoint
            if not (mask & (1 << column)) or mask & (1 << row):
                continue
            annihilated = mask ^ (1 << column)
            parity = ((mask & ((1 << column) - 1)).bit_count()
                      + (annihilated & ((1 << row) - 1)).bit_count()) % 2
            key = (ke + (flow == 0), kf + (flow == 1), annihilated | (1 << row))
            result[key] = result.get(key, F(0)) + (-1 if parity else 1) * value
    return {key: value for key, value in result.items() if value}


def electric(key: Key, central_endpoint: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ke, kf, mask = key
    density = tuple(sum((mask >> (4 * cell + 2 * spin + endpoint)) & 1
                        for spin in range(2)) - 1
                    for cell in range(3) for endpoint in range(2))
    ja = density[0]
    jb = density[2] + ke - (kf if central_endpoint == 0 else 0)
    jc = density[4] + kf
    divergence = (ja, -ja + ke,
                  jb - ke + (kf if central_endpoint == 0 else 0),
                  -jb + (kf if central_endpoint == 1 else 0), jc - kf, -jc)
    assert divergence == density
    return (ja, jb, jc, ke, kf), density


def he(state: State, central_endpoint: int) -> State:
    return {key: value * 64 * sum(j * j for j in electric(key, central_endpoint)[0])
            for key, value in state.items()}


def exact_toy(central_endpoint: int) -> dict[str, object]:
    omega = source()
    a = lambda v: hop(v, 0, 1, 1, 0, 0)
    b = lambda v: hop(v, 1, central_endpoint, 2, 0, 1)
    av, bv = a(omega), b(omega)
    ab, ba, aa = a(bv), b(av), a(av)
    wa, wb = inner(av, av), inner(bv, bv)
    gab, gba, dab = inner(ab, ab), inner(ba, ba), inner(ab, ba)
    for state in (omega, av, bv, ab, ba, aa):
        for key in state:
            electric(key, central_endpoint)
    assert inner(av, bv) == inner(omega, ab) == inner(av, ab) == inner(bv, ab) == 0
    z_ab = inner(combine((F(1), omega), (F(1), av), (F(1), bv), (F(1), ab)),
                 combine((F(1), omega), (F(1), av), (F(1), bv), (F(1), ab)))
    z_sym = 1 + wa + wb + (gab + gba + 2 * dab) / 4
    assert z_ab == 1 + wa + wb + gab
    assert z_ab != (1 + wa) * (1 + wb)
    assert dab > 0
    contact = inner(combine((F(1), ab), (F(-1), ba)), combine((F(1), ab), (F(-1), ba)))
    assert (contact > 0) == (central_endpoint == 0)
    exact_shift = combine((F(1), he(ab, central_endpoint)),
                          (F(-1), a(b(he(omega, central_endpoint)))), (F(-128), ab))
    repeated_shift = combine((F(1), he(aa, central_endpoint)),
                             (F(-1), a(a(he(omega, central_endpoint)))), (F(-256), aa))
    assert not exact_shift and not repeated_shift
    wrong_charge_shift = combine((F(1), he(ab, central_endpoint)),
                                 (F(-1), a(b(he(omega, central_endpoint)))), (F(-512), ab))
    assert inner(wrong_charge_shift, wrong_charge_shift) > 0
    prepared_increment = inner(ab, he(ab, central_endpoint)) / gab - inner(omega, he(omega, central_endpoint))
    assert prepared_increment != 128
    return {
        'scope': 'EXACT_12_MODE_TWO_SPIN_3_CELL_CAR_INTEGER_GAUSS_CONTROL_NOT_CURRENT_9672320_KERNEL',
        'central_endpoint': central_endpoint, 'Wa': str(wa), 'Wb': str(wb),
        'Gab': str(gab), 'Gba': str(gba), 'coherent_history_D': str(dab),
        'contact_norm_squared': str(contact), 'Zordered': str(z_ab), 'Zsymmetric': str(z_sym),
        'wrong_leg_product': str((1 + wa) * (1 + wb)),
        'ordered_log_x2y2': str(gab - wa * wb),
        'symmetric_log_x2y2': str((gab + gba + 2 * dab) / 4 - wa * wb),
        'unit_edge_operator_self_shift': 128, 'same_edge_q2_operator_self_shift': 256,
        'prepared_HE_increment': str(prepared_increment), 'prepared_increment_not_self_coefficient': True,
        'full_H_not_evaluated': True,
    }


def reference_boundary(data: dict[str, object]) -> dict[str, object]:
    profiles = data['profiles']
    assert isinstance(profiles, list)
    intervals = [root_interval(F(p['e_squared_exact'])) for p in profiles]
    a, u = intervals
    c = F(profiles[0]['c0_exact'])
    assert c == F(profiles[1]['c0_exact']) < 0 and a[1] < u[0]
    baseline = (-4 * a[1] - 252 * u[1], -4 * a[0] - 252 * u[0])
    delta = a[0] + c
    budget = F(-5155) - baseline[0]
    assert delta > 0 and budget < 3 * delta and budget < 2 * a[0] + delta
    return {
        'weak_whole_charged_floor': '-5155', 'baseline_interval': list(map(str, baseline)),
        'least_odd_cost_lower': str(delta), 'negative_free_budget_upper': str(budget),
        'three_odd_or_odd_plus_vector_cost_lower': str(3 * delta),
        'scalar_empty_full_plus_odd_cost_lower': str(2 * a[0] + delta),
        'classification': 'At totalnumber255/257 o is odd. Cost>=2eA*t+delta*o+2delta*j; t>=1 or o+2j>=3 exceeds the weak budget. Only ONE odd slot, no excited/scalar spectators, can be free-negative. All four spinor indices and both physical number1/3 embeddings retained.',
        'invariant_cover': 'For EVERY labelled odd slot use the existing entire unprojected Sym spectator x C4 jointD/R containing space, not the harmonic source image. Its -5120 lower applies at u=+/-1/2. Every negative free vector has minimum spectators, hence belongs to this invariant union. The reducing orthogonal complement has D>=-5155 and the electric square is positive. Therefore whole canonical255/257 H0(+/-1/2)>=-5155.',
        'not_new_charged_ground_identification': True,
    }


def integer_flow_control(n: int) -> dict[str, object]:
    assert n >= 4 and n % 2 == 0
    sites = [(p, q, r) for p in range(n) for q in range(n) for r in range(n)]
    index = {x: i for i, x in enumerate(sites)}
    total = [0] * len(sites)
    external = [0] * len(sites)
    internal_energy = 0
    external_energy = 0
    allocated = 0
    full_energy = 0
    internal: dict[tuple[int, int, int], int] = {}
    for x in sites:
        for axis in range(3):
            y = ((x[0] + (axis == 0)) % n,
                 (x[1] + (axis == 1)) % n,
                 (x[2] + (axis == 2)) % n)
            value = (index[x] + 2 * axis) % 5 - 2
            total[index[x]] -= value
            total[index[y]] += value
            full_energy += value * value
            if axis == 0 and x[0] % 2 == 0:
                internal[x] = value
            else:
                external[index[x]] -= value
                external[index[y]] += value
                external_energy += value * value
                allocated += 2 * 32 * value * value
    wrong_internal_sign_rejected = False
    for x, value in internal.items():
        y = (x[0] + 1, x[1], x[2])
        i, j = index[x], index[y]
        q = total[i] + total[j]
        rho = F(total[i] - total[j], 2)
        u = F(q, 2) - external[i]
        assert q == external[i] + external[j]
        assert value == -(rho + u)
        wrong_internal_sign_rejected |= value != rho + u
        internal_energy += value * value
    assert wrong_internal_sign_rejected
    assert internal_energy + external_energy == full_energy
    assert allocated == 64 * external_energy
    # Ten external raw incidences at a blocked vertex; count raw edges,
    # not paths, and do not collapse parallel edges between the same dimers.
    endpoint = [0, 1] + [s for s in (0, 1) for _ in range(4)]
    classes: dict[int, set[tuple[int, F]]] = {0: {(0, F(0))}, 1: set(), 2: set()}
    for edge, sign in itertools.product(range(10), (-1, 1)):
        d = [0, 0]
        d[endpoint[edge]] = sign
        classes[1].add((sum(d), F(d[1] - d[0], 2)))
    for a, b in itertools.combinations(range(10), 2):
        for sa, sb in itertools.product((-1, 1), repeat=2):
            d = [0, 0]
            d[endpoint[a]] += sa
            d[endpoint[b]] += sb
            classes[2].add((sum(d), F(d[1] - d[0], 2)))
    assert classes[1] == {(q, u) for q in (-1, 1) for u in (F(-1, 2), F(1, 2))}
    assert classes[2] == {(q, F(u)) for q in (-2, 0, 2) for u in (-1, 0, 1)}
    return {
        'N': n, 'raw_edges': 3 * n**3, 'matching_edges': len(internal),
        'HE_exact': 64 * full_energy,
        'HE_decomposed_exact': 64 * (internal_energy + external_energy),
        'external_endpoint_allocation_exact': allocated,
        'weighted_incidence_classes': {
            str(w): [[q, str(u)] for q, u in sorted(values)]
            for w, values in classes.items()
        },
        'wrong_internal_sign_rejected': wrong_internal_sign_rejected,
        'single_raw_absflow2_weighteddegree': 4,
        'wrong_single_raw_absflow2_degree1_rejected': 2**2 != 1,
        'scope': 'Independent exact integer Gauss/allocation control; not an H0 spectral calculation.',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = {}
    for name, expected in INPUTS.items():
        assert digest(ROOT / name) == expected, name
        data[name] = json.loads((ROOT / name).read_bytes())
    reviewed = data['charged_lower_review_attempt2_20261011_receipt.json']
    assert reviewed['status'] == 'VALID_SCOPED_LOWER_AND_IMAGE_CONDITIONAL_APPLICATION'
    assert all(b['status'] == 'VALID' and b['lambda1_floor_exact'] == '-5120' for b in reviewed['branches'])
    boundary = reference_boundary(data['local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'])
    upper = data['local_so5_ground_upper_complete_state_consumer_20261011_receipt.json']
    e_upper = F(upper['ground_energy_interval_exact'][1])
    gap = -5123 - e_upper
    assert gap > 0
    baseline_values = boundary['baseline_interval']
    delta_value = boundary['least_odd_cost_lower']
    assert isinstance(baseline_values, list) and isinstance(baseline_values[0], str)
    assert isinstance(delta_value, str)
    baseline = F(baseline_values[0])
    delta = F(delta_value)
    margins = {'w1_charged': F(-5155) + 32 - e_upper,
               'w2_abscharge2': baseline + 2 * delta + 64 - e_upper,
               'w_ge3': baseline + 96 - e_upper,
               'w2_neutral_affine_IF_floor_minus5187': F(-5187) + 64 - e_upper}
    assert all(value >= gap for value in margins.values())
    receipt = {
        'kind': 'NEW_EXACT_NORMALIZED_ROOT_CONTROL_AND_UNPERTURBED_GROUND_PREREQUISITE_COMPOSITION',
        'own_source_sha256': digest(Path(__file__)), 'input_sha256': INPUTS,
        'charged_weak_cover_parent_proof_pending_independent_review': boundary,
        'toy_controls': [exact_toy(0), exact_toy(1)],
        'integer_flow_controls': [integer_flow_control(n) for n in (4, 6, 8)],
        'primary_native_geometry': data['three_cell_native_proof_20261011.json']['exact_law_and_geometry']['global_candidate_geometry_only'],
        'conditional_global_reference': {
            'operator': 'FROZEN_PHI_FIXED_REF4_COMPACT_BETA0 CHARGED unperturbed H0 keeps onsite+inside-pair hopping+ALL native E64; V retains every other physical one/two-link path. No physical hopping/coupling adoption.',
            'pointwise_Gauss': 'q_v=external divergence summed over both sites; E_internal=-(rho_v+u_v), u_v=q_v/2-external_divergence_at_site0. Integer k yields weighted incidence w=sum k_e^2. Each external64 k^2 allocates32 k^2 to each endpoint.',
            'classes': 'w0:q0/u0 exact canonical ground. w1:q+/-1,u+/-1/2 weak charged bound. w2:q0,u0 or+/-1; |q|2,u0 or+/-1 with D>=B+2delta. w>=3: arbitrary charge/u, D>=B and allocation>=96. Incoming1 bound still REQUIRED.',
            'gap_lower_IF_affine_and_geometry_proved': str(gap),
            'pointwise_margins_exact': {name: str(value) for name, value in margins.items()},
            'affine_required_floor': '-5187', 'affine_certificate_present': False,
            'all_native_graph_operator_proof_reviewed': False,
            'global_H0_ground_admitted': False,
            'all_even_N_proof': 'For every even N>=4 choose disjoint axis0 matching dimers. In the native integer electric representation H0 preserves all external raw-edge k and cell particle numbers; Gauss fixes Q=d0+d1 and internal E=-(rho+u). Its direct-sum block is sum_v[D_v+64(rho_v+u_v)^2+32w_v]. Raw-edge degree is exactly ten, including parallel edges. The exhaustive w<=2 cases and free lower D>=B imply each nonzero-incidence cell lies at least gamma above E_g, IF the whole neutral affine +/-1 floor holds. With all k=0 each Q/u=0 and the accepted entire local canonical256 ground/gap applies, yielding unique product g and gap>=gamma. Any nonzero external flow has nonzero incidence and positive local margins. This is an unperturbed compact beta0 reference, NOT the physical hopping1 Hamiltonian.',
            'weak_charged_coverage_domain': 'Entire local canonical255/257 under positive electric square; maximal symmetric spectator containing spaces and every labelled number1/3 branch, not just a source image.',
            'physical_hopping1_ground_or_phase': False,
            'fullPhi_lattice_ground': False, 'all_Bloch_beta_ground': False,
            'allorder_normalized_linked_stability': False, 'sharp_photon': False,
        },
        'three_cell_transfer': {
            'squared_infidelity': upper['selected_ground_infidelity_upper_exact'],
            'formula': 'For aligned normalized source/g, product fidelity >=(1-I)^3; product projector distance squared<=3I and vector distance squared<=3I. Product residual<=3*(eta+Rayleigh-ground shift). This is a chosen neutral product-state transfer, NOT a global charged/external-flow ground assertion.',
            'eta': upper['normalized_exact_source_residual_upper_exact'],
        },
        'limitations': ['The 12-mode control is not the actual all1+63 geometric kernel.', 'Canonical weak charged cover is a NEW parent proof pending review.', 'Conditional global-reference theorem awaits actual affine lower and native graph operator proof.', 'Even an H0 gap does not prove physical hopping1 or physicalPhi ground, phase, stiffness, IR atom or commoncone.'],
    }
    assert all(digest(ROOT / name) == expected for name, expected in INPUTS.items())
    (ROOT / args.output).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'receipt': args.output, 'charged_weak_floor': '-5155',
                      'conditional_gap': str(gap), 'global_admitted': False, 'toy_cases': len(receipt['toy_controls'])}))


if __name__ == '__main__':
    main()
