from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = {
    'root_pair_reference_parent_attempt2_20261011_receipt.json': '226e2ec28a156ce3109136c48faefbfec356e86ebe76755b4319490631229feb',
    'local_so5_ground_current_moment_bounds_20261011_receipt.json': 'c6dbb8c839ce2a202be787dcec8f26ca50e4d91132a4cee08a5b8bada708a579',
    'selected_memory_moments_review_proof_20261011.json': '3d70f045dac50284abeaeb57f637e1918441ff3f9e4c6cda39bd40c799bea450',
    'joined_phi_u1_physical_charged_canonical_neutral_scoped_acceptance_20261011.json': '46d482d19a3cd409979b3a74d9237887c5bdf13d7e241ad9ea9fcf30d597841c',
}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def outward(lo: F, hi: F) -> list[str]:
    grid = 1 << 96
    left = F((lo * grid).__floor__(), grid)
    right = F((hi * grid).__ceil__(), grid)
    assert left <= lo <= hi <= right
    return [str(left), str(right)]


def face_signatures(n: int) -> dict[str, object]:
    assert n >= 4 and n % 2 == 0
    sites = list(itertools.product(range(n), repeat=3))
    index = {x: i for i, x in enumerate(sites)}
    internal = {3 * index[x] for x in sites if x[0] % 2 == 0}
    seen: set[tuple[tuple[int, int], ...]] = set()
    counts: dict[str, int] = {}
    for x in sites:
        for a, b in itertools.combinations(range(3), 2):
            xa = tuple((x[j] + (j == a)) % n for j in range(3))
            xb = tuple((x[j] + (j == b)) % n for j in range(3))
            edges = {3 * index[x] + a: 1, 3 * index[xa] + b: 1,
                     3 * index[xb] + a: -1, 3 * index[x] + b: -1}
            assert len(edges) == 4 and sum(k * k for k in edges.values()) == 4
            divergence = [0] * len(sites)
            for edge, value in edges.items():
                tail = sites[edge // 3]
                axis = edge % 3
                head = tuple((tail[j] + (j == axis)) % n for j in range(3))
                divergence[index[tail]] -= value
                divergence[index[head]] += value
            assert not any(divergence)
            signature = tuple(sorted((edge, k) for edge, k in edges.items() if edge not in internal))
            reverse = tuple((edge, -k) for edge, k in signature)
            assert signature and signature not in seen and reverse not in seen
            seen.update((signature, reverse))
            m = sum(edge in internal for edge in edges)
            assert m == (2 if a == 0 and x[0] % 2 == 0 else 0)
            label = f'{a}{b}:anchor_parity{x[0] % 2}:internal{m}'
            counts[label] = counts.get(label, 0) + 1
    assert len(seen) == 6 * n**3
    assert len(counts) == 6 and all(count == n**3 // 2 for count in counts.values())
    return {'N': n, 'positive_faces': 3 * n**3, 'signed_external_flow_sectors': len(seen),
            'face_classes': counts, 'D_inc_face_exact_zero': True,
            'distinct_external_flow_signatures': True,
            'sin_cube_boundary_norm_squared_on_reference': '3',
            'linear_Bianchi_is_not_sine_Bianchi': True}


def finite_ground_control(m: int) -> dict[str, object]:
    # Independent exact m-cell H=-sum sigma_x, rho=sigma_z, ground=all+.
    # Unnormalized all-one coordinates avoid floating square roots.
    size = 1 << m
    omega = [F(1)] * size
    norm = F(size)
    xdiag = [F(sum(1 - 2 * ((mask >> j) & 1) for j in range(m))) for mask in range(size)]

    def shifted(vector: list[F], sign: int) -> list[F]:
        return [(m + 256 + sign * 128 * xdiag[mask]) * vector[mask]
                - sum((vector[mask ^ (1 << j)] for j in range(m)), F(0))
                for mask in range(size)]

    averaged = []
    for degree in range(4):
        values = []
        for sign in (-1, 1):
            result = omega
            for _ in range(degree):
                result = shifted(result, sign)
            values.append(sum(result, F(0)) / norm)
        averaged.append(sum(values, F(0)) / 4)
    expected = [F(1, 2), F(128), F(32768 + 8192 * m),
                F(256**3 + 3 * 256 * 128**2 * m + 128**2 * m * 2, 2)]
    assert averaged == expected
    variance = 2 * averaged[2] - (2 * averaged[1])**2
    assert variance == 16384 * m
    return {'internal_cells': m, 'spectral_moments_exact': list(map(str, averaged)),
            'variance_exact': str(variance),
            'wrong_missing_internal_density_second_moment_rejected': m == 0 or averaged[2] != 32768,
            'scope': 'Exact finite ground/boost moment identity control, not actual canonical256 Hamiltonian.'}


def external_block_boost_control() -> dict[str, object]:
    initial = (1, 0, 0, 0)
    incidence = (1, 1, -1, -1)
    shift = 64 * sum((e + k)**2 - e**2 for e, k in zip(initial, incidence))
    zero_block_shift = 64 * sum(k * k for k in incidence)
    assert shift == 384 and zero_block_shift == 256 and shift != zero_block_shift
    return {'nonzero_external_block_shift_exact': shift,
            'wrong_unrestricted_internal_only_shift_exact': zero_block_shift,
            'missing_external_cross_exact': shift - zero_block_shift,
            'scope': 'Allowed single external E=1 face-edge Gauss block; internal E=0. It is not the reference ground-source block.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = {}
    for name, expected in INPUTS.items():
        assert digest(ROOT / name) == expected, name
        data[name] = json.loads((ROOT / name).read_bytes())
    reference = data['root_pair_reference_parent_attempt2_20261011_receipt.json']['conditional_global_reference']
    review = data['selected_memory_moments_review_proof_20261011.json']
    moments = data['local_so5_ground_current_moment_bounds_20261011_receipt.json']
    assert review['verdict'] == 'VALID' and review['not_absence_of_spectral_atoms'] is True
    local = [[F(value) for value in pair] for pair in moments['moment_intervals_exact']]
    rho2, rho_energy = local[0], local[1]
    assert 0 < rho2[0] <= rho2[1] and 0 < rho_energy[0] <= rho_energy[1]
    classes = []
    for m in (0, 2):
        second = [F(32768) + 8192 * m * value for value in rho2]
        third = [F(256**3, 2) + F(3 * 256 * 128**2 * m, 2) * rho2[j]
                 + F(128**2 * m, 2) * rho_energy[j] for j in (0, 1)]
        variance = [16384 * m * value for value in rho2]
        classes.append({'internal_matching_edges': m, 'mass_exact': '1/2',
                        'first_energy_moment_exact': '128',
                        'second_energy_moment_interval_exact': outward(*second),
                        'third_energy_moment_interval_exact': outward(*third),
                        'normalized_energy_variance_interval_exact': outward(*variance),
                        'exact_single_energy_delta256': m == 0,
                        'not_single_energy_saturation': m != 0,
                        'positive_variance_does_NOT_exclude_atoms': True})
    receipt = {
        'kind': 'CONDITIONAL_MATCHING_REFERENCE_GAUGE_FLUX_BLOCKED_MOMENTUM_SPECTRAL_COMPOSITION',
        'own_source_sha256': digest(Path(__file__)), 'input_sha256': INPUTS,
        'stage': 'NEW_PARENT_DERIVATION_PENDING_INDEPENDENT_OPERATOR_REVIEW_AND_AFFINE_ANTECEDENT',
        'operator': 'sin of elementary native face flux; bounded gauge-complete operator on compact beta0 frozenPhi matching CHARGED H0, not raw A/flux generator.',
        'stationary_ground_antecedent': reference,
        'exact_proof': {
            'flow': 'D inc_face=0, norm squared4. W_face shifts full integer E by inc_face, leaves matter occupations. H0 matter multiplication commutes W_face. The GLOBAL identity is W^* L W=L+256+128 sum_ALL_RAW_e inc_e E_e. First restrict its pullback to the invariant initial k_external=0 block; there the external cross term is zero and X=sum_internal inc_e E_e=-sum_internal signed rho_v. Both L and X preserve that pullback block. Only on this domain use L+256+128X for the product-ground moments; whole rho-flip symmetry makes odd X moments zero.',
            'moments': 'Each external + and - signature is orthogonal: mu mass1/2. m1=128; m2=32768+8192*m*<rho^2>; m3=256^3/2+(3*256*128^2*m/2)*<rho^2>+(128^2*m/2)*<rho Llocal rho>. Product-ground cross terms vanish, not charged-copy independence within a cell.',
            'spectral_domain': 'Finite second moment gives sinflux*g in D(L). L>=0 and nonzero-external-flow excitation support require the actual global ground antecedent, not stationarity alone. No spatial reflection positivity is assumed.',
            'momentum': 'H0 preserves every external raw-edge integer label. Distinct signed elementary faces have distinct external signatures for all even N>=4, hence every spectral cross measure between distinct faces vanishes. Blocked-cell Fourier B_p=cellcount^-1/2 sum exp(-ip.cell) B_face has the SAME diagonal measure at every blocked momentum p, for each of the six orientation/anchor-parity classes. Reference matching breaks one-site x translation; do not call this physical1 translation-invariant momentum dispersion.',
            'gapped_reference': 'With the conditional global H0 gap, every face measure lies above gamma. Four of six face classes have m0 internal matching edges and EXACT atom1/2 at energy256; two have m2 and strictly positive energy variance, not absence of atoms. No gapless/massless claim even as volume tends infinity.',
            'volume': 'Each diagonal face measure reduces to at most two fixed local canonical256 affine cells and integer external cost; untouched product cells cancel Eg_total exactly. This produces an N-independent reference spectral family, not an N-uniform linked perturbation at physical hopping1.',
        },
        'spectral_classes': classes,
        'integer_controls': [face_signatures(n) for n in (4, 6, 8)],
        'ground_boost_controls': [finite_ground_control(m) for m in (0, 2)],
        'external_block_boost_control': external_block_boost_control(),
        'linear_sine_Bianchi_control': 'Six oriented cube-face sinflux source vectors have distinct external signatures and norm squared1/2 each, so their signed sum has norm squared3, despite sum of LINEAR face phases=0. Do not impose a linear sine Bianchi.',
        'tails': 'mu((R,infinity)) <= min(1/2,128/R,m2_upper/R^2). Positive-frequency UPPER only, not stiffness or a pole.',
        'no_admission': {'physical_hopping1_phase': False, 'physicalPhi_lattice_ground': False,
                         'bulk_stiffness': False, 'IR_atom_massless_photon': False,
                         'normalized_allorder_linked_stability': False, 'commoncone': False,
                         'originalSOURCE': False, 'campaign_complete': False},
    }
    assert all(digest(ROOT / name) == expected for name, expected in INPUTS.items())
    (ROOT / args.output).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'receipt': args.output, 'blocked_momentum_measure_flat': True,
                      'reference_only': True, 'physical1_phase': False,
                      'second_moment_intervals': [item['second_energy_moment_interval_exact'] for item in classes]}))


if __name__ == '__main__':
    main()
