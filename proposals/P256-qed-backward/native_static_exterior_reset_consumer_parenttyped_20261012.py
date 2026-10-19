"""Native centre/CG/annihilation-Gauss and exact-ground reset-tail gates.
No H eigensolver, numerical minimizing ground or macro evolution backend.
Representation matrices are native SU4 representation controls, NOT PHP H.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, cast
import zipfile

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def su4_generators() -> list[sp.Matrix]:
    out: list[sp.Matrix] = []
    for i in range(4):
        for j in range(i+1, 4):
            a, b = sp.zeros(4), sp.zeros(4)
            a[i, j] = a[j, i] = sp.Rational(1, 2)
            b[i, j], b[j, i] = -sp.I/2, sp.I/2
            out.extend([a, b])
    for k in range(1, 4):
        entries = [1]*k+[-k]+[0]*(3-k)
        out.append(sp.diag(*entries)/sp.sqrt(2*k*(k+1)))
    assert len(out) == 15
    for i, a in enumerate(out):
        assert a.trace() == 0 and a.H == a
        for j, b in enumerate(out):
            assert sp.simplify((a*b).trace()) == (sp.Rational(1, 2) if i == j else 0)
    return out


def clifford_tube() -> dict[str, Any]:
    s1 = sp.Matrix([[0, 1], [1, 0]])
    s2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    s3 = sp.diag(1, -1)
    eye = sp.eye(2)
    gammas: list[Any] = [sp.kronecker_product(s1, eye), sp.kronecker_product(s2, eye),
                        sp.kronecker_product(s3, s1), sp.kronecker_product(s3, s2)]
    for i, a in enumerate(gammas):
        for j, b in enumerate(gammas):
            assert a*b+b*a == (2*sp.eye(4) if i == j else sp.zeros(4))
    # The generated complex Clifford monomials span ALL M4, proving that a
    # simultaneous unitary centralizer is scalar, then determinant1 gives Z4.
    monomials = [sp.eye(4)]
    for mask in range(1, 16):
        value = sp.eye(4)
        for i in range(4):
            if mask & (1 << i):
                value = value*gammas[i]
        monomials.append(value)
    rank = sp.Matrix.hstack(*[sp.Matrix(list(m)) for m in monomials]).rank()
    assert rank == 16
    generators = su4_generators()
    inertia: Any = sp.zeros(15)
    for i, a in enumerate(generators):
        for j, b in enumerate(generators):
            inertia[i, j] = sp.simplify(cast(Any, sum(
                ((a*g-g*a).H*(b*g-g*b)).trace() for g in gammas)))
    spectrum = inertia.eigenvals()  # Native 15x15 orbit metric, NOT H spectrum.
    assert spectrum == {sp.Integer(2): 4, sp.Integer(4): 6,
                        sp.Integer(6): 4, sp.Integer(8): 1}
    jordan = sp.simplify(cast(Any, 4*sum(((a*b+b*a)**2).trace()
                                       for a in gammas for b in gammas)))
    assert jordan == 256
    delta = Q(1, 2**16)
    normal_profile_gradient = Q(45**2*16, 4)*delta**-2  # pi^2<16.
    metric_lower = Q(1)-2*delta  # sqrt2<2 in perturbation lower bound.
    assert metric_lower > Q(99, 100)
    assert normal_profile_gradient < 2**46
    return {'scope': 'exact native Cl4 orbit/normal-tube algebra and metric',
            'Clifford_generated_M4_rank': rank, 'centralizer_SU4': 'Z4 exactly',
            'Phi_fields_centre_class': 0, 'HS_orbit_inertia_eigenvalues': {str(k): v for k, v in spectrum.items()},
            'q_metric_min_at_orbit': 4, 'normal_dimension': 45,
            'declared_tube_relative_radius': str(delta),
            'product_cos_normal_profile_H1_gradient_upper': str(normal_profile_gradient),
            'scaled_orbit_Jordan_over_lambda_R4': int(jordan),
            'tube_Jordan_bound_two_sites': '2^16 lambda R^4',
            'tube_alignment_plus_external_diagonal_bound': '512 alpha R^2',
            'no_Gauss_to_SO60_angular_bound': True,
            'tube_global_injectivity_and_normalization_proof': 'analytic in reset proof, not inferred from metric samples'}


def casimir(dynkin: tuple[int, int, int]) -> Q:
    a, b, c = dynkin
    inverse = ((Q(3, 4), Q(1, 2), Q(1, 4)),
               (Q(1, 2), Q(1), Q(1, 2)),
               (Q(1, 4), Q(1, 2), Q(3, 4)))
    v = (a, b, c)
    return cast(Q, sum(v[i]*inverse[i][j]*(v[j]+2) for i in range(3) for j in range(3))/2)


def centre(dynkin: tuple[int, int, int]) -> int:
    return (dynkin[0]+2*dynkin[1]+3*dynkin[2]) % 4


def representations(max_dynkin: int) -> dict[str, Any]:
    classes: list[tuple[int, int, int]] = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert [centre(x) for x in classes] == list(range(4))
    assert [casimir(x) for x in classes] == [Q(0), Q(15, 8), Q(5, 2), Q(15, 8)]
    checked = 0
    for a in range(max_dynkin+1):
        for b in range(max_dynkin+1):
            for c in range(max_dynkin+1):
                mu = (a, b, c)
                link = classes[centre(mu)]
                dual = tuple(reversed(link))
                sigma = cast(tuple[int, int, int], tuple(mu[i]+dual[i] for i in range(3)))
                assert centre(sigma) == 0
                assert casimir(sigma) <= 2*casimir(mu)+2*casimir(link)
                checked += 1
    return {'all_boundary_admissible_centre_pairs': '(q,-q); no others occur in Phi_x,Phi_y,internal-link Hilbert',
            'minimal_link_Dynkin_by_q': classes,
            'minimal_link_C2': [str(casimir(x)) for x in classes],
            'carrier_Phi_choice': 'sigma_x=mu_x+lambda*, sigma_y=mu_y+lambda (Cartan highest components)',
            'CG_existence': 'Cartan highest-vector component + compact-representation Hom reciprocity; explicit fundamental/adjoint CG below',
            'positive_exact_highest_weight_controls': checked,
            'norm_independent_of_magnetic_dimension': 'normalized Peter-Weyl isometries and normalized CG; I_Dmu retained'}


def cg_annihilation() -> dict[str, Any]:
    generators = su4_generators()
    fund = [np.array(a, dtype=complex) for a in generators]
    adj = []
    for a in generators:
        matrix = sp.Matrix(15, 15, lambda c, b: sp.simplify(
            2*(generators[c]*(a*generators[b]-generators[b]*a)).trace()))
        adj.append(np.array(matrix, dtype=complex))
    embeddings = []
    defects = []
    singlet_defects = []
    for rep in (fund, [-a.conjugate() for a in fund]):
        embedding = np.vstack(rep)/math.sqrt(15/8)
        assert np.max(abs(embedding.conj().T@embedding-np.eye(4))) < 1e-12
        full = [np.kron(adj[i], np.eye(4))+np.kron(np.eye(15), rep[i]) for i in range(15)]
        defect = max(float(np.max(abs(full[i]@embedding-embedding@rep[i]))) for i in range(15))
        assert defect < 1e-12
        # Actual annihilation Gauss with external dual, not mere covariance.
        singlet = sum((np.kron(embedding[:, j], np.eye(4)[:, j]) for j in range(4)), np.zeros(240, complex))/2
        gd = max(float(np.linalg.norm((np.kron(full[i], np.eye(4))
                 +np.kron(np.eye(60), -rep[i].conjugate()))@singlet)) for i in range(15))
        assert gd < 1e-12
        embeddings.append(embedding)
        defects.append(defect)
        singlet_defects.append(gd)
    # Multiplicity reset on native fundamental D, tested WITH its external
    # conjugate annihilation constraint. No Hamiltonian matrix is introduced.
    change = np.array([[0, 1], [0, 0]], complex)
    kraus = np.kron(np.eye(4), change)
    original = sum((np.kron(np.kron(np.eye(4)[:, j], [0, 1]), np.eye(4)[:, j])
                    for j in range(4)), np.zeros(32, complex))/2
    reset = np.kron(kraus, np.eye(4))@original
    assert abs(np.vdot(reset, reset)-1) < 1e-12
    reset_defect = 0.0
    for a in fund:
        gauss = np.kron(np.kron(a, np.eye(2)), np.eye(4))+np.kron(np.eye(8), -a.conjugate())
        reset_defect = max(reset_defect, float(np.linalg.norm(gauss@reset)))
        assert np.max(abs(np.kron(a, np.eye(2))@kraus-kraus@np.kron(a, np.eye(2)))) == 0
    assert reset_defect < 1e-12
    # Wrong magnetic trace/reset loses the correlation with exterior charge.
    wrong = np.zeros((4, 4), complex)
    wrong[0, 0] = 1
    wrong_vector = np.kron(np.kron(wrong, change), np.eye(4))@original
    wrong_defect = sum(float(np.linalg.norm(
        (np.kron(np.kron(a, np.eye(2)), np.eye(4))
         +np.kron(np.eye(8), -a.conjugate()))@wrong_vector))**2 for a in fund)
    assert wrong_defect > 0
    return {'scope': 'actual native SU4 representation/CG/Gauss controls; NOT H truncation',
            'mu_x_mu_y': ['4', '4bar'], 'link': '4 x 4bar', 'Phi_irreps': ['15', '15'],
            'CG_isometry_and_intertwiner_max_defect': defects,
            'external_dual_Gauss_annihilation_norms': singlet_defects,
            'multiplicity_Kraus_Gauss_annihilation_norm': reset_defect,
            'source_and_reset_norm': 1,
            'wrong_magnetic_reset_Gauss_defect_squared': wrong_defect,
            'CAR_U1_Ref_unchanged': 'Kraus acts Phi/link multiplicity ONLY; identities on these sectors'}

def native_threshold_jets(max_dynkin: int) -> dict[str, Any]:
    generators = su4_generators()
    fundamental_casimir = sum((a*a for a in generators), sp.zeros(4))
    assert fundamental_casimir == sp.Rational(15, 8)*sp.eye(4)
    phi = sp.diag(1, 1, -1, -1)
    mixed_jet_squared = sp.simplify(cast(Any, sum((phi*a).H.multiply(phi*a).trace()
                                                 for a in generators)))
    assert mixed_jet_squared == sp.Rational(15, 2)
    # Phi*d_Haar principal coefficient is NONZERO in the actual generators.
    # No kinetic projection or frozen link is used to relabel it as sqrt(h).
    def fundamental_products(mu: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        a, b, c = mu
        choices = [(a+1, b, c)]
        if a:
            choices.append((a-1, b+1, c))
        if b:
            choices.append((a, b-1, c+1))
        if c:
            choices.append((a, b, c-1))
        return choices
    count = 0
    kappa = Q(15, 4)
    for a in range(max_dynkin+1):
        for b in range(max_dynkin+1):
            for c in range(max_dynkin+1):
                mux = (a, b, c)
                muy = (c, b, a)
                cin = casimir(mux)+casimir(muy)
                for nx in fundamental_products(mux):
                    for ny_dual in fundamental_products(cast(tuple[int, int, int], tuple(reversed(muy)))):
                        ny = cast(tuple[int, int, int], tuple(reversed(ny_dual)))
                        cout = casimir(nx)+casimir(ny)
                        delta = cout-cin
                        if delta > kappa:
                            assert (delta-kappa)**2 <= 4*kappa*cin
                        if -delta > kappa:
                            assert (-delta-kappa)**2 <= 4*kappa*cout
                        assert (centre(nx)+centre(ny)) % 4 == 0
                        count += 1
    for threshold, charge in ((Q(64), kappa), (Q(64), Q(4)), (Q(256), Q(32))):
        assert threshold > charge and (threshold-charge)**2 > 4*charge*threshold
    return {'Cfund_exact': '15/8',
            'Phi_times_full15_Haar_first_jet_HS_squared': str(mixed_jet_squared),
            'Phi_U_commutator_retains_nonzero_mixed_derivative': True,
            'exact_fundamental_dual_Pieri_threshold_controls': count,
            'centre_pairs_preserved': True,
            'casimir_two_threshold_native_charge_checks': [64, 64, 256],
            'principal_current_order': 'r~h^1/4, Haar derivative~h^1/2, product~h^3/4; energy gap gives h^-1/4',
            'scope': 'actual native coefficient/tensor controls plus analytic block spectral gate, NOT numerical H spectrum'}


def native_boundary_budget(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lam, chi, fast = Q(2**8192), Q(1, 2**4096), Q(2**10240)
    assert [int(p['load']) for p in profiles] == [1, 1, 63]
    weights = [abs(Q(p['onsite']))+2*sum(abs(Q(t['c0']))+abs(Q(t['ci']))
               for t in p['paths']) for p in profiles]
    b = 4*(lam*weights[0]+weights[1]+63*weights[2])
    assert b < 512*lam
    r0 = 60*chi+11192*lam+Q(129, 10)*fast+262144
    c4 = 2*r0/lam
    kh = Q(3, 20)*chi
    cq = 1+100*r0/kh+2*15*264**2
    return {'all65_loads': [1, 1, 63], 'Ref4_existing_charge_conservatively_included': True,
            'B_sha256': digest(str(b).encode()), 'B_lt_512lambda': True,
            'R0_sha256': digest(str(r0).encode()), 'C4_sha256': digest(str(c4).encode()),
            'Ch': '1+67 R0+14 alpha sqrt(C4)',
            'Cq_exact_sha256': digest(str(cq).encode()),
            'Cq': '1+100 R0/kHaar+2090880',
            'boundary_alignment_edges': 10, 'external_link_plaquette_faces': 4,
            'D_alignment': '20 alpha (Ch*C4/lambda)^(1/4)',
            'D_inside_matter': '2B (Ch/lambda)^(1/4)',
            'D_outside_matter': '4B C4^(1/4)',
            'D_plaquette': '32 beta',
            'complete_boundary_variation': '3D sqrt(p)+D_outside p^(3/4)+D_plaquette p; outside-Phi internal-U paths NOT parity-killed',
            'fullE_U1_CAR_kinetics_outside_block_unchanged': True}


def scaling() -> dict[str, Any]:
    # All constants remain exact native dyadic/rational, no enormous floats.
    chi, lam, fast = Q(1, 2**4096), Q(2**8192), Q(2**10240)
    r0sq = Q(1, 2**4096)
    ckin = Q(2**96)
    a = ckin*(4*chi)/r0sq + 2**16*lam*r0sq*r0sq
    b = 512*(fast/40)*r0sq
    c = 1+Q(5, 2)*(Q(3, 20)*chi)
    assert a == 4*2**96+2**16
    assert b == Q(64, 5)*2**6144
    stages = []
    for p in (Q(1), Q(5, 4), Q(3, 2), Q(7, 4), Q(2)):
        gamma = min(p-Q(2, 3), p/2+Q(1, 4))
        supremum = 1+gamma
        stages.append({'input_h_and_external_Casimir_moment_p': str(p),
                       'carrier_tail_exponent': str(p-Q(2, 3)),
                       'Phi_U_low_high_current_exponent': str(p/2+Q(1, 4)),
                       'two_threshold_result_h_moment_supremum_not_endpoint': str(supremum),
                       'lift_from_local_graph_proved_in_packet': p <= 2})
        # Every selected next p is STRICTLY below the preceding supremum.
    cutoff = []
    for phi_moment in (Q(4), Q(7), Q(8), Q(10), Q(16), Q(17)):
        cutoff.append({'radial_moment_order': str(phi_moment),
                       'cone_R_M2_union_norm_exponent': str(3-phi_moment/2),
                       'complete_triangle_action_M_exponent': str(8-phi_moment/2),
                       'both_vanish_in_this_route': phi_moment > 16,
                       'proved_by_reset_bootstrap_and_graph_lift': phi_moment <= 8})
    return {'carrier_energy': 'A q^(2/3)+B q^(1/3)+C, q=1+C2(mu_x)+C2(mu_y)',
            'R': '2^-2048 q^(1/6)',
            'A_exact': str(a), 'B_exact_sha256': digest(str(b).encode()), 'B_bits': b.numerator.bit_length(),
            'C_exact_sha256': digest(str(c).encode()),
            'first_ground_h_tail': 'E[h 1_{h> M}] <= A Cq^(2/3) (Ch/M)^(1/3)+B Cq^(1/3)(Ch/M)^(2/3)+C Ch/M + complete boundary costs',
            'tail_is_not_arbitrary_high_moment': True,
            'two_threshold_bootstrap_strict_stages': stages,
            'supported_first_graph_lift': '1<=p<=2 for local h^p to internal Haar Casimir^p and radial r^(4p); higher lift not silently assumed',
            'current_gate': {'Phi': '||P>2M Phi P<=M|| <= c M^-1/2',
                             'U': '||P>2M U P<=M|| <= c M^-1/2+c/M',
                             'Phi_U': '||P>2M Phi U P<=M|| <= c M^-1/4; full Phi*Haar derivative retained',
                             'Casimir_U': 'P_{q>2J} U P_{q<=J}=0 for J>=64 via native tensor-Casimir support'},
            'actual_cutoff_exterior_retry': cutoff,
            'precise_remaining_operator_gate': 'R^3 quadratic-action triangle at R~M^2 still grows; Phi U current contains Phi*d_Haar, and first local graph lift does not supply r^p>16 or excitation-local summation/static density',
            'same_static_exterior_or_range_closed': False}


def radial_IMS_gate(cutoffs: list[int]) -> dict[str, Any]:
    rows = []
    for m in cutoffs:
        radius = m*m
        volume = (2*radius+1)**3
        norm2 = Q(volume, m**8)
        excitation = Q(8*volume, m**10)
        rows.append({'field_r_cutoff': m, 'patch_radius_div_fixed_cone_constant': radius,
                     'patch_sites': volume, 'source_norm_tail_squared_div_C8': str(norm2),
                     'exact_ground_excitation_form_tail_div_kphi_C8_upper': str(excitation),
                     'core_inserted_tail_form_first_term_div_a_gA_sqrt_C8': math.sqrt(float(norm2)),
                     'full_action_first_derivative_triangle_exponent': 3})
    assert rows[-1]['source_norm_tail_squared_div_C8'] != rows[0]['source_norm_tail_squared_div_C8']
    assert float(Q(rows[-1]['source_norm_tail_squared_div_C8'])) < float(Q(rows[0]['source_norm_tail_squared_div_C8']))
    return {'native_identity': 'q_L(f Omega)=kphi <sum |grad_q f|^2> for scalar radial f; all other full-H terms commute f',
            'cutoff': 'eta(r/M)=1 for r<=M,0 for r>=2M; r=|q|/sqrt2; grad_q r squared=1/2; eta phase derivative<=pi',
            'rows': rows,
            'crop_excitation_form_scaling_at_radial8': 'ground O(M^-4); A Omega graph source tail O(M^-1)+O(M^-4)',
            'actual_original_U_crop_comparison': 'U A Omega and U chi A Omega are uniformly close in norm AND excitation form because U preserves L form; the cropped vector is STILL evolved by ORIGINAL fullH',
            'missing_unitary_static_boundary': 'No uniformly compared Dirichlet/compressed finite-patch evolution or THIS static-core B_R(t). IMS small form is NOT small graph/action; (U-1)chi A Omega need not vanish with R.',
            'same_static_exterior_or_range_closed': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    if before != request['pins']:
        raise ValueError('reset input/source pin mismatch')
    with zipfile.ZipFile(ROOT/request['epoch_archive']) as bundle:
        profile_raw = bundle.read(request['profile'])
        profiles = json.loads(profile_raw)['primitive_profiles']
    result = {'schema': 1, 'kind': 'NATIVE_GROUND_SENSITIVE_ISOTYPIC_RESET_TAIL_AND_TWO_THRESHOLD_CONSUMER',
              'clifford_carrier': clifford_tube(),
              'SU4_centre_and_highest_weight': representations(request['max_dynkin']),
              'CG_and_annihilation_Gauss': cg_annihilation(),
              'ground_variational_and_cutoff_gates': scaling(),
              'native_boundary_and_Casimir_budgets': native_boundary_budget(profiles),
              'native_Phi_U_and_Casimir_threshold_jets': native_threshold_jets(request['max_dynkin']),
              'native_radial_IMS_assessment': radial_IMS_gate(request['field_cutoffs']),
              'profile_member_sha256': digest(profile_raw),
              'proof_sha256': before[request['proof']],
              'captured_revision': request['captured_revision'], 'assignment_layer': request['assignment_layer'],
              'source_input_sha256_before': before,
              'source_input_sha256_after': {name: digest((ROOT/name).read_bytes()) for name in request['pins']},
              'runtime': {'python': sys.executable, 'numpy': np.__version__, 'sympy': sp.__version__, 'elapsed_seconds': time.monotonic()-start},
              'native_H_or_ground_U_numerics': False,
              'finite_representation_controls_are_H_truncation': False,
              'source_coupling_boundary_adoption': False}
    if result['source_input_sha256_after'] != before:
        raise RuntimeError('reset input mutation')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'native_SU4_CG_and_annihilation_Gauss': True,
                      'ground_sensitive_reset_tail_and_two_threshold_attempted': True,
                      'same_static_exterior_closed': False,
                      'elapsed_seconds': result['runtime']['elapsed_seconds']}))


if __name__ == '__main__':
    main()
