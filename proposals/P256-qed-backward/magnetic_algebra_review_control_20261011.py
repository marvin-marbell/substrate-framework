"""Independent native-law algebra controls; no target entrypoint or ground solver."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import sparse
import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def incidence(n: int) -> tuple[Any, Any, Any]:
    """Direction-major independent tensor-difference construction."""
    shift = sparse.csr_matrix((np.ones(n, dtype=np.int64), (np.arange(n), (np.arange(n) + 1) % n)), shape=(n, n))
    delta = shift - sparse.eye(n, dtype=cast(Any, np.int64), format='csr')
    ident = sparse.eye(n, dtype=cast(Any, np.int64), format='csr')
    dx = sparse.kron(sparse.kron(delta, ident), ident, format='csr')
    dy = sparse.kron(sparse.kron(ident, delta), ident, format='csr')
    dz = sparse.kron(sparse.kron(ident, ident), delta, format='csr')
    zero = sparse.csr_matrix(dx.shape, dtype=np.int64)
    D = sparse.hstack((-dx.T, -dy.T, -dz.T), format='csr')
    C = sparse.bmat(((-dy, dx, zero), (-dz, zero, dx), (zero, -dz, dy)), format='csr')
    S = sparse.hstack((dz, -dy, dx), format='csr')
    return D, C, S


def lattice_control(n: int, seed: int) -> dict[str, Any]:
    D, C, S = incidence(n)
    v = n ** 3
    theta = np.random.default_rng(seed).integers(-13, 14, size=3 * v, dtype=np.int64)
    raw = C @ theta
    grid = theta.reshape(3, n, n, n)
    native = []
    for j, k in ((0, 1), (0, 2), (1, 2)):
        native.append(grid[j] + np.roll(grid[k], -1, axis=j) - np.roll(grid[j], -1, axis=k) - grid[k])
    assert np.array_equal(raw, np.asarray(native).reshape(-1))
    assert (S @ C).nnz == 0 and (C @ D.T).nnz == 0
    assert not np.any(S @ raw)
    surfaces = []
    for face, normal in ((0, 2), (1, 1), (2, 0)):
        for layer in range(n):
            sigma = np.zeros((3, n, n, n), dtype=np.int64)
            selection: list[Any] = [face, slice(None), slice(None), slice(None)]
            selection[normal + 1] = layer
            sigma[tuple(selection)] = 1
            row = sigma.reshape(-1)
            assert not np.any(C.T @ row) and int(row @ raw) == 0
            surfaces.append(dict(face=('xy', 'xz', 'yz')[face], layer=layer, flux_units=0))
    principal = (raw + 13) % 26 - 13
    defects = S @ principal
    assert np.all(defects % 26 == 0) and np.count_nonzero(defects) > 0
    assert int(defects.sum()) == 0
    sin_div = S @ np.sin(np.pi * raw / 13)
    assert np.max(np.abs(sin_div)) > .1
    deck = np.zeros(3 * v, dtype=np.int64)
    deck[0] = 26
    changed = C @ (theta + deck)
    assert np.count_nonzero(changed - raw) == 4
    assert np.array_equal((changed + 13) % 26 - 13, principal)
    # Positive controls reach the same assertions: corrupt one face orientation,
    # and use an open surface (one face), not a closed torus plane.
    corrupt = C.copy().tolil()
    first_column = int(C[0].indices[0])
    corrupt[0, first_column] = -corrupt[0, first_column]
    bad_bianchi = S @ corrupt.tocsr()
    bad_bianchi.eliminate_zeros()
    assert bad_bianchi.nnz > 0
    open_face = np.zeros(3 * v, dtype=np.int64)
    open_face[0] = 1
    assert np.count_nonzero(C.T @ open_face) == 4
    return dict(N=n, seed=seed, theta_pi_over_13_units=theta.tolist(), SC_exact_nnz=0, CDtranspose_exact_nnz=0,
                all_planes=surfaces, cube_charge_values=sorted(set((defects // 26).tolist())),
                cube_charge_nonzero_count=int(np.count_nonzero(defects)), net_charge=0,
                sin_bianchi_divergence_max=float(np.max(np.abs(sin_div))),
                deck_changed_raw_faces=4, deck_preserves_principal_and_Wilson=True,
                wrong_orientation_positive_control_nnz=int(bad_bianchi.nnz), open_surface_positive_control_boundary_edges=4,
                legal_packet='Project this finite real cochain to kerD without changing CTheta; thicken to a normalized transverse Gaussian, all-Phi invariant Gaussian/Haar constant, checkerboard wedge4/0 all64+neutral, Ref4, longitudinal equivariance. Finite-form legal configuration, not ground data.')


def rational_gauss_control() -> dict[str, Any]:
    D, C, _ = incidence(4)
    v = 64
    Ds = sp.Matrix(D.toarray())
    L = Ds * Ds.T
    J = sp.ones(v) / v
    Lplus = (L + J).inv(method='DM') - J
    k = sp.zeros(3 * v, 1)
    k[0] = 1
    drho = Ds * k
    longitudinal = Ds.T * Lplus * drho
    transverse = k - longitudinal
    assert Ds * longitudinal == drho and Ds * transverse == sp.zeros(v, 1)
    assert transverse.dot(longitudinal) == 0
    assert transverse[0] == sp.Rational(43, 64)
    # Source even cell is empty, target odd cell is filled: c_source^dag
    # c_target is a nonzero legal native +k hop, not a Pauli-blocked update.
    occupation = [0 if sum(x) % 2 == 0 else 4 for x in np.ndindex(4, 4, 4)]
    assert occupation[0] == 0 and occupation[16] == 4
    rho = sp.Matrix([64 * (number - 2) for number in occupation])
    EL = Ds.T * Lplus * rho
    assert Ds * EL == rho
    assert sp.Matrix(C.toarray()) * EL == sp.zeros(3 * v, 1)
    incoming = EL + transverse * sp.Rational(3, 8)
    after = incoming + k
    assert Ds * after == rho + drho
    assert Ds * (incoming + transverse) - (rho + drho) == -drho
    electric_increment = sp.expand(64 * (after.dot(after) - incoming.dot(incoming)))
    transverse_before = incoming - EL
    transverse_increment = 64 * ((transverse_before + transverse).dot(transverse_before + transverse) - transverse_before.dot(transverse_before))
    coulomb_increment = 64 * ((rho + drho).dot(Lplus * (rho + drho)) - rho.dot(Lplus * rho))
    assert electric_increment == transverse_increment + coulomb_increment
    # A pure-gradient fullE translation is not the Gauss transformation.
    lam = sp.zeros(v, 1)
    lam[0] = sp.Rational(1, 3)
    alpha = Ds.T * lam
    path_angle = alpha.dot(k)
    assert path_angle == sp.Rational(1, 3)
    assert (sp.Matrix(C.toarray()) * alpha) == sp.zeros(3 * v, 1)
    return dict(native_path_increment='k', source_minus_target_drho=list(map(str, drho)),
                fullE_identity_exact=True, fullE_deck_path_phase='exp(2pi i)=1',
                wrong_Pi_only_deck_path_phase='exp(2pi i * 43/64) != 1',
                P_L_k_edge0=str(longitudinal[0]), P_T_k_edge0=str(transverse[0]),
                legal_checkerboard_all64_rho_magnitude=128, Ref_wedge4_charge=0,
                electric_increment=str(electric_increment), transverse_increment=str(transverse_increment), collective_HC_increment=str(coulomb_increment),
                Pi_only_updated_Gauss_residual_nonzero=True,
                pure_gradient_translation=dict(Calpha_zero=True, path_angle=str(path_angle), hopping_phase='exp(i/3) != 1',
                                               actual_gauge_generator='lambda.(DE-rho)', pure_translation_generator='lambda.DE=lambda.rho on physical states'))


def car_transfer(mask: int, annihilate: int, create: int) -> tuple[int, int] | None:
    if not (mask >> annihilate) & 1 or (mask >> create) & 1:
        return None
    sign = (-1) ** ((mask & ((1 << annihilate) - 1)).bit_count())
    reduced = mask ^ (1 << annihilate)
    sign *= (-1) ** ((reduced & ((1 << create) - 1)).bit_count())
    return reduced | (1 << create), sign


def car_price_control() -> dict[str, Any]:
    # A legal fullPhi point Phi0=diag(1,1,-1,-1), Phi1..3=0,
    # U_path=I has four channels. No Clifford/frozen-field law is substituted.
    masks = [m for m in range(256) if m.bit_count() == 4]
    lookup = {m: i for i, m in enumerate(masks)}
    forward = np.zeros((len(masks), len(masks)), dtype=np.int64)
    B = (1, 1, -1, -1)
    for j, mask in enumerate(masks):
        for color, coeff in enumerate(B):
            action = car_transfer(mask, 4 + color, color)
            if action is not None:
                result, sign = action
                forward[lookup[result], j] += sign * coeff
    number_x = np.diag([(m & 15).bit_count() for m in masks])
    assert np.array_equal(number_x @ forward - forward @ number_x, forward)
    hermitian = forward + forward.T
    # Exact CAR matrix, checked spectral arithmetic only at this small consumer.
    eigenvalues = np.linalg.eigvalsh(hermitian)
    assert np.max(np.abs(eigenvalues)) < 4 + 1e-12 and abs(eigenvalues[-1] - 4) < 1e-12
    delta_norm = float(np.max(np.abs(-2 * eigenvalues)))
    assert delta_norm > 4
    assert delta_norm <= 16
    return dict(fixed_particle_number=4, retained_local_modes=8, CAR_charge_commutator_exact=True,
                B_operator_norm=1, B_nuclear_norm=4, summed_hop_plus_adjoint_norm=float(eigenvalues[-1]),
                phase_pi_actual_delta_norm=delta_norm,
                wrong_grouped_bound_4sin_times_Bopnorm=4,
                correct_target_bound_if_all4_scalar_channels_counted=16,
                sharper_grouped_bound_2sin_times_Bnuclearnorm=8,
                wrong_grouped_bound_rejected=True,
                hypothetical_equal_unit_B_all64_delta_norm=64 * 8,
                hypothetical_equal_unit_B_all64_scalar_bound=64 * 16,
                actual_role_weighted_grouped_price='2|sin(k.alpha/2)| (||B_active,k||_1 + 63 ||B_aux,k||_1)',
                actual_roles_not_assumed_equal=True,
                interpretation='The target explicitly says every spin/colour path. Its bound is safe for rank-one scalar CAR channels, NOT for one grouped c_xdag B4x4 c_y multiplied by ||B||op. Large-volume fullPhi values are not replaced by this pointwise operator control.')


def bundle_and_regular_control() -> dict[str, Any]:
    # Fixed nonzero n.b=1 gives a discontinuous invariant Weyl characteristic.
    s = Fraction(1, 32)
    assert s.denominator != 1
    t, alpha, B = sp.symbols('Theta alpha B', real=True)
    shifted_hop = sp.exp(sp.I * alpha) * B * sp.exp(sp.I * t)
    assert sp.simplify(shifted_hop.subs(alpha, 2 * sp.pi) - B * sp.exp(sp.I * t)) == 0
    assert sp.simplify(shifted_hop.subs(alpha, sp.Rational(1, 3)) - B * sp.exp(sp.I * t)) != 0
    return dict(deck_invariant_Weyl_small_s=str(s), character_forced_zero_at_s=True, character_at_zero=1,
                no_original_L2_normal_translation_eigenvacuum=True,
                charge1_flux_cocycle_integer_m='exp(i 2pi m)=1', noninteger_example='exp(i pi)=-1',
                charts_or_punctured_domains_required=True, original_disorder_map_supplied=False,
                finite_external_phase_absorption='For a specified periodic edge cochain A entering only native charge1 Peierls phases, H[A]=V(A) H[0] V(A)^*, since all theta-free bosonic terms and E^2 are unchanged. This is equality inside the original domain, not all-bundle equivalence or stiffness no-go.',
                sector_warning='An unconstrained finite lattice U1 phase field or movable Dirac sheet does not by itself retain a Chern label. Nontrivial bundle sectors require exact lifted charts/cocycles and constrained domains kept fixed under admissible transformations; no inter-sector map/domain/energy is supplied.')


def run(request_path: Path, output: Path) -> None:
    request = json.loads(request_path.read_text())
    inputs = {name: sha(ROOT / name) for name in request['inputs']}
    if inputs != request['inputs']:
        raise ValueError('Pinned input bytes changed')
    result = dict(kind='INDEPENDENT_ACTUAL_LAW_MAGNETIC_ALGEBRA_REVIEW_CONTROL_NOT_GROUND',
                  source_sha256=sha(Path(__file__)), request_sha256=sha(request_path), input_sha256=inputs,
                  lattice=[lattice_control(n, request['seed'] + n) for n in request['N']],
                  fullGauss=rational_gauss_control(), CAR_price=car_price_control(),
                  representations=bundle_and_regular_control(),
                  target_main_or_solver_run=False, full_ground_or_phase_or_pole_admitted=False)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(receipt=output.name, source_sha256=result['source_sha256'], N=request['N'], ground_or_pole=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.output)
