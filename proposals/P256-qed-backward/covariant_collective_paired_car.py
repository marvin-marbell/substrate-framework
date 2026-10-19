#!/usr/bin/env python3
"""NEW full-CAR ALL36 Ref0->Ref1 removal and its Ref4->Ref3 C partner.

The empty-reference neutral protocol has the SAME body operator as the frozen
filled-reference protocol, not an empty reference retrofitted onto its source.
Proof scope: compact native one-cycle/cut1/Phi3, no macro or real-EM admission.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, eye, kron

import covariant_native_collective_primitive_electric as frozen
from covariant_native_paired_charge_conjugation import conjugate_occupation

ROOT = Path(__file__).resolve().parent
FROZEN_SOURCE_SHA = '085c316c7ae4cf64d4cb9e09ee8429294db191602f26addacaabd0626c5681ea'
CAPSULE_PATH = ROOT / 'covariant_native_collective_primitive_electric_cut1_UE1_capsule_20261008.json'
CAPSULE_SHA = '6000a6b4a6d9e6e91c9b1f0c607a3b4708c41c878a7aa24cf08ff17182412b05'
CAPSULE_BYTES_AT_IMPORT = CAPSULE_PATH.read_bytes()
if hashlib.sha256(CAPSULE_BYTES_AT_IMPORT).hexdigest() != CAPSULE_SHA:
    raise RuntimeError('Frozen ALL36 primitive-electric capsule changed')
CAPSULE = json.loads(CAPSULE_BYTES_AT_IMPORT)
SOURCE_BYTES_AT_IMPORT: dict[str, bytes] = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if isinstance(filename, str):
        path = Path(filename).resolve()
        if path.suffix == '.py' and path.parent == ROOT:
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()
if hashlib.sha256(SOURCE_BYTES_AT_IMPORT['covariant_native_collective_primitive_electric.py']).hexdigest() != FROZEN_SOURCE_SHA:
    raise RuntimeError('Frozen original Ref4->Ref3 supplier changed')


def binary_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


BINARY_U_AT_IMPORT = [dict(path=row['disk_path'], bytes=Path(row['disk_path']).stat().st_size,
                         sha256=binary_hash(Path(row['disk_path'])), sector=sector,
                         copy_sign=row['copy_exchange_sign'], dimension=row['ALL_sector_columns'])
    for sector, rows in CAPSULE['actual_ALL_sector_factor_diagnostics'].items() for row in rows]
PATHS = ((0, 0, 0, 0), (-1, 0, 0, 0), (-1, -1, 0, 0), (0, 0, 0, 1))
SIGMA_X = np.array([[0., 1.], [1., 0.]])


def physical_conjugation(source: dict[str, Any], target: dict[str, Any], reference: bool) -> csr_matrix:
    """C on Phi3 and ALL36; reference=True maps Ref1 labels to Ref3 holes."""
    ns, nt = len(source['physical']), len(target['physical'])
    rows, columns, entries = [], [], []
    for column, (state, flux, electric) in enumerate(source['physical']):
        light, a, b = source['states'][state]
        lc, ls = conjugate_occupation(light, 8)
        ac, asign = conjugate_occupation(a, 4)
        bc, bsign = conjugate_occupation(b, 4)
        target_state = target['state_index'][(lc, ac, bc)]
        row = target['index'][(target_state, -flux)]
        if target['physical'][row][2] != tuple(-value for value in electric):
            raise ArithmeticError('C fails all-four-edge electric reversal')
        rho = frozen.collective.density(light, a, b, source['charge'])
        if frozen.collective.density(lc, ac, bc, target['charge']) != tuple(-value for value in rho):
            raise ArithmeticError('C fails collective density reversal')
        body_sign = ls * asign * bsign
        for node in range(3):
            if reference:
                # FULL20 Hodge: Ref1 crosses 16-7=9 complementary body modes.
                for ref in range(4):
                    rows.append((ref * 3 + node) * nt + row)
                    columns.append((ref * 3 + node) * ns + column)
                    entries.append(-body_sign * (-1) ** ref)
            else:
                rows.append(node * nt + row); columns.append(node * ns + column)
                entries.append(body_sign)
    return csr_matrix((entries, (rows, columns)),
                      shape=((12 if reference else 3) * nt, (12 if reference else 3) * ns))


def annihilator(neutral: dict[str, Any], removal: dict[str, Any], coefficients: np.ndarray,
                embedding: np.ndarray) -> csr_matrix:
    """r† W_inverse a_negative, ONLY NEW Ref0/background0 -> Ref1.

    CAR ordering is active8, aux1four, aux2four, staticRef4. The active
    annihilation leaves 3+2+2=7 body occupations before creating into EMPTY Ref.
    """
    nn, nr = len(neutral['physical']), len(removal['physical'])
    weights = (coefficients @ embedding.conj().T).conj()
    rows, columns, entries = [], [], []
    for column, (state, _, electric) in enumerate(neutral['physical']):
        light, a, b = neutral['states'][state]
        mask = sum(1 << mode for mode in light)
        for mode in light:
            target_light = tuple(m for m in light if m != mode)
            target = removal['state_index'][(target_light, a, b)]
            flux = tuple(e - w for e, w in zip(electric, PATHS[mode // 2]))
            row = removal['index'].get((target, flux[3]))
            if row is None:
                continue
            if removal['physical'][row][2] != flux:
                raise ArithmeticError('Reversed original Wilson path fails electric field')
            rho = frozen.collective.density(target_light, a, b, -1)
            if any(flux[x] - flux[(x - 1) % 4] != rho[x] for x in range(4)):
                raise ArithmeticError('Removal/static counterparticle violates ONE collective Gauss')
            below = (mask & ((1 << mode) - 1)).bit_count()
            sign = (-1) ** (below + len(target_light) + len(a) + len(b))
            for ref in range(4):
                for node in range(3):
                    rows.append((ref * 3 + node) * nr + row)
                    columns.append(node * nn + column)
                    entries.append(sign * weights[mode, ref])
    return csr_matrix((entries, (rows, columns)), shape=(12 * nr, 3 * nn))


def negative_coefficients(gamma: np.ndarray, packet: np.ndarray) -> np.ndarray:
    mean, _ = frozen.collective.native.square_jets(gamma, .09, 0.)
    values, vectors = eigh(mean[0] @ mean[0])
    negative = vectors[:, values > 1]
    if negative.shape[1] != 4:
        raise ArithmeticError('Specified mean negative band is not dimension four')
    template = np.kron(packet[:, None], np.eye(2))
    coefficients = negative @ negative.conj().T @ template
    return coefficients / np.linalg.norm(coefficients, 'fro')


def antiparticle_embedding(embedding: np.ndarray) -> np.ndarray:
    return embedding.conj() @ SIGMA_X


def full_hodge(occupied: tuple[int, ...], modes: int, spin_modes: int) -> tuple[tuple[int, ...], int]:
    """Independent full ordered-CAR Hodge; no body spin exchange on Ref."""
    complement = [mode for mode in range(modes) if mode not in occupied]
    wedge_parity = sum(mode - index for index, mode in enumerate(occupied))
    permuted = [mode ^ 1 if mode < spin_modes else mode for mode in complement]
    inversions = sum(x > y for index, x in enumerate(permuted) for y in permuted[index + 1:])
    return tuple(sorted(permuted)), (-1) ** (wedge_parity + inversions)


def body_occupation(state: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> tuple[int, ...]:
    light, a, b = state
    return light + tuple(8 + mode for mode in a) + tuple(12 + mode for mode in b)


def car_action(occupied: tuple[int, ...], mode: int, create: bool) -> tuple[tuple[int, ...], int]:
    if create == (mode in occupied):
        return occupied, 0
    sign = (-1) ** sum(m < mode for m in occupied)
    target = tuple(sorted((*occupied, mode))) if create else tuple(m for m in occupied if m != mode)
    return target, sign


def exact_car_proof(spaces: list[dict[str, Any]]) -> dict[str, Any]:
    body_checks = removal_checks = addition_checks = neutral_checks = source_checks = 0
    forbidden_filled_ref_creation_checks = 0
    for physical in spaces:
        for state in physical['states']:
            light, a, b = state
            lc, ls = conjugate_occupation(light, 8)
            ac, asign = conjugate_occupation(a, 4)
            bc, bsign = conjugate_occupation(b, 4)
            expected = body_occupation((lc, ac, bc)); body = body_occupation(state)
            observed, sign = full_hodge(body, 16, 16)
            if (observed, sign) != (expected, ls * asign * bsign):
                raise ArithmeticError('Full16 body crossfactor is not +1')
            body_checks += 1
            if physical['charge'] == -1:
                for ref in range(4):
                    target, full_sign = full_hodge(body + (16 + ref,), 20, 16)
                    ref3 = tuple(16 + r for r in range(4) if r != ref)
                    if (target, full_sign) != (expected + ref3, -sign * (-1) ** ref):
                        raise ArithmeticError('Full20 Ref1 Hodge candidate fails')
                    removal_checks += 1
            elif physical['charge'] == 1:
                for ref in range(4):
                    ref3 = tuple(16 + r for r in range(4) if r != ref)
                    if full_hodge(body + ref3, 20, 16) != (expected + (16 + ref,), sign * (-1) ** ref):
                        raise ArithmeticError('Full20 original Ref3 Hodge fails')
                    addition_checks += 1
            else:
                if full_hodge(body, 20, 16) != (expected + (16, 17, 18, 19), sign):
                    raise ArithmeticError('Empty neutral Ref C does not map to filled neutral Ref')
                if full_hodge(body + (16, 17, 18, 19), 20, 16) != (expected, sign):
                    raise ArithmeticError('Original filled neutral Ref C sign fails')
                neutral_checks += 2
                for mode in range(8):
                    below = sum(m < mode for m in light)
                    for ref in range(4):
                        if mode in light:
                            # a first, then r†: the latter crosses seven body modes.
                            after, first = car_action(body, mode, False)
                            _, second = car_action(after, 16 + ref, True)
                            if first * second != (-1) ** (below + 7):
                                raise ArithmeticError('Literal Ref0 removal CAR order fails')
                            filled = body + (16, 17, 18, 19)
                            after, _ = car_action(filled, mode, False)
                            _, forbidden = car_action(after, 16 + ref, True)
                            if forbidden != 0:
                                raise ArithmeticError('Creation on ORIGINAL filled Ref4 must vanish')
                            forbidden_filled_ref_creation_checks += 1
                        else:
                            # r first on Ref4, then a†: no creation on a filled Ref.
                            initial = body + (16, 17, 18, 19)
                            after, first = car_action(initial, 16 + ref, False)
                            _, second = car_action(after, mode, True)
                            if first * second != (-1) ** (below + ref):
                                raise ArithmeticError('Original Ref4 source CAR sign fails')
                        source_checks += 1
    return dict(full16_ALL_body_states_checked=body_checks, body_crossfactor=1,
                full20_Ref1_quartet_checked=removal_checks, full20_original_Ref3_quartet_checked=addition_checks,
                full20_neutral_Ref0_and_Ref4_checked=neutral_checks, literal_full20_source_sign_checks=source_checks,
                actual_forbidden_rdag_on_original_Ref4_zero_checks=forbidden_filled_ref_creation_checks,
                Ref1_cross_sign=-1, Ref1_cross_exponent=16 - 7,
                removal_C_sign='-bodySign*(-1)^r', original_Ref3_C_sign='bodySign*(-1)^r',
                removal_source_sign='(-1)^(below+7)', original_addition_source_sign='(-1)^(below+r)',
                reference_spin_swap=False, reference_label_is_unchanged=True)


def matter_permutation(source: dict[str, Any], target: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows, signs = [], []
    for light, a, b in source['states']:
        lc, ls = conjugate_occupation(light, 8)
        ac, asign = conjugate_occupation(a, 4)
        bc, bsign = conjugate_occupation(b, 4)
        rows.append(target['state_index'][(lc, ac, bc)]); signs.append(ls * asign * bsign)
    return np.argsort(rows), np.array(signs)


def point_proof(gamma: np.ndarray, removal: dict[str, Any], addition: dict[str, Any]) -> list[dict[str, Any]]:
    order, signs = matter_permutation(removal, addition)
    target_signs = signs[order]
    rows = []
    for normal in (-.09, 0., .09):
        actual = gamma.copy(); actual[1] *= 1 + normal
        for theta in (.37, 1.23, 2.71):
            ph, pp = frozen.collective.matched_tree_half(actual, theta, 0.)
            mh, mp = frozen.collective.matched_tree_half(actual, -theta, 0.)
            ah, ap = frozen.collective.auxiliary_path_half(actual, .04, complex((1 + np.exp(1j * theta)) / 2))
            bh, bp = frozen.collective.auxiliary_path_half(actual, .04, complex((1 + np.exp(-1j * theta)) / 2))
            A = frozen.canonical(ah, ap, np.array(removal['auxiliary']))
            B = frozen.canonical(bh, bp, np.array(addition['auxiliary']))
            three = np.kron(np.kron(frozen.canonical(ph, pp, np.array(removal['light'])), A), A)
            five = np.kron(np.kron(frozen.canonical(mh, mp, np.array(addition['light'])), B), B)
            transformed = three[np.ix_(order, order)] * target_signs[:, None] * target_signs[None, :]
            defect = float(np.linalg.norm(transformed - five) / np.linalg.norm(five))
            half_controls = []
            for label, positive, negative, pref, opposite_pref in (('active8', ph, mh, pp, mp), ('auxiliary4_each', ah, bh, ap, bp)):
                tau = np.eye(len(positive))[np.arange(len(positive)) ^ 1]
                expected = tau @ np.linalg.inv(positive).T @ tau
                block_order = np.r_[np.arange(0, len(positive), 2), np.arange(1, len(positive), 2)]
                blocks = []
                for half in (positive, negative):
                    transfer = (half @ half)[np.ix_(block_order, block_order)]
                    sites = len(half) // 2
                    Bmat = np.linalg.inv(transfer[:sites, :sites])
                    Fmat = -Bmat @ transfer[:sites, sites:]
                    blocks.append((Bmat, Fmat))
                Bplus, Fplus = blocks[0]; Bminus, Fminus = blocks[1]
                det_half = np.linalg.det(positive); det_T = np.linalg.det(positive @ positive)
                half_controls.append(dict(sector=label, det_half_real=float(det_half.real),
                    det_half_imag=float(det_half.imag), det_T_real=float(det_T.real),
                    det_T_imag=float(det_T.imag), det_half_one_abs_defect=float(abs(det_half - 1)),
                    det_T_one_abs_defect=float(abs(det_T - 1)), detB_pref=pref, opposite_detB_pref=opposite_pref,
                    pref_equal_abs_defect=abs(pref - opposite_pref),
                    detB_pref_squared_relative_defect=float(abs(np.linalg.det(Bplus) - pref ** 2) / pref ** 2),
                    B_transpose_relative_defect=float(np.linalg.norm(Bminus - Bplus.T) / np.linalg.norm(Bminus)),
                    F_negative_transpose_absolute_defect=float(np.linalg.norm(Fminus + Fplus.T)),
                    half_inverse_transpose_relative_defect=float(np.linalg.norm(negative - expected) / np.linalg.norm(negative))))
            control_keys = ('det_half_one_abs_defect', 'det_T_one_abs_defect', 'pref_equal_abs_defect',
                            'detB_pref_squared_relative_defect', 'B_transpose_relative_defect',
                            'F_negative_transpose_absolute_defect', 'half_inverse_transpose_relative_defect')
            if max([defect] + [c[key] for c in half_controls for key in control_keys]) > 1.e-10:
                raise ArithmeticError('Actual complete ALL36 nonzero-point SF C identity fails')
            rows.append(dict(Phi=normal, theta=theta, complete_matter_shape=list(three.shape),
                ALL36_before_ONE_Gauss=True, actual_complete_SF_C_relative_defect=defect, onebody_controls=half_controls))
    return rows


def sparse_norm(matrix: Any) -> float:
    return float(np.linalg.norm(matrix.data))


def run() -> dict[str, Any]:
    gamma, embedding, nodes, heat, _, metric = frozen.left.frame()
    removal, neutral, addition = (frozen.space(charge, 1) for charge in (-1, 0, 1))
    Cn = physical_conjugation(neutral, neutral, False)
    Cp = physical_conjugation(removal, addition, True)
    car = exact_car_proof([removal, neutral, addition])
    point_rows = point_proof(gamma, removal, addition)
    packets = [np.array(p) for p in np.eye(4)]
    packets.extend(np.array(CAPSULE['input_descriptor'][key]) for key in ('source_packet', 'receiver_packet'))
    source_rows = []
    Ebar = antiparticle_embedding(embedding)
    for number, packet in enumerate(packets):
        coefficients = negative_coefficients(gamma, packet)
        Sh = annihilator(neutral, removal, coefficients, Ebar)
        Sa = frozen.collective.source(neutral, addition, frozen.collective.native.source_coefficients(gamma, packet), embedding)
        transformed = Cp @ Sh @ Cn.T
        equal = sparse_norm(transformed - Sa); opposite = sparse_norm(transformed + Sa)
        norm = sparse_norm(Sa)
        sign = -1  # Derived full20 C/source order; the opposite sign is an adverse control.
        residual = transformed - sign * Sa
        column_norms = np.sqrt(np.asarray(abs(residual).power(2).sum(axis=0)).ravel())
        wrong = Cp @ annihilator(neutral, removal, coefficients, embedding) @ Cn.T
        wrong_embedding = min(sparse_norm(wrong - Sa), sparse_norm(wrong + Sa)) / norm
        if sparse_norm(residual) / norm > 1.e-10 or wrong_embedding < .1:
            raise ArithmeticError('Actual EVERYneutral-column source C/Ebar control fails')
        source_rows.append(dict(packet_index=number, packet=packet.tolist(), signed_source_identity=sign,
            equal_positive_Frobenius_defect=equal, equal_negative_Frobenius_defect=opposite,
            signed_relative_Frobenius_defect=sparse_norm(residual) / norm,
            maximum_ALLcolumn_absolute_defect=float(column_norms.max()),
            original_source_Frobenius_norm=norm, removal_source_Frobenius_norm=sparse_norm(Sh),
            ALL_neutral_columns=Sa.shape[1], ALL_Ref_quartet_rows=Sa.shape[0], actual_removal_nnz=Sh.nnz,
            wrong_unchanged_embedding_best_signed_relative_defect=wrong_embedding,
            all_four_coherent_Ref_channels=True))
    cn_squared = sparse_norm(Cn @ Cn - eye(3 * len(neutral['physical']), format='csr'))
    cp_unitary = sparse_norm(Cp.T @ Cp - eye(12 * len(removal['physical']), format='csr'))
    Phi_heat = kron(csr_matrix(heat), eye(len(neutral['physical']), format='csr'), format='csr')
    phi_commutator = sparse_norm(Cn @ Phi_heat - Phi_heat @ Cn)
    if phi_commutator:
        raise ArithmeticError('Actual Phi3 heat does not commute with physical C')
    if cn_squared or cp_unitary:
        raise ArithmeticError('Physical C signed permutation fails exact isometry/involution')
    removal_electric_heat = diags(np.tile([np.exp(-sum(e * e for e in E) / 2)
                                         for _, _, E in removal['physical']], 12), format='csr')
    addition_electric_heat = diags(np.tile([np.exp(-sum(e * e for e in E) / 2)
                                          for _, _, E in addition['physical']], 12), format='csr')
    electric_heat_defect = sparse_norm(Cp @ removal_electric_heat - addition_electric_heat @ Cp)
    if electric_heat_defect:
        raise ArithmeticError('Actual ALL4edge electric heat fails charged C intertwining')
    reference_rows = []
    for state in neutral['states']:
        light, a, b = state
        base = np.array(frozen.collective.density(light, a, b, 0))
        for occupation, background in ((0, 0), (4, 4)):
            explicit = base.copy(); explicit[0] += occupation - background
            if not np.array_equal(explicit, base):
                raise ArithmeticError('Neutral reference/background cancellation fails')
    for state in removal['states']:
        light, a, b = state
        body = np.array([sum(m // 2 == x for m in light) - 1 for x in range(4)])
        for aux in (a, b):
            for x in (0, 3):
                body[x] += sum((0 if m < 2 else 3) == x for m in aux) - 1
        body[0] += 1  # Ref1/background0 is the static counterparticle at site0.
        if tuple(body) != frozen.collective.density(light, a, b, -1):
            raise ArithmeticError('N3 charge-1 density omits actual static counterparticle')
    for label, physical in (('neutral', neutral), ('removal', removal), ('addition', addition)):
        reference_rows.append(dict(sector=label, ALL_matter_states=len(physical['states']),
            physical_base_dimension=len(physical['physical']), Phi3_dimension=3 * len(physical['physical']),
            ALL_edge_E_and_rho_reversal_verified=True))
    descriptor = dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',
        governing_raw_plan_epoch='6ebc34a912158254456c535bd10259fca04201b14f446a778bba68d5ef63ecf2',
        pinned_template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        frozen_source_sha256=FROZEN_SOURCE_SHA, frozen_capsule_sha256=CAPSULE_SHA,
        frozen_input_descriptor_sha256=CAPSULE['input_descriptor_sha256'], binary_U_at_IMPORT=BINARY_U_AT_IMPORT,
        NEW_removal_reference='Ref0/background0 -> Ref1; r† only on EMPTY reference',
        preserved_original_reference='Ref4/background4 -> Ref3; unchanged original addition source',
        CAR_order='active8, aux1four, aux2four, staticRef4', auxiliary_copies=2, ALL_auxiliary_states=36,
        active_particle_numbers=[3, 4, 5], dynamic_charged_flavours_including_active=3,
        ONE_Gauss_after_full_tensor=True, electric_cutoff=1, UE=1., no_bare_B_squared=True,
        Phi_nodes=nodes.tolist(), Phi3='threelevels ONEcoordinate', active_kappa=.09, auxiliary_kappa_each=.04,
        original_integer_paths=[list(p) for p in PATHS], removal_reverses_original_paths=True,
        source_packets=[p.tolist() for p in packets], Ebar='embedding.conj() @ sigma_x',
        old_conditional_kappa400_mean_and_metric_held_fixed=True,
        scope='Compact native onecycle/all36/Phi3/cut1 only; no macro/fullSU4/originalrealEM/massless/commoncone/largeN promotion')
    return dict(input_descriptor=descriptor,
        input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        sources_at_IMPORT={name:dict(sha256=hashlib.sha256(data).hexdigest(), bytes=data.decode()) for name, data in SOURCE_BYTES_AT_IMPORT.items()},
        pinned_capsule_at_IMPORT=dict(path=str(CAPSULE_PATH), sha256=CAPSULE_SHA, bytes=CAPSULE_BYTES_AT_IMPORT.decode()),
        runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        physical_mean_descriptor=metric, actual_full16_full20_CAR_proof=car,
        actual_complete_point_SF_C_rows=point_rows, actual_C_source_ALLcolumn_rows=source_rows,
        actual_physical_space_rows=reference_rows, actual_neutral_C_squared_defect=cn_squared,
        actual_charged_C_isometry_defect=cp_unitary,
        actual_neutral_reference_background_cancellation_states=2 * len(neutral['states']),
        actual_removal_static_counterparticle_density_states=len(removal['states']),
        actual_ALL4edge_electric_heat_C_intertwining_defect=electric_heat_defect,
        actual_Phi_heat_C_commutator_defect=phi_commutator, actual_antiparticle_embedding=dict(real=np.real(Ebar).tolist(), imag=np.imag(Ebar).tolist()),
        proof='For every real theta and each Phi3 node, B(-theta)=B(theta)^T and F(-theta)=-F(theta)^T. The block Schur transfer has detT=1 and T(-theta)=tau T(theta)^(-T) tau; its positive half inherits the identity. Exterior Hodge gives Gamma_N3(SFhalf) tensor BOTH complete Gamma_auxN2 factors mapped to N5 at -theta with equal detBpref. The full16 calculation proves the body crossfactor +1. The full20 calculation independently proves the static Ref1 crosssign -1, unchanged Ref labels, and the ORIGINAL Ref4-source CAR signs. E/n->-E and rho->-rho preserve ALL4-edge heat and the cutoff, hence ONEglobalGauss projection; Phi and its heat are untouched. These analytic identities extend the finite point controls to every real angle and Phi3 node, and F/K are conjugate in the same neutral normalization. Ref0/background0 and Ref4/background4 cancel identically in the neutral operator, NOT in their distinct source protocols. ALL source columns explicitly verify the signed C identity for four delta sites and the original normalized source/receiver. No spectral SVD or particle/pole/realEM/macro conclusion is asserted.',
        admitted_macro_fullSU4_originalrealEM_massless_commoncone=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result = run(); result['runtime_seconds'] = time.monotonic() - start
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(stage='actual_ALL36_fullCAR_C_proof_written', output=str(args.output),
        input_descriptor_sha256=result['input_descriptor_sha256'], runtime_seconds=result['runtime_seconds'],
        source_rows=result['actual_C_source_ALLcolumn_rows'])), flush=True)


if __name__ == '__main__':
    main()
