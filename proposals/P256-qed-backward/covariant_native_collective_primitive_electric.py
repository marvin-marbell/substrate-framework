#!/usr/bin/env python3
"""NEW two-copy quantum auxiliary primitive-electric transfer, BOTH copy sectors.

All 6 x 6 canonical auxiliary states precede ONE total Gauss projection.
Exact copy-exchange block diagonalization retains every singular direction;
no scalar sea squaring, occupation pruning, Gram squaring or fitted gap.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
import scipy
from scipy.linalg import eigh, svd
from scipy.sparse import block_diag, coo_matrix, csr_matrix
import covariant_native_collective_two_auxiliary as collective
import covariant_shared_electric_left_svd as left

ROOT = Path(__file__).resolve().parent
PLAN_PATH = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT = PLAN_PATH.read_bytes()
SOURCE_BYTES_AT_IMPORT: dict[str, bytes] = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if isinstance(filename, str):
        path = Path(filename)
        if path.suffix == '.py' and path.resolve().parent == ROOT:
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()


def space(charge: int, cutoff: int) -> dict[str, Any]:
    light = list(itertools.combinations(range(8), 4 + charge))
    aux = list(itertools.combinations(range(4), 2))
    states = list(itertools.product(light, aux, aux))
    physical = []
    for state, (occupied, a, b) in enumerate(states):
        rho = collective.density(occupied, a, b, charge)
        for n in range(-cutoff, cutoff + 1):
            electric = tuple(n + sum(rho[:j + 1]) for j in range(4))
            if max(map(abs, electric)) <= cutoff:
                physical.append((state, n, electric))
    return dict(charge=charge, light=light, auxiliary=aux, states=states,
                middle=[i for i, s in enumerate(states) if collective.density(*s, charge) == (0, 0, 0, 0)],
                physical=physical, index={(s, n): j for j, (s, n, _) in enumerate(physical)},
                state_index={s: j for j, s in enumerate(states)})


def copy_sector(physical: dict[str, Any], sign: int) -> dict[str, Any]:
    """Exact even-fermion species swap; both signs together resolve ALL36."""
    pairs = [(a, b) for a in range(6) for b in range(a, 6) if sign == 1 or a != b]
    pair_index = {pair: j for j, pair in enumerate(pairs)}
    rr, cc, vv = [], [], []
    for j, (a, b) in enumerate(pairs):
        rr.append(6 * a + b); cc.append(j); vv.append(1. if a == b else 1 / math.sqrt(2))
        if a != b:
            rr.append(6 * b + a); cc.append(j); vv.append(sign / math.sqrt(2))
    auxiliary_map = coo_matrix((vv, (rr, cc)), shape=(36, len(pairs))).toarray()
    aux_lookup = {a: j for j, a in enumerate(physical['auxiliary'])}
    rows, light_ids, pair_ids, flux, electric, pr, pc, pv = [], [], [], [], [], [], [], []
    for r, (state, n, E) in enumerate(physical['physical']):
        occupied, a, b = physical['states'][state]
        ia, ib = aux_lookup[a], aux_lookup[b]
        if ia > ib or (sign == -1 and ia == ib):
            continue
        column = len(rows)
        rows.append(r); light_ids.append(state // 36); pair_ids.append(pair_index[(ia, ib)])
        flux.append(n); electric.append(E)
        pr.append(r); pc.append(column); pv.append(1. if ia == ib else 1 / math.sqrt(2))
        if ia != ib:
            swapped = physical['state_index'][(occupied, b, a)]
            other = physical['index'][(swapped, n)]
            if physical['physical'][other][2] != E:
                raise ArithmeticError('Copy exchange does not preserve the actual global Gauss electric field')
            pr.append(other); pc.append(column); pv.append(sign / math.sqrt(2))
    base_map = coo_matrix((pv, (pr, pc)), shape=(len(physical['physical']), len(rows))).tocsr()
    embedding = block_diag((base_map, base_map, base_map), format='csr')
    occupied = np.array(physical['light'])
    light_ids_array = np.array(light_ids, dtype=int)
    return dict(sign=sign, pairs=pairs, auxiliary_map=auxiliary_map, embedding=embedding,
                light_ids=light_ids_array, pair_ids=np.array(pair_ids, dtype=int),
                flux=np.array(flux, dtype=int), electric=np.array(electric, dtype=int),
                light_n3=np.sum(occupied >= 6, axis=1)[light_ids_array], representative_rows=np.array(rows))


def canonical(half: np.ndarray, pref: float, occupied: np.ndarray) -> np.ndarray:
    return pref * np.linalg.det(half[occupied[:, None, :, None], occupied[None, :, None, :]])


def validate_copy_identity(gamma: np.ndarray, nodes: np.ndarray) -> list[dict[str, Any]]:
    """Actual all-column SF(theta) identity, not a scalar/species multiplier."""
    result = []
    for charge in (0, 1):
        physical = space(charge, 0)
        plus, minus = copy_sector(physical, 1), copy_sector(physical, -1)
        p, m = plus['embedding'], minus['embedding']
        completeness = p @ p.T + m @ m.T - scipy.sparse.eye(p.shape[0], format='csr')
        if completeness.nnz and np.max(abs(completeness.data)) > 1.e-14:
            raise ArithmeticError('Both copy sectors fail complete physical resolution')
        aux = np.array(physical['auxiliary']); occupied = np.array(physical['light'])
        n3 = np.sum(occupied >= 6, axis=1)
        for normal in nodes:
            actual = gamma.copy(); actual[1] *= 1 + float(normal)
            half, pref = collective.matched_tree_half(actual, 0., 0.)
            mean = canonical(half, pref, occupied)
            for theta in (.17, .73, 1.13):
                half, pref = collective.matched_tree_half(actual, theta, 0.)
                observed = canonical(half, pref, occupied)
                expected = mean * np.exp(1j * theta * (n3[:, None] - n3[None, :]))
                defect = float(np.linalg.norm(observed - expected) / np.linalg.norm(observed))
                half, pref = collective.auxiliary_path_half(actual, .04, complex((1 + np.exp(1j * theta)) / 2))
                A = canonical(half, pref, aux)
                both = np.kron(A, A)
                pp, mm = plus['auxiliary_map'], minus['auxiliary_map']
                cross = float(np.linalg.norm(pp.T @ both @ mm) / np.linalg.norm(both))
                reconstruction = pp @ (pp.T @ both @ pp) @ pp.T + mm @ (mm.T @ both @ mm) @ mm.T
                resolution = float(np.linalg.norm(both - reconstruction) / np.linalg.norm(both))
                if max(defect, cross, resolution) > 1.e-10:
                    raise ArithmeticError('Actual nonzero-field light/copy identity failed')
                result.append(dict(active_particles=4 + charge, normal=float(normal), theta=theta,
                                   actual_ALL_light_columns_phase_identity_relative_defect=defect,
                                   actual_ALL36_copy_cross_relative_defect=cross,
                                   actual_ALL36_copy_reconstruction_relative_defect=resolution))
    return result


def streamed_factor(gamma: np.ndarray, nodes: np.ndarray, heat: np.ndarray,
                    physical: dict[str, Any], sector: dict[str, Any], grid: int,
                    UE: float, path: Path) -> tuple[np.memmap, np.ndarray, dict[str, Any]]:
    nr = len(sector['flux'])
    factor = np.memmap(path, mode='w+', dtype=np.complex128, shape=(3 * nr, 3 * nr), order='F')
    light_ids, pairs, n3, flux = (sector[k] for k in ('light_ids', 'pair_ids', 'light_n3', 'flux'))
    electric_energy = UE * np.sum(sector['electric'] ** 2, axis=1)
    electric_half = np.exp(-electric_energy / 2)
    if np.any(electric_half <= 0):
        raise ArithmeticError('Physical electric input underflowed; no zero/fallback/clipping')
    occupied, auxiliary = np.array(physical['light']), np.array(physical['auxiliary'])
    paux = sector['auxiliary_map']
    lower = []
    for p, normal in enumerate(nodes):
        actual = gamma.copy(); actual[1] *= 1 + float(normal)
        half, pref = collective.matched_tree_half(actual, 0., 0.)
        mean = canonical(half, pref, occupied)
        samples = []
        minimum = math.inf
        for theta in 2 * math.pi * np.arange(grid) / grid:
            half, pref = collective.auxiliary_path_half(actual, .04, complex((1 + np.exp(1j * theta)) / 2))
            A = canonical(half, pref, auxiliary)
            minimum = min(minimum, float(eigh(A, eigvals_only=True)[0]))
            samples.append(paux.T @ np.kron(A, A) @ paux)
        lower.append(float(eigh(mean, eigvals_only=True)[0]) * minimum ** 2)
        coefficients = np.fft.fft(np.array(samples), axis=0) / grid
        for start in range(0, nr, 128):
            stop = min(nr, start + 128)
            delta = flux[:, None] - flux[None, start:stop] - n3[:, None] + n3[None, start:stop]
            block = mean[light_ids[:, None], light_ids[None, start:stop]] * coefficients[
                delta % grid, pairs[:, None], pairs[None, start:stop]]
            block *= electric_half[None, start:stop]
            for out in range(3):
                factor[out * nr:(out + 1) * nr, p * nr + start:p * nr + stop] = heat[out, p] * block
        factor.flush()
    if min(lower) <= 0:
        raise ArithmeticError('Nonpositive full quantum canonical auxiliary factor')
    middle = np.flatnonzero(np.all(sector['electric'] == 0, axis=1))
    columns = np.concatenate([middle + p * nr for p in range(3)])
    haar = np.asfortranarray(factor[:, columns])
    return factor, haar, dict(copy_exchange_sign=sector['sign'], ALL_sector_columns=3 * nr,
                              ALL_sector_auxiliary_basis_columns=len(sector['pairs']),
                              actual_point_fullcanonical_SF_positive_lower_bounds=lower,
                              actual_complete_E0_columns=len(columns), actual_factor_disk_bytes=factor.nbytes,
                              actual_electric_half_min=float(electric_half.min()),
                              actual_ALL4edge_electric_energy_max=float(electric_energy.max()), disk_path=str(path))


class Spectrum:
    """Both exact copy blocks, globally sorted, without a dense combined U."""
    def __init__(self, blocks: list[dict[str, Any]], dimension: int):
        self.blocks = blocks
        self.dimension = dimension
        unsorted = np.concatenate([b['q'] for b in blocks])
        order = np.argsort(-unsorted, kind='stable')
        inverse = np.argsort(order)
        self.q = unsorted[order]
        offset = 0
        for block in blocks:
            block['global_indices'] = inverse[offset:offset + len(block['q'])]
            offset += len(block['q'])

    def project(self, wave: np.ndarray) -> np.ndarray:
        result = np.empty(len(self.q), dtype=complex)
        for block in self.blocks:
            result[block['global_indices']] = left.project(block['V'], block['P'].T @ wave)
        return result

    def wave(self, coefficients: np.ndarray) -> np.ndarray:
        result = np.zeros(self.dimension, dtype=complex)
        for block in self.blocks:
            result += block['P'] @ left.physical_wave(block['V'], coefficients[block['global_indices']])
        return result


def observe(n: Spectrum, c: Spectrum, neutral: dict[str, Any], charged: dict[str, Any],
            S: csr_matrix, Sout: csr_matrix, nodes: np.ndarray, Hphi: np.ndarray) -> dict[str, Any]:
    lambdaN = float(n.q[0] ** 2)
    qn, qc = n.q / math.sqrt(lambdaN), c.q / math.sqrt(lambdaN)
    En, Ec = -2 * np.log(qn), -2 * np.log(qc)
    vacuum = np.zeros(len(qn), dtype=complex); vacuum[0] = 1.
    Omega = n.wave(vacuum)
    Bn, Bc = collective.native.curl(neutral, 3), collective.native.curl(charged, 3)
    rawph = Bn @ Omega
    b = n.project(rawph); ph = qn * b
    Zph = float(np.vdot(ph, ph).real)
    if Zph <= 0:
        raise ArithmeticError('Actual collective primitive electric photon preparation is zero')
    ph /= math.sqrt(Zph)
    incoming = n.wave(ph)
    raw = (S @ incoming).reshape(4, c.dimension)
    projected = np.array([c.project(w) for w in raw])
    Xi = qc[None, :] * projected
    M0 = float(np.vdot(Xi, Xi).real)
    if M0 <= 0:
        raise ArithmeticError('Actual collective primitive electric literal source is zero')
    E1 = float(np.sum(abs(Xi) ** 2 * Ec[None, :]))
    R1 = float(np.sum(abs(Xi) ** 2 * qc[None, :] ** 2))
    target = (Sout @ Omega).reshape(4, c.dimension)
    receiver = np.array([c.project(Bc @ w) for w in target])
    consumers = []
    for elapsed in (0., .5, 1., 2.):
        evolved = Xi * np.exp(-1j * elapsed * Ec)[None, :] / math.sqrt(M0)
        amplitude = complex(np.vdot(receiver, evolved))
        consumers.append(dict(time=elapsed, coherent_Ref4_receiver_real=amplitude.real,
                              coherent_Ref4_receiver_imag=amplitude.imag,
                              coherent_Ref4_receiver_probability=abs(amplitude) ** 2,
                              evolved_source_norm_squared=float(np.vdot(evolved, evolved).real)))
    sourcewave = np.array([c.wave(w) for w in Xi])
    null_source = np.array([w - c.wave(z) for w, z in zip(raw, projected)])
    null_photon = rawph - n.wave(b)
    return dict(actual_common_neutral_lambdaN=lambdaN, actual_positive_ranks=[len(qn), len(qc)],
                actual_ALL_neutral_singular_values=n.q.tolist(), actual_ALL_charged_singular_values=c.q.tolist(),
                actual_normalized_min_positive_singular_values=[float(qn[-1]), float(qc[-1])],
                actual_photon_Zph=Zph, actual_photon_energy_mean=float(np.sum(abs(ph) ** 2 * En)),
                actual_photon_spectral_rows=[dict(energy=float(e), normalized_weight=float(w)) for e, w in zip(En, abs(ph) ** 2)],
                actual_M0=M0, actual_E1=E1, actual_R1=R1, actual_energy_mean=E1 / M0,
                actual_transfer_Rayleigh_mean=R1 / M0,
                actual_raw_photon_Knull_norm_squared=float(np.vdot(null_photon, null_photon).real),
                actual_raw_source_Knull_norm_squared=float(np.vdot(null_source, null_source).real),
                actual_absolute_source_band_rows=[dict(absolute_threshold=t, source_fraction=float(np.sum(abs(Xi[:, Ec <= t]) ** 2) / M0)) for t in (.25, 2., 4.)],
                actual_same_literal_source_consumer_rows=consumers,
                actual_vacuum_loads=collective.marginals('actual_Omega', Omega, neutral, nodes, Hphi),
                actual_photon_loads=collective.marginals('actual_collective_photon', incoming, neutral, nodes, Hphi),
                actual_source_loads=collective.marginals('actual_literal_source', sourcewave, charged, nodes, Hphi),
                actual_copy_sector_vacuum_weights=[dict(sign=b['sign'], weight=float(np.sum(abs(vacuum[b['global_indices']]) ** 2))) for b in n.blocks],
                actual_copy_sector_source_weights=[dict(sign=b['sign'], weight=float(np.sum(abs(Xi[:, b['global_indices']]) ** 2) / M0)) for b in c.blocks])


def run(cutoff: int, grid: int, UE: float, scratch: Path) -> dict[str, Any]:
    gamma, embedding, nodes, heat, Hphi, metric = left.frame()
    identities = validate_copy_identity(gamma, nodes)
    neutral, charged = space(0, cutoff), space(1, cutoff)
    spectra, haars, diagnostics, mappings = [], [], {}, []
    for label, physical in (('neutral', neutral), ('charged', charged)):
        blocks, endpoint_blocks = [], []
        sector_diagnostics = []
        for sign in (1, -1):
            sector = copy_sector(physical, sign)
            path = scratch / f'N2_{label}_copy{sign:+d}_cut{cutoff}_leftU.complex128'
            print(json.dumps(dict(stage='streaming_ALL36_two_copy_exact_block', sector=label,
                                  copy_sign=sign, actual_block_dimension=3 * len(sector['flux']))), flush=True)
            factor, haar, diag = streamed_factor(gamma, nodes, heat, physical, sector, grid, UE, path)
            U, q, _ = svd(haar, full_matrices=False, lapack_driver='gesdd')
            if q[-1] <= 0:
                raise ArithmeticError('Actual full Haar endpoint lost a direction')
            endpoint_blocks.append(dict(V=U, q=q, P=sector['embedding'], sign=sign))
            del haar
            print(json.dumps(dict(stage='ALL_copy_sector_singulars_JOBUO_JOBVTN', sector=label, copy_sign=sign)), flush=True)
            singulars, abi = left.lapack_left(factor)
            blocks.append(dict(V=factor, q=singulars, P=sector['embedding'], sign=sign))
            mappings.append(factor)
            diag['actual_overwrite_SVD'] = abi
            sector_diagnostics.append(diag)
            left.evict(factor)
        dimension = 3 * len(physical['physical'])
        spectrum = Spectrum(blocks, dimension)
        if len(spectrum.q) != dimension:
            raise ArithmeticError('All36 copy-sector spectra fail complete canonical dimension')
        spectra.append(spectrum); haars.append(Spectrum(endpoint_blocks, dimension))
        diagnostics[label] = sector_diagnostics
    packet = np.array([.1, .8, .4, .1]); packet /= np.linalg.norm(packet)
    receiver = np.array([.5, .4, .1, .6]); receiver /= np.linalg.norm(receiver)
    S = collective.source(neutral, charged, collective.native.source_coefficients(gamma, packet), embedding)
    Sout = collective.source(neutral, charged, collective.native.source_coefficients(gamma, receiver), embedding)
    full = observe(spectra[0], spectra[1], neutral, charged, S, Sout, nodes, Hphi)
    endpoint = observe(haars[0], haars[1], neutral, charged, S, Sout, nodes, Hphi)
    descriptor = dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',
        raw_plan_sha256_at_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest(),
        branch='NEW true two-copy quantum auxiliary with declared primitive electric UE; NOT an induced-E explanation',
        total_charged_dynamic_flavours_including_tagged_active=3, quantum_auxiliary_copies=2,
        active_modes=8, active_particle_numbers=[4, 5], active_CAR_states=[70, 56],
        auxiliary_modes_each_copy=4, auxiliary_particle_number_each_copy=2,
        ALL_auxiliary_joint_CAR_states=36, exact_copy_exchange_basis_dimensions=[21, 15],
        BOTH_copy_exchange_sectors_retained=True, ONE_total_Gauss_after_ALL36=True,
        neutral_active_plus_aux_CAR_states=2520, charged_active_plus_aux_CAR_states=2016,
        Ref_CAR_modes=4, initial_Ref_occupation=4, prepared_Ref_occupation=3, static_Ref_site=0,
        F='PhiheatHalf [Gamma_light(SFhalf) tensor Gamma_aux1(SFhalf) tensor Gamma_aux2(SFhalf)] exp(-UE sum_ALL4edges E²/2) in ONE globalGauss; K=FFdag',
        primitive_electric_input=UE, NEW_declared_electric_cutoff=cutoff,
        electric_regulator_is_NOT_historical12=True, quadrature_grid=grid,
        active_kappa=.09, auxiliary_kappa_each=.04, light_homotopy=0.,
        both_auxiliary_path_weights=[.5, .5], auxiliary_integer_charge_each=1,
        source_packet=packet.tolist(), receiver_packet=receiver.tolist(),
        same_original_integer_Wilson_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        Pe=[.25,2.,4.], times=[0.,.5,1.,2.], transfer_duration=1., spatial_spacing=1.,
        Phi_nodes=nodes.tolist(), Phi3_is_three_levels_ONE_coordinate=True, Phi_sigma=.03, Phi_radius=.12,
        old_conditional_mean_and_metric_held_fixed_NOT_new_selfconsistent_law=True,
        operator_contract='SAME actual neutral lambdaN; Xi_ph=sqrt(KN/lambdaN) sin(theta) Omega/sqrtZph; Xi_ch=sqrt(KC/lambdaN) S_literal Xi_ph, ONE whole charged halfstep; M0=norm²Xi,E1=<Xi,-log(KC/lambdaN)Xi>,R1=<Xi,KC/lambdaN Xi>; coherent all4Ref receiver',
        Haar_endpoint='ALL total-E0 input columns of SAME NEW factor; UE→infinite endpoint at SAME regulator',
        exact_memory_method='BOTH complete copy-exchange sectors; disk Fortran factor; unsquared zgesvd JOBUO JOBVTN ALL sigma; no Vh/combinedU/Gram/clipping/pruning',
        no_bare_magnetic_action=True, no_speed_mass_quarter_fit=True,
        scope='Finite native one-cycle/electric/allCAR/Phi3. NOT untruncated EM/fullCartesianPhi/SU4/macro/massless/commoncone/largeN completion')
    result = dict(input_descriptor=descriptor,
                  input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
                  source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
                  source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
                  plan_bytes_at_import=PLAN_BYTES_AT_IMPORT.decode(), physical_mean_descriptor=metric,
                  runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                  actual_nonzero_field_ALL36_copy_identity_rows=identities,
                  actual_ALL_sector_factor_diagnostics=diagnostics,
                  actual_primitive_electric_ALL36_transfer=full, actual_same_factor_Haar_E0_endpoint=endpoint,
                  induced_electric_explanation_or_massless_commoncone_macro_admitted=False)
    for mapping in mappings:
        left.evict(mapping)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--electric-cutoff', type=int, required=True)
    parser.add_argument('--haar-grid', type=int, required=True)
    parser.add_argument('--primitive-electric-input', type=float, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.electric_cutoff < 1 or args.haar_grid <= 4 * args.electric_cutoff or args.primitive_electric_input <= 0:
        raise ValueError('Positive primitive input/cutoff and nonaliased declared field grid required')
    args.scratch.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result = run(args.electric_cutoff, args.haar_grid, args.primitive_electric_input, args.scratch)
    result['runtime_seconds'] = time.monotonic() - start
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(source_at_import=result['source_at_import'], input_descriptor_sha256=result['input_descriptor_sha256'],
                          runtime_seconds=result['runtime_seconds'],
                          physical={k:result['actual_primitive_electric_ALL36_transfer'][k] for k in ('actual_positive_ranks','actual_photon_energy_mean','actual_M0','actual_energy_mean','actual_transfer_Rayleigh_mean','actual_raw_source_Knull_norm_squared')})), flush=True)


if __name__ == '__main__':
    main()
