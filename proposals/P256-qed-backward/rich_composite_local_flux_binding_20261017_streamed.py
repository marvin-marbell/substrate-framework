"""Same chosen SU3/cloud/sea candidate, exact streamed weak-FORM contractions.
No weakened H, new state cap, physics change, or full primitive sparse H matrix.
"""
from __future__ import annotations
import gc
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from collections import defaultdict
from typing import Any
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_composite_local_flux_binding_20261017_'


def module(name: str, suffix: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT/(PREFIX+suffix))
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


old = module('binding_original_source_generators', 'execute.py')
model = old.model
native = module('binding_exact_native', 'native.py')


def parameters(law: Any) -> tuple[Any, ...]:
    if law.p['colour_magnetic'] != 0:
        raise ValueError('This native consumer owns the declared pure-electric regime, not nonzero bare plaquette coupling')
    energy = np.array(law.local_energies)
    x = np.array([[law.X[a, b] for b in range(19)] for a in range(19)])
    overlap = np.array([[law.overlap[a, b] for b in range(19)] for a in range(19)])
    x2 = np.array(law.X2)
    ordered = sorted(law.hops, key=lambda term: term[1])
    ptr = [0]
    for mode in range(72):
        ptr.append(ptr[-1]+sum(term[1] == mode for term in ordered))
    info = np.array([[target, edge, a, b, int(adjoint), target_site, source_site] for target, source, edge, a, b, adjoint, value, target_site, source_site, charge in ordered], dtype=np.int64)
    values = np.array([term[6] for term in ordered], dtype=complex)
    u_target = np.full((3, 3, 2, 19), -1, dtype=np.int64); u_value = np.zeros((3, 3, 2, 19))
    for (a, b, adjoint, initial), transitions in law.transitions.items():
        if len(transitions) > 1:
            raise ValueError('PW19 elemental transporter unexpectedly branches; cannot silently omit a matrix element')
        if transitions:
            u_target[a, b, int(adjoint), initial], u_value[a, b, int(adjoint), initial] = transitions[0]
    pairs = np.array([term[:3] for term in law.pairs], dtype=np.int64)
    pair_values = np.array([term[3] for term in law.pairs])
    return energy, x, x2, overlap, np.array(ptr, dtype=np.int64), info, values, u_target, u_value, pairs, pair_values, law.p['constituent_mass'], law.p['colour_electric'], law.p['deformation_gradient']


def key_arrays(keys: list[int]) -> tuple[Any, Any]:
    lo = np.fromiter((key & ((1 << 64)-1) for key in keys), dtype=np.uint64, count=len(keys))
    hi = np.fromiter((key >> 64 for key in keys), dtype=np.uint64, count=len(keys))
    return lo, hi


def native_encoding_receipt(law: Any, arguments: Any) -> dict[str, Any]:
    # A literal coefficient/action reconstruction inside this REAL new run,
    # not an approval replay and not a cut in the physical solver below.
    source, _ = law.baryon(0)
    first = law.action(source)
    seeds = [next(iter(source))]
    for predicate in [lambda k: any(((k >> (72+5*e)) & 31) != 0 for e in range(6)), lambda k: law.quantum_channel(k)[-1] == 1]:
        match = next((key for key in first if predicate(key)), None)
        if match is not None:
            seeds.append(match)
    selected = set(seeds)
    for key in seeds:
        selected.update(law.action_key(key))
    keys = sorted(selected); lookup = {key: i for i, key in enumerate(keys)}
    lo, hi = key_arrays(keys); table = native.make_table(lo, hi)
    identity = sparse.eye(len(keys), dtype=complex, format='csr')
    actual = native.weak_h(lo, hi, identity.indptr, identity.indices, identity.data, len(keys), table, *arguments)
    expected = np.zeros_like(actual)
    for col, key in enumerate(keys):
        for target, value in law.action_key(key).items():
            row = lookup.get(target)
            if row is not None:
                expected[row, col] += value
    error = float(np.max(abs(actual-expected)))
    if error > 2e-12:
        raise ValueError(f'Native representation changes the actual microscopic operator: {error}')
    return dict(selected_primitive_dimension=len(keys), max_literal_coefficient_error=error,
        scope='Diagnostic selection ONLY. Full declared source/relative/sea/colour union below is unchanged; all original numeric coefficients, including tiny values, retained.')


def build(law: Any, inp: dict[str, Any], stem: Path, args: Any) -> dict[str, Any]:
    columns, metadata = old.source_words(law, inp)
    keys = sorted(set().union(*(set(column) for column in columns)))
    lookup = {key: i for i, key in enumerate(keys)}
    rows, cols, values = [], [], []
    for col, wave in enumerate(columns):
        for key, value in wave.items():
            rows.append(lookup[key]); cols.append(col); values.append(value)
    raw_basis = sparse.csr_matrix((values, (rows, cols)), shape=(len(keys), len(columns)))
    del columns, rows, cols, values, lookup
    gram = (raw_basis.conj().T@raw_basis).toarray()
    channels: dict[Any, Any] = defaultdict(list)
    for i, row in enumerate(metadata):
        channels[tuple(row['channel'])].append(i)
    whitening, white_channels, metric_records = [], [], []
    for channel, selected in channels.items():
        ge, gv = eigh(gram[np.ix_(selected, selected)])
        keep = ge > inp['metric_relative_tolerance']*max(1., ge[-1])
        for k in np.flatnonzero(keep):
            vector = np.zeros(len(metadata), dtype=complex); vector[selected] = gv[:, k]/np.sqrt(ge[k])
            whitening.append(vector); white_channels.append(channel)
        metric_records.append(dict(channel=channel, raw_indices=selected, eigenvalues=ge.tolist(), retained=int(sum(keep)), discarded_metric_sum=float(sum(ge[~keep]))))
    transform = np.column_stack(whitening)
    basis = sparse.csr_matrix(raw_basis@sparse.csr_matrix(transform))
    del raw_basis, whitening
    lo, hi = key_arrays(keys)
    np.save(str(stem)+'keys_lo.npy', lo); np.save(str(stem)+'keys_hi.npy', hi)
    del keys
    for name, array in [('basis_indptr', basis.indptr), ('basis_indices', basis.indices), ('basis_values', basis.data)]:
        np.save(str(stem)+name+'.npy', array)
    shape = basis.shape
    del basis
    gc.collect()
    indptr = np.load(str(stem)+'basis_indptr.npy', mmap_mode='r')
    indices = np.load(str(stem)+'basis_indices.npy', mmap_mode='r')
    values_map = np.load(str(stem)+'basis_values.npy', mmap_mode='r')
    basis = sparse.csr_matrix((values_map, indices, indptr), shape=shape, copy=False)
    metric = (basis.conj().T@basis).toarray()
    np.savez_compressed(str(stem)+'source_checkpoint.npz', Gram=gram, whitening=transform, metric=metric, white_channels=np.array(white_channels))
    Path(str(stem)+'source_metrics.json').write_text(json.dumps(dict(raw_columns=metadata, metric_blocks=metric_records,
        primitive_support=shape[0], rank=shape[1], basis_nnz=basis.nnz), indent=2)+'\n')
    print(json.dumps(dict(stage='complete_source_checkpoint', primitive_support=shape[0], rank=shape[1], nnz=basis.nnz)), flush=True)
    table = native.make_table(lo, hi)
    raw_h = native.weak_h(lo, hi, indptr, indices, values_map, shape[1], table, *args)
    anti = float(np.max(abs(raw_h-raw_h.conj().T)))
    if anti > 2e-7:
        raise ValueError(f'Streamed H antihermiticity exceeds accumulation roundoff scope: {anti}')
    h = (raw_h+raw_h.conj().T)/2
    e, v = eigh(h, metric)
    np.savez_compressed(str(stem)+'spectrum.npz', H=h, raw_H=raw_h, metric=metric, eigenvalues=e, eigenvectors=v)
    return dict(lo=lo, hi=hi, table=table, basis=basis, H=h, metric=metric, eigenvalues=e, eigenvectors=v,
        raw_antihermiticity=anti, white_channels=white_channels, raw_metadata=metadata, Gram=gram, whitening=transform)


def source_coordinates(data: Any, wave: Any) -> Any:
    b = np.zeros(data['basis'].shape[1], dtype=complex)
    for key, amplitude in wave.items():
        row = native.find(data['table'], data['lo'], data['hi'], np.uint64(key & ((1 << 64)-1)), np.uint64(key >> 64))
        if row != -1:
            start, stop = data['basis'].indptr[row:row+2]
            b[data['basis'].indices[start:stop]] += data['basis'].data[start:stop].conj()*amplitude
    return b


def observe(law: Any, data: Any, vector: Any, x: Any) -> dict[str, Any]:
    result = native.physical_observe(data['lo'], data['hi'], data['table'], vector, x, np.array(law.top))
    numbers, density, sea, charge, phi, centres, onebody = result
    norm, compact, separated, radius, two_pairs, casimir, sz, cloud_n, top_diag, mixed, zero_pair = numbers
    independent = float(np.sum((onebody/zero_pair)**3)) if zero_pair else None
    return dict(norm=float(norm), total_charge=float(sum(charge)), inclusive_particle_density=density.tolist(), antiparticle_density=sea.tolist(),
        charge_density=charge.tolist(), quantum_phi_mean=phi.tolist(), zero_pair_compact_weight=float(compact),
        zero_pair_all_separated_weight=float(separated), zero_pair_weighted_pair_radius2=float(radius),
        at_least_two_sea_pairs=float(two_pairs), electric_Casimir=float(casimir), primitive_spin_z=float(sz),
        cloud_excitation_number=float(cloud_n), local_basis_top_indicator=float(top_diag), mixed_screening_pattern_weight=float(mixed),
        zero_pair_norm=float(zero_pair), compact_centre_weights=centres.tolist(),
        conditional_zero_pair_onebody=(onebody/zero_pair).tolist() if zero_pair else None,
        conditional_compact_probability=float(compact/zero_pair) if zero_pair else None,
        connected_compact_enhancement=float(compact/zero_pair/independent) if independent else None)


def spatial_rotation(law: Any, data: Any, ground_indices: Any) -> dict[str, Any]:
    # Source-orbit intertwining avoids enumerating an enormous rotated tensor.
    # R H^n B = H^n R B; every retained number projector is transported exactly.
    spin = .5*np.eye(2)-.5j*sum(law.sigma, np.zeros((2, 2), dtype=complex))
    spin = np.diag([1., -1.])@spin@np.diag([1., -1.])  # actual Clebsch source down phase
    permutation = [0, 2, 3, 1]
    metadata = data['raw_metadata']
    lookup = {(row['source'], str(row['degree']), tuple(row['channel'])): i for i, row in enumerate(metadata)}
    rows, cols, vals = [], [], []
    for j, row in enumerate(metadata):
        source = row['source']; family = source[0]; site = int(source[1]); oldspin = int(source[-1])
        channel = tuple(row['channel']); rotated = [0]*9
        for s in range(4):
            rotated[permutation[s]], rotated[4+permutation[s]] = channel[s], channel[4+s]
        rotated[-1] = channel[-1]
        for newspin in range(2):
            target_source = f'{family}{permutation[site]}_{newspin}'
            i = lookup.get((target_source, str(row['degree']), tuple(rotated)))
            if i is None:
                raise ValueError('Declared source/relative channel missing under the physical tetrahedral rotation')
            coefficient = spin[newspin, oldspin]*np.sqrt(metadata[i]['raw_component_norm2']/row['raw_component_norm2'])
            if family == 'S':
                coefficient *= np.sqrt(metadata[i]['raw_source_norm2']/row['raw_source_norm2'])
            rows.append(i); cols.append(j); vals.append(coefficient)
    raw_rotation = sparse.csr_matrix((vals, (rows, cols)), shape=data['Gram'].shape)
    w = data['whitening']
    rotation = w.conj().T@(data['Gram']@(raw_rotation@w))
    g = data['eigenvectors'][:, ground_indices]
    reduced = g.conj().T@rotation@g
    return dict(character=[float(np.trace(reduced).real), float(np.trace(reduced).imag)],
        matrix=[[[float(z.real), float(z.imag)] for z in row] for row in reduced],
        ground_block_unitarity_error=float(np.linalg.norm(reduced.conj().T@reduced-np.eye(len(reduced)))),
        triple_rotation_minus_identity_error=float(np.linalg.norm(np.linalg.matrix_power(reduced, 3)+np.eye(len(reduced)))),
        scope='Actual joint tetrahedral source-word/relative-position/link intertwiner; finite double-group character, NOT exact continuum J or an internal-spin-only rotation.')


def main() -> None:
    inp = json.loads(Path(sys.argv[1]).read_text()); stem = ROOT/(PREFIX+inp['epoch']+'_')
    start = time.time(); law = model.Law(inp['parameters']); args = parameters(law)
    encoding = native_encoding_receipt(law, args)
    Path(str(stem)+'native_encoding.json').write_text(json.dumps(encoding, indent=2)+'\n')
    source, raw_source = law.baryon(0); first = law.action(source)
    gauss = [float(model.inner(g, g).real) for site in range(4) for generator in range(8) for g in [law.gauss(first, site, generator)]]
    if max(gauss) > 1e-20:
        raise ValueError('Actual source/H violates non-Abelian Gauss')
    data = build(law, inp, stem, args)
    e, v = data['eigenvalues'], data['eigenvectors']; metric = data['metric']
    channels = np.array(data['white_channels']); counts = channels[:, :4]+channels[:, 4:8]
    selections = dict(three_separated_zero_pair=(channels[:, -1] == 0) & (np.max(counts, axis=1) <= 1),
        no_zero_pair_compact_support=~((channels[:, -1] == 0) & (np.max(counts, axis=1) == 3)))
    comparisons = {}
    for name, mask in selections.items():
        selected = np.flatnonzero(mask)
        energies = eigh(data['H'][np.ix_(selected, selected)], metric[np.ix_(selected, selected)], eigvals_only=True)
        comparisons[name] = dict(energy=float(energies[0]), cost=float(energies[0]-e[0]), dimension=len(selected))
    ground_indices = np.flatnonzero(abs(e-e[0]) < 1e-8)
    ground_records, spin_records = [], []
    for index in ground_indices:
        state = data['basis']@v[:, index]
        ground_records.append(observe(law, data, state, args[1]))
        s2 = native.physical_spin2(data['lo'], data['hi'], data['table'], state)
        spin_records.append([float(s2.real), float(s2.imag)])
    source_b = source_coordinates(data, source); coefficients = v.conj().T@source_b
    dipole_values = (2/3*channels[:, :4]-channels[:, 4:8]/3)@law.positions
    currents = [-1j*(dipole_values[:, axis, None]-dipole_values[None, :, axis])*data['H'] for axis in range(3)]
    contacts = [-(dipole_values[:, axis, None]-dipole_values[None, :, axis])**2*data['H'] for axis in range(3)]
    trajectory, phase_rows = [], []
    for t in inp['times']:
        coordinate = v@(coefficients*np.exp(-1j*e*t)); phase_rows.append(coordinate)
        observation = observe(law, data, data['basis']@coordinate, args[1])
        observation.update(absolute_time=t,
            owning_uniform_current=[float(np.vdot(coordinate, j@coordinate).real) for j in currents],
            owning_uniform_contact=[float(np.vdot(coordinate, k@coordinate).real) for k in contacts])
        trajectory.append(observation)
    rotation = spatial_rotation(law, data, ground_indices)
    np.savez_compressed(str(stem)+'phase_and_source.npz', source_overlap=source_b, coefficients=coefficients,
        trajectory_coordinates=np.array(phase_rows), times=np.array(inp['times']), currents=np.array(currents), contacts=np.array(contacts))
    bare_source = {}
    for key, amplitude in source.items():
        for cloud in range(2):
            bare_source[key | (cloud << model.CLOUD_SHIFT)] = amplitude*law.overlap[3, 0][cloud, 0]
    bare_b = source_coordinates(data, bare_source)
    result = dict(kind='ACTUAL_STREAMED_SAME_SU3_LOCAL_CLOUD_SEA_COMPOSITE_CANDIDATE', parameters=inp['parameters'],
        representation='EXACT weak-form contraction of all declared original coefficients/channels; weak is NOT weak coupling or a weakened H',
        source_raw_CAR_norm2=raw_source, first_H_max_all32_Gauss2=max(gauss), native_encoding=encoding,
        primitive_support=len(data['lo']), rank=len(e), metric_error=float(np.linalg.norm(metric-np.eye(len(e)))),
        raw_H_antihermiticity=data['raw_antihermiticity'], accumulation_scope='Raw matrix retained; only disclosed tiny floating accumulation Hermitian average before eigensolve.',
        eigenvalues=e.tolist(), ground_block_degeneracy=len(ground_indices), ground_records=ground_records,
        ground_block_primitive_spin2=spin_records, ground_block_joint_rotation=rotation,
        finite_same_H_comparisons=comparisons, compact_dressed_source_trajectory=trajectory,
        canonical_bare_source_local_projection_norm2=float(model.inner(bare_source, bare_source).real),
        canonical_bare_source_Galerkin_projection_norm2=float(np.vdot(bare_b, np.linalg.solve(metric, bare_b)).real),
        canonical_source_scope='Full bare canonical epsilon source on prepared empty-sea field reference has recorded CAR norm; dressed clock0 source is DISTINCT. Missing bare source strength is outside field/source ansatz, not removed physical strength.',
        separation_scope='Legal gauge-invariant SAME-H finite relative-sector constrained costs. Pair+single-inclusive comparison retains ALL generated sea/cloud/colour channels; not a global/infinite-volume breakup threshold.',
        EM_scope='Uniform external current/contact probe only, no dynamical EM/photon dressing; exact owning Peierls derivative in charge-channel-closed space, not momentum or imposed trimer current.',
        elapsed_seconds=time.time()-start, peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    Path(str(stem)+'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stage='physical_complete', E=float(e[0]), ground=ground_records[0], comparisons=comparisons,
        rotation=rotation, seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
