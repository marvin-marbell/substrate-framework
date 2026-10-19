"""Execute the actual non-Abelian colour/sea/local-field candidate first."""
from __future__ import annotations
import importlib.util
import json
import itertools
from pathlib import Path
import resource
import sys
import time
from typing import Any
from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_composite_local_flux_binding_20261017_'
spec = importlib.util.spec_from_file_location('local_flux_binding_law', ROOT/(PREFIX+'model.py'))
assert spec is not None and spec.loader is not None
model = importlib.util.module_from_spec(spec); sys.modules[spec.name] = model; spec.loader.exec_module(model)


def split_channels(law: Any, wave: Any) -> dict[Any, Any]:
    result: dict[Any, Any] = defaultdict(dict)
    for key, value in wave.items():
        result[law.quantum_channel(key)][key] = value
    return dict(result)


def source_words(law: Any, p: dict[str, Any], neutral: bool = False) -> tuple[list[Any], list[Any]]:
    columns, metadata = [], []
    seeds = [('vacuum', {0: 1.+0j}, 1.)] if neutral else [(f'B{site}_{spin}', *law.baryon(site, bool(spin))) for site in range(4) for spin in range(2)]
    if not neutral:
        for site in range(4):
            for spin in range(2):
                bare, _ = law.baryon(site, bool(spin))
                moved = law.action(law.action(bare, 'hop'), 'hop')
                separated = {key: value for key, value in moved.items()
                    if law.quantum_channel(key)[-1] == 0 and max(np.array(law.quantum_channel(key)[:4])+np.array(law.quantum_channel(key)[4:8])) <= 1}
                prepared, weight = model.normalized(separated)
                seeds.append((f'S{site}_{spin}', prepared, weight))
    for name, seed, raw_norm in seeds:
        word = seed
        depth = 1 if name.startswith('S') else p['word_depth']
        for degree in range(depth+1):
            for channel, component in split_channels(law, word).items():
                vector, norm = model.normalized(component)
                columns.append(vector); metadata.append(dict(source=name, degree=degree, channel=channel, raw_component_norm2=norm, raw_source_norm2=raw_norm))
            print(json.dumps(dict(stage='source_word', source=name, degree=degree, support=len(word), columns=len(columns))), flush=True)
            if degree < depth:
                action = law.action(word)
                shift = model.inner(word, action).real/model.inner(word, word).real
                for key, value in word.items():
                    action[key] = action.get(key, 0j)-shift*value
                word, _ = model.normalized(action)
        if p['wilson_loop_seed'] and not name.startswith('S'):
            loop: dict[int, complex] = defaultdict(complex)
            for face in law.faces:
                for oriented in [face, tuple(reversed(face))]:
                    for key, value in law.loop(seed, oriented).items():
                        loop[key] += value
            for channel, component in split_channels(law, dict(loop)).items():
                vector, norm = model.normalized(component)
                columns.append(vector); metadata.append(dict(source=name, degree='REAL_SUM_WILSON_TRIANGLES', channel=channel, raw_component_norm2=norm, raw_source_norm2=raw_norm))
    return columns, metadata


def build(law: Any, p: dict[str, Any], stem: Path, neutral: bool = False) -> dict[str, Any]:
    columns, metadata = source_words(law, p, neutral)
    keys = sorted(set().union(*(set(column) for column in columns)))
    lookup = {key: i for i, key in enumerate(keys)}
    rows, cols, values = [], [], []
    for col, wave in enumerate(columns):
        for key, value in wave.items():
            rows.append(lookup[key]); cols.append(col); values.append(value)
    a = sparse.csr_matrix((values, (rows, cols)), shape=(len(keys), len(columns)))
    del rows, cols, values, columns
    gram = (a.conj().T@a).toarray()
    by_channel: dict[Any, Any] = defaultdict(list)
    for i, row in enumerate(metadata):
        by_channel[tuple(row['channel'])].append(i)
    whitening = []; metric_records = []
    for channel, indices in by_channel.items():
        g = gram[np.ix_(indices, indices)]
        eigenvalues, eigenvectors = eigh(g)
        keep = eigenvalues > p['metric_relative_tolerance']*max(1., eigenvalues[-1])
        for k in np.flatnonzero(keep):
            vector = np.zeros(len(metadata), dtype=complex)
            vector[indices] = eigenvectors[:, k]/np.sqrt(eigenvalues[k]); whitening.append(vector)
        metric_records.append(dict(channel=channel, raw_indices=indices, eigenvalues=eigenvalues.tolist(), retained=int(sum(keep)), discarded_metric_sum=float(sum(eigenvalues[~keep]))))
    transform = np.column_stack(whitening)
    basis = sparse.csr_matrix(a@sparse.csr_matrix(transform))
    # Every basis vector is an exact SU3-invariant operator word/channel projection.
    # Only weak matrix elements within the source union are needed; this is NOT
    # an assertion that the primitive H image outside that union vanishes.
    hrows, hcols, hvalues, row_chunks, col_chunks, value_chunks = [], [], [], [], [], []
    for col, key in enumerate(keys):
        for target, value in law.action_key(key).items():
            row = lookup.get(target)
            if row is not None:
                hrows.append(row); hcols.append(col); hvalues.append(value)
        if len(hvalues) > 200000 or col == len(keys)-1:
            row_chunks.append(np.array(hrows, dtype=np.int32)); col_chunks.append(np.array(hcols, dtype=np.int32)); value_chunks.append(np.array(hvalues))
            hrows, hcols, hvalues = [], [], []
        if col % 10000 == 0:
            print(json.dumps(dict(stage='weak_primitive_H', source_column=col, support=len(keys), rank=basis.shape[1])), flush=True)
    union_h = sparse.csr_matrix((np.concatenate(value_chunks), (np.concatenate(row_chunks), np.concatenate(col_chunks))), shape=(len(keys), len(keys)))
    difference = union_h-union_h.conj().T
    error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.
    if error > 1e-10:
        raise ValueError(f'Literal primitive/field H is not Hermitian: {error}')
    h = (basis.conj().T@(union_h@basis)).toarray()
    actual_metric = (basis.conj().T@basis).toarray()
    metric_error = float(np.linalg.norm(actual_metric-np.eye(len(h))))
    eigenvalues, vectors = eigh(h, actual_metric)
    np.savez_compressed(str(stem)+'spectrum.npz', H=h, metric=actual_metric, raw_Gram=gram,
        whitening=transform, eigenvalues=eigenvalues, eigenvectors=vectors)
    sparse.save_npz(str(stem)+'basis.npz', basis)
    (Path(str(stem)+'primitive_keys.json')).write_text(json.dumps([str(key) for key in keys])+'\n')
    (Path(str(stem)+'source_metrics.json')).write_text(json.dumps(dict(columns=metadata, metric_blocks=metric_records, primitive_union_support=len(keys), rank=len(h), metric_error=metric_error), indent=2)+'\n')
    return dict(keys=keys, lookup=lookup, basis=basis, H=h, metric=actual_metric, eigenvalues=eigenvalues, eigenvectors=vectors,
        primitive_H=union_h, metric_error=metric_error, primitive_hermiticity=error, source_metadata=metadata)


def observe(law: Any, keys: list[int], vector: Any) -> dict[str, Any]:
    probabilities = np.abs(vector)**2
    density, sea, charge, x = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    pcompact, pseparated, radius, electric, spinz, mixed_local, two_pairs = 0., 0., 0., 0., 0., 0., 0.
    cloud_excited, top = 0., 0.
    for key, probability in zip(keys, probabilities):
        if probability == 0:
            continue
        bits = key & model.FERMION_MASK
        net, anti_number = [], 0
        for s in range(4):
            local = bits >> (18*s)
            u, d, anti = (local & 63).bit_count(), ((local >> 6) & 63).bit_count(), ((local >> 12) & 63).bit_count()
            net.append(u+d-anti); anti_number += anti
            density[s] += probability*(u+d+anti); sea[s] += probability*anti
            charge[s] += probability*(2/3*(u-anti)-d/3)
            n = u+d+anti; cloud = (key >> (model.CLOUD_SHIFT+s)) & 1
            x[s] += probability*law.X[n, n][cloud, cloud]
            cloud_excited += probability*cloud; top += probability*law.top[n][cloud]
            for species in range(3):
                up = ((local >> (6*species)) & 7).bit_count(); down = ((local >> (6*species+3)) & 7).bit_count()
                spinz += probability*(up-down)/2
        if anti_number == 0:
            pcompact += probability*(max(net) == 3)
            pseparated += probability*(max(net) == 1)
            for i, j in itertools.combinations(range(4), 2):
                radius += probability*net[i]*net[j]*np.sum((law.positions[i]-law.positions[j])**2)/3
        two_pairs += probability*(anti_number >= 2)
        electric += probability*4/3*sum(((key >> (model.FERMIONS+5*e)) & 31) != 0 for e in range(6))
        # A link-dressed state with local unequal particle/antiparticle numbers
        # is not forced into a local singlet sea; full Gauss couples it to flux.
        mixed_local += probability*(anti_number > 0 and any(n < 0 for n in net))
    lookup = {key: i for i, key in enumerate(keys)}
    x[:] = 0
    for col, key in enumerate(keys):
        counts = law.counts(key & model.FERMION_MASK)
        for site, n in enumerate(counts):
            initial = (key >> (model.CLOUD_SHIFT+site)) & 1
            for final in range(2):
                target = key ^ ((initial ^ final) << (model.CLOUD_SHIFT+site))
                row = lookup.get(target)
                if row is not None:
                    x[site] += float((vector[row].conjugate()*law.X[n, n][final, initial]*vector[col]).real)
    return dict(norm=float(sum(probabilities)), total_charge=float(sum(charge)), inclusive_particle_density=density.tolist(),
        antiparticle_density=sea.tolist(), charge_density=charge.tolist(),
        zero_pair_compact_weight=float(pcompact), zero_pair_all_separated_weight=float(pseparated),
        zero_pair_weighted_pair_radius2=float(radius), at_least_two_sea_pairs=float(two_pairs),
        electric_Casimir=float(electric), primitive_spin_z=float(spinz), cloud_excitation_number=float(cloud_excited),
        local_basis_top_population=float(top), quantum_phi_mean=x.tolist(), mixed_screening_pattern_weight=float(mixed_local))


def main() -> None:
    inp = json.loads(Path(sys.argv[1]).read_text()); stem = ROOT/(PREFIX+inp['epoch']+'_')
    start = time.time(); law = model.Law(inp['parameters'])
    source, raw = law.baryon(0)
    first = law.action(source)
    gauss = [float(model.inner(g, g).real) for site in range(4) for generator in range(8) for g in [law.gauss(first, site, generator)]]
    primitive = dict(bare_dressed_source_norm2=raw, first_H_support=len(first), first_H_max_Gauss2=max(gauss),
        local_cloud_energies=[e.tolist() for e in law.local_energies], bare_top=[x.tolist() for x in law.top])
    (Path(str(stem)+'primitive_receipt.json')).write_text(json.dumps(primitive, indent=2)+'\n')
    if max(gauss) > 1e-20:
        raise ValueError(f'Actual non-Abelian Gauss fails on H source: {max(gauss)}')
    print(json.dumps(dict(stage='actual_SU3_source_and_H', **{k: v for k, v in primitive.items() if k not in ['local_cloud_energies', 'bare_top']})), flush=True)
    data = build(law, inp, stem)
    fullground = data['basis']@data['eigenvectors'][:, 0]
    ground = observe(law, data['keys'], fullground)
    # Same-H separated relative-sector variational minimum. No illegal free colours.
    separated_rows = []
    for i, key in enumerate(data['keys']):
        channel = law.quantum_channel(key)
        if channel[-1] == 0 and max(np.array(channel[:4])+np.array(channel[4:8])) <= 1:
            separated_rows.append(i)
    sep_basis = data['basis'][separated_rows]
    sep_gram = (sep_basis.conj().T@sep_basis).toarray()
    ge, gv = eigh(sep_gram); keep = ge > inp['metric_relative_tolerance']*max(1., ge[-1])
    sep_t = gv[:, keep]/np.sqrt(ge[keep])
    selected_basis = sep_basis@sep_t
    sep_h = (selected_basis.conj().T@(data['primitive_H'][separated_rows][:, separated_rows]@selected_basis))
    sep_values = eigh(sep_h, eigvals_only=True)
    # Microscopic primitive hopping, not an imposed trimer term, moves this packet.
    source_coordinates = np.array([source.get(key, 0j) for key in data['keys']])
    coefficients = data['eigenvectors'].conj().T@(data['basis'].conj().T@source_coordinates)
    trajectories = []
    for t in inp['times']:
        coordinates = data['eigenvectors']@(coefficients*np.exp(-1j*data['eigenvalues']*t))
        state = data['basis']@coordinates
        row = observe(law, data['keys'], state); row['absolute_time'] = t; trajectories.append(row)
    result = dict(kind='ACTUAL_NONABELIAN_LOCAL_DEFORMATION_THREE_QUARK_VARIATIONAL_CANDIDATE', parameters=inp['parameters'],
        primitive=primitive, primitive_support=len(data['keys']), rank=len(data['H']), eigenvalues=data['eigenvalues'].tolist(), ground=ground,
        finite_same_H_separated_energy=float(sep_values[0]), finite_relative_cost=float(sep_values[0]-data['eigenvalues'][0]),
        separation_scope='Allowed gauge-invariant three-distinct-position zero-extra-pair variational states under the SAME microscopic H; constrained finite relative cost, not an infinite-volume fragmentation threshold.',
        compact_dressed_source_trajectory=trajectories, actual_metric_error=data['metric_error'],
        primitive_hermiticity=data['primitive_hermiticity'], elapsed_seconds=time.time()-start,
        peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (Path(str(stem)+'result.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stage='physical_complete', E=result['eigenvalues'][0], ground=ground, separated_E=result['finite_same_H_separated_energy'], seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
