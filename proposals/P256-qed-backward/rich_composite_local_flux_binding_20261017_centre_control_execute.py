"""Actual physical spectra/relative localization, not an approval replay."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_composite_local_flux_binding_20261017_'
spec = importlib.util.spec_from_file_location('local_flux_binding_model', ROOT/(PREFIX+'centre_control_model.py'))
assert spec is not None and spec.loader is not None
model = importlib.util.module_from_spec(spec); sys.modules[spec.name] = model; spec.loader.exec_module(model)


def main() -> None:
    input_path = Path(sys.argv[1]); inp = json.loads(input_path.read_text()); stem = ROOT/(PREFIX+inp['epoch']+'_')
    start = time.time(); candidate = model.Candidate(inp['parameters'])
    h = candidate.assemble()
    sparse.save_npz(str(stem)+'H.npz', h)
    print(json.dumps(dict(stage='actual_H', dimension=h.shape[0], nnz=h.nnz, seconds=time.time()-start)), flush=True)
    compact, compact_weight = candidate.source((0, 0, 0))
    separated, separated_weight = candidate.source((0, 3, 5))
    e, v = eigsh(h, k=inp['low_states'], which='SA', tol=inp['eigen_tolerance'], maxiter=inp['max_iterations'], v0=compact+0.13*separated)
    order = np.argsort(e); e, v = e[order], v[:, order]
    residuals = np.linalg.norm(h@v-v*e, axis=0)
    np.savez_compressed(str(stem)+'lowstates.npz', eigenvalues=e, vectors=v, residuals=residuals)
    ground = candidate.observe(v[:, 0])
    checkpoint = dict(kind='ACTUAL_NEW_LOCAL_FLUX_COUPLED_SPECTRAL_CANDIDATE_PARTIAL_UNTIL_FINAL_RECEIPT',
        input=str(input_path), parameters=inp['parameters'], dimension=candidate.dimension, H_nnz=h.nnz,
        eigenvalues=e.tolist(), residuals=residuals.tolist(), ground=ground,
        source_raw_spin_projection_weights=dict(compact=compact_weight, separated=separated_weight),
        elapsed_seconds=time.time()-start)
    (Path(str(stem)+'checkpoint.json')).write_text(json.dumps(checkpoint, indent=2)+'\n')
    print(json.dumps(dict(stage='first_physical_result', E=e[0], pair_rms=ground['pair_rms'], compact=ground['compact_all_three'], separated=ground['all_separated'], spin2=ground['spin_squared'], seconds=time.time()-start)), flush=True)
    # A same-H allowed separated configuration subspace; NOT an asymptotic threshold.
    allowed_body = np.max(np.array(candidate.body.counts), axis=1) <= 1
    selected = np.flatnonzero(np.repeat(allowed_body, candidate.block))
    hs = h[selected][:, selected]
    es, vs = eigsh(hs, k=1, which='SA', tol=inp['eigen_tolerance'], maxiter=inp['max_iterations'])
    separated_ground = np.zeros(candidate.dimension, dtype=complex); separated_ground[selected] = vs[:, 0]
    np.savez_compressed(str(stem)+'separated_minimum.npz', eigenvalues=es, selected=selected, vectors=vs)
    del hs
    # Spectral preparation is a declared initial state under the actual same H.
    # All included eigenvectors retain full relative positions/fields; no cluster projection.
    trajectories = {}
    for name, source in [('compact', compact), ('separated', separated)]:
        coefficients = v.conj().T@source
        raw_weight = float(np.vdot(coefficients, coefficients).real)
        if raw_weight <= 1e-20:
            raise ValueError(f'True null chosen spectral source {name}; no artificial normalization')
        coefficients /= np.sqrt(raw_weight)
        records = []
        for t in inp['times']:
            state = v@(coefficients*np.exp(-1j*e*t))
            observation = candidate.observe(state); observation['absolute_time'] = t
            records.append(observation)
        trajectories[name] = dict(spectral_raw_norm2=raw_weight, included_eigenstates=len(e), observations=records)
    # Vacuum/reference has no valence quarks. Its local sea/deformation is active.
    zero = (0,)*8
    href = candidate.field_block(zero).toarray()
    eref, vref = np.linalg.eigh(href)
    np.savez_compressed(str(stem)+'neutral_reference.npz', eigenvalues=eref, ground=vref[:, 0])
    source_weights = {}
    vacuum = vref[:, 0].reshape(candidate.ncloud, candidate.nflux)
    for site in range(8):
        # Bare normalized epsilon spinhalf baryon does not move primitive fields.
        # Its projected norm is measured, not relabelled as the full canonical norm1.
        overlap = candidate.local.local_product((site,), [candidate.local.overlaps[3, 0]])
        projected = overlap@vacuum
        source_weights[str(site)] = float(np.vdot(projected, projected).real)
    result: dict[str, Any] = dict(checkpoint, kind='EXECUTED_FINITE_LOCAL_CENTRE_FLUX_BINDING_CANDIDATE_NOT_PROTON',
        same_H_separated_minimum=float(es[0]), finite_separation_variational_cost=float(es[0]-e[0]),
        separation_meaning='Minimum same-H expectation over legal total-triality0 states with all3 positions distinct. A constrained finite relative cost, NOT free coloured fragments or an asymptotic breakup pole.',
        separated_minimum_observables=candidate.observe(separated_ground), trajectories=trajectories,
        reference_energy=float(eref[0]), addition_energy=float(e[0]-eref[0]),
        canonical_source_full_norm2=1., canonical_source_projected_norm2_by_site=source_weights,
        canonical_source_exterior_meaning='Full normalized bare epsilon source on empty valence band has CAR norm1; the remaining norm is outside the declared field/colour Galerkin ansatz, not deleted physical strength.',
        local_basis_energy_arrays=[x.tolist() for x in candidate.local.energies],
        local_basis_bare_top_probability=[np.diag(x['top']).tolist() for x in candidate.local.ops],
        colour_count_patterns=len(candidate.flux.cache), units='hbar=a=omega_reference=1; q_u=2/3,q_d=-1/3 primitive assumptions',
        elapsed_seconds=time.time()-start, peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (Path(str(stem)+'result.json')).write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(str(stem)+'representation.npz', positions=candidate.body.positions,
        states=np.array(candidate.body.states), counts=np.array(candidate.body.counts), cloud_bits=candidate.local.bits,
        local_vectors=np.array(candidate.local.vectors), local_eigenvalues=np.array(candidate.local.energies))
    print(json.dumps(dict(stage='physical_complete', epoch=inp['epoch'], E=float(e[0]), E_separated=float(es[0]),
        finite_separation_cost=float(es[0]-e[0]), seconds=time.time()-start, peak_KiB=result['peak_KiB'])), flush=True)


if __name__ == '__main__':
    main()
