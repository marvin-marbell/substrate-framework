#!/usr/bin/env python3
"""Actual periodic-mean fields in a finite open-cube Z4 flux comparison.

Closed-loop centre fluxes are physical graph data; four lifted centre links are
not four scalar gauge vacua. This probe is neither a bulk flux-gap nor phase proof.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag, expm

from centre_flux_sea import (centre, check, fluxes, heavy_energy, onebody,
                             primitive_pair_control, tree_edges, tree_fix)
from full_field_charge_bound import open_box
from periodic_bulk_frame import source_hashes as inherited_source_hashes
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis
from unrestricted_field_minimum import ALIGNMENT, BARE, STIFFNESS, matrices

ROOT = Path(__file__).resolve().parent


def source_hashes():
    return {**inherited_source_hashes(), **{
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('bulk_frame_flux.py', 'bulk_frame_flux.md')}}


def compare_frame(frame: NDArray[np.complex128], band, graph) -> dict[str, Any]:
    tree = tree_edges(graph)
    chords = tuple(i for i in range(len(graph.bonds)) if i not in tree)
    fields = np.tile(frame, (graph.volume, 1, 1, 1))
    check(float(np.linalg.norm(frame-frame.conj().transpose(0, 2, 1))), 'actual mean Hermiticity')
    check(float(np.linalg.norm(np.trace(frame, axis1=1, axis2=2))), 'actual mean tracelessness')
    site_gauge = np.array([(3*x+1)%4 for x in range(graph.volume)])
    gauge_matrix = np.kron(np.diag([centre(k) for k in site_gauge]), np.eye(4))
    rows = []
    gauge_error = 0.
    minimum_half_cut = float('inf')
    half = 2*graph.volume
    for labels in itertools.product(range(4), repeat=len(chords)):
        exponents = np.zeros(len(graph.bonds), dtype=int)
        exponents[list(chords)] = labels
        links = [centre(k)*np.eye(4) for k in exponents]
        h = onebody(graph, band, fields, links)
        energy, spectrum = heavy_energy(h, 1.)
        check(float(np.linalg.norm(h-h.conj().T)), 'actual graph Hermiticity')
        check(float(abs(np.trace(h))), 'actual graph trace zero')
        face_flux = fluxes(graph, exponents)
        changed = tuple((int(k)+int(site_gauge[x])-int(site_gauge[y]))%4
                        for k, (x, y) in zip(exponents, graph.bonds))
        changed_h = onebody(graph, band, fields, [centre(k)*np.eye(4) for k in changed])
        error = float(np.linalg.norm(changed_h-gauge_matrix@h@gauge_matrix.conj().T))
        check(error, 'actual centre-gauge covariance')
        gauge_error = max(gauge_error, error)
        if fluxes(graph, changed) != face_flux or tree_fix(graph, changed, tree) != tuple(exponents):
            raise AssertionError('Gauge invariant flux or unique tree representative changed')
        if (face_flux[5]-face_flux[4]-face_flux[3]+face_flux[2]+face_flux[1]-face_flux[0])%4:
            raise AssertionError('Oriented cube Bianchi constraint failed')
        gap = float(spectrum[half]-spectrum[half-1])
        minimum_half_cut = min(minimum_half_cut, gap)
        rows.append(dict(chord_exponents=list(labels), face_flux_exponents=list(face_flux),
                         unit_whole_heavy_energy=energy, occupied_empty_half_gap=gap,
                         positive_modes=int(np.sum(spectrum>0)), negative_modes=int(np.sum(spectrum<0))))
    energies = np.array([row['unit_whole_heavy_energy'] for row in rows])
    best = float(energies.min())
    tolerance = 2e-12*max(1.,float(np.max(abs(energies))))
    winners = np.flatnonzero(energies-best<=tolerance)
    above = energies[energies-best>tolerance]
    gram = np.einsum('mij,nji->mn',frame,frame).real/4
    elastic = graph.volume*STIFFNESS/2*float(np.sum(gram**2))
    # Every centre link commutes with the ACTUAL frame, so full W vanishes exactly.
    alignment = max(float(np.linalg.norm(fields[x]-u@fields[y]@u.conj().T))
                    for (x,y),u in zip(graph.bonds,[centre(k)*np.eye(4) for k in range(len(graph.bonds))]))
    check(alignment, 'actual full-field centre alignment zero')
    # A genuine noncentral local transformation also preserves the same operator.
    generators = matrix_unit_basis()/np.sqrt(8)
    gauges = [expm(1j*.19*np.einsum('a,aij->ij',np.sin(np.arange(15)*.43+.17*x),generators))
              for x in range(graph.volume)]
    flat_links = [np.eye(4,dtype=complex) for _ in graph.bonds]
    rotated_fields = np.array([u@frame@u.conj().T for u in gauges])
    rotated_links = [gauges[x]@gauges[y].conj().T for x,y in graph.bonds]
    flat_h = onebody(graph,band,fields,flat_links)
    rotated_h = onebody(graph,band,rotated_fields,rotated_links)
    gauge_block = np.asarray(block_diag(*gauges), dtype=np.complex128)
    noncentral_error = float(np.linalg.norm(rotated_h-gauge_block@flat_h@gauge_block.conj().T))
    check(noncentral_error,'actual noncentral local SU4 covariance')
    wrong_h = onebody(graph,band,fields,rotated_links)
    wrong_field_rotation = float(np.linalg.norm(wrong_h-gauge_block@flat_h@gauge_block.conj().T))
    if wrong_field_rotation <= 1e-5:
        raise AssertionError('Deleted actual field rotation failed to discriminate local covariance')
    return dict(frame_real=frame.real.tolist(),frame_imag=frame.imag.tolist(),
        configurations=len(rows),tree_edges=list(tree),chord_edges=list(chords),rows=rows,
        occupation='minus largest16 of actual32 onebody modes, via faithful primitive exterior identity; not N16 ED',
        minimum_unit_whole_heavy_energy=best,common_elastic_coefficient=elastic,
        minimum_total_leading_coefficient=best+elastic,full_centre_alignment_residual=alignment,
        numeric_minimizer_indices=[int(k) for k in winners],
        numeric_minimizer_face_fluxes=[rows[int(k)]['face_flux_exponents'] for k in winners],
        flat_unique_minimum=bool(len(winners)==1 and int(winners[0])==0),
        next_unit_energy_gap=float(above.min()-best) if len(above) else None,
        minimum_half_filling_cut_gap=minimum_half_cut,
        primitive_two_cell_binding=primitive_pair_control(band,frame),
        centre_gauge_covariance_max=gauge_error,noncentral_local_SU4_covariance=noncentral_error,
        omitted_field_rotation_covariance_defect=wrong_field_rotation,
        ranking_tolerance=tolerance,bare_colour_plaquette_added=False)


def run(input_receipt: Path):
    identities=source_hashes()
    payload=input_receipt.read_bytes()
    receipt=json.loads(payload)
    if receipt['bare_coefficients'] != {**BARE,'stiffness':STIFFNESS,'g0':0.,'Jpref':ALIGNMENT}:
        raise ValueError('Input must identify the actual shared bulk bare law')
    if receipt['source_sha256']['sea_selection.py'] != identities['sea_selection.py']:
        raise ValueError('Input coordinate basis owner changed')
    basis=matrix_unit_basis()
    band=RotationBand(**BARE)
    graph=open_box((2,2,2))
    results=[]
    for row in receipt['same_law_seed_branches']:
        point=row['validation_stationary_candidate']
        value=np.asarray(point['coordinate_values'],dtype=float)
        if value.shape!=(4,15) or not np.all(np.isfinite(value)):
            raise ValueError('Actual input must contain finite all60 coordinates')
        frame: NDArray[np.complex128]=matrices(value,basis)
        results.append(dict(branch=row['name'],input_mesh=point['grid'],
            input_stationarity_reference=dict(gradient_norm=point['gradient_norm'],
                source_sha256=receipt['source_sha256'],not_recomputed_by_this_flux_probe=True),
            flux=compare_frame(frame,band,graph)))
    if source_hashes()!=identities or input_receipt.read_bytes()!=payload:
        raise RuntimeError('Scientific source/proof/input changed during flux execution')
    return dict(source_sha256=identities,source_bindings_unchanged_during_run=True,
        input_receipt=dict(path=str(input_receipt),sha256=hashlib.sha256(payload).hexdigest()),
        geometry='open2x2x2 three-axis cube,12 links,5 independent cycles,1024 Z4 flux classes',
        status='Actual finite periodic-mean coordinates on finite closed-loop flux graph, NOT bulk phase',
        bare_coefficients=receipt['bare_coefficients'],comparisons=results,all_checks_pass=True,
        limits=['Mean stationarity belongs to its separately bound input; flux energies recomputed here.',
            'Centre link copies are gauge equivalent; cycle Wilson fluxes are distinct graph observables.',
            'No continuous SU4/link/field minimization or thermodynamic flux gap/deconfinement is proved.',
            'No physical quartet, quantum phase, charged lower gap or joined electron-photon completion.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-receipt',type=Path,default=ROOT/'periodic_bulk_frame_receipt.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.input_receipt)
    text=json.dumps(result,indent=2,allow_nan=False)
    args.output.write_text(text+'\n')
    print(text)


if __name__=='__main__':
    main()
