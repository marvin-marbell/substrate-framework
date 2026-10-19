#!/usr/bin/env python3
"""Run real research entrypoints and preserve revision-bound numerical evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
import sympy


HERE = Path(__file__).resolve().parent
BASELINE = '14e0b08f6099698b6d117840e65f86cbe70d4efb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Save the same JSON printed to stdout')
    args = parser.parse_args()
    results = {}
    # Receipt key, real entrypoint, CLI arguments; variants remain separate evidence.
    programs = [
        ('compton', 'compton', ()), ('constraint', 'constraint', ()),
        ('quantum_supplier', 'quantum_supplier', ()),
        ('lattice_compton', 'lattice_compton', ()), ('spin_transfer', 'spin_transfer', ()),
        ('bosonic_statistics', 'bosonic_statistics', ()),
        ('rotation_band', 'rotation_band', ()), ('dyon_discriminator', 'dyon_discriminator', ()),
        ('code_phase', 'code_phase', ()), ('sector_stability', 'sector_stability', ()),
        ('sea_selection', 'sea_selection', ('--improved',)),
        ('representation_stability', 'representation_stability', ()),
        ('soft_budget', 'soft_budget', ()),
        ('soft_budget_zero', 'soft_budget', ('--preferred', '0')),
        ('vacuum_feedback', 'vacuum_feedback', ()), ('vacuum_limit', 'vacuum_limit', ()),
        ('vacuum_log', 'vacuum_log', ()), ('photon_matching', 'photon_matching', ()),
        ('photon_matched_compton', 'photon_matched_compton', ()),
        ('angular_photon', 'angular_photon', ()), ('moving_frame', 'moving_frame', ()),
        ('soft_quantum', 'soft_quantum', ()), ('pauli_compton', 'pauli_compton', ()),
        ('photon_projector_trace', 'photon_projector_trace', ()),
        ('photon_infrared', 'photon_infrared', ()), ('elastic_join', 'elastic_join', ()),
        ('spin_orbital_locking', 'spin_orbital_locking', ()),
        ('local_orbital_stability', 'local_orbital_stability', ()),
        ('orbital_hydrodynamics', 'orbital_hydrodynamics', ()),
        ('quantum_orbital_sector', 'quantum_orbital_sector', ()),
        ('quantum_compact_photon', 'quantum_compact_photon', ()),
        ('compact_asymptotics', 'compact_asymptotics', ()),
        ('compact_charge_gap', 'compact_charge_gap', ()),
        ('quantum_spinor_plaquette', 'quantum_spinor_plaquette', ()),
        ('quantum_radial_spinor', 'quantum_radial_spinor', ()),
        ('photon_neutrality', 'photon_neutrality', ()),
        ('wilson_mass_geometry', 'wilson_mass_geometry', ()),
        ('quantum_gauge_mass', 'quantum_gauge_mass', ()),
        ('quantum_compton_response', 'quantum_compton_response', ()),
        ('orbital_gauge_locking', 'orbital_gauge_locking', ()),
        ('quantum_adjoint_mass', 'quantum_adjoint_mass',
         ('--output', 'quantum_adjoint_mass_receipt.json')),
        ('heavy_orbital_join', 'heavy_orbital_join', ('--mass-ratios', '.4', '.2')),
        ('gauged_rotation_fractionalization', 'gauged_rotation_fractionalization', ()),
        ('quantum_restoring_shapes', 'quantum_restoring_shapes', ()),
        ('quantum_su4_gauss_balanced', 'quantum_su4_gauss',
         ('--lambdas', '1', '--heavy-sign', '-1')),
        ('quantum_shape_resolvent', 'quantum_shape_resolvent',
         ('--output', 'quantum_shape_resolvent_receipt.json')),
        ('quantum_shape_narrow', 'quantum_shape_resolvent',
         ('--scales', '800000000', '--full-residual-target', '2e-10',
          '--eigen-tolerance', '2e-14', '--inner-tolerance', '2e-14',
          '--preconditioner-tolerance', '2e-14', '--resolvent-tolerance', '2e-12',
          '--output', 'quantum_shape_narrow_receipt.json')),
        ('exact_haar_alignment', 'exact_haar_alignment',
         ('--cutoffs', 'minimal', 'medium', 'full',
          '--output', 'exact_haar_alignment_receipt.json')),
        ('quantum_su4_photon_aligned', 'quantum_su4_photon',
         ('--R', '2', '--Jpref', '.2',
          '--output', 'quantum_su4_photon_aligned_receipt.json')),
        ('quantum_su4_photon_aligned_domains', 'quantum_su4_photon',
         ('--cutoffs', 'medium', 'full', '--numbers', '4', '--Jpref', '.2',
          '--operators-only', '--output', 'quantum_su4_photon_aligned_domains_receipt.json')),
        ('gauss_charge_intertwiner', 'gauss_charge_intertwiner',
         ('--lambdas', '1', '64', '--output', 'gauss_charge_intertwiner_receipt.json')),
        ('full_field_charge_bound', 'full_field_charge_bound',
         ('--shape', '2', '4', '4', '--output', 'full_field_charge_bound_receipt.json')),
        ('full_field_charge_bound_improved', 'full_field_charge_bound',
         ('--shape', '2', '4', '4', '--improved',
          '--output', 'full_field_charge_bound_improved_receipt.json')),
        ('gauss_born_bundle', 'gauss_born_bundle',
         ('--output', 'gauss_born_bundle_receipt.json')),
        ('full_band_gauss_section', 'full_band_gauss_section',
         ('--m', '.2', '--r', '.12', '--v', '.3', '--stiffness', '2',
          '--heavy-scale', '1', '--geometry',
          '--output', 'full_band_gauss_section_receipt.json')),
        ('gauss_photon_charge', 'gauss_photon_charge',
         ('--R', '2', '--fullground-receipt', 'quantum_su4_photon_aligned_receipt.json',
          '--output', 'gauss_photon_charge_receipt.json')),
        ('su4_sea_backreaction_midpoint', 'su4_sea_backreaction',
         ('--orders', '8', '10', '--output', 'su4_sea_backreaction_midpoint_receipt.json')),
        ('su4_sea_backreaction_split', 'su4_sea_backreaction',
         ('--orders', '8', '10', '--quadrature', 'split_gauss',
          '--output', 'su4_sea_backreaction_split_receipt.json')),
    ]
    artifacts = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.md')) + [HERE / 'proposal.yaml']
    evidence = dict(
        status='active_constructive_theorem_chain',
        execution_state='in_progress',
        source_baseline=BASELINE,
        runtime=dict(python=platform.python_version(), executable=sys.executable,
                     numpy=np.__version__, scipy=scipy.__version__, sympy=sympy.__version__),
        calculation_commands={},
        artifact_manifest_policy='Start-of-run workspace snapshot; completed calculation_commands identifies fresh executions.',
        artifact_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
        calculations=results)

    def save():
        payload = json.dumps(evidence, indent=2, sort_keys=True) + '\n'
        if args.output is not None:
            pending = args.output.with_name(args.output.name + '.tmp')
            pending.write_text(payload)
            pending.replace(args.output)
        return payload

    if args.output is not None:
        save()
    for key, program, arguments in programs:
        try:
            completed = subprocess.run([sys.executable, str(HERE / f'{program}.py'), *arguments],
                                       cwd=HERE, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            evidence['execution_state'] = 'failed'
            evidence['failed_calculation'] = dict(key=key, returncode=error.returncode,
                                                 stderr=error.stderr)
            if args.output is not None:
                save()
            print(error.stderr, file=sys.stderr)
            raise
        results[key] = json.loads(completed.stdout)
        evidence['calculation_commands'][key] = dict(entrypoint=f'{program}.py',
                                                    arguments=list(arguments))
        if args.output is not None:
            save()
    evidence['execution_state'] = 'complete'
    recorded = sorted(path for path in HERE.glob('*.json')
                      if path.name != 'results.json'
                      and (args.output is None or path.resolve() != args.output.resolve()))
    evidence['recorded_receipt_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in recorded}
    evidence['recorded_receipt_policy'] = 'Recorded receipts at completion; hashes alone do not identify fresh executions.'
    print(save(), end='')


if __name__ == '__main__':
    main()
