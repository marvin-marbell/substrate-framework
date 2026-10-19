#!/usr/bin/env python3
"""Complete finite real-cover mass scan, exact89 control and NEW bare-electric.

No mass floor, sector shift, source gate or pole inference. The input mass is
changed only through the captured CartesianCWilson constructor. Complete
spectra include chart and outside columns. All energies share ONE neutral norm.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.linalg import eigh, expm
from scipy.optimize import OptimizeResult, brentq, minimize_scalar

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = Path(__file__).read_bytes()
CORE_PATH = ROOT/'physical_c_bare_electric_kernel.py'
CORE_BYTES = CORE_PATH.read_bytes()
CORE_HASH = '6a66ca3a45b05e8df46cc1cec90aa775d97cd60341186639fdaf1dfd5fce9e64'
if hashlib.sha256(CORE_BYTES).hexdigest() != CORE_HASH:
    raise RuntimeError('Mass experiment requires the agreed NEW bare-electric epoch')
INPUT_NAMES = ('physical_c_real_cover_kernel_smooth_packet_capsule.json',
               'physical_c_completed_ledger_same_halfstep_capsule.json',
               'physical_alpha_compton_Cartesian_final_native_capsule.json')
INPUT_BYTES = {name: (ROOT/name).read_bytes() for name in INPUT_NAMES}
spec = importlib.util.spec_from_file_location('_physical_C_mass_bare_electric', CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError('Cannot import actual bare-electric primitive')
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
if core.SOURCE_BYTES != CORE_BYTES:
    raise RuntimeError('Core changed across IMPORT boundary')
reference = core.reference
alpha = core.alpha
OLD_MASS = core.BareElectricParameters().bare_mass
PLAN_ID = 'issue-222-induced-photon-continuation-20261008'


def projected_transfer_factor(p: Any, number: int, centre: int, field: dict[str, Any],
                              construction: str, coupling: str) -> dict[str, Any]:
    """Diagnostic coupling deletion ONLY, not a proposed new field law.

    All five coordinates, their original heat, full longitudinal Coulomb and
    full-phase Psi remain. Project only the phases seen by BOTH transfer
    potentials; retain whole scalar prefactor and valence at every point.
    """
    law, phi0, direction = reference.native_law(core.reference_parameters(p))
    graph = reference.real_graph()
    e = alpha.charge_contract(alpha.Parameters())['bare_e']
    _, masks = alpha.second_quantized(np.eye(8), number)
    size = len(masks)
    rho = np.array([[(mask & 15).bit_count()-2, (mask >> 4).bit_count()-(number-2)] for mask in masks], float)
    if np.max(abs(np.sum(rho, axis=1))) > 0:
        raise ArithmeticError('Diagnostic retains the exact original Ref Gauss compensation')
    coefficient = p.electric_coefficient*(1. if construction == 'NEW_bare_electric' else .5)
    coulomb = coefficient*e**2*np.einsum('ni,ij,nj->n', rho, graph['Lplus'], rho)
    active = slice(0, 2) if coupling == 'coexact_only' else slice(2, 5)
    charts = []; outsides = []; packets = []; chis = []
    for cover, y, xi in field['points']:
        source_phases = (e*graph['Q']@cover).reshape(2, 3)
        transfer_phases = (e*graph['Q'][:, active]@cover[active]).reshape(2, 3)
        colour = np.broadcast_to(np.eye(4), (2, 3, 4, 4)).astype(complex).copy()
        colour[0, 0] = np.exp(1j*np.pi*centre/2)*expm(1j*xi*alpha.matrix_unit_basis()[7]/math.sqrt(8))
        remainder = y*np.broadcast_to(direction, (2, 4, 4, 4))
        potential = law.potential(transfer_phases, remainder, colour)
        h, _ = alpha.second_quantized(potential['h'], number)
        chi = reference.cutoff(abs(y), p.chart_radius); chis.append(chi)
        charts.append(chi*expm((h+potential['pref']*np.eye(size))/2))
        outside = law.outside_potential(transfer_phases, np.broadcast_to(phi0, (2, 4, 4, 4))+remainder,
                                       colour, p.outside_couplings)
        h, _ = alpha.second_quantized(outside['H'], number)
        outsides.append(h+outside['offset']*np.eye(size))
        packets.append(law.source_packet_jet(source_phases, colour, (np.zeros((2, 3)), np.zeros((2, 3))))['psi'])
    basis = field['basis']; dim = field['dimension']*size
    chart = reference.compress(basis, np.array(charts))@np.kron(field['R'], np.diag(np.exp(-coulomb/2)))
    coupler = 2*(p.outside_couplings[0]+3*sum(p.outside_couplings[1:]))
    young = 3*coupler**(4/3)/(4**(4/3)*(p.outside_quartic/2)**(1/3))
    quartic = np.array([sum((np.linalg.norm(phi0+y*direction)**2/4)**2 for _ in range(2)) for _, y, _ in field['points']])
    h0 = np.kron(field['Houtside'], np.eye(size))+reference.compress(basis, np.array(outsides))
    h0 += np.kron(basis.T@(p.outside_quartic*quartic[:, None]*basis), np.eye(size))+2*young*np.eye(dim)
    h0 += np.kron(np.eye(field['dimension']), np.diag(coulomb))
    outside = np.kron(basis.T@(np.sqrt(1-np.array(chis)**2)[:, None]*basis), np.eye(size))@expm(-h0/2)
    factor = np.hstack((chart, outside)); k = factor@factor.conj().T
    values, vectors = eigh((k+k.conj().T)/2)
    if values[0] <= 0:
        raise ArithmeticError('Diagnostic complete positive spectrum unresolved; no clipping')
    centre_energy = p.centre_penalty*(1-math.cos(np.pi*centre/2))
    return dict(values=values*math.exp(-centre_energy), vectors=vectors, masks=masks,
        dimension=dim, complete_column_count=factor.shape[1], source_packet_blocks=np.array(packets),
        outside_H_min=float(eigh(h0, eigvals_only=True)[0]), outside_factor_norm=float(np.linalg.norm(outside, 'fro')))


def complete_case(p: Any, construction: str, centres: tuple[int, ...] = (0,),
                  transfer_coupling: str = 'full') -> tuple[dict[str, Any], dict[str, Any], dict[int, list[dict[str, Any]]]]:
    """All eigenpairs of each retained factor, not a source-selected subspace."""
    field = core.field_regulator(p) if construction == 'NEW_bare_electric' else reference.field_regulator(core.reference_parameters(p))
    law, _, _ = reference.native_law(core.reference_parameters(p))
    numbers = (4, 5, 3) if construction == 'NEW_bare_electric' else (4, 5)
    sectors = {}
    for number in numbers:
        if transfer_coupling != 'full':
            sectors[number] = [projected_transfer_factor(p, number, z, field, construction, transfer_coupling) for z in centres]
        elif construction == 'NEW_bare_electric':
            sectors[number] = [core.native_factor(p, number, z, field) for z in centres]
        else:
            sectors[number] = [reference.native_factor(core.reference_parameters(p), number, z, field) for z in centres]
    common = max(float(row['values'][-1]) for row in sectors[4])
    spectra = []
    for number in numbers:
        for z, row in zip(centres, sectors[number], strict=True):
            energy = -np.log(row['values']/common)
            spectra.append(dict(number=number, centre=z, complete_dimension=row['dimension'],
                complete_factor_columns=row['complete_column_count'],
                K_eigenvalues=row['values'].tolist(), energies_TRANSFER=energy.tolist(),
                energies_PHYSICAL=(energy/p.spacing).tolist(),
                minimum_energy_TRANSFER=float(energy[-1]),
                minimum_energy_PHYSICAL=float(energy[-1]/p.spacing),
                outside_H_min=row['outside_H_min'], outside_factor_norm=row['outside_factor_norm']))
    gaps = {str(number): [-math.log(float(row['values'][-1])/common) for row in sectors[number]] for number in numbers if number != 4}
    result = dict(construction=construction, parameters=asdict(p), centres=list(centres),
        B_global_lower=float(law.B_global_lower), primitive_kappa=float(law.kappa),
        ONE_neutral_lambda_N=common, complete_spectra=spectra,
        charged_top_gaps_TRANSFER=gaps,
        charged_top_gaps_PHYSICAL={n: [g/p.spacing for g in values] for n, values in gaps.items()},
        any_negative_charged_energy=any(g < 0 for values in gaps.values() for g in values),
        field_dimension=field['dimension'], real_coordinate_count=5,
        exact_halfheat_omitted_Hermite_operator_norm=field['actual_halfheat_omitted_Hermite_operator_norm'],
        exact89_dual_removal_implemented=construction == 'NEW_bare_electric')
    result['transfer_coupling'] = transfer_coupling
    result['field_quadrature_point_count'] = len(field['points'])
    return result, field, sectors


def source_evidence(p: Any, case: dict[str, Any], field: dict[str, Any], sectors: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Original coherent centre boundary; literal Psi and dual CAR removal."""
    common = case['ONE_neutral_lambda_N']
    law, _, _ = reference.native_law(core.reference_parameters(p))
    mean = law.potential(np.zeros((2, 3)), np.zeros((2, 4, 4, 4)),
                         np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy())
    h, _ = alpha.second_quantized(mean['h'], 4)
    _, vectors = eigh(h)
    sea = vectors[:, -1]
    prepared = []
    for neutral in sectors[4]:
        xi = np.zeros(neutral['dimension'], complex)
        xi[:70] = sea/2
        prepared.append(neutral['vectors']@((neutral['values']/common)**p.prefix_steps*(neutral['vectors'].conj().T@xi)))
    prefix_gram = sum(float(np.linalg.norm(s)**2) for s in prepared)
    if prefix_gram <= 0:
        raise ArithmeticError('Original coherent boundary prefix vanishes')
    prepared = [s/math.sqrt(prefix_gram) for s in prepared]
    preparations = [('original_neutral_packet_no_photon_insertion', prepared, 1.)]
    if field['photon_dimension'] == 6:
        incoming = np.zeros((6, 6))
        incoming[1, 0] = 1
        incoming = np.kron(np.kron(incoming, np.eye(p.normal_order)), np.eye(p.colour_order))
        photons = [(incoming@s.reshape(field['dimension'], 70)).reshape(-1) for s in prepared]
        photon_gram = sum(float(np.linalg.norm(s)**2) for s in photons)
        if photon_gram <= 0:
            raise ArithmeticError('Captured original photon packet insertion vanishes')
        preparations.append(('original_reference_photon_packet', [s/math.sqrt(photon_gram) for s in photons], photon_gram))
    observations = []
    for preparation, states, input_gram in preparations:
        for number in (5, 3):
            if number not in sectors:
                continue
            centre_rows = []
            for z, neutral, charged, state in zip(case['centres'], sectors[4], sectors[number], states, strict=True):
                coefficients = reference.source_coefficients(field, neutral['source_packet_blocks'])
                if number == 5:
                    operator = reference.creation_matrices(neutral['masks'], charged['masks'])
                else:
                    operator = reference.creation_matrices(charged['masks'], neutral['masks']).transpose(0, 2, 1)
                    coefficients = coefficients.conj()
                source = reference.apply_source(coefficients, operator, state)
                spectral = source@charged['vectors'].conj()
                raw_weights = np.sum(abs(spectral)**2, axis=0)
                weights = raw_weights*charged['values']/common
                energies = -np.log(charged['values']/common)
                transfer_window = (energies >= 0) & (energies <= 4)
                physical_window = (energies >= 0) & (energies <= 4*p.spacing)
                gap_branch = abs(energies-energies[-1]) <= 1e-8
                centre_rows.append(dict(centre=z, number=number,
                    complete_source_weights_after_ONE_halfstep=weights.tolist(),
                    complete_raw_source_weights=raw_weights.tolist(),
                    raw_source_Gram=float(np.linalg.norm(source)**2),
                    halfstep_Gram=float(sum(weights)), fullsource_E1_TRANSFER=float(weights@energies),
                    fullsource_E1_PHYSICAL=float(weights@energies/p.spacing),
                    gap_branch_TRANSFER=float(energies[-1]), gap_branch_raw_source_weight=float(sum(raw_weights[gap_branch])),
                    gap_branch_halfstep_source_weight=float(sum(weights[gap_branch])),
                    negative_energy_weight=float(sum(weights[energies < 0])),
                    literal_TRANSFER_Pe0_4_Gram=float(sum(weights[transfer_window])),
                    literal_PHYSICAL_Pe0_4_Gram=float(sum(weights[physical_window]))))
            gram = sum(row['halfstep_Gram'] for row in centre_rows)
            if gram <= 0:
                raise ArithmeticError('Complete original charged source vanishes')
            total = {key: sum(row[key] for row in centre_rows) for key in
                     ('raw_source_Gram', 'halfstep_Gram', 'fullsource_E1_TRANSFER', 'fullsource_E1_PHYSICAL',
                      'gap_branch_raw_source_weight', 'gap_branch_halfstep_source_weight',
                      'negative_energy_weight', 'literal_TRANSFER_Pe0_4_Gram', 'literal_PHYSICAL_Pe0_4_Gram')}
            observations.append(dict(preparation=preparation, number=number, input_packet_Gram=input_gram,
                centres=centre_rows, totals=total, fullsource_mean_TRANSFER=total['fullsource_E1_TRANSFER']/gram,
                fullsource_mean_PHYSICAL=total['fullsource_E1_PHYSICAL']/gram,
                literal_TRANSFER_Pe0_4_fraction=total['literal_TRANSFER_Pe0_4_Gram']/gram,
                literal_PHYSICAL_Pe0_4_fraction=total['literal_PHYSICAL_Pe0_4_Gram']/gram,
                negative_energy_fraction=total['negative_energy_weight']/gram))
    return dict(prefix_Gram=prefix_gram, observations=observations,
        source='CapturedPsi=[exp(iQtheta00)U00;I]/2, Ref4to3 addition; conjugate dual Ref0to1 removal. No source gate or sector shift.',
        units='Hphysical=-log(K/lambdaN)/a; transferPe[0,4] physical[0,400] at a=.01; physicalPe[0,4] transfer[0,.04].')


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = core.BareElectricParameters(photon_grade=args.scan_photon_grade,
        normal_order=args.scan_normal_order, colour_order=1, spacing=args.spacing)
    target = args.target_physical*args.spacing
    masses = args.mass or [-100., -50., -25., -15., -10., -5., OLD_MASS, 0., 5., 20.]
    attempts = []
    cache = {}
    def evaluate(mass: float, construction: str = 'NEW_bare_electric') -> dict[str, Any]:
        key = (construction, float(mass))
        if key not in cache:
            p = replace(base, bare_mass=float(mass))
            row, _, _ = complete_case(p, construction)
            cache[key] = row
            attempts.append(row)
            print(json.dumps(dict(stage='complete_spectrum', construction=construction, mass=mass,
                gaps_TRANSFER=row['charged_top_gaps_TRANSFER'], B_lower=row['B_global_lower'])), flush=True)
        return cache[key]
    for construction in ('exact_REFERENCE89_RPA', 'NEW_bare_electric'):
        for mass in masses:
            evaluate(mass, construction)
    selected = []
    selections = []
    for construction in ('exact_REFERENCE89_RPA', 'NEW_bare_electric'):
        def objective(mass: float) -> float:
            return evaluate(mass, construction)['charged_top_gaps_TRANSFER']['5'][0]
        optimum = cast(OptimizeResult, minimize_scalar(objective, bounds=(min(masses), max(masses)), method='bounded',
                                  options={'xatol': args.mass_tolerance, 'maxiter': 30}))
        optrow = evaluate(float(optimum.x), construction)
        chosen = float(optimum.x)
        ordered = sorted((row for (family, _), row in cache.items() if family == construction),
                         key=lambda r: r['parameters']['bare_mass'])
        brackets = []
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            gl, gr = left['charged_top_gaps_TRANSFER']['5'][0]-target, right['charged_top_gaps_TRANSFER']['5'][0]-target
            if gl*gr < 0:
                brackets.append([left['parameters']['bare_mass'], right['parameters']['bare_mass']])
        roots = []
        if brackets:
            for bracket in brackets:
                root = cast(float, brentq(lambda m: objective(m)-target, *bracket,
                                         xtol=args.mass_tolerance, full_output=False))
                roots.append(dict(bracket=bracket, mass=root, target_residual_TRANSFER=objective(root)-target))
            chosen = roots[0]['mass']
        selections.append(dict(construction=construction, sampled_domain=[min(masses), max(masses)],
            sampled_brackets=brackets, actual_roots=roots, bounded_minimizer_mass=float(optimum.x),
            bounded_minimizer_gap_TRANSFER=optrow['charged_top_gaps_TRANSFER']['5'][0],
            bounded_minimizer_gap_PHYSICAL=optrow['charged_top_gaps_TRANSFER']['5'][0]/args.spacing,
            optimizer_success=bool(optimum.success), optimizer_message=str(optimum.message),
            selection='actual_bracketed_root' if roots else 'bounded_minimum_candidate_NOT_root', selected_mass=chosen))
        selected.append((construction, chosen))
    candidate_checks = []
    for construction, chosen in selected:
        for mass, reason in ((OLD_MASS, 'original_input_control'), (chosen, 'target_derived_scan_candidate')):
            regulators = tuple(dict.fromkeys(((args.scan_photon_grade, args.scan_normal_order), (0, 3), (1, 3))))
            for grade, normal in regulators:
                p = replace(base, bare_mass=mass, photon_grade=grade, normal_order=normal)
                print(json.dumps(dict(stage='all_centres_candidate', construction=construction, mass=mass, grade=grade, normal=normal)), flush=True)
                row, field, sectors = complete_case(p, construction, (0, 1, 2, 3))
                row['candidate_reason'] = reason
                row['source_evidence'] = source_evidence(p, row, field, sectors)
                candidate_checks.append(row)
    return dict(scan_attempts=attempts, selections=selections, all_centres_candidate_checks=candidate_checks,
        allowed_mass_family=dict(B_global_lower='exp(a*m)-1/[8(3+exp(a*m))]>0',
            strict_mass_lower=math.log((math.sqrt(9.5)-3)/2)/args.spacing,
            attempted_domain=[min(masses), max(masses)], no_global_no_go=True),
        limitations=['Finite NF0, all5 real-coordinate quadrature order2 and complete retained Hermite spectra, not continuum convergence.',
            'One Cartesian normal and one colourLie direction; grade0 scan then grade1/normal3 refinement.',
            'Outside semigroup is exp(-P Hout P/2); no harmonic ground, artificial finite mass floor or K spectral clipping.',
            'Static Ref may bind and break translation. No physical electronpole, spin, translation or kineticmass claim.',
            'Exact89 control has addition only; dual removal is implemented only in NEW bare-electric primitive.',
            'Bounded minimization is an attempted finite-domain candidate, not a certified global minimum.'],
        physical_electron_pole_admitted=False)


def operator_cause(args: argparse.Namespace) -> dict[str, Any]:
    """Same-law quadrature convergence, then explicit altered-coupling control."""
    orders = args.quadrature_order or [2, 3, 4]
    couplings = ['full'] if args.investigation == 'quadrature' else ['full', 'coexact_only', 'harmonic_only']
    cases = []; convergence = []
    for construction, selected in [('exact_REFERENCE89_RPA', args.selected_reference_mass),
                                    ('NEW_bare_electric', args.selected_new_mass)]:
        for mass, reason in [(OLD_MASS, 'original_input_control'), (selected, 'grade1_bounded_minimum_NOT_root')]:
            for coupling in couplings:
                previous = None
                for order in orders:
                    p = core.BareElectricParameters(photon_grade=1, normal_order=1, colour_order=1,
                        quadrature_order=order, bare_mass=mass, spacing=args.spacing)
                    row, field, sectors = complete_case(p, construction, transfer_coupling=coupling)
                    row['candidate_reason'] = reason
                    row['source_evidence'] = source_evidence(p, row, field, sectors)
                    cases.append(row)
                    if previous is not None:
                        convergence.append(dict(construction=construction, mass=mass, transfer_coupling=coupling,
                            orders=[previous['parameters']['quadrature_order'], order],
                            addition_gap_change_TRANSFER=row['charged_top_gaps_TRANSFER']['5'][0]-previous['charged_top_gaps_TRANSFER']['5'][0],
                            neutral_lambda_relative_change=row['ONE_neutral_lambda_N']/previous['ONE_neutral_lambda_N']-1,
                            pureS_mean_change_PHYSICAL=row['source_evidence']['observations'][0]['fullsource_mean_PHYSICAL']-
                                previous['source_evidence']['observations'][0]['fullsource_mean_PHYSICAL']))
                    previous = row
                    print(json.dumps(dict(stage='operator_cause_case', construction=construction, mass=mass,
                        quadrature_order=order, transfer_coupling=coupling, gaps_TRANSFER=row['charged_top_gaps_TRANSFER'])), flush=True)
    return dict(investigation=args.investigation, operator_cause_cases=cases, convergence=convergence,
        coupling_deletion_semantics='Delete ONLY harmonic or coexact transfer coupling in chart and outside potentials. Keep all5 real fields, exact original heat and full Coulomb, original full-phase Psi and mean-field sea boundary. Whole scalarpref and Gamma valence retained.',
        source_constraint='Compare pureS and original photon-inserted source separately; photon-prepared dark gap branch is not absent pureS residue.',
        limitations=['Centre0 diagnostic only, not all4centre admission.', 'Fixed six-state photon Hermite basis; quadrature convergence is not basis/continuum convergence.',
            'Native L2 has link harmonic amplitude e*q/sqrt2; no large-volume replacement or fitted width is made.',
            'Altered transfer coupling is a causal control, not a physical Maxwell/electron construction.'],
        physical_electron_pole_admitted=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mass', type=float, action='append')
    parser.add_argument('--spacing', type=float, default=.01)
    parser.add_argument('--target-physical', type=float, default=1.)
    parser.add_argument('--mass-tolerance', type=float, default=.002)
    parser.add_argument('--scan-photon-grade', type=int, choices=(0, 1), default=0)
    parser.add_argument('--scan-normal-order', type=int, default=1)
    parser.add_argument('--investigation', choices=('mass', 'quadrature', 'mode_deletion'), default='mass')
    parser.add_argument('--quadrature-order', type=int, action='append')
    parser.add_argument('--selected-new-mass', type=float, default=-1.1134192897656376)
    parser.add_argument('--selected-reference-mass', type=float, default=-1.1806176541281859)
    args = parser.parse_args()
    start = time.time()
    descriptor = dict(arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        full_default_parameters=asdict(core.BareElectricParameters()), plan_id=PLAN_ID,
        pinned_template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        scoped_work_items=['joined-explanatory-campaign', 'shared-compton-transfer'],
        target_transfer=args.spacing*args.target_physical, original_bare_mass=OLD_MASS,
        core_epoch=CORE_HASH, exact_REFERENCE89_epoch=core.REFERENCE_HASH,
        scan_uses_complete_KN_KC_eigenpairs=True, no_source_fitted_shift=True,
        electric_coefficient=.02, exact89_Coulomb_coefficient=.01, NEW_Coulomb_coefficient=.02)
    input_json = json.dumps(descriptor, sort_keys=True, separators=(',', ':'))
    result = run(args) if args.investigation == 'mass' else operator_cause(args)
    source_map = {Path(__file__).name: SOURCE_BYTES, CORE_PATH.name: CORE_BYTES,
                  'captured_reference89.py': core.REFERENCE_BYTES,
                  'captured_ledger.py': reference.LEDGER_BYTES}
    source_map.update({'captured_Alpha56/'+name: data for name, data in alpha.SOURCE_BYTES.items()})
    startup = {Path(__file__).name: SOURCE_BYTES, CORE_PATH.name: CORE_BYTES, **INPUT_BYTES}
    execution: dict[str, Any] = dict(input_json=input_json, input_sha256=hashlib.sha256(input_json.encode()).hexdigest(),
        source_at_IMPORT={name: hashlib.sha256(data).hexdigest() for name, data in source_map.items()},
        source_bytes_at_IMPORT={name: data.decode() for name, data in source_map.items()},
        capsule_hashes_at_IMPORT={name: hashlib.sha256(data).hexdigest() for name, data in INPUT_BYTES.items()},
        startup_unchanged={name: (ROOT/name).read_bytes() == data for name, data in startup.items()},
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                     platform=platform.platform(), thread_environment={name: os.environ.get(name) for name in
                         ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}),
        runtime_seconds=time.time()-start)
    result['execution'] = execution
    if not all(execution['startup_unchanged'].values()):
        raise RuntimeError('Startup source or captured input changed during experiment')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stage='capsule_written', output=str(args.output), selections=result.get('selections', []),
        input_sha256=result['execution']['input_sha256'], runtime_seconds=result['execution']['runtime_seconds'])), flush=True)


if __name__ == '__main__':
    main()
