#!/usr/bin/env python3
"""One-flavour compact coupled current and path-dressed electron at finite cutoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np

from bulk_frame_compton import band_parameters, reconstruct_frame
from compact_dirac_measure import (GAMMA, I4, current_controls, dirac_matrix,
                                  gauge_transform, geometry, joint_log,
                                  links_from_angles, neutral_transfer_controls,
                                  norm, photon_log, positive_logdet, require,
                                  spatial_hamiltonian, wrap)
from rotation_band import MIXING, RotationBand
from sea_selection import SELECTED

ROOT = Path(__file__).resolve().parent
PLAN_REVISION = 'bbff677d7b2d3b5ddb68187ca868a67f676af68a0b51e8d2eb49b797d967b2e9'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def complex_record(value):
    value = np.asarray(value)
    return dict(real=value.real.tolist(), imag=value.imag.tolist())


def source_point(path: Path, spacing: float):
    payload = path.read_bytes()
    receipt = json.loads(payload)
    if not receipt['all_checks_pass'] or not receipt['source_bindings_unchanged']:
        raise ValueError('Successful immutable cutoff receipt required')
    # Bind the actual selected-family producers, not unrelated evolving proofs.
    for name in ('bulk_cutoff_matching.py', 'bulk_frame_compton.py', 'rotation_band.py',
                 'sea_selection.py', 'soft_budget.py', 'unrestricted_field_minimum.py'):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != receipt['source_sha256'][name]:
            raise ValueError(f'Cutoff receipt source changed: {name}')
    matches = [row for row in receipt['rows'] if row['a'] == spacing]
    if len(matches) != 1:
        raise ValueError('Choose one actual finite spacing from the cutoff receipt')
    row = matches[0]
    bare = RotationBand(**row['bare_band'])
    fields = np.sqrt(np.diag(row['Gram']))[:, None, None] * SELECTED
    band, unitary, reconstruction = reconstruct_frame(bare, fields)
    declared = RotationBand(**row['effective_band'])
    require(norm(np.array([band.a, band.m, band.r, band.v, band.e]) -
                 np.array([declared.a, declared.m, declared.r, declared.v, declared.e])),
            'Reconstructed actual receipt band', 1e-12)
    if band.improved or band.e != 1 or min(band.m, band.r, band.v) <= 0:
        raise ValueError('Positive nearest-axis one-flavour band with e=1 required')
    ue1, k1 = row['physical_Hamiltonian_UE1'], row['physical_Hamiltonian_K1']
    if min(ue1, k1) <= 0:
        raise ValueError('Positive declared Hamiltonian rotor coefficients required')
    binding = dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                   selected_spacing=spacing, family=receipt['family'], Gram=row['Gram'],
                   actual_field_coordinate_values=row['actual_field_coordinate_values'],
                   effective_band=band_parameters(band), UE1=ue1, K1=k1,
                   declared_g2=row['declared_g2'], frame_reconstruction=reconstruction,
                   reason=f'a={spacing} is an actual receipt point; default a=.2 with 3-site spatial sine momentum nonzero and dt=.2 avoids the stiff finest-cutoff Wilson hopping. No infrared/continuum approximation is used in compact sampling.')
    return bare, fields, band, unitary, ue1, k1, binding


def link_blocks(links, shape, neighbours, band, dt):
    """Exact forward/reverse blocks; reverse is NOT forward dagger."""
    coordinates, _ = geometry(shape)
    blocks = []
    for site in range(len(links)):
        for mu in range(4):
            phase = links[site, mu]
            if mu == 0 and coordinates[site, 0] == shape[0] - 1:
                phase = -phase
            w = 1 / dt if mu == 0 else band.r / band.a
            v = 1 / dt if mu == 0 else band.v / band.a
            forward = phase * (v * GAMMA[mu] - w * I4) / 2
            backward = phase.conjugate() * (-v * GAMMA[mu] - w * I4) / 2
            blocks.append((site, int(neighbours[site, mu]), forward, backward))
    return blocks


def local_trace(matrix, block, order):
    x, y, forward, backward = block
    sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
    if order == 1:
        return np.trace(matrix[sy, sx] @ (1j*forward) + matrix[sx, sy] @ (-1j*backward))
    return np.trace(matrix[sy, sx] @ (-forward) + matrix[sx, sy] @ (-backward))


def variation(blocks, coefficients, size, order=1):
    result = np.zeros((size, size), complex)
    for coefficient, block in zip(coefficients, blocks):
        x, y, forward, backward = block
        sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
        if order == 1:
            result[sx, sy] += coefficient * 1j * forward
            result[sy, sx] -= coefficient * 1j * backward
        else:
            result[sx, sy] -= coefficient * forward
            result[sy, sx] -= coefficient * backward
    return result


def dressed_observable(inverse, links, neighbours):
    x = 0
    middle = int(neighbours[x, 0])
    y = int(neighbours[middle, 1])
    path = links[x, 0] * links[middle, 1]
    block = inverse[4*x:4*x+4, 4*y:4*y+4]
    return path.conjugate() * block, (x, middle, y), path


def fermion_consumer(links, shape, neighbours, band, dt, phases, source, *, finite=False):
    matrix = dirac_matrix(links, shape, neighbours, band, dt)
    logdet, sign = positive_logdet(matrix)
    inverse = np.linalg.solve(matrix, np.eye(len(matrix)))
    require(norm(matrix @ inverse - np.eye(len(matrix))), 'Actual fermion inverse', 2e-10)
    blocks = link_blocks(links, shape, neighbours, band, dt)
    gauge_direction = (phases[:, None] - phases[neighbours]).ravel()
    first = np.array([local_trace(inverse, block, 1) for block in blocks])
    contact = np.array([local_trace(inverse, block, 2) for block in blocks])
    divergence = np.zeros(len(links), complex)
    for value, (x, y, _, _) in zip(first, blocks):
        divergence[x] += value
        divergence[y] -= value
    vg = variation(blocks, gauge_direction, len(matrix))
    chi = np.repeat(phases, 4)
    require(norm(vg - 1j*(chi[:, None]*matrix - matrix*chi[None, :])),
            'Exact nonlinear gauge-direction vertex', 2e-10)
    sg = inverse @ vg @ inverse
    bubble_g = np.array([local_trace(sg, block, 1) for block in blocks])
    hessian_g = gauge_direction * contact - bubble_g
    require(norm(divergence), 'Configuration-level current divergence', 2e-9)
    require(norm(hessian_g), 'Configuration-level Hessian gauge Ward', 2e-9)
    omitted_contact = norm(bubble_g)
    # The bad law is passed through the identical Ward assertion, not an unrelated metric.
    adverse_rejected = False
    try:
        require(omitted_contact, 'Configuration-level Hessian gauge Ward', 2e-9)
    except ValueError:
        adverse_rejected = True
    vb = variation(blocks, source, len(matrix))
    sb = inverse @ vb @ inverse
    response = np.dot(source**2, contact) - np.trace(sb @ vb)
    dressed, (x, middle, y), path = dressed_observable(inverse, links, neighbours)
    sl = (slice(4*x, 4*x+4), slice(4*y, 4*y+4))
    path_source = source[4*x] + source[4*middle+1]
    insertion = path.conjugate()*(-sb[sl] - 1j*path_source*inverse[sl])
    path_gauge = phases[x] - phases[y]
    dressed_gauge = path.conjugate()*(-sg[sl] - 1j*path_gauge*inverse[sl])
    require(norm(dressed_gauge), 'Dressed electron gauge-direction insertion', 2e-9)
    undressed_defect = norm(-sg[sl])
    undressed_rejected = False
    try:
        require(undressed_defect, 'Dressed electron gauge-direction insertion', 2e-9)
    except ValueError:
        undressed_rejected = True
    if not undressed_rejected or not adverse_rejected:
        raise ValueError('Wrong current/dressing control survived actual assertion')
    result: dict[str, Any] = dict(single_flavour_logdet=logdet, determinant_sign=sign,
        inverse_residual=norm(matrix @ inverse - np.eye(len(matrix))),
        current_link_derivative=complex_record(first.reshape(len(links), 4)),
        source_current=complex_record(np.dot(source, first)),
        determinant_source_Hessian=complex_record(response),
        current_divergence_norm=norm(divergence), Hessian_gauge_Ward_norm=norm(hessian_g),
        omitted_contact_Ward_defect=omitted_contact, omitted_contact_rejected=adverse_rejected,
        dressed_electron=complex_record(dressed), dressed_source_insertion=complex_record(insertion),
        dressed_gauge_insertion_norm=norm(dressed_gauge),
        undressed_insertion_Ward_defect=undressed_defect, undressed_insertion_rejected=undressed_rejected,
        path_sites=[x, middle, y], path_axes=[0, 1])
    if finite:
        transformed = gauge_transform(links, phases, neighbours)
        dm = dirac_matrix(transformed, shape, neighbours, band, dt)
        sm = np.linalg.solve(dm, np.eye(len(dm)))
        unitary = np.exp(1j*chi)
        covariance = norm(sm - unitary[:, None]*inverse*unitary.conj()[None, :])
        transformed_dressed, _, _ = dressed_observable(sm, transformed, neighbours)
        require(covariance, 'Finite electron inverse gauge covariance', 2e-9)
        require(norm(transformed_dressed - dressed), 'Finite dressed electron gauge invariance', 2e-9)
        step = 2e-4
        plus = links * np.exp(1j*step*source.reshape(links.shape))
        minus = links * np.exp(-1j*step*source.reshape(links.shape))
        dp = dirac_matrix(plus, shape, neighbours, band, dt)
        dn = dirac_matrix(minus, shape, neighbours, band, dt)
        sp = np.linalg.solve(dp, np.eye(len(dp)))
        sn = np.linalg.solve(dn, np.eye(len(dn)))
        fp = dressed_observable(sp, plus, neighbours)[0]
        fn = dressed_observable(sn, minus, neighbours)[0]
        first_error = norm((dp-dn)/(2*step)-vb)
        second = variation(blocks, source**2, len(matrix), order=2)
        contact_error = norm((dp-2*matrix+dn)/step**2-second)
        insertion_error = norm((fp-fn)/(2*step)-insertion)
        lp = positive_logdet(dp)[0]; ln = positive_logdet(dn)[0]
        hessian_error = abs((lp-2*logdet+ln)/step**2-response)
        for name, error, tolerance in (('Exact link variation', first_error, 2e-6),
                                      ('Exact contact variation', contact_error, 5e-6),
                                      ('Dressed current insertion', insertion_error, 2e-6),
                                      ('Determinant source Hessian', hessian_error, 5e-5)):
            require(float(error), name, tolerance)
        result['independent_finite_checks'] = dict(step=step, inverse_covariance=covariance,
            dressed_invariance=norm(transformed_dressed-dressed), link_variation_error=first_error,
            contact_variation_error=contact_error, dressed_insertion_error=insertion_error,
            determinant_Hessian_error=float(hessian_error))
    return result


def finite_step_current(spatial_shape, band, dt, seed):
    """Quantify derivative of H_dt versus actual source h, including its contact."""
    rng = np.random.default_rng(seed)
    sites = int(np.prod(spatial_shape))
    angles = rng.uniform(-.35, .35, (sites, 3))
    values, basis = np.linalg.eigh(MIXING)
    order = np.r_[np.flatnonzero(values > 0), np.flatnonzero(values < 0)]
    change = np.kron(np.eye(sites), basis[:, order])
    grouped = np.r_[[4*x+i for x in range(sites) for i in (0, 1)],
                    [4*x+i for x in range(sites) for i in (2, 3)]]
    change = change[:, grouped]
    rank = 2*sites
    def h(angle):
        shifted = angles.copy(); shifted[:, 0] += angle
        return change.conj().T @ spatial_hamiltonian(links_from_angles(shifted), spatial_shape, band) @ change
    def transfer_h(angle, step):
        hh = h(angle)
        w, f = hh[:rank, :rank], hh[:rank, rank:]
        b = np.eye(rank)+step*w; inv = np.linalg.solve(b, np.eye(rank)); c = step*f
        t = np.block([[inv, -inv@c], [-c.conj().T@inv, b+c.conj().T@inv@c]])
        eigenvalues, vectors = np.linalg.eigh(t)
        if eigenvalues[0] <= 0:
            raise ValueError('Positive exact static Wilson transfer required')
        return -(vectors*(np.log(eigenvalues)/step))@vectors.conj().T
    delta = 5e-4
    h0, hp, hn = h(0.), h(delta), h(-delta)
    j = (hp-hn)/(2*delta); c = (hp-2*h0+hn)/delta**2
    rows = []
    for step in (dt, dt/2, dt/4, dt/8):
        t0, tp, tn = transfer_h(0., step), transfer_h(delta, step), transfer_h(-delta, step)
        tj = (tp-tn)/(2*delta); tc = (tp-2*t0+tn)/delta**2
        rows.append(dict(dt=step, H_dt_source_h_error=norm(t0-h0),
                         current_relative_error=norm(tj-j)/norm(j),
                         contact_relative_error=norm(tc-c)/norm(c)))
    for key in ('H_dt_source_h_error', 'current_relative_error', 'contact_relative_error'):
        if not all(a[key] > b[key] for a, b in zip(rows, rows[1:])):
            raise ValueError(f'No finite-step source-current convergence: {key}')
    return dict(seed=seed, source='uniform spatial axis-1 link angle variation at nonzero compact background',
                finite_difference_step=delta, refinements=rows,
                physical_conversion='d/dA=e*a*d/dtheta; d2/dA2=(e*a)^2*d2/dtheta2',
                interpretation='Measured finite-step discrepancies, not exact finite-dt microscopic current equality')


def coupled_chain(shape, band, dt, ue1, k1, seed, burn, samples, updates, proposal):
    rng = np.random.default_rng(seed)
    coordinates, neighbours = geometry(shape)
    angles = np.zeros((len(coordinates), 4))
    links = links_from_angles(angles)
    phases = rng.uniform(-1., 1., len(links))
    # Non-pure-gauge source; its curl is independently nonzero.
    source_array = np.zeros_like(angles)
    source_array[:, 1] = .7 + .3*np.cos(2*np.pi*coordinates[:, 2]/shape[2])
    source = source_array.ravel()
    curl_norm = norm(source_array[:, 1] - source_array[neighbours[:, 2], 1])
    if curl_norm <= 1e-5:
        raise ValueError('Nontrivial transverse source required')
    weight, initial_terms = joint_log(links, shape, neighbours, band, dt, ue1, k1)
    initial_consumer = fermion_consumer(links, shape, neighbours, band, dt, phases, source, finite=True)
    rows = []; attempted = accepted = 0
    for sweep in range(burn+samples):
        for _ in range(updates):
            site, axis = int(rng.integers(len(links))), int(rng.integers(4))
            old = float(angles[site, axis])
            angles[site, axis] = float(wrap(old+rng.uniform(-proposal, proposal)))
            links[site, axis] = np.exp(1j*angles[site, axis])
            candidate, _ = joint_log(links, shape, neighbours, band, dt, ue1, k1)
            attempted += 1
            if np.log(rng.random()) < min(0., candidate-weight):
                weight = candidate; accepted += 1
            else:
                angles[site, axis] = old; links[site, axis] = np.exp(1j*old)
        if sweep >= burn:
            photon, photon_terms = photon_log(links, neighbours, ue1, k1, dt)
            consumer = fermion_consumer(links, shape, neighbours, band, dt, phases, source,
                                        finite=sweep == burn+samples-1)
            require(abs(weight-photon-consumer['single_flavour_logdet']), 'Coupled retained measure', 2e-9)
            rows.append(dict(sample=sweep-burn, joint_log_weight=weight, photon_terms=photon_terms,
                             angles=angles.tolist(), consumer=consumer))
    if accepted == 0 or len({json.dumps(r['angles']) for r in rows}) < 2:
        raise ValueError('No actual changing photon configurations reached interacting consumer')
    dressed = np.array([np.array(r['consumer']['dressed_electron']['real']) +
                        1j*np.array(r['consumer']['dressed_electron']['imag']) for r in rows])
    return dict(seed=seed, burn_sweeps=burn, samples=samples, updates_per_sweep=updates,
                proposal=proposal, attempted=attempted, accepted=accepted,
                source_link_coefficients=source_array.tolist(), gauge_probe_phases=phases.tolist(),
                source_curl_norm=curl_norm, initial_terms=initial_terms, initial_consumer=initial_consumer,
                finite_raw_dressed_mean=complex_record(dressed.mean(axis=0)),
                dressed_configuration_spread=float(np.sqrt(np.mean(np.abs(dressed-dressed.mean(axis=0))**2))),
                rows=rows, interpretation='Finite correlated raw Metropolis observations under actual photon weight times single det D. No equilibrium, mixing, ensemble response convergence, phase or scattering claim.')


def run(input_receipt, spacing=.2, dt=.2, shape=(2, 3, 3, 3), seed=22273,
        burn=3, samples=8, updates=16, proposal=.08) -> dict[str, Any]:
    if len(shape) != 4 or shape[0] < 2 or min(shape[1:]) < 3:
        raise ValueError('Require temporal size >=2 and spatial sizes >=3')
    if min(dt, proposal) <= 0 or not np.isfinite(dt+proposal) or min(samples, updates) < 2 or burn < 0:
        raise ValueError('Positive finite regulator and nontrivial bounded sampling required')
    bare, fields, band, unitary, ue1, k1, binding = source_point(input_receipt, spacing)
    names = ('compact_dirac_current.py', 'compact_dirac_current.md', 'compact_dirac_measure.py',
             'bulk_frame_compton.py', 'rotation_band.py', 'sea_selection.py')
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    result = dict(governing_plan='issue-222-continuing-backward-20260930', plan_revision=PLAN_REVISION,
                  template_revision=TEMPLATE_REVISION, work_item='campaign', source_sha256=hashes,
                  source_point=binding, shape=list(shape), dt=dt,
                  actual_source_current=current_controls(bare, fields, band, unitary),
                  neutral_transfer=neutral_transfer_controls(shape, band, dt, seed+1),
                  finite_step_source_current=finite_step_current(shape[1:], band, dt, seed+2),
                  coupled_current_electron=coupled_chain(shape, band, dt, ue1, k1, seed,
                                                       burn, samples, updates, proposal))
    if hashes != {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}:
        raise RuntimeError('Current producer/source changed during run')
    result.update(all_checks_pass=True, campaign_complete=False,
                  Gaussian_only_current_reliance_removed_at_finite_regulator=True,
                  flavour_count=1, absolute_determinant_used=False,
                  primitive_full_field_join=False, physical_B4_bar4_propagator=False,
                  neutral_hidden_marker_is_not_charged_electron=True,
                  limits=['Source-selected uniform classical fields remain frozen; compact U1 photons are sampled, not frozen.',
                          'Finite volume, finite dt Wilson fermion and photon product regulator; H_dt is not primitive microH.',
                          'Path-dressed conditional light electron is not an isolated primitive B4/bar4 or hidden neutral Pauli marker.',
                          'Declared Gaussian matching tunes bare UE1/K1; nonlinear compact ensemble photon speed and coupling are not re-established.',
                          'No mixing, thermodynamic phase, radiative scattering, full primitive field/colour/heavy Gauss joining or nonzero-coupling UV continuum.'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'bulk_cutoff_matching_receipt.json')
    parser.add_argument('--spacing', type=float, default=.2)
    parser.add_argument('--dt', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=22273)
    parser.add_argument('--burn', type=int, default=3)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--updates', type=int, default=16)
    parser.add_argument('--proposal', type=float, default=.08)
    parser.add_argument('--output', type=Path, default=ROOT/'compact_dirac_current_receipt.json')
    args = parser.parse_args()
    result = run(args.input, args.spacing, args.dt, seed=args.seed, burn=args.burn,
                 samples=args.samples, updates=args.updates, proposal=args.proposal)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(all_checks_pass=result['all_checks_pass'], campaign_complete=False,
                         output=str(args.output), accepted=result['coupled_current_electron']['accepted'],
                         attempted=result['coupled_current_electron']['attempted'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
