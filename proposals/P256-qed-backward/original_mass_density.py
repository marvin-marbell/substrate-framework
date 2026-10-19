#!/usr/bin/env python3
"""Static native beta-mass density check; no photon integration or phase claim."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
import numpy as np
import scipy
from original_charged_principal import original_frame
from original_compact_charge_bound import chiral_basis
from original_massive_photon_locality import locality_constants, source_hashes as locality_source_hashes, tree_onebody
from original_uniform_charge_gap import exterior_hamiltonian, temporal_green, time_evolution
from rotation_band import RotationBand
from bulk_quantum_selection import BulkQuantumLaw
from unrestricted_field_minimum import BARE


def accept(error, label, tolerance=2e-8):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')


def rejected(callback):
    try:
        callback()
    except AssertionError as error:
        return str(error)
    raise AssertionError('adverse reached and passed the actual accept assertion')


def eigendata(h, beta4):
    values, vectors = np.linalg.eigh(h)
    volume = len(h)//4
    accept(float(np.max(abs(values+values[::-1]))), 'actual mass-homotopy chiral pairs', 2e-6)
    local = vectors.reshape(volume, 4, len(h))
    weights = np.einsum('xai,ab,xbi->xi', local.conj(), beta4, local).real
    return values, weights


def source_density(data, period, theta):
    values, weights = data
    small = np.exp(-period*abs(values))
    occupation = np.empty(len(values), dtype=np.complex128)
    positive = values > 0
    occupation[positive] = 1/(1+small[positive]*np.exp(1j*theta))
    occupation[~positive] = small[~positive]*np.exp(-1j*theta)/(1+small[~positive]*np.exp(-1j*theta))
    density = weights@occupation
    accept(float(np.max(abs(density.imag))), 'real local physical beta density')
    return density.real


def log_difference(first, second, period, theta_first, theta_second):
    ep, er = first[first > 0], second[second > 0]
    xp, xr = np.exp(-period*ep), np.exp(-period*er)
    # Subtract paired energies BEFORE summing their common huge mass endpoint.
    return float(period*np.sum(ep-er)+np.sum(
        np.log1p(2*np.cos(theta_first)*xp+xp*xp)
        -np.log1p(2*np.cos(theta_second)*xr+xr*xr)))


def full_car_check(frame, band, beta4, period):
    h = np.asarray(tree_onebody(frame, band, .37), dtype=np.complex128)
    beta = np.kron(np.eye(2), beta4)
    rows = []
    for mass in (0., 1/period, 8/period):
        matrix = h+mass*beta
        data = eigendata(matrix, beta4)
        ground = float(np.sum(data[0][data[0] < 0]))
        trace, derivative, dimensions = 0j, 0j, []
        for number in range(9):
            sector, basis = exterior_hamiltonian(matrix, number)
            observable, _ = exterior_hamiltonian(beta, number)
            energies, vectors = np.linalg.eigh(sector)
            diagonal = np.einsum('ai,ab,bi->i', vectors.conj(), observable, vectors)
            weights = np.exp(-period*(energies-ground))*np.exp(.47j*(number-4))
            trace += complex(weights.sum())
            derivative += complex(weights@diagonal)
            dimensions.append(len(basis))
        assert sum(dimensions) == 256
        accept(abs(trace.imag)/max(1., abs(trace)), 'entire-CAR background-two positive real trace')
        assert trace.real > 0
        actual = float((-derivative/trace).real)
        predicted = float(source_density(data, period, .47).sum())
        accept(abs(actual-predicted), 'entire256CAR mass insertion versus native Green trace')
        rows.append(dict(mass=mass, dimensions=dimensions, CAR_density=actual,
                         Green_density=predicted, error=abs(actual-predicted)))
    return dict(entire_CAR_states=256, rows=rows,
                scope='entire two-site exterior algebra; conditional global-holonomy trace, not local photon/Haar Gauss integral')


def native_case(spacing, centre, seed):
    band = RotationBand(**{**BARE, 'a': spacing}, e=1.)
    selected, frame = original_frame(band, 5, np.pi/2*np.asarray(centre), True)
    gram = np.asarray(selected['G'], dtype=float)
    beta4 = chiral_basis(frame, gram, 1)[2]
    law = BulkQuantumLaw(5, band=band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.pi/2*np.asarray(centre))
    phases = np.random.default_rng(seed).uniform(-np.pi, np.pi, (law.volume, 3))
    reference = np.asarray(law.onebody(fields, links, np.zeros_like(phases)), dtype=np.complex128)
    native = np.asarray(law.onebody(fields, links, phases), dtype=np.complex128)
    beta = np.kron(np.eye(law.volume), beta4)
    period = .4*spacing
    constants = locality_constants(band, gram, frame, delta=period/8)
    mu, ch = constants['mu'], constants['Ch']
    scale = max(ch, 1/period)
    cap = 1000*scale
    initial_a, initial_r = eigendata(native, beta4), eigendata(reference, beta4)
    exact = np.array([log_difference(initial_a[0], initial_r[0], period, 0., 0.),
                      log_difference(initial_a[0], initial_a[0], period, .47, 0.),
                      log_difference(initial_a[0], initial_r[0], period, .47, 0.)])
    # Independently identify the same source with actual ordered evolution.
    controls = []
    for mass in (0., scale):
        matrix = native if mass == 0 else native+mass*beta
        data = initial_a if mass == 0 else eigendata(matrix, beta4)
        accept(max(0., mu+mass-float(np.min(abs(data[0])))), 'true beta mass Schur gap', 2e-7)
        evolution, prefixes, _ = time_evolution([matrix], np.full((1, law.volume), .47/period), np.array([period]))
        source = temporal_green(evolution, prefixes)[0]
        actual = np.array([np.trace(beta4@source[4*x:4*x+4, 4*x:4*x+4]) for x in range(law.volume)], dtype=np.complex128)
        predicted = source_density(data, period, .47)
        accept(float(np.max(abs(actual-predicted))), 'actual monodromy one-sided source beta insertion')
        epsilon = min(scale*1e-5, mu/10)
        plus = np.linalg.eigvalsh(matrix+epsilon*beta)
        minus = np.linalg.eigvalsh(matrix-epsilon*beta)
        finite = log_difference(plus, minus, period, .47, .47)/(2*epsilon)
        derivative = period*float(predicted.sum())
        accept(abs(finite-derivative)/max(1., abs(derivative)), 'actual physical logdet beta-mass derivative')
        controls.append(dict(mass=mass, source_error=float(np.max(abs(actual-predicted))),
                             logdet_derivative=finite, Green_derivative=derivative))
    wrong = np.linalg.eigvalsh(native+scale*np.eye(len(native)))
    wrong_mass = rejected(lambda: accept(max(0., mu+scale-float(np.min(abs(wrong)))),
                                         'true beta mass Schur gap', 2e-7))
    # Finite endpoint identity and infinite-endpoint homotopy, no raw exp(MT).
    nodes, weights = np.polynomial.legendre.leggauss(80)
    limit = cap/(cap+scale)
    integrals = np.zeros((3, law.volume))
    for node, weight in zip(nodes, weights):
        u = (node+1)*limit/2
        mass = scale*u/(1-u)
        da = eigendata(native+mass*beta, beta4)
        dr = eigendata(reference+mass*beta, beta4)
        ra, rr = source_density(da, period, 0.), source_density(dr, period, 0.)
        rh = source_density(da, period, .47)
        measure = weight*limit/2*scale/(1-u)**2
        integrals += measure*np.array([ra-rr, rh-ra, rh-rr])
    final_a, final_r = np.linalg.eigvalsh(native+cap*beta), np.linalg.eigvalsh(reference+cap*beta)
    endpoint = np.array([log_difference(final_a, final_r, period, 0., 0.),
                         log_difference(final_a, final_a, period, .47, 0.),
                         log_difference(final_a, final_r, period, .47, 0.)])
    recovered = -period*integrals.sum(axis=1)+endpoint
    accept(float(np.max(abs(recovered-exact))), 'finite-endpoint exact conditional mass-density integral', 3e-7)
    wrong_density = rejected(lambda: accept(float(np.max(abs(endpoint-exact))),
                                            'finite-endpoint exact conditional mass-density integral', 3e-7))
    # Static SAME-holonomy spatial difference has no1/2/3-hop phase difference on N>=5.
    # Conservatively bound the first possibly changing four-hop remainder.
    assert cap >= max(8*ch, 2/period)
    spatial_tail = period*law.volume*(8192/3)*ch**4/cap**3
    return dict(spacing=spacing, field_N=5, modes=500, centre=centre,
                native_Gram=gram.tolist(), native_gradient=float(selected['gradient_residual']),
                period=period, mu=mu, Ch=ch, beta_sha256=hashlib.sha256(beta4.tobytes()).hexdigest(),
                compact_phase_sha256=hashlib.sha256(phases.tobytes()).hexdigest(),
                mass_cap=cap, quadrature_nodes=80, exact_differences=exact.tolist(),
                endpoint_differences=endpoint.tolist(), density_recovered_differences=recovered.tolist(),
                integral_errors=abs(recovered-exact).tolist(),
                static_same_holonomy_four_hop_tail_bound=spatial_tail,
                actual_monodromy_and_mass_derivative=controls,
                wrong_identity_mass_rejected=wrong_mass, zero_density_rejected=wrong_density,
                independent_full_CAR=full_car_check(frame, band, beta4, period))


def native_source_hashes():
    return locality_source_hashes() | {
        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'original_mass_density_receipt.json')
    args = parser.parse_args()
    before = native_source_hashes()
    rows = [native_case(a, c, 2221030+i) for i, (a, c) in enumerate(
        ((1., (0, 0, 0)), (1., (1, 0, 2)), (.001, (0, 0, 0)), (.001, (1, 0, 2))))]
    assert before == native_source_hashes(), 'native source changed during density check'
    result = dict(plan='issue-222-continuing-backward-20260930',
                  assignment_revision='ae3462dd2e6af3ce4e30b03a879237c98cec4e531c209079dfb62a43af7bfd89',
                  template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
                  source_hashes=before,
                  runtime=dict(python=platform.python_version(), executable=sys.executable,
                               numpy=np.__version__, scipy=scipy.__version__), rows=rows,
                  all_checks=True,
                  scope='exact static conditional determinant mass homotopy only; no photon/localHaar integrated trace, Brownian density theorem, Maxwell-relative stability or phase')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks=True, cases=len(rows), compact_phase_proved=False)))


if __name__ == '__main__':
    main()
