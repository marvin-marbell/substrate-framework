#!/usr/bin/env python3
"""Whole-circle compact Riesz preparation with actual nonstationary energy and adverses."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
import numpy as np
import scipy
from scipy.integrate import solve_bvp, solve_ivp
from scipy.special import mathieu_a, mathieu_cem
from bulk_frame_compton import reconstruct_frame
from original_charged_principal import original_frame
from original_massive_photon_locality import source_hashes as locality_source_hashes
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE


def accept(error, label, tolerance=2e-7):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')
    return float(error)


def compact_case(ue, K, delta=.37, lambda_full=.08, packet_amplitude=.83):
    # Four native links on one contractible plaquette, complete divE=0 Gauss space:
    # all four electric numbers are the same integer. The exact operator is
    # 4ue(-d_phi^2)+K(1-cos phi) on the ENTIRE compact circle, not a finite Fock/rotor basis.
    kinetic = 4*ue
    q = -2*K/kinetic
    energy = K+kinetic*float(mathieu_a(0, q))/4
    def ground(phi):
        # SciPy argument is degrees; returned Mathieu differential-equation
        # derivative is calibrated below against the physical-radian finite difference.
        value, dx = mathieu_cem(0, q, np.asarray(phi)*90/np.pi)
        return np.asarray(value), np.asarray(dx)/2
    probes = np.array([.17, .37, 1.13, 2.])
    h = 1e-5
    finite = (ground(probes+h)[0]-ground(probes-h)[0])/(2*h)
    derivative_error = accept(float(np.max(abs(finite-ground(probes)[1]))), 'physical-radian Mathieu derivative', 2e-8)
    assert np.linalg.norm(finite) > .05 or K == 0
    # Inward log-ground integration preserves relative accuracy at the barrier.
    # Direct Mathieu evaluation there loses ~1e-5 relative accuracy at this SAME g.
    # No density flooring, changed coupling, or finite rotor basis.
    def log_ground_equation(phi, value):
        return [value[1], (K*(1-np.cos(phi))-energy)/kinetic-value[1]**2]
    log_ground = solve_ivp(log_ground_equation, (np.pi, 0.), [0., 0.],
                           rtol=2e-13, atol=2e-14, dense_output=True)
    assert log_ground.success, log_ground.message
    origin_log, origin_slope = log_ground.sol(0.)
    origin_value = float(ground(0.)[0])
    neumann_error = accept(abs(float(origin_slope)), 'inward ground Neumann origin', 2e-10)
    def stable_ground(phi):
        phi = np.asarray(phi)
        folded = np.where(phi <= np.pi, phi, 2*np.pi-phi)
        logarithm, slope = log_ground.sol(folded)
        value = origin_value*np.exp(logarithm-origin_log)
        prime = value*slope*np.where(phi <= np.pi, 1., -1.)
        return value, prime
    well_error = accept(float(np.max(abs(stable_ground(probes[:3])[0]/ground(probes[:3])[0]-1))),
                        'independent Mathieu well-region agreement', 2e-10)
    angle = np.arange(16384)*2*np.pi/16384
    raw, raw_prime = stable_ground(angle)
    assert np.all(raw > 0)
    norm = float(2*np.pi*np.mean(raw*raw))
    psi, psi_prime = raw/np.sqrt(norm), raw_prime/np.sqrt(norm)
    w = psi*psi
    ground_energy = float(2*np.pi*np.mean(kinetic*psi_prime**2+K*(1-np.cos(angle))*w))
    ground_error = accept(abs(ground_energy-energy), 'exact whole-circle ground quadratic form', 2e-9)
    lambda_cycle = lambda_full/4
    def equation(phi, value):
        # f=psi*S removes division by the exponentially small barrier density.
        # The same weighted Riesz problem becomes a smooth Schroedinger resolvent.
        y, yp = stable_ground(phi)
        potential = (K*(1-np.cos(phi))-energy)/kinetic
        chi_prime = packet_amplitude*np.cos(phi)
        chi_second = -packet_amplitude*np.sin(phi)
        # f_delta=psi*u_delta and f_zero=psi*u0; BOTH exact Riesz problems.
        return np.vstack((value[1], (potential+lambda_cycle)*value[0]+2*delta*yp,
                          value[3], (potential+lambda_cycle)*value[2]-y*chi_second-2*yp*chi_prime))
    def boundary(left, right):
        return np.array([left[0], right[0], left[2], right[2]])
    mesh = np.linspace(0, np.pi, 257)
    solution = solve_bvp(equation, boundary, mesh, np.zeros((4, len(mesh))), tol=1e-10, max_nodes=10000)
    assert solution.success, solution.message
    folded = np.where(angle <= np.pi, angle, 2*np.pi-angle)
    values = solution.sol(folded)
    folded_psi, folded_prime = stable_ground(folded)
    half_phase = values[0]/folded_psi
    phase = np.where(angle <= np.pi, half_phase, -half_phase)
    # Differentiate the represented phase, not the independent collocation f' state.
    # Their tiny absolute residual is amplified by division by the barrier ground.
    represented_prime = solution.sol(folded, 1)[0]
    gradient = (represented_prime-half_phase*folded_prime)/folded_psi
    collocation_gradient = (values[1]-half_phase*folded_prime)/folded_psi
    expectation = lambda value: float(2*np.pi*np.mean(w*value))
    response = expectation(gradient)
    dirichlet = expectation(gradient*gradient)
    regularizer = lambda_cycle*expectation(phase*phase)
    variational_error = accept(abs(delta*response-dirichlet-regularizer), 'normalized Riesz energy-response identity', 2e-8)
    shifted_kinetic = np.abs(psi_prime+1j*psi*gradient)**2
    prepared_energy = float(2*np.pi*np.mean(kinetic*shifted_kinetic+K*(1-np.cos(angle))*w))
    shift_error = accept(abs(prepared_energy-energy-kinetic*dirichlet), 'actual compact unitary full energy identity', 2e-9)
    accept(max(0., dirichlet-delta*delta), 'complete-gradient energy budget')
    periodic_gradient_error = accept(abs(float(2*np.pi*np.mean(gradient))), 'periodic phase integral', 2e-7)
    f_full = np.where(angle <= np.pi, values[0], -values[0])
    parts_error = accept(abs(4*delta*response+8*delta*float(2*np.pi*np.mean(raw_prime*f_full)/norm)),
                         'physical full-cycle response by independent periodic integration by parts', 2e-8)

    # A real nonstationary packet on this SAME complete rotor: psi*exp(i chi).
    # Its kinetic current is chi_prime, not zero as in the ground state.
    half_zero = values[2]/folded_psi
    zero_phase = np.where(angle <= np.pi, half_zero, -half_zero)
    zero_gradient = (solution.sol(folded, 1)[2]-half_zero*folded_prime)/folded_psi
    chi_prime = packet_amplitude*np.cos(angle)
    chi_second = -packet_amplitude*np.sin(angle)
    d_zero, p_zero = expectation(zero_gradient**2), expectation(zero_phase**2)
    j_zero = expectation(chi_prime*zero_gradient)
    cross_phase = expectation(zero_phase*phase)
    zero_riesz_error = accept(abs(d_zero+lambda_cycle*p_zero+j_zero),
                              'nonstationary kinetic-current cancelling Riesz identity', 2e-8)
    def phase_energy(prime):
        return float(2*np.pi*np.mean(kinetic*np.abs(psi_prime+1j*psi*prime)**2+K*(1-np.cos(angle))*w))
    packet_energy = phase_energy(chi_prime)
    baseline_energy = phase_energy(chi_prime+zero_gradient)
    packet_prepared_energy = phase_energy(chi_prime+zero_gradient+gradient)
    packet_shift = packet_prepared_energy-packet_energy
    completed_shift = kinetic*(dirichlet-d_zero-2*lambda_cycle*p_zero-2*lambda_cycle*cross_phase)
    packet_identity_error = accept(abs(packet_shift-completed_shift),
                                   'actual nonstationary compact-unitary square-completion identity', 2e-8)
    baseline_identity_error = accept(abs(baseline_energy-packet_energy+kinetic*(d_zero+2*lambda_cycle*p_zero)),
                                     'actual current-corrected baseline energy identity', 2e-8)
    accept(max(0.,packet_shift-kinetic*delta**2), 'actual nonstationary whole-energy input budget', 2e-9)
    accept(max(0.,baseline_energy-packet_energy), 'actual phase-corrected baseline does not raise energy', 2e-9)
    zero_periodic_error = accept(abs(float(2*np.pi*np.mean(zero_gradient))),
                                 'nonstationary correction is an actual periodic phase', 2e-7)
    # H on psi*exp(i chi), subtracting its own mean, independently proves nonstationarity.
    variance = float(2*np.pi*np.mean(kinetic**2*((chi_prime**2*psi)**2+(chi_second*psi+2*chi_prime*psi_prime)**2)))
    variance -= (packet_energy-energy)**2
    assert variance > 1e-6, 'selected packet unexpectedly stationary'
    naive_shift = phase_energy(chi_prime+gradient)-packet_energy
    wrong_stationarity_rejection = None
    if K != 0:
        try:
            accept(max(0.,naive_shift-kinetic*delta**2), 'wrong stationary formula on actual packet', 2e-9)
        except AssertionError as error:
            wrong_stationarity_rejection = str(error)
        else:
            raise AssertionError('actual nonstationary adverse did not reject the false input budget')
    packet_result = dict(amplitude=packet_amplitude, actual_energy=packet_energy,
                         energy_variance=variance, baseline_energy=baseline_energy,
                         prepared_energy=packet_prepared_energy, original_to_prepared_energy=packet_shift,
                         baseline_to_prepared_energy=packet_prepared_energy-baseline_energy,
                         input_energy_budget=kinetic*delta**2,
                         baseline_electric_shift_per_edge=expectation(zero_gradient),
                         soft_electric_shift_from_corrected_baseline=response,
                         total_electric_shift_per_edge=expectation(zero_gradient+gradient),
                         naive_phase_energy_increase=naive_shift,
                         false_stationarity_rejection=wrong_stationarity_rejection,
                         correction_riesz_error=zero_riesz_error,
                         square_completion_energy_error=packet_identity_error,
                         corrected_baseline_energy_error=baseline_identity_error,
                         correction_periodic_integral_error=zero_periodic_error)
    if K == 0:
        accept(abs(response)+dirichlet, 'uniform-ground zero response', 1e-12)
    else:
        assert response/delta > .95
    return dict(ue=ue, K=K, kinetic_cycle=kinetic, exact_ground_energy=energy,
                Mathieu_q=q, norm=norm, minimum_ground_density=float(w.min()),
                lambda_full=lambda_full, lambda_cycle=lambda_cycle, delta_edge=delta,
                target_edge_norm2=4*delta*delta, response_per_edge=response,
                response_fraction=response/delta, full_gradient_norm2=4*dirichlet,
                exact_energy_increase=prepared_energy-energy,
                upper_energy_budget=4*ue*delta*delta,
                radian_derivative_error=derivative_error, ground_form_error=ground_error,
                inward_neumann_error=neumann_error, independent_well_error=well_error,
                variational_error=variational_error, energy_identity_error=shift_error,
                periodic_phase_integral_error=periodic_gradient_error,
                collocation_periodic_integral_error=abs(float(2*np.pi*np.mean(collocation_gradient))),
                response_integration_by_parts_error=parts_error, nonstationary_packet=packet_result,
                bvp_nodes=len(solution.x), bvp_residual_max=float(solution.rms_residuals.max()),
                numerical_scope='continuous Mathieu full-rotor operator with adaptive ODE/quadrature; no finite Hamiltonian/Fock/rotor cutoff')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'original_compact_soft_response_receipt.json')
    args = parser.parse_args()
    bare = RotationBand(**{**BARE, 'a': 1.}, e=1.)
    selection, frame = original_frame(bare, 5, np.zeros(3), True)
    band, _, _ = reconstruct_frame(bare, frame)
    g = .3
    ue, K = band.v*g*g/2, band.v/(g*g)
    rows = [compact_case(ue, K), compact_case(ue, 0.)]
    illegal = abs(np.exp(2j*np.pi*.37)-1)
    assert illegal > 1
    result: dict[str, Any] = dict(plan='issue-222-continuing-backward-20260930',
        assignment_revision='a07663f778849e7081540356af537bb3ad2c5c435dd16f70466705ac800534cc',
        stationary_proof_sha256='63da25492d59b29eaede17dcd7eb15c83aa6b52bd1c4de789678e0c497a4b345',
        current_document_sha256=hashlib.sha256((ROOT/'original_massive_photon_locality.md').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_Gram=selection['G'].tolist(), photon=dict(ue=ue,K=K,g=g,a=1.),
        runtime=dict(python=platform.python_version(),executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__),
        source_hashes={**locality_source_hashes(), Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        illegal_fractional_rotor_plane_endpoint_mismatch=illegal, rows=rows, all_checks=True,
        scope='exact compact one-plaquette stationary and nonstationary comparator with uniform-weight and false-stationarity adverses; NOT original full-field/CAR ground, shell-uniform soft response, moving carrier or photon phase')
    target = args.output
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(target),all_checks=True,response_fractions=[r['response_fraction'] for r in rows], original_full_ground_verified=False)))


if __name__ == '__main__':
    main()
