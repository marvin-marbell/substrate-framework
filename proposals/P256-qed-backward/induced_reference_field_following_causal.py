#!/usr/bin/env python3
"""Full-Href retarded initial current moment on the true field-following source.

Electric gradients include the actual CAR ray tangent. Matter and connected
realGauss Coulomb remain separate. This finite-N2 Gaussian integral is not a
stationary spectral response, photon pole, residue or macro-source gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
if __name__ == '__main__':
    for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_name] = '1'
import numpy as np
from induced_reference_field_following import (build_native, conditional_slater, path_derivative,
    SOURCE_BYTES as FOLLOW_BYTES)
from induced_phi_em_causal_current import make_operators, commutator, SOURCE_BYTES as CAUSAL_BYTES

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**CAUSAL_BYTES, **FOLLOW_BYTES, Path(__file__).name: Path(__file__).read_bytes()}
SOURCE_HASHES = {name: hashlib.sha256(raw).hexdigest() for name, raw in SOURCE_BYTES.items()}


def measure_theta(native: dict[str, Any], theta: np.ndarray, eta: float,
                  direction: np.ndarray) -> dict[str, Any]:
    """Exact finite conditional moment; average over normalized photon density."""
    if native['N'] != 2:
        raise ValueError('This full causal backend explicitly supports N2, not a silent macro truncation')
    V = native['V']; u = native['UE']/native['a']; states = []
    matter = 0.; mean = 0.; covariance = 0.; contact = 0.; density_term = 0j
    comm_density_mean = np.zeros(V, complex); comm_density_cov = 0j
    for profile in native['Pprofiles']:
        state = conditional_slater(native, profile, theta, eta, section=False)
        h, P, lower, upper = state['physical_h'], state['P'], state['occupied'], state['empty']
        J = path_derivative(native, profile, theta, direction)
        load = profile['load']; rank = lower.shape[1]
        current_PH = upper.conj().T@J@lower
        C = path_derivative(native, profile, theta, direction, 2)
        mean += load*float(np.trace(P@J).real)
        covariance += load*float(np.sum(abs(current_PH)**2))
        contact += load*float(np.trace(P@C).real)
        hJ = h@J-J@h
        matter += load*float(np.trace(P@(hJ@J-J@hJ)).real)
        transitions = []; complement = np.eye(4*V)-P
        for x in range(V):
            mask = np.repeat((np.arange(V) == x).astype(float), 4)
            Ax = (mask[:, None]-mask[None, :])*J
            transitions.append(upper.conj().T@Ax@lower)
            comm_density_mean[x] += load*np.trace(P@Ax)
            Bx = Ax@J-J@Ax
            weights = np.repeat(native['Lplus'][x], 4)
            density_term += load*np.sum(np.diag(P@Bx@complement)*weights)
        packed = np.array(transitions).reshape(V, rank*rank)
        # Ax is anti-Hermitian, so its ordered connected covariance is negative.
        comm_density_cov -= load*np.einsum('xa,xy,ya->', packed.conj(), native['Lplus'], packed)
        state.update(profile=profile, J=J, load=load)
        states.append(state)
    density_pair = comm_density_cov+comm_density_mean@native['Lplus']@comm_density_mean
    coulomb_complex = u*(2*density_term.real+2*density_pair)
    electric = 0.; fixed_fibre_electric = 0.; tangent_correction = 0.
    for coordinate in native['Q'].T:
        derivative_mean = 0.; derivative_covariance = 0.; correction = 0.
        for state in states:
            profile, P = state['profile'], state['P']
            lower, upper, J = state['occupied'], state['empty'], state['J']
            actual_first = path_derivative(native, profile, theta, coordinate)
            derivative_current = path_derivative(native, profile, theta, direction, 2, other=coordinate)
            delta = state['values'][lower.shape[1]:, None]-state['values'][None, :lower.shape[1]]
            ray_PH = -eta*(upper.conj().T@actual_first@lower)/delta
            contact_PH = upper.conj().T@derivative_current@lower
            comm = J@derivative_current-derivative_current@J
            comm_PH = upper.conj().T@comm@lower
            load = state['load']
            derivative_mean += load*float(np.trace(P@derivative_current).real)
            derivative_covariance += load*float(np.sum(abs(contact_PH)**2))
            correction += load*float(np.vdot(ray_PH, comm_PH).real)
        scalar = derivative_mean**2+derivative_covariance
        fixed_fibre_electric -= 2*u*scalar
        tangent_correction -= 2*u*correction
        electric -= 2*u*(scalar+correction)
    return dict(current_mean=mean, conditional_current_covariance=covariance,
        raw_current_second_moment=covariance+mean**2, literal_contact=contact,
        retarded_initial_tangent=electric+matter+float(coulomb_complex.real),
        parts=dict(primitive_electric=electric, Peierls_CAR=matter, full_real_Gauss_Coulomb=float(coulomb_complex.real)),
        missing_actual_ray_tangent_shift=tangent_correction, wrong_fixed_fibre_electric=fixed_fibre_electric,
        Coulomb_hermitian_residual=float(abs(coulomb_complex.imag)))


def graph_bound(native: dict[str, Any], eta: float, variance: float,
                direction: np.ndarray) -> dict[str, Any]:
    """Conservative actual source graph, including field-dependent CAR derivatives."""
    V = native['V']; g = native['UE']/native['a']; M = native['M']
    B1 = np.zeros(M); B2 = np.zeros(M); gaps = []
    for profile in native['Pprofiles']:
        mass = profile['model'].m
        positive_onsite = float(np.linalg.eigvalsh(profile['onsite'])[-1])
        gaps.append((1-eta)*positive_onsite+eta*mass*np.sqrt(native['g0']))
        hopnorm = np.array([np.linalg.norm(p['hop'], 'fro') for p in profile['paths']])
        slopes = profile['path_Q']
        B1 += profile['load']*2*V*eta*(2*hopnorm@abs(slopes))
        B2 += profile['load']*2*V*eta*(2*hopnorm@(slopes**2))
    delta = 2*min(gaps)
    first = B1/delta; second = B2/delta+5*(B1/delta)**2
    free_gaussian_K = g*np.sqrt(M*(M+2))/(4*variance)
    actual_K = free_gaussian_K+g*(np.sum(first)/np.sqrt(variance)+np.sum(second))
    H, J = make_operators(dict(native=native, W=np.eye(4)), direction)
    loads = [p['load'] for p in native['Pprofiles']]
    def magnitude(coeff: np.ndarray) -> float:
        return float(sum(abs(coeff[r])*loads[r] for r in range(2)))
    matter = sum(magnitude(coeff) for coeff in H.values())
    Coulomb = g*V*128**2/native['laplacian_gap']
    cap = matter+Coulomb
    moments = np.zeros(5)
    for (_, _, phase), coeff in J.items():
        b = np.asarray(phase, float)
        norm = np.sqrt(float(b@native['PT']@b))
        moments += magnitude(coeff)*norm**np.arange(5)
    J0, J1, J2, J3, J4 = moments
    evolved_K = actual_K+2*cap; evolved_P = np.sqrt(evolved_K/g)
    KK = g*g*(4*J2*evolved_K/g+4*J3*evolved_P+J4)
    gradient = laplacian = 0.
    for (_, _, phase), coeff in H.items():
        b = np.asarray(phase, float); norm = np.sqrt(float(b@native['PT']@b)); amount = magnitude(coeff)
        gradient += norm*amount; laplacian += norm**2*amount
    mixed = g*(2*laplacian*J0+6*gradient*J1+4*cap*J2+4*gradient*J0*evolved_P+8*cap*J1*evolved_P)
    VV = 4*cap*cap*J0
    return dict(retarded_kernel_second_derivative_upper=float(2*J0*(KK+mixed+VV)),
        actual_initial_free_kinetic_norm_upper=float(actual_K), free_Gaussian_only_kinetic_norm=float(free_gaussian_K),
        uniform_preparation_manybody_gap=delta, potential_norm_upper=float(cap),
        formula='|chi_R(t)-t*mu| <= t²/2 * second_derivative_cap; source normalized, no stationarity assumed',
        source_graph='Parallel ground ray of Hprep: ||dOmega||<=B1/delta and ||ddOmega||<=B2/delta+5(B1/delta)^2. Gaussian product derivatives included. Actual Href potential includes fullrealGauss.',
        scope='Conservative analytic formula evaluated in binary64, not interval-certified arithmetic; no finiteclock/residue admission.')


def verify_event_CAR(native: dict[str, Any], theta: np.ndarray, eta: float,
                     direction: np.ndarray) -> float:
    """Independent shared event algebra vs dense original physical CAR action."""
    H, J = make_operators(dict(native=native, W=np.eye(4)), direction)
    operator = commutator(commutator(H, J), J)
    states = [conditional_slater(native, p, theta, eta, section=False) for p in native['Pprofiles']]
    observed = 0j
    for (i, j, phase), coefficients in operator.items():
        factor = np.exp(1j*np.asarray(phase)@theta)
        for r, profile in enumerate(native['Pprofiles']):
            observed += factor*coefficients[r]*profile['load']*states[r]['P'][j, i]
    dense = measure_theta(native, theta, eta, direction)['parts']['Peierls_CAR']
    return float(abs(observed-dense))


def verify_event_Coulomb(native: dict[str, Any], theta: np.ndarray, eta: float,
                         direction: np.ndarray) -> float:
    """Independent full64 event/Wick Coulomb algebra, no mean-density shortcut."""
    _, J = make_operators(dict(native=native, W=np.eye(4)), direction)
    states = [conditional_slater(native, p, theta, eta, section=False) for p in native['Pprofiles']]
    loads = [p['load'] for p in native['Pprofiles']]
    def one(i: int, j: int, phase: tuple[int, ...], coefficients: np.ndarray) -> complex:
        return complex(np.exp(1j*np.asarray(phase)@theta)*sum(
            loads[r]*coefficients[r]*state['P'][j, i] for r, state in enumerate(states)))
    total = 0j
    for (i, j, phase), a in J.items():
        charge = np.zeros(native['V']); charge[i//4] += 1; charge[j//4] -= 1
        weights = native['Lplus']@charge; diagonal = np.repeat(weights, 4)
        ci = float(charge@weights)
        density = [diagonal[:, None]*state['P']-state['P']@(diagonal[:, None]*state['P']) for state in states]
        for (k, l, q), b in J.items():
            combined = tuple(x+y for x, y in zip(phase, q))
            factor = np.exp(1j*np.asarray(combined)@theta)
            ma = sum(loads[r]*a[r]*s['P'][j, i] for r, s in enumerate(states))
            mb = sum(loads[r]*b[r]*s['P'][l, k] for r, s in enumerate(states))
            pair = ma*mb+sum(loads[r]*a[r]*b[r]*s['P'][l, i]*((1 if j == k else 0)-s['P'][j, k])
                for r, s in enumerate(states))
            charge2 = np.zeros(native['V']); charge2[k//4] += 1; charge2[l//4] -= 1
            total += 2*float(charge2@weights)*factor*pair
            for sign, left, right, allowed in ((1, i, l, j == k), (-1, k, j, l == i)):
                if allowed:
                    coefficients = sign*a*b
                    density_word = factor*sum(loads[r]*coefficients[r]*density[r][right, left] for r in range(2))
                    total += 2*density_word+ci*one(left, right, combined, coefficients)
    observed = native['UE']/native['a']*total
    dense = measure_theta(native, theta, eta, direction)['parts']['full_real_Gauss_Coulomb']
    return float(abs(observed-dense))




def run(mean: Path, *, eta: float = .4, variance: float = 2., samples: int = 4,
        seed: int = 20261008) -> dict[str, Any]:
    if samples < 2 or variance <= 0:
        raise ValueError('At least2 samples and positive density variance required')
    native = build_native(mean, 2)
    direction = np.zeros(3*native['V'])
    direction[1::3] = np.cos(np.pi*native['sites'][:, 0]); direction /= np.linalg.norm(direction)
    if np.linalg.norm(native['D']@direction) > 1e-12:
        raise ArithmeticError('Actual photon-current direction is not coexact')
    rng = np.random.default_rng(seed); rows = []
    for _ in range(samples):
        theta = np.sqrt(variance)*native['Q']@rng.standard_normal(native['M'])
        rows.append(dict(theta=theta.tolist(), observed=measure_theta(native, theta, eta, direction)))
    tangents = np.array([r['observed']['retarded_initial_tangent'] for r in rows])
    means = np.array([r['observed']['current_mean'] for r in rows])
    raw = np.array([r['observed']['raw_current_second_moment'] for r in rows])
    bound = graph_bound(native, eta, variance, direction)
    time_probe = .01*abs(float(np.mean(tangents)))/bound['retarded_kernel_second_derivative_upper']
    mean_variance = float(np.var(means, ddof=1))
    connected_plugin = float(np.mean(raw)-np.mean(means)**2)
    return dict(N=2, real_EM_coordinates=native['M'], all_charged_flavours=64, eta=eta, variance=variance,
        samples=samples, seed=seed, direction=direction.tolist(), rows=rows,
        retarded_initial_tangent_estimate=float(np.mean(tangents)), sampling_standard_error=float(np.std(tangents, ddof=1)/np.sqrt(samples)),
        current_mean_estimate=float(np.mean(means)), raw_current_second_moment_estimate=float(np.mean(raw)),
        within_Theta_current_covariance_estimate=float(np.mean([r['observed']['conditional_current_covariance'] for r in rows])),
        between_Theta_conditional_mean_variance_unbiased=mean_variance,
        connected_equal_time_plugin_estimate=connected_plugin,
        connected_equal_time_unbiased_estimate=connected_plugin+mean_variance/samples,
        source_mean_status='No exact C-zero assumption. Plug-in subtracts sampled mean squared and is biased; unbiased estimator adds sample conditional-mean variance/n. Raw moments retain between-Theta/all64 current correlations.',
        literal_contact_estimate=float(np.mean([r['observed']['literal_contact'] for r in rows])),
        source_graph_bound=bound, short_time_probe=time_probe,
        short_time_linear_kernel_estimate=time_probe*float(np.mean(tangents)),
        conservative_short_time_error_upper=time_probe**2/2*bound['retarded_kernel_second_derivative_upper'],
        independent_event_Coulomb_error=verify_event_Coulomb(native, np.asarray(rows[0]['theta']), eta, direction),
        independent_event_CAR_error=verify_event_CAR(native, np.asarray(rows[0]['theta']), eta, direction),
        source_hashes=SOURCE_HASHES, source_snapshots={name:raw.decode() for name,raw in SOURCE_BYTES.items()},
        state='Actual normalized realcoexact+harmonic Gaussian × fieldfollowing all64 projected-onsite CAR Slaters, realGaussCoulomb retained; eta preparation ONLY, physical Href unchanged.',
        validity='Actual fullHref initial retarded moment and conditional matrix/Wick Gaussian integrand; MC error not rigorousintegralcertificate. Nonstationary: no Fourierpole/residue/cone/macro source claim. Includesactual CAR ray derivative, never fixedtheta fibre or freeband response.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mean', type=Path, required=True); parser.add_argument('--eta', type=float, default=.4)
    parser.add_argument('--variance', type=float, default=2.); parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261008); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    result = run(args.mean, eta=args.eta, variance=args.variance, samples=args.samples, seed=args.seed)
    for name, raw in SOURCE_BYTES.items():
        if (ROOT/name).read_bytes() != raw:
            raise RuntimeError('Source changed during actual quantumcurrent calculation: '+name)
    result['elapsed_seconds'] = time.monotonic()-start
    result['input'] = dict(path=str(args.mean), sha256=hashlib.sha256(args.mean.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('retarded_initial_tangent_estimate','sampling_standard_error','independent_event_CAR_error','independent_event_Coulomb_error','elapsed_seconds')}))
    print('Natural exit:', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
