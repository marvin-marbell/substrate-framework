#!/usr/bin/env python3
"""Normalizable quantumEM × field-dependent all64 CAR variational source.

The state uses negative Slaters of Honsite + eta Hhop(theta), with the true
projected-onsite phase. eta defines PREPARATION only; every energy/current
uses original full Href. No Gibbs trace, compact rotor, vacuum or pole claim.
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
from induced_phi_em_gaussian_variational import build_native, matter_hamiltonian, SOURCE_BYTES as NATIVE_BYTES

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**NATIVE_BYTES, Path(__file__).name: Path(__file__).read_bytes()}
SOURCE_HASHES = {name: hashlib.sha256(raw).hexdigest() for name, raw in SOURCE_BYTES.items()}


def path_derivative(native: dict[str, Any], profile: dict[str, Any], theta: np.ndarray,
                    direction: np.ndarray, order: int = 1, *, other: np.ndarray | None = None) -> np.ndarray:
    """Actual Href directional current/contact, never preparation eta vertices."""
    if order not in (1, 2):
        raise ValueError('Literal current/contact order must be1 or2')
    theta, direction = np.asarray(theta, float), np.asarray(direction, float)
    if theta.shape != (3*native['V'],) or direction.shape != theta.shape:
        raise ValueError('theta and direction must be real physical edge vectors')
    other = direction if other is None else np.asarray(other, float)
    if other.shape != direction.shape:
        raise ValueError('Mixed-contact direction must be a real physical edge vector')
    result = np.zeros((4*native['V'], 4*native['V']), complex)
    for path in profile['paths']:
        edges = list(path['edges'])
        coefficient = 1j*np.sum(direction[edges])
        if order == 2:
            coefficient *= 1j*np.sum(other[edges])
        hop = coefficient*path['hop']*np.exp(1j*np.sum(theta[edges]))
        x, y = path['source'], path['target']
        result[4*x:4*x+4, 4*y:4*y+4] += hop
        result[4*y:4*y+4, 4*x:4*x+4] += hop.conj().T
    return result


def occupied_section(profile: dict[str, Any], occupied: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Exterior Slater equals Pfilled Omega_onsite / positive overlap.

    At this orthogonal-Phi frozen reference, Gamma5 anticommutes with h.
    In Gamma0/Gamma5 coordinates h=[[M,iK],[-iK,-M]], M>0. The unique
    contractive negative graph is X=-iY, Y Hermitian. This positive-projection
    section has Tr(W* dW)=0, not arbitrary eigensolver determinant phase.
    """
    V = occupied.shape[0]//4
    values, columns = np.linalg.eigh(profile['onsite'])
    base = columns[:, values < 0]
    if base.shape != (4, 2) or occupied.shape != (4*V, 2*V):
        raise ArithmeticError('Original half-filled CAR dimensions changed')
    B = np.einsum('sa,xsk->xak', base.conj(), occupied.reshape(V, 4, 2*V)).reshape(2*V, 2*V)
    gram = B@B.conj().T
    spectrum, rotation = np.linalg.eigh(gram)
    if spectrum[0] <= 0:
        raise ArithmeticError('Projected-onsite Slater overlap vanished')
    inverse_root = (rotation/np.sqrt(spectrum))@rotation.conj().T
    W = occupied@(B.conj().T@inverse_root)
    overlap = B@(B.conj().T@inverse_root)
    return W, dict(minimum_overlap_gram=float(spectrum[0]),
        orthonormal_error=float(np.max(abs(W.conj().T@W-np.eye(2*V)))),
        positive_phase_error=float(np.max(abs(overlap-overlap.conj().T))))


def conditional_slater(native: dict[str, Any], profile: dict[str, Any], theta: np.ndarray,
                       eta: float, *, section: bool = True) -> dict[str, Any]:
    """True theta-dependent bundle; preparation spectrum is NOT a physical pole."""
    if not 0 <= eta <= 1:
        raise ValueError('Preparation eta must lie in[0,1]')
    physical = matter_hamiltonian(native, profile, theta)
    preparation = eta*physical
    for x in range(native['V']):
        preparation[4*x:4*x+4, 4*x:4*x+4] += (1-eta)*profile['onsite']
    values, vectors = np.linalg.eigh(preparation)
    rank = 2*native['V']
    if not values[rank-1] < 0 < values[rank]:
        raise ArithmeticError('Half-filled preparation gap closed')
    occupied, empty = vectors[:, :rank], vectors[:, rank:]
    result: dict[str, Any] = dict(physical_h=physical, preparation_h=preparation, values=values, vectors=vectors,
                  occupied=occupied, empty=empty, P=occupied@occupied.conj().T, load=profile['load'])
    if section:
        result['W'], result['phase'] = occupied_section(profile, occupied)
    return result


def projector_tangents(state: dict[str, Any], first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Exact first/second determinant-ray tangents, without intra-band pole floors.

    first/second are PREPARATION derivatives (eta times actual Href paths).
    On the flat determinant section, dOmega=dGamma(T)Omega and
    ddOmega=[dGamma(Tprime)+dGamma(T)^2]Omega. The square is an OPERATOR
    square, including two-PH/cross-flavour terms, not the zero matrix T@T.
    """
    values, U = state['values'], state['vectors']; rank = len(values)//2
    lower, upper = U[:, :rank], U[:, rank:]
    A = U.conj().T@first@U
    Bcv = upper.conj().T@second@lower
    delta = values[rank:, None]-values[None, :rank]
    D = -A[rank:, :rank]/delta
    curvature = -(Bcv+2*(A[rank:, rank:]@D-D@A[:rank, :rank]))/delta
    T = upper@D@lower.conj().T
    Tprime = (upper@curvature@lower.conj().T
              -lower@(D.conj().T@D)@lower.conj().T+upper@(D@D.conj().T)@upper.conj().T)
    dP = T+T.conj().T
    ddP = (upper@curvature@lower.conj().T+lower@curvature.conj().T@upper.conj().T
           -2*lower@(D.conj().T@D)@lower.conj().T+2*upper@(D@D.conj().T)@upper.conj().T)
    return dict(PH=D, second_PH=curvature, T=T, Tprime=Tprime, dP=dP, ddP=ddP,
                metric=float(np.sum(abs(D)**2)), determinant_Berry_connection=0.,
                scope='Flat frozen-orthogonal-Phi reference only; second operator square must retain all PH pairs and all64 copies.')


def edge_stencils(native: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Local 16-spin supports; never allocate a full h matrix per edge."""
    incidence: list[list[tuple[int, int]]] = [[] for _ in range(3*native['V'])]
    for index, path in enumerate(profile['paths']):
        for edge in set(path['edges']):
            incidence[edge].append((index, path['edges'].count(edge)))
    stencils = []
    for entries in incidence:
        sites = sorted({profile['paths'][i][key] for i, _ in entries for key in ('source', 'target')})
        lookup = {site: j for j, site in enumerate(sites)}
        rows = np.array([4*x+s for x in sites for s in range(4)])
        terms = [(i, multiplicity, slice(4*lookup[profile['paths'][i]['source']], 4*lookup[profile['paths'][i]['source']]+4),
                  slice(4*lookup[profile['paths'][i]['target']], 4*lookup[profile['paths'][i]['target']]+4))
                 for i, multiplicity in entries]
        stencils.append(dict(rows=rows, terms=terms))
    return stencils


def energy_integrand(native: dict[str, Any], theta: np.ndarray, eta: float,
                     stencils: list[list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Exact finite-box CAR/Born/Gauss integrand of the explicitly defined state.

    Full-link derivative metric minus independently computed realGauss
    density metric is the coexact+harmonic gradient norm, by literal gauge
    Ward identity. Physical Coulomb is added, not silently removed.
    """
    if not 0 <= eta <= 1:
        raise ValueError('Preparation eta must lie in[0,1]')
    theta = np.asarray(theta, float)
    if theta.shape != (3*native['V'],) or not np.all(np.isfinite(theta)):
        raise ValueError('theta must be a finite real edge vector')
    V = native['V']; rank = 2*V; u = native['UE']/native['a']
    if stencils is None:
        stencils = [edge_stencils(native, p) for p in native['Pprofiles']]
    total_car = full_metric = coulomb_metric = 0.; profiles = []
    for profile, support in zip(native['Pprofiles'], stencils):
        state = conditional_slater(native, profile, theta, eta, section=False)
        physical, values = state['physical_h'], state['values']
        lower, upper, P = state['occupied'], state['empty'], state['P']
        blocks = P.reshape(V, 4, V, 4)
        occupation = P.diagonal().reshape(V, 4).sum(axis=1).real
        mean_rho_error = float(np.max(abs(occupation-2)))
        if mean_rho_error > 1e-10:
            raise ArithmeticError('Chiral reference density no longer neutral')
        density_cov = np.diag(occupation)-np.einsum('xayb,ybxa->xy', blocks, blocks, optimize=True).real
        gauss_metric = float(np.trace(native['Lplus']@density_cov))
        car = float(np.einsum('ij,ji->', P, physical).real)
        inv_gap2 = 1/(values[rank:, None]-values[None, :rank])**2
        phases = np.array([np.exp(1j*np.sum(theta[list(p['edges'])])) for p in profile['paths']])
        metric = 0.; buffers: dict[int, tuple[np.ndarray, ...]] = {}
        for stencil in support:
            rows = stencil['rows']; width = len(rows)
            if width not in buffers:
                buffers[width] = (np.zeros((width, width), complex), np.empty((width, rank), complex),
                    np.empty((width, rank), complex), np.empty((width, rank), complex),
                    np.empty((width, rank), complex), np.empty((rank, rank), complex), np.empty((rank, rank)))
            local, plus, minus, conjugate, product, transition, squared = buffers[width]
            local.fill(0)
            for index, multiplicity, x, y in stencil['terms']:
                hop = 1j*eta*multiplicity*phases[index]*profile['paths'][index]['hop']
                local[x, y] += hop; local[y, x] += hop.conj().T
            np.take(upper, rows, axis=0, out=plus); np.take(lower, rows, axis=0, out=minus)
            np.conjugate(plus, out=conjugate)
            np.matmul(local, minus, out=product); np.matmul(conjugate.T, product, out=transition)
            np.absolute(transition, out=squared); np.square(squared, out=squared)
            np.multiply(squared, inv_gap2, out=squared); metric += float(np.sum(squared))
        transverse = metric-gauss_metric
        if min(gauss_metric, transverse) < -1e-8:
            raise ArithmeticError('Physical realGauss/coexact metric notpositive')
        load = profile['load']; total_car += load*car; full_metric += load*metric; coulomb_metric += load*gauss_metric
        profiles.append(dict(label=profile['label'], load=load, actual_car_energy_per_V=car/V,
            full_link_state_metric_per_V=metric/V, coexact_harmonic_metric_per_V=transverse/V,
            realGauss_density_metric_per_V=gauss_metric/V, mean_rho_error=mean_rho_error,
            preparation_gap=float(min(-values[rank-1], values[rank]))))
    return dict(actual_car_energy_per_V=total_car/V, coexact_Born_Huang_per_V=u*(full_metric-coulomb_metric)/V,
        physical_Gauss_Coulomb_per_V=u*coulomb_metric/V, actual_Href_integrand_per_V=(total_car+u*full_metric)/V,
        profiles=profiles)


def equal_time_observables(native: dict[str, Any], theta: np.ndarray, eta: float,
                           directions: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Actual matter current/contact in the true joint-state conditional fibre.

    Average raw_second_moment over the photon density BEFORE subtracting the
    averaged mean outer product. Averaging only conditional_covariance would
    miss between-field/current cross-species correlations in this pure state.
    This equal-time calculation supplies no retarded pole or stationarity.
    """
    count = len(directions)
    mean = np.zeros(count)
    covariance = np.zeros((count, count), complex)
    contact = np.zeros((count, count))
    for profile in native['Pprofiles']:
        state = conditional_slater(native, profile, theta, eta, section=False)
        P, lower, upper = state['P'], state['occupied'], state['empty']
        currents = [path_derivative(native, profile, theta, d) for d in directions]
        transitions = [upper.conj().T@J@lower for J in currents]
        load = profile['load']
        for i, J in enumerate(currents):
            mean[i] += load*float(np.einsum('ij,ji->', P, J).real)
            for j, d in enumerate(directions):
                covariance[i, j] += load*np.vdot(transitions[i], transitions[j])
                C = path_derivative(native, profile, theta, directions[i], 2, other=d)
                contact[i, j] += load*float(np.einsum('ij,ji->', P, C).real)
    return dict(mean_current=mean, conditional_covariance=covariance, contact=contact,
                raw_second_moment=covariance+np.outer(mean, mean))


def compute_source(mean: str | Path | dict[str, Any], *, size: int = 6, eta: float = .4,
                   variance: float = .5, samples: int = 4, seed: int = 20261008) -> dict[str, Any]:
    """Gaussian integral by reproducible sampling, not a certified ground/pole."""
    if not np.isfinite(variance) or variance <= 0 or samples < 2:
        raise ValueError('Positive finite variance and at least2 samples required')
    native = build_native(mean, size)
    support = [edge_stencils(native, p) for p in native['Pprofiles']]
    rng = np.random.default_rng(seed); rows = []
    for _ in range(samples):
        xi = np.sqrt(variance)*rng.standard_normal(native['M'])
        theta = native['Q']@xi
        rows.append(dict(theta=theta.tolist(), result=energy_integrand(native, theta, eta, support)))
    values = np.array([r['result']['actual_Href_integrand_per_V'] for r in rows])
    kinetic = native['UE']/native['a']*native['M']/(4*variance*native['V'])
    return dict(size=size, V=native['V'], real_EM_coordinates=native['M'], eta=eta, variance=variance,
        seed=seed, samples=samples, rows=rows, actual_Href_energy_per_V_estimate=float(np.mean(values)+kinetic),
        sampling_standard_error=float(np.std(values, ddof=1)/np.sqrt(samples)), bare_EM_kinetic_per_V=kinetic,
        geometry_errors=native['geometry_errors'], frame_image=native['frame_image'],
        source_hashes=SOURCE_HASHES, source_snapshots={name:raw.decode() for name,raw in SOURCE_BYTES.items()},
        state_definition='Normalized Gaussian density variance*sI on all realcoexact+harmonic coordinates, times all64 projected-onsite negativeSlaters of Honsite+etaHhop(theta); realGauss lift and Ref4/background unchanged. eta is preparation ONLY, original fullHref energy/current. Determinant connection flat only at frozenorthogonalPhi reference; no arbitraryeigenvectorphase.',
        validity='Finitebox midpoint diagnostic. Sampling standard error is NOT a rigorous integral enclosure; no interacting ground, stationarity, photonresidue, commoncone or macro source admission. FullPhi corrections/prices not supplied by this calculation.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mean', type=Path, required=True)
    parser.add_argument('--size', type=int, default=6)
    parser.add_argument('--eta', type=float, default=.4)
    parser.add_argument('--variance', type=float, default=.5)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    result = compute_source(args.mean, size=args.size, eta=args.eta, variance=args.variance, samples=args.samples, seed=args.seed)
    for name, raw in SOURCE_BYTES.items():
        if (ROOT/name).read_bytes() != raw:
            raise RuntimeError('Source changed during scientific calculation: '+name)
    result['elapsed_seconds'] = time.monotonic()-start
    result['input'] = dict(path=str(args.mean), sha256=hashlib.sha256(args.mean.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('size','eta','variance','samples','actual_Href_energy_per_V_estimate','sampling_standard_error','elapsed_seconds')}))
    print('Natural exit:', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
