#!/usr/bin/env python3
"""Actual full-coupling V4 bare-electric Galerkin consumer, no dense Fock K.

Every canonical occupation is retained. One-body Hermitian SF is applied by
its exact exterior-power Givens circuit. Ritz/Lanczos residuals are reported;
finite Krylov source windows are estimates, never certified poles or roots.
Normal1 samples y=0, chi=1: outside factor vanishes in THIS regulator only.
The declared full outside law remains globally nonidentity and is not proved
by this sampled chart-only computation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
import numpy as np
import scipy
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
import numba
from numba import njit
ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = Path(__file__).read_bytes()
MASS_BYTES = (ROOT/'physical_c_bare_electric_mass.py').read_bytes()
import physical_c_bare_electric_mass as mass
if mass.SOURCE_BYTES != MASS_BYTES:
    raise RuntimeError('Mass dependency changed during import')
core = mass.core; reference = mass.reference; alpha = mass.alpha


def graph(volume: int) -> dict[str, np.ndarray]:
    """Periodic Vx1x1, normalized real quotient with 2(V-1)+3 modes."""
    targets = np.column_stack(((np.arange(volume)+1) % volume, np.arange(volume), np.arange(volume)))
    d = np.zeros((volume, 3*volume))
    for x in range(volume):
        d[x, 3*x] = 1; d[(x+1) % volume, 3*x] = -1
    # Deterministic orthonormal zero-mean transverse modes, not fitted widths.
    spatial = np.zeros((volume, volume-1))
    for j in range(volume-1):
        spatial[:j+1, j] = 1/math.sqrt((j+1)*(j+2))
        spatial[j+1, j] = -(j+1)/math.sqrt((j+1)*(j+2))
    q = np.zeros((3*volume, 2*(volume-1)+3))
    for direction in (1, 2):
        q[direction::3, (direction-1)*(volume-1):direction*(volume-1)] = spatial
    for direction in range(3):
        q[direction::3, 2*(volume-1)+direction] = 1/math.sqrt(volume)
    if max(np.linalg.norm(d@q), np.linalg.norm(q.T@q-np.eye(q.shape[1]))) > 1e-12:
        raise ArithmeticError('Actual volume quotient failed')
    return dict(D=d, Q=q, Lplus=np.linalg.pinv(d@d.T), targets=targets)


def occupations(modes: int, number: int) -> dict[str, np.ndarray]:
    occupied = np.array(list(itertools.combinations(range(modes), number)), dtype=np.int32)
    masks = np.array([sum(1 << int(j) for j in row) for row in occupied], dtype=np.int32)
    lookup = np.full(1 << modes, -1, dtype=np.int32); lookup[masks] = np.arange(len(masks))
    first = []; second = []; signs = []
    for p in range(modes):
        for q in range(p+1, modes):
            indexes = np.flatnonzero(((masks >> p) & 1) & ~((masks >> q) & 1))
            partners = lookup[masks[indexes] ^ (1 << p) ^ (1 << q)]
            between = ((1 << q)-1) ^ ((1 << (p+1))-1)
            first.append(indexes.astype(np.int32)); second.append(partners)
            signs.append(np.array([(-1)**(int(mask) & between).bit_count() for mask in masks[indexes]], np.int8))
    return dict(occupied=occupied, masks=masks, lookup=lookup, first=np.array(first),
                second=np.array(second), signs=np.array(signs))


def givens_frame(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row QR. Diagonal phases cancel around canonical eigenvalue diagonal."""
    work = u.copy(); c = []; s = []; pairs = []; modes = len(u)
    for p in range(modes):
        for q in range(p+1, modes):
            a, b = work[p, p], work[q, p]
            norm = math.sqrt(abs(a)**2+abs(b)**2)
            cp, sp = (1.+0j, 0j) if norm == 0 else (a.conjugate()/norm, b.conjugate()/norm)
            top = work[p].copy(); bottom = work[q].copy()
            work[p] = cp*top+sp*bottom; work[q] = -sp.conjugate()*top+cp.conjugate()*bottom
            c.append(cp); s.append(sp); pairs.append(p*(2*modes-p-1)//2+q-p-1)
    if np.linalg.norm(work-np.diag(np.diag(work))) > 2e-11:
        raise ArithmeticError('Actual onebody eigenspace QR failed')
    return np.array(c), np.array(s), np.array(pairs, np.int32)


@njit(cache=False)
def rotate(buffer, a, b, signs, c, s, adjoint):
    for j in range(len(a)):
        ia, ib, parity = a[j], b[j], signs[j]
        for r in range(buffer.shape[1]):
            x, y = buffer[ia, r], buffer[ib, r]
            if adjoint:
                buffer[ia, r] = c.conjugate()*x-s*parity*y
                buffer[ib, r] = s.conjugate()*parity*x+c*y
            else:
                buffer[ia, r] = c*x+s*parity*y
                buffer[ib, r] = -s.conjugate()*parity*x+c.conjugate()*y


@njit(cache=False)
def compressed_sf(vector, basis, cs, ss, gate_pairs, diagonal, first, second, signs):
    fields, count, batches = vector.shape
    output = np.zeros((fields, count, batches), np.complex128)
    buffer = np.empty((count, batches), np.complex128)
    for sample in range(len(basis)):
        for j in range(count):
            for r in range(batches):
                value = 0j
                for field in range(fields):
                    value += basis[sample, field]*vector[field, j, r]
                buffer[j, r] = value
        for gate in range(cs.shape[1]):
            pair = gate_pairs[gate]
            rotate(buffer, first[pair], second[pair], signs[pair], cs[sample, gate], ss[sample, gate], False)
        for j in range(count):
            for r in range(batches):
                buffer[j, r] *= diagonal[sample, j]
        for gate in range(cs.shape[1]-1, -1, -1):
            pair = gate_pairs[gate]
            rotate(buffer, first[pair], second[pair], signs[pair], cs[sample, gate], ss[sample, gate], True)
        for field in range(fields):
            weight = basis[sample, field]
            for j in range(count):
                for r in range(batches):
                    output[field, j, r] += weight*buffer[j, r]
    return output


class CompleteCarrier:
    def __init__(self, volume: int, number: int, bare_mass: float, quadrature: int):
        self.volume = volume; self.number = number; self.geometry = graph(volume)
        self.occ = occupations(4*volume, number); self.count = len(self.occ['masks'])
        self.p = core.BareElectricParameters(photon_grade=1, normal_order=1, colour_order=1,
                                            bare_mass=bare_mass, quadrature_order=quadrature)
        self.law = alpha.CartesianCWilson(alpha.gamma_frame(), alpha.Parameters(mass=bare_mass, spacing=.01), self.geometry['targets'])
        modes = self.geometry['Q'].shape[1]; coexact = 2*(volume-1)
        nodes, weights = reference.hermite_rule(quadrature)
        coordinates = np.array(list(itertools.product(nodes, repeat=modes)))
        probability_half = np.array([math.prod(row) for row in itertools.product(weights, repeat=modes)])
        self.basis = probability_half[:, None]*np.column_stack((np.ones(len(coordinates)), math.sqrt(2)*coordinates))
        if np.linalg.norm(self.basis.T@self.basis-np.eye(modes+1)) > 2e-11:
            raise ArithmeticError('Full real coordinate photon basis not orthonormal')
        inverse_width = np.r_[np.full(coexact, math.sinh(2*math.asinh(1.))), np.ones(3)]
        covers = coordinates/np.sqrt(inverse_width)
        half = 1+self.p.electric_coefficient*inverse_width/2
        full = 1+self.p.electric_coefficient*inverse_width
        vacuum = float(np.prod(half**(-.5)))
        self.field_heat = np.r_[vacuum, vacuum/half]
        full_diagonal = np.r_[np.prod(full**(-.5)), np.prod(full**(-.5))/full]
        self.heat_leak = float(np.sqrt(max(full_diagonal-self.field_heat**2)))
        colour = np.broadcast_to(np.eye(4), (volume, 3, 4, 4)).copy()
        remainder = np.zeros((volume, 4, 4, 4))
        cs = []; ss = []; diagonal = []; self.packets = []
        local_law, _, _ = reference.native_law(core.reference_parameters(self.p))
        self.onebody_logs = []
        for cover in covers:
            phases = (alpha.charge_contract(alpha.Parameters())['bare_e']*self.geometry['Q']@cover).reshape(volume, 3)
            potential = self.law.potential(phases, remainder, colour)
            ev, frame = eigh(potential['h'])
            c, s, self.gate_pairs = givens_frame(frame)
            cs.append(c); ss.append(s)
            diagonal.append(np.exp((potential['pref']+np.sum(ev[self.occ['occupied']], axis=1))/2))
            self.onebody_logs.append(potential['h'])
            packet = np.zeros((4*volume, 4), complex)
            packet[:8] = local_law.source_packet_jet(phases[:2], colour[:2], (np.zeros((2, 3)), np.zeros((2, 3))))['psi']
            self.packets.append(packet)
        self.cs = np.array(cs); self.ss = np.array(ss); self.diagonal = np.array(diagonal)
        self.packets = np.array(self.packets)
        e = alpha.charge_contract(alpha.Parameters())['bare_e']
        density = np.array([[((int(mask) >> (4*x)) & 15).bit_count()-2 for x in range(volume)] for mask in self.occ['masks']], float)
        density[:, 1] -= number-2*volume
        if np.max(abs(np.sum(density, axis=1))) > 0:
            raise ArithmeticError('Extended original static Ref compensation failed')
        self.coulomb = self.p.electric_coefficient*e**2*np.einsum('ni,ij,nj->n', density, self.geometry['Lplus'], density)
        self.heat_square = self.field_heat[:, None]**2*np.exp(-self.coulomb)[None, :]
        self.dimension = (modes+1)*self.count; self.calls = 0; self.elapsed = 0.
        self.SF_norm_bound = float(np.max(self.diagonal))
        self.K_norm_upper_bound = self.SF_norm_bound**2*float(np.max(self.heat_square))

    def apply_sf(self, vector: np.ndarray) -> np.ndarray:
        return compressed_sf(vector, self.basis, self.cs, self.ss, self.gate_pairs, self.diagonal,
                             self.occ['first'], self.occ['second'], self.occ['signs'])

    def apply_K(self, vector: np.ndarray, batches: int = 1) -> np.ndarray:
        start = time.time()
        state = vector.reshape(len(self.field_heat), self.count, batches)
        middle = self.apply_sf(state)
        middle *= self.heat_square[:, :, None]
        result = self.apply_sf(middle).reshape(-1)
        self.calls += 1; self.elapsed += time.time()-start
        return result

    def mean_boundary(self) -> np.ndarray:
        colour = np.broadcast_to(np.eye(4), (self.volume, 3, 4, 4)).copy()
        potential = self.law.potential(np.zeros((self.volume, 3)), np.zeros((self.volume, 4, 4, 4)), colour)
        _, frame = eigh(potential['h'])
        occupied = frame[:, -self.number:]
        canonical = np.linalg.det(occupied[self.occ['occupied']])
        state = np.zeros((len(self.field_heat), self.count), complex); state[0] = canonical
        return state.reshape(-1)/np.linalg.norm(state)

    def diagnostics(self) -> dict[str, Any]:
        return dict(volume=self.volume, number=self.number, complete_canonical_states=self.count,
            complete_field_dimension=len(self.field_heat), complete_total_dimension=self.dimension,
            real_quotient_modes=self.geometry['Q'].shape[1], quadrature_points=len(self.basis),
            all_full_transfer_couplings=True, B_global_lower=float(self.law.B_global_lower),
            exact_FREE_halfheat_omitted_Hermite_operator_norm=self.heat_leak,
            declared_full_K_operator_norm_upper_bound=self.K_norm_upper_bound,
            actual_K_applications=self.calls, actual_K_application_seconds=self.elapsed,
            normal1_sampled_outside_factor_zero=True, global_outside_law_nonidentity_not_deleted=True,
            source_support_sites=[0, 1], Ref_compensation_site=1)


@njit(cache=False)
def source_apply(state, coefficients, target, signs, destination_count):
    fields, original_count = state.shape
    result = np.zeros((fields, destination_count, 4), np.complex128)
    for mode in range(len(target)):
        for j in range(original_count):
            dest = target[mode, j]
            if dest < 0:
                continue
            parity = signs[mode, j]
            for outgoing in range(fields):
                for incoming in range(fields):
                    value = parity*state[incoming, j]
                    for ref in range(4):
                        result[outgoing, dest, ref] += coefficients[mode, ref, outgoing, incoming]*value
    return result.reshape(-1)


def apply_original_source(neutral: CompleteCarrier, charged: CompleteCarrier, state: np.ndarray) -> np.ndarray:
    coefficients = np.einsum('ni,nba,nj->baij', neutral.basis, neutral.packets, neutral.basis, optimize=True)
    target = np.full((4*neutral.volume, neutral.count), -1, np.int32); signs = np.zeros_like(target)
    addition = charged.number > neutral.number
    for mode in range(4*neutral.volume):
        for j, rawmask in enumerate(neutral.occ['masks']):
            mask = int(rawmask); occupied = bool(mask & (1 << mode))
            if occupied == addition:
                continue
            target[mode, j] = charged.occ['lookup'][mask ^ (1 << mode)]
            signs[mode, j] = (-1)**(mask & ((1 << mode)-1)).bit_count()
    if not addition:
        coefficients = coefficients.conj()
    return source_apply(state.reshape(len(neutral.field_heat), neutral.count), coefficients,
                        target, signs, charged.count)


class CompleteOperator(LinearOperator):
    """SciPy's documented subclass interface to the exact canonical action."""
    def __init__(self, carrier: CompleteCarrier):
        self.carrier = carrier
        super().__init__(dtype=np.dtype(np.complex128), shape=(carrier.dimension, carrier.dimension))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.carrier.apply_K(x)


def extremal(carrier: CompleteCarrier, args: argparse.Namespace) -> dict[str, Any]:
    operator = CompleteOperator(carrier)
    start = time.time(); converged = True
    try:
        values, vectors = eigsh(operator, k=1, which='LA', v0=carrier.mean_boundary(),
                               tol=args.eig_tolerance, maxiter=args.eig_maxiter, ncv=args.eig_ncv)
    except ArpackNoConvergence as failure:
        converged = False; values = failure.eigenvalues; vectors = failure.eigenvectors
    if values is None or vectors is None or len(values) == 0:
        raise ArithmeticError('No actual extremal Ritz pair converged; refusing a Rayleigh-as-norm anchor')
    index = int(np.argmax(values)); value = float(values[index].real); vector = vectors[:, index]
    residual = float(np.linalg.norm(carrier.apply_K(vector)-value*vector)/value)
    print(json.dumps(dict(stage='extremal_Ritz', number=carrier.number, volume=carrier.volume,
        value=value, relative_residual=residual, solver_converged=converged, K_calls=carrier.calls)), flush=True)
    return dict(largest_Ritz_value=value, relative_eigenpair_residual=residual, solver_converged=converged,
        solver_tolerance=args.eig_tolerance, solver_maxiter=args.eig_maxiter, runtime_seconds=time.time()-start,
        full_operator_norm_certified=False, analytical_K_norm_upper_bound=carrier.K_norm_upper_bound)


def spectral_probe(carrier: CompleteCarrier, source: np.ndarray, lambdaN: float, steps: int) -> dict[str, Any]:
    """Aggregate all4 Ref columns in one exact direct-sum Krylov chain."""
    gram = float(np.vdot(source, source).real)
    if gram <= 0:
        raise ArithmeticError('Original full Ref source vanishes')
    current = source/math.sqrt(gram); previous = np.zeros_like(current)
    basis = []; diagonal = []; offdiagonal = []; beta = 0.; next_beta = 0.
    for iteration in range(steps):
        basis.append(current.copy())
        work = carrier.apply_K(current, batches=4)/lambdaN
        a = float(np.vdot(current, work).real); work -= a*current+beta*previous
        # Reorthogonalization is needed for consumer band weights, not a floor.
        for old in basis:
            work -= np.vdot(old, work)*old
        next_beta = float(np.linalg.norm(work)); diagonal.append(a)
        if next_beta < 1e-13:
            break
        if iteration+1 < steps:
            offdiagonal.append(next_beta)
        previous, current, beta = current, work/next_beta, next_beta
    tridiagonal = np.diag(diagonal)
    if offdiagonal:
        tridiagonal += np.diag(offdiagonal, 1)+np.diag(offdiagonal, -1)
    values, frame = eigh(tridiagonal)
    if min(values) <= 0:
        raise ArithmeticError('Source Krylov positive Ritz nodes unresolved; no clipping')
    raw_weights = gram*abs(frame[0])**2
    weights = raw_weights*values; energy = -np.log(values)
    half_gram = float(sum(weights)); first = float(weights@energy)
    return dict(Krylov_steps_requested=steps, Krylov_dimension=len(diagonal),
        raw_source_Gram=gram, exact_ONE_halfstep_Gram=gram*diagonal[0],
        source_supported_Ritz_ratios=values.tolist(), aggregate_raw_source_Ritz_weights=raw_weights.tolist(),
        aggregate_HALF_source_Ritz_weights=weights.tolist(), energies_TRANSFER_estimates=energy.tolist(),
        fullsource_E1_PHYSICAL_estimate=first/.01, fullsource_mean_PHYSICAL_estimate=first/half_gram/.01,
        transfer_Pe0_4_fraction_estimate=float(sum(weights[(energy >= 0) & (energy <= 4)])/half_gram),
        physical_Pe0_4_fraction_estimate=float(sum(weights[(energy >= 0) & (energy <= .04)])/half_gram),
        negative_energy_fraction_estimate=float(sum(weights[energy < 0])/half_gram),
        final_Lanczos_beta=next_beta,
        hard_window_support_certified=False, finite_Krylov_not_complete_spectrum=True)


def dense_V2_calibration() -> dict[str, Any]:
    results = []; rng = np.random.default_rng(222)
    p = core.BareElectricParameters(photon_grade=1, normal_order=1, colour_order=1)
    field = core.field_regulator(p)
    for number in (4, 5, 3):
        carrier = CompleteCarrier(2, number, mass.OLD_MASS, 2)
        dense = core.native_factor(p, number, 0, field)
        # Captured canonical convention is identical combinations order.
        if list(map(int, carrier.occ['masks'])) != dense['masks']:
            raise ArithmeticError('Actual canonical order differs')
        vector = rng.normal(size=carrier.dimension)+1j*rng.normal(size=carrier.dimension)
        vector /= np.linalg.norm(vector)
        expected = dense['vectors']@(dense['values']*(dense['vectors'].conj().T@vector))
        actual = carrier.apply_K(vector)
        error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        sample = 0
        h, _ = alpha.second_quantized(carrier.onebody_logs[sample], number)
        pref = reference.native_law(core.reference_parameters(p))[0].potential(
            (alpha.charge_contract(alpha.Parameters())['bare_e']*carrier.geometry['Q']@
             (np.array(list(itertools.product(reference.hermite_rule(2)[0], repeat=5)))[sample]/
              np.sqrt(np.r_[np.full(2, math.sinh(2*math.asinh(1.))), np.ones(3)]))).reshape(2, 3),
            np.zeros((2, 4, 4, 4)), np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy())['pref']
        single = expm((h+pref*np.eye(carrier.count))/2)
        local = rng.normal(size=carrier.count)+1j*rng.normal(size=carrier.count)
        result = compressed_sf(local[None, :, None], np.ones((1, 1)), carrier.cs[:1], carrier.ss[:1],
            carrier.gate_pairs, carrier.diagonal[:1], carrier.occ['first'], carrier.occ['second'], carrier.occ['signs']).reshape(-1)
        sf_error = float(np.linalg.norm(result-single@local)/np.linalg.norm(single@local))
        if max(error, sf_error) > 2e-10:
            raise ArithmeticError('Exact exterior circuit fails actual V2 dense Gamma/K calibration')
        results.append(dict(number=number, actual_dense_Gamma_action_relative_error=sf_error,
            actual_dense_complete_K_action_relative_error=error, complete_canonical_states=carrier.count))
    return dict(rows=results, actual_captured_V2_positive_controls=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    calibration = dense_V2_calibration()
    print(json.dumps(dict(stage='actual_V2_dense_calibration', result=calibration)), flush=True)
    if args.calibration_only:
        return dict(actual_V2_dense_calibration=calibration)
    carriers = {}; spectra = {}; neutral_number = 2*args.volume
    charged_numbers = (neutral_number+1, neutral_number-1)
    for number in (neutral_number, *charged_numbers):
        start = time.time(); carrier = CompleteCarrier(args.volume, number, args.bare_mass, args.quadrature_order)
        print(json.dumps(dict(stage='actual_carrier_built', volume=args.volume, number=number, dimension=carrier.dimension,
            canonical_states=carrier.count, quadrature_points=len(carrier.basis), seconds=time.time()-start)), flush=True)
        spectra[number] = extremal(carrier, args); carriers[number] = carrier
    neutral = carriers[neutral_number]; common = spectra[neutral_number]['largest_Ritz_value']
    target = neutral.mean_boundary(); prefix_gram = 1.
    for _ in range(core.BareElectricParameters().prefix_steps):
        target = neutral.apply_K(target)/common
    prefix_gram = float(np.vdot(target, target).real); target /= math.sqrt(prefix_gram)
    photon = np.zeros_like(target).reshape(len(neutral.field_heat), neutral.count)
    photon[1] = target.reshape(len(neutral.field_heat), neutral.count)[0]
    photon_gram = float(np.vdot(photon, photon).real)
    if photon_gram <= 0:
        raise ArithmeticError('Captured projected photon insertion vanishes')
    preparations = [('pure_original_S', target), ('original_projected_photon_then_S', photon.reshape(-1)/math.sqrt(photon_gram))]
    source_rows = []
    for number in charged_numbers:
        carrier = carriers[number]
        for label, incoming in preparations:
            source = apply_original_source(neutral, carrier, incoming)
            probes = []
            for steps in args.source_steps:
                probe = spectral_probe(carrier, source, common, steps); probes.append(probe)
                print(json.dumps(dict(stage='actual_source_probe', volume=args.volume, number=number, preparation=label,
                    steps=steps, mean_PHYSICAL_estimate=probe['fullsource_mean_PHYSICAL_estimate'],
                    physical_Pe4_estimate=probe['physical_Pe0_4_fraction_estimate'])), flush=True)
            source_rows.append(dict(number=number, preparation=label, probes=probes))
    return dict(actual_V2_dense_calibration=calibration, parameters=dict(bare_mass=args.bare_mass,
        quadrature_order=args.quadrature_order, spacing=.01, UE=.02, volume=args.volume, centre=0, normal_order=1,
        colour_order=1, real_modes=neutral.geometry['Q'].shape[1], photon_dimension=len(neutral.field_heat)),
        actual_extremal_Ritz=spectra, ONE_neutral_Ritz_lambda_N=common,
        charged_top_gap_TRANSFER_estimates={str(n): -math.log(spectra[n]['largest_Ritz_value']/common) for n in charged_numbers},
        prefix_Gram=prefix_gram, original_projected_photon_Gram=photon_gram,
        original_full_Ref_source_probes=source_rows, actual_complete_carrier_diagnostics={str(n): c.diagnostics() for n, c in carriers.items()},
        global_full_family=core.full_family_definition(),
        limits=['No dense Fock operator allocated; ALL canonical occupations retained. Ritz solver residual is not a rigorous global eigenvalue enclosure.',
            'Source firstmoments and hard Pe windows are finite Lanczos spectral estimates, not complete spectral/pole claims.',
            'Normal1 y=0 makes sampled outsidefactor zero; declared global outside remains nonidentity, no outside proof inferred.',
            'Centre0, photonvacuum+ALL retained single-coordinate Hermite states and finite quadrature; no continuum, four-centre or kineticmass admission.',
            'Original projected photon insertion is a basis-limited packet, not a full creator/CCR claim.'],
        physical_mass1_or_electronpole_admitted=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bare-mass', type=float, default=mass.OLD_MASS)
    parser.add_argument('--volume', type=int, choices=(2, 4), default=4)
    parser.add_argument('--quadrature-order', type=int, default=2)
    parser.add_argument('--eig-tolerance', type=float, default=1e-6)
    parser.add_argument('--eig-maxiter', type=int, default=60)
    parser.add_argument('--eig-ncv', type=int, default=8)
    parser.add_argument('--source-steps', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--calibration-only', action='store_true')
    args = parser.parse_args(); start = time.time()
    inputs = json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, sort_keys=True)
    result = run(args)
    source_map = {Path(__file__).name: SOURCE_BYTES, 'physical_c_bare_electric_mass.py': MASS_BYTES,
        'physical_c_bare_electric_kernel.py': mass.CORE_BYTES, 'captured_reference89.py': core.REFERENCE_BYTES,
        'captured_ledger.py': reference.LEDGER_BYTES,
        **{'captured_Alpha56/'+name: data for name, data in alpha.SOURCE_BYTES.items()}}
    unchanged = {Path(__file__).name: Path(__file__).read_bytes() == SOURCE_BYTES,
                 'physical_c_bare_electric_mass.py': (ROOT/'physical_c_bare_electric_mass.py').read_bytes() == MASS_BYTES,
                 'physical_c_bare_electric_kernel.py': (ROOT/'physical_c_bare_electric_kernel.py').read_bytes() == mass.CORE_BYTES}
    unchanged.update({name: (ROOT/name).read_bytes() == data for name, data in mass.INPUT_BYTES.items()})
    if not all(unchanged.values()):
        raise RuntimeError('Source or input changed during actual V4 experiment')
    execution: dict[str, Any] = dict(source_at_IMPORT={name: hashlib.sha256(data).hexdigest() for name, data in source_map.items()},
        source_bytes_at_IMPORT={name: data.decode() for name, data in source_map.items()},
        input_json=inputs, input_sha256=hashlib.sha256(inputs.encode()).hexdigest(), startup_unchanged=unchanged,
        capsule_hashes_at_IMPORT={name: hashlib.sha256(data).hexdigest() for name, data in mass.INPUT_BYTES.items()},
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                     numba=numba.__version__, platform=platform.platform(),
                     thread_environment={name: os.environ.get(name) for name in
                         ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')}),
        runtime_seconds=time.time()-start,
        governing_plan=mass.PLAN_ID)
    result['execution'] = execution
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stage='capsule_written', output=str(args.output), runtime_seconds=time.time()-start,
        source_sha256=execution['source_at_IMPORT'][Path(__file__).name])), flush=True)


if __name__ == '__main__':
    main()
