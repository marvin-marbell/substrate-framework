#!/usr/bin/env python3
"""Different noncompact photon law: Coulomb source theorem and coupled diagnostic.

The CLI is intentionally run by the integration owner after all suppliers land.
No finite oscillator/Fourier result is promoted to a full microscopic theorem.
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
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.special import eval_genlaguerre, gammaln
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'a406dca7a4cc642318250c028a763daa96fa66a5999da5dae0c96e53910c5c91'


def accept(error, assertion, tolerance=2e-8):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{assertion}: {error} > {tolerance}')
    return float(error)


def reject(call):
    try:
        call()
    except AssertionError:
        return True
    raise AssertionError('Adverse implementation passed the physical assertion')


def basis(number):
    return [sum(1 << i for i in occupied) for occupied in itertools.combinations(range(8), number)]


def ladder(source, target, mode):
    index = {word: i for i, word in enumerate(target)}
    rows, columns, values = [], [], []
    for i, word in enumerate(source):
        if not word & (1 << mode):
            new = word | (1 << mode)
            if new in index:
                rows.append(index[new]); columns.append(i)
                values.append((-1.) ** ((word & ((1 << mode)-1)).bit_count()))
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(target), len(source)))


def occupation(words):
    return np.array([(word & 15).bit_count() for word in words], dtype=float)


def unit(words, i, j):
    below = basis(words[0].bit_count()-1)
    return ladder(below, words, i) @ ladder(below, words, j).getH()


def phase_galerkin(count, coefficient, length):
    """Actual matrix elements of exp(i coefficient z), not exp(i truncated z)."""
    argument = coefficient*length/np.sqrt(2)
    rows, columns = np.indices((count, count))
    low, high = np.minimum(rows, columns), np.maximum(rows, columns)
    difference = high-low
    return (np.exp(-argument**2/2+(gammaln(low+1)-gammaln(high+1))/2)
            *(1j*argument)**difference
            *eval_genlaguerre(low, difference, argument**2))


def photon(count, radius, ue, restoring, theta, *, compact=False, erase_harmonic=False):
    labels = np.arange(-radius, radius+1)
    length = (ue/restoring)**.25
    oscillator = sparse.diags(np.sqrt(ue*restoring)*(2*np.arange(count)+1))
    ih = sparse.eye(len(labels), format='csr')
    if compact:
        # A genuinely wrong substitution, tested against the same quadratic form.
        cos = phase_galerkin(count, np.sqrt(2), length).real
        nodes, weights = hermgauss(max(80, 4*count))
        poly = np.zeros((count, len(nodes))); poly[0] = np.pi**(-.25)
        if count > 1: poly[1] = np.sqrt(2)*nodes*poly[0]
        for n in range(1, count-1):
            poly[n+1] = np.sqrt(2/(n+1))*nodes*poly[n]-np.sqrt(n/(n+1))*poly[n-1]
        q2 = (poly*(weights*(length*nodes)**2))@poly.T
        oscillator = oscillator-sparse.csr_matrix(restoring*q2)+sparse.csr_matrix(restoring*(np.eye(count)-cos))
    kinetic = ue*(labels+theta)**2/6
    if erase_harmonic: kinetic[:] = 0
    result = sparse.kron(oscillator, ih)+sparse.kron(sparse.eye(count), sparse.diags(kinetic))
    transverse = np.array([1., -1., 0.])/np.sqrt(2)
    steps = (1, 1, -2)
    phases = []
    for value, step in zip(transverse, steps):
        shift = sparse.diags(np.ones(len(labels)-abs(step)), -step, shape=(len(labels), len(labels)), format='csr')
        phases.append(sparse.kron(sparse.csr_matrix(phase_galerkin(count, value, length)), shift, format='csr'))
    return result.tocsr(), phases, labels


def coupled(number, count, radius, ue, restoring, theta, hopping=.23, *, compact=False, frozen=False, erase_harmonic=False) -> dict[str, Any]:
    words = basis(number); n = occupation(words)
    ph, phases, labels = photon(count, radius, ue, restoring, theta, compact=compact, erase_harmonic=erase_harmonic)
    ip = sparse.eye(ph.shape[0], format='csr'); im = sparse.eye(len(words), format='csr')
    # Complete canonical occupations; fixed negative reference at site1 in N=5.
    electric = sparse.kron(sparse.diags(ue*(n-2)**2/3), ip, format='csr')
    oscillator = sparse.kron(im, ph, format='csr')
    light = sparse.csr_matrix(electric.shape, dtype=complex)
    for phase in phases:
        for colour in range(4):
            term = sparse.kron(unit(words, colour, colour+4), phase, format='csr')
            light += -hopping*(term+term.conjugate().transpose())
    if frozen:
        # An actual wrong consumer, not a flag claiming an unchanged ground.
        light = sparse.csr_matrix(light.shape, dtype=complex)
    return dict(H=(electric+oscillator+light).tocsr(), electric=electric, photon=oscillator,
                light=light, words=words, n=n, ph=ph, labels=labels, phases=phases)


def ground(model, tolerance=2e-9):
    H = model['H']; rng = np.random.default_rng(2221005)
    initial = rng.normal(size=H.shape[0])
    values, vectors = eigsh(H, k=1, which='SA', v0=initial, tol=tolerance/20, maxiter=10000)
    psi = vectors[:, 0]; energy = float(values[0])
    residual = np.linalg.norm(H@psi-energy*psi)
    accept(residual, 'actual coupled canonical eigen residual', tolerance*max(1., abs(energy)))
    return energy, psi, residual

def enlarged_residual(count, radius, ue, restoring, theta, energy, psi):
    """Leakage into a strictly larger true Galerkin basis; not full-space proof."""
    larger = coupled(4, count+2, radius+2, ue, restoring, theta)
    embedded = np.zeros((70, count+2, 2*(radius+2)+1), dtype=complex)
    cell = psi.reshape(70, count, 2*radius+1)
    embedded[:, :count, 2:-2] = cell
    vector = embedded.reshape(-1)
    residual = float(np.linalg.norm(larger['H']@vector-energy*vector))
    return dict(enlarged_residual=residual, oscillator_count=count+2, harmonic_radius=radius+2,
                retained_oscillator_edge_probability=float(np.sum(abs(cell[:, -2:, :])**2)),
                retained_harmonic_edge_probability=float(np.sum(abs(cell[:, :, [0, -1]])**2)),
                full_space_residual_certified=False)


def source_maps(neutral, charged, refs=4):
    ip = sparse.eye(neutral['ph'].shape[0], format='csr')
    return [sparse.csr_matrix(sparse.kron(ladder(neutral['words'], charged['words'], a), ip, format='csr'), copy=False) for a in range(refs)]


def measure(neutral, charged, psi, energy, *, refs=4):
    maps = source_maps(neutral, charged, refs)
    images = [a@psi for a in maps]
    weight = sum(float(np.vdot(v, v).real) for v in images)
    cost = sum(np.vdot(v, charged['H']@v).real for v in images)/weight-energy
    comm = sum(np.vdot(v, charged['H']@v-a@(neutral['H']@psi)).real for a, v in zip(maps, images))/weight
    electric = sum(np.vdot(v, charged['electric']@v-a@(neutral['electric']@psi)).real for a, v in zip(maps, images))/weight
    light = sum(np.vdot(v, charged['light']@v-a@(neutral['light']@psi)).real for a, v in zip(maps, images))/weight
    mean = np.vdot(psi, sparse.kron(sparse.diags(neutral['n']), sparse.eye(neutral['ph'].shape[0]))@psi).real
    second = np.vdot(psi, sparse.kron(sparse.diags((neutral['n']-2)**2), sparse.eye(neutral['ph'].shape[0]))@psi).real
    weighted = sum(np.vdot(psi, a.getH()@a@(neutral['H']@psi)).real for a in maps)/weight
    # Photon-alone Rayleigh difference need NOT be the electric commutator.
    ph_difference = sum(np.vdot(v, (charged['electric']+charged['photon'])@v).real for v in images)/weight-np.vdot(psi, (neutral['electric']+neutral['photon'])@psi).real
    neutral_light = float(np.vdot(psi, neutral['light']@psi).real)
    return dict(weight=weight, mean=float(mean), variance_about_two=float(second), cost=float(cost),
                commutator=float(comm), electric_commutator=float(electric), light_commutator=float(light),
                weighted_neutral=float(weighted), photon_alone_difference=float(ph_difference),
                neutral_light_energy=neutral_light, independent_light_commutator=-neutral_light/4)


def physical_assertions(report, ue, *, omit_cross=False):
    covariance = report['variance_about_two']/3
    predicted = ue*(1/3-(0 if omit_cross else covariance))
    return dict(norm=accept(abs(report['weight']-2), 'complete four-reference norm is exactly2'),
                density=accept(abs(report['mean']-2), 'C/translation symmetric neutral density2'),
                electric=accept(abs(report['electric_commutator']-predicted), 'retained signed Coulomb Gauss covariance'),
                light=accept(abs(report['light_commutator']-report['independent_light_commutator']), 'actual retained half-directed light hopping current'),
                full=accept(abs(report['cost']-report['electric_commutator']-report['light_commutator']), 'full source Rayleigh transfer, not photon energy'))


def coulomb_torus(size, displacement):
    modes = np.array(list(itertools.product(range(size), repeat=3)))
    k = 2*np.pi*modes/size
    eigenvalues = np.sum(2-2*np.cos(k), axis=1)
    keep = eigenvalues > 1e-14
    r = np.sum(2*(1-np.cos(k[keep]@np.asarray(displacement)))/eigenvalues[keep])/size**3
    return float(r)


def wavepacket_error(ue, physical_momentum_variance):
    """Symmetric direct-integral packet around minimizing Bloch fibre.

    The same periodic-cell vector is used in nearby fibres. Linear current is
    retained before symmetric integration; only its integral vanishes.
    """
    return ue*physical_momentum_variance


def packet_measurement(model, psi, ue, theta, width=.08) -> dict[str, Any]:
    """Normalized uniform Bloch density: exact quadratic Gauss quadrature."""
    if abs(theta)+width >= .5:
        raise ValueError('Packet support must stay inside this chosen Bloch cell')
    nodes, weights = np.polynomial.legendre.leggauss(8)
    weights = weights/2
    offsets = width*nodes
    labels = model['labels']
    count = model['ph'].shape[0]//len(labels)
    probabilities = np.sum(abs(psi.reshape(len(model['words']), count, len(labels)))**2, axis=(0, 1))
    differences = []
    for offset in offsets:
        # Actual fibre-H difference; every hop and restoring coordinate remains.
        delta = ue*((labels+theta+offset)**2-(labels+theta)**2)/6
        differences.append(float(probabilities@delta))
    actual = float(weights@np.asarray(differences))
    variance = float(weights@(offsets**2)/6)
    predicted = wavepacket_error(ue, variance)
    accept(abs(actual-predicted), 'same normalized Bloch-packet kinetic excess')
    return dict(physical_momentum_variance=variance, retained_energy_excess=actual,
                predicted_excess=predicted, Bloch_support=[theta-width, theta+width],
                integrated_linear_current=float(ue/3*(probabilities@(labels+theta))*(weights@offsets)),
                normalization=float(weights.sum()),
                infimum_scope='sampled finite Galerkin fibre, never a normalizable full-R ground')

def supplier_upper(ue, coulomb_r, light_bound, reference_kinetic_bound, reference_current_bound,
                   normal_error_neutral, normal_error_charged, wavepacket_excess):
    values = (ue, coulomb_r, light_bound, reference_kinetic_bound, reference_current_bound,
              normal_error_neutral, normal_error_charged, wavepacket_excess)
    if any(not np.isfinite(x) or x < 0 for x in values):
        raise ValueError('All supplied certified form/residual bounds must be finite nonnegative')
    return float(ue*coulomb_r+light_bound+reference_kinetic_bound+reference_current_bound+
                 normal_error_neutral+normal_error_charged+wavepacket_excess)


def run(count=8, radius=4, ue=.7, restoring=1.1, thetas=(-.25, 0., .25)):
    if count < 3 or radius < 2 or ue <= 0 or restoring <= 0:
        raise ValueError('Need count>=3, radius>=2 and positive independent comparison coefficients')
    fibres = []
    for theta in thetas:
        n = coupled(4, count, radius, ue, restoring, theta)
        energy, psi, residual = ground(n)
        fibres.append((energy, theta, n, psi, residual))
    energy, theta, neutral, psi, residual = min(fibres, key=lambda row: row[0])
    leakage = enlarged_residual(count, radius, ue, restoring, theta, energy, psi)
    charged = coupled(5, count, radius, ue, restoring, theta)
    report = measure(neutral, charged, psi, energy)
    checks = physical_assertions(report, ue)
    source_ladders = [ladder(neutral['words'], charged['words'], a) for a in range(4)]
    norm_operator = sum((a.getH()@a for a in source_ladders), sparse.csr_matrix((70, 70)))
    checks['full_reference_norm_operator'] = accept(sparse.linalg.norm(norm_operator-sparse.diags(4-neutral['n'])),
                                                   'entire canonical CAR creator norm operator')
    D = np.array([[1., 1., 1.], [-1., -1., -1.]])
    j = D.T@np.linalg.pinv(D@D.T)@np.array([1., -1.])
    checks['Coulomb_Gauss'] = accept(np.linalg.norm(D@j-[1., -1.]), 'actual Coulomb current solves full U1 Gauss')
    shift_ratio = np.exp(2j*np.pi*j[0])
    def phase_assertion(ratio):
        return accept(abs(ratio-shift_ratio), 'same noncompact Coulomb phase translation law')
    checks['noncompact_phase_translation'] = phase_assertion(shift_ratio)
    wrongrefs = measure(neutral, charged, psi, energy, refs=1)
    adverse = dict(omitted_three_references=reject(lambda: physical_assertions(wrongrefs, ue)),
                   omitted_negative_cross=reject(lambda: physical_assertions(report, ue, omit_cross=True)))
    adverse['compact_source_periodicity'] = reject(lambda: phase_assertion(1.))
    wrongcompact = coupled(4, count, radius, ue, restoring, theta, compact=True)
    frozen = coupled(4, count, radius, ue, restoring, theta, frozen=True)
    erased = coupled(4, count, radius, ue, restoring, theta, erase_harmonic=True)
    def action_assertion(model):
        return accept(np.linalg.norm(model['H']@psi-neutral['H']@psi), 'same primitive full coupled noncompact Hamiltonian action')
    adverse['compact_periodicity_substitution'] = reject(lambda: action_assertion(wrongcompact))
    adverse['frozen_light_H'] = reject(lambda: action_assertion(frozen))
    adverse['erased_harmonic_kinetic'] = reject(lambda: action_assertion(erased))
    packet = packet_measurement(neutral, psi, ue, theta)
    excess = packet['retained_energy_excess']
    adverse['erased_harmonic_wavepacket'] = reject(lambda: accept(abs(0-excess), 'same normalized Bloch-packet kinetic excess'))
    # C-invariant local vacuum/full mixture: no n=2 weight despite norm2.
    density_adversary: dict[str, Any] = dict(local_probabilities=[.5, 0., 0., 0., .5], mean=2., P2=0., creator_norm=2.)
    adverse['C_implies_P2_floor'] = reject(lambda: accept(abs(density_adversary['P2']-1), 'C-symmetry does not certify occupation2'))
    r_rows = [dict(N=N, r=coulomb_torus(N, (N//2, N//2, N//2))) for N in (5, 8, 12, 20)]
    return dict(law='A in R; Hph=UE E^2+(K/2)|curl A|^2; no compact cosine replacement',
                plan=PLAN, assignment_plan_revision=PLAN_REVISION,
                source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                               for name in ('noncompact_coulomb_carrier.py', 'noncompact_coulomb_carrier.md')},
                comparison_coefficients=dict(UE=ue, K=restoring, native_coefficients_selected=False),
                graph=dict(sites=2, links=3, D=[[1,1,1],[-1,-1,-1]], curl=[[1,-1,0]],
                           harmonic=[1/np.sqrt(6),1/np.sqrt(6),-2/np.sqrt(6)], harmonic_period=2*np.pi*np.sqrt(6),
                           harmonic_Bloch_labels=neutral['labels'].tolist(), transverse_oscillator_count=count),
                dimensions=dict(neutral_CAR=70, charged_CAR=56, reference=4,
                                neutral_quantum=neutral['H'].shape[0], charged_with_reference=4*charged['H'].shape[0]),
                neutral_fibres=[dict(theta=row[1], energy=row[0], residual=row[4]) for row in fibres],
                sampled_minimizer=dict(theta=theta, energy=energy, residual=residual, actual_global_infimum_certified=False),
                enlarged_basis_diagnostic=leakage,
                source=report, assertions=checks, adverse=adverse,
                harmonic_wavepacket=packet,
                C_density_adversary=density_adversary, cubic_torus_resistance_diagnostics=r_rows,
                decision='Adopt noncompact Coulomb source for the changed-law upper-bound branch; retain BO, reference and localization debts.',
                limitations=['Finite Galerkin diagnostic, not oscillator/Fourier convergence or global Bloch minimization proof.',
                    'Heavy wholeNH, fields60V, colour links/Gauss15V and all64 centres enter the analytic conditional supplier theorem, not this light/photon diagnostic.',
                    'Uniform upper compensated energy is not a mobile band, isolated pole or separated-reference limit.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oscillators', type=int, default=8)
    parser.add_argument('--harmonic-radius', type=int, default=4)
    parser.add_argument('--ue', type=float, default=.7)
    parser.add_argument('--restoring', type=float, default=1.1)
    args = parser.parse_args()
    print(json.dumps(run(args.oscillators, args.harmonic_radius, args.ue, args.restoring), indent=2))


if __name__ == '__main__':
    main()
