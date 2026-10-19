#!/usr/bin/env python3
"""Coupled neutral Slater supplier; native diagnostics and full-CAR loop comparison.

No scientific execution is performed on import. UE and K must be supplied by the
parent: neither coefficient is selected or inferred by this consumer.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import original_frame, portable
from original_compact_charge_bound import chiral_basis, metric_constants
from original_uniform_charge_gap import exterior_hamiltonian
from rotation_band import RotationBand
from sea_selection import coefficients
from unrestricted_field_minimum import BARE

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'a406dca7a4cc642318250c028a763daa96fa66a5999da5dae0c96e53910c5c91'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def close(actual, expected, label, tolerance=3e-8):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected)))
    scale = max(1., float(np.linalg.norm(expected)))
    if not math.isfinite(error) or error > tolerance*scale:
        raise AssertionError(f'{label}: relative residual {error/scale}')
    return error/scale


def adverse(label, assertion):
    try:
        assertion()
    except AssertionError as exc:
        return {'case': label, 'rejected_by_same_assertion': str(exc)}
    raise AssertionError(f'Adverse input escaped: {label}')


def negative_frame(beta, volume):
    values, vectors = eigh(beta)
    return np.kron(np.eye(volume), vectors[:, values < 0])


def polar_section(h, S):
    values, vectors = eigh(h)
    rank = h.shape[0]//2
    if values[rank-1] >= 0 or values[rank] <= 0:
        raise AssertionError('Actual complete negative cluster has wrong rank/gap')
    occupied = vectors[:, :rank]
    P = occupied@occupied.conj().T
    C = S.conj().T@P@S
    c, u = eigh(C)
    if c[0] <= 0:
        raise AssertionError('The fixed beta-minus overlap is singular')
    inverse_root = (u*c**-.5)@u.conj().T
    Z = P@S@inverse_root
    return {'values': values, 'vectors': vectors, 'P': P, 'C': C,
            'C_values': c, 'C_vectors': u, 'inverse_root': inverse_root,
            'Z': Z, 'sea': float(values[:rank].sum())}


def frame_assertions(data, h, S, beta, mu, B):
    Z, P = data['Z'], data['P']
    close(Z.conj().T@Z, np.eye(Z.shape[1]), 'complete Slater normalization')
    close(Z@Z.conj().T, P, 'actual negative projector, not positive/frozen sea')
    overlap = (1+mu/B)/2
    if float(data['C_values'][0]) < overlap-3e-8:
        raise AssertionError('negative-subspace Lyapunov overlap lower bound')
    coercive = (beta@h+h@beta)/2
    if eigh(coercive, eigvals_only=True, subset_by_index=[0, 0])[0] < mu-3e-8:
        raise AssertionError('canonical beta coercivity with correct polarity')
    close(np.trace(Z.conj().T@h@Z).real, data['sea'],
          'Slater expectation equals actual all-occupied sea energy')


def section_derivative(data, S, dh):
    values, vectors = data['values'], data['vectors']
    occupation = np.zeros(len(values)); occupation[:len(values)//2] = 1
    differences = values[:, None]-values[None, :]
    numerator = occupation[:, None]-occupation[None, :]
    divided = np.divide(numerator, differences, out=np.zeros_like(differences),
                        where=numerator != 0)
    dP = vectors@(divided*(vectors.conj().T@dh@vectors))@vectors.conj().T
    dC = S.conj().T@dP@S
    c, u = data['C_values'], data['C_vectors']
    # Exact divided difference of x^-1/2, including repeated eigenvalues.
    roots = np.sqrt(c)
    loewner = -1/(roots[:, None]*roots[None, :]*(roots[:, None]+roots[None, :]))
    d_inverse = u@(loewner*(u.conj().T@dC@u))@u.conj().T
    dZ = dP@S@data['inverse_root']+data['P']@S@d_inverse
    berry = np.trace(data['Z'].conj().T@dZ)
    metric = float(np.linalg.norm(dP)**2/2)
    omega_norm = float(np.linalg.norm(dZ)**2
                       -np.linalg.norm(data['Z'].conj().T@dZ)**2+abs(berry)**2)
    close(berry, 0., 'actual global polar section has zero Berry trace')
    close(omega_norm, metric, 'Slater derivative includes quantum metric')
    return {'dP': dP, 'dZ': dZ, 'berry': berry, 'metric': metric}


def slater_vector(Z, basis):
    return np.asarray([np.linalg.det(Z[[j for j in range(len(Z)) if state >> j & 1]])
                       for state in basis])


def eulerian_sum(power, rate):
    """Exact positive series sum_(R>=0) (R+1)^power exp(-rate R)."""
    q = math.exp(-rate)
    if power == 0:
        return 1/(1-q)
    polynomial = [1]
    for n in range(2, power+1):
        previous = polynomial
        polynomial = [(k+1)*(previous[k] if k < len(previous) else 0)
                      +(n-k)*(previous[k-1] if k else 0) for k in range(n)]
    return sum(coefficient*q**k for k, coefficient in enumerate(polynomial))/(1-q)**(power+1)


def tail_series(power, rate, first):
    """Exact translated polynomial tail, no numerical radius truncation."""
    return math.exp(-rate*first)*sum(math.comb(power, j)*first**(power-j)
                                    *eulerian_sum(j, rate) for j in range(power+1))


def sea_constants(frame, gram, band, size):
    """Conservative source-envelope constants, not measured optimal stiffness.

    See companion proof for every coefficient and the static T->infinity limit.
    The series are bounded by positive polynomials and summed analytically.
    """
    mu = metric_constants(band, gram, size**3)['mu']
    B = math.sqrt(gram[0, 0])*(band.m+6*band.r/band.a) \
        +abs(band.v)*sum(math.sqrt(gram[j, j]) for j in range(1, 4))/band.a
    hops = [-(band.r*frame[0]+1j*band.v*frame[j])/(2*band.a) for j in range(1, 4)]
    Ch = 2*sum(float(np.linalg.norm(F, 2)) for F in hops)
    gamma = mu/4
    aa = math.log1p(mu/(4*Ch))
    xi = mu/4; rho = mu-gamma-xi; T0 = 1/B
    C = math.exp(2)*max((8/T0+4*(B+gamma+xi))/rho, 4)
    Ptime = (1/math.tanh(aa/4))**3*(4/gamma)
    Jv = 1/band.a+Ch
    R0 = math.ceil(max(3, gamma*T0/2))+math.ceil(2*aa)
    schur = metric_constants(band, gram, size**3)['spatial_Schur']
    c = (band.m+3*band.r/band.a)*math.sqrt(schur)
    _, _, beta = chiral_basis(frame, gram, 1)
    O = (band.m+3*band.r/band.a)*(frame[0]-math.sqrt(schur)*beta)
    v_alpha = float(np.linalg.norm(O, 2))+Ch*math.exp(aa)
    qtail = max(c, math.log(2)/T0, gamma+8*v_alpha, gamma+4/band.a)
    Mtail = qtail-c
    Btail = 2**20*(v_alpha**2+band.a**-2+T0**-2)
    V0 = Mtail*(4*C+2)+Btail/(qtail-gamma)
    Hlow = 64*max(1., C)**3*(1+Jv*Ptime)**2
    H0 = Mtail*Hlow+Btail/(qtail-gamma)
    # L_R <= R/aa+1. S_R <= Scoef (R+1)^4, at every R>=0.
    Scoef = 2*(2/aa+5)**3*(1+2*Ch/gamma)
    Acoef = 512*max(1., C)**4*(1+Jv*Ptime)**2*Scoef*math.exp((1+2*aa)/2)
    # e_R=E_R+E_(R-1), conservatively extended to ALL R>=0.
    d4 = 2*Mtail*Acoef*math.exp(.5)
    d2 = 4*math.sqrt(Acoef*Btail)*math.exp(.25)
    L0 = math.ceil(R0/aa)
    moment = H0*(L0+1)**4+(1/aa+2)**4 \
        *(d4*eulerian_sum(8, .5)+d2*eulerian_sum(6, .25))
    CB = 9*math.pi**2*moment/4
    # 2L_R+1 >= N is noncontractible; cubes are restored, not cut away.
    threshold = math.floor(aa*(math.ceil((size-1)/2)-1))+1
    top = max(R0, threshold)
    RN = 2*(d4*tail_series(4, .5, top)+d2*tail_series(2, .25, top))
    if L0*2+1 >= size:
        RN += 2*V0
    CQ_axis = [(band.r**2*gram[0, 0]+band.v**2*gram[j, j])
               /(4*band.a**2*mu**2) for j in range(1, 4)]
    return {'mu': mu, 'B': B, 'Ch': Ch, 'alpha_a': aa, 'gamma': gamma,
            'T0': T0, 'R0': R0, 'L_R0': L0, 'R_top': top,
            'initial_wraps': L0*2+1 >= size, 'C_B': CB, 'R_N': RN,
            'C_Q_axis': CQ_axis, 'C_Q': max(CQ_axis),
            'static_source_envelope': {'C': C, 'H0': H0, 'V0': V0,
                                       'Acoef': Acoef, 'Btail': Btail,
                                       'd4': d4, 'd2': d2},
            'sums': 'positive analytic Eulerian sums; no deleted wrapping tail'}


def mean_domain(constants, UE, K):
    CB, RN, CQ = constants['C_B'], constants['R_N'], constants['C_Q']
    if K <= CB:
        return {'condition': 'K>C_B', 'satisfied_at_supplied_inputs': False,
                'conclusion': 'No coupled small-mean bound inferred from these constants'}
    g = math.sqrt(UE/(2*(K+CB)))
    bound = (3*math.sqrt(2*UE*(K+CB))+3*UE*CQ+2*RN)/(K-CB)
    return {'condition': 'K>C_B', 'satisfied_at_supplied_inputs': True,
            'theta_g': g, 'mean_sum_flux_over_V_upper': min(6., bound),
            'unclipped_upper': bound,
            'conclusion': 'Actual coupled ground mean only, not phase or HC'}


def native_controls(size, spacing, centre, UE, K, epsilon):
    band = RotationBand(**{**BARE, 'a': spacing})
    angles = np.asarray(centre)*math.pi/2
    selected, frame = original_frame(band, size, angles, True)
    gram = selected['G']
    law = BulkQuantumLaw(size=size, band=band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(angles)
    _, _, beta4 = chiral_basis(frame, gram, 1)
    beta = np.kron(np.eye(law.volume), beta4)
    S = negative_frame(beta4, law.volume)
    constants = sea_constants(frame, gram, band, size)
    phases = .31*np.sin(np.arange(3*law.volume).reshape(law.volume, 3)*.731+.27)
    h = law.onebody(fields, links, phases)
    data = polar_section(h, S)
    frame_assertions(data, h, S, beta, constants['mu'], constants['B'])
    density = np.asarray([np.trace(data['P'][4*x:4*x+4, 4*x:4*x+4]).real
                          for x in range(law.volume)])
    close(density, np.full(law.volume, 2.), 'actual local density equals background TWO')
    x, axis = law.volume//3, 1
    y = law.target[x, axis]
    sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
    F = -(band.r*frame[0]+1j*band.v*frame[axis+1])/(2*band.a)
    edge_hop = F@links[x, axis]*np.exp(1j*phases[x, axis])
    dh = np.zeros_like(h); dh[sx, sy] = 1j*edge_hop; dh[sy, sx] = -1j*edge_hop.conj().T
    derivative = section_derivative(data, S, dh)
    shifted = []
    for sign in (-1, 1):
        a = phases.copy(); a[x, axis] += sign*epsilon
        shifted.append(polar_section(law.onebody(fields, links, a), S))
    fd_h = (law.onebody(fields, links, phases+np.eye(law.volume*3)[3*x+axis].reshape(phases.shape)*epsilon)
            -law.onebody(fields, links, phases-np.eye(law.volume*3)[3*x+axis].reshape(phases.shape)*epsilon))/(2*epsilon)
    close(dh, fd_h, 'actual Peierls electric derivative, independently differenced', 2e-8)
    fd_Z = (shifted[1]['Z']-shifted[0]['Z'])/(2*epsilon)
    close(derivative['dZ'], fd_Z, 'independent global frame electric derivative', 2e-6)
    overlap = np.linalg.det(shifted[0]['Z'].conj().T@shifted[1]['Z'])
    fd_metric = (2-2*overlap.real)/(4*epsilon**2)
    close(fd_metric, derivative['metric'], 'full Slater electric metric from determinant overlap', 3e-4)
    if derivative['metric'] > constants['C_Q_axis'][axis]+3e-8:
        raise AssertionError('per-edge electric metric has no volume factor')
    chi = .29*np.cos(np.arange(law.volume)*.417)
    transformed = phases+chi[:, None]-chi[law.target]
    gauge_data = polar_section(law.onebody(fields, links, transformed), S)
    g = np.diag(np.repeat(np.exp(1j*chi), 4))
    gminus = np.diag(np.repeat(np.exp(1j*chi), 2))
    close(gauge_data['Z'], g@data['Z']@gminus.conj().T,
          'global smooth frame obeys every site U1 covariance', 2e-7)
    overlap_gauge = np.linalg.det(gauge_data['Z'].conj().T@g@data['Z'])
    close(overlap_gauge, np.exp(2j*chi.sum()), 'complete Slater background-two Gauss covariance', 2e-7)
    flux = []
    for j, k in itertools.combinations(range(3), 2):
        b = phases[:, j]+phases[law.target[:, j], k] \
            -phases[law.target[:, k], j]-phases[:, k]
        flux.extend(1-np.cos(b))
    zero = np.zeros_like(phases)
    flat = polar_section(law.onebody(fields, links, zero), S)
    alpha = np.asarray([.61, -.43, .29])
    toron_phases = np.tile(alpha/size, (law.volume, 1))
    toron = polar_section(law.onebody(fields, links, toron_phases), S)
    momenta = (2*math.pi*law.sites+angles+alpha)/(size*band.a)
    coeff = coefficients(band, momenta)
    independent_toron_sea = -2*float(np.sqrt(np.einsum('pi,ij,pj->p', coeff, gram, coeff)).sum())
    close(toron['sea'], independent_toron_sea, 'actual winding/toron sea energy versus independent native Fourier law')
    toron_difference = toron['sea']-flat['sea']
    if abs(toron_difference) < 1e-8:
        raise AssertionError('Selected toron adverse witness has unresolved sea response')
    if abs(data['sea']-flat['sea']) < 1e-8:
        raise AssertionError('Selected magnetic witness has unresolved sea response')
    sea_rhs = constants['C_B']*float(sum(flux))+law.volume*constants['R_N']
    if abs(data['sea']-flat['sea']) > sea_rhs+3e-8:
        raise AssertionError('actual sea magnetic bound with wrapping remainder retained')
    wrong_direction = polar_section(h, negative_frame(-beta4, law.volume))
    controls = [
        adverse('wrong beta-minus/P direction', lambda: frame_assertions(
            wrong_direction, h, negative_frame(-beta4, law.volume), beta,
            constants['mu'], constants['B'])),
        adverse('wrong beta sign', lambda: frame_assertions(
            data, h, S, -beta, constants['mu'], constants['B'])),
        adverse('omitted quantum metric', lambda: close(
            0., fd_metric, 'full Slater electric metric from determinant overlap', 3e-4)),
        adverse('dropped toron/wrapping response', lambda: close(
            flat['sea'], independent_toron_sea,
            'actual winding/toron sea energy versus independent native Fourier law', 1e-10)),
        adverse('frozen Slater sea energy', lambda: close(
            flat['sea'], np.trace(data['Z'].conj().T@h@data['Z']).real,
            'Slater expectation equals actual all-occupied sea energy', 1e-10))]
    return {'size': size, 'volume': law.volume, 'spacing': spacing, 'centre': centre,
            'centre_angles': angles, 'true_Gram': gram, 'Gram_selection': selected,
            'neutral_particles': 2*law.volume, 'native_onebody_dimension': len(h),
            'canonical_beta': beta4, 'fixed_beta_minus_site_frame': S[:4, :2],
            'C_min': float(data['C_values'][0]), 'C_lower': (1+constants['mu']/constants['B'])/2,
            'Slater_normalization_error': float(np.linalg.norm(data['Z'].conj().T@data['Z']-np.eye(2*law.volume))),
            'compact_phases': phases, 'site_gauge_chi': chi, 'Gauss_overlap': overlap_gauge,
            'all_site_density': density, 'actual_sea': data['sea'], 'flat_sea': flat['sea'],
            'toron_alpha': alpha, 'toron_sea': toron['sea'], 'toron_sea_difference': toron_difference,
            'independent_Fourier_toron_sea': independent_toron_sea,
            'magnetic_sum': float(sum(flux)),
            'sea_inequality_rhs': constants['C_B']*float(sum(flux))+law.volume*constants['R_N'],
            'edge': [int(x), int(axis), int(y)], 'edge_derivative_rank': int(np.linalg.matrix_rank(dh)),
            'edge_metric': derivative['metric'], 'finite_difference_metric': fd_metric,
            'edge_Berry_trace': derivative['berry'], 'epsilon': epsilon,
            'constants': constants, 'fixed_symbol_domain': mean_domain(constants, UE, K),
            'adverse': controls,
            'scope': 'N5 may be diagnostic only; full native static CAR and torons, not periodic quantum ground diagonalization'}


def loop_comparison(frame, gram, band, UE, K, cutoff, samples, epsilon):
    """Two sites/two parallel edges, actual endpoint coefficients, all70 NL=4.

    Geometry is a comparison, not the native N^3 lattice. Physical UE,K inputs
    are unchanged. Electric cutoff is a Galerkin diagnostic, never a law.
    """
    _, _, beta4 = chiral_basis(frame, gram, 1)
    S = negative_frame(beta4, 2)
    onsite = (band.m+3*band.r/band.a)*frame[0]
    F1 = -(band.r*frame[0]+1j*band.v*frame[1])/(2*band.a)
    F2 = -(band.r*frame[0]+1j*band.v*frame[2])/(2*band.a)
    def onebody(phi):
        hop = F1*np.exp(1j*phi)+F2
        return np.block([[onsite, hop], [hop.conj().T, onsite]])
    matter0, basis = exterior_hamiltonian(np.block([[onsite, F2], [F2.conj().T, onsite]]), 4)
    directed_h = np.zeros((8, 8), complex); directed_h[:4, 4:] = F1
    directed, _ = exterior_hamiltonian(directed_h+directed_h.conj().T, 4)
    # Split the Hermitian exterior lift into its actual left-creating part.
    charges = np.asarray([(state & 15).bit_count()-2 for state in basis])
    directed = np.where(charges[:, None]-charges[None, :] == 1, directed, 0.)
    dim = len(basis)
    def quantum_matrix(L):
        rows, cols, vals = [], [], []
        def block(row_l, col_l, matrix):
            i, j = np.nonzero(matrix)
            rows.extend((row_l*dim+i).tolist()); cols.extend((col_l*dim+j).tolist())
            vals.extend(matrix[i, j].tolist())
        for index, ell in enumerate(range(-L, L+1)):
            block(index, index, matter0+np.diag(UE*(ell**2+(charges-ell)**2)+K))
            if index < 2*L:
                block(index+1, index, directed-K*np.eye(dim)/2)
                block(index, index+1, directed.conj().T-K*np.eye(dim)/2)
        matrix = coo_matrix((vals, (rows, cols)), shape=((2*L+1)*dim,)*2).tocsr()
        close((matrix-matrix.conj().T).data, np.zeros_like((matrix-matrix.conj().T).data),
              'Gauss-eliminated coupled full-CAR loop Hamiltonian Hermiticity')
        return matrix
    grid = np.arange(samples)*2*math.pi/samples
    gtheta = math.sqrt(UE/(2*K))
    modes = np.arange(-max(cutoff+12, 24), max(cutoff+12, 24)+1)
    amplitudes = np.exp(-gtheta*modes**2/2)
    psi = (np.exp(1j*grid[:, None]*modes)@amplitudes).real/math.sqrt(float(np.dot(amplitudes, amplitudes)))
    dpsi = (np.exp(1j*grid[:, None]*modes)@(1j*modes*amplitudes)).real/math.sqrt(float(np.dot(amplitudes, amplitudes)))
    omegas, derivatives, seas = [], [], []
    for phi in grid:
        data = polar_section(onebody(phi), S)
        omega = slater_vector(data['Z'], basis)
        plus = slater_vector(polar_section(onebody(phi+epsilon), S)['Z'], basis)
        minus = slater_vector(polar_section(onebody(phi-epsilon), S)['Z'], basis)
        omegas.append(omega); derivatives.append((plus-minus)/(2*epsilon)); seas.append(data['sea'])
    omega = np.asarray(omegas); domega = np.asarray(derivatives); sea = np.asarray(seas)
    close(np.sum(abs(omega)**2, axis=1), np.ones(samples), 'entire70CAR Slater normalization')
    close(np.einsum('si,si->s', omega.conj(), domega), np.zeros(samples),
          'complete Slater Berry trace on loop slice', 2e-6)
    state = psi[:, None]*omega
    derivative_state = dpsi[:, None]*omega+psi[:, None]*domega
    edge2_state = -derivative_state+1j*charges[None, :]*state
    electric = UE*float(np.mean(np.sum(abs(derivative_state)**2+abs(edge2_state)**2, axis=1)))
    scalar_electric = 2*UE*float(np.mean(dpsi**2))
    metric_cost = electric-scalar_electric
    matter = float(np.mean(psi**2*sea))
    magnetic = K*float(np.mean(psi**2*(1-np.cos(grid))))
    norm = float(np.mean(psi**2))
    close(norm, 1., 'real periodic coupled trial normalization', 2e-10)
    physical_energy = (electric+matter+magnetic)/norm
    Fourier = np.fft.fft(state, axis=0)/samples
    indices = np.arange(-cutoff, cutoff+1)
    retained = Fourier[indices % samples].reshape(-1)
    retained_norm = float(np.vdot(retained, retained).real)
    H = quantum_matrix(cutoff)
    galerkin_energy = float(np.vdot(retained, H@retained).real/retained_norm)
    eigenvalues, eigenvectors = cast(tuple[np.ndarray, np.ndarray], eigsh(
        H, k=1, which='SA', tol=0, v0=np.ones((2*cutoff+1)*dim, dtype=complex)))
    ground = eigenvectors[:, 0]
    ground_energy = float(eigenvalues[0])
    close(H@ground, ground_energy*ground, 'actual coupled quantum ground eigen-residual', 2e-8)
    shaped = ground.reshape(2*cutoff+1, dim)
    phi_cos = float(np.vdot(shaped[1:], shaped[:-1]).real)
    if ground_energy > galerkin_energy+2e-8:
        raise AssertionError('actual coupled Galerkin ground lies below its actual coupled trial')
    error = abs(galerkin_energy-physical_energy)
    # Resolution is an explicit assertion, not a claimed infinite-electric run.
    if 1-retained_norm > 2e-6 or error > 2e-4*max(1., abs(physical_energy)):
        raise AssertionError('Increase declared loop electric cutoff/samples: coupled trial unresolved')
    accounting_tolerance = max(2e-8, 4*error/max(1., abs(physical_energy)))
    close(galerkin_energy, physical_energy,
          'actual coupled electric+sea+cosine energy accounting', accounting_tolerance)
    resolution = accounting_tolerance*max(1., abs(physical_energy), abs(galerkin_energy))
    if metric_cost <= 3*resolution:
        raise AssertionError('Quantum-metric adverse witness is below comparison resolution')
    frozen = float(np.mean(psi**2))*float(polar_section(onebody(0.), S)['sea'])
    if abs(frozen-matter) <= 3*resolution:
        raise AssertionError('Frozen-sea adverse witness is below comparison resolution')
    wrong = [adverse('omitted electric quantum metric', lambda: close(
                 scalar_electric+matter+magnetic, galerkin_energy,
                 'actual coupled electric+sea+cosine energy accounting', accounting_tolerance)),
             adverse('sea energy frozen at flat links', lambda: close(
                 electric+frozen+magnetic, galerkin_energy,
                 'actual coupled electric+sea+cosine energy accounting', accounting_tolerance))]
    # A second cutoff observes convergence without using it as a proof of the infinite rotor.
    higher = quantum_matrix(cutoff+2)
    hi_values = cast(np.ndarray, eigsh(higher, k=1, which='SA', tol=0,
                     v0=np.ones((2*cutoff+5)*dim, dtype=complex), return_eigenvectors=False))
    hi_ground = float(hi_values[0])
    return {'geometry': 'two sites, two parallel native-coefficient edges, NOT cubic native admission',
            'UE': UE, 'K': K, 'onsite': onsite, 'F1': F1, 'F2': F2,
            'all_sector_states': basis, 'particles': 4, 'CAR_dimension': dim,
            'Gauss_site_left_background_two_charge': charges,
            'Gauss_electric_integers': 'p1=ell, p2=n_left-2-ell; p1+p2=n_left-2',
            'electric_cutoff': cutoff, 'larger_cutoff': cutoff+2, 'samples': samples,
            'theta_g': gtheta, 'theta_fourier_modes': modes, 'theta_amplitudes': amplitudes,
            'trial_norm': norm, 'retained_trial_norm': retained_norm,
            'matter_energy': matter, 'frozen_flat_matter': frozen,
            'actual_electric_energy': electric, 'photon_scalar_electric_energy': scalar_electric,
            'quantum_metric_cost': metric_cost, 'actual_cosine_energy': magnetic,
            'coupled_trial_energy': physical_energy, 'Galerkin_trial_energy': galerkin_energy,
            'coupled_Galerkin_ground_energy': ground_energy,
            'larger_Galerkin_ground_energy': hi_ground,
            'coupled_Galerkin_ground_mean_flux': 1-phi_cos,
            'ground_residual': float(np.linalg.norm(H@ground-ground_energy*ground)),
            'adverse': wrong, 'infinite_electric_ground_certified': False}


def run(size=5, spacing=1., centre=(1, 0, 2), *, UE, K,
        epsilon=3e-4, cutoff=14, samples=128, all_centres=False, loop=True):
    if size < 5 or spacing <= 0 or UE <= 0 or K <= 0 or epsilon <= 0:
        raise ValueError('Need N>=5, a>0, explicit fixed UE,K>0 and epsilon>0')
    if cutoff < 2 or samples < 4*(cutoff+3):
        raise ValueError('Need resolved loop Fourier sampling, at least 4*(cutoff+3) samples')
    centres = list(itertools.product(range(4), repeat=3)) if all_centres else [tuple(centre)]
    native = [native_controls(size, spacing, item, UE, K, epsilon) for item in centres]
    result: dict[str, Any] = {'plan': PLAN, 'plan_revision': PLAN_REVISION,
        'template_revision': TEMPLATE_REVISION, 'inputs': {'size': size, 'spacing': spacing,
        'centres': centres, 'UE': UE, 'K': K, 'epsilon': epsilon, 'cutoff': cutoff,
        'samples': samples, 'loop': loop, 'all_centres': all_centres}, 'native': native,
        'retained_hypotheses': {'full_original_fields_and_SU4_Gauss': 'via source BO, not this static diagonalization',
        'all64_offsets_weights': 'not replaced; max constants/min baseline proof, no fabricated numerical weights',
        'N0': 'existence only; N5 diagnostics do not prove native admission',
        'physical_cosine': True, 'actual_compact_U1_Gauss_background': 2,
        'no_photon_phase_or_HC_claim': True},
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ('original_neutral_photon_trial.py', 'original_neutral_photon_trial.md',
                          'original_uniform_charge_gap.md', 'original_charged_principal.py',
                          'bulk_quantum_selection.py', 'original_slow_kinetic_bo.md')}}
    if all_centres:
        c = {'C_B': max(row['constants']['C_B'] for row in native),
             'R_N': max(row['constants']['R_N'] for row in native),
             'C_Q': max(row['constants']['C_Q'] for row in native)}
        result['all64_uniform_mean_domain'] = mean_domain(c, UE, K)
    if loop:
        band = RotationBand(**{**BARE, 'a': spacing})
        selected, frame = original_frame(band, size, np.asarray(centre)*math.pi/2, True)
        result['coupled_quantum_comparison'] = loop_comparison(
            frame, selected['G'], band, UE, K, cutoff, samples, epsilon/10)
    # Every observation/acceptance/adverse case is computed before serialization.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--centre', type=int, nargs=3, default=(1, 0, 2), choices=range(4))
    parser.add_argument('--UE', type=float, required=True)
    parser.add_argument('--K', type=float, required=True)
    parser.add_argument('--epsilon', type=float, default=3e-4)
    parser.add_argument('--cutoff', type=int, default=14)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--all-centres', action='store_true')
    parser.add_argument('--no-loop', action='store_true')
    args = parser.parse_args()
    result = run(args.size, args.spacing, tuple(args.centre), UE=args.UE, K=args.K,
                 epsilon=args.epsilon, cutoff=args.cutoff, samples=args.samples,
                 all_centres=args.all_centres, loop=not args.no_loop)
    print(json.dumps(portable(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
