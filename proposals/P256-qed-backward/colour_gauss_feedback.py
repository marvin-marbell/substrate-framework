#!/usr/bin/env python3
"""Actual primitive local colour moments; mean neutrality is not a Gauss state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np

from heavy_orbital_join import selected_saddle
from orbital_gauge_locking import gauge_mass
from quantum_adjoint_mass import primitive_code


def local_density(beta_lift, b, sign):
    """Four-mode Gaussian density from actual bosonic-code even generators."""
    if not 0 <= b < 1 or sign not in (-1, 1):
        raise ValueError('Require 0<=b<1 and a whole-H sign +/-1')
    values, vectors = np.linalg.eigh(beta_lift)
    logarithms = -2 * np.arctanh(sign * b) * values
    weights = np.exp(logarithms - np.max(logarithms))
    weights /= np.sum(weights)
    return (vectors * weights) @ vectors.conj().T


def primitive_moments(rho, charges, casimir, sectors):
    mean = np.einsum('ij,aji->a', rho, charges).real
    second = float(np.trace(rho @ casimir).real)
    pmf = np.array([np.trace(rho[np.ix_(sectors[n], sectors[n])]).real for n in range(5)])
    return mean, second, pmf


def run(orders, scales, light_ma, heavy_ma, stiffness):
    generators, _, q, _, sectors, primitive = primitive_code()
    charges = np.zeros((15, 16, 16), complex)
    for number, indices in sectors.items():
        for a in range(15):
            charges[a][np.ix_(indices, indices)] = q[number][a]
    casimir = sum(t @ t for t in charges)
    # Both species' colour operators are even. Their tensor product needs no
    # assumed inter-species ladder/CAR construction and makes no such claim.
    pair_charges = [np.kron(t, np.eye(6)) + np.kron(np.eye(6), t) for t in q[2]]
    pair_casimir = sum(t @ t for t in pair_charges)
    pair_values, pair_vectors = np.linalg.eigh(pair_casimir)
    singlets = pair_vectors[:, abs(pair_values) < 1e-10]
    dimensions = {str(c): int(np.sum(abs(pair_values - c) < 1e-10)) for c in (0, 4, 6)}
    if dimensions != {'0': 1, '4': 15, '6': 20}:
        raise AssertionError('Actual primitive two-plus-two Casimir decomposition failed')
    basis = np.sqrt(8) * generators
    rows = []
    for order in orders:
        for scale in scales:
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Require finite positive Lambda')
            saddles = {}
            for label, sign, mass in (('same_sign_unequal', 1, heavy_ma),
                                      ('opposite_sign_unequal', -1, heavy_ma),
                                      ('balanced_opposite_identical', -1, light_ma)):
                if mass not in saddles:
                    saddles[mass] = selected_saddle(order, light_ma, scale, stiffness, heavy_ma=mass)
                _, frame, saddle = saddles[mass]
                gram = np.array(saddle['selected_gram'])
                beta = frame[0] / np.sqrt(gram[0, 0])
                beta_coefficients = 2 * np.einsum('aij,ji->a', generators, beta).real
                beta_lift = np.einsum('a,aij->ij', beta_coefficients, charges)
                population = saddle['unprojected_local_centre_population']
                bh = population['species']['heavy']['local_density_projector_beta_average']
                bl = population['species']['light']['local_density_projector_beta_average']
                rh = local_density(beta_lift, bh, sign)
                rl = local_density(beta_lift, bl, 1)
                qh, ch, ph = primitive_moments(rh, charges, casimir, sectors)
                ql, cl, pl = primitive_moments(rl, charges, casimir, sectors)
                mean = qh + ql
                second = ch + cl + 2 * float(qh @ ql)
                closed_mean = -(sign * bh + bl) / 2 * np.einsum('aij,ji->a', generators, beta).real
                closed_second = .5 * (sign * bh + bl)**2 + 4 - (2 - bh**2 - bl**2) / 8
                pmf = np.convolve(ph, pl)
                centre = [float(np.sum(pmf[np.arange(9) % 4 == residue])) for residue in range(4)]
                n2 = sectors[2]
                r2 = np.kron(rh[np.ix_(n2, n2)] / ph[2], rl[np.ix_(n2, n2)] / pl[2])
                singlet_probability = float(np.trace(singlets.conj().T @ r2 @ singlets).real)
                mass_matrix, changes = gauge_mass(frame, generators, 1., 1.)
                # Phi_mu=sum_b u_mu,b B_b, Tr(B_b B_c)=4 delta_bc.
                # u and Pi_u are canonical; this is not the HS-normalized convention.
                tangent = np.einsum('bij,amji->mba', basis, 1j * changes).real.reshape(60, 15) / 4
                metric = tangent.T @ tangent
                multiplier = np.linalg.solve(metric, mean)
                momentum = -tangent @ multiplier
                coefficient = float(momentum @ momentum / 2)
                mass = scale**-.5
                residuals = dict(
                    beta_squared=float(np.linalg.norm(beta @ beta - np.eye(4))),
                    mean_formula=float(np.linalg.norm(mean - closed_mean)),
                    second_formula=abs(second - closed_second),
                    centre_population=float(np.linalg.norm(np.array(centre) - population['local_centre_probabilities'])),
                    canonical_metric=float(np.linalg.norm(metric - mass_matrix.real)),
                    classical_constraint=float(np.linalg.norm(tangent.T @ momentum + mean)))
                if max(residuals.values()) > 1e-9:
                    raise AssertionError(f'Primitive or canonical formula failure: {residuals}')
                rows.append(dict(profile=label, order=order, Lambda=scale, heavy_whole_H_sign=sign,
                                 light_ma=light_ma, heavy_ma=saddle['bare_heavy_ma'],
                                 heavy_beta_average=bh, light_beta_average=bl,
                                 mean_colour_squared=float(mean @ mean),
                                 matter_total_Casimir_expectation=second,
                                 matter_colour_variance=second - float(mean @ mean),
                                 local_centre_probabilities=centre,
                                 conditional_two_plus_two_matter_singlet_probability=singlet_probability,
                                 classical_mean_Gauss=dict(field_mass=mass, momentum_norm_squared=float(momentum @ momentum),
                                     coefficient_of_sqrt_Lambda=coefficient, minimum_kinetic_cost=coefficient / mass,
                                     metric_minimum_eigenvalue=float(np.linalg.eigvalsh(metric)[0])),
                                 residuals=residuals))
    here = Path(__file__).resolve().parent
    return dict(status='primitive_unprojected_local_colour_and_classical_mean_diagnostic',
                primitive=primitive, conditional_two_plus_two_Casimir_dimensions=dimensions, rows=rows,
                source_sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in
                               ('colour_gauss_feedback.py', 'heavy_orbital_join.py', 'quantum_adjoint_mass.py',
                                'orbital_gauge_locking.py')},
                hypotheses=['Frozen thermodynamic Slater seas at a common preferred-zero quartic saddle; no local Gauss projection.',
                            'Balanced whole-H polarity and identical bare coefficients are explicit hypotheses, not dynamically selected.',
                            'Classical field coordinates Phi_mu=sum u_mu,b B_b, Tr B_b B_c=4 delta; kinetic Pi_u^2/(2M), M=Lambda^-1/2.'],
                limitations=['Mean colour zero is not zero local matter Casimir and is not a quantum Gauss state.',
                             'Matter singlet probability is conditional on exactly two particles of each species; full Gauss permits field/link dressing.',
                             'Minimum mean-cancelling classical momentum cost is NOT a quantum ground-energy bound: Phi/Pi correlations and dressing remain.',
                             'No charged-gap conclusion follows; both polarities have positive finite Ritz differences in the separate continuous-SU4 one-link consumer.',
                             'No three-dimensional phase or mechanical spin protocol is proved.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[24, 32])
    parser.add_argument('--lambdas', type=float, nargs='+', default=[1., 8., 64.])
    parser.add_argument('--light-ma', type=float, default=.2)
    parser.add_argument('--heavy-ma', type=float, default=.8)
    parser.add_argument('--stiffness', type=float, default=1.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.orders, args.lambdas, args.light_ma, args.heavy_ma, args.stiffness)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
