#!/usr/bin/env python3
"""Actual N5 full10 common-m supplier and formal UV critical branch, not a cutover."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
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
from scipy.optimize import brentq
from scipy.linalg import null_space
from angular_photon_spectrum import brillouin_quadrature
from bulk_frame_compton import raw_h, reconstruct_frame
from local_orbital_stability import periodic_grid
from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import OriginalSpatial, original_frame, portable
from original_quantum_mass_diagnostic import bz_diagnostic, source_hashes as mass_source_hashes
from original_slow_kinetic_bo import jet_controls, normal_directions, source_hashes as bo_source_hashes
from rotation_band import RotationBand
from sea_selection import coefficients
from unrestricted_field_minimum import BARE, STIFFNESS, coordinates


def accept(error, label, tolerance):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')
    return float(error)


def native_supplier(t, a, cache):
    if t == 0:
        raise ValueError('Exact t=0 provider has a 0/0 zero-ray derivative; use explicit one-sided limits, not a fake frame')
    if t not in cache:
        bare = RotationBand(**{**BARE, 'a': a, 'm': t/a}, e=1.)
        selected, frame = original_frame(bare, 5, np.zeros(3), True)
        G = np.asarray(selected['G'])
        accept(float(selected['gradient_residual']), 'all ten native Gram gradients', 3e-12)
        accept(float(np.linalg.norm(G-np.diag(np.diag(G)))), 'actual cubic Gram symmetry', 3e-12)
        accept(float(np.ptp(np.diag(G)[1:])), 'actual spatial Gram equality', 3e-12)
        assert np.linalg.eigvalsh(G).min() > 0 and selected['full10_Gram']
        cache[t] = (bare, selected, frame)
    return cache[t]


def ultraviolet_coefficient(G, photon_d, quadrature) -> dict[str, Any]:
    q, weights = quadrature
    b0 = float(np.sqrt(G[0, 0]))
    r, c = BARE['r']*b0, BARE['v']*float(np.sqrt(G[1, 1]))
    qhat = 2*np.sin(q/2)
    Q2 = np.sum(qhat*qhat, axis=1)
    assert np.all(Q2 > 0)  # quadrature nodes only; native harmonic rotors are not removed
    Q = np.sqrt(Q2)
    M0 = r*np.sum(1-np.cos(q), axis=1)
    E = np.sqrt(M0*M0+c*c*np.sum(np.sin(q)**2, axis=1))
    T = np.sum(np.cos(q/2)**2*(1-qhat*qhat/Q2[:, None]), axis=1)
    terms = dict(contact=float(weights@(r/(2*Q))),
                 virtual=float(weights@(c*c*M0*T/(2*Q*E*(E+photon_d*Q)))),
                 Coulomb=float(weights@(photon_d*M0/(2*E*Q2))))
    return dict(C=sum(terms.values()), terms=terms, b0=b0, r=r, electron_c=c,
                photon_d=photon_d, reference_self=float(weights@(photon_d/(2*Q2))))


def winding_degree(t):
    values = [t+2*BARE['r']*n for n in range(4)]
    assert all(x != 0 for x in values)
    return int(round(sum(math.comb(3, n)*(-1)**n*np.sign(x)
                         for n, x in enumerate(values))/2))


def cusp_check(a, cache):
    eps = 2e-6
    frames = {t: native_supplier(t, a, cache)[1]['G']
              for t in (-2*eps, -eps, -eps/2, eps/2, eps, 2*eps)}
    Gmid = (frames[eps]+frames[-eps])/2
    Ghalf = (frames[eps/2]+frames[-eps/2])/2
    G0_est = 2*Ghalf-Gmid  # explicit O(eps^2) extrapolation; not exact zero-node provider
    q = 2*np.pi*periodic_grid(5)/5
    zero_bare = RotationBand(**{**BARE, 'a': 1., 'm': 0.}, e=1.)
    x = coefficients(zero_bare, q)
    basis = [np.diag(np.eye(4)[i]) for i in range(4)]
    for i, j in itertools.combinations(range(4), 2):
        unit = np.zeros((4, 4)); unit[i, j] = unit[j, i] = 1/np.sqrt(2)
        basis.append(unit)
    basis = np.asarray(basis)
    design = np.einsum('pi,aij,pj->pa', x, basis, x)
    radii = np.sqrt(np.einsum('pi,ij,pj->p', x, G0_est, x))
    zero_ray = np.all(x == 0, axis=1)
    assert len(x) == 125 and zero_ray.sum() == 1
    # At the exact x=0 ray its G objective/gradient/Hessian contribution is exactly zero.
    # All 125 rays remain in the normalization; no heavy/light mode or occupation is removed.
    inverse_cube = np.divide(1., radii**3, out=np.zeros_like(radii), where=~zero_ray)
    hessian = STIFFNESS*np.eye(10)+.5*np.einsum('pa,pb,p->ab', design, design, inverse_cube)/len(x)
    rhs = np.zeros(10); rhs[0] = 2/(125*np.sqrt(G0_est[0, 0]))
    predicted = np.einsum('a,aij->ij', np.linalg.solve(hessian, rhs), basis)
    right = (frames[2*eps]-frames[eps])/eps
    left = (frames[-eps]-frames[-2*eps])/eps
    error = accept(float(np.linalg.norm(right-left-predicted)), 'whole10 q0 cusp jump', 2e-5)
    assert np.linalg.norm(predicted) > .005
    return dict(epsilon=eps, all_source_rays=125, exact_zero_rays=1,
                G0_estimate=G0_est.tolist(), estimate_scope='one-sided extrapolation, not native provider evaluation at t=0',
                full10_hessian_minimum=float(np.linalg.eigvalsh(hessian).min()),
                measured_derivative_jump=(right-left).tolist(), predicted_derivative_jump=predicted.tolist(),
                jump_error=error, omitted_zero_ray_cusp_rejected=bool(np.linalg.norm(right-left) > .005))


def native_source_hashes():
    return mass_source_hashes() | bo_source_hashes() | {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def signed_heavy_controls(t, a, cache):
    """Actual whole500/250 negative-heavy source and full normal/contact jets."""
    assert -.0047 <= t <= -.0045, 'selected signed BO supplier interval'
    bare = native_supplier(t, a, cache)[0]
    law = BulkQuantumLaw(5, band=bare)
    rng = np.random.default_rng(2221203)
    rows = []
    for centre in ((0, 0, 0), (1, 0, 2)):
        angles = np.pi/2*np.asarray(centre)
        selected, frame = original_frame(bare, 5, angles, True)
        accept(float(selected['gradient_residual']), 'signed all-ten Gram stationarity', 3e-12)
        model = OriginalSpatial(bare, 5, law.basis, coordinates(frame, law.basis).reshape(-1), angles)
        accept(a*float(np.linalg.norm(model.gradient)), 'signed full-field/link native stationarity', 2e-8)
        soft = null_space(model.D.T, rcond=1e-12)
        assert soft.shape == (60, 45)
        uniform = model.hessian_parts(np.zeros(3))
        h0 = uniform['local']+uniform['sea']+uniform['contact']
        physical = soft.T@h0[:60, :60]@soft
        alpha = float(np.linalg.eigvalsh((physical+physical.conj().T)/2)[0])
        assert alpha > 0, 'actual signed45 physical uniform normals must restore'
        fields = np.tile(frame, (law.volume, 1, 1, 1))
        links = law.toron_links(angles)
        direction = normal_directions(model, law, rng)
        jets, heavy = jet_controls(model, law, fields, links, direction)
        gap_lower = 2*abs(bare.m)*np.sqrt(np.linalg.eigvalsh(selected['G'])[0])
        accept(max(0., float(gap_lower-heavy['gap'])), 'signed all-shift Bloch gap floor', 2e-8)
        rows.append(dict(centre=centre, all10_Gram=np.asarray(selected['G']).tolist(),
                         onebody_modes=4*law.volume, whole_NH=2*law.volume,
                         native_uniform45_minimum=alpha, analytic_Bloch_gap_floor=float(gap_lower),
                         actual_whole_heavy_gap=heavy['gap'], full_native_jet=jets,
                         complete_nonzero_q_normal_dimension=90,
                         scope='finiteN5 actual whole-heavy/source jet; not large-N globality, quantum normal vacuum or BO onset certificate'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'original_common_mass_diagnostic_receipt.json')
    args = parser.parse_args()
    before = native_source_hashes()
    a, g2, tref = .001, .09, .001*BARE['m']
    cache = {}
    bare, selected, frame = native_supplier(tref, a, cache)
    band, unitary, _ = reconstruct_frame(bare, frame)
    d_fixed = band.v
    orders = []
    for order in (6, 8):
        q, weights = brillouin_quadrature(order, 'split_gauss')
        assert weights is not None
        accept(abs(float(weights.sum())-1), 'normalized BZ measure', 1e-12)
        quadrature = (q, weights)
        original = ultraviolet_coefficient(selected['G'], d_fixed, quadrature)
        independent = bz_diagnostic(bare, band, frame, unitary, order)
        control = accept(abs(original['C']-independent['UV_frozen_frame_C']), 'positive native independent projector UV coefficient', 2e-12)
        def psi(t, retune=False):
            G = native_supplier(t, a, cache)[1]['G']
            d = BARE['v']*np.sqrt(G[1, 1]) if retune else d_fixed
            value = ultraviolet_coefficient(G, d, quadrature)
            return t*value['b0']+g2*value['C']
        root_solution = brentq(psi, -.08, -1e-7, xtol=1e-13, full_output=True)
        assert isinstance(root_solution, tuple) and root_solution[1].converged
        root = float(root_solution[0])
        bare_root, root_selected, root_frame = native_supplier(root, a, cache)
        value = ultraviolet_coefficient(root_selected['G'], d_fixed, quadrature)
        rest_h = raw_h(bare_root, root_frame, np.zeros(3))
        rest_values, rest_vectors = np.linalg.eigh(rest_h)
        positive_rest = rest_vectors[:, rest_values > 0]
        assert positive_rest.shape == (4, 2)
        heavy_rest_projector = positive_rest@positive_rest.conj().T
        beta0 = root_frame[0]/value['b0']
        negative_branch_projector = (np.eye(4)-beta0)/2
        rest_error = accept(float(np.linalg.norm(heavy_rest_projector-negative_branch_projector)),
                            'actual negative-bare heavy rest projector', 2e-10)
        wrong_positive_rest_error = float(np.linalg.norm(heavy_rest_projector-(np.eye(4)+beta0)/2))
        assert wrong_positive_rest_error > 1.
        accept(abs(psi(root)), 'formal common-root residual', 1e-12)
        frozen_root = -g2*original['C']/original['b0']
        retuned_solution = brentq(lambda t: psi(t, True), -.08, -1e-7, xtol=1e-13, full_output=True)
        assert isinstance(retuned_solution, tuple) and retuned_solution[1].converged
        retuned_root = float(retuned_solution[0])
        eps = 1e-6
        slope = (psi(root+eps)-psi(root-eps))/(2*eps)
        frozen_slope = value['b0']
        assert -.24 < root < 0 and winding_degree(root) == -1
        assert winding_degree(tref) == 0
        orders.append(dict(order=order, points=len(q), independent_projector_coefficient_error=control,
            root_t=root, common_bare_m=root/a, formal_positive_reference_coefficient=value,
            root_gradient=float(root_selected['gradient_residual']), native_Gram=root_selected['G'].tolist(),
            original_photon=dict(d=d_fixed, g2=g2, ue=d_fixed*g2/(2*a), K=d_fixed/(g2*a)),
            wrong_frozen_G_root=frozen_root, frozen_G_root_difference=root-frozen_root,
            different_retuned_photon_law_root=retuned_root, retuned_law_root_difference=retuned_root-root,
            common_source_slope=slope, wrong_frozen_source_slope=frozen_slope,
            bare_band_degree=winding_degree(root), original_bare_band_degree=winding_degree(tref),
            bare_rest_doublet_reversed=wrong_positive_rest_error > 1.,
            actual_bare_rest_spectrum=rest_values.tolist(),
            negative_bare_heavy_rest_projector_error=rest_error,
            wrong_positive_branch_heavy_rest_projector_error=wrong_positive_rest_error,
            target_split_is_not_negative_bare_spectrum=True))
    signed_native = signed_heavy_controls(orders[-1]['root_t'], a, cache)
    receipt: dict[str, Any] = dict(plan='issue-222-continuing-backward-20260930', assignment_revision='ae3462dd2e6af3ce4e30b03a879237c98cec4e531c209079dfb62a43af7bfd89',
        source_hashes=before, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, executable=sys.executable),
        field_N=5, all10=True, field_source_rays=125, orders=orders, cusp=cusp_check(a, cache), all_checks=True,
        actual_signed_heavy_controls=signed_native,
        signed_supplier_assignment_revision='e3199cae4be5095e35679118d501706f4349120f8f8fb3d37f733ad5cec6d67f',
        signed_supplier_proof_sha256='384ac5a935615d7b266d3998fb112dfa07b9922ae270a1170091f7f928f8d501',
        scope='actual full10 heavy Gram, finiteN5 whole500/250 signed-heavy source/normal/contact jets and formal one-loop common critical branch; original bare law unchanged; no N5 globality, negative-bare beta positivity, interacting mass, phase, UV or total spin claim')
    assert before == native_source_hashes(), 'native source changed during common-root check'
    args.output.write_text(json.dumps(portable(receipt), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks=True, roots=[row['root_t'] for row in orders],
                         degrees=[row['bare_band_degree'] for row in orders], cusp_error=receipt['cusp']['jump_error'])))


if __name__ == '__main__':
    main()
