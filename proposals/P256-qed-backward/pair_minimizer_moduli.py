#!/usr/bin/env python3
"""Exact pair-minimum classification and full native principal-form diagnostics.

The spectrum is a local quadratic calculation, not an exact quantum ground.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import block_diag, expm, null_space
from scipy.optimize import OptimizeResult, minimize_scalar

from full_band_gauss_section import FullBandGaussSection
from full_band_packet_bound import packet_point_controls
from pair_global_field_bound import coefficients, point, source_hashes as bound_hashes
from pair_separated_charge_transfer import full_field_angular_control
from rotation_band import RotationBand
from unrestricted_field_minimum import (ALIGNMENT, BARE, STIFFNESS, Functional,
    algebra_report, check, coordinates, derivative_controls, local_potential)

ROOT = Path(__file__).resolve().parent
RESOLUTION = 2e-7


def source_hashes():
    names = ('pair_minimizer_moduli.py', 'pair_minimizer_moduli.md',
             'pair_separated_charge_transfer.py', 'pair_separated_charge_transfer.md')
    return {**bound_hashes(), **{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in names}}


def canonical_fields(model, fraction):
    """One representative of EVERY minimum orbit for the executed 0<q<1/2 law."""
    c, _, t = coefficients(model)
    radius = ((c**4 + t**4) / STIFFNESS)**(1/3)
    q = t**4 / (c**4 + t**4)
    if not (0 < q < .5 and 0 <= fraction <= 1):
        raise ValueError('This classification chart requires 0<q<1/2 and fraction in [0,1]')
    delta = q * fraction
    aa, yy = np.sqrt([1-q+delta, 1-q-delta]), np.sqrt([q-delta, q+delta])
    A = np.diag([aa[0], -aa[0], aa[1], -aa[1]])
    Y = np.zeros((4, 4)); Y[0, 1] = Y[1, 0] = yy[0]; Y[2, 3] = Y[3, 2] = yy[1]
    frame = np.zeros((4, 4, 4), dtype=complex)
    frame[0], frame[1] = radius/c*A, radius/t*Y
    return np.tile(frame, (2, 1, 1, 1)), radius, q


def orbit_classifier(model, fields, link):
    """Recover the singular-value orbit invariant without choosing Clifford axes."""
    c, _, t = coefficients(model)
    radius = ((c**4 + t**4) / STIFFNESS)**(1/3)
    q = t**4 / (c**4+t**4)
    fixed = fields.copy(); fixed[1] = link @ fields[1] @ link.conj().T
    check(np.linalg.norm(fixed[0]-fixed[1]), 'minimum alignment classification')
    A, Y = c/radius*fixed[0, 0], t/radius*fixed[0, 1]
    check(np.linalg.norm(A@A+Y@Y-np.eye(4)), 'minimum involution classification')
    check(np.linalg.norm(A@Y+Y@A), 'minimum anticommutation classification')
    values, vectors = np.linalg.eigh(A)
    if np.sum(values > 0) != 2 or np.min(abs(values)) < 1e-3:
        raise AssertionError('Actual small-q positive/negative 2-plane chart failed')
    plus, minus = vectors[:, 2:], vectors[:, :2]
    S = plus.conj().T @ Y @ minus
    check(np.linalg.norm(plus.conj().T@Y@plus)+np.linalg.norm(minus.conj().T@Y@minus),
          'Y has only offdiagonal positive-negative blocks')
    squared = np.sort(np.linalg.svd(S, compute_uv=False)**2)
    check(abs(float(squared.sum())-2*q), 'complete orbit singular-value constraint')
    delta = float((squared[1]-squared[0])/2)
    return dict(delta=delta, fraction=delta/q, squared_singular_values=squared.tolist(),
                A_eigenvalues=values.tolist(),
                square_Clifford_defect=float(np.linalg.norm(A@A-np.trace(A@A)/4*np.eye(4))))


def full_tangents(model, fields):
    """135 native coordinates: Cartesian u (HS/4) and exp(i theta_a T_a) link."""
    D = np.array([coordinates(1j*(t@fields[1]-fields[1]@t), model.basis).reshape(-1)
                  for t in model.generators]).T
    pullback = np.column_stack((np.eye(120), np.vstack((np.zeros((60, 15)), D))))
    gauges = np.zeros((135, 30))
    gauges[:60, :15], gauges[60:120, 15:] = D, D
    gauges[120:, :15], gauges[120:, 15:] = np.eye(15), -np.eye(15)
    return pullback, gauges, D


def positive_spectrum(matrix):
    values = np.linalg.eigvalsh((matrix+matrix.T)/2)
    check(max(0., -float(values[0])), 'positive principal quadratic form', RESOLUTION)
    return values[values > RESOLUTION], values


def principal_spectrum(hessian, pullback, gauges, D, *, Lambda, ue4):
    """All positive oscillator modes survive the full local Gauss quotient.

    Divide frequencies by Lambda**(3/4); the remaining link cometric is eta.
    Covariant matter/Berry derivatives are lower differential order, not removed.
    """
    eta = 2*ue4/np.sqrt(Lambda)
    root_cometric = np.r_[np.ones(120), np.full(15, np.sqrt(eta))]
    full_hessian = pullback.T @ hessian @ pullback
    principal = root_cometric[:, None]*full_hessian*root_cometric[None, :]
    positive, all_values = positive_spectrum(principal)
    vertical = gauges/root_cometric[:, None]
    horizontal = null_space(vertical.T, rcond=1e-10)
    horizontal_positive, horizontal_values = positive_spectrum(horizontal.T@principal@horizontal)
    if len(positive) != len(horizontal_positive):
        raise AssertionError('A positive native oscillator was lost at the Gauss quotient')
    check(np.linalg.norm(positive-horizontal_positive), 'full versus horizontal Gauss frequencies', 2e-7)
    check(np.linalg.norm(principal@vertical), 'mass-normalized full Gauss Hessian null', 2e-7)
    # Independent 120D dual calculation; no sum-of-inertias approximation.
    small_metric = np.eye(120)
    small_metric[60:, 60:] += eta*D@D.T
    e, v = np.linalg.eigh(small_metric)
    metric_root = (v*np.sqrt(e))@v.T
    independent, _ = positive_spectrum(metric_root@hessian@metric_root)
    check(np.linalg.norm(positive-independent), '135D native versus 120D exact quotient cometric', 2e-7)
    cartesian, _ = positive_spectrum(hessian)
    coefficient = float(np.sqrt(positive).sum()/2)
    leading = float(np.sqrt(cartesian).sum()/2)
    bound = leading*np.sqrt(1+eta*np.linalg.norm(D, 2)**2)
    if coefficient < leading-2e-8 or coefficient > bound+2e-8:
        raise AssertionError('Exact principal harmonic leading-coefficient bounds failed')
    return dict(Lambda=Lambda,ue4=ue4,eta=eta,
        coordinate_dimension=135, gauge_rank=135-horizontal.shape[1],
        horizontal_linear_dimension=horizontal.shape[1],
        Hessian_positive_count=len(positive), full_Hessian_null_count=int(np.sum(abs(all_values)<=RESOLUTION)),
        horizontal_quadratic_null_count=int(np.sum(abs(horizontal_values)<=RESOLUTION)),
        frequency_divided_by_Lambda_three_quarters=np.sqrt(positive).tolist(),
        harmonic_half_trace_divided_by_Lambda_three_quarters=coefficient,
        Cartesian_half_trace=leading, exact_finite_eta_upper=bound,
        full_horizontal_frequency_residual=float(np.linalg.norm(positive-horizontal_positive)),
        Gauss_Hessian_null_residual=float(np.linalg.norm(principal@vertical)))


def native_energy(model, fields, theta):
    link = expm(1j*np.einsum('a,aij->ij', theta, model.generators))
    elastic, alignment = local_potential(fields, ((0,1),), (link,))
    h = model.onebody(fields, link, (0.,0.))[0]
    return elastic+alignment-float(np.linalg.eigvalsh(h)[4:].sum())


def full_hessian_controls(model, functional, fields, hessian, pullback, gauges):
    full = pullback.T@hessian@pullback
    errors = []
    for phase in (.23, .71):
        direction = np.sin(np.arange(135)*.37+phase); direction /= np.linalg.norm(direction)
        df = functional.fields(direction[:120]); theta = direction[120:]
        predicted = float(direction@full@direction)
        center = native_energy(model, fields, np.zeros(15))
        for step in (4e-4, 2e-4):
            finite = (native_energy(model, fields+step*df, step*theta)
                      -2*center+native_energy(model, fields-step*df, -step*theta))/step**2
            error = abs(finite-predicted)
            check(error, 'actual full field/link second derivative', 2e-7)
            errors.append(dict(phase=phase, step=step, actual_second_derivative=finite,
                               analytic_second_derivative=predicted, error=error))
    check(np.linalg.norm(full@gauges), 'all30 actual coordinate Gauss Hessian directions')
    # Holding fields frozen discards the real Cartesian Gauss terms.
    frozen = np.vstack((np.zeros((120,15)), np.eye(15)))
    wrong = float(np.linalg.norm(full@frozen))
    if wrong < 1e-3:
        raise AssertionError('Pure-link fake Gauss adverse control did not discriminate')
    return dict(full135_native_second_derivatives=errors,
                full_coordinate_Gauss_null_residual=float(np.linalg.norm(full@gauges)),
                omitted_Cartesian_Gauss_Hessian_defect=wrong)

def stabilizer_vacuum_control(model, fields, hessian, pullback, D):
    """Check residual isotropy of the NORMAL vacuum and the actual paired fibre."""
    kernel = null_space(D, rcond=1e-9)
    section = model.state(fields, np.eye(4), (0.,0.))
    full = pullback.T@hessian@pullback
    rows = []
    for vector in kernel.T:
        t = np.einsum('a,aij->ij', vector, model.generators)
        adjoint = 2*np.einsum('aij,bji->ab', model.generators,
                              1j*(t@model.generators-model.generators@t)).real
        action = block_diag(*([adjoint]*9))
        commutator = float(np.linalg.norm(full@action-action@full))
        charge = [model.lift(block_diag(t,t), species)@section.state for species in (0,1)]
        paired = float(np.linalg.norm(charge[0]+charge[1]))
        check(commutator, 'normal quadratic vacuum stabilizer invariance')
        check(paired, 'native paired stabilizer fibre character cancellation')
        rows.append(dict(normal_Hessian_isotropy_commutator=commutator,
            heavy_stabilizer_action_norm=float(np.linalg.norm(charge[0])),
            light_stabilizer_action_norm=float(np.linalg.norm(charge[1])),
            paired_stabilizer_action_norm=paired))
    return dict(stabilizer_dimension=kernel.shape[1], all_stabilizer_generators=rows,
        scope='Invariant local oscillator vacuum times actual paired fibre; no BO lower theorem')


def native_casimir_control(model, ue4):
    """A native fundamental matrix coefficient fixes the Haar kinetic normalization."""
    casimir = sum(t@t for t in model.generators)
    exact = 15/8*np.eye(4)
    check(np.linalg.norm(casimir-exact), 'native fundamental SU4 Laplacian normalization')
    wrong_generator = sum(b@b for b in model.basis)
    wrong = float(np.linalg.norm(wrong_generator-exact))
    if wrong < 1:
        raise AssertionError('B instead of T Haar kinetic adverse control did not discriminate')
    return dict(fundamental_Casimir_eigenvalues=np.linalg.eigvalsh(casimir).tolist(),
        fundamental_native_electric_energy=ue4*15/8,
        wrong_half_cometric_electric_energy=ue4*15/16,
        unnormalized_B_generator_Casimir_defect=wrong)



def projector_metric(model, fields):
    """Actual Slater quantum metric, including both heavy-top and light-bottom fills."""
    h = model.onebody(fields, np.eye(4), (0.,0.))
    spectra = [model.spectral(h[0],4,top=True)[1],model.spectral(h[1],4)[1]]
    metrics = []
    for index in range(137):
        df, du, dp = np.zeros_like(fields), np.zeros((4,4),complex), np.zeros(2)
        if index < 120:
            df.reshape(8,4,4)[index//15] = model.basis[index%15]
        elif index < 135:
            du = 1j*model.generators[index-120]
        else:
            dp[index-135] = 1
        changes = model.onebody_derivative(fields,np.eye(4),(0.,0.),df,du,dp)
        row = []
        for spectrum, dh in zip(spectra,changes):
            values, vectors, occupied, empty, _ = spectrum
            transition = vectors[:,empty].conj().T@dh@vectors[:,occupied]
            denominator = values[occupied][None,:]-values[empty,None]
            row.append(float(np.sum(abs(transition/denominator)**2)))
        metrics.append(row)
    result = np.array(metrics)
    section = model.state(fields,np.eye(4),(0.,0.))
    controls = []
    for index in (7, 91, 124, 135):
        df, du, dp = np.zeros_like(fields), np.zeros((4,4),complex), np.zeros(2)
        if index < 120:
            df.reshape(8,4,4)[index//15] = model.basis[index%15]
        elif index < 135:
            du = 1j*model.generators[index-120]
        else:
            dp[index-135] = 1
        derivative = model.derivative(section,fields,np.eye(4),(0.,0.),df,du,dp)
        horizontal = derivative-section.state*np.vdot(section.state,derivative)
        native = float(np.vdot(horizontal,horizontal).real)
        check(abs(native-result[index].sum()), 'primitive 4900 section versus Slater quantum metric')
        controls.append(dict(coordinate=index, primitive_metric=native,
                             spectral_metric=float(result[index].sum())))
    return dict(heavy_top4_light_bottom4_metric_trace_fields=result[:120].sum(axis=0).tolist(),
                metric_trace_colour=result[120:135].sum(axis=0).tolist(),
                metric_trace_phases=result[135:].sum(axis=0).tolist(),
                Born_Huang_field_coefficient_of_sqrt_Lambda=float(result[:120].sum()/2),
                actual_primitive_metric_controls=controls,
                scope='Local diagonal Born-Huang scalar only; not a full BO error or moduli Hamiltonian')


def soft_control(model, functional, fields, hessian):
    active = coordinates(fields[0,:2],model.basis)
    direction = null_space(active, rcond=1e-10)[:,0]
    soft = np.zeros((2,4,15)); soft[:,2] = direction
    check(np.linalg.norm(hessian@soft.reshape(-1)), 'symmetric unused exact quadratic soft direction')
    initial_elastic = local_potential(fields,((0,1),),(np.eye(4),))[0]
    rows=[]
    for step in (.03,.01):
        moved=fields+step*functional.fields(soft.reshape(-1))
        elastic=local_potential(moved,((0,1),),(np.eye(4),))[0]
        expected=STIFFNESS*step**4
        check(abs(elastic-initial_elastic-expected), 'actual quartic soft restoring term', 2e-15)
        rows.append(dict(step=step,elastic_excess=elastic-initial_elastic,exact_kappa_step_fourth=expected))
    return dict(soft_native_quartic_controls=rows,
                soft_quartic_energy_power=2/3, soft_localization_power=-1/12,
                scope='sqrt(Lambda) p^2 plus Lambda x^4 balance; coupled soft coefficient not solved')


def run(*, samples=33, ue4=.3):
    if samples < 5 or not np.isfinite(ue4) or ue4 <= 0:
        raise ValueError('At least5 samples and positive finite actual UE4 required')
    identities=source_hashes()
    band=RotationBand(**BARE)
    model=FullBandGaussSection(band,heavy_scale=1.)
    functional=Functional('pair',band,model,model.basis)
    casimir_control=native_casimir_control(model,ue4)
    rows=[]
    fractions=sorted(set(np.linspace(0,1,samples).tolist()+[.013,.37,.91,.999]))
    radius = q = None
    for fraction in fractions:
        fields,radius,q=canonical_fields(model,fraction)
        value=coordinates(fields,model.basis).reshape(-1)
        diagnostic=algebra_report(functional,value)
        check(diagnostic['gradient_norm'], 'actual full120 minimum stationarity')
        hessian=functional.hessian(value)
        positive, all_values=positive_spectrum(hessian)
        if len(positive)!=79 or np.sum(abs(all_values)<=RESOLUTION)!=41:
            raise AssertionError('Unexpected full native Hessian rank; do not hide a new soft mode')
        pullback,gauges,D=full_tangents(model,fields)
        principal=principal_spectrum(hessian,pullback,gauges,D,Lambda=1.,ue4=ue4)
        c,d,_=coefficients(model)
        aa=np.sqrt([1-q+q*fraction,1-q-q*fraction])
        predicted=np.sort([sign*radius+shift*d*radius/c*a
                           for a in aa for sign in (-1,1) for shift in (-1,1)])
        native_values=np.asarray(diagnostic['onebody_eigenvalues'])
        check(np.linalg.norm(native_values-predicted), 'complete native eight-band classification spectrum')
        row: dict[str, Any] = dict(fraction=fraction,
            actual_global=point(model,fields,np.eye(4),stiffness=STIFFNESS,alignment=ALIGNMENT),
            classification=orbit_classifier(model,fields,np.eye(4)),
            stabilizer_Lie_dimension=diagnostic['common_SU4_stabilizer_Lie_dimension'],
            gradient_norm=diagnostic['gradient_norm'],
            native_onebody_eigenvalues=native_values.tolist(),
            classified_eight_band_spectrum=predicted.tolist(),
            Cartesian_Hessian_eigenvalues=all_values.tolist(),
            Cartesian_harmonic_half_trace=float(np.sqrt(positive).sum()/2),
            native_full135_principal=principal)
        if fraction in (0.,.37,1.):
            row['actual_derivative_controls']=derivative_controls(functional,value)
            row['full135_Hessian_controls']=full_hessian_controls(model,functional,fields,hessian,pullback,gauges)
            row['actual_full_Gauss_primitive']=packet_point_controls(model,fields,np.eye(4),(0.,0.))
            row['actual_Cartesian_Gauss_control']=full_field_angular_control(model,fields,np.eye(4),(0.,0.))
            row['actual_Born_Huang_metric']=projector_metric(model,fields)
            row['actual_stabilizer_vacuum_control']=stabilizer_vacuum_control(model,fields,hessian,pullback,D)
            row['soft_control']=soft_control(model,functional,fields,hessian)
            row['native_kinetic_scaling']=[principal_spectrum(hessian,pullback,gauges,D,Lambda=scale,ue4=ue4)
                for scale in (1e4,1e12,1e24)]
        rows.append(row)
    assert radius is not None and q is not None
    # This search follows the COMPLETE orbit classification, but remains finite numerics.
    def coefficient(fraction):
        fields,_,_=canonical_fields(model,float(fraction))
        values,_=positive_spectrum(functional.hessian(coordinates(fields,model.basis).reshape(-1)))
        return float(np.sqrt(values).sum()/2)
    optimization=cast(OptimizeResult, minimize_scalar(
        coefficient,bounds=(0.,1.),method='bounded',options={'xatol':1e-11}))
    grid_best=min(rows,key=lambda row:row['Cartesian_harmonic_half_trace'])
    adverse=[]
    for fraction in (.013,.37,.999,1.):
        fields,_,_=canonical_fields(model,fraction)
        gl=expm(1j*np.einsum('a,aij->ij',.31*np.sin(np.arange(15)*.71+.3),model.generators))
        gr=expm(1j*np.einsum('a,aij->ij',.23*np.cos(np.arange(15)*.53+.2),model.generators))
        moved=fields.copy();moved[0]=gl@fields[0]@gl.conj().T;moved[1]=gr@fields[1]@gr.conj().T
        link=gl@gr.conj().T
        classified=orbit_classifier(model,moved,link)
        check(abs(classified['fraction']-fraction), 'noncentral independent-local-gauge orbit recovery')
        actual=point(model,moved,link,stiffness=STIFFNESS,alignment=ALIGNMENT)
        check(abs(actual['excess']), 'non-Cliff full-domain gauge minimum attainment')
        adverse.append(dict(fraction=fraction,actual_global=actual,classification=classified,
            actual_full_Gauss_primitive=packet_point_controls(model,moved,link,(.137,-.081)),
            actual_Cartesian_Gauss_control=full_field_angular_control(model,moved,link,(.137,-.081))))
    # Near-Cliff splitting: central second difference is diagnostic, not a derivative theorem.
    splitting=[]
    cliff=coefficient(0.)
    for step in (.02,.01,.005):
        second=2*(coefficient(step)-cliff)/step**2
        splitting.append(dict(fraction_step=step,even_split_second_difference=second))
    if not grid_best['Cartesian_harmonic_half_trace'] < cliff-1e-4:
        raise AssertionError('Actual non-Cliff harmonic discriminator failed')
    if source_hashes()!=identities:
        raise RuntimeError('Owner source/proof changed during moduli execution; rerun at frozen revision')
    return dict(source_sha256=identities,source_bindings_unchanged=True,
        parameters=dict(bare=BARE,stiffness=STIFFNESS,alignment=ALIGNMENT,ue4=ue4,samples=samples),
        radius=radius,q=q,exact_classical_minimum=-3*radius,
        native_Haar_Casimir_control=casimir_control,
        exact_classification_scope='Executed c>|d|, t!=0, J>0, 0<q<1/2; allR120 x SU4, modulo both complete local SU4 actions',
        exact_moduli='delta in [0,q], no extra minimum orbits; smooth15D unreduced active minimum manifold, singular orbit quotient interval',
        rows=rows,noncentral_nonCliff_adverse=adverse,
        finite_harmonic_prediction=dict(best_sample_fraction=grid_best['fraction'],
            Clifford_coefficient=cliff,endpoint_coefficient=coefficient(1.),
            bounded_search_fraction=float(optimization.x),bounded_search_coefficient=float(optimization.fun),
            bounded_search_success=bool(optimization.success),near_Cliff_split= splitting,
            conclusion='Numerics favor non-Cliff endpoint; neither sampled monotonicity nor bounded optimization is a selection theorem'),
        leading_coefficient_statement='For the exact local principal quadratic model, omega/Lambda^(3/4) tends sqrt of positive full120 Cartesian Hessian; all positive modes remain after complete Gauss. No full-quantum asymptotic theorem asserted.',
        remaining_premise='Uniform BO/semiclassical lower and admissible upper expansion on fullR120/SU4/T2 with soft26 directions and stabilizer-invariant normal vacuum, plus an analytic global minimizer proof for C(delta), would turn this diagnostic into quantum selection.',
        changed_construction_decision='Do not import a unique pair Clifford frame from invariant concentration. Carry complete singular-value moduli into the quantum construction; the actual leading principal form numerically disfavors Clifford.',
        tooling=dict(positive_AST='Functional.hessian via ast-grep',child_LSP='xd://lsp is not mounted; parent owns final native diagnostics'),
        all_checks_pass=True,quantum_ground_computed=False,campaign_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=33)
    parser.add_argument('--ue4',type=float,default=.3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run(samples=args.samples,ue4=args.ue4)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),all_checks_pass=report['all_checks_pass'],
        finite_harmonic_prediction=report['finite_harmonic_prediction'],
        quantum_ground_computed=False),indent=2))


if __name__ == '__main__':
    main()
