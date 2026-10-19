#!/usr/bin/env python3
"""Separated L insertion/R reference on native full fibres; bounded finite-Lambda cost.

The exact neutral ground enters the theorem, never a restricted-field packet.
Numerical sections below are differential-identity controls, not quantum grounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm, expm_frechet

from full_band_gauss_section import FullBandGaussSection
from pair_full_field_charge_transfer import charged_map, lift
from pair_full_field_concentration import constants, ground_moment_bound, source_hashes as inherited_hashes
from quantum_su4_gauss import norm, require_small, sparse_sum
from rotation_band import RotationBand
from unrestricted_field_minimum import ALIGNMENT, BARE, STIFFNESS, matrices, pair_clifford_reference

ROOT = Path(__file__).resolve().parent
CF = 15 / 8
LOCAL_LIGHT_CASIMIR_MAX = 5 / 2


def source_hashes():
    names = ('pair_separated_charge_transfer.py', 'pair_separated_charge_transfer.md',
             'pair_full_field_charge_transfer.py', 'pair_full_field_charge_transfer.md',
             'quantum_su4_photon.py', 'quantum_su4_photon.md',
             'quantum_su4_gauss.py', 'quantum_su4_gauss.md',
             'gauss_charge_intertwiner.py', 'gauss_charge_intertwiner.md')
    return {**inherited_hashes(), **{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                   for name in names}}


class SeparatedCharge:
    """Actual L marker words, transported to a distinct fixed R colour/EM reference."""

    def __init__(self, model, charge):
        if charge not in (-1, 1):
            raise ValueError('Charge must be -1 or +1')
        self.model, self.charge = model, charge
        self.matter, self.ladders, self.columns, _, self.boundary = charged_map(model, charge)
        self.left = self.ladders[:4]
        self.weight = (model.matter.light_left if charge == -1
                       else 4 - model.matter.light_left)
        self.light_Q = [sparse_sum(t[i, j] * model.matter.units[0, 1, i, j]
                                  for i in range(4) for j in range(4))
                        for t in model.generators]

    def coefficient(self, U):
        # Rows are external R components; columns are native L ladder labels.
        return U.conj().T if self.charge == -1 else U.T

    def assemble(self, F):
        return sparse_sum(sparse.kron(self.left[i], self.columns[j], format='csr') * F[j, i]
                          for j in range(4) for i in range(4)).tocsr()

    def phase(self, phases, path):
        return np.exp(1j * self.charge * (phases[0] if path == 0 else -phases[1]))

    def matrix(self, U, phases, path, *, fields=None, twist=0.):
        if twist:
            if fields is None:
                raise ValueError('Reference field twist requires all-field input')
            U = U @ expm(1j * twist * fields[1, 0])
        return self.phase(phases, path) * self.assemble(self.coefficient(U))

    def link_derivative(self, U, phases, path, t):
        # D=-i d/ds on U(s)=exp(i s t)U. Cartesian fields stay fixed.
        dU = 1j * t @ U
        dF = dU.conj().T if self.charge == -1 else dU.T
        return -1j * self.phase(phases, path) * self.assemble(dF)


def finite_bound(Lambda, law, *, stiffness, alignment, ue4, ue1):
    moment = ground_moment_bound(Lambda, law, stiffness=stiffness, alignment=alignment)
    deviation = np.sqrt(moment['expected_average_norm_deviation_squared_upper'])
    mean = np.asarray(law['optimal_average_norms']) + deviation
    light = float((law['light_u_coefficient'] * np.sqrt(mean[0])
                   + law['light_v_coefficient'] * np.sqrt(mean[1])) / 4)
    # Retain all positive kinetics in the global pointwise lower inequality.
    # Recover the positive excess directly, not from rounded total energies.
    kinetic = float(.75 * Lambda * moment['expected_global_remainder_upper'])
    cross = float(np.sqrt(LOCAL_LIGHT_CASIMIR_MAX * ue4 * kinetic))
    return dict(Lambda=Lambda, exact_neutral_total_kinetic_upper=kinetic,
                light_cost_upper=light, colour_cross_absolute_upper=cross,
                both_separated_charged_ground_minus_neutral_ground_upper=light + ue4 * CF + ue1 / 2 + cross,
                bound_scope='finite each Lambda; conservative upper bound grows O(Lambda^5/12), not uniform in Lambda',
                uniform_bound_obstruction='uncontrolled signed <Q_light,L dot J_field,L>; Gauss does not bound it by matter alone')


def vector(rng, size):
    value = rng.normal(size=size) + 1j * rng.normal(size=size)
    return value / np.linalg.norm(value)


def controls(model, fields, U, phases, *, ue4, ue1, twist=.7):
    rng = np.random.default_rng(222151)
    psi = vector(rng, model.matter.dimension)
    step = 2e-4
    reports = []
    for charge in (-1, 1):
        op = SeparatedCharge(model, charge)
        ib = sparse.eye(4, format='csr')
        im = sparse.eye(op.matter.dimension, format='csr')
        casimir = sparse_sum(q @ q for q in op.light_Q)
        local_casimirs = np.array([0., CF, LOCAL_LIGHT_CASIMIR_MAX, CF, 0.])
        casimir_residual = norm(casimir - sparse.diags(local_casimirs[model.matter.light_left]))
        require_small(casimir_residual, 'complete local light Fock Casimirs')
        residuals = dict(norm_operator=0., colour_Gauss=0., EM_Gauss=0., heavy_commutator=0.,
                         light_commutator=0., link_current=0., full_link_kinetic_commutator=0.,
                         full_phase_kinetic_commutator=0., corrected_full_colour_Gauss=0.,
                         corrected_link_current_unchanged=0., corrected_Cartesian_kinetic_commutator=0.)
        adverse = dict(omitted_boundary=0., wrong_phase_sign=0., omitted_colour_cross=0.,
                       reference_field_twist_missing_Cartesian_Gauss=0., reference_field_Cartesian_derivative=0.)
        # Complete fibres, including local n_L=0 and4; no local occupation restriction.
        for path in (0, 1):
            A = op.matrix(U, phases, path)
            image = A @ psi
            residuals['norm_operator'] = max(residuals['norm_operator'],
                norm(A.getH() @ A - sparse.diags(op.weight, dtype=float)))
            summed_fd = np.zeros_like(image)
            summed_exact = ue4 * CF * image
            cross = np.zeros_like(image)
            for a, t in enumerate(model.generators):
                DA = op.link_derivative(U, phases, path, t)
                residuals['link_current'] = max(residuals['link_current'],
                    norm((A.getH() @ DA + op.light_Q[a]) @ psi))
                jet = vector(rng, psi.size)
                plus = op.matrix(expm(1j * step * t) @ U, phases, path) @ (psi + 1j * step * jet)
                minus = op.matrix(expm(-1j * step * t) @ U, phases, path) @ (psi - 1j * step * jet)
                summed_fd += -ue4 * (plus - 2 * image + minus) / step**2
                cross += 2 * ue4 * (DA @ jet)
                for cell in range(2):
                    dU = 1j * t @ U if cell == 0 else -1j * U @ t
                    dF = dU.conj().T if charge == -1 else dU.T
                    dA = -1j * op.phase(phases, path) * op.assemble(dF)
                    target = sparse.kron(op.matter.Q[cell][a], ib, format='csr')
                    if cell == 1:
                        target += sparse.kron(im, sparse.csr_matrix(op.boundary[a]), format='csr')
                    right = target @ image - A @ (model.matter.Q[cell][a] @ psi)
                    residuals['colour_Gauss'] = max(residuals['colour_Gauss'], norm(right - dA @ psi))
                    if cell == 1:
                        adverse['omitted_boundary'] = max(adverse['omitted_boundary'],
                            norm(sparse.kron(op.matter.Q[cell][a], ib) @ image
                                 - A @ (model.matter.Q[cell][a] @ psi) - dA @ psi))
                    # Material second construction: all-field dependent reference twist.
                    V = expm(1j * twist * fields[1, 0])
                    dfield = 1j * (t @ fields[1, 0] - fields[1, 0] @ t) if cell == 1 else np.zeros((4, 4), complex)
                    dV = expm_frechet(1j * twist * fields[1, 0], 1j * twist * dfield, compute_expm=False)
                    B, dB = U @ V, dU @ V + U @ dV
                    dFB = dB.conj().T if charge == -1 else dB.T
                    At = op.matrix(U, phases, path, fields=fields, twist=twist)
                    dAt = -1j * op.phase(phases, path) * op.assemble(dFB)
                    residuals['corrected_full_colour_Gauss'] = max(residuals['corrected_full_colour_Gauss'],
                        norm(target @ (At @ psi) - At @ (model.matter.Q[cell][a] @ psi) - dAt @ psi))
                    if cell == 1:
                        fieldFB = (U @ dV).conj().T if charge == -1 else (U @ dV).T
                        adverse['reference_field_twist_missing_Cartesian_Gauss'] = max(
                            adverse['reference_field_twist_missing_Cartesian_Gauss'], norm(op.assemble(fieldFB) @ psi))
                Bt = U @ expm(1j * twist * fields[1, 0])
                At = op.matrix(U, phases, path, fields=fields, twist=twist)
                DAt = op.link_derivative(Bt, phases, path, t)
                residuals['corrected_link_current_unchanged'] = max(residuals['corrected_link_current_unchanged'],
                    norm((At.getH() @ DAt + op.light_Q[a]) @ psi))
            summed_exact += cross
            residuals['full_link_kinetic_commutator'] = max(residuals['full_link_kinetic_commutator'], norm(summed_fd - summed_exact))
            adverse['omitted_colour_cross'] = max(adverse['omitted_colour_cross'], norm(cross))
            shift = np.array([1., -1.])
            for cell in range(2):
                source = model.matter.light_left - 2 if cell == 0 else 2 - model.matter.light_left
                target_counts = (op.matter.light_left - 2 if cell == 0
                                 else op.matter.light_number - op.matter.light_left - 2 - charge)
                defect = sparse.kron(sparse.diags(target_counts, dtype=float), ib) @ image - A @ (source * psi)
                residuals['EM_Gauss'] = max(residuals['EM_Gauss'], norm(defect - charge * (1 if cell == 0 else -1) * image))
                adverse['wrong_phase_sign'] = max(adverse['wrong_phase_sign'], norm(defect + charge * (1 if cell == 0 else -1) * image))
            phase_fd = np.zeros_like(image)
            phase_exact = ue1 / 2 * image
            for axis in range(2):
                jet = vector(rng, psi.size)
                direction = np.eye(2)[axis]
                plus = op.matrix(U, phases + step * direction, path) @ (psi + 1j * step * jet)
                minus = op.matrix(U, phases - step * direction, path) @ (psi - 1j * step * jet)
                phase_fd += -ue1 / 2 * (plus - 2 * image + minus) / step**2
                momentum = charge if axis == 0 and path == 0 else -charge if axis == 1 and path == 1 else 0
                phase_exact += ue1 * momentum * (A @ jet)
            residuals['full_phase_kinetic_commutator'] = max(residuals['full_phase_kinetic_commutator'], norm(phase_fd - phase_exact))
            hH, hL = model.onebody(fields, U, phases)
            residuals['heavy_commutator'] = max(residuals['heavy_commutator'],
                norm(sparse.kron(lift(op.matter, hH, 0), ib) @ image - A @ (model.lift(hH, 0) @ psi)))
            F = op.coefficient(U) * op.phase(phases, path)
            terms = []
            for j in range(4):
                for i in range(4):
                    coefficients = hL[:, i] if charge == 1 else -hL[i, :]
                    ladder = sparse_sum(coefficients[k] * op.ladders[k] for k in range(8))
                    terms.append(F[j, i] * sparse.kron(ladder, op.columns[j], format='csr'))
            exact = sparse_sum(terms)
            residuals['light_commutator'] = max(residuals['light_commutator'],
                norm(sparse.kron(lift(op.matter, hL, 1), ib) @ image - A @ (model.lift(hL, 1) @ psi) - exact @ psi))
            # Exact second exponential derivative from a triangular block exponential.
            # These15 directions are all nonzero Cartesian directions of the twist;
            # the other105 derivative coefficients vanish identically.
            At = op.matrix(U, phases, path, fields=fields, twist=twist)
            twisted_image = At @ psi
            field_fd = np.zeros_like(image)
            field_exact = np.zeros_like(image)
            M = 1j * twist * fields[1, 0]
            for direction in model.basis:
                E = 1j * twist * direction
                block = np.zeros((12, 12), complex)
                for slot in range(3):
                    block[4 * slot:4 * slot + 4, 4 * slot:4 * slot + 4] = M
                block[:4, 4:8] = E
                block[4:8, 8:] = E
                dV = expm_frechet(M, E, compute_expm=False)
                d2V = 2 * expm(block)[:4, 8:]
                dB, d2B = U @ dV, U @ d2V
                dF = dB.conj().T if charge == -1 else dB.T
                d2F = d2B.conj().T if charge == -1 else d2B.T
                dA = op.phase(phases, path) * op.assemble(dF)
                d2A = op.phase(phases, path) * op.assemble(d2F)
                jet = vector(rng, psi.size)
                plus_fields, minus_fields = fields.copy(), fields.copy()
                plus_fields[1, 0] += step * direction
                minus_fields[1, 0] -= step * direction
                plus = op.matrix(U, phases, path, fields=plus_fields, twist=twist) @ (psi + 1j * step * jet)
                minus = op.matrix(U, phases, path, fields=minus_fields, twist=twist) @ (psi - 1j * step * jet)
                # Unit coefficient p_field²/2; multiply both sides by sqrtLambda
                # for the actual Hamiltonian. No inertia rotor is added.
                field_fd += -(plus - 2 * twisted_image + minus) / (2 * step**2)
                field_exact += -(d2A @ psi) / 2 - 1j * (dA @ jet)
                adverse['reference_field_Cartesian_derivative'] = max(
                    adverse['reference_field_Cartesian_derivative'], norm(dA @ psi))
            residuals['corrected_Cartesian_kinetic_commutator'] = max(
                residuals['corrected_Cartesian_kinetic_commutator'], norm(field_fd - field_exact))
        for label, value in residuals.items():
            require_small(value, label, 3e-6 if 'kinetic' in label else 3e-9)
        for label, value in adverse.items():
            if value < 1e-5:
                raise AssertionError(f'Adverse control vanished: {label}')
        n = model.matter.light_left
        W = op.weight
        em_mixed = ue1 / 2 + ue1 * charge * W * (n - 2) / 4
        reports.append(dict(charge=charge, native_dimensions=[model.matter.dimension, 4 * op.matter.dimension],
                            residuals=residuals, adverse=adverse,
                            complete_local_light_Casimir_residual=casimir_residual,
                            parallel_path_EM_cost_integrand_at_ground_norm2_by_n=[
                                float(em_mixed[np.flatnonzero(n == k)[0]]) for k in range(5)],
                            null_source_local_occupation=0 if charge == -1 else 4))
    return reports


def full_field_angular_control(model, fields, U, phases):
    section = model.state(fields, U, phases)
    full, omitted = [], []
    for cell in range(2):
        for a, t in enumerate(model.generators):
            df = np.zeros_like(fields)
            df[cell] = 1j * (t @ fields[cell] - fields[cell] @ t)
            dU = 1j * t @ U if cell == 0 else -1j * U @ t
            matter = 1j * (model.matter.Q[cell][a] @ section.state)
            joint = model.derivative(section, fields, U, phases, df, dU)
            link_only = model.derivative(section, fields, U, phases, dU=dU)
            full.append(norm(joint - matter))
            omitted.append(norm(link_only - matter))
    require_small(max(full), 'native full Cartesian-plus-link Gauss', 3e-8)
    if max(omitted) < 1e-4:
        raise AssertionError('Missing Cartesian angular momentum did not discriminate')
    return dict(full_all30_Gauss_residual=max(full), omitted_Cartesian_angular_Gauss_defect=max(omitted),
                scope='actual native section differential control, not an exact quantum ground')


def run(*, Lambdas, ue4=.3, ue1=.4, magnetic=.7):
    if min(ue4, ue1) <= 0 or magnetic < 0 or not all(np.isfinite(v) for v in (ue4, ue1, magnetic)):
        raise ValueError('Finite positive electric and nonnegative magnetic coefficients required')
    identities = source_hashes()
    model = FullBandGaussSection(RotationBand(**BARE), heavy_scale=1.)
    frame, _ = pair_clifford_reference(model, stiffness=STIFFNESS)
    anchor = np.tile(frame, (2, 1, 1, 1))
    law = constants(model, anchor, stiffness=STIFFNESS, alignment=ALIGNMENT,
                    ue4=ue4, ue1=ue1, magnetic=magnetic, chart_radius=.001)
    rng = np.random.default_rng(222152)
    inputs, points = [], []
    angular: dict[str, Any] | None = None
    for scale in (.02, .3):
        coefficients = rng.normal(size=(2, 4, 15)) * scale
        fields = anchor + matrices(coefficients, model.basis)
        U = np.asarray(expm(1j * np.einsum('a,aij->ij', rng.normal(size=15) * .35, model.generators)), dtype=complex)
        phases = rng.uniform(-np.pi, np.pi, 2)
        points.append(controls(model, fields, U, phases, ue4=ue4, ue1=ue1))
        inputs.append(dict(all120_perturbation=coefficients.tolist(), noncentral_link_real=np.real(U).tolist(),
                           noncentral_link_imag=np.imag(U).tolist(), phases=phases.tolist()))
        if scale == .02:
            angular = full_field_angular_control(model, fields, U, phases)
    assert angular is not None
    rows = [finite_bound(float(L), law, stiffness=STIFFNESS, alignment=ALIGNMENT,
                         ue4=ue4, ue1=ue1) for L in Lambdas]
    if source_hashes() != identities:
        raise RuntimeError('Scientific owners changed during execution')
    return dict(source_sha256=identities, source_bindings_unchanged=True,
                governing_plan='issue-222-continuing-backward-20260930', work_item='campaign',
                exact_domain='fullL2(R120 x SU4 x T2), NH4 NL4 neutral4900 -> NL3/5 full3920 x boundary4',
                separated_operator='L insertion, R reference: hole exp(-iA0 or+iA1) Udagger cL; particle conjugate dressing',
                charged_extension='same full-coordinate Hamiltonian; external compensating colour+EM source only at R',
                path_trial='incoherent equal mixture of the two separately physical path images, each norm squared2',
                controls=points, native_full_field_angular_control=angular, input_points=inputs,
                bounds=rows, all_checks_pass=True, campaign_complete=False,
                failed_uniform_bound='EM loop cross cancels using exact Gauss; colour cross retains Cartesian angular momentum',
                correction_attempt='reference field twist U exp(i beta Phi0,R), beta=.7: same colour cross, new Cartesian kinetic terms',
                limits=['Finite graph variational upper costs, not spectral charge gap, physical spin, mobile band or Coulomb excitation.',
                        'No uniform large-Lambda bound derived; finite each Lambda bound is valid but grows.',
                        'Native pointwise differential probes do not solve either full quantum ground.',
                        'No global no-go: only the pure-link and unitary reference-field-twist route diagnosed.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Lambda', type=float, nargs='+', default=[1e20, 1e30, 1e48])
    parser.add_argument('--ue4', type=float, default=.3)
    parser.add_argument('--ue1', type=float, default=.4)
    parser.add_argument('--magnetic', type=float, default=.7)
    parser.add_argument('--output', type=Path, default=ROOT / 'pair_separated_charge_transfer_receipt.json')
    args = parser.parse_args()
    result = run(Lambdas=args.Lambda, ue4=args.ue4, ue1=args.ue1, magnetic=args.magnetic)
    text = json.dumps(result, indent=2)
    args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
