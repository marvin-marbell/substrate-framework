#!/usr/bin/env python3
"""Native ALL36 fixed-field-frame external pulse connected source survival.

No moving SF/RootW frame, no Born kernel, no on-shell Compton admission.
The original coupled photon source is preserved; a separately named post-source
unitary pulse supplies a nonzero-disconnected adverse control.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
import scipy
from scipy.linalg import get_blas_funcs
from scipy.sparse.linalg import LinearOperator, expm_multiply
import covariant_native_collective_primitive_electric as base
import covariant_collective_spectral_recovery as recovery
import covariant_collective_paired_car as car
import covariant_collective_paired_source_current as current

zgemm = get_blas_funcs(('gemm',), dtype=np.complex128)[0]
ROOT = Path(__file__).resolve().parent
PLAN = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT = PLAN.read_bytes()
SOURCE_CAP = ROOT / 'covariant_native_collective_primitive_electric_cut1_UE1_capsule_20261008.json'
PINNED = {
    '--labels': '5d1f8116bee533a97adbe7e04885ffadaa7c281abc27cc3679c966af7935c14a',
    '--car-proof': '517a539a51759fe1cb5e45668b3c760bbcf1201abfc83ea2cbf4884748900acf',
    '--prior-consumer': 'af2fb68e0b98152aa34140bb0849b51db53a34c612ae056e9eeac9b8fc37f1de'}
ARGUMENT_BYTES_AT_IMPORT: dict[str, bytes] = {}
for index, argument in enumerate(sys.argv[:-1]):
    if argument in PINNED:
        ARGUMENT_BYTES_AT_IMPORT[argument] = Path(sys.argv[index + 1]).read_bytes()
SOURCE_CAP_BYTES_AT_IMPORT = SOURCE_CAP.read_bytes()
if hashlib.sha256(SOURCE_CAP_BYTES_AT_IMPORT).hexdigest() != '6000a6b4a6d9e6e91c9b1f0c607a3b4708c41c878a7aa24cf08ff17182412b05':
    raise RuntimeError('Actual neutral/charged cut1 quantum input changed')
SOURCE_BYTES_AT_IMPORT: dict[str, bytes] = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if isinstance(filename, str):
        path = Path(filename)
        if path.suffix == '.py' and path.resolve().parent == ROOT:
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()


def wave(c: base.Spectrum, coefficients: np.ndarray) -> np.ndarray:
    """Batch all reference channels without forming a combined left-U."""
    result = np.zeros((c.dimension, coefficients.shape[1]), dtype=complex, order='F')
    for block in c.blocks:
        local = zgemm(1., block['V'], np.asfortranarray(coefficients[block['global_indices']]))
        result += block['P'] @ local
    return result


def project(c: base.Spectrum, physical: np.ndarray) -> np.ndarray:
    result = np.empty((len(c.q), physical.shape[1]), dtype=complex, order='F')
    for block in c.blocks:
        local = np.asfortranarray(block['P'].T @ physical)
        result[block['global_indices']] = zgemm(1., block['V'], local, trans_a=2)
    return result


class PulseHamiltonian(LinearOperator):
    """H+h j or H+h L in the SAME complete fixed physical frame."""
    def __init__(self, c: base.Spectrum, energy: np.ndarray, field: np.ndarray,
                 coefficient: float, perturbation: str):
        super().__init__(dtype=np.dtype(complex), shape=(len(c.q), len(c.q)))
        self.c, self.energy, self.field = c, energy, field
        self.coefficient, self.perturbation = coefficient, perturbation
        self.actual_operator_batch_calls = 0

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        self.actual_operator_batch_calls += 1
        HX = self.energy[:, None] * X
        combined = wave(self.c, np.concatenate((X, HX), axis=1))
        Ecombined = project(self.c, self.field[:, None] * combined)
        EX, EHX = np.split(Ecombined, 2, axis=1)
        if self.perturbation == 'current':
            derivative = -1j * (self.energy[:, None] * EX - EHX)
        elif self.perturbation == 'contact':
            E2combined = project(self.c, self.field[:, None]**2 * combined)
            E2X, E2HX = np.split(E2combined, 2, axis=1)
            EHEX = project(self.c, self.field[:, None] * wave(self.c, self.energy[:, None] * EX))
            derivative = -E2HX + 2 * EHEX - self.energy[:, None] * E2X
        else:
            raise ValueError('Unknown physical perturbation')
        return HX + self.coefficient * derivative

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._matmat(x.reshape(-1, 1)).reshape(-1)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self._matvec(x)

    def _rmatmat(self, X: np.ndarray) -> np.ndarray:
        return self._matmat(X)


def exact_pulse(c: base.Spectrum, energy: np.ndarray, field: np.ndarray,
                physical_sources: np.ndarray, duration: float, epsilon: float,
                transform_source_too: bool = False) -> np.ndarray:
    """Exact nonlinear exp(i eps E) H exp(-i eps E), source fixed."""
    displaced = np.exp(-1j * epsilon * field)[:, None] * physical_sources
    if transform_source_too:
        displaced = np.exp(-1j * epsilon * field)[:, None] * (
            np.exp(1j * epsilon * field)[:, None] * physical_sources)
    coefficients = project(c, displaced)
    evolved = np.exp(-duration * energy)[:, None] * coefficients
    return np.real(np.sum(coefficients.conj() * evolved, axis=0))


def prepared_source(n: base.Spectrum, c: base.Spectrum, neutral: dict[str, Any],
                    charged: dict[str, Any], S: Any, field: np.ndarray,
                    post_source_pulse: float) -> tuple[np.ndarray, dict[str, Any]]:
    Omega, incoming, Z = current.photon_state(n, neutral)
    raw = (S @ incoming).reshape(4, c.dimension)
    Xi = (c.q / n.q[0])[:, None] * project(c, raw.T)
    M0 = float(np.vdot(Xi, Xi).real)
    if M0 <= 0:
        raise ArithmeticError('Actual original coupled photon source vanished')
    physical = wave(c, Xi) / np.sqrt(M0)
    displaced = np.exp(-1j * post_source_pulse * field)[:, None] * physical
    combined = np.concatenate((physical, displaced), axis=1)
    coefficients = project(c, combined)
    return coefficients, dict(actual_M0=M0, actual_Zph=Z,
        actual_original_source_energy_mean=float(np.sum(abs(Xi)**2 * (-2*np.log(c.q/n.q[0]))[:, None])/M0),
        actual_source_norms_all4Ref=[float(np.vdot(coefficients[:, start:start+4], coefficients[:, start:start+4]).real) for start in (0, 4)],
        actual_source_projection_absolute_defect=float(np.linalg.norm(wave(c, coefficients)-combined)))


def run(labels_path: Path, car_path: Path, prior_path: Path) -> dict[str, Any]:
    for flag, path in (('--labels', labels_path), ('--car-proof', car_path), ('--prior-consumer', prior_path)):
        data = ARGUMENT_BYTES_AT_IMPORT[flag]
        if path.read_bytes() != data or hashlib.sha256(data).hexdigest() != PINNED[flag]:
            raise RuntimeError(f'Actual frozen input changed: {flag}')
    n, plus, controls = recovery.load_recovered(SOURCE_CAP, labels_path)
    gamma, embedding, nodes, heat, Hphi, metric = base.left.frame()
    neutral, addition, removal = base.space(0, 1), base.space(1, 1), base.space(-1, 1)
    Cbody = car.physical_conjugation(removal, addition, False)
    minus = base.Spectrum([dict(V=b['V'], q=b['q'], P=Cbody.T @ b['P'], sign=b['sign']) for b in plus.blocks], 3*len(removal['physical']))
    packet = np.array([.1, .8, .4, .1]); packet /= np.linalg.norm(packet)
    Sa = base.collective.source(neutral, addition, base.collective.native.source_coefficients(gamma, packet), embedding)
    Sh = car.annihilator(neutral, removal, car.negative_coefficients(gamma, packet), car.antiparticle_embedding(embedding))
    direction = np.array([.1, .2, .3, .4])
    pulse, duration, steps = .17, .5, (.01, .005)
    energy = -2*np.log(plus.q/n.q[0])
    field = np.tile(np.array([E for _, _, E in addition['physical']]) @ direction, 3)
    minus_field = np.tile(np.array([E for _, _, E in removal['physical']]) @ direction, 3)
    sources, preparation = prepared_source(n, plus, neutral, addition, Sa, field, pulse)
    hole_sources, hole_preparation = prepared_source(n, minus, neutral, removal, Sh, minus_field, pulse)
    physical_sources, physical_holes = wave(plus, sources), wave(minus, hole_sources)
    ES = project(plus, field[:, None] * physical_sources)
    E2S = project(plus, field[:, None]**2 * physical_sources)
    thermal = np.exp(-duration * energy)[:, None]
    G0 = np.real(np.sum(sources.conj()*thermal*sources, axis=0))
    G1 = np.real(1j*np.sum(ES.conj()*thermal*sources-sources.conj()*thermal*ES, axis=0))
    G2 = np.real(np.sum(-E2S.conj()*thermal*sources+2*ES.conj()*thermal*ES-sources.conj()*thermal*E2S, axis=0))
    def group(values: np.ndarray) -> np.ndarray:
        return np.array([np.sum(values[:4]), np.sum(values[4:])])
    g0, g1, g2 = group(G0), group(G1), group(G2)
    disconnected = (g1/g0)**2
    exact_connected = g2/g0-disconnected
    print(json.dumps(dict(stage='actual_ALL36_fixed_frame_Frechet_source_contact', ALL_charged_columns=len(energy), global_neutral_lambda=float(n.q[0]**2), source_M0=preparation['actual_M0'])), flush=True)
    finite = []
    for step in steps:
        current_values, contact_values, calls = [], [], []
        for kind, destination in (('current', current_values), ('contact', contact_values)):
            for sign in (-1, 1):
                operator = PulseHamiltonian(plus, energy, field, sign*step, kind)
                evolved = expm_multiply(-duration*operator, sources,
                                       traceA=-duration*float(np.sum(energy)))
                destination.append(group(np.real(np.sum(sources.conj()*evolved, axis=0))))
                calls.append(dict(perturbation=kind, sign=sign, actual_operator_batch_calls=operator.actual_operator_batch_calls))
        ordered = (current_values[0]-2*g0+current_values[1])/step**2
        seagull = (contact_values[1]-contact_values[0])/(2*step)
        assembled = (ordered+seagull)/g0-disconnected
        exact_plus = group(exact_pulse(plus, energy, field, physical_sources, duration, step))
        exact_minus = group(exact_pulse(plus, energy, field, physical_sources, duration, -step))
        nonlinear_log = (np.log(exact_plus)-2*np.log(g0)+np.log(exact_minus))/step**2
        finite.append(dict(step=step, ALL_two_ordered_current_contribution=ordered.tolist(), actual_seagull_Frechet_contribution=seagull.tolist(), actual_disconnected_contribution=disconnected.tolist(), assembled_connected_log_response=assembled.tolist(), exact_nonlinear_LOG_pulse_finite_difference=nonlinear_log.tolist(), exact_analytic_LOG_pulse_response=exact_connected.tolist(), assembled_vs_exact_analytic_absolute_defect=abs(assembled-exact_connected).tolist(), nonlinear_LOG_vs_exact_analytic_absolute_defect=abs(nonlinear_log-exact_connected).tolist(), seagull_omission_absolute_error=abs(ordered/g0-disconnected-exact_connected).tolist(), disconnected_omission_absolute_error=abs((ordered+seagull)/g0-exact_connected).tolist(), actual_operator_calls=calls))
        print(json.dumps(dict(stage='actual_Frechet_contact_step_complete', step=step, assembled=assembled.tolist(), exact=exact_connected.tolist())), flush=True)
    controls_pulse = []
    for epsilon in (-.2, .2):
        original = group(exact_pulse(plus, energy, field, physical_sources, duration, epsilon))
        changed_basis = group(exact_pulse(plus, energy, field, physical_sources, duration, epsilon, True))
        hole = group(exact_pulse(minus, energy, minus_field, physical_holes, duration, epsilon))
        opposite_particle = group(exact_pulse(plus, energy, field, physical_sources, duration, -epsilon))
        controls_pulse.append(dict(epsilon=epsilon, actual_fixed_source_survival=original.tolist(), actual_joint_source_and_H_unitary_survival=changed_basis.tolist(), joint_unitary_invariance_absolute_defect=abs(changed_basis-g0).tolist(), actual_proper_Ref0_to1_source_survival=hole.tolist(), actual_ORIGINAL_source_C_opposite_pulse_absolute_defect=float(abs(hole[0]-opposite_particle[0]))))
    old = json.loads(ARGUMENT_BYTES_AT_IMPORT['--prior-consumer'])
    prior_particle = old['actual_particle_photon_source_and_timed_receiver']
    descriptor = dict(plan_id=PLAN.stem, raw_plan_sha256_at_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest(), input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in ARGUMENT_BYTES_AT_IMPORT.items()}, source_quantum_capsule_sha256=hashlib.sha256(SOURCE_CAP_BYTES_AT_IMPORT).hexdigest(), law='H(epsilon)=exp(i epsilon sum_e d_e Ephys_e) H_transfer exp(-i epsilon sum_e d_e Ephys_e), source FIXED after actualcoupledphoton preparation', current='j_d=-i[H,E_d]', contact='L_dd=-[E_d,[E_d,H]], allE commute in fullfixedcanonicalGauss frame', global_neutral_lambda=float(n.q[0]**2), direction_ALL4edges=direction.tolist(), positive_imaginary_time_duration=duration, finite_difference_steps=list(steps), adverse_control_post_source_physical_pulse=pulse, active_CAR_numbers=[3,4,5], ALL_auxiliary_states=36, charged_flavours_including_taggedactive=3, primitive_UE=1., compact_electric_cutoff=1, Phi3_mean='SAME oldconditional kappa400 mean/metric; not selfconsistent newlaw', kappa_active=.09, kappa_aux_each=.04, all_charged_columns=len(plus.q), all_Ref_channels=4, no_sigma_clipping_or_gate=True, scope='Native compactselectedCliff2/ALL36/finitePhi3/finiteEcut; external physical pulse AFTER fixedsource. NOT movingRootW/BKS, full4Dirac, originalrealEM, massless/commoncone, fullCARretarded response or on-shellComptonS-matrix. Positiveimaginarytime source SURVIVAL, NOT oldreal-time coherentreceiver probability; no overwrite/retag of oldsource orPe.', source_boundary_caveat='Original cut1 literal photon source rawboundary loss~0.492478 stands; no full-field/source-completeness admission.')
    return dict(input_descriptor=descriptor, input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(), source_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_BYTES_AT_IMPORT.items()}, source_bytes_at_import={k:v.decode() for k,v in SOURCE_BYTES_AT_IMPORT.items()}, plan_bytes_at_import=PLAN_BYTES_AT_IMPORT.decode(), runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()), physical_mean_descriptor=metric, actual_spectral_recovery_controls=controls, actual_original_source_preparation=preparation, actual_proper_ref0_to1_source_preparation=hole_preparation, actual_original_M0_difference_from_frozen_consumer=abs(preparation['actual_M0']-prior_particle['actual_M0']), actual_original_source_energy_difference_from_frozen_consumer=abs(preparation['actual_original_source_energy_mean']-prior_particle['actual_energy_mean']), variants=['ORIGINALcoupledphoton_source', 'SEPARATEpost_source_physicalpulse_0.17'], exact_source_survival=g0.tolist(), exact_LOG_first_derivative=(g1/g0).tolist(), exact_LOG_connected_second_derivative=exact_connected.tolist(), actual_Frechet_current_contact_rows=finite, actual_external_pulse_vs_basischange_controls=controls_pulse, originalrealEM_or_Compton_admitted=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--car-proof', type=Path, required=True)
    parser.add_argument('--prior-consumer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result = run(args.labels, args.car_proof, args.prior_consumer)
    result['runtime_seconds'] = time.monotonic()-start
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(input_descriptor_sha256=result['input_descriptor_sha256'], source_at_import=result['source_at_import'], runtime_seconds=result['runtime_seconds'], exact_connected=result['exact_LOG_connected_second_derivative'], final_Frechet_row=result['actual_Frechet_current_contact_rows'][-1])), flush=True)


if __name__ == '__main__':
    main()
