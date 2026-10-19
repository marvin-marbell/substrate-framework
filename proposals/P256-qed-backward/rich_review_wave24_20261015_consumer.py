"""Independent short physical uud-field event; no producer CLI or evolved input.
Taylor polynomial has a unitary-Duhamel terminal-defect error bound, not a
fixed-K schedule. The genuine OFF discriminator is exact tensor factorization.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from rich_review_wave19_20261015_all_m import AllM
from rich_spatial_colour_em_20261015_snapshot import load, matrix, source, charge_receiver
from rich_spatial_colour_em_20261015_model import assemble, observable

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave24_20261015_'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average(state, operator):
    return float(np.vdot(state, operator @ state).real)


def field_apply(state, unitary, shape):
    return np.einsum('ab,mfrb->mfra', unitary, state.reshape(shape), optimize=True).ravel()


def propagate(factors, state, duration, shift, tolerance):
    """Full-space Taylor action; all M blocks, no source/body/spin restriction.

    For self-adjoint A=H-shift and p_N(s), (d/ds+iA)p_N(s) is
    iA(-i s A)^N v/N!. Unitarity gives ||exp(-itA)v-p_N(t)||
    <= t^(N+1)||A^(N+1)v||/(N+1)! = ||next Taylor term||.
    Floating-point error is separately probed by four extra polynomial terms.
    """
    vector = factors.to_basis(state)
    result = np.zeros_like(vector)
    records = []
    for label, operator, indices, _ in factors.parts:
        value = vector[indices]
        if not np.any(value):
            records.append(dict(twice_M=label, initial_norm=0., degree=0, bound=0.))
            continue
        term = value.copy()
        total = value.copy()
        accepted = None
        for degree in range(1, 121):
            term = (-1j * duration / degree) * (operator @ term - shift * term)
            bound = float(np.linalg.norm(term))
            if accepted is None and degree >= 5 and bound < tolerance:
                accepted = (degree - 1, total.copy(), bound)
            total += term
            if accepted is not None and degree >= accepted[0] + 4:
                accepted_degree, lower, terminal_bound = accepted
                records.append(dict(twice_M=label, initial_norm=float(np.linalg.norm(value)), degree=degree,
                                    accepted_degree=accepted_degree, terminal_defect_bound=terminal_bound,
                                    four_extra_terms_difference=float(np.linalg.norm(total - lower)),
                                    final_term_norm=bound))
                result[indices] = total
                break
        else:
            raise RuntimeError('Short Taylor action did not reach the declared defect tolerance')
    return factors.from_basis(result), records


def marginal(state, effects, grid, shape):
    x = state.reshape(shape)
    result = np.zeros((len(effects), 2))
    for index, effect in enumerate(effects):
        applied = (effect @ x.reshape(shape[0], -1)).reshape(shape)
        for side, selected in enumerate([grid < 0, grid >= 0]):
            result[index, side] = np.vdot(x[:, selected], applied[:, selected]).real
    return result


def off_factorization(model, off, pulse, initial):
    """Prove each actual OFF term has no field factor except primitive H_E.

    Product Galerkin projection is body x phi x R x complete field shell.
    The only surviving field Hamiltonian term is I_material tensor H_E.
    This proof is valid even for an initially matter-field entangled state.
    """
    nr = model['base']['rotor']['dimension']
    ne = model['field']['dimension']
    electric = np.kron(np.eye(nr), model['field']['H_E'])
    field_terms = []
    errors = []
    for index, (matter, rq, phi) in enumerate(off.terms):
        value = rq.toarray()
        field_error = float(np.linalg.norm(value - electric))
        if not phi and field_error < 1e-13:
            identity_error = float(sparse.linalg.norm(matter - sparse.eye(off.nm)))
            if identity_error < 1e-13:
                field_terms.append(index)
                continue
        tensor = value.reshape(nr, ne, nr, ne)
        rotation = np.einsum('aibi->ab', tensor) / ne
        errors.append(float(np.linalg.norm(value - np.kron(rotation, np.eye(ne)))))
    if len(field_terms) != 1 or max(errors, default=0.) > 1e-11:
        raise RuntimeError('Actual OFF law does not factorize as claimed')
    return dict(field_only_electric_term_indices=field_terms, all_other_term_field_identity_errors=errors,
                maximum_field_identity_error=max(errors, default=0.), scalar_is_field_identity=off.include_scalar,
                initial_field_unitarity_error=float(np.linalg.norm(pulse.conj().T @ pulse - np.eye(ne))),
                same_initial_state=True, same_preparation=True, off_applies_from_original_t0=True,
                initial_norm=float(np.linalg.norm(initial)),
                theorem='Hoff=Hmaterial tensor I_field + I_material tensor HE. U acts only on field. For any initial entangled psi, reduced material densities of exp(-it Hoff)psi and exp(-it Hoff)Upsi agree exactly. Therefore every charge/phi PVM marginal, colour/scalar observable and S Sdag reverse-source observable has exactly zero pulse excess at every t in this finite law. Inclusive field receivers need not agree.',
                projected_assumptions='Full product body2108 x phi7 x R10 x field7, no selected total M or species/pole projection; actual rq matrices tested term by term; scalar H acts only phi. Both Peierls charge phase and rotor transverse connection removed by actual coupled=False assembly, not by turning off hops alone.')


def run():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    started = time.monotonic()
    state_json = ROOT / request['state_json']
    state_npz = ROOT / request['state_npz']
    metadata, data, model, laws = load(state_json, state_npz)
    name = 'baryon'
    h = laws[name]
    reference = data['neutral_reference']
    reference_residual = float(np.linalg.norm(laws['neutral'] @ reference - metadata['reference_energy'] * reference))
    insertion = source(data, model, name, request['source_key'])
    raw = insertion @ reference
    strength = float(np.linalg.norm(raw) ** 2)
    initial = raw / np.sqrt(strength)
    nr = model['base']['rotor']['dimension']
    ne = model['field']['dimension']
    nf = len(model['base']['scalar']['grid'])
    shape = (h.nm, nf, nr, ne)
    direction = np.asarray(request['drive_profile'], float)
    generator = sum(weight * q for weight, q in zip(direction, model['field']['Q']))
    pulse = expm(-1j * request['kick'] * generator)
    kicked = field_apply(initial, pulse, shape)
    rng = np.random.default_rng(240151)
    probe = rng.normal(size=h.shape[0]) + 1j * rng.normal(size=h.shape[0])
    probe /= np.linalg.norm(probe)
    rebuilt = assemble(model, name)
    owning_error = float(np.linalg.norm(rebuilt @ probe - h @ probe) / np.linalg.norm(h @ probe))
    if owning_error > 1e-11:
        raise RuntimeError('Snapshot and source-assembled H differ')
    off = assemble(model, name, coupled=False)
    causal = off_factorization(model, off, pulse, initial)
    factors = AllM(model, name, h)
    shift = average(initial, h)
    baseline, baseline_error = propagate(factors, initial, request['time'], shift, request['tolerance'])
    reaction, reaction_error = propagate(factors, kicked, request['time'], shift, request['tolerance'])
    op = model['base']['operators'][name]
    operators = {key: observable(model, name, matter=op[key]) for key in ['Q', 'rho0', 'rho1', 'C', 'B', 'N']}
    grid = model['base']['scalar']['grid']
    def measurements(state):
        values = {key: average(state, operator) for key, operator in operators.items()}
        x = state.reshape(shape)
        weights = np.sum(abs(x) ** 2, axis=(0, 2, 3))
        values.update(phi=float(weights @ grid), phi2=float(weights @ (grid ** 2)),
                      source_removal_strength=float(np.linalg.norm(insertion.H @ state) ** 2),
                      norm=float(np.linalg.norm(state)), energy=average(state, h))
        return values
    initial_base = measurements(initial)
    initial_pulse = measurements(kicked)
    base_values = measurements(baseline)
    pulse_values = measurements(reaction)
    labels, effects = charge_receiver(data, model, name)
    identity = sum(effects, sparse.csr_matrix(effects[0].shape))
    pvm_errors = [float(sparse.linalg.norm(effect @ effect - effect)) for effect in effects]
    charge_reconstruction = sum((q * effect for q, effect in zip(labels, effects)), sparse.csr_matrix(identity.shape))
    base_prob = marginal(baseline, effects, grid, shape)
    pulse_prob = marginal(reaction, effects, grid, shape)
    excess = {key: pulse_values[key] - base_values[key] for key in base_values}
    material_TV = float(np.sum(abs(pulse_prob - base_prob)) / 2)
    result = dict(kind='NEW_INDEPENDENT_PHYSICAL_THREE_QUARK_FIELD_CONSUMER', request=request,
                  reference_input=dict(json_sha256=sha(state_json), npz_sha256=sha(state_npz),
                                       status='Explicit declared original coupled reference protocol input, NOT an independent ground solve or evolved event input',
                                       energy=metadata['reference_energy'], norm=float(np.linalg.norm(reference)), eigen_residual=reference_residual),
                  source=dict(key=request['source_key'], raw_strength=strength,
                              construction='Apply exact snapshotted primitive physical site0 uud CAR source matrix to declared neutral reference, normalize once for preparation. No eigenpole/species filter, no borrowed evolved state.',
                              operator_matter_shape=list(insertion.matrix.shape), operator_matter_sha256=sha(state_npz)),
                  full_dimension=h.shape[0], shape=shape, owning_H_reassembly_relative_error=owning_error,
                  all_M=factors.evidence, causal_off=causal, initial_baseline=initial_base, initial_pulse=initial_pulse,
                  baseline=base_values, pulse=pulse_values, excess=excess,
                  charge_PVM=dict(physical_charge_labels=labels, physical_unit='qu=2/3 qd=-1/3, NOT integer3 labels or colour Casimir',
                                  identity_error=float(sparse.linalg.norm(identity-sparse.eye(h.nm))),
                                  idempotence_error_max=max(pvm_errors),
                                  physical_rho0_reconstruction_error=float(sparse.linalg.norm(charge_reconstruction-op['rho0'])),
                                  baseline_charge_phi=base_prob.tolist(), pulse_charge_phi=pulse_prob.tolist(),
                                  charge_phi_TV=material_TV, exact_off_charge_phi_TV=0.,
                                  normalization_errors=[float(abs(base_prob.sum()-1)), float(abs(pulse_prob.sum()-1))]),
                  propagation=dict(method='Own full compatible-M Taylor polynomial, unitary-Duhamel terminal defect bound plus four-extra-terms resolution check; no call to AllM.evolve or author propagation',
                                   baseline=baseline_error, pulse=reaction_error, shift=shift,
                                   roundoff_scope='Terminal bound is exact-arithmetic truncation bound, not a formal interval floating-point certificate; degree-extension, norm and energy checks separately recorded'),
                  provisional_verdict='PENDING', elapsed_seconds=time.monotonic()-started,
                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['provisional_verdict'] = ('FINITE_PHYSICAL_B_FEEDBACK_OBSERVED_AND_DEFEATED_BY_EXACT_GENUINE_OFF' if material_TV > 1e-9 and abs(excess['rho0']) > 1e-9 else 'NO_DECISION_CHANGING_MATERIAL_RESPONSE_AT_SELECTED_SHORT_TIME')
    (ROOT/(PREFIX+'provisional.json')).write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(ROOT/(PREFIX+'states.npz'), initial=initial, kicked=kicked, baseline=baseline, reaction=reaction)
    print(json.dumps(dict(verdict=result['provisional_verdict'], excess=excess, charge_phi_TV=material_TV,
                          maximum_initial_material_jump=max(abs(initial_pulse[key]-initial_base[key]) for key in ['rho0','C','B','phi','phi2','source_removal_strength']),
                          norm_errors=[abs(base_values['norm']-1),abs(pulse_values['norm']-1)],
                          elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    run()
