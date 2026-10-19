"""NEW actual mobile adjacent-sector/G/current/field-receiver consumer."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply
from rich_common_mobile_20261014_model import Mobile, MobileParameters
from rich_frame_photon_20261014_model import Parameters, spectrum, full_resolvent

ROOT = Path(__file__).resolve().parent


def norm(operator):
    return float(sp.linalg.norm(operator))


def expectation(operator, state):
    return float(np.vdot(state, operator @ state).real)


def evolve(operator, state, scale):
    return expm_multiply(scale * operator, state, traceA=scale * operator.diagonal().sum())


def main():
    inputs = json.loads((ROOT / 'rich_review_wave7_20261014_mobile_input.json').read_text())
    parameters = inputs['parameters']
    p = MobileParameters(base=Parameters(**parameters['base']), centres=tuple(tuple(row) for row in parameters['centres']), cm_hopping=parameters['cm_hopping'], core_mass=parameters['core_mass'])
    models = [Mobile(p, number) for number in (0, 1, 2)]
    neutral, mobile, pair = models
    values, vectors, ground_receipt = spectrum(neutral.H, count=4)
    reference = vectors[:, 0]
    sources = np.column_stack([neutral.source_to(mobile, mode) @ reference for mode in range(4)])
    car = []
    for mode in range(4):
        plus = mobile.source_to(pair, mode)
        minus = neutral.source_to(mobile, mode).getH()
        car.append(norm(plus.getH() @ plus + minus.getH() @ minus - mobile.I))
    k = np.array(inputs['wavevector'])
    rho = mobile.density_fourier(k)
    spatial = mobile.current_fourier(k)
    divergence = sum(k[a] * spatial[a] for a in range(3))
    continuity = norm(mobile.H @ rho - rho @ mobile.H + divergence)
    wrong_divergence = sum(k[a] * mobile.S[a] for a in range(3))
    z1, z2 = [values[0] + complex(*z) for z in inputs['frequencies']]
    r2, residual2 = full_resolvent(mobile.H, z2, sources)
    actual, residual_actual = full_resolvent(mobile.H, z1, ((z1-z2)*rho + divergence) @ r2)
    wrong, residual_wrong = full_resolvent(mobile.H, z1, ((z1-z2)*rho + wrong_divergence) @ r2)
    r1rho, residual_rho = full_resolvent(mobile.H, z1, rho @ sources)
    rhs = sources.conj().T @ (rho @ r2 - r1rho)
    ward = float(np.linalg.norm(sources.conj().T @ actual - rhs))
    wrong_ward = float(np.linalg.norm(sources.conj().T @ wrong - rhs))
    epsilon = inputs['derivative_step']
    derivatives = []
    for mode in range(2):
        shift = np.eye(2)[mode] * epsilon
        hp, hm = mobile.shifted(shift=shift), mobile.shifted(shift=-shift)
        derivatives.append({'current': norm((hp-hm)/(2*epsilon)+mobile.J[mode]), 'contact': norm((hp+hm-2*mobile.H)/epsilon**2-mobile.K[mode][mode])})
    cm = [-1j*(mobile.H @ x-x @ mobile.H) for x in mobile.X]
    total_position = []
    for axis in range(3):
        values_position = mobile.charge_data['charges'] @ mobile.charge_data['positions'][:, axis]
        total_position.append(sp.kron(mobile.If, sp.diags(values_position.ravel()), format='csr'))
    uniform_errors = []
    for axis in range(3):
        shift = np.eye(3)[axis]*epsilon
        derivative = -(mobile.shifted(uniform=shift)-mobile.shifted(uniform=-shift))/(2*epsilon)
        uniform_errors.append(norm(derivative-mobile.current_fourier(np.zeros(3))[axis]))
    position_errors = [norm(mobile.current_fourier(np.zeros(3))[axis]-1j*(mobile.H@total_position[axis]-total_position[axis]@mobile.H)) for axis in range(3)]
    source = sources @ np.asarray(inputs['source_coefficients'])
    source /= np.linalg.norm(source)
    field_generator = sum(value * operator for value, operator in zip(inputs['field_direction'], mobile.P))
    current_generator = sum(value * operator for value, operator in zip(inputs['current_direction'], mobile.J))
    spin = sum(value*operator for value, operator in zip(inputs['spin_axis'], mobile.S))
    spin_internal = sum(value*sum(side[axis] for side in mobile.spin) for axis, value in enumerate(inputs['spin_axis']))
    sval, su = eigh(spin_internal)
    project = su[:, sval>1e-8]@su[:, sval>1e-8].conj().T
    ps = mobile.mop(project)
    field_sign = (np.asarray(inputs['field_direction']) @ mobile.coordinates > 0).astype(float)
    pf = mobile.fop(sp.diags(field_sign))
    nleft = mobile.N[0]
    trajectories = {}
    for label, generator in (('field_only',field_generator), ('actual_current',current_generator), ('wrong_Noether_spin',spin), ('wrong_body_conditioner',mobile.N[0]@field_generator)):
        incoming = evolve(generator, source, -1j*inputs['pulse'])
        records = []
        for time in inputs['times']:
            state = evolve(mobile.H,incoming,-1j*time)
            probabilities = []
            for cell in (nleft, mobile.I-nleft):
                for spin_project in (ps,mobile.I-ps):
                    for field_project in (pf,mobile.I-pf):
                        probabilities.append(float(np.linalg.norm(cell@spin_project@field_project@state)**2))
            records.append({'time':time,'cell0':expectation(nleft,state),'X':[expectation(x,state) for x in mobile.X],
                            'CM_charge_current':[expectation(x,state) for x in cm], 'spin':expectation(spin,state),
                            'mode_current':[expectation(j,state) for j in mobile.J], 'field':[expectation(a,state) for a in mobile.A],
                            'probabilities_cell_spin_field':probabilities,'probability_sum':sum(probabilities)})
        trajectories[label]={'position_commutator':norm(generator@nleft-nleft@generator),'records':records}
    coherence = mobile.mop(mobile.cmhop+mobile.cmhop.getH())
    independent_error = norm(field_generator@coherence-coherence@field_generator)
    conditioned_error = norm((mobile.N[0]@field_generator)@coherence-coherence@(mobile.N[0]@field_generator))
    result={'kind':'NEW_WHOLE_H_MOBILE_SOURCE_CURRENT_TWO_FIELD_LEG_RECEIVER','inputs':inputs,
            'dimensions':[model.size for model in models],'matter_dimensions':[model.ns for model in models],
            'neutral_reference':{'energy':float(values[0]),**ground_receipt},'CAR_residuals':car,
            'source_Gram':[[str(x) for x in row] for row in sources.conj().T@sources],
            'continuity':continuity,'wrong_spin_continuity':norm(mobile.H@rho-rho@mobile.H+wrong_divergence),
            'whole_space_Ward':ward,'wrong_spin_Ward':wrong_ward,'resolvent_residuals':residual2+residual_actual+residual_wrong+residual_rho,
            'mode_derivatives':derivatives,'uniform_derivative_errors':uniform_errors,'total_position_current_errors':position_errors,
            'field_only_coherence_commutator':independent_error,'wrong_conditioner_coherence_commutator':conditioned_error,
            'field_coordinate_Gram':np.linalg.eigvalsh(np.column_stack([a@source for a in mobile.A]).conj().T@np.column_stack([a@source for a in mobile.A])).tolist(),
            'trajectories':trajectories,'scope':'Two fixed CM sites, genuine displacement/current, two global Gaussian 3D profiles and FULL64 supplier; not free COM, autonomous local field or Lorentz. Primitive core_mass input is not generated by scalar peer. No second active charged packet.'}
    (ROOT/'rich_review_wave7_20261014_mobile_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
