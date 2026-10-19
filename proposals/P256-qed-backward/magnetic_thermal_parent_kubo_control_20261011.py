"""Independent quantum/CAR canonical Kubo pressure and ground-block controls.

Two canonical four-mode species plus a three-level quantum field. This finite
METHOD is not full native Phi/SU4/Gauss, a native thermal state, or a photon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg
import sympy as sp

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_onebody(h: np.ndarray) -> np.ndarray:
    """Complete full-CAR action restricted only by exact canonical number2."""
    masks = [mask for mask in range(16) if mask.bit_count() == 2]
    index = {mask: row for row, mask in enumerate(masks)}
    out = np.zeros((6, 6), dtype=np.complex128)
    for column, mask in enumerate(masks):
        for create in range(4):
            for annihilate in range(4):
                if not (mask >> annihilate) & 1:
                    continue
                sign = (-1) ** ((mask & ((1 << annihilate)-1)).bit_count())
                removed = mask ^ (1 << annihilate)
                if (removed >> create) & 1:
                    continue
                sign *= (-1) ** ((removed & ((1 << create)-1)).bit_count())
                target = removed | (1 << create)
                out[index[target], column] += sign*h[create, annihilate]
    return out


def matrices(alpha: float = 0.0, averaged_roles: bool = False,
             omit_density_cross: bool = False,
             dephase_field_kinetic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I6 = np.eye(6)
    I36 = np.eye(36)
    I3 = np.eye(3)
    field = np.array([[0, 1/np.sqrt(2), 0],
                      [1/np.sqrt(2), 0, 1], [0, 1, 0]], dtype=np.float64)
    kinetic = np.diag([1.0, 3.0, 5.0])
    if dephase_field_kinetic:
        _, frame = linalg.eigh(field)
        kinetic = frame @ np.diag(np.diag(frame.T @ kinetic @ frame)) @ frame.T
    assert np.linalg.norm(kinetic @ field-field @ kinetic) > .1 or dephase_field_kinetic
    amp = I3+field/3
    staggered = canonical_onebody(np.diag([1.0, -1.0, 1.0, -1.0]))
    localrho = canonical_onebody(np.diag([1.0, 1.0, 0.0, 0.0]))-I6
    rho_a = np.kron(localrho, I6)
    rho_b = np.kron(I6, localrho)
    masses = (1.1, 1.1) if averaged_roles else (.2, 2.0)
    onsite = masses[0]*np.kron(staggered, I6)+masses[1]*np.kron(I6, staggered)
    phase = np.exp(1j*alpha)
    h = np.zeros((4, 4), dtype=np.complex128)
    j = h.copy(); k = h.copy()
    for x in range(3):
        h[x, x+1] = h[x+1, x] = -1
    h[3, 0] = -phase; h[0, 3] = -phase.conjugate()
    j[3, 0] = -1j*phase; j[0, 3] = 1j*phase.conjugate()
    k[3, 0] = phase; k[0, 3] = phase.conjugate()
    hopping = np.kron(canonical_onebody(h), I6)+np.kron(I6, canonical_onebody(h))
    current = np.kron(canonical_onebody(j), I6)+np.kron(I6, canonical_onebody(j))
    contact = np.kron(canonical_onebody(k), I6)+np.kron(I6, canonical_onebody(k))
    density = rho_a @ rho_a+rho_b @ rho_b if omit_density_cross else (rho_a+rho_b) @ (rho_a+rho_b)
    H = np.kron(kinetic, I36)+np.kron(I3, onsite+64*density)+np.kron(amp, hopping)
    return H, np.kron(amp, current), np.kron(amp, contact), np.kron(I3, rho_a @ rho_b)


def thermal_response(H: np.ndarray, J: np.ndarray, K: np.ndarray,
                     beta: float) -> dict[str, float]:
    energies, basis = linalg.eigh(H)
    weights = np.exp(-beta*(energies-energies[0]))
    weights /= weights.sum()
    j = basis.conj().T @ J @ basis
    k = basis.conj().T @ K @ basis
    meanj = float(np.real(weights @ np.diag(j)))
    contact = float(np.real(weights @ np.diag(k)))
    connected = 0.0
    for i in range(len(energies)):
        for q in range(len(energies)):
            diff = energies[q]-energies[i]
            divdiff = beta*(weights[i]+weights[q])/2 if abs(diff)<1e-10 else (weights[i]-weights[q])/diff
            connected += float(divdiff*abs(j[i, q])**2)
    connected -= beta*meanj**2
    return {'pressure_second': contact-connected, 'contact': contact,
            'connected_Duhamel': connected, 'current_mean': meanj}


def pressure(H: np.ndarray, beta: float) -> float:
    energies = linalg.eigvalsh(H)
    return float(energies[0]-np.log(np.exp(-beta*(energies-energies[0])).sum())/beta)


def quantum_canonical_control() -> dict[str, Any]:
    beta = .7
    H, J, K, cross = matrices()
    assert np.linalg.norm(H-H.conj().T) < 1e-12
    response = thermal_response(H, J, K, beta)
    step = 1/128
    values = {i: pressure(matrices(i*step)[0], beta) for i in (-2, -1, 0, 1, 2)}
    finite_second = (-values[2]+16*values[1]-30*values[0]+16*values[-1]-values[-2])/(12*step**2)
    assert abs(finite_second-response['pressure_second']) < 1e-5
    generator = np.kron(np.eye(3), np.kron(canonical_onebody(np.diag([0., .3, -.4, .8])), np.eye(6)))
    comm = generator @ H-H @ generator
    Jg = 1j*comm
    Kg = -(generator @ comm-comm @ generator)
    gradient = thermal_response(H, Jg, Kg, beta)
    assert gradient['contact'] > 1e-5
    assert abs(gradient['pressure_second']) < 1e-9
    assert abs(gradient['connected_Duhamel']-gradient['contact']) < 1e-9
    wrongs = {}
    for name, kwargs in [('averaged_roles', {'averaged_roles': True}),
                         ('omit_shared_density_cross', {'omit_density_cross': True}),
                         ('static_quantum_field', {'dephase_field_kinetic': True})]:
        h, j, k, _ = matrices(**kwargs)
        result = thermal_response(h, j, k, beta)
        defect = abs(result['pressure_second']-response['pressure_second'])
        assert defect > 1e-7, (name, defect)
        wrongs[name] = {'wrong_pressure_second': result['pressure_second'], 'defect': defect}
    assert response['connected_Duhamel'] > 1e-6
    assert abs(-response['connected_Duhamel']-response['pressure_second']) > 1e-6
    energies, frame = linalg.eigh(H)
    weights = np.exp(-beta*(energies-energies[0])); weights /= weights.sum()
    cross_expectation = float(np.real(weights @ np.diag(frame.conj().T @ cross @ frame)))
    return {'canonical_species': 2, 'modes_per_species': 4,
            'number_per_species': 2, 'field_quantum_levels': 3,
            'dimension': 108, 'inverse_temperature': beta, 'response': response,
            'finite_difference_second': finite_second,
            'pure_translation': gradient, 'wrong_variants': wrongs,
            'shared_density_cross_expectation': cross_expectation,
            'cross_expectation_is_not_a_connected_covariance_or_entanglement_claim': True,
            'numerics': 'binary64 finite quantum/CAR method, no interval enclosure',
            'not_full_native_Phi_SU4_Gauss_or_all64_pressure': True}


def degenerate_ground_pressure() -> dict[str, Any]:
    beta = sp.Symbol('beta_temperature', positive=True)
    alpha = sp.Symbol('alpha', real=True)
    Z = sp.exp(-beta*alpha)+sp.exp(beta*alpha)+sp.exp(-3*beta)
    second = sp.simplify(sp.diff(-sp.log(Z)/beta, alpha, 2).subs(alpha, 0))
    expected = -2*beta/(2+sp.exp(-3*beta))
    assert sp.simplify(second-expected) == 0
    assert sp.limit(second/beta, beta, sp.oo) == -1
    return {'H0': [0, 0, 3], 'J': [1, -1, 0], 'K': 0,
            'whole_ground_rank': 2, 'Qg_J_Pg': 0,
            'ground_block_current_is_not_scalar': True,
            'finite_beta_pressure_second_exact': str(expected),
            'zero_temperature_regular_pressure_inference_rejected': True,
            'native_ground_degeneracy_claimed': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert all(digest(ROOT/name) == pin for name, pin in request['pins'].items())
    result = {'kind': 'EXERCISED_QUANTUM_CANONICAL_KUBO_METHOD_NOT_NATIVE_THERMAL_PHASE',
              'source_sha256': digest(Path(__file__)),
              'quantum_canonical': quantum_canonical_control(),
              'degenerate_ground': degenerate_ground_pressure(),
              'native_ground_or_thermal_response_computed': False,
              'physical1_phase_stiffness_photon': False, 'campaign_complete': False}
    assert all(digest(ROOT/name) == pin for name, pin in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'quantum_method_dimension': 108,
                      'native_pressure': False, 'phase_or_photon': False}))


if __name__ == '__main__':
    main()
