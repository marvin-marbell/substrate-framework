"""Coherent full-chain contractions: no which-block classical mixture."""
from __future__ import annotations
from typing import Any
import numpy as np
from rich_recoil_current_wave_20261016_model import Law

I2 = np.eye(2)
N = np.diag([0., 1.])
JW = np.diag([1., -1.])
C = np.array([[0., 1.], [0., 0.]])
CD = C.T


def cores(psi: Any) -> list[np.ndarray]:
    return [psi.get_B(i, form='B').to_ndarray() for i in range(psi.L)]


def left(env: np.ndarray, bra: np.ndarray, ket: np.ndarray, op: np.ndarray) -> np.ndarray:
    return np.einsum('ab,apc,pq,bqd->cd', env, bra.conj(), op, ket, optimize=True)


def right(env: np.ndarray, bra: np.ndarray, ket: np.ndarray, op: np.ndarray) -> np.ndarray:
    return np.einsum('apc,pq,bqd,cd->ab', bra.conj(), op, ket, env, optimize=True)


def scalar_product(bra: list[np.ndarray], ket: list[np.ndarray], operators: list[np.ndarray]) -> complex:
    env = np.ones((1, 1), complex)
    for a, b, op in zip(bra, ket, operators):
        env = left(env, a, b, op)
    return complex(env[0, 0])


class CoherentWave:
    def __init__(self, wave: dict[int, Any]):
        self.wave = wave
        self.arrays = {r: cores(psi) for r, psi in wave.items()}
        self.weights = {r: complex(psi.norm) for r, psi in wave.items()}
        self.rights = {}
        for r, a in self.arrays.items():
            for s, b in self.arrays.items():
                envs = [np.ones((1, 1), complex)]*(len(a)+1)
                for i in reversed(range(len(a))):
                    envs[i] = right(envs[i+1], a[i], b[i], np.eye(a[i].shape[1]))
                self.rights[r, s] = envs

    def conditional_one_body(self, field_operator: np.ndarray, intrablock: bool = False) -> np.ndarray:
        result = np.zeros((108, 108), complex)
        for r, a in self.arrays.items():
            for s, b in self.arrays.items():
                factor = self.weights[r].conjugate()*self.weights[s]
                ends = self.rights[r, s]
                starts = [np.ones((1, 1), complex)]
                for i in range(len(a)):
                    starts.append(left(starts[-1], a[i], b[i], field_operator if i == 0 else I2))
                for i in range(108):
                    site = i+1
                    result[i, i] += factor*np.sum(left(starts[site], a[site], b[site], N)*ends[site+1])
                    env = left(starts[site], a[site], b[site], CD @ JW)
                    stop = 12*(i//12+1) if intrablock else 108
                    for j in range(i+1, stop):
                        target = j+1
                        result[i, j] += factor*np.sum(left(env, a[target], b[target], C)*ends[target+1])
                        env = left(env, a[target], b[target], JW)
        # The field operators used here are Hermitian and commute with fermions.
        result += np.triu(result, 1).conj().T
        return result

    def field_density(self) -> np.ndarray:
        dimension = next(iter(self.arrays.values()))[0].shape[1]
        rho = np.zeros((dimension, dimension), complex)
        for r, a in self.arrays.items():
            for s, b in self.arrays.items():
                factor = self.weights[r].conjugate()*self.weights[s]
                rho += factor*np.einsum('pc,qd,cd->qp', a[0][0].conj(), b[0][0], self.rights[r, s][1], optimize=True)
        return rho

    def momentum(self, law: Law) -> dict[str, Any]:
        characteristic = np.zeros((3, 3), complex)
        for r, a in self.arrays.items():
            for matter_power in range(3):
                for field_power in range(3):
                    ops = [np.diag(np.exp(-2j*np.pi*field_power*law.field['momentum']/3))]
                    ops += [np.diag([1., np.exp(-2j*np.pi*matter_power*k/3)]) for k in law.one['momentum']]
                    characteristic[matter_power, field_power] += abs(self.weights[r])**2*scalar_product(a, a, ops)
        joint = np.array([[np.real(sum(np.exp(2j*np.pi*(m*a+f*b)/3)*characteristic[a, b] for a in range(3) for b in range(3))/9) for f in range(3)] for m in range(3)])
        total = np.array([sum(joint[m, f] for m in range(3) for f in range(3) if (m+f) % 3 == r) for r in range(3)])
        return dict(joint_matter_field=joint.tolist(), matter=joint.sum(axis=1).tolist(), field=joint.sum(axis=0).tolist(), total=total.tolist())

    def sea_depletion(self, law: Law) -> list[float]:
        probabilities = np.zeros(2, complex)
        for r, a in self.arrays.items():
            env = [left(np.ones((1, 1), complex), a[0], a[0], np.eye(a[0].shape[1])), np.zeros((a[0].shape[2], a[0].shape[2]), complex)]
            for i in range(108):
                t = a[i+1]
                if law.one['occupied'][i]:
                    env = [left(env[0], t, t, N), left(env[1], t, t, N)+left(env[0], t, t, I2-N)]
                else:
                    env = [left(e, t, t, I2) for e in env]
            probabilities += abs(self.weights[r])**2*np.array([e[0, 0] for e in env])
        return [float(probabilities[0].real), float(probabilities[1].real), float(1-np.sum(probabilities).real)]


def mpo_mean(wave: dict[int, Any], operator: Any) -> complex:
    return complex(sum(abs(psi.norm)**2*operator.H_MPO.expectation_value(psi) for psi in wave.values()))


def observe(law: Law, wave: dict[int, Any], expensive: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frame = CoherentWave(wave)
    size = len(law.field['momentum'])
    one = frame.conditional_one_body(np.eye(size))
    rho_field = frame.field_density()
    u = law.one['orbitals']
    physical_diagonal = np.einsum('ai,ij,aj->a', u.conj(), one, u, optimize=True).real
    density = physical_diagonal.reshape(27, 4).sum(axis=1)
    pauli = [np.array([[0., 1.], [1., 0.]]), np.array([[0., -1j], [1j, 0.]]), np.diag([1., -1.])]
    spins = [u.conj().T @ np.kron(np.eye(27), np.kron(np.eye(2), p/2)) @ u for p in pauli]
    spin = [float(np.sum(s*one).real) for s in spins]
    norm = float(np.trace(rho_field).real)
    result: dict[str, Any] = dict(norm=norm, density=density.tolist(), total_particles=float(np.sum(density)),
        primitive_frame_spin=spin, reference_negative_band_holes=float(np.sum(1-np.real(one.diagonal())[law.one['occupied']])),
        reference_depletion_P0_P1_Pge2=frame.sea_depletion(law),
        momentum=frame.momentum(law), field_purity=float(np.trace(rho_field@rho_field).real),
        field_coordinates=[float(np.trace(rho_field@q).real) for q in law.field['coordinates']],
        scalar_mean=float(np.trace(rho_field@law.field['phi']).real),
        physical_energy=float(mpo_mean(wave, law.model).real), maximum_bond=max(max(psi.chi) for psi in wave.values()))
    raw = dict(one_body_cdag_c=one, field_density=rho_field)
    if expensive:
        conditional = {c: frame.conditional_one_body(law.field['coupling'][c], True) for c in range(2, 14)}
        currents = np.zeros(4, complex)
        spin_current = np.zeros(4, complex)
        contacts = np.zeros((4, 4), complex)
        for pol in range(2):
            for z in range(3):
                c = 2+2*(3*pol+z)
                profile = np.zeros(4)
                profile[2*pol:2*pol+2] = law.parameters['charge']*np.array([np.cos(2*np.pi*z/3), np.sin(2*np.pi*z/3)])
                cc = cs = sc = ss = 0j
                for block in range(9):
                    sl = slice(12*block, 12*block+12)
                    hc, hs = law.one['components'][block, c:c+2]
                    cz, sz = conditional[c][sl, sl], conditional[c+1][sl, sl]
                    primitive = spins[2][sl, sl]
                    cc += np.sum(hc*cz)
                    cs += np.sum(hc*sz)
                    sc += np.sum(hs*cz)
                    ss += np.sum(hs*sz)
                    spin_current += profile*(np.sum(((primitive@hc+hc@primitive)/2)*sz)-np.sum(((primitive@hs+hs@primitive)/2)*cz))
                currents += profile*(cs-sc)
                contacts -= np.outer(profile, profile)*(cc+ss)
        emission = []
        for operator in law.emission:
            qj = mpo_mean(wave, operator['QJ'])
            jq = mpo_mean(wave, operator['JQ'])
            emission.append(dict(polarization=operator['polarization'], profile_sign=operator['profile_sign'],
                actual_current_C3_charge=operator['current_C3_charge'], QJ=[qj.real, qj.imag], JQ=[jq.real, jq.imag]))
        result.update(currents=np.real(currents).tolist(), contacts=np.real(contacts).tolist(),
                      spin_resolved_current_form_factors=[np.real(.5*currents+spin_current).tolist(), np.real(.5*currents-spin_current).tolist()],
                      signed_current_field_emission=emission,
                      current_contact_imaginary_roundoff=float(max(np.max(abs(np.imag(currents))), np.max(abs(np.imag(contacts))))))
        raw.update({f'conditional_F{c}_cdag_c': x for c, x in conditional.items()})
    return result, raw
