#!/usr/bin/env python3
"""NEW transported-Phi Wilson/primitive root law; not frozen continuous-time Root.

The full positive extension is declared below, including its outside-chart block.
The selected photon supplier alone does not supply this SU4/Phi completion.
This executable calculates native edge-qubit transfer, pairing, source, current,
Gauss and finite-defect controls, then symbolic finite-family transfer bounds.
The radial successor removes the obsolete Gaussian cutoff-energy estimate and
places the actual heat-kernel cross transition INSIDE its Dirichlet chart ball.
Source admission still requires a nonzero charged-sector OS/range Gram; Gauss
covariance and a fixed-number source norm alone never supply a log-transfer energy.
No claim of an all-orders interacting Compton probability is made.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.abc
import importlib.machinery
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
import numpy as np
import scipy
from scipy.linalg import block_diag, expm

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ('covariant_root_transfer.py', 'covariant_induced_photon.py',
                'induced_root_transfer.py', 'induced_photon_response.py',
                'bosonic_statistics.py', 'sea_selection.py', 'rotation_band.py',
                'improved_phi_band.py', 'microscopic_native_orbit_lift.py',
                'local_orbital_stability.py')
SOURCE_BYTES_AT_IMPORT = {name: (ROOT/name).read_bytes() for name in SOURCE_NAMES}
SOURCE_HASHES_AT_IMPORT = {name: hashlib.sha256(value).hexdigest()
                         for name, value in SOURCE_BYTES_AT_IMPORT.items()}
LOADED_ROOT_CODE_MATCH = sys._getframe().f_code == compile(
    SOURCE_BYTES_AT_IMPORT['covariant_root_transfer.py'], __file__, 'exec',
    dont_inherit=True, optimize=sys.flags.optimize)
if not LOADED_ROOT_CODE_MATCH:
    raise ArithmeticError('Root loaded code differs from captured source bytes')

class CapsuleLoader(importlib.machinery.SourceFileLoader):
    """Compile the very bytes recorded, including transitive local suppliers."""
    def get_code(self, fullname: str) -> Any:
        data = Path(self.path).read_bytes()
        name = Path(self.path).name
        SOURCE_BYTES_AT_IMPORT[name] = data
        SOURCE_HASHES_AT_IMPORT[name] = hashlib.sha256(data).hexdigest()
        return compile(data, self.path, 'exec')


class CapsuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.origin and Path(spec.origin).parent == ROOT and spec.origin.endswith('.py'):
            spec.loader = CapsuleLoader(fullname, spec.origin)
        return spec


_capsule_finder = CapsuleFinder()
sys.meta_path.insert(0, _capsule_finder)
from bosonic_statistics import BosonicEdgeModel, tensor_pauli
from covariant_induced_photon import Parameters, WilsonConsumer, phi_frame, loop_coefficient
from sea_selection import matrix_unit_basis, SELECTED
from induced_photon_response import ResponseConfig, make_band
from induced_root_transfer import FlavorPauliGraph
sys.meta_path.remove(_capsule_finder)

PHYSICAL_COMPONENT_CONNECTION_LAW = 'physical-components-Dminus-20261008'


@dataclass(frozen=True)
class PhysicalComponentFrame:
    """A unitary component-frame jet Q, dQ; no independent colour/spin fibre."""
    q: np.ndarray
    dq: np.ndarray

    @property
    def connection(self) -> np.ndarray:
        return -self.dq@self.q.conj().T

    def vector(self, coordinate_vector: np.ndarray) -> np.ndarray:
        return self.q@coordinate_vector

    def operator(self, coordinate_operator: np.ndarray) -> np.ndarray:
        return self.q@coordinate_operator@self.q.conj().T

    def covariant_derivative(self, physical_vector: np.ndarray,
                             ordinary_derivative: np.ndarray) -> np.ndarray:
        return ordinary_derivative+self.connection@physical_vector

    def covariant_operator_derivative(self, physical_operator: np.ndarray,
                                      ordinary_derivative: np.ndarray) -> np.ndarray:
        a = self.connection
        return ordinary_derivative+a@physical_operator-physical_operator@a

    def source(self, coordinate_source: np.ndarray,
               neutral_frame: PhysicalComponentFrame) -> np.ndarray:
        return self.q@coordinate_source@neutral_frame.q.conj().T

    def covariant_source_derivative(self, physical_source: np.ndarray,
                                    ordinary_derivative: np.ndarray,
                                    neutral_frame: PhysicalComponentFrame) -> np.ndarray:
        return ordinary_derivative+self.connection@physical_source-physical_source@neutral_frame.connection

    def world_connection_generator(self) -> np.ndarray:
        """Ordinary -i*d = covariant -i*Dphys PLUS this Hermitian connection."""
        return -1j*self.dq@self.q.conj().T


def physical_component_friedrichs_form(node_frames: tuple[np.ndarray, ...],
                                      edge_frames: tuple[np.ndarray, ...],
                                      coordinate_factor: np.ndarray,
                                      coordinate_potential: np.ndarray) -> dict[str, np.ndarray]:
    """Actual physical A†A kinetic and potential in ONE component frame.

    In the continuum Aphys=Q (d-dlogk) Q†, Dphys=d-(dQ)Q†.
    The edge factor is its ground-factorized Galerkin form, including all
    endpoint parallel transports. This constructs the operator, not just a
    connection label. Dirichlet/Friedrichs domains are carried by the same Q.
    """
    nodes = block_diag(*node_frames)
    edges = block_diag(*edge_frames)
    factor = edges@coordinate_factor@nodes.conj().T
    kinetic = factor.conj().T@factor
    potential = nodes@coordinate_potential@nodes.conj().T
    return dict(factor=factor, kinetic=kinetic, potential=potential,
                Hamiltonian=kinetic+potential, component_unitary=nodes)


@dataclass(frozen=True)
class RootParameters:
    a: float = .01
    mass: float = 1.
    momentum: tuple[float, ...] = (.7, -.4, .3)
    native_cycle_penalty: float = 12.
    native_defect: float = .07
    normal_width_power: int = 1800
    tube_radius_power: int = 1600
    normal_frequency: float = 1.
    source_charge: int = 1


def positive_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    if values[0] <= 0:
        raise ArithmeticError('positive transfer required, not inverse Dirac matrix')
    return (vectors*np.log(values))@vectors.conj().T


def wilson_component_basis(gamma: np.ndarray) -> np.ndarray:
    """Fixed SU(4) reference embedding: (+gamma0, -gamma0) columns."""
    _, vectors = np.linalg.eigh(gamma[0])
    theta = np.concatenate((vectors[:, 2:], vectors[:, :2]), axis=1)
    return theta*np.exp(-.25j*np.angle(np.linalg.det(theta)))


def wilson_step(gamma: np.ndarray, a: float, mass: float,
                momentum: np.ndarray) -> tuple[np.ndarray, float, float]:
    consumer = WilsonConsumer(gamma, a, mass)
    b = 1+consumer.m0+float(np.sum(1-np.cos(momentum)))
    theta = wilson_component_basis(gamma)
    plus, minus = theta[:, :2], theta[:, 2:]
    f = plus.conj().T@(1j*np.einsum('j,jab->ab', np.sin(momentum), gamma[1:]))@minus
    l = np.block([[np.eye(2), -f], [np.zeros((2, 2)), np.eye(2)]])
    block = l.conj().T@np.diag([1/b, 1/b, b, b])@l
    return theta@block@theta.conj().T, b*b, consumer.energy(momentum)


def native_lift(model: BosonicEdgeModel, matrix: np.ndarray,
                offset: int = 0) -> np.ndarray:
    """Actual primitive even path polynomials, never an occupation isometry."""
    output = np.zeros_like(model.identity)
    anchor = offset
    for target, source in itertools.product(range(4), repeat=2):
        i, j = offset+target, offset+source
        if i == j:
            word = model.n(i)
        else:
            path = tuple(([j] if j != anchor else [])+[anchor]+([i] if i != anchor else []))
            word = model.bilinear(i, j, path=path)
        output += matrix[target, source]*word
    return output


def exterior(matrix: np.ndarray) -> np.ndarray:
    """Independent finite comparator; never used to construct native transfer."""
    states = [tuple(i for i in range(4) if mask & (1 << i)) for mask in range(16)]
    out = np.zeros((16, 16), complex)
    for r, occupied_r in enumerate(states):
        for c, occupied_c in enumerate(states):
            if len(occupied_r) == len(occupied_c):
                out[r, c] = np.linalg.det(matrix[np.ix_(occupied_r, occupied_c)]) if occupied_r else 1.
    return out


def car_comparator() -> list[np.ndarray]:
    """Signs only for discriminating particle-hole algebra, not primitive law."""
    out = []
    for i in range(4):
        c = np.zeros((16, 16), complex)
        for mask in range(16):
            if mask & (1 << i):
                c[mask ^ (1 << i), mask] = (-1)**((mask & ((1 << i)-1)).bit_count())
        out.append(c)
    return out


def pairing_control(t: np.ndarray, gamma: np.ndarray) -> dict[str, Any]:
    theta = wilson_component_basis(gamma)
    # The particle/hole comparator uses eigenspace labels, not raw components.
    h = -theta.conj().T@positive_log(t)@theta
    c = car_comparator()
    p, antiparticle = c[:2], [c[j].conj().T for j in (2, 3)]
    ordinary = sum((h[i, j]*c[i].conj().T@c[j] for i, j in itertools.product(range(4), repeat=2)), np.zeros((16, 16), complex))
    paired = np.trace(h[2:, 2:])*np.eye(16, dtype=complex)
    for i, j in itertools.product(range(2), repeat=2):
        paired += h[i, j]*p[i].conj().T@p[j]
        paired -= h[2+i, 2+j]*antiparticle[j].conj().T@antiparticle[i]
        paired += h[i, 2+j]*p[i].conj().T@antiparticle[j].conj().T
        paired += h[2+j, i]*antiparticle[j]@p[i]
    number_p = sum((x.conj().T@x for x in p), np.zeros((16, 16), complex))
    number_a = sum((x.conj().T@x for x in antiparticle), np.zeros((16, 16), complex))
    charge = number_p-number_a
    dropped = paired.copy()
    for i, j in itertools.product(range(2), repeat=2):
        dropped -= h[i, 2+j]*p[i].conj().T@antiparticle[j].conj().T
        dropped -= h[2+j, i]*antiparticle[j]@p[i]
    return dict(particle_modes=2, antiparticle_modes=2,
                exact_particle_hole_pairing_identity_error=float(np.linalg.norm(ordinary-paired)),
                actual_pair_creation_block_norm=float(np.linalg.norm(h[:2, 2:])),
                charge_commutator=float(np.linalg.norm(paired@charge-charge@paired)),
                wrong_no_pairing_transfer_error=float(np.linalg.norm(expm(-dropped)-expm(-paired))),
                particle_plus_antiparticle_number_commutator=float(np.linalg.norm(paired@(number_p+number_a)-(number_p+number_a)@paired)),
                convention='p=c_plus, antiparticle=c_minus^dag; electric charge n_p-n_antiparticle=n_Dirac-2. Pair pdag adag plus adjoint is REQUIRED; Dirac number conservation does not imply excitation-number conservation.')


def native_case(p: RootParameters, gamma: np.ndarray) -> dict[str, Any]:
    # Two onsite four-mode trees, reference bridge and one genuine extra cycle.
    model = BosonicEdgeModel(8, ((0, 1), (0, 2), (0, 3), (0, 4),
                                (4, 5), (4, 6), (4, 7), (1, 5)))
    t, determinant_b, energy = wilson_step(gamma, p.a, p.mass, np.array(p.momentum))
    logt = positive_log(t)
    lifted = native_lift(model, logt)
    transfer = determinant_b*expm(lifted)
    _, code = model.occupation_basis(signs=(1,))
    _, outside = model.occupation_basis(signs=(-1,))
    # Comparator is spectral only, and includes both reference parities.
    actual = np.linalg.eigvalsh(code.conj().T@transfer@code)
    expected = np.repeat(np.linalg.eigvalsh(determinant_b*exterior(t)), 8)
    spectrum_error = float(np.max(np.abs(np.sort(actual)-np.sort(expected))))
    s = model.stabilizers[0]
    penalty = expm(-p.native_cycle_penalty*(model.identity-s)/4)
    # The chord's Z commutes EVERY onsite even SU4 generator and source path,
    # while its anticommutation with the cycle supplies a real Q transition.
    chord = model.edge_index[(1, 5)]
    word = ['I']*len(model.edges)
    word[chord] = 'Z'
    d = tensor_pauli(word)
    half = math.sqrt(determinant_b)*expm(lifted/2)
    full = penalty@half@expm(p.native_defect*d)@half@penalty
    pp, qq = code.conj().T@full@code, outside.conj().T@full@outside
    pq = code.conj().T@full@outside
    generators = [native_lift(model, g)+native_lift(model, g, 4) for g in matrix_unit_basis()]
    source = sum((model.bilinear(c, c+4, path=tuple([c+4]+([4] if c else [])+[0]+([c] if c else [])))
                  for c in range(4)), np.zeros_like(model.identity))
    # Reference filled and actual negative-eigenvalue Wilson sea, no empty-sea upper.
    number_ref = sum((model.n(c) for c in range(4, 8)), np.zeros_like(model.identity))
    ref_filled = model.identity.copy()
    for c in range(4, 8):
        ref_filled = ref_filled@model.n(c)
    nactive = sum((model.n(c) for c in range(4)), np.zeros_like(model.identity))
    sector = code.conj().T@(ref_filled@(nactive-2*model.identity)@(nactive-2*model.identity)+100*(model.identity-ref_filled))@code
    ev, vec = np.linalg.eigh(sector)
    selected = code@vec[:, ev < 1e-8]
    vals, vectors = np.linalg.eigh(selected.conj().T@transfer@selected)
    sea = selected@vectors[:, -1]
    source_norm = float(np.linalg.norm(source@sea)**2)
    normalized = source@sea/math.sqrt(source_norm)
    charge = nactive+number_ref-6*model.identity
    # Primitive CURRENT AND CONTACT are derivatives of the actual positive step,
    # not the earlier nearest-neighbour Hamiltonian or just a copied CAR operator.
    def native_at(theta: np.ndarray) -> np.ndarray:
        ti, bi, _ = wilson_step(gamma, p.a, p.mass, theta)
        return bi*expm(native_lift(model, positive_log(ti)))
    direction = np.array([.3, -.2, .7])
    eps = 2e-4
    center = np.array(p.momentum)
    plus, minus = native_at(center+eps*direction), native_at(center-eps*direction)
    current = (plus-minus)/(2*eps)
    contact = (plus+minus-2*transfer)/(eps*eps)
    small_plus, small_minus = native_at(center+eps*direction/2), native_at(center-eps*direction/2)
    current_small = (small_plus-small_minus)/eps
    contact_small = 4*(small_plus+small_minus-2*transfer)/(eps*eps)
    source_covariance = max(float(np.linalg.norm(g@source-source@g)) for g in generators)
    determinant_defect = float(np.linalg.norm(transfer-expm(lifted)))
    e10 = model.bilinear(1, 0, path=(0, 1))
    e02 = model.bilinear(0, 2, path=(2, 0))
    e12 = model.bilinear(1, 2, path=(2, 0, 1))
    primitive_even_error = float(np.linalg.norm(e10@e02-e02@e10-e12))
    bare10, bare02 = model.hop(0, 1, False), model.hop(2, 0, False)
    commuting_error = float(np.linalg.norm(bare10@bare02-bare02@bare10-e12))
    # Actual TWO quantum photon coordinates, not just a classical Peierls jet.
    # The independent case uses two-level Galerkin field modes; the full
    # oscillator/compact-Haar spectrum is supplied by Photon.
    sigma, photon_energy = .04, .2
    hadamard = np.array([[1., 1.], [1., -1.]])/math.sqrt(2)
    rotate = np.kron(hadamard, hadamard)
    nodes = list(itertools.product((-sigma, sigma), repeat=2))
    quantum_blocks = np.array([native_at(center+np.array([x, y, 0.])) for x, y in nodes])
    quantum = np.einsum('ix,xab,jx->iajb', rotate, quantum_blocks, rotate).reshape(4*model.dimension, 4*model.dimension)
    occupation = np.array([0, 1, 1, 2])
    weights = np.repeat(np.exp(-photon_energy*occupation/2), model.dimension)
    quantum = weights[:, None]*quantum*weights[None, :]
    initial = np.zeros(4*model.dimension, complex)
    initial[:model.dimension] = normalized
    evolved = (quantum@initial).reshape(4, model.dimension)
    quantum_positive_lower = math.exp(-2*photon_energy)*min(float(np.linalg.eigvalsh(block)[0]) for block in quantum_blocks)
    # Literal INTEGER U1 Wilson-link source from separated charge1 Ref hole.
    # Full link is unitary on L2(U1); native interior Galerkin uses no wrapping.
    electric = np.diag(np.arange(-2, 3, dtype=float))
    link_raise = np.diag(np.ones(4), -1).astype(complex)
    flux_zero = np.array([0., 0., 1., 0., 0.])
    flux_final = link_raise@flux_zero
    electric_string_commutator = electric@link_raise-link_raise@electric-link_raise
    separated_state = np.outer(flux_final, source@sea/math.sqrt(source_norm))
    separated_norm = float(np.linalg.norm(separated_state))
    electric_final = float(np.vdot(flux_final, electric@flux_final).real)
    return dict(edge_qubits=len(model.edges), native_full_dimension=model.dimension,
                full_cycle_sector_dimensions=[code.shape[1], outside.shape[1]],
                actual_primitive_transfer_spectrum_error=spectrum_error,
                determinant_B_prefactor=determinant_b, wrong_omitted_prefactor_error=determinant_defect,
                positivity_min=float(np.linalg.eigvalsh(full)[0]),
                P_min=float(np.linalg.eigvalsh(pp)[0]), Q_norm=float(np.linalg.norm(qq, 2)),
                P_Q_transition_norm=float(np.linalg.norm(pq, 2)),
                actual_Q_to_P_spectral_ratio=float(np.linalg.norm(qq, 2)/np.linalg.eigvalsh(pp)[0]),
                defect_cycle_anticommutator=float(np.linalg.norm(d@s+s@d)),
                defect_SU4_commutator=max(float(np.linalg.norm(d@g-g@d)) for g in generators),
                source_SU4_commutator=source_covariance,
                source_EM_commutator=float(np.linalg.norm(charge@source-source@charge)),
                source_norm_squared_on_actual_native_Wilson_sea=source_norm,
                normalized_source_norm=float(np.linalg.norm(normalized)),
                source_active_number=float(np.vdot(normalized, nactive@normalized).real),
                source_Ref_number=float(np.vdot(normalized, number_ref@normalized).real),
                current_norm=float(np.linalg.norm(current)), contact_norm=float(np.linalg.norm(contact)),
                current_stepsize_relative_error=float(np.linalg.norm(current-current_small)/np.linalg.norm(current_small)),
                contact_stepsize_relative_error=float(np.linalg.norm(contact-contact_small)/np.linalg.norm(contact_small)),
                current_charge_commutator=float(np.linalg.norm(charge@current-current@charge)),
                pairing=pairing_control(t, gamma), free_pole_energy_lattice=energy,
                wrong_bare_mass_as_pole_error=abs(math.expm1(p.a*p.mass)-p.a*p.mass),
                exact_primitive_even_matrixunit_error=primitive_even_error,
                wrong_commuting_bare_X_even_algebra_error=commuting_error,
                actual_two_quantum_photon_transfer=dict(full_native_dimension=4*model.dimension,
                    physical_polarizations=2, source_state_norm=float(np.linalg.norm(initial)),
                    actual_positive_transfer_lower=quantum_positive_lower,
                    actual_one_photon_current_transition_norms=[float(np.linalg.norm(evolved[j])) for j in (1, 2)],
                    actual_two_photon_contact_transition_norm=float(np.linalg.norm(evolved[3])),
                    quantum_state_charge='Primitive source-normalized charged free sea tensor normalized two-mode field vacuum; both photon coordinates act as actual Galerkin quantum operators.',
                    transfer='e^(-Hph/2) T_matter(A_operator) e^(-Hph/2). Coordinate spectral calculus uses actual primitive T_matter at four joint nodes, not a source/current mock. Positive product lower follows actual block eigenvalues and field weights.',
                    finite_parameters=dict(coordinate_sigma=sigma, photon_energy=photon_energy),
                    scope='Native discriminating TWO-mode quantum-field case, not the huge induced-beta physical photon spectrum. Same-current packet consumer supplies that spectrum and temporal contact.'),
                actual_integer_U1_dressed_Ref_source=dict(Ref_operator_charge=1,
                    Ref_plus_neutral_background=4, Ref_minus_neutral_background=0,
                    source_normalized_state_norm=separated_norm,
                    actual_body_charge=1, actual_Ref_hole_charge=-1,
                    actual_link_electric_flux=electric_final,
                    exact_integer_Gauss_string_commutator=float(np.linalg.norm(electric_string_commutator))*float(np.linalg.norm(source)),
                    wrong_undressed_separated_source_Gauss_error=float(np.linalg.norm(source)),
                    full_source='sum_c a_cdag(x) [SU4 Wilson path]_(cd) [U1 Wilson path]^1 r_d(xRef)/sqrt2, with the SAME primitive Pauli path; body+1 and Refhole-1. Path integer flux has divergence body-minus-reference; no torus boundary or noninteger Haar phase.',
                    normalization='Filled Ref4 gives trace Wdag(I-rho_active)W=4-<n_active>=2 for ANY unitary SU4/U1 path, so division sqrt2 is exact, including separated source.',
                    native_scope='One-edge separated charge/flux INTERIOR Galerkin proof; full infinite integer rotor and arbitrary primitive paths are the actual family operator. Ref initial neutrality is NOT charge0 of its operator.'),
                finite_law='T_delta=e^(-C/2) T0^1/2 e^(epsilon Z_chord) T0^1/2 e^(-C/2); C=Delta(1-S)/2. All Q states remain. This is positive, not T0+indefinite defect.',
                source_scope='Actual native free-Wilson sea with filled Ref4; SU4-scalar source is exact. This fixed-frame sea itself is not a full-SU4 Gauss singlet; the transported chart section is required.',
                current_scope='Single Bloch-mode quantum-field jet; full spatial Peierls/gauge-current paths follow WilsonConsumer, not a local-spatial continuity proof from this cell.')


def transported_frame(gamma: np.ndarray, photon: Parameters) -> dict[str, Any]:
    basis = matrix_unit_basis()
    generator = sum((math.sin(i+1)*b for i, b in enumerate(basis)), np.zeros((4, 4), complex))
    v = expm(.07j*generator)
    gram = np.array(photon.gram)
    phi = np.sqrt(gram)[:, None, None]*SELECTED
    moved = np.array([v@x@v.conj().T for x in phi])
    moved_gamma = np.array([v@g@v.conj().T for g in gamma])
    beta = moved[0]/math.sqrt(gram[0])
    reconstructed = np.array([beta]+[1j*beta@moved[j]/math.sqrt(gram[j]) for j in (1, 2, 3)])
    tangent = np.column_stack([(1j*np.array([b@x-x@b for x in phi])).reshape(-1) for b in basis])
    singular = np.linalg.svd(np.vstack((tangent.real, tangent.imag)), compute_uv=False)
    clifford = max(float(np.linalg.norm(reconstructed[i]@reconstructed[j]+reconstructed[j]@reconstructed[i]-(2*np.eye(4) if i == j else 0))) for i, j in itertools.product(range(4), repeat=2))
    return dict(full_Phi_cartesian_coordinates=60, actual_SU4_orbit_rank=int(np.sum(singular > 1e-10)),
                normal_coordinates=45, tangent_min_singular_value=float(min(singular)),
                transported_Phi_gamma_error=float(np.linalg.norm(reconstructed-moved_gamma)),
                actual_transported_Clifford_error=clifford,
                selected_Gram_is_declared_new_law_centre=True,
                continuous_time_Phi_mean_imported=False,
                chart='Phi=V(Phi_star+Y)Vdag with Y normal to the 15-dimensional SU4 orbit; V modulo Z4. Fermion Fock fibre transforms by Gamma(V), not an independent spin4 tensor colour4.',
                chart_domain='Only the open tubular chart is normalized. Outside uses e^(-H_out) with full Cartesian Phi kinetic, positive quartic and actual linear Phi even-Fock couplings; it is not an identity/scalar/no-op replacement for a negative Wilson B.',
                tube_reach_lower=min(float(min(gram))/(512*float(np.linalg.norm(phi))), math.sqrt(float(min(gram)))/64),
                tube_reach_derivation='For any unitary V, sum_mu ||[V,SELECTED_mu]||F^2>=4||V-Tr(V)I/4||F^2, by the complete Clifford-grade channel spectrum. Near a centre element write V=z exp(iA), traceless Hermitian A: normal chord<=16||Phi_star||F ||A||F^2 and full chord squared>=gmin||A||F^2. Away from this neighbourhood chord>=sqrt(gmin)/32. Thus conservative reach min[gmin/(512||Phi_star||),sqrt(gmin)/64] is smaller than both estimates. The explicit configured radius must be inside it.',
                kinetic_metric='NEW physical-component product Friedrichs law: Q=Gamma(V), Dphys=d-(dQ)Qdag=Q d Qdag, Aphys=Q(d-dlogk)Qdag; BOTH kinetic and matter potential are in this SAME physical frame. Implemented PhysicalComponentFrame and physical_component_friedrichs_form. Outside retains full60 Cartesian kinetic; cross kernels are physical-component operators, not an unmatched coordinate kernel.',
                chart_source='The Z4 ambiguity in V is compensated by real centre-character colour-link packets. Adjoint Phi alone cannot carry that charge.')


def physical_component_geometry_native(p: RootParameters, gamma: np.ndarray,
                                       photon: Parameters) -> dict[str, Any]:
    """Execute the NEW physical kinetic, source kernels and noncommuting jets."""
    model = BosonicEdgeModel(8, ((0, 1), (0, 2), (0, 3), (0, 4),
                                (4, 5), (4, 6), (4, 7), (1, 5)))
    basis = matrix_unit_basis()
    a = basis[0]+.27*basis[3]+.15*basis[14]
    b = basis[1]+.31*basis[7]
    ar, br = .61*b+.13*a, .37*a-.19*b
    ga = native_lift(model, a)+native_lift(model, ar, 4)
    gb = native_lift(model, b)+native_lift(model, br, 4)
    noncommuting = float(np.linalg.norm(a@b-b@a))
    if noncommuting < .1:
        raise ArithmeticError('Physical-component jet is not a noncommuting SU4 curve')
    def curve(t: float) -> tuple[PhysicalComponentFrame, np.ndarray, np.ndarray]:
        left, right = expm(1j*t*ga), expm(1j*t*t*gb)
        q = left@right
        dq = 1j*ga@q+left@(2j*t*gb)@right
        return PhysicalComponentFrame(q, dq), expm(1j*t*a)@expm(1j*t*t*b), expm(1j*t*ar)@expm(1j*t*t*br)
    number_body = sum((model.n(c) for c in range(4)), np.zeros_like(model.identity))
    number_ref = sum((model.n(c) for c in range(4, 8)), np.zeros_like(model.identity))
    _, code = model.occupation_basis(signs=(1,))
    def canonical(nb: int, nr: int) -> np.ndarray:
        keep = np.diag((np.abs(np.diag(number_body)-nb) < 1e-8)
                       & (np.abs(np.diag(number_ref)-nr) < 1e-8))
        vectors, values, _ = np.linalg.svd(keep@code, full_matrices=False)
        return vectors[:, values > .5]
    neutral, charged = canonical(2, 4), canonical(3, 3)
    t1, detb, _ = wilson_step(gamma, p.a, p.mass, np.array(p.momentum))
    logt = positive_log(t1)
    t0 = detb*expm(native_lift(model, logt))
    w0 = expm(.21j*b)
    def path_source(w: np.ndarray) -> np.ndarray:
        result = np.zeros_like(model.identity)
        for c, d in itertools.product(range(4), repeat=2):
            ref = 4+d
            path = tuple(([ref] if ref != 4 else [])+[4, 0]+([c] if c else []))
            result += w[c, d]*model.bilinear(c, ref, path=path)/math.sqrt(2)
        return result
    s0 = path_source(w0)
    tn, tc = neutral.conj().T@t0@neutral, charged.conj().T@t0@charged
    eig, vectors = np.linalg.eigh(tn)
    v0 = neutral@vectors[:, -1]
    chord_word = ['I']*len(model.edges)
    chord_word[model.edge_index[(1, 5)]] = 'Z'
    chord = tensor_pauli(chord_word)
    rows = []
    phi = np.sqrt(np.array(photon.gram))[:, None, None]*SELECTED
    cartesian = np.zeros((60, 4, 4, 4), complex)
    for mu, g in itertools.product(range(4), range(15)):
        cartesian[15*mu+g, mu] = basis[g]
    tangent = np.column_stack([
        np.einsum('aijk,ijk->a', cartesian.conj(), 1j*(g@phi-phi@g)).real/4
        for g in basis])
    normal_basis = np.linalg.qr(tangent, mode='complete')[0][:, 15:]
    normal_coordinates = normal_basis@np.cos(np.arange(45)+.3)
    remainder = .002*np.einsum('a,aijk->ijk', normal_coordinates, cartesian)/np.linalg.norm(normal_coordinates)
    links0 = tuple(expm(1j*(.013+j*.007)*(basis[j]+.2*basis[j+4])) for j in range(3))
    phi_t0, _, _ = phi_wilson_step(gamma, .09, p.momentum, remainder, links0)
    theta = wilson_component_basis(gamma)
    pure_t, _, _ = phi_wilson_step(gamma, .09, p.momentum, np.zeros_like(remainder), (np.eye(4),)*3)
    pure_mass = math.log(1/(2*.09)-3)/p.a
    pure_reference_t, _, _ = wilson_step(gamma, p.a, pure_mass, np.array(p.momentum))
    basis_controls = dict(
        Theta_unitarity_error=float(np.linalg.norm(theta.conj().T@theta-np.eye(4))),
        Theta_SU4_determinant_error=float(abs(np.linalg.det(theta)-1)),
        selected_gamma0_eigenspace_embedding_error=float(np.linalg.norm(theta.conj().T@gamma[0]@theta-np.diag([1,1,-1,-1]))),
        pure_Clifford_field_physical_Wilson_error=float(np.linalg.norm(pure_t-pure_reference_t)))
    if max(basis_controls.values()) > 1e-10:
        raise ArithmeticError('Physical reference Theta embedding is not the actual Wilson law')
    for t in (-.31, .17, .43):
        jet, va, vr = curve(t)
        q, dq = jet.q, jet.dq
        physical_t = jet.operator(t0)
        ordinary_t = dq@t0@q.conj().T+q@t0@dq.conj().T
        physical_s = jet.source(s0, jet)
        ordinary_s = dq@s0@q.conj().T+q@s0@dq.conj().T
        direct_t = detb*expm(native_lift(model, va@logt@va.conj().T))
        rows.append(dict(parameter=t,
            Q_unitarity_error=float(np.linalg.norm(q.conj().T@q-model.identity)),
            Dminus_Q_ground_error=float(np.linalg.norm(jet.covariant_derivative(q@v0, dq@v0))),
            Dminus_operator_jet_error=float(np.linalg.norm(jet.covariant_operator_derivative(physical_t, ordinary_t))),
            Dminus_source_jet_error=float(np.linalg.norm(jet.covariant_source_derivative(physical_s, ordinary_s, jet))),
            actual_primitive_potential_frame_error=float(np.linalg.norm(direct_t-physical_t)),
            actual_integer_Wilson_source_frame_error=float(np.linalg.norm(path_source(va@w0@vr.conj().T)-physical_s)),
            physical_transfer_ordinary_derivative_norm=float(np.linalg.norm(ordinary_t)),
            physical_source_ordinary_derivative_norm=float(np.linalg.norm(ordinary_s)),
            statistical_cycle_descent_error=float(np.linalg.norm(q@model.stabilizers[0]-model.stabilizers[0]@q)),
            spatial_Z_descent_error=float(np.linalg.norm(q@chord-chord@q)),
            canonical_body_ref_range_error=float(np.linalg.norm(q@number_body-number_body@q)+np.linalg.norm(q@number_ref-number_ref@q)),
            world_connection_Hermiticity_error=float(np.linalg.norm(jet.world_connection_generator()-jet.world_connection_generator().conj().T)),
            actual_world_connection_on_parallel_ground=float(np.linalg.norm(jet.world_connection_generator()@q@v0))))
        dva = 1j*a@va+expm(1j*t*a)@(2j*t*b)@expm(1j*t*t*b)
        dvr = 1j*ar@vr+expm(1j*t*ar)@(2j*t*br)@expm(1j*t*t*br)
        direct_source_derivative = path_source(dva@w0@vr.conj().T+va@w0@dvr.conj().T)
        onebody = PhysicalComponentFrame(va, dva)
        moved_phi = np.array([va@z@va.conj().T for z in phi+remainder])
        moved_y = np.array([va@z@va.conj().T for z in remainder])
        dphi = np.array([dva@z@va.conj().T+va@z@dva.conj().T for z in phi+remainder])
        dy = np.array([dva@z@va.conj().T+va@z@dva.conj().T for z in remainder])
        direct_phi_t, _, _ = phi_wilson_step(
            np.array([va@g@va.conj().T for g in gamma]), .09, p.momentum,
            moved_y, tuple(va@u@va.conj().T for u in links0))
        physical_phi_t = onebody.operator(phi_t0)
        dphi_t = dva@phi_t0@va.conj().T+va@phi_t0@dva.conj().T
        step = 1e-5
        direct_transfers = []
        for shifted in (t-step, t+step):
            rotated = expm(1j*shifted*a)@expm(1j*shifted*shifted*b)
            value, _, _ = phi_wilson_step(
                np.array([rotated@g@rotated.conj().T for g in gamma]), .09,
                p.momentum, np.array([rotated@y@rotated.conj().T for y in remainder]),
                tuple(rotated@u@rotated.conj().T for u in links0))
            direct_transfers.append(value)
        direct_phi_derivative = (direct_transfers[1]-direct_transfers[0])/(2*step)
        rows[-1].update(
            actual_nonclifford_Phi_link_potential_covariance_error=float(np.linalg.norm(direct_phi_t-physical_phi_t)),
            actual_Phi_adjoint_connection_jet_error=max(float(np.linalg.norm(onebody.covariant_operator_derivative(z, dz))) for z, dz in zip(moved_phi, dphi)),
            actual_Phi_positive_T_connection_jet_error=float(np.linalg.norm(onebody.covariant_operator_derivative(physical_phi_t, dphi_t))),
            actual_integer_Wilson_source_derivative_error=float(np.linalg.norm(direct_source_derivative-ordinary_s)),
            actual_Phi_link_transfer_direct_derivative_error=float(np.linalg.norm(direct_phi_derivative-dphi_t)),
            normal_radial_domain_tangency_error=float(abs(np.vdot(moved_y, dy).real)),
            normal_radial_domain_invariance_error=float(abs(np.linalg.norm(moved_y)-np.linalg.norm(remainder))),
            joint_colour_link_domain_pullback_error=max(float(np.linalg.norm(va.conj().T@(va@u@va.conj().T)@va-u)) for u in links0))
    error_keys = [key for key in rows[0] if key.endswith('_error')]
    if max(row[key] for row in rows for key in error_keys) > 1e-8:
        raise ArithmeticError('Implemented physical-component kinetic/source jet failed')
    if min(row['physical_source_ordinary_derivative_norm'] for row in rows) < .01:
        raise ArithmeticError('Physical source jet is an inert frame surrogate')
    # Actual ground-factorized radial kinetic and potential in the SAME frame.
    nodes, radius = 7, .4
    x = np.linspace(-.75*radius, .75*radius, nodes)
    dx = float(x[1]-x[0])
    ground = np.exp(-x*x/4-x*x/(radius*radius*(radius*radius-x*x)))
    ground /= np.linalg.norm(ground)
    derivative = np.zeros((nodes-1, nodes))
    for j in range(nodes-1):
        derivative[j, j] = math.sqrt(ground[j+1]/ground[j])/dx
        derivative[j, j+1] = -math.sqrt(ground[j]/ground[j+1])/dx
    log_lambda0 = float(np.log(eig[-1]))
    hn = -positive_log(tn)+log_lambda0*np.eye(len(tn))
    hc = -positive_log(tc)+log_lambda0*np.eye(len(tc))
    full_nodes = tuple(curve(float(y))[0].q for y in x)
    full_edges = tuple(curve(float(y))[0].q for y in (x[:-1]+x[1:])/2)
    node_n = tuple(neutral.conj().T@q@neutral for q in full_nodes)
    edge_n = tuple(neutral.conj().T@q@neutral for q in full_edges)
    node_c = tuple(charged.conj().T@q@charged for q in full_nodes)
    edge_c = tuple(charged.conj().T@q@charged for q in full_edges)
    normal_h = .02*derivative.T@derivative
    form_n = physical_component_friedrichs_form(node_n, edge_n,
        math.sqrt(.02)*np.kron(derivative, np.eye(len(tn))), np.kron(np.eye(nodes), hn))
    form_c = physical_component_friedrichs_form(node_c, edge_c,
        math.sqrt(.02)*np.kron(derivative, np.eye(len(tc))), np.kron(np.eye(nodes), hc))
    un, uc = form_n['component_unitary'], form_c['component_unitary']
    h0n = np.kron(normal_h, np.eye(len(tn)))+np.kron(np.eye(nodes), hn)
    h0c = np.kron(normal_h, np.eye(len(tc)))+np.kron(np.eye(nodes), hc)
    half_n, half_c = expm(-form_n['Hamiltonian']/2), expm(-form_c['Hamiltonian']/2)
    half0n, half0c = expm(-h0n/2), expm(-h0c/2)
    phi0 = np.kron(np.outer(ground, ground), np.eye(len(tn)))
    phiphys = un@phi0@un.conj().T
    sn = charged.conj().T@s0@neutral
    source0 = np.kron(np.eye(nodes), sn)
    source_phys = uc@source0@un.conj().T
    # Literal TWO charged halfsteps, neutral boundary, and positive field filter.
    photon_filter = np.kron(np.diag(1+.3*x*x), np.eye(len(tc)))
    mphys = half_n@source_phys.conj().T@half_c@photon_filter@half_c@source_phys@half_n
    # Ref0->1/body2->1 uses the SAME neutral lambda0, never the charged top.
    neutral_minus, charged_minus = canonical(2, 0), canonical(1, 1)
    tnm = neutral_minus.conj().T@t0@neutral_minus
    tcm = charged_minus.conj().T@t0@charged_minus
    snm = charged_minus.conj().T@s0.conj().T@neutral_minus
    node_nm = tuple(neutral_minus.conj().T@q@neutral_minus for q in full_nodes)
    edge_nm = tuple(neutral_minus.conj().T@q@neutral_minus for q in full_edges)
    node_cm = tuple(charged_minus.conj().T@q@charged_minus for q in full_nodes)
    edge_cm = tuple(charged_minus.conj().T@q@charged_minus for q in full_edges)
    hnm = -positive_log(tnm)+log_lambda0*np.eye(len(tnm))
    hcm = -positive_log(tcm)+log_lambda0*np.eye(len(tcm))
    form_nm = physical_component_friedrichs_form(node_nm, edge_nm,
        math.sqrt(.02)*np.kron(derivative, np.eye(len(tnm))), np.kron(np.eye(nodes), hnm))
    form_cm = physical_component_friedrichs_form(node_cm, edge_cm,
        math.sqrt(.02)*np.kron(derivative, np.eye(len(tcm))), np.kron(np.eye(nodes), hcm))
    unm, ucm = form_nm['component_unitary'], form_cm['component_unitary']
    half_nm, half_cm = expm(-form_nm['Hamiltonian']/2), expm(-form_cm['Hamiltonian']/2)
    half0nm = expm(-(np.kron(normal_h, np.eye(len(tnm)))+np.kron(np.eye(nodes), hnm))/2)
    half0cm = expm(-(np.kron(normal_h, np.eye(len(tcm)))+np.kron(np.eye(nodes), hcm))/2)
    source0m = np.kron(np.eye(nodes), snm)
    source_physm = ucm@source0m@unm.conj().T
    photon_filterm = np.kron(np.diag(1+.3*x*x), np.eye(len(tcm)))
    mphysm = half_nm@source_physm.conj().T@half_cm@photon_filterm@half_cm@source_physm@half_nm
    m0m = half0nm@source0m.conj().T@half0cm@photon_filterm@half0cm@source0m@half0nm
    m0 = half0n@source0.conj().T@half0c@photon_filter@half0c@source0@half0n
    kphys, k0 = half_n@half_n, half0n@half0n
    xi = half_n@phiphys@np.linalg.matrix_power(kphys, 2)
    xi0 = half0n@phi0@np.linalg.matrix_power(k0, 2)
    transported_ground = un@np.kron(ground, vectors[:, -1])
    mixed_h = form_n['kinetic']+np.kron(np.eye(nodes), hn)
    form_errors = dict(
        kinetic_same_frame_error=float(np.linalg.norm(form_n['kinetic']-un@np.kron(normal_h, np.eye(len(tn)))@un.conj().T)),
        Hamiltonian_same_frame_error=float(np.linalg.norm(form_n['Hamiltonian']-un@h0n@un.conj().T)),
        heat_same_frame_error=float(np.linalg.norm(half_n-un@half0n@un.conj().T)),
        parallel_radial_ground_form_energy=float(np.vdot(transported_ground, form_n['Hamiltonian']@transported_ground).real),
        two_charged_halfstep_source_kernel_error=float(np.linalg.norm(mphys-un@m0@un.conj().T)),
        radial_boundary_Xi_frame_error=float(np.linalg.norm(xi-un@xi0@un.conj().T)),
        normalized_source_kernel_trace_error=abs(float(np.trace(mphys).real/np.trace(m0).real)-1),
        normalized_heat_power_trace_error=max(abs(float(np.trace(np.linalg.matrix_power(kphys, order)).real/np.trace(np.linalg.matrix_power(k0, order)).real)-1) for order in (2, 8)))
    if max(abs(value) for value in form_errors.values()) > 1e-8:
        raise ArithmeticError('Physical A†A kinetic, potential or positive source kernel mismatch')
    minus_error = float(np.linalg.norm(mphysm-unm@m0m@unm.conj().T))
    if minus_error > 1e-8:
        raise ArithmeticError('Ref0/body-minus source violates same-neutral-lambda0 frame kernel')
    mixed_energy = float(np.vdot(transported_ground, mixed_h@transported_ground).real)
    if mixed_energy <= 1e-5:
        raise ArithmeticError('Untransported potential failed to expose mixed-frame kinetic defect')
    # Centre descent uses real character packets, not matter-only Γ(z).
    total4 = np.diag((np.abs(np.diag(number_body+number_ref)-4) < 1e-8).astype(float))
    seed = total4@code[:, :].sum(axis=1)
    seed /= np.linalg.norm(seed)
    nb = np.rint(np.diag(number_body).real).astype(int)
    nr = np.rint(np.diag(number_ref).real).astype(int)
    packet = np.zeros((model.dimension, 4), complex)
    packet[np.arange(model.dimension), nb % 4] = seed
    centre_error = 0.
    wrong_centre = 0.
    for zb, zr in itertools.product((1, 1j, -1, -1j), repeat=2):
        matter_phase = zb**nb*zr**nr
        character_phase = np.array([(np.conj(zb)*zr)**q for q in range(4)])
        centre_error = max(centre_error, float(np.linalg.norm(matter_phase[:, None]*packet*character_phase[None, :]-packet)))
        wrong_centre = max(wrong_centre, float(np.linalg.norm(matter_phase[:, None]*packet-packet)))
    jet, va, _ = curve(.17)
    pure_c_errors = []
    for scale in (.73, 1.2):
        scaled = scale*np.array([va@z@va.conj().T for z in phi])
        beta = scaled[0]/(scale*math.sqrt(photon.gram[0]))
        recovered = np.array([beta]+[1j*beta@scaled[j]/(scale*math.sqrt(photon.gram[j])) for j in (1, 2, 3)])
        pure_c_errors.append(float(np.linalg.norm(recovered-np.array([va@g@va.conj().T for g in gamma]))))
    # C is a field/statistical projector, not a Fock-frame drive.
    pure_c = PhysicalComponentFrame(jet.q, np.zeros_like(jet.q))
    pure_c_connection = float(np.linalg.norm(pure_c.connection))
    if centre_error > 1e-10 or wrong_centre < .1 or max(pure_c_errors) > 1e-10:
        raise ArithmeticError('Physical-component centre or pure-scale descent failed')
    plus_error = float(np.linalg.norm(number_body@s0-s0@number_body-s0))
    ref_error = float(np.linalg.norm(number_ref@s0-s0@number_ref+s0))
    if max(plus_error, ref_error) > 1e-10:
        raise ArithmeticError('Actual signed primitive source lost body+1/Refhole-1 charge')
    return dict(connection_law=PHYSICAL_COMPONENT_CONNECTION_LAW, connection_implemented=True,
        native_primitive_dimension=model.dimension, neutral_canonical_dimension=len(tn), charged_canonical_dimension=len(tc),
        noncommuting_SU4_generator_norm=noncommuting, curve_rows=rows, form_controls=form_errors,
        physical_reference_basis_ABI='T4=Theta Tblock ThetaDag; Theta columns(+gamma0,-gamma0), detTheta=1; bp,bm are eigenspace blocks, not raw-component matrices',
        actual_reference_basis_controls=basis_controls,
        actual_native_normal_domain=dict(cartesian_dimensions=60,normal_dimensions=45,
            normal_rank=int(np.linalg.matrix_rank(normal_basis)),
            same_radial_colour_faces_and_intersections=True,component_box_used=False),
        actual_mixed_frame_potential_ground_energy=mixed_energy,
        actual_physical_kinetic_change_norm=float(np.linalg.norm(form_n['kinetic']-np.kron(normal_h, np.eye(len(tn))))),
        actual_signed_canonical_source_kernels=dict(
            plus=dict(Ref_transition='4->3',body_transition='2->3',two_charged_halfsteps=True,
                      kernel_trace=float(np.trace(mphys).real),frame_error=form_errors['two_charged_halfstep_source_kernel_error']),
            minus=dict(Ref_transition='0->1',body_transition='2->1',two_charged_halfsteps=True,
                       kernel_trace=float(np.trace(mphysm).real),frame_error=minus_error),
            common_neutral_log_lambda0=log_lambda0,charged_top_used_as_normalization=False),
        actual_native_positive_source_kernel_trace=float(np.trace(mphys).real),
        actual_centre_character_descent_error=centre_error, wrong_matter_only_centre_descent_error=wrong_centre,
        actual_pure_C_normalized_Phi_frame_errors=pure_c_errors, actual_pure_C_connection_norm=pure_c_connection,
        actual_source_body_plus1_error=plus_error,
        actual_source_Ref_hole_minus1_error=ref_error,
        radial_projector='Pphi_phys=Q Pphi_coordinate Qdag; identity on ALL centres and statistical sectors. Canonical Xi uses this transported projector, not a scalar projector in mismatched physical components.',
        continuum_form='Dphys=d-(dQ)Qdag; Aphys=Q(d-dlogk)Qdag. On the physical bundle Q includes centre-character compensation, carries the Friedrichs/zero-trace domain, and transports BOTH kinetic and potential. Hphys=Q Hcoordinate Qdag is the NEW declared law; it is not an equivalence to the old Dplus Hamiltonian.',
        source_API='PhysicalComponentFrame.source(S0,neutral_frame), covariant_source_derivative; rectangular charged/neutral canonical ranges and TWO charged positive halfsteps are kept. Source body+1/Refhole-1 and photon/receiver functional recipes unchanged.',
        source_scope='Executed native canonical algebra/positive kernel, not a full-SU4 inserted-Haar macro Gram theorem; full Gauss wedge2-link and domain controls are independently retained.',
        reported_historical_defect=dict(owner='Centre/Main',source_epoch='ec0730f1/94474467',Dplus_defect=1.59387,Dminus_error=1.25e-16,reexecuted=False),
        scientific_admission=False)

def phi_wilson_step(gamma: np.ndarray, kappa: float, momentum: tuple[float, ...],
                    remainder: np.ndarray, colour_links: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical-reference positive T; Bplus/Bminus remain eigenspace blocks."""
    theta = wilson_component_basis(gamma)
    plus, minus = theta[:, :2], theta[:, 2:]
    links = [expm(1j*p*np.eye(4))@u for p, u in zip(momentum, colour_links)]
    b = np.eye(4)/(2*kappa)-sum((u+u.conj().T for u in links), np.zeros((4, 4), complex))/2
    b += kappa*remainder[0]+kappa*np.linalg.norm(remainder)**2*np.eye(4)
    bp, bm = plus.conj().T@b@plus, minus.conj().T@b@minus
    c = sum((g@u-u.conj().T@g for g, u in zip(gamma[1:], links)), np.zeros((4, 4), complex))/2
    c += kappa*remainder[0]+kappa*sum((y@u-u.conj().T@y for y, u in zip(remainder[1:], links)), np.zeros((4, 4), complex))/2
    f = plus.conj().T@c@minus
    l = np.block([[np.eye(2), -f], [np.zeros((2, 2)), np.eye(2)]])
    block = l.conj().T@np.block([[np.linalg.inv(bp), np.zeros((2, 2))],
                               [np.zeros((2, 2)), bm]])@l
    return theta@block@theta.conj().T, bp, bm


def nonclifford_chart_native(gamma: np.ndarray, photon: Parameters) -> dict[str, Any]:
    """Explicit noncommuting colour/normal positive completion, no global gamma."""
    phi = np.sqrt(np.array(photon.gram))[:, None, None]*SELECTED
    generators = matrix_unit_basis()
    tangent = np.column_stack([(1j*np.array([g@x-x@g for x in phi])).reshape(-1) for g in generators])
    real_tangent = np.vstack((tangent.real, tangent.imag))
    candidate = np.array([sum((math.sin((mu+1)*(j+1))*g for j, g in enumerate(generators)), np.zeros((4, 4), complex)) for mu in range(4)], dtype=complex)
    vector = candidate.reshape(-1)
    real_vector = np.concatenate((np.real(vector), np.imag(vector)))
    normal = real_vector-real_tangent@np.linalg.lstsq(real_tangent, real_vector, rcond=None)[0]
    normal *= 1e-5/np.linalg.norm(normal)
    y = (normal[:64]+1j*normal[64:]).reshape(4, 4, 4)
    identity_links = (np.eye(4), np.eye(4), np.eye(4))
    colour = expm(.03j*sum((math.sin(j+1)*g for j, g in enumerate(generators)), np.zeros((4, 4), complex)))
    kappa = 1/(2*(math.expm1(.01)+4))
    t, bp, bm = phi_wilson_step(gamma, kappa, (.3, .2, .1), y, (colour, *identity_links[1:]))
    t0, _, _ = phi_wilson_step(gamma, kappa, (.3, .2, .1), np.zeros_like(y), (colour, *identity_links[1:]))
    arbitrary = phi+y
    normalized = arbitrary/np.sqrt(np.einsum('mij,mji->m', arbitrary, arbitrary).real/4)[:, None, None]
    naive = np.array([normalized[0]]+[1j*normalized[0]@normalized[j] for j in (1, 2, 3)])
    defect = max(float(np.linalg.norm(naive[i]@naive[j]+naive[j]@naive[i]-(2*np.eye(4) if i == j else 0)))
                 for i, j in itertools.product(range(4), repeat=2))
    return dict(actual_normal_coordinate_norm=float(np.linalg.norm(y)),
                actual_normal_tangent_inner_error=float(np.linalg.norm(real_tangent.T@normal)),
                actual_noncommuting_colour_gamma0=float(np.linalg.norm(colour@gamma[0]-gamma[0]@colour)),
                actual_positive_transfer_min=float(np.linalg.eigvalsh(t)[0]),
                actual_Bplus_Bminus_difference=float(np.linalg.norm(bp-bm)),
                actual_full_Phi_remainder_transfer_change=float(np.linalg.norm(t-t0)),
                actual_gauge_dependent_prefactor=float(np.linalg.det(bp).real),
                wrong_arbitrary_normalization_Clifford_defect=defect,
                functional_form='Actual Bplus/Bminus and offdiagonal F construction specified in the full family; finite normal1e-5 is BELOW the quantitative tube reach, colour link is not assumed to commute with gamma0.',
                scope='Explicit finite material-completion comparator. Full-family radius/width are taken from the configuration-bound symbolic budgets, not rounded to zero here; this finite1e-5 case is not the huge-family state.')

def gauss_bundle_native() -> dict[str, Any]:
    """Actual nonzero full-SU4 trial with a real wedge2 Wilson-link Hilbert."""
    pairs = list(itertools.combinations(range(4), 2))
    c = car_comparator()
    selector = np.eye(16)[:, [sum(1 << i for i in pair) for pair in pairs]]
    generators = [selector.conj().T@sum((g[i, j]*c[i].conj().T@c[j]
                   for i, j in itertools.product(range(4), repeat=2)), np.zeros((16, 16), complex))@selector
                  for g in matrix_unit_basis()]
    epsilon = np.zeros((6, 6), complex)
    for i, pair in enumerate(pairs):
        complement = tuple(j for j in range(4) if j not in pair)
        sequence = (*pair, *complement)
        inversions = sum(sequence[a] > sequence[b] for a in range(4) for b in range(a+1, 4))
        epsilon[i, pairs.index((complement[0], complement[1]))] = (-1)**inversions
    # Peter-Weyl sqrt(6) D_wedge2(U) matrix elements form a genuine36-dimensional
    # link subspace of L2(SU4,Haar), NOT an extra fermionic colour multiplicity.
    state = np.einsum('ia,jb->ijab', np.eye(6), epsilon)/6
    errors = []
    for g in generators:
        left = np.einsum('ik,kjab->ijab', g, state)-np.einsum('ak,ijkb->ijab', g.T, state)
        right = np.einsum('jk,ikab->ijab', g, state)+np.einsum('bk,ijak->ijab', g, state)
        errors.extend((float(np.linalg.norm(left)), float(np.linalg.norm(right))))
    # Transport actual Peter-Weyl links AND matter in one Gauss representation.
    la, lb = generators[0]+.27*generators[3]+.15*generators[14], generators[1]+.31*generators[7]
    ra, rb = .61*lb+.13*la, .37*la-.19*lb
    ql = expm(.17j*la)@expm(.17*.17j*lb)
    qr = expm(.17j*ra)@expm(.17*.17j*rb)
    physical_state = np.einsum('ik,jl,am,bn,klmn->ijab', ql, qr, ql.conj(), qr, state)
    physical_errors = []
    for g in generators:
        gl, gr = ql@g@ql.conj().T, qr@g@qr.conj().T
        left = np.einsum('ik,kjab->ijab', gl, physical_state)-np.einsum('ak,ijkb->ijab', gl.T, physical_state)
        right = np.einsum('jk,ikab->ijab', gr, physical_state)+np.einsum('bk,ijak->ijab', gr, physical_state)
        physical_errors.extend((float(np.linalg.norm(left)), float(np.linalg.norm(right))))
    occupations = []
    for color in range(4):
        values = np.array([int(color in pair) for pair in pairs])
        occupations.append(float(np.sum(np.abs(state)**2*values[:, None, None, None])))
    centre_single = sum((1j**(2*j) for j in range(4)))/4
    centre_pair = sum((1j**(4*j) for j in range(4)))/4
    return dict(actual_SU4_link_representation='wedge2 left/right Peter-Weyl matrix elements',
                actual_link_subspace_dimension=36, actual_site_wedge2_dimensions=[6, 6],
                normalized_full_Gauss_state_norm=float(np.linalg.norm(state)),
                actual_all15_left_right_Gauss_generator_error=max(errors),
                actual_physical_component_all15_Gauss_error=max(physical_errors),
                actual_physical_component_PeterWeyl_state_norm=float(np.linalg.norm(physical_state)),
                matter_occupations_per_color=occupations,
                colocated_Ref_plus_source_norm_squared=sum(1-x for x in occupations),
                colocated_Ref_minus_source_norm_squared=sum(occupations),
                actual_link_electric_Casimir=2.5,
                wrong_odd_cell_adjoint_only_centre_projection_norm=abs(centre_single),
                even_dimer_centre_projection_norm=abs(centre_pair),
                scope='Actual full-SU4 number-sector trial/positive control, not a substituted filled Wilson sea. The prepared free sea uses the centre-character bundle below.')

def centre_current_native(gamma: np.ndarray) -> dict[str, Any]:
    """Actual Z4 Gauss link states, cycle coherence and Wilson centre branches."""
    assignments = list(itertools.product(range(4), repeat=3))
    edges = ((0, 1), (1, 2), (0, 2))
    def divergence(q: tuple[int, ...]) -> np.ndarray:
        result = np.zeros(3, dtype=int)
        for value, (source, target) in zip(q, edges):
            result[source] += value
            result[target] -= value
        return result % 4
    def packet(n: np.ndarray, tree_only: bool = False) -> np.ndarray:
        allowed = np.array([np.array_equal(divergence(q), n % 4) and (not tree_only or q[2] == 0)
                            for q in assignments], float)
        return allowed/math.sqrt(float(allowed@allowed))
    charge = np.array([1, 2, 1])
    state = packet(charge)
    moved_charge = charge.copy()
    moved_charge[0] -= 1
    moved_charge[2] += 1
    # Physical 0->2 current shifts the ACTUAL chord character q02 by -1.
    shift = np.zeros((64, 64), complex)
    for column, q in enumerate(assignments):
        moved = (*q[:2], (q[2]-1) % 4)
        shift[assignments.index(moved), column] = 1
    coherent_overlap = abs(np.vdot(packet(moved_charge), shift@state))
    tree_overlap = abs(np.vdot(packet(moved_charge, True), shift@packet(charge, True)))
    wrong = packet((-charge) % 4)
    wrong_gauss = max(abs(1j**int(charge[x]-divergence(q)[x])-1)
                      for amplitude, q in zip(wrong, assignments) if amplitude for x in range(3))
    # Actual Fourier transform of all cycle characters annihilates nonflat
    # centre holonomy. Tree packets alone do NOT perform this projection.
    fourier = np.array([[1j**sum(qi*zi for qi, zi in zip(q, z))/8
                         for q in assignments] for z in assignments])
    centre_state = fourier@state
    nonflat = np.array([(z[0]+z[1]-z[2]) % 4 != 0 for z in assignments])
    uncorrelated = fourier@packet(charge, True)
    _, basis = np.linalg.eigh(gamma[0])
    gpm = basis[:, 2:].conj().T@gamma[1]@basis[:, :2]
    def step(phases: np.ndarray) -> np.ndarray:
        transport = np.zeros((3, 3), complex)
        for x in range(3):
            transport[x, (x+1) % 3] = phases[x]
        b = np.kron(np.eye(2), (2+math.expm1(.01))*np.eye(3)-(transport+transport.conj().T)/2)
        f = np.kron(gpm, (transport-transport.conj().T)/2)
        l = np.block([[np.eye(6), -f], [np.zeros((6, 6)), np.eye(6)]])
        return l.conj().T@np.block([[np.linalg.inv(b), np.zeros((6, 6))],
                                  [np.zeros((6, 6)), b]])@l
    baseline = step(np.ones(3))
    covariance = 0.
    wrong_pole = 0.
    for z in itertools.product(range(4), repeat=3):
        g = 1j**np.array(z)
        phase = g*np.conj(np.roll(g, -1))
        transform = np.kron(np.eye(4), np.diag(g))
        transported = step(phase)
        covariance = max(covariance, float(np.linalg.norm(transported-transform@baseline@transform.conj().T)))
        unadjusted = phase.copy()
        unadjusted[-1] = 1
        wrong_pole = max(wrong_pole, float(np.linalg.norm(np.linalg.eigvalsh(step(unadjusted))-np.linalg.eigvalsh(baseline))))
    return dict(actual_centre_quantum_Hilbert=64, Gauss_cycle_packet_norm=float(np.linalg.norm(state)),
                actual_chord_current_overlap=coherent_overlap,
                wrong_fixed_tree_current_overlap=tree_overlap,
                wrong_subtree_sign_Gauss_phase_error=float(wrong_gauss),
                actual_nonflat_centre_flux_probability=float(np.linalg.norm(centre_state[nonflat])**2),
                wrong_uncorrelated_tree_chord_flux_probability=float(np.linalg.norm(uncorrelated[nonflat])**2),
                actual_ALL64_centre_branch_Wilson_transfer_covariance_error=covariance,
                wrong_unadjusted_chord_Wilson_pole_spectrum_error=wrong_pole,
                convention='Orient root->child: n_x+q_in-sum q_out=0; hence q_tree=-sum_subtree n. Sum coherently over ALL divergence-compatible q assignments, including every cycle. Fourier support is centre-flat pure gauge, NOT independent four-peaked links.')


def exact_chart_ground() -> dict[str, Any]:
    """Integrate the actual radial operator in both width/radius regimes."""
    from scipy.integrate import quad
    rows = []
    for radius in (8., .4):
        def values(x: float) -> tuple[float, float, float]:
            denominator = radius*radius-x*x
            # Subtract logk(0), preserving the normalized state even when sigma>R.
            logk = -x*x/4-x*x/(radius*radius*denominator)
            first = -x/2-2*x/(denominator*denominator)
            second = -.5-2/(denominator*denominator)-8*x*x/(denominator**3)
            return math.exp(logk), first, second
        norm = quad(lambda x: values(x)[0]**2, -radius, radius, epsabs=1e-12)[0]
        kinetic = quad(lambda x: 2*values(x)[0]**2*values(x)[1]**2, -radius, radius, epsabs=1e-12)[0]/norm
        potential = quad(lambda x: 2*values(x)[0]**2*(values(x)[2]+values(x)[1]**2), -radius, radius, epsabs=1e-12)[0]/norm
        variance = quad(lambda x: x*x*values(x)[0]**2, -radius, radius, epsabs=1e-12)[0]/norm
        rows.append(dict(native_radius_over_width=radius,
                         actual_chart_kinetic_expectation=kinetic,
                         actual_chart_potential_expectation=potential,
                         actual_chart_ground_energy_error=abs(kinetic+potential),
                         actual_normalized_ground_variance=variance,
                         rigorous_coordinate_variance_upper=1/(1+4/radius**4)))
    return dict(native_radial_width_regimes=rows,
                actual_chart_ground_energy_error=max(row['actual_chart_ground_energy_error'] for row in rows),
                chart_ground_formula='k_d(X)=exp[-||X||^2/(4sigma^2)-sigma^2/(R^2-||X||^2)] in the ball,0 outside. Evaluate normalized logk-logk(0), not an underflow-zero unnormalized exponential. H=2Omega sigma^2 sum_i A_i†A_i; its Friedrichs form has EXACT ground energy0 because k and every derivative vanish at the boundary and A_i k=0. No Gaussian cutoff-energy estimate is used. Normal blockdimension45/R=r/2; colour Lie blockdimension15/R=8sigma on each link; V remains gauge-null.',
                actual_normal_excitation_gap_lower=1.,
                gap_proof='Hess(-logk)>=I[1/(2sigma^2)+2sigma^2/R^4]. Weighted Poincare/Bakry-Emery gives conservative gap>=Omega(1+4sigma^4/R^4); coordinate variance<=[sigma^-2+4sigma^2/R^4]^-1. Thus sigma>R produces a NARROWER ground, width<=R^2/(2sigma), and a LARGER gap, not Gaussian variance sigma^2. All excited states remain.',
                massive_centre_sector_law='Add J=sum_localcolourfaces(1-Re centreholonomy)+sum3noncontractiblecolourloops(1-Re centreholonomy), J>=Omega, extended smoothly across SU4 charts. In character basis the coherent cycle packet is the unique centre-flat prepared ground; all other physical centre flux/toron states are finite positive sectors. These are NOT the8 statistical torons, which are retained.')


def charge_weighted_phi_budget(p: RootParameters, photon: Parameters, charge: int = 1,
                              electromagnetic_order: int = 0, *,
                              active_kappa: float | None = None) -> dict[str, Any]:
    """Renew species-resolved analytic/projective and mixed-current budgets."""
    naux = 10**(photon.beta_power+4*photon.kappa_power)//16
    kactive = (max(1/(2*(math.expm1(a*m)+4)) for a in photon.spacings for m in photon.masses)
               if active_kappa is None else active_kappa)
    margin = 1-6*kactive
    if not 0 < kactive < 1/6:
        raise ValueError('positive Wilson B requires actual active kappa in (0,1/6)')
    def logmoment(power: int) -> float:
        auxiliary = math.log10(naux)-power*photon.kappa_power
        active = power*math.log10(kactive)+electromagnetic_order*math.log10(charge)
        return max(auxiliary, active)+math.log10(1+10**(-abs(auxiliary-active)))
    # Relative square-root Sylvester integral needs only a logarithmic condition
    # factor of the ONE-body matrix, never the exponential Fock condition number.
    constant_log = 24*math.log10(2)+2*math.log10(1+photon.kappa_power*math.log(10)+math.log(64))
    active_constant_log = (24*math.log10(2)
                           +2*math.log10(1-math.log(kactive)+abs(math.log(margin))+math.log(64))
                           +(electromagnetic_order+3)*max(0., math.log10(.25/margin)))
    def coefficient_moment(power: int) -> float:
        auxiliary = constant_log+math.log10(naux)-power*photon.kappa_power
        active = active_constant_log+power*math.log10(kactive)+electromagnetic_order*math.log10(charge)
        return max(auxiliary, active)+math.log10(1+10**(-abs(auxiliary-active)))
    volume_log = 3*math.log10(photon.spatial_size)
    slab_log = volume_log+math.log10(photon.time_slices)
    effective_width = max(p.normal_width_power, 2*p.tube_radius_power-p.normal_width_power)
    colour_step = volume_log+coefficient_moment(1)-p.normal_width_power
    normal_step = volume_log+coefficient_moment(2)-p.tube_radius_power
    return dict(active_charge=str(charge), auxiliary_charge=1, electromagnetic_derivative_order=electromagnetic_order,
                physical_component_connection_law=PHYSICAL_COMPONENT_CONNECTION_LAW,
                operator_theorem_scope='NEW physical Dminus pullback law; native framed kinetic/positive kernel verified separately. This coefficient ledger is analytic, not a numerical simulation of the huge family and not old-epoch theorem promotion.',
                coupling_constant_log10=constant_log,
                actual_active_kappa=kactive, actual_active_normalized_B_margin=margin,
                active_coupling_constant_log10=active_constant_log,
                projective_step_colour_log10=2*volume_log+coefficient_moment(1)-p.normal_width_power,
                projective_step_normal_log10=2*volume_log+coefficient_moment(2)-p.tube_radius_power,
                projective_slab_colour_log10=volume_log+slab_log+coefficient_moment(1)-p.normal_width_power,
                projective_slab_normal_log10=volume_log+slab_log+coefficient_moment(2)-p.tube_radius_power,
                projective_slab_density_trace_error_log10=math.log10(16)+volume_log+slab_log+max(
                    coefficient_moment(1)-p.normal_width_power,
                    coefficient_moment(2)-p.tube_radius_power),
                projective_local_coordinate_budget='On45normal/site use sup|Y_a|<=r and ||G_a||<=2: coordinate projective sum<=90Vr (NOT sqrt45 operator sum). On15Lie/link, R=8sigma gives sum sup|X_a|||G_a||<=240sigma; exponential series gives ||deltaU||pi<=480sigma. Three links/site: deltaB_pi<=500kappaVr+1440Vsigma, deltaF_pi<=360kappaVr+1440Vsigma, including the kappa||Y||² term for tiny r. Baseline Bmin>=1/(8kappa); relative Cholesky E_pi<=2^18 V(kappa²r+kappasigma)<1/2 per species. General active margin uses2kappa/(1-6kappa) and its stated inverse-margin multiplier, not8kappa>1 winding or accretive-D claims.',
                projective_resolvent_proof='In C(Y,X) tensor_pi B(A,Fock,centre), E=T0^-1/2 deltaT T0^-1/2. The sqrt difference timesT0^-1/2 is integral t^1/2(T0+t)^-1 T0^1/2 E sum_j[-T0/(T0+t) E]^j (T0+t)^-1 /pi. Neumann converges absolutely for||E||pi<1/2. Splitting t atlambda_min/max gives integral<=[3+logcondT0], hence ||r-I||pi<=2eta_pi[3+logcondT0]. No global Fock condition number or exponential-monotonicity step.',
                projective_exterior_species_proof='Each onebody r=I+sum f_alpha D_alpha has exterior/Fock projective norm <=(1+zeta_s)^(4V)-1 by antisymmetric tensor expansion; common coefficient words are grouped across occupation blocks, so no2^(4V) factor. Species multiplication costs product_s(1+zeta_s)^(4V), i.e weightedSUM NF*kappa_aux² / NF*kappa_aux versus the ONE active kappa. Scalar detB prefactor uses its own absolutely convergent logdet Neumann expansion. Coordinate-l1 V BEFORE exterior4V is paid byV². Displayed2^24(1+logkappa+ln64)^2 exceeds these coefficients; noNF².',
                physical_projective_tensor_frame='NEW physical-component Dminus law is pulled back by the actual Q=Gamma(V) physical-bundle isometry INCLUDING centre-character descent, not separate spin x colour. Dphys=d-(dQ)Qdag gives Qdag Dphys Q=d exactly. Haar/Gauss, canonical/source ranges, kinetic, matter potential, radial Pphi and source halfsteps use the SAME Q. Only in this consistent pullback does scalar radial heat N act on45normal/site and15Lie/link identically for every character and K0=N tensor Kf0. Kf0 retains compactU1 Haar, centre penalties and SAME statistical defects/all8 torons. These budgets are rerun for this NEW Hamiltonian, not inherited from bd4d/931.',
                projective_Schatten_bridge='For R-I=sum f_alpha tensor D_alpha with sum||f||||D||<=delta_series, X=Nhalf R Kf0half and X0=Nhalf Kf0half give ||X-X0||_(2p)<=delta_series||K0||_p^1/2 by Holder/tensor norms. Thus ||Knew-K0||_p<=(2delta+delta²)||K0||_p. Atp=Nt, telescope gives trace-norm power difference<=Nt epsilon(1+epsilon)^(Nt-1)TrK0^Nt; reverse Schatten triangle gives TrKnew^Nt>=(1-epsilon)^NtTrK0^Nt. Normalized density error<=2Nt epsilon(1+epsilon)^(Nt-1)/(1-epsilon)^Nt. With delta_series<=colour+normal<=2max and the verified tiny p epsilon, epsilon<=6max and the displayed16Nt max bound conservatively includes both budgets. This is not inferred from supnorm alone.',
                projective_postsource_rule='Apply ACTUAL bounded CP filters with retained Gram g>0: normalized density error<=2||F||² delta_rho/g before taking honest square-root state/observable price. Pphi, toron conditioning, photon clipping, charged source halfsteps/current must use their OWN actual positive Gram. FixedN2 bare norm1 is not g; no automatic full physical-source admission.',
                prepared_boundary_purifier_proof='For evenNt,m=Nt/2 andp=Nt+1, Xi=Knewhalf Pphi Knew^m. BKS/Ando positive-square-root Schatten inequality plusHolder gives ||(Knewhalf-K0half)Pphi Knew^m||HS<=sqrt(epsilon)(1+epsilon)^m sqrtTrK0^p. Power telescope gives the second term m epsilon(1+epsilon)^(m-1)sqrtTrK0^p. Pphi commutes K0=N tensor Kf0, so ||Xi0||HS²=TrK0^p*gphi_p: no global Tmin or matter trace ratio. Normalized canonical purifier difference<=2h/sqrt(gphi_p), h=sqrt(epsilon)(1+epsilon)^m+m epsilon(1+epsilon)^(m-1); reduced-state trace error<=4h/sqrt(gphi_p). Actual source/charged CP filters AFTER this still pay their actual retained Gram.',
                prepared_ground_filter_proof='Pphi means ONLY continuous radial grounds, tensorIDENTITY on ALL centre/flat/nonflat statistical sectors retained in Kf0; it does NOT select unique flat colourcentre. Normal k² has stronglyconvex -logdensity Hessian>=sigma^-2+4sigma²/R4. Colour exponential-ball Haar Jacobian J has -logJ convex on the tiny Lieball (positive even-power trace series ofadX); k²J obeys same bound. Caffarelli Gaussian-to-target contraction on convex Dirichlet balls gives weighted Dirichlet minmax eigenlevel>=OUlevel, not merely spectral GAP. Conservatively each coordinatelevel>=Omega n; total90V product trace<=prod_i(1-exp(-pOmega_i))^-1. Within chart gphi_p>=prod_i(1-exp(-pOmega_i))>=1-90V exp(-pOmega_min), withOmega_min>=1 andp=Nt+1 actual1,048,577. For full chartDIRECTSUMoutside reference divide by1+TrTout^p/TrTchart^p, bounded by the explicit outside heat/Haar/primitive-entropy proof. Dampedcross corrections are paid separately. Any ADDITIONAL flatcentre/toron filter needs its OWN retained Gram; no automatic AlphaQ or mattertrace equality acrosscentres.',
                prepared_boundary_purifier_error_log10=math.log10(32)+max(
                    2*volume_log+coefficient_moment(1)-p.normal_width_power,
                    2*volume_log+coefficient_moment(2)-p.tube_radius_power)/2,
                prepared_boundary_density_trace_error_log10=math.log10(64)+max(
                    2*volume_log+coefficient_moment(1)-p.normal_width_power,
                    2*volume_log+coefficient_moment(2)-p.tube_radius_power)/2,
                prepared_boundary_Gram_lower_used=.25,
                prepared_boundary_trace_powers=[photon.time_slices+1, photon.time_slices+2],
                prepared_boundary_constant_proof='Let d be the SUM of normal and colour projective step budgets, conservatively <=2max. Transfer Schatten epsilon<=3d<=6max; the actual Outside supplier verifies the dampedcross relative contribution<=max and radial fullsector gphi>=1/4. Thus total epsilon<=8max. In the verified p sqrt(epsilon)<<1 regime h<=2sqrt(epsilon), normalized purifier<=8sqrt(epsilon)<=32sqrt(max), density<=64sqrt(max). The old4sqrt(max) display omitted these constants and is not reused. Charged/photon/toron Grams remain separate.',
                prepared_boundary_Nt_plus2_proof='For the additional NEUTRAL boundary halfstep Xi=K Pphi K^m andp=Nt+2, Holder/telescope gives h<=(m+1)epsilon(1+epsilon)^m and the SAME retained radial Gram gphi,p>=1/4. The verified p sqrt(epsilon)<<1 makes this smaller than the common32sqrt(max) purifier budget. This is NOT a replacement of an actual CHARGED halfstep/source kernel by neutral K; actual charged CP filters still need their own Gram.',
                actual_scalar_angular_heat_operator_norm_upper=4/math.e,
                actual_scalar_angular_Friedrichs_proof='Physical normal-coordinate generator is lambdaR tensorAdS with operatornorm<=2; colourLie generatorAdS hasnorm<=1. Both preserve their ground measures and compact balls. Unique Gaussian-to-target Caffarelli contraction is equivariant under these rotations (normal radial; colour HaarJ isAd-invariant). Dirichlet minmax therefore compares eigenlevels WITHIN each angular-weight sector. Gaussian polynomial degree n has |m|<=2n and conservative level>=Omega n; summing independent blocks gives commuting Hphi>=|Lscalar|/2 withOmega>=1. Thus ||Lscalar exp(-Hphi/2)||<=sup_E2E exp(-E/2)=4/e, independentNF/dimension; Lscalar Pphi=0 and Qphi N Qphi<=e^-1. This is the declared product-coordinate Friedrichs chart law, not a claim about the original Cartesian Phi kinetic.',
                normal_charge_moment_log10=logmoment(2), colour_charge_moment_log10=logmoment(1),
                full_step_relative_root_colour_log10=colour_step,
                full_step_relative_root_normal_log10=normal_step,
                full_slab_colour_operator_log10=slab_log+coefficient_moment(1)-p.normal_width_power,
                full_slab_normal_operator_log10=slab_log+coefficient_moment(2)-p.tube_radius_power,
                full_slab_reference_ground_normal_operator_log10=slab_log+coefficient_moment(2)-effective_width+math.log10(45)/2,
                direct_relative_transfer_proof='Bactual>=(1-6kappa)/(2kappa)-kappa||Y||; baseline kappa<=1/8 gives1/(8kappa), while kappa<=7/48 gives the renewed1/(16kappa) bound. Use2kappa/(1-6kappa) for general margin, including actual negativebare active mass. Ldelta/Dnew whitening bounds relative transfer order; Γ exterior preserves it incldetB. Logmonotonicity is valid; exponential monotonicity is NOT. Active8kappa>1 hopping/accretive arguments are not restored.',
                relative_square_root_proof='For D=sqrtA-sqrtB, sqrtA D+D sqrtB=A-B. If ||B^-1/2(A-B)B^-1/2||<=eta<1/2, the Sylvester integral split at inverse sqrtlambda_max/min gives ||D B^-1/2||<=2eta[2+log(sqrtcondB)+eta]. Onebody cond<=256/kappa², so ONLY logkappa enters. sqrtGamma(T1)=Gamma(sqrtT1); telescoping exterior factors and the scalar detB gives uniform R=Tf_new^1/2 Tf0^-1/2 with ||R-I|| bounded above. The displayed very conservative constant absorbs64sigma link width, dimensions and nEM<=2 chain derivatives.',
                Haar_bridge='For fixedPhi, Knew=R K0 Rdag with the SAME bareP_Haar. Thus compressedA=P_Haar Tf P_Haar retains relative Loewner on its common nonzero range; Knew range is the actual nearidentity image of ranK0, not an assumed source invariant. No global Tmin or exponential operator monotonicity. With normalheat Nhalf outside, Knew=Nhalf R Kf0 Rdag Nhalf. Ideal Tf0 ignoresY/X, centre currents permute identical packets, so [N,Kf0]=0 and ||N||=1 with ground1: ||Knew-K0||/||Kf0||<=2deltaR+deltaR². Field/source thermal normalizations and postselection still require their actual bounded kernel/Gram proof.',
                low_energy_normal_escape='Qphi K0 Qphi<=exp(-Omega)||Kf0||. For ACTUAL lowenergy projector P_E with P_E Knew P_E>=exp(-Ecut)||Kf0||, Ecut<Omega, X=Qphi P_E solves AX-XB=-Qphi deltaK P_E. Ordered Sylvester integral gives ||Qphi P_E||<=(2deltaR+deltaR²)/(exp(-Ecut)-exp(-Omega)) WITHOUT sqrt(dimension), valid for any coherent/mixed source in P_E and all real-time evolution. Shellprob<=2p_reference+2||Qphi P_E||². Actual source energywindow/Gram/highenergyweight still must be proved.',
                charge_semantics='Every n-th EM derivative of active Peierls U^Q contributesQ^n; auxiliaries staycharge1. Normal moment=Naux*kappa_aux²+kappa_active² Q^n; colour moment=Naux*kappa_aux+kappa_active Q^n. Bare sourceunitary norm unchanged; source/dressing flux derivative and selfenergy prices are separate.',
                scope='Actual microscopic chart coefficient/relative-root bounds. No automatic unconditional full-state, physical-alpha, normalized projected source or all-orders amplitude admission.')


def projective_heat_native(gamma: np.ndarray) -> dict[str, Any]:
    """Test the actual Schatten/normalized-trace bridge through radial heat+Haar."""
    nodes, radius = 7, .4
    coordinates = np.linspace(-.75*radius, .75*radius, nodes)
    spacing = float(coordinates[1]-coordinates[0])
    ground = np.exp(-coordinates**2/4-coordinates**2/(radius**2*(radius**2-coordinates**2)))
    ground /= np.linalg.norm(ground)
    derivative = np.zeros((nodes-1, nodes))
    for j in range(nodes-1):
        derivative[j, j] = math.sqrt(ground[j+1]/ground[j])/spacing
        derivative[j, j+1] = -math.sqrt(ground[j]/ground[j+1])/spacing
    normal_h = .02*derivative.T@derivative
    field_e = np.linalg.eigvalsh(normal_h)
    model = BosonicEdgeModel(4, ((0, 1), (0, 2), (0, 3)))
    number = sum((model.n(c) for c in range(4)), np.zeros_like(model.identity))
    left = model.n(0)+model.n(1)
    indices = np.flatnonzero(np.abs(np.diag(number)-2) < 1e-8)
    selector = np.eye(model.dimension)[:, indices]
    haar = np.diag((np.abs(np.diag(selector.conj().T@left@selector)-1) < 1e-8).astype(float))
    basis = matrix_unit_basis()
    ga = selector.conj().T@native_lift(model, basis[0]+.27*basis[3])@selector
    gb = selector.conj().T@native_lift(model, basis[1]+.31*basis[7])@selector
    def fibre(y: float) -> np.ndarray:
        return expm(.3j*y*ga)@expm(.09j*y*y*gb)
    node_frames = tuple(fibre(float(y)) for y in coordinates)
    edge_frames = tuple(fibre(float(y)) for y in (coordinates[:-1]+coordinates[1:])/2)
    physical_form = physical_component_friedrichs_form(node_frames, edge_frames,
        math.sqrt(.02)*np.kron(derivative, np.eye(6)), np.zeros((6*nodes, 6*nodes), complex))
    component_unitary = physical_form['component_unitary']
    physical_normal_half = expm(-physical_form['kinetic']/2)
    _, frame = np.linalg.eigh(gamma[0])
    plus = frame[:, 2:]
    direction = plus@np.diag([1., -1.])@plus.conj().T
    def half(y: float) -> np.ndarray:
        normal = np.zeros((4, 4, 4), complex)
        normal[0] = y*direction
        t, bp, _ = phi_wilson_step(gamma, .03, (.3, .2, .1), normal, (np.eye(4),)*3)
        eigen, vectors = np.linalg.eigh(t)
        h = -math.log(float(np.linalg.det(bp).real))*model.identity-native_lift(model, (vectors*np.log(eigen))@vectors.conj().T)
        return selector.conj().T@expm(-h/2)@selector
    ideal_half = half(0.)
    kf0 = ideal_half@haar@ideal_half
    eigen, vectors = np.linalg.eigh(ideal_half)
    inverse = (vectors/eigen)@vectors.conj().T
    rows = []
    for width in (1e-4, 3e-5):
        halves = [half(width*float(value)) for value in coordinates]
        blocks = [h@inverse for h in halves]
        # Indicators on actual Galerkin nodes give an EXPLICIT separable sum;
        # this native finite pi budget is not a supnorm relabel.
        delta_series = sum(float(np.linalg.norm(r-np.eye(6), 2)) for r in blocks)
        relative = np.zeros((6*nodes, 6*nodes), complex)
        for j, r in enumerate(blocks):
            relative[6*j:6*(j+1), 6*j:6*(j+1)] = r
        relative_phys = component_unitary@relative@component_unitary.conj().T
        kf_phys = component_unitary@np.kron(np.eye(nodes), kf0)@component_unitary.conj().T
        field_half = physical_normal_half
        k0 = field_half@kf_phys@field_half
        k1 = field_half@relative_phys@kf_phys@relative_phys.conj().T@field_half
        scale = float(np.linalg.eigvalsh(k0)[-1])
        k0, k1 = k0/scale, k1/scale
        epsilon = 2*delta_series+delta_series**2
        schatten = []
        for order in (1, 2, 8):
            reference = float(np.sum(np.linalg.svd(k0, compute_uv=False)**order)**(1/order))
            actual = float(np.sum(np.linalg.svd(k1-k0, compute_uv=False)**order)**(1/order))/reference
            schatten.append(dict(order=order, actual_relative_difference=actual, projective_bound=epsilon))
        def positive_half(k: np.ndarray) -> np.ndarray:
            levels, modes = np.linalg.eigh((k+k.conj().T)/2)
            return (modes*np.sqrt(np.maximum(levels, 0)))@modes.conj().T
        half0, half1 = positive_half(k0), positive_half(k1)
        phi_projector = component_unitary@np.kron(np.outer(ground, ground), np.eye(6))@component_unitary.conj().T
        powers = []
        for steps, boundary_power in itertools.product((2, 8, 16), (.5, 1.)):
            reference_power = steps+int(2*boundary_power)
            p0 = np.linalg.matrix_power(k0, reference_power)
            p1 = np.linalg.matrix_power(k1, reference_power)
            z0, z1 = float(np.trace(p0).real), float(np.trace(p1).real)
            error = float(np.linalg.norm(p1/z1-p0/z0, 'nuc'))
            bound = 2*reference_power*epsilon*(1+epsilon)**(reference_power-1)/(1-epsilon)**reference_power
            boundary0, boundary1 = (half0, half1) if boundary_power == .5 else (k0, k1)
            xi0 = boundary0@phi_projector@np.linalg.matrix_power(k0, steps//2)
            xi1 = boundary1@phi_projector@np.linalg.matrix_power(k1, steps//2)
            norm0, norm1 = float(np.linalg.norm(xi0, 'fro')), float(np.linalg.norm(xi1, 'fro'))
            reference_ground_gram = norm0**2/z0
            m = steps//2
            raw_h = (math.sqrt(epsilon)*(1+epsilon)**m+m*epsilon*(1+epsilon)**(m-1)
                     if boundary_power == .5 else (m+1)*epsilon*(1+epsilon)**m)
            h = raw_h/math.sqrt(reference_ground_gram)
            xi_difference = float(np.linalg.norm(xi1/norm1-xi0/norm0, 'fro'))
            prepared_trace_error = float(np.linalg.norm(xi1@xi1.conj().T/norm1**2-xi0@xi0.conj().T/norm0**2, 'nuc'))
            powers.append(dict(time_steps=steps, boundary_heat_power=boundary_power,
                               reference_trace_power=reference_power, actual_trace_ratio=z1/z0,
                               actual_ground_filter_retained_Gram=reference_ground_gram,
                               actual_boundary_prepared_normalized_purifier_difference=xi_difference,
                               actual_boundary_prepared_density_trace_error=prepared_trace_error,
                               BKS_Holder_normalized_purifier_bound=2*h,
                               BKS_Holder_prepared_density_trace_bound=4*h,
                               actual_normalized_density_trace_error=error,
                               actual_unnormalized_power_trace_error=float(np.linalg.norm(p1-p0, 'nuc'))/z0,
                               projective_power_bound=reference_power*epsilon*(1+epsilon)**(reference_power-1),
                               projective_normalized_density_bound=bound))
        rows.append(dict(physical_coordinate_scale=width, actual_separable_coefficient_sum=delta_series,
                         actual_Schatten_rows=schatten, actual_normalized_power_rows=powers))
    return dict(actual_joint_dimension=6*nodes, native_normal_frequency=.01,
                physical_component_connection_law=PHYSICAL_COMPONENT_CONNECTION_LAW,
                physical_radial_kinetic_frame_error=float(np.linalg.norm(physical_form['kinetic']-component_unitary@np.kron(normal_h, np.eye(6))@component_unitary.conj().T)),
                Pphi_in_actual_physical_components=True,
                actual_normal_ground_residual=float(np.linalg.norm(normal_h@ground)),
                actual_normal_energy_levels=field_e.tolist(), actual_heat_Haar_rows=rows,
                scope='NEW physical-component A†A kinetic/potential/Haar and Pphi in the SAME Q frame. Actual primitive positive Tf, literal rank-null integer-Gauss Haar, separable coefficient budgets, all-p Schatten/normalizedTrace and BKS-Xi are executed. This finite basis transport does not substitute a fullSU4 microscopic source Gram. Full45+45 macro uses the renewed analytic projective coefficient theorem, not the native indicator count.')


def relative_haar_native(gamma: np.ndarray) -> dict[str, Any]:
    """Actual primitive congruence/range controls and the exponent-order falsifier."""
    model = BosonicEdgeModel(4, ((0, 1), (0, 2), (0, 3)))
    number = sum((model.n(c) for c in range(4)), np.zeros_like(model.identity))
    left = model.n(0)+model.n(1)
    indices = np.flatnonzero(np.abs(np.diag(number)-2) < 1e-8)
    selector = np.eye(model.dimension)[:, indices]
    bare = np.diag((np.abs(np.diag(selector.conj().T@left@selector)-1) < 1e-8).astype(float))
    rows = []
    direction = sum((math.sin(j+1)*g for j, g in enumerate(matrix_unit_basis())), np.zeros((4, 4), complex))
    direction /= np.linalg.norm(direction)
    def power(matrix: np.ndarray, exponent: float) -> np.ndarray:
        values, vectors = np.linalg.eigh(matrix)
        return (vectors*(values**exponent))@vectors.conj().T
    for kappa in (.1, .03, .01, .003):
        for kind in ('normal', 'colour'):
            width = 1e-5
            y = np.zeros((4, 4, 4), complex)
            if kind == 'normal':
                _, frame = np.linalg.eigh(gamma[0])
                plus = frame[:, 2:]
                y[0] = width*plus@np.diag([1., -1.])@plus.conj().T
            links = (expm(1j*width*direction) if kind == 'colour' else np.eye(4), np.eye(4), np.eye(4))
            t0, bp0, _ = phi_wilson_step(gamma, kappa, (.3, .2, .1), np.zeros_like(y), (np.eye(4),)*3)
            t1, bp1, _ = phi_wilson_step(gamma, kappa, (.3, .2, .1), y, links)
            eigen0, v0 = np.linalg.eigh(t0)
            eigen1, v1 = np.linalg.eigh(t1)
            h0 = -math.log(float(np.linalg.det(bp0).real))*model.identity-native_lift(model, (v0*np.log(eigen0))@v0.conj().T)
            h1 = -math.log(float(np.linalg.det(bp1).real))*model.identity-native_lift(model, (v1*np.log(eigen1))@v1.conj().T)
            tf0 = selector.conj().T@expm(-h0)@selector
            tf1 = selector.conj().T@expm(-h1)@selector
            half0, half1 = power(tf0, .5), power(tf1, .5)
            r = half1@power(tf0, -.5)
            relative = float(np.linalg.norm(r-np.eye(6), 2))
            k0, k1 = half0@bare@half0, half1@bare@half1
            a0, a1 = bare@tf0@bare, bare@tf1@bare
            selected = np.flatnonzero(np.diag(bare))
            a0p = a0[np.ix_(selected, selected)]
            a1p = a1[np.ix_(selected, selected)]
            whitened = power(a0p, -.5)@a1p@power(a0p, -.5)
            p0 = half0[:, selected]@np.linalg.inv(a0p)@half0[:, selected].conj().T
            p1 = half1[:, selected]@np.linalg.inv(a1p)@half1[:, selected].conj().T
            rows.append(dict(kappa=kappa, coordinate=kind,
                             actual_primitive_log_variation=float(np.linalg.norm(h1-h0, 2)),
                             actual_log_derivative_scaled=float(np.linalg.norm(h1-h0, 2))/(width*kappa**(2 if kind == 'normal' else 1)),
                             actual_relative_halfstep_norm=relative,
                             actual_Haar_compressed_relative_order_error=float(np.max(np.abs(np.log(np.linalg.eigvalsh(whitened))))),
                             actual_normalized_K_difference=float(np.linalg.norm(k1-k0, 2)/np.linalg.norm(k0, 2)),
                             actual_congruence_bound=2*relative+relative*relative,
                             actual_range_projector_change=float(np.linalg.norm(p1-p0, 2)),
                             actual_range_kernel_dimension=2,
                             representative_Gauss='Two2spin sites, fixedN2, E=nleft-1, Eother=-E. Actual compactHaar in this integerGauss sector projects nleft=1; physical cycle/Phi/SU4 macro not inferred.'))
    h = np.diag([0., 40.])
    changed = h+.001*np.array([[0., 1.], [1., 0.]])
    raw0, raw1 = float(expm(-h)[1, 1]), float(expm(-changed)[1, 1])
    return dict(actual_primitive_rows=rows,
                wrong_exp_of_close_H_projected_trace=raw0,
                actual_exp_of_close_H_projected_trace=raw1,
                actual_close_H_norm=.001,
                wrong_exp_order_Haar_amplification=raw1/raw0,
                scope='Relative Cholesky transfer and literal Haar compression executed with primitive edge-qubit Hf. The2site2spin native is a range discriminator, not a fullSU4 thermal-model substitute.')


def full_gauss_bundle(p: RootParameters, photon: Parameters) -> dict[str, Any]:
    naux = 10**(photon.beta_power+4*photon.kappa_power)//16
    volume = photon.spatial_size**3
    return dict(finite_full_Gauss_positive_control=gauss_bundle_native(),
                full_global_centre_condition=(2*(naux+1)*volume) % 4 == 0,
                physical_Hilbert='L2(normal-bundle tube x SU4 colour links x compact U1 links) tensor primitive even Fock, plus full Cartesian outside block; orthogonal LOCAL SU4 and U1 Haar Gauss projection. Source and defect commute with that projection.',
                source_embedding='For every occupation component of the ACTUAL filled Wilson sea, use the coherent normalized sum over ALL Z4 link-character assignments with divergence q=n_total mod4. A tree solution has q_edge=-sum_subtree n. On every link f_q(U)=sum_z z^(-q) k(zU), Haar normalized; distinct q are orthogonal and have IDENTICAL norms because centre bump supports are disjoint. Include every cycle character, not a fixed tree flux. Matter transforms by Gamma(V_x), links by V_xdag U_xy V_y.',
                actual_centre_single_valuedness='Changing V_x to z_x V_x multiplies the local Fock occupation by z_x^n_x and changes adjacent f_q by exactly the inverse character, using the tree divergence equation. This is independent of the local occupation fluctuations in the filled sea. A purely adjoint Phi state without these link packets fails.',
                full_SU4_Gauss_norm='Gauge transformations V_x->g_x V_x and U_xy->g_x U_xy g_y^dag leave all transformed link arguments and covariant matter sections unchanged. Normalized fibre Fock states and normalized orthogonal f_q packets give an isometric physical embedding; Haar orientation volume is normalized to1. No additional charged species or spin multiplicity.',
                full_U1_Gauss_dressing='For each matter/Ref occupation component rho with integer sumrho=0, choose integer tree electric flux E_tree with divergence E=rho. Multiply normalized coexact/harmonic compact photon wavefunction by the LITERAL integer Wilson string exp(i E_tree·A). This restores full U1 Gauss without changing norm. Non-tree hopping produces actual Wilson loops on the photon state; these U1 fields remain EXACT, not approximated by the colour packet width.',
                source_actual_Ref='Conditional GAUGE-ISOMETRIC freeWilson-sea embedding only: co-located Splus=sum a_cdag r_c/sqrt2 with Ref4, Sminus=adjoint with Ref0. It preserves total siteoccupation/q. BareGram=(4-<nactive>)/2 or<nactive>/2. Translation-invariant conditional free sea has<nactive>=2. This is NOT the actual fullthermal or charged OS/range Gram; those must be measured/matched separately.',
                spatial_current_transport='A physical colour Wilson link shifts q_e and endpoint occupation, PERMUTING the coherent sum with unit overlap. Its coordinate support is correlated PUREGAUGE centre assignments; all64 native branch T1 covariances are checked, including wrong-chord controls. Fifteen Lie coordinates with Rcolour=8sigma give ||X||op<=64sigma, hence ||exp(iX)-I||<=64sigma. All species/path coefficients enter the4096 ledger. U1 Peierls fields remain exact quantum operators.',
                sigma=f'1e-{p.normal_width_power}', compact_support_not_tail=True,
                prepared_colour_modes=45*volume,
                colour_link_law='NEW physical radial operators Hkin_phys=Q[2Omega sigma^2 sum Acoord†Acoord]Qdag, Acoord=partial-partial logk, Aphys=Q Acoord Qdag with Dphys=d-(dQ)Qdag. k=exp[-||X||²/(4sigma²)-sigma²/(R²-||X||²)]. Fifteen Lie/link Rcolour8sigma,45normal/site Rnormal=r/2. BOTH matter potential and radial projector are transported by the same Q, including real centre-character descent. Identical central copies/J retain all colour sectors; full Cartesian/SU4 outside and nonzero physical cross remain; statistical8torons are not discarded.',
                prepared_connection='NEW implemented physical-component Dphys=d-(dQ)Qdag=Q d Qdag, Q=Gamma(V); Qdag Dphys(Q psi0)=d psi0. Hkin_phys=Q Hkin_coordinate Qdag and Vmatter_phys=Q Vmatter_coordinate Qdag. Source=Qcharged Scoordinate Qneutraldag with BOTH charged halfsteps and their canonical/Gauss ranges transported. Old bd4d/931 plus-form cancellation claim is withdrawn, not a coordinate convention change.',
                consumer_remainder_log10=charge_weighted_phi_budget(p, photon)['projective_slab_colour_log10'],
                admitted_scope='Constructed NEW physical-domain embedding and declared massive chart law. The exact interacting full-Gauss ground is not identified with the prepared sea.')


def outside_cartesian_native() -> dict[str, Any]:
    """A real positive/nontrivial full-field operator, including forbidden-B control."""
    size = 9
    annihilate = np.diag(np.sqrt(np.arange(1, size)), 1).astype(complex)
    coordinate = (annihilate+annihilate.conj().T)/math.sqrt(2)
    momentum = (annihilate-annihilate.conj().T)/(1j*math.sqrt(2))
    matter = BosonicEdgeModel(4, ((0, 1), (0, 2), (0, 3)))
    coupling = native_lift(matter, SELECTED[0])
    field = .2*(momentum@momentum)+coordinate@coordinate+coordinate@coordinate@coordinate@coordinate
    full = np.kron(field, matter.identity)+.7*np.kron(coordinate, coupling)
    eigen = np.linalg.eigvalsh(full)
    positive = expm(-.15*(full-eigen[0]*np.eye(full.shape[0])))
    bare_B = np.eye(size)-.7*coordinate
    xvalues, xbasis = np.linalg.eigh(coordinate)
    pin_field = (xbasis*(np.abs(xvalues) < 1.5))@xbasis.conj().T
    boundary = (xbasis*((np.abs(xvalues) > .75) & (np.abs(xvalues) < 1.5)))@xbasis.conj().T
    pin = np.kron(pin_field, matter.identity)
    pout = np.eye(full.shape[0])-pin
    reference = expm(-.2*(momentum@momentum+coordinate@coordinate))
    forward = np.kron((np.eye(size)-pin_field)@reference@boundary, matter.identity)
    defect = (forward+forward.conj().T)/2
    blocked_h = pin@full@pin+pout@full@pout
    blocked_h -= np.linalg.eigvalsh(blocked_h)[0]*np.eye(full.shape[0])
    half = expm(-.075*blocked_h)
    penalty = pin+math.exp(-2)*pout
    damped_defect = penalty@defect@penalty
    joined = penalty@half@expm(.07*damped_defect)@half@penalty
    virtual_undamped = pin@(expm(.07*defect)-np.eye(full.shape[0]))@pin
    virtual_damped = pin@(expm(.07*damped_defect)-np.eye(full.shape[0]))@pin
    # The retired outside-only shell cannot couple either Dirichlet block.
    outside_field = np.eye(size)-pin_field
    inactive_forward = np.kron(outside_field@reference@outside_field, matter.identity)
    inactive_defect = (inactive_forward+inactive_forward.conj().T)/2
    inactive = penalty@half@expm(.07*inactive_defect)@half@penalty
    number = sum((matter.n(c) for c in range(4)), np.zeros_like(matter.identity))
    total_number = np.kron(np.eye(size), number)
    return dict(Cartesian_field_Galerkin=size, actual_primitive_dimension=matter.dimension,
                actual_joined_dimension=full.shape[0], actual_full_H_lower=float(eigen[0]),
                actual_positive_T_min=float(np.linalg.eigvalsh(positive)[0]),
                actual_field_matter_noncommutator=float(np.linalg.norm(full@np.kron(momentum, matter.identity)-np.kron(momentum, matter.identity)@full)),
                wrong_linear_Phi_as_positive_B_min=float(np.linalg.eigvalsh(bare_B)[0]),
                actual_boundary_defect_norm=float(np.linalg.norm(defect, 2)),
                actual_damped_boundary_defect_norm=float(np.linalg.norm(damped_defect, 2)),
                wrong_final_penalty_only_chart_virtual_excursion=float(np.linalg.norm(virtual_undamped, 2)),
                actual_inside_exponential_damped_chart_virtual_excursion=float(np.linalg.norm(virtual_damped, 2)),
                actual_boundary_defect_charge_commutator=float(np.linalg.norm(defect@total_number-total_number@defect)),
                actual_full_field_chart_transition_norm=float(np.linalg.norm(pout@joined@pin, 2)),
                wrong_outside_only_shell_transition_norm=float(np.linalg.norm(pout@inactive@pin, 2)),
                actual_full_field_joined_positive_min=float(np.linalg.eigvalsh(joined)[0]),
                chart_transition_law='Chart radiusR=1.5, tube r=3: actual inner boundary shell r/4<|Y|<r/2. Dcross=(Poutside exp(-Href) chi_inner +adjoint)/2. W_exit=Pin+exp(-Delta/2)Pout is applied INSIDE exp(epsilon W_exit Dcross W_exit), as well as the outer penalty. Final penalty alone leaves unsuppressed virtual chart→outside→chart terms. NativeDelta4 gives nonzero damped transition; global exactDelta huge remainsFINITE. Outside-only shell has zero transition. Fullfamily cross uses Cartesian/SU4 referenceheat and commutesGauss.',
                full_field_not_identity_transfer=float(np.linalg.norm(positive-np.eye(full.shape[0]))),
                full_global_outside_operator='Outside the JOINT chart (normal |Y|<r/2 AND every colourLie |X_e|<8sigma nearZ4), Hout uses EXACT Cartesian kinetic -1/2 sum_(60V) partial_Phi^2 in Tr(Phi²)/4 orthonormal coordinates, Lambda_out(sum_mu Tr(Phi_mu²)/4)^2, positive Eaux n, actual linearPhi and SU4/U1 Wilson hops, plus coefficientONE times the positive SU4link Laplacian normalized Tr(G_a²)/4=1 (fundamental Casimir15). Tout=e^(-(Hout+Bout)/2) P_U1Haar e^(-(Hout+Bout)/2) thenFULLGauss. Same zero-bareMaxwell law, no E²/B². Colour complement even with Phi stillnearorbit belongs to OUTSIDE, not an undefined annulus/no-op.',
                scope='Actual quantumPhi-only Galerkin BEFORE adding the U1 Haar kernel; finite coefficients differ from the huge family. The separate cyclic U1 trace probe executes the actual smoothing correction.')


def outside_u1_trace_native() -> dict[str, Any]:
    """Execute trace growth on a physical compact cycle, versus literal Haar."""
    field_size, angles = 5, 64
    annihilate = np.diag(np.sqrt(np.arange(1, field_size)), 1).astype(complex)
    x, momentum = (annihilate+annihilate.conj().T)/math.sqrt(2), 1j*(annihilate.conj().T-annihilate)/math.sqrt(2)
    field = .2*momentum@momentum+x@x+x@x@x@x
    model = BosonicEdgeModel(4, ((0, 1), (0, 2), (0, 3)))
    number = sum((model.n(c) for c in range(4)), np.zeros_like(model.identity))
    indices = np.flatnonzero(np.abs(np.diag(number)-2) < 1e-8)
    selector = np.eye(model.dimension)[:, indices]
    zero_charge = np.flatnonzero(
        (np.abs(np.diag(model.n(0))-1) < 1e-8)
        & (np.abs(np.diag(model.n(1))-1) < 1e-8)
        & (np.abs(np.diag(model.n(2))) < 1e-8)
        & (np.abs(np.diag(model.n(3))) < 1e-8))
    bare_matter = selector.conj().T@np.eye(model.dimension)[:, zero_charge]
    bare = np.kron(np.eye(field_size), bare_matter@bare_matter.conj().T)
    shift = 1+2*float(np.linalg.norm(x, 2))
    half_blocks = []
    for j in range(angles):
        theta = 2*math.pi*j/angles
        one = np.diag([.3, -.3, .2, -.2]).astype(complex)
        one[0, 1], one[1, 0] = .2*np.exp(1j*theta), .2*np.exp(-1j*theta)
        one[1, 2] = one[2, 1] = one[2, 0] = one[0, 2] = .2
        primitive = selector.conj().T@native_lift(model, one)@selector
        h = np.kron(field, np.eye(6))+.7*np.kron(x, primitive)+shift*np.eye(6*field_size)
        half_blocks.append(expm(-h/2))
    fourier = np.fft.fft(np.array(half_blocks), axis=0)/angles
    direct_trace = float(np.mean([np.trace(h@h).real for h in half_blocks]))
    corrected_trace = float(np.mean([np.trace(h@bare@h).real for h in half_blocks]))
    rows = []
    for cutoff in (0, 1, 2, 4, 8, 12):
        retained = [fourier[n % angles] for n in range(-cutoff, cutoff+1)]
        haar_trace = sum(float(np.trace(h@bare@h.conj().T).real) for h in retained)
        rows.append(dict(cycle_Fourier_cutoff=cutoff,
                         wrong_unsmoothed_positive_direct_integral_trace=(2*cutoff+1)*direct_trace,
                         corrected_literal_Haar_sandwich_trace=haar_trace,
                         corrected_missing_trace=corrected_trace-haar_trace))
    tail = sum(float(np.trace(h@bare@h.conj().T).real) for n, h in enumerate(fourier)
               if 16 <= n <= angles-16)
    return dict(actual_field_Galerkin=field_size, actual_primitive_dimension=model.dimension,
                actual_fixedN2_dimension=6, actual_compact_cycle_quadrature=angles,
                actual_bare_Haar_matter_rank=int(np.linalg.matrix_rank(bare_matter)),
                actual_full_corrected_outside_trace=corrected_trace,
                actual_direct_integral_trace_per_Fourier_mode=direct_trace,
                actual_cycle_trace_rows=rows, observed_high_Fourier_alias_budget=tail,
                literal_operator='Tout(A,Aprime)=exp[-(Hout(A)+Bout)/2] P_rho0 exp[-(Hout(Aprime)+Bout)/2]. Full-link bareHaar+Gauss on this triangle reduces to cycleFourier0 AND rho0 matter. Unsmoothed exp(-Hout(A)) is multiplication and its truncated trace grows exactly2cut+1; finite exit penalty never cures it.',
                physical_cycle_domain='Three-site triangle with backgrounds(1,1,0), four native modes and fixedN2. Integer treeflux restores Gauss for every occupation; one independent integer cycleflux remains. Allsite Gauss is retained after gauge fixing, not an unconstrained angle invented for this probe. FullSU4/source branch not inferred.',
                trace_class_proof='For the full outside block, Phi quartic/kinetic and positive SU4 colour Laplacian give uniformly traceclass e^-Hout(A) over compactU1 A. ||e^-Hout/2 P_Haar||HS²=integral_dA Tr_Phi,colour,Fock e^-Hout(A)<infinity; same after orthogonal Gauss. Thus Tout is traceclass. Chart has massive normal/colour heat smoothing and the SAME Haar. Bounded exp(epsilon D) sandwich of T0^1/2 preserves traceclass. Full thermal trace is therefore defined on the physical quotient; unsmoothed c280 outside claim is explicitly withdrawn.')


def positive_trace_preparation(gamma: np.ndarray, photon: Parameters) -> dict[str, Any]:
    """Prepare actual coupled quantum-field/Fock density, not a free-sea isometry."""
    kappa, steps, field_size = .03, 8, 3
    pole_mass = math.log(1/(2*kappa)-3)
    annihilate = np.diag(np.sqrt(np.arange(1, field_size)), 1).astype(complex)
    coordinate = .04*(annihilate+annihilate.conj().T)
    nodes, rotation = np.linalg.eigh(coordinate)
    blocks = []
    for node in nodes:
        t, prefactor, _ = wilson_step(gamma, 1., pole_mass, np.array([.3+node, .2, .1]))
        blocks.append(prefactor*exterior(t))
    raw = np.einsum('ix,xab,jx->iajb', rotation, np.array(blocks), rotation.conj()).reshape(48, 48)
    weights = np.repeat(np.exp(-.2*np.arange(field_size)/2), 16)
    raw = weights[:, None]*raw*weights[None, :]
    eigenvalues, vectors = np.linalg.eigh(raw)
    transfer = raw/eigenvalues[-1]
    powered = (vectors*((eigenvalues/eigenvalues[-1])**steps))@vectors.conj().T
    grand = powered/np.trace(powered).real
    occupation = np.tile(np.array([mask.bit_count() for mask in range(16)]), field_size)
    canonical_selector = np.diag((occupation == 2).astype(float))
    canonical_raw = canonical_selector@powered@canonical_selector
    canonical = canonical_raw/np.trace(canonical_raw).real
    outside_probability = float(np.trace(grand@np.diag((occupation != 2).astype(float))).real)
    retained = np.kron(coordinate@coordinate, np.eye(16))
    canonical_retained = float(np.trace(canonical@retained).real)
    grand_retained = float(np.trace(grand@retained).real)
    free_weights = np.exp(-.2*steps*np.arange(field_size))
    free_field_retained = float(np.sum(free_weights*np.diag(coordinate@coordinate).real)/np.sum(free_weights))
    naux = 10**(photon.beta_power+4*photon.kappa_power)//16
    fourvolume = photon.spatial_size**3*photon.time_slices
    winding_log = (math.log10(8*fourvolume)+math.log10(naux)
                   +photon.time_slices*(math.log10(16)-photon.kappa_power)
                   -math.log10(photon.time_slices))
    return dict(actual_quantum_Fock_field_dimension=48, actual_canonical_trace_dimension=18,
                native_kappa=kappa, native_time_steps=steps,
                actual_normalized_positive_T_min=float(np.linalg.eigvalsh(transfer)[0]),
                actual_prepared_density_trace=float(np.trace(canonical).real),
                actual_prepared_density_min_observed=float(np.linalg.eigvalsh(canonical_raw[np.ix_(occupation == 2, occupation == 2)])[0]),
                actual_prepared_density_positive_lower_from_factorization=float((eigenvalues[0]/eigenvalues[-1])**steps),
                actual_grand_wrong_number_probability=outside_probability,
                actual_canonical_retained_photon_second_moment=canonical_retained,
                actual_grand_retained_photon_second_moment=grand_retained,
                actual_canonical_grand_retained_difference=abs(canonical_retained-grand_retained),
                wrong_unprepared_free_field_second_moment=free_field_retained,
                wrong_unprepared_retained_state_difference=abs(canonical_retained-free_field_retained),
                actual_canonical_mean_Dirac_number=float(np.trace(canonical@np.diag(occupation)).real),
                native_scope='Actual positive FULL coupled Wilson-Fock/quantum-field power and canonical trace, conditional Clifford mode. No frozen auxiliary Slater state or mock trace is used. Separate full-Gauss native controls and full-family construction provide its physical-domain completion.',
                full_prepared_state='FORMAL full-law rho=PiNs P_G chi K^Nt chi P_G PiNs/actualTrace. It is traceclass ONLY with the repaired outside SAME-U1-Haar sandwich; unsmoothed c280 outside thermal claim is withdrawn. The actual prepared scientist instead executes rho=Khalf R K^Nt Rdag Khalf/actualTrace, R=Pphi_chart f_toron, then photon and charged OS halfsteps; this CONDITIONED preparation is not an unconditional full ground assertion. ALLspecies includingactive once, protected Ref4/0 and finite statistical/normal/outside dynamics retained.',
                normalization_positive_control='The fixed-number Gauss/Haar-range trial has positive transfer expectation, so the denominator is nonzero. It is NOT the normalized free-heavy-sea embedding. Actual canonical-positiveT free heavy density variance is O(Naux*kappa^4), whereas the wrong nearest-Dirac-Slater variance is O(Naux*kappa^2); neither free covariance admits a physical projected thermal vacuum.',
                auxiliary_conditional_state='Auxiliaries are traced/projected in the SAME positive microscopic T; conditional coherent elimination is the full Wilson determinant. All virtual density/current fluctuations retained. FullPhi departure uses the NEW kappa²-normal/kappa-colour relative-Cholesky/root bounds, not the retired unweighted1e-418 state-transfer assertion.',
                actual_full_family_canonical_winding_error_log10=winding_log,
                canonical_winding_proof='Insert species charge fugacity exp[i theta(N_s-2V)]. Every nonwinding heavy logdet loop is independent of theta; temporal winding requires length>=Nt. Sum norms using16k gives eta<=8 V4 Naux(16k)^Nt/[Nt(1-16k)]. Fourier projection to each zero relative-charge sector differs normalized retained grandcanonical correlations by<=4eta/(1-eta). This is a proof about the exact positive-transfer trace, NOT a free-sea-ground assertion. The factor16 conservatively covers noncommuting colour projectors; the separately priced chart departure is not discarded.',
                source_normalization_in_prepared_state='Bare filledRef4 UNITARY-Wilsonline source S/sqrt2 has Gram=(4-<n_active(x)>)/2, equal1 on fixed-onsiteN2 trial ONLY or after an actual symmetry/expectation proof. General packets need their full cross-source Gram. Physical admission additionally requires nonzero Tr(Tcharged^1/2 S rho_T Sdag Tcharged^1/2), filter loss and current matching on the nonzeroT range. Gauss invariance and bare norm do not imply finite logT energy; no global Tmin is asserted.',
                full_physical_source_receiver='chi_out S_worldline T^n S_worldline_dag chi_in with the actual integer SU4/U1/primitive path, staticRefworldline and canonical/source-background projectors. The joined packet scientist evaluates its grade2/current/contact consumer; tree Compton alone is not the whole density.')


def topology_and_centre(photon: Parameters) -> dict[str, Any]:
    cfg = ResponseConfig(flavors=2)
    _, _, law = make_band(cfg)
    graph = FlavorPauliGraph(law, 2)
    observed = graph.topology()
    # Exact Pauli X support determines the Z_e statistical syndrome.
    # Species-labelled cubic plaquettes alone distinguish all spatial edges.
    spatial_syndromes = tuple(tuple(i for i, face in enumerate(graph.plaquettes)
                                  if (face.x >> edge) & 1)
                             for edge in graph.spatial_edges)
    if any(len(syndrome) != 4 for syndrome in spatial_syndromes) or len(set(spatial_syndromes)) != len(graph.spatial_edges):
        raise ArithmeticError('Actual cubic spatial defects have zero/duplicate plaquette syndromes')
    # A degree-two square is a genuine adverse control: all edge syndromes
    # coincide, so blindly replacing the second moment by 1/E is false.
    square = BosonicEdgeModel(4, ((0, 1), (1, 2), (2, 3), (0, 3)))
    flat = (square.identity+square.stabilizers[0])/2
    zsum = np.zeros_like(square.identity)
    for edge in range(len(square.edges)):
        word = ['I']*len(square.edges)
        word[edge] = 'Z'
        zsum += tensor_pauli(word)/len(square.edges)
    duplicate_error = float(np.linalg.norm(flat@zsum@zsum@flat-flat/len(square.edges), 2))
    if duplicate_error < .5:
        raise ArithmeticError('Duplicate-syndrome square failed to reject false 1/E second moment')
    naux = 10**(photon.beta_power+4*photon.kappa_power)//16
    species = naux+1
    volume = photon.spatial_size**3
    spatial_count = 3*species*volume
    epsilon = 1/16
    # exp(eps)-1-eps <= eps² exp(eps)/2, with no huge float conversion.
    return_log = 2*math.log10(epsilon)+epsilon/math.log(10)-math.log10(2)-math.log10(spatial_count)
    modes, edges = 4*species*volume+4, (7*species-1)*volume+4
    cycles = edges-modes+1
    # Global SU4 centre projects to zero unless total fundamental occupation is 0 mod4.
    residues = {str(v): (2*species*v+4) % 4 for v in (1, 3**3, 18**3, volume)}
    return dict(native_topology_positive_control=observed,
                statistical_defect_budget=dict(
                    full_spatial_edges_exact=str(spatial_count),
                    epsilon_total=epsilon,
                    exact_support='ONLY all species spatial anchor edges; no onsite, Ref or nearest-species-tree edges. Dstat=(sum_e Z_e)/Espatial and the exponential uses epsilon_total*Dstat.',
                    actual_native_spatial_edges=len(graph.spatial_edges),
                    actual_native_unique_spatial_syndromes=len(set(spatial_syndromes)),
                    actual_native_plaquettes_per_spatial_edge=4,
                    actual_native_nonspatial_edges_excluded=len(graph.edges)-len(graph.spatial_edges),
                    wrong_degree2_duplicate_syndrome_second_moment_error=duplicate_error,
                    exact_flat_moments='P Dstat P=0; P Dstat² P=P/Espatial. P retains ALL8 statistical torons, ALL Fock occupations and ALL Phi/colour/centre states; it imposes only local statistical-cycle flatness.',
                    all_size_proof='For L>=3 each spatial edge belongs to four species-labelled elementary faces, two for each transverse direction. Their basepoints/orientations identify the edge uniquely; different species have disjoint face labels. Thus every Z_e has a nonzero syndrome and P Z_e Z_f P=delta_ef P, independently of redundant face constraints. Nearest-species rectangles only add constraints, and no tree/reference edge is summed.',
                    flat_return_relative_upper_formula='0<=P exp(epsilon Dstat)P-P<=[exp(epsilon)-1-epsilon]P/Espatial <=epsilon² exp(epsilon)P/(2Espatial). Integral remainder uses P Dstat exp(tDstat) Dstat P and ||Dstat||<=1; no exp operator monotonicity.',
                    flat_return_relative_log10_upper=return_log,
                    flat_return_whole_slab_relative_log10_upper=math.log10(photon.time_slices)+return_log+math.log10(2),
                    flat_to_nonflat_exponential_norm_log10_upper=math.log10(epsilon)+epsilon/math.log(10)-math.log10(spatial_count)/2,
                    single_path_length_upper=3*photon.spatial_size//2,
                    single_path_Dstat_commutator_relative_log10_upper=math.log10(3*photon.spatial_size)-math.log10(spatial_count),
                    single_path_exponential_commutator_relative_log10_upper=math.log10(epsilon)+epsilon/math.log(10)+math.log10(3*photon.spatial_size)-math.log10(spatial_count),
                    source_commutator_rule='For one actual integer/primitive path, [Dstat,Spath]=-(2/Espatial) Spath sum_traversed Z_e, hence relative norm<=2ell/Espatial; Duhamel adds epsilon exp(epsilon). A coherent wholepacket requires its ACTUAL coefficient-l1/nuclear sum or a separate proved CAR/source-kernel orthogonality bound; individual relative bounds are not silently promoted to ||sum c Spath||.',
                    source_current_algebra='Z_e commutes onsite even SU4/U1 charge and co-local onsite/Ref source paths; it anticommutes primitive A_e and paths traversing e. General separated sources and spatial currents do NOT commute Dstat. Photon link-coordinate observables commute Z_e. Canonical source/path kernels must stay exact; no independent-statistical-fibre factorization or scalar RN/source admission follows.'),
                finite_helper_scope='Only primitive graph topology reused; its neutral-heavy continuous-time coefficients are NOT covariant physics.',
                covariant_charged_species=str(species), neutral_heavy_added=0,
                full_volume=str(volume), full_modes=str(modes), full_edge_qubits=str(edges),
                independent_cycles=str(cycles), local_constraint_rank=str(cycles-3),
                retained_statistical_torons=3, retained_flat_statistical_histories=8,
                all_defect_and_occupation_sectors_retained=True,
                symbolic_rank_derivation='V-1 spanning spatial tree per species, three cubic H1 cycles, onsite/species/reference trees; plaquette+nearest-species rectangles identify all species torons. Rank E-M+1-3. Reference adds four modes and four tree edges, no cycle.',
                actual_global_SU4_centre_residue=residues,
                odd_volume_wrong_Gauss_norm=0., even_volume_centre_passes_not_sufficient=True,
                local_Gauss_completion='Pair wedge2 heavy/charged same-species trial across disjoint nearest-neighbour dimers on even spatial box via the actual wedge2 SU4 Wilson link. This is a number-sector trial, NOT the momentum filled sea. Prepared Wilson sea requires gauge-averaged transported chart state.',
                fixed_species_number=2*volume, source_background_per_site=str(2*species),
                Ref_plus='filled4 to3', Ref_minus='empty0 to1')


def family_bounds(p: RootParameters, photon: Parameters) -> dict[str, Any]:
    """Renew HUGE multiplicity bounds; do not recycle Lambda256 or gap2^80."""
    naux = 10**(photon.beta_power+4*photon.kappa_power)//16
    species = naux+1
    volume = photon.spatial_size**3
    fourvolume = volume*photon.time_slices
    kmax = max(1/(2*(math.expm1(a*m)+4)) for a in photon.spacings for m in photon.masses)
    bmin = 1-6*kmax
    # Hermitian positive B and arbitrary bounded offdiagonal F, not Clifford globally.
    bmax = 1+6*kmax
    fmax = 6*kmax
    log_one = math.log(bmax/bmin)+2*math.log1p(fmax)
    log_aux = photon.kappa_power*math.log(10)+math.log(2)
    # Four modes per cell; safe log-norm ledger for detB and Gamma(T1).
    cap = 16*volume*(int(math.ceil(log_one))+naux*int(math.ceil(log_aux)))
    cap_decimal = str(cap)
    # A penalty larger than twice the FULL log-transfer cap, not a spectral gap
    # borrowed from a different Hamiltonian. Scalar normalization is harmless.
    # Dimer Fock connection is parallel in V; colour derivatives cost at most
    # 4096 Omega V S^2 sigma^2. Keep this HUGE-species term before rounding.
    colour_scale = 10**(2*p.normal_width_power)
    colour_trial_numerator = 4096*int(math.ceil(p.normal_frequency))*volume*species*species
    colour_trial_upper = (colour_trial_numerator+colour_scale-1)//colour_scale
    field_trial = 90*volume+colour_trial_upper
    delta = 4*cap+4*field_trial+256
    exponent = math.log10(fourvolume)+math.log10(4096)+math.log10(naux)-p.tube_radius_power
    width_ratio_log = 2*(p.normal_width_power-p.tube_radius_power)
    effective_width_power = max(p.normal_width_power, 2*p.tube_radius_power-p.normal_width_power)
    field_kinetic_log = math.log10(2*p.normal_frequency)-2*p.normal_width_power
    gap_gain_log = math.log10(64)+4*(p.tube_radius_power-p.normal_width_power)
    tail_prefactor_log = math.log10(volume)+45*math.log10(4)
    dimension_log = math.log10(60*volume)
    quartic_gap_log = (math.log10(3/4)-2*math.log10(2)/3
                       +4*dimension_log/3+(4000-math.log10(volume)-math.log10(2))/3)
    half_quartic_shift_log = (math.log10(.75)
                             +(4*(math.log10(64*volume)+math.log10(naux)-photon.kappa_power)
                               -(4000-math.log10(volume))-math.log10(2))/3)
    colour_heat_trace = 0.
    for a in range(5):
        for b in range(5-a):
            for c in range(5-a-b):
                representation_dimension = (a+1)*(b+1)*(c+1)*(a+b+2)*(b+c+2)*(a+b+c+3)/12
                laplace_eigenvalue = 3*a*a+4*b*b+3*c*c+4*a*b+2*a*c+4*b*c+12*a+16*b+12*c
                colour_heat_trace += representation_dimension**2*math.exp(-laplace_eigenvalue)
    primitive_edges = (7*species-1)*volume+4
    return dict(auxiliary_count=str(naux), auxiliary_count_digits=len(str(naux)),
                physical_component_connection_law=PHYSICAL_COMPONENT_CONNECTION_LAW,
                actual_auxiliary_pole_energy_lattice=photon.kappa_power*math.log(10)-math.log(2),
                pole_formula='Eaux=log(1/(2k)-3); m0aux=1/(2k)-4 is a bare coefficient, NOT the transfer energy.',
                charge_per_species=p.source_charge, active_B_lower=bmin,
                primitive_hopping_l1_active_bound=8*kmax,
                primitive_hopping_l1_auxiliary_total_log10=math.log10(naux)+math.log10(8)-photon.kappa_power,
                full_B_F_log_transfer_norm_cap_exact=cap_decimal,
                renewed_cycle_penalty_exact=str(delta),
                all_sector_spectral_ratio_formula='||QT_deltaQ||/lambda_max(PT_deltaP) <= exp(2K+Etrial+2epsilon-Delta). No false P_min or globally-invertible Haar transfer bound: the bare U1 Haar kernel is rank-one. Use a FULL Gauss fixed-number-sector trial n_s(x)=2 (even-box colour dimers/centre-cycle bundle), Ref4/0, constant U1 Haar, actual normal/colour chart fields. It lies in P_Haar range. ||P_Haar Tfhalf trial||^2>=|<trial Tfhalf trial>|^2>=exp(-K), and field Jensen costs Etrial. This trial is NOT the filled momentum sea. Q is every nonflat statistical sector; P retains all8 torons and field excitations.',
                spectral_ratio_log_upper_exact=str(2*cap+field_trial-delta)+' + 2epsilon',
                defect_total_norm_bound='epsilon=1/16 from epsilon/(number spatial edges) sum Z_e; each Z preserves onsite SU4, U1 Gauss and charge but changes adjacent statistical plaquettes.',
                field_trial_radial_ground_energy_exact=0.,
                field_trial_colour_derivative_energy_upper=str(colour_trial_upper),
                actual_full_chart_field_trial_upper=str(field_trial),
                chart_trial_energy_derivation='All species have EXACT onsite n_s(x)=2; Ref4/0 and constant U1 Haar give rho=0. The radial A†A ground has exactlyZERO energy, with flat Dirichlet boundary. Multiplying the normalized wedge2 SU4 dimer representation costs at most4096 Omega V S^2 sigma^2 in actual colour derivatives; its exact rounded bound is included in Etrial, before huge multiplicity is discarded. Parallel D cancels all V-orientation derivatives. Remaining chart curvature costs<=90V. The old Gaussian cutoff correction and45V Omega/2 oscillator energy are WITHDRAWN.',
                field_trial_is_not_original_Root_ground=True,
                declared_full_normal_field_law='H_Y=sum45 2Omega sigma^2 A†A with compact radialk and actual excited spectrum. Outside Hout has all60 Cartesian kinetic/quartic, linearPhi even-Fock and mass-resolved Eaux n on SU4/U1 Wilson paths. Tout=e^-Hout/2 P_U1Haar e^-Hout/2 after coercive lower shift and Gauss. No bareMaxwell E²/B² and no arbitraryPhi normalization.',
                prepared_normal_width=f'1e-{p.normal_width_power}', tube_radius=f'1e-{p.tube_radius_power}',
                full_field_normal_chart_tail='For one45dim blockR=r/2, integrate numerator overB_R and lower-normalization overB_(r/8): Prob(|Y|>=r/4)<=4^45 exp[-3r^2/(128sigma^2)-(32/15)sigma^2/r^2]. Union overV cells, then capped at1. Actual shell r/4<|Y|<r/2 lies INSIDE chart; outside begins |Y|>=r/2. D=Poutside e^(-Href) chi_shell+adjoint, divided by2, is a genuine nonzero quantum transition. A separate normalized tail is required for colour or full interacting thermal density; this reference-ground bound is not renamed a thermal admission.',
                normal_tail_union_log10_prefactor=tail_prefactor_log,
                normal_tail_negative_exponent_terms=[
                    dict(coefficient=3/128, power10=width_ratio_log),
                    dict(coefficient=32/15, power10=-width_ratio_log)],
                normal_coordinate_variance_upper_formula='[sigma^-2+64sigma^2/r^4]^-1',
                actual_normal_width_upper_power10=effective_width_power,
                actual_normal_width_upper_formula='min(sigma,r^2/(8sigma)); power descriptor conservatively omits1/8',
                normal_field_kinetic_coefficient_log10=field_kinetic_log,
                normal_gap_lower_log10=max(0., gap_gain_log)+math.log10(p.normal_frequency),
                normal_ground_weighted_matter_remainder_log10=exponent+p.tube_radius_power-effective_width_power+math.log10(45)/2,
                weighted_tail_commutator_scope='For bounded J and normalized reference ground rho, ||[D,J]sqrt(rho)||HS<=2||J||sqrt(p_shell) when J is parallel in chart and smoothly extended/cut off. Full thermal rho, clock derivative costs, range support, and OS/source filter Grams require actual matching; they are not supplied by this reference measure.',
                total_full_slab_matter_normal_remainder_log10=exponent,
                matter_remainder_formula='4096 V4(Naux+1) r includes inverse-B derivatives (active Bmin>=1/4), full Fock coefficient norms, contact and normalized source products BEFORE comparing to amplitude beta^-1.',
                normal_remainder_scope='Tube Bplus=Pplus(Bspatial+kY0+k||Y||^2)Pplus, Bminus=the Pminus block; F=Pplus[Cspatial+kY0+k sum_j (Y_j T_j-T_jdag Y_j)/2]Pminus. These are actual spatial-index-contracted Phi/Wilson paths, not a fixed(1,1,1) vector. B±>=Bmin-k||Y0||>=Bmin/2. T1=Ldag diag(Bplus^-1,Bminus)L and det(Bplus)Gamma(T1) are the positive extension. At Y0/colourlocked, exactly Wilson; general noncommuting colour/Y departure is paid, not claimed to be identical Grassmann D globally.',
                normalized_source_operator_cap=4,
                current_contact_full_phi_coefficient_log10=math.log10(64*fourvolume)+math.log10(species),
                chart_crossing_positive_extension='Chart product normalballsR=r/2 AND colourLieballs8sigma near eachcentre; complement actual fullCartesian/SU4 operator. C exactDelta on exitedPhiORcolour and nonflatstatistical sectors. Define Wexit=Pchart+exp(-Delta/2)Poutside, Dbar=Wexit Dcross Wexit, ||Dbar||<=exp(-Delta/2), where Dcross is offdiagonal inside-shell Cartesian/SU4 referenceheat normalized<=1. NEW declared T=e^-C/2 T0half exp[epsilon(Dstat+Dbar)] T0half e^-C/2. Ideal retains SAME Dstat on allsector histories; crossnot discarded, finite nonzero. Duhamel ||exp(epsilon(Dstat+Dbar))-exp(epsilonDstat)||<=epsilon exp(2epsilon)exp(-Delta/2). This explicitly prices virtual exit excursions INSIDE exponential; merely penalizing final outside amplitude would not. Colour-shell tail neednot be tiny.',
                outside_quartic_Lambda='1e4000',
                outside_U1_kernel='SAME barecompactU1Haar rank-one field kernel on ALLlinks, plus fullGauss. Outside unsmoothed direct-integral is not used; actual cycle Fourier trace-growth/convergent-Haar falsifier included.',
                outside_linear_matter_load_log10=math.log10(64*volume)+math.log10(naux)-photon.kappa_power,
                outside_lower_bound='For global field radius r, sum_site r_site^4>=r^4/V. Thus Lambda_eff=1e4000/V and Hout>=-(3/4)L^(4/3)/(4Lambda_eff)^(1/3), L<=64V(Naux*kappa+1). Its exact log-scale lower shift is recorded; auxiliary positive Eaux n is retained. No old Lambda256 is copied.',
                absolute_normal_state_transfer_error_log10=exponent+math.log10(64),
                separated_integer_string_length_upper=3*photon.spatial_size//2,
                separated_integer_string_induced_energy_upper=3*photon.spatial_size/(4*10.**photon.beta_power*min(photon.spacings)),
                separated_reference_energy_scope='Explicit induced quadratic electric STRING trial upper length/(2 beta a), not a claimed optimal compact interacting Coulomb eigenvalue. It is a static/reference phase/energy price; connected amplitude costs must be multiplied by their interaction grade, not treated as an unscaled amplitude error. Source remains globally body+1/Refhole-1 neutral.',
                relative_to_beta_inverse_amplitude_error_log10=exponent+math.log10(64)+photon.beta_power,
                outside_lower_shift_log10=math.log10(.75)+(4*(math.log10(64*volume)+math.log10(naux)-photon.kappa_power)-(4000-math.log10(volume))-math.log10(4))/3,
                outside_Cartesian_kinetic_coefficient=.5,
                outside_quartic_gap_log10=quartic_gap_log,
                outside_half_quartic_linear_shift_log10=half_quartic_shift_log,
                outside_quartic_gap_proof='D=60V, Lambda_eff=1e4000/V. Half-quartic absorption retains full positive Eaux*n and coefficientONE colourLap: Ebare>=3/[4*2^(2/3)] D^(4/3)(Lambda_eff/2)^(1/3)-shift. Joint Dirichlet/Gauss only raises this bound. Bare E need NOT dominate K: kappa400 makes baregap log1356 smaller than cap log1774. Actual exitedPhi OR colour endpoint carries exp(-Delta), Delta=4K+4Etrial+256, so q/lambda0<=exp[-Ebare-Delta+K+Etrial+epsilon]. The outside supplier checks the actual load, endpoint budget and normalized full primitive trace; no new quartic law or outside ground substitution.',
                outside_SU4_Laplacian_coefficient=1,
                outside_SU4_unit_time_heat_trace_s_le4=colour_heat_trace,
                outside_SU4_unit_time_heat_trace_upper=1.0001,
                outside_SU4_heat_trace_proof='L2(SU4) PeterWeyl multiplicity dim(a,b,c)^2, dim=(a+1)(b+1)(c+1)(a+b+2)(b+c+2)(a+b+c+3)/12 and eigenG=3a²+4b²+3c²+4ab+2ac+4bc+12a+16b+12c. For sumlabels=s>=5, eigen>=2s²+12s, dim<=(s+3)^6 and count=(s+1)(s+2)/2; positive tail<2*21*8^12*exp(-110)<1e-35. Displayed actual35representation sum plus this bound<1.0001. Same normalization fundamental Casimir15, not arbitrary tiny positive Laplacian.',
                outside_full_primitive_log_dimension_log10=math.log10(primitive_edges)+math.log10(math.log(2)),
                outside_bare_thermal_negative_exponent_log10=quartic_gap_log+math.log10(photon.time_slices),
                outside_thermal_negative_exponent_log10=math.log10(delta)+math.log10(photon.time_slices),
                outside_thermal_chart_trial_positive_exponent_log10=math.log10(cap+field_trial)+math.log10(photon.time_slices),
                outside_bare_gap_dominates_full_cap=quartic_gap_log > math.log10(cap+field_trial),
                outside_endpoint_residual_exact=str(delta-cap-field_trial),
                outside_thermal_trace_proof='At fixed compactA, optimized harmonicPhi plus coefficientONE SU4Lap bounds Tr exp(-pHout)<=2^E exp[-p(Ebare)](1-exp(-pomega))^-60V (1.0001)^(3V), E=(7species-1)V+4, omega=(60*1e4000)^(1/3). Positive Eaux*n and every finite Fock sector remain. Nonzero Tout spectrum is PHaar exp(-Hout) PHaar, a compactA average; Schatten convexity and fullGauss compression give the SAME bound, NOT direct-integral trace. Actual outside block has exp(-Delta) endpoint and bounded statistical exponential <=exp(epsilon), hence relative chart trace exponent is E ln2+thermalprefactors-p(Ebare+Delta-K-Etrial-epsilon). Existing Delta alone defeats full entropy and chart trial even when the OLD baregap-dominance route fails. Nonzero damped cross is paid by a relative exponential/Schatten perturbation, not discarded. The Outside supplier computes fresh Nt and Nt+1 estimates for this exact configuration.',
                full_field_bound_scope='Explicit NEW field law, full-Gauss prepared-domain construction and uniform coefficient/spectral bounds. Exact full interacting ground and all-orders Compton are not claimed.')


def joined_compton(p: RootParameters, gamma: np.ndarray) -> dict[str, Any]:
    consumer = WilsonConsumer(gamma, p.a, p.mass)
    cases = [consumer.compton(ratio, angle, direction_axis=axis)
             for ratio, angle, axis in ((.15, .4, 2), (.5, 1.2, 1), (1., 2.2, 0))]
    # Source normalization and source/current map use exact SAME b, E, rank2 poles.
    return dict(actual_same_source_current_contact_Compton=cases,
                source_operator='Ref-normalized primitive scalar source sum_c a_cdag r_c/sqrt2 selects the charged free-sea excitation; external pole residues1/b then map that canonical excitation to WilsonConsumer.',
                two_photon_polarizations=2, particle_spin_states=2,
                relation_to_native='Native T=detB exp(dGamma(log T1)) gives precisely the T1 positive/negative-time residues. Temporal equal-time contact is part of the primitive Wilson action, not an imported nearest-neighbour H.',
                observed_max_Ward=max(max(c['Ward_in'], c['Ward_out'], c['double_Ward']) for c in cases),
                observed_min_wrong_temporal_contact=min(c['adverse_controls']['temporal_contact'] for c in cases),
                scope='Connected Born coefficient with conditional chart/free positive transfer. Full-SU4/normal-field source-state transfer and all-orders finite-g probability are NOT proved.')


def all_statistical_toron_poles(p: RootParameters, photon: Parameters,
                               gamma: np.ndarray) -> dict[str, Any]:
    consumer = WilsonConsumer(gamma, p.a, p.mass)
    cases = []
    for history in itertools.product((0, 1), repeat=3):
        momentum = math.pi*np.array(history)/photon.spatial_size
        pole, spinors, b, energy = consumer.pole(momentum)
        step, _, _ = wilson_step(gamma, p.a, p.mass, momentum)
        cases.append(dict(history=list(history), momentum=momentum.tolist(),
                          actual_Wilson_pole_inverse_error=float(np.linalg.norm(consumer.inverse(pole)@spinors)),
                          positive_transfer_min=float(np.linalg.eigvalsh(step)[0]),
                          actual_spin_multiplicity=spinors.shape[1],
                          energy_physical=energy/p.a, canonical_source_residue=1/b))
    return dict(actual_all8_flat_history_poles=cases,
                maximum_source_drift_momentum_physical=math.sqrt(3)*math.pi/(photon.spatial_size*p.a),
                scope='All statistical spin structures retained and actual source poles calculated. Selected periodic-sector Born rows above are not relabeled as collisions at the shifted source momenta. The finite source drift must enter packet/recoil budgets in the joined consumer.')


def run(p: RootParameters, photon: Parameters) -> dict[str, Any]:
    start = time.monotonic()
    gamma, frame = phi_frame(photon)
    native = native_case(p, gamma)
    transported = transported_frame(gamma, photon)
    physical_geometry = physical_component_geometry_native(p, gamma, photon)
    topology = topology_and_centre(photon)
    bounds = family_bounds(p, photon)
    couplings = charge_weighted_phi_budget(p, photon)
    if max(couplings['projective_step_colour_log10'],
           couplings['projective_step_normal_log10'])/2+math.log10(photon.time_slices+2) >= -2:
        raise ArithmeticError('Declared chart does not satisfy the small projective/BKS regime')
    haar_relative = relative_haar_native(gamma)
    projective_heat = projective_heat_native(gamma)
    compton = joined_compton(p, gamma)
    gauss = full_gauss_bundle(p, photon)
    cartesian = outside_cartesian_native()
    outside_u1 = outside_u1_trace_native()
    centres = centre_current_native(gamma)
    ground = exact_chart_ground()
    preparation = positive_trace_preparation(gamma, photon)
    nonclifford = nonclifford_chart_native(gamma, photon)
    if native['actual_primitive_transfer_spectrum_error'] > 1e-8:
        raise ArithmeticError('native primitive transfer != Wilson Fock spectrum')
    if native['positivity_min'] <= 0 or native['P_Q_transition_norm'] <= 0:
        raise ArithmeticError('finite extra-sector transfer must be positive and dynamic')
    if max(native['source_SU4_commutator'], native['source_EM_commutator'], native['defect_SU4_commutator']) > 1e-8:
        raise ArithmeticError('primitive source/defect Gauss algebra failed')
    if abs(native['source_norm_squared_on_actual_native_Wilson_sea']-2) > 1e-8:
        raise ArithmeticError('actual sea source norm is not2')
    if native['exact_primitive_even_matrixunit_error'] > 1e-8 or native['wrong_commuting_bare_X_even_algebra_error'] <= 1:
        raise ArithmeticError('commuting control failed to distinguish primitive statistics')
    if native['pairing']['wrong_no_pairing_transfer_error'] <= .01:
        raise ArithmeticError('pairing control did not distinguish consumer')
    if max(gauss['finite_full_Gauss_positive_control']['actual_all15_left_right_Gauss_generator_error'],
           gauss['finite_full_Gauss_positive_control']['actual_physical_component_all15_Gauss_error']) > 1e-8:
        raise ArithmeticError('actual full SU4 Wilson-link state failed Gauss')
    if cartesian['actual_positive_T_min'] <= 0 or cartesian['wrong_linear_Phi_as_positive_B_min'] >= 0:
        raise ArithmeticError('outside Cartesian positive exponential control failed')
    if cartesian['actual_full_field_chart_transition_norm'] <= 0 or cartesian['actual_full_field_joined_positive_min'] <= 0:
        raise ArithmeticError('outside/chart transition is a no-op or not positive')
    if cartesian['wrong_outside_only_shell_transition_norm'] > 1e-10:
        raise ArithmeticError('outside-only shell unexpectedly couples chart')
    if cartesian['actual_inside_exponential_damped_chart_virtual_excursion'] >= .1*cartesian['wrong_final_penalty_only_chart_virtual_excursion']:
        raise ArithmeticError('finite exit penalty fails to suppress virtual chart excursions')
    if abs(centres['actual_chord_current_overlap']-1) > 1e-10 or centres['wrong_fixed_tree_current_overlap'] > 1e-10:
        raise ArithmeticError('centre Gauss dressing destroyed the primitive current')
    if centres['actual_ALL64_centre_branch_Wilson_transfer_covariance_error'] > 1e-10:
        raise ArithmeticError('coadjusted centre branches fail the ACTUAL Wilson transfer')
    if ground['actual_chart_ground_energy_error'] > 1e-9:
        raise ArithmeticError('actual normal-field ground factorization failed')
    if any(row['actual_normalized_ground_variance'] > row['rigorous_coordinate_variance_upper']+1e-10
           for row in ground['native_radial_width_regimes']):
        raise ArithmeticError('radial ground violates curvature variance bound')
    if haar_relative['wrong_exp_order_Haar_amplification'] < 1e6:
        raise ArithmeticError('Haar amplification falsifier did not reject exp monotonicity')
    if any(row['actual_normalized_K_difference'] > row['actual_congruence_bound']+1e-9
           for row in haar_relative['actual_primitive_rows']):
        raise ArithmeticError('actual relative-Haar congruence bound failed')
    for row in projective_heat['actual_heat_Haar_rows']:
        if any(value['actual_relative_difference'] > value['projective_bound']+1e-10
               for value in row['actual_Schatten_rows']):
            raise ArithmeticError('actual heat/Haar Schatten bridge failed')
        if any(value['actual_normalized_density_trace_error'] > value['projective_normalized_density_bound']+1e-10
               or value['actual_unnormalized_power_trace_error'] > value['projective_power_bound']+1e-10
               for value in row['actual_normalized_power_rows']):
            raise ArithmeticError('actual normalized physical trace bridge failed')
        if any(value['actual_boundary_prepared_normalized_purifier_difference'] > value['BKS_Holder_normalized_purifier_bound']+1e-10
               or value['actual_boundary_prepared_density_trace_error'] > value['BKS_Holder_prepared_density_trace_bound']+1e-10
               for value in row['actual_normalized_power_rows']):
            raise ArithmeticError('actual boundary-prepared BKS/Schatten bridge failed')
    trace_rows = outside_u1['actual_cycle_trace_rows']
    if trace_rows[-1]['wrong_unsmoothed_positive_direct_integral_trace'] < 20*trace_rows[0]['wrong_unsmoothed_positive_direct_integral_trace']:
        raise ArithmeticError('unsmoothed U1 cycle trace did not diverge')
    if abs(trace_rows[-1]['corrected_missing_trace']) > 1e-10:
        raise ArithmeticError('actual outside Haar Fourier trace has not converged')
    if nonclifford['actual_positive_transfer_min'] <= 0 or nonclifford['actual_full_Phi_remainder_transfer_change'] <= 1e-12:
        raise ArithmeticError('normal-field completion is not a real positive operator')
    if preparation['actual_normalized_positive_T_min'] <= 0 or abs(preparation['actual_prepared_density_trace']-1) > 1e-10:
        raise ArithmeticError('actual canonical full quantum preparation failed')
    if preparation['wrong_unprepared_retained_state_difference'] <= 1e-10:
        raise ArithmeticError('preparation control failed to distinguish borrowed free field')
    source_sha = SOURCE_HASHES_AT_IMPORT['covariant_root_transfer.py']
    inputs = dict(root=asdict(p), photon=asdict(photon),
                  physical_component_connection_law=PHYSICAL_COMPONENT_CONNECTION_LAW,
                  positive_transfer_matrix_basis='physical-reference-Theta-block-ThetaDag-20261008')
    canonical = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
    return dict(branch='NEW transported-Phi covariant Root; declared quantum/metric/potential inputs',
                governing_plan='issue-222-induced-photon-delta13-continuation-20261008',
                governing_revision='d21e32f5b786dc66274299d2b39a81fd9675c4a4572ab82f19586755b6d5da40',
                configuration_ABI=dict(label=f'kappa{photon.kappa_power}_sigma{p.normal_width_power}_r{p.tube_radius_power}',
                    canonical_parameters_sha256=hashlib.sha256(canonical).hexdigest(),
                    physical_component_connection_law=PHYSICAL_COMPONENT_CONNECTION_LAW,
                    positive_transfer_matrix_basis=inputs['positive_transfer_matrix_basis'],
                    Hamiltonian_epoch='NEW physical-component kinetic AND potential; bd4d/931 cancellation-invalid',
                    source_charge=p.source_charge, auxiliary_charge=1,
                    auxiliary_count_exact=str(10**(photon.beta_power+4*photon.kappa_power)//16),
                    same_functional_preparation=True, identical_historical_state=False),
                microscopic_source_admission=dict(source_Gram_lower=None,photon_Gram_lower=None,
                    energy_band_Gram_lower=None,high_energy_weight_upper=None,admitted=False),
                parameters=inputs, frame=frame, native=native, transported_Phi=transported,
                actual_physical_component_geometry=physical_geometry,
                topology_and_Gauss=topology, renewed_family_bounds=bounds,
                exact_selected_frame_plaquette=loop_coefficient(gamma), joined_consumer=compton,
                actual_full_Gauss_bundle=gauss, actual_outside_Cartesian_operator=cartesian,
                actual_outside_U1_trace_control=outside_u1,
                actual_centre_current_transfer=centres, actual_massive_chart_field=ground,
                actual_positive_quantum_preparation=preparation,
                actual_weighted_Phi_coupling=couplings,
                actual_primitive_Haar_relative_control=haar_relative,
                actual_projective_heat_Haar_Schatten_control=projective_heat,
                actual_nonclifford_Phi_transfer=nonclifford,
                actual_statistical_toron_sources=all_statistical_toron_poles(p, photon, gamma),
                scientific_admission=False,
                unmet=['All-orders normalized finite-g scattering probability is owned by the joined packet consumer; the connected Born coefficient is not that probability.',
                       'The prepared construction is a new declared field/connection law, NOT a derivation of the frozen continuous-time Phi potential or a proof of its interacting vacuum.'],
                elapsed_seconds=time.monotonic()-start,
                capsule=dict(source_sha256=source_sha, input_sha256=hashlib.sha256(canonical).hexdigest(),
                             loaded_code_matches_captured_source=LOADED_ROOT_CODE_MATCH,
                             source_sha256_at_IMPORT=SOURCE_HASHES_AT_IMPORT,
                             source_bytes_at_IMPORT={name: data.decode() for name, data in SOURCE_BYTES_AT_IMPORT.items()},
                             python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                             executable=sys.executable,
                             executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'covariant_root_transfer_capsule_physical_Dminus.json')
    parser.add_argument('--normal-width-power', type=int, default=RootParameters().normal_width_power)
    parser.add_argument('--tube-radius-power', type=int, default=RootParameters().tube_radius_power)
    parser.add_argument('--kappa-power', type=int, default=250)
    parser.add_argument('--beta-power', type=int, default=150)
    parser.add_argument('--spatial-size', type=int, default=2**24)
    parser.add_argument('--time-slices', type=int, default=2**20)
    args = parser.parse_args()
    result = run(RootParameters(normal_width_power=args.normal_width_power,
                                tube_radius_power=args.tube_radius_power),
                 Parameters(kappa_power=args.kappa_power, beta_power=args.beta_power,
                            spatial_size=args.spatial_size, time_slices=args.time_slices))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(natural_exit=0, output=str(args.output), scientific_admission=False,
                         source_sha256=result['capsule']['source_sha256'],
                         input_sha256=result['capsule']['input_sha256'],
                         native_positive_min=result['native']['positivity_min'],
                         native_PQ=result['native']['P_Q_transition_norm'],
                         source_norm2=result['native']['source_norm_squared_on_actual_native_Wilson_sea'],
                         elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()
