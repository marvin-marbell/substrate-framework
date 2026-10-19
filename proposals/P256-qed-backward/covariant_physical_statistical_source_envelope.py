#!/usr/bin/env python3
"""Standalone physical Dstat/source envelope for frozen Source2a and Root6911.

The common cosh scalar is removed once, not once per polymer or toron.
No reference Slater identity is an interacting source admission.  The native
carrier is an exact Pauli subgroup/syndrome carrier, not a degree-two ring.
"""
from __future__ import annotations

import argparse
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

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parent
FROZEN_SOURCE = '2a0ecfb2341fceb7866d5062cf33d1da71b2fafa4b148829d21d14f0650fe307'
FROZEN_ROOT = '6911f0283dc0f84f4cc05202deb67dcd3266d5c64ca1a05e149f1567451e0f9c'
SELECTED_CAP = 'covariant_prepared_gauss_state_conditioned_source_physical_Dminus_verified_20261008.json'
SELECTED_CAP_SHA = 'ae1c756921a90752bf580d4c149d8c4c258224f291303dcc969245f9d543db1b'
CAPTURED: dict[str, bytes] = {Path(__file__).name: Path(__file__).read_bytes()}


class CaptureLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        data = Path(self.path).read_bytes()
        CAPTURED[Path(self.path).name] = data
        return compile(data, self.path, 'exec')


class CaptureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.origin and Path(spec.origin).parent == ROOT and spec.origin.endswith('.py'):
            spec.loader = CaptureLoader(fullname, spec.origin)
        return spec


_finder = CaptureFinder()
sys.meta_path.insert(0, _finder)
import numpy as np
import mpmath as mp
from scipy.linalg import expm
import covariant_root_transfer as root
from covariant_induced_photon import Parameters, WilsonConsumer, phi_frame
from induced_root_transfer import FlavorPauliGraph, raw_polynomial
from microscopic_native_orbit_lift import PauliWord
from improved_phi_band import ImprovedPhiLaw
sys.meta_path.remove(_finder)


def require(value: bool, reason: str) -> None:
    if not value:
        raise ArithmeticError(reason)


def family_parameters() -> tuple[Any, Any]:
    return root.RootParameters(normal_width_power=2000, tube_radius_power=2100), Parameters(kappa_power=400)


def standalone_transfer(tf0half: np.ndarray, C: np.ndarray, Dstat: np.ndarray,
                        epsilon: float = 1/16) -> np.ndarray:
    """Literal positive sandwich; caller supplies the actual sum Z_e/M."""
    outer = expm(-C/2)
    return outer@tf0half@expm(epsilon*Dstat)@tf0half@outer


def statistical_prices() -> dict[str, Any]:
    """Uniform operator bound after ONE common scalar; all torons retained."""
    _, photon = family_parameters()
    volume = photon.spatial_size**3
    species = 10**1750//16+1
    M = 3*volume*species
    with mp.workdps(100):
        eps = mp.mpf(1)/16
        x = eps/M
        tau = mp.tanh(x)
        # log1p retains the tiny positive answer that log(cosh(x)) rounds away.
        log_cosh = M*mp.log1p(2*mp.sinh(x/2)**2)
        polymer_exponent = M*(48*tau)**6/(1-48*tau)
        closed = mp.expm1(polymer_exponent)
        return dict(M_exact=str(M), species_exact=str(species), volume_exact=str(volume), epsilon='1/16',
            edge_exponent_log10=float(mp.log10(x)), tau_log10=float(mp.log10(tau)),
            common_scalar='c=cosh(epsilon/M)^M', common_scalar_log=mp.nstr(log_cosh, 40),
            common_scalar_minus_one_log10=float(mp.log10(mp.expm1(log_cosh))),
            common_scalar_slab_log=mp.nstr(photon.time_slices*log_cosh, 40),
            common_scalar_slab_minus_one_log10=float(mp.log10(mp.expm1(photon.time_slices*log_cosh))),
            common_scalar_removed_exactly_once=True, dual_face_adjacency_degree=12,
            rooted_animal_constant=48, macro_minimum_connected_closed_component=6,
            polymer_exponent=mp.nstr(polymer_exponent, 40), closed_relative_upper_log10=float(mp.log10(closed)),
            closed_Nt_plus2_relative_upper_log10=float(mp.log10(mp.expm1((photon.time_slices+2)*polymer_exponent))),
            flat_to_nonflat_exponential_norm_log10_upper=float(mp.log10(eps)+eps/mp.log(10)-mp.log10(M)/2),
            formula='P exp(epsilon Dstat) P=c sum_{S cubic-closed} tau^|S| P Z_S P; terms violating full local species/Ref constraints vanish additionally.',
            support='ONLY spatial anchor edges of EVERY species; no onsite, nearest-species-tree or reference edges.',
            projection_proof='The Z_e commute and square to I, so the product expansion is exact. A term anticommutes a local cycle iff its primal edge mask meets that cycle oddly; P Z_S P=0 in that case. Species cubic closure is necessary. Extra constraints remove terms, never add terms. Each retained compression has norm <=1. No separate-character positivity is used.',
            animal_proof='Dual faces are adjacent when sharing a dual edge. Each face has four edges, each incident to at most three other faces: degree12. A rooted size-k animal has a deterministic ordered spanning tree; plane-tree shapes <=4^(k-1), and each tree step <=12 choices. Hence <=48^(k-1)<=48^k. Root every connected component at its least face. Relax disjointness and closure to independent polymers: product(1+tau^size)<=exp(sum tau^size)<=exp[M sum_{k>=6}(48tau)^k]. This bounds the nonempty-subset norm by exp[M(48tau)^6/(1-48tau)]-1.',
            minimum_size_proof='A cubic cocycle is a Z2 closed dual cubical surface. For a component with k<6<=L, every fundamental adjacency loop has length<=k<L and cannot wind a periodic coordinate. Thus the entire closed component has a consistent finite lift to Z^3. Its primal cocycle is a finite vertex cut delta A, choosing the finite side (Z^3 has one end). Every nonempty finite A has at least two boundary edges in each axis: take an occupied coordinate line and its two extreme vertices. Hence the cut has at least6 edges/faces. The six faces of a dual cube attain equality. Noncontractible sheets have area>=L^2 and remain retained.',
            ordering='W_C Tf0half exp(epsilon Dstat) Tf0half W_C, W_C=exp(-C/2). Statistical cycles commute every primitive A/B and hence full even matter, canonical projectors, SF/positive roots, integer sources and field/Haar/Gauss factors. Individual Z_e do NOT commute spatial hops or separated sources.',
            sectors='P imposes only local statistical flatness. ALL8 common statistical torons, all Fock occupations and all Phi/SU4/U1 sectors retained.',
            actual_macro_Grams=None, microscopic_source_admission_proved=False)


def nonflat_penalty_prices() -> dict[str, Any]:
    """Full ROOT neutral-trial normalization, never charged contraction."""
    rp, photon = family_parameters()
    bounds = root.family_bounds(rp, photon)
    K = int(bounds['full_B_F_log_transfer_norm_cap_exact'])
    E = int(bounds['actual_full_chart_field_trial_upper'])
    Delta = int(bounds['renewed_cycle_penalty_exact'])
    M = 3*photon.spatial_size**3*(10**1750//16+1)
    require(Delta == 4*K+4*E+256, 'Actual Root exact Delta ledger')
    with mp.workdps(100):
        eps = mp.mpf(1)/16
        q_exp = 2*K+E-Delta
        off_exp = 2*K+E-Delta//2
        return dict(K_exact=str(K), Etrial_exact=str(E), Delta_exact=str(Delta),
            colour_trial_exact=bounds['field_trial_colour_derivative_energy_upper'],
            diagonal_ratio_log_e_integer=str(q_exp), diagonal_ratio_log_e_fraction='1/8',
            diagonal_ratio_log10_upper=mp.nstr((mp.mpf(q_exp)+2*eps)/mp.log(10), 50),
            offblock_ratio_log_e_integer=str(off_exp),
            offblock_ratio_log10_upper=mp.nstr((mp.mpf(off_exp)+2*eps)/mp.log(10)+mp.log10(eps)-mp.log10(M)/2, 50),
            offblock_cancellation_identity='2K+Etrial-Delta/2=-Etrial-128',
            trial='Actual neutral fixedN2/site/species Gauss trial: even-box wedge2 SU4 Wilson dimers, Ref4/0, constant compact-U1 Haar, actual radial ground and colour chart. rho=0, canonical N_s=2V. NOT the original Root ground or filled momentum sea.',
            trial_norm_proof='The positive matter factor has log-norm ledger K and inverse bound on the finite matter fibre ONLY. The neutral trial lies in bareHaar range: ||PHaar Tfhalf trial||^2 >= |<trial,Tfhalf trial>|^2 >=exp(-K). Field Jensen costs Etrial. Including the statistical factor gives lambda_neutral>=exp(-K-Etrial-epsilon). Haar itself has a nullspace; no global Tmin exists.',
            diagonal_proof='Every nonflat endpoint has W_C<=exp(-Delta/2). Thus ||QTQ||<=exp(K+epsilon-Delta), divided by the actual neutral trial lambda gives exp(2K+Etrial+2epsilon-Delta). Q contains ALL nonflat occupations/fields, not a charged low-energy subset.',
            offblock_proof='Cycle preservation gives [Tf0half,P]=0. Unique nonzero spatial-edge syndromes imply P Dstat^2 P=P/M. Thus ||Q exp(epsilon Dstat)P||<=epsilon exp(epsilon)/sqrt(M). ONE nonflat endpoint contributes exp(-Delta/2), and ||Tf0half||^2<=exp(K). Divide by the SAME neutral trial lower bound: epsilon/sqrt(M) exp(2K+Etrial+2epsilon-Delta/2). No charged range/halfstep is assumed contractive.',
            full_Phi_scope='Uniform Root chart/field norm and neutral trial ledger. Damped chart-cross Dbar is a SEPARATE perturbation, not included in standalone exp(epsilon Dstat). Outside Phi/colour endpoints are still penalized by the same exact C.',
            induced_field_scope='No Maxwell replacement, induced Gaussian field state, source/toron Gram, weighted BKS or interacting Slater admission supplied.',
            actual_macro_Grams=None, microscopic_source_admission_proved=False)


def beam_projectors(size: int) -> list[dict[str, Any]]:
    """Literal Source2a momenta/projectors; illustrative reference amplitudes."""
    _, p = family_parameters()
    gamma, _ = phi_frame(p)
    consumer = WilsonConsumer(gamma, .01, 1.)
    offsets = np.arange(-1, 2)
    # Photon packet reference 978-980, NOT a separately pinned Source2a f_k.
    # Macro bounds below do not depend on this illustrative native choice.
    amplitudes = np.exp(-offsets*offsets/4)
    amplitudes /= np.linalg.norm(amplitudes)
    rows = []
    for eta in itertools.product((0, 1), repeat=3):
        beam = np.column_stack((offsets*2*math.pi/size, np.zeros(3), np.zeros(3)))+math.pi*np.array(eta)/size
        projectors, energies = [], []
        for k in beam:
            T1, _, _ = root.wilson_step(gamma, .01, 1., k)
            values, vectors = np.linalg.eigh(T1)
            require(int(np.sum(values < 1)) == 2, 'Canonical stable rank2 source projector')
            frame = vectors[:, values < 1]
            projectors.append(frame@frame.conj().T)
            energies.append(consumer.energy(k))
        rows.append(dict(eta=eta, beam=beam, amplitudes=amplitudes, projectors=projectors, energies=np.array(energies)))
    return rows


def coherent_source_prices() -> dict[str, Any]:
    """Wholepacket coefficient-nuclear sum; no single-path relative shortcut."""
    _, p = family_parameters()
    volume = p.spatial_size**3
    M = 3*volume*(10**1750//16+1)
    beam = beam_projectors(p.spatial_size)
    rows = []
    max_ledger = 0.
    for row in beam:
        weighted_norm2 = math.exp(float(max(row['energies'])))
        max_ledger = max(max_ledger, weighted_norm2)
        rows.append(dict(statistical_toron=list(row['eta']), momenta=row['beam'].tolist(),
            energies=row['energies'].tolist(), exp_E_half=np.exp(row['energies']/2).tolist(),
            coefficient_Frobenius_sum_squared_uniform_upper=weighted_norm2,
            projector_idempotence_error=max(float(np.linalg.norm(P@P-P)) for P in row['projectors'])))
    require(max(float(max(r['energies'])) for r in rows) < .011, 'Frozen macro source energies below .011')
    # C_x has four body/reference indices.  Nuclear <=2 Frobenius; Parseval
    # on the toron-shifted Fourier basis gives sum_x ||C_x||F^2 exactly.
    nuclear_sum_cap = 2*math.sqrt(volume*max_ledger)
    length = 3*p.spatial_size//2
    with mp.workdps(100):
        log_D = mp.log10(2*length)+mp.log10(nuclear_sum_cap)-mp.log10(M)
        log_exp = log_D+mp.log10(mp.mpf(1)/16)+mp.mpf(1)/(16*mp.log(10))
        return dict(all8_beam_rows=rows, canonical_projector_before_integer_paths=True,
            original_source_preserved=True, integer_path_length_upper=length,
            amplitude_scope='Uniform over ALL complex three-beam coefficients with sum|a_k|^2=1. No Source2a f_k is replaced or asserted pinned.',
            amplitude_authority='Source2a1773 explicitly specifies beam l2 norm1 and Ee<.011. exp(-offset^2/4) appears in Photon978-980 as a reference packet; used only by native illustrative smoke.',
            whole_coherent_nuclear_sum_upper=nuclear_sum_cap,
            source_Dstat_commutator_norm_log10_upper=float(log_D),
            source_exponential_commutator_norm_log10_upper=float(log_exp),
            single_path_only_exponential_log10_upper=float(mp.log10(2*length)+mp.log10(mp.mpf(1)/16)+mp.mpf(1)/(16*mp.log(10))-mp.log10(M)),
            sqrt_volume_nuclear_price_log10=math.log10(nuclear_sum_cap),
            exact_source='Spre=sum_x,alpha,beta C_x[alpha,beta] a_alpha^dag(x; SAME integer SU4/U1/primitive path to Ref0) r_beta. C_x=1/sqrt(2V) sum_{3 beams} a_k exp(Ee(k)/2) exp(ik.(x-carrier)) P_el(k), carrier=(L/4,0,0). Transport P_el covariantly BEFORE the integer path. Ref4->3 and bodycharge1 unchanged.',
            coefficient_proof='For each toron the three Fourier characters are orthogonal. P_el has rank2, so sum_x||C_x||F^2=sum_k|a_k|^2 exp(Ee(k)) exactly, <=exp(max_k Ee(k)) for EVERY complex l2-normalized three-beam coefficient vector. Unitary Phi/SU4/U1 transport common to all beams at x preserves Frobenius/nuclear norm. ||C_x||nuc<=2||C_x||F; Cauchy gives sum_x||C_x||nuc<=2sqrt(V)exp(max_k Ee(k)/2). SVD at each x expresses the actual source into rank1 even primitive transfers of norm<=1. This is a WHOLE coherent source bound uniform over the original unknown coefficients; no factorization of its interacting Gram.',
            commutator_proof='Z_e commutes coefficient field matrices, local canonical projectors, onsite CAR and SU4/U1 link-coordinate multipliers. It anticommutes path_A precisely on its spatial X support. Therefore ||[Dstat,Spre]||<=2/M sum_x ell_x||C_x||nuc<=2ell_max nuclear_sum_cap/M. Duhamel gives ||[exp(epsilon Dstat),Spre]||<=epsilon exp(epsilon)||[Dstat,Spre]||. Never divide by an unproved ||Spre|| lower bound.',
            filledRef4_chart_CAR_norm_squared_upper=max_ledger,
            filledRef4_chart_CAR_scope='Uniform over all l2-normalized three-beam coefficients: exact bare source norm cap on transported chart; not charged halfstep or Pe retention, not macro Gram1.',
            fullPhi_outside_global_source_cap=8,
            fullPhi_outside_exponential_commutator_upper=16*math.exp(1/16),
            fullPhi_outside_scope='Source2a outside extension has global normcap8. Without an outside coefficient/parallel-path ledger only trivial ||[exp(epsilon Dstat),S]||<=16exp(epsilon) is claimed outside. Chart nuclear-price dilution is NOT silently applied to arbitrary fullPhi source extensions. Actual outside-state weight and inserted Grams remain separate.',
            induced_field_scope='Integer induced-U1 paths remain literal unitaries. No Gaussian reference source/field measure substituted for the physical kernel.',
            actual_macro_Grams=None, microscopic_source_admission_proved=False)


def shortest_body_to_reference(law: Any, site: int) -> tuple[int, ...]:
    coords = law.sites[site].copy()
    path = [site]
    # Source2a literal x,z,y order, negative direction on half-box ties.
    for axis in (0, 2, 1):
        coordinate = int(coords[axis])
        signed = -coordinate if coordinate <= law.size//2 else law.size-coordinate
        for _ in range(abs(signed)):
            coords[axis] = (coords[axis]+(1 if signed > 0 else -1)) % law.size
            path.append(law.index[tuple(coords)])
    require(path[-1] == 0, 'Integer source path terminates at Ref site0')
    return tuple(path)


def native_cubic_smoke(size: int = 6) -> dict[str, Any]:
    """Actual bit-packed primitive cubic graph and faithful six-Z carrier."""
    require(size >= 6, 'Native cubic minimum-six probe needs L>=6, not a ring')
    law = ImprovedPhiLaw(size)
    graph = FlavorPauliGraph(law, 1)
    spatial_set = set(graph.spatial_edges)
    generators = (*graph.B, *graph.A.values())
    cycles = (*graph.cycles, *graph.local_cycles)
    cycle_errors = sum(word.anticommutes(cycle) for word in generators for cycle in cycles)
    require(cycle_errors == 0, 'All actual primitive bilinears preserve statistical cycles')
    syndromes = [sum(((face.x >> edge)&1) << j for j, face in enumerate(graph.plaquettes)) for edge in graph.spatial_edges]
    require(len(set(syndromes)) == len(syndromes) and all(syndromes), 'Actual distinct nonzero cubic syndromes')
    star_edges = tuple(e for e in graph.incident[graph.mode(1, 0, 0)] if e in spatial_set)
    require(len(star_edges) == 6, 'Actual spatial cubic star has six edges')
    star_mask = sum(1 << e for e in star_edges)
    joint_mask = star_mask | sum(1 << e for e in graph.incident[graph.mode(0, 0, 0)] if e in spatial_set)
    star = PauliWord(z=star_mask)
    joint = PauliWord(z=joint_mask)
    star_cubic_errors = sum(star.anticommutes(face) for face in graph.plaquettes)
    star_full_errors = sum(star.anticommutes(face) for face in graph.local_cycles)
    require(star_cubic_errors == 0 and star_full_errors > 0, 'Extra species constraints only reduce star support')
    require(not any(joint.anticommutes(face) for face in graph.local_cycles), 'Joint species star survives full local constraints')
    toron_rows = []
    for axis in range(3):
        mask = 0
        for species in range(graph.species):
            for x, coord in enumerate(law.sites):
                if int(coord[axis]) == size-1:
                    y = int(law.target[x, axis])
                    edge = graph.edge_index[tuple(sorted((graph.mode(species, x, 0), graph.mode(species, y, 0))))]
                    mask |= 1 << edge
        sheet = PauliWord(z=mask)
        local_errors = sum(sheet.anticommutes(face) for face in graph.local_cycles)
        body = tuple(graph.mode(1, law.index[tuple(j if a == axis else 0 for a in range(3))], 0) for j in range(size))
        winding_loop = graph.cycle(body)
        require(local_errors == 0 and sheet.anticommutes(winding_loop), 'Closed toron sheet flips winding but no local cycle')
        toron_rows.append(dict(axis=axis, spatial_support=mask.bit_count(), local_cycle_errors=local_errors,
            winding_flip=sheet.anticommutes(winding_loop)))
    # Exact regular representation of the native six Z involutions.  The
    # diagonal P tests the actual plaquette syndrome of each subgroup element;
    # it retains the closed-star +/- occupation characters, not only one.
    dimension = 1 << 6
    localZ = []
    indices = np.arange(dimension)
    for bit in range(6):
        matrix = np.zeros((dimension, dimension))
        matrix[indices ^ (1 << bit), indices] = 1
        localZ.append(matrix)
    masks = [sum(1 << e for bit, e in enumerate(star_edges) if (subset >> bit)&1) for subset in range(dimension)]
    keep = np.array([not any((mask&face.x).bit_count()%2 for face in graph.plaquettes) for mask in masks])
    require(np.flatnonzero(keep).tolist() == [0, 63], 'Only empty/full six-star subsets are cubic-closed')
    P = np.diag(keep.astype(float))
    a = .13
    Zstar = np.linalg.multi_dot(localZ)
    native_A = graph.A[graph.edges[star_edges[0]]]
    Acarrier = np.diag([(-1.)**((mask&native_A.x).bit_count()%2) for mask in masks])
    physical_half = expm(.11*Acarrier)
    Ccarrier = 4*(np.eye(dimension)-P)
    local_Dstat = np.add.reduce(localZ)/len(graph.spatial_edges)
    ordered = standalone_transfer(physical_half, Ccarrier, local_Dstat)
    outer = expm(-Ccarrier/2)
    wrong_order = outer@expm(local_Dstat/16)@physical_half@physical_half@outer
    ordering_adverse = float(np.linalg.norm(ordered-wrong_order, 2))
    require(ordering_adverse > 1e-8, 'Moving statistical exponential past actual primitive half changes transfer')
    require(float(np.linalg.eigvalsh(ordered)[0]) > 0, 'Literal finite sandwich remains positive')
    exponential = expm(a*sum(localZ))
    scalar = math.cosh(a)**6
    predicted = scalar*(P+math.tanh(a)**6*P@Zstar@P)
    projection_error = float(np.linalg.norm(P@exponential@P-predicted, 2))
    require(projection_error < 2e-14, 'Observable native subgroup cosh/closed-star projection')
    # Source2a coherent coefficients on the finite cubic graph.  Its native
    # momenta/energies are not renamed macro prices.  All eight sectors run.
    spatial_mask = sum(1 << e for e in graph.spatial_edges)
    source_cycle_cache: dict[tuple[int, int], int] = {}
    source_rows = []
    for row in beam_projectors(size):
        terms = []
        nuclear_sum = 0.
        for x, coords in enumerate(law.sites):
            phase = np.exp(1j*(row['beam']@(coords-np.array([size/4, 0., 0.]))))
            C = sum((amp*math.exp(float(E)/2)*ph*proj for amp,E,ph,proj in zip(row['amplitudes'],row['energies'],phase,row['projectors'])), np.zeros((4, 4), complex))/math.sqrt(2*law.volume)
            nuclear_sum += float(np.linalg.norm(C, 'nuc'))
            body = shortest_body_to_reference(law, x)
            anchors = tuple(graph.mode(1, site, 0) for site in reversed(body))
            for alpha, beta in itertools.product(range(4), repeat=2):
                if abs(C[alpha, beta]) < 1e-16:
                    continue
                path = ((graph.ref[beta], graph.ref[0]) if beta else (graph.ref[0],))+anchors
                if alpha:
                    path += (graph.mode(1, x, alpha),)
                term = graph.bilinear(graph.mode(1, x, alpha), graph.ref[beta], complex(C[alpha, beta]), path)
                terms.extend(term.words)
        polynomial = raw_polynomial(terms)
        commutator_terms = []
        source_cycle_errors = 0
        for (px, pz), coefficient in polynomial.items():
            word = PauliWord(px, pz)
            key = px, pz
            if key not in source_cycle_cache:
                source_cycle_cache[key] = sum(word.anticommutes(cycle) for cycle in graph.local_cycles)
            source_cycle_errors += source_cycle_cache[key]
            traversed = px & spatial_mask
            while traversed:
                bit = traversed & -traversed
                Z = PauliWord(z=bit)
                commutator_terms.append((2*coefficient/len(graph.spatial_edges), Z@word))
                traversed ^= bit
        require(source_cycle_errors == 0, 'Actual coherent integer source preserves local cycles')
        commutator = raw_polynomial(commutator_terms)
        hs_lower = math.sqrt(sum(abs(c)**2 for c in commutator.values()))
        require(hs_lower > 1e-6, 'Adverse source MUST NOT commute Dstat')
        upper = 2*(3*size//2)*nuclear_sum/len(graph.spatial_edges)
        require(hs_lower <= upper, 'Observable full-edge normalized HS commutator below nuclear bound')
        source_rows.append(dict(statistical_toron=list(row['eta']), actual_Pauli_source_terms=len(polynomial),
            actual_commutator_terms=len(commutator), source_cycle_errors=source_cycle_errors,
            Dstat_source_commutator_normalized_HS=hs_lower, commutator_operator_norm_lower=hs_lower,
            whole_coherent_source_nuclear_sum=nuclear_sum, commutator_operator_norm_upper=upper))
    return dict(size=size, species=graph.species, modes=graph.modes, edges=len(graph.edges),
        spatial_edges=len(graph.spatial_edges), nonspatial_edges_excluded=len(graph.edges)-len(graph.spatial_edges),
        cycle_primitive_commutation_errors=cycle_errors, cubic_syndromes_unique=len(set(syndromes)),
        star_spatial_support=len(star_edges), star_cubic_errors=star_cubic_errors,
        star_additional_species_cycle_errors=star_full_errors, joint_species_star_support=joint_mask.bit_count(),
        joint_species_star_full_local_errors=0, toron_sheet_rows=toron_rows,
        retained_statistical_torons=8, Z_involution_error=float(np.linalg.norm(localZ[0]@localZ[0]-np.eye(dimension))),
        exact_native_Z_subgroup_dimension=dimension, projected_subgroup_dimension=int(np.sum(keep)),
        cosh_closed_star_projection_operator_error=projection_error,
        scalar_subtracted_flat_operator_eigenvalues=np.linalg.eigvalsh(predicted[np.ix_(keep, keep)]/scalar).tolist(),
        literal_ordering_adverse_operator_error=ordering_adverse,
        literal_sandwich_positive_min=float(np.linalg.eigvalsh(ordered)[0]),
        literal_ordering_scope='Exact actual native A/selected-six-Z subalgebra carrier, using epsilon=1/16 and native full M denominator. Other spatial Z are omitted ONLY in this finite ordering probe, not in macro bound.',
        wrong_flat_identity_operator_error=float(np.linalg.norm(P@exponential@P/scalar-P, 2)),
        finite_carrier_scope='Faithful regular representation of SIX actual cubic-star spatial Z involutions and their native cubic syndromes, retaining both closed-star characters. Not the full 2^E edge Hilbert space. Whole graph primitive/cycle/source algebra is exact symbolic Pauli. No degree2-ring minimum-six claim.',
        native_source_amplitude_scope='Illustrative normalized exp(-offset^2/4) reference coefficients from Photon978-980, not asserted pinned Source2a f_k. Source2a momenta, canonical projector/path ordering and compensation are preserved; macro pricing is independent of this illustrative choice.',
        all8_actual_coherent_source_adverse_rows=source_rows,
        actual_macro_Grams=None, microscopic_source_admission_proved=False)


def run() -> dict[str, Any]:
    start = time.perf_counter()
    for name, expected in (('covariant_prepared_gauss_state.py', FROZEN_SOURCE), ('covariant_root_transfer.py', FROZEN_ROOT)):
        data = (ROOT/name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, 'Frozen scientific source mismatch: '+name)
        CAPTURED[name] = data
    cap_data = (ROOT/SELECTED_CAP).read_bytes()
    require(hashlib.sha256(cap_data).hexdigest() == SELECTED_CAP_SHA, 'Original ae1 selected capsule, not dd73 final')
    payload: dict[str, Any] = dict(plan='issue-222-induced-photon-delta13-continuation-20261008',
        governing_revision='1124ce49865f8f7cc5f912ef9569edc52ad3227fb532fd78d5637658c923b990',
        interface_revision='standalone-physical-statistical-source-envelope-20261009',
        frozen_source_sha256=FROZEN_SOURCE, frozen_root_sha256=FROZEN_ROOT,
        selected_original_capsule=dict(path=SELECTED_CAP, sha256=SELECTED_CAP_SHA),
        parameters=dict(kappa_power=400, beta_power=150, Naux_recipe='10**1750//16', L=2**24, Nt=2**20,
            normal_width='1e-2000', tube_radius='1e-2100'),
        statistical=statistical_prices(), nonflat=nonflat_penalty_prices(), source=coherent_source_prices(),
        native=native_cubic_smoke(), actual_macro_Grams=None, microscopic_source_admission_proved=False,
        limitations=['Standalone statistical exponential only; damped chart-cross/fullPhi perturbation and induced field comparison remain separate.',
            'No postsource, Pe, photon, f_R or charged halfstep retained Gram proved; no change of original source, preparation, receiver or clock.',
            'FullPhi outside source extension receives only global normcap commutator, not the chart nuclear dilution.',
            'Neutral Gauss trial proves denominator/norm lower only, not original state identification.',
            'No global Tmin, scalar RN-to-Loewner, weighted BKS, independent-character positivity or interacting Slater assumption.'])
    # Root's inner loader captured the transitive modules while it was ahead
    # of our finder.  Reuse those exact compiled bytes, not post-run rereads.
    CAPTURED.update(root.SOURCE_BYTES_AT_IMPORT)
    sources = {}
    for name, data in sorted(CAPTURED.items()):
        require((ROOT/name).read_bytes() == data, 'Source changed during runtime: '+name)
        sources[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), captured_source=data.decode())
    payload['provenance'] = dict(executable=sys.executable, python=platform.python_version(), argv=sys.argv,
        sources=sources, numpy=np.__version__, elapsed_seconds=time.perf_counter()-start,
        runtime_completed=True, natural_exit_expected_code=0)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Only NEW named JSON capsules may be written')
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), closed_log10=result['statistical']['closed_relative_upper_log10'],
        source_commutator_log10=result['source']['source_exponential_commutator_norm_log10_upper'],
        native_projection_error=result['native']['cosh_closed_star_projection_operator_error'],
        all8_source_adverse_verified=True, microscopic_source_admission_proved=False), indent=2))


if __name__ == '__main__':
    main()
