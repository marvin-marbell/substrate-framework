#!/usr/bin/env python3
"""Local Pauli original-field law and finite auxiliary-sector transfer.

No microscopic occupation-sign operators: CAR occurs in code_isometry comparison
only. Large-family counts/budgets are symbolic; native matrices are 8-edge probes.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, kron as sparse_kron, eye as sparse_eye
from scipy.sparse.linalg import expm_multiply, norm as sparse_norm

from bosonic_statistics import (BosonicEdgeModel, tensor_pauli, reference_hop,
                               exchange_report)
from local_orbital_stability import scalar_paths
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis
from noncompact_coulomb_carrier import phase_galerkin

ROOT = Path(__file__).resolve().parent
NSTAR = 5 * 2**96


def original_endpoint(phi_x, phi_y, link, axis, band):
    """Actual h[x,y] endpoint average; link convention from BulkQuantumLaw."""
    _, paths = scalar_paths(band)
    scalar = next(s for d, s in paths if d[axis] == 1)
    fx = np.einsum('m,mij->ij', scalar, phi_x)
    fy = np.einsum('m,mij->ij', scalar, phi_y)
    return (fx @ link + link @ fy) / 2


class OriginalBosonicLaw:
    """Lazy full heavy/light original-Phi coefficient supplier.

    Each output is a local Pauli-even bilinear coefficient, not a CAR matrix.
    Site callbacks retrieve only the selected site and its three neighbours.
    Fields remain arbitrary Cartesian traceless Hermitian, links arbitrary SU4,
    and EM is a real noncompact coordinate. No selected Dirac frame is input.
    """
    def __init__(self, size, band, heavy_scale, penalty, defect_coupling):
        if size < 3 or min(heavy_scale, penalty, defect_coupling) <= 0:
            raise ValueError('Finite positive scales and periodic N>=3 required')
        self.size, self.band, self.heavy_scale = size, band, heavy_scale
        self.penalty, self.defect_coupling = penalty, defect_coupling
        self.onsite, self.paths = scalar_paths(band)

    def local_matter_coefficients(self, site, field_at, colour_link_at, em_at):
        site = tuple(site)
        phi = field_at(site)
        for species, scale in (('H', -self.heavy_scale), ('L', 1)):
            onsite = scale*self.onsite*phi[0]
            for i,j in itertools.product(range(4), repeat=2):
                yield ((species,site,i),(species,site,j),onsite[i,j],None)
            for axis in range(3):
                target=list(site);target[axis]=(target[axis]+1)%self.size
                target=tuple(target)
                coefficient=scale*original_endpoint(phi,field_at(target),colour_link_at(site,axis),axis,self.band)
                if species == 'L':
                    coefficient=coefficient*np.exp(1j*em_at(site,axis))
                for i,j in itertools.product(range(4), repeat=2):
                    # h[x,y] creates x and annihilates y, matching actual owner.
                    yield ((species,site,i),(species,target,j),coefficient[i,j],(site,axis))
                    yield ((species,target,j),(species,site,i),np.conj(coefficient[i,j]),(site,axis))

    def field_potential_at(self, phi, outgoing):
        """Full quartic/alignment local potential; colour magnetic is per face."""
        gram=np.einsum('mij,nji->mn',phi,phi).real/4
        quartic=self.heavy_scale*(2/self.band.a)*np.sum(gram*gram)/2
        alignment=0.
        for neighbour,link in outgoing:
            difference=phi-np.einsum('ij,mjk,kl->mil',link,neighbour,link.conj().T)
            alignment += .2*self.heavy_scale**1.25*np.sum(abs(difference)**2)/8
        return quartic+alignment

    def full_form_contract(self):
        """Operator sum, on the original full physical Friedrichs/Gauss domain."""
        return dict(Phi_kinetic='Lambda^-1/2 sum p_Phi^2/2 on R^(60V)',
                    colour_electric='.3 Lambda^-1/2 sum E4^2/2 on SU4^(3V)',
                    quartic='Lambda (2/a) sum ||Tr(Phi_mu Phi_nu)/4||_F^2/2',
                    alignment='.2 Lambda^5/4 sum ||Phi_x-U Phi_y Udagger||_F^2/8',
                    colour_magnetic='.7 Lambda^5/4 sum (4-ReTr U_face)',
                    EM='original positive quadratic real curl, Coulomb and harmonic electric; no compact replacement',
                    matter='local_matter_coefficients lifted by -i path_A n_source(1-n_target)',
                    loops='Delta sum local faces/bridge rectangles (1-S)/2',
                    dynamics='sum nonzero spatial-edge Z_e, bounded total beta; field independent',
                    reference='four local tree modes, actual wedge3/bar4 or wedge1/4 and singlet source',
                    Gauss='field adjoint+colour left/right electric+onsite tree-even H/L/ref charges; EM divergence minus light/background/ref charge',
                    domain='full Phi and SU4 and real EM; kappa0 or explicit paired harmonic history')


class NativeOriginalModel(BosonicEdgeModel):
    """Eight modes, local bulk/ref trees, two bridges and one auxiliary loop."""
    def __init__(self):
        super().__init__(8, tuple((i, i+1) for i in range(7)) + ((0, 7),))
        self.tree_edges = tuple((i, i+1) for i in range(7))
        self.bulk_modes = tuple(range(4))
        self.reference_modes = tuple(range(4, 8))
        self.defect_edge = (0, 7)
        word = ['I'] * len(self.edges)
        word[self.edge_index[self.defect_edge]] = 'Z'
        self.defect = tensor_pauli(word)

    def tree_bilinear(self, target, source):
        if target == source:
            return self.n(source)
        step = 1 if target > source else -1
        path = tuple(range(source, target+step, step))
        return self.bilinear(target, source, path=path)

    def encode_even(self, matrix, modes=None):
        modes = tuple(range(self.vertices)) if modes is None else tuple(modes)
        matrix = np.asarray(matrix, dtype=complex)
        if matrix.shape != (len(modes), len(modes)):
            raise ValueError('Coefficient matrix/mode mismatch')
        result = np.zeros_like(self.identity)
        for i, target in enumerate(modes):
            for j, source in enumerate(modes):
                if matrix[i, j] != 0:
                    result += matrix[i, j] * self.tree_bilinear(target, source)
        return result

    def code_isometry(self, number=None):
        """Comparator-only global phase alignment, valid for each even filling."""
        masks, basis = self.occupation_basis(number)
        phases = np.zeros(len(masks), dtype=complex)
        primitives = [basis.conj().T @ self.hop(i, j) @ basis
                      for i, j in self.tree_edges]
        references = [reference_hop(self.vertices, masks, i, j)
                      for i, j in self.tree_edges]
        for root in range(len(masks)):
            if abs(phases[root]) > 0:
                continue
            phases[root] = 1
            queue = deque([root])
            while queue:
                column = queue.popleft()
                for q, f in zip(primitives, references):
                    for a, b in ((q, f), (q.conj().T, f.conj().T)):
                        for row in np.flatnonzero(abs(b[:, column]) > .5):
                            if abs(phases[row]) == 0:
                                phases[row] = a[row, column]*phases[column]/b[row, column]
                                queue.append(int(row))
        return masks, basis * phases[None, :]


class ReferenceEmbedding(TypedDict):
    number: np.ndarray
    neutral: np.ndarray
    charged: np.ndarray
    charge: np.ndarray
    unwanted: np.ndarray
    projectors: list[np.ndarray]
    representation: str


def reference_sector_embedding(model, modes=None, branch=1) -> ReferenceEmbedding:
    """Actual local wedge3 bar4 / wedge1 4, not an odd parity spectator.

    Plus branch neutral n_ref=4, charge q_ref=n_ref-4 on admitted 3/4.
    Minus branch neutral n_ref=0, charge q_ref=n_ref on admitted 0/1.
    Other occupations have finite gauge-invariant penalties, not hard deletion.
    """
    modes = model.reference_modes if modes is None else tuple(modes)
    if len(modes) != 4 or branch not in (-1, 1):
        raise ValueError('Four reference colours and branch +/-1 required')
    ident = model.identity
    nref = np.zeros_like(ident)
    for mode in modes:
        nref += model.n(mode)
    projectors = []
    for count in range(5):
        p = np.zeros_like(ident)
        for occupied in itertools.combinations(modes, count):
            term = ident.copy()
            for i in modes:
                term = term @ (model.n(i) if i in occupied else ident-model.n(i))
            p += term
        projectors.append(p)
    neutral = 4 if branch == 1 else 0
    charged = 3 if branch == 1 else 1
    return ReferenceEmbedding(number=nref, neutral=projectors[neutral], charged=projectors[charged],
                charge=nref-neutral*ident, unwanted=ident-projectors[neutral]-projectors[charged],
                projectors=projectors, representation='bar4' if branch == 1 else '4')


def reference_source(model, coefficients, branch=1) -> tuple[np.ndarray,list[np.ndarray]]:
    """Four internal reference components retained with actual source norm."""
    embedding = reference_sector_embedding(model, branch=branch)
    coefficients = np.asarray(coefficients, dtype=complex)
    if coefficients.shape != (4,):
        raise ValueError('Actual source needs all four colour coefficients')
    pieces = []
    for bulk, ref in zip(model.bulk_modes, model.reference_modes):
        operator = model.tree_bilinear(bulk, ref) if branch == 1 else model.tree_bilinear(ref, bulk)
        pieces.append(embedding['charged'] @ operator @ embedding['neutral'])
    result = np.zeros_like(model.identity)
    for coefficient,piece in zip(coefficients,pieces):
        result += coefficient*piece
    return result,pieces


def finite_gap_budget(vacuum_lower, code_vacuum_upper, window, defect_norm, penalty):
    """Complete Q lower bound, relative to the ACTUAL code vacuum.

    H0 >= vacuum_lower uniformly in fields/loop sectors. On Q the local cycle
    penalty is >=penalty. Bounded D has norm beta; no full-H norm is assumed.
    """
    beta = defect_norm
    gap = penalty + vacuum_lower - code_vacuum_upper - window - beta
    if gap <= 0:
        raise ValueError('Finite penalty does not dominate vacuum advantage/window')
    return dict(gap=gap, beta=beta, leakage_amplitude=beta/gap,
                schur_residual=beta*beta/gap,
                vacuum_advantage=code_vacuum_upper-vacuum_lower)


def weighted_transfer_constants(gap, beta, graph_norm, time):
    """Resolvent/source graph transfer constants, not generic relative errors.

    B must obey ||B P|| <= graph_norm and
    ||B Q(H_Q-E)^-1|| <= graph_norm/gap. This is the bounded-observable
    contract. A quartic-weighted Phi graph uses the distinct 3/4-power API.
    """
    if min(gap, graph_norm) <= 0 or beta < 0:
        raise ValueError('Positive gap and certified observable graph required')
    return dict(leakage=beta/gap, energy_residual=beta*beta/gap,
                observable_image_bound=2*graph_norm*beta/gap,
                time_amplitude_bound=abs(time)*beta*beta/gap+4*beta/gap)


def quartic_weighted_transfer_budget(graph_log2_upper=3972, gap_log2=11892,
                                    beta_log2=-4, time=6) -> dict[str,Any]:
    """Actual supplied weighted-Phi/current/Lie allocation, no full-H norm.

    For z=Q(H_Q-E)^-1 D P psi, ||z||<=beta/G and
    <z,(H_Q-E)z><=beta^2/G. The supplied quartic form gives
    ||rho z|| <= C beta/G^(3/4), with its vacuum/coercivity remainder paid
    separately. graph_log2_upper includes C AND actual source/current/Lie jets.
    It is not a bounded full-H coefficient and not a bare internal j matrix.
    """
    error_log = Fraction(3+graph_log2_upper+beta_log2)-Fraction(3*gap_log2,4)
    if error_log.denominator != 1:
        error_log = Fraction(-((-error_log.numerator)//error_log.denominator))
    exponent=int(error_log)
    error=Fraction(2**exponent) if exponent>=0 else Fraction(1,2**(-exponent))
    beta = Fraction(2**beta_log2) if beta_log2>=0 else Fraction(1,2**(-beta_log2))
    graph = Fraction(2**graph_log2_upper) if graph_log2_upper>=0 else Fraction(1,2**(-graph_log2_upper))
    ordinary = graph*(4*beta+abs(time)*beta*beta)/2**gap_log2
    ue=Fraction.from_float(1.5429442570866347e-148)/2**4096
    target=ue/(NSTAR**3*2**40)
    # Coexact A has a HALF-power energy graph, unlike quartic Phi. Actual
    # G>=U_trial>=Lambda^(5/4), with selected log2 Lambda>=180375, pays it.
    c=Fraction(1130202988249777,9007199254740992)
    coexact_A_square=ue*NSTAR**2/(8*c*c)  # curl lambda_min>=16/N^2
    if coexact_A_square>=1:
        raise AssertionError('Original coexact photon coercivity constant is not below one')
    em_gap_floor=max(gap_log2,5*180375//4)
    em_exponent=3+graph_log2_upper+beta_log2-em_gap_floor//2
    em_error=Fraction(2**em_exponent) if em_exponent>=0 else Fraction(1,2**(-em_exponent))
    if error+em_error+ordinary>=target:
        raise ValueError('Supplied quartic graph/gap does not meet actual UE/V target')
    ratio=error/target
    total_ratio=(error+em_error+ordinary)/target
    ratio_log=ratio.numerator.bit_length()-ratio.denominator.bit_length()+1
    total_ratio_log=total_ratio.numerator.bit_length()-total_ratio.denominator.bit_length()+1
    return dict(graph_log2_upper=graph_log2_upper,weighted_gap_power='3/4',
                gap_log2=gap_log2,beta_log2=beta_log2,
                local_defect_coefficient=str((Fraction(2**beta_log2) if beta_log2>=0
                                              else Fraction(1,2**(-beta_log2)))/(6*NSTAR**3)),
                weighted_error_log2_upper=exponent,
                coexact_EM_weighted_gap_power='1/2',
                coexact_EM_gap_log2_lower=em_gap_floor,
                coexact_EM_error_log2_upper=em_exponent,
                coexact_EM_graph_constant_below_one=True,
                actual_log2_Lambda_lower=180375,
                harmonic_domain='kappa0/explicit paired history; no harmonic oscillator coercivity claimed',
                weighted_error_over_actual_target=float(ratio) if ratio_log>=-1022 else None,
                weighted_ratio_log2_upper=ratio_log,
                total_error_over_actual_target=float(total_ratio) if total_ratio_log>=-1022 else None,
                total_ratio_log2_upper=total_ratio_log,strictly_positive_error=True,
                bare_schur_time_log2_upper=int(np.ceil(np.log2(abs(time)+1)))+2*beta_log2-gap_log2,
                normal_receipt_sha256='dc7d6277001394ea4c7b96aa6d8874e889e50b830fea47d6a4ca22db8ebbafcf',
                scope='actual supplied source/current/Lie and retained EM graphs; separate full normal/EM admission remains owned upstream',
                gap_is_lower_bound_not_actual_selected_value=True,
                actual_high_gap_contract=large_family_gap_contract(gap_log2),
                penalty='B_N(Lambda)+U_trial(Lambda)+window+beta+actual_G')


def original_microscopic_loop_budget():
    """Actual source19 original-Omega graphs, not prepared-Gamma relabelling."""
    result=quartic_weighted_transfer_budget(65536,131072,-4,6)
    result.update(microscopic_graph_receipt_sha256=
                  '48b45ba330635169bde84020f16b52805faf995d2767cbc7c7121b83f4ae5fdc',
                  actual_input='B a_dagger[g] Omega_original, not U_g or Gamma response',
                  source_sites=19,tree_word=NSTAR//2+2,eta=18,alpha='2^371',
                  actual_source_q6_ceils=[4,427,850,1274,1698,2122,2547],
                  source19_angular1_log2=1494,source19_angular2_log2=2184,
                  source19_two_current_two_angular_log2=3187,
                  own_scalar_orbital_coefficient_upper=192,
                  U_g_uniform_unit_L2_complex_coexact=True,
                  RAW_profile_orbital_graph_not_copied=True,
                  actual_after_B_photon_half_graph_log2_upper=600,
                  charged_source_norm_lower='1/sqrt8',
                  normal_core_tube_log2_prefactor=27868,
                  normal_core_tube='2J Lambda^-1/5; selected actual finite Lambda retained',
                  scope='original microscopic source19 auxiliary-loop transfer; physical source/current/Noether and full receiver composed by PhysicalTransfer')
    if result['total_ratio_log2_upper']>=-20000:
        raise AssertionError('Original microscopic loop envelope lost actual UE/V margin')
    return result


def quartic_vacuum_coefficients_from_l1(volume,kappa,heavy_l1,light_l1):
    """Actual heavy/light coefficient l1 caps; supports explicitly changed bands."""
    c0=Fraction(kappa)/128
    heavy,light=Fraction(heavy_l1),Fraction(light_l1)
    return (volume*(heavy*heavy/(2*c0)+c0/8),
            volume*heavy*light/c0,volume*light*light/(2*c0))


def quartic_vacuum_coefficients(volume,band):
    """Exact B_N(Lambda)=a_N Lambda+b_N+c_N/Lambda, across ALL sectors."""
    onsite,paths=scalar_paths(band)
    coefficient=4*abs(Fraction(float(onsite)))
    for _,scalar in paths:
        coefficient+=8*sum((abs(Fraction(float(z.real)))+abs(Fraction(float(z.imag)))
                            for z in scalar),Fraction())
    return quartic_vacuum_coefficients_from_l1(volume,Fraction(2)/Fraction(float(band.a)),
                                              coefficient,coefficient)


def gauss_trial_vacuum_upper_contract():
    """Admissible explicit ORIGINAL-law upper, not the frozen matrix vacuum."""
    volume=NSTAR**3
    return dict(U_trial='V*(300*Lambda+(132/5)*Lambda^(5/4)+15*Lambda^(-1/2)+4*c)',
                V=str(volume),c='1130202988249777/9007199254740992',
                real_field_coordinates=60,
                field_wave='normalized exp(-sum Cartesian C_mu_a^2/2)',
                field_variance='1/2',mean_Gram_square=300,
                colour_links='normalized constant Haar; electric Casimir zero',
                alignment_per_site='90*J=18*Lambda^(5/4)',
                colour_magnetic_per_site='12*K4=(42/5)*Lambda^(5/4)',
                photon='original coexact Fock vacuum, kappa=0; E_ph<4*c*V',
                matter='fixed onsite (NH,NL)=(2,2) SU4 singlet; reference neutral singlet',
                charged_trial='at reference site: plus (H2,L3,ref3) or minus (H2,L1,ref1) SU4 singlet',
                Gauss='all local colour generators annihilate; light+reference EM rho=0',
                matter_energy_expectation=0,reference_penalty_expectation=0,
                local_loop_penalty_expectation=0,defect_expectation=0,
                scope='vacuum upper only; charge co-located for trial, never substituted for physical B input')


def large_family_gap_contract(minimum_gap_log2=131072):
    """Finite full-Phi dominance, NOT the source envelope used as a full-H norm."""
    band=RotationBand(a=1.,m=.2,r=.12,v=.3)
    a,b,c=quartic_vacuum_coefficients(NSTAR**3,band)
    return dict(vacuum_lower='-B_N(Lambda)',
                B_N='a_N*Lambda+b_N+c_N/Lambda',
                a_N=str(a),b_N=str(b),c_N=str(c),
                actual_Lambda='unchanged selected finite normal/phase Lambda tower',
                actual_code_vacuum_upper=gauss_trial_vacuum_upper_contract(),
                G=f'max(2^{minimum_gap_log2}, B_N(Lambda)+U_trial(Lambda)+window+beta+1)',
                Delta='B_N(Lambda)+U_trial(Lambda)+window+beta+G',
                result='Q(H-E)Q >= G; G dominates explicit full-Phi advantage and trial cost',
                no_advantage_assumption=False,full_Phi_operator_norm_claimed=False)


def quartic_vacuum_lower(volume, heavy_scale, band):
    """Exact rational coarse global form lower bound, O(volume*Lambda).

    r_x^2=sum_mu ||Phi_mu(x)||_F^2; kappa||G||^2/2>=kappa*r_x^4/128.
    The original slow law multiplies quartic and heavy matter by Lambda.
    ||even lift|| <= entrywise coefficient sum, in EVERY auxiliary sector.
    """
    lam=Fraction(heavy_scale)
    a,b,c=quartic_vacuum_coefficients(volume,band)
    return -(a*lam+b+c/lam)


def symbolic_graph_contract(size=NSTAR):
    """No lattice/mode/edge list allocation. Onsite trees and cubic anchor graph."""
    volume = size**3
    # Each site H/L four-mode trees (6), H-L bridge (1); H and L cubic (6).
    vertices = 8*volume+4
    edges = 13*volume+4  # reference tree3 + one light-anchor attachment
    return dict(size=size, volume=volume, modes=vertices, edge_qubits=edges,
                independent_cycles=edges-vertices+1, maximum_degree=11,
                onsite_paths_max_edges=3, spatial_bilinear_path_max_edges=7,
                onsite_tree_edges_per_site=6, spatial_anchor_edges_per_site=6,
                cycle_law='all elementary H/L anchor plaquettes AND local H-L bridge rectangles; redundant local constraints allowed',
                noncontractible_holonomies=3,flat_auxiliary_spin_structures=8,
                holonomy_law='H-L rectangles glue T3 x interval; 8 auxiliary spin structures, independent of 64 original SU4 centre wells',
                local_penalty_gap='each nontrivial contractible syndrome costs at least Delta',
                topological_scope='local defect can mix holonomy histories through excited flux; Schur bounds apply to full flat direct sum, prepared + history transfer is finite-time',
                parity='product B=+1; full heavy/light/reference total parity, not a spectator',
                gauge='onsite tree even algebra gives exact su4 in all spatial-loop sectors')


def topology_probe(size=3):
    """GF2 local-cycle rank, no Hilbert matrix; bounded native positive control."""
    if not 3<=size<=5:
        raise ValueError('Native cycle proof uses only N=3..5, never Nstar')
    sites=tuple(itertools.product(range(size),repeat=3))
    def neighbour(site,axis):
        result=list(site);result[axis]=(result[axis]+1)%size
        return tuple(result)
    edges=[]
    for species in range(2):
        for site in sites:
            for axis in range(3):
                edges.append(tuple(sorted(((species,site),(species,neighbour(site,axis))))))
    edges.extend(((0,site),(1,site)) for site in sites)
    lookup={edge:j for j,edge in enumerate(edges)}
    def edge_bit(left,right):
        return 1<<lookup[tuple(sorted((left,right)))]
    faces=[]
    for species in range(2):
        for site in sites:
            for a,b in itertools.combinations(range(3),2):
                sa,sb=neighbour(site,a),neighbour(site,b)
                sab=neighbour(sa,b)
                cycle=((species,site),(species,sa),(species,sab),(species,sb))
                mask=0
                for left,right in zip(cycle,(*cycle[1:],cycle[0])):
                    mask ^= edge_bit(left,right)
                faces.append(mask)
    rectangles=[]
    for site in sites:
        for axis in range(3):
            target=neighbour(site,axis)
            rectangles.append(edge_bit((0,site),(0,target))^edge_bit((1,site),(1,target))
                              ^edge_bit((0,site),(1,site))^edge_bit((0,target),(1,target)))
    def rank(rows):
        pivots={}
        for value in rows:
            while value:
                pivot=value.bit_length()-1
                if pivot not in pivots:
                    pivots[pivot]=value;break
                value ^= pivots[pivot]
        return len(pivots)
    cycles=len(edges)-2*len(sites)+1
    local_rank=rank((*faces,*rectangles))
    wrong_debt=cycles-rank(faces)
    if cycles-local_rank!=3 or wrong_debt<=3:
        raise AssertionError('Local rectangle/torus topology contract failed')
    return dict(size=size,anchor_vertices=2*len(sites),anchor_edges=len(edges),
                cycle_rank=cycles,local_constraint_rank=local_rank,
                retained_holonomies=cycles-local_rank,
                omitted_bridge_rectangles_holonomies=wrong_debt)


def physical_budget_family(log2_lambda=180375, time=6, source_log2=2856, graph_log2_upper=None):
    """Finite/nonzero dyadic successor below the actual tiny UE/V budget.

    Gap chosen from vacuum advantage plus a dyadic excess. Norm beta <=1
    remains nonzero; penalty is finite although astronomically large. Any
    supplied source/current/Noether graph logs enter before choosing excess.
    """
    v = NSTAR**3
    ue = Fraction.from_float(1.5429442570866347e-148)/2**4096
    target = ue/(v*2**40)
    target_log_floor = target.numerator.bit_length()-target.denominator.bit_length()-1
    graph_log = source_log2 if graph_log2_upper is None else max(source_log2,graph_log2_upper)
    time_log = max(1,int(np.ceil(np.log2(2+abs(time)))))
    gap_log = graph_log-target_log_floor+16+time_log
    # beta=1 total; distribute nonzero 1/M over all local spatial defects.
    return dict(log2_lambda=log2_lambda, target='UE/(V*2^40)',
                target_log2_floor=target_log_floor, supplied_graph_log2_upper=graph_log,
                required_gap_log2=gap_log, total_defect_norm_upper=1,
                local_defect_coefficient=f'1/{6*v}',
                gap_is_lower_bound_not_actual_selected_value=True,
                actual_high_gap_contract=large_family_gap_contract(gap_log),
                penalty='B_N(Lambda)+U_trial(Lambda)+window+beta+actual_G',
                observable_error_log2_upper=1+graph_log-gap_log,
                time_error_log2_upper=time_log-gap_log,
                status=('actual bounded source-value contribution only; Noether/current graph input still required'
                        if graph_log2_upper is None else
                        'conditional on caller-certified weighted source/current/Noether graph; no full-H norm'))


def finite_exchange_probe(phi,band,x,p,photon) -> dict[str,Any]:
    """Spatial two-charge exchange with finite nonzero flux dynamics.

    Single-colour spatial Galerkin panel; full colour/ref source is separately
    exercised in native_probe. Every pulse coefficient is an original Phi
    endpoint entry; no fermion/Dirac exchange endpoint is supplied.
    """
    model=BosonicEdgeModel(4,((0,1),(1,2),(2,3),(0,3)))
    word=['I']*len(model.edges);word[model.edge_index[(0,3)]]='Z'
    defect=tensor_pauli(word)
    theta,rotations=np.linalg.eigh(x)
    projectors=[np.outer(rotations[:,j],rotations[:,j].conj()) for j in range(3)]
    basis=matrix_unit_basis()
    link=expm(.19j*basis[3])
    cloud=phase_galerkin(3,.27,1.)
    potential=np.zeros((3,3),complex)
    law=OriginalBosonicLaw(3,band,1.,31.,.13)
    coefficients=[]
    for axis in (0,1):
        coefficient=np.zeros((3,3),complex)
        for value,projector in zip(theta,projectors):
            shifted=phi+.07*value*np.roll(phi,1,axis=0)
            endpoint=original_endpoint(shifted,shifted*.91,link,axis,band)
            coefficient += endpoint[0,0]*projector
        coefficients.append(coefficient)
    for value,projector in zip(theta,projectors):
        shifted=phi+.07*value*np.roll(phi,1,axis=0)
        potential += law.field_potential_at(shifted,[(shifted*.91,link)])*projector
    field=.65*(p@p+np.diag([0.,0.,1.5]))+potential
    background=np.kron(model.identity,np.kron(field,np.eye(3))+np.kron(np.eye(3),photon))
    background+=np.kron(model.loop_penalty(31.)+.13*defect,np.eye(9))
    pair=-1j*model.path_A((0,1,2))@(model.identity-model.n(0))@(model.identity-model.n(2))
    source=np.kron(pair,np.eye(9))
    results={}
    for name,signs in (('code',None),('false_sector',(-1,))):
        neutral=np.kron(model.state((),signs),np.eye(9)[0])
        charged=source@neutral
        total_time=0.
        for (start,end),axis,orientation in zip(((0,1),(2,3),(1,2),(3,0)),(0,0,1,1),(1,-1,1,-1)):
            coefficient=coefficients[axis].conj().T if orientation==1 else coefficients[axis]
            phase=cloud.conj().T if orientation==1 else cloud
            forward=np.kron(model.hop(start,end),np.kron(coefficient,phase))
            base=original_endpoint(phi,phi*.91,link,axis,band)[0,0]
            duration=np.pi/(2*abs(base))
            charged=expm(-1j*duration*(background-forward-forward.conj().T))@charged
            total_time+=duration
        neutral=expm(-1j*total_time*background)@neutral
        target=source@neutral
        amplitude=np.vdot(target,charged)/np.vdot(target,target)
        results[name]=dict(real=float(amplitude.real),imag=float(amplitude.imag),
                           magnitude=float(abs(amplitude)),time=float(total_time))
    if results['code']['real']>=-.8 or results['false_sector']['real']<=.8:
        raise AssertionError(('finite coupled exchange must discriminate',results))
    return dict(beta=.13,penalty=31.,photon_phase='.27',results=results,
                source_history='same neutral finite-field/gauge/defect history; primitive pair source',
                scope='single-colour spatial exchange panel; fullRef4 is native_probe')


def native_probe():
    model = NativeOriginalModel()
    ident = model.identity
    band = RotationBand(a=1., m=.2, r=.12, v=.3)
    basis = matrix_unit_basis()
    # Arbitrary full Cartesian traceless Phi; no imported Clifford endpoints.
    coordinates = np.array([[.17*np.sin((m+1)*(j+1)) for j in range(15)] for m in range(4)])
    phi = np.einsum('ma,aij->mij', coordinates, basis)
    onsite, _ = scalar_paths(band)
    colour = expm(.19j*basis[3])
    endpoint = original_endpoint(phi, phi*.91, colour, 0, band)
    # Native spatial-cell Galerkin: the physical endpoint coefficient retains
    # its alternate loop path. The reference has no artificial field endpoint.
    spatial = model.stabilizers[0] @ model.encode_even(endpoint+endpoint.conj().T, model.bulk_modes)
    matter = model.encode_even(onsite*phi[0], model.bulk_modes) + spatial
    # One genuine compact SU4 colour-coordinate Galerkin jet and real noncompact
    # EM oscillator, with field endpoint coefficients multiplying encoded hops.
    x = np.array([[0.,1.,0.],[1.,0.,np.sqrt(2.)],[0.,np.sqrt(2.),0.]])/np.sqrt(2)
    p = np.array([[0.,-1j,0.],[1j,0.,-1j*np.sqrt(2.)],[0.,1j*np.sqrt(2.),0.]])/np.sqrt(2)
    photon = np.diag(np.arange(3)+.5)
    cloud = phase_galerkin(3,.27,1.)
    momentum_squared = p@p+np.diag([0.,0.,1.5])
    source, pieces = reference_source(model, [1,.3j,-.2+.4j,.6])
    ref = reference_sector_embedding(model)
    # Source is gauge invariant only after its field/colour dressing. Native
    # scalar contraction sum colour components is separately checked below.
    scalar_source, _ = reference_source(model, np.ones(4))
    scalar_matter = matter + 17*ref['unwanted']
    neutral = model.state((4,5,6,7))
    sourced = source@neutral
    source_norm = np.linalg.norm(sourced)
    sourced /= source_norm
    total_number = sum((model.n(i) for i in range(8)), np.zeros_like(ident))
    colour_comm = []
    scalar_comm = []
    lie = []
    bulk_generators=[model.encode_even(t,model.bulk_modes) for t in basis]
    reference_generators=[model.encode_even(t,model.reference_modes) for t in basis]
    generators=[left+right for left,right in zip(bulk_generators,reference_generators)]
    for g in generators:
        colour_comm.append(np.linalg.norm(g@model.defect-model.defect@g))
        scalar_comm.append(np.linalg.norm(g@scalar_source-scalar_source@g))
    basis_norm=np.einsum('aij,aji->a',basis,basis)
    for a,b in itertools.combinations(range(15),2):
        bracket=basis[a]@basis[b]-basis[b]@basis[a]
        structure=np.einsum('aij,ji->a',basis,bracket)/basis_norm
        for representation in (bulk_generators,reference_generators):
            expected=np.zeros_like(ident)
            for coefficient,generator in zip(structure,representation):
                expected += coefficient*generator
            ga,gb=representation[a],representation[b]
            lie.append(np.linalg.norm(ga@gb-gb@ga-expected))
    projector = model.sector_projector()
    coupling = .13
    penalty = 31.
    # Independent quantum field/colour coordinate and real photon coordinate.
    # The SU4 link and ALL four Phi matrices vary in the field-coordinate jet.
    theta, rotations = np.linalg.eigh(x)
    field_projectors = [np.outer(rotations[:,j],rotations[:,j].conj()) for j in range(3)]
    forward = csr_matrix((model.dimension*9,model.dimension*9),dtype=complex)
    local_field = np.zeros((3,3),dtype=complex)
    native_field_law = OriginalBosonicLaw(3,band,1.,penalty,coupling)
    for value, field_projector in zip(theta,field_projectors):
        shifted_phi = phi + .07*value*np.roll(phi,1,axis=0)
        shifted_colour = expm(1j*(.19+.09*value)*basis[3])
        shifted_endpoint = original_endpoint(shifted_phi,shifted_phi*.91,shifted_colour,0,band)
        encoded = model.stabilizers[0]@model.encode_even(shifted_endpoint,model.bulk_modes)
        forward += sparse_kron(sparse_kron(csr_matrix(encoded),csr_matrix(field_projector)),csr_matrix(cloud))
        local_field += native_field_law.field_potential_at(shifted_phi,[(shifted_phi*.91,shifted_colour)])*field_projector
    field = .65*momentum_squared+local_field
    link = forward+forward.conj().T
    h0 = sparse_kron(csr_matrix(scalar_matter-spatial),sparse_eye(9))
    h0 += sparse_kron(csr_matrix(ident),csr_matrix(np.kron(field,np.eye(3))+np.kron(np.eye(3),photon)))+link
    h0 += sparse_kron(csr_matrix(model.encode_even(onsite*phi[2],model.bulk_modes)),csr_matrix(np.kron(.07*x,np.eye(3))))
    initial = np.kron(np.kron(sourced,np.array([1.,0.,0.])),cloud@np.array([1.,0.,0.]))
    h = h0+sparse_kron(csr_matrix(model.loop_penalty(penalty)+coupling*model.defect),sparse_eye(9))
    final = expm_multiply(-.7j*h,initial)
    outside = sparse_kron(csr_matrix(ident-projector),sparse_eye(9))
    leakage = float(np.vdot(final,outside@final).real)
    current=-1j*(forward-forward.conj().T)
    contact=-(forward+forward.conj().T)
    eps=1e-4
    phased=lambda a:np.exp(1j*a)*forward+np.exp(-1j*a)*forward.conj().T
    derivative_error=sparse_norm((phased(eps)-phased(-eps))/(2*eps)+current)
    contact_error=sparse_norm((phased(eps)+phased(-eps)-2*link)/eps**2-contact)
    masks,W=model.code_isometry(4)
    WP = np.kron(W,np.eye(9))
    WQ = np.kron(model.defect@W,np.eye(9))
    pc = WP.conj().T@(h0@WP)
    qc = WQ.conj().T@(h0@WQ)
    spectrum_p = np.linalg.eigvalsh(pc)
    spectrum_q = np.linalg.eigvalsh(qc)
    native_gap = finite_gap_budget(float(min(spectrum_p[0],spectrum_q[0])),
                                  float(spectrum_p[0]),
                                  float(spectrum_p[-1]-spectrum_p[0]),
                                  coupling,penalty)
    native_transfer = weighted_transfer_constants(native_gap['gap'],coupling,1.,.7)
    exact_final = expm_multiply(-.7j*h0,initial)
    state_difference = float(np.linalg.norm(final-exact_final))
    low_spectrum = np.linalg.eigvalsh(np.block([[pc,coupling*np.eye(len(pc))],
                                              [coupling*np.eye(len(pc)),qc+penalty*np.eye(len(pc))]]))[:len(pc)]
    spectral_difference = float(np.max(abs(low_spectrum-spectrum_p)))
    law = OriginalBosonicLaw(3,band,2.,31.,.13)
    local_coefficients = list(law.local_matter_coefficients((0,0,0),lambda s:phi,
                                                          lambda s,a:colour,lambda s,a:.17))
    coefficient_boundary_error = max(abs(coef-np.exp(.17j)*original_endpoint(phi,phi,colour,path[1],band)[target[2],origin[2]])
                                    for target,origin,coef,path in local_coefficients
                                    if target[0]=='L' and path is not None and target[1]==(0,0,0))
    native_lower = quartic_vacuum_lower(27,2,band)
    if native_lower >= 0:
        raise AssertionError('Quartic vacuum lower bound must retain negative matter advantage')
    transfer=[]
    for target,origin in ((0,7),(2,5),(1,3),(4,6)):
        transfer.append(np.linalg.norm(model.tree_bilinear(target,origin)@W-W@reference_hop(8,masks,origin,target)))
    # Actual wedge3 -T^T comparison uses one simultaneous signed hole basis,
    # not independent eigenvalue matches for each generator.
    holes=[]
    mask_lookup = {mask:i for i,mask in enumerate(masks)}
    for i in range(4):
        occupied=tuple(j for j in range(4,8) if j!=i+4)+(0,)
        mask=sum(1<<j for j in occupied)
        holes.append((-1)**i*W[:,mask_lookup[mask]])
    H=np.column_stack(holes)
    bar_errors=[]
    for t in basis:
        actual=H.conj().T@model.encode_even(t,model.reference_modes)@H
        bar_errors.append(np.linalg.norm(actual+t.T))
    missing_source, _ = reference_source(model,[1,0,0,0])
    missing_ref_error = np.linalg.norm((source-missing_source)@neutral)
    frozen_current_error = sparse_norm(current)
    colour_removed_endpoint = original_endpoint(phi,phi*.91,np.eye(4),0,band)
    boundary_error = np.linalg.norm(endpoint-colour_removed_endpoint)
    gx,gy=expm(.23j*basis[2]),expm(-.31j*basis[5])
    transformed=original_endpoint(gx@phi@gx.conj().T,gy@(phi*.91)@gy.conj().T,
                                  gx@colour@gy.conj().T,0,band)
    gauge_covariance_error=float(np.linalg.norm(transformed-gx@endpoint@gy.conj().T))
    phase_adversary=float(np.linalg.norm(cloud-expm(.27j*x)))
    ph = ident.copy()
    for i in range(0,8,2):
        ph = ph @ model.A(i,i+1)
    # Bipartite density phase is essential: density complement alone is NOT
    # the original CAR-PH comparison on all off-diagonal even bilinears.
    ph = model.region_parity((1,3,5,7)) @ ph
    ph_density_error = max(np.linalg.norm(ph@n.conj()@ph.conj().T-(ident-n)) for n in
                           (model.n(i) for i in range(8)))
    ph_defect_error = np.linalg.norm(ph@model.defect.conj()@ph.conj().T-model.defect)
    ph_loop_error=max(np.linalg.norm(ph@s.conj()@ph.conj().T-s) for s in model.stabilizers)
    minus_ref=reference_sector_embedding(model,branch=-1)
    minus_source,_=reference_source(model,np.conj([1,.3j,-.2+.4j,.6]),branch=-1)
    ph_source_error=np.linalg.norm(ph@source.conj()@ph.conj().T+minus_source)
    field_part=sparse_kron(csr_matrix(ident),csr_matrix(np.kron(field,np.eye(3))+np.kron(np.eye(3),photon)))
    plus_reference_part=sparse_kron(csr_matrix(17*ref['unwanted']),sparse_eye(9))
    minus_reference_part=sparse_kron(csr_matrix(17*minus_ref['unwanted']),sparse_eye(9))
    matter_part=h0-field_part-plus_reference_part
    paired_h=-matter_part+field_part+minus_reference_part
    paired_h+=sparse_kron(csr_matrix(model.loop_penalty(penalty)+coupling*model.defect),sparse_eye(9))
    ph_full=sparse_kron(csr_matrix(ph),sparse_eye(9))
    ph_full_error=sparse_norm(ph_full@h.conj()@ph_full.conj().T-paired_h)
    # A folded single spatial cell can lose the GLOBAL zero-trace cancellation.
    # It must not be advertised as a full PH-ground witness. Test the actual
    # two-distinct-site coefficient instead, with primitive Pauli operators.
    actual_bond=model.encode_even(onsite*phi[0],model.bulk_modes)
    actual_bond+=model.encode_even(onsite*.91*phi[0],model.reference_modes)
    for i,j in itertools.product(range(4),repeat=2):
        term=endpoint[i,j]*model.tree_bilinear(model.bulk_modes[i],model.reference_modes[j])
        actual_bond+=term+term.conj().T
    ph_original_bond_error=np.linalg.norm(ph@actual_bond.conj()@ph.conj().T+actual_bond)
    # Reference modes above label the second physical site ONLY for this
    # coefficient/algebra comparison. The evolved physical reference is static.
    # Full-sector Gauss Ward: field/link covariance + exact onsite SU4 action.
    # Include alternate spatial loop, finite nonzero D and same EM derivatives.
    rotation=expm(1j*(.23*bulk_generators[2]-.31*reference_generators[5]))
    forward_before=np.zeros_like(ident);forward_after=np.zeros_like(ident)
    for i,j in itertools.product(range(4),repeat=2):
        hop=model.tree_bilinear(model.bulk_modes[i],model.reference_modes[j])
        forward_before += endpoint[i,j]*hop
        forward_after += transformed[i,j]*hop
    forward_before=model.stabilizers[0]@forward_before
    forward_after=model.stabilizers[0]@forward_after
    fw=sparse_kron(csr_matrix(forward_before),csr_matrix(cloud))
    fw_after=sparse_kron(csr_matrix(forward_after),csr_matrix(cloud))
    onsite_before=model.encode_even(onsite*phi[0],model.bulk_modes)+model.encode_even(onsite*.91*phi[0],model.reference_modes)
    onsite_after=model.encode_even(onsite*gx@phi[0]@gx.conj().T,model.bulk_modes)
    onsite_after+=model.encode_even(onsite*.91*gy@phi[0]@gy.conj().T,model.reference_modes)
    finite_part=model.loop_penalty(penalty)+coupling*model.defect
    ward_before=sparse_kron(csr_matrix(onsite_before+finite_part),sparse_eye(3))+fw+fw.conj().T
    ward_after=sparse_kron(csr_matrix(onsite_after+finite_part),sparse_eye(3))+fw_after+fw_after.conj().T
    rotation_full=sparse_kron(csr_matrix(rotation),sparse_eye(3))
    ward_h_error=sparse_norm(rotation_full@ward_before@rotation_full.conj().T-ward_after)
    ward_current_error=sparse_norm(rotation_full@(-1j*(fw-fw.conj().T))@rotation_full.conj().T
                                   -(-1j*(fw_after-fw_after.conj().T)))
    ward_contact_error=sparse_norm(rotation_full@(-fw-fw.conj().T)@rotation_full.conj().T
                                   -(-fw_after-fw_after.conj().T))
    square=BosonicEdgeModel(4,((0,1),(1,2),(2,3),(0,3)))
    moves=((0,1),(2,3),(1,2),(3,0))
    exchange={name:exchange_report(square,(0,2),moves,signs=signs,dressed=dressed)
              for name,signs,dressed in [('code',None,True),('false_sector',(-1,),True),('undressed',None,False)]}
    checks=dict(exact_colour_all_sectors=max(lie)<1e-10,
                gauge_preserving_defect=max(colour_comm)<1e-10,
                scalar_full_ref_source=max(scalar_comm)<1e-10,
                charge_preserving_defect=np.linalg.norm(total_number@model.defect-model.defect@total_number)<1e-10,
                actual_nonzero_flux=np.linalg.norm((ident-projector)@model.defect@projector)>1,
                finite_dynamics_leaks=leakage>1e-8,
                path_transfer=max(transfer)<1e-10,
                bar4_spectrum=max(bar_errors)<1e-10,
                current_contact=derivative_error<1e-6 and contact_error<1e-5,
                source_norm=abs(source_norm-np.linalg.norm([1,.3j,-.2+.4j,.6]))<1e-10,
                exchange_discriminates=abs(exchange['code']['hop_return_amplitude']['real']+1)<1e-10
                    and abs(exchange['false_sector']['hop_return_amplitude']['real']-1)<1e-10
                    and abs(exchange['undressed']['hop_return_amplitude']['real']-1)<1e-10)
    checks.update(reference_adversary=missing_ref_error>.1,
                  frozen_current_adversary=frozen_current_error>.1,
                  colour_boundary_adversary=boundary_error>1e-3,
                  PH_defect_and_density=max(ph_density_error,ph_defect_error)<1e-10)
    checks.update(native_gap=native_gap['gap']>0,
                  schur_spectrum=spectral_difference<=native_gap['schur_residual'],
                  full_state_transfer=state_difference<=native_transfer['time_amplitude_bound'],
                  local_original_coefficients=coefficient_boundary_error<1e-12)
    checks.update(original_SU4_endpoint_covariance=gauge_covariance_error<1e-12,
                  exact_photon_phase_not_compressed_exponential=phase_adversary>.01,
                  projected_photon_source_norm_retained=abs(np.linalg.norm(final)-np.linalg.norm(initial))<1e-10)
    checks.update(actual_Phi_PH_reference_source=max(ph_loop_error,ph_source_error,ph_original_bond_error)<1e-10,
                  folded_native_not_full_PH_ground=ph_full_error>1e-6)
    checks['full_auxiliary_nonzero_defect_Ward']=max(ward_h_error,ward_current_error,ward_contact_error)<1e-10
    finite_exchange = finite_exchange_probe(phi,band,x,p,photon)
    checks['finite_nonzero_field_gauge_exchange'] = (finite_exchange['results']['code']['real']<-.8
                                                   and finite_exchange['results']['false_sector']['real']>.8)
    checks = {key: bool(value) for key,value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(checks=checks,primitive_qubits=8,native_dimension=h.shape[0],
                finite_defect=coupling,finite_penalty=penalty,time=.7,
                source_norm=source_norm,loop_leakage=leakage,
                finite_coupled_exchange=finite_exchange,
                full_sector_SU4_generator_pairs_per_site=105,
                full_nonzero_defect_Ward=dict(H=float(ward_h_error),current=float(ward_current_error),
                                             contact=float(ward_contact_error),code_projection_used=False),
                original_SU4_endpoint_covariance=gauge_covariance_error,
                full_SU4_Lie_residual=max(lie),defect_colour_commutator=max(colour_comm),
                scalar_source_colour_commutator=max(scalar_comm),
                path_comparison=max(transfer),reference_bar4_spectral_residual=max(bar_errors),
                first_derivative_error=derivative_error,second_derivative_error=contact_error,
                missing_reference_error=float(missing_ref_error),
                frozen_current_error=float(frozen_current_error),
                removed_colour_boundary_error=float(boundary_error),
                PH_density_residual=float(ph_density_error),PH_defect_residual=float(ph_defect_error),
                native_gap=native_gap,native_transfer=native_transfer,
                spectral_deviation=spectral_difference,state_deviation=state_difference,
                local_coefficient_residual=float(coefficient_boundary_error),
                quartic_lower_native=str(native_lower),full_form=law.full_form_contract(),
                PH_source_residual=float(ph_source_error),PH_loop_residual=float(ph_loop_error),
                PH_original_distinct_site_bond_residual=float(ph_original_bond_error),
                folded_single_cell_PH_trace_remainder=float(ph_full_error),
                prepared_input_norm=float(np.linalg.norm(initial)),
                exact_photon_phase_adversary=float(np.linalg.norm(cloud-expm(.27j*x))),
                exchange=exchange,
                Galerkin_limit='independent three-level field/SU4-coordinate and real photon; no cutoff defines full law',
                input=dict(coordinates=coordinates.tolist(),band=dict(a=1,m=.2,r=.12,v=.3),
                           quartet=['1','0.3i','-0.2+0.4i','0.6'],EM_cloud=.27,
                           Phi_coordinate_jet=.07,SU4_coordinate_jet=.09,
                           field_kinetic=.65,field_levels=3,photon_levels=3,
                           finite_defect=coupling,finite_penalty=penalty,time=.7))


def run():
    native=native_probe()
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('bosonic_original_law_join.py','bosonic_original_law_join.md','bosonic_statistics.py',
             'bulk_quantum_selection.py','local_orbital_stability.py','rotation_band.py','sea_selection.py',
             'noncompact_coulomb_carrier.py')}
    return dict(status='constructive original-field bosonic finite-loop successor with actual microscopic source19 allocation',
                microscopic_CAR=False,comparison_CAR=True,source_hashes=hashes,
                input_sha256=hashlib.sha256(json.dumps(native['input'],sort_keys=True).encode()).hexdigest(),
                native=native,graph=symbolic_graph_contract(),budget=physical_budget_family(),
                joined_weighted_budget=quartic_weighted_transfer_budget(),
                original_microscopic_budget=original_microscopic_loop_budget(),
                topology_positive_controls=[topology_probe(3),topology_probe(4)],
                debt=['fundamental tensor-product quantum law and coefficients are inputs',
                      'finite penalty/defect is an explicit successor, not identical original bare dynamics',
                      'full physical receiver/normal/Noether composition remains owned by physical transfer supplier',
                      'torus flat holonomies require admitted sector/history; no long-loop local penalty',
                      'native Galerkin proves mechanism, not unrestricted quantum vacuum or physical spin'])


if __name__ == '__main__':
    print(json.dumps(run(),indent=2,sort_keys=True))
