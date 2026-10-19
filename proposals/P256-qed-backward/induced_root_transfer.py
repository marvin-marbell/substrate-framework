#!/usr/bin/env python3
"""NEW N-total charged-flavour primitive Pauli law, full-Phi and induced-current transfer.

N includes the active flavour's filled negative band. No old Maxwell coercivity,
old vacuum upper, endpoint CAR on Q, or whole-particle conclusion is inherited.
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
import numpy as np
from scipy.linalg import expm
from bosonic_statistics import BosonicEdgeModel, reference_alignment, reference_hop
from microscopic_native_orbit_lift import NativePauliGraph, PauliWord, EvenTerm
from induced_photon_response import ResponseConfig, make_band, finite_mode, LOADED_SOURCE_SHA256
from improved_phi_band import ImprovedPhiLaw
from rotation_band import RotationBand
from local_orbital_stability import scalar_paths
from sea_selection import matrix_unit_basis

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ('induced_root_transfer.py', 'induced_photon_response.py', 'improved_phi_band.py',
                'microscopic_native_orbit_lift.py', 'bosonic_statistics.py', 'rotation_band.py',
                'local_orbital_stability.py', 'sea_selection.py')
LOADED_HASHES = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
LOADED_HASHES['induced_photon_response.py'] = LOADED_SOURCE_SHA256


def rank(words: Any, edge_count: int) -> int:
    pivots: dict[int, int] = {}
    for word in words:
        value = word.x | (word.z << edge_count)
        while value:
            pivot = value.bit_length()-1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    return len(pivots)


def raw_polynomial(words: Any) -> dict[tuple[int,int],complex]:
    """Exact full-edge Pauli polynomial, with NO statistical cycle quotient."""
    result: dict[tuple[int,int],complex]={}
    for coefficient,word in words:
        key=word.x,word.z
        result[key]=result.get(key,0j)+coefficient*1j**word.phase
    return {key:value for key,value in result.items() if abs(value)>1e-12}


def full_tree_commutator(graph: NativePauliGraph,left: EvenTerm,right: EvenTerm) -> float:
    words=[(a*b,x@y) for a,x in left.words for b,y in right.words]
    words += [(-a*b,y@x) for a,x in left.words for b,y in right.words]
    if left.source==right.target:
        words.extend((-a,w) for a,w in graph.bilinear(left.target,right.source,left.coefficient*right.coefficient).words)
    if left.target==right.source:
        words.extend((a,w) for a,w in graph.bilinear(right.target,left.source,left.coefficient*right.coefficient).words)
    return sum(abs(value) for value in raw_polynomial(words).values())


class FlavorPauliGraph(NativePauliGraph):
    """T3 x species-tree, onsite colour trees and one actual reference tree.

    Reuses the exact primitive path/even-polynomial operations, not the old
    two-species constructor or old three-toron assumption. No dense global H.
    """
    def __init__(self, law: ImprovedPhiLaw, flavors: int, reference_site: int = 0):
        self.law, self.volume, self.reference_site = law, law.volume, reference_site
        self.flavors, self.species = flavors, flavors+1
        self.modes = 4*self.species*self.volume+4
        self.ref = tuple(range(self.modes-4, self.modes))
        edges = []
        for species in range(self.species):
            for site in range(self.volume):
                anchor = self.mode(species, site, 0)
                edges.extend((anchor, self.mode(species, site, c)) for c in range(1, 4))
                edges.extend((anchor, self.mode(species, int(law.target[site, j]), 0)) for j in range(3))
        # Nearest species-tree bridges glue statistical, not EM, torons.
        edges.extend((self.mode(s, x, 0), self.mode(s+1, x, 0))
                     for s in range(self.species-1) for x in range(self.volume))
        edges.extend((self.ref[0], self.ref[c]) for c in range(1, 4))
        edges.append((self.mode(1, reference_site, 0), self.ref[0]))
        self.edges = tuple(sorted(tuple(sorted(e)) for e in edges))
        if len(set(self.edges)) != len(self.edges):
            raise ValueError('Simple primitive graph required')
        self.edge_index = {e: j for j, e in enumerate(self.edges)}
        incident: list[list[int]] = [[] for _ in range(self.modes)]
        for j, pair in enumerate(self.edges):
            for v in pair:
                incident[v].append(j)
        self.incident = tuple(tuple(js) for js in incident)
        # Identical primitive_pauli_words law, bit-packed to avoid O(E^2) letters.
        self.B = tuple(PauliWord(z=sum(1 << e for e in es)) for es in self.incident)
        self.A = {}
        for e, pair in enumerate(self.edges):
            z = 0
            for v in pair:
                for earlier in self.incident[v]:
                    if earlier >= e:
                        break
                    z ^= 1 << earlier
            self.A[pair] = PauliWord(1 << e, z)
        self.parent: dict[int, int | None] = {0: None}
        tree = set()
        queue = deque([0])
        while queue:
            v = queue.popleft()
            for e in self.incident[v]:
                pair = self.edges[e]
                w = pair[1] if pair[0] == v else pair[0]
                if w not in self.parent:
                    self.parent[w] = v
                    queue.append(w)
                    tree.add(pair)
        if len(self.parent) != self.modes:
            raise ValueError('Primitive graph disconnected')
        self.cycles, self.cycle_windings = [], []
        for pair in self.edges:
            if pair not in tree:
                path = self.tree_path(pair[1], pair[0])
                vertices = (pair[0], *path[:-1])
                self.cycles.append(self.cycle(vertices))
                self.cycle_windings.append(self.winding((*vertices, vertices[0])))
        self.plaquettes, bridges = [], []
        for s in range(self.species):
            for x in range(self.volume):
                for j, k in itertools.combinations(range(3), 2):
                    y, z = int(law.target[x, j]), int(law.target[x, k])
                    yz = int(law.target[y, k])
                    self.plaquettes.append(self.cycle(tuple(self.mode(s, t, 0) for t in (x,y,yz,z))))
        for s in range(self.species-1):
            for x in range(self.volume):
                for j in range(3):
                    y = int(law.target[x,j])
                    bridges.append(self.cycle((self.mode(s,x,0), self.mode(s,y,0),
                                               self.mode(s+1,y,0), self.mode(s+1,x,0))))
        self.local_cycles = self.plaquettes+bridges
        self.spatial_edges = tuple(self.edge_index[tuple(sorted((self.mode(s,x,0),
            self.mode(s,int(law.target[x,j]),0))))] for s in range(self.species)
            for x in range(self.volume) for j in range(3))
        self.epsilon = 1/(16*len(self.spatial_edges))
        self.central_bases = {(1,1,1): self._central_basis((1,1,1))}
        self._decoded_terms = {}

    def winding(self, path: tuple[int, ...]) -> tuple[int, int, int]:
        counts = [0,0,0]
        for source, target in zip(path[:-1], path[1:]):
            if source >= self.modes-4 or target >= self.modes-4:
                continue
            if source//(4*self.volume) != target//(4*self.volume):
                continue
            x, y = (source%(4*self.volume))//4, (target%(4*self.volume))//4
            delta = self.law.sites[y]-self.law.sites[x]
            for j in range(3):
                if abs(int(delta[j])) == self.law.size-1:
                    counts[j] ^= 1
        return tuple(counts)  # type: ignore[return-value]

    def block_decode(self, terms: list[EvenTerm], species: int,
                     history: tuple[int,...] = (1,1,1), defect_edge: int | None = None,
                     defect_mask: int = 0) -> np.ndarray:
        if history not in self.central_bases:
            self.central_bases[history] = self._central_basis(history)
        offset = 4*species*self.volume
        out = np.zeros((4*self.volume,4*self.volume),complex)
        for term in terms:
            coefficient = term.coefficient
            if term.target != term.source:
                actual = self.path_A(term.path)
                canonical = self.path_A(self.tree_path(term.source,term.target))
                coefficient *= self.central_value(actual@canonical.adjoint(),history)
                mask=defect_mask | (0 if defect_edge is None else 1 << defect_edge)
                if (actual.x & mask).bit_count()%2:
                    coefficient *= -1
            out[term.target-offset,term.source-offset] += coefficient
        return out

    def topology(self) -> dict[str, Any]:
        e, v = len(self.edges), self.modes
        local_rank = rank(self.local_cycles,e)
        plaq_rank = rank(self.plaquettes,e)
        joint_rank = rank((*self.B[:-1],*self.cycles),e)
        if e-v+1-local_rank != 3 or joint_rank != e:
            raise AssertionError('Renewed full flavour cycle/occupation rank failed')
        primitive_errors = 0
        for pair, word in self.A.items():
            primitive_errors += int(word@word != PauliWord())
            primitive_errors += sum(not word.anticommutes(self.B[x]) for x in pair)
        # Shared-vertex A relations are local; disjoint supports can only overlap
        # in endpoint earlier Z strings, never a disjoint endpoint's X edge.
        for edges in self.incident:
            for a,b in itertools.combinations(edges,2):
                primitive_errors += int(not self.A[self.edges[a]].anticommutes(self.A[self.edges[b]]))
        product = PauliWord()
        for b in self.B:
            product = product@b
        primitive_errors += int(product != PauliWord())
        primitive_errors += sum(word@word != PauliWord() for word in (*self.cycles,*self.local_cycles))
        return dict(charged_flavors=self.flavors, active_is_tagged_charged_flavor=True,
            heavy_neutral_species=1, actual_reference_modes=4, modes=v, edge_qubits=e,
            independent_cycles=e-v+1, local_constraint_rank=local_rank,
            derived_common_statistical_torons=3, flat_histories=2**3,
            joint_occupation_cycle_rank=joint_rank, primitive_algebra_errors=primitive_errors,
            omitted_bridge_unfixed_cycles=e-v+1-plaq_rank,
            omitted_bridge_extra_cycles=e-v+1-plaq_rank-3,
            flux_space='statistical Z2 cycles distinct from EM real U1 and SU4 colour centre spaces',
            D='epsilon sum Z_e over every species spatial anchor edge', epsilon=self.epsilon,
            beta=len(self.spatial_edges)*self.epsilon,
            full_edge_Hilbert_dimension=f'2^{e}', dense_global_allocation=False)


def coefficient_caps(band: RotationBand) -> dict[str,F]:
    onsite,paths = scalar_paths(band)
    matter = 4*abs(F(float(onsite)))
    current = F(0)
    contact = F(0)
    for displacement,scalar in paths:
        length = int(np.sum(abs(displacement)))
        cap = 8*sum((abs(F(float(z.real)))+abs(F(float(z.imag))) for z in scalar),F(0))
        matter += cap
        current += length*cap
        contact += length*length*cap
    return dict(matter=matter,current=current,contact=contact)


def renewed_bounds(config: ResponseConfig, volume: int, heavy_scale: int = 256,
                   field_sigma: float = .5) -> dict[str,Any]:
    """Explicit all-sector quartic lower and Gauss trial, not a vacuum claim.

    r=sum_mu ||Phi_mu||_F^2 square root. Each Phi is traceless Hermitian,
    SU4 paths unitary and EM paths modulus one. Every Pauli hop has norm <=1.
    """
    light,_,_ = make_band(config)
    heavy = RotationBand(a=config.a,m=.2,r=.12,v=.3,improved=False)
    lc,hc = coefficient_caps(light),coefficient_caps(heavy)
    groups=config.species or ((config.flavors,config.m),)
    species_caps=[(count,mass,coefficient_caps(RotationBand(a=config.a,m=mass,
        r=config.r,v=config.v,e=1.,improved=True))) for count,mass in groups]
    totals={key:sum((count*cap[key] for count,_,cap in species_caps),F(0))
            for key in ('matter','current','contact')}
    lam, kappa = F(heavy_scale),F(2)/F(config.a)
    c0 = kappa/128
    K = lam*hc['matter']+totals['matter']
    # r <= r^2+1/4 and (c0*Lambda/2)r^4-Kr^2 >= -K^2/(2c0Lambda).
    per_site_B = K*K/(2*c0*lam)+K/4
    B = volume*per_site_B
    sigma2 = F(field_sigma)**2
    # Matrix basis has Tr(T_a T_b)=4 delta_ab. r^2=4 sum_60 C^2.
    # E r^4 =16*60*62*sigma^4. ||G||^2 <=r^4/16.
    quartic_trial = lam*kappa*F(60*62,2)*sigma2*sigma2
    phi_kinetic = F(60,8)/sigma2/max(1,math.isqrt(heavy_scale))
    # Round positive Lambda^(5/4) upward for the rational trial upper.
    lambda54 = F(math.ceil(heavy_scale**1.25))
    alignment_trial = F(1,5)*lambda54*3*480*sigma2/8
    color_magnetic_trial = F(7,10)*lambda54*3*4
    # All coexact and harmonic real coordinates retained; E p^2=1/4.
    physical_coordinates = 2*volume+1
    ue_link = F(config.inertia)/F(config.a)
    photon_trial = ue_link*physical_coordinates/4
    trial = volume*(quartic_trial+phi_kinetic+alignment_trial+color_magnetic_trial)+photon_trial
    J = totals['current']
    C = totals['contact']
    # Ref source sum over four colour transfers has norm <=4 in P AND Q;
    # occupation projector source-normalization is evaluated independently.
    return dict(volume=volume, heavy_scale=heavy_scale, charged_flavors=config.flavors,
        coefficient_l1={k:str(v) for k,v in lc.items()}, heavy_coefficient_l1={k:str(v) for k,v in hc.items()},
        species_coefficient_ledger=[dict(count=count,mass=mass,coefficient_l1={k:str(v) for k,v in cap.items()})
                                   for count,mass,cap in species_caps],
        charged_total_coefficient_l1={k:str(v) for k,v in totals.items()},
        wrong_all_active_matter_load_defect=str(totals['matter']-config.flavors*lc['matter']),
        K=str(K), quartic_c0=str(c0), retained_quartic_coercivity=str(c0*lam/2),
        vacuum_lower=-float(B), B_exact=str(B), lower_derivation='c0 Lambda r^4-Kr >= (c0 Lambda/2)r^4-[K^2/(2c0 Lambda)+K/4]',
        current_l1=str(J),contact_l1=str(C),source_Ref_operator_cap=4,
        field_trial=dict(C_coordinate_variance=float(sigma2),real_C_coordinates=60*volume,
            phi_quartic_upper=float(quartic_trial)*volume,phi_kinetic_upper=float(phi_kinetic)*volume,
            alignment_upper=float(alignment_trial)*volume,color_magnetic_upper=float(color_magnetic_trial)*volume,
            colour_links='constant Haar, electric Casimir zero',
            matter='heavy empty; pairwise charged wedge2 SU4 singlets, Ref wedge4 (+) or wedge0 (-)',
            gauss='rho=sum charged(n-2)+nRef-neutralRef; every trial cell rho=0; all local SU4 singlets',
            charged_trial='apply sum_c primitive taggedflavor[c]^dagger Ref[c] (+), adjoint (-); nonzero SU4 scalar, exact colocated rho=0',
            matter_expectation=0,reference_penalty=0,loop_penalty=0,defect_expectation=0,
            EM_preparation='normalized width1 Gaussian on all coexact/harmonic real coordinates; no magnetic potential',
            bare_UE_phys=config.inertia,bare_UE_link=float(ue_link),physical_EM_coordinates=physical_coordinates,
            actual_bare_electric_upper=float(photon_trial),global_oscillator_ground_asserted=False),
        code_trial_upper=float(trial),code_trial_upper_exact=str(trial),
        pairwise_trial_requires_even_flavors=True,
        no_old_four_c_photon_upper=True)


def native_current(config: ResponseConfig, graph: FlavorPauliGraph) -> dict[str,Any]:
    _,frame,law = make_band(config)
    rng = np.random.default_rng(222)
    fields = np.tile(frame,(law.volume,1,1,1))+.021*law.fields(rng.normal(size=(law.volume,4,15)))
    links = np.asarray([[expm(.03j*sum(rng.normal()*b for b in law.generators)) for _ in range(3)] for _ in range(law.volume)])
    a = rng.normal(scale=.025,size=(law.volume,3))
    first = rng.normal(size=a.shape)
    second = rng.normal(size=a.shape)
    errors = {}
    cached = []
    for name,u,w in (('H',None,None),('current',first,None),('contact',first,second)):
        terms = graph.path_terms(law,fields,links,a,1,first=u,second=w)
        actual = graph.block_decode(terms,1)
        direct = law.onebody(fields,links,a,first=u,second=w)
        errors[name] = float(np.linalg.norm(actual-direct))
        cached.append((terms,actual))
    group_errors=[]
    representative=1
    for count,mass in (config.species or ((config.flavors,config.m),)):
        group_band=RotationBand(a=config.a,m=mass,r=config.r,v=config.v,e=1.,improved=True)
        group_law=ImprovedPhiLaw(law.size,group_band)
        terms=graph.path_terms(group_law,fields,links,a,representative)
        decoded=graph.block_decode(terms,representative)
        error=float(np.linalg.norm(decoded-group_law.onebody(fields,links,a)))
        wrong_mass=float(np.linalg.norm(decoded-cached[0][1]))
        if error>3e-7 or (mass!=config.m and wrong_mass<1e-6):
            raise AssertionError('Actual mass-resolved primitive comparison or wrong-active-mass control failed')
        group_errors.append(dict(count=count,mass=mass,representative_species=representative,H_error=error,
                                 wrong_all_active_mass_H_defect=wrong_mass))
        representative+=count
    jterms,j = cached[1]
    dx=1e-5
    numerical=(law.onebody(fields,links,a+dx*first)-law.onebody(fields,links,a-dx*first))/(2*dx)
    errors['same_H_current_derivative'] = float(np.linalg.norm(j-numerical))
    # Q is a different flux sector, not claimed to be endpoint CAR.
    edge = next(e for e in graph.spatial_edges if graph.edges[e][0]//(4*graph.volume)==1)
    wrong_cycle=graph.block_decode(jterms,1,defect_edge=edge)
    cycle_price=float(np.linalg.norm(wrong_cycle-j))
    other_history=graph.block_decode(jterms,1,history=(-1,1,1))
    history_price=float(np.linalg.norm(other_history-j))
    if history_price<=1e-7:
        raise AssertionError('Noncontractible statistical history failed to affect current')
    if cycle_price <= 1e-7:
        raise AssertionError('Actual current failed to detect wrong cycle')
    sheet=PauliWord(z=sum(1 << graph.edge_index[tuple(sorted((graph.mode(1,x,0),
        graph.mode(1,int(law.target[x,0]),0))))] for x in range(law.volume)
        if law.sites[x,0]==law.size-1))
    plaquette_violations=sum(sheet.anticommutes(w) for w in graph.plaquettes)
    bridge_violations=sum(sheet.anticommutes(w) for w in graph.local_cycles[len(graph.plaquettes):])
    omitted_bridge_image=float(np.linalg.norm(graph.block_decode(jterms,1,defect_mask=sheet.z)-j))
    if plaquette_violations or not bridge_violations or omitted_bridge_image<1e-7:
        raise AssertionError('Omitted bridge control did not produce a valid extra flavour flux/current')
    # Full Pauli polynomial, not a cycle quotient: local colour tree algebra
    # genuinely holds in every spatial flux sector.
    colours=tuple(graph.mode(1,0,c) for c in range(4))
    su4_error=max(full_tree_commutator(graph,graph.bilinear(colours[i],colours[k],1),
        graph.bilinear(colours[k],colours[j],1)) for i,j,k in itertools.product(range(4),repeat=3))
    mode=finite_mode(config,np.array([.05,.035,.02]),velocity=False)
    consumers=[]
    energies,eigenvectors=np.linalg.eigh(cached[0][1])
    active=eigenvectors[:,2*law.volume]
    occupied=eigenvectors[:,:2*law.volume]
    sea_projector=occupied@occupied.conj().T
    charged_projector=sea_projector+np.outer(active,active.conj())
    target=eigenvectors[:,2*law.volume+1]
    momentum=np.asarray(mode['q'])
    site_wave=np.cos(config.a*(law.sites@momentum))
    density=np.diag(np.repeat(site_wave,4))
    density_terms=graph.lift_matrix(density,tuple(graph.mode(1,x,c) for x in range(law.volume) for c in range(4)))
    primitive_density=graph.block_decode(density_terms,1)
    if np.linalg.norm(primitive_density-density)>1e-12:
        raise AssertionError('Primitive density/A0 vertex disagrees with onsite charge')
    for pole in mode['two_polarizations']:
        vector=np.asarray(pole['polarization'])
        profile=config.a*vector[None,:]*np.cos(config.a*((law.sites@momentum)[:,None]+momentum[None,:]/2))
        primitive=graph.block_decode(graph.path_terms(law,fields,links,a,1,first=profile),1)
        direct=law.onebody(fields,links,a,first=profile)
        amputator=math.sqrt(pole['residue']/(2*pole['omega']))
        scalar=float(pole['scalar_polarization'])
        vertex=scalar*primitive_density-primitive
        image=amputator*vertex
        # Active-flavour sea background plus actual Ref charge-1 at site0.
        background=amputator*scalar*(2*float(np.sum(site_wave))+float(site_wave[graph.reference_site]))
        mean=np.trace(charged_projector@image)-background
        connected=np.trace(charged_projector@image@(np.eye(len(image))-charged_projector)@image).real
        selected_state_image=math.sqrt(max(0.,float(abs(mean)**2+connected)))
        transition=complex(np.vdot(target,image@active))
        if selected_state_image<=0:
            raise AssertionError('Native charged code state did not consume polarized photon current')
        consumers.append(dict(omega=pole['omega'],residue=pole['residue'],Aphys_amputator=amputator,
            scalar_polarization=scalar,
            native_primitive_photon_current_norm=float(np.linalg.norm(image)),
            same_current_direct_error=float(np.linalg.norm(amputator*(vertex-(scalar*density-direct)))),
            missing_residue_image_error=float(np.linalg.norm((amputator-1)*vertex)),
            omitted_A0_density_image_defect=float(np.linalg.norm(amputator*scalar*primitive_density)),
            active_code_Slater_current_image_norm=selected_state_image,
            active_particle_transition=[transition.real,transition.imag],
            active_native_energy=float(energies[2*law.volume]),
            state_scope='actual fixed-field code Slater sector with filled negative doublet plus one tagged particle; Ref compensation; not full field Gauss-prepared sea'))
    if max(errors.values())>3e-6 or su4_error>1e-10:
        raise AssertionError('Actual primitive H/current/contact/algebra comparison failed')
    return dict(native_size=law.size,all_Cartesian_Phi_coordinates=60*law.volume,
        independent_SU4_and_EM_links=True,errors=errors,onsite_even_algebra_error=su4_error,
        actual_mass_resolved_primitive_groups=group_errors,
        wrong_Q_cycle_current_defect=cycle_price,nontrivial_statistical_toron_current_defect=history_price,
        photon_consumers=consumers,
        photon_mode_momentum=mode['q'],
        native_input_ledger=dict(seed=222,Cartesian_Phi_perturbation=.021,
            SU4_generator_coordinate_scale=.03,EM_link_coordinate_scale=.025,
            sea_occupied_doublet_dimension=2*law.volume,active_species=1),
        omitted_bridge_control=dict(preserves_all_species_plaquettes=plaquette_violations==0,
            actual_bridge_violations=bridge_violations,current_image_defect=omitted_bridge_image),
        physical_consumer='both induced polarizations normalized sqrt(Z/(2omega)); local current jet A_link=a Aphys and actual charged code Slater image, not an exact N3 plane wave',
        actual_same_path_parameters=asdict(config),endpoint_CAR_claim_on_all_Q=False)


def gauss_source(graph: FlavorPauliGraph) -> dict[str,Any]:
    """Actual primitive SU4 pair singlet and flavour/Ref Gauss source algebra."""
    tree=BosonicEdgeModel(8,((0,1),(0,2),(0,3),(0,4),(4,5),(4,6),(4,7)))
    def local_bilinear(target: int,source: int) -> np.ndarray:
        if target==source:
            return tree.n(source)
        anchor=0 if source<4 else 4
        path=tuple(([source] if source!=anchor else [])+[anchor]+([target] if target!=anchor else []))
        return tree.bilinear(target,source,path=path)
    _,occupation=tree.occupation_basis(4)
    left_number=sum((tree.n(c) for c in range(4)),np.zeros_like(tree.identity))
    right_number=sum((tree.n(c) for c in range(4,8)),np.zeros_like(tree.identity))
    selector=occupation.conj().T@((left_number-2*tree.identity)@(left_number-2*tree.identity)
                                 +(right_number-2*tree.identity)@(right_number-2*tree.identity))@occupation
    numbers,vectors=np.linalg.eigh(selector)
    allowed=occupation@vectors[:,numbers<1e-8]
    generators=[]
    for T in matrix_unit_basis():
        G=np.zeros_like(tree.identity)
        for offset in (0,4):
            for i,j in itertools.product(range(4),repeat=2):
                G+=T[i,j]*local_bilinear(offset+i,offset+j)
        generators.append(G)
    casimir=sum((G@G for G in generators),np.zeros_like(tree.identity))
    eigenvalues,eigenvectors=np.linalg.eigh(allowed.conj().T@casimir@allowed)
    state=allowed@eigenvectors[:,0]
    gauss_error=max(float(np.linalg.norm(G@state)) for G in generators)
    occupations=[float(np.vdot(state,tree.n(c)@state).real) for c in range(4)]
    # Ref plus starts filled: ||sum a_c^dag r_c psi||^2=sum <1-n_c>=2.
    # Ref minus starts empty: adjoint norm squared=sum <n_c>=2.
    source_plus=sum(1-n for n in occupations)
    source_minus=sum(occupations)
    colours=tuple(graph.mode(1,graph.reference_site,c) for c in range(4))
    source=[term for c in range(4) for term in graph.bilinear(colours[c],graph.ref[c],1).words]
    def commutator(words: Any) -> float:
        result=[(a*b,x@y) for a,x in words for b,y in source]
        result += [(-a*b,y@x) for a,x in words for b,y in source]
        return sum(abs(c) for c in raw_polynomial(result).values())
    colour_errors=[]
    for T in matrix_unit_basis():
        words=[]
        for modes in (colours,graph.ref):
            for i,j in itertools.product(range(4),repeat=2):
                if T[i,j]!=0:
                    words.extend(graph.bilinear(modes[i],modes[j],complex(T[i,j])).words)
        colour_errors.append(commutator(words))
    em_words=[term for mode in (*colours,*graph.ref)
              for term in graph.bilinear(mode,mode,1).words]
    em_error=commutator(em_words)
    if gauss_error>1e-8 or max(colour_errors)>1e-8 or em_error>1e-8 or min(source_plus,source_minus)<1.9:
        raise AssertionError('Actual flavour/source Gauss construction failed')
    return dict(pairwise_primitive_SU4_singlet_error=gauss_error,
        pair_site_flavour_occupations=occupations,neutral_background_per_cell=2*graph.flavors,
        source_Ref_plus_norm_squared=source_plus,source_Ref_minus_norm_squared=source_minus,
        actual_source_SU4_commutator_error=max(colour_errors),actual_source_EM_commutator_error=em_error,
        plus_reference='nRef=4 neutral, nRef=3 charge-1',minus_reference='nRef=0 neutral, nRef=1 charge+1',
        full_Gauss_trial='tensor actual onsite pair singlets, scalar Cartesian Gaussian Phi, constant Haar links, zero-charge Gaussian coexact/harmonic A',
        charged_trial='nonzero primitive scalar source applied to that exact full-Gauss trial; same field energy, matter expectation zero',
        onsite_algebra_applies_all_spatial_flux_sectors=True,
        source_norm_claim_scope='explicit trial, not interacting ground/source receiver')


def dense_native(config: ResponseConfig) -> dict[str,Any]:
    """Exact full bosonic cyclic cell with real photon jet and nonzero flux D."""
    model=BosonicEdgeModel(4,((0,1),(1,2),(2,3),(0,3)))
    masks,basis,_,alignment=reference_alignment(model,2)
    one=np.zeros((4,4),complex)
    _,frame,law=make_band(config)
    block=np.einsum('m,mij->ij',law.paths[0][1],frame)
    colour_pair=np.unravel_index(np.argmax(abs(block)),block.shape)
    coefficient=complex(block[colour_pair])
    for k,(i,j) in enumerate(model.edges):
        one[j,i]=coefficient*np.exp(.11j*(k+1));one[i,j]=one[j,i].conjugate()
    H=model.quadratic(one)
    J=np.zeros_like(H);C=np.zeros_like(H);car=np.zeros((len(masks),len(masks)),complex)
    commuting=np.zeros_like(car)
    for i,j in model.edges:
        hop=model.hop(i,j)
        J+=1j*one[j,i]*hop-1j*one[i,j]*hop.conj().T
        C-=one[j,i]*hop+one[i,j]*hop.conj().T
        car+=1j*one[j,i]*reference_hop(4,masks,i,j)-1j*one[i,j]*reference_hop(4,masks,j,i)
        commuting+=1j*one[j,i]*reference_hop(4,masks,i,j,False)-1j*one[i,j]*reference_hop(4,masks,j,i,False)
    current_error=float(np.linalg.norm(J@basis-basis@car))
    wrong_commuting=float(np.linalg.norm(car-commuting))
    # Real A quantum jet; only E^2 is bare. Peierls exponential is exact.
    p=np.array([[0,-1j,0],[1j,0,-1j*math.sqrt(2)],[0,1j*math.sqrt(2),0]])/math.sqrt(2)
    x=np.array([[0,1,0],[1,0,math.sqrt(2)],[0,math.sqrt(2),0]])/math.sqrt(2)
    ue_link=config.inertia/config.a
    incidence=np.zeros((4,len(model.edges)))
    for e,(i,j) in enumerate(model.edges):
        incidence[i,e]=1;incidence[j,e]=-1
    transverse=np.array([.5,-.5,.5,.5])  # unit cycle mode in sorted edge order
    if np.linalg.norm(incidence@transverse)>1e-12:
        raise AssertionError('Native photon coordinate is not transverse')
    green=np.linalg.pinv(incidence@incidence.T,hermitian=True)
    rho=[model.n(i)-model.identity/2 for i in range(4)]
    coulomb=sum((ue_link*green[i,j]*(rho[i]@rho[j])
                 for i,j in itertools.product(range(4),repeat=2)),np.zeros_like(model.identity))
    longitudinal=[sum((float((incidence.T@green)[e,i])*rho[i] for i in range(4)),np.zeros_like(model.identity))
                  for e in range(len(model.edges))]
    number=sum((model.n(i) for i in range(4)),np.zeros_like(model.identity))
    number_two=np.diag((abs(np.diag(number)-2)<1e-10).astype(float))
    gauss_error=max(float(np.linalg.norm((sum((incidence[i,e]*longitudinal[e] for e in range(len(model.edges))),
        np.zeros_like(model.identity))-rho[i])@number_two)) for i in range(4))
    h0=np.kron(model.identity,ue_link*p@p)+np.kron(coulomb,np.eye(3))
    coupled_J=np.zeros_like(h0)
    coupled_C=np.zeros_like(h0)
    field_potential=np.zeros_like(h0)
    shifted_plus=np.zeros_like(h0);shifted_minus=np.zeros_like(h0)
    dx=1e-4
    for e,(i,j) in enumerate(model.edges):
        hop=model.hop(i,j)
        phase=expm(1j*transverse[e]*x)  # A_link=e_trans*x, charge one
        forward=np.kron(one[j,i]*hop,phase)
        backward=np.kron(one[i,j]*hop.conj().T,phase.conj().T)
        field_potential+=forward+backward
        coupled_J+=1j*transverse[e]*(forward-backward)
        coupled_C-=transverse[e]**2*(forward+backward)
        shifted_plus+=np.exp(1j*dx*transverse[e])*forward+np.exp(-1j*dx*transverse[e])*backward
        shifted_minus+=np.exp(-1j*dx*transverse[e])*forward+np.exp(1j*dx*transverse[e])*backward
    h0+=field_potential
    derivative_error=float(np.linalg.norm((shifted_plus-shifted_minus)/(2*dx)-coupled_J))
    contact_error=float(np.linalg.norm((shifted_plus+shifted_minus-2*field_potential)/dx**2-coupled_C))
    P=np.kron(model.sector_projector(),np.eye(3));Q=np.eye(len(P))-P
    beta=.0625
    # Actual Z_e commutes with every occupation and Gauss charge.
    z=np.diag([1 if not (n&1) else -1 for n in range(model.dimension)])
    D=beta*np.kron(z,np.eye(3))
    lower=float(np.min(np.linalg.eigvalsh(h0)))
    trial=np.kron(model.state((0,2)),np.array([1.,0.,0.]))
    upper=float(np.vdot(trial,h0@trial).real)
    window=1.;gap=128.;penalty=gap+upper-lower+window+beta
    full=h0+penalty*Q+D
    code=h0
    initial=trial
    t=.4
    evolved=expm(-1j*t*full)@initial
    ideal=expm(-1j*t*code)@initial
    leakage=float(np.linalg.norm(Q@evolved))
    image_error=float(np.linalg.norm(coupled_J@(evolved-ideal)))
    schur=beta*beta/gap
    bound=(4*beta+t*beta*beta)/gap
    zero_penalty_Q=float(np.linalg.norm(Q@expm(-1j*t*(h0+D))@initial))
    if (alignment>1e-10 or current_error>1e-10 or wrong_commuting<1e-4
            or derivative_error>1e-6 or contact_error>1e-5 or gauss_error>1e-9 or leakage<=0
            or zero_penalty_Q<=leakage or leakage>bound
            or image_error>float(np.linalg.norm(coupled_J,2)*bound)):
        raise AssertionError('Native primitive adverse or finite-gap control failed')
    return dict(primitive_edge_dimension=model.dimension,physical_photon_jet_dimension=3,
        native_cell_coefficient=dict(actual_one_link_Phi_component=[coefficient.real,coefficient.imag],
                                     selected_colour_pair=list(map(int,colour_pair)),
                                     full_3D_path_comparison_owned_by_native_current=True),
        alignment_error=alignment,current_CAR_comparator_error=current_error,
        wrong_commuting_current_defect=wrong_commuting,actual_bare_kinetic_trial=float(np.vdot(trial,np.kron(model.identity,ue_link*p@p)@trial).real),
        exact_native_EM_Gauss_residual=gauss_error,
        actual_Coulomb_trial=float(np.vdot(trial,np.kron(coulomb,np.eye(3))@trial).real),
        native_EM_field_domain='unit-normalized transverse cycle coordinate; longitudinal E=Incidence^T Lplus rho; number-two physical neutral sector',
        full_H_primitive=True,current_contact_same_H=True,contact_norm=float(np.linalg.norm(coupled_C)),
        current_derivative_error=derivative_error,contact_derivative_error=contact_error,
        finite_positive_beta=beta,noncommuting_DH_norm=float(np.linalg.norm(D@h0-h0@D)),
        noncommuting_DJ_norm=float(np.linalg.norm(D@coupled_J-coupled_J@D)),
        actual_allsector_H0_lower=lower,actual_code_trial_upper=upper,penalty=penalty,gap_lower=gap,
        schur_residual_bound=schur,time=t,observed_Q_amplitude=leakage,
        current_image_error=image_error,current_image_bound=float(np.linalg.norm(coupled_J,2)*bound),
        evolution_amplitude_bound=bound,zero_penalty_Q_control=zero_penalty_Q,
        scope='exact finite Pauli/quantumfield Galerkin; not unrestricted field ground or global physical pole')


def weighted_transfer(config:ResponseConfig, bounds:dict[str,Any], current:dict[str,Any]) -> dict[str,Any]:
    beta=F(1,16)
    window=F(1)
    B=F(bounds['B_exact'])
    upper=F(bounds['code_trial_upper_exact'])
    c=F(bounds['retained_quartic_coercivity'])
    J=F(bounds['current_l1']);C=F(bounds['contact_l1'])
    photon_price=max(p['Aphys_amputator'] for p in current['photon_consumers'])
    # quartic graph: ||r z|| <= c^-1/4 (beta^2/G+C0 beta^2/G^2)^1/4 (beta/G)^1/2.
    # r_total^4 <= V sum r_x^4. C0=B+E+beta pays the bounded defect.
    volume=bounds['volume']
    amplitude=F(photon_price)
    coefficient=4*max(F(config.a)*J*amplitude,F(config.a)**2*C*amplitude**2)*volume
    density_coefficient=4*max(abs(F(p['scalar_polarization'])*F(p['Aphys_amputator']))
        for p in current['photon_consumers'])*(6*config.flavors*volume+4)
    gap=F(2**80)
    penalty=B+upper+window+beta+gap
    E=upper+window
    radius_graph=(volume/float(c))**.25*float(beta)*float(gap)**(-.75)*(1+float((B+E+beta)/gap))**.25
    density_error=2*float(density_coefficient*beta/gap)
    image=2*float(coefficient)*radius_graph+density_error
    ordinary=float(8*beta/gap)
    failed_gap=F(128)+(-B)-upper-window-beta
    if failed_gap>0 or image>=1e-8:
        raise AssertionError('Renewed quartic/gap adverse control failed')
    return dict(finite_positive_beta=str(beta),local_defect_epsilon=f'1/(16*3*(Ntotal+1)*V)',
        controlled_complement='Q_stat = nonflat statistical Pauli cycle sectors ONLY',
        retained_sea_particle_holes='in P_stat dynamics, not removed by Delta; Q_sea/interband gap is NOT this Schur gap',
        actual_D_commutes_Phi_and_Gauss=True,gap_exact=str(gap),penalty_exact=str(penalty),
        vacuum_advantage_paid_exact=str(B+upper),finite_window=str(window),
        leakage_bound=float(beta/gap),schur_bound=float(beta*beta/gap),
        current_contact_source_Ref_and_photon_coefficient=float(coefficient),
        bounded_density_A0_Ref_coefficient=float(density_coefficient),bounded_density_A0_image_bound=density_error,
        physical_current_a_factor=config.a,physical_contact_two_leg_a2_factor=config.a**2,
        weighted_Phi_gap_power='3/4',rho_resolvent_graph=radius_graph,
        current_contact_photon_image_bound=image,bounded_source_image_bound=ordinary,
        failed_old_small_penalty_gap=float(failed_gap),old_small_penalty_rejected=True,
        photon_coercivity='none global; photon here is actual finite-q spectral CURRENT amputator, not an A-coordinate bound',
        full_unbounded_Phi_allsector_form_lower=True,
        scope='finite volume source/current/contact aux Schur with Phi multiplication graph; no arbitrary time evolved state graph asserted')


def run(config:ResponseConfig) -> dict[str,Any]:
    if config.flavors%2:
        raise ValueError('Selected explicit pairwise Gauss trial requires even Ntotal')
    _,_,law=make_band(config)
    graph=FlavorPauliGraph(law,config.flavors)
    topology=graph.topology()
    if topology['primitive_algebra_errors']:
        raise AssertionError('Primitive algebra failed')
    current=native_current(config,graph)
    bounds=renewed_bounds(config,law.volume)
    gauss=gauss_source(graph)
    native=dense_native(config)
    transfer=weighted_transfer(config,bounds,current)
    changed=[n for n,h in LOADED_HASHES.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    if changed:
        raise RuntimeError(f'Loaded source changed during calculation: {changed}')
    return dict(branch='NEW no-bare-curl induced root transfer, not CURRENT427',
        parameters=asdict(config),Ntotal_semantics='ALL filled negative Dirac bands INCLUDING tagged active flavour',
        governing_plan='issue-222-induced-photon-continuation-20261008',
        scientific_assignment_epoch='f43102786fc92abe69dd80ebcd9c28b290787094276bf5b801c29fe66140dee2',
        quantum_inputs='complex tensor product, bosonic edge Pauli statistics, canonical R60V/SU4/real-U1 quantization',
        topology=topology,native_current=current,renewed_fullPhi_bounds=bounds,native_full_coupled=native,
        flavour_Gauss_source=gauss,
        weighted_current_photon_transfer=transfer,
        debts=['No full interacting quantum ground, no global real-A coercivity.',
               'Finite-q induced pole is leading collective/RPA mode, not an eternal full-law pole.',
               'Native full-H jet and auxiliary Schur do not close selfenergy/vertex/scattering/spin/spacetime campaign.'],
        capsule=dict(loaded_source_sha256=LOADED_HASHES,input_sha256=hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode()).hexdigest(),
            python=sys.version,executable=sys.executable,
            executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            numpy=np.__version__,platform=platform.platform()))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a',type=float,default=.25)
    parser.add_argument('--flavors',type=int,default=64)
    parser.add_argument('--order',type=int,default=4)
    parser.add_argument('--inertia',type=float,default=1.)
    parser.add_argument('--species',type=str,default='',help='Count:mass pairs, e.g. 1:0.2,63:2; total includes tagged active flavour')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    groups=tuple((int(item.split(':')[0]),float(item.split(':')[1])) for item in args.species.split(',')) if args.species else ()
    config=ResponseConfig(a=args.a,flavors=args.flavors,orders=(args.order,),inertia=args.inertia,species=groups)
    result=run(config)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(f'Natural exit: {args.output}; primitive H/current/contact, both induced photon modes and finite weighted gap checked',flush=True)


if __name__=='__main__':
    main()
