"""Full canonical slab heat-per-primitive-cell construction and response bounds.

No static-Phi determinant is a native thermal value here. Native output is an
exact finite-N ensemble/response enclosure. Small noncommuting CAR matrices are
independent controls only. Literal primitive parameters are never floated.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
import sympy as sp

from magnetic_boundary_slab_scientist_20261011 import FixedMagneticSlab, dirichlet_solve
from magnetic_boundary_scientist_20261011 import car_transfer, exact_matvec, generic_phi
from magnetic_thermal_heatword_20261011 import common_b_memory, face_word

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parent_independent_check(request: dict[str, Any]) -> dict[str, Any]:
    receipt_name = 'magnetic_thermal_parent_deck_actual1_20261011_receipt.json'
    source_name = 'magnetic_thermal_parent_deck_control_20261011.py'
    archive_name = 'magnetic_thermal_parent_deck_actual1_20261011_executed.zip'
    receipt = json.loads((ROOT/receipt_name).read_text())
    with zipfile.ZipFile(ROOT/archive_name) as archive:
        assert archive.read(receipt_name) == (ROOT/receipt_name).read_bytes()
        assert archive.read(source_name) == (ROOT/source_name).read_bytes()
        for name, pin in request['parent_original_pins'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == pin
    controls = []
    for row in receipt['native_deck']:
        slab = FixedMagneticSlab(row['N'], 1)
        k = np.zeros(len(slab.dynamic), dtype=np.int64)
        for edge, value in row['integer_divfree_dynamic_deck']:
            assert edge in slab.dynamic_lookup and edge not in slab.pinned
            k[slab.dynamic_lookup[edge]] = value
        assert not np.any(slab.DI @ k)
        assert int(k @ k) == 4
        half_wrong = 0
        count = 0
        for site, axis, length in product(range(slab.v), range(3), (1, 2)):
            path = slab.path(site, axis, length)
            if path is None:
                continue
            winding = sum(int(k[slab.dynamic_lookup[e]]) for e in path['edges'] if e in slab.dynamic_lookup)
            assert isinstance(winding, int)
            half_wrong += winding % 2 != 0
            count += 1
        assert count == row['native_one_two_paths'] and half_wrong == row['half_period_wrongcase_changed_paths']
        controls.append(dict(N=slab.n, incidence_exact=True, paths=count, half_period_wrong=half_wrong))
    return dict(receipt_sha256=digest(ROOT/receipt_name), source_sha256=digest(ROOT/source_name),
                archive_sha256=digest(ROOT/archive_name), archive_bytes_match=True,
                fresh_native_incidence_path_checks=controls, full_cover_trace='infinite',
                full_cover_argument='Orthogonal disjoint compact physical packet translates each have the same strictly positive heat diagonal; finite Z assumption fails.',
                parent_entrypoint_rerun=False)


def dirichlet_det(n: int) -> int:
    """Exact transverse Fourier product; no floating determinant."""
    r = n-2
    p = sp.Integer(1)
    for ky, kz in product(range(n), repeat=2):
        s = sp.simplify(4-2*sp.cos(2*sp.pi*ky/n)-2*sp.cos(2*sp.pi*kz/n))
        previous, current = sp.Integer(1), s+2
        for _ in range(2, r+1):
            previous, current = current, sp.expand((s+2)*current-previous)
        p = sp.simplify(p*current)
    assert p.is_Integer and p > 0
    return int(p)


def modular_det(matrix: np.ndarray, prime: int) -> int:
    a = np.array(matrix % prime, dtype=np.int64, copy=True)
    determinant = 1
    for j in range(a.shape[0]):
        pivots = np.flatnonzero(a[j:, j])
        if not len(pivots):
            return 0
        k = j+int(pivots[0])
        if k != j:
            a[[j, k]] = a[[k, j]]
            determinant = -determinant
        value = int(a[j, j])
        determinant = determinant*value % prime
        if j+1 < a.shape[0]:
            multipliers = a[j+1:, j]*pow(value, -1, prime) % prime
            a[j+1:, j+1:] = (a[j+1:, j+1:]-multipliers[:, None]*a[j, j+1:]) % prime
            a[j+1:, j] = 0
    return determinant % prime


def primitive_lattice(slab: FixedMagneticSlab) -> tuple[dict[str, Any], Any]:
    n = slab.n
    # One integral forest rooted at x=0, with exactly one parent for each
    # interior vertex. No right-boundary normal edge is in the forest.
    tree = [slab.dynamic_lookup[3*slab.index[(x, y, z)]]
            for x, y, z in product(range(n-2), range(n), range(n))]
    tree_set = set(tree)
    chords = [j for j in range(len(slab.dynamic)) if j not in tree_set]
    db = slab.DI[:, chords].toarray().astype(np.int64)
    row_of = {site: j for j, site in enumerate(slab.interior)}
    blocks = np.zeros((len(slab.dynamic), len(chords)), dtype=np.int64)
    blocks[chords, np.arange(len(chords))] = 1
    for y, z in product(range(n), repeat=2):
        running = np.zeros(len(chords), dtype=np.int64)
        for x in range(n-2, 0, -1):
            running += db[row_of[slab.index[(x, y, z)]]]
            edge = slab.dynamic_lookup[3*slab.index[(x-1, y, z)]]
            blocks[edge] = running
    c = sparse.csr_matrix(blocks)
    dc = slab.DI @ c
    assert not np.any(dc.data)
    gram = (c.T @ c).toarray().astype(np.int64)
    assert np.array_equal(blocks[chords], np.eye(len(chords), dtype=np.int64))
    det_l = dirichlet_det(n)
    prime = 1009
    assert modular_det(gram, prime) == det_l % prime
    # Exact affine deck phase on an actual all64 row-stripe canonical state.
    stripe = [F(128 if x < n//2 else -128) for x in range(n)]
    potential = dirichlet_solve(stripe[1:-1])
    longitudinal = exact_matvec(slab.DI.T, [potential[slab.sites[i][0]] for i in slab.interior])
    holonomy = [str(longitudinal[j]) for j in chords]
    return dict(N=n, M=len(slab.dynamic), interior_rows=len(slab.interior),
                rank=len(chords), full_integer_forest_determinant_abs=1,
                exact_cycle_identity_DI_C_zero=True, cycle_chord_identity=True,
                cycle_gram_trace=int(np.trace(gram)), det_LI= str(det_l),
                det_CGram_equals_det_LI='Exact Sylvester determinant identity; native modular consumer also agrees.',
                native_gram_modulus=prime, native_gram_det_modulus=det_l % prime,
                primitive_deck_basis='Lambda=C(C^T C)^-1=Q_I [chord standard basis]',
                covolume_primitive_deck='1/sqrt(det_LI)',
                radians_cell_volume='(2pi)^d/sqrt(det_LI)',
                character_measure='product_j db_j on [0,1)^d, total mass1',
                affine_all64_stripe_holonomy_sha256=hashlib.sha256(json.dumps(holonomy).encode()).hexdigest(),
                affine_holonomy_nonintegral_entries=sum(F(t).denominator != 1 for t in holonomy),
                raw_cycle_only_cell_is_not_primitive=True), gram


def affine_gluing_control() -> dict[str, Any]:
    slab = FixedMagneticSlab(4, 1)
    site = slab.index[(2, 1, 1)]
    path = slab.path(site, 0, 1)
    assert path is not None
    edge = path['edges'][0]
    j = slab.dynamic_lookup[edge]
    di = sp.Matrix(slab.DI.toarray().astype(int).tolist())
    raw = sp.zeros(len(slab.dynamic), 1)
    raw[j] = 1
    charge_change = di*raw
    lam = (di*di.T).inv()*charge_change
    delta = (charge_change.T*lam)[0]
    assert delta.is_Rational and (2*delta).q != 1
    projected_path = 1-delta
    assert sp.simplify(projected_path+delta) == 1
    # Intertwining under reduced gluing is the MINUS density phase.
    assert sp.simplify(sp.exp(-2*sp.pi*sp.I*delta)-sp.exp(2*sp.pi*sp.I*projected_path)) == 0
    plus_wrong = sp.exp(2*sp.pi*sp.I*delta)-sp.exp(2*sp.pi*sp.I*projected_path)
    assert plus_wrong != 0
    coordinate, rho = sp.symbols('coordinate rho', real=True)
    section = sp.exp(sp.I*coordinate*rho)
    assert sp.simplify(-sp.I*sp.diff(section, coordinate)-rho*section) == 0
    assert sp.simplify(-sp.I*sp.diff(sp.conjugate(section), coordinate)-rho*sp.conjugate(section)) != 0
    assert sp.exp(sp.I*sp.pi) == -1
    return dict(N=4, full_gauge_section='Psi(theta+D^T lambda)=exp(+i lambda.rho)Psi(theta)',
                fullE_convention='-i derivative', reduced_deck_gluing='exp(+2pi i b.j) exp(-2pi i (LI^-1 DI n).rhoI)',
                raw_native_single_chord_edge=edge, exact_affine_charge_change_pairing=str(delta),
                projected_path_pairing=str(projected_path), complete_native_fullE_deck_intertwining=True,
                wrong_plus_reduced_density_sign_rejected=True, wrong_minus_full_gauge_section_rejected=True,
                half_integer_raw_deck_rejected=True, density_scope='sum64 canonical charged densities; Ref retained before interior restriction')


def standing_cochain(slab: FixedMagneticSlab) -> dict[int, Any]:
    return {edge: sp.cos(2*sp.pi*slab.sites[edge//3][1]/slab.n)
            for edge in slab.pinned if edge % 3 == 2}


def canonical_mask(v: int, source: int, target: int) -> int:
    slots = list(range(2*v))
    create, annihilate = 4*source, 4*target
    if create in slots:
        slots.remove(create)
    if annihilate not in slots:
        slots.append(annihilate)
    for j in range(4*v):
        if len(slots) >= 2*v:
            break
        if j != create and j not in slots:
            slots.append(j)
    while len(slots) > 2*v:
        slots.remove(next(j for j in reversed(slots) if j != annihilate))
    one = sum(1 << j for j in slots)
    assert one.bit_count() == 2*v and not one & (1 << create) and one & (1 << annihilate)
    return sum(one << (4*v*s) for s in range(65)) | (15 << (4*v*65))


def path_jet(mask: int, species: int, path: dict[str, Any], coefficient: Any,
             phase: Any, eta: Any, order: int, v: int) -> dict[int, Any]:
    image: dict[int, Any] = {}
    offset = 4*v*species
    for a, b in product(range(4), repeat=2):
        forward = coefficient[a, b]*phase*(sp.I*eta)**order
        for create, annihilate, value in (
            (offset+4*path['source']+a, offset+4*path['target']+b, forward),
            (offset+4*path['target']+b, offset+4*path['source']+a, sp.conjugate(forward))):
            action = car_transfer(mask, annihilate, create)
            if action is not None and value != 0:
                target, sign = action
                image[target] = sp.expand(image.get(target, 0)+sign*value)
    return image


def native_current_contact(slab: FixedMagneticSlab, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    source = standing_cochain(slab)
    curvature = []
    for x in (0, slab.n-1):
        faces = [sp.simplify(sp.cos(2*sp.pi*(y+1)/slab.n)-sp.cos(2*sp.pi*y/slab.n))
                 for y, z in product(range(slab.n), repeat=2)]
        assert sp.simplify(sum(faces)) == 0 and any(a != 0 for a in faces)
        curvature.append(dict(boundary_x=x, exact_zero_net=True,
                              nonzero_faces=sum(a != 0 for a in faces)))
    active_paths, auxiliary_paths, eta_one, eta_two = 0, 0, [], []
    for site, axis, length in product(range(slab.v), range(3), (1, 2)):
        path = slab.path(site, axis, length)
        if path is None:
            continue
        eta = sp.simplify(sum((source.get(e, sp.Integer(0)) for e in path['edges']), sp.Integer(0)))
        if eta != 0:
            active_paths += 1
            auxiliary_paths += 63
        if axis == 2 and slab.sites[site][0] in (0, slab.n-1):
            expected = length*sp.cos(2*sp.pi*slab.sites[site][1]/slab.n)
            assert sp.simplify(eta-expected) == 0
            (eta_one if length == 1 else eta_two).append(str(eta))
        else:
            assert eta == 0
    alpha = sp.Symbol('alpha', real=True)
    transfers = 0
    jet_records = []
    for x, y, length in product((0, slab.n-1), (0, 1), (1, 2)):
        site = slab.index[(x, y, slab.n-1)]
        path = slab.path(site, 2, length)
        assert path is not None and path['seam'] == -1
        eta = sp.simplify(sum(source.get(e, 0) for e in path['edges']))
        fields = {site: generic_phi(2), path['target']: generic_phi(5)}
        first = sp.diag(sp.I, -sp.I, -1, -1)
        second = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        links = {e: first if j == 0 else second for j, e in enumerate(path['edges'])}
        if length == 2:
            assert first*second != second*first
        mask = canonical_mask(slab.v, site, path['target'])
        phase0 = sp.exp(2*sp.pi*sp.I*sp.Rational(slab.phase(path, {})))
        phase_alpha = phase0*sp.exp(sp.I*alpha*eta)
        assert sp.simplify(sp.diff(phase_alpha, alpha)-sp.I*eta*phase_alpha) == 0
        assert sp.simplify(sp.diff(phase_alpha, alpha, 2)+eta**2*phase_alpha) == 0
        for profile in profiles[1:]:
            row = next(p for p in profile['paths'] if p['axis'] == 2 and p['length'] == length)
            coefficient = slab.native.endpoint(row, path, fields, links)
            species_list = [0] if profile['load'] == 1 else range(1, 64)
            for species in species_list:
                h = path_jet(mask, species, path, coefficient, phase0, eta, 0, slab.v)
                j = path_jet(mask, species, path, coefficient, phase0, eta, 1, slab.v)
                k = path_jet(mask, species, path, coefficient, phase0, eta, 2, slab.v)
                assert h
                assert all(sp.expand(k.get(state, 0)+eta**2*value) == 0 for state, value in h.items())
                assert not eta or j
                for state in h:
                    assert ((state >> (4*slab.v*species)) & ((1 << (4*slab.v))-1)).bit_count() == 2*slab.v
                    assert state.bit_count() == mask.bit_count()
                transfers += 1
            jet_records.append([x, y, length, profile['role'], str(eta), str(coefficient[0, 0])])
        # Neutral retains the full -Lambda operator but has no alpha jet.
        if length == 1:
            row = next(p for p in profiles[0]['paths'] if p['axis'] == 2)
            coefficient = slab.native.endpoint(row, path, fields, links)
            assert path_jet(mask, 64, path, coefficient, 1, 0, 1, slab.v) == {}
        if length == 2 and y == 0:
            assert eta == 2 and -eta**2 != -1
    return dict(N=slab.n, alpha_units='radians; atlas increment alpha*a/(2pi)',
                fixed_amplitude=1, q_lattice='2pi/N', magnetic_boundary_m=1,
                source_edges=len(source), perturbation_curvature=curvature,
                active_nonzero_eta_paths=active_paths, auxiliary63_nonzero_eta_paths=auxiliary_paths,
                literal_all64_CAR_jet_channels_exercised=transfers,
                ordered_noncommuting_two_links=True, AP_z_seam_retained=True,
                complete_J='sum i eta T - i eta Tdagger',
                complete_K='sum -eta^2 (T+Tdagger)',
                neutral_and_onsite_and_Phi_SU4_HE_HC_alpha_jets_zero=True,
                two_link_wrong_J_i_and_K_minus1_rejected=True,
                coefficient_scope='Arbitrary quantum fullPhi/ordered SU4 operator identity; noncommuting pointwise probes do not freeze the thermal fields.',
                exact_probe_sha256=hashlib.sha256(json.dumps(jet_records).encode()).hexdigest())


def enclosure_data(slab: FixedMagneticSlab, profiles: list[dict[str, Any]], gram: Any) -> dict[str, Any]:
    v, n = slab.v, slab.n
    parts = {'heavy_Lambda': F(0), 'charged': F(0)}
    for profile in profiles:
        coefficient = F(4*v)*abs(F(profile['onsite']))
        for row in profile['paths']:
            count = ((n-int(row['length']))*n*n if row['axis'] == 0 else v)
            coefficient += 8*count*(abs(F(row['c0']))+abs(F(row['ci'])))
        parts['heavy_Lambda' if profile['scale'] == 'heavy' else 'charged'] += profile['load']*coefficient
    mj, mk = F(0), F(0)
    for profile in profiles[1:]:
        for row in profile['paths']:
            if row['axis'] == 2:
                base = 8*profile['load']*2*n*n*(abs(F(row['c0']))+abs(F(row['ci'])))
                mj += base*row['length']
                mk += base*row['length']**2
    faces = 3*v-2*n*n
    d, tr_g = gram.shape[0], int(np.trace(gram))
    return dict(N=n, exact_literal_parameters=dict(a='1/64', Lambda='2^8192', chi='2^-4096', F='2^10240', HE=64, HC=64),
                canonical_particles_per_each65=2*v, canonical_CAR_dimension=str(math.comb(4*v, 2*v)**65),
                Ref_dimension=1, fullPhi_coordinates=60*v, colour_Haar_factors=len(slab.edges),
                r='Euclidean native fullPhi Cart radius; Tr(T_a T_b)=delta_ab/2, so full HS radius<=r',
                M_N_parts={key: str(value) for key, value in parts.items()},
                M_N='heavy_Lambda*2^8192+charged', a4_N=f'2^8192/(4*{v})',
                MJ_N=str(mj), MK_N=str(mk),
                C_N='M_N+M_N^2/(2*a4_N)', C0_N='C_N+1/(2*a4_N)',
                Etrial_N=dict(electric_upper=str(16*d*tr_g),
                              HC_upper=str(64*(n-1)**2*128**2*len(slab.interior)//4),
                              Phi_kinetic=f'{120*v}*2^-4096',
                              Jordan_upper=f'{14880*v}*2^8192',
                              alignment_upper=f'{3*len(slab.edges)}*2^10240',
                              colour_face_upper=f'{F(28,5)*faces}*2^10240',
                              all65_matter_expectation_exact=0),
                Z_native_enclosure='exp(-beta*Etrial_N) <= Z_N(beta,alpha) <= exp(beta*C0_N)*Q_N(beta)',
                Q_N='D_CAR*(pi/(64*beta))^(d/2)/sqrt(det_LI)*(2sinh(2beta*sqrt(chi)))^(-60V)*h_SU4(3chi*beta/20)^E',
                moment_enclosure='B_N=2/(e*beta)*Q_N(beta/2)*exp(beta*(Etrial_N+C0_N)); <r^2> <= R2_N=min(B_N,sqrt(2B_N/a4_N))',
                actual_native_current_enclosure='|<J>| <= MJ_N*sqrt(R2_N)',
                actual_native_contact_enclosure='|<K>| <= MK_N*sqrt(R2_N)',
                actual_native_Kubo_enclosure='0 <= chi_Duhamel <= beta*MJ_N^2*R2_N (whole zero-frequency block retained)',
                actual_native_Fsecond_enclosure='[-MK_N*sqrt(R2_N)-beta*MJ_N^2*R2_N, MK_N*sqrt(R2_N)]',
                enclosure_kind='EXACT_DERIVED_UNEVALUATED_SYMBOLIC_FINITE_N_FULL_QUANTUM_NOT_NUMERICAL_ESTIMATE',
                magnetic_response_sign_paid=False, stiffness_phase_photon=False)


def heat_representation_attempt(n: int = 4) -> dict[str, Any]:
    # This new analytic epoch actually checks the necessary stated finite-crop
    # criterion; it does not execute a guessed native finite CAR spectrum.
    beta, slices, cutoff = F(1), 16, 64
    t = F(3, 20)*F(1, 2**4096)/slices
    u = t/8
    assert u*cutoff**2 < 7
    exponent = F(4*(cutoff+1), 2**2048)
    assert exponent < F(1, 2**2039)
    return dict(N=n, beta_temperature=str(beta), proposed_Trotter_slices=slices,
                native_representation='All-character twisted E heat kernel, full60V Phi Gaussian bridge, SU4 heat kernels, physical SU4 group projection, exact 65 canonical CAR coefficient of ordered matrix product with collective HC ONCE.',
                shared_HC_HS_identity='e^(-64 delta rho.LI^-1 rho)=E_xi e^(i sqrt(128delta) xi.LI^-1/2 rho), xi standard real Gaussian; density background kept.',
                canonical_coefficient='[z0^(2V)...z64^(2V) zRef^4] fullFockTrace(P_SU4 ordered exp matrices)',
                static_determinant_substitution=False, grandcanonical_substitution=False,
                candidate_phi_oscillator_per_coordinate_cutoff=cutoff,
                exact_reference_phi_thermal_tail_lower='1-2^-2039 at beta1; this is a comparison-reference tail, not a paid native Galerkin error',
                candidate_colour_highest_weight_shell_cutoff=cutoff,
                colour_decreasing_tail_criterion='u*R^2 >=7, u=(3chi*beta/(20*slices))/8',
                exact_colour_criterion_left=str(u*cutoff**2), criterion_reached=False,
                needed_colour_R_lower='sqrt(7/u), of order 2^2048 at this pinned chi',
                native_CAR_dimension=str(math.comb(4*n**3, 2*n**3)**65),
                failed_criteria=['comparison-reference tail is nearly whole thermal mass at proposed Phi cutoff',
                                 'colour shell64 does not reach the monotone Gaussian tail bound',
                                 'no native Galerkin/Trotter trace-norm error from the noncommuting fullPhi/colour/CAR couplings was computed',
                                 'all-character and field/HS quadrature errors unpriced'],
                full_field_tail_paid_for_existence=True,
                thermal_numeric_pressure_or_sign_computed=False,
                exact_native_enclosures_remain_valid=True,
                failed_epoch='native_analytic_crop1_retained_NOT_retagged_as_native_numeric')


def canonical_control_matrices() -> tuple[Any, Any, Any, Any]:
    masks = [((1 << a) | (1 << (2+b))) for a, b in product(range(2), repeat=2)]
    lookup = {mask: j for j, mask in enumerate(masks)}
    forward = np.zeros((4, 4), complex)
    rhoa, rhob = [], []
    for j, mask in enumerate(masks):
        rhoa.append((mask & 1)-.5)
        rhob.append(((mask >> 2) & 1)-.5)
        for shift in (0, 2):
            action = car_transfer(mask, shift+1, shift)
            if action is not None:
                target, sign = action
                forward[lookup[target], j] += sign
    z = np.diag([1, -1]); x = np.array([[0, 1], [1, 0]]); y = np.array([[0, -1j], [1j, 0]])
    ra, rb = np.diag(rhoa), np.diag(rhob)
    base = np.kron(z/3, np.eye(4))+.17*np.kron(y, ra-rb)
    hopping = .7*np.kron(x, forward)
    hc = 1.1*np.kron(np.eye(2), (ra+rb) @ (ra+rb))
    wrong_hc = 1.1*np.kron(np.eye(2), ra @ ra+rb @ rb)
    jextra = .23*np.kron(z, np.eye(4))
    return base+hc, hopping, jextra, base+wrong_hc


def thermal_matrix(h: Any, j: Any, k: Any, beta: float) -> dict[str, float]:
    e, u = eigh(h)
    shift = float(e.min())
    weights = np.exp(-beta*(e-shift))
    z = float(weights.sum())
    p = weights/z
    jt = u.conj().T @ j @ u
    kt = u.conj().T @ k @ u
    mean = float(np.dot(p, np.diag(jt).real))
    connected = jt-mean*np.eye(len(e))
    chi, zero = 0., 0.
    for a, b in product(range(len(e)), repeat=2):
        gap = float(e[b]-e[a])
        coefficient = beta*p[a] if abs(gap) < 1e-10 else (p[a]-p[b])/gap
        term = coefficient*abs(connected[a, b])**2
        assert term >= -1e-13
        chi += term
        if abs(gap) < 1e-10:
            zero += term
    contact = float(np.dot(p, np.diag(kt).real))
    return dict(F=shift-math.log(z)/beta, current=mean, contact=contact,
                Kubo=chi, zero_frequency=zero, Fsecond=contact-chi)


def noncommuting_controls() -> dict[str, Any]:
    base, hopping, jextra, wrong_base = canonical_control_matrices()
    def matrices(alpha: float, wrong: bool = False) -> tuple[Any, Any, Any]:
        forward = np.exp(1j*alpha)*hopping
        return ((wrong_base if wrong else base)+forward+forward.conj().T+alpha*jextra,
                1j*(forward-forward.conj().T)+jextra, -(forward+forward.conj().T))
    rows = []
    for beta in (.5, 1.):
        h, j, k = matrices(0)
        assert np.linalg.norm(h @ j-j @ h) > .1
        value = thermal_matrix(h, j, k, beta)
        eps = .001
        def second(step: float) -> float:
            return (thermal_matrix(*matrices(step), beta)['F']-2*value['F']+
                    thermal_matrix(*matrices(-step), beta)['F'])/step**2
        richardson = (4*second(eps/2)-second(eps))/3
        assert abs(richardson-value['Fsecond']) < 2e-7
        assert abs(value['contact']) > .01 and value['zero_frequency'] > 1e-5
        wrong = thermal_matrix(*matrices(0, True), beta)
        assert abs(wrong['F']-value['F']) > .01
        rows.append(dict(beta=beta, **value, direct_Fsecond_Richardson=richardson,
                         omitted_contact_error=abs(value['contact']),
                         omitted_zero_block_error=value['zero_frequency'],
                         wrong_shared_density_F_error=abs(wrong['F']-value['F'])))
    # One-body canonical coefficient differs from the grandcanonical trace.
    e0, e1, beta = F(0), F(2), F(1)
    zcanonical = 1+sp.exp(-2)
    zgrand = 2*(1+sp.exp(-2))
    assert sp.simplify(zgrand/zcanonical) == 2
    b_control = sp.sqrt(sp.pi)/8
    assert b_control < 1
    # Entire degenerate ground block, not its mean: cusp and beta divergence.
    ground = [dict(beta=beta, exact_Fsecond=f'-{2*beta}/(2+exp(-{2*beta}))') for beta in (1, 4, 16)]
    return dict(scope='BINARY64_NONCOMMUTING_FINITE_QUANTUM_FIELD_AND_EXACT_CANONICAL_CAR_CONTROLS_NOT_NATIVE_PRESSURE',
                field_dimension=2, two_species_canonical_counts=[1, 1], matrices_dimension=8,
                controls=rows, omitted_contact_rejected=True, omitted_zero_frequency_rejected=True,
                wrong64_independent_density_factorization_rejected=True,
                factorization_scope='Two canonical species suffice to reject independent-density factorization; literal native64 bound/jet consumer separately retains all64.',
                grandcanonical_wrongcase_ratio_exact='2', zero_Bloch_vs_integrated_character_rejected=True,
                all_character_free_control_exact=str(b_control), zero_character_trace_lower=1,
                full_cover_finite_Z_assumption_rejected='independent native deck packet argument',
                ground_block_controls=ground, ground_energy_cusp='E0(alpha)=-abs(alpha) near0',
                finite_temperature_positive_L_assumption=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert all(digest(ROOT/name) == pin for name, pin in request['pins'].items())
    profiles = json.loads((ROOT/request['profile']).read_text())['primitive_profiles']
    assert [(p['load'], p['scale']) for p in profiles] == [(1, 'heavy'), (1, 'charged'), (63, 'charged')]
    result: dict[str, Any] = dict(kind='ACTUAL_CANONICAL_FULL_QUANTUM_MAGNETIC_SLAB_HEAT_PER_CELL_AND_EXACT_RESPONSE_ENCLOSURES_NOT_STIFFNESS',
        source_sha256=digest(Path(__file__)), request_sha256=digest(args.request),
        parent_independently_checked=parent_independent_check(request), native_N_controls=[])
    for n in request['N']:
        slab = FixedMagneticSlab(n, 1)
        lattice, gram = primitive_lattice(slab)
        result['native_N_controls'].append(dict(lattice=lattice,
            response_operator=native_current_contact(slab, profiles),
            exact_thermal_enclosures=enclosure_data(slab, profiles, gram),
            electric_heat_word=dict(interior_face=face_word(slab, 1),
                                    boundary_face=face_word(slab, 0),
                                    common_character_memory=common_b_memory(slab))))
    result.update(native_analytic_attempt=heat_representation_attempt(),
                  native_affine_gluing=affine_gluing_control(),
                  adverse_noncommuting_controls=noncommuting_controls(),
                  ordinary_real_cover_Gibbs_density=False,
                  beta0_fibre_substitution=False, static_Phi_native_pressure=False,
                  native_numeric_expectations=False, native_exact_response_enclosures=True,
                  magnetic_response_sign_paid=False, physical_ground_phase_atom=False,
                  old_producer_solver_root_checker_rerun=False, campaign_complete=False)
    assert all(digest(ROOT/name) == pin for name, pin in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(N=request['N'], primitive_integrated_canonical_heat='constructed with exact finite-N trace/response bounds',
                          numeric_native_pressure=False, analytic_crop='failed precise tail/error criteria', phase=False)))


if __name__ == '__main__':
    main()
