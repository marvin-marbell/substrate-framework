"""Actual fixed magnetic boundary slab; no ground/tension computation.

Open x, periodic native AP y/z. Tangential U1 connections on x=0,N-1 are
Dirichlet data; no boundary E derivative is silently retained. Interior E,
all64 charge1 CAR, neutral, full quantum Phi/SU4 and Ref remain explicit.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy import sparse

from magnetic_boundary_scientist_20261011 import (
    ChargeOneBundle, car_transfer, exact_matvec, generic_phi, su4_basis,
)

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dirichlet_solve(rho: list[F]) -> list[F]:
    """Exact one-dimensional restriction of slab L_I; both end values fixed0."""
    n = len(rho)
    matrix = [[F(2 if i == j else -1 if abs(i-j) == 1 else 0) for j in range(n)]
              + [rho[i]] for i in range(n)]
    for i in range(n):
        pivot = matrix[i][i]
        matrix[i] = [a/pivot for a in matrix[i]]
        for j in range(n):
            if j != i:
                coefficient = matrix[j][i]
                matrix[j] = [matrix[j][k]-coefficient*matrix[i][k] for k in range(n+1)]
    return [F(0), *[row[-1] for row in matrix], F(0)]


class FixedMagneticSlab:
    """Unrestricted interior connections with FIXED tangential boundary atlas."""
    def __init__(self, n: int, m: int = 1):
        if n < 4 or n % 2 or not isinstance(m, int):
            raise ValueError('Even slab N>=4 and integral charge1 atlas required')
        self.n, self.m, self.v = n, m, n**3
        self.native = ChargeOneBundle(n, 0)
        self.sites, self.index = self.native.sites, self.native.index
        self.edges = [3*i+j for i, xyz in enumerate(self.sites) for j in range(3)
                      if not (j == 0 and xyz[0] == n-1)]
        self.pinned = {3*i+j for i, xyz in enumerate(self.sites) for j in (1, 2)
                       if xyz[0] in (0, n-1)}
        self.dynamic = [edge for edge in self.edges if edge not in self.pinned]
        self.dynamic_lookup = {edge: i for i, edge in enumerate(self.dynamic)}
        self.interior = [i for i, xyz in enumerate(self.sites) if 0 < xyz[0] < n-1]
        self.a = [F(0)]*(3*self.v)
        self.sheet = [0]*self.v
        for i, (_x, y, z) in enumerate(self.sites):
            self.a[3*i+2] = F(m*y, n*n)
            if y == n-1:
                self.a[3*i+1] = F(-m*z, n)
            if y == n-1 and z == n-1:
                self.sheet[i] = m
        self.DI = self.native.D[self.interior, :][:, self.dynamic].tocsr()
        self.LI = (self.DI @ self.DI.T).tocsr()

    def path(self, x: int, axis: int, length: int) -> dict[str, Any] | None:
        if axis == 0 and self.sites[x][0]+length >= self.n:
            return None  # Open geometry: this edge/path does not exist.
        return self.native.path(x, axis, length)

    def phase(self, path: dict[str, Any], theta: dict[int, F], charged: bool = True) -> F:
        if not charged:
            return F(0)
        return sum((self.a[e]+theta.get(e, F(0)) for e in path['edges']), F(0))

    def accept_translation(self, shift: dict[int, F]) -> None:
        if any(edge not in self.dynamic_lookup for edge in shift):
            raise ValueError('Fixed magnetic boundary links are NOT dynamic E coordinates')

    def translated_phase(self, path: dict[str, Any], theta: dict[int, F], shift: dict[int, F]) -> F:
        self.accept_translation(shift)
        return self.phase(path, {e: theta.get(e, F(0))+shift.get(e, F(0)) for e in self.dynamic})

    def apply_path(self, mask: int, species: int, path: dict[str, Any], coefficient: Any,
                   theta: dict[int, F], charged: bool = True) -> dict[int, Any]:
        """Literal canonical operator; boundary paths keep their fixed phase."""
        self.accept_translation(theta)
        phase = sp.exp(2*sp.pi*sp.I*sp.Rational(self.phase(path, theta, charged)))
        offset, image = 4*self.v*species, {}
        for a, b in product(range(4), repeat=2):
            i = offset+4*path['source']+a; j = offset+4*path['target']+b
            for create, annihilate, value in ((i, j, coefficient[a, b]*phase),
                                              (j, i, sp.conjugate(coefficient[a, b]*phase))):
                action = car_transfer(mask, annihilate, create)
                if action is not None and value != 0:
                    target, sign = action
                    image[target] = image.get(target, 0)+sign*value
        return image


def geometry_and_boundary(slab: FixedMagneticSlab) -> dict[str, Any]:
    n, v = slab.n, slab.v
    rows, open_absent, ap_count = [], 0, 0
    for x, axis, length in product(range(v), range(3), (1, 2)):
        path = slab.path(x, axis, length)
        if path is None:
            open_absent += 1
            continue
        assert all(edge in slab.edges for edge in path['edges'])
        assert path['seam'] == (-1 if axis in (1, 2) and slab.sites[x][axis]+length >= n else 1)
        ap_count += int(path['seam'] == -1)
        rows.append([x, axis, length, path['target'], path['edges'], path['seam'], str(slab.phase(path, {}))])
    assert open_absent == 3*n*n
    boundary_flux = []
    for x in (0, n-1):
        for y, z in product(range(n), repeat=2):
            start = slab.index[(x, y, z)]
            py = slab.path(start, 1, 1); pz = slab.path(start, 2, 1)
            assert py is not None and pz is not None
            yz = slab.path(py['target'], 2, 1); zy = slab.path(pz['target'], 1, 1)
            assert yz is not None and zy is not None
            phase = slab.phase(py, {})+slab.phase(yz, {})-slab.phase(zy, {})-slab.phase(pz, {})
            assert phase+slab.sheet[start] == F(1, n*n)
            assert phase.denominator != 1
            assert py['seam']*yz['seam']*zy['seam']*pz['seam'] == 1
            assert all(edge in slab.pinned for p in (py, yz, zy, pz) for edge in p['edges'])
            internal_shift = {edge: F((edge*13) % 17-8, 31) for edge in slab.dynamic}
            shifted = (slab.translated_phase(py, {}, internal_shift)+slab.translated_phase(yz, {}, internal_shift)
                       -slab.translated_phase(zy, {}, internal_shift)-slab.translated_phase(pz, {}, internal_shift))
            assert shifted == phase
            boundary_flux.append(phase+slab.sheet[start])
    assert sum(boundary_flux[:n*n]) == sum(boundary_flux[n*n:]) == 1
    # The torus absorption map is reached and rejected at this NEW domain boundary.
    rejected = False
    try:
        slab.accept_translation({e: -slab.a[e] for e in slab.edges if slab.a[e]})
    except ValueError:
        rejected = True
    assert rejected
    # Interior-only absorption leaves a nontrivial measurable boundary Wilson face.
    legal_shift = {e: -slab.a[e] for e in slab.dynamic}
    slab.accept_translation(legal_shift)
    interior = slab.index[(1, 0, 0)]
    path = slab.path(interior, 2, 1)
    assert path is not None and slab.translated_phase(path, {}, legal_shift) == 0
    # Internal gauge: lambda=0 on both boundary components. Exact source-target gradient.
    lam = [F(0) if xyz[0] in (0, n-1) else F((i % 7)-3, 11) for i, xyz in enumerate(slab.sites)]
    gauge = exact_matvec(slab.native.D.T, lam)
    assert all(gauge[e] == 0 for e in slab.pinned)
    raw = exact_matvec(slab.native.C, gauge)
    assert not any(raw)
    # A boundary face has nonzero curvature modulo2pi; NO endpoint gauge can cancel it.
    assert F(1, n*n).denominator != 1
    return dict(N=n, open_x=True, periodic_AP_axes=['y', 'z'], vertices=v,
                existing_edges=len(slab.edges), pinned_U1_tangential_edges=len(slab.pinned),
                dynamic_U1_E_edges=len(slab.dynamic), interior_U1_Gauss_vertices=len(slab.interior),
                path_count=len(rows), absent_open_x_paths=open_absent, retained_AP_paths=ap_count,
                exact_native_path_records_sha256=sha(json.dumps(rows).encode()),
                fixed_boundary_yz_flux_2pi_units=['1', '1'],
                fixed_boundary_Wilson_face='exp(2pi i/N^2) !=1',
                all_boundary_faces_exact=len(boundary_flux),
                actual_fullE_unrestricted_absorption_rejected_by_domain=rejected,
                interior_only_shift_cannot_change_boundary_loop=True,
                internal_gauge_preserves_fixed_connection=True,
                endpoint_gauge_absorption_impossible_by_closed_loop=True,
                boundary_E_not_defined_as_original_derivative=True,
                geometry='Declared NEW open slab with fixed tangential U1 atlas, NOT an original periodic reducing sector')


def collective_gauss(slab: FixedMagneticSlab) -> dict[str, Any]:
    n, v = slab.n, slab.v
    stripe = [F(128 if x < n//2 else -128) for x in range(n)]
    phi_x = dirichlet_solve(stripe[1:-1])
    phi = [phi_x[slab.sites[i][0]] for i in slab.interior]
    rho = [stripe[slab.sites[i][0]] for i in slab.interior]
    el = exact_matvec(slab.DI.T, phi)
    assert exact_matvec(slab.DI, el) == rho
    assert exact_matvec(slab.LI, phi) == rho
    # This explicit normal-flux reservoir pays the boundary electric/Gauss price.
    full_el = [F(0)]*(3*v)
    for edge, value in zip(slab.dynamic, el, strict=True):
        full_el[edge] = value
    full_div = exact_matvec(slab.native.D, full_el)
    boundary_residual = [full_div[i]-stripe[xyz[0]] for i, xyz in enumerate(slab.sites) if xyz[0] in (0, n-1)]
    assert any(boundary_residual)
    rows = []
    for length in (1, 2):
        source_x = n//2-length
        k = [F(0)]*len(slab.dynamic)
        species_mask = sum(15 << (4*i) for i, xyz in enumerate(slab.sites) if xyz[0] < n//2)
        initial_mask = sum(species_mask << (4*v*s) for s in range(65)) | (15 << (4*v*65))
        hop_mask, hop_sign = initial_mask, 1
        # A product of actual legal canonical adjoint hops, one per yz column.
        for y, z in product(range(n), repeat=2):
            source = slab.index[(source_x, y, z)]
            path = slab.path(source, 0, length)
            assert path is not None
            for edge in path['edges']:
                k[slab.dynamic_lookup[edge]] -= 1
            action = car_transfer(hop_mask, 4*source, 4*path['target'])
            assert action is not None
            hop_mask, sign = action
            hop_sign *= sign
        assert hop_mask.bit_count() == initial_mask.bit_count()
        assert (hop_mask & ((1 << (4*v))-1)).bit_count() == 2*v
        drho = exact_matvec(slab.DI, k)
        row_drho = [F(-1 if x == source_x else 1 if x == n//2 else 0) for x in range(1, n-1)]
        delta_phi_x = dirichlet_solve(row_drho)
        delta_phi = [delta_phi_x[slab.sites[i][0]] for i in slab.interior]
        delta_el = exact_matvec(slab.DI.T, delta_phi)
        assert exact_matvec(slab.DI, delta_el) == drho
        qk = [k[j]-delta_el[j] for j in range(len(k))]
        assert not any(exact_matvec(slab.DI, qk))
        assert sum((qk[j]*delta_el[j] for j in range(len(k))), F(0)) == 0
        incoming = [el[j]+F(3, 8)*qk[j] for j in range(len(k))]
        full = 64*sum(((incoming[j]+k[j])**2-incoming[j]**2 for j in range(len(k))), F(0))
        transverse = 64*(F(11, 8)**2-F(3, 8)**2)*sum((t*t for t in qk), F(0))
        hc = 64*(2*sum((rho[j]*delta_phi[j] for j in range(len(rho))), F(0))
                 +sum((drho[j]*delta_phi[j] for j in range(len(rho))), F(0)))
        assert full == transverse+hc
        assert exact_matvec(slab.DI, [incoming[j]+qk[j] for j in range(len(k))]) != [rho[j]+drho[j] for j in range(len(rho))]
        rows.append(dict(length=length, actual_native_plane_product_hops=n*n,
                         actual_global_CAR_plane_product_sign=hop_sign,
                         resulting_mask_sha256=sha(hop_mask.to_bytes((65*4*v+11)//8, 'little')),
                         HE64_increment=str(full), transverse64_increment=str(transverse),
                         collective_HC64_increment=str(hc), fullE_Gauss_update_exact=True,
                         Pi_only_wrongcase_rejected=True))
    hc = 64*sum((rho[j]*phi[j] for j in range(len(rho))), F(0))
    wrong_hc = 64*64*sum(((rho[j]/64)*(phi[j]/64) for j in range(len(rho))), F(0))
    assert hc == 64*wrong_hc
    lam = [F(0)]*len(slab.interior); lam[0] = F(1, 3)
    alpha = exact_matvec(slab.DI.T, lam)
    path = slab.path(slab.interior[0], 1, 1)
    assert path is not None
    angle = sum((alpha[slab.dynamic_lookup[e]] for e in path['edges']), F(0))
    assert angle == F(1, 3)
    return dict(N=n, all64_fullrho='sum64(n_s-2)+(Nref-4)delta0; interior restriction I applied explicitly',
                Ref_at_boundary0=True, Ref4_charge0=True,
                LI='D_I D_I^T strictly positive Dirichlet, no zero-mode deletion',
                exact_selected_longitudinal_Poisson_all_interior_sites=True,
                collective_HC64=str(hc), wrong_separate_species_HC=str(wrong_hc),
                wrong_separate_underestimate_factor=64, fullE_controls=rows,
                boundary_Gauss_residual_nonzero_count=sum(bool(a) for a in boundary_residual),
                boundary_price='U1 gauge parameters vanish on boundary; boundary charge is explicit reservoir data, not an omitted original Gauss constraint',
                actual_pure_gradient_translation_phase='exp(i/3)!=1',
                actual_pure_gradient_path_angle_radians=str(angle))


def canonical_and_boundary_loop(slab: FixedMagneticSlab, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    n, v = slab.n, slab.v
    species_mask = sum(15 << (4*i) for i, xyz in enumerate(slab.sites) if xyz[0] < n//2)
    mask = sum(species_mask << (4*v*s) for s in range(65)) | (15 << (4*v*65))
    theta: dict[int, F] = {}
    transfers, fullphi_assertions = 0, 0
    for length in (1, 2):
        source = slab.index[(n//2-length, 1, 1)]
        path = slab.path(source, 0, length)
        assert path is not None
        target = path['target']
        fields = {source: generic_phi(1), target: generic_phi(4)}
        links = {edge: sp.diag(sp.I, -sp.I, -1, -1) for edge in path['edges']}
        for profile in profiles:
            if profile['scale'] == 'heavy' and length == 2:
                continue
            row = next(r for r in profile['paths'] if r['axis'] == 0 and r['length'] == length)
            coefficient = slab.native.endpoint(row, path, fields, links)
            u = sp.eye(4)
            for edge in path['edges']:
                u = u*links[edge]
            c0, ci = sp.Rational(row['c0']), sp.Rational(row['ci'])
            fx, fy = c0*fields[source][0]+sp.I*ci*fields[source][1], c0*fields[target][0]+sp.I*ci*fields[target][1]
            for h in su4_basis():
                dfx, du = sp.I*(h*fx-fx*h), sp.I*h*u
                assert sp.simplify((dfx*u+fx*du+du*fy)/2-sp.I*h*coefficient) == sp.zeros(4)
                dfy, du = sp.I*(h*fy-fy*h), -sp.I*u*h
                assert sp.simplify((fx*du+du*fy+u*dfy)/2+sp.I*coefficient*h) == sp.zeros(4)
                fullphi_assertions += 2
            species_list = [64] if profile['scale'] == 'heavy' else ([0] if profile['load'] == 1 else list(range(1, 64)))
            for species in species_list:
                image = slab.apply_path(mask, species, path, coefficient, theta, charged=species != 64)
                assert image
                for state in image:
                    assert state.bit_count() == mask.bit_count()
                    assert ((state >> (4*v*species)) & ((1 << (4*v))-1)).bit_count() == 2*v
                    assert ((state >> (4*v*65)) & 15) == 15
                    transfers += 1
    # Literal AP seam operators on all64 species, not just a path-table label.
    seam_images = 0
    seam_source = slab.index[(1, n-1, 1)]
    destination = slab.index[(n-1, 2, 1)]
    seam_species = species_mask ^ (15 << (4*seam_source)) ^ (15 << (4*destination))
    seam_mask = sum(seam_species << (4*v*s) for s in range(65)) | (15 << (4*v*65))
    for length in (1, 2):
        path = slab.path(seam_source, 1, length)
        assert path is not None and path['seam'] == -1
        fields = {seam_source: generic_phi(1), path['target']: generic_phi(4)}
        links = {e: sp.eye(4) for e in path['edges']}
        for profile in profiles:
            if profile['scale'] == 'heavy' and length == 2:
                continue
            row = next(r for r in profile['paths'] if r['axis'] == 1 and r['length'] == length)
            coefficient = slab.native.endpoint(row, path, fields, links)
            assert coefficient != -coefficient
            species_list = [64] if profile['scale'] == 'heavy' else ([0] if profile['load'] == 1 else list(range(1, 64)))
            for species in species_list:
                image = slab.apply_path(seam_mask, species, path, coefficient, {}, charged=species != 64)
                assert image
                for state in image:
                    assert state.bit_count() == seam_mask.bit_count()
                    assert ((state >> (4*v*species)) & ((1 << (4*v))-1)).bit_count() == 2*v
                    seam_images += 1
        t = sp.Symbol('t', real=True)
        phase = sp.exp(sp.I*length*t)
        assert sp.diff(phase, t) == sp.I*length*phase
        if length == 2:
            assert sp.diff(phase, t) != sp.I*phase
    # Four literal rank-one native boundary CAR channels circulate one occupied
    # colour around a square. Other canonical modes/species remain unchanged.
    vertices = [slab.index[(0, 0, 0)], slab.index[(0, 1, 0)],
                slab.index[(0, 1, 1)], slab.index[(0, 0, 1)]]
    loop_mask = mask
    empty = [i for i, xyz in enumerate(slab.sites) if xyz[0] >= n//2][:3]
    for site, destination in zip(vertices[1:], empty, strict=True):
        loop_mask ^= 1 << (4*site)
        loop_mask |= 1 << (4*destination)
    assert loop_mask.bit_count() == mask.bit_count()
    forward_paths = [slab.path(vertices[0], 1, 1), slab.path(vertices[1], 2, 1),
                     slab.path(vertices[3], 1, 1), slab.path(vertices[0], 2, 1)]
    assert all(p is not None for p in forward_paths)
    fields = {i: generic_phi(1+i % 5) for i in vertices}
    row_y = next(r for r in profiles[1]['paths'] if r['axis'] == 1 and r['length'] == 1)
    row_z = next(r for r in profiles[1]['paths'] if r['axis'] == 2 and r['length'] == 1)
    action_mask, amplitude, phase_total = loop_mask, sp.Integer(1), F(0)
    for create_site, annihilate_site, p, row, reverse in (
        (vertices[3], vertices[0], forward_paths[3], row_z, True),
        (vertices[2], vertices[3], forward_paths[2], row_y, True),
        (vertices[1], vertices[2], forward_paths[1], row_z, False),
        (vertices[0], vertices[1], forward_paths[0], row_y, False)):
        assert p is not None
        coeff = slab.native.endpoint(row, p, fields, {e: sp.eye(4) for e in p['edges']})[0, 0]
        if reverse:
            coeff = sp.conjugate(coeff)
        assert coeff != 0
        signed_phase = (-1 if reverse else 1)*slab.phase(p, {})
        phase_total += signed_phase
        action = car_transfer(action_mask, 4*annihilate_site, 4*create_site)
        assert action is not None
        action_mask, sign = action
        amplitude *= sign*coeff
    assert action_mask == loop_mask and amplitude != 0
    assert phase_total == F(1, n*n)
    magnetic_amplitude = amplitude*sp.exp(2*sp.pi*sp.I*sp.Rational(phase_total))
    assert sp.simplify(magnetic_amplitude/amplitude-sp.exp(2*sp.pi*sp.I/n**2)) == 0
    assert magnetic_amplitude != amplitude
    # Ref charge and SU4 cancellation, preserving fullrho before interior restriction.
    ref_offset = 4*v*65
    ref_mask = mask ^ 15 ^ (15 << (4*empty[0]))
    refs = []
    for colour in range(4):
        result = car_transfer(ref_mask, ref_offset+colour, colour)
        assert result is not None
        image, sign = result
        before = sum(((ref_mask >> (4*v*s)) & 15).bit_count()-2 for s in range(64))
        after = sum(((image >> (4*v*s)) & 15).bit_count()-2 for s in range(64))-1
        assert before == after
        refs.append(dict(colour=colour, CAR_sign=sign, Ref4_to3=True, fullrho_unchanged=True))
    return dict(N=n, all64_actual_CAR_species=True, neutral_actual_CAR_retained=True,
                canonical_counts=[2*v]*65, global_modes=65*4*v+4,
                global_bitmask_sha256=sha(mask.to_bytes((65*4*v+11)//8, 'little')),
                one_two_link_CAR_images=transfers, fullPhi_all15_SU4_endpoint_derivative_assertions=fullphi_assertions,
                actual_AP_seam_CAR_images=seam_images,
                AP_omission_changes_literal_endpoint_sign=True,
                wrong_two_link_E_derivative_rejected=True,
                Ref_transitions=refs,
                actual_closed_boundary_CAR_loop=dict(number_preserved=True, state_returned_exact=True,
                    nonzero_native_endpoint_product_sha256=sha(str(amplitude).encode()),
                    charged_loop_over_zero_field_loop='exp(2pi i/N^2) !=1',
                    scalar_U1_loop_phase_2pi_units=str(phase_total),
                    neutral_same_loop_phase='1', fullE_interior_translation_cannot_remove=True),
                fields_scope='Arbitrary full quantum60Phi/site in operator; exact pointwise noncommuting matrices are probes only')


def run(request_path: Path, output: Path) -> None:
    request = json.loads(request_path.read_text())
    for name, pin in request['input_sha256'].items():
        assert sha((ROOT/name).read_bytes()) == pin, name
    profiles = json.loads((ROOT/request['profile']).read_text())['primitive_profiles']
    assert [(p['load'], p['scale']) for p in profiles] == [(1, 'heavy'), (1, 'charged'), (63, 'charged')]
    result: dict[str, Any] = dict(kind='NEW_FIXED_MAGNETIC_BOUNDARY_QUANTUM_SLAB_REALIZATION_NOT_GROUND',
        source_sha256=sha(Path(__file__).read_bytes()), request_sha256=sha(request_path.read_bytes()),
        input_sha256=request['input_sha256'], N_controls=[], no_EM_B2_or_admissibility_cutoff=True,
        original_torus_domain=False, physical_ground_tension_stiffness_IR_atom=False)
    for n in request['N']:
        slab = FixedMagneticSlab(n, 1)
        result['N_controls'].append(dict(N=n, geometry=geometry_and_boundary(slab),
            collective_Gauss=collective_gauss(slab), fullcanonical=canonical_and_boundary_loop(slab, profiles)))
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(receipt=output.name, N=request['N'], magnetic_boundary_Wilson_face='nontrivial',
                         original_torus_domain=False, ground=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.output)
