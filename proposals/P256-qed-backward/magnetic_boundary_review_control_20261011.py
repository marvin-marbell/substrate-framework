"""Independent exact NEW-boundary consumer; never imports historical producers."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any
import zipfile

import sympy as s

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def transfer(mask: int, annihilate: int, create: int) -> tuple[int, int] | None:
    if not (mask >> annihilate) & 1:
        return None
    sign = (-1) ** (mask & ((1 << annihilate)-1)).bit_count()
    mask ^= 1 << annihilate
    if (mask >> create) & 1:
        return None
    sign *= (-1) ** (mask & ((1 << create)-1)).bit_count()
    return mask | (1 << create), sign


def consume_boundary(n: int, edges: list[tuple[tuple[int, ...], int]],
                     pinned: set[int], shift: dict[int, F]) -> None:
    expected = {i for i, (xyz, axis) in enumerate(edges)
                if xyz[0] in (0, n-1) and axis in (1, 2)}
    require(pinned == expected, 'Pinned tangential boundary set is incomplete or changed')
    require(all(i in range(len(edges)) and i not in pinned for i in shift),
            'Translation changes absent or pinned boundary DOF')


def consume_density(species: list[s.Matrix], actual: s.Matrix) -> None:
    require(len(species) == 64, 'All64 charged densities required')
    require(actual == sum(species, s.zeros(actual.rows, 1)), 'Collective density omitted or misaggregated')


def must_reject(call: Any) -> str:
    try:
        call()
    except ValueError as error:
        return str(error)
    raise AssertionError('Adverse mutation reached accept boundary but was accepted')


def exact_control(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    n = 4
    sites = list(product(range(n), repeat=3))
    edges = [(xyz, axis) for xyz in sites for axis in range(3)
             if not (axis == 0 and xyz[0] == n-1)]
    lookup = {edge: i for i, edge in enumerate(edges)}
    pinned = {i for i, (xyz, axis) in enumerate(edges) if xyz[0] in (0, n-1) and axis in (1, 2)}
    dynamic = [i for i in range(len(edges)) if i not in pinned]
    interior = [xyz for xyz in sites if 0 < xyz[0] < n-1]
    site_index = {xyz: i for i, xyz in enumerate(sites)}
    def target(xyz: tuple[int, ...], axis: int) -> tuple[int, ...]:
        values = list(xyz); values[axis] = (values[axis]+1) % n
        return tuple(values)
    def atlas(xyz: tuple[int, ...], axis: int) -> F:
        return F(xyz[1], n*n) if axis == 2 else F(-xyz[2], n) if axis == 1 and xyz[1] == n-1 else F(0)
    background = {i: atlas(xyz, axis) for i, (xyz, axis) in enumerate(edges)}
    consume_boundary(n, edges, pinned, {})
    omitted = pinned-{min(pinned)}
    omitted_rejection = must_reject(lambda: consume_boundary(n, edges, omitted, {}))
    full_shift = {i: -a for i, a in background.items() if a}
    full_rejection = must_reject(lambda: consume_boundary(n, edges, pinned, full_shift))
    legal_shift = {i: -background[i] for i in dynamic}
    consume_boundary(n, edges, pinned, legal_shift)
    face_phases = []
    for x in (0, n-1):
        for y, z in product(range(n), repeat=2):
            a = (x, y, z); b = target(a, 1); d = target(a, 2)
            loop = [(lookup[(a, 1)], 1), (lookup[(b, 2)], 1),
                    (lookup[(d, 1)], -1), (lookup[(a, 2)], -1)]
            require(all(i in pinned for i, _ in loop), 'Boundary loop used dynamic DOF')
            phase = sum((background[i]*sign for i, sign in loop), F(0))
            require((phase-F(1, n*n)).denominator == 1, 'Boundary Wilson curvature changed')
            require(sum((legal_shift.get(i, F(0))*sign for i, sign in loop), F(0)) == 0,
                    'Legal translation changed boundary Wilson loop')
            ap = (-1 if y == n-1 else 1)**2 * (-1 if z == n-1 else 1)**2
            require(ap == 1, 'Closed AP signs did not cancel')
            face_phases.append(str(phase))
    D = s.zeros(len(interior), len(dynamic))
    row = {xyz: i for i, xyz in enumerate(interior)}
    for j, edge in enumerate(dynamic):
        xyz, axis = edges[edge]; endpoint = target(xyz, axis)
        if xyz in row: D[row[xyz], j] += 1
        if endpoint in row: D[row[endpoint], j] -= 1
    L = D*D.T
    require(all(p > 0 for p in L.LDLdecomposition(hermitian=False)[1].diagonal()), 'Dirichlet Laplacian not positive')
    inverse = L.inv()
    species = [s.Matrix([2 if xyz[0] == 1 else -2 for xyz in interior]) for _ in range(64)]
    rho = sum(species, s.zeros(len(interior), 1))
    consume_density(species, rho)
    omitted_density = must_reject(lambda: consume_density(species, rho-species[-1]))
    separate_density = must_reject(lambda: consume_density(species, species[0]))
    longitudinal = D.T*inverse*rho
    require(D*longitudinal == rho, 'Exact collective Gauss failed')
    hc = 64*(rho.T*inverse*rho)[0]
    separate_hc = sum((64*(r.T*inverse*r)[0] for r in species), s.Integer(0))
    require(hc == 64*separate_hc and hc != separate_hc, 'Species HC adverse control did not discriminate')
    k = s.zeros(len(dynamic), 1)
    edge = lookup[((1, 0, 0), 0)]; k[dynamic.index(edge)] = -1
    drho = D*k; delta = D.T*inverse*drho; qk = k-delta
    require(D*qk == s.zeros(len(interior), 1) and (qk.T*delta)[0] == 0, 'Orthogonal split failed')
    incoming = longitudinal+s.Rational(3, 8)*qk
    full_delta = 64*((incoming+k).dot(incoming+k)-incoming.dot(incoming))
    transverse_delta = 64*(s.Rational(11, 8)**2-s.Rational(3, 8)**2)*qk.dot(qk)
    hc_delta = 64*((rho+drho).dot(inverse*(rho+drho))-rho.dot(inverse*rho))
    require(full_delta == transverse_delta+hc_delta, 'HE/collective HC64 identity failed')
    require(D*(incoming+qk) != rho+drho, 'Pi-only update silently accepted')
    # Full colour-summed oriented native hopping, not a selected rank-one channel.
    # A legal pointwise Phi probe makes other-colour hops Pauli blocked; law is not frozen.
    vertices = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]
    base = sum(15 << (4*i) for i, xyz in enumerate(sites) if xyz[0] < n//2)
    for xyz in vertices[1:]: base ^= 1 << (4*site_index[xyz])
    for xyz in [(2, 0, 0), (2, 0, 1), (2, 0, 2)]: base |= 1 << (4*site_index[xyz])
    mask = sum(base << (4*n**3*j) for j in range(65)) | (15 << (4*n**3*65))
    require(all(((mask >> (4*n**3*j)) & ((1 << (4*n**3))-1)).bit_count() == 2*n**3 for j in range(65)),
            'Canonical all1+63+neutral sector changed')
    loop_results = []
    for species_id in range(65):
        profile = profiles[0] if species_id == 64 else profiles[1] if species_id == 0 else profiles[2]
        active = {mask: s.Integer(1)}; angle = F(0)
        for create_xyz, annihilate_xyz, origin, axis, reverse in [
            (vertices[3], vertices[0], vertices[0], 2, True),
            (vertices[2], vertices[3], vertices[3], 1, True),
            (vertices[1], vertices[2], vertices[1], 2, False),
            (vertices[0], vertices[1], vertices[0], 1, False)]:
            record = next(r for r in profile['paths'] if r['axis'] == axis and r['length'] == 1)
            coefficient = s.Rational(record['c0'])*s.diag(1, 2, 3, -6)
            image: dict[int, Any] = {}
            for state, amplitude in active.items():
                for a, b in product(range(4), repeat=2):
                    value = coefficient[a, b]
                    if reverse: value = s.conjugate(coefficient[b, a])
                    if value == 0: continue
                    action = transfer(state, 4*n**3*species_id+4*site_index[annihilate_xyz]+b,
                                      4*n**3*species_id+4*site_index[create_xyz]+a)
                    if action:
                        final, sign = action
                        image[final] = image.get(final, s.Integer(0))+amplitude*sign*value
            active = {state: amplitude for state, amplitude in image.items() if amplitude != 0}
            require(bool(active), 'Full colour-summed native loop annihilated state')
            angle += (-1 if reverse else 1)*atlas(origin, axis) if species_id != 64 else F(0)
        require(set(active) == {mask} and active[mask] != 0, 'Native loop did not return canonical state')
        require(angle == (F(0) if species_id == 64 else F(1, 16)), 'Charged/neutral loop phase wrong')
        loop_results.append({'species': species_id, 'angle_2pi_units': str(angle),
                             'nonzero_endpoint_product_sha256': digest(str(active[mask]).encode())})
    # Arbitrary noncommuting traceless Hermitian probes and ordered two-link SU4.
    basis = []
    for i in range(4):
        for j in range(i+1, 4):
            h = s.zeros(4); h[i,j] = h[j,i] = 1; basis.append(h)
            h = s.zeros(4); h[i,j] = s.I; h[j,i] = -s.I; basis.append(h)
    for j in range(1,4):
        basis.append(s.diag(*([1]*j+[-j]+[0]*(3-j))))
    fx = basis[0]+s.I*basis[2]+basis[12]
    fy = basis[5]+2*s.I*basis[1]+basis[14]
    u1 = s.diag(s.I, -s.I, -1, -1)
    u2 = s.eye(4); u2[0,0] = u2[1,1] = 0; u2[0,1] = 1; u2[1,0] = -1
    u = u1*u2; B = (fx*u+u*fy)/2
    covariance_assertions = 0
    for h in basis:
        dfx = s.I*(h*fx-fx*h); du = s.I*h*u
        require((dfx*u+fx*du+du*fy)/2 == s.I*h*B, 'Boundary source SU4/Phi covariance failed')
        dfy = s.I*(h*fy-fy*h); du = -s.I*u*h
        require((fx*du+du*fy+u*dfy)/2 == -s.I*B*h, 'Boundary target SU4/Phi covariance failed')
        covariance_assertions += 2
    for h in basis:
        require(u1*(s.I*h*u2)+(-s.I*u1*h)*u2 == s.zeros(4), 'Intermediate colour endpoint did not cancel')
        covariance_assertions += 1
    # Exact tail witness: in d real Phi dimensions psi~r^(-(d+5)/2).
    # Norm and quartic form are finite, but linear native multiplier's quartic form is not.
    d = 60
    squared_tail = -(d+5)
    exponents = {str(k): d-1+k+squared_tail for k in (0, 4, 6)}
    require(exponents == {'0': -6, '4': -2, '6': 0}, 'Tail witness algebra failed')
    return {'N': n, 'existing_edges': len(edges), 'pinned_edges': len(pinned), 'dynamic_edges': len(dynamic),
            'boundary_faces': len(face_phases), 'face_phases_sha256': digest(json.dumps(face_phases).encode()),
            'adverse_omitted_pinned_rejected': omitted_rejection, 'unrestricted_absorption_rejected': full_rejection,
            'adverse_omitted_collective_density_rejected': omitted_density,
            'adverse_single_species_density_rejected': separate_density,
            'HC64_collective': str(hc), 'HC64_separate_species': str(separate_hc), 'ratio': '64',
            'HE64_increment': str(full_delta), 'transverse64_increment': str(transverse_delta),
            'HC64_increment': str(hc_delta), 'Pi_only_rejected': True,
            'full_colour_summed_oriented_native_CAR_loops': loop_results,
            'all15_boundary_and_intermediate_Phi_SU4_covariance_assertions': covariance_assertions,
            'closed_form_invariance_counterexample': {'Phi_dimension': d, 'radial_moment_exponents': exponents,
                'norm_finite': True, 'quartic_form_finite': True, 'linear_multiplier_quartic_form_finite': False,
                'meaning': 'Native linear-Phi multipliers preserve smooth compact core, NOT entire closed form domain'},
            'scope': 'NEW boundary law/domain; pointwise full native operator coefficients, not ground or tension'}


def run(request: Path, output: Path) -> None:
    payload = json.loads(request.read_text())
    for name, pin in payload['input_sha256'].items():
        require(digest((ROOT/name).read_bytes()) == pin, 'Input hash changed: '+name)
    with zipfile.ZipFile(ROOT/payload['slab_archive']) as archive:
        profile_raw = archive.read(payload['profile_member'])
        profiles = json.loads(profile_raw)['primitive_profiles']
        archived_source = archive.read('magnetic_boundary_slab_scientist_20261011.py')
        archived_helper = archive.read('magnetic_boundary_scientist_20261011.py')
    with zipfile.ZipFile(ROOT/payload['torus_archive']) as archive:
        torus_source = archive.read('magnetic_boundary_scientist_20261011.py')
    result = {'kind': 'INDEPENDENT_EXACT_MAGNETIC_BOUNDARY_REVIEW_CONTROL_NOT_ADMISSION',
              'source_sha256': digest(Path(__file__).read_bytes()), 'request_sha256': digest(request.read_bytes()),
              'input_sha256': payload['input_sha256'], 'archived_profile_sha256': digest(profile_raw),
              'slab_archived_source_sha256': digest(archived_source),
              'slab_archived_helper_sha256': digest(archived_helper),
              'torus_original_archived_source_sha256': digest(torus_source),
              'control': exact_control(profiles), 'historical_producer_import_or_execution': False,
              'ground_phase_stiffness_pole_admitted': False}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'control': output.name, 'exact_adverse_controls': 'passed', 'ground': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.output)
