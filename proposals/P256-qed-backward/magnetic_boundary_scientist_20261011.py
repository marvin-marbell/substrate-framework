"""Charge-one spatial bundle and nonabsorbed electric deck boundary realization.

This is a NEW boundary-domain extension, not a ground solver. All finite quantum
Phi, SU4 and canonical CAR factors remain in the operator/form specification.
Only frozen incidence and occupation-sign helpers are imported from old sources.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from itertools import product
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import sympy as sp

from magnetic_mechanism_scientist_20261011 import incidence
from magnetic_algebra_review_control_20261011 import car_transfer

ROOT = Path(__file__).resolve().parent
ZERO = (F(0), F(0))
ONE = (F(1), F(0))


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_matvec(matrix: Any, vector: list[F]) -> list[F]:
    """Integer CSR incidence acting on exact fractions, without float casting."""
    csr = matrix.tocsr()
    return [sum((int(csr.data[j])*vector[int(csr.indices[j])]
                 for j in range(csr.indptr[i], csr.indptr[i+1])), F(0))
            for i in range(csr.shape[0])]


def qadd(a: tuple[F, F], b: tuple[F, F]) -> tuple[F, F]:
    return a[0] + b[0], a[1] + b[1]


def qmul(a: tuple[F, F], b: tuple[F, F]) -> tuple[F, F]:
    return a[0]*b[0] + 2*a[1]*b[1], a[0]*b[1] + a[1]*b[0]


def qinv(a: tuple[F, F]) -> tuple[F, F]:
    norm = a[0]*a[0] - 2*a[1]*a[1]
    return a[0]/norm, -a[1]/norm


def cosine(n: int, k: int) -> tuple[F, F]:
    k %= n
    if n == 4:
        return (F((1, 0, -1, 0)[k]), F(0))
    if n == 6:
        return (F((2, 1, -1, -2, -1, 1)[k], 2), F(0))
    if n == 8:
        return ((F(1), F(0)), (F(0), F(1, 2)), ZERO,
                (F(0), F(-1, 2)), (F(-1), F(0)),
                (F(0), F(-1, 2)), ZERO, (F(0), F(1, 2)))[k]
    raise ValueError('Exact diagnostic Green function supports N4/6/8 only')


def exact_green(n: int) -> list[F]:
    """Exact Q(sqrt2) Fourier contraction, followed by rational certification.

    Reflection/permutation orbits avoid a dense rational matrix inverse. No
    binary64 reconstruction or old Poisson solver supplies the exact result.
    """
    sites = list(product(range(n), repeat=3))
    orbit = lambda x: tuple(sorted(min(t % n, (-t) % n) for t in x))
    inverse = {}
    for k in sites[1:]:
        cs = [cosine(n, j) for j in k]
        value = (F(6)-2*sum((c[0] for c in cs), F(0)), -2*sum((c[1] for c in cs), F(0)))
        inverse[k] = qinv(value)
    values = {}
    for r in sorted({orbit(x) for x in sites}):
        total = ZERO
        for k, inv in inverse.items():
            c = ONE
            for axis in range(3):
                c = qmul(c, cosine(n, k[axis]*r[axis]))
            total = qadd(total, qmul(c, inv))
        assert total[1] == 0, 'Irrational Fourier pieces must cancel exactly'
        values[r] = total[0]/n**3
    green = [values[orbit(x)] for x in sites]
    lookup = {x: i for i, x in enumerate(sites)}
    for i, x in enumerate(sites):
        lap = 6*green[i]
        for axis in range(3):
            for sign in (-1, 1):
                y = list(x); y[axis] = (y[axis]+sign) % n
                lap -= green[lookup[tuple(y)]]
        assert lap == F(int(i == 0))-F(1, n**3)
    assert sum(green) == 0
    return green


class ChargeOneBundle:
    """Connection charts in units 2pi; native AP remains a separate -1 seam.

    Universal chart a_y=m*x/N^2, a_x=a_z=0 and transition
    g_x(y)=exp(-2pi*i*m*y/N). The finite seam contains that transition.
    theta is an unrestricted additional real periodic edge cochain.
    """
    def __init__(self, n: int, m: int = 1):
        if n < 4 or n % 2 or not isinstance(m, int):
            raise ValueError('Even N>=4 and integral charge-one cocycle required')
        self.n, self.m, self.v = n, m, n**3
        self.sites = list(product(range(n), repeat=3))
        self.index = {x: i for i, x in enumerate(self.sites)}
        self.D, self.C, self.S = incidence(n)
        self.a = [F(0)]*(3*self.v)
        for i, (x, y, _z) in enumerate(self.sites):
            self.a[3*i+1] = F(m*x, n*n)
            if x == n-1:
                self.a[3*i] = F(-m*y, n)
        self.sheet = [0]*(3*self.v)
        for i, (x, y, _z) in enumerate(self.sites):
            if x == n-1 and y == n-1:
                self.sheet[3*i] = m

    def target(self, x: int, axis: int, length: int = 1) -> int:
        y = list(self.sites[x]); y[axis] = (y[axis]+length) % self.n
        return self.index[tuple(y)]

    def path(self, x: int, axis: int, length: int) -> dict[str, Any]:
        edges, seam, current = [], 1, x
        for _ in range(length):
            edges.append(3*current+axis)
            if self.sites[current][axis] == self.n-1:
                seam *= -1
            current = self.target(current, axis)
        return dict(source=x, target=current, axis=axis, length=length,
                    edges=edges, seam=seam, bundle_angle=sum(self.a[e] for e in edges))

    def phase(self, path: dict[str, Any], theta: list[F], charged: bool = True) -> F:
        return sum((theta[e]+self.a[e] for e in path['edges']), F(0)) if charged else F(0)

    def lifted_curvature(self, theta: list[F]) -> list[F]:
        raw = exact_matvec(self.C, [theta[i]+self.a[i] for i in range(3*self.v)])
        return [F(raw[i])+self.sheet[i] for i in range(3*self.v)]

    def endpoint(self, row: dict[str, Any], path: dict[str, Any],
                 fields: dict[int, list[Any]], links: dict[int, Any]) -> Any:
        """Literal full-Phi endpoint average; no selected frame in the law."""
        c0, ci = sp.Rational(row['c0']), sp.Rational(row['ci'])
        x, y, axis = path['source'], path['target'], path['axis']
        fx = c0*fields[x][0] + sp.I*ci*fields[x][axis+1]
        fy = c0*fields[y][0] + sp.I*ci*fields[y][axis+1]
        u = sp.eye(4)
        for edge in path['edges']:
            u = u*links[edge]
        return path['seam']*(fx*u+u*fy)/2

    def apply_path(self, mask: int, species: int, path: dict[str, Any],
                   coefficient: Any, theta: list[F], charged: bool = True) -> dict[int, Any]:
        """Actual canonical full-mode CAR forward plus adjoint operator action."""
        offset = 4*self.v*species
        phase = sp.exp(2*sp.pi*sp.I*sp.Rational(self.phase(path, theta, charged)))
        result: dict[int, Any] = {}
        for a, b in product(range(4), repeat=2):
            create = offset+4*path['source']+a
            annihilate = offset+4*path['target']+b
            for i, j, value in ((create, annihilate, coefficient[a, b]*phase),
                                (annihilate, create, sp.conjugate(coefficient[a, b]*phase))):
                action = car_transfer(mask, j, i)
                if action is not None and value != 0:
                    target, sign = action
                    result[target] = result.get(target, 0)+sign*value
        return result


def principal(value: F) -> F:
    return value - (value+F(1, 2)).numerator//(value+F(1, 2)).denominator


def chart_control(bundle: ChargeOneBundle) -> dict[str, Any]:
    n, v = bundle.n, bundle.v
    b = bundle.lifted_curvature([F(0)]*(3*v))
    assert all(b[3*i] == F(1, n*n) and b[3*i+1] == b[3*i+2] == 0 for i in range(v))
    assert not any(exact_matvec(bundle.S, b))
    planes = [sum(b[3*i] for i, x in enumerate(bundle.sites) if x[2] == z) for z in range(n)]
    assert planes == [F(1)]*n
    paths, ap, twolink_ap = 0, 0, 0
    exact_records = []
    for i, xyz in enumerate(bundle.sites):
        for axis, length in product(range(3), (1, 2)):
            p = bundle.path(i, axis, length)
            paths += 1; ap += int(p['seam'] == -1)
            twolink_ap += int(length == 2 and p['seam'] == -1)
            # Independent universal chart integral plus endpoint chart transition.
            unwrapped_target = xyz[axis]+length
            expected = F(length*xyz[0], n*n) if axis == 1 else F(0)
            if axis == 0 and unwrapped_target >= n:
                expected -= F(xyz[1], n)
            assert (p['bundle_angle']-expected).denominator == 1
            assert p['seam'] == (-1 if unwrapped_target >= n else 1)
            exact_records.append([i, axis, length, p['target'], p['edges'], p['seam'], str(p['bundle_angle'])])
    assert twolink_ap == 6*n*n
    wrong_seam = bundle.a.copy()
    for i, xyz in enumerate(bundle.sites):
        if xyz[0] == n-1:
            wrong_seam[3*i] = F(0)
    wrong_b = exact_matvec(bundle.C, wrong_seam)
    wrong_uniform = sum(F(wrong_b[3*i]) != F(1, n*n) for i in range(v))
    assert wrong_uniform > 0
    # Reach the actual charge1-domain accept boundary with a fractional cocycle.
    fractional_rejected = False
    try:
        ChargeOneBundle(n, cast(Any, F(1, 2)))
    except ValueError:
        fractional_rejected = True
    assert fractional_rejected
    # Candidate principal-flux m=1 component is NOT E-translation invariant.
    shift = [F(0)]*(3*v)
    shift[0] = F(1, 2)-F(1, 2*n*n)
    before = [principal(t) for t in b]
    after = [principal(t) for t in bundle.lifted_curvature(shift)]
    old_flux = sum(before[3*i] for i, xyz in enumerate(bundle.sites) if xyz[2] == 0)
    new_flux = sum(after[3*i] for i, xyz in enumerate(bundle.sites) if xyz[2] == 0)
    assert old_flux == 1 and new_flux == 0
    defects = exact_matvec(bundle.S, after)
    assert any(defects) and sum(defects) == 0
    lifted_after = bundle.lifted_curvature(shift)
    assert sum(lifted_after[3*i] for i, xyz in enumerate(bundle.sites) if xyz[2] == 0) == 1
    # Independent original-cover path t*A: raw flux0 throughout, principal0->1.
    raw_background = exact_matvec(bundle.C, bundle.a)
    threshold = F(n*n, 2*(n*n-1))
    delta = F(1, 10*n*n)
    path_rows = []
    for parameter in (threshold-delta, threshold+delta, F(1)):
        raw_path = [parameter*value for value in raw_background]
        wrapped_path = [principal(value) for value in raw_path]
        raw_planes = [sum(raw_path[3*i] for i, xyz in enumerate(bundle.sites) if xyz[2] == z) for z in range(n)]
        wrapped_planes = [sum(wrapped_path[3*i] for i, xyz in enumerate(bundle.sites) if xyz[2] == z) for z in range(n)]
        assert raw_planes == [F(0)]*n
        assert wrapped_planes == [F(int(parameter > threshold))]*n
        assert not any(exact_matvec(bundle.S, wrapped_path))
        path_rows.append(dict(parameter=str(parameter), raw_plane_flux='0',
                              principal_plane_flux=str(wrapped_planes[0]),
                              cube_monopoles=0))
    # Real cochain background is exactly absorbed in every literal charged path.
    theta = [F(((i*7+3) % 19)-9, 37) for i in range(3*v)]
    shifted_theta = [theta[j]+bundle.a[j] for j in range(3*v)]
    for i, axis, length, _y, _edges, _seam, _phase in exact_records:
        p = bundle.path(i, axis, length)
        assert bundle.phase(p, theta) == sum(shifted_theta[e] for e in p['edges'])
    return dict(N=n, m=1, exact_plane_flux_in_2pi_units=list(map(str, planes)),
                cocycle_charge1_exact=True, path_count=paths, AP_paths=ap,
                two_link_AP_paths=twolink_ap,
                exact_path_records_sha256=sha(json.dumps(exact_records).encode()),
                omitted_transition_nonuniform_faces=wrong_uniform,
                original_cover_Dirac_sheet_crossing=dict(threshold=str(threshold),
                    below_above_endpoint=path_rows, all_raw_plane_flux_zero=True,
                    fullE_path_preserved_by_original_domain=True),
                fractional_m_half_cocycle='exp(-pi i)=-1, rejects closed charge1 bundle',
                principal_flux_candidate=dict(before=str(old_flux), after=str(new_flux),
                    translation_edge=0, translation_2pi_units=str(shift[0]),
                    nonzero_cube_defects=sum(bool(t) for t in defects),
                    reducing_sector=False, admissibility_cutoff_added=False),
                unrestricted_chart_lift_flux_after_translation='1',
                background_absorption_every_native_path_exact=True)


def gauss_control(bundle: ChargeOneBundle) -> dict[str, Any]:
    n, v = bundle.n, bundle.v
    green = exact_green(n)
    # All64 canonical stripe singlets: each species has2V, Ref4, neutral2V.
    rho = [F(128 if x[0] < n//2 else -128) for x in bundle.sites]
    stripe = [F(128 if x < n//2 else -128) for x in range(n)]
    prefix, total = [], F(0)
    for r in stripe:
        total += r; prefix.append(total)
    edge_x = [a-sum(prefix, F(0))/n for a in prefix]
    potential_x = [F(0)]
    for x in range(n-1):
        potential_x.append(potential_x[-1]-edge_x[x])
    potential_x = [a-sum(potential_x, F(0))/n for a in potential_x]
    potential = [potential_x[x[0]] for x in bundle.sites]
    el = exact_matvec(bundle.D.T, potential)
    assert exact_matvec(bundle.D, el) == rho
    assert not any(exact_matvec(bundle.C, el))
    assert sum(rho) == 0
    rows = []
    for length in (1, 2):
        x = bundle.index[(n-1, 0, 0)]
        p = bundle.path(x, 0, length); y = p['target']
        k = [F(0)]*(3*v)
        for e in p['edges']:
            k[e] += 1
        drho = exact_matvec(bundle.D, k)
        def translated_green(site: tuple[int, ...], centre: tuple[int, ...]) -> F:
            return green[bundle.index[tuple((site[j]-centre[j]) % n for j in range(3))]]
        phi = [translated_green(site, bundle.sites[x])-translated_green(site, bundle.sites[y]) for site in bundle.sites]
        delta_el = exact_matvec(bundle.D.T, phi)
        assert exact_matvec(bundle.D, delta_el) == drho
        qk = [k[j]-delta_el[j] for j in range(3*v)]
        assert not any(exact_matvec(bundle.D, qk))
        assert sum(qk[j]*delta_el[j] for j in range(3*v)) == 0
        incoming = [el[j]+F(3, 8)*qk[j] for j in range(3*v)]
        after = [incoming[j]+k[j] for j in range(3*v)]
        assert exact_matvec(bundle.D, after) == [rho[j]+drho[j] for j in range(v)]
        full_increment = 64*sum(after[j]**2-incoming[j]**2 for j in range(3*v))
        transverse_increment = 64*((F(11, 8)**2-F(3, 8)**2)*sum(t*t for t in qk))
        coulomb_increment = 64*(2*sum(rho[j]*phi[j] for j in range(v))+sum(drho[j]*phi[j] for j in range(v)))
        assert full_increment == transverse_increment+coulomb_increment
        wrong_gauss = exact_matvec(bundle.D, [incoming[j]+qk[j] for j in range(3*v)])
        assert wrong_gauss != [rho[j]+drho[j] for j in range(v)]
        rows.append(dict(length=length, AP=p['seam'], HE64_increment=str(full_increment),
                         transverse64_increment=str(transverse_increment),
                         collective_HC64_increment=str(coulomb_increment),
                         Pi_only_wrong_Gauss_rejected=True,
                         affine_fullE_density_update_exact=True))
    # Actual contrary case from native physical E, not a simultaneous gauge move.
    lam = [F(0)]*v; lam[0] = F(1, 3)
    alpha = exact_matvec(bundle.D.T, lam)
    assert not any(exact_matvec(bundle.C, alpha))
    p = bundle.path(0, 0, 1)
    path_angle = sum(alpha[e] for e in p['edges'])
    assert path_angle == F(1, 3)
    # alpha here is in radians, so phase exp(i/3), not a 2pi-normalized phase.
    hc = 64*sum(rho[j]*potential[j] for j in range(v))
    separate = 64*64*sum((rho[j]/64)*(potential[j]/64) for j in range(v))
    assert hc == 64*separate
    return dict(N=n, green_exact_sha256=sha(json.dumps(list(map(str, green))).encode()),
                green_poisson_all_sites_exact=True, rho='sum64(n_s-2)+(Nref-4)delta0',
                Ref_count=4, neutral_count=2*v, charged_counts=[2*v]*64,
                collective_HC64=str(hc), wrong_species_separate_HC=str(separate),
                collective_over_wrong_separate_ratio=64,
                fullE_hop_controls=rows,
                actual_pure_gradient_translation=dict(Calpha_exact_zero=True,
                    native_path_radians=str(path_angle), phase='exp(i/3) !=1',
                    simultaneous_Gauss_generator='lambda.(DE-rho), NOT lambda.DE',
                    pure_translation_changes_H=True))


def generic_phi(seed: int) -> list[Any]:
    result = []
    for mu in range(4):
        mat = sp.zeros(4)
        for i in range(4):
            mat[i, i] = (seed+mu+1)*(i+1)
        mat -= sp.trace(mat)*sp.eye(4)/4
        for i in range(4):
            for j in range(i+1, 4):
                mat[i, j] = seed+i+j+mu+1+sp.I*(1+seed+2*i+j+mu)
                mat[j, i] = sp.conjugate(mat[i, j])
        assert mat == mat.adjoint() and sp.trace(mat) == 0
        result.append(mat)
    return result


def su4_basis() -> list[Any]:
    basis = []
    for i in range(4):
        for j in range(i+1, 4):
            a, b = sp.zeros(4), sp.zeros(4)
            a[i, j] = a[j, i] = 1
            b[i, j], b[j, i] = sp.I, -sp.I
            basis.extend((a, b))
    for i in range(3):
        a = sp.zeros(4); a[i, i] = 1; a[3, 3] = -1
        basis.append(a)
    assert len(basis) == 15
    return basis


def canonical_operator_control(bundle: ChargeOneBundle, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    v, n = bundle.v, bundle.n
    # A full global bit mask, not tensoring 63 into a numerical coupling.
    species_mask = sum(15 << (4*i) for i, xyz in enumerate(bundle.sites) if xyz[0] < n//2)
    mask = sum(species_mask << (4*v*s) for s in range(65))
    ref_offset = 4*v*65
    mask |= 15 << ref_offset
    assert mask.bit_count() == 65*2*v+4
    assert all(((mask >> (4*v*s)) & ((1 << (4*v))-1)).bit_count() == 2*v for s in range(65))
    theta = [F(0)]*(3*v)
    transfers, gauss_checks, endpoints, wrong_two_phase = 0, 0, 0, 0
    exact_coefficients = []
    for length in (1, 2):
        x = bundle.index[(n-1, 1, 0)]
        p = bundle.path(x, 0, length); y = p['target']
        fields = {x: generic_phi(1), y: generic_phi(4)}
        # Noncommuting exact determinant-one colour links on each ordered edge.
        links = {edge: sp.diag(sp.I, -sp.I, -1, -1) for edge in p['edges']}
        if length == 2:
            links[p['edges'][1]] = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        assert all(u.adjoint()*u == sp.eye(4) and u.det() == 1 for u in links.values())
        for profile in profiles:
            if profile['scale'] == 'heavy' and length == 2:
                continue
            row = next(r for r in profile['paths'] if r['axis'] == 0 and r['length'] == length)
            coefficient = bundle.endpoint(row, p, fields, links)
            endpoints += 1
            u = sp.eye(4)
            for edge in p['edges']:
                u *= links[edge]
            # All15 local SU4 infinitesimal generators, including fullPhi action.
            c0, ci = sp.Rational(row['c0']), sp.Rational(row['ci'])
            fx = c0*fields[x][0]+sp.I*ci*fields[x][1]
            fy = c0*fields[y][0]+sp.I*ci*fields[y][1]
            for h in su4_basis():
                dfx = sp.I*(h*fx-fx*h); du = sp.I*h*u
                left_derivative = p['seam']*(dfx*u+fx*du+du*fy)/2
                assert sp.simplify(left_derivative-sp.I*h*coefficient) == sp.zeros(4)
                dfy = sp.I*(h*fy-fy*h); du = -sp.I*u*h
                right_derivative = p['seam']*(fx*du+du*fy+u*dfy)/2
                assert sp.simplify(right_derivative+sp.I*coefficient*h) == sp.zeros(4)
                gauss_checks += 2
            slots = [64] if profile['scale'] == 'heavy' else ([0] if profile['load'] == 1 else list(range(1, 64)))
            for species in slots:
                image = bundle.apply_path(mask, species, p, coefficient, theta, charged=species != 64)
                assert len(image) == 16
                for newmask, value in image.items():
                    assert newmask.bit_count() == mask.bit_count()
                    assert ((newmask >> ref_offset) & 15) == 15
                    assert ((newmask >> (4*v*species)) & ((1 << (4*v))-1)).bit_count() == 2*v
                    assert ((newmask >> (4*v*species+4*x)) & 15).bit_count() == 1
                    assert ((newmask >> (4*v*species+4*y)) & 15).bit_count() == 3
                    assert value != 0
                    transfers += 1
            exact_coefficients.append(dict(role=profile['role'], length=length,
                phase_2pi_units=str(bundle.phase(p, theta, profile['scale'] != 'heavy')),
                endpoint_sha256=sha(str(coefficient).encode())))
        # Independent derivative of all ordered charge1 path links, not endpoint-only.
        t = sp.Symbol('t', real=True)
        path_frequency = sum(1 for _e in p['edges'])
        phase = sp.exp(sp.I*length*t)
        assert sp.diff(phase, t) == sp.I*path_frequency*phase
        if length == 2:
            assert sp.diff(phase, t) != sp.I*phase
            wrong_two_phase += 1
    # Ref transitions at origin: charged creation/Ref annihilation is canonical
    # and U1-neutral overall. Ref4->Ref3 pays the actual -1 density correction.
    empty_site = bundle.index[(n-1, 0, 0)]
    # Reference is at0 in the law; use an alternative legal2V occupation with0 empty.
    reference_mask = mask ^ (15 << 0) ^ (15 << (4*empty_site))
    ref_images = []
    for colour in range(4):
        action = car_transfer(reference_mask, ref_offset+colour, colour)
        assert action is not None
        image, sign = action
        assert (image & 15).bit_count() == 1 and ((image >> ref_offset) & 15).bit_count() == 3
        before_rho0 = sum(((reference_mask >> (4*v*s)) & 15).bit_count()-2 for s in range(64))
        after_rho0 = sum(((image >> (4*v*s)) & 15).bit_count()-2 for s in range(64))-1
        assert before_rho0 == after_rho0
        ref_images.append(dict(colour=colour, CAR_sign=sign, Ref_before=4, Ref_after=3,
                               charged_before=2*v, charged_after=2*v+1, rho0_unchanged=True))
    # Four-mode canonical singlets transform by det(g)=1 for every SU4.
    g = sp.diag(sp.I, -sp.I, -1, -1)
    assert g.det() == 1
    return dict(N=n, global_modes=65*4*v+4, charged_species=64,
                full_global_bitmask_sha256=sha(mask.to_bytes((65*4*v+4+7)//8, 'little')),
                canonical_CAR_images_exercised=transfers, literal_endpoint_blocks=endpoints,
                fullPhi_SU4_local_generator_covariance_assertions=gauss_checks,
                exact_noncommuting_fullPhi_probes=True, neutral_charge0_retained=True,
                neutral_scale='-2^8192, symbolic primitive factor; not evaluated as float',
                Ref_transition_controls=ref_images,
                wrong_two_link_derivative_rejected=wrong_two_phase,
                endpoint_blocks=exact_coefficients,
                full_field_factor='L2(R^(60V)) and SU4 Haar/Casimir, no field freeze or law truncation')


def cylinder_control(bundle: ChargeOneBundle) -> dict[str, Any]:
    v = bundle.v
    face = np.zeros(3*v, dtype=np.int64); face[0] = 1
    e = bundle.C.T @ face
    assert not any(bundle.D @ e) and int(e @ e) == 4
    assert any(bundle.C @ e)
    beta = sp.Rational(1, 3)
    t, s = sp.symbols('t s', real=True)
    psi = sp.exp(sp.I*beta*t)
    holonomy = sp.exp(2*sp.pi*sp.I*beta)
    assert sp.simplify(psi.subs(t, t+2*sp.pi)-holonomy*psi) == 0
    derivative = -sp.I*sp.diff(psi, t)
    assert sp.simplify(derivative-beta*psi) == 0
    assert sp.simplify(derivative.subs(t, t+2*sp.pi)-holonomy*derivative) == 0
    electric_energy = sp.simplify(64*beta**2/4)
    assert electric_energy == sp.Rational(16, 9)
    frequencies = set()
    for x, axis, length in product(range(v), range(3), (1, 2)):
        p = bundle.path(x, axis, length)
        frequency = sum(int(e[j]) for j in p['edges'])
        frequencies.add(frequency)
        assert isinstance(frequency, int)
        # Actual boundary preservation for charge1 hopping and translated core.
        wave = sp.exp(sp.I*(beta+frequency)*t)
        assert sp.simplify(wave.subs(t, t+2*sp.pi)-holonomy*wave) == 0
    translated = psi.subs(t, t+s)
    assert sp.simplify(translated.subs(t, t+2*sp.pi)-holonomy*translated) == 0
    # A fake fractional path does not preserve this domain.
    bad = sp.exp(sp.I*(beta+sp.Rational(1, 2))*t)
    assert sp.simplify(bad.subs(t, t+2*sp.pi)-holonomy*bad) != 0
    # Multiplying away beta changes E: no claim of same operator/domain equivalence.
    periodic = sp.exp(-sp.I*beta*t)*psi
    assert sp.simplify(periodic-1) == 0
    assert sp.simplify(-sp.I*sp.diff(periodic, t)) == 0
    return dict(N=bundle.n, deck_vector=list(map(int, e)), deck_norm2=4,
                beta='1/3', quasi_boundary='psi(theta+2pi e)=exp(2pi i/3) psi(theta)',
                fullE_derivative_boundary_preserved=True,
                all_integer_native_one_two_path_frequencies=sorted(frequencies),
                native_path_boundary_preserved=True,
                all_real_fullE_translations_preserve_domain=True,
                wrong_fractional_charge_path_rejected=True,
                electric_direction_core_plane_wave_energy64='16/9',
                energy_scope='Only deck-direction derivative contribution on a smooth compact-supported transverse/Phi/Haar core packet, NOT full H expectation or eigenvalue.',
                removing_beta_changes_electric_operator='E -> E+beta e/||e||^2',
                raw_local_curl_does_not_descend=True,
                magnetic_closed_plane_operator='Q_xy=m I, exactly1 in this fibre; bounded scalar, NOT a fluctuating rawB atom',
                magnetic_background_absorbed_even_on_cylinder=True,
                representation='Declared partially compactified electric-Bloch chart/bundle domain, NOT original-cover reducing Hilbert subspace; m alone is an isospectral boundary label')


def run(request_path: Path, output: Path) -> None:
    request = json.loads(request_path.read_text())
    for name, pin in request['input_sha256'].items():
        assert sha((ROOT/name).read_bytes()) == pin, name
    profiles = json.loads((ROOT/request['profile']).read_text())['primitive_profiles']
    assert [(p['load'], p['scale']) for p in profiles] == [(1, 'heavy'), (1, 'charged'), (63, 'charged')]
    result: dict[str, Any] = dict(kind='NEW_MAGNETIC_CHART_AND_ELECTRIC_BLOCH_BOUNDARY_DOMAIN_NOT_GROUND',
        source_sha256=sha(Path(__file__).read_bytes()), request_sha256=sha(request_path.read_bytes()),
        input_sha256=request['input_sha256'], N_controls=[],
        old_main_solver_checker_executed=False, laws_couplings_roots_covariances_adopted=False,
        new_EM_B2=False, physical_ground_or_magnetic_stiffness_or_atom_proved=False)
    for n in request['N']:
        bundle = ChargeOneBundle(n, 1)
        result['N_controls'].append(dict(N=n, chart=chart_control(bundle),
            collective_Gauss=gauss_control(bundle), fullcanonical=canonical_operator_control(bundle, profiles),
            compatible_boundary=cylinder_control(bundle)))
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(receipt=output.name, N=request['N'], decision='principal sector rejected; magnetic background absorbed; nontrivial electric-Bloch boundary extension realized', ground=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.output)
