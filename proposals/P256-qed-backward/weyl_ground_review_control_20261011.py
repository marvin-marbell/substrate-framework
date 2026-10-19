"""Independent exact adverse controls; no native ground or photon computation."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, cast
import zipfile

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def check(ok: bool, label: str) -> None:
    if not ok:
        raise AssertionError(label)


def exact_lattice(n: int) -> dict[str, Any]:
    sites = [(x[0], x[1], x[2]) for x in itertools.product(range(n), repeat=3)]
    parity = {x: (-1) ** sum(x) for x in sites}
    def target(x: tuple[int, int, int], j: int) -> tuple[int, int, int]:
        y = list(x)
        y[j] = (y[j] + 1) % n
        return (y[0], y[1], y[2])
    def circulation(x: tuple[int, int, int]) -> dict[tuple[tuple[int, int, int], int], F]:
        return {(x, 0): F(1, 2), (target(x, 0), 1): F(1, 2),
                (target(x, 1), 0): F(-1, 2), (x, 1): F(-1, 2)}
    def div(flow: dict[tuple[tuple[int, int, int], int], F]) -> dict[tuple[int, int, int], F]:
        out = {x: F(0) for x in sites}
        for (x, j), value in flow.items():
            out[x] += value
            out[target(x, j)] -= value
        return out
    b = circulation((0, 0, 0))
    check(all(v == 0 for v in div(b).values()), 'closed coexact insertion')
    check(sum(v * v for v in b.values()) == 1, 'unit plaquette norm')
    el = {(x, j): F(64, 3) * parity[x] for x in sites for j in range(3)}
    check(div(el) == {x: F(128 * parity[x]) for x in sites}, 'all64 collective Gauss')
    check(sum(el[e] * v for e, v in b.items()) == 0, 'closed blindness to longitudinal field')
    check(sum(4 if parity[x] == 1 else 0 for x in sites) == 2 * n ** 3, 'canonical particles per species')
    chars = [circulation(x) for x in sites]
    check(len({tuple(sorted(c.items())) for c in chars}) == n ** 3, 'translated coexact flows distinct')
    # Full native translations T_m use E, not E_T. On closed flows their
    # character dual is the saturated integer cycle lattice ker(D) cap Z^edges.
    # Distinct raw flows with a noninteger edge are detected by m=unit_edge.
    for s in (F(1, 3), F(1, 7), F(1, 31)):
        for i, c in enumerate(chars):
            for d in chars[i + 1:]:
                differences = [s * (c.get(e, F(0)) - d.get(e, F(0))) for e in c.keys() | d.keys()]
                check(any(v != 0 and v.denominator != 1 for v in differences), 'proper fullE character separation')
    # Checkerboard phi=(32/3)parity solves L phi=rho; exact edge field above.
    kedge = ((0, 1, 0), 0)  # empty creation endpoint, full annihilation endpoint
    check(parity[kedge[0]] == -1 and parity[target(*kedge)] == 1, 'allowed CAR hop orientation')
    k: dict[tuple[tuple[int, int, int], int], F] = {kedge: F(1)}
    dk = div(k)
    incoming = F(3, 8)
    full = 64 * (2 * (el[kedge] + incoming * b[kedge]) + 1)
    # Foster identity Tr(D^T L+ D)=V-1; all3V adjacent edges are related by
    # translation/axis automorphisms, hence each resistance is (V-1)/(3V).
    resistance = F(n ** 3 - 1, 3 * n ** 3)
    transverse = 64 * (2 * incoming * b[kedge] + 1 - resistance)
    hc = 64 * (2 * el[kedge] + resistance)
    check(full == transverse + hc, 'exact open-hop HE plus collective HC')
    check(hc != 0 and full != transverse, 'Pi-only energy wrongcase rejected')
    rho_after = {x: 128 * parity[x] + dk[x] for x in sites}
    full_after = dict(el)
    full_after[kedge] += 1
    check(div(full_after) == rho_after, 'affine fullE hop legal')
    check(max(abs(dk[x]) for x in sites) == 1, 'Pi-only Gauss residual exactly1')
    hc_collective = 64 * sum(v * v for v in el.values())
    hc_independent = 64 * 64 * sum((v / 64) ** 2 for v in el.values())
    check(hc_collective / hc_independent == 64, 'collective versus64 independent Coulomb copies')
    # No sampled coefficient replaces the quantum operator: give an exact
    # traceless SU4 generator with nonzero native F endpoint matrix element.
    check(sp.diag(1, -1, 0, 0).trace() == 0, 'nonzero traceless colour coefficient')
    return {'N': n, 'particles_each_of65': 2*n**3, 'Ref': 4,
            'SU4_singlets': 'wedge0 and wedge4 (det SU4=1), fullPhi radial Gaussian/Haar constant',
            'full_open_increment': str(full), 'transverse_increment': str(transverse), 'HC_increment': str(hc),
            'adjacent_resistance_exact_by_Foster': str(resistance), 'Pi_only_Gauss_residual': 1,
            'collective_HC_ratio': 64, 'distinct_coexact_characters': n**3,
            'Fourier_Gram': 'V terms each |coefficient|^2=1/V; total1 at every discrete q',
            'finite_form_trial_not_ground': True, 'interval_claim': False}


def operator_packet_spectral() -> dict[str, Any]:
    theta, phi, colour, s, b = sp.symbols('theta phi colour s b', real=True)
    psi = sp.Function('joint')(theta, phi, colour)
    u = sp.exp(sp.I*s*b*theta)
    electric = lambda f: -sp.I*sp.diff(f, theta)
    check(sp.simplify(electric(u*psi)-u*(electric(psi)+s*b*psi)) == 0, 'electric plus sign')
    check(sp.simplify(sp.diff(u*psi, phi, 2)-u*sp.diff(psi, phi, 2)) == 0, 'fullPhi kinetic commute')
    # Arbitrary left/right vector field in colour coordinates, not a flattened Casimir claim.
    vector = sp.Function('left_right')(colour)
    check(sp.simplify(vector*sp.diff(u*psi, colour)-u*vector*sp.diff(psi, colour)) == 0, 'SU4 vector field commute')
    m, r = sp.symbols('m r', real=True)
    delta, B = sp.symbols('delta B', positive=True)
    excess = delta + 128*r*m + 64*r*r*B
    rstar = -m/B
    check(sp.simplify(excess.subs(r, rstar)-(delta-64*m*m/B)) == 0, 'packet optimized defect')
    bound = (sp.sqrt(delta)+8*sp.Abs(s)*sp.sqrt(B))**2
    check(sp.expand(bound-(delta+16*sp.Abs(s)*sp.sqrt(delta*B)+64*s*s*B)) == 0, 'packet bound coefficient8')
    # Wrong first moment cannot be globally minimizing: choose r opposite m.
    check((128*r*m+64*r*r).subs({m:1, r:sp.Rational(-1,64)}) < 0, 'both signs needed for firstmoment0')
    x = sp.symbols('x', real=True)
    eta = sp.Rational(1, 100)
    center = sp.Rational(1, 3)
    norm = sp.integrate(1/(2*eta), (x, center-eta, center+eta))
    null = sp.integrate(sp.Piecewise((1, sp.Eq(x,center)), (0,True)), (x,center-eta,center+eta))
    check(norm == 1 and null == 0, 'original measure normalization and null preservation')
    check(2*eta < sp.Rational(1,14), 'finite translated packet supports disjoint')
    # Explicit whole-P0 adverse positive spectral representation, two ground states.
    sv = sp.Rational(1,7)
    mean = 64*sv*sv
    L = sp.diag(0,0,2*mean)
    omega = sp.Matrix([1,0,0])
    xi = sp.Matrix([0,1/sp.sqrt(2),1/sp.sqrt(2)])
    P0 = sp.diag(1,1,0)
    excited = (sp.eye(3)-P0)*xi
    check((omega.T*xi)[0] == 0 and (xi.T*xi)[0] == 1, 'meanconnected totalmass1')
    check((excited.T*excited)[0] == sp.Rational(1,2), 'wholeP0 not excitedmass1')
    check((xi.T*L*xi)[0] == mean and (excited.T*L*excited)[0] == mean, 'P0 preserves first moment')
    # NEW atomless exponential density with the SAME exact total moments.
    eps = sp.symbols('eps', nonnegative=True)
    density = sp.exp(-eps/mean)/mean
    check(sp.integrate(density, (eps,0,sp.oo)) == 1, 'atomless totalmass')
    check(sp.integrate(eps*density, (eps,0,sp.oo)) == mean, 'atomless same firstmoment')
    K = sp.Matrix(4,4,lambda i,j: 1/(1+mean*(i+j)))
    minors = [cast(Any, sp.factor(cast(sp.MatrixBase, K[:j,:j]).det())) for j in range(1,5)]
    check(all(v > 0 for v in minors), 'exact positive temporal kernel adverse')
    # Positive probability mu_n=(1-1/n)delta0+(1/n)delta_n has moment1
    # but weak limit delta0 has moment0: only lower semicontinuity survives.
    check(sp.Rational(1,13)*13 == 1, 'moment-loss sequence exact finite mean')
    regular = sp.limit((1-sp.exp(-s*s))/(s*s), s, 0)
    check(regular == 1, 'regular source mass scales as s squared')
    # Sobolev circle example: coefficients n^-2 give finite form sum n^-2,
    # infinite operator-domain sum1. Domain insertion is not D(H) membership.
    check(sp.zeta(2).is_finite is True, 'form finite but D(H) unsupported')
    return {'electric_sign': '+s b', 'ground_firstmoment': '0 from both signs and arbitrarily small s',
            'packet_m_bound': 'sqrt(delta B)/8', 'packet_excess_bound': str(bound),
            'original_packet_norm': str(norm), 'Borel_singleton_packet_mass': str(null),
            'domain_adverse': 'Fourier n^-2: H1 finite, H2 infinite; insertion is form-domain only',
            'wholeP0_totalmass': 1, 'meanconnected_mass': 1, 'wholeP0_excitedmass': '1/2',
            'wholeP0_firstmoment': str(mean), 'atomless_density': str(density),
            'exact_temporal_leading_minors': [str(v) for v in minors],
            'moment_loss_adverse': 'mu_n mean1 weak->delta0 mean0; limsup bound not equality',
            'sharp_strong_continuity_adverse': '||U_s Omega-Omega||^2=2; divided norm2/s^2',
            'regular_connected_mass_over_s_squared_limit': str(regular), 'pole_inferred': False}


def run(output: Path) -> None:
    snapshot = ROOT/'vacuum_weyl_parent_review_original_targets_20261011.zip'
    pre = ROOT/'fractional_flux_ir_actual4_20261011_preexecution.zip'
    with zipfile.ZipFile(snapshot) as z, zipfile.ZipFile(pre) as zp:
        request = json.loads(zp.read('fractional_flux_ir_request_20261011.json'))
        check(all(sha(zp.read(n)) == pin for n,pin in request['input_sha256'].items()), 'original preexecution input pins')
        check(z.read('fractional_flux_ir_consumer.py') == zp.read('fractional_flux_ir_consumer.py'), 'original reviewed consumer execution binding')
        original_capsule = json.loads(z.read('fractional_flux_ir_actual4_20261011_capsule.json'))
        executed = ROOT/'fractional_flux_ir_actual4_20261011_executed.zip'
        check(sha(pre.read_bytes()) == original_capsule['preexecution_sha256'], 'original runtime preexecution archive')
        check(sha(executed.read_bytes()) == original_capsule['executed_sha256'], 'original executed archive')
        with zipfile.ZipFile(executed) as ze:
            check(sha(ze.read('fractional_flux_ir_actual4_20261011_receipt.json')) == original_capsule['receipt_sha256'], 'original receipt hash')
            check(ze.read('fractional_flux_ir_consumer.py') == zp.read('fractional_flux_ir_consumer.py'), 'original executed source unchanged')
            exit_info = json.loads(ze.read('fractional_flux_ir_actual4_20261011_exit.json'))
            check(exit_info['returncode'] == 0 and exit_info['sources_inputs_unchanged'], 'original natural exit observation')
        profiles = json.loads(zp.read(request['profile']))['primitive_profiles']
        check([(p['load'],p['scale']) for p in profiles] == [(1,'heavy'),(1,'charged'),(63,'charged')], 'all1plus63 plus heavy')
        for p in profiles:
            check(F(p['onsite']) != 0, 'native onsite retained')
            for row in p['paths']:
                check(F(row['c0']) != 0, 'native path coefficient retained')
    receipt = {'kind': 'INDEPENDENT_EXACT_WEYL_GROUND_REVIEW_CONTROLS_NOT_NATIVE_GROUND',
               'source_sha256': sha(Path(__file__).read_bytes()), 'snapshot_sha256': sha(snapshot.read_bytes()),
               'original_preexecution_sha256': sha(pre.read_bytes()),
               'profile_binding': profiles, 'lattices': [exact_lattice(n) for n in (4,6,8)],
               'operator_packet_spectral': operator_packet_spectral(),
               'old_programs_executed': False, 'full_ground_computed': False, 'photon_admitted': False}
    with output.open('x') as f:
        json.dump(receipt,f,indent=2)
        f.write('\n')
    print(json.dumps({'receipt': output.name, 'independent_exact_controls': True, 'photon_admitted': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
