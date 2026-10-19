#!/usr/bin/env python3
"""NEW all-centre thermal bridge on the frozen bd4d Wilson/Fock bundle.

Charge-one matter kernels are translated, NOT the complete Hamiltonian: the
colour J potential is retained.  The original photon/receiver is not silently
identified with its translated counterpart.  The executable constructs both,
prices their difference at the actual bounded CP consumer, and retains all
centre classes and all eight statistical sectors.  No interacting-vacuum or
all-orders scattering closure is asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parent
SNAPSHOT = {p.stem: p.read_bytes() for p in ROOT.glob('*.py')}
COMPILED: dict[str, str] = {}
FROZEN_ROOT = 'bd4d5e1d0e9fcfadef72cc9dba2cc4f7e6efacc6a577ee331fcd5b82574e5e49'


class RecordedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str):
        data = SNAPSHOT[fullname]
        COMPILED[fullname] = hashlib.sha256(data).hexdigest()
        return compile(data, self.path, 'exec')


class RecordedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None):
        if fullname not in SNAPSHOT:
            return None
        return importlib.util.spec_from_loader(fullname, RecordedLoader(fullname, str(ROOT/(fullname+'.py'))))


finder = RecordedFinder()
sys.meta_path.insert(0, finder)
import numpy as np
import covariant_root_transfer as root
import covariant_prepared_gauss_state as field
from covariant_induced_photon import Parameters, phi_frame
sys.meta_path.remove(finder)
# Root temporarily installs its own exact-byte loader; record those modules too.
for name, data in root.SOURCE_BYTES_AT_IMPORT.items():
    stem = Path(name).stem
    if data != SNAPSHOT[stem]:
        raise ArithmeticError('Inconsistent transitive source snapshots')
    COMPILED[stem] = hashlib.sha256(data).hexdigest()
COMPILED[Path(__file__).stem] = hashlib.sha256(SNAPSHOT[Path(__file__).stem]).hexdigest()
if COMPILED['covariant_root_transfer'] != FROZEN_ROOT:
    raise ArithmeticError('Wrong frozen Root ABI')


def check(ok: bool, reason: str) -> None:
    if not ok:
        raise ArithmeticError(reason)


def centre_graph(size: int = 3) -> dict[str, Any]:
    """T^3 cochain discriminator, including NONFLAT centres and gauge cocycle.

    If d(z-z')=0 and all three based periods vanish, path integration from a
    basepoint produces g with z-z'=dg.  Every path difference is a sum of
    plaquette boundaries and the three fundamental cycles.  This works over
    Z/4, not just over a field; no division or false mod4 rank argument.
    """
    sites = list(itertools.product(range(size), repeat=3))
    index = {x: i for i, x in enumerate(sites)}
    def neighbor(x, mu):
        y = list(x); y[mu] = (y[mu]+1) % size
        return index[tuple(y)]
    def features(z):
        faces = []
        for i, x in enumerate(sites):
            for mu, nu in itertools.combinations(range(3), 2):
                faces.append(int((z[i, mu]+z[neighbor(x, mu), nu]-z[neighbor(x, nu), mu]-z[i, nu]) % 4))
        loops = []
        for mu in range(3):
            total = 0
            for k in range(size):
                x = [0, 0, 0]; x[mu] = k
                total += z[index[tuple(x)], mu]
            loops.append(int(total % 4))
        return tuple(faces+loops)
    rng = np.random.default_rng(222104)
    errors = []
    cocycle_errors = []
    # Actual integer Gauss characters on a graph with all cycle fluxes retained.
    E = rng.integers(-3, 4, size=(len(sites), 3))
    divergence = E.sum(axis=1).copy()
    for i, x in enumerate(sites):
        for mu in range(3):
            divergence[neighbor(x, mu)] -= E[i, mu]
    for _ in range(48):
        z = rng.integers(0, 4, size=(len(sites), 3))
        g = rng.integers(0, 4, size=len(sites))
        dg = np.array([[g[i]-g[neighbor(x, mu)] for mu in range(3)] for i, x in enumerate(sites)])
        errors.append(features(z) != features((z+dg) % 4))
        cocycle_errors.append(abs(1j**int(np.sum(E*dg))-1j**int(divergence@g)))
    seam = np.zeros((len(sites), 3), dtype=int)
    for i, x in enumerate(sites):
        if x[0] == size-1:
            seam[i, 0] = 1
    curl = np.zeros_like(seam); curl[0, 0] = 1
    check(not any(errors) and max(cocycle_errors) < 1e-12, 'Centre quotient or integer background cocycle failed')
    check(features(seam)[-3:] == (1, 0, 0) and not any(features(seam)[:-3]), 'Noncontractible quarter seam failed')
    check(any(features(curl)[:-3]), 'Nonflat plaquette positive control failed')
    return dict(size=size, sites=len(sites), links=3*len(sites), gauge_quotient_classes_exact=str(4**(2*len(sites)+1)),
        native_gauge_feature_errors=sum(errors), integer_background_cocycle_max_error=max(cocycle_errors),
        quarter_seam_features=features(seam), nonflat_link_features=features(curl),
        quotient_injection_proof='Equal ALL plaquette curvatures and three based H1 periods imply closed zero-period difference; integrate on paths to obtain a vertex Z4 coboundary. Cube faces and H1 generate path relations on periodic cubic T3. Gauge redundancy 4^(V-1) cancels on physical quotient; independent classes 4^(E-V+1)=4^(2V+1).',
        background_cocycle='U_delta exp(i E.A)=exp(i E.delta)exp(i E.A). For delta=pi/2 dg, E.dg=(div E).g. The Fock/Ref charge phase cancels this by integer Gauss, including background b: rho=sum Q n + Qref nref-b. Ref4 background contributes z^4=1, never replace it by Ref0. No loss of fluctuating occupations or cycle characters.',
        statistical_sectors_retained=8)


def quartic_native(gamma: np.ndarray) -> dict[str, Any]:
    """Actual four-spin Wilson determinant on the same open 2^4 box as Photon."""
    sites = list(itertools.product(range(2), repeat=4)); index = {x: j for j, x in enumerate(sites)}
    rng = np.random.default_rng(222)
    A = rng.uniform(-1, 1, (len(sites), 4)); z = rng.integers(0, 4, A.shape)
    delta = z*math.pi/2
    def hopping(values, centres, charge):
        h = np.zeros((64, 64), complex)
        for x, site in enumerate(sites):
            for mu in range(4):
                if site[mu] == 1:
                    continue
                other = list(site); other[mu] += 1; y = index[tuple(other)]
                link = (1j**int(centres[x, mu]))*np.exp(1j*charge*values[x, mu])
                h[4*x:4*x+4, 4*y:4*y+4] += (np.eye(4)-gamma[mu])*link
                h[4*y:4*y+4, 4*x:4*x+4] += (np.eye(4)+gamma[mu])*link.conjugate()
        return h
    def curvature(values):
        rows = []
        for x, site in enumerate(sites):
            for mu, nu in itertools.combinations(range(4), 2):
                if site[mu] or site[nu]:
                    continue
                xm, xn = list(site), list(site); xm[mu] += 1; xn[nu] += 1
                rows.append(values[x, mu]+values[index[tuple(xm)], nu]-values[index[tuple(xn)], mu]-values[x, nu])
        return np.array(rows)
    hc = hopping(A, z, 1); shifted = hopping(A+delta, np.zeros_like(z), 1)
    zero = hopping(np.zeros_like(A), np.zeros_like(z), 1)
    cost = float(np.sum(1-np.cos(curvature(A+delta))))
    rows = []
    for kappa in (.005, .01, .02):
        _, logc = np.linalg.slogdet(np.eye(64)-kappa*hc)
        _, logs = np.linalg.slogdet(np.eye(64)-kappa*shifted)
        _, log0 = np.linalg.slogdet(np.eye(64)-kappa*zero)
        action = float(log0-logc); leading = 16*kappa**4*cost
        tail = 2*(64/6)*(8*kappa)**6/(1-(8*kappa)**2)
        check(abs(action-leading) <= tail, 'Quartic centre holonomy did not reach physical determinant')
        rows.append(dict(kappa=kappa, actual_action=action, quartic_plaquette_action=leading, uniform_tail=tail, absorption_error=float(abs(logc-logs))))
    Q2 = hopping(A, z, 2); wrong2 = hopping(A+delta, np.zeros_like(z), 2)
    epsilon = 1e-5; direction = np.zeros_like(A); direction[0, 0] = epsilon
    jc = (hopping(A+direction, z, 1)-hopping(A-direction, z, 1))/(2*epsilon)
    js = (hopping(A+delta+direction, np.zeros_like(z), 1)-hopping(A+delta-direction, np.zeros_like(z), 1))/(2*epsilon)
    q2error = float(np.linalg.norm(Q2-wrong2))
    check(bool(np.linalg.norm(hc-shifted) < 1e-12 and np.linalg.norm(jc-js) < 1e-8 and q2error > 1), 'Charge/current discriminator failed')
    original_curl = np.sin(curvature(A)); transformed_curl = np.sin(curvature(A+delta))
    return dict(open_box=[2]*4, actual_nonflat_curvatures=(curvature(delta)/(math.pi/2)).round().astype(int).tolist(),
        full4spin_hopping_absorption_error=float(np.linalg.norm(hc-shifted)), actual_link_current_absorption_error=float(np.linalg.norm(jc-js)),
        quartic_rows=rows, wrong_Q2_active_aux1_absorption_error=q2error,
        original_vs_transformed_Curl_difference=float(np.linalg.norm(original_curl-transformed_curl)),
        Q2_obstruction='Aux1 fixes exp(i delta)=z. Active Q2 then exp(2i delta)=z², not z for z=i. Qmod4=2 matter occupation/Ref background is not centre-equivalent to charge1. No charge2 admission.',
        exact_full_kernel_scope='Pointwise fundamental Wilson blocks with arbitrary Phi and SU4 links: z U exp(i A)=U exp(i(A+arg z)). Primitive even lift, determinant and functional calculus inherit equality. It is not equality of J(z)+matter with J(0)+matter.')


def reference_quarter_control() -> dict[str, Any]:
    """Literal primitive Ref4->3 source versus an illegal quarter-charge hole."""
    model = root.BosonicEdgeModel(8, ((0, 1), (0, 2), (0, 3), (0, 4),
                                    (4, 5), (4, 6), (4, 7), (1, 5)))
    source = sum((model.bilinear(c, c+4, path=tuple([c+4]+([4] if c else [])+[0]+([c] if c else [])))
                  for c in range(4)), np.zeros_like(model.identity))
    na = sum((model.n(c) for c in range(4)), np.zeros_like(model.identity))
    nr = sum((model.n(c) for c in range(4, 8)), np.zeros_like(model.identity))
    physical_charge = na+nr
    wrong_quarter_charge = na+nr/4
    actual = float(np.linalg.norm(physical_charge@source-source@physical_charge))
    wrong = float(np.linalg.norm(wrong_quarter_charge@source-source@wrong_quarter_charge))
    alpha = math.pi/2
    active = np.exp(1j*alpha*np.diag(na))
    reference = np.exp(1j*alpha*np.diag(nr))
    body = active[:, None]*source*active.conj()[None, :]
    refhole = reference[:, None]*source*reference.conj()[None, :]
    body_error = float(np.linalg.norm(body-1j*source))
    hole_error = float(np.linalg.norm(refhole+1j*source))
    quarter = np.exp(1j*alpha*np.diag(nr)/4)
    wrong_hole = quarter[:, None]*source*quarter.conj()[None, :]
    wrong_hole_error = float(np.linalg.norm(wrong_hole+1j*source))
    check(actual < 1e-12 and body_error < 1e-12 and hole_error < 1e-12 and wrong > 1 and wrong_hole_error > 1,
          'Actual primitive reference does not discriminate quarter-charge reassignment')
    return dict(full_primitive_dimension=model.dimension, actual_source_U1_commutator=actual,
        actual_quarter_centre_body_phase_error=body_error, actual_quarter_centre_Refhole_phase_error=hole_error,
        wrong_Ref_charge_onequarter_Gauss_commutator=wrong, wrong_Ref_quarter_hole_centre_error=wrong_hole_error,
        wrong_fractional_Wilson_Haar_periodicity_error=abs(np.exp(2j*math.pi/4)-1),
        neutral_reference_background=4, source_reference_occupation=3, actual_Ref_operator_charge=1,
        meaning='Quarter refers to z=i central holonomy, NOT charge1/4. The fundamental Ref annihilator has charge1, so its hole transforms by -i; Ref4 background remains unchanged.')


def toron_shift_control(cutoff: int = 48) -> dict[str, Any]:
    """Execute actual bounded Haar toron-window Grams in all 4^3 toron classes."""
    beta, L, Nt, R = 20., 4, 8, .7
    n = np.arange(-cutoff, cutoff+1); energy = n*n/(2*beta*L)
    weights = np.exp(-Nt*energy); weights /= weights.sum()
    nodes, quadrature = field.roots_legendre(240)
    window = (1-nodes*nodes)**2; diff = n[:, None]-n[None, :]
    def matrix(alpha):
        # f(theta+alpha) has support centred at -alpha, including the seam.
        theta = R*nodes-alpha
        return R*np.einsum('q,ijq,q->ij', quadrature, np.exp(-1j*diff[:, :, None]*theta), window)/(2*math.pi)
    F = matrix(0.); base = (F*weights)@F.conj().T; gram = float(np.trace(base).real)
    base /= gram; rows = []
    for centre in range(4):
        alpha = math.pi*centre/2; d = np.exp(1j*n*alpha)
        shifted = matrix(alpha)
        density = (shifted*weights)@shifted.conj().T
        actual_gram = float(np.trace(density).real); density /= actual_gram
        correct = d[:, None]*base*d.conj()[None, :]
        # Original window retained as a different observable.
        original_window = float(np.trace(density@F@F).real)
        shifted_window = float(np.trace(density@shifted@shifted).real)
        error = float(np.linalg.norm(shifted-d[:, None]*F*d.conj()[None, :]))
        check(bool(error < 1e-12 and np.linalg.norm(density-correct) < 1e-12 and abs(actual_gram/gram-1) < 1e-12),
              'Actual toron window/Haar CP Gram translation failed')
        rows.append(dict(centre=centre, actual_Gram=actual_gram, relative_Gram_error=abs(actual_gram/gram-1),
            literal_Haar_window_covariance_error=error, original_window_probability=original_window,
            transformed_window_probability=shifted_window))
    all64: list[dict[str, Any]] = [dict(centre_torons=z, positive_joint_filter_Gram=float(np.prod([rows[c]['actual_Gram'] for c in z])))
             for z in itertools.product(range(4), repeat=3)]
    check(all(r['positive_joint_filter_Gram'] > 0 for r in all64), 'Toron CP deleted a centre class')
    return dict(beta=beta, L=L, time_slices=Nt, window_radius=R, Fourier_cutoff=cutoff,
        one_toron=rows, all64_toron_classes=all64, unfiltered_thermal_modes_retained=len(n),
        continuum_one_window_Haar_probability=128*R/(315*math.pi),
        construction='Literal multiplier f(theta+arg z), integer Fourier electric basis. H=n²/(2 beta L) commutes translation. Thermal mode weights remain positive. Finite Fourier projection is stated, no wrapping of electric charge.')


def shifted_range(gamma: np.ndarray, grid: int, cutoff: int, charge: int, alpha: float):
    base = field.native_range(gamma, .09, grid, cutoff, charge)
    states, middle = base['states'], [c for c in base['states'] if field.rho_sites(c, charge) == (0, 0, 0, 0)]
    samples = []
    for theta in 2*math.pi*np.arange(grid)/grid:
        half, pref, _, _, _ = field.square_wilson(gamma, .09, float(theta+alpha))
        samples.append(math.sqrt(pref)*np.linalg.det(half[np.asarray(states)[:, None, :, None], np.asarray(middle)[None, :, None, :]]))
    coefficients = np.fft.fft(np.asarray(samples), axis=0)/grid
    V = np.array([coefficients[n % grid, i, :] for i, n, _ in base['physical']])
    phases = np.exp(1j*alpha*np.array([n for _, n, _ in base['physical']]))
    U = np.diag(phases)
    error = float(np.linalg.norm(V-U@base['V'])/np.linalg.norm(V))
    return base, U, V, error


def shifted_current(gamma: np.ndarray, space, grid: int, alpha: float) -> list[np.ndarray]:
    """Independently differentiate the actual shifted Fock Hamiltonian and detB."""
    epsilon = 2e-4; jets = []
    for theta in 2*math.pi*np.arange(grid)/grid:
        _, p0, h0, _, _ = field.square_wilson(gamma, .09, float(theta+alpha))
        _, pp, hp, _, _ = field.square_wilson(gamma, .09, float(theta+alpha+epsilon))
        _, pm, hm, _, _ = field.square_wilson(gamma, .09, float(theta+alpha-epsilon))
        current = field.matter_operator((hp-hm)/(2*epsilon), space['states'])
        current -= (math.log(pp)-math.log(pm))/(2*epsilon)*np.eye(len(space['states']))
        contact = field.matter_operator((hp+hm-2*h0)/epsilon**2, space['states'])
        contact -= (math.log(pp)+math.log(pm)-2*math.log(p0))/epsilon**2*np.eye(len(space['states']))
        jets.append((current, contact))
    coefficients = np.fft.fft(np.asarray(jets), axis=0)/grid
    return [np.array([[coefficients[(n-m) % grid, order, i, j] for j, m, _ in space['physical']]
                      for i, n, _ in space['physical']]) for order in range(2)]


def native_allsector(gamma4: np.ndarray, grid: int, cutoff: int, steps: int) -> dict[str, Any]:
    """Actual integer-Gauss/nonzero-range CP state, no occupation-state replacement.

    Reuses the consumer's square Wilson minors, physical basis, canonical source
    and Ref contraction.  The separate primitive/full-SU4 checks bind its spin
    convention to Root.  This finite square is a discriminator, NOT the macro
    cubic-volume implementation or a full-SU4 approximation certificate.
    """
    gamma = field.two_spin_frame(gamma4)
    neutral = field.native_range(gamma, .09, grid, cutoff, 0)
    charged = field.native_range(gamma, .09, grid, cutoff, 1)
    Un, Uc = neutral['u'], np.kron(np.eye(2), charged['u'])
    scale = neutral['values'][0]; ln = neutral['values']/scale; lc = np.tile(charged['values']/scale, 2)
    w = ln**steps; w /= w.sum(); rho = np.diag(w)
    holonomy = np.zeros((len(neutral['physical']),)*2, complex)
    for j, (i, n, _) in enumerate(neutral['physical']):
        row = neutral['basis_index'].get((i, n+1))
        if row is not None:
            holonomy[row, j] = 1
    raw_B = (holonomy-holonomy.conj().T)/(2j)
    B = Un.conj().T@raw_B@Un; hn = np.diag(np.sqrt(ln)); hc = np.diag(np.sqrt(lc))
    photon = hn@B.conj().T@rho@B@hn; gph = float(np.trace(photon).real); photon /= gph
    packet = np.array([.1, .8, .4, .1]); packet /= np.linalg.norm(packet)
    coeff = field.source_coefficients(gamma, .09, packet, 1)
    raw_S, defect = field.source_matrix(neutral, charged, coeff)
    S = Uc.conj().T@raw_S@Un; state = hc@S@photon@S.conj().T@hc
    gs = float(np.trace(state).real); state /= gs
    J, contact = field.native_current(gamma, .09, grid, neutral)
    # A fixed charged physical receiver is NOT secretly translated. Its two
    # separated electric-flux components distinguish a quarter-loop phase.
    physical_density = Uc@state@Uc.conj().T
    dimension = len(charged['physical'])
    body_density = physical_density[:dimension, :dimension]+physical_density[dimension:, dimension:]
    pairs = [(j, charged['basis_index'][i, n+1]) for j, (i, n, _) in enumerate(charged['physical'])
             if (i, n+1) in charged['basis_index']]
    check(bool(pairs), 'Physical receiver has no coherent cycle pair')
    pair = max(pairs, key=lambda pair: abs(body_density[pair[0], pair[1]]))
    vector = np.zeros(dimension, complex)
    vector[pair[0]] = 1/math.sqrt(2)
    vector[pair[1]] = np.exp(-1j*np.angle(body_density[pair[0], pair[1]]))/math.sqrt(2)
    receiver_raw = np.kron(np.eye(2), np.outer(vector, vector.conj()))
    rows = []; covariance = []; source_error = []; original_error = []
    native_p = steps+1; Jscale = .11
    centre_weights = np.exp(-native_p*Jscale*(1-np.cos(math.pi*np.arange(4)/2)))
    centre_weights /= centre_weights.sum()
    range_cache = {}
    current_cache = {}
    # All eight statistical labels remain; their square pullbacks are +/-
    # boundary seams. No selected label is relabelled as an ALL-sector result.
    for eta in itertools.product((0, 1), repeat=3):
        for z in range(4):
            key = (2*eta[0]+z) % 4
            alpha = key*math.pi/2
            if key not in range_cache:
                range_cache[key] = (shifted_range(gamma, grid, cutoff, 0, alpha),
                                    shifted_range(gamma, grid, cutoff, 1, alpha))
                current_cache[key] = shifted_current(gamma, neutral, grid, alpha)
            (nspace, dn, Vn, en), (cspace, dc0, Vc, ec) = range_cache[key]
            dc = np.kron(np.eye(2), dc0)
            uzn, uzc = dn@Un, dc@Uc
            # Positive transfer including literal bare-Haar kernel, NOT H=onebody.
            Kn = Vn@Vn.conj().T/scale; Kc = np.kron(np.eye(2), Vc@Vc.conj().T/scale)
            covariance.append(float(np.linalg.norm(Kn-dn@(neutral['V']@neutral['V'].conj().T/scale)@dn.conj().T)))
            physical_source = np.zeros_like(raw_S)
            for j, (_, _, E) in enumerate(nspace['physical']):
                for row, (_, _, F) in enumerate(cspace['physical']):
                    # Ref is at vertex0. The literal Wilson string supplies
                    # exp(i alpha*(F3-E3)); Ref4->3 itself is NOT a quarter charge.
                    phase = np.exp(1j*alpha*(F[3]-E[3]))
                    physical_source[row::len(cspace['physical']), j] = raw_S[row::len(cspace['physical']), j]*phase
            source_error.append(float(np.linalg.norm(physical_source-dc@raw_S@dn.conj().T)))
            Sz = uzc.conj().T@physical_source@uzn
            transformed_B = dn@raw_B@dn.conj().T
            Bz = uzn.conj().T@transformed_B@uzn
            phz = hn@Bz.conj().T@rho@Bz@hn; gzph = float(np.trace(phz).real); phz /= gzph
            sz = hc@Sz@phz@Sz.conj().T@hc; gzsource = float(np.trace(sz).real); sz /= gzsource
            Oz = uzc.conj().T@(dc@receiver_raw@dc.conj().T)@uzc
            original_B = uzn.conj().T@raw_B@uzn
            original_error.append(float(np.linalg.norm(original_B-B)))
            raw_Jz, raw_Cz = current_cache[key]
            Jz, Cz = uzn.conj().T@raw_Jz@uzn, uzn.conj().T@raw_Cz@uzn
            rows.append(dict(statistical_sector=eta, centre_class=z, retained_thermal_weight=float(centre_weights[z]/8),
                Haar_range_covariance_error=max(en, ec), neutral_nonzero_rank=len(ln), charged_nonzero_rank=len(lc),
                translated_photon_Gram=gzph, translated_source_Gram=gzsource,
                photon_Gram_relative_error=abs(gzph/gph-1), source_Gram_relative_error=abs(gzsource/gs-1),
                source_density_covariance_error=float(np.linalg.norm(sz-state)),
                translated_receiver=float(np.trace(sz@Oz).real), original_receiver=float(np.trace(sz@(uzc.conj().T@receiver_raw@uzc)).real),
                translated_current=float(np.trace(phz@Jz).real), current_squared=float(np.trace(phz@Jz@Jz).real),
                translated_contact=float(np.trace(phz@Cz).real),
                actual_Fock_current_covariance_error=float(np.linalg.norm(Jz-J)),
                actual_Fock_contact_covariance_error=float(np.linalg.norm(Cz-contact)),
                density_trace=float(np.trace(sz).real), positive_kernel_min_nonzero=float(min(ln[-1], lc[-1])),
                actual_kernel_norm=float(np.linalg.norm(Kc))))
    off = []
    for alpha in (.37, -.81):
        _, _, _, error = shifted_range(gamma, grid, cutoff, 0, alpha)
        off.append(dict(phase=alpha, actual_Haar_range_covariance_error=error))
    check(max(r['Haar_range_covariance_error'] for r in rows) < 1e-10, 'Quarter seam nonzero-Haar range failed')
    check(max(r['source_density_covariance_error'] for r in rows) < 1e-10 and max(source_error) < 1e-12, 'Source/current CP construction failed')
    check(max(r['actual_Fock_current_covariance_error'] for r in rows) < 1e-8
          and max(r['actual_Fock_contact_covariance_error'] for r in rows) < 1e-5, 'Actual shifted Fock/current contact covariance failed')
    check(max(r['photon_Gram_relative_error'] for r in rows) < 1e-10 and max(r['source_Gram_relative_error'] for r in rows) < 1e-10, 'Actual CP Grams changed under translation')
    check(max(original_error) > .1*np.linalg.norm(B) and defect == 0 and all(r['retained_thermal_weight'] > 0 for r in rows), 'Original-receiver obstruction or all-sector retention disappeared')
    check(max(r['actual_Haar_range_covariance_error'] for r in off) < 1e-8, 'Off-cycle translation/FFT aliasing failed')
    receiver_difference = max(abs(r['translated_receiver']-r['original_receiver']) for r in rows)
    check(receiver_difference > 1e-12, 'Original physical receiver fails to discriminate a centre seam')
    return dict(native_geometry='Actual consumer invariant2spin square, all centre cycle classes x eight retained statistical pullbacks',
        grid=grid, electric_cutoff=cutoff, steps=steps, native_centre_Jscale=Jscale, sectors=rows,
        centre_partition_exact=float(sum(np.exp(-native_p*Jscale*(1-np.cos(math.pi*np.arange(4)/2))))),
        all32_weight_sum=sum(r['retained_thermal_weight'] for r in rows),
        receiver_definition='Fixed native coherent projector on a same-occupation adjacent integer cycle-flux pair, selecting the largest actual baseline source coherence before all centre runs; tensor identity on Ref. Not a replacement for the macro packet receiver.',
        receiver_flux_pair=pair, original_vs_transformed_receiver_max_difference=receiver_difference,
        actual_Gauss_defect=defect, primitive_source_string_covariance_error=max(source_error),
        actual_Haar_transfer_covariance_error=max(covariance), off_cycle_controls=off,
        untransformed_Curl_max_operator_difference=max(original_error),
        untransformed_Curl_relative_operator_difference=max(original_error)/float(np.linalg.norm(B)),
        allsector_photon_Gram=sum(r['retained_thermal_weight']*r['translated_photon_Gram'] for r in rows),
        allsector_source_Gram=sum(r['retained_thermal_weight']*r['translated_source_Gram'] for r in rows),
        allsector_receiver=sum(r['retained_thermal_weight']*r['translated_receiver'] for r in rows),
        original_receiver_retained=sum(r['retained_thermal_weight']*r['original_receiver'] for r in rows),
        construction='Direct sum rho_z ∝ exp(-(steps+1)J(z)) U_z rho_base U_z†. Every weight strictly positive. B_z=U_n B U_n†, S_z=U_ch S U_n†, O_z=U_ch O U_ch†; O_original is separately evaluated, not modified.',
        limitation='Only one cycle is sampled by this spatial square; eta_y,eta_z are retained degeneracy labels, not claims of three resolved native cubic photon modes. Macro T3 quotient and all8stats use the exact graph argument and Root/Field reference. Finite cutoff/range boundary retained exactly.')


def thermal_consumer_bound(p: Parameters, *, photon_cap: float = 2e6, source_cap: float = 8.,
                           photon_gram: float = .9, source_gram: float = .5,
                           selected_probability: float = .14443728261154) -> dict[str, Any]:
    """Consumer-ready rigorous logarithmic bound, retaining enormous multiplicity.

    J has one unit per nonzero face/period. Root's 1-Re(i^c) is >=this.
    Injective class features have constraints, so summing independent features
    OVERCOUNTS, never drops entropy. Bounds use exp/log inequalities and remain
    meaningful when exp(-p) underflows. The exact weight formula, not a rounded
    zero, is the physical construction.
    """
    volume = p.spatial_size**3; count = 3*volume+3; exponent = p.time_slices+1
    R = p.spatial_size*1e-50
    toron_probability = (128*R/(315*math.pi))**3
    logweight = math.log10(3*count)-exponent/math.log(10)
    # Z-1 <= x exp(x), x=3 M exp(-p); x<1 gives <=2x.
    check(logweight < -1, 'Small-centre branch required for stated factor2 upper')
    raw_tail_log10 = math.log10(2)+logweight
    retained_gram = toron_probability*photon_gram*source_gram*selected_probability
    cp_norm = photon_cap*source_cap
    cp_amplification_log10 = 2*math.log10(cp_norm)-math.log10(retained_gram)
    source_tail_log10 = min(0., raw_tail_log10+cp_amplification_log10)
    purifier_log10 = math.log10(2)+source_tail_log10/2
    family = root.charge_weighted_phi_budget(root.RootParameters(normal_width_power=1500, tube_radius_power=1600), p)
    root_trace_log10 = math.log10(2)+cp_amplification_log10+family['projective_slab_density_trace_error_log10']
    return dict(spatial_volume_exact=str(volume), centre_features_exact=str(count), all_quotient_classes_exact=f'4^(2*{volume}+1)',
        inverse_temperature=exponent, statistical_sectors_retained=8,
        actual_J='sum_ALL3V_faces(1-Re z_face)+sum3basedH1loops(1-Re z_loop), unit gap; all sectors retained',
        J_source_binding=dict(frozen_source_sha256=FROZEN_ROOT, function='exact_chart_ground',
            normal_frequency=root.RootParameters().normal_frequency,
            chart_interpretation='Identical central radial chart copies; J is the discrete central face/period value. Its declared smooth continuation outside the charts is NOT silently equated to this reference.',
            new_centre_penalty_added=False),
        exact_partition_bound='Z_J(p)<= (1+2exp(-p)+exp(-2p))^(3V+3) <= (1+3exp(-p))^(3V+3)',
        partition_proof='Each class injects into face+period feature vector. J additive in these features. Sum over the image <= sum over ALL independent vectors. A feature contributes1,exp(-p),exp(-2p),exp(-p); positive constrained sums preserve inequality. Charge1 Haar-translated matter partitions equal within each statistical sector; statistical sector partition factors cancel without discarding their eight positive weights.',
        log10_Zminus1_upper=raw_tail_log10, log10_nonidentity_centre_probability_upper=raw_tail_log10,
        CP=dict(photon_operator_cap=photon_cap, source_operator_cap=source_cap,
            actual_photon_Gram_lower=photon_gram, actual_source_Gram_lower=source_gram,
            toron_filter_probability=toron_probability, selected3_probability=selected_probability,
            actual_retained_joint_Gram_lower=retained_gram, amplification_log10=cp_amplification_log10,
            postsource_centre_probability_log10_upper=source_tail_log10,
            original_vs_transformed_receiver_trace_error_log10_upper=math.log10(2)+source_tail_log10,
            original_receiver_vs_flat_conditional_purifier_error_log10_upper=purifier_log10),
        full_Root_reference_trace_price_log10=root_trace_log10,
        full_Root_reference_proof='Full K need not intertwine centre translation. Add the frozen Root normalized projective/Schatten trace error after actual CP norm²/Gram. This is independent of thermal centre retention; no endpoint-only Dcross damping or globalTmin invented.',
        operator_API=dict(unitary='(U_z psi)(A)=psi(A+pi*z/2); Fourier matrix diag exp(i*pi*E.z/2)',
            source='S_z=U_ch,z S_pre U_n,z†, literal U1+SU4 strings, transported positive-band projectors and charge1 Ref4->3',
            photon='B_z(A)=B125clip(A+pi*z/2), with coexact Curl(A+pi*z/2), NOT Curl(A)',
            toron='f_z(A)=f_three_physical_torons(A+pi*z/2); translation commutes positive functional calculus/range projection but not the unshifted window',
            receiver='O_z=U_ch,z O_original U_ch,z† is a separate explicit observable; original O_original retained and centre tail priced',
            Pe='Pe_z=U_ch,z Pe_base U_ch,z† for reference; full actual Pe is separately controlled by Root relative kernel'),
        full_equivalence_obstruction='Matter charge1 bundle equivalence is true; full J-weighted law is NOT equivalent to centre-zero law: spectrum carries J(z), nonflat photon curvature and toron/source windows shift, and frozen cross-block D covariance is not assumed. This construction retains explicit weights and transformed operators rather than claiming full equivalence.',
        admitted_scope='All-centre chart/reference CP consumer plus frozen Root relative error. Actual Field gph>=.9 and gsource>=.5 are required inputs, not proved by this entropy bound. Outside angular control, full interacting vacuum and all-orders scattering remain separate.')


def full_haar_proof() -> dict[str, Any]:
    return dict(unitary='Product compact U1 Haar is invariant under EVERY link translation, including nonflat z. On full colour-link Haar the combined map W_z psi(X,U,A)=psi(X,z^-1 U,A+arg z) is exactly unitary and leaves every charge1 U exp(iA) unchanged. On identified central radial chart copies its field part is U_z. No small-chart or onebody-only unitarity argument.',
        Gauss='Left/right local SU4 action commutes central link multiplication. U1 gauge additions commute link translations. U_delta therefore commutes both orthogonal Gauss projectors with unchanged integer background and Ref. Functional calculus gives U ran(K)=ran(UKU†), including actual nullspace/nonzero range; no global lower eigenvalue.',
        primitive='Fundamental onsite Gamma(V), root.native_lift and primitive even path polynomials are unchanged. Every charged matter Wilson link appears as U exp(i QA). For ALL Q=1 species, z is absorbed identically. SU4 adjoint Phi and transported connection are centre-invariant. Determinant, minors, literal Haar middle projector and boundary positive halfsteps inherit equality.',
        cocycle='On a vertex-centre change g, E.dg=div(E).g and rho=sum Qn+Qref nref-b. Keep the integer background character exp(-i b.g*pi/2) and actual Ref occupation. Global consistency requires total fundamental centre residue0, as frozen Root proves on its even spatial box; it is not removed by choosing a tree.',
        external_sources='A separated charge1 source path changes E by the literal integer path; U_ch S U_n† multiplies each matrix element by exp(i delta.(Eout-Ein)). This equals the product of translated Wilson links. FullRef4->3 has a fundamental hole, never a charge1/4 hole. Native primitive source checks all15 SU4 and U1 commutators.',
        obstruction='Translations do not eliminate colour J or preserve original geometric Curl/toron window. J weights retained explicitly. The outside/crossing law is not claimed invariant without its own intertwining proof; actual frozen Root error is paid at source/receiver.')


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, default=64)
    parser.add_argument('--cutoff', type=int, default=3)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--root-capsule', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    raw = args.root_capsule.read_bytes(); capsule = json.loads(raw)
    check(capsule['capsule']['source_sha256_at_IMPORT']['covariant_root_transfer.py'] == FROZEN_ROOT,
          'Selected input capsule is not the frozen Root execution')
    p = Parameters(); gamma, _ = phi_frame(p)
    primitive = root.native_case(root.RootParameters(normal_width_power=1500, tube_radius_power=1600), gamma)
    fullgauss = root.gauss_bundle_native()
    check(primitive['source_SU4_commutator'] < 1e-8 and primitive['source_EM_commutator'] < 1e-8, 'Primitive fullRef source lost Gauss')
    check(fullgauss['actual_all15_left_right_Gauss_generator_error'] < 1e-8, 'Actual full SU4 Gauss trial failed')
    graph = centre_graph(); quartet = quartic_native(gamma)
    reference_control = reference_quarter_control(); toron_control = toron_shift_control()
    native = native_allsector(gamma, args.grid, args.cutoff, args.steps)
    bounds = thermal_consumer_bound(p)
    unchanged = all((ROOT/(name+'.py')).read_bytes() == SNAPSHOT[name] for name in COMPILED)
    check(unchanged, 'Executed supplier changed during run; rerun, do not relabel receipt')
    result: dict[str, Any] = dict(plan='issue-222-induced-photon-delta13-continuation-20261008',
        assignment_revision='d13d7d28ebfd7cb8386e7442897c596481aeab95cd55def1a8b3a60c1e89f4dd',
        branch='NEW retained-centre physical thermal CP bridge; frozen Root untouched',
        full_Haar_bundle_proof=full_haar_proof(), quotient=graph, actual_quartic=quartet,
        actual_allsector_consumer=native, thermal_consumer=bounds,
        actual_Ref_quarter_controls=reference_control, actual_all64_toron_filter_Grams=toron_control,
        frozen_J_input_binding=capsule['actual_massive_chart_field']['massive_centre_sector_law'],
        primitive_fullRef_controls=primitive, actual_full_SU4_Gauss_control=fullgauss,
        input=dict(path=str(args.root_capsule), sha256=hashlib.sha256(raw).hexdigest(),
            capsule_top_level_fields=sorted(capsule), frozen_root_sha256=FROZEN_ROOT),
        compiled_source_hashes=COMPILED, captured_source_bytes={name: SNAPSHOT[name].decode() for name in COMPILED},
        compiled_source_unchanged=unchanged,
        runtime=dict(executable=sys.executable, executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            python=sys.version, numpy=np.__version__, platform=platform.platform()),
        elapsed_seconds=time.monotonic()-started,
        scope='Executed corrected all-sector physical chart/reference construction and priced frozen full-Root continuation; not full interacting vacuum/scattering closure.')
    args.output.write_text(json.dumps(portable(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), source_sha256=COMPILED[Path(__file__).stem],
        root_sha256=FROZEN_ROOT, input_sha256=result['input']['sha256'],
        actual_sector_weights_sum=native['all32_weight_sum'], photon_Gram=native['allsector_photon_Gram'],
        source_Gram=native['allsector_source_Gram'], original_Curl_obstruction=native['untransformed_Curl_max_operator_difference'],
        Q2_adverse=quartet['wrong_Q2_active_aux1_absorption_error'],
        postsource_centre_probability_log10=bounds['CP']['postsource_centre_probability_log10_upper'],
        elapsed_seconds=result['elapsed_seconds'])))
    print('Natural exit: full-Haar/Gauss bridge, retained centre thermal/source/current/receiver construction')


if __name__ == '__main__':
    main()
