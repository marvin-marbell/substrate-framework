#!/usr/bin/env python3
"""All-background Wilson-relative bounds and an improved admitted Maxwell family."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
import scipy
from scipy.linalg import block_diag, expm, lu_factor, lu_solve

from bulk_quantum_selection import BulkQuantumLaw
from noncompact_maxwell_phase import (
    first_source_response, native_jets, ordered_hessian, phase_constants,
)
from original_charged_principal import original_frame, portable
from original_compact_charge_bound import chiral_basis
from rotation_band import RotationBand
from original_massive_photon_locality import ContinuousGreen

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
REVISION = '14d4836b213e2bdca4c489ec23d8489fdb5ffcb76ebfa1a4fad87f7216b9e8c0'


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def close(a, b, label, tolerance=3e-9):
    error = float(np.linalg.norm(a-b)/max(1., np.linalg.norm(b)))
    require(error < tolerance, f'{label}: {error}')
    return error


def metric_budget(g):
    s = float(g[0, 0]-g[0, 1:]@np.linalg.solve(g[1:, 1:], g[1:, 0]))
    tilt = math.sqrt(max(0., float(g[0, 0])-s)/s)
    kv = math.sqrt(.3**2*float(np.trace(g[1:, 1:]))/(2*.2*.12*s))
    return dict(s=s, tilt=tilt, spatial=kv, kappa=kv+tilt,
                diagonal=np.diag(g).tolist())


def anchored_family(s_floor=.01, speed=.1, *, tight=False):
    """Evaluate the existing anchored theorem at sharper legal inputs/onset.

    Universal mode retains the predecessor's SAME Schur/Gram hypotheses.
    Tight mode additionally declares G00<=.43, Gjj<=.065, tilt<=.002.
    The finite consumer does not establish either infinite-family admission.
    """
    m, r, v = .2, .12, .3
    tstar = (2*math.sqrt(2)*math.sqrt((m+6*r)**2+3*v*v))**(2/3)
    g00 = .43 if tight else tstar
    gjj = .065 if tight else tstar
    # s <= G00. Without a tilt hypothesis, G00-s <= G00-s_floor.
    odd0 = (m+3*r)*(.002*math.sqrt(g00) if tight else math.sqrt(g00-s_floor))
    c_upper = (m+3*r)*math.sqrt(g00)
    ch = 3*(r*math.sqrt(g00)+v*math.sqrt(gjj))
    b = (m+3*r)*math.sqrt(g00)+ch
    mu = m*math.sqrt(s_floor)
    candidates = []
    for gf, xf in ((.25, .25), (.2, .5), (.15, .6), (.2, .6), (.25, .5)):
        gamma, xi = gf*mu, xf*mu
        alpha = math.log1p(xi/ch)
        va = odd0+ch*math.exp(alpha)
        qt = max(c_upper, math.log(2), gamma+8*va, gamma+4)
        # qt-c <= qt-(m+3r)*sqrt(s_floor): do not discard the true subtraction.
        mt = qt-(m+3*r)*math.sqrt(s_floor)
        delta = min(1/8, 1/(4*(b+mt+gamma+xi)))
        cbar = math.e/(delta*(mu-gamma-xi))
        p = 16/gamma*(1+2/math.expm1(alpha/2))**3
        jet = max((r*math.sqrt(g00)+v*math.sqrt(gjj))/2, 1.)
        hlow = 16*(2*cbar)**3*jet**2*math.exp(6*alpha)*p*p+8*(2*cbar)**2*jet*math.exp(4*alpha)*p
        h = mt*hlow+(16384*va*va+2048)/(qt-gamma)
        x4 = max(0., 4/alpha-1)
        ce0 = 8*h/(math.e**2*gamma**2)
        cb0 = 2*64**2*h*(1+x4)**4*math.exp(-alpha*x4)
        ce_base, cb_base = 2*ce0, 2*cb0
        seam_budget = max(ce_base, cb_base)/100
        tau = max(1., (alpha+1)/gamma)
        xa, xt = max(0., 48/alpha-1), max(0., 48/gamma-1)
        log_r = 20*math.log(2)+math.log(h)+alpha/2+12*math.log1p(xa)-alpha*xa/4+12*math.log1p(xt)-gamma*xt/4
        bs, bt = alpha/64, gamma/64
        def seam_log(n):
            x, y = -bs*n, -bt*tau*n*n
            return log_r+2*math.log(n)-math.log(8)+max(x, y)+math.log1p(math.exp(-abs(x-y)))
        onset = max(5, math.ceil(2/bs), math.ceil(math.sqrt(1/(bt*tau))))
        low, high = onset, onset
        while seam_log(high) > math.log(seam_budget):
            high *= 2
        while high-low > 1:
            middle = (low+high)//2
            if seam_log(middle) <= math.log(seam_budget):
                high = middle
            else:
                low = middle
        n0 = high
        ce, cb = ce_base+seam_budget, cb_base+seam_budget
        he = 16*max(ce, cb/speed**2, 1.)
        candidates.append(dict(s_floor=s_floor, G00_upper=g00, Gjj_upper=gjj,
            tilt_upper=.002 if tight else None, tight=tight, mu=mu, Ch=ch, B=b,
            onsite_odd_upper=odd0, c_upper=c_upper, gamma=gamma, xi=xi, alpha=alpha,
            qtail=qt, Mtail_upper=mt, Cbar=cbar, H=h, CE_bulk=ce_base, CB_bulk=cb_base,
            log_Rstar=log_r, seam_budget=seam_budget, N0_seam=n0, tau=tau,
            log_seam_at_N0=seam_log(n0), CE=ce, CB=cb, UE=1/(2*he), K=speed**2*he,
            bare_speed=speed, rho=max(ce/he, cb/(speed**2*he)),
            gamma_fraction=gf, xi_fraction=xf))
    return max(candidates, key=lambda c: c['UE'])


def primitive_cone_envelope(*, m_hat: float, r: float, v: float,
                            s_lower: float, g00_upper: float,
                            spatial_diagonal_upper: tuple[float, float, float],
                            tilt_root_upper: float,
                            xi_fraction: float) -> dict[str, float]:
    """Dimensionless aH kernel/moments; no physical coupling or metric admission.

    Sources are spatial link angle theta and b_hat=a*a0; time is t_hat=t/a.
    tilt_root_upper bounds sqrt(G00-s), not sqrt((G00-s)/s).
    All outputs are hatted except dimensionless site distances.
    """
    require(m_hat > 0 and r > 0 and s_lower > 0 and g00_upper >= s_lower,
            'positive hatted Wilson/Schur input bounds')
    require(all(value > 0 for value in spatial_diagonal_upper)
            and tilt_root_upper >= 0 and 0 < xi_fraction < .5,
            'positive spatial bounds and strict weighted-cone budget')
    roots = tuple(math.sqrt(value) for value in spatial_diagonal_upper)
    mu = m_hat*math.sqrt(s_lower)
    ch = sum(r*math.sqrt(g00_upper)+abs(v)*root for root in roots)
    odd_row = (m_hat+6*r)*tilt_root_upper+abs(v)*sum(roots)
    odd_jet = sum(r*tilt_root_upper+abs(v)*root for root in roots)
    jet = max(1., max((r*math.sqrt(g00_upper)+abs(v)*root)/2 for root in roots))
    xi = xi_fraction*mu
    alpha = math.log1p(xi/ch)
    radius = (odd_row+xi)/(math.sqrt(mu*mu-2*mu*xi+odd_row*odd_row)+mu-xi)
    require(radius < 1, 'weighted beta-cone graph stays strictly interior')
    cinv = (1+radius*radius)/(1-radius*radius)
    gamma = (mu-xi)/2
    t0 = max(1., math.log(2)/(mu-xi))
    point = 4*cinv**1.5
    p = 16/gamma*(1+2/math.expm1(alpha/2))**3
    triangle = 16*odd_row*(2*point)**3*jet**2*math.exp(6*alpha)*p*p
    contact = 8*odd_row*(2*point)**2*jet*math.exp(4*alpha)*p
    anchor_first = 16*(2*point)**2*jet*odd_jet*math.exp(4*alpha)*p
    anchor_contact = 8*(2*point)*odd_jet*math.exp(2*alpha)
    h = triangle+contact+anchor_first+anchor_contact
    x4 = max(0., 4/alpha-1)
    ce_bulk = 16*h/(math.e**2*gamma*gamma)
    cb_bulk = 4*64**2*h*(1+x4)**4*math.exp(-alpha*x4)
    xa, xt = max(0., 48/alpha-1), max(0., 48/gamma-1)
    log_r = (20*math.log(2)+math.log(h)+alpha/2
             +12*math.log1p(xa)-alpha*xa/4+12*math.log1p(xt)-gamma*xt/4)
    return dict(m_hat=m_hat, r=r, v=v, mu=mu, Ch=ch,
        odd_row=odd_row, odd_jet=odd_jet, native_jet=jet, xi=xi,
        alpha=alpha, gamma=gamma, cone_radius=radius, cone_angle_inverse=cinv,
        point_kernel_bound=point, T0=t0, P=p, H=h, triangle=triangle,
        native_contact=contact, anchor_first=anchor_first, anchor_contact=anchor_contact,
        CE_bulk=ce_bulk, CB_bulk=cb_bulk, log_Rstar=log_r,
        mass_only_jet_squared=1+3*r*r*g00_upper,
        mass_only_contact=r*math.sqrt(g00_upper), xi_fraction=xi_fraction)


def primitive_wilson_energy_envelope(*, size_floor: int, m_hat: float,
                                   r: float, v: float, s_lower: float,
                                   g00_upper: float,
                                   spatial_diagonal_upper: tuple[float, float, float],
                                   tilt_root_upper: float,
                                   xi_fraction: float) -> dict[str, float]:
    """Source-marked Wilson heat evaluator; all constants remain hatted."""
    require(size_floor >= 3, 'three-dimensional periodic heat-return floor')
    core = primitive_cone_envelope(m_hat=m_hat, r=r, v=v, s_lower=s_lower,
        g00_upper=g00_upper, spatial_diagonal_upper=spatial_diagonal_upper,
        tilt_root_upper=tilt_root_upper, xi_fraction=xi_fraction)
    heat_time = 4**(2/3)*math.pi/(8*r*math.sqrt(s_lower))
    infinite_part = 3*heat_time
    zero_mode_part = 4/(size_floor**3*core['mu'])
    cm = infinite_part+zero_mode_part
    bw = ((m_hat+3*r)*math.sqrt(g00_upper)
          +core['Ch']*math.exp(core['alpha'])+core['gamma'])
    delta = min(core['T0']/8, 1/(2*bw))
    coercivity_ratio = 1-(core['xi']+core['gamma'])/core['mu']
    energy_point = math.e*cm/(delta*coercivity_ratio)
    old_point = core['point_kernel_bound']
    point = min(old_point, energy_point)
    q = point/old_point
    result = dict(core)
    result['triangle'] = core['triangle']*q**3
    result['native_contact'] = core['native_contact']*q*q
    result['anchor_first'] = core['anchor_first']*q*q
    result['anchor_contact'] = core['anchor_contact']*q
    result['H'] = sum(result[key] for key in
                      ('triangle', 'native_contact', 'anchor_first', 'anchor_contact'))
    hratio = result['H']/core['H']
    result['CE_bulk'] = core['CE_bulk']*hratio
    result['CB_bulk'] = core['CB_bulk']*hratio
    result['log_Rstar'] = core['log_Rstar']+math.log(hratio)
    result['point_kernel_bound'] = point
    result.update(size_floor=float(size_floor), Wilson_heat_split_time=heat_time,
        mass_diagonal_infinite_part=infinite_part, mass_diagonal_zero_mode_part=zero_mode_part,
        mass_inverse_diagonal_upper=cm, source_transport_B=bw, source_interval=delta,
        energy_relative_coercivity=coercivity_ratio, energy_point_candidate=energy_point,
        cone_point_candidate=old_point, H_gain=core['H']/result['H'])
    return result


def even_loop_family(s_floor=.01, speed=.1, *, tight=False):
    """Odd-interpolation anchored envelope; no auxiliary-mass Hlow or Htail."""
    m, r, v = .2, .12, .3
    tstar = (2*math.sqrt(2)*math.sqrt((m+6*r)**2+3*v*v))**(2/3)
    g00, gjj = (.43, .065) if tight else (tstar, tstar)
    tilt_root = .002*math.sqrt(g00) if tight else math.sqrt(g00-s_floor)
    candidates = []
    for fraction in (.1, .2, .25, .3, .35, .4):
        core = primitive_cone_envelope(m_hat=m, r=r, v=v, s_lower=s_floor,
            g00_upper=g00, spatial_diagonal_upper=(gjj, gjj, gjj),
            tilt_root_upper=tilt_root, xi_fraction=fraction)
        mu, alpha, gamma = core['mu'], core['alpha'], core['gamma']
        ce_bulk, cb_bulk = core['CE_bulk'], core['CB_bulk']
        mass_jet2, mass_contact = core['mass_only_jet_squared'], core['mass_only_contact']
        epsilon = max(ce_bulk, cb_bulk)/100
        tau = max(core['T0'], (alpha+1)/gamma)
        log_r = core['log_Rstar']
        bs, bt = alpha/64, gamma/64
        def paid_tail_log(n):
            x, y = -bs*n, -bt*tau*n*n
            seam = log_r+2*math.log(n)-math.log(8)+max(x, y)+math.log1p(math.exp(-abs(x-y)))
            period = tau*n*n
            # Mass-only actual ordered finiteT remainder, including all holonomies.
            thermal = 5*math.log(n)+math.log(period*mass_jet2+mass_contact)-mu*period
            return max(seam, thermal)+math.log1p(math.exp(-abs(seam-thermal)))
        floor = max(5, math.ceil(2/bs), math.ceil(math.sqrt(1/(bt*tau))),
                    math.ceil(math.sqrt(7/(2*mu*tau))))
        lo, hi = floor, floor
        while paid_tail_log(hi) > math.log(epsilon):
            hi *= 2
        while hi-lo > 1:
            mid = (lo+hi)//2
            if paid_tail_log(mid) <= math.log(epsilon):
                hi = mid
            else:
                lo = mid
        ce, cb = ce_bulk+epsilon, cb_bulk+epsilon
        he = 16*max(ce, cb/speed**2, 1.)
        candidates.append(dict(core, s_floor=s_floor, G00_upper=g00, Gjj_upper=gjj,
            tight=tight, tilt_upper=.002 if tight else None,
            seam_and_thermal_budget=epsilon, N0_seam=hi, tau=tau,
            log_paid_tail_at_N0=paid_tail_log(hi), CE=ce, CB=cb,
            UE=1/(2*he), K=speed**2*he, bare_speed=speed,
            rho=max(ce/he, cb/(speed**2*he)), xi_fraction=fraction,
            construction='anchored even odd-band interpolation plus actual ordered mass thermal remainder'))
    return max(candidates, key=lambda c: c['UE'])


def ordered_even_loop_consumer(size=5, seed=2221057):
    """Independent continuous native loop integral, source Hessian and adverse."""
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    law = BulkQuantumLaw(size, band=band)
    centre = np.asarray([1, 2, 3])
    selected, frame = original_frame(band, size, centre*math.pi/2, True)
    g = np.asarray(selected['G'])
    beta = np.kron(np.eye(law.volume), chiral_basis(frame, g, 1)[2])
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(centre*math.pi/2)
    rng = np.random.default_rng(seed)
    phases = rng.normal(size=(3, law.volume, 3))*4.9
    temporal = rng.normal(size=(3, law.volume))*.37
    durations = np.asarray([.31, .43, .37])
    probe = np.zeros((law.volume, 3))
    probe[:, 1] = np.cos(2*math.pi*law.sites[:, 0]/size)/math.sqrt(law.volume/2)
    zero = np.zeros(law.volume)
    def split_at(source):
        h = [law.onebody(fields, links, a+source) for a in phases]
        masses = [(beta@matrix+matrix@beta)/2 for matrix in h]
        odds = [matrix-beta@mass for matrix, mass in zip(h, masses)]
        return h, masses, odds
    hs, masses, odds = split_at(np.zeros_like(probe))
    jets = [native_jets(law, h, probe) for h in hs]
    mass_first = [(beta@j+j@beta)/2 for j, _ in jets]
    mass_second = [(beta@c+c@beta)/2 for _, c in jets]
    odd_first = [j-beta@mp for (j, _), mp in zip(jets, mass_first)]
    odd_second = [c-beta@mpp for (_, c), mpp in zip(jets, mass_second)]
    def source_hessian(lam):
        h = [beta@m+lam*d for m, d in zip(masses, odds)]
        first = [beta@mp+lam*dp for mp, dp in zip(mass_first, odd_first)]
        second = [beta@mpp+lam*dpp for mpp, dpp in zip(mass_second, odd_second)]
        return ordered_hessian(h, temporal, durations, first, second, zero)['full']
    full, mass_only, opposite = source_hessian(1.), source_hessian(0.), source_hessian(-1.)
    parity_error = abs(full-opposite)/max(abs(full), 1e-14)
    require(parity_error < 2e-8, 'actual ordered closed odd-band traces vanish')
    target = full-mass_only
    eps = .004
    def loop_score(lam, source, *, freeze_anchor=False):
        _, ms, ds = split_at(source)
        h = [beta@m+lam*d for m, d in zip(ms, ds)]
        return first_source_response(h, temporal, durations, odds if freeze_anchor else ds, zero)
    quadrature = []
    for order in (4, 8):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        integral = 0.
        for node, weight in zip((nodes+1)/2, weights/2):
            fminus = loop_score(node, -eps*probe)
            fzero = loop_score(node, np.zeros_like(probe))
            fplus = loop_score(node, eps*probe)
            second = (fplus-2*fzero+fminus)/(eps*eps)
            integral += weight*second
        quadrature.append(dict(order=order, actual_loop_Hessian=integral,
                               error=abs(integral-target)/max(abs(target), 1e-14)))
    require(quadrature[-1]['error'] < 2e-5, 'anchored loop Hessian reaches independently differentiated actual determinant')
    chi = np.cos(2*math.pi*law.sites[:, 0]/size+.37)/math.sqrt(law.volume/2)
    ward = chi[:, None]-chi[law.target]
    ward_base = loop_score(.5, np.zeros_like(ward))
    correct = (loop_score(.5, eps*ward)+loop_score(.5, -eps*ward)-2*ward_base)/(eps*eps)
    wrong = (loop_score(.5, eps*ward, freeze_anchor=True)
             +loop_score(.5, -eps*ward, freeze_anchor=True)-2*ward_base)/(eps*eps)
    require(abs(correct) < 2e-7, 'anchored insertion differentiated with its source/current/contact satisfies Ward')
    require(abs(wrong) > 1e-4, 'frozen odd anchor jets fail the same anchored Ward boundary')
    # Resolved pointwise kernel control on the actual noncommuting history.
    green = ContinuousGreen(hs, temporal, durations)
    k = green.kernel(.74, .17)
    x, y = 0, int(law.target[law.target[0, 0], 1])
    actual_kernel = float(np.linalg.norm(k[4*x:4*x+4, 4*y:4*y+4], ord=2))
    budget = metric_budget(g)
    native_mu = .2*math.sqrt(budget['s'])
    d = (.2+6*.12)*math.sqrt(max(0., g[0, 0]-budget['s']))+.3*sum(np.sqrt(np.diag(g)[1:]))
    ch = sum(.12*math.sqrt(g[0, 0])+.3*math.sqrt(g[j, j]) for j in range(1, 4))
    xi = .25*native_mu
    alpha = math.log1p(xi/ch)
    r = (d+xi)/(math.sqrt(native_mu**2-2*native_mu*xi+d*d)+native_mu-xi)
    cinv = (1+r*r)/(1-r*r)
    rate = native_mu-xi
    point = 2*cinv**1.5/(1-math.exp(-rate*float(durations.sum())))
    spatial_distance = float(np.minimum(abs(law.sites[x]-law.sites[y]), size-abs(law.sites[x]-law.sites[y])).sum())
    temporal_distance = min(.74-.17, float(durations.sum())-(.74-.17))
    envelope = point*math.exp(-alpha*spatial_distance-rate*temporal_distance)
    require(actual_kernel <= envelope, 'cone-derived point kernel bounds actual ordered native source')
    return dict(size=size, modes=4*law.volume, centre=centre, seed=seed,
        phases_sha256=hashlib.sha256(phases.tobytes()).hexdigest(),
        temporal_sha256=hashlib.sha256(temporal.tobytes()).hexdigest(), durations=durations,
        full_Hessian=full, mass_only_Hessian=mass_only, odd_loop_Hessian=target,
        opposite_odd_Hessian=opposite, even_band_relative_error=parity_error,
        quadrature=quadrature, anchored_Ward_Hessian=correct, frozen_anchor_Ward_Hessian=wrong,
        ordered_point_kernel=actual_kernel, all_background_point_envelope=envelope,
        scope='actual continuous noncommuting native determinant loop/current/contact; no Gibbs samples or global metric certification')


def source_hashes():
    names = ('noncompact_wilson_photon_envelope.py', 'noncompact_wilson_photon_envelope.md',
             'noncompact_maxwell_phase.py', 'noncompact_maxwell_phase.md',
             'original_massive_photon_locality.md', 'original_massive_photon_locality.py', 'bulk_quantum_selection.py',
             'original_charged_principal.py', 'original_compact_charge_bound.py', 'rotation_band.py')
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def cutoff_cone_consumer(spacing, size=5, seed=2221071):
    """Finite-a native point kernel WITHOUT relative-Neumann admission."""
    require(0 < spacing <= .01, 'declared near-Dirac fixed-cutoff branch')
    band = RotationBand(a=spacing, m=.2, r=.12, v=.3, improved=False)
    law = BulkQuantumLaw(size, band=band)
    centre = np.asarray([1, 2, 3])
    selected, frame = original_frame(band, size, centre*math.pi/2, True)
    g = np.asarray(selected['G'])
    s = float(g[0, 0]-g[0, 1:]@np.linalg.solve(g[1:, 1:], g[1:, 0]))
    beta = np.kron(np.eye(law.volume), chiral_basis(frame, g, 1)[2])
    mu = .2*math.sqrt(s)
    tilt_root = math.sqrt(max(0., g[0, 0]-s))
    tilt = tilt_root/math.sqrt(s)
    kv = math.sqrt(.3**2*float(np.trace(g[1:, 1:]))/(2*.2*.12*spacing*s))
    delta_trace = 3*.2*.12*spacing*s/(.3**2*float(np.trace(g[1:, 1:])))
    close(np.asarray(kv*kv), np.asarray(3/(2*delta_trace)), 'spacing-correct relative applicability algebra')
    require(kv+tilt > 1, 'near-Dirac branch lies outside relative uniform admission')
    ch = sum((.12*math.sqrt(g[0, 0])+.3*math.sqrt(g[j, j]))/spacing for j in range(1, 4))
    d = (.2+6*.12/spacing)*tilt_root+.3/spacing*sum(np.sqrt(np.diag(g)[1:]))
    core = primitive_cone_envelope(m_hat=spacing*.2, r=.12, v=.3,
        s_lower=s, g00_upper=float(g[0, 0]),
        spatial_diagonal_upper=(float(g[1, 1]), float(g[2, 2]), float(g[3, 3])),
        tilt_root_upper=tilt_root, xi_fraction=.25)
    xi, alpha = core['xi']/spacing, core['alpha']
    radius, cinv = core['cone_radius'], core['cone_angle_inverse']
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(centre*math.pi/2)
    flat = law.onebody(fields, links, np.zeros((law.volume, 3)))
    mass = (beta@flat+flat@beta)/2
    odd = flat-beta@mass
    ev, uv = np.linalg.eigh(mass)
    require(float(ev[0]) >= mu-2e-8, 'actual finite-a native Wilson positive mass')
    x = (uv/np.sqrt(ev))@uv.conj().T
    relative_r = x@beta@odd@x
    spectrum, vectors = np.linalg.eigh(-1j*relative_r)
    index = int(np.argmax(abs(spectrum)))
    flat_radius = abs(float(spectrum[index]))
    require(flat_radius > 1, 'actual flat native relative-Neumann spectral radius exceeds one')
    z = vectors[:, index]
    exact = np.linalg.solve(np.eye(len(flat))+relative_r, z)
    term, partial = z.copy(), z.copy()
    for _ in range(20):
        term = -relative_r@term
        partial += term
    wrong_error = float(np.linalg.norm(partial-exact))
    require(wrong_error > 1, 'using inadmissible relative series fails the actual inverse boundary')
    rng = np.random.default_rng(seed)
    phases = rng.normal(size=(3, law.volume, 3))*4.9
    # Physical temporal source and durations: finite fixed-cutoff geometry.
    temporal = rng.normal(size=(3, law.volume))*.37/spacing
    durations = spacing*np.asarray([.31, .43, .37])
    h = [law.onebody(fields, links, a) for a in phases]
    green = ContinuousGreen(h, temporal, durations)
    kernel = green.kernel(.74*spacing, .17*spacing)
    y = int(law.target[law.target[0, 0], 1])
    actual = float(np.linalg.norm(kernel[:4, 4*y:4*y+4], ord=2))
    point = 2*cinv**1.5/(1-math.exp(-(mu-xi)*float(durations.sum())))
    spatial_distance = float(np.minimum(abs(law.sites[0]-law.sites[y]), size-abs(law.sites[0]-law.sites[y])).sum())
    temporal_distance = min(.57*spacing, float(durations.sum())-.57*spacing)
    envelope = point*math.exp(-alpha*spatial_distance-(mu-xi)*temporal_distance)
    require(actual <= envelope, 'cutoff-scaled cone envelope reaches actual ordered native kernel')
    probe = np.zeros((law.volume, 3))
    probe[:, 1] = np.cos(2*math.pi*law.sites[:, 0]/size)/math.sqrt(law.volume/2)
    current, contact = native_jets(law, h[0], probe)
    eps = .0001
    hp = law.onebody(fields, links, phases[0]+eps*probe)
    hm = law.onebody(fields, links, phases[0]-eps*probe)
    first_error = close(current, (hp-hm)/(2*eps), 'actual physical finite-a same current', 3e-8)
    contact_error = close(contact, (hp-2*h[0]+hm)/(eps*eps), 'actual physical finite-a same contact', 2e-6)
    return dict(a=spacing, size=size, modes=4*law.volume, Gram=g, Schur=s,
        native_stationarity_residual=selected['gradient_residual'],
        delta_trace=delta_trace, isotropic_Gordon_comparator=2/(1+delta_trace),
        kappa_spatial=kv, kappa_total=kv+tilt, relative_uniform_admitted=False,
        actual_flat_relative_radius=flat_radius, inadmissible_series_error=wrong_error,
        mu=mu, odd_row=d, Ch=ch, xi=xi, cone_radius=radius,
        spatial_lattice_rate=alpha, spatial_physical_rate=alpha/spacing,
        temporal_physical_rate=mu-xi, finiteT_point_prefactor=point,
        actual_ordered_point_kernel=actual, paid_point_envelope=envelope,
        current_error=first_error, contact_error=contact_error, durations=durations,
        phases_sha256=hashlib.sha256(phases.tobytes()).hexdigest(),
        temporal_sha256=hashlib.sha256(temporal.tobytes()).hexdigest(),
        primitive_hatted_envelope=core,
        scope='actual finite-cutoff native point/current consumer; Gordon is an isotropic carrier comparator, not physical mechanical spin measurement; no near-Dirac CE/CB or uniform UV claim')


def native_consumer(size=5, centre=(1, 2, 3), seed=2221057):
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    law = BulkQuantumLaw(size, band=band)
    selected, frame = original_frame(band, size, np.asarray(centre)*math.pi/2, True)
    g = np.asarray(selected['G'])
    budget = metric_budget(g)
    require(budget['kappa'] < 1, 'native selected relative odd contraction')
    beta4 = chiral_basis(frame, g, 1)[2]
    beta = np.kron(np.eye(law.volume), beta4)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.asarray(centre)*math.pi/2)
    rng = np.random.default_rng(seed)
    phases = rng.normal(size=(3, law.volume, 3))*4.9
    temporal = rng.normal(size=(3, law.volume))*.37
    durations = np.asarray([.31, .43, .37])
    masses, invroots, odds, hs, laps, heat, rows = [], [], [], [], [], [], []
    for a in phases:
        shifts = []
        for j in range(3):
            t = np.zeros((law.volume, law.volume), dtype=complex)
            for x in range(law.volume):
                t[x, law.target[x, j]] = links[x, j, 0, 0]*np.exp(1j*a[x, j])
            shifts.append(t)
        lj = [2*np.eye(law.volume)-t-t.conj().T for t in shifts]
        sj = [(t-t.conj().T)/(2j) for t in shifts]
        sine_defects = [float(np.linalg.eigvalsh(l-s@s)[0]) for l, s in zip(lj, sj)]
        require(min(sine_defects) > -2e-9, 'all-phase S_j squared <= native L_j')
        lap = sum(lj)
        w = .2*np.eye(law.volume)+.12*lap/2
        mass_small = math.sqrt(budget['s'])*w
        mass = np.kron(mass_small, np.eye(4))
        odd = np.kron(w, frame[0]-math.sqrt(budget['s'])*beta4)
        odd += .3*sum(np.kron(s, phi) for s, phi in zip(sj, frame[1:]))
        h = law.onebody(fields, links, a)
        reconstruction = close(beta@mass+odd, h, 'true native Wilson mass plus odd reconstruction')
        values, vectors = np.linalg.eigh(mass_small)
        ir = np.kron((vectors/np.sqrt(values))@vectors.conj().T, np.eye(4))
        relative = ir@odd@ir
        actual_k = float(np.max(np.abs(np.linalg.eigvalsh(relative))))
        require(actual_k <= budget['kappa']+2e-9, 'derived all-phase relative norm controls native sample')
        anti = close(beta@odd, -odd@beta, 'native odd beta anticommutation')
        vac = -2*float(np.trace(mass_small).real)
        expected_vac = -2*law.volume*math.sqrt(budget['s'])*(.2+3*.12)
        close(np.asarray(vac), np.asarray(expected_vac), 'phase-independent whole-half mass vacuum')
        masses.append(mass); invroots.append(ir); odds.append(odd); hs.append(h); laps.append(lap)
        heat.append(mass_small)
        rows.append(dict(relative_norm=actual_k, mass_min=float(values[0]), reconstruction=reconstruction,
                         beta_odd_error=anti, sine_form_min=min(sine_defects), mass_vacuum=vac))
    # Time-ordered scalar Wilson heat, not exp of the average Hamiltonian.
    free_lap = np.zeros_like(laps[0])
    for j in range(3):
        t = np.zeros_like(free_lap)
        t[np.arange(law.volume), law.target[:, j]] = 1
        free_lap += 2*np.eye(law.volume)-t-t.conj().T
    free_mass = math.sqrt(budget['s'])*(.2*np.eye(law.volume)+.12*free_lap/2)
    ordered, positive = np.eye(law.volume, dtype=complex), np.eye(law.volume, dtype=complex)
    for mass, dt in zip(heat, durations):
        ordered = np.asarray(expm(-dt*mass))@ordered
        positive = np.asarray(expm(-dt*free_mass))@positive
    heat_excess = float(np.max(np.abs(ordered)-np.real(positive)))
    require(heat_excess < 2e-10, 'ordered native Wilson heat diamagnetism')
    # A spectral time Galerkin: exact skew derivative, noncommuting mass history.
    derivative = np.asarray([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)/2
    time_dimension = 3*len(masses[0])
    ib = np.asarray(block_diag(*invroots))
    bb = np.kron(np.eye(3), beta)
    dodd = np.asarray(block_diag(*odds))
    temporal4 = np.diag(np.repeat(temporal.reshape(-1), 4))
    skew_time = np.kron(derivative, beta)-1j*bb@temporal4
    skew_odd = bb@dodd
    b0 = ib@skew_time@ib
    r = ib@skew_odd@ib
    q = np.eye(time_dimension)+b0
    actual = q+r
    congruence = close((actual+actual.conj().T)/2, np.eye(time_dimension), 'both-sided mass congruence identity coercivity')
    z = rng.normal(size=time_dimension)+1j*rng.normal(size=time_dimension)
    z /= np.linalg.norm(z)
    exact = np.linalg.solve(actual, z)
    require(np.linalg.norm(exact) <= 1+1e-9, 'mass-weighted Euclidean inverse contraction')
    factor = lu_factor(q)
    term = lu_solve(factor, z)
    partial = term.copy()
    power = 100
    for _ in range(power):
        term = -lu_solve(factor, r@term)
        partial += term
    error = float(np.linalg.norm(exact-partial))
    bound = budget['kappa']**(power+1)/(1-budget['kappa'])
    require(error <= bound, 'native noncommuting relative Neumann remainder')
    wrong = ib@skew_time
    wrong_coercivity = float(np.linalg.norm((wrong+wrong.conj().T)/2))
    require(wrong_coercivity > 1e-3, 'one-sided mass scaling adverse reaches varying-M skew boundary')
    wrong_odd = dodd/(.2*math.sqrt(budget['s']))
    wrong_norm = float(np.max(np.abs(np.linalg.eigvalsh(wrong_odd[:len(hs[0]), :len(hs[0])]))))
    require(wrong_norm > budget['kappa'], 'bare mass replacing relative Wilson mass fails advertised bound')
    # Actual same endpoint current, contact, occupied-empty Wick covariance.
    probe = np.zeros((law.volume, 3))
    probe[:, 1] = np.cos(2*math.pi*law.sites[:, 0]/size)/math.sqrt(law.volume/2)
    current, contact = native_jets(law, hs[0], probe)
    eps = .0001
    hp = law.onebody(fields, links, phases[0]+eps*probe)
    hm = law.onebody(fields, links, phases[0]-eps*probe)
    current_error = close(current, (hp-hm)/(2*eps), 'native current finite difference', 3e-8)
    contact_error = close(contact, (hp-2*hs[0]+hm)/(eps*eps), 'native contact finite difference', 2e-6)
    # Continue the selected differentiated-relative-Wilson construction.
    mass = masses[0]
    x = invroots[0]
    mp = (beta@current+current@beta)/2
    mpp = (beta@contact+contact@beta)/2
    dp = current-beta@mp
    dpp = contact-beta@mpp
    mass_values, mass_vectors = np.linalg.eigh(mass)
    roots = np.sqrt(mass_values)
    y = (mass_vectors*roots)@mass_vectors.conj().T
    yp = mass_vectors@((mass_vectors.conj().T@mp@mass_vectors)/(roots[:, None]+roots[None, :]))@mass_vectors.conj().T
    ypp = mass_vectors@((mass_vectors.conj().T@(mpp-2*yp@yp)@mass_vectors)/(roots[:, None]+roots[None, :]))@mass_vectors.conj().T
    xp = -x@yp@x
    xpp = 2*x@yp@x@yp@x-x@ypp@x
    relative_p = xp@beta@odds[0]@x+x@beta@dp@x+x@beta@odds[0]@xp
    relative_pp = (xpp@beta@odds[0]@x+x@beta@odds[0]@xpp
                   +2*xp@beta@odds[0]@xp+2*xp@beta@dp@x
                   +2*x@beta@dp@xp+x@beta@dpp@x)
    relative_shifted = []
    jet_eps = .002
    for sign in (-1, 1):
        shifted_h = law.onebody(fields, links, phases[0]+sign*jet_eps*probe)
        shifted_m = (beta@shifted_h+shifted_h@beta)/2
        ev, uv = np.linalg.eigh(shifted_m)
        shifted_x = (uv/np.sqrt(ev))@uv.conj().T
        shifted_d = shifted_h-beta@shifted_m
        relative_shifted.append(shifted_x@beta@shifted_d@shifted_x)
    rp_fd = (relative_shifted[1]-relative_shifted[0])/(2*jet_eps)
    rpp_fd = (relative_shifted[1]-2*x@beta@odds[0]@x+relative_shifted[0])/(jet_eps**2)
    relative_first_error = close(relative_p, rp_fd, 'differentiated relative Wilson current', 2e-7)
    relative_second_error = close(relative_pp, rpp_fd, 'contact-completed relative Wilson second jet', 2e-7)
    wrong_relative_error = float(np.linalg.norm(x@beta@dp@x-relative_p)/np.linalg.norm(relative_p))
    require(wrong_relative_error > .01, 'frozen mass weights fail actual relative current')
    close(y@y, mass, 'mass square-root owner')
    values, vectors = np.linalg.eigh(hs[0])
    occupied = vectors[:, values < 0]
    empty = vectors[:, values > 0]
    block = empty.conj().T@current@occupied
    variance = float(np.linalg.norm(block)**2)
    trace_j2 = float(np.trace(current@current).real)
    trace_formula = 2*sum((.12**2*g[0, 0]+.3**2*g[j+1, j+1])*float(np.sum(probe[:, j]**2)) for j in range(3))
    trace_identity_error = close(np.asarray(trace_j2), np.asarray(trace_formula), 'all-phase native current quadratic trace')
    wick_bound = trace_j2/2
    susceptibility = float(2*np.sum(np.abs(block)**2/(values[values > 0, None]-values[None, values < 0])))
    gap_bound = trace_j2/(2*.2*math.sqrt(budget['s']))
    require(0 < variance <= wick_bound, 'actual canonical whole-half Wick current variance')
    require(susceptibility <= gap_bound, 'same current occupied-empty static susceptibility')
    # Wrong current keeps only the spatial Clifford odd endpoint.
    wrong_frame = frame.copy(); wrong_frame[0] = 0
    wrong_h = law.onebody(np.tile(wrong_frame, (law.volume, 1, 1, 1)), links, phases[0])
    wrong_current = native_jets(law, wrong_h, probe)[0]
    wrong_current_error = float(np.linalg.norm(wrong_current-current)/np.linalg.norm(current))
    require(wrong_current_error > .01, 'odd-only current omits the actual Wilson derivative')
    all64 = []
    for theta in np.ndindex(4, 4, 4):
        native, _ = original_frame(band, size, np.asarray(theta)*math.pi/2, True)
        budget64 = metric_budget(np.asarray(native['G']))
        require(budget64['kappa'] < 1, 'sampled all64 tilted Gram relative contraction')
        all64.append(dict(centre=theta, Gram=native['G'], **budget64))
    return dict(size=size, modes=4*law.volume, spatial_dimension=3, centre=centre, seed=seed,
        phases_sha256=hashlib.sha256(phases.tobytes()).hexdigest(), temporal_sha256=hashlib.sha256(temporal.tobytes()).hexdigest(),
        durations=durations, Gram=g, budget=budget, all64_finite_samples=all64, spatial_rows=rows,
        heat_diamagnetic_excess=heat_excess, mass_history_commutator=float(np.linalg.norm(masses[0]@masses[1]-masses[1]@masses[0])),
        temporal_Galerkin=dict(coercivity_error=congruence, inverse_vector_norm=float(np.linalg.norm(exact)),
                              Neumann_power=power, actual_error=error, analytic_error_bound=bound),
        current=dict(variance=variance, Wick_bound=wick_bound, susceptibility=susceptibility,
                     gap_bound=gap_bound, first_difference_error=current_error, contact_difference_error=contact_error,
                     quadratic_trace_identity_error=trace_identity_error,
                     relative_first_jet_error=relative_first_error, relative_second_jet_error=relative_second_error),
        adverse=dict(one_sided_congruence_Hermitian_defect=wrong_coercivity,
                     bare_mass_odd_norm=wrong_norm, odd_only_current_relative_error=wrong_current_error,
                     frozen_relative_mass_current_error=wrong_relative_error),
        scope='native3D operator/current; temporal Galerkin checks the derived identity, not continuum determinant or Gibbs sampling')


def wilson_energy_consumer(spacing=.001, size=5, seed=2221072):
    """Native source-marked heat/ordered evaluator with finite-zero-mode adverse."""
    require(0 < spacing <= .001 and size == 5, 'declared N5 near-Dirac marked-source probe')
    band = RotationBand(a=spacing, m=.2, r=.12, v=.3, improved=False)
    law = BulkQuantumLaw(size, band=band)
    centre = np.asarray([1, 2, 3])
    selected, frame = original_frame(band, size, centre*math.pi/2, True)
    g = np.asarray(selected['G'])
    s = float(g[0, 0]-g[0, 1:]@np.linalg.solve(g[1:, 1:], g[1:, 0]))
    beta = np.kron(np.eye(law.volume), chiral_basis(frame, g, 1)[2])
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(centre*math.pi/2)
    energy = primitive_wilson_energy_envelope(size_floor=size,
        m_hat=spacing*.2, r=.12, v=.3, s_lower=s,
        g00_upper=float(g[0, 0]),
        spatial_diagonal_upper=(float(g[1, 1]), float(g[2, 2]), float(g[3, 3])),
        tilt_root_upper=math.sqrt(max(0., g[0, 0]-s)), xi_fraction=.25)
    cancel = np.zeros((law.volume, 3))
    for j in range(3):
        cancel[law.sites[:, j] == size-1, j] = -centre[j]*math.pi/2
    h0 = spacing*law.onebody(fields, links, cancel)
    m0 = (beta@h0+h0@beta)/2
    free_lap = np.zeros((law.volume, law.volume))
    for j in range(3):
        t = np.zeros_like(free_lap)
        t[np.arange(law.volume), law.target[:, j]] = 1
        free_lap += 2*np.eye(law.volume)-t-t.T
    free_mass = math.sqrt(s)*(spacing*.2*np.eye(law.volume)+.12*free_lap/2)
    mass_identity = close(m0, np.kron(free_mass, np.eye(4)), 'actual native harmonic path gives free Wilson mass')
    mass_inverse = np.linalg.inv(m0)
    actual_diag = float(np.max(np.real(np.diag(mass_inverse))))
    require(actual_diag <= energy['mass_inverse_diagonal_upper'], '3D marked heat bounds actual mass inverse diagonal')
    require(actual_diag > energy['mass_diagonal_infinite_part'],
            'dropping finiteN harmonic zero mode fails the actual marked heat boundary')
    rng = np.random.default_rng(seed)
    phases = rng.normal(size=(3, law.volume, 3))*4.9
    temporal = rng.normal(size=(3, law.volume))*.37
    durations = np.asarray([.31, .43, .37])
    h = [np.asarray(spacing*law.onebody(fields, links, a), dtype=np.complex128) for a in phases]
    diagonal_rows = []
    for matrix in h:
        mass = (beta@matrix+matrix@beta)/2
        diagonal_rows.append(float(np.max(np.real(np.diag(np.linalg.inv(mass))))))
    require(max(diagonal_rows) <= energy['mass_inverse_diagonal_upper'], 'same heat bound covers each curved history')
    green = ContinuousGreen(h, temporal, durations)
    target = int(law.target[law.target[0, 0], 1])
    source_basis = np.eye(4*law.volume, dtype=complex)[:, :4]
    target_basis = np.eye(4*law.volume, dtype=complex)[:, 4*target:4*target+4]
    t0, s0 = .74, .17
    period = float(durations.sum())
    delta = min(period/8, 1/(2*energy['source_transport_B']))
    ps, pt = green.fundamental(s0), green.fundamental(t0)
    source_at_s = np.linalg.solve(ps, source_basis)
    exact = target_basis.conj().T@green.kernel(t0, s0)@source_basis
    transported, frozen = np.zeros((4, 4), dtype=complex), np.zeros((4, 4), dtype=complex)
    nodes = np.asarray([.5-1/(2*math.sqrt(3)), .5+1/(2*math.sqrt(3))])
    for u in s0-delta*nodes:
        f = green.fundamental(float(u))@source_at_s
        for time in t0+delta*nodes:
            pv = green.fundamental(float(time))
            adjoint_target = np.linalg.solve(pv.conj().T, pt.conj().T@target_basis)
            k = green.kernel(float(time), float(u))
            transported += adjoint_target.conj().T@k@f/4
            frozen += target_basis.conj().T@k@source_basis/4
    identity_error = float(np.linalg.norm(transported-exact)/np.linalg.norm(exact))
    wrong_error = float(np.linalg.norm(frozen-exact)/np.linalg.norm(exact))
    require(identity_error < 2e-9, 'actual native short-source transport reaches point Green evaluator')
    require(wrong_error > 1e-3, 'frozen source windows fail the same native evaluator boundary')
    point = (math.exp(2*energy['source_transport_B']*delta)
             *energy['mass_inverse_diagonal_upper']/(delta*energy['energy_relative_coercivity']))
    distance = float(np.minimum(abs(law.sites[target]-law.sites[0]), size-abs(law.sites[target]-law.sites[0])).sum())
    time_distance = min(t0-s0, period-(t0-s0))
    envelope = point*math.exp(-energy['alpha']*distance-energy['gamma']*time_distance)
    actual_point = float(np.linalg.norm(exact, ord=2))
    require(actual_point <= envelope, 'localized Wilson-energy point envelope covers ordered native kernel')
    require(energy['H_gain'] > 1e5, 'marked heat materially shrinks same-box anchored moments')
    wrap_t, wrap_s = period-.03, .03
    wrap_ps, wrap_pt = green.fundamental(wrap_s), green.fundamental(wrap_t)
    wrap_source = np.linalg.solve(wrap_ps, source_basis)
    wrap_exact = target_basis.conj().T@green.kernel(wrap_t, wrap_s)@source_basis
    wrap_good = np.zeros((4, 4), dtype=complex)
    wrap_bad = np.zeros((4, 4), dtype=complex)
    def lifted_fundamental(value):
        turn = math.floor(value/period)
        fundamental = green.fundamental(float(value-turn*period))
        if turn == -1:
            return np.linalg.solve(green.u.T, fundamental.T).T
        if turn == 1:
            return fundamental@green.u
        require(turn == 0, 'single-period native window lift')
        return fundamental
    for node in nodes:
        good_u, bad_u = wrap_s+delta*node, wrap_s-delta*node
        good_f = green.fundamental(float(good_u))@wrap_source
        bad_f = lifted_fundamental(float(bad_u))@wrap_source
        for target_node in nodes:
            good_v, bad_v = wrap_t-delta*target_node, wrap_t+delta*target_node
            good_pv = green.fundamental(float(good_v))
            good_g = np.linalg.solve(good_pv.conj().T, wrap_pt.conj().T@target_basis)
            wrap_good += good_g.conj().T@green.kernel(float(good_v), float(good_u))@good_f/4
            bad_pv = lifted_fundamental(float(bad_v))
            bad_g = np.linalg.solve(bad_pv.conj().T, wrap_pt.conj().T@target_basis)
            u_turn, v_turn = math.floor(bad_u/period), math.floor(bad_v/period)
            bad_k = (-1)**(v_turn-u_turn)*green.kernel(
                float(bad_v-v_turn*period), float(bad_u-u_turn*period))
            wrap_bad += bad_g.conj().T@bad_k@bad_f/4
    wrap_error = float(np.linalg.norm(wrap_good-wrap_exact)/np.linalg.norm(wrap_exact))
    bad_wrap_error = float(np.linalg.norm(wrap_bad-wrap_exact)/np.linalg.norm(wrap_exact))
    require(wrap_error < 2e-9, 'cyclic-order preserving native windows retain the AP source branch')
    require(bad_wrap_error > 1e-3, 'blind outward windows cross the AP jump and fail the same evaluator')
    return dict(a=spacing, size=size, modes=4*law.volume, seed=seed, Gram=g,
        actual_free_mass_inverse_diagonal=actual_diag, actual_curved_mass_diagonals=diagonal_rows,
        native_mass_identity_error=mass_identity, energy_hatted_envelope=energy,
        evaluator_transport_error=identity_error, frozen_evaluator_error=wrong_error,
        near_seam_evaluator_transport_error=wrap_error,
        wrong_AP_window_evaluator_error=bad_wrap_error,
        actual_ordered_point_kernel=actual_point, finiteT_energy_point_envelope=envelope,
        source_interval=delta, durations_hat=durations,
        phases_sha256=hashlib.sha256(phases.tobytes()).hexdigest(),
        temporal_sha256=hashlib.sha256(temporal.tobytes()).hexdigest(),
        scope='actual native3D source-marked Wilson heat and ordered Green; finite zero mode and source transport retained; no physical coupling selection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--loop-only', action='store_true',
                        help='Execute the new ordered even-band loop consumer without repeating the baseline.')
    parser.add_argument('--cutoff-cone', action='store_true',
                        help='Execute the physical-regime weighted-cone fork without a1 baseline/loop reruns.')
    parser.add_argument('--energy-marked', action='store_true',
                        help='Execute the new Wilson-energy/source-marked heat consumer without old consumer reruns.')
    parser.add_argument('--spacings', type=float, nargs='+')
    args = parser.parse_args()
    require(args.size >= 5, 'genuine3D N>=5')
    require(sum((args.loop_only, args.cutoff_cone, args.energy_marked)) <= 1, 'choose one scientific consumer mode')
    spacings = args.spacings or ([.001] if args.energy_marked else [.01, .001])
    before = source_hashes()
    start = time.monotonic()
    universal = anchored_family()
    tight = anchored_family(.40, tight=True)
    predecessor = phase_constants()
    native = None if args.loop_only or args.cutoff_cone or args.energy_marked else native_consumer(args.size)
    loops = ordered_even_loop_consumer(args.size) if args.loop_only else None
    cutoff = [cutoff_cone_consumer(a, args.size) for a in spacings] if args.cutoff_cone else None
    energy = [wilson_energy_consumer(a, args.size) for a in spacings] if args.energy_marked else None
    universal_loop = even_loop_family()
    tight_loop = even_loop_family(.40, tight=True)
    require(source_hashes() == before, 'exact sources unchanged through natural consumer exit')
    result = dict(plan=PLAN, revision=REVISION, source_hashes=before,
        runtime=dict(python=platform.python_version(), executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__),
        input=dict(size=args.size, centre=[1, 2, 3],
                   seed=2221072 if args.energy_marked else 2221071 if args.cutoff_cone else 2221057,
                   mass=.2, Wilson=.12, speed=.3,
                   spacings=spacings if args.cutoff_cone or args.energy_marked else None),
        universal_same_metric_family=universal, tighter_declared_metric_family=tight,
        improvement_same_metric=universal['UE']/predecessor['UE'],
        improvement_tighter_metric=tight['UE']/predecessor['UE'], predecessor_UE=predecessor['UE'],
        native=native, ordered_even_loops=loops,
        cutoff_weighted_cone=cutoff,
        Wilson_energy_marked=energy,
        universal_even_loop_family=universal_loop, tighter_even_loop_family=tight_loop,
        loop_gain_same_metric=universal_loop['UE']/universal['UE'],
        loop_gain_tighter_metric=tight_loop['UE']/tight['UE'],
        seconds=time.monotonic()-start, checks=True,
        claims=dict(all_background_relative_bound='derived', continuum_heat_bound='derived',
                    anchored_Hessian='derived odd-interpolation anchored envelope with inherited Ward filling and paid seams',
                    global_metric_admission_from_samples=False, target_UE_1e_4=False,
                    full_interacting_current_variance_from_Wick=False,
                    uniform_UV_Maxwell_window=False, near_Dirac_phase_constants=False))
    print(json.dumps(portable(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
