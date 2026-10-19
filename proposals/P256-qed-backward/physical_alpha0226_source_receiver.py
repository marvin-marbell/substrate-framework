#!/usr/bin/env python3
"""Original-source positive-range pullback and finite log-survival receiver.

A supplied chart remains an APPROXIMATION to the complete midpoint operator.
Its source is Qdag m(s) from the original boundary/fullK/CAR family, never a
replacement vacuum or own-sector normalization. Missing source jets and full
operator complements must be priced outside this module. The positive source
coordinate is A^-1/2 m, NOT the normalized half-source m. ORIGINALfield
Pancharatnam transport and both ordered exponential contacts are retained.
This is a finite source receiver, not an on-shell photon/Compton amplitude.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.linalg
import physical_alpha0226_midpoint_operator as core

SOURCE_BYTES = Path(__file__).read_bytes()


def pulled_original_source_jets(a: np.ndarray, ax: np.ndarray, ay: np.ndarray,
                                axy: np.ndarray, source: dict[str, np.ndarray],
                                physical: dict) -> dict[str, np.ndarray]:
    """Differentiate Qpolar(s) A(s)^-1/2 Qdag m(s), including source/prep jets."""
    values, frame = np.linalg.eigh((a+a.conj().T)/2)
    if values[0] <= 0:
        raise ArithmeticError('Original positive source needs strict positivity; no floor')
    s = (frame*np.sqrt(values))@frame.conj().T
    inv = (frame/np.sqrt(values))@frame.conj().T
    sx = scipy.linalg.solve_sylvester(s, s, ax)
    sy = scipy.linalg.solve_sylvester(s, s, ay)
    sxy = scipy.linalg.solve_sylvester(s, s, axy-sx@sy-sy@sx)
    ix = -inv@sx@inv
    iy = -inv@sy@inv
    ixy = inv@sx@inv@sy@inv+inv@sy@inv@sx@inv-inv@sxy@inv
    m, mx, my, mxy = (source[k] for k in ('', 'x', 'y', 'xy'))
    n = inv@m
    nx = ix@m+inv@mx
    ny = iy@m+inv@my
    nxy = ixy@m+ix@my+iy@mx+inv@mxy
    ox, oy, qxy = (physical[k] for k in ('Omega_x', 'Omega_y', 'Pancharatnam_Qxy'))
    return {'': n, 'x': nx+ox@n, 'y': ny+oy@n,
            'xy': nxy+ox@ny+oy@nx+qxy@n}


def connected_survival(physical: dict, source: dict[str, np.ndarray],
                       observation_time: float) -> dict:
    """Actual finite Tr(source† exp(-T H) source) jets and connected log.

    Both current insertion orders and the Hxy seagull are supplied by the
    noncommuting exponential, and ALL eight moving-source product terms are
    carried. CommonGLOBALneutral lambda is frozen in H, so no vacuum jets.
    """
    h, hx, hy, hxy = (physical[k] for k in ('H', 'physical_x', 'physical_y', 'physical_xy'))
    t = observation_time
    e, ex, ey, exy = core.consumer.ALPHA.matrix_exp_jet(-t*h, -t*hx, -t*hy, -t*hxy)
    p, px, py, pxy = (source[k] for k in ('', 'x', 'y', 'xy'))
    pairing = lambda left, matrix, right: float(np.trace(left.conj().T@matrix@right).real)
    z = pairing(p, e, p)
    if z <= 0:
        raise ArithmeticError('Actual finite positive survival lost positivity; no fallback')
    zx = 2*pairing(px, e, p)+pairing(p, ex, p)
    zy = 2*pairing(py, e, p)+pairing(p, ey, p)
    operator_mixed = pairing(p, exy, p)
    source_mixed = (2*pairing(pxy, e, p)+2*pairing(px, e, py)
                    +2*pairing(px, ey, p)+2*pairing(py, ex, p))
    zxy = operator_mixed+source_mixed
    return dict(observation_time=t, Z=z, Zx=zx, Zy=zy, Zxy=zxy,
                operator_mixed_over_Z=operator_mixed/z,
                original_source_mixed_over_Z=source_mixed/z,
                disconnected_ZxZy_over_Z2=zx*zy/z**2,
                connected_log_mixed=zxy/z-zx*zy/z**2,
                source_positive_range_Gram=float(np.vdot(p, p).real),
                interpretation='Finite ORIGINALsource receiver in supplied chart. No on-shell photons, fitted pole, own-sector source normalization or full-log certification.')
