#!/usr/bin/env python3
"""Original Phi positive directional form on the once-seeded paired orbit."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, cast
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
from scipy.linalg import expm
from prepared_neutral_shared_photon_state import load_shared_native_cache
from prepared_neutral_changing_paired_pulse import source_point, coupled_interval
from prepared_neutral_ambient_metric import algebra_coordinates
from unrestricted_field_minimum import coordinates
from original_charged_principal import portable


def cofactor(matrix: np.ndarray) -> np.ndarray:
    """Polynomial cofactor, including singular occupied cross matrices."""
    u, s, vh = np.linalg.svd(matrix)
    products = np.array([np.prod(np.delete(s, i)) for i in range(len(s))])
    orientation = np.linalg.det(u)*np.linalg.det(vh)
    return orientation*(u.conj()*products)@vh.conj()


def _cross_blocks(light: np.ndarray, reference: np.ndarray, sea: int) -> dict[str, np.ndarray]:
    """Wick-before-compression; CAR particle/hole signs live in cofactors."""
    n = len(light)
    occupied = light[:sea, :sea]
    det = np.linalg.det(occupied)
    # Bordered determinant identity is polynomial: no occupied/field inverse.
    particle = det*light[sea:, sea:]-light[sea:, :sea]@cofactor(occupied).T@light[:sea, sea:]
    neutral = np.diag([det*np.linalg.det(reference), np.linalg.det(light[:sea+4, :sea+4])])
    plus = np.kron(cofactor(reference), particle)
    minus = np.kron(reference, cofactor(light[:sea+4, :sea+4]))
    charged = np.zeros((4*(n-sea)+4*(sea+4),)*2, complex)
    split = len(plus)
    charged[:split, :split] = plus
    charged[split:, split:] = minus
    return dict(neutral=neutral, charged=charged)


def full_cross_kernels(model: Any, units_left: dict[str, np.ndarray],
                       units_right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Full physical RAW overlaps in the ORIGINAL fixed Slater fibre.

    Neutral sea54/Ref4 and sea58/Ref0 are orthogonal. Charged ordering is
    ref-hole outer/light-empty inner, then Ref-particle outer/light-hole inner.
    Full heavy108 occupied54 determinant multiplies EVERY entry. Section
    gauges and endpoint bank recapture are never part of this ABI.
    """
    fibre = model.seed_fibre
    orbitals = fibre.orbitals
    light = orbitals.conj().T@units_left['light'].conj().T@units_right['light']@orbitals
    r = model.reference_frame
    reference = r.conj().T@units_left['reference'].conj().T@units_right['reference']@r
    heavy = model.heavy_orbitals
    heavy_cross = heavy.conj().T@units_left['heavy'].conj().T@units_right['heavy']@heavy
    determinant = np.linalg.det(heavy_cross)
    return {name: determinant*block for name, block in _cross_blocks(light, reference, fibre.sea_count).items()}


def full_cross_overlap(model: Any, units_left: dict[str, np.ndarray],
                       units_right: dict[str, np.ndarray], left: np.ndarray,
                       right: np.ndarray, sector: str) -> complex:
    """Contract fixed-fibre vectors; callers supply original LAB or held functions."""
    return complex(np.vdot(left, full_cross_kernels(model, units_left, units_right)[sector]@right))


def _wick_control(model: Any, left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, Any]:
    """Independent full occupied-minor CAR overlaps on actual108/Ref4 families."""
    f = model.seed_fibre
    l = f.orbitals.conj().T@left['light'].conj().T@right['light']@f.orbitals
    r = model.reference_frame.conj().T@left['reference'].conj().T@right['reference']@model.reference_frame
    h = model.heavy_orbitals
    hd = np.linalg.det(h.conj().T@left['heavy'].conj().T@right['heavy']@h)
    kernels = full_cross_kernels(model, left, right)
    errors = []
    for sector, offset, size in ((1, 0, f.dimensions[1]), (-1, f.dimensions[1], f.dimensions[-1])):
        ids = np.unique(np.r_[0, 1, size//4-1, size//2, size-2, size-1])+offset
        for i in ids:
            for j in ids:
                ml, mr = f.charged_masks[i], f.charged_masks[j]
                il = [k for k in range(f.n) if ml & (1 << k)]
                ir = [k for k in range(f.n) if mr & (1 << k)]
                rl = [k for k in range(4) if ml & (1 << (f.n+k))]
                rr = [k for k in range(4) if mr & (1 << (f.n+k))]
                wick = hd*f.charged_car_phases[i]*f.charged_car_phases[j]*np.linalg.det(l[np.ix_(il, ir)])*np.linalg.det(r[np.ix_(rl, rr)])
                errors.append(abs(wick-kernels['charged'][i, j]))
    return dict(full108_Ref4_CAR_minor_max_error=float(max(errors)), sampled_cross_entries=len(errors),
                light_PH_cross_norm=float(np.linalg.norm(l[:f.sea_count, f.sea_count:])),
                Ref_cross_offdiagonal_norm=float(np.linalg.norm(r-np.diag(np.diag(r)))))


def sparse_car_control() -> dict[str, Any]:
    """Independent ten-mode full Fock CAR exponential, not identity wiring."""
    import itertools
    rng = np.random.default_rng(222754)
    a = rng.normal(size=(6, 6))+1j*rng.normal(size=(6, 6)); a = .11*(a+a.conj().T)
    r = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)); r = .17*(r+r.conj().T)
    body = np.zeros((10, 10), complex); body[:6, :6] = a; body[6:, 6:] = r
    masks = [sum(1 << i for i in comb) for comb in itertools.combinations(range(10), 6)]
    index = {mask: i for i, mask in enumerate(masks)}
    H = np.zeros((len(masks), len(masks)), complex)
    def act(mask: int, i: int, create: bool) -> tuple[int, int]:
        if bool(mask & (1 << i)) == create: return mask, 0
        sign = (-1)**((mask & ((1 << i)-1)).bit_count())
        return mask ^ (1 << i), sign
    for col, mask in enumerate(masks):
        for j in range(10):
            m, s = act(mask, j, False)
            if not s: continue
            for i in range(10):
                mm, ss = act(m, i, True)
                if ss: H[index[mm], col] += s*ss*body[i, j]
    neutral = [(1 << 2)-1 | (15 << 6), (1 << 6)-1]
    charged = []; signs = []
    for q in (1, -1):
        for ref in range(4):
            for i in (range(2, 6) if q == 1 else range(6)):
                m, s = act(neutral[0 if q == 1 else 1], 6+ref if q == 1 else i, False)
                mm, ss = act(m, i if q == 1 else 6+ref, True)
                charged.append(index[mm]); signs.append(s*ss)
    evolution = expm(-1j*H)
    exact = np.asarray(signs)[:, None]*evolution[np.ix_(charged, charged)]*np.asarray(signs)[None, :]
    kernel = _cross_blocks(expm(-1j*a), expm(-1j*r), 2)['charged']
    projected = np.asarray(signs)[:, None]*H[np.ix_(charged, charged)]*np.asarray(signs)[None, :]
    wrong = expm(-1j*projected)
    return dict(full_sparseCAR_cross_error=float(np.linalg.norm(exact-kernel)),
                projection_before_Slater_wrong_error=float(np.linalg.norm(wrong-kernel)),
                nontrivial_PH_generator_norm=float(np.linalg.norm(a[:2, 2:])), Ref_generator_offdiagonal_norm=float(np.linalg.norm(r-np.diag(np.diag(r)))))


def relative_initial_factor(model: Any, base: dict[str, Any], sample: dict[str, Any]) -> float:
    q, qb = sample['q'], base['q']
    if base['cutoff'] == 0 or sample['cutoff'] == 0:
        raise ValueError('Directional family outside original compact support')
    return float(np.exp(-(q@q-qb@qb)/(2*model.sigma**2)+.5*(base['point']['log_density']-sample['point']['log_density']))*sample['cutoff']/base['cutoff'])


def endpoint_tangent(model: Any, base: dict[str, Any], minus: dict[str, Any],
                     plus: dict[str, Any], h: float) -> np.ndarray:
    """Full2835 native Cartesian Phi and LEFT Haar link tangent at base endpoint."""
    p0 = base['native_provider'](base['time_interval'][1])['raw']
    pm = minus['native_provider'](minus['time_interval'][1])['raw']
    pp = plus['native_provider'](plus['time_interval'][1])['raw']
    dphi = coordinates((pp['phi_endpoint']-pm['phi_endpoint'])/(2*h), model.law.basis).reshape(model.law.volume, 60)
    du = (pp['links_endpoint']-pm['links_endpoint'])/(2*h)
    generator = -1j*du@p0['links_endpoint'].conj().swapaxes(-1, -2)
    dl = algebra_coordinates(model.law, generator).reshape(model.law.volume, 45)
    return np.column_stack((dphi, dl)).ravel()


def frame_cancellation(model: Any, data: dict[str, Any], frame: dict[str, Any],
                       base: dict[str, Any]) -> dict[str, Any]:
    """Verify full frame products on each varied function, not a fitted phase."""
    gauge = frame['section_inverse']['gauge']
    block = np.zeros((4*len(gauge),)*2, complex)
    for i, g in enumerate(gauge): block[4*i:4*i+4, 4*i:4*i+4] = g
    products = {name: float(np.linalg.norm(block@frame['unitaries'][name]-frame['raw_unitaries'][name]))
                for name in ('light', 'heavy')}
    products['reference'] = float(np.linalg.norm(gauge[model.reference_site]@frame['unitaries']['reference']-frame['raw_unitaries']['reference']))
    coefficient = {s: float(np.linalg.norm(frame['boost_inverse_coefficient']*frame['carrier_sample'][s]-frame['physical_sample'][s])/np.linalg.norm(frame['physical_sample'][s]))
                   for s in ('neutral', 'charged')}
    kernels = full_cross_kernels(model, frame['raw_unitaries'], frame['raw_unitaries'])
    factor = relative_initial_factor(model, base['initial_sample'], frame['initial_sample'])
    state_rows = {}
    for name, sector, state in (('neutral', 'neutral', data['neutral_state']),
                                ('source19', 'charged', data['source19_normalized_source_state']),
                                ('SOFT', 'charged', data['SOFT_normalized_source_state'])):
        v = factor*frame['physical_sample'][sector]@state
        state_rows[name] = dict(full_cross_self_norm=float(np.vdot(v, kernels[sector]@v).real),
                               fixed_coefficient_dimension=len(state))
    heavy = model.heavy_orbitals
    heavy_amplitude = np.linalg.det(heavy.conj().T@frame['raw_unitaries']['heavy']@heavy)
    if max(products.values()) > 1e-10 or max(coefficient.values()) > 1e-12:
        raise AssertionError('Section gauge/W product cancellation failed')
    return dict(initial_q_norm=float(np.linalg.norm(frame['initial_q'])),
        initial_relative_Gaussian_Haar_cutoff_factor=factor, scalar_integral=frame['scalar_integral'],
        scalar_phase=frame['scalar_phase'], heavy_occupied54_absolute_overlap_with_seed=heavy_amplitude,
        full_section_gauge_times_section_U_minus_RAW_U=products,
        W_inverse_times_carrier_minus_physical_relative_errors=coefficient,
        halfdensity_log=frame['common_halfdensity_log'], state_rows=state_rows)


def directional_row(model: Any, data: dict[str, Any], base: dict[str, Any],
                    minus: dict[str, Any], plus: dict[str, Any], h: float) -> dict[str, Any]:
    tangent = endpoint_tangent(model, base, minus, plus, h)
    massnorm = float(tangent@(model.chart.mass*tangent))
    kernels = {pair: full_cross_kernels(model, a['raw_unitaries'], b['raw_unitaries'])
               for pair, a, b in (('mm', minus, minus), ('pp', plus, plus), ('mp', minus, plus))}
    rows = {}
    for name, sector, state in (('neutral', 'neutral', data['neutral_state']),
                                ('source19', 'charged', data['source19_normalized_source_state']),
                                ('SOFT', 'charged', data['SOFT_normalized_source_state'])):
        vectors = []
        for frame in (minus, plus):
            factor = relative_initial_factor(model, base['initial_sample'], frame['initial_sample'])
            vectors.append(factor*frame['physical_sample'][sector]@state)
        vm, vp = vectors
        nm = float(np.vdot(vm, kernels['mm'][sector]@vm).real)
        np_ = float(np.vdot(vp, kernels['pp'][sector]@vp).real)
        cross = complex(np.vdot(vm, kernels['mp'][sector]@vp))
        difference = nm+np_-2*cross.real
        wrongcross = cross*minus['scalar_phase']*plus['scalar_phase'].conjugate()
        wrong = nm+np_-2*wrongcross.real
        if difference < 0: raise ArithmeticError('Negative full cross finite difference; roundoff is not clipped')
        rows[name] = dict(minus_self_norm=nm, plus_self_norm=np_, cross=cross,
            derivative_norm_squared=difference/(4*h*h), kinetic_directional_lowerbound=model.chi/2*difference/(4*h*h*massnorm),
            wrong_drop_original_scalar_phase_lowerbound=model.chi/2*wrong/(4*h*h*massnorm),
            subtraction_condition=(nm+np_+2*abs(cross.real))/difference,
            scalar_integral_plus_minus_difference=plus['scalar_integral']-minus['scalar_integral'],
            principal_full_cross_phase=float(np.angle(cross)))
    return dict(h=h, h_over_sigma=h/model.sigma, RAW_endpoint_tangent_mass_norm_squared=massnorm,
                RAW_endpoint_tangent_dimension=len(tangent), rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--h-divisors', type=int, nargs='+', default=[32, 64])
    args = parser.parse_args(); started = time.perf_counter()
    model, data = load_shared_native_cache(args.cache_npz, args.cache_manifest)
    q, em, g = source_point(model, data)
    direction = g['Y0']/np.linalg.norm(g['Y0'])
    clock = 566; t0 = .37*clock; t1 = t0+.031
    base = coupled_interval(model, data, q, em, clock, t0, t1, args.steps)
    controls = sparse_car_control()
    if controls['full_sparseCAR_cross_error'] > 1e-10 or controls['projection_before_Slater_wrong_error'] < .01:
        raise AssertionError('Independent CAR overlap/adverse projection control failed')
    rows = []; wick = []
    families = [frame_cancellation(model, data, base, base)]
    for divisor in args.h_divisors:
        h = model.sigma/divisor
        minus = coupled_interval(model, data, q-h*direction, em, clock, t0, t1, args.steps)
        plus = coupled_interval(model, data, q+h*direction, em, clock, t0, t1, args.steps)
        families.extend(frame_cancellation(model, data, f, base) for f in (minus, plus))
        rows.append(directional_row(model, data, base, minus, plus, h))
        # Actual physical RAW Ref is H_ref=0, so independently exercise a
        # nontrivial physical Ref cross family, without changing the science.
        rng = np.random.default_rng(222756)
        r = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)); r = .13*(r+r.conj().T)
        control_right = dict(plus['raw_unitaries'], reference=expm(-1j*r))
        wick.append(_wick_control(model, minus['raw_unitaries'], control_right))
        if wick[-1]['full108_Ref4_CAR_minor_max_error'] > 1e-9:
            raise AssertionError('Actual full108/Ref4 CAR cross family failed')
    root = Path(__file__).parent
    sources = {Path(cast(str, m.__file__)).name: hashlib.sha256(Path(cast(str, m.__file__)).read_bytes()).hexdigest()
               for m in tuple(sys.modules.values()) if getattr(m, '__file__', None) and Path(cast(str, m.__file__)).parent == root}
    loaded_stable = all(hashlib.sha256((root/name).read_bytes()).hexdigest() == sha for name, sha in sources.items())
    result = dict(scope='ONE original Phi positive directional lowerbound at one actual coupled endpoint; not cometric trace or L2',
        clock=clock, time_interval=[t0, t1], steps=args.steps, parameters=dict(sigma=model.sigma, chi=model.chi, Lambda=model.Lambda),
        direction='full mean physical Y0 / Euclidean norm Y0; all normal coordinates retained', refinements=rows, full_physical_families=families,
        sparseCAR_control=controls, actual_fullWick_controls=wick,
        initial_relative_factor='exp[-(qvar²-qbase²)/(2sigma²)] sqrt[rhobase/rhovar] cutoffvar/cutoffbase',
        neutral_state_dimension=214, fixed_globally_normalized_SOURCE_state_dimension=1728,
        no_refilter_normalizer_bank_recapture_phasefit_polarrepair_floor_or_clipping=True,
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(), manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        loaded_sources=sources, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        loaded_sources_stable_at_result=loaded_stable,
        unpaid=['full Phi cometric trace and global L2 form', 'whole-H propagation and full pulse', 'finite-h/CF4 numerical remainder', 'physical 2pi and Born', 'Root/Gauss/legal SOFT/quantum law and parameter closure'],
        wall_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(portable(result), indent=2, sort_keys=True))
    print(json.dumps(portable(result), sort_keys=True))


if __name__ == '__main__':
    main()
