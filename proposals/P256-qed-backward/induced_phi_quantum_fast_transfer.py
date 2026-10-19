"""Outward real-time raw Phi35 coordinate transfer in a NEW coupled PH Gaussian.

Exact algebraic POINT interpretation of the primitive binary64 scalar paths and
binary64 constructor mean. This is NOT an upgrade of frozen NumPy spectra or
an interacting mean. All other Phi/SU4/Pi coordinates are frozen. No original
SOURCE, common cone, interacting eR, or full-vacuum admission is made.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.abc
import importlib.machinery
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES: dict[str, bytes] = {Path(__file__).name: Path(__file__).read_bytes()}


class CapturedLoader(importlib.machinery.SourceFileLoader):
    """Execute the captured source, not a possibly cached bytecode epoch."""
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes()
        SOURCE_BYTES[Path(self.path).name] = raw
        return self.source_to_code(raw, self.path)


class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.origin is not None:
            source = Path(spec.origin)
            if source.parent == ROOT and source.suffix == '.py':
                spec.loader = CapturedLoader(fullname, str(source))
                return spec
        return None


_finder = CapturedFinder()
sys.meta_path.insert(0, _finder)
import induced_phi_quantum_oscillator_backreaction as supplier
sys.meta_path.remove(_finder)
import mpmath as mp
import numpy as np

iv: Any = mp.iv
mp.mp.dps = 80
sys.set_int_max_str_digits(0)
FROZEN_RECEIPT_SHA = 'e7a765bb739d29133a1f96bf807ceccdd18f6a19a743944b28372b34f4c5afd3'
FROZEN_SUPPLIER_SHA = '404dfbee04a0f8d0ba449c396881675c5763464f0aa81061b69bffc2e6b3ec4c'
DIGITS = 1250


def exact(value: Any) -> Any:
    f = value if isinstance(value, Fraction) else Fraction(value)
    return iv.mpf(f.numerator) / f.denominator


def binary(value: Any) -> Fraction:
    return Fraction.from_float(float(value))


def endpoint(value: Any, which: int) -> Fraction:
    sign, mantissa, exponent, _ = value._mpi_[which]
    return Fraction((-1)**sign * mantissa) * Fraction(2)**exponent


def enclosure(lower: Fraction, upper: Fraction) -> Any:
    if lower > upper:
        raise ArithmeticError('Empty outward interval')
    return iv.mpf([exact(lower).a, exact(upper).b])


def bounds(value: Any) -> dict[str, str]:
    lo, hi = endpoint(value, 0), endpoint(value, 1)
    def display(f: Fraction) -> str:
        return str(mp.nstr(mp.mpf(f.numerator) / f.denominator, 40))
    return dict(lower=str(lo), upper=str(hi), lower_decimal=display(lo), upper_decimal=display(hi))


def frozen_number(value: dict[str, str]) -> Fraction:
    return (Fraction(value['lower']) + Fraction(value['upper'])) / 2


def point_mean(mean: dict[str, Any]) -> tuple[Fraction, Fraction]:
    enc = mean['centre_enclosure']
    values = [binary((supplier.loops.rational(enc[name+'_lower']) +
                      supplier.loops.rational(enc[name+'_upper'])) / 2)
              for name in ('g0', 'gs')]
    return values[0], values[1]


def primitive_profiles(mean: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = supplier.primitive.PhiConfig(Lambda=1., **mean['microscopic_config'])
    records = []
    for load, model, role in supplier.primitive.models(cfg):
        onsite, paths = supplier.primitive.scalar_paths(model)
        scalar = []
        for displacement, coefficients in paths:
            support = np.flatnonzero(displacement)
            if len(support) != 1:
                raise ArithmeticError('Primitive path is not axial')
            axis = int(support[0]); length = int(displacement[axis])
            if length not in (1, 2) or any(complex(coefficients[j]) != 0 for j in range(1, 4) if j != axis+1):
                raise ArithmeticError('Actual kinetic coefficient is not axis+1')
            scalar.append(dict(axis=axis, length=length,
                               c0=binary(complex(coefficients[0]).real),
                               ci=binary(complex(coefficients[axis+1]).imag)))
        records.append(dict(role=role, load=int(load), model=model,
                            onsite=binary(onsite), paths=scalar,
                            raw_paths=paths, scale='heavy' if role.startswith('neutral') else 'charged'))
    if [p['load'] for p in records] != [1, 1, 63] or records[0]['model'].improved:
        raise ArithmeticError('Actual one neutral heavy / all64 charged profile contract changed')
    return records


def algebraic_ph(profile: dict[str, Any], g0: Fraction, gs: Fraction) -> list[dict[str, Any]]:
    root0, roots = iv.sqrt(exact(g0)), iv.sqrt(exact(gs))
    halfroot = iv.sqrt(exact(2)) / 2
    cos1, sin1, sin2 = (1, -1, -1, 1), (1, 1, -1, -1), (1, -1, 1, -1)
    bands = {}
    for k in product(range(4), repeat=3):
        xi0 = exact(profile['onsite']); hs = [iv.mpf(0), iv.mpf(0), iv.mpf(0)]
        for p in profile['paths']:
            axis, length = p['axis'], p['length']
            cosine = cos1[k[axis]] * halfroot if length == 1 else iv.mpf(0)
            sine = sin1[k[axis]] * halfroot if length == 1 else exact(sin2[k[axis]])
            xi0 += 2 * exact(p['c0']) * cosine
            hs[axis] -= 2 * exact(p['ci']) * sine
        h = [root0 * xi0, *(roots * x for x in hs)]
        energy = iv.sqrt(sum(x*x for x in h))
        bands[k] = dict(xi0=xi0, h=h, E=energy)
    points = []
    for k, incoming in bands.items():
        for direction in (-1, 1):
            out = ((k[0]+direction) % 4, k[1], k[2]); outgoing = bands[out]
            hi, ho = incoming['h'], outgoing['h']
            dot = sum(ho[j]*hi[j] for j in (0, 1, 2)) - ho[3]*hi[3]
            xiavg = (incoming['xi0'] + outgoing['xi0']) / 2
            weight = xiavg*xiavg / (16*64) * (1-dot/(incoming['E']*outgoing['E']))
            if endpoint(weight, 0) < 0:
                raise ArithmeticError('Algebraic PH weight lost positivity')
            points.append(dict(k_in=k, q_sign=direction, k_out=out,
                               delta=incoming['E']+outgoing['E'], weight=weight))
    if len(points) != 128:
        raise ArithmeticError('Both q directions / full AP N4 PH support required')
    return points


def native_ph(mean: dict[str, Any], profiles: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Independent literal 256x256 AP h/F consumer, no algebraic projectors."""
    native = supplier.quantum.build_native(mean, 4)
    zero = np.zeros(3*native['V']); mode = supplier.pauli.magnetic_shape(native)
    rows = []
    for p in profiles:
        if p['scale'] == 'charged':
            matches = [s for s in native['Pprofiles'] if s['label'] == p['role']]
            if len(matches) != 1:
                raise ArithmeticError('Missing native charged profile')
            profile = matches[0]; sign = 1
        else:
            paths = []
            for displacement, coefficients in p['raw_paths']:
                support = [s for s in native['Pprofiles'][0]['paths']
                           if np.array_equal(s['displacement'], displacement)]
                if len(support) != 64:
                    raise ArithmeticError('Actual heavy path/seam support mismatch')
                paths.extend({**s, 'scalars': coefficients,
                              'hop': s['seam'] * np.einsum('m,mij->ij', coefficients, native['frame'])}
                             for s in support)
            profile = dict(model=p['model'], load=1, label=p['role'],
                           onsite=float(p['onsite'])*native['frame'][0], paths=paths)
            sign = -1
        h = sign * supplier.quantum.matter_hamiltonian(native, profile, zero)
        force = sign * supplier.pauli.operator(native, profile, zero, mode)
        energy, vectors = np.linalg.eigh(h)
        if not energy[127] < 0 < energy[128]:
            raise ArithmeticError('Literal physical filled band not separated')
        amplitude = vectors[:, :128].conj().T @ force @ vectors[:, 128:]
        weight = abs(amplitude)**2
        delta = energy[None, 128:] - energy[:128, None]
        rows.append(dict(role=p['role'], load=p['load'], physical_filling='upper h band for physical -Lambda h' if sign == -1 else 'negative h band',
                         unscaled_Frobenius_squared=float(weight.sum()),
                         unscaled_static_bubble=float(np.sum(2*weight/delta)),
                         unscaled_first_PH_moment=float(np.sum(weight*delta)),
                         unscaled_dynamic_bubbles={str(om): float(np.sum(2*delta*weight/(delta*delta+om*om))) for om in (0, 1, 10, 100)},
                         literal_PH_matrix_shape=list(weight.shape), literal_gap=float(delta.min())))
    return native, rows


def compare_ph(profiles: list[dict[str, Any]], points: list[list[dict[str, Any]]], native: list[dict[str, Any]], frozen: dict[str, Any]) -> list[dict[str, Any]]:
    frozen_rows = frozen['actual_quantum_Phi_heavy_charged_PH_oscillator_backreaction']['actual_ALLspecies_spectral_force_and_energy_consumers']
    comparisons = []
    for p, pts, measured in zip(profiles, points, native, strict=True):
        old = next(row for row in frozen_rows if row['role'] == p['role'])
        weight = sum(row['weight'] for row in pts)
        bubble = sum(2*row['weight']/row['delta'] for row in pts)
        first = sum(row['weight']*row['delta'] for row in pts)
        scale = Fraction(2)**8192 if p['scale'] == 'heavy' else Fraction(1)
        frozen_w = float(frozen_number(old['actual_unscaled_Phi35_PH_Frobenius_squared']))
        frozen_b = float(frozen_number(old['actual_numerical_spectral_Phi35_static_bubble']) / (p['load']*scale))
        values = []
        for label, algebraic, actual in [('native Frobenius', weight, measured['unscaled_Frobenius_squared']),
                                          ('native static', bubble, measured['unscaled_static_bubble']),
                                          ('native first PH moment', first, measured['unscaled_first_PH_moment']),
                                          ('frozen e7a Frobenius', weight, frozen_w),
                                          ('frozen e7a unscaled static', bubble, frozen_b)]:
            relative = abs(float(endpoint(algebraic, 0))-actual) / max(abs(actual), 1e-300)
            if relative > 2e-11:
                raise ArithmeticError(f'{p["role"]}: independent {label} PH mismatch {relative}')
            values.append(dict(quantity=label, observed=actual, relative_difference=relative))
        dynamic = []
        for om in (0, 1, 10, 100):
            value = sum(2*row['delta']*row['weight']/(row['delta']**2+om*om) for row in pts)
            actual = measured['unscaled_dynamic_bubbles'][str(om)]
            relative = abs(float(endpoint(value, 0))-actual) / max(abs(actual), 1e-300)
            if relative > 2e-11:
                raise ArithmeticError('Actual PH shape consumer disagrees')
            dynamic.append(dict(frequency=om, algebraic=bounds(value), observed_native=actual, relative_difference=relative))
        comparisons.append(dict(role=p['role'], load=p['load'], algebraic_unscaled_static=bounds(bubble),
                                algebraic_unscaled_Frobenius_squared=bounds(weight),
                                algebraic_unscaled_first_PH_moment=bounds(first), comparisons=values,
                                actual_frequency_shape_consumer=dynamic, native=measured,
                                validation_scope='Numerical agreement only; e7a binary64 eigenvalues remain observed controls, never exact phase inputs.'))
    return comparisons


def coupled_transfer(profiles: list[dict[str, Any]], points: list[list[dict[str, Any]]], g0: Fraction) -> dict[str, Any]:
    Lambda = exact(Fraction(2)**8192); G = exact(8*Fraction(2)**-4096)
    kappa = exact(Fraction(2)**10240) * 2/40 + Lambda*exact(g0)/(2*exact(Fraction(1, 64)))
    A = G*kappa; free_omega = iv.sqrt(A); free_var = G/(2*free_omega)
    modes = []
    for p, pts in zip(profiles, points, strict=True):
        scale = Lambda if p['scale'] == 'heavy' else exact(1)
        for row in pts:
            delta = scale*row['delta']; w = p['load']*scale*scale*row['weight']
            modes.append(dict(delta=delta, d2=delta*delta, numerator=2*delta*w,
                              scale=p['scale']))
    heavy = [m for m in modes if m['scale'] == 'heavy']
    charged = [m for m in modes if m['scale'] == 'charged']
    Bstatic = sum(m['numerator']/m['d2'] for m in modes)
    epsilon_static = Bstatic/kappa
    if endpoint(epsilon_static, 1) >= 1:
        raise ArithmeticError('Full coupled positive Gaussian Schur complement failed')
    heavy_gap2 = min(endpoint(m['d2'], 0) for m in heavy)
    charged_top2 = max(endpoint(m['d2'], 1) for m in charged)
    ratio = A/exact(heavy_gap2)
    if endpoint(ratio, 1) >= 1:
        raise ArithmeticError('Fast pole is not below every heavy gap')
    heavy_static = sum(m['numerator']/m['d2'] for m in heavy)
    # Factor two makes the lower-end sign strict while keeping a heavy-only price.
    epsilon_heavy_dynamic = 2*heavy_static/kappa/(1-ratio)
    lower = 1-endpoint(epsilon_heavy_dynamic, 1)
    initial = enclosure(lower, Fraction(1))
    if lower <= 0 or charged_top2 >= endpoint(A*exact(lower), 0):
        raise ArithmeticError('Chargedgap << fastpole << heavygap separation failed')

    def fixed(z: Any) -> Any:
        return 1-sum(m['numerator']/(m['d2']-A*z) for m in modes)/kappa

    def derivative(z: Any) -> Any:
        return G*sum(m['numerator']/(m['d2']-A*z)**2 for m in modes)

    top_value = fixed(exact(1)); low_value = fixed(exact(lower))
    top_sign = top_value-1; low_sign = low_value-exact(lower)
    contraction = derivative(initial)
    if endpoint(top_sign, 1) >= 0 or endpoint(low_sign, 0) <= 0 or endpoint(contraction, 1) >= Fraction(1, 2):
        raise ArithmeticError('Outward pole existence/contraction sign criterion failed')
    z = initial; iteration_rows = []
    for iteration in range(12):
        image = fixed(z)
        lo = max(endpoint(z, 0), endpoint(image, 0))
        hi = min(endpoint(z, 1), endpoint(image, 1))
        new = enclosure(lo, hi)
        iteration_rows.append(dict(iteration=iteration+1, width=bounds(new.b-new.a),
                                   inclusion='z_root in previous bracket intersect F(previous bracket)'))
        if endpoint(new.b-new.a, 1) == endpoint(z.b-z.a, 1):
            z = new
            break
        z = new
    x = A*z; omega = iv.sqrt(x)
    Z = G/(1+derivative(z)); fast_weight = Z/(2*omega)
    var_upper = free_var/(1-epsilon_static)
    tail_upper = endpoint(var_upper, 1)-endpoint(fast_weight, 0)
    if tail_upper < 0:
        raise ArithmeticError('Gaussian all-other-poles spectral measure became negative')
    tail = exact(tail_upper)
    residual = fixed(z)-z
    residual_price = iv.mpf([-1, 1])*exact(max(abs(endpoint(residual, 0)), abs(endpoint(residual, 1)))) / (1-contraction)
    relative_tail = tail/free_var
    times = []
    for t in (Fraction(0), Fraction(1, 4096), Fraction(4096)):
        phase = omega*exact(t)
        re = fast_weight*iv.cos(phase)
        im = -fast_weight*iv.sin(phase)
        re_full = re + enclosure(-tail_upper, tail_upper)
        im_full = im + enclosure(-tail_upper, tail_upper)
        if t == 0:
            # Positive spectral measure and resolvent monotonicity sharpen t=0.
            re_full = enclosure(max(endpoint(free_var, 0), endpoint(fast_weight, 0)), endpoint(var_upper, 1))
            im_full = exact(0)
        phase_width = phase.b-phase.a
        times.append(dict(time=str(t), physical_time_units='primitive continuous-time H units, hbar=1; clock4096 unchanged',
                          omega_t_outward=bounds(phase), phase_width_price=bounds(phase_width),
                          fast_cos=bounds(iv.cos(phase)), fast_minus_sin=bounds(-iv.sin(phase)),
                          fast_raw_real=bounds(re), fast_raw_imag=bounds(im),
                          actual_raw_kernel_real=bounds(re_full), actual_raw_kernel_imag=bounds(im_full),
                          ALL_other_poles_complex_disk_radius_upper=bounds(tail),
                          phase_only_kernel_absolute_error_upper=bounds(fast_weight*phase_width),
                          raw_coordinate_normalization='No Gamma whitening, no charged phase, no own source normalization.'))
    return dict(G=bounds(G), kappa=bounds(kappa), bare_frequency_squared=bounds(A),
                free_variance=bounds(free_var), epsilon_static=bounds(epsilon_static),
                epsilon_heavy_dynamic=bounds(epsilon_heavy_dynamic), charged_PH_gap_squared_upper=str(charged_top2),
                heavy_PH_gap_squared_lower=str(heavy_gap2),
                charged_denominator_upper=bounds(exact(charged_top2)-A*exact(lower)),
                heavy_denominator_lower=bounds(exact(heavy_gap2)-A),
                initial_z_bracket=bounds(initial), F_one_minus_one=bounds(top_sign), F_lower_minus_lower=bounds(low_sign),
                contraction_absolute_derivative=bounds(contraction), intersection_iterations=iteration_rows,
                zroot_outward=bounds(z), fixed_point_residual=bounds(residual), root_distance_residual_price=bounds(residual_price),
                collective_omega_squared=bounds(x), collective_omega=bounds(omega), coordinate_residue_Z=bounds(Z),
                fast_Phi_spectralweight=bounds(fast_weight), total_coupled_variance_envelope=dict(lower=bounds(free_var), upper=bounds(var_upper)),
                ALL_other_poles_spectralweight_upper=bounds(tail), tail_relative_to_free_variance=bounds(relative_tail),
                full_times_rawcoordinate_transfer=times,
                exact_positive_PH_Hamiltonian='H=G Pi_Phi^2/2+kappa Phi^2/2+sum Delta bdag b+Phi sum sqrt(F^2)(b+bdag); each of 128 points/profile with spin-summed F^2 and actual loads. Vacuum energy subtracted only globally.',
                resolvent_proof='Mass-normalized force matrix D00=Gkappa, Djj=Delta_j^2, D0j=sqrt(2G Delta_j F_j^2). Schur complement kappa-Bstatic>0 proves D positive. R00(s)=G/[Gkappa-s-G sum 2Delta F^2/(Delta^2-s)]; real isolated fast root residue Z. Spectral theorem gives Phi covariance sum Z_j/(2omega_j), all nonnegative. Euclidean Schur resolvent lies between free and free/(1-eps_static), hence same variance bound after integration. Subtract fastweight LOWER from total variance UPPER to price ALL other poles; no pole omitted. K(t)=sum positive weight_j exp(-i omega_j t).',
                units='Phi is actual Cart-unit Phi35 mode; canonical [Phi,Pi_Phi]=i, hbar1; gap and omega inverse primitive time; F=dH/dPhi, kappa energy/Phi^2, G energy/Pi_Phi^2. Physical heavy gaps Lambda delta and weights Lambda^2 w; active1+aux63 charged unscaled.')


def run(path: Path) -> dict[str, Any]:
    started = time.monotonic(); raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_RECEIPT_SHA:
        raise ArithmeticError('Frozen e7a oscillator input required')
    if hashlib.sha256(SOURCE_BYTES[Path(supplier.__file__).name]).hexdigest() != FROZEN_SUPPLIER_SHA:
        raise ArithmeticError('Frozen oscillator source404d required')
    frozen = json.loads(raw); mean = frozen['input_descriptor']['actual_root_mean_input']
    iv.dps = DIGITS
    g0, gs = point_mean(mean)
    profiles = primitive_profiles(mean)
    points = [algebraic_ph(p, g0, gs) for p in profiles]
    native, native_rows = native_ph(mean, profiles)
    validation = compare_ph(profiles, points, native_rows, frozen)
    physical = coupled_transfer(profiles, points, g0)
    desc = dict(plan=frozen['input_descriptor']['plan'],
                scientific_execution_revision=frozen['input_descriptor']['scientific_execution_revision'],
                governing_plan_revision_at_binding='452ad8053dd6efcbc7d30b55443c1da57c4e6b7e6866b58b11fc61284c6d0961',
                input_receipts=dict(frozen_oscillator=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())),
                supplier_epochs=dict(frozen_oscillator_source=FROZEN_SUPPLIER_SHA, frozen_oscillator_receipt=FROZEN_RECEIPT_SHA),
                declared_NEW_POINT_parameters=dict(g0_exact_binary64=str(g0), gs_exact_binary64=str(gs),
                    g0_float=float(g0), gs_float=float(gs),
                    interpretation='Exactly interpret float(midpoint of frozen constructor centre enclosure); algebraic sqrt thereafter. NEW microscopic Gaussian point, not true interacting mean or upgraded old NumPy spectra.',
                    native_numpy_frame_squared_trace=dict(g0=native['g0'], gs=native['gs'])),
                primitive_binary64_scalar_profiles=[dict(role=p['role'], load=p['load'], onsite=str(p['onsite']),
                    paths=[{**s, 'c0': str(s['c0']), 'ci': str(s['ci'])} for s in p['paths']]) for p in profiles],
                scalar_Phi35_mode='f=sqrt(2/64)sin(pi x/2); deltaPhi0=Gamma0(iGamma1Gamma2)/sqrt8; other components0; Cart unit normal',
                source_vertex='F_q=xi0avg Gamma0 iGamma1Gamma2/sqrt8 times +/-i/sqrt(2V), q=+/-pi/2 x; summed spin w=xi0avg^2/(16V)*(1-(h0o h0i+h1o h1i+h2o h2i-h3o h3i)/(Eo Ei))',
                actual_parameters=dict(Lambda='2^8192', chi='2^-4096', a='1/64', g_Pi_AND_Coulomb='64', N=4, original_clock_UNCHANGED='4096'),
                exact_AP_trig='k=(2pi n+pi)/4; cos1 signed sqrt2/2, sin1 signed sqrt2/2, cos2=0, sin2 signed1',
                vacuum='ONE common neutral positive coupled Phi+PH Gaussian vacuum; used equally at every time/source. No charged-sector own phase or normalizer.',
                scope='NEW scalarPhi35 + all64 charged/heavy PH Gaussian, frozen otherPhi/SU4/Pi. Not original SOURCE or full interacting vacuum; no commoncone/eR/anomaly promotion.',
                outward_decimal_precision=DIGITS, physical_times=['0', '1/4096', '4096'])
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(),
                actual_independent_native_PH_shape_consumers=validation,
                actual_realtime_coupled_Phi_coordinate_transfer=physical,
                exact_algebraic_PH_points=[dict(role=p['role'], load=p['load'],
                    rows=[dict(k_in=r['k_in'], q_sign=r['q_sign'], k_out=r['k_out'], unscaled_delta=bounds(r['delta']), unscaled_spin_PH_weight=bounds(r['weight'])) for r in pts])
                    for p, pts in zip(profiles, points, strict=True)],
                missing_full_law_obligations=['nonlinear Phi bdag b couplings beyond linear PH Gaussian',
                    'full Phi quartic / nonuniform field and all45/SU4 quantum dynamics',
                    'primitive Pi/Coulomb interactions and Noether/Gauss physical common-vacuum lift',
                    'stationary full interacting electron/photon poles, common cone, interacting eR and physical scattering closure'],
                source_at_import={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name, data in SOURCE_BYTES.items()},
                source_unchanged_since_import={name: (ROOT/name).read_bytes() == data for name, data in SOURCE_BYTES.items()},
                runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, mpmath=mp.__version__, interval_digits=iv.dps),
                elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oscillator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.oscillator)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source epoch changed during actual real-time transfer')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    transfer = result['actual_realtime_coupled_Phi_coordinate_transfer']
    print(json.dumps(dict(input_sha256=result['input_sha256'],
        times=[dict(time=t['time'], real=t['actual_raw_kernel_real'], imag=t['actual_raw_kernel_imag'],
                    phase_width=t['phase_width_price']) for t in transfer['full_times_rawcoordinate_transfer']],
        tail_relative=transfer['tail_relative_to_free_variance'], elapsed_seconds=result['elapsed_seconds']), indent=2))
