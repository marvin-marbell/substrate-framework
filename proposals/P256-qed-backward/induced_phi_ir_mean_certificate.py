#!/usr/bin/env python3
"""Exact-finite discrete-measure enclosure for the NEW issue222 IR box.

The Gaussian rules integrate consecutive INTEGER indices, not Lebesgue measure.
The physical centres are implicit finite ten-component roots. Displayed midpoints
are computational witnesses only. All-sector selection is deliberately unresolved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import mpmath as mp
from mpmath.libmp import to_str
import numpy as np
from scipy.linalg import eigh
from induced_phi_backreaction import PhiConfig, SELECTED, clifford_cartesian_basis, models, scalar_paths, IMPORT_BYTES

ROOT = Path(__file__).resolve().parent
OWN_BYTES = Path(__file__).read_bytes()
SOURCE_BYTES = dict(IMPORT_BYTES)
SOURCE_BYTES[Path(__file__).name] = OWN_BYTES
N = 262144
EXPONENT = 8192
ORDER = 10
LD = np.longdouble
# mpmath's interval context has incomplete third-party operator stubs.
# Keep the dynamic boundary here; all scientific bounds remain actual intervals.
IV: Any = mp.iv
PI = LD('3.1415926535897932384626433832795028841971693993751')
G_WITNESS = (Fraction('0.2901803527975645'), Fraction('0.10293343620392745'))
GMIN = Fraction(1, 10)
GMAX = Fraction(3, 10)
ROUND_PRICE = Fraction(1, 10**10)
RULE_RECORDS: dict[int, dict[str, Any]] = {}
HEAVY_ONSITE, HEAVY_PATHS = scalar_paths(models(PhiConfig())[0][1])
HEAVY_MASS = LD(float(-2*HEAVY_PATHS[0][1][0].real))
HEAVY_SPEED = LD(float(-2*HEAVY_PATHS[0][1][1].imag))


def encoded(value: Fraction | int | float) -> dict[str, str]:
    f = value if isinstance(value, Fraction) else Fraction(value)
    return {'numerator': str(f.numerator), 'denominator': str(f.denominator),
            'decimal_display': str(float(f))}


def upper(value: Any) -> Fraction:
    """Outward upper endpoint of a genuine MPInterval evaluation."""
    return Fraction(str(to_str(value._mpi_[1], 90, strip_zeros=False))) + Fraction(1, 10**85)


def iv(value: Fraction | int | str) -> Any:
    if isinstance(value, Fraction):
        return IV.mpf(value.numerator) / value.denominator
    return IV.mpf(value)


@lru_cache(maxsize=None)
def rule(count: int):
    """Hahn/Gram Jacobi rule with interval residual/orthogonality/gap proof.

    Scale consecutive integers to [-1,1]. Exact Jacobi coefficients are rational
    squared offdiagonals. Approximate eigensystems are NOT presumed exact rules.
    An interval residual and a disjoint eigenvalue gap certify both nodes/weights.
    """
    order = min(ORDER, count)
    if count == 1:
        return np.array([0], dtype=LD), np.array([1], dtype=LD)
    mp.mp.dps = 75
    IV.dps = 70
    matrix = mp.zeros(order)
    actual = [[iv(0) for _ in range(order)] for _ in range(order)]
    for k in range(1, order):
        square = Fraction(k*k*(count*count-k*k), (4*k*k-1)*(count-1)**2)
        matrix[k-1, k] = matrix[k, k-1] = mp.sqrt(mp.mpf(square.numerator)/square.denominator)
        actual[k-1][k] = actual[k][k-1] = IV.sqrt(iv(square))
    eigenvalues, vectors = mp.eigsy(matrix)
    ds = [Fraction(str(mp.nstr(eigenvalues[k], 65))) for k in range(order)]
    vs = [[Fraction(str(mp.nstr(vectors[j, k], 65))) for k in range(order)] for j in range(order)]
    residual = iv(0)
    orthogonal = iv(0)
    for j, k in product(range(order), repeat=2):
        entry = sum(actual[j][l]*iv(vs[l][k]) for l in range(order))-iv(vs[j][k]*ds[k])
        residual += abs(entry)**2
        entry = iv(sum(vs[l][j]*vs[l][k] for l in range(order)) - int(j == k))
        orthogonal += abs(entry)**2
    residual_bound = upper(IV.sqrt(residual))
    orthogonal_bound = upper(IV.sqrt(orthogonal))
    # Nearest orthogonal polar factor differs by <=2||V^T V-I||.
    spectral_error = 4*residual_bound+16*orthogonal_bound
    gap = min(ds[k+1]-ds[k] for k in range(order-1))-2*spectral_error
    if gap <= 0 or orthogonal_bound >= Fraction(1, 100):
        raise ArithmeticError('Discrete Jacobi eigenvalue certification failed')
    vector_error = 8*(residual_bound+orthogonal_bound)/gap
    weight_error = 4*vector_error+4*orthogonal_bound
    nodes = np.array([mp.nstr(eigenvalues[k], 65) for k in range(order)], dtype=LD)
    weights = np.array([mp.nstr(vectors[0, k]**2, 65) for k in range(order)], dtype=LD)
    # Cast perturbations are separately paid; this is a checked platform property.
    if spectral_error > Fraction(1, 10**50) or weight_error > Fraction(1, 10**50):
        raise ArithmeticError('Actual rule precision does not support arithmetic envelope')
    RULE_RECORDS[count] = dict(count=count, order=order,
        exact_squared_offdiagonal='k^2 (M^2-k^2)/((4k^2-1)(M-1)^2)',
        interval_residual_upper=encoded(residual_bound),
        interval_orthogonality_upper=encoded(orthogonal_bound),
        interval_spectral_error_upper=encoded(spectral_error),
        certified_minimum_gap_lower=encoded(gap),
        nodewise_weight_error_upper=encoded(weight_error),
        sum_weights_witness=str(sum(weights)),
        exactness_degree=2*order-1,
        theorem='Spectral measure of exact finite Jacobi matrix at e0; positive Gauss weights; exact discrete integer moments.')
    return nodes, weights


def trig(theta):
    """Polynomial sin/cos, no libm error assumption.

    |theta|<=pi; alternating Taylor through degrees 55/54. The exact remainders
    are <=pi^56/56!<2^-140; gamma_300 double-extended arithmetic is <2^-50.
    All callers pay a much larger absolute 1e-14 trigonometric enclosure.
    """
    square = theta*theta
    sine = theta.copy()
    cosine = np.ones_like(theta)
    st = theta.copy()
    ct = np.ones_like(theta)
    for k in range(1, 28):
        st *= -square/((2*k)*(2*k+1))
        ct *= -square/((2*k-1)*(2*k))
        sine += st
        cosine += ct
    return sine, cosine


def checked_sqrt(value):
    result = np.sqrt(value)
    residual = np.max(np.abs(result*result-value)/(value+LD('1e-60')))
    if residual > LD('1e-17') or np.min(value) <= 0:
        raise ArithmeticError('Square-root residual does not fit arithmetic envelope')
    return result


def cell_certificate(cell):
    """A actual-band analytic polydisc and Cauchy remainder for this finite cell."""
    centres = np.array([(lo+hi)/2 for lo, hi, _ in cell], dtype=LD)*2*PI/N
    halfwidth = max(LD(hi-lo)*PI/N for lo, hi, _ in cell)
    radius = 4*halfwidth
    if radius >= LD('0.75'):
        return None
    s, c = trig(centres)
    x0 = LD('.2')+LD('7.68')*np.sum(1-c)
    spatial = LD('19.2')*s
    real_square = LD(str(float(G_WITNESS[0])))*x0*x0+LD(str(float(G_WITNESS[1])))*np.sum(spatial*spatial)
    real_radius = checked_sqrt(np.array([real_square]))[0]
    rlower = real_radius-LD('1e-9')
    # |sin(w)|<=sinh(r)<=r/(1-r^2), |cos(w)-1|<=r^2/(2(1-r^2)).
    sh = radius/(1-radius*radius)
    ch = radius*radius/(2*(1-radius*radius))
    dc = np.abs(s)*sh+np.abs(c)*ch+LD('1e-12')
    ds = np.abs(c)*sh+np.abs(s)*ch+LD('1e-12')
    delta_square = LD('.3')*((LD('7.68')*np.sum(dc))**2+np.sum((LD('19.2')*ds)**2))
    delta = checked_sqrt(np.array([2*delta_square]))[0]+LD('1e-9')
    if delta > rlower/2:
        return None
    xbound = abs(x0)+np.sum(np.abs(spatial))+LD('7.68')*np.sum(dc)+LD('19.2')*np.sum(ds)+LD('1e-8')
    # Bounds for each second/fourth component and radius, on the polydisc.
    function_bound = max(2*xbound*xbound/rlower, 8*xbound**4/rlower**3, 2*xbound)
    tau = LD('.25')
    error = 6*function_bound*tau**(2*ORDER)/(1-tau)**3
    mass = math.prod((hi-lo+1)*mult for lo, hi, mult in cell)/N**3
    return float(error*mass), float(mass), float(rlower), float(delta)


def cells():
    segments = ((0, 0, 1), (1, N//2-1, 2), (N//2, N//2, 1))
    pending = list(product(segments, repeat=3))
    accepted = []
    total_error = LD(0)
    while pending:
        cell = pending.pop()
        certificate = cell_certificate(cell)
        if certificate is not None:
            accepted.append((cell, certificate))
            total_error += LD(certificate[0])
            continue
        axis = max(range(3), key=lambda j: cell[j][1]-cell[j][0])
        lo, hi, mult = cell[axis]
        if lo == hi:
            raise ArithmeticError('Analytic band gap failed on a singleton integer cell')
        mid = (lo+hi)//2
        for bounds in ((lo, mid, mult), (mid+1, hi, mult)):
            child = list(cell)
            child[axis] = bounds
            pending.append(tuple(child))
    represented_points = sum(math.prod((hi-lo+1)*mult for lo, hi, mult in cell)
                             for cell, _ in accepted)
    if represented_points != N**3:
        raise ArithmeticError('Adaptive integer boxes do not partition the exact finite measure')
    # The only numerical accumulation here is positive, with <1e6 additions.
    return accepted, Fraction(str(float(total_error*LD('1.00000001'))))+ROUND_PRICE


def evaluate(accepted):
    g = np.array([str(float(G_WITNESS[0])), *([str(float(G_WITNESS[1]))]*3)], dtype=LD)
    second = np.zeros(4, dtype=LD)
    fourth = np.zeros((4, 4), dtype=LD)
    energy = LD(0)
    points = 0
    for cell, certificate in accepted:
        coords = []
        weights = []
        for lo, hi, mult in cell:
            nodes, w = rule(hi-lo+1)
            coords.append(((LD(lo+hi)+LD(hi-lo)*nodes)/2)*2*PI/N)
            weights.append(w*LD((hi-lo+1)*mult)/N)
        grid = np.meshgrid(*coords, indexing='ij')
        theta = np.stack([z.ravel() for z in grid], axis=1)
        w = (weights[0][:, None, None]*weights[1][None, :, None]*weights[2][None, None, :]).ravel()
        sine, cosine = trig(theta)
        x = np.empty((len(w), 4), dtype=LD)
        x[:, 0] = LD(float(HEAVY_ONSITE))-HEAVY_MASS*np.sum(cosine, axis=1)
        x[:, 1:] = HEAVY_SPEED*sine
        square = x*x
        radius = checked_sqrt(square@g)
        second += np.sum(w[:, None]*square/radius[:, None], axis=0)
        for i, j in product(range(4), repeat=2):
            fourth[i, j] += np.sum(w*square[:, i]*square[:, j]/radius**3)
        energy += np.sum(w*radius)
        points += len(w)
    return second, fourth, energy, points


def normal_stiffness(second, fourth):
    """Frozen full60 witness Hessian, compared to exact HEAVY-classical root.

    Diagonal000 parity kills ONLY odd spatial moments. Shifted-sector odd/quarter
    moments are bounded by the separate finite Fourier-alias certificate.
    """
    basis = clifford_cartesian_basis()
    tensors = np.rint(np.einsum('nij,ajk,mkl,bli->namb', SELECTED, basis, SELECTED, basis).real/4).astype(int)
    g = np.array([float(G_WITNESS[0]), *([float(G_WITNESS[1])]*3)], dtype=LD)
    scales = checked_sqrt(g)
    h = np.zeros((60, 60), dtype=LD)
    for mu, a, nu, b in product(range(4), range(15), range(4), range(15)):
        value = -second[mu] if mu == nu and a == b else LD(0)
        for i, j in product(range(4), repeat=2):
            indices = (mu, nu, i, j)
            if any(indices.count(axis)%2 for axis in (1, 2, 3)):
                continue
            if len(set(indices)) == 1:
                moment = fourth[mu, mu]
            elif len(set(indices)) == 2 and all(indices.count(k) == 2 for k in set(indices)):
                moment = fourth[min(indices), max(indices)]
            else:
                continue
            value += int(tensors[i, a, j, b])*scales[i]*scales[j]*moment
        value += 256*((g[a] if mu == nu and a == b and a < 4 else 0)+
            (scales[nu]*scales[mu] if a == nu and b == mu else 0)+
            (g[mu] if mu == nu and a == b else 0))
        h[15*mu+a, 15*nu+b] = value
    eigenvalues, vectors = eigh(np.asarray(h, dtype=float))
    v = np.asarray(vectors, dtype=LD)
    d = np.asarray(eigenvalues, dtype=LD)
    residual = np.max(np.sum(np.abs(h@v-v*d), axis=1))+LD('1e-10')
    orthogonal = np.max(np.sum(np.abs(v.T@v-np.eye(60, dtype=LD)), axis=1))+LD('1e-12')
    error = 4*residual+8*np.max(np.sum(np.abs(h), axis=1))*orthogonal
    if orthogonal >= LD('.01'):
        raise ArithmeticError('Actual full60 eigensystem lacks orthogonality certificate')
    return eigenvalues, Fraction(str(float(error*LD('1.00000001')))), dict(
        dimension=60, normal_dimension=45, gauge_dimension=15,
        full_tensor='Frozen7ef precise_uniform_normal_packet Cartesian tensor formula; full second/fourth even moments, no two-variable normal substitute',
        residual_infinity_upper=encoded(Fraction(str(float(residual*LD('1.00000001'))))),
        orthogonality_infinity_upper=encoded(Fraction(str(float(orthogonal*LD('1.00000001'))))),
        spectral_inclusion_error_upper=encoded(Fraction(str(float(error*LD('1.00000001'))))),
        eigenvalues_diagnostic=eigenvalues.tolist(),
        proof='Real symmetric residual + polar orthogonalization. ONLY at the exact heavy-classical stationary root do the heavy/quartic 15 orbit eigenvalues vanish, making separated upper45 its exact normal spectrum. Witness-to-heavy-root Hessian error is priced. Coupled-root normal compression needs the separate explicit centre/projector transfer price below; no heavy-only zero identity is asserted at the coupled root.')


def alias_certificate():
    """Uniform shifted N-grid versus000 finite Fourier alias, all ten moments.

    In |Im theta_j|<1/4096, G in [0.1,0.3] and all three bands: one-link heavy
    delta-h<0.009, improved charged delta-h<0.022, real R>0.063.
    Hence positive square-root branch remains separated. The bound uses N, not
    any N6 ranking or continuum quadrature. exp(-64)<2^-92 follows ln2<.694.
    """
    IV.dps = 70
    r = iv(Fraction(1, 4096))
    sh = (IV.exp(r)-IV.exp(-r))/2
    ch = (IV.exp(r)+IV.exp(-r))/2-1
    # ACTUAL improved RotationBand: x0=m+(r/a)sum(1-cos)^2,
    # spatial=(v/a)sin, NOT a different improved derivative stencil.
    dc = sh+ch
    heavy_delta = IV.sqrt(iv(Fraction(3, 5))*((iv('7.68000000000001')*3*dc)**2+3*(iv('19.20000000000001')*dc)**2))
    charged_delta = IV.sqrt(iv(Fraction(3, 5))*((iv('7.68000000000001')*3*(4*dc+dc*dc))**2+3*(iv('31.938304836061')*dc)**2))
    gap_lower = Fraction(63, 1000)
    hd, cd = upper(heavy_delta), upper(charged_delta)
    if max(hd, cd) >= gap_lower/2:
        raise ArithmeticError('Actual all-profile complex strip gap failed')
    q = Fraction(1, 2**92)
    # Sum all nonzero integer alias vectors, and difference of arbitrary phases.
    multiplier = 2*((1+2*q/(1-q))**3-1)
    cap = Fraction(10**9)  # xnorm<=150, R>=.0315: includes all fourth moments.
    deviation = multiplier*cap
    return deviation, dict(strip_width=encoded(Fraction(1, 4096)),
        heavy_complex_delta_h_upper=encoded(hd), charged_complex_delta_h_upper=encoded(cd),
        real_band_radius_lower=encoded(gap_lower), complex_band_radius_lower=encoded(gap_lower/2),
        Fourier_tail_q_upper=encoded(q), alias_multiplier_upper=encoded(multiplier),
        component_function_cap=encoded(cap), all64_moment_component_deviation_upper=encoded(deviation),
        finite_N=N, proof='Exact shifted trapezoidal Fourier alias: each k=N*l contributes exp(-N|l|_1/4096); difference of phases costs2; (1+2q/(1-q))^3-1. Quarter twists keep all offdiagonal moments; no zeroing.')


def calculate() -> dict[str, Any]:
    started = time.monotonic()
    if np.finfo(LD).eps > LD(2)**-63:
        raise RuntimeError('Certified double-extended arithmetic is required')
    cfg = asdict(PhiConfig())
    cfg.pop('Lambda')
    descriptor = dict(finite_size=N, Lambda_exponent=EXPONENT, gaussian_order=ORDER,
        microscopic_config=cfg, clock=4096, cutoff_bits=2048,
        G_computational_witness=[encoded(x) for x in G_WITNESS],
        actual_binary64_scalar_paths=[dict(role=role, onsite=encoded(float(onsite)),
            paths=[dict(displacement=[int(z) for z in displacement],
                coefficients=[dict(real=encoded(float(z.real)), imag=encoded(float(z.imag))) for z in scalars])
                for displacement, scalars in paths])
            for _, band, role in models(PhiConfig())
            for onsite, paths in [scalar_paths(band)]],
        exact_measure='N^-3 sum over integers0..N-1 in each coordinate; even reflection endpoints retained',
        quantum_inputs='hbar1/CAR/CCR and selected microscopic parameters declared inputs',
        original_colour_kinetic='.3 chi E4^2, cometric .6 chi',
        primitive_fields='Real noncompact L2(R^(2V+1)), real-gradient Gauss/Coulomb; UEphys1, UElink1/a; no bare EM B^2',
        heavy_occupation='neutral upper-h occupied NH=2V with Hamiltonian -Lambda RotationBand',
        charged_species='TOTAL64 INCLUDING active1 m.2 +aux63 m2; improvedTrue',
        reference='Ref4 actual onsite singlet, totalnumbermod4=0, neutral U1 sector')
    descriptor_bytes = json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()
    accepted, cauchy_error = cells()
    second, fourth, energy, points = evaluate(accepted)
    # |x|<=60 and |x|>=.199. Trig polynomial absolute error<=2^-48;
    # exact binary64 scalar-path coefficients are continued without refitting.
    # Thus ||delta x||/||x||<=2^-37; moment relative error<=32*2^-37.
    # second<=200, fourth<=2000. Positive tensor sums are computed cellwise:
    # gamma_(1000*(ORDER^3+cells)) covers point operations, within-cell reduction
    # and across-cell accumulation without incorrectly multiplying independent
    # pointwise errors by the total number of evaluated points.
    operation_count = 1000*(ORDER**3+len(accepted))
    unit_roundoff = Fraction(1, 2**63)
    gamma = operation_count*unit_roundoff/(1-operation_count*unit_roundoff)
    arithmetic_error = 2000*(Fraction(32, 2**37)+gamma)
    moment_error = cauchy_error+arithmetic_error
    g0, gs = G_WITNESS
    f0 = abs(128*g0-Fraction(str(second[0])))
    fs = abs(128*gs-Fraction(str(sum(second[1:])/3)))
    classical_gradient = 2*(f0+3*fs+4*moment_error)
    # Actual omitted-from-evaluation terms are retained EXACTLY in the enclosure.
    charged_force = Fraction(64*150*150*100, 2**EXPONENT)
    fast_force = Fraction(10**6, 2**(5*EXPONENT//8))
    fast_hessian = Fraction(10**8, 2**(5*EXPONENT//8))
    convexity = Fraction(128)-fast_hessian
    root_radius = (classical_gradient+charged_force+fast_force)/convexity
    if root_radius >= Fraction(1, 1000):
        raise ArithmeticError('Finite root enclosure is not an interior Gram neighbourhood')
    alias_error, aliases = alias_certificate()
    sector_radius = 8*alias_error/convexity
    allroot_radius = root_radius+sector_radius
    eigs, eigen_error, eigen_record = normal_stiffness(second, fourth)
    # Row-sum tensor price: at most16 terms per entry, g<=.3 and60 entries.
    tensor_error = Fraction(60)*(1+16*Fraction(3, 10))*moment_error
    # Exact000 remains diagonal/cubic. For rho<.001, second relative changes
    # <=10rho, fourth<=20rho, and sqrtGi sqrtGj absolute change<=2rho.
    # Thus a row-sum bound from the ACTUAL enclosed moments controls H_000.
    # Shifted fullG perturbations are separately paid with the global1e9 norm cap.
    second_cap = max(Fraction(str(abs(z))) for z in second)+moment_error
    fourth_cap = max(Fraction(str(abs(z))) for z in fourth.ravel())+moment_error
    root_hessian_lipschitz = 60*(10*second_cap+16*(2+Fraction(31, 100)*20)*fourth_cap)+10000
    root_shape_error = root_hessian_lipschitz*root_radius+10**9*sector_radius
    # The exact heavy-classical root has genuine15 gauge zeros. Its normal
    # compression transfers to the COUPLED root with both H and projector paid.
    # Sigma_min(O)>=sqrt(.05)>.22; ||deltaO||<=4||deltaC||, ||deltaC||<=2rho
    # give ||deltaP||<=120rho. Global H norm<1e6, H derivative<2e7;
    # 2||H||*120+2e7 <1e9 bounds the compressed-normal transfer.
    coupled_centre_shift = (charged_force+fast_force)/convexity
    coupled_normal_transfer = Fraction(10**9)*coupled_centre_shift
    phase_error = Fraction(10**6)*alias_error
    charged_stiffness = Fraction(10**10, 2**EXPONENT)
    fast_stiffness = Fraction(10**8, 2**(5*EXPONENT//8))
    total_stiffness_error = eigen_error+tensor_error+root_shape_error+phase_error+coupled_normal_transfer+charged_stiffness+fast_stiffness
    slow_lower = Fraction(float(eigs[15]))-total_stiffness_error
    slow_upper = Fraction(float(eigs[-1]))+total_stiffness_error
    if slow_lower <= 0:
        raise ArithmeticError('Actual full45 allsector normal stiffness did not remain positive')
    root_definition = ('G_t is the unique interior root in the certified Frobenius ball of '
        '128 G-(1/N^3)sum_{p=2pi(n+t/4)/N,profiles} w x(p)x(p)^T/sqrt(x(p)^T G x(p))'
        '+2^-5120 dFfast/dG=0; wheavy=1,wactive=2^-8192,waux=63*2^-8192. '
        'Ffast=(1/N^3)sum_q [45sqrt(lap/5)/2 +sum_{nonemptyS subset{0,1,2,3}} '
        '(sqrt(.12rS+.21lap)+sqrt(.12rS+.20lap)/2)], rS=.5sum_{i in S}eig_i(G); '
        'lap=4sum_j sin(pi q_j/N)^2. Full TEN Frobenius Sym4 components, not two-variable projection. '
        'ClassicalF globally128-convex; fast Hess norm<=1e8*2^-5120 in the enclosing ball, '
        'so the coupled root is locally unique.000 is cubic by exact discrete parity/permutation.')
    centre = dict(g0_lower=encoded(g0-root_radius), g0_upper=encoded(g0+root_radius),
        gs_lower=encoded(gs-root_radius), gs_upper=encoded(gs+root_radius),
        all64_Gram_radius_upper=encoded(sector_radius),
        minimum_Gram_eigenvalue_lower=encoded(gs-allroot_radius),
        trace_Gram_upper=encoded(g0+3*gs+2*allroot_radius),
        numeric_witness_only=[encoded(g0), encoded(gs)],
        reference000_root_distance_upper=encoded(root_radius),
        bound_witness='All endpoints exact rational n/d. g0/gs refer to exact000 coupled root; all64radius means ||G_t-G_000_exact||F, never seed-distance. Weyl min and Frobenius trace price cover all64.')
    f_witness = 64*(g0*g0+3*gs*gs)-2*Fraction(str(energy))
    # Energy derivative norm<=1e3 in ball; displacement and cubature errors paid.
    energy_error = 1000*allroot_radius+2*moment_error+charged_force+fast_force+10*alias_error
    rows = [dict(twist=list(t), exact_root_definition=root_definition,
        energy_interval=dict(lower=encoded(f_witness-energy_error), upper=encoded(f_witness+energy_error),
            normalization='E_D,t/(Lambda V); original uniform sea+quartic+canonical fastZPE; Ref common energy convention'),
        Gram_Frobenius_distance_to_exact000_upper=encoded(sector_radius),
        selection_status='unresolved') for t in product(range(4), repeat=3)]
    source = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes_utf8': data.decode()}
              for name, data in SOURCE_BYTES.items()}
    return dict(branch='NEW actual exact-finite N262144 original-law Phi/colour localwell mean enclosure; NOT bulk substitution or ground/photon achievement',
        input_descriptor=descriptor, input_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        source_at_import=source, own_source_at_import=source[Path(__file__).name],
        runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
            mpmath=mp.__version__, platform=platform.platform(), longdouble_epsilon=str(np.finfo(LD).eps)),
        elapsed_seconds=time.monotonic()-started, finite_size=N, Lambda_exponent=EXPONENT,
        microscopic_config=cfg,
        actual_finite_grid_quadrature_certificate=dict(kind='Positive Gauss quadrature of exact uniform DISCRETE integer-grid measure',
            finite_volume=N**3, reflection='1/N[f(0)+f(N/2)+2sum_{n=1}^{N/2-1} f(n)] percoordinate; endpoint planes/edges/corners retained',
            adaptive_cell_count=len(accepted), actual_evaluated_tensor_nodes=points,
            rule_order=ORDER, rules=list(RULE_RECORDS.values()),
            cauchy_component_remainder_upper=encoded(cauchy_error), arithmetic_component_error_upper=encoded(arithmetic_error),
            total_component_moment_error_upper=encoded(moment_error),
            cauchy_proof='Each cell has actual one-link h0^2=R0^2 I; ||delta h||<=Rlower/2 on polydisc radius4halfwidth, positive square-root branch and |R|>=Rlower/2. Positive tensor quadrature exact coordinate degree2n-1; error<=6 Mfunc tau^(2n)/(1-tau)^3, tau1/4. Full polynomial Taylor, not uncontrolled NumPy/libm sin/cos.',
            arithmetic_proof='IEEE double-extended basic ops, checked eps<=2^-63; sin/cos degree55/54 remainder pi^56/56!<2^-140, gamma300*exp(pi)<2^-48; exact binary64 scalar-path continuation; pi conversion included; sqrt candidates certified by observed squared residual<=1e-17. With |x|>=.199, relative x error<=2^-37, second<=200/fourth<=2000, relative moment price32*2^-37. Actual cellwise positive reduction gamma_(1000*(ORDER^3+cellcount)) adds the displayed rational arithmetic error. MPInterval Jacobi residual/orth/gap certifies rules and cast errors are included.',
            arithmetic_operation_count=operation_count, arithmetic_gamma_upper=encoded(gamma),
            second_moments_diagnostic=[str(z) for z in second], fourth_even_moments_diagnostic=[[str(z) for z in row] for row in fourth],
            finite_full10_gradient_upper=encoded(classical_gradient+charged_force+fast_force),
            charged_gradient_upper=encoded(charged_force), canonical_fast_gradient_upper=encoded(fast_force),
            canonical_fast_concave_Hessian_upper=encoded(fast_hessian), local_coupled_convexity_lower=encoded(convexity),
            canonical_fast_pressure_derivation=dict(
                finite_sum='EXACT N^-3 sum, lap(q)>=0; same sup applies to every actual finite q, not a continuum replacement',
                orbit_radius_lower=encoded(Fraction(1, 20)), frequency_lower=encoded(Fraction(7, 100)),
                spectral_radius_gradient_norm_upper=encoded(1),
                scalar_frequency_gradient_upper=encoded(Fraction(9, 100)/Fraction(7, 100)),
                scalar_frequency_second_derivative_upper=encoded(Fraction(54, 10000)/Fraction(7, 100)**3),
                coupled_pressure_witness='15 nonempty subsets; rS=.5sum eigG, ||drS||F<=1, rS>=.05; t,l>=.006>.07^2. Gradient <=15*.09/.07<20; Hessian spectral divided-difference bound <=15*.0054/.07^3<240. Retained1e6/1e8 bounds are larger, not underflow deletion. Cartesian fullnormal fast-price1e8 additionally bounds the O(C) polynomial/cometric derivatives with orbit gap.05.',
                slow_zero_point_energy_witness='45 finite uniform harmonic modes; normalized slowZPE<=1e8*2^-6144/V, already enclosed by larger retained fast energy price1e6*2^-5120'),
            all64_N_dependent_alias_certificate=aliases),
        exact_centre_definition=root_definition, centre_enclosure=centre,
        slow_stiffness_lower=encoded(slow_lower), slow_stiffness_upper=encoded(slow_upper),
        full45_stiffness_certificate=dict(**eigen_record,
            tensor_moment_error_upper=encoded(tensor_error), root_and_normal_projector_error_upper=encoded(root_shape_error),
            quarter_phase_moment_error_upper=encoded(phase_error), charged_stiffness_error_upper=encoded(charged_stiffness),
            canonical_fast_stiffness_error_upper=encoded(fast_stiffness), total_error_upper=encoded(total_stiffness_error),
            coupled_centre_shift_from_heavy_root_upper=encoded(coupled_centre_shift),
            coupled_normal_compression_transfer_upper=encoded(coupled_normal_transfer),
            exact_normal_consumer='N0(C_t)^T H_t(C_t) N0(C_t), C_t=sqrt(G_t), N0=null(O(C_t)^T); full45. Sortedfull60 identity used ONLY at exact heavy-classical root, then actual coupled-root/projector shift transferred.',
            projector_transfer_witness='Orbit singular gap>=sqrt(.05)>.22; ||deltaO||<=4||deltaC||, ||deltaC||<=2rho, projector change<=120rho. Global H norm<1e6, H derivative<2e7, so compressed derivative<1e9. Charged+fast force/convexity gives displayed actual transfer distance.',
            scope='ALL64 exact implicit roots including quarter fullG; all45 normals, not only g0/gs. Actual heavy moments plus finiteN alias/root/charged/fast prices.'),
        all64_sector_rows=rows,
        sector_policy=dict(kind='prepared_uniform_64_mixture', weights='exact1/64 each',
            selection='UNRESOLVED: energy intervals overlap; no extrapolation of N6 preferred222',
            physical_definition='rho=(1/64)sum_t |Psi_t(G_t,central-seam loops)><Psi_t(G_t,central-seam loops)|',
            meaning='arbitrary prepared mixture, NOT energy degeneracy, thermal state, global ground or photon vacuum; each E_D,t phase cancels separately in density'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = calculate()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(finite_size=report['finite_size'], elapsed_seconds=report['elapsed_seconds'],
        input_sha256=report['input_sha256'], own_sha256=report['own_source_at_import']['sha256'],
        centre_enclosure=report['centre_enclosure'], slow_stiffness_lower=report['slow_stiffness_lower'],
        slow_stiffness_upper=report['slow_stiffness_upper'], sector_count=len(report['all64_sector_rows']),
        adaptive_cells=report['actual_finite_grid_quadrature_certificate']['adaptive_cell_count']), indent=2))


if __name__ == '__main__':
    main()
