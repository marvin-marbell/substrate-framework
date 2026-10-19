#!/usr/bin/env python3
"""Exact-rational normal/BO budgets for the original noncompact-photon law.

The native N>=25 Gram enclosure is not a global scalar-phase certificate.
Global admission is supplied separately by noncompact_quantitative_scalar_phase.
No harmonic energy replaces any of the 64 exact localized scalar energies.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
PLAN_REVISION = '168949e5f2df57656471db8e62d4ae809ab290b1d09ea28c620f52349324a549'


def rational(value: Any) -> F:
    """Decimal strings are exact; binary floating diagnostics are not certificates."""
    if isinstance(value, float):
        raise TypeError('Use an exact integer/Fraction/decimal string, not a float')
    return F(value)


def sqrt_lower(x: F) -> F:
    if x < 0:
        raise ValueError('Negative radicand')
    scale = 1 << 64
    return F(math.isqrt(x.numerator*x.denominator*scale*scale), x.denominator*scale)


def sqrt_upper(x: F) -> F:
    lo = sqrt_lower(x)
    if lo*lo == x:
        return lo
    return lo+F(1, x.denominator*(1 << 64))


def ceil_log2(x: F | int) -> int:
    """Exact outward logarithm: x<=2**answer, including subunit rationals."""
    x = F(x)
    if x <= 0:
        raise ValueError('Positive logarithm input required')
    k = x.numerator.bit_length()-x.denominator.bit_length()
    if k >= 0:
        return k if x.numerator <= (x.denominator << k) else k+1
    return k if (x.numerator << -k) <= x.denominator else k+1


def _inverse(matrix: list[list[F]]) -> list[list[F]]:
    n = len(matrix)
    a = [row[:]+[F(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            raise ValueError('Native coefficient design is not full rank')
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [entry/scale for entry in a[j]]
        for i in range(n):
            if i != j:
                scale = a[i][j]
                a[i] = [x-scale*y for x, y in zip(a[i], a[j])]
    return [row[n:] for row in a]


# Gaussian-rational Laurent coefficients; no complex binary arithmetic.
Gaussian = tuple[F, F]
Exponent = tuple[int, int, int]
Polynomial = dict[Exponent, Gaussian]


def _multiply(a: Polynomial, b: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for i, (ar, ai) in a.items():
        for j, (br, bi) in b.items():
            key = tuple(i[k]+j[k] for k in range(3))
            exponent: Exponent = (key[0], key[1], key[2])
            cr, ci = result.get(exponent, (F(0), F(0)))
            result[exponent] = (cr+ar*br-ai*bi, ci+ar*bi+ai*br)
    return {key: value for key, value in result.items() if value != (0, 0)}


@lru_cache(maxsize=16)
def coefficient_design(spacing: F = F(1)) -> dict[str, Any]:
    """Exact avg(x_mu*x_nu*x_rho*x_sigma), every real shift and N>=5."""
    a = rational(spacing)
    if not 0 < a <= 1:
        raise ValueError('Original proof uses 0<a<=1')
    m, r, v = F(1, 5), F(3, 25), F(3, 10)
    zero = (0, 0, 0)
    x: list[Polynomial] = [{zero: (m+3*r/a, F(0))}]
    sine: list[Polynomial] = []
    for axis in range(3):
        positive = tuple(int(j == axis) for j in range(3))
        negative = tuple(-int(j == axis) for j in range(3))
        kp: Exponent = (positive[0], positive[1], positive[2])
        km: Exponent = (negative[0], negative[1], negative[2])
        x[0][kp] = x[0][km] = (-r/(2*a), F(0))
        sine.append({kp: (F(0), -v/(2*a)), km: (F(0), v/(2*a))})
    x.extend(sine)
    pairs = list(itertools.combinations_with_replacement(range(4), 2))
    p = [_multiply(x[i], x[j]) for i, j in pairs]
    design = [[_multiply(left, right).get(zero, (F(0), F(0))) for right in p] for left in p]
    if any(imag for row in design for _, imag in row):
        raise AssertionError('Real native design acquired an imaginary part')
    real = [[value[0] for value in row] for row in design]
    # Positive leading principal pivots are an exact Sylvester certificate.
    work = [row[:] for row in real]
    pivots = []
    for j in range(10):
        pivot = work[j][j]
        if pivot <= 0:
            raise AssertionError('Native all-ten design is not positive definite')
        pivots.append(pivot)
        for i in range(j+1, 10):
            for k in range(j+1, 10):
                work[i][k] -= work[i][j]*work[j][k]/pivot
    inverse = _inverse(real)
    eta = 1/sum(inverse[i][i] for i in range(10))
    return dict(pairs=pairs, exact_matrix=real, positive_LDL_pivots=pivots,
                richness_lower=eta, grid_onset=5, all_real_shifts=True,
                proof='Fourier degree <=4 per axis; no nonzero multiple of N>=5')


@dataclass(frozen=True)
class GramBounds:
    lower: F
    upper: F
    provenance: str
    onset: int = 5

    @classmethod
    def native_a1(cls) -> GramBounds:
        # NativeMetric Arb128 all-real-shift enclosure: spatial >=.062770690,
        # temporal >=.418273883, cross Frobenius <=.000319264. Gershgorin/
        # Weyl imply the deliberately wider [1/20,1/2] operator enclosure.
        return cls(F(1, 20), F(1, 2), 'NativeMetricAdmission all-ten Arb128, N>=25', 25)

    def validate(self, size: int) -> None:
        if not 0 < self.lower <= self.upper or size < self.onset or not self.provenance:
            raise ValueError('A positive actual all64 Gram enclosure and its N scope are required')


@dataclass(frozen=True)
class NormalBudget:
    size: int
    spacing: F
    gram: GramBounds
    richness: F
    alpha: F
    stiff: F
    strong_upper: F
    leading_upper: F
    heavy_gap: F
    gauge_gap: F
    radius: F
    chart_majorant: F
    comparison: F
    vacuum: F
    momentum: F
    soft_moment: F
    stiff_moment: F
    frequency: F
    gap_loss: F
    residual_kinetic: F
    residual_light: F
    current_lipschitz: F
    agmon: F
    cutoff_log2: int
    local_log2_t: int
    exterior_radius: F
    exterior_light_constant: F
    exterior_light_relative: F

    @property
    def volume(self) -> int:
        return self.size**3

    @property
    def dimension(self) -> int:
        return 90*self.volume

    def powers(self, t: int) -> dict[str, F]:
        """Lambda=t^80 makes every original exponent an exact integer power."""
        if t < 2 or t < (1 << self.local_log2_t):
            raise ValueError('Use a legal t above the returned local onset')
        # Cutoff is an analytic majorant, not a floating exp evaluation.
        cutoff = F(1 << self.cutoff_log2, t**5) if self.cutoff_log2 < 100000 else None
        if cutoff is None:
            raise OverflowError('Use choose_lambda for a symbolic/dyadic huge-parameter certificate')
        d = self.residual_kinetic/F(t**5)+self.residual_light/F(t**25)+cutoff
        return dict(chi=F(1, t**40), epsilon=F(1, t**16),
                    zero_point_upper=45*self.volume*sqrt_upper(self.leading_upper+self.strong_upper)*t**30,
                    scalar_vacuum_excess_upper=self.vacuum*t**30,
                    normal_momentum_upper=self.momentum*t**35,
                    xi_moment_upper=self.stiff_moment/F(t**35),
                    z_moment_upper=self.soft_moment/F(t**25), residual_upper=d,
                    current_image_difference_upper=self.current_lipschitz*self.soft_moment/F(t**25),
                    complement_gap_lower=self.frequency*t**20/4-self.gap_loss*t**14-self.exterior_light_constant)


def native_normal_budget(size: int, spacing: Any = '1', gram: GramBounds | None = None) -> NormalBudget:
    """Computable original-source constants, conditional only on stated Gram scope.

    Global scalar-phase admission is NOT inferred from this function.
    """
    a = rational(spacing)
    if size < 5 or not 0 < a <= 1:
        raise ValueError('N>=5 and0<a<=1 required')
    if gram is None:
        if a != 1:
            raise ValueError('Supply the actual all64 Gram enclosure at this spacing')
        gram = GramBounds.native_a1()
    gram.validate(size)
    volume, dim = size**3, 105*size**3
    g, G = gram.lower, gram.upper
    kappa, m, r, v = 2/a, F(1, 5), F(3, 25), F(3, 10)
    cx2 = (m+6*r/a)**2+3*(v/a)**2
    cx = sqrt_upper(cx2)
    rmax = sqrt_upper(G*cx2)
    eta = coefficient_design(a)['richness_lower']
    # Canonical complete60 linearized Clifford spectrum: 0(15),4(10),6(20),8(15).
    alpha = 4*g*g/G*min(kappa, eta/(2*rmax**3))
    # Complete mass-horizontal .6 link cometric, not a Euclidean slice.
    stiff = F(1, 5)*min(F(16, size**2), F(3, 10)*g)
    strong = 10+G
    delta = 2*m*sqrt_lower(g)
    gauge = g/2
    sf = (3*r*r+3*v*v)*G/(a*a)
    leading = 6*kappa*G+(60*cx2+30*sf)/delta+sqrt_upper(5400*(r*r+v*v)/(a*a)+450*sf)
    rho = sqrt_upper(4*volume*G)
    # Explicit finite-jet analytic-chart lemma in the accompanying proof.
    h = F(1 << 12)*dim*dim*(1+rho+cx)
    s = min(F(1, 4), sqrt_lower(gauge), sqrt_lower(stiff), delta/8)
    chart = (1024*h/s)**32
    radius = min(F(1, 64), (s/(1024*h))**8, delta/(64*volume*h))
    compare = (1 << 16)*dim**3*chart**4*(1+1/alpha+1/stiff)*(4*volume+3)**2
    zp = 45*volume*sqrt_upper(leading+strong)
    vacuum = 4*zp+4*compare
    momentum = sqrt_upper(8*(vacuum+compare))
    zm = sqrt_upper(8*(vacuum+compare)/alpha)
    xim = sqrt_upper(8*(vacuum+compare)/stiff)
    frequency = min(sqrt_lower(alpha/2), sqrt_lower(stiff/2))
    gap_loss = 8*compare*(vacuum+1)
    # Endpoint light/current/contact are bounded in compact A but NOT globally in Phi.
    lip = 32*(2*volume+1)*cx*(1+rho)*chart
    residual_k = 16*compare*(momentum+1)
    residual_l = lip*(zm+xim)
    agmon = min(F(1, 64), sqrt_lower(min(alpha, stiff))/256)
    polynomial_degree = dim+10
    inverse_power = (80*polynomial_degree+5+27)//28
    # exp(-u)<=j!u^-j<=j^j u^-j: no floating exponential used.
    cutoff_log = (ceil_log2(64*compare*(1+vacuum))
                  +inverse_power*(ceil_log2(inverse_power)+ceil_log2(1/agmon)))
    schur = 2*(leading+2*leading*leading/alpha)/stiff
    local_k = max(1, (ceil_log2(schur)+19)//20,
                  (ceil_log2(4/radius)+15)//16,
                  (ceil_log2(64*compare)+15)//16,
                  (ceil_log2(64*vacuum/min(alpha, stiff))+17)//18,
                  (ceil_log2(64*vacuum/delta)+49)//50,
                  (ceil_log2(64*gap_loss/frequency)+5)//6)
    # Whole raw fixed-NL sector, worst of q=0,+1,-1; bar4/4 reference included.
    A4 = kappa/(8*volume)
    BH, BL = 4*volume*cx, 2*(2*volume+1)*cx
    # integer cube upper is outward, avoids a fractional-power diagnostic.
    cube = 2*BH/A4
    rho0 = max(F(1), F(1 << ((ceil_log2(cube)+2)//3)))
    C0, C1 = BL*rho0, 2*BL/A4
    return NormalBudget(size, a, gram, eta, alpha, stiff, strong, leading,
                        delta, gauge, radius, chart, compare, vacuum, momentum,
                        zm, xim, frequency, gap_loss, residual_k, residual_l,
                        lip, agmon, cutoff_log, local_k, rho0, C0, C1)


def whole_heavy_projector_jets(budget: NormalBudget, max_order: int = 6) -> dict[str, Any]:
    """Rank-one WHOLE fixed2V Slater Riesz/Kato estimates on the regular tube."""
    if not 0 <= max_order <= 8:
        raise ValueError('This finite-jet construction covers orders0..8')
    dim, number = 105*budget.volume, 2*budget.volume
    rho = sqrt_upper(4*budget.volume*budget.gram.upper)
    endpoint = 16*(1+sqrt_upper((F(1, 5)+F(18, 25)/budget.spacing)**2
                              +3*(F(3, 10)/budget.spacing)**2))*(1+rho)
    # Product/exponential endpoint derivatives <=q!2^q endpoint; whole dGamma<=NH.
    scale = 64*number*endpoint/budget.heavy_gap
    return dict(whole_occupation=number, onebody_dimension=4*budget.volume,
                all_normal_coordinates=dim, unscaled_cluster_gap_lower=budget.heavy_gap/2,
                endpoint_jet_upper={q: math.factorial(q)*2**q*endpoint for q in range(max_order+1)},
                whole_Slater_projector_jet_upper={q: 2*math.factorial(q)*scale**q for q in range(max_order+1)},
                Kato_section_jet_upper={q: math.factorial(q)*(4*scale)**(2*q+2) for q in range(max_order+1)},
                arbitrary_nonuniform_fields_in_tube=True, EM_independent=True,
                no_per_momentum_filling_constraint=True)


def orientation_source_jets(length: int, eta: Any, alpha: Any, frame_squared: Any,
                            initial_metric_lower: Any, max_order: int = 6) -> dict[str, Any]:
    """CentreSourceEnergy's whole-Haar analytic Cauchy bounds, exact outward.

    These are Frechet/ordered-link jets, NOT the total physical Noether generator.
    """
    e, al, b2, n = map(rational, (eta, alpha, frame_squared, initial_metric_lower))
    if length < 0 or e <= 0 or al <= 0 or b2 <= 0 or n <= 0 or not 1 <= max_order <= 6:
        raise ValueError('Positive actual orientation parameters and source Gram floor required')
    scale = 1+al*sqrt_upper((length+1)*b2)+e*sqrt_upper(F(length))
    jets = {0: 1/sqrt_lower(n)}
    for q in range(1, max_order+1):
        # q/2 mark; factor4 covers the source owner's quartet q convention.
        jets[q] = 4*2**(q+1)*math.factorial(q)*(16*q*scale)**q/sqrt_lower(n)
    return dict(normalized_quartet_source_jet_upper=jets, whole_product_Haar=True,
                all_Phi=True, total_physical_Noether_bound=False,
                provenance='CentreSourceEnergy whole-Haar Cauchy inequality, q<=6')


def source_transfer_multiplier(source_jets: dict[int, F], graph_norm: Any,
                               detector_norm: Any = 1, max_order: int = 6) -> F:
    """Leibniz majorant for graph-weighted source/current insertions.

    graph_norm is the actual positive weighted R_B aOmega graph norm. A source
    derivative or total-Noether graph norm is never inferred from unweighted ||aOmega||.
    """
    graph, detector = rational(graph_norm), rational(detector_norm)
    if graph <= 0 or detector <= 0 or not 0 <= max_order <= 6:
        raise ValueError('Positive actual graph/detector norms required')
    if set(range(max_order+1))-source_jets.keys():
        raise ValueError('Every source derivative through requested order is needed')
    jets = [rational(source_jets[q]) for q in range(max_order+1)]
    if any(j < 0 for j in jets):
        raise ValueError('A jet majorant cannot be negative')
    return graph*detector*sum(math.comb(max_order, q)*jets[q] for q in range(max_order+1))


def _dyadic_requirement(coefficient: F, bits: int, power: int) -> int:
    return max(1, (ceil_log2(coefficient)+bits+power-1)//power)


def choose_lambda(budget: NormalBudget, target_error: Any, time: Any = 1,
                  norm_budget: Any = 1, requested_gap: Any = 1,
                  source_multiplier: Any = 1, phase: dict[str, Any] | None = None,
                  kinetic_exponent: Any = '-1/2', energy_subtraction: str = 'exact_all64',
                  exterior: str = 'quartic_full_domain') -> dict[str, Any]:
    """Explicit finite Lambda for requested transfer/resolvent/current budget.

    Native local admission is returned even when the global phase packet is absent;
    that result is labelled conditional and is not microscopic admission. A supplied
    phase packet must retain all64 centres and the full reducible/unbounded competitors.
    A huge finite Lambda is an exact power expression, not inf/overflow/frozen cutoff.
    """
    error, duration, norm, gap, multiplier = map(rational,
        (target_error, time, norm_budget, requested_gap, source_multiplier))
    if error <= 0 or duration < 0 or norm <= 0 or gap <= 0 or multiplier <= 0:
        raise ValueError('Positive requested error/norm/gap and nonnegative time required')
    if rational(kinetic_exponent) != F(-1, 2):
        raise ValueError('The entire-photon residual uses chi=Lambda^(-1/2), not a fast kinetic law')
    if energy_subtraction != 'exact_all64':
        raise ValueError('Every unknown exact Etheta offset must remain; no harmonic-vac substitution')
    if exterior != 'quartic_full_domain':
        raise ValueError('Unbounded Phi is part of the domain; no global constant light norm')
    # Three residual summands, current image error, and time evolution share one budget.
    weight = (1+duration)*norm*multiplier
    bits = max(1, ceil_log2(16*weight/error))
    k = max(budget.local_log2_t,
            _dyadic_requirement(budget.residual_kinetic, bits, 5),
            _dyadic_requirement(budget.residual_light, bits, 25),
            (budget.cutoff_log2+bits+4)//5,
            _dyadic_requirement(budget.current_lipschitz*budget.soft_moment, bits, 25),
            (ceil_log2(64*gap/budget.frequency)+19)//20,
            (ceil_log2(64*budget.exterior_light_constant/budget.frequency)+19)//20,
            (ceil_log2(2*budget.exterior_light_relative)+79)//80)
    phase_expression = None
    global_exponent = None
    if phase is not None:
        if (budget.spacing != 1 or phase.get('N') != budget.size
                or budget.size < int(phase['N_phase'])
                or phase.get('all_64_native_full10_metrics') is not True
                or phase.get('centre_offsets_retained') is not True):
            raise ValueError('The actual a1 all64 global scalar-phase N scope does not match')
        full = phase['full_domain']
        height = int(full['height_inner_exponent'])
        restoration = full['global_quadratic_restoration']
        if (restoration['exponent_sign'] != -1
                or restoration['exponent_coefficient'] != 8
                or restoration['exponent_tower']['inner_exponent'] != height+1
                or 'entire field/link domain' not in full['statement']):
            raise ValueError('The actual scalar-phase full-domain height law does not match')
        # c=2^-D, D=8*2^(2^(height+1)). Chart coordinates can shrink ambient
        # core distances by J, so pay c/J^2 as well as vacuum, light and IMS.
        # Lambda^(.6-.375)=Lambda^(9/40): log2Lambda >=(40/9)(D+log2C).
        # 80*(D+k+ell) is larger, also above the phase onset 8*2^(2^height).
        ell = max(1, ceil_log2(256*(budget.vacuum+budget.comparison
                      +budget.exterior_light_constant+gap+1)*budget.chart_majorant**2))
        global_exponent = dict(local_log2_t=k, exterior_log2_t=ell,
                               coercivity_reciprocal_exponent_coefficient=8,
                               coercivity_reciprocal_inner_exponent=height+1)
        phase_expression = f'2^(80*({k}+{ell}+8*2^(2^({height+1}))))'
    expression = f'2^({80*k})' if phase_expression is None else phase_expression
    return dict(Lambda_expression=expression, local_log2_Lambda=80*k,
                global_log2_Lambda_expression=global_exponent,
                Lambda_is_finite=True, exact_integer_power=True,
                target_error=str(error), time=str(duration), norm_budget=str(norm),
                source_multiplier=_json(multiplier), requested_complement_gap=str(gap),
                transfer_error_upper=str(error/2), current_image_error_upper=str(error/16),
                residual_upper=str(error/(4*weight)), complement_gap_lower=str(gap),
                global_microscopic_admission=phase is not None,
                scalar_phase_scope='conditional_on_global_packet' if phase is None else phase,
                native_Gram_scope=budget.gram.provenance, N=budget.size,
                physical_normal_dimension=budget.dimension, soft_count=45,
                stiff_count=budget.dimension-45, all15V_Gauss=True,
                entire_effective_photon_Hilbert=True, raw_fixed_NL_CAR=True,
                opposite_bar4_or_4_reference=True, all64_exact_offsets_retained=True,
                Etheta_values_computed=False, frozen_harmonic_vacuum_used=False,
                unbounded_Phi_exterior_retained=True,
                proof='Exact logarithmic inequalities in accompanying analytic derivation')


def resolvent_budget(residual: Any, complement_gap: Any, real_z: Any,
                     imaginary_z: Any) -> F:
    d, g, re, im = map(rational, (residual, complement_gap, real_z, imaginary_z))
    if d < 0 or im == 0 or g <= abs(re):
        raise ValueError('Nonreal z and complement gap>|Re z| required')
    return 3*d/(im*im)+1/(g-abs(re))


def microscopic_charge_lower(effective_mu: Any, neutral_residual: Any,
                             charged_residual: Any, charged_complement_gap: Any,
                             neutral_effective_upper: Any) -> dict[str, Any]:
    """Untruncated charged-minus-neutral infimum transfer by a Schur bound.

    Inputs are proved sector-wise W residual/gap bounds and an actual neutral
    effective trial upper. Scalar offsets must be common and retained all64.
    This does not require a charged pole, a photon cutoff or a light projection.
    """
    mu, d0, dq, g, upper = map(rational, (effective_mu, neutral_residual,
        charged_residual, charged_complement_gap, neutral_effective_upper))
    if mu <= 0 or min(d0, dq) < 0:
        raise ValueError('Positive effective threshold and nonnegative sector residuals required')
    threshold_upper = upper+mu
    if g <= threshold_upper:
        raise ValueError('Charged complement must lie above the neutral trial plus threshold')
    denominator = g-threshold_upper
    complement_loss = dq*dq/denominator
    lower = mu-d0-dq-complement_loss
    return dict(effective_threshold_lower=mu, neutral_upper_loss=d0,
                charged_image_lower_loss=dq, charged_complement_schur_loss=complement_loss,
                schur_denominator_lower=denominator,
                charged_minus_neutral_infimum_lower=lower, strictly_positive=lower > 0,
                both_compensated_charge_signs=True, all64_common_exact_offsets_required=True,
                untruncated_noncompact_harmonic_fibres=True, raw_full_CAR=True,
                opposite_reference_retained=True, isolated_charged_pole_assumed=False)


def native_microscopic_charge_budget(budget: NormalBudget, electric: Any,
                                    magnetic: Any, target_error: Any = None,
                                    phase: dict[str, Any] | None = None) -> dict[str, Any]:
    """Positive finite-Lambda original-law compensated mass capability.

    NativeMetric's actual outward Schur endpoint is .4182722595473932.
    The nine-decimal floor below rounds DOWN; .418272260 would round up.
    """
    if budget.spacing != 1 or budget.gram != GramBounds.native_a1():
        raise ValueError('The actual NativeMetric a1 all64 N>=25 Schur scope is required')
    schur = F('0.418272259')
    mu = sqrt_lower(schur)/5
    error = mu/16 if target_error is None else min(rational(target_error), mu/16)
    if error <= 0:
        raise ValueError('Positive target error required')
    upper = neutral_trial_upper(budget, electric, magnetic)
    gap = upper+mu+1
    choice = choose_lambda(budget, error, time=0, requested_gap=gap, phase=phase)
    # choice certifies d<=error/4 in every sector and g>=gap. No source vector
    # or sampled charged Ritz value replaces these entire-domain inequalities.
    d = error/4
    transferred = microscopic_charge_lower(mu, d, d, gap, upper)
    return dict(N=budget.size, spacing=budget.spacing, native_Schur_floor=schur,
                native_effective_mu_lower=mu, neutral_effective_trial_upper=upper,
                finite_Lambda=choice, transfer=transferred,
                microscopic_mass_admitted=(choice['global_microscopic_admission']
                                            and transferred['strictly_positive']),
                generic_BO_proof_review_is_separate=True,
                effective_threshold_source='noncompact_charge_threshold.md sections1-4',
                actual_unknown_Etheta_offsets_retained=True)


def neutral_trial_upper(budget: NormalBudget, electric: Any, magnetic: Any) -> F:
    """Explicit finite neutral photon/light trial, common exact scalar offset.

    Matched noncompact Coulomb law is UE*rho G rho, not its obsolete half.
    Harmonic coordinates use an actual normalized Gaussian, not a zero-mode vacuum.
    """
    ue, ub = rational(electric), rational(magnetic)
    if ue <= 0 or ub <= 0:
        raise ValueError('Positive actual noncompact photon coefficients required')
    volume = budget.volume
    light = 2*(2*volume+1)*sqrt_upper((F(1, 5)+F(18, 25)/budget.spacing)**2
                                   +3*(F(3, 10)/budget.spacing)**2)
    rho = sqrt_upper(4*volume*budget.gram.upper)+1
    return (light*rho+3*volume*sqrt_upper(12*ue*ub)/2
            +3*ue/4+ue*volume*budget.size**2/4)


def state_normal_moments(budget: NormalBudget, electric: Any, magnetic: Any,
                         relative_energy: Any, norm_squared: Any = 1) -> dict[str, Any]:
    """Actual Omega/resolvent graph moment inequality with the negative baseline kept."""
    energy, n2 = rational(relative_energy), rational(norm_squared)
    if energy < 0 or n2 <= 0:
        raise ValueError('Nonnegative relative graph energy and positive squared norm required')
    trial = neutral_trial_upper(budget, electric, magnetic)
    finite = energy+(trial+budget.exterior_light_constant+1)*n2
    return dict(field_momentum_squared_upper_expression=
                '2*Lambda^(1/2)*(finite+vacuum_coefficient*Lambda^(3/8))',
                link_momentum_squared_upper_expression=
                '(10/3)*Lambda^(1/2)*(finite+vacuum_coefficient*Lambda^(3/8))',
                finite=finite, vacuum_coefficient=budget.vacuum*n2,
                original_chi=True, exact_all64_scalar_offsets=True,
                actual_ground_or_resolvent_graph=True, no_U0_equals_zero_assumption=True)


def physical_noether_filter_budget(budget: NormalBudget, log2_lambda: int,
                                  field_velocity_sup: Any, link_velocity_sup: Any,
                                  source_field_graph: Any, source_link_graph: Any,
                                  photon_orbital_reference_graph: Any,
                                  actual_projected_matrix_norm: Any,
                                  filter_eighth_derivative_L1: Any,
                                  temporal_halo: Any = 0, energy_tail: Any = 0,
                                  source_norm: Any = 1) -> dict[str, Any]:
    """Weighted actual-Noether graph/filter bound, not an intertwining equivalence.

    Velocities and projected matrix must come from the actual full-Gauss/mechanical
    connection. Photon/orbital/reference graph includes harmonic coordinate moments;
    a photon energy cutoff alone does not control those coordinates.
    The energy window is <=half the compressed-complement gap. Applying this
    to full microscopic spectral filtering also needs its actual graph mixing
    estimate; the bounded Hamiltonian residual does not supply that estimate.
    """
    yp, yl, bp, bl, ph, projected, f8, halo, etail, norm = map(rational,
        (field_velocity_sup, link_velocity_sup, source_field_graph, source_link_graph,
         photon_orbital_reference_graph, actual_projected_matrix_norm,
         filter_eighth_derivative_L1, temporal_halo, energy_tail, source_norm))
    if log2_lambda < 80*budget.local_log2_t or min(yp, yl, bp, bl, ph, projected, halo, etail) < 0 or f8 <= 0 or norm <= 0:
        raise ValueError('Actual finite-source graph bounds and a legal Lambda are required')
    leading = (yp+yl)*budget.momentum*norm
    fixed = yp*bp+yl*bl+ph+projected*norm
    graph_log = 1+max(ceil_log2(max(leading, F(1)))+(7*log2_lambda+15)//16,
                      ceil_log2(max(fixed, F(1))))
    # For E<=g/2 and g>=frequency Lambda^(1/4)/8, g-E>=frequency Lambda^(1/4)/16.
    floor_gap_log = -ceil_log2(16/budget.frequency)+log2_lambda//4
    tail_log = ceil_log2(f8)+graph_log+8*((log2_lambda+31)//32-floor_gap_log)
    return dict(weighted_Jphys_W_graph_upper_expression=f'2^({graph_log})',
                offnormal_time_smear_upper_expression=f'2^({tail_log})',
                offnormal_log2_upper=tail_log, tau='Lambda^(-1/32)', smear_order=8,
                energy_window='E <= admitted_complement_gap/2',
                scoped_halo_energy_tail_upper=halo+etail,
                projected_matrix_norm=projected, leading_normal_graph_coefficient=leading,
                fixed_source_photon_graph_coefficient=fixed,
                leading_decay_power='Lambda^(-21/16)',
                actual_projection_is_caller_owned=True,
                spectral_filter_operator='compressed K_C versus effective energy window',
                microscopic_spectral_graph_mixing_established=False,
                bounded_Hamiltonian_residual_implies_Noether_intertwining=False,
                bare_local_Phi_two_pi_identification=False)


def source_bindings() -> dict[str, str]:
    names = ('noncompact_quantitative_normal_budget.py', 'noncompact_quantitative_normal_budget.md',
             'original_slow_kinetic_bo.py', 'original_slow_kinetic_bo.md',
             'original_normal_gap_bound.py', 'original_normal_gap_bound.md',
             'original_charged_principal.py', 'bulk_quantum_selection.py',
             'sea_selection.py', 'noncompact_centre_source_energy.py',
             'noncompact_quantitative_scalar_phase.py', 'noncompact_quantitative_scalar_phase.md',
             'noncompact_charge_threshold.py', 'noncompact_charge_threshold.md',
             'noncompact_native_metric_admission.py', 'noncompact_native_metric_admission.md')
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def native_consumer() -> dict[str, Any]:
    """Run actual full500 heavy/full105 horizontal/current path, not phase proof."""
    import numpy as np
    from scipy.linalg import null_space
    from bulk_quantum_selection import BulkQuantumLaw
    from original_charged_principal import OriginalSpatial, charge_form, original_frame
    from original_normal_gap_bound import reduced_matrices, native_current_controls
    from original_slow_kinetic_bo import whole_heavy, normal_directions, jet_controls
    from rotation_band import RotationBand
    from sea_selection import matrix_unit_basis
    from unrestricted_field_minimum import BARE, coordinates

    band = RotationBand(**{**BARE, 'a': 1.})
    size = 5
    angles = np.pi/2*np.array([1, 2, 3])
    _, frame = original_frame(band, size, angles, True)
    basis = matrix_unit_basis()
    model = OriginalSpatial(band, size, basis, coordinates(frame, basis).reshape(-1), angles)
    law = BulkQuantumLaw(size, band=band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(angles)
    q = 2*np.pi*np.array([1, 0, 0])/size
    gauge = model.gauge_columns(q)
    form = charge_form(gauge, 1., .3)
    Z = null_space((np.sqrt(form['mass'])[:, None]*gauge).conj().T)
    if form['rank'] != 15 or Z.shape != (105, 90):
        raise AssertionError('Native physical full-normal/Gauss consumer was narrowed')
    B, _ = reduced_matrices(model, q, model.alignment(q))
    conditional = GramBounds(F(1, 100), F(1), 'N5 diagnostic enclosure, NOT native/global admission')
    bound = native_normal_budget(size, gram=conditional)
    if np.linalg.eigvalsh(B)[0] < float(bound.stiff):
        raise AssertionError('Derived all-q alignment lower failed its actual consumer')
    direction = normal_directions(model, law, np.random.default_rng(222501))
    jets, _ = jet_controls(model, law, fields, links, direction)
    whole = whole_heavy(law.onebody(fields, links), 2*law.volume)
    current = native_current_controls(band, size, frame, angles)
    # Adverse exterior: actual row/quartic relative bound grows, a false global
    # constant light norm is contradicted by the literal source Phi scaling.
    h0 = law.onebody(fields, links)
    enlarged = law.onebody(32*fields, links)
    scaled_error = float(np.linalg.norm(enlarged-32*h0))
    if scaled_error > 1e-8 or np.linalg.norm(enlarged) <= 16*np.linalg.norm(h0):
        raise AssertionError('Unbounded native Phi adverse case did not exercise growth')
    # Adverse kinetic on the actual105-coordinate colour alignment form:
    # after factoring the intended chi, a fast field law multiplies its
    # field cometric by Lambda while retaining the declared colour cometric.
    fast_form = charge_form(gauge, 1e8, .3)
    fast_Z = null_space((np.sqrt(fast_form['mass'])[:, None]*gauge).conj().T)
    fast_root = np.sqrt(fast_form['a'])
    fast_transform = fast_root[:, None]*fast_Z
    fast_B = fast_transform.conj().T@model.alignment(q)@fast_transform
    kinetic_trace_ratio = float(np.trace(fast_B).real/np.trace(B).real)
    if fast_form['rank'] != 15 or fast_Z.shape != (105, 90) or kinetic_trace_ratio <= 2:
        raise AssertionError('Wrong kinetic law did not change the actual full-Gauss principal')
    # Radial Phi scaling has exact quartic+(linear whole-heavy) potential.
    # A quadratic oscillator Gaussian is not an eigenfunction of this
    # literal native potential: the nonzero cubic/quartic remainder survives.
    native_G = np.einsum('mij,nji->mn', frame, frame).real/4
    quartic = law.volume*float(np.sum(native_G*native_G))
    radial_step = .04
    literal_V = quartic*(1+radial_step)**4+whole['energy']*(1+radial_step)
    quadratic_V = (quartic+whole['energy']+(4*quartic+whole['energy'])*radial_step
                   +6*quartic*radial_step**2)
    harmonic_remainder = float(literal_V-quadratic_V)
    if harmonic_remainder <= 1e-6:
        raise AssertionError('Frozen harmonic-vacuum adverse case missed native anharmonicity')
    adverse = {}
    adverse_cases: list[tuple[str, dict[str, Any]]] = [
        ('wrong_kinetic', {'kinetic_exponent': '1/2'}),
        ('harmonic_vacuum', {'energy_subtraction': 'harmonic'}),
        ('bounded_exterior', {'exterior': 'bounded_Phi'})]
    for name, kwargs in adverse_cases:
        try:
            choose_lambda(bound, '1/100', **kwargs)
        except ValueError:
            adverse[name] = True
        else:
            raise AssertionError(f'{name} changed the actual theorem silently')
    return dict(N=size, centre=[1, 2, 3], all15_Gauss=form['rank'], horizontal_dimension=Z.shape[1],
                physical_normal_dimension=90*size**3, full_heavy_matrix_dimension=len(whole['values']),
                whole_heavy_occupation=2*law.volume, whole_heavy_gap=whole['gap'],
                heavy_curvature_consumer=jets, current_consumer=current,
                actual_unbounded_Phi_scaling_error=scaled_error, adverse=adverse,
                wrong_kinetic_actual_principal_trace_ratio=kinetic_trace_ratio,
                harmonic_vacuum_native_potential_remainder=harmonic_remainder,
                harmonic_remainder_is_not_an_exact_Etheta_calculation=True,
                N5_global_or_outward_stationarity_certified=False,
                diagnostic_scope='Native full source path only; analytic rational theorem is separate')


def _json(value: Any) -> Any:
    if isinstance(value, F):
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 12000:
            return dict(exact_numerator_hex=hex(value.numerator), exact_denominator_hex=hex(value.denominator))
        return str(value)
    if isinstance(value, GramBounds):
        return _json(value.__dict__)
    if isinstance(value, NormalBudget):
        return _json(value.__dict__)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if hasattr(value, 'tolist'):
        return _json(value.tolist())
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=25)
    parser.add_argument('--error', default='1/1000')
    parser.add_argument('--time', default='6')
    parser.add_argument('--native-consumer', action='store_true')
    parser.add_argument('--global-phase', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    budget = native_normal_budget(args.size)
    phase = None
    if args.global_phase:
        from noncompact_quantitative_scalar_phase import phase_bounds
        phase = phase_bounds(args.size)
    result: dict[str, Any] = dict(governing_plan=PLAN, assignment_revision=PLAN_REVISION,
                  analytic_normal_budget=budget, exact_design=coefficient_design(),
                  whole_heavy_Kato=whole_heavy_projector_jets(budget),
                  finite_Lambda=choose_lambda(budget, args.error, args.time, phase=phase),
                  native_consumer=native_consumer() if args.native_consumer else None,
                  source_sha256=source_bindings(), campaign_complete=False)
    payload = json.dumps(_json(result), indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload+'\n')
    print(json.dumps(dict(N=args.size, physical_normals=budget.dimension,
                         local_log2_Lambda=result['finite_Lambda']['local_log2_Lambda'],
                         global_microscopic_admission=phase is not None,
                         native_consumer_exercised=args.native_consumer)))


if __name__ == '__main__':
    main()
