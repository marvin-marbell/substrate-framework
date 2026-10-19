#!/usr/bin/env python3
"""Exact charged/neutral flight after paired photon amputation, with rational bounds."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.sparse.linalg import expm_multiply

from noncompact_native_current import Coupled, accept, adverse, exterior
from noncompact_photon_amputation_graph import creators
from original_charged_principal import original_frame
from rotation_band import RotationBand
from sea_selection import SELECTED
import noncompact_inclusive_scattering_join as joined
import noncompact_quantitative_normal_budget as normal
import noncompact_quantitative_scalar_phase as scalar_phase

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
Q = Fraction


def rational(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Q.from_float(value)
    return Q(str(value))


def sqrt_upper(value: Fraction, bits: int = 256) -> Fraction:
    """A directed rational square root, including subnormal/large-volume regimes."""
    if value < 0:
        raise ValueError('nonnegative radicand required')
    if value == 0:
        return Q(0)
    shift = bits + max(0, (value.denominator.bit_length()-value.numerator.bit_length()+1)//2)
    scale = 1 << shift
    numerator = value.numerator*scale*scale
    ceiling = (numerator+value.denominator-1)//value.denominator
    root = math.isqrt(ceiling)
    if root*root < ceiling:
        root += 1
    result = Q(root, scale)
    if result*result < value:
        raise AssertionError('directed square root lost its upper direction')
    return result


def round_upper(value: Fraction, bits: int = 256) -> Fraction:
    if value < 0:
        raise ValueError('positive directed rounding required')
    shift = max(0, bits+value.denominator.bit_length()-value.numerator.bit_length())
    scale = 1 << shift
    numerator = value.numerator*scale
    return Q((numerator+value.denominator-1)//value.denominator, scale)


def exp_upper(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError('nonnegative exponential argument required')
    term = total = Q(1)
    count = max(80, 2*math.ceil(value)+1)
    for k in range(1, count+1):
        term *= value/k
        total += term
    ratio = value/(count+1)
    return round_upper(total+term*ratio/(1-ratio))


def decimal_bound(value: Fraction, upper: bool=True) -> str:
    with localcontext() as context:
        context.prec = 70
        context.rounding = ROUND_CEILING if upper else ROUND_FLOOR
        return str(Decimal(value.numerator)/Decimal(value.denominator))


def display_rational(value: Fraction) -> dict[str, str]:
    return dict(decimal_upper=decimal_bound(value),decimal_lower=decimal_bound(value,False),
        numerator=str(value.numerator),denominator=str(value.denominator))


def encode(value: Any) -> Any:
    if isinstance(value, Fraction):
        return display_rational(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, Jet):
        return dict(norm=value.values[0], derivatives=list(value.values[1:]), light_commutator=value.light)
    raise TypeError(type(value).__name__)


def touchard(order: int, value: Fraction) -> Fraction:
    row = [Q(1)]
    for n in range(1, order+1):
        row = [Q(0)]+[row[k-1]+k*(row[k] if k < len(row) else 0) for k in range(1, n+1)]
    return sum((coefficient*value**k for k, coefficient in enumerate(row)), Q(0))


@dataclass(frozen=True)
class Jet:
    """Whole-CAR norm jets per harmonic axis and an A-weighted commutator bound."""
    values: tuple[Fraction, ...]
    light: Fraction

    def product(self, other: Jet) -> Jet:
        values = tuple(sum((math.comb(q, k)*self.values[k]*other.values[q-k]
            for k in range(q+1)), Q(0)) for q in range(len(self.values)))
        return Jet(values, self.light*other.values[0]+self.values[0]*other.light)


def free_conjugated(values: tuple[Fraction, ...], time: Fraction, hjet: Fraction) -> tuple[Fraction, ...]:
    """Only the one-body coefficients are conjugated; no extensive H0 derivative."""
    flow = [touchard(q, time*hjet) for q in range(len(values))]
    return tuple(sum((math.factorial(q)//(math.factorial(a)*math.factorial(b)*math.factorial(q-a-b))
        *flow[a]*values[b]*flow[q-a-b] for a in range(q+1) for b in range(q-a+1)), Q(0))
        for q in range(len(values)))


def interval_end(text: str, upper: bool) -> Fraction:
    if text.startswith('['):
        middle, radius = text[1:-1].split(' +/- ')
        return Q(middle)+(Q(radius) if upper else -Q(radius))
    return Q(text)


def compact_descriptor(payload: dict[str, Any]) -> dict[str, Any]:
    descriptor = payload['result']['descriptor'] if 'result' in payload else payload
    n = int(descriptor['N'])
    if n < 5 or int(descriptor['V']) != n**3:
        raise ValueError('actual native N>=5 and exact V=N^3 required')
    relative = [[int(displacement[j])+int(descriptor['origin'][j])-int(descriptor['reference'][j])
                 for j in range(3)] for displacement in descriptor['compact_displacement']]
    radius = max(abs(component) for row in relative for component in row)
    word_length = max(sum(abs(component) for component in row) for row in relative)
    if radius != int(descriptor['source_max_relative_coordinate']):
        raise ValueError('actual source-relative coordinates disagree with source jet descriptor')
    fields, frequencies = [], []
    for beam in ('incoming', 'outgoing_annihilation'):
        rows = descriptor[beam]
        if sum((Q(row['weight_squared']) for row in rows), Q(0)) != 1:
            raise ValueError('physical photon packet has nonunit CCR norm')
        if any(row['polarization'] != [0, 0, 1] or row['mode_id'][2] != 0 for row in rows):
            raise ValueError('this native flight contract uses the supplied z-polarized transverse cones')
        lower = [interval_end(row['omega_interval'], False) for row in rows]
        if min(lower) <= 0:
            raise ValueError('positive hard-band frequencies required')
        weights = [sqrt_upper(Q(row['weight_squared'])) for row in rows]
        fields.append(sum((weight*sqrt_upper(1/frequency) for weight, frequency in zip(weights, lower)), Q(0)))
        frequencies.extend(lower)
    family = descriptor['physical_family']
    division = int(family.get('division_exponent', 0))
    ue0 = rational(family.get('UE0_exact', family['UE']))
    if rational(family['UE']) != ue0/Q(2)**division:
        raise ValueError('selected TOTAL division exponent disagrees with actual UE')
    if rational(family['K']) != rational(family['c'])**2/(2*rational(family['UE'])):
        raise ValueError('actual physical K must equal exact c^2/(2UE)')
    source_norm = max(rational(descriptor['source_harmonic_normalized_axis_jets_q0_to_q6'][0][0]),
        Q(2)**int(descriptor['screened_q6']['normalized_power_of_two_upper_exponents'][0]))
    endpoint_slack = 128*interval_end(descriptor['actual_full10_Gram_distance_interval'],True)+Q(1,2**40)
    return dict(N=n, V=n**3, tau=rational(descriptor['tau'])+endpoint_slack,
        mu=rational(descriptor['mu'])-endpoint_slack, B=rational(descriptor['B'])+endpoint_slack,
        c=rational(family['c']), UE0=ue0, division_exponent=division, source_norm=source_norm,
        omega_min=min(frequencies), field_sum=max(fields),
        native_endpoint_interval_slack=endpoint_slack,
        source_count=len(descriptor['compact_displacement']), source_relative_axis=radius,
        source_relative_coordinates=relative, actual_SU4_source_word_length=word_length,
        exact_mode_ids=[row['mode_id'] for row in descriptor['incoming']],
        declaration=descriptor['physical_output'], source_descriptor=descriptor)


def quadratic_occupation_density(ue: Fraction, c: Fraction, tau: Fraction,
                                 mu: Fraction, volume: int, energy_shift: Fraction,
                                 normal_allowance: Fraction) -> dict[str, Fraction]:
    """Full-CAR gap/Young + scalar-paid Rayleigh comparison; O(UE), not O(sqrtUE)."""
    gstar = Q(13, 16)
    sigma0_squared = ue*sqrt_upper(gstar)/c
    # The actual microscopic H_EM <= Q+S_N form and ||W*Omega||^2>=1/2
    # supply this variance.  No frozen-phase covariance substitutes for it.
    sigma_squared = 4*gstar/(c*c/(2*ue))*energy_shift
    quadratic_cost = 4*tau+8*tau*tau/mu
    polar_trial = 12*ue*tau*tau/(mu*mu)
    energy_density = (3*quadratic_cost*sigma_squared+12*tau*sigma0_squared
        +2*ue*gstar+polar_trial/volume+normal_allowance/volume)
    density = 2*energy_density/mu
    return dict(coexact_variance_upper=sigma_squared, trial_coexact_variance_upper=sigma0_squared,
        pointwise_quadratic_CAR_cost=quadratic_cost, scalar_paid_polar_trial=polar_trial,
        normal_energy_allowance=normal_allowance,
        neutral_excitation_density_upper=density, occupation_energy_density_upper=energy_density)


def normalized_flight_bound(descriptor: dict[str, Any], time: Fraction) -> dict[str, Any]:
    """Complete fixed-body static raw flight/cloud errors; actual global Gram stays explicit."""
    if time < 0:
        raise ValueError('nonnegative physical flight time required')
    n, volume = int(descriptor['N']), int(descriptor['V'])
    tau, mu, b, c, omega = (rational(descriptor[key]) for key in ('tau', 'mu', 'B', 'c', 'omega_min'))
    division = int(descriptor['division_exponent'])
    ue = rational(descriptor['UE0'])/Q(2)**division
    k = c*c/(2*ue)
    number = 4*volume  # Every CAR state, not a ground-state current variance.
    # q_min^2=4sin^2(pi/N)>=16/N^2, without enormous-grid enumeration.
    cc_neutral = ue*volume*n*n/4
    cc_charged = ue*(4*volume+5)*n*n/16
    cc_cross = cc_neutral+cc_charged
    # Pure-colour ALL64 vacuum subtraction and quarter-graph normal admission.
    # Its two half-graph conditions and normal allowance are external exact debts.
    positive_shift = 16*volume+cc_neutral+3*ue/(2*volume)
    normal_allowance = ue*volume/Q(2)**40
    spectral_width = ue*ue/(Q(2)**40*volume*volume)
    effective_ground_residual = 4*normal_allowance+spectral_width
    occupation = quadratic_occupation_density(ue, c, tau, mu, volume, positive_shift,
        effective_ground_residual)
    sigma = sqrt_upper(occupation['coexact_variance_upper'])
    delta = volume*occupation['neutral_excitation_density_upper']/2
    leakage = sqrt_upper(delta)
    gentle = 2*leakage+2*delta  # ACTUAL normalized marginal, including its excited mass.
    root_shift = sqrt_upper(positive_shift)
    hjet = 2*tau
    propagation_l1 = exp_upper(6*tau*time)
    source_norm = rational(descriptor['source_norm'])
    source_l1 = source_norm*sqrt_upper(Q(int(descriptor['source_count'])))*propagation_l1
    source_values = tuple(source_norm*sum((math.comb(q, r)*touchard(r, time*hjet)
        *Q(descriptor['source_relative_axis'])**(q-r) for r in range(q+1)), Q(0)) for q in range(7))
    source = Jet(source_values, 6*tau*source_l1)
    field = rational(descriptor['field_sum'])*sqrt_upper(Q(1,volume))
    onebody_current, onebody_contact = 2*tau*field, 2*tau*field*field
    current_values = free_conjugated(tuple(Q(number)*onebody_current for _ in range(7)), time, hjet)
    contact_values = free_conjugated(tuple(Q(number)*onebody_contact for _ in range(7)), time, hjet)
    current = Jet(current_values, 48*tau*volume*onebody_current*propagation_l1**2)
    contact = Jet(contact_values, 48*tau*volume*onebody_contact*propagation_l1**2)
    decay = omega+2*mu
    p1, p2 = hjet/(2*mu), hjet/(2*mu)+2*hjet*hjet/(mu*mu)
    cloud_values = [Q(number)*onebody_current/decay,
        number*onebody_current*((1+4*p1)/decay+2*hjet/decay**2),
        number*onebody_current*((1+8*p1+4*p2+32*p1*p1)/decay
            +(6*hjet+16*p1*hjet)/decay**2+8*hjet*hjet/decay**3)]
    strip = mu/(8*hjet+2*mu)
    cp = 8*(b+mu)/(3*mu)
    for q in range(3, 7):
        cloud_values.append(number*math.factorial(q)*cp*cp*2*onebody_current/((omega+mu)*strip**q))
    spatial_strip = mu/(48*tau+2*mu)
    cloud_row_l1 = 4*cp*cp*sqrt_upper((1+1/spatial_strip)**3)*onebody_current/(omega+mu)
    cloud = Jet(free_conjugated(tuple(cloud_values), time, hjet),
        48*tau*volume*cloud_row_l1*propagation_l1**2)

    def residual(jet: Jet) -> Fraction:
        return (sigma*jet.light+(cc_cross+effective_ground_residual)*jet.values[0]
            +2*sqrt_upper(ue/Q(volume))*root_shift*sqrt_upper(Q(3))*jet.values[1]
            +ue/Q(volume)*3*jet.values[2])

    def flight(jet: Jet) -> Fraction:
        return time*residual(jet)

    j, z, u, s = current.values[0], contact.values[0], cloud.values[0], source.values[0]
    js, zs, us = current.product(source), contact.product(source), cloud.product(source)
    jjs, jus = current.product(js), current.product(us)
    # Actual vertex replacement on the ground, paid before any photon-factor extraction.
    vertex = 8*tau*3*volume*field*sigma
    contact_vertex = 8*tau*3*volume*field*field*sigma
    pair_0 = number*onebody_current/decay
    pair_1 = cloud_values[1]
    pair_2 = cloud_values[2]
    # Band heat superposition and positive particle-hole propagation; no backward-energy growth.
    initial_pair_mismatch = 2*number*onebody_current*leakage+vertex
    pair_residual_integral = (48*tau*volume*cloud_row_l1*sigma+2*cc_neutral*pair_0
        +2*sqrt_upper(ue/Q(volume))*root_shift*sqrt_upper(Q(3))*pair_1
        +ue/Q(volume)*3*pair_2)
    packet_amputation_error = spectral_width*sqrt_upper((spectral_width+positive_shift)/(ue*omega**3))
    cloud_error = (initial_pair_mismatch+pair_residual_integral
        +effective_ground_residual*cloud_values[0])/omega+packet_amputation_error
    actual_cloud_norm = 2*j/omega
    real = time*time*s*(j*j*flight(source)+j*flight(js)+flight(jjs)+2*vertex*j*s)
    contact_error = time*s*(z*flight(source)+flight(zs)+contact_vertex*s)
    pre = time*s*(j*(s*cloud_error+flight(us))+vertex*s*u+flight(jus))
    post = time*(s*cloud_error*j*s+s*u*(j*flight(source)+vertex*s+flight(js)))
    exchange = s*s*cloud_error*(2*actual_cloud_norm+cloud_error)+s*u*flight(us)
    conditional_norm = s*s*(time*z+time*time*j*j+2*time*j*u+u*u)
    sea_error = gentle*conditional_norm
    # Source-photon graph payment on a band resolvent, not ||RB Omega|| ||a Omega||.
    heat_energy = j*sqrt_upper(Q(2)/omega)  # sqrt(pi/(2e))<1, hence safe <=sqrt(2).
    cloud_positive_graph = sqrt_upper(heat_energy**2+positive_shift*actual_cloud_norm**2)
    cloud_field_graph = sqrt_upper(Q(2)*Q(13, 16)/k)*cloud_positive_graph
    rb_cloud = (6*tau*source_l1*cloud_field_graph+cc_cross*s*actual_cloud_norm
        +2*sqrt_upper(ue/Q(volume))*sqrt_upper(Q(3))*source.values[1]*cloud_positive_graph
        +ue/Q(volume)*3*source.values[2]*actual_cloud_norm)
    source_graph_error = 2*time*time*s*j*rb_cloud
    raw_error = real+contact_error+pre+post+exchange+sea_error+source_graph_error
    actual_kernel_norm = s*s*(time*z+time*time*j*j+2*time*j*actual_cloud_norm+actual_cloud_norm**2)
    gram_coefficient_norm = 2*s*s*actual_cloud_norm**2
    gram_error = s*s*cloud_error*(2*actual_cloud_norm+cloud_error)+gentle*s*s*u*u
    bare_forward_error = s*flight(source)+gentle*s*s+ue*actual_kernel_norm
    # Finite-alpha Born G0 is NOT reset to I. The coherent reference owns its
    # value. Here only the additional actual-ground change is estimated.
    g0_ground_error = s*s*gentle
    gram_radius = g0_ground_error+ue*gram_coefficient_norm
    if gram_radius >= Q(1, 4):
        whitening_error = None
        total = None
    else:
        # On eigenvalues >=1/4: |f'|<=4, |f''|<=24 for f(t)=t^-1/2.
        # Exact 2x2 divided differences preserve the UE factor in the offGram.
        # Input is raw and normalized ONLY by its actual norm.
        m0_upper = s*s
        whitening_error = (4*gram_error*m0_upper+4*gram_coefficient_norm*bare_forward_error
            +24*gram_radius*gram_coefficient_norm*m0_upper+4*gram_radius*actual_kernel_norm)
        whitening_error = 2*whitening_error+4*gram_radius*(2*actual_kernel_norm
            +4*gram_coefficient_norm*m0_upper+2*raw_error)
        total = 4*raw_error+whitening_error
    return dict(N=n,V=volume,time=time,
        physical_family=dict(c=c,UE=ue,K=k,UE0=rational(descriptor['UE0']),
            total_division_exponent=division, definition='UE=UE0/2^TOTAL; K=c^2/(2UE), positive exact rationals'),
        source_relative_axis=descriptor['source_relative_axis'],
        actual_SU4_source_word_length=descriptor['actual_SU4_source_word_length'],
        quadratic_occupation=occupation, actual_Omega_sea_projection_delta=delta,
        actual_harmonic_marginal_gentle_multiplier=gentle, positive_energy_shift=positive_shift,
        normal_ground_operator_residual_allowance=normal_allowance,
        near_infimum_spectral_width=spectral_width,near_infimum_amputation_error=packet_amputation_error,
        normalized_source_norm_upper=source_norm,
        entire_Coulomb_neutral_upper=cc_neutral, entire_Coulomb_charged_upper=cc_charged,
        source_jet=source,current_jet=current,contact_jet=contact,cloud_jet=cloud,
        initial_current_vertex_replacement=vertex, conditional_cloud_error=cloud_error,
        real_both_histories_error=real,contact_error=contact_error,
        outgoing_ket_pre_error=pre,incoming_bra_post_error=post,cross_cloud_exchange_error=exchange,
        full_sea_actual_marginal_error=sea_error,source_photon_graph_error=source_graph_error,
        raw_static_coefficient_error=raw_error,actual_raw_kernel_norm=actual_kernel_norm,
        leading_output_Gram_error=gram_error,leading_output_Gram_norm=gram_coefficient_norm,
        bare_static_forward_error=bare_forward_error,actual_Gram_radius=gram_radius,
        additional_ground_G0_error=g0_ground_error,
        Born_global_G0_retained=True,Born_normalized_Gram_lower_required=Q(1,2),
        output_only_whitening_error=whitening_error,total_normalized_coefficient_error=total,
        volume_rescaled_error=None if total is None else volume*total,
        complete_fixed_body_static_amputation_bound=total is not None,
        actual_global_receiver_reference_required='global Born G0 and input n retained; coherent output-only divided difference, never fibrewise whitening',
        microscopic_weighted_W_error_paid=False,actual_ground_simulated=False,
        source_graph_is_not_plain_state_error_product=True)

def microscopic_sandwich_bound(descriptor: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Full-Phi transfer with currents on opposite source legs; no J-word graph guess."""
    n, volume, time = int(descriptor['N']),int(descriptor['V']),body['time']
    if n % 2 or n < 2**96:
        raise ValueError('pure-colour NL0 vacuum needs the actual even legal-volume family')
    ue = body['physical_family']['UE']
    omega, tau = descriptor['omega_min'],descriptor['tau']
    budget = normal.native_normal_budget(n)
    phase = scalar_phase.phase_bounds(n)
    shift = body['positive_energy_shift']
    cx = normal.sqrt_upper((Q(1,5)+6*Q(3,25))**2+3*Q(3,10)**2)
    a4 = Q(1,4*volume)
    slope = (8*volume+2)*cx
    radius = Q(2)**max(0,(normal.ceil_log2(2*slope/a4)+2)//3)
    cd = slope*radius
    cq = sqrt_upper(sqrt_upper(2/a4))
    crho = sqrt_upper(sqrt_upper(2*(cd+budget.vacuum+shift+budget.exterior_light_constant+1)/a4))
    k0 = descriptor['source_norm']
    exponents = descriptor['source_descriptor']['screened_q6']['normalized_power_of_two_upper_exponents']
    # Tr(BaBb)/4=delta -> physical Cartesian norm; native Ta=Ba/sqrt8.
    k1,k2 = Q(4)*Q(2)**int(exponents[1]),Q(4)*Q(2)**int(exponents[2])
    source_dimension = 75*(int(descriptor['source_count'])+1)*(descriptor['actual_SU4_source_word_length']+1)
    bh1,bh2 = k0*descriptor['source_relative_axis'],k0*descriptor['source_relative_axis']**2
    field = descriptor['field_sum']*sqrt_upper(Q(1,volume))
    jr,zr = 16*volume*(1+tau)*field,16*volume*(1+tau)*field*field
    j,z = body['current_jet'].values[0],body['contact_jet'].values[0]
    jfull = jr*(crho+cq)
    packet = sqrt_upper((1+shift)/(ue*omega**3))
    birth_norm = sqrt_upper(1+shift/omega)
    omega_max = max(interval_end(row['omega_interval'],True)
        for beam in ('incoming','outgoing_annihilation') for row in descriptor['source_descriptor'][beam])
    birth_q = omega_max*birth_norm+sqrt_upper(ue)*jfull+sqrt_upper(1+(1+shift)/omega)
    nmax = max(Q(1),jfull/omega+packet,2*j/omega,birth_norm)
    qmax = max(Q(1),jfull+omega*packet,2*j,birth_q)
    emax = nmax*qmax
    photon_half = sqrt_upper(emax+shift*nmax*nmax)
    kinetic = sqrt_upper(2*emax)+sqrt_upper(2*(budget.vacuum+shift+budget.exterior_light_constant+1))*nmax
    rho = (crho+cq)*(nmax+sqrt_upper(emax))
    coulomb = body['entire_Coulomb_neutral_upper']+body['entire_Coulomb_charged_upper']
    harmonic = 2*sqrt_upper(ue/Q(volume))*sqrt_upper(Q(3))*bh1*photon_half+3*ue/Q(volume)*bh2*nmax
    # Actual stationary-source COMMUTATOR, not its free-flow residual RB.
    # +2K0 additionally pays Bdot if an interaction-picture source is used.
    rb = 2*k1*kinetic+source_dimension*k2*nmax+4*(1+tau)*k0*rho
    rb += (coulomb+2)*k0*nmax+harmonic
    rb_effective = (coulomb+4)*k0*nmax+harmonic
    source_half = sqrt_upper(k0*k0*nmax*qmax+k0*nmax*rb)
    effective_source_half = sqrt_upper(k0*k0*nmax*qmax+k0*nmax*rb_effective)
    defect_half = source_half+effective_source_half+k0*nmax
    cloud_weight = (1+20*jfull)/omega+2*j/omega**2+packet
    source_weight = max(4*k0+1,k0*cloud_weight+nmax)+time*k0*nmax
    quarter_weight = cq*sqrt_upper(defect_half)*sqrt_upper(source_weight)
    current_weight = k0*nmax+jr*(crho*source_weight+quarter_weight)
    contact_weight = k0*nmax+zr*(crho*source_weight+quarter_weight)
    source_cap = k0*nmax
    current_cap = max(j*source_cap,jr*(crho+cq)*(source_cap+source_half))
    contact_cap = max(z*source_cap,zr*(crho+cq)*(source_cap+source_half))
    real_weight = time*time*(2*current_cap*current_weight+time*j*j*source_cap**2)
    contact_word_weight = time*(2*source_cap*contact_weight+2*source_weight*contact_cap)
    cloud_words_weight = 2*time*(2*source_weight*current_cap+2*source_cap*current_weight
        +time*source_cap**2*j)
    exchange_weight = 2*source_cap*source_weight+time*source_cap**2
    raw_weight = real_weight+contact_word_weight+cloud_words_weight+exchange_weight
    gram_weight = 2*source_cap*source_weight
    bare_weight = 2*source_cap*source_weight+time*source_cap**2
    g0_weight = (1+ue)*gram_weight
    kn,gn = body['actual_raw_kernel_norm']+1,body['leading_output_Gram_norm']+1
    m0 = k0*k0
    whitening_weight = 2*(4*gram_weight*m0+4*gn*bare_weight
        +24*g0_weight*gn*m0+4*g0_weight*kn)
    whitening_weight += 4*g0_weight*(4*kn+4*gn*m0)
    total_weight = 1+4*raw_weight+whitening_weight
    target = ue/(Q(volume)*Q(2)**40)
    delta = min(Q(1,64),target/(8*total_weight),body['normal_ground_operator_residual_allowance']/8,
        1/(16*source_weight),1/(16*g0_weight))
    multiplier = max(Q(1),field,field*field,64*k0,
        4*budget.chart_majorant*k1*(budget.soft_moment+budget.stiff_moment)/budget.residual_light)
    certificate = normal.choose_lambda(budget,delta,time,1,shift+16,multiplier,phase)
    base_k = certificate['local_log2_Lambda']//80
    extra = dict(quarter_graph=(normal.ceil_log2(1/delta)+19)//20,
        photon_half_graph=(normal.ceil_log2(budget.current_lipschitz**2)+79)//80,
        photon_half_graph_vacuum=(normal.ceil_log2(budget.current_lipschitz**2*budget.vacuum)+49)//50)
    k = max(base_k,*extra.values())
    global_exponent = dict(certificate['global_log2_Lambda_expression'])
    global_exponent['local_log2_t'] = k
    ell,height = global_exponent['exterior_log2_t'],global_exponent['coercivity_reciprocal_inner_exponent']
    certificate['Lambda_expression'] = f'2^(80*({k}+{ell}+8*2^(2^({height}))))'
    certificate['local_log2_Lambda'] = 80*k
    certificate['global_log2_Lambda_expression'] = global_exponent
    spectral_width = min(body['near_infimum_spectral_width'],delta*delta*ue/(Q(2)**40*volume*volume))
    transfer = delta*total_weight
    charge = normal.microscopic_charge_lower(normal.sqrt_lower(Q('0.418272259'))/5,
        delta,delta,shift+16,shift)
    if charge['charged_minus_neutral_infimum_lower'] <= Q(3,25) or transfer > target or spectral_width > delta:
        raise AssertionError('original microscopic sandwich/charge/spectral admission failed')
    return dict(N=n,V=volume,original_kinetic_exponent=Q(-1,2),source_Cartesian_dimension_upper=source_dimension,
        source_Cartesian_first_upper=k1,source_Cartesian_second_upper=k2,
        source_global_norm_upper=k0,rho_constant=crho,rho_quarter_constant=cq,
        scalar_vacuum_coefficient=budget.vacuum,quartic_exterior_constant=budget.exterior_light_constant,
        actual_full_current_ground_norm_upper=jfull,
        actual_source_charged_energy_half_upper=source_half,
        actual_prepared_input_energy_half_upper=2*source_half,
        actual_global_receiver_energy_half_upper=3*source_half,
        source_energy_uses_commutator_and_Bdot=True,actual_RB_U_half_graph_assumed=False,
        normal_certificate=certificate,additional_exact_log_requirements=extra,
        common_microscopic_charge=charge,normal_small_parameter=delta,
        selected_near_infimum_spectral_width=spectral_width,
        real_both_histories_weight=real_weight,contact_weight=contact_word_weight,
        both_PRE_POST_weight=cloud_words_weight,exchange_weight=exchange_weight,
        source_propagation_norm_error_upper=delta*source_weight,
        source_propagation_current_error_upper=delta*current_weight,
        normalized_full_Phi_coefficient_error_upper=transfer,
        volume_rescaled_full_Phi_error_upper=volume*transfer,
        quarter_graph_not_half_graph_closure=True,original_static_two_vector_Gram=True,
        every_exact_ALL64_scalar_offset_retained=True,
        neutral_state_domain='C-real neutral kappa0 fibre or explicit kappa/-kappa spectral pair; never arbitrary fixed moving kappa',
        actual_full_Phi_word_transfer_paid=True,actual_legal_volume_state_simulated=False,
        proof='source half-energy plus rho-quarter interpolation; both real currents on opposite source legs')





class FiniteFlightMap:
    """FFT-defined smooth full-CAR map; every finite harmonic projection is retained."""
    def __init__(self, neutral: Coupled, charged: Coupled, time: float, kind: str,
                 samples: int = 128, orders: int = 48):
        self.neutral = neutral
        self.target = neutral if kind.startswith('neutral') else charged
        self.kind,self.time = kind,time
        creation = creators(neutral,charged)
        neutral_basis = np.array([exterior(np.eye(8,dtype=complex)[i:i+1].T
            @np.eye(8,dtype=complex)[j:j+1],neutral.rows) for i in range(8) for j in range(8)])
        charged_basis = np.array([exterior(np.eye(8,dtype=complex)[i:i+1].T
            @np.eye(8,dtype=complex)[j:j+1],charged.rows) for i in range(8) for j in range(8)])
        alpha = np.array([1.,.3j,-.2+.4j,.6]);alpha /= np.linalg.norm(alpha)
        seed = np.zeros((8,4),complex)
        seed[:4] = sum(alpha[j]*([np.eye(4)]+[SELECTED[0]@SELECTED[i] for i in range(1,4)])[j] for j in range(4))/math.sqrt(2)
        onsite = np.kron(np.eye(2),(.2+3*.12)*neutral.frame[0])
        hops=[]
        for axis in range(3):
            hop=np.zeros((8,8),complex)
            hop[:4,4:]=-(.12*neutral.frame[0]+1j*.3*neutral.frame[axis+1])/2
            hops.append(hop)
        f=np.array([1.,-1.,0.])/math.sqrt(2*neutral.omega)
        matrices,dots=[],[]
        for angle in 2*math.pi*np.arange(samples)/samples:
            h=onsite.copy();current=np.zeros((8,8),complex);contact=np.zeros((8,8),complex)
            for axis,(hop,nu) in enumerate(zip(hops,(1,1,-2))):
                phase=np.exp(1j*(nu*angle+neutral.external[axis]))
                h += phase*hop+phase.conjugate()*hop.conj().T
                current += 1j*f[axis]*(phase*hop-phase.conjugate()*hop.conj().T)
                contact -= f[axis]**2*(phase*hop+phase.conjugate()*hop.conj().T)
            values,vectors=np.linalg.eigh(h)
            flow=(vectors*np.exp(-1j*time*values))@vectors.conj().T
            mark=flow@seed*np.exp(-1j*angle)
            b=np.einsum('ir,icd->rcd',mark,creation)
            bd=np.einsum('ir,icd->rcd',h@mark,creation)
            if 'cloud' in kind:
                vertex=vectors.conj().T@current@vectors
                tangent=np.zeros((8,8),complex)
                for a in np.flatnonzero(values>0):
                    for z in np.flatnonzero(values<0):
                        tangent[a,z]=-vertex[a,z]/(values[a]-values[z]+neutral.omega)
                onebody=flow@(vectors@tangent@vectors.conj().T)@flow.conj().T
            else:
                onebody=flow@(contact if 'contact' in kind else current)@flow.conj().T
            idot=h@onebody-onebody@h
            basis=neutral_basis if kind.startswith('neutral') or 'cloud' in kind else charged_basis
            op=np.einsum('ij,ijcd->cd',onebody,basis.reshape(8,8,basis.shape[-2],basis.shape[-1]))
            opd=np.einsum('ij,ijcd->cd',idot,basis.reshape(8,8,basis.shape[-2],basis.shape[-1]))
            if kind.startswith('neutral'):
                matrix,dot=op[None,:,:],opd[None,:,:]
            elif 'cloud' in kind:
                matrix,dot=b@op,bd@op+b@opd
            else:
                matrix,dot=op@b,opd@b+op@bd
            matrices.append(matrix);dots.append(dot)
        self.orders=np.arange(-orders,orders+1)
        self.coefficients=np.fft.fft(np.asarray(matrices),axis=0)[self.orders%samples]/samples
        self.idot_coefficients=np.fft.fft(np.asarray(dots),axis=0)[self.orders%samples]/samples
        self.references=self.coefficients.shape[1]

    def apply(self, vector: np.ndarray, idot: bool=False) -> np.ndarray:
        source=vector.reshape(self.neutral.n,self.neutral.l,self.neutral.d)
        out=np.zeros((self.references,self.target.n,self.target.l,self.target.d),complex)
        coefficients=self.idot_coefficients if idot else self.coefficients
        for order,coefficient in zip(self.orders,coefficients):
            if abs(order)>=self.neutral.l:
                continue
            if order>=0:
                left,right=slice(order,self.neutral.l),slice(0,self.neutral.l-order)
            else:
                left,right=slice(0,self.neutral.l+order),slice(-order,self.neutral.l)
            out[:,:,left] += np.einsum('nld,rcd->rnlc',source[:,right],coefficient,optimize=True)
        return out.reshape(self.references,self.target.size)

    def residual(self, vector: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        created=self.apply(vector)
        total=self.apply(vector,True)-np.array([self.target.action(row) for row in created])+self.apply(self.neutral.action(vector))
        charged=created.reshape(self.references,self.target.phdim,self.target.d)
        neutral=self.neutral.reshape(vector)
        coulomb=-charged*self.target.coulomb+self.apply((neutral*self.neutral.coulomb).ravel()).reshape(charged.shape)
        p0=np.tile(self.neutral.UE*(self.neutral.ell+self.neutral.zeta)**2/6,self.neutral.n)
        p1=np.tile(self.target.UE*(self.target.ell+self.target.zeta)**2/6,self.target.n)
        harmonic=-charged*p1[None,:,None]+self.apply((neutral*p0[:,None]).ravel()).reshape(charged.shape)
        return total,coulomb.reshape(created.shape),harmonic.reshape(created.shape)


def trace(model: Coupled) -> float:
    return float(model.phenergy.sum()*model.d+model.phdim*np.trace(model.onsite).real+model.phdim*model.coulomb.sum())


def propagate(model: Coupled, rows: np.ndarray, time: float) -> np.ndarray:
    if time==0:
        return rows.copy()
    return np.asarray(expm_multiply(-1j*time*model.operator,rows.T,traceA=-1j*time*trace(model))).T


def finite_science(n: int=3, cutoff: int=8, duration: float=1., nodes: int=8) -> dict[str, Any]:
    band=RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    _,frame=original_frame(band,5,np.array([1,2,3])*math.pi/2,True)
    ue,k=.0001,28.361421309064075
    neutral=Coupled(n,cutoff,0,4,ue,k,frame)
    charged=Coupled(n,cutoff,0,5,ue,k,frame)
    energies,vectors,eigen_errors=neutral.solve(1)
    energy=float(energies[0]);omega=vectors[:,0]
    records=[]
    for kind in ('neutral_current','charged_current','neutral_cloud','charged_cloud','charged_contact'):
        first=FiniteFlightMap(neutral,charged,0.,kind)
        last=FiniteFlightMap(neutral,charged,duration,kind)
        actual=propagate(first.target,first.apply(omega),duration)
        predicted=np.exp(-1j*energy*duration)*last.apply(omega)
        defect=actual-predicted
        integrals=[]
        integrals_c=[];integrals_p=[]
        for order in (nodes,2*nodes):
            x,w=np.polynomial.legendre.leggauss(order)
            total=np.zeros_like(defect);coulomb=np.zeros_like(defect);harmonic=np.zeros_like(defect)
            for at,weight in zip((x+1)*duration/2,w*duration/2):
                field=FiniteFlightMap(neutral,charged,float(at),kind)
                residual,c,p=field.residual(omega)
                scale=1j*weight*np.exp(-1j*energy*at)
                total += scale*propagate(field.target,residual,float(duration-at))
                coulomb += scale*propagate(field.target,c,float(duration-at))
                harmonic += scale*propagate(field.target,p,float(duration-at))
            integrals.append(total);integrals_c.append(coulomb);integrals_p.append(harmonic)
        quadrature=float(np.linalg.norm(integrals[1]-integrals[0]))
        budget=max(3e-8,4*quadrature+duration*max(eigen_errors)*float(np.linalg.norm(first.coefficients)))
        equality=accept(np.linalg.norm(defect-integrals[1]),'WHOLE charged/neutral current-state Duhamel flight',budget)
        missing_c=adverse(np.linalg.norm(defect-integrals[1]+integrals_c[1]),'WHOLE charged/neutral current-state Duhamel flight',budget)
        missing_p=adverse(np.linalg.norm(defect-integrals[1]+integrals_p[1]),'WHOLE charged/neutral current-state Duhamel flight',budget)
        missing_light=adverse(np.linalg.norm(defect-integrals_c[1]-integrals_p[1]),
            'WHOLE charged/neutral coexact-Peierls current-state flight',budget)
        records.append(dict(kind=kind,references=first.references,initial_norm=float(np.linalg.norm(first.apply(omega))),
            exact_flight_defect=float(np.linalg.norm(defect)),Duhamel_identity_error=equality,
            numerical_budget=budget,quadrature_difference=quadrature,
            missing_Coulomb_reference_rejected=missing_c,missing_harmonic_electric_rejected=missing_p,
            missing_coexact_Peierls_flight_rejected=missing_light,
            exact_Coulomb_flight_norm=float(np.linalg.norm(integrals_c[1])),
            exact_harmonic_flight_norm=float(np.linalg.norm(integrals_p[1]))))
    return dict(n=n,M=cutoff,time=duration,UE=ue,K=k,neutral_dimension=neutral.size,charged_dimension=charged.size,
        whole_neutral_number=4,whole_charged_number=5,source_filter=False,ground_energy=energy,
        spatial_sites=2,coexact_photon_dimension=n,harmonic_dimension=2*cutoff+1,
        ground_eigenpair_errors=eigen_errors,ground_sha256=hashlib.sha256(omega.tobytes()).hexdigest(),
        actual_full_CAR_flight=records,
        scope='two-site full CAR/coexact-photon/harmonic coupled BODY maps; not a legal-volume simulation',
        isolated_charged_pole_required=False)


def native_far_setup(size: int=5) -> dict[str, Any]:
    """Literal mobile Coulomb/harmonic + SU4 tree source, with RIGHT=(0,D,D)."""
    data = joined.setup(size)
    law, momentum, origin = data['law'],data['momentum'],data['origin']
    reference = int(law.index[(0,int(origin[1]),int(origin[2]))])
    _,_,baseline = joined.graph(size,data['centre'],data['frame'],data['band'])
    fields = np.tile(data['frame'],(law.volume,1,1,1))
    links = law.toron_links(np.asarray(data['centre'])*math.pi/2)
    green_matrix,_,_ = joined.green(law)
    alpha = np.array([1.,.3j,-.2+.4j,.6]);alpha /= np.linalg.norm(alpha)
    def mark(eta: np.ndarray) -> np.ndarray:
        marks = joined.source_marks(law,fields,links,eta,reference,baseline,np.zeros(3),green_matrix)
        return momentum.to_momentum(np.einsum('a,air->ir',alpha,marks))/math.sqrt(2)
    source = mark(data['eta'])
    recoil = data['eta']*np.exp(1j*(law.sites-law.sites[reference])@data['transfer'])
    detector = mark(recoil)
    accept(abs(np.linalg.norm(momentum.project(source,False))**2-1),
        'far RIGHT genuine mobile tree source complete bar4 CT norm')
    return {**data,'reference':reference,'full_source':source,'full_detector':detector,
        'source':momentum.project(source,False),'detector':momentum.project(detector,False)}


def native_quadratic_witness(data: dict[str, Any]) -> dict[str, Any]:
    """Exact quadratic-CAR spectral reduction minimizes over ALL 2^(4V) occupations."""
    h, law = data['h'], data['law']
    energies, vectors = np.linalg.eigh(h)
    negative = energies < 0
    projector = vectors[:, negative]@vectors[:, negative].conj().T
    number = int(negative.sum())
    gap = float(np.min(abs(energies)))*(1-1e-12)
    tau = max(float(np.linalg.norm(h[4*x:4*x+4,4*int(law.target[x,j]):4*int(law.target[x,j])+4],2))
              for x in range(law.volume) for j in range(3))*(1+1e-12)
    field = np.zeros((law.volume,3))
    field[:,2] = .03*np.cos(2*math.pi*law.sites[:,0]/law.size)
    changed = joined.literal_peierls(law,h,field,np.zeros_like(field),1.,0.).toarray()
    difference = changed-h
    norm_squared = float(np.vdot(difference,difference).real)
    trace_pd = float(np.trace(projector@difference).real)
    sum_phase_squared = float(np.vdot(field,field).real)
    constant = -float(energies[negative].sum())-gap*number/2-trace_pd
    trial_operator = changed-gap*(np.eye(len(h))-2*projector)/2
    trial_energies = np.linalg.eigvalsh(trial_operator)
    full_CAR_minimum_without_Young = float(trial_energies[trial_energies<0].sum())+constant
    full_CAR_minimum = full_CAR_minimum_without_Young+norm_squared/gap
    accept(max(0.,-full_CAR_minimum),'quadratic gap/Young form on EVERY native CAR occupation',3e-9)
    rejected = adverse(-full_CAR_minimum_without_Young,
        'quadratic gap/Young form on EVERY native CAR occupation',3e-9)
    accept(max(0.,norm_squared-8*tau*tau*sum_phase_squared),
        'native unaliased hopping HS quadratic phase bound',3e-10)
    accept(max(0.,-trace_pd-4*tau*sum_phase_squared),
        'coexact filled-sea quadratic phase trace bound',3e-10)
    linear = np.zeros_like(h)
    for x in range(law.volume):
        for j in range(3):
            y = int(law.target[x,j]);a,b = slice(4*x,4*x+4),slice(4*y,4*y+4)
            linear[a,b] += 1j*field[x,j]*h[a,b]
            linear[b,a] -= 1j*field[x,j]*h[b,a]
    linear_trace = accept(abs(np.trace(projector@linear)),
        'filled-sea coexact linear current cancels without removing scalar Esea',3e-10)
    return dict(N=law.size,onebody_dimension=len(h),whole_negative_occupation=number,
        full_CAR_states=f'2^{len(h)}',positive_band_restriction=False,
        field_amplitude=.03,field='z link cos(2pi Xx/N), native coexact zero-mean',
        gap=gap,endpoint_norm=tau,difference_HS_squared=norm_squared,Tr_P_D=trace_pd,
        scalar_Esea=float(energies[negative].sum()),coexact_linear_trace_error=linear_trace,
        minimum_over_every_CAR_occupation=full_CAR_minimum,
        same_consumer_without_Young_rejected=rejected,
        scope='executed native finite input; global quadratic-form proof is algebraic')


def static_native_bridge(data: dict[str, Any], duration: float=6., steps: int=64) -> dict[str, Any]:
    """Independent bordered-exterior STATIC readout, all sea columns and full bar4."""
    momentum, law = data['momentum'],data['law']
    static = dict(data)
    static['full_detector'] = momentum.evolve(data['full_detector'],-duration)
    static['outgoing'] = {**data['outgoing'],'modes':[{**row,
        'coefficient':row['coefficient']*np.exp(-1j*row['omega']*duration)}
        for row in data['outgoing']['modes']]}
    coarse,_ = joined.prepared_reduced_output(static,duration,16)
    fine,gap = joined.prepared_reduced_output(static,duration,32)
    pre,_ = joined.preparation_tangent(momentum,static['outgoing'],0.)
    post,_ = joined.preparation_tangent(momentum,joined.annihilation_packet(data['incoming']),duration)
    def finite_difference(resolution: int) -> np.ndarray:
        return joined.mixed_difference(lambda x,y:joined.exterior_ratio(momentum,data['h'],
            data['incoming'],static['outgoing'],data['full_source'],static['full_detector'],
            duration,resolution,x,y,pre=pre,post=post,full_output=True),4e-4)
    literal,literal_fine = finite_difference(steps),finite_difference(2*steps)
    step_error = float(np.linalg.norm(literal_fine-literal))
    quadrature = float(np.linalg.norm(fine['full']-coarse['full']))
    budget = max(4e-7,4*step_error+4*quadrature)
    identity = accept(np.linalg.norm(fine['full']-literal_fine),
        'STATIC fullbar4 contour versus independent complete exterior',budget)
    raw = complex(np.vdot(static['full_detector'],fine['full']))
    speed = data['incoming']['modes'][0]['omega']/np.linalg.norm(1-np.exp(-1j*data['incoming']['modes'][0]['k']))
    forward = joined.packet(law,0,float(speed),data['origin']+np.array([1,0,0]))
    gram_data = {**data,'incoming':forward}
    gram_parts,_ = joined.prepared_reduced_output(gram_data,0.,4)
    gram = complex(np.vdot(data['full_detector'],gram_parts['full']))
    gp,_ = joined.preparation_tangent(momentum,data['outgoing'],0.)
    gq,_ = joined.preparation_tangent(momentum,joined.annihilation_packet(forward),0.)
    literal_gram = joined.mixed_difference(lambda x,y:joined.exterior_ratio(momentum,data['h'],
        forward,data['outgoing'],data['full_source'],data['full_detector'],0.,1,x,y,
        pre=gp,post=gq),4e-4)
    gram_error = accept(abs(gram-literal_gram),'STATIC actual forward/recoil receiver offGram cloud',4e-8)
    photon_overlap = sum(np.conj(left['coefficient'])*right['coefficient']
        *np.exp(-1j*right['omega']*duration) for left,right in zip(forward['modes'],data['incoming']['modes']))
    bare_electron = complex(np.vdot(data['full_source'],
        momentum.project(momentum.evolve(data['full_source'],duration),False)))
    bare = photon_overlap*bare_electron
    bare_literal = joined.exterior_ratio(momentum,data['h'],data['incoming'],static['outgoing'],
        data['full_source'],momentum.evolve(data['full_source'],-duration),duration,1,0.,0.)
    bare_error = accept(abs(bare_electron-bare_literal),'STATIC forward fullbar4 zero-drive amplitude',3e-11)
    coefficient = raw-.5*gram*bare
    old,_ = joined.prepared_reduced_output(data,duration,32)
    moving = complex(np.vdot(data['full_detector'],old['full']))
    moving0,_ = joined.prepared_reduced_output(data,0.,4)
    old_connected = moving-complex(np.vdot(data['full_detector'],moving0['full']))
    wrong = adverse(abs(coefficient-old_connected),'ORIGINAL STATIC output-only global whitening target',budget)
    hole = adverse(abs(np.vdot(momentum.project(static['full_detector'],True),fine['full'])),
        'STATIC unfiltered negative final detector',budget)
    return dict(N=law.size,T=duration,reference=law.sites[data['reference']],complete_occupied_columns=2*law.volume,
        all_reference_rows=4,input_whitened=False,detector_static=True,source_positive_filter=False,
        raw_static_C=raw,receiver_T0_offGram=gram,bare_forward_photon_overlap=photon_overlap,
        bare_forward_electron_overlap=bare_electron,bare_forward_mx0=bare,
        output_only_whitened_leading_coefficient=coefficient,independent_complete_exterior_error=identity,
        independent_receiver_offGram_error=gram_error,independent_bare_forward_error=bare_error,
        numerical_budget=budget,finite_step_difference=step_error,quadrature_difference=quadrature,
        moving_CminusC0_wrong_target_rejected=wrong,negative_final_detector_omission_rejected=hole,
        positive_particle_hole_plus_hard_photon_gap=gap,
        global_actual_nu_integration_performed=False,coherent_global_product_order='meanC-0.5meanG*meanM0',
        scope='executed finite static three-kernel identity, not a legal-volume observation')


def hashes() -> dict[str,str]:
    paths={Path(__file__).resolve(),ROOT/'noncompact_scattering_flight.md'}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and Path(filename).resolve().parent==ROOT:
            paths.add(Path(filename).resolve())
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sys.set_int_max_str_digits(0)  # Exact rational witnesses exceed Python's display-only limit.
    parser.add_argument('--descriptor',type=Path,required=True)
    parser.add_argument('--time',default='6')
    parser.add_argument('--rescaled-target',default='0.0001')
    parser.add_argument('--native-steps',type=int,default=64)
    parser.add_argument('--bounds-only',action='store_true')
    parser.add_argument('--n',type=int,default=3)
    parser.add_argument('--M',type=int,default=8)
    parser.add_argument('--finite-time',type=float,default=1.)
    parser.add_argument('--nodes',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    before=hashes()
    descriptor=compact_descriptor(json.loads(args.descriptor.read_text()))
    bound=normalized_flight_bound(descriptor,Q(args.time))
    target=Q(args.rescaled_target)
    if target <= 0 or bound['volume_rescaled_error'] is None or bound['volume_rescaled_error'] > target:
        raise AssertionError('actual selected TOTAL family fails requested normalized flight budget')
    microscopic=microscopic_sandwich_bound(descriptor,bound)
    leading=descriptor['source_descriptor']['leading_static_receiver']
    reference_margin=rational(leading['every_actual_normalized_nu_GLOBAL_leading_rescaled_lower'])
    total_error=bound['volume_rescaled_error']+microscopic['volume_rescaled_full_Phi_error_upper']
    margin=reference_margin-total_error
    if margin <= 0:
        raise AssertionError('original coherent GLOBAL receiver loses its positive lower')
    amplitude=bound['physical_family']['UE']*margin/Q(descriptor['V'])
    science=None
    if not args.bounds_only:
        native=native_far_setup(5)
        science=dict(body=finite_science(args.n,args.M,args.finite_time,args.nodes),
            native_quadratic=native_quadratic_witness(native),
            original_static_global_bridge=static_native_bridge(native,float(args.time),args.native_steps))
    after=hashes()
    if before!=after:
        raise AssertionError('Flight/source/current dependencies changed during execution')
    result=dict(plan=PLAN,template_revision=TEMPLATE_REVISION,source_sha256=after,
        descriptor_sha256=hashlib.sha256(args.descriptor.read_bytes()).hexdigest(),
        runtime=dict(executable=sys.executable,python=sys.version,argv=sys.argv),
        bound=bound,microscopic_sandwich=microscopic,actual_finite_science=science,
        global_receiver=dict(reference_margin=reference_margin,volume_rescaled_total_error=total_error,
            positive_margin=margin,amplitude_lower=amplitude,probability_lower=amplitude*amplitude,
            exact_two_coherent_vectors=True,all_bar4_references_internal=True,
            input_whitened=False,detector_static=True,actual_finite_energy_state_constructed_by_spectral_filter=True,
            legal_volume_probability_simulated=False),
        physical_receiver_probability_certified=True,
        certification_scope='derived finite-parameter microscopic original-state theorem, with executed finite identities below',
        LAW05_closed=False)
    text=json.dumps(result,default=encode,indent=2,allow_nan=False)
    args.output.write_text(text+'\n')
    print(json.dumps(dict(output=str(args.output),source_sha256=after,
        selected_TOTAL_division=bound['physical_family']['total_division_exponent'],
        body_volume_error_upper=decimal_bound(bound['volume_rescaled_error']),
        full_Phi_volume_error_upper=decimal_bound(microscopic['volume_rescaled_full_Phi_error_upper']),
        amplitude_lower=decimal_bound(amplitude,False),probability_lower=decimal_bound(amplitude*amplitude,False),
        executed_science=science),default=encode,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
