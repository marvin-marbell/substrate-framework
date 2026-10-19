#!/usr/bin/env python3
"""Unfiltered original-Phi source spectral tails and source-valued flight."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING, localcontext
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

import noncompact_quantitative_normal_budget as normal
import noncompact_quantitative_scalar_phase as scalar
from noncompact_scattering_flight import (compact_descriptor, normalized_flight_bound,
    sqrt_upper as precise_sqrt_upper, free_conjugated)
from noncompact_native_current import Coupled, accept, adverse, exterior
from noncompact_photon_amputation_graph import creators
from original_charged_principal import original_frame
from rotation_band import RotationBand
from sea_selection import SELECTED

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
F = Fraction


def sqrt_upper(value: Fraction) -> Fraction:
    """Outward root without materializing a multi-million-bit integer square."""
    if value<0:
        raise ValueError('Nonnegative graph form required')
    if value==0:
        return F(0)
    if max(value.numerator.bit_length(),value.denominator.bit_length())>16384:
        return F(2)**((normal.ceil_log2(value)+1)//2)
    return precise_sqrt_upper(value)


def dyadic_upper(value: Fraction) -> Fraction:
    return F(2)**normal.ceil_log2(value) if value else F(0)


def dyadic_value(text: str) -> Fraction:
    if not text.startswith('2^'):
        raise ValueError('Supplier must provide an exact dyadic graph cap')
    return F(2)**int(text[2:])


@dataclass(frozen=True)
class SourceGraph:
    """Actual SOURCE graphs, never a positive-half graph of R_B u."""
    norm: Fraction
    first: Fraction
    half: Fraction
    commutator: Fraction
    jets: tuple[Fraction, ...]


def source_graph(descriptor: dict[str, Any], body: dict[str, Any],
                 photon: dict[str, Any], raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Original chi-weighted normal, Coulomb and harmonic source product rule."""
    n, volume = descriptor['N'], descriptor['V']
    budget = normal.native_normal_budget(n)
    ue, omega, tau = body['physical_family']['UE'], descriptor['omega_min'], descriptor['tau']
    shift = body['positive_energy_shift']
    cx = sqrt_upper((F(1, 5)+6*F(3, 25))**2+3*F(3, 10)**2)
    a4 = F(1, 4*volume)
    slope = (8*volume+2)*cx
    radius = F(2)**max(0, (normal.ceil_log2(2*slope/a4)+2)//3)
    cd = slope*radius
    selected=photon['selected']
    lambda_floor=int(selected['actual_global_Lambda_floor_log2'])
    vacuum_log=normal.ceil_log2(2*(budget.vacuum+shift+budget.exterior_light_constant+1))
    if lambda_floor<8*vacuum_log:
        raise AssertionError('Actual unchanged Lambda does not pay chi-weighted source momentum')
    # The accepted SAME Lambda pays sqrt(2Cvac) Lambda^-1/16<=1
    # and Cvac Lambda^-5/8<=1. Neither zero-point energy is discarded.
    rho0 = sqrt_upper(sqrt_upper(2*(cd+shift+budget.exterior_light_constant+2)/a4))
    rhoquarter = sqrt_upper(sqrt_upper(2/a4))
    field = descriptor['field_sum']*sqrt_upper(F(1, volume))
    current_coefficient = 16*volume*(1+tau)*field
    contact_coefficient = current_coefficient*field
    current_ground = current_coefficient*(rho0+rhoquarter)
    cloud_norm=dyadic_value(selected['U_norm_upper'])
    cloud_first=dyadic_value(selected['QU_norm_upper'])
    birth_norm=sqrt_upper(1+dyadic_value(selected['all_mode_photon_number_upper']))
    birth_first=dyadic_value(selected['actual_photon_birth_Q_first_graph_upper'])
    rho2_inputs=dict(cloud=dyadic_value(selected['rho_squared_U_graph_upper']),
        birth=dyadic_value(selected['rho_squared_vector_norm_upper'])+1)
    if raw is None:
        exponents = descriptor['source_descriptor']['screened_q6']['normalized_power_of_two_upper_exponents']
        support, word = descriptor['source_count'], descriptor['actual_SU4_source_word_length']
        relative = descriptor['source_relative_axis']
        source_identity = dict(eta=18, profile=19, detector_source=True)
    else:
        if (raw['N'] != n or raw['source_profile']['profile_native_N'] != 41
                or raw['source_profile']['screen_eta'] != 18
                or raw['source_profile']['alpha'] != 2**371):
            raise ValueError('RAW profile41/screen18/alpha371 must be the actual same-N distinct source')
        exponents = raw['source_q6_upper_exponents']
        support, word = 41**3, raw['source_profile']['tree_max']
        relative = word
        source_identity = dict(eta=18, profile_lattice=41, support_radius=20,
            source_count_upper=support, detector_source=False)
    jets = tuple(F(4)*F(2)**int(exponent) for exponent in exponents)
    jets = free_conjugated(jets,body['time'],2*tau)
    k0 = max(descriptor['source_norm'], jets[0])
    k1, k2 = jets[1:3]
    dimension = 75*(support+1)*(word+1)
    coulomb = body['entire_Coulomb_neutral_upper']+body['entire_Coulomb_charged_upper']
    graphs: dict[str, SourceGraph] = {}
    for label, n0, q0 in (('birth', birth_norm, birth_first), ('cloud', cloud_norm, cloud_first)):
        e0 = sqrt_upper(n0*q0)
        photon_half = sqrt_upper(n0*q0+shift*n0*n0)
        # chi^1/2 ||p u|| <= sqrt(2)||Q^1/2 u||
        # The actual selected Lambda, not an asymptotic wish, pays this coefficient.
        kinetic = sqrt_upper(2*n0*q0)+n0
        rho=sqrt_upper(n0*rho2_inputs[label])
        harmonic = 2*sqrt_upper(ue/F(volume))*sqrt_upper(F(3))*k0*relative*photon_half
        harmonic += 3*ue/F(volume)*k0*relative**2*n0
        commutator = 2*k1*kinetic+dimension*k2*n0+4*(1+tau)*k0*rho+coulomb*k0*n0+harmonic
        first = k0*q0+commutator+shift*k0*n0
        graphs[label] = SourceGraph(dyadic_upper(k0*n0), dyadic_upper(first),
            dyadic_upper(sqrt_upper(k0*n0*first)),dyadic_upper(commutator),
            tuple(dyadic_upper(jet) for jet in jets))
    return dict(source_identity=source_identity, graphs=graphs, positive_shift=shift,
        rho_constant=dyadic_upper(rho0),rho_quarter_constant=dyadic_upper(rhoquarter),
        current_coefficient=dyadic_upper(current_coefficient),
        contact_coefficient=dyadic_upper(contact_coefficient),occupation_coercivity=descriptor['mu']/2,
        current_ground=current_ground, Cchart=2*budget.chart_majorant,
        source_dimension=dimension, physical_word=word, source_filter=False,
        chi_vacuum_coefficient_paid_to=1,chi_vacuum_log2_Lambda_requirement=8*vacuum_log,
        residual_positive_half_assumed=False)


def spectral_tail(graph: SourceGraph, cutoff: Fraction) -> dict[str, Fraction]:
    """For K>=0: ||1_(L,infty)(K)s||<=||Ks||/L; energy-half<=||Ks||/sqrt L."""
    if cutoff <= 0:
        raise ValueError('Positive spectral decomposition threshold required')
    return dict(norm=min(graph.norm, graph.first/cutoff),
        half=min(graph.half, sqrt_upper(graph.first**2/cutoff)),
        occupation_energy=graph.first**2/cutoff)


def source_flight_certificate(descriptor: dict[str, Any], body: dict[str, Any],
                              raw: dict[str, Any], metric: Fraction,photon: dict[str, Any]) -> dict[str, Any]:
    """A common finite original Lambda pays source tails, effective flow and observables."""
    n, volume, time = descriptor['N'], descriptor['V'], body['time']
    budget = normal.native_normal_budget(n)
    target = body['physical_family']['UE']/(volume*F(2)**40)
    sources = dict(scattering19=source_graph(descriptor,body,photon),
        RAW41=source_graph(descriptor,body,photon,raw))
    max_norm = max(graph.norm for entry in sources.values() for graph in entry['graphs'].values())
    max_first = max(graph.first for entry in sources.values() for graph in entry['graphs'].values())
    max_half = max(graph.half for entry in sources.values() for graph in entry['graphs'].values())
    jcoef = max(entry['current_coefficient'] for entry in sources.values())
    zcoef = max(entry['contact_coefficient'] for entry in sources.values())
    r0 = max(entry['rho_constant'] for entry in sources.values())
    rq = max(entry['rho_quarter_constant'] for entry in sources.values())
    # Opposite-source-leg current sandwiches, real/contact/PRE/POST/exchange,
    # then coherent GLOBAL2x2 divided differences. No current word graph is assumed.
    current_cap = jcoef*(r0*max_norm+rq*sqrt_upper(max_norm*max_half))
    contact_cap = zcoef*(r0*max_norm+rq*sqrt_upper(max_norm*max_half))
    observable_weight = metric*(1+max_norm+max_first+current_cap+contact_cap)**4*(1+time)**4
    # Norm defect z has bounded half graph <=2 max_half, not a vanishing residual half graph.
    # rho z <=r0||z||+rq sqrt(||z|| ||Khalf z||). Squaring the norm tolerance pays it.
    delta_log=max(6,2*normal.ceil_log2(128*observable_weight*(1+r0+rq+max_half)/target))
    delta=F(2)**(-delta_log)
    cutoff_log = max(1, normal.ceil_log2(16*max_first/delta),
        2*normal.ceil_log2(16*max_first/delta))
    cutoff = F(2)**cutoff_log
    source_jet = max(sum(math.comb(6, q)*graph.jets[q] for q in range(7))
        for entry in sources.values() for graph in entry['graphs'].values())
    # Photon creation sees the H_EM-half ground mismatch, and a cloud sees
    # sqrt(||z||) through Jz. Pay these ROOTS explicitly rather than identifying
    # a small Hamiltonian residual with a small positive-half residual.
    low_graph_multiplier = dyadic_upper(metric*(1+source_jet)*(1+max_first+max_half)*(1+1/descriptor['omega_min'])**3)
    trial = dyadic_upper(normal.neutral_trial_upper(budget,body['physical_family']['UE'],body['physical_family']['K']))
    light_slope = dyadic_upper((8*volume+2)*sqrt_upper((F(1,5)+6*F(3,25))**2+3*F(3,10)**2))
    deep_log=8*normal.ceil_log2(128*low_graph_multiplier*(1+trial+light_slope+r0+r0*r0)/delta)
    # Ask the frozen normal supplier for its ordinary exact certificate, then
    # strengthen its power inequalities by INTEGER logarithms. Passing a
    # multi-million-digit decimal tolerance to its display API is unnecessary.
    choice = normal.choose_lambda(budget, target, time=time,
        requested_gap=body['positive_energy_shift']+16,
        source_multiplier=metric, phase=scalar.phase_bounds(n))
    dlog=deep_log+normal.ceil_log2(4*(1+time)*(1+max_norm)*low_graph_multiplier)
    d = F(2)**(-dlog)
    gap=8*cutoff
    neutral_tail = d/gap
    moment2 = r0*r0
    photon_tail_form = d*d/gap+(trial+d)*neutral_tail**2
    photon_tail_form += light_slope*sqrt_upper(moment2)*sqrt_upper(neutral_tail**3)
    birth_low_error = sqrt_upper(neutral_tail**2+photon_tail_form/descriptor['omega_min'])
    current_low_error = jcoef*sqrt_upper(moment2*neutral_tail)+F(2)**(-deep_log)
    cloud_low_error = (current_low_error+d*(1+max_first))/descriptor['omega_min']**2
    # u=W*Omega is not normalized: psi=u/||u||. Photon creation/cloud must
    # also pay the actual scalar normalization difference, not only Pperp Omega.
    birth_low_error += 2*neutral_tail**2*(max_norm+birth_low_error)
    cloud_low_error += 2*neutral_tail**2*(1+max_norm+max_first)/descriptor['omega_min']**2
    cy, orbital = F(2)**64, F(2)**50
    boundary_good1 = (source_jet+1)*cy*2*budget.chart_majorant
    boundary_good2 = (source_jet+1)*cy**3*2*budget.chart_majorant
    value_coefficient = max(source_jet*2*budget.chart_majorant*(budget.soft_moment+budget.stiff_moment)*(1+max_first),
        boundary_good1,boundary_good2,orbital*(1+max_first))
    value_log = 5*normal.ceil_log2(64*value_coefficient*low_graph_multiplier/delta)
    extra_normal_log = max(16*(normal.ceil_log2(budget.residual_kinetic)+dlog+3),
        (16*(normal.ceil_log2(budget.residual_light)+dlog+3)+4)//5,
        16*(budget.cutoff_log2+dlog+3),
        (16*(normal.ceil_log2(budget.current_lipschitz*budget.soft_moment)+deep_log+3)+4)//5,
        4*(cutoff_log+normal.ceil_log2(128/budget.frequency)))
    required_log=max(180375,value_log,extra_normal_log)
    # The ALREADY SELECTED global exponent contains 640*2^(2^height).
    # Since 2^(2^height)>=height, this elementary finite lower floor proves
    # the stronger source requests without changing k, ell, Lambda or chi.
    global_exponent=choice['global_log2_Lambda_expression']
    existing_tower_floor=640*global_exponent['coercivity_reciprocal_inner_exponent']
    if max(choice['local_log2_Lambda'],existing_tower_floor)<required_log:
        raise AssertionError('Selected scalar-phase Lambda does not cover source strengthening')
    choice['source_strengthening'] = dict(residual_log2_upper=-dlog,
        requested_spectral_cutoff_log2=cutoff_log,extra_log2_Lambda_requirement=extra_normal_log,
        existing_global_log2_floor=existing_tower_floor,selected_Lambda_unchanged=True,
        unchanged_original_kinetic=True)
    source_value_error = delta/(64*low_graph_multiplier)
    boundary_bad = source_jet*cy**2*(1+moment2)*neutral_tail
    physical_boundary_error = 2*source_value_error+boundary_bad+orbital*d*(1+max_first)
    if physical_boundary_error > delta/8:
        raise AssertionError('Literal source Lie/orbital/scalar boundary exceeds original-Phi budget')
    initialization_error = low_graph_multiplier*(birth_low_error+cloud_low_error+source_value_error)
    if initialization_error > delta/8:
        raise AssertionError('Actual photon/source low-W mismatch exceeds its ROOT-paid budget')
    # All these are proved envelopes at the displayed SAME finite Lambda. The huge
    # scalar-phase exponent is never converted to a floating number or materialized.
    error = initialization_error+time*d*low_graph_multiplier+delta/8
    current_error = jcoef*(r0*error+rq*sqrt_upper(error*2*max_half))
    contact_error = zcoef*(r0*error+rq*sqrt_upper(error*2*max_half))
    occupation_error = 16*volume*(2*max_norm+error)*error
    full_word_error = metric*(1+time)**4*(1+max_norm+current_cap+contact_cap)**3*(
        error+current_error+contact_error+occupation_error+physical_boundary_error)
    if full_word_error >= target:
        raise AssertionError('Actual positive graph multipliers exceed the charged-source transfer target')
    for entry in sources.values():
        entry['tails'] = {name:spectral_tail(graph, cutoff) for name,graph in entry['graphs'].items()}
        entry['low_spectral_normal_complement_upper'] = d*(1+max_first)/(gap-cutoff)
        entry['true_high_normal_source_norm_upper'] = entry['low_spectral_normal_complement_upper']+max_first/cutoff
        entry['physical_Lie_orbital_scalar_boundary_error_upper'] = physical_boundary_error
        entry['physical_CY_upper'] = cy
        entry['profile_orbital_q6_upper'] = orbital
        entry['source_norm_flight_error_upper'] = error
        entry['source_current_flight_error_upper'] = current_error
        entry['source_contact_flight_error_upper'] = contact_error
        entry['source_occupation_expectation_error_upper'] = occupation_error
        entry['positive_light_occupation_energy_coercivity'] = dict(
            gap=descriptor['mu']/2,
            neutral_excitation_density=body['quadratic_occupation']['neutral_excitation_density_upper'],
            full_source_occupation_upper=min(4*volume*max_norm**2,
                2*(max_first*max_norm+(trial+body['positive_energy_shift'])*max_norm**2)/descriptor['mu'])+occupation_error,
            scalar_paid_positive_light_trial=trial,
            bad_tube_occupation_uses_bounded_full_CAR=True,
            effective_Gamma_ground_uniqueness_used=False)
    return dict(N=n,V=volume,time=time,target=target,metric_multiplier=metric,sources=sources,
        exact_high_source_retained=True,high_source_spectral_cutoff_log2=cutoff_log,
        high_source_energy_invariant=True,hardware_filter=False,normal_certificate=choice,
        source_value_log2_requirement=max(180375,value_log),Cchart='2J',
        low_graph_multiplier=low_graph_multiplier,normal_norm_tolerance=delta,
        actual_neutral_tail_norm=neutral_tail,actual_photon_tail_form=photon_tail_form,
        actual_birth_low_W_error=birth_low_error,actual_cloud_low_W_error=cloud_low_error,
        source_initialization_error=initialization_error,normal_residual_norm_upper=d,
        low_W_is_quasimode=True,RB_U_positive_half_inferred=False,
        source_current_and_occupation_transfer_upper=full_word_error,
        occupied_light_coercivity_lower=descriptor['mu']/2,
        canonical_neutral_momentum_cancels_operatorially=True,
        unbounded_neutral_Pi_variance_used=False,
        fibre_scope='uniform real shift; paired kappa/-kappa normalized envelope from Photon supplier',
        selected_full_volume_simulated=False)


def _dense(model: Coupled) -> np.ndarray:
    eye = np.eye(model.size,dtype=complex)
    return np.column_stack([model.action(column) for column in eye.T])


def native_source_consumer(stiffness: float = 256., time: float = 6.) -> dict[str, Any]:
    """Actual radial Phi + nonlinear coexact Peierls + harmonic + entire light CAR slice.

    This is an independent finite original-law Galerkin identity consumer. It does
    not simulate the selected 60V Cartesian problem or certify its global phase.
    """
    band = RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    _,frame = original_frame(band,5,np.array([1,2,3])*math.pi/2,True)
    ue,k = .015, .53
    neutral = Coupled(2,1,.17,4,ue,k,frame)
    charged = Coupled(2,1,.17,5,ue,k,frame)
    h0,h1 = _dense(neutral),_dense(charged)
    # Homogeneous radial slice of the ACTUAL quartic ||Tr Phi_mu Phi_nu/4||².
    # Heavy occupied energy is exactly linear for positive radial scale. Keeping
    # it gives a cubic/quartic radial well; no substituted harmonic potential.
    grams = np.einsum('mij,nji->mn',frame,frame).real/4
    a4 = 2*float(np.sum(grams*grams))
    onsite = np.kron(np.eye(2),(.2+3*.12)*frame[0])
    onebody_hops=[]
    heavy_onebody=onsite.copy()
    for axis in range(3):
        hop=np.zeros((8,8),complex)
        hop[:4,4:]=-(.12*frame[0]+1j*.3*frame[axis+1])/2
        onebody_hops.append(hop)
        heavy_onebody += hop+hop.conj().T
    heavy_values = np.linalg.eigvalsh(heavy_onebody)
    heavy_linear = -float(np.sum(heavy_values[heavy_values>0]))
    centre = (-heavy_linear/(4*a4))**(1/3)
    radial_dimension = 5
    ladder = np.diag(np.sqrt(np.arange(1,radial_dimension)),1)
    x = (ladder+ladder.T)/math.sqrt(2)
    p = -1j*(ladder-ladder.T)/math.sqrt(2)
    grid, rotation = eigh(x)
    width = stiffness**(-3/8)/3
    radial = centre+width*grid
    if radial.min() <= 0:
        raise AssertionError('Positive radial heavy-cluster slice required')
    p2 = rotation.conj().T@(p@p)@rotation
    chi = stiffness**(-1/2)
    radial_h = chi/(2*width*width)*p2+np.diag(stiffness*(a4*radial**4+heavy_linear*radial))
    radial_values,radial_vectors = eigh(radial_h)
    radial_h -= radial_values[0]*np.eye(radial_dimension)
    normal_ground = radial_vectors[:,0]
    matrices=[]
    for model,h in ((neutral,h0),(charged,h1)):
        diagonal = np.diag((model.phenergy[:,None]+model.coulomb[None,:]).ravel())
        matrices.append(np.kron(radial_h,np.eye(model.size))+
            np.kron(np.eye(radial_dimension),diagonal)+np.kron(np.diag(radial),h-diagonal))
    hn,hc = matrices
    solver=cast(Callable[...,Any],eigsh)
    en,on = solver(hn,k=1,which='SA',tol=1e-11,v0=np.random.default_rng(222).normal(size=hn.shape[0]))
    e0 = float(en[0]);omega = on[:,0]
    ground_error = accept(np.linalg.norm(hn@omega-e0*omega),'full radial/coexact/harmonic/CAR eigenpair',2e-8)
    ec,vc = eigh(hc)
    shift = max(1.,e0-float(ec[0])+1.)
    positive = ec-e0+shift
    kc = hc+(shift-e0)*np.eye(hc.shape[0])
    w = np.kron(normal_ground[:,None],np.eye(charged.size))
    heff = w.conj().T@hc@w
    ee,ve = eigh(heff)
    residual = hc@w-w@heff
    current_weights = np.array([1.,-1.,0.])/math.sqrt(2*neutral.omega)
    jn0 = np.column_stack([neutral.current(column,current_weights) for column in np.eye(neutral.size).T])
    jc0 = np.column_stack([charged.current(column,current_weights) for column in np.eye(charged.size).T])
    jn,jc = np.kron(np.diag(radial),jn0),np.kron(np.diag(radial),jc0)
    an = np.kron(np.diag(np.sqrt(np.arange(1,neutral.n)),1),np.eye(neutral.l*neutral.d))
    birth = np.kron(np.eye(radial_dimension),an.conj().T)@omega
    uvalues,uvectors = eigh(hn)
    cloud = uvectors@((uvectors.conj().T@(jn@omega))/(uvalues-e0+neutral.omega))
    qn = hn-e0*np.eye(hn.shape[0])
    cloud_graph_error = accept(np.linalg.norm((qn+neutral.omega*np.eye(hn.shape[0]))@cloud-jn@omega),'actual neutral photon resolvent cloud',2e-8)
    cc = creators(neutral,charged)
    # B_t uses the ACTUAL harmonic one-body flow and Wilson phase, not a
    # time-independent source copy. FFT coefficients are compressed only after
    # constructing the smooth map; all four opposite-reference columns remain.
    def source_at(at: float) -> np.ndarray:
        samples=64
        marks=[]
        seed=np.zeros((8,4),complex)
        alpha=np.array([1.,.3j,-.2+.4j,.6])
        alpha/=np.linalg.norm(alpha)
        seed[:4]=sum(alpha[q]*([np.eye(4)]+[SELECTED[0]@SELECTED[j] for j in range(1,4)])[q]
            for q in range(4))/math.sqrt(2)
        for angle in 2*math.pi*np.arange(samples)/samples:
            onebody=onsite.copy()
            for hop,nu in zip(onebody_hops,(1,1,-2)):
                phase=np.exp(1j*nu*angle)
                onebody += phase*hop+phase.conjugate()*hop.conj().T
            values,vectors=eigh(centre*onebody)
            mark=((vectors*np.exp(-1j*at*values))@vectors.conj().T)@seed*np.exp(-1j*angle)
            marks.append(np.einsum('ir,icd->rcd',mark,cc))
        coefficients=np.fft.fft(np.asarray(marks),axis=0)/samples
        rows=np.zeros((4,charged.size,neutral.size),complex)
        for order in range(-neutral.l+1,neutral.l):
            shift_h=np.zeros((neutral.l,neutral.l))
            for column in range(neutral.l):
                if 0<=column+order<neutral.l:
                    shift_h[column+order,column]=1
            for ref in range(4):
                rows[ref] += np.kron(np.eye(neutral.n),np.kron(shift_h,coefficients[order%samples,ref]))
        return np.array([np.kron(np.eye(radial_dimension),row) for row in rows])
    source_full = source_at(0.)
    source_last = source_at(time)
    threshold = float(np.quantile(positive,.2))
    high = positive>threshold
    def flow(rows: np.ndarray) -> np.ndarray:
        return (vc@(np.exp(-1j*time*ec)[:,None]*(vc.conj().T@rows.T))).T
    def expectation(rows: np.ndarray, operator: np.ndarray) -> float:
        return float(sum(np.vdot(row,operator@row).real for row in rows))
    occupation_samples=[]
    occupied_gaps=[]
    for angle in 2*math.pi*np.arange(64)/64:
        onebody=onsite.copy()
        for hop,nu in zip(onebody_hops,(1,1,-2)):
            phase=np.exp(1j*nu*angle)
            onebody += phase*hop+phase.conjugate()*hop.conj().T
        light_values,light_vectors=eigh(centre*onebody)
        ppositive=light_vectors[:,light_values>0]@light_vectors[:,light_values>0].conj().T
        occupation_samples.append(exterior(ppositive,charged.rows))
        occupied_gaps.append(float(np.min(light_values[light_values>0])))
    occupation_coefficients=np.fft.fft(np.asarray(occupation_samples),axis=0)/64
    occupation_base=np.zeros((charged.size,charged.size),complex)
    for order in range(-charged.l+1,charged.l):
        shift_h=np.zeros((charged.l,charged.l))
        for column in range(charged.l):
            if 0<=column+order<charged.l:
                shift_h[column+order,column]=1
        occupation_base += np.kron(np.eye(charged.n),np.kron(shift_h,occupation_coefficients[order%64]))
    number=np.kron(np.eye(radial_dimension),occupation_base)
    records=[]
    for label,input_state in (('birth',birth),('cloud',cloud)):
        prepared = source_full@input_state
        coefficients = prepared@vc.conj()
        low = (vc[:,~high]@coefficients[:,~high].T).T
        tail = (vc[:,high]@coefficients[:,high].T).T
        actual = flow(prepared)
        low_out,high_out = flow(low),flow(tail)
        identity = accept(np.linalg.norm(actual-low_out-high_out),'unfiltered true low/high source flight',2e-9)
        graph_first = float(np.linalg.norm(prepared@kc.T))
        tail_norm = float(np.linalg.norm(tail))
        tail_half = float(np.linalg.norm(coefficients[:,high]*np.sqrt(positive[high])[None,:]))
        accept(max(0.,tail_norm-graph_first/threshold),'SOURCE first graph pays high norm',2e-9)
        accept(max(0.,tail_half-graph_first/math.sqrt(threshold)),'SOURCE first graph pays high half energy',2e-9)
        invariance = accept(abs(np.linalg.norm(high_out)-tail_norm),'retained high source norm invariant',2e-9)
        effective_initial = prepared@w.conj()
        normal_low=effective_initial@w.T
        normal_high=prepared-normal_low
        true_normal_identity=accept(np.linalg.norm(actual-flow(normal_low)-flow(normal_high)),
            'true W/complement unfiltered source decomposition',2e-9)
        normal_high_norm=float(np.linalg.norm(normal_high))
        normal_high_half=float(np.linalg.norm((normal_high@vc.conj())*np.sqrt(positive)[None,:]))
        effective = (ve@(np.exp(-1j*time*ee)[:,None]*(ve.conj().T@effective_initial.T))).T
        lifted = effective@w.T
        difference = float(np.linalg.norm(actual-lifted))
        initial_error = float(np.linalg.norm(prepared-effective_initial@w.T))
        # Uniform restricted Duhamel norm bound computed on the ACTUAL residual,
        # not a guessed half graph or a copied operator expectation.
        residual_cap = float(np.linalg.norm(residual,2))
        duhamel = initial_error+time*residual_cap*float(np.linalg.norm(effective_initial))
        accept(max(0.,difference-duhamel),'low-W source Duhamel envelope',2e-8)
        missing_high = adverse(np.linalg.norm(actual-low_out),'unfiltered true low/high source flight',2e-9)
        missing_normal = adverse(np.linalg.norm(actual-lifted),'full radial source evolution',2e-9)
        boundary = np.array([jc@row-row@jn for row in source_full])
        actual_boundary = boundary@input_state
        wrong_boundary = adverse(np.linalg.norm(actual_boundary),'same physical source/current boundary',2e-9)
        commutator = np.array([hc@row-row@hn for row in source_full])@input_state
        source_graph_identity = accept(np.linalg.norm(prepared@kc.T-(source_full@(qn@input_state)+commutator+shift*prepared)),
            'actual SOURCE graph product rule',2e-8)
        last_prepared=source_last@input_state
        last_first=float(np.linalg.norm(last_prepared@kc.T))
        records.append(dict(source=label,source_reference_count=4,input_norm=float(np.linalg.norm(input_state)),
            source_norm=float(np.linalg.norm(prepared)),source_first_graph=graph_first,
            source_graph_identity_error=source_graph_identity,high_source_norm=tail_norm,
            high_source_half_energy=tail_half,high_source_invariance_error=invariance,
            decomposition_error=identity,actual_source_current=expectation(actual,jc),
            effective_source_current=expectation(lifted,jc),actual_source_occupation=expectation(actual,number),
            effective_source_occupation=expectation(lifted,number),low_W_norm_error=difference,
            actual_advected_source_norm=float(np.linalg.norm(last_prepared)),
            actual_advected_source_first_graph=last_first,
            advected_source_sha256=hashlib.sha256(last_prepared.tobytes()).hexdigest(),
            true_normal_decomposition_error=true_normal_identity,
            true_high_normal_source_norm=normal_high_norm,
            true_high_normal_source_half_energy=normal_high_half,
            low_W_Duhamel_upper=duhamel,current_boundary_norm=float(np.linalg.norm(actual_boundary)),
            wrong_missing_high_rejected=missing_high,wrong_missing_normal_rejected=missing_normal,
            wrong_missing_current_boundary_rejected=wrong_boundary,
            source_sha256=hashlib.sha256(prepared.tobytes()).hexdigest()))
    return dict(stiffness=stiffness,chi=chi,time=time,source_filter=False,radial_dimension=radial_dimension,
        radial_grid=radial.tolist(),radial_potential='original quartic Gram plus exact homogeneous heavy Slater energy',
        neutral_dimension=hn.shape[0],charged_dimension=hc.shape[0],ground_error=ground_error,
        cloud_graph_error=cloud_graph_error,normal_residual_norm=float(np.linalg.norm(residual)),
        actual_current_nonlinear_Peierls=True,actual_harmonic_and_Coulomb_retained=True,
        sampled_positive_light_band_minimum=min(occupied_gaps),
        native_scope='finite homogeneous radial-Phi slice; not selected full Cartesian phase simulation',
        records=records,ground_sha256=hashlib.sha256(omega.tobytes()).hexdigest())


def encode(value: Any) -> Any:
    if isinstance(value,F):
        exponent=normal.ceil_log2(value) if value>0 else None
        if exponent is None:
            return dict(log2_upper=None,decimal_upper='0')
        with localcontext() as context:
            context.prec=70
            context.rounding=ROUND_CEILING
            context.Emax=999999999
            context.Emin=-999999999
            upper=str(Decimal(2)**exponent)
        result=dict(log2_upper=exponent,dyadic_upper=f'2^({exponent})',decimal_upper=upper)
        if max(value.numerator.bit_length(),value.denominator.bit_length())<=12000:
            result['exact_rational']=str(value)
        return result
    if isinstance(value,SourceGraph):
        return dict(norm=encode(value.norm),first=encode(value.first),half=encode(value.half),
            commutator=encode(value.commutator),jets=[encode(jet) for jet in value.jets])
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor',type=Path)
    parser.add_argument('--raw-receipt',type=Path)
    parser.add_argument('--normal-packet',type=Path)
    parser.add_argument('--photon-packet',type=Path)
    parser.add_argument('--photon-receipt',type=Path)
    parser.add_argument('--metric',default='3.61861435924626937868268721417267178786515019177895711431167e859')
    parser.add_argument('--time',default='6')
    parser.add_argument('--native-consumer',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    sys.set_int_max_str_digits(0)
    result: dict[str, Any] = dict(plan=PLAN,proof_pair='noncompact_original_source_flight.py/.md')
    inputs = [path for path in (args.descriptor,args.raw_receipt,args.normal_packet,
        args.photon_packet,args.photon_receipt) if path]
    if args.descriptor:
        if not args.raw_receipt or not args.normal_packet or not args.photon_packet or not args.photon_receipt:
            parser.error('Selected flight requires distinct RAW41, corrected Normal, and actual Photon packet/receipt')
        packet = json.loads(args.normal_packet.read_text())
        if packet['normal_capability']['source_value_log2_Lambda_requirement'] != 180375:
            raise ValueError('Superseded source chart receipt: require Cchart2J/log2Lambda180375')
        descriptor = compact_descriptor(json.loads(args.descriptor.read_text()))
        body = normalized_flight_bound(descriptor,F(args.time))
        raw = json.loads(args.raw_receipt.read_text())
        photon_packet=json.loads(args.photon_packet.read_text())
        photon_bytes=args.photon_receipt.read_bytes()
        if hashlib.sha256(photon_bytes).hexdigest()!=photon_packet['receipt_sha256']:
            raise ValueError('Actual photon receipt differs from its stable supplier packet')
        photon=json.loads(photon_bytes)
        pselected=photon['selected']
        if (pselected['N']!=descriptor['N'] or F(pselected['UE'])!=body['physical_family']['UE']
                or F(pselected['K'])!=body['physical_family']['K']
                or pselected['selected_time']!=body['time']):
            raise ValueError('Photon supplier must use the SAME actual microscopic source family/history')
        result['selected_source_flight'] = source_flight_certificate(descriptor,body,raw,F(args.metric),photon)
        phase=body['physical_family']['UE']/descriptor['V']*dyadic_value(pselected['packet']['T6_phase_rescaled_error_upper'])
        combined=result['selected_source_flight']['source_current_and_occupation_transfer_upper']+phase
        if combined>=result['selected_source_flight']['target']:
            raise AssertionError('Actual source/current/occupation plus paired-Bloch phase exceeds far target')
        result['paired_photon_source_join']=dict(physical_phase_error_upper=phase,
            combined_transfer_upper=combined,actual_normalized_packet=pselected['packet'],
            original_two_fixed_outputs_and_GLOBAL2x2=photon_packet['new_observed_finite_consumer']['GLOBAL_whitening_after_fibre_integration'])
    if args.native_consumer:
        result['native_consumer'] = native_source_consumer(time=float(args.time))
    if not args.descriptor and not args.native_consumer:
        parser.error('Exercise the selected quantitative path or native source consumer')
    source_paths={Path(__file__).resolve(),ROOT/'noncompact_original_source_flight.md'}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.parent==ROOT and path.suffix=='.py':
                source_paths.add(path)
    result['source_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    result['input_sha256'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    args.output.write_text(json.dumps(result,default=encode,indent=2,allow_nan=False)+'\n')
    selected = result.get('selected_source_flight')
    print(json.dumps(dict(native_consumer_exercised=args.native_consumer,
        selected_source_transfer_exercised=selected is not None,
        selected_target_met=selected is not None and selected['source_current_and_occupation_transfer_upper']<selected['target'],
        output=str(args.output))))


if __name__ == '__main__':
    main()
