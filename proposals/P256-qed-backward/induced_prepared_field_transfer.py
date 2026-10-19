#!/usr/bin/env python3
"""NEW induced law: exact all-flavour sea-fibre electric leakage and local photon.

This is a prepared finite-field-domain comparison, never an equilibrium vacuum.
The native coupled-coordinate calculation keeps the full determinant Fock Gram;
its omitted particle-hole electric component is measured, not set to zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from itertools import combinations
from fractions import Fraction
import platform
import sys
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ('induced_prepared_field_transfer.py', 'induced_photon_response.py',
                'improved_phi_band.py', 'rotation_band.py', 'sea_selection.py',
                'unrestricted_field_minimum.py', 'induced_root_transfer.py',
                'noncompact_quantitative_normal_budget.py')
LOADED_HASHES = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
SOURCE_SNAPSHOT = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}

import numpy as np
import scipy
from scipy.linalg import eigh, expm
from scipy.optimize import brentq
from scipy.special import beta
from induced_photon_response import ResponseConfig, SpectralResponse, make_band
from unrestricted_field_minimum import local_potential
from improved_phi_band import ImprovedPhiLaw
from rotation_band import RotationBand
from induced_root_transfer import renewed_bounds
from noncompact_quantitative_normal_budget import ceil_log2

PLAN = 'issue-222-induced-photon-continuation-20261008'
ASSIGNMENT_REVISION = 'f43102786fc92abe69dd80ebcd9c28b290787094276bf5b801c29fe66140dee2'


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ArithmeticError(message)


def real(value: Any) -> float:
    return float(np.real(value))


def packet_moments(radius: float) -> dict[str, float]:
    """Normalized C1 compact packet (1-z^2)^2, extended by zero in H2(R)."""
    norm = float(beta(.5, 5))
    d1 = 16*float(beta(1.5, 3))/norm
    d2 = (16*2-96*2/3+144*2/5)/norm
    return dict(radius=radius, normalization_squared=1/(radius*norm),
                coordinate_squared=radius**2*float(beta(1.5, 5))/norm,
                gradient_squared=d1/radius**2, laplacian_squared=d2/radius**4,
                boundary_value=0., boundary_derivative=0.)


def gaussian_quantum(mass: np.ndarray, position: np.ndarray) -> dict[str, Any]:
    """Exact normalized canonical joint oscillator functional, not physical CAR."""
    pv, pu = np.linalg.eigh(position)
    check(float(pv.min()) > 0, 'Positive joint quantum position Hessian')
    root = pu@np.diag(np.sqrt(pv))@pu.T
    inverse = pu@np.diag(1/np.sqrt(pv))@pu.T
    lam, vectors = np.linalg.eigh(root@mass@root)
    check(float(lam.min()) > 0, 'Positive joint quantum kinetic Hessian')
    xcov = .5*inverse@(vectors@np.diag(np.sqrt(lam))@vectors.T)@inverse
    pcov = .25*np.linalg.inv(xcov)
    return dict(omega=np.sqrt(lam), xcov=xcov, pcov=pcov,
                energy=.5*float(np.sqrt(lam).sum()), precision=np.linalg.inv(2*xcov))


def bin_channels(delta: np.ndarray, weight: np.ndarray, bins: int):
    keep = weight > float(np.max(weight))*1e-14
    d, w = delta[keep], weight[keep]
    if not len(w):
        return np.array([]), np.array([]), 0., 0., np.full(len(delta),-1,dtype=int)
    edges = np.geomspace(float(d.min())*(1-1e-12), float(d.max())*(1+1e-12), bins+1)
    index = np.clip(np.searchsorted(edges, d)-1, 0, bins-1)
    total = np.bincount(index, weights=w, minlength=bins)
    # Preserve actual raw current strength AND the static bubble. A harmonic
    # gap mean supplies these two moments without changing contact. Dynamic
    # electric/pole compression errors remain measured, not matched away.
    moment = np.bincount(index, weights=w/d, minlength=bins)
    present = total > 1e-20
    effective = total[present]/moment[present]
    # Measured actual finite-grid spectral spread, not fitted pole precision.
    representative = np.ones(bins)
    representative[present] = effective
    spread = float(np.sqrt(np.sum(w*(d-representative[index])**2)/np.sum(w)))
    participation = float(np.sum(w)**2/np.sum(w*w))
    assignment = np.full(len(delta),-1,dtype=int)
    assignment[keep] = np.cumsum(present)[index]-1
    return effective, total[present], spread, participation, assignment


def centre_sector(size: int, flavors: int) -> dict[str, Any]:
    """Exact global centre character of the selected fixed-number seas."""
    volume = size**3
    occupied = 2*volume*(flavors+1)+4
    character = sum((1j)**((k*occupied) % 4) for k in range(4))/4
    return dict(volume=volume, heavy_particles=2*volume, charged_particles=2*volume*flavors,
        Ref_particles=4, total_particles=occupied, total_mod4=occupied % 4,
        exact_centre_projection=float(character.real), positive=character == 1,
        constraint='global SU4 centre fixes adjoint Phi and every link; Haar projection vanishes unless total fundamental number is 0 mod4')


def heavy_pair_control(config: ResponseConfig, size: int) -> dict[str, Any]:
    """Actual fixed-NH2V nearest-pair Wilson trial and centre controls."""
    _, _, law = make_band(config)
    generators = np.asarray(law.generators)
    pairs = tuple(combinations(range(4),2))
    basis = np.zeros((16,6),complex)
    epsilon = np.zeros((6,6))
    for a,(i,j) in enumerate(pairs):
        basis[4*i+j,a],basis[4*j+i,a] = 1/math.sqrt(2),-1/math.sqrt(2)
        for b,(k,l) in enumerate(pairs):
            if len({i,j,k,l}) == 4:
                order=(i,j,k,l)
                epsilon[a,b]=(-1)**sum(order[u]>order[v] for u,v in combinations(range(4),2))
    def wedge(u):
        return np.array([[u[i,k]*u[j,l]-u[i,l]*u[j,k] for k,l in pairs] for i,j in pairs])
    reps=np.array([basis.conj().T@(np.kron(t,np.eye(4))+np.kron(np.eye(4),t))@basis for t in generators])
    casimir=np.sum(reps@reps,axis=0)
    link=expm(1j*(.31*generators[0]-.19*generators[5]))
    amplitude=wedge(link)@epsilon/math.sqrt(6)
    step=1e-5
    errors=[]
    for t,rep in zip(generators,reps):
        left=(wedge(expm(1j*step*t)@link)-wedge(expm(-1j*step*t)@link))@epsilon/(2*step*math.sqrt(6))
        right=(wedge(link@expm(-1j*step*t))-wedge(link@expm(1j*step*t)))@epsilon/(2*step*math.sqrt(6))
        errors.extend((float(np.linalg.norm(left-1j*rep@amplitude)),
                       float(np.linalg.norm(right-1j*amplitude@rep.T))))
    positive=centre_sector(size,config.flavors)
    adverse=centre_sector(3,config.flavors)
    check(positive['positive'], 'Filled-heavy physical state requires centre-admissible volume')
    check(not adverse['positive'], 'Odd-volume filled65-species centre projection must vanish')
    casimir_error=float(np.linalg.norm(casimir-2.5*np.eye(6)))
    marginal_error=abs(float(np.linalg.norm(amplitude)**2)-1)
    check(max(errors)<1e-8 and casimir_error<1e-12 and marginal_error<1e-12,
          'Actual exterior2 heavy Wilson Gauss state/Casimir/Haar marginal')
    return dict(centre=positive, odd_volume_adverse=adverse, local_Gauss_derivative_error=max(errors),
        pair_amplitude='(exterior2 Uxy) epsilon / sqrt6, on disjoint nearest pairs in even x direction',
        pair_normalization_error=marginal_error, Casimir=2.5,Casimir_error=casimir_error,
        fixed_particles_each_site=2, heavy_total_particles=2*size**3,
        independent_Haar_marginal_proof='Tr amplitude^dagger amplitude=1 pointwise; all link probabilities remain product Haar',
        heavy_hopping_mean=0, Phi_matter_mean=0,
        omitted_Wilson_Gauss_defect=math.sqrt(float(np.trace(amplitude.conj().T@casimir@amplitude).real)),
        added_original_colour_kinetic='(3/8) volume Lambda^(-1/2)',
        LAW05_colour_coefficient='.3 chi, flat colour cometric .6 chi',
        half_colour_control_relative_defect=.5,
        scope='fixed NH2V sector trial, NOT the filled-heavy BO Slater or its energy')


def joint_collective(config: ResponseConfig, size: int, bins: int) -> dict[str, Any]:
    """Prepared PH-enriched transverse + ALL longitudinal Coulomb Gaussians.

    The finite microscopic CAR state is the explicitly defined degree-K
    collective polynomial image, not an asserted bosonic vacuum. Its
    normalization is the positive exact Fock Gram integral; zero-PH amplitude
    supplies a calculable positive lower. Finite-N and nonlinear action
    bounds below determine whether the preparation transfers dynamically.
    """
    sector=centre_sector(size,config.flavors)
    check(sector['positive'], 'Collective fullGauss preparation needs nonzero global centre projection')
    V = size**3
    integers = np.asarray([(x,y,z) for x in range(size) for y in range(size) for z in range(size)])
    pin = 2*math.pi*integers/size
    groups = config.species or ((config.flavors, config.m),)
    models = []
    from dataclasses import replace
    for count, mass_value in groups:
        model = SpectralResponse(replace(config, m=mass_value, flavors=count, species=()), 2)
        em, pm, um = model.projectors(pin)
        models.append((count, model, em, pm, um))
    qt = np.array([2*math.pi/size, 0., 0.])
    delta, weight = [], []
    contact, ward, participation_denominator, mixed_sum = 0., 0., 0., 0j
    for count, model, em, pm, um in models:
        ep, pp, up = model.projectors(pin+qt)
        j, _ = model.vertices(pin+qt/2, qt)
        _, c = model.vertices(pin, qt)
        # Standing cos has half the plane-wave spectral norm.
        g = np.trace(pm@j[:,1]@up@j[:,1]+um@j[:,1]@pp@j[:,1], axis1=1, axis2=2).real
        delta.extend((em+ep).tolist())
        weight.extend((count*g/2).tolist())
        participation_denominator += float(np.sum((count*g/2)**2))/count
        mixed_sum += count*complex(np.trace(pm@j[:,1]@up,axis1=1,axis2=2).sum())
        contact += count/2*float(np.trace(pm@c[:,1], axis1=1, axis2=2).real.sum())
        ward = max(ward, float(np.max(np.abs(2*math.sin(qt[0]/2)*j[:,0]-(model.h(pin+qt)-model.h(pin))))))
    d, w = np.asarray(delta), np.asarray(weight)
    stiffness = contact-float(np.sum(w/d))
    check(stiffness > 0, 'Actual native IR coexact stiffness')
    epsbare = config.a*V/(4*config.inertia)
    bright = w > float(w.max())*1e-12
    threshold = float(d[bright].min())
    def inverse(x):
        return epsbare*x-stiffness+float(np.sum(w*x/(d*(d*d-x))))
    root = cast(float, brentq(inverse, 0., threshold**2*(1-1e-10), xtol=1e-25, full_output=False))
    exact_omega = math.sqrt(root)
    exact_residue = 1/(epsbare+float(np.sum(w*d/(d*d-root)**2)))
    effective, total, spread, participation, _ = bin_channels(d, w, bins)
    flavour_participation = float(np.sum(w)**2/participation_denominator)
    check(abs(mixed_sum) < 1e-8, 'Actual axisaligned transverse/density Schur parity')
    # Original microscopic current chart: no sea-coordinate unitary silently
    # omitted from J_CAR. Equivalent shifted-momentum form requires that
    # unitary on the physical state, so use the unshifted positive Hessian.
    kinetic = np.diag(np.r_[1/epsbare, effective])
    position = np.diag(np.r_[contact, effective])
    position[0,1:] = position[1:,0] = np.sqrt(total)
    transverse = gaussian_quantum(kinetic, position)
    xcov, pcov = transverse['xcov'], transverse['pcov']
    occupation_T = .5*(np.diag(xcov)[1:]+np.diag(pcov)[1:]-1)
    # All nonzero real density Fourier degrees retained, not only selected q.
    density_occupation, density_energy, density_log_norm = 0., 0., 0.
    density_spread, density_largest, minimum_density_frequency = 0., 0., math.inf
    density_modes = 0
    hole_load = [np.zeros((V,4,4),complex) for _ in models]
    for iq in integers[1:]:
        qtheta = 2*math.pi*iq/size
        dl, wl, hole_kernels = [], [], []
        for count, model, em, pm, _ in models:
            ep, _, up = model.projectors(pin+qtheta)
            g = np.trace(pm@up, axis1=1, axis2=2).real
            dl.extend((em+ep).tolist())
            # rho_cos = sqrt(N sum trace(P- P+)) x, xvac variance1/2.
            wl.extend((count*np.maximum(g,0.)).tolist())
            hole_kernels.append(pm@up@pm)
        dd, ww, sp, _, assignment = bin_channels(np.asarray(dl), np.asarray(wl), bins)
        lap = 4*float(np.sum(np.sin(qtheta/2)**2))
        density_vector = np.sqrt(ww)
        density_position = np.diag(dd)+4*config.inertia/(config.a*lap*V)*np.outer(density_vector,density_vector)
        density = gaussian_quantum(np.diag(dd), density_position)
        xx, pp = density['xcov'], density['pcov']
        occ = .5*(np.diag(xx)+np.diag(pp)-1)
        for group_index, kernel in enumerate(hole_kernels):
            ix = assignment[group_index*V:(group_index+1)*V]
            ratio = np.zeros(V)
            valid = ix >= 0
            ratio[valid] = np.maximum(occ[ix[valid]],0.)/ww[ix[valid]]
            hole_load[group_index] += ratio[:,None,None]*kernel
        density_occupation += float(occ.sum())
        density_energy += density['energy']
        density_spread = max(density_spread, sp)
        density_largest = max(density_largest, float(occ.max()))
        minimum_density_frequency = min(minimum_density_frequency, float(density['omega'].min()))
        # Gaussian vacuum probability det(Cov+I/2)^(-1/2), exact zero mean.
        density_log_norm -= .5*(float(np.linalg.slogdet(xx+.5*np.eye(len(dd)))[1])+
                                 float(np.linalg.slogdet(pp+.5*np.eye(len(dd)))[1]))
        density_modes += len(dd)
    hole_predictions = [dict(count=count,mass=mass,maximum_per_flavour_valence_occupation=float(np.linalg.eigvalsh(load).max()),
                            total_per_flavour_valence_occupation=float(np.trace(load,axis1=1,axis2=2).real.sum()))
                        for (count,mass),load in zip(groups,hole_load)]
    hole_trace = sum(row['count']*row['total_per_flavour_valence_occupation'] for row in hole_predictions)
    check(abs(hole_trace-density_occupation) < 1e-7*(1+density_occupation), 'Actual modewise density occupation accounting')
    pol_x, pol_p = xcov[1:,1:], pcov[1:,1:]
    transverse_log_norm = -.5*(float(np.linalg.slogdet(pol_x+.5*np.eye(len(effective)))[1])+
                                float(np.linalg.slogdet(pol_p+.5*np.eye(len(effective)))[1]))
    meanK = max(0., float(occupation_T.sum())+density_occupation)
    cutoff = math.ceil(8*(meanK+1))
    # Exact bilinear CAR commutator: [B_i,B_j^dagger]-delta_ij is a sum of
    # occupied/empty number matrices divided by flavour multiplicity. This
    # conservative whole degree-K bound does NOT claim independent HP spins.
    min_count = min(count for count, _ in groups)
    commutator_upper = 2*cutoff/min_count
    physical_norm_lower_log2 = math.floor((density_log_norm+transverse_log_norm)/math.log(2))-2
    sigma = math.sqrt(float(xcov[0,0]))
    # Actual range-two microscopic Peierls third jet. Uniform fixed-number
    # operator remainder is retained; no Wick cancellation presumed.
    jet3, matter_operator, sea_Wick_third_variance = 0., 0., 0.
    for count, model, _, pm, _ in models:
        jet3 += 4*V*count*2*sum(length**3*np.linalg.norm(t,2)
                               for axis,length,t in model.paths if axis == 1)
        matter_operator += 4*V*count*(abs(model.law.onsite)+2*sum(np.linalg.norm(t,2) for _,_,t in model.paths))
        # Literal third Peierls jet for cos(x)^3: +/-q and +/-3q weights.
        # This Wick identity is exact on the actual filled Slater sea only.
        third = np.zeros((V,4,4),complex)
        for axis,length,t in model.paths:
            if axis == 1:
                term = (1j*length)**3*np.exp(1j*length*pin[:,1])[:,None,None]*t
                third += term+term.conj().transpose(0,2,1)
        for harmonic,amplitude in ((-3,.125),(-1,.375),(1,.375),(3,.125)):
            _, _, upper = model.projectors(pin+harmonic*qt)
            sea_Wick_third_variance += count*amplitude**2*float(np.trace(pm@third@upper@third,axis1=1,axis2=2).real.sum())
    nonlinear_action = jet3*math.sqrt(15)*sigma**3/6
    gaussian_bin_error = spread*math.sqrt(3*(float(occupation_T.sum())+1)**2)
    whole_Coulomb_norm = config.inertia/config.a*V*(2*config.flavors+4)**2/(4*math.sin(math.pi/size)**2)
    period = 2*math.pi/exact_omega
    field_boundary_radius = 8*sigma
    electric_graph = 32*(cutoff+3)**2*(1+float(np.linalg.norm(transverse['precision'],2)))**2/epsbare
    electric_graph += packet_moments(field_boundary_radius)['laplacian_squared']/epsbare
    # The microscopic image is normalized by its actual Gram, not by a
    # bosonic norm. A small Gram lower means action can be amplified.
    # Each normalized PH creation bilinear has norm <=4V sqrt(Ntotal).
    # The complete degree-K CAR polynomial map is bounded by its term count,
    # K! and this actual fixed-number operator cap; no bosonic echo norm.
    map_log = cutoff*((4*V*config.flavors+1).bit_length()+
                       (density_modes+len(effective)+1).bit_length()+cutoff.bit_length())
    action_log2 = math.ceil(math.log2(1+matter_operator+jet3+whole_Coulomb_norm+electric_graph))+map_log-physical_norm_lower_log2//2+16
    controlled_clock = '1/2^'+str(max(0,action_log2)+8)
    return dict(config=config.__dict__, native_size=size, native_volume=V,
        measure='EXACT periodic flat pin=2pi*l/N; contact atpin, bubble atpin+q/2',
        q_physical=(qt/config.a).tolist(), current_vertex_Ward_error=ward,
        actual_static_stiffness=stiffness, pair_gap_current_support=threshold,
        actual_spectral_omega=exact_omega, actual_spectral_residue=exact_residue,
        compressed_omega=float(transverse['omega'][0]),
        pole_relative_compression_error=abs(float(transverse['omega'][0])/exact_omega-1),
        canonical_H='P^2/(2epsbare)+contact A^2/2+A sum sqrt(W_j)x_j+sum Delta_j(x_j^2+p_j^2)/2; each real densityq adds UElink rho_q^2/lambdaq',
        normalized_joint_bosonic_wave='det(Omega/pi)^(1/4) exp(-x.Omega.x/2), Omega=(2CovX)^-1',
        transverse_covariance_x=xcov, transverse_covariance_p=pcov,
        prepared_photon_one='sqrt(2omega) Q_low times jointGaussian; nonvacuum energyomega; NO Slater-only photon substitution',
        RPA_prepared_quantum_excess_energy=float(transverse['omega'][0]), transverse_PH_occupation=float(occupation_T.sum()),
        transverse_density_mixed_trace=abs(mixed_sum), actual_flavour_participation=flavour_participation,
        all_longitudinal_real_degrees=V-1, longitudinal_collective_modes=density_modes,
        longitudinal_PH_occupation=density_occupation, longitudinal_max_channel_occupation=density_largest,
        longitudinal_quantum_energy=density_energy, longitudinal_min_frequency=minimum_density_frequency,
        density_modewise_occupation_matrix_predictions=hole_predictions,
        modewise_prediction_scope='Reconstructed from joint BOSONIC Gaussian and actual finite-Bloch occupied projectors. NOT yet a certified physical-CAR tail or Gram-transfer bound.',
        density_Gauss_action='rho_cos/sin collective bilinears; eliminate E_L=gradL+rho, retain squeezed H_C in state and H; no rho=0 expectation replacement',
        actual_weight_participation=participation, collective_degree_cutoff=cutoff,
        physical_microscopic_state='W[J_CAR Pi_{K<=cutoff} jointGaussian * H2 fieldboundary * normalpacket * heavySlater * Ref4]/sqrt(exact_full_Fock_Gram_integral)',
        physical_CAR_map='For bin j, B_j^dagger=sum_{species,flavour,k,u,o inbin}(J_uo or density_uo)c_u^dagger c_o/sqrt(sum|vertex_uo|^2); J_CAR maps products of bosonic creators to these exact primitive bilinears on ALL filled valence determinants, with no independent-spin claim',
        exact_normalization=1, normalization_lower_log2=physical_norm_lower_log2,
        global_centre_sector=sector,
        normalization_proof='degree0 PH sector is orthogonal to all higher sectors and has Gaussian vacuum probability; therefore exact mapped Gram integral is strictlypositive finite',
        finite_N_CAR_commutator_upper=commutator_upper, finite_N_CAR_small=commutator_upper < .1,
        current_spectral_bin_rms=spread, density_spectral_bin_rms_max=density_spread,
        microscopic_Peierls_third_jet=jet3, actual_Gaussian_Peierls_action_upper=nonlinear_action,
        exact_baseline_sea_Wick_third_variance=sea_Wick_third_variance,
        baseline_sea_Wick_Peierls_action=math.sqrt(max(0.,sea_Wick_third_variance))*math.sqrt(15)*sigma**3/6,
        Wick_scope='Exact filled baselineSlater variance, NOT a Wick theorem for the entangled truncated J_CAR state. Dressing correction must still be controlled.',
        retained_field_boundary_radius=field_boundary_radius, bare_electric_graph_upper=electric_graph,
        actual_Gaussian_bin_action_upper=gaussian_bin_error,
        photon_period=period, unamplified_nonlinear_error_over_period=period*nonlinear_action,
        microscopic_image_H_action_upper_log2=action_log2, controlled_microscopic_clock_exact=controlled_clock,
        fullfield_resolved_photon_transfer_admitted=False,
        explicit_failure='Whole CAR cutoff/normalization and actual Peierls action bounds do not admit a photon period at this finite64 state. This is a constructed normalized PH/Gauss preparation with priced failure, NOT a microscopic photon-phase theorem.')


def adverse_field_controls(config: ResponseConfig, size: int) -> dict[str, Any]:
    """Actual Peierls large-shift, discrete Gauss and retained-boundary probes."""
    _, frame, _ = make_band(config)
    law = ImprovedPhiLaw(size, make_band(config)[0])
    phi = np.tile(frame, (law.volume,1,1,1))
    links = np.tile(np.eye(4), (law.volume,3,1,1))
    zero = np.zeros((law.volume,3))
    shifted = zero.copy()
    # Integer plaquette circulation: divergence and harmonic average both0,
    # hence a genuine coexact large shift, not an uncancelled toron/gauge mode.
    shifted[0,0] = 2*math.pi
    shifted[int(law.target[0,0]),1] = 2*math.pi
    shifted[int(law.target[0,1]),0] = -2*math.pi
    shifted[0,1] = -2*math.pi
    shift_defect = float(np.linalg.norm(law.onebody(phi,links,shifted)-law.onebody(phi,links,zero)))
    check(shift_defect < 1e-10, 'Actual independent 2pi link periodicity')
    lap = np.zeros((law.volume,law.volume))
    for x in range(law.volume):
        for k in range(3):
            y = int(law.target[x,k])
            lap[x,x] += 1; lap[y,y] += 1
            lap[x,y] -= 1; lap[y,x] -= 1
    ev, u = np.linalg.eigh(lap)
    green = (u[:,1:]/ev[1:])@u[:,1:].T
    charge = np.zeros(law.volume)
    charge[0], charge[1] = 1., -1.
    compensated = float(np.linalg.norm(lap@green@charge-charge))
    wrong = np.zeros_like(charge); wrong[0] = 1.
    uncancelled = float(np.linalg.norm(lap@green@wrong-wrong))
    check(compensated < 1e-10 and uncancelled > .01, 'Actual compensated Gauss/adverse zero mode')
    nodes = 17
    uniform = np.ones(nodes)/math.sqrt(nodes)
    dirichlet = 2*np.eye(nodes)-np.eye(nodes,k=1)-np.eye(nodes,k=-1)
    wrapped = dirichlet.copy(); wrapped[0,-1]=wrapped[-1,0]=-1
    boundary_price = real(np.vdot(uniform,(dirichlet-wrapped)@uniform))
    check(boundary_price > .01, 'Wrong phase wrapping must delete actual boundary kinetic')
    return dict(independent_link_2pi_H_defect=shift_defect,
        shifted_link_norm_squared=float(np.linalg.norm(shifted)**2),
        global_quadratic_coercivity_rejected=True,
        actual_compensated_Gauss_residual=compensated,
        uncancelled_source_without_Ref_Gauss_residual=uncancelled,
        wrong_periodic_phase_boundary_energy_defect=boundary_price,
        all_controls_discriminate=True)


def torus_alias_budget(config: ResponseConfig, tolerance: Fraction = Fraction(1,10**10)) -> dict[str, Any]:
    """Finite torus minus EXACT continuum: analytic bound, not quadrature proof.

    Analytic h^2=E^2 I in the selected Clifford scalar span keeps occupied
    projectors holomorphic when 4rho sum ell||t||<=M/16. Fourier contours
    then bound every nonzero grid alias. Numerical supplier quadrature is
    NOT silently promoted into a rigorous continuum enclosure.
    """
    from dataclasses import replace
    groups = config.species or ((config.flavors,config.m),)
    strips, static_bounds, epsilon_bounds, rows = [], [], [], []
    for count, mass in groups:
        child = replace(config,m=mass,flavors=count,species=())
        _, _, law = make_band(child)
        check(config.g0 > .25 and max(config.g0,config.gs) < 1, 'Explicit frame norm/gap enclosure')
        M = Fraction(float(mass))/2
        sums = [Fraction(0),Fraction(0),Fraction(0)]
        for disp, scalar in law.paths:
            length = int(np.max(disp))
            cap = sum((abs(Fraction(float(z.real)))+abs(Fraction(float(z.imag))) for z in scalar),Fraction(0))
            for n in range(3):
                sums[n] += length**n*cap
        rho = min(Fraction(1,20),M/(64*sums[1]))
        hmax = abs(Fraction(float(law.onsite)))+2*sums[0]
        pmax = 1+2*hmax/M
        jmax, cmax = 4*sums[1],4*sums[2]
        Fstatic = 4*pmax*cmax+16*pmax*pmax*jmax*jmax/M
        Feps = 64*pmax*pmax*jmax*jmax/(M**3)
        strips.append(rho); static_bounds.append(count*Fstatic); epsilon_bounds.append(count*Feps)
        rows.append(dict(count=count,mass=mass,gap_M_lower_exact=str(M),analytic_strip_exact=str(rho),
                         static_holomorphic_integrand_upper_exact=str(Fstatic),
                         epsilon_holomorphic_integrand_upper_exact=str(Feps)))
    strip = min(strips)
    total = max(sum(static_bounds),sum(epsilon_bounds))/Fraction(float(config.a))
    bits = ceil_log2(96*total/tolerance)+2
    size = (bits*strip.denominator+strip.numerator-1)//strip.numerator
    bound = 96*total/Fraction(1 << bits)
    check(bound <= tolerance, 'Actual analytic alias budget')
    mode_integer = max(1,round(size*config.a*.05/(2*math.pi)))
    q = 2*math.pi*mode_integer/(size*config.a)
    return dict(config=config.__dict__,species_analytic_enclosures=rows,torus_size=size,torus_volume=size**3,
        physical_box=size*config.a,mode_integer=mode_integer,actual_q_physical=q,
        analytic_grid_alias_physical_static_and_epsilon_upper_exact=str(bound),
        target_error_exact=str(tolerance),all_nonzero_aliases_summed=True,
        proof='sum_alias exp(-rho N |l|1)=[(1+2/(exp(rho N)-1))^3-1] <=96*2^-bits; actual species weighted before 1/a',
        certified_comparison='exact finite BZ sum versus exact continuum integral only',
        NOT_certified='numerical quadrature remainder, photon pole error, fullfield phase, microscopic HP transfer',
        no_native_dense_allocation_required=True)


class NativeSeaWell:
    """Actual finite periodic ImprovedPhiLaw, all occupied states and N flavours.

    A=theta*cos(2pi*x/N) on y links has zero harmonic component. Block Fourier
    reduction in y,z is exact, not a sampled bulk-BZ determinant replacement.
    """
    def __init__(self, config: ResponseConfig, size: int):
        self.config, self.size = config, size
        self.band, self.frame, _ = make_band(config)
        self.law = ImprovedPhiLaw(size, self.band)
        self.volume = size**3
        self.pattern = np.cos(2*math.pi*np.arange(size)/size)
        self.pattern_norm2 = self.volume/2
        self.blocks = [(2*math.pi*y/size, 2*math.pi*z/size)
                       for y in range(size) for z in range(size)]
        self.paths = [(int(np.flatnonzero(d)[0]), int(np.max(d)),
                       np.einsum('m,mij->ij', s, self.frame)) for d, s in self.law.paths]
        self.epsilon_bare = config.a*self.pattern_norm2/(2*config.inertia)
        self.hzero = [self.h(0., k) for k in self.blocks]
        self.energy_zero = sum(float(np.linalg.eigvalsh(h)[:2*size].sum()) for h in self.hzero)
        from dataclasses import replace
        self.children = [NativeSeaWell(replace(config,m=mass,flavors=count,species=()),size)
                         for count,mass in config.species]
        self.flavour_weights = [count for count,_ in (config.species or ((config.flavors,config.m),))
                                for _ in self.blocks]

    def h(self, theta: float, k: tuple[float, float], derivative: int = 0) -> np.ndarray:
        n = self.size
        matrix = np.zeros((4*n, 4*n), complex)
        if derivative == 0:
            for x in range(n):
                matrix[4*x:4*x+4, 4*x:4*x+4] = self.law.onsite*self.frame[0]
        for axis, length, t in self.paths:
            for x in range(n):
                if axis == 0:
                    if derivative:
                        continue
                    y = (x+length) % n
                    term = t
                else:
                    y = x
                    phase_rate = length*self.pattern[x] if axis == 1 else 0.
                    phase = np.exp(1j*(length*k[axis-1]+theta*phase_rate))
                    term = (1j*phase_rate)**derivative*phase*t
                matrix[4*x:4*x+4, 4*y:4*y+4] += term
                matrix[4*y:4*y+4, 4*x:4*x+4] += term.conj().T
        return matrix

    def sea(self, theta: float):
        if self.children:
            parts = [child.sea(theta) for child in self.children]
            return sum(e for e,_ in parts), [v for _,vectors in parts for v in vectors]
        energies, occupied = [], []
        for k in self.blocks:
            e, u = eigh(self.h(theta, k))
            energies.append(float(e[:2*self.size].sum()))
            occupied.append(u[:, :2*self.size])
        return self.config.flavors*(sum(energies)-self.energy_zero), occupied

    def potential(self, theta: float) -> float:
        return self.sea(theta)[0]

    def response(self, theta: float) -> dict[str, Any]:
        deltas, weights = [], []
        contact, metric, gap = 0., 0., math.inf
        for model, k in ((model,k) for model in (self.children or [self]) for k in model.blocks):
            e, u = eigh(model.h(theta, k))
            n = 2*self.size
            occ, empty = u[:, :n], u[:, n:]
            j, c = model.h(theta, k, 1), model.h(theta, k, 2)
            delta = e[n:][None, :]-e[:n, None]
            g2 = np.abs(occ.conj().T@j@empty)**2
            deltas.append(delta.ravel())
            weights.append((2*model.config.flavors*g2).ravel())
            contact += model.config.flavors*real(np.trace(occ.conj().T@c@occ))
            metric += model.config.flavors*float(np.sum(g2/delta**2))
            gap = min(gap, float(np.min(delta)))
        d, w = np.concatenate(deltas), np.concatenate(weights)
        stiffness = contact-float(np.sum(w/d))
        induced_inertia = float(np.sum(w/d**3))
        result: dict[str, Any] = dict(contact=contact, paramagnetic=float(np.sum(w/d)),
            curvature=stiffness, quantum_metric=metric, pair_gap=gap,
            epsilon_bare=self.epsilon_bare, epsilon_induced=induced_inertia,
            finitebox=True, q_physical=2*math.pi/(self.size*self.config.a))
        if stiffness > 0:
            def inverse(x):
                return self.epsilon_bare*x-stiffness+float(np.sum(w*x/(d*(d*d-x))))
            result.update(local_omega=math.sqrt(stiffness/(self.epsilon_bare+induced_inertia)),
                          omit_contact_curvature=-float(np.sum(w/d)), omit_poles_curvature=contact)
            # The lowest GLOBAL gap may be a dark q=0 pair. The standing
            # coexact current has only +/-q support. Never use a dark gap
            # as proof that the actual current pole fails.
            coupled = w > float(np.max(w))*1e-12
            coupled_gap = float(np.min(d[coupled]))
            upper = coupled_gap*coupled_gap*(1-1e-10)
            result['current_coupled_gap'] = coupled_gap
            result['below_pair_pole'] = inverse(upper) > 0
            if result['below_pair_pole']:
                pole = cast(float, brentq(inverse, 0., upper, xtol=1e-25, full_output=False))
                result.update(omega=math.sqrt(pole),
                              residue=1/(self.epsilon_bare+float(np.sum(w*d/(d*d-pole)**2))))
        return result

    def overlap(self, left: list[np.ndarray], right: list[np.ndarray]) -> complex:
        # Product over EVERY transverse momentum, occupied orbital and flavour.
        logabs, phase = 0., 0j
        for a, b, count in zip(left, right, self.flavour_weights):
            sign, logdet = np.linalg.slogdet(a.conj().T@b)
            if abs(sign) == 0:
                return 0j
            logabs += count*float(logdet)
            phase += count*np.log(complex(sign))
        return complex(np.exp(logabs+phase))

    def prepared(self, center: float, radius: float, nodes: int) -> dict[str, Any]:
        """Native coupled field Galerkin with exact outside-sea Fock residual.

        Electric finite difference is an explicit coordinate UV regularization.
        Boundary nodes are retained Dirichlet zeros, NOT periodic phase wrapping.
        At each field node H_matter has its exact normalized all-flavour Slater.
        Q is electric departure from these fibres; Q is not Root code Q.
        """
        x = np.linspace(center-radius, center+radius, nodes+2)[1:-1]
        dx = float(x[1]-x[0])
        matter, seas = zip(*(self.sea(float(y)) for y in x))
        # Independent phase alignment of the full Slater fibre; not scalar energy subtraction.
        aligned = [list(s) for s in seas]
        for i in range(1, len(aligned)):
            ov = self.overlap(aligned[i-1], aligned[i])
            phase = np.angle(ov)/self.flavour_weights[0]
            aligned[i][0] = aligned[i][0].copy()
            aligned[i][0][:, 0] *= np.exp(-1j*phase)
        gram = np.eye(nodes, dtype=complex)
        for offset in (1, 2):
            for i in range(nodes-offset):
                z = self.overlap(aligned[i], aligned[i+offset])
                gram[i, i+offset] = z
                gram[i+offset, i] = z.conjugate()
        kinetic = 1/(2*self.epsilon_bare*dx*dx)
        h = np.diag(np.asarray(matter)+2*kinetic).astype(complex)
        for i in range(nodes-1):
            h[i, i+1] = -kinetic*gram[i, i+1]
            h[i+1, i] = h[i, i+1].conjugate()
        # At node i the only omitted electric vectors are adjacent sea fibres.
        q2 = np.zeros((nodes, nodes), complex)
        for i in range(nodes):
            neighbors = [j for j in (i-1, i+1) if 0 <= j < nodes]
            for j in neighbors:
                for k in neighbors:
                    q2[j, k] += kinetic**2*(gram[j, k]-gram[j, i]*gram[i, k])
        q2 = (q2+q2.conj().T)/2
        e, v = eigh(h)
        states = []
        for n in range(2):
            state = v[:, n]
            residual2 = max(0., real(np.vdot(state, q2@state)))
            field_variance = real(np.vdot(state, (x-center)**2*state))
            boundary_shell = float(abs(state[0])**2+abs(state[-1])**2)
            electric = real(np.vdot(state, (h-np.diag(matter))@state))
            states.append(dict(normalization=real(np.vdot(state, state)), Peierls_plus_bare_electric_energy=float(e[n]),
                sea_matter_energy=real(np.vdot(state, np.asarray(matter)*state)), bare_electric_energy=electric,
                full_Fock_Q_residual=math.sqrt(residual2), field_variance=field_variance,
                endpoint_shell_probability=boundary_shell))
        omega = float(e[1]-e[0])
        transition = complex(np.vdot(v[:, 0], (x-center)*v[:, 1]))
        rpa = self.response(center)
        frequency_reference = float(rpa['omega'] if rpa['below_pair_pole'] else rpa['local_omega'])
        residue_reference = float(rpa['residue'] if rpa['below_pair_pole'] else 1/(rpa['epsilon_bare']+rpa['epsilon_induced']))
        period = 2*math.pi/frequency_reference
        # Retain full Gauss-Coulomb operator, not just its density expectation.
        coulomb_upper = self.config.inertia/self.config.a*self.volume*(2*self.config.flavors+4)**2/(4*math.sin(math.pi/self.size)**2)
        # Outside the interval, the discrete full electric law couples each
        # endpoint to its neighbouring retained external field node.
        boundary_residuals = [kinetic*math.sqrt(s['endpoint_shell_probability']) for s in states]
        total_residual = sum(s['full_Fock_Q_residual'] for s in states)+sum(boundary_residuals)+2*coulomb_upper
        finiteT = min(period, .05/max(total_residual, 1e-300))
        errors = [total_residual*t for t in (0., finiteT/2, finiteT)]
        # This is the actual projected coupledfield propagator, not oscillator dynamics substituted for H.
        trajectory = []
        initial = (v[:, 0]+v[:, 1])/math.sqrt(2)
        coefficients = v.conj().T@initial
        for t in np.linspace(0., finiteT, 5):
            propagated = v@(np.exp(-1j*e*t)*coefficients)
            trajectory.append(dict(time=float(t), norm=real(np.vdot(propagated, propagated)),
                coordinate=real(np.vdot(propagated, (x-center)*propagated)),
                electric_Q_norm=math.sqrt(max(0., real(np.vdot(propagated, q2@propagated)))),
                energy=real(np.vdot(propagated, h@propagated))))
        return dict(nodes=nodes, radius=radius, dx=dx, field_domain='retained Dirichlet interval; NOT compact wrapping',
            sea_fock_dimension_not_allocated=True, exact_species=self.config.flavors,
            occupied_per_species=2*self.volume, states=states, microscopic_quasimode_gap=omega,
            microscopic_dipole_squared=abs(transition)**2, rpa=rpa,
            below_pair_spectral_pole=rpa['below_pair_pole'],
            frequency_reference='actual belowpair RPA pole' if rpa['below_pair_pole'] else 'FAILED local expansion, no belowpair pole admitted',
            pole_relative_error=abs(omega/frequency_reference-1),
            residue_relative_error=abs((2*omega*abs(transition)**2)/residue_reference-1),
            retained_Coulomb_operator_upper=coulomb_upper, retained_boundary_residuals=boundary_residuals,
            finiteT=finiteT, photon_period=period, controlled_period_fraction=finiteT/period,
            full_microscopic_Duhamel_errors=errors, actual_projected_native_trajectory=trajectory,
            nonvacuum='normalized (prepared0+prepared1)/sqrt2; onequantum state separately retained',
            quantum_excess_energy=omega, full_Gauss_energy_excess_interval=[omega-2*coulomb_upper,omega+2*coulomb_upper],
            endpoint_leakage_not_deleted=True,
            physical_full_continuum_discretization_proved=False)


def fullfield_descriptor(config: ResponseConfig, size: int, field_radius: float,
                         heavy_scale: int = 256, normal_radius: Fraction | None = None,
                         exact_mean_precision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Constructive finite-energy functional, with NEW explicit costs for Ntotal.

    The quotient chart is not required to be centred at a stationary Phi field.
    It is a prepared wavepacket. W is the full Haar/Gauss chart isometry, not an
    old normal-ground or old coexact Fock insertion.
    """
    V, D = size**3, 90*size**3
    sector=centre_sector(size,config.flavors)
    check(size % 2 == 0 and sector['positive'], 'FullGauss filled-heavy state requires even physical torus')
    _, frame, light = make_band(config)
    heavy = ImprovedPhiLaw(size, RotationBand(a=config.a, m=.2, r=.12, v=.3, improved=False))
    Llight = abs(float(light.onsite))+2*sum(float(np.sum(np.abs(s))) for _, s in light.paths)
    species = config.species or ((config.flavors, config.m),)
    Llight_total = sum(count*(Llight+mass-config.m) for count,mass in species)
    Llight = max(Llight+mass-config.m for _,mass in species)
    Lheavy = abs(float(heavy.onsite))+2*sum(float(np.sum(np.abs(s))) for _, s in heavy.paths)
    # With rho^2=sum_xmu Tr(Phi_xmu^2)/4, occupied fixed-number operator norm.
    Lambda = heavy_scale
    linear = 4*V*(Fraction(float(Llight_total))+Lambda*Fraction(float(Lheavy)))
    quartic = Fraction(Lambda,4*V)/Fraction(float(config.a))
    root_shift = linear*linear/(2*quartic)+linear/4
    # Direct site-local gauge slice and product Haar coordinates, NOT the old
    # four-inverse normal-eigenbundle construction. Orbit Gram>=gmin/2.
    d = 105*V
    rho_chart = math.ceil(math.sqrt(4*max(config.g0,config.gs)))
    hchart = 4096*15*15*(1+rho_chart+math.ceil(max(Llight,Lheavy)))
    check(min(config.g0, config.gs) > .05, 'Explicit chart Gram lower')
    orbit_lower = Fraction(float(min(config.g0,config.gs)))/2
    orbit = np.array([[1j*(t@p-p@t) for p in frame] for t in light.generators])
    orbit_metric = np.einsum('amij,bmji->ab',orbit,orbit).real/4
    orbit_min = float(np.linalg.eigvalsh(orbit_metric).min())
    check(orbit_min >= float(orbit_lower)*(1-1e-10), 'Actual all15 gauge-orbit Gram lower')
    base_log = ceil_log2(1024*hchart/orbit_lower)
    chart_log = 8*base_log
    eta_exponent = 2*base_log+(D.bit_length()+1)//2+2
    eta = Fraction(1, 1 << eta_exponent)
    if normal_radius is not None:
        eta = min(eta,normal_radius/(math.isqrt(D)+1))
        eta_exponent = ceil_log2(1/eta)
    phi_packet = dict(radius_exact=str(eta), normalization_squared_exact=str(Fraction(315,256)/eta),
                      coordinate_squared_exact=str(eta*eta/11), gradient_squared_exact=str(3/(eta*eta)),
                      laplacian_squared_exact=str(Fraction(63,2)/(eta**4)), boundary_value=0., boundary_derivative=0.)
    em_packet = packet_moments(field_radius)
    harmonic_packet = packet_moments(field_radius)
    chi = Lambda**-.5
    bare_normal_kinetic = Fraction(1,2*math.isqrt(Lambda))*D*3/(eta*eta)
    phi_kinetic_log = ceil_log2(bare_normal_kinetic)
    modes = 2*(V-1)
    electric = config.inertia/config.a*((modes+3)*3+8)/field_radius**2
    rho_center = math.sqrt(V*(config.g0+3*config.gs))
    rho_max = math.ceil(rho_center)+1
    lambda54 = math.isqrt(math.isqrt(Lambda**5))+1
    scalar = Fraction(2*Lambda)/Fraction(float(config.a))*rho_max**4+lambda54*3*rho_max**2+Fraction(7,10)*lambda54*3*8*V
    fermion_energy = linear*rho_max
    # Exact density occupation bounds; inverse Laplacian includes actual a.
    # rho_x=n_L,total-2Ntotal + Ref compensation, 0<=n_species<=4.
    density_squared = V*(2*config.flavors+4)**2
    coulomb = Fraction(float(config.inertia))/Fraction(float(config.a))*density_squared*size*size/16
    multiplication_upper = scalar+fermion_energy+Fraction(float(electric))+coulomb
    encoding_defect = (linear*Fraction(str(exact_mean_precision['float_consumer_frame_operator_error']))
                       if exact_mean_precision else Fraction(0))
    defect_beta=Fraction(1,16)
    multiplication_upper += encoding_defect+defect_beta
    # Full chart/Kato connection and ALL charged species, not flat kinetic.
    current_endpoint = 2*sum(int(np.max(disp))*float(np.sum(np.abs(s))) for disp, s in light.paths)
    field_delta = current_endpoint*math.sqrt(modes+3)*field_radius
    gap_lower = config.m*math.sqrt(config.g0)-field_delta
    check(gap_lower > 0, 'Prepared allmode field support preserves actual Slater gap')
    endpoint = 16*(1+max(Llight,Lheavy))*(1+rho_max)
    kato_scale = math.ceil(64*(2*V)*endpoint/gap_lower)
    section2_upper = 2*(4*kato_scale)**6
    product_jet_upper = 4*(config.flavors+1)**2*section2_upper**2
    representation = 4*V*(config.flavors+1)+4
    # q<=2 chart coefficients <=q! J^(q+1); new representation norm paid.
    kinetic_graph_log = (16+2*(D+modes+4).bit_length()+4*chart_log+
                         2*representation.bit_length()+max(2*eta_exponent+D.bit_length()+8,
                         ceil_log2(product_jet_upper), math.ceil(math.log2(1+8*em_packet['laplacian_squared']))))
    full_energy_log = max(kinetic_graph_log,ceil_log2(1+multiplication_upper))+2
    full_action_log = full_energy_log+4
    if Lambda.bit_length() < 700:
        root_trial = renewed_bounds(config,V,heavy_scale=Lambda)
    else:
        # SAME explicit Gauss pair-singlet/Haar/Gaussian trial, evaluated
        # rationally beyond binary64; neither negative-sea nor ground claim.
        sigma2 = Fraction(1,4)
        qt = Fraction(Lambda)*2/Fraction(float(config.a))*Fraction(60*62,2)*sigma2*sigma2
        pk = Fraction(60,8)/sigma2/math.isqrt(Lambda)
        al = Fraction(1,5)*lambda54*3*480*sigma2/8
        cm = Fraction(7,10)*lambda54*3*4
        ph = Fraction(float(config.inertia))/Fraction(float(config.a))*(2*V+1)/4
        trial = V*(qt+pk+al+cm)+ph
        root_trial = dict(code_trial_upper_exact=str(trial),code_trial_upper_log2=ceil_log2(trial),
            actual_Gauss_pair_singlet_trial=True,heavy_empty=True,negative_sea_trial=False,
            Phi_coordinate_variance='1/4',SU4_Haar=True,all_coexact_and_harmonic_gaussians=True,
            bare_electric_upper_exact=str(ph),ordinary_full_field_Hamiltonian_retained=True)
    heavy_pair_kinetic=Fraction(3,8)*V/math.isqrt(Lambda)
    fixed_heavy_upper=Fraction(root_trial['code_trial_upper_exact'])+heavy_pair_kinetic
    fixed_heavy_trial=dict(code_trial_upper_exact=str(fixed_heavy_upper),
        code_trial_upper_log2=ceil_log2(fixed_heavy_upper), heavy_particles=2*V,
        heavy_empty=False,negative_sea_trial=False,nearest_pair_Wilson_exterior2=True,
        heavy_pair_colour_kinetic_upper_exact=str(heavy_pair_kinetic),
        actual_same_NH_sector=True,other_factors='Root pairedcharged singlets, Ref4/0, isotropic Phi Gaussian, Haar links, allreal EM Gaussians',
        colour_magnetic_Haar_mean=4,colour_electric_coefficient='.3 chi (LAW05)',
        centre=sector)
    current = 4*V*config.flavors*rho_max*2*sum(int(np.max(d))*float(np.sum(np.abs(s))) for d, s in light.paths)
    contact = 4*V*config.flavors*rho_max*2*sum(int(np.max(d))**2*float(np.sum(np.abs(s))) for d, s in light.paths)
    # Energy-form interpolation prices unbounded Phi-linear insertions.
    # H+B >= c rho^4 AND UElink sum E_T^2, on the entire physical domain.
    retained_c=quartic/2
    B=root_shift+defect_beta
    graph_upper=Fraction(1 << full_action_log)+B
    C=max(Fraction(float(current/rho_max)),Fraction(float(contact/rho_max)))
    source_h_delta=endpoint*math.sqrt(D)*float(eta)+field_delta
    source_Gram_lower=2-32*source_h_delta/(config.m*math.sqrt(config.g0))
    check(source_Gram_lower>1, 'Actual localized sea source Fock Gram stays positive')
    # Factor16 includes ||S||<=4 and normalization of S psi; the additional
    # linear term prices changed normalization, never merely the unnormalized image.
    fourth_coefficient=(16*C)**4*2*graph_upper/retained_c
    epsilon_bits=max(8,(ceil_log2(fourth_coefficient)+34)//3,
                     ceil_log2(8192*C*rho_max)+1)
    full_clock=Fraction(1,1 << (full_action_log+epsilon_bits))
    epsilon=Fraction(1,1 << epsilon_bits)
    fourth_error=fourth_coefficient*epsilon**3
    normalization_error=32*C*rho_max*epsilon
    check(fourth_error<=Fraction(1,256**4) and normalization_error<=Fraction(1,256),
          'Actual weighted current/contact and normalized source finiteclock')
    em_radius2=Fraction(float(field_radius))**2
    ue_link=Fraction(float(config.inertia))/Fraction(float(config.a))
    drift2=4*ue_link*graph_upper*full_clock**2
    escape=Fraction(1,128)+drift2/(128*(modes+3)*em_radius2)
    check(escape<Fraction(1,100), 'FiniteT realfield escape probability bound')
    weighted=dict(retained_quartic_exact=str(retained_c),B_exact=str(B),
        defect_beta_exact=str(defect_beta),epsilon_bits=epsilon_bits,
        graph_H_plus_B_upper_exact=str(graph_upper),source_Gram_lower=source_Gram_lower,
        source_operator_cap=4,source_normalized_insertions=True,
        source='sum_c tagged_c^dagger Ref_c on Ref4 (+); adjoint on Ref0 (-), each with its selected neutral Ref charge background. Not both signs on Ref4',
        source_charge_and_centre='colocated tagged/Ref transfer is SU4 scalar, cancels U1 rho operator and preserves total fundamental number',
        current_and_contact_error_upper='1/128',fourth_weighted_error_exact=str(fourth_error),
        source_normalization_error_exact=str(normalization_error),
        interpolation='||rho delta|| <= c^-1/4 ||delta||^3/4 ||(H+B)delta||^1/4; ||(H+B)delta||<=2||(H+B)psi||. Factor16 includes normalized primitive source, not Wick or RPA',
        unweighted_Duhamel_error_exact=str(epsilon),
        realfield_escape_probability_upper_exact=str(escape),
        escape_radius_squared_exact=str(256*(modes+3)*em_radius2),
        escape_proof='[H,A_alpha]=-2i UElink E_alpha; conserved form energy bounds all E_T. Initial support ||A||<=sqrt(M)R; squared drift<=4UElink E T^2; Markov outside16sqrt(M)R. No global A coercivity asserted')
    photon_packet=dict(state1='sqrt11 A_alpha/R times prepared0; odd factor, orthogonal and exactly normalized',
        coordinate_variance0_exact=str(em_radius2/11),coordinate_variance1_exact=str(3*em_radius2/13),
        momentum_variance0_exact=str(3/em_radius2),momentum_variance1_exact=str(11/em_radius2),
        laplacian_squared1_exact=str(Fraction(495,2)/em_radius2**2),
        physical_standing_wave_leg=float(field_radius/config.a*math.sqrt(2/(11*V))),
        canonical_RPA_leg_matched=False,
        meaning='actual prepared nonvacuum H2 field insertion. Its measured leg is not silently replaced by sqrt(Z/(2omega)); no photon-period transfer admitted')
    fields = np.tile(frame, (V, 1, 1, 1))
    # light lead law has N3; build correct-size law for actual scalar evaluation.
    scalar_law = ImprovedPhiLaw(size, light.band)
    bonds = [(x, int(scalar_law.target[x, k])) for x in range(V) for k in range(3)]
    elastic, align = local_potential(fields, bonds, [np.eye(4) for _ in bonds])
    return dict(identity='INDUCED-prepared-localized-fullGauss-v1', total_charged_species=config.flavors,
        active_species='one of Ntotal, its valence determinant INCLUDED, not extra empty Dirac species',
        heavy_species=1, original_heavy_EM_neutral=True, Ref='one local Ref4 singlet (+neutral branch), or Ref0 (-neutral branch); branch-specific neutral charge background, actual colocated source compensation',
        domain='Phi Cartesian H2 bump, SU4 full Haar chart H2 bump, A_coexact R^(2(V-1)), A_h R^3, exact Gauss quotient',
        microscopic_H='chi PiPhi^2/2+.3chi E_SU4^2+Lambda Vquartic(kappa=2/a)+Lambda^(5/4) Walign+.7Lambda^(5/4) W_SU4plaquette-Lambda dGamma(hHeavy(a))+sum_Ntotal dGamma(hLight(Phi,U,exp(iA)))+UEphys/a E_U1^2+H_C+H_aux+H_Ref; NO bare curl^2. Canonical LAW05 colour and Phi kinetics retained',
        heavy_scale_in_bound=Lambda, no_heavy_scale_omitted=True,
        Lambda=Lambda, chi=chi, UEphys=config.inertia, UElink=config.inertia/config.a,
        Gauss='div E_U1-rho_L,total-rho_Ref=0; SU4 div E4-rho_heavy-rho_alllight-rho_Ref=0, solved by full W, not zero density expectation',
        longitudinal='E_L=grad L+ rho; H_C=UElink rho^T L+rho retained',
        state='W[product normalized H2 compact bumps * whole heavy occupied2V positive Slater * product_Ntotal whole negative2V Slaters * auxiliarycode * Ref4]/norm; photon0/1 localized factors replace one coexact bump; full coupled evolution is H, not the sea projection',
        normalization=1., normalization_proof='product Beta(.5,5) integrals, normalized complete Slaters, Haar chart W isometry on centre-admissible sector; sum local centre deck branches, global centre character1. No Gaussian photon vacuum assumed',
        gauss_norm=0., gauss_proof='exact constraint by W Coulomb/colour dressing with local Z4 deck branches; connected-link stabilizer is global Z4 and selected total character1. Uncancelled Ref and oddV are adverse controls',
        global_centre_sector=sector,
        prepared_not_ground=True, periodic_potential_not_globally_coercive=True,
        config=config.__dict__, preparation_heavy_scale_exact=heavy_scale,
        supplied_total_normal_radius_exact=str(normal_radius) if normal_radius is not None else None,
        no_stationarity_or_mean_precision_claim=True,
        centre_precision_input=exact_mean_precision,
        centre_definition='serialized decimal frame scales' if exact_mean_precision else 'binary64 config frame',
        float_frame_encoding_H_defect_upper=encoding_defect,
        exact_mean_usage='actual centre is the serialized decimal Clifford frame; binary64 is only used for broad outward-majorants, and its matter-H defect is explicitly included' if exact_mean_precision else None,
        fixed_NH2V_Gauss_trial=fixed_heavy_trial,
        Root_heavy_empty_trial_scope='full Fock variational upper ONLY, not this fixed NH2V sector',
        normal_dimension=D, coexact_modes=modes, harmonic_modes=3, full_H2_boundary_retained=True,
        normal_packet=phi_packet, allmode_EM_packet=em_packet, harmonic_packet=harmonic_packet,
        quartic_coefficient=quartic, linear_matter_coefficient=linear, renewed_Root_lower_shift=root_shift,
        rho_support_upper=rho_max, bare_kinetic_normal_upper_log2=phi_kinetic_log,
        bare_allmode_electric_upper=electric, Coulomb_action_operator_upper=coulomb,
        scalar_upper=scalar, matter_upper=fermion_energy,
        full_energy_upper_log2=full_energy_log, full_H_action_upper_log2=full_action_log,
        new_chart=dict(dimension=d, local_gauge_dimension=15, h=hchart,
                       orbit_Gram_lower_exact=str(orbit_lower), actual_orbit_Gram_min=orbit_min,
                       J_upper_log2=chart_log,radius_upper_choice_log2=-2*base_log,eta_exponent=eta_exponent,
                       charge_representation_upper=representation,species_jets=product_jet_upper,
                       Slater_gap_lower=gap_lower,minimum_claimed=False,
                       gauge_slice='F_a=sum_mu Tr(i[T_a,Phi_mu0]deltaPhi_mu)/4=0; direct local IFT and product exponential Haar',
                       jet_proof='DF_orbit>=gmin/2; q<=2 adjoint/exponential jets bounded by h. Two inverse operations on radius(orbit_lower/(1024h))^2 have q!J^(q+1) coefficient/cometric/connection jets, J=(1024h/orbit_lower)^8. No normal-eigenbundle projector or vacuum inserted.'),
        repaired_fullfield_clock_exact=str(full_clock), fullfield_norm_comparison_error_upper='1/128',
        weighted_source_current_contact=weighted, prepared_photon_packet=photon_packet,
        finiteclock_meaning='Each actual fullGauss prepared state has ||H psi|| <=2^full_H_action_upper_log2; Duhamel bounds its departure from the static prepared functional. Common Phi phase not subtracted. Extremely short existence bound, NOT transfer of the BO residual or a resolved photon period.',
        alternate_exact_fullGauss_trial=root_trial,
        actual_center_quartic=elastic/config.a, actual_center_alignment=align,
        current_operator_upper=current, contact_operator_upper=contact,
        field_photon_leg_fluctuation_upper=math.sqrt(em_packet['coordinate_squared']),
        normal_backreaction_not_stationary=True, fullfield_photon_period_admitted=False,
        quantum_law_and_parameters_inputs=True,
        adverse=dict(global_harmonic='independent 2pi link shifts leave determinant bounded/periodic, A norm diverges: no coercive quadratic lower bound',
                     uncancelled_Gauss=dict(source_charge=1, Ref_compensation=0, Gauss_defect=1),
                     wrong_phase_domain='periodic wrap identifies noncompact boundary, deletes electric escape term; not selected law'))


def portable(data: Any) -> Any:
    if isinstance(data, Fraction):
        return str(data)
    if isinstance(data, dict):
        return {k: portable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [portable(v) for v in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    return data


def run(size: int, nodes: int) -> dict[str, Any]:
    adverse = NativeSeaWell(ResponseConfig(a=.0625, flavors=64), 3)
    adverse_response = adverse.response(0.)
    check(adverse_response['curvature'] < 0, 'Wrong-small-box adverse must be unstable')
    initial = NativeSeaWell(ResponseConfig(a=.0625, flavors=64), size)
    zero_response = initial.response(0.)
    center = 0.
    scan = [(float(x), initial.potential(float(x))) for x in np.linspace(-.1, .1, 9)]
    response = initial.response(center)
    check(response['curvature'] > 0, 'Executed larger-physical-box repair must have a positive local well')
    width_initial = math.sqrt(1/(response['epsilon_bare']+response['epsilon_induced'])/(2*response['local_omega']))
    failed = initial.prepared(center, min(.7, 7*width_initial), nodes)
    # Explicit new electric parameter, not a common-cone or fitted-speed repair.
    repair_config = ResponseConfig(a=.0625, flavors=64, inertia=1e-16)
    repaired = NativeSeaWell(repair_config, size)
    repaired_response = repaired.response(center)
    width = math.sqrt(float(repaired_response['residue'])/(2*float(repaired_response['omega'])))
    repaired_native = repaired.prepared(center, 7*width, nodes)
    refined = repaired.prepared(center, 7*width, 2*nodes+1)
    descriptor = fullfield_descriptor(repair_config, size, 1e-8)
    descriptor['native_sea_fibre_width_NOT_inherited']=True
    joint_initial = joint_collective(ResponseConfig(a=1., flavors=64), 18, 8)
    joint_repair = joint_collective(ResponseConfig(a=1., flavors=64, species=((1,.2),(63,2.))), 18, 16)
    controls = adverse_field_controls(ResponseConfig(a=.0625, flavors=64), size)
    canonical_alias = torus_alias_budget(ResponseConfig(a=.015625, flavors=64, species=((1,.2),(63,2.))))
    phi_primary_config = ResponseConfig(a=.015625,flavors=64,species=((1,.2),(63,2.)),
        g0=.3605245769279699,gs=.13278766227047106)
    phi_primary_field = fullfield_descriptor(phi_primary_config,size,1e-8)
    heavy_control=heavy_pair_control(phi_primary_config,size)
    phi_bo_path=ROOT/'induced_phi_backreaction_even_BO.json'
    phi_bo_bytes=phi_bo_path.read_bytes()
    phi_bo=json.loads(phi_bo_bytes)
    bo=phi_bo
    exact_mean=bo['exact_heavy_mean_precision']
    bo_config=ResponseConfig(a=.015625,flavors=64,species=((1,.2),(63,2.)),
        g0=float(exact_mean['g0_decimal']),gs=float(exact_mean['gs_decimal']))
    bo_field=fullfield_descriptor(bo_config,6,1e-8,heavy_scale=1<<816,
        normal_radius=Fraction(str(bo['support_radius'])),exact_mean_precision=exact_mean)
    bo_field['external_Phi_capsule']=dict(path=str(phi_bo_path),sha256=hashlib.sha256(phi_bo_bytes).hexdigest(),
        source_at_import=phi_bo['source_at_import'],normal_tangent_residual_not_transferred=True)
    check(abs(refined['states'][0]['normalization']-1)<1e-10, 'Prepared state normalization')
    check(refined['quantum_excess_energy'] > 0, 'Nonvacuum excess energy')
    check(refined['controlled_period_fraction'] > failed['controlled_period_fraction'], 'Material electric/preparation repair must improve native finiteclock transfer')
    imported_names = sorted({Path(str(getattr(module,'__file__',''))).name for module in tuple(sys.modules.values())
                             if Path(str(getattr(module,'__file__',''))).parent == ROOT and
                             Path(str(getattr(module,'__file__',''))).suffix == '.py'})
    imported_hashes = {name: SOURCE_SNAPSHOT[name] for name in imported_names}
    check(all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest for name,digest in imported_hashes.items()),
          'Imported project source changed from preimport snapshot')
    final_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
    check(final_hashes == LOADED_HASHES, 'Source changed during execution; not loaded-source certified')
    return dict(plan=PLAN, assignment_revision=ASSIGNMENT_REVISION,
        law_branch='NEW no bare magnetic term; normalized fullGauss prepared, finite field UV regulator',
        config_initial=initial.config.__dict__, config_repair=repair_config.__dict__,
        zero_field_failed_regime=adverse_response, repaired_zero_field_response=zero_response, actual_selected_phase=center,
        actual_selected_phase_potential=initial.potential(center), raw_cycle_scan=scan,
        initial_sea_fibre_failure=failed, executed_material_repair=repaired_native,
        refined_material_repair=refined,
        refinement=dict(gap_relative_change=abs(refined['microscopic_quasimode_gap']/repaired_native['microscopic_quasimode_gap']-1),
                        residual_ratio=refined['states'][1]['full_Fock_Q_residual']/repaired_native['states'][1]['full_Fock_Q_residual']),
        fullfield=descriptor,
        actual_Phi_newmean_primary_fullGauss_preparation=phi_primary_field,
        exact_even_Phi_BO_centre_fullGauss_preparation=bo_field,
        actual_filled_heavy_Gauss_pair_and_centre_controls=heavy_control,
        joint_UE1_IR_native_initial=joint_initial, joint_UE1_IR_mass_state_repair=joint_repair,
        actual_domain_Gauss_controls=controls,
        canonical_mass_small_a_torus_alias=canonical_alias,
        smallUE_not_hardconsumer_repair=True,
        distinction='finitebox positive local flux well and smallUE prepared window are NOT original UE1 fullBZ photon, commoncone repair, equilibrium vacuum, or fullfield photon-period theorem',
        remaining_debt=['fullfield stationary Phi/SU4 phase and full coupled spectral dressing',
                        'continuum field regulator convergence including boundary escape',
                        'larger physical box shared bulkIR response at UE1',
                        'smallUE repair slows photon cone and does not solve commoncone'],
        source_hashes=LOADED_HASHES, actual_imported_source_hashes=imported_hashes, source_unchanged_during_run=True,
        input=dict(native_size=size,nodes=nodes,joint_size=18,joint_bins_initial=8,joint_bins_repair=16),
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        all_scoped_checks_pass=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=6)
    parser.add_argument('--nodes', type=int, default=41)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.nodes)
    args.output.write_text(json.dumps(portable(result), indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(phase=result['actual_selected_phase'], failed_period_fraction=result['initial_sea_fibre_failure']['controlled_period_fraction'],
        repair_period_fraction=result['refined_material_repair']['controlled_period_fraction'],
        repair_pole_error=result['refined_material_repair']['pole_relative_error'],
        repair_residue_error=result['refined_material_repair']['residue_relative_error']), indent=2))
    print(f'Natural exit: {args.output}; prepared state and actual full-Fock leakage retained', flush=True)


if __name__ == '__main__':
    main()
