#!/usr/bin/env python3
"""Smooth full-SU4/Phi dressed source; complete local-loop and cloud consumers.

This changes the source, never the native Hamiltonian or its domain.  A chart
is admitted on a singular-gap/polar guard, not asserted on arbitrary frames.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any
from types import SimpleNamespace

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from bulk_quantum_selection import BulkQuantumLaw
from noncompact_charged_packet_transport import bundle_evolve, divergence, green, source_gram
from noncompact_marked_metric import maps_from_marks
from noncompact_finite_packet_spin import lift
from noncompact_neutral_supplier import particle_hole
from noncompact_centre_source_energy import orientation_cloud
from noncompact_physical_rotation_join import (
    continuum_packet_consumer, mechanical_connection, screened_boundary,
)
from original_centre_charge import CENTRES, packet, packet_derivatives
from noncompact_mechanical_pulse import (
    SmoothPulse, geometry, native_modes, mechanical_flow,
    smooth_band_value, smooth_band_divided_difference, radial_profile,
)
from original_total_angular_transfer import scalar_orbital, endpoint_flow, primitive_flow
from unrestricted_field_minimum import coordinates
from original_charged_principal import original_frame, portable
from original_physical_rotation_lift import QUARTER_TURN, make_lift, restore_links
from physical_spin_boundary import physical_cubic
from rotation_band import RotationBand
from sea_selection import SELECTED

ROOT = Path(__file__).resolve().parent


def require(error: float, label: str, tolerance: float = 3e-7) -> float:
    error = float(error)
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')
    return error


def frame_chart(phi: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Equivariant least-Clifford intertwiner and polar chart, up to Z4.

    Off-Clifford normals are retained.  The least singular vector must remain
    simple and its polar factor nonsingular; these are computable domain guards.
    Whitening uses all ten Gram entries, not an assumed diagonal frame.
    """
    gram = np.einsum('mij,nji->mn', phi, phi).real / 4
    values, modes = np.linalg.eigh(gram)
    if values[0] <= 1e-10:
        raise ValueError('Phi Gram guard failed')
    gamma = np.einsum('mn,nij->mij', (modes/np.sqrt(values))@modes.T, phi)
    system = np.concatenate([np.kron(np.eye(4), g)-np.kron(s.T, np.eye(4))
                             for g, s in zip(gamma, SELECTED)])
    _, singular, vh = np.linalg.svd(system)
    candidate = vh[-1].conj().reshape(4, 4, order='F')
    left, polar_values, right = np.linalg.svd(candidate)
    gap = float(singular[-2]**2-singular[-1]**2)
    if gap <= 1e-8 or polar_values[-1] <= 1e-8:
        raise ValueError('least-intertwiner/polar guard failed')
    unitary = left@right
    unitary *= np.exp(-.25j*np.angle(np.linalg.det(unitary)))
    return unitary, dict(Gram_min=float(values[0]), singular_squared_gap=gap,
        polar_min=float(polar_values[-1]), nonClifford_residual=float(singular[-1]))


def centre_scalar(v: np.ndarray, eta: float,
                  generators: np.ndarray) -> tuple[complex, np.ndarray]:
    """ORIGINAL whole-group psi1/psi0 and its fifteen left derivatives.

    The common exp(-4eta) normalization in packet cancels exactly.  This is a
    smooth ratio, not nearest-centre projection.  No U1 theta enters this map.
    """
    if not 0 < eta <= 80:
        raise ValueError('finite packet eta must lie in (0,80]')
    p0, p1 = packet(v, eta, 0), packet(v, eta, 1)
    if p0.real <= 0 or abs(p0.imag) > 1e-12:
        raise ValueError('whole-group denominator lost positivity')
    d0 = packet_derivatives(v, eta, 0, generators)
    d1 = packet_derivatives(v, eta, 1, generators)
    return p1/p0, (d1*p0-p1*d0)/(p0*p0)


def scalar_gradient_bound(v: np.ndarray, eta: float) -> float:
    scores = (CENTRES.conj()*np.trace(v)).real
    dominant = int(np.argmax(scores))
    wrong = np.delete(scores, dominant)
    # |d_a ReTr(zV)| <= sqrt(2), fifteen canonical generators.
    return float(min(2*eta*math.sqrt(30),
                     4*eta*math.sqrt(30)*np.exp(eta*(wrong-scores[dominant])).sum()))


def orientation_cloud_derivative_majorants(length: int, eta: float, alpha: float,
                                          Cartesian_frame_squared: float,
                                          initial_metric_lower: float,
                                          max_order: int = 6,
                                          logarithmic: bool = False) -> dict[str, Any]:
    """Conservative ordered source derivatives, NOT a total Noether graph norm.

    Uses B coordinates HS/4, canonical all15 link words, q/2 unit-HS marks.
    Four q-convention marks and the frozen initial inverse are included below.
    Contracting with field/orbital/photon vectorfields needs separate moments.
    """
    if (length < 1 or eta <= 0 or alpha <= 0 or Cartesian_frame_squared <= 0
            or initial_metric_lower <= 0 or not 0 <= max_order <= 6):
        raise ValueError('positive source/domain/metric constants and order0..6 required')
    field_scale=4*alpha*math.sqrt(Cartesian_frame_squared*(length+1))
    link_scale=2*math.sqrt(15*length)*(1+4*eta)
    if logarithmic:
        unit_logs=[math.lgamma(q+1)/math.log(2)+q*math.log2(field_scale+link_scale)
                   for q in range(max_order+1)]
        normalized_logs=[2+value-.5*math.log2(initial_metric_lower) for value in unit_logs]
        return dict(Cartesian_basis_metric='Tr(BaBb)/4=delta; Ta=Ba/sqrt8',
            unit_HS_q_over_2_log2=unit_logs,
            quartet_q_convention_initial_normalized_log2=normalized_logs,
            normalized_power_of_two_upper_exponents=[math.ceil(value)+1 for value in normalized_logs],
            conservative_rounding_margin_bits=1, total_physical_Noether_bound=False,
            unpaid='actual Noether contractions, photon/harmonic and energy-tail graphs')
    return dict(Cartesian_basis_metric='Tr(BaBb)/4=delta; Ta=Ba/sqrt8',
        field_scale=field_scale,link_scale=link_scale,
        unit_HS_q_over_2=[math.factorial(q)*(field_scale+link_scale)**q
                          for q in range(max_order+1)],
        quartet_q_convention_initial_normalized=[
            4*math.factorial(q)*(field_scale+link_scale)**q/math.sqrt(initial_metric_lower)
            for q in range(max_order+1)],
        total_physical_Noether_bound=False,
        unpaid='Lie-field contraction moments, orbital/photon generators, reference/boundary localization')


def source_paths(law: Any, reference: int) -> list[list[tuple[int, int, int, int]]]:
    paths = []
    for x in range(law.volume):
        path, y = [], x
        for axis in range(3):
            while law.sites[y, axis] != law.sites[reference, axis]:
                if law.sites[y, axis] < law.sites[reference, axis]:
                    edge, sign = y, 1
                    target = int(law.target[y, axis])
                else:
                    position = law.sites[y].copy(); position[axis] -= 1
                    edge, sign = int(law.index[tuple(position)]), -1
                    target = edge
                path.append((y, target, edge, sign* (axis+1)))
                y = target
        paths.append(path)
    return paths


def screened_marks(law: Any, fields: np.ndarray, links: np.ndarray, a: np.ndarray,
                   eta: np.ndarray, reference: int, screen_eta: float,
                   harmonic: np.ndarray | None = None) -> np.ndarray:
    """Full bar4RIGHT, primitive quaternion labels, original Coulomb real phase.

    Harmonic argument, when supplied, owns exp(-i(x-R).a_h) and must NOT also
    be inserted in the Coulomb argument.  SU4 centre seams remain in links.
    """
    paths = source_paths(law, reference)
    needed = {reference}
    for x in np.flatnonzero(eta):
        needed.add(int(x))
        needed.update(y for step in paths[x] for y in step[:2])
    charts = {x: frame_chart(fields[x])[0] for x in needed}
    spins = np.asarray([-.5j*SELECTED[j]@SELECTED[k]
                        for j, k in ((2, 3), (3, 1), (1, 2))])
    labels = np.asarray([np.eye(4), *(2j*s for s in spins)])
    g, _, _ = green(law)
    chi = g@divergence(law, a)
    phases = np.exp(1j*(chi-chi[reference]))
    if harmonic is not None:
        phases *= np.exp(-1j*(law.sites-law.sites[reference])@harmonic)
    marks = np.zeros((4, 4*law.volume, 4), complex)
    for x in np.flatnonzero(eta):
        scalar = 1.+0j
        for y, target, edge, encoded in paths[x]:
            axis, sign = abs(encoded)-1, np.sign(encoded)
            transport = links[edge, axis] if sign > 0 else links[edge, axis].conj().T
            scalar *= centre_scalar(charts[y].conj().T@transport@charts[target],
                                    screen_eta, law.generators)[0]
        marks[:, 4*x:4*x+4] = (eta[x]*phases[x]*scalar)*(
            charts[int(x)]@labels@charts[reference].conj().T)
    return marks


def native_consumer(law: Any, frame: np.ndarray, screen_eta: float) -> dict[str, Any]:
    rng = np.random.default_rng(22210651)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    fields += .003*law.fields(rng.normal(size=(law.volume, 4, 15)))
    links = np.asarray([[expm(.08j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))
                         for _ in range(3)] for _ in range(law.volume)])
    a = .05*rng.normal(size=(law.volume, 3))
    reference = 0
    cloud = law.index[(1, 1, 1)]
    eta = np.zeros(law.volume, complex); eta[cloud] = 1
    marks = screened_marks(law, fields, links, a, eta, reference, screen_eta)
    ct_source_error = require(np.linalg.norm(screened_marks(law,-fields,links,a,eta,
        reference,screen_eta)-marks), 'actual whole Phi-sign CT-even screened source')
    gauge = np.asarray([expm(.4j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))
                        for _ in range(law.volume)])
    moved_fields = gauge[:, None]@fields@gauge[:, None].conj().swapaxes(-1,-2)
    moved_links = gauge[:, None]@links@gauge[law.target].conj().swapaxes(-1,-2)
    moved = screened_marks(law, moved_fields, moved_links, a, eta, reference, screen_eta)
    target = np.einsum('xij,axjr,rk->axik', gauge,
        marks.reshape(4,law.volume,4,4), gauge[reference].conj().T).reshape(marks.shape)
    covariance = require(np.linalg.norm(moved-target), 'fullSU4/Phi source equivariance')
    charts = [frame_chart(f)[0] for f in fields]
    u = charts[cloud].conj().T@links[cloud,0]@charts[int(law.target[cloud,0])]
    value, analytic = centre_scalar(u, screen_eta, law.generators)
    step = 1e-5
    numerical = np.asarray([(centre_scalar(expm(1j*step*t)@u,screen_eta,law.generators)[0]
                             -centre_scalar(expm(-1j*step*t)@u,screen_eta,law.generators)[0])/(2*step)
                            for t in law.generators])
    derivative_error = require(np.linalg.norm(numerical-analytic), 'full15 scalar Casimir derivative')
    seams = max(abs(centre_scalar(z*u,screen_eta,law.generators)[0]-z*value) for z in CENTRES)
    require(seams,'centre character')
    # Arbitrary independent chart branch choices at every site telescope.
    z = CENTRES[rng.integers(0,4,size=law.volume)]
    branch_product = 1.+0j; original_product = 1.+0j
    for x,y,edge,encoded in source_paths(law,reference)[cloud]:
        v = charts[x].conj().T@links[edge,abs(encoded)-1]@charts[y]
        original_product *= centre_scalar(v,screen_eta,law.generators)[0]
        branch_product *= centre_scalar(z[x].conjugate()*z[y]*v,screen_eta,law.generators)[0]
    branch = require(abs(z[cloud]*branch_product*z[reference].conjugate()-original_product),
                     'independent Phi centre-chart branch cancellation')
    h = law.onebody(fields,links,a)
    values,modes = np.linalg.eigh(h)
    p = modes[:,values<0]@modes[:,values<0].conj().T
    metric = (source_gram(marks,p)+source_gram(marks,np.eye(len(h))-p))/2
    hs = np.einsum('air,bir->ab',marks.conj(),marks)/2
    require(np.linalg.norm(metric-hs),'fullCAR CT initial metric remeasurement')
    # Execute actual complete 12-mode exterior CAR with all bar4 components.
    small = marks.reshape(4,law.volume,4,4)[:,[cloud,reference,law.index[(1,0,0)]]].reshape(4,12,4)
    rows = tuple(itertools.combinations(range(12),6))
    charged = tuple(itertools.combinations(range(12),7))
    creators = maps_from_marks(small,rows,charged)
    psi = rng.normal(size=len(rows))+1j*rng.normal(size=len(rows)); psi/=np.linalg.norm(psi)
    ct = particle_hole(rows,12)@psi.conj()
    columns = [np.column_stack([b@v for b in creators]) for v in (psi,ct)]
    actual = sum(c.conj().T@c for c in columns)/2
    car = require(np.linalg.norm(actual-hs),'actual complete exterior CAR metric')
    wrong = sum(c[:len(charged)].conj().T@c[:len(charged)] for c in columns)/2
    # Derivatives of the actual source along physical Phi, full Gauss and normal.
    window = np.zeros(law.volume); window[cloud]=1
    ell,mechanical = mechanical_connection(law,frame,window)
    df = np.zeros_like(fields); df[:,1]=-window[:,None,None]*fields[:,2]
    df[:,2]=window[:,None,None]*fields[:,1]
    vf = 1j*(ell[:,None]@fields-fields@ell[:,None])
    vu = 1j*(ell[:,None]@links-links@ell[law.target])
    physical=[]; horizontal=[]; vertical=[]
    for sign in (-1,1):
        physical.append(screened_marks(law,fields+sign*step*df,links,a,eta,reference,screen_eta))
        hu = np.asarray([[expm(-sign*step*vu[x,j]@links[x,j].conj().T)@links[x,j]
                          for j in range(3)] for x in range(law.volume)])
        horizontal.append(screened_marks(law,fields+sign*step*(df-vf),hu,a,eta,reference,screen_eta))
        q = np.asarray([expm(sign*step*1j*e) for e in ell])
        vertical.append(screened_marks(law,q[:,None]@fields@q[:,None].conj().swapaxes(-1,-2),
            q[:,None]@links@q[law.target].conj().swapaxes(-1,-2),a,eta,reference,screen_eta))
    dp,dh,dv = [(v[1]-v[0])/(2*step) for v in (physical,horizontal,vertical)]
    exact = (1j*np.einsum('xij,axjr->axir',ell,marks.reshape(4,law.volume,4,4))
             -1j*np.einsum('axir,rj->axij',marks.reshape(4,law.volume,4,4),ell[reference])).reshape(marks.shape)
    gauss_error=require(np.linalg.norm(dv-exact),'actual full15 Gauss source/right derivative')
    noether_error=require(np.linalg.norm(dp-dv-dh),'actual Phi physical=Gauss+horizontal derivative')
    # Freeze initial metric ONCE; never normalize a final/seam metric.
    eigenvalues, eigenmodes = np.linalg.eigh(metric)
    initial_inverse = (eigenmodes/np.sqrt(eigenvalues))@eigenmodes.conj().T
    def derivative_form(derivative: np.ndarray) -> np.ndarray:
        return initial_inverse@(
            np.einsum('air,bir->ab',derivative.conj(),derivative)/2)@initial_inverse
    path = source_paths(law,reference)[cloud]
    needed = sorted({cloud,reference,*(x for edge in path for x in edge[:2])})
    phi_form=np.zeros((4,4),complex); link_form=np.zeros((4,4),complex)
    phi_densities=[]
    for site in needed:
        for mu in range(4):
            for basis in law.basis:
                changed=[]
                for sign in (-1,1):
                    f=fields.copy(); f[site,mu]+=sign*step*basis
                    changed.append(screened_marks(law,f,links,a,eta,reference,screen_eta))
                derivative=(changed[1]-changed[0])/(2*step)
                phi_form+=derivative_form(derivative)
                if site==cloud:
                    phi_densities.append(float(np.linalg.norm(derivative)**2))
    for _,_,edge,encoded in path:
        axis=abs(encoded)-1
        for generator in law.generators:
            changed=[]
            for sign in (-1,1):
                u_changed=links.copy()
                u_changed[edge,axis]=expm(sign*step*1j*generator)@links[edge,axis]
                changed.append(screened_marks(law,fields,u_changed,a,eta,reference,screen_eta))
            derivative=(changed[1]-changed[0])/(2*step)
            link_form+=derivative_form(derivative)
    # Native current from SAME actual Peierls Hamiltonian and source.
    direction=rng.normal(size=a.shape)
    current=(law.onebody(fields,links,a+step*direction)-law.onebody(fields,links,a-step*direction))/(2*step)
    cg=(law.onebody(moved_fields,moved_links,a+step*direction)-law.onebody(moved_fields,moved_links,a-step*direction))/(2*step)
    big=sparse.block_diag([sparse.csr_matrix(q) for q in gauge]).toarray()
    current_error=require(np.linalg.norm(cg-big@current@big.conj().T),'actual source native current gauge action')
    source_direction=(screened_marks(law,fields,links,a+step*direction,eta,reference,screen_eta)
                     -screened_marks(law,fields,links,a-step*direction,eta,reference,screen_eta))/(2*step)
    chi=green(law)[0]@divergence(law,direction)
    require(np.linalg.norm(source_direction-1j*(chi[cloud]-chi[reference])*marks),'actual Coulomb source derivative')
    # SAME new-source orbital evolution and real photon/electric receiver.
    rf,ru,ra,permutation=physical_cubic(law,fields,links,a,QUARTER_TURN)
    rh=law.onebody(rf,ru,ra)
    orbital_H=require(np.linalg.norm(rh-permutation@h@permutation.T),'screened source actual orbital native H')
    rotated_marks=np.einsum('ij,ajr->air',permutation,marks)
    evolved=bundle_evolve((values,modes),marks,.17)
    rv,rm=np.linalg.eigh(rh)
    evolved_rotated=bundle_evolve((rv,rm),rotated_marks,.17)
    orbital_source=require(np.linalg.norm(evolved_rotated-np.einsum('ij,ajr->air',permutation,evolved)),
                           'SAME screened source native mobile orbital consumer')
    photon=rng.normal(size=a.shape); electric=rng.normal(size=a.shape)
    def photon_energy(vector: np.ndarray, momentum: np.ndarray) -> float:
        curl=sum(np.linalg.norm(vector[law.target[:,i],j]-vector[:,j]
            -vector[law.target[:,j],i]+vector[:,i])**2 for i,j in ((0,1),(0,2),(1,2)))
        return float(np.linalg.norm(momentum)**2+curl)
    _,_,rp,_=physical_cubic(law,fields,links,photon,QUARTER_TURN)
    _,_,re,_=physical_cubic(law,fields,links,electric,QUARTER_TURN)
    photon_error=require(abs(photon_energy(rp,re)-photon_energy(photon,electric)),
                         'SAME world real photon coordinate/electric rotation')
    return dict(gauge_error=covariance,centre_branch_error=branch,centre_character_error=seams,
        scalar_full15_error=derivative_error,scalar_full15_density=float(np.linalg.norm(analytic)**2),
        scalar_gradient_bound=scalar_gradient_bound(u,screen_eta),source_initial_G=metric,
        source_initial_G_eigenvalues=np.linalg.eigvalsh(metric),actual_wholeCAR_error=car,
        whole_Phi_CT_error=ct_source_error,source_metric_not_borrowed2I=True,
        screened_mobile_orbital_H_error=orbital_H,screened_mobile_orbital_source_error=orbital_source,
        actual_photon_vector_electric_error=photon_error,
        omitted_bar4_metric_defect=float(np.linalg.norm(wrong-hs)),
        native_current_gauge_error=current_error,physical_Noether_error=noether_error,
        complete_Gauss_error=gauss_error,horizontal_source_norm=float(np.linalg.norm(dh)),
        Cartesian_Phi_cloud_60_densities=phi_densities,
        Cartesian_Phi_cloud_gradient_sum=sum(phi_densities),mechanical=mechanical,
        all_path_and_RIGHT_Cartesian_Phi_kinetic_matrix=phi_form,
        all_path_full15_colour_kinetic_matrix=link_form,
        angular_horizontal_initial_metric_matrix=derivative_form(dh),
        original_slow_extra_energy_per_chi=.5*phi_form+.3*link_form,
        derivative_scope=dict(Phi_sites=needed,Cartesian_coordinates=60*len(needed),
                              colour_edges=len(path),canonical_link_coordinates=15*len(path)),
        unpaid='neutral expectation and kinetic product cross terms, guard cutoff derivative, photon/harmonic expectation, potential/sea changes')


def local_loop_consumer(frame: np.ndarray, generators: np.ndarray,
                        screen_eta: float, samples: int) -> dict[str, Any]:
    """Actual centre seam on ordinary f=1 charged cloud, f=0 RIGHT.

    Telescoping identity links leave one boundary V=Spin(theta)^dagger.
    Interior word length cannot remove this crossing or its source zero.
    """
    spin=-.5j*SELECTED[1]@SELECTED[2]
    angles=np.linspace(0,2*math.pi,samples+1)
    values=[]; derivatives=[]; scalar_gradients=[]
    for angle in angles:
        chart=expm(1j*angle*spin)
        v=chart.conj().T
        scalar,gradient=centre_scalar(v,screen_eta,generators)
        coefficient=2*np.einsum('aij,ji->a',generators,-spin).real
        ds=np.dot(coefficient,gradient)
        values.append(chart*scalar)
        derivatives.append(1j*spin@chart*scalar+chart*ds)
        scalar_gradients.append(float(np.linalg.norm(gradient)))
    start,end=values[0],values[-1]
    require(np.linalg.norm(end-start),'ordinary actual closed2pi source +I')
    midpoint=values[samples//2]
    require(np.linalg.norm(midpoint),'actual centre-seam source zero')
    wrong=float(np.linalg.norm(end+start)**2)
    # ODE horizontal contribution in source-end interaction picture: ds/dtheta.
    scalar_values=np.asarray([centre_scalar(expm(-1j*t*spin),screen_eta,generators)[0] for t in angles])
    horizontal_length=float(np.sum(abs(np.diff(scalar_values))))
    labels=np.asarray([np.eye(4),*(SELECTED[j]@SELECTED[k]
                                 for j,k in ((2,3),(3,1),(1,2)))])
    thin_rows=[]
    for length in (8,80,800):
        connection=np.linspace(1,0,length+1)
        dword=sum(1j*(connection[j]-connection[j+1])*spin for j in range(length))
        thin_horizontal=-labels@dword
        norm=float(np.linalg.norm(thin_horizontal)**2/2)
        require(abs(norm-2),'actual thinWilson length-independent CT horizontal norm2')
        thin_rows.append(dict(length=length,thin_full_quartet_CT_horizontal_norm_squared=norm,
            screened_initial_G_scalar=2*abs(scalar_values[0])**(2*length),
            screened_normalized_horizontal_variation=horizontal_length/abs(scalar_values[0])))
    return dict(samples=samples,endpoint_plus_error=float(np.linalg.norm(end-start)),
        intrinsic_minus_defect_squared=wrong,seam_source_norm=float(np.linalg.norm(midpoint)),
        intrinsic_minus_CT_metric_defect_squared=wrong/2,
        thin_word_adverse_controls=thin_rows,
        source_initial_norm_squared=float(np.linalg.norm(start)**2),
        angular_full15_max=max(scalar_gradients),horizontal_scalar_total_variation=horizontal_length,
        horizontal_variation_lower=2*abs(scalar_values[0]),
        source_Noether_full_derivative_max=max(float(np.linalg.norm(d)) for d in derivatives),
        spin_J2_or_minusI_admitted=False,
        reason='closed ordinary Phi loop crosses scalar zero; initial metric cannot normalize a spin intertwiner across it')


def dynamical_cloud_consumer(law: Any, frame: np.ndarray, screen_eta: float) -> dict[str, Any]:
    """Different route: dynamical Yukawa-screened field AND electric-link cloud.

    Solve actual mass-horizontal screened response, then corotate links with
    the resultant continuous profile.  No endpoint or neutral-history reset.
    """
    fields=np.tile(frame,(law.volume,1,1,1)); links=np.tile(np.eye(4),(law.volume,3,1,1))
    cloud=law.index[(1,1,1)]; reference=0
    window=np.zeros(law.volume); window[cloud]=1
    ell, report=mechanical_connection(law,frame,window)
    spin=-.5j*SELECTED[1]@SELECTED[2]
    profile=np.asarray([np.vdot(spin,q).real/np.vdot(spin,spin).real for q in ell])
    eta=np.zeros(law.volume); eta[cloud]=1
    original=screened_marks(law,fields,links,np.zeros((law.volume,3)),eta,reference,screen_eta)
    endpoint_gauge=np.asarray([expm(2j*math.pi*f*spin) for f in profile])
    moved_fields=endpoint_gauge[:,None]@fields@endpoint_gauge[:,None].conj().swapaxes(-1,-2)
    moved_links=endpoint_gauge[:,None]@links@endpoint_gauge[law.target].conj().swapaxes(-1,-2)
    moved=screened_marks(law,moved_fields,moved_links,np.zeros((law.volume,3)),eta,reference,screen_eta)
    expected=np.einsum('xij,axjr,rk->axik',endpoint_gauge,original.reshape(4,law.volume,4,4),
                       endpoint_gauge[reference].conj().T).reshape(original.shape)
    error=require(np.linalg.norm(moved-expected),'dynamical screened field/link cloud source')
    # Its finite link kinetic norm is paid with ORIGINAL 1/.6 mass, not fixed EFT.
    tangent=1j*(ell[:,None]@links-links@ell[law.target])
    kinetic=float(2*np.linalg.norm(tangent)**2/.6)
    # Actual SAME-neutral exterior CAR transport, then complete Gauss identification.
    selected_sites=[cloud,reference,law.index[(1,0,0)]]
    old_small=original.reshape(4,law.volume,4,4)[:,selected_sites].reshape(4,12,4)
    new_small=moved.reshape(4,law.volume,4,4)[:,selected_sites].reshape(4,12,4)
    rows=tuple(itertools.combinations(range(12),6))
    charged=tuple(itertools.combinations(range(12),7))
    b0=maps_from_marks(old_small,rows,charged)
    b1=maps_from_marks(new_small,rows,charged)
    onebody=sparse.block_diag([sparse.csr_matrix(2*math.pi*profile[x]*spin)
                              for x in selected_sites]).toarray()
    gn=lift(onebody,rows); gc=lift(onebody,charged)
    rng=np.random.default_rng(22210653)
    neutral=rng.normal(size=len(rows))+1j*rng.normal(size=len(rows))
    neutral/=np.linalg.norm(neutral)
    solver: Any=expm_multiply
    moved_neutral=solver(1j*gn,neutral,traceA=1j*gn.diagonal().sum())
    old_columns=np.column_stack([b@neutral for b in b0]).reshape(4,len(charged),4)
    new_columns=np.column_stack([b@moved_neutral for b in b1]).reshape(4,len(charged),4)
    inverse_charged=solver(-1j*gc,new_columns.transpose(1,0,2).reshape(len(charged),16),
                           traceA=-1j*gc.diagonal().sum()).reshape(len(charged),4,4).transpose(1,0,2)
    restored=np.einsum('rj,jca->rca',endpoint_gauge[reference].T,inverse_charged)
    complete_error=require(np.linalg.norm(restored-old_columns),
                           'actual cloud fullCAR plus bar4 SAME-neutral Gauss identification')
    false_spin=float(np.linalg.norm(restored+old_columns))
    return dict(source_covariance_error=error,actual_screened_profile=profile,
        source_response=float(profile[cloud]),outside_reference_response=float(profile[reference]),
        link_cloud_mass_norm_squared=kinetic,field_endpoint_defect=float(np.linalg.norm(moved_fields-fields)),
        guard_link_endpoint_defect=float(np.linalg.norm(moved_links-links)),
        relative_source_minus_defect=float(np.linalg.norm(moved+original)),mechanical=report,
        actual_fullCAR_same_neutral_complete_Gauss_error=complete_error,
        wrong_source_end_only_minus_defect_after_Gauss=false_spin,
        decision='pure mechanical Gauss cloud is finite but vertical; cannot become physical spin by source-end rotation',
        selected_successor='normalized covariant wholeCartesian Haar orientation cloud from CentreSourceEnergy',
        unproved='physical angular holonomy of cloud relative SAME neutral; concentration does not change closed-loop +I')


def orientation_cloud_consumer(frame: np.ndarray, generators: np.ndarray,
                               screen_eta: float, samples: int = 512) -> dict[str, Any]:
    """Whole-Cartesian posterior cloud, finite full-Haar product quadrature.

    Coupled rotated nodes prove the quadrature covariance identity, NOT its
    Haar convergence.  This route uses no polar chart or frame-domain guard.
    """
    rng=np.random.default_rng(22210652)
    raw=rng.normal(size=(samples,2,4,4))+1j*rng.normal(size=(samples,2,4,4))
    nodes,triangular=np.linalg.qr(raw)
    diagonal=np.diagonal(triangular,axis1=-2,axis2=-1)
    nodes*= (diagonal/abs(diagonal))[:,:,None,:]
    nodes[:,:,:,0]/=np.linalg.det(nodes)[:,:,None]
    spins=np.asarray([-.5j*SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2))])
    labels=np.asarray([np.eye(4),*(2j*s for s in spins)])
    chirality=SELECTED[0]@SELECTED[1]@SELECTED[2]@SELECTED[3]
    nodes=np.concatenate((nodes,nodes@chirality),axis=0)
    phi=np.tile(frame,(2,1,1,1))
    phi+=.06*np.einsum('xma,aij->xmij',rng.normal(size=(2,4,15)),generators)
    link=expm(.37j*np.einsum('a,aij->ij',rng.normal(size=15),generators))
    alpha=2.
    templates=np.tile(frame,(2,1,1,1))
    def evaluate(fields: np.ndarray, u: np.ndarray, gs: np.ndarray):
        jets=[orientation_cloud(fields,u[None],q,screen_eta,alpha,gs,templates)
              for q in labels]
        return 2*np.asarray([jet['mean'] for jet in jets]),jets
    value,jets=evaluate(phi,link,nodes)
    sign_partner,sign_jets=evaluate(-phi,link,nodes)
    ct_error=require(np.linalg.norm(sign_partner-value),
                     'actual paired-Haar cloud globally Phi-sign CT even')
    ct_gradient_error=require(np.linalg.norm(np.asarray([jet['Cartesian_gradient'] for jet in sign_jets])
        +np.asarray([jet['Cartesian_gradient'] for jet in jets])),
        'actual whole120Cartesian paired cloud CT-odd derivatives')
    gauge=np.asarray([expm(.5j*np.einsum('a,aij->ij',rng.normal(size=15),generators)) for _ in range(2)])
    moved_phi=gauge[:,None]@phi@gauge[:,None].conj().swapaxes(-1,-2)
    moved,_=evaluate(moved_phi,gauge[0]@link@gauge[1].conj().T,gauge[None]@nodes)
    covariance=require(np.linalg.norm(moved-gauge[0]@value@gauge[1].conj().T),
                       'wholeCartesian screened cloud matched fullHaar-node covariance')
    direction=rng.normal(size=(2,4,15))
    df=np.einsum('xma,aij->xmij',direction,generators)
    gradients=2*np.asarray([jet['Cartesian_gradient'] for jet in jets])
    analytic=np.einsum('axcij,xc->aij',gradients,direction.reshape(2,60)/math.sqrt(8))
    step=1e-5
    fd=(evaluate(phi+step*df,link,nodes)[0]-evaluate(phi-step*df,link,nodes)[0])/(2*step)
    error=require(np.linalg.norm(fd-analytic),'wholeCartesian posterior covariance source derivative')
    zero,_=evaluate(np.zeros_like(phi),link,nodes)
    gram=np.einsum('aij,bij->ab',value.conj(),value)/2
    # The entire neutral exterior CAR and complete bar4 consume cloud coefficients.
    marks=np.concatenate((value,np.zeros((4,8,4))),axis=1)
    rows=tuple(itertools.combinations(range(12),6))
    charged=tuple(itertools.combinations(range(12),7))
    creators=maps_from_marks(marks,rows,charged)
    neutral=rng.normal(size=len(rows))+1j*rng.normal(size=len(rows))
    neutral/=np.linalg.norm(neutral)
    partner=particle_hole(rows,12)@neutral.conj()
    columns=[np.column_stack([b@v for b in creators]) for v in (neutral,partner)]
    actual=sum(c.conj().T@c for c in columns)/2
    car=require(np.linalg.norm(actual-gram),'actual orientation cloud completeCAR initial metric')
    # Consume supplier's complete120Cartesian/all15 link jets at this same G0.
    eigenvalues,eigenmodes=np.linalg.eigh(gram)
    inverse=(eigenmodes/np.sqrt(eigenvalues))@eigenmodes.conj().T
    link_gradients=2*np.asarray([jet['link_gradient'] for jet in jets])
    phi_form=inverse@np.einsum('axcij,bxcij->ab',gradients.conj(),gradients)@inverse/2
    link_form=inverse@np.einsum('aecij,becij->ab',link_gradients.conj(),link_gradients)@inverse/2
    link_direction=np.einsum('a,aij->ij',rng.normal(size=15),generators)
    coefficients=2*np.einsum('aij,ji->a',generators,link_direction).real
    link_analytic=np.einsum('aecij,c->aij',link_gradients,coefficients)
    link_fd=(evaluate(phi,expm(1j*step*link_direction)@link,nodes)[0]
             -evaluate(phi,expm(-1j*step*link_direction)@link,nodes)[0])/(2*step)
    link_error=require(np.linalg.norm(link_fd-link_analytic),'actual original all15 cloud link derivative')
    physical_source_rows=[]
    for axis,(j,k) in enumerate(((2,3),(3,1),(1,2))):
        tangent=np.zeros_like(phi); tangent[0,j]=-phi[0,k]; tangent[0,k]=phi[0,j]
        tangent_coordinates=coordinates(tangent,math.sqrt(8)*generators).reshape(2,60)
        contracted=np.einsum('axcij,xc->aij',gradients,tangent_coordinates)
        numeric=(evaluate(phi+step*tangent,link,nodes)[0]
                 -evaluate(phi-step*tangent,link,nodes)[0])/(2*step)
        physical_error=require(np.linalg.norm(numeric-contracted),
            'actual Haar wholeCartesian jets contracted with physical Phi vector tangent')
        raw=inverse@np.einsum('aij,bij->ab',value.conj(),-1j*contracted)/2@inverse
        derivative=maps_from_marks(np.concatenate((-1j*contracted,np.zeros((4,8,4))),
                                                axis=1),rows,charged)
        crossed=sum(c.conj().T@np.column_stack([b@v for b in derivative])
                    for c,v in zip(columns,(neutral,partner)))/2
        car_error=require(np.linalg.norm(inverse@crossed@inverse-raw),
            'actual fullCAR physical source Lie derivative at frozen initial Gram')
        physical_source_rows.append(dict(axis=axis,
            actual_physical_Cartesian_source_derivative_error=physical_error,
            entire_CAR_initial_metric_contraction_error=car_error,
            primitive_source_Lie_derivative_initial_matrix=raw,
            source_coefficient_derivative_not_projected_total_Noether=True))
    loop_phi=phi.copy()
    loop_phi[0]=expm(2j*math.pi*spins[2])@phi[0]@expm(-2j*math.pi*spins[2])
    endpoint=evaluate(loop_phi,link,nodes)[0]
    loop=require(np.linalg.norm(endpoint-value),'wholeCartesian cloud SAME ordinary2pi +')
    return dict(base_fullSU4_product_Haar_samples=samples,paired_total_nodes=len(nodes),alpha=alpha,
        wholeCartesian_Phi_sign_CT_error=ct_error,
        whole120Cartesian_CT_gradient_error=ct_gradient_error,
        ordered_source_derivative_majorants=orientation_cloud_derivative_majorants(
            1,screen_eta,alpha,float(np.linalg.norm(frame)**2/4),float(eigenvalues[0])),
        matched_node_gauge_error=covariance,Cartesian_derivative_error=error,
        supplier_all15_link_derivative_error=link_error,
        actual_physical_Cartesian_source_flow_contractions=physical_source_rows,
        all120Cartesian_initial_metric_energy_matrix=phi_form,
        all15link_initial_metric_energy_matrix=link_form,
        original_slow_extra_energy_per_chi=.5*phi_form+.3*link_form,
        fullCartesian_Laplacian=2*np.asarray([jet['Cartesian_Laplacian'] for jet in jets]),
        full15link_Laplacian=2*np.asarray([jet['link_Laplacian'] for jet in jets]),
        initial_G=gram,initial_G_eigenvalues=np.linalg.eigvalsh(gram),
        actual_completeCAR_error=car,zeroPhi_source_finite_norm=float(np.linalg.norm(zero)),
        ordinary2pi_plus_error=loop,
        source_not_thinWilson=True,polar_guard_used=False,
        finite_quadrature_not_Haar_convergence_proof=True,
        physical_halfspin_proved=False,
        decision='different entire dynamical orientation cloud survives source/domain checks, but cannot erase the closed ordinary-loop obstruction')


def physical_cocycle_and_flux_consumer(law: Any, frame: np.ndarray,
                                      screen_eta: float) -> dict[str, Any]:
    """Physical cube extension and exact original-Gauss boundary charge.

    Global restored rotations are ACTUAL physical orbital/vector/link/photon
    operations. Their separated centre flux character is exact; individual
    localized physical Noether operators still require an energy/clustering
    estimate and are not inferred from this algebra.
    """
    fields=np.tile(frame,(law.volume,1,1,1))
    links=np.tile(np.eye(4),(law.volume,3,1,1)).astype(complex)
    a=np.zeros((law.volume,3))
    cloud=law.index[(1,1,1)]; reference=0
    eta=np.zeros(law.volume); eta[cloud]=1
    marks=screened_marks(law,fields,links,a,eta,reference,screen_eta)
    gram=np.einsum('mij,nji->mn',frame,frame).real/4
    rx=np.diag([1,-1,-1]); ry=np.diag([-1,1,-1])
    h=law.onebody(fields,links,a)
    rng=np.random.default_rng(22210654)
    photon=.03*rng.normal(size=a.shape)
    electric=.04*rng.normal(size=a.shape)
    operations=[]; references=[]; errors=[]
    world_f,world_u,world_a=fields.copy(),links.copy(),a.copy()
    world_photon,world_electric=photon.copy(),electric.copy()
    for rotation in (rx,ry,rx,ry):
        lift_r=make_lift(law,np.zeros(3,dtype=int),rotation,gram,frame)
        physical_f,physical_u,world_a,permutation=physical_cubic(
            law,world_f,world_u,world_a,rotation)
        _,_,world_photon,_=physical_cubic(law,world_f,world_u,world_photon,rotation)
        _,_,world_electric,_=physical_cubic(law,world_f,world_u,world_electric,rotation)
        local=sparse.block_diag([sparse.csr_matrix(g) for g in lift_r.local]).toarray()
        operation=local@permutation
        old_h=law.onebody(world_f,world_u,a)
        world_f=lift_r.fields(physical_f)
        world_u=restore_links(law,lift_r,physical_u)
        new_h=law.onebody(world_f,world_u,world_a)
        errors.append(require(np.linalg.norm(new_h-operation@old_h@operation.conj().T),
                              'actual physical pi orbital plus fullGauss native intertwiner'))
        operations.append(operation)
        references.append(lift_r.local[reference])
    onebody_cocycle=np.eye(len(h),dtype=complex)
    reference_cocycle=np.eye(4,dtype=complex)
    for operation,ref in zip(operations,references):
        onebody_cocycle=operation@onebody_cocycle
        reference_cocycle=ref@reference_cocycle
    require(np.linalg.norm(onebody_cocycle+np.eye(len(h))),
            'derived physical orthogonalpi charged-leg cocycle centreminus')
    require(np.linalg.norm(reference_cocycle+np.eye(4)),
            'actual bar4 reference orthogonalpi cocycle centreminus')
    world_error=require(np.linalg.norm(world_f-fields)+np.linalg.norm(world_u-links)
        +np.linalg.norm(world_a-a)+np.linalg.norm(world_photon-photon)
        +np.linalg.norm(world_electric-electric),'actual complete physical world cocycle closure')
    # Whole native CAR: occupied exterior determinant, one added creation mode.
    values,modes=np.linalg.eigh(h)
    occupied=modes[:,values<0]
    p=occupied@occupied.conj().T
    neutral_phase=np.linalg.det(occupied.conj().T@onebody_cocycle@occupied)
    require(abs(neutral_phase-1),'SAME whole54mode neutral determinant cocycle')
    positive=marks-np.einsum('ij,ajr->air',p,marks)
    initial=source_gram(marks,p)
    vals,vecs=np.linalg.eigh(initial)
    inverse=(vecs/np.sqrt(vals))@vecs.conj().T
    moved=np.einsum('ij,ajr->air',onebody_cocycle,positive)
    charged_overlap=inverse@np.einsum('air,bir->ab',positive.conj(),moved)@inverse*neutral_phase
    charged_error=require(np.linalg.norm(charged_overlap+np.eye(4)),
                          'actual fullCAR charged-minus-SAME-neutral orthogonalpi cocycle')
    paired=np.einsum('air,rj->aij',moved,reference_cocycle.conj().T)
    paired_overlap=inverse@np.einsum('air,bir->ab',positive.conj(),paired)@inverse*neutral_phase
    pair_error=require(np.linalg.norm(paired_overlap-np.eye(4)),
                       'actual fullCAR fullbar4 global physical cocycle plus')
    # Centre boundary flux is a real SU4 link translation, not label relabelling.
    region=np.zeros(law.volume,dtype=bool); region[cloud]=True
    centre=np.where(region,-1.,1.)
    shifted_links=centre[:,None,None,None]*links*centre[law.target][:,:,None,None]
    flux_marks=screened_marks(law,fields,shifted_links,a,eta,reference,screen_eta)
    flux_error=require(np.linalg.norm(flux_marks+marks),
                       'actual wholeSU4 boundary electric centreflux q1 source character')
    hflux=law.onebody(fields,shifted_links,a)
    z=np.repeat(centre,4)
    hflux_error=require(np.linalg.norm(hflux-z[:,None]*h*z[None,:]),
                        'original fullGauss centreflux matter boundary identity')
    # Complete operator Ward on all neutral/charged exterior rows; no parity
    # eigenstate or manually selected initial source occupancy is assumed.
    sites=[cloud,reference,law.index[(1,0,0)]]
    small=marks.reshape(4,law.volume,4,4)[:,sites].reshape(4,12,4)
    flux_small=flux_marks.reshape(4,law.volume,4,4)[:,sites].reshape(4,12,4)
    rows=tuple(itertools.combinations(range(12),6))
    charged=tuple(itertools.combinations(range(12),7))
    b0=maps_from_marks(small,rows,charged)
    bf=maps_from_marks(flux_small,rows,charged)
    pn=sparse.diags([(-1)**sum(mode<4 for mode in row) for row in rows],dtype=float)
    pc=sparse.diags(np.tile([(-1)**sum(mode<4 for mode in row) for row in charged],4),dtype=float)
    ward=max(float(sparse.linalg.norm(f@pn-pc@b)) for f,b in zip(bf,b0))
    require(ward,'actual completeCAR boundaryflux times matter Gauss Ward')
    fixed_neutral_cost=float(np.trace(p@(hflux-h)).real)
    return dict(physical_pi_axes_commute=True,spatial_vectors_links_photons_reference_rotated=True,
        physical_native_intertwiner_errors=errors,full_world_cocycle_closure_error=world_error,
        native_CAR_modes=len(h),neutral_occupied_modes=len(occupied.T),
        SAME_neutral_exterior_determinant_cocycle=neutral_phase,
        charged_leg_CAR_cocycle_error=charged_error,global_fullbar4_cocycle_error=pair_error,
        original_boundary_flux_q1_error=flux_error,fullCAR_Gauss_flux_charge_Ward_error=ward,
        original_boundary_flux_H_covariance_error=hflux_error,
        flux_alone_native_H_defect=float(np.linalg.norm(hflux-h)),
        flux_alone_fixed_SAME_neutral_energy_cost=fixed_neutral_cost,
        localized_individual_physical_Noether_operators_admitted=False,
        unpaid='original deconfinedGamma well plus separated-region clustering/boundary angular localization estimate; global lift and exact Gauss charge alone do not supply it',
        selected_protocol='actual cubic physical rotation followed by boundary-centreflux interferometry against SAME neutral; individual localization remains conditional')


def dynamical_local_frame_protocol(law: Any, frame: np.ndarray,
                                  screen_eta: float) -> dict[str, Any]:
    """Actual native closed-Phi drive and positive-band Kato Berry protocol.

    No source is rebuilt/re-normalized during the loop. Full charged and SAME
    neutral CAR transport use the same time-dependent native Hamiltonian.
    This externally prescribed frame protocol is not an Euler cloud solution
    or a microscopic spatial SO3 rotation on a finite cubic grid.
    """
    fields=np.tile(frame,(law.volume,1,1,1))
    links=np.tile(np.eye(4),(law.volume,3,1,1))
    a=np.zeros((law.volume,3))
    cloud=law.index[(1,1,1)]; reference=0
    eta=np.zeros(law.volume); eta[cloud]=1
    source=screened_marks(law,fields,links,a,eta,reference,screen_eta)
    h0=law.onebody(fields,links,a)
    e0,v0=np.linalg.eigh(h0)
    occupied=v0[:,e0<0]; positive=v0[:,e0>0]@v0[:,e0>0].conj().T
    prepared=np.einsum('ij,ajr->air',positive,source)
    metric=np.einsum('air,bir->ab',prepared.conj(),prepared)
    vals,vecs=np.linalg.eigh(metric)
    initial_inverse=(vecs/np.sqrt(vals))@vecs.conj().T
    def physical_fields(angle: float) -> np.ndarray:
        moved=fields.copy()
        moved[cloud,1]=math.cos(angle)*frame[1]-math.sin(angle)*frame[2]
        moved[cloud,2]=math.sin(angle)*frame[1]+math.cos(angle)*frame[2]
        return moved
    rows=[]; previous_kato=None; previous_dynamic=None
    physical_coordinate_norm=float(np.linalg.norm(frame[1:3])**2/4)
    for steps in (32,64):
        kato=np.eye(len(h0),dtype=complex)
        dynamic=np.eye(len(h0),dtype=complex)
        prior=positive
        min_gap=float(min(abs(e0))); max_projector_step=0.
        duration=.4; dt=duration/steps
        alignment_W_max=0.; fixed_neutral_sea_shift_max=0.
        for n in range(1,steps+1):
            angle=2*math.pi*n/steps
            moved_fields=physical_fields(angle)
            h=law.onebody(moved_fields,links,a)
            transported=links[:,:,None]@moved_fields[law.target]@links[:,:,None].conj().swapaxes(-1,-2)
            alignment_W_max=max(alignment_W_max,float(np.linalg.norm(moved_fields[:,None]-transported)**2))
            values,modes=np.linalg.eigh(h)
            fixed_neutral_sea_shift_max=max(fixed_neutral_sea_shift_max,
                float(np.sum(values[values<0])-np.sum(e0[e0<0])))
            min_gap=min(min_gap,float(min(abs(values))))
            if np.sum(values>0)!=len(h)//2 or min_gap<=1e-8:
                raise ValueError('actual local-frame Berry path loses positive-band gap')
            current=modes[:,values>0]@modes[:,values>0].conj().T
            overlap=current@prior+(np.eye(len(h))-current)@(np.eye(len(h))-prior)
            left,singular,right=np.linalg.svd(overlap)
            if singular[-1]<=1e-8:
                raise ValueError('Kato step needs refinement before comparison')
            max_projector_step=max(max_projector_step,float(np.linalg.norm(current-prior,ord=2)))
            kato=(left@right)@kato; prior=current
            midpoint=2*math.pi*(n-.5)/steps
            hm=law.onebody(physical_fields(midpoint),links,a)
            em,vm=np.linalg.eigh(hm)
            propagator=(vm*np.exp(-1j*dt*em))@vm.conj().T
            dynamic=propagator@dynamic
        berry=np.einsum('ij,ajr->air',kato,prepared)
        neutral_berry=np.linalg.det(occupied.conj().T@kato@occupied)
        # Exact determinant/Schur-complement charged-minus-SAME-neutral overlap.
        neutral_block=occupied.conj().T@dynamic@occupied
        neutral_amplitude=np.linalg.det(neutral_block)
        schur=dynamic-dynamic@occupied@np.linalg.solve(neutral_block,occupied.conj().T@dynamic)
        relative=np.einsum('ij,ajr->air',schur,prepared)
        dynamic_overlap=initial_inverse@np.einsum('air,bir->ab',prepared.conj(),relative)@initial_inverse
        berry_overlap=initial_inverse@np.einsum('air,bir->ab',prepared.conj(),berry)@initial_inverse
        unitary_error=require(np.linalg.norm(dynamic.conj().T@dynamic-np.eye(len(h0))),
                              'actual same-H native driven charged neutral unitary')
        rows.append(dict(steps=steps,duration=duration,actual_gap_min=min_gap,
            Kato_projector_step_max=max_projector_step,actual_native_unitary_error=unitary_error,
            actual_alignment_W_max=alignment_W_max,
            original_alignment_energy_max_per_Lambda_1p25=.2/8*alignment_W_max,
            fixed_neutral_sea_shift_max=fixed_neutral_sea_shift_max,
            unprojected_field_drive_energy_times_chi=.5*(2*math.pi/duration)**2*physical_coordinate_norm,
            SAME_neutral_Berry_amplitude=neutral_berry,SAME_neutral_dynamic_amplitude=neutral_amplitude,
            fullbar4_relative_Berry_overlap=berry_overlap,
            fullbar4_relative_dynamic_overlap=dynamic_overlap,
            full_source_Berry_plus_defect=float(np.linalg.norm(berry-prepared)),
            full_source_Berry_minus_defect=float(np.linalg.norm(berry+prepared)),
            Berry_mesh_change=None if previous_kato is None else float(np.linalg.norm(kato-previous_kato)),
            dynamic_mesh_change=None if previous_dynamic is None else float(np.linalg.norm(dynamic-previous_dynamic))))
        previous_kato=kato; previous_dynamic=dynamic
    closure=require(np.linalg.norm(law.onebody(physical_fields(2*math.pi),links,a)-h0),
                    'actual driven Phi native world closes without source seam reset')
    return dict(protocol='physical localPhi frame loop; exact native matter Schrodinger and Kato positive band',
        closed_native_world_error=closure,RIGHT_fixed=True,source_rebuilt_or_final_metric_inverse=False,
        full15_Phi_Noether_scope='same physical tangent already measured in native_consumer; external drive and photon/field backreaction not solved',
        rows=rows,physical_local_halfspin_admitted=False,
        unpaid='coupled dynamical field/photon cloud solution, adiabatic gap/mesh extrapolation and individual physical symmetry localization',
        distinction='a dynamically transported state need not share the smooth source coefficient loop; neither its Berry phase nor source-end spin is assumed')


def true_spectral_field_graph(energy_shifted: float, chi: float,
                              quartic_coercivity: float, quartic_constant: float,
                              cometric_laplacian_constant: float,
                              Cartesian_dimension: int) -> dict[str, float]:
    """Actual full spectral graph, not a product of separate first moments.

    H_R=T+V>=0, V>=a*rho^4-c and sum t_i D_i^2 V<=D(1+rho^2).
    Arguments are coefficients of the ORIGINAL shifted operator, not fitted
    ground-state data. The proof and native coefficient accounting are in §10.
    """
    if min(energy_shifted, quartic_constant, cometric_laplacian_constant) < 0:
        raise ValueError('nonnegative shifted energy/coercivity constants required')
    if chi <= 0 or quartic_coercivity <= 0 or Cartesian_dimension < 1:
        raise ValueError('positive physical cometric/quartic coercivity required')
    rho_fourth=(energy_shifted+quartic_constant)/quartic_coercivity
    rho_squared_graph=math.sqrt(rho_fourth)
    T_graph=math.sqrt(energy_shifted**2
                     +cometric_laplacian_constant*(1+rho_squared_graph))
    momentum_squared_graph=2*T_graph/chi
    weighted_Phi_momentum=math.sqrt(
        momentum_squared_graph*rho_squared_graph+Cartesian_dimension)
    return dict(rho_fourth_moment=rho_fourth, rho_squared_graph=rho_squared_graph,
        total_kinetic_graph=T_graph, Cartesian_Laplacian_graph=momentum_squared_graph,
        actual_Phi_momentum_graph=weighted_Phi_momentum)


def physical_spectral_filter_bound(gap: float, energy_window: float,
                                   duration: float, actual_J_graph: float) -> dict[str, Any]:
    """Same compact r8 clock on TRUE spectral blocks of the physical H.

    No compressed normal complement is silently substituted for a spectral
    projector. Light-to-light/virtual-normal matrix elements are not removed.
    """
    if gap <= energy_window or min(duration, actual_J_graph) <= 0:
        raise ValueError('separated actual spectral blocks and finite J graph required')
    pulse=SmoothPulse(order=8)
    derivative=pulse.derivatives[8]
    L1_upper=float(sum(abs(c)/(j+1) for j,c in enumerate(derivative.coef)))
    upper=min(1.,L1_upper/((gap-energy_window)*duration)**8)*actual_J_graph
    return dict(high_actual_spectral_block_bound=upper,
        eighth_derivative_L1_upper=L1_upper, clock_order=8,
        high_projector='1_[gap,infinity)(H-E0)', low_projector='1_[0,E](H-E0)',
        compressed_normal_projector_used=False,
        low_low_projected_matrix_not_deduced_from_this_bound=True)


def refined_noether_transfer_selector(budget: Any, error: Any,
                                      field_link_velocity_coefficient: Any,
                                      photon_orbital_reference_graph: Any,
                                      source_norm: Any = 1,
                                      energy_window: Any = 1) -> dict[str, Any]:
    """Exact local selector for the weighted RNC/physical-Noether terms in §12.

    Global exterior coercivity, source concentration, actual photon/orbital
    graph and lattice/energy tails are separate inputs, not inferred here.
    """
    from fractions import Fraction
    from noncompact_quantitative_normal_budget import _json, ceil_log2, rational
    error,Y,ph,source,E=map(rational,(error,field_link_velocity_coefficient,
                                   photon_orbital_reference_graph,source_norm,energy_window))
    if min(error,Y,source) <= 0 or min(ph,E) < 0:
        raise ValueError('actual finite physical graph coefficients and positive target required')
    normal_gap=budget.frequency/8
    heavy_gap=budget.heavy_gap/4
    bounded=16*budget.comparison
    normal=16*budget.comparison*(budget.momentum+1)
    common=16*budget.comparison*source*(1+Y+ph)*(
        1+budget.vacuum+budget.exterior_light_constant+E)
    normal_resolvent=common*(1+1/normal_gap)**2
    heavy_resolvent=common*(1+1/heavy_gap)**2
    normal_leading=normal_resolvent*normal
    heavy_leading=heavy_resolvent*budget.residual_kinetic
    normal_other=normal_resolvent*(budget.residual_light+bounded)
    k=max(budget.local_log2_t,(ceil_log2(16*normal_leading/error)+5)//6,
          (ceil_log2(16*normal_other/error)+9)//10,
          (ceil_log2(16*heavy_leading/error)+24)//25,
          (ceil_log2(16*heavy_resolvent*bounded/error)+59)//60,
          (ceil_log2(64*(E+budget.exterior_light_constant+1)/budget.frequency)+19)//20)
    t=1<<max(1,k)
    normal_upper=normal_leading/Fraction(t**6)+normal_other/Fraction(t**10)
    heavy_upper=heavy_leading/Fraction(t**25)+heavy_resolvent*bounded/Fraction(t**60)
    if normal_upper+heavy_upper > error:
        raise AssertionError('weighted Noether selector misses its own local target')
    return _json(dict(local_Lambda_expression=f'2^({80*max(1,k)})',
        local_log2_Lambda=80*max(1,k),
        normal_RNC_residual_expression='C_RN*Lambda^(-21/80)+C_L*Lambda^(-5/16)+C_B*Lambda^(-1/2)',
        normal_RNC_residual_coefficient=normal,
        normal_weighted_Noether_upper=normal_upper,
        localized_heavy_weighted_Noether_upper=heavy_upper,
        normal_mix_exponent='-3/40', localized_heavy_mix_exponent='-5/16',
        exterior_safe_heavy_mix_exponent='-9/80',
        exterior_safe_heavy_coefficient=dict(prefactor=common,coercivity_cost='(1+c_phase^(-1))^2'),
        required_exterior_global_selector='pay actual all64 c_phase and IMS/Agmon polynomial in addition to local exponent',
        actual_RNC_connection_zero_required=True,
        vertical_Gauss_charge_energy_retained=True,
        unbounded_normal_heavy_cross_treated_as_closed_form=True,
        full_physical_Noether_admission=False))


def orientation_core_halfspin_bound(length: int, eta: float, gram_min: float,
                                    frame_squared: float, error: float) -> dict[str, Any]:
    """Finite full-Haar concentration, source Gram and physical-core spin bound.

    A Gaussian-to-Haar QR small-ball proof supplies a conservative r^32
    mass lower bound. Nearest centres occur ONLY in the proof, not the source.
    The source remains the supplier's smooth full-group orientation integral.
    """
    if length < 1 or min(eta,gram_min,frame_squared,error) <= 0 or error >= 1:
        raise ValueError('finite full-Haar core parameters and error in (0,1) required')
    log_scalar=math.log1p(-2/(math.exp(4*eta)+1))
    core_amplitude=math.exp(length*log_scalar)
    if core_amplitude <= 0:
        raise ValueError('choose eta so the finite path core amplitude is representable')
    C=2*math.sqrt(length+1)*(1+64*eta*length)
    radius=error*core_amplitude/(64*C)
    log_Haar_ball=-16*math.log(math.pi)-9+32*math.log(2/(4096*math.sqrt(32)))
    exponent=0
    while True:
        alpha=2.**exponent
        log_tail=-log_Haar_ball+2*frame_squared+16*math.log(alpha)-alpha*gram_min*radius**2/100
        if log_tail <= math.log(radius**2/16):
            break
        exponent+=1
    tail=math.exp(log_tail) if log_tail > -745 else math.ulp(0.)
    distance_second=radius**2+16*tail
    perturbation=C*math.sqrt(distance_second)
    singular_lower=core_amplitude-perturbation
    spin_error=4*perturbation/singular_lower
    if singular_lower <= 0 or spin_error > error:
        raise AssertionError('finite Haar concentration failed source metric/spin target')
    return dict(alpha_expression=f'2^{exponent}', alpha=alpha, path_edges=length,
        centre_packet_eta=eta, core_amplitude=core_amplitude,
        Haar_ball_log_lower_coefficient=log_Haar_ball,
        original_whole_group_near_centre_radius=radius,
        posterior_tail_log_upper=log_tail, mark_operator_perturbation=perturbation,
        q_over_2_CT_Gram_lower=singular_lower**2/2,
        physical_core_halfspin_generator_error=spin_error,
        finite_alpha_field_link_jets_remain_payable=True,
        not_a_nearest_centre_source=True,
        normal_field_link_variation_and_photon_orbital_tails_excluded=True)


def physical_raw_noether_geometry_majorant(outer_radius: float = 64.,
                                         spacing: float = 1.) -> dict[str, Any]:
    """Original local Phi/vector, full15 plaquette and literal sine-flow jets.

    This bounds the UNPROJECTED vector field, not total angular variance.
    Phi coefficients have linear growth; compact U and sine derivatives are
    bounded. A Gauss-vertical change of representative is unnecessary.
    """
    if outer_radius <= 0 or spacing <= 0:
        raise ValueError('positive actual support and native spacing required')
    side=2*math.ceil(outer_radius/spacing+8)+1
    coordinates=108*side**3
    root=math.isqrt(coordinates)+1
    # Each output uses fewer than 4096 input coordinates: at most 64
    # plaquette links *15, twelve transports *15, and seven Phi blocks *60.
    # q<=2 gives at most the square of this count. Products have length<=4;
    # norm(generator)<=1, |sin^(q)|<=1, and ||Phi_core_component||<=2.
    raw=16*root*4096**2*math.ceil(4096*(1+outer_radius/spacing))
    exponent=max(64,(raw-1).bit_length()+3)
    return dict(actual_raw_vector_q0_q1_q2_coefficient_upper=raw,
        selected_CY_expression=f'2^{exponent}', selected_CY_log2=exponent,
        active_coordinate_upper=coordinates, stencil_input_coordinate_upper=4096,
        field_growth='(1+rho) for raw vector jets; (1+rho^2) for two Lie words',
        core_Phi_component_HS_upper=2, derivative_halo=8,
        Gauss_vertical_projection_or_inverse_cometric_used=False,
        total_J_noise_or_variance_bounded_or_deleted=False,
        original_full15_and_literal_sine_flow_retained=True)


def orientation_core_point_source_bound(core: dict[str, Any],
                                        initial_norm_lower: float,
                                        neutral_photon_creation_upper: float = 1.) -> dict[str, Any]:
    """Uniform whole-Haar source VALUE error, including compact rotation poses.

    |Ds_eta|<=4eta in Hilbert-Schmidt norm on the WHOLE group. Centre
    characters telescope before estimating distances, so independent Z4
    branches never enter the error. This is not a photon crossed-Gram order.
    """
    if min(initial_norm_lower,neutral_photon_creation_upper) <= 0:
        raise ValueError('actual frozen source norm and creation bound required')
    log_tail=core['posterior_tail_log_upper']
    tail=math.exp(log_tail) if log_tail > -745 else math.ulp(0.)
    rms=math.sqrt(core['original_whole_group_near_centre_radius']**2+16*tail)
    coefficient=2+8*core['centre_packet_eta']*core['path_edges']
    raw=coefficient*rms
    return dict(canonical_q_over_2_source_HS_error_upper=raw,
        four_primitive_mark_stack_error_upper=4*raw,
        normalized_source_vector_comparison_upper=2*raw*
            neutral_photon_creation_upper/initial_norm_lower,
        whole_group_scalar_first_jet_HS_upper=4*core['centre_packet_eta'],
        centre_branches_cancelled_before_bound=True,
        source_carrier_HS_normalization=1,
        actual_two_detector_global_Gram_not_replaced=True,
        compact_rotation_pose_requires_same_posterior_concentration=True)


def physical_orientation_lie_majorant(core: dict[str, Any], profile_variation: float,
                                     initial_norm_lower: float,
                                     neutral_photon_creation_upper: float = 1.) -> dict[str, Any]:
    """Literal Phi core Lie jet, by a whole-Haar change of variables.

    The actual core vector tangent must equal i[S,Phi]; off-core Cartesian,
    clover, orbital and energy tails are separate, retained inputs.
    """
    if profile_variation < 0 or min(initial_norm_lower,neutral_photon_creation_upper) <= 0:
        raise ValueError('actual radial path variation and frozen norm required')
    eta=core['centre_packet_eta']
    tail=math.exp(core['posterior_tail_log_upper']) if core['posterior_tail_log_upper'] > -745 else math.ulp(0.)
    rms=math.sqrt(core['original_whole_group_near_centre_radius']**2+16*tail)
    hessian=4*eta+48*eta**2
    raw=(4+8*profile_variation*hessian)*rms
    return dict(full_Haar_physical_core_Lie_numerator_upper=raw,
        frozen_metric_physical_core_Lie_upper=raw*neutral_photon_creation_upper/initial_norm_lower,
        whole_SU4_scalar_Hessian_upper=hessian,
        radial_path_profile_total_variation=profile_variation,
        centre_scalar_traceless_first_jet_exactly_zero=True,
        fundamental_thin_link_first_jet_not_zero=True,
        physical_generator_inserted_as_end_factor=False,
        original_core_vector_tangent_and_actual_localization_required=True)


def _graph_log2(value: Any) -> float:
    if value == 0:
        return -math.inf
    if value < 0:
        raise ValueError('nonnegative physical graph required')
    if hasattr(value,'numerator') and hasattr(value,'denominator'):
        return math.log2(value.numerator)-math.log2(value.denominator)
    return math.log2(value)


def _graph_float(log_value: float) -> float | None:
    if log_value == -math.inf:
        return 0.
    return math.exp2(log_value) if -1074 <= log_value <= 1023 else None


def relative_photon_noether_graph(log2_UE: float, photon_speed: float,
                                  photon_energy_graph: float,
                                  velocity_max: float, spacing: float,
                                  phase_geometry_norm: float,
                                  initial_norm_lower: float,
                                  coexact_singular_lower: float | None = None) -> dict[str, Any]:
    """RAW source on SAME neutral photon history; retain its nonzero boundary.

    K curl(A)^2<=H_EM with K=c^2/(2UE) pays the original oscillator form.
    No total-J variance is suppressed, and no floating underflow substitutes
    zero coupling. The supplied graph is for the ACTUAL prepared photon state.
    """
    if min(photon_speed,spacing,initial_norm_lower) <= 0:
        raise ValueError('original positive photon coefficients and frozen norm required')
    if min(photon_energy_graph,velocity_max,phase_geometry_norm) < 0:
        raise ValueError('actual nonnegative energy and geometry graphs required')
    coefficient=8*velocity_max*phase_geometry_norm/(spacing*photon_speed*initial_norm_lower)
    if coexact_singular_lower is not None:
        if coexact_singular_lower <= 0:
            raise ValueError('actual positive coexact curl singular value required')
        coefficient+=12*math.sqrt(2)*velocity_max/(spacing*photon_speed*
            initial_norm_lower*coexact_singular_lower)
    coefficient_log=_graph_log2(coefficient)+.5*_graph_log2(photon_energy_graph)
    log_bound=.5*log2_UE+coefficient_log
    constant=_graph_float(coefficient_log)
    return dict(relative_photon_Noether_upper_log2=log_bound,
        relative_photon_Noether_coefficient_before_sqrt_UE=constant,
        relative_photon_Noether_coefficient_before_sqrt_UE_log2=coefficient_log,
        relative_photon_Noether_dyadic_upper_exponent=math.ceil(log_bound)+1 if math.isfinite(log_bound) else None,
        actual_UE_log2=log2_UE, exact_SAME_neutral_operator_noise_cancellation=True,
        Coulomb_Gauss_charge_and_harmonic_momentum_insertions_retained=True,
        literal_sine_clover_bounded_by_real_curl_not_replaced=True,
        raw_source_coexact_commutation_exact=True,
        independently_filtered_source_requires_its_full_nonzero_filter_jet=True,
        actual_scalar_orbital_coexact_A_debt_included=coexact_singular_lower is not None,
        photon_energy_graph_is_actual_prepared_state_input=True)


def weighted_photon_curl_graph(log2_UE: float, photon_speed: float,
                              rho_squared_graph: float, shifted_H_graph: float,
                              rho_squared_expectation: float,
                              rho_cubed_expectation: float, light_coefficient: float,
                              chi: float, state_norm: float = 1.) -> dict[str, Any]:
    """Actual mixed IMS graph, not a product of unrelated first moments.

    rho=sqrt(1+sum Phi^2), K_col-Emin>=-1, |H_light|<=B_L rho.
    rho commutes H_EM/H_light; the gradient IMS cost is chi*||psi||^2/2.
    """
    values=(rho_squared_graph,shifted_H_graph,rho_squared_expectation,
            rho_cubed_expectation,light_coefficient,chi,state_norm)
    if photon_speed <= 0 or min(values) < 0:
        raise ValueError('actual positive speed and finite nonnegative full graphs required')
    terms=(_graph_log2(rho_squared_graph)+_graph_log2(shifted_H_graph),
           _graph_log2(rho_squared_expectation),
           _graph_log2(light_coefficient)+_graph_log2(rho_cubed_expectation),
           _graph_log2(chi)+2*_graph_log2(state_norm)-1)
    maximum=max(terms)
    log_rhs=(maximum+math.log2(sum(math.exp2(term-maximum) for term in terms))
             if maximum != -math.inf else -math.inf)
    log_bound=.5*log2_UE+.5*(1+log_rhs)-math.log2(photon_speed)
    return dict(rho_curl_A_norm_upper_log2=log_bound,
        actual_mixed_IMS_rhs=_graph_float(log_rhs), actual_mixed_IMS_rhs_log2=log_rhs,
        rho_curl_A_dyadic_upper_exponent=math.ceil(log_bound)+1 if math.isfinite(log_bound) else None,
        full_H_minus_Emin_graph_required=True, K_col_lower_bound_minus_one_required=True,
        rho_squared_and_first_H_graphs_not_separate_first_moments=True,
        harmonic_momentum_shift_and_Coulomb_charge_not_discarded=True)


def ground_coexact_photon_number_graph(log2_UE: float, mode_count: int,
                                      frequency_min: float,
                                      current_squared_per_mode: float) -> dict[str, Any]:
    """Exact ACTUAL-ground pullthrough, with the original Peierls current.

    (Q+omega_k)a_k Omega=-sqrt(UE/(2omega_k))J_k Omega, Q>=0.
    This is not the first-order Ug replacement for Omega.
    """
    if mode_count < 1 or min(frequency_min,current_squared_per_mode) <= 0:
        raise ValueError('all actual coexact modes and current/energy graphs required')
    log_number=(log2_UE+math.log2(mode_count*current_squared_per_mode/2)-
                3*math.log2(frequency_min))
    return dict(actual_ground_total_coexact_number_upper_log2=log_number,
        normalized_mode_creation_norm_upper_log2=.5+.5*max(0.,log_number),
        creation_norm_upper_two_admitted=log_number <= 0,
        actual_ground_resolvent_Q_plus_omega=True,
        first_order_current_cloud_substituted_for_ground=False,
        all_modes_retained=mode_count,
        full_Phi_current_requires_actual_rho_squared_expectation=True)


def embedded_raw_radial_receiver(band: RotationBand) -> dict[str, Any]:
    """Supplier eta41 embedded on the SAME legal lattice, with an explicit halo.

    Only the scalar profile is embedded. No native41 normal/BO factor or
    detector metric is transferred. The moderate Bloch grid is a localization
    comparison for the fixed admitted Gamma, never a native Nstar simulation.
    """
    N=41
    sites=np.indices((N,)*3).reshape(3,-1).T
    geometry_law=SimpleNamespace(size=N,band=band,sites=sites)
    eta,guard,_=radial_profile(geometry_law,np.full(3,N//2),.25,2.4,10.,17.)
    halo=53
    source=np.zeros((halo,)*3,complex)
    source[6:47,6:47,6:47]=eta.reshape((N,)*3)
    positions=np.indices(source.shape).astype(float)-halo//2
    def orbital(axis: int, value: np.ndarray) -> np.ndarray:
        result=np.zeros_like(value)
        for j in range(3):
            velocity=np.cross(np.eye(3)[axis],positions.reshape(3,-1).T)[:,j].reshape(source.shape)
            result+=velocity*(np.roll(value,-1,axis=j)-np.roll(value,1,axis=j))/(2j*band.a)
        return result
    first=[orbital(j,source) for j in range(3)]
    second=sum((orbital(j,first[j]) for j in range(3)),np.zeros_like(source))
    algebra=[]
    for j,k,l in ((0,1,2),(1,2,0),(2,0,1)):
        algebra.append(float(np.linalg.norm(orbital(j,first[k])-orbital(k,first[j])-1j*first[l])))
    eta_cut_norm=float(np.linalg.norm(source))
    require(abs(eta_cut_norm-1),'actual embedded eta41 norm')
    mass_action=band.m*source
    spatial_squared=0.
    for j in range(3):
        mass_action+=band.r*(source-(np.roll(source,-1,axis=j)+np.roll(source,1,axis=j))/2)
        derivative=(np.roll(source,-1,axis=j)-np.roll(source,1,axis=j))/(2j)
        spatial_squared+=band.v**2*float(np.linalg.norm(derivative)**2)
    coefficient_energy_squared=float(np.linalg.norm(mass_action)**2)+spatial_squared
    # The actual all-toron Gamma enclosure has ||G||<1/2. For quaternion Q,
    # Tr(Q† Gamma_mu Q)=0, so P+ has norm²1/2 and the same energy RMS.
    particle_energy_upper=math.sqrt(.5*coefficient_energy_squared)
    halo_radius=math.sqrt(3)*(20+6)
    if halo_radius >= 48:
        raise AssertionError('six literal derivative halos escape the actual plateau')
    # Keep the common far-reference exp(-i D Ah) SYMBOLIC. Its local phase
    # differences cancel the original Peierls Ah and longitudinal edges.
    harmonic=np.asarray([.317,-.221,.163])
    chi=.03*np.sin(.17*positions[0]+.11*positions[1]-.13*positions[2])
    local_phase=np.exp(1j*(chi-np.einsum('j,jxyz->xyz',harmonic,positions)))
    dressed=source*local_phase
    cancellation=[]
    missing_phase=[]
    for axis in range(3):
        result=np.zeros_like(source)
        wrong=np.zeros_like(source)
        for j in range(3):
            velocity=np.cross(np.eye(3)[axis],positions.reshape(3,-1).T)[:,j].reshape(source.shape)
            angle=chi-np.roll(chi,-1,axis=j)+harmonic[j]
            result+=velocity*(np.exp(1j*angle)*np.roll(dressed,-1,axis=j)-
                np.exp(-1j*np.roll(angle,1,axis=j))*np.roll(dressed,1,axis=j))/(2j*band.a)
            wrong+=velocity*(np.exp(1j*harmonic[j])*np.roll(source,-1,axis=j)-
                np.exp(-1j*harmonic[j])*np.roll(source,1,axis=j))/(2j*band.a)
        cancellation.append(require(np.linalg.norm(result-local_phase*first[axis]),
                            'original raw Coulomb/harmonic covariant orbital cancellation'))
        missing_phase.append(float(np.linalg.norm(wrong-first[axis])))
    if min(missing_phase) <= .01:
        raise AssertionError('missing harmonic source phase control is not discriminating')
    # Fixed admitted continuum trial used by the unchanged scattering source.
    g0,gs=.4187253908699568,.0632221974274214
    auxiliary=129
    raw=np.zeros((auxiliary,)*3,complex)
    center=auxiliary//2
    raw[center-20:center+21,center-20:center+21,center-20:center+21]=eta.reshape((N,)*3)
    transformed=np.fft.fftn(raw)
    momenta=2*np.pi*np.fft.fftfreq(auxiliary,d=band.a)
    mass=np.full(raw.shape,band.m+3*band.r)
    energy_squared=np.zeros(raw.shape)
    for j in range(3):
        shape=[1,1,1]; shape[j]=auxiliary
        p=momenta.reshape(shape)*band.a
        mass-=band.r*np.cos(p)
        energy_squared+=gs*band.v**2*np.sin(p)**2
    energy=np.sqrt(g0*mass**2+energy_squared)
    coefficients=np.empty((5,)+raw.shape,complex)
    coefficients[0]=raw/2
    coefficients[1]=np.fft.ifftn(transformed*math.sqrt(g0)*mass/(2*energy))
    for j in range(3):
        shape=[1,1,1]; shape[j]=auxiliary
        coefficients[j+2]=np.fft.ifftn(transformed*math.sqrt(gs)*band.v*
            np.sin(momenta.reshape(shape)*band.a)/(2*energy))
    matrices=np.asarray([np.eye(4),*SELECTED])
    q=np.asarray([np.eye(4),*(SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2)))])/2
    beta=np.asarray([1.,.3j,-.2+.4j,.6])
    Q=np.einsum('a,aij->ij',beta/np.linalg.norm(beta),q)
    density=np.zeros(raw.shape)
    # Factor the actual 16-component all-reference state before squaring.
    # Summing signed Gram cross-terms loses the tiny outside probability.
    columns=np.asarray([(matrix@Q).ravel() for matrix in matrices]).T
    _,singular,right=np.linalg.svd(columns,full_matrices=False)
    for row,scale in zip(right,singular):
        amplitude=np.einsum('m,mxyz->xyz',row,coefficients)
        density+=scale**2*abs(amplitude)**2
    actual_norm=float(density.sum())
    require(abs(actual_norm-.5),'actual all-ref fixed-quartet particle band Gram')
    coords=np.indices(raw.shape).astype(float)-center
    radius_squared=np.sum(coords**2,axis=0)
    outside=float(density[radius_squared>64**2].sum()/actual_norm)
    # Exact analytic strip .5 and projector bound8; all nonzero period images.
    source_positions=sites-N//2
    alias_prefactor=2*math.exp(-.5*auxiliary)/(1-math.exp(-.5*auxiliary))
    alias_point=0.
    for count in (1,2,3):
        for axes in itertools.combinations(range(3),count):
            weight=np.sum(abs(source_positions[:,axes]),axis=1)
            weighted_l1=float(np.sum(abs(eta)*np.exp(.5*weight)))
            alias_point+=alias_prefactor**count*math.exp(.5*64*math.sqrt(count))*weighted_l1
    alias_local=8*math.sqrt(float(np.sum(radius_squared<=64**2)))*alias_point/math.sqrt(actual_norm)
    return dict(profile_native_N=41, embedded_support_Chebyshev_radius=20,
        finite_profile_sample_count=N**3, actual_nonzero_profile_samples=int(np.count_nonzero(eta)),
        exact_legal_N=5*2**96, legal_origin='(D,D,D), D=Nstar//2',
        legal_reference='(0,D,D)', source_word_max_expression='D+60',
        field_rotation_plateau_radius=48, field_rotation_outer_radius=64,
        source_and_six_derivative_halo_inside_plateau=True,
        six_derivative_halo_Euclidean_radius=halo_radius,
        actual_Coulomb_harmonic_orbital_cancellation_errors=cancellation,
        missing_harmonic_phase_control_defects=missing_phase,
        common_far_reference_harmonic_phase_evaluated_separately=False,
        embedded_raw_eta_orbital_norms=[float(np.linalg.norm(v)) for v in first],
        embedded_raw_eta_second_angular_norm=float(np.linalg.norm(second)),
        embedded_raw_eta_SU2_algebra_defects=algebra,
        six_orbital_jet_upper=2**50,
        actual_fixed_internal_quartet=beta, complete_bar4_reference=True,
        fixed_quartet_particle_band_initial_G0=actual_norm,
        Gamma_particle_energy_first_graph_comparison_upper=particle_energy_upper,
        Gamma_particle_plus_incoming_photon_graph_comparison_upper=particle_energy_upper+.15,
        original_fullPhi_source_Q_graph_not_inferred_from_Gamma_energy=True,
        bare_charge_probability_inside_64=1-outside,
        bare_charge_probability_outside_64=outside,
        moderate_Bloch_period=auxiliary, finite_Bloch_alias_local_state_upper=alias_local,
        raw_eta_removed_by_supplier_guard_norm=float(np.linalg.norm((1-guard)*eta)),
        Gamma_trial_Gram=dict(g0=g0,gs=gs),
        moderate_Gamma_comparison_not_interacting_G0=True,
        canonical_TWO_receiver_POVM_coefficients_and_rank_changed=False,
        transferred_native41_normal_or_BO_constants=False)


def mechanical_core_noether_consumer(law: Any, frame: np.ndarray,
                                     screen_eta: float) -> dict[str, Any]:
    """Full15 Gauss core compression, real-normal subtraction and boundary RIGHT.

    The three generators are SOLVED from Cartesian vector tangents. They are
    not inserted as a bare spin matrix. Large screening graphs use this native
    frame coefficient; they are not claimed to have this minimizing Gram.
    """
    metric_mass=np.r_[np.ones(60),np.full(45,1/.6)]
    D=np.asarray([coordinates(1j*(t@frame-frame@t),law.basis).reshape(60).real
                  for t in law.generators]).T
    G=np.vstack((D,np.zeros((45,15))))
    inertia=G.T@(metric_mass[:,None]*G)
    _,_,vh=np.linalg.svd(G.T*np.sqrt(1/metric_mass)[None,:],full_matrices=True)
    Z=np.sqrt(1/metric_mass)[:,None]*vh[15:].T
    A0=np.linalg.solve(inertia,G.T@(metric_mass[:,None]*Z))
    origin_error=require(np.linalg.norm(A0),'actual full15 mass-horizontal RNC connection zero')
    first_jets=[]
    for z in Z.T:
        df=np.einsum('ma,aij->mij',z[:60].reshape(4,15),law.basis)
        dD=np.asarray([coordinates(1j*(t@df-df@t),law.basis).reshape(60).real
                       for t in law.generators]).T
        dlinks=[]
        for j in range(3):
            K=np.einsum('a,aij->ij',z[60+15*j:75+15*j],law.generators)
            variation=-1j*(K@law.generators-law.generators@K)
            dlinks.append(2*np.einsum('aij,bji->ab',law.generators,variation).real)
        dG=np.vstack((dD,*dlinks))
        first_jets.append(np.linalg.solve(inertia,dG.T@(metric_mass[:,None]*Z)))
    connection_first_jet=float(np.linalg.norm(np.asarray(first_jets)))
    generators=[]
    vector_errors=[]
    for j,k in ((2,3),(3,1),(1,2)):
        tangent=np.zeros_like(frame); tangent[j]=-frame[k]; tangent[k]=frame[j]
        y=np.r_[coordinates(tangent,law.basis).reshape(60).real,np.zeros(45)]
        solved=np.linalg.solve(inertia,G.T@(metric_mass*y))
        s=np.einsum('a,aij->ij',solved,law.generators)
        vector_errors.append(require(np.linalg.norm(1j*(s@frame-frame@s)-tangent),
            'Gauss-solved original physical Cartesian vector generator'))
        generators.append(s)
    spins=np.asarray(generators)
    fields=np.tile(frame,(law.volume,1,1,1))
    links=np.tile(np.eye(4,dtype=complex),(law.volume,3,1,1)); a=np.zeros((law.volume,3))
    cloud=law.index[(1,1,1)]; reference=0
    eta=np.zeros(law.volume); eta[cloud]=1
    marks=screened_marks(law,fields,links,a,eta,reference,screen_eta)
    h=law.onebody(fields,links,a)
    energies,bands=np.linalg.eigh(h)
    f=smooth_band_value(energies,1e-5)
    F=(bands*f)@bands.conj().T
    full_Phi_Ward=[]
    for s,(j,k) in zip(spins,((2,3),(3,1),(1,2))):
        tangent=np.zeros_like(fields)
        tangent[:,j]=-fields[:,k]; tangent[:,k]=fields[:,j]
        dh=endpoint_flow(law,fields,links,a,(tangent,np.zeros_like(links),np.zeros_like(a)))
        S=np.kron(np.eye(law.volume),s)
        herror=require(np.linalg.norm(dh-1j*(S@h-h@S)),
                       'literal allCartesian Phi current Gauss-derived Spin Ward')
        dF=bands@(smooth_band_divided_difference(energies,1e-5)*
                  (bands.conj().T@dh@bands))@bands.conj().T
        filter_error=require(np.linalg.norm(dF-1j*(S@F-F@S)),
                             'actual full-band Phi filter Frechet Ward')
        source_tangent=tangent.copy(); source_tangent[reference]=0
        step=2e-6
        draw=(screened_marks(law,fields+step*source_tangent,links,a,eta,reference,screen_eta)-
              screened_marks(law,fields-step*source_tangent,links,a,eta,reference,screen_eta))/(2*step)
        raw_error=require(np.linalg.norm(draw-1j*np.einsum('ij,ajr->air',S,marks)),
                          'literal Phi source first jet with fixed outside RIGHT')
        full_Phi_Ward.append(dict(actual_original_Phi_current_error=herror,
            actual_full_band_filter_jet_error=filter_error,
            actual_fixed_RIGHT_raw_source_Phi_jet_error=raw_error,
            global_filter_Ward_not_local_reference_tail_certificate=True))
    point=marks[:,4*cloud:4*cloud+4]
    hs=np.einsum('aij,bij->ab',point.conj(),point)/2
    eig,modes=np.linalg.eigh(hs); initial_inverse=(modes/np.sqrt(eig))@modes.conj().T
    def compress(value: np.ndarray) -> np.ndarray:
        return initial_inverse@np.einsum('aij,bij->ab',point.conj(),value)/2@initial_inverse
    left=np.asarray([compress(s@point) for s in spins])
    right=np.asarray([compress(point@s) for s in spins])
    require(np.linalg.norm(sum(s@s for s in left)-.75*np.eye(4)),
            'derived source-label half Casimir from actual fullGauss vector tangent')
    # Complete CAR, same neutral and its genuine particle-hole partner.
    small=np.concatenate((point,np.zeros((4,8,4))),axis=1)
    rows=tuple(itertools.combinations(range(12),6))
    charged=tuple(itertools.combinations(range(12),7))
    creators=maps_from_marks(small,rows,charged)
    rng=np.random.default_rng(22210659)
    neutral=rng.normal(size=len(rows))+1j*rng.normal(size=len(rows))
    neutral/=np.linalg.norm(neutral)
    partner=particle_hole(rows,12)@neutral.conj()
    columns=[np.column_stack([b@v for b in creators]) for v in (neutral,partner)]
    car_metric=sum(c.conj().T@c for c in columns)/2
    car_error=require(np.linalg.norm(car_metric-hs),'core Noether fullCAR CT initial metric')
    screening=screened_boundary(frame,[17,33,65,129,257])
    screened_rows=[]
    for entry in screening['ordinary_rigid_plateau_rows']:
        source=entry['source_coefficient']; ref=entry['fixed_RIGHT_coefficient']
        matrices=source*left-ref*right
        eps=float(max(np.linalg.norm(matrices[j]-left[j],ord=2) for j in range(3)))
        casimir=float(np.linalg.norm(sum(s@s for s in matrices)-.75*np.eye(4),ord=2))
        algebra=float(max(np.linalg.norm(matrices[j]@matrices[k]-matrices[k]@matrices[j]
                         -1j*matrices[l],ord=2) for j,k,l in ((0,1,2),(1,2,0),(2,0,1))))
        loop=float(np.linalg.norm(expm(-2j*math.pi*matrices[2])+np.eye(4),ord=2))
        screened_rows.append(dict(**entry, actual_projected_Noether_matrices=matrices,
            half_generator_error=eps, SU2_commutator_error=algebra,
            J_squared_error=casimir, projected_two_pi_minus_error=loop,
            operator_generator_upper=(abs(1-source)+abs(ref))/2,
            RIGHT_boundary_electric_term_retained=True))
    # Independent complete exterior-CAR consumption of the final boundary term.
    entry=screened_rows[-1]
    actual=[]
    for s in spins:
        value=entry['source_coefficient']*(s@point)-entry['fixed_RIGHT_coefficient']*(point@s)
        derivative=maps_from_marks(np.concatenate((value,np.zeros((4,8,4))),axis=1),
                                  rows,charged)
        crossed=sum(c.conj().T@np.column_stack([b@v for b in derivative])
                    for c,v in zip(columns,(neutral,partner)))/2
        actual.append(initial_inverse@crossed@initial_inverse)
    matrix_car=require(np.linalg.norm(np.asarray(actual)-entry['actual_projected_Noether_matrices']),
                       'fullCAR screened Noether and complete bar4 boundary matrix')
    positions=(law.sites-law.sites[cloud]+law.size//2)%law.size-law.size//2
    velocity=np.column_stack((-positions[:,1],positions[:,0],np.zeros(law.volume)))
    orbital=scalar_orbital(law,links,a,velocity)
    orbital_point=require(np.linalg.norm(np.einsum('ij,ajr->air',orbital,marks)),
                          'actual scalar-orbital action on centred point source')
    gram=np.einsum('mij,nji->mn',frame,frame).real/4
    haar=orientation_core_halfspin_bound(6,screen_eta,float(np.linalg.eigvalsh(gram)[0]),
                                       float(np.trace(gram)),1e-3)
    return dict(full15_inertia_min=float(np.linalg.eigvalsh(inertia)[0]),
        actual_RNC_normal_connection_origin_error=origin_error,
        actual_RNC_normal_connection_first_jet_Frobenius=connection_first_jet,
        physical_vector_tangent_errors=vector_errors, actual_initial_fullCAR_metric=hs,
        initial_metric_fullCAR_error=car_error, final_screened_Noether_fullCAR_error=matrix_car,
        actual_scalar_orbital_point_action_error=orbital_point,
        screened_core_rows=screened_rows,
        actual_continuum_Haar_orientation_core_bound=haar,
        actual_fullPhi_current_filter_and_raw_source_Ward=full_Phi_Ward,
        physical_source_Lie_bound=physical_orientation_lie_majorant(haar,2.,math.sqrt(float(eig[0]))),
        projected_scope='mass-horizontal fullGauss W core; common real normal vacuum and heavy scalar Berry background subtracted with SAME neutral',
        genuine_photon_orbital_reference_graph_and_true_spectral_transfer_still_required=True,
        source_end_spin_inserted=False, micro_cubic_SO3_claimed=False)


def local_photon_noether_consumer(law: Any, frame: np.ndarray,
                                  screen_eta: float) -> dict[str, Any]:
    """Actual local real-Cartan photon/current/source and U1 Gauss ordering.

    Uniform harmonic A is annihilated, not silently bounded by its energy.
    Global vector rotations are different and retain their harmonic A-P graph.
    """
    rng=np.random.default_rng(22210661)
    fields=np.tile(frame,(law.volume,1,1,1))+.002*law.fields(
        rng.normal(size=(law.volume,4,15)))
    links=np.asarray([[expm(.04j*np.einsum('a,aij->ij',rng.normal(size=15),law.generators))
                        for j in range(3)] for x in range(law.volume)])
    a=rng.normal(scale=.7,size=(law.volume,3))
    cloud=law.index[(1,1,1)]; reference=0
    eta=np.zeros(law.volume); eta[cloud]=1
    positions=(law.sites-law.sites[cloud]+law.size//2)%law.size-law.size//2
    weight=(np.sum(positions**2,axis=1)<=1).astype(float)
    velocity=weight[:,None]*np.column_stack(
        (-positions[:,1],positions[:,0],np.zeros(law.volume)))
    if weight[reference] or np.linalg.norm(velocity[cloud]) or np.linalg.norm(velocity[reference]):
        raise AssertionError('local physical angular source/reference geometry invalid')
    flow=mechanical_flow(law,fields,links,a,velocity,weight)
    harmonic=np.array([.31,-.27,.19])
    shifted_flow=mechanical_flow(law,fields,links,a+harmonic,velocity,weight)
    null_error=require(np.linalg.norm(shifted_flow[2]-flow[2]),
                       'actual local real-Cartan Noether annihilates harmonic A')
    sine_defect=float(np.linalg.norm(flow[2]-primitive_flow(
        law,fields,links,a,velocity,weight)[2]))
    if sine_defect<=1e-5:
        raise AssertionError('compact sine photon flow failed to discriminate actual real law')
    step=2e-6
    K=-1j*flow[1]@links.conj().swapaxes(-1,-2)
    plus_links=np.asarray([[expm(1j*step*K[x,j])@links[x,j]
                           for j in range(3)] for x in range(law.volume)])
    minus_links=np.asarray([[expm(-1j*step*K[x,j])@links[x,j]
                            for j in range(3)] for x in range(law.volume)])
    dh=endpoint_flow(law,fields,links,a,flow)
    numeric=(law.onebody(fields+step*flow[0],plus_links,a+step*flow[2])
             -law.onebody(fields-step*flow[0],minus_links,a-step*flow[2]))/(2*step)
    current_error=require(np.linalg.norm(numeric-dh),
                          'actual all-term local physical endpoint current')
    def source(value: np.ndarray) -> np.ndarray:
        mean=value.mean(axis=0)
        return screened_marks(law,fields,links,value-mean,eta,reference,screen_eta,
                              harmonic=mean)
    marks=source(a)
    photon_derivative=(source(a+step*flow[2])-source(a-step*flow[2]))/(2*step)
    g,_,_=green(law)
    change=g@divergence(law,flow[2])
    displacement=law.sites[cloud]-law.sites[reference]
    scalar=float(change[cloud]-change[reference]-displacement@flow[2].mean(axis=0))
    source_error=require(np.linalg.norm(photon_derivative-1j*scalar*marks),
                         'actual Coulomb harmonic source photon Noether contribution')
    # Canonical lift differs by grad(lambda(A)); the complete periodic trace
    # of its field-dependent Gauss coefficient is zero, including ordering.
    trace=.5*sum(float((velocity[:,j]-velocity[law.target[:,j],j]).sum())
                 for j in range(3))
    trace_error=require(abs(trace),'periodic field-dependent U1 Gauss ordering trace')
    phnorm=float(np.linalg.norm(flow[2]))
    if phnorm<=1e-5 or abs(scalar)<=1e-5:
        raise AssertionError('nonzero local photon/source angular positive control missed')
    return dict(actual_real_photon_flow_norm=phnorm,
        full_current_derivative_error=current_error,
        harmonic_A_null_error=null_error, compact_sine_wrong_law_defect=sine_defect,
        actual_Coulomb_harmonic_scalar_Noether_source_term=scalar,
        actual_photon_source_derivative_error=source_error,
        field_dependent_U1_Gauss_ordering_trace_error=trace_error,
        reference_outside_and_fixed=True,
        local_harmonic_AP_needed=False,
        remaining_local_photon_graph='coexact E-A plus harmonic p_h times curl A and longitudinal Coulomb E times curl A',
        global_harmonic_AP_and_scattering_source_Dq_debts_retained=True,
        low_energy_Coulomb_boundary_term_not_discarded=True)


def same_neutral_photon_commutator_consumer(law: Any, frame: np.ndarray,
                                         screen_eta: float) -> dict[str, Any]:
    """Actual literal flow, fullbar4 CAR and A-dependent SAME neutral current.

    This is an operator-identity witness on the physical Gauss quotient,
    not a replacement for the interacting neutral ground/energy graph.
    """
    rng=np.random.default_rng(22210663)
    fields=np.tile(frame,(law.volume,1,1,1))+.002*law.fields(
        rng.normal(size=(law.volume,4,15)))
    links=np.asarray([[expm(.04j*np.einsum('a,aij->ij',rng.normal(size=15),law.generators))
                        for _ in range(3)] for _ in range(law.volume)])
    cloud=law.index[(1,1,1)]; reference=0
    sites=[cloud,reference,law.index[(1,0,0)]]
    indices=np.concatenate([np.arange(4*x,4*x+4) for x in sites])
    eta=np.zeros(law.volume); eta[cloud]=1
    base=screened_marks(law,fields,links,np.zeros((law.volume,3)),eta,
                        reference,screen_eta,harmonic=np.zeros(3))
    small=base.reshape(4,law.volume,4,4)[:,sites].reshape(4,12,4)
    rows=tuple(itertools.combinations(range(12),6))
    charged=tuple(itertools.combinations(range(12),7))
    creators=maps_from_marks(small,rows,charged)
    seed=rng.normal(size=len(rows))+1j*rng.normal(size=len(rows))
    seed/=np.linalg.norm(seed)
    occupations=np.asarray([[sum(mode//4==s for mode in row)-2
                             for s in range(3)] for row in rows])
    div=np.zeros((law.volume,3*law.volume))
    for x in range(law.volume):
        for j in range(3):
            div[x,3*x+j]=1
            div[int(law.target[x,j]),3*x+j]-=1
    g,_,_=green(law)
    harmonic=np.tile(np.eye(3),(law.volume,1))/math.sqrt(law.volume)
    transverse=np.eye(3*law.volume)-div.T@g@div-harmonic@harmonic.T
    harmonic_projector=harmonic@harmonic.T
    profile=(law.sites-law.sites[cloud]+law.size//2)%law.size-law.size//2
    cutoff=(np.sum(profile**2,axis=1)<=1).astype(float)
    velocity=cutoff[:,None]*np.column_stack((-profile[:,1],profile[:,0],np.zeros(law.volume)))
    photon=transverse@rng.normal(size=3*law.volume)
    photon/=np.linalg.norm(photon)
    a0=rng.normal(scale=.7,size=(law.volume,3))
    results=[]
    for UE in (1.,.25,.0625):
        a=math.sqrt(UE)*a0
        tangent=primitive_flow(law,fields,links,a,velocity,cutoff)[2]
        ordering=0.
        for coordinate_index in range(3*law.volume):
            direction=np.zeros_like(a)
            direction.ravel()[coordinate_index]=1
            vf=primitive_flow(law,fields,links,a+2e-6*direction,velocity,cutoff)[2]
            vb=primitive_flow(law,fields,links,a-2e-6*direction,velocity,cutoff)[2]
            ordering+=(vf.ravel()[coordinate_index]-vb.ravel()[coordinate_index])/4e-6
        tflat=tangent.ravel()
        coexact=(transverse@tflat).reshape(a.shape)
        phase_coexact=g@divergence(law,coexact)
        phase_null=require(np.linalg.norm(phase_coexact)+np.linalg.norm(coexact.mean(axis=0)),
                           'RAW source exact physical coexact commutation')
        def phase(value: np.ndarray) -> float:
            chi=g@divergence(law,value)
            return float(chi[cloud]-chi[reference]-
                (law.sites[cloud]-law.sites[reference])@value.mean(axis=0))
        scalar=phase(tangent)
        def neutral(value: np.ndarray) -> np.ndarray:
            flat=value.ravel()
            chi=g@(div@flat)
            reduced=(transverse@flat).reshape(value.shape)+value.mean(axis=0)
            onebody=law.onebody(fields,links,reduced)[np.ix_(indices,indices)]
            current_history=expm_multiply(-.17j*lift(onebody,rows),seed)
            gauge=np.exp(1j*(occupations@chi[sites]))
            coordinate=transverse@flat
            baseline=transverse@a.ravel()
            physical=coordinate+harmonic_projector@flat
            physical_baseline=baseline+harmonic_projector@a.ravel()
            gaussian=math.exp(-(np.dot(physical,physical)-
                np.dot(physical_baseline,physical_baseline))/(2*UE))
            onephoton=np.dot(photon,coordinate)/np.dot(photon,baseline)
            return gaussian*onephoton*gauge*current_history
        step=2e-6
        minus,plus=neutral(a-step*tangent),neutral(a+step*tangent)
        psi=neutral(a)
        Jn=-1j*((plus-minus)/(2*step)+ordering*psi/2)
        bminus=np.exp(1j*phase(a-step*tangent))
        bplus=np.exp(1j*phase(a+step*tangent))
        bzero=np.exp(1j*phase(a))
        errors=[]; charged_norms=[]; relative_norms=[]
        for B in creators:
            Jc=-1j*((bplus*(B@plus)-bminus*(B@minus))/(2*step)+ordering*bzero*(B@psi)/2)
            relative=Jc-bzero*(B@Jn)
            target=scalar*bzero*(B@psi)
            errors.append(require(np.linalg.norm(relative-target),
                'actual current SAME neutral photon commutator',2e-8))
            charged_norms.append(float(np.linalg.norm(Jc)))
            relative_norms.append(float(np.linalg.norm(relative)))
        if np.linalg.norm(Jn)<=1e-4 or abs(scalar)<=1e-6:
            raise AssertionError('neutral angular covariance/nonzero boundary control missed')
        results.append(dict(UE=UE, literal_photon_flow_norm=float(np.linalg.norm(tangent)),
            nonzero_scalar_Coulomb_harmonic_boundary=scalar,
            coexact_phase_commutation_error=phase_null,
            literal_field_ordering_divergence=ordering,
            actual_fullCAR_relative_identity_error=max(errors),
            SAME_neutral_angular_noise_norm=float(np.linalg.norm(Jn)),
            charged_angular_norm_without_neutral_subtraction=max(charged_norms),
            relative_photon_norm=max(relative_norms),
            relative_norm_divided_by_sqrt_UE=max(relative_norms)/math.sqrt(UE)))
    return dict(rows=results, complete_CAR_modes=12, complete_RIGHT_dimension=4,
        neutral_current_history='actual native Peierls h[A_T+a_h] exterior propagator and full Gauss phase',
        source_coexact_commutation=True, neutral_noise_variance_deleted=False,
        full_interacting_ground_graph_claimed=False,
        explicit_F_h_source_would_require_nonzero_filter_jet=True)


def normal_noether_clock_consumer(law: Any, frame: np.ndarray) -> dict[str, Any]:
    """Actual ALL90V quadratic normal Noether one-phonon norm, same r8 clock.

    Forces are squared moduli: no phases/wavefunction are fabricated from them.
    This diagnostic does not simulate the full nonlinear quantum vacuum.
    """
    weight=np.zeros(law.volume); weight[law.index[(1,1,1)]]=1
    _,_,chat,gauss_error=geometry(law,frame,weight)
    pulse=SmoothPulse(order=8)
    rows=[]
    for exponent in (64,80):
        scale=2.**exponent; chi=scale**(-.5); duration=scale**(-1/32)
        frequencies,forces,controls=native_modes(law,frame,(0,0,0),chat,scale)
        contributions=forces*frequencies/(2*chi)
        unfiltered=math.sqrt(float(contributions.sum()))
        multiplier=pulse.integral(frequencies*duration)
        filtered=math.sqrt(float(np.dot(contributions,abs(multiplier)**2)))
        gap=float(frequencies.min())
        upper=physical_spectral_filter_bound(gap,0.,duration,unfiltered)
        require(max(0.,filtered-upper['high_actual_spectral_block_bound']),
                'native full normal Noether spectral r8 upper')
        rows.append(dict(log2_Lambda=exponent, original_chi=chi, same_r8_duration=duration,
            actual_complete_normal_gap=gap,
            unfiltered_physical_horizontal_Noether_onephonon_norm=unfiltered,
            filtered_physical_horizontal_Noether_onephonon_norm=filtered,
            spectral_r8_operator_upper=upper['high_actual_spectral_block_bound'],
            ALL90V_normal_controls=controls))
    return dict(original_full15_mass_Gauss_error=gauss_error, rows=rows,
        exact_quadratic_ground_to_onephonon_norm=True,
        relative_phases_reconstructed_from_squared_forces=False,
        full_nonlinear_coupled_quantum_Noether_certificate=False,
        missing_from_diagnostic='quadratic/angular pair creation, nonlinear normal jets, actual photon/orbital/harmonic and full physical spectral transfer')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=3)
    parser.add_argument('--screen-eta',type=float,default=4.)
    parser.add_argument('--loop-samples',type=int,default=400)
    parser.add_argument('--normal-noether',action='store_true')
    parser.add_argument('--embedded-radial-receiver',action='store_true')
    args=parser.parse_args()
    if args.loop_samples%2 or args.loop_samples<20:
        raise ValueError('even loop-samples>=20 required to exercise exact seam')
    band=RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    _,frame=original_frame(band,args.size,np.zeros(3),True)
    law=BulkQuantumLaw(args.size,band=band)
    same_core=orientation_core_halfspin_bound(5*2**95+60,18.,.025,2.,.001)
    report=dict(source='fullSU4 smooth Phi-polar centre ratio, not thin Wilson or source-end spin',
        parameters=vars(args),native=native_consumer(law,frame,args.screen_eta),
        local_loop=local_loop_consumer(frame,law.generators,args.screen_eta,args.loop_samples),
        dynamical_cloud=dynamical_cloud_consumer(law,frame,args.screen_eta),
        orientation_cloud=orientation_cloud_consumer(frame,law.generators,args.screen_eta),
        physical_cocycle_and_flux=physical_cocycle_and_flux_consumer(law,frame,args.screen_eta),
        dynamical_local_frame=dynamical_local_frame_protocol(law,frame,args.screen_eta),
        mechanical_core_Noether=mechanical_core_noether_consumer(law,frame,args.screen_eta),
        native_normal_Noether_clock=normal_noether_clock_consumer(law,frame) if args.normal_noether else None,
        linear_real_Cartan_photon_diagnostic=local_photon_noether_consumer(law,frame,args.screen_eta),
        SAME_neutral_actual_photon_commutator=same_neutral_photon_commutator_consumer(
            law,frame,args.screen_eta),
        embedded_RAW_radial_receiver=embedded_raw_radial_receiver(band) if args.embedded_radial_receiver else None,
        original_raw_Noether_geometry=physical_raw_noether_geometry_majorant(64.,1.),
        SAME_N_full_Haar_core=same_core,
        SAME_N_source_value_comparison=orientation_core_point_source_bound(
            same_core,.4892165624,1025/1024),
        SAME_N_physical_Phi_Lie=physical_orientation_lie_majorant(
            same_core,2.,.4892165624,1025/1024),
        orbital_positive_band=continuum_packet_consumer(frame,band),
        actual_law_unchanged=True,finite_cubic_grid_microSO3_claim=False,
        physical_Noether_delivery_scope='literal source, normal, EM, orbital and localization inputs; full flight composed by consumers',
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'noncompact_screened_gauss_dressing.py','noncompact_screened_gauss_dressing.md',
            'noncompact_centre_source_energy.py','original_centre_charge.py',
            'noncompact_physical_rotation_join.py','original_charged_principal.py',
            'noncompact_mechanical_pulse.py','original_total_angular_transfer.py',
            'noncompact_quantitative_normal_budget.py',
            'noncompact_marked_metric.py','noncompact_finite_packet_spin.py')})
    print(json.dumps(portable(report),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
