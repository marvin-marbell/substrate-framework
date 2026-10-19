#!/usr/bin/env python3
"""Explicit cutoff family: selected uniform fields, same-current Gaussian photon/Pauli response."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from angular_photon_spectrum import ContinuousSea, SeaTensor
from bulk_frame_compton import (
    band_parameters, current_excited_gauss, primitive_frame_controls,
    reconstruct_frame, source_hashes as inherited_hashes,
)
from periodic_bulk_frame import midpoint_grid
from photon_matched_compton import forward_at_pole
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, matrix_unit_basis, rank_report
from soft_budget import select_gram
from unrestricted_field_minimum import BARE, STIFFNESS, check, coordinates

ROOT=Path(__file__).resolve().parent


def source_hashes():
    return {**inherited_hashes(),**{
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('soft_budget.py','soft_budget.md','bulk_global_gram.py','bulk_global_gram.md',
                     'angular_photon_spectrum.py','bulk_cutoff_matching.py','bulk_cutoff_matching.md')}}


def dimensionless_bare(a):
    return RotationBand(a=1.,m=a*BARE['m'],r=BARE['r'],v=BARE['v'],e=1.,improved=False)


def selected_gram(a,grid):
    x=coefficients(dimensionless_bare(a),midpoint_grid(grid,1.))
    solution=select_gram(x,STIFFNESS,0.)
    G=solution['G']
    radii=np.sqrt(np.einsum('pm,mn,pn->p',x,G,x))
    gradient=STIFFNESS*G-np.einsum('pi,pj,p->ij',x,x,1/radii)/len(x)
    check(float(np.linalg.norm(gradient)),'actual full10 Gram stationarity',1e-10)
    richness=rank_report(x)
    if richness['quadratic_rank']!=10 or np.any(np.diag(G)<=0):
        raise AssertionError('Four nonzero actual selected Clifford scales and rank10 required')
    return solution,float(np.linalg.norm(gradient)),richness


def photon_coefficients(infrared, speed, g2) -> dict[str,float]:
    ce=1/(g2*speed)-infrared['electric']
    cb=speed/g2-infrared['magnetic']
    if not np.isfinite(ce+cb) or min(ce,cb)<=0:
        raise ValueError('Fixed target coupling requires nonpositive bare rotor stiffness at this cutoff')
    return dict(dimensionless_lagrangian_electric=float(ce),dimensionless_magnetic=float(cb),
        effective_electric=float(ce+infrared['electric']),effective_magnetic=float(cb+infrared['magnetic']),
        dimensionless_Hamiltonian_electric=float(1/ce))


def photon_profile(sea, q, matched, band, *, length=None) -> dict[str,Any]:
    raw=sea.spectrum(q)
    tensor=SeaTensor(raw.q,raw.contact,raw.channels,
        matched['dimensionless_lagrangian_electric'],matched['dimensionless_magnetic'],raw.normalization)
    modes: dict[str,Any]=tensor.modes()
    khat=tensor.khat
    restoring=float(np.linalg.eigvalsh(tensor.inverse(0.)).min())
    restoring_Ward=float(np.linalg.norm(tensor.static@khat)/(np.linalg.norm(khat)*restoring))
    comparisons=[]
    for branch in modes['branches']:
        comparisons.append(dict(physical_energy=branch['energy']/band.a,
            physical_phase_speed=branch['energy']/np.linalg.norm(q),
            lattice_IR_velocity_ratio=branch['energy']/(band.v*np.linalg.norm(khat)),
            physical_A_inverse_residue=branch['inverse_residue'],
            physical_link_inverse_residue=band.a*branch['inverse_residue'],
            physical_volume_field_leg=branch['source'],pole_residual=branch['pole_residual']))
    checks=dict(positive_residue=all(branch['inverse_residue']>=matched['dimensionless_lagrangian_electric']>0 for branch in modes['branches']),
        below_pair_threshold=all(branch['energy']<tensor.gap for branch in modes['branches']),
        lattice_corrected_IR_speeds=all(abs(branch['lattice_IR_velocity_ratio']-1)<1e-3 for branch in comparisons),
        actual_restoring_scaled_Ward=restoring_Ward<1e-3)
    if not all(checks.values()):
        raise AssertionError(f'Unresolved actual finite photon tensor at physical q={q/band.a}: {checks}, Ward={restoring_Ward}')
    result=dict(physical_q=(q/band.a).tolist(),dimensionless_q=q.tolist(),
        modes=modes,physical_profiles=comparisons,restoring_scaled_Ward_vector=restoring_Ward,
        contact_scaled_Ward=modes['gradient_relative_residual'],checks=checks)
    if length is not None:
        first=comparisons[0]
        pole=dict(L=length,band_parameters=band_parameters(band),momentum=float(q[0]/band.a),
            photon_energy=first['physical_energy'],canonical_field_leg_factor=first['physical_volume_field_leg'],
            hamiltonian_UE1=matched['dimensionless_Hamiltonian_electric']/band.a,
            link_residue_inverse=first['physical_link_inverse_residue'],
            physical_field_residue_inverse=first['physical_A_inverse_residue'],checks=checks)
        response=forward_at_pole(band,pole)
        if not all(response['checks'].values()):
            raise AssertionError(f'Same-current native Pauli response failed: {response["checks"]}')
        # tau=c*t, canonical external legs divide by sqrt(c), H_tau=H/c.
        # Thus amplitude_tau=amplitude_t/c^2 and M_tau=M_t/c.
        result['actual_shared_forward_Pauli']=response
        result['Lorentz_time_normalized_forward_squared']=response['pauli_forward_squared']/band.v**6
        result['declared_soft_continuum_target']=4*band.e**4/(matched['effective_electric']*band.v)**2
        result['finite_Wilson_soft_ratio']=(1+band.a*band.r*band.m/band.v**2)**2
    return result


def run(*,spacings,gram_grid=32,validation_grid=48,sea_order=16,validation_sea_order=24,g2=.1,momentum=.1):
    if (not spacings or any(not np.isfinite(a) or not 0<a<=1 for a in spacings)
            or not np.isfinite(g2+momentum) or min(g2,momentum)<=0 or momentum>=.2):
        raise ValueError('0<a<=1 cutoff trajectory, positive finite g² and physical momentum<.2 required')
    if min(gram_grid,validation_grid)<4 or min(sea_order,validation_sea_order)<8 or validation_sea_order<sea_order:
        raise ValueError('Gram grids>=4 and ordered sea quadratures>=8 required')
    identities=source_hashes()
    rows=[]
    basis=matrix_unit_basis()
    limit_solution,limit_residual,limit_richness=selected_gram(0.,validation_grid)
    limit_mass=BARE['m']*np.sqrt(limit_solution['G'][0,0])
    limit_speed=BARE['v']*np.sqrt(limit_solution['G'][1,1])
    for a in spacings:
        coarse,coarse_residual,_=selected_gram(a,gram_grid)
        solution,residual,richness=selected_gram(a,validation_grid)
        fields=np.sqrt(np.diag(solution['G']))[:,None,None]*SELECTED
        bare=RotationBand(a=a,m=BARE['m'],r=BARE['r'],v=BARE['v'],e=1.,improved=False)
        band,unitary,reconstruction=reconstruct_frame(bare,fields)
        unit=RotationBand(a=1.,m=a*band.m,r=band.r,v=band.v,e=1.,improved=False)
        current=primitive_frame_controls(dimensionless_bare(a),unit,fields,unitary)
        coarse_sea=ContinuousSea(sea_order,band=unit,chi_e=1.,chi_b=1.,quadrature='split_gauss')
        coarse_IR=coarse_sea.infrared()
        del coarse_sea
        sea=ContinuousSea(validation_sea_order,band=unit,chi_e=1.,chi_b=1.,quadrature='split_gauss')
        infrared=sea.infrared()
        matched=photon_coefficients(infrared,band.v,g2)
        profiles=[]
        for radius in (momentum,momentum/2):
            length=max(2,int(round(2*np.pi/(a*radius))))
            q=np.array([2*np.pi/length,0.,0.])
            profiles.append(photon_profile(sea,q,matched,band,length=length))
        profiles.append(photon_profile(sea,a*momentum*np.array([.8,.6,0.]),matched,band))
        dispersion=[]
        for p in (np.array([momentum,0.,0.]),momentum*np.array([.8,.6,0.]),momentum*np.array([.3,.2,.4])):
            exact=band.energy(p)
            finite_Dirac=float(np.sqrt(band.m**2+band.v**2*np.dot(p,p)))
            limiting_Dirac=float(np.sqrt(limit_mass**2+limit_speed**2*np.dot(p,p)))
            dispersion.append(dict(physical_p=p.tolist(),actual_source_energy=exact,
                relativistic_finite_coefficients=finite_Dirac,mesh_limit_Dirac_energy=limiting_Dirac,
                finite_coeff_Dirac_error=abs(exact-finite_Dirac),mesh_limit_error=abs(exact-limiting_Dirac)))
        unused_speed=float(np.sqrt((a*.7+infrared['magnetic'])/(1/(a*.4)+infrared['electric'])))
        bad_g2=1e4
        rejected=False
        try:
            photon_coefficients(infrared,band.v,bad_g2)
        except ValueError:
            rejected=True
        if not rejected:
            raise AssertionError('Actual negative-bare-stiffness target incorrectly accepted')
        rows.append(dict(a=a,physical_stiffness=STIFFNESS/a,bare_band=band_parameters(bare),
            effective_band=band_parameters(band),actual_field_coordinate_values=coordinates(fields,basis).tolist(),
            Gram=solution['G'].tolist(),full_Gram_gradient_norm=residual,
            physical_uniform_energy=solution['value']/a,quadratic_richness=richness,
            Gram_refinement=dict(coarse_grid=gram_grid,fine_grid=validation_grid,
                Gram_difference=float(np.linalg.norm(solution['G']-coarse['G'])),coarse_gradient=coarse_residual,
                status='floating mesh refinement, not interval continuum certification'),
            actual_frame_reconstruction=reconstruction,actual_dimensionless_primitive_current_control=current,
            current_units='energy=a Hphys, momentum=theta=a pphys; native physical matrix/current/contact follow exact scaling',
            Wilson_relative_inertia_debt=a*band.r*band.m/band.v**2,
            lowest_other_corner_mass=band.energy(np.array([np.pi/a,0.,0.])),
            physical_dispersion=dispersion,sea_infrared=infrared,
            infrared_refinement=dict(coarse_order=sea_order,fine_order=validation_sea_order,coarse=coarse_IR,
                electric_difference=abs(infrared['electric']-coarse_IR['electric']),
                magnetic_difference=abs(infrared['magnetic']-coarse_IR['magnetic']),status='numerical quadrature difference, not a certified enclosure'),
            matched_photon=matched,physical_Hamiltonian_UE1=matched['dimensionless_Hamiltonian_electric']/a,
            physical_Hamiltonian_K1=matched['dimensionless_magnetic']/a,
            declared_g2=g2,original_constant_UE1_K1_unretuned_IR_speed=unused_speed,
            negative_bare_target_control=dict(g2=bad_g2,rejected=True),actual_photon_and_forward_consumers=profiles))
        del sea
    finest=min(spacings)
    fine_solution,_,_=selected_gram(finest,validation_grid)
    fine_fields=np.sqrt(np.diag(fine_solution['G']))[:,None,None]*SELECTED
    gauss=current_excited_gauss(dimensionless_bare(finest),fine_fields)
    if source_hashes()!=identities:
        raise RuntimeError('Source/proof family changed during execution')
    return dict(source_sha256=identities,source_bindings_unchanged=True,
        governing_plan='issue-222-continuing-backward-20260930',work_item='campaign',
        family='EXPLICIT different bare trajectory: fixed primitive m.2/r.12/v.3, kappa(a)=2/a, source-selected uniform fields; retuned UE1/K1',
        field_law='all60 uniform Cartesian fields, soft full Gram quartic, global TOTAL half heavy filling, identity colour links',
        photon_law='same selected-band quadratic free-sea + Gaussian rotor, actual raw current/contact, positive Hamiltonian coefficients',
        finite_mesh_limiting_coefficients=dict(grid=validation_grid,Gram=limit_solution['G'].tolist(),
            mass=float(limit_mass),speed=float(limit_speed),full_gradient=limit_residual,richness=limit_richness,
            status='finite-mesh estimate of continuum trajectory endpoint, no interval integral certificate'),
        rows=rows,finest_current_excited_primitive_neutral_Gauss=gauss,all_checks_pass=True,
        campaign_complete=False,limits=['Original a1 UE1=0.4,K1=0.7 point is NOT retroactively speed-matched; photon inputs here are explicitly retuned.',
            'Source-selected classical uniform mean, not nonuniform/full primitive quantum field or compact phase proof.',
            'Gaussian matching is declared parameter tuning, not a mechanism predicting coupling or velocity.',
            'Fixed nonzero g² UV continuum is not supplied: positive logarithmic vacuum response eventually exhausts positive bare electric stiffness.',
            'Primitive Pauli forward consumer retains neutral hidden marker, not isolated charged B4/bar4 propagation or mechanical spin.',
            'No full nonlinear photon/matter radiative closure, thermodynamic phase or full joined campaign completion.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a',type=float,nargs='+',default=[1.,.2,.05,.01,.005,.001])
    parser.add_argument('--gram-grid',type=int,default=32)
    parser.add_argument('--validation-grid',type=int,default=48)
    parser.add_argument('--sea-order',type=int,default=16)
    parser.add_argument('--validation-sea-order',type=int,default=24)
    parser.add_argument('--g2',type=float,default=.1)
    parser.add_argument('--momentum',type=float,default=.1)
    parser.add_argument('--output',type=Path,default=ROOT/'bulk_cutoff_matching_receipt.json')
    args=parser.parse_args()
    result=run(spacings=args.a,gram_grid=args.gram_grid,validation_grid=args.validation_grid,
        sea_order=args.sea_order,validation_sea_order=args.validation_sea_order,g2=args.g2,momentum=args.momentum)
    text=json.dumps(result,indent=2);args.output.write_text(text+'\n');print(text)


if __name__=='__main__':
    main()
