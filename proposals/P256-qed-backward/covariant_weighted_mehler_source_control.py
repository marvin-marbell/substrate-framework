#!/usr/bin/env python3
"""NEW conditional leading-Gaussian/physical-Root Bloch-fiber source control.

This is not the Source2a microscopic backend. The field coefficients both come
from its primitive kappa^4 Wilson word: beta=16*Naux*kappa_aux^4. A single
coexact mode is coupled to an actual physical four-component Root fiber;
canonical exterior matrices retain every occupation and all four Ref channels.
Finite Hermite compressions are comparison laws, not microscopic admissions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_hermite, factorial, roots_hermite

import covariant_canonical_gauss_boundary_kernel as supplier

HERE=Path(__file__).resolve().parent
SOURCE_SHA='2a0ecfb2341fceb7866d5062cf33d1da71b2fafa4b148829d21d14f0650fe307'
ROOT_SHA='6911f0283dc0f84f4cc05202deb67dcd3266d5c64ca1a05e149f1567451e0f9c'
SELECTED='covariant_prepared_gauss_state_conditioned_source_physical_Dminus_verified_20261008.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hermitian(a):
    return (a+a.conj().T)/2


def positive_power(a,power):
    values,vectors=eigh(hermitian(a))
    if values[0]<=0:
        raise ValueError('The native comparison matrix must be strictly positive')
    return (vectors*values**power)@vectors.conj().T


def creation(coefficients):
    neutral=supplier.occupations(4,2)
    charged=supplier.occupations(4,3)
    positions={state:i for i,state in enumerate(charged)}
    result=np.zeros((4*len(charged),len(neutral)),complex)
    for ref in range(4):
        for column,state in enumerate(neutral):
            for mode in range(4):
                if mode in state:
                    continue
                target=tuple(sorted((*state,mode)))
                sign=(-1)**sum(j<mode for j in state)
                result[ref*len(charged)+positions[target],column]+=sign*coefficients[mode,ref]
    return result


def fiber(kappa,momentum,field):
    gamma,_=supplier.phi_frame(supplier.Parameters())
    phase=np.array(momentum)+np.array([0.,field,0.])
    links=[np.exp(1j*p)*np.eye(4) for p in phase]
    half,whole_scalar,transfer=supplier.physical_spatial_onebody(
        gamma,kappa,links,[np.zeros((4,4),complex) for _ in range(4)],np.eye(4))
    return half,math.sqrt(whole_scalar),transfer


def logarithm(a):
    values,vectors=eigh(hermitian(a))
    return (vectors*np.log(values))@vectors.conj().T


def projected(values,phi,weights):
    q,left,right=values.shape
    modes=phi.shape[1]
    return np.einsum('ia,i,ib,iuv->aubv',phi,weights,phi,values,optimize=True).reshape(modes*left,modes*right)


def native_control(modes,nodes,kappa,beta,omega,nt) -> dict[str,Any]:
    energy=2*math.asinh(omega/2)
    s=beta*math.sinh(energy)
    sigma=math.sqrt(1/(2*s))
    lam=math.exp(-energy/2)
    ratio=math.exp(-energy)
    y,weights=roots_hermite(nodes)
    weights=weights/math.sqrt(math.pi)
    phi=np.column_stack([eval_hermite(n,y)/math.sqrt(2.**n*factorial(n)) for n in range(modes)])
    if np.linalg.norm(phi.T@(weights[:,None]*phi)-np.eye(modes),2)>1e-10:
        raise ValueError('Hermite multiplication compression is not isometric')
    momentum=(.006,.004,.002)
    h0,p0,t0=fiber(kappa,momentum,0.)
    states2=supplier.occupations(4,2)
    states3=supplier.occupations(4,3)
    eig=eigh(h0,eigvals_only=True)
    common=p0*eig[-1]*eig[-2]
    f0=p0*supplier.exterior(h0,states2)/common
    fc0=p0*supplier.exterior(h0,states3)/common
    tv,tu=eigh(t0)
    electron=-math.log(float(max(tv[tv<1])))
    projector0=tu[:,tv<1]@tu[:,tv<1].conj().T
    source0=creation(math.exp(electron/2)*projector0/math.sqrt(2))
    fs=[];fcs=[];sources=[];bounds=[];source_bounds=[]
    for field in y/math.sqrt(s):
        half,scalar,transfer=fiber(kappa,momentum,float(field))
        fs.append(hermitian(scalar*supplier.exterior(half,states2)/common))
        fcs.append(np.kron(np.eye(4),hermitian(scalar*supplier.exterior(half,states3)/common)))
        values,vectors=eigh(transfer)
        band=vectors[:,values<1]@vectors[:,values<1].conj().T
        coefficients=math.exp(electron/2)*band/math.sqrt(2)
        sources.append(creation(coefficients))
        generator=2*np.linalg.norm(logarithm(half)-logarithm(h0),2)+abs(math.log(scalar/p0))
        bounds.append(math.expm1(generator))
        source_bounds.append(float(np.linalg.norm(coefficients-math.exp(electron/2)*projector0/math.sqrt(2),'fro')))
    fs=np.array(fs);fcs=np.array(fcs);sources=np.array(sources)
    fm=projected(fs,phi,weights)
    fcm=projected(fcs,phi,weights)
    sm=projected(sources,phi,weights)
    gm=np.diag(lam*ratio**np.arange(modes))
    gneutral=np.kron(gm,np.eye(6));gcharged=np.kron(gm,np.eye(16))
    kn=hermitian(fm@gneutral@fm)
    kc=hermitian(fcm@gcharged@fcm)
    kref=np.kron(gm,f0@f0)
    source_ref=np.kron(np.eye(modes),source0)
    kcref=np.kron(gm,np.kron(np.eye(4),fc0@fc0))
    ground=np.zeros((modes,modes));ground[0,0]=1
    prep=np.kron(ground,np.eye(6))
    field_b=np.diag(np.sqrt(np.arange(1,modes)),1)+np.diag(np.sqrt(np.arange(1,modes)),-1)
    b=np.kron(field_b,np.eye(6))
    # SAME one-mode precompensation as Source2a: exp(E) times
    # sqrt(2 beta sinh E)/omega times the physical Curl factor omega*x.
    b=math.exp(energy)*b
    reference_values=eigh(kref,eigvals_only=True)/lam
    reference_traces={str(power):float(np.sum(reference_values**power)) for power in (nt,nt+1,nt+2)}
    def gramtower(neutral,charged,source):
        xi=positive_power(neutral,.5)@prep@positive_power(neutral,nt/2)
        photon=positive_power(neutral,.5)@b@xi
        source_image=source@photon
        z0=float(np.linalg.norm(xi,'fro')**2)
        zph=float(np.linalg.norm(photon,'fro')**2)
        lambda0=float(eigh(neutral,eigvals_only=True)[-1])
        m0=float(np.linalg.norm(source_image,'fro')**2/zph)
        m1=float(np.linalg.norm(positive_power(charged,.5)@source_image,'fro')**2/(lambda0*zph))
        return dict(Z0=z0,Zph=zph,normalized_photon_Gram=zph/z0,M0=m0,M1=m1,
                    M1_minus_exp_minus_quarter_M0=m1-math.exp(-.25)*m0,
                    common_neutral_lambda0=lambda0,charged_max_over_common_neutral=float(eigh(charged,eigvals_only=True)[-1]/lambda0)),xi,photon
    actual,xi,photon=gramtower(kn,kc,sm)
    reference,xiref,photonref=gramtower(kref,kcref,source_ref)
    epsf=float(max(np.linalg.norm(f-f0,2) for f in fs))
    epscar=epsf*(2+epsf)
    actual_perturb=float(np.linalg.norm(kn-kref,2)/lam)
    if actual_perturb>epscar*(1+1e-7)+1e-12:
        raise ValueError('Measured CAR kernel perturbation violates weighted compression bound')
    return dict(hermite_modes=modes,quadrature_nodes=nodes,epoch='NEW leading-Gaussian physical-Root Bloch-fiber comparison; not Source2a backend',
        native_beta=beta,body_kappa=kappa,time_slices=nt,mode_omega=omega,mode_energy=energy,
        field_sigma=sigma,leading_ground_lambda=lam,mehler_ratio=ratio,
        temporal_coefficient_beta=beta,spatial_coefficient_beta=beta,
        actual=actual,reference=reference,
        photon_operator='exp(E) sqrt(2 beta sinh E) x, selected-mode physical precompensation; no global B cap',
        reference_all_canonical_normalized_Tr=reference_traces,
        reference_odd_over_even_Tr=reference_traces[str(nt+1)]/reference_traces[str(nt)],
        pointwise_neutral_SF_difference=epsf,all_occupation_log_Gamma_enclosure=max(bounds),
        CAR_weighted_kernel_bound=epscar,measured_kernel_norm_over_lambda=actual_perturb,
        noncommutation_norm=float(np.linalg.norm(fm@gneutral-gneutral@fm,2)),
        actual_source_operator_difference=float(np.linalg.norm(sm-source_ref,2)),source_CAR_frobenius_bound=max(source_bounds),
        relative_prepared_HS_difference=float(np.linalg.norm(xi-xiref,'fro')/np.linalg.norm(xiref,'fro')),
        relative_B_graph_difference=float(np.linalg.norm(b@(xi-xiref),'fro')/np.linalg.norm(b@xiref,'fro')),
        relative_photon_HS_difference=float(np.linalg.norm(photon-photonref,'fro')/np.linalg.norm(photonref,'fro')),
        canonical_neutral_states=6,canonical_charged_states_per_Ref=4,Ref_channels=4,
        neutral_common_SF_half_scale=common,charged_own_max_normalization=False,
        Gaussian_polynomial_moments=dict(second=sigma**2,fourth=3*sigma**4,joint_second= sigma**4*(1+2*ratio**2)),
        comparison_preparation='Xi0=Kneutral^1/2 (|Gaussian-ground><Gaussian-ground| tensor I_Nc=2) Kneutral^(Nt/2); original Source2a preparation unchanged',
        limits=['One physical Bloch fiber, not full spatial CAR backend','Co-local Ref in this native control, not macro Wilson-string Gram','Finite Hermite comparison has no physical Haar-null admission','No physical fullPhi/statistical envelope imported'])


def macro_envelope(kappa):
    mp.mp.dps=100
    volume=mp.mpf(2)**72;length=mp.mpf(2)**24;nt=mp.mpf(2)**20
    beta=mp.mpf(10)**150;delta=mp.mpf(10)**-50
    bmin=1/(2*mp.mpf(str(kappa)))-3;bmax=bmin+6
    tmin=1/(16*bmax)
    dtcap=24*bmax+48*max(1,1/bmin**2)
    c=(2*volume)*dtcap*delta/(2*tmin)+3*volume*delta/bmin
    sf=mp.expm1(c)
    # On the declared tube; outside and long jumps retain their separate prices.
    epssf=sf*(2+sf)
    qmin=2*mp.sin(mp.pi/length)
    charge=4*volume/(beta*qmin**2)
    radius=length*mp.mpf(10)**-50;mass=beta*length
    harmonic_price=9*(nt+3)/(2*mass*radius**2)
    def row(value):
        return dict(value=mp.nstr(value,24),log10=mp.nstr(mp.log10(value),24))
    return dict(scope='Conditional mean-field Gaussian tube CAR envelope only; fullPhi/statistical kernel domination remains separate',
        active_kappa=kappa,bmin=mp.nstr(bmin,24),T_min_used_only_for_onebody_log=mp.nstr(tmin,24),
        all_occupation_SF_generator=row(c),SF_kernel_relative_error=row(epssf),
        retained_MF_scale='SF kernel bound multiplies ||F_N^0(theta)||² in the ONE neutral-common normalization; no independently shifted flat-holonomy or charged constants',
        contraction_proof='||Fprime Pq F-F0prime F0|| <= epsSF ||F0prime|| ||F0|| + ||I-Pq|| ||Fprime|| ||F||; Pq is positive diagonal, not commuted through matter and not a Loewner inference',
        all_Nt_SF_telescoping_relative_error=row(mp.expm1((nt+3)*mp.log1p(epssf))),
        positive_diagonal_Gauss_charge_I_difference_upper=row(charge),
        original_Source2a_toron_filter_overlap_defect_at_Nt_plus_3=row(harmonic_price),
        original_filter_shape='Source2a f_physical_torons bump; this price uses its stated 9/(2 m R²) reference energy bound, not a Gaussian retag',
        original_R=mp.nstr(radius,24),original_mass=mp.nstr(mass,24),
        harmonic_weight='1 is a Schur weight, not an L² ground state',
        Gaussian_norm_tau_natural_width=float((1+nt)**(-mp.mpf(3)/4)),
        Gaussian_norm_tau_new_heat_width=float(2**(-mp.mpf(3)/4)),
        Gaussian_new_width_squared='Nt/(beta L): explicit NEW preparation cost, not original Source2a',
        Gaussian_same_scale_original_R_tau=mp.nstr((1+nt/(mass*radius**2))**(-mp.mpf(3)/4),90),
        no_extensive_vacuum_norm_loss=True,macro_source_admitted=False,
        actual_macro_M0=None,actual_macro_M1=None,actual_macro_normalized_photon_Gram=None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.perf_counter()
    if digest(HERE/'covariant_prepared_gauss_state.py')!=SOURCE_SHA or digest(HERE/'covariant_root_transfer.py')!=ROOT_SHA:
        raise ValueError('Frozen Source2a/Root6911 provenance changed')
    # Body kappa is the exact selected Source2a maximum, not an auxiliary value.
    kappa=.1248905293249106
    omega=2*math.sin(.005/2)
    rows=[native_control(m,n,kappa,1e13,omega,32) for m,n in ((12,48),(20,64),(28,80))]
    mp.mp.dps=100
    auxk=mp.mpf(10)**-32;auxn=10**141//16;auxb=1/(2*auxk)-3;r=1/auxb**2
    aux_terms=4*r**34+r**68
    aux_log=auxn*mp.log1p(aux_terms)
    report=dict(source_sha=SOURCE_SHA,root_sha=ROOT_SHA,consumer_sha=digest(__file__),
        selected_capsule=SELECTED,selected_capsule_sha=digest(HERE/SELECTED),
        governing_plan_revision='7650a3636f3d39a74243a986afb1683be8fae895c3f5afb77720b05330363a51',
        leading_law='KW=exp(-beta spatial_faces/2) exp(-beta temporal_links) exp(-beta spatial_faces/2), beta=16 Naux kappa_aux^4; quadratic coexact mode E=2asinh(omega/2)',
        leading_law_scope='Primitive mean-field kappa^4 field law, not imported RPA; non-Gaussian and fullPhi/statistical remainders are NOT silently absorbed',
        rows=rows,macro_conditional_envelope=macro_envelope(kappa),
        new_auxiliary_comparison_epoch=dict(kappa_power=32,auxiliary_count=str(auxn),beta=1e13,
            six_canonical_occupations_per_species=True,reference_partition_log=mp.nstr(aux_log,24),
            field_independent_spectator_only=True,not_interacting_Slater=True),
        largest_successive_M0_difference=max(abs(a['actual']['M0']-b['actual']['M0']) for a,b in zip(rows,rows[1:])),
        largest_successive_M1_difference=max(abs(a['actual']['M1']-b['actual']['M1']) for a,b in zip(rows,rows[1:])),
        runtime_seconds=time.perf_counter()-started,macro_source_admitted=False)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=args.output,runtime_seconds=report['runtime_seconds'],rows=[r['actual'] for r in rows],macro_source_admitted=False)))


if __name__=='__main__':
    main()
