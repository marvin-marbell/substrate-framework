#!/usr/bin/env python3
"""Proof-only interior prices for unchanged Source2a Pe and all9 Root tuples.

No reference Gram is substituted for an interacting Gram. Original nominal
source projectors and prefactors stay nominal; actual Root per-case charge
energies, projector overlaps and common SF scales are independently consumed.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy.linalg import eigh
import mpmath as mp
import covariant_actual_source_band_consumer as consumer
import covariant_canonical_gauss_boundary_kernel as supplier
import covariant_physical_mf_vacuum_winding as meanfield
from covariant_induced_photon import Parameters,WilsonConsumer,phi_frame

ROOT=Path(__file__).resolve().parent


def stable_band(gamma: np.ndarray,a: float,mass: float,k: np.ndarray) -> tuple[np.ndarray,float]:
    kappa=1/(2*(math.expm1(a*mass)+4))
    links=[np.exp(1j*value)*np.eye(4) for value in k]
    _,_,T=supplier.physical_spatial_onebody(gamma,kappa,links,[np.zeros((4,4),complex) for _ in range(4)],np.eye(4))
    v,u=eigh(T);selected=v<1
    if np.count_nonzero(selected)!=2:raise ValueError('Original canonical positive-band rank changed')
    return u[:,selected]@u[:,selected].conj().T,-math.log(float(max(v[selected])))


def full_holonomy_source_price(a: float,mass: float,L: int,source_lower: float,
                              source_cap: float,photon_energy: float,soft_retained: float) -> dict[str,Any]:
    """Keep literal shortest integer paths, even at nonzero flat holonomy."""
    radius=128
    # Exact even-L DFT magnitude:
    # |sin(theta/2)|/[L|sin((theta-2pi n)/(2L))|]. Keep the antipodal jump.
    # For |n|<=L/4, |c_n|²<=1/[8(|n|-.5)²]; remaining frequencies <=4/L.
    # The coherent beam shifts shrink the core by1 and l1<=sqrt3 costs3.
    one_axis=1/(4*(radius-1.5))+4/L
    outside=9*one_axis
    core_p=math.sqrt(3)*(2*math.pi*(radius+1)+2*math.pi)/L
    incoming_p=math.sqrt(3)*3*math.pi/L
    case_projector=core_p/math.expm1(2*a*mass)
    nominal_projector=math.sqrt(3)*incoming_p/math.expm1(.02)
    projection=(case_projector+nominal_projector)**2
    retained_source=source_lower-source_cap*(outside+projection)
    b=math.exp(a*mass)
    case_energy=2*math.asinh(.5*math.sqrt(((b-1+core_p**2/2)**2+core_p**2)/b))
    M1=retained_source*math.exp(-case_energy-photon_energy)*soft_retained
    pe=(M1-math.exp(-.25)*source_cap)/(1-math.exp(-.25))
    if retained_source<=0 or pe<=0:raise ValueError('Unfiltered-flat-holonomy source price failed')
    return dict(proof_Fourier_core_radius=radius,all_continuous_and_flat_centre_holonomies=True,
        all_3_coherent_beams=True,source_path_jump_retained=True,
        actual_uniform_1D_Wilson_DFT_tail_upper=one_axis,
        full_coherent_3D_electron_winding_outside_squared_norm_upper=source_cap*outside,
        exact_positive_band_projector_core_loss_upper=source_cap*projection,
        reference_raw_source_M0_lower=retained_source,
        reference_source_M1_lower=M1,reference_original_PE25_Gram_lower=pe,
        reference_charged_core_energy_upper=case_energy,
        reference_interior_high_source_tail_squared_norm_upper=source_cap*outside,
        proof='Flat SU4-centre/statistical sheets act as scalar holonomies in the mean-Clifford reference. Keep literal x,z,y shortest integer paths. Their antipodal jump gives the exact even-L DFT. Product tail and l1<=sqrt3 price EVERY coherent three-beam source. For canonical stable T, sin mixing angle<=|sin p|/(B²-1); comparing each case and the unchanged nominal projector to gamma0+ gives the squared projection loss. No centre/toron is selected or independently normalized.',
        scope='Cold canonical reference only; actual thermal remainder is separately priced by Nt/Nt+1/Nt+2 MF rows. Original operator/source comparison, aux complement/interference, normal fields and real-Gauss insertions remain required.')


def wilson_seam_smoke(gamma: np.ndarray,p: Parameters) -> dict[str,Any]:
    rng=np.random.default_rng(222091);rows=[];max_formula=0.;max_projector_excess=0.
    for L in (4096,8192):
        x=np.arange(L);signed=np.where(x<=L//2,x,x-L)
        momenta=np.fft.fftfreq(L)*L;radius=128
        for theta in (-math.pi,-math.pi/2,-.31,.37,math.pi/2,math.pi):
            primitive=np.exp(1j*theta*signed/L)
            coeff=np.fft.fft(primitive)/L
            denominator=L*(1-np.exp(1j*(theta-2*math.pi*momenta)/L))
            exact=(np.exp(-1j*theta)-1)*np.exp(1j*(theta-2*math.pi*momenta)*(L//2+1)/L)/denominator
            max_formula=max(max_formula,float(np.max(abs(coeff-exact))))
            beams=rng.standard_normal(3)+1j*rng.standard_normal(3);beams/=np.linalg.norm(beams)
            wave=primitive*sum(a*np.exp(2j*math.pi*k*(x-L/4)/L) for a,k in zip(beams,(-1,0,1)))
            spectrum=np.fft.fft(wave)/L
            tail=float(np.sum(abs(spectrum[abs(momenta)>radius])**2))
            upper=3*(1/(4*(radius-1.5))+4/L)
            if tail>upper+1e-12:raise ValueError('Actual coherent shortest-Wilson seam exceeds its uniform tail')
            rows.append(dict(L=L,theta=theta,actual_whole_coherent_tail=tail,proved_upper=upper))
    core=math.sqrt(3)*(2*math.pi*130)/p.spatial_size
    for a,mass in itertools.product(p.spacings,p.masses):
        Pzero,_=stable_band(gamma,a,mass,np.zeros(3))
        for _ in range(32):
            momentum=rng.uniform(-core/math.sqrt(3),core/math.sqrt(3),3)
            projector,_=stable_band(gamma,a,mass,momentum)
            B=math.exp(a*mass)+float(np.sum(1-np.cos(momentum)))
            bound=float(np.linalg.norm(np.sin(momentum)))/(B*B-1)
            actual=float(np.linalg.norm(projector-Pzero,2))
            max_projector_excess=max(max_projector_excess,actual-bound)
            if actual>bound+1e-12:raise ValueError('Actual physical canonical projector violates graph mixing bound')
    return dict(actual_shortest_path_DFT_rows=rows,exact_DFT_formula_max_error=max_formula,
        actual_physical_Root_projector_rows=288,max_projector_bound_excess=max_projector_excess,
        scope='NEW numerical proof probe; no original source/preparation change. Actual antipodal jump and all coherent phases retained.')


def run() -> dict[str,Any]:
    start=time.perf_counter()
    cfg=json.loads((ROOT/'covariant_prepared_gauss_state_input_kappa400_sigma2000_r2100.json').read_text())
    selected=json.loads((ROOT/consumer.SELECTED).read_text());macro=selected['macro']
    p=Parameters(**cfg['photon_parameters']);gamma,_=phi_frame(p)
    nominal=WilsonConsumer(gamma,.01,1.)
    photonE=np.array([nominal.photon_energy(np.array(q)) for q in macro['photon']['momenta']])
    packetmax=float(photonE.max());soft=macro['finite_box_thermal_partition']['mean_energy_upper']
    # The frozen reference photon boundary lower includes its ACTUAL toron
    # kinetic budget. Do not infer a zero toron energy from Gaussian width.
    retained=macro['photon']['reference_soft_spectator_boundary_Gram_lower']
    toronmean=1-soft-retained
    originalsourcecap=macro['source']['filledRef4_CAR_chart_operator_norm_squared_upper']
    lowerenergy=.00175;upperenergy=.2;cut=macro['exact_energy_preparation']['energy_cut']
    if cut!=.25:raise ValueError('Original Pe changed')
    rows=[]
    for a,mass in itertools.product(p.spacings,p.masses):
        beam=[]
        for eta in itertools.product((0,1),repeat=3):
            for offset in (-1,0,1):
                k=(np.array([offset,0,0])*2+np.array(eta))*math.pi/p.spatial_size
                Pnom,Enom=stable_band(gamma,.01,1.,k)
                Pcase,Ecase=stable_band(gamma,a,mass,k)
                overlap=float(np.trace(Pnom@Pcase).real/2)
                beam.append(dict(eta=list(eta),offset=offset,original_nominal_energy=Enom,
                    actual_case_energy=Ecase,actual_canonical_projector_overlap=overlap,
                    reference_raw_source_Gram=math.exp(Enom)*overlap,
                    reference_charged_boundary_factor=math.exp(Enom-Ecase)*overlap))
        m0min=min(row['reference_raw_source_Gram'] for row in beam)
        m0max=max(row['reference_raw_source_Gram'] for row in beam)
        m1min=min(row['reference_charged_boundary_factor'] for row in beam)*math.exp(-packetmax)*retained
        pe=(m1min-math.exp(-cut)*originalsourcecap)/(1-math.exp(-cut))
        if pe<=0:raise ValueError('All9 original-source reference Rayleigh price failed')
        mf=meanfield.reference_budget(a,mass,p.spatial_size,p.time_slices)
        holonomy=full_holonomy_source_price(a,mass,p.spatial_size,m0min,originalsourcecap,packetmax,retained)
        # Each grade2 source hard state has <=3 fermionic excitations. For all
        # p, 0<=dE/d(am)<=1: dE/dlogB=(B²-1-s²)/sqrt(D- D+), and dlogB/dam<=1.
        # Thus the frozen full125/all8 upper is enlarged, never recalibrated by
        # treating the new kappa-max comparison mass as the nominal source.
        hard=macro['exact_energy_preparation']['actual_grade2_support']['uniform_all8_full125_grade2_energy_upper']+3*max(0,a*mass-.01)
        reserve=upperenergy-hard-.02-1e-6
        if reserve<=0:raise ValueError('Proof interior needs a paid positive source energy buffer')
        partition=macro['finite_box_thermal_partition']['rows']
        chernoff=min((r['logZ_upper']-partition[0]['logZ_lower']-(p.time_slices-r['inverse_temperature'])*reserve)/math.log(10)
            for r in partition[1:])
        with mp.workdps(100):
            L=mp.mpf(p.spatial_size);R=L*mp.mpf('1e-50');bodymass=mp.mpf(10)**150*L
            threshold=mp.sqrt(2*bodymass*mp.mpf('.02')/3)/2
            bump=mp.log10(mp.mpf('393.75')/mp.pi)-5*mp.log10(R*threshold)
            warm=mp.log10(4)-mp.mpf('.9')*p.time_slices*mp.mpf('.02')/(12*mp.log(10))
            gap=2*mp.sinh(mp.mpf(str(a*mass)))
            jet=8*16*(1+mp.mpf('1.5')+3*512/(gap*L)+100*30720/L)**3
            sourcejet=mp.log10(mp.mpf(2)**40*jet**2/mp.mpf('.5'))
            torontail=mp.log10(3)+max(bump,warm)+mp.log10(1+mp.mpf(10)**(-abs(bump-warm)))+sourcejet
        photonclipprice=math.log10(macro['photon']['packet_operator_norm_upper']**2/retained)+math.log10(originalsourcecap/.5)
        # Lower-energy outliers are absent in the EXACT free common-SF
        # reference because the charged norm gap is case-specific. Full law
        # tails remain NULL until operator/source differences are paid.
        rows.append(dict(a=a,m=mass,original_nominal_source_preserved=True,all8_times3_beams=beam,
            reference_raw_M0_lower=m0min,reference_raw_M0_upper=m0max,
            reference_full_halfstep_M1_lower=m1min,reference_original_PE25_Gram_lower=pe,
            all_flat_holonomy_source_price=holonomy,
            reference_grade2_hard_support_upper=hard,proof_soft_energy_reserve=reserve,
            proof_reference_photon_high_tail_log10_upper=chernoff+photonclipprice,
            proof_reference_toron_source_high_tail_log10_upper=float(torontail),
            original_physical_band_separation=float(gap),reference_toron_source_jet_cap=float(jet),
            free_common_SF_charged_contraction_upper=float(mf['SF_charged_common_neutral_norm_upper']),
            free_common_full_K_charged_contraction_upper=float(mf['SF_charged_common_neutral_norm_upper'])**2,
            reference_low_energy_source_outlier_zero=True,
            free_MF_winding_log10=mf['winding']['log10_vacuum_twist_error_upper']))
    return dict(governing_plan_revision=consumer.PLAN_REVISION,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_Source2a_sha256=consumer.digest(ROOT/'covariant_prepared_gauss_state.py'),
        frozen_Root6911_sha256=consumer.digest(ROOT/'covariant_root_transfer.py'),
        selected_capsule_sha256=consumer.digest(ROOT/consumer.SELECTED),
        existing_Pe_energy_band=[0.,cut],proof_interior_energy_band=[lowerenergy,upperenergy],
        witness_changes_preparation=False,all_parameter_tuple_reference_prices=rows,
        actual_Wilson_seam_and_canonical_projector_smoke=wilson_seam_smoke(gamma,p),
        original_all125_photon_modes=len(photonE),actual_source_coherence_scope='Bounds uniform over ALL complex l2-normalized original3 beams and all8 statistical torons; nominal source projector/prefactors not replaced by per-case source.',
        reference_M1_scope='Zero-continuous/colour-holonomy beam rows are exact free reference contractions, NOT uniform original-source bounds. The separate all_flat_holonomy_source_price retains every literal Wilson-path jump and its Fourier/projection loss. Both include complete charged boundary and actual soft/toron lower; microscopic Gauss/Phi/auxiliary/CAR comparison errors remain separate.',
        reference_tail_scope='Proof witness strictly INSIDE original Pe; hard support enlarged using proven global Wilson mass Lipschitz bound. Actual original toron bump and corrected per-case gap price its source jets. Full operator K-null/above-lambda0/outlier tails are NOT assigned these reference values.',
        Nt_Nt1_Nt2_scope='Positive free spectator energy makes its normalized Gibbs Chernoff tail nonincreasing with added inverse time. Actual inserted J trace/power ratios are not inferred from this monotonicity.',
        macro_actual_M0=None,macro_actual_M1=None,macro_actual_Pe_Gram=None,macro_source_admitted=False,
        elapsed_seconds=time.perf_counter()-start)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run();args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),elapsed_seconds=result['elapsed_seconds'],
        worst_zero_holonomy_reference_PE25_lower=min(r['reference_original_PE25_Gram_lower'] for r in result['all_parameter_tuple_reference_prices']),
        worst_all_holonomy_reference_PE25_lower=min(r['all_flat_holonomy_source_price']['reference_original_PE25_Gram_lower'] for r in result['all_parameter_tuple_reference_prices']),
        macro_source_admitted=False)))


if __name__=='__main__':main()
