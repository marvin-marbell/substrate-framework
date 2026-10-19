#!/usr/bin/env python3
"""Discriminate bare screened SOURCE from Peierls-expanded photon functions.

Original fn/Gamma preparation and its accepted fast graph are unchanged.
The genuinely new execution checks literal source/Peierls/mixed jets and the
actual generic creator callback, not a historical scientific rerun.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import expm
from functional_source_photon_bank import (
    FunctionalSourcePhotonBank, Full108ReferenceMarks, FullPacketWeylTerm,
)
from prepared_neutral_weyl_hard_fock import GaussianFunction, ProjectionPolynomial
from prepared_neutral_shared_photon_state import load_shared_native_cache
from noncompact_screened_gauss_dressing import screened_marks, divergence, green
from prepared_guard_colour_chart import Jet2, onebody_coordinate_jets
from improved_phi_band import NSTAR, UE, C
from raw_source_relative_noether_graph import runtime_binding

EXPECTED = {
    'cache_npz': '10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3',
    'cache_manifest': '90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a',
    'original_legal': '10925ce66ab64c8f4c6a8fd8574baf183539be5827ac4dd495545357b28ab03e',
    'transport_capsule': '1c80504c0006c3145a5e5e18891d62e9453a23300f12d378e2d765ec74476224',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational_log2_upper(value: Fraction) -> int:
    n,d=value.numerator,value.denominator
    k=n.bit_length()-d.bit_length()
    fits=n<=d<<k if k>=0 else n<<(-k)<=d
    return k if fits else k+1


def generic_extension(legal: dict[str,Any]) -> dict[str,Any]:
    """Conditional operator-image extension; NOT a term in bare preparation."""
    phase_squared=9*UE*NSTAR**3/C
    delta_log=(rational_log2_upper(phase_squared)+1)//2
    graph=legal['science']['NEW_true_photon_initial_fast_graph']
    return dict(
        bare_original_graph_UNCHANGED=graph,
        bare_source_C_g='[a_g,Bbare]=0 EXACTLY: div Pcoeq=0; separate harmonic phase held, SU4 links are NOT U1 Peierls links',
        generic_C_g='For an ACTUAL finite Peierls multiplier B_eff=sum_path M_path(Phi,U) exp(i ell_path.A), C_g=[a_g,B_eff]=D_g B_eff/sqrt2; no such term is added to Bbare',
        derivative_contract='sumlegs/path is independent of Phi/SU4. Thus D_Phi^j C_g=i(sumlegs.g)/sqrt2 times D_Phi^j M_path, j<=2, with absolute path/Pauli-l1 envelopes (NOT cancellation-based operator norms)',
        canonical_photon_normalization='A=sum_m sqrt(2UE/omega_m) u_m y_m; [a,B_eff]=D_y B_eff/sqrt2',
        physical_omega_min_lower='4c/N for nonzero coexact modes; harmonics have no coexact creator',
        conservative_phase_insertion_squared='9 UE N^3/c, using ||ell||2<=||ell||1<=3N and omega_min>=c/N (weaker than actual4c/N)',
        conservative_phase_insertion_norm_log2_upper=delta_log,
        actual_N=str(NSTAR),actual_c=str(C),actual_UE_log2_upper=rational_log2_upper(UE),
        missing_term_IF_nonzero='a Gamma B_eff f tau=Gamma B_eff a f tau+Gamma C_g f tau; K Gamma C_g f tau=Gamma C_g K f tau+[K,Gamma C_g]f tau',
        mixed_fast_commutator='chi g D(Gamma C_g).Pboost +chi Delta_g(Gamma C_g)/2 +actual beta/body covariant commutator; retain original Gamma first Lambda^1/4 and second Lambda^1/2 jets',
        generic_annihilation_fast_power_IF_nonzero='Lambda^1/2 (not the bare source Lambda^1/8 pullthrough); same ultimate creator Lambda^1/2 power via exact CCR, constants must include ACTUAL expanded coefficients and mixed jets',
        no_generic_graph_admission_without_actual_expanded_jet_binding=True,
        original_fn_Gamma_SOURCE_norm_parameter_and_state_NOT_changed=True,
        no_photon_weighted_boundary_or_scattering_claim=True,
    )


def native_scope(model: Any,data: dict[str,Any]) -> dict[str,Any]:
    bank=FunctionalSourcePhotonBank(model,data)
    packet=np.asarray(bank.packet)
    if abs(np.vdot(packet,packet).real-1)>1e-10:
        raise ValueError('Actual original incoming packet must have its original unit norm')
    probe=packet.real
    if np.linalg.norm(probe)<1e-10:
        probe=packet.imag
    probe=probe/np.linalg.norm(probe)
    phi=model.chart.mean_phi;links=model.chart.mean_links
    a0=np.zeros((model.law.volume,3));harmonic=np.zeros(3)
    em_direction=(model.em['legs']@probe).reshape(model.law.volume,3)
    green_matrix,_,_=green(model.law)
    coulomb_coefficients=np.column_stack([green_matrix@divergence(model.law,model.em['legs'][:,j].reshape(model.law.volume,3)) for j in range(52)])
    coulomb_coefficients-=coulomb_coefficients[model.reference_site]
    bare=[]
    for profile in ('source19','SOFT'):
        eta=model.source_bank.profiles[profile]
        value=screened_marks(model.law,phi,links,a0,eta,model.reference_site,18.,harmonic)
        moved=screened_marks(model.law,phi,links,a0+em_direction,eta,model.reference_site,18.,harmonic)
        bare.append(dict(profile=profile,all4word_max_difference=float(np.max(np.abs(moved-value))),
                         exact_coexact_source_word='zeros52 from div Pcoeq=0',all_Ref4_preserved=True))
    if max(row['all4word_max_difference'] for row in bare)>1e-11:
        raise AssertionError('Literal bare SOURCE is not the claimed coexact Coulomb source')
    paths=list(model.law.terms(phi,links,a0))
    frequencies=[sum((model.em['legs'][3*x+axis] for x,axis in path),np.zeros(52)) for _,_,_,path in paths]
    scores=[abs(k@packet)*np.linalg.norm(block) for k,(_,_,block,_) in zip(frequencies,paths)]
    chosen=int(np.argmax(scores));x,y,block,path=paths[chosen];frequency=frequencies[chosen]
    if scores[chosen]<=1e-12:
        raise AssertionError('Actual literal Peierls path has no discriminating original packet derivative')
    size=4*model.law.volume
    onsite=np.zeros((size,size),complex)
    for site in range(model.law.volume):
        onsite[4*site:4*site+4,4*site:4*site+4]=model.law.onsite*phi[site,0]
    operator_terms=[(onsite,np.zeros(52))]
    for (px,py,pblock,_),pfrequency in zip(paths,frequencies):
        forward=np.zeros((size,size),complex)
        forward[4*px:4*px+4,4*py:4*py+4]=pblock
        operator_terms.extend(((forward,pfrequency),(forward.conj().T,-pfrequency)))
    point=np.sin(np.arange(52)+.3)*.025
    constant=ProjectionPolynomial.constant(52)
    field_point=(model.em['legs']@point).reshape(model.law.volume,3)
    actual_h=model.law.onebody(phi,links,field_point)
    d_h=(model.law.onebody(phi,links,field_point,first=(model.em['legs']@packet.real).reshape(model.law.volume,3))
         +1j*model.law.onebody(phi,links,field_point,first=(model.em['legs']@packet.imag).reshape(model.law.volume,3)))
    callback=[]
    for profile in ('source19','SOFT'):
        for word in range(4):
            raw=model.source_bank.seed_fibre.source_vectors_at(phi,links,np.zeros(3),a0,
                        model.source_bank.profiles[profile],np.eye(4)[word])
            marks=Full108ReferenceMarks(raw['full108_plus_coefficients'],raw['full108_minus_coefficients'])
            bare_term=FullPacketWeylTerm(GaussianFunction(constant,np.zeros(52)),marks)
            observed_plus=np.zeros_like(marks.plus);observed_minus=np.zeros_like(marks.minus)
            for pbody,pfrequency in operator_terms:
                expanded=bare_term.light_product(pbody,pfrequency)
                physical_creator=expanded.incoming_creator(packet)
                polynomial,_=physical_creator.function.polynomial.evaluate(point)
                scalar=polynomial*np.exp(1j*(physical_creator.function.phase@point))
                observed_plus+=scalar*expanded.marks.plus
                observed_minus+=scalar*expanded.marks.minus
            original_image=marks.light_action(actual_h)
            derivative_image=marks.light_action(d_h)
            multiplier=np.sqrt(2)*(packet@point)
            expected_plus=multiplier*original_image.plus-derivative_image.plus/np.sqrt(2)
            expected_minus=multiplier*original_image.minus-derivative_image.minus/np.sqrt(2)
            error=float(np.sqrt(np.linalg.norm(observed_plus-expected_plus)**2+np.linalg.norm(observed_minus-expected_minus)**2))
            wrong_error=float(np.sqrt(np.linalg.norm(observed_plus-multiplier*original_image.plus)**2+np.linalg.norm(observed_minus-multiplier*original_image.minus)**2))
            callback.append(dict(profile=profile,word=word,bothmaps_and_full108_Ref4=True,
                actual_full_creator_physical_image_error=error,
                rejected_multiplication_only_physical_image_norm=wrong_error,
                ALL_original_onsite_forward_reverse_Peierls_terms=len(operator_terms)))
    if max(row['actual_full_creator_physical_image_error'] for row in callback)>1e-10:
        raise AssertionError('Actual generic creator callback omits derivative or has wrong sign')
    if min(row['rejected_multiplication_only_physical_image_norm'] for row in callback)<=1e-15:
        raise AssertionError('Full original all4/bothmaps Hamiltonian images failed the adverse discriminator')
    # True Cartesian Phi + SU4 path direction, with independent coexact EM direction.
    algebra=np.zeros_like(links);edge,axis=path[0];algebra[edge,axis]=model.law.generators[0]
    dp=np.zeros_like(phi);dp[x,axis+1]=model.law.basis[0]
    dl=1j*np.einsum('xaij,xajk->xaik',algebra,links)
    ddl=-np.einsum('xaij,xajk,xakl->xail',algebra,algebra,links)
    zero_phi=Jet2(phi,np.zeros_like(phi),np.zeros_like(phi))
    phi_jet=Jet2(phi,dp,np.zeros_like(phi))
    zero_link=Jet2(links,np.zeros_like(links),np.zeros_like(links))
    link_jet=Jet2(links,dl,ddl)
    zero_em=Jet2(a0,np.zeros_like(a0),np.zeros_like(a0))
    em_jet=Jet2(a0,em_direction,np.zeros_like(a0))
    all_jets=onebody_coordinate_jets(model.law,phi_jet,link_jet,em_jet)
    spatial=onebody_coordinate_jets(model.law,phi_jet,link_jet,zero_em)
    electromagnetic=onebody_coordinate_jets(model.law,zero_phi,zero_link,em_jet)
    mixed=(all_jets.second-spatial.second-electromagnetic.second)/2
    fd=[]
    def h(s: float,t: float) -> np.ndarray:
        moved=links.copy();moved[edge,axis]=expm(1j*s*algebra[edge,axis])@links[edge,axis]
        return model.law.onebody(phi+s*dp,moved,a0+t*em_direction)
    for step in (2.**-7,2.**-8):
        numerical=(h(step,step)-h(step,-step)-h(-step,step)+h(-step,-step))/(4*step**2)
        fd.append(dict(step=step,absolute_error=float(np.linalg.norm(numerical-mixed)),
                       relative_error=float(np.linalg.norm(numerical-mixed)/np.linalg.norm(mixed))))
    if np.linalg.norm(mixed)<1e-12 or fd[-1]['relative_error']>2e-4:
        raise AssertionError('Actual mixed Phi/colour/coexact Peierls jet was not observed')
    return dict(scope='NEW native source-contract/creator/mixed-jet discriminator; not full-H/timeflow or re-execution of T13/T14',
        original_packet_norm_squared=float(np.vdot(packet,packet).real),
        all52_Coulomb_coexact_coefficient_max=float(np.max(np.abs(coulomb_coefficients))),
        bare_SOURCE=bare,actual_Peierls_path=list(path),actual_path_length=len(path),
        actual_phase_packet_dot=dict(real=float((frequency@packet).real),imag=float((frequency@packet).imag)),
        original_generic_creator_all4_bothmaps=callback,
        actual_mixed_Phi_colour_coexact_onebody_norm=float(np.linalg.norm(mixed)),mixed_second_jet_controls=fd,
        OMIT_mixed_jet_branch_norm=float(np.linalg.norm(mixed)),
        coexact_probe='real or imaginary component of ORIGINAL packet; probe only, original preparation/incoming packet unchanged')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED:
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-archive',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    for name,expected in EXPECTED.items():
        if sha(getattr(args,name))!=expected:
            raise ValueError('Not the original frozen supplier: '+name)
    legal=json.loads(args.original_legal.read_bytes())
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    science: dict[str,Any]=dict(native=native_scope(model,data),generic_extension=generic_extension(legal),
        precise_scope='Bare screened SOURCE commutes coexact creator. Generic H/J/G/F functions require full factored derivative; no invalidation or repair of historical10925.',
        physical_frame_contract='In a genuinely field-dependent fullFock frame use Udagger a† U=a†−Udagger(Dg U)/sqrt2 and transform fast/body/field operators consistently; no eight-ray recapture or discarded PH connection',
        original_legal_source_and_receipt_NOT_modified=True,full_spatial2pi_Born_weighted_boundary_NOT_claimed=True)
    binding=runtime_binding(args.runtime_archive)
    result=dict(science=science,argv=sys.argv,source_sha256=sha(Path(__file__)),
        inputs_sha256={name:sha(getattr(args,name)) for name in EXPECTED},runtime_binding=binding,
        original_legal_receipt_sha256=sha(args.original_legal),wall_seconds=time.perf_counter()-started,
        Python=sys.version.split()[0],metadata_allowlist='explicit argv/input/source/runtime/scientific quantities only; no ENV')
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(args.output),output_sha256=sha(args.output),source_sha256=result['source_sha256'],
        bare_B_coexact_commutator_scope_correct=True,actual_mixed_jet_norm=science['native']['actual_mixed_Phi_colour_coexact_onebody_norm'],
        original_LEGAL_graph_unchanged=True,wall_seconds=result['wall_seconds']),indent=2))


if __name__=='__main__':
    main()
