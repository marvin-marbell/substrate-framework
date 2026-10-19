#!/usr/bin/env python3
"""Exercise original mechanical core, source-section adverse and transported band.

No original SOURCE/neutral state is renormalized or re-filtered. Native control
is not a LEGAL gate. The alternative finite-group packet construction retains
orbital pullback and centre gluing after the smooth SOURCE-section route fails.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from fractions import Fraction
import time
from typing import Any
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import expm
from physical_core_spin_module import (core_from_field,screened_noether,uniform_screen,
    packet_action,frame_path_gluing,positive_band_group,fundamental_colour_energy,legal_screening,
    finite_vertical_transport,finite_source_intertwiner,aligned_field_invariants,normal_compensated_source_jet,
    screened_axis_noether,fixed_boost_angular_comparison,mean_packet_rotation,physical_group_ledger,
    continuum_packet_group_control,flat_P_group_extension,encoded_reference_number_proof)
from prepared_neutral_shared_photon_state import load_shared_native_cache
from prepared_neutral_moving_matter_frame import mean_matter_generators
from prepared_neutral_phi_word_channels import coefficient_channels
from ordered_sector_reference import original_packet_input,covariant_packet_marks
from improved_phi_band import ImprovedPhiLaw,selected_band
from raw_source_relative_noether_graph import runtime_binding
from original_charged_principal import portable
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_full_onebody import full_coordinate_light_operators


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(error: float,label: str,upper: float=3e-7) -> None:
    if not np.isfinite(error) or error>upper:raise AssertionError(f'{label}: {error} > {upper}')


def native(model: Any,data: dict[str,Any]) -> tuple[dict[str,Any],dict[str,np.ndarray]]:
    law=model.law;frame=model.chart.mean_phi[0];core=core_from_field(law,frame)
    check(max(core['physical_vector_tangent_residuals']),'original field Noether tangent')
    check(core['SU2_commutator_error'],'derived Lie algebra')
    check(core['rest_Casimir_half_error'],'actual light positive rest band Casimir')
    g=mean_matter_generators(model,data);guard=json.loads(g['guard_path'].read_bytes())
    alpha=np.asarray(guard['alpha']);velocity=np.asarray(guard['velocity'])
    fields=model.chart.mean_phi;links=model.chart.mean_links;phases=np.zeros((law.volume,3))
    result=screened_noether(law,fields,links,phases,alpha,velocity,model.reference_site,model.original.reference_frame)
    fft=uniform_screen(law,core,alpha,2)
    check(result['equation_residual'],'actual screened equation')
    check(result['mass_horizontal_error'],'actual full mass orthogonality')
    check(float(np.linalg.norm(result['matrices']-fft)),'independent Fourier same screened operator')
    sample=model.basis_at(np.zeros(model.chart.normal_dim),np.zeros(55))
    arrays={'spins4':core['spins'],'rest_spins2':core['rest_spins'],'Phi_inertia15':core['Phi_inertia'],
        'mechanical_omega405':result['omega'],'mechanical_body27':result['matrices'],
        'true_Ref_RIGHT4':result['Ref_RIGHT'],'orbital108':result['orbital108'],'horizontal_RAW':result['horizontal_RAW']}
    profiles=[]
    for profile in ('source19','SOFT'):
        packetplus,packetminus=original_packet_input(model,profile)
        generic=covariant_packet_marks(model,sample,packetplus,packetminus)
        source=model.source_bank.seed_fibre.source_vectors_at(fields,links,sample['theta'],sample['a'],model.source_bank.profiles[profile])
        check(float(np.linalg.norm(generic['plus']-source['full108_plus_coefficients'])),'unchanged original packet input')
        check(float(np.linalg.norm(generic['minus']-source['full108_minus_coefficients'])),'unchanged original hole packet input')
        S=source['full108_plus_coefficients'].reshape(law.volume,4,4)
        pieces=packet_action(S,result['matrices'],result['Ref_RIGHT'],result['orbital108'])
        T=source['full108_minus_coefficients'].reshape(4,law.volume,4).transpose(1,0,2)
        hole=packet_action(T,result['matrices'],result['Ref_RIGHT'],result['orbital108'],charge=-1)
        _,metadata=coefficient_channels(data,'charged',profile);Z=metadata['global_SOURCE_normalizer']
        profiles.append(dict(profile=profile,original_globalZ=Z,raw_quartet=[1.,.3j,-.2+.4j,.6],
            raw_original_mark_norm=float(np.linalg.norm(S)),
            component_raw_norms={k:float(np.linalg.norm(v)) for k,v in pieces.items()},
            omission_Ref_RIGHT_defect=float(np.linalg.norm(pieces['Ref_RIGHT'])),
            minus_component_raw_norms={k:float(np.linalg.norm(v)) for k,v in hole.items()},
            component_norms_are_NOT_global_state_norms=True,full108_Ref4=True))
        for key,value in pieces.items():arrays[profile+'_'+key]=value
        for key,value in hole.items():arrays[profile+'_minus_'+key]=value
    # Adverse 1: a scalar smooth source section is exactly periodic. Its centre
    # compensator is measured at the actual core site; it cannot be dropped.
    site=int(np.argmax(abs(model.source_bank.profiles['source19'])))
    S0=model.source_bank.seed_fibre.source_vectors_at(fields,links,sample['theta'],sample['a'],model.source_bank.profiles['source19'])['full108_plus_coefficients']
    d=2.4;step=2e-6;s=core['spins'][2]
    def source_at(angle: float) -> np.ndarray:
        moved=fields.copy();u=expm(1j*angle*s);moved[site]=u@frame@u.conj().T
        return model.source_bank.seed_fibre.source_vectors_at(moved,links,sample['theta'],sample['a'],model.source_bank.profiles['source19'])['full108_plus_coefficients']
    final=source_at(2*np.pi);mid=source_at(d);fd=(source_at(d+step)-source_at(d-step))/(2*step)
    pure=np.zeros_like(mid);pure[4*site:4*site+4]=1j*s@mid[4*site:4*site+4]
    section: dict[str,Any]=dict(original_SOURCE_section_two_pi_plus_error=float(np.linalg.norm(final-S0)),
        wrong_SOURCE_section_minus_defect=float(np.linalg.norm(final+S0)),
        actual_scalar_word_jet_not_pure_left_defect=float(np.linalg.norm(fd-pure)),
        original_core_mark_norm=float(np.linalg.norm(S0[4*site:4*site+4])),
        seam_pi_core_mark_norm=float(np.linalg.norm(source_at(np.pi)[4*site:4*site+4])),
        result='Smooth original SOURCE section is periodic; bare quaternion-left halfspin identification fails. Retain centre/scalar compensation and test actual transported module instead.')
    check(section['original_SOURCE_section_two_pi_plus_error'],'actual source section periodicity')
    if section['wrong_SOURCE_section_minus_defect']<1e-4:raise AssertionError('Non-discriminating source-section adverse')
    gluing=frame_path_gluing(frame,s)
    check(gluing['retained_lift_minus_error'],'continued actual polar lift')
    # Adverse 2: rotating both endpoints produces integer global pair spin.
    full: Any=np.asarray([np.kron(s,np.eye(4))-np.kron(np.eye(4),s.T) for s in core['spins']])
    pair_casimir=np.linalg.eigvalsh(sum(s@s for s in full))
    pair_loop=float(np.linalg.norm(expm(-2j*np.pi*full[2])-np.eye(16),2))
    check(pair_loop,'neutral global pair integer control')
    # Different construction: an arbitrary packet transported by the derived
    # Noether fundamental plus scalar orbital pullback, not source-label fitting.
    bandcontrols=[positive_band_group(law,frame,core,p,np.array([.3,-.2,.4]))
                  for p in (np.array([2e-7,5e-8,-4e-8]),np.array([.8,.5,-.3]))]
    angle1=np.array([.31,-.17,.2]);angle2=np.array([-.11,.38,.13])
    U1=expm(-1j*np.einsum('j,jab->ab',angle1,core['spins']))
    U2=expm(-1j*np.einsum('j,jab->ab',angle2,core['spins']))
    from scipy.spatial.transform import Rotation
    R1=Rotation.from_rotvec(angle1).as_matrix();R2=Rotation.from_rotvec(angle2).as_matrix()
    # This point is generic, not an original packet retune. It tests f module.
    p=np.array([.11,-.23,.19]);rng=np.random.default_rng(222107);W=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    def f(k: np.ndarray) -> np.ndarray:
        return np.exp(-np.dot(k,k))*(1+.4j*k[0]+.2*k[1]**2)*W
    sequential=U1@U2@f(R2.T@R1.T@p)
    combined_angle=Rotation.from_matrix(R1@R2).as_rotvec()
    Ucombined=expm(-1j*np.einsum('j,jab->ab',combined_angle,core['spins']))
    joined=Ucombined@f((R1@R2).T@p)
    group_error=float(np.linalg.norm(sequential-joined))
    wrong=float(np.linalg.norm(U1@U2@f(p)-joined))
    check(group_error,'arbitrary packet group orbital composition')
    if wrong<1e-6:raise AssertionError('Non-discriminating orbital adverse')
    colour=fundamental_colour_energy(core,law,site,model.reference_site)
    arrays['continuous_colour_energy16']=colour['continuous_pair_energy_operator_over_chi']
    return dict(core={key:value for key,value in core.items() if key not in ('spins','rest_spins','positive_rest_frame','Phi_orbit_columns','Phi_inertia','Gauss_coefficients')},
        native_N=law.size,native_chi=model.chi,cache_static_boost=dict(Omega=model.boost['Omega'],Omega_dot=model.boost['Omega_dot'],Y0_norm=float(np.linalg.norm(model.boost['Y0']))),
        guard_sha256=sha(g['guard_path']),screen_equation=result['equation'],
        screening_controls=dict(equation_residual=result['equation_residual'],horizontal_error=result['mass_horizontal_error'],
            all15_Fourier_error=float(np.linalg.norm(result['matrices']-fft)),
            Ref_RIGHT_norm=float(np.linalg.norm(result['Ref_RIGHT'])),Ref_RIGHT_not_zero=True),
        original_profiles=profiles,failed_source_section_route=section,frame_lift_gluing=gluing,
        neutral_pair_control=dict(Casimir_eigenvalues=pair_casimir,two_pi_plus_error=pair_loop),
        alternative_packet_group=dict(group_composition_error=group_error,wrong_omitted_orbital_defect=wrong,
            actual_band_controls=bandcontrols,core_representation='derived fundamental4=two copies of j1/2; positive rest band2 is one copy',
            physical_fullfield_intertwining_NOT_deduced_from_module_algebra=True),
        fundamental_colour=colour,native_is_control_NOT_admission_gate=True),arrays


def aligned_native(model: Any,data: dict[str,Any]) -> tuple[dict[str,Any],dict[str,np.ndarray]]:
    """NEW finite aligned field/link factor; old Phi-only science is not rerun."""
    q,em,g=source_point(model,data);sample=model.basis_at(q,em)
    guard=json.loads(g['guard_path'].read_bytes());point=sample['point']
    flow=screened_noether(model.law,point['phi'],point['links'],sample['a'],
        np.asarray(guard['alpha']),np.asarray(guard['velocity']),model.reference_site,model.original.reference_frame)
    packets=(model.photon_basis.packets@data['packet_templates'].T).T
    incoming=(model.em['legs']@packets[0]).reshape(model.law.volume,3)/np.sqrt(2)
    outgoing=(model.em['legs']@packets[2].conj()).reshape(model.law.volume,3)/np.sqrt(2)
    current0=full_coordinate_light_operators(model.law,point['phi'],point['links'],sample['a'],incoming,outgoing)
    heavy0=model.heavy_law.onebody(point['phi'],point['links'],np.zeros_like(sample['a']))
    occupied=model.original.heavy_projector
    initial_heavy_trace=float(np.trace(occupied@heavy0).real)
    rows=[];arrays={'original_q':q,'original_em55':em,'mechanical_matrices':flow['matrices']}
    mean=model.chart.evaluate(np.zeros(model.chart.normal_dim))
    n=model.chart.normal_dim
    fixed_covector=np.linalg.solve(model.chi*mean['cometric'][:n,:n],g['Y0'])
    jets=[normal_compensated_source_jet(model,sample,flow,fixed_covector,profile)
          for profile in ('source19','SOFT')]
    for jet in jets:
        for error in jet['rows'][-1]['mechanical_CORE_orbital_RIGHT_identity_errors']:
            check(error,'actual original functional SOURCE normal-kinetic/Noether identity',upper=1e-5)
    rng=np.random.default_rng(22210731)
    arbitrary=rng.normal(size=(model.law.volume,4,4))+1j*rng.normal(size=(model.law.volume,4,4))
    arbitrary_minus=arbitrary.conj().transpose(0,2,1)
    generic0=covariant_packet_marks(model,sample,arbitrary,arbitrary_minus)
    for angle in (.7,np.pi,2*np.pi):
        transport=finite_vertical_transport(model,sample,flow['matrices'],float(angle))
        invariant=aligned_field_invariants(model.law,sample,transport)
        for key,error in invariant.items():check(error,'actual aligned gauge orbit '+key)
        U=transport['body']
        current=full_coordinate_light_operators(model.law,transport['fields'],transport['links'],sample['a'],incoming,outgoing)
        currents={key:float(np.linalg.norm(current[key]-U@current0[key]@U.conj().T))
                  for key in ('H','J','C')}
        for key,error in currents.items():check(error,'same ORIGINAL Peierls '+key+' covariance')
        heavy=model.heavy_law.onebody(transport['fields'],transport['links'],np.zeros_like(sample['a']))
        heavy_error=float(np.linalg.norm(heavy-U@heavy0@U.conj().T))
        moved_occupied=U@occupied@U.conj().T
        transported_trace=float(np.trace(moved_occupied@heavy).real)
        heavy_trace_error=abs(transported_trace-initial_heavy_trace)
        check(heavy_error,'full original heavy108 covariance')
        check(heavy_trace_error,'SAME heavy determinant exterior transport, not re-ground')
        moved_sample={**sample,'point':{**point,'phi':transport['fields'],'links':transport['links']}}
        generic=covariant_packet_marks(model,moved_sample,arbitrary,arbitrary_minus)
        R=transport['reference']
        generic_errors=dict(plus=float(np.linalg.norm(generic['plus']-U@generic0['plus']@R.conj().T)),
                            minus=float(np.linalg.norm(generic['minus']-R@generic0['minus']@U.conj().T)))
        for key,error in generic_errors.items():check(error,'full arbitrary packet/Ref coframe ABI '+key)
        source_rows=[]
        for profile in ('source19','SOFT'):
            value=finite_source_intertwiner(model,sample,transport,profile)
            for key in ('exact_field_factor_plus_error','exact_field_factor_minus_error',
                        'full_Gauss_trivial_plus_error','full_Gauss_trivial_minus_error'):
                check(value[key],'literal original finite SOURCE '+key)
            source_rows.append(dict(profile=profile,**{key:entry for key,entry in value.items()
                if key not in ('plus','minus','original_plus','original_minus')}))
            arrays[f'{profile}_angle{angle}_plus']=value['plus']
            arrays[f'{profile}_angle{angle}_minus']=value['minus']
        rows.append(dict(angle=float(angle),fullfield_regular_orbit=invariant,
            ORIGINAL_H_J_contact_errors=currents,heavy108_covariance_error=heavy_error,
            true_transported_neutral_heavy_trace_error=heavy_trace_error,
            wrong_frozen_heavy_projector_trace_defect=abs(float(np.trace(occupied@heavy).real)-initial_heavy_trace),
            source_operator_intertwiners=source_rows,
            arbitrary_full_packet_Ref_module_errors=generic_errors,
            local_finite_factor_NOT_compact_full_G_group=True))
    return dict(scope='Finite covariantly aligned fullPhi/link mechanical FIELD factor at actual original nonzero field and all55 EM; not original driven-H trajectory or global SO3',
        screening_controls=dict(equation_residual=flow['equation_residual'],horizontal_error=flow['mass_horizontal_error']),
        actual_point_q_norm=float(np.linalg.norm(q)),actual_EM55_norm=float(np.linalg.norm(em)),rows=rows,
        normal_kinetic_SOURCE_operator_jets=jets,
        relative_B0_correction='When B0 is genuinely constant-horizontal, DY0 B0=0; nonzero Omega_start and betadot(a.q) are COMMON and not charged as B0-relative costs. Original functional B(q) jet residuals here are not charged again after initial B-B0 transfer.',
        exact_operator_comparison='C_f(g.X)=U_L(g) C_f(X) U_R(g)^dagger, both S/T; full field+matter Gauss transport cancels. FIELD factor acts as core/reference compensator only after normal kinetic cancellation.',
        full_state_comparison='||U_C^V(t) B0(f) Xi - B0(f_t) U_N^V(t) Xi|| <= initial SAME-Z B-B0 error + integral ||[H_C^V B0-B0(H_N^V+h_packet)-i dotB0] Xi_t|| dt',
        remaining='A must bind actual boost/parallel-frame covariant normal kinetic cross and residual, slow originalH/EM/Coul/current. Finite aligned Gauss factor alone is not physical spin or whole compact2pi.',
        earlier_Phi_only_fixed_link_curve_is_NOT_original_Gamma_or_driven_transport=True,
        no_heavy_recapture_no_source_renormalization=True),arrays


def group_native(model: Any,data: dict[str,Any]) -> tuple[dict[str,Any],dict[str,np.ndarray]]:
    """NEW same-neutral module/all64 operator controls; not native orbital-grid SO3."""
    q,em,g=source_point(model,data);actual=model.basis_at(q,em)
    mean=model.basis_at(np.zeros(model.chart.normal_dim),em)
    guard=json.loads(g['guard_path'].read_bytes());alpha=np.asarray(guard['alpha'])
    axis_list=np.r_[np.eye(3),np.array([[1.,2.,3.]])/np.sqrt(14)]
    frame=model.chart.mean_phi[0];core=core_from_field(model.law,frame)
    matrices=[];axis_rows=[]
    n=model.chart.normal_dim
    fixed_covector=np.linalg.solve(model.chi*mean['point']['cometric'][:n,:n],g['Y0'])
    screens=np.asarray([uniform_screen(model.law,core,alpha,j) for j in range(3)])
    for axis in axis_list:
        velocity=alpha[:,None]*np.cross(axis,model.law.sites-np.asarray(guard['origin']))
        mean_flow=screened_axis_noether(model.law,mean['point']['phi'],mean['point']['links'],mean['a'],
            alpha,velocity,model.reference_site,axis,model.original.reference_frame)
        fft=np.einsum('j,jxab->xab',axis,screens)
        check(float(np.linalg.norm(fft-mean_flow['matrices'])),'arbitrary axis ORIGINAL Gauss versus Fourier')
        matrices.append(mean_flow['matrices'])
        actual_flow=screened_axis_noether(model.law,actual['point']['phi'],actual['point']['links'],actual['a'],
            alpha,velocity,model.reference_site,axis,model.original.reference_frame)
        # Y0 is the original z pulse; other axes are coefficient controls, not a new initial phase.
        angular=fixed_boost_angular_comparison(model,actual,actual_flow,mean_flow,fixed_covector)
        check(angular['Cartesian_vs_actual_chart_mechanical_error'],'actual whole normal/Gauss chart coefficient')
        for row in angular['rows']:check(row['exact_mechanical_decomposition_error'],'actual fixed-boost coefficient identity')
        axis_rows.append(dict(axis=axis,actual_fixed_boost_angular=angular))
    centre_rows=[];max_field=0.;max_packet=0.;max_exterior=0.
    for number in range(64):
        sector=np.array([(number//(4**j))%4 for j in range(3)])
        links=mean['point']['links'].copy()
        for j,k in enumerate(sector):
            links[model.law.sites[:,j]==model.law.size-1,j]*=1j**int(k)
        alias=number%4
        z=np.asarray([1j**int(alias*np.sum(x)%4) for x in model.law.sites])
        links=z[:,None,None,None]*links*z[model.law.target].conj()[:,:,None,None]
        sample={**mean,'point':{**mean['point'],'links':links}}
        # All sectors exercise the endpoint; all axes are represented and also proved linearly.
        K=matrices[number%len(axis_list)]
        field=finite_vertical_transport(model,sample,K,2*np.pi)
        profiles=[]
        for profile in ('source19','SOFT'):
            born=finite_source_intertwiner(model,sample,field,profile)
            field_error=max(born[key] for key in ('exact_field_factor_plus_error','exact_field_factor_minus_error',
                'full_Gauss_trivial_plus_error','full_Gauss_trivial_minus_error'))
            check(field_error,'all64 ORIGINAL scalar centre word telescope bothmaps endpoint')
            plus,minus=original_packet_input(model,profile)
            packet=mean_packet_rotation(model,sample,plus,minus,K,2*np.pi)
            for key in ('plus_error','minus_error','actual_seeded_neutral_full108_exterior_norm_squared_difference',
                        'derived_Ref3_hole_exterior_error','derived_neutral_Ref4_singlet_error'):
                check(packet[key],'all64 same-neutral fullCAR/exterior '+key)
            max_field=max(max_field,field_error)
            max_packet=max(max_packet,packet['plus_error'],packet['minus_error'])
            max_exterior=max(max_exterior,packet['actual_seeded_neutral_full108_exterior_norm_squared_difference'])
            profiles.append(dict(profile=profile,field_factor_endpoint_error=field_error,packet_exterior=packet))
        centre_rows.append(dict(colour_centre=sector,alias_control=alias,axis=axis_list[number%len(axis_list)],profiles=profiles))
    rng=np.random.default_rng(2222189)
    arbitrary=rng.normal(size=(model.law.volume,4,4))+1j*rng.normal(size=(model.law.volume,4,4))
    generic=[]
    for axis,K in zip(axis_list,matrices):
        row=mean_packet_rotation(model,mean,arbitrary,arbitrary.conj().transpose(0,2,1),K,.73)
        for key in ('plus_error','minus_error','actual_seeded_neutral_full108_exterior_norm_squared_difference',
                    'derived_Ref3_hole_exterior_error','derived_neutral_Ref4_singlet_error'):
            check(row[key],'arbitrary full packet SAME-neutral module '+key)
        generic.append(dict(axis=axis,**row))
    return dict(scope='NEW exact internal-CAR/exterior module; original native cache/neutral unchanged, other centre/alias rows are operator controls not newly prepared states',
        screening_controls=dict(all64_field_endpoint_error=max_field,all64_internal_packet_error=max_packet,
            all64_transported_neutral_exterior_norm_error=max_exterior),
        arbitrary_axes_actual_coefficients=axis_rows,all64_colour_centre_and_alias_controls=centre_rows,
        arbitrary_full_packet_controls=generic,
        ordinary_colour64_NOT_statisticalflat8=True,
        orbital_unitarity_scope='Continuum packet pullback is unitary; arbitrary-axis interpolation on a finite108 grid is NOT claimed unitary. Internal U(x) and derived Ref R are unitary in original CAR metric. LEGAL lowmomentum/alias prices are separate.',
        actual_spatial_group_conditional_on_A_H_Noether_integral=True,
        full_field_matter_Gauss_triviality_NOT_physical_core_admission=True),dict(mechanical_axes=np.asarray(matrices))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-npz','cache-manifest','geometry','bridge','c2','metric','output','arrays','runtime-archive'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--finite-aligned-only',action='store_true')
    parser.add_argument('--mean-group-only',action='store_true')
    parser.add_argument('--packet-family',type=Path)
    parser.add_argument('--flat-P-extension-only',dest='flat_P_extension_only',action='store_true')
    parser.add_argument('--group-control',type=Path)
    parser.add_argument('--auxiliary',type=Path)
    args=parser.parse_args();started=time.monotonic()
    if sum((args.mean_group_only,args.finite_aligned_only,args.flat_P_extension_only))>1:
        parser.error('Select only one NEW scientific consumer path')
    expected={'cache_npz':'10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3',
        'cache_manifest':'90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a',
        'geometry':'de3b71e6ab6d2d0c2da71e762c84ea3716ae5f6ac9b169669a8d87a9fdb7d4d2',
        'bridge':'008231eb67c9c6b1620fe3e110504234646b8a934bc6501f89872fedf34dfcd2',
        'c2':'10925ce66ab64c8f4c6a8fd8574baf183539be5827ac4dd495545357b28ab03e',
        'metric':'a79cb016133d178ac716314e60dbdb63fc04a0b95f9fa4df6cdfd1e4304a0ffd'}
    inputs={}
    for key,value in expected.items():
        path=getattr(args,key);actual=sha(path)
        if actual!=value:raise ValueError('Not original bound input '+key)
        inputs[key]=dict(path=str(path),sha256=actual)
    geometry=json.loads(args.geometry.read_bytes())
    if args.mean_group_only:
        if args.packet_family is None:parser.error('--mean-group-only requires the exact frozen --packet-family')
        packet_sha=sha(args.packet_family)
        if packet_sha!=geometry['result']['result']['frozen_input_sha256']:
            raise ValueError('Not the frozen selected ORIGINAL narrower LEGAL packet descriptor')
        inputs['packet_family']=dict(path=str(args.packet_family),sha256=packet_sha)
    if args.flat_P_extension_only:
        if args.group_control is None or args.auxiliary is None:
            parser.error('--flat-P-extension-only requires --group-control and --auxiliary')
        for key,digest in (
            ('group_control','fb44aa491671db6ef5fc33b78c83c0680dfa394b58c16045018dadb5d55d2726'),
            ('auxiliary','306a7608bce161a4c83ae94baf5254185de5d17b048fd36f3e17d4ea2bcff1c2')):
            path=getattr(args,key);actual=sha(path)
            if actual!=digest:raise ValueError('Not the frozen original '+key+' receipt')
            inputs[key]=dict(path=str(path),sha256=actual)
        frozen=json.loads(args.group_control.read_bytes())
        for key in expected:
            if frozen['inputs'][key]['sha256']!=inputs[key]['sha256']:
                raise ValueError('Group extension changed its original '+key)
        extension=flat_P_group_extension(frozen,json.loads(args.auxiliary.read_bytes()))
        model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
        number=encoded_reference_number_proof(model,data)
        if number['edge_number_star_incidence_errors']:
            raise AssertionError('Encoded edge/number-star incidence failed')
        for row in number['operator_classes'].values():
            check(row['maximum_exact_Pauli_commutator_l1_error'],'FULL encoded nRef selection')
            if not row['nonzero_even_terms']:raise AssertionError('Empty number conservation class')
        extension['NEW_full_encoded_reference_number_proof']=number
        binding=runtime_binding(args.runtime_archive)
        np.savez_compressed(args.arrays,nRef_class_errors=np.asarray([
            row['maximum_exact_Pauli_commutator_l1_error'] for row in number['operator_classes'].values()]))
        report=dict(plan=frozen['plan'],revision=frozen['revision'],work_items=frozen['work_items'],
            inputs=inputs,flat_P_extension=extension,runtime_binding=binding,
            inherited_native_controls_REUSED_not_rerun=inputs['group_control'],
            arrays_path=str(args.arrays),arrays_sha256=sha(args.arrays),
            program_wall_seconds=time.monotonic()-started,campaign_OPEN=True,
            outcome='ALL8 CAR-P group controls extended with ORIGINAL Delta/2 Q; actual Noether/H/EM/current joined comparison remains required.')
        args.output.write_text(json.dumps(portable(report),indent=2,allow_nan=False))
        print(json.dumps(dict(output=str(args.output),flat_P_extension=extension,
            program_wall_seconds=report['program_wall_seconds']),indent=2))
        return
    binding=runtime_binding(args.runtime_archive)
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    if args.mean_group_only:observed,arrays=group_native(model,data)
    elif args.finite_aligned_only:observed,arrays=aligned_native(model,data)
    else:observed,arrays=native(model,data)
    bridge=json.loads(args.bridge.read_bytes());C2=json.loads(args.c2.read_bytes())
    legal=legal_screening(geometry,bridge,C2)
    # LEGAL parameter factory is explicitly separate from native cache.
    metric=json.loads(args.metric.read_bytes())
    band,frame,calibration=selected_band(metric);law=ImprovedPhiLaw(3,band)
    legal_core=core_from_field(law,frame)
    legal.update(LEGAL_trial_core=dict(Phi_inertia_min=legal_core['Phi_inertia_min'],
        derived_commutator_error=legal_core['SU2_commutator_error'],positive_rest_Casimir_half_error=legal_core['rest_Casimir_half_error']),
        LEGAL_trial_vs_actual_mean_debt=calibration,
        actual_SOFT_band_control=positive_band_group(law,frame,legal_core,
            np.array([2.31e-7,-4.9e-8,8.1e-8]) if args.mean_group_only else np.array([2e-7,8e-8,-4e-8]),
            np.array([-.17,.63,.29]) if args.mean_group_only else np.array([.3,-.2,.4])),
        native_3_used_ONLY_to_supply_SU4_basis_NOT_legal_geometry=True)
    legal.update(actual_all64_Gram_enclosure=metric['certificate']['all_real_shifts_all_N_ge_N_metric_box'],
        actual_frame_distance_upper=float(Fraction(calibration['frame_distance_upper'])),
        distinct_sectors=dict(colour_centre64='Retained original centre seams; continuous screened inverse does not settle them',
            statistical_flat8='Distinct microscopic exchange histories; corrected Delta/2 bounded-unitary auxiliary admission may transfer group observables, not an unbounded G-image gate'),
        remaining_quantitative_contract=dict(
            scalar_SOURCE_seam='Earlier Phi-only/fixed-link section curve is potentially outside original aligned Gamma; it discriminates shortcuts, NOT physical-route impossibility. NEW all64 aligned field and internal packet words telescope exactly; physical Noether/H comparison remains explicit.',
            original_Gamma_initial_BminusB0_squared_log2=C2['science']['initial_mean_body_SOURCE_comparison']['initial_SOURCE_same_Z_error_squared_log2_upper'],
            whole_tube_exit_log2_upper=C2['science']['full_pulse_neutral_and_SOURCE_actual_controls']['WHOLE_actual_tube_exit_probability_log2_upper'],
            needed_finite_group_intertwiner_margin=1e-4-legal['conditional_joined_group_upper'],
            bound_not_supplied='No number is assigned to an unproved fullfield/neutral RIGHT/source-scalar group intertwiner'))
    if args.mean_group_only:
        packet_family=json.loads(args.packet_family.read_bytes())
        legal['bounded_same_neutral_physical_group']=physical_group_ledger(
            legal,geometry,C2,law,frame,calibration,packet_family,metric)
        continuum=continuum_packet_group_control(legal_core,float(packet_family['packet_family']['records'][-1]['width']))
        check(continuum['composition_error'],'NEW continuous nonconstant complex packet group')
        for error in continuum['arbitrary_axis_CORE_LEFT_two_pi_minus_errors']:
            check(error,'allaxes coreLEFT minus after continuous orbital/derivedRIGHT subtraction')
        if continuum['omitted_orbital_pullback_defect']<=.01:
            raise AssertionError('NEW nonconstant fullpacket orbital adverse did not discriminate')
        legal['NEW_continuous_generic_packet_control']=continuum
        legal['conditional_joined_OLD_bridge_plus_NEW_group_upper']=(
            legal['inherited_physical_bridge_upper']+
            legal['bounded_same_neutral_physical_group']['conditional_bounded_same_neutral_module_norm_upper'])
        legal['physical_admission_still_requires_actual_A_T_residual_integrals']=True
    np.savez_compressed(args.arrays,allow_pickle=False,**arrays)
    report=dict(plan='issue-222-mercury-synthesis-continuation-20261007',revision='f9a076e0785f47644962a38b973e6b87a329d6ad594a4345a4728f4c689aa3be',
        work_items=['soft-spatial-closed-rotation','campaign'],inputs=inputs,native=observed,LEGAL=legal,
        runtime_binding=binding,arrays_path=str(args.arrays),arrays_sha256=sha(args.arrays),
        program_wall_seconds=time.monotonic()-started,campaign_OPEN=True,
        outcome='Actual-law mechanical core or NEW covariantly aligned finite field factor exercised; physical group/H intertwiner remains explicit and unproved.')
    text=json.dumps(portable(report),indent=2,allow_nan=False);args.output.write_text(text)
    print(json.dumps(dict(output=str(args.output),native=observed['screening_controls'],LEGAL=legal['conditional_joined_group_upper'],
        campaign_OPEN=True,program_wall_seconds=report['program_wall_seconds']),indent=2))

if __name__=='__main__':main()
