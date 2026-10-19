"""Conditional full-Fock interaction reference on the unchanged ordered packet.

The neutral reference is U_matter(q,A,t) W_field(t). Its SOURCE quasimode
is U_matter(q,A,t) B(q,A) W_field(t) Xi0, NOT W_field(t) B Xi0.
W_field is B's metaplectic operator with its transported compact remainder.
U_matter is an exterior power of ORIGINAL full108 onebody exponentials,
not a new occupied sea. Its counterphase removes only the held neutral
expectation already in W_field. Coulomb/nonquadratic/curved kinetic/auxiliary
and the differential SOURCE commutator remain explicit residual terms.
"""
from __future__ import annotations
from copy import copy
from typing import Any
import math
import numpy as np
from scipy.linalg import eigh, expm_frechet
from common_field_quadratic_reference import gaussian_polynomial_abi, modal_packet
from prepared_guard_colour_chart import onebody_coordinate_jets
from prepared_neutral_full_onebody import full_coordinate_light_operators
from prepared_neutral_slater_fibre import slater_expectation
from prepared_neutral_source_relative_embedding import relative_source
from prepared_guard_colour_chart import Jet2, guarded_frame_jet, centre_word_jet, full_phi_frame_jets
from prepared_neutral_spatial_charge_transfer import GAMMA5
from prepared_neutral_phi_word_channels import coefficient_channels
from prepared_neutral_canonical_source import LABELS
from prepared_neutral_full_field import BETA


def occupied_blocks(h: np.ndarray, occupied: np.ndarray) -> dict[str, Any]:
    """Diagonalize WITHIN the frozen occupied/complement spaces; no recapture."""
    p=occupied@occupied.conj().T
    values, vectors=eigh(np.eye(len(p))-p)
    empty=vectors[:,values>.5]
    eo,vo=eigh(occupied.conj().T@h@occupied)
    ee,ve=eigh(empty.conj().T@h@empty)
    o,e=occupied@vo,empty@ve
    return dict(occupied=o,empty=e,occupied_energy=eo,empty_energy=ee,
                gap=ee[:,None]-eo[None,:],
                invariant_subspace_defect=float(np.linalg.norm(e.conj().T@h@o)),
                projector_preservation=float(np.linalg.norm(o@o.conj().T-p)))


def ph_response(block: dict[str,Any], perturbation: np.ndarray, t: float) -> dict[str,Any]:
    """Actual retained PH action and its gap resolvent, before any elimination."""
    coupling=block['empty'].conj().T@perturbation@block['occupied']
    gap=block['gap']
    nonzero=np.abs(gap)>1e-12
    # Entire divided difference, with its zero-gap limit: no gap clipping.
    response=np.empty_like(coupling)
    response[nonzero]=np.expm1(-1j*t*gap[nonzero])/gap[nonzero]*coupling[nonzero]
    response[~nonzero]=-1j*t*coupling[~nonzero]
    positive=bool(np.min(gap)>0)
    schur=-np.sum(abs(coupling)**2/gap) if positive else None
    return dict(coupling=coupling,gap=gap,first_order_amplitudes=response,
                retained_action_norm=float(np.linalg.norm(coupling)),
                first_order_PH_norm_squared=float(np.vdot(response,response).real),
                Schur_energy= None if schur is None else float(schur.real),
                positive_gap=positive,gap_min=float(np.min(gap)),gap_max=float(np.max(gap)),
                resolvent_dressing_norm_squared=float(np.sum(abs(coupling/gap)**2)) if positive else None,
                actual_retained_PH_before_elimination=True)


def quadratic_frame(h: np.ndarray, dh: np.ndarray, t: float) -> dict[str,np.ndarray]:
    """Exact onebody exponential and differential connection, all Fock powers."""
    u,du=expm_frechet(-1j*t*h,-1j*t*dh)
    connection=-1j*u.conj().T@du
    return dict(U=u,dU=du,connection=connection)


def field_direction(model: Any,sample: dict[str,Any],normal: np.ndarray,
                    dem: np.ndarray) -> dict[str,Any]:
    original=model.original
    jet=model.chart.directional_jet(sample['q'],normal,sample['point'])
    a1,_=original.electromagnetic_field(dem)
    zero_phi=np.zeros_like(sample['point']['phi']);zero_links=np.zeros_like(sample['point']['links'])
    phi=Jet2(sample['point']['phi'],jet['phi_first'],zero_phi)
    links=Jet2(sample['point']['links'],jet['links_first'],zero_links)
    az=np.zeros_like(sample['a'])
    light=onebody_coordinate_jets(model.law,phi,links,Jet2(sample['a'],a1,az))
    heavy=onebody_coordinate_jets(original.heavy_law,phi,links,Jet2(az,az,az))
    return dict(light_first=light.first,heavy_first=-model.Lambda*heavy.first,
                metric_first=jet['metric_first'],log_density_first=jet['log_density_first'],
                full_Jacobian_connection_retained=True,second_jet_NOT_claimed=True)


def transported_fibre(fibre: Any,u: np.ndarray) -> Any:
    moved=copy(fibre)
    moved.orbitals=u@fibre.orbitals
    moved.empty=u@fibre.empty
    moved.occupied={charge:u@orb for charge,orb in fibre.occupied.items()}
    moved.projectors={charge:u@p@u.conj().T for charge,p in fibre.projectors.items()}
    return moved


def determinant_observation(occupied: np.ndarray,u: np.ndarray,h: np.ndarray) -> dict[str,Any]:
    moved=u@occupied;p=moved@moved.conj().T
    sign,logabs=np.linalg.slogdet(occupied.conj().T@moved)
    return dict(exact_Fock_survival_probability=float(np.exp(2*logabs)),
                occupied_subspace_leakage=float(np.trace((np.eye(len(p))-occupied@occupied.conj().T)@p).real),
                energy=float(slater_expectation(p,h).real),
                energy_variance=float(np.linalg.norm((np.eye(len(p))-p)@h@moved)**2),
                overlap_phase=float(np.angle(sign)),full_multipair_exterior_power=True)


def ordered_family(model: Any,data: dict[str,Any],sample: dict[str,Any],profile: str,
                   t: float,normal: np.ndarray,dem: np.ndarray,K: np.ndarray,R: np.ndarray,
                   reference: dict[str,Any],packet: dict[str,Any],y0: np.ndarray,clock: float) -> dict[str,Any]:
    """A function-valued ABI: call at fields, NEVER freeze its point arrays."""
    original=model.original;phi,links=sample['point']['phi'],sample['point']['links']
    hH=-model.Lambda*original.heavy_law.onebody(phi,links,np.zeros_like(sample['a']))
    hL=model.law.onebody(phi,links,sample['a'])
    held_heavy=slater_expectation(original.heavy_projector,hH).real
    held_light=np.array([slater_expectation(model.fibre.projectors[q],hL).real for q in (1,-1)])
    counterphase=np.exp(1j*t*(held_heavy+held_light))
    jets=field_direction(model,sample,normal,dem)
    hf=quadratic_frame(hH,jets['heavy_first'],t);lf=quadratic_frame(hL,jets['light_first'],t)
    moved=transported_fibre(model.fibre,lf['U'])
    original_neutral=sample['neutral']@data['neutral_state']
    original_charged=sample['charged']@data[profile+'_normalized_source_state']
    modal=modal_packet(reference,packet,y0)
    gaussian=gaussian_polynomial_abi(reference,packet,modal,t,clock)
    x=reference['modes'].T@sample['q']-gaussian['modal_gaussian_centre_q']
    polynomial=packet['a']+gaussian['modal_linear_coefficients'].T@x
    exponent=(gaussian['phase']['log_amplitude_prefactor']+
              1j*gaussian['phase']['total_common_phase']+
              1j*gaussian['modal_gaussian_centre_p']@x+
              .5j*np.sum(gaussian['modal_q_precision']*x*x)+
              .5*np.dot(sample['q'],sample['q'])/packet['sigma']**2)
    neutral=counterphase*np.exp(exponent)*polynomial
    _,metadata=coefficient_channels(data,'charged',profile)
    globalZ=metadata['global_SOURCE_normalizer']
    literal=model.source_at(phi,links,sample['theta'],sample['a'],profile)['source_matrix']
    charged=literal@neutral/globalZ
    initial_function_error=float(max(np.linalg.norm(neutral-original_neutral),np.linalg.norm(charged-original_charged))) if t==0 else None
    if t==0 and initial_function_error is not None and initial_function_error>1e-9:
        raise AssertionError('Metaplectic SOURCE quasimode changed the original initial function')
    # These coefficients label U acting on EVERY occupied sea orbital and
    # particle/hole, not compression of exp(h) to the original active rows.
    h0H=-model.Lambda*original.massive_onebody(model.chart.mean_phi,model.chart.mean_links)
    h0L=model.law.onebody(model.chart.mean_phi,model.chart.mean_links,np.zeros_like(sample['a']))
    banks={}
    for charge,index in ((1,0),(-1,1)):
        block=occupied_blocks(h0L,model.fibre.occupied[charge])
        banks[str(charge)]=dict(PH=ph_response(block,hL-h0L,t),
            exact=determinant_observation(model.fibre.occupied[charge],lf['U'],hL),
            frozen_block_defect=block['invariant_subspace_defect'])
    heavyblock=occupied_blocks(h0H,original.heavy_orbitals)
    heavy=dict(PH=ph_response(heavyblock,hH-h0H,t),
               exact=determinant_observation(original.heavy_orbitals,hf['U'],hH),
               frozen_block_defect=heavyblock['invariant_subspace_defect'])
    n2=float(np.vdot(neutral,neutral).real);c2=float(np.vdot(charged,charged).real)
    def excess(charged_operator: np.ndarray,neutral_operator: np.ndarray) -> float:
        # Remove each matching neutral BANK before the ONE charged-state norm.
        # Separately normalized neutral/charged expectations can change branch
        # weights and spuriously leave a common sea contribution.
        numerator=np.vdot(charged,charged_operator@charged)
        if np.linalg.norm(neutral_operator-np.diag(np.diag(neutral_operator)))>1e-10:
            raise ValueError('Neutral-bank subtraction requires the actual block-diagonal operator')
        for index,charge in enumerate((1,-1)):
            amplitudes=charged[moved.charged_slices[charge]]
            numerator-=neutral_operator[index,index]*np.vdot(amplitudes,amplitudes)
        return float((numerator/c2).real)
    def light_excess(op: np.ndarray) -> float:
        return excess(moved.onebody(op,'charged'),moved.onebody(op,'neutral'))
    coulomb_excess=excess(moved.coulomb(original.green,model.ue,'charged'),
                          moved.coulomb(original.green,model.ue,'neutral'))
    density_excess=[]
    for x in range(model.law.volume):
        density=np.zeros((108,108));density[4*x:4*x+4,4*x:4*x+4]=np.eye(4)
        density_excess.append(light_excess(density))
    ref_excess=excess(moved.reference(np.eye(4),'charged'),moved.reference(np.eye(4),'neutral'))
    packets=(model.photon_basis.packets@data['packet_templates'].T).T
    incoming=(model.em['legs']@packets[0]).reshape(model.law.volume,3)/math.sqrt(2)
    outgoing=(model.em['legs']@packets[2].conj()).reshape(model.law.volume,3)/math.sqrt(2)
    emops=full_coordinate_light_operators(model.law,phi,links,sample['a'],incoming,outgoing)
    effectiveK=lf['U'].conj().T@K@lf['U']+lf['connection']
    gimages=[];coreimages=[];refimages=[];jimages=[];cimages=[]
    for word in np.eye(4,dtype=complex):
        physical=relative_source(model,sample,profile,word,effectiveK,R,normal,dem,2.**-19)
        core=relative_source(model,sample,profile,word,effectiveK,np.zeros((4,4)),normal,dem,2.**-19)
        jrow=relative_source(model,sample,profile,word,lf['U'].conj().T@emops['J']@lf['U'],np.zeros((4,4)),np.zeros_like(normal),np.zeros_like(dem),2.**-19)
        crow=relative_source(model,sample,profile,word,lf['U'].conj().T@emops['C']@lf['U'],np.zeros((4,4)),np.zeros_like(normal),np.zeros_like(dem),2.**-19)
        for index in range(2):
            gimages.append(physical['relative_matrix'][:,index]*neutral[index]/globalZ)
            coreimages.append(core['relative_matrix'][:,index]*neutral[index]/globalZ)
            refimages.append((physical['relative_matrix'][:,index]-core['relative_matrix'][:,index])*neutral[index]/globalZ)
            jimages.append(jrow['relative_matrix'][:,index]*neutral[index]/globalZ)
            cimages.append(crow['relative_matrix'][:,index]*neutral[index]/globalZ)
    # A creator on this EM-dependent family is NOT multiplication alone.
    # incoming already includes the canonical 1/sqrt2 derivative factor.
    photonframe=quadratic_frame(hL,emops['J'],t)
    photonbody=lf['U'].conj().T@photonframe['dU']
    photonweak=model.fibre.onebody(photonbody,'charged')
    phaseslope=np.zeros(len(charged),complex)
    for charge in (1,-1):
        phaseslope[model.fibre.charged_slices[charge]]=1j*t*slater_expectation(model.fibre.projectors[charge],emops['J'])
    derivativeweak=photonweak+np.diag(phaseslope)
    multiplication=math.sqrt(2)*np.dot(packets[0],sample['em'][:52])*charged
    creatorprojected=multiplication-derivativeweak@charged
    outside=model.fibre.bilinear_product(photonbody.conj().T,photonbody,'charged')-photonweak.conj().T@photonweak
    creatoroutside=float(np.vdot(charged,outside@charged).real)
    return dict(time=t,U_heavy=hf['U'],U_light=lf['U'],heavy=heavy,light=banks,
        heavy_connection=hf['connection'],light_connection=lf['connection'],
        metric_log_density_connection=jets['log_density_first'],metric_connection_norm=float(np.linalg.norm(jets['metric_first'])),
        neutral_coordinates=neutral,charged_coordinates=charged,neutral_counterphase=counterphase,
        held_expectation_removed_ONCE=True,exact_initial_identity_error=float(max(np.linalg.norm(hf['U']-np.eye(len(hH))),np.linalg.norm(lf['U']-np.eye(len(hL))))) if t==0 else None,
        particle_excess_onebody_energy=light_excess(hL),particle_excess_original_J=light_excess(emops['J']),particle_excess_original_contact=light_excess(emops['C']),
        particle_excess_Coulomb_energy=coulomb_excess,
        particle_excess_matter_plus_Coulomb_energy=light_excess(hL)+coulomb_excess,
        physical_light_density_excess=np.asarray(density_excess),FARRef_number_excess=ref_excess,
        global_EM_charge_excess=float(sum(density_excess)+ref_excess),
        full_field_kinetic_particle_excess_NOT_certified=True,
        excess_definition='Matching transported neutral BANK removed BEFORE the ONE charged norm; not separately normalized unrelated branches. Absolute neutral RIGHT/PH remains in the source-relative action/residual ledger.',
        source_relative_G_images=np.column_stack(gimages),source_relative_J_images=np.column_stack(jimages),source_relative_contact_images=np.column_stack(cimages),
        source_relative_core_G_images=np.column_stack(coreimages),source_relative_FARRef_right_images=np.column_stack(refimages),
        incoming_photon_projected_image=creatorprojected,incoming_photon_derivative_body108=photonbody,
        incoming_photon_outside_PH_norm_squared=creatoroutside,
        incoming_photon_full_norm_squared=float(np.vdot(creatorprojected,creatorprojected).real+creatoroutside),
        incoming_photon_multiplication_only_error=float(np.linalg.norm(derivativeweak@charged)),
        canonical_creator_derivative_retained=True,
        canonical_photon_packets=packets,field_reference=gaussian,
        common_heavy_phase_NOT_removed_from_absolute_flow=True,
        metaplectic_SOURCE_core_evaluated_at_actual_fields=True,
        reference_order='U B(q,A) W_field Xi0; differential [H_field,B] remains in whole-H residual',
        initial_function_embedding_error=initial_function_error,
        common_field_wave_log_amplitude=float(exponent.real),
        original_IC_function_preserved=True,full_H_residual_closed=False)


def mean_mobile_symbol(model: Any,momenta: np.ndarray) -> dict[str,Any]:
    """Evaluate primitive paths, not a guessed mass or new Clifford theorem."""
    phi=model.chart.mean_phi[0];law=model.law
    symbols=[];bands=[]
    for k in momenta:
        h=law.onsite*phi[0]
        for displacement,scalar in law.paths:
            term=np.einsum('m,mij->ij',scalar,phi)*np.exp(1j*displacement@k)
            h=h+term+term.conj().T
        symbols.append(h);bands.append(np.linalg.eigvalsh(h))
    rest=np.min(np.abs(bands[0]))
    return dict(momenta=momenta,symbols=np.asarray(symbols),bands=np.asarray(bands),
                rest_addition_energy=float(rest),actual_Phi0_HS_scale=float(np.sqrt(np.trace(phi[0]@phi[0]).real/4)),
                primitive_mass_coefficient=float(law.band.m),onsite_NOT_rest_mass=True,
                interacting_particle_pole_NOT_claimed=True)


def mean_source_gradient(model: Any,profile: str,em: np.ndarray,
                         word: np.ndarray | None=None) -> dict[str,Any]:
    """Full2430 mean derivative from actual frame jets and literal SOURCE.

    At the uniform locked mean each edge scalar has zero FIRST derivative
    on su4. This is checked on all frame and link tangent generators below.
    It does not remove interior/colour SECOND derivatives away from the mean.
    """
    phi,links=model.chart.mean_phi,model.chart.mean_links
    if max(np.linalg.norm(phi-phi[0]),np.linalg.norm(links-np.eye(4)))>1e-12:
        raise ValueError('Mean derivative simplification requires the actual uniform flat seed')
    if word is None:
        word=BETA
    a,theta=model.original.electromagnetic_field(em)
    seed=model.source_bank.seed_fibre
    literal=seed.source_vectors_at(phi,links,theta,a,model.source_bank.profiles[profile],word)
    plus=literal['full108_plus_coefficients'].reshape(model.law.volume,4,4)
    minus=literal['full108_minus_coefficients'].reshape(4,model.law.volume,4).transpose(1,0,2)
    raw=model.chart.mean_basis.reshape(model.law.volume,105,model.chart.normal_dim)[:,:60]
    zero=np.zeros_like(phi[0]);identity=np.eye(4,dtype=complex)
    generators=[];scalar_derivatives=[]
    for charge in (1,-1):
        mean=phi[0] if charge==1 else -GAMMA5@phi[0]@GAMMA5
        row=[]
        for m in range(4):
            for basis in model.law.basis:
                tangent=np.zeros_like(mean);tangent[m]=basis
                if charge==-1:tangent=-GAMMA5@tangent@GAMMA5
                frame,_=guarded_frame_jet(Jet2(mean,tangent,zero))
                body=frame.first@frame.value.conj().T
                row.append(body)
                scalar_derivatives.append(abs(centre_word_jet(Jet2(identity,body,np.zeros_like(body)),18.).first))
        generators.append(np.asarray(row))
    for generator in model.law.generators:
        scalar_derivatives.append(abs(centre_word_jet(Jet2(identity,1j*generator,np.zeros_like(generator)),18.).first))
    scalar_defect=float(max(scalar_derivatives))
    if scalar_defect>1e-10:
        raise AssertionError('Original per-edge SOURCE has nonzero leading scalar derivative: must retain its interior action')
    plusbody=np.einsum('xan,aij->xnij',raw,generators[0],optimize=True)
    minusbody=np.einsum('xan,aij->xnij',raw,generators[1],optimize=True)
    ref=seed.reference_site
    dp=plusbody@plus[:,None]-plus[:,None]@plusbody[ref][None]
    # Literal minus producer -Gamma5 M^dagger Gamma5 reverses endpoint order.
    left=-GAMMA5@minusbody[ref].conj().transpose(0,2,1)@GAMMA5
    right=GAMMA5@minusbody.conj().transpose(0,1,3,2)@GAMMA5
    dm=left[None]@minus[:,None]+minus[:,None]@right
    return dict(plus_gradient=dp.transpose(1,0,2,3).reshape(model.chart.normal_dim,108,4),
                minus_gradient=dm.transpose(1,2,0,3).reshape(model.chart.normal_dim,4,108),
                plus_mean=plus.reshape(108,4),minus_mean=minus.transpose(1,0,2).reshape(4,108),
                peredge_scalar_first_zero_defect=scalar_defect,
                all_normal_directions=model.chart.normal_dim,
                EM55_HELD_NOT_zeroed=True,interior_colour_second_jets_NOT_removed=True)


def legal_reference_control(science: dict[str,Any],filter_receipt: dict[str,Any]) -> dict[str,Any]:
    """Compare mean-body SOURCE on the ORIGINAL Gamma-supported initial state.

    No global bounded-B assertion: the initial support is the original regular
    union. Later exact charged unitary propagation preserves this INITIAL
    difference. It does not estimate unbounded B tails at later times.
    """
    ledger=filter_receipt['ledger']
    if int(science['selected_log2_Lambda_lower'])!=int(ledger['selected_log2_lambda_lower']):
        raise ValueError('Original LEGAL parameter differs from reference control')
    if not science['actual_original_state']['no_new_ground_no_recenter_no_sigma_retune_no_refilter']:
        raise ValueError('Reference control requires the original postfilter function')
    lower=ledger['source_relative_to_actual_Xi_norm_lower']
    from fractions import Fraction
    norm_lower=Fraction(int(lower['exact_numerator_hex'],16),int(lower['exact_denominator_hex'],16))
    if norm_lower!=Fraction(1,4):
        raise ValueError('This exact normalization payment binds the original norm lower1/4')
    first=int(ledger['global_union']['source_first_log2'])
    d2=int(science['initial_actual_postfilter_controls']['neutral']['actual_postfilter_union_distance_second_moment_log2_upper'])
    error_square=2*first+d2+4
    if error_square>=-40:
        raise AssertionError('Original finite selected Lambda does not pay mean-body initial SOURCE overlap')
    return dict(actual_original_initial_state=science['actual_original_state'],
        selected_log2_Lambda_lower=science['selected_log2_Lambda_lower'],
        original_clock=science['clock'],original_Gamma_support=ledger['physical_preparation'],
        original_support_radius='2^r0 Lambda^(-1/4), all original regular-union centres; not the larger analysis tube',
        original_support_r0_log2=ledger['global_union']['r0_log2'],
        original_SOURCE_normalizer_lower=str(norm_lower),
        actual_initial_mean_body_SOURCE_difference_squared_log2_upper=error_square,
        identity='||(B-B0)Xi/Z_SOURCE||² <=16 M1² <dist_union²>_Xi on ORIGINAL Gamma support; ||U_C(t)(B-B0)Xi/Z_SOURCE|| unchanged exactly',
        B0='branchwise actual mean Phi/colour body with SAME EM55/harmonic/sourceprofile/quartet/FARRef; no physical A set to zero and no B0 renormalization',
        source_first_operator_graph_log2_upper=first,
        initial_union_distance_squared_log2_upper=d2,
        actual_whole_tube=science['full_pulse_neutral_and_SOURCE_actual_controls'],
        actual_true_photon_unweighted_tube=science['full_pulse_true_photon_unweighted_actual_controls'],
        actual_source_field_join=science['source_field_join'],
        reference_is_comparison_NOT_substituted_initial_function=True,
        uniform_unbounded_SOURCE_outside_tail_NOT_paid=True,
        full_H_reference_difference_and_physical_group_NOT_certified=True)




def covariant_packet_marks(model: Any,sample: dict[str,Any],plus_packet: np.ndarray,
                           minus_packet: np.ndarray) -> dict[str,np.ndarray]:
    """Full coframe packet/Ref-ancilla module using the ORIGINAL identity SOURCE.

    f(x) is a4x4 packet matrix (light spinor rows, Ref ancilla columns), possibly
    a field function. All scalar screened words/EM phases are reused from the
    literal producer; no new Wilson/charge/source law is copied here.
    """
    volume=model.law.volume
    if plus_packet.shape!=(volume,4,4) or minus_packet.shape!=(volume,4,4):
        raise ValueError('Full coframe packet and Ref4 ancilla required at every site')
    phi,links=sample['point']['phi'],sample['point']['links']
    identity=model.source_bank.seed_fibre.source_vectors_at(phi,links,sample['theta'],sample['a'],
        np.ones(volume,dtype=complex),np.array([1.,0.,0.,0.],complex))
    plus0=identity['full108_plus_coefficients'].reshape(volume,4,4)
    minus0=identity['full108_minus_coefficients'].reshape(4,volume,4).transpose(1,0,2)
    zeros=np.zeros_like(phi)
    frames,_=full_phi_frame_jets(Jet2(phi,zeros,zeros))
    negative=-GAMMA5@phi@GAMMA5
    dualframes,_=full_phi_frame_jets(Jet2(negative,zeros,zeros))
    born=frames.value@plus_packet@frames.value.conj().transpose(0,2,1)
    dualborn=GAMMA5@dualframes.value@minus_packet@dualframes.value.conj().transpose(0,2,1)@GAMMA5
    plus=born@plus0;minus=minus0@dualborn
    return dict(plus=plus.reshape(108,4),minus=minus.transpose(1,0,2).reshape(4,108))


def original_packet_input(model: Any,profile: str) -> tuple[np.ndarray,np.ndarray]:
    """Original quartet/profile as ONE module input, unchanged and unnormalized."""
    raw=2*np.einsum('a,aij->ij',BETA,LABELS)
    packet=model.source_bank.profiles[profile][:,None,None]*raw
    return packet,packet.conj().transpose(0,2,1)
