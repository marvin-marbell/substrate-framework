"""NEW actual functional photon/global coexact metric and original LAB consumer.
No original supplier/H/G science rerun; a new physical function bank is exercised.
Normal/theta quadrature is explicit finite debt, full52 moments are analytic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast
import numpy as np
from scipy.linalg import eigh
from original_charged_principal import portable
from original_total_angular_transfer import scalar_orbital
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_shared_photon_state import load_shared_native_cache
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_source_relative_embedding import relative_source
from prepared_neutral_phi_word_channels import coefficient_channels
from prepared_neutral_full_onebody import full_coordinate_light_operators
from prepared_neutral_quartet_functions import full_slater_relative_spin
from prepared_neutral_full_field import BETA
from prepared_neutral_weyl_hard_fock import GaussianFunction, ProjectionPolynomial, gaussian_kernel
from functional_source_photon_bank import (FunctionalSourcePhotonBank,coexact_metric,
    original_LAB_effect_moments,LAB_scalar_plane_kernels,creator_function,
    Full108ReferenceMarks,FullPacketWeylTerm,full_packet_mark_action)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_bank(model: Any,data: dict[str,Any],bank: FunctionalSourcePhotonBank,sample: dict[str,Any],profile: str) -> np.ndarray:
    """Existing ORIGINAL microscopic callbacks, not reconstructed Fourier values."""
    C,metadata=coefficient_channels(data,'charged',profile);base=sample['charged']@C
    psi=sample['neutral']@data['neutral_state'];Z=metadata['global_SOURCE_normalizer'];point=sample['point']
    pull=physical_chart_pullback(model,sample,bank.alpha,bank.velocity);gauge=model.gauge_operators(pull['gauge'],'charged');em=model.noether_bank.evaluate(sample['em'][:52])
    K=gauge['body']+scalar_orbital(model.law,point['links'],sample['a'],bank.velocity)+np.diag(np.repeat(em['lambda_C'],4));R=gauge['reference'];dem=np.r_[em['flow'],em['harmonic']]
    gs=[]
    for word in range(4):
        g_image=relative_source(model,sample,profile,np.eye(4)[word],K,R,pull['normal'],dem,bank.source_step)
        for branch in range(2):gs.append(g_image['relative_matrix'][:,branch]*psi[branch]/Z)
    legs=[(model.em['legs']@packet).reshape(model.law.volume,3)/math.sqrt(2) for packet in (bank.packets[0],bank.packets[1],bank.packets[2].conj())]
    primitives=[full_coordinate_light_operators(model.law,point['phi'],point['links'],sample['a'],leg)['J'] for leg in legs]
    primitives.append(full_coordinate_light_operators(model.law,point['phi'],point['links'],sample['a'],legs[0],legs[2])['C'])
    js=[];pairs=[(charge,word) for charge in (1,-1) for word in range(4)]
    for primitive in primitives:
        def row(beta: np.ndarray) -> np.ndarray:
            src=model.source_at(point['phi'],point['links'],sample['theta'],sample['a'],profile)
            if not np.array_equal(beta,BETA):src=model.source_bank.physical_source(model.fibre,model.reference_frame,point['phi'],point['links'],sample['theta'],sample['a'],profile,beta)
            return full_slater_relative_spin(model.fibre,src,primitive)['relative']
        coherent=row(np.asarray(BETA));unit=[row(np.eye(4)[word]) for word in range(4)]
        columns=[coherent@psi/Z]
        for charge,word in pairs[1:]:columns.append(unit[word]@(psi*np.array([charge==1,charge==-1]))/Z)
        js.append(np.column_stack(columns))
    return np.column_stack((base,np.column_stack(gs),*js))


def projection(metric: np.ndarray,base_count: int,input_index: int) -> dict[str,Any]:
    old=metric[:base_count,:base_count];diag=np.real(np.diag(old));scale=np.sqrt(diag)
    conditioned=old/scale[:,None]/scale[None,:];spectrum=eigh(conditioned,eigvals_only=True)
    norm=float(metric[input_index,input_index].real);cross=metric[:base_count,input_index]/scale
    projected=None
    if spectrum[0]>0:projected=float(np.vdot(cross,np.linalg.solve(conditioned,cross)).real)
    return dict(actual_input_norm_squared=norm,conditioned_old_metric_eigenvalues=spectrum,
        physical_projection_squared=projected,physical_outside_squared=None if projected is None else norm-projected,
        projection_fraction=None if projected is None else projected/norm,
        coordinate_diagonal_equilibration_NOT_state_SOURCEZ_or_effect_renormalization=True,
        no_metric_jitter_clipping_or_photon_fallback=True)


def generic_packet_science(model: Any,data: dict[str,Any],bank: FunctionalSourcePhotonBank) -> dict[str,Any]:
    """Changed full packet, actual Hmean/J/contact/orbital, true RIGHT/PH."""
    q,em,_=source_point(model,data);sample=model.basis_at(q,em);point=sample['point']
    pull=physical_chart_pullback(model,sample,bank.alpha,bank.velocity)
    gauge=model.gauge_operators(pull['gauge'],'charged');noe=model.noether_bank.evaluate(em[:52])
    body=gauge['body']+scalar_orbital(model.law,point['links'],sample['a'],bank.velocity)+np.diag(np.repeat(noe['lambda_C'],4))
    reference=gauge['reference'];dem=np.r_[noe['flow'],noe['harmonic']]
    leg=(model.em['legs']@bank.packet).reshape(model.law.volume,3)/math.sqrt(2)
    operators=full_coordinate_light_operators(model.law,point['phi'],point['links'],sample['a'],leg,leg.conj())
    current,contact=operators['J'],operators['C'];h=model.law.onebody(point['phi'],point['links'],sample['a'])
    output={}
    for profile in ('source19','SOFT'):
        source=model.source_at(point['phi'],point['links'],sample['theta'],sample['a'],profile)
        initial=Full108ReferenceMarks(source['full108_plus_coefficients'],source['full108_minus_coefficients'])
        changed=initial.light_action(h)
        def at(q: np.ndarray,y: np.ndarray) -> Full108ReferenceMarks:
            p=model.chart.embedding(q);a,theta=model.electromagnetic_field(y)
            b=model.source_at(p['phi'],p['links'],theta,a,profile)
            marks=Full108ReferenceMarks(b['full108_plus_coefficients'],b['full108_minus_coefficients'])
            return marks.light_action(model.law.onebody(p['phi'],p['links'],a))
        step=2.**-19/max(1.,float(np.linalg.norm(pull['normal'])),float(np.linalg.norm(dem)))
        plus=at(q+step*pull['normal'],em+step*dem);minus=at(q-step*pull['normal'],em-step*dem)
        derivative=Full108ReferenceMarks((plus.plus-minus.plus)/(2*step),(plus.minus-minus.minus)/(2*step))
        result=full_packet_mark_action(model.fibre,changed.plus,changed.minus,derivative.plus,derivative.minus,body,reference)
        controls={}
        for name,matrix in (('Hmean',bank.fibre.h0),('original_J',current),('original_contact',contact),('actual_orbital_L',scalar_orbital(model.law,point['links'],sample['a'],bank.velocity))):
            raw=changed.light_action(matrix).physical_source(model.fibre)
            control=full_slater_relative_spin(model.fibre,changed.physical_source(model.fibre),matrix)
            controls[name]=dict(full108_packet_action_norm=float(np.linalg.norm(raw['source_matrix'])),
                original_full_Fock_CAR_error=control['exact_CAR_error'],
                true_neutral_RIGHT_outside_norm=control['neutral_right_outside_norm'],
                beforecompression_PH_outside_norm_squared=control['outside_PH_spin_norm_squared'],
                orbital_packet_action_NOT_intrinsic_spin=True)
        C,metadata=coefficient_channels(data,'charged',profile);psi=sample['neutral']@data['neutral_state'];z=metadata['global_SOURCE_normalizer']
        old=sample['charged']@C;u,s,_=np.linalg.svd(old,full_matrices=False);frame=u[:,s>np.finfo(float).eps*max(old.shape)*s[0]]
        full=changed.physical_source(model.fibre)['source_matrix']@psi/z
        outside=full-frame@(frame.conj().T@full)
        # D_g H = sqrt2 J_g; original SOURCE commutes coexact. Do NOT fit Hf.
        d=initial.light_action(math.sqrt(2)*current).physical_source(model.fibre)['source_matrix']@psi/z
        actual=math.sqrt(2)*(bank.packet@em[:52])*full-d/math.sqrt(2)
        functions=[];hopping=np.zeros_like(h)
        for x,y,block,path in model.law.terms(point['phi'],point['links'],sample['a']):
            frequency=sum((model.em['legs'][3*site+axis] for site,axis in path),np.zeros(52))
            for xx,yy,b,k in ((x,y,block,frequency),(y,x,block.conj().T,-frequency)):
                matrix=np.zeros_like(h);matrix[4*xx:4*xx+4,4*yy:4*yy+4]=b
                # Remove the actual point phase to expose the coefficient at y=0.
                matrix*=np.exp(-1j*(k@em[:52]));hopping+=np.exp(1j*(k@em[:52]))*matrix
                functions.append(FullPacketWeylTerm(GaussianFunction(ProjectionPolynomial.constant(52),k),initial.light_action(matrix)))
        functions.append(FullPacketWeylTerm(GaussianFunction(ProjectionPolynomial.constant(52),np.zeros(52)),initial.light_action(h-hopping)))
        computed=np.zeros_like(full)
        for term in functions:
            created=term.incoming_creator(bank.packet);p,_=created.function.polynomial.evaluate(em[:52])
            computed+=p*np.exp(1j*(created.function.phase@em[:52]))*(created.marks.physical_source(model.fibre)['source_matrix']@psi/z)
        error=float(np.linalg.norm(computed-actual))
        if error>1e-9:raise ValueError('Full generic H-packet canonical creator lost DgH/current')
        output[profile]=dict(changed_full108xRef_packet_Hf_norm=float(np.linalg.norm(full)),
            changed_packet_outside_initial8_squared=float(np.vdot(outside,outside).real),
            full_generic_creator_vs_ORIGINAL_DgH_sqrt2J_error=error,
            omitted_current_derivative_error=float(np.linalg.norm(d/math.sqrt(2))),
            actual_generic_relative_Noether_matrix_norm=float(np.linalg.norm(result['relative_matrix'])),
            true_generic_neutral_RIGHT_PH_norm=float(np.linalg.norm(result['actual_neutral_RIGHT_PH'])),
            generic_Noether_CAR_control=result['original_full_Fock_CAR_and_beforecompression_PH_controls'],
            actual_Hmean_J_contact_orbital_controls=controls,
            one_original_SOURCE_initial_globalZ_only=True,no_profilefit_recapture_or_8ray_filter=True,
            no_spinor_assignment_or_physical_Hmean_static_currentdrive_claim=True)
    return output


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','manifest','frozen-LAB-effect','output-npz'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--normal-theta-nodes',type=int,default=4)
    p.add_argument('--packet-actions-only',action='store_true');args=p.parse_args()
    if args.normal_theta_nodes<2 or args.normal_theta_nodes%2:raise ValueError('Antithetic fullnormal/theta Gaussian nodes required')
    paths={name:getattr(args,name.replace('-','_')) for name in ('cache','manifest','frozen-LAB-effect')};before={name:digest(path) for name,path in paths.items()}
    if before['frozen-LAB-effect']!='3182b40a78e7eb6cfd26d4803fc9f57b67b1ea78d2a44757f773217fbf8b5b84':raise ValueError('Original frozen F2/E2 bytes changed')
    model,data=load_shared_native_cache(args.cache,args.manifest);bank=FunctionalSourcePhotonBank(model,data)
    with np.load(args.frozen_LAB_effect,allow_pickle=False) as stored:f2=stored['frozen_E2_coefficients'].copy()
    q0,em0,_=source_point(model,data);sample=model.basis_at(q0,em0);validation={}
    if args.packet_actions_only:
        print(json.dumps(portable(dict(input_sha256=before,generic_fullpacket=generic_packet_science(model,data,bank),
            photon_operator_same_original_J_current=True,
            original_driven_H_minus_OmegaG_NOT_retroconjugated_or_replaced=True,
            fullfield_PH_weighted_domain_propagation_and_spin_mass_Born_closure_NOT_claimed=True)),sort_keys=True))
        return
    for profile in ('source19','SOFT'):
        pack=bank.compile(sample,profile);value=pack.values(em0[:52]);direct=direct_bank(model,data,bank,sample,profile)
        direct_error=float(np.linalg.norm(value[:,:48]-direct));creator=[];g=bank.packet
        for step in (2.**-9,2.**-10):
            derivative=np.zeros_like(direct);gaussian=math.exp(-float(em0[:52]@em0[:52])/2)
            for direction,factor in ((g.real,1),(g.imag,1j)):
                plus,minus=em0[:52]+step*direction,em0[:52]-step*direction
                derivative+=factor*(pack.values(plus)[:,:48]*math.exp(-float(plus@plus)/2)-pack.values(minus)[:,:48]*math.exp(-float(minus@minus)/2))/(2*step)
            actual=((g@em0[:52])*direct-derivative/gaussian)/math.sqrt(2)
            creator.append(dict(step=step,full_gaussian_creator_absolute_error=float(np.linalg.norm(actual-value[:,48:])),
                full_gaussian_creator_relative_error=float(np.linalg.norm(actual-value[:,48:])/np.linalg.norm(value[:,48:]))))
        wrong=math.sqrt(2)*(g@em0[:52])*direct
        validation[profile]=dict(functional_fullG_J_contact_vs_ORIGINAL_callbacks_error=direct_error,
            full_creator_errors=creator,wrong_multiplication_only_creator_error=float(np.linalg.norm(wrong-value[:,48:])),
            actual_factor_derivative_norm=float(np.linalg.norm(pack.coexact_directional(em0[:52],g))),
            actual_48_functional_NOT_held_pointcolumns=True,
            no_copied_spinor_RefPauli_or_particle_spin_assignment=True)
        if direct_error>2e-8:raise ValueError('Functional actual G/J/contact producer mismatch')
    probe=bank.compile(sample,'source19');phases=np.asarray([term.function.phase for term in probe.terms[:len(probe.terms)//2]])
    plane,creator=LAB_scalar_plane_kernels(model,phases,bank.packet);functions=model.noether_bank.gaussian_functions();checks=[]
    for column in (0,len(phases)//3,len(phases)-1):
        phase=phases[column];base=GaussianFunction(ProjectionPolynomial.constant(52),phase);raised=creator_function(base,bank.packet)
        for row in (0,1,5,15,len(functions)-1):
            left=functions[row]
            a=sum(gaussian_kernel(term,base) for term in left.terms);b=sum(gaussian_kernel(term,raised) for term in left.terms)
            checks.append(dict(scalar=row,term=column,plane_error=abs(a-plane[row,column]),creator_error=abs(b-creator[row,column])))
    if max(max(row['plane_error'],row['creator_error']) for row in checks)>1e-8:raise ValueError('Full52 original finite Gaussian scalar ABI mismatch')
    rng=np.random.default_rng(2222030);qnodes=rng.standard_normal((args.normal_theta_nodes//2,model.chart.normal_dim))*model.sigma/math.sqrt(2);theta=rng.standard_normal((args.normal_theta_nodes//2,3))/math.sqrt(2)
    global_metric={profile:np.zeros((96,96),complex) for profile in ('source19','SOFT')};global_row={profile:np.zeros(96,complex) for profile in global_metric};effectnorm=0.;records=[]
    for pair,(q,t) in enumerate(zip(qnodes,theta)):
        for sign in (-1,1):
            em=np.r_[np.zeros(52),sign*t];node=model.basis_at(sign*q,em);weight=node['importance_weight']/args.normal_theta_nodes;norm_this=[]
            for profile in global_metric:
                pack=bank.compile(node,profile);metric=coexact_metric(pack);lab=original_LAB_effect_moments(model,data,node,pack,f2)
                global_metric[profile]+=weight*metric;global_row[profile]+=weight*lab['LAB_cross_row'];norm_this.append(lab['coexact_exact_original_LAB_norm_squared'])
                if lab['original_LAB_function_reconstruction_error']>1e-11:raise ValueError('Original fixed LABY was advected/reset/refit')
            effectnorm+=weight*norm_this[0]
            records.append(dict(pair=pair,sign=sign,qnorm=float(np.linalg.norm(node['q'])),theta=node['em'][52:],cutoff=node['cutoff'],same_LAB_norm_profile_difference=abs(norm_this[0]-norm_this[1])))
    results={};payload={'normal_qnodes':qnodes,'harmonic_theta_nodes':theta,'neutral_exact_uncut_Gram28':np.eye(28),'neutral_original_incoming_IC':bank.initial_coefficients('neutral')}
    for profile,metric in global_metric.items():
        n=48;proj=projection(metric,n,n);initial=bank.initial_coefficients(profile);cross=global_row[profile]
        response=float(abs(cross@initial)**2/float(np.vdot(initial,metric@initial).real))
        results[profile]=dict(full52_exact_Phi_theta_quad_metric_projection=proj,
            original_SOURCE_plus_incoming_function_norm_squared=float(metric[48,48].real),
            original_SOURCE_no_photon_function_norm_squared=float(metric[0,0].real),
            original_LAB_unretuned_initial_response_quotient=response,
            NOT_Born_or_evolution=True,global_metric_antihermitian_norm=float(np.linalg.norm(metric-metric.conj().T)))
        payload[profile+'_global_Gram96']=metric;payload[profile+'_fixed_original_LAB_cross_row']=cross;payload[profile+'_original_incoming_IC']=initial
    if any(digest(path)!=before[name] for name,path in paths.items()):raise RuntimeError('Original law/state/detector input changed')
    np.savez_compressed(args.output_npz,**cast(Any,payload))
    print(json.dumps(portable(dict(input_sha256=before,validation=validation,scalar_full52_kernel_checks=checks,
        normal_theta_nodes=records,global_results=results,original_LAB_global_norm_squared_estimate=effectnorm,
        original_E0_contractivity_NOT_assumed_or_receiver_renormalized=True,
        actual_neutral_original_plus_incoming_norm_squared=float(np.vdot(bank.initial_coefficients('neutral'),bank.initial_coefficients('neutral')).real),
        factor_jets_not_Gaussian_Haar_cutoff_preapplied=True,all52_exact_all55_harmonic_retained=True,
        normal_theta_global_quadrature_debt=True,source_Phi_geometry_FD_debt=True,
        fullH_M_connection_reference_AND_photon_propagation_next_owner_not_claimed=True,
        no_supplier_H_G_CLI_or_preparation_rerun=True,output_npz_sha256=digest(args.output_npz),
        no_mass_spin_statistics_Born_Root_LEGAL_or_whole_closure_claim=True)),sort_keys=True))


if __name__=='__main__':main()
