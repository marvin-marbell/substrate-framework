#!/usr/bin/env python3
"""Evaluate the coupled conditional Fock/metaplectic reference; not a pole claim."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from common_field_covariance_consumer import consume
from common_field_quadratic_reference import fixed_seed_potential, seed_packet
from native_closed_rotation_quantum import physical_chart_pullback
from original_total_angular_transfer import scalar_orbital
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_shared_photon_state import load_shared_native_cache
from ordered_sector_reference import (ordered_family, mean_mobile_symbol, field_direction,
    occupied_blocks, ph_response, quadratic_frame, determinant_observation, mean_source_gradient,
    legal_reference_control, covariant_packet_marks, original_packet_input)
from raw_source_relative_noether_graph import runtime_binding
from prepared_neutral_phi_word_channels import coefficient_channels
from functional_source_photon_bank import full_packet_mark_action
from prepared_neutral_source_relative_embedding import relative_source
from prepared_neutral_full_onebody import full_coordinate_light_operators
from prepared_neutral_full_field import BETA


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value: Any) -> Any:
    if isinstance(value,np.ndarray):
        if np.iscomplexobj(value):return dict(real=value.real.tolist(),imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
    if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [portable(v) for v in value]
    return value




def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-npz','cache-manifest','covariance-npz','covariance-receipt','legal-control','original-filter','output-npz','output-json','runtime-archive'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--duration',type=float,default=1e-5)
    args=p.parse_args();started=time.perf_counter()
    if sha(args.cache_npz)!='10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3':raise ValueError('Wrong frozen IC')
    if sha(args.cache_manifest)!='90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a':raise ValueError('Wrong frozen manifest')
    binding=runtime_binding(args.runtime_archive)
    legal_receipt=json.loads(args.legal_control.read_bytes())
    if sha(args.original_filter)!='9fdf6a3f5f38a7150f50489ce6e9fa413844fd0da6ddb43b9a4a6c741badedc3':
        raise ValueError('Not the original actual postfilter function')
    if legal_receipt['inputs_sha256']['filter']!=sha(args.original_filter):
        raise ValueError('LEGAL reference receipt does not bind the original filter')
    if not legal_receipt['runtime_binding']['all_actual_callable_code_matches_archived_source']:
        raise ValueError('LEGAL control has no actual callable/source binding')
    legal=legal_reference_control(legal_receipt['science'],json.loads(args.original_filter.read_bytes()))
    covariance_observation=consume(args.covariance_npz,args.covariance_receipt,args.duration)
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    q,em,g=source_point(model,data);sample=model.basis_at(q,em)
    guard=json.loads(g['guard_path'].read_bytes())
    pull=physical_chart_pullback(model,sample,np.asarray(guard['alpha']),np.asarray(guard['velocity']))
    noether=model.noether_bank.evaluate(em[:52]);gauge=model.gauge_operators(pull['gauge'],'charged')
    orbital=scalar_orbital(model.law,sample['point']['links'],sample['a'],np.asarray(guard['velocity']))
    K=gauge['body']+orbital+np.diag(np.repeat(noether['lambda_C'],4))
    R=gauge['reference'];dem=np.r_[noether['flow'],noether['harmonic']]
    with np.load(args.covariance_npz,allow_pickle=False) as stored:
        reference={name:stored[name].copy() for name in ('eigenvalues','modes','gradient')}
        y0=stored['Y0'].copy();scalars=stored['scalars'].copy()
    reference.update(chi=float(scalars[0]),E0=float(scalars[4]))
    packet=seed_packet(model,data);clock=float(scalars[5])
    frozen_scalar=sum(fixed_seed_potential(model,packet,q).values())
    scalar_reference=reference['E0']+reference['gradient']@q+.5*np.sum(reference['eigenvalues']*(reference['modes'].T@q)**2)
    # PH action of all six original field-polynomial directions, with every
    # occupied/empty heavy108 and both complete light banks retained.
    derivative_rows=[];derivative_arrays={}
    for index,direction in enumerate(model.original.directions.T):
        jet=field_direction(model,sample,direction,np.zeros_like(em))
        PHH=(np.eye(108)-model.original.heavy_projector)@jet['heavy_first']@model.original.heavy_projector
        norms={str(charge):float(np.linalg.norm((np.eye(108)-projector)@jet['light_first']@projector)) for charge,projector in model.fibre.projectors.items()}
        derivative_rows.append(dict(direction=index,heavy_PH_force_norm=float(np.linalg.norm(PHH)),light_PH_force_norm=norms,
                                    log_density_first=jet['log_density_first'],metric_first_norm=float(np.linalg.norm(jet['metric_first']))))
        derivative_arrays['heavy_PH_force_'+str(index)]=PHH
    arrays={'actual_q':q,'actual_em55':em,'normal_Noether_flow':pull['normal'],'EM_Noether_flow55':dem,'K108':K,'R4':R}
    arrays.update(derivative_arrays)
    meanpoint=model.chart.evaluate(np.zeros(model.chart.normal_dim))
    arrays['mean_cometric_normal_gauge']=meanpoint['normal_gauge_cometric']
    arrays['mean_metric_normal_gauge']=meanpoint['normal_gauge_metric']
    arrays['mean_cometric_normal_normal']=meanpoint['cometric'][:model.chart.normal_dim,:model.chart.normal_dim]
    mean_geometry=dict(full_dimension=model.chart.raw_dim,normal_dimension=model.chart.normal_dim,
        normal_cometric_identity_defect=float(np.linalg.norm(arrays['mean_cometric_normal_normal']-np.eye(model.chart.normal_dim))),
        normal_gauge_cometric_norm=float(np.linalg.norm(arrays['mean_cometric_normal_gauge'])),
        normal_gauge_metric_norm=float(np.linalg.norm(arrays['mean_metric_normal_gauge'])),
        nonzero_SOURCE_gradient_NOT_excluded_by_zero_metric_cross=True)
    mean_source_controls=[]
    rngmean=np.random.default_rng(2221010)
    testdirection=rngmean.normal(size=model.chart.normal_dim);testdirection/=np.linalg.norm(testdirection)
    hmean=2.**-19
    for profile in ('source19','SOFT'):
        jet=mean_source_gradient(model,profile,em)
        _,channel_metadata=coefficient_channels(data,'charged',profile)
        globalZ=channel_metadata['global_SOURCE_normalizer']
        raw_norms={name:float(np.linalg.norm(jet[name])) for name in ('plus_gradient','minus_gradient')}
        points=[model.chart.embedding(sign*hmean*testdirection) for sign in (-1,1)]
        a,theta=model.original.electromagnetic_field(em)
        values=[model.source_bank.seed_fibre.source_vectors_at(point['phi'],point['links'],theta,a,model.source_bank.profiles[profile]) for point in points]
        errors={}
        for mark,name in (('full108_plus_coefficients','plus_gradient'),('full108_minus_coefficients','minus_gradient')):
            fd=(values[1][mark]-values[0][mark])/(2*hmean)
            prediction=np.einsum('a,aij->ij',testdirection,jet[name])
            errors[name]=float(np.linalg.norm(fd-prediction))
            if errors[name]>1e-6:raise AssertionError('Full mean SOURCE derivative disagrees with literal producer: '+name)
            arrays[profile+'_mean_'+name]=jet.pop(name)
        arrays[profile+'_mean_plus_marks']=jet.pop('plus_mean')
        arrays[profile+'_mean_minus_marks']=jet.pop('minus_mean')
        mean_source_controls.append(dict(profile=profile,full2430_gradient_controls=errors,
            raw_mark_gradient_norms=raw_norms,ORIGINAL_global_SOURCE_normalizer=globalZ,
            mark_gradient_over_original_globalZ_norms={name:value/globalZ for name,value in raw_norms.items()},
            actual_neutral_preimage_required_for_state_graph=True,**jet))
    packet_controls=[]
    packets=(model.photon_basis.packets@data['packet_templates'].T).T
    incoming=(model.em['legs']@packets[0]).reshape(model.law.volume,3)/np.sqrt(2)
    outgoing=(model.em['legs']@packets[2].conj()).reshape(model.law.volume,3)/np.sqrt(2)
    operators=full_coordinate_light_operators(model.law,sample['point']['phi'],sample['point']['links'],sample['a'],incoming,outgoing)
    mean_light=model.law.onebody(model.chart.mean_phi,model.chart.mean_links,sample['a'])
    for profile in ('source19','SOFT'):
        ppacket,mpacket=original_packet_input(model,profile)
        marks=covariant_packet_marks(model,sample,ppacket,mpacket)
        source=model.source_at(sample['point']['phi'],sample['point']['links'],sample['theta'],sample['a'],profile)
        errors=dict(plus=float(np.linalg.norm(marks['plus']-source['full108_plus_coefficients'])),
                    minus=float(np.linalg.norm(marks['minus']-source['full108_minus_coefficients'])))
        if max(errors.values())>1e-9:raise AssertionError('Generic packet module changed original SOURCE input')
        physical=relative_source(model,sample,profile,BETA,K,R,pull['normal'],dem,2.**-19)
        rows={}
        for name,body,ref,dplus,dminus in (
                ('H_light',operators['H'],np.zeros((4,4)),np.zeros_like(marks['plus']),np.zeros_like(marks['minus'])),
                ('H_mean_light_full_EM',mean_light,np.zeros((4,4)),np.zeros_like(marks['plus']),np.zeros_like(marks['minus'])),
                ('L_packet',orbital,np.zeros((4,4)),np.zeros_like(marks['plus']),np.zeros_like(marks['minus'])),
                ('G_physical',K,R,physical['derivative_marks']['plus'],physical['derivative_marks']['minus']),
                ('J_original',operators['J'],np.zeros((4,4)),np.zeros_like(marks['plus']),np.zeros_like(marks['minus'])),
                ('contact_original',operators['C'],np.zeros((4,4)),np.zeros_like(marks['plus']),np.zeros_like(marks['minus']))):
            action=full_packet_mark_action(model.fibre,marks['plus'],marks['minus'],dplus,dminus,body,ref)
            arrays[profile+'_generic_'+name+'_relative_SOURCE_matrix']=action['relative_matrix']
            arrays[profile+'_generic_'+name+'_actual_neutral_RIGHT_PH']=action['actual_neutral_RIGHT_PH']
            arrays[profile+'_generic_'+name+'_plus_packet_action']=action['light_packet_action'].plus
            arrays[profile+'_generic_'+name+'_minus_packet_action']=action['light_packet_action'].minus
            rows[name]=action['original_full_Fock_CAR_and_beforecompression_PH_controls']
        packet_controls.append(dict(profile=profile,initial_input_mark_errors=errors,
            full_mark_complex_domain_per_charge=108*4,original_input_NOT_retuned=True,
            full_mean_light_orbital_current_and_Ref_operators_act_on_packet=True,
            outside_fixed8_rays_is_NOT_intrinsic_spin_failure=True,action_controls=rows))
    # A source-point slice is not the full2430 packet. Also consume the literal
    # first original RNG222 quadrature node, no recenter/filter/state change.
    rng=np.random.default_rng(222)
    original_qnodes=rng.standard_normal((7,2430))*model.sigma/np.sqrt(2)
    original_enodes=rng.standard_normal((7,55))/np.sqrt(2)
    cloudq,cloudem=-original_qnodes[0],-original_enodes[0]
    cloud=model.chart.embedding(cloudq);clouda,_=model.original.electromagnetic_field(cloudem)
    meanH=-model.Lambda*model.original.massive_onebody(model.chart.mean_phi,model.chart.mean_links)
    cloudH=-model.Lambda*model.original.massive_onebody(cloud['phi'],cloud['links'])
    cloudblock=occupied_blocks(meanH,model.original.heavy_orbitals)
    cloudPH=ph_response(cloudblock,cloudH-meanH,args.duration)
    cloudU=quadratic_frame(cloudH,np.zeros_like(cloudH),args.duration)['U']
    cloudexact=determinant_observation(model.original.heavy_orbitals,cloudU,cloudH)
    for name in ('coupling','gap','first_order_amplitudes'):
        arrays['original_cloud_heavy_'+name]=cloudPH.pop(name)
    arrays['original_cloud_q']=cloudq;arrays['original_cloud_em55']=cloudem
    meanL=model.law.onebody(model.chart.mean_phi,model.chart.mean_links,np.zeros_like(clouda))
    cloudL=model.law.onebody(cloud['phi'],cloud['links'],clouda)
    cloudUL=quadratic_frame(cloudL,np.zeros_like(cloudL),args.duration)['U']
    cloudlight={}
    for charge in (1,-1):
        bankPH=ph_response(occupied_blocks(meanL,model.fibre.occupied[charge]),cloudL-meanL,args.duration)
        for name in ('coupling','gap','first_order_amplitudes'):
            arrays['original_cloud_light'+str(charge)+'_'+name]=bankPH.pop(name)
        cloudlight[str(charge)]=dict(PH=bankPH,exact=determinant_observation(model.fibre.occupied[charge],cloudUL,cloudL))
    cloudscalar=sum(fixed_seed_potential(model,packet,cloudq).values())
    cloudref=reference['E0']+reference['gradient']@cloudq+.5*np.sum(reference['eigenvalues']*(reference['modes'].T@cloudq)**2)
    cloud_observation=dict(original_rng=222,original_node=0,original_sign=-1,
        full2430_point_norm=float(np.linalg.norm(cloudq)),original_radius=model.radius,
        heavy_PH=cloudPH,heavy_exact=cloudexact,light=cloudlight,
        scalar_H_minus_Href=float(cloudscalar-cloudref),point_not_global_integral=True)
    observations=[]
    for profile in ('source19','SOFT'):
        for t in (0.,args.duration/2,args.duration):
            row=ordered_family(model,data,sample,profile,t,pull['normal'],dem,K,R,reference,packet,y0,clock)
            if t==0 and row['exact_initial_identity_error']>1e-10:raise AssertionError('Original IC altered')
            for name in ('U_heavy','U_light'):
                if np.linalg.norm(row[name].conj().T@row[name]-np.eye(108))>1e-8:raise AssertionError('Nonunitary Fock frame')
            if not np.isfinite(row['particle_excess_onebody_energy']):raise AssertionError('Nonfinite physical particle excess')
            if abs(row['global_EM_charge_excess'])>1e-8:raise AssertionError('Lost physical FARRef charge compensation')
            if not np.isfinite(row['particle_excess_matter_plus_Coulomb_energy']):raise AssertionError('Nonfinite original Coulomb excess')
            prefix=profile+'_'+str(t)
            for name in ('U_heavy','U_light','heavy_connection','light_connection','source_relative_G_images','source_relative_core_G_images','source_relative_FARRef_right_images','source_relative_J_images','source_relative_contact_images','incoming_photon_projected_image','incoming_photon_derivative_body108','neutral_coordinates','charged_coordinates','neutral_counterphase'):
                arrays[prefix+'_'+name]=row.pop(name)
            for species in ('heavy','light'):
                banks=[row['heavy']] if species=='heavy' else list(row['light'].values())
                for bankindex,bank in enumerate(banks):
                    for name in ('coupling','gap','first_order_amplitudes'):
                        arrays[prefix+'_'+species+str(bankindex)+'_'+name]=bank['PH'].pop(name)
            arrays[prefix+'_gaussian_precision']=row['field_reference']['modal_q_precision']
            arrays[prefix+'_gaussian_linear']=row['field_reference']['modal_linear_coefficients']
            gauss=row.pop('field_reference')
            row.update(profile=profile,fielddomain=dict(radial_q_second_moment=gauss['moments']['radial_q_second_moment'],radius=packet['radius'],compact_remainder=gauss['cutoff_remainder']),
                       continuous_common_field_phase=gauss['phase'])
            observations.append(row)
    symbol=mean_mobile_symbol(model,np.array([[0.,0.,0.],[.01,0.,0.],[.02,0.,0.],[0.,.02,0.]]))
    arrays['mean_light_symbols']=symbol.pop('symbols')
    np.savez_compressed(args.output_npz,**arrays)
    receipt=dict(scope='NEW full108 exterior-Fock conditional interaction reference joined to actual all2430 covariance/SOURCE-relative G/original J/contact/gamma-in. Not full-H evolution or a pole.',
        inputs={name:dict(path=str(getattr(args,name)),sha256=sha(getattr(args,name))) for name in ('cache_npz','cache_manifest','covariance_npz','covariance_receipt','legal_control','original_filter')},
        sources={name:sha(Path(__file__).with_name(name)) for name in ('ordered_sector_reference.py','ordered_sector_reference_consumer.py')},
        runtime_binding=binding,runtime_capture='actual imported callable code/source archived BEFORE science; concurrent later provider edits do not rewrite this executed epoch',
        duration=args.duration,point_normal_norm=float(np.linalg.norm(q)),native_scalar_H_minus_Href=float(frozen_scalar-scalar_reference),
        covariance=covariance_observation,full108_PH_field_derivatives=derivative_rows,original_full2430_packet_node=cloud_observation,observations=observations,mean_mobile_band=symbol,
        mean_physical_geometry=mean_geometry,mean_SOURCE_full2430_gradient=mean_source_controls,
        generic_covariant_packet_module=packet_controls,
        original_LEGAL_reference_control=legal,
        reference_definition='Neutral U W_field Xi0; SOURCE quasimode U B(q,A) W_field Xi0, NOT W_field B Xi0. Original SOURCE commutator remains explicit residual; compact remainder transported; scalar counterphase already in coordinates exactly once.',
        residual_ledger=dict(retained='actual full h_H/h_L PH+multiPH action; all2430 B covariance; literal SOURCE/G/J/contact/gamma-in; actual directional full metric/density/Fock connection',
            scalar='actual Vfixed(q)-Vquadratic(q) evaluated at original source point',
            unpaid='full covariant kinetic commutator over2835 directions and its second jets, Coulomb multipair/resolvent, source d_t under metaplectic q+p, full spatial G closure, fullaux/all8 and legal-volume/source/field-domain graph payment',
            norm_of_whole_H_residual_NOT_certified=True,absolute_PH_not_particle_relative_failure=True),
        no_recentre_no_reground_no_refilter_no_mass_or_spin_fit=True,spin_definition='Actual SOURCE-relative G with neutral RIGHT and locking, not assigned Pauli. This one-axis evaluated family does NOT certify three-axis Casimir or spatial2pi.',
        output_npz_sha256=sha(args.output_npz),wall_seconds=time.perf_counter()-started,natural_exit=True,campaign_open=True)
    args.output_json.write_text(json.dumps(portable(receipt),indent=2)+'\n')
    print(json.dumps(portable(dict(wall_seconds=receipt['wall_seconds'],scalar_residual=receipt['native_scalar_H_minus_Href'],mean_band=symbol,
        observations=[dict(time=x['time'],profile=x['profile'],energy=x['particle_excess_onebody_energy'],J=x['particle_excess_original_J'],heavy=x['heavy']) for x in observations])),indent=2))

if __name__=='__main__':main()
