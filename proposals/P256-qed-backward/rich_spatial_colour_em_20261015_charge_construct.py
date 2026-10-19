"""PRIMARY new charge-orbit CAR/colour/phi/SO3/quantum-EM H and reference.
Raw physical Q-site projector packets, never the frozen bandpair control or its state.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from rich_spatial_colour_charge_20261015_model import packet_matrix
from rich_spatial_colour_charge_20261015_snapshot import load as load_native_body
from rich_spatial_colour_em_20261015_join import from_body,assemble,angular,native_source,native_C,native_charge_receiver
from rich_spatial_colour_em_20261015_snapshot import save_sparse,save_factor,save_body_primitives
from rich_spatial_colour_em_20261015_spectrum import lowest_total_M,integer_rotation_multiplets


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());p=dict(request);started=time.time();stored={};sectors={};results={};source_results={};native_records={};spectra={};laws={}
    startpath=inp.with_name(inp.name.replace('_input.json','_start.json'));binding=json.loads(startpath.read_text()) if startpath.exists() else {'binding':'No capture start receipt in direct invocation'}
    def checkpoint(stage):
        payload=dict(kind='PRIMARY_RAW_SITE_CHARGE_ORBIT_SPATIAL_QUANTUM_EM_H',stage=stage,parameters=p,source_input_recipe=binding,native_representation=native_records,sectors=sectors,results=results,sources=source_results,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if 'model' in locals_outer:
            payload.update(V_self=locals_outer['model']['V_self'],V_cross=locals_outer['model']['V_cross'])
        if 'reference_energy' in locals_outer:payload['reference_energy']=locals_outer['reference_energy']
        out.write_text(json.dumps(payload,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)
    locals_outer={}
    def body_checkpoint(stage,obj):
        if stage.endswith('_basis'):
            name=stage[:-6];masks,packets=packet_matrix(obj.seeds)
            stored[name+'_native_seed_masks']=np.array(masks,np.uint64);save_sparse(stored,name+'_native_seed_packets',packets)
            stored[name+'_native_charge3_site0']=obj.h;stored[name+'_native_twice_Sz']=obj.m
            for charge,white in obj.blocks.items():
                stored[name+'_native_white_'+str(charge)]=white;save_sparse(stored,name+'_native_Gram_'+str(charge),obj.grams[charge])
            native_records[name]=dict(dimension=obj.dimension,unprojected_seed_packets=len(obj.seeds),unprojected_seed_mask_support=len(masks),physical_isometry_error=obj.isometry_error,rank_records=obj.rank_records,seed_rank_records=obj.seed_rank_records,representation='|alpha,h>=sum_i W_i,alpha,h P_h(raw Q_em,site0)|seed_i>. P_h precedes any band cut. Sparse SEED packets are NOT full physical-orbit Fock expansions; coherent all-band-pair orders are implicit in exact charge-projector algebra.')
        checkpoint('native_'+stage)
    cache=inp.parent/request['native_body_cache'];cache_hash=hashlib.sha256(cache.read_bytes()).hexdigest()
    if cache_hash!=request['native_body_cache_sha256']:raise ValueError('Native body artifact differs from the explicitly pinned trusted cache')
    body=load_native_body(cache)
    body_parameters=dict(body['parameters'])
    physical_keys=['vertex_separation','path_height','path_quadrature','orientation_azimuth_points','orientation_polar_points','orientation_inertia','scalar_points','scalar_extent','scalar_metric','scalar_quartic','scalar_quadratic','scalar_coupling','electric','plaquette','vector_velocity','rank_absolute','rank_relative']
    for key in physical_keys:
        if key in request and key in body_parameters and request[key]!=body_parameters[key]:
            raise ValueError('Requested body parameter conflicts with exact native operator cache: '+key)
    p={key:body_parameters[key] for key in physical_keys};p.update(request)
    results['native_body_cache']=dict(path=cache.name,sha256=cache_hash,meaning='Exact trusted native numeric operators/isometry/source metrics only. NO supplier energies, reference or state loaded into the extended H.')
    for name in ['neutral','lepton','baryon']:body_checkpoint(name+'_basis',body['native'][name])
    model=from_body(body,p);locals_outer['model']=model;base=model['base'];field=model['field'];rotor=base['rotor'];scalar=base['scalar'];nr=rotor['dimension'];ne=field['dimension']
    stored.update(scalar_H=scalar['H'],scalar_grid=scalar['grid'],field_basis=np.array(field['basis']),field_HE=field['H_E'],field_C=field['C'],field_boundary=field['boundary'],rotor_R=rotor['R'],rotor_weights=rotor['weights'],rotor_basis=rotor['basis'],route_coefficients=model['coefficients']['values'],route_body_coefficients=model['coefficients']['body'],electric_raw_Gram=model['coefficients']['raw_electric_gram'],vertices=base['geometry']['vertices'])
    for axis in range(3):save_sparse(stored,'rotor_L'+str(axis),rotor['L'][axis]);stored['field_F'+str(axis)]=field['F'][axis]
    for axis in range(6):stored['field_Q'+str(axis)]=field['Q'][axis];stored['field_Pi'+str(axis)]=field['Pi'][axis]
    for edge,path in enumerate(base['geometry']['paths']):stored['path'+str(edge)]=path
    for name,sector in base['sectors'].items():
        op=base['operators'][name];h=assemble(model,name);laws[name]=h;save_factor(stored,name+'_H',h);save_body_primitives(stored,name,op)
        labels,effects=native_charge_receiver(model,name);stored[name+'_charge_labels']=labels
        for index,effect in enumerate(effects):save_sparse(stored,name+'_charge_effect_'+str(index),effect)
        identity=sparse.eye(sector['dimension']);charge_closure=op['density_delta2']-op['density_delta']@op['density_delta']
        sectors[name]=dict(dimension=h.shape[0],matter_dimension=sector['dimension'],term_count=len(h.terms),Q_em=float(op['Q'].diagonal().real.mean()),site_charge_labels=labels,site_charge_PVM_closure=float(sparse.linalg.norm(sum(effects)-identity)),native_D2_vs_D_squared_Frobenius=float(sparse.linalg.norm(charge_closure)),native_isometry_error=sector['native_isometry'].isometry_error,full_body_M=True,full_material_SO3_dimension=nr,complete_field_shell_dimension=ne,Gauss_scope='Occupied seed is global colour singlet/PW vacuum; raw colour-covariant hopping/electric/loop/source images and EM charge projectors commute colour Gauss. Native packet construction preserves that physical kernel. No invented numerical raw-Fock-Gauss-zero certificate.')
        stored[name+'_native_charge3_site0']=sector['charge3_site0'];stored[name+'_native_twice_Sz']=sector['twice_Sz']
        checkpoint('constructed_'+name)
    for name,values in base['source_matter'].items():
        for key,source_matrix in values.items():
            save_sparse(stored,name+'_source_'+key+'_matter',source_matrix);save_sparse(stored,name+'_source_'+key+'_full_metric',base['source_metric_full'][name][key])
    for name,value in base['C_matter'].items():save_sparse(stored,name+'_C_matter',value)
    checkpoint('all_H_source_metric_C_charge_PVM_before_eigensolvers')
    for name in ['neutral','lepton','baryon']:
        h=laws[name];e,v,solver=lowest_total_M(model,name,h,0 if name=='neutral' else 1,p);spectra[name]=(e,v);stored[name+'_e']=e;stored[name+'_v']=v
        jops=[angular(model,name,axis) for axis in range(3)];probe=np.random.default_rng(222).normal(size=h.shape[0])+1j*np.random.default_rng(333).normal(size=h.shape[0]);probe/=np.linalg.norm(probe)
        results[name]=dict(ground_energy_in_complete_total_M_sector=float(e[0]),stationary_solver=solver,eigen_residuals=[float(np.linalg.norm(h@v[:,index]-e[index]*v[:,index])) for index in range(len(e))],physical_J2=[float(sum(np.linalg.norm(j@v[:,index])**2 for j in jops)) for index in range(len(e))],HJ_random_commutators=[float(np.linalg.norm(h@(j@probe)-j@(h@probe))) for j in jops],Gram_error=float(np.linalg.norm(v.conj().T@v-np.eye(v.shape[1]))))
        checkpoint('solved_'+name)
        print(json.dumps(dict(stage='solved',name=name,dimension=h.shape[0],M_solver=solver,energies=e[:6].tolist(),J2=results[name]['physical_J2'][:6],peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)
    reference=spectra['neutral'][1][:,0];e0=float(spectra['neutral'][0][0]);stored['neutral_reference']=reference;locals_outer['reference_energy']=e0
    mode_e,mode_v,mode_labels,mode_evidence=integer_rotation_multiplets(model,'neutral',*spectra['neutral'],p)
    stored['neutral_modes_e']=mode_e;stored['neutral_modes_v']=mode_v;stored['neutral_modes_JM']=mode_labels;results['neutral_rotation_multiplets']=mode_evidence
    results['neutral_mode_eigen_residual']=float(np.linalg.norm(laws['neutral']@mode_v-mode_v*mode_e[None,:]))
    cn=native_C(model,'neutral');results['reference_C']=dict(overlap_real=float(np.vdot(reference,cn@reference).real),overlap_imag=float(np.vdot(reference,cn@reference).imag),unitarity_on_reference=float(abs(np.linalg.norm(cn@reference)-1)))
    for name in ['lepton','baryon']:
        anti='anti'+name;c=native_C(model,name);e,v=spectra[name];av=c@v;spectra[anti]=(e,av);stored[anti+'_e']=e;stored[anti+'_v']=av
        results[anti]=dict(C_H_on_low_poles=float(np.linalg.norm(laws[anti]@av-c@(laws[name]@v))),eigen_residual=float(np.linalg.norm(laws[anti]@av-av*e[None,:])))
        results[name]['C_source_operators']={}
        for key in base['source_matter'][name]:
            sa=base['source_matter'][anti][key];s=base['source_matter'][name][key];c0=base['C_matter']['neutral'];ca=base['C_matter'][name]
            results[name]['C_source_operators'][key]=float(sparse.linalg.norm(sa@c0-ca@s))
        checkpoint('C_'+name)
    for name,values in base['source_matter'].items():
        source_results[name]={};h=laws[name];e,v=spectra[name]
        for key,body_source in values.items():
            insertion=native_source(model,name,key);raw=insertion@reference;strength=float(np.linalg.norm(raw)**2);metric=base['source_metric_full'][name][key]
            x=reference.reshape(metric.shape[0],-1);full=float(np.sum(x.conj()*(metric@x)).real);m=float(key.split(':')[-1])
            qtarget=base['operators'][name]['Q'];qneutral=base['operators']['neutral']['Q'];charge_error=float(sparse.linalg.norm(qtarget@body_source-body_source@qneutral-(sectors[name]['Q_em']-sectors['neutral']['Q_em'])*body_source))
            entry=dict(physical_source_component_M=m,raw_full_strength=full,raw_retained_strength=strength,source_projection_omitted_norm_squared=full-strength,raw_preparation_energy=float(np.vdot(raw,h@raw).real/strength-e0),source_Q_em_error=charge_error,raw_metric_vs_retained_Frobenius=float(sparse.linalg.norm(metric-body_source.conj().T@body_source)))
            if abs(m-.5)<1e-12:
                amplitudes=v.conj().T@raw;weights=abs(amplitudes)**2;bright=np.flatnonzero(weights>p['bright_tolerance'])
                entry.update(low_total_M_half_gaps=(e-e0).tolist(),low_raw_weights=weights.tolist(),outside_low_window_strength=float(strength-weights.sum()))
                if len(bright):
                    index=bright[0];pole=abs(e-e[index])<p['degeneracy_tolerance'];state=v[:,pole]@amplitudes[pole];state/=np.linalg.norm(state);stored[name+'_source_'+key+'_pole']=state
                    entry.update(lowest_computed_bright_gap=float(e[index]-e0),pole_raw_weight=float(weights[pole].sum()),pole_J2=float(sum(np.linalg.norm(angular(model,name,axis)@state)**2 for axis in range(3))),pole_eigen_residual=float(np.linalg.norm(h@state-e[index]*state)))
                else:entry['lowest_computed_bright_gap']=None
            else:entry['spectral_scope']='Native source component retained and raw norm/cost measured, but no pole decomposition in the DIFFERENT totalM+.5 display window; full all-M H and resolvent interface remain available.'
            source_results[name][key]=entry
        checkpoint('sources_'+name)
    checkpoint('complete')
    print(json.dumps(dict(stage='complete',reference_energy=e0,sectors=sectors,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)


if __name__=='__main__':main()
