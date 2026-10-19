"""Actual leading primitive source-connected current vertex, no source/R PAP."""
from pathlib import Path
import gc,json,pickle,resource,sys,time
import numpy as np
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_cloud_probe import neutral_state,inner
from rich_common_source_vertex_20261016_fields import so3_words,direction_fields
from rich_common_source_vertex_20261016_words import sources,evaluate
from rich_common_source_vertex_20261016_moments import connected,native_comparison


def serial(value):
    if isinstance(value,np.ndarray):return {'real':value.real.tolist(),'imag':value.imag.tolist()}if np.iscomplexobj(value)else value.tolist()
    if isinstance(value,(complex,np.complexfloating)):return {'real':float(value.real),'imag':float(value.imag)}
    if isinstance(value,dict):return {str(k):serial(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v)for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def active_control(field,source_edge,receiver_edge):
    # Derive the controlled word algebra with Tphi=Trot=0 but ORIGINAL active
    # matter source/hops, potential, plaquette and unshared HE remaining.
    embedding=field['embedding'];phi=field['scalar']['phi'];zero_phi=np.zeros_like(phi);kp=phi@(zero_phi@phi-phi@zero_phi)-(zero_phi@phi-phi@zero_phi)@phi;coordinate=field['coordinate'];source=np.einsum('a,iaxy->ixy',field['graph']['axes'][source_edge],coordinate);receiver=np.einsum('a,iaxy->ixy',field['graph']['axes'][receiver_edge],coordinate);zero_rotation=np.zeros_like(field['kinetic']);vv=[]
    for i in range(3):
        for j in range(3):
            a=source[i];b=receiver[j];vv.append((a@zero_rotation@b-a@b@zero_rotation-zero_rotation@b@a+b@zero_rotation@a)@embedding)
    return {'Phi_double_commutator':kp,'rotor_double_commutator_lifts':np.asarray(vv),'kinetic_word_norm':float(np.linalg.norm(kp)+np.linalg.norm(vv)),
        'assertion':'Remove ONLY identified sharedPhi kinetic and mechanicalTrot; keep ALL original scalar/vector matter pulse/hops,scalarpotential,plaquette,HE. With commutingcoordinates, Hw(theta) commutes every quarkoperator and unsharedHE; reverse likewise. Thus actual crossfirst and all retardedwords vanish,forANYstate. Nativebody/RmatrixPAP may still respond and is a DIFFERENT finite model. SourceDeltaH remains active and its actual norms are retained.'}


def save_direction(out,field,raw,con,control):
    np.savez(out+'_field_lifts.npz',R_lifts=np.asarray([x[1]for x in field['lifts']]),Phi_lifts=np.asarray([x[2]for x in field['lifts']]),names=np.asarray([x[0]for x in field['lifts']]),R_Gram=field['Rgram'],Phi_Gram=field['Phigram'],R_complement_Gram=field['Rcomplement'],control_Phi=control['Phi_double_commutator'],control_rotor=control['rotor_double_commutator_lifts'])
    arrays={'connected_mean_u2_components':con['connected_mean_u2_components'],'connected_covariance_u2':con['connected_Weyl_current_covariance_u2_tensor']}
    for source,item in raw.items():
        arrays[source+'_Gram']=item['Gram'];arrays[source+'_R_complement_Gram']=item['R_complement_Gram'];arrays[source+'_new_material_Gram']=item['new_raw_support_Gram']
    np.savez(out+'_raw_tensor.npz',**arrays)


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time();data=load(config['neutral_basis_snapshot']);primitive=primitive_for(data);p=data['p'];coupling=load(config['neutral_coupling_snapshot']);states=np.load(config['neutral_states_snapshot']);omega=neutral_state(data,coupling,states['states'][:,0]);del data,coupling,states;gc.collect();rho_states=sources(primitive,omega);norms={name:float(inner(state,state).real)for name,state in rho_states.items()};field=so3_words(p);native=native_comparison(config,field['scalar']);result={'kind':'PRIMITIVE_CANONICAL_CONNECTED_CURRENT_VERTEX_BEFORE_SOURCE_AND_R_PROJECTION','parameters':p,'configuration':config,'raw_source_traces':norms,'field_word_receipt':field['receipt'],'native_PAP_comparison':native,'directions':{},'source_order':'rho0=OmegaOmega†;rhoB=Bdag0rho0B0 ORIGINALunmaskedhalftripletM+.5 epsilon/sqrt6;rhoL=Ldag2rho0L2 DOWN-WHITEflavour physicalSPIN-UP coefficient1;rhoBL=Bdag0Ldag2rho0L2B0,LactsFIRST. Allfour freshlyacted on same actualN32Omega,notcompactB/Lsite0/native sectoralternation.','functional':'Z(s,t;eta,theta,u)=Tr exp(i sum_a eta_a J_a) Utheta(u)[rho0+s rhoB+t rhoL+st rhoBL]Utheta†. Mixedsourceintensity log derivatives with exact rawtraces; NOTamputatedS-matrix. Conditional currentcomponent/Weylcovariance leadingu² evaluated,allpartitions retained.','law':'Same primitivegeometryH/p,exactinputPhi7finitealgebra,full commutingSO3 coordinate-word/derivative completion with degreecertificatej<=3,allCAR/PW19 reference and generatedoutputs. R10/sourcePAP is a distinct reviewedfiniteapproximation; no hiddenfundamentalnoncommutinggeometrylawfork.','limits':'Leadingoperatorwords,NOT integrated finite-u pulse trajectory. Primitive charges/spin/SU3/3+1/gauging,negative scalar quadratic/volume andmechanicaldensity INPUTS; no Maxwell/Coulomb/photon/labrotationalEM/freeCOM/field-onlyspin/mass/binding/lifetime/proton claim. Severe initialN32onehop order/Phi7/PW19 inputs remain; no ground/range permission gate. Other47/48/body408/CURRENT427/inducedlaws separate.'}
    np.savez(out+'_field_basis.npz',basis=field['basis'],rotations=field['rotations'],weights=field['weights'],input_embedding=field['embedding'],right_generators=field['right_generators'],Trot=field['kinetic'],Phi=field['scalar']['phi'],Tphi=field['scalar']['kinetic']/p['scalar_volume_ratio'])
    for direction in ('white','quark'):
        source_edge=2 if direction=='white'else 0;receiver_edge=0 if direction=='white'else 2;lift=direction_fields(field,source_edge,receiver_edge);raw={}
        for source_name,state in rho_states.items():
            raw[source_name]=evaluate(primitive,state,lift,direction,config['theta'],out+'_'+direction+'_'+source_name);print(json.dumps({'stage':'PRIMITIVE_MATCHED_SOURCE_VERTEX_COMPLETE','direction':direction,'source':source_name,'trace':norms[source_name],'full_raw_output_partitions':len(raw[source_name]['output_count_partitions'])}),flush=True)
        con=connected(raw);control=active_control(field,source_edge,receiver_edge);control['actual_active_source_DeltaH_norm2']={name:con['raw'][name]['active_source_DeltaH_norm2']for name in rho_states};control['connected_mean_u2_from_zero_kinetic_word']=float(control['kinetic_word_norm']);control['meaning']='Zero derived kinetic word above,not a supplied sourceoff/null. Actual unchangedsourceDeltaH normsNONZERO; allphi/R/sea/core outputs retained in full calculation.';save_direction(out+'_'+direction,lift,raw,con,control);result['directions'][direction]={'field_receipt':lift['receipt'],'canonical_connected':con,'active_source_control':control,'output_receipts':{name:{k:v for k,v in item.items()if k not in ('Gram','R_complement_Gram','new_raw_support_Gram')}for name,item in raw.items()}};Path(out+'_checkpoint.json').write_text(json.dumps(serial(result),indent=2)+'\n');print(json.dumps({'stage':'ACTUAL_CANONICAL_PRIMITIVE_VERTEX','direction':direction,'connected_current_u2':con['connected_mean_u2_total'],'connected_Weylvariance_u2':con['connected_Weyl_current_variance_u2_total'],'control_word_norm':control['kinetic_word_norm']}),flush=True);del raw;gc.collect()
    with open(out+'_CAR_coordinates.pkl','xb')as stream:pickle.dump({'primitive':primitive,'sources':rho_states,'field_lifts':'field_basis.npz plus direction_field_lifts.npz','meaning':'Actual localFock orthonormal coordinate dictionary afterALLprimitivewords; rawoutputs refer these IDs. No globalFockenumeration/sourceP. All source traces and signs original.'},stream,protocol=5)
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started};Path(out+'_result.json').write_text(json.dumps(serial(result),indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
