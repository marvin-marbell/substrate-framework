"""Fresh primitive field/CAR words and raw-logZ source cumulants, no output replay."""
from pathlib import Path
from itertools import product
import gc
import json
import pickle
import resource
import time
import numpy as np
from scipy import sparse
from rich_primitive_vertex_review_20261016_fields import construct,direction,gram
from rich_primitive_vertex_review_20261016_car import sources,PairAlgebra,SourceContraction
ROOT=Path(__file__).resolve().parent
P='rich_primitive_vertex_review_20261016_'
ORDER=('0','B','L','BL')
POWERS=((0,0),(1,0),(0,1),(1,1))


def serial(value):
    if isinstance(value,np.ndarray):return dict(real=value.real.tolist(),imag=value.imag.tolist()) if np.iscomplexobj(value) else value.tolist()
    if isinstance(value,(complex,np.complexfloating)):return dict(real=float(value.real),imag=float(value.imag))
    if isinstance(value,dict):return {str(key):serial(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [serial(item) for item in value]
    if isinstance(value,np.generic):return value.item()
    return value


def multiply(a,b):
    out=np.zeros((2,2),complex)
    for i,j,k,l in product(range(2),repeat=4):
        if i+k<2 and j+l<2:out[i+k,j+l]+=a[i,j]*b[k,l]
    return out


def polynomial(values):
    out=np.zeros((2,2),complex)
    for name,power in zip(ORDER,POWERS):out[power]=values[name]
    return out


def inverse(values):
    n=polynomial(values);out=np.zeros((2,2),complex);out[0,0]=1/n[0,0]
    for i,j in ((1,0),(0,1),(1,1)):
        out[i,j]=-sum(n[a,b]*out[i-a,j-b] for a in range(i+1) for b in range(j+1) if a or b)/n[0,0]
    if np.linalg.norm(multiply(n,out)-np.array([[1.,0.],[0.,0.]]))>1e-12:raise AssertionError('Raw trace formal inverse failed')
    return out


def connected(raw,norms):
    inv=inverse(norms);inv2=multiply(inv,inv);mean=np.zeros(4);cov=np.zeros((4,4));weights={}
    for name,power in zip(ORDER,POWERS):
        basis=np.zeros((2,2));basis[power]=1;weights[name]=float(multiply(basis,inv)[1,1].real)
    for a in range(4):
        va=polynomial({name:raw[name]['V'][a] for name in ORDER});ma=polynomial({name:raw[name]['M'][a] for name in ORDER});mean[a]=float((-.5*multiply(va,inv)[1,1]).real)
        for b in range(4):
            vb=polynomial({name:raw[name]['V'][b] for name in ORDER});mb=polynomial({name:raw[name]['M'][b] for name in ORDER});av=polynomial({name:(raw[name]['A'][a,b]+raw[name]['A'][b,a]).real for name in ORDER})
            cov[a,b]=float((-.5*multiply(av,inv)[1,1]+.5*multiply(multiply(ma,vb)+multiply(mb,va),inv2)[1,1]).real)
    return dict(mean_components=mean,mean_total=float(mean.sum()),Weyl_covariance_u2=cov,current_variance_u2_total=float(cov.sum()),raw_source_weights=weights,trace_inverse_polynomial=inv,trace_inverse2_polynomial=inv2)


def receiver_and_source(q,w,source,theta):
    receiver=w if source=='quark' else q;active=q if source=='quark' else w
    jfull=[receiver.operator('current',i,0.) for i in range(4)];j=[receiver.single('current',i,0.) for i in range(4)];dfull=[active.operator('delta',i,theta) for i in range(4)];d=[active.single('delta',i,theta) for i in range(4)]
    return receiver,active,jfull,j,dfull,d


def material_order(source,receiver_matrix,source_matrix):
    return (source_matrix,receiver_matrix) if source=='quark' else (receiver_matrix,source_matrix)


def raw_moments(contraction,q,w,field,lifts,source,theta):
    receiver,active,jfull,j,dfull,d=receiver_and_source(q,w,source,theta);iq=sparse.eye(len(q.pairs),format='csr');iw=sparse.eye(len(w.pairs),format='csr');e=field['E'];raw=dict(M=np.zeros(4,complex),V=np.zeros(4,complex),A=np.zeros((4,4),complex),vertex_10=np.zeros(10,complex))
    products={(a,b):receiver.product(jfull[a],jfull[b]) for a,b in product(range(4),repeat=2)}
    for a in range(4):
        aq,bw=(iq,j[a]) if source=='quark' else (j[a],iw);r=e.conj().T@lifts['receiver'][a][0];phi=lifts['receiver'][a][1];raw['M'][a]=contraction.expectation(aq,bw,r,phi)
    for index,(i,jindex,rout,pout) in enumerate(lifts['vertices']):
        aq,bw=material_order(source,j[jindex],d[i]);value=contraction.expectation(aq,bw,e.conj().T@rout,pout);raw['vertex_10'][index]=value;raw['V'][jindex]+=value
        for a in range(4):
            aq,bw=material_order(source,products[a,jindex],d[i]);r,phi=gram(lifts['receiver'][a],(rout,pout));raw['A'][a,jindex]+=contraction.expectation(aq,bw,r,phi)
    # Physically wrong finite CCR replacement, independently exercised below.
    aq,bw=material_order(source,j[0],d[0]);canonical=np.eye(7)/(field['p']['scalar_metric']*field['p']['scalar_volume_ratio']);raw['wrong_CCR_SS']=contraction.expectation(aq,bw,np.eye(10),canonical)
    return raw


def source_action_norms(contraction,q,w,field,lifts,source,theta):
    active=q if source=='quark' else w;other=sparse.eye(len(w.pairs) if source=='quark' else len(q.pairs),format='csr');result={}
    for kind in ('delta','current','contact'):
        operators=[active.operator(kind,i,theta) for i in range(4)];value=0.+0j
        for a,b in product(range(4),repeat=2):
            kernel=active.product(operators[a],operators[b]);aq,bw=(kernel,other) if source=='quark' else (other,kernel);r,phi=gram(lifts['source'][a],lifts['source'][b]);value+=contraction.expectation(aq,bw,r,phi)
        result[kind+'_norm2']=float(value.real)
    if result['delta_norm2']<1e-5:raise AssertionError('Mediator OFF control lacks an active matter pulse')
    return result


def selected_rich_norm(contraction,q,w,field,lifts,source,theta):
    receiver,active,jfull,j,dfull,d=receiver_and_source(q,w,source,theta);e=field['E'];projector=e@e.conj().T;v00=lifts['vertices'][0];v11=next(v for v in lifts['vertices'] if v[0]==1 and v[1]==1);channels=(v00,v11);terms=np.zeros((2,2),complex);higher=0.
    for a,left in enumerate(channels):
        for b,right in enumerate(channels):
            i,jindex,lr,lp=left;k,lindex,rr,rp=right
            kr=receiver.product(jfull[jindex],jfull[lindex]);ks=active.product(dfull[i],dfull[k]);aq,bw=material_order(source,kr,ks);r,phi=gram((lr,lp),(rr,rp));terms[a,b]=contraction.expectation(aq,bw,r,phi)
            if a==b==1:higher=float(contraction.expectation(aq,bw,lr.conj().T@(np.eye(84)-projector)@rr,phi).real)
    # Save the exact compact factorization for the selected output, not author Vpsi.
    for a in (0,1):
        sparse.save_npz(ROOT/(P+source+f'_selected_receiverK{a}.npz'),jfull[a]);sparse.save_npz(ROOT/(P+source+f'_selected_sourceK{a}.npz'),dfull[a])
    np.savez(ROOT/(P+source+'_selected_rich_raw.npz'),coherent_selected_Gram=terms,R_lifts=np.array([v[2] for v in channels]),Phi_lifts=np.array([v[3] for v in channels]),input_coefficients=contraction.values,input_keys=np.array(contraction.keys),quark_input_pairs=np.array(q.pairs),white_input_pairs=np.array(w.pairs),quark_output_pairs=np.array(q.output_pairs),white_output_pairs=np.array(w.output_pairs))
    return dict(selected_channels=['V00','V11'],coherent_selected_norm2=float(terms.sum().real),individual_norms2=np.diag(terms).real.tolist(),selected_V11_R_above_input_j1_norm2=higher,scope='Own full-pair KdagK plus exact84field lifts retains ALL intermediate/outgoing CAR/R/Phi/PW in these SELECTED rich outputs and their SS/VV interference. Full10-channel Vpsi norm/newmaterial tensor NOT independently rebuilt; no90minute27-output replay or source-return criterion.')


def mediator_off(field,lifts,active_norms):
    phi=field['Phi'];zero_phi=np.zeros((7,7));kp=2*phi@zero_phi@phi-phi@phi@zero_phi-zero_phi@phi@phi;e=field['E'];zt=np.zeros((84,84));source_edge=lifts['receipt']['source_edge'];receiver_edge=lifts['receipt']['receiver_edge'];a=np.einsum('a,iaxy->ixy',field['graph']['axes'][source_edge],field['R']);b=np.einsum('a,iaxy->ixy',field['graph']['axes'][receiver_edge],field['R']);words=np.array([(a[i]@zt@b[j]-a[i]@b[j]@zt-zt@b[j]@a[i]+b[j]@zt@a[i])@e for i,j in product(range(3),repeat=2)])
    norm=float(np.linalg.norm(kp)+np.linalg.norm(words))
    if norm>1e-13 or active_norms['delta_norm2']<=1e-5:raise AssertionError('Actual active-source mediator control failed')
    return dict(shared_kinetic_word_norm=norm,unchanged_active_joint_DeltaH_norm2=active_norms['delta_norm2'],definition='Only Tphi/Trot removed. Actual matter source/hops, scalar potential, plaquette and unshared HE retained. Operatorial zero primitive vertex for ANYstate, not sourceOFF/native PAP null.')


def main():
    started=time.time();primitive,packets,norms=sources(ROOT);field=construct(primitive.p);q=PairAlgebra(primitive,packets,'quark');w=PairAlgebra(primitive,packets,'white')
    # All actual quark/white bilinears have disjoint member modes, before any P.
    q_modes={index for (edge,f,white,c,d),hops in primitive.maps.items() if edge==0 and not white for ops,spin,maps in hops for index,flag in ops};w_modes={index for (edge,f,white,c,d),hops in primitive.maps.items() if edge==2 and white for ops,spin,maps in hops for index,flag in ops}
    if q_modes&w_modes:raise AssertionError('Owning CAR words are not disjoint even bilinears')
    result=dict(kind='INDEPENDENT_PRIMITIVE_CANONICAL_CONNECTED_RETARDED_VERTEX',raw_source_traces=norms,field_receipt=field['receipt'],CAR_pair_receipts=dict(quark=q.receipt(),white=w.receipt()),actual_CAR_cross_mode_intersection=sorted(q_modes&w_modes),directions={})
    np.savez(ROOT/(P+'field_raw.npz'),basis=field['basis'],rotations=field['rotations'],weights=field['weights'],input_embedding=field['E'],Trot=field['T'],Phi=field['Phi'],Kphi=field['Kphi'])
    for source in ('white','quark'):
        lift=direction(field,2 if source=='white' else 0,0 if source=='white' else 2);raw={};bl=None
        for name in ORDER:
            contraction=SourceContraction(packets[name],q,w);raw[name]=raw_moments(contraction,q,w,field,lift,source,.3)
            if name=='BL':bl=contraction
            print(json.dumps(dict(stage='own_primitive_pair_moments',direction=source,source=name,raw_vertex_components=serial(raw[name]['V']))),flush=True)
        con=connected(raw,norms);actions=source_action_norms(bl,q,w,field,lift,source,.3);rich=selected_rich_norm(bl,q,w,field,lift,source,.3);control=mediator_off(field,lift,actions)
        wrong={name:{**item,'V':item['V'].copy()} for name,item in raw.items()}
        for name in ORDER:wrong[name]['V'][0]=raw[name]['wrong_CCR_SS']
        ccr=connected(wrong,norms)['mean_total'];naive=float((-.5*(raw['BL']['V'].sum()/norms['BL']-raw['B']['V'].sum()/norms['B']-raw['L']['V'].sum()/norms['L']+raw['0']['V'].sum()/norms['0'])).real)
        bad=dict(Phi_CCR_substitution_connected_mean=ccr,naive_normalized_sector_alternating_mean=naive,actual_connected_mean=con['mean_total']);rejections={}
        for key in ('Phi_CCR_substitution_connected_mean','naive_normalized_sector_alternating_mean'):
            if abs(bad[key]-con['mean_total'])>1e-10:
                try:assert abs(bad[key]-con['mean_total'])<1e-10,'Wrong physical construction changed independently reconstructed canonical vertex: '+key
                except AssertionError as error:rejections[key]=str(error)
        if not rejections:raise AssertionError('Neither wrong physical construction reached a discriminating canonical consumer')
        arrays=dict(raw_source_traces=np.array([norms[name] for name in ORDER]),M=np.array([raw[name]['M'] for name in ORDER]),V=np.array([raw[name]['V'] for name in ORDER]),A=np.array([raw[name]['A'] for name in ORDER]),vertex_10=np.array([raw[name]['vertex_10'] for name in ORDER]),connected_mean_components=con['mean_components'],connected_Weyl_covariance=con['Weyl_covariance_u2'],wrong_Phi_CCR_SS=np.array([raw[name]['wrong_CCR_SS'] for name in ORDER]))
        np.savez(ROOT/(P+source+'_moments_raw.npz'),**arrays);result['directions'][source]=dict(field=lift['receipt'],connected=con,active_source_actions=actions,selected_rich_output=rich,mediator_OFF=control,wrong_cases=bad,wrong_case_assertions=rejections)
        (ROOT/(P+'partial_result.json')).write_text(json.dumps(serial(result),indent=2)+'\n');print(json.dumps(dict(stage='own_primitive_connected_result',direction=source,result=serial(result['directions'][source]))),flush=True);gc.collect()
    # Secondary comparison only AFTER independent primitive reconstruction above.
    comparisons={}
    for source in ('white','quark'):
        producer=np.load(ROOT/('rich_common_source_vertex_20261016_actual1_'+source+'_raw_tensor.npz'));our=result['directions'][source]['connected'];comparisons[source]=dict(mean_components_difference=float(np.linalg.norm(our['mean_components']-producer['connected_mean_u2_components'])),covariance_tensor_difference=float(np.linalg.norm(our['Weyl_covariance_u2']-producer['connected_covariance_u2'])),scope='Producer raw tensor is secondary output comparison ONLY, never a primary field/CAR/moment input.')
    result['secondary_producer_comparison']=comparisons;result['resource']=dict(elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['limits']='Canonical source-intensity logZ cumulant, leading u² mean/covariance moments only. No integrated pulse/scattering, S-matrix/amputation, force/probability/mass/photon/Maxwell/Coulomb/binding/proton/lifetime/freeCOM or derived primitive field-only spin. Original Phi7/PW19/onehopOmega/charges/spin/action cuts remain. Old native PAP slopes are a separate finite matrix law, not replaced or reverified. No full27-output/newmaterial tensor independently rebuilt.'
    (ROOT/(P+'result.json')).write_text(json.dumps(serial(result),indent=2)+'\n');print(json.dumps(dict(kind=result['kind'],resource=result['resource'])),flush=True)


if __name__=='__main__':main()
