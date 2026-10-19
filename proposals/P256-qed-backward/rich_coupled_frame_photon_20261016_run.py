"""Fresh state/source, two-leg current/field responses and real field-only kicks."""
import json
import gc
from math import factorial
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply
import rich_coupled_frame_photon_20261016_model as m

ROOT=Path(__file__).resolve().parent
P='rich_coupled_frame_photon_20261016_'


def expect(v,a):return float(np.vdot(v,a@v).real)


def evolve(h,v,t):
    return expm_multiply(-1j*t*h,v,traceA=-1j*t*h.diagonal().sum())


def matter_apply(model,a,v):return (a@v.reshape(model.shape)).reshape(-1)


def kick_vacuum(model,state,axis,epsilon):
    """Exact unbounded exp(i epsilon q_axis)|0>, then P and normalize."""
    f=np.zeros(model.nfield,dtype=np.complex128)
    for i,occupation in enumerate(model.basis):
        if any(occupation[b] for b in range(3) if b!=axis):continue
        n=occupation[axis]
        f[i]=np.exp(-epsilon**2/4)*(1j*epsilon/np.sqrt(2))**n/np.sqrt(factorial(n))
    retained=float(np.linalg.norm(f)**2)
    matter=state.reshape(model.shape)[:,model.basis.index((0,0,0))]
    return np.kron(matter,f/np.sqrt(retained)),1-retained


def measure(model,v,vertices):
    hm=model.matter;x=v.reshape(model.shape)
    ev=lambda a:float(np.vdot(x,a@x).real)
    h=model.H@v
    gx=x.reshape(hm.shape+(model.nfield,))
    topg=max(z[0] for z in hm.g['labels']);topR=max(z[0] for z in hm.material['labels'])
    gweight=sum(float(np.linalg.norm(gx[:,i])**2) for i,z in enumerate(hm.g['labels']) if z[0]==topg)
    rweight=sum(float(np.linalg.norm(gx[:,:,i])**2) for i,z in enumerate(hm.material['labels']) if z[0]==topR)
    return dict(norm=float(np.vdot(v,v).real),energy=float(np.vdot(v,h).real),r=ev(hm.r),force=ev(hm.force),beta=ev(hm.beta),HE=ev(hm.HE),Hg=ev(hm.Hg),HR=ev(hm.HR),Hrelative=ev(hm.Hrelative),Hfield=expect(v,model.Hfield),J2=sum(float(np.linalg.norm(a@v)**2) for a in model.J),Jmatter2=sum(float(np.linalg.norm(matter_apply(model,a,v))**2) for a in hm.J),density=[ev(a) for a in hm.n],dipole=[ev(a) for a in model.D],material_current=[-ev(a) for a in vertices],field_coordinate=[float(np.vdot(x,x@a.T).real) for a in model.q],electric=[expect(v,model.electric(a)) for a in range(3)],electric_velocity=[expect(v,model.electric(a))/model.parameters['cavity_inertia'] for a in range(3)],g_top_weight=gweight,R_top_weight=rweight,**model.exterior_image(v))


def main():
    raw={};result=dict(kind='ACTUAL_NEW_SINGLE_LAW_NONCOMPACT_VECTOR_CAVITY_FRAME',parameters=m.DEFAULTS,protocol_times=[0.,.08,.16],sign='q_s=n_s-b_s; J=-d_A Hvelocity at fixed canonical q,Pi,LR; length H=Hm+(Pi+D)^2/(2I)',receiver='Uniform lab E at x=(0,0,0) and x=(0,0,1): identical mode function by declaration, not propagating/nonforward scattering')
    reference=m.Model(4);neutral=reference.preparation();raw['neutral_input']=neutral
    nr_vertices,nr_contacts=reference.source_vertices()
    neutral_states=np.array([neutral,evolve(reference.H,neutral,.08),evolve(reference.H,neutral,.16)])
    raw['neutral_states']=neutral_states
    result['neutral']=dict(shape=reference.shape,dynamics=[measure(reference,v,nr_vertices) for v in neutral_states])
    result['branches']={}
    for number,create,name,component in [(5,True,'addition',0),(3,False,'removal',1)]:
        model=m.Model(number);hm=model.matter;vertices,contact=model.source_vertices()
        packets=[model.source_from(reference,mu,create)@neutral for mu in range(8)]
        weights=[float(np.linalg.norm(v)**2) for v in packets]
        v=packets[component]/np.sqrt(weights[component]);raw[name+'_input']=v
        b=model.source_from(reference,component,create)
        source_connections={}
        for label,a,ar in [('HE',hm.HE,reference.matter.HE),('Hg',hm.Hg,reference.matter.Hg),('HR',hm.HR,reference.matter.HR),('Hradial',hm.Hradial,reference.matter.Hradial),('Hmass',hm.Hmass,reference.matter.Hmass),('Hrelative',hm.Hrelative,reference.matter.Hrelative)]:
            source_connections[label]=float(np.linalg.norm(matter_apply(model,a,b@neutral)-b@matter_apply(reference,ar,neutral)))
        source_connections['Hfield']=float(np.linalg.norm(model.Hfield@(b@neutral)-b@(reference.Hfield@neutral)))
        source_connections['fullH']=float(np.linalg.norm(model.H@(b@neutral)-b@(reference.H@neutral)))
        # Canonical field-only momentum kick: no CAR/g/R/r operation at t=0.
        eps=.23;kicked=[kick_vacuum(model,v,a,eps) for a in range(2)]
        kicks=[pair[0] for pair in kicked]
        starts=np.column_stack([v,*kicks]);mid=evolve(model.H,starts,.08);end=evolve(model.H,starts,.16)
        raw[name+'_starts']=starts;raw[name+'_mid']=mid;raw[name+'_end']=end
        protocols={label:[measure(model,starts[:,i],vertices),measure(model,mid[:,i],vertices),measure(model,end[:,i],vertices)] for i,label in enumerate(['baseline','field_kick_x','field_kick_y'])}
        # Both actual legs and time orders; normalize by the SAME fresh source.
        # C_ba(t,0)=<psi(t)|Vb U(t) Va|psi(0)>; no stationary S matrix claimed.
        legs=[-matter_apply(model,a,v) for a in vertices[:2]]+[model.electric(a)@v for a in range(2)]
        propagated=evolve(model.H,np.column_stack(legs),.16)
        kernels=np.zeros((4,4),dtype=np.complex128)
        for out in range(4):
            left=-matter_apply(model,vertices[out],end[:,0]) if out<2 else model.electric(out-2)@end[:,0]
            for inc in range(4):kernels[out,inc]=np.vdot(left,propagated[:,inc])
        raw[name+'_two_leg_matrix']=kernels;raw[name+'_two_leg_propagated']=propagated
        raw[name+'_reverse_timeorder_matrix']=kernels.conj()
        raw[name+'_retarded_commutator']=2*kernels.imag
        raw[name+'_contact_xy']=np.array([[np.vdot(v,matter_apply(model,contact[a,bb],v)) for bb in range(2)] for a in range(2)])
        raw[name+'_source_connections']=np.array(list(source_connections.values()))
        # Correct source-contact action versus the tempting projected D products.
        premature={};mechanical={};noether=[];projected_current_defect=[]
        for a in range(2):
            d=model.D[a];h0=hm.H
            wrong=-(d@(d@h0)-2*d@h0@d+h0@d@d)
            premature[str(a)]=float(np.linalg.norm(matter_apply(model,contact[a,a]-wrong,v)))
            mechanical[str(a)]=float(np.linalg.norm(matter_apply(model,1j*(d@hm.HR-hm.HR@d),v)))
            for state in [v,end[:,0]]:
                dv=matter_apply(model,d,state)
                dyn=1j*(model.H@dv-matter_apply(model,d,model.H@state))
                projected_current_defect.append(float(np.linalg.norm(dyn+matter_apply(model,vertices[a],state))))
        for state in [v,end[:,0]]:noether.append(max(float(np.linalg.norm(model.H@(j@state)-j@(model.H@state))) for j in model.J))
        # Actual H hopping/path-off control. Same complete input, no observable deleted.
        hoff=model.H-model.lift(hm.Hrelative)
        off=evolve(hoff,kicks[0],.16);raw[name+'_hopping_off_end']=off
        hfieldoff=model.Hmatter+sp.kron(model.identity,model.p2,format='csr')/(2*model.parameters['cavity_inertia'])
        fieldoff=evolve(hfieldoff,kicks[0],.16);raw[name+'_field_path_off_end']=fieldoff
        result['branches'][name]=dict(number=number,charge=number-4,source_component=component,source_weights=weights,source_weight_sum=sum(weights),source_connections=source_connections,source_energy_cost=expect(v,model.H)-expect(neutral,reference.H),source_cost_is_not_inertial_mass=True,shape=model.shape,protocols=protocols,field_kick_work=[expect(k,model.H)-expect(v,model.H) for k in kicks],initial_density_differences=[float(np.max(abs(np.array(protocols[label][0]['density'])-np.array(protocols['baseline'][0]['density'])))) for label in ['field_kick_x','field_kick_y']],two_leg_labels=['physical_current_x_negative_vertex','physical_current_y_negative_vertex','electric_x','electric_y'],two_leg_real=kernels.real.tolist(),two_leg_imag=kernels.imag.tolist(),contact_real=raw[name+'_contact_xy'].real.tolist(),contact_imag=raw[name+'_contact_xy'].imag.tolist(),premature_projected_contact_error=premature,mechanical_vertex_norm=mechanical,projected_current_vs_full_source_error=projected_current_defect,projected_total_J_H_commutator=noether,hopping_off_end=measure(model,off,vertices),hopping_on_off_packet_difference=float(np.linalg.norm(end[:,1]-off)),same_input_hopping_off=True)
        result['branches'][name]['retarded_commutator']=raw[name+'_retarded_commutator'].tolist()
        result['branches'][name]['response_normalization']='Unit-norm fresh charged B packet; canonical q,Pi with hbar=1, I=2; no LSZ/free photon frequency normalization. Reverse order is elementwise conjugate C*, not silently omitted; retarded -i(C-C*)=2ImC. Current-source response also has instantaneous full K contact.'
        result['branches'][name]['field_path_off_end']=measure(model,fieldoff,vertices)
        result['branches'][name]['field_path_on_off_packet_difference']=float(np.linalg.norm(end[:,1]-fieldoff))
        result['branches'][name]['field_path_off_meaning']='Actual control H=H40+Pi^2/(2I): D removed from complete electric square, same fresh input/kick, not deleting measured electric or current observable.'
        result['branches'][name]['feedback']={label:{key:(np.array(protocols[label][-1][key])-np.array(protocols['baseline'][-1][key])).tolist() for key in ['r','Hg','HR','HE','density','material_current','electric']} for label in ['field_kick_x','field_kick_y']}
        result['branches'][name]['field_kick_discarded_weight']=[pair[1] for pair in kicked]
        result['branches'][name]['field_kick_projection']='Exact unbounded coherent kick on field vacuum, then numerical P and unit normalization; no claim exp(i epsilon PqP) equals P exp(i epsilon q)P.'
        print(name,json.dumps(result['branches'][name]),flush=True)
        np.savez_compressed(ROOT/(P+name+'_raw.npz'),**{k:val for k,val in raw.items() if k.startswith(name)})
        (ROOT/(P+'result.json')).write_text(json.dumps(result,indent=2)+'\n')
        del model,vertices,contact,packets,propagated,starts,mid,end,off,hoff,b,legs;gc.collect()
    np.savez_compressed(ROOT/(P+'neutral_raw.npz'),neutral_input=neutral,neutral_states=neutral_states)
    print(json.dumps(dict(result=P+'result.json',branches=list(result['branches']))),flush=True)


if __name__=='__main__':main()
