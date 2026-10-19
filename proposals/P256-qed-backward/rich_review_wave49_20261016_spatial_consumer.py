"""Own fresh states on active48 H; physical fixed-canonical source variations."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, expm_multiply
import rich_spatial_frame_photon_20261016_model as law
import rich_spatial_frame_photon_20261016_fields as fields

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def evolve(m,state,time,hopping=True,motion=True):
    def action(v):return m.apply(np.asarray(v).reshape(-1),hopping,motion)
    op=LinearOperator((m.dimension,m.dimension),matvec=lambda v:-1j*time*action(v),rmatvec=lambda v:1j*time*action(v),dtype=complex)
    return expm_multiply(op,state,traceA=0)


def physical_hops(m):
    """Independent geometry reconstruction, not producer's private cache tuples."""
    hm=m.matter;sites=hm.geo['sites'];lookup={tuple(v):i for i,v in enumerate(sites)}
    for axis in range(3):
        for source in range(8):
            position=sites[source].copy();position[axis]^=1;target=lookup[tuple(position)]
            if target<source:continue
            gamma=hm.geo['gamma'][axis][target,source]
            if not abs(gamma):continue
            phi=law.kernel.segment(m.x[:,source],m.x[:,target],m.modes)[0]
            disp=fields.displacement(m.basis,m.basis,-phi)
            one=np.zeros((8,8),complex);one[target,source]=gamma
            hop=hm.sector.bilinear(one).tocoo()
            for ell in range(3):
                coefficient=m.rot[:,ell,:]@(-2*np.cos(m.parameters['k']))
                yield hop,hm.g['O'][ell][axis].toarray(),phi,coefficient,disp


def external_action(m,state,external):
    """Rebuild original velocity minimal source independently, canonical Pi fixed."""
    x=state.reshape(m.shape);nf=m.nfield;nm=m.nm;R=m.R;ls=[a.toarray() for a in R['L']]
    out=(m.base@x.reshape(-1,nf)).reshape(m.shape)
    out+=np.einsum('...i,ji->...j',x,m.p2,optimize=True)/(2*m.parameters['cavity_inertia'])
    ext=fields.basis(m.parameters['field_total_cut']+1)
    embed=np.zeros((len(ext),nf),complex)
    for j,v in enumerate(m.basis):embed[ext.index(v),j]=1
    qimage=m.qe+external[:,None,None]*embed[None,:,:]
    qshort=np.asarray(m.q)+external[:,None,None]*np.eye(nf)[None,:,:]
    for f,conn in enumerate(m.connQ):
        block=np.kron(R['casimir'].toarray(),np.eye(nf)).astype(complex)
        for a in range(3):
            for l in range(8):
                coefficient=law.mult(R,conn[:,l,a]);block-=np.kron(ls[a]@coefficient+coefficient@ls[a],qshort[l])
        image=np.einsum('tla,lek->taek',conn,qimage,optimize=True)
        square=np.einsum('taei,taej->tij',image.conj(),image,optimize=True)
        block+=law.mult(R,square).transpose(0,1,3,2).reshape(nm*nf,nm*nf)
        out[f]+=(x[f].reshape(-1,nm*nf)@block.T).reshape(x[f].shape)/(2*m.parameters['material_inertia'])
    for hop,g,phi,coeff,disp in physical_hops(m):
        factor=np.exp(-1j*(phi@external))
        matrix=law.mult(R,(coeff*factor)[:,None,None]*disp).transpose(0,1,3,2).reshape(nm*nf,nm*nf)
        for reverse in (False,True):
            a=matrix.conj().T if reverse else matrix
            y=(x.reshape(-1,nm*nf)@a.T).reshape(m.shape)
            y=np.einsum('ij,fjrma->firma',g,y,optimize=True)
            rows,cols,values=(hop.col,hop.row,hop.data.conj()) if reverse else (hop.row,hop.col,hop.data)
            for row,col,value in zip(rows,cols,values):out[row]+=value*y[col]
    return out.ravel()


def observe(m,state):
    norm=float(np.vdot(state,state).real);x=state.reshape(m.shape)
    probabilities=np.sum(abs(x)**2,axis=(1,2,3,4))/norm
    charge=probabilities@m.Q
    spin=[]
    for L in m.matter.g['L']:
        image=np.einsum('ij,fjrma->firma',L.toarray(),x,optimize=True)
        spin.append(float(np.vdot(x,image).real/norm))
    current=[float(-np.vdot(state,m.vertex(state,a)).real/norm) for a in (0,4)]
    energy=float(np.vdot(state,m.apply(state)).real/norm)
    return dict(norm2=norm,energy=energy,physical_charge_density=charge.tolist(),g_LEFT=spin,physical_current_qin_qout=current,field_q=[float(np.vdot(state,m.field(state,a,'q')).real/norm) for a in (0,4)],field_pi=[float(np.vdot(state,m.field(state,a,'p')).real/norm) for a in (0,4)],field_shell_weight=float(sum(np.linalg.norm(x[...,i])**2 for i,v in enumerate(m.basis) if sum(v)==m.parameters['field_total_cut'])/norm))


def run():
    ref=law.Model(4);neutral=evolve(ref,ref.preparation(),.021)
    records={};arrays={'neutral_state':neutral}
    for number,create in ((5,True),(3,False)):
        m=law.Model(number);strengths=[];images=[]
        for component in range(8):
            image=m.source(ref,neutral,component,create);strengths.append(float(np.vdot(image,image).real));images.append(image)
        mu=int(np.argmax(strengths));state=images[mu]/np.sqrt(strengths[mu]);del images
        beforekick_energy=float(np.vdot(state,m.apply(state)).real)
        kick=expm(.17j*m.q[0]+.11j*m.q[4]);state=(state.reshape(-1,m.nfield)@kick.T).ravel()
        final=evolve(m,state,.037);nohop=evolve(m,state,.037,hopping=False);off=evolve(m,state,.037,hopping=False,motion=False)
        step=2e-4;unit=np.eye(8);zero=np.zeros(8)
        h0=external_action(m,state,zero);hp=external_action(m,state,step*unit[0]);hm=external_action(m,state,-step*unit[0])
        fd=(hp-hm)/(2*step);contact=(hp+hm-2*h0)/step**2
        v=m.vertex(state,0);c=m.vertex(state,0,0)
        corners=[external_action(m,state,step*(a*unit[0]+b*unit[4])) for a,b in ((1,1),(1,-1),(-1,1),(-1,-1))]
        mixed=(corners[0]-corners[1]-corners[2]+corners[3])/(4*step**2)
        vc=m.vertex(state,0,4)
        checks=dict(base_action_relative=float(np.linalg.norm(h0-m.apply(state))/max(np.linalg.norm(h0),1e-14)),physical_current_fd_relative=float(np.linalg.norm(fd-v)/max(np.linalg.norm(v),1e-14)),full_contact_fd_relative=float(np.linalg.norm(contact-c)/max(np.linalg.norm(c),1e-14)),mixed_contact_fd_relative=float(np.linalg.norm(mixed-vc)/max(np.linalg.norm(vc),1e-14)),hopping_OFF_convection_current_norm=float(np.linalg.norm(m.vertex(state,0,hopping=False))),whole_charge_motion_OFF_current_norm=float(np.linalg.norm(m.vertex(state,0,hopping=False,motion=False))),wrong_omitted_rotational_connection_current_error=float(np.linalg.norm(v-m.vertex(state,0,motion=False))),same_H_energy_drift=float(np.vdot(final,m.apply(final)).real-np.vdot(state,m.apply(state)).real),source_kick_work=float(np.vdot(state,m.apply(state)).real-beforekick_energy))
        assert checks['base_action_relative']<1e-11,checks
        assert checks['physical_current_fd_relative']<1e-6 and checks['full_contact_fd_relative']<1e-4 and checks['mixed_contact_fd_relative']<1e-4,checks
        assert checks['wrong_omitted_rotational_connection_current_error']>1e-5 and checks['whole_charge_motion_OFF_current_norm']==0,checks
        response_time=.013;response_state=evolve(m,state,response_time)
        in_image=evolve(m,m.vertex(state,0),response_time)
        out_image=evolve(m,m.vertex(state,4),response_time)
        order1=np.vdot(m.vertex(response_state,4),in_image)
        order2=np.vdot(m.vertex(response_state,0),out_image)
        checks['physical_current_time_orders']=[[float(order1.real),float(order1.imag)],[float(order2.real),float(order2.imag)]]
        checks['response_time']=response_time
        branch='addition' if create else 'removal'
        records[branch]=dict(number=number,dimension=m.dimension,source_strengths_all_bands=strengths,selected_component=mu,Gauss_error=m.matter.sector.gauss_error,initial=observe(m,state),final=observe(m,final),hopping_OFF=observe(m,nohop),whole_charge_motion_OFF=observe(m,off),state_feedback_norm=float(np.linalg.norm(final-off)),checks=checks)
        for name,array in dict(initial=state,final=final,hopping_OFF=nohop,whole_OFF=off,source_current=v,source_contact=c,mixed_contact=vc,fd_current=fd,fd_contact=contact).items():arrays[branch+'_'+name]=array
        print(branch,json.dumps(records[branch]),flush=True)
    np.savez_compressed(ROOT/(P+'spatial_consumer_raw.npz'),**arrays)
    result=dict(kind='OWN_FRESH_ACTIVE48_FULL_CAR8_QUANTUM_G_R_R_FIELD_CONSUMER',records=records,limits='Global g control, full CAR8 N3/4/5 and all8 source bands; not local47 or fullCAR32. Eight real quantum field coordinates, fixed velocity-canonical variation includes path and rotating ion+fermion source. Finite field shell1, angular1/1.5, material1 and 5radial nodes; no bulk photon, globalSO3 closure, mass or convergence claim.')
    (ROOT/(P+'spatial_consumer_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
