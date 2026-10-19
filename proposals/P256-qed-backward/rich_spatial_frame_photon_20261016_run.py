"""Fresh neutral/charged spatial experiment; no inherited state or solved output."""
import gc
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from scipy.linalg import expm
import rich_spatial_frame_photon_20261016_model as m
import rich_spatial_frame_photon_20261016_fields as fields
import rich_spatial_frame_photon_20261016_kernel as kernel
ROOT=Path(__file__).resolve().parent
P='rich_spatial_frame_photon_20261016_'


def evolve(apply,state,t,steps=16):
    """Hermitian Lanczos exponential, actual terminal residual recorded."""
    norm=np.linalg.norm(state)
    if norm<1.e-15:return state.copy(),dict(krylov=0,residual=0.)
    q=state/norm;old=np.zeros_like(q);beta=0.;b=0.;vectors=[];alpha=[];betas=[]
    for i in range(steps):
        vectors.append(q);w=apply(q)-beta*old;a=float(np.vdot(q,w).real);alpha.append(a);w-=a*q
        # Two-pass full orthogonalization: no unreported Lanczos ghost states.
        for _ in range(2):
            for v in vectors:w-=np.vdot(v,w)*v
        b=float(np.linalg.norm(w))
        if i==steps-1 or b<1.e-13:break
        betas.append(b);old=q;q=w/b;beta=b
    tr=np.diag(alpha).astype(complex)
    if betas:tr+=np.diag(betas,1)+np.diag(betas,-1)
    coeff=expm(-1j*t*tr)[:,0]
    out=sum(c*v for c,v in zip(coeff,vectors))*norm
    return out,dict(krylov=len(vectors),terminal_residual=float(abs(b*coeff[-1])*norm),norm_defect=float(np.linalg.norm(out)-norm))


def expectation(v,w):return float(np.vdot(v,w).real)


def measure(model,v,hopping=True,motion=True):
    x=v.reshape(model.shape);hm=model.matter
    prob=np.sum(abs(x)**2,axis=(1,2,3,4));rprob=np.sum(abs(x)**2,axis=(0,1,3,4))
    electric=[expectation(v,model.field(v,a)) for a in range(8)]
    spin=[];spin2=0.
    for a in hm.g['L']:
        w=np.einsum('ij,fjrma->firma',a.toarray(),x,optimize=True)
        spin.append(expectation(v,w.reshape(-1)));spin2+=float(np.linalg.norm(w)**2)
    qmean=[expectation(v,model.field(v,a,'q')) for a in range(8)]
    receiver=kernel.profiles(np.array([[0.,0.,0.],[.8,-.4,1.1]]),model.modes)
    current=[-expectation(v,model.vertex(v,a,hopping=hopping,motion=motion)) for a in (0,2,4,6)]
    return dict(norm=float(np.linalg.norm(v)**2),energy=expectation(v,model.apply(v,hopping,motion)),r=float(rprob@hm.rnodes),density=(prob@hm.sector.n).tolist(),physical_charge=float(prob@model.Q.sum(axis=1)),spin=spin,spin2=spin2,electric=electric,field_coordinate=qmean,receiver_electric=np.einsum('sla,l->sa',receiver,electric).tolist(),physical_current=current,field_top_weight=float(sum(np.linalg.norm(x[...,i])**2 for i,n in enumerate(model.basis) if sum(n)==model.parameters['field_total_cut'])),g_top_weight=float(sum(np.linalg.norm(x[:,i])**2 for i,z in enumerate(hm.g['labels']) if z[0]==max(y[0] for y in hm.g['labels']))),R_top_weight=float(sum(np.linalg.norm(x[:,:,:,i])**2 for i,z in enumerate(model.R['labels']) if z[0]==max(y[0] for y in model.R['labels']))))


def kick(model,v,axis,eps):
    # Finite-basis unitary is exactly field-only, with continuum comparator separately.
    unitary=expm(1j*eps*model.q[axis]);out=(v.reshape(model.shape)@unitary.T).reshape(-1)
    c=np.zeros(8);c[axis]=eps
    exact=(v.reshape(model.shape)@fields.displacement(model.basis,model.basis,c).T).reshape(-1)
    return out,dict(type='unitary exp(i eps P a P), finite field regulator',continuum_compression_difference=float(np.linalg.norm(out-exact)),continuum_discarded_probability=float(np.linalg.norm(v)**2-np.linalg.norm(exact)**2))


def main():
    label=sys.argv[1] if len(sys.argv)>1 else 'actual1';p=dict(m.DEFAULTS)
    if label=='field2':p['field_total_cut']=2
    if label=='R2':p['material_level']=2
    t=.08 if label=='actual1' else .04
    result=dict(kind='ACTUAL_NEW_FINITE_Q_FULL_QUANTUM_MINIMAL_VELOCITY',parameters=p,t=t,modes=kernel.default_modes().tolist(),charge='Q=ions-fermions; addition charge -1, removal +1',field_regulator='unbounded Hermite primitives/exponentials before projection; finite-basis unitary kick explicitly distinguished',source='radial-root path exp(-i a F_s) times B40; source products precede cutoff')
    reference=m.Model(4,p);neutral=reference.preparation();neutral_end,err=evolve(reference.apply,neutral,t)
    np.savez_compressed(ROOT/(P+label+'_neutral_raw.npz'),input=neutral,end=neutral_end)
    result['neutral']=dict(initial=measure(reference,neutral),end=measure(reference,neutral_end),evolution=err,dimension=reference.dimension)
    result['branches']={}
    for number,create,name,mu in [(5,True,'addition',0),(3,False,'removal',1)]:
        model=m.Model(number,p);raw={}
        packets=[model.source(reference,neutral,a,create) for a in range(8)]
        weights=[float(np.linalg.norm(v)**2) for v in packets];v=packets[mu]/np.sqrt(weights[mu]);del packets
        pulse,kickinfo=kick(model,v,0,.23)
        end,e0=evolve(model.apply,v,t);pulsed,e1=evolve(model.apply,pulse,t)
        raw.update(input=v,kicked_input=pulse,end=end,kicked_end=pulsed)
        a0=measure(model,v);a1=measure(model,end);b0=measure(model,pulse);b1=measure(model,pulsed)
        # Fixed-canonical current/electric/spin legs for qin and nonparallel qout.
        legs=[-model.vertex(v,0),-model.vertex(v,4),model.field(v,0),model.field(v,4)]
        propagated=[];leg_errors=[]
        for leg in legs:
            w,e=evolve(model.apply,leg,t);propagated.append(w);leg_errors.append(e)
        left=[-model.vertex(end,0),-model.vertex(end,4),model.field(end,0),model.field(end,4)]
        C=np.array([[np.vdot(l,r) for r in propagated] for l in left],dtype=np.complex128)
        K=np.array([[np.vdot(v,model.vertex(v,a,b)) for b in (0,4)] for a in (0,4)],dtype=np.complex128)
        raw.update(two_leg=C,reverse_timeorder=C.conj(),retarded=2*C.imag,full_contact=K)
        # Every control uses its OWN Hamiltonian and matching canonical observable.
        offbase,eo0=evolve(lambda z:model.apply(z,False,True),v,t)
        offpulse,eo1=evolve(lambda z:model.apply(z,False,True),pulse,t)
        barebase,eb0=evolve(lambda z:model.apply(z,False,False),v,t)
        barepulse,eb1=evolve(lambda z:model.apply(z,False,False),pulse,t)
        raw.update(hopping_off_base=offbase,hopping_off_pulse=offpulse,charge_motion_off_base=barebase,charge_motion_off_pulse=barepulse)
        # Actual source-H mismatch, with source connection and all canonical field terms.
        bv=model.source(reference,reference.apply(neutral),mu,create)
        connections=dict(full_H=float(np.linalg.norm(model.apply(v)*np.sqrt(weights[mu])-bv)))
        # Independent physical primitive components: tree E, g centrifugal, r force, R/source.
        x=v.reshape(model.shape);qprob=np.sum(abs(x)**2,axis=(1,2,3,4));rprob=np.sum(abs(x)**2,axis=(0,1,3,4))
        force=-p['radial_stiffness']*(model.matter.rnodes-p['radial_centre'])+3/(4*p['radial_mass']*model.matter.rnodes**3)
        force_mean=float(rprob@force)
        for a in model.matter.g['L']:
            av=np.einsum('ij,fjrma->firma',a.toarray(),x,optimize=True)
            force_mean+=float(np.sum(abs(av)**2/model.matter.rnodes[None,None,:,None,None]**3))/p['radial_mass']
        beta_diag=np.real(np.diag(model.matter.geo['beta']));force_mean-=p['root_coupling']*float(qprob@(model.matter.sector.n@beta_diag))
        # Real closed square Wilson loop, which a uniform-only rephasing sets to zero.
        loop=model.x[:,[0,4,6,2,0],:];flux=kernel.path_integrals(loop,model.modes)
        W=fields.displacement(model.basis,model.basis,-flux)
        loopop=m.mult(model.R,W).transpose(0,1,3,2).reshape(model.nm*model.nfield,-1)
        loopv=(x.reshape(-1,model.nm*model.nfield)@loopop.T).reshape(-1)
        geometric=dict(loop_flux_rms=float(np.sqrt(np.sum(model.R['weights'][:,None]*flux**2))),actual_Wilson_real=float(np.vdot(v,loopv).real),wrong_uniform_Wilson_real=1.,physical_Wilson_defect=float(np.linalg.norm(loopv-v)),residual_connection_rms=float(np.sqrt(np.mean([np.linalg.norm(g['residual_connection'])**2 for g in model.geom]))))
        branch: dict[str,Any]
        branch=dict(number=number,dimension=model.dimension,source_component=mu,source_weights=weights,source_weight_sum=sum(weights),source_energy_cost=a0['energy']-result['neutral']['initial']['energy'],source_connections=connections,source_radial_force=force_mean,baseline=[a0,a1],field_kick=[b0,b1],kick_info=kickinfo,two_leg_labels=['JinT1cos','JoutT1cos','EinT1cos','EoutT1cos'],two_leg_real=C.real.tolist(),two_leg_imag=C.imag.tolist(),retarded=(2*C.imag).tolist(),contact_real=K.real.tolist(),contact_imag=K.imag.tolist(),evolution=[e0,e1,*leg_errors,eo0,eo1,eb0,eb1],hopping_off=[measure(model,offbase,False,True),measure(model,offpulse,False,True)],charge_motion_off=[measure(model,barebase,False,False),measure(model,barepulse,False,False)],geometry=geometric)
        branch['kick_feedback']={key:(np.asarray(b1[key])-np.asarray(a1[key])).tolist() for key in ['r','density','spin','spin2','electric','receiver_electric','physical_current']}
        branch['spin_current_order']=dict(initial_spin_current=[np.vdot(np.einsum('ij,fjrma->firma',model.matter.g['L'][2].toarray(),x,optimize=True).reshape(-1),legs[1]).real.item()])
        raw['source_H_image']=model.apply(v)*np.sqrt(weights[mu])-bv
        result['branches'][name]=branch
        np.savez_compressed(ROOT/(P+label+'_'+name+'_raw.npz'),**raw)
        (ROOT/(P+label+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(name,json.dumps(dict(dimension=model.dimension,energy=a0['energy'],out_feedback=branch['kick_feedback']['electric'][4],loop=geometric,residual=max(e['terminal_residual'] for e in branch['evolution']))),flush=True)
        del model,v,pulse,end,pulsed,legs,left,propagated,raw;gc.collect()
    print(json.dumps(dict(label=label,complete=True)),flush=True)


if __name__=='__main__':main()
