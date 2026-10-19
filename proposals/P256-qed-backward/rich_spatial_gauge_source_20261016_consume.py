"""New finite external-B experiment in declared spatial source completion."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from rich_local_frame_carrier_20261015_model import Model
from rich_dressed_local_carrier_20261015_source import cool, energy
from rich_review_wave31_20261015_krylov import propagate
from rich_spatial_gauge_source_20261016_family import SpatialFamily, quadrature, project

ROOT=Path(__file__).resolve().parent
P='rich_spatial_gauge_source_20261016_'


def pack(x):
    if isinstance(x,np.ndarray):
        return dict(real=x.real.tolist(),imag=x.imag.tolist()) if np.iscomplexobj(x) else x.tolist()
    if isinstance(x,np.generic):
        return x.item()
    if isinstance(x,complex):
        return dict(real=x.real,imag=x.imag)
    if isinstance(x,dict):
        return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [pack(v) for v in x]
    return x


def dump(name,value):
    (ROOT/(P+name+'.json')).write_text(json.dumps(pack(value),indent=2)+'\n')


def matrix(op,states):
    action=[op@s for s in states]
    return np.array([[np.vdot(a,b) for b in action] for a in states])


def filter_state(h,v,shift):
    states,gains,report=cool(h,v,[.35],40,30,shift)
    return states[0],float(np.exp(2*gains[0])),report


def main():
    start=time.monotonic(); config=json.loads((ROOT/(P+'input.json')).read_text())
    for name,expected in config['expected_input_SHA256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Changed frozen input '+name)
    interface=json.loads((ROOT/config['interface_file']).read_text())
    nm=Model(6,interface['parameters']); hn=nm.law()[0]
    with np.load(ROOT/interface['state_file']) as data:
        neutral=data['neutral'].copy()
    matched,_,neutral_filter=filter_state(hn,neutral,interface['filter_shift'])
    r,w,basis=quadrature(); gram=np.einsum('qi,q,qj->ij',basis,w,basis)
    result=dict(kind='NEW_EXTERNAL_SPATIAL_GAUGE_CURRENT_RESPONSE_NOT_ZEEMAN_EIGENSTATE',
        input=config,neutral_original=energy(hn,neutral),neutral_matched=energy(hn,matched),
        neutral_filter=neutral_filter,haar_gram_error=float(np.linalg.norm(gram-np.eye(10))),
        branches={},numerics=[])
    arrays={}
    for branch,number in [('addition',7),('removal',5)]:
        m=Model(number,interface['parameters']); h=m.law()[0]
        states=[]; weights=[]; reports=[]
        for spin in range(2):
            raw=m.source_from(nm,0,branch,component=spin,orbital=0)@matched
            state,weight,report=filter_state(h,raw,interface['filter_shift'])
            states.append(state);weights.append(weight);reports.append(report)
        lower=(m.J[0]-1j*m.J[1])@states[0]
        output=dict(dimension=m.dimension,charge=float(sum(np.vdot(states[0],q@states[0]).real for q in m.Q)),
            raw_filter_weights=weights,filter_reports=reports,prepared_energies=[energy(h,s) for s in states],
            lowering_overlap=complex(np.vdot(states[1],lower)),lowering_residual=float(np.linalg.norm(lower-np.vdot(states[1],lower)*states[1])),
            S=[matrix(a,states) for a in m.S],material_L=[matrix(a,states) for a in m.material_L],
            J=[matrix(a,states) for a in m.J],axes={})
        for axis,direction in zip('xyz',np.eye(3)):
            family=SpatialFamily(m,direction); dh,contact,hop,mechanical=family.derivatives()
            mu=-matrix(dh,states); k=matrix(contact,states)
            block=dict(moment=mu,hopping_moment=-matrix(hop,states),convective_moment=-matrix(mechanical,states),
                contact=k,wrong_compressed_contact=matrix(family.wrong_contact(),states),
                neutral_original_moment=None,neutral_matched_moment=None)
            for baseline,ref in [('original',neutral),('matched',matched)]:
                nf=SpatialFamily(nm,direction); ndh=nf.derivatives()[0]
                block['neutral_'+baseline+'_moment']=-float(np.vdot(ref,ndh@ref).real)
            # Every physical LEFT polarization, not scalar-cycle spin labels.
            polarizations={'z+':states[0],'z-':states[1],
                'x+':(states[0]+states[1])/np.sqrt(2),'x-':(states[0]-states[1])/np.sqrt(2),
                'y+':(states[0]+1j*states[1])/np.sqrt(2),'y-':(states[0]-1j*states[1])/np.sqrt(2)}
            block['polarizations']={label:dict(S=[float(np.vdot(v,a@v).real) for a in m.S],
                J=[float(np.vdot(v,a@v).real) for a in m.J],moment=-float(np.vdot(v,dh@v).real)) for label,v in polarizations.items()}
            if axis=='z':
                eps=1e-3; hp=family.law(eps); hm=family.law(-eps); v=states[0]
                first=(hp@v-hm@v)/(2*eps); second=(hp@v+hm@v-2*h@v)/eps**2
                block['own_H_first_derivative_error']=float(np.linalg.norm(first-dh@v))
                block['own_H_contact_error']=float(np.linalg.norm(second-contact@v))
                block['zero_H_action_error']=float(np.linalg.norm(family.law(0)@v-h@v))
                block['complete_vs_compressed_contact_action']=float(np.linalg.norm((contact@v)-(family.wrong_contact()@v)))
                block['prepared_energy_slopes']={str(field):(energy(family.law(field),v)['energy']-energy(h,v)['energy'])/field for field in [-.04,.04]}
                probe=family.law(.08)
                time_states,report=propagate(probe,v,[.18],24,18)
                free_states,free_report=propagate(h,v,[.18],24,18)
                block['finite_time_B08_t018']=dict(B_energy=energy(probe,time_states[0]),free_energy=energy(h,free_states[0]),
                    ray_distance=float(np.linalg.norm(time_states[0]-free_states[0])),
                    delta_S=[float(np.vdot(time_states[0],a@time_states[0]).real-np.vdot(free_states[0],a@free_states[0]).real) for a in m.S],
                    delta_material_L=[float(np.vdot(time_states[0],a@time_states[0]).real-np.vdot(free_states[0],a@free_states[0]).real) for a in m.material_L],
                    magnetic_moment_B_minus_free=-float(np.vdot(time_states[0],dh@time_states[0]).real-np.vdot(free_states[0],dh@free_states[0]).real),
                    propagation=report,free_propagation=free_report)
                arrays[branch+'_B_state']=time_states[0]
                # Constant A is pure gradient. Projected dipole commutator is
                # deliberately NOT substituted for full-coordinate source.
                const=SpatialFamily(m,direction,'constant'); cd=const.derivatives()[0]; dip=const.dipole()
                comm=1j*(dip@(h@v)-h@(dip@v))
                block['constant_A_fixed_projection_complement_norm']=float(np.linalg.norm(cd@v-comm))
                block['constant_A_direct_source_norm']=float(np.linalg.norm(cd@v))
                # Differential London identity at every quadrature point:
                # (-i X - a Lambda) exp(i a D) psi = exp(i a D)(-i X)psi.
                # For D=sum Q a.x, X D=Lambda; hopping endpoint phases cancel.
                q=m.sector.charge[0]; D=np.einsum('v,qva,a->q',q,const.x,direction)
                lam=np.einsum('v,qva->qa',q,const.connection)
                angle=.17; london=np.exp(1j*angle*D)
                derivative_errors=[]
                for a in np.eye(3):
                    step=1e-5
                    longitudinal=np.einsum('qva,a->qv',const.x,a)[:,:,None]*a*(1-np.cos(step))
                    xp=const.x*np.cos(step)+np.cross(a,const.x)*np.sin(step)+longitudinal
                    xm=const.x*np.cos(step)-np.cross(a,const.x)*np.sin(step)+longitudinal
                    dp=np.einsum('v,qva,a->q',q,xp,direction)
                    dm=np.einsum('v,qva,a->q',q,xm,direction)
                    numeric=(np.exp(1j*angle*dp)-np.exp(1j*angle*dm))/(2*step)
                    exact=1j*angle*(lam@a)*london
                    derivative_errors.append(float(np.max(abs(numeric-exact))))
                block['London_covariant_derivative_pointwise_errors']=derivative_errors
                endpoint=[]
                for edge,(i,j) in enumerate([(0,1),(1,2),(2,0)]):
                    delta=D+const.x[:,i]@direction-const.x[:,j]@direction
                    endpoint.append(float(np.max(abs(np.exp(1j*angle*delta)-london*np.exp(-1j*angle*const.phases[edge])))))
                block['London_hopping_endpoint_covariance_errors']=endpoint
            output['axes'][axis]=block
        result['branches'][branch]=output
        arrays[branch+'_prepared_up']=states[0];arrays[branch+'_prepared_down']=states[1]
        dump('partial_result',result)
        print(json.dumps(dict(branch=branch,elapsed=time.monotonic()-start)),flush=True)
        del m,h,states,family;gc.collect()
    result['elapsed_seconds']=time.monotonic()-start
    np.savez_compressed(ROOT/(P+'raw_arrays.npz'),**arrays)
    dump('result',result)
    print(json.dumps(dict(complete=True,elapsed=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    main()
