"""New physical consumer of exact original spatial actual2 retained law/state."""
import json
from pathlib import Path
import resource
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,expm_multiply
from rich_spatial_quantum_sea_20261015_model import geometry,fields,one_body,source_packet,source_operator


def norm(a):return float(sparse.linalg.norm(a) if sparse.issparse(a) else np.linalg.norm(a))


def main():
    inp,out=map(Path,sys.argv[1:3]);req=json.loads(inp.read_text());p=json.loads(Path(req['owner_input']).read_text());owner=json.loads(Path(req['owner_json']).read_text());data=np.load(req['owner_npz']);geo=geometry(p);field=fields(p);nb=len(field['coordinates']);u=data['orbitals'];geo['orbitals']=u;geo['transformed']=np.array([u.conj().T@a@u for a in geo['components']]);geo['onsite']=u.conj().T@(3*p['wilson']*geo['components'][0])@u;offset=float(np.sum(geo['reference_values'][:54]));sec={}
    for name,key in [('neutral','neutral_masks'),('positive','positive_masks'),('negative','negative_masks')]:
        masks=[int(x) for x in data[key]];ops=one_body(masks,np.concatenate([geo['onsite'][None],geo['transformed']]));sec[name]=dict(masks=masks,operators=ops)
    coefficients=np.vstack([np.ones(nb),field['coefficients']])
    def matter(name,vector,c):
        a=vector.reshape(len(sec[name]['masks']),nb);value=np.zeros_like(a,dtype=complex)
        for op,f in zip(sec[name]['operators'],c):value+=(op@a)*f
        return value.ravel()
    def law(name,c):
        n=len(sec[name]['masks']);dimension=n*nb
        def act(v):
            a=np.asarray(v).reshape(n,nb);return ((field['boson']@a.T).T-offset*a).ravel()+matter(name,v,c)
        def mm(v):return np.column_stack([act(x) for x in v.T])
        h=LinearOperator((dimension,dimension),matvec=act,rmatvec=act,matmat=mm,rmatmat=mm,dtype=complex)
        trace=n*field['boson'].diagonal().sum()-offset*dimension+sum(op.diagonal().sum()*f.sum() for op,f in zip(sec[name]['operators'],c))
        return h,trace
    hn,_=law('neutral',coefficients);h,tr=law('positive',coefficients);ground=data['neutral_ground'];en=owner['main']['energies'][0];binding=norm(hn@ground-en*ground)
    f=source_packet(geo,dict(p,packet_momentum=req['packet_momentum']));sp=source_operator(sec['neutral'],sec['positive'],f,True);sm=source_operator(sec['neutral'],sec['negative'],f,False);a0=ground.reshape(len(sec['neutral']['masks']),nb);raw=(sp@a0).ravel();hole=(sm@a0).ravel();initial=raw/norm(raw);occupation=one_body(sec['neutral']['masks'],np.outer(f,f.conj()))[0];raw_minus=float(np.vdot(a0,occupation@a0).real);raw_plus=1-raw_minus
    # n_x /55 is a complete physical POVM for sampling one actual fermion.
    density=one_body(sec['positive']['masks'],np.array([u[4*i:4*i+4].conj().T@u[4*i:4*i+4] for i in range(27)]))
    def observe(v):
        a=v.reshape(len(sec['positive']['masks']),nb);joint=np.array([np.sum(a.conj()*(op@a),axis=0).real/55 for op in density]);prob=np.sum(abs(a)**2,axis=0)
        current=[-float(np.vdot(v,matter('positive',v,np.vstack([np.zeros(nb),d]))).real) for d in field['derivative']]
        return dict(density=55*joint.sum(axis=1),joint=joint,field=prob,phi2=float(prob@(field['coordinates'][:,0]**2)),current=np.array(current))
    phase=np.exp(1j*req['kick']*(field['coordinates'][:,2]-.3*field['coordinates'][:,3]));kicked=(initial.reshape(len(sec['positive']['masks']),nb)*phase).ravel();first=observe(initial);records=[];states=[]
    for t in req['times']:
        a=expm_multiply(-1j*t*h,initial,traceA=-1j*t*tr);b=expm_multiply(-1j*t*h,kicked,traceA=-1j*t*tr);oa=observe(a);ob=observe(b);states.append(b);delta=oa['density']-first['density']
        records.append(dict(time=t,norms=[norm(a),norm(b)],particle_number=float(ob['density'].sum()),sampled_particle_field_POVM_sum=float(ob['joint'].sum()),sampled_particle_field_POVM_min=float(ob['joint'].min()),POVM_TV=float(abs(ob['joint']-oa['joint']).sum()/2),density_excess_l2=norm(ob['density']-oa['density']),field_probability_L1=float(abs(ob['field']-oa['field']).sum()),phi2_excess=ob['phi2']-oa['phi2'],current_excess=(ob['current']-oa['current']).tolist(),baseline_axis_motion=[norm(np.bincount(geo['cells'][:,axis],weights=delta,minlength=3)) for axis in range(3)]))
    # Independent external phase offsets, not a polynomial defined by stored J/K.
    def shifted(axis,delta):
        c=coefficients.copy();k=2
        for spatial in range(3):
            for level in (range(3) if spatial<2 else [0]):
                profile=p['charge']*(np.cos(2*np.pi*level/3) if axis%2==0 else np.sin(2*np.pi*level/3)) if spatial==axis//2 else 0.
                cs=np.cos(profile*delta);sn=np.sin(profile*delta);c[k]=coefficients[k]*cs-coefficients[k+1]*sn;c[k+1]=coefficients[k]*sn+coefficients[k+1]*cs;k+=2
        return c
    eps=1e-3;derivatives=[];contacts=[]
    for axis in range(4):
        hp=matter('positive',initial,shifted(axis,eps));hm=matter('positive',initial,shifted(axis,-eps));h0=matter('positive',initial,coefficients);j=-matter('positive',initial,np.vstack([np.zeros(nb),field['derivative'][axis]]));k=matter('positive',initial,np.vstack([np.zeros(nb),field['contact'][axis]]));derivatives.append(norm((hp-hm)/(2*eps)+j));contacts.append(norm((hp+hm-2*h0)/eps**2-k))
    center=int(np.argmin(np.sum((field['coordinates']-np.array([p['phi_center'],0,0,0,0]))**2,axis=1)));frozen=np.repeat(coefficients[:,center,None],nb,axis=1);hf,tf=law('positive',frozen);t=req['times'][-1];fa=observe(expm_multiply(-1j*t*hf,initial,traceA=-1j*t*tf));fb=observe(expm_multiply(-1j*t*hf,kicked,traceA=-1j*t*tf))
    result=dict(kind='INDEPENDENT_ORIGINAL_SPATIAL_ACTUAL2_NEW_SOURCE_AND_FIELD_EVENT',request=req,dimensions={n:len(s['masks'])*nb for n,s in sec.items()},literal_saved_orbital_frame='Reused exact owner orbital bytes, not an independently rotated degenerate eigbasis.',neutral_same_state_retained_eigen_residual=binding,raw_CAR=dict(creation=raw_plus,removal=raw_minus,sum=raw_plus+raw_minus),retained_source=dict(creation=norm(raw)**2,removal=norm(hole)**2,lost_creation=raw_plus-norm(raw)**2,lost_removal=raw_minus-norm(hole)**2),physical_charge_of_creation=p['charge'],source_field_preparation='First normalize explicitly retained projected CAR source. Then exact field-only diagonal unitary, no source re-projection/conditioning or renormalization.',density_number_identity_error=norm(sum(density)-55*sparse.eye(len(sec['positive']['masks']))),current_phase_derivative_errors=derivatives,contact_phase_derivative_errors=contacts,samples=records,frozen_sameinitial_density_excess=norm(fb['density']-fa['density']),frozen_physical_field_source_current='ZERO for the explicitly field-independent frozen-H law; full-law J diagnostic not substituted.',owner_omitted_H_norm_not_recomputed=owner['main']['neutral_projection']['absolute_H_projection_residual'],peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits='Full108 modes/54 occupied reference, finite fixed determinant selection71/329/324. Small retained eigen residual does NOT bound missing-H/source errors; source CAR raw sum1 is NOT retained source sum. Signed vacuum-subtracted charge excess not probability. Sampled-fermion n/55 joint POVM not added-particle-position PVM. Same severe field boxes/zero nonuniform longitudinal/supplied scalar order and internalframe inputs; not allpair/physicalNoEther/bulkphoton/COM.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),initial=initial,kicked=kicked,states=np.array(states));print(json.dumps(result))


if __name__=='__main__':main()
