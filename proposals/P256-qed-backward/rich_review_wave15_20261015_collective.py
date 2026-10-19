"""Wave15 same-law Gaussian/CAR/Gram/H2 and NEW physical packet consumer."""
import json
from pathlib import Path
import resource
import sys
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply
from rich_spatial_collective_20261015_model import (build,packet,source_matrices,source_state,
    receiver_matrices,fermionic_leakage,install_operator,whiten,exterior_basis,determinant_vectors,second_quantized)


def norm(a):return float(sparse.linalg.norm(a) if sparse.issparse(a) else np.linalg.norm(a))
def mean(v,a):return float(np.vdot(v,a@v).real)


def physical_observe(model,state,density):
    sec=model['positive'];a=state.reshape(sum(sec['sizes']),-1);prob=np.sum(abs(a)**2,axis=0)
    joint=np.array([np.sum(a.conj()*(op@a),axis=0).real/55 for op in density])
    currents=[]
    for derivative in model['field']['derivative']:
        total=0.
        for b,n in enumerate(sec['sizes']):
            c=state[sec['offsets'][b]:sec['offsets'][b+1]].reshape(n,-1);action=np.zeros_like(c)
            for f,d in zip(sec['matrices'][b],np.vstack([np.zeros(len(prob)),derivative])):action-=(f@c)*d
            total+=float(np.vdot(c,action).real)
        currents.append(total)
    return dict(norm=norm(state),sampled_particle_site_field_POVM=joint,number_density=55*joint.sum(axis=1),field_probability=prob,phi2=float(prob@(model['field']['coordinates'][:,0]**2)),currents=currents)


def particle_hole_matrix(n):
    _,masks,_=exterior_basis(n);_,target,lookup=exterior_basis(12-n);rows=[];values=[]
    for mask in masks:
        occupied=[i for i in range(12) if mask&(1<<i)];full=(1<<12)-1;sign=1
        for i in reversed(occupied):
            local=i%4;m=local//2;s=local%2;j=4*(i//4)+(local^3)
            sign*=(-1 if m==s else 1)*(-1)**((full&((1<<j)-1)).bit_count());full^=1<<j
        rows.append(lookup[full]);values.append(sign)
    return sparse.csr_matrix((values,(rows,range(len(masks)))),shape=(len(target),len(masks)))


def conjugation_audit(model,initial,plus,source_packet):
    blocks=model['blocks'];index={k:i for i,k in enumerate(model['momenta'])};partner=[index[((-k)%3,(-l)%3)] for k,l in model['momenta']];g=model['gindex'];c6=particle_hole_matrix(6);c7=particle_hole_matrix(7)
    local=[b['v0'].conj().T@(c6@blocks[r]['v0']) for b,r in zip(blocks,partner)]
    raw=np.ones_like(local[0])
    for c in local:raw*=c
    x=model['neutral']['whiteners'][0];cn=x.conj().T@raw@x;charged=[]
    for b,r in enumerate(partner):
        factor=np.ones_like(raw)
        for i,c in enumerate(local):
            if i!=b:factor*=c
        rawc=blocks[b]['vm'].conj().T@(c7@blocks[r]['vp']);rawc*=factor[g[:,None],g[None,:]]
        charged.append(model['negative']['whiteners'][b].conj().T@rawc@model['positive']['whiteners'][r])
    coordinates=model['field']['coordinates'];lookup={tuple(np.round(q,10)):i for i,q in enumerate(coordinates)};reflection=np.array([lookup[tuple(np.round(np.r_[q[0],-q[1:]],10))] for q in coordinates]);nb=len(coordinates)
    def cneutral(v):return (cn@v.reshape(len(cn),nb)[:,reflection]).ravel()
    def cplus(v):
        sec=model['positive'];out=[]
        for b,r in enumerate(partner):out.append((charged[b]@v[sec['offsets'][r]:sec['offsets'][r+1]].reshape(sec['sizes'][r],nb)[:,reflection]).ravel())
        return np.concatenate(out)
    rng=np.random.default_rng(15);test=rng.normal(size=len(plus))+1j*rng.normal(size=len(plus));test/=norm(test)
    y=np.array([[0,-1j],[1j,0]]);u=np.kron(y,y);u12=np.kron(np.eye(3),u);conjugate_packet=[u12@source_packet[r].conj() for r in partner]
    sm=source_matrices(model,-1,conjugate_packet);neutral_c=cneutral(model['neutral']['ground']);neutral=model['neutral'];old=neutral['ground'];neutral['ground']=neutral_c;direct=source_state(model,-1,sm);neutral['ground']=old
    returned=cplus(plus)
    return dict(neutral_C_unitarity=norm(cn.conj().T@cn-np.eye(len(cn))),charged_C_unitarity=float(np.sqrt(sum(norm(c.conj().T@c-np.eye(c.shape[1]))**2 for c in charged))),whole_H_C_relative=norm(model['negative']['h']@cplus(test)-cplus(model['positive']['h']@test))/max(1.,norm(model['positive']['h']@test)),direct_CAR_source_intertwiner=norm(returned-direct),direct_CAR_source_raw_strength=norm(direct)**2,C_defined_raw_strength=norm(returned)**2,meaning='Actual occupied N53 holes from canonical particle-hole action, k->-k and four EM-coordinate reflection. No empty-valence copy; phase/sign map derives exact local exterior CAR.')


def local_product_controls(model):
    # Brute two-block 924^2 Fock vector is a working positive control for exact
    # overlap, one-body and H2 cross terms; no inverse Slater overlap.
    b0,b1=model['blocks'][:2];q=model['center'];h=[]
    for b in [b0,b1]:h.append(sum(c*op for c,op in zip(model['coefficients'][:,q],b['operators'][6])))
    v=[b['v0'][:,:2] for b in [b0,b1]];full=np.column_stack([np.kron(v[0][:,i],v[1][:,i]) for i in range(2)])
    image=[]
    for i in range(2):
        a=full[:,i].reshape(924,924);image.append((h[0]@a+(h[1]@a.T).T).ravel())
    image=np.column_stack(image);grams=[a.conj().T@a for a in v];first=[a.conj().T@(op@a) for a,op in zip(v,h)];square=[(op@a).conj().T@(op@a) for a,op in zip(v,h)]
    h2=square[0]*grams[1]+grams[0]*square[1]+2*first[0]*first[1]
    duplicate=np.ones((2,2));x,info=whiten(duplicate,1e-10)
    # Make actually orthogonal Slaters joined by a nonzero current matrix element.
    raw=model['local_components'][0];ref=sum(c*a for c,a in zip(model['coefficients'][:,q],raw));_,u=eigh(ref);occ=u[:,:6];empty=u[:,6:]
    current=-sum(c*a for c,a in zip(np.r_[0.,model['field']['derivative'][0,:,q]],raw));transition=empty.conj().T@current@occ;p,r=np.unravel_index(np.argmax(abs(transition)),transition.shape);other=occ.copy();other[:,r]=empty[:,p]
    dv=determinant_vectors([occ,other],exterior_basis(6)[0]);op=second_quantized(current,exterior_basis(6)[1],exterior_basis(6)[2]);element=dv[:,0].conj()@(op@dv[:,1])
    return dict(two_block_exact_Gram_error=norm(full.conj().T@full-grams[0]*grams[1]),two_block_H_matrix_error=norm(full.conj().T@image-first[0]*grams[1]-grams[0]*first[1]),two_block_H2_matrix_error=norm(image.conj().T@image-h2),wrong_drop_cross_H2_error=norm(image.conj().T@image-square[0]*grams[1]-grams[0]*square[1]),singular_Gram=info,singular_Gram_whitened_norm=norm(x.conj().T@duplicate@x-np.eye(1)),orthogonal_Slater_overlap=float(abs(np.vdot(dv[:,0],dv[:,1]))),finite_current_between_orthogonal_Slaters=float(abs(element)),meaning='Literal exterior-Fock positive controls reject dropping H2 cross terms or treating singular raw Gaussian coefficients as orthonormal probabilities.')


def main():
    inp,out=map(Path,sys.argv[1:3]);req=json.loads(inp.read_text());config=json.loads(Path(req['owner_input']).read_text());owner=json.loads(Path(req['owner_json']).read_text());data=np.load(req['owner_npz']);model=build(config);neutral=model['neutral'];neutral['ground']=data['neutral_ground'];neutral['energy']=owner['energies']['neutral']
    print(json.dumps(dict(stage='model',dimensions={n:model[n]['h'].shape[0] for n in ['neutral','positive','negative']})),flush=True)
    binding=norm(neutral['h']@neutral['ground']-neutral['energy']*neutral['ground']);source_packet=packet(model,dict(config,packet_momentum=req['packet_momentum']));sm=source_matrices(model,1,source_packet);plus=source_state(model,1,sm);initial=plus/norm(plus)
    caudit=conjugation_audit(model,initial,plus,source_packet);controls=local_product_controls(model);residual=fermionic_leakage(model)
    density=receiver_matrices(model);sec=model['positive'];coordinates=model['field']['coordinates'];phase=np.exp(1j*req['kick']*(coordinates[:,1]+.4*coordinates[:,4]));kicked=(initial.reshape(sum(sec['sizes']),-1)*phase).ravel()
    records=[];states=[];first=physical_observe(model,initial,density)
    for t in req['times']:
        a=expm_multiply(-1j*t*sec['h'],initial,traceA=-1j*t*sec['trace']);b=expm_multiply(-1j*t*sec['h'],kicked,traceA=-1j*t*sec['trace']);oa=physical_observe(model,a,density);ob=physical_observe(model,b,density);states.append(b);delta=oa['number_density']-first['number_density']
        # Deliberately remove inter-k density coherences: x/y marginals then cannot move.
        aa=a.reshape(sum(sec['sizes']),-1);dephased=[];bound=np.cumsum([0]+sec['sizes'])
        for op in density:
            dephased.append(sum(float(np.vdot(aa[bound[i]:bound[i+1]],op[bound[i]:bound[i+1],bound[i]:bound[i+1]]@aa[bound[i]:bound[i+1]]).real) for i in range(9)))
        records.append(dict(time=t,norms=[oa['norm'],ob['norm']],number_sum=float(ob['number_density'].sum()),sampled_particle_field_POVM_sum=float(ob['sampled_particle_site_field_POVM'].sum()),sampled_particle_field_POVM_min=float(ob['sampled_particle_site_field_POVM'].min()),joint_POVM_TV=float(abs(ob['sampled_particle_site_field_POVM']-oa['sampled_particle_site_field_POVM']).sum()/2),density_excess_l2=norm(ob['number_density']-oa['number_density']),field_probability_L1=float(abs(ob['field_probability']-oa['field_probability']).sum()),phi2_excess=ob['phi2']-oa['phi2'],current_excess=(np.array(ob['currents'])-oa['currents']).tolist(),baseline_axis_motion=[norm(np.bincount(model['geo']['cells'][:,axis],weights=delta,minlength=3)) for axis in range(3)],dephased_xy_spread=[float(np.ptp(np.bincount(model['geo']['cells'][:,axis],weights=dephased,minlength=3))) for axis in range(2)]))
    frozen_coeff=np.repeat(model['coefficients'][:,model['center'],None],len(coordinates),axis=1);install_operator(sec,model,frozen_coeff);t=req['times'][-1];a=expm_multiply(-1j*t*sec['h'],initial,traceA=-1j*t*sec['trace']);b=expm_multiply(-1j*t*sec['h'],kicked,traceA=-1j*t*sec['trace']);fa=physical_observe(model,a,density);fb=physical_observe(model,b,density)
    result=dict(kind='NEW_WAVE15_SAME_PARAMETER_COLLECTIVE_PHYSICAL_CONSUMER_NOT_ALLPAIR_CLOSURE',request=req,neutral_same_state_eigen_residual=binding,source_raw_norm=norm(plus)**2,source_C=caudit,exact_exterior_controls=controls,independent_same_state_H2_leakage=residual,gram_max_defect=max(i['whitened_gram_defect'] for name in ['neutral','positive','negative'] for i in model[name]['gram_info']),samples=records,frozen_sameinitial_control=dict(density_excess_l2=norm(fb['number_density']-fa['number_density']),phi2_excess=fb['phi2']-fa['phi2'],physical_field_source_current='ZERO for the explicitly field-independent frozen-H law; original-law current matrices only diagnostic, not substituted.'),peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits='Same108 CAR/54 sea/405 field coordinates,11 Gaussian collective span. Exact local all-pair coherent Slaters NOT complete arbitrary allpair dynamics. H2 residual neutral only, variational gaps not bounds. Sampled-particle POVM n_site/55 is a uniform draw from55 actual fermions, NOT source-position PVM; signed vacuum-subtracted excess not probability. Singleq/internalframe/zero longitudinal/finitefield inputs unchanged, no full NoEther/photons/COM claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),initial=initial,kicked=kicked,states=np.array(states));print(json.dumps(dict(stage='complete',result=result)),flush=True)


if __name__=='__main__':main()
