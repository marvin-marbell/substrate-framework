"""Wave15 NEW full depth2/E4/scalar17 source/event audit of frozen construct1."""
import json
from pathlib import Path
import resource
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply,gmres,LinearOperator
from rich_quantum_common_field_20261015_model import rotor,operators,law,compressed,energy_projection_residual,conjugate_sector,neighbours
from rich_quantum_common_field_20261015_sources import compressed_source
from rich_quantum_common_field_20261015_boundary import full_source_strength,omitted_population
from rich_common_colour_lepton_20261015_model import scalar_field,gauss_operators
from rich_colour_composite_20261014_model import loop_field
from rich_quantum_common_field_20261015_calculate import neutral_C


def norm(a):return float(sparse.linalg.norm(a) if sparse.issparse(a) else np.linalg.norm(a))
def mean(v,a):return float(np.vdot(v,a@v).real)
def load(data,key):return sparse.csr_matrix((data[key+'_data'],data[key+'_indices'],data[key+'_indptr']),shape=tuple(data[key+'_shape']))


def reconstruct(data,name,count,scalar,loop,p):
    blocks=[];offset=0
    for i in range(count):
        key=name+'_block'+str(i);masks=data[key+'_masks'].tolist();projection=load(data,key+'_P');q,g=gauss_operators(masks,loop);sector=dict(masks=masks,P=projection,Q=q,G=g,dimension=projection.shape[1]);op=operators(sector,loop);v=data[key+'_V'];dim=v.shape[1]
        blocks.append(dict(sector=sector,op=op,H0=law(op,scalar,p)[0],V=v,offset=offset,dimension=dim,signature=tuple(data[key+'_signature'])))
        offset+=dim
    signatures=[b['signature'] for b in blocks];omitted=sorted(set().union(*(neighbours(s) for s in signatures))-set(signatures))
    return dict(blocks=blocks,dimension=offset,omitted=omitted,anti=False)


def receiver(sector,state,scalar,field):
    size=len(field['electric']);nf=len(scalar['grid']);prob=np.zeros((2,2,3,3));bins=[field['electric']<0,field['electric']==0,field['electric']>0]
    for b in sector['blocks']:
        a=state[b['offset']*field['size']:(b['offset']+b['dimension'])*field['size']].reshape(b['dimension'],field['size']);full=(b['V']@a).reshape(b['sector']['dimension'],nf,size,size);ec,u=np.linalg.eigh(b['op']['C']);full=np.einsum('ab,bfxy->afxy',u.conj().T,full)
        for i,j,k,l in np.ndindex(prob.shape):prob[i,j,k,l]+=norm(full[np.ix_(ec<1e-8 if i==0 else ec>=1e-8,scalar['grid']<0 if j==0 else scalar['grid']>=0,bins[k],bins[l])])**2
    return prob


def field_shift_audit(h,state,field,matter):
    numbers=field['electric'];size=len(numbers);zero=int(np.flatnonzero(numbers==0)[0]);center=zero*size+zero;nb=field['size'];n=field['E'];ident=sparse.eye(matter,format='csr');rows=np.arange(matter)*nb;coefficient={};deltas=sorted({(sign*k,0) for k in [1,2,3] for sign in [-1,1]}|{(0,sign*k) for k in [1,2,3] for sign in [-1,1]}|{(sign*k,sign*k) for k in [1,2,3] for sign in [-1,1]})
    m0=h[rows+center][:,rows+center];rebuilt=sparse.kron(m0,field['identity'])+sparse.kron(ident,field['H']);big=np.arange(numbers[0]-3,numbers[-1]+4);index={int(x):i for i,x in enumerate(big)};bigsize=len(big);outside=np.zeros((matter,bigsize*bigsize),complex);a=state.reshape(matter,nb);inside=[]
    for x in numbers:
        for y in numbers:inside.append(index[int(x)]*bigsize+index[int(y)])
    for dx,dy in deltas:
        ix=int(np.flatnonzero(numbers==dx)[0]);iy=int(np.flatnonzero(numbers==dy)[0]);c=h[rows+ix*size+iy][:,rows+center];coefficient[dx,dy]=c
        rr=[];cc=[];targets=[]
        for q,(x,y) in enumerate(( (x,y) for x in numbers for y in numbers)):
            tx=int(x)+dx;ty=int(y)+dy;targets.append(index[tx]*bigsize+index[ty])
            if tx in numbers and ty in numbers:rr.append(int(np.flatnonzero(numbers==tx)[0])*size+int(np.flatnonzero(numbers==ty)[0]));cc.append(q)
        shift=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(nb,nb));rebuilt+=sparse.kron(c,shift);outside[:,targets]+=c@a
    outside[:,inside]=0
    electric=[np.tile(op.diagonal().real,matter)/3 for op in n];j=[];k=[]
    for number in electric:
        diagonal=sparse.diags(number);j.append(1j*(h@diagonal-diagonal@h));k.append(-(diagonal@diagonal@h-2*diagonal@h@diagonal+h@diagonal@diagonal))
    nx,ny=[sparse.diags(a) for a in electric];mixed=-(nx@ny@h-nx@h@ny-ny@h@nx+h@nx@ny)
    return dict(full_H_shift_rebuild_error=norm(rebuilt-h),actual_electric_omitted_H_norm=norm(outside),extended_cut=int(big[-1]),harmonic_shift_coefficient_norms={str(d):norm(c) for d,c in coefficient.items()},source_units='theta has integer n, a=3theta, so physical current uses n/3; replacing n/3 by n multiplies J by3.'),j,k,mixed


def solve(matrix,rhs):
    diagonal=matrix.diagonal();pre=LinearOperator(matrix.shape,matvec=lambda x:x/diagonal,dtype=complex);value,info=gmres(matrix,rhs,M=pre,rtol=1e-10,atol=1e-12,restart=100,maxiter=1000)
    return value,dict(info=int(info),residual=norm(matrix@value-rhs))


def main():
    inp,out=map(Path,sys.argv[1:3]);req=json.loads(inp.read_text());owner=json.loads(Path(req['owner_json']).read_text());p=owner['parameters'];data=np.load(req['owner_npz']);scalar=scalar_field(p);loop=loop_field();field=rotor(p);neutral=reconstruct(data,'neutral',owner['results']['neutral']['retained_gauge_blocks'],scalar,loop,p);ground=data['neutral_stationary'];hn=load(data,'neutral_H');en=owner['results']['neutral']['ground_energy'];reflection=sparse.csr_matrix(np.fliplr(np.eye(field['size'])));cn=sparse.kron(neutral_C(neutral,scalar),reflection,format='csr');size=len(field['electric']);u=sparse.diags(np.ones(size-1),-1,shape=(size,size));ux=sparse.kron(u,sparse.eye(size));uy=sparse.kron(sparse.eye(size),u);generator=(ux+ux.conj().T)/2+.4*(uy-uy.conj().T)/(2j);results={};stored={}
    root_response=[];neutral_identity=sparse.eye(hn.shape[0]);jneutral=[load(data,'neutral_J'+str(a)) for a in range(2)];kneut=[load(data,'neutral_K'+str(a)) for a in range(2)]
    for frequency in req['response_frequencies']:
        terms=[];errors=[]
        for sign in [-1,1]:
            columns=[];matrix=(frequency+1j*p['eta'])*neutral_identity+sign*(hn-en*neutral_identity)
            for j in jneutral:
                x,err=solve(matrix,j@ground);columns.append(x);errors.append(err)
            terms.append(np.column_stack([j@ground for j in jneutral]).conj().T@np.column_stack(columns))
        chi=terms[0]-terms[1].T;root_response.append(dict(frequency=frequency,paramagnetic_real=chi.real.tolist(),paramagnetic_imag=chi.imag.tolist(),diagonal_contacts=[mean(ground,k) for k in kneut],solver=errors,all_retained_virtuals=hn.shape[0]))
    for name in ['lepton','baryon']:
        sector=reconstruct(data,name,owner['results'][name]['retained_gauge_blocks'],scalar,loop,p);h=load(data,name+'_H');j=[load(data,name+'_J'+str(a)) for a in range(2)];k=[load(data,name+'_K'+str(a)) for a in range(2)];source=sparse.kron(compressed_source(neutral,sector,name,scalar),field['identity'],format='csr');raw=source@ground;initial=raw/norm(raw);kicked=expm_multiply(-1j*req['kick']*generator,initial.reshape(sector['dimension'],field['size']).T).T.ravel();neutral['target_blocks']=sector['blocks'];rawnorm=full_source_strength(neutral,name,ground,scalar,field,loop)
        # Independent direct CAR removal, not C-defined source by construction.
        anti=dict(sector);anti['blocks']=[]
        for b in sector['blocks']:
            ab=dict(b);ab['sector']=conjugate_sector(b['sector'],loop);anti['blocks'].append(ab)
        removal=sparse.kron(compressed_source(neutral,anti,name,scalar,removal=True),field['identity'],format='csr');ca=sparse.kron(sparse.eye(sector['dimension']),reflection);ah=load(data,'anti'+name+'_H')
        field_audit,jderived,kderived,kxy=field_shift_audit(h,kicked,field,sector['dimension']);obs={key:sparse.kron(compressed(sector,key,scalar),field['identity']) for key in ['phi2','C','positive','Q']};records=[];states=[]
        for t in req['times']:
            a=expm_multiply(-1j*t*h,initial);b=expm_multiply(-1j*t*h,kicked);states.append(b);pa=receiver(sector,a,scalar,field);pb=receiver(sector,b,scalar,field)
            records.append(dict(time=t,norms=[norm(a),norm(b)],physical_material_excess={key:mean(b,v)-mean(a,v) for key,v in obs.items()},current_excess=[mean(b,v)-mean(a,v) for v in j],POVM_sum=float(pb.sum()),POVM_min=float(pb.min()),POVM_TV=float(abs(pb-pa).sum()/2),reverse_source_strength_excess=norm(source.conj().T@b)**2-norm(source.conj().T@a)**2))
        q=obs['Q'];chosen=-1 if name=='lepton' else 1;pop=omitted_population(sector,states[-1],scalar,field,p);energy=energy_projection_residual(sector,states[-1],scalar,field,p);green,err=solve((req['source_frequency']+en+1j*p['eta'])*sparse.eye(h.shape[0])-h,raw)
        results[name]=dict(dimension=h.shape[0],population_depth=p['population_depth'],states_per_signature=p['states_per_signature'],scalar_points=p['scalar_points'],physical_Q=chosen,charge_operator_error=norm(q-chosen*sparse.eye(h.shape[0])),source_Q_Ward=norm(q@source-chosen*source),source_rebuild_vs_saved=norm(source-load(data,name+'_source')),retained_raw_source_strength=norm(raw)**2,full_CAR_source=rawnorm,total_omitted_source_norm=float(np.sqrt(max(0,rawnorm['full_unprojected_CAR_strength']-norm(raw)**2))),C_H_error=norm(ah@ca-ca@h),neutral_C_unitarity=norm(cn.conj().T@cn-sparse.eye(cn.shape[0])),direct_CAR_removal_vs_C_source=norm(removal-ca@source@cn.conj().T),direct_removal_raw_strength=norm(removal@ground)**2,field_shift_audit=field_audit,physical_current_n_over3_errors=[norm(a-b) for a,b in zip(j,jderived)],positive_contact_errors=[norm(a-b) for a,b in zip(k,kderived)],nonzero_mixed_contact_norm=norm(kxy),wrong_harmonic_as_physical_charge_current_error=norm(3*jderived[0]-j[0]),nonlinear_rotated_population_commutator=norm(h@obs['positive']-obs['positive']@h),population_final=pop,energy_projection_final=energy,samples=records,complete_projected_source_G=dict(frequency=req['source_frequency'],real=float(np.vdot(raw,green).real),imag=float(np.vdot(raw,green).imag),solver=err,all_retained_virtuals=h.shape[0]));stored[name+'_initial']=initial;stored[name+'_kicked']=kicked;stored[name+'_states']=np.array(states)
        print(json.dumps(dict(stage='full_source_event',sector=name,result=results[name])),flush=True)
    result=dict(kind='WAVE15_INDEPENDENT_FULL_AUTHOR_CUT_NEW_HADAMARD_SOURCE_QUANTUM_EVENT',request=req,neutral_eigen_residual=norm(hn@ground-en*ground),signed_neutral_retarded_response=root_response,results=results,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits='EXACT construct1 depth2/energy4/scalar17/electriccut3, not Wave14 smaller control. New full physical source state normalized only before independent EM-only unitary. All retained sparse-H evolution/resolvents, physical inclusive loop/phi/electric POVM. Source strength and spectral window not probability. Population/energy/electric omission actual and nonzero: no allpair closure. ChosenQ fundamental1/3 units, primitive CAR/ordered frame/spherical density and compact global fields remain inputs; no bulkphotons/COM/NoEther/QCD/lepton=proton/Nature claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)


if __name__=='__main__':main()
