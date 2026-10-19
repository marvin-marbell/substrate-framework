"""Independent fresh reduced regulator control of the NEW nonlinear quantum law."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh,expm_multiply,cg,LinearOperator
from rich_quantum_common_field_20261015_model import build,rotor,expand_sector,vector_maps,whole_law,compressed,omitted_vector_norm,energy_projection_residual,conjugate_sector,operators,law,conjugation,signatures
from rich_quantum_common_field_20261015_sources import compressed_source,full_source_strength
from rich_quantum_common_field_20261015_feedback import field_overflow
from rich_common_colour_lepton_20261015_model import charge_conjugate


def norm(a):return float(sparse.linalg.norm(a) if sparse.issparse(a) else np.linalg.norm(a))
def mean(v,a):return float(np.vdot(v,a@v).real)
def ground(h):
    e,v=eigsh(h,k=1,which='SA',tol=1e-11,maxiter=20000);return float(e[0]),v[:,0]


def receiver(sector,psi,scalar,field):
    size=len(field['electric']);bins=[field['electric']<0,field['electric']==0,field['electric']>0];out=np.zeros((2,2,3,3))
    for b in sector['blocks']:
        full=(b['V']@psi[b['offset']*field['size']:(b['offset']+b['dimension'])*field['size']].reshape(b['dimension'],field['size'])).reshape(b['sector']['dimension'],len(scalar['grid']),size,size)
        e,u=np.linalg.eigh(b['op']['C']);full=np.einsum('ab,bfxy->afxy',u.conj().T,full)
        for a,c,d,f in np.ndindex(out.shape):out[a,c,d,f]+=norm(full[np.ix_(e<1e-8 if a==0 else e>=1e-8,scalar['grid']<0 if c==0 else scalar['grid']>=0,bins[d],bins[f])])**2
    return out


def main():
    inp,out=map(Path,sys.argv[1:3]);req=json.loads(inp.read_text());p=json.loads(Path(req['owner_input']).read_text());p.update(req['parameter_overrides']);base=build(p);scalar=base['scalar'];field=rotor(p);loop=base['loop'];sectors={};laws={};states={};results={}
    for name in ['neutral','lepton','baryon']:
        sector=expand_sector(base['sectors'][name],loop,scalar,p);sector['vector']=vector_maps(sector,scalar);h,j,k=whole_law(sector,scalar,field,p);sectors[name]=sector;laws[name]=(h,j,k)
        print(json.dumps(dict(stage='built',sector=name,dimension=h.shape[0],blocks=len(sector['blocks']),layers=sector['layers'])),flush=True)
    hn,jn,kn=laws['neutral'];en,reference=ground(hn);states['neutral']=reference
    # Signed static diagonal current/contact root from complete retained H; not low-pole sum.
    signed=[]
    op=LinearOperator(hn.shape,matvec=lambda x:hn@x-en*x+reference*np.vdot(reference,x),dtype=complex)
    for a in range(2):
        rhs=jn[a]@reference;rhs-=reference*np.vdot(reference,rhs);solution,info=cg(op,rhs,rtol=1e-10,atol=1e-12,maxiter=10000)
        signed.append(dict(contact=mean(reference,kn[a]),paramagnetic=-2*float(np.vdot(rhs,solution).real),signed_static=mean(reference,kn[a])-2*float(np.vdot(rhs,solution).real),solver_info=int(info),residual=norm(op@solution-rhs),all_retained_virtuals=hn.shape[0]))
    size=len(field['electric']);u=sparse.diags(np.ones(size-1),-1,shape=(size,size));ux=sparse.kron(u,sparse.eye(size));uy=sparse.kron(sparse.eye(size),u);generator=(ux+ux.conj().T)/2+(uy-uy.conj().T)/(4j)
    neutral=sectors['neutral'];nb=neutral['blocks'];cm=[[None]*len(nb) for _ in nb]
    for j,b in enumerate(nb):
        for i,a in enumerate(nb):
            cm[i][j]=sparse.csr_matrix((a['dimension'],b['dimension']),dtype=complex)
            if a['signature']!=signatures(charge_conjugate(b['sector']['masks'][0])[0]):continue
            rawc=conjugation(b['sector']['masks'],a['sector']['masks'])
            c=a['sector']['P'].conj().T@rawc@b['sector']['P']
            cm[i][j]=sparse.csr_matrix(a['V'].conj().T@(sparse.kron(c,sparse.eye(len(scalar['grid'])))@b['V']))
    cn=sparse.kron(sparse.bmat(cm),sparse.csr_matrix(np.fliplr(np.eye(field['size']))))
    for name in ['lepton','baryon']:
        sector=sectors[name];h,j,k=laws[name];source=sparse.kron(compressed_source(sectors['neutral'],sector,name,scalar),field['identity']);raw=source@reference;state=raw/norm(raw);kicked=expm_multiply(-.37j*generator,state.reshape(-1,field['size']).T).T.ravel();sectors['neutral']['target_blocks']=sector['blocks'];sourceevidence=full_source_strength(sectors['neutral'],name,reference,scalar,field,loop)
        def external(x,y):
            shifted=dict(field);shifted['shifts']={q:[a*np.exp(1j*(q/3)*delta) for a,delta in zip(v,[x,y,x+y])] for q,v in field['shifts'].items()};return whole_law(sector,scalar,shifted,p)[0]
        step=1e-3;de=[];ce=[]
        for axis in range(2):
            args=[0.,0.];args[axis]=step;hp=external(*args);hm=external(*[-x for x in args]);de.append(norm((hp-hm)/(2*step)+j[axis])/max(1.,norm(j[axis])));ce.append(norm((hp+hm-2*h)/step**2-k[axis])/max(1.,norm(k[axis])))
        mixed=(external(step,step)-external(step,-step)-external(-step,step)+external(-step,-step))/(4*step**2)
        expected=sparse.csr_matrix(h.shape,dtype=complex)
        for harmonic in [2,-1,-3]:
            shift=field['shifts'][harmonic][2];expected-=p['vector_velocity']*(harmonic/3)**2*sparse.kron(sector['vector'][2,harmonic],(shift-shift.conj().T)/(2j))
        obs={key:sparse.kron(compressed(sector,key,scalar),field['identity']) for key in ['phi2','C','Q']};records=[];events=[]
        for t in req['times']:
            a=expm_multiply(-1j*t*h,kicked);b=expm_multiply(-1j*t*h,state);events.append(a);pa=receiver(sector,a,scalar,field);pb=receiver(sector,b,scalar,field)
            records.append(dict(time=t,norm=norm(a),energy_drift=mean(a,h)-mean(kicked,h),material_excess={key:mean(a,x)-mean(b,x) for key,x in obs.items()},current_excess=[mean(a,x)-mean(b,x) for x in j],reverse_source_excess=norm(source.conj().T@a)**2-norm(source.conj().T@b)**2,POVM_sum=float(pa.sum()),POVM_total_variation=float(abs(pa-pb).sum()/2)))
        results[name]=dict(dimension=h.shape[0],layers=sector['layers'],original_positive_source=sourceevidence,retained_raw_source_strength=norm(raw)**2,source_projection_omitted_norm=float(np.sqrt(max(0,sourceevidence['full_unprojected_CAR_strength']-norm(raw)**2))),current_derivative_errors=de,contact_derivative_errors=ce,mixed_contact_norm=norm(expected),mixed_contact_derivative_error=norm(mixed-expected)/max(1.,norm(expected)),population_final=omitted_vector_norm(sector,events[-1],scalar,field,loop,p),energy_projection_final=energy_projection_residual(sector,events[-1],scalar,field,p),samples=records)
        anti=dict(sector);anti['blocks']=[];anti['anti']=True
        for block in sector['blocks']:
            a=dict(block);a['sector']=conjugate_sector(block['sector'],loop);a['op']=operators(a['sector'],loop);a['H0']=law(a['op'],scalar,p)[0];anti['blocks'].append(a)
        anti['vector']=vector_maps(anti,scalar);ah,aj,ak=whole_law(anti,scalar,field,p)
        ca=sparse.kron(sparse.eye(sector['dimension']),sparse.csr_matrix(np.fliplr(np.eye(field['size']))))
        direct_remove=sparse.kron(compressed_source(neutral,anti,name,scalar,removal=True),field['identity'])
        results[name]['independent_source_C']=dict(H=norm(ah@ca-ca@h),J=[norm(aj[a]@ca+ca@j[a]) for a in range(2)],neutral_C_unitarity=norm(cn.conj().T@cn-sparse.eye(cn.shape[0])),direct_raw_removal_vs_C_source=norm(direct_remove-ca@source@cn.conj().T),direct_removal_raw_strength=norm(direct_remove@reference)**2)
        results[name]['electric_overflow_final']=field_overflow(sector,events[-1],scalar,field,p)
        states[name+'_initial']=state;states[name+'_kick']=kicked;states[name+'_event']=np.array(events)
        print(json.dumps(dict(stage='event',sector=name,result=results[name])),flush=True)
    result=dict(kind='INDEPENDENT_NONLINEAR_QUANTUM_COMMON_REGULATOR_CONTROL_NOT_AUTHOR_ACTUAL2',parameters=p,neutral_energy=en,neutral_eigen_residual=norm(hn@reference-en*reference),signed_neutral_static_response=signed,charge_harmonic_nonzeros={str(q):[a.nnz for a in v] for q,v in field['shifts'].items()},results=results,limits='Populationdepth1/energy2/scalar7 are disclosed changed regulators, NOT author depth2/scalar17 acceptance. Electriccut3 retains true electron phase. Full Hadamard original-positive sources and primitive raw factors; unfiltered source event rather than low source pole. Exact retained propagation and physical inclusive POVM; residuals explicitly nonclosed. No imported G/D, COM, photons, Nature, SU4=QCD or lepton=proton.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**states)


if __name__=='__main__':main()
