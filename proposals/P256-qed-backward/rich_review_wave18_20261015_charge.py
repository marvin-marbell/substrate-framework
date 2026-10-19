"""NEW native material-charge/source/current consumer; trusted pinned snapshots only.
No old producer, current-commutator replacement, source-strength fit or particle PVM.
"""
from itertools import product
from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib,json,resource,sys,time
import numpy as np
from scipy import sparse
from scipy.linalg import expm,svd
from scipy.sparse.linalg import expm_multiply
from rich_spatial_colour_charge_20261015_snapshot import load
from rich_spatial_colour_charge_20261015_model import (law as original_law,route_current,BodyLift,
    charge_grams,forward,electric_image,insert,CHARGES3)
from rich_review_wave18_20261015_law import physical_law
ROOT=Path(__file__).resolve().parent


def real(x: Any)->float:return float(np.real(x))


def norm(x: Any)->float:return float(np.linalg.norm(x.data)) if sparse.issparse(x) else float(np.linalg.norm(x))


def sha(path: Path)->str:
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def mean(state: Any,operator: Any)->float:return real(np.vdot(state,operator@state))


def pvm(state: Any,basis: Any,nf: int)->Any:
    weights=np.sum(abs(state.reshape(basis.dimension,-1))**2,axis=1);labels=sorted(set(basis.h));values=np.array([sum(weights[basis.h==h]) for h in labels])
    return dict(charge3=[int(h) for h in labels],physical_charge=[float(h/3) for h in labels],probabilities=values.tolist(),sum=float(values.sum()),mean=float(np.array(labels)@values/3),minimum_unclipped=float(values.min()))


def distance(left: Any,right: Any)->Any:
    delta=np.array(left['probabilities'])-right['probabilities'];l1=float(sum(abs(delta)));return dict(L1=l1,TV=l1/2,L2=float(np.linalg.norm(delta)),mean_difference=left['mean']-right['mean'])


def evolve(h: Any,state: Any,t: float)->Any:return expm_multiply(-1j*t*h,state,traceA=-1j*t*h.trace())


@lru_cache(maxsize=None)
def pattern(mask: int)->Any:
    occ=[];numbers=[];choices=[]
    for f,s,c in product(range(2),range(2),range(4)):
        pair=[16*f+8*m+4*s+c for m in range(2)];present=[m for m,i in enumerate(pair) if mask&(1<<i)];numbers.append(len(present));choices.append(present[0] if len(present)==1 else 0);occ.extend(pair[m] for m in present)
    inversions=sum(a>b for k,a in enumerate(occ) for b in occ[k+1:]);return tuple(numbers),tuple(choices),(-1)**inversions


def raw_characteristic(left: Any,right: Any,chi: float)->complex:
    total=0j
    for lm,lv in left.items():
        ln,lc,ls=pattern(lm)
        for rm,rv in right.items():
            rn,rc,rs=pattern(rm)
            if ln!=rn:continue
            value=complex(ls*rs)
            for number,a,b,q3 in zip(ln,lc,rc,CHARGES3):
                phase=np.exp(1j*chi*q3/3)
                if number==2:value*=phase
                elif number==1:value*=(1+phase)/2 if a==b else (phase-1)/2
            total+=value*np.vdot(lv,rv)
    return total


def charge_algebra(data: Any)->Any:
    output={}
    for name in ['neutral','lepton','baryon','antilepton','antibaryon']:
        basis=data['native'][name];errors=[]
        for i,j in [(0,0),(min(1,len(basis.seeds)-1),min(1,len(basis.seeds)-1)),(0,min(1,len(basis.seeds)-1))]:
            for chi in [.37,2*np.pi,6*np.pi]:
                direct=raw_characteristic(basis.seeds[i],basis.seeds[j],chi);native=sum(np.exp(1j*chi*h/3)*g[i,j] for h,g in basis.grams.items());errors.append(abs(direct-native))
        grams=[w.conj().T@basis.grams[h]@w for h,w in basis.blocks.items()];normalization=max(float(np.linalg.norm(g-np.eye(len(g)))) for g in grams)
        period_witness=abs(raw_characteristic(basis.seeds[0],basis.seeds[0],2*np.pi)-raw_characteristic(basis.seeds[0],basis.seeds[0],0.))
        output[name]=dict(dimension=basis.dimension,seeds=len(basis.seeds),physical_native_Gram_defect=normalization,independent_pair_exponential_vs_raw_Laurent_error=max(errors),wrong_2pi_period_witness=float(period_witness),physical_period=6*np.pi,charge3_labels=[int(h) for h in sorted(set(basis.h))],twice_Sz_labels=[int(m) for m in sorted(set(basis.m))],meaning='Independent canonical CAR reordering plus exact2x2 active-pair exponentials. Fractions are q=h/3; physical U_chi BEFORE any band cut, not exp(compressedQ).')
    return output


def orbit_high_pairs(data: Any)->Any:
    basis=data['native']['neutral'];sea=sum(1<<(16*f+8+4*s+c) for f,s,c in product(range(2),range(2),range(4)));vac=np.zeros(19,complex);vac[0]=1;grams=charge_grams(basis.seeds,[{sea:vac}]);state=np.zeros(basis.dimension,complex)
    for h,w in basis.blocks.items():
        if h in grams:state[basis.slices[h]]=np.asarray(w.conj().T@grams[h].toarray()).ravel()
    chi=np.pi;turned=np.exp(1j*chi*basis.h/3)*state;op=data['operators']['neutral'];positive=op['N']/2+sum(m[0]+m[0].conj().T for _,m in op['hops'][0])/2
    expected=sum(np.sin(chi*float(q)/6)**2 for q in CHARGES3)
    return dict(bare_seed_native_norm2=real(np.vdot(state,state)),orbit_state_native_norm2=real(np.vdot(turned,turned)),native_positive_band_count=mean(turned,positive),independent_pair_Bernoulli_positive_band_count=float(expected),charge_PVM_change=distance(pvm(turned,basis,1),pvm(state,basis,1)),meaning='Actual bare occupied sea rotated by physical U_site(pi) inside native orbit. Positive-band mean8 cannot fit old pair1. This is kinematic orbit closure, not fixed-field H gauge symmetry, dynamics or photon preparation.')


def source_checks(data: Any,omega: Any,nf: int,h: Any,e0: float)->Any:
    out={};prepared={};cn=data['C_matter']['neutral'];rhos=data['operators'];a0=omega.reshape(data['native']['neutral'].dimension,-1)
    for name,values in data['source_matter'].items():
        entries={};op=rhos[name];jz=BodyLift(op['J'][2],nf);j2=BodyLift(op['J2'],nf)
        for key,matrix in values.items():
            raw=(matrix@a0).ravel();strength=real(np.vdot(raw,raw));full=real(np.vdot(a0,data['source_metric_full'][name][key]@a0));j=float(key.split(':')[1]);m=float(key.split(':')[2])
            source_energy=mean(raw,h[name])/strength
            entries[key]=dict(retained_raw_strength=strength,full_native_raw_SdagS_strength=full,omission=full-strength,source_Jz_residual=norm(jz@raw-m*raw)/np.sqrt(strength),source_J2_residual=norm(j2@raw-j*(j+1)*raw)/np.sqrt(strength),source_mean_energy=source_energy,preparation_cost=source_energy-e0,
                physical_total_Q=mean(raw,BodyLift(op['Q'],10*nf))/strength,physical_total_N=mean(raw,BodyLift(op['N'],10*nf))/strength)
            if key=='0:0.5:0.5':prepared[name]=raw/np.sqrt(strength)
            if name.startswith('anti'):
                positive='lepton' if name=='antilepton' else 'baryon';entries[key]['direct_C_source_operator_error']=norm(matrix-data['source_matter'][positive][key]@cn.conj().T)
        out[name]=entries
    return out,prepared


def raw_conjugate(packet: Any)->Any:
    out={}
    for mask,value in packet.items():
        target=(1<<32)-1;sign=1
        for i in reversed([j for j in range(32) if mask&(1<<j)]):
            k=i^12;sign*=(-1)**(((i%8)//4)+(target&((1<<k)-1)).bit_count());target^=1<<k
        changed=value.copy();changed[1:10]=value[10:19];changed[10:19]=value[1:10];out[target]=sign*changed
    return out


def C_seed_evidence(data: Any)->Any:
    result={}
    for name,anti in [('lepton','antilepton'),('baryon','antibaryon')]:
        error=0.
        for original,actual in zip(data['native'][name].seeds,data['native'][anti].seeds):
            expected=raw_conjugate(original)
            for mask in set(expected)|set(actual):error=max(error,float(np.linalg.norm(expected.get(mask,np.zeros(19))-actual.get(mask,np.zeros(19)))))
        result[name]=dict(actual_raw_C_seed_error=error,native_charge_sign_error=norm(data['operators'][anti]['rho0']+data['operators'][name]['rho0']),meaning='Explicit particle-hole CAR annihilation signs and PW3/bar3 swap verify actual C-image native seeds. Identity positive-to-anti coordinate matrix is not assumed to prove physical C.')
    return result


def native_ward(data: Any,name: str,p: Any,h: Any)->Any:
    op=data['operators'][name];r=data['rotor'];g=data['geometry'];q=np.repeat(op['rho0'].diagonal(),10);terms=[]
    for edge in range(2):
        coordinates=[r['integrate'](g['tangents'][:,edge,a]) for a in range(3)]
        for charge,matrices in op['hops'][edge]:
            if charge:terms.append((charge,g['weights'][edge],matrices,[f.conj().T.tocsr() for f in matrices],coordinates))
    sums=np.zeros(4);rotation_one=np.zeros(3);rotation_phi=np.zeros(3)
    for first in range(0,op['Q'].shape[0],8):
        last=min(first+8,op['Q'].shape[0]);start,stop=10*first,10*last;shape=(stop-start,h.one.shape[1]);jo=sparse.csr_matrix(shape,dtype=complex);jp=jo.copy()
        for charge,weight,matrices,adjoints,coordinates in terms:
            jp+=-1j*charge*p['scalar_coupling']*weight*sparse.kron(matrices[0][first:last]-adjoints[0][first:last],sparse.eye(10),format='csr')
            for axis in range(3):jo+=charge*p['vector_velocity']*weight*sparse.kron(matrices[axis+1][first:last]+adjoints[axis+1][first:last],coordinates[axis],format='csr')
        for slot,(matrix,current) in enumerate([(h.one,jo),(h.phi,jp)]):
            a=matrix[start:stop];delta=1j*(a.multiply(q[None,:])-a.multiply(q[start:stop,None]))+current;sums[slot]+=norm(delta)**2
            skew=a-matrix[:,start:stop].conj().T;sums[slot+2]+=norm(skew)**2
        for axis,j in enumerate(op['J']):
            rotation_one[axis]+=norm(j[start:stop]@h.one-h.one[start:stop]@j)**2
            rotation_phi[axis]+=norm(j[start:stop]@h.phi-h.phi[start:stop]@j)**2
    off_body=op['electric']-p['plaquette']*op['B'];rho=op['rho0']
    return dict(original_graph_Ward_one=float(np.sqrt(sums[0])),original_graph_Ward_phi=float(np.sqrt(sums[1])),all_hops_off_material_charge_body_commutator=norm(off_body@rho-rho@off_body),body_electric_material_charge_commutator=norm(op['electric']@rho-rho@op['electric']),owning_H_one_Hermitian_defect=float(np.sqrt(sums[2])),owning_H_phi_Hermitian_defect=float(np.sqrt(sums[3])),full_rotation_one_errors=np.sqrt(rotation_one).tolist(),full_rotation_phi_errors=np.sqrt(rotation_phi).tolist(),method='COMPLETE streamed80-row original physical graph-current formula, H commutator/Hermiticity and all-M rotation coefficients. No full large sparse subtraction, no J redefinition.')
 

def stream_grams(left: Any,right: Any)->Any:
    from rich_spatial_colour_charge_20261015_model import packet_matrix,pair_pattern,charge_polynomial
    lm,la=packet_matrix(left);rm,ra=packet_matrix(right);lp=[pair_pattern(m) for m in lm];rp=[pair_pattern(m) for m in rm];groups={}
    for j,(single,double,choice,sign) in enumerate(rp):groups.setdefault((single,double),[]).append(j)
    output={};identity=sparse.eye(19,format='csr')
    for start in range(0,len(lm),4):
        stop=min(start+4,len(lm));entries={}
        for i in range(start,stop):
            single,double,choice,sign=lp[i];ns=tuple(sum(bool(single&(1<<a)) and CHARGES3[a]==q for a in range(16)) for q in (-3,-1,0,2));offset=sum(int(CHARGES3[a]) for a in range(16) if double&(1<<a))
            for j in groups.get((single,double),[]):
                diff=choice^rp[j][2];ds=tuple(sum(bool(diff&(1<<a)) and CHARGES3[a]==q for a in range(16)) for q in (-3,-1,0,2))
                if ds[2]:continue
                for h,value in charge_polynomial(ns,ds,offset):
                    rows,cols,values=entries.setdefault(h,([],[],[]));rows.append(i-start);cols.append(j);values.append(value*sign*rp[j][3])
        bra=la[19*start:19*stop].conj().T
        for h,(rows,cols,values) in entries.items():
            kernel=sparse.csr_matrix((values,(rows,cols)),shape=(stop-start,len(rm)));piece=(bra@(sparse.kron(kernel,identity,format='csr')@ra)).toarray()
            if h in output:output[h]+=piece
            else:output[h]=piece
    return output


def identical(a: Any,b: Any)->bool:
    return a.shape==b.shape and np.array_equal(a.indptr,b.indptr) and np.array_equal(a.indices,b.indices) and np.array_equal(a.data,b.data)


def difference_norm(a: Any,b: Any)->float:
    if identical(a,b):return 0.
    return float(np.sqrt(sum(norm(a[i:i+80]-b[i:i+80])**2 for i in range(0,a.shape[0],80))))


def build_H(data: Any,p: Any)->Any:
    h={}
    for name,op in data['operators'].items():
        candidate=original_law(op,data['rotor'],data['scalar'],p,data['geometry']);positive='lepton' if name=='antilepton' else 'baryon' if name=='antibaryon' else None
        if positive is not None and identical(candidate.one,h[positive].one) and identical(candidate.phi,h[positive].phi):h[name]=h[positive]
        else:h[name]=candidate
    return h


def coefficient_difference(original: Any,factored: Any)->Any:
    squared=np.zeros(2)
    for first in range(0,factored.n,8):
        last=min(first+8,factored.n);start,stop=10*first,10*last;shape=(stop-start,10*factored.n);blocks=[sparse.csr_matrix(shape,dtype=complex) for _ in range(2)]
        for c,f,r,phi in factored.terms:blocks[int(phi)]+=c*sparse.kron(f[first:last],sparse.csr_matrix(r),format='csr')
        for c,m in factored.linear:blocks[0]+=c*m[start:stop]
        for k,name in enumerate(['one','phi']):squared[k]+=norm(getattr(original,name)[start:stop]-blocks[k])**2
    return dict(one=float(np.sqrt(squared[0])),phi=float(np.sqrt(squared[1])),scope='COMPLETE explicit coefficient-row differences, no subtraction of large H norms; scalar Hphi shared separately. Floating CSR cleanup/summation differences are retained, not hidden.')


def factor_equivalence(data: Any,p: Any,old: Any,new: Any)->Any:
    out={}
    for name in ['neutral','lepton','baryon']:out[name+'_H0']=coefficient_difference(old[name],new[name])
    args=(data['operators']['neutral'],data['rotor'],data['scalar'],p,data['geometry']);fields=np.array(p['A_pulse']+p['B_pulse'])
    out['neutral_finite_A_B']=coefficient_difference(original_law(*args,fields=fields),physical_law(*args,fields=fields))
    for a in range(6):out['neutral_source_'+str(a)]=coefficient_difference(original_law(*args,fields=fields,derivative=(a,)),physical_law(*args,fields=fields,derivative=(a,)))
    for a,b in [(0,0),(0,3),(4,5)]:out['neutral_contact_'+str((a,b))]=coefficient_difference(original_law(*args,fields=fields,derivative=(a,b)),physical_law(*args,fields=fields,derivative=(a,b)))
    return out


def raw_metric(basis: Any,images: Any,shifts: Any)->Any:
    result=sparse.lil_matrix((basis.dimension,)*2,dtype=complex)
    for a,left in enumerate(images):
        for b,right in enumerate(images):
            grams=stream_grams(left,right)
            for hout,g in grams.items():
                hl=hout-shifts[a];hr=hout-shifts[b]
                if hl in basis.blocks and hr in basis.blocks:result[basis.slices[hl],basis.slices[hr]]+=basis.blocks[hl].conj().T@g@basis.blocks[hr]
    return result.tocsr()


def omissions(data: Any,omega: Any,prepared: Any,p: Any)->Any:
    neutral=data['native']['neutral'];images=[electric_image(x,data['loop'],p,data['geometry']) for x in neutral.seeds];metric=raw_metric(neutral,[images],[0]);a=omega.reshape(neutral.dimension,-1);electric=data['operators']['neutral']['electric'];raw=real(np.vdot(a,metric@a));ret=norm(electric@a)**2
    basis=data['native']['lepton'];q=-1.;forward_images=[{m:-1j*q*v for m,v in forward(x,data['loop'],0,1,True,np.eye(2)).items()} for x in basis.seeds];backward_images=[{m:1j*q*v for m,v in forward(x,data['loop'],0,1,True,np.eye(2),True).items()} for x in basis.seeds];full=raw_metric(basis,[forward_images,backward_images],[3,-3]);f=data['operators']['lepton']['hops'][0][3][1][0];current=-1j*q*(f-f.conj().T);a=prepared['lepton'].reshape(basis.dimension,10,-1)*data['scalar']['grid'][None,None,:];a=a.reshape(basis.dimension,-1);factor=p['scalar_coupling']*data['geometry']['weights'][0];fullnorm=factor**2*real(np.vdot(a,full@a));retained=factor**2*norm(current@a)**2
    return dict(neutral_colour_electric_H_component=dict(full_raw_norm2=raw,projected_norm2=ret,omitted_norm2=raw-ret),lepton_tree_scalar_original_current_component=dict(full_raw_norm2=fullnorm,projected_norm2=retained,omitted_norm2=fullnorm-retained,metric_Hermiticity=norm(full-full.conj().T)),meaning='Actual full raw CAR/PW19 operator-image SdagS before target native projection. Scalar colourless tree route-current component only, NOT total Cartesian/mechanical current. Colour-electric H component only, NOT a total H omission/error bound. Signed roundoff retained; total H/current tails remain unbounded.')


def seed_only_wrongcase(data: Any,t: float)->Any:
    basis=data['native']['neutral'];v=np.zeros((basis.dimension,len(basis.seeds)),complex)
    for h,w in basis.blocks.items():v[basis.slices[h]]=w.conj().T@basis.grams[h].toarray()
    u,s,_=svd(v,full_matrices=False);keep=s>1e-11*s[0];u=u[:,keep];coeff=np.exp(.31j*np.arange(u.shape[1]));coeff/=np.linalg.norm(coeff);initial=u@coeff;e=data['operators']['neutral']['electric'];correct=expm_multiply(-1j*t*e,initial,traceA=-1j*t*e.diagonal().sum());wrong=u@(expm(-1j*t*(u.conj().T@e@u))@coeff)
    return dict(seed_rank=u.shape[1],native_rank=basis.dimension,true_native_material_charge_PVM_change=distance(pvm(correct,basis,1),pvm(initial,basis,1)),wrong_seed_only_material_charge_PVM_change=distance(pvm(wrong,basis,1),pvm(initial,basis,1)),meaning='NEW diagnostic seed-only projection on same owning colour-electric term, not an old producer/checker replay. Compressing BEFORE physical charge completion can create material-charge drift even with all hopping absent. No full-law no-go.')


def current_contact(data: Any,name: str,state: Any,p: Any)->Any:
    op=data['operators'][name];args=(op,data['rotor'],data['scalar'],p,data['geometry']);direction=np.array([.2,-.3,.4,.1,.15,-.2]);epsilon=3e-4
    hp=physical_law(*args,fields=epsilon*direction);hm=physical_law(*args,fields=-epsilon*direction);h0=physical_law(*args)
    j=sum(direction[a]*(physical_law(*args,derivative=(a,))@state) for a in range(6));k=sum(direction[a]*direction[b]*(physical_law(*args,derivative=(a,b))@state) for a,b in product(range(6),repeat=2))
    off_j=[physical_law(*args,derivative=(a,),path_scale=(0.,0.)) for a in range(3)];zero=sparse.csr_matrix(op['Q'].shape);shape=(10*op['Q'].shape[0],)*2
    offop=dict(op,connection=dict(op['connection'],linear=[sparse.csr_matrix(shape) for _ in range(6)],products={a:zero for a in range(3)}))
    fully_off=[physical_law(offop,*args[1:],derivative=(a,),path_scale=(0.,0.)) for a in range(3)]
    return dict(original_source_first_direction_error=norm((hp@state-hm@state)/(2*epsilon)+j)/norm(j),original_full_contact_second_direction_error=norm((hp@state+hm@state-2*(h0@state))/epsilon**2-k)/norm(k),all_hops_off_mechanical_current_action_norms=[norm(x@state) for x in off_j],hops_AND_rotor_EM_coupling_off_current_action_norms=[norm(x@state) for x in fully_off],meaning='All graph hops off does NOT remove rigid-rotor EM current. Only explicit simultaneous rotor-connection decoupling control has zero EM source current. Public current remains original physical source derivative, not i[Hcut,Qcut].')


def finite_current_response(data: Any,name: str,state: Any,p: Any,lag: float,h: Any)->Any:
    args=(data['operators'][name],data['rotor'],data['scalar'],p,data['geometry'])
    def vectors(s: Any)->Any:return [physical_law(*args,derivative=(a,))@s for a in range(3)]+[(s.reshape(-1,len(data['scalar']['grid']))*data['scalar']['grid'][None,:]).ravel()]
    end=expm_multiply(-1j*lag*h,np.column_stack([state]+vectors(state)),traceA=-1j*lag*h.trace());ordered=np.column_stack(vectors(end[:,0])).conj().T@end[:,1:];commutator=-1j*(ordered-ordered.conj())
    return dict(lag=lag,ordered=dict(real=ordered.real.tolist(),imag=ordered.imag.tolist()),signed_commutator=dict(real=commutator.real.tolist(),imag=commutator.imag.tolist()),instantaneous_original_contact=[[mean(state,physical_law(*args,derivative=(a,b))) for b in range(3)] for a in range(3)],nonstationarity=norm(h@state-state*np.vdot(state,h@state)),meaning='Original Cartesian current plus actual quantum scalar field4x4 nonstationary two-time response on entire retained H. No low-pole virtual replacement, bare-polarization/photon claim or commutator-current patch.')


def selected_raw_source_reconstruction(data: Any)->Any:
    out={}
    for name,key in [('lepton','0:0.5:0.5'),('baryon','0:1.5:0.5'),('baryon','split2:1.5:-1.5')]:
        item=data['source_definitions'][name][key];neutral=data['native']['neutral'];images=[insert(seed,item['terms']) for seed in neutral.seeds];metric=raw_metric(neutral,[images],[item['shift']]);target=data['native'][name];grams=stream_grams(target.seeds,images);operator=sparse.lil_matrix(data['source_matter'][name][key].shape,dtype=complex)
        for hr,wr in neutral.blocks.items():
            hl=hr+item['shift']
            if hl in target.blocks and hl in grams:operator[target.slices[hl],neutral.slices[hr]]=target.blocks[hl].conj().T@grams[hl]@wr
        operator=operator.tocsr()
        out[name+':'+key]=dict(full_raw_SdagS_reconstruction_error=norm(metric-data['source_metric_full'][name][key]),native_source_image_reconstruction_error=norm(operator-data['source_matter'][name][key]))
    return out


def main()->None:
    started=time.monotonic();request:Any=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);body=ROOT/request['body']
    limit=request['memory_limit_MiB']*1024**2;resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    if sha(body)!=request['body_sha256']:raise ValueError('Trusted local body bytes differ from author-pinned digest; not unpickled')
    data=load(body);p=dict(data['parameters']);nf=len(data['scalar']['grid']);captured=np.load(ROOT/request['neutral_state']);omega=captured['vectors'][:,0];e0=float(captured['energies'][0]);h=build_H(data,p);states=dict(neutral_reference=omega);reference=dict(energy=e0,H_state_residual=norm(h['neutral']@omega-e0*omega),J2=mean(omega,BodyLift(data['operators']['neutral']['J2'],nf)),meaning='Actual new native reference selected in stationary M0 block; public H and all source/response operations retain all M. Not a global all-Fock vacuum theorem.')
    cn=data['C_matter']['neutral'];reference['C_neutral_unitarity_error']=norm(cn.conj().T@cn-sparse.eye(cn.shape[0]));reference['C_neutral_eigenstate_overlap']=float(abs(np.vdot(omega,data['C']['neutral']@omega)))
    owning=np.load(ROOT/request['H_snapshot']);binding={}
    for name,operator in h.items():
        differences={}
        for part in ['one','phi']:
            key=name+'_H0_'+part;saved=sparse.csr_matrix((owning[key+'_data'],owning[key+'_indices'],owning[key+'_indptr']),shape=tuple(owning[key+'_shape']))
            differences[part]=difference_norm(getattr(operator,part),saved);del saved
        binding[name]=differences
    algebra=charge_algebra(data);orbit=orbit_high_pairs(data);sources,prepared=source_checks(data,omega,nf,h,e0);ward={name:native_ward(data,name,p,h[name]) for name in ['neutral','lepton','baryon']};cseed=C_seed_evidence(data);events={};C={}
    factors={name:physical_law(op,data['rotor'],data['scalar'],p,data['geometry']) for name,op in data['operators'].items()};equivalence=factor_equivalence(data,p,h,factors);h=factors;del operator
    for name in ['lepton','baryon']:
        state=prepared[name];basis=data['native'][name];op=data['operators'][name];initial=pvm(state,basis,nf);off=physical_law(op,data['rotor'],data['scalar'],p,data['geometry'],path_scale=(0.,0.));timepoint=request['duration'];base=evolve(h[name],state,timepoint);control=evolve(off,state,timepoint);kicked=(state.reshape(-1,nf)@expm(-1j*p['field_kick']*data['scalar']['P']).T).ravel();driven=evolve(h[name],kicked,timepoint);receiver=data['sources'][name]['1:0.5:0.5']@omega;receiver/=np.linalg.norm(receiver)
        events[name]=dict(initial=initial,baseline=pvm(base,basis,nf),all_hops_off=pvm(control,basis,nf),all_hops_off_material_PVM_change=distance(pvm(control,basis,nf),initial),scalar_field_only_t0_PVM_change=distance(pvm(kicked,basis,nf),initial),field_only_later_excess=distance(pvm(driven,basis,nf),pvm(base,basis,nf)),conditional_CAR_capture_initial=float(abs(np.vdot(receiver,state))**2),conditional_CAR_capture_hops_off=float(abs(np.vdot(receiver,control))**2),meaning='Inclusive MATERIAL site-charge PVM, not added-particle location/capture. Capture can vary without material charge transfer. Field-only scalar unitary has no initial material event but may drive later conditional response.')
        def phi_mean(s: Any)->float:return real(np.sum(abs(s.reshape(-1,nf))**2*data['scalar']['grid'][None,:]))
        electric=BodyLift(op['electric'],10*nf);route_base=[];route_driven=[]
        for edge in range(2):
            current=route_current(op,data['rotor'],data['scalar'],p,data['geometry'],edge);route_base.append(mean(base,current));route_driven.append(mean(driven,current));del current
        events[name]['scalar_colour_loop_response']=dict(phi_baseline=phi_mean(base),phi_driven=phi_mean(driven),colour_electric_baseline=mean(base,electric),colour_electric_driven=mean(driven,electric),original_route_current_baseline=route_base,original_route_current_driven=route_driven)
        apulse=physical_law(op,data['rotor'],data['scalar'],p,data['geometry'],fields=np.array(p['A_pulse']+[0.,0.,0.]));aafter=evolve(apulse,state,request['pulse_duration']);bafter=evolve(h[name],state,request['pulse_duration']);afinal=evolve(h[name],aafter,timepoint);bfinal=evolve(h[name],bafter,timepoint)
        del apulse
        events[name]['actual_external_A_pulse']=dict(duration=request['pulse_duration'],fields=p['A_pulse'],material_charge_excess=distance(pvm(afinal,basis,nf),pvm(bfinal,basis,nf)),phi_excess=phi_mean(afinal)-phi_mean(bfinal),colour_electric_excess=mean(afinal,electric)-mean(bfinal,electric),meaning='Actual3D original line-integral/rotor source Hamiltonian during pulse, followed by same H0. Matched total elapsed baseline, not quantum EM/photon preparation.')
        states[name+'_A_pulse_final']=afinal;states[name+'_A_matched_final']=bfinal
        states.update({name+'_initial':state,name+'_baseline':base,name+'_all_hops_off':control,name+'_scalar_kicked':kicked,name+'_scalar_driven':driven})
        anti='antilepton' if name=='lepton' else 'antibaryon';c=data['C'][name];fields=np.array(p['A_pulse']+p['B_pulse']);cstate=c@state
        anti_action=physical_law(data['operators'][anti],data['rotor'],data['scalar'],p,data['geometry'],fields=-fields)@cstate
        positive_action=c@(physical_law(op,data['rotor'],data['scalar'],p,data['geometry'],fields=fields)@state)
        C[name]=dict(owning_H_zero_field_action_error=norm(h[anti]@cstate-c@(h[name]@state)),finite_original_field_action_error=norm(anti_action-positive_action),current_odd_action_errors=[norm(physical_law(data['operators'][anti],data['rotor'],data['scalar'],p,data['geometry'],derivative=(a,))@cstate+c@(physical_law(op,data['rotor'],data['scalar'],p,data['geometry'],derivative=(a,))@state)) for a in range(3)])
        anti_end=evolve(h[anti],prepared[anti],timepoint);strength=sources[name]['0:0.5:0.5']['retained_raw_strength'];anti_strength=sources[anti]['0:0.5:0.5']['retained_raw_strength'];addition=strength*np.vdot(state,base)*np.exp(1j*e0*timepoint);removal=anti_strength*np.vdot(prepared[anti],anti_end)*np.exp(1j*e0*timepoint);retarded=-1j*(addition+np.conj(removal))
        events[name]['raw_odd_source_time_sample']=dict(time=timepoint,addition=[float(addition.real),float(addition.imag)],C_removal=[float(removal.real),float(removal.imag)],retarded_anticommutator=[float(retarded.real),float(retarded.imag)],raw_addition_strength=strength,raw_C_removal_strength=anti_strength,meaning='Unnormalized full-retained-H time correlations and odd-source anticommutator; composite trilinear source is NOT assigned elementary Dirac/CAR normalization or fitted pole.')
        states[anti+'_time_sample']=anti_end
    checkpoint=output.with_name(output.name.replace('_result.json','_checkpoint.npz'));np.savez_compressed(checkpoint,**states);checkpoint.with_suffix('.json').write_text(json.dumps(dict(kind='NEW18_NATIVE_SOURCE_AND_MATERIAL_EVENT_SNAPSHOT_BEFORE_RAW_OMISSION_WORK',reference=reference,events=events,body_sha256=sha(body)),indent=2)+'\n')
    result=dict(kind='NEW_INDEPENDENT_NATIVE_PHYSICAL_CHARGE_ORBIT_BODY_CONSUMER',request=request,reference=reference,native_charge_algebra=algebra,coherent_high_band_pair_orbit=orbit,all_native_sources=sources,native_original_current_Ward=ward,actual_C_seed_evidence=cseed,C_owning_law=C,events=events,meaningful_seed_only_wrongcase=seed_only_wrongcase(data,request['duration']),raw_omission_witnesses=omissions(data,omega,prepared,p),original_current_contact_and_decoupling_controls=current_contact(data,'lepton',prepared['lepton'],p),
        owning_H_snapshot_reconstruction=binding,
        original_law_factored_equivalence=equivalence,
        selected_full_raw_source_reconstruction=selected_raw_source_reconstruction(data),
        original_current_scalar_nonstationary_response={name:finite_current_response(data,name,states[name+'_baseline'],p,request['response_lag'],h[name]) for name in ['lepton','baryon']},
        limits=['Same microscopic law/couplings but explicitly NEW finite charge-orbit seed projection; no old source/H/energy relabelling.','Exact material charge closure does not establish full R-dependent gauge/Cartesian Ward closure, H-range completeness or full total-current omission bound.','Physical rho=h/3 with6pi period, SU3 colour-electric/T generators, material R and external A/B are different typed quantities. No base Coulomb/self/skin/neutralizer, extra I/S2/baryon hop or source-strength fit.','All-M raw sources and C are consumed; source costs not elementary masses, no unique halfspin/proton or positron=proton claim.','No quantum EM/photon/bulk/COM/Nature admission; new EM integration and FrameOrbit outside18. Frozen old adverse controls remain regime-specific ground truth, not rerun or global no-go.'],elapsed_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),**states);print(json.dumps(dict(result=output.name,ward=ward,orbit=orbit,events={name:{k:v for k,v in entry.items() if k.endswith(('change','excess'))} for name,entry in events.items()},raw_omissions=result['raw_omission_witnesses'],peak_RSS_KiB=result['peak_RSS_KiB'],elapsed=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
