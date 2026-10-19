"""Literal cached direct-TT H, exact transverse-momentum fiber, all canonical spinors.
No old producer replay, borrowed free propagator, auxiliary photon number or normalization.
"""
from typing import Any
from itertools import product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator,eigsh,expm_multiply
from rich_spatial_translation_20261015_model import build as raw_build
from rich_spatial_orthogonal_20261015_tensor import local_frame,cp_train,charged_train,overlap_cp_batch,transfer
from rich_spatial_collective_20261015_model import exterior_basis,annihilators,second_quantized


def build(config: Any,law: Any,reconstruction: Any)->Any:
    raw:Any=raw_build(law);site=raw['momenta'].index(tuple(config['transverse_indices']));local=[]
    for b,block in enumerate(raw['blocks']):
        frames={6:local_frame(block['v0'],config['local_singular_relative_cut'])}
        if b==site:
            for number,key in [(7,'vp'),(5,'vm')]:frames[number]=local_frame(block[key],config['local_singular_relative_cut'])
        local.append(frames)
    coordinates=[u[6]['coordinates'] for u in local]
    neutral=cp_train(coordinates,list(range(9)),config['tensor_singular_relative_cut'])
    prefix=cp_train(coordinates,[b for b in range(9) if b!=site],config['tensor_singular_relative_cut'])
    trains=[neutral]+[charged_train(prefix,local[site][number]['coordinates'],raw['gindex'],site,config['tensor_singular_relative_cut']) for number in [7,5]]
    sectors={};bindings={}
    for name,t,rank in zip(['neutral','positive','negative'],trains,[27,162,162]):
        if t['rank']!=rank:raise ValueError(f'Changed physical rank {name}/{t["rank"]}')
        index=0 if name=='neutral' else site;cached_r=reconstruction[f'{name}_physical_CP_R_{index}']
        difference=float(np.linalg.norm(t['R']-cached_r)/np.linalg.norm(cached_r))
        if difference>1e-11:raise ValueError(f'Cached physical basis mismatch {name}: {difference}; no silent basis splice')
        f=reconstruction[name+'_projected_components'][index]
        sectors[name]=dict(train=t,matrices=f,n=rank)
        bindings[name]=dict(exact_cached_R_relative_difference=difference,rank=rank,source_coordinate_inverse_defect=t['right_inverse_defect'])
    field=raw['field'];model=dict(sectors=sectors,raw=raw,local=local,field=field,selected_site=site,law=law,basis_bindings=bindings)
    field_difference=max(float(np.linalg.norm(k.toarray()-reconstruction['field_terms'][i])) for i,k in enumerate(field['coupling']))
    bare_difference=float(np.linalg.norm(field['boson'].toarray()-reconstruction['field_boson']))
    if max(field_difference,bare_difference)>1e-12:raise ValueError('Owning cached field-law mismatch; no imported propagator splice')
    bindings['field']=dict(cached_Peierls_component_difference=field_difference,cached_bare_field_difference=bare_difference)
    for sec in sectors.values():install(sec,model,False)
    return model


def install(sec: Any,model: Any,decoupled: bool=False)->Any:
    field=model['field'];n=sec['n'];nb=field['dimension'];dimension=n*nb;terms=list(field['coupling'])
    if decoupled:
        identity=sparse.eye(nb,format='csr');terms=[identity,terms[1]]
        for _ in range(7):terms.extend([identity,sparse.csr_matrix((nb,nb))])
    f=sec['matrices'];b=field['boson'];offset=model['raw']['offset'];adj_f=[x.conj().T for x in f];adj_k=[x.conj().T for x in terms];bd=b.conj().T
    def action(matrix: Any,adjoint: bool=False)->Any:
        matrix=np.asarray(matrix);vector=matrix.ndim==1;x=matrix.reshape(n,nb,-1);columns=x.shape[-1]
        bo=bd if adjoint else b;out=(bo@x.transpose(1,0,2).reshape(nb,n*columns)).reshape(nb,n,columns).transpose(1,0,2)-offset*x
        for fc,kc in zip(adj_f if adjoint else f,adj_k if adjoint else terms):
            if kc.nnz:
                left=(fc@x.reshape(n,nb*columns)).reshape(n,nb,columns)
                out+=(kc@left.transpose(1,0,2).reshape(nb,n*columns)).reshape(nb,n,columns).transpose(1,0,2)
        result=out.reshape(dimension,columns)
        return result[:,0] if vector else result
    h=LinearOperator((dimension,dimension),matvec=action,matmat=action,rmatvec=lambda v:action(v,True),rmatmat=lambda v:action(v,True),dtype=complex)
    sec.update(h=h,trace=n*b.diagonal().sum()-offset*dimension+sum(np.trace(fc)*kc.diagonal().sum() for fc,kc in zip(f,terms)),decoupled=decoupled)
    return h


def canonical_sources(model: Any)->Any:
    site=model['selected_site'];neutral=model['sectors']['neutral']['train'];result={name:[] for name in ['positive','negative']};occupations={name:[] for name in ['neutral','positive','negative']}
    for kz in range(3):
        for spinor in range(4):
            f=np.zeros(12,complex);f[spinor::4]=np.exp(2j*np.pi*kz*np.arange(3)/3)/np.sqrt(3)
            for name,number in [('positive',7),('negative',5)]:
                train=model['sectors'][name]['train'];coordinates=[]
                for b in train['order']:
                    v=model['raw']['blocks'][b]['v0']
                    if b==site:
                        ops=annihilators(7,exterior_basis(7)[1],exterior_basis(6)[2]) if number==7 else annihilators(6,exterior_basis(6)[1],exterior_basis(5)[2])
                        v=sum(value*(a.conj().T@v if number==7 else a@v) for value,a in zip(f if number==7 else f.conj(),ops))
                    coordinates.append((model['local'][b][number if b==site else 6]['frame'].conj().T@v)[:,:,None])
                result[name].append(overlap_cp_batch(train,coordinates)[:,:,0]@neutral['inverse'])
            for name,number in [('neutral',6),('positive',7),('negative',5)]:
                train=model['sectors'][name]['train'];environment=np.ones((1,1),complex)
                for core,b in zip(train['cores'],train['order']):
                    operator=None
                    if b==site:
                        basis=exterior_basis(number);op=second_quantized(np.outer(f,f.conj()),basis[1],basis[2]);u=model['local'][b][number]['frame'];operator=u.conj().T@(op@u)
                    environment=transfer(core,environment,operator)
                occupations[name].append(environment)
    model['sources']={name:np.array(s) for name,s in result.items()};model['occupations']={name:np.array(s) for name,s in occupations.items()}
    return model['sources']


def reference(model: Any,cached_neutral: Any,decoupled: bool=False)->Any:
    if not decoupled:
        sec=model['sectors']['neutral'];e,v=eigsh(sec['h'],k=1,which='SA',tol=1e-11,maxiter=5000,v0=cached_neutral)
        omega=v[:,0];energy=float(e[0]);factor=None
    else:
        field=model['field'];np_=field['nphi'];ne=field['shell_dimension'];em=field['em_kinetic'];bare=field['boson'].toarray().reshape(np_,ne,np_,ne)
        phi=(np.einsum('peqe->pq',bare)-np.trace(em)*np.eye(np_))/ne
        em_values,em_vectors=eigh(em);factor={};offset=model['raw']['offset']
        for name,sec in model['sectors'].items():
            fs=sec['matrices'];constant=fs[0]+sum(fs[c] for c in range(2,16,2))
            h=np.kron(np.eye(sec['n']),phi)+np.kron(constant,np.eye(np_))+np.kron(fs[1],np.diag(model['law']['coupling']*field['phi']))-offset*np.eye(sec['n']*np_)
            values,vectors=eigh(h);factor[name]=dict(values=values,vectors=vectors)
        energy=float(factor['neutral']['values'][0]+em_values[0]);omega=np.kron(factor['neutral']['vectors'][:,0],em_vectors[:,0])
        factor['em']=dict(values=em_values,vectors=em_vectors)
        factor['bare_tensor_reconstruction_norm']=float(np.linalg.norm(field['boson'].toarray()-np.kron(phi,np.eye(ne))-np.kron(np.eye(np_),em)))
    for sec in model['sectors'].values():sec['E0']=energy
    means=[];field=model['field'];a=omega.reshape(27,field['dimension'])
    for q in field['coordinates']:means.append(float(np.real(np.vdot(a,(q@a.T).T))))
    field['centered']=[q-means[i]*sparse.eye(field['dimension'],format='csr') for i,q in enumerate(field['coordinates'])]
    return dict(omega=omega,energy=energy,field_means=means,factor=factor,
        H_state_residual=float(np.linalg.norm(model['sectors']['neutral']['h']@omega-energy*omega)),
        reference_semantics='Own exactly factorized matter/Phi x EM reference; scalar coupling NOT frozen.' if decoupled else 'Fresh ground of literal coupled owning Hamiltonian.')


def evolve(model: Any,ref: Any,name: str,vectors: Any,time: float,adjoint: bool=False,chunk: int=16)->Any:
    vectors=np.asarray(vectors);one=vectors.ndim==1
    if one:vectors=vectors[:,None]
    if time==0:return vectors[:,0].copy() if one else vectors.copy()
    sec=model['sectors'][name];result=np.empty_like(vectors,complex)
    if ref['factor'] is not None:
        factor=ref['factor'];m=factor[name];em=factor['em'];nm=len(m['values']);ne=len(em['values'])
        phase=np.exp((1j if adjoint else -1j)*time*(m['values'][:,None]+em['values'][None,:]-ref['energy']))
        for start in range(0,vectors.shape[1],chunk):
            end=min(vectors.shape[1],start+chunk);a=(m['vectors'].conj().T@vectors[:,start:end].reshape(nm,-1)).reshape(nm,ne,-1)
            a=np.einsum('ef,mfv->mev',em['vectors'].conj().T,a,optimize=True)*phase[:,:,None]
            a=np.einsum('ef,mfv->mev',em['vectors'],a,optimize=True)
            result[:,start:end]=(m['vectors']@a.reshape(nm,-1)).reshape(sec['n']*model['field']['dimension'],-1)
    else:
        h=sec['h'];dimension=h.shape[0]
        shifted=LinearOperator(h.shape,matvec=lambda x:h@x-ref['energy']*x,matmat=lambda x:h@x-ref['energy']*x,
            rmatvec=lambda x:h.H@x-ref['energy']*x,rmatmat=lambda x:h.H@x-ref['energy']*x,dtype=complex)
        operator=(1j*time)*shifted.H if adjoint else (-1j*time)*shifted;trace=(1j*time*np.conj(sec['trace']-ref['energy']*dimension)) if adjoint else (-1j*time*(sec['trace']-ref['energy']*dimension))
        for start in range(0,vectors.shape[1],chunk):result[:,start:start+chunk]=expm_multiply(operator,vectors[:,start:start+chunk],traceA=trace)
    return result[:,0] if one else result


def act(model: Any,name: str,operator: int,vectors: Any)->Any:
    vectors=np.asarray(vectors);one=vectors.ndim==1
    if one:vectors=vectors[:,None]
    field=model['field'];nb=field['dimension'];sec=model['sectors'][name];a=vectors.reshape(sec['n'],nb,-1)
    if operator<4:
        q=field['centered'][operator];out=(q@a.transpose(1,0,2).reshape(nb,-1)).reshape(nb,sec['n'],-1).transpose(1,0,2);target=name
    else:
        create=operator>=16;index=operator-16 if create else operator-4
        if name=='neutral':target='positive' if create else 'negative';s=model['sources'][target][index]
        elif name=='positive' and not create:target='neutral';s=model['sources']['positive'][index].conj().T
        elif name=='negative' and create:target='neutral';s=model['sources']['negative'][index].conj().T
        else:raise ValueError('Four-point request left actual N54/N55/N53 sectors')
        out=(s@a.reshape(sec['n'],-1)).reshape(s.shape[0],nb,-1)
    out=out.reshape(-1,vectors.shape[1]);return target,(out[:,0] if one else out)


def adjoint_operator(op: int)->int:return op if op<4 else (op+12 if op<16 else op-12)


def field_cut_weights(model: Any)->Any:
    states=model['field']['shell_states'];cut=max(sum(s) for s in states);extended=sorted([x for x in product(range(cut+2),repeat=4) if sum(x)<=cut+1],key=lambda x:(sum(x),x));lookup={x:i for i,x in enumerate(extended)};width=model['law']['shell_width'];maps=[]
    for axis in range(4):
        q=np.zeros((len(extended),len(states)),complex)
        for j,state in enumerate(states):
            if state[axis]:
                target=list(state);target[axis]-=1;q[lookup[tuple(target)],j]+=width*np.sqrt(state[axis]/2)
            target=list(state);target[axis]+=1;q[lookup[tuple(target)],j]+=width*np.sqrt((state[axis]+1)/2)
        maps.append(q)
    model['field']['before_cut_Q_products']=[[sparse.kron(sparse.eye(model['field']['nphi']),a.conj().T@b,format='csr') for b in maps] for a in maps]
    return maps
