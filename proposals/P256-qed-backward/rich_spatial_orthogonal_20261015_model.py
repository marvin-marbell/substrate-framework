"""SAME physical CP span, actual orthonormal tensor basis, direct owning operators.
Never Cdag Hraw C, H averaging, polar-unitary repair or assumed adjoint.
"""
from typing import Any
from itertools import product
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from rich_spatial_translation_20261015_model import build as old_build
from rich_spatial_collective_20261015_model import exterior_basis,annihilators,second_quantized
from rich_spatial_orthogonal_20261015_tensor import (local_frame,cp_train,charged_train,project_sum,
    physical_norm,overlap_cp_batch,project_product_batches)


def build(config: Any,law: Any)->Any:
    old:Any=old_build(law);locals=[]
    for block in old['blocks']:
        frames={}
        for number,key in [(6,'v0'),(7,'vp'),(5,'vm')]:
            data=local_frame(block[key],config['local_singular_relative_cut']);u=data['frame']
            data['operators']=np.array([u.conj().T@(op@u) for op in block['operators'][number]])
            frames[number]=data
        locals.append(frames)
    coordinates=[local[6]['coordinates'] for local in locals]
    train=cp_train(coordinates,list(range(9)),config['tensor_singular_relative_cut'])
    if train['rank']!=27:raise ValueError(f'Physical neutral rank changed: {train["rank"]}, required27')
    neutral=dict(trains=[train],matrices=[project_sum(train,[local[6]['operators'] for local in locals])],sizes=[27])
    charged={name:dict(trains=[],matrices=[],sizes=[]) for name in ['positive','negative']}
    for b in range(9):
        prefix=cp_train(coordinates,[r for r in range(9) if r!=b],config['tensor_singular_relative_cut'])
        for name,number in [('positive',7),('negative',5)]:
            t=charged_train(prefix,locals[b][number]['coordinates'],old['gindex'],b,config['tensor_singular_relative_cut'])
            if t['rank']!=162:raise ValueError(f'Physical charged rank changed: {name}/{b}/{t["rank"]}, required162')
            operators=[local[6]['operators'] for local in locals];operators[b]=locals[b][number]['operators']
            charged[name]['trains'].append(t);charged[name]['matrices'].append(project_sum(t,operators));charged[name]['sizes'].append(162)
            print('Direct tensor projection',name,b,'rank',t['rank'],flush=True)
    model=dict(geo=old['geo'],momenta=old['momenta'],transforms=old['transforms'],field=old['field'],offset=old['offset'],
        blocks=old['blocks'],locals=locals,gindex=old['gindex'],gaussian_count=old['gaussian_count'],local_translations=old['local_translations'],
        neutral=neutral,positive=charged['positive'],negative=charged['negative'],span_evidence={},orthogonality={})
    for name in ['neutral','positive','negative']:
        sec=model[name];sec['old_to_new']=[];evidence=[];norms=[]
        for b,t in enumerate(sec['trains']):
            mapping=t['R']@old[name]['whiteners'][b];sec['old_to_new'].append(mapping)
            singular=np.linalg.svd(mapping,compute_uv=False);norm=physical_norm(t)
            evidence.append(dict(rank=t['rank'],smallest_physical_state_singular=t['smallest_state_singular'],
                CP_reconstruction_discard_bound=sum(step['discarded_norm'] for step in t['steps']),right_inverse_defect=t['right_inverse_defect'],
                old_to_new_map_singular_min=float(singular[-1]),old_to_new_map_singular_max=float(singular[0]),
                old_to_new_map_condition=float(singular[0]/singular[-1]),old_physical_metric_departure=float(np.linalg.norm(mapping.conj().T@mapping-np.eye(len(mapping))))))
            norms.append(float(np.linalg.norm(norm-np.eye(t['rank']))))
        model['span_evidence'][name]=evidence;model['orthogonality'][name]=norms
        sec['matter_translation']=[]
        for b,t in enumerate(sec['trains']):
            z=[]
            for site in t['order']:
                number=6 if name=='neutral' or site!=b else (7 if name=='positive' else 5)
                key='v0' if number==6 else ('vp' if number==7 else 'vm');v=old['blocks'][site][key]
                if name!='neutral' and site!=b:v=v[:,old['gindex']]
                z.append((locals[site][number]['frame'].conj().T@(old['local_translations'][number]@v))[:,:,None])
            sec['matter_translation'].append((overlap_cp_batch(t,z)[:,:,0])@t['inverse'])
        install_operator(sec,model)
    return model


def install_operator(sec: Any,model: Any,frozen: bool=False)->Any:
    nb=model['field']['dimension'];sizes=sec['sizes'];offsets=np.cumsum([0]+[n*nb for n in sizes]);dimension=int(offsets[-1]);terms=model['field']['frozen' if frozen else 'coupling'];boson=model['field']['boson']
    sec['offsets']=offsets
    def action(vector: Any,adjoint: bool=False)->Any:
        vector=np.asarray(vector).reshape(-1);out=np.empty(dimension,complex)
        for b,n in enumerate(sizes):
            first,last=offsets[b:b+2];a=vector[first:last].reshape(n,nb)
            bo=boson.conj().T if adjoint else boson;result=(bo@a.T).T-model['offset']*a
            for f,k in zip(sec['matrices'][b],terms):
                if k.nnz:
                    if adjoint:f,k=f.conj().T,k.conj().T
                    result+=(k@(f@a).T).T
            out[first:last]=result.ravel()
        return out
    def matmat(matrix: Any)->Any:return np.column_stack([action(v) for v in matrix.T])
    def adjmatmat(matrix: Any)->Any:return np.column_stack([action(v,True) for v in matrix.T])
    sec['h']=LinearOperator((dimension,dimension),matvec=action,rmatvec=lambda v:action(v,True),matmat=matmat,rmatmat=adjmatmat,dtype=complex)
    sec['trace']=sum(n*boson.diagonal().sum()-model['offset']*n*nb+sum(np.trace(f)*k.diagonal().sum() for f,k in zip(fs,terms)) for n,fs in zip(sizes,sec['matrices']))
    return sec['h']


def operator_quality(model: Any,name: str)->Any:
    sec=model[name];terms=model['field']['coupling'];boson=model['field']['boson'];norms=[];skews=[];components=[]
    def kron_norm(pairs: Any)->float:
        value=0j
        for f,k in pairs:
            for g,l in pairs:value+=np.vdot(f,g)*np.sum(k.conj().multiply(l))
        return float(np.sqrt(max(float(np.real(value)),0.)))
    for fs,n in zip(sec['matrices'],sec['sizes']):
        pairs=[(np.eye(n),boson),(model['offset']*-np.eye(n),sparse.eye(boson.shape[0],format='csr'))]+list(zip(fs,terms))
        skew=[]
        for f,k in pairs:
            skew.append((f-f.conj().T,k));skew.append((f.conj().T,k-k.conj().T))
        norms.append(kron_norm(pairs));skews.append(kron_norm(skew));components.append([float(np.linalg.norm(f-f.conj().T)) for f in fs])
    return dict(whole_H_Frobenius=float(np.linalg.norm(norms)),whole_H_skew_Frobenius=float(np.linalg.norm(skews)),
        relative_whole_H_skew=float(np.linalg.norm(skews)/np.linalg.norm(norms)),per_owning_component_skew=components,
        method='Exact Kronecker Frobenius contractions of DIRECT tensor-projected matrices; skew built as deltaF tensorK+Fdag tensordeltaK, no cancellation of large H norms, no dense joint-H allocation, no posthoc averaging.')


def source_matrices(model: Any,sign: int,f: Any)->Any:
    name='positive' if sign==1 else 'negative';number=7 if sign==1 else 5;out=[];neutral=model['neutral']['trains'][0]
    for b,t in enumerate(model[name]['trains']):
        z=[]
        for site in t['order']:
            v=model['blocks'][site]['v0']
            if site==b:
                annihilation=annihilators(7,exterior_basis(7)[1],exterior_basis(6)[2]) if sign==1 else annihilators(6,exterior_basis(6)[1],exterior_basis(5)[2])
                v=sum(value*(op.conj().T@v if sign==1 else op@v) for value,op in zip(f[b] if sign==1 else f[b].conj(),annihilation))
            z.append((model['locals'][site][number if site==b else 6]['frame'].conj().T@v)[:,:,None])
        out.append(overlap_cp_batch(t,z)[:,:,0]@neutral['inverse'])
    return out


def neutral_occupation(model: Any,f: Any)->Any:
    operators=[];basis=exterior_basis(6)
    for b,u in enumerate(model['locals']):
        op=second_quantized(np.outer(f[b],f[b].conj()),basis[1],basis[2]);frame=u[6]['frame'];operators.append([frame.conj().T@(op@frame)])
    return project_sum(model['neutral']['trains'][0],operators)[0]


def rotation_structure(model: Any,structure: Any)->Any:
    lookup={tuple(k):i for i,k in enumerate(model['momenta'])};permutation=[lookup[(-ky%3,kx)] for kx,ky in model['momenta']];inverse=np.argsort(permutation);rotations={}
    phase={}
    for n in [5,6,7]:
        masks=exterior_basis(n)[1];phase[n]=np.exp(-.5j*np.pi*np.array([sum(.5 if i%2==0 else -.5 for i in range(12) if m&(1<<i)) for m in masks]))
    for name in ['neutral','positive','negative']:
        sec=model[name];correct=[];wrong=[];mapping=[0] if name=='neutral' else permutation
        for b,tket in enumerate(sec['trains']):
            r=mapping[b];tbra=sec['trains'][r];key='v0' if name=='neutral' else ('vp' if name=='positive' else 'vm');number=6 if name=='neutral' else (7 if name=='positive' else 5)
            for omit_spatial,destination in [(False,correct),(True,wrong)]:
                bra=sec['trains'][b] if omit_spatial else tbra;z=[]
                for site in bra['order']:
                    source_site=site if omit_spatial else int(inverse[site]);n=number if name!='neutral' and source_site==b else 6
                    v=model['blocks'][source_site][key if n!=6 else 'v0']
                    if name!='neutral' and n==6:v=v[:,model['gindex']]
                    z.append((model['locals'][site][n]['frame'].conj().T@(phase[n][:,None]*v))[:,:,None])
                destination.append(overlap_cp_batch(bra,z)[:,:,0]@tket['inverse'])
        rotations[name]=dict(matrices=correct,mapping=mapping,spin_only=wrong,particle_parity=0 if name=='neutral' else 1)
    rotations['field']=sparse.kron(sparse.eye(model['field']['nphi']),structure['rotation'],format='csr')
    return rotations


def spin_receiver(model: Any,points: int=56)->Any:
    outputs={};normalization=[];hermiticity=[]
    phases=np.exp(2j*np.pi*np.outer(np.arange(points),np.arange(7))/points)
    local_actions={}
    for site in range(9):
        for number,key in [(6,'v0'),(7,'vp')]:
            masks=exterior_basis(number)[1];up=np.array([sum(1 for i in range(0,12,2) if m&(1<<i)) for m in masks]);v=model['blocks'][site][key];u=model['locals'][site][number]['frame'];translated=v;values=[]
            for l in range(3):
                values.append(np.array([u[up==n].conj().T@translated[up==n] for n in range(7)]));translated=model['local_translations'][number]@translated
            local_actions[site,number]=np.einsum('ku,luij->ijkl',phases,np.array(values),optimize=True).reshape(u.shape[1],v.shape[1],points*3)
    for name in ['neutral','positive']:
        effects=[]
        for b,t in enumerate(model[name]['trains']):
            coords=[]
            for site in t['order']:
                n=7 if name=='positive' and site==b else 6;z=local_actions[site,n]
                if name=='positive' and n==6:z=z[:,model['gindex']]
                coords.append(z)
            char=project_product_batches(t,coords,chunk=4).reshape(points,3,t['rank'],t['rank'])
            e=np.fft.ifft(np.fft.fft(char,axis=0)/points,axis=1);effects.append(e)
            if name=='positive':
                normalization.append(float(np.linalg.norm(np.sum(e,axis=(0,1))-np.eye(t['rank']))));hermiticity.append(float(np.max(np.linalg.norm(e-np.swapaxes(e.conj(),-1,-2),axis=(-2,-1)))))
            print('Physical direct tensor spin POVM',name,b,flush=True)
        outputs[name]=effects
    return dict(effects=outputs['positive'],neutral_effects=outputs['neutral'][0],spin_values=(np.arange(points)-27.5).tolist(),neutral_spin_values=(np.arange(points)-27.).tolist(),
        normalization_defects=normalization,maximum_effect_hermiticity_defects=hermiticity,
        meaning='Physical orthonormal TT bra contracted with ambient full-CAR spin-count/Tz acted CP columns, then SINGLE state-coordinate right-inverse map. No ill-conditioned Gram sandwich, compressed-Pauli PVM, clipping or probability conditioning.')
