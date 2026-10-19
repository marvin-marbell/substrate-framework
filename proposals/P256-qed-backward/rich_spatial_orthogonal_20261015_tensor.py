"""Physical local isometries and CP-column tensor-train SVD; no Gram sandwich."""
from typing import Any
import numpy as np
from scipy.linalg import svd


def local_frame(v: Any,relative_cut: float)->Any:
    u,s,vh=svd(v,full_matrices=False,lapack_driver='gesvd');keep=s>relative_cut*s[0]
    frame=u[:,keep];coordinates=s[keep,None]*vh[keep]
    return dict(frame=frame,coordinates=coordinates,singular_values=s,
        discarded_norm=float(np.linalg.norm(s[~keep])),reconstruction_norm=float(np.linalg.norm(v-frame@coordinates)),
        isometry_defect=float(np.linalg.norm(frame.conj().T@frame-np.eye(np.count_nonzero(keep)))))


def append_step(cores: Any,r: Any,z: Any,relative_cut: float)->Any:
    previous=r.shape[0];physical=z.shape[0]
    matrix=(r[:,None,:]*z[None,:,:]).reshape(previous*physical,-1)
    u,s,vh=svd(matrix,full_matrices=False,lapack_driver='gesvd');keep=s>relative_cut*s[0]
    core=u[:,keep].reshape(previous,physical,np.count_nonzero(keep));remainder=s[keep,None]*vh[keep]
    return cores+[core],remainder,dict(singular_values=s.tolist(),retained=int(np.count_nonzero(keep)),
        discarded_norm=float(np.linalg.norm(s[~keep])),isometry_defect=float(np.linalg.norm(u[:,keep].conj().T@u[:,keep]-np.eye(np.count_nonzero(keep)))))


def cp_train(coordinates: Any,order: Any,relative_cut: float)->Any:
    cores=[];r=np.ones((1,coordinates[0].shape[1]),complex);steps=[]
    for site in order:
        cores,r,info=append_step(cores,r,coordinates[site],relative_cut);steps.append(info)
    return finish_train(cores,r,order,steps)


def finish_train(cores: Any,r: Any,order: Any,steps: Any)->Any:
    # R has independent orthonormal rows up to singular scaling from final SVD.
    # Its right inverse maps ONLY the true CP column span, never a Gram inverse.
    u,s,vh=svd(r,full_matrices=False,lapack_driver='gesvd')
    inverse=vh.conj().T@(u.conj().T/s[:,None])
    return dict(cores=cores,R=r,inverse=inverse,order=list(order),steps=steps,
        smallest_state_singular=float(s[-1]),rank=r.shape[0],
        right_inverse_defect=float(np.linalg.norm(r@inverse-np.eye(len(s)))))


def charged_train(prefix: Any,charged_coordinates: Any,gindex: Any,site: int,relative_cut: float)->Any:
    cores,r,info=append_step(prefix['cores'],prefix['R'][:,gindex],charged_coordinates,relative_cut)
    return finish_train(cores,r,prefix['order']+[site],prefix['steps']+[info])


def transfer(core: Any,environment: Any,operator: Any=None)->Any:
    # General bra/ket bond matrix, not assumed Hermitian.
    weighted=np.einsum('lm,mpj->lpj',environment,core,optimize=True)
    if operator is not None:weighted=np.einsum('pq,lqj->lpj',operator,weighted,optimize=True)
    return core.reshape(-1,core.shape[-1]).conj().T@weighted.reshape(-1,core.shape[-1])


def project_sum(train: Any,operators_by_site: Any)->Any:
    count=len(operators_by_site[train['order'][0]]);environments=np.zeros((count,1,1),complex)
    norm=np.ones((1,1),complex)
    for core,site in zip(train['cores'],train['order']):
        result=[]
        for c,operator in enumerate(operators_by_site[site]):
            inherited=transfer(core,environments[c])
            inserted=transfer(core,norm,operator)
            result.append(inherited+inserted)
        environments=np.array(result)
        norm=transfer(core,norm)
    return environments


def physical_norm(train: Any)->Any:
    e=np.ones((1,1),complex)
    for core in train['cores']:e=transfer(core,e)
    return e


def overlap_cp_batch(train: Any,coordinates: Any)->Any:
    """<physical orthonormal B_i|product z_site,a,batch>, norm-one CP ket."""
    columns=coordinates[0].shape[1];batch=coordinates[0].shape[2];environment=np.ones((1,columns,batch),complex)
    for core,z in zip(train['cores'],coordinates):
        matrix=(environment[:,None,:,:]*z[None,:,:,:]).reshape(core.shape[0]*core.shape[1],columns*batch)
        environment=(core.reshape(-1,core.shape[-1]).conj().T@matrix).reshape(core.shape[-1],columns,batch)
    return environment


def project_product_batches(train: Any,coordinate_batches: Any,chunk: int=4)->Any:
    total=coordinate_batches[0].shape[2];rank=train['rank'];result=np.empty((total,rank,rank),complex)
    for start in range(0,total,chunk):
        end=min(total,start+chunk);overlap=overlap_cp_batch(train,[z[:,:,start:end] for z in coordinate_batches])
        result[start:end]=np.einsum('iab,aj->bij',overlap,train['inverse'],optimize=True)
    return result
