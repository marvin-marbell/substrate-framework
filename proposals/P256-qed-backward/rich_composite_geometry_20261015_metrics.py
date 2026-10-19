"""Actual local FULL-Fock and native coordinate isometry metrics, no assumed I."""
import numpy as np


def isometry(basis,primitive):
    native=0.;dropped=0.
    if hasattr(basis,'storage'):
        for block in basis.storage.values():
            q=block.q[:len(block.keys),:block.rank]
            native=max(native,float(np.linalg.norm(q.conj().T@q-np.eye(block.rank))))
            tiny=np.where(abs(q)<=2e-14,q,0.)
            materialized=q-tiny
            native=max(native,float(np.linalg.norm(materialized.conj().T@materialized-np.eye(block.rank))))
            if block.rank:dropped=max(dropped,float(np.max(np.linalg.norm(tiny,axis=0))))
    else:
        for indices in basis.blocks.values():
            keys=sorted({k for i in indices for k in basis.vectors[i]});lookup={k:j for j,k in enumerate(keys)}
            q=np.zeros((len(keys),len(indices)),complex)
            for column,index in enumerate(indices):
                for key,value in basis.vectors[index].items():q[lookup[key],column]=value
            native=max(native,float(np.linalg.norm(q.conj().T@q-np.eye(len(indices)))))
    local=0.;rows=[]
    if hasattr(primitive,'local'):
        for lab,indices in primitive.local.blocks.items():
            keys=sorted({k for i in indices for k in primitive.local.vectors[i]});lookup={k:j for j,k in enumerate(keys)}
            q=np.zeros((len(keys),len(indices)),complex)
            for column,index in enumerate(indices):
                for key,value in primitive.local.vectors[index].items():q[lookup[key],column]=value
            error=float(np.linalg.norm(q.conj().T@q-np.eye(len(indices))));local=max(local,error)
            rows.append({'local_counts_Sz':lab,'dimension':len(indices),'full_local_Fock_Gram_error':error})
    tensor_bound=float(np.expm1(4*np.log1p(local)))
    return {'native_coordinate_Gram_max_block_Frobenius':native,'max_materialized_column_tiny_omission_norm':dropped,
        'local_full_Fock_Gram_max_block_Frobenius':local,'local_blocks':rows,
        'four_site_tensor_Gram_operator_upper':tensor_bound,
        'native_physical_Gram_operator_upper':native+(1+native)*tensor_bound,
        'meaning':'Local16-mode raw Fock embeddings measured per exact local-count/Sz block. Four-site tensor Gram product bound plus actual native block Gram. No global2^64 expansion or assumed local metric identity. Bounds concern this isometry only, NOT fullH/current range or gap error.'}
