"""Exact raw CAR image metrics without materializing all Laurent mask pairs.
Pattern-wise charged-single Walsh transforms implement the SAME physical P_h.
Neutral-charge single choices stay orthogonal; no old band projector is applied.
"""
import numpy as np
from scipy import sparse
from rich_spatial_colour_charge_20261015_model import pair_pattern,CHARGES3


def charge_grams_streamed(images):
    """Return <image_i|P_h(raw site charge)|image_j>, holding PW19 fixed.

    A pattern fixes singly/doubly occupied mirror pairs. On its charged singles,
    P_h is a Walsh-Hadamard transform followed by an exact SITE-occupation mask.
    This is the native charge_polynomial kernel (.5,.5 or -.5,.5), not a
    projector built from compressed Q. One pattern is expanded at a time.
    """
    count=len(images);zero_mask=sum(1<<a for a,q in enumerate(CHARGES3) if q==0)
    groups={}
    for mask in set().union(*(image.keys() for image in images)):
        single,double,choice,sign=pair_pattern(mask)
        key=(single,double,choice&zero_mask)
        groups.setdefault(key,[]).append((mask,choice,sign))
    grams={};raw=np.zeros((count,count),complex);max_bytes=0;max_charged_singles=0
    for (single,double,_),members in groups.items():
        charged=[a for a,q in enumerate(CHARGES3) if q and single&(1<<a)]
        rows=1<<len(charged);x=np.zeros((rows,19,count),complex)
        max_bytes=max(max_bytes,x.nbytes);max_charged_singles=max(max_charged_singles,len(charged))
        for mask,choice,sign in members:
            index=sum(((choice>>a)&1)<<i for i,a in enumerate(charged))
            for column,image in enumerate(images):
                value=image.get(mask)
                if value is not None:x[index,:,column]=sign*value
        flat=x.reshape(-1,count);raw+=flat.conj().T@flat
        step=1
        while step<rows:
            pairs=x.reshape(-1,2*step,19,count);left=pairs[:,:step].copy()
            pairs[:,:step]=left+pairs[:,step:];pairs[:,step:]=left-pairs[:,step:]
            step*=2
        charges=np.full(rows,sum(int(CHARGES3[a]) for a in range(16) if double&(1<<a)),int)
        sites=np.arange(rows)
        for i,a in enumerate(charged):charges+=int(CHARGES3[a])*(((sites>>i)&1)==0)
        for h in np.unique(charges):
            selected=x[charges==h].reshape(-1,count)
            grams.setdefault(int(h),np.zeros((count,count),complex))
            grams[int(h)]+=selected.conj().T@selected/rows
    total=sum(grams.values(),np.zeros_like(raw))
    return grams,dict(pattern_count=len(groups),max_charged_singles=max_charged_singles,max_pattern_tensor_bytes=max_bytes,raw_seed_Gram_norm=float(np.linalg.norm(raw)),sum_charge_projectors_Gram_error=float(np.linalg.norm(total-raw)),method='Exact pattern-wise charged-single Walsh transform; native pair_pattern canonical CAR signs, full raw site-charge masks and PW19 amplitudes. No all-mask-pair Laurent list or raw all-pattern state is materialized.')


def native_image_metric(basis,images,shift):
    grams,evidence=charge_grams_streamed(images);blocks=[]
    for h,w in basis.blocks.items():
        gram=grams.get(h+shift,np.zeros((len(images),len(images)),complex))
        blocks.append(sparse.csr_matrix(w.conj().T@gram@w))
    metric=sparse.block_diag(blocks,format='csr')
    evidence.update(native_dimension=basis.dimension,physical_integer3_charge_shift=int(shift),metric_Hermiticity_Frobenius=float(sparse.linalg.norm(metric-metric.conj().T)))
    return metric,evidence,grams
