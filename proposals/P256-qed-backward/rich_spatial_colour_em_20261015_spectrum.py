"""Stationary solver only: exact TOTAL M reduction of the FULL coupled angular H.
Never a fixed-body-M tensor with spectator photons. All propagation/resolvents stay full.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,eigsh
from rich_spatial_colour_em_20261015_model import FactorOperator

def _total_M_operator(h,body_labels,rq_labels,twice_M):
    """Exact factor contractions on grouped TOTAL-M coordinates, without a dense H."""
    groups=[];selected=[];offset=0
    for label in np.unique(body_labels):
        body=np.flatnonzero(body_labels==label);rq=np.flatnonzero(rq_labels+label==twice_M)
        if not len(rq):continue
        indices=(body[:,None,None]*(h.nf*h.nrq)+np.arange(h.nf)[None,:,None]*h.nrq+rq[None,None,:]).ravel()
        groups.append(dict(body=body,rq=rq,part=slice(offset,offset+len(indices))))
        selected.append(indices);offset+=len(indices)
    blocks=[]
    for a,b,phi in h.terms:
        for target,g in enumerate(groups):
            for origin,f in enumerate(groups):
                left=a[g['body']][:,f['body']]
                if not left.nnz:continue
                right=b[g['rq']][:,f['rq']]
                if right.nnz:blocks.append((target,origin,left,right,phi))
    def apply(vectors):
        vectors=np.asarray(vectors);single=vectors.ndim==1
        if single:vectors=vectors[:,None]
        columns=vectors.shape[1];out=np.zeros(vectors.shape,complex)
        x=[vectors[g['part']].reshape(len(g['body']),h.nf,len(g['rq']),columns) for g in groups]
        y=[out[g['part']].reshape(len(g['body']),h.nf,len(g['rq']),columns) for g in groups]
        if h.include_scalar:
            for source,target in zip(x,y):target+=np.einsum('fg,mgkc->mfkc',h.scalar['H'],source,optimize=True)
        for target,origin,a,b,phi in blocks:
            ni=len(groups[target]['body']);nj=len(groups[origin]['body']);ri=len(groups[target]['rq']);rj=len(groups[origin]['rq'])
            left=(a@x[origin].reshape(nj,-1)).reshape(ni,h.nf,rj,columns)
            right=(b@left.transpose(2,0,1,3).reshape(rj,-1)).reshape(ri,ni,h.nf,columns).transpose(1,2,0,3)
            if phi:right*=h.scalar['grid'][None,:,None,None]
            y[target]+=right
        return out[:,0] if single else out
    return LinearOperator((offset,offset),matvec=apply,matmat=apply,rmatvec=apply,dtype=complex),np.concatenate(selected)



def lowest_total_M(model,name,h,twice_M,p):
    base=model['base'];field=model['field'];rotor=base['rotor'];op=base['operators'][name]
    sz=op['S'][2];body_diagonal=sz.diagonal().real;off=sz-sparse.diags(body_diagonal)
    off_norm=float(sparse.linalg.norm(off));label_error=float(np.max(abs(2*body_diagonal-np.rint(2*body_diagonal))))
    if off_norm>1e-9 or label_error>1e-9:
        raise ValueError('Native orbit API must preserve body2S_z blocks for exact total-M stationary solver; no dense/false-M fallback: '+str((off_norm,label_error)))
    body_labels=np.rint(2*body_diagonal).astype(int)
    lr,ur=np.linalg.eigh(rotor['L'][2].toarray());lf,uf=np.linalg.eigh(field['F'][2]);u=np.kron(ur,uf)
    rq_labels=(np.rint(2*lr).astype(int)[:,None]+np.rint(2*lf).astype(int)[None,:]).ravel()
    transformed=FactorOperator(h.nm,h.scalar,h.nrq,[(a,u.conj().T@b.toarray()@u,phi) for a,b,phi in h.terms],h.include_scalar)
    reduced,selected=_total_M_operator(transformed,body_labels,rq_labels,twice_M)
    def embed(vector):
        full=np.zeros(h.shape[0],complex);full[selected]=np.asarray(vector).ravel();return full
    rng=np.random.default_rng(2221015);probe=rng.normal(size=len(selected))+1j*rng.normal(size=len(selected));probe/=np.linalg.norm(probe)
    image=transformed@embed(probe);leakage=float(np.linalg.norm(image-embed(image[selected])))
    if leakage>1e-8:
        raise ValueError('Actual H does not preserve claimed TOTAL-M symmetry; cannot truncate it into a fake rotation sector: '+str(leakage))
    count=min(p['low_eigenstates'],len(selected)-2);ncv=min(len(selected),max(2*count+8,32))
    e,v=eigsh(reduced,k=count,ncv=ncv,which='SA',tol=p['eigen_tolerance'],maxiter=24000,v0=rng.normal(size=len(selected)))
    q,s,_=np.linalg.svd(v,full_matrices=False);q=q[:,s>1e-10*s[0]];small=q.conj().T@(reduced@q);e,w=np.linalg.eigh((small+small.conj().T)/2);v=q@w
    full=np.zeros((h.shape[0],v.shape[1]),complex);full[selected]=v
    full=np.einsum('ab,mfbc->mfac',u,full.reshape(h.nm,h.nf,h.nrq,-1),optimize=True).reshape(h.shape[0],-1)
    evidence=dict(full_H_dimension=h.shape[0],stationary_total_M=twice_M/2,stationary_sector_dimension=len(selected),actual_symmetry_leakage=leakage,body_M_diagonal_error=off_norm,body_M_quantization_error=label_error,scope='Stationary eigenvectors only, reconstructed into FULL physical body+SO3+field space. No Jhalf constraint: totalM+.5 includes all J>=half. Currents/resolvents/field-only moving states retain ALL M. Rotational degeneracy is resolved without an incomplete ARPACK multiplet projector.')
    return e,full,evidence


def integer_rotation_multiplets(model,name,energies,representatives,p):
    """Complete neutral TOTAL-M0 representatives by ACTUAL full J ladders."""
    from rich_spatial_colour_em_20261015_model import angular
    generators=[angular(model,name,axis) for axis in range(3)]
    j2=sum(generator@(generator@representatives) for generator in generators)
    vectors=[];labels=[];output_energies=[];evidence=[];done=set()
    for first,energy in enumerate(energies):
        if first in done:continue
        indices=np.flatnonzero(abs(energies-energy)<p['degeneracy_tolerance']);done.update(indices.tolist())
        v=representatives[:,indices];small=v.conj().T@j2[:,indices];values,rotation=np.linalg.eigh((small+small.conj().T)/2);v=v@rotation
        for column,value in enumerate(values):
            j=(np.sqrt(1+4*value)-1)/2;integer=int(round(float(j)));error=abs(j-integer)
            if error>1e-7:raise ValueError('Neutral angular representative is not a resolved integer-J multiplet: '+str((value,error)))
            centre=v[:,column];vectors.append(centre);labels.append((integer,0));output_energies.append(float(energy))
            upward=centre;downward=centre
            for m in range(integer):
                upward=(generators[0]@upward+1j*(generators[1]@upward))/np.sqrt((integer-m)*(integer+m+1))
                downward=(generators[0]@downward-1j*(generators[1]@downward))/np.sqrt((integer-m)*(integer+m+1))
                vectors.extend([upward,downward]);labels.extend([(integer,m+1),(integer,-m-1)]);output_energies.extend([float(energy)]*2)
            evidence.append(dict(energy=float(energy),J=integer,J_quantization_error=float(error)))
    full=np.column_stack(vectors)
    return np.array(output_energies),full,np.array(labels),dict(representatives=representatives.shape[1],complete_multiplet_vectors=full.shape[1],Gram_error=float(np.linalg.norm(full.conj().T@full-np.eye(full.shape[1]))),multiplets=evidence,scope='Actual NoEther ladders complete ALL M of the computed neutral low-window multiplets. No new H term, spin penalty, state postselection or low-virtual response truncation.')
