"""Exact retained total-J fibres and sparse factor snapshots, no j penalty."""
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy import sparse
from rich_rotcov_gauge_20261015_sparse import SparseRoundoff


def body_isometry(full,radial_count):
    # The supplied full isometry is V_body tensor I_radial.
    return full[::radial_count,::radial_count].tocsr()


def explicit(operator,target_isometry,source_isometry=None):
    source_isometry=target_isometry if source_isometry is None else source_isometry
    vt=body_isometry(target_isometry,operator.nr);vs=body_isometry(source_isometry,operator.nr);factory=SparseRoundoff()
    result=sparse.csr_matrix((target_isometry.shape[1],source_isometry.shape[1]),dtype=complex);cache_t={};cache_s={}
    def restrict(v,rows,cache):
        key=tuple(rows)
        if key not in cache:
            indices=(rows[:,None]*operator.ng+np.arange(operator.ng)[None,:]).ravel();cache[key]=v[indices]
        return cache[key]
    for rows,cols,matter,angle,radius in operator.blocks:
        left=restrict(vt,rows,cache_t);right=restrict(vs,cols,cache_s)
        angular=factory.kron(matter,factory.csr(angle));reduced=left.conj().T@angular@right
        result+=factory.kron(reduced,factory.csr(radius))
    if operator.field is not None:
        for j,matrix in operator.field.items():
            rows=np.flatnonzero(operator.source['spin']['j']==j)
            left=restrict(vt,rows,cache_t);right=restrict(vs,rows,cache_s)
            result+=factory.kron(left.conj().T@right,factory.csr(matrix))
    return result.tocsr(),factory.snapshot()


def factorized(operator,isometry):
    """Exact J projection, coalescing body factors before radial expansion."""
    v=body_isometry(isometry,operator.nr);factory=SparseRoundoff();cache={};groups={}
    def restrict(rows):
        key=tuple(rows)
        if key not in cache:
            indices=(rows[:,None]*operator.ng+np.arange(operator.ng)[None,:]).ravel();cache[key]=v[indices]
        return cache[key]
    def add(body,radius):
        radius=np.asarray(radius);key=(radius.shape,radius.dtype.str,radius.tobytes())
        if key in groups:groups[key][0]+=body
        else:groups[key]=[body.tocsr(),radius]
    for rows,cols,matter,angle,radius in operator.blocks:
        left=restrict(rows);right=restrict(cols)
        angular=factory.kron(matter,factory.csr(angle))
        add(left.conj().T@angular@right,radius)
    if operator.field is not None:
        for j,radius in operator.field.items():
            selected=restrict(np.flatnonzero(operator.source['spin']['j']==j))
            add(selected.conj().T@selected,radius)
    terms=[(factory.csr(body),radius) for body,radius in groups.values()]
    count=v.shape[1];nr=operator.nr
    class Projected(LinearOperator):
        def __init__(self):super().__init__(dtype=np.dtype(complex),shape=(count*nr,count*nr))
        def _apply(self,x,adjoint=False):
            x=np.asarray(x).reshape(count,nr);out=np.zeros_like(x,dtype=complex)
            for body,radius in terms:
                if adjoint:out+=(body.conj().T@x)@radius.conj()
                else:out+=(body@x)@radius.T
            return out.ravel()
        def _matvec(self,x):return self._apply(x)
        def _rmatvec(self,x):return self._apply(x,True)
    result=Projected();rng=np.random.default_rng(122231);x=rng.normal(size=result.shape[0])+1j*rng.normal(size=result.shape[0]);x/=np.linalg.norm(x)
    direct=isometry.conj().T@(operator@(isometry@x));actual=result@x
    ledger=dict(sparse_roundoff=factory.snapshot(),factor_groups=len(terms),stored_body_nonzeros=sum(body.nnz for body,radius in terms),projection_action_relative_error=float(np.linalg.norm(direct-actual)/max(np.linalg.norm(direct),np.finfo(float).tiny)),meaning='Same complete finite-J/radial operator; no radial Kronecker expansion, no matrix-element or state deletion.')
    return result,ledger


def csr_snapshot(output,key,matrix):
    matrix=matrix.tocsr();output[key+'_data']=matrix.data;output[key+'_indices']=matrix.indices;output[key+'_indptr']=matrix.indptr;output[key+'_shape']=np.array(matrix.shape)


def snapshot_operator(operator,output,prefix):
    for i,(rows,cols,mat,angle,radius) in enumerate(operator.blocks):
        key=f'{prefix}_term{i}';output[key+'_rows']=rows;output[key+'_cols']=cols;csr_snapshot(output,key+'_matter',mat);output[key+'_orientation']=angle;output[key+'_radial']=radius
    if operator.field is not None:
        for j,matrix in operator.field.items():output[prefix+f'_field_j{int(2*j)}']=matrix
    output[prefix+'_dimension']=np.array(operator.shape)
