"""Numeric, trusted-local snapshots of exact carrier Galerkin factors.

No producer replay, model globals, closures, old H/states or injected accessors.
The established multiplicity/angular/radial decomposition is retained verbatim
in mathematical content; this class makes its DATA an explicit public API.
"""
import hashlib,pickle
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from rich_rotcov_gauge_20261015_sparse import SparseRoundoff


class Factors(LinearOperator):
    def __init__(self,rows,cols,nr,terms,provenance=None):
        self.rows=rows;self.cols=cols;self.nr=nr;self.terms=terms;self.provenance={} if provenance is None else provenance
        super().__init__(dtype=np.dtype(complex),shape=(rows*nr,cols*nr))

    def _apply(self,x,adjoint=False):
        x=np.asarray(x).reshape(self.rows if adjoint else self.cols,self.nr);y=np.zeros((self.cols if adjoint else self.rows,self.nr),complex)
        for left,right,a,b,c in self.terms:
            if adjoint:left,right,a,b,c=right,left,a.conj().T,b.conj().T,c.conj().T
            z=x[right['indices']];z=(a@z.reshape(right['multiplets'],-1)).reshape(left['multiplets'],right['channels'],self.nr)
            z=(b@z.transpose(1,0,2).reshape(right['channels'],-1)).reshape(left['channels'],left['multiplets'],self.nr).transpose(1,0,2);z=np.einsum('ian,mn->iam',z,c,optimize=True)
            y[left['indices']]+=z
        return y.ravel()

    def _matvec(self,x):return self._apply(x)
    def _rmatvec(self,x):return self._apply(x,True)
    def _matmat(self,x):return np.column_stack([self@x[:,i] for i in range(x.shape[1])])

    def save(self,path):
        path=Path(path)
        with path.open('xb') as handle:pickle.dump(dict(format='CARRIER_MULTIPLICITY_ANGULAR_RADIAL_FACTORS_V1',rows=self.rows,cols=self.cols,nr=self.nr,terms=self.terms,provenance=self.provenance),handle,protocol=5)
        return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,shape=list(self.shape))


def load(path,sha256):
    """ONLY a trusted-local pinned numeric snapshot; pickle is not external I/O."""
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:raise ValueError('Carrier factor snapshot digest mismatch')
    with path.open('rb') as handle:data=pickle.load(handle)
    if data['format']!='CARRIER_MULTIPLICITY_ANGULAR_RADIAL_FACTORS_V1':raise ValueError('Unsupported carrier factor snapshot')
    return Factors(data['rows'],data['cols'],data['nr'],data['terms'],data['provenance'])


def compile(operator,vt,vs=None):
    """Same exact decomposition as the frozen pure helper; no giant CSR H."""
    vs=vt if vs is None else vs;nr=operator.nr;ng=operator.ng;target=vt[::nr,::nr].tocsr();source=vs[::nr,::nr].tocsr()
    def layout(spin,v):
        result={}
        for j in sorted(set(spin['j'])):
            multiplets=[a for a in spin['multiplets'] if a['j']==j];maps=[];reference=None
            for a in multiplets:
                local=v[a['start']*ng:a['stop']*ng];columns=np.unique(local.indices);w=local[:,columns].tocsr()
                if reference is None:reference=w
                elif w.shape!=reference.shape or sparse.linalg.norm(w-reference)>1e-12:raise ValueError('Projection is not multiplicity-separable')
                maps.append(columns)
            count=int(round(2*j+1));result[float(j)]=dict(indices=np.array(maps,dtype=int),W=[reference[i*ng:(i+1)*ng] for i in range(count)],multiplets=len(multiplets),channels=reference.shape[1])
        return result
    factory=SparseRoundoff();tl=layout(operator.target['spin'],target);sl=layout(operator.source['spin'],source);terms=[]
    def add(left,right,matter,angle,radius):
        angle=factory.csr(angle)
        if matter.nnz and angle.nnz:terms.append([left,right,matter,angle,radius])
    for rows,cols,matter,angle,radius in operator.blocks:
        j=float(operator.target['spin']['j'][rows[0]]);k=float(operator.source['spin']['j'][cols[0]]);left=tl[j];right=sl[k];dj=int(round(2*j+1));dk=int(round(2*k+1))
        for mi in range(dj):
            for ni in range(dk):
                a=matter[mi::dj,ni::dk].tocsr()
                if not a.nnz:continue
                angular=left['W'][mi].conj().T@sparse.csr_matrix(angle)@right['W'][ni];add(left,right,a,angular,radius)
    if operator.field is not None:
        for j,radius in operator.field.items():
            left=tl[j];right=sl[j];angular=sum((a.conj().T@b for a,b in zip(left['W'],right['W'])),sparse.csr_matrix((left['channels'],right['channels']))).toarray();add(left,right,sparse.eye(left['multiplets'],right['multiplets'],format='csr'),angular,radius)
    result=Factors(target.shape[1],source.shape[1],nr,terms);rng=np.random.default_rng(22215);x=rng.normal(size=result.shape[1])+1j*rng.normal(size=result.shape[1]);x/=np.linalg.norm(x);direct=vt.conj().T@(operator@(vs@x));actual=result@x
    receipt=dict(factor_groups=len(terms),stored_multiplicity_nonzeros=sum(a.nnz for l,r,a,b,c in terms),angular_nonzeros=sum(b.nnz for l,r,a,b,c in terms),sparse_roundoff=factory.snapshot(),projection_action_relative_error=float(np.linalg.norm(direct-actual)/max(np.linalg.norm(direct),np.finfo(float).tiny)),construction='Exact existing multiplicity/angular/radial factor architecture, explicit numeric snapshot API; no second Hilbert space or giant H CSR')
    result.provenance=receipt
    return result,receipt
