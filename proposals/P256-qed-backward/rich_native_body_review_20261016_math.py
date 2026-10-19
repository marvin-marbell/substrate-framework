"""Independent fixed-state weak-H physics; not hot source-EOM Gdot dynamics."""
from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator,expm_multiply


class FieldHamiltonian(LinearOperator):
    def __init__(self,c0,cphi,hphi,grid):
        self.c0=sparse.csr_matrix(c0);self.cp=sparse.csr_matrix(cphi)
        self.hp=np.asarray(hphi,complex);self.grid=np.asarray(grid);self.n=self.c0.shape[0];self.nf=len(grid)
        self.trace=float((self.nf*self.c0.diagonal().sum()+self.grid.sum()*self.cp.diagonal().sum()+self.n*np.trace(self.hp)).real)
        super().__init__(dtype=np.dtype(complex),shape=(self.n*self.nf,)*2)
    def apply(self,value,adjoint=False):
        a=np.asarray(value);z=a.reshape(self.n,self.nf,-1)
        c0=self.c0.conj().T if adjoint else self.c0;cp=self.cp.conj().T if adjoint else self.cp;hp=self.hp.conj().T if adjoint else self.hp
        out=(c0@z.reshape(self.n,-1)).reshape(z.shape)
        out+=(cp@(z*self.grid[None,:,None]).reshape(self.n,-1)).reshape(z.shape)
        out+=np.einsum('fg,ngk->nfk',hp,z)
        return out.reshape(a.shape)
    def _matvec(self,value):return self.apply(value)
    def _matmat(self,value):return self.apply(value)
    def _rmatvec(self,value):return self.apply(value,True)
    def _rmatmat(self,value):return self.apply(value,True)


def metric_roots(gram,labels):
    groups=defaultdict(list)
    for i,label in enumerate(labels):groups[tuple(label['counts'])].append(i)
    rows=[];cols=[];roots=[];inverses=[];minimum=float('inf');maximum=0.;cross=gram.copy().tocoo()
    label_array=np.array([label['counts'] for label in labels]);bad=np.any(label_array[cross.row]!=label_array[cross.col],axis=1)
    if np.any(abs(cross.data[bad])>1e-12):raise AssertionError('Gram crosses physical count projectors')
    for indices in groups.values():
        g=gram[indices][:,indices].toarray();w,u=eigh(g);minimum=min(minimum,float(w[0]));maximum=max(maximum,float(w[-1]))
        if w[0]<=0:raise AssertionError('Nonpositive physical Gram')
        s=(u*np.sqrt(w))@u.conj().T;t=(u*(1/np.sqrt(w)))@u.conj().T
        r,c=np.indices(g.shape);rows.extend(np.asarray(indices)[r.ravel()]);cols.extend(np.asarray(indices)[c.ravel()]);roots.extend(s.ravel());inverses.extend(t.ravel())
    s=sparse.coo_matrix((roots,(rows,cols)),shape=gram.shape).tocsr();t=sparse.coo_matrix((inverses,(rows,cols)),shape=gram.shape).tocsr()
    identity=sparse.eye(gram.shape[0],format='csr')
    receipt=dict(count_blocks=len(groups),minimum_eigenvalue=minimum,maximum_eigenvalue=maximum,sqrt_metric_residual=float(sparse.linalg.norm(s@s-gram)),inverse_sqrt_metric_residual=float(sparse.linalg.norm(t@gram@t-identity)),Gram_minus_I=float(sparse.linalg.norm(gram-identity)))
    return s,t,receipt


def data_from_labels(labels):
    count=np.array([row['counts'] for row in labels],int).reshape(-1,4,4)
    colour=count[:,:,0]+count[:,:,2];white=count[:,:,1]+count[:,:,3]
    compact=np.all(np.sort(colour,axis=1)==[6,6,6,9],axis=1);fragment=np.all(np.sort(colour,axis=1)==[6,7,7,7],axis=1)
    return dict(counts=count,colour=colour,white=white,Q=(2*count[:,:,0]-count[:,:,2])/3-count[:,:,3],B=(colour-6)/3,C=compact,F=fragment,Csite=compact[:,None]&(colour==9))


def peierls(matrix,data,edge,derivative=0,theta=0.,charge='Q'):
    """Independently identify exact one-category transfer on predeclared bond."""
    start,end=((0,1),(1,2),(2,3),(0,3))[edge]
    coo=matrix.tocoo();count=data['counts'];change=count[coo.row]-count[coo.col]
    total_change=change.sum(axis=2)
    mask=(abs(total_change[:,end])==1)&(total_change[:,start]==-total_change[:,end])
    mask&=np.sum(abs(change),axis=(1,2))==2
    for site in range(4):
        if site not in (start,end):mask&=np.all(change[:,site]==0,axis=1)
    q=data[charge][coo.row,end]-data[charge][coo.col,end]
    factor=np.exp(1j*theta*q)-1 if derivative==0 else -1j*q if derivative==1 else -(q*q)
    return sparse.coo_matrix((coo.data[mask]*factor[mask],(coo.row[mask],coo.col[mask])),shape=matrix.shape).tocsr()


def evolve(operator,state,times):
    states=[];previous=0.;current=state.copy()
    for t in times:
        if t!=previous:current=expm_multiply((-1j*(t-previous))*operator,current,traceA=-1j*(t-previous)*operator.trace)
        states.append(current.copy());previous=t
    return states


def readout(state,data,operators,scalar,hamiltonian):
    n=len(data['C']);nf=len(scalar['grid']);z=state.reshape(n,nf);weights=np.sum(abs(z)**2,axis=1);phi=np.sum(abs(z)**2,axis=0);rho=z.conj().T@z
    norm=float(weights.sum());values=dict(norm=norm,compact=float(weights@data['C']),fragmented=float(weights@data['F']),compact_site=(weights@data['Csite']).tolist(),Qsite=(weights@data['Q']).tolist(),Bsite=(weights@data['B']).tolist(),colour_site=(weights@data['colour']).tolist(),white_site=(weights@data['white']).tolist(),Phi=float(phi@scalar['grid']),Phi2=float(phi@(scalar['grid']**2)),Phi_boundary=float(phi[0]+phi[-1]),Phi_purity=float(np.trace(rho@rho).real/norm**2),energy=float(np.vdot(state,hamiltonian@state).real))
    for name,matrix in operators.items():values[name]=float(np.vdot(z,matrix@z).real)
    return values


def count_distribution(state,data):
    weights=np.sum(abs(state.reshape(len(data['C']),-1))**2,axis=1);result=defaultdict(float)
    for counts,weight in zip(data['counts'],weights):result[tuple(counts.reshape(-1))]+=float(weight)
    return result


def total_variation(a,b):return sum(abs(a.get(key,0.)-b.get(key,0.)) for key in a.keys()|b.keys())/2
