"""Positive ordered Clifford orbit: r³ measure, field LEFT spin, RIGHT Gauss.

Right gauging is CHOSEN through a time-local a0 constraint, not derived from a
global symmetry or the metric. The metric fixes inertia, not an arbitrary j
penalty. All source and radial products are integrated before projection.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy import sparse
from scipy.special import eval_genlaguerre,gammaln,roots_genlaguerre
from rich_frame_gauss_20261014_model import angular,cg
from rich_rotcov_gauge_20261015_sparse import SparseRoundoff


class RadialForm:
    """Friedrichs quadratic form; j-adapted regular Laguerre functions on r>0.

    f_nj=r^beta L_n^(beta+1)(a r²)e^(-a r²/2),
    beta=-1+sqrt(1+2j(j+1)). No r=epsilon or inverse-radius clamp exists.
    a is a Galerkin scale: its auxiliary oscillator potential is SUBTRACTED.
    """
    def __init__(self,p):
        self.count=p['radial_basis'];self.scale=p['radial_scale'];self.mass=16*p['matrix_metric'];self.quartic=p['scalar_quartic'];self.quadratic=p['scalar_quadratic'];self.quadrature=p['radial_quadrature']

    @staticmethod
    def beta(j):return -1+np.sqrt(1+2*j*(j+1))

    @lru_cache(None)
    def moment(self,j,k,power):
        bj,bk=self.beta(j),self.beta(k);alpha=1+(bj+bk+power)/2
        nodes,weights=roots_genlaguerre(self.quadrature,alpha)
        n=np.arange(self.count);nj=np.exp((gammaln(n+1)-gammaln(n+bj+2))/2);nk=np.exp((gammaln(n+1)-gammaln(n+bk+2))/2)
        left=np.array([eval_genlaguerre(i,bj+1,nodes) for i in n]);right=np.array([eval_genlaguerre(i,bk+1,nodes) for i in n])
        return self.scale**(-power/2)*nj[:,None]*(left*weights[None,:])@right.T*nk[None,:]

    @lru_cache(None)
    def operators(self,j):
        n=np.arange(self.count);r2=self.moment(j,j,2);r4=self.moment(j,j,4);inverse=self.moment(j,j,-2)
        total_kinetic=np.diag(self.scale/self.mass*(2*n+self.beta(j)+2))-self.scale**2*r2/(2*self.mass)
        frame=j*(j+1)*inverse/self.mass;radial=total_kinetic-frame;potential=self.quartic*r4-self.quadratic*r2
        return dict(H=total_kinetic+potential,kinetic=total_kinetic,radial=radial,frame=frame,potential=potential,r=self.moment(j,j,1),r2=r2,inverse_r2=inverse)

    def values(self,j,r):
        beta=self.beta(j);t=self.scale*np.asarray(r)**2;n=np.arange(self.count)
        norms=np.sqrt(2)*self.scale**((beta+2)/2)*np.exp((gammaln(n+1)-gammaln(n+beta+2))/2)
        return np.array([norms[i]*np.asarray(r)**beta*eval_genlaguerre(i,beta+1,t)*np.exp(-t/2) for i in n]).T


def spin_basis(spins,commuting_charge=None):
    """All internal right-spin multiplets, with phases fixed by lowering."""
    if commuting_charge is not None:
        charges=np.asarray(commuting_charge);dimension=len(charges);factory=SparseRoundoff();matrices=[sparse.csr_matrix(a) for a in spins]
        off_charge=max(float(np.linalg.norm(a.tocoo().data[charges[a.tocoo().row]!=charges[a.tocoo().col]])) for a in matrices)
        if off_charge>1e-8:raise ValueError(f'Native charge does not commute chosen RIGHT Spin: {off_charge}')
        rows=[];cols=[];values=[];labels=[];multiplets=[];blocks=[];offset=0
        for h in sorted(set(charges)):
            selected=np.flatnonzero(charges==h);part=spin_basis([a[selected][:,selected] for a in matrices])
            if not part['completion']:raise ValueError('Native right-spin packet is not a complete SU2 representation')
            local=factory.csr(part['U']).tocoo();rows.extend(selected[local.row]);cols.extend(offset+local.col);values.extend(local.data)
            shift=len(multiplets)
            labels.extend((identifier+shift,j,m) for identifier,j,m in part['labels'])
            multiplets.extend(dict(id=x['id']+shift,j=x['j'],start=x['start']+offset,stop=x['stop']+offset,charge3=int(h)) for x in part['multiplets'])
            blocks.append(dict(charge3=int(h),dimension=len(selected),unitarity=part['unitarity']))
            offset+=len(selected)
        u=factory.csr(sparse.csr_matrix((values,(rows,cols)),shape=(dimension,dimension)))
        converted=[factory.csr(u.conj().T@a@u) for a in matrices]
        return dict(U=u,labels=labels,multiplets=multiplets,S=converted,j=np.array([x[1] for x in labels]),m=np.array([x[2] for x in labels]),unitarity=float(sparse.linalg.norm(u.conj().T@u-sparse.eye(dimension))),completion=offset==dimension,sparse_roundoff=factory.snapshot(),charge_blocks=blocks,right_spin_charge_commutator=off_charge)
    matrices=[a.toarray() if sparse.issparse(a) else np.asarray(a) for a in spins];dimension=len(matrices[0]);casimir=sum(a@a for a in matrices)
    mz,w=eigh(matrices[2]);lower=matrices[0]-1j*matrices[1];columns=[];labels=[];multiplets=[]
    for twice in range(int(round(2*np.max(mz)))+1):
        j=twice/2;selection=abs(mz-j)<1e-7
        if not np.any(selection):continue
        space=w[:,selection];values,vectors=eigh(space.conj().T@casimir@space)
        for column in np.flatnonzero(abs(values-j*(j+1))<1e-7):
            top=space@vectors[:,column];steps=[top]
            for m in np.arange(j,-j,-1):steps.append(lower@steps[-1]/np.sqrt(j*(j+1)-m*(m-1)))
            start=len(columns);identifier=len(multiplets)
            for m,v in zip(np.arange(-j,j+1),reversed(steps)):columns.append(v);labels.append((identifier,j,float(m)))
            multiplets.append(dict(id=identifier,j=j,start=start,stop=len(columns)))
    u=np.column_stack(columns);factory=SparseRoundoff();converted=[factory.csr(u.conj().T@a@u) for a in matrices]
    return dict(U=u,labels=labels,multiplets=multiplets,S=converted,j=np.array([x[1] for x in labels]),m=np.array([x[2] for x in labels]),unitarity=float(np.linalg.norm(u.conj().T@u-np.eye(dimension))),completion=len(columns)==dimension,sparse_roundoff=factory.snapshot())


def gauss_frame(spin,js):
    """Actual raw L²(SU2) x CAR isometry, normalized D* Peter-Weyl basis.

    P|alpha,j,M> = sum_n |j,M,n> tensor |alpha,j,n>/sqrt(2j+1).
    RIGHT generator=-s_j^T, CAR right=+s_j; LEFT generator=+s_j.
    """
    basis=[(j,float(m),float(n)) for j in js for m in np.arange(-j,j+1) for n in np.arange(-j,j+1)];lookup={x:i for i,x in enumerate(basis)};d=len(spin['labels']);rows=[];cols=[];values=[]
    for multiplet in spin['multiplets']:
        j=multiplet['j'];start=multiplet['start']
        for mi,m in enumerate(np.arange(-j,j+1)):
            for ni,n in enumerate(np.arange(-j,j+1)):
                rows.append(lookup[(j,float(m),float(n))]*d+start+ni);cols.append(start+mi);values.append(1/np.sqrt(2*j+1))
    projector=sparse.csr_matrix((values,(rows,cols)),shape=(len(basis)*d,d))
    left=[sparse.block_diag([sparse.kron(angular(j)[a],sparse.eye(int(2*j+1))) for j in js],format='csr') for a in range(3)]
    right=[sparse.block_diag([sparse.kron(sparse.eye(int(2*j+1)),-angular(j)[a].T) for j in js],format='csr') for a in range(3)]
    gauss=[sparse.kron(a,sparse.eye(d))+sparse.kron(sparse.eye(len(basis)),sparse.csr_matrix(b)) for a,b in zip(right,spin['S'])]
    compressed=[projector.conj().T@sparse.kron(a,sparse.eye(d))@projector for a in left]
    return dict(basis=basis,P=projector,L=left,R=right,G=gauss,left=compressed,Gauss=max(float(sparse.linalg.norm(a@projector)) for a in gauss),isometry=float(sparse.linalg.norm(projector.conj().T@projector-sparse.eye(d))),left_equals_reduced=max(float(sparse.linalg.norm(a-b)) for a,b in zip(compressed,spin['S'])))


def wigner_multiplier(source_frame,target_frame,k,K,q):
    """P D^(k)*_(Kq)(g) P with normalized Haar and complete M/right indices."""
    rows=[];cols=[];values=[];lookup={x:i for i,x in enumerate(target_frame['basis'])}
    for column,(j,m,n) in enumerate(source_frame['basis']):
        for jp in sorted(set(x[0] for x in target_frame['basis'])):
            key=(jp,float(m+K),float(n+q))
            if key in lookup:
                coefficient=np.sqrt((2*j+1)/(2*jp+1))*cg(j,m,k,K,jp)*cg(j,n,k,q,jp)
                if coefficient:rows.append(lookup[key]);cols.append(column);values.append(coefficient)
    return sparse.csr_matrix((values,(rows,cols)),shape=(len(target_frame['basis']),len(source_frame['basis'])))


def dressed_source(source,target,tensors,k,K):
    """Actual Wigner/CAR source, not an assumed reduced CAR/spin insertion."""
    zero=source['frame'];one=target['frame'];d0=len(source['spin']['labels']);d1=len(target['spin']['labels']);image=sparse.csr_matrix((len(one['basis'])*d1,d0),dtype=complex)
    for q,matrix in tensors.items():
        image+=sparse.kron(wigner_multiplier(zero,one,k,K,q),matrix,format='csr')@zero['P']
    reduced=one['P'].conj().T@image
    # Bare CAR source with unchanged frame coordinate uses identity between
    # common Peter-Weyl labels; its Gauss defect is a meaningful wrong-source control.
    common={x:i for i,x in enumerate(one['basis'])};rows=[];cols=[]
    for i,x in enumerate(zero['basis']):
        if x in common:rows.append(common[x]);cols.append(i)
    identity=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(one['basis']),len(zero['basis'])))
    bare_image=sparse.kron(identity,tensors[K],format='csr')@zero['P']
    return reduced.tocsr(),dict(raw_source_Gauss=max(float(sparse.linalg.norm(g@image)) for g in one['G']),reduced_matches_CAR=float(sparse.linalg.norm(reduced-tensors[K])),source_frame_projection_leak=float(sparse.linalg.norm(image-one['P']@reduced)),bare_without_Wigner_Gauss=max(float(sparse.linalg.norm(g@bare_image)) for g in one['G']))


def material_spin(rotor):
    return spin_basis(rotor['L'])


def coupled_isometry(spin,material,J,M):
    """One complete (J,M) symmetry fibre, never a fitted Casimir penalty."""
    d=len(spin['labels']);ng=len(material['labels']);rows=[];cols=[];values=[];column=0
    for a in spin['multiplets']:
        j=a['j']
        for b in material['multiplets']:
            l=b['j']
            if J<abs(j-l)-1e-8 or J>j+l+1e-8 or abs(M)>J:continue
            for mi,m in enumerate(np.arange(-j,j+1)):
                n=M-m
                if abs(n)<=l+1e-8:
                    ni=int(round(n+l));value=cg(j,float(m),l,float(n),J)
                    if value:rows.append((a['start']+mi)*ng+b['start']+ni);cols.append(column);values.append(value)
            column+=1
    return sparse.csr_matrix((values,(rows,cols)),shape=(d*ng,column))


def coordinate_noether(sector,number):
    """Differentiate ACTUAL g coordinate wavefunctions, not an assigned spin."""
    from rich_frame_gauss_20261014_model import rotation
    frame=sector['frame'];d=len(sector['spin']['labels']);rng=np.random.default_rng(12231)
    vector=rng.normal(size=d)+1j*rng.normal(size=d);vector/=np.linalg.norm(vector)
    def wave(alpha,beta,gamma,v,centre=False):
        coefficients=(frame['P']@v).reshape(len(frame['basis']),d)
        matrices={j:rotation(j,alpha,beta,gamma).conj() for j in set(x[0] for x in frame['basis'])}
        values=np.array([np.sqrt(2*j+1)*matrices[j][int(round(m+j)),int(round(n+j))]*((-1)**int(round(2*j)) if centre else 1) for j,m,n in frame['basis']])
        return values@coefficients
    a,b,c=.37,.52,-.23;step=1e-5
    differentiated=1j*(wave(a-step,b,c,vector)-wave(a+step,b,c,vector))/(2*step)
    expected=wave(a,b,c,sector['spin']['S'][2]@vector)
    return dict(actual_left_g_derivative=float(np.linalg.norm(differentiated-expected)),centre_field_sewing=float(np.linalg.norm(wave(a,b,c,vector,True)-(-1)**number*wave(a,b,c,vector))),primitive_CAR_physical_spin='ZERO in this variant; S_CAR enters RIGHT Gauss only; physical generator is field LEFT differentiation compressed by the actual isometry')


def radial_form_readout(radial,j):
    """Independent weighted derivative quadratic form, with no origin clamp."""
    beta=radial.beta(j);n=np.arange(radial.count);nodes,weights=roots_genlaguerre(max(radial.quadrature,radial.count+2),beta)
    norms=np.exp((gammaln(n+1)-gammaln(n+beta+2))/2)
    lag=np.array([eval_genlaguerre(i,beta+1,nodes) for i in n])
    derivative=np.array([-eval_genlaguerre(i-1,beta+2,nodes) if i else np.zeros(len(nodes)) for i in n])
    polynomial=beta*lag+2*nodes[None,:]*derivative-nodes[None,:]*lag
    gradient=radial.scale/(2*radial.mass)*norms[:,None]*(polynomial*weights[None,:])@polynomial.T*norms[None,:]
    frame=j*(j+1)/radial.mass*radial.moment(j,j,-2);actual=radial.operators(j)
    return dict(j=float(j),origin_exponent=float(beta),measure='r^3 dr dHaar',radial_Gram_error=float(np.linalg.norm(radial.moment(j,j,0)-np.eye(radial.count))),derivative_quadratic_form_vs_LB=float(np.linalg.norm(gradient+frame-actual['kinetic'])),frame_inertia='M_rad*r^2/2',no_origin_epsilon=True,trial_domain='j-adapted regular Friedrichs branch r^beta_j times Laguerre polynomial/Gaussian; positive half-line ordered branch only')
