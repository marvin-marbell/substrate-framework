"""One new RIGHT-gauged r^3 matrix-frame/native BODY/material/six-field law.
Primitive CAR spin is internal RIGHT only; field LEFT supplies physical spin.
No inherited state, gap, H, sigma.B or magnetic oscillator potential.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from rich_common_frame_orbit_20261015_native import build as frame_build
from rich_common_frame_orbit_20261015_frame import spin_basis,coupled_isometry,dressed_source
from rich_common_frame_orbit_20261015_model import FormOperator,source_tensors
from rich_spatial_colour_em_20261015_fields import field_shell,route_coefficients,endpoint_connections,phase_values,mechanical_kernels,gaussian_green
from rich_spatial_colour_charge_20261015_snapshot import load


def physical_phase(rotor,geometry,coefficients,field,edge,q,derivative=(),spin_axis=None):
    """Complex spherical Frame basis requires bra conjugation BEFORE both cuts."""
    c=coefficients['values'][:,edge];values=phase_values(c,q,field);factor=np.ones(len(c),complex)
    for axis in derivative:factor*=1j*q*c[:,axis]
    if spin_axis is not None:factor*=geometry['tangents'][:,edge,spin_axis]
    matrix=np.einsum('qr,q,qef,qs->resf',rotor['basis'].conj(),rotor['weights'],values*factor[:,None,None],rotor['basis'],optimize=True)
    size=rotor['dimension']*field['dimension']
    return matrix.reshape(size,size)


def build(p):
    model=frame_build(p);rotor=model['rotor'];field=field_shell(p['em_cut'],p['em_basis_scale']);nf=field['dimension'];nr=10
    ang=[sparse.kron(rotor['L'][i],sparse.eye(nf))+sparse.kron(sparse.eye(nr),field['F'][i]) for i in range(3)]
    rqspin=spin_basis(ang);u=sparse.csr_matrix(rqspin['U']);factory=model['sparse']
    model.update(physical_rotor=rotor,field=field,rqspin=rqspin,rqU=u,
        coefficients=route_coefficients(model['geometry'],rotor,p),connections=endpoint_connections(model['geometry'],rotor,p),phase_cache={},mechanical_cache={})
    model['rotor']=dict(basis=np.zeros((1,nr*nf)))
    model['Vself']=float(gaussian_green(0.,p['charge_sigma'],p['epsilon']));model['Vcross']=float(gaussian_green(p['vertex_separation'],p['charge_sigma'],p['epsilon']))
    payload=load(__import__('pathlib').Path(__file__).resolve().parent/p['native_body'])
    model['C_matter']=payload['C_matter']
    def rq(matrix):return factory.csr(u.conj().T@sparse.csr_matrix(matrix)@u)
    model['rq']=rq
    model['rqobs']={kind:[rq(sparse.kron(a,sparse.eye(nf)) if kind=='material' else sparse.kron(sparse.eye(nr),a)) for a in values] for kind,values in [('material',rotor['L']),('field',field['F']),('Q',field['Q']),('Pi',field['Pi'])]}
    model['rqobs']['HE']=rq(sparse.kron(sparse.eye(nr),field['H_E']));model['rqobs']['boundary']=rq(sparse.kron(sparse.eye(nr),field['boundary']))
    return model


def fibre(model,name,J,M):
    v=coupled_isometry(model['sectors'][name]['spin'],model['rqspin'],J,M)
    return sparse.kron(v,sparse.eye(model['radial'].count),format='csr')


def law(model,name,coupled=True,derivative=(),frame_factor=1.,route_derivative=None):
    p=model['parameters'];s=model['sectors'][name];op=s['op'];rot=model['physical_rotor'];nf=model['field']['dimension'];nrq=10*nf;rq=model['rq'];eye=np.eye(nrq);identity=sparse.eye(len(s['spin']['j']));terms=[];order=len(derivative)+(route_derivative is not None)
    def add(a,b,power=0):terms.append((a,b,power))
    if order==0:
        longitudinal=((model['Vself']+model['Vcross'])*(op['Q']@op['Q'])+(model['Vself']-model['Vcross'])*op['density_delta2'])/4
        add(op['electric']-p['plaquette']*op['B']+longitudinal,eye)
        add(identity,rq(sparse.kron(rot['L2']/(2*p['orientation_inertia']),sparse.eye(nf))+sparse.kron(sparse.eye(10),model['field']['H_E'])))
    if coupled and route_derivative is None:
        key=tuple(derivative)
        if key not in model['mechanical_cache']:model['mechanical_cache'][key]=mechanical_kernels(rot,model['field'],model['connections'],p['orientation_inertia'],derivative=derivative)
        mechanical=model['mechanical_cache'][key]
        for a,b in zip([op['Q'],op['density_delta']],mechanical['linear']):add(a,rq(b))
        for a,b in zip([op['Q']@op['Q'],op['Q']@op['density_delta'],op['density_delta2']],mechanical['quadratic']):add(a,rq(b))
    for edge in range(2):
        if route_derivative is not None and edge!=route_derivative:continue
        for charge,matrices in op['hops'][edge]:
            q=charge if coupled else 0.
            for axis in [None,0,1,2]:
                key=(edge,q,tuple(derivative),axis)
                if key not in model['phase_cache']:model['phase_cache'][key]=rq(physical_phase(rot,model['geometry'],model['coefficients'],model['field'],edge,q,derivative,axis))
                angle=model['phase_cache'][key]
                if route_derivative is not None:angle=1j*q*angle
                matter=(p['scalar_coupling']*matrices[0] if axis is None else 1j*p['vector_velocity']*matrices[axis+1])*model['geometry']['weights'][edge]
                # BOTH scalar and vector hopping are continuous matrix-amplitude r.
                add(matter,angle,1);add(matter.conj().T,angle.conj().T,1)
    if order==1:terms=[(-a,b,n) for a,b,n in terms]
    radial=None if order else {float(j):model['radial'].operators(float(j))['radial']+frame_factor*model['radial'].operators(float(j))['frame']+model['radial'].operators(float(j))['potential'] for j in set(s['spin']['j'])}
    return FormOperator(s,s,model['rotor'],model['radial'],terms,radial)


def observable(model,name,matter=None,rq=None):
    s=model['sectors'][name];d=len(s['spin']['j']);ng=len(model['rqspin']['j'])
    return FormOperator(s,s,model['rotor'],model['radial'],[(sparse.eye(d) if matter is None else matter,np.eye(ng) if rq is None else rq,0)])


def source(model,name,k,m,site):
    tensors=source_tensors(model,name,'lepton' if k==.5 else 'baryon',k,site)
    zero=model['sectors']['neutral'];one=model['sectors'][name]
    matrix,audit=dressed_source(zero,one,tensors,k,m)
    return FormOperator(zero,one,model['rotor'],model['radial'],[(matrix,np.eye(len(model['rqspin']['j'])),0)]),audit


def compress(operator,vt,vs=None):
    """Exact multiplicity x angular-channel x radial factors, never giant CSR."""
    vs=vt if vs is None else vs;nr=operator.nr;ng=operator.ng
    target=vt[::nr,::nr].tocsr();source_v=vs[::nr,::nr].tocsr()
    def layout(spin,v):
        result={}
        for j in sorted(set(spin['j'])):
            multiplets=[a for a in spin['multiplets'] if a['j']==j];maps=[];reference=None
            for a in multiplets:
                local=v[a['start']*ng:a['stop']*ng];columns=np.unique(local.indices);w=local[:,columns].tocsr()
                if reference is None:reference=w
                elif w.shape!=reference.shape or sparse.linalg.norm(w-reference)>1e-12:raise ValueError('Projector not multiplicity-separable')
                maps.append(columns)
            count=int(round(2*j+1));result[float(j)]=dict(indices=np.array(maps,dtype=int),W=[reference[i*ng:(i+1)*ng] for i in range(count)],multiplets=len(multiplets),channels=reference.shape[1])
        return result
    from rich_rotcov_gauge_20261015_sparse import SparseRoundoff
    factory=SparseRoundoff();tl=layout(operator.target['spin'],target);sl=layout(operator.source['spin'],source_v);terms=[]
    def add(left,right,matter,angle,radius):
        angle=factory.csr(angle)
        if matter.nnz and angle.nnz:terms.append([left,right,matter,angle,radius])
    for rows,cols,matter,angle,radius in operator.blocks:
        j=float(operator.target['spin']['j'][rows[0]]);k=float(operator.source['spin']['j'][cols[0]]);left=tl[j];right=sl[k];dj=int(round(2*j+1));dk=int(round(2*k+1))
        for mi in range(dj):
            for ni in range(dk):
                a=matter[mi::dj,ni::dk].tocsr()
                if not a.nnz:continue
                angular=left['W'][mi].conj().T@sparse.csr_matrix(angle)@right['W'][ni]
                add(left,right,a,angular,radius)
    if operator.field is not None:
        for j,radius in operator.field.items():
            left=tl[j];right=sl[j]
            angular=sum((a.conj().T@b for a,b in zip(left['W'],right['W'])),sparse.csr_matrix((left['channels'],right['channels']))).toarray()
            add(left,right,sparse.eye(left['multiplets'],right['multiplets'],format='csr'),angular,radius)
    class Projected(LinearOperator):
        def __init__(self):super().__init__(dtype=np.dtype(complex),shape=(target.shape[1]*nr,source_v.shape[1]*nr))
        def _apply(self,x,adjoint=False):
            x=np.asarray(x).reshape(target.shape[1] if adjoint else source_v.shape[1],nr);y=np.zeros((source_v.shape[1] if adjoint else target.shape[1],nr),complex)
            for left,right,a,b,c in terms:
                if adjoint:left,right,a,b,c=right,left,a.conj().T,b.conj().T,c.conj().T
                z=x[right['indices']];z=(a@z.reshape(right['multiplets'],-1)).reshape(left['multiplets'],right['channels'],nr)
                z=(b@z.transpose(1,0,2).reshape(right['channels'],-1)).reshape(left['channels'],left['multiplets'],nr).transpose(1,0,2);z=np.einsum('ian,mn->iam',z,c,optimize=True)
                y[left['indices']]+=z
            return y.ravel()
        def _matvec(self,x):return self._apply(x)
        def _rmatvec(self,x):return self._apply(x,True)
        def _matmat(self,x):return np.column_stack([self@x[:,i] for i in range(x.shape[1])])
    result=Projected();rng=np.random.default_rng(22215);x=rng.normal(size=result.shape[1])+1j*rng.normal(size=result.shape[1]);x/=np.linalg.norm(x)
    direct=vt.conj().T@(operator@(vs@x));actual=result@x
    return result,dict(factor_groups=len(terms),stored_multiplicity_nonzeros=sum(a.nnz for l,r,a,b,c in terms),angular_nonzeros=sum(b.nnz for l,r,a,b,c in terms),sparse_roundoff=factory.snapshot(),projection_action_relative_error=float(np.linalg.norm(direct-actual)/max(np.linalg.norm(direct),np.finfo(float).tiny)),construction='Exact multiplicity x sparse angular-channel x radial tensor factors, existing scale-relative roundoff ledger; no collapsed CSR, duplicated byte-key arrays or state/direction deletion.')


class DirectSum(LinearOperator):
    def __init__(self,blocks):
        self.blocks=blocks;self.ends=np.cumsum([0]+[x.shape[0] for x in blocks]);super().__init__(dtype=np.dtype(complex),shape=(int(self.ends[-1]),)*2)
    def _matvec(self,x):return np.concatenate([a@x[l:r] for a,l,r in zip(self.blocks,self.ends[:-1],self.ends[1:])])
    def _rmatvec(self,x):return np.concatenate([a.rmatvec(x[l:r]) for a,l,r in zip(self.blocks,self.ends[:-1],self.ends[1:])])
    def _matmat(self,x):return np.column_stack([self@x[:,i] for i in range(x.shape[1])])
