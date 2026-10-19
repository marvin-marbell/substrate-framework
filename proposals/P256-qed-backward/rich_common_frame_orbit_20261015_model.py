"""NEW common CAR/PW19 matter with quantum radial/Clifford-frame orientation.

The native band-pair sectors are a labelled conditional control, not the charge
orbit successor and not an integrated photon model. Primitive spin indices are
RIGHT gauge indices only; physical spin is actual LEFT field differentiation.
"""
from itertools import product
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from rich_common_colour_lepton_20261015_model import gauge_sector,mode,string,one_body
from rich_colour_vector_loop_20261015_model import masks_for
from rich_colour_composite_20261014_model import loop_field
from rich_spatial_colour_root_20261015_model import so3,graph,matter_ops,conjugate_sector,connection_data,phase_matrix,site_source,spatial_source_terms
from rich_rotcov_gauge_20261015_sparse import SparseRoundoff
from rich_common_frame_orbit_20261015_frame import RadialForm,spin_basis,gauss_frame,material_spin,dressed_source,coupled_isometry


def convert(spin,matrix,factory):
    u=spin['U'];return factory.csr(u.conj().T@(matrix@u))


def build(p):
    if p.get('representation')=='native_charge_orbit':
        from rich_common_frame_orbit_20261015_native import build as native_build
        return native_build(p)
    factory=SparseRoundoff();loop=loop_field();radial=RadialForm(p);native_rotor=so3(p);material=material_spin(native_rotor);um=material['U']
    def integrate(values):return factory.csr(um.conj().T@native_rotor['integrate'](values)@um)
    rotor=dict(native_rotor,L=material['S'],L2=sum(a@a for a in material['S']),basis=native_rotor['basis']@um,integrate=integrate)
    geometry=graph(p,rotor)
    natives={'neutral':gauge_sector(masks_for((0,0,0,0),p['pair_cut']),loop),'baryon':gauge_sector(masks_for((2,1,0,0),p['pair_cut']),loop),'lepton':gauge_sector(masks_for((0,0,0,1),p['pair_cut']),loop)}
    for name,anti in [('baryon','antibaryon'),('lepton','antilepton')]:natives[anti]=conjugate_sector(natives[name],loop)
    sectors={}
    for name,native in natives.items():
        old=matter_ops(native,loop,p,geometry,with_povm=False);spin=spin_basis(old['S'])
        op={key:convert(spin,old[key],factory) for key in ['electric','B','Q','N','density_delta','density_delta2','rho0','rho1','PW_boundary','matter_octet']}
        op['hops']=[[(q,[convert(spin,a,factory) for a in matrices]) for q,matrices in edge] for edge in old['hops']]
        op['S']=spin['S'];op['connection']=connection_data(op,rotor,p)
        sectors[name]=dict(native=native,spin=spin,op=op,colour_Gauss=native['gauge_error'],right_spin_completion=spin['completion'],native_hop_Gauss=old['raw_hop_Gauss_image_errors'])
    largest=max(max(s['spin']['j']) for s in sectors.values());frame_js=[x/2 for x in range(int(round(2*largest))+1)]
    for s in sectors.values():s['frame']=gauss_frame(s['spin'],frame_js)
    return dict(parameters=p,loop=loop,radial=radial,rotor=rotor,material=material,geometry=geometry,sectors=sectors,sparse=factory,frame_js=frame_js)


class FormOperator(LinearOperator):
    """Separable Galerkin products with j-dependent radial domains, no dense H."""
    def __init__(self,source,target,rotor,radial,terms=(),field=None):
        self.source=source;self.target=target;self.radial=radial;self.ng=len(rotor['basis'][0]);self.nr=radial.count
        self.ds=len(source['spin']['j']);self.dt=len(target['spin']['j']);self.blocks=[];self.field=field
        for matrix,orientation,power in terms:
            matrix=sparse.csr_matrix(matrix)
            for j in sorted(set(target['spin']['j'])):
                rows=np.flatnonzero(target['spin']['j']==j)
                for k in sorted(set(source['spin']['j'])):
                    cols=np.flatnonzero(source['spin']['j']==k);block=matrix[rows][:,cols].tocsr()
                    if block.nnz:self.blocks.append((rows,cols,block,np.asarray(orientation.toarray() if sparse.issparse(orientation) else orientation),radial.moment(float(j),float(k),power)))
        super().__init__(dtype=np.dtype(complex),shape=(self.dt*self.ng*self.nr,self.ds*self.ng*self.nr))

    def _apply(self,value,adjoint=False):
        x=np.asarray(value).reshape(self.dt if adjoint else self.ds,self.ng,self.nr);y=np.zeros((self.ds if adjoint else self.dt,self.ng,self.nr),complex)
        for rows,cols,mat,angle,radius in self.blocks:
            if adjoint:rows,cols,mat,angle,radius=cols,rows,mat.conj().T,angle.conj().T,radius.conj().T
            moved=(mat@x[cols].reshape(len(cols),-1)).reshape(len(rows),self.ng,self.nr)
            moved=np.einsum('ab,ibn->ian',angle,moved,optimize=True)
            y[rows]+=np.einsum('ian,mn->iam',moved,radius,optimize=True)
        if self.field is not None:
            for j in sorted(set(self.source['spin']['j'])):
                rows=np.flatnonzero(self.source['spin']['j']==j);matrix=self.field[float(j)]
                y[rows]+=np.einsum('ian,mn->iam',x[rows],matrix.conj().T if adjoint else matrix,optimize=True)
        return y.ravel()
    def _matvec(self,x):return self._apply(x)
    def _rmatvec(self,x):return self._apply(x,True)
    def _matmat(self,x):return np.column_stack([self._matvec(x[:,i]) for i in range(x.shape[1])])
    def diagonal(self):
        diagonal=np.zeros((self.ds,self.ng,self.nr),complex)
        for rows,cols,mat,angle,radius in self.blocks:
            if np.array_equal(rows,cols):diagonal[rows]+=mat.diagonal()[:,None,None]*np.diag(angle)[None,:,None]*np.diag(radius)[None,None,:]
        if self.field is not None:
            for j,matrix in self.field.items():diagonal[self.source['spin']['j']==j]+=np.diag(matrix)[None,None,:]
        return diagonal.ravel()
    def trace(self):return np.sum(self.diagonal())


def law(model,name,fields=None,derivative=(),frame_factor=1.,vector_factor=1.,hop_factor=1.):
    p=model['parameters'];s=model['sectors'][name];op=s['op'];rotor=model['rotor'];g=model['geometry'];factory=model['sparse'];d=len(s['spin']['j']);identity=sparse.eye(d,format='csr');ng=10;orient_identity=np.eye(ng)
    fields=np.zeros(6) if fields is None else np.asarray(fields,float);order=len(derivative);terms=[]
    def add(matrix,orientation,power=0,coefficient=1.):
        if coefficient:terms.append((factory.csr(coefficient*matrix),factory.csr(orientation),power))
    if order==0:
        add(op['electric']-p['plaquette']*op['B'],orient_identity)
        add(identity,rotor['L2']/(2*p['orientation_inertia']))
    connection=op['connection']
    for k in range(6):
        coefficient=fields[k] if order==0 else (1. if order==1 and derivative[0]==k else 0.)
        if coefficient:
            matter=op['density_delta'] if k<3 else op['Q']
            for axis in range(3):
                alpha=rotor['integrate'](connection['values'][k,axis]);angular=rotor['L'][axis]@alpha+alpha@rotor['L'][axis]
                add(matter,angular,coefficient=-coefficient/(2*p['orientation_inertia']))
    for k in range(6):
        for l in range(6):
            coefficient=fields[k]*fields[l] if order==0 else (2*fields[l] if order==1 and derivative[0]==k else (2. if order==2 and derivative==(k,l) else 0.))
            if coefficient:
                kind=int(k>=3)+int(l>=3);moment=rotor['integrate'](np.sum(connection['values'][k]*connection['values'][l],axis=0))
                add(connection['products'][kind],moment,coefficient=coefficient/(2*p['orientation_inertia']))
    for edge in range(2):
        scale=g['weights'][edge]*hop_factor
        for q,matrices in op['hops'][edge]:
            angle=phase_matrix(rotor,g,edge,q,fields,derivative)
            add(matrices[0],angle,1,p['scalar_coupling']*scale);add(matrices[0].conj().T,angle.conj().T,1,p['scalar_coupling']*scale)
            for axis in range(3):
                angle=phase_matrix(rotor,g,edge,q,fields,derivative,axis);coefficient=1j*p['vector_velocity']*scale*vector_factor
                # NEW continuous Phi law: the vector route also carries r.
                add(matrices[axis+1],angle,1,coefficient);add(matrices[axis+1].conj().T,angle.conj().T,1,coefficient.conjugate())
    if order==1:terms=[(-a,b,power) for a,b,power in terms]
    field=None
    if order==0:field={float(j):model['radial'].operators(float(j))['radial']+frame_factor*model['radial'].operators(float(j))['frame']+model['radial'].operators(float(j))['potential'] for j in set(s['spin']['j'])}
    result=FormOperator(s,s,rotor,model['radial'],terms,field);result.terms=terms
    return result


def observable(model,name,matter=None,orientation=None,power=0,field_kind=None):
    s=model['sectors'][name];matter=sparse.eye(len(s['spin']['j'])) if matter is None else matter;orientation=np.eye(10) if orientation is None else orientation
    field=None if field_kind is None else {float(j):model['radial'].operators(float(j))[field_kind] for j in set(s['spin']['j'])}
    return FormOperator(s,s,model['rotor'],model['radial'],[] if field_kind else [(matter,orientation,power)],field)


def coupled(model,name,J,M=None):
    s=model['sectors'][name];M=J if M is None else M
    body=coupled_isometry(s['spin'],model['material'],J,M)
    return sparse.kron(body,sparse.eye(model['radial'].count),format='csr')


def reduced(operator,isometry):
    class Coupled(LinearOperator):
        def __init__(self):super().__init__(dtype=np.dtype(complex),shape=(isometry.shape[1],)*2)
        def _matvec(self,x):return isometry.conj().T@(operator@(isometry@x))
        def _rmatvec(self,x):return isometry.conj().T@(operator.H@(isometry@x))
        def _matmat(self,x):return np.column_stack([self._matvec(x[:,i]) for i in range(x.shape[1])])
    return Coupled()


def source_tensors(model,name,kind,J,site=0):
    neutral=model['sectors']['neutral'];target=model['sectors'][name];removal=name.startswith('anti');factory=model['sparse']
    if 'native_source_matter' in model:
        tensors={}
        for label,matrix in model['native_source_matter'][name].items():
            placement,j,m=label.split(':')
            if placement==str(site) and float(j)==J:tensors[float(m)]=factory.csr(target['spin']['U'].conj().T@matrix@neutral['spin']['U'])
        if len(tensors)!=int(2*J+1):raise ValueError('Published native source does not contain the complete required RIGHT multiplet')
        return tensors
    seed=site_source(neutral['native'],target['native'],site,kind,J,removal)
    seed=factory.csr(target['spin']['U'].conj().T@(seed@neutral['spin']['U']))
    plus_t=target['spin']['S'][0]+1j*target['spin']['S'][1];plus_n=neutral['spin']['S'][0]+1j*neutral['spin']['S'][1]
    current=seed;m=.5
    while m<J-1e-8:
        current=(plus_t@current-current@plus_n)/np.sqrt((J-m)*(J+m+1));m+=1
    tensors={float(J):factory.csr(current)};minus_t=plus_t.conj().T;minus_n=plus_n.conj().T
    while m>-J+1e-8:
        current=(minus_t@current-current@minus_n)/np.sqrt((J+m)*(J-m+1));m-=1;tensors[float(m)]=factory.csr(current)
    return tensors


def source(model,name,kind,J,M=None,site=0):
    M=J if M is None else M;tensors=source_tensors(model,name,kind,J,site)
    neutral=model['sectors']['neutral'];target=model['sectors'][name]
    dressed,readout=dressed_source(neutral,target,tensors,J,float(M))
    operator=FormOperator(neutral,target,model['rotor'],model['radial'],[(dressed,np.eye(10),0)])
    operator.matrix=dressed;operator.tensors=tensors;operator.readout=readout
    return operator


def full_raw_source_gram(model,name,kind,J,M=None,site=0):
    """No target pair/radial projector: complete polynomial image Gram."""
    M=J if M is None else M;neutral=model['sectors']['neutral'];target=model['sectors'][name];removal=name.startswith('anti')
    if 'native_source_metric_full' in model:
        candidates=[matrix for label,matrix in model['native_source_metric_full'][name].items() if label.split(':')[0]==str(site) and float(label.split(':')[1])==J and float(label.split(':')[2])==M]
        if len(candidates)!=1:raise ValueError('Exact full raw native source metric is not uniquely supplied')
        gram=model['sparse'].csr(neutral['spin']['U'].conj().T@candidates[0]@neutral['spin']['U'])
        return FormOperator(neutral,neutral,model['rotor'],model['radial'],[(gram,np.eye(10),0)]),None
    # Build the supplied M=+1/2 CAR polynomial in an augmented mask image,
    # then form all tensor components by raw right-spin commutators if needed.
    terms=spatial_source_terms(site,kind,J,removal);source_masks=neutral['native']['masks'];targets=set()
    for mask in source_masks:
        for operations,coefficient in terms:
            step=string(mask,operations)
            if step is not None and coefficient:targets.add(step[0])
    # Closing the image under spin keeps all tensor components and makes the
    # unprojected norm physical without assuming fixed spin-axis populations.
    sigma=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1.,-1.])]
    spin_terms=[[(mode(f,m,a,c),mode(f,m,b,c),s[a,b]/2) for f,m,a,b,c in product(range(2),range(2),range(2),range(2),range(4)) if s[a,b]!=0] for s in sigma]
    while True:
        added=set(targets)
        for mask in targets:
            for row in spin_terms:
                for a,b,value in row:
                    step=string(mask,[(b,False),(a,True)])
                    if step is not None and value:added.add(step[0])
        if added==targets:break
        targets=added
    targets=sorted(targets);lookup={m:i for i,m in enumerate(targets)};rows=[];cols=[];values=[]
    for column,mask in enumerate(source_masks):
        for operations,coefficient in terms:
            step=string(mask,operations)
            if step is not None:rows.append(lookup[step[0]]);cols.append(column);values.append(coefficient*step[1])
    raw=sparse.csr_matrix((values,(rows,cols)),shape=(len(targets),len(source_masks)))
    image=sparse.kron(raw,sparse.eye(19),format='csr')@neutral['native']['P'];image=image@neutral['spin']['U']
    sz_target=sparse.kron(one_body(targets,spin_terms[2]),sparse.eye(19),format='csr');plus_target=sparse.kron(one_body(targets,spin_terms[0])+1j*one_body(targets,spin_terms[1]),sparse.eye(19),format='csr')
    plus_n=neutral['spin']['S'][0]+1j*neutral['spin']['S'][1];m=.5
    while m<J-1e-8:image=(plus_target@image-image@plus_n)/np.sqrt((J-m)*(J+m+1));m+=1
    while m>M+1e-8:image=(plus_target.conj().T@image-image@plus_n.conj().T)/np.sqrt((J+m)*(J-m+1));m-=1
    gram=model['sparse'].csr(image.conj().T@image)
    return FormOperator(neutral,neutral,model['rotor'],model['radial'],[(gram,np.eye(10),0)]),len(targets)


def route_current(model,name,edge):
    """Physical -dH/dtheta_edge from the original Peierls source, NOT a commutator."""
    p=model['parameters'];s=model['sectors'][name];terms=[];scale=model['geometry']['weights'][edge];rotor=model['rotor'];g=model['geometry'];factory=model['sparse']
    for q,matrices in s['op']['hops'][edge]:
        angle=phase_matrix(rotor,g,edge,q,np.zeros(6))
        terms.append((factory.csr(-1j*q*p['scalar_coupling']*scale*matrices[0]),angle,1))
        terms.append((factory.csr(1j*q*p['scalar_coupling']*scale*matrices[0].conj().T),angle.conj().T,1))
        for axis in range(3):
            angle=phase_matrix(rotor,g,edge,q,np.zeros(6),axis=axis);coefficient=q*p['vector_velocity']*scale
            terms.append((factory.csr(coefficient*matrices[axis+1]),angle,1));terms.append((factory.csr(coefficient*matrices[axis+1].conj().T),angle.conj().T,1))
    return FormOperator(s,s,rotor,model['radial'],terms)
