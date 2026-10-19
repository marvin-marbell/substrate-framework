"""One four-position occupied CAR36/radial/frame/colour/EM law.

The extra material cell has four colourless spin modes and ALL its Fock states.
Root CAR32/PW19 native packets are architecture/isometries, never old states.
"""
from itertools import product
from pathlib import Path
import hashlib
import numpy as np
from scipy import sparse
from rich_spatial_colour_charge_20261015_snapshot import load
from rich_spatial_colour_charge_20261015_model import insert,charge_grams
from rich_common_colour_lepton_20261015_model import string,mode
from rich_common_frame_orbit_20261015_frame import RadialForm,spin_basis,material_spin,coupled_isometry
from rich_common_frame_orbit_20261015_model import FormOperator
from rich_frame_field_join_20261015_model import compress
from rich_spatial_colour_root_20261015_model import so3,graph
from rich_spatial_colour_em_20261015_fields import field_shell,profiles,phase_values,gaussian_green
from rich_rotcov_gauge_20261015_sparse import SparseRoundoff
from rich_frame_gauss_20261014_model import angular

ROOT=Path(__file__).resolve().parent
NROOT={'antilepton':15,'neutral':16,'lepton':17}
SIGMA=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1.,-1.])]


def extra_operator(left,right,terms):
    lookup={m:i for i,m in enumerate(left)};rr=[];cc=[];vv=[]
    for j,mask in enumerate(right):
        for operations,value in terms:
            result=string(mask,operations)
            if result is not None and result[0] in lookup:
                rr.append(lookup[result[0]]);cc.append(j);vv.append(value*result[1])
    return sparse.csr_matrix((vv,(rr,cc)),shape=(len(left),len(right)))


def combined_spin(root,ext):
    """CG change of basis, preserving every multiplicity and M, no dense solve."""
    columns=[];labels=[];multiplets=[];offset=0
    largest=max(root['j'])+max(ext['j'])
    parity=(int(round(2*max(root['j'])))+int(round(2*max(ext['j']))))%2
    for twice in range(parity,int(round(2*largest))+1,2):
        j=twice/2
        fibres=[coupled_isometry(root,ext,j,float(m)) for m in np.arange(-j,j+1)]
        count=fibres[0].shape[1]
        for a in range(count):
            start=offset;identifier=len(multiplets)
            for m,v in zip(np.arange(-j,j+1),fibres):
                columns.append(v[:,a]);labels.append((identifier,j,float(m)));offset+=1
            multiplets.append(dict(id=identifier,j=j,start=start,stop=offset))
    u=sparse.hstack(columns,format='csr')
    return dict(U=u,labels=labels,multiplets=multiplets,j=np.array([x[1] for x in labels]),m=np.array([x[2] for x in labels]))


def build(p):
    path=ROOT/p['native_body']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=p['native_body_sha256']:raise ValueError('Pinned BODY changed')
    data=load(path);factory=SparseRoundoff();radial=RadialForm(p)
    rawrot=so3(p);material=material_spin(rawrot);um=material['U']
    rotor=dict(rawrot,L=material['S'],L2=sum(a@a for a in material['S']),basis=rawrot['basis']@um)
    rotor['integrate']=lambda values:factory.csr(rotor['basis'].conj().T@((rotor['weights']*values)[:,None]*rotor['basis']))
    field=field_shell(p['em_cut'],p['em_basis_scale']);nf=field['dimension']
    rqspin=spin_basis([sparse.kron(rotor['L'][a],sparse.eye(nf))+sparse.kron(sparse.eye(10),field['F'][a]) for a in range(3)])
    ru=sparse.csr_matrix(rqspin['U']);rq=lambda a:factory.csr(ru.conj().T@sparse.csr_matrix(a)@ru)
    roots={}
    for name in NROOT:
        op=data['operators'][name];sp=spin_basis(op['S'],data['sectors'][name]['charge3_site0']);u=sp['U']
        roots[name]=dict(spin=sp,op=op,U=u)
    masks={n:[m for m in range(16) if m.bit_count()==n] for n in range(5)};ext={}
    for n,ms in masks.items():
        spins=[extra_operator(ms,ms,[([(2*s+r,True),(2*s+t,False)],sigma[r,t]/2) for s,r,t in product(range(2),repeat=3) if sigma[r,t]!=0]) for sigma in SIGMA]
        ext[n]=spin_basis(spins)
    # Direct raw CAR contractions across root charge sectors. Both annihilator
    # and creator directions use exact native P_h, not compressed exponentials.
    creation={};rawmetrics={}
    for source,target in [('antilepton','neutral'),('neutral','lepton')]:
        bs=data['native'][source];bt=data['native'][target]
        for site,spin in product(range(2),repeat=2):
            terms=[([(mode(1,m,spin,3),True)],(-1)**(site*m)/np.sqrt(2)) for m in range(2)]
            images=[insert(seed,terms) for seed in bs.seeds]
            a=bs.projected(images,bt,-3 if site==0 else 0)
            creation[source,target,site,spin]=roots[target]['U'].conj().T@a@roots[source]['U']
            rawmetrics[source,target,site,spin]=bs.image_metric(images,-3 if site==0 else 0)
    sectors={}
    for name,total in [('neutral',18),('lepton',19),('antilepton',17)]:
        blocks=[];offset=0
        for root,nr in NROOT.items():
            n=total-nr
            if n not in masks:continue
            sp=combined_spin(roots[root]['spin'],ext[n]);u=sparse.kron(roots[root]['U'],sparse.csr_matrix(ext[n]['U']),format='csr')@sp['U']
            blocks.append(dict(root=root,n=n,spin=sp,U=u,start=offset,stop=offset+u.shape[1]));offset+=u.shape[1]
        labels=[];multiplets=[]
        for b in blocks:
            shift=len(multiplets)
            labels.extend((a+shift,j,m) for a,j,m in b['spin']['labels'])
            multiplets.extend(dict(id=x['id']+shift,j=x['j'],start=x['start']+b['start'],stop=x['stop']+b['start']) for x in b['spin']['multiplets'])
        sp=dict(labels=labels,j=np.array([x[1] for x in labels]),m=np.array([x[2] for x in labels]),multiplets=multiplets)
        sectors[name]=dict(spin=sp,blocks=blocks,dimension=offset,total_N=total)
    def assemble(name,fn):
        ss=sectors[name];rows=[]
        for left in ss['blocks']:
            row=[]
            for right in ss['blocks']:
                a=fn(left,right)
                row.append(sparse.csr_matrix((left['stop']-left['start'],right['stop']-right['start'])) if a is None else factory.csr(left['U'].conj().T@a@right['U']))
            rows.append(row)
        return sparse.bmat(rows,format='csr')
    def local_root(key):
        return lambda l,r:sparse.kron(data['operators'][l['root']][key],sparse.eye(len(masks[l['n']])),format='csr') if l is r else None
    for name,s in sectors.items():
        op={key:assemble(name,local_root(key)) for key in ['electric','B','Q','rho0','rho1','PW_boundary']}
        op['S']=[sparse.block_diag([angular(a['j'])[axis] for a in s['spin']['multiplets']],format='csr') for axis in range(3)]
        rho=[op['rho0'],op['rho1']]
        for site in range(2):
            rho.append(assemble(name,lambda l,r,site=site:sparse.kron(sparse.eye(data['sectors'][l['root']]['dimension']),sparse.diags([1-sum(bool(m&(1<<(2*site+t))) for t in range(2)) for m in masks[l['n']]]),format='csr') if l is r else None))
        op['rho']=rho;op['Q']=sum(rho);op['hops']=[]
        for edge in range(2):
            op['hops'].append([(q,[assemble(name,lambda l,r,a=a:sparse.kron(a[l['root']],sparse.eye(len(masks[l['n']])),format='csr') if l is r else None) for a in [{root:data['operators'][root]['hops'][edge][ch][1][axis] for root in NROOT} for axis in range(4)]]) for ch,q in enumerate([2/3,0.,-1/3,-1.])])
        # New bridge 1 -> 2 and new-cell internal bond 2 -> 3.
        for edge in [2,3]:
            matrices=[]
            for sigma in [np.eye(2)]+SIGMA:
                def primitive(l,r,sigma=sigma,edge=edge):
                    if edge==3:
                        if l is not r:return None
                        a=extra_operator(masks[l['n']],masks[r['n']],[([(2+s0,True),(t,False)],sigma[s0,t]) for s0,t in product(range(2),repeat=2) if sigma[s0,t]!=0])
                        return sparse.kron(sparse.eye(data['sectors'][l['root']]['dimension']),a,format='csr')
                    if NROOT[l['root']]!=NROOT[r['root']]-1 or l['n']!=r['n']+1:return None
                    result=None
                    for outspin,inspin in product(range(2),repeat=2):
                        if sigma[outspin,inspin]==0:continue
                        # d† c = (-1)^(Nroot-1) c tensor d†, root modes first.
                        c=creation[l['root'],r['root'],1,inspin].conj().T
                        # undo root spin basis; l.U/r.U already include it.
                        c=roots[l['root']]['U']@c@roots[r['root']]['U'].conj().T
                        d=extra_operator(masks[l['n']],masks[r['n']],[([(outspin,True)],1.)])
                        item=(-1)**(NROOT[r['root']]-1)*sigma[outspin,inspin]*sparse.kron(c,d,format='csr')
                        result=item if result is None else result+item
                    return result
                matrices.append(assemble(name,primitive))
            op['hops'].append([(-1.,matrices)])
        s['op']=op
    geometry=graph(p,rotor);vertices=np.array([[-.5,0,0],[.5,0,0],[1.5,0,0],[2.5,0,0.]])
    geometry['vertices']=vertices;geometry['paths']+= [vertices[[1,2]],vertices[[2,3]]]
    geometry['weights']=np.concatenate([geometry['weights'],[1.,1.]])
    geometry['tangents']=np.concatenate([geometry['tangents'],np.einsum('qia,ea->qei',rotor['R'],np.array([[1.,0,0],[1.,0,0]]))],axis=1)
    nodes,weights=np.polynomial.legendre.leggauss(p['path_quadrature'])
    geometry['quadrature'] += [dict(points=(a+b)/2+nodes[:,None]*(b-a)/2,dl=weights[:,None]*(b-a)/2) for a,b in [(vertices[1],vertices[2]),(vertices[2],vertices[3])]]
    coeff=[]
    for route in geometry['quadrature']:
        body=np.einsum('nmi,ni->m',profiles(route['points'],p['em_width'],p['epsilon'],p['charge_sigma']),route['dl'])
        coeff.append(np.concatenate([np.einsum('qia,a->qi',rotor['R'],body[k:k+3]) for k in [0,3]],axis=1))
    points=np.einsum('qia,sa->qsi',rotor['R'],vertices)
    conn=np.cross(points[:,:,None,:],profiles(points,p['em_width'],p['epsilon'],p['charge_sigma'])).transpose(1,0,3,2)
    model=dict(parameters=p,sectors=sectors,radial=radial,physical_rotor=rotor,field=field,rqspin=rqspin,rqU=ru,rq=rq,rotor=dict(basis=np.zeros((1,10*nf))),geometry=geometry,coefficients=coeff,connections=conn,sparse=factory,roots=roots,ext=ext,masks=masks,data=data,creation=creation,rawmetrics=rawmetrics,cache={})
    return model


def law(model,name,bridge=True,external=None,derivative=None):
    p=model['parameters'];s=model['sectors'][name];op=s['op'];rot=model['physical_rotor'];field=model['field'];rq=model['rq'];nf=field['dimension'];eye=np.eye(10*nf);external=np.zeros(6) if external is None else np.asarray(external);terms=[]
    def add(a,b,power=0):terms.append((a,b,power))
    if derivative is None:
        matter=op['electric']-p['plaquette']*op['B']
        for a,b in product(range(4),repeat=2):
            v=float(gaussian_green(np.linalg.norm(model['geometry']['vertices'][a]-model['geometry']['vertices'][b]),p['charge_sigma'],p['epsilon']))
            matter=matter+op['rho'][a]@op['rho'][b]*v/2
        add(matter,eye)
        add(sparse.eye(s['dimension']),rq(sparse.kron(rot['L2']/(2*p['orientation_inertia']),sparse.eye(nf))+sparse.kron(sparse.eye(10),field['H_E'])))
    key=('mechanical',tuple(external),derivative)
    if key not in model['cache']:
        linear=[];quadratic=[]
        for site,c in enumerate(model['connections']):
            matrix=np.zeros_like(eye,complex)
            for axis,mu in product(range(3),range(6)):
                r=rot['integrate'](c[:,axis,mu]);a=rot['L'][axis]@r+r@rot['L'][axis]
                q=field['Q'][mu]+external[mu]*np.eye(nf) if derivative is None else (np.eye(nf) if derivative==mu else np.zeros((nf,nf)))
                matrix-=sparse.kron(a,q).toarray()/(2*p['orientation_inertia'])
            linear.append(rq(matrix))
        for left in range(4):
            for right in range(left,4):
                matrix=np.zeros_like(eye,complex)
                for mu,nu in product(range(6),repeat=2):
                    c=model['connections'];r=rot['integrate'](np.sum(c[left][:,:,mu]*c[right][:,:,nu],axis=1))/(2*p['orientation_inertia'])
                    if not r.nnz:continue
                    q=(field['QQ'][mu][nu]+external[mu]*field['Q'][nu]+external[nu]*field['Q'][mu]+external[mu]*external[nu]*np.eye(nf)) if derivative is None else ((field['Q'][nu]+external[nu]*np.eye(nf))*(derivative==mu)+(field['Q'][mu]+external[mu]*np.eye(nf))*(derivative==nu))
                    if np.any(q):matrix+=sparse.kron(r,q).toarray()*(1 if left==right else 2)
                quadratic.append((left,right,rq(matrix)))
        model['cache'][key]=(linear,quadratic)
    linear,quadratic=model['cache'][key]
    for site,matrix in enumerate(linear):add(op['rho'][site],matrix)
    for left,right,matrix in quadratic:add(op['rho'][left]@op['rho'][right],matrix)
    for edge,channels in enumerate(op['hops']):
        if edge==2 and not bridge:continue
        c=model['coefficients'][edge]
        for charge,matrices in channels:
            for axis in [None,0,1,2]:
                key=('phase',edge,charge,tuple(external),derivative,axis)
                if key not in model['cache']:
                    values=phase_values(c,charge,field)*np.exp(1j*charge*(c@external))[:,None,None]
                    if derivative is not None:values*= (1j*charge*c[:,derivative])[:,None,None]
                    factor=1 if axis is None else model['geometry']['tangents'][:,edge,axis]
                    b=np.einsum('qr,q,qef,qs->resf',rot['basis'].conj(),rot['weights'],values*(factor if np.isscalar(factor) else factor[:,None,None]),rot['basis'],optimize=True).reshape(10*nf,10*nf)
                    model['cache'][key]=rq(b)
                b=model['cache'][key]
                a=(p['scalar_coupling']*matrices[0] if axis is None else 1j*p['vector_velocity']*matrices[axis+1])*model['geometry']['weights'][edge]
                add(a,b,1);add(a.conj().T,b.conj().T,1)
    if derivative is not None:terms=[(-a,b,n) for a,b,n in terms]
    radial=None if derivative is not None else {float(j):model['radial'].operators(float(j))['H'] for j in set(s['spin']['j'])}
    return FormOperator(s,s,model['rotor'],model['radial'],terms,radial)


def fibre(model,name,J,M):
    return sparse.kron(coupled_isometry(model['sectors'][name]['spin'],model['rqspin'],J,M),sparse.eye(model['radial'].count),format='csr')


def source_matrix(model,target,site,m):
    """Literal c† or spinor epsilon.c on the SAME CAR36 Fock space."""
    source=model['sectors']['neutral'];dest=model['sectors'][target];removal=target=='antilepton';rows=[]
    spin=0 if m==.5 else 1
    if removal:spin=1-spin
    coefficient=-1. if removal and m==-.5 else 1.
    for l in dest['blocks']:
        row=[]
        for r in source['blocks']:
            a=None
            if site<2 and l['n']==r['n']:
                if not removal and NROOT[l['root']]==NROOT[r['root']]+1:c=model['creation'][r['root'],l['root'],site,spin]
                elif removal and NROOT[l['root']]==NROOT[r['root']]-1:c=model['creation'][l['root'],r['root'],site,spin].conj().T
                else:c=None
                if c is not None:
                    c=model['roots'][l['root']]['U']@c@model['roots'][r['root']]['U'].conj().T
                    a=sparse.kron(c,sparse.eye(len(model['masks'][l['n']])),format='csr')
            elif site>=2 and l['root']==r['root'] and l['n']==r['n']+(-1 if removal else 1):
                d=extra_operator(model['masks'][l['n']],model['masks'][r['n']],[([(2*(site-2)+spin,not removal)],(-1)**NROOT[r['root']])])
                a=sparse.kron(sparse.eye(model['data']['sectors'][r['root']]['dimension']),d,format='csr')
            row.append(sparse.csr_matrix((l['stop']-l['start'],r['stop']-r['start'])) if a is None else coefficient*l['U'].conj().T@a@r['U'])
        rows.append(row)
    return model['sparse'].csr(sparse.bmat(rows,format='csr'))


def source(model,target,site,m):
    # Wigner multiplier/Gauss sewing: natural intertwiner. The reduced tensor
    # is c_M; its covariance and LEFT coordinate differentiation are measured.
    a=source_matrix(model,target,site,m);zero=model['sectors']['neutral'];one=model['sectors'][target]
    return FormOperator(zero,one,model['rotor'],model['radial'],[(a,np.eye(10*model['field']['dimension']),0)])


def observable(model,name,matter=None,rq=None,power=0):
    s=model['sectors'][name]
    return FormOperator(s,s,model['rotor'],model['radial'],[(sparse.eye(s['dimension']) if matter is None else matter,np.eye(10*model['field']['dimension']) if rq is None else rq,power)])


def prepared_reference(model):
    s=model['sectors']['neutral'];b=next(b for b in s['blocks'] if b['root']=='neutral');basis=model['data']['native']['neutral']
    sea=sum(1<<mode(f,1,spin,c) for f,spin,c in product(range(2),range(2),range(4)))
    vacuum=np.zeros(19,complex);vacuum[0]=1;grams=charge_grams(basis.seeds,[{sea:vacuum}]);root=np.zeros(basis.dimension,complex)
    for h,w in basis.blocks.items():
        if h in grams:root[basis.slices[h]]=(w.conj().T@grams[h].toarray())[:,0]
    # Both spins occupy the negative-energy dimer orbital (d2-d3)/sqrt2.
    ext=np.zeros(6,complex);lookup={m:i for i,m in enumerate(model['masks'][2])}
    for i,j in product(range(2),repeat=2):
        result=string(0,[(2*j+1,True),(2*i,True)])
        ext[lookup[result[0]]]+=(-1)**(i+j)*result[1]/2
    matter=np.zeros(s['dimension'],complex);matter[b['start']:b['stop']]=b['U'].conj().T@np.kron(root,ext)
    if np.linalg.norm(matter[s['spin']['j']!=0])>1e-8:raise ValueError('Prepared sea is not a scalar')
    angle=np.zeros(10*model['field']['dimension'],complex);angle[0]=1;angle=model['rqU'].conj().T@angle
    _,rv=np.linalg.eigh(model['radial'].operators(0.)['H']);rad=rv[:,0]
    value=np.einsum('i,a,n->ian',matter,angle,rad).ravel();return value/np.linalg.norm(value)
