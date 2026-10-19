"""One new moving-shell / correlated radial supplier / primitive-electric law.

Both full j=1 TE and TM irreps are actual transverse spatial profiles.  All
products are Galerkin projections of continuum products, not products of cut
coordinate matrices.  Primitive bindings and charged bundle are inputs.
"""
from itertools import product
from math import factorial
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, eye, kron
from scipy.spatial.transform import Rotation
from scipy.special import eval_hermite, roots_genlaguerre
from rich_physical_noether_20261014_model import oscillator, sphere
from rich_frame_holonomy_20261014_model import raw_frame
from rich_rotcov_induced_20261015_connections import shell_connections


def comm(a,b):return a@b-b@a

def symbasis(osc):
    unused,u=eigh(osc['L'][2]);return u


def quadrature(osc,length,nr):
    r,wr=roots_genlaguerre(nr,.5);directions,wa=sphere(6,12)
    points=(length*np.sqrt(r)[:,None,None]*directions[None,:,:]).reshape(-1,3)
    weights=(2/np.sqrt(np.pi)*wr[:,None]*wa[None,:]).ravel()
    polys=np.ones((len(points),len(osc['basis'])))
    gradients=np.zeros((len(points),len(osc['basis']),3))
    for c,ns in enumerate(osc['basis']):
        for axis,n in enumerate(ns):polys[:,c]*=eval_hermite(n,points[:,axis]/length)/np.sqrt(2**n*factorial(n))
        for axis in range(3):
            term=np.ones(len(points))
            for j,n in enumerate(ns):
                value=eval_hermite(n,points[:,j]/length)
                if j==axis:value=(2*n*eval_hermite(n-1,points[:,j]/length) if n else np.zeros(len(points)))-points[:,j]/length*value
                term*=value/np.sqrt(2**n*factorial(n))
            gradients[:,c,axis]=term/length
    u=symbasis(osc);polys=polys@u;gradients=np.einsum('xci,cb->xbi',gradients,u)
    def integrate(value):return np.einsum('xa,x,xb,x...->ab...',polys.conj(),weights,polys,value,optimize=True)
    def kinetic(mass):return np.einsum('xai,x,xbi->ab',gradients.conj(),weights,gradients,optimize=True)/(2*mass)
    def linear(connection,mass,charge):
        op=-1j*np.einsum('xa,x,xki,xbi->kab',polys.conj(),weights,connection,gradients,optimize=True)
        return -charge*(op+op.conj().transpose(0,2,1))/(2*mass)
    return dict(points=points,weights=weights,polys=polys,u=u,integrate=integrate,kinetic=kinetic,linear=linear)


def profiles(points,width,epsilon):
    """A_TE,a=f(e_a cross x)/2; A_TM,a=curl A_TE,a. Unit electric metric."""
    r2=np.sum(points*points,axis=1);f=np.exp(-r2/(2*width**2))
    te=np.array([f[:,None]*np.cross(np.eye(3)[a],points)/2 for a in range(3)]).transpose(1,0,2)
    tm=f[:,None,None]*((1-r2/(2*width**2))[:,None,None]*np.eye(3)[None,:,:]+points[:,None,:]*points[:,:,None]/(2*width**2))
    metrics=np.array([epsilon*np.pi**1.5*width**5/4]*3+[epsilon*5*np.pi**1.5*width**3/8]*3)
    return np.concatenate([te,tm],axis=1)/np.sqrt(metrics)[None,:,None],metrics


def field_space(cut,scale):
    basis=[n for n in product(range(cut+1),repeat=6) if sum(n)<=cut]
    extended=[n for n in product(range(cut+3),repeat=6) if sum(n)<=cut+2]
    index={n:i for i,n in enumerate(extended)};selected=[index[n] for n in basis];a=[]
    for axis in range(6):
        op=np.zeros((len(extended),len(extended)),complex)
        for c,n in enumerate(extended):
            if n[axis]:
                target=list(n);target[axis]-=1;op[index[tuple(target)],c]=np.sqrt(n[axis])
        a.append(op)
    q=[scale*(v+v.conj().T)/np.sqrt(2) for v in a];p=[-1j*(v-v.conj().T)/(scale*np.sqrt(2)) for v in a]
    project=lambda op:op[np.ix_(selected,selected)]
    l=[]
    for family in range(2):l.append([project(-1j*(a[3*family+(i+1)%3].conj().T@a[3*family+(i+2)%3]-a[3*family+(i+2)%3].conj().T@a[3*family+(i+1)%3])) for i in range(3)])
    unused,u=eigh(l[0][2]+l[1][2]);convert=lambda op:u.conj().T@project(op)@u
    return dict(basis=basis,u=u,Q=[convert(v) for v in q],P=[convert(v) for v in p],QQ=[[convert((v@w+w@v)/2) for w in q] for v in q],E=convert(sum(v@v/2 for v in p)),F=[[u.conj().T@v@u for v in family] for family in l])


def supplier(p):
    osc=oscillator(p['supplier_cut']);n=len(osc['basis']);ident=np.eye(n)
    qs=[quadrature(osc,length,p['radial_quadrature']) for length in p['supplier_lengths']]
    one=[];lin=[];quad=[];angular=[]
    for f,q in enumerate(qs):
        x=q['points'];r2=np.sum(x*x,axis=1);avec,metrics=profiles(x,p['field_width'],p['epsilon'])
        if p.get('supplier_path')=='tangent':
            unit=x/np.sqrt(r2)[:,None]
            avec-=np.einsum('xai,xi->xa',avec,unit)[:,:,None]*unit[:,None,:]
        if p.get('supplier_path')=='omit_TE':avec[:,:3,:]=0
        if p.get('supplier_path')=='constant_gauge':
            avec[:,:,:]=0;avec[:,:3,:]=np.eye(3)[None,:,:]
        potential=p['supplier_binding'][f]*r2/2-p['charge']**2*p['ion_charge']/(4*np.pi*p['epsilon']*np.sqrt(r2+p['coulomb_softening']**2))
        one.append(q['kinetic'](p['supplier_masses'][f])+q['integrate'](potential))
        lin.append(q['linear'](avec,p['supplier_masses'][f],-p['charge']))
        quad.append(p['charge']**2*q['integrate'](np.einsum('xai,xbi->xab',avec,avec))/(2*p['supplier_masses'][f]))
        angular.append([q['u'].conj().T@v@q['u'] for v in osc['L']])
    x,y=[q['points'] for q in qs]
    distance=np.sqrt(np.sum((x[:,None,:]-y[None,:,:])**2,axis=2)+p['coulomb_softening']**2)
    density=[np.einsum('xa,xb,x->xab',q['polys'].conj(),q['polys'],q['weights']) for q in qs]
    coulomb=p['charge']**2/(4*np.pi*p['epsilon'])*np.einsum('xab,xy,ycd->acbd',density[0],1/distance,density[1],optimize=True).reshape(n*n,n*n)
    # Icosahedral angular cubature exactly twirls ranks <= 4 for s+p retained
    # products; this integrates the rotation-invariant Coulomb kernel, not a
    # vector cancellation or alteration of the supplier law.
    if p['supplier_cut']!=1:raise ValueError('Declared exact angular cubature is only implemented for complete s+p supplier shells')
    from scipy.linalg import expm
    scalar=np.zeros_like(coulomb,complex)
    for rot in Rotation.create_group('I').as_rotvec():
        transforms=[expm(-1j*sum(rot[a]*angular[f][a] for a in range(3))) for f in range(2)]
        u=np.kron(*transforms);scalar+=u@coulomb@u.conj().T/60
    h=np.kron(one[0],ident)+np.kron(ident,one[1])+p['correlation']*scalar
    b=[np.kron(lin[0][a],ident)+np.kron(ident,lin[1][a]) for a in range(6)]
    c=[[np.kron(quad[0][:,:,a,b],ident)+np.kron(ident,quad[1][:,:,a,b]) for b in range(6)] for a in range(6)]
    ls=[[np.kron(angular[0][a],ident) for a in range(3)],[np.kron(ident,angular[1][a]) for a in range(3)]]
    radial=[np.kron(qs[0]['integrate'](np.sum(x*x,axis=1)),ident),np.kron(ident,qs[1]['integrate'](np.sum(y*y,axis=1)))]
    return dict(H=h,linear=b,quadratic=c,L=ls,r2=radial,coulomb=scalar,dimension=n*n,primitive_dimension=n)


def build(p):
    frame=raw_frame();unused,uf=eigh(frame['S'][2]+10*frame['N']);transform=lambda op:uf.conj().T@op@uf
    spin=[transform(op) for op in frame['S']];sources=[transform(op) for op in frame['source']]
    number=transform(frame['N']);charge=number-2*np.eye(16)
    mass=p['mass']*sum(sources[a]@sources[a+2].conj().T+sources[a+2]@sources[a].conj().T for a in range(2))
    osc=oscillator(p['geometry_cut']);g=quadrature(osc,p['length'],p['radial_quadrature']);ng=len(osc['basis']);nb=len(p['radii'])
    sup=supplier(p);ns=sup['dimension'];field=field_space(p['field_cut'],p['field_scale']);nf=len(field['basis'])
    dims=[16,nb,ng,ns,nf];ident=[eye(n,format='csr',dtype=complex) for n in dims]
    def embed(ops):
        out=csr_matrix([[1.]])
        for a,n in enumerate(dims):out=kron(out,csr_matrix(ops[a]) if a in ops else ident[a],format='csr')
        return out
    bodydim=16*nb*ng
    def body(m,b,op):
        project=np.zeros((nb,nb));project[b,b]=1
        return kron(csr_matrix(m),kron(csr_matrix(project),csr_matrix(op)),format='csr')
    base=csr_matrix((bodydim,bodydim),dtype=complex);linear=[base.copy() for _ in range(6)];quadratic=[[base.copy() for _ in range(6)] for _ in range(6)]
    translation=[base.copy() for _ in range(6)];rotation=[base.copy() for _ in range(6)]
    x=g['points'];one=np.eye(16);ig=np.eye(ng)
    for bi,radius in enumerate(p['radii']):
        inertia=2*p['rotating_mass']*radius**2/3
        base+=body(mass+sum(v@v for v in spin)/(2*inertia)+p['charge']**2*charge@charge/(8*np.pi*p['epsilon']*radius),bi,ig)
        base+=body(one,bi,g['kinetic'](p['reduced_mass'])+p['binding']*g['integrate'](np.sum(x*x,axis=1))/2)
        alpha,beta,metrics=shell_connections(x,radius,p['field_width'],p['epsilon'])
        tl=g['linear'](alpha,p['reduced_mass'],p['charge'])
        for a in range(6):
            tr=body(charge,bi,tl[a]);ro=sum(body(-p['charge']*charge@spin[i]/inertia,bi,g['integrate'](beta[:,a,i])) for i in range(3))
            linear[a]+=tr+ro;translation[a]+=tr;rotation[a]+=ro
        tensor=p['charge']**2*g['integrate'](np.einsum('xai,xbi->xab',alpha,alpha)/(2*p['reduced_mass'])+np.einsum('xai,xbi->xab',beta,beta)/(2*inertia))
        for a in range(6):
            for b in range(6):quadratic[a][b]+=body(charge@charge,bi,tensor[:,:,a,b])
    radii=np.array(p['radii']);step=radii[1]-radii[0]
    shape=np.diag(p['shape_binding']*(radii-p['preferred_radius'])**2+1/(p['shape_mass']*step**2))
    shape+=np.diag(np.full(nb-1,-1/(2*p['shape_mass']*step**2)),1)+np.diag(np.full(nb-1,-1/(2*p['shape_mass']*step**2)),-1)
    base+=kron(eye(16),kron(csr_matrix(shape),eye(ng)),format='csr')
    # Breathing and internal-flavour transfer are local: both internal charged
    # channels occupy the SAME uniform shell; radial flux of either transverse
    # family averages to zero. There is no inter-radius core hopping to omit.
    hbare=kron(base,eye(ns*nf),format='csr')+embed({3:sup['H']})+embed({4:field['E']})
    h=hbare.copy();current=[];contact=[]
    supplier_current=[]
    for a in range(6):
        h+=kron(linear[a],kron(eye(ns),csr_matrix(field['Q'][a])),format='csr')
        h+=p['supplier_coupling']*embed({3:sup['linear'][a],4:field['Q'][a]})
        for b in range(6):
            h+=kron(quadratic[a][b],kron(eye(ns),csr_matrix(field['QQ'][a][b])),format='csr')
            h+=p['supplier_coupling']**2*embed({3:sup['quadratic'][a][b],4:field['QQ'][a][b]})
    for a in range(6):
        cur=-kron(linear[a],eye(ns*nf),format='csr')-p['supplier_coupling']*embed({3:sup['linear'][a]})
        sc=-p['supplier_coupling']*embed({3:sup['linear'][a]})
        row=[]
        for b in range(6):
            cur-=kron(quadratic[a][b]+quadratic[b][a],kron(eye(ns),csr_matrix(field['Q'][b])),format='csr')
            dia=p['supplier_coupling']**2*embed({3:sup['quadratic'][a][b]+sup['quadratic'][b][a]})
            cur-=p['supplier_coupling']**2*embed({3:sup['quadratic'][a][b]+sup['quadratic'][b][a],4:field['Q'][b]})
            sc-=p['supplier_coupling']**2*embed({3:sup['quadratic'][a][b]+sup['quadratic'][b][a],4:field['Q'][b]})
            row.append(kron(quadratic[a][b]+quadratic[b][a],eye(ns*nf),format='csr')+dia)
        current.append(cur.tocsr());supplier_current.append(sc.tocsr());contact.append(row)
    s=[embed({0:v}) for v in spin];l=[embed({2:g['u'].conj().T@v@g['u']}) for v in osc['L']]
    sl=[[embed({3:v}) for v in family] for family in sup['L']];fl=[[embed({4:v}) for v in family] for family in field['F']]
    j=[s[a]+l[a]+sum(family[a] for family in sl+fl) for a in range(3)]
    return dict(H=h.tocsr(),bare=hbare,J=j,S=s,L=l,SL=sl,F=fl,N=embed({0:number}),sources=[embed({0:v}) for v in sources],current=current,contact=contact,supplier_current=supplier_current,Q=[embed({4:v}) for v in field['Q']],P=[embed({4:v}) for v in field['P']],X=[embed({2:p['length']*g['u'].conj().T@v@g['u']}) for v in osc['Q']],PX=[embed({2:g['u'].conj().T@v@g['u']/p['length']}) for v in osc['P']],radius=embed({1:np.diag(radii)}),supplier_r2=[embed({3:v}) for v in sup['r2']],supplier_energy=embed({3:sup['H']}),field_energy=embed({4:field['E']}),coulomb=embed({3:sup['coulomb']}),dims=dims,field=field,supplier=sup,geometry=osc,frame=frame,metrics=metrics,parameters=p,translation=translation,rotation=rotation)


def shifted_h(model,external):
    """Independent coefficient assembly under Q -> Q + external I."""
    h=model['H'].copy();ns=model['dims'][3];nf=model['dims'][4];p=model['parameters']
    body_identity=eye(np.prod(model['dims'][:3]),format='csr')
    for a in range(6):
        h+=external[a]*kron(model['translation'][a]+model['rotation'][a],eye(ns*nf),format='csr')
        h+=p['supplier_coupling']*external[a]*kron(body_identity,kron(csr_matrix(model['supplier']['linear'][a]),eye(nf)),format='csr')
        for b in range(6):
            # Contact is a field-independent multiplication operator; the
            # derivative term is its symmetric original quadratic coefficient.
            h+=external[a]*(model['contact'][a][b]@model['Q'][b])
            h+=external[a]*external[b]*model['contact'][a][b]/2
    return h
