"""Charged breathing-shell frame bundle, moving geometry and complete l=1 EM.

Only raw scalar-colour Gauss/CAR frame helpers are inherited. No producer loads.
The Z2 charged bundle and mechanical shell law are explicit new physical inputs.
"""
from __future__ import annotations
from itertools import product
from math import factorial
import numpy as np
from scipy.linalg import expm
from scipy.special import eval_hermite, roots_genlaguerre
from scipy.sparse import csr_matrix, eye, kron
from rich_frame_holonomy_20261014_model import raw_frame


def oscillator(cut):
    basis = [n for n in product(range(cut+1), repeat=3) if sum(n) <= cut]
    index = {n:i for i,n in enumerate(basis)}
    a = []
    for axis in range(3):
        op = np.zeros((len(basis),len(basis)),complex)
        for col,n in enumerate(basis):
            if n[axis]:
                target=list(n); target[axis]-=1
                op[index[tuple(target)],col]=np.sqrt(n[axis])
        a.append(op)
    q=[(v+v.conj().T)/np.sqrt(2) for v in a]
    mom=[-1j*(v-v.conj().T)/np.sqrt(2) for v in a]
    angular=[-1j*(a[(i+1)%3].conj().T@a[(i+2)%3]-a[(i+2)%3].conj().T@a[(i+1)%3]) for i in range(3)]
    return dict(basis=basis,a=a,Q=q,P=mom,L=angular,number=np.diag([sum(n) for n in basis]))


def sphere(nz=8,nphi=16):
    z,w=np.polynomial.legendre.leggauss(nz)
    phi=2*np.pi*np.arange(nphi)/nphi
    directions=np.array([[np.sqrt(1-v*v)*np.cos(t),np.sqrt(1-v*v)*np.sin(t),v] for v in z for t in phi])
    weights=np.repeat(w/2,nphi)/nphi
    return directions,weights


def geometry_quadrature(osc,p):
    radial,wr=roots_genlaguerre(p['radial_quadrature'],.5)
    directions,wa=sphere()
    points=(p['length']*np.sqrt(radial)[:,None,None]*directions[None,:,:]).reshape(-1,3)
    weights=(2/np.sqrt(np.pi)*wr[:,None]*wa[None,:]).reshape(-1)
    polynomials=np.ones((len(points),len(osc['basis'])))
    for col,ns in enumerate(osc['basis']):
        for axis,n in enumerate(ns):
            polynomials[:,col]*=eval_hermite(n,points[:,axis]/p['length'])/np.sqrt(2**n*factorial(n))
    def integrate(values):
        return np.einsum('xa,x,xb,x...->ab...',polynomials,weights,polynomials,values,optimize=True)
    return points,integrate


def shell_connections(points,b,width):
    """Average actual A(X+b n) and b n cross A over the material shell.

    A(x;Q)=exp(-x^2/2w^2) (Q cross x)/2.  a_X=alpha Q cross X/2;
    a_Omega=beta Q. No spin magnetic moment is independently supplied.
    """
    radius=np.linalg.norm(points,axis=1)
    z,w=np.polynomial.legendre.leggauss(40)
    values=np.exp(-(radius[:,None]**2+b*b+2*b*radius[:,None]*z)/(2*width**2))
    f0=values@(w/2); nz=values@(w*z/2); nzz=values@(w*z*z/2)
    transverse=(f0-nzz)/2
    alpha=f0+b*nz/radius
    scalar=b/2*(radius*nz+b*(f0-transverse))
    tensor=b/2*(-nz/radius-b*(nzz-transverse)/radius**2)
    beta=scalar[:,None,None]*np.eye(3)+tensor[:,None,None]*points[:,:,None]*points[:,None,:]
    return alpha,beta


def build(p):
    frame=raw_frame(); g=oscillator(p['geometry_cut']); fld=oscillator(p['field_cut'])
    extended=oscillator(p['field_cut']+2); selected=[extended['basis'].index(n) for n in fld['basis']]
    points,integrate=geometry_quadrature(g,p)
    ng=len(g['basis']); nf=len(fld['basis']); shapes=p['radii']; nb=len(shapes)
    im=np.eye(16); ig=np.eye(ng); iff=np.eye(nf)
    nt=[sum(frame['source'][2*t+a]@frame['source'][2*t+a].conj().T for a in range(2)) for t in range(2)]
    charges=[a-im for a in nt]
    mass=p['mass']*sum(frame['source'][a]@frame['source'][a+2].conj().T+frame['source'][a+2]@frame['source'][a].conj().T for a in range(2))
    s=frame['S']; s2=sum(a@a for a in s)
    metric=p['epsilon']*np.pi**1.5*p['field_width']**5/4
    stiffness=5*np.pi**1.5*p['field_width']**3/(8*p['mu'])
    omega=np.sqrt(stiffness/metric)
    fq=[a/np.sqrt(metric*omega) for a in fld['Q']]
    fp=[a*np.sqrt(metric*omega) for a in fld['P']]
    fullq=[a/np.sqrt(metric*omega) for a in extended['Q']]
    qq=[[((fullq[a]@fullq[b]+fullq[b]@fullq[a])/2)[np.ix_(selected,selected)] for b in range(3)] for a in range(3)]
    field_h=omega*(fld['number']+1.5*iff)
    linear=[csr_matrix((16*nb*ng,16*nb*ng),dtype=complex) for _ in range(3)]
    quadratic=[[csr_matrix((16*nb*ng,16*nb*ng),dtype=complex) for _ in range(3)] for _ in range(3)]
    h0=csr_matrix((16*nb*ng,16*nb*ng),dtype=complex)
    translation_linear=[a.copy() for a in linear]; rotation_linear=[a.copy() for a in linear]
    def mg(m,b,op):
        projector=np.zeros((nb,nb)); projector[b,b]=1
        return kron(csr_matrix(m),kron(csr_matrix(projector),csr_matrix(op)),format='csr')
    kinetic=np.diag([(sum(n)+1.5)/(2*p['reduced_mass']*p['length']**2) for n in g['basis']])
    radius2=integrate(np.sum(points*points,axis=1))
    base_geometry=kinetic+p['binding']*radius2/2
    shell_records=[]
    for bindex,radius in enumerate(shapes):
        inertia=2*p['rotating_mass']*radius**2/3
        h0+=mg(mass+s2/(2*inertia),bindex,ig)+mg(im,bindex,base_geometry)
        connections=[shell_connections(points,radius*ratio,p['field_width']) for ratio in p['shell_ratios']]
        shell_records.append(connections)
        # Smooth spherical-shell longitudinal self/cross Coulomb energy.
        for t in range(2):
            for u in range(2):
                coefficient=p['charge']**2/(8*np.pi*p['epsilon']*radius*max(p['shell_ratios'][t],p['shell_ratios'][u]))
                h0+=mg(coefficient*(charges[t]@charges[u]),bindex,ig)
        for t,(alpha,beta) in enumerate(connections):
            alphamat=integrate(alpha)
            betamat=integrate(beta)
            for axis in range(3):
                trans=mg(-p['charge']*charges[t]/(2*p['reduced_mass']),bindex,alphamat@g['L'][axis])
                rot=sum(mg(-p['charge']*charges[t]@s[i]/inertia,bindex,betamat[:,:,i,axis]) for i in range(3))
                linear[axis]+=trans+rot; translation_linear[axis]+=trans; rotation_linear[axis]+=rot
        for t,(at,bt) in enumerate(connections):
            for u,(au,bu) in enumerate(connections):
                transl=at[:,None,None]*au[:,None,None]*(np.sum(points*points,axis=1)[:,None,None]*np.eye(3)-points[:,:,None]*points[:,None,:])/4
                rotating=np.einsum('xij,xik->xjk',bt,bu)
                tensor=integrate(transl/(2*p['reduced_mass'])+rotating/(2*inertia))
                for a in range(3):
                    for b in range(3):
                        quadratic[a][b]+=mg(p['charge']**2*charges[t]@charges[u],bindex,tensor[:,:,a,b])
    shape=np.diag([p['shape_binding']*(r-p['preferred_radius'])**2 for r in shapes]).astype(complex)
    step=shapes[1]-shapes[0]
    shape+=np.diag(np.full(nb,2/(2*p['shape_mass']*step**2)))
    shape+=np.diag(np.full(nb-1,-1/(2*p['shape_mass']*step**2)),1)+np.diag(np.full(nb-1,-1/(2*p['shape_mass']*step**2)),-1)
    h0+=kron(csr_matrix(im),kron(csr_matrix(shape),csr_matrix(ig)),format='csr')
    h=kron(h0,csr_matrix(iff))+kron(eye(16*nb*ng),csr_matrix(field_h))
    h+=sum(kron(linear[a],csr_matrix(fq[a])) for a in range(3))
    h+=sum(kron(quadratic[a][b],csr_matrix(qq[a][b])) for a in range(3) for b in range(3))
    currents=[-kron(linear[a],csr_matrix(iff))-sum(kron(quadratic[a][b]+quadratic[b][a],csr_matrix(fq[b])) for b in range(3)) for a in range(3)]
    contacts=[[-kron(quadratic[a][b]+quadratic[b][a],csr_matrix(iff)) for b in range(3)] for a in range(3)]
    embed_m=lambda a:kron(csr_matrix(a),eye(nb*ng*nf),format='csr')
    embed_g=lambda a:kron(eye(16*nb),kron(csr_matrix(a),eye(nf)),format='csr')
    embed_f=lambda a:kron(eye(16*nb*ng),csr_matrix(a),format='csr')
    spin=[embed_m(a) for a in s]; orbital=[embed_g(a) for a in g['L']]; field_l=[embed_f(a) for a in fld['L']]
    j=[spin[a]+orbital[a]+field_l[a] for a in range(3)]
    # Canonical charge conjugation is constructed on the same complete CAR16.
    eps=np.array([[0,1],[-1,0]],complex); cu=np.kron(np.diag([1,-1]),eps)
    full=np.eye(16,dtype=complex)[:,-1]; cf=np.zeros((16,16),complex)
    for state in range(16):
        v=full.copy()
        for a in reversed(range(4)):
            if state&(1<<a):v=sum(cu[a,b]*frame['source'][b].conj().T for b in range(4))@v
        cf[:,state]=v
    c=kron(csr_matrix(cf),kron(eye(nb*ng),csr_matrix(np.diag([(-1)**sum(n) for n in fld['basis']]))),format='csr')
    radial_forward=p['mass']*sum(frame['source'][a]@frame['source'][a+2].conj().T for a in range(2))
    return dict(H=h.tocsr(),H0=h0,linear=linear,quadratic=quadratic,current=currents,contact=contacts,
                J=j,S=spin,L=orbital,F=field_l,X=[embed_g(p['length']*a) for a in g['Q']],PX=[embed_g(a/p['length']) for a in g['P']],Q=[embed_f(a) for a in fq],P=[embed_f(a) for a in fp],
                source=[embed_m(a) for a in frame['source']],N=embed_m(frame['N']),Nt=[embed_m(a) for a in nt],radial_current=embed_m(1j*(radial_forward-radial_forward.conj().T)),C=c,Csource=cu,frame=frame,
                geometry=g,field=fld,ng=ng,nf=nf,nb=nb,metric=metric,stiffness=stiffness,omega=omega,
                field_energy=embed_f(field_h),radius=kron(eye(16),kron(csr_matrix(np.diag(shapes)),eye(ng*nf))),
                translation_current=[-kron(a,csr_matrix(iff)) for a in translation_linear],rotation_current=[-kron(a,csr_matrix(iff)) for a in rotation_linear],parameters=p)


def shifted_h(model,external):
    """External physical A amplitude; exact same-H mechanical source derivative."""
    p=model['parameters']; nf=model['nf']; iff=eye(nf)
    result=model['H'].copy()
    for a in range(3):
        result+=external[a]*kron(model['linear'][a],iff)
        for b in range(3):
            qa=model['field']['Q'][a]/np.sqrt(model['metric']*model['omega'])
            qb=model['field']['Q'][b]/np.sqrt(model['metric']*model['omega'])
            delta=external[a]*qb+external[b]*qa+external[a]*external[b]*np.eye(nf)
            result+=kron(model['quadratic'][a][b],csr_matrix(delta))
    return result.tocsr()


def rotation_geometry(points,material_directions,quaternion,radius,axis,angle):
    """Actual configuration action: positions/routes, frame matrix and EM vectors."""
    axis=np.asarray(axis,float); axis/=np.linalg.norm(axis)
    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    rotation=expm(angle*cross)
    pauli=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    lift=expm(-.5j*angle*sum(axis[a]*pauli[a] for a in range(3)))
    matrix=lift@quaternion
    body_rotation=np.array([[np.trace(pauli[a]@quaternion@pauli[b]@quaternion.conj().T).real/2 for b in range(3)] for a in range(3)])
    routes=points[:,None,:]+radius*(material_directions@body_rotation.T)[None,:,:]
    return dict(rotation=rotation,lift=lift,frame=matrix,positions=points@rotation.T,
                routes=routes@rotation.T,matrixfield=matrix@pauli[2]@matrix.conj().T)


def electromagnetic_profile(points,amplitude,width):
    """Actual transverse mode fields, not a label on a spin operator."""
    radius2=np.sum(points*points,axis=-1)
    profile=np.exp(-radius2/(2*width**2))
    potential=profile[...,None]*np.cross(amplitude,points)/2
    magnetic=profile[...,None]*((1-radius2/(2*width**2))[...,None]*amplitude+points*np.sum(points*amplitude,axis=-1)[...,None]/(2*width**2))
    return potential,magnetic
