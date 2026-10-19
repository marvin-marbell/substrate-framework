"""Explicit versioned moving-shell / neutral-supplier / primitive-E constructor.

Only the needed supplier-transverse-OFF construction is ported. The sparse
factory is an ordinary required API argument. Frozen source remains untouched;
all basis, geometry integrals and raw Gauss data are reused as pure helpers.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye
from rich_frame_holonomy_20261014_model import raw_frame
from rich_rotcov_induced_20261015_model import oscillator,quadrature,supplier,field_space
from rich_rotcov_induced_20261015_connections import shell_connections


def build(p,sparse):
    csr=sparse.csr;kron=sparse.kron
    frame=raw_frame();unused,uf=eigh(frame['S'][2]+10*frame['N']);transform=lambda op:uf.conj().T@op@uf
    spin=[transform(op) for op in frame['S']];sources=[transform(op) for op in frame['source']]
    number=transform(frame['N']);charge=number-2*np.eye(16)
    mass=p['mass']*sum(sources[a]@sources[a+2].conj().T+sources[a+2]@sources[a].conj().T for a in range(2))
    osc=oscillator(p['geometry_cut']);g=quadrature(osc,p['length'],p['radial_quadrature']);ng=len(osc['basis']);nb=len(p['radii'])
    sup=supplier(p);ns=sup['dimension'];field=field_space(p['field_cut'],p['field_scale']);nf=len(field['basis'])
    dims=[16,nb,ng,ns,nf];ident=[eye(n,format='csr',dtype=complex) for n in dims]
    def embed(ops):
        out=csr([[1.]])
        for a,n in enumerate(dims):out=kron(out,csr(ops[a]) if a in ops else ident[a])
        return out
    bodydim=16*nb*ng
    def body(m,b,op):
        project=np.zeros((nb,nb));project[b,b]=1
        return kron(csr(m),kron(csr(project),csr(op)))
    base=csr((bodydim,bodydim),dtype=complex);linear=[base.copy() for _ in range(6)];quadratic=[[base.copy() for _ in range(6)] for _ in range(6)]
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
    base+=kron(eye(16),kron(csr(shape),eye(ng)))
    hbare=kron(base,eye(ns*nf))+embed({3:sup['H']})+embed({4:field['E']});h=hbare.copy()
    for a in range(6):
        h+=kron(linear[a],kron(eye(ns),csr(field['Q'][a])))
        for b in range(6):h+=kron(quadratic[a][b],kron(eye(ns),csr(field['QQ'][a][b])))
    current=[];contact=[]
    for a in range(6):
        cur=-kron(linear[a],eye(ns*nf));row=[]
        for b in range(6):
            coefficient=quadratic[a][b]+quadratic[b][a]
            cur-=kron(coefficient,kron(eye(ns),csr(field['Q'][b])))
            row.append(kron(coefficient,eye(ns*nf)))
        current.append(cur.tocsr());contact.append(row)
    s=[embed({0:v}) for v in spin];l=[embed({2:g['u'].conj().T@v@g['u']}) for v in osc['L']]
    sl=[[embed({3:v}) for v in family] for family in sup['L']];fl=[[embed({4:v}) for v in family] for family in field['F']]
    j=[s[a]+l[a]+sum(family[a] for family in sl+fl) for a in range(3)]
    return dict(H=h.tocsr(),bare=hbare,J=j,S=s,L=l,SL=sl,F=fl,N=embed({0:number}),sources=[embed({0:v}) for v in sources],current=current,contact=contact,Q=[embed({4:v}) for v in field['Q']],P=[embed({4:v}) for v in field['P']],X=[embed({2:p['length']*g['u'].conj().T@v@g['u']}) for v in osc['Q']],PX=[embed({2:g['u'].conj().T@v@g['u']/p['length']}) for v in osc['P']],radius=embed({1:np.diag(radii)}),supplier_r2=[embed({3:v}) for v in sup['r2']],supplier_energy=embed({3:sup['H']}),field_energy=embed({4:field['E']}),dims=dims,field=field,supplier=sup,geometry=osc,frame=frame,metrics=metrics,parameters=p,translation=translation,rotation=rotation)
