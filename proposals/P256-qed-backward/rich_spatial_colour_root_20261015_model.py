"""NEW two-vertex spatial CAR32, noncoplanar routes, local colour Gauss and SO3.
The original ordered mirror legs ARE the spatial vertices in this new law.
A band-pair Galerkin cut is explicit and need not preserve local EM Ward identities.
"""
from __future__ import annotations
from itertools import product
from math import comb
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from rich_common_colour_lepton_20261015_model import (
    mode, apply, string, one_body, gauge_sector, gauss_operators, scalar_field, common_root)
from rich_colour_vector_loop_20261015_model import masks_for, frame
from rich_colour_composite_20261014_model import generators, loop_field
from rich_quantum_common_field_20261015_sources import source_terms


def clean(matrix,tol=1e-12):
    out=sparse.csr_matrix(matrix);out.data[abs(out.data)<tol]=0;out.eliminate_zeros();return out


def project(sector,matrix):
    return clean(sector['P'].conj().T@matrix@sector['P'])


def epsilon(a,b,c):
    return 0 if len({a,b,c})<3 else 1 if (a,b,c) in [(0,1,2),(1,2,0),(2,0,1)] else -1


def new_C_mask(mask):
    """Original SITE Z tensor i sigma_y; Hadamard BAND X tensor i sigma_y."""
    target=(1<<32)-1;amplitude=1
    for index in reversed([i for i in range(32) if mask&(1<<i)]):
        f=index//16;m=(index%16)//8;s=(index%8)//4;c=index%4
        target,sign=apply(target,mode(f,1-m,1-s,c),False)
        amplitude*=sign*((-1)**s)
    return target,amplitude


def raw_C(source_masks,target_masks):
    lookup={m:i for i,m in enumerate(target_masks)};rows=[];values=[]
    for mask in source_masks:
        target,value=new_C_mask(mask);rows.append(lookup[target]);values.append(value)
    matter=sparse.csr_matrix((values,(rows,range(len(rows)))),shape=(len(target_masks),len(source_masks)))
    swap=np.zeros((19,19));swap[0,0]=1;swap[1:10,10:]=np.eye(9);swap[10:,1:10]=np.eye(9)
    return sparse.kron(matter,swap,format='csr')


def conjugate_sector(sector,loop):
    masks=sorted(new_C_mask(mask)[0] for mask in sector['masks']);p=raw_C(sector['masks'],masks)@sector['P']
    qs,gs=gauss_operators(masks,loop)
    return dict(masks=masks,P=p,Q=qs,G=gs,dimension=p.shape[1],raw_dimension=p.shape[0],
        gauge_error=max(float(sparse.linalg.norm(g@p)) for g in gs))


def so3(p):
    """Complete j0 plus j1=span{sqrt3 R_ia}, not a pinned axis or S2 rotor."""
    count=p['orientation_azimuth_points'];z,w=np.polynomial.legendre.leggauss(p['orientation_polar_points'])
    rotations=[];weights=[]
    rz=lambda a:np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    for a,g,(x,weight) in product(2*np.pi*np.arange(count)/count,2*np.pi*np.arange(count)/count,zip(z,w)):
        beta=np.arccos(x);ry=np.array([[np.cos(beta),0.,np.sin(beta)],[0.,1.,0.],[-np.sin(beta),0.,np.cos(beta)]])
        rotations.append(rz(a)@ry@rz(g));weights.append(weight/(2*count*count))
    rotations=np.array(rotations);weights=np.array(weights);basis=np.column_stack([np.ones(len(weights)),np.sqrt(3)*rotations.reshape(-1,9)])
    angular=[]
    for axis in range(3):
        vector=np.array([[-1j*epsilon(axis,i,j) for j in range(3)] for i in range(3)])
        matrix=np.zeros((10,10),complex);matrix[1:,1:]=np.kron(vector,np.eye(3));angular.append(clean(matrix))
    def integrate(values):
        return clean(basis.T@((weights*np.asarray(values))[:,None]*basis))
    coordinates=[[integrate(rotations[:,i,a]) for a in range(3)] for i in range(3)]
    return dict(R=rotations,weights=weights,basis=basis,dimension=10,L=angular,L2=sparse.diags([0.]+[2.]*9,format='csr'),
        boundary=sparse.diags([0.]+[1.]*9,format='csr'),coordinate=coordinates,
        gram_error=float(np.linalg.norm(basis.T@(weights[:,None]*basis)-np.eye(10))),integrate=integrate)


def graph(p,rotor):
    d=p['vertex_separation'];h=p['path_height'];r0=np.array([-d/2,0.,0.]);r1=-r0
    paths=[np.array([r0,r1]),np.array([r0,[-d/6,h,0.],[d/6,0.,h],r1])]
    lengths=np.array([sum(np.linalg.norm(b-a) for a,b in zip(path[:-1],path[1:])) for path in paths])
    areas=np.array([sum((np.cross(a,b) for a,b in zip(path[:-1],path[1:])),np.zeros(3))/2 for path in paths])
    tangent=np.array([(path[1]-path[0])/np.linalg.norm(path[1]-path[0]) for path in paths])
    displacement=r1-r0;rd=np.einsum('qia,a->qi',rotor['R'],displacement)
    coefficients=np.stack([np.column_stack([rd,np.einsum('qia,a->qi',rotor['R'],area)]) for area in areas],axis=1)
    tangents=np.einsum('qia,ea->qei',rotor['R'],tangent)
    nodes,quadrature_weights=np.polynomial.legendre.leggauss(p.get('path_quadrature',4))
    quadrature=[]
    for path in paths:
        points=[];deltas=[]
        for a,b in zip(path[:-1],path[1:]):
            points.extend((a+b)/2+nodes[:,None]*(b-a)/2)
            deltas.extend(quadrature_weights[:,None]*(b-a)/2)
        quadrature.append(dict(points=np.array(points),dl=np.array(deltas)))
    return dict(vertices=np.array([r0,r1]),paths=paths,lengths=lengths,areas=areas,tangents=tangents,
        coefficients=coefficients,weights=d/lengths,displacement=displacement,body_tangent=tangent,quadrature=quadrature)


def changed_masks(masks,terms):
    return sorted(set(masks)|{step[0] for mask in masks for a,b,_ in terms
        if (step:=string(mask,[(a,True),(b,False)])) is not None})


def density_terms():
    # rho_site1-rho_site0=-sum q(dplus†dminus+h.c.).
    return [(mode(f,m,s,c),mode(f,1-m,s,c),-([2/3,2/3,2/3,0.] if f==0 else [-1/3,-1/3,-1/3,-1.])[c])
        for f,m,s,c in product(range(2),range(2),range(2),range(4))]


def density_image(sector):
    """Exact raw local-density image beyond the pair cut; explicit integration API."""
    terms=density_terms();targets=changed_masks(sector['masks'],terms);lookup={m:i for i,m in enumerate(targets)}
    inclusion=sparse.csr_matrix((np.ones(len(sector['masks'])),([lookup[m] for m in sector['masks']],range(len(sector['masks'])))),
        shape=(len(targets),len(sector['masks'])))
    embedding=sparse.kron(inclusion,sparse.eye(19),format='csr')@sector['P']
    image=sparse.kron(one_body(sector['masks'],terms,targets),sparse.eye(19),format='csr')@sector['P']
    return dict(target_masks=targets,embedding=embedding,delta_image=image)


def site_colour_terms(t):
    site1=np.array([[.5,-.5],[-.5,.5]])
    return [(mode(f,m,s,a),mode(f,n,s,b),site1[m,n]*t[a,b])
        for f,m,n,s,a,b in product(range(2),range(2),range(2),range(2),range(3),range(3)) if t[a,b]!=0]


def electric_and_density(sector,loop,p,g):
    masks=sector['masks'];n=len(masks);cols=19*n;eye=sparse.eye(19,format='csr')
    terms=[term for t in generators() for term in site_colour_terms(t)]+density_terms()
    targets=changed_masks(masks,terms);lookup={mask:i for i,mask in enumerate(targets)}
    inclusion=sparse.csr_matrix((np.ones(n),([lookup[m] for m in masks],range(n))),shape=(len(targets),n))
    same=sparse.kron(inclusion,eye,format='csr');pp=sector['P']
    tree=sparse.csr_matrix((sector['dimension'],)*2,dtype=complex);q1square=tree.copy();cross=tree.copy()
    for a,t in enumerate(generators()):
        q=sparse.kron(one_body(masks,site_colour_terms(t),targets),eye,format='csr')@pp
        r=sparse.kron(inclusion,loop['R'][a],format='csr')@pp
        full=q+r;tree+=full.conj().T@full;q1square+=q.conj().T@q;cross+=q.conj().T@r+r.conj().T@q
    c=project(sector,sparse.kron(sparse.eye(n),loop['C'],format='csr'))
    delta_raw=sparse.kron(one_body(masks,density_terms(),targets),eye,format='csr')@pp
    delta=clean(pp.conj().T@same.conj().T@delta_raw);delta2=clean(delta_raw.conj().T@delta_raw)
    electric=clean(p['electric']*(g['lengths'][0]*tree+g['lengths'][1]*c)/2)
    return dict(electric=electric,tree_C=clean(tree),Q1_C=clean(q1square),electric_cross=clean(cross),C=c,
        density_delta=delta,density_delta2=delta2,electric_decomposition_error=float(sparse.linalg.norm(tree-q1square-cross-c)),
        intermediate_raw_masks=len(targets),local_G1_identity_error=0.,local_G0_error=sector['gauge_error'],
        local_Gauss='Native U links run1->0, with G0=Q0+L_tree+L_cycle and G1=Q1+R_tree+R_cycle. Tree U=I gives R_tree=-(Q1+R_cycle), L_tree=Q1+R_cycle. Thus G1=0 operator identity, G0=globalG, E_tree(0->1)=Q1+R_cycle. Full Q1^2 and Q1-R cross terms include all intermediate masks, not (P Q1 P)^2.')


def raw_forward(sector,loop,edge,f,colourless,spin_matrix):
    masks=sector['masks'];n=len(masks);result=sparse.csr_matrix((19*n,19*n),dtype=complex)
    bridge=np.array([[.5,.5],[-.5,-.5]]) # c_site1† c_site0 in band coordinates.
    members=[3] if colourless else range(3)
    for a,b in product(members,repeat=2):
        if (edge==0 or colourless) and a!=b:continue
        terms=[(mode(f,m,s,a),mode(f,n,t,b),bridge[m,n]*spin_matrix[s,t])
            for m,n,s,t in product(range(2),repeat=4) if spin_matrix[s,t]!=0]
        body=one_body(masks,terms)
        # Forward physical0->1 uses Udagger because native PW link is1->0.
        field=np.eye(19) if edge==0 or colourless else loop['U'][b,a].conj().T
        result+=sparse.kron(body,field,format='csr')
    return result


def spatial_source_terms(site,kind,spin=.5,removal=False):
    terms=[]
    for operations,amplitude in source_terms(kind,spin):
        sites=[site]*len(operations) if isinstance(site,int) else site
        amplitude*=(-1)**sum(x*((index%16)//8) for x,(index,_) in zip(sites,operations))
        if removal:
            mapped=[]
            for index,_ in operations:
                amplitude*=(-1)**((index%8)//4);mapped.append((index^12,False))
            operations=mapped
        terms.append((operations,amplitude))
    return terms


def site_source(sector,target,site,kind,spin=.5,removal=False):
    terms=spatial_source_terms(site,kind,spin,removal)
    lookup={mask:i for i,mask in enumerate(target['masks'])};rows=[];columns=[];values=[]
    for j,mask in enumerate(sector['masks']):
        for operations,amplitude in terms:
            step=string(mask,operations)
            if step is not None and step[0] in lookup:
                rows.append(lookup[step[0]]);columns.append(j);values.append(amplitude*step[1])
    raw=sparse.csr_matrix((values,(rows,columns)),shape=(len(target['masks']),len(sector['masks'])))
    return clean(target['P'].conj().T@sparse.kron(raw,sparse.eye(19),format='csr')@sector['P'])


def pair_pattern(mask):
    singles=0;doubles=0;choice=0;ranks=[]
    for f,s,c in product(range(2),range(2),range(4)):
        pair=(f*2+s)*4+c;occ=[m for m in range(2) if mask&(1<<mode(f,m,s,c))]
        if len(occ)==2:doubles|=1<<pair
        elif len(occ)==1:
            singles|=1<<pair
            if occ[0]:choice|=1<<pair
    for index in range(32):
        if mask&(1<<index):ranks.append((((index//16)*2+(index%8)//4)*4+index%4)*2+(index%16)//8)
    sign=(-1)**sum(a>b for j,a in enumerate(ranks) for b in ranks[j+1:])
    return singles,doubles,choice,sign


def polynomial(single,distance,double):
    a=np.array([comb(single-distance,k) for k in range(single-distance+1)],float)
    b=np.array([comb(distance,k)*(-1)**(distance-k) for k in range(distance+1)],float)
    return np.pad(np.polynomial.polynomial.polymul(a,b)/2**single,(double,0))


def spatial_povms(sector):
    masks=sector['masks'];patterns=[pair_pattern(m) for m in masks];groups={}
    for j,(single,double,choice,sign) in enumerate(patterns):groups.setdefault((single,double),[]).append(j)
    measures={'total':(1<<16)-1,'u':sum(1<<((0*2+s)*4+c) for s,c in product(range(2),range(3))),
        'd':sum(1<<((1*2+s)*4+c) for s,c in product(range(2),range(3))),
        'e':sum(1<<((1*2+s)*4+3) for s in range(2))}
    effects={};closure={};cache={}
    for label,measure in measures.items():
        capacity=16 if label=='total' else 2 if label=='e' else 6
        rows=[[] for _ in range(capacity+1)];cols=[[] for _ in range(capacity+1)];values=[[] for _ in range(capacity+1)]
        for (single,double),indices in groups.items():
            ns=(single&measure).bit_count();nd=(double&measure).bit_count()
            for j in indices:
                for i in indices:
                    difference=patterns[i][2]^patterns[j][2]
                    if difference&~measure:continue
                    distance=(difference&measure).bit_count();key=(ns,distance,nd)
                    if key not in cache:cache[key]=polynomial(*key)
                    coefficient=cache[key]*patterns[i][3]*patterns[j][3]
                    for number,value in enumerate(coefficient):
                        if value:
                            rows[number].append(i);cols[number].append(j);values[number].append(value)
        matrices=[]
        for rr,cc,vv in zip(rows,cols,values):
            raw=sparse.csr_matrix((vv,(rr,cc)),shape=(len(masks),)*2)
            matrices.append(project(sector,sparse.kron(raw,sparse.eye(19),format='csr')))
        effects[label]=matrices
        closure[label]=float(sparse.linalg.norm(sum(matrices)-sparse.eye(sector['dimension'])))
    return dict(effects=effects,closure=closure,
        meaning='Exact original-site fermion-number PVM compressed into band-pair Galerkin subspace: a physical POVM here, not generally idempotent. Polynomial generating function z^ndouble(1+z)^(nsingle-distance)(z-1)^distance/2^nsingle includes ALL original-site Fock numbers without expanding all band determinants.')


class TensorOperator(LinearOperator):
    """Two matter-orientation body matrices times I/phi, plus optional Hphi."""
    def __init__(self,one,phi,scalar,field_energy=False):
        self.one=clean(one);self.phi=clean(phi);self.scalar=scalar;self.field_energy=field_energy
        self.nb=self.one.shape[0];self.nf=len(scalar['grid'])
        super().__init__(dtype=np.dtype(complex),shape=(self.nb*self.nf,)*2)
    def _matvec(self,value):
        x=np.asarray(value).reshape(self.nb,self.nf)
        y=self.one@x+(self.phi@x)*self.scalar['grid'][None,:]
        if self.field_energy:y+=x@self.scalar['H'].T
        return np.asarray(y).ravel()
    def _rmatvec(self,value):
        x=np.asarray(value).reshape(self.nb,self.nf)
        y=self.one.conj().T@x+(self.phi.conj().T@x)*self.scalar['grid'][None,:]
        if self.field_energy:y+=x@self.scalar['H'].conj()
        return np.asarray(y).ravel()
    def _matmat(self,value):return np.column_stack([self._matvec(value[:,j]) for j in range(value.shape[1])])
    def diagonal(self):
        out=self.one.diagonal()[:,None]+self.phi.diagonal()[:,None]*self.scalar['grid'][None,:]
        if self.field_energy:out=out+np.diag(self.scalar['H'])[None,:]
        return out.ravel()
    def trace(self):return self.diagonal().sum()


def matter_ops(sector,loop,p,g,with_povm=True):
    n=len(sector['masks']);ns=sector['dimension'];ops=electric_and_density(sector,loop,p,g)
    sigma=frame()['sigma'];spins=[]
    for s in sigma:
        terms=[(mode(f,m,a,c),mode(f,m,b,c),s[a,b]/2) for f,m,a,b,c in product(range(2),range(2),range(2),range(2),range(4)) if s[a,b]!=0]
        spins.append(project(sector,sparse.kron(one_body(sector['masks'],terms),sparse.eye(19),format='csr')))
    positive=np.array([sum(bool(mask&(1<<mode(f,0,s,c))) for f,s,c in product(range(2),range(2),range(4))) for mask in sector['masks']],float)
    numbers=np.array([mask.bit_count() for mask in sector['masks']],float)
    qvalues=np.array([sum(([2/3,2/3,2/3,0.] if i//16==0 else [-1/3,-1/3,-1/3,-1.])[i%4] for i in range(32) if mask&(1<<i)) for mask in sector['masks']])
    diagonal=lambda values:project(sector,sparse.kron(sparse.diags(values,dtype=float),sparse.eye(19),format='csr'))
    ops.update(S=spins,S2=clean(sum(s@s for s in spins)),positive=diagonal(positive),N=diagonal(numbers),Q=diagonal(qvalues))
    ops['rho0']=clean((ops['Q']-ops['density_delta'])/2);ops['rho1']=clean((ops['Q']+ops['density_delta'])/2)
    ops['B']=project(sector,sparse.kron(sparse.eye(n),loop['B'],format='csr'))
    ops['PW_boundary']=project(sector,sparse.kron(sparse.eye(n),loop['boundary'],format='csr'))
    ops['matter_octet']=project(sector,sparse.kron(sum(q@q for q in sector['Q'])/3,sparse.eye(19),format='csr'))
    hops=[];errors=[]
    for edge in range(2):
        edge_hops=[];raw_combined=sparse.csr_matrix((19*n,)*2,dtype=complex)
        for f in range(2):
            for colourless in [False,True]:
                q=([0.,-1.] if colourless else [2/3,-1/3])[f]
                matrices=[]
                for index,s in enumerate([np.eye(2)]+sigma):
                    raw=raw_forward(sector,loop,edge,f,colourless,s)
                    matrices.append(project(sector,raw));raw_combined+=(index+1)*raw
                edge_hops.append((q,matrices))
        image=raw_combined@sector['P'];errors.append(max(float(sparse.linalg.norm(gg@image)) for gg in sector['G']))
        hops.append(edge_hops)
    ops['hops']=hops;ops['raw_hop_Gauss_image_errors']=errors
    if with_povm:ops['povm']=spatial_povms(sector)
    return ops


def connection_data(op,rotor,p):
    r=rotor['R'][:,:,0];nq=len(r);d=p['vertex_separation'];values=np.zeros((6,3,nq))
    for i,a in product(range(3),repeat=2):
        values[i,a]=d/2*sum(epsilon(a,b,i)*r[:,b] for b in range(3))
        values[3+i,a]=d*d/8*((1. if a==i else 0.)-r[:,a]*r[:,i])
    charge=float(op['Q'].diagonal().real.mean());identity=sparse.eye(op['Q'].shape[0],format='csr')
    matter=[op['density_delta']]*3+[op['Q']]*3
    linear=[]
    for k in range(6):
        matrix=sparse.csr_matrix((op['Q'].shape[0]*10,)*2,dtype=complex)
        for a in range(3):
            alpha=rotor['integrate'](values[k,a]);angular=rotor['L'][a]@alpha+alpha@rotor['L'][a]
            matrix-=sparse.kron(matter[k],angular,format='csr')/(2*p['orientation_inertia'])
        linear.append(clean(matrix))
    products={0:op['density_delta2'],1:charge*op['density_delta'],2:charge*charge*identity}
    return dict(values=values,linear=linear,products=products)


def rotor_quadratic(op,rotor,p,k,l):
    conn=op['connection'];moment=rotor['integrate'](np.sum(conn['values'][k]*conn['values'][l],axis=0))
    product_type=int(k>=3)+int(l>=3)
    return clean(sparse.kron(conn['products'][product_type],moment,format='csr')/(2*p['orientation_inertia']))


def phase_matrix(rotor,g,edge,q,fields,derivative=(),axis=None):
    coefficient=g['coefficients'][:,edge];values=np.exp(1j*q*(coefficient@fields))
    for k in derivative:values*=1j*q*coefficient[:,k]
    if axis is not None:values*=g['tangents'][:,edge,axis]
    return rotor['integrate'](values)


def law(op,rotor,scalar,p,g,fields=None,derivative=(),path_scale=(1.,1.)):
    fields=np.zeros(6) if fields is None else np.asarray(fields,float);nb=op['Q'].shape[0]*10
    one=sparse.csr_matrix((nb,nb),dtype=complex);phi=one.copy();order=len(derivative)
    if order==0:
        one+=sparse.kron(op['electric']-p['plaquette']*op['B'],sparse.eye(10),format='csr')
        one+=sparse.kron(sparse.eye(op['Q'].shape[0]),rotor['L2'],format='csr')/(2*p['orientation_inertia'])
        for k in range(6):
            if fields[k]:one+=fields[k]*op['connection']['linear'][k]
        for k in range(6):
            if fields[k]:
                for l in range(6):
                    if fields[l]:one+=fields[k]*fields[l]*rotor_quadratic(op,rotor,p,k,l)
    elif order==1:
        k=derivative[0];one+=op['connection']['linear'][k]
        for l in range(6):
            if fields[l]:one+=2*fields[l]*rotor_quadratic(op,rotor,p,k,l)
    else:one+=2*rotor_quadratic(op,rotor,p,*derivative)
    for edge in range(2):
        if not path_scale[edge]:continue
        scale=g['weights'][edge]*path_scale[edge]
        for q,matrices in op['hops'][edge]:
            angle=phase_matrix(rotor,g,edge,q,fields,derivative)
            forward=sparse.kron(matrices[0],angle,format='csr')
            phi+=p['scalar_coupling']*scale*(forward+forward.conj().T)
            for axis in range(3):
                angle=phase_matrix(rotor,g,edge,q,fields,derivative,axis)
                forward=sparse.kron(matrices[axis+1],angle,format='csr')
                one+=1j*p['vector_velocity']*scale*(forward-forward.conj().T)
    sign=-1 if order==1 else 1
    return TensorOperator(sign*one,sign*phi,scalar,field_energy=(order==0))


def route_current(op,rotor,scalar,p,g,edge):
    nb=op['Q'].shape[0]*10;one=sparse.csr_matrix((nb,nb),dtype=complex);phi=one.copy()
    for q,matrices in op['hops'][edge]:
        f=sparse.kron(matrices[0],sparse.eye(10),format='csr')
        phi-=1j*q*p['scalar_coupling']*g['weights'][edge]*(f-f.conj().T)
        for i in range(3):
            coordinate=rotor['integrate'](g['tangents'][:,edge,i]);f=sparse.kron(matrices[i+1],coordinate,format='csr')
            one+=q*p['vector_velocity']*g['weights'][edge]*(f+f.conj().T)
    return TensorOperator(one,phi,scalar)


def observable(matter,rotor,scalar,orientation_matrix=None):
    coordinate=sparse.eye(10) if orientation_matrix is None else orientation_matrix
    one=clean(sparse.kron(matter,coordinate,format='csr'))
    return TensorOperator(one,sparse.csr_matrix(one.shape,dtype=complex),scalar)


def source_operator(source,rotor,scalar):
    return clean(sparse.kron(sparse.kron(source,sparse.eye(10),format='csr'),sparse.eye(len(scalar['grid'])),format='csr'))


def build(p,with_povm=True):
    loop=loop_field();scalar=scalar_field(p);rotor=so3(p);geometry=graph(p,rotor)
    sectors={'neutral':gauge_sector(masks_for((0,0,0,0),p['pair_cut']),loop),
        'baryon':gauge_sector(masks_for((2,1,0,0),p['pair_cut']),loop),
        'lepton':gauge_sector(masks_for((0,0,0,1),p['pair_cut']),loop)}
    for name,anti in [('baryon','antibaryon'),('lepton','antilepton')]:sectors[anti]=conjugate_sector(sectors[name],loop)
    ops={}
    for name,sector in sectors.items():
        op=matter_ops(sector,loop,p,geometry,with_povm);op['connection']=connection_data(op,rotor,p)
        op['J']=[clean(sparse.kron(s,sparse.eye(10),format='csr')+sparse.kron(sparse.eye(sector['dimension']),l,format='csr')) for s,l in zip(op['S'],rotor['L'])]
        op['J2']=clean(sum(j@j for j in op['J']));ops[name]=op
    sources={}
    for name,kind,removal in [('baryon','baryon',False),('lepton','lepton',False),('antibaryon','baryon',True),('antilepton','lepton',True)]:
        sources[name]={f'{site}:{spin}':source_operator(site_source(sectors['neutral'],sectors[name],site,kind,spin,removal),rotor,scalar)
            for site in range(2) for spin in ([.5,1.5] if kind=='baryon' else [.5])}
        if kind=='baryon':
            for label,sites in [('split1',(0,0,1)),('split2',(0,1,1))]:
                for spin in [.5,1.5]:
                    sources[name][f'{label}:{spin}']=source_operator(site_source(sectors['neutral'],sectors[name],sites,kind,spin,removal),rotor,scalar)
    c={}
    for name,anti in [('neutral','neutral'),('baryon','antibaryon'),('lepton','antilepton')]:
        matter=clean(sectors[anti]['P'].conj().T@raw_C(sectors[name]['masks'],sectors[anti]['masks'])@sectors[name]['P'])
        c[name]=source_operator(matter,rotor,scalar)
    root=common_root();root['inputs']='NEW spatial law: original ordered mirror legs are physical vertices, not internal rest-mass labels. Quantum phi is bond order; actual Y/frame transport hops between sites. Primitive repeated4, fixed3+1, flavour/Q/gauging remain inputs. Lepton route now IDENTITY, not old TrU/3. No inherited energies, sources, old C or global vacuum.'
    return dict(loop=loop,scalar=scalar,rotor=rotor,geometry=geometry,sectors=sectors,operators=ops,sources=sources,C=c,root=root)
