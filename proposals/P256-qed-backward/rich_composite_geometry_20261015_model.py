"""NEW occupied CAR64 tetrahedral-cycle constituent/flux model; no old state reuse.
Packets are sparse FULL physical-site CAR determinants x PW19, not valence particles.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from itertools import product
from math import sqrt
import numpy as np
from scipy import sparse
from rich_colour_composite_20261014_model import generators, loop_field, epsilon
from rich_common_colour_lepton_20261015_model import apply, string, scalar_field
from rich_spatial_colour_root_20261015_model import so3

Packet = dict[tuple[int, int], complex]
PAULI = (np.array([[0,1],[1,0]],complex), np.array([[0,-1j],[1j,0]],complex), np.diag([1.,-1.]))
EDGES = ((0,1),(1,2),(2,3),(0,3))


def mode(site, flavour, spin, member):
    return 16*site+8*flavour+4*spin+member


def unpack(index):
    site, rest = divmod(index,16)
    flavour, rest = divmod(rest,8)
    spin, member = divmod(rest,4)
    return site,flavour,spin,member


def trim(packet, tol=2e-14):
    return {k:complex(v) for k,v in packet.items() if abs(v)>tol}


def add(*terms):
    out=defaultdict(complex)
    for coefficient, packet in terms:
        for key,value in packet.items():out[key]+=coefficient*value
    return trim(out)


def inner(left,right):
    if len(left)>len(right):return np.conjugate(inner(right,left))
    return sum(np.conjugate(value)*right.get(key,0.) for key,value in left.items())


@lru_cache(maxsize=262144)
def label(mask):
    counts=[0]*16;twice_m=0
    for index in range(64):
        if mask&(1<<index):
            site,f,s,c=unpack(index);counts[4*site+2*f+(c==3)]+=1;twice_m+=1-2*s
    return tuple(counts)+(twice_m,)


def split(packet):
    out={}
    for key,value in packet.items():out.setdefault(label(key[0]),{})[key]=value
    return out


def geometry(p):
    vertices=np.array([[1,1,1],[1,-1,-1],[-1,-1,1],[-1,1,-1]],float)/sqrt(8)
    if p['geometry']=='stretched':vertices[3]+=np.array([0.,0.,2.])
    elif p['geometry']!='tetrahedron':raise ValueError('Unknown physical geometry')
    vertices-=vertices.mean(axis=0)
    displacement=np.array([vertices[b]-vertices[a] for a,b in EDGES])
    lengths=np.linalg.norm(displacement,axis=1);axes=displacement/lengths[:,None]
    areas=np.array([np.cross(vertices[a],vertices[b])/2 for a,b in EDGES])
    # m_vertex=3 fixes trace-average two-site old I=1; no spin/order fitting.
    inertia=p['vertex_mass_density']*sum((np.dot(r,r)*np.eye(3)-np.outer(r,r) for r in vertices),np.zeros((3,3)))
    return dict(vertices=vertices,edges=EDGES,lengths=lengths,axes=axes,weights=1/lengths,
        areas=areas,displacement=displacement,inertia=inertia,inertia_inverse=np.linalg.inv(inertia))


class Primitives:
    def __init__(self,p):
        self.p=p;self.graph=geometry(p);self.loop=loop_field();self.ts=generators();self.maps={}
        for edge in range(4):
            for f,white,component,dagger in product(range(2),(False,True),range(4),(False,True)):
                spin=np.eye(2) if component==0 else PAULI[component-1]
                transitions=[]
                for a,b,s,t in product(([3] if white else range(3)),([3] if white else range(3)),range(2),range(2)):
                    if (edge!=3 or white) and a!=b:continue
                    if spin[s,t]==0:continue
                    start,end=EDGES[edge]
                    operations=[(mode(end,f,s,a),True),(mode(start,f,t,b),False)]
                    field=np.eye(19) if edge!=3 or white else self.loop['U'][b,a].conj().T
                    coefficient=spin[s,t]
                    if dagger:
                        operations=[(mode(start,f,t,b),True),(mode(end,f,s,a),False)]
                        field=field.conj().T;coefficient=coefficient.conjugate()
                    maps=self.field_map(field)
                    transitions.append((operations,coefficient,maps))
                self.maps[edge,f,white,component,dagger]=transitions
        self.colour_terms={}
        for sites in (tuple(range(4)),(1,2,3),(2,3),(3,)):
            self.colour_terms[sites]=[[(mode(v,f,s,a),mode(v,f,s,b),t[a,b])
                for v,f,s,a,b in product(sites,range(2),range(2),range(3),range(3)) if t[a,b]!=0] for t in self.ts]
        self.field_maps={k:self.field_map(self.loop[k]) for k in ('B','C','boundary')}
        self.field_maps.update({('R',a):self.field_map(self.loop['R'][a]) for a in range(8)})
        self.field_maps.update({('G',a):self.field_map(self.loop['L'][a]+self.loop['R'][a]) for a in range(8)})

    @staticmethod
    def field_map(matrix):
        return [[(int(r),complex(matrix[r,c])) for r in np.flatnonzero(abs(matrix[:,c])>1e-14)] for c in range(19)]

    @staticmethod
    def field(packet,maps):
        out=defaultdict(complex)
        for (mask,pw),value in packet.items():
            for target,amplitude in maps[pw]:out[mask,target]+=amplitude*value
        return trim(out)

    @staticmethod
    def fermion(packet,terms):
        out=defaultdict(complex)
        for (mask,pw),value in packet.items():
            for a,b,coefficient in terms:
                step=string(mask,[(a,True),(b,False)])
                if step is not None:out[step[0],pw]+=coefficient*step[1]*value
        return trim(out)

    def hop(self,packet,edge,f,white,component,dagger=False):
        out=defaultdict(complex)
        for operations,coefficient,maps in self.maps[edge,f,white,component,dagger]:
            for (mask,pw),value in packet.items():
                step=string(mask,operations)
                if step is not None:
                    for q,amplitude in maps[pw]:out[step[0],q]+=coefficient*step[1]*amplitude*value
        return trim(out)

    def collective(self,packet,component=0,body_axis=0,charge=None,edge_only=None):
        terms=[]
        for edge,f,white in product(range(4),range(2),(False,True)):
            if edge_only is not None and edge!=edge_only:continue
            q=([0.,-1.] if white else [2/3,-1/3])[f]
            factor=self.graph['weights'][edge]
            if charge=='EM':factor*=q
            elif charge=='baryon':factor*=0. if white else 1/3
            elif charge is not None:raise ValueError('Typed current charge unknown')
            if component:factor*=self.graph['axes'][edge,body_axis]
            forward=self.hop(packet,edge,f,white,component)
            backward=self.hop(packet,edge,f,white,component,True)
            if charge is None:
                terms.extend(((factor if component==0 else 1j*factor,forward),
                    (factor if component==0 else -1j*factor,backward)))
            else:
                # -d/dtheta of exp(i q theta)F plus adjoint.
                terms.extend(((-1j*factor if component==0 else factor,forward),
                    (1j*factor if component==0 else factor,backward)))
        return add(*terms)

    def electric(self,packet):
        out=[]
        for edge,sites in enumerate(((1,2,3),(2,3),(3,))):
            for a in range(8):
                def action(v):
                    return add((1,self.fermion(v,self.colour_terms[sites][a])),(1,self.field(v,self.field_maps['R',a])))
                out.append((self.p['electric']*self.graph['lengths'][edge]/2,action(action(packet))))
        out.append((self.p['electric']*self.graph['lengths'][3]/2,self.field(packet,self.field_maps['C'])))
        return add(*out)

    def gauss(self,packet,a):
        return add((1,self.fermion(packet,self.colour_terms[tuple(range(4))][a])),
            (1,self.field(packet,self.field_maps['G',a])))

    def spin(self,packet,component):
        terms=[(mode(v,f,s,c),mode(v,f,t,c),PAULI[component][s,t]/2)
            for v,f,c,s,t in product(range(4),range(2),range(4),range(2),range(2)) if PAULI[component][s,t]!=0]
        return self.fermion(packet,terms)


def conjugate(packet):
    out=defaultdict(complex)
    for (mask,pw),value in packet.items():
        target=(1<<64)-1;amplitude=1
        for index in reversed([i for i in range(64) if mask&(1<<i)]):
            v,f,s,c=unpack(index)
            target,sign=apply(target,mode(v,f,1-s,c),False)
            amplitude*=sign*((-1)**(v+s))
        q=0 if pw==0 else pw+9 if pw<10 else pw-9
        out[target,q]+=amplitude*value
    return trim(out)


def occupied_seed():
    # Every vertex has8 occupied modes/6colours, no empty-valence-particle vacuum.
    mask=sum(1<<mode(v,v%2,s,c) for v,s,c in product(range(4),range(2),range(4)))
    return {(mask,0):1.+0j}


def neutral_generators(primitives):
    """Full degree2 physical H ingredients in neutral J0-compatible S0/S1 range.
    R j0+j1 cannot couple neutral S2 to J0. No charged spin family selected out.
    """
    o=occupied_seed();x=primitives.collective(o);result=[o,x,primitives.collective(x)]
    v={(i,a):primitives.collective(o,i+1,a) for i,a in product(range(3),repeat=2)}
    result.extend(v.values())
    result.extend(primitives.collective(value) for value in v.values())
    result.extend(primitives.collective(x,i+1,a) for i,a in product(range(3),repeat=2))
    for a,b in product(range(3),repeat=2):
        result.append(add(*[(1,primitives.collective(v[i,b],i+1,a)) for i in range(3)]))
        for k in range(3):
            result.append(add(*[(epsilon(k,i,j),primitives.collective(v[j,b],i+1,a))
                for i,j in product(range(3),repeat=2) if epsilon(k,i,j)]))
    b=primitives.field(o,primitives.field_maps['B']);result.append(b)
    result.append(primitives.field(b,primitives.field_maps['B']))
    result.append(primitives.electric(b))
    for value in [x,*v.values()]:
        result.append(primitives.field(value,primitives.field_maps['B']))
        result.append(primitives.electric(value))
    result.append(primitives.collective(b))
    result.extend(primitives.collective(b,i+1,a) for i,a in product(range(3),repeat=2))
    return [value for packet in result if packet for value in (packet,conjugate(packet))]


def spin_amplitudes(kind,twice_m):
    if kind=='threehalf':
        return {spins:1/sqrt(3) for spins in product(range(2),repeat=3) if sum(1-2*s for s in spins)==twice_m} if abs(twice_m)==1 else {((0,0,0) if twice_m==3 else (1,1,1)):1.}
    if twice_m not in (-1,1):raise ValueError('Halfspin M')
    if kind=='half_triplet':
        out={(0,0,1):sqrt(2/3),(0,1,0):-sqrt(1/6),(1,0,0):-sqrt(1/6)}
    elif kind=='half_singlet':out={(0,1,0):1/sqrt(2),(1,0,0):-1/sqrt(2)}
    else:raise ValueError('Spin source unknown')
    if twice_m==1:return out
    # S_- of a spinhalf highest state; no arbitrary all-spin-flip phase.
    lower=defaultdict(complex)
    for spins,value in out.items():
        for i in range(3):
            if spins[i]==0:
                target=list(spins);target[i]=1;lower[tuple(target)]+=value
    return trim(lower)


def source_terms(positions,kind,twice_m):
    out=[]
    for colours in product(range(3),repeat=3):
        e=epsilon(*colours)
        if e:
            for spins,value in spin_amplitudes(kind,twice_m).items():
                out.append(([(mode(v,f,s,c),True) for v,f,s,c in zip(positions,(0,0,1),spins,colours)],e*value/sqrt(6)))
    return out


def source(packet,positions,kind,twice_m,removal=False):
    out=defaultdict(complex)
    for operations,value in source_terms(positions,kind,twice_m):
        if removal:
            transformed=[]
            for index,_ in operations:
                v,f,s,c=unpack(index);value*=(-1)**(v+s)
                transformed.append((mode(v,f,1-s,c),False))
            operations=transformed
        for (mask,pw),amplitude in packet.items():
            step=string(mask,operations)
            if step is not None:out[step[0],pw]+=value*step[1]*amplitude
    return trim(out)


def source_specs():
    for a,b,c in product(range(4),repeat=3):
        if a>b:continue
        for kind in ('half_triplet','half_singlet','threehalf'):
            if kind=='half_singlet' and a==b:continue
            for m in ((-3,-1,1,3) if kind=='threehalf' else (-1,1)):
                yield ((a,b,c),kind,m)


class PacketBasis:
    """Streaming twice-MGS within exact site-flavour/colourwhite counts and Sz.
    No dense global Gram and no global Fock enumeration. Numerical-null cut only.
    """
    def __init__(self):
        self.vectors=[];self.labels=[];self.blocks={};self.inverted={};self.attempts=0;self.nulls=0;self.minimum_accepted=1.

    def append(self,packet):
        added=0
        for lab,part in split(packet).items():
            self.attempts+=1;n0=float(inner(part,part).real)
            if n0<1e-26:self.nulls+=1;continue
            block=self.blocks.setdefault(lab,[]);inverse=self.inverted.setdefault(lab,{})
            v=part.copy()
            for _ in range(2):
                coefficients=defaultdict(complex)
                for key,value in v.items():
                    for index,a in inverse.get(key,()):coefficients[index]+=np.conjugate(a)*value
                for index,coefficient in coefficients.items():
                    for key,value in self.vectors[index].items():v[key]=v.get(key,0.)-coefficient*value
                v=trim(v)
            norm=float(inner(v,v).real)
            if norm<1e-20*n0 or norm<1e-25:self.nulls+=1;continue
            self.minimum_accepted=min(self.minimum_accepted,norm/n0)
            v={key:value/sqrt(norm) for key,value in v.items()};index=len(self.vectors)
            self.vectors.append(v);self.labels.append(lab);block.append(index)
            for key,value in v.items():inverse.setdefault(key,[]).append((index,value))
            added+=1
        return added

    def coordinates(self,packet):
        out=defaultdict(complex)
        for lab,part in split(packet).items():
            inverse=self.inverted.get(lab,{})
            for key,value in part.items():
                for index,a in inverse.get(key,()):out[index]+=np.conjugate(a)*value
        return trim(out)

    def matrix(self,action,target=None):
        target=self if target is None else target
        rows=[];columns=[];values=[]
        for j,packet in enumerate(self.vectors):
            for i,value in target.coordinates(action(packet)).items():rows.append(i);columns.append(j);values.append(value)
        return sparse.csr_matrix((values,(rows,columns)),shape=(len(target.vectors),len(self.vectors)))

    def receipt(self):
        return dict(dimension=len(self.vectors),blocks=len(self.blocks),max_block=max(map(len,self.blocks.values()),default=0),
            packet_nonzeros=sum(map(len,self.vectors)),raw_support=len({k for p in self.vectors for k in p}),
            attempts=self.attempts,numerical_nulls=self.nulls,minimum_accepted_relative_norm2=self.minimum_accepted,
            exact_labels='per-site upcolour/upwhite/downcolour/downwhite counts plus total2Sz; h/Q/baryon are derived physical units')


def field_and_rotor(p):
    rotor=so3(p);scalar=scalar_field(p)
    # One uniform scalar mode over twice the old2-vertex volume.
    volume=p['scalar_volume_ratio']
    scalar['H']=scalar['kinetic']/volume+volume*scalar['potential']
    return scalar,rotor
