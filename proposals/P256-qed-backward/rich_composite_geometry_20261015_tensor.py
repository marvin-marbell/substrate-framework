"""Material alternative: correlated local six-colour/two-white occupied singlets.
Exact adaptive LOCAL Fock coordinates factor the4-site sea; no global expansion.
The microscopic CAR64/Phi/colour/geometry Hamiltonian is unchanged.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import product
from math import sqrt,comb
import numpy as np
import rich_composite_geometry_20261015_model as raw


def local_label(mask):
    counts=[0]*4;m=0
    for i in range(16):
        if mask&(1<<i):
            f=i//8;s=(i%8)//4;c=i%4;counts[2*f+(c==3)]+=1;m+=1-2*s
    return tuple(counts)+(m,)


def local_seed():
    coloured=defaultdict(complex)
    for udown in range(4):
        up={};down={}
        for spins in product(range(2),repeat=3):
            if sum(spins)==udown:
                for colours in product(range(3),repeat=3):
                    e=raw.epsilon(*colours)
                    if e:
                        step=raw.string(0,[(4*s+c,True) for s,c in zip(spins,colours)])
                        up[step[0]]=up.get(step[0],0.)+e*step[1]/sqrt(6*comb(3,udown))
            if sum(spins)==3-udown:
                for colours in product(range(3),repeat=3):
                    e=raw.epsilon(*colours)
                    if e:
                        step=raw.string(0,[(8+4*s+c,True) for s,c in zip(spins,colours)])
                        down[step[0]]=down.get(step[0],0.)+e*step[1]/sqrt(6*comb(3,3-udown))
        nu=sqrt(sum(abs(v)**2 for v in up.values()));nd=sqrt(sum(abs(v)**2 for v in down.values()))
        for a,b in product(up,down):coloured[a|b]+=(-1)**udown*up[a]*down[b]/(2*nu*nd)
    out=defaultdict(complex)
    for mask,value in coloured.items():
        for operations,coefficient in [([(3,True),(15,True)],1/sqrt(2)), ([(7,True),(11,True)],-1/sqrt(2))]:
            step=raw.string(mask,operations)
            out[step[0]]+=value*coefficient*step[1]
    norm=sqrt(sum(abs(v)**2 for v in out.values()))
    return raw.trim({k:v/norm for k,v in out.items()})


class LocalBasis:
    def __init__(self):
        self.vectors=[];self.labels=[];self.blocks={};self.inverse={};self.cache={};self.discarded_norm2=0.
        coefficients=self.resolve(local_seed())
        if set(coefficients)!={0}:raise ArithmeticError('Local occupied seed coordinate')

    def resolve(self,packet):
        groups={}
        for mask,value in packet.items():groups.setdefault(local_label(mask),{})[mask]=value
        result=defaultdict(complex)
        for lab,part in groups.items():
            inverse=self.inverse.setdefault(lab,{});v=part.copy();original=float(raw.inner(part,part).real)
            for _ in range(2):
                c=defaultdict(complex)
                for key,value in v.items():
                    for index,a in inverse.get(key,()):c[index]+=np.conjugate(a)*value
                for index,a in c.items():
                    result[index]+=a
                    for key,value in self.vectors[index].items():v[key]=v.get(key,0.)-a*value
                v=raw.trim(v)
            norm=float(raw.inner(v,v).real)
            if norm>max(1e-25,1e-22*original):
                v={key:value/sqrt(norm) for key,value in v.items()};index=len(self.vectors)
                self.vectors.append(v);self.labels.append(lab);self.blocks.setdefault(lab,[]).append(index)
                for key,value in v.items():inverse.setdefault(key,[]).append((index,value))
                result[index]+=sqrt(norm)
            else:self.discarded_norm2+=norm
        return raw.trim(result)

    def fermion(self,index,create,column):
        key=(index,create,column)
        if key not in self.cache:
            out=defaultdict(complex)
            for mask,value in self.vectors[column].items():
                step=raw.apply(mask,index,create)
                if step is not None:out[step[0]]+=value*step[1]
            self.cache[key]=self.resolve(raw.trim(out))
        return self.cache[key]

    def conjugate(self,column):
        key=('C',column)
        if key not in self.cache:
            out=defaultdict(complex)
            for mask,value in self.vectors[column].items():
                target=(1<<16)-1;amplitude=1
                for index in reversed([i for i in range(16) if mask&(1<<i)]):
                    f=index//8;s=(index%8)//4;c=index%4
                    target,sign=raw.apply(target,8*f+4*(1-s)+c,False);amplitude*=sign*((-1)**s)
                out[target]+=amplitude*value
            self.cache[key]=self.resolve(raw.trim(out))
        return self.cache[key]

    def receipt(self):
        return {'dimension':len(self.vectors),'max_local_block':max(map(len,self.blocks.values())),
            'raw_local_nonzeros':sum(map(len,self.vectors)),'cached_local_operations':len(self.cache),
            'local_roundoff_discarded_norm2_sum':self.discarded_norm2,
            'meaning':'Adaptive ORTHONORMAL local16-mode Fock coordinates, no local physical-state/energy cut. Every exact c/cdag image is resolved including its new orthogonal remainder BEFORE global packet projection. Global2^64 sea never expanded.'}


class Primitives(raw.Primitives):
    def __init__(self,p,local=None):
        super().__init__(p);self.local=LocalBasis() if local is None else local

    def coordinate_operations(self,key,operations):
        current={key:1.+0j}
        for index,create in reversed(operations):
            site,local_index=divmod(index,16);out=defaultdict(complex)
            for state,value in current.items():
                parity=sum(sum(self.local.labels[state[v]][:4]) for v in range(site))%2
                for target,a in self.local.fermion(local_index,create,state[site]).items():
                    changed=list(state);changed[site]=target;out[tuple(changed)]+=(-1)**parity*value*a
            current=raw.trim(out)
        return current

    def field(self,packet,maps):
        out=defaultdict(complex)
        for key,value in packet.items():
            for pw,a in maps[key[-1]]:out[key[:-1]+(pw,)]+=value*a
        return raw.trim(out)

    def fermion(self,packet,terms):
        out=defaultdict(complex)
        for key,value in packet.items():
            for a,b,coefficient in terms:
                for target,sign in self.coordinate_operations(key,[(a,True),(b,False)]).items():out[target]+=value*coefficient*sign
        return raw.trim(out)

    def hop(self,packet,edge,f,white,component,dagger=False):
        out=defaultdict(complex)
        for operations,coefficient,maps in self.maps[edge,f,white,component,dagger]:
            for key,value in packet.items():
                for target,sign in self.coordinate_operations(key,operations).items():
                    for pw,a in maps[target[-1]]:out[target[:-1]+(pw,)]+=value*coefficient*sign*a
        return raw.trim(out)

    def conjugate(self,packet):
        out=defaultdict(complex)
        for key,value in packet.items():
            state={key:value}
            for site in range(4):
                changed=defaultdict(complex)
                for original,a in state.items():
                    n=sum(self.local.labels[original[site]][:4])
                    for target,b in self.local.conjugate(original[site]).items():
                        row=list(original);row[site]=target;changed[tuple(row)]+=a*b*((-1)**(site*n))
                state=raw.trim(changed)
            for row,a in state.items():
                pw=row[-1];pw=0 if pw==0 else pw+9 if pw<10 else pw-9
                out[row[:-1]+(pw,)]+=a
        return raw.trim(out)

    def label(self,key):
        labs=[self.local.labels[i] for i in key[:4]]
        return tuple(x for lab in labs for x in lab[:4])+(sum(lab[-1] for lab in labs),)

    def source(self,packet,positions,kind,m2,removal=False):
        out=defaultdict(complex)
        for operations,coefficient in raw.source_terms(positions,kind,m2):
            if removal:
                transformed=[]
                for index,_ in operations:
                    v,f,s,c=raw.unpack(index);coefficient*=(-1)**(v+s)
                    transformed.append((raw.mode(v,f,1-s,c),False))
                operations=transformed
            for key,value in packet.items():
                for target,a in self.coordinate_operations(key,operations).items():out[target]+=value*coefficient*a
        return raw.trim(out)


class PacketBasis(raw.PacketBasis):
    def __init__(self,primitive):
        super().__init__();self.primitive=primitive

    def split(self,packet):
        out={}
        for key,value in packet.items():out.setdefault(self.primitive.label(key),{})[key]=value
        return out

    def append(self,packet):
        added=0
        for lab,part in self.split(packet).items():
            self.attempts+=1;n0=float(raw.inner(part,part).real)
            if n0<1e-26:self.nulls+=1;continue
            block=self.blocks.setdefault(lab,[]);inverse=self.inverted.setdefault(lab,{})
            v=part.copy()
            for _ in range(2):
                coefficients=defaultdict(complex)
                for key,value in v.items():
                    for index,a in inverse.get(key,()):coefficients[index]+=np.conjugate(a)*value
                for index,a in coefficients.items():
                    for key,value in self.vectors[index].items():v[key]=v.get(key,0.)-a*value
                v=raw.trim(v)
            norm=float(raw.inner(v,v).real)
            if norm<1e-20*n0 or norm<1e-25:self.nulls+=1;continue
            self.minimum_accepted=min(self.minimum_accepted,norm/n0)
            v={key:value/sqrt(norm) for key,value in v.items()};index=len(self.vectors)
            self.vectors.append(v);self.labels.append(lab);block.append(index)
            for key,value in v.items():inverse.setdefault(key,[]).append((index,value))
            added+=1
        return added

    def coordinates(self,packet):
        out=defaultdict(complex)
        for lab,part in self.split(packet).items():
            inverse=self.inverted.get(lab,{})
            for key,value in part.items():
                for index,a in inverse.get(key,()):out[index]+=np.conjugate(a)*value
        return raw.trim(out)

    def receipt(self):
        result=super().receipt();result['raw_support_meaning']='ORTHONORMAL local-Fock tensor/PW coordinates, not enumerated global Slaters'
        result['local_basis']=self.primitive.local.receipt()
        return result


def neutral_generators(primitive):
    o={(0,0,0,0,0):1.+0j};x=primitive.collective(o);result=[o,x,primitive.collective(x)]
    v={(i,a):primitive.collective(o,i+1,a) for i,a in product(range(3),repeat=2)}
    result.extend(v.values());result.extend(primitive.collective(value) for value in v.values())
    result.extend(primitive.collective(x,i+1,a) for i,a in product(range(3),repeat=2))
    for a,b in product(range(3),repeat=2):
        result.append(raw.add(*[(1,primitive.collective(v[i,b],i+1,a)) for i in range(3)]))
        for k in range(3):result.append(raw.add(*[(raw.epsilon(k,i,j),primitive.collective(v[j,b],i+1,a)) for i,j in product(range(3),repeat=2) if raw.epsilon(k,i,j)]))
    b=primitive.field(o,primitive.field_maps['B']);result.append(b)
    result.append(primitive.field(b,primitive.field_maps['B']));result.append(primitive.electric(b))
    for value in [x,*v.values()]:
        result.append(primitive.field(value,primitive.field_maps['B']));result.append(primitive.electric(value))
    result.append(primitive.collective(b));result.extend(primitive.collective(b,i+1,a) for i,a in product(range(3),repeat=2))
    return [value for packet in result if packet for value in (packet,primitive.conjugate(packet))]
