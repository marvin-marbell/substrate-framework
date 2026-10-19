"""Same finite packet family, coordinate-once dense SMALL block storage.
No global dense Gram; no source/state/rank/coupling cut to repair an allocation.
"""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from math import sqrt
import numpy as np
from rich_composite_geometry_20261015_model import PacketBasis as MatrixProtocol,split,trim


class Block:
    def __init__(self):
        self.keys=[];self.lookup={};self.q=np.zeros((0,0),complex);self.rank=0;self.columns=[]

    def rows(self,keys):
        for key in keys:
            if key not in self.lookup:self.lookup[key]=len(self.keys);self.keys.append(key)
        needed=len(self.keys)
        if needed>self.q.shape[0]:
            size=max(16,self.q.shape[0]*2)
            while size<needed:size*=2
            new=np.zeros((size,max(4,self.q.shape[1])),complex)
            new[:self.q.shape[0],:self.rank]=self.q[:,:self.rank];self.q=new

    def append(self,packet):
        self.rows(packet);n=len(self.keys);v=np.zeros(n,complex)
        for key,value in packet.items():v[self.lookup[key]]=value
        original=float(np.vdot(v,v).real)
        if original<1e-26:return None,0.
        q=self.q[:n,:self.rank]
        for _ in range(2):v-=q@(q.conj().T@v)
        norm=float(np.vdot(v,v).real)
        if norm<1e-20*original or norm<1e-25:return None,0.
        if self.rank==self.q.shape[1]:
            new=np.zeros((self.q.shape[0],max(4,2*self.rank)),complex);new[:,:self.rank]=self.q[:,:self.rank];self.q=new
        column=self.rank;self.q[:n,column]=v/sqrt(norm);self.rank+=1
        return column,norm/original

    def coordinates(self,packet):
        v=np.zeros(len(self.keys),complex)
        for key,value in packet.items():
            row=self.lookup.get(key)
            if row is not None:v[row]+=value
        return self.q[:len(self.keys),:self.rank].conj().T@v


class Vectors(Sequence):
    def __init__(self,basis):self.basis=basis
    def __len__(self):return len(self.basis.labels)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        lab,column=self.basis.locations[index];block=self.basis.storage[lab]
        values=block.q[:len(block.keys),column]
        return {block.keys[i]:complex(values[i]) for i in np.flatnonzero(abs(values)>2e-14)}


class PacketBasis(MatrixProtocol):
    def __init__(self,primitive=None):
        self.primitive=primitive;self.storage={};self.blocks={};self.labels=[];self.locations=[]
        self.attempts=0;self.nulls=0;self.minimum_accepted=1.

    @property
    def vectors(self):return Vectors(self)

    def split(self,packet):
        if self.primitive is None:return split(packet)
        out={}
        for key,value in packet.items():out.setdefault(self.primitive.label(key),{})[key]=value
        return out

    def append(self,packet):
        count=0
        for lab,part in self.split(packet).items():
            self.attempts+=1;block=self.storage.setdefault(lab,Block());column,relative=block.append(part)
            if column is None:self.nulls+=1;continue
            index=len(self.labels);self.labels.append(lab);self.locations.append((lab,column))
            self.blocks.setdefault(lab,[]).append(index);block.columns.append(index)
            self.minimum_accepted=min(self.minimum_accepted,relative);count+=1
        return count

    def coordinates(self,packet):
        result={}
        for lab,part in self.split(packet).items():
            block=self.storage.get(lab)
            if block is None:continue
            for index,value in zip(block.columns,block.coordinates(part)):
                if abs(value)>2e-14:result[index]=complex(value)
        return result

    def receipt(self):
        result={'dimension':len(self.labels),'blocks':len(self.blocks),'max_block':max(map(len,self.blocks.values()),default=0),
            'packet_nonzeros':sum(int(np.count_nonzero(abs(b.q[:len(b.keys),:b.rank])>2e-14)) for b in self.storage.values()),
            'raw_coordinate_rows_allocated':sum(len(b.keys) for b in self.storage.values()),
            'numeric_storage_bytes':sum(b.q.nbytes for b in self.storage.values()),
            'attempts':self.attempts,'numerical_nulls':self.nulls,'minimum_accepted_relative_norm2':self.minimum_accepted,
            'storage':'Exact same physical-count/Sz packet-family streaming twice-MGS; raw coordinate keys ONCE per block and complex128 matrix SMALL block columns. NO global Gram/inverse-copy dictionary, no changed sources/null thresholds/states/couplings.'}
        if self.primitive is not None:result['local_basis']=self.primitive.local.receipt()
        return result
