"""Exact SAME native packet family with sparse CSC block isometry columns.
Fixes the observed dense-block/conjugation allocation, NOT a physical cut.
"""
from collections.abc import Sequence
from math import sqrt
import numpy as np
from scipy import sparse
from rich_composite_geometry_20261015_compact import PacketBasis as DenseBasis


class Block:
    def __init__(self):
        self.keys=[];self.lookup={};self.matrix=sparse.csc_matrix((0,0),dtype=complex);self.rank=0;self.columns=[];self.tiny_discarded_norm2=0.
    @property
    def q(self):return self.matrix.toarray()
    def append(self,packet):
        for key in packet:
            if key not in self.lookup:self.lookup[key]=len(self.keys);self.keys.append(key)
        n=len(self.keys)
        if self.matrix.shape[0]!=n:self.matrix.resize((n,self.rank))
        v=np.zeros(n,complex)
        for key,value in packet.items():v[self.lookup[key]]=value
        original=float(np.vdot(v,v).real)
        if original<1e-26:return None,0.
        for _ in range(2):
            # qdag v=conj(qT conj(v)); NEVER allocate a dense conjugated q.
            coefficient=np.conjugate(self.matrix.T@np.conjugate(v));v-=self.matrix@coefficient
        norm=float(np.vdot(v,v).real)
        if norm<1e-20*original or norm<1e-25:return None,0.
        v/=sqrt(norm);keep=np.flatnonzero(abs(v)>2e-14)
        self.tiny_discarded_norm2+=float(np.linalg.norm(v[abs(v)<=2e-14])**2)
        column=sparse.csc_matrix((v[keep],keep,[0,len(keep)]),shape=(n,1))
        self.matrix=sparse.hstack((self.matrix,column),format='csc');index=self.rank;self.rank+=1
        return index,norm/original
    def coordinates(self,packet):
        v=np.zeros(len(self.keys),complex)
        for key,value in packet.items():
            row=self.lookup.get(key)
            if row is not None:v[row]+=value
        return np.conjugate(self.matrix.T@np.conjugate(v))


class Vectors(Sequence):
    def __init__(self,basis):self.basis=basis
    def __len__(self):return len(self.basis.labels)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        lab,column=self.basis.locations[index];block=self.basis.storage[lab];v=block.matrix.getcol(column)
        return {block.keys[int(row)]:complex(value) for row,value in zip(v.indices,v.data)}


class PacketBasis(DenseBasis):
    @property
    def vectors(self):return Vectors(self)
    def append(self,packet):
        added=0
        for lab,part in self.split(packet).items():
            self.attempts+=1;block=self.storage.setdefault(lab,Block());column,relative=block.append(part)
            if column is None:self.nulls+=1;continue
            index=len(self.labels);self.labels.append(lab);self.locations.append((lab,column));self.blocks.setdefault(lab,[]).append(index);block.columns.append(index)
            self.minimum_accepted=min(self.minimum_accepted,relative);added+=1
        return added
    def receipt(self):
        result={'dimension':len(self.labels),'blocks':len(self.blocks),'max_block':max(map(len,self.blocks.values()),default=0),
            'packet_nonzeros':sum(b.matrix.nnz for b in self.storage.values()),'raw_coordinate_rows_allocated':sum(len(b.keys) for b in self.storage.values()),
            'numeric_storage_bytes':sum(b.matrix.data.nbytes+b.matrix.indices.nbytes+b.matrix.indptr.nbytes for b in self.storage.values()),
            'tiny_coefficient_discarded_norm2_sum':sum(b.tiny_discarded_norm2 for b in self.storage.values()),
            'attempts':self.attempts,'numerical_nulls':self.nulls,'minimum_accepted_relative_norm2':self.minimum_accepted,
            'storage':'SparseCSC actual isometry blocks, keysONCE; twiceMGS with conjugated VECTOR instead of allocating dense q.conj.T. SAME2e-14 materialized-vector roundoff tolerance and relative1e-20 norm2 numerical-null threshold. No physical source/state/cut/coupling reduction.'}
        if self.primitive is not None:result['local_basis']=self.primitive.local.receipt()
        return result


def convert(old):
    result=PacketBasis(old.primitive);result.labels=old.labels;result.locations=old.locations;result.blocks=old.blocks
    result.attempts=old.attempts;result.nulls=old.nulls;result.minimum_accepted=old.minimum_accepted
    discarded=0.
    for lab in list(old.storage):
        previous=old.storage.pop(lab);block=Block();block.keys=previous.keys;block.lookup=previous.lookup;block.rank=previous.rank;block.columns=previous.columns
        q=previous.q[:len(block.keys),:block.rank];block.matrix=sparse.csc_matrix(q)
        tiny=abs(block.matrix.data)<=2e-14;lost=float(np.linalg.norm(block.matrix.data[tiny])**2)
        block.matrix.data[tiny]=0;block.matrix.eliminate_zeros();block.tiny_discarded_norm2=lost;discarded+=lost
        result.storage[lab]=block
        del q,previous
    return result,discarded
