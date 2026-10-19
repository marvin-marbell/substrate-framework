"""Rich source/sea/R/PW body isometry tensor FULL independent Phi7.
SparseCSC count/J blocks reuse the established twiceMGS/null/2e-14 convention.
Record the actual operator lift: |phi_out><phi_in| P_counts G_source on ALL inputs.
Count pieces complete the carrier gauge orbit, NOT a particle leakage definition.
"""
from pathlib import Path
import gc,json
import numpy as np
from scipy import sparse
from rich_composite_geometry_20261015_sparse_basis import Block
from rich_composite_geometry_20261015_dressing_sources import descriptors,apply_source
from rich_composite_geometry_20261015_cloud_probe import inner
from rich_composite_geometry_20261015_dressing_store import DiskBlocks,update_pickle


class OperatorBlock(Block):
    def __init__(self):
        super().__init__();self.lifts=[];self.nulls=0;self.null_residual_norm2=0.;self.minimum_accepted=1.
    def append_operator(self,packet,descriptor):
        for key in packet:
            if key not in self.lookup:self.lookup[key]=len(self.keys);self.keys.append(key)
        n=len(self.keys)
        if self.matrix.shape[0]!=n:self.matrix.resize((n,self.rank))
        v=np.zeros(n,complex)
        for key,value in packet.items():v[self.lookup[key]]=value
        original=float(np.vdot(v,v).real);coordinates=np.zeros(self.rank,complex);lift={descriptor:1.+0j}
        if original<1e-26:self.nulls+=1;self.null_residual_norm2+=original;return coordinates
        for _ in range(2):
            coefficient=np.conjugate(self.matrix.T@np.conjugate(v));coordinates+=coefficient;v-=self.matrix@coefficient
            for column,value in enumerate(coefficient):
                if value:
                    for source,weight in self.lifts[column].items():lift[source]=lift.get(source,0.)-value*weight
        norm=float(np.vdot(v,v).real)
        if norm<1e-20*original or norm<1e-25:self.nulls+=1;self.null_residual_norm2+=norm;return coordinates
        scale=np.sqrt(norm);v/=scale;keep=np.flatnonzero(abs(v)>2e-14);self.tiny_discarded_norm2+=float(np.linalg.norm(v[abs(v)<=2e-14])**2)
        column=sparse.csc_matrix((v[keep],keep,[0,len(keep)]),shape=(n,1));self.matrix=sparse.hstack((self.matrix,column),format='csc');self.rank+=1
        self.lifts.append({key:value/scale for key,value in lift.items() if value!=0});self.minimum_accepted=min(self.minimum_accepted,norm/original)
        return np.r_[coordinates,scale]


def construct(primitive,omega,config,scalar,rotor,out):
    storage=DiskBlocks(out);probes=[];generators=list(descriptors(config));nf=primitive.p['scalar_points']
    for index,descriptor in enumerate(generators):
        state=apply_source(primitive,omega,descriptor,scalar,rotor);raw=float(inner(state,state).real);pieces={};count_parts={}
        for key,field in state.items():count_parts.setdefault(primitive.label(key)[:-1],{})[key]=field
        for counts,part in count_parts.items():
            label=(descriptor['twice_J'],counts);block=None
            for phi in range(nf):
                packet={}
                for key,field in part.items():
                    values=field[:,phi]
                    for r in np.flatnonzero(values):packet[key+(int(r),)]=complex(values[r])
                if not packet:continue
                if block is None:block=storage[label] if label in storage else storage.create(label,OperatorBlock())
                pieces.setdefault(label,{})[phi]=block.append_operator(packet,(index,phi))
        probes.append({'descriptor':descriptor,'raw_strength':raw,'pieces':pieces});storage.flush(True);del state,count_parts;gc.collect();statistics=storage.stats()
        if descriptor['family']=='eom' or index==len(generators)-1:
            update_pickle(out+'_body_construction_checkpoint.pkl',{'p':primitive.p,'primitive':primitive,'storage':storage,'probes':probes,'generators':generators,'next_generator':index+1,'config':config})
        print(json.dumps({'stage':'rich_source_FULL_independent_Phi_release','generator':index,'family':descriptor['family'],'twice_J':descriptor['twice_J'],
            'body_dimension':statistics['dimension'],'body_nonzeros':statistics['nnz'],'body_count_J_blocks':len(storage),'storage':'EXACT disk blocks, not a physical cut'}),flush=True)
    offset=0;labels=[]
    for label in storage:
        block=storage[label];block.columns=list(range(offset,offset+block.rank));offset+=block.rank;block.matrix=block.matrix.tocsr()
        labels.extend({'twice_J':label[0],'counts':label[1]} for _ in range(block.rank));storage.flush(True)
    statistics=storage.stats()
    receipt={'body_dimension':offset,'full_independent_Phi':nf,'full_coordinate_dimension':offset*nf,'count_J_blocks':len(storage),'source_generators':len(generators),
        'body_numeric_bytes':statistics['numeric_bytes'],'body_nonzeros':statistics['nnz'],'max_body_count_block':statistics['max_rank'],
        'tiny_body_coefficient_discarded_norm2_sum':statistics['tiny_discarded_norm2'],'null_input_residual_norm2_sum':statistics['null_residual_norm2'],
        'numerical_nulls':statistics['nulls'],'minimum_accepted_relative_norm2':statistics['minimum_accepted'],
        'storage':'Flat disk-backed count/J blocks, one mutable CSC block during construction; blockwise CSR conversion, never global CSC+CSR residency. ALL blocks included in natural capture. Same68 source lists/Phi/null tolerances/couplings as failed in-memory variant.',
        'definition':'span of ALL7 Phi slices of B,(2J.Rd)B,complexTrU B on20 declared anchors plus LOCAL bare D=[H0,B] on4 compact anchors/both spins; tensor FULL independent7 Phi output coordinates. PhiB/PphiB included by full matrix-unit factor, not a twolevel scalar replacement. Entire actual occupied Omega enters every source before count-orbit completion. Additional severe source/sea/R/PW body-rank and anchor cuts remain; no full all-pair/native range claim.',
        'operator_lift':'Each q_body tensor|phi_out> has literal sum lift[(source_generator,phi_in)] |phi_out><phi_in| P_counts G_source operator extension. Sparse2e-14 trim/null discarded norms disclosed; no arbitrary state-only extension or fitted energy term.'}
    Path(out+'_basis_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return storage,labels,probes,generators,receipt


def source_coordinates(storage,probes,nbody,nphi):
    result=[]
    for probe in probes:
        vector=np.zeros((nbody,nphi),complex)
        for label,columns in probe['pieces'].items():
            columns_global=storage.meta[label]['columns']
            for phi,values in columns.items():vector[np.asarray(columns_global[:len(values)]),phi]=values
        result.append({**{key:value for key,value in probe.items() if key!='pieces'},'coordinates':vector})
    return result
