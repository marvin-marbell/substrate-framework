"""Sparse weak primitive H on rich body(CAR/PW/R) tensor independentPhi7.
No global dense Gram/operator/Hpsi cloud; source-count blocks bound memory.
FULL local colour squares precede Galerkin target selection. Source R/U fields
receive actual mechanical/electric/plaquette/hop action, no frozen derivative.
"""
from functools import lru_cache
from itertools import product
from pathlib import Path
import gc,json,time
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_cloud_probe import LocalElectric


class Collector:
    def __init__(self,dimension):self.dimension=dimension;self.rows=[];self.columns=[];self.values=[];self.discarded_norm2=0.
    def add_block(self,rows,columns,value,threshold):
        keep=abs(value)>threshold;self.discarded_norm2+=float(np.linalg.norm(value[~keep])**2);r,c=np.nonzero(keep)
        if len(r):self.rows.append(np.asarray(rows,dtype=np.int32)[r]);self.columns.append(np.asarray(columns,dtype=np.int32)[c]);self.values.append(value[r,c])
    def finish(self):
        if not self.values:return sparse.csr_matrix((self.dimension,self.dimension),dtype=complex)
        result=sparse.coo_matrix((np.concatenate(self.values),(np.concatenate(self.rows),np.concatenate(self.columns))),shape=(self.dimension,self.dimension)).tocsr()
        self.rows.clear();self.columns.clear();self.values.clear();return result


def assemble(primitive,storage,labels,rotor,out):
    started=time.time();dimension=len(labels);p=primitive.p;electric=LocalElectric(primitive)
    names=('Gram','rotation','electric','plaquette','quark_vector','white_vector','quark_scalar','white_scalar','loopB','loopC')
    collectors={name:Collector(dimension) for name in names};generators=angular.right_angular(rotor);inverse=np.asarray(primitive.graph['inertia_inverse']);kinetic=np.zeros((10,10),complex)
    for a,b in np.ndindex(3,3):kinetic+=inverse[a,b]*(generators[a]@generators[b]).toarray()/2
    axes=np.asarray(primitive.graph['axes']);weights=np.asarray(primitive.graph['weights']);rotations={(edge,i):sum((axes[edge,a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for edge,i in product(range(4),range(3))}
    empty=(np.empty(0,dtype=np.int32),np.zeros((0,10),complex))
    def packet(block,key):
        pieces=[];columns=set();matrix=block.matrix
        for r in range(10):
            row=block.lookup.get(key+(r,))
            if row is not None:
                start,end=matrix.indptr[row:row+2];ix=matrix.indices[start:end];values=matrix.data[start:end]
                pieces.append((r,ix,values));columns.update(map(int,ix))
        if not columns:return empty
        columns=np.asarray(sorted(columns),dtype=np.int32);position={int(column):index for index,column in enumerate(columns)};fields=np.zeros((len(columns),10),complex)
        for r,indices,values in pieces:
            for column,value in zip(indices,values):fields[position[int(column)],r]=value
        return columns,fields
    counters={'primitive_hop_calls':0,'output_count_channels_outside_model_skipped':0,'raw_transitions_projected':0,'raw_source_keys':0}
    storage.flush(True)
    for source_number,label in enumerate(storage):
        if storage.meta[label]['rank']==0:continue
        block=storage.load_readonly(label);j,counts=label;source_packets={}
        for key in dict.fromkeys(key[:-1] for key in block.keys):
            value=packet(block,key)
            if len(value[0]):source_packets[key]=value
        counters['raw_source_keys']+=len(source_packets);target_block=block;matrices={}
        @lru_cache(maxsize=512)
        def target_packet(key):
            return source_packets.get(key,empty) if target_block is block else packet(target_block,key)
        def project(target,source_columns,fields,entries):
            rows,bra=target_packet(target)
            if len(rows)==0:return
            value=bra.conj()@fields.T;index=np.ix_(rows,source_columns)
            for name,coefficient in entries:
                matrix=matrices.get(name)
                if matrix is None:matrix=np.zeros((target_block.rank,block.rank),complex);matrices[name]=matrix
                matrix[index]+=coefficient*value
            counters['raw_transitions_projected']+=1
        def finish_pair():
            for name,matrix in matrices.items():collectors[name].add_block(target_block.columns,block.columns,matrix,0. if name=='Gram' else 2e-14)
            matrices.clear();target_packet.cache_clear()
        for key,(columns,fields) in source_packets.items():
            project(key,columns,fields,[('Gram',1.)]);project(key,columns,fields@kinetic.T,[('rotation',1.)])
            for target,value in electric.action(key).items():project(target,columns,fields,[('electric',value)])
            for pw,value in primitive.field_maps['B'][key[-1]]:project(key[:-1]+(pw,),columns,fields,[('plaquette',-p['plaquette']*value),('loopB',value)])
            for pw,value in primitive.field_maps['C'][key[-1]]:project(key[:-1]+(pw,),columns,fields,[('loopC',value)])
        finish_pair()
        for edge,(start,end) in enumerate(model.EDGES):
            for flavour,iswhite,dagger in product(range(2),(False,True),(False,True)):
                changed=list(counts);category=2*flavour+int(iswhite);direction=-1 if dagger else 1;changed[4*start+category]-=direction;changed[4*end+category]+=direction;target_label=(j,tuple(changed))
                if target_label not in storage or storage.meta[target_label]['rank']==0:counters['output_count_channels_outside_model_skipped']+=len(source_packets);continue
                target_block=storage.load_readonly(target_label);target_packet.cache_clear()
                for key,(columns,fields) in source_packets.items():
                    transformed=[p['scalar_coupling']*weights[edge]*fields]
                    transformed.extend(1j*p['vector_velocity']*weights[edge]*(fields@rotations[edge,i].T) for i in range(3))
                    for component in range(4):
                        image=primitive.hop({key:1.+0j},edge,flavour,iswhite,component,dagger);counters['primitive_hop_calls']+=1
                        coefficient=-1 if component and dagger else 1;name=('white' if iswhite else 'quark')+('_vector' if component else '_scalar')
                        for target,value in image.items():project(target,columns,transformed[component],[(name,coefficient*value)])
                finish_pair()
        target_packet.cache_clear();del source_packets,target_block,block;gc.collect()
        print(json.dumps({'stage':'rich_source_primitive_body_tensor_H','source_count_block':source_number,'count_blocks':len(storage),'elapsed_seconds':time.time()-started,**counters}),flush=True)
    operators={};errors={};discarded={}
    for name in names:
        matrix=collectors[name].finish();discarded[name]=collectors[name].discarded_norm2;difference=matrix-matrix.conj().T;errors[name]=float(np.linalg.norm(difference.data));sparse.save_npz(out+'_raw_'+name+'.npz',matrix)
        operators[name]=(matrix+matrix.conj().T)/2;sparse.save_npz(out+'_op_'+name+'.npz',operators[name]);del matrix
    receipt={'dimension':dimension,'nnz':{name:value.nnz for name,value in operators.items()},'raw_Hermiticity_Frobenius':errors,'entry_roundoff_discarded_norm2':discarded,
        'metric':'ALL actual Gram entries retained, including tiny offdiagonal values; generalized metric kept through analysis. No assumedI or source EOM Hermitization.',
        'roundoff':'Raw operators retained; charged weakH analysis uses tiny Hermitian average. Non-Gram entry2e-14 sparse roundoff only, discarded Frobenius squared explicit. No physical source/field/mass/spin/coupling cut or current replacement.',
        'storage':'One pinned source body block and one target block at a time, target-pair traversal prevents disk-cache thrashing. All source packets for ONLY current count cached; no global body CSC+CSR/dense lookup residency.',
        'projection':'Full primitive action and local colour Q appliedTWICE before target selection. Source-count support skips are Galerkin weak-matrix selection, NOT full-H closure or particle leakage test. Actual allPhi7 independent factor, allR10/PW19 inside body profiles; severe operator/sea/body-R rank remains.',
        'elapsed_seconds':time.time()-started,**counters}
    Path(out+'_assembly_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return operators,receipt
