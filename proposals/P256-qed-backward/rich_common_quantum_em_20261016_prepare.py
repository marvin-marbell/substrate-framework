"""NEW-H ingredients from immutable CAR/PW/R isometries, never solved states.
Extra physical chord is acted before target selection. Canonical source raw Gram
includes ALL CAR outputs, so its source-space compression is measured explicitly.
"""
from collections import defaultdict
from functools import lru_cache
from itertools import product
from pathlib import Path
import gc,json,pickle,sys,time,zipfile,io,resource
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_sparse_basis import Block
from rich_composite_geometry_20261015_cloud_probe import embedding
from rich_composite_geometry_20261015_dressing_operators import Collector
from rich_composite_geometry_20261015_factors import load_h
from rich_common_native_species_20261016_sources import operator_terms,algebra_receipt
from rich_common_quantum_em_20261016_geometry import construct

OLD_NEUTRAL='rich_composite_geometry_20261015_correlated_onehop_neutral1_neutral.pkl'
OLD_COUPLING='rich_composite_geometry_20261015_correlated_onehop_reference1_J0_coupling.pkl'
OLD_FACTORS='rich_composite_geometry_20261015_correlated_onehop_reference1_J0_H0.npz'
OLD_JOINT='rich_common_native_species_20261016_joint1'


def load(path):
    with open(path,'rb')as stream:return pickle.load(stream)


def save(path,value):
    with open(path,'wb')as stream:pickle.dump(value,stream,protocol=5)


def block_of(storage,label):
    return storage.load_readonly(label)if hasattr(storage,'load_readonly')else storage[label]


def packets(block):
    result={};matrix=block.matrix.tocsr()
    for key in dict.fromkeys(key[:-1]for key in block.keys):
        pieces=[];columns=set()
        for r in range(10):
            row=block.lookup.get(key+(r,))
            if row is None:continue
            a,b=matrix.indptr[row:row+2];ix=matrix.indices[a:b];values=matrix.data[a:b];pieces.append((r,ix,values));columns.update(map(int,ix))
        if not columns:continue
        columns=np.asarray(sorted(columns),dtype=np.int32);where={int(c):i for i,c in enumerate(columns)};fields=np.zeros((len(columns),10),complex)
        for r,indices,values in pieces:
            for column,value in zip(indices,values):fields[where[int(column)],r]=value
        result[key]=(columns,fields)
    return result


def neutral_blocks(data,coupling):
    keys,matrix=embedding(data['basis']);labels=coupling['labels'];groups=defaultdict(list)
    for i,row in enumerate(labels):groups[(0,tuple(row['counts']))].append(i)
    result={};body=[(matrix@u).tocsr()for u in coupling['U']]
    for label,columns in groups.items():
        stacked=sparse.vstack([value[:,columns]for value in body],format='csr');keep=np.flatnonzero(np.diff(stacked.indptr));block=Block();block.keys=[keys[int(i)%len(keys)]+(int(i)//len(keys),)for i in keep];block.lookup={key:i for i,key in enumerate(block.keys)};block.matrix=stacked[keep].tocsr();block.rank=len(columns);block.columns=columns;result[label]=block
    return result


def append_chord(primitive):
    """Tree-gauged SU3 path0->1->2, direct physical charge chord0->2."""
    for flavour,white,component,dagger in product(range(2),(False,True),range(4),(False,True)):
        spin=np.eye(2)if component==0 else model.PAULI[component-1];terms=[]
        for member in((3,)if white else range(3)):
            for s,t in product(range(2),repeat=2):
                coefficient=complex(spin[s,t])
                if not coefficient:continue
                operations=[(model.mode(2,flavour,s,member),True),(model.mode(0,flavour,t,member),False)]
                if dagger:operations=[(index,not create)for index,create in reversed(operations)];coefficient=coefficient.conjugate()
                terms.append((operations,coefficient,primitive.field_map(np.eye(19))))
        primitive.maps[(4,flavour,white,component,dagger)]=terms


def chord_operators(primitive,storage,labels,rotor,out):
    append_chord(primitive);dimension=len(labels);p=primitive.p;positions=np.asarray(construct()['positions']);axis=positions[2]-positions[0];rotation=[sum((axis[a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray()for a in range(3)),np.zeros((10,10),complex))for i in range(3)];collectors={name:Collector(dimension)for name in('vector','scalar')};start=time.time();skipped=0
    for number,label in enumerate(storage):
        source=block_of(storage,label)
        if not source.rank:continue
        source_packets=packets(source);j,counts=label
        for flavour,white,dagger in product(range(2),(False,True),(False,True)):
            changed=list(counts);category=2*flavour+int(white);direction=-1 if dagger else 1;changed[category]-=direction;changed[8+category]+=direction;target_label=(j,tuple(changed))
            if target_label not in storage:skipped+=len(source_packets);continue
            target=block_of(storage,target_label);target_packets=packets(target);values={name:np.zeros((target.rank,source.rank),complex)for name in collectors}
            for key,(columns,fields)in source_packets.items():
                transformed=[p['scalar_coupling']*fields]+[1j*p['vector_velocity']*(fields@r.T)for r in rotation]
                for component in range(4):
                    image=primitive.hop({key:1.+0j},4,flavour,white,component,dagger);name='vector'if component else'scalar';sign=-1 if component and dagger else 1
                    for key_out,coefficient in image.items():
                        packet=target_packets.get(key_out)
                        if packet is None:continue
                        rows,bra=packet;values[name][np.ix_(rows,columns)]+=sign*coefficient*(bra.conj()@transformed[component].T)
            for name,value in values.items():collectors[name].add_block(target.columns,source.columns,value,0.)
            del target,target_packets,values
        if number%8==0:print(json.dumps({'stage':'NEW_physical_chord_before_target_selection','source_block':number,'blocks':len(storage),'seconds':time.time()-start}),flush=True)
        del source_packets,source;gc.collect()
    result={name:collector.finish()for name,collector in collectors.items()};receipt={'full_before_target_hop':True,'new_colour_route':'tree0->1->2','new_charge_path':'direct0->2','source_count_channels_not_in_declared_space':skipped,'hermiticity_errors':{name:float(np.linalg.norm((value-value.conj().T).data))for name,value in result.items()},'numerical_element_pruning':0.}
    for name,value in result.items():sparse.save_npz(out+'_chord_'+name+'.npz',value)
    return result,receipt


def body_metric(storage,dimension):
    collector=Collector(dimension)
    for label in storage:
        block=block_of(storage,label);value=(block.matrix.conj().T@block.matrix).toarray();collector.add_block(block.columns,block.columns,value,0.)
    return collector.finish()


def metric_powers(gram,labels):
    groups=defaultdict(list)
    for i,row in enumerate(labels):groups[tuple(row['counts'])].append(i)
    half=Collector(len(labels));inverse=Collector(len(labels));smallest=float('inf');largest=0.
    for indices in groups.values():
        block=gram[indices][:,indices].toarray();values,vectors=np.linalg.eigh(block);smallest=min(smallest,float(values.min()));largest=max(largest,float(values.max()))
        if values.min()<=0:raise ValueError('Actual body metric is not positive')
        half.add_block(indices,indices,(vectors*np.sqrt(values))@vectors.conj().T,0.);inverse.add_block(indices,indices,(vectors/np.sqrt(values))@vectors.conj().T,0.)
    return half.finish(),inverse.finish(),{'minimum_eigenvalue':smallest,'maximum_eigenvalue':largest}


def prepare(mode,out):
    geometry=construct();Path(out+'_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    if mode=='neutral':
        data=load(OLD_NEUTRAL);coupling=load(OLD_COUPLING);primitive=data['primitive'];storage=neutral_blocks(data,coupling);labels=coupling['labels'];old=load_h(OLD_FACTORS);counts=np.asarray([row['counts']for row in labels]);coo=old.c0.tocoo();same=np.all(counts[coo.row]==counts[coo.col],axis=1);static=sparse.coo_matrix((coo.data[same],(coo.row[same],coo.col[same])),shape=coo.shape).tocsr();vector=old.c0-static;scalar_hop=old.cphi
        save(out+'_neutral_raw_blocks.pkl',{'blocks':storage,'labels':labels})
    elif mode=='joint':
        data=load(OLD_JOINT+'_body_basis.pkl');primitive=data['primitive'];storage=data['storage'];storage.relocate(Path.cwd());labels=data['labels'];static=sum((sparse.load_npz(OLD_JOINT+'_op_'+name+'.npz')for name in('rotation','electric','plaquette')),sparse.csr_matrix((len(labels),len(labels)),dtype=complex));vector=sum((sparse.load_npz(OLD_JOINT+'_op_'+name+'.npz')for name in('quark_vector','white_vector')),sparse.csr_matrix(static.shape,dtype=complex));scalar_hop=sum((sparse.load_npz(OLD_JOINT+'_op_'+name+'.npz')for name in('quark_scalar','white_scalar')),sparse.csr_matrix(static.shape,dtype=complex))
    else:raise ValueError('neutral or joint required')
    scalar,rotor=model.field_and_rotor(primitive.p);gram=body_metric(storage,len(labels));half,inverse,metric=metric_powers(gram,labels);chord,receipt=chord_operators(primitive,storage,labels,rotor,out)
    for name,value in{'static':static,'vector':vector,'scalar_hop':scalar_hop,'Gram':gram,'half':half,'inversehalf':inverse}.items():sparse.save_npz(out+'_'+name+'.npz',value)
    np.savez_compressed(out+'_scalar.npz',H=scalar['H'],grid=scalar['grid'])
    clean_labels=[{'counts':list(row['counts']),'twice_J':0 if mode=='neutral'else 2}for row in labels]
    Path(out+'_body.json').write_text(json.dumps({'mode':mode,'labels':clean_labels,'parameters':primitive.p,'metric':metric,'chord':receipt,'old_stationary_state_files_opened':False,'old_energy_values_used':False,'basis_limitation':'Selected source-body CAR/PW/R isometry, explicit NEW fundamental finite-matrix law. NO unprojected full-CAR spectral claim.'},indent=2)+'\n')
    print(json.dumps({'stage':'NEW_H_ingredients_complete','mode':mode,'body_dimension':len(labels),'metric':metric,'chord':receipt,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)


def source(neutral_prefix,joint_prefix,out):
    neutral=load(neutral_prefix+'_neutral_raw_blocks.pkl');data=load(OLD_JOINT+'_body_basis.pkl');primitive=data['primitive'];storage=data['storage'];storage.relocate(Path.cwd());labels=data['labels'];nlabels=neutral['labels'];terms=operator_terms({'sector':36,'B_site':0,'L_site':2});rows=[];columns=[];values=[];gram=Collector(len(nlabels));counter={'raw_output_keys':0,'source_blocks_not_in_target':0};archive=zipfile.ZipFile(out+'_full_CAR_source_blocks.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3)
    original=load(OLD_NEUTRAL)['primitive'].local.vectors
    if len(original)>len(primitive.local.vectors)or any(a!=b for a,b in zip(original,primitive.local.vectors)):raise ValueError('Raw CAR coordinate prefixes differ')
    del original
    for number,(label,block)in enumerate(neutral['blocks'].items()):
        count=list(label[1]);count[0]+=2;count[2]+=1;count[11]+=1;target_label=(2,tuple(count));image={}
        for key,(ix,fields)in packets(block).items():
            for operations,coefficient in terms:
                for target,value in primitive.coordinate_operations(key,operations).items():
                    output=image.get(target)
                    if output is None:output=np.zeros((block.rank,10),complex);image[target]=output
                    output[ix]+=coefficient*value*fields
        g=np.zeros((block.rank,block.rank),complex)
        for field in image.values():g+=field.conj()@field.T
        gram.add_block(block.columns,block.columns,g,0.);counter['raw_output_keys']+=len(image)
        buffer=io.BytesIO();np.savez_compressed(buffer,columns=np.asarray(block.columns),keys=np.asarray(list(image),dtype=np.int64),coefficients=np.asarray(list(image.values())));archive.writestr('source_block_'+str(number)+'.npz',buffer.getvalue())
        if target_label not in storage:counter['source_blocks_not_in_target']+=1;continue
        target=block_of(storage,target_label);target_packets=packets(target);value=np.zeros((target.rank,block.rank),complex)
        for key,field in image.items():
            packet=target_packets.get(key)
            if packet is None:continue
            ix,bra=packet;value[ix]+=bra.conj()@field.T
        r,c=np.nonzero(value);rows.extend(np.asarray(target.columns)[r]);columns.extend(np.asarray(block.columns)[c]);values.extend(value[r,c]);del image,target_packets,target;gc.collect()
        print(json.dumps({'stage':'NEW_reference_independent_canonical_source','block':number,'blocks':len(neutral['blocks']),**counter}),flush=True)
    archive.close();weak=sparse.coo_matrix((values,(rows,columns)),shape=(len(labels),len(nlabels))).tocsr();ni=sparse.load_npz(neutral_prefix+'_inversehalf.npz');ji=sparse.load_npz(joint_prefix+'_inversehalf.npz');operator=ji@weak@ni;fullgram=ni@gram.finish()@ni
    sparse.save_npz(out+'_source_weak.npz',weak);sparse.save_npz(out+'_source_orth.npz',operator);sparse.save_npz(out+'_source_full_CAR_gram_orth.npz',fullgram)
    save(out+'_source_primitive.pkl',primitive)
    receipt={'source':'canonical Bdag0 Ldag2; L FIRST; actual CAR source on arbitrary new reference, NOT old source-state coefficients','source_algebra':algebra_receipt({'sector':36,'B_site':0,'L_site':2,'twice_J':2}),**counter,'source_shape':list(operator.shape),'full_CAR_Gram_saved':True,'flux_Wilson_dressing_applied_later_on_actual_new_reference':True,'old_solved_states_used':False};Path(out+'_source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


def main():
    if sys.argv[1]=='source':source(*sys.argv[2:5])
    else:prepare(sys.argv[1],sys.argv[2])


if __name__=='__main__':main()
