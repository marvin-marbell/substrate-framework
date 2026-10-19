"""Full CAR intermediate words with compact exact field lifts, not source PAP.
Output-count partitions are memory slices ONLY; every reachable word output kept.
"""
from collections import defaultdict
from functools import lru_cache
from itertools import product
import gc,json,pickle,zipfile
import numpy as np
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_cloud_probe import add,inner
from rich_composite_geometry_20261015_source_eom import bare_state


def sources(primitive,omega):
    baryon=model.source_terms([0,0,0],'half_triplet',1);lepton=[([(model.mode(2,1,0,3),True)],1.+0j)];white=bare_state(primitive,omega,lepton)
    return {'0':omega,'B':bare_state(primitive,omega,baryon),'L':white,'BL':bare_state(primitive,white,baryon)}


def steps(species):return [(flavour,dagger)for flavour in ((1,)if species=='white'else(0,1))for dagger in (False,True)]


def changed(counts,species,edge,flavour,dagger):
    values=list(counts);start,end=model.EDGES[edge];category=2*flavour+int(species=='white');direction=-1 if dagger else 1;values[4*start+category]-=direction;values[4*end+category]+=direction
    maximum=2 if species=='white'else 6
    if values[4*start+category]<0 or values[4*end+category]<0 or values[4*start+category]>maximum or values[4*end+category]>maximum:return None
    return tuple(values)


def coefficient(primitive,species,edge,component,flavour,dagger,kind,theta):
    q=(-1. if flavour==1 else 0.)if species=='white'else(2/3 if flavour==0 else-1/3);direction=-1 if dagger else 1;base=primitive.graph['weights'][edge]*(primitive.p['scalar_coupling']if component==0 else 1j*primitive.p['vector_velocity']*direction);phase=np.exp(1j*direction*q*theta)
    if kind=='delta':return base*(phase-1)
    if kind=='current':return -1j*direction*q*phase*base
    if kind=='contact':return -q*q*phase*base
    raise ValueError('Owning microscopic delta/current/contact required')


def inner_lift(left,right,rgram,phigram,only_new=None):
    if not left or not right:return 0.+0j
    result=0.+0j
    for key,value in left.items():
        if key not in right or(only_new is not None and key in only_new):continue
        result+=np.vdot(value,rgram@right[key]@phigram)
    return result


def evaluate(primitive,state,field,direction,theta,out):
    source_species=direction;receiver_species='quark'if direction=='white'else'white';source_edge=2 if source_species=='white'else 0;receiver_edge=0 if receiver_species=='quark'else 2;names=[item[0]for item in field['lifts']];indices={name:i for i,name in enumerate(names)};parts=defaultdict(list)
    for key,value in state.items():parts[primitive.label(key)[:-1]].append((key,value))
    plan=defaultdict(list)
    for counts in parts:
        plan[counts].append(('identity',counts,None,None))
        for receiver_step in steps(receiver_species):
            target=changed(counts,receiver_species,receiver_edge,*receiver_step)
            if target is not None:plan[target].append(('receiver',counts,None,receiver_step))
        for source_step in steps(source_species):
            target=changed(counts,source_species,source_edge,*source_step)
            if target is not None:plan[target].append(('source',counts,source_step,None))
            for receiver_step in steps(receiver_species):
                first=changed(counts,receiver_species,receiver_edge,*receiver_step);target=None if first is None else changed(first,source_species,source_edge,*source_step)
                if target is not None:plan[target].append(('vertex',counts,source_step,receiver_step))
    calls=0
    @lru_cache(maxsize=8192)
    def hop(key,species,edge,component,flavour,dagger):
        nonlocal calls
        calls+=1;return primitive.hop({key:1.+0j},edge,flavour,species=='white',component,dagger)
    gram=np.zeros((len(names),len(names)),complex);r_complement=np.zeros_like(gram);new_support=np.zeros_like(gram);counts_receipts=[];max_rows=0;source_modes=set();receiver_modes=set()
    for species,edge,target in ((source_species,source_edge,source_modes),(receiver_species,receiver_edge,receiver_modes)):
        for flavour,dagger in steps(species):
            for component in range(4):
                for operations,spin,maps in primitive.maps[edge,flavour,species=='white',component,dagger]:
                    target.update(index for index,_ in operations)
                    if any(any(pw!=index or abs(value-1)>1e-14 for pw,value in column)for index,column in enumerate(maps)):raise ValueError('Chosen vertex is owning tree0/white2; nonidentity PW word requires its actual complement algebra')
    with zipfile.ZipFile(out+'_outputs.zip','x',compression=zipfile.ZIP_DEFLATED,allowZip64=True)as archive:
        for count_index,(counts,entries)in enumerate(sorted(plan.items())):
            outputs=[{}for _ in names]
            for kind,input_counts,source_step,receiver_step in entries:
                for key,value in parts[input_counts]:
                    if kind=='identity':add(outputs[0],key,1.,value);continue
                    if kind=='receiver':
                        for j in range(4):
                            factor=coefficient(primitive,receiver_species,receiver_edge,j,*receiver_step,'current',0.)
                            for target,car in hop(key,receiver_species,receiver_edge,j,*receiver_step).items():add(outputs[indices['receiver_'+str(j)]],target,factor*car,value)
                    elif kind=='source':
                        for i in range(4):
                            image=hop(key,source_species,source_edge,i,*source_step)
                            for name,source_kind in (('source_delta','delta'),('source_current','current'),('source_contact','contact')):
                                factor=coefficient(primitive,source_species,source_edge,i,*source_step,source_kind,theta)
                                for target,car in image.items():add(outputs[indices[name+'_'+str(i)]],target,factor*car,value)
                    else:
                        for i,j in [(0,0)]+list(product(range(1,4),repeat=2)):
                            factor=coefficient(primitive,source_species,source_edge,i,*source_step,'delta',theta)*coefficient(primitive,receiver_species,receiver_edge,j,*receiver_step,'current',0.)
                            for intermediate,first in hop(key,receiver_species,receiver_edge,j,*receiver_step).items():
                                for target,second in hop(intermediate,source_species,source_edge,i,*source_step).items():add(outputs[indices[f'vertex_{i}{j}']],target,factor*first*second,value)
            active=[i for i,packet in enumerate(outputs)if packet];rows=sum(len(outputs[i])for i in active);max_rows=max(max_rows,rows);local=np.zeros_like(gram)
            for a,left in enumerate(active):
                for right in active[a:]:
                    entry=inner_lift(outputs[left],outputs[right],field['Rgram'][left,right],field['Phigram'][left,right]);comp=inner_lift(outputs[left],outputs[right],field['Rcomplement'][left,right],field['Phigram'][left,right]);outside=inner_lift(outputs[left],outputs[right],field['Rgram'][left,right],field['Phigram'][left,right],state)
                    local[left,right]=entry;gram[left,right]+=entry;r_complement[left,right]+=comp;new_support[left,right]+=outside
                    if left!=right:local[right,left]=entry.conjugate();gram[right,left]+=entry.conjugate();r_complement[right,left]+=comp.conjugate();new_support[right,left]+=outside.conjugate()
            if active:
                with archive.open(f'counts_{count_index:04d}.pkl','w',force_zip64=True)as stream:pickle.dump({'counts':counts,'channels':{names[i]:outputs[i]for i in active},'field_lifts':'own direction_field_lifts.npz; raw10x7 coefficients times exact84x10 Rlift and7x7Philift','scope':'ALLoutputs for this memory countpartition,not source/bare-return projection'},stream,protocol=5)
            counts_receipts.append({'counts':counts,'channel_rows':rows,'vertex_norm2':float(np.sum(local[17:,17:]).real)});del outputs;gc.collect()
            print(json.dumps({'stage':'PRIMITIVE_VERTEX_OUTPUT_PARTITION','direction':direction,'count':count_index,'formal_counts':len(plan),'channel_rows':rows,'hop_calls':calls}),flush=True)
    return {'Gram':gram,'R_complement_Gram':r_complement,'new_raw_support_Gram':new_support,'channels':names,'raw_trace':float(inner(state,state).real),'output_count_partitions':counts_receipts,'maximum_partition_channel_rows':max_rows,'actual_primitive_hop_calls':calls,'CAR_cross_mode_intersection':sorted(source_modes&receiver_modes),'CAR_first_commutator':'All source/receiver are even disjoint canonical bilinears; actual owning map modeintersection empty implies exactzero CARcontractions. Full fieldfirstcomm computed separately before R/sourceP.','outputs':'Exact factorized rich CAR/PW/R<=3/Phi7 channels retained; neither elasticreturn nor native sourceP used. RawR10coefficients in outputzip are lifted by stored84x10 matrices,not projectedR10states.'}
