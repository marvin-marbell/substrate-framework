"""Continue ALL288 sources from actual same-family sparse checkpoint, no cutback."""
import json,pickle,resource,sys,time
from pathlib import Path
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_sparse_basis import PacketBasis


def checkpoint(path,data):
    temporary=Path(path+'.pending')
    with open(temporary,'wb') as f:pickle.dump(data,f,protocol=5)
    temporary.replace(path)


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    with open(config['checkpoint_snapshot'],'rb') as f:data=pickle.load(f)
    with open(config['neutral_snapshot'],'rb') as f:neutral=pickle.load(f)
    if data['p']!=neutral['p']:raise ValueError('Changed same-family law')
    basis=data['basis']
    if not isinstance(basis,PacketBasis):raise TypeError('Sparse same-family checkpoint required')
    specs=list(model.source_specs());first=data['next_source']
    for i in range(first,len(specs)):
        for packet in neutral['generators']:basis.append(model.source(packet,*specs[i]))
        print(json.dumps({'stage':'actual_same_family_remaining_sources','source':i,'source_count':len(specs),'dimension':len(basis.vectors)}),flush=True)
        if (i+1)%8==0:checkpoint(out+'_continuation_checkpoint.pkl',{'p':data['p'],'basis':basis,'next_source':i+1,'neutral_snapshot':config['neutral_snapshot']})
    data.update(basis=basis,sources=specs,next_source=len(specs),neutral_snapshot=config['neutral_snapshot'])
    with open(out+'_charged.pkl','xb') as f:pickle.dump(data,f,protocol=5)
    result={'kind':'ACTUAL_COMPLETE_SAME_K2_OCCUPIED_MOTT_ALL_SOURCE_FAMILY','parameters':data['p'],'continued_from':config['checkpoint_snapshot'],
        'first_source':first,'completed_source_count':len(specs),'charged':basis.receipt(),
        'meaning':'ALL original positions/spins/sources and SAME neutralK2 family/couplings/null thresholds; resumes completed computation, not a source/state/physicalcut fallback.',
        'resource':{'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
