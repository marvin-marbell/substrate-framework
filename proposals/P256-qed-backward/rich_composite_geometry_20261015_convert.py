"""Convert an actual SAME-family completed source checkpoint, no physics restart."""
import json,pickle,resource,sys,time
from pathlib import Path
from rich_composite_geometry_20261015_sparse_basis import convert


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    with open(config['checkpoint_snapshot'],'rb') as f:data=pickle.load(f)
    before={'dimension':len(data['basis'].vectors),'dense_padded_numeric_bytes':sum(b.q.nbytes for b in data['basis'].storage.values()),'next_source':data['next_source']}
    basis,dropped=convert(data['basis']);data['basis']=basis
    data['representation_conversion']={'from':config['checkpoint_snapshot'],'next_source':data['next_source'],
        'roundoff_tiny_coefficient_norm2_sum':dropped,'meaning':'Same actual native source prefix and ALL remaining source scope. Only2e-14 already-used materialized-vector roundoff coefficients removed; no energy/spin/state/coupling/source or virtual cut.'}
    with open(out+'_checkpoint.pkl','xb') as f:pickle.dump(data,f,protocol=5)
    result={'kind':'ACTUAL_SAME_NATIVE_SOURCE_CHECKPOINT_SPARSE_REPRESENTATION_CONVERSION','before':before,'after':basis.receipt(),
        'conversion':data['representation_conversion'],'resource':{'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
