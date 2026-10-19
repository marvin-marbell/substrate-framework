"""Material stretched-geometry continuation with a FIXED physical isometry.
No spectrum/reference is copied: the new H must be built and solved afterwards.
"""
import hashlib,json,pickle,resource,sys,time
from pathlib import Path
import numpy as np
from rich_composite_geometry_20261015_model import geometry


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time();source=Path(config['basis_snapshot'])
    with open(source,'rb') as f:data=pickle.load(f)
    old=dict(data['p']);new=dict(old);new['geometry']='stretched'
    digest=hashlib.sha256()
    with open(source,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):digest.update(block)
    data['family_origin']={'file':source.name,'sha256':digest.hexdigest(),'family_parameters':old,
        'meaning':'SAME fixed occupied/source packet isometry across geometries, not reranked/retuned and NOT new-H eigenstates. Geometry/H couplings are separately NEW parameters. No variational/convergence comparison.'}
    data['p']=new
    if 'primitive' in data:
        primitive=type(data['primitive'])(new,data['primitive'].local);data['primitive']=primitive;data['basis'].primitive=primitive
    path=out+'_basis.pkl'
    with open(path,'xb') as f:pickle.dump(data,f,protocol=5)
    graph=geometry(new)
    result={'kind':'ACTUAL_FIXED_NATIVE_ISOMETRY_STRETCHED_GEOMETRY_PREPARATION_NOT_NEW_SPECTRUM',
        'parameters':new,'family_origin':data['family_origin'],'basis_dimension':len(data['basis'].vectors),
        'new_physical_geometry':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in graph.items()},
        'not_inherited':'NO H/eigenstate/energy/reference transferred; build NEW raw primitive operators and solve own occupied/charged H. Scalar volume stays4fixed material cells, massdensity3 fixed; metric lengths/tangents/flux energies/hops and FULL inertia tensor change.',
        'resource':{'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
