"""Actual constructor and early occupied-reference snapshot before response science."""
import json,pickle,resource,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from rich_spatial_colour_charge_20261015_model import build,law,TensorOperator,clean
from rich_spatial_colour_charge_20261015_snapshot import save
from rich_spatial_colour_root_20261015_calculate import lowest,save_tensor


def stationary(h,op,rotor,scalar,M,count,p):
    # Exact TOTAL M used for stationary eigensolve ONLY; public/body/full H all-M.
    mr,u=np.linalg.eigh(rotor['L'][2].toarray());mr=np.round(mr,8)
    n=op['Q'].shape[0];transform=sparse.kron(sparse.eye(n),sparse.csr_matrix(u),format='csr')
    labels=(op['S_z_labels'][:,None]+mr[None,:]).ravel();ids=np.flatnonzero(abs(labels-M)<1e-7)
    transform=transform[:,ids]
    one=clean(transform.conj().T@h.one@transform);phi=clean(transform.conj().T@h.phi@transform)
    restricted=TensorOperator(one,phi,scalar,True)
    energies,vectors=lowest(restricted,count,p)
    residuals=[float(np.linalg.norm(restricted@vectors[:,j]-energies[j]*vectors[:,j])) for j in range(len(energies))]
    orth=float(np.linalg.norm(vectors.conj().T@vectors-np.eye(len(energies))))
    full=[]
    for v in vectors.T:
        full.append((transform@v.reshape(len(ids),len(scalar['grid']))).ravel())
    return energies,np.column_stack(full),dict(total_M=M,stationary_dimension=restricted.shape[0],full_dimension=h.shape[0],residuals=residuals,orthogonality=orth)


def main():
    ip=Path(sys.argv[1]);prefix=Path(sys.argv[2]);p=json.loads(ip.read_text());start=time.time();stage_records={}
    def checkpoint(stage,value):
        path=Path(str(prefix)+'_'+stage+'.pkl')
        with path.open('xb') as handle:pickle.dump(value,handle,protocol=5)
        item=dict(stage=stage,elapsed_seconds=time.time()-start,bytes=path.stat().st_size)
        if hasattr(value,'dimension'):item.update(dimension=value.dimension,seeds=len(value.seeds),isometry_error=value.isometry_error)
        stage_records[stage]=item;Path(str(prefix)+'_stages.json').write_text(json.dumps(stage_records,indent=2)+'\n');print(json.dumps(item),flush=True)
    data=build(p,checkpoint)
    save(data,p,str(prefix)+'_body.pkl')
    arrays={};h={}
    for name,op in data['operators'].items():
        h[name]=law(op,data['rotor'],data['scalar'],p,data['geometry']);save_tensor(arrays,name+'_H0',h[name])
    arrays['scalar_grid']=data['scalar']['grid'];arrays['scalar_H']=data['scalar']['H'];np.savez_compressed(str(prefix)+'_H0.npz',**arrays)
    preliminary=dict(kind='ACTUAL_CHARGE_ORBIT_NATIVE_BODY_AND_FULL_H_SNAPSHOT_BEFORE_EIGEN_RESPONSE',body_snapshot=str(prefix)+'_body.pkl',parameters=p,sectors={name:dict(dimension=basis.dimension,seed_count=len(basis.seeds),isometry_error=basis.isometry_error,charge_values=[int(x) for x in sorted(set(basis.h))],twice_Sz_values=[int(x) for x in sorted(set(basis.m))],seed_rank_records=basis.seed_rank_records,charge_rank_records=basis.rank_records) for name,basis in data['native'].items()},scope='Exact physical P_h orbit, no compressed-charge exponential or old energies/states. Complete all M; chosen finite packet family and numerical-null rank cuts disclosed. No science admission.')
    Path(str(prefix)+'_preflight.json').write_text(json.dumps(preliminary,indent=2)+'\n')
    print(json.dumps(dict(stage='BODY_AND_H_READY',body=preliminary['body_snapshot'],elapsed=time.time()-start)),flush=True)
    e,v,info=stationary(h['neutral'],data['operators']['neutral'],data['rotor'],data['scalar'],0.,p['neutral_low_eigenstates'],p)
    np.savez_compressed(str(prefix)+'_neutral_state.npz',energies=e,vectors=v)
    result=dict(kind='NEW_ACTUAL_OCCUPIED_CHARGE_ORBIT_REFERENCE_EARLY_HANDOFF',neutral_energy=float(e[0]),eigen=info,neutral_J2=float(np.vdot(v[:,0],sparse.kron(data['operators']['neutral']['J2'],sparse.eye(len(data['scalar']['grid'])))@v[:,0]).real),resource=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-start),body=preliminary['body_snapshot'],states=str(prefix)+'_neutral_state.npz',scope='Own new common quantum reference, not inherited frozen band or EM field reference; whole H/native snapshots precede eigen/response. Stationary M0 block only, full basis/multiplets remain in H and all responses.')
    Path(str(prefix)+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
