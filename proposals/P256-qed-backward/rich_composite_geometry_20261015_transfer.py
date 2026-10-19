"""Actual compact-sector multi-hop amplitudes, never assigned baryon hopping."""
import json,resource,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from rich_composite_geometry_20261015_model import field_and_rotor
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load,physical_arrays,nonhop


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    data=load(config['basis_snapshot']);primitive=primitive_for(data);p=data['p'];scalar,rotor=field_and_rotor(p)
    coupling=load(config['coupling_snapshot']);ops=load(config['operators_snapshot']);h=sparse.load_npz(config['H_snapshot'])
    spectrum=np.load(config['states_snapshot']);e=spectrum['energies'];ground=np.flatnonzero(abs(e-e[0])<1e-7)
    pole=spectrum['states'][:,ground]/np.sqrt(len(ground));nf=len(scalar['grid']);arrays=physical_arrays(coupling)
    mask=np.repeat(arrays['Csite'][:,0],nf);prepared=mask[:,None]*pole;norm=float(np.linalg.norm(prepared)**2)
    result={'kind':'ACTUAL_ORIGINAL_CONSTITUENT_HOP_COMPACT_TRANSFER_MOMENTS','parameters':p,'preparation_strength':norm,
        'meaning':'P_Csite_s Hhop^k P_Csite0 on an actual conditioned ground-pole projector, retaining ALL selected intermediate Phi/sea/flux/R states. Norm² has ENERGY^(2k), NOT transfer probability or fitted autonomous baryon-hop coefficient. Full-H time motion is separately measured.','moments':[]}
    if norm>1e-20:
        prepared/=np.sqrt(norm);hhop=h-nonhop(ops,coupling,primitive,scalar,rotor);v=prepared
        for k in range(1,7):
            v=hhop@v
            result['moments'].append({'primitive_hop_order':k,
                'compact_target_norm2':[float(np.linalg.norm(v*np.repeat(arrays['Csite'][:,s],nf)[:,None])**2) for s in range(4)],
                'total_norm2':float(np.linalg.norm(v)**2)})
    else:result['actual_failure']='No compact-site0 weight; no source/capture/state substitution.'
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
