"""NEW actual raw-seed colour-Gauss witness before charge projection; no trimming."""
from itertools import product
from pathlib import Path
from typing import Any
import hashlib,json,pickle,resource,sys,time
import numpy as np
from rich_spatial_colour_charge_20261015_model import generators,loop_field,CHARGES3
ROOT=Path(__file__).resolve().parent


def sha(path: Path)->str:
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def image(packet: Any,t: Any,field: Any)->Any:
    out={mask:field@value for mask,value in packet.items()}
    terms=[(16*f+8*m+4*s+a,16*f+8*m+4*s+b,t[a,b]) for f,m,s,a,b in product(range(2),range(2),range(2),range(3),range(3)) if t[a,b]!=0]
    for mask,value in packet.items():
        for a,b,c in terms:
            if not mask&(1<<b):continue
            intermediate=mask^(1<<b)
            if intermediate&(1<<a):continue
            target=intermediate|(1<<a);sign=(-1)**((mask&((1<<b)-1)).bit_count()+(intermediate&((1<<a)-1)).bit_count())
            if target not in out:out[target]=np.zeros(19,complex)
            out[target]+=sign*c*value
    return out


def main()->None:
    start=time.monotonic();request:Any=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);loop=loop_field();ts=generators();result={}
    for name,record in request['native_basis'].items():
        path=ROOT/record['file']
        if sha(path)!=record['sha256']:raise ValueError('Trusted native basis differs from captured constructor digest')
        with path.open('rb') as stream:basis=pickle.load(stream)
        raw=[];bounds=[];map_norms=[float(np.linalg.svd(w,compute_uv=False)[0]) for w in basis.blocks.values()]
        for a,t in enumerate(ts):
            norms=[]
            for packet in basis.seeds:
                acted=image(packet,t,loop['L'][a]+loop['R'][a]);norms.append(float(np.sqrt(sum(np.vdot(v,v).real for v in acted.values()))))
            raw.append(dict(generator=a,maximum_seed_image_norm=max(norms),all_seed_image_Frobenius=float(np.linalg.norm(norms))))
            bounds.append(float(np.linalg.norm(norms)*np.linalg.norm(map_norms)))
        result[name]=dict(native_dimension=basis.dimension,actual_raw_seed_count=len(basis.seeds),raw_Gauss_generators=raw,conservative_native_Gauss_Frobenius_upper_estimates=bounds,charge_projector_commutes_colour=True,
            meaning='Actual canonical full CAR32 total colour charge plus PW19(L+R) applied BEFORE band/charge projection, retaining all tiny coefficients. P_h commutes colour because triplet charges coincide. ||P_h G V W_h||F<=||G V||F||W_h||op, summed over orthogonal h; computed floating estimate, not interval proof or local-field continuum theorem.')
    spread=max(float(np.ptp(CHARGES3[4*i:4*i+3])) for i in range(4));document=dict(kind='NEW_INDEPENDENT_RAW_NATIVE_SEED_COLOUR_GAUSS_AND_CHARGE_COMMUTATION',request=request,physical_triplet_charge3_spread=spread,sectors=result,local_tree_scope='Tree Gauss G1 is solved by E_tree=Q_site1+R_cycle; G0 is the actually tested global colour Gauss. SU3 E/T are not physical EM charge. No new coupling/constraint or old producer/checker CLI.',elapsed_seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(document,indent=2)+'\n');print(json.dumps(dict(result=output.name,raw_max={name:max(v['maximum_seed_image_norm'] for v in entry['raw_Gauss_generators']) for name,entry in result.items()},elapsed=document['elapsed_seconds'],peak_RSS_KiB=document['peak_RSS_KiB'])),flush=True)


if __name__=='__main__':main()
