"""Full N1xPW19 primitive charge/Gauss identities, before ANY orbit projection.
CAR bilinear charge commutators lift these one-particle identities to every N.
This distinguishes original owning current charges from a block-labelled Ward patch.
"""
import json,resource,sys,time
from pathlib import Path
from itertools import product
import numpy as np
from scipy import sparse
from rich_spatial_colour_charge_20261015_model import (
    loop_field,packet_matrix,forward,charge_site0_image,frame,generators,mode)
from rich_spatial_colour_root_20261015_model import one_body


def frobenius(matrix):return float(np.sqrt(np.vdot(matrix.data,matrix.data).real))


def main():
    request=json.loads(Path(sys.argv[1]).read_text());prefix=Path(sys.argv[2]);started=time.time();loop=loop_field();packets=[]
    for i,a in product(range(32),range(19)):
        vector=np.zeros(19,complex);vector[a]=1;packets.append({1<<i:vector})
    masks,_=packet_matrix(packets)
    # Embed nonzero charge-image rows, including no false charged-neutrino slots,
    # into the COMPLETE physical one-particle row basis.
    rm,rr=packet_matrix([charge_site0_image(x) for x in packets]);lookup={m:i for i,m in enumerate(masks)}
    inclusion=sparse.csr_matrix((np.ones(len(rm)),([lookup[m] for m in rm],range(len(rm)))),shape=(len(masks),len(rm)))
    rho=sparse.kron(inclusion,sparse.eye(19),format='csr')@rr
    gauss=[]
    for a,t in enumerate(generators()):
        terms=[(mode(f,m,s,c),mode(f,m,s,d),t[c,d]) for f,m,s,c,d in product(range(2),range(2),range(2),range(3),range(3)) if t[c,d]!=0]
        gauss.append(sparse.kron(one_body(masks,terms),sparse.eye(19),format='csr')+sparse.kron(sparse.eye(32),loop['L'][a]+loop['R'][a],format='csr'))
    records=[]
    for edge,f,white in product(range(2),range(2),range(2)):
        q=([2/3,0.] if f==0 else [-1/3,-1.])[white]
        for axis,spin in enumerate([np.eye(2)]+frame()['sigma']):
            images=[forward(x,loop,edge,f,white,spin) for x in packets];tm,op=packet_matrix(images)
            embedding=sparse.csr_matrix((np.ones(len(tm)),([lookup[m] for m in tm],range(len(tm)))),shape=(32,len(tm)))
            op=sparse.kron(embedding,sparse.eye(19),format='csr')@op
            records.append(dict(edge=edge,flavour=f,colourless=bool(white),spin_component=axis,physical_q=q,raw_site_charge_commutator=frobenius(rho@op-op@rho+q*op),raw_global_colour_commutators=[frobenius(g@op-op@g) for g in gauss],full_operator_norm=frobenius(op)))
    result=dict(kind='ORIGINAL_FULL_N1_PW19_PRIMITIVE_CHARGE_GAUSS_WITNESS_NOT_NATIVE_BLOCK_PATCH',inputs=request,dimension=608,records=records,scope='Actual model.forward and actual raw physical charge images on COMPLETE one-particle CAR32xPW19; no band or orbit compression. [rho0,c1dag Udag c0]=-q c1dag Udag c0 and global colour commutators checked BEFORE declared h-block selection. Bilinear charge algebra proves same identity in every N. Not a whole-H/current range closure or vacuum/source spectrum.',resource=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-started))
    Path(str(prefix)+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(result=str(prefix)+'_result.json',resource=result['resource'])))

if __name__=='__main__':main()
