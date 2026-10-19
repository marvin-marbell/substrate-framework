from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix

from affine_canonical_lower_20261011 import construct, ROOT, FLOOR
from neutral_return_lower_control_20261011 import (
    Q, ZERO, iv, add, sub, mul, scale, mid, endpoints, digest,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',required=True)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args();assert args.tag.startswith('affine_canonical_')
    start=time.monotonic();receipt=json.loads((ROOT/args.receipt).read_text())
    result=[]
    for family in receipt['families']:
        if not family['proven_lower_form_failure']:continue
        kind=family['kind'];roles=tuple(family['roles']);u=int(family['u_exact'])
        matrix_path=ROOT/family['matrix_path'];witness_path=ROOT/family['witness_path']
        assert digest(matrix_path)==family['matrix_sha256']
        assert digest(witness_path)==family['witness_sha256']
        matrix=json.loads(matrix_path.read_text())
        witness=np.load(witness_path)
        base,states,index,outputs,neighbors,_,_=construct(kind,roles,family['output_budget'],family['support_budget'])
        D=[sub(add(base,cost),iv(FLOOR)) for _,cost,_ in states]
        L=list(map(int,witness['negative']));positive=[j for j,x in enumerate(D) if x[0]>0]
        rows={j:r for r,j in enumerate(outputs)};rr=[];cc=[];vv=[]
        for c,j in enumerate(positive):
            if j in rows:rr.append(rows[j]);cc.append(c);vv.append(float(u))
            for key,value in neighbors(states[j][0]):
                target=index.get(key)
                if target in rows:rr.append(rows[target]);cc.append(c);vv.append(mid(value))
        BQ=coo_matrix((vv,(rr,cc)),shape=(len(outputs),len(positive))).tocsr()
        center=np.array([[float(Fraction(x,Q)) for x in row] for row in matrix['center_integer']])
        values,vectors=np.linalg.eigh(center);v=vectors[:,0]
        x=np.zeros(len(states));x[L]=v
        x[positive]=-64*(BQ.T@(witness['Y']@v))/np.array([mid(D[j]) for j in positive])
        # The trial coefficients are exact binary rationals. The whole image
        # of rho+u is formed, including outputs OUTSIDE the retained Schur rows.
        XI=[iv(float(z)) for z in x]
        norm=ZERO;free=ZERO;image={}
        for j,(key,_,_) in enumerate(states):
            if x[j]==0:continue
            sq=mul(XI[j],XI[j]);norm=add(norm,sq);free=add(free,mul(D[j],sq))
            image[key]=add(image.get(key,ZERO),scale(XI[j],u))
            for target,value in neighbors(key):
                image[target]=add(image.get(target,ZERO),mul(value,XI[j]))
        density=ZERO
        for value in image.values():density=add(density,mul(value,value))
        form=add(free,scale(density,64))
        assert norm[0]>0
        upper=Fraction(form[1],norm[0]) if form[1]>=0 else Fraction(form[1],norm[1])
        lower=Fraction(form[0],norm[1]) if form[0]>=0 else Fraction(form[0],norm[0])
        stem=f'{args.tag}_{kind}_{"".join(map(str,roles))}_u{u:+d}'
        trial=ROOT/(stem+'_trial.npz')
        np.savez_compressed(trial,x=x,states=witness['states'],u=np.array(u))
        item={'kind':kind,'roles':roles,'u_exact':str(u),'number_sector':256,
              'support_columns':len(states),'complete_output_rows':len(image),
              'retained_schur_output_rows':len(outputs),'full_output_not_PHP':True,
              'norm_squared_exact_interval':endpoints(norm),
              'H_minus_floor_form_exact_interval':endpoints(form),
              'Rayleigh_minus_floor_exact_interval':[str(lower),str(upper)],
              'Rayleigh_H_upper_exact':str(Fraction(FLOOR)+upper),
              'proves_floor_false':upper<0,'trial_path':trial.name,'trial_sha256':digest(trial),
              'matrix_path':matrix_path.name,'matrix_sha256':digest(matrix_path),
              'witness_path':witness_path.name,'witness_sha256':digest(witness_path)}
        result.append(item);print(json.dumps(item),flush=True)
    path=ROOT/(args.tag+'_receipt.json')
    with path.open('x') as f:
        json.dump({'kind':'EXACT_RATIONAL_FINITE_SUPPORT_FULL_OUTPUT_VARIATIONAL_CONTROL',
                   'input_receipt':args.receipt,'input_receipt_sha256':digest(ROOT/args.receipt),
                   'trials':result,'source_sha256':digest(Path(__file__)),
                   'no_ground_eigenvector_identification':True,
                   'elapsed_seconds':time.monotonic()-start},f,indent=2)


if __name__=='__main__':
    main()
