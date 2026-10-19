"""Exact original periodic N4 affine deck control; no slab/source execution."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);args=ap.parse_args()
    req=json.loads((ROOT/args.request).read_text())
    captured={name:(ROOT/name).read_bytes() for name in req['pins']}
    for name,raw in captured.items():assert hashlib.sha256(raw).hexdigest()==req['pins'][name],name
    sites=list(itertools.product(range(4),repeat=3));origin=(0,0,0);end=(1,0,0)
    roots=[(1,0),(0,1),(-1,0),(0,-1)]
    eigen=[0,2,4,2]
    potential={}
    for x in sites:
        real=F(0);imag=F(0)
        for p in sites:
            eig=sum(eigen[j] for j in p)
            if not eig:continue
            a,b=roots[sum(j*k for j,k in zip(p,x))%4]
            c,d=roots[(-p[0])%4]
            # exp(i p.x)*(1-exp(-i p_axis))/ell, normalized byV.
            real+=F(a*(1-c)+b*d,64*eig)
            imag+=F(b*(1-c)-a*d,64*eig)
        assert imag==0
        potential[x]=real
    assert sum(potential.values())==0
    rho={x:int(x==origin)-int(x==end) for x in sites}
    for x in sites:
        lap=F(0)
        for axis in range(3):
            for sign in (-1,1):
                y=list(x);y[axis]=(y[axis]+sign)%4
                lap+=potential[x]-potential[tuple(y)]
        assert lap==rho[x]
    r=potential[origin]-potential[end]
    assert r==F(21,64)
    peierls=1-r
    assert peierls==F(43,64)
    assert (peierls+r).denominator==1
    wrong=peierls-r
    assert wrong==F(11,32) and wrong.denominator!=1
    second=(2,0,0);long=potential[origin]-potential[second]
    projected_path=1-long
    assert (projected_path+long).denominator==1
    assert (projected_path-long).denominator!=1
    result={'kind':'EXACT_ORIGINAL_PERIODIC_N4_MEAN_ZERO_AFFINE_DECK_NOT_SLAB',
            'vertices':64,'edges':192,'incidence_rank':63,'transverse_dimension':129,
            'mean_zero_poisson_equation_every_site_reached':True,
            'single_edge_longitudinal_turns':str(r),'single_edge_Peierls_turns':str(peierls),
            'full_section_plus_total_turns':str(peierls+r),
            'wrong_forward_plus_residual_turns':str(wrong),
            'native_length2_longitudinal_turns':str(long),
            'native_length2_projected_Peierls_turns':str(projected_path),
            'native_length2_correct_minus_intertwining':True,
            'native_length2_wrong_forward_plus_rejected':True,
            'original_compatible_character_retained':True,'input_pins':req['pins'],
            'original_functions_executed':False,'scientific_admission':False,'campaign_complete':False}
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    (ROOT/req['receipt']).write_text(json.dumps(result,indent=2)+'\n')
    print('periodic mean-zero native NN/length2 affine forward-gluing boundary controls completed')


if __name__=='__main__':
    main()
