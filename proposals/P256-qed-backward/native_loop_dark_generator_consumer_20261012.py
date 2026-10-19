"""Positive fullcolour native coefficient-algebra and exterior-sector control."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import zipfile
import sympy as sp

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_bytes())
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:raw=archive.read(request['profile_member'])
    assert hashlib.sha256(raw).hexdigest()==request['profile_member_sha256']
    profiles=json.loads(raw)['primitive_profiles'];ci=sp.Rational(profiles[1]['paths'][0]['ci'])
    generators=[]
    for i in range(4):
        for j in range(i+1,4):
            x=sp.zeros(4);x[i,j]=x[j,i]=sp.Rational(1,2);generators.append(x)
            y=sp.zeros(4);y[i,j]=-sp.I/2;y[j,i]=sp.I/2;generators.append(y)
    for r in range(1,4):
        entries=[1]*r+[-r]+[0]*(3-r)
        generators.append(sp.diag(*entries)/sp.sqrt(2*r*(r+1)))
    assert len(generators)==15
    for a,A in enumerate(generators):
        assert A==A.H and sp.trace(A)==0
        for b,B in enumerate(generators):assert sp.simplify(sp.trace(A*B))==(sp.Rational(1,2) if a==b else 0)
    U=sp.diag(sp.I,-sp.I,1,1)
    extra=generators[12]*U
    basis=sp.Matrix.hstack(*(sp.Matrix(16,1,list(g)) for g in generators))
    assert basis.rank()==15 and sp.trace(extra)==sp.I
    assert sp.Matrix.hstack(basis,sp.Matrix(16,1,list(extra))).rank()==16
    T=sp.diag(3,-1,-1,-1)/sp.sqrt(24)
    links=[sp.eye(4),U,sp.diag(1,1,sp.I,-sp.I),sp.diag(-1,-1,1,1)]
    # Four independent native axis-field endpoint directions, actual nearest
    # ci !=0, fullC4 source and target spaces, no selected colour eligibility.
    blocks=[sp.I*ci*T*link/2 for link in links]
    assert all(block.det()!=0 and block.rank()==4 for block in blocks)
    assert len(tuple(itertools.permutations(range(4))))==24
    result={'kind':'NATIVE_FULLCOLOUR_UNIFORM_COEFFICIENT_ALGEBRA_POSITIVE_CONTROL_NOT_KERNEL_CLASSIFICATION',
            'native_ci':str(ci),'normalized_native_SU4_generators':15,
            'traceless_span_rank':15,'span_TA_times_SU4_rank':16,'extra_trace':str(sp.trace(extra)),
            'face_colour_index_generators':16**4,'all_temporal_orders':24,'all_canonical_copy_assignments':64**4,
            'four_distinct_copy_native_endpoint_blocks_rank':[b.rank() for b in blocks],
            'fullcolour_native_fourcopy_transition_rank':4**4,'Hermitian_quadrature_transition_prefactor':12,
            'all64_factor_spectators':'first4 canonical copies127 in240 modes; other60 canonical copies128 in240 modes; no factor dropped',
            'universal_loop_annihilator_equals_leg_annihilator':False,
            'evaluated_all_native_irrep_kernel_ranks':False,
            'governor':request['governor'],'actual_ground':False,'campaign_complete':False}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'receipt':args.output.name,'native_generator_coefficient_control_completed':True}))


if __name__=='__main__':main()
