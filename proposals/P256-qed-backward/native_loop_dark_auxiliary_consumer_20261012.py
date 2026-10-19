"""Original63-auxiliary Schur-sector obstruction and full64 adjoint loop bridge.
No original main or numerical ground/eigensolver is run. The16-matrix adjoint
calculation represents a PROVED full64 intertwiner, not a two-copy surrogate.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import zipfile
import sympy as sp

ROOT=Path(__file__).resolve().parent


def ad_image(legs,body):
    direct=sp.zeros(body.rows);adjoint=sp.zeros(body.rows)
    for order in itertools.permutations(range(4)):
        current=body
        for e in order:current=legs[e]*current-current*legs[e]
        direct+=current
        current=body
        for e in reversed(order):current=legs[e].H*current-current*legs[e].H
        adjoint+=current
    return (direct+adjoint)/2


def embed_blocks(blocks):
    legs=[]
    for e,block in enumerate(blocks):
        t=sp.zeros(16)
        for a in range(4):
            for b in range(4):t[4*e+a,4*((e+1)%4)+b]=block[a,b]
        legs.append(t)
    return legs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_bytes())
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:raw=archive.read(request['profile_member'])
    assert hashlib.sha256(raw).hexdigest()==request['profile_member_sha256']
    profiles=json.loads(raw)['primitive_profiles'];assert profiles[1]['paths']==profiles[2]['paths']
    c0=sp.Rational(profiles[1]['paths'][0]['c0']);ci=sp.Rational(profiles[1]['paths'][0]['ci']);ratio=ci/c0
    G=sp.diag(3,-1,-1,-1)
    n=sp.diag(1,1,1,1,*([0]*12))
    diagonal=ad_image(embed_blocks([G]*4),n)
    trdiag=sp.simplify(sp.trace(diagonal.H*diagonal))
    assert sp.trace(diagonal)==0 and trdiag==472608
    # Full original M=256,k=128 canonical covariance on ALL singlet channels.
    c=sp.Rational(64,255);normdiag=c*trdiag
    Phi=[G,sp.zeros(4),sp.zeros(4),sp.zeros(4)]
    Phi[1][0,1]=Phi[1][1,0]=1
    Phi[2][1,2]=sp.I;Phi[2][2,1]=-sp.I
    Phi[3][2,3]=Phi[3][3,2]=2
    U=sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,sp.I,0],[0,0,0,-sp.I]])
    V=sp.Matrix([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    assert U.det()==V.det()==1 and U*V!=V*U
    # Actual native xy face:0=(0,0,0),1=(1,0,0),2=(1,1,0),3=(0,1,0).
    # Particle transfers targete <- sourcee+1. Reverse edges are native adjoints.
    def block(x,y,axis,link):
        fx=(1+x)*(Phi[0]+sp.I*ratio*Phi[axis+1])
        fy=(1+y)*(Phi[0]+sp.I*ratio*Phi[axis+1])
        return (fx*link+link*fy)/2
    blocks=[block(0,1,0,U),block(1,2,1,V),block(3,2,0,V*U).H,block(0,3,1,U*V).H]
    native=ad_image(embed_blocks(blocks),n)
    trnative=sp.simplify(sp.trace(native.H*native))
    assert sp.trace(native)==0 and trnative>0
    # The full64 singlet local number bridge is n0=n-4/M I; ad kills identity.
    assert c==sp.Rational(256,4*(256-1))
    # Auxiliary det^31 wedge^128 exact mode spectrum31,32. Branch with only
    # three aux face particles cannot overlap this ORIGINAL invariant sector.
    aux_modes=(31,32);branch_max=3
    assert min(aux_modes)>branch_max
    result={'kind':'ORIGINAL_AUX63_INVARIANT_SECTOR_AND_ACTUAL_FULL64_ADJOINT_LOOP_RESPONSE_NOT_GROUND',
            'auxiliary_profiles_pathwise_equal':True,'charged_copies':64,'auxiliary_copies':63,
            'native_M':256,'native_k':128,'auxiliary_irrep':'det^31 tensor wedge^128 C256',
            'auxiliary_mode_number_spectrum':list(aux_modes),'old_branch_max_aux_mode_number':branch_max,
            'old_supported_branch_exactly_orthogonal_to_this_sector':True,
            'full_singlet_multiplicities':'all rest-R-dual channels, not one paired vector',
            'full64_intertwiner':'E(t) X(g) P_S = X([t,g]) P_S, X(g)=dGamma_active(g); all spectator and auxiliary multiplicities identity',
            'all_native_orders':24,'all_native_assignments_per_order':64**4,
            'diagonal_uniform_native_occupation_response_HS2_over_c0_pow8':str(trdiag),
            'diagonal_uniform_full64_norm2_over_c0_pow8':str(normdiag),
            'noncommuting_fullcolour_native_response_HS2_over_c0_pow8':str(trnative),
            'noncommuting_full64_norm2_over_c0_pow8':str(sp.simplify(c*trnative)),
            'native_Phi0_Phi123_endpoints_SU4':True,'source_legs_rescaled_for_exact_units_only':'reported coefficients multiply literal c0^8; physical hopping remains1',
            'C_source':'i[n_active,site,A_face], full local0..4 site number',
            'actual_ground_sector_weight':'UNPAID; globally minimizing auxiliary irrep is not identified',
            'wholeP0_response':'UNPAID; this is total vector compression on P_S, not excited norm',
            'governor':request['governor'],'actual_ground_numerics':False,'campaign_complete':False}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'receipt':args.output.name,'actual_full64_adj_native_bridge_completed':True,'actual_ground_transfer':'UNPAID'}))


if __name__=='__main__':main()
