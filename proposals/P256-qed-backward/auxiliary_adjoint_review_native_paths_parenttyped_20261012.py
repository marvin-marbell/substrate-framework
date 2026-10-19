"""New exact NN/two/current support and signed-identity native review controls."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import zipfile
import itertools
import sympy as s

ROOT=Path(__file__).resolve().parent


def path(start,axis,length,n):
    point=start; links=[]; seam=1
    for _ in range(length):
        links.append((point,axis))
        if point[axis]==n-1:seam=-seam
        next_point=list(point); next_point[axis]=(next_point[axis]+1)%n; point=tuple(next_point)
    return point,tuple(links),seam


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--request',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); request=json.loads(args.request.read_bytes())
    with zipfile.ZipFile(ROOT/(request['epoch_prefix']+'_preexecution.zip')) as sealed:
        with zipfile.ZipFile(io.BytesIO(sealed.read(request['profile_archive']))) as archive:
            profiles=json.loads(archive.read(request['profile_member']))['primitive_profiles']
    rows=[]
    for n in (4,6,8):
        site=(n-1,0,0); face=(site,(0,0,0),(0,1,0),(n-1,1,0))
        face_origins=((face[0],0),(face[1],1),(face[3],0),(face[0],1))
        seams=[path(start,axis,1,n)[2] for start,axis in face_origins]
        assert seams==[-1,1,-1,1]
        current=[]; occupation=[]; support=set()
        selected_link=(site,0)
        for start in itertools.product(range(n),repeat=3):
            for index,p in enumerate(profiles[1]['paths']):
                end,links,seam=path(start,p['axis'],p['length'],n)
                if selected_link in links:current.append((start,end,index,seam));support.update((start,end))
                if site in (start,end):occupation.append((start,end,index,seam))
        expected={((site[0]-1)%n,0,0),site,((site[0]+1)%n,0,0),((site[0]+2)%n,0,0)}
        assert len(current)==3 and support==expected and len(occupation)==12
        assert sorted(profiles[1]['paths'][i]['length'] for _,_,i,_ in current)==[1,2,2]
        # Two primitive length2 orientations alias at N4; they stay distinct.
        length2=[(a,b) for a,b,i,seam in occupation if profiles[1]['paths'][i]['length']==2 and profiles[1]['paths'][i]['axis']==0]
        assert len(length2)==2
        assert ((length2[0][0],length2[0][1])==(length2[1][1],length2[1][0]))==(n==4)
        # Enumerate complete graph matter locality, including intermediate links.
        corners=set(face); graphsites: set[tuple[int, ...]]=set(corners)
        for start in itertools.product(range(n),repeat=3):
            for p in profiles[1]['paths']:
                end,links,seam=path(start,p['axis'],p['length'],n)
                if start in corners or end in corners:
                    graphsites.update((start,end));graphsites.update(x for x,_ in links)
        assert len(graphsites)<=52
        rows.append({'N':n,'wrapped_face_AP_seams':seams,'face_AP_product':s.prod(seams),
                     'current_primitive_paths':len(current),'current_sites':sorted(support),
                     'occupation_incident_primitive_paths':len(occupation),
                     'complete_face_incident_sites':len(graphsites),
                     'N4_length2_reverse_aliases_retained':n==4})
    # Nonzero complex resolvent numerator distinguishes the rationale sign.
    omega=s.Matrix([1,0,0]); p0=s.diag(1,0,0); q0=s.eye(3)-p0
    number=s.Matrix([[2,1,s.I],[1,2,0],[-s.I,0,2]])
    probe=s.Matrix([[0,1+s.I,2],[1-s.I,0,0],[2,0,0]])
    z=(omega.H*number*q0*probe*omega)[0]
    signed=(omega.H*s.I*(number*probe-probe*number)*omega)[0]
    ground=(omega.H*s.I*(number*p0*probe-probe*p0*number)*omega)[0]
    assert s.expand(z)==1+s.I and s.expand(signed)==-2 and ground==0
    assert s.expand(signed-ground)==-2*s.im(s.expand(z))
    assert s.im(s.expand(z)) != s.expand(signed-ground)/2
    coefficient=8*sum(abs(s.Rational(p['c0']))+abs(s.Rational(p['ci'])) for p in profiles[1]['paths'])
    result={'kind':'INDEPENDENT_NATIVE_PATH_SUPPORT_AND_NONZERO_COMPLEX_SIGN_CONTROL',
            'rows':rows,'Dn_coefficient_times_C85_pow_2over17':str(coefficient),
            'signed_rationale':{'z':str(s.expand(z)),'c':str(s.expand(signed)),'exactP0_term':str(ground),
                                'identity':'c-p=-2 Im z','wrong_positive_half_rejected':True,
                                'displayed_squared_resolvent_lower_unchanged':True},
            'all65_neutral_Ref_once_sharedE_retained':True,'original_main_executed':False,
            'native_ground_or_wholeP0_Gram':False,'governor':request['governor']}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,default=str);stream.write('\n')
    print(json.dumps({'receipt':args.output.name,'native_path_and_sign_controls_completed':True}))


if __name__=='__main__':
    main()
