"""Independent exact all-field cap/tree/canonical consumers; no ground solver."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import zipfile
import sympy as s


def adj(a):
    return a.conjugate().T


def clean(a):
    return a.applyfunc(s.cancel)


def hs2(a):
    return s.cancel(s.trace(a*adj(a)))


def joint2(p):
    return sum(hs2(a) for a in p)


def jordan(p):
    return s.cancel(4*sum(s.trace((a*b+b*a)**2) for a in p for b in p))


def native(fields, edges, m, c0, ci):
    h = s.zeros(4*len(fields))
    for x,p in enumerate(fields):
        h[4*x:4*x+4,4*x:4*x+4] = m*p[0]
    for x,y,a,u,ap in edges:
        f = clean(ap*(c0*(fields[x][0]*u+u*fields[y][0])+s.I*ci*(fields[x][a+1]*u+u*fields[y][a+1]))/2)
        h[4*x:4*x+4,4*y:4*y+4] += f
        h[4*y:4*y+4,4*x:4*x+4] += adj(f)
    return h


def partitions(n,r):
    q = n//r
    lengths = [r]*q
    for j in range(n-q*r):
        lengths[j%q] += 1
    assert sum(lengths)==n and all(r<=v<=2*r-1 for v in lengths)
    starts = set(itertools.accumulate([0]+lengths[:-1]))
    counts = [sum((x-shift)%n in starts for shift in range(n)) for x in range(n)]
    assert counts==[q]*n and s.Rational(q,n)<=s.Rational(1,r)
    return {'N':n,'R':r,'lengths':lengths,'cut_probability':str(s.Rational(q,n))}


def controls(profile):
    m=s.Rational(profile['onsite']); c0=s.Rational(profile['paths'][0]['c0']); ci=s.Rational(profile['paths'][0]['ci'])
    assert len(profile['paths'])==3 and all(p['length']==1 and s.Rational(p['c0'])==c0 and s.Rational(p['ci'])==ci for p in profile['paths'])
    b=abs(c0)+abs(ci); d=2*abs(m)+12*b
    eye=s.eye(4)
    sx=s.Matrix([[0,1],[1,0]]); sy=s.Matrix([[0,-s.I],[s.I,0]]); sz=s.diag(1,-1)
    beta=s.kronecker_product(sz,s.eye(2)); alpha=[s.kronecker_product(sx,t) for t in (sx,sy,sz)]
    body=[beta/2,alpha[0]/2,(beta+alpha[1])/2,alpha[2]/2]
    assert joint2(body)==5 and jordan(body)>0
    units=[(3*eye+4*s.I*beta)/5,(5*eye+12*s.I*alpha[0])/13,(8*eye+15*s.I*alpha[1])/17]
    for u in units:
        assert clean(u*adj(u))==eye and s.cancel(u.det())==1
    assert clean(units[0]*units[1]-units[1]*units[0])!=s.zeros(4)
    vertices=list(itertools.product(range(2),repeat=3)); ix={x:i for i,x in enumerate(vertices)}
    def advance(x,a):
        y=list(x);y[a]+=1;return tuple(y)
    links={}
    for x in vertices:
        for a in range(3):
            if x[a]==0:
                links[x,a]=units[(a+sum(x))%3]
    def path(x):
        return [a for a in range(3) for _ in range(x[a])]
    def product(word):
        x=(0,0,0);out=eye
        for a in word:
            out=clean(out*links[x,a]);x=advance(x,a)
        return out
    transports={x:product(path(x)) for x in vertices}
    curvature=s.S.Zero
    for x in vertices:
        for a in range(3):
            for j in range(a+1,3):
                if x[a]==x[j]==0:
                    face=clean(links[x,a]*links[advance(x,a),j]*adj(links[advance(x,j),a])*adj(links[x,j]))
                    cost=s.cancel(4-s.re(s.trace(face)))
                    assert cost>=0 and hs2(face-eye)==2*cost
                    curvature+=cost
    residual=s.S.Zero; decompositions=[]
    for (x,a),u in links.items():
        y=advance(x,a);word=path(x)+[a];w0=product(word);terms=[];facecost=s.S.Zero
        for k in range(len(word)-2,-1,-1):
            if word[k]<=a:break
            before=word[:k];start=(0,0,0)
            for j in before:start=advance(start,j)
            j=word[k]
            aa=clean(links[start,j]*links[advance(start,j),a])
            bb=clean(links[start,a]*links[advance(start,a),j])
            pref=product(before)
            conjugated=clean(pref*aa*adj(bb)*adj(pref))
            old=product(word);word[k],word[k+1]=word[k+1],word[k];new=product(word)
            assert clean(old*adj(new))==conjugated
            terms.append(conjugated);facecost+=hs2(aa*adj(bb)-eye)
        assert word==path(y)
        mult=eye
        for t in terms:mult=clean(mult*t)
        target=clean(transports[x]*u*adj(transports[y]))
        assert mult==target and hs2(target-eye)<=len(terms)*facecost
        residual+=hs2(target-eye)
        decompositions.append({'tail':x,'axis':a,'conjugated_faces':len(terms),'residual_HS2':str(hs2(target-eye))})
    assert curvature>0 and residual<=576*curvature
    scales=[s.Integer(1),s.Integer(2),s.Rational(3,2),s.Integer(3),s.Integer(1),s.Integer(2),s.Integer(1),s.Rational(5,2)]
    fields=[[t*p for p in body] for t in scales]
    capped=[body[:] for _ in vertices]
    transformed=[[clean(transports[x]*p*adj(transports[x])) for p in capped[ix[x]]] for x in vertices]
    alignment=s.S.Zero;uncapped_alignment=s.S.Zero
    edges=[];flat=[];gaugeedges=[]
    # A seam crossing an open local coordinate interval is a scalar sign. Gauge it on that contractible interval, not on the torus.
    signs={x:(-1 if x[0] else 1) for x in vertices}
    for (x,a),u in links.items():
        y=advance(x,a);ap=signs[x]*signs[y]
        edges.append((ix[x],ix[y],a,u,ap));flat.append((ix[x],ix[y],a,eye,1))
        uprime=clean(transports[x]*u*adj(transports[y]))
        gaugeedges.append((ix[x],ix[y],a,uprime,1))
        alignment+=sum(hs2(p-u*q*adj(u)) for p,q in zip(capped[ix[x]],capped[ix[y]]))
        uncapped_alignment+=sum(hs2(p-u*q*adj(u)) for p,q in zip(fields[ix[x]],fields[ix[y]]))
    assert alignment<=uncapped_alignment
    assert sum(jordan(p) for p in capped)<=sum(jordan(p) for p in fields)
    h=native(capped,edges,m,c0,ci);hg=native(transformed,gaugeedges,m,c0,ci)
    gauge=s.diag(*(signs[x]*transports[x] for x in vertices))
    assert clean(hg-gauge*h*adj(gauge))==s.zeros(32)
    mean=[clean(sum((p[j] for p in transformed),s.zeros(4))/8) for j in range(4)]
    deviation=sum(joint2([a-z for a,z in zip(p,mean)]) for p in transformed)
    assert joint2(mean)<=5 and deviation<=48*alignment
    # Actual open-box arbitrary-matrix constant functional, not a Clifford replacement.
    h0=native([mean]*8,flat,m,c0,ci)
    assert clean(h0-adj(h0))==s.zeros(32)
    edge=flat[0]
    cut=native([body]*8,[edge],0,c0,ci)
    assert cut.rank()==8
    hop=cut[4*edge[0]:4*edge[0]+4,4*edge[1]:4*edge[1]+4]
    assert hs2(hop)<=b*b*5
    # Exact Jordan directional derivative and an adverse deletion of complete cross pairs.
    t=s.Symbol('t',real=True)
    direction=[alpha[1]/7,beta/9,alpha[2]/11,alpha[0]/13]
    derivative=s.diff(jordan([p+t*v for p,v in zip(body,direction)]),t).subs(t,0)
    assert derivative**2<=64**2*5**3*joint2(direction)
    diagonal=4*sum(s.trace((2*p*p)**2) for p in body)
    assert jordan(body)>diagonal
    # Every GLOBAL canonical occupation: enumerate all choose(12,6), including negative levels and compensation.
    blocks=[[9,-3,-3,-3],[3,-1,-1,-1],[1,s.Rational(-1,3),s.Rational(-1,3),s.Rational(-1,3)]]
    levels=[m*z for block in blocks for z in block]
    canonical_h=native([[s.diag(*block),s.zeros(4),s.zeros(4),s.zeros(4)] for block in blocks],[],m,c0,ci)
    assert list(canonical_h.diagonal())==levels
    choices=list(itertools.combinations(range(12),6))
    winner=max(choices,key=lambda rows:sum(levels[j] for j in rows))
    top=sum(levels[j] for j in winner)
    forced=sum(sum(sorted([m*z for z in block],reverse=True)[:2]) for block in blocks)
    positive=sum(max(z,0) for z in levels)
    assert top!=forced and top!=positive and any(levels[j]<0 for j in winner)
    occupations=[sum(j//4==a for j in winner) for a in range(3)]
    density=[s.Rational(2,3)*int(j in winner)+s.Rational(1,3)*int(j in choices[0]) for j in range(12)]
    assert sum(density)==6 and all(0<=z<=1 for z in density) and sum(g*z for g,z in zip(density,levels))<=top
    R=2**12;L=2**32;M=s.Integer(2**14)
    return {'neutral':{'m':str(m),'c0':str(c0),'ci':str(ci),'b':str(b),'D':str(d)},'partition_controls':[partitions(n,r) for n,r in [(4,1),(4,2),(10,3),(14,4),(18,5)]], 'noncommuting_native_cube':{'vertices':8,'edges':12,'faces':6,'curvature':str(curvature),'residual_HS2_sum':str(residual),'conjugated_face_decompositions':decompositions,'joint_radius_squared':5,'alignment_capped':str(alignment),'alignment_original':str(uncapped_alignment),'tree_mean_deviation_squared':str(deviation),'mean_joint_HS2':str(joint2(mean)),'endpoint_gauge_AP_exact':True,'open_constant_matrix_dimension':h0.rows,'cut_rank':cut.rank(),'ordered_Jordan':str(jordan(body)),'diagonal_only_Jordan':str(diagonal),'Jordan_directional_derivative':str(derivative)},'canonical':{'enumerated_complete_occupations':len(choices),'dimension':12,'particles':6,'top':str(top),'forced_two_per_box':str(forced),'positive_only':str(positive),'winning_box_occupations':occupations,'negative_levels_retained':True,'actual_conditional_density_control':True},'first_gate':{'R':R,'L':L,'boundary':str(12*b*L/R),'tail':str(d*M/L**3),'fails_minus21_gate':bool(12*b*L/R>21),'no_go':False},'scope':'Exact finite native controls accompany independent analytical all-even-N proof; no ground vector, optimizer, old CLI, periodic minimum, Clifford projection, or global Poincare.'}


def main():
    sys.set_int_max_str_digits(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text());root=args.request.parent
    for path,digest in request['pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    with zipfile.ZipFile(root/request['original_archive']) as z:
        data=z.read('neutral_body_parent_primitive_profiles_20261012.json')
        assert hashlib.sha256(data).hexdigest()==request['profile_sha256']
        profile=json.loads(data)['primitive_profiles'][0]
    result=controls(profile)
    result['request_sha256']=hashlib.sha256(args.request.read_bytes()).hexdigest()
    result['scientific_admission']=False;result['campaign_complete']=False
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Independent exact native cap/tree/conjugated curvature/AP/global canonical controls passed; capped gate failed, not a no-go.')


if __name__=='__main__':
    main()
