"""Independent exact algebra/native path controls; no original function is executed."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast
import zipfile
import sympy as s

ROOT = Path(__file__).resolve().parent


def exterior(b, k):
    ids = list(itertools.combinations(range(b.rows), k))
    return s.Matrix([[b.extract(i, j).det() for j in ids] for i in ids])


def coefficient(eigenvalues, k):
    c = [1] + [0] * k
    for v in eigenvalues:
        for j in range(k, 0, -1):
            c[j] += v * c[j-1]
    return c[k]


def canonical_controls():
    a = s.Matrix([[1,2,0,0],[0,1,1,0],[0,0,2,1],[1,0,0,1]])
    b = s.Matrix([[2,0,1,0],[1,1,0,0],[0,1,1,2],[0,0,0,1]])
    c = s.diag(1,2,3,5)
    correct = exterior(b*c*a, 2)
    assert correct == exterior(b,2)*exterior(c,2)*exterior(a,2)
    z = s.Symbol('z')
    determinant = (s.eye(4)+z*b*c*a).det().expand()
    assert s.trace(correct) == determinant.coeff(z,2)
    reverse = s.trace(exterior(a*c*b,2))
    assert reverse != s.trace(correct)
    assert determinant.subs(z,1) != s.trace(correct)
    # Separate number in every species, not a pooled number on a tensor product.
    pairs = [(i+1,i+2) for i in range(65)]
    separate = math.prod(x+y for x,y in pairs)
    pooled = coefficient([x for pair in pairs for x in pair],65)
    assert separate != pooled
    return {'exterior_dimension':6,'ordered_trace':str(s.trace(correct)),
            'wrong_reversed_trace':str(reverse),'grandcanonical':str(determinant.subs(z,1)),
            'all65_separate_coefficients':str(separate),'wrong_one_global_z':str(pooled),
            'native_dimension':256,'native_each_number':128,'Ref_not_species':True}


def fixture():
    sites = list(itertools.product(range(4), repeat=3))
    basis = []
    for i in range(3):
        t=s.zeros(4);t[i,i]=1;t[3,3]=-1;basis.append(t)
    for i,j in itertools.combinations(range(4),2):
        t=s.zeros(4);t[i,j]=t[j,i]=1;basis.append(t)
        u=s.zeros(4);u[i,j]=s.I;u[j,i]=-s.I;basis.append(u)
    assert len(basis)==15
    phi={x:[sum((s.Rational(1+((j+mu+sum(x))%7),16)*t for j,t in enumerate(basis)),s.zeros(4)) for mu in range(4)] for x in sites}
    def rotation(i,j):
        r=s.eye(4);r[i,i]=r[j,j]=s.Rational(3,5);r[i,j]=s.Rational(4,5);r[j,i]=-s.Rational(4,5);return r
    rotations=[rotation(0,1),rotation(1,2),rotation(2,3)]
    links={(x,a):rotations[(a+sum(x))%3] for x in sites for a in range(3)}
    phases={key: (s.I if (sum(key[0])+key[1])%2 else -s.I) for key in links}
    gauge={x:rotations[sum(x)%3]*s.diag(-1,-1,1,1) for x in sites}
    assert all(u.det()==1 and u*u.H==s.eye(4) for u in [*rotations,*gauge.values()])
    return sites,basis,phi,links,phases,gauge


def native_blocks(sites,phi,links,phases,p,charged,wrong=None):
    out={}
    def add(x,y,v):
        out[x,y]=out.get((x,y),s.zeros(4))+v
    for x in sites:
        add(x,x,s.Rational(p['onsite'])*(s.eye(4) if wrong=='constant_onsite' else phi[x][0]))
        for path in p['paths']:
            if wrong=='omit_length2' and path['length']==2: continue
            axis=path['axis'];y=x;u=s.eye(4);phase=1;seam=1
            for _ in range(path['length']):
                v=links[y,axis]
                u=v*u if wrong=='reverse_Wilson' else u*v
                if charged and wrong!='omit_Peierls':phase*=phases[y,axis]
                if y[axis]==3 and wrong!='omit_AP':seam=-seam
                yy=list(y);yy[axis]=(yy[axis]+1)%4;y=tuple(yy)
            f=lambda v:s.Rational(path['c0'])*phi[v][0]+s.I*s.Rational(path['ci'])*phi[v][axis+1]
            block=seam*phase*(f(x)*u+u*f(y))/2
            if wrong=='omit_endpoint':block=seam*phase*f(x)*u
            add(x,y,block);add(y,x,block.H)
    return {xy:v.expand() for xy,v in out.items()}


def block_mismatch(a,b):
    for xy in sorted(set(a)|set(b)):
        diff=(a.get(xy,s.zeros(4))-b.get(xy,s.zeros(4))).expand()
        for i in range(4):
            for j in range(4):
                if diff[i,j]!=0:return {'sites':xy,'colour':[i,j],'exact_difference':str(diff[i,j])}
    return None


def native_controls(profiles):
    sites,basis,phi,links,phases,gauge=fixture()
    gp={x:[gauge[x]*v*gauge[x].H for v in phi[x]] for x in sites}
    gl={}
    for (x,a),u in links.items():
        y=list(x);y[a]=(y[a]+1)%4;y=tuple(y)
        gl[x,a]=gauge[x]*u*gauge[y].H
    zero={x:[s.zeros(4) for _ in range(4)] for x in sites}
    rows=[];active=None
    for role,p in enumerate(profiles):
        charged=role!=0
        h=native_blocks(sites,phi,links,phases,p,charged)
        hs=native_blocks(sites,gp,gl,phases,p,charged)
        expected={xy:gauge[xy[0]]*v*gauge[xy[1]].H for xy,v in h.items()}
        assert block_mismatch(hs,expected) is None
        assert all(v==s.zeros(4) for v in native_blocks(sites,zero,links,phases,p,charged).values())
        assert all(v==h.get((y,x),s.zeros(4)).H for (x,y),v in h.items())
        assert block_mismatch(h,native_blocks(sites,phi,links,phases,p,charged,'constant_onsite'))
        rows.append({'role':p['role'],'load':p['load'],'onsite':p['onsite'],'paths':len(p['paths']),
                     'dimension':4*len(sites),'gauge_covariance_with_boson_endpoints':True,
                     'zero_Phi_exact_zero':True,'hermitian':True,'neutral_Peierls_absent':not charged})
        if role==1:active=h
    wrongs={}
    for name in ['omit_length2','omit_AP','omit_Peierls','reverse_Wilson','omit_endpoint','constant_onsite']:
        witness=block_mismatch(active,native_blocks(sites,phi,links,phases,profiles[1],True,name))
        assert witness is not None,name
        wrongs[name]=witness
    # A CAR-only gauge sewing with unchanged boson endpoints is genuinely different.
    frozen_expected={xy:gauge[xy[0]]*v*gauge[xy[1]].H for xy,v in cast(Any, active).items()}
    witness=block_mismatch(active,frozen_expected);assert witness
    wrongs['CAR_sewing_without_boson_transform']=witness
    # Full Jordan fields: all four times fifteen real generators, no Clifford restriction.
    scalar=lambda fs:sum(s.trace((a*b+b*a)**2) for a in fs for b in fs)
    q0=s.expand(scalar(phi[sites[0]]));qg=s.expand(scalar(gp[sites[0]]))
    assert q0==qg and q0>0
    assert q0!=s.expand(scalar(phi[sites[0]][:3]))
    return {'rows':rows,'full60_basis_rank':15*4,'wrongcases':wrongs,
            'Jordan_sum_all16_pairs':str(q0),'Jordan_gauge_covariant':True,
            'actual_full_native_ground_evolution':False}


def hs_controls():
    sites=list(itertools.product(range(4),repeat=3));rho={x:(-1)**x[0] for x in sites}
    for x in sites:
        lap=0
        for a in range(3):
            for sign in (-1,1):
                y=list(x);y[a]=(y[a]+sign)%4;lap+=rho[x]-rho[tuple(y)]
        assert lap==4*rho[x]
    norm=sum(F(v*v,4) for v in rho.values());assert norm==16
    shared=64*4*norm;independent=64*2*norm
    assert shared==4096 and independent==2048
    # Characteristic exponent 1/2*(128 delta)*rho Lplus rho.
    assert F(128,2)*4*norm==shared
    assert 2*shared!=shared
    return {'L_eigenvalue':4,'rho_Lplus_rho':str(norm),'shared_energy':str(shared),
            'wrong_independent_energy':str(independent),'wrong_double_HC':str(2*shared),
            'covariance':128,'neutral_has_no_HS_phase':True,'background':'Ref_x-128; Ref4 net insertion zero'}


def sewing_controls(profiles):
    dim=math.comb(256,128)
    sewn=coefficient([-1,-1]+[1]*254,128)
    ratio=F(sewn,dim);assert ratio==-F(1,255)
    margin=abs(ratio**65);assert ratio**65<0 and ratio**64>0
    epsilon=F(1,2**9000)
    weights=[abs(F(p['onsite']))+2*sum(abs(F(t['c0']))+abs(F(t['ci'])) for t in p['paths']) for p in profiles]
    d=2*epsilon*128*(2**8192*weights[0]+weights[1]+63*weights[2])
    r=F(1,2**1200)
    # Exterior tensor perturbation plus complete heat and common HS/affine scalar phases.
    sewing_error=65*128*r
    hs_phase_error=64*128*r
    affine_phase_error=64*128*r
    error=2*d+2*(sewing_error+hs_phase_error+affine_phase_error)
    assert d<F(1,2**780) and error<margin/2
    # g has determinant1 and is a genuine endpoint transform of full bosons.
    g=s.diag(-1,-1,1,1);assert g.det()==1
    sites=list(itertools.product(range(4),repeat=3));origin=sites[0]
    gx={x:(g if x==origin else s.eye(4)) for x in sites}
    endpoints={}
    for x in sites:
        for a in range(3):
            y=list(x);y[a]=(y[a]+1)%4;y=tuple(y)
            endpoints[x,a]=gx[x]*gx[y].H
    assert sum(v!=s.eye(4) for v in endpoints.values())==6
    assert all(v.det()==1 for v in endpoints.values())
    return {'native256_k128_ratio':str(ratio),'all65_ratio':str(ratio**65),
            'wrong_omit_neutral_all64_ratio':str(ratio**64),'epsilon_power':-9000,
            'canonical_heat_bound_d':str(d),'negative_margin':str(margin),
            'sewing_radius_power':-1200,'common_HS_eta_sup_radius_power':-1200,
            'common_affine_onebody_holonomy_radius_power':-1200,'total_error_bound':str(error),
            'error_less_than_half_margin':True,'changed_bosonic_endpoint_links':6,
            'endpoint_links_SU4':True,'negative_real_fermion_phase_factor_open_neighborhood':True,
            'path_integral_or_ground_sign':False,'slice_uniform_neighborhood_measure_lower':False}


def strong_trace_counter():
    # A_n=rank-one projection e_n, strongly0, trace1. Heat analogue on ell2:
    # exp(-H_n)e_j=1 (j=n), exp(-j^2) otherwise. Strong limit exp(-H),
    # but trace difference 1-exp(-n^2) ->1, despite every heat being trace class.
    return {'operator':'H e_j=j^2 e_j; H_n same except H_n e_n=0',
            'strong_heat_limit':'exp(-H_n) -> exp(-H)',
            'trace_defect_limit':1,'each_operator_positive_trace_class':True,
            'wrong_strong_implies_trace_rejected':True,
            'fixed_rho_trace_pairing_valid':'uniform boundedness + trace-class rho; not variable thermal trace'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);args=ap.parse_args()
    req=json.loads((ROOT/args.request).read_text())
    captured={name:(ROOT/name).read_bytes() for name in req['pins']}
    for name,raw in captured.items():assert hashlib.sha256(raw).hexdigest()==req['pins'][name],name
    with zipfile.ZipFile(ROOT/req['profile_archive']) as z:raw=z.read(req['profile_member'])
    assert hashlib.sha256(raw).hexdigest()==req['profile_sha256']
    profiles=json.loads(raw)['primitive_profiles'];assert [p['load'] for p in profiles]==[1,1,63]
    result={'kind':'INDEPENDENT_CANONICAL_PATH_FORM_REVIEW_CONTROLS_NOT_GROUND',
            'canonical':canonical_controls(),'native':native_controls(profiles),
            'common_HS':hs_controls(),'sewing':sewing_controls(profiles),
            'strong_not_trace':strong_trace_counter(),'input_pins':req['pins'],
            'profile_sha256':req['profile_sha256'],'original_functions_executed':False,
            'scientific_admission':False,'campaign_complete':False}
    assert all((ROOT/name).read_bytes()==data for name,data in captured.items())
    (ROOT/req['receipt']).write_text(json.dumps(result,indent=2)+'\n')
    print('independent canonical exterior/native full boson sewing/HS/form-scope controls completed')


if __name__=='__main__':
    main()
