"""New exact scientific review controls; no frozen main or native ground is run."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, cast
import zipfile
import sympy as s

ROOT = Path(__file__).resolve().parent
ORDERS = tuple(itertools.permutations(range(4)))


def comm(a, b):
    return a*b-b*a


def insert(body, slot, copies):
    return s.kronecker_product(*[body if j == slot else s.eye(2) for j in range(copies)])


def collective(body, copies):
    return sum((insert(body, j, copies) for j in range(copies)), s.zeros(2**copies))


def representation_control():
    pauli = [s.Matrix([[0,1],[1,0]])/2, s.Matrix([[0,-s.I],[s.I,0]])/2, s.diag(1,-1)/2]
    ga = [collective(t, 5) for t in pauli]
    ca = sum((g*g for g in ga), s.zeros(32))
    pa = (ca-s.Rational(15,4)*s.eye(32))*(ca-s.Rational(35,4)*s.eye(32))/24
    assert pa*pa == pa and pa.H == pa and s.trace(pa) == 10
    for a in (s.Matrix([[1,s.I],[2,3]]), s.diag(1,0), s.diag(0,1)):
        assert comm(pa, collective(a,5)) == s.zeros(32)
    for a in (s.diag(1,0), s.diag(0,1)):
        number = collective(a,5)
        assert (number-2*s.eye(32))*(number-3*s.eye(32))*pa == s.zeros(32)
    sewing = s.diag(s.I,1)
    wa = s.kronecker_product(*([sewing]*5))
    assert comm(wa,pa) == s.zeros(32)
    assert s.trace(pa*wa) == 5*sewing.det()**2*s.trace(sewing)
    assert s.trace(pa*wa) != 5*s.trace(sewing)
    gs = [collective(t,6) for t in pauli]
    cs = sum((g*g for g in gs), s.zeros(64))
    ps = (s.eye(64)-cs/2)*(s.eye(64)-cs/6)*(s.eye(64)-cs/12)
    assert ps*ps == ps and ps.H == ps and s.trace(ps) == 5
    assert s.kronecker_product(s.eye(2),pa)*ps == ps
    g = s.Matrix([[1,2+s.I],[3*s.I,-1]])
    h = s.Matrix([[s.I,4],[-1,-s.I]])
    xg = insert(g,0,6)*ps
    xh = insert(h,0,6)*ps
    assert ps*xg.H*xh*ps == s.trace(g.H*h)*ps/2
    for t in pauli+[s.Matrix([[0,1],[0,0]])]:
        difference = collective(t,6)*xg-insert(comm(t,g),0,6)*ps
        assert difference.applyfunc(s.expand) == s.zeros(64)
    ws = s.kronecker_product(*([sewing]*6))
    assert ws*ps == sewing.det()**3*ps
    # A species-dependent connection is not a collective connection.
    assert comm(ps,insert(pauli[2],0,6)) != s.zeros(64)
    return {'auxiliary_control':'five genuine wedge^1 C2 factors, full dimension32',
            'auxiliary_isotypic_rank':10,'auxiliary_multiplicity':5,'auxiliary_shift':2,
            'all_common_singlet_control':'six genuine factors, full dimension64',
            'common_singlet_rank':5,'all_multiplicity_complex_cross_compression':True,
            'determinant_sewing_required':True,'active_specific_connection_rejected':True,
            'native63_scope':'general63/M=4V proof analytic, not executed63 Fock matrix'}


def nested(legs, body, reverse=False):
    out = s.zeros(body.rows)
    for order in ORDERS:
        current = body
        for j in (reversed(order) if reverse else order):
            current = comm(legs[j].H if reverse else legs[j], current)
        out += current
    return out


def expanded(legs, body, reverse=False):
    out = s.zeros(body.rows)
    for order in ORDERS:
        order = tuple(reversed(order)) if reverse else order
        matrices = [legs[j].H if reverse else legs[j] for j in order]
        for mask in range(16):
            left = s.eye(body.rows); right = s.eye(body.rows)
            for j, t in enumerate(matrices):
                if mask >> j & 1:
                    right = right*t
                else:
                    left = t*left
            out += (-1)**mask.bit_count()*left*body*right
    return out


def face_legs(blocks):
    result=[]
    for j,b in enumerate(blocks):
        t=s.zeros(16)
        t[4*j:4*j+4,4*((j+1)%4):4*((j+1)%4)+4]=b
        result.append(t)
    return result


def adjoint_control(profiles):
    c0=s.Rational(profiles[1]['paths'][0]['c0'])
    ci=s.Rational(profiles[1]['paths'][0]['ci'])
    onsite=s.diag(1,1,1,1,*([0]*12))
    diagonal=s.diag(3,-1,-1,-1)
    uniform=face_legs([diagonal]*4)
    uniform_image=(nested(uniform,onsite)+nested(uniform,onsite,True))/2
    hs=s.simplify(s.trace(uniform_image.H*uniform_image))
    assert hs==472608
    phi=[diagonal,s.Matrix([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]),
         s.Matrix([[0,0,0,0],[0,0,s.I,0],[0,-s.I,0,0],[0,0,0,0]]),
         s.Matrix([[0,0,0,0],[0,0,0,0],[0,0,0,2],[0,0,2,0]])]
    u=s.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,s.I,0],[0,0,0,-s.I]])
    v=s.Matrix([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    assert u.H*u==v.H*v==s.eye(4) and u.det()==v.det()==1 and u*v!=v*u
    def block(x,y,axis,link):
        a=phi[0]+s.I*(ci/c0)*phi[axis+1]
        return ((1+x)*a*link+(1+y)*link*a)/2
    blocks=[block(0,1,0,u),block(1,2,1,v),block(3,2,0,v*u).H,block(0,3,1,u*v).H]
    # At origin(N-1,0,0), both opposite x legs cross the native AP seam.
    seams=(-1,1,-1,1); phases=(s.I,-1,1,1)
    native=face_legs([b*seam*phase for b,seam,phase in zip(blocks,seams,phases)])
    direct=nested(native,onsite); adjoint=nested(native,onsite,True)
    assert direct==expanded(native,onsite) and adjoint==expanded(native,onsite,True)
    image=(direct+adjoint)/2
    assert s.trace(image)==0
    native_hs=s.simplify(s.trace(image.H*image))
    assert native_hs>0 and math.prod(seams)==1
    # Retain nontrivial closed Peierls phase: it is not an AP cancellation.
    assert image!=(nested(face_legs(blocks),onsite)+nested(face_legs(blocks),onsite,True))/2
    c=s.Rational(256,4*255)
    assert c*hs==s.Rational(10082304,85)
    return {'actual_M':256,'actual_k':128,'all24_orders':len(ORDERS),
            'all64_assignments_per_order':64**4,'independent_left_right_expansion_checked':True,
            'uniform_HS2_over_c0_pow8':str(hs),'uniform_full64_compression':str(c*hs),
            'noncommuting_wrapped_complex_native_HS2_over_c0_pow8':str(native_hs),
            'noncommuting_full64_compression':str(s.simplify(c*native_hs)),
            'AP_seams':list(seams),'Peierls_loop_phase':str(math.prod(phases)),
            'four_colour_Px_trace':s.trace(onsite),'original_c0':str(c0),
            'scope':'native point controls and general analytic intertwiner, not ground stratum'}


def domain_signed_control(profiles):
    m=s.symbols('m',positive=True)
    s0,cg,ss=s.symbols('S Cg ss',positive=True)
    # Constants/jets/tails scale8; degree and incident support do not change.
    beta=(1-17*ss)/4
    assert beta.subs(ss,s.Rational(1,34))>0 and beta.subs(ss,s.Rational(1,17))==0
    assert s.simplify((8*s0)**(1-ss)*(17*8*cg)**ss/(s0**(1-ss)*(17*cg)**ss))==8
    assert s.simplify((s0*m**s.Rational(-1,4)).subs(m,m**s.Rational(4,17)))==s0*m**s.Rational(-1,17)
    paths=profiles[1]['paths']
    assert len(paths)==6 and {(p['axis'],p['length']) for p in paths}==set(itertools.product(range(3),(1,2)))
    coeff=sum(abs(s.Rational(p['c0']))+abs(s.Rational(p['ci'])) for p in paths)
    # Ground-channel term can exactly cancel a signed commutator expectation.
    omega=s.Matrix([1,0,0]); p0=s.diag(1,1,0); q0=s.eye(3)-p0
    number=s.Matrix([[2,1,0],[1,2,0],[0,0,2]])
    a=s.Matrix([[0,s.I,0],[-s.I,0,0],[0,0,0]])
    signed=(omega.H*s.I*comm(number,a)*omega)[0]
    ground=(omega.H*s.I*(number*p0*a-a*p0*number)*omega)[0]
    assert signed==ground==2 and q0*a*omega==s.zeros(3,1)
    # Dirichlet double commutator identity and exact resolvent Cauchy on a
    # separate declared finite non-native method representation.
    l=s.diag(0,2,5); n=s.Matrix([[2,1,s.I],[1,2,0],[-s.I,0,2]])
    a=s.Matrix([[0,1+s.I,2],[1-s.I,0,0],[2,0,0]])
    v=n*omega-omega*(omega.H*n*omega)[0]; j=a*omega
    dn=(v.H*l*v)[0]
    assert dn==(omega.H*comm(n,comm(l,n))*omega)[0]/2==7
    eta=s.Rational(3,2); actual=(j.H*(l+eta*s.eye(3)).inv()*j)[0]
    lower=abs((v.H*j)[0])**2/(dn+eta*(v.H*v)[0])
    assert actual>=lower and (v.H*v)[0]<=16
    return {'norm_and_all_covariant_jet_factor':8,'local_incident_support_upper':52,
            'strict_domain':'0<s<1/17; endpoint has zero geometric exponent',
            'fractional_K_factor':8,'active_paths_in_Dn':6,
            'Dn_coefficient_times_C85_pow_2over17':str(8*coeff),
            'incident_paths_count':12,'current_link_support_sites':4,
            'signed_commutator':str(signed),'exact_ground_commutator':str(ground),
            'wrong_dropping_P0_predicts_positive':True,'actual_excited_norm':0,
            'finite_resolvent_cauchy_actual':str(actual),'finite_resolvent_cauchy_lower':str(lower),
            'domain_claim_conditional_on_inherited_same_L_core':True}


def complex_ground_controls():
    # Independently derive the parent's all-real-character discriminant.
    k=s.symbols('k',real=True); a=s.Integer(256); coupling=s.Integer(1024)
    difference=s.expand((coupling+a*k*k)**2-(coupling**2+4*a*a*k*k))
    assert difference==65536*k**4+262144*k**2
    ground=s.Matrix([1,1,0])/s.sqrt(2); other=s.Matrix([0,0,1]); excited=s.Matrix([1,-1,0])/s.sqrt(2)
    p=ground*ground.H+other*other.H; q=s.eye(3)-p; ec=s.diag(-4,4,0)
    b=s.zeros(3); b[2,0]=1; probe=b+b.H
    assert comm(ec,b)==4*b and p*ec*p==s.zeros(3)
    inverse=(ground.H*ec*(excited*excited.H/(2*coupling))*ec*ground)[0]
    assert inverse==s.Rational(1,128)<=s.Rational(1,64)
    assert q*probe*ground==s.zeros(3,1) and q*probe*other!=s.zeros(3,1)
    assert (ground.H*probe*ground)[0]==0 and (ground.H*probe**2*ground)[0]==s.Rational(1,2)
    vectors=[s.Matrix([1,s.I]),s.Matrix([s.I,-1])]
    gram=s.Matrix([[(v.H*w)[0] for w in vectors] for v in vectors])
    assert gram.det()==0 and cast(Any, s.re(gram)).det()==4
    return {'parent_countercontrol_rederived_not_rerun':True,'all_real_character_discriminant':str(difference),
            'closed_E_inverse':str(inverse),'inverse_upper':'1/64',
            'A_selected_ground_excited_Gram':0,'A_total_Gram':'1/2','other_ground_A_excitation':True,
            'complex_rank_deficient_Gram':str(gram),'wrong_real_only_determinant':4,
            'ground_backend':False,'mean_subtraction_invalid':True}


def immutable_helpers(archive):
    raw=archive.read('native_loop_dark_consumer_20261012.py')
    tree=ast.parse(raw)
    # Select pure helper definitions/imports only; omit main and top-level calls.
    nodes: list[ast.stmt]=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)) and not (isinstance(node,ast.FunctionDef) and node.name=='main')]
    nodes += [node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ORDERS' for t in node.targets)]
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    namespace: dict[str, Any]={'__name__':'immutable_scientific_helpers','__file__':str(ROOT/'native_loop_dark_consumer_20261012.py')}
    exec(compile(module,'immutable_scientific_helpers','exec'),namespace)
    return namespace


def polarization_controls(archive):
    helpers=immutable_helpers(archive)
    # Independent direct canonical trace positive control for the determinant
    # helper, using an actual full 8-mode/4-particle exterior matrix.
    body=s.zeros(8); body[4:8,0:4]=s.diag(3,-1,-1,-1)
    acted,_=helpers['wedge_action'](body,4)
    exact=s.trace(acted*acted.H)/math.comb(8,4)
    slot=helpers['one_species_moments'](((1,0),(0,1)),8)[-1]
    assert Fraction(int(s.numer(exact)),int(s.denom(exact)))==slot
    prior=json.loads(archive.read('native_loop_dark_polarization_actual1_20261012_receipt.json'))
    rows=[]
    for row in prior['rows']:
        n=row['N']; modes=4*n**3; root=s.exp(2*s.pi*s.I/n)
        first=helpers['face']((0,0,0),(1,2),n)
        # NEW exact native evaluations; combine duplicate displacement patterns
        # before expensive word traces. No prior diagonal/cross result is input.
        entries=[]
        for axes in ((1,2),(2,0)):
            patterns={}
            for d in itertools.product(range(n),repeat=3):
                pattern=helpers['covariance_pattern'](first,helpers['face'](d,axes,n))
                patterns[pattern]=patterns.get(pattern,s.S.Zero)+root**d[2]
            value=s.S.Zero
            for pattern,weight in patterns.items():
                weight=s.simplify(s.expand_complex(weight))
                if weight==0:continue
                coefficient=helpers['face_covariance'](*pattern,modes)
                value+=s.Rational(coefficient.numerator,coefficient.denominator)*weight
            entries.append(s.simplify(s.expand_complex(value)))
        diagonal,cross=entries
        gram: Any=s.Matrix([[diagonal,cross],[s.conjugate(cross),diagonal]])
        prior_gram: Any=cast(Any, s.sympify)(row['conditional_full_complex_polarization_Gram_over_native_c0_pow8'],locals={'Matrix':s.Matrix})
        assert (gram-prior_gram).applyfunc(s.simplify)==s.zeros(2)
        assert diagonal>0 and s.simplify(gram.det())>0
        rows.append({'N':n,'M':modes,'k':modes//2,'V':n**3,'V_inverse_sqrt_source':True,
                     'complex_Gram':str(gram),'rank':2,'original_receipt_agrees':True,
                     'native_ground_spectrum_or_IR':False})
    return {'rows':rows,'independent_full_exterior_trace_positive_control':str(exact),
            'all64_separately_canonical_factors':True,'neutral_and_Ref_identity_retained':True,
            'rank2_scope':'normalized finite conditional equal-time covariance only',
            'no_finite_P0_transport':True,'old_main_executed':False}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--request',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); request=json.loads(args.request.read_bytes())
    pre=ROOT/(request['epoch_prefix']+'_preexecution.zip')
    with zipfile.ZipFile(pre) as sealed:
        snapshot_raw=sealed.read(request['snapshot'])
        assert hashlib.sha256(snapshot_raw).hexdigest()==request['pins'][request['snapshot']]
        import io
        with zipfile.ZipFile(io.BytesIO(snapshot_raw)) as frozen:
            for name,digest in request['snapshot_pins'].items():
                assert hashlib.sha256(frozen.read(name)).hexdigest()==digest
            with zipfile.ZipFile(io.BytesIO(sealed.read(request['profile_archive']))) as profile_archive:
                raw=profile_archive.read(request['profile_member'])
                assert hashlib.sha256(raw).hexdigest()==request['profile_member_sha256']
                profiles=json.loads(raw)['primitive_profiles']
            assert [p['load'] for p in profiles]==[1,1,63]
            assert profiles[1]['paths']==profiles[2]['paths']
            result={'kind':'INDEPENDENT_EXACT_AUXILIARY_ADJOINT_REVIEW_NOT_NATIVE_GROUND',
                    'representation':representation_control(),'adjoint_native':adjoint_control(profiles),
                    'domain_signed':domain_signed_control(profiles),'complex_ground':complex_ground_controls(),
                    'conditional_polarization':polarization_controls(frozen),
                    'governor':request['governor'],'scientific_admission':False,'campaign_complete':False}
    with args.output.open('x') as stream: json.dump(result,stream,indent=2,default=str); stream.write('\n')
    print(json.dumps({'receipt':args.output.name,'native_exact_and_complex_controls_completed':True,'actual_ground_transfer':'UNPAID'}))


if __name__=='__main__':
    main()
