"""Independent exact coefficient and non-native ordered-word adverse controls."""
from __future__ import annotations
import hashlib
import itertools
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from fractions import Fraction
import sympy as s

ROOT = Path(__file__).resolve().parent

def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    snapshot = (ROOT / request['snapshot']).read_bytes()
    assert digest(snapshot) == request['snapshot_sha256']
    outer = zipfile.ZipFile(BytesIO(snapshot))
    before = {name: digest(outer.read(name)) for name in request['target_pins']}
    assert before == request['target_pins']
    pre = zipfile.ZipFile(BytesIO(outer.read('native_history_ground_actual2_20261011_preexecution.zip')))
    old_request = json.loads(pre.read('native_history_ground_actual2_20261011_request.json'))
    epoch = zipfile.ZipFile(BytesIO(pre.read(old_request['epoch_archive'])))
    raw = epoch.read(old_request['profile_member'])
    profiles = json.loads(raw)['primitive_profiles']
    weights = [abs(Fraction(p['onsite'])) + 2*sum((abs(Fraction(r['c0'])) + abs(Fraction(r['ci'])) for r in p['paths']), Fraction()) for p in profiles]
    lam = 2**8192
    b = 4*(lam*weights[0] + weights[1] + 63*weights[2])
    assert b < 512*lam and sum(p['load'] for p in profiles) == 65
    assert b != 4*(lam*weights[0]+weights[1]+weights[2])
    # Independent full three-dimensional grounded solve: occupation transfer is -1.
    sites = list(itertools.product(range(4), repeat=3)); index = {x:i for i,x in enumerate(sites)}
    edges = [(x,a) for x in sites for a in range(3)]
    def next_site(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        return tuple((v+1)%4 if j==axis else v for j,v in enumerate(x))
    d = s.zeros(64,192)
    for i,(x,a) in enumerate(edges):
        d[index[x],i]=1; d[index[next_site(x,a)],i]=-1
    o=(0,0,0); neighbour=(1,0,0); literal=edges.index((o,0))
    transfer=s.zeros(192,1); transfer[literal]=-1
    drho=d*transfer
    potential=s.zeros(64,1)
    potential[:63,0]=(d*d.T)[:63,:63].inv(method='DM')*drho[:63,:]
    longitudinal=d.T*potential; transverse=transfer-longitudinal
    assert d*transverse==s.zeros(64,1)
    assert longitudinal[literal]==-s.Rational(21,64) and transverse[literal]==-s.Rational(43,64)
    exterior=[i for i in range(192) if i!=literal and longitudinal[i]!=0]
    assert exterior and all(transverse[i]+longitudinal[i]==0 for i in exterior)
    # Exact native theta=2^(-distance/4), compare in Q[z]/(2z^4-1).
    z=s.Symbol('z'); theta=[z**sum(min(v,4-v) for v in x) for x,_ in edges]
    full=sum(theta[i]*(transverse[i]+longitudinal[i])**2 for i in range(192))
    frozen=sum(theta[i]*transverse[i]**2 for i in range(192))
    assert full==1 and s.rem(s.expand(frozen-full),2*z**4-1,z)!=0
    k=s.Matrix([64*(-1)**sum(x) if a==0 else 0 for x,a in edges])
    rho=s.Matrix([128*(-1)**sum(x) for x in sites]); assert d*k==rho
    occupations=[4 if sum(x)%2==0 else 0 for x in sites]
    assert sum(occupations)==128 and occupations[index[o]]==4 and occupations[index[neighbour]]==0
    moved=occupations[:]; moved[index[o]]-=1; moved[index[neighbour]]+=1
    assert sum(moved)==128 and s.Matrix([a-b for a,b in zip(moved,occupations)])==drho
    delta=64*sum(theta[i]*(2*k[i]*transfer[i]+transfer[i]**2) for i in range(192))
    no_cross=64*sum(theta[i]*transfer[i]**2 for i in range(192))
    assert delta==-8128 and no_cross==64 and d*(k+transfer)==rho+drho
    # Prior +1 moves from empty neighbour to filled origin, so is forbidden.
    assert occupations[index[neighbour]]==0 and occupations[index[o]]==4
    # Generic independent field coordinates, ordered Haar position, all literal profiles.
    p,q,a,u=s.symbols('p q a u',real=True)
    t=s.diag(s.Rational(1,2),-s.Rational(1,2),0,0)
    x=s.Matrix([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])/2
    link=s.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    second=s.diag(s.I,-s.I,1,1)
    jet_rows=0
    for profile in profiles:
        scale=-s.Integer(lam) if profile['scale']=='heavy' else s.Integer(1)
        for row in profile['paths']:
            c0,ci=s.Rational(row['c0']),s.Rational(row['ci'])
            fx=c0*p*t+s.I*ci*q*x; fy=c0*(q+1)*x+s.I*ci*(p-1)*t
            path=link if row['length']==1 else link*second
            phase=s.exp(-s.I*a*row['length']) if profile['scale']=='charged' else s.Integer(1)
            tensor=scale*phase*(fx*path+path*fy)/2
            envelope=s.exp(-(p*p+q*q)/2)
            product=envelope*tensor
            assert s.simplify(s.diff(product,p,2)-(s.diff(envelope,p,2)*tensor+2*s.diff(envelope,p)*s.diff(tensor,p)))==s.zeros(4)
            if profile['scale']=='charged':
                assert s.simplify(s.diff(tensor,a,2)+row['length']**2*tensor)==s.zeros(4)
            if row['length']==2:
                first=s.I*t*link*second; last=s.I*link*t*second
                assert first!=last and path!=second*link
            jet_rows+=1
    # Products and adjoints are checked with a new matrix triple and Weyl phase.
    aa=s.Matrix([[0,1,0],[0,0,2],[0,0,0]])
    bb=aa.T; cc=s.diag(2,3,5)
    assert aa*bb!=bb*aa and (aa*bb).T==bb.T*aa.T
    shifted=s.exp(s.I*(a+u))*bb
    correct=s.exp(-s.I*a)*aa*shifted
    omitted=s.exp(-s.I*a)*aa*(s.exp(s.I*a)*bb)
    assert s.simplify(correct-s.exp(s.I*u)*omitted)==s.zeros(3)
    assert s.simplify(correct.subs(u,s.pi)-omitted)==-2*aa*bb
    # NEW NON-NATIVE method matrices: each finite word has actual right-to-left order.
    l=s.diag(0,2,5); qmat=s.diag(1,3,7); omega=s.Matrix([1,0,0])
    letters=[aa,bb,cc]; bounds=[5,5,38]; words=0
    for length in range(5):
        for ids in itertools.product(range(3),repeat=length):
            vector=omega; n2=s.Integer(1); energy=s.Integer(1); g2=s.Integer(0)
            for j in ids:
                letter=letters[j]; phase=s.diag(1,(-1)**j,s.I**j)
                evolved=phase*letter*phase.conjugate().T
                comm=l*letter-letter*l
                c2=sum(abs(v)**2 for v in comm)
                # Q>=I, Q<=7I, exact Frobenius norm upper bounds.
                g2=2*bounds[j]*g2+2*c2*energy
                energy=7*bounds[j]*energy; n2=bounds[j]*n2
                vector=evolved*vector
            assert (vector.conjugate().T*vector)[0]<=n2
            assert (vector.conjugate().T*qmat*vector)[0]<=energy
            assert ((l*vector).conjugate().T*(l*vector))[0]<=g2
            words+=1
    # Omission of LA Omega fails on a transported word, even when initial Omega is ground.
    right=bb*omega
    assert l*right!=s.zeros(3,1)
    left=cc
    actual=l*left*right; omitted=(l*left-left*l)*right
    assert actual!=omitted and actual-omitted==left*l*right
    # Packet leading U(t1) cannot be dropped.
    packet=s.Matrix([1,1,0])/s.sqrt(2); phase=s.diag(1,-1,1)
    packet_form=(packet.T*qmat*packet)[0]
    packet_graph2=((l*packet).T*(l*packet))[0]
    assert packet_form==2 and packet_graph2==2
    transported=phase*packet
    assert (transported.T*qmat*transported)[0]==packet_form
    evolved=phase.conjugate().T*bb*phase
    correct_word=evolved*packet
    factored_word=phase.conjugate().T*bb*transported
    wrong_word=phase.conjugate().T*bb*packet
    assert correct_word==factored_word and correct_word!=wrong_word
    actual_packet_form=(correct_word.conjugate().T*qmat*correct_word)[0]
    actual_packet_graph2=((l*correct_word).conjugate().T*(l*correct_word))[0]
    assert actual_packet_form==s.Rational(31,2) and actual_packet_graph2==52
    comm=l*bb-bb*l
    comm2=sum(abs(v)**2 for v in comm)
    assert actual_packet_form<=7*bounds[1]*packet_form
    assert actual_packet_graph2<=2*bounds[1]*packet_graph2+2*comm2*packet_form
    receipt={'kind':'INDEPENDENT_NATIVE_COEFFICIENT_AND_NON_NATIVE_WORD_CONTROLS','target_hashes':before,
             'profile_sha256':digest(raw),'B_sha256':digest(str(b).encode()),'weights':[str(v) for v in weights],
             'native_delta_fullE':str(delta),'omitted_cross_delta':str(no_cross),
             'native_tL':str(longitudinal[literal]),'native_tT':str(transverse[literal]),
             'exterior_cancelled_edges':len(exterior),'field_Haar_fullE_profile_rows':jet_rows,
             'packet_form':str(packet_form),'packet_initial_graph2':str(packet_graph2),
             'packet_correct_word_form':str(actual_packet_form),
             'packet_correct_word_graph2':str(actual_packet_graph2),
             'new_method_words':words,'adverse_rejections':['occupation_frozen','weighted_cross_omitted','prior_plus1_canonical_transfer','Haar_position_reversed','Weyl_shift_omitted','matrix_order_reversed','LAOmega_omitted','packet_initial_propagator_omitted'],
             'native_ground_or_fullH_evolution':False,'scientific_admission':False}
    Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'new_method_words':words,'native_rows':jet_rows,'adverse_cases':8,'ground_computed':False}))

if __name__=='__main__':
    main()
