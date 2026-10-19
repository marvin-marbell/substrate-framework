"""Independent exact algebra/domain controls, never a native ground solver."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import cast
import zipfile
import sympy as s

ROOT = Path(__file__).resolve().parent

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def eq(a, b) -> bool:
    if isinstance(a, s.MatrixBase):
        return all(s.simplify(v) == 0 for v in a - b)
    return s.simplify(a - b) == 0

def record(m):
    return [[str(s.simplify(m[i,j])) for j in range(m.cols)] for i in range(m.rows)]

def spectral_controls():
    # Two stationary directions and three positive directions; no eigensolver.
    L = s.diag(0, 0, 1, 3, 7)
    omega = s.eye(5)[:,0]
    P0 = s.diag(1,1,0,0,0)
    Q = s.eye(5)-P0
    A = s.Matrix([[2,1,1,s.I,0],[1,-1,0,0,0],[1,0,3,1,0],[-s.I,0,1,-2,s.I],[0,0,0,-s.I,1]])
    B = s.Matrix([[-1,2*s.I,-2-s.I,0,1],[-2*s.I,0,0,0,0],[-2+s.I,0,1,s.I,2],[0,0,-s.I,2,0],[1,0,2,0,-3]])
    assert A == A.H and B == B.H
    eta=s.Rational(2,5)
    R=Q*(L+eta*s.eye(5)).inv()*Q
    X=s.Matrix.hstack(Q*A*omega,Q*B*omega)
    G=X.H*X
    Y=R*X
    T=X.H*R*X; U=Y.H*Y; K=Y.H*L*Y
    assert eq(K,T-eta*U)
    z=s.Matrix([s.Rational(1,4)+s.I/3,-s.Rational(2,5)+s.I/7])
    raw=omega+z[0]*A*omega+z[1]*B*omega
    V=s.Matrix.hstack(A*omega,B*omega)
    mean=s.Matrix([2,-1])
    norm=1+2*s.re((mean.T*z)[0])+(z.H*(V.H*V)*z)[0]
    assert eq((raw.H*raw)[0],norm)
    assert eq((raw.H*L*raw)[0],(z.H*(V.H*L*V)*z)[0])
    assert not eq((z.H*G*z)[0],(z.H*s.re(G)*z)[0])
    assert not eq(Q*A*omega,(s.eye(5)-omega*omega.H)*A*omega)
    t=s.Matrix([s.Rational(1,3)+s.I/5,-s.Rational(1,4)+s.I/6])
    S=s.Rational(3,7)*A-s.Rational(2,9)*B
    psi=omega+Y*t
    den=1+(t.H*U*t)[0]
    a=(omega.H*S*omega)[0]
    linear=(t.H*Y.H*S*omega)[0]
    numerator=(t.H*K*t)[0]-a-2*s.re(linear)-(t.H*Y.H*S*Y*t)[0]
    direct=(psi.H*(L-S)*psi)[0]/(psi.H*psi)[0]
    assert eq(direct,numerator/den)
    assert not eq(direct,(t.H*K*t)[0]/den-a-2*s.re(linear)-(t.H*Y.H*S*Y*t)[0])
    j=s.symbols('j', real=True)
    y=Y[:,0]; u=U[0,0]; k=K[0,0]; aa=(omega.H*A*omega)[0]
    cubic=(y.H*A*y)[0]
    trial=omega+j*y
    exact=(trial.H*(L-j*A)*trial)[0]/(trial.H*trial)[0]
    formula=(j*j*k-j*aa-2*j*j*T[0,0]-j**3*cubic)/(1+j*j*u)
    assert eq(exact,formula) and cubic != 0
    assert not eq(exact,(j*j*k-j*aa-2*j*j*T[0,0])/(1+j*j*u))
    assert not eq(exact,j*j*k-j*aa-2*j*j*T[0,0]-j**3*cubic)
    M=X.H*(L+eta*s.eye(5))*X
    lower=G*M.inv()*G
    diff=s.simplify(T-lower)
    assert diff[0,0]>=0 and diff[1,1]>=0 and s.simplify(diff.det())>=0
    assert s.re(T[0,1])<0 and s.im(T[0,1])!=0 and s.simplify(T.det())>0
    # Singular support with a deliberately redundant complex column.
    D=s.Matrix([[1,0,s.I],[0,1,2]])
    Xs=X*D; Gs=Xs.H*Xs; Ms=Xs.H*(L+eta*s.eye(5))*Xs
    Ts=Xs.H*R*Xs; plus=Ms.pinv()
    ds=s.simplify(Ts-Gs*plus*Gs)
    assert Ms.det()==0 and eq(Ms*plus*Ms,Ms)
    for subset in itertools.chain(itertools.combinations(range(3),1),itertools.combinations(range(3),2),[tuple(range(3))]):
        assert s.simplify(ds.extract(subset,subset).det())>=0
    null=s.Matrix([-s.I,-2,1]); assert eq(Xs*null,s.zeros(5,1)) and eq(Gs*null,s.zeros(3,1))
    # Dropping range compatibility makes the purported variational supremum infinite.
    incompatible=s.diag(1,0); wrongG=s.eye(2); v=s.Matrix([0,j])
    invalid_sup=2*(s.Matrix([0,1]).T*wrongG*v)[0]-(v.T*incompatible*v)[0]
    assert invalid_sup==2*j
    return {'scope':'Finite exact method control, NOT native ground or native H','wholeP0_rank':2,
            'G':record(G),'T':record(T),'M':record(M),'T_minus_supported_lower':record(diff),
            'complex_negative_cross':str(T[0,1]),'singular_rank':Ms.rank(),'singular_support_lower':record(Gs*plus*Gs),
            'general_complex_source_energy':str(s.simplify(direct)), 'single_source_energy':str(s.factor(formula)),
            'signed_mean':str(aa),'nonzero_cubic':str(cubic),
            'rejected':['mean-only P0','real-only complex cross','source mean outside denominator','cubic deletion','denominator deletion','singular ordinary inverse','pseudoinverse without kernel/range compatibility']}

def generators():
    result=[]
    for a in range(4):
        for b in range(a+1,4):
            t=s.zeros(4); t[a,b]=t[b,a]=s.Rational(1,2); result.append(t)
            t=s.zeros(4); t[a,b]=-s.I/2; t[b,a]=s.I/2; result.append(t)
    for n in range(1,4):
        result.append(s.diag(*([1]*n+[-n]+[0]*(3-n)))/s.sqrt(2*n*(n+1)))
    assert len(result)==15 and eq(sum((t*t for t in result),s.zeros(4)),15*s.eye(4)/8)
    return result

def primitive_controls(profiles):
    ts=generators(); U=s.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]])
    V=s.diag(s.I,-s.I,1,1)
    q=s.symbols('q',real=True); checks=0; endpoint_checks=0
    assert [len(p['paths']) for p in profiles]==[3,6,6] and [p['load'] for p in profiles]==[1,1,63]
    assert len({p['onsite'] for p in profiles})==3
    roles=[{'role':i,'profile':0 if i==0 else 1 if i==1 else 2,'scale':'-Lambda' if i==0 else '1'} for i in range(65)]
    for profile in profiles:
        for path in profile['paths']:
            c0=s.Rational(path['c0']); ci=s.I*s.Rational(path['ci'])
            Fx=c0*ts[0]+ci*ts[3]; Fy=c0*ts[2]+ci*ts[5]
            factors=[U] if path['length']==1 else [U,V]
            W=U if len(factors)==1 else U*V
            leg=(Fx*W+W*Fy)/2
            for mu in range(4):
                coefficient=c0 if mu==0 else ci if mu==path['axis']+1 else 0
                for t in ts:
                    for endpoint in (0,1):
                        dx=coefficient*t if endpoint==0 else s.zeros(4)
                        dy=coefficient*t if endpoint==1 else s.zeros(4)
                        varied=((Fx+q*dx)*W+W*(Fy+q*dy))/2
                        assert eq(varied.diff(q),(dx*W+W*dy)/2) and eq(varied.diff(q,2),s.zeros(4))
                        endpoint_checks+=1
            for slot in range(len(factors)):
                for t in ts:
                    jet=s.eye(4)+q*s.I*t-q*q*t*t/2
                    variedW=(jet*U if slot==0 else U*jet*V) if len(factors)==1 or slot==1 else jet*U*V
                    dW=(s.I*t*U if slot==0 else U*s.I*t*V) if len(factors)==1 or slot==1 else s.I*t*U*V
                    ddW=(-t*t*U if slot==0 else -U*t*t*V) if len(factors)==1 or slot==1 else -t*t*U*V
                    varied=(Fx*variedW+variedW*Fy)/2
                    d=(Fx*dW+dW*Fy)/2; dd=(Fx*ddW+ddW*Fy)/2
                    assert eq(varied.diff(q).subs(q,0),d) and eq(varied.diff(q,2).subs(q,0),dd)
                    assert eq(varied.H.diff(q).subs(q,0),d.H)
                    assert eq(varied.H.diff(q,2).subs(q,0),dd.H)
                    checks+=1
            assert eq((-leg).H,-leg.H) # wrapped AP cocycle retains directed-adjoint binding
    assert not eq(U*ts[0]*V,ts[0]*U*V)
    x=s.symbols('x', real=True)
    X=s.Matrix([[x,s.I*x*x],[-s.I*x*x,-x]])
    connection=s.I*s.Matrix([[0,1],[1,0]])
    f=s.Matrix([1+x*x,s.I*x])
    def D(v): return v.diff(x)+connection*v
    dx=X.diff(x)+connection*X-X*connection
    dd=dx.diff(x)+connection*dx-dx*connection
    lhs=-D(D(X*f))+X*D(D(f)); rhs=-dd*f-2*dx*D(f)
    assert eq(lhs,rhs) and not eq(lhs,-dd*f)
    assert not eq(lhs,-X.diff(x,2)*f-2*X.diff(x)*f.diff(x))
    # Neutral commutes charged A, but does not commute neutral occupation projector.
    P=s.diag(1,0); hn=s.Matrix([[0,1],[1,0]]); I=s.eye(2)
    nc=s.kronecker_product(hn,I); ac=s.kronecker_product(I,hn); pn=s.kronecker_product(P,I)
    assert eq(nc*ac-ac*nc,s.zeros(4)) and not eq(nc*pn-pn*nc,s.zeros(4))
    # Local total number commutes onsite colour; intercell transfers do not.
    occupation=s.diag(1,1,1,1,0,0,0,0)
    onsite=s.diag(*([1,2,3,4]*2)); hopping=s.zeros(8); hopping[0,4]=hopping[4,0]=1
    assert eq(onsite*occupation-occupation*onsite,s.zeros(8))
    assert not eq(hopping*occupation-occupation*hopping,s.zeros(8))
    assert not eq((onsite+hopping)*occupation-occupation*(onsite+hopping),s.zeros(8))
    # Four ordered slots, second derivatives and cross insertions, all 24 words.
    legs=[U,V,U*V,V*U]; first=ts[:4]; second=[-t*t for t in first]
    cross_rejections=0
    for order in itertools.permutations(range(4)):
        product=s.eye(4)
        for e in order: product=(legs[e]+q*first[e]+q*q*second[e]/2)*product
        d=s.zeros(4); dd=s.zeros(4); diagdd=s.zeros(4)
        for slot in range(4):
            v=s.eye(4); w=s.eye(4)
            for pos,e in enumerate(order):
                v=(first[e] if pos==slot else legs[e])*v
                w=(second[e] if pos==slot else legs[e])*w
            d+=v; dd+=w; diagdd+=w
        for a,b in itertools.combinations(range(4),2):
            v=s.eye(4)
            for pos,e in enumerate(order):v=(first[e] if pos in (a,b) else legs[e])*v
            dd+=2*v
        assert eq(product.diff(q).subs(q,0),d) and eq(product.diff(q,2).subs(q,0),dd)
        if not eq(dd,diagdd):cross_rejections+=1
    assert cross_rejections>0
    return {'literal_profile_path_count':15,'distinct_canonical_roles':roles,'all60_Phi_both_endpoints_checks':endpoint_checks,
            'all15_Haar_each_path_position_and_adjoint_checks':checks,'word_orders':24,'cross_second_rejections':cross_rejections,
            'expanded_word_count':24*64**4,'rejected':['Berry removed','right derivative removed','internal Haar insertion moved','cross second jets deleted','neutral projector commutator deleted','local occupation diagonalization','AP adjoint mismatch'],
            'scope':'Exact primitive matrix/connection examples and symbolic identities, not physical quantum-ground evaluation'}

def geometry_ward_controls():
    vertices=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),(2,0,0),(2,1,0)]
    K=set(vertices)
    for v in vertices:
        for axis in range(3):
            for offset in (-2,-1,1,2):
                z=list(v);z[axis]+=offset;K.add(cast(tuple[int, int, int], tuple(z)))
    radius=max(sum(abs(a) for a in z) for z in K)
    assert len(K)<=78 and radius==5
    assert s.Rational(1,4)**4 < s.Rational(1,2)**5
    ca=s.Matrix([1,1,1,1,0,0,0]);cb=s.Matrix([0,-1,0,0,1,1,1]);C=s.Matrix.hstack(ca,cb)
    edges=[(0,1),(1,2),(2,3),(3,0),(1,4),(4,5),(5,2)]
    incidence=s.zeros(6,7)
    for k,(a,b) in enumerate(edges):incidence[a,k]=-1;incidence[b,k]=1
    assert eq(incidence*C,s.zeros(6,2))
    J=C.T*C;assert J==s.Matrix([[4,-1],[-1,4]])
    # Longitudinal plus transverse must recombine before closing.
    D=s.Matrix([[-1,1]]);PL=D.T*(D*D.T).inv()*D;PT=s.eye(2)-PL;openpath=s.Matrix([1,0])
    assert eq(PT*openpath+PL*openpath,openpath) and not eq(PT*openpath,openpath)
    # Electric right action on a Fourier word shifts every raw E by its incidence.
    momenta=s.symbols('p0:7',real=True)
    shift=sum((momenta[i]+ca[i])**2-momenta[i]**2 for i in range(7))
    assert eq(shift,4+2*sum(ca[i]*momenta[i] for i in range(7))) and not eq(shift,4)
    mu=s.diag(2,-3);u=s.Matrix([s.Rational(1,3),s.Rational(2,5)])
    v=C*mu*u;h=v; B0=s.Rational(160)
    optimized=(h.T*v)[0]**2/(4*(B0/32)*(h.T*h)[0])
    assert eq(optimized,(8/B0)*(u.T*mu*J*mu*u)[0])
    assert not eq(mu*J*mu,s.diag(2,3)*J*s.diag(2,3))
    assert eq(s.zeros(2)*J*s.zeros(2),s.zeros(2))
    # Real-direction PSD does not lift to complex Hermitian order.
    GA=s.Matrix([[1,s.I],[-s.I,1]]);W=s.eye(2)
    assert s.re(GA)==W and GA.det()==0
    complexu=s.Matrix([1,s.I]);assert (complexu.H*GA*complexu)[0]==0 and (complexu.H*W*complexu)[0]==2
    # all-Q radial extraction with nonnormalized vector and arbitrary radii.
    m=78;lamtau=s.Rational(16);norm2=s.Rational(9)
    radii=[s.Rational(i%5) for i in range(m)]
    radial4=norm2*sum(r**4 for r in radii)
    qQ=norm2+lamtau*radial4/2
    lhs=norm2*sum(radii)**2
    # Squared inequality avoids irrational comparison while preserving exact homogeneity.
    assert lhs**2 <= m**3*(2/lamtau)*qQ**2
    scaled=s.Rational(7,3);assert (scaled**2*lhs)**2 <= m**3*(2/lamtau)*(scaled**2*qQ)**2
    return {'incident_union_count':len(K),'incident_radius':radius,'safe_tau':'1/4','J':record(J),'J_eigenvalues':[3,5],
            'signed_Ward_matrix':record(mu*J*mu),'optimized_h':record(h),'Ward_coefficient':'8/B0','real_not_complex_counter':record(GA),
            'wholeP0_E_zero':'Inherited source-specific global-minimum closed-electric energy-Riesz premise, not finiteP0 limit',
            'all_Q_radial_inequality':'||sum_K r_z psi|| <= m^(3/4)*(2/(Lambda*tau))^(1/4)*sqrt(q_Q(psi))',
            'rejected':['theta=1/2 for twoface radius5','fullE transverse-only on open leg','electric right action omitted','absolute ground means substituted','real lower promoted to complex matrix','zero means promoted positive','normalized-only all-Q extraction']}

def domain_controls():
    x=s.symbols('x',real=True)
    # Bounded gauge-invariant is not sufficient: sharp scalar projector creates a jump.
    # Weak derivative of 1_(x>0) has delta mass, which is not an L2 weak derivative.
    jump=s.diff(s.Heaviside(x),x)
    assert jump==s.DiracDelta(x)
    # Resolvent smoothing requires Hilbert source only; uncut source need not have energy form.
    n=s.symbols('n',integer=True,positive=True);eta=s.symbols('eta',positive=True)
    assert s.simplify(n*(1/(n+eta)) - (1-eta/(n+eta)))==0
    # X_n=1/n, L_n=n: norm sum1/n^2 finite; energy sum1/n divergent; graph sum1 divergent.
    # R_eta X has L image n/(n+eta)*1/n <=1/n, genuinely D(L).
    return {'bounded_projector_counter':str(jump),'scope':'Native gauge-invariant sharp Phi-radial projector has the same jump obstruction on a smooth nonzero sphere section; not an actualground counterstate',
            'uncut_Hilbert_not_form_sequence':'L_n=n, X_n=1/n; sum|X|^2 converges but sum n|X|^2 diverges',
            'resolvent_domain_identity':'L R_eta=1-eta R_eta; graph bounded on Hilbert source',
            'fractional_endpoint_counter':'E_n=2^n, p_n=2^(-2n/17)/n: tail <=C T^(-2/17), endpoint moment sum E_n^(2/17)p_n=sum1/n diverges',
            'rejected':['any bounded gauge-invariant projector preserves form/domain','Hilbert or strict fractional source implies uncut form/D(L)','resolvent domain requires uncut D(L)','strict endpoint extended to equality','uncut source-energy cubic without A-domain payment'],
            'no_native_domain_nonexistence_or_no_go':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();req=json.loads(args.request.read_bytes())
    captured={name:(ROOT/name).read_bytes() for name in req['pins']}
    assert all(sha(raw)==req['pins'][name] for name,raw in captured.items())
    with zipfile.ZipFile(ROOT/req['profile_archive']) as z:raw=z.read(req['profile_member'])
    assert sha(raw)==req['profile_member_sha256']
    profiles=json.loads(raw)['primitive_profiles']
    result={'kind':'INDEPENDENT_CARRIER_ACTION_WARD_EXACT_CONTROLS_NOT_ADMISSION','spectral':spectral_controls(),
            'primitives':primitive_controls(profiles),'geometry_ward':geometry_ward_controls(),'domain':domain_controls(),
            'input_pins':req['pins'],'source_selected':True,'native_ground_computed':False,'physical1_positive_lower':False,
            'scientific_admission':False,'campaign_complete':False}
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    result['inputs_unchanged']=True
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'independent_controls_complete':True,'inputs_unchanged':True,'native_ground_computed':False}))

if __name__=='__main__':
    main()
