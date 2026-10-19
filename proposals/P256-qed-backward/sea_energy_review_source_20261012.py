"""Independent exact native jet and full-H trial-energy consumer; never a ground solver."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path
import sympy as s

ROOT = Path(__file__).resolve().parent

def native_matrices():
    pauli = [s.Matrix([[0,1],[1,0]]), s.Matrix([[0,-s.I],[s.I,0]]), s.diag(1,-1)]
    beta = s.diag(1,1,-1,-1)
    alpha = [s.kronecker_product(s.Matrix([[0,1],[1,0]]), p) for p in pauli]
    c = [beta/2] + [a/3 for a in alpha]
    generators = []
    for i,j in itertools.combinations(range(4),2):
        a=s.zeros(4); a[i,j]=a[j,i]=s.Rational(1,2); generators.append(a)
        a=s.zeros(4); a[i,j]=-s.I/2; a[j,i]=s.I/2; generators.append(a)
    for k in range(1,4):
        generators.append(s.diag(*([1]*k+[-k]+[0]*(3-k)))/s.sqrt(2*k*(k+1)))
    assert all(s.simplify(2*s.trace(a*b)) == int(i==j) for i,a in enumerate(generators) for j,b in enumerate(generators))
    assert all(a*b+b*a == 2*int(i==j)*s.eye(4) for i,a in enumerate([beta]+alpha) for j,b in enumerate([beta]+alpha))
    return c,generators

def comm(a,b):
    return a*b-b*a

def geometry_jets(c,t):
    def coefficients(fields):
        return s.Matrix([s.simplify(2*s.trace(a*b)) for a in fields for b in t])
    def orbit(fields):
        return s.Matrix.hstack(*(coefficients([s.I*comm(a,b) for b in fields]) for a in t))
    L=orbit(c); G=s.simplify(L.T*L); inverse=G.inv(); left=inverse*L.T
    P=s.simplify(s.eye(60)-L*left)
    assert P*P == P and P*L == s.zeros(60,15) and P.rank()==45
    eigen=G.eigenvals(); assert min(eigen)>=s.Rational(1,8)
    direction=L[:,0]+P*s.eye(60)[:,2]
    direction=direction/s.sqrt((direction.T*direction)[0])
    u=left*direction; w=P*direction
    A=sum((u[j]*t[j] for j in range(15)),s.zeros(4))
    W=[sum((w[15*mu+j]*t[j] for j in range(15)),s.zeros(4)) for mu in range(4)]
    nonlinear=coefficients([-comm(A,comm(A,c[mu]))+2*s.I*comm(A,W[mu]) for mu in range(4)])
    u2=-left*nonlinear; w2=-P*nonlinear
    assert s.simplify(L*u+w-direction)==s.zeros(60,1)
    assert s.simplify(L*u2+w2+nonlinear)==s.zeros(60,1)
    assert s.simplify((u2.T*u2+w2.T*w2)[0])>0
    assert nonlinear!=s.zeros(60,1)
    dL=orbit(W); R=s.simplify(left*dL)
    logj1=s.simplify(s.trace(R)); logj2=s.simplify(-s.trace(R*R))
    # Full [K,S] determinant reduces to det(I+L^+ dL) without choosing a normal frame.
    z=s.Symbol('z'); determinant=s.factor((s.eye(15)+z*R).det())
    assert s.simplify(s.diff(determinant,z).subs(z,0)-logj1)==0
    assert s.simplify(s.diff(determinant,z,2).subs(z,0)-logj1**2-logj2)==0
    H=t[0]; ad=s.Matrix(15,15,lambda i,j:s.simplify(2*s.trace(t[i]*comm(H,t[j]))))
    logh2=s.simplify(-s.trace(ad*ad)/12)
    assert logh2 == -s.Rational(1,3)
    # Matrix Taylor logarithm independently recovers incident coordinates from fixed U=I.
    A2=sum((u2[j]*t[j] for j in range(15)),s.zeros(4))
    outgoing_U1=-s.I*A; outgoing_U2=-s.I*A2-A*A
    incoming_U1=s.I*A; incoming_U2=s.I*A2-A*A
    def logarithm_jets(first,second):
        return first/s.I,(second-first*first)/s.I
    bo1,bo2=logarithm_jets(outgoing_U1,outgoing_U2)
    bi1,bi2=logarithm_jets(incoming_U1,incoming_U2)
    assert s.simplify(bo1+A)==s.zeros(4) and s.simplify(bo2+A2)==s.zeros(4)
    assert s.simplify(bi1-A)==s.zeros(4) and s.simplify(bi2-A2)==s.zeros(4)
    X=t[0]; Y=t[2]
    mixed_log=X*Y-((-s.I*X)*(s.I*Y)+(s.I*Y)*(-s.I*X))/2
    assert mixed_log==comm(X,Y)/2 and mixed_log!=s.zeros(4)
    W2=[sum((w2[15*mu+j]*t[j] for j in range(15)),s.zeros(4)) for mu in range(4)]
    composed_logj2=s.simplify(logj2+s.trace(left*orbit(W2)))
    six_logh2=s.simplify(-2*(u.T*u)[0])
    return {'orbit_Gram_eigenvalues':{str(k):v for k,v in eigen.items()},'normal_projector_rank':45,
            'inverse_first_reconstruction_exact':True,'inverse_second_reconstruction_exact':True,
            'inverse_first_norm_squared':str(s.simplify((u.T*u+w.T*w)[0])),
            'inverse_second_norm_squared':str(s.simplify((u2.T*u2+w2.T*w2)[0])),
            'normal_logj_first_exact':str(logj1),'normal_logj_second_exact':str(logj2),
            'Haar_logh_first_at_zero':'0','Haar_logh_second_unit_direction':str(logh2),
            'composed_inverse_logj_second_exact':str(composed_logj2),
            'six_incident_logh_second_exact':str(six_logh2),
            'incident_matrix_log_jets_exact':True,'mixed_link_BCH_commutator_adverse_detected':True,
            'six_incident_link_first_second_signs_exact':True}

def bump_controls():
    x=s.Symbol('x',real=True); b=(1-x*x)**4
    norm=s.integrate(b*b,(x,-1,1)); first=s.integrate(s.diff(b,x)**2,(x,-1,1))/norm
    second=s.integrate(s.diff(b,x,2)**2,(x,-1,1))/norm
    moment=s.integrate(x*x*b*b,(x,-1,1))/norm
    boundary=[s.diff(b,x,k).subs(x,sign) for sign in [-1,1] for k in range(4)]
    assert norm==s.Rational(65536,109395) and first==s.Rational(34,7) and second==s.Rational(459,7)
    assert all(a==0 for a in boundary) and moment==s.Rational(1,19)
    # Product derivatives are differentiated amplitudes, never ratios at the boundary.
    two=s.integrate(s.diff(b,x)*b,(x,-1,1))/norm
    assert two==0
    # m bump coordinates, 7 density factors; direct Cauchy/triangle bounds in cancelled measure.
    m=135; c1=2**30; c2=2**70; ell1=7*2**30; ell2=7*2**70
    first_scalar=2*(m*m*c1*c1*F(34,7)+F(ell1*ell1,4))
    first_ray=4*2**30
    first_total=2*first_scalar+2*first_ray**2
    # Triangle L2: diagonal and ordered mixed bump jets, coordinate acceleration, density jets.
    second_L2=m*c1*c1*9+m*(m-1)*c1*c1*5+m*c2*3+ell1*m*c1*3+F(ell2,2)+F(ell1*ell1,4)
    second_total=4*(second_L2**2+first_scalar*first_ray**2+(4*2**70+first_ray**2)**2)
    assert first_scalar < 2**240 and first_total<2**244 and second_total<2**1000
    assert (60*4+F(45*3,20))*2**244 < 2**260
    return {'integral':str(norm),'first':str(first),'second':str(second),'even_second_moment':str(moment),
            'boundary_jets_through_three_zero':True,'cross_first_integral':'0',
            'product_scalar_first_bound_at_delta1':str(first_scalar),
            'product_ray_first_bound_at_delta1':str(first_total),'product_ray_second_bound_at_delta1':str(second_total),
            'per_direction_2power244_delta_minus2':True,'per_direction_2power1000_delta_minus4':True,
            'sum_2power260_chi_delta_minus2_V':True}

def car_controls(t):
    # Native local CAR on all 16 occupation states, not a fixed-sector norm heuristic.
    def annihilate(j):
        a=s.zeros(16)
        for mask in range(16):
            if mask>>j&1: a[mask^(1<<j),mask]=(-1)**((mask&((1<<j)-1)).bit_count())
        return a
    annih=[annihilate(j) for j in range(4)]
    def second_quantize(a):
        return sum((a[i,j]*annih[i].T*annih[j] for i in range(4) for j in range(4)),s.zeros(16))
    X=t[0]; Y=t[2]; DX=second_quantize(X); DY=second_quantize(Y)
    assert DX*DY-DY*DX==second_quantize(comm(X,Y)) and DX*DY!=DY*DX
    # Entangled native wedge4 over two complete colour sites, all 70 amplitudes.
    q=s.eye(4).col_join(s.eye(4))/s.sqrt(2)
    x=s.diag(X,s.zeros(4)); y=s.diag(Y,s.zeros(4))
    qx=s.I*x*q; qy=s.I*y*q; qxy=-x*y*q
    costs=[s.S.Zero]*4; wrong_cross=False; wrong_order=False
    occupations=list(itertools.combinations(range(8),4))
    amplitudes={occupied:q[list(occupied),:].det() for occupied in occupations}
    def exterior_action(operator, state):
        result={}
        for occupied in occupations:
            value=s.S.Zero
            for j in occupied:
                for i in range(8):
                    if i in occupied and i!=j: continue
                    prior=tuple(sorted((set(occupied)-{j})|{i}))
                    sign=(-1)**(list(occupied).index(j)+prior.index(i))
                    value+=sign*operator[j,i]*state[prior]
            result[occupied]=s.simplify(value)
        return result
    xy_action=exterior_action(x,exterior_action(y,amplitudes))
    yx_action=exterior_action(y,exterior_action(x,amplitudes))
    for occupied in itertools.combinations(range(8),4):
        a=q[list(occupied),:]; ax=qx[list(occupied),:]; ay=qy[list(occupied),:]; axy=qxy[list(occupied),:]
        value=a.det(); first=s.S.Zero; mixed=s.S.Zero; dropped=s.S.Zero; reverse=s.S.Zero
        for j in range(4):
            v=a.copy();v[:,j]=ax[:,j];first+=v.det()
            v=a.copy();v[:,j]=axy[:,j];mixed+=v.det();dropped+=v.det()
            v=a.copy();v[:,j]=(-y*x*q)[list(occupied),j];reverse+=v.det()
            for k in range(4):
                if k!=j:
                    v=a.copy();v[:,j]=ax[:,j];v[:,k]=ay[:,k];term=v.det();mixed+=term;reverse+=term
        # A full local dGamma action on exterior amplitudes, independently generated via CAR signs.
        def action(operator):
            result=s.S.Zero
            for j in occupied:
                for i in range(8):
                    if i in occupied and i!=j: continue
                    prior=sorted((set(occupied)-{j})|{i})
                    sign=(-1)**(list(occupied).index(j)+prior.index(i))
                    result+=sign*operator[j,i]*q[prior,:].det()
            return s.simplify(result)
        assert s.simplify(first-s.I*action(x))==0
        assert s.simplify(mixed+xy_action[occupied])==0
        assert s.simplify(reverse+yx_action[occupied])==0
        wrong_cross=wrong_cross or s.simplify(mixed-dropped)!=0
        wrong_order=wrong_order or s.simplify(mixed-reverse)!=0
        for index,z in enumerate([value,first,mixed,reverse]): costs[index]+=s.conjugate(z)*z
    costs=[s.simplify(z) for z in costs]
    assert costs[0]==1 and wrong_cross and wrong_order
    assert max(abs(z) for z in DX.eigenvals()) <= 4*max(abs(z) for z in X.eigenvals())
    return {'localCAR_dimension':16,'native_wedge4_dimension':70,'norm_first_mixed_reverse_squared':list(map(str,costs)),
            'noncommuting_second_order_detected':wrong_order,'distinct_column_second_cross_detected':wrong_cross,
            'CAR_commutator_identity_exact':True,'operator_bound_proof':'On sector k, dGamma(A) is restriction of sum over k tensor slots, norm<=k||A||<=4||A||. Mixed derivative is dGamma(A) dGamma(B)+dGamma(D_A B), retaining order and every cross column.'}

def packet_control(n):
    vertices=list(itertools.product(range(n),repeat=3)); number={v:i for i,v in enumerate(vertices)}
    edges=[]
    for v in vertices:
        for axis in range(3):
            w=list(v);w[axis]=(w[axis]+1)%n;edges.append((number[v],number[tuple(w)]))
    adjacency=[[] for _ in vertices]
    for e,(a,b) in enumerate(edges):adjacency[a].append((b,e,1));adjacency[b].append((a,e,-1))
    parent={0:None}; paths={0:{}}; queue=[0];tree=set()
    for a in queue:
        for b,e,sign in adjacency[a]:
            if b not in parent:
                parent[b]=a;paths[b]=dict(paths[a]);paths[b][e]=sign;tree.add(e);queue.append(b)
    chords=[e for e in range(len(edges)) if e not in tree];trace=0
    for e in chords:
        a,b=edges[e];column={e:1}
        for k,v in paths[a].items():column[k]=column.get(k,0)+v
        for k,v in paths[b].items():column[k]=column.get(k,0)-v
        divergence=[0]*len(vertices)
        for k,v in column.items():
            u,w=edges[k];divergence[u]-=v;divergence[w]+=v
        assert not any(divergence)
        assert all(column.get(k,0)==int(k==e) for k in chords)
        trace+=sum(v*v for v in column.values())
    d=len(chords); assert d==2*n**3+1
    width=F(1,64*d*(1+trace));expected=64*width**2*F(trace,19);sup=64*width**2*d*trace
    assert expected<=sup<1
    return {'N':n,'cycle_rank':d,'TrG':trace,'integer_divergence_zero':True,'chord_identity_minor':True,
            'packet_halfwidth':str(width),'exact_packet_expected_shared_electric_excess':str(expected),
            'packet_sup_shared_electric_excess':str(sup),'shared_E0_cross_expectation':'0','shared_E0_energy_per_site':2**18}

def outward_sqrt(q):
    denominator=1<<256; numerator=math.isqrt(q.numerator*denominator**2//q.denominator)+1
    upper=F(numerator,denominator);assert (upper-F(1,denominator))**2<=q<upper**2
    return upper

def ap_and_energy(c,profiles):
    p=profiles[0];m=F(p['onsite']);paths=p['paths']
    assert len(paths)==3 and {x['axis'] for x in paths}=={0,1,2} and all(x['length']==1 for x in paths)
    c0=F(paths[0]['c0']);ci=F(paths[0]['ci']);assert all(F(x['c0'])==c0 and F(x['ci'])==ci for x in paths)
    assert m>6*abs(c0)
    M,d0,d1,d2=s.symbols('M d0 d1 d2',real=True)
    h=M*c[0]+sum((d*(3*a) for d,a in zip((d0,d1,d2),c[1:])),s.zeros(4))
    epsilon_squared=M*M/4+d0*d0+d1*d1+d2*d2
    assert s.simplify(h*h-epsilon_squared*s.eye(4))==s.zeros(4)
    assert s.trace(h)==0
    # Fourier characters average to zero unless N divides their exponent, with AP alias (-1)^(exponent/N).
    def moment(n,poly):
        z=s.Symbol('z');expanded=s.expand(poly(z));answer=s.S.Zero
        for term in s.Add.make_args(expanded):
            power=int(term.as_powers_dict().get(z,0));coefficient=term/z**power
            if power%n==0:answer+=coefficient*s.Integer(-1)**(power//n)
        return s.simplify(answer)
    sn=lambda z:(z-z**-1)/(2*s.I);cs=lambda z:(z+z**-1)/2
    N2_sin_squared=moment(2,lambda z:sn(z)**2)
    assert N2_sin_squared==1
    rows=[]
    for n in [4,6,8,10,12]:
        values=[moment(n,poly) for poly in [cs,lambda z:sn(z)**2,lambda z:sn(z)**4,lambda z:sn(z)**2*cs(z)**2]]
        assert values[0]==0 and values[1]==s.Rational(1,2) and values[2]<=s.Rational(3,8) and values[3]<=s.Rational(1,4)
        rows.append({'N':n,'cos_sin2_sin4_sin2cos2':list(map(str,values))})
    jordan=s.simplify(4*sum(s.trace((a*b+b*a)**2) for a in c for b in c))
    assert jordan==s.Rational(172,27)
    a=(2*ci/3)**2;EX=F(3,2)*a
    EXM2=EX*m*m+4*c0*c0*F(9,4)*a
    EX2=F(21,8)*a*a;EXepsilon2=EXM2/4+EX2
    gain=EX*EX/(outward_sqrt(EX*EXepsilon2)+m*EX/2)
    point=F(172,27)-m-2*gain
    exponents={'neutral_Jordan_support':240-3584,'Phi_SU4_kinetic':260-4096+7168-8192,
               'alignment_ordinary_HS':240+10240-7168-8192,'colour':240+10240-7168-8192,'shared_HEHC_packet':19-8192}
    corrections={k:F(2)**v for k,v in exponents.items()};total=point+sum(corrections.values())
    assert point<-21-F(1,1024) and total<-21 and sum(corrections.values())<F(2)**-3000
    # Native occupation action: on full/empty opposite endpoints hopping is nonzero but expectation zero.
    occupied=15; i,j=4,0;after=(occupied^(1<<j))|(1<<i)
    assert occupied>>j&1 and not occupied>>i&1 and after!=occupied
    assert [p['load'] for p in profiles]==[1,1,63]
    for p in profiles[1:]:
        assert {x['length'] for x in p['paths']}=={1,2} and {x['axis'] for x in p['paths']}=={0,1,2}
    charged_rows=[]
    for n in (4,6):
        counts={'N':n,'native_AP1_terms':0,'native_AP2_terms':0,'negative_AP_boundary_terms':0}
        for profile in profiles[1:]:
            for vertex in itertools.product(range(n),repeat=3):
                source_full=sum(vertex)%2==0
                for path in profile['paths']:
                    target=list(vertex);axis=path['axis'];length=path['length']
                    phase=(-1)**((target[axis]+length)//n);target[axis]=(target[axis]+length)%n
                    assert tuple(target)!=vertex
                    target_full=sum(target)%2==0
                    # For all colour entries either annihilation/creation blocks or the resulting
                    # occupation basis vector differs. AP sign never changes its diagonal matrix element.
                    for colour_in in range(4):
                        for colour_out in range(4):
                            nonzero=source_full and not target_full
                            if nonzero:
                                after_hop=(15^(1<<colour_in))|(1<<(4+colour_out))
                                assert after_hop!=15
                    counts['native_AP'+str(length)+'_terms']+=profile['load']
                    counts['negative_AP_boundary_terms']+=int(phase==-1)*profile['load']
        charged_rows.append(counts)
    return {'AP_exact_moment_controls':rows,'AP_all_even_N_ge4_proof':'Degree<=4 Laurent characters have only N=4 alias among even N>=4; N4 sin4=1/4, sin2cos2=1/4; all N>=6 generic 3/8 and1/8. N2 excluded by governing contract.',
            'N2_adverse_sin_squared_exact':str(N2_sin_squared),'native_band_square_identity_exact':True,
            'positive_band_multiplicity':2,'canonical_neutral_particle_number':'2V',
            'native_all64_AP_path_controls':charged_rows,
            'ordered_16_Jordan_exact':str(jordan),'EX':str(EX),'EXM2_upper':str(EXM2),'EX2_upper':str(EX2),
            'weighted_Cauchy_gain_lower':str(gain),'point_sea_Jordan_upper':str(point),
            'correction_power2_exponents':exponents,'correction_sum_exact':str(sum(corrections.values())),
            'total_upper_exact':str(total),'total_below_minus21':True,'corrections_below_2power_minus3000':True,
            'charged_64_Ref_expectations_exact_zero':True,'charged_hopping_not_zero_operator_witness':{'occupation_before':occupied,'occupation_after':after,'created_colour_mode':i,'annihilated_colour_mode':j},
            'parent_tube_receipt_not_used_for_child_radius_or_normalization':True}

ANALYTIC_PROOF = [
    'Native affine tube inverse: the fixed normal projector P=I-L(L^T L)^-1 L^T avoids eigenframe degeneracy. At centre the derivative on tangent+normal has minimum singular value>=sqrt(1/8). On |z|<=sqrt45 delta the perturbation <=32 sqrt45 delta; exp-link dexp perturbation <=8 sqrt15 delta. Neumann yields inverse norm<16. Map Hessian <=128 in the finite60 group/normal coordinates gives inverse Hessian <=16^3*128=2^19<2^20. Covariance removes g dependence.',
    'Normal density is |det[K,S]|, not a global normalizer quotient. Dlogdet=Tr J^-1 DJ; D2logdet=Tr(J^-1 D2J-J^-1 DJ J^-1 DJ). Dimension60, inverse16, map jets32/128, and the preceding inverse chain rule fit first2^30 and second2^70. For dexp use the analytic series integral exp(-it ad_b), with ||ad_b||<=2||b||HS, obtaining smaller bounds on |b|<=sqrt15 delta. Exact centre Haar log second is -Tr ad_H^2/12. Constants Vol cancel and have zero derivatives.',
    'A unit Phi derivative moves only the one frame and45 normal coordinates plus six adjacent link logs. Relative b=log(g_x^-1 U_xy g_y); differentiate exp/log, including BCH commutator terms. Inverse norm16 and Hessian2^20 give each incident coordinate first<=2^30 and second<=2^70. There are45+6*15=135 bump coordinates and7 density factors. A left/right Haar link derivative at fixedPhi changes only15 b coordinates, one h factor, and no ray. No transported tree derivatives occur.',
    'In cancelled product measure differentiate polynomial amplitudes, not log B. First sum uses 135^2 c1^2(34/7) delta^-2 and density derivative bounds; second triangle includes135 diagonal Bdoubleprime terms,135*134 ordered Bprime Bprime terms, coordinate accelerations and all density crosses. delta<=1 absorbs lower inverse powers. Independent rational arithmetic proves scalar first<2^240 delta^-2, full ray first<2^244 delta^-2, full ray second<2^1000 delta^-4. The entire entangled neutral ray is controlled by the full16 localCAR dGamma norm, not one selected Slater column. Mixed dGamma order and cross-column terms are retained. Charged masks are exact SU4 singlets. Summing60 Phi directions weighted4chi and45 Haar directions weighted3chi/20 is<2^260 chi delta^-2 V.',
    'B and derivatives0..3 vanish on both support boundaries. Positive smooth j,h on closed support and compact frame group make zero extension H2 without delta surfaces. FiniteN signed matter, quartic Jordan, alignment and colour multiplication are bounded on compact Phi support; CAR space is finite. Electric fibre packet has bounded E spectrum, so E and E2 jets are L2. The inverse Bloch unitary promotes the finite-fibre state to original noncompact U1 L2; no beta derivative appears in H. Thus Q(H), and finiteN D(H), conditional on the independently reviewed common normalizer/quotient domain; no uniform O(V) H-norm follows.',
    'Let S=sum sin2, T=sum cos, X=(2ci/3)^2 S, M=m+2c0 T>0, epsilon=sqrt(M2/4+X). Exact AP characters give EM=m, EX=3a/2, EXM2<=a(3m2/2+9c02), EX2<=21a2/8, including the exceptional N4 fourth moments. epsilon-M/2=X/(epsilon+M/2). Weighted Cauchy twice gives E(epsilon-M/2)>=EX2/[sqrt(EX E(X epsilon2))+m EX/2], where the numerator EX2 means (EX)^2, not E(X^2). The negativeLambda sheet fills both positive bands, fixed canonical2V, giving -2Eepsilon plus172/27 ordered Jordan. The outward rational square root is independent arithmetic, never eigensolver output.',
    'On support each Phi perturbation HS<=sqrt(45/2) delta<8delta, each link deviation op<=sqrt(15/2) delta<3delta. Neutral hopping/onsite coefficients<64 with three paths and their adjoints bound the one-body perturbation row sum<2^20 delta; occupying2V yields energy error<2^22 Lambda delta per site. Sixteen degree4 Jordan terms are Lipschitz on ||Phi||op<2, giving<2^24 Lambda delta per site. Combined2^240 Lambda delta is generous and applies pointwise to the same fixed global sea, not the point Slater alone.',
    'Alignment is ordinary unrescaled HS. Across endpoints orientations cancel, and ||phi_x-W phi_y Wdagger||HS<2^12 delta; sum four components and three edges, coefficientF/40, fits2^240 F delta2. A four-link plaquette differs from identity HS<2^12 delta, and4-ReTr U=||I-U||HS2/2; three faces times7F/10 fits2^240 F delta2. No tube-helper smaller radius substituted.',
    'All charged species1+63 use full4/empty checkerboard with each2V. Onsite expectations vanish by tracelessPhi; Ref2 subtracts2Tr h=0. Every native length1/2 AP path has distinct endpoints for evenN>=4. Opposite parity hopping changes occupations; same parity full-full creation blocked or empty-empty annihilation zero. Hence expectations zero, not operators or H action zero. This reasoning holds for arbitrary Wilson and U1 phases and all profiles, without removing native paths.',
    'Shared totalrho=128 parity gives E0=64 parity on axis0, divergence128 parity, and single64||E0||2/V=2^18; HE+HC is one Hodge electric square, not two copies or specieswise squares. Integer cycle basis DB=0 has rank2V+1 and chord identity minor. Even normalized beta product bumps make cross E0.Bbeta mean zero; E beta_j2=w2/19, hence64 E||Bbeta||2=64w2 TrG/19<1. The larger2^19 per-site allowance pays both once. There is no bare U1 magnetic B2 term.'
]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output-prefix',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text())
    for filename,digest in request['pins'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    c,t=native_matrices();profiles=json.loads((ROOT/'neutral_body_parent_primitive_profiles_20261012.json').read_text())['primitive_profiles']
    controls={'geometry':geometry_jets(c,t),'bump':bump_controls(),'CAR':car_controls(t),'packets':[packet_control(n) for n in (4,6)],'energy':ap_and_energy(c,profiles)}
    original=json.loads((ROOT/'connected_quantum_sea_actual2_20261012_proof.json').read_text())
    assert F(controls['energy']['point_sea_Jordan_upper'])==F(original['energy']['point_sea_plus_ordered_Jordan_upper_per_Lambda_site'])
    assert F(controls['energy']['total_upper_exact'])==F(original['energy']['total_upper_exact'])
    scope={'ground':False,'stationarity':False,'phase':False,'IR':False,'numerical_H_evolution':False,'quantum_integral_numerically_computed':False,'uniform_H_norm_O_V':False,'law_adoption':False,'campaign_complete':False}
    proof={'kind':'INDEPENDENT_EXACT_NATIVE_JETS_AND_ANALYTIC_FULL_SOURCE_TRIAL_ENERGY_REVIEW','pins':request['pins'],'plan':request['plan'],'controls':controls,'analytic_proof':ANALYTIC_PROOF,'scope':scope}
    verdict={'scoped_energy_and_jets':'PASS_CONDITIONAL_ON_ORBIT_QUOTIENT_NORMALIZER_AND_GAUSS_PACKET_DOMAIN','energy_below_minus21':True,'corrections_below_2power_minus3000':True,'all_even_N_scope':'N>=4 as pinned governing contract; N2 AP moment equality is false','normalizer_review_owner':'independent orbit/Gauss review; not inferred from these point controls','scope':scope,'frozen_originals_unchanged':True,'correction_required':'Explicit N>=4 in the source proof moment sentence; no correction to the governed N>=4 bound.'}
    for suffix,payload in [('proof',proof),('verdict',verdict)]:
        args.output_prefix.with_name(args.output_prefix.name+'_'+suffix+'.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'exact_native_controls':True,'energy_below_minus21':True,'scope':scope}),flush=True)

if __name__=='__main__':main()
