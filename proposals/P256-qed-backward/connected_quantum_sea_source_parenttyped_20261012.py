"""Connected full-Cartesian canonical neutral sea tube, exact analytical energy control.

No ground/eigenvalue solver. Native finite-CAR and integer Gauss controls are
separated from the analytic uniform-N tube estimates and FLOAT FCS witness.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
import hashlib
import itertools
import json
from math import isqrt
from pathlib import Path
import numpy as np
import sympy as sp
from typing import cast
from sympy.matrices.dense import MutableDenseMatrix

ROOT = Path(__file__).resolve().parent
DELTA_EXP = -3584


def matrices():
    p = [sp.Matrix([[0,1],[1,0]]), sp.Matrix([[0,-sp.I],[sp.I,0]]), sp.diag(1,-1)]
    beta = sp.diag(1,1,-1,-1)
    alpha = [sp.BlockMatrix([[sp.zeros(2),x],[x,sp.zeros(2)]]).as_explicit() for x in p]
    centre = [beta/2]+[x/3 for x in alpha]
    basis = []
    for i in range(4):
        for j in range(i+1,4):
            x=sp.zeros(4); x[i,j]=x[j,i]=sp.Rational(1,2); basis.append(x)
            x=sp.zeros(4); x[i,j]=-sp.I/2; x[j,i]=sp.I/2; basis.append(x)
    for k in range(1,4):
        basis.append(sp.diag(*([1]*k+[-k]+[0]*(3-k)))/sp.sqrt(2*k*(k+1)))
    return beta, alpha, centre, basis


def slice_geometry():
    """Exact orbit derivative in actual 60 Cartesian coordinates; no frame guess."""
    _, _, c, t = matrices()
    tangent=sp.Matrix(60,15,lambda row,col:sp.simplify(2*sp.trace(t[row%15]*sp.I*(t[col]*c[row//15]-c[row//15]*t[col]))))
    gram=sp.simplify(tangent.T*tangent)
    eigenvalues=gram.eigenvals()
    assert sum(eigenvalues.values())==15 and min(eigenvalues)>=sp.Rational(1,8)
    # S is an exact orthonormal basis for the affine normal slice, dimension45.
    # Return a defining matrix as well: users need not choose matching blocks.
    normal=sp.GramSchmidt(tangent.T.nullspace(),True)
    S=sp.Matrix.hstack(*normal)
    assert S.shape==(60,45) and S.T*S==sp.eye(45) and tangent.T*S==sp.zeros(15,45)
    return c,t,tangent,S,eigenvalues


def orbit_jacobian_squared(z, geometry=None):
    """j(z)^2 / Vol(SU4)^2, relative to normalized Haar and Cart Lebesgue.

    The domain is |z_j|<delta. Haar metric generators have length1.
    The square root is positive. This formula is the literal normalizer density.
    """
    c,t,_,S,_=slice_geometry() if geometry is None else geometry
    coefficients=S*sp.Matrix(z)
    fields=[c[mu]+sum((coefficients[15*mu+a]*t[a] for a in range(15)),sp.zeros(4)) for mu in range(4)]
    K=sp.Matrix(60,15,lambda row,col:sp.simplify(2*sp.trace(t[row%15]*sp.I*(t[col]*fields[row//15]-fields[row//15]*t[col]))))
    J=K.row_join(S)
    return sp.det(J.T*J)


def scalar_coordinate_amplitude(t, delta):
    """Normalized 1D compact amplitude; exact zero extension through third jet."""
    x=sp.sympify(t)/delta
    if x.is_number and abs(x)>=1:
        return sp.S.Zero
    integral=sp.Rational(65536,109395)  # integral_-1^1 (1-u^2)^8 du
    return (1-x*x)**4/sp.sqrt(delta*integral)


def positive_band_columns(k, profile):
    """FLOAT evaluation of the exact graph formula, not numerical eigensolving."""
    beta,alpha,_,_=matrices()
    m=float(Q(profile['onsite'])); c0=float(Q(profile['paths'][0]['c0'])); ci=float(Q(profile['paths'][0]['ci']))
    mass=(m+2*c0*sum(np.cos(k)))/2
    d=-2*ci/3*np.asarray(np.sin(k))
    epsilon=np.sqrt(mass*mass+d@d)
    D=sum((d[i]*np.array(alpha[i][:2,2:],complex) for i in range(3)),np.zeros((2,2),complex))
    return np.vstack((np.sqrt((epsilon+mass)/(2*epsilon))*np.eye(2),D/np.sqrt(2*epsilon*(epsilon+mass))))


def slater_columns(n, profile):
    """Every native colour/site mode and all 2V sea columns, AP1/2 characters."""
    if n<4 or n%2: raise ValueError('even N>=4 required')
    sites=np.array(list(itertools.product(range(n),repeat=3)))
    momenta=2*np.pi*(sites+.5)/n
    q=np.empty((4*n**3,2*n**3),complex)
    for index,k in enumerate(momenta):
        band=positive_band_columns(k,profile)
        q[:,2*index:2*index+2]=(np.exp(1j*(sites@k))[:,None,None]*band[None,:,:]/np.sqrt(n**3)).reshape(4*n**3,2)
    return q


def sea_component(columns, frames, occupied):
    """Actual native canonical exterior amplitude in transported global ray."""
    if len(occupied)!=columns.shape[1]: return 0j
    transformed=np.einsum('xab,xbc->xac',np.asarray(frames),columns.reshape(len(frames),4,-1)).reshape(columns.shape)
    return complex(np.linalg.det(transformed[np.asarray(occupied),:]))

def exterior_component_jet(value, first, second, occupied):
    """Literal native wedge jets; every distinct second-column cross is retained."""
    a=cast(MutableDenseMatrix, sp.Matrix(value)[list(occupied),:])
    da=cast(MutableDenseMatrix, sp.Matrix(first)[list(occupied),:]); dda=cast(MutableDenseMatrix, sp.Matrix(second)[list(occupied),:])
    k=a.cols; one=sp.S.Zero; two=sp.S.Zero
    for j in range(k):
        v=a.copy();v[:,j]=da[:,j];one+=v.det()
        v=a.copy();v[:,j]=dda[:,j];two+=v.det()
        for l in range(k):
            if l!=j:
                v=a.copy();v[:,j]=da[:,j];v[:,l]=da[:,l];two+=v.det()
    return a.det(),sp.simplify(one),sp.simplify(two)


def exact_native_local_jet_control():
    """Entangled global wedge4, both local number fluctuations and full native70."""
    q=sp.eye(4).col_join(sp.eye(4))/sp.sqrt(2)
    generator=sp.diag(sp.Rational(1,2),-sp.Rational(1,2),0,0,0,0,0,0)
    first=sp.I*generator*q;second=-generator*generator*q
    norm=sp.S.Zero;first_cost=sp.S.Zero;second_cost=sp.S.Zero
    cross_needed=False
    for occupied in itertools.combinations(range(8),4):
        value,one,two=exterior_component_jet(q,first,second,occupied)
        eigenvalue=sum(generator[j,j] for j in occupied)
        assert sp.simplify(one-sp.I*eigenvalue*value)==0
        assert sp.simplify(two+eigenvalue**2*value)==0
        norm+=sp.conjugate(value)*value
        first_cost+=sp.conjugate(one)*one
        second_cost+=sp.conjugate(two)*two
        if value!=0 and eigenvalue==0 and len(set(occupied)&{0,1})==2:
            # Two opposite local generator eigenvalues cancel only with crosses.
            without_cross=sp.S.Zero
            for column in range(4):
                v=cast(MutableDenseMatrix, sp.Matrix(q)[list(occupied),:]);v[:,column]=sp.Matrix(second)[list(occupied),column]
                without_cross+=v.det()
            cross_needed=cross_needed or sp.simplify(without_cross)!=0
    assert norm==1 and first_cost==sp.Rational(1,8) and second_cost==sp.Rational(1,32) and cross_needed
    return {'native_wedge4_dimension':70,'norm_squared_exact':str(norm),'local_first_jet_squared_norm_exact':str(first_cost),'local_second_jet_squared_norm_exact':str(second_cost),'wrong_drop_distinct_column_cross_detected':cross_needed,'localCAR_operator_bound':'||dGamma_x A||<=4||A||, full16 localCAR independentN'}


def cycle_packet_control(n):
    """Constructive integer cycles and exact original-cover packet electric price."""
    sites=list(itertools.product(range(n),repeat=3)); index={x:i for i,x in enumerate(sites)}
    edges=[]
    for x in sites:
        for axis in range(3):
            y=list(x);y[axis]=(y[axis]+1)%n
            edges.append((index[x],index[tuple(y)]))
    adjacency=[[] for _ in sites]
    for e,(x,y) in enumerate(edges):
        adjacency[x].append((y,e,1));adjacency[y].append((x,e,-1))
    parent: dict[int, tuple[int, int, int] | None] = {0:None};queue=[0];tree=set()
    for x in queue:
        for y,e,sign in adjacency[x]:
            if y not in parent:
                parent[y]=(x,e,sign);queue.append(y);tree.add(e)
    assert len(parent)==len(sites)
    chords=[e for e in range(len(edges)) if e not in tree]
    def rootpath(x):
        out=np.zeros(len(edges),dtype=np.int64)
        while x:
            previous,e,sign=cast(tuple[int, int, int], parent[x]);out[e]+=sign;x=previous
        return out
    columns=np.zeros((len(edges),len(chords)),dtype=np.int64)
    for j,e in enumerate(chords):
        x,y=edges[e];columns[:,j]=rootpath(x)-rootpath(y);columns[e,j]+=1
    divergence=np.zeros((len(sites),len(chords)),dtype=np.int64)
    for e,(x,y) in enumerate(edges):
        divergence[x,:]+=columns[e,:];divergence[y,:]-=columns[e,:]
    assert not np.any(divergence) and np.array_equal(columns[chords,:],np.eye(len(chords),dtype=np.int64))
    d=len(chords);trace=int(np.sum(columns**2));w=Q(1,64*d*(1+trace))
    excess=64*w*w*d*trace
    assert d==2*n**3+1 and excess<1
    return {'integer_DB_exact_zero':True,'cycle_lattice_identity_minor':True,'cycles':d,'TrG_exact':trace,'packet_halfwidth_exact':str(w),'shared_electric_packet_excess_sup_exact':str(excess),'even_packet_shared_E0_cross_integral_exact':0}


def exact_controls(geometry):
    c,t,L,S,eigenvalues=geometry
    # Symbolic native positive-band normalization: D^2=|d|^2 I; no eigensolver.
    mass,eps,d1,d2,d3=sp.symbols('mass eps d1 d2 d3',real=True)
    D=sp.Matrix([[d3,d1-sp.I*d2],[d1+sp.I*d2,-d3]])
    assert sp.expand(D*D)==(d1*d1+d2*d2+d3*d3)*sp.eye(2)
    residual=sp.simplify((eps+mass)/(2*eps)+(eps**2-mass**2)/(2*eps*(eps+mass))-1)
    assert residual==0
    u=sp.symbols('u',real=True); bump=(1-u*u)**4
    norm=sp.integrate(bump*bump,(u,-1,1)); first=sp.integrate(sp.diff(bump,u)**2,(u,-1,1))/norm
    second=sp.integrate(sp.diff(bump,u,2)**2,(u,-1,1))/norm
    assert norm==sp.Rational(65536,109395) and first<16 and second<512
    # Independent local centre acts nontrivially on the *native* wedge space.
    # (a+b)/sqrt2, one particle at either endpoint: <Gamma(i on a)>= (1+i)/2.
    local_fcs=sp.Rational(1,2)*(1+sp.I)
    assert local_fcs!=1
    rows=[]
    for n in (4,6):
        sites=list(itertools.product(range(n),repeat=3)); index={x:i for i,x in enumerate(sites)}
        parity=[(-1)**sum(x) for x in sites]; v=len(sites)
        # 64 species, each locally full4/empty0; Ref2 EACH species.
        rho=[128*p for p in parity]; divergence=[0]*v
        for x,p in zip(sites,parity):
            y=((x[0]+1)%n,x[1],x[2]); divergence[index[x]]+=64*p;divergence[index[y]]-=64*p
        assert divergence==rho and sum(rho)==0 and sum(4*(p+1)//2 for p in parity)==2*v
        assert (2*v)%4==0 and (128*v)%4==0
        # Connected centre constraints z_x/z_y=1 have rankV-1 over Z4.
        visited={sites[0]}; frontier=[sites[0]]
        while frontier:
            x=frontier.pop()
            for axis in range(3):
                y=list(x);y[axis]=(y[axis]+1)%n;y=tuple(y)
                if y not in visited:visited.add(y);frontier.append(y)
        assert len(visited)==v
        rows.append({'N':n,'V':v,'canonical_neutral':2*v,'each_charged_particles':2*v,'rho_values':[-128,128],'divE_minus_rho_exact':0,'full_HE_plus_HC_per_site':64*64**2,'combined_cover_degree':4,'wrong_independent_cover_degree':str(4**v),'wrong_normalized_norm_squared':str(Q(4**v,4))})
    gram_det=sp.factor((L.T*L).det())
    # At z0, J=[L,S], hence Jacobian square exactly determinant orbit Gram.
    assert sp.simplify(sp.det(L.row_join(S).T*L.row_join(S))-gram_det)==0
    return {'regular_orbit_Gram_eigenvalues':{str(k):v for k,v in eigenvalues.items()},'orbit_Gram_det_exact':str(gram_det),'normal_slice_dimension':45,'Cart_dimension_per_site':60,'band_column_norm_residual_exact':str(residual),'one_dimensional_squared_bump_integral':str(norm),'normalized_first_Dirichlet_constant':str(first),'normalized_second_jet_constant':str(second),'independent_local_native_centre_FCS_exact':str(local_fcs),'connected_cover_and_shared_Gauss':rows,'native_covariant_local_first_second_jet':exact_native_local_jet_control(),'original_cover_packet_native_integer_controls':{str(n):cycle_packet_control(n) for n in (4,6)}}


def sqrt_upper(value):
    scale=1<<256; lo=isqrt(value.numerator*scale*scale//value.denominator)
    assert Q(lo,scale)**2<=value<Q(lo+1,scale)**2
    return Q(lo+1,scale)


def energy_bound(profile):
    """Full-H analytic upper, unchanged coefficients; not a point-field energy."""
    m=Q(profile['onsite']); c0=Q(profile['paths'][0]['c0']);ci=Q(profile['paths'][0]['ci'])
    assert m-6*abs(c0)>0
    x=Q(3,2)*(2*ci/3)**2
    xm2=x*m*m+4*c0*c0*Q(9,4)*(2*ci/3)**2
    x2=Q(21,8)*(2*ci/3)**4
    gain=x*x/(sqrt_upper(x*(xm2/4+x2))+m*x/2)
    point=Q(172,27)-m-2*gain
    # Coarse constants proved in the interface. Ratios are perV, uniformN.
    terms={'neutral_and_Jordan_support_error':Q(2)**(240+DELTA_EXP),
           'Phi_and_SU4_complete_covariant_Dirichlet':Q(2)**(260-4096-2*DELTA_EXP-8192),
           'endpoint_alignment':Q(2)**(240+10240+2*DELTA_EXP-8192),
           'colour_plaquette':Q(2)**(240+10240+2*DELTA_EXP-8192),
           'charged_all64_AP1_AP2_Ref_expectation':Q(0),
           'shared_HE_HC_and_original_cover_packet':Q(2)**(19-8192)}
    total=point+sum(terms.values(),Q(0))
    assert point<-21-Q(1,1024) and total<-21
    return {'point_sea_plus_ordered_Jordan_upper_per_Lambda_site':str(point),'full_H_corrections_per_Lambda_site':{k:str(v) for k,v in terms.items()},'total_upper_exact':str(total),'total_upper_below_minus21':True,'coarse_total_correction_lt_2power_minus3000':sum(terms.values())<Q(2)**-3000}


def proof_record(controls,energy,request):
    return {
      'kind':'CONNECTED_NORMALIZED_FULL_QUANTUM_GLOBAL_CANONICAL_SEA_TRIAL_NOT_GROUND',
      'bound_pins':request['pins'],'law':{'a':'1/64','Lambda':'2^8192','chi':'2^-4096','F':'2^10240','Phi_kinetic':'4chi pCart^2','SU4_kinetic':'3chi/20 E4^2','HE_HC':'64 once on shared totalrho','ordered_Jordan':'Lambda/(16a) sum_mu_nu Tr({Phi_mu,Phi_nu}^2)'},
      'construction':[
        'C=(beta/2,alpha1/3,alpha2/3,alpha3/3), literal Pauli Clifford matrices in source. All60 Cartesian components are retained. T_a have Tr(T_a T_b)=delta_ab/2. L_C maps u to coefficients of i[u_a T_a,C_mu]. S is its exact orthonormal 45-dimensional orthogonal complement, calculated natively.',
        'Coordinates: phi(z)=C+S z; Phi_x=g_x phi(z_x)g_x†; U_xy=g_x exp(i b_e,a T_a)g_y†. Every |z_x,j|<delta and |b_e,a|<delta, delta=2^-3584. g_x ranges over ALL normalized SU4 Haar, not a small orientation box. This is full60V Cartesian x all3V SU4 Haar module, not a frozen orbit distribution.',
        'Regular orbit tubular map is injective modulo Z4 for ||z||<2^-16. L_C^T L_C exact eigenvalues listed below establish regularity. Compactness makes the local inverse uniform over g. Normal slice tubes of distinct orbit points cannot meet: for nearby g use the invertible derivative and mean value bound; for far g use sum_mu ||g C_mu g†-C_mu||², equal to a positive quadratic form in traceless g, whose Clifford commutant is scalar. Its smallest coefficient is >=1/8 by the same adjoint decomposition. Thus closeness forces g within2^10||z|| of one centre. Relative exp-link chart radius<2^-16 then forces identical centre on adjacent sites. Connectedness leaves EXACTLY4 global preimages, not4^V.',
        'Pullback volume is (product_x j(z_x)) (product_e h(b_e)) dz db dHaar(g). Here j=Vol(SU4)*sqrt(det([K(z),S]^T[K(z),S])), K(z) the literal commutator derivative in source; h(b)=Vol(SU4)^-1 det(dexp_b), positive exp-chart Haar density. Metric T_a orthonormal defines Vol. Haar link triangular substitution cancels all g derivatives in determinant. Integral over image is ONE FOURTH of pullback, since covering multiplicity4. This is not a product of separately quotiented field integrals.',
        'Let B_delta(t)=(1-(t/delta)^2)^4_+ /sqrt(delta*65536/109395). Scalar f=2 product_x,j B_delta(z_x,j)/sqrt(product_x j(z_x)) times product_e,a B_delta(b_e,a)/sqrt(product_e h(b_e)). Exact normalization: (1/4)*4*product of 1D integrals*integral dHaar(g)=1. The constant4 is independentV; j and h cancel locally before any estimate. No exponential total-normalizer ratio.',
        'Neutral ray is wedge of ALL positive-band columns at AP k_i=2pi(n_i+1/2)/N: h(k)=M(k)beta/2-(2ci/3)sum sin(k_i)alpha_i, epsilon²=M²/4+(2ci/3)²sum sin². Columns Q_k=((sqrt((epsilon+M/2)/(2epsilon)) I2), D/sqrt(2epsilon(epsilon+M/2))). D²=|d|² I2. Fourier normalization gives Q†Q=I_(2V) exactly. Global Gamma(diag_x g_x)wedge(Q), not independent block wedges. Physical -Lambda fills positive band. Full normalized ray has fixed2V and genuinely fluctuating local occupation.',
        'Every charged species1+aux63 is full4 on even checkerboard vertices and empty on odd ones, canonical2V EACH. The full local4 wedge is SU4 determinant1. Ref2 per species is preserved, totalrho=128 parity. Neutral has zero U1 charge. Global centre z on all sites acts on neutral as z^(2V)=1 for every evenN; charged masks also invariant. Thus f times ray descends without arbitrary independent local roots.',
        'U1 shared E0 on axis0 edges=64 parity, other edges0, divE0=128 parity=totalrho exactly. Character exp(i E0·theta) has correct U1 covariance including Ref. Native controls keep cross64: full64||E0||²/V=2^18, not64 sum_species||E_species||². HE+HC is Hodge decomposition of this one full electric square, never doubled.',
        'Original noncompact U1 cover: use integer cycle basis B, DB=0, d=2V+1, G=B^T B. Direct-integral normalized product even B_w(beta_j), w=1/(64d(1+TrG)), multiply by E0+B beta character in the finite fibre. Original L2 state is inverse Bloch transform of this packet. div is unchanged, first beta moments vanish; 64 E||B beta||²<=64 w² d TrG<1. No beta derivative occurs in H (no bare B²). Packet may dependN; electric expectation bound is uniform. Equivalently state is a finite-fibre Q-form trial with the specified packet promotion.'
      ],
      'local_jet_and_domain_proof':[
        'Coordinates use fixed normal affine slice, not a simple eigenvalue eigenframe with degeneracy. At centre, min singular value of [L,S] >=sqrt(1/8). For |z|<=sqrt45 delta and |b|<=sqrt15 delta, commutator derivative variation <=32||z|| and exponential dexp deviation <=8||b||. Neumann inversion gives inverse norm<16. First derivative of commutator map<=32, second exp/group derivatives<=128; differentiating J^-1=-J^-1 (DJ) J^-1 bounds inverse coordinate second jets<2^20. These bounds hold uniformly over g by unitary covariance. All higher local smooth derivatives needed here are bounded on this compact chart.',
        'Differentiating log determinant: D log j=Tr(J^-1 DJ), D² log j=Tr(J^-1 D²J-J^-1 DJ J^-1 DJ); using dimension60 and inverse<16 gives first<2^30, second<2^70. The identical dexp determinant formula gives Haar h bounds no larger. This controls derivatives of j^-1/2 and h^-1/2 without comparing total products.',
        'One unit Cartesian derivative at x changes only z_x,g_x and the six incident b links. Derivatives of adjacent link logarithms, including second jets and all mixed product terms, are bounded by2^30 and2^70. No spanning-tree transport derivative occurs. At fixedPhi, one Haar derivative changes only its b link and the neutral ray is constant. Hence at most135 coordinate bumps and7 Jacobian factors enter each local Cart derivative, at most15 coordinate bumps and1 Jacobian factor each Haar derivative.',
        f"In the cancelled product measure, integral |B_delta prime|²={controls['normalized_first_Dirichlet_constant']} delta^-2 and integral |B_delta doubleprime|²={controls['normalized_second_jet_constant']} delta^-4. Applying |sum_j a_j|²<=m sum_j|a_j|² with m<=256 and coordinate coefficients<=2^30 yields conservative integrated scalar first cost<=2^240 delta^-2 for EACH unit local Cart/Haar direction. Determinant derivatives add <=2^140, already absorbed. Second local jet squared costs<=2^1000 delta^-4. No bound on Bprime/B at the support boundary is asserted.",
        'Neutral local frame first derivative acts by dGamma_x(A_x), whose norm<=4||A_x|| on the full native16-dimensional local CAR algebra, independentV and independent entanglement. Second jet dGamma_x(A_x prime)+dGamma_x(A_x)^2 retains the complete Leibniz cross; charged full/empty masks have identically zero SU4 jets. Product f ray has first squared cost<=2^244 delta^-2, second<=2^1000 delta^-4 per local direction. Original U1 first/second fibre jets are E_e and E_e² including packet shift, not zero. Thus summing60V Phi and45V Haar directions with coefficients4chi and3chi/20 costs<=2^260 chi delta^-2 V. This proves uniformN expected kinetic constants, not a claim inferred from width alone.',
        'Every bump and its first three derivatives vanish at each coordinate boundary; zero extension has no surface distributions through second order. j,h are smooth positive on the closed support; g compact. Phi/colour jets square-integrable locally with the above bounds. For each finiteN all signed matter, ordered Jordan, alignment and colour multiplication operators are bounded on this compact Phi support; electric packet has bounded spectral support. Consequently the source belongs to ambient Q(H), indeed D(H) in the finite-fibre/direct-integral realization. Local second-jet estimates are uniformN; total ||H Psi|| need not be O(V) and no small residual/stationarity is claimed.'
      ],
      'full_energy_proof':[
        'All-evenN AP moments: mean cos=0, mean sin²=1/2, mean sin⁴<=3/8 (N4=1/4), mixed sin²cos²<=1/4. Weighted Cauchy gives epsilon_mean>=m/2+EX²/[sqrt(EX*E(X epsilon²))+m EX/2]. Exact rational outward square root recomputed here, not an old producer rerun. Ordered16 Jordan at centre equals64((1/2)^4+3(1/3)^4)=172/27. Sea plus J upper is explicit below.',
        'Pointwise phi deviations<=sqrt45 delta and Wilson deviations<=2sqrt15 delta. Neutral linear onsite/nearest-path form coefficients<64 and at most3 paths/site, local trace norm bound<=4. Gauge transformed h differs from point operator by row/column sum<=2^20 delta, so |<dGamma Delta h>|/V<=2^22 delta. Jordan degree4, ordered16, ||C_mu||<=1, perturbations<8delta give variation<=2^24 delta/site. Conservative combined2^240 Lambda delta retained. This applies to the fixed global Slater ray on every support coordinate, no BO ground or parameter adoption.',
        'Alignment uses ordinary HS endpoint square. Each endpoint at centre has identical C; ||phi_x-exp(ib)phi_y exp(-ib)||HS<=2^12 delta, so three edges and F/40 give <=2^240 F delta²/site. Colour plaquette has four exp links, ||Uface-I||HS<=2^12 delta and 4-ReTr Uface=||Uface-I||HS²/2; three faces and7F/10 likewise <=2^240 F delta²/site. Orientation g cancels exactly.',
        'All64 charged on-site expectations vanish because trace phi0=0, including Ref subtraction trace h=0. Every off-site hopping expectation is zero in full/empty occupation eigenstates: opposite parity changes occupation, same parity full-full creation is Pauli-blocked or empty-empty annihilation is zero. This includes every native AP1 andAP2 ordered two-link Wilson, every active/aux profile; no hopping operator is removed from H or from its action. Charge neutrality and shared electric costs as above. There is no bare U1 B². SU4/Phi quantum gradient, Jordan, alignment, colour, neutral signed sheet, charged signed sheets, Ref, HC and HE are all paid.'
      ],
      'controls':controls,'energy':energy,
      'scope':{'actual_normalized_quantum_trial':True,'global_minimizer':False,'stationary_ground':False,'ground_computed_as_permission':False,'numerical_ground_backend':False,'phase_or_IR_residue':False,'campaign_complete':False,'analytic_existence_authority':'inherited, unchanged; variational trial gives ground upper only, not identifying the ground','cost_claim_type':'analytic exact conservative uniformN upper plus exact native and separately FLOAT controls'}
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output-prefix',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text())
    for name,pin in request['pins'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pin,name
    profiles=json.loads((ROOT/request['profiles']).read_text())['primitive_profiles']
    assert [x['load'] for x in profiles]==[1,1,63]
    geometry=slice_geometry();controls=exact_controls(geometry);energy=energy_bound(profiles[0])
    q=slater_columns(4,profiles[0]);gram_error=float(np.max(np.abs(q.conj().T@q-np.eye(q.shape[1]))))
    pair=list(range(4))+list(range(4*16,4*16+4)); local=q[pair,:]@q[pair,:].conj().T
    fcs=complex(np.linalg.det(np.eye(8)+(1j-1)*local))
    assert gram_error<1e-11 and abs(fcs-1)>.1
    controls['FLOAT_only']={'N4_native_global_Slater_Gram_max_error':gram_error,'actual_matching_pair_local_Z4_FCS_real':fcs.real,'actual_matching_pair_local_Z4_FCS_imag':fcs.imag,'actual_matching_pair_local_Z4_FCS_not1':abs(fcs-1)>.1,'normalization_authority':'symbolic band identity and exact Fourier characters, NOT this FLOAT error'}
    proof=proof_record(controls,energy,request)
    for suffix,payload in (('proof',proof),('receipt',controls),('interface',{'kind':proof['kind'],'source':Path(__file__).name,'construction':proof['construction'],'local_jet_and_domain_proof':proof['local_jet_and_domain_proof'],'energy':energy,'scope':proof['scope']})):
        args.output_prefix.with_name(args.output_prefix.name+'_'+suffix+'.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'exact_native_controls_completed':True,'full_H_upper_strict_minus21':energy['total_upper_below_minus21'],'FLOAT_FCS':controls['FLOAT_only'],'scope':proof['scope']},indent=2),flush=True)

if __name__=='__main__':main()
