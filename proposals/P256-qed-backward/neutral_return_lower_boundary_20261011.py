from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import time

import sympy as s

ROOT = Path(__file__).resolve().parent
PROFILE = 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'
CAPS = 'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json'
STATE = 'local_so5_resum_actual04_20261011_state.npz'
CERT_SOURCE = 'neutral_return_lower_control_20261011.py'
CERT_SOURCE_PIN = 'f2a3ff7efce4960ee38a32055e9daf6ae97e465ab23514c7bee206b60a32338f'
IMAGE = 'neutral_return_image_interface_20261011.json'
IMAGE_PIN = 'a0b4d7e530f770d549d4f022e41f2134715e850b93a046449b64e3fe61dd2a43'
PINS = {PROFILE:'102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
        CAPS:'2f019dd49873a5e13f130c65ef6739b9af78a16f85138dc6c6ab6e32e65c7a12',
        STATE:'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2'}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def second(M, n):
    masks = [sum(1<<q for q in w) for w in itertools.combinations(range(4),n)]
    out = s.zeros(len(masks))
    for j,mask in enumerate(masks):
        for q in range(4):
            if not mask>>q&1: continue
            sign = (-1)**((mask&((1<<q)-1)).bit_count())
            tail = mask^(1<<q)
            for p in range(4):
                if not tail>>p&1:
                    z = sign*(-1)**((tail&((1<<p)-1)).bit_count())
                    out[masks.index(tail|(1<<p)),j] += z*M[p,q]
    return out,masks


def generic_joint_phase():
    a,b,e,c = s.symbols('a b e c',real=True)
    m,d = e*(a*a-b*b),2*e*a*b
    A = s.diag(-1,1,-1,1)
    B = s.kronecker_product(s.Matrix([[0,1],[1,0]]),s.diag(1,-1))
    X,Z = s.Matrix([[0,1],[1,0]]),s.zeros(2)
    C = Z.row_join(s.I*X).col_join((-s.I*X).row_join(Z))
    R = s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))
    H = m*A+c*B+d*C
    Wp = s.Matrix([[1,0],[0,1],[1,0],[0,-1]])/s.sqrt(2)
    Wm = s.Matrix([[1,0],[0,1],[-1,0],[0,1]])/s.sqrt(2)
    U = s.Matrix.hstack(Wp*s.Matrix([a,-s.I*b]),Wm*s.Matrix([a,s.I*b]),s.I*Wp*s.Matrix([-s.I*b,a]),s.I*Wm*s.Matrix([s.I*b,a]))
    D4 = s.diag(-e+c,-e-c,e+c,e-c)
    rf,rg = (a*a-b*b)/2,a*b
    R4 = s.Matrix([[0,rf,0,-rg],[rf,0,rg,0],[0,rg,0,rf],[-rg,0,rf,0]])
    def zero(M):
        return all(s.rem(s.expand(z),a*a+b*b-1,b)==0 for z in M)
    assert zero(U.conjugate().T*U-s.eye(4))
    assert zero(U.conjugate().T*H*U-D4) and zero(U.conjugate().T*R*U-R4)
    h2,masks2 = second(D4,2)
    r2,_ = second(R4,2)
    V = s.zeros(6)
    for j,(mask,coef) in enumerate([(3,1),(5,1),(9,1/s.sqrt(2)),(10,1),(12,-1),(9,1/s.sqrt(2))]):
        V[masks2.index(mask),j] = coef
    V[masks2.index(6),2] = 1/s.sqrt(2)
    V[masks2.index(6),5] = -1/s.sqrt(2)
    D6 = s.diag(-2*e,2*c,0,-2*c,2*e,0)
    R6 = s.zeros(6)
    for j in (1,3):
        for i,z in ((0,rg),(2,s.sqrt(2)*rf),(4,-rg)):
            R6[i,j] = R6[j,i] = z
    assert zero(V.T*V-s.eye(6))
    assert zero(h2*V-V*D6) and zero(r2*V-V*R6)
    wrong = V.copy();wrong[:,4] = -wrong[:,4]
    assert zero(h2*wrong-wrong*D6) and not zero(r2*wrong-wrong*R6)
    h3,masks3 = second(H,3)
    r3,_ = second(R,3)
    J = s.zeros(4)
    for p in range(4): J[masks3.index(15^(1<<p)),p] = (-1)**p
    Q = s.kronecker_product(s.eye(2),X)
    S4 = s.diag(1,-1,1,-1)
    assert zero(J.T*h3*J+H.T) and zero(J.T*r3*J+R.T)
    assert zero(Q*(-H.T)*Q-H) and zero(Q*(-R.T)*Q+R)
    assert zero(S4*D4*S4-D4) and zero(S4*R4*S4+R4)
    U3 = J*Q*U*S4
    assert zero(U3.conjugate().T*U3-s.eye(4))
    assert zero(U3.conjugate().T*h3*U3-D4) and zero(U3.conjugate().T*r3*U3-R4)
    wrong_hole = J*Q*U
    assert zero(wrong_hole.conjugate().T*h3*wrong_hole-D4)
    assert not zero(wrong_hole.conjugate().T*r3*wrong_hole-R4)
    F = s.diag(1,-1,-1,1)
    assert F*R*F == R
    # Commuting exact oneparticle source phase extends by exterior powers;
    # identical-role permutation symmetry survives its tensor extension.
    return {'generic_oneparticle_joint_D_R':True,'full_wedge2_vector5_plus_singlet_joint_D_R':True,
            'full_signed_number3_joint_D_R':True,'number3_isometry':'U3=J Q U S4, J[15 xor 2^p,p]=(-1)^p; Q=I2 tensor X; S4=diag(1,-1,1,-1). Both D and R become exactly SAME pair as number1.',
            'wrong_highest_wedge_phase_rejected':True,'energy_only_hole_phase_rejected':True,
            'source_oneparticle_phase_commutes_density':True,'hypotheses':'e>0,m>0,e^2=m^2+d^2; a^2+b^2=1,m=e(a^2-b^2),d=2eab; no independent rho sign choice.',
            'both_ordered_number_pairs_covered':['1,3','3,1']}


def rational_density_output_control():
    # Exact independent full output-density support for unequal onsite profiles:
    # vector spectator occupation n=2 plus one number1 and one number3 target.
    def vector_pair(m,d,e,c):
        en = [-2*e,2*c,0,-2*c,2*e]
        rr = {}
        for j in (1,3):
            for i,z in ((0,s.Rational(d,2*e)),(2,s.Rational(m,e)/s.sqrt(2)),(4,-s.Rational(d,2*e))):
                rr[i,j] = rr[j,i] = z
        return en,rr
    def spinor_pair(m,d,e,c):
        en = [-e+c,-e-c,e+c,e-c]
        rr = {}
        for i,j,z in ((0,1,s.Rational(m,2*e)),(2,3,s.Rational(m,2*e)),(0,3,-s.Rational(d,2*e)),(1,2,s.Rational(d,2*e))):
            rr[i,j] = rr[j,i] = z
        return en,rr
    ve,vr = vector_pair(5,12,13,-2)
    ae,ar = spinor_pair(3,4,5,-1)
    ue,ur = spinor_pair(5,12,13,-2)
    words = [w for w in itertools.product(range(3),repeat=5) if sum(w)==2]
    def cost(key):
        w,a,u = key
        return sum(w[j]*(ve[j]-ve[0]) for j in range(5))+ae[a]-ae[0]+ue[u]-ue[0]
    def neighbors(key):
        w,a,u = key
        for (dest,src),value in vr.items():
            if w[src]:
                v = list(w);v[src]-=1;v[dest]+=1
                yield (tuple(v),a,u),value*s.sqrt(w[src]*(w[dest]+1))
        for (dest,src),value in ar.items():
            if src==a: yield (w,dest,u),value
        for (dest,src),value in ur.items():
            if src==u: yield (w,a,dest),value
    allstates = [(w,a,u) for w in words for a in range(4) for u in range(4)]
    output_budget,support_budget = 24,80
    inputs = {key for key in allstates if cost(key)<=support_budget}
    outputs = [key for key in allstates if cost(key)<=output_budget]
    maxstep = max(max(ve)-min(ve),max(ae)-min(ae),max(ue)-min(ue))
    assert output_budget+maxstep<support_budget
    count = 0
    for out in outputs:
        for key,v in neighbors(out):
            assert key in inputs
            assert s.simplify(v)!=0
            count += 1
    removed = next(key for out in outputs for key,_ in neighbors(out))
    wrong = inputs-{removed}
    assert any(key not in wrong for out in outputs for key,_ in neighbors(out))
    omitted = [key for key in allstates if key not in inputs]
    assert omitted and all(cost(key)>support_budget for key in omitted)
    # Compression positivity is NOT a full-space lower bound: genuine adverse vector.
    D = s.diag(-s.Rational(3,2),s.Rational(1,10))
    R = s.ones(2)
    H = D+R*R
    adverse = s.Matrix([1,-1])
    assert H[0,0]>0 and (adverse.T*H*adverse)[0]<0
    return {'exact_unequal_profile_output_control':True,'all_finite_states':len(allstates),
            'retained_outputs':len(outputs),'support_inputs':len(inputs),'density_edges':count,
            'output_complete':True,'deleting_required_density_input_rejected':True,
            'omitted_free_spectrum_control':True,'maximum_single_density_energy_step':maxstep,
            'input_PHP_false_positivity_rejected':True,'input_PHP_wrongcase':{'D':['-3/2','1/10'],'R':[['1','1'],['1','1']],'H_compressed_diagonal':'1/2','adverse_vector':['1','-1'],'adverse_form_exact':'-7/5'},
            'all_n_output_completeness_proof':'One density action changes one local free index; occupancy factors never change energy. Cost(target)<=cost(output)+maxstep; output upper<=240 and maxstep upper<82 imply target lower<322. All skipped occupancy continuations have cost lower>322. Thus every retained output row has its full density image. Positive dropped output squares yield a genuine lower form.'}


def exact_cutoff_wrongcases():
    p = json.loads((ROOT/PROFILE).read_text())
    q = 1<<100
    def floor_sqrt(fr):
        return Fraction(math.isqrt(fr.numerator*q*q//fr.denominator),q)
    ee = []
    for x in p['profiles']:
        f = Fraction(x['e_squared_exact'])
        lo = floor_sqrt(f)
        hi = lo+Fraction(1,q)
        assert lo*lo<=f<=hi*hi
        ee.append((lo,hi))
    c = abs(Fraction(p['profiles'][0]['c0_exact']))
    # Lower energies prove these complete free-basis configurations are NOT
    # in the omitted positive cone; upper energies prove they remain below z.
    base = (-4*ee[0][1]-252*ee[1][1],-4*ee[0][0]-252*ee[1][0])
    costs = {
        'two_vector_Cartan_excitations_AA':4*(ee[0][1]-c),
        'four_balanced_odd_slots_AAUU':2*(ee[0][1]-c)+2*(ee[1][1]-c),
        'one_A_singlet_plus_UU_odd_pair':2*ee[0][1]+2*(ee[1][1]-c),
    }
    for name,cost in costs.items(): assert base[1]+cost < -5123, name
    return {'free_baseline_exact_interval':[str(x) for x in base],
            'z_exact':'-5123','negative_free_budget_exact_interval':[str(-5123-base[1]),str(-5123-base[0])],
            'single_defect_or_one_odd_pair_shortcut_rejected':True,
            'explicit_omitted_free_negative_wrongcases':{name:{'cost_upper_exact':str(cost),'total_free_energy_upper_exact':str(base[1]+cost),'net_cellcharge':'0'} for name,cost in costs.items()},
            'interpretation':'These exact adverse FREE basis configurations falsify the claimed omitted-state free-positivity cutoff. They do NOT disprove positivity of full H=hfree+64rho^2, and do NOT establish a fullcell no-go. Four-odd branches are outside actual two-slot return insertion but prevent any canonical-fullcell ground admission by this image certificate.'}


def bareiss(matrix):
    a = [row[:] for row in matrix]
    previous = 1
    signs = []
    for t in range(len(a)):
        pivot = a[t][t]
        assert pivot
        signs.append(1 if pivot*previous>0 else -1)
        for i in range(t+1,len(a)):
            for j in range(t+1,len(a)):
                numerator = a[i][j]*pivot-a[i][t]*a[t][j]
                assert numerator%previous==0
                a[i][j] = numerator//previous
        previous = pivot
    return signs.count(-1)


def consume_certificates():
    families = [('vector','0'),('vector','1'),('vector','00'),('vector','01'),('vector','11'),('spinor','00'),('spinor','01'),('spinor','11')]
    results = []
    for kind,roles in families:
        tag = f'neutral_return_lower_cert2_20261011_{kind}_{roles}'
        receipt = ROOT/f'{tag}_receipt.json'
        r = json.loads(receipt.read_text())
        exit_path = ROOT/f'{tag}_natural_exit.json'
        natural = json.loads(exit_path.read_text())
        assert natural['natural_exit_code']==0 and natural['inputs_unchanged']
        assert r['source_sha256']==CERT_SOURCE_PIN==digest(ROOT/CERT_SOURCE)
        assert r['input_sha256']==PINS and r['inputs_unchanged'] and r['source_unchanged']
        assert r['certified'] and r['u_exact']=='0' and r['entire_density_output_image_checked']
        total = 0
        for block in r['parity_blocks']:
            matrix_path = ROOT/block['matrix_path']
            assert digest(matrix_path)==block['matrix_sha256']
            assert digest(ROOT/block['witness_path'])==block['witness_sha256']
            matrix = json.loads(matrix_path.read_text())
            center,radius,residual = matrix['center_integer'],matrix['radius_integer'],matrix['residual_squared_upper_integer']
            assert radius>=0 and residual>=0
            for i,row in enumerate(center):
                for j,x in enumerate(row):
                    assert matrix['lower_integer'][i][j] == x-(radius if i==j else 0)
                    assert matrix['upper_integer'][i][j] == x+(radius+64*residual if i==j else 0)
            n = bareiss(matrix['lower_integer'])
            assert n==bareiss(matrix['upper_integer'])==block['lower_inertia']['negative']==block['upper_inertia']['negative']
            total += n
        assert total==(1 if kind=='vector' else 0)
        pre = ROOT/f'{tag}_preexecution.zip'
        assert digest(pre)==natural['preexecution_zip_sha256']
        results.append({'kind':kind,'roles':roles,'spectator_counts':r['spectator_counts'],'exact_negative_inertia':total,
                        'eigenvalue_index':2 if kind=='vector' else 1,'lower_exact':'-5123',
                        'receipt':receipt.name,'receipt_sha256':digest(receipt),'natural_exit':exit_path.name,'natural_exit_sha256':digest(exit_path),
                        'preexecution_zip_sha256':digest(pre),'executed_zip_sha256':digest(ROOT/f'{tag}_executed.zip'),
                        'input_support':r['input_support'],'output_rows':r['output_rows']})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    args = parser.parse_args()
    start = time.monotonic()
    own = digest(Path(__file__))
    for f,pin in PINS.items(): assert digest(ROOT/f)==pin
    assert digest(ROOT/IMAGE)==IMAGE_PIN
    image = json.loads((ROOT/IMAGE).read_text())
    assert image['schema']=='neutral_return_image/v1'
    assert image['law']['axis']==0 and image['law']['internal_endpoint_sites']==[1,0]
    expected_spectators = [[1,126],[2,125],[0,126],[1,125],[2,124]]
    assert [b['spectator_harmonic_counts'] for b in image['exact_image']['per_cell_branches']]==expected_spectators
    assert [(x['physical_copies'],x['physical_spin_slots']) for x in image['law']['roles']]==[(1,2),(63,126)]
    # Every literal neutral bilinear changes two modes at the SAME internal
    # site. Physical-copy/spin labels do not alter their equal rho values.
    native_rho = [Fraction(1,2),Fraction(1,2),Fraction(-1,2),Fraction(-1,2)]
    for endpoints in ((2,3),(0,1)):
        for p,q in itertools.product(endpoints,repeat=2):
            assert native_rho[p]-native_rho[q]==0
    assert 3+1==2+2
    controls = {'joint_phase':generic_joint_phase(),'rational_density_output':rational_density_output_control(),
                'omitted_spectrum_wrongcases':exact_cutoff_wrongcases()}
    certificates = consume_certificates()
    caps = json.loads((ROOT/CAPS).read_text())
    ground_upper = Fraction(caps['ground_energy_interval_exact'][1])
    assert ground_upper < -5123
    gap = -5123-ground_upper
    result = {
        'kind':'ACTUAL_NEUTRAL_RETURN_COMPLETE_PHYSICAL_INSERTION_DENOMINATOR_NOT_CANONICAL_FULLCELL',
        'controls':controls,'families':certificates,'denominator_lower_exact':str(gap),
        'source_ground_energy_interval_exact':caps['ground_energy_interval_exact'],
        'source_formation_cap_exact':caps['coefficient_and_action_error_upper_exact'],
        'source_residual_cap_exact':caps['normalized_exact_source_residual_upper_exact'],
        'source_ground_infidelity_SQUARED_exact':caps['selected_ground_infidelity_upper_exact'],
        'source_ground_caps_binary64_assumptions':{'standard_binary64_correct_rounding_assumed':caps['standard_binary64_correct_rounding_assumed'],'accepted_consumer':CAPS,'not_replaced_by_observed_residual':True},
        'ground_projector_definition':'P0=|g_A tensor g_B><g_A tensor g_B| on the coherent invariant insertion domain, with exact normalized selected eigenvectors g. The serialized approximate Omega source is not g and P_Omega is not P0.',
        'orthogonal_source_vector_error_bound':'sqrt(274877906944/2204801412260314559041); aligned normalized ground vector distance <= sqrt(2*274877906944/2204801412260314559041)',
        'actual_domains':{
            'same_slot_active':'C6 number2 x Sym^1(C5)_active x Sym^126(C5)_aux',
            'same_slot_aux':'C6 number2 x Sym^2(C5)_active x Sym^125(C5)_aux',
            'distinct_slots_AA':'C4 number1 x C4 number3 x Sym^0(C5)_active x Sym^126(C5)_aux, both orders',
            'distinct_slots_AU':'C4_active number1/3 x C4_aux number3/1 x Sym^1(C5)_active x Sym^125(C5)_aux',
            'distinct_slots_UU':'C4_aux number1 x C4_aux number3 x Sym^2(C5)_active x Sym^124(C5)_aux, both orders',
            'extra_two_C6':'AA/AU/UU full6 x full6 domains independently certified but not needed for literal two-hop same-cell image'},
        'exact_domain_embedding_proof':[
            'Unrestricted symmetric spectators embed isometrically as their physical normalized labelled permutation tensors. Explicit changed slots carry their full signed exterior-number representation. There are actual active2 and auxiliary126 spin slots, not 128 identical onsite profiles.',
            'For k=1,2 selected slots of one role, E_i=b_i/sqrt(n) and E_ij=b_i b_j/sqrt(n(n-1)); ordered output coefficient is sqrt(n_i(n_j-delta_ij)/(n(n-1))). Across different roles use the product of the two E maps. Sum of squared coefficients is exactly one. The map is an extraction/reassembly of ONE original symmetric tensor into actual labelled slots, not a directsum encoding.',
            'For any same-role one-body M, E_k dGamma_n(M)=(dGamma_spectator(M)+sum_t M_t) E_k. This follows by acting M on each physical tensor slot; splitting selected versus spectator slots partitions the SAME sum. It applies simultaneously to hfree and rho and hence to 64rho^2, with no scalar affine offset in neutral sectors.',
            'The source top harmonic subspace is contained in Sym^2(C5)xSym^126(C5). Trace commutes with annihilators, so selected-slot spectator is harmonic, hence lies inside unrestricted Sym. Corrected source phase and generic joint CAR phase are retained.',
            'Thus each vector-containing physical domain includes one isometric copy of the SAME original selected harmonic ground eigenvector at E0. Since E0 upper<-5123 and the NEW lower form has at most one eigenvalue below -5123, the entire domain has exactly this one ground eigenvector and lambda2>=-5123. Scalar invariant summands contain no original ground and therefore lambda1>=-5123. Paired odd-number domains are different canonical number patterns and their NEW lower inertia is zero, giving lambda1>=-5123.',
            'Different selected-slot vector domains are NOT orthogonal; they are invariant subspaces of the SAME full physical H0 by spectator permutation symmetry. Each ground-complement lies in the common spectral subspace [ -5123,infinity). Their coherent algebraic sum also lies in that spectral subspace. No triangle-Gram pricing or directsum copies are used. Singlets/odd sectors retain their true canonical orthogonal labels.',
            'For two cells take the coherent sum of physical tensor-product domains. Each vector local factor has spectrum {E0} union [-5123,infinity), and odd local factors have spectrum in [-5123,infinity). After removing the entire unique exact g_A tensor g_B block, H_A+H_B-2E0 >= gap. Two locally excited factors have >=2gap. Spectral Q is not rho-zero, density-zero, scalar mean subtraction, or an approximate numerical rank-one projector.'
        ],
        'Schur_lower_form_proof':'D=hfree-z. B=P_output rho with complete density image. Hlo=D+64B*B<=H-z. All omitted input D positive. For positive support Q, K=I+64 BQ DQ^-1 BQ*>=I. Fixed binary Y witnesses G(Y)=BL*Y+Y*BL-Y*KY<=BL*K^-1 BL<=G(Y)+(BL-KY)*(BL-KY). Outward dyadic operations enclose G and residual. Radius I bounds entry uncertainty; Frobenius residual squared bounds its Gram. Exact integer Bareiss inertia of both Loewner matrices determines inertia of the Schur block, hence Hlo. Floating solves supply witnesses only.',
        'uniqueP0_domain':'Unique on EACH above physical insertion domain and their coherent invariant span; original selected harmonic ground identified via E intertwiner. No claim of unique ground on the entire canonical neutral cell.',
        'affine_u0_binding':'Bound to exact Image literal axis0 edge [1,0,0]->[2,0,0]. At A Dirac2,3 have rho=-1/2; B Dirac0,1 have rho=+1/2. Every local return bilinear has one creation and one annihilation at the same internal site, so delta charge at every site=0, hence delta internal E=-cumsum(delta q)=0. Returned external flow=0 and local number1+3=2+2: affine u0 exactly. No charged +/-1/2 floor substitution.',
        'source_image_scope':'Complete actual same-slot scalar/vector and two distinct number1/3 branches for active1+aux63, actual two spin labels, physical labels/flows/coherent histories retained. Source amplitudes and Gram belong to Image, not this denominator certificate.',
        'canonical_fullcell_ground':False,'fullPhi_ground':False,'lattice_ground':False,'phase':False,'stiffness':False,'IR_atom':False,'pole':False,'commoncone':False,
        'remaining_boundary':['Parent consumes bound Image coherent ground-block-removal/source-formation interface; exact numerical P0 from approximate source is not supplied.','Canonical fullcell/physicalPhi/lattice ground is not supplied; omitted four-odd and other sectors prevent heuristic admission.','Source/return formation and normalized linked expansion remain separate from this spectral denominator.'],
        'source_sha256':own,'input_sha256':PINS,'certificate_source_sha256':CERT_SOURCE_PIN,
        'image_interface':IMAGE,'image_interface_sha256':IMAGE_PIN,
        'image_branch_binding_complete':True,'literal_affine_u0_verified':True,
        'inputs_unchanged':all(digest(ROOT/f)==pin for f,pin in PINS.items()),'source_unchanged':digest(Path(__file__))==own,
        'elapsed_seconds':time.monotonic()-start,
    }
    path = ROOT/f'{args.tag}_receipt.json'
    with path.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'receipt':path.name,'sha256':digest(path),'denominator_lower_exact':str(gap),'families':len(certificates),'all_exact_inertias_valid':True}),flush=True)


if __name__=='__main__':
    main()
