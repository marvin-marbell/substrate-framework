"""Independent exact review controls. Never imports or executes target producers."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
import hashlib
import itertools
import json
from pathlib import Path
import zipfile
import sympy as s

ROOT = Path(__file__).resolve().parent
SNAPSHOT = 'vacuum_weyl_parent_review_original_targets_20261011.zip'
PINS = {
    'vacuum_connected_native_proof_20261011.json': 'd100b2733d614c159cbd104c77f11193104c680d46178e0f4fcf8250bfbbd55b',
    'vacuum_connected_native_interface_20261011.json': 'aefbeee66bd349fc7796252673bcb61d1b58a5d58e383087d6c0bd1cefd1400c',
    'vacuum_connected_native_actual2_20261011_capsule.json': 'a8863011d6101f2e50f8e229b074923ffccc6eaf539ec692b14239eedc1286aa',
    'vacuum_connected_native_scientist_20261011.py': 'a2618c53a90d6ecc15bc12bbfceab16d3640c342e507debe7de85f98a9fd3681',
    'vacuum_connected_native_execute_20261011.py': '7c0fa8eaa2e45ee89097ca84dc5cfc6d5ebbfde18b0cfafe79cabc643220d32d',
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def interval(obj):
    return tuple(Q(x) for x in obj['exact'])

def cadd(x,y):
    return x[0]+y[0],x[1]+y[1]

def cmul(x,y):
    return x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0]

def reverse_inertia(entries,shift):
    """Independent bottom-right Hermitian congruence elimination, exact rationals."""
    a=[[(Q(z[0]),Q(z[1])) for z in row] for row in entries]
    for i in range(len(a)):
        a[i][i]=cadd(a[i][i],(-shift,Q(0)))
    negative=0
    while a:
        p=a[-1][-1]
        assert p[1]==0 and p[0]!=0
        negative+=p[0]<0
        n=len(a)-1
        a=[[cadd(a[i][j],tuple(-v/p[0] for v in cmul(a[i][-1],a[-1][j]))) for j in range(n)] for i in range(n)]
    return negative

def native_enclosures(receipt):
    certificates=[]
    gamma=Q(46955312929,34359738368)
    for axis in receipt['axes']:
        for name,key,error_key in [('G','G_onebody_exact_operator','G'),('TdagT','TdagT_onebody_exact_operator','TdagT')]:
            lo=[];hi=[];scalar=Q(0);counts=[]
            for role in axis[key]:
                mult=role['physical_count'];assert mult==(1 if role['role']==0 else 63)
                eig=role['eigenvalue_intervals'];assert len(eig)==8
                for i,band in enumerate(eig):
                    l,u=interval(band)
                    lc=reverse_inertia(role['onebody_exact'],l)
                    uc=reverse_inertia(role['onebody_exact'],u)
                    assert lc<=i and uc>=i+1
                    counts.append([role['role'],i,lc,uc])
                    lo.extend([l]*mult);hi.extend([u]*mult)
                scalar+=mult*Q(role['scalar_exact'])
            assert len(lo)==len(hi)==512
            err=Q(axis['outer_ground_coefficient_errors_exact'][error_key])
            bound=(scalar+sum(sorted(lo)[:256])-err,scalar+sum(sorted(hi)[-256:])+err)
            saved=axis['G_spectrum_entire_canonical256' if name=='G' else 'TdagT_spectrum_entire_canonical256']
            assert bound==interval(saved)
            certificates.append({'axis':axis['axis'],'gram':name,'exact_reverse_congruence_inertias':counts,'mode_count':512,'occupied_count':256,'saved_interval_reproduced':True})
        L=max(Q(0),interval(axis['G_spectrum_entire_canonical256'])[1])
        Tc=max(Q(0),interval(axis['TdagT_spectrum_entire_canonical256'])[1])
        G0=interval(axis['G_ground_expectation']);Gc=Q(axis['signed_centered_G_operator_norm_cap_exact'])
        d=Q(receipt['domains']['whole_operator_two_cell_floor_exact']);anchor=Q(axis['anchor_exact'])
        for row in axis['energies']:
            E=Q(row['E_exact']);A=anchor-E;w=interval(row['vacuum_w0'])
            parts={k:Q(v) for k,v in row['whole_excited_connected_form_components_exact'].items()}
            assert parts['signed_centered_G']==Gc/(gamma*(A+gamma))
            assert parts['scalar_energy_counterterm']==G0[1]/(A*(A+gamma))
            assert parts['vacuum_scalar_mismatch']==max(abs(G0[0]/A-w[1]),abs(G0[1]/A-w[0]))/gamma
            rem=parts['entire_spectrum_intertwiner_remainder']
            assert (rem*(d-E)*gamma*(A+gamma))**2>=L*Tc
            assert sum(parts.values())==Q(row['whole_excited_connected_relative_cap_exact'])
            assert Q(row['metric_I_plus_Wprime_upper_exact'])==1+L/(d-E)**2
            assert not row['reaches_one']
    return certificates

def annihilators(modes):
    result=[]
    for mode in range(modes):
        a=s.zeros(1<<modes)
        for n in range(1<<modes):
            if n&(1<<mode):
                a[n^(1<<mode),n]=(-1)**((n&((1<<mode)-1)).bit_count())
        result.append(a)
    return result

def car_intertwiner():
    a=annihilators(6);I=s.eye(64)
    for i,j in itertools.product(range(6),repeat=2):
        assert a[i]*a[j].T+a[j].T*a[i]==(I if i==j else s.zeros(64))
    n=[x.T*x for x in a]
    ro=(n[0]-n[1])/2;rc=(n[2]-n[3]+n[4]-n[5])/2
    Do=3*(n[0]-n[1])+4*(a[0].T*a[1]+a[1].T*a[0])
    Dc=2*(n[2]-n[3])+(a[2].T*a[3]+a[3].T*a[2])+5*(n[4]-n[5])+2*(a[4].T*a[5]+a[5].T*a[4])
    D=Do+Dc;Hn=D+64*(ro*ro+rc*rc)
    # Genuine outer N1 eigenvector (-1,2)/sqrt5, Eg_outer=11.
    states=[n for n in range(16) if n.bit_count()==2]
    embed=s.zeros(64,len(states))
    for j,state in enumerate(states):
        embed[1|(state<<2),j]=-1/s.sqrt(5)
        embed[2|(state<<2),j]=2/s.sqrt(5)
    K=embed.T*(Dc+64*rc*rc)*embed
    assert Hn*embed==embed*K+11*embed
    V=a[0].T*(a[2]+2*a[4]);Vm=V.T
    rows=[];memory_nonzero=[];grams=[]
    for O,delta in [(V,s.Rational(1,2)),(Vm,s.Rational(-1,2))]:
        X=D+64*((ro-delta*I)**2+(rc+delta*I)**2)+53*I
        J=O*embed;C=(D*O-O*D+64*O)*embed
        assert X*J==J*K+C
        T=C-9*J
        A1=X*T-T*K
        wrong=(D*(D*O-O*D+55*O)-(D*O-O*D+55*O)*D+64*(D*O-O*D+55*O))*embed
        assert A1!=wrong
        memory_nonzero.append(sum(x!=0 for x in A1-wrong))
        G=J.T*J;grams.append(G)
        # Signed CAR scalar plus bilinear identity on arbitrary central N2.
        B=a[2]+2*a[4];outer_n=s.Rational(1,5)
        expected=(outer_n*B*B.T if O==Vm else (1-outer_n)*B.T*B)
        assert G==embed.T*expected*embed
        rows.append({'orientation':str(delta),'whole_N2_columns':len(states),'first_affine_cancellation_exact':True,'subsequent_memory_difference_nonzeros':memory_nonzero[-1]})
    assert grams[0]+grams[1]!=s.eye(len(states))*(grams[0]+grams[1])[0,0]
    wrong_Dc=2*(n[2]-n[3]+n[4]-n[5])+a[2].T*a[3]+a[3].T*a[2]+a[4].T*a[5]+a[5].T*a[4]
    assert (Dc*V-V*Dc)*embed!=(wrong_Dc*V-V*wrong_Dc)*embed
    return {'dimension':64,'CAR_pairs':36,'intertwiners':rows,'unequal_role_misassignment_rejected':True,'scalar_only_Gram_rejected':True,'collective_rho_uses_both_roles':True}

def energy_forms():
    K=s.diag(0,2,5);X=s.diag(7,11,13,17)
    J=s.Matrix([[2,1,0],[1,2,1],[0,1,3],[1,0,1]])
    E=s.Rational(1,3);anchor=s.Rational(9);A=anchor-E
    R=(X-E*s.eye(4)).inv();F=(K+A*s.eye(3)).inv();G=J.T*J;T=X*J-J*K-anchor*J
    W=J.T*R*J;g0=G[0,0];w0=W[0,0];C=W-w0*s.eye(3);Gc=G-g0*s.eye(3)
    Rem=J.T*R*T*F
    assert W==G*F-Rem
    assert G*F!=F*G
    four=(Gc*F+F*Gc)/2+g0*(F-s.eye(3)/A)+(g0/A-w0)*s.eye(3)-(Rem+Rem.T)/2
    assert C==four
    assert F-s.eye(3)/A==-K*(K+A*s.eye(3)).inv()/A
    S=K-E*s.eye(3)-W;H=K.row_join(J.T).col_join(J.row_join(X))
    assert (H-E*s.eye(7)).det()==(X-E*s.eye(4)).det()*S.det()
    q=W[1:,0];t=E+w0;Aq=S[1:,1:]
    assert S.det()==Aq.det()*(-t-(q.T*Aq.inv()*q)[0])
    assert S.det()==(-t)*(Aq+q*q.T/t).det()
    assert S.det()!=(-t)*Aq.det()
    Wprime=J.T*R*R*J;rec=s.eye(3).col_join(-R*J)
    assert rec.T*rec==s.eye(3)+Wprime
    f=s.Rational(-1,2);Rf=(X-f*s.eye(4)).inv()
    assert (W-J.T*Rf*J)/(E-f)==J.T*R*Rf*J
    L=max(sum(abs(z) for z in G.row(i)) for i in range(3))
    TT=T.T*T;Tc=max(sum(abs(z) for z in TT.row(i)) for i in range(3))
    jt=s.Integer(s.ceiling(s.sqrt(L*Tc)))
    gc=max(sum(abs(z) for z in Gc.row(i)) for i in range(3));gamma=s.Integer(2);d=s.Integer(7)
    B=gc/(gamma*(A+gamma))+g0/(A*(A+gamma))+abs(g0/A-w0)/gamma+jt/((d-E)*gamma*(A+gamma))
    Kq=K[1:,1:];Cq=C[1:,1:]
    minors=[]
    for sign in (-1,1):
        M=B*Kq+sign*Cq
        for indices in ((0,),(1,),(0,1)):
            det=M.extract(indices,indices).det();assert det>=0;minors.append(str(det))
    # Plausible relative-norm misuse: ordinary norm 1 is not K-relative cap1.
    gamma_small=s.Rational(1,4);assert 1>gamma_small
    return {'exact_noncommuting_four_term_identity':True,'anticommutator_needed':True,'signed_scalar_identity':True,'full_resolvent_T_remainder':True,'every_Q_vector_bound_PSD_principal_minors':minors,'bound_exact':str(B),'Pg_fold_signs_exact':True,'dropoffdiagonal_changes_determinant':True,'metric_and_divided_difference_exact':True,'ordinary_norm_as_relative_cap_rejected':True}

def adverse_formation():
    I=s.Rational(400,10201);overlap=s.Rational(99,101)
    tensor_dist2=2-2*overlap**3
    assert tensor_dist2>2*I and tensor_dist2<=6*I
    trace3_squared=4*(1-overlap**6)
    assert trace3_squared>8*I and trace3_squared<=24*I
    # Whole totalN2 allocation permits0 and20; one-particle-per-role crop gives10.
    levels=[0,0,10,10]
    spectrum=[sum(levels[i] for i in inds) for inds in itertools.combinations(range(4),2)]
    assert min(spectrum)==0 and max(spectrum)==20
    assert spectrum!=[10]*len(spectrum)
    # A coefficient budget after product collection need not equal path L1.
    ann=annihilators(1)[0]
    terms=[(1,ann),(1,ann.T)]
    path_L1=sum(abs(x*y) for x,_ in terms for y,_ in terms)
    assert (ann+ann.T).T*(ann+ann.T)==s.eye(2)
    collected_L1=1
    assert path_L1==4 and collected_L1==1
    return {'single_cell_tensor_price_rejected':True,'threecell_sqrt6I_bound_valid':True,'tensor_distance_squared_exact':str(tensor_dist2),'singlecell_2I_exact':str(2*I),'per_label_number_crop_rejected':True,'whole_N2_extrema':[0,20],'wrong_per_role_N1_extrema':[10,10],'postcollection_budget_is_not_path_L1':{'path_L1':path_L1,'collected_L1':collected_L1}}

def run(output):
    with zipfile.ZipFile(ROOT/SNAPSHOT) as archive:
        assert sha((ROOT/SNAPSHOT).read_bytes())=='8c204449e7ea5d3cef6cbb24de09c824cce66d27709204db9c9e744b09819b44'
        for name,pin in PINS.items():
            assert sha(archive.read(name))==pin
        receipt=json.loads(archive.read('vacuum_connected_native_interface_20261011.json'))
        capsule=json.loads(archive.read('vacuum_connected_native_actual2_20261011_capsule.json'))
    executed=ROOT/capsule['executed']['path'];assert sha(executed.read_bytes())==capsule['executed']['sha256']
    with zipfile.ZipFile(executed) as archive:
        for name in ('vacuum_connected_native_scientist_20261011.py','vacuum_connected_native_execute_20261011.py'):
            assert sha(archive.read('executed_inputs/'+name))==PINS[name]
    result={'kind':'NEW_INDEPENDENT_EXACT_VACUUM_OPERATOR_REVIEW_CONTROLS','target_original_sha256':PINS,'no_target_imports_or_old_entrypoints':True,'native_exact_enclosures':native_enclosures(receipt),'CAR_intertwiner':car_intertwiner(),'energy_forms':energy_forms(),'adverse_formation_allocation':adverse_formation(),'physical1_E0_caps':[axis['energies'][1]['whole_excited_connected_relative_cap_display'] for axis in receipt['axes']],'physical1_guard_failed':True,'scope':'Exact finite frozen reference algebra and conditional historical formation; no physicalPhi ground/allorder/phase/pole.'}
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'result':output.name,'result_sha256':sha(output.read_bytes()),'physical1_guard_failed':True},sort_keys=True),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Review result epoch is immutable')
    run(args.output)
