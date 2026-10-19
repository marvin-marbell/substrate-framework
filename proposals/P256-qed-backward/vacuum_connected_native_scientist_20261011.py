"""New entire-canonical energy-dependent vacuum-connected cell Schur calculation.
Frozen helpers are imported as libraries only; no historical entrypoint executes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import sympy as sp
import central_neutral_vertex_scientist_20261011 as car

ROOT = Path(__file__).resolve().parent
PREFIX = 'vacuum_connected_native_'
ZERO, ONE = car.ZERO, car.ONE
frozen = car.frozen


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def dagger(terms):
    return [((z[0], -z[1]), tuple((c,s,p,not cr) for c,s,p,cr in reversed(w)), m)
            for z,w,m in terms]


def projected_product(kernel, left, right, outer, role, price, beta):
    """Exact signed CAR collection, scalar kept, arbitrary central labelled N.

    Different outer label returns vanish by the exact outer Nj=2 reducing state.
    NO central number relation, harmonic projection, spin crop or Pg deletion.
    """
    symbolic = defaultdict(lambda: ZERO)
    raw = F(0)
    pair_l1 = sum((car.norm(z) for z,_,_ in left), F(0)) * sum((car.norm(z) for z,_,_ in right), F(0))
    for z,w,m in car.multerms(dagger(left), right):
        if any(m):
            raise ArithmeticError('This Gram accepts no hidden rho memory')
        raw += car.norm(z)
        sign = (-1)**sum(w[i][0] > w[j][0] for i in range(len(w)) for j in range(i+1,len(w)))
        ow = tuple((s,p,cr) for c,s,p,cr in w if c == outer)
        cw = tuple(x for x in w if x[0] == 1)
        balance = defaultdict(int)
        for s,p,cr in ow:
            balance[s] += 1 if cr else -1
        if any(balance.values()):
            continue
        symbolic[cw,ow] = car.ca(symbolic[cw,ow], car.cs(z,sign))
    out = defaultdict(lambda: ZERO)
    l1 = F(0)
    for (cw,ow),z in symbolic.items():
        if z == ZERO:
            continue
        out[cw] = car.ca(out[cw], car.cm(z, car.dec(kernel.local_memory(ow,0,0))))
        l1 += car.norm(z)
    terms = car.collect([(z,w,(0,0,0)) for w,z in out.items() if z != ZERO])
    if left == right:
        # The exact contracted Gram is Hermitian. Its Hermitian proxy has no
        # greater error than the existing coefficient enclosure (contractivity
        # of A -> (A+A*)/2); no measured antisymmetry is silently discarded.
        terms = car.collect(car.scale(terms,F(1,2))+car.scale(dagger(terms),F(1,2)))
    if role:
        terms = car.reindex(terms,-role)
    formation_l1 = max(pair_l1, raw)
    error = l1*price + formation_l1*beta
    return terms,error,{'pre_normal_order_term_pair_L1_exact':str(pair_l1),
                        'post_normal_order_pre_outer_collection_L1_exact':str(raw),
                        'formation_L1_exact':str(formation_l1),'signed_outer_L1_exact':str(l1),
                        'whole_central_labelled_number_unrestricted':True,'scalar_retained':True}


def inertia(matrix, shift):
    """Exact unpivoted Hermitian LDL; a zero pivot requests a wider bracket."""
    n = len(matrix)
    L = [[ZERO for _ in range(n)] for _ in range(n)]
    pivots = []
    for i in range(n):
        p = car.ca(matrix[i][i],(-shift,F(0)))
        for j in range(i):
            p = car.ca(p, car.cs(car.cm(L[i][j],(L[i][j][0],-L[i][j][1])),-pivots[j]))
        if p[1] or not p[0]:
            raise ArithmeticError('Nonreal or zero exact LDL pivot')
        pivots.append(p[0])
        L[i][i] = ONE
        for k in range(i+1,n):
            v = matrix[k][i]
            for j in range(i):
                v = car.ca(v,car.cs(car.cm(L[k][j],(L[i][j][0],-L[i][j][1])),-pivots[j]))
            L[k][i] = car.cs(v,1/p[0])
    return sum(p < 0 for p in pivots)


def quadratic(terms):
    h = [[ZERO for _ in range(8)] for _ in range(8)]
    scalar = ZERO
    for z,w,m in terms:
        if any(m):
            raise ArithmeticError('Quadratic price cannot delete totalrho')
        if not w:
            scalar = car.ca(scalar,z)
        elif len(w)==2 and w[0][3] and not w[1][3] and all(c==1 for c,s,p,cr in w):
            i,j = 4*w[0][1]+w[0][2],4*w[1][1]+w[1][2]
            h[i][j] = car.ca(h[i][j],z)
        else:
            raise ArithmeticError(('Not a fullCAR quadratic',w))
    if scalar[1] or any(h[i][j] != (h[j][i][0],-h[j][i][1]) for i in range(8) for j in range(8)):
        raise ArithmeticError('Exact decoded quadratic is not Hermitian')
    approximate = np.linalg.eigvalsh(np.array([[car.floating(z) for z in r] for r in h]))
    intervals = []
    for i,x in enumerate(approximate):
        center = F(float(x)); radius = F(1,1<<30)
        for attempt in range(48):
            lo,hi = center-radius,center+radius
            try:
                lower_count,upper_count = inertia(h,lo),inertia(h,hi)
            except ArithmeticError:
                radius *= 2
                continue
            if lower_count <= i and upper_count >= i+1:
                intervals.append((lo,hi))
                break
            radius *= 2
        else:
            raise ArithmeticError('Cannot certify entire onebody spectrum')
    return scalar[0],intervals,h


def canonical_spectrum(roles, errors):
    """All 512 central modes and total N256; no per-label N2 constraint."""
    scalar = F(0); lower=[]; upper=[]; payload=[]
    for role,mult in ((0,1),(1,63)):
        c,e,h = quadratic(roles[role])
        scalar += mult*c
        lower.extend([v[0] for v in e]*mult)
        upper.extend([v[1] for v in e]*mult)
        payload.append({'role':role,'physical_count':mult,'scalar_exact':str(c),
                        'onebody_exact':[[[str(x) for x in z] for z in row] for row in h],
                        'eigenvalue_intervals':[frozen.encode(v) for v in e]})
    error = errors[0]+63*errors[1]
    lo = scalar+sum(sorted(lower)[:256])-error
    hi = scalar+sum(sorted(upper)[-256:])+error
    return (lo,hi),payload,error


def expectation(kernel, roles, price):
    value = 0j; budget=F(0)
    for role,mult in ((0,1),(1,63)):
        terms=car.reindex(roles[role],role)
        value += mult*kernel.gram_terms([(1.,(),(0,0,0))],[(car.floating(z),w,m) for z,w,m in terms])
        budget += mult*sum(car.norm(z) for z,w,m in terms)
    if abs(value.imag)>1e-9:
        raise ArithmeticError('Nonreal Hermitian expectation')
    return frozen.interval(value.real,price*budget)


def source_moments(kernel, pairs, price):
    centers=[0.,0.,0.]; budgets=[F(0),F(0),F(0)]
    for role,mult,first,action in pairs:
        for i,(left,right) in enumerate(((first,first),(first,action),(action,action))):
            lv=[(car.floating(z),w,m) for z,w,m in left]
            rv=[(car.floating(z),w,m) for z,w,m in right]
            v=kernel.gram_terms(lv,rv)
            if abs(v.imag)>1e-7*max(1.,abs(v.real)):
                raise ArithmeticError('Nonreal native charged moment')
            centers[i] += mult*v.real
            budgets[i] += mult*sum(car.norm(z) for z,w,m in left)*sum(car.norm(z) for z,w,m in right)
    return [frozen.interval(v,price*b) for v,b in zip(centers,budgets)],centers,budgets


def exact_controls():
    """Independent exact Schur, fullCAR, multiplicity and totalrho controls."""
    K=sp.diag(0,sp.Rational(3,2)); X=sp.diag(3,5,7)
    J=sp.Matrix([[6,0],[0,0],[sp.Rational(1,2),sp.Rational(1,2)]])
    E=sp.symbols('E'); R=(X-E*sp.eye(3)).inv(); W=J.T*R*J
    w0=W[0,0]; C=W-w0*sp.eye(2); S=K-E*sp.eye(2)-W
    H=K.row_join(J.T).col_join(J.row_join(X))
    if sp.simplify((H-E*sp.eye(5)).det()-(X-E*sp.eye(3)).det()*S.det()) != 0:
        raise ArithmeticError('Independent exact Schur determinant failed')
    if sp.simplify(S-(K-(E+w0)*sp.eye(2)-C)) != sp.zeros(2):
        raise ArithmeticError('Vacuum normalization failed')
    a=sp.Rational(9); Fk=(K+(a-E)*sp.eye(2)).inv(); T=X*J-J*K-a*J
    if sp.simplify(W-(J.T*J*Fk-J.T*R*T*Fk)) != sp.zeros(2):
        raise ArithmeticError('Exact energy intertwiner failed')
    if sp.simplify(sp.diff(W,E)-J.T*R*R*J) != sp.zeros(2):
        raise ArithmeticError('Derivative metric failed')
    wrong_metric=sp.eye(2)-sp.diff(W,E)
    assert wrong_metric.subs(E,0)[0,0]<0
    reconstruction=(-R*J).col_join(sp.eye(2))
    assert sp.simplify(reconstruction.T*reconstruction-(sp.eye(2)+sp.diff(W,E)))==sp.zeros(2)
    # Deliberately equal diagonal W: Q C Q=0, nonzero Pg/Qg coupling.
    Je=sp.Matrix([[2,2],[0,0],[1,1]]); We=Je.T*R*Je; Ce=We-We[0,0]*sp.eye(2)
    assert Ce[1,1]==0 and Ce[0,1]!=0
    assert sp.simplify((K-E*sp.eye(2)-We).det()-(-E-We[0,0])*(K[1,1]-E-We[0,0])) != 0
    assert W.subs(E,sp.Rational(1,4)) != W.subs(E,0)
    # Independent CAR matrices, not the symbolic normal-ordering helper.
    ann=[]
    for mode in range(4):
        A=sp.zeros(16)
        for n in range(16):
            if n & (1<<mode):
                A[n^(1<<mode),n]=(-1)**((n & ((1<<mode)-1)).bit_count())
        ann.append(A)
    assert all(ann[i]*ann[j].T+ann[j].T*ann[i]==(sp.eye(16) if i==j else sp.zeros(16)) for i in range(4) for j in range(4))
    c0,c1,c2,c3=ann; rho=(c0.T*c0+c1.T*c1-c2.T*c2-c3.T*c3)/2
    O=c2.T*c0; delta=sp.Rational(-1)
    assert rho*O-O*rho==delta*O
    u=sp.Rational(1,2)
    lhs=64*((rho+u*sp.eye(16))**2*O-O*rho**2)
    rhs=64*(delta+u)**2*O+128*(delta+u)*O*rho
    assert lhs==rhs and lhs != 64*(delta+u)**2*O
    # A1+U63 is an operator sum over copies, NOT A+U or 64A.
    A=sp.diag(1,3); U=sp.diag(2,5)
    exact_trace=A.trace()+63*U.trace()
    assert exact_trace != (A+U).trace() and exact_trace != 64*A.trace()
    return {'exact_schur_determinant':True,'exact_intertwining_identity':True,
            'scalar_only_and_drop_offdiagonal_rejected':True,'zero_QC_Q_nonzero_PQ':True,
            'frozen_energy_rejected':True,'wrong_derivative_sign_rejected':True,
            'independent_fullCAR_dimension':16,'totalrho_deletion_rejected':True,
            'wrong_1plus63_and_64active_rejected':True,
            'equal_diagonal_connected_matrix':[[str(x) for x in row] for row in Ce.tolist()],
            'metric_exact':[[str(x) for x in row] for row in (sp.eye(2)+sp.diff(W,E)).tolist()],
            'control_only_not_native_proof':True}


def run(epoch):
    started=time.monotonic()
    request=json.loads((ROOT/f'{PREFIX}{epoch}_request.json').read_bytes())
    for name,pin in request['input_sha256'].items():
        if sha(ROOT/name)!=pin:
            raise ArithmeticError(('Pin mismatch',name))
    iface=json.loads((ROOT/'central_neutral_vertex_interface_20261011.json').read_bytes())
    if iface['controls']['actual_unequal_role_commutator'][1]['wrong_template_active_h_difference']<=0:
        raise ArithmeticError('Corrected actual2 auxiliary profile discriminator absent')
    boundary=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
    _,_,_,exact=frozen.formation.derive_physical(boundary)
    matrices={}
    for axis in range(3):
        M=np.array([[complex(frozen.formation.sp.sympify(z)) for z in r] for r in exact[axis]])
        for s,t in itertools.product(range(2),repeat=2):
            matrices[axis,s,t]=(-1)**(s+t)*M
    inventory,_=frozen.native.inventory(8)
    edges=[next(e for e in inventory if e['axis']==a and e['central_site']==0 and e['central_is_tail']) for a in (1,2)]
    kernel=frozen.DynamicKernel(); star=frozen.Star(kernel,edges,matrices); central=car.Central(kernel,star)
    gamma=F(46955312929,34359738368); cap=F(1,65536)
    infidelity=(cap/gamma)**2
    if infidelity!=F(274877906944,2204801412260314559041):
        raise ArithmeticError('Accepted cap-to-ground-distance mismatch')
    eta=frozen.sqrt_hi(2*infidelity)
    density=json.loads((ROOT/'three_cell_density_formation_typed3_20261011_interface.json').read_bytes())
    delta=max(F(r['delta_trace_norm_exact']) for r in density['errors'].values())
    beta=2500*frozen.formation.gamma(8192)+frozen.formation.gamma(1<<24)+128*frozen.formation.gamma(256)
    price=delta+beta+2*eta
    # Charged moments are on a tensor-product cell ground, not one cell.
    # Reuse the accepted three-cell extraction bound, conservative for two.
    eta3=frozen.sqrt_hi(6*infidelity)
    beta3=2500*frozen.formation.gamma(8192)+frozen.formation.gamma(1<<28)+128*frozen.formation.gamma(256)
    source_price=3*delta+beta3+2*eta3
    Egupper=F(-176071894972193,34359738368)
    b=64-10240-2*Egupper; d=64-5120-5155-2*Egupper
    assert d==F(iface['excited_transfer']['uniform_operator_single_floor_exact'])
    axes=[]
    for edge in range(2):
        anchor=car.CANDIDATES[edge]
        grams={name:[] for name in ('G','Tgram','JC')}; errors={name:[] for name in grams}
        decomposition=[]; pairs=[]; rho_memory=[]
        for role,mult in ((0,1),(1,63)):
            combined={name:[] for name in grams}; combined_error={name:F(0) for name in grams}
            for orientation in (-1,1):
                first=[(car.dec(z),w,(0,0,0)) for z,w in star.root(edge,orientation,role)]
                flow,u=star.flow_u((1,edge,orientation,1-edge,0))
                C=central.action(first,flow,u)
                if any(any(m) for z,w,m in C):
                    raise ArithmeticError('Fixed endpoint electric cancellation fails')
                T=car.collect(C+car.scale(first,-anchor))
                outer=0 if edge==0 else 2
                for name,left,right in (('G',first,first),('Tgram',T,T),('JC',first,C)):
                    terms,error,account=projected_product(kernel,left,right,outer,role,price,beta)
                    combined[name]+=terms;combined_error[name]+=error
                    decomposition.append({'role':role,'count':mult,'orientation':orientation,'operator':name,
                                          'projection':account})
                pairs.append((role,mult,first,C))
                next_action=central.action(T,flow,u)
                rho_memory.append({'role':role,'orientation':orientation,
                                   'intertwiner_T_terms':len(T),'next_commutator_terms':len(next_action),
                                   'TOTALrho_terms':sum(any(m) for z,w,m in next_action),
                                   'TOTALrho_cap_exact':'128',
                                   'T_exact':[{'coefficient_exact':[str(x) for x in z],'word':w} for z,w,m in T],
                                   'next_action_exact':[{'coefficient_exact':[str(x) for x in z],'word':w,'TOTALrho_right_powers':m} for z,w,m in next_action]})
            for name in grams:
                grams[name].append(car.collect(combined[name]));errors[name].append(combined_error[name])
        Gspec,Gpayload,Gerror=canonical_spectrum(grams['G'],errors['G'])
        Tspec,Tpayload,Terror=canonical_spectrum(grams['Tgram'],errors['Tgram'])
        G0=expectation(kernel,grams['G'],price)
        G0=(max(F(0),G0[0]-Gerror),G0[1]+Gerror)
        Gcenter=max(abs(Gspec[0]-G0[1]),abs(Gspec[1]-G0[0]))
        L=max(F(0),Gspec[1]); Tcap=max(F(0),Tspec[1])
        JT=frozen.sqrt_hi(L*Tcap)
        moments,centers,budgets=source_moments(kernel,pairs,source_price)
        # Polynomial is a diagnostic/ground-column candidate ONLY. It is never K^-1.
        energy_rows=[]
        for E in (F(-1),F(0),gamma/4):
            ae=anchor-E;de=d-E;be=b-E
            M0=(max(F(0),moments[0][0]),moments[0][1])
            M1=frozen.add(moments[1],frozen.scale(M0,-E))
            variance=frozen.add(moments[2],frozen.add(frozen.scale(moments[1],-2*anchor),frozen.scale(M0,anchor**2)))
            varhi=max(F(0),variance[1])
            poly=frozen.add(frozen.scale(M0,2/ae+E/ae**2),frozen.scale(moments[1],-1/ae**2))
            wlo=M0[0]**2/M1[1]
            whi=min(M0[1]/be,poly[1]+varhi/(ae**2*be))
            if wlo>whi:
                raise ArithmeticError('Whole-spectrum vacuum interval empty')
            w0=(wlo,whi)
            # Entire Qg form: signed G-G0I, scalar energy counterterm,
            # exact resolvent remainder and exact-vacuum scalar mismatch.
            signed=Gcenter/(gamma*(ae+gamma))
            scalar=G0[1]/(ae*(ae+gamma))
            mismatch=max(abs(G0[0]/ae-w0[1]),abs(G0[1]/ae-w0[0]))/gamma
            remainder=JT/(de*gamma*(ae+gamma))
            connected=signed+scalar+mismatch+remainder
            # Retain Pg/Qg; ground-column polynomial error is separately paid.
            roles=[];q_outer_error=F(0)
            for role,mult in ((0,1),(1,63)):
                roles.append(car.collect(car.scale(grams['G'][role],2/ae+E/ae**2)+car.scale(grams['JC'][role],-1/ae**2)))
                q_outer_error+=mult*((2/ae+abs(E)/ae**2)*errors['G'][role]+errors['JC'][role]/ae**2)
            # Gram uses the complete correlated fourCAR fields, not copy factorization.
            p=expectation(kernel,roles,price)
            q2=0.;qbudget=F(0)
            for l,r,mult in ((0,0,1),(1,1,63),(0,1,63),(1,0,63),(1,2,3906)):
                lt=car.reindex(roles[int(l>0)],l);rt=car.reindex(roles[int(r>0)],r)
                q2+=mult*kernel.gram_terms([(car.floating(z),w,m) for z,w,m in lt],[(car.floating(z),w,m) for z,w,m in rt]).real
                qbudget+=mult*central.norms(lt)*central.norms(rt)
            p2=car.square_interval(p)
            qpoly=frozen.sqrt_hi(max(F(0),F(q2)+price*qbudget-p2[0]))
            source_inverse_error=frozen.sqrt_hi(varhi)*max(F(1),ae/be-1)/ae**2
            qcap=qpoly+frozen.sqrt_hi(L)*source_inverse_error+2*q_outer_error
            metric=L/de**2
            # General folded term has a denominator, NEVER assumed nonzero/positive.
            t=frozen.add((E,E),w0)
            denominator=min(abs(t[0]),abs(t[1])) if not t[0]<=0<=t[1] else F(0)
            fold=qcap**2/(denominator*gamma) if denominator else None
            energy_rows.append({'E_exact':str(E),'physical_hopping_exact':'1','vacuum_w0':frozen.encode(w0),
                                'whole_excited_connected_form_components_exact':{
                                    'signed_centered_G':str(signed),'scalar_energy_counterterm':str(scalar),
                                    'vacuum_scalar_mismatch':str(mismatch),'entire_spectrum_intertwiner_remainder':str(remainder)},
                                'whole_excited_connected_relative_cap_exact':str(connected),
                                'whole_excited_connected_relative_cap_display':float(connected),
                                'whole_excited_connected_paid_conditionally':True,'reaches_one':connected<1,
                                'Qg_W_Pg_norm_cap_exact':str(qcap),'q_is_not_entire_operator_certificate':True,
                                'folded_rank1_relative_cap_exact':str(fold) if fold is not None else None,
                                'fold_denominator_E_plus_w0':frozen.encode(t),
                                'metric_I_plus_Wprime_upper_exact':str(1+metric),
                                'derivative_metric_positive':True,
                                'polynomial_f_is_NOT_inverse':True,'whole_spectral_polynomial_source_error_exact':str(source_inverse_error)})
        axes.append({'axis':edges[edge]['axis'],'anchor_exact':str(anchor),
                     'G_spectrum_entire_canonical256':frozen.encode(Gspec),
                     'G_ground_expectation':frozen.encode(G0),
                     'signed_centered_G_operator_norm_cap_exact':str(Gcenter),
                     'TdagT_spectrum_entire_canonical256':frozen.encode(Tspec),
                     'G_onebody_exact_operator':Gpayload,'TdagT_onebody_exact_operator':Tpayload,
                     'outer_ground_coefficient_errors_exact':{'G':str(Gerror),'TdagT':str(Terror)},
                     'native_charged_moments':[frozen.encode(v) for v in moments],
                     'native_charged_moment_centers':centers,'projection_accounting':decomposition,
                     'fullCAR_TOTALrho_intertwiner':rho_memory,'energies':energy_rows})
    result={'kind':'NEW_NATIVE_VACUUM_CONNECTED_ENERGY_DEPENDENT_ENTIRE_CANONICAL_CELL_TRANSFER',
            'epoch':epoch,'governing_plan':request['governing_plan'],
            'input_sha256':request['input_sha256'],'source_sha256':sha(Path(__file__)),
            'corrected_actual2_auxiliary_profile_consumed':True,
            'formation':{'accepted_cell_residual_exact':str(cap),'included_formation_exact':'1/262144',
                         'infidelity_from_accepted_cap_exact':str(infidelity),'ground_distance_cap_exact':str(eta),
                         'density_trace_price_exact':str(delta),'new_beta_exact':str(beta),
                         'charged_multicell_ground_distance_exact':str(eta3),
                         'charged_multicell_beta_exact':str(beta3),'charged_multicell_moment_price_exact':str(source_price),
                         'conditional_binary64_no_FTZ':True,'observed_eta_or_residual_not_used':True},
            'domains':{'outer_source_specific_reducing_floor_exact':'-5120','arbitrary_central_whole_charged_floor_exact':'-5155',
                       'source_two_cell_floor_exact':str(b),'whole_operator_two_cell_floor_exact':str(d),
                       'central':'Entire total N256 of 512 modes; arbitrary spin/labelled-number/scalar/exterior sectors',
                       'totalrho_not_independent_copy_fields':True,'physical_hopping':'1'},
            'axes':axes,'independent_adverse_controls':exact_controls(),
            'scope':'Finite frozen matching reference single-charge cell Schur, NOT complete interacting spatial Schur expansion',
            'whole_excited_operator_paid_conditionally':True,
            'physical1_gate_reaches_one':all(r['reaches_one'] for a in axes for r in a['energies']),
            'higher_charge_intercell_domains_paid':False,'physicalPhi_ground':False,
            'bulk_stiffness':False,'allorder_phase':False,'photon_promoted':False,'campaign_complete':False,
            'missing_mechanism':'Signed energy-form cancellation in the entire-canonical remainder Jdag R T, together with interacting vacuum root/offdiagonal folds and multicharge linked spatial domains. Failure of this sufficient guard is not a no-go.',
            'elapsed_seconds':time.monotonic()-started}
    result['sources_inputs_unchanged']=all(sha(ROOT/n)==p for n,p in request['input_sha256'].items())
    if not result['sources_inputs_unchanged']:
        raise ArithmeticError('Immutable runtime input changed')
    dump(ROOT/f'{PREFIX}{epoch}_receipt.json',result)
    dump(ROOT/f'{PREFIX}interface_20261011.json',result)
    print(json.dumps({'receipt':f'{PREFIX}{epoch}_receipt.json',
                      'physical1_caps':[a['energies'][1]['whole_excited_connected_relative_cap_display'] for a in axes],
                      'reaches_one':result['physical1_gate_reaches_one']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    run(parser.parse_args().epoch)
