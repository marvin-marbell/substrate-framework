"""Native outer-Pg central neutral vectors; frozen imports, no old CLI/solver."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import shared_star_dynamic_scientist_20261011 as frozen

ROOT = Path(__file__).resolve().parent
PREFIX = 'central_neutral_vertex_'
ZERO = (F(0), F(0))
ONE = (F(1), F(0))
CANDIDATES = [F(229374216504249,2199023255552), F(3669987464067985,35184372088832)]

def dec(z):
    return F(float(complex(z).real)), F(float(complex(z).imag))

def ca(a,b):
    return a[0]+b[0], a[1]+b[1]

def cm(a,b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]

def cs(a,b):
    return a[0]*b, a[1]*b

def norm(z):
    return abs(z[0])+abs(z[1])

def floating(z):
    return complex(float(z[0]),float(z[1]))

def collect(terms):
    out = defaultdict(lambda: ZERO)
    for z,w,m in terms:
        for sign,v in frozen.normal(w):
            out[v,m] = ca(out[v,m],cs(z,sign))
    return [(z,w,m) for (w,m),z in out.items() if z != ZERO]

def multerms(left,right):
    return collect([(cm(a,b),w+v,tuple(x+y for x,y in zip(lm,rm)))
                    for a,w,lm in left for b,v,rm in right])

def reindex(terms,copy):
    return [(z,tuple((c,s+2*copy,p,cr) for c,s,p,cr in w),m) for z,w,m in terms]

def scale(terms,c):
    return [(cs(z,c),w,m) for z,w,m in terms]

def canonical(word):
    """Exact Nj=2 relation, before projection/coefficients/norm pricing."""
    for slot in sorted({s for _,s,_,_ in word}):
        number = ((1,slot,3,True),(1,slot,3,False))
        if word == number:
            return [(2,())]+[(-1,((1,slot,p,True),(1,slot,p,False))) for p in range(3)]
    return [(1,word)]

class Central:
    def __init__(self,k,star):
        self.k,self.star = k,star
        rows=json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())['profiles']
        self.h=[]
        for row in rows:
            m,c,d=[F(row[t]) for t in ('m_exact','c0_exact','d_exact')]
            h=[[ZERO for _ in range(4)] for _ in range(4)]
            for p in range(4):h[p][p]=(m*(-1 if p%2==0 else 1),F(0))
            for p in range(2):
                h[p][p+2]=(c*(1 if p==0 else -1),F(0));h[p+2][p]=h[p][p+2]
                h[p][3-p]=(F(0),d);h[3-p][p]=(F(0),-d)
            self.h.append(h)

    def action(self,terms,flow=(0,0),u=(F(0),F(0),F(0))):
        out=[]
        for z,w,memory in terms:
            if any(memory):raise ArithmeticError('Only one central K action: no uncomputed rho2 field')
            change=[F(0)]*3
            for c,s,p,cr in w:change[c]+=(1 if cr else -1)*F(1 if p<2 else -1,2)
            for i,(c,s,p,cr) in enumerate(w):
                h=self.h[0 if s<2 else 1]
                for q in range(4):
                    coeff=h[q][p] if cr else cs(h[p][q],-1)
                    if coeff!=ZERO:out.append((cm(z,coeff),w[:i]+((c,s,q,cr),)+w[i+1:],memory))
            constant=64*sum(F(x)**2 for x in flow)
            for cell in range(3):
                delta=change[cell]+u[cell];constant+=64*delta*delta
                if delta:out.append((cs(z,128*delta),w,tuple(int(j==cell) for j in range(3))))
            if constant:out.append((cs(z,constant),w,memory))
        return collect(out)

    def vertex(self,edge,role,scalar_only=False,row_gram=False):
        a=CANDIDATES[edge];c0,c1=2/a,-1/a**2
        if scalar_only:c0,c1=1/a,F(0)
        if row_gram:c0,c1=F(1),F(0)
        symbolic=defaultdict(lambda: ZERO)
        raw=F(0);dropped_different=0
        for orientation in (-1,1):
            first=[(dec(z),w,(0,0,0)) for z,w in self.star.root(edge,orientation,role)]
            returning=[(dec(z),w,(0,0,0)) for z,w in self.star.root(edge,-orientation,role)]
            flow,u=self.star.flow_u((1,edge,orientation,1-edge,0))
            action=self.action(first,flow,u)
            if any(any(m) for _,_,m in action):raise ArithmeticError('Initial electric cancellation failed')
            vector=collect(scale(first,c0)+scale(action,c1))
            for z,w,m in multerms(returning,vector):
                raw+=norm(z)
                outer=0 if edge==0 else 2
                inv=sum(w[i][0]>w[j][0] for i in range(len(w)) for j in range(i+1,len(w)))
                ow=tuple((s,p,cr) for c,s,p,cr in w if c==outer)
                cw=tuple(x for x in w if x[0]==1)
                balance=defaultdict(int)
                for s,p,cr in ow:balance[s]+=1 if cr else -1
                if any(balance.values()):dropped_different+=1;continue
                for sign,v in ([(1,cw)] if row_gram else canonical(cw)):
                    if not v and not row_gram:continue  # Only QB source scalars vanish.
                    key=v,ow
                    symbolic[key]=ca(symbolic[key],cs(z,sign*(-1)**inv))
        coefficients=defaultdict(lambda: ZERO);coefficient_budget=F(0)
        for (cw,ow),z in symbolic.items():
            if z==ZERO:continue
            expectation=dec(self.k.local_memory(ow,0,0))
            coefficients[cw]=ca(coefficients[cw],cm(z,expectation))
            coefficient_budget+=norm(z)
        terms=collect([(z,w,(0,0,0)) for w,z in coefficients.items() if z!=ZERO])
        # Relabel one auxiliary physical copy to template slots0,1.
        if role:terms=reindex(terms,-role)
        return terms,{'preprojection_raw_L1_exact':str(raw),'postcanonical_outer_coefficient_L1_exact':str(coefficient_budget),
                      'different_label_return_terms_removed':dropped_different,'nonzero_symbolic_coefficients':sum(z!=ZERO for z in symbolic.values()),
                      'central_collected_terms':len(terms),'scalar_deleted_by_QB_before_any_price':not row_gram}

    def norms(self,terms):
        return sum(norm(z)*128**sum(m) for z,w,m in terms)

    def moments(self,roles,active_only=False):
        action=[reindex(self.action(reindex(terms,role)),-role) for role,terms in enumerate(roles)]
        results=[];budgets=[]
        patterns=[(0,0,1),(1,1,63),(0,1,63),(1,0,63),(1,2,3906)]
        for li,ri in ((0,0),(0,1),(1,1)):
            value=0j;budget=F(0)
            for l,r,mult in patterns:
                if active_only and (l or r):continue
                ls=(roles if li==0 else action)[int(l>0)]
                rs=(roles if ri==0 else action)[int(r>0)]
                lt=reindex(ls,l);rt=reindex(rs,r)
                lv=[(floating(z),w,m) for z,w,m in lt];rv=[(floating(z),w,m) for z,w,m in rt]
                value+=mult*self.k.gram_terms(lv,rv)
                budget+=mult*self.norms(lt)*self.norms(rt)
            if abs(value.imag)>1e-7*max(1,abs(value.real)):raise ArithmeticError(('nonreal moment',value))
            results.append(value.real);budgets.append(budget)
        p=0j;pb=F(0)
        for role,mult in ((0,1),(1,63)):
            terms=reindex(roles[role],role)
            p+=mult*self.k.gram_terms([(1.,(),(0,0,0))],[(floating(z),w,m) for z,w,m in terms])
            pb+=mult*self.norms(terms)
        return results,budgets,p,pb,action

def square_interval(x):
    return (F(0) if x[0]<=0<=x[1] else min(x[0]**2,x[1]**2),max(x[0]**2,x[1]**2))

def enclose(central,roles,price,gap):
    m,b,p,pb,action=central.moments(roles)
    # A nonselfadjoint polynomial vertex can have complex ground amplitude.
    pr=frozen.interval(p.real,pb*price);pi=frozen.interval(p.imag,pb*price)
    pp=frozen.add(square_interval(pr),square_interval(pi))
    mi=[frozen.interval(v,cap*price) for v,cap in zip(m,b)]
    mi[0]=frozen.add(mi[0],frozen.scale(pp,F(-1)));mi[0]=(max(F(0),mi[0][0]),max(F(0),mi[0][1]))
    projected=m[0]-abs(p)**2
    coefficient=F(float(projected/m[1])) if m[1]>0 else F(0)
    residual=frozen.add(frozen.add(mi[0],frozen.scale(mi[1],-2*coefficient)),frozen.scale(mi[2],coefficient**2))
    error=frozen.sqrt_hi(mi[0][1]*max(F(0),residual[1]))/gap
    response=frozen.add(frozen.scale(mi[0],coefficient),(-error,error))
    response=(max(F(0),response[0]),min(mi[0][1]/gap,response[1]))
    if response[0]>response[1]:raise ArithmeticError('Empty whole-spectrum enclosure')
    return {'moments_float':m,'central_ground_amplitude_float':[p.real,p.imag],
            'central_ground_amplitude_intervals':[frozen.encode(pr),frozen.encode(pi)],'q_norm_squared_center_float':projected,
            'moments_exact_intervals':[frozen.encode(x) for x in mi],'postcollect_L1_moment_prices_exact':[str(x) for x in b],
            'inverse_candidate_exact':str(coefficient),'residual_square':frozen.encode(residual),'sourceweighted_response':frozen.encode(response),
            'scalar_candidate_response_float':float(coefficient)*projected,'all_spectrum_tail_paid':True,
            'first_action_terms':[len(x) for x in action],'collective_rho_terms':[sum(any(m) for z,w,m in x) for x in action]},response,mi[0][1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def run(epoch):
    started=time.monotonic()
    request=json.loads((ROOT/f'{PREFIX}{epoch}_request.json').read_bytes())
    for name,pin in request['input_sha256'].items():
        if sha(ROOT/name)!=pin:raise ArithmeticError(('immutable pin mismatch',name))
    boundary=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
    _,_,_,exact=frozen.formation.derive_physical(boundary)
    matrices={}
    for axis in range(3):
        M=np.asarray([[complex(frozen.formation.sp.sympify(z)) for z in row] for row in exact[axis]])
        for s,t in itertools.product(range(2),repeat=2):matrices[axis,s,t]=(-1)**(s+t)*M
    inventory,_=frozen.native.inventory(8)
    edges=[next(e for e in inventory if e['axis']==a and e['central_site']==0 and e['central_is_tail']) for a in (1,2)]
    k=frozen.DynamicKernel();star=frozen.Star(k,edges,matrices);central=Central(k,star)
    density=json.loads((ROOT/'three_cell_density_formation_typed3_20261011_interface.json').read_bytes())
    delta=max(F(r['delta_trace_norm_exact']) for r in density['errors'].values())
    eta=frozen.sqrt_hi(2*F(274877906944,2204801412260314559041))
    beta=2500*frozen.formation.gamma(8192)+frozen.formation.gamma(1<<24)+128*frozen.formation.gamma(256)
    price=delta+beta+2*eta
    gap=F(46955312929,34359738368)
    vertices=[];metadata=[];coefficient_errors=[];Vnorm=[];Jnorm=[];Jgram=[]
    for edge in range(2):
        roles=[];rows=[];error=F(0)
        for role,mult in ((0,1),(1,63)):
            terms,row=central.vertex(edge,role);roles.append(terms);rows.append(row)
            error+=mult*(F(row['postcanonical_outer_coefficient_L1_exact'])*price+F(row['preprojection_raw_L1_exact'])*beta)
        vertices.append(roles);metadata.append(rows);coefficient_errors.append(error)
        Vnorm.append(2*sum(F(r['count'])*F(next(p['fullCAR_hop_norm_upper_exact'] for p in r['paths'] if p['axis']==edges[edge]['axis'] and p['length']==1)) for r in boundary['roles']))
        rowcap=F(0);gramrows=[]
        for role,mult in ((0,1),(1,63)):
            terms,row=central.vertex(edge,role,row_gram=True)
            uncertainty=F(row['postcanonical_outer_coefficient_L1_exact'])*price+F(row['preprojection_raw_L1_exact'])*beta
            cap=central.norms(terms)+uncertainty
            rowcap+=mult*cap
            gramrows.append({'role':role,'count':mult,'full_central_Fock_row_gram_norm_upper_exact':str(cap),
                             'ground_outer_transfer_error_exact':str(uncertainty),'scalar_retained':True,
                             'no_fixed_central_labelled_number_simplification':True,'projection_accounting':row})
        Jnorm.append(frozen.sqrt_hi(rowcap));Jgram.append(gramrows)
    outputs={};responses={};norms={}
    complex_groups=[(name,[collect(a+[(cm(z,(F(0),sign)),w,m) for z,w,m in b]) for a,b in zip(*vertices)]) for name,sign in [('plusi',F(1)),('minusi',F(-1))]]
    for name,roles in [('a',vertices[0]),('b',vertices[1]),('plus',[collect(a+b) for a,b in zip(*vertices)]),('minus',[collect(a+scale(b,-1)) for a,b in zip(*vertices)])]+complex_groups:
        row,response,qnorm=enclose(central,roles,price,gap);outputs[name]=row;responses[name]=response;norms[name]=frozen.sqrt_hi(qnorm)
    central_cross=frozen.scale(frozen.add(responses['plus'],frozen.scale(responses['minus'],F(-1))),F(1,2))
    imaginary_cross=frozen.scale(frozen.add(responses['minusi'],frozen.scale(responses['plusi'],F(-1))),F(1,2))
    outer=json.loads((ROOT/'charged_outer_vector_interface_20261011.json').read_bytes())
    eps=[]
    for edge,a in enumerate(CANDIDATES):
        row=outer['axes'][edge]
        if row['polynomial_coefficients_exact']!=[str(2/a),str(-1/a**2)]:
            raise ArithmeticError('Charged interface polynomial differs from executed vertex')
        eps.append(F(row['epsilon_upper_exact']))
    transfer=[coefficient_errors[i]+Jnorm[i]*eps[i] for i in range(2)]
    transfer_radius=2*(transfer[0]*norms['b']+transfer[1]*norms['a']+transfer[0]*transfer[1])/gap
    actual=frozen.add(central_cross,(-transfer_radius,transfer_radius))
    control={}
    action_controls=[]
    for role,terms in enumerate(vertices[0]):
        actual_terms=reindex(terms,role)
        rational=central.action(actual_terms)
        reference=k.action([(floating(z),w,m) for z,w,m in actual_terms],(0,0),(F(0),F(0),F(0)))
        rd={(w,m):floating(z) for z,w,m in rational};fd={(w,m):z for z,w,m in reference}
        discrepancy=sum(abs(rd.get(key,0)-fd.get(key,0))*128**sum(key[1]) for key in rd.keys()|fd.keys())
        allowed=float(beta*(central.norms(actual_terms)+central.norms(rational)))
        if discrepancy>allowed:raise ArithmeticError('Independent native role commutator failed')
        wrong=reindex(central.action(terms),role)
        wd={(w,m):floating(z) for z,w,m in wrong}
        wrong_discrepancy=sum(abs(rd.get(key,0)-wd.get(key,0))*128**sum(key[1]) for key in rd.keys()|wd.keys())
        action_controls.append({'role':role,'rational_vs_frozen_native_action_L1_difference':discrepancy,
                                'conditional_formation_allowance':allowed,'wrong_template_active_h_difference':wrong_discrepancy})
    control['actual_unequal_role_commutator']=action_controls
    for name in ('plus','minus'):
        roles=[collect(a+(b if name=='plus' else scale(b,-1))) for a,b in zip(*vertices)]
        baseline=outputs[name]['moments_float']
        k.independent=True;k.local_memory.cache_clear();independent=central.moments(roles)[0]
        k.independent=False;k.local_memory.cache_clear();k.delete_memory=True;nomemory=central.moments(roles)[0]
        k.delete_memory=False;active=central.moments(roles,True)[0]
        control[name]={'correlated_baseline':baseline,'wrong_independent_copy':independent,'wrong_deleted_rho':nomemory,'wrong_only_active1':active}
    scalar_vertices=[[central.vertex(e,r,True)[0] for r in (0,1)] for e in (0,1)]
    scalar_plus=[collect(a+b) for a,b in zip(*scalar_vertices)]
    control['wrong_outer_scalar_replacement_plus_moments']=central.moments(scalar_plus)[0]
    control['Pground_not_sourceQ']={'unprojected_plus_norm':outputs['plus']['moments_float'][0],'priced_trueQ_center':outputs['plus']['q_norm_squared_center_float']}
    b=F(150034528033,34359738368)
    profiles=json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())['profiles']
    eupper=[frozen.sqrt_hi(F(row['e_squared_exact'])) for row in profiles]
    elower=[F(row['e_squared_exact'])/bound for row,bound in zip(profiles,eupper)]
    free_base=-4*eupper[0]-252*eupper[1]
    removal=min(lo-abs(F(row['c0_exact'])) for row,lo in zip(profiles,elower))
    free_charged2=free_base+2*removal
    ground_upper=-5120-b
    charged2=free_charged2-ground_upper
    mixed={'opposite_sign_floor_exact':str(128+2*b),'same_sign_floor_exact':str(128+2*b+charged2),
           'free_number254258_envelope_exact':str(free_charged2),'free_number256_base_lower_exact':str(free_base),
           'minimum_one_particle_removal_price_exact':str(removal),'central_positive_neutral_retained':True,
           'centerPgB_allowed_and_not_deleted':True,'domains_proof':'central_neutral_vertex_proof_20261011.json'}
    encoded=[]
    for edge,roles in enumerate(vertices):
        encoded.append({'axis':edge+1,'roles':[{'role':role,'count':1 if role==0 else 63,'terms':[{'coefficient_re_exact':str(z[0]),'coefficient_im_exact':str(z[1]),'word':w,'rho_right_powers':m} for z,w,m in terms]} for role,terms in enumerate(roles)]})
    operator_single_floor=64+2*b-35
    if operator_single_floor<=0:raise ArithmeticError('Wholecentral charged floor not positive')
    uniform_forms=[cap*cap/(operator_single_floor*gap) for cap in Jnorm]
    result={'kind':'ACTUAL_NATIVE_OUTER_GROUND_REDUCED_CENTRAL_NEUTRAL_VERTEX','status':'CONDITIONAL_NATIVE_ENCLOSURE_NO_PHYSICAL1_PHASE',
            'plan':request['plan'],'source_sha256':sha(Path(__file__)),'input_sha256':request['input_sha256'],
            'edges':edges,'polynomial_outer_coefficients_exact':[[str(2/a),str(-1/a**2)] for a in CANDIDATES],
            'native_central_operator_vectors':encoded,'canonical_projection_accounting':metadata,
            'formation':{'source_distance_exact':str(eta),'density_delta_exact':str(delta),'new_operation_beta_exact':str(beta),'moment_price_exact':str(price),
                         'conditional_binary64':True,'observed_residual_not_used':True,'cell_residual_including_formation':'1/65536','formation_component':'1/262144'},
            'central_q_resolvents':outputs,'polynomial_neutral_2Re_cross':frozen.encode(central_cross),
            'outer_coefficient_transfer_errors_exact':[str(x) for x in coefficient_errors],
            'degree1_whole_spectral_outer_epsilons_exact':[str(x) for x in eps],
            'outer_ground_row_operator_gram':Jgram,'outer_ground_row_norm_caps_exact':[str(x) for x in Jnorm],
            'coarse_full_V_norm_caps_exact':[str(x) for x in Vnorm],
            'actual_outer_resolvent_2Re_cross':frozen.encode(actual),'actual_cross_transfer_radius_exact':str(transfer_radius),
            'polynomial_neutral_2Im_cross':frozen.encode(imaginary_cross),
            'actual_outer_resolvent_2Im_cross':frozen.encode(frozen.add(imaginary_cross,(-transfer_radius,transfer_radius))),
            'mixed_nonzero_flow_domains':mixed,'controls':control,
            'excited_transfer':{'source_domain':'span{q_a,q_b}; full central total256 direct sum of labelled-number/scalar/vector/exterior sectors, never harmonic-only',
                                'paid_sourceweighted_R_matrix_by_polarization':True,
                                'uniform_operator_single_floor_exact':str(operator_single_floor),
                                'uniform_excited_form_relative_caps_exact':[str(x) for x in uniform_forms],
                                'uniform_excited_form_relative_caps_display':[float(x) for x in uniform_forms],
                                'form_domain':'All phi in QB D(K_B^1/2) on the entire neutral canonical256 cell. |<phi,W_e phi>| <= cap_e <phi,K_B phi>; outer charged reducing union, center whole255/257 weakfloor -5155.',
                                'uniform_form_bound_proved_conditionally':True,'allorder_excited_transfer_proved':False,
                                'remaining_input':'A sufficiently SMALL source-weighted excited transfer plus linked spatial histories; current rigorous entire-excited relative caps fail contraction at physical1. Ground-column q norms do not imply a smaller allorder cap.'},
            'elapsed_seconds':time.monotonic()-started}
    result['source_and_inputs_unchanged']=all(sha(ROOT/name)==pin for name,pin in request['input_sha256'].items())
    dump(ROOT/f'{PREFIX}{epoch}_receipt.json',result)
    dump(ROOT/f'{PREFIX}interface_20261011.json',result)
    print(json.dumps({'receipt':f'{PREFIX}{epoch}_receipt.json','actual_cross_interval':result['actual_outer_resolvent_2Re_cross']['display'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epoch',required=True)
    run(parser.parse_args().epoch)
