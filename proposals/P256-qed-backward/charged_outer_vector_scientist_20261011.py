"""NEW native degree-one charged operator vectors; pure frozen imports, no solve."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import sympy as sp
import shared_star_dynamic_scientist_20261011 as frozen
import harmonic_car_bridge_scientist_20261011 as car

ROOT=Path(__file__).resolve().parent
PREFIX='charged_outer_vector_'
INPUTS=sorted(set(frozen.INPUTS+car.EXTRA+list(car.PINS)+[
 'shared_star_dynamic_scientist_20261011.py','shared_star_dynamic_actual2_20261011_receipt.json',
 'shared_star_dynamic_proof_20261011.json','charged_stieltjes_proof_20261011.json',
 'charged_stieltjes_actual1_20261011_receipt.json','charged_cell_heat_typed3_20261011_interface.json',
 'charged_lower_review_attempt2_20261011_receipt.json','charged_output_lower_attempt3_20261011_receipt.json']))
FROZEN_STAR='2f02fee85495c4ea6ad6b687add5e0ebf1a9fb975c01205bc82f3c98d3c29fa3'
FROZEN_RECEIPT='3f690a820fbc84095804d0470386911dd8a9298c396fb9f2c915da870fdfe789'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(name):return json.loads((ROOT/name).read_bytes())
def enc(x):return {'exact':[str(v) for v in x],'display':[float(v) for v in x]}
def mag(z):return abs(F(float(z.real)))+abs(F(float(z.imag)))
def l1(terms):return sum((mag(z)*128**sum(m) for z,w,m in terms),F(0))
def sumiv(xs):
 out=(F(0),F(0))
 for x in xs:out=frozen.add(out,x)
 return out

def image(terms):
 return [{'coefficient_exact':str(sp.simplify(z)), 'word':[list(a) for a in w], 'right_total_rho_powers':list(m)} for z,w,m in terms if z!=0]

def symbolic_action(terms,flow,u,hs):
 out={}
 def put(z,w,m):out[w,m]=out.get((w,m),sp.S.Zero)+z
 for z,w,m in terms:
  if any(m):raise ArithmeticError('No dropping noncommuting rho memory')
  changes=[F(0)]*3
  for cell,slot,p,creation in w:changes[cell]+=(1 if creation else -1)*F(1 if p<2 else -1,2)
  for i,(cell,slot,p,creation) in enumerate(w):
   h=hs[0 if slot<2 else 1]
   for q in range(4):
    a=h[q,p] if creation else -h[p,q]
    if a:put(z*a,w[:i]+((cell,slot,q,creation),)+w[i+1:],m)
  const=64*sum(v*v for v in flow)
  for cell in range(3):
   delta=changes[cell]+u[cell];const+=64*delta*delta
   if delta:put(z*sp.Rational(128*delta),w,tuple(int(j==cell) for j in range(3)))
  if const:put(z*sp.Rational(const),w,m)
 return [(sp.simplify(z),w,m) for (w,m),z in out.items() if z!=0]

def domain_controls():
 J,maps,h4,ops,ann,V=car.car_exact()
 P=V*V.T
 invariant=all(sp.simplify(o*P-P*o)==sp.zeros(16) for o in ops)
 N=sum((a.T*a for a in ann),sp.zeros(16))
 labelled=all(sp.simplify(o*N-N*o)==sp.zeros(16) for o in ops)
 assert invariant and labelled
 return {'exact_W5_spectator_projector_commutes_with_all_four_generators':invariant,
 'exact_individual_Nj_commutes_with_all_four_generators':labelled,
 'selected_charged_spinor_dimensions':[4,4],
 'spectator_domain':'Full unprojected Sym(C5) after removing exactly one labelled spin replica: active selected spectators(1,126), auxiliary selected(2,125); includes every harmonic degree, no harmonic cutoff.',
 'reducing_argument':'W5 is invariant under each dGamma(h) and R. Sums of identical per-role onebody and R preserve spectator permutation symmetry. The total affine square preserves the same spaces. Nj and external integer-flow projections commute with H0, so all copy/spin/orientation source sectors reduce K and are mutually spectral-orthogonal. Creation/annihilation selected Nj=3/1 uses complete four-component spinor. The reviewed -5120 bound is applied only to these charged containing modules.',
 'source_domain':'Exact g lies in the frozen canonical Sym^2(C5) x Sym^126(C5) embedding. Each all-four-spin monomial lies in the stated charged modules; f(K) and its residual retain them. Literal stored s is distinctg and has the same module support.',
 'charged_floor_provenance':['charged_lower_review_attempt2_20261011_receipt.json','charged_output_lower_attempt3_20261011_receipt.json','charged_stieltjes_proof_20261011.json']}

def run(epoch):
 started=time.monotonic();pins={n:sha(ROOT/n) for n in INPUTS}
 assert pins['shared_star_dynamic_scientist_20261011.py']==FROZEN_STAR
 assert pins['shared_star_dynamic_actual2_20261011_receipt.json']==FROZEN_RECEIPT
 assert all(pins[n]==p for n,p in frozen.native.PINS.items())
 assert all(pins[n]==p for n,p in car.PINS.items())
 domain=domain_controls()
 old=load('shared_star_dynamic_actual2_20261011_receipt.json')
 upper=load('local_so5_ground_upper_complete_state_consumer_20261011_receipt.json')
 Elo,Ehi=map(F,upper['ground_energy_interval_exact'])
 b=64-10240-2*Ehi;assert b==F(old['single_flow_floor_exact']) and b>0
 profiles=load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json')['profiles']
 D=sum((F(4*r['spin_replicas'])*sum(abs(F(r[key])) for key in ('m_exact','c0_exact','d_exact')) for r in profiles),F(0))
 U=2*(D+64*F(257,2)**2-Elo)+64
 domain.update(single_flow_floor_exact=str(b),full_finite_containing_K_upper_exact=str(U),upper_derivation='Each of128 replicas: ||dGamma(h)||<=4(|m|+|c|+|d|); ||Rtotal||<=128 on full four-mode Fock, |u|=1/2; X upper=2(D+64(128+1/2)^2-Eg_lower)+64. This broad full-Fock upper is not a sampled spectral cutoff.')
 boundary=load('charged_first_shell_actual04_20261011_receipt.json')['all64_boundary']
 _,_,_,exact=frozen.formation.derive_physical(boundary)
 matrices={}
 exactmat={}
 for axis in range(3):
  M=sp.Matrix([[sp.sympify(z) for z in row] for row in exact[axis]])
  for s,t in itertools.product(range(2),repeat=2):
   exactmat[axis,s,t]=(-1)**(s+t)*M
   matrices[axis,s,t]=np.asarray(exactmat[axis,s,t]).astype(complex)
 inventory,_=frozen.native.inventory(8)
 edges=[next(e for e in inventory if e['axis']==axis and e['central_site']==0 and e['central_is_tail']) for axis in (1,2)]
 k=frozen.DynamicKernel();star=frozen.Star(k,edges,matrices)
 hs=[]
 for r in profiles:hs.append(sum((sp.Rational(F(r[key]))*v for key,v in zip(('m_exact','c0_exact','d_exact'),car.car_exact()[2][:3])),sp.zeros(4)))
 price=F(old['formation_and_transfer']['combined_relative_L1_expectation_price_exact'])
 rows=[];aggregates=[]
 for edge,axis in enumerate((1,2)):
  a=F(old['single_roots'][edge]['alpha_exact']);c0=2/a;c1=-1/a**2
  factor=max(F(1),a/b-1)
  agg=[(F(0),F(0)) for _ in range(5)]
  for orientation in (-1,1):
   for copy,count,role in ((0,1,'active'),(1,63,'aux')):
    root=[(z,w,(0,0,0)) for z,w in star.root(edge,orientation,copy)]
    flow,u=star.flow_u((1,edge,orientation,1-edge,0))
    A=k.action(root,flow,u)
    assert not any(any(m) for z,w,m in A)
    B=k.action(A,flow,u)
    vectors=[root,A,B]
    pairs=[(root,root),(root,A),(A,A),(A,B),(B,B)]
    centers=[float(k.gram_terms(l,r).real) for l,r in pairs]
    budgets=[l1(l)*l1(r) for l,r in pairs]
    mi=[frozen.interval(v,L*price) for v,L in zip(centers,budgets)]
    for j in range(5):agg[j]=frozen.add(agg[j],frozen.scale(mi[j],count))
    variance=sumiv([mi[2],frozen.scale(mi[1],-2*a),frozen.scale(mi[0],a*a)])
    eps=frozen.sqrt_hi(max(F(0),variance[1]))*factor/a**2
    # r=v-Kf(K)v has coefficients(1,-c0,-c1). All rho memories in B remain.
    ri=sumiv([mi[0],frozen.scale(mi[1],-2*c0),frozen.scale(mi[2],c0*c0-2*c1),frozen.scale(mi[3],2*c0*c1),frozen.scale(mi[4],c1*c1)])
    rcap=frozen.sqrt_hi(max(F(0),ri[1]));eps=min(eps,rcap/b)
    xs=[]
    for i,j in itertools.product(range(4),repeat=2):
     e=edges[edge];M=exactmat[axis,e['central_site'],e['outer_site']]
     ct=1;ch=0 if edge==0 else 2
     w=((ct,2*copy+i//2,2*e['central_site']+i%2,True),(ch,2*copy+j//2,2*e['outer_site']+j%2,False))
     z=M[i,j]
     if orientation==-1:z=sp.conjugate(z);w=tuple((cell,slot,p,not cr) for cell,slot,p,cr in reversed(w))
     if z:xs.append((z,w,(0,0,0)))
    sx1=symbolic_action(xs,flow,u,hs);sx2=symbolic_action(sx1,flow,u,hs)
    xsproxy=frozen.collect([(float(c0)*z,w,m) for z,w,m in root]+[(float(c1)*z,w,m) for z,w,m in A])
    pairdist=2*F(load('charged_cell_heat_typed3_20261011_interface.json')['ground_distance_exact'][1])
    proxycap=pairdist*(abs(c0)*l1(root)+abs(c1)*l1(A))
    rows.append({'axis':axis,'orientation':orientation,'role':role,'physical_copy_count':count,'all_four_spin_blocks':True,'flow':list(flow),'affine_u_exact':[str(v) for v in u],
      'polynomial_coefficients_exact':[str(c0),str(c1)],'moments_g_intervals':[enc(v) for v in mi],'moment_centers_conditional_float':centers,
      'residual_norm_upper_exact':str(rcap),'residual_over_charged_floor_exact':str(rcap/b),'weighted_full_spectrum_error_upper_exact':str(eps),
      'whole_inverse_vector_norm_upper_exact':str(frozen.sqrt_hi(max(F(0),mi[0][1]))/b),
      'same_source_scalar_error_upper_exact':str(frozen.sqrt_hi(max(F(0),variance[1]))/(a*b)),
      'actual_degree1_source_proxy_terms':len(xsproxy),'source_proxy_to_true_g_norm_upper_exact':str(proxycap),
      'source_images_exact':{'O':image(xs),'KO':image(sx1),'K2O':image(sx2)},
      'memory_preserved_in_K2O':any(any(m) for z,w,m in sx2),
      'source_proxy_semantics':'(c0O+c1C)s where C=[H0,O]+64O with exact affine rho terms. x_g=(c0O+c1C)g=f(K)Og. Not f(K)Os and not a rename of s to g.',
      'formation_assumptions':'Frozen field GR/RR formation and source-to-g bounded polynomial model, conditional binary64; inherited exact norm normalization and residual1/65536 including1/262144, Rayleigh shift retained.'})
  var=sumiv([agg[2],frozen.scale(agg[1],-2*a),frozen.scale(agg[0],a*a)])
  ep=frozen.sqrt_hi(max(F(0),var[1]))*factor/a**2
  oldeps=F(old['single_roots'][edge]['whole_charged_outer_resolvent_error_exact'])
  aggregates.append({'axis':axis,'polynomial_coefficients_exact':[str(c0),str(c1)],'anchor_exact':str(a),
   'all1_plus63_both_orientations_moments_g':[enc(v) for v in agg],
   'epsilon_upper_exact':str(ep),'epsilon_upper_display':float(ep),'frozen_scalar_epsilon_exact':str(oldeps),'frozen_scalar_epsilon_display':float(oldeps),
   'strict_improvement_certified_conditionally':ep<oldeps,'ratio_to_frozen_scalar_display':float(ep/oldeps),
   'whole_inverse_vector_norm_upper_exact':str(frozen.sqrt_hi(max(F(0),agg[0][1]))/b),
   'strict_better_than_whole_inverse_bound':ep<frozen.sqrt_hi(max(F(0),agg[0][1]))/b})
 result={'kind':'ACTUAL_NATIVE_DEGREE1_CHARGED_OUTER_OPERATOR_VECTOR_INTERFACE','epoch':epoch,'natural_main_finished':True,
 'governing_plan':{'id':'issue-222-induced-photon-delta13-continuation-20261008','captured_revision':'c6e1ec4424d6c94ad36f5ceca2d5bd72fbbff3ec25d964a0c15a2fe9e565f060','template':'general','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','work_item':'induced-photon'},
 'input_sha256':pins,'own_source_sha256':sha(Path(__file__)),'domain':domain,'edges':edges,'sectors':rows,'axes':aggregates,
 'formation':old['formation_and_transfer'],
 'full_spectrum_error_proof':'For x>=b>0, 1/x-(2/a-x/a^2)=(x-a)^2/(a^2x). Thus squared error <= max(1,a/b-1)^2/a^4 integral(x-a)^2 dmu. This is a global weighted residual inequality, not a fourth-moment tail deletion. Unweighted residual v-Kf(K)v uses full K2O norm and rho-memory moments M0..M4. Its residual/b bound is also reported. All source spectral mass is covered.',
 'coefficient_selection':'Frozen scalar alpha is an exact rational anchor, not optimized or retagged Ritz. Nonzero c1 applies actual Dirac commutator, and K2O retains subsequent total-rho memory.',
 'conditional_formation_note':'Price inherited from pinned accepted shared-star GR/RR field formation; new Gram/action arithmetic bounded by its same beta and L1 model. Machine attestation unsupported. No eigensolver/old scientific main/root/checker is invoked.',
 'missing_for_sharper_residual':'No missing field for the executed degree1 residual: K2O has only first right-rho powers; GR/RR fields suffice. Higher degree would need additional noncommuting Noether/third-field data, not inferred here.',
 'nonclaims':['No certified Ritz solve, physical1 native sign, phase or massless pole.','No ground column promoted to excited-to-excited all-order transfer bound.','No sourceg/fullPhi adoption or coupling change.'],
 'elapsed_seconds':time.monotonic()-started}
 assert all(sha(ROOT/n)==p for n,p in pins.items())
 for name in (f'{PREFIX}{epoch}_receipt.json',f'{PREFIX}interface_20261011.json'):
  with (ROOT/name).open('x') as out:json.dump(result,out,indent=2);out.write('\n')
 print(json.dumps({'axes':[{k:r[k] for k in ('axis','epsilon_upper_display','frozen_scalar_epsilon_display','strict_improvement_certified_conditionally')} for r in aggregates],'natural_main_finished':True}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epoch',required=True);run(p.parse_args().epoch)
