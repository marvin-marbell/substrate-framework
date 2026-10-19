"""Outward formation and S0-only replacement of immutable native164 moments.
Consumes, never reruns, the pinned producer; arithmetic prices are explicit.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
PRIMARY = 'three_cell_native_actual3_20261011_receipt.json'
PINS = {
 PRIMARY: '1616f84b0a4c09f7ab1905d0b158601cd2fa7e082a4ee4e1312b35e1dcc7a741',
 'three_cell_native_scientist_20261011.py': 'bed006710905adc3e4574d3cdde42ba3c29ce9d59677bef3c2cbcaa47f628557',
 'three_cell_native_proof_20261011.json': 'c67fa1780acd241d0f7869e34d4d3e7979f3199b57763392354817d28c7fb043',
 'charged_first_shell_actual04_20261011_receipt.json': 'b48aed4cb67ae2fd98e0b75ee4c58053244c68e1e4bd47c4cb97c9fdf58e4f02',
 'connected_root_exterior_actual1_20261011_moments.npz': '4c6c6e0da22e0d9d405c37935332bb9548fbf2df5f429ec66b50aa671f8af67d',
 'harmonic_car_bridge_typed_attempt3_20261011_receipt.json': 'be07db74d79630fd62866e4a44eb32e8b9217eaf59bf2ce1f26fdc247e91ff21',
 'connected_root_exterior_scientist_20261011.py': 'c61389c52cd146bf1c4ecc189c7b46b2c964d53c1036a73a975b9859f3651e62',
 'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz': 'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
}
PATTERNS = [(0,0,1,'same_active'), (1,1,63,'same_auxiliary'),
 (0,1,63,'distinct_active_auxiliary'), (1,0,63,'distinct_auxiliary_active'),
 (1,2,3906,'distinct_auxiliary_auxiliary')]
U = Q(1, 2**53)

def gamma(n):
 return n*U/(1-n*U)

def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def sqrt_hi(x):
 x = Q(x)
 if x <= 0: return Q(0)
 scale = 2**80
 k = math.isqrt(x.numerator*scale*scale//x.denominator)
 if Q(k*k, scale*scale) < x: k += 1
 return Q(k, scale)

class Interval:
 def __init__(self, lo, hi=None):
  lo=Q(lo); hi=lo if hi is None else Q(hi); assert lo<=hi
  scale=2**96
  self.lo=Q((lo.numerator*scale)//lo.denominator,scale)
  self.hi=Q(-((-hi.numerator*scale)//hi.denominator),scale)
 def __add__(self, other):
  other = wrap(other); return Interval(self.lo+other.lo, self.hi+other.hi)
 __radd__ = __add__
 def __neg__(self): return Interval(-self.hi, -self.lo)
 def __sub__(self, other): return self+-wrap(other)
 def __mul__(self, other):
  other = wrap(other); a = [x*y for x in (self.lo,self.hi) for y in (other.lo,other.hi)]
  return Interval(min(a), max(a))
 __rmul__ = __mul__
 def __truediv__(self, other):
  other = wrap(other); assert not other.lo <= 0 <= other.hi
  return self*Interval(1/other.hi,1/other.lo)
 def square(self):
  return Interval(0 if self.lo <= 0 <= self.hi else min(self.lo**2,self.hi**2), max(self.lo**2,self.hi**2))
 def encoded(self): return [str(self.lo), str(self.hi)]

def wrap(x): return x if isinstance(x, Interval) else Interval(x)

def around(x, error, positive=False):
 x=Q(x); return Interval(max(0,x-error) if positive else x-error,x+error)

def log_point(x):
 assert x>0
 z=(x-1)/(x+1); az=abs(z)
 assert az<Q(1,2), 'The exact log series is intentionally scoped to these finite kernels'
 value=2*sum((z**(2*k+1)/Q(2*k+1) for k in range(24)), Q(0))
 tail=2*az**49/(49*(1-az**2))
 return Interval(value-tail,value+tail)

def log_interval(x): return Interval(log_point(x.lo).lo,log_point(x.hi).hi)

def derive_physical(boundary):
 with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',allow_pickle=False) as z:
  phi=z['Phi_binary_section'][1]
 def exact(z): return sp.Rational(Q(float(z.real)))+sp.I*sp.Rational(Q(float(z.imag)))
 H=sp.Matrix([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])
 frame=H.T*sp.Matrix([[exact(z) for z in row] for row in phi])*H/2
 norm=sp.sqrt(sp.trace(frame*frame)/4)
 G=sp.simplify(frame/norm)
 e=sp.eye(4)
 S=sp.Matrix.hstack(e[:,0],G*e[:,0],e[:,1],G*e[:,1])
 assert S.T.conjugate()*S==e
 P=sp.diag(1,-1,1,-1)
 prices={}; L={}; matrices={}
 for axis in range(3):
  Sa=Q(0); role_details=[]
  for role in boundary['roles']:
   path=next(p for p in role['paths'] if p['axis']==axis and p['length']==1)
   M=sp.Matrix([[sp.Rational(Q(a))+sp.I*sp.Rational(Q(b)) for a,b in row] for row in path['matrix_exact']])
   rank=M.rank(); frob=Q(str(sp.trace(M.T.conjugate()*M)))
   Sa+=Q(role['count'])*rank*frob
   role_details.append({'role':role['role'],'count':role['count'],'rank':rank,'Frobenius_square_exact':str(frob),'rankFrob_square_exact':str(rank*frob)})
   rotated=P*S.T.conjugate()*M*S*P
   # Exact S is a permutation: entrywise l1 price is endpoint/orientation invariant.
   l1=sum((abs(Q(str(sp.re(z))))+abs(Q(str(sp.im(z)))) for z in rotated),Q(0))
   if role['role']=='active': L[axis]=l1; matrices[axis]=[[str(z) for z in rotated.row(i)] for i in range(4)]
   else:
    assert l1==L[axis]
    assert [[str(z) for z in rotated.row(i)] for i in range(4)]==matrices[axis]
  prices[axis]={'Sa_exact':str(Sa),'role_prices':role_details}
 return S,prices,L,matrices

def metric(a,b,g,t):
 t2=t*t; t4=t2*t2
 F=1+t2*(a+b)+t4*g
 leg=(1+t2*a)*(1+t2*b)
 C=g-a*b
 return {'F':F,'leg_product':leg,'connected_coefficient':C,
  'source_connected_over_full_norm':t4*C/F,'FS_real_mixed':t2*C/F.square(),
  'mixed_log_derivative':4*t2*C/F.square(),
  'connected_log':log_interval(F/leg)}

def complex_probe(vals,t):
 # Bra amplitudes(it,t), ket(t,it), so the fourth-order phase is+1.
 a,b=vals['Ga'],vals['Gb']; t2=t*t; t4=t2*t2
 Fab=1+t2*(a+b)+t4*vals['Gab']; Fba=1+t2*(a+b)+t4*vals['Gba']
 norms=Fab*Fba
 denominator=Interval(max(0,sqrt_hi(norms.lo)-Q(1,2**80)),sqrt_hi(norms.hi))
 real=1+t4*vals['ReD']; imag=t2*(b-a)+t4*vals['ImD']
 same_real=1+t4*vals['Gab']; same_imag=t2*(b-a)
 return {'bra_amplitudes':'(i*t,t)','ket_amplitudes':'(t,i*t)',
  'AB_BA_normalized_kernel_real':(real/denominator).encoded(),
  'AB_BA_normalized_kernel_imag':(imag/denominator).encoded(),
  'AB_AB_normalized_kernel_real':(same_real/Fab).encoded(),
  'AB_AB_normalized_kernel_imag':(same_imag/Fab).encoded()}

def error_price(La,Lb,errors,ordered):
 # Every cell-local exact compressed CAR word is a contraction. beta pays
 # U decoding, partial trace, W compression, tensor op and complex einsum.
 beta=2500*gamma(8192)
 kappa=64*gamma(8192)
 reduction=gamma(2**20 if ordered else 8192)
 total=Q(0)
 patterns=PATTERNS if ordered else [(0,0,1,'active'),(1,1,63,'auxiliary')]
 for cf,cg,mult,_ in patterns:
  ra=0 if cf==0 else 1; rb=0 if cg==0 else 1
  central=('AA' if ra==rb==0 else 'UU' if ra==rb==1 else 'AU')
  if ordered:
   ds=[errors['AA' if ra==0 else 'UU'],errors[central],errors['AA' if rb==0 else 'UU']]
   coeff=La**2*Lb**2; degree=4
  else:
   ds=[errors['AA' if ra==0 else 'UU']]*2
   coeff=La**2; degree=2
  # Deliberately charges two-slot delta even where the word uses one slot.
  factor=math.prod(1+d+beta*(1+d) for d in ds)
  total+=mult*coeff*((1+kappa)**degree*factor*(1+reduction)-1)
 return total

def ground_errors(a,b,ab,ba,Sa,Sb):
 d=Q(1,50000)
 ra=sqrt_hi(Sa)*d; rb=sqrt_hi(Sb)*d; rab=sqrt_hi(Sa*Sb)*d
 return {'Ga':2*sqrt_hi(a.hi)*ra+ra*ra,'Gb':2*sqrt_hi(b.hi)*rb+rb*rb,
 'Gab':2*sqrt_hi(ab.hi)*rab+rab*rab,'Gba':2*sqrt_hi(ba.hi)*rab+rab*rab,
 'ReD':(sqrt_hi(ab.hi)+sqrt_hi(ba.hi))*rab+rab*rab}

def dicke_control():
 state={s:Q(1) for s in itertools.combinations(range(6),3)}
 target={}
 for s,z in state.items():
  for f in set(range(6))-set(s):
   out=tuple(sorted((*s,f))); target[out]=target.get(out,Q(0))+z
 norm=sum(z*z for z in state.values()); after=sum(z*z for z in target.values())
 assert after/norm==12 and after/norm>6
 return {'domain':'six physical copies, each one A and one B mode; total number6, not fixed number in each cell/copy',
 'Dicke_k':3,'T3_allB_each_amplitude':'6','unscaled_input_norm_square':str(norm),
 'unscaled_output_norm_square':str(after),'ratio_exact':str(after/norm),'wrong_global_Sa':'6',
 'fixed_cell_number_hypothesis_essential':True}

def run(density_path,output):
 density=json.loads(density_path.read_bytes())
 errors={role:Q(density['errors'][role]['delta_trace_norm_exact']) for role in ('AA','UU','AU')}
 assert all(e>=0 for e in errors.values())
 density_sha=sha(density_path)
 before={name:sha(ROOT/name) for name in PINS}; assert before==PINS
 p=json.loads((ROOT/PRIMARY).read_bytes()); boundary=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
 assert Q(density['common_N0_exact'])==Q(p['source_normalization_divisor_float'])
 S,prices,L,matrices=derive_physical(boundary)
 I=Q(274877906944,2204801412260314559041)
 assert 3*I<Q(1,50000)**2
 assert 3*I/(1-3*I)<Q(1,50000)**2
 assert len(p['classes'])==164 and len(p['edges'])==10
 rows=[]; seen=set()
 for row in p['classes']:
  ea,eb=[p['edges'][i] for i in row['edges']]
  assert ea['outer_cell']!=eb['outer_cell'] and row['cells'][0]!=row['cells'][2]
  key=tuple(row['edges']+row['orientations']); assert key not in seen;seen.add(key)
  ax,bx=ea['axis'],eb['axis']; aerr=error_price(L[ax],L[ax],errors,False); berr=error_price(L[bx],L[bx],errors,False)
  gerr=error_price(L[ax],L[bx],errors,True)
  source={name:around(row[name],err,True) for name,err in [('Ga',aerr),('Gb',berr),('Gab',gerr),('Gba',gerr)]}
  source['ReD']=around(row['ordered_overlap'][0],gerr)
  source['ImD']=around(row['ordered_overlap'][1],gerr)
  Sa=Q(prices[ax]['Sa_exact']); Sb=Q(prices[bx]['Sa_exact'])
  ge=ground_errors(source['Ga'],source['Gb'],source['Gab'],source['Gba'],Sa,Sb)
  ground={name:Interval(max(0,v.lo-ge[name]) if name in ('Ga','Gb','Gab','Gba') else v.lo-ge[name],v.hi+ge[name]) for name,v in source.items() if name!='ImD'}
  # The same overlap-error cap also bounds the imaginary part.
  ground['ImD']=Interval(source['ImD'].lo-ge['ReD'],source['ImD'].hi+ge['ReD'])
  t=Q(row['t_diagnostic']); phases={}
  for label,vals in [('normalized_s',source),('reference_g',ground)]:
   sym=(vals['Gab']+vals['Gba']+2*vals['ReD'])*Q(1,4)
   sym=Interval(max(0,sym.lo),sym.hi)
   orders={}
   for order,g in [('AB',vals['Gab']),('BA',vals['Gba']),('coherent_half_sum',sym)]:
    orders[order]={name:value.encoded() for name,value in metric(vals['Ga'],vals['Gb'],g,t).items()}
   phases[label]={'moments':{name:v.encoded() for name,v in vals.items()},'coherent_half_sum_norm_square':sym.encoded(),'orders':orders,'complex_ordered_probe':complex_probe(vals,t)}
  rows.append({'edges':row['edges'],'orientations':row['orientations'],'cells':row['cells'],
   'central_endpoint_class':row['central_endpoint_class'],'t_exact_binary':str(t),
   'formation_errors':{'Ga':str(aerr),'Gb':str(berr),'ordered_Gram_and_overlap':str(gerr)},
   'reference_replacement_errors':{name:str(v) for name,v in ge.items()},'intervals':phases})
 # Common-leg normalization and ordered contact are active wrong-implementation controls.
 first=rows[0]['intervals']['reference_g']['orders']['AB']
 norm_delta=wrap(0)
 F=Interval(*first['F']); leg=Interval(*first['leg_product']); norm_delta=F-leg
 contact=[]
 for row in rows:
  v=row['intervals']['normalized_s']['moments']; ct=Interval(*v['Gab'])+Interval(*v['Gba'])-2*Interval(*v['ReD'])
  if ct.lo>0: contact.append({'edges':row['edges'],'orientations':row['orientations'],'commutator_square':ct.encoded()})
 wrong_role_error=error_price(L[0],L[1],{k:errors['AA'] for k in errors},True)
 actual_role_error=Q(rows[0]['formation_errors']['ordered_Gram_and_overlap'])
 role_difference=around(p['controls']['wrong64active_absolute_error_float'],wrong_role_error+actual_role_error)
 controls={'wrong_global_domain':dicke_control(),'wrong_central_order_nonzero_contact_intervals':contact,
  'wrong_leg_normalization_first_class_difference':norm_delta.encoded(),
  'wrong64active_normalized_s_Gab_difference':role_difference.encoded(),
  'wrong64active_rejected_conditionally':role_difference.lo>0,
  'parallel_distinct_raw_edges_same_outercell_excluded':4,'accepted_distinct_outercell_raw_pairs':41}
 cert=all(Interval(*r['intervals']['reference_g']['orders']['AB']['source_connected_over_full_norm']).hi<0 for r in rows)
 result={'kind':'CONDITIONAL_OUTWARD_SOURCE_FORMATION_AND_S0_REFERENCE_TRANSFER',
  'density_interface_path':str(density_path),'density_interface_SHA':density_sha,
  'density_trace_norm_errors_exact':{k:str(v) for k,v in errors.items()},
  'source_and_input_SHA':before,'own_source_SHA':sha(Path(__file__)),
  'physical_S_exact':[[str(z) for z in S.row(i)] for i in range(4)],'physical_endpoint00_M_exact':matrices,
  'domain_prices':prices,'coefficient_l1_prices':{str(k):str(v) for k,v in L.items()},
  'arithmetic_prices':{'u_exact':str(U),'local_beta_exact':str(2500*gamma(8192)),
   'coefficient_kappa_exact':str(64*gamma(8192)),'single_sum_gamma_exact':str(gamma(8192)),
   'ordered_sum_gamma_exact':str(gamma(2**20))},
  'single_squared_infidelity_exact':str(I),'three_cell_squared_infidelity_exact':str(1-(1-I)**3),
  'three_cell_aligned_vector_distance_upper_exact':'1/50000',
  'classes':rows,'controls':controls,'all164_AB_source_connected_over_full_norm_negative_conditionally':cert,
  'complete_firstH_primary_price_exact':p['complete_action']['outward_sourceweighted_complete_firstH_upper_exact'],
  'diagnostic_definition':'source_connected_over_full_norm=t^4*(g-Ga*Gb)/F, not normalized-state sector-indicator covariance t^4*(g-Ga*Gb)/F^2; FS/mixed-log are separate.',
  'nonclaims':['No source-restricted bound on physical lattice ground','referenceg frozen local canonical256 selected harmonic2x126 only','no accepted eta from observed residual','no finite-pair envelope across unlike functionals','no allorder normalized N-uniform stability/phase/stiffness/IRatom/commoncone/pole/originalSOURCE'],
  'source_and_inputs_unchanged':before=={name:sha(ROOT/name) for name in PINS} and density_sha==sha(density_path),'natural_main_finished':True}
 assert result['source_and_inputs_unchanged']
 output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'receipt':output.name,'classes':len(rows),'conditional_all164_negative':cert,'nonzero_order_controls':len(contact),'natural_main_finished':True}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--density',required=True);parser.add_argument('--output',required=True)
 args=parser.parse_args();run(Path(args.density),Path(args.output))
