"""Independent exact arithmetic and full-CAR adverse controls; never imports producers."""
import argparse
import zipfile
from fractions import Fraction as R
from pathlib import Path
import hashlib
import itertools
import json
import math
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
NAMES = {
 'three_cell_metric_transfer_proof_20261011.json':'3d45c4b98e0c3c236882575d05a90129b7e5c352d3ef53a6dd83922f79583b1e',
 'three_cell_metric_transfer_consumer_20261011.py':'a726c6187cafc3951e4c86d05777fe36e965029d284eb89bbcd6a502ecc4b11d',
 'three_cell_metric_transfer_actual1_20261011_receipt.json':'5f060dfa360457b96de85d4e11b1a4e79be9ec1b5e29d38449b456275358f3f0',
 'three_cell_metric_transfer_actual1_20261011_executed.zip':'be1732b5b202d1f228de76b475f74125443f9c535a73fffd8c149877db771b19',
 'three_cell_density_formation_interface_20261011.json':'c40506cde408eb241602bd2e79696382c7cf0b3f9b55fee8d913930bbecd9318',
 'three_cell_native_actual3_20261011_receipt.json':'1616f84b0a4c09f7ab1905d0b158601cd2fa7e082a4ee4e1312b35e1dcc7a741',
 'charged_first_shell_actual04_20261011_receipt.json':'b48aed4cb67ae2fd98e0b75ee4c58053244c68e1e4bd47c4cb97c9fdf58e4f02',
 'harmonic_car_bridge_typed_attempt3_20261011_receipt.json':'be07db74d79630fd62866e4a44eb32e8b9217eaf59bf2ce1f26fdc247e91ff21',
 'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz':'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
}

def digest(name):
 if name=='three_cell_metric_transfer_consumer_20261011.py':
  with zipfile.ZipFile(ROOT/'three_cell_metric_transfer_actual1_20261011_executed.zip') as archive:
   raw=archive.read(name)
 else:raw=(ROOT/name).read_bytes()
 return hashlib.sha256(raw).hexdigest()

def load(name):
 return json.loads((ROOT/name).read_bytes())

class Box:
 def __init__(self, left, right=None):
  a,b=R(left),R(left if right is None else right)
  assert a<=b
  scale=1<<96
  self.a=R(a.numerator*scale//a.denominator,scale)
  self.b=R((b.numerator*scale+b.denominator-1)//b.denominator,scale)
 def __add__(self,o):
  o=box(o);return Box(self.a+o.a,self.b+o.b)
 __radd__=__add__
 def __neg__(self):return Box(-self.b,-self.a)
 def __sub__(self,o):return self+-box(o)
 def __mul__(self,o):
  o=box(o);products=[a*b for a in (self.a,self.b) for b in (o.a,o.b)]
  return Box(min(products),max(products))
 __rmul__=__mul__
 def __truediv__(self,o):
  o=box(o);assert o.a*o.b>0
  return self*Box(1/o.b,1/o.a)
 def sq(self):return Box(0 if self.a<=0<=self.b else min(self.a**2,self.b**2),max(self.a**2,self.b**2))
 def pair(self):return [str(self.a),str(self.b)]

def box(o):return o if isinstance(o,Box) else Box(o)

def upper_root(a):
 a=R(a);assert a>=0
 scale=1<<80
 k=math.isqrt(a.numerator*scale**2//a.denominator)
 return R(k+(k*k*a.denominator<a.numerator*scale**2),scale)

def log_end(a):
 z=(a-1)/(a+1);assert abs(z)<R(1,2)
 terms=[2*z**n/n for n in range(1,48,2)]
 remainder=2*abs(z)**49/(49*(1-z*z))
 return Box(sum(terms,R(0))-remainder,sum(terms,R(0))+remainder)

def log_box(a):return Box(log_end(a.a).a,log_end(a.b).b)

def metrics(a,b,c,t):
 v=t*t;F=1+v*(a+b)+v*v*c;legs=(1+v*a)*(1+v*b);conn=c-a*b
 return {'F':F,'leg_product':legs,'connected_coefficient':conn,
 'normalized_covariance':v*v*conn/F,'FS_real_mixed':v*conn/F.sq(),
 'mixed_log_derivative':4*v*conn/F.sq(),'connected_log':log_box(F/legs)}

def gamma(n):return R(n,2**53-n)

def price(la,lb,delta,order):
 total=R(0);beta=2500*gamma(8192);kap=64*gamma(8192)
 patterns=[('AA','AA','AA',1),('UU','UU','UU',63),('AA','AU','UU',63),('UU','AU','AA',63),('UU','UU','UU',3906)] if order else [('AA','AA','AA',1),('UU','UU','UU',63)]
 for pat in patterns:
  if order:
   roles=pat[:3];mult=pat[3];coeff=la*la*lb*lb;degree=4
  else:
   roles=pat[:2];mult=pat[3];coeff=la*la;degree=2
  factor=math.prod((1+delta[r])*(1+beta) for r in roles)
  total+=mult*coeff*((1+kap)**degree*factor*(1+gamma(2**20 if order else 8192))-1)
 return total

def check_physical(receipt):
 bridge=load('harmonic_car_bridge_typed_attempt3_20261011_receipt.json')
 J=sp.Matrix([[sp.sympify(z) for z in row] for row in bridge['controls']['joint_n1_native_embedding']])
 assert J.T.conjugate()*J==sp.eye(5)
 assert all(sum(x!=0 for x in J.row(i))==1 for i in range(5))
 assert all(x in (-1,0,1) for x in J)
 V=sp.zeros(16,5)
 for col,mask in enumerate([5,3,12,10]):V[mask,col]=1
 V[9,4]=V[6,4]=1/sp.sqrt(2)
 W=V*J*sp.diag(1,-sp.I,1,1,sp.I)
 assert W.T.conjugate()*W==sp.eye(5)
 U=sp.zeros(25,15);col=0
 for i in range(5):
  for j in range(i,5):
   U[5*i+j,col]=1 if i==j else 1/sp.sqrt(2)
   if i!=j:U[5*j+i,col]=1/sp.sqrt(2)
   col+=1
 assert U.T*U==sp.eye(15)
 E=sp.kronecker_product(W,W)*U
 # Absolute scalar trace path sums: like-role 225*8; unlike-role625*4.
 col_l1=[sum((abs(x) for x in E.col(i)),sp.Integer(0)) for i in range(15)]
 assert all(sp.simplify(x*x)<=8 for x in col_l1)
 like_path_cap=15**2*8;unlike_path_cap=25**2*4
 assert max(like_path_cap,unlike_path_cap)<=2500
 with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',allow_pickle=False) as z:
  phi=z['Phi_binary_section'][1]
 exact=lambda x:sp.Rational(R(float(x.real)))+sp.I*sp.Rational(R(float(x.imag)))
 H=sp.Matrix([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])
 frame=H.T*sp.Matrix([[exact(x) for x in row] for row in phi])*H/2
 amplitude=sp.sqrt(sp.trace(frame*frame)/4)
 assert sp.Rational(1,4)<amplitude<sp.Rational(1,2)
 G=sp.simplify(frame/amplitude)
 S=sp.Matrix.hstack(sp.eye(4)[:,0],G[:,0],sp.eye(4)[:,1],G[:,1])
 assert S.T*S==sp.eye(4)
 assert [[str(x) for x in S.row(i)] for i in range(4)]==receipt['physical_S_exact']
 boundary=load('charged_first_shell_actual04_20261011_receipt.json')['all64_boundary']
 P=sp.diag(1,-1,1,-1);prices={};ls={}
 for axis in range(3):
  total=R(0)
  for role in boundary['roles']:
   raw=next(p for p in role['paths'] if p['axis']==axis and p['length']==1)['matrix_exact']
   M=sp.Matrix([[sp.Rational(R(a))+sp.I*sp.Rational(R(b)) for a,b in row] for row in raw])
   assert sum(x!=0 for x in M)<=8
   total+=role['count']*M.rank()*R(str(sp.trace(M.T.conjugate()*M)))
   rotated=P*S.T*M*S*P
   assert [[str(x) for x in rotated.row(i)] for i in range(4)]==receipt['physical_endpoint00_M_exact'][str(axis)]
   ls[axis]=sum(abs(R(str(sp.re(x))))+abs(R(str(sp.im(x)))) for x in rotated)
  prices[axis]=total
  assert str(total)==receipt['domain_prices'][str(axis)]['Sa_exact']
  assert str(ls[axis])==receipt['coefficient_l1_prices'][str(axis)]
 return prices,ls,{'W_isometry':True,'J_signed_permutation':True,'S_permutation':True,'Phi_amplitude_exact':str(amplitude),'like_role_absolute_path_cap':like_path_cap,'unlike_role_absolute_path_cap':unlike_path_cap}

def car_apply(mask,mode,create):
 if bool(mask&(1<<mode))==create:return None
 return mask^(1<<mode),(-1)**((mask&((1<<mode)-1)).bit_count())

def hop(state):
 out={}
 for mask,coef in state.items():
  for f in range(6):
   first=car_apply(mask,2*f+1,False)
   if first is None:continue
   second=car_apply(first[0],2*f,True)
   if second is not None:out[second[0]]=out.get(second[0],0)+coef*first[1]*second[1]
 return {m:z for m,z in out.items() if z}

def dicke():
 state={sum(1<<(2*f+1) for f in range(6)):1}
 for _ in range(3):state=hop(state)
 after=hop(state)
 n=sum(z*z for z in state.values());m=sum(z*z for z in after.values())
 assert len(state)==20 and set(abs(z) for z in state.values())=={6}
 assert R(m,n)==12>6
 return {'full_12_mode_CAR':True,'input_norm_square':n,'output_norm_square':m,'ratio':'12','wrong_global_squared_cap':'6'}

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--output',required=True)
 args=parser.parse_args()
 before={n:digest(n) for n in NAMES};assert before==NAMES
 r=load('three_cell_metric_transfer_actual1_20261011_receipt.json')
 raw=load('three_cell_native_actual3_20261011_receipt.json')
 interface=load('three_cell_density_formation_interface_20261011.json')
 ds={role:R(interface['errors'][role]['delta_trace_norm_exact']) for role in ('AA','UU','AU')}
 assert R(raw['source_normalization_divisor_float'])==R(interface['common_N0_exact'])
 prices,ls,physical=check_physical(r)
 I=R(274877906944,2204801412260314559041);I3=1-(1-I)**3
 assert I3<=3*I and 3*I/(1-3*I)<R(1,50000)**2
 edges=raw['edges'];accepted={(i,j) for i in range(10) for j in range(i+1,10) if edges[i]['outer_cell']!=edges[j]['outer_cell']}
 assert len(accepted)==41
 expected={(i,j,oa,ob) for i,j in accepted for oa,ob in itertools.product((-1,1),repeat=2)}
 seen=set();counts={};contact_count=0;mutant_rejections=0;fields=0
 for a,b in zip(raw['classes'],r['classes'],strict=True):
  key=tuple(a['edges']+a['orientations']);seen.add(key)
  assert a['edges']==b['edges'] and a['orientations']==b['orientations']
  assert a['cells'][0]!=a['cells'][2]
  ax,bx=[edges[i]['axis'] for i in a['edges']]
  pa=price(ls[ax],ls[ax],ds,False);pb=price(ls[bx],ls[bx],ds,False);pc=price(ls[ax],ls[bx],ds,True)
  assert b['formation_errors']=={'Ga':str(pa),'Gb':str(pb),'ordered_Gram_and_overlap':str(pc)}
  source={}
  for n,p in [('Ga',pa),('Gb',pb),('Gab',pc),('Gba',pc)]:
   v=R(a[n]);source[n]=Box(max(0,v-p),v+p)
  for n,ix in [('ReD',0),('ImD',1)]:
   v=R(a['ordered_overlap'][ix]);source[n]=Box(v-pc,v+pc)
  ra=upper_root(prices[ax])/50000;rb=upper_root(prices[bx])/50000;rab=upper_root(prices[ax]*prices[bx])/50000
  gerr={'Ga':2*upper_root(source['Ga'].b)*ra+ra*ra,'Gb':2*upper_root(source['Gb'].b)*rb+rb*rb,
   'Gab':2*upper_root(source['Gab'].b)*rab+rab*rab,'Gba':2*upper_root(source['Gba'].b)*rab+rab*rab,
   'ReD':(upper_root(source['Gab'].b)+upper_root(source['Gba'].b))*rab+rab*rab}
  assert b['reference_replacement_errors']=={n:str(p) for n,p in gerr.items()}
  ground={n:Box(max(0,v.a-gerr[n]) if n.startswith('G') else v.a-gerr[n],v.b+gerr[n]) for n,v in source.items() if n!='ImD'}
  ground['ImD']=Box(source['ImD'].a-gerr['ReD'],source['ImD'].b+gerr['ReD'])
  t=R(a['t_diagnostic']);assert str(t)==b['t_exact_binary']
  for phase,vals in [('normalized_s',source),('reference_g',ground)]:
   actual=b['intervals'][phase]
   assert actual['moments']=={n:v.pair() for n,v in vals.items()}
   coh=(vals['Gab']+vals['Gba']+2*vals['ReD'])*R(1,4);coh=Box(max(0,coh.a),coh.b)
   assert actual['coherent_half_sum_norm_square']==coh.pair()
   for order,g in [('AB',vals['Gab']),('BA',vals['Gba']),('coherent_half_sum',coh)]:
    derived={n:v.pair() for n,v in metrics(vals['Ga'],vals['Gb'],g,t).items()}
    assert derived==actual['orders'][order]
    fields+=len(derived)
    cv=Box(*derived['normalized_covariance']);sign='negative' if cv.b<0 else 'positive' if cv.a>0 else 'contains_zero'
    counts.setdefault(phase+'_'+order,{'negative':0,'positive':0,'contains_zero':0})[sign]+=1
   fab=metrics(vals['Ga'],vals['Gb'],vals['Gab'],t)['F'];fba=metrics(vals['Ga'],vals['Gb'],vals['Gba'],t)['F']
   norms=fab*fba;den=Box(max(0,upper_root(norms.a)-R(1,2**80)),upper_root(norms.b))
   v=t*t
   kernels={'AB_BA_normalized_kernel_real':((1+v*v*vals['ReD'])/den).pair(),
    'AB_BA_normalized_kernel_imag':((v*(vals['Gb']-vals['Ga'])+v*v*vals['ImD'])/den).pair(),
    'AB_AB_normalized_kernel_real':((1+v*v*vals['Gab'])/fab).pair(),
    'AB_AB_normalized_kernel_imag':(v*(vals['Gb']-vals['Ga'])/fab).pair()}
   assert all(actual['complex_ordered_probe'][n]==p for n,p in kernels.items());fields+=4
  ct=source['Gab']+source['Gba']-2*source['ReD']
  if ct.a>0:contact_count+=1
  # Wrong incoherent mean differs from coherent mean by ||ABs-BAs||^2/4.
  incoherent=(source['Gab']+source['Gba'])*R(1,2)
  coherent=Box(*b['intervals']['normalized_s']['coherent_half_sum_norm_square'])
  if (incoherent-coherent).a>0:mutant_rejections+=1
 assert seen==expected and len(seen)==164
 assert counts['reference_g_AB']=={'negative':40,'positive':40,'contains_zero':84}
 # Trace equality alone cannot imply a local expectation error bound.
 rho=sp.diag(1,0);wrong=sp.diag(0,1);observable=sp.diag(1,-1)
 assert sp.trace(rho-wrong)==0 and abs(sp.trace(observable*(rho-wrong)))==2
 # Complex coherent weight alpha=1,beta=i exposes imaginary ordered overlap.
 synthetic_D=sp.Rational(1,4)+sp.I*sp.Rational(1,3)
 exact_coherent=2+2*sp.re(sp.I*synthetic_D)
 wrong_conjugate=2+2*sp.re(sp.I*sp.conjugate(synthetic_D))
 assert exact_coherent!=wrong_conjugate
 # Omitting the log tail cannot enclose log(2): the positive omitted terms exceed zero.
 assert ((R(2)-1)/(R(2)+1))**49>0
 assert all(digest(n)==h for n,h in before.items())
 out={'kind':'INDEPENDENT_EXACT_SOURCE_METRIC_REVIEW_CONTROL','status':'PASS_AT_CONDITIONAL_SCOPE',
 'input_SHA':before,'own_source_SHA':digest(Path(__file__).name),'physical_controls':physical,
 'frozen_target_source_archive':'three_cell_metric_transfer_actual1_20261011_executed.zip; target producer is the immutable a726 epoch, not the renamed live producer',
 'Sa_exact':{str(k):str(v) for k,v in prices.items()},'Sa_display':{str(k):float(v) for k,v in prices.items()},
 'all164_interval_fields_exact_match':True,'metric_and_kernel_fields_checked':fields,'class_counts':counts,
 'controls':{'global_domain_Dicke':dicke(),'ordered_contact_strict_positive_classes':contact_count,
 'wrong_incoherent_mean_rejected_classes':mutant_rejections,'trace_only_price_rejected_expectation_difference':2,
 'complex_overlap_conjugation_exact':str(exact_coherent),'complex_overlap_conjugation_wrong':str(wrong_conjugate),
 'log_tail_omission_rejected_by_positive_first_omitted_term':True,'all_164_distinct_outer_geometry_exhaustive':True},
 'three_cell_squared_infidelity_exact':str(I3),'vector_price':'1/50000',
 'native_formation_is_conditional_not_historical_machine_attestation':True,'unchanged_inputs':True,'natural_main_finished':True}
 Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'fields':fields,'counts':counts,'contact_controls':contact_count,'natural_main_finished':True}),flush=True)

if __name__=='__main__':main()
