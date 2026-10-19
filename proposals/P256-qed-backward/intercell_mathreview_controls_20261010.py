"""NEW source-bound mathematical controls; never calls an old producer/run/solver."""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
from itertools import combinations
import hashlib
import json
import math
import sys
import numpy as np
from scipy.linalg import expm, block_diag, expm_frechet
from threadpoolctl import threadpool_limits
import spatial_intercell_source as source
import spatial_intercell_complete_action_parent as parent

ROOT = Path(__file__).resolve().parent
PREFIX = 'intercell_mathreview_'
FROZEN = {
 'spatial_intercell_source.py':'2c323b844cfafb27692b3e7e5f107272477c4448c92481b65b932768b7886713',
 'spatial_intercell_actual_attempt2_20261010_receipt.json':'a2f6dab92f2974292bb2c25c3b0968c80d35936b3d7fcc17391763c00526da17',
 'spatial_intercell_complete_action_parent.py':'fc822f2d15b27f4fe78c34e7c6bda643db864ce5d2e15b4de4ad712f8ddbbf6a',
 'spatial_intercell_complete_action_parent_attempt1_20261010_receipt.json':'951da598084f935725a63ef29b179ec54d3c8a87db1f17925462e0b85d2226df',
}

def sha(raw):return hashlib.sha256(raw).hexdigest()
def trace_norm(a):return float(np.linalg.svd(a,compute_uv=False).sum())
def operator(a):return float(np.linalg.norm(a,2))
def hermitian(rng,d):
 a=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));return (a+a.conj().T)/2

def rational_controls(receipt,profiles):
 bnd=receipt['uniform_bound'];g0=F(bnd['exact_g0']);gs=F(bnd['exact_gs'])
 assert 0<g0<=F(3,10) and F(1,10)<=gs<=F(11,100)
 assert F(3,10)**2<=gs<=F(1,3)**2 and g0<=F(11,20)**2
 paths=profiles[1]['paths'];s=F(1,128)
 a={(p['axis'],p['length']):abs(p['c0'])*F(11,20)+abs(p['ci'])/3 for p in paths}
 B=2*sum(a.values());z=next(p for p in paths if (p['axis'],p['length'])==(2,1))
 b=F(21,25)*abs(z['ci'])-F(11,40)*abs(z['c0'])-abs(z['ci'])/6
 edge=max(a[(j,1)]+2*a[(j,2)] for j in range(3))
 D=16*edge+32*B*s;D2=16*edge+64*s*edge+32*B*s+64*B*s*s
 H=[abs(p['onsite'])*F(11,20)+B for p in profiles[1:]]
 roles=sum(p['load']*h for p,h in zip(profiles[1:],H))
 negative=64*b*b;quadratic=4*roles*B*B+192*64*(2*s*B*D+D*D)
 quartic=192*4096*B*B*D*D;tau=F(1,262144)
 delta=-negative*tau+quadratic*tau*tau+quartic*tau**4
 b1=8*s+tau*D;b2=8*s+16*s*tau*D+tau*D2+tau*tau*D*D
 action=2*roles+192*(64*b2+4096*b1*b1)
 exact={'b_lower':b,'S_operator_upper':B,'local_S_prime_trace_upper':D,'local_S_second_trace_upper':D2,'negative_slope_per_V':-negative,'quadratic_upper_per_V':quadratic,'quartic_upper_per_V':quartic,'tau':tau,'strict_delta_H_upper_per_V':delta,'complete_body_action_norm_upper_per_V':action}
 assert all(F(bnd[k]['exact'])==v for k,v in exact.items()) and delta<-negative*tau/2
 assert -negative*2*tau+quadratic*(2*tau)**2+quartic*(2*tau)**4>=-negative*tau
 assert [p['load'] for p in profiles[1:]]==[1,63]
 assert profiles[1]['paths']==profiles[2]['paths'] and profiles[1]['onsite']!=profiles[2]['onsite']
 wrong_roles=64*H[0]
 assert roles-wrong_roles>0
 counts=[]
 for N in [4,6,8,10]:
  sites=list(np.ndindex((N,N,N)));cells={(x,y,z):(x//2,y//2,z) for x,y,z in sites}
  pairs={(cells[v],cells[(v[0],v[1],(v[2]+1)%N)]) for v in sites}
  both=pairs|{(b,a) for a,b in pairs}
  assert len(both)==N**3//2 and len(pairs)*2==len(both)
  assert all((z+2)%N not in ((z+1)%N,(z-1)%N) for z in range(N))
  # Each literal edge appears once in length1 and twice in length2.
  appearances={}
  for site in sites:
   for axis in range(3):
    for length in (1,2):
     for step in range(length):
      t=list(site);t[axis]=(t[axis]+step)%N;k=(tuple(t),axis,length)
      appearances[k]=appearances.get(k,0)+1
  assert all(n==k[2] for k,n in appearances.items())
  counts.append({'N':N,'directed_z_cell_pairs':len(both),'lower_qhp_coefficient':str(F(len(both),N**3)),'edge_path_multiplicities':[1,2]})
 assert (0+1)%2==(0-1)%2
 return {'exact':{k:str(v) for k,v in exact.items()},'display':{k:float(v) for k,v in exact.items()},'load_wrong64active_shortfall':str(roles-wrong_roles),'wrong_undirected_pair_factor':2,'excluded_N2_colliding_forward_reverse':True,'counts':counts}


def independent_h(native,profile,theta,edge=None,second=False,no_AP=False):
 V=native['V'];N=native['N'];sites=native['sites'];idx={tuple(v):i for i,v in enumerate(sites)}
 h=np.zeros((4*V,4*V),complex)
 if edge is None:h+=np.kron(np.eye(V),float(profile['onsite'])*native['frame'][0])
 for disp,coef in profile['raw_paths']:
  axis=int(np.flatnonzero(disp)[0]);length=int(disp[axis]);hop=np.einsum('m,mij->ij',coef,native['frame'])
  for x,v in enumerate(sites):
   target=idx[tuple((v+disp)%N)];edges=[]
   for step in range(length):
    t=v.copy();t[axis]=(t[axis]+step)%N;edges.append(3*idx[tuple(t)]+axis)
   count=edges.count(edge)
   if edge is not None and count==0:continue
   seam=1 if no_AP or v[axis]+length<N else -1
   q=seam*np.exp(1j*sum(theta[e] for e in edges))*hop
   if edge is not None:q*=(-count*count if second else 1j*count)
   a=slice(4*x,4*x+4);b=slice(4*target,4*target+4);h[a,b]+=q;h[b,a]+=q.conj().T
 return h


def native_controls(native,profiles,rng):
 V=native['V'];d=4*V;theta=rng.uniform(-math.pi,math.pi,3*V)
 c=source.construction(native,theta);P=c['P'];U0=c['U0'];p0=c['p0'];Q=np.eye(d)-P
 h=independent_h(native,profiles[1],theta);mask=c['mask'];hic=np.where(mask,0,h)
 assert np.max(abs(h-source.path_matrix(native,profiles[1],theta)))<1e-12
 S=hic@P-P@hic;dot=P@S-S@P;nq=float(np.linalg.norm(Q@hic@P)**2)
 role_slopes=[]
 for p in profiles[1:]:
  hr=independent_h(native,p,theta);der=complex(np.trace(hr@dot));local=complex(np.trace(np.where(mask,hr,0)@dot))
  assert abs(der+2*nq)<1e-7 and abs(local)<1e-12
  role_slopes.append({'load':p['load'],'onsite':str(p['onsite']),'first_derivative':der.real,'local_first_derivative_abs':abs(local)})
 gam=native['frame']/np.sqrt(np.trace(native['frame']@native['frame'],axis1=1,axis2=2).real/4)[:,None,None]
 z=next(p for p in profiles[1]['paths'] if (p['axis'],p['length'])==(2,1));v=float(z['c0'])*native['frame'][0]+1j*float(z['ci'])*native['frame'][3]
 hop_norm=operator(v);wrong_hop=math.sqrt(float(z['c0'])**2*native['g0']+float(z['ci'])**2*native['gs'])
 assert hop_norm>wrong_hop+1
 assert abs(hop_norm-(abs(float(z['c0']))*math.sqrt(native['g0'])+abs(float(z['ci']))*math.sqrt(native['gs'])))<1e-10
 cells=source.seed.tiles(native);ca,cb=cells[0],cells[1]
 ia=np.array([4*x+j for x in ca for j in range(4)]);ib=np.array([4*x+j for x in cb for j in range(4)])
 pa=P[np.ix_(ia,ia)];pb=P[np.ix_(ib,ib)];p00=np.kron(np.eye(4),(np.eye(4)-gam[0])/2)
 hab=hic[np.ix_(ia,ib)];A=(np.eye(16)-pa)@hab@pb;A0=(np.eye(16)-p00)@hab@p00
 expected=math.sqrt(8*float(z['ci'])**2*native['gs'])
 perturb=64/128*wrong_hop
 assert abs(np.linalg.norm(A0)-expected)<1e-10 and np.linalg.norm(A-A0)<=perturb
 assert operator(pa-p00)<=8/128 and operator(pb-p00)<=8/128
 edge=next(iter(c['derivatives']));inds,du,dp=c['derivatives'][edge]
 r0=np.zeros_like(P);r0[np.ix_(inds,inds)]=-1j*U0[np.ix_(inds,inds)].conj().T@du
 pe=np.zeros_like(P);pe[np.ix_(inds,inds)]=dp
 he=np.where(mask,0,independent_h(native,profiles[1],theta,edge));sp=he@P-P@he+hic@pe-pe@hic
 r1=1j*U0.conj().T@sp@U0
 linear_mean=abs(np.trace(p0@r1));linear_wick=abs(np.trace(p0@r0@(np.eye(d)-p0)@r1))
 wrong_linear=float(np.trace(p0@r0@(np.eye(d)-p0)@r0).real)
 assert linear_mean<1e-12 and linear_wick<1e-12 and wrong_linear>1e-5
 assert abs(np.trace(p0@r0))<1e-12
 # Magnetic translation checks against independently constructed AP signed permutations.
 flat=np.zeros(3*V);cf=source.construction(native,flat);hf=independent_h(native,profiles[1],flat)
 full=parent.translations(native,4);seen=[];maxmove=wrongmove=0.;translation_error=0.
 idx={tuple(v):i for i,v in enumerate(native['sites'])}
 for k,(x,y,z0) in enumerate((x,y,z0) for x in range(0,4,2) for y in range(0,4,2) for z0 in range(4)):
  shift=np.array([x,y,z0]);T=np.zeros((d,d))
  for j,vv in enumerate(native['sites']):
   tgt=idx[tuple((vv+shift)%4)];sgn=(-1)**int(np.sum((vv+shift)//4));T[4*tgt:4*tgt+4,4*j:4*j+4]=sgn*np.eye(4)
  translation_error=max(translation_error,float(np.max(abs(parent.move(hf,full[k])-T@hf@T.T))))
  maxmove=max(maxmove,float(np.max(abs(T@hf@T.T-hf))),float(np.max(abs(T@cf['U0']@T.T-cf['U0']))))
  wrongmove=max(wrongmove,float(np.max(abs(abs(T)@hf@abs(T).T-hf))))
  for site in cells[0]:
   tgt=idx[tuple((native['sites'][site]+shift)%4)]
   seen.extend(3*tgt+j for j in range(3))
 assert sorted(seen)==list(range(3*V)) and maxmove<1e-10 and translation_error<1e-10 and wrongmove>1
 second_edge=3*cells[0][0]
 hp2=independent_h(native,profiles[1],flat,second_edge,second=True)
 assert np.max(abs(hp2-parent.path_second(native,profiles[1],flat,second_edge)))<1e-12
 # Pure phase shared-copy Wick control: 64 copies of exp(i faceflux), ||face||^2=4.
 copy_price=64*64**2*4;independent_price=64*64*4
 assert copy_price==1048576 and copy_price/independent_price==64
 return {'N':4,'field_seed':'uniform deterministic seed; not Haar quadrature','roles':role_slopes,'qhp_HS_squared':nq,'same_field_all64_first_derivative':-128*nq,'z_pair_onsite_HS':float(np.linalg.norm(A0)),'z_pair_expected_HS':expected,'z_pair_perturbation_HS':float(np.linalg.norm(A-A0)),'z_pair_perturbation_bound':perturb,'hop_operator':hop_norm,'wrong_nonnormal_sqrt_operator':wrong_hop,'offcell_linear_mean_abs':linear_mean,'offcell_linear_wick_abs':linear_wick,'wrong_intracell_linear_wick':wrong_linear,'seed_Berry_mean_abs':abs(np.trace(p0@r0)),'translation_group_order':len(full),'edge_representatives':12,'full_edges':len(seen),'translation_defect':maxmove,'independent_translation_implementation_defect':translation_error,'wrong_unsigned_translation_defect':wrongmove,'shared_phase_fullE_price':copy_price,'wrong_independent_phase_price':independent_price}


def exponential_controls(rng):
 d=8;S=1j*hermitian(rng,d);S/=operator(S)
 v=rng.normal(size=d)+1j*rng.normal(size=d);v/=np.linalg.norm(v);Sp=1j*np.outer(v,v.conj());tau=.4
 derivative=expm_frechet(-tau*S,-tau*Sp,compute_expm=False)
 singular=np.linalg.svd(derivative,compute_uv=False);rank=int(np.sum(singular>1e-13))
 # A rank-one insertion spreads beyond rank one through noncommuting Duhamel transport.
 assert rank>1
 X=-1j*expm(-tau*S).conj().T@derivative;r1=1j*Sp
 delta=trace_norm(X);remainder=trace_norm(X-tau*r1)
 assert delta<=tau*trace_norm(Sp)+1e-12 and remainder<=tau*tau*operator(S)*trace_norm(Sp)+1e-12
 A=1j*hermitian(rng,4);A*=.8/operator(A);E=1j*hermitian(rng,4);E*=.3/operator(E);F2=1j*hermitian(rng,4);F2*=.2/operator(F2)
 block=np.zeros((12,12),complex)
 for j in range(3):block[4*j:4*j+4,4*j:4*j+4]=A
 block[:4,4:8]=E;block[4:8,8:]=E;block[:4,8:]=F2/2
 exact=expm(block);poly,first,second=parent.exponential_derivatives(A,E,F2)
 a=operator(A);e=operator(E);f=operator(F2)
 errors=[operator(poly-exact[:4,:4]),operator(first-exact[:4,4:8]),operator(second-2*exact[:4,8:])]
 bounds=[math.exp(a)*a**9/math.factorial(9),math.exp(a)*e*a**8/math.factorial(8),math.exp(a)*(f*a**8/math.factorial(8)+e*e*a**7/math.factorial(7))]
 assert all(x<=b for x,b in zip(errors,bounds))
 wrong=expm(A)@(F2+E@E);wrongerr=operator(wrong-2*exact[:4,8:]);assert wrongerr>1e-3
 # F=0 must still have a second-derivative tail from TWO E insertions.
 block[:4,8:]=0;exact0=expm(block);_,_,second0=parent.exponential_derivatives(A,E,np.zeros_like(A))
 omit_E2_tail_error=operator(second0-2*exact0[:4,8:]);assert omit_E2_tail_error>1e-8
 return {'insertion_rank':1,'exponential_derivative_rank':rank,'derivative_singular_values':singular.tolist(),'Duhamel_trace_delta':delta,'Duhamel_trace_remainder':remainder,'Duhamel_bounds':[tau,tau*tau],'noncommuting_series_errors':errors,'analytic_series_tail_bounds':bounds,'wrong_commuting_second_derivative_error':wrongerr,'wrong_omitted_two_insertion_tail_error':omit_E2_tail_error,'analytic_tail_is_not_roundoff_enclosure':True}


def fock_one_body(h,basis,index):
 out=np.zeros((len(basis),len(basis)),complex);d=len(h)
 for j,word in enumerate(basis):
  for q in range(d):
   if not word>>q&1:continue
   after=word^(1<<q);s1=(-1)**((word&((1<<q)-1)).bit_count())
   for p in range(d):
    if after>>p&1:continue
    dest=after|(1<<p);s2=(-1)**((after&((1<<p)-1)).bit_count());out[index[dest],j]+=s1*s2*h[p,q]
 return out


def firstH_fock_controls(rng):
 n=2;G=1.7;d=4;occ=[0,1];empty=[2,3]
 hs=[hermitian(rng,d),hermitian(rng,d)];Rs=[hermitian(rng,d) for _ in range(3)];Ts=[hermitian(rng,d) for _ in range(3)]
 basis=[sum(1<<j for j in choice) for choice in combinations(range(n*d),n*2)];index={w:i for i,w in enumerate(basis)}
 word=sum(1<<j for j in [0,1,4,5]);ref=np.zeros(len(basis));ref[index[word]]=1
 action=fock_one_body(block_diag(*hs),basis,index)@ref
 for R,T in zip(Rs,Ts):
  RF=fock_one_body(block_diag(*([R]*n)),basis,index);TF=fock_one_body(block_diag(*([T]*n)),basis,index)
  action+=G*(RF@RF-1j*TF)@ref
 ph=np.array([(w&sum(1<<j for j in [2,3,6,7])).bit_count() for w in basis]);norms=[float(np.vdot(action[ph==j],action[ph==j]).real) for j in range(3)]
 zero=sum(np.trace(h[np.ix_(occ,occ)]) for h in hs);el=np.zeros((2,2),complex);Ks=[]
 for R,T in zip(Rs,Ts):
  K=R[np.ix_(empty,occ)];Ks.append(K);roo=R[np.ix_(occ,occ)];ree=R[np.ix_(empty,empty)];mu=n*np.trace(roo)
  zero+=G*(mu*mu+n*np.vdot(K,K).real-1j*n*np.trace(T[np.ix_(occ,occ)]))
  el+=G*(2*mu*K+ree@K-K@roo-1j*T[np.ix_(empty,occ)])
 one=sum(np.vdot(h[np.ix_(empty,occ)]+el,h[np.ix_(empty,occ)]+el).real for h in hs)
 pair=0j;ind=0j;noexchange=0j
 for K in Ks:
  for L in Ks:
   Z=K.conj().T@L;tr=np.trace(Z);exchange=np.trace(Z@Z)
   pair+=2*G*G*(n*n*tr*tr-n*exchange)
   ind+=2*G*G*n*(tr*tr-exchange);noexchange+=2*G*G*n*n*tr*tr
 expected=[abs(zero)**2,float(one),float(pair.real)]
 assert max(abs(a-b) for a,b in zip(norms,expected))<1e-9 and np.linalg.norm(action[ph>2])==0
 wrong_average=n*np.vdot((hs[0][np.ix_(empty,occ)]+hs[1][np.ix_(empty,occ)])/n+el,(hs[0][np.ix_(empty,occ)]+hs[1][np.ix_(empty,occ)])/n+el).real
 assert one-wrong_average>1 and abs(pair-ind)>1 and abs(pair-noexchange)>1
 # Separate Gaussian source has genuine THREE-PH support, even though first H has only 0/1/2 PH.
 U=expm(np.block([[np.zeros((3,3)), -.4*np.eye(3)],[.4*np.eye(3),np.zeros((3,3))]]));W=U[:,:3]
 amp3=complex(np.linalg.det(W[3:,:]));assert abs(amp3)>1e-2
 return {'Fock_orbitals':8,'copies':n,'basis_dimension':len(basis),'electric_directions':3,'direct_firstH_PH_norms':norms,'Gaussian_formula_PH_norms':expected,'maximum_formula_defect':max(abs(a-b) for a,b in zip(norms,expected)),'wrong_independent_twoPH':float(ind.real),'wrong_no_exchange_twoPH':float(noexchange.real),'wrong_role_averaged_onePH':float(wrong_average),'source_gaussian_threePH_amplitude_abs':abs(amp3),'firstH_012PH_is_action_identity_not_source_crop':True}


def main():
 captured={n:(ROOT/n).read_bytes() for n in FROZEN}
 assert all(sha(captured[n])==h for n,h in FROZEN.items())
 receipt=json.loads(captured['spatial_intercell_actual_attempt2_20261010_receipt.json'])
 input_name='induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json'
 input_raw=(ROOT/input_name).read_bytes();cfg=json.loads(input_raw)['input_descriptor']['original_micro_config_UNCHANGED']
 g0=F(receipt['uniform_bound']['exact_g0']);gs=F(receipt['uniform_bound']['exact_gs'])
 # Pure assembly from already saved rationals/config: no root, mean solver or old delivered producer.
 mean={'microscopic_config':cfg,'centre_enclosure':{name+'_'+edge:{'numerator':str(v.numerator),'denominator':str(v.denominator)} for name,v in [('g0',g0),('gs',gs)] for edge in ['lower','upper']}}
 profiles=source.seed.field.full.scalar.fast.primitive_profiles(mean)
 rng=np.random.default_rng(222101047)
 with threadpool_limits(limits=1):
  native=source.seed.field.full.supplier.quantum.build_native(mean,4)
  controls={'rational':rational_controls(receipt,profiles),'native':native_controls(native,profiles,rng),'exponential':exponential_controls(rng),'firstH_Fock':firstH_fock_controls(rng)}
 binding={**{n:sha(b) for n,b in captured.items()},input_name:sha(input_raw)}
 binding.update({n:sha(b) for n,b in source.seed.SOURCE_BYTES.items()});binding.update({n:sha(b) for n,b in source.seed.field.CAPTURED.items()})
 binding[Path(__file__).name]=sha(Path(__file__).read_bytes())
 assert all((ROOT/n).read_bytes()==b for n,b in captured.items())
 result={'kind':'NEW_INDEPENDENT_INTERCELL_MATH_CONSUMER_CONTROLS','seed':222101047,'source_input_hashes':binding,'runtime':{'python':sys.executable,'version':sys.version,'numpy':np.__version__},'old_producers_or_solvers_run':False,'controls':controls,'scope':'finite mathematical/consumer adverse controls plus independent exact rational arithmetic; not Haar variance, stationarity or physical approval','natural_exit':True}
 path=ROOT/(PREFIX+'controls_20261010.json');path.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'capsule':str(path),'sha256':sha(path.read_bytes()),'control_groups':list(controls),'exact_delta':controls['rational']['exact']['strict_delta_H_upper_per_V'],'Fock_defect':controls['firstH_Fock']['maximum_formula_defect']}))

if __name__=='__main__':main()
