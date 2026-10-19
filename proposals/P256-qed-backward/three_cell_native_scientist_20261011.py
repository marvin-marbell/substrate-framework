"""Distinct physical-edge three-cell coherent CAR kernel and complete-action bounds.
Only new finite frozen-source science; no lattice ground or linked theorem.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import bulk_electric_phase_consumer as geometry
import connected_root_exterior_scientist_20261011 as car

ROOT = Path(__file__).resolve().parent
DENSITY = 'connected_root_exterior_actual1_20261011_moments.npz'
PINS = {
 DENSITY:'4c6c6e0da22e0d9d405c37935332bb9548fbf2df5f429ec66b50aa671f8af67d',
 'local_so5_resum_actual04_20261011_state.npz':'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2',
 'charged_first_shell_actual04_20261011_receipt.json':'b48aed4cb67ae2fd98e0b75ee4c58053244c68e1e4bd47c4cb97c9fdf58e4f02',
 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json':'102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
 'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz':'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
 'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json':'2f019dd49873a5e13f130c65ef6739b9af78a16f85138dc6c6ab6e32e65c7a12',
 'harmonic_car_bridge_typed_attempt3_20261011_receipt.json':'be07db74d79630fd62866e4a44eb32e8b9217eaf59bf2ce1f26fdc247e91ff21',
 'connected_root_exterior_graph_interface_corrected_20261011.json':'becd093cdcb851d60678fdc89180d249d95cbd4217364eecbe2e191a8e3fcc03',
}
PATTERNS = [(0,0,1,'same_active'),(1,1,63,'same_auxiliary'),
 (0,1,63,'distinct_active_auxiliary'),(1,0,63,'distinct_auxiliary_active'),
 (1,2,3906,'distinct_auxiliary_auxiliary')]

def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def enc(a):
 a=np.asarray(a);return np.stack([a.real,a.imag],axis=-1).tolist()

def cell(x):
 return (x[0]//2,x[1],x[2])

def inventory(n):
 box=geometry.IntegerBox(n); B=(2,2,2); points=[(4,2,2),(5,2,2)]
 edges=[]
 for site,p in enumerate(points):
  for axis,sgn in itertools.product(range(3),[-1,1]):
   other=box.step(p,axis,sgn)
   if cell(other)==B:continue
   tail,head=(p,other) if sgn==1 else (other,p)
   inc=box.path(tail,axis,1)
   expected=np.zeros(box.volume,dtype=np.int64)
   expected[box.index[tail]]=-1;expected[box.index[head]]=1
   assert np.array_equal(-box.divergence(inc),expected)
   edges.append({'axis':axis,'tail':tail,'head':head,'central_site':site,
    'central_is_tail':sgn==1,'outer_site':other[0]%2,'outer_cell':cell(other),
    'raw_edge':int(np.flatnonzero(inc)[0])})
 rows=[]
 for i,j in itertools.combinations(range(len(edges)),2):
  a,b=edges[i],edges[j]
  if a['outer_cell']==b['outer_cell']:continue
  for oa,ob in itertools.product([-1,1],repeat=2):
   flow=np.zeros(3*box.volume,dtype=np.int64)
   flow[a['raw_edge']]=-oa;flow[b['raw_edge']]=-ob
   delta=-box.divergence(flow)
   selected=[a['outer_cell'],B,b['outer_cell']]
   charges=[];us=[];sites=[]
   for c in selected:
    p=(2*c[0],c[1],c[2]);q=box.step(p,0,1)
    d0=int(delta[box.index[p]]);d1=int(delta[box.index[q]])
    charge=d0+d1;internal=int(flow[3*box.index[p]])
    # Ebase=-rho, rho'=rho+d0-charge/2, E'=Ebase+internal.
    u=Q(charge,2)-d0-internal
    charges.append(charge);us.append(str(u));sites.append([d0,d1])
   assert sum(charges)==0 and np.count_nonzero(flow)==2
   assert int(flow@flow)==2 and all(flow[3*box.index[(2*c[0],c[1],c[2])]]==0 for c in selected)
   rows.append({'edges':[i,j],'orientations':[oa,ob],'cells':selected,
    'cell_charge':charges,'canonical_numbers':[256+q for q in charges],
    'delta_site_counts':sites,'affine_u':us,'external_HE_increment_exact':'128',
    'once_shared_integer_flow':[[a['raw_edge'],-oa],[b['raw_edge'],-ob]],
    'central_endpoint_class':'same' if a['central_site']==b['central_site'] else 'opposite',
    'central_incoming_u_applicable_neutral':us[1] if charges[1]==0 else None})
 return edges,rows

def all_native_gauss(n):
 box=geometry.IntegerBox(n)
 internal={3*box.index[x] for x in box.sites if x[0]%2==0}
 external=set(range(3*box.volume))-internal
 paths=[];native_inside=native_outside=0
 for x,axis,length in itertools.product(box.sites,range(3),(1,2)):
  y=box.step(x,axis,length);inc=box.path(x,axis,length)
  delta=-box.divergence(-inc);expected=np.zeros(box.volume,dtype=np.int64)
  expected[box.index[x]]=1;expected[box.index[y]]=-1
  assert np.array_equal(delta,expected)
  inside=cell(x)==cell(y)
  if inside:
   assert axis==0 and length==1 and x[0]%2==0
   native_inside+=1
  else:
   native_outside+=1
   paths.append({'axis':axis,'length':length,'tail_site':x[0]%2,'head_site':y[0]%2,
    'external_raw_count':len(set(np.flatnonzero(inc))&external)})
 # Independent integer witness covering every raw edge, not just root paths.
 rng=np.random.default_rng(20261011);E=rng.integers(-3,4,3*box.volume,dtype=np.int64)
 extE=E.copy();extE[list(internal)]=0
 totalrho=-box.divergence(E);dext=-box.divergence(extE)
 internal_square=0;allocated=0
 for x in box.sites:
  if x[0]%2:continue
  y=box.step(x,0,1);i,j=box.index[x],box.index[y]
  charge=int(totalrho[i]+totalrho[j]);rho=Q(int(totalrho[i]-totalrho[j]),2)
  u=Q(charge,2)-int(dext[i])
  assert Q(int(E[3*i]))==-(rho+u)
  internal_square+=(rho+u)**2
 for edge in external:
  tail=box.sites[int(box.tail[edge])];head=box.sites[int(box.head[edge])]
  assert cell(tail)!=cell(head)
  allocated+=2*32*int(E[edge])**2
 assert Q(int(E@E))==internal_square+int(extE@extE)
 assert allocated==64*int(extE@extE)
 counts=Counter(tuple(p.values()) for p in paths)
 return {'N':n,'raw_internal_matching_edges':len(internal),'raw_external_edges':len(external),
  'native_inside_pair_paths':native_inside,'native_V_paths':native_outside,
  'all_positive_one_two_link_paths':native_inside+native_outside,
  'path_classes':[dict(zip(['axis','length','tail_site','head_site','external_raw_count'],key))|{'count':value} for key,value in sorted(counts.items())],
  'D_out':'row-col = -IntegerBox.divergence',
  'exact_identity':'Q=d0+d1; u=Q/2-d0=(d1-d0)/2; E_internal=-(rho+u); HE+HC=64[sum_cells(rho+u)^2+sum_external k_e^2]',
  'allocation':'Each raw external edge contributes32k_e^2 to each blocked endpoint, total64k_e^2, regardless of primitive-path decompositions.',
  'weighted_degree_cases':{'0':'Q0,u0','1':'Q+/-1,u+/-1/2','2':'Q0,u0/+/-1 OR Q+/-2,u0/+/-1',
   'at_least3':'allocation>=96; weak lower applicability belongs to parent','single_edge_abs_k2':'weighteddegree4, NOT degree1'},
  'integer_witness_HE':int(64*(E@E)),'integer_witness_decomposition_HE':str(64*(internal_square+int(extE@extE))),
  'integer_witness_external_allocation':allocated,'no_globalH0_ground_assumed':True,
  'scope':'Literal fixedRef4/beta0/frozenPhi integer-flow geometry; not real-cover/allbeta/fullPhi ground'}

class Kernel:
 def __init__(self):
  self.W=car.slot_embedding()
  self.ann=[]
  for p in range(4):
   a=np.zeros((16,16))
   for m in range(16):
    if m>>p&1:a[m^(1<<p),m]=(-1)**((m&((1<<p)-1)).bit_count())
   self.ann.append(a)
  source_receipt=json.loads((ROOT/'harmonic_car_bridge_typed_attempt3_20261011_receipt.json').read_bytes())
  self.N0=float(source_receipt['source_norm_float'])
  with np.load(ROOT/DENSITY,allow_pickle=False) as z:
   self.two={}
   for roles,key in [((0,0),'active_pair'),((1,1),'auxiliary_pair'),((0,1),'mixed_pair')]:
    U=car.pair_basis(roles[0]==roles[1]);d=U.shape[1]
    self.two[roles]=(U@z[key][:d,:d].T@U.T)/self.N0
  self.two[1,0]=self.two[0,1].reshape(5,5,5,5).transpose(1,0,3,2).reshape(25,25)
  self.one={r:np.einsum('ikjk->ij',self.two[r,r].reshape(5,5,5,5)) for r in (0,1)}
  self.defects={str(r):float(abs(np.trace(v)-1)) for r,v in self.one.items()}
 @lru_cache(maxsize=None)
 def local(self,word):
  if not word:return 1.+0j
  bal=Counter()
  for slot,p,c in word:bal[slot]+=1 if c else -1
  if any(bal.values()):return 0j
  slots=sorted(bal)
  assert len(slots)<=2,'Higher density required; do not approximate it'
  inv=sum(word[i][0]>word[j][0] for i in range(len(word)) for j in range(i+1,len(word)))
  ops=[]
  for slot in slots:
   op=np.eye(16,dtype=complex)
   for owner,p,c in word:
    if owner==slot:op=op@(self.ann[p].T if c else self.ann[p])
   ops.append(self.W.conj().T@op@self.W)
  role=lambda slot:0 if slot<2 else 1
  op=ops[0] if len(ops)==1 else np.kron(*ops)
  rho=self.one[role(slots[0])] if len(slots)==1 else self.two[role(slots[0]),role(slots[1])]
  return (-1)**inv*np.einsum('ij,ji->',op,rho)
 def moment(self,word):
  inv=sum(word[i][0]>word[j][0] for i in range(len(word)) for j in range(i+1,len(word)))
  result=complex((-1)**inv)
  for c in range(3):
   result*=self.local(tuple((slot,p,cr) for owner,slot,p,cr in word if owner==c))
   if not result:return 0j
  return result
 def gram(self,left,right):
  total=0j
  for a,w in left:
   dagger=tuple((c,s,p,not cr) for c,s,p,cr in reversed(w))
   for b,v in right:total+=a.conjugate()*b*self.moment(dagger+v)
  return total

def root_terms(edge,orientation,outer,M,copy,spin0=False):
 # Frozen physical positive path creates TAIL and annihilates HEAD.
 ct=1 if edge['central_is_tail'] else outer
 ch=outer if edge['central_is_tail'] else 1
 st=edge['central_site'] if ct==1 else edge['outer_site']
 sh=edge['central_site'] if ch==1 else edge['outer_site']
 out=[]
 for i,j in itertools.product(range(4),repeat=2):
  if spin0 and (i//2 or j//2):continue
  z=M[i,j]
  if z==0:continue
  a=(ct,2*copy+i//2,2*st+i%2,True)
  b=(ch,2*copy+j//2,2*sh+j%2,False)
  out.append((complex(z), (a,b)) if orientation==1 else (complex(z.conjugate()), ((ch,b[1],b[2],True),(ct,a[1],a[2],False))))
 return out

def product(a,b):
 return [(x*y,w+v) for x,w in a for y,v in b]

def contact_terms(a,b):
 # Exact even-CAR commutator, not a substituted endpoint propagator.
 # [c†i cj,c†k cl]=delta_jk c†i cl-delta_li c†k cj.
 combined={}
 for z,(i,j) in a:
  for w,(h,l) in b:
   for coincident,left,right,sign in [(j[:3]==h[:3],i,l,1),(l[:3]==i[:3],h,j,-1)]:
    if coincident:
     word=(left,right)
     combined[word]=combined.get(word,0j)+sign*z*w
 return [(z,w) for w,z in combined.items() if z!=0]

def pair_quantities(k,e,f,oa,ob,matrices,spin0=False):
 Ga=Gb=Gab=Gba=overlap=contact=0j;pieces={}
 for cf,cg,mult,name in PATTERNS:
  a=root_terms(e,oa,0,matrices[e['axis'],e['central_site'],e['outer_site']],cf,spin0)
  b=root_terms(f,ob,2,matrices[f['axis'],f['central_site'],f['outer_site']],cg,spin0)
  ab=product(a,b);ba=product(b,a)
  aa=k.gram(ab,ab);bb=k.gram(ba,ba);cross=k.gram(ab,ba)
  Gab+=mult*aa;Gba+=mult*bb;overlap+=mult*cross
  pieces[name]={'multiplicity':mult,'Gab':float(aa.real),'Gba':float(bb.real),'order_overlap':enc(cross)}
 # Each single-root sector has exactly its own physical copy label, including spin.
 for cp,mult in [(0,1),(1,63)]:
  a=root_terms(e,oa,0,matrices[e['axis'],e['central_site'],e['outer_site']],cp,spin0)
  b=root_terms(f,ob,2,matrices[f['axis'],f['central_site'],f['outer_site']],cp,spin0)
  Ga+=mult*k.gram(a,a);Gb+=mult*k.gram(b,b)
  ct=contact_terms(a,b)
  contact+=mult*k.gram(ct,ct)
 assert max(abs(z.imag) for z in [Ga,Gb,Gab,Gba])<1e-4
 comm=float((Gab+Gba-2*overlap.real).real)
 return {'Ga':float(Ga.real),'Gb':float(Gb.real),'Gab':float(Gab.real),'Gba':float(Gba.real),
  'ordered_overlap':enc(overlap),'commutator_norm_square':comm,
  'explicit_CAR_contact_norm_square':float(contact.real),
  'coherent_orders_vs_contact_absolute_error':abs(comm-float(contact.real)),'patterns':pieces}

def direct_control(k,matrices):
 # Independent globally signed twelve-mode CAR; native W5, one specified spin.
 e={'axis':0,'central_site':1,'outer_site':0,'central_is_tail':True}
 f={'axis':1,'central_site':1,'outer_site':1,'central_is_tail':False}
 v=np.array([1,2j,-1,3,1j],complex);v/=np.linalg.norm(v)
 oldone,oldtwo=k.one,k.two
 k.one={0:np.outer(v,v.conj()),1:np.outer(v,v.conj())}
 k.two={(a,b):np.kron(k.one[a],k.one[b]) for a,b in itertools.product([0,1],repeat=2)}
 k.local.cache_clear()
 a=root_terms(e,1,0,matrices[0,1,0],0,True)
 b=root_terms(f,1,2,matrices[1,1,1],0,True)
 omega=np.kron(np.kron(k.W@v,k.W@v),k.W@v)
 def apply(terms,state):
  result=np.zeros_like(state)
  for z,word in terms:
   for ix in np.flatnonzero(state):
    mask=int(ix);sign=1
    for c,s,p,cr in reversed(word):
     # kron first cell has high bits; independent global basis order C,B,A.
     mode=4*(2-c)+p
     tr=car.apply_car(mask,mode,cr)
     if tr is None:sign=0;break
     mask,sg=tr;sign*=sg
    if sign:result[mask]+=z*sign*state[ix]
  return result
 ab=apply(a,apply(b,omega));ba=apply(b,apply(a,omega))
 actual=[np.vdot(ab,ab),np.vdot(ba,ba),np.vdot(ab,ba)]
 factor=[k.gram(product(a,b),product(a,b)),k.gram(product(b,a),product(b,a)),k.gram(product(a,b),product(b,a))]
 error=max(abs(x-y) for x,y in zip(actual,factor));assert error<1e-6
 contact=float(np.linalg.norm(ab-ba)**2);assert contact>1e-6
 k.one,k.two=oldone,oldtwo;k.local.cache_clear()
 return {'scope':'NEW one-labelled-spin native complex-source control, NOT actual64 source moments',
  'direct_global_CAR_vs_factorized_max_error':float(error),'wrong_central_order_contact_error_square':contact,
  'native_opposite_endpoint_zero_contact':'covered separately by actual computed commutator',
  'fullCAR_native_W5_no_highweight_fit':True}

def role_and_charge_controls(k,e,f,orientations,matrices,first):
 oldone,oldtwo=k.one,k.two
 k.one={r:oldone[0] for r in (0,1)}
 k.two={(r,s):oldtwo[0,0] for r,s in itertools.product((0,1),repeat=2)}
 k.local.cache_clear()
 wrong=pair_quantities(k,e,f,*orientations,matrices)
 k.one,k.two=oldone,oldtwo;k.local.cache_clear()
 # Full shared rho includes every one-slot square and every ordered pair.
 rho=np.diag([.5,.5,-.5,-.5])
 lifted=sum((k.ann[i].T@k.ann[j]*rho[i,j] for i,j in itertools.product(range(4),repeat=2)),np.zeros((16,16)))
 R=k.W.conj().T@lifted@k.W;R2=k.W.conj().T@lifted@lifted@k.W
 r2=2*np.trace(R2@k.one[0])+126*np.trace(R2@k.one[1])
 for a,b,m in [(0,0,2),(0,1,252),(1,0,252),(1,1,15750)]:
  r2+=m*np.trace(np.kron(R,R)@k.two[a,b])
 assert abs(r2.imag)<1e-8
 return {'wrong64active_Gab_float':wrong['Gab'],
  'actual1plus63_Gab_float':first['Gab'],
  'wrong64active_absolute_error_float':abs(wrong['Gab']-first['Gab']),
  'drop_auxiliary_Gab_float':first['patterns']['same_active']['Gab'],
  'complete_rho_all_second_moment_from_correlated_densities_float':float(r2.real),
  'sharedB_electric_duplicate_extra_source_price_float':64*float(r2.real),
  'literal_neutral_B_control':{'delta_site':[1,-1],'Q':0,'rho_prime_shift':'1','u':'-1',
   'actual_internal_E_delta':'0','wrong_u0_internal_HE_at_rho0':'64','correct_internal_HE_at_rho0':'0'},
  'literal_charged_B_control':{'delta_site':[2,0],'Q':2,'number':258,'rho_prime_shift':'1','u':'-1',
   'actual_external_HE_two_distinct_edges':'128','wrong_same_edge_import':'256'},
  'outside_ABC_disjoint_transverse_edge_action_norm_square_float':2*first['Ga'],
  'exterior_nonzero_proof':'Choose a far axis1 unit edge with both blocked endpoints outside A/B/C. Its two orientations create unique external-flow signatures, orthogonal to every other native first-H term and root flow. It acts on an untouched normalized product source, hence contribution2Ga, not a missing upper bound masquerading as a lower.'}

def complete_action(boundary,profiles,n,charges):
 # Rigorous finite ambient FIRST-H norm bound on the actual rooted input support.
 # Matter operators can send the state anywhere; their full Fock norms are paid.
 V=n**3;cells=V//2
 onsite=4*sum(Q(p['count'])*abs(Q(p['m_exact'])) for p in profiles)
 pathK={}
 for axis,length in itertools.product(range(3),(1,2)):
  pathK[axis,length]=sum(Q(p['count'])*Q(next(z['fullCAR_hop_norm_upper_exact'] for z in p['paths'] if z['axis']==axis and z['length']==length)) for p in boundary['roles'])
 # Every positive native path, including internal matching and outside A/B/C.
 hopping=V*sum(pathK.values())
 matter=V*onsite+hopping
 # All source cells carry ONE bounded internal tree E=-rho, |rho|<=128.
 # Distinct external unit edges add two orthogonal integer entries, not q2=256.
 electric=64*(cells*128**2+2)
 absolute=matter+electric
 exterior=hopping-sum(pathK[a,1] for a in charges)
 return {'scope':'EXACT finite-ambient full FIRST-H operator-norm majorant on rooted input E support; NOT variance or uniform linked bound',
  'N':n,'ambient_physical_sites':V,'ambient_blocked_cells':cells,
  'matter_onsite_upper_exact':str(V*onsite),'all_primitive_one_two_link_hopping_upper_exact':str(hopping),
  'full_E_HE_HC_combined_upper_exact':str(electric),'full_E_uses_orthogonal_transverse_and_longitudinal_parts_once':True,
  'complete_firstH_norm_upper_exact':str(absolute),'selected_two_edge_only_hopping_upper_exact':str(hopping-exterior),
  'outside_selected_edges_hopping_upper_exact':str(exterior),'all64_once_shared_roles':True,
  'PHP_squared':False,'sampled_tail':False,'electric_source_internal_support_bound':128,
  'action_density_need':'No third-slot extraction needed for this rigorous full-action norm upper. Computing sharper full V-action squared WOULD require third-slot density for generic exterior terms and is not claimed.'}

def run(epoch,n,geometry_only=False):
 start=time.monotonic();assert n>=8 and n%2==0
 pins={name:sha(ROOT/name) for name in PINS};assert pins==PINS
 edges,rows=inventory(n)
 if geometry_only:
  out={'edges':edges,'rows':rows,'N':n,'D_out_verified_sign':'-IntegerBox.divergence = row-col; Dinc=-1tail,+1head',
   'flow_law':'positive T creates tail, removes head => flow=-inc; affine u=Qcell/2-delta_n_site0-flow_internal',
   'neutral_incoming_u':sorted({r['affine_u'][1] for r in rows if r['cell_charge'][1]==0}),
   'charged_numbers':sorted({r['canonical_numbers'][1] for r in rows if r['cell_charge'][1]!=0})}
  (ROOT/f'three_cell_native_{epoch}_geometry.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('edges','rows')}));return
 boundary=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
 profiles=json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())['profiles']
 with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',allow_pickle=False) as z:phi=z['Phi_binary_section']
 had=np.array([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])/math.sqrt(2)
 frame=np.einsum('ia,mij,jb->mab',had,phi,had);gamma=frame[1]/math.sqrt(np.trace(frame[1]@frame[1]).real/4)
 S=np.column_stack([np.eye(4)[:,0],gamma@np.eye(4)[:,0],np.eye(4)[:,1],gamma@np.eye(4)[:,1]])
 P=np.diag([1,-1,1,-1]);matrices={};physical={}
 for axis in range(3):
  z=next(p for p in boundary['roles'][0]['paths'] if p['axis']==axis and p['length']==1)
  M=np.array([[complex(float(Q(a)),float(Q(b))) for a,b in row] for row in z['matrix_exact']])
  physical[str(axis)]=z['matrix_exact']
  for sa,sb in itertools.product(range(2),repeat=2):matrices[axis,sa,sb]=(-1)**(sa+sb)*P@(S.conj().T@M@S)@P
 k=Kernel();control=direct_control(k,matrices)
 old=json.loads((ROOT/'connected_root_exterior_graph_interface_corrected_20261011.json').read_bytes())['graph_interface']['finite_pair_candidate']
 t=float(old['t']);cache={};quantities=[]
 for ix,row in enumerate(rows):
  e,f=[edges[j] for j in row['edges']];oa,ob=row['orientations']
  key=(e['axis'],e['central_site'],e['outer_site'],e['central_is_tail'],f['axis'],f['central_site'],f['outer_site'],f['central_is_tail'],oa,ob)
  if key not in cache:cache[key]=pair_quantities(k,e,f,oa,ob,matrices)
  q=cache[key];a,b,g=q['Ga'],q['Gb'],q['Gab']
  Z=1+t*t*(a+b)+t**4*g;leg=(1+t*t*a)*(1+t*t*b)
  coeff=g-a*b
  quantities.append(row|q|{'t_diagnostic':t,'shared_F_t_t':Z,'wrong_leg_product_F':leg,
   'log_x2_y2_coefficient':coeff,'shared_normalized_covariance_t':t**4*coeff/Z,
   'leg_scaled_log_coefficient_t':t**4*coeff/leg,
   'F_exact_polynomial':'1+x^2 Ga+y^2 Gb+x^2 y^2 Gab; x,y real; charge sectors force every other cross term zero',
   'metric_x_y':'partial_x partial_y log F = 4xy (Gab-GaGb)/F^2; normalized FS mixed metric = xy(Gab-GaGb)/F^2',
   'F_reverse':'replace Gab by Gba; central order cannot be silently interchanged'})
  if ix%20==0:print(json.dumps({'class':ix,'total':len(rows)}),flush=True)
 first=quantities[0];e,f=[edges[j] for j in first['edges']]
 collapsed=pair_quantities(k,e,f,*first['orientations'],matrices,spin0=True)
 action=complete_action(boundary,profiles,n,[e['axis'],f['axis']])
 role_controls=role_and_charge_controls(k,e,f,first['orientations'],matrices,first)
 I=Q(274877906944,2204801412260314559041);eta=Q(1,65536);shift=Q(46955575073,1538631694057472)
 # Three independent CELL sources, not independent slots/copies: squared overlap (1-I)^3.
 vector_cap=Q(1,40000);assert 3*I<vector_cap**2
 K={axis:sum(Q(p['count'])*Q(next(z['fullCAR_hop_norm_upper_exact'] for z in p['paths'] if z['axis']==axis and z['length']==1)) for p in boundary['roles']) for axis in range(3)}
 O=(1+Q(t)*max(K.values()))**2
 ground_price=O*vector_cap
 # Exact outward positive-sum source prices, using the preserved producer model.
 moments=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['actual_selected_source_moments']
 gamma_reduction=Q(moments['positive_sum_roundoff_relative_upper_exact'])
 assert Q(moments['source_moments_float'][1])/(1-gamma_reduction)**2<1
 assert Q(moments['source_moments_float'][3])/(1-gamma_reduction)**2<2
 exterior_cells=n**3//2-3
 # Root sources factor across cells, NOT across physical slots.
 # Exterior electric fourth moment <=Cext*2+Cext(Cext-1)*1.
 eext_square=Q(64**2)*(2*exterior_cells+exterior_cells*(exterior_cells-1))
 eext_upper=Q(math.isqrt(eext_square.numerator//eext_square.denominator)+1)
 assert eext_upper**2>=eext_square
 # P commutes the literal matching internal E, which remains bounded128.
 # ||P rho_c Omega||<=||P|| ||rho_c Omega||<=O, and F>=1 by charges.
 eselected_upper=3*64*128*O
 sourceweighted_action=Q(action['matter_onsite_upper_exact'])+Q(action['all_primitive_one_two_link_hopping_upper_exact'])+eext_upper+eselected_upper+128
 action['outward_sourceweighted_complete_firstH_upper_exact']=str(sourceweighted_action)
 action['outward_exterior_electric_norm_upper_exact']=str(eext_upper)
 action['outward_selected_threecell_electric_norm_upper_exact']=str(eselected_upper)
 action['outward_source_moment_prices']={'rho2_upper_exact':'1','rho4_upper_exact':'2',
  'gamma_positive_reduction_exact':str(gamma_reduction),
  'model':'Conditional same standard correct-rounding binary64 source positive-reduction model; integer rho^4<=2^28 exact, retained gamma16d bounds product/sum/normalization.',
  'proof':'F>=1 exactly by outer-cell charge orthogonality. Outside3 cells remain product source; sum rho^2 has square moment C*m4+C(C-1)*m2^2. In3 selected cells literal E commutes each external root and |E|<=128; triangle gives3*64*128*||P||. Full HE/HC remains once shared.',
  'improves_absolute_full_action_majorant':sourceweighted_action<Q(action['complete_firstH_norm_upper_exact'])}
 # Entire-ground transfer still cannot establish the small kernel envelope.
 cap_joint=max(K.values())**2
 gram_ground_error_norm=cap_joint*vector_cap
 worst_gram_error=2*cap_joint*gram_ground_error_norm+gram_ground_error_norm**2
 action['outward_envelope_attempt']={'ordered_product_norm_upper_exact':str(cap_joint),
  'threecell_ground_product_vector_replacement_upper_exact':str(gram_ground_error_norm),
  'Gram_ground_replacement_upper_exact':str(worst_gram_error),
  't4_Gram_replacement_upper_exact':str(Q(t)**4*worst_gram_error),
  'decision':'This exact outward source-to-ground error cap is larger than the observed1e-4 kernel. It changes admission to FAIL_TO_CERTIFY, not physical failure. Density formation uncertainty is additionally unpriced, never replaced by this cap.'}
 transfer={'scope':'EXACT conditional algebraic transfer caps; actual extracted moments remain FLOAT',
  'single_squared_infidelity_upper_exact':str(I),'three_cell_squared_infidelity_upper_exact':str(1-(1-I)**3),
  'three_cell_phase_aligned_vector_distance_upper_exact':str(vector_cap),
  'proof':'d3^2=2[1-(1-I)^(3/2)]<=3I for 0<=I<=1; verify3I<(1/40000)^2. Do not use I as a vector bound.',
  'three_cell_residual_and_reference_shift_exact':str(3*(eta+shift)),
  'eta_exact_includes_formation':str(eta),'Rayleigh_ground_shift_exact':str(shift),
  'root_polynomial_norm_upper_exact':str(O),'rooted_unnormalized_vector_ground_replacement_upper_exact':str(ground_price),
  'normalization_rule':'F>=1 exactly from four orthogonal outer-cell charge sectors of the normalized source, so normalized threecell ground-source replacement <=2epsilon. This does NOT enclose actual4c6c moment formation.',
  'formation_prices':'No outward formation/normalization interval on4c6c densities; no true-ground numeric kernel admission.'}
 values=[q['leg_scaled_log_coefficient_t'] for q in quantities]
 out={'kind':'NEW_DISTINCT_NATIVE_THREECELL_ACTUAL64_COHERENT_FLOAT_KERNEL_COMPLETE_FINITE_ACTION_UPPER',
  'N':n,'edges':edges,'classes':quantities,'physical_native_matrices_exact':physical,'spin_frame_float':enc(S),
  'all_native_blocked_graph_gauss':all_native_gauss(n),
  'density_scope':'FLOAT archived normalized correlated one/two-slot extraction; no interval promotion',
  'density_trace_defects_float':k.defects,'source_normalization_divisor_float':k.N0,
  'allcopy_ordered_patterns':PATTERNS,
  'higher_density_proof':'Outer A and C have one hop each. Their fixed per-slot number2 forces bra/ket copy and outer spin equality. Central B then has at most four odd factors, hence at most two balanced slots. Same-copy cross-spin histories retained. All five copy patterns sum4096 ordered pairs.',
  'complete_action':action,'ground_transfer':transfer,
  'controls':control|role_controls|{'wrong_same_edge_external_selfterm':256,'actual_distinct_external_selfterm':128,
   'wrong_leg_product_max_error':max(abs(q['shared_F_t_t']-q['wrong_leg_product_F']) for q in quantities),
   'central_contact_max_commutator_square':max(q['commutator_norm_square'] for q in quantities),
   'actual_spin0_drop_Gab':collapsed['Gab'],'actual_allspin_Gab':first['Gab'],
   'wrong_exterior_deletion_missing_norm_majorant_exact':action['outside_selected_edges_hopping_upper_exact'],
   'blind_probe':'Opposite-endpoint orders commute by disjoint central physical modes; observed tiny FLOAT commutator is NONDISCRIMINATING against a general order bug.'},
  'envelope_decision':{'old_same_edge_log_kernel':old['second_graph_pair_kernel'],
   'actual_distinct_leg_scaled_log_range_float':[min(values),max(values)],
   'max_absolute_distinct_float':max(map(abs,values)),
   'improves_old_constant_at_tested_t_float':max(map(abs,values))<abs(old['second_graph_pair_kernel']),
   'not_same_functional':'Old same-edge graph has exchange folds and a symmetrized degree2 coefficient t^2/2. New ordered two-distinct-edge polynomial coefficient t^2; values are explicitly function-specific, not interchangeable envelopes.',
   'decision_scope':'Finite FLOAT source kernel observations only; outward density uncertainty and complete finite-action price do not prove N-uniform envelope or linked stability.'},
  'source_and_input_SHA':pins,'own_source_SHA':sha(Path(__file__)),
  'source_and_inputs_unchanged':all(sha(ROOT/name)==pin for name,pin in pins.items()),
  'elapsed_seconds':time.monotonic()-start,'natural_main_finished':True,
  'nonclaims':['physicalPhi','lattice_ground','allorder_linked_stability','phase','IRatom','pole','commoncone','originalSOURCE','law_adoption']}
 path=ROOT/f'three_cell_native_{epoch}_receipt.json'
 with path.open('x') as stream:json.dump(out,stream,indent=2);stream.write('\n')
 print(json.dumps({'receipt':path.name,'classes':len(rows),'envelope':out['envelope_decision'],'source_and_inputs_unchanged':out['source_and_inputs_unchanged']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epoch',required=True);p.add_argument('--N',type=int,default=8);p.add_argument('--geometry-only',action='store_true')
 args=p.parse_args();run(args.epoch,args.N,args.geometry_only)
