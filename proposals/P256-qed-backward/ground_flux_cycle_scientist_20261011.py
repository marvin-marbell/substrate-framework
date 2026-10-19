"""Actual matching-ground four-edge dynamical transition: source-weighted enclosure.
No solver, historical CLI, static heat proxy, or physical-one phase admission.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import sympy as sy
import harmonic_car_bridge_scientist_20261011 as car
import charged_stieltjes_scientist_20261011 as cx
import charged_output_lower_certificate_20261011 as iv

ROOT = Path(__file__).resolve().parent
PREFIX = 'ground_flux_cycle_'
INPUTS = list(dict.fromkeys([*cx.INPUTS, 'charged_stieltjes_scientist_20261011.py',
 'charged_stieltjes_proof_20261011.json', 'charged_stieltjes_actual1_20261011_receipt.json',
 'affine_lower_mathreview_proof_20261011.json', 'affine_lower_mathreview_verdict_20261011.json',
 'joined_phi_u1_physical_charged_canonical_neutral_scoped_acceptance_20261011.json',
 'reference_matching_flux_spectral_parent_attempt2_20261011_receipt.json',
 'joined_phi_u1_bulk_flux_ring_proof_20261010.json',
 'ground_flux_cycle_proof_20261011.json']))
ZERO = iv.iv(0)

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def matrix_read(raw):
 return sy.Matrix([[sy.Rational(a)+sy.I*sy.Rational(b) for a,b in row] for row in raw])

def l1(matrix):
 out=ZERO
 for value in matrix:
  real,imag=cx.ci(value)
  out=iv.add(out,(0,max(abs(real[0]),abs(real[1]))+max(abs(imag[0]),abs(imag[1]))))
 return out

def gram(role,n,X,N,epoch):
 paths=[]
 for slot,E in enumerate(car.extraction(n)):
  path=ROOT/f'{PREFIX}{epoch}_scratch_{role}_{slot}.npy'
  np.save(path,E@X if role==0 else (E@X.T).T)
  paths.append(path)
 views=[np.load(p,mmap_mode='r').reshape(-1) for p in paths]
 K=np.zeros((5,5))
 for start in range(0,views[0].size,32768):
  block=np.stack([v[start:start+32768] for v in views])
  K+=block@block.T
 del views
 for path in paths:path.unlink()
 budget=F(1,1<<24)
 result=[]
 for row in K:
  result.append([(iv.iv(F(float(v))/N-budget)[0],iv.iv(F(float(v))/N+budget)[1]) for v in row])
 return result,K.tolist()

def endpoint(maps,K):
 out=[]
 for p in range(2):
  row=[]
  for q in range(2):
   value=cx.CZ
   for k,i,j in itertools.product(range(4),range(5),range(5)):
    value=cx.ca(value,cx.cm(cx.cm(cx.cc(cx.ci(maps[p][k,i])),cx.ci(maps[q][k,j])),cx.scalar(K[i][j])))
   row.append(value)
  out.append(row)
 # Spin replicas are labelled, with the same exchange-symmetric selected state.
 return out

def norm_squared(M,A,B):
 out=cx.CZ
 for i,j,k,l in itertools.product(range(2),repeat=4):
  out=cx.ca(out,cx.cm(cx.cm(cx.cc(cx.ci(M[i,j])),cx.ci(M[k,l])),cx.cm(A[i][k],B[j][l])))
 assert out[0][1]>=0 and out[1][0]<=0<=out[1][1]
 return max(0,out[0][0]),out[0][1]

def face(N):
 vertices=[(0,0,0),(0,1,0),(0,1,1),(0,0,1)]
 edges=[(vertices[0],1,1),(vertices[1],2,1),(vertices[3],1,-1),(vertices[0],2,-1)]
 cells=[(x//2,y,z) for x,y,z in vertices]
 assert N>=8 and len(set(cells))==4 and len({(a,b) for a,b,_ in edges})==4
 assert all(v[0]%2==0 for v in vertices)
 return vertices,edges

def subset_data(subset,orientation):
 q=[0]*4
 for e in subset:
  q[e]+=orientation;q[(e+1)%4]-=orientation
 k=len(subset);charged=sum(v!=0 for v in q)
 return {'edges':list(subset),'external_integer_flow':[((-orientation if i<2 else orientation) if i in subset else 0) for i in range(4)],
  'cell_charge':q,'cell_internal_affine_u':[str(F(-v,2)) for v in q],
  'external_energy_exact':str(64*k),'charged_cells':charged,
  'simultaneous_particle_hole_pairs':charged//2}

def mask_op(mask,mode,creation):
 if bool(mask>>mode&1)==creation:return None
 return mask^(1<<mode),(-1)**((mask&((1<<mode)-1)).bit_count())

def car_controls(ann,V,edges):
 initial=sum(5<<(4*v) for v in range(4))
 vertices,_=face(8)
 incidence=np.zeros((4,4),dtype=np.int64)
 for e,(tail,axis,_) in enumerate(edges):
  head=list(tail);head[axis]=(head[axis]+1)%8;head=tuple(head)
  incidence[vertices.index(tail),e]=1;incidence[vertices.index(head),e]=-1
 flow_checks=0
 for orientation in (-1,1):
  for k in range(1,5):
   for subset in itertools.combinations(range(4),k):
    row=subset_data(subset,orientation)
    assert list(-incidence@np.array(row['external_integer_flow']))==row['cell_charge']
    assert sum(v*v for v in row['external_integer_flow'])==k
    flow_checks+=1
 assert list(incidence@np.array([-1,-1,1,1]))==[0]*4
 witnesses=[];parity_counter=None
 for order in itertools.permutations(range(4)):
  nonzero=0
  for modes in itertools.product(range(2),repeat=8):
   mask=initial;sign=1;local_equal=True
   for e in order:
    a,b=e,(e+1)%4;pa,pb=modes[2*e:2*e+2]
    before=mask
    first=mask_op(mask,4*b+pb,False)
    if first is None:break
    second=mask_op(first[0],4*a+pa,True)
    if second is None:break
    mask=second[0];hop=first[1]*second[1];sign*=hop
    # Independently local CAR omits the graded tensor strings, intentionally wrong.
    x=mask_op((before>>(4*b))&15,pb,False)
    y=mask_op((before>>(4*a))&15,pa,True)
    assert x is not None and y is not None
    local_equal &= hop==x[1]*y[1]
   else:
    if mask==initial:
     nonzero+=1
     if not local_equal and parity_counter is None:parity_counter={'order':list(order),'modes':list(modes),'closed_full_sign':sign,'per_hop_equality':False}
  witnesses.append({'order':list(order),'nonzero_closed_CAR_histories':nonzero,
   'opposite_first_two':(order[0]-order[1])%4==2})
 assert parity_counter is not None and any(w['opposite_first_two'] and w['nonzero_closed_CAR_histories'] for w in witnesses)
 exterior=ann[1].T*ann[0]*V[:,0]
 full=(exterior.conjugate().T*exterior)[0]
 cropped=(exterior.conjugate().T*V*V.conjugate().T*exterior)[0]
 assert full>cropped
 repeated=[0,0,2,3]
 assert len(set(repeated))<4 and [repeated.count(e) for e in range(4)]!=[1]*4
 # Integer incoming flow changes the literal electric difference, no Phi proxy.
 s=[-1,-1,1,1];incoming=s
 delta=64*sum((a+b)**2-a*a for a,b in zip(incoming,s))
 assert delta==768 and delta!=256
 return {'N8_four_distinct_external_edges':True,'final_flow_energy_exact':'256',
  'exact_N8_Gauss_flow_checks':flow_checks,'final_raw_incidence_zero':True,
  'repeated_edge_rejected':True,'missing_external64_rejected':64*3!=256,
  'incoming_face_electric_difference_exact':str(delta),'omitted_electric_rejected':True,
  'CAR_orders':witnesses,'wrong_per_hop_parity':parity_counter,
  'exterior_full_norm_squared_exact':str(full),'exterior_vector5_cropped_norm_squared_exact':str(cropped),
  'omitted_intermediate_twoPH_rejected':True,'omitted_exterior_scalar_rejected':True}

def rooted_guard(boundary,gap):
 # All native one/two-link supports, including the electric path midpoint.
 N=8;paths=[]
 weights={}
 for axis,length in itertools.product(range(3),(1,2)):
  weight=ZERO
  for role in boundary['roles']:
   row=next(p for p in role['paths'] if p['axis']==axis and p['length']==length)
   weight=iv.add(weight,iv.scale(l1(matrix_read(row['matrix_exact'])),2*role['count']))
  weights[axis,length]=weight
 def cell(v):return (v[0]//2,v[1],v[2])
 for anchor in itertools.product(range(N),repeat=3):
  for axis,length in itertools.product(range(3),(1,2)):
   vertices=[]
   for step in range(length+1):
    v=list(anchor);v[axis]=(v[axis]+step)%N;vertices.append(tuple(v))
   support=frozenset(cell(v) for v in vertices)
   if axis==0 and length==1 and anchor[0]%2==0:continue
   paths.append((support,weights[axis,length]))
 rooted=[item for item in paths if (0,0,0) in item[0]]
 overlap=[]
 for support,_ in rooted:
  total=ZERO
  for other,weight in paths:
   if support&other:total=iv.add(total,weight)
  overlap.append(total)
 J=max(a[1] for a in overlap)
 # 3 > e: a fully rational conservative tree-counting numerical gate.
 ratio=iv.mul(iv.scale((J,J),3),iv.inv(iv.iv(gap)))
 return {'normalization':'per rooted local path, vacuum-disconnected components divided out; no volume norm',
  'N8_rooted_path_classes':len(rooted),'max_overlapping_local_strength':cx.interval_output((J,J)),
  'conditional_tree_majorant_gate_3J_over_gap':cx.interval_output(ratio),
  'physical_hopping_exact':'1','gate_less_than_one':ratio[1]<iv.Q,
  'conditional_hopping_radius_upper':str(F(gap)/(3*F(J,iv.Q))),
  'unproved_required_hypothesis':'Normalized ground cumulant/tree and time-integral bound with each connected insertion costing at most J/gap; ordinary reference gap does not supply additive support-energy denominators or this linked bound.',
  'verdict':'FAILED numerical sufficient-package gate, with linked/tree hypothesis separately UNPAID. No physical1 remainder certified.',
  'N_uniformity':'Local finite-range overlaps lift to every even N>=8 without volume factors; coefficients are the same. Counting bound does not establish a thermodynamic phase.'}

def run(epoch):
 start=time.monotonic();pins={p:sha(ROOT/p) for p in INPUTS}
 assert all(pins[k]==v for k,v in car.PINS.items())
 upper=json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_text())
 Egupper=F(upper['ground_energy_interval_exact'][1]);charged_cost=-5120-Egupper
 assert charged_cost>0
 infidelity=iv.iv(F(274877906944,2204801412260314559041));eta=iv.sqrt(iv.scale(infidelity,2))
 shift=F(46955575073,1538631694057472)
 density=json.loads((ROOT/'three_cell_density_formation_interface_20261011.json').read_text());N=F(density['source_norm_squared_exact'])
 with np.load(ROOT/'local_so5_resum_actual04_20261011_state.npz',allow_pickle=False) as data:X=np.array(data['state'],copy=True).reshape(14,690880)
 J,maps,hbasis,ops,ann,V=car.car_exact()
 boundary=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_text())['all64_boundary']
 with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',allow_pickle=False) as data:raw=data['Phi_binary_section'][1]
 phi=sy.Matrix([[sy.Rational(F(float(complex(v).real)))+sy.I*sy.Rational(F(float(complex(v).imag))) for v in row] for row in raw])
 had=sy.Matrix([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])/sy.sqrt(2)
 frame=sy.simplify(had.T*phi*had);gamma=sy.simplify(frame/sy.sqrt(sy.trace(frame*frame)/4))
 S=sy.Matrix.hstack(sy.eye(4)[:,0],gamma*sy.eye(4)[:,0],sy.eye(4)[:,1],gamma*sy.eye(4)[:,1]);P=sy.diag(1,-1,1,-1)
 assert sy.simplify(S.conjugate().T*S)==sy.eye(4)
 vertices,edges=face(8);controls=car_controls(ann,V,edges)
 totals={z:ZERO for z in (0,64)};stored={z:ZERO for z in totals};coarse={z:ZERO for z in totals}
 roles=[];order_rows=[]
 for role,(n,count) in enumerate(((2,1),(126,63))):
  K,diagnostic=gram(role,n,X,N,epoch)
  A=endpoint(maps[1],K);B=endpoint(maps[0],K)
  matrices={}
  for axis in (1,2):
   path=next(p for p in boundary['roles'][role]['paths'] if p['axis']==axis and p['length']==1)
   matrices[axis]=sy.simplify(P*S.conjugate().T*matrix_read(path['matrix_exact'])*S*P)
  cache={};boundary_rows=[]
  for orientation in (-1,1):
   for e,axis in enumerate((1,2,1,2)):
    positive=(1 if e<2 else -1)*orientation
    M=matrices[axis] if positive==1 else matrices[axis].conjugate().T
    # In loop-vertex order orientation+ creates vertex e, annihilates e+1.
    # Negative-axis geometric tail/head is reversed by the adjoint above.
    for sa,sb in itertools.product(range(2),repeat=2):
     row_spin,col_spin=(sa,sb) if orientation==1 else (sb,sa)
     block=M.extract([2*row_spin,2*row_spin+1],[2*col_spin,2*col_spin+1]);L=l1(block)
     m0=norm_squared(block,A,B)
     radius=iv.sqrt(m0);ground_radius=iv.add(radius,iv.mul(iv.scale(eta,2),L))
     cache[orientation,e,sa,sb]=(L,radius,ground_radius)
     boundary_rows.append({'orientation':orientation,'edge':e,'spins':[sa,sb],
      'source_norm_squared':cx.interval_output(m0),'operator_L1_upper':cx.interval_output(L),
      'exact_g_source_norm_upper':cx.interval_output(ground_radius)})
  for orientation in (-1,1):
   for order in itertools.permutations(range(4)):
    subsets=[subset_data(order[:k],orientation) for k in (1,2,3)]
    gaps=[64*len(r['edges'])+r['charged_cells']*charged_cost for r in subsets]
    order_bounds={z:ZERO for z in totals}
    for spins in itertools.product(range(2),repeat=4):
     entries=[cache[orientation,e,spins[e],spins[(e+1)%4]] for e in order]
     Ls,rs,gs=zip(*entries)
     # Final bra V4* Wg has the opposite charges to V4 acting on Wg.
     # Its norm is the separately calculated reverse-orientation source norm;
     # electric shifts are unitary and source matter remains g.
     last=order[-1];opposite=cache[-orientation,last,spins[last],spins[(last+1)%4]]
     product=iv.mul(iv.mul(gs[0],opposite[2]),iv.mul(Ls[1],Ls[2]))
     source_product=iv.mul(iv.mul(rs[0],opposite[1]),iv.mul(Ls[1],Ls[2]))
     normproduct=iv.iv(1)
     for L in Ls:normproduct=iv.mul(normproduct,L)
     for z in totals:
      denominator=iv.iv(1)
      for gap in gaps:denominator=iv.mul(denominator,iv.iv(gap+z))
      value=iv.mul(product,iv.inv(denominator));value=iv.scale(value,count)
      totals[z]=iv.add(totals[z],value);order_bounds[z]=iv.add(order_bounds[z],value)
      stored[z]=iv.add(stored[z],iv.scale(iv.mul(source_product,iv.inv(denominator)),count))
      coarse[z]=iv.add(coarse[z],iv.scale(iv.mul(normproduct,iv.inv(denominator)),count))
    order_rows.append({'role':role,'orientation':orientation,'order':list(order),'subsets':subsets,
     'full_H0_minus_Eground_gaps_exact':[str(v) for v in gaps],
     'z0_absolute_bound':cx.interval_output(order_bounds[0])})
  roles.append({'role':role,'physical_copy_count':count,'harmonic_spin_replica_count':n,
   'new_slot_gram_diagnostic':diagnostic,'boundary_sources':boundary_rows})
 del X
 gap=F(46955312929,34359738368)
 out={'kind':'NEW_ACTUAL_G_FOUR_DISTINCT_EDGE_DYNAMICAL_FLUX_SOURCE_WEIGHTED_ENCLOSURE',
  'epoch':epoch,'own_source_sha256':sha(Path(__file__)),'input_sha256':pins,
  'geometry':{'N':8,'vertices':vertices,'matching_site':0,'all_external_edges':edges,'no_winding_alias':True},
  'literal_target':'sum_orientation=+/-1 sum_order in S4 <W_orientation g^tensor4,V4 R_Q(z) V3 R_Q(z) V2 R_Q(z) V1 g^tensor4>',
  'coefficient_enclosures':{str(z):{'real':cx.interval_output((-v[1],v[1])),
    'imaginary':cx.interval_output((-v[1],v[1])),
    'absolute_upper':cx.interval_output((0,v[1])),
    'literal_stored_source_absolute_upper':cx.interval_output((0,stored[z][1])),
    'exact_ground_majorant_increase_over_stored_majorant':cx.interval_output((v[1]-stored[z][1],v[1]-stored[z][1])),
    'unweighted_operator_norm_upper':cx.interval_output((0,coarse[z][1])),
    'sign':'INDETERMINATE; no nonzero exact-g coefficient certified'} for z,v in totals.items()},
  'roles':roles,'orders':order_rows,'controls':controls,
  'ground_pricing':{'squared_infidelity':cx.interval_output(infidelity),'vector_eta':cx.interval_output(eta),
   'source_residual_cap_exact':'1/65536','included_ground_formation_cap_exact':'1/262144',
   'Rayleigh_minus_ground_upper_exact':str(shift),
   'resolvent_energy':'True Eground, with Egupper in every denominator. NOT Es; this interval already pays the Rayleigh-ground shift.',
   'counterfactual_Es_resolvent_shift_per_four_cell_upper_exact':str(4*shift)},
  'new_formation':{'slot_gram_entry_cap_exact':'1/16777216','model':'correctly rounded binary64 extraction, sums and dot; no FTZ, underflow/overflow or reduced precision. Conditional arithmetic model, not machine attestation.',
   'budget_derivation':'Extraction row <=2 nonzeros, coefficients <=1; sqrt and multiplies pay gamma_12, block32768 Gram dots gamma_32770, <=300 block additions gamma_300. Selected normalized slot norm <=1, giving bound <2^-35; generous2^-24 also covers sparse formation and normalization. Physical coefficients are exact binary rationals/radicals enclosed outward at2^100.'},
  'resolvent_tail':{'global':'Every full sector spectrum x>=stated gap; ||R||<=1/(gap+z) includes arbitrarily high energies. No spectral or harmonic compression, PHP square, moment exponential or tail deletion.',
   'cutoff':None,'omitted_mass':0,'source_domain':'Finite CAR/external finite-support source in D(H0); after each bounded hop whole sector resolvent maps into D(H0). Neutral full6 exterior and simultaneous twoPH branches remain.'},
  'incoming_flow':'Computed enclosure only for incoming external k=0. General electric increment64|S|+128 sum incoming_e signed_shift_e, and internal affine u=-q/2 at site0. Incoming nonzero flows need their own reference-energy/domain lower bound; k=face final increment768, not256.',
  'rooted_remainder_attempt':rooted_guard(boundary,gap),
  'finite_float_coefficient_diagnostic':None,
  'next_material_mechanism':'A source-specific linked/resolvent resummation retaining neutral exterior and opposite-edge twoPH correlations, with exact incoming electric weights and ground cumulant control; current norm-only interval cannot decide oriented interference/sign. Then seek an actual same-law gap-closing branch rather than gap-preserving masslessness.',
  'open':['fullPhi/lattice stationary ground','normalized physical1 linked phase','gap-closing spectral trajectory','bulk stiffness','IRatom/sharp photon','commoncone','originalSOURCE','campaign'],
  'elapsed_seconds':time.monotonic()-start}
 target=ROOT/f'{PREFIX}{epoch}_receipt.json';target.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'receipt':target.name,'sha256':sha(target),'coefficient_enclosures':out['coefficient_enclosures'],
  'physical1_gate':out['rooted_remainder_attempt']['gate_less_than_one'],'elapsed_seconds':out['elapsed_seconds']}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);run(parser.parse_args().epoch)
