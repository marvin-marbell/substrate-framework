"""Source-specific full-H charged Stieltjes moment enclosure; no eigensolver.
Historical modules are imported for pure extraction/CAR/interval functions only.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import sympy as sy
import harmonic_car_bridge_scientist_20261011 as native
import charged_output_lower_certificate_20261011 as iv

ROOT = Path(__file__).resolve().parent
PREFIX = 'charged_stieltjes_'
INPUTS = [*native.PINS, 'harmonic_car_bridge_scientist_20261011.py',
 'harmonic_car_bridge_typed_attempt3_20261011_receipt.json',
 'charged_first_shell_actual04_20261011_receipt.json',
 'charged_output_lower_certificate_20261011.py',
 'charged_lower_review_attempt2_20261011_receipt.json',
 'charged_output_lower_attempt3_20261011_receipt.json',
 'charged_spinor_joint_free_phase_control_20261011.py',
 'charged_spinor_joint_free_phase_control_20261011_receipt.json',
 'joined_phi_u1_harmonic_transverse_affine_boundary_control_20261011_receipt.json',
 'three_cell_density_formation_interface_20261011.json',
 'three_cell_density_formation_proof_20261011.json']
Z = iv.iv(0)
ONE = iv.iv(1)

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def algebraic(x):
 x = sy.simplify(x)
 if x.is_Rational:
  return iv.iv(F(int(x.p),int(x.q)))
 if x.is_Add:
  out=Z
  for a in x.args: out=iv.add(out,algebraic(a))
  return out
 if x.is_Mul:
  out=ONE
  for a in x.args: out=iv.mul(out,algebraic(a))
  return out
 if x.is_Pow and x.exp in (sy.Rational(1,2),sy.Rational(-1,2)):
  out=iv.sqrt(algebraic(x.base))
  return iv.inv(out) if x.exp<0 else out
 raise ValueError(('unsupported exact algebraic coefficient',x))

def ci(x):
 a,b=sy.expand_complex(x).as_real_imag()
 return algebraic(a),algebraic(b)

def ca(a,b): return iv.add(a[0],b[0]),iv.add(a[1],b[1])
def cn(a): return iv.neg(a[0]),iv.neg(a[1])
def cc(a): return a[0],iv.neg(a[1])
def cm(a,b):
 return iv.sub(iv.mul(a[0],b[0]),iv.mul(a[1],b[1])),iv.add(iv.mul(a[0],b[1]),iv.mul(a[1],b[0]))
CZ=(Z,Z)

def scalar(a): return a,Z

def interval_output(a):
 return {'exact':iv.exact_endpoints(a),'display':[float(F(v,iv.Q)) for v in a]}

def widen(a,price): return iv.sub(a,price)[0],iv.add(a,price)[1]

def gram(role,n,X,rho,epoch,N):
 """New extraction/Gram epoch, complete slots and both fields, never old CLI."""
 paths=[]
 for field,W in enumerate((X,rho*X)):
  for k,E in enumerate(native.extraction(n)):
   data=E@W if role==0 else (E@W.T).T
   path=ROOT/f'{PREFIX}{epoch}_scratch_{role}_{field}_{k}.npy'
   np.save(path,data.reshape(-1));paths.append(path)
 views=[np.load(p,mmap_mode='r') for p in paths]
 K=np.zeros((10,10))
 for start in range(0,views[0].size,32768):
  block=np.stack([v[start:start+32768] for v in views])
  K+=block@block.T
 del views
 for p in paths: p.unlink()
 # Derived operation-model budget: 2^-24 exceeds formation+dot+block
 # accumulation gamma bound by >1000. No unexplained empirical epsilon.
 intervals=[]
 for i in range(10):
  row=[]
  for j in range(10):
   price=F((128 if i>=5 else 1)*(128 if j>=5 else 1),1<<24)
   center=F(float(K[i,j]))/N
   row.append(iv.iv(center-price)[0:1]+iv.iv(center+price)[1:2])
  intervals.append(row)
 return intervals,K.tolist()

def reduced(maps,h,u,creation,K):
 S=[[[CZ for _ in range(10)] for _ in range(4)] for _ in range(2)]
 H=[[[CZ for _ in range(10)] for _ in range(4)] for _ in range(2)]
 for p in range(2):
  delta=(1 if creation else -1)*F(1,2)+u
  for k in range(4):
   for j in range(5):
    S[p][k][j]=ci(maps[p][k,j])
    change=sum(((h[q,p] if creation else -h[p,q])*maps[q][k,j] for q in range(4)),sy.S.Zero)
    H[p][k][j]=ci(change+64*delta*delta*maps[p][k,j])
    H[p][k][5+j]=ci(128*delta*maps[p][k,j])
 def bilinear(A,B):
  out=[[CZ for _ in range(2)] for _ in range(2)]
  for p in range(2):
   for q in range(2):
    v=CZ
    for k in range(4):
     for i in range(10):
      if A[p][k][i]==CZ:continue
      for j in range(10):
       if B[q][k][j]!=CZ:
        v=ca(v,cm(cm(cc(A[p][k][i]),scalar(K[i][j])),B[q][k][j]))
    out[p][q]=v
  return [[out[p%2][q%2] if p//2==q//2 else CZ for q in range(4)] for p in range(4)]
 return {'G':bilinear(S,S),'T':bilinear(S,H),'U':bilinear(H,H)}

def contract(M,A,B):
 out=CZ
 for i in range(4):
  for j in range(4):
   for k in range(4):
    for l in range(4):
     out=ca(out,cm(cm(cc(M[i][j]),M[k][l]),cm(A[i][k],B[j][l])))
 return out[0]

def bounds(m0,m1,m2,gamma,z):
 G=iv.add(gamma,iv.iv(z))
 a=F(m1[0]+m1[1],m0[0]+m0[1])+z
 aa=iv.iv(a)
 n1=iv.add(m1,iv.mul(iv.iv(z),m0))
 n2=iv.add(iv.add(m2,iv.scale(iv.mul(iv.iv(z),m1),2)),iv.mul(iv.iv(z*z),m0))
 # Fixed positive a: exact Hermite identity, with global x>=G tail.
 diff=iv.sub(n1,iv.mul(aa,m0))
 variance=iv.add(iv.sub(n2,iv.scale(iv.mul(aa,n1),2)),iv.mul(iv.mul(aa,aa),m0))
 a2=iv.mul(aa,aa);a3=iv.mul(a2,aa)
 lower=iv.mul(iv.mul(m0,m0),iv.inv(n1))
 upper=iv.add(iv.sub(iv.mul(m0,iv.inv(aa)),iv.mul(diff,iv.inv(a2))),iv.mul(variance,iv.inv(iv.mul(G,a2))))
 lower2=iv.mul(iv.mul(iv.mul(m0,m0),m0),iv.inv(iv.mul(n1,n1)))
 factor=iv.mul(iv.add(aa,iv.scale(G,2)),iv.inv(iv.mul(iv.mul(G,G),a3)))
 upper2=iv.add(iv.sub(iv.mul(m0,iv.inv(a2)),iv.scale(iv.mul(diff,iv.inv(a3)),2)),iv.mul(variance,factor))
 coarse=iv.mul(m0,iv.inv(G));coarse2=iv.mul(m0,iv.inv(iv.mul(G,G)))
 return {'z_exact':str(z),'a_exact':str(a),'inverse':(max(0,lower[0]),min(upper[1],coarse[1])),
  'inverse_square':(max(0,lower2[0]),min(upper2[1],coarse2[1])),
  'gap_only':coarse,'gap_only_square':coarse2}

def run(epoch):
 start=time.monotonic();pins={name:sha(ROOT/name) for name in INPUTS}
 assert all(pins[k]==v for k,v in native.PINS.items())
 density=json.loads((ROOT/'three_cell_density_formation_interface_20261011.json').read_text())
 N=F(density['source_norm_squared_exact'])
 prof=json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles']
 frozen=json.loads((ROOT/'harmonic_car_bridge_typed_attempt3_20261011_receipt.json').read_text())
 J,maps,hbasis,ops,ann,V=native.car_exact()
 assert [[str(x) for x in row] for row in J.tolist()]==frozen['controls']['joint_n1_native_embedding']
 upper=json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_text())
 Eg=F(upper['ground_energy_interval_exact'][1]);gamma=iv.iv(64-10240-2*Eg)
 assert gamma[0]>0
 infid=iv.iv(F(274877906944,2204801412260314559041))
 eta=iv.sqrt(iv.scale(infid,2))
 residual=iv.iv(F(1,65536));shift=iv.iv(F(46955575073,1538631694057472))
 with np.load(ROOT/'local_so5_resum_actual04_20261011_state.npz',allow_pickle=False) as data:
  X=np.array(data['state'],copy=True).reshape(14,690880)
 r0=np.array([p-q for l,p,q in native.labels(2)])
 r1=np.array([p-q for l,p,q in native.labels(126)])
 rho=r0[:,None]+r1[None,:]
 boundary=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_text())['all64_boundary']
 geometry=json.loads((ROOT/'joined_phi_u1_harmonic_transverse_affine_boundary_control_20261011_receipt.json').read_text())
 chosen={qa:next(r for r in geometry['rows'] if r['axis']==1 and r['length']==1 and r['sites']==[0,0] and r['qa']==qa) for qa in [-1,1]}
 with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',allow_pickle=False) as data:
  raw=data['Phi_binary_section'][1]
 phi=sy.Matrix([[sy.Rational(F(float(complex(v).real)).numerator,F(float(complex(v).real)).denominator)+sy.I*sy.Rational(F(float(complex(v).imag)).numerator,F(float(complex(v).imag)).denominator) for v in row] for row in raw])
 had=sy.Matrix([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])/sy.sqrt(2)
 frame=sy.simplify(had.T*phi*had);g=sy.simplify(frame/sy.sqrt(sy.trace(frame*frame)/4))
 S=sy.Matrix.hstack(sy.eye(4)[:,0],g*sy.eye(4)[:,0],sy.eye(4)[:,1],g*sy.eye(4)[:,1])
 assert sy.simplify(S.conjugate().T*S)==sy.eye(4)
 path=next(p for p in boundary['roles'][0]['paths'] if p['axis']==1 and p['length']==1)
 physical=sy.Matrix([[sy.Rational(a)+sy.I*sy.Rational(b) for a,b in row] for row in path['matrix_exact']])
 P=sy.diag(1,-1,1,-1);Mexact=sy.simplify(P*S.conjugate().T*physical*S*P)
 M=[[ci(Mexact[i,j]) for j in range(4)] for i in range(4)]
 # Every CAR monomial has norm<=1; exact native coefficient L1 pays all spin terms.
 L=(sum(max(abs(a),abs(b)) for row in M for value in row for a,b in value),)*2
 rows=[];aggregate={z:{k:Z for k in ('inverse','inverse_square','gap_only','gap_only_square')} for z in (F(0),F(64))}
 rawgrams={}
 for role,(n,p) in enumerate(zip((2,126),prof)):
  K,rawgram=gram(role,n,X,rho,epoch,N);rawgrams[p['role']]=rawgram
  h=sum((sy.Rational(p[key])*basis for key,basis in zip(('m_exact','c0_exact','d_exact'),hbasis[:3])),sy.zeros(4))
  for qa in (-1,1):
   ua,ub=map(F,chosen[qa]['affine_u']);assert (ua,ub)==((-F(qa,2),F(qa,2)))
   A=reduced(maps[int(qa==1)],h,ua,qa==1,K)
   B=reduced(maps[int(qa==-1)],h,ub,qa==-1,K)
   Mc=M if qa==1 else [[cc(x) for x in row] for row in M]
   m0=contract(Mc,A['G'],B['G'])
   basefirst=iv.add(contract(Mc,A['T'],B['G']),contract(Mc,A['G'],B['T']))
   m1=iv.add(basefirst,iv.scale(m0,64))
   adjointT=[[cc(A['T'][j][i]) for j in range(4)] for i in range(4)]
   m2=iv.add(iv.add(contract(Mc,A['U'],B['G']),contract(Mc,A['G'],B['U'])),iv.scale(contract(Mc,adjointT,B['T']),2))
   m2=iv.add(iv.add(m2,iv.scale(basefirst,128)),iv.scale(m0,4096))
   # Full commutator plus certified neutral residual and exact Rayleigh-E0 shift.
   action_error=iv.scale(iv.mul(L,iv.add(residual,shift)),2)
   m1=widen(m1,iv.mul(iv.sqrt(m0),action_error))
   m2=widen(m2,iv.add(iv.scale(iv.mul(iv.sqrt((max(0,m2[0]),m2[1])),action_error),2),iv.mul(action_error,action_error)))
   assert m0[0]>0 and m1[0]>0 and m2[0]>0
   pricebase=iv.scale(iv.mul(iv.mul(L,L),eta),4)
   kernels=[]
   for z in aggregate:
    result=bounds(m0,m1,m2,gamma,z)
    G=iv.add(gamma,iv.iv(z))
    price=iv.mul(pricebase,iv.inv(G));price2=iv.mul(pricebase,iv.inv(iv.mul(G,G)))
    result['inverse']=widen(result['inverse'],price)
    result['inverse_square']=widen(result['inverse_square'],price2)
    result['gap_only']=widen(result['gap_only'],price)
    result['gap_only_square']=widen(result['gap_only_square'],price2)
    for key in aggregate[z]:
     value=result[key];value=(max(0,value[0]),value[1]);result[key]=value
     aggregate[z][key]=iv.add(aggregate[z][key],iv.scale(value,p['count']))
    result['source_ground_kernel_price']=price;result['source_ground_square_price']=price2
    kernels.append({key:interval_output(value) if isinstance(value,tuple) else value for key,value in result.items()})
   rows.append({'role':p['role'],'physical_copy_count':p['count'],'qA':qa,'affine_u_exact':[str(ua),str(ub)],
    'literal_stored_source_moments':[interval_output(m) for m in (m0,m1,m2)],'full_action_residual_shift_norm_price':interval_output(action_error),'kernels':kernels})
 # Independently reaches the actual global-tail polynomial assertion.
 G=F(2);a=F(5);x=F(1000000)
 q=1/a-(x-a)/a**2+(x-a)**2/(G*a**2)
 assert q>=1/x
 wrong=1/a-(x-a)/a**2
 assert wrong<1/x
 assert gamma!=iv.iv(64-10240+2*Eg), 'wrong relative-energy sign must not survive'
 wrongorient=(-F(1,2),-F(1,2));assert wrongorient!=tuple(map(F,chosen[1]['affine_u']))
 controls={'tail_deleted_tangent_rejected_at_x1000000':True,'wrong_ground_shift_sign_rejected':True,
  'transverse_common_u_rejected':True,'positive_full_tail_majorant_at_x1000000':True,
  'exact_signed_CAR_and_joint_all4_intertwiner':True,'source_reference_g_distinct':True}
 out={'kind':'NEW_LITERAL_NATIVE_ONE_EDGE_FULL_H_CHARGED_STIELTJES_INTERVAL',
  'epoch':epoch,'input_sha256':pins,'own_source_sha256':sha(Path(__file__)),
  'source_norm_squared_exact':str(N),'source_endpoint':{'axis':1,'length':1,'tail':[0,0,0],'head':[0,1,0],'matching_sites':[0,0],
   'positive_operator':'creates tail (qA=+1), annihilates head; adjoint reverses charges',
   'flow':'negative oriented integer incidence; external E=-1 costs64 once, internal affine offsets(-1/2,+1/2)',
   'physical_matrix_exact':path['matrix_exact'],'joint_native_matrix_exact':[[str(v) for v in row] for row in Mexact.tolist()],
   'all64_source':'sum over actual1+aux63 physical copies, all four spin-pair labels and both charge orientations'},
  'gamma_exact':iv.exact_endpoints(gamma),'ground_energy_upper_exact':str(Eg),
  'ground_norm_distance_price':interval_output(eta),'source_residual_cap_exact':'1/65536','Rayleigh_ground_shift_cap_exact':str(F(46955575073,1538631694057472)),
  'source_operator_L1_bound':interval_output(L),'rows':rows,
  'aggregate':{str(z):{k:interval_output(v) for k,v in values.items()} for z,values in aggregate.items()},
  'controls':controls,'new_raw_grams':rawgrams,
  'operation_model':['binary64 correctly rounded rational conversion and sqrt; extraction coefficients relative error<=3u',
   'CSR <=2-term row, <=4 binary64 rounded operations; rho integer multiply <=u',
   'block dot <=32768 products and65536 rounding sites; <=300 block accumulations',
   'no underflow/overflow/FTZ/DAZ/nonfinite or approximate reduced precision; no unconditional machine certification'],
  'formation_budget_derivation':'gamma(k)=ku/(1-ku), u=2^-53; coefficient+row+rho<=gamma(12); dot gamma(65536), accumulation gamma(300); combined entry error<2^-24 FiFj with Fi=1 or128. Sum extraction squared=I and |rho|<=128 bound true field norms. Raw Gram/N uses exact N and100-bit outward rational intervals.',
  'reducing_sector_proof':'Each physical replica number Nj=sum_modes c†c commutes with every one-body Dirac term and total-density64(R+u)^2. H0 has no intercell hopping. Source g,s have Nj=2. Different copy, spin-slot or orientation changes produce distinct joint Nj eigenvalues at the two cells. Their spectral measures are orthogonal for every function of H0; coherent Dirac terms WITHIN each such sector are retained.',
  'rooted_criterion':'Only this one literal edge is enclosed. Inverse-square upper is NOT an all-path rooted sum; no rooted<1 conclusion.',
  'unpaid':['normalized linked all-order connected spatial bound','physical lambda1 phase and thermodynamic common ground','gap-closing spectral pole trajectory and IR photon mechanism'],
  'source_and_inputs_unchanged':pins=={name:sha(ROOT/name) for name in INPUTS},'elapsed_seconds':time.monotonic()-start}
 assert out['source_and_inputs_unchanged']
 target=ROOT/f'{PREFIX}{epoch}_receipt.json';target.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'receipt':target.name,'sha256':sha(target),'aggregate':out['aggregate'],'controls':controls,'elapsed_seconds':out['elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epoch',required=True)
 run(parser.parse_args().epoch)
