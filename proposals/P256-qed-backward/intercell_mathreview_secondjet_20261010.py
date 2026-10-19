"""Independent complete-curvature and ordered-cell second-jet controls."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
import spatial_intercell_source as source
import spatial_intercell_complete_action_parent as parent
ROOT=Path(__file__).resolve().parent

def sha(raw):return hashlib.sha256(raw).hexdigest()
def jets(A,E,F):
 d=len(A);block=np.zeros((3*d,3*d),complex)
 for k in range(3):block[k*d:(k+1)*d,k*d:(k+1)*d]=A
 block[:d,d:2*d]=E;block[d:2*d,2*d:]=E;block[:d,2*d:]=F/2
 answer=expm(block);return answer[:d,:d],answer[:d,d:2*d],2*answer[:d,2*d:]

def main():
 names=['spatial_intercell_source.py','spatial_intercell_complete_action_parent.py','spatial_intercell_actual_attempt2_20261010_receipt.json']
 bound={n:sha((ROOT/n).read_bytes()) for n in names}
 assert bound[names[0]]=='2c323b844cfafb27692b3e7e5f107272477c4448c92481b65b932768b7886713'
 assert bound[names[1]]=='fc822f2d15b27f4fe78c34e7c6bda643db864ce5d2e15b4de4ad712f8ddbbf6a'
 assert bound[names[2]]=='a2f6dab92f2974292bb2c25c3b0968c80d35936b3d7fcc17391763c00526da17'
 # Clifford frame comes from the already saved mean, not any root/mean producer.
 from fractions import Fraction as F
 receipt=json.loads((ROOT/names[2]).read_bytes())['uniform_bound']
 vals=[F(receipt['exact_g0']),F(receipt['exact_gs'])]
 config_path=ROOT/'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json'
 config=json.loads(config_path.read_bytes())['input_descriptor']['original_micro_config_UNCHANGED']
 mean={'microscopic_config':config,'centre_enclosure':{name+'_'+edge:{'numerator':str(v.numerator),'denominator':str(v.denominator)} for name,v in zip(['g0','gs'],vals) for edge in ['lower','upper']}}
 with threadpool_limits(limits=1):
  native=source.seed.field.full.supplier.quantum.build_native(mean,4)
  gam=native['frame']/np.sqrt(np.trace(native['frame']@native['frame'],axis1=1,axis2=2).real/4)[:,None,None]
  theta=np.array([.37,-.29,.81,-.14]);gates=[];first=[];second=[]
  for k,(x,y,axis) in enumerate(((0,1,1),(1,3,2),(2,3,1),(0,2,2))):
   a=slice(4*x,4*x+4);b=slice(4*y,4*y+4)
   S=np.zeros((16,16),complex);Sp=np.zeros_like(S);v=np.exp(1j*theta[k])*1j*gam[0]@gam[axis]
   S[a,b]=v;S[b,a]=-v.conj().T;Sp[a,b]=1j*v;Sp[b,a]=1j*v.conj().T
   g,ge,gee=jets(-source.seed.T*S,-source.seed.T*Sp,source.seed.T*S)
   gates.append(g);first.append(ge);second.append(gee)
  U=np.eye(16,dtype=complex);du=[];ddu=[]
  for g in gates:U=g@U
  for edge in range(4):
   one=np.eye(16,dtype=complex);two=one.copy()
   for k in range(4):one=(first[k] if k==edge else gates[k])@one;two=(second[k] if k==edge else gates[k])@two
   du.append(one);ddu.append(two)
  _,seed_u,seed_du=source.seed.cell(theta,gam);parent_second=parent.cell_second(theta,gam)
  defects={'U':float(np.max(abs(U-seed_u))),'first':float(np.max(abs(np.array(du)-seed_du))),'second':float(np.max(abs(np.array(ddu)-parent_second)))}
  assert max(defects.values())<1e-13
  wrong_second=float(np.max(abs(np.array(ddu)+parent_second)));assert wrong_second>.01
  p0=np.kron(np.eye(4),(np.eye(4)-gam[0])/2)
  psecond=[uu@p0@U.conj().T+2*u@p0@u.conj().T+U@p0@uu.conj().T for u,uu in zip(du,ddu)]
  wrong_cross=[uu@p0@U.conj().T+U@p0@uu.conj().T for uu in ddu]
  wrong_cross_defect=float(np.max(abs(np.array(psecond)-np.array(wrong_cross))));assert wrong_cross_defect>1e-5
  # Rank-one P exactly saturates ||[S,[S,P]]||1 <=4||S||^2 rank(P).
  P=np.diag([1.,0.]);S=np.array([[0.,-1.],[1.,0.]]);comm=S@P-P@S;double=S@comm-comm@S
  double_trace=float(np.linalg.svd(double,compute_uv=False).sum());assert double_trace==4
  tau=.01;Pt=expm(-tau*S)@P@expm(tau*S);h=np.diag([-1.,1.]);matter_remainder=float(np.trace(h@(Pt-P)))
  assert tau*tau<matter_remainder<=2*tau*tau
 result={'kind':'NEW_ORDERED_CELL_SECOND_JET_AND_RANK_CURVATURE_ADVERSE_CONTROLS','source_input_hashes':{**bound,config_path.name:sha(config_path.read_bytes()),Path(__file__).name:sha(Path(__file__).read_bytes())},'cell_nonzero_phase_second_jet_defects':defects,'wrong_seed_second_sign_defect':wrong_second,'wrong_missing_projector_second_cross_defect':wrong_cross_defect,'rank1_double_commutator_trace_norm':double_trace,'correct_rank1_bound':4,'wrong_rank1_bound':2,'matter_remainder_at_tau001':matter_remainder,'correct_rank1_matter_Taylor_bound':2*tau*tau,'wrong_half_matter_Taylor_bound':tau*tau,'runtime_python':sys.executable,'old_producers_run':False}
 path=ROOT/'intercell_mathreview_secondjet_20261010.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'capsule':str(path),'sha256':sha(path.read_bytes()),'second_jet_defects':defects}))

if __name__=='__main__':main()
