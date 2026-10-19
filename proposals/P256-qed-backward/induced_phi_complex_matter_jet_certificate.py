"""Actual complex occupied/PH90 jets; conditional root, never source admission.

The receipt contains the proof, outward constants and adverse actual-N4 controls.
No root solver is called. Frozen helper files are imported read-only.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import mpmath as mp
import numpy as np
from scipy.linalg import expm, sqrtm
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import induced_phi_nonuniform_fermionic_source_unitary as source
import induced_phi_gaussian_action_certificate as gaussian

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).read_bytes()
R = F(1, 1 << 20)
TUBE = F(1, 1 << 16)
POLAR = F(1, 1 << 12)
sys.set_int_max_str_digits(0)
PROOF = {
 'conditional_inputs': 'Archived exact positive Clifford mean box, exact normal_geometry Q0 recipe, microscopic binary coefficients, static sea and q0 slow lower remain hypotheses. No root existence, whole source/action/domain, ground, pole or cone adoption.',
 'polar': 'O is the actual real Cartesian orbit matrix continued LINEARLY over complex coordinates. L=O^T O, P=I-O L^-1 O^T, N=P Q0, Q=N(N^T N)^-1/2. All transposes are algebraic; no conjugation of coordinates. Root ||O||<2, sigma_min(O)^2>2/5 from h>1/20,w<1/3. ||delta O||<=||s||: actual slow injection /8 and commutator Casimir sqrt15/4<1. At p=1/4096, delta L<=4p+p²=b. ||P-P0||<=b/(2/5)+4b/((2/5)(2/5-b))=eta. Both P,P0 are complex symmetric idempotents, so N^T N=Q0^T P Q0; ||N^T N-I||<=eta. Binomial series gives ||Q||<=(1+eta)/sqrt(1-eta)<9/8. This independently proves actual polar holomorphy and normal frame; not occupied radial Kato.',
 'nonlinear_h': 'Actual Phi_x=Phi0+Q0 s/8+Q(s) f_x; Fourier cosine/sine transform is real orthogonal, f zero-mean, links exp(i T.C_e), 64 sites/192 edges/one AND two links/AP seams. In the complex ball ||(s,f,C)||_2<=t, site Cartesian field displacement <=(1/8+9/8)t<2t, generator opnorm<=t/sqrt2<t. Both orientations continue by inverse ordered links and conjugating FIXED microscopic coefficients, never complex coordinates. Each mean hop has norm Hpath; each endpoint adds <=2|c|1 t. Ordered path and inverse path norms <=exp(2t)<=1/(1-2t). Schur row gives delta h<=2|onsite|t+2 sum[(Hpath+2|c|1 t)/(1-2t)-Hpath]. This complex bound includes both endpoints, all5760 directions, support and3R.',
 'contour': 'Let d=mean gap/2, H=mean row norm. Rectangle with real sides -H-d/2,-d/2 and imaginary sides +/-d/2 encloses exactly the negative sheet. Mean resolvent<=2/d, length=2H+2d<=4(H+d). delta h<d/4 implies Neumann resolvent<=4/d on this SAME fixed contour throughout complex tube. P=(2pi i)^-1 integral (z-h)^-1 dz is holomorphic and rank128. ||P||<=4(H+d)/d using pi>3. Complex h need not be normal or Hermitian; no Weyl eigenvalue inference is used for complex h.',
 'projector_jets': 'At any ||r||<=rho=3R, margin a=TUBE-rho. Multivariate discs with n radii a/n yield ||D^n h||<=n^n(H+delta)/a^n. Contour product rule: P1<=L Rz² H1/(2pi); P2<=L(2Rz³ H1²+Rz²H2)/(2pi); P3<=L(6Rz^4 H1³+6Rz³H1 H2+Rz²H3)/(2pi). L/(2pi)<=H+d. These are mixed multilinear operator norms. Tensor conversion costs sqrt(number of coordinates)^n, explicitly retained.',
 'matter_hessian': 'E=Tr(h P). Differentiating the Riesz integral gives E1=Tr(P h1) and E2=Tr(P h2)+Tr(P1 h1). At the real mean, P1 in the occupied/unoccupied basis has entries h1/(eocc-eunocc). Thus E2=contact minus BOTH +/-q particle-hole quotients of build_species/response/contact_q. Fourier 1/8, occupiedrank128 and PH90 are unchanged. For slow coefficients K(s), use separate probe u: Phi=Phi0+Q0 s/8+Q(s)uPhi, U=exp(iT.uC); K(s)=D_u² E(s,0), NOT the Hessian of the tied nonlinear chart. Derivatives of Q(s) are consequently retained, but no spurious chart gradient-curvature term is added. On ||s||+||u||<=TUBE the same contour bound holds. |E|<=256(H+delta)Pmax. Cauchy gives K_j<= (j+2)^(j+2) Emax/a^(j+2), j=0..3, for arbitrary normalized probe/slow directions. For the tied nonlinear source chart the same scalar Cauchy jets hold separately. The uniform case has all64q blocks; nonuniform case is the full5760 matrix, never falsely block diagonal.',
 'scale': 'The signed h includes negative heavy sign. Negative physical -Lambda h occupies the upper ORIGINAL h sheet. Substitution z=Lambda zeta cancels Lambda exactly in P and all P derivatives; [d(I-P),I-P]=[dP,P]. Ephys=|scale| E_signed; vertices/contact scale by |scale|, response numerator scales squared and physical gap once, so static K_j scales by |scale| and dynamical frequency is omega/|scale|. Heavy neutral charge does not remove its Phi/colour sea stiffness.',
 'tensor': 'K_j is a multilinear slow-to-fast-bilinear norm, hence fast operator bound. Each90 block HS<=sqrt90 K_j<10 K_j; all64 block directsum HS<=76 K_j. Slow tensor HS<=76*7^j K_j (sqrt45<7). Fully nonuniform ambient derivative tensor uses 76^j and full matrix HS<=76 K_j; full Hessian has5760, never a single-q 1/8 bound.',
 'budget_trace': 'Frozen gauss full_field_observations lines273-286 use v1=sum load*90(vPhi+vC),v2=sum load*90(PC+CC),riesz3=128(8v2+24v1v2/gap+48v1^3/gap²),C=2^24(1+riesz3+bound)^3(1+gap^-1)^4(1+hmin^-1)^4. That affine price is NOT promoted. Replacement C below explicitly dominates K0..K3, contour jets, tensor factors, sea, inverse gap/Horbit/slow lower and covariance norm comparison; it is an outward conservative common price for matter force inputs, not a proof of the parent whole action. Physical budgets retain 2^1024,2^2048 forces, times slowstd2^-3072 and omitted shifts2^-7168,2^-6144. New Gaussian derivative recurrence consumes full K1/K2 including heavy Lambda and charged unit contributions, and primitive G jets; AGA=K and C=.5A^-1, not scalar or commuting substitutes.',
 'source_boundary': 'Actual source_prices lines119-126 per-q D1 premise was refuted by global normal witness. No automatic adoption of it here: new Riesz jets cover the actual nonlinear source chart; real radial occupied W envelopes remain the independently repaired global_occupied certificate. This receipt removes the complex matter derivative premise, not root/static positivity/Haar/action/source propagation requirements.',
 'force_inequality': 'For E_ZPE=.5Tr sqrt(G^1/2 K G^1/2), |dE|<=n/4(||A|| ||dG||+||A^-1|| ||dK||), by AGA=K and Sylvester cyclic trace. Forces are evaluated POINTWISE at accepted rootRECT, not across a putative adiabatic Gaussian family on3R. Fast matter force scale2^1024, slow ZPEforce2^2048 and mixed block contribution2^0 follow the explicit block inverse-root derivation in induced_phi_complex_matter_jet_certificate_force_consumer.py. That consumer is authoritative for integration; the generic C prices here are comparison outputs, NOT adopted force certificates. Actual slowstd2^-3072 and mean Jacobian128Lambda remain explicit. No added all3R slowKpositivity premise: root positive pencil gives a local analytic germ.',
 'primitive_consumer': 'Primitive Hessians are polynomial Gram/ordered-link Hessians. The native primitive_matrix has K=128 J^T J+32 sum g_mu column_mu^T column_mu, Q opnorm<9/8, total g<2/3 and shift<TUBE; ||J||<4 therefore ||Kprimitive||<4096. Fast90 K consists of .025 q² I,.2 H and .35(q² I-dd*) with q²<=12, H=O^T O/8, ||O||<2+TUBE, hence norm<4096. Cauchy gives their slow derivative bounds j^j*4096/margin^j, with physical Lambda and Lambda^5/4 restored. G inverse jets use Horbit>=gmin/2-TUBE. The Riccati A1/A2 and covariance recurrence is POINTWISE at rootRECT using the accepted static pencil lower there. Fixed f_A(root), Xi_i(root) and centred cubic P are not replaced by f_A(slow); collar-wide complex matter K_j bounds are used for Taylor/source jets, not to demand an adiabatic slow-penc il on the collar.',
}

def digest(raw: bytes) -> str:
 return hashlib.sha256(raw).hexdigest()

def number(x: F) -> dict[str, str]:
 return {'upper':str(x),'decimal':str(mp.nstr(mp.mpf(x.numerator)/x.denominator,30))}

def analytic_orbit(frame: np.ndarray, T: np.ndarray) -> np.ndarray:
 comm=1j*(T[:,None]@frame[None]-frame[None]@T[:,None])
 return (2*np.einsum('amij,bji->mba',comm,T)).reshape(60,15)

def analytic_frame(frame: np.ndarray,T: np.ndarray,Q0: np.ndarray) -> np.ndarray:
 O=analytic_orbit(frame,T);P=np.eye(60)-O@np.linalg.solve(O.T@O,O.T)
 N=P@Q0;S=np.asarray(sqrtm(N.T@N),dtype=complex)
 return np.linalg.solve(S.T,N.T).T

def analytic_fields(data: dict[str,Any],s: np.ndarray,f: np.ndarray,C: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
 T=data['geom']['M']/np.sqrt(8)
 frame=data['native']['frame']+np.einsum('a,amij->mij',s,data['geom']['deltas'])/8
 Q=analytic_frame(frame,T,data['geom']['Q'])
 eta=(f@Q.T).reshape(64,4,15)
 fields=frame[None]+np.einsum('xma,aij->xmij',eta,T)
 links=np.asarray([expm(1j*x) for x in np.einsum('ea,aij->eij',C,T)])
 return fields,links

def analytic_h(profile: dict[str,Any],native: dict[str,Any],fields: np.ndarray,links: np.ndarray,wrong: str='') -> np.ndarray:
 h=np.zeros((256,256),complex)
 for x in range(64):h[4*x:4*x+4,4*x:4*x+4]=float(profile['onsite'])*fields[x,0]
 for path in field.native_paths(profile,native):
  x,y=int(path['source']),int(path['target']);U=np.eye(4,dtype=complex)
  edges=path['edges'][:1] if wrong=='two_link' else path['edges']
  for edge in edges:U=U@links[edge]
  Ui=np.linalg.inv(U);c=path['coefficients']
  Fx=np.einsum('m,mij->ij',c,fields[x]);Fy=np.einsum('m,mij->ij',c,fields[y])
  hop=path['seam']*(Fx@U+U@(Fx if wrong=='endpoint' else Fy))/2
  Fxr=np.einsum('m,mij->ij',c.conj(),fields[x]);Fyr=np.einsum('m,mij->ij',c.conj(),fields[y])
  reverse=path['seam']*(Ui@Fxr+Fyr@Ui)/2
  if wrong=='conjugate':reverse=hop.conj().T
  h[4*x:4*x+4,4*y:4*y+4]+=hop
  h[4*y:4*y+4,4*x:4*x+4]+=reverse
 return (-1 if profile['scale']=='heavy' else 1)*h

def projector(h: np.ndarray) -> np.ndarray:
 # Newton matrix sign: inverse only, analytic on the same two spectral sheets.
 X=h/max(1.,float(np.linalg.norm(h,2)))
 for _ in range(18):X=(X+np.linalg.inv(X))/2
 return (np.eye(len(h))-X)/2

def controls(data: dict[str,Any]) -> dict[str,Any]:
 rng=np.random.default_rng(222101064)
 def unit(shape: tuple[int,...]) -> np.ndarray:
  a=rng.normal(size=shape);return a/np.linalg.norm(a)
 s,f,C=unit((45,)),unit((64,45)),unit((192,15));f-=f.mean(axis=0);f/=np.linalg.norm(f)
 # Nonuniform complex point outside packet, still within the certified tube.
 scale=float(TUBE)/4;s*=scale;f*=scale;C*=scale
 p=data['profiles'][1];native=data['native']
 a,l=analytic_fields(data,s,f,C);h=analytic_h(p,native,a,l)
 reference=field.assemble(p,native,a,l)
 P=projector(h);w,V=np.linalg.eigh(reference);Pr=V[:,:128]@V[:,:128].conj().T
 out: dict[str,Any]={'real_operator_agreement':float(np.linalg.norm(h-reference)), 'real_projector_agreement':float(np.linalg.norm(P-Pr)), 'rank_trace':float(np.trace(P).real),'idempotence':float(np.linalg.norm(P@P-P))}
 z=1+0.6j;ac,lc=analytic_fields(data,z*s,z*f,z*C);hc=analytic_h(p,native,ac,lc);Pc=projector(hc)
 out['complex_commutation']=float(np.linalg.norm(hc@Pc-Pc@hc));out['complex_idempotence']=float(np.linalg.norm(Pc@Pc-Pc))
 out['complex_nonhermiticity']=float(np.linalg.norm(hc-hc.conj().T))
 eps=.001
 def H(t: complex,wrong: str='') -> np.ndarray:
  aa,ll=analytic_fields(data,t*s,t*f,t*C);return analytic_h(p,native,aa,ll,wrong)
 dreal=(H(z+eps)-H(z-eps))/(2*eps)
 dimag=(H(z+1j*eps)-H(z-1j*eps))/(2j*eps)
 out['holomorphic_CR_defect']=float(np.linalg.norm(dreal-dimag))
 wr=(H(z+eps,'conjugate')-H(z-eps,'conjugate'))/(2*eps)
 wi=(H(z+1j*eps,'conjugate')-H(z-1j*eps,'conjugate'))/(2j*eps)
 out['wrong_antiholomorphic_CR_defect']=float(np.linalg.norm(wr-wi))
 out['wrong_endpoint_error']=float(np.linalg.norm(H(z,'endpoint')-hc))
 out['wrong_two_link_error']=float(np.linalg.norm(H(z,'two_link')-hc))
 frame=native['frame']+np.einsum('a,amij->mij',z*s,data['geom']['deltas'])/8
 Q=analytic_frame(frame,data['geom']['M']/np.sqrt(8),data['geom']['Q'])
 out['complex_polar_transpose_isometry']=float(np.linalg.norm(Q.T@Q-np.eye(45)))
 out['wrong_frozen_polar_fast_displacement']=float(np.linalg.norm(f@(Q-data['geom']['Q']).T))
 out['wrong_frozen_polar_displacement_over_scale_squared']=out['wrong_frozen_polar_fast_displacement']/scale**2
 # Mixed/nonuniform scalar-energy Hessian versus Riesz projector product rule.
 u=unit((45,));v=unit((192,15));step=2e-4
 out['finite_control_scope']={'complex_point_coordinate_scale':scale,'complex_continuation_tube':float(TUBE),'mixed_FD_step':step,'mixed_FD_inside_certified_tube':False,'floating_controls_are_interval_proof':False}
 def energy(b: float,c: float) -> float:
  aa,ll=analytic_fields(data,s+b*u,f,C+c*v);hh=analytic_h(p,native,aa,ll)
  return float(np.trace(hh@projector(hh)).real)
 mixed=(energy(step,step)-energy(step,-step)-energy(-step,step)+energy(-step,-step))/(4*step**2)
 mixed2=(energy(step/2,step/2)-energy(step/2,-step/2)-energy(-step/2,step/2)+energy(-step/2,-step/2))/(step**2)
 out['nonuniform_mixed_energy_hessian']=mixed2;out['mixed_refinement_defect']=abs(mixed-mixed2)
 # Every q/90 operator, all65 profiles in the producer; BOTH response terms.
 def operator(b: float,c: float) -> np.ndarray:
  aa,ll=analytic_fields(data,s+b*u,f,C+c*v);return analytic_h(p,native,aa,ll)
 hs=(operator(step,0)-operator(-step,0))/(2*step)
 hcjet=(operator(0,step)-operator(0,-step))/(2*step)
 hsc=(operator(step,step)-operator(step,-step)-operator(-step,step)+operator(-step,-step))/(4*step**2)
 occupation=(w<0).astype(float);den=w[:,None]-w[None,:]
 kernel=np.zeros((256,256));mask=occupation[:,None]!=occupation[None,:]
 kernel[mask]=(occupation[:,None]-occupation[None,:])[mask]/den[mask]
 Ps=V@(kernel*(V.conj().T@hs@V))@V.conj().T
 product=float(np.trace(P@hsc+Ps@hcjet).real)
 out['mixed_Riesz_product_rule']=product
 out['mixed_energy_vs_Riesz_defect']=abs(product-mixed2)
 heavy=data['profiles'][0]
 hh=analytic_h(heavy,native,a,l);Ph=projector(hh)
 out['heavy_upper_original_sheet_defect']=float(np.linalg.norm(Ph-(np.eye(256)-projector(-hh))))
 rows=[]
 for qi in field.QINTEGERS:
  r=field.build_species(p,native,data['geom'],qi);direction=unit((90,))
  plus=np.einsum('kaij,kbij,kij->ab',r['PH_plus'].conj(),r['PH_plus'],1/r['gap_plus'])
  minus=np.einsum('kaij,kbij,kij->ab',r['PH_minus'].conj(),r['PH_minus'],1/r['gap_minus'])
  expected=r['contact']-plus-minus
  rows.append({'q':qi.tolist(),'contact_minus_both_PH_defect':float(np.max(abs(expected-r['sea']))),'wrong_omitted_minus_response':float(np.linalg.norm(minus)),'PH_shape':list(r['PH_plus'].shape)})
 out['all64q_fullPH90']=rows
 if max(out['real_operator_agreement'],out['real_projector_agreement'],out['complex_idempotence'],out['complex_commutation'],out['complex_polar_transpose_isometry'])>1e-9:raise ArithmeticError('Actual analytic continuation failed')
 if out['holomorphic_CR_defect']>1e-9 or out['wrong_frozen_polar_displacement_over_scale_squared']<1e-3 or min(out[k] for k in ('wrong_antiholomorphic_CR_defect','wrong_endpoint_error','wrong_two_link_error'))<1e-12:raise ArithmeticError('Adverse continuation/path/connection controls blind: '+json.dumps({k:v for k,v in out.items() if k!='all64q_fullPH90'}))
 if out['mixed_refinement_defect']>1e-4 or out['mixed_energy_vs_Riesz_defect']>1e-4 or out['heavy_upper_original_sheet_defect']>1e-9 or max(r['contact_minus_both_PH_defect'] for r in rows)>1e-9:raise ArithmeticError('Actual mixed/fullPH consumer failed')
 return out

def run() -> dict[str,Any]:
 start=time.monotonic();mp.mp.dps=100;field.full.iv.dps=100
 srcpath=ROOT/'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'
 raw=srcpath.read_bytes();src=json.loads(raw)
 if digest(raw)!='da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6':raise ArithmeticError('Pinned source input changed')
 inputs={str(srcpath):raw}
 for name in ('matter','quantum'):
  binding=src['input_descriptor'][name];path=ROOT/Path(binding['path']).name;b=path.read_bytes()
  if digest(b)!=binding['sha256']:raise ArithmeticError('Archived input mismatch')
  inputs[str(path)]=b
 matter=json.loads(inputs[str(ROOT/Path(src['input_descriptor']['matter']['path']).name)])
 quantum=json.loads(inputs[str(ROOT/Path(src['input_descriptor']['quantum']['path']).name)])
 saved=src['input_descriptor']['exact_NEW_allq_BO_mean'];centres=[F(x) for x in saved['centre_exact_rational']]
 data=field.load_mean(g_override=[centres[0],*([centres[1]]*3)],mean_descriptor={'law':'Archived NEW allq mean, no solve or adoption'})
 data['g']=[field.full.enclosure(F(saved['parameters'][k]['lower']),F(saved['parameters'][k]['upper'])) for k in ('g0','gs')]
 data['g']=[data['g'][0],*([data['g'][1]]*3)]
 for key in ('root_receipt','full_receipt'):
  path=Path(data['descriptor'][key]['path']);inputs[str(path)]=path.read_bytes()
 ex,iv=field.full.exact,field.full.iv
 gmin=min(F(field.full.bounds(g)['lower']) for g in data['g']);gmaxsum=sum(F(field.full.bounds(g)['upper']) for g in data['g'])
 if not gmin/2>F(1,20) or not gmaxsum/2<F(1,3):raise ArithmeticError('Required stronger polar root box unavailable')
 b=4*POLAR+POLAR**2;eta=b/F(2,5)+4*b/(F(2,5)*(F(2,5)-b))
 if (1+eta)**2/(1-eta)>=F(9,8)**2:raise ArithmeticError('Actual complex polar series bound failed')
 rows=[];K=[F(0)]*4;v1=F(0);v2=F(0);sea=F(0);minimum=None
 margin=TUBE-3*R
 for profile in data['profiles']:
  bounds=field.exact_operator_bounds(profile,data['g']);d=F(bounds['minimum_gap_unscaled']['lower'])/2
  H=abs(ex(profile['onsite']))*iv.sqrt(data['g'][0]);delta=ex(2*abs(profile['onsite'])*TUBE)
  for path in profile['paths']:
   hop=abs(ex(path['c0']))*iv.sqrt(data['g'][0])+abs(ex(path['ci']))*iv.sqrt(data['g'][path['axis']+1])
   c=abs(path['c0'])+abs(path['ci']);H+=2*hop
   delta+=2*((hop+ex(2*c*TUBE))/(1-ex(2*TUBE))-hop)
  Hhi=F(field.full.bounds(H)['upper']);delta_hi=F(field.full.bounds(delta)['upper'])
  if delta_hi>=d/4:raise ArithmeticError('Complex resolvent Neumann margin fails')
  Pmax=4*(Hhi+d)/d;Rz=4/d;length=Hhi+d
  hj=[F(n**n)*(Hhi+delta_hi)/margin**n for n in (1,2,3)]
  pj=[length*Rz**2*hj[0],length*(2*Rz**3*hj[0]**2+Rz**2*hj[1]),length*(6*Rz**4*hj[0]**3+6*Rz**3*hj[0]*hj[1]+Rz**2*hj[2])]
  Emax=256*(Hhi+delta_hi)*Pmax
  kj=[F((j+2)**(j+2))*Emax/margin**(j+2) for j in range(4)]
  for j in range(4):K[j]+=profile['load']*kj[j]
  v1+=profile['load']*90*(F(bounds['Phi_vertex_operator_upper']['upper'])+F(bounds['colour_vertex_operator_upper']['upper']))
  v2+=profile['load']*90*(F(bounds['PC_contact_entry_upper']['upper'])+F(bounds['CC_contact_entry_upper']['upper']))
  sea+=profile['load']*F(bounds['sea90_operator_upper']['upper']);minimum=d*2 if minimum is None else min(minimum,d*2)
  rows.append({'role':profile['role'],'load':profile['load'],'heavy_negative_sheet':profile['scale']=='heavy','mean_distance_lower':str(d),'mean_H_upper':number(Hhi),'complex_delta_h_upper':number(delta_hi),'Neumann_ratio_upper':number(2*delta_hi/d),'projector_norm_upper':number(Pmax),'projector_jets':[number(x) for x in pj],'matter_K0_to_K3_unscaled':[number(x) for x in kj],'normalized_projector_physical_scale_cancels':True,'physical_K_multiplier':'2^8192' if profile['scale']=='heavy' else '1'})
  rows[-1]['complex_zero_singular_value_lower']=str(d-delta_hi)
  rows[-1]['complex_zero_resolvent_upper']=number(1/(d-delta_hi))
  rows[-1]['physical_independent_CAR_copies']=profile['load']
 if minimum is None:raise ArithmeticError('No profiles')
 oldr=128*(8*v2+24*v1*v2/minimum+48*v1**3/minimum**2)
 slowlo=F(quantum['actual_full_field_covariance']['certified_q0_slow_positive_lower']['lower'])
 # A common outward replacement, keeping physical heavy scaling symbolic.
 C=F(2**24)*(1+oldr+sea+sum(76*7**j*K[j] for j in range(4)))**3*(1+1/minimum)**4*(1+2/gmin+1/slowlo)**4
 budgets={name:number(C*F(2)**power) for name,power in [('matter_fast_ZPE_force_upper',1024),('slowPhi_ZPE_force_upper',2048),('matter_fast_force_times_slow_std_upper',-2048),('slowPhi_ZPE_force_times_slow_std_upper',-1024),('mean_root_residual_shift_omitted_fastPH_upper',-7168),('mean_root_residual_shift_omitted_slowPhi_upper',-6144)]}
 # Consume NEW jets in the same noncommuting full physical Riccati inequality.
 data['frozen_slow_lower']=2*slowlo
 primitive=gaussian.outward_bounds(data,matter)['bounds']
 Gmax=F(8)+F(3,10)+12/(gmin/2-TUBE);G1=12*(F(2)+TUBE)/(4*(gmin/2-TUBE)**2);G2=12*(2*((F(2)+TUBE)/4)**2/(gmin/2-TUBE)**3+F(1,4)/(gmin/2-TUBE)**2)
 gmax=ex(Gmax)*ex(F(2)**-4096);gminphys=ex(F(3,10))*ex(F(2)**-4096)
 kmin=ex(min(slowlo,gmin/20))*ex(F(2)**8192)
 kmax=ex(4096+sea+sum(K))*ex(F(2)**10240)
 a=iv.sqrt(kmax/gminphys);ell=(gmax/gminphys)/(2*iv.sqrt(gminphys*kmin))
 kp=[ex(K[j])*(ex(F(2)**8192)+1)+ex(F(j**j)*4096/margin**j)*(ex(F(2)**10240)+ex(F(2)**8192)) for j in (1,2)]
 gp=[ex(G1)*ex(F(2)**-4096),ex(G2)*ex(F(2)**-4096)]
 a1=ell*(kp[0]+a*a*gp[0]);a2=ell*(kp[1]+2*a1*a1*gmax+4*a*a1*gp[0]+a*a*gp[1])
 amin=iv.sqrt(kmin/gmax);c1=a1/(2*amin**2);c2=a1*a1/amin**3+a2/(2*amin**2)
 with threadpool_limits(limits=1):finite=controls(data)
 captures={**field.CAPTURED,**source.SOURCE_BYTES,Path(gaussian.__file__).name:Path(gaussian.__file__).read_bytes(),Path(__file__).name:SELF}
 stable={name:{'sha256':digest(bytes_),'bytes_utf8':bytes_.decode(),'unchanged':(ROOT/name).read_bytes()==bytes_} for name,bytes_ in captures.items()}
 if not all(x['unchanged'] for x in stable.values()) or not all(Path(path).read_bytes()==bytes_ for path,bytes_ in inputs.items()):raise ArithmeticError('Immutable epoch changed')
 return {'status':'ACTUAL_COMPLEX_MATTER_JET_CERTIFICATE_CONDITIONAL_ROOT_STATIC_PENCIL_NOT_SOURCE_ADMISSION','proof':PROOF,'dimensions':{'normal':5760,'slow':45,'q':64,'PH':90,'occupied_per_copy':128,'profiles':len(rows)},'domain':{'support':str(R),'collar':str(3*R),'complex_tube':str(TUBE),'complex_Cauchy_margin':str(margin),'polar_complex_radius':str(POLAR),'polar_eta_upper':str(eta)},'profiles':rows,'all_species_unscaled_K0_K1_K2_K3':[number(x) for x in K],'all64q_slow_tensor_HS_K1_K2_K3':[number(76*7**j*K[j]) for j in (1,2,3)],'full_nonuniform_tensor_HS_K1_K2_K3':[number(76**(j+1)*K[j]) for j in (1,2,3)],'frozen_budget_trace':{'v1':str(v1),'v2':str(v2),'riesz3':str(oldr),'minimum_gap':str(minimum),'replacement_coefficient':number(C),'old_affine_price_adopted':False},'actual_force_input_budgets':budgets,'full_physical_Gaussian_jet_consumer':{'A1':field.full.bounds(a1),'A2':field.full.bounds(a2),'Cov1':field.full.bounds(c1),'Cov2':field.full.bounds(c2),'primitive_input_bounds':primitive,'scope':'Outward full PH90/5760 Riccati jet norm recurrence; conservative global worst-pencil bound, not parent weighted whole action.'},'finite_actual_controls':finite,'input_capsule':{path:{'sha256':digest(bytes_),'bytes_utf8':bytes_.decode()} for path,bytes_ in inputs.items()},'source_at_execution':stable,'plan_revision':'34c80e78e7fbc97c60c8ac8294df069a56f4dc459f43879d17c8862dce6bd246','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','runtime':{'executable':sys.executable,'python':sys.version},'elapsed_seconds':time.monotonic()-start}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 result=run();args.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'profiles':result['dimensions']['profiles'],'domain':result['domain'],'force_times_slowstd':{k:v['decimal'] for k,v in result['actual_force_input_budgets'].items() if 'times' in k},'elapsed_seconds':result['elapsed_seconds']},indent=2))
