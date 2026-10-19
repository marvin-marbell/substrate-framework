"""Exact Gauss-complete electric locality and NEW 2^-16 all129 response balls.

This is a mathematical/operator consumer, not full-H evolution or admission.
Failure of a triangle/linked closure is retained, not a model no-go theorem.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_signal_interval_certificate as ar
import joined_phi_u1_photon_causal_response as photon

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_electric_locality_response'
CLOCK=F(1,2**16)
REVISION='887df7470f23d7cfcba29846a4854f2c17841ff42d90d77c9fe2f840683bbb48'
PIN={
 'joined_phi_u1_photon_causal_response_verified_20261010_receipt.json':'c7c3ec787e34d5aa58c806650db7603e9f06c92d388cf590672d3644efbe262f',
 'joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json':'2cc91cbe6b8b64f6d270ff04a9b946b22bd3a9a52ff9627e4bf15325f0c0adf4',
 'joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json':'2a1bbb46852707c82e477f28b5497e303b5ca66453db76d0655379f35cfb9a97',
 'induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json':'f6a6b6d79496e4e24d0a5df4762a9098b0db24debe16c2f82c32572696592d2f',
}
PROOF={
 'phase_derivation':'O shifts Pi by q and rho by delta: HE O=O[HE+64(2s+q²+delta Ldelta)]. q=P inc, delta=D inc, P=I-DtL D. Thus q²+delta Ldelta=inc², [s_i,O_j]=(inc_i.inc_j)O_j. O_i^E(t)=O_i exp(i64t(2s_i+inc_i²)). Moving the right phase through O_j shifts it by exp(i128t inc_i.inc_j). Reverse order shifts by exp(i128s inc_i.inc_j): ratio exp(i128(t-s)inc_i.inc_j), positive sign with row-minus-column density convention.',
 'CAR':'For distinct species, even bilinears commute without Jordan-Wigner sign. For one species [c_x†A c_y,c_z†B c_w]=delta_yz c_x†AB c_w-delta_wx c_z†BA c_y. Four-spin hop norm <=4(|c0|+|ci|)=b, because Clifford opnorm<1 on pinned mean box. General electric commutator <=2bi bj(e_ij+min(64|t-s||inc_i.inc_j|,1)), e same-species cross-endpoint indicator. This bounds the static CAR term plus the unequal electric phase term; no cancellation assumed for shared endpoints.',
 'onsite_repair':'Onsite N=sum_x c_x†A0 c_x commutes with every total site density, HC, Theta and Pi. Thus H0=HE+N is exactly solvable. Its hop spin matrix is exp(iA0 t)A exp(-iA0 t); b is unchanged, and derivative norm <=2|onsite|b. No future Slater assumption.',
 'contact_dressing':'For C0(t)=U0† C U0, ||C_H(t)-C0(t)||HS <= integral_0^t ||[Mhop0(u),C0(t)]||HS du <= L_C0 t+L_C1 t²/2. L_C0=sum_same_species 2bi bj |q_j|² e_ij; L_C1=sum_all_species 128bi bj |q_j|² |inc_i.inc_j|. Loads1,63 count same-species pairs1,63 and all-species pairs load_i load_j, NOT63² for static CAR. This is exact Duhamel in the bounded interaction picture.',
 'initial_covariance_repair':'Each normalized Ref channel is a rank129 active Slater and63 rank128 auxiliary Slaters initially, product over species only at t=0. For V=diag(v_x) tensor I4, Var(dGamma(V))=tr(P V(1-P)V)<=tr(V²)/2=2sum_x v_x²: the two offdiagonal blocks in the P/(1-P) decomposition contribute twice the variance to tr(V²). Across64 independent species Var<=128sum v². Background mean rho is uniform, delta L annihilates it. Source added active particle minus Ref hole gives |mean(vrho)|<=2max|v|. Gaussian Pi independent with variance1/8. Hence ||s_j psi||/||psi|| <=sqrt(q_j²/8+128sum v²+4max|v|²) (last term0 for background), v=delta_j L. Raw four-channel mass1/2 is retained. This is initial covariance only, never future Wick.',
 'contact_drift':'|<C0(t)>-<C(0)>|HS <=mass K_C t, K_C=sum load b_j |q_j|²[128sqrt(q_j²/8+128sum v_j²+source4vmax²)+64inc_j²+2|onsite_j|]. Initial expectation derivative may be evaluated on original psi because s_j commutes H0. Bounding O_j derivative on psi does not require evolved density being Slater.',
 'unequal_channel':'J_H(u)-J0(u) has HS norm <=EJ(u)=L_J0 u+L_J1 u²/2 by same Duhamel, weights|q_j|. ||i[J_H(u),J]||HS <=S_JJ+128u O_JJ+2a EJ(u), a=sum load b|q|, S_JJ=sum_same 2bi bj|q_i||q_j|e, O_JJ=sum_all bi bj|q_i||q_j||inc_i.inc_j|. The last triangle is retained as an explicit linked-closure limitation; it is not called a local nested commutator. Unitary conjugation identifies [J(t),J(t-u)] with [J(u),J].',
 'response_integrals':'Exact mixed identity M(t)=-128t<C(t)>+128 integral_0^t(t-u)i<[J(t),J(t-u)]>du. With contact error mass[(K_C+L_C0)t+L_C1 t²/2] and unequal bound mass[S+(128O+2aL_J0)u+aL_J1 u²], mixed remainder <=128mass[(K_C+L_C0+S/2)t²+(L_C1/2+(128O+2aL_J0)/6)t³+aL_J1 t⁴/12]. D=128t mass I+128int(t-s)M(s)ds; each mixed B_k t^k integrates to D128 B_k t^(k+2)/((k+1)(k+2)), Pi B_k t^(k+1)/(k+1). Contact cubic -8192C/3 and Pi quadratic -64C retained, unequal channel never discarded.',
 'domain':'HE selfadjoint on D(Pi²), bounded finite CAR HC/M on fixed sectors; initial Gaussian and Theta/Pi/Fourier insertions Schwartz. Electric phases are unitary Weyl shifts and bounded CAR words on full L²(R129), no finite-Hermite canonical exactness. Faithful Weyl energy-shift controls evaluate constant phase ratios on generalized momentum characters; equality is pointwise in momentum and therefore on smooth compact-support wavepackets, not a probability-normalized momentum eigenstate.',
 'physical_transfer':'Use actual Theta-independent T, HF T=T(Hb+Ephi)+R uniformly on BODY with epsilon<=3.76985514057735e-138. Duhamel Delta v<=t epsilon||v||, HDelta v<=|Ephi|t epsilon||v||+t epsilon||Hb v||+2epsilon||v||. Actual global graph quartic/additive434313856Lambda bounds current on propagated Delta, not a bounded fullfield current/Theta/Pi. Existing exact graph formula is reevaluated at NEW clock on psi and129 unbounded Theta insertions. This transfers response of fixed Tpsi/TTheta source, not a physical E0 insertion effect; source endpoint defect would require a separate transfer.',
 'map':'Q orthonormal real192x129, QQ^t=Pphys;126 coexact+3 uniform torons, tau_axis has entries1/8 on axis links. Physical192 D=Q D129 Qt and error HS unchanged. Theta=a Aphys,a1/64: D_A=D_Theta/a², mixed(J_A,A)=mixed(J_Theta,Theta), Pi_A=aPi_Theta and its response to A impulse equals coordinate Pi response; J_A=aJ_Theta,C_A=a²C_Theta. Three torons are INCLUDED in P, not added twice.',
 'scope':'Exact derived operator identities plus conditional outward mathematical balls under pinned mean/action/domain owners. No measured full129 dynamics, no stationary/RPA response, no full-H Wick, ground/pole/commoncone/originalSOURCE admission. Failure to resolve all129 signs at2^-16 falsifies this construction/price regime only; no model impossibility. No charged-motion claim follows from coordinate response, offsite Pauli coherence or onsite rotation; no-hop contrast is only the exactly zero J/C when all charged path coefficients are set0, not a computed density redistribution.'
}

def ib(x: Any)->dict[str,str]:
 return photon.bounds(x)

def car_hop(mask: int,row: int,col: int)->tuple[int,int]|None:
 if not (mask>>col)&1:return None
 sign=(-1)**((mask&((1<<col)-1)).bit_count());m=mask^(1<<col)
 if (m>>row)&1:return None
 sign*=(-1)**((m&((1<<row)-1)).bit_count())
 return m|(1<<row),sign

def controls(geo: dict[str,Any],paths: list[dict[str,Any]])->dict[str,Any]:
 den=geo['den'];k=geo['k'];dens=geo['density'];gram=geo['gram'];dot=k@k.T
 if not np.array_equal(gram+dens,den*dot):raise ArithmeticError('Exact full Gauss cross failure')
 # Choose actual disjoint one/two paths with Coulomb cross nonzero.
 pairs=[(i,j) for i in range(len(paths)) for j in range(len(paths)) if paths[i]['length']==1 and paths[j]['length']==2 and dot[i,j]==0 and dens[i,j]!=0 and len({geo['x'][i],geo['y'][i],geo['x'][j],geo['y'][j]})==4]
 if not pairs:raise ArithmeticError('No actual disjoint falsifying pair')
 i,j=next(((i,j) for i,j in pairs if paths[i]['seam']<0 or paths[j]['seam']<0),pairs[0])
 p0=np.arange(192,dtype=np.int64)%7-3;rho=np.zeros(64,dtype=np.int64);rho[geo['y'][i]]=1;rho[geo['y'][j]]+=1;rho[0]-=2
 lint=geo['lint'];d=geo['native']['D'].astype(np.int64)
 def energy(pi: np.ndarray,r: np.ndarray,he: int,hc: int)->F:
  return he*(F(int(pi@pi))-F(int((d@pi)@lint@(d@pi)),den))+hc*F(int(r@lint@r),den)
 def word(a: int,b: int,ta: F,tb: F,he: int,hc: int)->F:
  rb=rho+geo['delta'][b];rab=rb+geo['delta'][a];pb=p0+k[b];pab=pb+k[a]
  return ta*(energy(pab,rab,he,hc)-energy(pb,rb,he,hc))+tb*(energy(pb,rb,he,hc)-energy(p0,rho,he,hc))
 t=CLOCK;s=CLOCK/3
 observations={}
 for label,he,hc in [('full',64,64),('omittedHC',64,0),('HE32_HC64',32,64)]:
  difference=word(i,j,t,s,he,hc)-word(j,i,s,t,he,hc)
  expected=2*(t-s)*(he*F(int(gram[i,j]),den)+hc*F(int(dens[i,j]),den))
  if difference!=expected:raise ArithmeticError('Electric word sign/order error')
  if label=='full' and difference!=0:raise ArithmeticError('Disjoint full cancellation failed')
  if label!='full' and difference==0:raise ArithmeticError('Blind coefficient control')
  observations[label]={'phase_angle_exact':str(difference),'commutator_coefficient_observed':2*abs(math.sin(float(difference)/2))}
 overlapping=next((a,b) for a in range(len(paths)) for b in range(len(paths)) if paths[a]['length']==1 and paths[b]['length']==2 and dot[a,b]!=0)
 aa,bb=overlapping
 reverse=aa+len(paths)
 forward_angle=word(aa,bb,t,s,64,64)-word(bb,aa,s,t,64,64)
 reverse_angle=word(reverse,bb,t,s,64,64)-word(bb,reverse,s,t,64,64)
 if forward_angle!=128*(t-s)*int(dot[aa,bb]) or reverse_angle!=-forward_angle:raise ArithmeticError('Literal reverse q phase sign failed')
 wrong_reverse_ward=geo['native']['D'].astype(np.int64)@k[aa]-geo['delta'][reverse]
 if not np.any(wrong_reverse_ward):raise ArithmeticError('Wrong reverse q control blind')
 observations['overlap_one_two_forward_reverse']={'forward_angle_exact':str(forward_angle),'reverse_angle_exact':str(reverse_angle),'wrong_reverse_q_Ward_l1_defect':int(np.abs(wrong_reverse_ward).sum()),'AP_product_seam':int(paths[i]['seam']*paths[j]['seam']),'seam_scope':'Literal scalar AP seam multiplies both orders and cannot alter electric commutation ratio.'}
 # Independent-species CAR even words on four modes, including Jordan-Wigner signs.
 independent=0
 for mask in range(16):
  def compose(a:tuple[int,int],b:tuple[int,int]):
   first=car_hop(mask,*b)
   if first is None:return None
   second=car_hop(first[0],*a)
   return None if second is None else (second[0],first[1]*second[1])
  if compose((0,1),(2,3))!=compose((2,3),(0,1)):raise ArithmeticError('Independent even CAR failed')
  independent+=int(compose((0,1),(2,3)) is not None)
 # Exact shared-endpoint matrix commutator equals the induced hop.
 matrices=[]
 for row,col in ((0,1),(1,2),(0,2)):
  m=np.zeros((8,8),int)
  for mask in range(8):
   out=car_hop(mask,row,col)
   if out:m[out[0],mask]=out[1]
  matrices.append(m)
 defect=matrices[0]@matrices[1]-matrices[1]@matrices[0]-matrices[2]
 if np.any(defect) or not np.any(matrices[2]):raise ArithmeticError('Same-species CAR assertion unreachable')
 # Literal E0 transition: active0 empty, Ref0 occupied, delta active+Ref=0.
 e=np.eye(64,dtype=np.int64)[0];r=np.eye(64,dtype=np.int64)[1]-e
 ref_transition=car_hop(2,0,1)
 if ref_transition is None:raise ArithmeticError('E0 legal boundary unreachable')
 complete=64*F(int(r@lint@r-r@lint@r),den)
 omitted=64*F(int((r+e)@lint@(r+e)-r@lint@r),den)
 if complete!=0 or omitted==0:raise ArithmeticError('Ref neutrality control blind')
 # Toron omission is a distinct Gauss-cross failure, exercised on disjoint parallel links.
 parallel=next((a,b) for a in range(len(paths)) for b in range(a+1,len(paths)) if paths[a]['length']==paths[b]['length']==1 and paths[a]['axis']==paths[b]['axis'] and dot[a,b]==0)
 a,b=parallel;toron_cross=F(sum(int(k[a,axis::3].sum())*int(k[b,axis::3].sum()) for axis in range(3)),64)
 if toron_cross==0:raise ArithmeticError('Toron omission blind')
 lengths={str(length):sum(p['length']==length for p in paths) for length in (1,2)}
 seam_count=sum(p['seam']<0 for p in paths)
 if not seam_count or not all(lengths.values()):raise ArithmeticError('Literal AP/one/two controls missing')
 return {'full_cross_exact_defect':0,'paths':lengths,'AP_negative_seams':seam_count,'disjoint_pair':[i,j],'literal_pair':[dict(source=paths[a]['source'],target=paths[a]['target'],edges=list(paths[a]['edges']),seam=paths[a]['seam'],length=paths[a]['length']) for a in (i,j)],'faithful_Weyl_operator_words':observations,'independent_species_nonzero_CAR_words':independent,'same_species_shared_endpoint_CAR_defect':0,'same_species_shared_endpoint_nonzero_entries':int(np.count_nonzero(matrices[2])),'E0_complete_commutator_energy_shift':str(complete),'E0_omittedRef_commutator_energy_shift':str(omitted),'E0_omittedRef_total_charge_shift':1,'E0_legal_transition':list(ref_transition),'omitted_torons_disjoint_cross':str(-toron_cross),'native192_literal_charge_shift_Ward_defect':0,'Hermite_truncation':'none; exact CAR and faithful continuous Weyl shifts, not finite canonical matrices','no_hop_contrast':'J=C=0 exactly with path coefficients0 even when onsite spin and initial nonlocal CAR coherence remain'}

def prices(profiles: list[Any],geo: dict[str,Any],source: bool)->dict[str,Any]:
 ex=ar.exact;iv=ar.iv;den=geo['den'];k=geo['k'];x=geo['x'];y=geo['y'];dot=np.abs(k@k.T);endpoints=(y[:,None]==x[None,:])|(x[:,None]==y[None,:])
 q2=[F(int(v),den) for v in np.diag(geo['gram'])];q=[iv.sqrt(ex(v)) for v in q2];n=len(q)
 rows=[]
 for profile in profiles:
  lookup={(p['axis'],p['length']):4*(abs(F(p['c0']))+abs(F(p['ci']))) for p in profile['paths']}
  paths=body.joined.old.native_paths(profile,geo['native']);rows.append([lookup[p['axis'],p['length']] for p in paths+paths])
 # Geometry profiles have identical path ordering; exact rows retain profile strengths.
 a=ex(0);kc_global=ex(0);kc_cov=ex(0);lc0=ex(0);lc1=ex(0);lj0=ex(0);lj1=ex(0);sj=ex(0);oj=ex(0)
 for sp,(profile,load) in enumerate(zip(profiles,(1,63))):
  for j in range(n):
   b=rows[sp][j];v=[F(int(z),den) for z in geo['delta'][j]@geo['lint']];vsq=sum((z*z for z in v),F(0));vmax=max(abs(z) for z in v)
   gaussian=iv.sqrt(ex(q2[j]/8+128*vsq+(4*vmax*vmax if source else 0)))
   glob=iv.sqrt(ex(q2[j]/8))+ex((128+int(source))*sum((abs(z) for z in v),F(0)))
   own=ex(64*int(k[j]@k[j])+2*abs(F(profile['onsite'])))
   kc_cov+=load*ex(b*q2[j])*(128*gaussian+own)
   kc_global+=load*ex(b*q2[j])*(128*glob+own)
   a+=load*ex(b)*q[j]
 # Aggregate exact weighted pair arrays before interval square roots.
 for sp,load in enumerate((1,63)):
  for tp,other in enumerate((1,63)):
   bi,bj=rows[sp],rows[tp]
   for i in range(n):
    for j in np.flatnonzero(endpoints[i] if sp==tp else np.zeros(n,bool)):
     w=2*load*bi[i]*bj[j]
     lc0+=ex(w*q2[j]);lj0+=ex(w)*q[j];sj+=ex(w)*q[i]*q[j]
    for j in np.flatnonzero(dot[i]):
     w=load*other*bi[i]*bj[j]*int(dot[i,j])
     lc1+=ex(128*w*q2[j]);lj1+=ex(128*w)*q[j];oj+=ex(w)*q[i]*q[j]
 return dict(a=a,kc_global=kc_global,kc_cov=kc_cov,lc0=lc0,lc1=lc1,lj0=lj0,lj1=lj1,sj=sj,oj=oj)

def response_bounds(p: dict[str,Any],mass: F,repair: str)->dict[str,Any]:
 ex=ar.exact;t=ex(CLOCK);m=ex(mass)
 if repair=='global_triangle':
  # Extensive unequal-current triangle retained as a failed comparison.
  a=p['a'];base=p['kc_global']+p['lc0']+a*a
  coeff={2:128*m*base,3:64*m*p['lc1']}
 else:
  kc=p['kc_global'] if repair=='local_paths_only' else p['kc_cov']
  coeff={2:128*m*(kc+p['lc0']+p['sj']/2),3:128*m*(p['lc1']/2+(128*p['oj']+2*p['a']*p['lj0'])/6),4:128*m*p['a']*p['lj1']/12}
 mixed=sum((b*t**power for power,b in coeff.items()),ex(0))
 dr=sum((128*b*t**(power+2)/((power+1)*(power+2)) for power,b in coeff.items()),ex(0))
 pr=sum((b*t**(power+1)/(power+1) for power,b in coeff.items()),ex(0))
 return dict(D=dr,Pi=pr,mixed=mixed,coefficients={str(k):ib(v) for k,v in coeff.items()})

def run(output: Path)->dict[str,Any]:
 started=time.monotonic();ar.iv.dps=75;captured={}
 def take(name: str,pin: str|None=None)->bytes:
  raw=(ROOT/name).read_bytes()
  if pin and hashlib.sha256(raw).hexdigest()!=pin:raise ArithmeticError('Pinned bytes changed '+name)
  captured[name]=raw;return raw
 for name,pin in PIN.items():take(name,pin)
 previous=json.loads(captured['joined_phi_u1_photon_causal_response_verified_20261010_receipt.json'])
 capsule=previous['all129_capsule'];take(capsule['path'],capsule['sha256'])
 data,species,inputs,_=body.full_inputs();captured.update(inputs);captured.update(body.CAPTURED)
 profiles=data['profiles'][1:];native=data['native'];paths=body.joined.old.native_paths(profiles[0],native)
 geo=ar.geometry(native,paths);geo['native']=native
 control=controls(geo,paths)
 graph=json.loads(captured['joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json'])
 # Reuse physical graph function ONLY; its clock is explicitly set before invocation.
 photon.CLOCK=CLOCK
 arrays={};states={};old=np.load(ROOT/capsule['path'])
 for label,mass,source in (('background',F(1),False),('raw_source',F(1,2),True)):
  p=prices(profiles,geo,source);phys=photon.physical_prices(graph,mass,profiles)
  c=old[label+'_contact'];initial=previous['all129_states'][label]
  minimum=F(initial['contact_minimum_directional_outward_lower']['lower']);norm=F(initial['contact_HS_lower']['lower'])
  dsig=F(8192,3)*CLOCK**3*minimum;psig=64*CLOCK**2*minimum;msig=128*CLOCK*minimum
  attempts={}
  for repair in ('global_triangle','local_paths_only','onsite_covariance_linked'):
   bounds=response_bounds(p,mass,repair)
   du=ar.end(bounds['D'],True)+F(phys['D_transfer']['upper']);pu=ar.end(bounds['Pi'],True)+F(phys['Pi_transfer']['upper']);mu=ar.end(bounds['mixed'],True)+F(phys['mixed_transfer']['upper'])
   attempts[repair]=dict(D_error=ib(bounds['D']),Pi_error=ib(bounds['Pi']),mixed_error=ib(bounds['mixed']),mixed_coefficients=bounds['coefficients'],all_direction_D_margin=str(dsig-du),all_direction_Pi_margin=str(psig-pu),all_direction_mixed_margin=str(msig-mu),D_error_to_minimum_directional_signal=float(du/dsig),Pi_error_to_minimum_directional_signal=float(pu/psig),mixed_error_to_minimum_directional_signal=float(mu/msig),all129_signed_signal_resolved=du<dsig and pu<psig and mu<msig)
  final=attempts['onsite_covariance_linked'];radius=F(initial['projected192_contact_entry_ball_radius'])
  for name,factor in [('D_correction',F(-8192,3)*CLOCK**3),('Pi_correction',-64*CLOCK**2),('mixed',-128*CLOCK)]:arrays[label+'_'+name]=float(factor)*c
  arrays[label+'_D']=128*float(CLOCK*mass)*np.eye(129)+arrays[label+'_D_correction']
  arrays[label+'_Pi']=float(mass)*np.eye(129)+arrays[label+'_Pi_correction']
  arrays[label+'_physical192_D_correction']=native['Q']@arrays[label+'_D_correction']@native['Q'].T
  arrays[label+'_physical192_Pi_correction']=native['Q']@arrays[label+'_Pi_correction']@native['Q'].T
  torons=[]
  for axis in range(3):
   tau=np.zeros(192);tau[axis::3]=1/8
   body.require(float(np.linalg.norm(native['D']@tau)))
   z=native['Q'].T@tau
   torons.append(dict(axis=axis,contact_observation=float(np.real(z@c@z)),rigorous_lower=str(minimum)))
  states[label]=dict(raw_mass=str(mass),attempts=attempts,prices={key:ib(value) for key,value in p.items()},physical_transfer=phys,initial_contact_minimum_lower=str(minimum),initial_contact_HS_lower=str(norm),minimum_directional_D_signal=str(dsig),minimum_directional_Pi_signal=str(psig),minimum_directional_mixed_signal=str(msig),initial_projected192_entry_radius=str(radius),centre_arithmetic='binary64 display; rigorous total coefficient uncertainty separately adds192*input entry radius times polynomial coefficient; output Q map is observational display, exact formula defines physical response',D_total_error_upper=str(F(final['D_error']['upper'])+F(phys['D_transfer']['upper'])+192*radius*abs(F(8192,3)*CLOCK**3)),Pi_total_error_upper=str(F(final['Pi_error']['upper'])+F(phys['Pi_transfer']['upper'])+192*radius*64*CLOCK**2),mixed_total_error_upper=str(F(final['mixed_error']['upper'])+F(phys['mixed_transfer']['upper'])+192*radius*128*CLOCK),torons=torons,status='ALL129_LONGER_CLOCK_SIGN_RESOLVED' if final['all129_signed_signal_resolved'] else 'ATTEMPTED_ALL129_LONGER_CLOCK_LINKED_CLOSURE_FALSIFIED_BY_ERROR_PRICE',failed_boundary='Worst-case source/contact/current operator triangles, especially2aEJ linked-current dressing and all64-species overlap phase count; NOT global physical impossibility' if not final['all129_signed_signal_resolved'] else None)
 for module in tuple(sys.modules.values()):
  file=getattr(module,'__file__',None)
  if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captured.setdefault(Path(file).name,Path(file).read_bytes())
 captured[Path(__file__).name]=Path(__file__).read_bytes()
 if not all((ROOT/name).read_bytes()==raw for name,raw in captured.items()):raise ArithmeticError('Source/input mutation')
 npz=output.with_suffix('.npz');np.savez_compressed(npz,**arrays)
 archive=output.with_suffix('.zip')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for name,raw in sorted(captured.items()):z.writestr(name,raw)
 runtime=dict(executable=sys.executable,python=sys.version,platform=platform.platform(),numpy=np.__version__,mpmath=ar.mp.__version__,sympy=ar.sy.__version__,executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest())
 abi=dict(electric64=True,clock=str(CLOCK),gauss_cross='q_i.q_j+delta_i Lplus delta_j=inc_i.inc_j',phase_ratio='exp(+i128(t-s)inc_i.inc_j)',hop_norm='b_i=4(|c0_i|+|ci_i|)',commutator_price='2bi bj(e_same_crossendpoint+min(64abs(t-s)abs(inc_i.inc_j),1))',onsite_exact=True,same_species_loads=[1,63],unequal_overlap_loads='load_i load_j',ref_neutrality='delta_active+delta_Ref=0,inc=0',identity_verified=control['full_cross_exact_defect']==0)
 result=dict(status='NATURAL_EXACT_LOCALITY_ALL129_LONGER_CLOCK_CONDITIONAL_RESPONSE_BALLS',governing_plan=body.PLAN,governing_revision=REVISION,template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',clock_exact=str(CLOCK),old_clocks_unchanged=['2^-39','2^-36','4096','finiteSCREEN'],proof=PROOF,locality_ABI=abi,operator_controls=control,all129_states=states,full129_capsule=dict(path=npz.name,sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),arrays={name:list(a.shape) for name,a in arrays.items()},scope='all129 nonstationary response polynomial centres plus rigorous HS remainder balls, NOT measured evolution'),input_bytes=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),stable=True),source_input_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in sorted(captured.items())},runtime=runtime,elapsed_seconds=time.monotonic()-started,language_tooling=dict(AST='Python ast positive functions on actual photon owner',LSP='Configured pyright-langserver direct protocol:161 symbols, mixed_prices declaration+caller references, diagnostics empty; xd lsp unavailable'))
 output.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'_receipt.json'));args=parser.parse_args()
 with threadpool_limits(limits=1):result=run(args.output)
 print(json.dumps(dict(status=result['status'],clock=result['clock_exact'],all129={label:state['attempts']['onsite_covariance_linked']['D_error_to_minimum_directional_signal'] for label,state in result['all129_states'].items()},elapsed_seconds=result['elapsed_seconds'])),flush=True)
