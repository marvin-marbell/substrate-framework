"""Explicit full PH90 slow/fast inverse-root force bounds from complex jets.

Consumes the scientific jet receipt; does not adopt its generic common C price.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import mpmath as mp

ROOT=Path(__file__).resolve().parent
SELF=Path(__file__).read_bytes()
PROOF={
 'normalization':'B=G^1/2 K G^1/2, E=.5Tr sqrtB, dE=.25Tr B^-1/2 dB. chi=2^-4096, Lambda=2^8192. Nonzero q fast B=2^6144 Bf0+2^4096 Bmatter+Bunit*2^-4096. At q0 slow Bs has scale2^4096, fast Bf2^6144, cross2^4096. The 63 auxiliary profiles/load and heavy negative occupied sheet are retained by K0/K1. Unscaled K_j are actual derivative bounds, never fitted coefficients.',
 'sqrtG_derivative':'For G0>=gm I, ||D sqrtG0||<=G1/(2sqrtgm). Consequently ||Bpart||<=gM kpart and ||D Bpart||<=gM kpart1+sqrt(gM/gm)G1 kpart0. This handles noncommuting G and the actual normal-polar connection through K1; no frozen G shortcut.',
 'block_resolvent':'For T=B+tI, block inverse Schur formula. If ||Bpc||²/(lambda_min Bs lambda_min Bf)<1/2 and both diagonals have positive lower bs,bf, ||Rss||<=2/(bs+t),||Rff||<=2/(bf+t),||Rsf||<=2c/((bs+t)(bf+t)). Relative to diagonal reference Bf0, ||Rff-Rf0||<=2 deltaf/(bf+t)^2+2c²/((bf+t)^2(bs+t)), provided bf lower is chosen below both full/reference fast diagonals. Identity B^-1/2=(1/pi)int_0^infty t^-1/2(B+t)^-1dt yields ||(B^-1/2)ss||<=2/sqrtbs, ||(B^-1/2)sf||<=2c/[sqrtbs sqrtbf(sqrtbs+sqrtbf)]<=2c/(sqrtbs bf), ||(B^-1/2)ff-Bf0^-1/2||<=(deltaf+c²/bs)/bf^1.5. Integral int t^-1/2/(bf+t)^2 =pi/(2bf^1.5). This is an actual positive-matrix bound, not a named resolvent premise.',
 'force':'Subtract ONLY the leading fast derivative .25Tr Bf0^-1/2 dBf0. Trace block product gives residual <=ns/2 Bs1/sqrtbs +nf/2 deltaf1/sqrtbf +nf/4 (deltaf0+c0²/bs) Bf01/bf^1.5 +min(ns,nf)c0 c1/(sqrtbs bf). Restore scales: first term2^2048, middle two2^1024, cross term2^0. Unscaled bs,bf halves chosen below diagonal lower; matter fast relative perturb2^-2048 and unit ratio2^-10240. Whole matrix PH90 mixing is retained rather than set M=0. For all64q, use ns=45,nf=5715; these counts also overbound q0 alone. Multiplication by the slow Gaussian SCALE2^-3072 gives powers-1024,-2048 and cross-3072; source_weighting supplies covariance/corrector/normalization. Division by classical mean Jacobian128Lambda gives root-shift powers-6144,-7168,-8192. These are matter/slow coefficient forces, NOT full parent nonlinear action bounds.',
 'source_weighting':'2^-3072 is a Gaussian STD SCALE, not a unit-variance claim. At q0, Gpp=8 chi I and Gpc=0; C=.5 G^1/2 B^-1/2 G^1/2 and the Schur bound give Cpp<=8/sqrt(bs) times2^-6144. For degree3 P, Holder gives ||(F.s)P f||<=3^(7/4)||F.s f|| ||P||2<9||F.s f|| ||P||2. Thus each force action on UNNORMALIZED b6(1-P)f gains sqrt(8/sqrt(bs))*(1+9p). The direct rational lower1-24TrC/R²-4p>1/2 licenses normalization norm upper2; prepared total is priced separately. Mean-root gradient shift has no source variance/normalization multiplier. Fixed fA/Xi/source are not changed.',
 'primitive':'For Gram primitive Hessian on the polar slow tube ||Kprimitive||<4096; fast K90 .025q²,.2H,.35curl has norm<4096. Derivative common upper4096/margin from complex Cauchy. G0 max<=8+.3+12/Hlower and G1<=12(Omax+TUBE)/(4 Hlower²), Hlower=gmin/2-TUBE. Exact mean box supplies these root values. Static sea norm includes fullPH90/all65: three primitive classes have loads1,1,63. Only the accepted rootRECT static q0 slow lower is used, never positivity throughout3R.',
 'scope':'Forces and Riccati derivatives evaluated POINTWISE at accepted rootRECT/mean, with root static pencil lower as the separate hypothesis. Fixed f_A(root), fixed Xi_i(root), centred cubic P are not replaced by adiabatic f_A(slow). Positive local germ follows from continuity; the explicit germ radius below follows Weyl on real s and is not a new premise. Complex matter/Riesz jets themselves are uniform on all5760 support+3R. Parent owns whole Gaussian/cubic/Haar/source domain. No SOURCE/ground/pole/cone/current admission.'}

def run(path: Path) -> dict[str,Any]:
 mp.mp.dps=120;mp.iv.dps=120;sys.set_int_max_str_digits(0)
 raw=path.read_bytes();jet=json.loads(raw)
 inputs=jet['input_capsule'];src=next(json.loads(v['bytes_utf8']) for k,v in inputs.items() if k.endswith('resumed_actual_receipt.json'))
 quantum=next(json.loads(v['bytes_utf8']) for k,v in inputs.items() if k.endswith('gauss_quantum_final_actual_receipt.json'))
 pars=src['input_descriptor']['exact_NEW_allq_BO_mean']['parameters'];gmin=min(F(pars[k]['lower']) for k in ('g0','gs'))
 slow=F(quantum['actual_full_field_covariance']['certified_q0_slow_positive_lower']['lower'])
 k=[F(v['upper']) for v in jet['all_species_unscaled_K0_K1_K2_K3']]
 margin=F(jet['domain']['complex_Cauchy_margin']);tube=F(jet['domain']['complex_tube'])
 germ=min(margin,slow/(4*(2*k[1]+4096/margin)))
 iv:Any=mp.iv
 def ex(x: F) -> Any:return iv.mpf(x.numerator)/iv.mpf(x.denominator)
 def record(x:Any) -> dict[str,str]:
  lo=F(x.a._mpi_[0][1])*F(2)**x.a._mpi_[0][2] if x.a._mpi_[0][0]==0 else -F(x.a._mpi_[0][1])*F(2)**x.a._mpi_[0][2]
  hi=F(x.b._mpi_[1][1])*F(2)**x.b._mpi_[1][2] if x.b._mpi_[1][0]==0 else -F(x.b._mpi_[1][1])*F(2)**x.b._mpi_[1][2]
  return {'lower':str(lo),'upper':str(hi),'upper_decimal':str(mp.nstr(mp.mpf(hi.numerator)/hi.denominator,30))}
 h=gmin/2-tube;gm=ex(F(3,10));gM=ex(8+F(3,10)+12/h);g1=ex(12*(2+tube)/(4*h*h))
 prim0=ex(F(4096));prim1=prim0/ex(margin)
 matter0=ex(k[0])*2;matter1=ex(k[1])*2
 deriv=lambda k0,k1:gM*k1+iv.sqrt(gM/gm)*g1*k0
 bs=gm*ex(slow)/2;bf=gm*ex(gmin/20)/2
 s1=deriv(prim0+matter0,prim1+matter1)
 f01=deriv(prim0,prim1)
 delta0=gM*matter0;delta1=deriv(matter0,matter1)
 c0=delta0;c1=delta1
 ratio=c0*c0/(bs*bf)*ex(F(2)**-2048)
 if ratio.b>=iv.mpf('0.5'):raise ArithmeticError('Physical Schur margin fails')
 ns,nf=45,5715
 slowforce=ex(F(ns,2))*s1/iv.sqrt(bs)
 fastforce=ex(F(nf,2))*delta1/iv.sqrt(bf)+ex(F(nf,4))*(delta0+c0*c0/bs)*f01/bf**iv.mpf('1.5')
 crossforce=ex(F(ns))*c0*c1/(iv.sqrt(bs)*bf)
 total=slowforce*ex(F(2)**2048)+fastforce*ex(F(2)**1024)+crossforce
 gaussian=jet['full_physical_Gaussian_jet_consumer']['primitive_input_bounds']
 p=F(gaussian['corrector_L2']['upper'])
 trc=F(gaussian['full_trace_covariance']['upper'])
 radius=F(jet['domain']['support'])
 normalizer_lower=1-24*trc/radius**2-4*p
 if normalizer_lower<=F(1,2):raise ArithmeticError('Compact corrected source normalization bound not licensed')
 slowcov=ex(F(8))/iv.sqrt(bs)
 source_multiplier=iv.sqrt(slowcov)*(1+9*ex(p))
 # A legal scalar Gaussian G=8,K=1/4 has STD^4=G/(4K)=8,
 # rejecting a unit-STD assumption without claiming an actual ZPE violation.
 scalar_std_fourth=F(8)/(4*F(1,4))
 if scalar_std_fourth<=1:raise ArithmeticError('Wrong unit-STD control was blind')
 budgets={
  'slow_force_bumped_corrected_Gaussian_action_upper':record(slowforce*ex(F(2)**-1024)*source_multiplier),
  'matter_fast_force_bumped_corrected_Gaussian_action_upper':record(fastforce*ex(F(2)**-2048)*source_multiplier),
  'mixed_cross_force_bumped_corrected_Gaussian_action_upper':record(crossforce*ex(F(2)**-3072)*source_multiplier),
  'total_force_bumped_corrected_Gaussian_action_upper':record(total*ex(F(2)**-3072)*source_multiplier),
  'prepared_total_force_action_upper':record(2*total*ex(F(2)**-3072)*source_multiplier),
  'mean_shift_omitted_coefficients':record(total/ex(F(128)*F(2)**8192))}
 if path.read_bytes()!=raw or Path(__file__).read_bytes()!=SELF:raise ArithmeticError('Immutable consumer epoch changed')
 return {
  'status':'DERIVED_EXPLICIT_PH90_BLOCK_FORCE_CONSUMER_AT_ROOT_RECT',
  'proof':PROOF,'pointwise_positive_real_germ_radius':str(germ),
  'requires_3R_slow_pencil_positivity':False,
  'inputs':{'jet_path':str(path),'jet_sha256':hashlib.sha256(raw).hexdigest(),'own_sha256':hashlib.sha256(SELF).hexdigest(),'own_bytes_utf8':SELF.decode()},
  'outward_constants':{k:record(v) for k,v in {
   'gmin':gm,'Gmax':gM,'G1':g1,'bs':bs,'bf':bf,'Bs1':s1,'Bf01':f01,
   'deltaf0':delta0,'deltaf1':delta1,'Bpc0':c0,'Bpc1':c1,'Schur_ratio':ratio,
   'slow_force_coefficient':slowforce,'fast_force_coefficient':fastforce,
   'mixed_force_coefficient':crossforce,'slow_covariance_scaled_upper':slowcov,
   'force_to_bumped_corrected_Gaussian_multiplier':source_multiplier}.items()},
  'actual_action_input_budgets':budgets,
  'source_preparation':{'normalizer_lower_above_half':normalizer_lower>F(1,2),'normalization_norm_upper':'2','corrector_L2_upper':str(p),
   'budget_semantics':'Component/total bumped-corrected Gaussian actions are BEFORE normalization. Prepared total includes normalization upper2. Mean shift is a gradient bound, not a source action. No actual source/normalization change.'},
  'controls':{'scalar_G':'8','scalar_K':'1/4','scalar_STD_fourth_power':str(scalar_std_fourth),
   'wrong_unit_STD_bound_rejected':scalar_std_fourth>1,'scope':'Exact scalar Gaussian unit-conversion falsifier, NOT an actual N4 ZPE price violation.'},
  'generic_C_price_adopted':False,
  'runtime':{'executable':sys.executable,'python':sys.version},
  'arithmetic':'120-digit outward interval, exact rational endpoints'}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--jet',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 result=run(args.jet);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v['upper_decimal'] for k,v in result['actual_action_input_budgets'].items()},indent=2))
