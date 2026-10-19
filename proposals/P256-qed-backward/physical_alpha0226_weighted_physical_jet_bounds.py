#!/usr/bin/env python3
"""Derivative-aware source Stieltjes ENERGY probe under frozen ORIGINAL Alpha0226.

For B=A+tI, z=Q(Q*BQ)^-1 Q*m, r=m-Bz, E=m*B^-1m-m*z=r*B^-1r.
Differentiation of this identity (Q fixed) gives
 E_xy=2Re<r_xy,Rr>+2Re<r_x,Rr_y>
      -2Re<r_x,R A_y Rr>-2Re<r_y,R A_x Rr>
      +<r,(R A_y R A_x R+R A_x R A_y R-R A_xy R)r>.
With d_j=||r_j||/sqrt(t), k_j=||A_j||/t, its absolute bound is
 2d_xy d+2d_x d_y+2k_y d_x d+2k_x d_y d+(2k_x k_y+k_xy)d^2.
This sees a first residual jet even when r=0. Integrating E or its jets is
legitimate only with integrable majorants for EVERY jet and both endpoints.
No finite shifted probe is a full log or physical-current certificate.

The repair enriches Galerkin with whole-callback first/mixed residual jets,
not just base Krylov. Physical chart transport and pulled source
are separately accounted. Full physical transport terms are bounded, not
silently identified with this chart. Numerical bounds are not directed-rounded.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, cast
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import physical_alpha0226_midpoint_operator as core
import physical_alpha0226_source_resolvent as resolvent
import physical_alpha0226_physical_log_jets as logjets
import physical_alpha0226_source_receiver as receiver

ROOT = Path(__file__).parent
JET_KEYS = ('', 'x', 'y', 'xy')
EXPECTED = {'physical_alpha0226_midpoint_operator.py':'80045bd63ac3eba197f183793ead16323980810fd53dfb62cdbd9129b71dde91',
 'physical_alpha0226_physical_log_jets.py':'2c638095122d3d0fdc8d90d81f19cd3e858381d1db19e64db406958071ae3597',
 'physical_alpha0226_source_receiver.py':'080143bc0e23c31f5da04e309f886918fed270801f63abab2a0cd5a1190597cc',
 'physical_alpha0226_source_resolvent.py':'a363dd6103b0c0a4d5a94048c09f4706625b82d62d5241dcfe0fac016419b713'}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pairing(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.vdot(a,b).real)

def assignments(key: str) -> list[tuple[str,str,str]]:
    result=[]
    for positions in itertools.product(range(3), repeat=len(key)):
        suffix=['','','']
        for letter, position in zip(key,positions,strict=True):
            suffix[position]+=letter
        result.append(tuple(suffix))
    return result

class GramJetAction:
    """Exact tensor product rule for COMPLETE P TdagT P and ALL jets."""
    def __init__(self, model: core.Alpha0226Midpoint):
        self.model=model
        self.grams=[]
        self.sparsity=[]
        self.norm_upper={key:0. for key in JET_KEYS}
        self.cross_upper={key:0. for key in ('x','y','xy')}
        for parts,w in zip(model.factors,model.weights,strict=True):
            node=[]
            sn=[]
            gn=[]
            for part in parts:
                s={key:part['SF'+key] for key in JET_KEYS}
                g={'':s[''].conj().T@s[''],
                   'x':s['x'].conj().T@s['']+s[''].conj().T@s['x'],
                   'y':s['y'].conj().T@s['']+s[''].conj().T@s['y'],
                   'xy':s['xy'].conj().T@s['']+s['x'].conj().T@s['y']+s['y'].conj().T@s['x']+s[''].conj().T@s['xy']}
                node.append({k:scipy.sparse.csr_matrix(v) for k,v in g.items()})
                self.sparsity.append({k:float(np.count_nonzero(v)/v.size) for k,v in g.items()})
                gn.append({k:float(np.linalg.norm(v,2)) for k,v in g.items()})
                sn.append({k:float(np.linalg.norm(v,2)) for k,v in s.items()})
            self.grams.append(node)
            for key in JET_KEYS:
                self.norm_upper[key]+=float(w)*sum(float(np.prod([gn[j][k] for j,k in enumerate(row)])) for row in assignments(key))
            base=float(np.prod([s[''] for s in sn]))
            for key in ('x','y','xy'):
                right=sum(float(np.prod([sn[j][k] for j,k in enumerate(row)])) for row in assignments(key))
                self.cross_upper[key]+=float(w)*base*right
    def apply(self, v: np.ndarray, key: str='') -> np.ndarray:
        full=self.model.embed(v);total=np.zeros_like(full)
        for node,w in zip(self.grams,self.model.weights,strict=True):
            for suffix in assignments(key):
                total+=w*resolvent.tensor_product(cast(Any,node[0][suffix[0]]),cast(Any,node[1][suffix[1]]),cast(Any,node[2][suffix[2]]),full)
        return self.model.pack(total)


def energy_bound(residual: dict[str,np.ndarray], upper: dict[str,float], shift: float) -> dict[str,Any]:
    d={k:float(np.linalg.norm(v))/np.sqrt(shift) for k,v in residual.items()}
    k={j:upper[j]/shift for j in ('x','y','xy')}
    terms={'second_residual':2*d['xy']*d[''], 'BOTH_first_residuals':2*d['x']*d['y'],
           'Ay_cross':2*k['y']*d['x']*d[''], 'Ax_cross':2*k['x']*d['y']*d[''],
           'BOTH_ordered_operator_products':2*k['x']*k['y']*d['']**2,
           'operator_contact':k['xy']*d['']**2}
    return {'energy_upper':d['']**2,'first_x_upper':2*d['x']*d['']+k['x']*d['']**2,
            'first_y_upper':2*d['y']*d['']+k['y']*d['']**2,
            'mixed_absolute_upper':sum(terms.values()), 'mixed_terms':terms,'weighted_residual_jets':d,
            'scope':'Exact-arithmetic majorant using t>0, not a hidden spectral floor; float residual/norm evaluation not interval certified.'}


def galerkin(q: np.ndarray, aq: np.ndarray, source: dict, shift: float,
             action: GramJetAction) -> dict:
    """Exact differentiated Galerkin via only necessary FOUR-Ref actions."""
    a=logjets.hermitian(q.conj().T@aq)
    solve=lambda rhs:np.linalg.solve(a+shift*np.eye(q.shape[1]),q.conj().T@rhs)
    u={'':solve(source[''])};z={'':q@u['']}
    axz=action.apply(z[''],'x');ayz=action.apply(z[''],'y')
    u['x']=solve(source['x']-axz);u['y']=solve(source['y']-ayz)
    z['x']=q@u['x'];z['y']=q@u['y']
    axyz=action.apply(z[''],'xy');axzy=action.apply(z['y'],'x');ayzx=action.apply(z['x'],'y')
    rhsxy=source['xy']-axyz-axzy-ayzx
    u['xy']=solve(rhsxy);z['xy']=q@u['xy']
    r={'':source['']-aq@u['']-shift*z[''],
       'x':source['x']-axz-aq@u['x']-shift*z['x'],
       'y':source['y']-ayz-aq@u['y']-shift*z['y'],
       'xy':rhsxy-aq@u['xy']-shift*z['xy']}
    value={ '':pairing(source[''],z['']),
            'x':pairing(source['x'],z[''])+pairing(source[''],z['x']),
            'y':pairing(source['y'],z[''])+pairing(source[''],z['y']),
            'xy':pairing(source['xy'],z[''])+pairing(source['x'],z['y'])+pairing(source['y'],z['x'])+pairing(source[''],z['xy'])}
    wrong=source['xy']-aq@u['xy']-shift*z['xy']
    return {'z':z,'r':r,'value':value,'omitted_operator_contact_and_first_jets_residual':wrong}


def physical_accounting(z: Any, source: dict, alpha: float, upper: dict, cross: dict) -> dict:
    a,ax,ay,axy=(z[k] for k in ('A','Ax','Ay','Axy'))
    physical=logjets.physical_range_jets(a,z['Cx'],z['Cy'],z['Cxy'],z['Dxy'],ax,ay,axy)
    pulled=receiver.pulled_original_source_jets(a,ax,ay,axy,{k:z['Q'].conj().T@v for k,v in source.items()},physical)
    h,hx,hy,hxy=(physical[k] for k in ('H','canonical_x','canonical_y','canonical_xy'))
    ox,oy=physical['Omega_x'],physical['Omega_y']
    sigma=physical['Pancharatnam_Qxy']-(ox@oy+oy@ox)/2
    comm=lambda x,y:x@y-y@x
    terms={'canonical_hxy':hxy,'Omega_x_canonical_hy':comm(ox,hy),'Omega_y_canonical_hx':comm(oy,hx),
           'sigma_xy_H':comm(sigma,h),'BOTH_double_commutators':(comm(ox,comm(oy,h))+comm(oy,comm(ox,h)))/2}
    first={'canonical_hx':hx,'Omega_x_H':comm(ox,h),'canonical_hy':hy,'Omega_y_H':comm(oy,h)}
    p=pulled[''];den=pairing(p,p)
    chart={'mixed_channels':{k:{'norm':float(np.linalg.norm(v,2)),'source_weighted':pairing(p,v@p)/den} for k,v in terms.items()},
           'first_channels':{k:{'norm':float(np.linalg.norm(v,2)),'source_weighted':pairing(p,v@p)/den} for k,v in first.items()},
           'Omega_x_norm':float(np.linalg.norm(ox,2)),'Omega_y_norm':float(np.linalg.norm(oy,2)),
           'sigma_xy_norm':float(np.linalg.norm(sigma,2)),
           'pulled_source_jet_norms':{k:float(np.linalg.norm(v)) for k,v in pulled.items()},
           'wrong_omit_all_transport_contact_norm':float(np.linalg.norm(physical['physical_xy']-hxy,2)),
           'wrong_omit_sigma_weighted_contact':pairing(p,terms['sigma_xy_H']@p)/den,
           'mixed_channel_sum_defect':float(np.linalg.norm(sum(terms.values())-physical['physical_xy'])),
           'scope':'Actual persisted original finite immersion chart only; no fullphysical operator retagging.'}
    bx,by,bxy=(upper[k] for k in ('x','y','xy'));beta=upper['']
    hbound=max(abs(np.log(alpha)),abs(np.log(beta)))/.01
    hxb=bx/(.01*alpha);hyb=by/(.01*alpha);hxyb=bxy/(.01*alpha)+bx*by/(.01*alpha**2)
    oxb=(cross['x']+bx/2)/alpha;oyb=(cross['y']+by/2)/alpha
    sigmab=cross['xy']/alpha+(cross['x']*by+cross['y']*bx)/(2*alpha**2)+np.sqrt(beta)*(bxy/(2*alpha**1.5)+3*bx*by/(4*alpha**2.5))
    inv=alpha**-.5;ix=bx/(2*alpha**1.5);iy=by/(2*alpha**1.5);ixy=bxy/(2*alpha**1.5)+3*bx*by/(4*alpha**2.5)
    mn={k:float(np.linalg.norm(v)) for k,v in source.items()}
    nb=inv*mn[''];nxb=ix*mn['']+inv*mn['x'];nyb=iy*mn['']+inv*mn['y']
    nxyb=ixy*mn['']+ix*mn['y']+iy*mn['x']+inv*mn['xy']
    full={'H':hbound,'canonical_hx':hxb,'canonical_hy':hyb,'canonical_hxy':hxyb,
          'Omega_x':oxb,'Omega_y':oyb,'sigma_xy':sigmab,
          'Omega_x_H':2*oxb*hbound,'Omega_y_H':2*oyb*hbound,
          'Omega_x_hy':2*oxb*hyb,'Omega_y_hx':2*oyb*hxb,'sigma_xy_H':2*sigmab*hbound,
          'BOTH_double_commutators':4*oxb*oyb*hbound,
          'pulled_source':nb,'pulled_source_x':nxb+oxb*nb,'pulled_source_y':nyb+oyb*nb,
          'pulled_source_xy':nxyb+oxb*nyb+oyb*nxb+(sigmab+oxb*oyb)*nb}
    return {'chart':chart,'full_original_conditional_norm_upper':full,
            'full_bound_hypotheses':'Frozen byte-positive complete A>=alpha I; actual tensor norm uppers; exact arithmetic Sylvester/Frechet bounds. Float callbacks/norms not outward-rounded.',
            'physical_certificate':False,'unpaid':'Fullnorm estimates are far too large to price chart-complement physical contact or positive source; finite shift source-energy advance below does not remove this debt.'}


def abstract_control() -> dict:
    delta=1e-12;t=.01
    # At k=0 Q=e1, r=0, r_x=-e2/(1+t), r_xy=0.
    rx=np.array([[0.],[-1/(1+t)]],complex)
    residual={'':np.zeros_like(rx),'x':rx,'y':rx,'xy':np.zeros_like(rx)}
    bound=energy_bound(residual,{'x':1.,'y':1.,'xy':0.},t)
    exact=2/((1+t)**2*(delta+t))
    return {'delta':delta,'shift':t,'base_energy':0.,'full_minus_chart_mixed_resolvent':exact,
            'derivative_energy_bound':bound,'wrong_base_only_contact_bound':0.,
            'integrated_log_contact_magnitude':2*(-np.log(delta)-1+delta)/(1-delta)**2,
            'scope':'Analytic control A=[[1,k],[k,delta]], b=e1, same first direction twice; no Alpha substitution.'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--shifts',type=float,nargs='+',default=[.01,.001]);args=parser.parse_args()
    start=time.time();own_bytes=Path(__file__).read_bytes()
    inputs=list(EXPECTED)+['physical_alpha0226_source_resolvent_m64_final_20261009_capsule.json','physical_alpha0226_source_resolvent_m64_final_20261009_capsule.npz',
        'physical_alpha0226_inverse_adapted_source_family_capsule.json','physical_alpha0226_inverse_adapted_source_point_capsule.json','physical_alpha0226_inverse_adapted_source_point_capsule.npz',
        'physical_alpha0226_complete_integer_source_response_capsule.json']
    inputs.append('physical_alpha0226_weighted_physical_jet_sparse_timeout_receipt.json')
    hashes={name:sha(ROOT/name) for name in inputs}
    for name,value in EXPECTED.items():
        if hashes[name]!=value:raise ArithmeticError('Frozen supplier changed: '+name)
    z=np.load(ROOT/'physical_alpha0226_inverse_adapted_source_point_capsule.npz')
    source={k:z['source'+k] for k in JET_KEYS}
    model=core.Alpha0226Midpoint(order=64,jets=True)
    if np.linalg.norm(model.source_mid-source[''])>1e-14:raise ArithmeticError('Original source bytes mismatch')
    action=GramJetAction(model)
    witness=json.loads((ROOT/'physical_alpha0226_weighted_physical_jet_sparse_timeout_receipt.json').read_text())
    action_hash=hashlib.sha256(inspect.getsource(GramJetAction).strip().encode()).hexdigest()
    if action_hash!=witness['GramJetAction_source_sha256']:raise ArithmeticError('Interrupted callback action bytes changed')
    if any(hashes[name]!=value for name,value in witness['input_sha256'].items()):raise ArithmeticError('Interrupted callback input bytes changed')
    equivalence=witness['observed_completed_component']['alljet_callback_equivalence_relative']
    q=z['Q'];aq=z['AQ']
    print(json.dumps({'resumed_component_witness':sha(ROOT/'physical_alpha0226_weighted_physical_jet_sparse_timeout_receipt.json'),
                      'elapsed_seconds':time.time()-start}),flush=True)
    cases=[];vectors={}
    for shift in args.shifts:
        if shift<=0:raise ValueError('Stieltjes probe requires actual t>0, not a floor')
        old=galerkin(q,aq,source,shift,action)
        print(json.dumps({'shift':shift,'baseline_residual_jet_norms':{k:float(np.linalg.norm(v)) for k,v in old['r'].items()}}),flush=True)
        enrichment=[old['r'][key]/np.linalg.norm(old['r'][key],axis=0) for key in ('x','y','xy')]
        packed=np.column_stack([q,*enrichment])
        repaired_q,rr,pivot=cast(Any,scipy.linalg.qr)(packed,mode='economic',pivoting=True)
        diag=np.abs(np.diag(rr));rank=int(np.sum(diag>diag[0]*2e-13));repaired_q=repaired_q[:,:rank]
        repaired_aq=action.apply(repaired_q)
        repaired=galerkin(repaired_q,repaired_aq,source,shift,action)
        old_bound=energy_bound(old['r'],action.norm_upper,shift)
        new_bound=energy_bound(repaired['r'],action.norm_upper,shift)
        difference={k:repaired['value'][k]-old['value'][k] for k in JET_KEYS}
        # This bound compares two approximate charts through the SAME full scalar.
        pair_bound={k:old_bound[k]+new_bound[k] for k in ('energy_upper','first_x_upper','first_y_upper','mixed_absolute_upper')}
        actual_wrong=float(np.linalg.norm(repaired['omitted_operator_contact_and_first_jets_residual']-repaired['r']['xy']))
        cases.append({'shift':shift,'old_rank':q.shape[1],'derivative_enriched_rank':rank,'qr_pivot':pivot.tolist(),
          'old_source_response_jets':old['value'],'repaired_source_response_jets':repaired['value'],'observed_repaired_minus_old_jets':difference,
          'old_fullcallback_energy_jet_bound':old_bound,'repaired_fullcallback_energy_jet_bound':new_bound,'two_chart_difference_bounds':pair_bound,
          'omitted_Axy_and_BOTH_Afirst_solution_jets_control':{'residual_difference_norm':actual_wrong,
              'wrong_contact_residual_norm':float(np.linalg.norm(repaired['omitted_operator_contact_and_first_jets_residual'])),
              'complete_contact_residual_norm':float(np.linalg.norm(repaired['r']['xy']))},
          'derivative_enrichment':'Fixed Q extended by ALL four-Ref actual residual x/y/xy columns; one whole A Q action; exactly differentiated Galerkin using cached necessary four-Ref whole operator-jet actions.',
          'noninteger_Stieltjes_interpretation':'These are pointwise resolvent integrand jets. No integral endpoint/quadrature/log/current certificate.'})
        vectors['shift_'+str(shift).replace('.','p')+'_Q']=repaired_q
        for key,value in repaired['r'].items():vectors['shift_'+str(shift).replace('.','p')+'_residual'+key]=value
        print(json.dumps({'shift':shift,'rank':rank,'old_contact_bound':old_bound['mixed_absolute_upper'],'new_contact_bound':new_bound['mixed_absolute_upper'],'omitted_jet_control':actual_wrong}),flush=True)
    alpha=float(json.loads((ROOT/'physical_alpha0226_source_resolvent_m64_final_20261009_capsule.json').read_text())['represented_Gram_lower_bound']['exact_byte_Gram_alpha_lower'])
    # Point jetsTrue alpha differs in last bits; choose the independently recovered smaller matching value.
    alpha=min(alpha,4.361197765882947e-21)
    physical=physical_accounting(z,source,alpha,action.norm_upper,action.cross_upper)
    output=Path(args.output);vector_path=output.with_suffix('.npz');np.savez_compressed(vector_path,**vectors)
    report={'derivation':__doc__,'plan':{'id':'issue-222-induced-photon-delta13-continuation-20261008','work_item':'shared-compton-transfer','pinned_revision':'2de065e033f83c33591752f5778876d9e28acce0578dda17e5295a0abe572e15'},
      'descriptor':model.descriptor(),'input_sha256':hashes,'actual_alljet_callback_equivalence_relative':equivalence,
      'callback_equivalence_scope':'Explicitly bound completed component observation from prior interrupted NEW probe. GramJetAction class/input bytes match; no rerun to confirm. Prior process was NOT natural0.',
      'actual_complete_operator_jet_norm_upper':action.norm_upper,'actual_complete_immersion_cross_norm_upper':action.cross_upper,
      'actual_factor_exact_zero_sparsity':action.sparsity,
      'abstract_zero_base_residual_contact_falsifier':abstract_control(),'actual_physical_chart_and_fullnorm_accounting':physical,'actual_weighted_jet_cases':cases,
      'vectors':{'path':str(vector_path),'sha256':sha(vector_path)},
      'nonclaims':['No source or covariance adoption, parameter fit, fixed point, source gate, spectral floor, Q-ground hypothesis or weak-native bound.',
       'Full original physical current/contact and noninteger source-response closure remain OPEN; positive source/range transport and low-shift Stieltjes endpoint estimates are unpaid.',
       'Integer total identities and finitechart FD agreement do not certify individual physical operators.',
       'All comparisons are finite M64 seed20261009 whole44336 callback derivatives; no continuousHaar, Phi/statistical/macro or on-shell Compton certificate.'],
      'execution':{'source_at_import_sha256':hashlib.sha256(own_bytes).hexdigest(),'source_bytes_at_import':own_bytes.decode(),
        'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'runtime':platform.platform(),
        'input_sha256':hashlib.sha256(json.dumps(model.descriptor(),sort_keys=True).encode()).hexdigest(),
        'elapsed_seconds':time.time()-start,'startup_unchanged':Path(__file__).read_bytes()==own_bytes and all(sha(ROOT/name)==value for name,value in hashes.items()),
        'natural_exit':'Wrapper must bind observed process returncode; receipt is written only after all new comparisons.'}}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'capsule':str(output),'capsule_sha256':sha(output),'elapsed_seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
