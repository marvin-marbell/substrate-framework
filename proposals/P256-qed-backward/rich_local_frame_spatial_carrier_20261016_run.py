"""Actual full-H MPS images, matched background and two-way field intervention."""
from __future__ import annotations
import gc
import faulthandler
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np
from scipy.linalg import expm
from tenpy.networks.mps import MPS
from rich_local_frame_spatial_carrier_20261016_model import Model, PREFIX, physical_element
ROOT=Path(__file__).resolve().parent


def save(name,data):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(data,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def store(name,state):
    payload=dict(format='TENPY_1_1_1_FINITE_MPS_RAW_TENSORS',B=state._B,S=state._S,form=state.form,norm=state.norm,bc=state.bc,unit_cell_width=state.unit_cell_width,site_dimensions=[s.dim for s in state.sites])
    with (ROOT/(PREFIX+name+'.pkl')).open('wb') as f:pickle.dump(payload,f,protocol=5)


def load_state(name,model):
    with (ROOT/(PREFIX+name+'.pkl')).open('rb') as f:data=pickle.load(f)
    if data['site_dimensions']!=[s.dim for s in model.sites]:raise ValueError('State/operator physical dimensions differ')
    return MPS(model.sites,data['B'],data['S'],bc=data['bc'],form=data['form'],norm=data['norm'],unit_cell_width=data['unit_cell_width'])


def normalize(psi):
    raw=float(psi.overlap(psi).real)
    psi.canonical_form(renormalize=False);psi.norm/=np.sqrt(raw)
    return raw


def apply(H,psi,chi):
    image=psi.copy()
    err=H.apply(image,{'compression_method':'zip_up','trunc_params':{'chi_max':chi,'svd_min':1e-12}})
    if image.norm!=0:image.canonical_form(renormalize=False)
    record=dict(image_norm2=float(image.overlap(image).real),chi=max(image.chi))
    if hasattr(err,'pieces'):record.update(piece_images=err.pieces,svd_eps_sum=float(err.eps),meaning=err.meaning)
    else:record['svd_eps']=float(err.eps)
    return image,record


def step(model,psi,dt,chi):
    """Same full H, exact bounded-MPO sum; numerical splitting disclosed."""
    result=psi.copy()
    ledger=model.H.evolve(result,dt,chi)
    ledger['final_norm2']=normalize(result)
    return result,ledger


def outgoing(model,state):
    """Actual populated angular/field edge images; no global convergence claim."""
    rows=[]
    for v in range(4):
        O=model.g['O'][0][0].toarray()
        # Multiplication by O00 has exact squared norm integral; compression
        # cannot replace P O² P by (P O P)².
        f=model.g['F'];w=model.g['weights']
        # The quadrature's own O data came from genuine SU2 rotations. Recover
        # O00 values from fundamental D and evaluate their products directly.
        from rich_gauge_band_frame_20261016_model import jmat
        J=jmat(.5);P=[2*a for a in J]
        values=[]
        for a,b,c in model.g['points']:
            u=expm(-1j*a*J[2])@expm(-1j*b*J[1])@expm(-1j*c*J[2])
            values.append(np.trace(P[0]@u@P[0]@u.conj().T).real/2)
        exact=f.conj().T@(w[:,None]*np.asarray(values)[:,None]**2*f)
        leakage=(exact-O.conj().T@O);leakage=(leakage+leakage.conj().T)/2
        rows.append(float(physical_element(state,model.observable(model.gsites[v],leakage),state).real))
    # All eight Q image deficits see the unrepresented total-occupation shell.
    field=[]
    for a in range(8):
        leak=model.QQ[a,a]-model.Q[a]@model.Q[a]
        field.append(float(physical_element(state,model.observable(model.rfsite,np.kron(np.eye(model.nR),leak)),state).real))
    return dict(actual_local_O00_outgoing_norm2=rows,actual_field_Q_outgoing_norm2=field,scope='Actual-state individual raw multiplication images outside Galerkin regulator, NOT norm of total exterior H image, not a convergence bound. Full H images inside regulator are evolved; no occupied angular shell declared safe.')


def polarization(model,create):
    indices=np.flatnonzero(np.diag(model.geo['beta'])==(1 if create else -1))
    generator=-model.geo['spin'][2].T if create else model.geo['spin'][2]
    values,vectors=np.linalg.eigh(generator[np.ix_(indices,indices)])
    result=np.zeros(8,complex);result[indices]=vectors[:,-1]
    return result


def main():
    start=time.time()
    if '--trace-build' in sys.argv:faulthandler.dump_traceback_later(1200,repeat=False)
    if '--resume' in sys.argv:
        with (ROOT/(PREFIX+'model.pkl')).open('rb') as f:m=pickle.load(f)
    else:
        if '--resume-build' in sys.argv:m=Model.resume_build(ROOT/(PREFIX+'build_checkpoint.pkl'))
        else:m=Model()
        save('build',dict(parameters=m.parameters,signature=m.signature,positions=m.positions.tolist(),modes=m.modes.tolist(),Fock_modes=32,Fock_sector_dimensions={'N15':565722720,'N16':601080390,'N17':565722720},CAR_representation='All32 binary local CAR modes, explicit Jordan-Wigner, no virtual-configuration projection',local_g_labels=m.g['labels'],material_labels=m.material['labels'],field_basis=m.basis.tolist(),bounded_group_MPO_graph_bonds=m.H.chi,full_graph_storage_audit=m.graph_audit,operator_representation='Exact lazy sum of bounded-channel MPO groups; no full-MPO tensor allocation',terms=len(m.terms),time=time.time()-start))
        with (ROOT/(PREFIX+'model.pkl')).open('wb') as f:pickle.dump(m,f,protocol=5)
    print('MODEL_READY',m.signature,len(m.terms),max(m.H.chi),flush=True)
    if '--trace-build' in sys.argv:faulthandler.cancel_dump_traceback_later()
    if '--build-only' in sys.argv:return
    addpol=polarization(m,True);rempol=polarization(m,False)
    if '--evolve-prepared' in sys.argv:
        preparation=json.loads((ROOT/(PREFIX+'preparation_result.json')).read_text())
        states={name:load_state(name+'_t0',m) for name in preparation['absolute']}
        neutral=states['neutral'];addition=states['addition']
        addnorm=preparation['source_raw_norm2']['addition'];remnorm=preparation['source_raw_norm2']['removal'];unpolarized_norm=preparation['source_raw_norm2']['addition_unpolarized']
        add_quality=preparation['source_quality']['addition'];rem_quality=preparation['source_quality']['removal'];unpol_quality=preparation['source_quality']['addition_unpolarized']
    else:
        neutral=m.seed()
        addition,addnorm=m.source(neutral,0,addpol,True,chi=12);add_quality=dict(m.last_source_quality)
        removal,remnorm=m.source(neutral,0,rempol,False,chi=12);rem_quality=dict(m.last_source_quality)
        unpolarized,unpolarized_norm=m.source(neutral,0,0,True,chi=12);unpol_quality=dict(m.last_source_quality)
        kicked=addition.copy();kick=expm(.28j*np.kron(np.eye(m.nR),m.Q[0]));kicked.apply_local_op(m.rfsite,m.register(m.rfsite,kick),unitary=True,renormalize=False);normalize(kicked)
        states={'neutral':neutral,'addition':addition,'removal':removal,'addition_field_kick':kicked,'addition_unpolarized':unpolarized}
    result=dict(kind='ACTUAL_FOUR_LOCAL_SEWN_CAR32_AND_EIGHT_FINITE_Q_FIELD_SAME_H',law=m.signature,architecture='Physical charged32 microsites x_s(R)=R xbody_s; not lab-fixed texture alternative',source_raw_norm2={'addition':addnorm,'removal':remnorm},times=[0.,.04,.08],trajectories={},limits=['Finite g j<=2, material l<=2, eight-field total occupation<=1, r3 Friedrichs grid, finite chi and normalized RK2; no stationary ground, pole, photon, inertial mass or global angular convergence claim.','Whole-CAR32 representation permits all N15/N16/N17 virtual configurations; finite chi compresses entanglement and errors are measured, not projected-source evolution.','Scalar edge current is distinct from physical lab-mode derivative, which includes all microscopic paths and rotating ions.'])
    result['source_polarizations']={name:[[float(z.real),float(z.imag)] for z in vector] for name,vector in [('addition',addpol),('removal',rempol)]}
    result['operator_cache_reused']='--resume' in sys.argv
    result['source_raw_norm2']['addition_unpolarized']=unpolarized_norm
    result['unpolarized_control']='Original cube-index source: legitimate halfspin/multiplicity-entangled preparation, not a failed spin state or a Hamiltonian repair.'
    result['source_quality']={'addition':add_quality,'removal':rem_quality,'addition_unpolarized':unpol_quality}
    if '--evolve-prepared' not in sys.argv:
        preparation=dict(law=m.signature,source_raw_norm2=result['source_raw_norm2'],source_quality=result['source_quality'],source_polarizations=result['source_polarizations'],absolute={name:m.measure(psi,energy=False) for name,psi in states.items()},scope='Actual32CAR/fourlocalg/r/R/eightfield preparation only; NO transport/evolution claim.')
        for name,psi in states.items():store(name+'_t0',psi)
        save('preparation_result',preparation);print('PREPARATION_COMPLETE',json.dumps(preparation),flush=True)
    if '--prepare-only' in sys.argv:return
    for name,psi in states.items():
        measurements=[m.measure(psi)];ledgers=[]
        for k in range(2):
            psi,ledger=step(m,psi,.04,12);ledgers.append(ledger);measurements.append(m.measure(psi));store(name+'_t'+str(k+1),psi)
            print('EVOLVED',name,k+1,json.dumps(measurements[-1]),flush=True)
        result['trajectories'][name]=dict(measurements=measurements,steps=ledgers,outgoing=outgoing(m,psi));save('result_partial',result)
        states[name]=psi
    for name in ('addition','removal','addition_unpolarized'):
        result['trajectories'][name]['excess_matched_background']=[{key:(np.asarray(a[key])-np.asarray(b[key])).tolist() for key in ('n_cells','local_LEFT','local_r2','field_Q')} for a,b in zip(result['trajectories'][name]['measurements'],result['trajectories']['neutral']['measurements'])]
    result['field_to_carrier']=[{key:(np.asarray(a[key])-np.asarray(b[key])).tolist() for key in ('n_cells','local_LEFT','local_r2','field_Q')} for a,b in zip(result['trajectories']['addition_field_kick']['measurements'],result['trajectories']['addition']['measurements'])]
    result['carrier_to_field']=result['trajectories']['addition']['excess_matched_background']
    # Independent bond increase on the same preparation and fullH image, not
    # differences between independently optimized states.
    fine,_=m.source(neutral,0,addpol,True,chi=20);coarse,_=step(m,fine,.04,12);fine,ledger=step(m,fine,.04,20)
    result['same_input_chi12_20']=dict(distance_squared=float(coarse.overlap(coarse).real+fine.overlap(fine).real-2*coarse.overlap(fine).real),fine_ledger=ledger)
    store('chi20_addition_t1',fine)
    half=addition.copy();half_ledgers=[]
    for k in range(2):
        half,small_ledger=step(m,half,.02,12);half_ledgers.append(small_ledger)
    coarse_dt=load_state('addition_t1',m)
    coarse_measure=m.measure(coarse_dt,energy=False);half_measure=m.measure(half,energy=False)
    overlap=coarse_dt.overlap(half)
    result['same_input_dt004_002']=dict(chi=12,coarse_dt=.04,fine_dt=.02,time=.04,physical_observable_differences={key:(np.asarray(half_measure[key])-np.asarray(coarse_measure[key])).tolist() for key in ('n_cells','local_LEFT','local_r2','field_Q')},raw_phase_restored_distance_squared=float(2-2*overlap.real),phase_aligned_distance_squared=float(2-2*abs(overlap)),half_step_ledgers=half_ledgers,meaning='Short actual local charge/LEFT/field dt/half-dt discriminator, including splitting plus compression uncertainty; not norm/energy-only or a convergence theorem.')
    store('dt002_addition_t1',half)
    # Physical nonforward lab current/contact, not the scalar-edge derivative.
    result['physical_lab_sources']={}
    for a in (0,4):
        source=m.lab_source(a);contact=m.lab_source(a,a)
        result['physical_lab_sources'][str(a)]={name:dict(current=float(physical_element(psi,source,psi).real),contact=float(physical_element(psi,contact,psi).real)) for name,psi in states.items()}
        del source,contact;gc.collect()
    result['elapsed_seconds']=time.time()-start;save('result',result);print('SCIENCE_COMPLETE',flush=True)


if __name__=='__main__':main()
