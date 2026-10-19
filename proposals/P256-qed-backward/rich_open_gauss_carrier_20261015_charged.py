"""Actual two signed Green branches and same-state charge/field/current motion."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.special import jv
from rich_open_gauss_carrier_20261015_model import NeutralCube, maximum, commutator
ROOT=Path(__file__).resolve().parent
P='rich_open_gauss_carrier_20261015_'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def pair(z):
    return [float(np.real(z)),float(np.imag(z))]
def chebyshev(H,psi,t):
    diag=H.diagonal()
    radius=np.asarray(abs(H).sum(axis=1)).ravel()-abs(diag)
    low=float(np.min(diag-radius));high=float(np.max(diag+radius))
    center=(low+high)/2;scale=(high-low)/2
    A=(H-sparse.eye(H.shape[0])*center)/scale
    degree=int(np.ceil(scale*abs(t)))+100
    a=psi.astype(complex);b=A@a
    out=jv(0,scale*t)*a+2*(-1j)*jv(1,scale*t)*b
    for k in range(2,degree+1):
        c=2*(A@b)-a
        out+=2*((-1j)**k)*jv(k,scale*t)*c
        a,b=b,c
    return np.exp(-1j*center*t)*out,dict(gershgorin=[low,high],degree=degree,bessel_tail_estimate=float(2*np.sum(np.abs(jv(np.arange(degree+1,degree+201),scale*t)))))
def main():
    start=time.monotonic()
    config=json.loads((ROOT/(P+'input.json')).read_text())
    neutral=NeutralCube()
    data=np.load(ROOT/(P+'neutral_reference.npz'))
    if str(data['basis_SHA256'])!=neutral.basis_hash or str(data['model_SHA256'])!=sha(ROOT/(P+'model.py')):
        raise ValueError('Reference basis/model mismatch')
    omega=data['psi'];energy=float(data['energy'])
    times=np.asarray(config['times'])
    results={};states={'reference':omega};early={}
    for label,up,creation in [('addition',5,True),('removal',3,False)]:
        model=NeutralCube(up,4)
        sources=[neutral.charged_source(model,site,0,creation) for site in range(8)]
        raw=np.column_stack([s@omega for s in sources])
        norms=np.sum(abs(raw)**2,axis=0)
        prepared=raw[:,0]/np.sqrt(norms[0])
        trajectory=expm_multiply(-1j*model.H,prepared,start=0,stop=float(times[-1]),num=len(times),traceA=-1j*float(model.H.diagonal().sum()))
        records=[]
        for t,state in zip(times,trajectory):
            obs=model.measure(state)
            amplitude=raw.T.conj()@state*np.sqrt(norms[0])*np.exp(1j*energy*t)
            # Addition positive-frequency exp(-i gap t); removal negative-frequency exp(+i gap t).
            if not creation:
                amplitude=amplitude.conj()
            obs.update(time=float(t),signed_green_from_anchor=[pair(z) for z in amplitude])
            records.append(obs)
        initial=records[0]
        for obs in records:
            obs['delta_right_charge_own_t0']=obs['right_charge']-initial['right_charge']
            obs['delta_right_electric_energy_own_t0']=obs['right_electric_energy']-initial['right_electric_energy']
            obs['delta_flux_own_t0']=(np.asarray(obs['electric_flux'])-initial['electric_flux']).tolist()
        early[label]=dict(dimension=model.dimension,raw_source_norm_squared=norms.tolist(),initial=initial,endpoint=records[-1],reference_energy=energy,conditional_charge_semantics='All changes relative to this normalized charged source own t0, not neutral reference; preexisting entangled remote charge and offsite Green retained')
        (ROOT/(P+'provisional_synthesis.json')).write_text(json.dumps(dict(kind='FIRST_ACTUAL_SIGNED_CHARGED_SOURCE_MOTION',branches=early,scope='Finite charged field candidate, no physical Noether embedding or particle mass'),indent=2)+'\n')
        controls={}
        for name,H in [('path_cut',model.Hcut),('hopping_off',model.Hoff)]:
            control=expm_multiply(-1j*float(times[-1])*H,prepared,traceA=-1j*float(times[-1])*float(H.diagonal().sum()))
            obs=model.measure(control)
            QR=sparse.diags(np.sum(model.rho[:,model.right_sites],axis=1))
            controls[name]=dict(endpoint=obs,delta_right_charge_own_t0=obs['right_charge']-initial['right_charge'],delta_right_electric_energy_own_t0=obs['right_electric_energy']-initial['right_electric_energy'],right_charge_conservation_commutator=maximum(commutator(H,QR)),full_minus_control_right_charge=records[-1]['right_charge']-obs['right_charge'],same_prepared_state=True,energy_observable_scope='endpoint energy/right_owned_energy above are full-H diagnostics, NOT control conserved own energy; own control energy below',own_energy_initial=float(np.vdot(prepared,H@prepared).real),own_energy_endpoint=float(np.vdot(control,H@control).real))
            states[label+'_'+name+'_endpoint']=control
        independent,resolution=chebyshev(model.H,prepared,float(times[-1]))
        resolution['endpoint_state_L2_difference']=float(np.linalg.norm(independent-trajectory[-1]))
        resolution['endpoint_right_charge_difference']=model.measure(independent)['right_charge']-records[-1]['right_charge']
        QR=sparse.diags(np.sum(model.rho[:,model.right_sites],axis=1))
        dQR=1j*commutator(model.H,QR)
        direct_rates=[float(np.vdot(s,dQR@s).real) for s in trajectory]
        fluxdelta=np.asarray(records[-1]['electric_flux'])-initial['electric_flux']
        rhodelta=np.asarray(records[-1]['rho'])-initial['rho']
        algebra=model.algebra()
        algebra['finite_time_gauss_change']=float(np.max(abs(model.full_incidence@fluxdelta+rhodelta)))
        algebra['source_adjoint_max']=max(maximum(neutral.charged_source(model,j,0,creation).T-model.charged_source(neutral,j,0,not creation)) for j in (0,4))
        results[label]=dict(dimension=model.dimension,basis_SHA256=model.basis_hash,H_nnz=model.H.nnz,raw_source_norm_squared=norms.tolist(),raw_gram=[[pair(z) for z in row] for row in raw.T.conj()@raw],samples=records,controls=controls,resolution=resolution,algebra=algebra,right_charge_derivative_from_commutator=direct_rates,max_norm_drift=max(abs(r['norm']-1) for r in records),max_energy_drift=max(abs(r['energy']-initial['energy']) for r in records),source_energy_minus_reference=initial['energy']-energy,boundary_charging_cost=1/32,loop_ReW_prepared=float(np.vdot(prepared,model.source@prepared).real))
        states[label+'_initial']=prepared;states[label+'_endpoint']=trajectory[-1];states[label+'_independent_endpoint']=independent
        states[label+'_basis_old_indices']=model.old_indices
        states[label+'_basis_fock']=model.fock
        states[label+'_basis_electric']=model.electric
        del trajectory,model,raw,sources
    anticomm=np.asarray(results['addition']['raw_source_norm_squared'])+np.asarray(results['removal']['raw_source_norm_squared'])
    output=dict(kind='ACTUAL_COMMON_H_BOTH_SIGNED_GREEN_AND_CHARGED_FIELD_MOTION',reference=dict(energy=energy,residual=float(data['residual']),SHA256=sha(ROOT/(P+'neutral_reference.npz')),basis_SHA256=neutral.basis_hash),branches=results,raw_local_dressed_anticommutator=anticomm.tolist(),spinor_scope='Both particle-number signs for same primitive up CAR component. Down component is spin-rotation partner under spin-independent H and singlet-seeded neutral reference; no computed physical-frame Noether isometry, no electron completion.',green_convention='addition A_j0=<S+j Omega|exp[-i(H9-E0)t]|S+0 Omega>; removal B_0j=conjugate(<S-j Omega|exp[-i(H7-E0)t]|S-0 Omega>), negative-frequency transposed spatial component; all raw source weights retained',physical_scope='N9 rho total+1 has boundary E-1; N7 rho total-1 has boundary E+1; rho is occupation-minus-ion, electric charge sign may be opposite. No positron or mass claim.',elapsed_seconds=time.monotonic()-start)
    (ROOT/(P+'charged_result.json')).write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(ROOT/(P+'charged_states.npz'),**states)
    print(json.dumps(dict(branches={k:dict(endpoint_right_charge=v['samples'][-1]['right_charge'],delta_right_charge=v['samples'][-1]['delta_right_charge_own_t0'],resolution=v['resolution'],algebra=v['algebra']) for k,v in results.items()},elapsed_seconds=output['elapsed_seconds'])),flush=True)
if __name__=='__main__':
    main()
