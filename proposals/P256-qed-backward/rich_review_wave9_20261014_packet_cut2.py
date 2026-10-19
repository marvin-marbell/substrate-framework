"""NEW joint field-momentum cutoff2 packet; no cutoff1 producer replay."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, expm_multiply, norm
from rich_field_packet_20261014_connected import ConnectedPacket
from rich_field_packet_20261014_model import PacketParameters, spatial_windows
from rich_frame_photon_20261014_model import Parameters

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave9_20261014_packet_cut2'


def evolve(h,state,time):
    return expm_multiply(-1j*time*h,state,traceA=-1j*time*h.diagonal().sum())


def expectation(op,state):
    return float(np.vdot(state,op@state).real)


def measure(m,pair,windows,first):
    rows=[]
    axis=np.array([.43,-.51,.745]);axis/=np.linalg.norm(axis)
    s=sum(axis[a]*sum(local[a] for local in m.spin) for a in range(3));sv,su=eigh(s)
    spin_projectors=[su[:,sv>0]@su[:,sv>0].conj().T,su[:,sv<0]@su[:,sv<0].conj().T]
    mask=windows[2].diagonal().real>=windows[0].diagonal().real
    for state in pair.T:
        tensor=state.reshape(m.nfield,m.ns,64);pop=np.sum(abs(tensor)**2,axis=(1,2))
        electric=np.array([float(pop@w.diagonal().real) for w in windows])
        rows.append(dict(electric=electric.tolist(),first=float(pop@first.diagonal().real),cell=[expectation(n,state) for n in m.N],spin=[expectation(s,state) for s in m.S],cm_current=[expectation(j,state) for j in m.cm_current()],
                         norm=float(np.vdot(state,state).real),energy=expectation(m.H,state),
                         inclusive=[float(np.einsum('fmk,mn,fnk->',tensor[selected].conj(),m.number[c]@ps,tensor[selected]).real) for c in range(2) for ps in spin_projectors for selected in (mask,~mask)]))
    pulse,reference=rows;excess=np.array(pulse['electric'])-reference['electric']
    return dict(pulse=pulse,reference=reference,excess=excess.tolist(),excess_centroid=float((pulse['first']-reference['first'])/sum(excess)) if abs(sum(excess))>1e-13 else None,
                cell_contrast=(np.array(pulse['cell'])-reference['cell']).tolist(),spin_contrast=(np.array(pulse['spin'])-reference['spin']).tolist())


def main():
    data=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    p=PacketParameters(base=Parameters(**data['base']),cutoff=2,qwall=data['qwall'],supplier_spacing=data['supplier_spacing'])
    grams,first=spatial_windows(p)
    models=[ConnectedPacket(p,n) for n in range(3)];n0,m,n2=models
    prep=m.H-data['well']*m.N[0]-.029*(.43*m.S[0]-.51*m.S[1]+.745*m.S[2])
    rng=np.random.default_rng(92221014)
    e,v=eigsh(prep,k=1,which='SA',v0=rng.normal(size=m.size)+1j*rng.normal(size=m.size),tol=2e-10,maxiter=30000)
    g=v[:,0];eg=float(e[0])
    coeff=(.8*grams[0,0]-.6*grams[0,1])/m.G[0,0]
    q=sum(coeff[a]*m.Q[a] for a in range(6));qe,qv=eigh(q.toarray())
    u=(qv*np.exp(1j*data['strength']*qe))@qv.conj().T
    initial=(u@g.reshape(m.nfield,m.internal)).ravel();pair=np.column_stack([initial,g])
    windows=[m.electric_window(gw) for gw in grams];moment=m.electric_window(sum(first))
    cut=ConnectedPacket(p,1,bridge=False)
    records={};saved=dict(initial=initial,reference_initial=g)
    for label,c,h in [('full',m,m.H),('same_initial_physical_cut',cut,cut.H),('same_initial_bare_electric',m,m.E)]:
        both=pair.copy();entries=[];last=0.
        for time in data['times']:
            both=evolve(h,both,time-last);last=time
            row=measure(c,both,windows,moment);row['time']=time;entries.append(row)
        records[label]=entries;saved[label+'_final_pulse']=both[:,0];saved[label+'_final_reference']=both[:,1]
    plus=np.column_stack([m.source_to(n2,a)@g for a in range(4)])
    minus=np.column_stack([n0.source_to(m,a).getH()@g for a in range(4)])
    returned_plus=evolve(n2.H,plus,data['source_time']);returned_minus=evolve(n0.H,minus,data['source_time'])
    pg=plus.conj().T@plus;mg=minus.conj().T@minus
    k=np.array([.29,-.33,.21]);rho=m.density_fourier(k);current=m.current_fourier(k)
    continuation=float(norm(m.H@rho-rho@m.H+sum(k[a]*current[a] for a in range(3))))
    encode=lambda a:dict(real=a.real.tolist(),imag=a.imag.tolist())
    # This old result is a frozen observed comparison, never an old CLI replay or borrowed state.
    previous=json.loads((ROOT/data['frozen_cut1_result']).read_text())
    old=previous['trajectories']['interacting'][-1]
    result=dict(kind='NEW_CHANGED_JOINT_FIELD_MOMENTUM_REGULATOR_PACKET_AUDIT',parameters=asdict(p),dimensions=[c.size for c in models],field_momentum_tuples=m.indices.tolist(),
                actual_prepared_H_residual=float(np.linalg.norm(prep@g-eg*g)),reference_event_H_residual=float(np.linalg.norm(m.H@g-expectation(m.H,g)*g)),
                prepared_energy=eg,physical_Gram=m.G.tolist(),full_dual_metric=m.Minv.tolist(),quadrature_error=float(np.linalg.norm(sum(grams)-m.G)),
                bare_regional_E_commutators=[float(norm(m.E@m.fop(w)-m.fop(w)@m.E)) for w in windows],
                pulse_far_energy_operator_change=float(np.linalg.norm(u.conj().T@windows[2].toarray()@u-windows[2].toarray())),
                physical_continuity=continuation,centre_charge_current_error=[float(norm(m.cm_current()[a]+1j*(m.H@m.X[a]-m.X[a]@m.H))) for a in range(3)],
                source_CAR_Gram=float(np.linalg.norm(pg+mg.T-np.eye(4))),wrong_removal_untransposed=float(np.linalg.norm(pg+mg-np.eye(4))),
                complete_N0_N2_time_kernels=dict(time=data['source_time'],addition=encode(plus.conj().T@returned_plus),removal_RAW=encode(minus.conj().T@returned_minus)),
                trajectories=records,frozen_cut1_comparison=dict(path=data['frozen_cut1_result'],time=old['time'],excess=old['excess'],centroid=old['excess_centroid'],source='Immutable independent Wave8 result, same base/protocol, only changed field Fourier momentum regulator; own reference recomputed, no state transferred.'),
                interpretation='NEW larger joint Fourier-field momentum regulator, not oscillatorFockN or CAR particleN. Paired interacting-state regional energy/spin/motion and inclusive physical detector under entire sharedH. Bareelectric diagonal identity measured. One increase is finite-regulator sensitivity, not convergence/asymptotic/freephoton proof. Source kernels are nonstationary reference-anchored, raw removal transpose needed for canonicalG.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**saved,plus=plus,minus=minus,field_unitary=u)
    print(json.dumps({k:result[k] for k in ('dimensions','actual_prepared_H_residual','reference_event_H_residual','bare_regional_E_commutators','physical_continuity','source_CAR_Gram')},indent=2))
    print(json.dumps({label:[dict(time=row['time'],centroid=row['excess_centroid'],excess=row['excess'],cell=row['cell_contrast'],spin=row['spin_contrast']) for row in entries] for label,entries in records.items()},indent=2))


if __name__=='__main__':
    main()
