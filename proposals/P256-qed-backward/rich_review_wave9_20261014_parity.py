"""NEW occupied parity/source-bright stationary pole and actual FIELD-only PVM."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, expm_multiply, norm, splu
from rich_parity_sea_field_20261014_model import build, momentum_basis, fourier_source

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave9_20261014_parity'


def mean(op,state):
    return float(np.vdot(state,op@state).real)


def evolve(h,state,time):
    return expm_multiply(-1j*time*h,state,traceA=-1j*time*h.diagonal().sum())


def groups(e):
    start=0
    while start<len(e):
        end=start+1
        while end<len(e) and abs(e[end]-e[start])<2e-8:
            end+=1
        yield np.arange(start,end)
        start=end


def inclusive(m,state):
    axis=np.array([.43,-.51,.745]);axis/=np.linalg.norm(axis)
    s=sum(axis[a]*m['S'][a] for a in range(3))
    # Exact odd N7 spectrum +/-1/2,+/-3/2,+/-5/2: inclusive sign, never state filter.
    abscissa=np.array([.5,1.5,2.5]);coeff=np.linalg.solve(np.column_stack([np.ones(3),abscissa**2,abscissa**4]),1/abscissa)
    x=s@state;y=s@(s@x);z=s@(s@y)
    signed=coeff[0]*x+coeff[1]*y+coeff[2]*z
    spin_states=[(state+signed)/2,(state-signed)/2]
    count=np.rint(m['nums'][0].diagonal().real).astype(int)
    # Whole source-smeared A detector with c=G0/G00, not a mode overlap projector.
    c=m['geometry']['gram'][0]/m['geometry']['gram'][0,0]
    field=sum(c[a]*m['Q'][a] for a in range(2)).diagonal().real>=0
    probabilities=[]
    for occupation in range(5):
        for psi in spin_states:
            for fieldmask in (field,~field):
                selected=(count==occupation)&fieldmask
                probabilities.append(float(np.vdot(psi[selected],psi[selected]).real))
    return dict(probabilities=probabilities,sum=sum(probabilities),norm=float(np.vdot(state,state).real),order='Nsite0=0..4 x physical obliqueSpin+/- x actual whole3D profile-smeared A+/-')


def main():
    p=json.loads((ROOT/(PREFIX+'_input.json')).read_text());m=build(p);h=m['H']
    sectors={};neutral=[]
    for number in (5,6,7):
        sectors[number]=[]
        for momentum in range(3):
            u=momentum_basis(m,number,momentum);e,v=eigh((u.getH()@h@u).toarray())
            sectors[number].append(dict(U=u,e=e,v=v,m=momentum))
            if number==6:
                neutral.append(float(e[0]))
    eg=min(neutral);references=[];ref_m=[]
    for block in sectors[6]:
        for i in np.flatnonzero(abs(block['e']-eg)<2e-8):
            references.append(block['U']@block['v'][:,i]);ref_m.append(block['m'])
    refs=np.column_stack(references);rank=len(references)
    number_min=[]
    for n in range(13):
        if n in sectors:
            number_min.append(min(float(b['e'][0]) for b in sectors[n]));continue
        values=[]
        for momentum in range(3):
            u=momentum_basis(m,n,momentum)
            if not u.shape[1]:
                continue
            block=(u.getH()@h@u).tocsr()
            values.append(float(eigh(block.toarray(),eigvals_only=True)[0]) if block.shape[0]<=2 else float(eigsh(block,k=1,which='SA',tol=1e-11,return_eigenvectors=False)[0]))
        number_min.append(min(values))
    source_records=[];canonical=[];stationary_candidates=[]
    for momentum in range(3):
        source=fourier_source(m,momentum);car=np.zeros((4,4),complex)
        for label,number,ops in [('addition',7,source),('removal',5,[a.getH() for a in source])]:
            row=dict(source_momentum=momentum,number=number,all_virtual_states=sum(len(b['e']) for b in sectors[number]),blocks=[])
            for block in sectors[number]:
                u=block['U'];e=block['e'];v=block['v']
                amplitudes=[v.conj().T@(u.getH()@np.column_stack([op@g for op in ops])) for g in references]
                raw=sum(a.conj().T@a for a in amplitudes)/rank
                car+=raw if label=='addition' else raw.T
                for ix in groups(e):
                    residue=sum(a[ix].conj().T@a[ix] for a in amplitudes)/rank
                    weight=float(np.trace(residue).real)
                    if weight<=1e-10:
                        continue
                    basis=u@v[:,ix];vm=basis.conj().T@(m['V']@basis);jm=[basis.conj().T@(j@basis) for j in m['J']]
                    item=dict(target_momentum=block['m'],gap=float(e[ix[0]]-eg),multiplet_rank=len(ix),source_residue=weight,
                              velocity_eigenvalues=np.linalg.eigvalsh(vm).tolist(),velocity_matrix_norm=float(np.linalg.norm(vm)),current_matrix_norms=[float(np.linalg.norm(j)) for j in jm],
                              S2_eigenvalues=np.linalg.eigvalsh(basis.conj().T@(m['S2']@basis)).tolist())
                    row['blocks'].append(item)
                    if number==7 and block['m']==0:
                        stationary_candidates.append((weight,block,ix,item))
            source_records.append(row)
        canonical.append(float(np.linalg.norm(car-np.eye(4))))
    # Select strongest ACTUAL source pole, not dark floor/averaged opposite velocities.
    weight,charged,ix,carrier=max(stationary_candidates,key=lambda entry:entry[0])
    carrier_basis=charged['U']@charged['v'][:,ix];g7=carrier_basis[:,0];e7=float(charged['e'][ix[0]])
    u=charged['U'];hc=(u.getH()@h@u).tocsr();gc=np.asarray(u.getH()@g7)
    qe,qv=eigh(m['M']);sqrtM=(qv*np.sqrt(qe))@qv.T;inverse=(qv/np.sqrt(qe))@qv.T
    qcolumns=np.column_stack([(u.getH()@q@u)@gc for q in m['Q']])@sqrtM
    jcolumns=np.column_stack([(u.getH()@j@u)@gc for j in m['J']])@inverse
    columns=np.column_stack([qcolumns,jcolumns]);columns-=gc[:,None]*(gc.conj()@columns)
    amplitudes=charged['v'].conj().T@columns;gaps=charged['e']-e7;z=p['frequency']+1j*p['eta']
    retarded=amplitudes.conj().T@(amplitudes/(z-gaps)[:,None])-(amplitudes.conj().T@(amplitudes/(z+gaps)[:,None])).T
    pos=(z+e7)*sp.eye(len(gc))-hc;neg=(z-e7)*sp.eye(len(gc))+hc
    xp=splu(pos.tocsc()).solve(columns);xm=splu(neg.tocsc()).solve(columns)
    direct=columns.conj().T@xp-(columns.conj().T@xm).T
    mixed_blocks=[]
    for group in groups(charged['e']):
        if abs(gaps[group[0]])<2e-8:
            continue
        t=amplitudes[group];cross=t[:,:2].conj().T@t[:,2:]
        singular=np.linalg.svd(cross,compute_uv=False)
        if max(singular)>1e-9:
            mixed_blocks.append(dict(signed_gap=float(gaps[group[0]]),singular_values=singular.tolist()))
    # Non-filtered localized source packet and field-only unitary on SAME initial body state.
    seed=(m['source'][0]+(.21+.29j)*m['source'][5])@references[0];seed/=np.linalg.norm(seed)
    coeff=(.6*m['geometry']['gram'][0]-.8*m['geometry']['gram'][1])/m['geometry']['gram'][0,0]
    q=sum(coeff[a]*m['Q'][a] for a in range(2));unitary_phase=np.exp(1j*p['pulse']*q.diagonal())
    pulse=unitary_phase*seed;wrong_phase=np.exp(1j*p['pulse']*(q@m['nums'][0]).diagonal());wrong=wrong_phase*seed
    actual_j=.6*m['J'][0]-.8*m['J'][1]
    inserted=actual_j@pulse;strength=float(np.vdot(inserted,inserted).real);inserted/=np.sqrt(strength)
    noether=sum(np.array([.43,-.51,.745])[a]*m['S'][a] for a in range(3));wronginsert=noether@pulse;wrongstrength=float(np.vdot(wronginsert,wronginsert).real);wronginsert/=np.sqrt(wrongstrength)
    initial=np.column_stack([seed,pulse,wrong,inserted,wronginsert]);evolved=initial.copy();last=0.;events=[]
    for time in p['times']:
        evolved=evolve(h,evolved,time-last);last=time
        event=dict(time=time,states={})
        for label,state in zip(['reference','FIELD_only','wrong_body_conditioner','FIELD_plus_actual_current_conditional','FIELD_plus_wrong_Noether_conditional'],evolved.T):
            event['states'][label]=dict(norm=float(np.vdot(state,state).real),number=[mean(n,state) for n in m['nums']],spin=[mean(s,state) for s in m['S']],S2=mean(m['S2'],state),field=[mean(q,state) for q in m['Q']],current=[mean(j,state) for j in m['J']],velocity=mean(m['V'],state),inclusive=inclusive(m,state))
        events.append(event)
    coherence=m['source'][4]@m['source'][0].getH();wrongq=q@m['nums'][0]
    adverse=build(p,mirror=False,frame=m['frame'])
    result=dict(kind='NEW_LITERAL_PARITY_OCCUPIED_BRIGHT_CURRENT_AND_PHYSICAL_FIELD_EVENT_AUDIT',parameters=p,
                all_number_ground_energies=number_min,actual_lowest_number=[n for n,value in enumerate(number_min) if abs(value-min(number_min))<1e-8],neutral_energy=eg,neutral_rank=rank,neutral_momenta=ref_m,
                neutral_S2=float(np.trace(refs.conj().T@(m['S2']@refs)).real/rank),source_CAR_errors=canonical,source_all_bright_blocks=source_records,
                selected_strong_source_carrier=carrier,carrier_reference_residual=float(np.linalg.norm(h@g7-e7*g7)),carrier_reference_energy=e7,
                true_P_H=float(norm(m['P']@h@m['P'].getH()-h)),true_P_current=[float(norm(m['P']@j@m['P'].getH()+j)) for j in m['J']],
                P_reference_leakage=float(np.linalg.norm(m['P']@refs-refs@(refs.conj().T@(m['P']@refs)))),
                C_H=float(norm(m['CF']@h@m['CF'].getH()-h)),antiunitary_T_H=float(norm(m['AT']@h.conjugate()@m['AT'].getH()-h)),
                frame_S_H=[float(norm(s@h-h@s)) for s in m['S']],frame_reference_leakage=[float(np.linalg.norm(s@refs-refs@(refs.conj().T@(s@refs)))) for s in m['S']],
                adverse_missing_mirror_P_H=float(norm(adverse['P']@adverse['H']@adverse['P'].getH()-adverse['H'])),
                actual_profile_Gram_eigenvalues=np.linalg.eigvalsh(m['geometry']['gram']).tolist(),actual_route_source_rank=int(np.linalg.matrix_rank(np.array([route['line'] for route in m['geometry']['routes']]))),
                complete_carrier_current_virtual_states=len(charged['e']),current_spectral_Gram=float(np.linalg.norm(amplitudes.conj().T@amplitudes-columns.conj().T@columns)),retarded_direct_error=float(np.linalg.norm(retarded-direct)),mixed_current_bright_blocks=mixed_blocks,
                independent_field_coherence_commutator=float(norm(q@coherence-coherence@q)),wrong_body_coherence_commutator=float(norm(wrongq@coherence-coherence@wrongq)),
                actual_current_insertion_strength=strength,wrong_Noether_insertion_strength=wrongstrength,physical_events=events,
                scope='Entire occupied neutral/source and strong actual source-pole reference separately bound under SAME H. Strong stationary pole is not automatically firstbright or global lowest charged pole/restmass. Two independent transverse Ax shapes are not orthogonal polarizations at common k. Actual field-only UNITARY with untouched source-packet reference; actual occupation/spin/smeared-field inclusive PVM. Current-conditioned event strength separate from probability, not abs-one-state modal overlap. Mirror vector cancellation may conserve spin/no magnetic torque; no full spatial NoEther/Lorentz/freephoton/inertialmass/g2 admission.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),neutral_reference=refs,strong_carrier_reference=g7,physical_initial=initial,physical_final=evolved)
    print(json.dumps({key:result[key] for key in ('actual_lowest_number','neutral_energy','neutral_rank','neutral_S2','source_CAR_errors','selected_strong_source_carrier','true_P_H','adverse_missing_mirror_P_H','frame_S_H','frame_reference_leakage','actual_route_source_rank','retarded_direct_error','independent_field_coherence_commutator','wrong_body_coherence_commutator','actual_current_insertion_strength')},indent=2))


if __name__=='__main__':
    main()
