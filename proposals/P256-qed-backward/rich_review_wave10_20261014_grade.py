"""NEW mirror-graded vector/common-q fields, complete occupied sources and PVM."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply, norm, splu
from rich_graded_mirror_20261014_model import build, momentum_basis, fourier_source, field_values

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave10_20261014_grade'


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


def receiver(m,state,axis):
    s=sum(axis[a]*m['S'][a] for a in range(3));x=s@state;y=s@(s@x);z=s@(s@y)
    abscissa=np.array([.5,1.5,2.5]);c=np.linalg.solve(np.column_stack([np.ones(3),abscissa**2,abscissa**4]),1/abscissa)
    signed=c[0]*x+c[1]*y+c[2]*z;spin=[(state+signed)/2,(state-signed)/2]
    count=np.rint(m['nums'][0].diagonal().real).astype(int);q=[a.diagonal().real>=0 for a in m['Q']]
    probabilities=[]
    for number in range(5):
        for psi in spin:
            for a in (q[0],~q[0]):
                for b in (q[1],~q[1]):
                    mask=(count==number)&a&b;probabilities.append(float(np.vdot(psi[mask],psi[mask]).real))
    return dict(probabilities=probabilities,sum=sum(probabilities),norm=float(np.vdot(state,state).real),order='Nsite0=0..4 x oblique fullphysicalSpin+/- x actual common-q field-smeared A0+/- x A1+/-; all remaining states traced')


def main():
    p=json.loads((ROOT/(PREFIX+'_input.json')).read_text());m=build(p);h=m['H'];all_sectors={};references=[]
    for number in (5,6,7):
        all_sectors[number]=[]
        for momentum in range(3):
            u=momentum_basis(m,number,momentum);hc=(u.getH()@h@u).tocsr();e,v=eigh(hc.toarray())
            all_sectors[number].append(dict(U=u,H=hc,e=e,v=v,m=momentum))
    eg=min(float(b['e'][0]) for b in all_sectors[6]);ref_m=[]
    for block in all_sectors[6]:
        for index in np.flatnonzero(abs(block['e']-eg)<2e-8):
            references.append(block['U']@block['v'][:,index]);ref_m.append(block['m'])
    refs=np.column_stack(references);rank=len(references);z=p['frequency']+1j*p['eta']
    source_rows=[];greens=[];candidates=[]
    for momentum in range(3):
        ops=fourier_source(m,momentum);car=np.zeros((4,4),complex);green=car.copy();direct_green=car.copy()
        for label,number,operators,sign in [('addition',7,ops,-1),('removal',5,[a.getH() for a in ops],1)]:
            row=dict(source_momentum=momentum,number=number,sector_floors=[],bright=[])
            for block in all_sectors[number]:
                u,e,v,hc=block['U'],block['e'],block['v'],block['H'];row['sector_floors'].append(dict(target_momentum=block['m'],gap=float(e[0]-eg)))
                columns=[np.asarray(u.getH()@np.column_stack([a@g for a in operators])) for g in references]
                amp=[v.conj().T@c for c in columns];gram=sum(a.conj().T@a for a in amp)/rank
                car+=gram if label=='addition' else gram.T
                denominator=z+sign*(e-eg)
                kernel=sum(a.conj().T@(a/denominator[:,None]) for a in amp)/rank
                matrix=(z-sign*eg)*sp.eye(len(e))+sign*hc;factor=splu(matrix.tocsc())
                direct=sum(c.conj().T@factor.solve(c) for c in columns)/rank
                green+=kernel if label=='addition' else kernel.T;direct_green+=direct if label=='addition' else direct.T
                for indices in groups(e):
                    residue=sum(a[indices].conj().T@a[indices] for a in amp)/rank;weight=float(np.trace(residue).real)
                    if weight<=1e-10:
                        continue
                    basis=u@v[:,indices];velocity=basis.conj().T@(m['V']@basis);js=[basis.conj().T@(j@basis) for j in m['J']]
                    item=dict(target_momentum=block['m'],gap=float(e[indices[0]]-eg),source_residue=weight,multiplet_rank=len(indices),
                              velocity_eigenvalues=np.linalg.eigvalsh(velocity).tolist(),velocity_matrix_norm=float(np.linalg.norm(velocity)),current_matrix_norms=[float(np.linalg.norm(j)) for j in js],
                              S2_eigenvalues=np.linalg.eigvalsh(basis.conj().T@(m['S2']@basis)).tolist())
                    row['bright'].append(item)
                    if number==7 and block['m']==0:
                        candidates.append((item,block,indices,basis))
            source_rows.append(row)
        greens.append(dict(source_momentum=momentum,CAR_error=float(np.linalg.norm(car-np.eye(4))),canonical_G_real=green.real.tolist(),canonical_G_imag=green.imag.tolist(),independent_resolvent_error=float(np.linalg.norm(green-direct_green))))
    candidate,charged,ix,basis=min(candidates,key=lambda item:item[0]['gap'])
    axis=np.asarray(p['spin_axis'],float);axis/=np.linalg.norm(axis);s=sum(axis[a]*m['S'][a] for a in range(3))
    spin_e,spin_v=eigh(basis.conj().T@(s@basis));polarized=[basis@spin_v[:,0],basis@spin_v[:,-1]]
    e7=float(charged['e'][ix[0]]);g=polarized[-1];u=charged['U'];gc=np.asarray(u.getH()@g);hc=charged['H'];e,v=charged['e'],charged['v'];gap=e-e7
    q=[(u.getH()@a@u).tocsr() for a in m['Q']];js=[(u.getH()@a@u).tocsr() for a in m['J']]
    cols=np.column_stack([a@gc for a in q+js]);cols-=gc[:,None]*(gc.conj()@cols);amp=v.conj().T@cols
    retarded=amp.conj().T@(amp/(z-gap)[:,None])-(amp.conj().T@(amp/(z+gap)[:,None])).T
    pos=(z+e7)*sp.eye(len(e))-hc;neg=(z-e7)*sp.eye(len(e))+hc
    xp=splu(pos.tocsc()).solve(cols);xm=splu(neg.tocsc()).solve(cols);direct=cols.conj().T@xp-(cols.conj().T@xm).T
    def leg(coeff):
        a=amp[:,:2]@np.asarray(coeff);positive=a*(gap>2e-8);residue=float(np.vdot(positive,positive).real);state=v@positive/np.sqrt(residue)
        energy=float(np.dot(abs(positive)**2,gap)/residue);variance=float(np.dot(abs(positive)**2,(gap-energy)**2)/residue)
        return dict(state=state,residue=residue,energy=energy,variance=variance,field=np.array([np.vdot(gc,op@state) for op in q]),current=np.array([np.vdot(gc,op@state) for op in js]))
    incoming=leg([1.,.73j]);outgoing=leg([.61,-1j]);ai=incoming['field'];ao=outgoing['field'].conj()
    ji=sum(ai[a]*js[a] for a in range(2));jo=sum(ao[a]*js[a] for a in range(2));k=[[u.getH()@m['K'][a][b]@u for b in range(2)] for a in range(2)]
    contact=sum(ao[a]*ai[b]*(k[a][b]@gc) for a in range(2) for b in range(2))
    rp=splu(((e7+incoming['energy']+1j*p['eta'])*sp.eye(len(e))-hc).tocsc()).solve(ji@gc)
    rm=splu(((e7-outgoing['energy']+1j*p['eta'])*sp.eye(len(e))-hc).tocsc()).solve(jo@gc)
    reaction=contact+jo@rp+ji@rm
    no_vector=build(p,vector_scale=0.,frame=m['frame']);kick=sum(p['field_kick'][a]*m['E'][a] for a in range(2))
    records={};saved={'neutral_reference':refs,'carrier_reference':g};pulses=[]
    for label,base in zip(('spin_minus','spin_plus'),polarized):
        pulse=evolve(-kick,base,1.);pulses.append(pulse);saved[label+'_reference']=base;saved[label+'_FIELD_initial']=pulse;records[label]={}
        for control,law,jcurrent,velocity in [('full',h,m['J'],m['V']),('same_initial_vector_removed',no_vector['H'],no_vector['J'],no_vector['V'])]:
            pair=np.column_stack([pulse,base]);last=0.;rows=[]
            for time in p['times']:
                pair=evolve(law,pair,time-last);last=time;states=[]
                for state in pair.T:
                    states.append(dict(spin=[mean(op,state) for op in m['S']],S2=mean(m['S2'],state),current=[mean(op,state) for op in jcurrent],velocity=mean(velocity,state),field=[mean(op,state) for op in m['Q']],inclusive=receiver(m,state,axis)))
                rows.append(dict(time=time,pulse=states[0],reference=states[1],spin_contrast=(np.array(states[0]['spin'])-states[1]['spin']).tolist(),current_contrast=(np.array(states[0]['current'])-states[1]['current']).tolist()))
            records[label][control]=rows;saved[label+'_'+control+'_final']=pair
    packet=(m['source'][0]+(.23+.31j)*m['source'][1]+.37*m['source'][4])@references[0];packet/=np.linalg.norm(packet)
    field_packet=evolve(-kick,packet,1.);both=np.column_stack([field_packet,packet]);last=0.;motion=[]
    for time in p['times']:
        both=evolve(h,both,time-last);last=time
        motion.append(dict(time=time,cell_contrast=[mean(n,both[:,0])-mean(n,both[:,1]) for n in m['nums']],spin_contrast=[mean(s,both[:,0])-mean(s,both[:,1]) for s in m['S']],inclusive=receiver(m,both[:,0],axis)))
    transfer=[];sewing=np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],complex)
    for a,b in zip(m['transfers'][::2],m['transfers'][1::2]):
        vector=np.vstack([a['matrix'][:2],b['matrix'][2:]])
        transfer.append(dict(graded_vector_norm=float(np.linalg.norm(vector)),scalar_vector_anticommutator=float(np.linalg.norm(sewing@vector+vector@sewing)),inverse_mirror_rotation=float(np.linalg.norm(a['rotation'].conj().T-b['rotation']))))
    step=2e-5;derivatives=[];wrong=[]
    for a in range(2):
        shift=np.zeros(2);shift[a]=step;hp=build(p,external=shift,frame=m['frame'])['H'];hm=build(p,external=-shift,frame=m['frame'])['H'];d=(hp-hm)/(2*step)
        derivatives.append(float(norm(d+m['J'][a])));wrong.append(float(norm(d+m['S'][a])))
    points=np.array([[0.,.17,.11],[p['spacing']/7,.17,.11]]);fields=field_values(points,p)
    selected_lower=[row for record in source_rows if record['number']==7 for row in record['bright'] if row['gap']<candidate['gap']-2e-8]
    summary=lambda leg:{key:leg[key] for key in ('residue','energy','variance')}
    result=dict(kind='NEW_GRADED_VECTOR_OCCUPIED_COMMON_Q_CURRENT_AND_PHYSICAL_SPIN_EVENT',parameters=p,selected_neutral_E=eg,neutral_rank=rank,neutral_momenta=ref_m,neutral_frame_S2=float(np.trace(refs.conj().T@(m['S2']@refs)).real/rank),
                canonical_source_G=greens,all_source_bright_records=source_rows,first_m0_source_bright=candidate,all_lower_charged_bright=selected_lower,
                polarized_frame_spin_eigenvalues=spin_e.tolist(),ordered_mirror_vectors=transfer,
                actual_common_q=m['geometry']['common_q'],actual_field_point_dot_products=np.einsum('si,si->s',fields[0],fields[1]).tolist(),actual_field_q_dots=[(field@np.asarray(m['geometry']['common_q'])).tolist() for field in fields],
                actual_field_Gram=m['geometry']['gram'].tolist(),actual_current_Gram_eigenvalues=np.linalg.eigvalsh(cols[:,2:].conj().T@cols[:,2:]).tolist(),
                true_P_H=float(norm(m['P']@h@m['P'].getH()-h)),true_C_H=float(norm(m['CF']@h@m['CF'].getH()-h)),true_T_H=float(norm(m['AT']@h.conjugate()@m['AT'].getH()-h)),
                H_frame_spin=[float(norm(h@s-s@h)) for s in m['S']],physical_current_derivative=derivatives,wrong_NoEther_as_current_derivative=wrong,
                full_current_virtual_states=len(e),retarded_independent_error=float(np.linalg.norm(retarded-direct)),wrong_positive_only_retarded_error=float(np.linalg.norm(retarded-cols.conj().T@xp)),
                actual_normalized_positive_field_legs=dict(incoming=summary(incoming),outgoing=summary(outgoing)),actual_all_virtual_reaction_strength=float(np.vdot(reaction,reaction).real),omitted_contact_state_error=float(np.linalg.norm(contact)),omitted_negative_order_error=float(np.linalg.norm(ji@rm)),
                field_only_spin_commutator=float(norm(kick@s-s@kick)),actual_polarized_FIELD_events=records,localized_unfiltered_source_motion=motion,
                scope='Actual first m0 BRIGHT multiplet/all lower bright states, not selected excited dominance or dark floor. Physical frame source/whole occupied angular continuum distinct from full spatial NoEther. Common +/-q orthogonal transverse standing waves, finiteT3 and DVR/globalfields, not traveling/asymptotic photons. Actual independent unitary FIELD prep, both pulse/baseline under same full/off law, full40-outcome physical PVM. Reaction strength conditional/not crosssection. No massfit, bareB2/freegamma, sourcejhalf filter, Lorentz/restmass/species admission or splicing other roots.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**saved,localized_source_initial=packet,localized_field_initial=field_packet)
    print(json.dumps({key:result[key] for key in ('selected_neutral_E','neutral_frame_S2','first_m0_source_bright','polarized_frame_spin_eigenvalues','ordered_mirror_vectors','actual_current_Gram_eigenvalues','true_P_H','true_C_H','true_T_H','H_frame_spin','physical_current_derivative','retarded_independent_error','actual_all_virtual_reaction_strength')},indent=2))


if __name__=='__main__':
    main()
