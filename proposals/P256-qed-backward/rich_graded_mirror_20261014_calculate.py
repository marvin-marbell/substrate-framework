"""NEW whole-H occupied G±, graded physical current and coherent field event."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply
from rich_graded_mirror_20261014_model import build, momentum_basis, fourier_source


def norm(a):
    return float(np.linalg.norm(a.data if hasattr(a,'tocsr') else a))


def enc(a):
    a=np.asarray(a)
    return dict(real=a.real.tolist(),imag=a.imag.tolist())


def expectation(a,v):
    return float(np.vdot(v,a@v).real)


def clusters(e):
    start=0
    while start<len(e):
        end=start+1
        while end<len(e) and e[end]-e[start]<2e-8:end+=1
        yield start,end
        start=end


def sectors(model,number):
    result=[]
    for m in range(model['parameters']['sites']):
        u=momentum_basis(model,number,m); h=(u.getH()@model['H']@u).tocsr()
        e,v=eigh(h.toarray(),driver='evd',overwrite_a=True,check_finite=False)
        result.append(dict(U=u,H=h,e=e,v=v,m=m))
    return result


def source_spectra(model,all_sectors,refs,eg,z):
    records={}; greens={}; vertices={}; saved={}; bright=[]
    for momentum in range(model['parameters']['sites']):
        operators=fourier_source(model,momentum); green=np.zeros((4,4),complex); car=green.copy(); vv=[green.copy(),green.copy()]
        records[str(momentum)]={}
        for label,number,ops in [('addition',2*model['parameters']['sites']+1,operators),('removal',2*model['parameters']['sites']-1,[d.getH() for d in operators])]:
            poles=[]; groups=[]
            for b in all_sectors[number]:
                amps=[b['v'].conj().T@(b['U'].getH()@np.column_stack([d@r for d in ops])) for r in refs]
                saved[f'source{momentum}_{label}_m{b["m"]}']=np.stack(amps)
                gaps=b['e']-eg
                for i,gap in enumerate(gaps):
                    residue=sum(np.outer(a[i].conj(),a[i]) if label=='addition' else np.outer(a[i],a[i].conj()) for a in amps)/len(refs)
                    green+=residue/(z-gap if label=='addition' else z+gap);car+=residue
                    poles.append(dict(target_m=b['m'],gap=float(gap),residue=enc(residue),weight=float(np.trace(residue).real)))
                for start,end in clusters(b['e']):
                    rr=sum(a[start:end].conj().T@a[start:end] for a in amps)/len(refs)
                    weight=float(np.trace(rr).real)
                    if weight<1e-10:continue
                    w=b['U']@b['v'][:,start:end]
                    vel=w.conj().T@(model['V']@w); currents=[w.conj().T@(j@w) for j in model['J']]
                    item=dict(N=number,source_m=momentum,target_m=b['m'],gap=float(gaps[start]),degeneracy=end-start,residue=enc(rr),weight=weight,
                              velocity_matrix=enc(vel),velocity_eigenvalues=np.linalg.eigvalsh(vel).tolist(),current_matrices=[enc(j) for j in currents],
                              current_norms=[norm(j) for j in currents],S2_eigenvalues=np.linalg.eigvalsh(w.conj().T@(model['S2']@w)).tolist(),
                              whole_block_rest=norm(vel)<1e-8 and max(norm(j) for j in currents)<1e-8)
                    groups.append(item);bright.append(item)
                for axis,j in enumerate(model['J']):
                    je=b['v'].conj().T@(b['U'].getH()@j@b['U']@b['v'])
                    denominator=z-gaps if label=='addition' else z+gaps
                    for amp in amps:
                        routed=amp/denominator[:,None]
                        vertex=routed.conj().T@je@routed
                        vv[axis]+=vertex/len(refs)
            records[str(momentum)][label]=dict(all_poles=poles,all_bright_clusters=groups,weight=sum(p['weight'] for p in poles),lowest_bright_gap=min(g['gap'] for g in groups),sector_floors=[dict(m=b['m'],gap=float(b['e'][0]-eg)) for b in all_sectors[number]])
        greens[str(momentum)]=dict(G=enc(green),CAR_error=norm(car-np.eye(4)))
        vertices[str(momentum)]=dict(source_resolvent_current_bilinear=[enc(a) for a in vv],meaning='Advanced-retarded inclusive current bilinear, not analytic GJG: includes occupied addition and removal with own signed denominator.')
    return records,greens,vertices,saved,bright


def response_and_event(model,block,index,p):
    u=block['U'];h=block['H'];e=block['e'];v=block['v'];g=v[:,index];e0=e[index];gap=e-e0;z=complex(*p['frequency'])
    ops={key:[(u.getH()@a@u).tocsr() for a in model[key]] for key in ('Q','E','J','S')}
    velocity=(u.getH()@model['V']@u).tocsr()
    contacts=[[(u.getH()@model['K'][a][b]@u).tocsr() for b in range(2)] for a in range(2)]
    amplitudes=v.conj().T@np.column_stack([a@g for a in ops['Q']+ops['J']])
    amplitudes[abs(gap)<2e-8]=0
    retarded=amplitudes.conj().T@(amplitudes/(z-gap)[:,None])-(amplitudes.conj().T@(amplitudes/(z+gap)[:,None])).T
    poles=[]
    for start,end in clusters(e):
        amp=amplitudes[start:end]
        poles.append(dict(gap=float(gap[start]),degeneracy=end-start,residue=enc(amp.conj().T@amp),mixed_singular_values=np.linalg.svd(amp[:,:2].conj().T@amp[:,2:],compute_uv=False).tolist()))
    def leg(coeff):
        aa=amplitudes[:,:2]@np.asarray(coeff);positive=gap>2e-8;aa[~positive]=0
        weight=float(np.vdot(aa,aa).real);state=v@aa/np.sqrt(weight)
        return dict(state=state,residue=weight,energy=float(np.dot(abs(aa)**2,gap)/weight),field=np.array([np.vdot(g,a@state) for a in ops['Q']]),current=np.array([np.vdot(g,a@state) for a in ops['J']]))
    incoming=leg([1.,1j]);outgoing=leg([1.,-.61j]);ai=incoming['field'];ao=outgoing['field'].conj()
    ji=sum(ai[a]*ops['J'][a] for a in range(2));jo=sum(ao[a]*ops['J'][a] for a in range(2))
    def resolvent(state,shift):return v@((v.conj().T@state)/(e0+shift+1j*p['frequency'][1]-e))
    contact=sum(ao[a]*ai[b]*(contacts[a][b]@g) for a in range(2) for b in range(2))
    positive=jo@resolvent(ji@g,incoming['energy']);negative=ji@resolvent(jo@g,-outgoing['energy']);reaction=contact+positive+negative
    final_amplitudes=v.conj().T@reaction
    inclusive=[dict(energy=float(e[start]-e0),strength=float(np.linalg.norm(final_amplitudes[start:end])**2),degeneracy=end-start) for start,end in clusters(e)]
    end=index+1
    while end<len(e) and abs(e[end]-e0)<2e-8:end+=1
    start=index
    while start>0 and abs(e[start-1]-e0)<2e-8:start-=1
    w=v[:,start:end];direction=np.array(p['spin_axis']);saxis=sum(direction[a]*ops['S'][a] for a in range(3))
    polarized_e,polarized_v=eigh(w.conj().T@(saxis@w))
    # The polarization uses a coherent eigenvector of physical S in the WHOLE
    # selected degenerate charged block; neither mean opposite velocities nor a
    # source-spin projection is substituted for the rest test.
    baselines=[w@polarized_v[:,j] for j in (0,len(polarized_e)-1)]
    kick=1j*sum(p['field_kick'][a]*ops['E'][a] for a in range(2))
    no_vector=build(p,vector_scale=0.,frame=model['frame']);hc=(u.getH()@no_vector['H']@u).tocsr()
    event={};states={};trajectories=[]
    for label,base in zip(('spin_minus','spin_plus'),baselines):
        kicked=expm_multiply(kick,base,traceA=kick.diagonal().sum())
        event[label]={};trajectories.append({})
        states[label+'_baseline']=u@base;states[label+'_field_only_prepared']=u@kicked
        for control,initial,law in [('baseline',base,h),('field_only_unitary',kicked,h),('same_initial_vector_removed',kicked,hc)]:
            physical_current=ops['J'] if control!='same_initial_vector_removed' else [(u.getH()@a@u).tocsr() for a in no_vector['J']]
            physical_velocity=velocity if control!='same_initial_vector_removed' else (u.getH()@no_vector['V']@u).tocsr()
            rows=[]
            for time in p['times']:
                state=expm_multiply(-1j*time*law,initial,traceA=-1j*time*law.diagonal().sum())
                rows.append(dict(time=time,norm=float(np.linalg.norm(state)),energy=expectation(law,state),velocity=expectation(physical_velocity,state),current=[expectation(a,state) for a in physical_current],original_law_current_detector=[expectation(a,state) for a in ops['J']],spin=[expectation(a,state) for a in ops['S']],field_Q=[expectation(a,state) for a in ops['Q']],field_E=[expectation(a,state) for a in ops['E']],outgoing_global_leg_projector_probability=float(abs(np.vdot(outgoing['state'],state))**2)))
            event[label][control]=rows
        trajectories[-1]=event[label]
    differences=[]
    for i,time in enumerate(p['times']):
        differences.append(dict(time=time,spin_dependent_induced_current=(np.array(trajectories[1]['field_only_unitary'][i]['current'])-np.array(trajectories[1]['baseline'][i]['current'])-np.array(trajectories[0]['field_only_unitary'][i]['current'])+np.array(trajectories[0]['baseline'][i]['current'])).tolist(),spin_plus_change=(np.array(trajectories[1]['field_only_unitary'][i]['spin'])-np.array(trajectories[1]['baseline'][i]['spin'])).tolist()))
    statelegs={name:dict(residue=x['residue'],energy=x['energy'],field_amplitude=enc(x['field']),current_amplitude=enc(x['current'])) for name,x in [('incoming',incoming),('outgoing',outgoing)]}
    states.update(reference=u@g,incoming_global_leg=u@incoming['state'],outgoing_global_leg=u@outgoing['state'],inclusive_reaction=u@reaction)
    return dict(reference_m=block['m'],reference_E=float(e0),dimension=len(e),all_field_current_poles=poles,retarded_field_current=enc(retarded),own_legs=statelegs,
                reaction=dict(contact_norm=norm(contact),positive_order_norm=norm(positive),negative_order_norm=norm(negative),all_final_strength=float(np.linalg.norm(final_amplitudes)**2),no_contact_strength=norm(positive+negative)**2,no_negative_strength=norm(contact+positive)**2,all_final_branches=inclusive,meaning='Current/contact resolvent strength, not normalized scattering probability.'),
                physical_polarization_eigenvalues=polarized_e.tolist(),physical_field_only_event=event,polarization_contrasts=differences,
                initial_field_kick_spin_commutator=norm(kick@saxis-saxis@kick)),states


def run(p):
    model=build(p);h=model['H'];sea=2*p['sites'];all_sectors={n:sectors(model,n) for n in (sea-1,sea,sea+1)}
    eg=min(b['e'][0] for b in all_sectors[sea]);refs=[]
    for b in all_sectors[sea]:
        refs.extend(b['U']@b['v'][:,j] for j in np.flatnonzero(abs(b['e']-eg)<2e-8))
    reference=np.column_stack(refs);ident=eye(h.shape[0]);f=model['frame'];parity=model['P']
    raw=dict(source_CAR=max(norm(a.getH()@b+b@a.getH()-(ident if i==j else 0)) for i,a in enumerate(model['source']) for j,b in enumerate(model['source'])),
             raw_Gauss=max(norm(g@f['P']) for g in f['G']),bare_source_physical_gram=max(norm(f['P'].conj().T@a@f['P']) for a in f['bare']),
             parity_square=norm(parity@parity-ident),parity_H=norm(parity@h@parity.getH()-h),parity_current=[norm(parity@a@parity.getH()+a) for a in model['J']],
             parity_velocity=norm(parity@model['V']@parity.getH()+model['V']),parity_axial_spin=[norm(parity@a@parity.getH()-a) for a in model['S']],
             translation_H=norm(model['T']@h-h@model['T']),reference_parity_residual=norm(parity@reference-reference@(reference.conj().T@(parity@reference))),
             S2_H=norm(model['S2']@h-h@model['S2']),spin_H=[norm(a@h-h@a) for a in model['S']],
             continuity=[norm(1j*(h@n-n@h)-(model['edgeJ'][(j-1)%p['sites']]-model['edgeJ'][j])) for j,n in enumerate(model['nums'])],
             physical_current_norms=[norm(a) for a in model['J']],vector_forward_norms=[norm(a) for i,a in enumerate(model['forwards']) if i%5!=0])
    raw['C_H']=norm(model['CF']@h@model['CF'].getH()-h)
    raw['T_H']=norm(model['AT']@h.conjugate()@model['AT'].getH()-h)
    raw['T_axial_spin']=[norm(model['AT']@s.conjugate()@model['AT'].getH()+s) for s in model['S']]
    raw['P_source_sewing']=max(norm(parity@model['source'][4*j+a]@parity.getH()-model['source'][4*((-j)%p['sites'])+(a+2)%4]) for j in range(p['sites']) for a in range(4))
    raw['C_source_sewing']=max(norm(model['CF']@model['source'][4*j+a]@model['CF'].getH()-sum(model['source_C'][a,b]*model['source'][4*j+b].getH() for b in range(4))) for j in range(p['sites']) for a in range(4))
    raw['source_adjoint_spin']=[enc(np.array([[np.trace(f['source'][b].conj().T@(s@f['source'][a]-f['source'][a]@s))/8 for b in range(4)] for a in range(4)])) for s in f['S']]
    raw['N_sector_charge']={str(n):p['charge']*(n-sea) for n in all_sectors}
    sewing=np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],complex)
    transfer_records=[]
    for i in range(0,len(model['transfers']),2):
        a=model['transfers'][i];b=model['transfers'][i+1];vector=np.zeros((4,4),complex)
        vector[:2]=a['matrix'][:2];vector[2:]=b['matrix'][2:]
        transfer_records.append(dict(ordered_vector=enc(vector),vector_norm=norm(vector),mirror_inverse=norm(a['rotation'].conj().T-b['rotation']),scalar_vector_anticommutator=norm(sewing@vector+vector@sewing)))
    raw['ordered_mirror_transfers']=transfer_records
    delta=1e-5;raw['current_derivative']=[]
    for a in range(2):
        shift=np.zeros(2);shift[a]=delta
        hp=build(p,external=shift,frame=f)['H'];hm=build(p,external=-shift,frame=f)['H']
        raw['current_derivative'].append(norm((hp-hm)/(2*delta)+model['J'][a]))
    spectra,greens,vertices,states,bright=source_spectra(model,all_sectors,refs,eg,complex(*p['frequency']))
    pair=[];pair_ret=np.zeros((2,2),complex)
    for b in all_sectors[sea]:
        amps=[b['v'].conj().T@(b['U'].getH()@np.column_stack([a@r for a in model['J']])) for r in refs]
        states[f'pair_current_m{b["m"]}']=np.stack(amps)
        for start,end in clusters(b['e']):
            residue=sum(a[start:end].conj().T@a[start:end] for a in amps)/len(refs);gap=float(b['e'][start]-eg)
            pair.append(dict(m=b['m'],gap=gap,residue=enc(residue),degeneracy=end-start))
            if gap>2e-8:pair_ret+=residue/(complex(*p['frequency'])-gap)-residue.T/(complex(*p['frequency'])+gap)
    m0=[b for b in bright if b['N']==sea+1 and b['target_m']==0]
    candidate=min(m0,key=lambda b:b['gap']);charged=all_sectors[sea+1][0];index=int(np.argmin(abs(charged['e']-eg-candidate['gap'])))
    em,emstates=response_and_event(model,charged,index,p);states.update(emstates);states['neutral_reference']=reference
    for n,items in all_sectors.items():
        for b in items:states[f'N{n}_m{b["m"]}_energies']=b['e'];states[f'N{n}_m{b["m"]}_vectors']=b['v']
    lower=[b for b in bright if b['N']==sea+1 and b['gap']<candidate['gap']-2e-8]
    local_spinor=model['source'][0]+1j*model['source'][1]+.4*model['source'][2]
    packet=local_spinor@refs[0];packet/=np.linalg.norm(packet)
    field_generator=1j*sum(p['field_kick'][a]*model['E'][a] for a in range(2))
    field_packet=expm_multiply(field_generator,packet,traceA=field_generator.diagonal().sum())
    baseline_density=np.array([expectation(n,refs[0]) for n in model['nums']])
    motion={}
    for label,initial in [('local_source_packet',packet),('same_packet_field_only_unitary',field_packet)]:
        states[label]=initial;motion[label]=[]
        for time in p['times']:
            state=expm_multiply(-1j*time*h,initial,traceA=-1j*time*h.diagonal().sum())
            excess=np.array([expectation(n,state) for n in model['nums']])-baseline_density
            motion[label].append(dict(time=time,norm=float(np.linalg.norm(state)),site_excess_number=excess.tolist(),charge=float(p['charge']*sum(excess)),velocity=expectation(model['V'],state),current=[expectation(j,state) for j in model['J']],spin=[expectation(s,state) for s in model['S']],S2=expectation(model['S2'],state),electric_energy=expectation(model['electric'],state)))
    return dict(parameters=p,dimensions={str(n):[len(b['e']) for b in items] for n,items in all_sectors.items()},neutral_E=eg,neutral_degeneracy=len(refs),raw=raw,
                geometry={key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in model['geometry'].items() if key!='routes'},
                route_lines=[r['line'].tolist() for r in model['geometry']['routes']],source_spectra=spectra,occupied_G=greens,source_current_vertices=vertices,
                pair_feedback=dict(all_poles=pair,retarded=enc(pair_ret),sum_bright_strength=float(sum(np.trace(np.array(a['residue']['real'])) for a in pair))),
                selected_lowest_m0_sourcebright=candidate,all_sourcebright_lower_charged_branches=lower,physical_EM=em,physical_spatial_source_motion=motion,
                limits=['Finite periodic T1 matter with noncoplanar routes in a T3 field box, not free three-dimensional COM.','Two exactly orthogonal transverse T3 plane-wave directions at the SAME common q, but global coordinates and finite box, not asymptotic photons.','All CAR occupations and all physical frame irreps retained; no initial jhalf projection. Higher angular frame sectors beyond raw complete physical N0..4 representation are not explored.','Frame/CAR/colour/centre and couplings remain inputs. Scalar interchannel transfer is a geometric hopping input, not a derived mass.','Every source-bright lower branch is retained; no gap, SO3 or rest admission gate. No inertia from dark floors or finite flux.','Own positive-frequency global legs and both virtual time orders/contact are finite response strengths, not an S matrix.']),states


if __name__=='__main__':
    p=json.loads(Path(sys.argv[1]).read_text());result,states=run(p);stem=sys.argv[2]
    Path(stem+'_result.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(stem+'_states.npz',**states)
    print(json.dumps(dict(neutral_E=result['neutral_E'],dimensions=result['dimensions'],parity_H=result['raw']['parity_H'],current_norms=result['raw']['physical_current_norms'],candidate=result['selected_lowest_m0_sourcebright'],lower_bright_count=len(result['all_sourcebright_lower_charged_branches']),contrasts=result['physical_EM']['polarization_contrasts'])))
