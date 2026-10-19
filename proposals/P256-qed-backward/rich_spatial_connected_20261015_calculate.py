"""Actual owning G,D and unamputated connected B-Q-Q-Bdag tensor, all24 T orders."""
from typing import Any
from itertools import permutations,product
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from rich_spatial_connected_20261015_model import (build,install,canonical_sources,reference,evolve,act,adjoint_operator,field_cut_weights)
from rich_spatial_helicity_20261015_model import field_structure
ROOT=Path(__file__).resolve().parent


def pair_catalog()->Any:
    catalogs={};lookups={}
    for family in ['right','left']:
        keys={name:[] for name in ['neutral','positive','negative']};lookup={}
        for i,j in product(range(28),repeat=2):
            # All requested incoming creators have kz0; their adjoints on the
            # left are kz0 annihilators. This only removes UNUSED pair states.
            if family=='right' and any(op>=20 for op in [i,j]):continue
            if family=='left' and any(8<=op<16 for op in [i,j]):continue
            first=0 if i<4 else (-1 if i<16 else 1);second=0 if j<4 else (-1 if j<16 else 1)
            if first and second and first==second:continue
            total=first+second
            if total not in [-1,0,1]:continue
            name={-1:'negative',0:'neutral',1:'positive'}[total];lookup[i,j]=(name,len(keys[name]));keys[name].append((i,j))
        catalogs[family]=keys;lookups[family]=lookup
    return catalogs,lookups


def field_expect(model: Any,name: str,state: Any,operator: Any)->complex:
    a=state.reshape(model['sectors'][name]['n'],-1);return np.vdot(a,(operator@a.T).T)


def body_expect(model: Any,name: str,state: Any,operator: Any)->complex:
    a=state.reshape(model['sectors'][name]['n'],-1);return np.vdot(a,operator@a)


def full_action_weight(model: Any,ref: Any,name: str,operator: int,state: Any)->float:
    norm=float(np.vdot(state,state).real)
    if operator<4:
        mu=ref['field_means'][operator];q=model['field']['coordinates'][operator]
        square=model['field']['before_cut_Q_products'][operator][operator]-2*mu*q+mu*mu*sparse.eye(q.shape[0],format='csr')
        return float(field_expect(model,name,state,square).real)
    create=operator>=16;index=operator-16 if create else operator-4
    occupied=float(body_expect(model,name,state,model['occupations'][name][index]).real)
    return norm-occupied if create else occupied


def single_states(model: Any,ref: Any)->Any:
    omega=ref['omega'];groups={name:[] for name in ['neutral','positive','negative']};indices={}
    for op in range(28):
        name,v=act(model,'neutral',op,omega);indices[op]=(name,len(groups[name]));groups[name].append(v)
    return {name:np.column_stack(v) for name,v in groups.items()},indices


def pair_states(model: Any,ref: Any,keys: Any,first: Any,first_indices: Any)->Any:
    states={};audits={}
    for name,catalog in keys.items():
        columns=[];data=[]
        for i,j in catalog:
            source,index=first_indices[i];before=first[source][:,index];target,after=act(model,source,j,before)
            if target!=name:raise ValueError('Incorrect actual CAR intermediate sector')
            full=full_action_weight(model,ref,source,j,before);retained=float(np.vdot(after,after).real)
            columns.append(after);data.append([float(np.vdot(before,before).real),full,retained,full-retained])
        states[name]=np.column_stack(columns);audits[name]=np.array(data)
    return states,audits


def two_points(model: Any,ref: Any,times: Any,singles: Any)->Any:
    propagated={};mplus=[];mminus=[];mfield=[]
    for t in times:
        data={name:evolve(model,ref,name,v,float(t)) for name,v in singles.items()};propagated[round(float(t),12)]=data
        mplus.append(singles['positive'].conj().T@data['positive']);mminus.append(singles['negative'].conj().T@data['negative']);mfield.append(singles['neutral'].conj().T@data['neutral'])
    return dict(propagated=propagated,positive=np.array(mplus),negative=np.array(mminus),field=np.array(mfield),times=np.array(times))


def four_point(model: Any,ref: Any,spacing: float,points: Any,keys: Any,lookup: Any,first_indices: Any,singles: Any,orders: Any)->Any:
    forward=points['propagated'][round(float(spacing),12)]
    backward={name:evolve(model,ref,name,v,spacing,True) for name,v in singles.items()}
    right,audit_right=pair_states(model,ref,keys['right'],forward,first_indices);left,audit_left=pair_states(model,ref,keys['left'],backward,first_indices)
    middle={name:evolve(model,ref,name,v,spacing) for name,v in right.items()};grams={name:left[name].conj().T@v for name,v in middle.items()}
    shape=(len(orders),3,4,4,4,4);full=np.empty(shape,complex);disconnected=np.empty(shape,complex)
    time_lookup={round(float(t),12):i for i,t in enumerate(points['times'])}
    for order_index,order in enumerate(orders):
        slots=np.argsort(order);fermion_sign=1 if slots[0]>slots[3] else -1
        bt=time_lookup[round(abs(int(slots[0])-int(slots[3]))*spacing,12)];qt=time_lookup[round(abs(int(slots[1])-int(slots[2]))*spacing,12)]
        for kout,alpha,beta,a,b in product(range(3),range(4),range(4),range(4),range(4)):
            operators=[4+4*kout+alpha,a,b,16+beta]
            rkey=(operators[order[0]],operators[order[1]]);lkey=(adjoint_operator(operators[order[3]]),adjoint_operator(operators[order[2]]))
            rn,ri=lookup['right'][rkey];ln,li=lookup['left'][lkey]
            if rn!=ln:raise ValueError('Unbalanced physical four-point charge sectors')
            full[order_index,kout,alpha,beta,a,b]=fermion_sign*grams[rn][li,ri]
            fermion=points['positive'][bt,4*kout+alpha,beta] if fermion_sign==1 else -points['negative'][bt,beta,4*kout+alpha]
            boson=points['field'][qt,a,b] if slots[1]>slots[2] else points['field'][qt,b,a]
            disconnected[order_index,kout,alpha,beta,a,b]=fermion*boson
    return full,disconnected,dict(forward= audit_right,adjoint_bra=audit_left,middle_norm2={name:np.sum(abs(v)**2,axis=0) for name,v in middle.items()})


def fourier_legs(tensor: Any)->Any:
    outgoing=np.zeros((2,2,4),complex);incoming=np.zeros_like(outgoing)
    for pol,signindex in product(range(2),range(2)):
        eta=[-1,1][signindex];outgoing[pol,signindex,2*pol]=1/np.sqrt(2);outgoing[pol,signindex,2*pol+1]=-1j*eta/np.sqrt(2)
        incoming[pol,signindex,2*pol]=1/np.sqrt(2);incoming[pol,signindex,2*pol+1]=1j*eta/np.sqrt(2)
    return np.einsum('pksrab,lda,meb->pksrlmde',tensor,outgoing,incoming,optimize=True)


def two_point_frequency(points: Any,frequencies: Any)->Any:
    times=points['times'];plus=points['positive'];minus=points['negative'];field=points['field'];g=[];d=[];gr=[];dr=[]
    for omega in frequencies:
        phase=np.exp(1j*omega*times)[:,None,None];negative_phase=np.exp(-1j*omega*times)[:,None,None]
        g.append(np.trapezoid(phase*(-1j*plus),x=times,axis=0)+np.trapezoid(negative_phase*(1j*minus.transpose(0,2,1)),x=times,axis=0))
        d.append(np.trapezoid(phase*(-1j*field),x=times,axis=0)+np.trapezoid(negative_phase*(-1j*field.transpose(0,2,1)),x=times,axis=0))
        gr.append(np.trapezoid(phase*(-1j*(plus+minus.conj())),x=times,axis=0));dr.append(np.trapezoid(phase*(-1j*(field-field.conj())),x=times,axis=0))
    return dict(G_T_window=np.array(g),D_T_window=np.array(d),G_R_window=np.array(gr),D_R_window=np.array(dr),
        convention='G_T=-i<T BBdag>, D_T=-i<T deltaQ deltaQ>. Negative-time fermion uses MINUS reversed removal order. G_R uses anticommutator, D_R commutator. Finite time window, not exact infinite resolvents.')


def spectral_window(model: Any,ref: Any,singles: Any,count: int)->Any:
    result={}
    for name in ['neutral','positive','negative']:
        sec=model['sectors'][name];v0=np.cos(.37*np.arange(sec['h'].shape[0]))+1j*np.sin(.19*np.arange(sec['h'].shape[0]));e,u=eigsh(sec['h'],k=count,which='SA',tol=1e-9,maxiter=5000,v0=v0)
        ordering=np.argsort(e);e=e[ordering];u=u[:,ordering];vectors=singles[name];overlaps=u.conj().T@vectors;groups=[];start=0
        while start<len(e):
            end=start+1
            while end<len(e) and abs(e[end]-e[start])<1e-7:end+=1
            residue=overlaps[start:end].conj().T@overlaps[start:end]
            groups.append(dict(energy_difference=float(np.mean(e[start:end])-ref['energy']),multiplicity=end-start,residue_real=residue.real.tolist(),residue_imag=residue.imag.tolist(),trace_weight=float(np.trace(residue).real),last_group_may_be_window_truncated=end==len(e)))
            start=end
        result[name]=dict(groups=groups,raw_source_norm2=float(np.linalg.norm(vectors)**2),retained_low_window_weight=float(np.linalg.norm(overlaps)**2),unresolved_high_weight=float(np.linalg.norm(vectors)**2-np.linalg.norm(overlaps)**2),
            meaning='Same-H low-eigenwindow diagnostic ONLY; complete degenerate residues except potentially final window edge. Full time G/D and connected subtraction do NOT use this truncated window.')
    return result


def finite_cut_eom(model: Any,ref: Any,singles: Any)->Any:
    structure=field_structure(model,model['law']);np_=model['field']['nphi'];output={}
    for name,state in [('neutral',ref['omega']),('positive',singles['positive'][:,0]),('negative',singles['negative'][:,0])]:
        sec=model['sectors'][name];a=state.reshape(sec['n'],-1);hstate=sec['h']@state;records=[]
        def field_action(k: Any,v: Any)->Any:return (k@v.reshape(sec['n'],-1).T).T.ravel()
        def term_action(terms: Any)->Any:
            out=np.zeros_like(a)
            for f,k in zip(sec['matrices'],terms):
                if k.nnz:out+=(k@(f@a).T).T
            return out.ravel()
        for axis in range(4):
            q=model['field']['coordinates'][axis];p=sparse.kron(sparse.eye(np_),structure['p'][axis],format='csr');qstate=field_action(q,state);pstate=field_action(p,state);current=-term_action(model['field']['derivatives'][axis])
            eq=1j*(sec['h']@qstate-field_action(q,hstate))-pstate/(model['law']['volume']/2)
            ep=1j*(sec['h']@pstate-field_action(p,hstate))-current
            records.append(dict(Q_velocity_defect_norm=float(np.linalg.norm(eq)),Pi_current_defect_norm=float(np.linalg.norm(ep)),raw_current_expectation=[float(np.vdot(state,current).real),float(np.vdot(state,current).imag)]))
        output[name]=dict(raw_state_norm2=float(np.vdot(state,state).real),profiles=records)
    return output


def main()->None:
    started=time.time();inp=Path(sys.argv[1]);config:Any=json.loads(inp.read_text());output=Path(sys.argv[2]);law:Any=json.loads((ROOT/config['law_input']).read_text());orth:Any=json.loads((ROOT/config['orthogonal_input']).read_text());config.update({k:orth[k] for k in ['local_singular_relative_cut','tensor_singular_relative_cut']})
    reconstruction=np.load(ROOT/config['reconstruction']);model=build(config,law,reconstruction);sources=canonical_sources(model);field_cut_weights(model)
    keys,lookup=pair_catalog();orders=list(permutations(range(4)));spacings=np.array(config['time_spacings']);times=np.array(sorted(set(round(float(s)*m,12) for s in spacings for m in range(4))))
    car_max=0.
    for s in range(12):
        for t in range(12):car=sources['positive'][s].conj().T@sources['positive'][t]+sources['negative'][t].conj().T@sources['negative'][s]-(np.eye(27) if s==t else 0);car_max=max(car_max,float(np.linalg.norm(car)))
    arrays={};reports={};cached_neutral=reconstruction['neutral_ground'];state_snapshot={}
    for decoupled,label in [(False,'coupled'),(True,'EM_off_factorized')]:
        for sec in model['sectors'].values():install(sec,model,decoupled)
        ref=reference(model,cached_neutral,decoupled);single,first_indices=single_states(model,ref)
        state_snapshot[label+'_reference']=ref['omega'];state_snapshot[label+'_positive_sources']=single['positive'];state_snapshot[label+'_negative_sources']=single['negative'];state_snapshot[label+'_field_sources']=single['neutral']
        np.savez_compressed(output.with_name(output.name.replace('_result.json','_states.npz')),**state_snapshot)
        output.with_name(output.name.replace('_result.json','_states.json')).write_text(json.dumps(dict(kind='OWNING_REFERENCE_AND_ALL_RAW_SOURCE_STATES_BEFORE_EXPENSIVE_RESPONSE',current_arm=label,reference_energy=ref['energy'],H_state_residual=ref['H_state_residual'],reference_semantics=ref['reference_semantics'],basis_bindings=model['basis_bindings'],field_means=ref['field_means'],source_state_norms={name:np.sum(abs(v)**2,axis=0).tolist() for name,v in single.items()},input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest()),indent=2)+'\n')
        two=two_points(model,ref,times,single);frequency=two_point_frequency(two,config['two_point_frequencies'])
        for name in ['positive','negative','field']:arrays[label+'_two_point_'+name]=two[name]
        for name,value in frequency.items():
            if isinstance(value,np.ndarray):arrays[label+'_'+name]=value
        sourceweights={}
        for mode in range(12):
            occ=float(body_expect(model,'neutral',ref['omega'],model['occupations']['neutral'][mode]).real);pn=float(np.vdot(single['positive'][:,mode],single['positive'][:,mode]).real);mn=float(np.vdot(single['negative'][:,mode],single['negative'][:,mode]).real)
            sourceweights[str(mode)]=dict(raw_addition=1-occ,raw_removal=occ,retained_addition=pn,retained_removal=mn,omitted_addition=1-occ-pn,omitted_removal=occ-mn)
        before=np.array([[field_expect(model,'neutral',ref['omega'],q) for q in row] for row in model['field']['before_cut_Q_products']]);retained=single['neutral'].conj().T@single['neutral']
        before_centered=before-np.outer(ref['field_means'],ref['field_means'])
        spectra=spectral_window(model,ref,single,config['spectral_eigenwindow']) if not decoupled else dict(meaning='Control reference/evolution uses its own factorized matter-Phi and EM eigensystems; no coupled-state reuse.',factorization_residual=ref['factor']['bare_tensor_reconstruction_norm'])
        eom=finite_cut_eom(model,ref,single) if not decoupled else None
        full_list=[];disc_list=[];connected_list=[];norm_reports=[]
        for spacing in spacings:
            full,disc,audits=four_point(model,ref,float(spacing),two,keys,lookup,first_indices,single,orders);connected=full-disc
            full_list.append(fourier_legs(full));disc_list.append(fourier_legs(disc));connected_list.append(fourier_legs(connected))
            for direction,groups in audits.items():
                for name,data in groups.items():arrays[f'{label}_tau{spacing}_{direction}_{name}']=data
            norm_reports.append(dict(spacing=float(spacing),full_centered_norm=float(np.linalg.norm(full)),disconnected_norm=float(np.linalg.norm(disc)),connected_norm=float(np.linalg.norm(connected)),maximum_connected_magnitude=float(np.max(abs(connected)))))
            arrays[label+'_four_full_centered']=np.array(full_list);arrays[label+'_four_disconnected']=np.array(disc_list);arrays[label+'_four_connected']=np.array(connected_list)
            np.savez_compressed(output.with_name(output.name.replace('_result.json','_checkpoint.npz')),**arrays)
            output.with_name(output.name.replace('_result.json','_checkpoint.json')).write_text(json.dumps(dict(kind='ACTUAL_COMPLETED_TIME_SURFACES_BEFORE_REMAINING_EXPENSIVE_STAGES',current_arm=label,completed_spacings=[float(s) for s in spacings[:len(full_list)]],basis_bindings=model['basis_bindings'],norms=norm_reports,elapsed_seconds=time.time()-started),indent=2)+'\n')
            print('Completed connected time surface',label,float(spacing),norm_reports[-1],flush=True)
        tensor=np.array(connected_list);ray=[]
        for frequency_tuple in config['four_point_frequency_tuples']:
            weights=[]
            for order in orders:slots=np.argsort(order);weights.append(float(np.dot(frequency_tuple,slots)))
            phase=np.exp(1j*spacings[:,None]*np.array(weights)[None,:]);ray.append(np.trapezoid(phase.reshape(len(spacings),24,*([1]*7))*tensor,x=spacings,axis=0))
        arrays[label+'_four_time_ray_frequency']=np.array(ray)
        allowed=[];forbidden=[]
        for out_sign,in_sign in product(range(2),repeat=2):
            expected=([-1,1][in_sign]-[-1,1][out_sign])%3
            allowed.append(tensor[:,:,expected,:,:,:,:,out_sign,in_sign].ravel())
            for other in range(3):
                if other!=expected:forbidden.append(tensor[:,:,other,:,:,:,:,out_sign,in_sign].ravel())
        causal=orders.index((3,2,1,0));backward=tensor[-1,causal,2,:,:,:,:,0,1]
        reports[label]=dict(reference=dict(energy=ref['energy'],H_state_residual=ref['H_state_residual'],semantics=ref['reference_semantics']),field_means=ref['field_means'],source_weights=sourceweights,
            raw_G_CAR_ground_defect=float(np.linalg.norm(single['positive'].conj().T@single['positive']+(single['negative'].conj().T@single['negative']).T-np.eye(12))),
            field_before_cut_uncentered_Gram_real=before.real.tolist(),field_before_cut_uncentered_Gram_imag=before.imag.tolist(),field_centered_retained_Gram_real=retained.real.tolist(),field_centered_retained_Gram_imag=retained.imag.tolist(),
            field_before_cut_centered_Gram_real=before_centered.real.tolist(),field_before_cut_centered_Gram_imag=before_centered.imag.tolist(),field_omitted_Gram_real=(before_centered-retained).real.tolist(),field_omitted_Gram_imag=(before_centered-retained).imag.tolist(),
            finite_cut_EOM=eom,spectral_window=spectra,surfaces=norm_reports,allowed_connected_Frobenius=float(np.linalg.norm(np.concatenate(allowed))),forbidden_connected_Frobenius=float(np.linalg.norm(np.concatenate(forbidden))),
            backward_allowed_causal_max_magnitude=float(np.max(abs(backward))),backward_allowed_causal_spinor_polarization_real=backward.real.tolist(),backward_allowed_causal_spinor_polarization_imag=backward.imag.tolist(),
            equal_time_dressing_norm=float(np.linalg.norm(tensor[0])),last_minus_equal_time_dynamical_variation_norm=float(np.linalg.norm(tensor[-1]-tensor[0])))
    result=dict(kind='SAME_H_OWNING_G_D_CONNECTED_CHARGED_TWO_FIELD_TENSOR_NOT_ONSHELL_COMPTON',input=config,law_input=law,basis_bindings=model['basis_bindings'],
        exact_transverse_fiber=dict(indices=config['transverse_indices'],selected_source_site=model['selected_site'],physical_momentum=[2*np.pi*config['transverse_indices'][0]/3,2*np.pi*config['transverse_indices'][1]/3],
            explanation='Exact invariant charged transverse-momentum fiber, not COM tensor internal factorization or discarded virtual modes. Neutral27 and charged162 GLOBAL nine-Fock-product TT states retain all54 occupied sea dynamics; EM q has zero transverse momentum. All four canonical spinor components and all three kz retained; hole physical momentum is opposite its annihilated canonical mode.'),
        canonical_CAR_operator_max_defect=car_max,operator_order_labels=['B_out','Q_out','Q_in','Bdag_in'],chronological_orders=[list(o) for o in orders],fermion_time_order_signs=[1 if list(o).index(0)>list(o).index(3) else -1 for o in orders],
        source_mode_labels=[dict(kz_index=k,spinor=s,mirror=s//2,primitive_frame_component=s%2) for k,s in product(range(3),range(4))],
        raw_pair_catalog={family:{name:[list(k) for k in v] for name,v in catalog.items()} for family,catalog in keys.items()},two_point_times=times.tolist(),reports=reports,
        tensor_axes=['time_spacing','chronological_order','kz_out','spinor_out','spinor_in','polarization_out','polarization_in','q_out_sign','q_in_sign'],
        array_semantics='Complex NPZ contains COMPLETE centered full, owning disconnected and connected tensors, both Green branches/field two-points, finite time/frequency data and raw unnormalized pair-action weights. No intermediate vector normalization or energy/source filtering.',
        connected_convention='C_T=<T B deltaQ_out deltaQ_in Bdag> - <T B Bdag><T deltaQ_out deltaQ_in>. Fields centered using OWN stationary neutral reference means, same constants in ALL sectors; charge-changing lower disconnected means vanish. G_T=-i fermion2point, D_T=-i centered field2point; four-point here has NO extra-i. Both source CAR time orders and all24 operator permutations retained.',
        field_leg_convention='Incoming Q_(eta_in), outgoing Q_(-eta_out), Q_(eta)=(Qcos+i eta Qsin)/sqrt2. eta signs[-1,+1], pol[x,y]. Allowed source recoil kz_out=kz_in+eta_in-eta_out mod3, kz_in0. Full tensors retain BOTH allowed and forbidden channels without filtering.',
        frequency_semantics='Two-point G_T/D_T and retarded windows are signed-time finite integrals. Four-point frequency tuples contract actual times along each ordered uniform-spacing ray, NOT exhaustive independent three-frequency vertex or LSZ amplitude.',
        controls='Complete EM-off law keeps quantum scalar coupling and has OWN exactly factorized matter-Phi x EM ground. Same general24-order consumer, not hardcoded connectedzero; previous entangled coupled state is never reused as factorization control.',
        limitations=['Unamputated finite connected correlator, not a pure one-photon scattering amplitude/cross section or isolated electron spinflip. Spinor indices are canonical mirror/frame components, not physical particle-spin outcomes.',
            'Current derivatives and momentum-carrying source/field insertions are distinct operators linked by owning dynamics/selection rules, NOT Pcm=J and no COM/internal factorization imposed.',
            'Full-H field correlator already contains all retained Peierls/contact physics; no seagull added twice, no borrowed freeG/D, no scalar/contact substitute. Explicit finite-cut EOM and full-before-cut field/source weight deficits retained.',
            'Same27/162 projected space, severe shell2/oneq/three-cell regulator and omitted-H.723842 remain; source/C4 maps~1e-10..1e-9 distinct from whole-H roundoff. Other colour-law57.38% current omission is NOT this TT law.',
            'Low-eigenwindow diagnostics do not replace full time G/D or prove photon/particle isolation; no bulk/Lorentz/freeCOM/strong/proton/Nature promotion.'],
        elapsed_seconds=time.time()-started,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    np.savez_compressed(output.with_suffix('.npz'),**arrays);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(result=str(output),reports={k:dict(allowed=v['allowed_connected_Frobenius'],forbidden=v['forbidden_connected_Frobenius'],backward=v['backward_allowed_causal_max_magnitude']) for k,v in reports.items()},elapsed=result['elapsed_seconds'],peak_resident_KiB=result['peak_resident_KiB'])),flush=True)


if __name__=='__main__':main()
