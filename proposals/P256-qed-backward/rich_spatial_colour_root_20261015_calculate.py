"""Actual spatial graph source, current, magnetic and inclusive-density consumer.
Snapshots exact factorized H before eigensolvers and states before resolvents.
"""
from __future__ import annotations
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh,expm_multiply,gmres,LinearOperator
from rich_spatial_colour_root_20261015_model import (
    build,law,route_current,observable,TensorOperator,clean,spatial_source_terms,string)


def norm(x):
    return float(sparse.linalg.norm(x)) if sparse.issparse(x) else float(np.linalg.norm(x))


def mean(state,op):return float(np.vdot(state,op@state).real)

def cjson(value):return dict(real=np.asarray(value).real.tolist(),imag=np.asarray(value).imag.tolist())


def save_matrix(stored,name,matrix):
    matrix=sparse.csr_matrix(matrix)
    for key in ['data','indices','indptr']:stored[name+'_'+key]=getattr(matrix,key)
    stored[name+'_shape']=matrix.shape


def save_tensor(stored,name,operator):
    save_matrix(stored,name+'_one',operator.one);save_matrix(stored,name+'_phi',operator.phi)
    stored[name+'_field_energy']=int(operator.field_energy)


def lowest(h,count,p):
    k=min(count,h.shape[0]-2);v0=np.random.default_rng(222).normal(size=h.shape[0])
    e,v=eigsh(h,k=k,which='SA',tol=p['eigen_tolerance'],maxiter=24000,ncv=min(h.shape[0],max(4*k,100)),v0=v0)
    u,s,_=np.linalg.svd(v,full_matrices=False);basis=u[:,s>1e-10*s[0]]
    small=basis.conj().T@(h@basis);e,w=np.linalg.eigh((small+small.conj().T)/2)
    return e,basis@w


def lifted_body(body,scalar):
    return TensorOperator(body,sparse.csr_matrix(body.shape,dtype=complex),scalar)


def full_source_strength(reference,neutral,model,site,kind,spin,removal):
    terms=spatial_source_terms(site,kind,spin,removal);targets=set();entries=[]
    for j,mask in enumerate(neutral['masks']):
        for operations,amplitude in terms:
            step=string(mask,operations)
            if step is not None:targets.add(step[0]);entries.append((step[0],j,amplitude*step[1]))
    targets=sorted(targets);lookup={mask:i for i,mask in enumerate(targets)}
    raw=sparse.csr_matrix(([v for m,j,v in entries],([lookup[m] for m,j,v in entries],[j for m,j,v in entries])),shape=(len(targets),len(neutral['masks'])))
    tensor=reference.reshape(neutral['dimension'],-1);full=neutral['P']@tensor
    image=sparse.kron(raw,sparse.eye(19),format='csr')@full
    return dict(full_unprojected_strength=norm(image)**2,target_raw_masks=len(targets),all_original_mirror_terms=2 if kind=='lepton' else 8)


def dynamics(state,h,times,operators,receivers):
    current=state.copy();previous=0.;states=[]
    for t in times:
        if t!=previous:
            delta=t-previous;current=expm_multiply((-1j*delta)*h,current,traceA=-1j*delta*h.trace())
        states.append(current.copy());previous=t
    states=np.column_stack(states)
    data={key:[mean(states[:,i],op) for i in range(len(times))] for key,op in operators.items()}
    capture={key:[float(abs(np.vdot(receiver,states[:,i]))**2) for i in range(len(times))] for key,receiver in receivers.items()}
    return states,data,capture


def number_povm(state,op,model):
    # Trace orientation/phi only after applying each exact compressed physical effect.
    x=state.reshape(op['Q'].shape[0],-1);out={}
    for name,effects in op['povm']['effects'].items():
        probabilities=[float(np.sum(x.conj()*(effect@x)).real) for effect in effects]
        out[name]=dict(probabilities=probabilities,sum=float(sum(probabilities)),minimum=float(min(probabilities)))
    return out


def ward_operators(h,op,rotor,scalar,p,g):
    rho=clean(sparse.kron(op['rho0'],sparse.eye(10),format='csr'))
    routes=[route_current(op,rotor,scalar,p,g,e) for e in range(2)]
    one=clean(1j*(h.one@rho-rho@h.one)+sum(j.one for j in routes))
    phi=clean(1j*(h.phi@rho-rho@h.phi)+sum(j.phi for j in routes))
    defect=TensorOperator(one,phi,scalar)
    return defect,routes,dict(local_continuity_one_body_norm=norm(one),local_continuity_phi_body_norm=norm(phi),
        meaning='Raw graph identity dot rho0=-sum_e J_e. The BAND-pair projection does not commute physical-site density, so these measured defects must not be suppressed or called exact projected Ward conservation.')


def solve_resolvent(h,energy,rhs,w,p):
    z=w+energy+1j*p['eta'];diagonal=z-h.diagonal()
    matrix=LinearOperator(h.shape,matvec=lambda v:z*v-h@v,rmatvec=lambda v:z.conjugate()*v-h@v,dtype=complex)
    preconditioner=LinearOperator(h.shape,matvec=lambda v:v/diagonal,dtype=complex)
    iterations=[0]
    def callback(value):iterations[0]+=1
    solution,info=gmres(matrix,rhs,M=preconditioner,rtol=p['resolvent_tolerance'],atol=p['resolvent_tolerance']*1e-2,
        restart=p['gmres_restart'],maxiter=p['gmres_cycles'],callback=callback,callback_type='pr_norm')
    residual=norm(matrix@solution-rhs)
    return solution,dict(info=int(info),iterations=iterations[0],absolute_residual=residual,relative_residual=residual/max(norm(rhs),1e-30))


def response_tensor(h,currents,contacts,state,p):
    energy=mean(state,h);images=[j@state for j in currents];frequencies=p['frequencies'];greens={};diagnostics={}
    for w in frequencies:
        matrix=np.zeros((3,3),complex);details=[]
        for j,rhs in enumerate(images):
            solution,status=solve_resolvent(h,energy,rhs,w,p);details.append(status)
            for i,left in enumerate(images):matrix[i,j]=np.vdot(left,solution)
        greens[w]=matrix;diagnostics[str(w)]=details
    chi=[]
    for w in frequencies:chi.append(greens[w]+greens[-w].conj())
    contact=np.array([[mean(state,contacts(i,j)) for j in range(3)] for i in range(3)])
    return dict(frequencies=frequencies,eta=p['eta'],retained_virtual_dimension=h.shape[0],reference_energy=energy,
        retarded=cjson(chi),physical_contact=contact.tolist(),kernel_with_contact=cjson(np.array(chi)+contact),
        current_Gram=cjson([[np.vdot(a,b) for b in images] for a in images]),solves=diagnostics,
        scope='Three Cartesian physical vector-potential currents; ALL retained virtuals in full tensor-aware linear systems, no low-eigenstate cutoff. Second signed retarded order is entrywise conjugate G_ij(-w), exchanging the spectral current order. Nonconverged solve info/residual is retained, never replaced by a low-pole/contact proxy.')


def spin_component_response(h,current,contact,state,p):
    image=current@state;energy=mean(state,h);greens={};details={}
    for w in p['frequencies']:
        solution,status=solve_resolvent(h,energy,image,w,p)
        greens[w]=np.vdot(image,solution);details[str(w)]=status
    chi=[greens[w]+greens[-w].conjugate() for w in p['frequencies']]
    return dict(component='physical Cartesian I_z',retained_virtual_dimension=h.shape[0],
        frequencies=p['frequencies'],retarded=cjson(chi),physical_contact=mean(state,contact),solves=details)


def main():
    inp,out=map(Path,sys.argv[1:3]);p=json.loads(inp.read_text());started=time.time();model=build(p)
    scalar=model['scalar'];rotor=model['rotor'];g=model['geometry'];nf=len(scalar['grid']);nbrot=10
    stem=str(out).removesuffix('_result.json');snapshot={};laws={};ops=model['operators'];sector_results={};observables={}
    snapshot['scalar_H']=scalar['H'];snapshot['scalar_grid']=scalar['grid'];snapshot['rotor_R']=rotor['R'];snapshot['rotor_weights']=rotor['weights'];snapshot['rotor_basis']=rotor['basis']
    for name,sector in model['sectors'].items():
        op=ops[name];h=law(op,rotor,scalar,p,g);laws[name]=h
        save_tensor(snapshot,name+'_H0',h);save_matrix(snapshot,name+'_P',sector['P']);snapshot[name+'_masks']=np.array(sector['masks'],np.uint64)
        snapshot[name+'_initial_Ritz_seed']=np.random.default_rng(222).normal(size=h.shape[0])
        for edge,channels in enumerate(op['hops']):
            for channel,(q,matrices) in enumerate(channels):
                for spin,matrix in enumerate(matrices):save_matrix(snapshot,f'{name}_hop_{edge}_{channel}_{spin}',matrix)
        for key,source in model['sources'].get(name,{}).items():save_matrix(snapshot,name+'_source_'+key,source)
        obs={key:observable(op[key],rotor,scalar) for key in ['rho0','rho1','N','Q','S2','positive','C','B','PW_boundary','matter_octet','tree_C','Q1_C','electric_cross']}
        obs['J2']=lifted_body(op['J2'],scalar);obs['L2']=observable(sparse.eye(sector['dimension']),rotor,scalar,rotor['L2'])
        obs['orientation_boundary']=observable(sparse.eye(sector['dimension']),rotor,scalar,rotor['boundary'])
        z=sparse.csr_matrix((sector['dimension']*10,)*2,dtype=complex)
        obs['phi']=TensorOperator(z,sparse.eye(z.shape[0]),scalar)
        # phi2 and DVR boundary are direct tensor observables, not modified laws.
        obs['phi2']=LinearOperator(h.shape,matvec=lambda v,n=sector['dimension']: (v.reshape(n*10,nf)*scalar['grid'][None,:]**2).ravel(),dtype=complex)
        obs['scalar_boundary']=LinearOperator(h.shape,matvec=lambda v,n=sector['dimension']: (v.reshape(n*10,nf)*np.diag(scalar['boundary'])[None,:]).ravel(),dtype=complex)
        for k in range(3):
            obs['I'+str(k)]=law(op,rotor,scalar,p,g,derivative=(k,))
            obs['mu'+str(k)]=law(op,rotor,scalar,p,g,derivative=(3+k,))
            zero=sparse.csr_matrix(z.shape,dtype=complex)
            obs['Irot'+str(k)]=TensorOperator(-op['connection']['linear'][k],zero,scalar)
            obs['murot'+str(k)]=TensorOperator(-op['connection']['linear'][3+k],zero,scalar)
        defect,routes,ward=ward_operators(h,op,rotor,scalar,p,g)
        obs['route0']=routes[0];obs['route1']=routes[1]
        obs['density_transport_current']=TensorOperator(routes[0].one+routes[1].one,routes[0].phi+routes[1].phi,scalar)
        obs['circulation_current']=TensorOperator((routes[1].one-routes[0].one)/2,(routes[1].phi-routes[0].phi)/2,scalar)
        observables[name]=obs
        rotation=dict(one=[norm(j@h.one-h.one@j) for j in op['J']],phi=[norm(j@h.phi-h.phi@j) for j in op['J']])
        sector_results[name]=dict(dimension=h.shape[0],matter_dimension=sector['dimension'],raw_CAR_masks=len(sector['masks']),
            gauss_error=sector['gauge_error'],local_G0_error=op['local_G0_error'],local_G1_identity_error=op['local_G1_identity_error'],
            graph_Gauss_derivation=op['local_Gauss'],electric_decomposition_error=op['electric_decomposition_error'],
            electric_intermediate_raw_masks=op['intermediate_raw_masks'],raw_hop_Gauss_errors=op['raw_hop_Gauss_image_errors'],
            spatial_POVM_closure=op['povm']['closure'],rotation=rotation,local_continuity_operator_defect=ward)
        save_tensor(snapshot,name+'_continuity_defect',defect)
    np.savez_compressed(stem+'_operators.npz',**snapshot)
    preflight_C={}
    for name,anti in [('neutral','neutral'),('baryon','antibaryon'),('lepton','antilepton')]:
        cb=model['C'][name][::nf,::nf]
        preflight_C[name]=dict(one=norm(laws[anti].one@cb-cb@laws[name].one),phi=norm(laws[anti].phi@cb-cb@laws[name].phi))
    Path(stem+'_preflight.json').write_text(json.dumps(dict(stage='EXACT_H_OPERATOR_SNAPSHOT_BEFORE_HEAVY_DIAGNOSTICS',parameters=p,
        dimensions={name:h.shape[0] for name,h in laws.items()},graph=dict(vertices=g['vertices'].tolist(),paths=[x.tolist() for x in g['paths']],lengths=g['lengths'].tolist(),areas=g['areas'].tolist()),
        orientation_Gram_error=rotor['gram_error'],C_H0=preflight_C,local_Gauss={name:s['gauge_error'] for name,s in model['sectors'].items()},peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
    del snapshot
    spectra={};state_arrays={}
    for name,h in laws.items():
        e,v=lowest(h,p['low_eigenstates'],p);spectra[name]=(e,v);state_arrays[name+'_low_energies']=e;state_arrays[name+'_low_vectors']=v
        result=sector_results[name];obs=observables[name]
        result.update(ground_energy=float(e[0]),ground={key:mean(v[:,0],operator) for key,operator in obs.items()},
            low_energies=e.tolist(),low_J2=[mean(v[:,i],obs['J2']) for i in range(len(e))],
            eigen_residuals=[norm(h@v[:,i]-e[i]*v[:,i]) for i in range(len(e))],low_orthogonality=norm(v.conj().T@v-np.eye(len(e))))
    en,vn=spectra['neutral'];reference=vn[:,0];state_arrays['reference']=reference
    source_results={};source_states={};pole_states={};raw_receivers={}
    placements={'0':0,'1':1,'split1':(0,0,1),'split2':(0,1,1)}
    for name,sources in model['sources'].items():
        e,v=spectra[name];source_results[name]={};kind='lepton' if 'lepton' in name else 'baryon';removal=name.startswith('anti')
        raw_receivers[name]={}
        for key,source in sources.items():
            placement,spin_text=key.split(':');spin=float(spin_text);state=source@reference;strength=norm(state)**2;weights=abs(v.conj().T@state)**2
            bright=np.flatnonzero(weights>p['bright_tolerance']*max(strength,1e-30))
            if not len(bright):raise RuntimeError('No source-bright low pole: '+name+'/'+key)
            energy=float(e[bright[0]]);cluster=abs(e-energy)<p['degeneracy_tolerance'];pole=v[:,cluster]@(v[:,cluster].conj().T@state);pole/=norm(pole)
            full=full_source_strength(reference,model['sectors']['neutral'],model,placements[placement],kind,spin,removal)
            source_results[name][key]=dict(raw_strength=strength,full_CAR=full,projection_strength_loss=full['full_unprojected_strength']-strength,
                preparation_cost=mean(state,laws[name])/strength-en[0],lowest_bright_gap=energy-en[0],raw_pole_residue=float(weights[cluster].sum()),
                conditional_pole_fraction=float(weights[cluster].sum()/strength),uncomputed_high_source_strength=strength-float(weights.sum()),
                source_J2_residual=norm(observables[name]['J2']@state-spin*(spin+1)*state),pole_J2=mean(pole,observables[name]['J2']),
                pole_mu=[mean(pole,observables[name]['mu'+str(k)]) for k in range(3)],pole_N=mean(pole,observables[name]['N']),pole_Q=mean(pole,observables[name]['Q']),
                pole_eigen_residual=norm(laws[name]@pole-energy*pole))
            source_states[name,key]=state;pole_states[name,key]=pole;state_arrays[name+'_source_state_'+key]=state;state_arrays[name+'_pole_'+key]=pole
            if spin==.5:raw_receivers[name][placement]=state/norm(state)
        labels=list(raw_receivers[name]);vectors=np.column_stack([raw_receivers[name][key] for key in labels])
        source_results[name]['spatial_preparation_Gram']=dict(labels=labels,matrix=cjson(vectors.conj().T@vectors),
            meaning='Site-labelled source preparations are NOT exclusive position projectors; occupied-sea overlap at t0 is retained, not called transfer.')
    np.savez_compressed(stem+'_states.npz',**state_arrays)
    cchecks={};cn=model['C']['neutral'];nbody=cn.shape[0]//nf;ix=np.arange(0,cn.shape[0],nf);cnb=cn[ix][:,ix]
    cchecks['neutral']=dict(H_one=norm(laws['neutral'].one@cnb-cnb@laws['neutral'].one),H_phi=norm(laws['neutral'].phi@cnb-cnb@laws['neutral'].phi),
        reference_overlap=float(abs(np.vdot(reference,cn@reference))**2))
    for name,anti in [('baryon','antibaryon'),('lepton','antilepton')]:
        c=model['C'][name];ix=np.arange(0,c.shape[0],nf);cb=c[ix][:,ix];h=laws[name];ha=laws[anti]
        probe=np.array(p['A_pulse']+p['B_pulse']);hp=law(ops[name],rotor,scalar,p,g,probe);ham=law(ops[anti],rotor,scalar,p,g,-probe)
        cchecks[name]=dict(H_one=norm(ha.one@cb-cb@h.one),H_phi=norm(ha.phi@cb-cb@h.phi),
            finite_field_one=norm(ham.one@cb-cb@hp.one),finite_field_phi=norm(ham.phi@cb-cb@hp.phi),
            local_rho0=norm(ops[anti]['rho0']@cb[::10,::10]+cb[::10,::10]@ops[name]['rho0']),
            direct_sources={key:norm(model['sources'][anti][key]@cn-c@source) for key,source in model['sources'][name].items()})
        # Derivative C checks are factorized, not fabricated anti responses.
        current_checks=[]
        for k in range(6):
            jp=law(ops[name],rotor,scalar,p,g,derivative=(k,));ja=law(ops[anti],rotor,scalar,p,g,derivative=(k,))
            current_checks.append(dict(one=norm(ja.one@cb+cb@jp.one),phi=norm(ja.phi@cb+cb@jp.phi)))
        cchecks[name]['physical_current_and_moment']=current_checks
    reactions={};responses={};form_factors={};diagnostics={};event_arrays={};times=np.array(p['times'])
    for name in ['neutral','baryon','lepton','antibaryon','antilepton']:
        op=ops[name];h=laws[name];obs=observables[name]
        initial=reference if name=='neutral' else raw_receivers[name]['0'];receivers={} if name=='neutral' else raw_receivers[name]
        baseline,base,capture=dynamics(initial,h,times,obs,receivers);event_arrays[name+'_baseline']=baseline
        defect,routes,ward=ward_operators(h,op,rotor,scalar,p,g)
        diagnostics[name]=dict(continuity_defect_on_preparation=norm(defect@initial),
            current_route_sum_on_preparation=norm((routes[0]@initial)+(routes[1]@initial)),
            physical_count_POVM_initial=number_povm(initial,op,model),physical_count_POVM_baseline_final=number_povm(baseline[:,-1],op,model))
        dipole_defects=[]
        for axis in range(3):
            dipole=observable(op['density_delta'],rotor,scalar,p['vertex_separation']/2*rotor['coordinate'][axis][0])
            defect_one=clean(1j*(h.one@dipole.one-dipole.one@h.one)-obs['I'+str(axis)].one)
            defect_phi=clean(1j*(h.phi@dipole.one-dipole.one@h.phi)-obs['I'+str(axis)].phi)
            dipole_defects.append(dict(on_preparation=norm(TensorOperator(defect_one,defect_phi,scalar)@initial),one_body_norm=norm(defect_one),phi_body_norm=norm(defect_phi)))
        diagnostics[name]['Cartesian_dipole_velocity_minus_current']=dipole_defects
        reactions[name]=dict(times=times.tolist(),baseline=base,source_capture_baseline=capture,
            capture_meaning='Normalized overlap with actual site/split CAR source PREPARATION states is a two-outcome rank1 receiver, not exclusive baryon presence. Initial Gram overlap is explicitly nonzero; report time variation and count POVMs separately.')
        for label,fields in [('A',np.array(p['A_pulse']+[0.,0.,0.])),('B',np.array([0.,0.,0.]+p['B_pulse']))]:
            fields*=(-1 if name.startswith('anti') else 1)
            hp=law(op,rotor,scalar,p,g,fields);kicked=expm_multiply((-1j*p['pulse_duration'])*hp,initial,traceA=-1j*p['pulse_duration']*hp.trace())
            states,data,received=dynamics(kicked,h,times,obs,receivers);event_arrays[name+'_'+label+'_event']=states
            reactions[name][label]=dict(fields=fields.tolist(),duration=p['pulse_duration'],energy_injection=mean(kicked,h)-mean(initial,h),
                norm_error=abs(norm(kicked)-1),pulse=data,excess={key:(np.array(value)-base[key]).tolist() for key,value in data.items()},
                source_capture=received,physical_count_POVM_final=number_povm(states[:,-1],op,model))
        unitary=expm(-1j*p['field_kick']*scalar['P']);field_state=(initial.reshape(h.nb,nf)@unitary.T).ravel()
        states,data,received=dynamics(field_state,h,times,obs,receivers);event_arrays[name+'_field_event']=states
        reactions[name]['field_only']=dict(energy_injection=mean(field_state,h)-mean(initial,h),
            initial_matter_geometry_changes={key:mean(field_state,obs[key])-mean(initial,obs[key]) for key in ['rho0','rho1','S2','J2','L2','C','tree_C','orientation_boundary']},
            excess={key:(np.array(value)-base[key]).tolist() for key,value in data.items()},source_capture=received,
            physical_count_POVM_final=number_povm(states[:,-1],op,model),preparation='Unitary on SAME quantum phi only; actual occupied matter/loop/material orientation untouched initially. SAME-state H0 baseline, no reset.')
        # Actual Fourier charge density and magnetic moment from current, never Clifford labels.
        pole=reference if name=='neutral' else pole_states[name,'0:0.5'];rho=[]
        for wavevector in p['wavevectors']:
            kvec=np.array(wavevector);values=[]
            for site in range(2):
                positions=np.einsum('qia,a->qi',rotor['R'],g['vertices'][site])
                factor=rotor['integrate'](np.exp(1j*(positions@kvec)));density=observable(op['rho'+str(site)],rotor,scalar,factor)
                values.append(np.vdot(pole,density@pole))
            rho.append(sum(values))
        form_factors[name]=dict(wavevectors=p['wavevectors'],raw_charge_form_factor=cjson(rho),
            total_charge=mean(pole,obs['Q']),magnetic_moment=[mean(pole,obs['mu'+str(k)]) for k in range(3)],
            meaning='Actual physical exp(ik dot R r_site) charge density; magnetic moment=-dH/dB includes path area current AND gauged mechanical rotation, not a Clifford bivector. Finite graph, not radius/infinite-volume binding.')
        if name in ['neutral','baryon','lepton']:
            currents=[obs['I'+str(k)] for k in range(3)]
            contacts=lambda i,j,op=op:law(op,rotor,scalar,p,g,derivative=(i,j))
            responses[name]=response_tensor(h,currents,contacts,pole,p)
            if name=='baryon':
                responses['baryon_threehalf_Iz']=spin_component_response(h,obs['I2'],contacts(2,2),pole_states[name,'0:1.5'],p)
    # Two material controls: remove the noncoplanar colour route; remove all primitive hops.
    controls={}
    for name in ['baryon','lepton']:
        op=ops[name];initial=raw_receivers[name]['0'];obs=observables[name]
        control={}
        for label,scale in [('cycle_path_off',(1.,0.)),('all_primitive_hops_off',(0.,0.))]:
            hc=law(op,rotor,scalar,p,g,path_scale=scale);states,data,received=dynamics(initial,hc,times,obs,raw_receivers[name])
            control[label]=dict(actual_density_history=data['rho1'],source_capture=received,
                energy_of_same_preparation=mean(initial,hc),state_final_count_POVM=number_povm(states[:,-1],op,model),
                interpretation='Same actual prepared state, changed microscopic path law. NOT a recomputed vacuum/gap. If density changes with ALL hopping off, that is exposed Galerkin local-Ward contamination, NOT physical composite hopping.')
        controls[name]=control
    np.savez_compressed(stem+'_result.npz',**event_arrays)
    separation={name:{key:source_results[name][key]['preparation_cost'] for key in ['0:0.5','split1:0.5','split2:0.5','1:0.5']} for name in ['baryon','antibaryon']}
    payload=dict(kind='NEW_SPATIAL_CONSTITUENT_COLOUR_GRAPH_PHYSICAL_CONSUMER_NOT_PHOTON_OR_PROTON',parameters=p,
        root={key:value for key,value in model['root'].items() if key not in ['cliff','Phi','Z']},
        geometry=dict(vertices=g['vertices'].tolist(),paths=[x.tolist() for x in g['paths']],lengths=g['lengths'].tolist(),areas=g['areas'].tolist(),hopping_weights=g['weights'].tolist(),
            source='theta_e=int_path A dot dl=A_uniform dot R displacement+B dot R area_e, A(r)=A_uniform+B cross r/2; charge q in charge1 units, fundamental1/3 explicit. No compact photon or holonomy-as-polarization identification.',orientation_Gram_error=rotor['gram_error']),
        sectors=sector_results,sources=source_results,C=cchecks,spatial_diagnostics=diagnostics,reactions=reactions,
        Cartesian_current_response=responses,form_factors_and_moments=form_factors,controls=controls,
        finite_colocalized_split_source_costs=separation,
        spin_gap_difference=source_results['baryon']['0:1.5']['lowest_bright_gap']-source_results['baryon']['0:0.5']['lowest_bright_gap'],
        resource=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-started),
        limitations='Original mirror legs are actual sites in a NEW spatial law, phi bond order not inherited rest mass; identity lepton paths replace TrU/3. No elementary B hopping, neutral binder, fitted spin exchange or fixed spin tag. Every hop is primitive quark/lepton CAR hopping; real colour electric cross terms from local Gauss. Band-pair truncation can violate physical-site continuity, explicitly measured and hop-off diagnosed. Quantum scalar/PW19/full-j1 SO3 retained, higher pairs/irreps/matrix Phi/Z/continuum omitted. Actual external Cartesian A and B current/contact/magnetic response, NOT quantum photons. Physical Noether is rigid material geometry+frame, not translational COM or infinite-volume proton binding. Global vacuum/all-number/real electron-proton identity not claimed.')
    out.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(dimensions={name:v['dimension'] for name,v in sector_results.items()},sources={name:{key:{k:v[k] for k in ['raw_strength','lowest_bright_gap','preparation_cost','pole_J2']} for key,v in ss.items() if key!='spatial_preparation_Gram'} for name,ss in source_results.items()},spin_gap_difference=payload['spin_gap_difference'],C=cchecks,resource=payload['resource']),indent=2))


if __name__=='__main__':main()
