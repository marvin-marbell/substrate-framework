"""New finite-bandwidth charged/field packet consumer; frozen helpers are pure imports."""
from __future__ import annotations
import hashlib,json,resource,sys,time,gc
from pathlib import Path
from itertools import product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh,expm
from scipy.sparse.linalg import eigsh
from rich_mobile_packet_20261015_model import (geometry,field,tangent,source_sector,project,install,
    apply_field,apply_terms,expectation,translation,momentum)
ROOT=Path(__file__).resolve().parent


def digest(p: Path) -> str:return hashlib.sha256(p.read_bytes()).hexdigest()
def savejson(p: Path,x: dict) -> None:p.write_text(json.dumps(x,indent=2)+'\n')
def norm2(x: np.ndarray) -> float:return float(np.vdot(x,x).real)


def step(h, state: np.ndarray, dt: float, dimension: int) -> tuple:
    """Restarted physical Lanczos; full double reorthogonalization, no H repair."""
    n=len(state);beta0=np.linalg.norm(state);q=np.empty((dimension,n),complex);alpha=[];beta=[];last=0.0
    q[0]=state/beta0
    for j in range(dimension):
        z=h@q[j];a=np.vdot(q[j],z);alpha.append(a.real)
        for repeat in range(2):z-=q[:j+1].T@(q[:j+1].conj()@z)
        last=float(np.linalg.norm(z))
        if j+1==dimension or last<2e-13:break
        beta.append(last);q[j+1]=z/last
    m=len(alpha);small=np.diag(alpha).astype(complex)
    if m>1:small+=np.diag(beta[:m-1],1)+np.diag(beta[:m-1],-1)
    e,u=eigh(small);coeff=(u@((u[0].conj())*np.exp(-1j*dt*e)))*beta0
    result=coeff@q[:m]
    # Residual of the actual time-dependent projected exponential at endpoint.
    defect=last*abs(coeff[-1]);del q
    return result,dict(dimension=m,endpoint_Schrodinger_residual=float(defect),norm_drift=norm2(result)-norm2(state),small_min=float(e[0]),small_max=float(e[-1]))


def physical_receivers(geo: dict, sec: dict, c: dict) -> None:
    u=geo['orbitals'];density=[];current=[];nc=180
    for z in range(5):
        n=np.zeros((nc,nc),complex);a=np.zeros_like(n)
        for site,x in enumerate(geo['cells']):
            if x[2]==z:n[4*site:4*site+4,4*site:4*site+4]=np.eye(4)
        for site,target,axis,level,block in geo['links']:
            if axis==2 and geo['cells'][site,2]==z:a[4*site:4*site+4,4*target:4*target+4]+=block
        density.append(project(sec,u.conj().T@n@u));current.append(project(sec,u.conj().T@(1j*(a-a.conj().T))@u))
    spin4=-.5j*geo['cliff'][1]@geo['cliff'][2]
    sec['density']=density;sec['current_z']=current;sec['spin']=project(sec,u.conj().T@np.kron(np.eye(45),spin4)@u)
    sec['pair_count']=np.array([sum(bool(mask&(1<<j)) for j in range(90,180)) for mask in sec['masks']])
    sec['translation']=translation(sec,geo)


def field_measure(f: dict,state: np.ndarray) -> dict:
    means_q=[float(np.vdot(state,apply_field(state,q)).real) for q in f['q']]
    means_p=[float(np.vdot(state,apply_field(state,p)).real) for p in f['p']]
    profiles=[];energy=[];energy_retained=[];coord=[]
    for z in range(5):
        prof=np.zeros((2,8))
        for band,pol in product(range(2),repeat=2):
            q=2*np.pi*(band+1)/5;prof[pol,4*band+2*pol:4*band+2*pol+2]=[np.cos(q*z),np.sin(q*z)]
        profiles.append((prof@np.array(means_p)/f['inertia']).tolist());coord.append((prof@np.array(means_q)).tolist())
        full=0.;ret=0.
        for p in prof:
            op=np.einsum('a,aij->ij',p,f['prect'])/f['inertia'];small=np.einsum('a,aij->ij',p,f['p'])/f['inertia']
            full+=float(np.vdot(state,apply_field(state,op.conj().T@op)).real)/2
            ret+=float(np.vdot(state,apply_field(state,small@small)).real)/2
        energy.append(full);energy_retained.append(ret)
    x=state.reshape(-1,5,45);diagonal=np.sum(abs(x)**2,axis=(0,1))
    cut=[float(sum(diagonal[j] for j,s in enumerate(f['states']) if sum(s)==n)) for n in range(3)]
    qloss=[]
    for q,qr in zip(f['q'],f['qrect']):qloss.append(float(np.vdot(state,apply_field(state,qr.conj().T@qr-q@q)).real))
    return dict(Q=means_q,P=means_p,electric_profile=profiles,coordinate_profile=coord,electric_energy_density_before_cut=energy,
        electric_energy_density_square_of_retained=energy_retained,electric_energy_integral=float(9*sum(energy)),
        electric_kinetic=float(np.vdot(state,apply_field(state,f['kinetic'])).real),shell_basis_weights=cut,
        coordinate_action_omission=qloss,polarization_angular=float(np.vdot(state,apply_field(state,f['pol'])).real))


def observe(geo: dict,f: dict,sec: dict,h,state: np.ndarray,vacdensity: np.ndarray,c: dict) -> dict:
    density=np.array([expectation(state,o) for o in sec['density']]);flux=np.array([expectation(state,o) for o in sec['current_z']]);hs=h@state
    derivative=[]
    for op in sec['density']:
        x=state.reshape(-1,225);acted=(op@x).ravel();derivative.append(float(-2*np.vdot(hs,acted).imag))
    continuity=np.array(derivative)-(np.roll(flux,1)-flux)
    currentQ=[float(np.vdot(state,-apply_terms(sec,d,state)).real) for d in f['derivatives']]
    md=momentum(sec,f,state);x=state.reshape(-1,225);population=np.sum(abs(x)**2,axis=1)
    return dict(norm2=norm2(state),energy=float(np.vdot(state,hs).real),number_density_z=density.tolist(),
        charge_density_z=(c['charge']*density).tolist(),random_fermion_z_PVM=(density/(90 if sec['name']=='neutral' else 91)).tolist(),
        signed_excess_number_z=(density-vacdensity).tolist(),number_flux_z=flux.tolist(),charge_flux_z=(c['charge']*flux).tolist(),
        density_time_derivative=derivative,physical_continuity_defect=continuity.tolist(),physical_continuity_L2=float(np.linalg.norm(continuity)),
        transverse_profile_current=currentQ,whole_sea_canonical_Sz=expectation(state,sec['spin']),
        actual_positive_reference_band_occupancy=float(sec['pair_count']@population),momentum=md,field=field_measure(f,state))


def symmetry_selection(geo: dict,neutral: dict) -> dict:
    # True C4 action in the full 180-mode one-body space; no polar completion.
    spin4=-.5j*geo['cliff'][1]@geo['cliff'][2];r4=expm(-.5j*np.pi*spin4);rotation=np.zeros((180,180),complex)
    lookup={tuple(x):i for i,x in enumerate(geo['cells'])}
    for site,(x,y,z) in enumerate(geo['cells']):
        target=lookup[(-int(y)%3,int(x),int(z))];rotation[4*target:4*target+4,4*site:4*site+4]=r4
    r=geo['orbitals'].conj().T@rotation@geo['orbitals'];pairs=[]
    for m in neutral['masks']:
        pp=[i for i in range(90,180) if m&(1<<i)];hh=[j for j in range(90) if not m&(1<<j)]
        if len(pp)==2:pairs.append((pp,hh))
    weights=[]
    for pp,hh in pairs:
        weights.append(float(sum(abs(np.linalg.det(r[np.ix_(pt,pp)])*np.linalg.det(r[np.ix_(ht,hh)]).conjugate())**2 for pt,ht in pairs)))
    return dict(two_pair_count=len(pairs),C4_two_pair_retained_weights=weights,C4_two_pair_min_retained=min(weights),
        rotation_positive_negative_mix=float(np.linalg.norm(r[:90,90:])),
        translation='Every physical determinant is an exact C5 translation eigenvector, so the retained family is translation complete.',
        charge_conjugation='Selected two-pair family is NOT completed under an independently declared charge-conjugation operator. N89 sources are actual CAR removal, not a claimed C-paired spectrum.',
        consequence='C4 and C nonclosure preclude interpreting polarization/spin asymmetry as symmetry-protected electron spin response. Full-sea canonical spin remains a literal observable.')


def main() -> None:
    started=time.time();inputpath=Path(sys.argv[1]);prefix=Path(sys.argv[2]);c=json.loads(inputpath.read_text());files=[]
    def path(suffix: str) -> Path:return Path(str(prefix)+suffix)
    def snapshot(suffix: str,**arrays) -> None:
        p=path(suffix);np.savez_compressed(p,**arrays);files.append(p.name)
    geo=geometry(c);f=field(c,geo);neutral=tangent(geo);indices=np.flatnonzero(np.all(geo['blocks']==c['transverse_source_momentum'],axis=1))
    positive,sp=source_sector(neutral,indices,True);negative,sm=source_sector(neutral,indices,False)
    sectors={'neutral':neutral,'positive':positive,'negative':negative};structure={}
    # Exact physical mask/source checkpoint BEFORE eigensolver or trajectories.
    for name,sec in sectors.items():
        sec['name']=name;savejson(path('_'+name+'_masks.json'),dict(masks=[str(m) for m in sec['masks']],N={'neutral':90,'positive':91,'negative':89}[name]));files.append(path('_'+name+'_masks.json').name)
    savejson(path('_family.json'),dict(selection=neutral['selection'],symmetry=symmetry_selection(geo,neutral),input_sha256=digest(inputpath)));files.append(path('_family.json').name)
    snapshot('_primitive.npz',orbitals=geo['orbitals'],onebody_components=geo['transformed'],momenta=geo['momenta'],reference_energies=geo['energies'],
        field_terms=f['terms'],field_off=f['off'],field_boson=f['boson'],field_q=f['q'],field_p=f['p'],field_qrect=f['qrect'],field_prect=f['prect'])
    for name,sec in sectors.items():
        sec['matrices']=[]
        for k,h in enumerate(geo['transformed']):
            op=project(sec,h);sec['matrices'].append(op);p=path(f'_{name}_Hcomponent{k}.npz');sparse.save_npz(p,op);files.append(p.name)
        structure[name]=dict(matter_dimension=len(sec['masks']),joint_dimension=len(sec['masks'])*225,component_skew_max=max(float(sparse.linalg.norm(m-m.conj().T)) for m in sec['matrices']))
        print('Physical sector complete',name,structure[name],flush=True)
    for name,ops in [('add',sp),('remove',sm)]:
        for j,op in enumerate(ops):p=path(f'_{name}_source{j}.npz');sparse.save_npz(p,op);files.append(p.name)
    # Negative sector is physically constructed and snapshotted, not silently replaced by empty holes.
    del negative['matrices'];gc.collect()
    for sec in [neutral,positive]:physical_receivers(geo,sec,c)
    hn=install(neutral,f,geo['offset']);n=len(neutral['masks'])*225
    v0=np.zeros(n,complex);v0[neutral['masks'].index(geo['sea'])*225:][:225]=np.tile(eigh(f['kinetic'])[1][:,0],5)/np.sqrt(5)
    e,v=eigsh(hn,k=1,which='SA',v0=v0,tol=c['eigensolver_tolerance'],maxiter=4000);omega=v[:,0];e0=float(e[0]);res=float(np.linalg.norm(hn@omega-e0*omega));del v
    vacdensity=np.array([expectation(omega,o) for o in neutral['density']]);a=omega.reshape(-1,225)
    # Canonical full physical source, all four spinor components and all five kz.
    spinor=np.array(c['canonical_spinor_real'])+1j*np.array(c['canonical_spinor_imag']);spinor/=np.linalg.norm(spinor)
    physical=np.zeros((45,4),complex)
    for site,(x,y,z) in enumerate(geo['cells']):
        dz=(z-c['source_center_z']+2.5)%5-2.5
        physical[site]=spinor*np.exp(-dz*dz/(2*c['source_width']**2)+1j*c['source_kz']*z+2j*np.pi*x/3)
    physical=physical.ravel();physical/=np.linalg.norm(physical);coef=geo['orbitals'][:,indices].conj().T@physical
    create=sum((z*op for z,op in zip(coef,sp)),start=sparse.csr_matrix(sp[0].shape));remove=sum((z.conjugate()*op for z,op in zip(coef,sm)),start=sparse.csr_matrix(sm[0].shape))
    rawplus=(create@a).ravel();rawminus=(remove@a).ravel();pn=norm2(rawplus);mn=norm2(rawminus);charge=rawplus/np.sqrt(pn)
    rawadd=np.array([norm2((op@a).ravel()) for op in sp]);rawremove=np.array([norm2((op@a).ravel()) for op in sm]);car_error=float(sparse.linalg.norm(sum((op.conj().T@op for op in sp),start=sparse.csr_matrix((len(neutral['masks']),)*2))+sum((op.conj().T@op for op in sm),start=sparse.csr_matrix((len(neutral['masks']),)*2))-20*sparse.eye(len(neutral['masks']))))
    # A bandwidth-containing field-ONLY Weyl unitary. Amplitudes do not borrow a dispersion.
    qdrive=np.zeros(8);pdrive=np.zeros(8)
    for band in range(2):
        phase=2*np.pi*(band+1)*c['field_center_z']/5
        qdrive[4*band:4*band+2]=c['field_Q_amplitude']*np.array([np.cos(phase),np.sin(phase)])/np.sqrt(2)
        pdrive[4*band:4*band+2]=c['field_P_amplitude']*np.array([-np.sin(phase),np.cos(phase)])/np.sqrt(2)
        qdrive[4*band+2:4*band+4]=.5*c['field_Q_amplitude']*np.array([-np.sin(phase),np.cos(phase)])/np.sqrt(2)
        pdrive[4*band+2:4*band+4]=.5*c['field_P_amplitude']*np.array([np.cos(phase),np.sin(phase)])/np.sqrt(2)
    generator=np.einsum('a,aij->ij',pdrive,f['q'])-np.einsum('a,aij->ij',qdrive,f['p']);drive=expm(1j*generator)
    pulse=apply_field(charge,drive);neutralpulse=apply_field(omega,drive)
    snapshot('_initial_states.npz',neutral=omega,charged=charge,charged_field=pulse,neutral_field=neutralpulse,raw_add_packet=rawplus,raw_remove_packet=rawminus,
        source_coefficients=coef,source_physical=physical,field_drive=drive,field_Qdrive=qdrive,field_Pdrive=pdrive,raw_add_weights=rawadd,raw_remove_weights=rawremove)
    rho=a@a.conj().T;remove_gram=np.empty((20,20),complex)
    for i,j in product(range(20),repeat=2):
        op=(sm[i].conj().T@sm[j]).tocoo();remove_gram[i,j]=np.sum(op.data*rho[op.col,op.row])
    add_gram=np.eye(20)-remove_gram.T
    canonical=np.zeros((180,20),complex)
    for kz,s in product(range(5),range(4)):
        wave=np.zeros((45,4),complex)
        for site,(x,y,z) in enumerate(geo['cells']):wave[site,s]=np.exp(2j*np.pi*(x/3+kz*z/5))/np.sqrt(45)
        canonical[:,4*kz+s]=wave.ravel()
    canonical_coordinates=geo['orbitals'][:,indices].conj().T@canonical
    canonical_add=canonical_coordinates.conj().T@add_gram@canonical_coordinates
    canonical_remove=canonical_coordinates.T@remove_gram@canonical_coordinates.conj()
    snapshot('_source_grams.npz',orbital_add=add_gram,orbital_remove=remove_gram,canonical_add=canonical_add,canonical_remove=canonical_remove,canonical_coordinates=canonical_coordinates)
    initial=dict(neutral_energy=e0,neutral_residual=res,source_packet_raw_add=pn,source_packet_raw_remove=mn,source_sum=pn+mn,
        raw_orbital_add=rawadd.tolist(),raw_orbital_remove=rawremove.tolist(),orbital_CAR_sum_operator_defect=car_error,
        field_unitarity_defect=float(np.linalg.norm(drive.conj().T@drive-np.eye(45))),
        source_profile_probability=np.sum(abs(physical.reshape(3,3,5,4))**2,axis=(0,1,3)).tolist(),
        t0_matter_density_change=[expectation(pulse,o)-expectation(charge,o) for o in positive['density']],
        raw_canonical_kz_spinor_add=np.diag(canonical_add).real.reshape(5,4).tolist(),raw_canonical_kz_spinor_remove=np.diag(canonical_remove).real.reshape(5,4).tolist(),
        scope='Canonical source preparation is normalized ONCE after storing raw CAR weights; no virtual/source/string weight normalization.')
    savejson(path('_initial.json'),initial);files.append(path('_initial.json').name)
    print('Exact source/reference snapshot complete',initial,flush=True)
    # Actual newly coupled coordinate-source energy scales: spectral moments use full owning H.
    scales=[]
    for axis,q in enumerate(f['q']):
        mean=np.vdot(omega,apply_field(omega,q));source=apply_field(omega,q)-mean*omega;weight=norm2(source);energy_source=hn@source-e0*source
        scales.append(dict(axis=axis,band=axis//4+1,raw_retained_weight=weight,spectral_centroid=float(np.vdot(source,energy_source).real/weight),
            spectral_second_moment=float(norm2(energy_source)/weight),spectral_width=float(np.sqrt(max(0,norm2(energy_source)/weight-(np.vdot(source,energy_source).real/weight)**2)))))
    savejson(path('_scales.json'),dict(kind='NEW_COUPLED_FULL_H_COORDINATE_SOURCE_SPECTRAL_MOMENTS_NOT_FREE_PHOTON_FREQUENCIES',scales=scales));files.append(path('_scales.json').name)
    times=np.linspace(0,c['duration'],c['time_points']);arms={};diagnostics={};statecaps=[]
    for label,sec,start,off in [('charged_no_incoming',positive,charge,False),('charged_incoming',positive,pulse,False),
                               ('charged_same_state_EMoff_no_incoming',positive,charge,True),('charged_same_state_EMoff_incoming',positive,pulse,True),
                               ('neutral_incoming',neutral,neutralpulse,False)]:
        h=install(sec,f,geo['offset'],off);state=start.copy();records=[];steps=[];stored={}
        for j,t in enumerate(times):
            if j:state,d=step(h,state,float(times[j]-times[j-1]),c['krylov_dimension']);steps.append(d)
            record=observe(geo,f,sec,h,state,vacdensity,c);records.append(record)
            # Full exact output states at three declared anchors; physical observations at every point.
            if j in [0,len(times)//2,len(times)-1]:
                p=path(f'_{label}_state{j}.npz');np.savez_compressed(p,state=state,time=t);files.append(p.name);statecaps.append(p.name)
                if sec['name']=='positive':
                    x=state.reshape(-1,225);pop=np.sum(abs(x)**2,axis=1);weights=[]
                    for jj,op in zip(indices,sp):
                        full=float(sum(pop[r] for r,m in enumerate(sec['masks']) if m&(1<<int(jj))));retained=norm2((op.conj().T@x).ravel())
                        weights.append(dict(orbital=int(jj),full_removal_norm2=full,retained_neutral_return_norm2=retained,omitted_difference=full-retained))
                    record['raw_reverse_source_weights']=weights
            savejson(path('_checkpoint.json'),dict(completed_arms=arms,current_arm=label,times=times[:j+1].tolist(),current_records=records,step_diagnostics=steps,initial=initial));
            print('Natural physical trajectory',label,'time',float(t),'continuity',record['physical_continuity_L2'],flush=True)
        arms[label]=records;diagnostics[label]=steps;del state,h;gc.collect()
    vacfield=field_measure(f,omega);comparisons={}
    for rootname in ['charged','charged_same_state_EMoff']:
        a0=arms[rootname+'_no_incoming'];a1=arms[rootname+'_incoming'];comparisons[rootname]=dict(
            density_L2=[float(np.linalg.norm(np.array(p['number_density_z'])-b['number_density_z'])) for p,b in zip(a1,a0)],
            current_L2=[float(np.linalg.norm(np.array(p['number_flux_z'])-b['number_flux_z'])) for p,b in zip(a1,a0)],
            whole_sea_Sz_difference=[p['whole_sea_canonical_Sz']-b['whole_sea_canonical_Sz'] for p,b in zip(a1,a0)],
            matter_momentum_L2=[float(np.linalg.norm(np.array(p['momentum']['matter'])-b['momentum']['matter'])) for p,b in zip(a1,a0)])
    result=dict(kind='COMPLETED_NEW_SAME_PRIMITIVE_H_180CAR_90SEA_TWO_BAND_FIELD_CHARGED_PACKET_EXPERIMENT',input=c,structure=structure,initial=initial,
        coupled_scales=scales,times=times.tolist(),vacuum_density_z=vacdensity.tolist(),vacuum_field=vacfield,arms=arms,comparisons=comparisons,
        numerical_steps=diagnostics,exact_state_checkpoints=statecaps,artifact_files=files,
        limitations=['Selected nonlinear two-pair CI is not old nonlinear Gaussian TT span and is not all-Fock closure.',
            'C4/C nonclosure of selected higher-pair family is measured/disclosed; no symmetry-protected spin/helicity/asymptotic photon interpretation.',
            'Source localized only in z; xy is declared exact transverse plane-wave fiber. Signed vacuum-subtracted charge is not a position PVM.',
            'Random fermion n_z/N PVM samples whole filled sea, not isolated added charge. Canonical whole-sea spin is not electron spinflip.',
            'Shell2 field projection makes finite field EOM and Peierls/CCR defects; actual physical continuity/current artifacts are retained.',
            'No supplied Maxwell magnetic potential, no old poles, no Pcm=J, no infinite-time/bulk/Lorentz/Nature claim.'],
        elapsed_seconds=time.time()-started,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    savejson(path('_result.json'),result);print(json.dumps(dict(result=path('_result.json').name,elapsed_seconds=result['elapsed_seconds'],peak_resident_KiB=result['peak_resident_KiB'],comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
