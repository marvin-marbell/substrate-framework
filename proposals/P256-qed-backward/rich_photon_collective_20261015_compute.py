"""NEW coupled larger-ring quantum-sea modes and total-energy packet consumer."""
from itertools import product
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import eigsh, expm_multiply
from rich_photon_collective_20261015_model import geometry, family, onebody, field, install, field_action, lanczos
from rich_mobile_packet_20261015_model import shell_states

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'

def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def spectral(operator,source,energy,steps):
    norm=float(np.vdot(source,source).real);record=lanczos(operator,source,steps);ee=record['energies']-energy;ww=record['weights'];tri=np.diag(record['alpha'])+np.diag(record['beta'],1)+np.diag(record['beta'],-1);_,rot=eigh(tri)
    residual=record['last_beta']*abs(rot[-1]);groups=[]
    for k in np.argsort(ee):
        if ww[k]<1e-13:continue
        if groups and abs(ee[k]-groups[-1]['energy'])<1e-7:
            groups[-1]['weight']+=float(ww[k]);groups[-1]['residual']=max(groups[-1]['residual'],float(residual[k]));groups[-1]['quadrature_multiplicity']+=1
        else:groups.append(dict(energy=float(ee[k]),weight=float(ww[k]),residual=float(residual[k]),quadrature_multiplicity=1))
    centroid=float(np.dot(ee,ww)/norm);width=float(np.sqrt(max(0,np.dot((ee-centroid)**2,ww)/norm)));largest=max(groups,key=lambda x:x['weight'])
    grid=np.linspace(0,2.5,251);eta=.025;response=np.array([sum(w*(1/(z+1j*eta-e)-1/(z+1j*eta+e)) for e,w in zip(ee,ww)) for z in grid])
    return dict(raw_norm=norm,weight_sum=float(ww.sum()),centroid=centroid,width=width,largest_group=largest,largest_fraction=largest['weight']/norm,groups=groups,krylov_dimension=record['dimension'],alpha=record['alpha'],beta=record['beta'],frequency_grid=grid.tolist(),retarded_real=response.real.tolist(),retarded_imag=response.imag.tolist(),eta=eta,meaning='Full-owning-H source Lanczos measure; every quadrature group and physical Ritz residual retained. Unconverged groups are NOT exact poles. Retarded expression retains positive and negative frequency signs; no lowwindow replacement.')

def run():
    mode=sys.argv[1];stem=PREFIX+mode;started=time.monotonic();c=json.loads((ROOT/(stem+'_physics.json')).read_text());geo=geometry(c);masks=family(geo);n=len(masks);f=field(c,geo);nf=len(f['states']);nb=5*nf
    dump(ROOT/(stem+'_checkpoint.json'),dict(phase='geometry_and_exact_family',matter=n,boson=nb,joint=n*nb,elapsed=time.monotonic()-started))
    dump(ROOT/(stem+'_masks.json'),[str(m) for m in masks]);matrices=[]
    for k,h in enumerate(geo['transformed']):
        m=onebody(masks,h);matrices.append(m);sparse.save_npz(ROOT/(stem+f'_Hcomponent{k}.npz'),m)
    np.savez_compressed(ROOT/(stem+'_primitive.npz'),orbitals=geo['orbitals'],components=geo['components'],transformed=geo['transformed'],profiles=geo['profiles'],momenta=geo['momenta'],energies=geo['energies'],q=f['q'],p=f['p'],p2=f['p2'],phi=f['phi'],phi_h=f['phi_h'],kinetic=f['kinetic'],terms=np.array(f['terms']),boson=f['boson'],offset=geo['offset'],states=np.array(f['states']))
    hamiltonian=install(matrices,f['terms'],f['boson'],geo['offset']);off=install(matrices,f['off'],f['boson'],geo['offset'])
    ephi,uphi=eigh(f['phi_h']);eelectric,uelectric=eigh(f['kinetic']);initial=np.zeros((n,5,nf),complex);initial[masks.index(geo['sea'])]=np.outer(uphi[:,0],uelectric[:,0]);initial=initial.ravel()
    energy,vectors=eigsh(hamiltonian,k=1,which='SA',v0=initial,tol=c['eigensolver_tolerance'],ncv=12,maxiter=4000);ground=vectors[:,0];e0=float(energy[0]);residual=float(np.linalg.norm(hamiltonian@ground-e0*ground));del vectors
    np.savez_compressed(ROOT/(stem+'_ground.npz'),state=ground,energy=e0)
    dump(ROOT/(stem+'_checkpoint.json'),dict(phase='own_coupled_ground',energy=e0,residual=residual,joint=n*nb,elapsed=time.monotonic()-started))
    # Literal local terms: onsite split by layer, transverse bonds at their layer;
    # z bonds half assigned to each endpoint. Uniform Phi energy is shared 1/5.
    local_onsite=[];local_mass=[];local_z=[];density=[];current=[];u=geo['orbitals'];nc=geo['nc'];beta=geo['cliff'][0]
    for z in range(5):
        onsite=np.zeros((nc,nc),complex);mass=np.zeros_like(onsite);zhop=np.zeros_like(onsite);den=np.zeros_like(onsite);jz=np.zeros_like(onsite)
        for site,coord in enumerate(geo['cells']):
            if coord[2]==z:
                sl=slice(4*site,4*site+4);mass[sl,sl]=beta;onsite[sl,sl]=3*c['wilson']*beta;den[sl,sl]=np.eye(4)
        for site,end,axis,level,hop in geo['links']:
            if axis!=2:continue
            a=int(geo['cells'][site,2]);b=int(geo['cells'][end,2]);sl=slice(4*site,4*site+4);tl=slice(4*end,4*end+4)
            weight=.5*(int(a==z)+int(b==z));zhop[sl,tl]+=weight*hop;zhop[tl,sl]+=weight*hop.conj().T
            if a==z:jz[sl,tl]+=1j*hop;jz[tl,sl]+=-1j*hop.conj().T
        for collection,op in ((local_onsite,onsite),(local_mass,mass),(local_z,zhop),(density,den),(current,jz)):
            transformed=u.conj().T@op@u;transformed[abs(transformed)<2e-12]=0;collection.append(onebody(masks,transformed))
    # Before-cut P_i P_j via exact shellcut+1 rectangular action.
    large=shell_states(c['shell_cutoff']+1);ll={s:i for i,s in enumerate(large)};prect=[]
    for axis in range(8):
        pr=np.zeros((len(large),nf),complex)
        for j,s in enumerate(f['states']):
            t=list(s);t[axis]+=1;pr[ll[tuple(t)],j]=1j*np.sqrt(t[axis]/2)/c['shell_width']
            if s[axis]:t=list(s);t[axis]-=1;pr[ll[tuple(t)],j]=-1j*np.sqrt(s[axis]/2)/c['shell_width']
        prect.append(pr)
    profiles=np.zeros((2,5,8));electric_energy=[]
    for z in range(5):
        for band,pol in product(range(2),repeat=2):profiles[pol,z,4*band+2*pol:4*band+2*pol+2]=[np.cos(2*np.pi*(band+1)*z/5),np.sin(2*np.pi*(band+1)*z/5)]
        en=np.zeros((nf,nf),complex)
        for pol in range(2):
            rect=sum(profiles[pol,z,a]*prect[a] for a in range(8))/f['inertia'];en+=geo['volume']/5*(rect.conj().T@rect)/2
        electric_energy.append(en)
    energy_identity=float(np.linalg.norm(sum(electric_energy)-f['kinetic']))
    shell=np.array([sum(s) for s in f['states']]);positive=np.array([sum(bool(m&(1<<i)) for i in range(geo['sea_count'],nc)) for m in masks]);matter_momentum=np.array([sum(int(geo['momenta'][i,2]) for i in range(nc) if m&(1<<i))%5 for m in masks])
    def boson_apply(state,op):return (op@state.reshape(n,nb).T).T.ravel()
    def mean_matter(state,op):
        x=state.reshape(n,nb);return float(np.vdot(x,op@x).real)
    def mixed(state,op,term):
        x=state.reshape(n,nb);return float(np.vdot(x,(term@(op@x).T).T).real)
    def transverse_current(state,axis):
        x=state.reshape(n,nb);out=np.zeros_like(x)
        for k in range(2,len(matrices),2):
            coefficient=geo['profiles'][k,axis]
            if coefficient:out+=coefficient*((sparse.csr_matrix(f['terms'][k+1])@(matrices[k]@x).T).T-(sparse.csr_matrix(f['terms'][k])@(matrices[k+1]@x).T).T)
        return out.ravel()
    def observe(state,timepoint,law):
        x=state.reshape(n,5,nf);prob=abs(x)**2;matterphi=prob.sum(axis=2);terms=[];scalar=float(np.vdot(x,np.einsum('ab,mbf->maf',f['phi_h'],x,optimize=True)).real);electric=float(np.vdot(x,np.einsum('ab,mpb->mpa',f['kinetic'],x,optimize=True)).real)
        # Use direct action, avoiding a density-matrix transpose convention ambiguity.
        meansq=np.array([np.vdot(state,field_action(state,q,nf)).real for q in f['q']]);meansp=np.array([np.vdot(state,field_action(state,p,nf)).real for p in f['p']])
        for z in range(5):
            onsite=mean_matter(state,local_onsite[z]);phi_coupling=mixed(state,local_mass[z],sparse.csr_matrix(f['terms'][1]));zlinks=mean_matter(state,local_z[z]);transverse=0.
            for axis in range(2):
                k=2+2*(5*axis+z);transverse+=mixed(state,matrices[k],sparse.csr_matrix(f['off' if law=='off' else 'terms'][k]));transverse+=mixed(state,matrices[k+1],sparse.csr_matrix(f['off' if law=='off' else 'terms'][k+1]))
            ez=float(np.vdot(state,field_action(state,electric_energy[z],nf)).real);terms.append(dict(onsite=onsite,Phi_matter=phi_coupling,z_bond=zlinks,transverse_Peierls=transverse,global_Phi=scalar/5,electric=ez,reference=-geo['offset']/5,total=onsite+phi_coupling+zlinks+transverse+scalar/5+ez-geo['offset']/5))
        actual=float(np.vdot(state,(off if law=='off' else hamiltonian)@state).real);joint=np.zeros((5,5));amplitudes=(x.reshape(-1,nf)@f['translation_vectors'].conj()).reshape(x.shape);weights=(abs(amplitudes)**2).sum(axis=1)
        for m,k in product(range(5),repeat=2):joint[m,k]=weights[matter_momentum==m][:,f['translation_values']==k].sum()
        return dict(time=timepoint,norm=float(np.vdot(state,state).real),energy=actual,profile=terms,profile_sum_defect=sum(t['total'] for t in terms)-actual,density=[mean_matter(state,d) for d in density],z_number_current=[mean_matter(state,j) for j in current],transverse_current=[float(np.vdot(state,transverse_current(state,a)).real) for a in range(8)] if law=='coupled' else [0.]*8,Q=meansq.tolist(),P=meansp.tolist(),meanA=(profiles@meansq).tolist(),meanE=(profiles@(meansp/f['inertia'])).tolist(),electric=electric,scalar=scalar,Phi_distribution=matterphi.sum(axis=0).tolist(),Phi_mean=float(matterphi.sum(axis=0)@f['phi']),positive_reference_occupation=float(prob.sum(axis=(1,2))@positive),shell_distribution=[float(prob[:,:,shell==s].sum()) for s in range(c['shell_cutoff']+1)],joint_matter_field_C5=joint.tolist(),total_C5=[float(sum(joint[m,k] for m,k in product(range(5),repeat=2) if (m+k)%5==a)) for a in range(5)])
    vacuum=observe(ground,0.,'coupled');dump(ROOT/(stem+'_ground_observed.json'),vacuum)
    # Complete canonical source Gram in one transverse fibre, raw addition/removal.
    ids=np.flatnonzero(np.all(geo['blocks']==(1,0),axis=1));rho=np.empty((20,20),complex);groundx=ground.reshape(n,nb);lookup={mask:k for k,mask in enumerate(masks)}
    for a,i0 in enumerate(ids):
        i=int(i0)
        for b,j0 in enumerate(ids):
            j=int(j0);rows=[];cols=[];signs=[]
            for column,mask in enumerate(masks):
                if not mask&(1<<j):continue
                first=mask^(1<<j)
                if first&(1<<i):continue
                destination=lookup.get(first|(1<<i))
                if destination is not None:
                    rows.append(destination);cols.append(column);signs.append((-1)**(((mask&((1<<j)-1)).bit_count()+(first&((1<<i)-1)).bit_count())%2))
            rho[a,b]=np.einsum('ij,ij,i->',groundx[rows].conj(),groundx[cols],np.array(signs),optimize=True) if rows else 0.
    canonical=geo['orbitals'][:,ids]
    np.savez_compressed(ROOT/(stem+'_source_grams.npz'),orbital_ids=ids,removal=rho,addition=np.eye(20)-rho.T,canonical_orbitals=canonical,CAR_sum=np.eye(20),all_virtual_pair_count=(c['transverse']**2)*100)
    # Own spectra of every real coordinate and literal transverse current.
    spectra=[]
    for axis in range(8):
        qsource=field_action(ground,f['q'][axis],nf);qsource-=np.vdot(ground,qsource)*ground;j_source=transverse_current(ground,axis);j_source-=np.vdot(ground,j_source)*ground
        qrecord=spectral(hamiltonian,qsource,e0,c['spectral_steps']);jrecord=spectral(hamiltonian,j_source,e0,c['spectral_steps']);contact=0.
        for k in range(2,len(matrices),2):
            coefficient=geo['profiles'][k,axis]
            contact-=coefficient**2*(mixed(ground,matrices[k],sparse.csr_matrix(f['terms'][k]))+mixed(ground,matrices[k+1],sparse.csr_matrix(f['terms'][k+1])))
        spectra.append(dict(axis=axis,band=axis//4+1,polarization='x' if axis%4<2 else 'y',quadrature='cos' if axis%2==0 else 'sin',coordinate=qrecord,literal_current=jrecord,contact=contact))
        dump(ROOT/(stem+'_spectra.json'),dict(spectra=spectra,energy=e0,scope='Every completed axis uses its own full retained H, no old poles. Real cos/sin coordinates span both q signs; no centroid is promoted to a pole.'))
    # Unconditioned field-only unitary: both q magnitudes and both polarizations.
    center=2.;qdrive=np.zeros(8);pdrive=np.zeros(8)
    for band,pol in product(range(2),repeat=2):
        phase=2*np.pi*(band+1)*center/5;amplitude=c['pulse_amplitude']/(band+1)*(1. if pol==0 else .5);qdrive[4*band+2*pol:4*band+2*pol+2]=amplitude*np.array([np.cos(phase),np.sin(phase)]);pdrive[4*band+2*pol:4*band+2*pol+2]=c['pulse_momentum']/(band+1)*(1. if pol==0 else .5)*np.array([-np.sin(phase),np.cos(phase)])
    drive=expm(1j*sum(pdrive[a]*f['q'][a]-qdrive[a]*f['p'][a] for a in range(8)));pulse=field_action(ground,drive,nf);unitarity=float(np.linalg.norm(drive.conj().T@drive-np.eye(nf)));np.savez_compressed(ROOT/(stem+'_preparation.npz'),ground=ground,pulse=pulse,drive=drive,qdrive=qdrive,pdrive=pdrive)
    matterphi=sparse.kron(matrices[0],sparse.eye(5))+sparse.kron(matrices[1],sparse.diags(c['coupling']*f['phi']))
    for k in range(2,len(matrices),2):matterphi+=sparse.kron(matrices[k],sparse.eye(5))
    matterphi+=sparse.kron(sparse.eye(n),sparse.csr_matrix(f['phi_h']))-geo['offset']*sparse.eye(n*5);trace=float(matterphi.diagonal().sum().real);dt=c['dt'];fieldstep=expm(-1j*dt*f['kinetic'])
    def offstep(state):
        x=state.reshape(n*5,nf);out=expm_multiply(-1j*dt*matterphi,x,traceA=-1j*dt*trace);return (fieldstep@out.T).T.ravel()
    arms=dict(coupled_pulse=pulse.copy(),same_state_off_pulse=pulse.copy(),same_state_off_baseline=ground.copy());records={key:[] for key in arms};steps={key:[] for key in arms};times=np.arange(c['time_points'])*dt
    for index,t in enumerate(times):
        for arm,state in arms.items():
            records[arm].append(observe(state,float(t),'coupled' if arm=='coupled_pulse' else 'off'))
            if index in (0,len(times)//2,len(times)-1):np.savez_compressed(ROOT/(stem+f'_{arm}_state{index}.npz'),state=state,time=t)
        dump(ROOT/(stem+'_checkpoint.json'),dict(phase='actual_packet',index=index,time=float(t),ground=vacuum,records=records,lanczos_steps=steps,elapsed=time.monotonic()-started))
        if index<len(times)-1:
            arms['coupled_pulse'],receipt=lanczos(hamiltonian,arms['coupled_pulse'],c['trajectory_krylov'],dt);steps['coupled_pulse'].append(receipt)
            for arm in ('same_state_off_pulse','same_state_off_baseline'):arms[arm]=offstep(arms[arm])
    result=dict(kind='NEW_ACTUAL_LARGER_RING_QUANTUM_SEA_PHI_TWO_BAND_PACKET',physics=c,geometry=dict(volume=geo['volume'],CAR=nc,filled_sea=geo['sea_count'],matter_dimension=n,boson_dimension=nb,joint_dimension=n*nb,all_individual_virtual_pairs=c['transverse']**2*100,all_double_pairs_in_self_C4_fibre=2025),energy=e0,eigensolver_residual=residual,ground=vacuum,spectra=spectra,records=records,lanczos_steps=steps,field_unitarity=unitarity,electric_profile_operator_sum_defect=energy_identity,wall_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits=['ALL individual virtuals in every transverse fibre, but higher multipairs are full only in(0,0). Not unrestricted dynamical Fock closure.','C4-invariant family by full single-pair space plus full double-pair space in self-C4(0,0); C completion not asserted.','Finite field cut and uniform quantumPhi; total-energy local share of Phi is bookkeeping, not local transport.','Every spectral group has Ritz residual; unconverged quadratures not exact poles or asymptotic photons.','Unconditioned two-band field-only unitary, no source filtering/postselection.','C5 recoil is modular lattice momentum, not current or continuum Pcm.'])
    dump(ROOT/(stem+'_data.json'),result);print(json.dumps(dict(mode=mode,energy=e0,residual=residual,joint=n*nb,wall_seconds=result['wall_seconds'],peak_RSS_KiB=result['peak_RSS_KiB'])),flush=True)

if __name__=='__main__':run()
