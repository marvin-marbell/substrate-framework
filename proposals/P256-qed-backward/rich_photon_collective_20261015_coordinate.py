"""Material alternative: diagonal nonlinear Peierls Q, finite-difference electric law.
All8 real coordinates/two bands/two polarizations, same rich quantum sea/Phi.
Finite Q-domain and h are explicit; NOT continuum/canonical or C5-complete proof.
"""
from itertools import product
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import LinearOperator, eigsh, expm_multiply
from rich_photon_collective_20261015_model import geometry, family, onebody, field_action, lanczos

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'

def lattice(radius):
    # Exact sparse L1 lattice, avoiding a 5**8 Cartesian allocation.
    nodes={(0,)*8};front=nodes.copy()
    for _ in range(radius):
        new=set()
        for node in front:
            for axis,sign in product(range(8),(-1,1)):
                changed=list(node);changed[axis]+=sign;new.add(tuple(changed))
        front=new-nodes;nodes|=new
    return sorted(nodes)

def write_json(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def coordinate_field(c,geo):
    nodes=lattice(c['Q_radius']);extended=lattice(c['Q_radius']+1);look={s:i for i,s in enumerate(nodes)};extlook={s:i for i,s in enumerate(extended)};n=len(nodes);step=c['Q_step'];inertia=geo['volume']/2;kinetic=np.zeros((n,n),complex);momenta=[];gradients=[]
    for axis in range(8):
        momentum=np.zeros((n,n),complex);gradient=np.zeros((len(extended),n),complex)
        for j,node in enumerate(nodes):
            kinetic[j,j]+=1/(inertia*step**2);gradient[extlook[node],j]-=1/step;lower=list(node);lower[axis]-=1;gradient[extlook[tuple(lower)],j]+=1/step
            for sign in (-1,1):
                changed=list(node);changed[axis]+=sign;i=look.get(tuple(changed))
                if i is not None:kinetic[i,j]-=1/(2*inertia*step**2);momentum[i,j]=1j*sign/(2*step)
        momenta.append(momentum);gradients.append(-1j*gradient)
    coordinates=np.array(nodes,dtype=float).T*step;phi=c['phi_center']+2*c['phi_width']*(np.arange(5)-2)/5;pk=2*np.pi*np.fft.fftfreq(5,d=2*c['phi_width']/5);ft=np.exp(2j*np.pi*np.outer(np.arange(5),np.arange(5))/5)/np.sqrt(5);pp=ft@np.diag(pk)@ft.conj().T;phi_h=pp@pp/(2*geo['volume']*c['scalar_inertia'])+np.diag(geo['volume']*c['scalar_stiffness']*(phi-c['phi_center'])**2/2)
    factors=[np.ones((5,n)),np.broadcast_to(c['coupling']*phi[:,None],(5,n))]
    for k in range(2,len(geo['components']),2):
        phase=geo['profiles'][k]@coordinates;factors.extend([np.broadcast_to(np.cos(phase),(5,n)),np.broadcast_to(np.sin(phase),(5,n))])
    profiles=np.zeros((2,5,8));energy=[]
    for z in range(5):
        electric=np.zeros((n,n),complex)
        for band,pol in product(range(2),repeat=2):profiles[pol,z,4*band+2*pol:4*band+2*pol+2]=[np.cos(2*np.pi*(band+1)*z/5),np.sin(2*np.pi*(band+1)*z/5)]
        for pol in range(2):
            rect=sum(profiles[pol,z,a]*gradients[a] for a in range(8))/inertia;electric+=(geo['volume']/5)*rect.conj().T@rect/2
        energy.append(electric)
    return dict(nodes=np.array(nodes),coordinates=coordinates,momenta=np.array(momenta),kinetic=kinetic,phi=phi,phi_h=phi_h,factors=np.array(factors),profiles=profiles,electric_profile=energy,energy_identity=float(np.linalg.norm(sum(energy)-kinetic)),inertia=inertia)

def run():
    started=time.monotonic();mode=sys.argv[1];stem=PREFIX+mode;c=json.loads((ROOT/(stem+'_physics.json')).read_text());geo=geometry(c);masks=family(geo);n=len(masks);f=coordinate_field(c,geo);nf=len(f['nodes']);nb=5*nf;matrices=[]
    write_json(ROOT/(stem+'_masks.json'),[str(m) for m in masks])
    for k,primitive in enumerate(geo['transformed']):
        matrix=onebody(masks,primitive);matrices.append(matrix);sparse.save_npz(ROOT/(stem+f'_Hcomponent{k}.npz'),matrix)
    np.savez_compressed(ROOT/(stem+'_primitive.npz'),**{key:geo[key] for key in ('orbitals','components','transformed','profiles','momenta','energies')},**{key:f[key] for key in ('nodes','coordinates','momenta','kinetic','phi','phi_h','factors','profiles') if key not in ('momenta','profiles')},field_momenta=f['momenta'],spatial_profiles=f['profiles'],electric_profile=np.array(f['electric_profile']),offset=geo['offset'])
    kfield=sparse.csr_matrix(f['kinetic']);scalar=sparse.csr_matrix(f['phi_h']);factors=f['factors'].reshape(len(matrices),nb)
    def install(off=False):
        weights=factors.copy()
        if off:
            for k in range(2,len(matrices),2):weights[k]=1.;weights[k+1]=0.
        def action(vector):
            x=np.asarray(vector).reshape(n,5,nf);out=(kfield@x.reshape(n*5,nf).T).T.reshape(x.shape)+np.einsum('ab,mbf->maf',f['phi_h'],x,optimize=True)-geo['offset']*x;flat=x.reshape(n,nb);result=out.reshape(n,nb)
            for matrix,weight in zip(matrices,weights):result+=(matrix@flat)*weight
            return result.ravel()
        return LinearOperator((n*nb,n*nb),matvec=action,rmatvec=action,dtype=complex)
    h=install();hoff=install(True);ephi,uphi=eigh(f['phi_h']);ebare,ubare=eigh(f['kinetic']);initial=np.zeros((n,5,nf),complex);initial[masks.index(geo['sea'])]=np.outer(uphi[:,0],ubare[:,0]);e,v=eigsh(h,k=1,which='SA',v0=initial.ravel(),tol=1e-8,ncv=12,maxiter=4000);ground=v[:,0];e0=float(e[0]);residual=float(np.linalg.norm(h@ground-e0*ground));del v,initial
    np.savez_compressed(ROOT/(stem+'_ground.npz'),state=ground,energy=e0);write_json(ROOT/(stem+'_checkpoint.json'),dict(phase='own_coordinate_vacuum',energy=e0,residual=residual,joint=n*nb,elapsed=time.monotonic()-started))
    # Literal local matter partition as in the oscillator construction.
    local=[];density=[];zcurrent=[];u=geo['orbitals'];nc=geo['nc'];beta=geo['cliff'][0]
    for z in range(5):
        onsite=np.zeros((nc,nc),complex);mass=np.zeros_like(onsite);zhop=np.zeros_like(onsite);den=np.zeros_like(onsite);jz=np.zeros_like(onsite)
        for site,coord in enumerate(geo['cells']):
            if coord[2]==z:
                sl=slice(4*site,4*site+4);onsite[sl,sl]=3*c['wilson']*beta;mass[sl,sl]=beta;den[sl,sl]=np.eye(4)
        for site,end,axis,level,hop in geo['links']:
            if axis!=2:continue
            a=geo['cells'][site,2];b=geo['cells'][end,2];weight=.5*(int(a==z)+int(b==z));sl=slice(4*site,4*site+4);tl=slice(4*end,4*end+4);zhop[sl,tl]+=weight*hop;zhop[tl,sl]+=weight*hop.conj().T
            if a==z:jz[sl,tl]+=1j*hop;jz[tl,sl]+=-1j*hop.conj().T
        projected=[]
        for op in (onsite,mass,zhop,den,jz):
            transformed=u.conj().T@op@u;transformed[abs(transformed)<2e-12]=0;projected.append(onebody(masks,transformed))
        local.append(projected[:3]);density.append(projected[3]);zcurrent.append(projected[4])
    matterlabels=np.array([sum(int(geo['momenta'][i,2]) for i in range(nc) if mask&(1<<i))%5 for mask in masks]);virtual=np.array([sum(bool(mask&(1<<i)) for i in range(geo['sea_count'],nc)) for mask in masks]);angular=np.zeros((nf,nf),complex)
    for band,pol in product(range(2),repeat=2):
        cc,ss=4*band+2*pol,4*band+2*pol+1;angular+=(band+1)*(f['coordinates'][cc,:,None]*f['momenta'][ss]-f['coordinates'][ss,:,None]*f['momenta'][cc])
    def current_action(state,axis):
        x=state.reshape(n,nb);out=np.zeros_like(x)
        for k in range(2,len(matrices),2):
            coefficient=geo['profiles'][k,axis]
            if coefficient:out+=coefficient*((matrices[k]@x)*factors[k+1]-(matrices[k+1]@x)*factors[k])
        return out.ravel()
    def observe(state,t,off=False):
        x=state.reshape(n,5,nf);flat=x.reshape(n,nb);prob=abs(x)**2;weights=prob.sum(axis=(1,2));fieldweights=prob.sum(axis=(0,1));q=f['coordinates']@fieldweights;pmean=np.array([np.vdot(state,field_action(state,m,nf)).real for m in f['momenta']]);scalarenergy=float(np.vdot(x,np.einsum('ab,mbf->maf',f['phi_h'],x,optimize=True)).real);profiles=[]
        def matter(matrix):return float(np.vdot(flat,matrix@flat).real)
        for z in range(5):
            onsite,mass,zlinks=local[z];terms=dict(onsite=matter(onsite),Phi_matter=float(np.vdot(flat,(mass@flat)*factors[1]).real),z_bond=matter(zlinks),transverse_Peierls=0.,global_Phi=scalarenergy/5,electric=float(np.vdot(state,field_action(state,f['electric_profile'][z],nf)).real),reference=-geo['offset']/5)
            for pol in range(2):
                k=2+2*(5*pol+z)
                for index in (k,k+1):terms['transverse_Peierls']+=float(np.vdot(flat,(matrices[index]@flat)*(float(index%2==0) if off else factors[index])).real)
            terms['total']=sum(terms.values());profiles.append(terms)
        energy=float(np.vdot(state,(hoff if off else h)@state).real)
        return dict(time=t,norm=float(np.vdot(state,state).real),energy=energy,profile=profiles,profile_sum_defect=sum(p['total'] for p in profiles)-energy,density=[matter(m) for m in density],z_number_current=[matter(m) for m in zcurrent],transverse_current=[0.]*8 if off else [float(np.vdot(state,current_action(state,a)).real) for a in range(8)],Q=q.tolist(),P=pmean.tolist(),meanA=(f['profiles']@q).tolist(),meanE=(f['profiles']@(pmean/f['inertia'])).tolist(),Phi_distribution=prob.sum(axis=(0,2)).tolist(),Phi_mean=float(prob.sum(axis=(0,2))@f['phi']),positive_reference_occupation=float(weights@virtual),matter_C5_PVM=[float(weights[matterlabels==a].sum()) for a in range(5)],field_angular_mean=float(np.vdot(state,field_action(state,angular,nf)).real),Q_boundary_weight=float(fieldweights[np.abs(f['nodes']).sum(axis=1)==c['Q_radius']].sum()))
    vacuum=observe(ground,0.);write_json(ROOT/(stem+'_ground_observed.json'),vacuum)
    ids=np.flatnonzero(np.all(geo['blocks']==(1,0),axis=1));rho=np.zeros((20,20),complex);gx=ground.reshape(n,nb);mask_lookup={mask:k for k,mask in enumerate(masks)}
    for a,i0 in enumerate(ids):
        i=int(i0)
        for b,j0 in enumerate(ids):
            j=int(j0);rows=[];cols=[];signs=[]
            for column,mask in enumerate(masks):
                if not mask&(1<<j):continue
                first=mask^(1<<j)
                if first&(1<<i):continue
                destination=mask_lookup.get(first|(1<<i))
                if destination is not None:rows.append(destination);cols.append(column);signs.append((-1)**(((mask&((1<<j)-1)).bit_count()+(first&((1<<i)-1)).bit_count())%2))
            rho[a,b]=np.einsum('ij,ij,i->',gx[rows].conj(),gx[cols],np.array(signs),optimize=True) if rows else 0.
    np.savez_compressed(ROOT/(stem+'_source_grams.npz'),orbital_ids=ids,removal=rho,addition=np.eye(20)-rho.T,canonical_orbitals=geo['orbitals'][:,ids])
    # Exact low-group projectors AND full-H source Lanczos measures with raw tails.
    qsource=[];jsource=[]
    for axis in range(8):
        q=(ground.reshape(n,5,nf)*f['coordinates'][axis]).ravel();q-=np.vdot(ground,q)*ground;j=current_action(ground,axis);j-=np.vdot(ground,j)*ground;qsource.append(q);jsource.append(j)
    lowe,lowv=eigsh(h,k=13,which='SA',v0=ground+np.random.default_rng(222).standard_normal(n*nb)*1e-4,tol=2e-8,ncv=28,maxiter=4000);lowv,_=np.linalg.qr(lowv);acted=np.column_stack([h@lowv[:,k] for k in range(len(lowe))]);lowe,rotation=eigh(lowv.conj().T@acted);lowv=lowv@rotation;groups=[]
    for k,energy in enumerate(lowe):
        vector=lowv[:,k];res=float(np.linalg.norm(h@vector-energy*vector));qw=np.array([abs(np.vdot(vector,s))**2 for s in qsource]);jw=np.array([abs(np.vdot(vector,s))**2 for s in jsource])
        if not groups or abs(energy-e0-groups[-1]['energy'])>1e-6:groups.append(dict(energy=float(energy-e0),multiplicity=0,coordinate_axis_weights=[0.]*8,current_axis_weights=[0.]*8,max_residual=0.))
        group=groups[-1];group['multiplicity']+=1;group['coordinate_axis_weights']=(np.array(group['coordinate_axis_weights'])+qw).tolist();group['current_axis_weights']=(np.array(group['current_axis_weights'])+jw).tolist();group['max_residual']=max(group['max_residual'],res)
    for k,group in enumerate(groups):group['complete_below_upper_edge']=k<len(groups)-1 and group['max_residual']<1e-6
    np.savez_compressed(ROOT/(stem+'_lowgroups.npz'),energies=lowe,vectors=lowv);del lowv,acted
    spectra=[]
    for axis in range(8):
        record=dict(axis=axis,coordinate={},literal_current={})
        contact=0.
        for k in range(2,len(matrices),2):
            coefficient=geo['profiles'][k,axis]
            if coefficient:contact-=coefficient**2*float(np.vdot(gx,(matrices[k]@gx)*factors[k]+(matrices[k+1]@gx)*factors[k+1]).real)
        record['contact']=contact
        for label,source in (('coordinate',qsource[axis]),('literal_current',jsource[axis])):
            measure=lanczos(h,source,c['spectral_steps']);record[label]=dict(raw_norm=float(np.vdot(source,source).real),energies=(measure['energies']-e0).tolist(),weights=measure['weights'].tolist(),alpha=measure['alpha'],beta=measure['beta'],last_beta=measure['last_beta'],meaning='Full retained H measure, all raw weight; not replaced by low groups. Negative frequency retarded counterpart uses identical weights and opposite pole sign.')
        spectra.append(record);write_json(ROOT/(stem+'_spectra.json'),dict(groups=groups,spectra=spectra))
    del qsource,jsource
    # A coordinate phase is an EXACT field-only local kick in this representation.
    pdrive=np.zeros(8)
    for band,pol in product(range(2),repeat=2):
        phase=2*np.pi*(band+1)*2/5;pdrive[4*band+2*pol:4*band+2*pol+2]=c['pulse_momentum']/(band+1)*(1. if pol==0 else .5)*np.array([np.cos(phase),np.sin(phase)])
    drive=np.exp(1j*pdrive@f['coordinates']);pulse=(ground.reshape(n,5,nf)*drive).ravel();np.savez_compressed(ROOT/(stem+'_preparation.npz'),ground=ground,pulse=pulse,drive=drive,pdrive=pdrive)
    # Exact SAME entangled-state complete EMoff, retaining quantum sea and Phi.
    matterphi=sparse.kron(matrices[0],sparse.eye(5))+sparse.kron(matrices[1],sparse.diags(c['coupling']*f['phi']))
    for k in range(2,len(matrices),2):matterphi+=sparse.kron(matrices[k],sparse.eye(5))
    matterphi+=sparse.kron(sparse.eye(n),scalar)-geo['offset']*sparse.eye(n*5);trace=float(matterphi.diagonal().sum().real);fieldstep=expm(-1j*c['dt']*f['kinetic']);arms=dict(coupled_pulse=pulse.copy(),same_state_off_pulse=pulse.copy(),same_state_off_baseline=ground.copy());records={a:[] for a in arms};steps=[]
    for index in range(c['time_points']):
        t=index*c['dt']
        for arm,state in arms.items():
            records[arm].append(observe(state,t,arm!='coupled_pulse'))
            if index in (0,(c['time_points']-1)//2,c['time_points']-1):np.savez_compressed(ROOT/(stem+f'_{arm}_state{index}.npz'),state=state,time=t)
        write_json(ROOT/(stem+'_checkpoint.json'),dict(phase='actual_coordinate_packet',ground=vacuum,records=records,steps=steps,elapsed=time.monotonic()-started))
        if index<c['time_points']-1:
            arms['coupled_pulse'],receipt=lanczos(h,arms['coupled_pulse'],c['trajectory_krylov'],c['dt']);steps.append(receipt)
            for arm in ('same_state_off_pulse','same_state_off_baseline'):
                x=arms[arm].reshape(n*5,nf);x=expm_multiply(-1j*c['dt']*matterphi,x,traceA=-1j*c['dt']*trace);arms[arm]=(fieldstep@x.T).T.ravel()
    # Own bare-coordinate kinetic exact degenerate coordinate-source groups.
    bsource=np.column_stack([f['coordinates'][a]*ubare[:,0] for a in range(8)]);amplitude=ubare.conj().T@bsource;baregroups=[]
    for k,energy in enumerate(ebare):
        if not baregroups or abs(energy-baregroups[-1]['energy'])>1e-9:baregroups.append(dict(energy=float(energy),gap=float(energy-ebare[0]),multiplicity=0,axis_weights=[0.]*8))
        baregroups[-1]['multiplicity']+=1;baregroups[-1]['axis_weights']=(np.array(baregroups[-1]['axis_weights'])+abs(amplitude[k])**2).tolist()
    eom=[];hground=h@ground
    for axis in range(8):
        qstate=(ground.reshape(n,5,nf)*f['coordinates'][axis]).ravel();pstate=field_action(ground,f['momenta'][axis],nf);qdef=1j*(h@qstate-(hground.reshape(n,5,nf)*f['coordinates'][axis]).ravel())-pstate/f['inertia'];pdef=1j*(h@pstate-field_action(hground,f['momenta'][axis],nf))-current_action(ground,axis);eom.append(dict(axis=axis,Qdot_minus_P_over_I_norm=float(np.linalg.norm(qdef)),Pdot_minus_literal_J_norm=float(np.linalg.norm(pdef))))
    result=dict(kind='COMPLETED_MATERIAL_COORDINATE_FIELD_ALTERNATIVE',physics=c,geometry=dict(volume=geo['volume'],CAR=nc,sea=geo['sea_count'],matter=n,field=nf,scalar=5,joint=n*nb),energy=e0,eigensolver_residual=residual,ground=vacuum,groups=groups,spectra=spectra,records=records,lanczos_steps=steps,bare_groups=baregroups,field_EOM=eom,electric_profile_operator_sum_defect=f['energy_identity'],wall_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits=['Finite-difference Q-domain L1 radius2,Dirichlet,finiteh; not continuum oscillator equivalence.','All Peierls Q functions commute exactly; Qdot=Pcentral/I exact. Pdot-J may have finiteh/boundary defects,reported.','Electric energy profile uses common before-domain forward-gradient rectangular maps; meanE uses Hermitian central momentum. Difference is explicit finiteh regularization,not fakepositivephotonenergy.','Q lattice breaks exact C5 rotations of Fourier quadratures; no totalC5conservation/jointfieldmomentumPVM claim. MatterC5PVM and fieldangularmean are separate finite observables.','Same complete individualvirtual CI and quantumPhi; independent higherPH outside self-C4(0,0) remain omitted.'])
    write_json(ROOT/(stem+'_data.json'),result);print(json.dumps(dict(mode=mode,energy=e0,residual=residual,joint=n*nb,wall_seconds=result['wall_seconds'],peak_RSS_KiB=result['peak_RSS_KiB'])),flush=True)

if __name__=='__main__':run()
