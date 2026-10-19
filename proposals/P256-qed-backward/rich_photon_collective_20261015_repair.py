"""Controlled SHORT same-H/same-t0 numerical repair; no physical cut or law change.
Independent temporal grids + integrated physical Krylov defects, not conservation alone.
"""
from itertools import product
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.integrate import quad
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import LinearOperator, expm_multiply
from rich_photon_collective_20261015_model import geometry, onebody, install, field_action
from rich_mobile_packet_20261015_model import shell_states

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'

def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def krylov_step(operator,state,duration,directions):
    norm=float(np.linalg.norm(state));v=state/norm;previous=np.zeros_like(v);last=0.;basis=[];alpha=[];beta=[];corrections=[]
    for index in range(directions):
        basis.append(v);w=operator@v-last*previous;diagonal=np.vdot(v,w);a=float(diagonal.real);w-=a*v;correction=abs(diagonal.imag)
        for _ in range(2):
            for retained in basis:
                coefficient=np.vdot(retained,w);w-=coefficient*retained;correction+=abs(coefficient)
        following=float(np.linalg.norm(w));alpha.append(a);corrections.append(float(correction))
        if following<1e-12 or index==directions-1:break
        beta.append(following);previous=v;v=w/following;last=following
    tridiagonal=np.diag(alpha)+np.diag(beta,1)+np.diag(beta,-1);energies,rotation=eigh(tridiagonal);weights=rotation[-1]*rotation[0].conj()
    def residual(t):return float(norm*following*abs(np.dot(weights,np.exp(-1j*t*energies))))
    integrated,error=quad(residual,0,duration,epsabs=1e-14,epsrel=1e-7,limit=100);coefficients=rotation@(np.exp(-1j*duration*energies)*rotation[0].conj());out=np.zeros_like(state)
    for vector,coefficient in zip(basis,coefficients):out+=norm*coefficient*vector
    recurrence=norm*duration*float(np.linalg.norm(corrections));receipt=dict(duration=duration,directions=len(alpha),endpoint_residual=residual(duration),integrated_residual_estimate=float(integrated),residual_quadrature_reported_error=float(error),measured_reorthogonalization_correction_allowance=recurrence,local_defect_indicator=float(integrated+abs(error)+recurrence),norm=float(np.linalg.norm(out)),meaning='Duhamel integrated physical Krylov residual estimate plus measured recurrence corrections; quadrature/roundoff NOT a rigorous machine-proof bound. Independent temporal-grid full-state/observable comparisons remain separately measured.')
    return out,receipt

def advance(operator,state,duration,directions,tolerance,depth=0):
    out,receipt=krylov_step(operator,state,duration,directions)
    if receipt['local_defect_indicator']<=tolerance:return out,[receipt]
    if depth>=10:raise RuntimeError(f'Cannot meet local defect indicator: {receipt}')
    middle,left=advance(operator,state,duration/2,directions,tolerance/2,depth+1);out,right=advance(operator,middle,duration/2,directions,tolerance/2,depth+1)
    return out,[dict(rejected_parent_step=receipt,reason='integrated defect indicator exceeds declared tolerance'),*left,*right]

def run():
    started=time.monotonic();mode=sys.argv[1];stem=PREFIX+mode;request=json.loads((ROOT/(stem+'_physics.json')).read_text());base=PREFIX+request['base_mode'];c=json.loads((ROOT/(base+'_physics.json')).read_text());kind=request['representation'];archive=np.load(ROOT/(base+'_primitive.npz'));masks=[int(s) for s in json.loads((ROOT/(base+'_masks.json')).read_text())];n=len(masks);profiles=archive['profiles'];matrices=[sparse.load_npz(ROOT/(base+f'_Hcomponent{k}.npz')) for k in range(len(profiles))];offset=float(archive['offset']);phi=archive['phi'];phi_h=archive['phi_h'];kinetic=archive['kinetic'];orbitals=archive['orbitals'];momenta_matter=archive['momenta'];ground_file=np.load(ROOT/(base+'_ground.npz'));ground=ground_file['state'];e0=float(ground_file['energy']);preparation=np.load(ROOT/(base+'_preparation.npz'));pulse=preparation['pulse'];nf=len(kinetic);nb=5*nf;volume=5*c['transverse']**2;inertia=volume/2
    if kind=='oscillator':
        q=archive['q'];p=archive['p'];terms=[sparse.csr_matrix(term) for term in archive['terms']];boson=archive['boson'];offterms=[sparse.eye(nb,format='csr'),terms[1]]
        for _ in range(2,len(matrices),2):offterms.extend([sparse.eye(nb,format='csr'),sparse.csr_matrix((nb,nb),dtype=complex)])
        h=install(matrices,terms,boson,offset);hoff=install(matrices,offterms,boson,offset);states=[tuple(map(int,s)) for s in archive['states']];large=shell_states(c['shell_cutoff']+1);look={s:i for i,s in enumerate(large)};rect=[]
        for axis in range(8):
            pr=np.zeros((len(large),nf),complex)
            for j,s in enumerate(states):
                target=list(s);target[axis]+=1;pr[look[tuple(target)],j]=1j*np.sqrt(target[axis]/2)/c['shell_width']
                if s[axis]:target=list(s);target[axis]-=1;pr[look[tuple(target)],j]=-1j*np.sqrt(s[axis]/2)/c['shell_width']
            rect.append(pr)
        field_profiles=np.zeros((2,5,8));electric_profile=[]
        for z in range(5):
            operator=np.zeros((nf,nf),complex)
            for band,pol in product(range(2),repeat=2):field_profiles[pol,z,4*band+2*pol:4*band+2*pol+2]=[np.cos(2*np.pi*(band+1)*z/5),np.sin(2*np.pi*(band+1)*z/5)]
            for pol in range(2):
                derivative=sum(field_profiles[pol,z,a]*rect[a] for a in range(8))/inertia;operator+=(volume/5)*derivative.conj().T@derivative/2
            electric_profile.append(operator)
        annihilation=[(qq/c['shell_width']+1j*c['shell_width']*pp)/np.sqrt(2) for qq,pp in zip(q,p)];angular=np.zeros((nf,nf),complex)
        for band,pol in product(range(2),repeat=2):
            cc,ss=4*band+2*pol,4*band+2*pol+1;angular+=(band+1)*1j*(annihilation[ss].conj().T@annihilation[cc]-annihilation[cc].conj().T@annihilation[ss])
        ell,field_rotation=eigh(angular);field_labels=np.rint(ell).astype(int)%5;nodes=None
        def term_action(index,x,off=False):return ((offterms if off else terms)[index]@x.T).T
    else:
        coordinates=archive['coordinates'];q=np.array([np.diag(axis) for axis in coordinates]);p=archive['field_momenta'];factors=archive['factors'].reshape(len(matrices),nb);electric_profile=list(archive['electric_profile']);field_profiles=archive['spatial_profiles'];nodes=archive['nodes'];kfield=sparse.csr_matrix(kinetic);angular=np.zeros((nf,nf),complex)
        for band,pol in product(range(2),repeat=2):
            cc,ss=4*band+2*pol,4*band+2*pol+1;angular+=(band+1)*(coordinates[cc,:,None]*p[ss]-coordinates[ss,:,None]*p[cc])
        def term_action(index,x,off=False):
            if off and index>=2:return x if index%2==0 else np.zeros_like(x)
            return x*factors[index]
        def make(off=False):
            def action(vector):
                x=np.asarray(vector).reshape(n,5,nf);out=(kfield@x.reshape(n*5,nf).T).T.reshape(x.shape)+np.einsum('ab,mbf->maf',phi_h,x,optimize=True)-offset*x;result=out.reshape(n,nb);flat=x.reshape(n,nb)
                for index,matrix in enumerate(matrices):result+=term_action(index,matrix@flat,off)
                return result.ravel()
            return LinearOperator((n*nb,n*nb),matvec=action,rmatvec=action,dtype=complex)
        h=make();hoff=make(True);field_rotation=None;field_labels=None
    del archive
    geo=geometry(c)
    for key in ('components','transformed','orbitals','reference'):del geo[key]
    nc=len(orbitals);local=[];density=[];zcurrent=[]
    for z in range(5):
        onsite=np.zeros((nc,nc),complex);mass=np.zeros_like(onsite);zlinks=np.zeros_like(onsite);den=np.zeros_like(onsite);current=np.zeros_like(onsite)
        for site,coordinate in enumerate(geo['cells']):
            if coordinate[2]==z:
                sl=slice(4*site,4*site+4);onsite[sl,sl]=3*c['wilson']*geo['cliff'][0];mass[sl,sl]=geo['cliff'][0];den[sl,sl]=np.eye(4)
        for site,end,axis,level,hop in geo['links']:
            if axis!=2:continue
            a=geo['cells'][site,2];b=geo['cells'][end,2];fraction=.5*(int(a==z)+int(b==z));sl=slice(4*site,4*site+4);tl=slice(4*end,4*end+4);zlinks[sl,tl]+=fraction*hop;zlinks[tl,sl]+=fraction*hop.conj().T
            if a==z:current[sl,tl]+=1j*hop;current[tl,sl]+=-1j*hop.conj().T
        projected=[]
        for primitive in (onsite,mass,zlinks,den,current):
            transformed=orbitals.conj().T@primitive@orbitals;transformed[abs(transformed)<2e-12]=0;projected.append(onebody(masks,transformed))
        local.append(projected[:3]);density.append(projected[3]);zcurrent.append(projected[4])
    matter_labels=np.array([sum(int(momenta_matter[i,2]) for i in range(nc) if mask&(1<<i))%5 for mask in masks]);virtual=np.array([sum(bool(mask&(1<<i)) for i in range(nc//2,nc)) for mask in masks]);electric_identity=float(np.linalg.norm(sum(electric_profile)-kinetic))
    def jaction(state,axis):
        x=state.reshape(n,nb);out=np.zeros_like(x)
        for k in range(2,len(matrices),2):
            coefficient=profiles[k,axis]
            if coefficient:out+=coefficient*(term_action(k+1,matrices[k]@x)-term_action(k,matrices[k+1]@x))
        return out.ravel()
    def observe(state,t,off=False,transfer=False):
        x=state.reshape(n,5,nf);flat=x.reshape(n,nb);prob=abs(x)**2;field_gram=x.reshape(n*5,nf).conj().T@x.reshape(n*5,nf);field_expect=lambda op:float(np.einsum('ab,ab->',field_gram,op).real);qmean=np.array([field_expect(op) for op in q]);pmean=np.array([field_expect(op) for op in p]);scalar_state=np.einsum('ab,mbf->maf',phi_h,x,optimize=True).reshape(n,nb);scalar_energy=float(np.vdot(flat,scalar_state).real);parts=[];layer_states=[];jmean=np.zeros(8)
        for z in range(5):
            onsite,mass,zlinks=local[z];onstate=onsite@flat;massstate=term_action(1,mass@flat);zstate=zlinks@flat;transverse=np.zeros_like(flat)
            for pol in range(2):
                k=2+2*(5*pol+z);mcos=matrices[k]@flat;msin=matrices[k+1]@flat;transverse+=term_action(k,mcos,off)+term_action(k+1,msin,off)
                if not off:
                    force=float(np.vdot(flat,term_action(k+1,mcos)-term_action(k,msin)).real);jmean+=profiles[k]*force
            electric=field_expect(electric_profile[z]);record=dict(onsite=float(np.vdot(flat,onstate).real),Phi_matter=float(np.vdot(flat,massstate).real),z_bond=float(np.vdot(flat,zstate).real),transverse_Peierls=float(np.vdot(flat,transverse).real),global_Phi=scalar_energy/5,electric=electric,reference=-offset/5*float(np.vdot(state,state).real));record['total']=sum(record.values());parts.append(record)
            if transfer:layer_states.append((onstate+massstate+zstate+transverse+scalar_state/5+field_action(state,electric_profile[z],nf).reshape(n,nb)-offset/5*flat).ravel())
        acted=(hoff if off else h)@state;energy=float(np.vdot(state,acted).real);weights=prob.sum(axis=(1,2));matter=[float(weights[matter_labels==label].sum()) for label in range(5)];record=dict(time=t,norm=float(np.vdot(state,state).real),energy=energy,profile=parts,profile_sum_defect=sum(p['total'] for p in parts)-energy,density=[float(np.vdot(flat,op@flat).real) for op in density],z_number_current=[float(np.vdot(flat,op@flat).real) for op in zcurrent],transverse_current=jmean.tolist(),Q=qmean.tolist(),P=pmean.tolist(),meanA=(field_profiles@qmean).tolist(),meanE=(field_profiles@(pmean/inertia)).tolist(),Phi_distribution=prob.sum(axis=(0,2)).tolist(),Phi_mean=float(prob.sum(axis=(0,2))@phi),positive_reference_occupation=float(weights@virtual),matter_C5_PVM=matter,electric=sum(p['electric'] for p in parts),scalar=scalar_energy)
        if kind=='oscillator':
            amplitudes=(x.reshape(n*5,nf)@field_rotation.conj()).reshape(n,5,nf);joint=np.zeros((5,5));fieldweights=(abs(amplitudes)**2).sum(axis=1)
            for a,b in product(range(5),repeat=2):joint[a,b]=fieldweights[matter_labels==a][:,field_labels==b].sum()
            record['joint_matter_field_C5']=joint.tolist();record['total_C5']=[float(sum(joint[a,b] for a,b in product(range(5),repeat=2) if (a+b)%5==label)) for label in range(5)]
        else:record['field_angular_mean']=field_expect(angular);record['Q_boundary_weight']=float(prob.sum(axis=(0,1))[abs(nodes).sum(axis=1)==c['Q_radius']].sum())
        if transfer:
            exchange=np.array([[-2*np.vdot(layer_states[w],layer_states[z]).imag for w in range(5)] for z in range(5)]);derivative=np.array([-2*np.vdot(acted,layer_states[z]).imag for z in range(5)]);record['energy_transfer']=dict(layer_action_sum_defect=float(np.linalg.norm(sum(layer_states)-acted)),antisymmetric_matrix=exchange.tolist(),layer_energy_derivative=derivative.tolist(),row_sum_defect=float(np.linalg.norm(exchange.sum(axis=1)-derivative)),meaning='i<[H_w,H_z]> for declared bond/globalPhi partition; includes NONLOCAL transfer,not a nearest-neighbor local energy current.')
        return record
    ground_residual=float(np.linalg.norm(h@ground-e0*ground));vacuum=observe(ground,0.,transfer=True);initial=observe(pulse,0.,transfer=True);dump(ROOT/(stem+'_checkpoint.json'),dict(phase='same_H_t0_reconstructed',base=request['base_mode'],ground=vacuum,pulse=initial,ground_residual=ground_residual,elapsed=time.monotonic()-started))
    # Actual same-state full matterPhi/electric Kronecker-sum controls.
    matterphi=sparse.kron(matrices[0],sparse.eye(5))+sparse.kron(matrices[1],sparse.diags(c['coupling']*phi))
    for k in range(2,len(matrices),2):matterphi+=sparse.kron(matrices[k],sparse.eye(5))
    matterphi+=sparse.kron(sparse.eye(n),sparse.csr_matrix(phi_h))-offset*sparse.eye(n*5);trace=float(matterphi.diagonal().sum().real)
    def off_evolve(initial_state,t):
        state=expm_multiply(-1j*t*matterphi,initial_state.reshape(n*5,nf),traceA=-1j*t*trace);return (expm(-1j*t*kinetic)@state.T).T.ravel()
    states=dict(coarse=pulse.copy(),fine=pulse.copy());records=dict(coarse=[initial],fine=[initial],same_state_off_pulse=[observe(pulse,0.,True)],same_state_off_baseline=[observe(ground,0.,True)]);steps=dict(coarse=[],fine=[]);comparisons=[];times=request['times'];last=0.
    np.savez_compressed(ROOT/(stem+'_t0.npz'),ground=ground,pulse=pulse)
    for sample,t in enumerate(times[1:],start=1):
        for grid in ('coarse','fine'):
            resume=request.get('resume_coarse')
            if grid=='coarse' and sample==1 and resume is not None:
                saved=np.load(ROOT/resume['file'])
                if float(saved['time'])!=t or saved['state'].shape!=ground.shape:raise RuntimeError('Preserved coarse state doesnotmatch declared time/fullH dimensions')
                states[grid]=saved['state'];steps[grid].append(dict(time=t,preserved_completed_state=resume['file'],individual_step_receipts='UNKNOWN: original failed fine-grid run exited before serializing coarse receipts',not_replayed=True))
            else:
                maximum=request[grid+'_dt'];directions=request[grid+'_krylov'];duration=t-last;count=int(np.ceil(duration/maximum));dt=duration/count
                for _ in range(count):states[grid],receipt=advance(h,states[grid],dt,directions,request[grid+'_step_defect_tolerance']);steps[grid].extend(receipt)
            records[grid].append(observe(states[grid],t,transfer=grid=='fine'));np.savez_compressed(ROOT/(stem+f'_{grid}_state{sample}.npz'),state=states[grid],time=t)
            dump(ROOT/(stem+'_checkpoint.json'),dict(phase='controlled_grid_anchor',grid=grid,time=t,ground=vacuum,records=records,comparisons=comparisons,steps=steps,elapsed=time.monotonic()-started))
        overlap=np.vdot(states['fine'],states['coarse']);aligned=states['coarse']*np.exp(-1j*np.angle(overlap));coarse=records['coarse'][-1];fine=records['fine'][-1];comparisons.append(dict(time=t,raw_state_L2=float(np.linalg.norm(states['coarse']-states['fine'])),phase_aligned_state_L2=float(np.linalg.norm(aligned-states['fine'])),overlap_real=float(overlap.real),overlap_imag=float(overlap.imag),total_energy_profile_L2=float(np.linalg.norm(np.array([p['total'] for p in coarse['profile']])-np.array([p['total'] for p in fine['profile']]))),current_L2=float(np.linalg.norm(np.array(coarse['transverse_current'])-np.array(fine['transverse_current']))),meanE_L2=float(np.linalg.norm(np.array(coarse['meanE'])-np.array(fine['meanE']))),Phi_mean_difference=coarse['Phi_mean']-fine['Phi_mean'],meaning='Independent temporal grids and Krylov directions from SAME exact t0/H; empirical full-state/observable agreement,not an all-roundoff rigorous error theorem.'))
        for arm,start_state in (('same_state_off_pulse',pulse),('same_state_off_baseline',ground)):
            state=off_evolve(start_state,t);records[arm].append(observe(state,t,True));np.savez_compressed(ROOT/(stem+f'_{arm}_state{sample}.npz'),state=state,time=t)
        last=t;dump(ROOT/(stem+'_checkpoint.json'),dict(phase='controlled_short_interval',time=t,ground=vacuum,records=records,comparisons=comparisons,steps=steps,elapsed=time.monotonic()-started))
    # Physical finite-cut EOM, not a commutator definition of current.
    eom=[]
    for label,state in (('ground',ground),('fine_final',states['fine'])):
        hstate=h@state
        for axis in (0,4):
            qstate=field_action(state,q[axis],nf);pstate=field_action(state,p[axis],nf);qdef=1j*(h@qstate-field_action(hstate,q[axis],nf))-pstate/inertia;pdef=1j*(h@pstate-field_action(hstate,p[axis],nf))-jaction(state,axis);eom.append(dict(state=label,axis=axis,Qdot_minus_P_over_I_norm=float(np.linalg.norm(qdef)),Pdot_minus_literal_J_norm=float(np.linalg.norm(pdef))))
    bare_e,bare_v=eigh(kinetic);bare_sources=np.column_stack([qq@bare_v[:,0] for qq in q]);projection=bare_v.conj().T@bare_sources;bare_groups=[]
    for index,energy in enumerate(bare_e):
        if not bare_groups or abs(energy-bare_groups[-1]['energy'])>1e-9:bare_groups.append(dict(energy=float(energy),gap=float(energy-bare_e[0]),multiplicity=0,axis_weights=[0.]*8))
        bare_groups[-1]['multiplicity']+=1;bare_groups[-1]['axis_weights']=(np.array(bare_groups[-1]['axis_weights'])+abs(projection[index])**2).tolist()
    source_spectra=json.loads((ROOT/(base+'_spectra.json')).read_text());result=dict(kind='NATURALLY_COMPLETED_CONTROLLED_SHORT_SAME_H_NUMERICAL_REPAIR',base_mode=request['base_mode'],representation=kind,physics=c,numerical_request=request,geometry=dict(volume=volume,CAR=nc,sea=nc//2,matter=n,field=nf,scalar=5,joint=n*nb),energy=e0,eigensolver_residual=ground_residual,ground=vacuum,records=records,lanczos_steps=steps,comparisons=comparisons,field_EOM=eom,bare_groups=bare_groups,electric_profile_operator_sum_defect=electric_identity,source_spectra=source_spectra,groups=source_spectra.get('groups',[]),spectra=source_spectra['spectra'],wall_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits=['SHORT same-H t0 interval only,not full0..10/16/24 rerun or long-distance propagation claim.','Old dt1/K28 outputs preserved as numericallyuncontrolled,NOT negativephysicaltransport evidence.','Full-state coarse/fine agreement plus integrateddefect measured; no rigorous continuum/projection/globalmachineerror theorem.','No physical geometry/cut/coupling/state change; all source/operator inputs pinned.','Original representation-specific quantumPhi/globalenergy/CI/coordinate-domain caveats remain.'])
    dump(ROOT/(stem+'_data.json'),result);print(json.dumps(dict(mode=mode,wall_seconds=result['wall_seconds'],peak_RSS_KiB=result['peak_RSS_KiB'],comparisons=comparisons)),flush=True)

if __name__=='__main__':run()
