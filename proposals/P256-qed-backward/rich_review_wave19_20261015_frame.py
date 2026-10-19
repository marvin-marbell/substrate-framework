"""Independent selected native radial6 Frame variant, never primary EM integration."""
import gc
import json
from pathlib import Path
import pickle
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh,expm_multiply,gmres
from scipy.special import roots_genlaguerre,eval_genlaguerre,gammaln
import rich_common_frame_orbit_20261015_model as f
from rich_common_frame_orbit_20261015_reduce import explicit,factorized
from rich_common_frame_orbit_20261015_frame import coordinate_noether


def norm(x):return float(sparse.linalg.norm(x)) if sparse.issparse(x) else float(np.linalg.norm(x))
def mean(x,a):return float(np.vdot(x,a@x).real)
def cj(x):return dict(real=np.asarray(x).real.tolist(),imag=np.asarray(x).imag.tolist())


def eigenspace(h,count=10):
    e,v=eigsh(h,k=min(count,h.shape[0]-2),which='SA',v0=np.random.default_rng(31915).normal(size=h.shape[0]),tol=2e-10,ncv=min(h.shape[0],max(4*count,50)),maxiter=15000)
    q,s,_=np.linalg.svd(v,full_matrices=False);q=q[:,s>1e-8*s[0]];small=q.conj().T@(h@q);e,u=np.linalg.eigh((small+small.conj().T)/2);v=q@u
    return e,v,dict(dimension=h.shape[0],orthogonality=norm(v.conj().T@v-np.eye(len(e))),eigen_residual=max(norm(h@v[:,i]-e[i]*v[:,i]) for i in range(len(e))))


def generators(model,name):
    s=model['sectors'][name]
    return [f.observable(model,name,matter=a) for a in s['frame']['left']],[f.observable(model,name,orientation=a) for a in model['rotor']['L']]


def j2(state,left,orbital):return sum((a@(a@state+b@state)+b@(a@state+b@state) for a,b in zip(left,orbital)),np.zeros_like(state))


def radial_audit(radial,j):
    # Independent higher-order weighted derivative integration in the native r³ form.
    beta=-1+np.sqrt(1+2*j*(j+1));n=np.arange(radial.count);nodes,w=roots_genlaguerre(64,beta);normal=np.exp((gammaln(n+1)-gammaln(n+beta+2))/2)
    lag=np.array([eval_genlaguerre(i,beta+1,nodes) for i in n]);dl=np.array([-eval_genlaguerre(i-1,beta+2,nodes) if i else np.zeros_like(nodes) for i in n]);derivative=beta*lag+2*nodes*dl-nodes*lag
    kinetic=radial.scale/(2*radial.mass)*normal[:,None]*(derivative*w)@derivative.T*normal[None,:]
    frame=j*(j+1)/radial.mass*radial.moment(j,j,-2);actual=radial.operators(j)
    return dict(j=j,beta=beta,Gram_error=norm(radial.moment(j,j,0)-np.eye(radial.count)),weighted_derivative_plus_frame_error=norm(kinetic+frame-actual['kinetic']),min_radial_kinetic_eigenvalue=float(np.linalg.eigvalsh((kinetic+kinetic.T)/2).min()),frame_inertia='Mrad*r²/2',measure='r³dr Haar',origin_epsilon=False)


def conjugations(model,body):
    result={}
    for name,native in body['C_matter'].items():
        anti='neutral' if name=='neutral' else 'anti'+name
        source=model['sectors'][name];target=model['sectors'][anti];matrix=target['spin']['U'].conj().T@native@source['spin']['U']
        result[name]=f.FormOperator(source,target,model['rotor'],model['radial'],[(matrix,np.eye(10),0)])
    return result


def main():
    inp,out=map(Path,sys.argv[1:3]);case=json.loads(inp.read_text());p=json.loads((inp.parent/case['owner_input']).read_text());start=time.monotonic();model=f.build(p);saved={};rows={};all_spectra={};neutral=f.law(model,'neutral');candidates=[];neutral_blocks={}
    for J in range(int(round(max(model['sectors']['neutral']['spin']['j'])+1))+1):
        v=f.coupled(model,'neutral',J,0.)
        if not v.shape[1]:continue
        h,ledger=explicit(neutral,v);e,u,meta=eigenspace(h,case['low_eigenstates']);candidates.append(dict(J=J,energy=float(e[0]),numerics=meta));neutral_blocks[J]=(h,e,u,v)
    candidate=min(candidates,key=lambda x:x['energy']);J0=candidate['J'];h,e,u,v=neutral_blocks[J0];ground=v@u[:,0];energy=e[0];saved['ground']=ground
    left0,orb0=generators(model,'neutral');radial=model['radial'];gauss={name:dict(right_Gauss=s['frame']['Gauss'],isometry=s['frame']['isometry'],left_compression=s['frame']['left_equals_reduced'],coordinate=coordinate_noether(s,int(round(s['op']['N'].diagonal().real.mean())))) for name,s in model['sectors'].items()}
    with (inp.parent/p['native_body']).open('rb') as handle:body=pickle.load(handle)
    cs=conjugations(model,body);del body;gc.collect();source_results={};states={};spectra={};controls={};C_audits={}
    for name,kind,spins in [('lepton','lepton',[.5]),('antilepton','lepton',[.5]),('baryon','baryon',[.5,1.5]),('antibaryon','baryon',[.5,1.5])]:
        s=model['sectors'][name];hfull=f.law(model,name);left,orb=generators(model,name);spectra[name]={};controls[name]={}
        for J in spins:
            family={}
            for M in np.arange(-J,J+1):
                source=f.source(model,name,kind,J,float(M));raw=source@ground;strength=norm(raw)**2;created=raw/np.sqrt(strength)
                gram=f.FormOperator(model['sectors']['neutral'],model['sectors']['neutral'],model['rotor'],radial,[(source.matrix.conj().T@source.matrix,np.eye(10),0)])
                rawgram,_=f.full_raw_source_gram(model,name,kind,J,float(M));family[str(float(M))]=dict(**source.readout,raw_retained_strength=strength,matter_unprojected_radial_strength=mean(ground,gram),native_full_polynomial_strength=mean(ground,rawgram),J2=float(np.vdot(created,j2(created,left,orb)).real),J2_residual=norm(j2(created,left,orb)-J*(J+1)*created),conditional_cost=mean(created,hfull)-energy)
                if M==J:states[name,str(J)]=created;saved[name+'_'+str(J)]=created
            source_results[name+':'+str(J)]=family
            iso=f.coupled(model,name,J,J);projection=factorized if kind=='baryon' else explicit;projected,ledger=projection(hfull,iso);e,u,meta=eigenspace(projected,case['low_eigenstates']);frame=f.observable(model,name,field_kind='frame');fm,unused=projection(frame,iso);eo,uo,om=eigenspace(projected-fm,case['low_eigenstates']);spectra[name][str(J)]=dict(ground_energy=float(e[0]),gap=float(e[0]-energy),numerics=meta,projection=ledger);controls[name][str(J)]=dict(ground_energy=float(eo[0]),numerics=om,meaning='Same finite radial6 basis/domain and coefficients, derived field kinetic removed only; not unlimited radial theory.')
            del projected,fm,u,uo;print(json.dumps({'frame_spectrum':name,'J':J,'gap':spectra[name][str(J)]['gap']}),flush=True)
    rng=np.random.default_rng(919);field_tests=[]
    for a in range(3):
        probe=rng.normal(size=neutral.shape[0])+1j*rng.normal(size=neutral.shape[0]);probe/=norm(probe);total=left0[a]@probe+orb0[a]@probe
        field_tests.append(norm(neutral@total-left0[a]@(neutral@probe)-orb0[a]@(neutral@probe))/max(1.,norm(neutral@probe)))
    for name in ['neutral','lepton','baryon']:
        anti='neutral' if name=='neutral' else 'anti'+name;c=cs[name];h=f.law(model,name);ha=f.law(model,anti);x=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]);x/=norm(x)
        C_audits[name]=dict(H_action_relative=norm(ha@(c@x)-c@(h@x))/max(1.,norm(h@x)),C_isometry_action=norm(c.H@(c@x)-x))
        if name!='neutral':
            checks={}
            for J in ([.5] if name=='lepton' else [.5,1.5]):
                for M in np.arange(-J,J+1):
                    pos=f.source(model,name,name,J,float(M));neg=f.source(model,anti,name,J,float(M));checks[f'{J}:{M}']=norm(neg@(cs['neutral']@ground)-c@(pos@ground))
            C_audits[name]['direct_source_action']=checks
    currents=[f.law(model,'neutral',derivative=(a,)) for a in range(6)];derivative=[];contact=[];step=case['derivative_step'];response=[]
    for a in range(6):
        ext=np.zeros(6);ext[a]=step;hp=f.law(model,'neutral',ext);hm=f.law(model,'neutral',-ext);sec=f.law(model,'neutral',derivative=(a,a));derivative.append(norm((hp@ground-hm@ground)/(2*step)+currents[a]@ground)/max(1.,norm(currents[a]@ground)));contact.append(norm((hp@ground+hm@ground-2*(neutral@ground))/step**2-sec@ground)/max(1.,norm(sec@ground)))
        if a in [2,5]:
            hc,_,_,iso=neutral_blocks[1];image=iso.conj().T@(currents[a]@ground);answers=[];residuals=[]
            for w in case['frequencies']:
                pair=[];rr=[]
                for z in [energy+w+1j*case['eta'],energy-w-1j*case['eta']]:
                    operator=z*sparse.eye(hc.shape[0],format='csr')-hc;solution,flag=gmres(operator,image,rtol=2e-10,atol=1e-12,restart=100,maxiter=400)
                    if flag:raise RuntimeError('Full frame signed resolvent did not converge')
                    pair.append(solution);rr.append(norm(operator@solution-image))
                answers.append(np.vdot(image,pair[0]+pair[1]));residuals.append(rr)
            response.append(dict(channel=a,virtual_dimension=hc.shape[0],strength=norm(image)**2,positive_Hsecond_contact=mean(ground,sec),retarded=cj(answers),residuals=residuals))
    rho=f.observable(model,'neutral',matter=model['sectors']['neutral']['op']['rho0']);route=[f.route_current(model,'neutral',edge) for edge in range(2)];graph_image=1j*(neutral@(rho@ground)-rho@(neutral@ground))+sum((x@ground for x in route),np.zeros_like(ground))
    dipole=f.observable(model,'neutral',matter=model['sectors']['neutral']['op']['density_delta'],orientation=model['rotor']['integrate'](model['rotor']['R'][:,2,0]));scale=p['vertex_separation']/2;lab_image=1j*scale*(neutral@(dipole@ground)-dipole@(neutral@ground))-currents[2]@ground
    nohop=f.law(model,'neutral',hop_factor=0.);jnohop=f.law(model,'neutral',derivative=(2,),hop_factor=0.);lab_nohop=1j*scale*(nohop@(dipole@ground)-dipole@(nohop@ground))-jnohop@ground
    ward=dict(material_graph_norm=norm(graph_image),lab_dipole_current_norm=norm(lab_image),physical_current_image_norm=norm(currents[2]@ground),lab_defect_over_current=norm(lab_image)/norm(currents[2]@ground),lab_no_hops_mechanical_defect=norm(lab_nohop),lab_hopping_piece_defect=norm(lab_image-lab_nohop),meaning='Norms on the same unit newly solved ground, not probabilities/operator norms. Lab dipole and current are compressed Galerkin operators. Material site graph Ward differs from physical lab dipole Ward; no gauge closure inferred.')
    name='lepton';s=model['sectors'][name];initial=states[name,'0.5'];shape=(len(s['spin']['j']),10,radial.count);rotated=initial.reshape(shape)*np.exp(-1j*case['frame_kick']*s['spin']['m'])[:,None,None];prepared=np.zeros_like(rotated)
    for j in set(s['spin']['j']):
        rows=s['spin']['j']==j;prepared[rows]=rotated[rows]@expm(-1j*case['radial_kick']*radial.moment(float(j),float(j),1)).T
    prepared=prepared.ravel();event_states=[np.zeros_like(prepared) for _ in case['times']];baselines=[np.zeros_like(prepared) for _ in case['times']];components=[]
    for twice in range(1,int(round(2*(max(s['spin']['j'])+1)))+1,2):
        J=twice/2;iso=f.coupled(model,name,J,.5)
        if not iso.shape[1]:continue
        hc,_=explicit(f.law(model,name),iso);pair0=np.column_stack([iso.conj().T@prepared,iso.conj().T@initial]);components.append(dict(J=J,weight=norm(pair0[:,0])**2,dimension=hc.shape[0]))
        for i,t in enumerate(case['times']):
            evolved=expm_multiply(-1j*t*hc,pair0,traceA=-1j*t*hc.diagonal().sum());event_states[i]+=iso@evolved[:,0];baselines[i]+=iso@evolved[:,1]
    h=f.law(model,name);left,orb=generators(model,name);obs={'r2':f.observable(model,name,field_kind='r2'),'frame_energy':f.observable(model,name,field_kind='frame'),'radial_energy':f.observable(model,name,field_kind='radial'),'colour_electric':f.observable(model,name,matter=s['op']['electric']),'plaquette':f.observable(model,name,matter=s['op']['B'])};event=[];byid={x['id']:x['charge3'] for x in s['spin']['multiplets']};charge_labels=np.array([byid[x[0]] for x in s['spin']['labels']])
    for t,x,b in zip(case['times'],event_states,baselines):
        weights=np.sum(abs(x.reshape(shape))**2,axis=(1,2));pw={str(int(q)):float(weights[charge_labels==q].sum()) for q in sorted(set(charge_labels))};hstate=h@x
        event.append(dict(time=t,norm=norm(x)**2,excess={key:mean(x,a)-mean(b,a) for key,a in obs.items()},field_left=[mean(x,a) for a in left],material_orbit=[mean(x,a) for a in orb],torque_total=[float(2*np.vdot(a@x+bop@x,hstate).imag) for a,bop in zip(left,orb)],charge_PVM_h_equals_3rho0=pw,charge_PVM_sum=sum(pw.values())))
    ordering={name:dict(full=spectra[name]['1.5']['ground_energy']-spectra[name]['0.5']['ground_energy'],frame_off=controls[name]['1.5']['ground_energy']-controls[name]['0.5']['ground_energy']) for name in ['baryon','antibaryon']}
    for row in ordering.values():row['positive_increment']=row['full']-row['frame_off']
    saved.update(prepared=prepared,event_states=np.array(event_states),baseline_states=np.array(baselines))
    result=dict(kind='WAVE19_NEW_NATIVE_RADIAL6_FRAME_VARIANT_NOT_PRIMARY_EM',case=case,parameters=p,reference=dict(energy=float(energy),J=J0,candidates=candidates),radial=[radial_audit(radial,float(j)) for j in model['frame_js']],raw_frame_Gauss_Noether=gauss,sources=source_results,spectra=spectra,frame_off=controls,ordering=ordering,actual_C=C_audits,H_total_Noether_relative_probes=field_tests,current=dict(derivative=derivative,positive_contact=contact,full_signed_responses=response,frequencies=case['frequencies'],eta=case['eta']),Ward=ward,event=dict(components=components,preparation='Unfiltered actual source, field LEFT-g z rotation and radial unitary only; nonstationary same initial baseline, not photons or primitive-CAR rotation.',records=event),resource=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.monotonic()-start),limits='Chosen RIGHT gauging/CAR/topology and minimal cone quantization inputs. Primitive CAR physical spin zero; do not double-count old S. Native colour provenance supplied, not reconstructed here. Native6 off already positive; n4 sign reversal not convergence/necessity. Frame variant different law from primary SpatialColourEM.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**saved);print(json.dumps(dict(ordering=ordering,resource=result['resource'],Ward=ward)),flush=True)


if __name__=='__main__':main()
