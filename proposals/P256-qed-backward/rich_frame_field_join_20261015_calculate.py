"""Fresh finite spectra/source/field-only reaction from the new joined H."""
import json
from pathlib import Path
import sys,time
import numpy as np
from scipy import sparse,linalg
from scipy.sparse.linalg import eigsh
from rich_frame_field_join_20261015_model import build,law,fibre,observable,source,compress
from rich_common_frame_orbit_20261015_frame import coordinate_noether,radial_form_readout
ROOT=Path(__file__).resolve().parent


def real(z):return float(np.real(z))
def expect(a,x):return real(np.vdot(x,a@x))
def norm(x):return float(np.linalg.norm(x))
def eigen(h,k,tol):
    rng=np.random.default_rng(2221015);v=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]);e,u=eigsh(h,k=k,which='SA',tol=tol,v0=v,ncv=max(30,3*k));order=np.argsort(e);e=e[order];u=u[:,order]
    return e,u,[norm(h@u[:,i]-e[i]*u[:,i]) for i in range(k)]


def krylov(h,x,t,steps):
    """Hermitian Lanczos with full reorthogonalization and visible final defect."""
    length=norm(x);q=x/length;basis=[];alpha=[];beta=[];old=np.zeros_like(q);last=0.
    for iteration in range(steps):
        basis.append(q);w=h@q-(beta[-1]*old if beta else 0);a=real(np.vdot(q,w));alpha.append(a);w-=a*q
        for _ in range(2):
            for b in basis:w-=np.vdot(b,w)*b
        last=norm(w)
        if last<1e-13:break
        old=q;q=w/last
        if iteration+1<steps:beta.append(last)
    tridiagonal=np.diag(alpha)+np.diag(beta,1)+np.diag(beta,-1);e,u=linalg.eigh(tridiagonal);coeff=u@(np.exp(-1j*t*e)*u[0].conj())
    result=sum((c*b for c,b in zip(coeff,basis)),np.zeros_like(x))*length
    return result,dict(dimension=len(basis),last_residual=last,last_exponential_coefficient=float(abs(coeff[-1])),endpoint_Schrodinger_defect=float(last*abs(coeff[-1])*length),norm=norm(result))


def m_isometry(model,name,M):
    spin=model['sectors'][name]['spin'];rq=model['rqspin'];nr=model['radial'].count
    selected=np.flatnonzero(abs((spin['m'][:,None]+rq['m'][None,:]).ravel()-M)<1e-8)
    v=sparse.csr_matrix((np.ones(len(selected)),(selected,np.arange(len(selected)))),shape=(len(spin['m'])*len(rq['m']),len(selected)))
    return sparse.kron(v,sparse.eye(nr),format='csr')


def main():
    input_path=Path(sys.argv[1]);out=Path(sys.argv[2]);p=json.loads(input_path.read_text());start=time.time();model=build(p);npz={};result={'parameters':p,'dimensions':{n:len(s['spin']['j']) for n,s in model['sectors'].items()},'field_dimension':model['field']['dimension'],'rq_unitarity':model['rqspin']['unitarity'],'native_payload':model['native_payload']}
    result['frame']={n:dict(Gauss=s['frame']['Gauss'],left=s['frame']['left_equals_reduced'],coordinate=coordinate_noether(s,16 if n=='neutral' else 17 if n=='lepton' else 15)) for n,s in model['sectors'].items() if n in ['neutral','lepton','antilepton']}
    result['radial']=[radial_form_readout(model['radial'],float(j)) for j in sorted(set(model['sectors']['lepton']['spin']['j']))]
    rotor=model['physical_rotor'];basis=rotor['basis'];weights=rotor['weights'];result['complex_rotor_projection']={'correct_bra_conjugate_Gram_error':norm(basis.conj().T@(weights[:,None]*basis)-np.eye(10)),'wrong_real_basis_shortcut_defect':norm(basis.T@(weights[:,None]*basis)-np.eye(10)),'law':'physical_phase uses conjugated spherical bra before joint R/Q projection'}
    v0=fibre(model,'neutral',0.,0.);h0,a0=compress(law(model,'neutral'),v0);e0,u0,r0=eigen(h0,p['eigenstates'],p['eigen_tolerance']);z=u0[:,0];fullz=v0@z
    result['reference']={'condition':'J=0 conditional reference, not global vacuum','energies':e0.tolist(),'residuals':r0,'dimension':h0.shape[0],'factorization':a0};npz.update(reference=z,reference_energies=e0)
    result['reference']['full_H_outside_total_J_fibre']=norm(law(model,'neutral')@fullz-v0@(h0@z))
    # A checkpoint exists before charged spectra and all response propagation.
    np.savez_compressed(out.with_name(out.stem+'_reference.npz'),**npz)
    rows={};seeds={}
    for name in ['lepton','antilepton']:
        v=fibre(model,name,.5,.5);h,a=compress(law(model,name),v);e,u,r=eigen(h,p['eigenstates'],p['eigen_tolerance']);npz[name+'_energies']=e;npz[name+'_vectors']=u
        off,ao=compress(law(model,name,coupled=False),v);eo,uo,ro=eigen(off,p['eigenstates'],p['eigen_tolerance'])
        rows[name]={'J':.5,'M':.5,'dimension':h.shape[0],'energies':e.tolist(),'gaps':(e-e0[0]).tolist(),'residuals':r,'full_transverse_off_energies':eo.tolist(),'full_transverse_off_residuals':ro,'factorization':a,'sources':[]}
        rows[name]['full_H_outside_total_J_fibre']=norm(law(model,name)@(v@u[:,0])-v@(h@u[:,0]))
        for site in [0,1]:
            for m in [-.5,.5]:
                sm,sa=source(model,name,.5,m,site);raw=sm@fullz;vm=fibre(model,name,.5,m);image=vm.conj().T@raw;weight=norm(image)**2
                entry={'site':site,'M':m,'norm_squared':weight,'angular_fibre_leakage':norm(raw-vm@image),'source_Gauss':sa,'energy_cost':expect(h,image)/weight-e0[0]}
                if m==.5:
                    entry['pole_weights']=abs(u.conj().T@image)**2;entry['pole_weights']=entry['pole_weights'].tolist();entry['window_fraction']=float(sum(entry['pole_weights'])/weight)
                rows[name]['sources'].append(entry)
                npz[f'{name}_source_site{site}_m{int(2*m)}']=image
                if site==0 and m==.5:seeds[name]=raw/norm(raw)
    result['stationary']=rows
    result['C_spectral_max']=float(np.max(abs(np.array(rows['lepton']['energies'])-np.array(rows['antilepton']['energies']))))
    # Both CAR +/- source norms are computed, not silently normalized away.
    result['CAR_source_sum']=[dict(site=site,M=m,addition=next(x['norm_squared'] for x in rows['lepton']['sources'] if x['site']==site and x['M']==m),removal=next(x['norm_squared'] for x in rows['antilepton']['sources'] if x['site']==site and x['M']==m)) for site in [0,1] for m in [-.5,.5]]
    np.savez_compressed(out.with_name(out.stem+'_sources.npz'),**npz)
    out.write_text(json.dumps(result,indent=2)+'\n')
    events={}
    for name in ['lepton','antilepton']:
        print('event',name,flush=True);vm=m_isometry(model,name,.5);h,ha=compress(law(model,name),vm);off,oa=compress(law(model,name,coupled=False),vm)
        seed=vm.conj().T@seeds[name];op=model['sectors'][name]['op'];obs={}
        for label,a,b in [('rho0',op['rho0'],None),('LEFT_z',op['S'][2],None),('material_z',None,model['rqobs']['material'][2]),('field_z',None,model['rqobs']['field'][2]),('HE',None,model['rqobs']['HE']),('boundary',None,model['rqobs']['boundary'])]:obs[label]=compress(observable(model,name,a,b),vm)[0]
        for i in range(6):
            obs['Q'+str(i)]=compress(observable(model,name,rq=model['rqobs']['Q'][i]),vm)[0];obs['Pi'+str(i)]=compress(observable(model,name,rq=model['rqobs']['Pi'][i]),vm)[0]
        # Every external source derivative is retained for current/contact response.
        currents=[compress(law(model,name,derivative=(i,)),vm)[0] for i in range(6)]
        contacts=[compress(law(model,name,derivative=(i,i)),vm)[0] for i in range(6)]
        kick=obs['Q2'];kicked,kledger=krylov(kick,seed,-p['field_kick'],p['kick_steps'])
        baseline={key:expect(a,seed) for key,a in obs.items()};initial={key:expect(a,kicked) for key,a in obs.items()}
        npz[name+'_event_seed']=seed;npz[name+'_field_kicked']=kicked
        def measure(x):
            q={key:expect(a,x) for key,a in obs.items()};q['currents']=[expect(a,x) for a in currents];q['contact']=[expect(a,x) for a in contacts];q['energy']=expect(h,x);q['norm']=norm(x);q['Jz_sum']=q['LEFT_z']+q['material_z']+q['field_z'];return q
        paths={}
        for tag,operator,state in [('full_kicked',h,kicked),('full_no_kick',h,seed),('same_state_full_transverse_off',off,kicked),('full_transverse_off_no_kick',off,seed)]:
            history=[]
            for t in p['times']:
                if t==0:x=state;ledger={'norm':norm(state)}
                else:x,ledger=krylov(operator,state,t,p['evolution_steps'])
                history.append(dict(time=t,observables=measure(x),propagation=ledger));npz[f'{name}_{tag}_t{t}']=x
            paths[tag]=history
        x=kicked;eh=h@x;response=[]
        for i in range(6):
            # Equal-time current/contact and dynamical response are distinct.
            jx=currents[i]@x;response.append(dict(axis=i,current=real(np.vdot(x,jx)),current_variance=real(np.vdot(jx,jx))-expect(currents[i],x)**2,contact=expect(contacts[i],x),dcurrent_dt=real(1j*(np.vdot(eh,jx)-np.vdot(jx,eh)))))
        q=obs['Q2'];pi=obs['Pi2'];ccr=norm(q@(pi@x)-pi@(q@x)-1j*x)
        field_velocity=1j*(np.vdot(eh,q@x)-np.vdot(q@x,eh));force=1j*(np.vdot(eh,pi@x)-np.vdot(pi@x,eh))
        events[name]=dict(dimension=h.shape[0],M=.5,retained_J='ALL compatible J in full M=1/2 fibre, no response-sector deletion',factorization=ha,kick=kledger,t0_before=baseline,t0_after=initial,t0_matter_difference={key:initial[key]-baseline[key] for key in ['rho0','LEFT_z','material_z']},trajectories=paths,response=response,CCR_state_defect=ccr,Qdot_minus_Pi=real(field_velocity)-expect(pi,x),Pidot_minus_canonical_current=real(force)-expect(currents[2],x))
        result['events']=events;np.savez_compressed(out.with_suffix('.npz'),**npz);out.write_text(json.dumps(result,indent=2)+'\n')
        del h,off,obs,currents,contacts
    result['elapsed_seconds']=time.time()-start
    result['interpretation']='One new finite law, conditional J0 reference. Field shell is electric kinetic only: its bare finite gaps are artifacts, not induced photons. Quantization/gauging/CAR/native cut are chosen inputs. No Nature/global ground/proton/QCD/Lorentz claim.'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'elapsed':result['elapsed_seconds'],'C_spectral_max':result['C_spectral_max'],'events':list(events)}),flush=True)
if __name__=='__main__':main()
