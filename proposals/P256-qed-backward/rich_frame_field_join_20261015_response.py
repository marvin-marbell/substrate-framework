"""Natural same-new-H CAR/C/full angular response and finite gauge defects."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse,linalg
from rich_frame_field_join_20261015_model import build,law,fibre,observable,compress,source,physical_phase
from rich_frame_field_join_20261015_calculate import norm,expect,m_isometry,real,krylov
from rich_common_frame_orbit_20261015_model import FormOperator
ROOT=Path(__file__).resolve().parent


def correlated_feedback(model,name,states,times):
    vm=m_isometry(model,name,.5);s=model['sectors'][name];op=s['op'];operators={}
    for key in ['B','electric','PW_boundary']:
        operators['colour_'+key]=compress(observable(model,name,matter=op[key]),vm)[0]
    operators['fieldLEFT_Casimir']=compress(observable(model,name,matter=sum(a@a for a in op['S'])),vm)[0]
    for key in ['r','r2','frame','radial']:
        radial={float(j):model['radial'].operators(float(j))[key] for j in set(s['spin']['j'])}
        operators['matrix_amplitude_'+key]=compress(FormOperator(s,s,model['rotor'],model['radial'],[],radial),vm)[0]
    operators['colourB_times_matrix_r2']=compress(FormOperator(s,s,model['rotor'],model['radial'],[(op['B'],np.eye(280),2)]),vm)[0]
    operators['colourB_times_LEFT_Casimir']=compress(observable(model,name,matter=op['B']@sum(a@a for a in op['S'])),vm)[0]
    output=[]
    for t in times:
        values={}
        for tag in ['full_kicked','full_no_kick','same_state_full_transverse_off','full_transverse_off_no_kick']:
            x=states[f'{name}_{tag}_t{t}'];values[tag]={key:expect(a,x) for key,a in operators.items()}
            row=values[tag];row['connected_colourB_matrix_r2']=row['colourB_times_matrix_r2']-row['colour_B']*row['matrix_amplitude_r2'];row['connected_colourB_LEFT_Casimir']=row['colourB_times_LEFT_Casimir']-row['colour_B']*row['fieldLEFT_Casimir']
        delta={key:values['full_kicked'][key]-values['full_no_kick'][key] for key in values['full_kicked']}
        offdelta={key:values['same_state_full_transverse_off'][key]-values['full_transverse_off_no_kick'][key] for key in values['full_kicked']}
        output.append({'time':t,'values':values,'EM_kick_intervention':delta,'proper_full_transverse_off_intervention':offdelta})
    return {'meaning':'Actual external-field-only kick followed by SAME new H: physical LEFT-spin sector, quantum matrix-amplitude and colour-loop/electric feedback. r is matrix amplitude, not charge radius. No scalar-only-drive novelty claim.','trajectories':output}


def basis_covariance(model):
    """Bounded actual-join sibling check, never patching frozen real-R10 API."""
    from rich_spatial_colour_root_20261015_model import so3
    from rich_spatial_colour_em_20261015_fields import phase_kernel,mechanical_kernels
    rot=model['physical_rotor'];realrot=so3(model['parameters']);u=np.asarray(model['material']['U']);bridge=np.kron(u,np.eye(28));g=model['geometry'];f=model['field'];c=model['coefficients']
    def difference(a,b):return norm(a-bridge.conj().T@b@bridge)
    rows=[]
    for edge,q in [(0,1.),(1,1/3)]:
        for derivative in [(),(2,),(2,5)]:
            for axis in [None,0,1,2]:
                actual=physical_phase(rot,g,c,f,edge,q,derivative,axis);control=phase_kernel(realrot,g,c,f,edge,q,derivative,axis)
                rows.append({'edge':edge,'charge':q,'derivative':list(derivative),'spin_axis':axis,'absolute_error':difference(actual,control)})
    mechanical=[]
    for derivative in [(),(2,),(2,5)]:
        a=mechanical_kernels(rot,f,model['connections'],model['parameters']['orientation_inertia'],derivative=derivative);b=mechanical_kernels(realrot,f,model['connections'],model['parameters']['orientation_inertia'],derivative=derivative)
        mechanical.append({'derivative':list(derivative),'linear_errors':[difference(x,y) for x,y in zip(a['linear'],b['linear'])],'quadratic_errors':[difference(x,y) for x,y in zip(a['quadratic'],b['quadratic'])]})
    values=rot['R'][:,2,0]
    density_error=norm(rot['integrate'](values).toarray()-u.conj().T@realrot['integrate'](values).toarray()@u)
    return {'bridge_unitarity':norm(bridge.conj().T@bridge-np.eye(280)),'metric_Gram':norm(rot['basis'].conj().T@(rot['weights'][:,None]*rot['basis'])-np.eye(10)),'wrong_unconjugated_Gram_control':norm(rot['basis'].T@(rot['weights'][:,None]*rot['basis'])-np.eye(10)),'R_spin_phase_current_contact':rows,'charged_mechanical_current_contact':mechanical,'lab_density_R_multiplier_covariance':density_error,'BODY_and_source_metric_rule':'Native BODY, density, spin and raw source metrics use U.conj().T A U; Wigner-source/Gauss uses P.conj().T; radial metric is real and independent. No other real-basis-as-adjoint shortcut occurs on the actual join paths.'}


def spectral(h,x,frequencies,eta,steps,offset):
    weight=norm(x)**2
    if weight<1e-25:return dict(weight=weight,frequencies=frequencies,G=[[0.,0.] for _ in frequencies],dimension=0)
    q=x/np.sqrt(weight);basis=[];alpha=[];beta=[];old=np.zeros_like(q);last=0.
    for iteration in range(steps):
        basis.append(q);w=h@q-(beta[-1]*old if beta else 0);a=real(np.vdot(q,w));alpha.append(a);w-=a*q
        for _ in range(2):
            for b in basis:w-=np.vdot(b,w)*b
        last=norm(w)
        if last<1e-12:break
        old=q;q=w/last
        if iteration+1<steps:beta.append(last)
    e,u=linalg.eigh_tridiagonal(np.array(alpha),np.array(beta));weights=weight*abs(u[0])**2;g=[sum(weights/(omega+1j*eta-(e-offset))) for omega in frequencies]
    residuals=[float(abs(last*np.linalg.solve((omega+1j*eta+offset)*np.eye(len(alpha))-(np.diag(alpha)+np.diag(beta,1)+np.diag(beta,-1)),np.eye(len(alpha))[:,0])[-1])*np.sqrt(weight)) for omega in frequencies]
    return dict(weight=weight,dimension=len(alpha),projected_energies=e.tolist(),projected_weights=weights.tolist(),frequencies=frequencies,eta=eta,G=[[real(z),float(z.imag)] for z in g],linear_system_absolute_residual=residuals,last_lanczos_beta=last)


def main():
    p=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);statepath=ROOT/p['state_file'];digest=hashlib.sha256(statepath.read_bytes()).hexdigest()
    if digest!=p['state_sha256']:raise ValueError('Exact owning state changed')
    states=np.load(statepath);meta=json.loads((ROOT/p['result_file']).read_text());model=build(p);result={'state_file':statepath.name,'state_sha256':digest,'parameters':p,'CAR':[],'source_G':{},'response':{},'gauge':{}}
    if hashlib.sha256((ROOT/p['result_file']).read_bytes()).hexdigest()!=p['result_sha256']:raise ValueError('Exact owning metadata changed')
    field=model['field'];result['finite_field']={'bare_electric_kinetic_eigenvalues':np.linalg.eigvalsh(field['H_E']).tolist(),'CCR_operator_norms':[float(np.linalg.norm(field['Q'][i]@field['Pi'][i]-field['Pi'][i]@field['Q'][i]-1j*np.eye(28),2)) for i in range(6)],'precut_QQ_minus_cut_Q_squared_norms':[float(np.linalg.norm(field['QQ'][i][i]-field['Q'][i]@field['Q'][i],2)) for i in range(6)],'interpretation':'Bare H_E has artificial finite-shell discrete gaps; no magnetic stiffness or induced-photon identification.'}
    result['complex_basis_covariance']=basis_covariance(model)
    v0=fibre(model,'neutral',0.,0.);z=states['reference'];fullz=v0@z;E0=meta['reference']['energies'][0]
    for site in [0,1]:
        for m in [-.5,.5]:
            entry={'site':site,'M':m};total=0.
            for name in ['lepton','antilepton']:
                gram=next(mat for label,mat in model['native_source_metric_full'][name].items() if label.split(':')[0]==str(site) and float(label.split(':')[1])==.5 and float(label.split(':')[2])==m)
                u=model['sectors']['neutral']['spin']['U'];g=u.conj().T@gram@u;raw=expect(observable(model,'neutral',matter=g),fullz);sm,sa=source(model,name,.5,m,site);projected=norm(sm@fullz)**2
                entry[name]=dict(raw_norm_squared=raw,retained_norm_squared=projected,projection_loss=raw-projected);total+=raw
            entry['raw_CAR_sum']=total;entry['raw_CAR_defect']=abs(total-1.);result['CAR'].append(entry)
    # Native C is unitary particle-hole intertwining, NOT complex conjugation.
    sn=model['sectors']['lepton']['spin'];sa=model['sectors']['antilepton']['spin'];cm=sa['U'].conj().T@model['C_matter']['lepton']@sn['U'];crq=model['rq'](sparse.kron(sparse.eye(10),model['field']['C']));C=FormOperator(model['sectors']['lepton'],model['sectors']['antilepton'],model['rotor'],model['radial'],[(cm,crq,0)])
    vl=fibre(model,'lepton',.5,.5);x=vl@states['lepton_vectors'][:,0];cx=C@x;hl=law(model,'lepton');ha=law(model,'antilepton')
    result['C_intertwining']={'state_norm':norm(cx),'H_C_minus_C_H':norm(ha@cx-C@(hl@x)),'charge_C_plus_C_charge':norm(observable(model,'antilepton',matter=model['sectors']['antilepton']['op']['Q'])@cx+C@(observable(model,'lepton',matter=model['sectors']['lepton']['op']['Q'])@x))}
    out.write_text(json.dumps(result,indent=2)+'\n')
    del hl,ha,C,x,cx
    for name in ['lepton','antilepton']:
        v=fibre(model,name,.5,.5);h,ledger=compress(law(model,name),v);seed=states[name+'_source_site0_m1'];result['source_G'][name]=spectral(h,seed,p['frequencies'],p['eta'],p['response_steps'],E0)
        ground=states[name+'_vectors'][:,0];full=v@ground;energy=meta['stationary'][name]['energies'][0];op=model['sectors'][name]['op'];targets={.5:h};vs={}
        for J in [.5,1.5]:
            for M in np.arange(-J,J+1):vs[J,float(M)]=fibre(model,name,J,float(M))
            if J!=.5:targets[J]=compress(law(model,name),vs[J,J])[0]
        angular=[]
        for i in range(6):
            current=law(model,name,derivative=(i,))
            image=current@full;projection=np.zeros_like(image);pieces=[]
            for (J,M),vt in vs.items():
                y=vt.conj().T@image;w=norm(y)**2;projection+=vt@y
                pieces.append({'J':J,'M':M,'weight':w})
            row={'component':i,'full_weight':norm(image)**2,'angular_projection_leakage':norm(image-projection),'sectors':pieces}
            # z components have only M=+1/2; both output Js retained exactly.
            row['contact']=expect(law(model,name,derivative=(i,i)),full)
            if i in [2,5]:
                row['G_by_J']={str(J):spectral(targets[J],vs[J,.5].conj().T@image,p['frequencies'],p['eta'],p['response_steps'],energy) for J in [.5,1.5]}
                combined=np.sum([np.array(x['G'])[:,0]+1j*np.array(x['G'])[:,1] for x in row['G_by_J'].values()],axis=0)
                row['contact_plus_retarded_current_kernel']=[]
                for index,omega in enumerate(p['frequencies']):
                    opposite=p['frequencies'].index(-omega);value=row['contact']+combined[index]+combined[opposite].conjugate()
                    row['contact_plus_retarded_current_kernel'].append({'omega':omega,'real':real(value),'imaginary':float(value.imag)})
                row['kernel_meaning']='Same already-coupled finite H, contact plus retarded current susceptibility, NOT an unprojected Ward identity or separately induced free photon stiffness.'
            angular.append(row)
        result['response'][name]=angular
        hf=law(model,name);rho=observable(model,name,matter=op['rho0']);flow=sum((law(model,name,route_derivative=e)@full for e in range(2)),np.zeros_like(full));drho=1j*(hf@(rho@full)-rho@(hf@full))
        # Constituent-path continuity is measured independently, not forced by defining J=i[H,X].
        result['gauge'][name]={'rho0dot_plus_path_current_state_defect':norm(drho+flow),'rho0dot_minus_path_current_wrong_sign_control':norm(drho-flow)}
        # Full physical-space CCR and electric force; no same-M projection of transverse channels.
        ccr=[];force=[]
        for i in range(6):
            q=observable(model,name,rq=model['rqobs']['Q'][i]);pi=observable(model,name,rq=model['rqobs']['Pi'][i]);x=full;hx=hf@x;qx=q@x;px=pi@x
            ccr.append(norm(q@px-pi@qx-1j*x));force.append(norm(1j*(hf@px-pi@hx)-law(model,name,derivative=(i,))@x))
        result['gauge'][name].update(CCR_state_defects=ccr,canonical_Pidot_minus_current_action_defects=force)
        vm=m_isometry(model,name,.5);control=[]
        matter_observables={label:compress(observable(model,name,matter=matrix),vm)[0] for label,matrix in [('rho0',op['rho0']),('LEFT_z',op['S'][2])]}
        for t in p['times']:
            x=states[f'{name}_full_transverse_off_no_kick_t{t}']
            kicked=states[f'{name}_same_state_full_transverse_off_t{t}']
            control.append({'time':t,'off_no_kick':{key:expect(a,x) for key,a in matter_observables.items()},'off_kick_minus_off_no_kick':{key:expect(a,kicked)-expect(a,x) for key,a in matter_observables.items()}})
        result['gauge'][name]['proper_off_intervention_control']=control
        out.write_text(json.dumps(result,indent=2)+'\n')
        result.setdefault('correlated_EM_LEFT_radial_colour_feedback',{})[name]=correlated_feedback(model,name,states,p['times'])
        out.write_text(json.dumps(result,indent=2)+'\n')
        del h,targets,current,hf,matter_observables
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'CAR_max':max(x['raw_CAR_defect'] for x in result['CAR']),'C':result['C_intertwining']}),flush=True)
if __name__=='__main__':main()
