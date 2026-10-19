"""Actual multitrajectory local-rich spatial response, evolving baseline+tangent.
No profile-truncated field, global primitive-pair cap or control-ground refit.
"""
from pathlib import Path
import json,sys,time,resource
import numpy as np
from scipy.integrate import solve_ivp
from rich_connected_longwave_20261017_phase import Phase


class Layout:
    def __init__(self,m):self.m=m;self.sizes=[len(ids)**2 for ids in m.groups]+[len(f['bare'])**2 for f in m.fields]+[81,81];self.ends=np.r_[0,np.cumsum(self.sizes)];self.size=int(self.ends[-1])
    def pack(self,state):
        r,fields,a,p=state;return np.concatenate([r[np.ix_(ids,ids)].ravel()for ids in self.m.groups]+[x.ravel()for x in fields]+[a,p]).astype(complex)
    def unpack(self,y):
        chunks=[y[a:b]for a,b in zip(self.ends[:-1],self.ends[1:])];r=np.zeros((256,256),complex)
        for ids,x in zip(self.m.groups,chunks):r[np.ix_(ids,ids)]=x.reshape(len(ids),len(ids))
        fields=[x.reshape(len(f['bare']),len(f['bare']))for x,f in zip(chunks[4:16],self.m.fields)];return r,fields,chunks[-2].real,chunks[-1].real


class Observers:
    def __init__(self,m):
        self.m=m;self.d=np.array([l['displacement']for l in m.g['links']]);p=np.asarray(m.g['positions']);mid=np.array([p[l['start']]+np.array(l['displacement'])/2 for l in m.g['links']]);self.cell=(np.floor(mid[:,0]/np.sqrt(2)+.5).astype(int)%4);self.raw=np.asarray(m.g['probe_profiles_raw']);self.uniform=np.asarray(m.g['uniform_profiles']);self.inverses=[np.linalg.pinv(self.d[self.cell==c].T@self.d[self.cell==c])for c in range(4)]
    def tangent(self,state,tangent,jet):
        m=self.m;r,fields,a,p=state;dr,df,da,dp=tangent;j=jet['current'];e=jet['Eint'];et=m.P@e;el=e-et;ebase=m.T@m.density(r)+m.K@p;cellj=[];celle=[];cellt=[];celll=[];energy=[]
        for c in range(4):
            select=self.cell==c;d=self.d[select];inv=self.inverses[c];cellj.append(d.T@j[select]);celle.append(inv@d.T@e[select]/3);cellt.append(inv@d.T@et[select]/3);celll.append(inv@d.T@el[select]/3);energy.append(ebase[select]@e[select])
        return {'Qsite':jet['Nint']/3,'Qcell':jet['Nint'].reshape(4,4).sum(axis=1)/3,'current_edge':j,'Efull_edge':e/3,'Etransverse_edge':et/3,'Elong_edge':el/3,'cell_current_vector':np.array(cellj),'cell_Efull_vector':np.array(celle),'cell_Etransverse_vector':np.array(cellt),'cell_Elong_vector':np.array(celll),'cell_electric_energy':np.array(energy),'q_current':self.raw.T@j,'q_Efull':self.raw.T@e/3,'q_ET':self.raw.T@et/3,'uniform_E':self.uniform.T@e/3,'Phi':np.array([np.trace(df[3*c]@np.diag(m.phigrid)).real for c in range(4)]),'R_trace':np.array([sum(np.trace(df[3*c+1]@m.rotor['coordinate'][i][i].toarray()).real for i in range(3))for c in range(4)]),'colour_ReTrV':np.array([np.trace(df[3*c+2]@m.loop['B']).real for c in range(4)]),'energy':np.array(jet['energy']),'Gauss_max':np.array(np.max(abs(m.d@e-jet['Nint']))/3)}


def load_reference(path):
    with np.load(path)as x:return {'rho':x['rho'],'U':x['U'],'V':x['V'],'psis':[x[f'psi_{i}']for i in range(12)]}


def moments(base,response,strength):
    mean=strength*response.mean(axis=0);sem=strength*response.std(axis=0,ddof=1)/np.sqrt(len(response));b=base-base.mean(axis=0);r=response-response.mean(axis=0);contributions=2*strength*b*r*len(base)/(len(base)-1)
    return {'source_mean':mean.tolist(),'MonteCarlo_standard_error':sem.tolist(),'baseline_symmetrized_variance':base.var(axis=0,ddof=1).tolist(),'source_variance_change_LINEAR':contributions.mean(axis=0).tolist(),'variance_change_MC_standard_error':(contributions.std(axis=0,ddof=1)/np.sqrt(len(base))).tolist()}


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text());out=cfg['out'];started=time.time();m=Phase();layout=Layout(m);observers=Observers(m);reference=load_reference(cfg['reference']);rng=np.random.default_rng(cfg['seed']);times=np.asarray(cfg['times']);strength=cfg['source_strength'];saved={};receipts=[];pos=np.asarray(m.g['positions']);profile=np.zeros(256);first=m.site//4==0;profile[first]=pos[m.site[first],0]*m.q[first]/3
    for sample in range(cfg['samples']):
        state=m.sample(reference,rng);r,fields,a,p=state;tangent=(1j*(profile[:,None]-profile[None,:])*r,[np.zeros_like(x)for x in fields],np.zeros(81),np.zeros(81));initial=np.r_[layout.pack(state),layout.pack(tangent)]
        if sample==0:np.savez_compressed(out+'_native_initial_sample0.npz',state_and_source_tangent=initial,source_profile=profile)
        for control,name in((False,'full'),(True,'active_cotree_Uidentity')):
            calls=0;max_ward=0.
            def rhs(t,y):
                nonlocal calls,max_ward
                s=layout.unpack(y[:layout.size]);ds=layout.unpack(y[layout.size:]);b,d,j=m.jet(s,ds,control=control);calls+=1
                if not control:max_ward=max(max_ward,j['native_full_H_field_current_Ward'])
                return np.r_[layout.pack(b),layout.pack(d)]
            begin=time.time();solution=solve_ivp(rhs,(0,float(times[-1])),initial,t_eval=times,rtol=cfg['rtol'],atol=cfg['atol'],max_step=cfg['max_step'])
            if not solution.success:raise RuntimeError(solution.message)
            base=[];response=[];casimir=[];traces=[];hermiticity=[]
            for column in solution.y.T:
                s=layout.unpack(column[:layout.size]);ds=layout.unpack(column[layout.size:]);_,_,j=m.jet(s,ds,control=control);base.append(m.probes(s,control=control));response.append(observers.tangent(s,ds,j));rr,ff,_,_=s;casimir.append([np.trace(rr[np.ix_(ids,ids)]@rr[np.ix_(ids,ids)]).real for ids in m.groups]+[np.trace(x@x).real for x in ff]);traces.append([np.trace(rr[np.ix_(ids,ids)]).real for ids in m.groups]+[np.trace(x).real for x in ff]);hermiticity.append(max(np.linalg.norm(rr-rr.conj().T),max(np.linalg.norm(x-x.conj().T)for x in ff)))
            for kind,rows in(('base',base),('response_per_unit_source',response)):
                for key in rows[0]:saved.setdefault(name+'_'+kind+'_'+key,[]).append(np.array([row[key]for row in rows]))
            if sample==0:np.savez_compressed(out+'_'+name+'_sample0_states.npz',times=times,state_and_source_tangent=solution.y)
            energy=np.array([b['energy']for b in base]);de=np.array([b['energy']for b in response]);receipt={'sample':sample,'generator':name,'nfev':solution.nfev,'seconds':time.time()-begin,'energy_drift_max':float(np.max(abs(energy-energy[0]))),'tangent_energy_drift_max':float(np.max(abs(de-de[0]))),'fermion_and_local_field_Casimir_drift':float(np.max(abs(np.array(casimir)-casimir[0]))),'number_and_local_field_trace_drift':float(np.max(abs(np.array(traces)-traces[0]))),'Hermiticity_max':float(max(hermiticity)),'full_generator_field_current_Ward_max':max_ward if not control else None};receipts.append(receipt);print(json.dumps({'stage':'ACTUAL_rich_spatial_source_response','sample':sample,'samples':cfg['samples'],'generator':name,'nfev':solution.nfev,'seconds':receipt['seconds'],'energy_drift':receipt['energy_drift_max']}),flush=True)
        raw={k:np.asarray(v)for k,v in saved.items()};np.savez_compressed(out+'_checkpoint_observers.npz',times=times,**raw);Path(out+'_checkpoint_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    raw={k:np.asarray(v)for k,v in saved.items()};np.savez_compressed(out+'_sample_observers.npz',times=times,**raw);summary={}
    for name in('full','active_cotree_Uidentity'):
        summary[name]={key:moments(raw[name+'_base_'+key],raw[name+'_response_per_unit_source_'+key],strength)for key in('cell_current_vector','cell_Efull_vector','cell_Etransverse_vector','cell_Elong_vector','cell_electric_energy','Qcell','q_current','q_Efull','q_ET','uniform_E','Phi','R_trace','colour_ReTrV')}
    contrast={}
    for key in('cell_current_vector','cell_Etransverse_vector','q_ET','q_current'):
        v=raw['full_response_per_unit_source_'+key]-raw['active_cotree_Uidentity_response_per_unit_source_'+key];contrast[key]={'source_difference_full_minus_active_control':(strength*v.mean(axis=0)).tolist(),'paired_MonteCarlo_standard_error':(strength*v.std(axis=0,ddof=1)/np.sqrt(len(v))).tolist()}
    covariance={}
    for name in('full','active_cotree_Uidentity'):
        b=raw[name+'_base_q_ET'];r=raw[name+'_response_per_unit_source_q_ET'];covariance[name]=[]
        for t in range(len(times)):
            bb=b[:,t]-b[:,t].mean(axis=0);rr=r[:,t]-r[:,t].mean(axis=0);covariance[name].append({'time':float(times[t]),'baseline_q_ET_covariance':(bb.T@bb/(len(bb)-1)).tolist(),'source_q_ET_covariance_change_LINEAR':(strength*(bb.T@rr+rr.T@bb)/(len(bb)-1)).tolist()})
    result={'kind':'ACTUAL_CONNECTED_LOCAL_RICH_LIE_POISSON_TWA_WAVE_RESPONSE','samples':cfg['samples'],'seed':cfg['seed'],'times':times.tolist(),'source_strength':strength,'source':'Nonconstant UNIVERSAL cell0 physical dipole impulse; source derivative along each actual nonlinear trajectory, not static Hessian/refitted control. O(f²) omitted.','reference':cfg['reference'],'same_sampled_physical_input_for_full_and_control':True,'local_Phi_R_PW_quantum_covariances_and_ALL81canonical_cycles':True,'no_global_primitive_pair_cap_or_profile_tying':True,'initial_current_region_support':'Dipole pulse changes currents only on bonds touchingcell0; midpointcell1/3 may include adjacent physical bonds. Cell2 is opposite; no receiver moved. InitialfullE/density source change zero.','cell_readout':'Midpoint-region current sum d_e J_e; E vector least-squares native link-flux reconstruction using actual local gram. Not continuum photon/Poynting observable.','responses':summary,'paired_active_control_contrasts':contrast,'q_ET_covariance':covariance,'numerical_receipts':receipts,'approximation_limits':'TWA/finite-field symmetrized sampling is an approximate quantum response, not full nonlinear compact evolution. Continuous momentum trajectories are not integer spectra. Quasimatices can be nonpositive, never clipped or called probabilities. Joint rho*R*V currents retained, native product-reference current image compared separately. Residual compact-offset/Coulomb/colour/material control response is NOT all transverse radiation. Finite tube/time/MonteCarlo signal not photon poles, universal velocity, mass, lifetime or full3DNoether proof.','resources':{'seconds':time.time()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'ACTUAL_rich_spatial_response_complete','samples':cfg['samples'],'seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
