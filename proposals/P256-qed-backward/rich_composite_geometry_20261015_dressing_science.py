"""Physical charged weak-H/G dynamics of the NEW rich dressing model.
NOT a Hermitized source-EOM K. Full independentPhi7 and actual post-MGS metric;
original microscopic source current/contact, colourOFF, ALLhopOFF, fieldkick.
"""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve,eigh,expm
from scipy.sparse.linalg import LinearOperator,eigsh,expm_multiply
from rich_composite_geometry_20261015_factors import PhiHamiltonian
from rich_composite_geometry_20261015_cloud_variational import arrays
import rich_composite_geometry_20261015_model as model


class WeakPhi(LinearOperator):
    def __init__(self,c0,cphi,gram,hphi,grid):
        self.c0=sparse.csr_matrix(c0);self.cphi=sparse.csr_matrix(cphi);self.gram=sparse.csr_matrix(gram)
        self.hphi=np.asarray(hphi,complex);self.grid=np.asarray(grid,float);self.n=self.c0.shape[0];self.nphi=len(self.grid)
        self.adj0=self.c0.conj().T;self.adjphi=self.cphi.conj().T;self.gram_adj=self.gram.conj().T
        super().__init__(dtype=np.dtype(complex),shape=(self.n*self.nphi,)*2)
    def action(self,value,adjoint=False):
        vector=np.asarray(value);columns=1 if vector.ndim==1 else vector.shape[1];tensor=vector.reshape(self.n,self.nphi,columns)
        c0=self.adj0 if adjoint else self.c0;cphi=self.adjphi if adjoint else self.cphi;g=self.gram_adj if adjoint else self.gram
        result=(c0@tensor.reshape(self.n,-1)).reshape(tensor.shape);result+=(cphi@(tensor*self.grid[None,:,None]).reshape(self.n,-1)).reshape(tensor.shape)
        field=np.einsum('ab,nbk->nak',self.hphi.conj().T if adjoint else self.hphi,tensor);result+=(g@field.reshape(self.n,-1)).reshape(tensor.shape)
        return result.reshape(vector.shape)
    def _matvec(self,value):return self.action(value)
    def _matmat(self,value):return self.action(value)
    def _rmatvec(self,value):return self.action(value,True)
    def _rmatmat(self,value):return self.action(value,True)


def metric_inverse(gram,labels):
    groups=defaultdict(list)
    for column,label in enumerate(labels):groups[tuple(label['counts'])].append(column)
    rows=[];columns=[];values=[];minimum=1.;maximum_error=0.
    for keep in groups.values():
        block=gram[keep][:,keep].toarray();factor=cho_factor(block,lower=True);inverse=cho_solve(factor,np.eye(len(keep)));minimum=min(minimum,float(np.linalg.eigvalsh(block)[0]));maximum_error=max(maximum_error,float(np.linalg.norm(block@inverse-np.eye(len(keep)))))
        r,c=np.indices(block.shape);rows.extend(np.asarray(keep)[r.ravel()]);columns.extend(np.asarray(keep)[c.ravel()]);values.extend(inverse.ravel())
    inverse=sparse.coo_matrix((values,(rows,columns)),shape=gram.shape).tocsr()
    return inverse,{'minimum_count_metric_eigenvalue':minimum,'maximum_count_inverse_residual':maximum_error,'count_blocks':len(groups)}


class TensorMetric(LinearOperator):
    def __init__(self,gram,nphi):
        self.gram=gram;self.adjoint_gram=gram.conj().T;self.n=gram.shape[0]
        super().__init__(dtype=np.dtype(complex),shape=(self.n*nphi,)*2)
    def action(self,value,adjoint=False):
        array=np.asarray(value);operator=self.adjoint_gram if adjoint else self.gram
        return (operator@array.reshape(self.n,-1)).reshape(array.shape)
    def _matvec(self,value):return self.action(value)
    def _matmat(self,value):return self.action(value)
    def _rmatvec(self,value):return self.action(value,True)
    def _rmatmat(self,value):return self.action(value,True)


def metric_operator(gram,nphi):return TensorMetric(gram,nphi)


def spectrum(c0,cphi,gram,inverse,scalar,k=8):
    h=WeakPhi(c0,cphi,gram,scalar['H'],scalar['grid']);metric=metric_operator(gram,len(scalar['grid']));inv=metric_operator(inverse,len(scalar['grid']));count=min(k,h.shape[0]-2)
    rng=np.random.default_rng(422);v0=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0])
    energy,vectors=eigsh(h,k=count,M=metric,Minv=inv,which='SA',tol=2e-10,maxiter=5000,v0=v0);order=np.argsort(energy);energy=energy[order];vectors=vectors[:,order]
    residual=[float(np.linalg.norm(h@vectors[:,i]-energy[i]*(metric@vectors[:,i]))) for i in range(count)]
    return energy,vectors,residual


def evolution(c0,cphi,inverse,scalar):
    return PhiHamiltonian(inverse@c0,inverse@cphi,scalar['H'],scalar['grid'])


def evolve(operator,state,times):
    current=state.copy();previous=0.;result=[];trace=np.sum(operator.diagonal())
    for t in times:
        dt=t-previous
        if dt:current=expm_multiply((-1j*dt)*operator,current,traceA=-1j*dt*trace)
        result.append(current.copy());previous=t
    return result


def body_expectation(state,matrix,nphi):
    tensor=state.reshape(matrix.shape[0],nphi);return float(np.vdot(tensor,matrix@tensor).real)


def readout(state,gram,data,operators,scalar):
    nf=len(scalar['grid']);tensor=state.reshape(gram.shape[0],nf);dual=gram@tensor;weights=np.real(np.sum(tensor.conj()*dual,axis=1));phi=np.real(np.sum(tensor.conj()*dual,axis=0))
    return {'norm':float(np.sum(weights)),'compact':float(weights@data['C']),'fragmented':float(weights@data['F']),'compact_site':(weights@data['Csite']).tolist(),
        'Q_em_site_mean':(weights@data['Q']).tolist(),'baryon_site_mean':(weights@data['B']).tolist(),'ncolour_site_mean':(weights@data['colour']).tolist(),'nwhite_site_mean':(weights@data['white']).tolist(),
        'nwhite_site_variance':(weights@(data['white']**2)-(weights@data['white'])**2).tolist(),'Phi':float(phi@scalar['grid']),'Phi2':float(phi@(scalar['grid']**2)),
        'Phi_boundary':float(phi[0]+phi[-1]),'Phi_energy':float(np.vdot(tensor,gram@(tensor@scalar['H'].T)).real),
        **{name:body_expectation(state,operators[name],nf) for name in ('loopB','loopC','electric','rotation')}}


def count_probabilities(state,gram,data,nphi):
    tensor=state.reshape(gram.shape[0],nphi);weights=np.real(np.sum(tensor.conj()*(gram@tensor),axis=1));result=defaultdict(float)
    for counts,weight in zip(data['counts'],weights):result[tuple(counts.reshape(-1))]+=float(weight)
    return result


def connection(matrix,data,edge,order,theta=0.,charge='EM'):
    coo=matrix.tocoo();start,end=model.EDGES[edge];number=data['counts'].sum(axis=2);left=number[coo.row,start]-number[coo.col,start];right=number[coo.row,end]-number[coo.col,end];mask=(abs(left)==1)&(left+right==0)
    for site in range(4):
        if site not in (start,end):mask&=number[coo.row,site]==number[coo.col,site]
    density=data['Q'] if charge=='EM' else data['B'];q=density[coo.row,end]-density[coo.col,end];factor=np.exp(1j*theta*q)-1 if order==0 else -1j*q if order==1 else -q*q
    return sparse.coo_matrix((coo.data[mask]*factor[mask],(coo.row[mask],coo.col[mask])),shape=matrix.shape).tocsr()


def tensor_frobenius(first,second,grid):
    value=len(grid)*np.linalg.norm(first.data)**2+float(grid@grid)*np.linalg.norm(second.data)**2+2*float(np.sum(grid))*np.real(first.conjugate().multiply(second).sum())
    return float(np.sqrt(value))


def stationary_channels(c0,cphi,gram,inverse,data,scalar):
    result={}
    for name in ('C','F'):
        keep=np.flatnonzero(data[name]);result[name]={'body_dimension':len(keep),'full_Phi_dimension':len(keep)*len(scalar['grid'])}
        if len(keep):
            energy,_,residual=spectrum(c0[keep][:,keep],cphi[keep][:,keep],gram[keep][:,keep],inverse[keep][:,keep],scalar,1);result[name].update(minimum_energy=float(energy[0]),generalized_residual=residual[0])
    if result['C']['body_dimension'] and result['F']['body_dimension']:result['F_minus_C']=result['F']['minimum_energy']-result['C']['minimum_energy']
    result['meaning']='NONNESTED whole-sea occupation-channel discriminator, NOT a free/asymptotic breakup threshold or tagged-three-particle position PVM.'
    return result


def nohop_evolve(c0,gram,labels,scalar,state,times):
    nf=len(scalar['grid']);tensor=state.reshape(len(labels),nf);groups=defaultdict(list)
    for index,label in enumerate(labels):groups[tuple(label['counts'])].append(index)
    blocks=[]
    for keep in groups.values():
        energy,vectors=eigh(c0[keep][:,keep].toarray(),gram[keep][:,keep].toarray());amplitudes=vectors.conj().T@(gram[keep][:,keep]@tensor[keep]);blocks.append((keep,energy,vectors,amplitudes))
    result=[]
    for t in times:
        value=np.zeros_like(tensor);field=expm(-1j*t*scalar['H'])
        for keep,energy,vectors,amplitudes in blocks:value[keep]=(vectors@(np.exp(-1j*t*energy)[:,None]*amplitudes))@field.T
        result.append(value.reshape(-1))
    return result


def analyze(operators,labels,probes,scalar,p,reference_energy,config,out):
    all_data=arrays(labels);result={'reference_selected_energy':reference_energy,'sectors':{}}
    for j in (1,3):
        keep=np.flatnonzero(all_data['J']==j);local_labels=[labels[index] for index in keep];data={name:value[keep] for name,value in all_data.items()};ops={name:value[keep][:,keep] for name,value in operators.items()};gram=ops['Gram'];inverse,metric=metric_inverse(gram,local_labels)
        c0=ops['rotation']+ops['electric']+ops['plaquette']+ops['quark_vector']+ops['white_vector'];cphi=ops['quark_scalar']+ops['white_scalar'];hop0=ops['quark_vector']+ops['white_vector'];hopphi=cphi;nf=len(scalar['grid'])
        energy,vectors,residual=spectrum(c0,cphi,gram,inverse,scalar);ground=np.flatnonzero(abs(energy-energy[0])<1e-8);h=WeakPhi(c0,cphi,gram,scalar['H'],scalar['grid']);metric_full=metric_operator(gram,nf);source_results={};compact_source=None
        for probe in probes:
            if probe['descriptor']['twice_J']!=j:continue
            source_vector=probe['coordinates'][keep].reshape(-1);dual=metric_full@source_vector;retained=float(np.vdot(source_vector,dual).real);residue=float(np.linalg.norm(vectors[:,ground].conj().T@dual)**2);key=str(probe['descriptor'])
            source_results[key]={'raw_canonical_strength':probe['raw_strength'],'retained_strength':retained,'omitted_strength_signed':probe['raw_strength']-retained,'raw_observed_ground_cluster_residue':residue,'conditional_ground_fraction':residue/retained if retained else None,'preparation_energy':float(np.vdot(source_vector,h@source_vector).real/retained) if retained else None}
            if probe['descriptor']['family']=='bare' and probe['descriptor']['spec'][0]==[0,0,0]:compact_source=source_vector
        if compact_source is None:raise ValueError('Pinned canonical compact0 source absent; no receiver/state fallback')
        prepared=(compact_source.reshape(len(keep),nf)*data['Csite'][:,0,None]).reshape(-1);support=float(np.vdot(prepared,metric_full@prepared).real)
        if support<=0:raise ValueError('Pinned compact0 has no support; no source fallback')
        prepared/=np.sqrt(support);initial_counts=count_probabilities(prepared,gram,data,nf);full=evolution(c0,cphi,inverse,scalar);off0=c0-ops['electric'];off=evolution(off0,cphi,inverse,scalar);off_energy,off_vectors,off_residual=spectrum(off0,cphi,gram,inverse,scalar)
        trajectories={'full':evolve(full,prepared,p['times']),'colour_electric_off':evolve(off,prepared,p['times']),
            'ALL_hops_off':nohop_evolve(ops['rotation']+ops['electric']+ops['plaquette'],gram,local_labels,scalar,prepared,p['times'])}
        events={}
        for name,states in trajectories.items():
            events[name]=[]
            for t,state in zip(p['times'],states):
                weights=count_probabilities(state,gram,data,nf);tv=sum(abs(weights.get(key,0.)-initial_counts.get(key,0.)) for key in weights.keys()|initial_counts.keys())/2
                events[name].append({'t':t,**readout(state,gram,data,ops,scalar),'material_count_PVM_TV_from_preparation':tv})
        sector={'body_dimension':len(keep),'full_independent_Phi_dimension':len(keep)*nf,'metric':metric,'minimum_energy':float(energy[0]),'gap_from_OWN_selected_reference':float(energy[0]-reference_energy),
            'low_poles':energy.tolist(),'generalized_residuals':residual,'observed_ground_cluster_members':len(ground),'observed_ground_cluster_closed_at_lowpole_boundary':bool(energy[-1]-energy[0]>=1e-8),
            'ground':readout(vectors[:,0],gram,data,ops,scalar),'channels':stationary_channels(c0,cphi,gram,inverse,data,scalar),'sources':source_results,'compact0_conditioning_strength':support,
            'finite_prepared_events':events,'electric_off':{'minimum_energy':float(off_energy[0]),'generalized_residuals':off_residual,'ground':readout(off_vectors[:,0],gram,data,ops,scalar),'channels':stationary_channels(off0,cphi,gram,inverse,data,scalar),'meaning':'Same source/body/reference/Phi space, only electric term removed; not guaranteed nonbinding.'}}
        result['sectors'][str(j)]=sector;Path(out+'_first_physics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'FIRST_RICH_DRESSING_COMPACT_MOTION','twice_J':j,'minimum_energy':sector['minimum_energy'],'events':events}),flush=True)
        ward={};currents={};contacts={}
        for charge in ('EM','baryon'):
            density=data['Q'] if charge=='EM' else data['B'];div0=[sparse.csr_matrix(c0.shape,dtype=complex) for _ in range(4)];divphi=[value.copy() for value in div0];currents[charge]={};contacts[charge]={}
            for edge,(start,end) in enumerate(model.EDGES):
                j0=connection(hop0,data,edge,1,charge=charge);jphi=connection(hopphi,data,edge,1,charge=charge);div0[start]+=j0;div0[end]-=j0;divphi[start]+=jphi;divphi[end]-=jphi
                current=WeakPhi(j0,jphi,sparse.csr_matrix(gram.shape),np.zeros_like(scalar['H']),scalar['grid']);contact=WeakPhi(connection(hop0,data,edge,2,charge=charge),connection(hopphi,data,edge,2,charge=charge),sparse.csr_matrix(gram.shape),np.zeros_like(scalar['H']),scalar['grid'])
                currents[charge][str(edge)]=[float(np.vdot(state,current@state).real) for state in trajectories['full']];contacts[charge][str(edge)]=float(np.vdot(prepared,contact@prepared).real)
            ward[charge]=[]
            for site in range(4):
                q=sparse.diags(density[:,site]);w0=1j*(c0@q-q@c0)+div0[site];wphi=1j*(cphi@q-q@cphi)+divphi[site];ward[charge].append(tensor_frobenius(w0,wphi,scalar['grid']))
        theta=config['material_EM_angle'];duration=config['material_EM_duration'];pulse=evolution(c0+connection(hop0,data,3,0,theta),cphi+connection(hopphi,data,3,0,theta),inverse,scalar);em=evolve(pulse,prepared,[duration])[0];same_age=evolve(full,prepared,[duration])[0]
        field_initial=(prepared.reshape(len(keep),nf)@expm(-1j*p['field_kick']*scalar['P']).T).reshape(-1)
        sector['material_EM_pulse_events']=[{'t':t,**readout(state,gram,data,ops,scalar)} for t,state in zip(p['times'],evolve(full,em,p['times']))]
        sector['same_age_no_EM_events']=[{'t':t,**readout(state,gram,data,ops,scalar)} for t,state in zip(p['times'],evolve(full,same_age,p['times']))]
        sector['field_only_kick_events']=[{'t':t,**readout(state,gram,data,ops,scalar)} for t,state in zip(p['times'],evolve(full,field_initial,p['times']))]
        sector['original_Peierls_current']=currents;sector['original_Peierls_contact']=contacts;sector['Ward_after_ALL_MGS_and_actual_metric']=ward
        sector['source_law']='Original microscopic physicalQ chord phases BEFORE projection, fixed zero-A source/body isometry. Actual count gauge-orbit closure and independentPhi factor preserve source continuity; J/contact direct derivatives, Ward measured separately. No lab3D/mechanicalEM/photon/COM claim.'
        np.savez(out+f'_J{j}_lowstates.npz',energies=energy,vectors=vectors,prepared=prepared)
        Path(out+'_science_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    result['threehalf_minus_half']=result['sectors']['3']['minimum_energy']-result['sectors']['1']['minimum_energy'];return result
