"""Actual stationary quark-occupation/colour discriminator and full-H motion.
No constrained-channel energy is renamed an asymptotic breakup threshold.
"""
from __future__ import annotations
import json,pickle,resource,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh,expm_multiply
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_runtime import primitive_for,conjugate_packet


def load(path):
    with open(path,'rb') as f:return pickle.load(f)


def low(h,count=3,tolerance=1e-10):
    if h.shape[0]<6:
        e,v=np.linalg.eigh(h.toarray());e=e[:count];v=v[:,:count]
    else:
        e,v=eigsh(h,k=min(count,h.shape[0]-2),which='SA',tol=tolerance)
        order=np.argsort(e);e=e[order];v=v[:,order]
    return e,v,[float(np.linalg.norm(h@v[:,i]-e[i]*v[:,i])) for i in range(len(e))]


def probabilities(state,dimension,nphi):
    return np.sum(abs(state.reshape(dimension,nphi,-1))**2,axis=(1,2))


def physical_arrays(coupling):
    counts=np.array([row['counts'] for row in coupling['labels']],float).reshape(-1,4,4)
    ncolour=counts[:,:,0]+counts[:,:,2];nwhite=counts[:,:,1]+counts[:,:,3]
    q=2*counts[:,:,0]/3-counts[:,:,2]/3-counts[:,:,3]
    baryon=(ncolour-6)/3
    compact=np.sort(ncolour,axis=1).astype(int).tolist()
    c=np.array([x==[6,6,6,9] for x in compact]);f=np.array([x==[6,7,7,7] for x in compact])
    csite=np.stack([c&(ncolour[:,s]==9) for s in range(4)],axis=1)
    holesA=32-sum(counts[:,v,2*(v%2)]+counts[:,v,2*(v%2)+1] for v in range(4))
    holesB=32-sum(counts[:,v,2*(1-v%2)]+counts[:,v,2*(1-v%2)+1] for v in range(4))
    return dict(counts=counts,ncolour=ncolour,nwhite=nwhite,Q=q,B=baryon,C=c,F=f,Csite=csite,
        holes=np.minimum(holesA,holesB))


def mean(state,op,nphi):
    n=op.shape[0];matrix=state.reshape(n,nphi,-1)
    transformed=(op@matrix.reshape(n,-1)).reshape(matrix.shape)
    return float(np.vdot(matrix,transformed).real)


def readout(state,coupling,scalar,arrays,observables):
    n=coupling['dimension'];nf=len(scalar['grid']);p=probabilities(state,n,nf)
    result={'norm':float(p.sum()),'compact_quark_occupation_channel':float(p@arrays['C']),
        'fragmented_quark_occupation_channel':float(p@arrays['F']),
        'compact_site_PVM':(p@arrays['Csite']).tolist(),
        'Q_em_site_mean':(p@arrays['Q']).tolist(),'baryon_site_mean':(p@arrays['B']).tolist(),
        'ncolour_site_mean':(p@arrays['ncolour']).tolist(),'nwhite_site_mean':(p@arrays['nwhite']).tolist(),
        'nwhite_site_variance':(p@(arrays['nwhite']**2)-(p@arrays['nwhite'])**2).tolist(),
        'hole_order_relative_nearest_occupied_Mott_seed':{str(int(h)):float(p[arrays['holes']==h].sum()) for h in np.unique(arrays['holes'])},
        'baryon_site_covariance':((arrays['B'].T*p)@arrays['B']-np.outer(p@arrays['B'],p@arrays['B'])).tolist(),
        'meaning':'TRUE joint quark-occupation sector/PVM and whole-sea densities; NOT EM-charge sector, colour-Casimir or a three-added-particle position PVM.'}
    tensor=state.reshape(n,nf,-1);rho=np.einsum('nfg,nhg->fh',tensor.conj(),tensor)
    weights=np.diag(rho).real;phi=float(weights@scalar['grid'])
    result.update(phi=phi,phi_variance=float(weights@(scalar['grid']**2)-phi**2),
        scalar_reduced_purity=float(np.trace(rho@rho).real),
        scalar_boundary=float(weights@np.diag(scalar['boundary'])))
    for name,op in observables.items():result[name]=mean(state,op,nf)
    return result


def nonhop(ops,coupling,primitive,scalar,rotor):
    body=sparse.eye(ops['B'].shape[0],format='csr');k=angular.right_angular(rotor)
    kinetic=sparse.csr_matrix((10,10),dtype=complex)
    for a,b in np.ndindex(3,3):kinetic+=primitive.graph['inertia_inverse'][a,b]*(k[a]@k[b])/2
    c=angular.project_product(coupling,ops['electric']-primitive.p['plaquette']*ops['B'],sparse.eye(10))
    c+=angular.project_product(coupling,body,kinetic)
    nf=len(scalar['grid'])
    return sparse.kron(c,sparse.eye(nf),format='csr')+sparse.kron(sparse.eye(c.shape[0]),scalar['H'],format='csr')


def evolve(h,initial,times):
    state=initial.copy();previous=0.;trace=h.diagonal().sum();result=[]
    for t in times:
        if t!=previous:state=expm_multiply(-1j*(t-previous)*h,state,traceA=-1j*(t-previous)*trace)
        result.append(state.copy());previous=t
    return result


def channel_spectra(h,arrays,nphi):
    result={}
    for name in ('C','F'):
        keep=np.flatnonzero(np.repeat(arrays[name],nphi))
        if len(keep)==0:
            result[name]={'dimension':0,'meaning':'Actual selected family has NO support: no invented threshold or fallback.'};continue
        energies,states,res=low(h[keep][:,keep])
        result[name]={'dimension':len(keep),'energies':energies.tolist(),'residuals':res}
    if all(result[k]['dimension'] for k in ('C','F')):
        result['fragmented_minus_compact']=result['F']['energies'][0]-result['C']['energies'][0]
    result['meaning']='Two NONNESTED gauge-invariant regional quark-occupation constrained Hamiltonians, all retained Phi/R/colour within each. Contrast is a discriminator, NOT free coloured breakup threshold or binding from Efull<Econstrained.'
    return result


def current_matrix(basis,primitive,coupling,rotor,scalar,edge,charge):
    nf=len(scalar['grid']);component=basis.matrix(lambda v:primitive.collective(v,charge=charge,edge_only=edge))
    jphi=primitive.p['scalar_coupling']*angular.project_product(coupling,component,sparse.eye(10))
    j0=sparse.csr_matrix(jphi.shape,dtype=complex)
    for i,a in np.ndindex(3,3):
        component=basis.matrix(lambda v,i=i,a=a:primitive.collective(v,i+1,a,charge=charge,edge_only=edge))
        j0+=primitive.p['vector_velocity']*angular.project_product(coupling,component,rotor['coordinate'][i][a])
    return sparse.kron(jphi,scalar['phi'],format='csr')+sparse.kron(j0,sparse.eye(nf),format='csr')


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    neutral=load(config['neutral_basis_snapshot']);charged=load(config['charged_basis_snapshot']);p=neutral['p']
    if charged['p']!=p:raise ValueError('Mismatched candidate law')
    primitive=primitive_for(charged);primitive_for(neutral,primitive)
    scalar,rotor=model.field_and_rotor(p);nf=len(scalar['grid'])
    uN=load(config['neutral_coupling_snapshot']);hN=sparse.load_npz(config['neutral_H_snapshot'])
    lowN=np.load(config['neutral_states_snapshot']);omega=lowN['states'][:,0];E0=float(lowN['energies'][0])
    cbody=neutral['basis'].matrix(lambda v:conjugate_packet(primitive,v))
    cN=angular.project_product(uN,cbody,sparse.eye(10));cOmega=(cN@omega.reshape(uN['dimension'],nf)).reshape(-1)
    plus=omega+cOmega
    if np.linalg.norm(plus)>1e-8:omega=plus/np.linalg.norm(plus);Cparity=1
    else:omega=omega/np.linalg.norm(omega);Cparity=-1
    np.savez(out+'_reference.npz',state=omega,energy=E0,Cparity=Cparity)
    result={'kind':'ACTUAL_OCCUPIED_COMPOSITE_CHANNEL_STRUCTURE_MOTION_CONSUMER','parameters':p,
        'reference':{'energy':E0,'Cparity':Cparity,'same_H_eigen_residual':float(np.linalg.norm(hN@omega-E0*omega)),
            'C_error':float(np.linalg.norm((cN@omega.reshape(uN['dimension'],nf)).reshape(-1)-Cparity*omega)),
            'meaning':'Own new J0 stationary occupied Mott/Krylov reference. C-even choice within actual degenerate branch if supported; NO invented tunnelling or exact negative-band-sea claim.'},'sectors':{}}
    ops=load(config['charged_operators_snapshot'])
    for j2 in (1,3):
        coupling=load(config[f'charged_J{j2}_coupling_snapshot']);h=sparse.load_npz(config[f'charged_J{j2}_H_snapshot'])
        poles=np.load(config[f'charged_J{j2}_states_snapshot']);energies=poles['energies'];states=poles['states']
        ground=np.flatnonzero(abs(energies-energies[0])<1e-7);pole=states[:,ground]/np.sqrt(len(ground))
        arrays=physical_arrays(coupling)
        observables={name:angular.project_product(coupling,ops[name],sparse.eye(10)) for name in ('electric','B','C')}
        sector={'ground_energy':float(energies[0]),'gap':float(energies[0]-E0),'ground_multiplicity_at_fixed_M':len(ground),
            'pole_scope':'Uniform mixed projector over observed degenerate fixed-M groundspace, NOT fitted source/receiver or full ground-domain theorem.',
            'ground':readout(pole,coupling,scalar,arrays,observables),'channels':channel_spectra(h,arrays,nf)}
        E=observables['electric'];hcolour_off=h-sparse.kron(E,sparse.eye(nf),format='csr')
        eo,vo,res=low(hcolour_off);sector['colour_electric_off']={'ground_energy':float(eo[0]),'residuals':res,
            'ground':readout(vo[:,:1],coupling,scalar,arrays,observables),'channels':channel_spectra(hcolour_off,arrays,nf),
            'meaning':'Same occupied packet/Phi/plaquette/hops; remove colour-electric term ONLY. NOT guaranteed nonbinding because common quantumPhi/holonomy remain.'}
        Path(out+f'_J{j2}_structure.json').write_text(json.dumps(sector,indent=2)+'\n')
        if j2==1:
            mask=np.repeat(arrays['Csite'][:,0],nf)
            prepared=pole*mask[:,None];strength=float(np.linalg.norm(prepared)**2)
            sector['compact_site0_preparation_strength']=strength
            if strength>1e-20:
                prepared/=np.sqrt(strength)
                scenarios={'compact_site0':prepared,
                    'baryon_phase_kick':np.exp(1j*p['density_kick']*np.repeat(arrays['B'][:,0],nf))[:,None]*prepared,
                    'field_only_kick':(prepared.reshape(coupling['dimension'],nf,-1).transpose(0,2,1)@expm(-1j*p['field_kick']*scalar['P']).T).transpose(0,2,1).reshape(prepared.shape)}
                h_off=nonhop(ops,coupling,primitive,scalar,rotor)
                snapshots={};events={}
                for name,initial in list(scenarios.items())+[('ALL_hops_off',prepared)]:
                    evolved=evolve(h_off if name=='ALL_hops_off' else h,initial,p['times']);snapshots[name]=evolved
                    events[name]=[{'t':t,**readout(v,coupling,scalar,arrays,observables)} for t,v in zip(p['times'],evolved)]
                    print(json.dumps({'stage':'actual_full_H_motion','scenario':name,'times':p['times']}),flush=True)
                sector['events']=events
                np.savez(out+'_motion_states.npz',**{f'{name}_{i}':v for name,values in snapshots.items() for i,v in enumerate(values)})
                currents={};ward={}
                for charge in ('baryon','EM'):
                    divergence=[sparse.csr_matrix(h.shape,dtype=complex) for _ in range(4)]
                    currents[charge]={}
                    for edge,(a,b) in enumerate(model.EDGES):
                        current=current_matrix(charged['basis'],primitive,coupling,rotor,scalar,edge,charge)
                        currents[charge][str(edge)]={name:[float(np.vdot(v,current@v).real) for v in values] for name,values in snapshots.items() if name!='ALL_hops_off'}
                        divergence[a]+=current;divergence[b]-=current
                    densities=arrays['B'] if charge=='baryon' else arrays['Q']
                    errors=[]
                    for s in range(4):
                        density=np.repeat(densities[:,s],nf);coo=h.tocoo()
                        comm=sparse.csr_matrix((1j*(density[coo.col]-density[coo.row])*coo.data,(coo.row,coo.col)),shape=h.shape)
                        errors.append(float(sparse.linalg.norm(comm+divergence[s])))
                    ward[charge]=errors
                sector['original_material_bond_currents']=currents;sector['original_material_Ward_Frobenius']=ward
                sector['current_scope']='Original -dH/d(edge phase), typed physical EM fractions or baryon1/3. MATERIAL BOND currents, NOT full lab Cartesian/mechanical current or an exercised continuum EM source.'
            else:sector['motion_failure']='Actual chosen groundspace has zero compact-site0 channel support; no fitted state/capture substitution.'
        result['sectors'][str(j2)]=sector
        Path(out+'_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    result['threehalf_minus_half']=result['sectors']['3']['ground_energy']-result['sectors']['1']['ground_energy']
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    result['limits']='Chosen occupied Mott/sea/pair packet family, PW19, properSO3j0+j1, scalarDVR7. Quark occupation channels not EM/colour8/Casimir or three-added-particle positions. Contrast not asymptotic binding; source gaps/prep costs not thresholds. Full H/current-family tails/family bias unbounded. No baryonmass/COM/proton/Nature/photons claim.'
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
