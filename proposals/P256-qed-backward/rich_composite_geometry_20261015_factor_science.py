"""Actual complete-field compactness/channel/current/motion consumer.
All Phi modes and all intermediate states of each chosen Noether block retained.
"""
from __future__ import annotations
import json,resource,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_runtime import primitive_for,conjugate_packet
from rich_composite_geometry_20261015_factors import PhiHamiltonian,load_h
from rich_composite_geometry_20261015_stream import operator_matrix
from rich_composite_geometry_20261015_science import load,low,physical_arrays,readout,evolve,probabilities


def nonhop_factor(ops,coupling,primitive,scalar,rotor):
    body=sparse.eye(ops['B'].shape[0],format='csr');k=angular.right_angular(rotor)
    kinetic=sparse.csr_matrix((10,10),dtype=complex)
    for a,b in np.ndindex(3,3):kinetic+=primitive.graph['inertia_inverse'][a,b]*(k[a]@k[b])/2
    c=angular.project_product(coupling,ops['electric']-primitive.p['plaquette']*ops['B'],sparse.eye(10))
    c+=angular.project_product(coupling,body,kinetic)
    return PhiHamiltonian(c,sparse.csr_matrix(c.shape,dtype=complex),scalar['H'],scalar['grid'])


def channels(h,arrays):
    result={}
    for name in ('C','F'):
        keep=np.flatnonzero(arrays[name])
        if not len(keep):result[name]={'dimension':0,'meaning':'No actual family support; no fabricated threshold/state.'};continue
        restricted=h.restrict_body(keep);e,v,res=low(restricted)
        result[name]={'dimension':restricted.shape[0],'energies':e.tolist(),'residuals':res}
    if all(result[k]['dimension'] for k in ('C','F')):result['fragmented_minus_compact']=result['F']['energies'][0]-result['C']['energies'][0]
    result['meaning']='Nonnested ACTUAL quark-occupation sectors (9,6,6,6) versus (7,7,7,6), full retained quantumPhi/R/colour. Contrast is NOT an asymptotic/free coloured breakup threshold or a three-added-particle position claim.'
    return result


def bond_action(primitive,packet,edge,charge,component=0,body_axis=0,order=1,angle=0.):
    terms=[]
    for f,white in np.ndindex(2,2):
        white=bool(white);q=([0.,-1.] if white else [2/3,-1/3])[f] if charge=='EM' else (0. if white else 1/3)
        coefficient=(np.exp(1j*q*angle)-1) if order==0 else -1j*q if order==1 else -q*q
        weight=primitive.graph['weights'][edge]
        if component:weight*=primitive.graph['axes'][edge,body_axis]
        a=primitive.hop(packet,edge,f,white,component);b=primitive.hop(packet,edge,f,white,component,True)
        terms.extend(((weight*coefficient*(1j if component else 1),a),
            (weight*coefficient.conjugate()*(-1j if component else 1),b)))
    return model.add(*terms)


def bond_factor(basis,primitive,coupling,rotor,scalar,edge,charge,order=1,angle=0.):
    raw,_=operator_matrix(basis,lambda v:bond_action(primitive,v,edge,charge,order=order,angle=angle))
    cphi=primitive.p['scalar_coupling']*angular.project_product(coupling,raw,sparse.eye(10))
    c0=sparse.csr_matrix(cphi.shape,dtype=complex)
    for i,a in np.ndindex(3,3):
        raw,_=operator_matrix(basis,lambda v,i=i,a=a:bond_action(primitive,v,edge,charge,i+1,a,order,angle))
        c0+=primitive.p['vector_velocity']*angular.project_product(coupling,raw,rotor['coordinate'][i][a])
    return PhiHamiltonian(c0,cphi,np.zeros_like(scalar['H']),scalar['grid'])


def count_pvm(state,coupling,nphi):
    probability=probabilities(state,coupling['dimension'],nphi);result={}
    for row,value in zip(coupling['labels'],probability):
        key=tuple(row['counts']);result[key]=result.get(key,0.)+float(value)
    return result


def pvm_change(left,right):
    values=np.array([left.get(k,0.)-right.get(k,0.) for k in set(left)|set(right)])
    return {'L1':float(sum(abs(values))),'L2':float(np.linalg.norm(values)),'TV':float(sum(abs(values))/2)}


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    neutral=load(config['neutral_basis_snapshot']);charged=load(config['charged_basis_snapshot']);p=neutral['p']
    if charged['p']!=p:raise ValueError('Mismatch of actual law parameters')
    primitive=primitive_for(charged);primitive_for(neutral,primitive);scalar,rotor=model.field_and_rotor(p);nf=len(scalar['grid'])
    uN=load(config['neutral_coupling_snapshot']);hN=load_h(config['neutral_H_snapshot']);nstate=np.load(config['neutral_states_snapshot'])
    omega=nstate['states'][:,0];E0=float(nstate['energies'][0]);cbody=neutral['basis'].matrix(lambda v:conjugate_packet(primitive,v))
    cN=angular.project_product(uN,cbody,sparse.eye(10));cOmega=(cN@omega.reshape(uN['dimension'],nf)).reshape(-1)
    plus=omega+cOmega
    if np.linalg.norm(plus)>1e-8:omega=plus/np.linalg.norm(plus);Cparity=1
    else:omega/=np.linalg.norm(omega);Cparity=-1
    np.savez(out+'_reference.npz',state=omega,energy=E0,Cparity=Cparity)
    ground_energies={j:float(np.load(config[f'charged_J{j}_states_snapshot'])['energies'][0]) for j in (1,3)}
    delta=ground_energies[3]-ground_energies[1]
    dynamics=[1,3] if abs(delta)<1e-7 else [min(ground_energies,key=ground_energies.get)]
    result={'kind':'ACTUAL_FULL_PHI_OCCUPIED_COMPOSITE_STRUCTURE_MOTION_AND_BOND_SOURCE_CONSUMER','parameters':p,
        'reference':{'energy':E0,'Cparity':Cparity,'same_H_residual':float(np.linalg.norm(hN@omega-E0*omega)),
            'C_error':float(np.linalg.norm((cN@omega.reshape(uN['dimension'],nf)).reshape(-1)-Cparity*omega)),
            'meaning':'Own actual stationary J0 occupied Mott/correlated-family reference, NOT exactnegative-band/globalallN vacuum. C eigenchoice within actual degenerate sector, not fabricated tunnelling.'},
        'threehalf_minus_half':delta,'dynamics_twice_J':dynamics,
        'spin_scope':'Dynamics uses LOWER of the actual computed half/threehalf pole branches (both if tied), never preference-fitted halfspin. Higher dark J/global-law ordering unproved. Both required source multiplets remain.','sectors':{}}
    ops=load(config['charged_operators_snapshot'])
    for j2 in (1,3):
        coupling=load(config[f'charged_J{j2}_coupling_snapshot']);h=load_h(config[f'charged_J{j2}_H_snapshot'])
        if not isinstance(h,PhiHamiltonian):raise TypeError('This consumer requires declared full-Phi factor artifact')
        spectrum=np.load(config[f'charged_J{j2}_states_snapshot']);e=spectrum['energies'];indices=np.flatnonzero(abs(e-e[0])<1e-7)
        pole=spectrum['states'][:,indices]/np.sqrt(len(indices));arrays=physical_arrays(coupling)
        observables={name:angular.project_product(coupling,ops[name],sparse.eye(10)) for name in ('electric','B','C')}
        sector={'energy':float(e[0]),'gap':float(e[0]-E0),'observed_fixed_M_ground_multiplicity':len(indices),
            'ground':readout(pole,coupling,scalar,arrays,observables),'channels':channels(h,arrays)}
        off=h.remove_static(observables['electric']);eo,vo,res=low(off)
        sector['colour_electric_off']={'energy':float(eo[0]),'residuals':res,'channels':channels(off,arrays),
            'ground':readout(vo[:,:1],coupling,scalar,arrays,observables),
            'meaning':'SAME fullPhi/sea/holonomy/plaquette/source family, electric term removed ONLY. Not guaranteed nonbinding; genuine quadratic control separate.'}
        del off,vo
        Path(out+f'_J{j2}_structure.json').write_text(json.dumps(sector,indent=2)+'\n')
        if j2 in dynamics:
            mask=np.repeat(arrays['Csite'][:,0],nf);prepared=mask[:,None]*pole;weight=float(np.linalg.norm(prepared)**2)
            sector['compact_site0_conditioning_weight']=weight
            if weight>1e-20:
                prepared/=np.sqrt(weight);hoff=nonhop_factor(ops,coupling,primitive,scalar,rotor)
                hhop=PhiHamiltonian(h.c0-hoff.c0,h.cphi,np.zeros_like(h.hphi),h.grid);v=prepared
                sector['primitive_compact_transfer_moments']=[]
                for k in range(1,7):
                    v=hhop@v;sector['primitive_compact_transfer_moments'].append({'order':k,
                        'target_norm2':[float(np.linalg.norm(v*np.repeat(arrays['Csite'][:,s],nf)[:,None])**2) for s in range(4)]})
                del hhop,v
                sector['transfer_moment_meaning']='P_Cs Hhop^k P_C0 actual primitive virtual processes; ENERGY^(2k) norms, NOT time probabilities or assigned effective baryon hopping.'
                fieldkick=expm(-1j*p['field_kick']*scalar['P'])
                scenarios={'compact_site0':prepared,'baryon_phase_kick':np.exp(1j*p['density_kick']*np.repeat(arrays['B'][:,0],nf))[:,None]*prepared,
                    'field_only_kick':(prepared.reshape(coupling['dimension'],nf,-1).transpose(0,2,1)@fieldkick.T).transpose(0,2,1).reshape(prepared.shape),
                    'ALL_hops_off':prepared}
                theta=config['material_EM_angle'];duration=config['material_EM_duration']
                source=bond_factor(charged['basis'],primitive,coupling,rotor,scalar,3,'EM',0,theta)
                pulse=PhiHamiltonian(h.c0+source.c0,h.cphi+source.cphi,h.hphi,h.grid)
                scenarios['material_EM_bond_pulse']=expm_multiply(-1j*duration*pulse,prepared,traceA=-1j*duration*pulse.diagonal().sum())
                scenarios['same_age_no_EM_pulse']=expm_multiply(-1j*duration*h,prepared,traceA=-1j*duration*h.diagonal().sum())
                del pulse,source
                snapshots={};events={};initial=count_pvm(prepared,coupling,nf)
                for name,first in scenarios.items():
                    values=evolve(hoff if name=='ALL_hops_off' else h,first,p['times']);snapshots[name]=values
                    events[name]=[{'t':t,**readout(v,coupling,scalar,arrays,observables),
                        'count_PVM_change_from_compact_preparation':pvm_change(count_pvm(v,coupling,nf),initial)} for t,v in zip(p['times'],values)]
                    print(json.dumps({'stage':'actual_occupied_compact_motion','twice_J':j2,'scenario':name}),flush=True)
                sector['events']=events
                sector['EM_probe_scope']='Finite MATERIAL U1 bond connection on chord, fractional physical q, full owning first/second source derivatives. Independent material-link source, NOT an exercised lab3D A profile, full Cartesian/mechanical current, continuum Maxwell or photon claim. EMpulse and noEM baseline have SAME extra pulse duration before listed times.'
                np.savez(out+f'_J{j2}_motion_states.npz',**{f'{name}_{i}':v for name,values in snapshots.items() for i,v in enumerate(values)})
                current_table={};contact={};ward={}
                for charge in ('baryon','EM'):
                    zero=sparse.csr_matrix(h.c0.shape,dtype=complex);div0=[zero.copy() for _ in range(4)];divphi=[zero.copy() for _ in range(4)]
                    current_table[charge]={};contact[charge]={}
                    for edge,(a,b) in enumerate(model.EDGES):
                        current=bond_factor(charged['basis'],primitive,coupling,rotor,scalar,edge,charge)
                        current_table[charge][str(edge)]={name:[float(np.vdot(v,current@v).real) for v in values] for name,values in snapshots.items() if name!='ALL_hops_off'}
                        div0[a]+=current.c0;div0[b]-=current.c0;divphi[a]+=current.cphi;divphi[b]-=current.cphi
                        del current
                        second=bond_factor(charged['basis'],primitive,coupling,rotor,scalar,edge,charge,2)
                        contact[charge][str(edge)]=float(np.vdot(pole,second@pole).real);del second
                    density=arrays['B'] if charge=='baryon' else arrays['Q'];ward[charge]=[]
                    for site in range(4):
                        d=density[:,site];a=h.c0.tocoo();b=h.cphi.tocoo()
                        r0=sparse.csr_matrix((1j*(d[a.col]-d[a.row])*a.data,(a.row,a.col)),shape=h.c0.shape)+div0[site]
                        rp=sparse.csr_matrix((1j*(d[b.col]-d[b.row])*b.data,(b.row,b.col)),shape=h.cphi.shape)+divphi[site]
                        value=nf*sparse.linalg.norm(r0)**2+float(sum(h.grid**2))*sparse.linalg.norm(rp)**2+2*float(sum(h.grid))*float(r0.conjugate().multiply(rp).sum().real)
                        ward[charge].append(float(np.sqrt(value)))
                    del div0,divphi
                sector['original_material_bond_current']=current_table;sector['owning_bond_contact_on_pole']=contact;sector['original_material_Ward_full_field_Frobenius']=ward
                sector['current_scope']='Direct derivatives of ORIGINAL individual CAR64 Peierls hops BEFORE native/Noether projection. Full-field Ward independently compared, not current defined from commutator. Active-bond currents NOT mislabeled as off-H current; ALL-hop-off PVM is separate.'
            else:sector['motion_failure']='Actual compact-site0 conditioning support absent; no state/source/receiver replacement.'
        result['sectors'][str(j2)]=sector
        Path(out+'_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
        del h,pole
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    result['limits']='Finite chosen occupied order/source family, PW19, SO3j0+j1, fullPhiDVR7. Real sea/hole/white/Phi/flux dynamics, NOT inert trimer spectators. Occupation channels not EM charge/colour8/Casimir or3-added-particle position. Contrast finite discriminator not breakup threshold; fullH/current tails and family bias unbounded. No inferred baryonmass/newspinI/proton/COM/Lorentz/photons/wholeissue claim.'
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
