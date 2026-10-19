"""NEW severe count-resolved source-of-dressed-reference cloud model W_j.
W_j=span{P_counts S_i† Omega}; NOT the proposed independent Hhop-cloud V_j.
Full primitive matrix entries are streamed into W, never a huge Hpsi catalogue.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import product
from pathlib import Path
import gc,json,resource,sys,time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh,expm,solve
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_compile import save
from rich_composite_geometry_20261015_cloud_probe import LocalElectric,source,neutral_state,inner


def specs(config):
    for j in (1,3):
        for position in config['compact_anchors']:
            yield j,(position,'half_triplet' if j==1 else 'threehalf',j)
        for position in config['separated_anchors']:
            for kind in (('half_triplet','half_singlet') if j==1 else ('threehalf',)):
                yield j,(position,kind,j)


def subtract(state,coefficient,other):
    if coefficient==0:return
    for key,value in other.items():
        if key in state:state[key]-=coefficient*value
        else:state[key]=-coefficient*value


def construct(primitive,omega,config,out):
    groups={};vectors=[];labels=[];probes=[];minimum=1.;nulls=0
    for index,(j,spec) in enumerate(specs(config)):
        state=source(primitive,omega,spec);parts={}
        for key,field in state.items():parts.setdefault(primitive.label(key)[:-1],{})[key]=field
        for counts,part in parts.items():
            block=groups.setdefault((j,counts),[]);original=float(inner(part,part).real)
            if original<1e-25:nulls+=1;continue
            value={key:field.copy() for key,field in part.items()}
            for _ in range(2):
                for column in block:subtract(value,inner(vectors[column],value),vectors[column])
            norm=float(inner(value,value).real)
            if norm<max(1e-25,1e-20*original):nulls+=1;continue
            for field in value.values():field/=np.sqrt(norm)
            block.append(len(vectors));vectors.append(value);labels.append({'twice_J':j,'counts':counts})
            minimum=min(minimum,norm/original)
        probes.append({'twice_J':j,'spec':spec,'raw':state,'strength':float(inner(state,state).real)})
        print(json.dumps({'stage':'new_dressed_source_cloud_count_basis','source_index':index,'dimension':len(vectors),'source_material_count_pieces':len(parts)}),flush=True)
    dimension=len(vectors)
    for probe in probes:
        probe['dual_amplitudes']=np.array([inner(vector,probe['raw']) if label['twice_J']==probe['twice_J'] else 0. for vector,label in zip(vectors,labels)],complex)
        del probe['raw']
    lookup={}
    for column,vector in enumerate(vectors):
        for key,field in vector.items():lookup.setdefault(key,[]).append((column,field))
    del vectors;gc.collect()
    for key,values in lookup.items():
        columns,fields=zip(*values);lookup[key]=(np.asarray(columns,dtype=np.int32),np.stack(fields))
    receipt={'dimension':dimension,'count_blocks':len(groups),'sources':len(probes),'raw_rows':len(lookup),
        'coefficient_bytes':sum(fields.nbytes for _,fields in lookup.values()),'maximum_raw_row_multiplicity':max(len(columns) for columns,_ in lookup.values()),
        'null_parts':nulls,'minimum_accepted_relative_norm2':minimum,
        'meaning':'Twice MGS within physical spin-summed16-count/J sectors;1e-20 relative and1e-25 absolute norm2 null thresholds. NO Sz splitting of the coupled field state. All source-field coefficients retained; additional W cloud/anchor span cut explicit.'}
    Path(out+'_basis_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return lookup,labels,probes,receipt


def assemble(primitive,lookup,labels,scalar,rotor,out):
    dimension=len(labels);names=('Gram','field','electric','plaquette','quark_hop','white_hop','Phi','Phi2','Pphi','boundary','loopB','loopC')
    matrices={name:np.zeros((dimension,dimension),complex) for name in names}
    target_counts={tuple(label['counts']) for label in labels};p=primitive.p;electric=LocalElectric(primitive)
    inverse=np.asarray(primitive.graph['inertia_inverse'],float);generators=angular.right_angular(rotor);kinetic=np.zeros((10,10),complex)
    for a,b in np.ndindex(3,3):kinetic+=inverse[a,b]*(generators[a]@generators[b]).toarray()/2
    axes=np.asarray(primitive.graph['axes']);weights=np.asarray(primitive.graph['weights'])
    rotations={(edge,i):sum((axes[edge,a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for edge,i in product(range(4),range(3))}
    counters={'primitive_hop_calls':0,'count_channels_outside_W_skipped':0,'raw_transitions_into_W':0};started=time.time()
    def project(target,columns,fields,entries):
        present=lookup.get(target)
        if present is None:return
        rows,bra=present;value=bra.reshape(len(rows),-1).conj()@fields.reshape(len(columns),-1).T;index=np.ix_(rows,columns)
        for name,coefficient in entries:matrices[name][index]+=coefficient*value
        counters['raw_transitions_into_W']+=1
    for row,(key,(columns,fields)) in enumerate(lookup.items()):
        counts=primitive.label(key)[:-1];unit={key:1.+0j}
        project(key,columns,fields,[('Gram',1.)])
        field_action=np.einsum('ij,bjf->bif',kinetic,fields)+fields@scalar['H'].T
        project(key,columns,field_action,[('field',1.)])
        for name,operator in (('Phi',scalar['phi']),('Phi2',scalar['phi']@scalar['phi']),('Pphi',scalar['P']),('boundary',scalar['boundary'])):
            project(key,columns,fields@operator.T,[ (name,1.) ])
        for target,value in electric.action(key).items():project(target,columns,fields,[('electric',value)])
        for pw,value in primitive.field_maps['B'][key[-1]]:project(key[:-1]+(pw,),columns,fields,[('plaquette',-p['plaquette']*value),('loopB',value)])
        for pw,value in primitive.field_maps['C'][key[-1]]:project(key[:-1]+(pw,),columns,fields,[('loopC',value)])
        for edge,(start,end) in enumerate(model.EDGES):
            transformed=[p['scalar_coupling']*weights[edge]*fields*scalar['grid'][None,None,:]]
            transformed.extend(1j*p['vector_velocity']*weights[edge]*np.einsum('ij,bjf->bif',rotations[edge,i],fields) for i in range(3))
            for flavour,iswhite,dagger in product(range(2),(False,True),(False,True)):
                changed_counts=list(counts);category=2*flavour+int(iswhite);direction=-1 if dagger else 1
                changed_counts[4*start+category]-=direction;changed_counts[4*end+category]+=direction
                if tuple(changed_counts) not in target_counts:counters['count_channels_outside_W_skipped']+=1;continue
                for component in range(4):
                    image=primitive.hop(unit,edge,flavour,iswhite,component,dagger);counters['primitive_hop_calls']+=1
                    coefficient=-1 if component and dagger else 1;name='white_hop' if iswhite else 'quark_hop'
                    for target,value in image.items():project(target,columns,transformed[component],[(name,coefficient*value)])
        if row%256==0:print(json.dumps({'stage':'actual_new_W_full_H_contraction','raw_row':row,'raw_rows':len(lookup),'elapsed_seconds':time.time()-started,**counters}),flush=True)
    raw_errors={name:float(np.linalg.norm(matrix-matrix.conj().T)) for name,matrix in matrices.items()}
    np.savez(out+'_raw_matrices.npz',**matrices)
    matrices={name:(value+value.conj().T)/2 for name,value in matrices.items()}
    return matrices,{'raw_Hermiticity_Frobenius':raw_errors,'roundoff_Hermitization':'Both raw entries retained in raw_matrices NPZ; analysis uses their Hermitian average. No mass/spin/receiver fitting or clipping.','elapsed_seconds':time.time()-started,**counters,
        'projection':'Target-support/count skips are EXACT matrix-entry selection for the explicitly NEW W, not full-H range closure. FULL local colour generators act TWICE before final target selection. No all-Hpsi arrays are allocated.'}


def arrays(labels):
    counts=np.array([row['counts'] for row in labels],float).reshape(-1,4,4);colour=counts[:,:,0]+counts[:,:,2];white=counts[:,:,1]+counts[:,:,3]
    compact=np.all(np.sort(colour,axis=1)==[6,6,6,9],axis=1);fragmented=np.all(np.sort(colour,axis=1)==[6,7,7,7],axis=1)
    return {'counts':counts,'colour':colour,'white':white,'Q':2*counts[:,:,0]/3-counts[:,:,2]/3-counts[:,:,3],
        'B':(colour-6)/3,'C':compact,'F':fragmented,'Csite':compact[:,None]&(colour==9),
        'J':np.array([row['twice_J'] for row in labels])}


def expectation(state,matrix):return float(np.vdot(state,matrix@state).real)


def readout(state,gram,data,observables,initial_count=None):
    norm=expectation(state,gram);weights=np.real(np.conjugate(state)*(gram@state));result={'norm':norm,
        'compact':float(weights@data['C']),'fragmented':float(weights@data['F']),'compact_site':(weights@data['Csite']).tolist(),
        'baryon_site_mean':(weights@data['B']).tolist(),'Q_em_site_mean':(weights@data['Q']).tolist(),
        'ncolour_site_mean':(weights@data['colour']).tolist(),'nwhite_site_mean':(weights@data['white']).tolist(),
        'nwhite_site_variance':(weights@(data['white']**2)-(weights@data['white'])**2).tolist()}
    for name,matrix in observables.items():result[name]=expectation(state,matrix)
    result['Phi_variance']=result['Phi2']-result['Phi']**2
    if initial_count is not None:
        current=defaultdict(float)
        for counts,weight in zip(data['counts'],weights):current[tuple(counts.reshape(-1))]+=float(weight)
        result['material_count_PVM_TV_from_preparation']=sum(abs(current.get(key,0.)-initial_count.get(key,0.)) for key in current.keys()|initial_count.keys())/2
    return result


def spectrum(h,gram):
    energy,vectors=eigh(h,gram);return energy,vectors


def evolve(h,gram,state,times,spectral=None):
    energy,vectors=spectrum(h,gram) if spectral is None else spectral
    amplitude=vectors.conj().T@(gram@state)
    return [vectors@(np.exp(-1j*t*energy)*amplitude) for t in times]


def source_connection(hop,data,edge,order,angle=0.,charge='EM'):
    start,end=model.EDGES[edge];number=data['counts'].sum(axis=2);left=number[:,start,None]-number[None,:,start];right=number[:,end,None]-number[None,:,end]
    mask=(abs(left)==1)&(left+right==0)
    for site in range(4):
        if site not in (start,end):mask&=number[:,site,None]==number[None,:,site]
    density=data['Q'] if charge=='EM' else data['B'];q=density[:,end,None]-density[None,:,end]
    factor=(np.exp(1j*angle*q)-1) if order==0 else -1j*q if order==1 else -q*q
    return hop*mask*factor


def channels(h,gram,data):
    result={}
    for name in ('C','F'):
        selected=np.flatnonzero(data[name]);result[name]={'dimension':len(selected)}
        if len(selected):result[name]['minimum_energy']=float(eigh(h[np.ix_(selected,selected)],gram[np.ix_(selected,selected)],eigvals_only=True)[0])
    if all(result[name]['dimension'] for name in ('C','F')):result['F_minus_C']=result['F']['minimum_energy']-result['C']['minimum_energy']
    result['meaning']='NONNESTED selected whole-sea quark-count sectors, NOT an asymptotic/free breakup threshold or unique3-added-particle position PVM.'
    return result


def analyze(matrices,labels,probes,p,reference_energy,config,out):
    full=matrices['field'].copy()
    for name in ('electric','plaquette','quark_hop','white_hop'):full+=matrices[name]
    gram=matrices['Gram'];data=arrays(labels)
    hop=matrices['quark_hop']+matrices['white_hop'];result={'reference_energy':reference_energy,'sectors':{}}
    for j in (1,3):
        keep=np.flatnonzero(data['J']==j);ix=np.ix_(keep,keep);h=full[ix];g=gram[ix];local={name:value[keep] for name,value in data.items()}
        observables={name:matrices[name][ix] for name in ('Phi','Phi2','loopB','loopC','electric')}
        energy,vectors=spectrum(h,g);ground=np.flatnonzero(abs(energy-energy[0])<1e-8)
        sector={'dimension':len(keep),'minimum_energy':float(energy[0]),'gap_from_OWN_reference':float(energy[0]-reference_energy),
            'generalized_eigen_residual':float(np.linalg.norm(h@vectors[:,0]-energy[0]*(g@vectors[:,0]))),
            'ground':readout(vectors[:,0],g,local,observables),'channels':channels(h,g,local),'sources':{}}
        compact_probe=None
        for probe in probes:
            if probe['twice_J']!=j:continue
            dual=probe['dual_amplitudes'][keep];coordinate=solve(g,dual,assume_a='her');strength=float(np.vdot(dual,coordinate).real)
            residue=float(np.linalg.norm(vectors[:,ground].conj().T@dual)**2);key=str(probe['spec'])
            sector['sources'][key]={'full_raw_strength':probe['strength'],'W_retained_strength':strength,'omitted_strength_signed':probe['strength']-strength,
                'raw_ground_residue':residue,'conditional_ground_fraction':residue/strength if strength else None,
                'preparation_energy':expectation(coordinate,h)/strength if strength else None}
            if probe['spec'][0]==[0,0,0]:compact_probe=coordinate
        if compact_probe is None:raise ValueError('Declared compact source0 absent; no receiver replacement')
        prepared=local['Csite'][:,0]*compact_probe;support=expectation(prepared,g)
        sector['compact0_conditioning_strength']=support
        if support<=0:raise ValueError('Actual W compact0 preparation absent; no source/state fallback')
        prepared/=np.sqrt(support);weights=np.real(prepared.conj()*(g@prepared));initial_count=defaultdict(float)
        for counts,weight in zip(local['counts'],weights):initial_count[tuple(counts.reshape(-1))]+=float(weight)
        off=h-matrices['electric'][ix];off_energy,off_vectors=spectrum(off,g)
        sector['electric_off']={'minimum_energy':float(off_energy[0]),'ground':readout(off_vectors[:,0],g,local,observables),'channels':channels(off,g,local),
            'meaning':'SAME W/reference state and fields, only electric term removed; NOT guaranteed nonbinding.'}
        times=p['times'];events={}
        nohop=matrices['field'][ix]+matrices['electric'][ix]+matrices['plaquette'][ix]
        trajectories={}
        for name,operator in (('full',h),('colour_electric_off',off),('ALL_hops_off',nohop)):
            spectral=(energy,vectors) if name=='full' else (off_energy,off_vectors) if name=='colour_electric_off' else None
            trajectories[name]=evolve(operator,g,prepared,times,spectral)
            events[name]=[{'t':t,**readout(state,g,local,observables,initial_count)} for t,state in zip(times,trajectories[name])]
        sector['finite_prepared_events']=events;result['sectors'][str(j)]=sector
        Path(out+'_first_physics_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'stage':'FIRST_ACTUAL_NEW_W_COMPACT_SURVIVAL_TRANSFER_AND_CHANNELS','twice_J':j,'sector':sector}),flush=True)
        currents={};contacts={};ward={};selected_hop=hop[ix]
        for charge in ('EM','baryon'):
            density=local['Q'] if charge=='EM' else local['B'];divergence=[np.zeros_like(h) for _ in range(4)];currents[charge]={};contacts[charge]={}
            for edge,(start,end) in enumerate(model.EDGES):
                current=source_connection(selected_hop,local,edge,1,charge=charge);second=source_connection(selected_hop,local,edge,2,charge=charge)
                divergence[start]+=current;divergence[end]-=current
                currents[charge][str(edge)]=[expectation(state,current) for state in trajectories['full']]
                contacts[charge][str(edge)]=expectation(prepared,second)
            ward[charge]=[float(np.linalg.norm(1j*(h*density[None,:,site]-density[:,None,site]*h)+divergence[site])) for site in range(4)]
        theta=config['material_EM_angle'];duration=config['material_EM_duration'];pulse=h+source_connection(selected_hop,local,3,0,theta)
        em_initial=evolve(pulse,g,prepared,[duration])[0];same_age=evolve(h,g,prepared,[duration],(energy,vectors))[0]
        field_initial=evolve(matrices['Pphi'][ix],g,prepared,[p['field_kick']])[0]
        sector['material_EM_pulse_events']=[{'t':t,**readout(state,g,local,observables)} for t,state in zip(times,evolve(h,g,em_initial,times,(energy,vectors)))]
        sector['same_age_no_EM_events']=[{'t':t,**readout(state,g,local,observables)} for t,state in zip(times,evolve(h,g,same_age,times,(energy,vectors)))]
        sector['field_only_kick_events']=[{'t':t,**readout(state,g,local,observables)} for t,state in zip(times,evolve(h,g,field_initial,times,(energy,vectors)))]
        sector['original_Peierls_current']=currents;sector['original_Peierls_contact']=contacts;sector['Ward_after_MGS_and_actual_metric']=ward
        sector['source_law']='Original full primitive material link exp(i q theta) BEFORE W projection. Each fixed-count matrix block has unique single-hop edge/class and q=end physicalQ change, so its phase factors uniformly out of the exact raw contraction. J=-dH/dtheta and contact=d2H/dtheta2, NOT current defined from a commutator. Material source independentR; no lab3D A/mechanicalEM/photon claim.'
        np.savez(out+f'_J{j}_spectrum.npz',energies=energy,vectors=vectors,Gram=g,H=h,prepared=prepared)
        Path(out+'_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    result['threehalf_minus_half']=result['sectors']['3']['minimum_energy']-result['sectors']['1']['minimum_energy']
    return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    neutral=load(config['neutral_basis_snapshot']);primitive=primitive_for(neutral);p=neutral['p'];coupling=load(config['neutral_coupling_snapshot']);states=np.load(config['neutral_states_snapshot'])
    omega=neutral_state(neutral,coupling,states['states'][:,0]);reference_energy=float(states['energies'][0]);del neutral,coupling,states;gc.collect()
    scalar,rotor=model.field_and_rotor(p);lookup,labels,probes,basis=construct(primitive,omega,config,out)
    save(out+'_W_basis.pkl',{'p':p,'primitive':primitive,'lookup':lookup,'labels':labels,'probes':probes,'reference_energy':reference_energy,'basis_receipt':basis})
    matrices,assembly=assemble(primitive,lookup,labels,scalar,rotor,out)
    np.savez(out+'_matrices.npz',**matrices)
    metric={'coordinate_Gram_Frobenius':float(np.linalg.norm(matrices['Gram']-np.eye(len(labels)))),
        'count_metric_commutator_max':max(float(np.linalg.norm(matrices['Gram']*np.asarray([row['counts'][a] for row in labels])[None,:]-np.asarray([row['counts'][a] for row in labels])[:,None]*matrices['Gram'])) for a in range(16)),
        'scope':'Actual post-MGS full-field coordinate Gram and physical spin-summed count commutation. All analysis uses generalized H/G dynamics and true count sectors, not assumed metricI. Local Fock roundoff remains disclosed; no full-H range closure.'}
    result={'kind':'NEW_SEVERE_DRESSED_REFERENCE_SOURCE_COUNT_CLOUD_CONDITIONAL_COMPOSITE_MODEL','parameters':p,'basis':basis,'assembly':assembly,'metric':metric,
        'ansatz':'W_j=span{P_counts S_i† OWN dressedOmega}, four compact+four cyclic separated physical anchors; both half copies at distinct-u anchors and threehalf. Omega ALREADY carries actual quark/white sea-hole/Phi/PW/R cloud. NO additional independent HhopOmega cloud, whose measured materialization cost was1.88GB/one source. This is a NEW additional severe field/sea-cloud+anchor span cut, NOT old68440 completion or a full native/full-law survival claim.',
        'Noether':'Source tensor rankj applied to actual J0Omega, spin-summed P_counts commute physical J=Lleft+SCAR. Stored highestM=j represents each proper physical multiplet, not borrowed Frame labels or fitted spin penalty; otherM covariance is by construction, not a new global-spin catalogue. No freeCOM/continuum spatial Noether claim.'}
    result.update(analyze(matrices,labels,probes,p,reference_energy,config,out))
    result['limits']='Conditional finite variational W with additional severe Omega-linked field/sea cloud and cyclic-anchor cut; primitivePhi7/R10/PW19 coordinates retained INSIDE vectors, not the full independent tensor virtual space. No claimed H/current range closure, variational nesting with68440 or globalvacuum/spin ordering. C/F are whole-sea quark-count channels, not3-added-particle position or asymptotic threshold. Colour-off need not be nonbinding; exact quadratic control separate. Source strengths/residues and probabilities separate. Primitive negative scalar quadratic/volume/mechanical mass density/SU3/charge/gauging inputs unchanged. No mass rebase, fitted binder/spin/receiver, proton, Lorentz/freeCOM/photon/whole-issue claim. Existing fuller native consumers remain running, not silently replaced.'
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
