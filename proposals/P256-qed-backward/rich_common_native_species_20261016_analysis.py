"""Full native generalized-metric dynamics and physical species source response."""
import numpy as np
from scipy import sparse
from rich_composite_geometry_20261015_cloud_variational import arrays
from rich_composite_geometry_20261015_dressing_science import metric_inverse,evolution,evolve,readout,count_probabilities,tensor_frobenius,metric_operator,spectrum,WeakPhi
import rich_composite_geometry_20261015_model as model


def native(operators,labels,sector,scalar):
    keep=np.array([i for i,label in enumerate(labels) if sum(label['counts'])==sector],int);local_labels=[labels[i] for i in keep];data=arrays(local_labels);ops={name:matrix[keep][:,keep].tocsr() for name,matrix in operators.items()};gram=ops['Gram'];inverse,receipt=metric_inverse(gram,local_labels)
    c0=ops['rotation']+ops['electric']+ops['plaquette']+ops['quark_vector']+ops['white_vector'];cphi=ops['quark_scalar']+ops['white_scalar']
    data['Qwhite']=-data['counts'][:,:,3];data['Qquark']=2*data['counts'][:,:,0]/3-data['counts'][:,:,2]/3
    return {'keep':keep,'labels':local_labels,'data':data,'ops':ops,'gram':gram,'inverse':inverse,'metric':receipt,'c0':c0,'cphi':cphi,'scalar':scalar,'nf':len(scalar['grid'])}


def coefficients(probe,space):return probe['coordinates'][space['keep']].reshape(-1)


def norm(value,space):return float(np.vdot(value,metric_operator(space['gram'],space['nf'])@value).real)


def prepare(probe,space):
    state=coefficients(probe,space);strength=norm(state,space)
    if strength<=0:raise ValueError('Declared unmasked canonical source absent; no state/receiver fallback')
    return state/np.sqrt(strength),strength


def hopping(space,species):
    if species not in ('quark','white'):raise ValueError('Actual microscopic quark/white species required')
    return space['ops'][species+'_vector'],space['ops'][species+'_scalar']


def source_matrix(matrix,data,edge,order,theta=0.,species='total'):
    coo=matrix.tocoo();start,end=model.EDGES[edge];number=data['counts'].sum(axis=2);delta=number[coo.row]-number[coo.col];mask=(abs(delta[:,start])==1)&(delta[:,start]+delta[:,end]==0)
    for site in range(4):
        if site not in (start,end):mask&=delta[:,site]==0
    density=data['Q'] if species=='total' else data['Q'+species];q=density[coo.row,end]-density[coo.col,end];phase=np.exp(1j*theta*q)
    factor=phase-1 if order==0 else -1j*q*phase if order==1 else -q*q*phase
    return sparse.coo_matrix((coo.data[mask]*factor[mask],(coo.row[mask],coo.col[mask])),shape=matrix.shape).tocsr()


def source_pair(space,species,edge,order,theta=0.,hops_on=True):
    first,second=hopping(space,species)
    if not hops_on:first=first*0;second=second*0
    return source_matrix(first,space['data'],edge,order,theta,species),source_matrix(second,space['data'],edge,order,theta,species)


def expectation_pair(state,first,second,space):
    tensor=state.reshape(-1,space['nf']);image=first@tensor+(second@(tensor*space['scalar']['grid'][None,:]));return float(np.vdot(tensor,image).real)


def observe(state,space):
    result=readout(state,space['gram'],space['data'],space['ops'],space['scalar']);tensor=state.reshape(-1,space['nf']);dual=space['gram']@tensor;weights=np.real(np.sum(tensor.conj()*dual,axis=1));white=space['data']['white'];colour=space['data']['colour']
    result['Qwhite_site_mean']=(weights@space['data']['Qwhite']).tolist();result['Qquark_site_mean']=(weights@space['data']['Qquark']).tolist();result['white_excess_site_mean']=(weights@(white-2)).tolist();result['Phi_momentum']=float(np.vdot(tensor,dual@space['scalar']['P'].T).real)
    for excess in (-1,1):
        channel=np.all(np.sort(white,axis=1)==([1,2,2,2] if excess<0 else [2,2,2,3]),axis=1);result['white_'+('hole'if excess<0 else'addition')+'_site_probability']=(weights@(channel[:,None]&(white==2+excess))).tolist()
    result['quark_compact_site_probability']=result.pop('compact_site');result['quark_compact_probability']=result.pop('compact');result['quark_fragmented_probability']=result.pop('fragmented')
    result['field_carrier_note']='SharedPhi/R/PW observables include actual whole quark/white sea; counts are material whole-sea PVM, not tagged particles. NetQ and compact probabilities DIFFER.'
    return result


def currents(state,space,active_species=None,active_edge=None,theta=0.,hops_on=True):
    result={}
    for species in ('quark','white'):
        result[species]=[]
        for edge in range(4):
            angle=theta if species==active_species and edge==active_edge else 0.
            first,second=source_pair(space,species,edge,1,angle,hops_on);result[species].append(expectation_pair(state,first,second,space))
    return result


def ward(space):
    result={}
    for species in ('quark','white'):
        first,second=hopping(space,species);div0=[sparse.csr_matrix(first.shape,dtype=complex)for _ in range(4)];divphi=[sparse.csr_matrix(first.shape,dtype=complex)for _ in range(4)]
        for edge,(start,end)in enumerate(model.EDGES):
            j0,jphi=source_pair(space,species,edge,1)
            div0[start]+=j0;div0[end]-=j0;divphi[start]+=jphi;divphi[end]-=jphi
        result[species]=[]
        for site in range(4):
            q=sparse.diags(space['data']['Q'+species][:,site]);w0=1j*(space['c0']@q-q@space['c0'])+div0[site];wphi=1j*(space['cphi']@q-q@space['cphi'])+divphi[site];result[species].append(tensor_frobenius(w0,wphi,space['scalar']['grid']))
    return result


def make_generator(space,delta0=None,deltaphi=None,hops_on=True):
    c0=space['c0']if hops_on else space['ops']['rotation']+space['ops']['electric']+space['ops']['plaquette'];cphi=space['cphi']if hops_on else space['cphi']*0
    if delta0 is not None:c0=c0+delta0
    if deltaphi is not None:cphi=cphi+deltaphi
    return evolution(c0,cphi,space['inverse'],space['scalar'])


def source_profile(probe,space,energies,vectors,reference):
    source=coefficients(probe,space);metric=metric_operator(space['gram'],space['nf']);dual=metric@source;strength=float(np.vdot(source,dual).real);ground=np.flatnonzero(abs(energies-energies[0])<1e-8);residue=float(np.linalg.norm(vectors[:,ground].conj().T@dual)**2);weak=WeakPhi(space['c0'],space['cphi'],space['gram'],space['scalar']['H'],space['scalar']['grid'])
    return {'descriptor':probe['descriptor'],'raw_source_strength':probe['raw_strength'],'retained_strength':strength,'omitted_signed':probe['raw_strength']-strength,'raw_ground_cluster_residue':residue,'conditional_ground_fraction':residue/strength if strength else None,'preparation_Rayleigh':float(np.vdot(source,weak@source).real/strength)if strength else None,'meaning':'Literal canonical strengths NOTprobabilities; Dsource extraenergy². Spectrum/Rayleigh relative hotselectedreference NOTphysicalmass/gap or stationarytwo-time source pole.'}
