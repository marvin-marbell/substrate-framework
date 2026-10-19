"""Independent common-species metric/current physics, reusing frozen state math."""
import numpy as np
from scipy import sparse
from rich_native_body_review_20261016_math import FieldHamiltonian,metric_roots,data_from_labels,evolve,readout,count_distribution,total_variation


def species_data(labels):
    data=data_from_labels(labels);c=data['counts'];data['Qquark']=(2*c[:,:,0]-c[:,:,2])/3;data['Qwhite']=-c[:,:,3]
    w=data['white'];add=np.all(np.sort(w,axis=1)==[2,2,2,3],axis=1);hole=np.all(np.sort(w,axis=1)==[1,2,2,2],axis=1)
    data['white_addition_site']=add[:,None]&(w==3);data['white_hole_site']=hole[:,None]&(w==1)
    return data


def source_matrix(matrix,data,edge,order,theta,species):
    start,end=((0,1),(1,2),(2,3),(0,3))[edge];coo=matrix.tocoo();change=data['counts'][coo.row]-data['counts'][coo.col];number=change.sum(axis=2)
    mask=(abs(number[:,end])==1)&(number[:,start]==-number[:,end])&(np.sum(abs(change),axis=(1,2))==2)
    for site in range(4):
        if site not in (start,end):mask&=np.all(change[:,site]==0,axis=1)
    charge=data['Q'+species];q=charge[coo.row,end]-charge[coo.col,end];phase=np.exp(1j*theta*q)
    coefficient=phase-1 if order==0 else -1j*q*phase if order==1 else -q*q*phase
    return sparse.coo_matrix((coo.data[mask]*coefficient[mask],(coo.row[mask],coo.col[mask])),shape=matrix.shape).tocsr()


def native_space(meta,allops,number):
    import rich_composite_geometry_20261015_model as model
    keep=np.array([i for i,row in enumerate(meta['labels']) if sum(row['counts'])==number]);labels=[meta['labels'][i] for i in keep];data=species_data(labels)
    ops={name:matrix[keep][:,keep].tocsr() for name,matrix in allops.items()};g=ops['Gram'];s,t,receipt=metric_roots(g,labels);scalar,_=model.field_and_rotor(meta['p'])
    physical={name:(t@matrix@t).tocsr() for name,matrix in ops.items() if name!='Gram'}
    c0=physical['rotation']+physical['electric']+physical['plaquette']+physical['quark_vector']+physical['white_vector'];cp=physical['quark_scalar']+physical['white_scalar']
    return dict(meta=meta,N=number,keep=keep,labels=labels,data=data,g=g,s=s,t=t,metric=receipt,ops=ops,physical=physical,scalar=scalar,c0=c0,cp=cp,n=len(keep),nf=len(scalar['grid']),pair_cache={})


def pair(space,species,edge,order,theta=0.,hops_on=True):
    key=(species,edge,order,float(theta),hops_on)
    if key not in space['pair_cache']:
        t=space['t'];data=space['data'];ops=space['ops']
        space['pair_cache'][key]=tuple((t@source_matrix(ops[species+'_'+component] if hops_on else ops[species+'_'+component]*0,data,edge,order,theta,species)@t).tocsr() for component in ('vector','scalar'))
    return space['pair_cache'][key]


def pair_operator(space,pieces):
    return FieldHamiltonian(pieces[0],pieces[1],np.zeros_like(space['scalar']['H']),space['scalar']['grid'])


def generator(space,mode='full',delta=None):
    physical=space['physical'];scalar=space['scalar']
    if mode=='full':c0=space['c0'];cp=space['cp'];hp=scalar['H']
    elif mode=='ALLhopOFF':c0=physical['rotation']+physical['electric']+physical['plaquette'];cp=sparse.csr_matrix(space['g'].shape);hp=scalar['H']
    elif mode=='coordinate_only':
        c0=physical['plaquette']+physical['quark_vector']+physical['white_vector'];cp=space['cp'];hp=space['meta']['p']['scalar_volume_ratio']*scalar['potential']
    else:raise ValueError('Explicit physical generator required')
    if delta is not None:c0=c0+delta[0];cp=cp+delta[1]
    return FieldHamiltonian(c0,cp,hp,scalar['grid'])


def prepare(space,descriptor):
    probe=next(probe for probe in space['meta']['probes'] if probe['descriptor']==descriptor);x=probe['coordinates'][space['keep']];raw=float(np.vdot(x,space['g']@x).real)
    if raw<=0:raise AssertionError('Pinned unmasked source has zero metric norm')
    return (space['s']@x/np.sqrt(raw)).reshape(-1),dict(descriptor=descriptor,declared_raw_strength=probe['raw_strength'],retained_raw_strength=raw,omitted_signed=probe['raw_strength']-raw)


def observe(space,state,h,active_species=None,active_edge=None,theta=0.,hops_on=True):
    data=space['data'];z=state.reshape(space['n'],space['nf']);weights=np.sum(abs(z)**2,axis=1)
    result=readout(state,data,space['physical'],space['scalar'],h)
    for name in ('Qquark','Qwhite','white_addition_site','white_hole_site'):result[name+'_site' if name.startswith('Q') else name]=(weights@data[name]).tolist()
    result['Pphi']=float(np.vdot(z,z@space['scalar']['P'].T).real);result['Phi_energy']=float(np.vdot(z,z@space['scalar']['H'].T).real);result['current']={}
    for species in ('quark','white'):
        result['current'][species]=[]
        for edge in range(4):
            angle=theta if active_species==species and active_edge==edge else 0.
            op=pair_operator(space,pair(space,species,edge,1,angle,hops_on));result['current'][species].append(float(np.vdot(state,op@state).real))
    if active_species is not None:
        op=pair_operator(space,pair(space,active_species,active_edge,2,theta,hops_on));result['source_contact']=float(np.vdot(state,op@state).real)
    return result


def continuity(space):
    result={};nf=space['nf'];grid=space['scalar']['grid'];ops=space['ops'];c0=ops['rotation']+ops['electric']+ops['plaquette']+ops['quark_vector']+ops['white_vector'];cp=ops['quark_scalar']+ops['white_scalar']
    for species in ('quark','white'):
        div0=[sparse.csr_matrix(space['g'].shape,dtype=complex) for _ in range(4)];divp=[value.copy() for value in div0]
        for edge,(start,end) in enumerate(((0,1),(1,2),(2,3),(0,3))):
            a=source_matrix(ops[species+'_vector'],space['data'],edge,1,0.,species);b=source_matrix(ops[species+'_scalar'],space['data'],edge,1,0.,species);div0[start]+=a;div0[end]-=a;divp[start]+=b;divp[end]-=b
        result[species]=[]
        for site in range(4):
            q=sparse.diags(space['data']['Q'+species][:,site]);a=1j*(c0@q-q@c0)+div0[site];b=1j*(cp@q-q@cp)+divp[site]
            result[species].append(float(np.sqrt(nf*sparse.linalg.norm(a)**2+np.dot(grid,grid)*sparse.linalg.norm(b)**2)))
    if max(value for rows in result.values() for value in rows)>2e-9:raise AssertionError('Actual species continuity failed')
    return result
