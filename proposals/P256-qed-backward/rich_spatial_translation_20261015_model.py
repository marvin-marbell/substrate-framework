"""Joint lattice-translation Gaussian orbit x complete four-coordinate shells.
Oscillators are basis functions ONLY; physical field energy is before-cut P^2.
"""
from itertools import product
from math import factorial
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.special import eval_genlaguerre
from scipy.sparse.linalg import LinearOperator
from rich_spatial_collective_20261015_model import (build as collective_build,exterior_basis,
    annihilators,multiply_except,whiten,packet,source_matrices,receiver_matrices,dense_body_operator,solve)
from rich_spatial_collective_20261015_calculate import neutral_one_body


def local_translation(n):
    _,masks,lookup=exterior_basis(n);rows=[];cols=[];values=[]
    for j,mask in enumerate(masks):
        moved=[(i+4)%12 for i in range(12) if mask&(1<<i)]
        inversions=sum(a>b for k,a in enumerate(moved) for b in moved[k+1:])
        target=sum(1<<i for i in moved)
        rows.append(lookup[target]);cols.append(j);values.append((-1)**inversions)
    return sparse.csr_matrix((values,(rows,cols)),shape=(len(masks),)*2)


def shell_fields(config):
    cutoff=config['shell_cutoff'];width=config['shell_width'];volume=config['volume']
    states=sorted([x for x in product(range(cutoff+1),repeat=4) if sum(x)<=cutoff],key=lambda x:(sum(x),x))
    lookup={x:i for i,x in enumerate(states)};ns=len(states);identity=np.eye(ns)
    annihilation=[]
    for axis in range(4):
        a=np.zeros((ns,ns),complex)
        for j,state in enumerate(states):
            if state[axis]:
                target=list(state);target[axis]-=1;a[lookup[tuple(target)],j]=np.sqrt(state[axis])
        annihilation.append(a)
    coordinates=[width*(a+a.conj().T)/np.sqrt(2) for a in annihilation]
    # P^2 is formed BEFORE truncation: (2n+1-a^2-a^dagger^2)/(2 width^2).
    # In particular the diagonal includes the raising path beyond the last shell.
    em_kinetic=np.zeros((ns,ns),complex)
    for axis,a in enumerate(annihilation):
        number=np.diag([state[axis] for state in states])
        em_kinetic+=(2*number+identity-a@a-a.conj().T@a.conj().T)/(4*(volume/2)*width**2)
    lz=np.zeros((ns,ns),complex)
    for c,s in [(0,1),(2,3)]:lz+=1j*(annihilation[s].conj().T@annihilation[c]-annihilation[c].conj().T@annihilation[s])
    translation=expm(-2j*np.pi*lz/3)
    nphi=config['phi_points'];span=config['phi_width'];center=config['phi_center']
    phi=center+2*span*(np.arange(nphi)-(nphi-1)/2)/nphi
    p=2*np.pi*np.fft.fftfreq(nphi,d=2*span/nphi)
    f=np.exp(2j*np.pi*np.outer(np.arange(nphi),np.arange(nphi))/nphi)/np.sqrt(nphi)
    pmat=f@np.diag(p)@f.conj().T
    phi_h=pmat@pmat/(2*volume*config['scalar_inertia'])+np.diag(volume*config['scalar_stiffness']*(phi-center)**2/2)
    boson=sparse.kron(phi_h,sparse.eye(ns),format='csr')+sparse.kron(sparse.eye(nphi),em_kinetic,format='csr')
    identity_full=sparse.eye(nphi*ns,format='csr')
    def displacement(vector):
        result=np.ones((ns,ns),complex)
        for axis,c in enumerate(vector):
            a=c*width/np.sqrt(2)
            element=np.zeros((cutoff+1,cutoff+1),complex)
            for n in range(cutoff+1):
                for m in range(cutoff+1):
                    lo,hi=min(n,m),max(n,m)
                    element[n,m]=np.exp(-a*a/2)*np.sqrt(factorial(lo)/factorial(hi))*(1j*a)**(hi-lo)*eval_genlaguerre(lo,hi-lo,a*a)
            result*=element[np.array(states)[:,axis,None],np.array(states)[None,:,axis]]
        return result
    coupling=[identity_full,sparse.kron(sparse.diags(config['coupling']*phi),sparse.eye(ns),format='csr')]
    derivatives=[[sparse.csr_matrix(identity_full.shape) for _ in range(16)] for _ in range(4)]
    contacts=[[[sparse.csr_matrix(identity_full.shape) for _ in range(16)] for _ in range(4)] for _ in range(4)]
    component=2
    for axis in range(3):
        for level in (range(3) if axis<2 else [0]):
            profile=np.zeros(4)
            if axis<2:profile[2*axis:2*axis+2]=config['charge']*np.array([np.cos(2*np.pi*level/3),np.sin(2*np.pi*level/3)])
            d=displacement(profile);cos=(d+d.conj().T)/2;sin=(d-d.conj().T)/(2j)
            fullcos=sparse.kron(sparse.eye(nphi),cos,format='csr');fullsin=sparse.kron(sparse.eye(nphi),sin,format='csr')
            coupling += [fullcos,fullsin]
            for a in range(4):
                derivatives[a][component]=-profile[a]*fullsin;derivatives[a][component+1]=profile[a]*fullcos
                for b in range(4):
                    contacts[a][b][component]=-profile[a]*profile[b]*fullcos
                    contacts[a][b][component+1]=-profile[a]*profile[b]*fullsin
            component+=2
    frozen=[identity_full,config['coupling']*center*identity_full]
    for i in range(7):frozen += [identity_full,sparse.csr_matrix(identity_full.shape)]
    field_q=[sparse.kron(sparse.eye(nphi),q,format='csr') for q in coordinates]
    return dict(shell_states=states,shell_dimension=ns,dimension=nphi*ns,phi=phi,nphi=nphi,
        boson=boson,coupling=coupling,frozen=frozen,derivatives=derivatives,contacts=contacts,
        coordinates=field_q,em_translation=translation,em_lz=lz,
        field_translation=sparse.kron(sparse.eye(nphi),translation,format='csr'),
        q_drive=sparse.kron(sparse.eye(nphi),expm(1j*config['kick_strength']*coordinates[0]),format='csr'),
        phi_operator=sparse.kron(sparse.diags(phi),sparse.eye(ns),format='csr'),
        phi2_operator=sparse.kron(sparse.diags(phi**2),sparse.eye(ns),format='csr'),
        em_kinetic=em_kinetic)


def build(config):
    # Pure immutable helper constructs the eleven GLOBAL diabatic Slaters only;
    # no old consumer/producer is replayed and no old namespace is modified.
    base=collective_build(config);translations={n:local_translation(n) for n in [5,6,7]}
    blocks=[]
    for old in base['blocks']:
        vectors={}
        for n,key in [(6,'v0'),(7,'vp'),(5,'vm')]:
            v=old[key];t=translations[n]
            vectors[key]=np.column_stack([v,t@v,t@(t@v)])
        v0,vp,vm=vectors['v0'],vectors['vp'],vectors['vm']
        a6=annihilators(6,exterior_basis(6)[1],exterior_basis(5)[2]);a7=annihilators(7,exterior_basis(7)[1],exterior_basis(6)[2])
        blocks.append(dict(**vectors,operators=old['operators'],gram=v0.conj().T@v0,
            creation=np.array([vp.conj().T@(a.conj().T@v0) for a in a7]),
            removal=np.array([vm.conj().T@(a@v0) for a in a6])))
    count=3*base['gaussian_count'];grams=[b['gram'] for b in blocks];gindex=np.repeat(np.arange(count),6)
    raw_gram=multiply_except(grams,[]);raw_f=np.zeros((16,count,count),complex);local=[]
    for b,block in enumerate(blocks):
        v=block['v0'];f=np.array([v.conj().T@(op@v) for op in block['operators'][6]])
        local.append(f);raw_f+=f*multiply_except(grams,[b])[None]
    x,info=whiten(raw_gram,config['gram_tolerance'])
    neutral=dict(grams=[raw_gram],whiteners=[x],gram_info=[info],matrices=[np.array([x.conj().T@f@x for f in raw_f])])
    sectors=[]
    for sign in [1,-1]:
        sec=dict(grams=[],whiteners=[],gram_info=[],matrices=[])
        for b,block in enumerate(blocks):
            n,key=(7,'vp') if sign==1 else (5,'vm');v=block[key];g=v.conj().T@v
            outside=multiply_except(grams,[b])[gindex[:,None],gindex[None,:]];gram=g*outside
            matrices=np.array([v.conj().T@(op@v) for op in block['operators'][n]])*outside[None]
            for r in range(9):
                if r!=b:matrices+=local[r][:,gindex[:,None],gindex[None,:]]*g[None]*multiply_except(grams,[b,r])[gindex[:,None],gindex[None,:]][None]
            x,info=whiten(gram,config['gram_tolerance']);sec['grams'].append(gram);sec['whiteners'].append(x);sec['gram_info'].append(info)
            sec['matrices'].append(np.array([x.conj().T@f@x for f in matrices]))
        sectors.append(sec)
    field=shell_fields(config)
    model=dict(geo=base['geo'],transforms=base['transforms'],momenta=base['momenta'],blocks=blocks,grams=grams,
        local_components=base['local_components'],offset=base['offset'],neutral=neutral,positive=sectors[0],negative=sectors[1],
        gaussian_count=count,gindex=gindex,field=field,local_translations=translations,
        orbit_names=[f'{name}_Tz{l}' for l in range(3) for name in base['generator_names']])
    # Translation is formed from the physical local Fock permutation, not inferred
    # from energy degeneracy or a guessed coefficient permutation.
    tg=[b['v0'].conj().T@(translations[6]@b['v0']) for b in blocks]
    raw_t=multiply_except(tg,[]);neutral['matter_translation']=[neutral['whiteners'][0].conj().T@raw_t@neutral['whiteners'][0]]
    for sign,sec in zip([1,-1],sectors):
        sec['matter_translation']=[]
        for b,block in enumerate(blocks):
            n,key=(7,'vp') if sign==1 else (5,'vm');v=block[key]
            raw=(v.conj().T@(translations[n]@v))*multiply_except(tg,[b])[gindex[:,None],gindex[None,:]]
            x=sec['whiteners'][b];sec['matter_translation'].append(x.conj().T@raw@x)
    for sec in [neutral]+sectors:install_operator(sec,model)
    return model


def apply_terms(sec,terms,state):
    result=np.zeros_like(state,dtype=complex)
    for b,n in enumerate(sec['sizes']):
        start,stop=sec['offsets'][b:b+2];a=state[start:stop].reshape(n,-1);out=np.zeros_like(a,dtype=complex)
        for f,k in zip(sec['matrices'][b],terms):
            if k.nnz:out+=(k@(f@a).T).T
        result[start:stop]=out.ravel()
    return result


def install_operator(sec,model,frozen=False):
    nb=model['field']['dimension'];sizes=[len(f[0]) for f in sec['matrices']];offsets=np.cumsum([0]+[n*nb for n in sizes]);dimension=int(offsets[-1])
    sec.update(sizes=sizes,offsets=offsets)
    terms=model['field']['frozen' if frozen else 'coupling'];boson=model['field']['boson']
    def action(vector):
        vector=np.asarray(vector).reshape(-1);out=apply_terms(sec,terms,vector)
        for b,n in enumerate(sizes):
            start,stop=offsets[b:b+2];a=vector[start:stop].reshape(n,nb)
            out[start:stop]+=((boson@a.T).T-model['offset']*a).ravel()
        return out
    def matmat(matrix):return np.column_stack([action(v) for v in matrix.T])
    sec['h']=LinearOperator((dimension,dimension),matvec=action,rmatvec=action,matmat=matmat,rmatmat=matmat,dtype=complex)
    sec['trace']=sum(n*boson.diagonal().sum()-model['offset']*n*nb+sum(np.trace(f)*k.diagonal().sum() for f,k in zip(fs,terms)) for n,fs in zip(sizes,sec['matrices']))
    return sec['h']


def source_state(model,sign,matrices):
    neutral=model['neutral'];a=neutral['ground'].reshape(neutral['sizes'][0],model['field']['dimension'])
    return np.concatenate([(m@a).ravel() for m in matrices])


def translate(model,sec,state,matter_power=0,field_power=0):
    result=np.empty_like(state);tf=model['field']['field_translation']
    for b,n in enumerate(sec['sizes']):
        start,stop=sec['offsets'][b:b+2];a=state[start:stop].reshape(n,-1)
        for _ in range(matter_power%3):a=sec['matter_translation'][b]@a
        for _ in range(field_power%3):a=(tf@a.T).T
        result[start:stop]=a.ravel()
    return result


def momentum_probabilities(model,sec,state):
    characteristic=np.empty((3,3),complex)
    for a,b in product(range(3),repeat=2):characteristic[a,b]=np.vdot(state,translate(model,sec,state,a,b))
    joint=np.empty((3,3))
    for m,f in product(range(3),repeat=2):
        joint[m,f]=np.real(sum(np.exp(2j*np.pi*(m*a+f*b)/3)*characteristic[a,b] for a,b in product(range(3),repeat=2))/9)
    total=np.array([sum(joint[m,f] for m,f in product(range(3),repeat=2) if (m+f)%3==r) for r in range(3)])
    return dict(joint_matter_field=joint.tolist(),matter=joint.sum(axis=1).tolist(),field=joint.sum(axis=0).tolist(),total=total.tolist(),
        meaning='Exact commuting C3 translation projectors, momenta2pi*r/3 modulo reciprocal lattice; oscillator basis occupation is NOT photon number.')


def symmetry_data(model,sec,state):
    tm=sec['matter_translation'];tf=model['field']['em_translation'];h=sec['h']
    translated=translate(model,sec,state,1,1)
    comm=h@translated-translate(model,sec,h@state,1,1)
    return dict(matter_unitarity=max(float(np.linalg.norm(t.conj().T@t-np.eye(len(t)))) for t in tm),
        matter_order3=max(float(np.linalg.norm(t@t@t-np.eye(len(t)))) for t in tm),
        field_unitarity=float(np.linalg.norm(tf.conj().T@tf-np.eye(len(tf)))),field_order3=float(np.linalg.norm(tf@tf@tf-np.eye(len(tf)))),
        joint_H_commutator_on_state=float(np.linalg.norm(comm)),relative_joint_H_commutator=float(np.linalg.norm(comm)/max(np.linalg.norm(h@state),1e-30)))
