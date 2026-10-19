"""Collective nonorthogonal Gaussian fork of the immutable spatial-sea law.
Exact twelve-mode block exterior algebra avoids inverse-overlap singularities.
"""
from __future__ import annotations
from itertools import combinations, product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import LinearOperator, eigsh
from rich_spatial_quantum_sea_20261015_model import geometry, fields


def exterior_basis(n):
    indices=np.array(list(combinations(range(12),n)),int)
    masks=[sum(1<<int(i) for i in row) for row in indices]
    return indices,masks,{m:i for i,m in enumerate(masks)}


def annihilators(n,source,target):
    matrices=[]
    for orbital in range(12):
        rows=[];cols=[];data=[]
        for j,mask in enumerate(source):
            if mask & (1<<orbital):
                rows.append(target[mask^(1<<orbital)]);cols.append(j)
                data.append((-1)**((mask&((1<<orbital)-1)).bit_count()))
        matrices.append(sparse.csr_matrix((data,(rows,cols)),shape=(len(target),len(source))))
    return matrices


def second_quantized(h,masks,lookup):
    rows=[];cols=[];data=[]
    for j,mask in enumerate(masks):
        for b in range(12):
            if not mask&(1<<b):continue
            first=mask^(1<<b);sign=(-1)**((mask&((1<<b)-1)).bit_count())
            for a in range(12):
                if first&(1<<a) or abs(h[a,b])<1e-13:continue
                final=first|(1<<a);parity=(-1)**((first&((1<<a)-1)).bit_count())
                rows.append(lookup[final]);cols.append(j);data.append(sign*parity*h[a,b])
    return sparse.csr_matrix((data,(rows,cols)),shape=(len(masks),)*2)


def determinant_vectors(orbitals,indices):
    return np.linalg.det(np.asarray(orbitals)[:,indices,:]).T


def multiply_except(grams,excluded):
    result=np.ones_like(grams[0])
    for block,g in enumerate(grams):
        if block not in excluded:result*=g
    return result


def whiten(gram,tolerance):
    values,vectors=eigh((gram+gram.conj().T)/2)
    cutoff=max(float(values[-1])*tolerance,1e-13)
    keep=values>cutoff
    x=vectors[:,keep]/np.sqrt(values[keep])
    return x,dict(raw_dimension=len(values),retained_dimension=int(np.count_nonzero(keep)),
                  eigenvalues=values.tolist(),cutoff=cutoff,discarded=int(np.count_nonzero(~keep)),
                  whitened_gram_defect=float(np.linalg.norm(x.conj().T@gram@x-np.eye(np.count_nonzero(keep)))))


def build(config):
    geo=geometry(config);field=fields(config);cells=geo['cells']
    transforms=[];momenta=[];local_components=[];reference=[];occupied=[];empty=[]
    onsite=3*config['wilson']*geo['components'][0]
    for kx,ky in product(range(3),repeat=2):
        w=np.zeros((108,12),complex)
        for site,(x,y,z) in enumerate(cells):
            w[4*site:4*site+4,4*z:4*z+4]=np.eye(4)*np.exp(2j*np.pi*(kx*x+ky*y)/3)/3
        transforms.append(w);momenta.append((kx,ky))
        components=np.array([w.conj().T@h@w for h in np.concatenate([onsite[None],geo['components']])])
        local_components.append(components)
        ref=w.conj().T@geo['reference']@w
        e,u=eigh(ref);reference.append(e)
        if np.count_nonzero(e<0)!=6:raise ValueError('Six occupied modes required in each true spatial block')
        occupied.append(u[:,:6]);empty.append(u[:,6:])
    local_components=np.array(local_components)
    coefficients=np.vstack([np.ones(len(field['coordinates'])),field['coefficients']])
    center=int(np.argmin(np.sum((field['coordinates']-np.array([config['phi_center'],0,0,0,0]))**2,axis=1)))
    # Five collective occupied-empty generators: actual M and all four physical currents.
    operators=[local_components[:,1]]
    for derivative in field['derivative']:
        operators.append(-np.einsum('c,bcij->bij',np.r_[0.,derivative[:,center]],local_components))
    generators=[];norms=[]
    for op in operators:
        blocks=[]
        for u,h in zip(occupied,op):
            p=u@u.conj().T
            a=(np.eye(12)-p)@h@p
            blocks.append(a-a.conj().T)
        norm=np.sqrt(sum(np.linalg.norm(a)**2/2 for a in blocks))
        norms.append(float(norm));generators.append(np.array(blocks)/norm)
    gaussians=[[u for u in occupied]];complements=[[v for v in empty]];names=['reference_sea']
    for j,k in enumerate(generators):
        for direction in [-1,1]:
            rotations=[expm(direction*config['collective_step']*a) for a in k]
            gaussians.append([r@u for r,u in zip(rotations,occupied)])
            complements.append([r@v for r,v in zip(rotations,empty)])
            names.append(f'{["mass","Jxc","Jxs","Jyc","Jys"][j]}_{direction:+d}')
    count=len(gaussians)
    fock={n:exterior_basis(n) for n in [5,6,7]}
    blocks=[]
    for b in range(9):
        v0=determinant_vectors([g[b] for g in gaussians],fock[6][0])
        plus=[];minus=[]
        for g,v in zip(gaussians,complements):
            for j in range(6):
                plus.append(np.column_stack([v[b][:,j],g[b]]))
                minus.append(np.delete(g[b],j,axis=1))
        vp=determinant_vectors(plus,fock[7][0]);vm=determinant_vectors(minus,fock[5][0])
        local_operators={n:[second_quantized(h,fock[n][1],fock[n][2]) for h in local_components[b]] for n in [5,6,7]}
        a6=annihilators(6,fock[6][1],fock[5][2]);a7=annihilators(7,fock[7][1],fock[6][2])
        creation=np.array([vp.conj().T@(a.conj().T@v0) for a in a7])
        removal=np.array([vm.conj().T@(a@v0) for a in a6])
        blocks.append(dict(v0=v0,vp=vp,vm=vm,operators=local_operators,
            gram=v0.conj().T@v0,creation=creation,removal=removal))
    grams=[b['gram'] for b in blocks];neutral_gram=multiply_except(grams,[])
    neutral_f=np.zeros((len(coefficients),count,count),complex)
    neutral_local=[]
    for b,block in enumerate(blocks):
        local=np.array([block['v0'].conj().T@(op@block['v0']) for op in block['operators'][6]])
        neutral_local.append(local);neutral_f+=local*multiply_except(grams,[b])[None]
    x0,info0=whiten(neutral_gram,config['gram_tolerance'])
    neutral=dict(grams=[neutral_gram],whiteners=[x0],gram_info=[info0],
                 matrices=[np.array([x0.conj().T@f@x0 for f in neutral_f])])
    charged=[];gindex=np.repeat(np.arange(count),6)
    for sign in [1,-1]:
        sector=dict(grams=[],whiteners=[],gram_info=[],matrices=[])
        for b,block in enumerate(blocks):
            v=block['vp' if sign==1 else 'vm'];n=7 if sign==1 else 5
            local_gram=v.conj().T@v
            outside=multiply_except(grams,[b])[gindex[:,None],gindex[None,:]]
            gram=local_gram*outside
            matrices=np.array([v.conj().T@(op@v) for op in block['operators'][n]])*outside[None]
            for r in range(9):
                if r==b:continue
                factor=multiply_except(grams,[b,r])[gindex[:,None],gindex[None,:]]*local_gram
                matrices+=neutral_local[r][:,gindex[:,None],gindex[None,:]]*factor[None]
            x,info=whiten(gram,config['gram_tolerance'])
            sector['grams'].append(gram);sector['whiteners'].append(x);sector['gram_info'].append(info)
            sector['matrices'].append(np.array([x.conj().T@f@x for f in matrices]))
        charged.append(sector)
    offset=float(np.sum(np.array(reference)[:,:6]))
    model=dict(geo=geo,field=field,transforms=transforms,momenta=momenta,blocks=blocks,grams=grams,
               coefficients=coefficients,local_components=local_components,neutral_local=neutral_local,
               offset=offset,neutral=neutral,positive=charged[0],negative=charged[1],
               generator_names=names,generator_norms=norms,gaussian_count=count,gindex=gindex,center=center)
    for sec in [neutral]+charged:install_operator(sec,model)
    return model


def install_operator(sec,model,coefficients=None):
    coefficients=model['coefficients'] if coefficients is None else coefficients
    nb=coefficients.shape[1];sizes=[len(x[0]) for x in sec['matrices']]
    offsets=np.cumsum([0]+[n*nb for n in sizes]);dimension=int(offsets[-1]);boson=model['field']['boson']
    def action(vector):
        vector=np.asarray(vector).reshape(-1);out=np.empty(dimension,complex)
        for b,(n,f) in enumerate(zip(sizes,sec['matrices'])):
            a=vector[offsets[b]:offsets[b+1]].reshape(n,nb)
            result=(boson@a.T).T-model['offset']*a
            for matrix,coefficient in zip(f,coefficients):result+=(matrix@a)*coefficient
            out[offsets[b]:offsets[b+1]]=result.ravel()
        return out
    def matmat(matrix):return np.column_stack([action(v) for v in matrix.T])
    h=LinearOperator((dimension,dimension),matvec=action,rmatvec=action,matmat=matmat,rmatmat=matmat,dtype=complex)
    trace=sum(n*boson.diagonal().sum()-model['offset']*n*nb+sum(np.trace(f)*np.sum(c) for f,c in zip(fs,coefficients)) for n,fs in zip(sizes,sec['matrices']))
    sec.update(h=h,sizes=sizes,offsets=offsets,trace=trace,coefficients=coefficients)
    return h


def solve(sec,tolerance):
    h=sec['h'];e,v=eigsh(h,k=1,which='SA',tol=tolerance,maxiter=5000,v0=np.ones(h.shape[0],complex)/np.sqrt(h.shape[0]))
    sec.update(energy=float(e[0]),ground=v[:,0],eigensolver_residual=float(np.linalg.norm(h@v[:,0]-e[0]*v[:,0])))


def packet(model,config):
    x=model['geo']['cells'].astype(float);dx=(x-np.array(config['packet_center'])+1.5)%3-1.5
    f=np.zeros((27,4),complex)
    f[:,0]=np.exp(-np.sum(dx*dx,axis=1)/(2*config['packet_width']**2)+1j*x@np.array(config['packet_momentum']))
    f[:,2]=.3j*f[:,0];f=f.ravel();f/=np.linalg.norm(f)
    return [w.conj().T@f for w in model['transforms']]


def source_matrices(model,sign,source_packet):
    sec=model['positive' if sign==1 else 'negative'];x0=model['neutral']['whiteners'][0];out=[]
    for b,block in enumerate(model['blocks']):
        raw=np.einsum('i,ijk->jk',source_packet[b] if sign==1 else source_packet[b].conj(),block['creation' if sign==1 else 'removal'])
        raw*=multiply_except(model['grams'],[b])[model['gindex']]
        out.append(sec['whiteners'][b].conj().T@raw@x0)
    return out


def source_state(model,sign,matrices):
    neutral=model['neutral'];nb=len(model['field']['coordinates']);a=neutral['ground'].reshape(neutral['sizes'][0],nb)
    return np.concatenate([(m@a).ravel() for m in matrices])


def dense_body_operator(model,local_matrices):
    """Global N=55 density, including inter-kx/ky charge coherences."""
    sec=model['positive'];g=model['gindex'];count=model['gaussian_count'];sizes=sec['sizes'];bound=np.cumsum([0]+sizes)
    result=np.zeros((bound[-1],bound[-1]),complex)
    for b in range(9):
        v=model['blocks'][b]['vp'];localg=v.conj().T@v
        raw=np.zeros_like(localg)
        for r in range(9):
            if r==b:
                f=second_quantized(local_matrices[r,r],exterior_basis(7)[1],exterior_basis(7)[2])
                raw+=(v.conj().T@(f@v))*multiply_except(model['grams'],[b])[g[:,None],g[None,:]]
            else:
                v0=model['blocks'][r]['v0'];basis=exterior_basis(6)
                f=second_quantized(local_matrices[r,r],basis[1],basis[2]);a=v0.conj().T@(f@v0)
                raw+=a[g[:,None],g[None,:]]*localg*multiply_except(model['grams'],[b,r])[g[:,None],g[None,:]]
        x=sec['whiteners'][b];result[bound[b]:bound[b+1],bound[b]:bound[b+1]]=x.conj().T@raw@x
        for r in range(9):
            if r==b:continue
            raw=np.zeros((6*count,6*count),complex)
            c1=model['blocks'][b]['creation'];c2=model['blocks'][r]['creation']
            for i,j in zip(*np.where(abs(local_matrices[b,r])>1e-13)):
                raw+=local_matrices[b,r,i,j]*c1[i][:,g]*c2[j][:,g].conj().T
            raw*=multiply_except(model['grams'],[b,r])[g[:,None],g[None,:]]
            result[bound[b]:bound[b+1],bound[r]:bound[r+1]]=x.conj().T@raw@sec['whiteners'][r]
    return (result+result.conj().T)/2


def receiver_matrices(model):
    results=[]
    for site,(x,y,z) in enumerate(model['geo']['cells']):
        local=np.zeros((9,9,12,12),complex)
        for b,(kx,ky) in enumerate(model['momenta']):
            for r,(lx,ly) in enumerate(model['momenta']):
                phase=np.exp(2j*np.pi*((lx-kx)*x+(ly-ky)*y)/3)/9
                local[b,r,4*z:4*z+4,4*z:4*z+4]=phase*np.eye(4)
        results.append(dense_body_operator(model,local))
    return results


def fermionic_leakage(model):
    sec=model['neutral'];nb=len(model['field']['coordinates']);white=sec['ground'].reshape(sec['sizes'][0],nb)
    raw=sec['whiteners'][0]@white;count=model['gaussian_count'];coeff=model['coefficients']
    actions=[np.array([op@b['v0'] for op in b['operators'][6]]) for b in model['blocks']]
    fullnorm=0.
    for q in range(nb):
        local=[];squares=[]
        for b,block in enumerate(model['blocks']):
            a=np.einsum('c,cij->ij',coeff[:,q],actions[b])
            local.append(block['v0'].conj().T@a);squares.append(a.conj().T@a)
        h2=np.zeros((count,count),complex)
        for b in range(9):
            h2+=squares[b]*multiply_except(model['grams'],[b])
            for r in range(9):
                if r!=b:h2+=local[b]*local[r]*multiply_except(model['grams'],[b,r])
        fullnorm+=float(np.real(np.vdot(raw[:,q],h2@raw[:,q])))
    projected=np.zeros_like(white)
    for f,c in zip(sec['matrices'][0],coeff):projected+=(f@white)*c
    retained=float(np.linalg.norm(projected)**2)
    return dict(absolute_H_projection_residual=float(np.sqrt(max(fullnorm-retained,0.))),
        original_fermion_H_action_norm2=fullnorm,retained_fermion_H_action_norm2=retained,
        method='Exact local exterior-algebra H^2 plus cross-block products, no overlap inverse, no all-global-Fock enumeration; same field grid',
        meaning='Norm((1-P)H psi); boson kinetic stays in Gaussian tensor span. Not a gap bound. Includes discarded Gram directions.')
