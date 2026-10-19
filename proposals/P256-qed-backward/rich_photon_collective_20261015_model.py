"""Fresh larger-ring all-virtual nonlinear CI with quantum Phi and two-band field.
No supplied magnetic term: literal Peierls exponentials projected before field cut.
"""
from itertools import product, combinations
from math import factorial
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm
from scipy.special import eval_genlaguerre
from scipy.sparse.linalg import LinearOperator
from rich_mobile_packet_20261015_model import shell_states
from rich_common_colour_lepton_20261015_model import common_root

def geometry(c):
    length=c['transverse'];nz=5;volume=length*length*nz;nc=4*volume;sea_count=nc//2
    cliff=common_root()['cliff'];cells=np.array(list(product(range(length),range(length),range(nz))));lookup={tuple(v):i for i,v in enumerate(cells)}
    mass=np.kron(np.eye(volume),cliff[0]);components=[3*c['wilson']*mass,mass];profiles=[np.zeros(8),np.zeros(8)];labels=['onsite','mass'];links=[]
    for axis in range(3):
        for level in (range(nz) if axis<2 else [0]):
            forward=np.zeros((nc,nc),complex)
            for site,coord in enumerate(cells):
                if axis<2 and coord[2]!=level:continue
                target=coord.copy();target[axis]=(target[axis]+1)%([length,length,nz][axis]);end=lookup[tuple(target)]
                hop=-.5j*c['velocity']*cliff[axis+1]-.5*c['wilson']*cliff[0]
                forward[4*site:4*site+4,4*end:4*end+4]+=hop;links.append((site,end,axis,level,hop))
            profile=np.zeros(8)
            if axis<2:
                for band in (1,2):profile[4*(band-1)+2*axis:4*(band-1)+2*axis+2]=c['charge']*np.array([np.cos(2*np.pi*band*level/nz),np.sin(2*np.pi*band*level/nz)])
            components.extend([forward+forward.conj().T,1j*(forward-forward.conj().T)]);profiles.extend([profile,profile]);labels.extend([f'hop{axis}_{level}_cos',f'hop{axis}_{level}_sin'])
    reference=components[0]+c['coupling']*c['phi_center']*mass+sum(components[2::2]);orbitals=[];energies=[];momenta=[];blocks=[]
    for kx,ky,kz in product(range(length),range(length),range(nz)):
        wave=np.vstack([np.eye(4)*np.exp(2j*np.pi*(kx*x/length+ky*y/length+kz*z/nz))/np.sqrt(volume) for x,y,z in cells]);ev,u=eigh(wave.conj().T@reference@wave)
        for j in range(4):orbitals.append(wave@u[:,j]);energies.append(ev[j]);momenta.append((kx,ky,kz));blocks.append((kx,ky))
    order=np.argsort(np.array(energies)>=0,kind='stable');u=np.array(orbitals).T[:,order];energies=np.array(energies)[order];momenta=np.array(momenta)[order];blocks=np.array(blocks)[order]
    transformed=np.array([u.conj().T@h@u for h in components]);transformed[abs(transformed)<2e-12]=0
    return dict(cells=cells,cliff=cliff,components=np.array(components),transformed=transformed,profiles=np.array(profiles),labels=labels,reference=reference,orbitals=u,energies=energies,momenta=momenta,blocks=blocks,sea_count=sea_count,sea=(1<<sea_count)-1,offset=float(energies[:sea_count].sum()),links=links,volume=volume,nc=nc)

def family(geo):
    sea=geo['sea'];ns=geo['sea_count'];masks={sea};blocks=sorted(set(map(tuple,geo['blocks'])))
    for block in blocks:
        ids=np.flatnonzero(np.all(geo['blocks']==block,axis=1));occ=[int(i) for i in ids if i<ns];empty=[int(i) for i in ids if i>=ns]
        for i,j in product(empty,occ):masks.add(sea^(1<<i)^(1<<j))
        if block==(0,0):
            for (i,k),(j,l) in product(combinations(empty,2),combinations(occ,2)):masks.add(sea^(1<<i)^(1<<k)^(1<<j)^(1<<l))
    return sorted(masks)

def onebody(masks,h):
    nc=len(h);look={m:i for i,m in enumerate(masks)};rows=[];cols=[];vals=[];diagonal=np.diag(h)
    targets=[[(int(i),h[i,j]) for i in np.flatnonzero(abs(h[:,j])>1e-12) if i!=j] for j in range(nc)]
    for column,mask in enumerate(masks):
        occ=[j for j in range(nc) if mask&(1<<j)];value=sum(diagonal[j] for j in occ)
        if abs(value)>1e-12:rows.append(column);cols.append(column);vals.append(value)
        for j in occ:
            first=mask^(1<<j);sign=-1 if (mask&((1<<j)-1)).bit_count()%2 else 1
            for i,element in targets[j]:
                if first&(1<<i):continue
                destination=look.get(first|(1<<i))
                if destination is not None:
                    parity=-1 if (first&((1<<i)-1)).bit_count()%2 else 1;rows.append(destination);cols.append(column);vals.append(sign*parity*element)
    return sparse.csr_matrix((vals,(rows,cols)),shape=(len(masks),len(masks)))

def field(c,geo):
    cut=c['shell_cutoff'];states=shell_states(cut);n=len(states);look={s:i for i,s in enumerate(states)};width=c['shell_width'];inertia=geo['volume']/2;a=[]
    for axis in range(8):
        op=np.zeros((n,n),complex)
        for j,s in enumerate(states):
            if s[axis]:
                t=list(s);t[axis]-=1;op[look[tuple(t)],j]=np.sqrt(s[axis])
        a.append(op)
    q=np.array([width*(v+v.conj().T)/np.sqrt(2) for v in a]);p=np.array([-1j*(v-v.conj().T)/(np.sqrt(2)*width) for v in a]);p2=np.array([(np.diag([2*s[k]+1 for s in states])-v@v-v.conj().T@v.conj().T)/(2*width**2) for k,v in enumerate(a)])
    kinetic=p2.sum(axis=0)/(2*inertia);phi=c['phi_center']+2*c['phi_width']*(np.arange(5)-2)/5;pk=2*np.pi*np.fft.fftfreq(5,d=2*c['phi_width']/5);ft=np.exp(2j*np.pi*np.outer(np.arange(5),np.arange(5))/5)/np.sqrt(5);pp=ft@np.diag(pk)@ft.conj().T
    phi_h=pp@pp/(2*geo['volume']*c['scalar_inertia'])+np.diag(geo['volume']*c['scalar_stiffness']*(phi-c['phi_center'])**2/2)
    terms=[np.eye(5*n),np.kron(np.diag(c['coupling']*phi),np.eye(n))];off=[terms[0],terms[1]]
    for k in range(2,len(geo['components']),2):
        displacement=np.ones((n,n),complex)
        for axis,coefficient in enumerate(geo['profiles'][k]):
            x=coefficient*width/np.sqrt(2);element=np.zeros((cut+1,cut+1),complex)
            for nn,mm in product(range(cut+1),repeat=2):
                lo,hi=min(nn,mm),max(nn,mm);element[nn,mm]=np.exp(-x*x/2)*np.sqrt(factorial(lo)/factorial(hi))*(1j*x)**(hi-lo)*eval_genlaguerre(lo,hi-lo,x*x)
            idx=np.array(states)[:,axis];displacement*=element[idx[:,None],idx[None,:]]
        co=(displacement+displacement.conj().T)/2;si=(displacement-displacement.conj().T)/(2j);terms.extend([np.kron(np.eye(5),co),np.kron(np.eye(5),si)]);off.extend([np.eye(5*n),np.zeros((5*n,5*n))])
    angular=np.zeros((n,n),complex)
    for band,pol in product(range(2),repeat=2):
        cc,ss=4*band+2*pol,4*band+2*pol+1;angular+=(band+1)*1j*(a[ss].conj().T@a[cc]-a[cc].conj().T@a[ss])
    ell,rot=eigh(angular)
    return dict(states=states,q=q,p=p,p2=p2,kinetic=kinetic,phi=phi,phi_h=phi_h,boson=np.kron(phi_h,np.eye(n))+np.kron(np.eye(5),kinetic),terms=terms,off=off,inertia=inertia,angular=angular,translation_values=np.rint(ell).astype(int)%5,translation_vectors=rot)

def install(matrices,terms,boson,offset):
    n=matrices[0].shape[0];nb=len(boson);boson=sparse.csr_matrix(boson);terms=[sparse.csr_matrix(t) for t in terms]
    def apply(vector):
        x=np.asarray(vector).reshape(n,nb);result=(boson@x.T).T-offset*x
        for matter,term in zip(matrices,terms):
            if term.nnz:result+=(term@(matter@x).T).T
        return result.ravel()
    return LinearOperator((n*nb,n*nb),matvec=apply,rmatvec=apply,dtype=complex)

def field_action(state,op,n):
    return (op@state.reshape(-1,n).T).T.ravel()

def expectation(state,operator):return float(np.vdot(state,operator@state).real)

def lanczos(operator,state,steps,dt=None):
    norm=np.linalg.norm(state);v=state/norm;vectors=[];alpha=[];beta=[];previous=np.zeros_like(v);last=0.
    for iteration in range(steps):
        vectors.append(v);w=operator@v-last*previous;value=float(np.vdot(v,w).real);w-=value*v
        # Full reorthogonalization; final endpoint residual retained, not a global bound.
        for _ in range(2):
            for basis in vectors:w-=np.vdot(basis,w)*basis
        following=float(np.linalg.norm(w));alpha.append(value)
        if following<1e-12 or iteration==steps-1:break
        beta.append(following);previous=v;v=w/following;last=following
    tri=np.diag(alpha)+np.diag(beta,1)+np.diag(beta,-1);energies,rotation=eigh(tri);weights=norm**2*abs(rotation[0])**2
    if dt is None:return dict(energies=energies,weights=weights,last_beta=following,dimension=len(alpha),alpha=alpha,beta=beta)
    coefficients=rotation@(np.exp(-1j*dt*energies)*rotation[0].conj());output=np.zeros_like(state)
    for basis,coefficient in zip(vectors,coefficients):output+=norm*coefficient*basis
    return output,dict(dimension=len(alpha),endpoint_residual=float(norm*following*abs(coefficients[-1])),norm=float(np.linalg.norm(output)))
