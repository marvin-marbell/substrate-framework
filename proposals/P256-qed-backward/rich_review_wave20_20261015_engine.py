"""Full owning spans; independent tensor action, CAR checks and chronological DAG.
Pure frozen physical basis/source helpers only; no producer consumer is imported.
"""
from typing import Any
from itertools import permutations,product
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,expm_multiply
from rich_spatial_connected_20261015_model import build,canonical_sources
from rich_spatial_translation_20261015_model import shell_fields
from rich_spatial_orthogonal_20261015_tensor import cp_train
from rich_spatial_collective_20261015_model import exterior_basis,annihilators,second_quantized


class OwningH(LinearOperator):
    def __init__(self,body: Any,field: Any,offset: float):
        self.body=body;self.field=field;self.offset=offset;self.n=body.shape[1];self.nb=field['dimension'];self.factors=[(f,k) for f,k in zip(body,field['coupling']) if k.nnz];self.bare=field['boson'];self.tr=self.n*self.bare.diagonal().sum()-offset*self.n*self.nb+sum(np.trace(f)*k.diagonal().sum() for f,k in self.factors)
        super().__init__(dtype=np.dtype(complex),shape=(self.n*self.nb,)*2)
    def action(self,value: Any,adjoint: bool=False)->Any:
        value=np.asarray(value);one=value.ndim==1;x=value.reshape(self.n,self.nb,-1);count=x.shape[-1];b=self.bare.conj().T if adjoint else self.bare;out=(b@x.transpose(1,0,2).reshape(self.nb,-1)).reshape(self.nb,self.n,count).transpose(1,0,2)-self.offset*x
        for f,k in self.factors:
            if adjoint:f=f.conj().T;k=k.conj().T
            left=(f@x.reshape(self.n,-1)).reshape(self.n,self.nb,count);out+=(k@left.transpose(1,0,2).reshape(self.nb,-1)).reshape(self.nb,self.n,count).transpose(1,0,2)
        result=out.reshape(self.shape[0],count);return result[:,0] if one else result
    def _matvec(self,v: Any)->Any:return self.action(v)
    def _matmat(self,v: Any)->Any:return self.action(v)
    def _rmatvec(self,v: Any)->Any:return self.action(v,True)
    def _rmatmat(self,v: Any)->Any:return self.action(v,True)


def norm2(value: Any)->float:return float(np.vdot(value,value).real)


def field_act(engine: Any,name: str,matrix: Any,value: Any)->Any:
    value=np.asarray(value);one=value.ndim==1;a=value.reshape(engine.n[name],engine.nb,-1);out=(matrix@a.transpose(1,0,2).reshape(engine.nb,-1)).reshape(engine.nb,engine.n[name],-1).transpose(1,0,2).reshape(value.shape)
    return out.ravel() if one else out


def mixed_train(bra: Any,ket: Any,operators: Any)->Any:
    environment=np.ones((1,1),complex)
    for cb,ck,site in zip(bra['cores'],ket['cores'],bra['order']):
        weighted=np.einsum('ab,bpc->apc',environment,ck,optimize=True)
        if site in operators:weighted=np.einsum('pq,aqc->apc',operators[site],weighted,optimize=True)
        environment=cb.reshape(-1,cb.shape[-1]).conj().T@weighted.reshape(-1,ck.shape[-1])
    return environment


def independent_source_projection(model: Any)->Any:
    site=model['selected_site'];neutral=model['sectors']['neutral']['train'];coordinates=[entry[6]['coordinates'] for entry in model['local']];result={};checks={}
    for name,number in [('positive',7),('negative',5)]:
        train=model['sectors'][name]['train'];ordered=cp_train(coordinates,train['order'],model['config_tensor_cut']);rotation=ordered['R']@neutral['inverse'];u0=model['local'][site][6]['frame'];ut=model['local'][site][number]['frame'];ops=annihilators(7,exterior_basis(7)[1],exterior_basis(6)[2]) if number==7 else annihilators(6,exterior_basis(6)[1],exterior_basis(5)[2]);matrices=[]
        for kz,spinor in product(range(3),range(4)):
            f=np.zeros(12,complex);f[spinor::4]=np.exp(2j*np.pi*kz*np.arange(3)/3)/np.sqrt(3);raw=sum(value*(a.conj().T if number==7 else a) for value,a in zip(f if number==7 else f.conj(),ops));local=ut.conj().T@(raw@u0);matrices.append(mixed_train(train,ordered,{site:local})@rotation)
        result[name]=np.array(matrices);checks[name]=dict(independent_mixed_TT_vs_owning_CP_source_Frobenius=float(np.linalg.norm(result[name]-model['sources'][name])),neutral_order_rotation_isometry_error=float(np.linalg.norm(rotation.conj().T@rotation-np.eye(27))))
    return checks


class Engine:
    def __init__(self,config: Any,law: Any,reconstruction: Any):
        self.model=build(config,law,reconstruction);self.model['config_tensor_cut']=config['tensor_singular_relative_cut'];canonical_sources(self.model);self.source=self.model['sources'];self.n={name:sec['n'] for name,sec in self.model['sectors'].items()};self.nb=self.model['field']['dimension'];self.law=law;self.own_field=self.model['field'];self.fields={'coupled':self.own_field,'EM_off_factorized':shell_fields(dict(law,charge=0.))};self.H={arm:{name:OwningH(sec['matrices'],field,self.model['raw']['offset']) for name,sec in self.model['sectors'].items()} for arm,field in self.fields.items()};self.occupations=self.model['occupations'];self.projection_check=independent_source_projection(self.model)
    def set_reference(self,arm: str,omega: Any)->Any:
        self.arm=arm;self.omega=omega;self.energy=float(np.vdot(omega,self.H[arm]['neutral']@omega).real);self.means=np.array([np.vdot(omega,field_act(self,'neutral',q,omega)).real for q in self.own_field['coordinates']]);self.Q=[q-float(mu)*sparse.eye(self.nb,format='csr') for q,mu in zip(self.own_field['coordinates'],self.means)]
        return dict(energy=self.energy,H_state_residual=float(np.linalg.norm(self.H[arm]['neutral']@omega-self.energy*omega)),field_means=self.means.tolist(),reference_norm2=norm2(omega))
    def propagate(self,name: str,vectors: Any,t: float,adjoint: bool=False)->Any:
        vectors=np.asarray(vectors);one=vectors.ndim==1
        if one:vectors=vectors[:,None]
        if t==0:return vectors[:,0].copy() if one else vectors.copy()
        h=self.H[self.arm][name];shift=LinearOperator(h.shape,matvec=lambda v:h@v-self.energy*v,matmat=lambda v:h@v-self.energy*v,rmatvec=lambda v:h.H@v-self.energy*v,rmatmat=lambda v:h.H@v-self.energy*v,dtype=complex);coefficient=(1j if adjoint else -1j)*t;operator=coefficient*(shift.H if adjoint else shift);trace=coefficient*(np.conj(h.tr) if adjoint else h.tr)-coefficient*self.energy*h.shape[0];out=np.empty_like(vectors,complex)
        for start in range(0,vectors.shape[1],16):out[:,start:start+16]=expm_multiply(operator,vectors[:,start:start+16],traceA=trace)
        return out[:,0] if one else out
    def act(self,name: str,op: int,state: Any)->Any:
        if op<4:return name,field_act(self,name,self.Q[op],state)
        create=op>=16;mode=op-(16 if create else 4)
        if name=='neutral':target='positive' if create else 'negative';matrix=self.source[target][mode]
        elif name=='positive' and not create:target='neutral';matrix=self.source[name][mode].conj().T
        elif name=='negative' and create:target='neutral';matrix=self.source[name][mode].conj().T
        else:raise ValueError('Source string left actual54/55/53 sectors')
        shape=np.asarray(state).shape;value=matrix@state.reshape(self.n[name],-1);return target,value.reshape((self.n[target]*self.nb,)+shape[1:])
    def full_action_norm2(self,name: str,op: int,state: Any)->float:
        if op<4:
            q=self.rectangular_Q()[op];a=state.reshape(self.n[name],self.own_field['nphi'],-1);extended=np.einsum('es,nps->npe',q,a,optimize=True);padded=np.zeros_like(extended);padded[:,:,:a.shape[-1]]=a;return norm2(extended-self.means[op]*padded)
        mode=op-(16 if op>=16 else 4);a=state.reshape(self.n[name],-1);occupied=float(np.vdot(a,self.occupations[name][mode]@a).real);return norm2(state)-occupied if op>=16 else occupied
    def rectangular_Q(self)->Any:
        if hasattr(self,'rectQ'):return self.rectQ
        states=self.own_field['shell_states'];cut=max(map(sum,states));extended=sorted([s for s in product(range(cut+2),repeat=4) if sum(s)<=cut+1],key=lambda s:(sum(s),s));lookup={s:i for i,s in enumerate(extended)};self.rectQ=[]
        for axis in range(4):
            q=np.zeros((len(extended),len(states)),complex)
            for column,state in enumerate(states):
                for delta in [-1,1]:
                    target=list(state);target[axis]+=delta
                    if target[axis]>=0:q[lookup[tuple(target)],column]+=self.law['shell_width']*np.sqrt((state[axis]+(delta==1))/2)
            self.rectQ.append(q)
        return self.rectQ


def dagger(op: int)->int:return op if op<4 else op+12 if op<16 else op-12


def request_dag()->Any:
    orders=list(permutations(range(4)));requests=[];pairs={'ket':set(),'bra':set()}
    for oi,order in enumerate(orders):
        for kz,alpha,beta,a,b in product(range(3),range(4),range(4),range(4),range(4)):
            physical=[4+4*kz+alpha,a,b,16+beta];ket=(physical[order[0]],physical[order[1]]);bra=(dagger(physical[order[3]]),dagger(physical[order[2]]));pairs['ket'].add(ket);pairs['bra'].add(bra);requests.append(((oi,kz,alpha,beta,a,b),ket,bra))
    return orders,requests,{name:sorted(value) for name,value in pairs.items()}


def fourier(raw: Any)->Any:
    incoming=np.zeros((2,2,4),complex);outgoing=np.zeros_like(incoming)
    for pol,index in product(range(2),range(2)):
        eta=[-1,1][index];incoming[pol,index,2*pol]=outgoing[pol,index,2*pol]=1/np.sqrt(2);incoming[pol,index,2*pol+1]=1j*eta/np.sqrt(2);outgoing[pol,index,2*pol+1]=-1j*eta/np.sqrt(2)
    return np.einsum('...ab,lsa,mtb->...lmst',raw,outgoing,incoming,optimize=True)


def run_surface(engine: Any,tau: float,orders: Any,requests: Any,pairs: Any,singles: Any,first: Any,two: Any)->Any:
    propagated={direction:{name:engine.propagate(name,value,tau,direction=='bra') for name,value in singles.items()} for direction in ['ket','bra']};states={};indices={};audits={};catalog={}
    for direction in ['ket','bra']:
        columns={name:[] for name in engine.n};indices[direction]={};records={name:[] for name in engine.n};catalog[direction]={name:[] for name in engine.n}
        for op1,op2 in pairs[direction]:
            source,column=first[op1];before=propagated[direction][source][:,column];target,after=engine.act(source,op2,before);indices[direction][op1,op2]=(target,len(columns[target]));columns[target].append(after);full=engine.full_action_norm2(source,op2,before);records[target].append([norm2(before),full,norm2(after),full-norm2(after)]);catalog[direction][target].append([op1,op2])
        states[direction]={name:np.column_stack(value) for name,value in columns.items()};audits[direction]={name:np.array(value) for name,value in records.items()}
    evolved={name:engine.propagate(name,value,tau) for name,value in states['ket'].items()};grams={name:states['bra'][name].conj().T@value for name,value in evolved.items()};full=np.zeros((24,3,4,4,4,4),complex);disc=np.zeros_like(full)
    for index,ket,bra in requests:
        oi,kz,alpha,beta,a,b=index;kn,ki=indices['ket'][ket];bn,bi=indices['bra'][bra]
        if kn!=bn:raise ValueError('Unbalanced actual CAR sector')
        order=orders[oi];slot={label:order.index(label) for label in range(4)};sign=1 if slot[0]>slot[3] else -1;full[index]=sign*grams[kn][bi,ki];td=round(abs(slot[0]-slot[3])*tau,12);qd=round(abs(slot[1]-slot[2])*tau,12);fermion=two[td]['positive'][4*kz+alpha,beta] if sign==1 else -two[td]['negative'][beta,4*kz+alpha];boson=two[qd]['neutral'][a,b] if slot[1]>slot[2] else two[qd]['neutral'][b,a];disc[index]=fermion*boson
    return full,disc,audits,catalog
