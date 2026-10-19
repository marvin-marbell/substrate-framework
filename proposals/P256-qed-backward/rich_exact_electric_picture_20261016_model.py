"""SAME frozen48 microscopic H in exact unbounded electric moving bases.
No old evolved state, cut free propagator, new restoring force or field deletion.
"""
from collections import OrderedDict
import numpy as np
import rich_spatial_frame_photon_20261016_model as frozen
import rich_spatial_frame_photon_20261016_fields as primitive
import rich_exact_electric_picture_20261016_fields as exact

DEFAULTS=dict(frozen.DEFAULTS)


class Model:
    def __init__(self,number,parameters=None):
        self.fixed=f=frozen.Model(number,DEFAULTS if parameters is None else parameters)
        self.parameters=f.parameters;self.shape=f.shape;self.dimension=f.dimension
        self.basis=f.basis;self.nfield=f.nfield;self.matter=f.matter;self.R=f.R
        self.nf=f.nf;self.ng=f.ng;self.nr=f.nr;self.nm=f.nm
        self.cache=OrderedDict();self.zeros=np.zeros(8);self.eye=np.eye(self.nfield)
        self.exterior=primitive.basis(self.parameters['field_total_cut']+1)
        self.images=exact.primitive_images(self.exterior,self.basis,0.)
        Kvalues=[];Cvalues=[]
        L=[a.toarray() for a in self.R['L']]
        for conn in f.connQ:
            M=frozen.mult(self.R,conn)
            Kvalues.append(np.array([sum(L[a]@M[:,lam,a,:]+M[:,lam,a,:]@L[a] for a in range(3)) for lam in range(8)]))
            C=np.einsum('tla,tma->tlm',conn,conn,optimize=True)
            Cvalues.append(frozen.mult(self.R,C))
        self.K=np.asarray(Kvalues);self.C=np.asarray(Cvalues)

    def impulse(self,axis=None,strength=.23):
        value=np.zeros(8)
        if axis is not None:value[axis]=strength
        return value

    def snapshot(self,time,impulse=None):
        eps=self.zeros if impulse is None else np.asarray(impulse);key=(float(time),tuple(eps))
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        inertia=self.parameters['cavity_inertia'];tau=time/inertia
        image=exact.primitive_images(self.exterior,self.basis,time,inertia,eps)
        b=np.asarray(self.fixed.q)+tau*(np.asarray(self.fixed.pi)+eps[:,None,None]*self.eye)
        # C is projected full Lambda_l Lambda_m, NEVER products of projected Lambda.
        quadratic=np.einsum('filmj,lmab->fiajb',self.C,image['bb'],optimize=True)
        linear=np.einsum('flij,lab->fiajb',self.K,b,optimize=True)
        rot=(quadratic-linear).reshape(self.nf,self.nm*self.nfield,self.nm*self.nfield)
        rot+=np.kron(self.R['casimir'].toarray(),self.eye)[None,:,:]
        rot/=2*self.parameters['material_inertia']
        hops=[]
        for edge in range(12):
            phi=self.fixed.hops[3*edge][3]
            D=exact.displacement(self.basis,self.basis,-phi,time,inertia,eps)
            for ell in range(3):
                hop,g,_,_,coeff=self.fixed.hops[3*edge+ell]
                rf=frozen.mult(self.R,coeff[:,None,None]*D).transpose(0,1,3,2).reshape(self.nm*self.nfield,-1)
                hops.append((hop,g,rf))
        value=dict(time=float(time),impulse=eps.copy(),b=b,rot=rot,hops=hops)
        self.cache[key]=value
        while len(self.cache)>2:self.cache.popitem(last=False)
        return value

    def hopping(self,state,snapshot,axis=None,second=None):
        x=np.asarray(state,dtype=np.complex128).reshape(self.shape);out=np.zeros_like(x)
        for term,(hop,g,rf) in enumerate(snapshot['hops']):
            if axis is not None:
                _,_,_,phi,coeff=self.fixed.hops[term]
                factor=-1j*phi[:,axis]
                if second is not None:factor*=(-1j*phi[:,second])
                D=exact.displacement(self.basis,self.basis,-phi,snapshot['time'],self.parameters['cavity_inertia'],snapshot['impulse'])
                rf=frozen.mult(self.R,(coeff*factor)[:,None,None]*D).transpose(0,1,3,2).reshape(self.nm*self.nfield,-1)
            for reverse in (False,True):
                matrix=rf.conj().T if reverse else rf
                rows,cols,values=(hop.col,hop.row,hop.data.conj()) if reverse else (hop.row,hop.col,hop.data)
                y=(x[cols].reshape(-1,self.nm*self.nfield)@matrix.T).reshape(len(cols),self.ng,self.nr,self.nm,self.nfield)
                y=np.einsum('ij,fjrma->firma',g,y,optimize=True)
                for k,(row,value) in enumerate(zip(rows,values)):out[row]+=value*y[k]
        return out.reshape(-1)

    def apply(self,state,time,impulse=None,hopping=True,coupled=True):
        """H_I^W(t), NOT physical H energy and NOT removal of physical H0."""
        x=np.asarray(state,dtype=np.complex128).reshape(self.shape)
        out=(self.fixed.base@x.reshape(-1,self.nfield)).reshape(self.shape)
        if coupled:
            snap=self.snapshot(time,impulse)
            out+=np.einsum('fgrj,fij->fgri',x.reshape(self.nf,self.ng,self.nr,-1),snap['rot'],optimize=True).reshape(self.shape)
        else:
            out+=np.einsum('...ja,ij->...ia',x,self.R['casimir'].toarray(),optimize=True)/(2*self.parameters['material_inertia'])
        if hopping:out+=self.hopping(state,self.snapshot(time,impulse)).reshape(self.shape)
        return out.reshape(-1)

    def field(self,state,axis,time=0.,impulse=None,kind='p'):
        eps=self.zeros if impulse is None else np.asarray(impulse)
        p=self.fixed.pi[axis]+eps[axis]*self.eye
        operator=p if kind=='p' else self.fixed.q[axis]+time/self.parameters['cavity_inertia']*p
        return (np.asarray(state).reshape(self.shape)@operator.T).reshape(-1)

    def free_energy(self,state,impulse=None):
        eps=self.zeros if impulse is None else np.asarray(impulse)
        pe=self.images['p']+eps[:,None,None]*self.images['embedding']
        p2=np.einsum('lei,lej->ij',pe.conj(),pe,optimize=True)
        return (np.asarray(state).reshape(self.shape)@p2.T/(2*self.parameters['cavity_inertia'])).reshape(-1)

    def physical_energy(self,state,time,impulse=None,hopping=True,coupled=True):
        return self.apply(state,time,impulse,hopping,coupled)+self.free_energy(state,impulse)

    def vertex(self,state,axis,time,impulse=None,second=None,hopping=True,coupled=True):
        snap=self.snapshot(time,impulse);out=np.zeros(self.shape,dtype=np.complex128)
        if coupled:
            if second is None:
                blocks=2*np.einsum('fimj,mab->fiajb',self.C[:,:,axis,:,:],snap['b'],optimize=True)
                blocks=blocks.reshape(self.nf,self.nm*self.nfield,-1)-np.asarray([np.kron(k[axis],self.eye) for k in self.K])
            else:blocks=2*np.einsum('fij,ab->fiajb',self.C[:,:,axis,second,:],self.eye,optimize=True).reshape(self.nf,self.nm*self.nfield,-1)
            blocks/=2*self.parameters['material_inertia']
            x=np.asarray(state).reshape(self.nf,self.ng,self.nr,-1)
            out+=np.einsum('fgrj,fij->fgri',x,blocks,optimize=True).reshape(self.shape)
        if hopping:out+=self.hopping(state,snap,axis,second).reshape(self.shape)
        return out.reshape(-1)

    def preparation(self):return self.fixed.preparation()

    def source(self,reference,state,component,create,time=0.,impulse=None):
        """Physical Wilson source at ABSOLUTE time in the same protocol bases."""
        if time==0.:return self.fixed.source(reference.fixed,state,component,create)
        eps=self.zeros if impulse is None else impulse
        tg=tuple(sorted(set(x[0] for x in self.matter.g['labels'])))
        rg=tuple(sorted(set(x[0] for x in reference.matter.g['labels'])))
        gs=frozen.source_matrices(tg,rg,create)
        out=np.zeros(self.shape,dtype=np.complex128);x=np.asarray(state).reshape(reference.shape)
        for site in range(8):
            car=self.matter.sector.source(reference.matter.sector,site,create).tocoo()
            D=exact.displacement(self.basis,reference.basis,(-1 if create else 1)*self.fixed.F[:,site],time,self.parameters['cavity_inertia'],eps)
            rf=frozen.mult(self.R,D).transpose(0,1,3,2).reshape(self.nm*self.nfield,reference.nm*reference.nfield)
            y=(x.reshape(-1,reference.nm*reference.nfield)@rf.T).reshape(reference.nf,reference.ng,self.nr,self.nm,self.nfield)
            y=np.einsum('ij,fjrma->firma',gs[component,site],y,optimize=True)
            for row,col,value in zip(car.row,car.col,car.data):out[row]+=value*y[col]
        return out.reshape(-1)

    def overlap(self,left,right,left_impulse,right_impulse):
        matrix=exact.protocol_overlap(self.basis,self.basis,left_impulse,right_impulse)
        return np.vdot(left,(np.asarray(right).reshape(self.shape)@matrix.T).reshape(-1))
