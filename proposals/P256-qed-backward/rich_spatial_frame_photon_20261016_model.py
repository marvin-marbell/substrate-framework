"""Actual finite-q minimal-velocity global-frame law, exact plane-path Peierls.
Full CAR sectors, SU2 g and positive quantum r from frozen pure Band40 helpers.
Net physical Q=ions-fermions. No ancestor state/producer is executed.
"""
from functools import lru_cache
import numpy as np
from scipy import sparse as sp
import rich_gauge_band_frame_20261016_model as band
import rich_spatial_frame_photon_20261016_kernel as kernel
import rich_spatial_frame_photon_20261016_fields as fields
import rich_spatial_frame_photon_20261016_quadrature as quadrature
import rich_local_frame_spatial_carrier_20261016_angular as projection

DEFAULTS=dict(band.DEFAULTS,field_total_cut=1,cavity_inertia=2.,material_scale=1.,material_level=1)
DEFAULTS['wave_number_scale']=1.
DEFAULTS['radial_nodes']=[.3,.6,.9,1.2,1.5]


def rotations(points):
    a,b,c=np.asarray(points).T
    def rz(t):
        z=np.zeros_like(t);o=z+1;return np.array([[np.cos(t),-np.sin(t),z],[np.sin(t),np.cos(t),z],[z,z,o]]).transpose(2,0,1)
    def ry(t):
        z=np.zeros_like(t);o=z+1;return np.array([[np.cos(t),z,np.sin(t)],[z,o,z],[-np.sin(t),z,np.cos(t)]]).transpose(2,0,1)
    return rz(a)@ry(b)@rz(c)


@lru_cache(maxsize=8)
def source_matrices(target_js,reference_js,create):
    from scipy.linalg import expm
    common=band.angular(tuple(sorted(set(target_js+reference_js))))
    target=band.angular(target_js);reference=band.angular(reference_js)
    ti=[common['labels'].index(x) for x in target['labels']]
    ri=[common['labels'].index(x) for x in reference['labels']]
    spin=band.geometry()['spin']
    values=np.array([expm(-1j*a*spin[2])@expm(-1j*b*spin[1])@expm(-1j*c*spin[2]) for a,b,c in common['points']])
    if not create:values=values.conj()
    return np.einsum('ti,t,tuv,tj->uvij',common['F'][:,ti].conj(),common['weights'],values,common['F'][:,ri],optimize=True)


def mult(R,values):
    values=np.asarray(values);shape=values.shape[1:]
    if 'gaunt' not in R:
        return np.einsum('ti,t,t...,tj->i...j',R['F'].conj(),R['weights'],values,R['F'],optimize=True)
    n=len(R['labels'])
    coefficients=R['forward']@values.reshape(len(values),-1)
    out=(R['gaunt']@coefficients).reshape((n,n)+shape)
    return out.transpose((0,)+tuple(range(2,out.ndim))+(1,))


class Model:
    def __init__(self,number,parameters=None):
        self.parameters=p=dict(DEFAULTS if parameters is None else parameters)
        bp=dict(p);bp['angular_level']=p['angular_level']
        self.matter=hm=band.Model(number,bp)
        self.R=R=projection.material(p['material_level'])
        # Same H40 coefficients; R level can be independently raised.
        self.nf,self.ng,_,self.nr=hm.shape;self.nm=len(R['labels'])
        self.basis=fields.basis(p['field_total_cut']);self.nfield=len(self.basis)
        self.shape=(self.nf,self.ng,self.nr,self.nm,self.nfield);self.dimension=int(np.prod(self.shape))
        self.rot=rotations(R['points']);self.x=np.einsum('tab,sb->tsa',self.rot,(hm.geo['sites']-.5)*p['material_scale'])
        self.modes=kernel.default_modes();self.modes[:,:3]*=p.get('wave_number_scale',1.)
        self.geom=[kernel.geometry(x,self.modes) for x in self.x]
        self.F=np.array([g['F'] for g in self.geom]);self.conn=np.array([g['connection'] for g in self.geom])
        self.Q=-hm.sector.q
        self.q=[fields.primitive(self.basis,self.basis,a,'q').toarray() for a in range(8)]
        self.pi=[fields.primitive(self.basis,self.basis,a,'p').toarray() for a in range(8)]
        ext=fields.basis(p['field_total_cut']+1)
        self.qe=np.array([fields.primitive(ext,self.basis,a,'q').toarray() for a in range(8)])
        self.qq=np.einsum('lei,mej->lmij',self.qe.conj(),self.qe,optimize=True)
        pe=[fields.primitive(ext,self.basis,a,'p') for a in range(8)]
        self.p2=sum((a.conj().T@a).toarray() for a in pe)
        # Base excludes material rotational and hopping terms, rebuilt before cut.
        fi,gi,ri,mi=sp.eye(self.nf),sp.eye(self.ng),sp.eye(self.nr),sp.eye(self.nm)
        self.base=band.tensor(sp.diags(np.sum(hm.sector.electric**2,axis=1)/(2*p['electric_inertia'])),gi,ri,mi)
        hr=hm.Hradial[:self.nr,:self.nr]
        self.base+=band.tensor(fi,gi,hr,mi)
        self.base+=band.tensor(fi,hm.g['casimir'],sp.diags(1/(2*p['radial_mass']*hm.rnodes**2)),mi)
        self.base+=p['root_coupling']*band.tensor(hm.sector.bilinear(hm.geo['beta']),gi,sp.diags(hm.rnodes),mi)
        self.connQ=np.einsum('fs,tsla->ftla',self.Q,self.conn)
        self.rotblocks=self.rotational_blocks()
        self.hops=[]
        for axis in range(3):
            for source in range(8):
                target=source^(1<<(2-axis))
                if target<source:continue
                # Site array ordering has bit 2-axis associated with Cartesian axis.
                gamma=hm.geo['gamma'][axis][target,source]
                if not abs(gamma):continue
                phi=kernel.segment(self.x[:,source],self.x[:,target],self.modes)[0]
                disp=fields.displacement(self.basis,self.basis,-phi)
                one=np.zeros((8,8),complex);one[target,source]=gamma
                hop=hm.sector.bilinear(one).tocoo()
                for ell in range(3):
                    coeff=self.rot[:,ell,:]@(-2*np.cos(p['k']))
                    rf=mult(R,coeff[:,None,None]*disp).transpose(0,1,3,2).reshape(self.nm*self.nfield,-1)
                    self.hops.append((hop,hm.g['O'][ell][axis].toarray(),rf,phi,coeff))

    @lru_cache(maxsize=16)
    def rotational_blocks(self,source_axis=None,second_axis=None):
        R=self.R;nm=self.nm;nfld=self.nfield;blocks=[]
        ls=[a.toarray() for a in R['L']];eye=np.eye(nfld)
        for conn in self.connQ:
            if source_axis is None:
                block=np.kron(R['casimir'].toarray(),eye).astype(complex)
                for axis in range(3):
                    for lam in range(8):
                        m=mult(R,conn[:,lam,axis])
                        block-=np.kron(ls[axis]@m+m@ls[axis],self.q[lam])
                coefficients=np.einsum('tla,tma->tlm',conn,conn,optimize=True)
                quadratic=np.einsum('ilmj,lmab->iajb',mult(R,coefficients),self.qq,optimize=True).reshape(nm*nfld,nm*nfld)
            elif second_axis is None:
                lam=source_axis;block=np.zeros((nm*nfld,)*2,complex)
                for axis in range(3):
                    m=mult(R,conn[:,lam,axis]);block-=np.kron(ls[axis]@m+m@ls[axis],eye)
                cross=2*np.einsum('ta,tka->tk',conn[:,lam,:],conn,optimize=True)
                quadratic=np.einsum('ilj,lab->iajb',mult(R,cross),np.asarray(self.q),optimize=True).reshape(nm*nfld,nm*nfld)
            else:
                block=np.zeros((nm*nfld,)*2,complex)
                val=2*np.einsum('ta,ta->t',conn[:,source_axis,:],conn[:,second_axis,:])
                quadratic=np.kron(mult(R,val),eye)
            block=block+quadratic
            blocks.append(block/(2*self.parameters['material_inertia']))
        return np.asarray(blocks)

    @lru_cache(maxsize=1)
    def edge_profile(self,edge):
        phi=self.hops[3*edge][3]
        return fields.displacement(self.basis,self.basis,-phi)

    @lru_cache(maxsize=192)
    def hop_source_matrix(self,index,source_axis,second_axis):
        _,_,_,phi,coeff=self.hops[index]
        factor=-1j*phi[:,source_axis]
        if second_axis is not None:factor*=(-1j*phi[:,second_axis])
        disp=self.edge_profile(index//3)
        return mult(self.R,(coeff*factor)[:,None,None]*disp).transpose(0,1,3,2).reshape(self.nm*self.nfield,-1)

    def hop_action(self,state,source_axis=None,second_axis=None):
        x=np.asarray(state,dtype=complex).reshape(self.shape);out=np.zeros_like(x)
        for term,(hop,g,rf,phi,coeff) in enumerate(self.hops):
            if source_axis is not None:rf=self.hop_source_matrix(term,source_axis,second_axis)
            for reverse in (False,True):
                matrix=rf.conj().T if reverse else rf
                rows,cols,values=(hop.col,hop.row,hop.data.conj()) if reverse else (hop.row,hop.col,hop.data)
                y=(x[cols].reshape(-1,self.nm*self.nfield)@matrix.T).reshape(len(cols),self.ng,self.nr,self.nm,self.nfield)
                y=np.einsum('ij,fjrma->firma',g,y,optimize=True)
                for index,(row,value) in enumerate(zip(rows,values)):out[row]+=value*y[index]
        return out.reshape(-1)

    def apply(self,state,hopping=True,motion=True):
        x=np.asarray(state,dtype=complex).reshape(self.shape);nfld=self.nfield
        # base has ordering F,g,r,R, field last.
        out=(self.base@x.reshape(-1,nfld)).reshape(self.shape)
        out+=np.einsum('...i,ji->...j',x,self.p2,optimize=True)/(2*self.parameters['cavity_inertia'])
        if motion:out+=np.einsum('fgrj,fij->fgri',x.reshape(self.nf,self.ng,self.nr,-1),self.rotblocks,optimize=True).reshape(self.shape)
        else:
            out+=np.einsum('...ja,ij->...ia',x,self.R['casimir'].toarray(),optimize=True)/(2*self.parameters['material_inertia'])
        if hopping:out+=self.hop_action(state).reshape(self.shape)
        return out.reshape(-1)

    def vertex(self,state,axis,second=None,hopping=True,motion=True):
        out=np.zeros(self.shape,complex);x=np.asarray(state).reshape(self.nf,self.ng,self.nr,-1)
        if motion:out+=np.einsum('fgrj,fij->fgri',x,self.rotational_blocks(axis,second),optimize=True).reshape(self.shape)
        if hopping:out+=self.hop_action(state,axis,second).reshape(self.shape)
        return out.reshape(-1)

    def preparation(self):
        v=self.matter.preparation().reshape(self.matter.shape)
        if self.nm!=self.matter.shape[2]:
            w=np.zeros((self.nf,self.ng,self.nm,self.nr),complex);w[:,:,0,:]=v[:,:,0,:];v=w
        f=np.zeros(self.nfield);f[0]=1
        return np.kron(v.transpose(0,1,3,2).reshape(-1),f)

    def source(self,reference,state,component,create):
        out=np.zeros(self.shape,complex);x=state.reshape(reference.shape)
        tg=tuple(sorted(set(x[0] for x in self.matter.g['labels'])))
        rg=tuple(sorted(set(x[0] for x in reference.matter.g['labels'])))
        gs=source_matrices(tg,rg,create)
        for site in range(8):
            g=gs[component,site]
            car=self.matter.sector.source(reference.matter.sector,site,create).tocoo()
            coeff=(-1 if create else 1)*self.F[:,site,:]
            disp=fields.displacement(self.basis,reference.basis,coeff)
            rf=mult(self.R,disp).transpose(0,1,3,2).reshape(self.nm*self.nfield,reference.nm*reference.nfield)
            y=(x.reshape(-1,reference.nm*reference.nfield)@rf.T).reshape(reference.nf,reference.ng,self.nr,self.nm,self.nfield)
            y=np.einsum('ij,fjrma->firma',g,y,optimize=True)
            for row,col,value in zip(car.row,car.col,car.data):out[row]+=value*y[col]
        return out.reshape(-1)

    def field(self,state,axis,kind='p'):
        x=state.reshape(self.shape);a=self.pi[axis] if kind=='p' else self.q[axis]
        return (x@a.T).reshape(-1)
