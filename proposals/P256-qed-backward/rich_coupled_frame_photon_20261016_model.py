"""NEW noncompact lab-vector zero-mode/material law; pure declarations only.

Velocity: Hv=U Hm U†+Pi²/(2I), U=exp(i q.D), Xs=R(site_s-1/2).
Length: H=Hm+(Pi+D)²/(2I). No magnetic or q² restoring term.
All three vector modes retained; x,y are transverse to selected lab z.
"""
from itertools import product
from functools import lru_cache
import numpy as np
from scipy import sparse as sp
import rich_gauge_band_frame_20261016_model as band

DEFAULTS=dict(band.DEFAULTS,cavity_inertia=2.,field_total_cut=2,material_scale=1.)


def field_basis(cut):
    return [v for v in product(range(cut+1),repeat=3) if sum(v)<=cut]


def field_operator(target,reference,axis,kind):
    lookup={v:i for i,v in enumerate(target)};rows=[];cols=[];values=[]
    for col,v in enumerate(reference):
        for sign in [-1,1]:
            w=list(v);w[axis]+=sign
            if min(w)<0 or tuple(w) not in lookup:continue
            value=np.sqrt((v[axis]+(sign>0))/2)
            if kind=='p':value*=1j*sign
            rows.append(lookup[tuple(w)]);cols.append(col);values.append(value)
    return sp.csr_matrix((values,(rows,cols)),shape=(len(target),len(reference)))


@lru_cache(maxsize=4)
def material_multiplication(level):
    angular=band.angular(tuple(range(level+1)))
    return angular


class Model:
    def __init__(self,number,parameters=None):
        self.parameters=p=dict(DEFAULTS if parameters is None else parameters)
        self.matter=hm=band.Model(number,p)
        self.basis=field_basis(p['field_total_cut']);self.nfield=len(self.basis)
        self.shape=(hm.dimension,self.nfield);self.dimension=hm.dimension*self.nfield
        self.identity=sp.eye(hm.dimension,format='csr');self.fi=sp.eye(self.nfield,format='csr')
        self.q=[field_operator(self.basis,self.basis,a,'q') for a in range(3)]
        self.p=[field_operator(self.basis,self.basis,a,'p') for a in range(3)]
        self.body_dipole=hm.sector.q@((hm.geo['sites']-.5)*p['material_scale'])
        self.D=self.dipoles(hm.material)
        d2=np.sum(self.body_dipole**2,axis=1)
        self.D2=hm.lift_f(sp.diags(d2))
        # Full Pi² BEFORE projection: intermediate field shell is cut+2.
        exterior=field_basis(p['field_total_cut']+2)
        pe=[field_operator(exterior,self.basis,a,'p') for a in range(3)]
        self.p2=band.clean(sum(a.conj().T@a for a in pe))
        inertia=p['cavity_inertia']
        self.Hfield=band.clean(sp.kron(self.D2,self.fi)/(2*inertia)+sp.kron(self.identity,self.p2)/(2*inertia)+sum(sp.kron(self.D[a],self.p[a])/inertia for a in range(3)))
        self.Hmatter=sp.kron(hm.H,self.fi,format='csr')
        self.H=band.clean(self.Hmatter+self.Hfield)
        self.Jfield=[]
        qe=[field_operator(exterior,self.basis,a,'q') for a in range(3)]
        for a in range(3):
            b=(a+1)%3;c=(a+2)%3
            self.Jfield.append(band.clean(qe[b].conj().T@pe[c]-qe[c].conj().T@pe[b]))
        self.J=[band.clean(sp.kron(hm.J[a],self.fi)+sp.kron(self.identity,self.Jfield[a])) for a in range(3)]

    def dipoles(self,angular):
        hm=self.matter;nf,ng,_,nr=hm.shape
        return [band.clean(sum(band.tensor(sp.diags(self.body_dipole[:,b]),sp.eye(ng),angular['O'][a][b],sp.eye(nr)) for b in range(3))) for a in range(3)]

    def lift(self,operator):return sp.kron(operator,self.fi,format='csr')

    def electric(self,axis):
        return band.clean(sp.kron(self.identity,self.p[axis])+sp.kron(self.D[axis],self.fi))

    def preparation(self):
        f=np.zeros(self.nfield);f[self.basis.index((0,0,0))]=1
        return np.kron(self.matter.preparation(),f)

    def source_from(self,reference,component=0,create=True):
        # Root-radial Wilson material source in velocity gauge is U_N B U_4†.
        return sp.kron(self.matter.source_from(reference.matter,component,create),self.fi,format='csr')

    def exterior_image(self,state):
        """Exact field-exterior image of unbounded polynomial H on this packet."""
        ext=field_basis(self.parameters['field_total_cut']+2)
        embed=sp.csr_matrix(([1.]*len(self.basis),([ext.index(v) for v in self.basis],range(len(self.basis)))),shape=(len(ext),len(self.basis)))
        pp=[field_operator(ext,self.basis,a,'p') for a in range(3)]
        # Pi² image needs one extra shell for its intermediate image.
        middle=field_basis(self.parameters['field_total_cut']+1)
        p2=sp.csr_matrix(sum(field_operator(ext,middle,a,'p')@field_operator(middle,self.basis,a,'p') for a in range(3)))
        x=state.reshape(self.shape);inertia=self.parameters['cavity_inertia']
        out=((self.matter.H+self.D2/(2*inertia))@x)@embed.T
        out+=(x@p2.T)/(2*inertia)
        for a in range(3):out+=(self.D[a]@x)@pp[a].T/inertia
        projected=x@embed.T
        keep=[ext.index(v) for v in self.basis];outside=[i for i in range(len(ext)) if i not in keep]
        return dict(field_exterior_norm=float(np.linalg.norm(out[:,outside])),full_H_norm=float(np.linalg.norm(out)),field_shell_weight=float(sum(np.linalg.norm(x[:,i])**2 for i,v in enumerate(self.basis) if sum(v)==self.parameters['field_total_cut'])),embedding_norm=float(np.linalg.norm(projected)))

    def source_vertices(self):
        """P d_A Hvelocity P and P d²_A Hvelocity P, NOT projected products.

        D changes material l by at most one. R extension lmax+1 contains
        every intermediate contributing to P[D,[D,Hm]]P. g/r/Fock unchanged.
        This is a source-image extension, not a second evolved Hamiltonian.
        """
        hm=self.matter;level=self.parameters['angular_level'];angular=material_multiplication(level+1)
        nf,ng,nm,nr=hm.shape;ne=len(angular['labels']);di=self.dipoles(angular)
        index=[angular['labels'].index(v) for v in hm.material['labels']]
        em=sp.csr_matrix(([1.]*nm,(index,range(nm))),shape=(ne,nm))
        embedding=band.tensor(sp.eye(nf),sp.eye(ng),em,sp.eye(nr))
        fi,gi,ri=sp.eye(nf),sp.eye(ng),sp.eye(nr)
        # Hm extension reconstructed from the SAME declared coefficients.
        hr=band.tensor(fi,gi,angular['casimir']/(2*self.parameters['material_inertia']),ri)
        base=band.tensor(sp.diags(np.sum(hm.sector.electric**2,axis=1)/(2*self.parameters['electric_inertia'])),gi,sp.eye(ne),ri)
        base+=band.tensor(fi,gi,sp.eye(ne),hm.Hradial[:nr,:nr])
        base+=band.tensor(fi,hm.g['casimir'],sp.eye(ne),sp.diags(1/(2*self.parameters['radial_mass']*hm.rnodes**2)))
        base+=self.parameters['root_coupling']*band.tensor(hm.sector.bilinear(hm.geo['beta']),gi,sp.eye(ne),sp.diags(hm.rnodes))
        hopping=sp.csr_matrix(hr.shape,dtype=complex)
        for i,a,ell in product(range(3),repeat=3):
            hopping+=-2*np.cos(self.parameters['k'][a])*band.tensor(hm.sector.bilinear(hm.geo['gamma'][i]),hm.g['O'][ell][i],angular['O'][ell][a],ri)
        he=band.clean(base+hr+hopping)
        de=[d@embedding for d in di];himage=he@embedding
        vertices=[band.clean(1j*(de[a].conj().T@himage-himage.conj().T@de[a])) for a in range(3)]
        contact={}
        for a,b in [(0,0),(0,1),(1,0),(1,1),(2,2)]:
            # -D_a D_b H - H D_b D_a + D_a H D_b + D_b H D_a.
            dd=di[a]@de[b]
            contact[a,b]=band.clean(-dd.conj().T@himage-himage.conj().T@dd+de[a].conj().T@(he@de[b])+de[b].conj().T@(he@de[a]))
        return vertices,contact
