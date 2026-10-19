"""Fresh CAR32, four locally sewn Peter-Weyl cone fields and spatial photons.
All RF multiplications are integrated BEFORE angular/Hermite compression.
No solved ancestor state, assigned fermion spin or Maxwell magnetic term.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
import hashlib
import pickle
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.special import eval_genlaguerre, factorial
from tenpy.linalg import np_conserved as npc
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOEnvironment, MPOGraph
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingModel
from rich_gauge_band_frame_20261016_model import geometry, angular
from rich_local_frame_spatial_carrier_20261016_angular import material, project
from rich_local_frame_spatial_carrier_20261016_operator import PhysicalOperatorSum
import rich_spatial_frame_photon_20261016_kernel as spatial

PREFIX='rich_local_frame_spatial_carrier_20261016_'
CENTRES=np.array([[0.,0.,0.],[1.3,0.,0.],[.2,1.2,0.],[.3,.4,1.1]])
EDGES=[(0,1),(1,2),(2,3),(3,0),(0,2),(1,3)]


def physical_element(bra, operator, ket):
    """MPOEnvironment excludes norm metadata; overlap includes it."""
    if isinstance(operator,PhysicalOperatorSum):return operator.matrix_element(bra,ket)
    return complex(np.conj(bra.norm)*ket.norm*MPOEnvironment(bra,operator,ket).full_contraction(0))


def field_basis(cap):
    return np.array([p for p in product(range(cap+1),repeat=8) if sum(p)<=cap],int)


def field_moments(basis):
    """P Q_a Q_b P and P Pi_a² P, with intermediate states NOT cut."""
    lookup={tuple(n):i for i,n in enumerate(basis)};d=len(basis)
    Q=np.zeros((8,d,d));QQ=np.zeros((8,8,d,d));PP=np.zeros((8,d,d))
    for col,n in enumerate(basis):
        for a in range(8):
            for direction in (-1,1):
                middle=n.copy();middle[a]+=direction
                if middle[a]<0:continue
                f=np.sqrt((n[a]+(direction==1))/2)
                row=lookup.get(tuple(middle))
                if row is not None:Q[a,row,col]+=f
                for b in range(8):
                    for dr in (-1,1):
                        last=middle.copy();last[b]+=dr
                        if last[b]<0:continue
                        row=lookup.get(tuple(last))
                        if row is None:continue
                        z=f*np.sqrt((middle[b]+(dr==1))/2)
                        QQ[b,a,row,col]+=z
                        if a==b:PP[a,row,col]+=-direction*dr*z
    return Q,QQ,PP


def displacement(basis, coefficients):
    """Exact Hermite matrix of exp(i c.Q), not exp(i c.PQP)."""
    c=np.asarray(coefficients);nq=len(c);d=len(basis)
    if d==9 and np.all(basis.sum(axis=1)<=1):
        vacuum=int(np.flatnonzero(basis.sum(axis=1)==0)[0])
        excited=np.flatnonzero(basis.sum(axis=1)==1)
        x=c[:,np.argmax(basis[excited],axis=1)]/np.sqrt(2)
        out=np.zeros((nq,d,d),complex);out[:,vacuum,vacuum]=1.
        out[:,vacuum,excited]=1j*x;out[:,excited,vacuum]=1j*x
        out[:,excited[:,None],excited[None,:]]=np.eye(8)-x[:,:,None]*x[:,None,:]
        return out*np.exp(-np.sum(c*c,axis=1)/4)[:,None,None]
    out=np.ones((nq,d,d),complex)
    for a in range(8):
        x=c[:,a]/np.sqrt(2)
        for i,n in enumerate(basis[:,a]):
            for j,m in enumerate(basis[:,a]):
                lo=min(n,m);hi=max(n,m)
                out[:,i,j]*=np.exp(-x*x/2)*np.sqrt(factorial(lo)/factorial(hi))*(1j*x)**(hi-lo)*eval_genlaguerre(lo,hi-lo,x*x)
    return out


class Model:
    def __init__(self, field_cap=1, gmax=2., material_max=2, hopping=.45, internal=.35, nangle=24, nbeta=16):
        self.parameters=dict(field_cap=field_cap,gmax=gmax,material_max=material_max,hopping=hopping,internal=internal,radial_mass=2.,material_inertia=6.,electric_inertia=3.,charge_shell_radius=.08,boundary_radius=20.,nangle=nangle,nbeta=nbeta)
        self.geo=geometry();self.modes=spatial.default_modes()
        # Translating spherical-shell profiles have this exact plane-wave form
        # factor; no extra internally rigid rotation/spin of each shell.
        self.modes[:,3:6]*=np.sinc(.08*np.linalg.norm(self.modes[:,:3],axis=1)/np.pi)[:,None]
        self.rfrecipes={}
        self.positions=(CENTRES[:,None,:]+.24*(self.geo['sites'][None,:,:]-.5)).reshape(32,3)
        self.ions=np.tile(self.geo['ions'],4)
        self.g=angular(tuple(np.arange(0,gmax+.01,.5)))
        self.material=material(material_max,nangle,nbeta)
        self.rnodes=np.array([.45,.9,1.35]);self.basis=field_basis(field_cap)
        self.Q,self.QQ,self.PP=field_moments(self.basis)
        self.ng=len(self.g['labels']);self.nR=len(self.material['labels']);self.nf=len(self.basis)
        self.RFdim=self.nR*self.nf
        # Independent finite-q cubature, SAME retained complete R multiplets.
        self.rotations=self.material['rotations']
        self.lab=np.einsum('qab,sb->qsa',self.rotations,self.positions)
        self.kernel=[spatial.geometry(x,self.modes) for x in self.lab]
        self.connection=np.array([x['connection'] for x in self.kernel])
        self.F=np.array([x['F'] for x in self.kernel])
        self.sites=[];self.ops=[];self.fsites=[];self.gsites=[];self.rsites=[]
        self.chinfo=npc.ChargeInfo([1,2,2,2,2],['N','sew0','sew1','sew2','sew3'])
        for v in range(4):
            self.rsites.append(self.add_site(np.zeros((3,5),int),{}))
            q=np.zeros((self.ng,5),int);q[:,v+1]=[round(2*x[0])%2 for x in self.g['labels']]
            self.gsites.append(self.add_site(q,{}))
            for s in range(8):
                q=np.zeros((2,5),int);q[:,0]=[0,1];q[:,v+1]=[0,1]
                self.fsites.append(self.add_site(q,{'N':np.diag([0.,1.]),'Qnet':np.diag([self.ions[8*v+s],self.ions[8*v+s]-1.]),'JW':np.diag([1.,-1.])}))
        self.rfsite=self.add_site(np.zeros((self.RFdim,5),int),{})
        self.lat=Lattice([1],self.sites,bc='open',bc_MPS='finite')
        self.terms=[];self.hop_terms=[];self.local_terms=[];self.pair_starts=set()
        self._build()
        self.graph_audit=self.graph_metadata(self.terms)
        self.H=self.mpo(self.terms)
        self.signature=hashlib.sha256(repr(self.parameters).encode()+self.positions.tobytes()+self.modes.tobytes()).hexdigest()

    def checkpoint(self,stage):
        self.stage=stage
        path=Path(__file__).resolve().parent/(PREFIX+'build_checkpoint.pkl')
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as f:pickle.dump(self,f,protocol=5)
        temporary.replace(path)
        print('CHECKPOINT',stage,flush=True)

    @classmethod
    def resume_build(cls,path):
        with Path(path).open('rb') as f:self=pickle.load(f)
        if self.stage=='local':
            self._build_D();self.checkpoint('D')
        first=int(self.stage[4:])+1 if self.stage.startswith('edge') else 0
        self._build_transport(first)
        self.graph_audit=self.graph_metadata(self.terms)
        self.H=self.mpo(self.terms)
        self.signature=hashlib.sha256(repr(self.parameters).encode()+self.positions.tobytes()+self.modes.tobytes()).hexdigest()
        return self

    def add_site(self,q,ops):
        leg=npc.LegCharge.from_qflat(self.chinfo,np.asarray(q,dtype=np.int64))
        i=len(self.sites);self.sites.append(Site(leg,[str(x) for x in range(len(q))],sort_charge=True,**ops));self.ops.append({});return i

    def register(self,i,A):
        A=np.asarray(A,complex).copy();A[abs(A)<2e-12]=0
        key=hashlib.sha256(A.tobytes()).hexdigest()
        if key not in self.ops[i]:
            name='X'+str(len(self.ops[i]));self.sites[i].add_op(name,A,need_JW=False,permute_dense=True);self.ops[i][key]=name
        return self.ops[i][key]

    def term(self,coefficient,factors):
        return coefficient,[(self.register(i,A),i) for i,A in sorted(factors.items())]

    def coupling_model(self,terms):
        model=CouplingModel(self.lat)
        for value,term in terms:
            if abs(value)>1e-14:model.add_local_term(value,[(op,[0,i]) for op,i in term])
        return model

    def graph(self,terms):
        model=self.coupling_model(terms)
        return MPOGraph.from_terms((model.all_onsite_terms(),model.all_coupling_terms(),model.exp_decaying_terms),self.lat.mps_sites(),self.lat.bc_MPS,unit_cell_width=self.lat.mps_unit_cell_width)

    def graph_sizes(self,terms):
        return [len(x) for x in self.graph(terms).states]

    def graph_metadata(self,terms):
        sizes=self.graph_sizes(terms);dims=[s.dim for s in self.sites]
        dense=[16*sizes[i]*sizes[i+1]*d*d for i,d in enumerate(dims)]
        return dict(graph_virtual_channels=sizes,physical_dimensions=dims,dense_complex128_storage_upper_bytes=dense,dense_upper_total_bytes=sum(dense),dense_upper_max_site_bytes=max(dense),actual_charge_block_allocation='Not allocated; upper bound ignores conserved-charge sparsity. No proved attribution of prior SIGKILL.',selected_representation='Exact lazy sum of bounded-channel MPO groups, never this full tensor.')

    def mpo_small(self,terms):
        """Build an MPO, including empty or cancelled sums on this lattice.

        A zero operator still needs the physical sites and conserved-charge
        legs; an empty coupling graph cannot infer those legs in TeNPy.
        """
        model=self.coupling_model(terms)
        onsite=model.all_onsite_terms();couplings=model.all_coupling_terms()
        onsite.remove_zeros();couplings.remove_zeros()
        if onsite.to_TermList().terms or couplings.to_TermList().terms:
            return model.calc_H_MPO()
        zero=self.coupling_model([(1.,[('Id',0)])]).calc_H_MPO()
        zero.set_W(0,zero.get_W(0)*0.)
        return zero

    def mpo(self,terms):
        terms=list(terms)
        return PhysicalOperatorSum(self,terms) if len(terms)>24 else self.mpo_small(terms)

    def project(self,values):
        """Values[q,field_bra,field_ket], actual Haar bra and full RF product."""
        return project(self.material,np.asarray(values))

    def rscalar(self,values):
        return self.project(np.asarray(values)[:,None,None]*np.eye(self.nf))

    def tag_rf(self,matrix,recipe):
        a=np.asarray(matrix,complex).copy();a[abs(a)<2e-12]=0
        self.rfrecipes[hashlib.sha256(a.tobytes()).hexdigest()]=recipe

    @lru_cache(maxsize=768)
    def phase(self,origin,target,axis=-1,ell=-1):
        phi=spatial.segment(self.lab[:,origin,:],self.lab[:,target,:],self.modes)[0]
        values=displacement(self.basis,-phi)
        if axis>=0:values*=self.rotations[:,ell,axis,None,None]
        out=self.project(values)
        self.tag_rf(out,('phase',origin,target,axis,ell,False))
        self.tag_rf(out.conj().T,('phase',origin,target,axis,ell,True))
        return out

    def fermions(self,operations):
        factors={}
        c=np.array([[0.,1.],[0.,0.]])
        for mode,create in operations:
            for before in range(mode):
                i=self.fsites[before];factors[i]=factors.get(i,np.eye(2))@np.diag([1.,-1.])
            i=self.fsites[mode];factors[i]=factors.get(i,np.eye(2))@(c.T if create else c)
        return factors

    def add_pair(self,value,factors,collection=None):
        target=self.terms if collection is None else collection
        if collection is None:self.pair_starts.add(len(self.terms))
        t=self.term(value,factors);u=self.term(np.conj(value),{i:a.conj().T for i,a in factors.items()})
        target.extend([t,u]);return t,u

    def _build(self):
        p=self.parameters;nr=3;h=.45;r=self.rnodes;mass=p['radial_mass'];I=p['material_inertia']
        radial=(2*np.eye(nr)-np.eye(nr,k=1)-np.eye(nr,k=-1))/(2*mass*h*h)+np.diag(1.5*(r-.9)**2+3/(8*mass*r*r))
        for v in range(4):
            self.terms.append(self.term(1.,{self.rsites[v]:radial}))
            self.terms.append(self.term(1.,{self.rsites[v]:np.diag(1/(2*mass*r*r)),self.gsites[v]:self.g['casimir'].toarray()}))
            for s in range(8):
                self.terms.append(self.term(.7*self.geo['beta'][s,s],{self.rsites[v]:np.diag(r),self.fsites[8*v+s]:np.diag([0.,1.])}))
        LR=[np.kron(a.toarray(),np.eye(self.nf)) for a in self.material['L']]
        field=np.kron(np.eye(self.nR),self.PP.sum(axis=0)/(2*p['electric_inertia']))
        self.terms.append(self.term(1.,{self.rfsite:sum(a@a for a in LR)/(2*I)+field}))
        # Nonoverlapping spherical charge shells, exact grounded-sphere Green
        # kernel (harmonic mean-value property), including rotating integer ions.
        B=p['boundary_radius'];a=.08
        def image(s,t):
            y=self.positions[t];rho=np.linalg.norm(y)
            return B/rho/np.linalg.norm(self.positions[s]-B*B*y/(rho*rho))
        for s in range(32):
            Qs=np.diag([self.ions[s],self.ions[s]-1.])
            self.terms.append(self.term(.5*(1/a-image(s,s)),{self.fsites[s]:Qs@Qs}))
            for t in range(s):
                Qt=np.diag([self.ions[t],self.ions[t]-1.]);d=np.linalg.norm(self.positions[s]-self.positions[t])
                self.terms.append(self.term(1/d-image(s,t),{self.fsites[s]:Qs,self.fsites[t]:Qt}))
        # Physical rotating ions AND electrons: raw {L,Lambda} and Lambda².
        for s in range(32):
            Qs=np.diag([self.ions[s],self.ions[s]-1.]);cross=np.zeros((self.RFdim,)*2,complex)
            for axis in range(3):
                lam=self.project(np.einsum('ql,lab->qab',self.connection[:,s,:,axis],self.Q))
                cross-= (LR[axis]@lam+lam@LR[axis])/(2*I)
            self.tag_rf(cross,('cross',s))
            self.terms.append(self.term(1.,{self.fsites[s]:Qs,self.rfsite:cross}))
            for t in range(s+1):
                Qt=np.diag([self.ions[t],self.ions[t]-1.])
                values=np.einsum('qla,qma,lmij->qij',self.connection[:,s],self.connection[:,t],self.QQ,optimize=True)
                op=self.project(values)/I
                self.tag_rf(op,('square',s,t))
                factors={self.rfsite:op,self.fsites[s]:Qs}
                if t==s:factors[self.fsites[s]]=Qs@Qs;factor=.5
                else:factors[self.fsites[t]]=Qt;factor=1.
                self.terms.append(self.term(factor,factors))
            if s%8==7:print('MECHANICAL_SITE',s,flush=True)
        # Intra-cell actual cube pi-flux paths, all8 modes, no band projection.
        for v,axis,ell,i in product(range(4),range(3),range(3),range(3)):
            gamma=self.geo['gamma'][i]
            for s,t in zip(*np.nonzero(np.triu(abs(gamma)>1e-12,1))):
                factors=self.fermions([(8*v+int(s),True),(8*v+int(t),False)])
                factors[self.gsites[v]]=self.g['O'][ell][i].toarray()
                factors[self.rfsite]=self.phase(8*v+int(t),8*v+int(s),axis,ell)
                self.add_pair(-p['internal']*gamma[s,t]/3,factors)
        print('LOCAL_TERMS',len(self.terms),flush=True)
        self.local_terms=list(self.terms)
        self.checkpoint('local')
        self._build_D();self.checkpoint('D')
        self._build_transport()

    def _build_D(self):
        # Geometry-derived S_a²=1/4: exact exponential, not a new spin law.
        spin=self.geo['spin'];points=self.g['points'];eye=np.eye(8)
        D=np.broadcast_to(eye,(len(points),8,8)).astype(complex).copy()
        for axis,angles in ((2,points[:,0]),(1,points[:,1]),(2,points[:,2])):
            rotation=np.cos(angles/2)[:,None,None]*eye-2j*np.sin(angles/2)[:,None,None]*spin[axis]
            D=np.einsum('qab,qbc->qac',D,rotation,optimize=True)
        f=self.g['F'];w=self.g['weights'];self.D={}
        for mu,s in product(range(8),repeat=2):
            if np.max(abs(D[:,mu,s]))>1e-10:self.D[mu,s]=f.conj().T@(w[:,None]*D[:,mu,s,None]*f)

    def _build_transport(self,first=0):
        p=self.parameters
        for edge,(v,u) in enumerate(EDGES):
            if edge<first:continue
            for mu in range(8):
                for s in range(8):
                    if (mu,s) not in self.D:continue
                    for t in range(8):
                        if (mu,t) not in self.D:continue
                        factors=self.fermions([(8*u+s,True),(8*v+t,False)])
                        factors[self.gsites[u]]=self.D[mu,s]
                        factors[self.gsites[v]]=self.D[mu,t].conj().T
                        factors[self.rfsite]=self.phase(8*v+t,8*u+s)
                        pair=self.add_pair(-p['hopping'],factors)
                        self.hop_terms.extend([(pair[0],edge,1),(pair[1],edge,-1)])
            print('INTERCELL_EDGE',edge,len(self.terms),flush=True)
            self.checkpoint('edge'+str(edge))

    def seed(self):
        labels=[]
        for v in range(4):labels += ['1','0']+[str(x) for x in self.ions[8*v:8*v+8]]
        labels+=['0']
        psi=MPS.from_product_state(self.sites,labels,dtype=complex,unit_cell_width=1)
        radial=np.exp(-2*(self.rnodes-.9)**2);radial/=np.linalg.norm(radial)
        for i in self.rsites:
            op=np.eye(3,dtype=complex);op[:,1]=radial
            psi.apply_local_op(i,self.register(i,op),unitary=False,renormalize=False)
        psi.canonical_form(renormalize=False);psi.norm=1.
        return psi

    def source(self,psi,cell=0,component=0,create=True,chi=12):
        coefficients=np.eye(8,dtype=complex)[component] if np.isscalar(component) else np.asarray(component,complex)
        terms=[]
        for s in range(8):
            matrix=sum((coefficients[mu]*(self.D[mu,s] if create else self.D[mu,s].conj().T) for mu in range(8) if (mu,s) in self.D),np.zeros((self.ng,)*2,complex))
            if np.linalg.norm(matrix)<1e-14:continue
            factors=self.fermions([(8*cell+s,create)])
            factors[self.gsites[cell]]=matrix
            # Transverse path from the common interior root; eliminated
            # longitudinal Dirac dressing joins that root to the boundary.
            rf=self.project(displacement(self.basis,(-1 if create else 1)*self.F[:,8*cell+s]))
            factors[self.rfsite]=rf;terms.append(self.term(1.,factors))
        out=psi.copy();self.mpo_small(terms).apply_naively(out)
        raw=float(out.overlap(out).real)
        if raw==0:
            out.norm=0.
            self.last_source_quality=dict(exact_regulated_source_norm2=0.,retained_source_norm2=0.,svd_eps=0.)
            return out,raw
        error=out.compress_svd({'chi_max':chi,'svd_min':1e-12})
        retained=float(out.overlap(out).real)
        self.last_source_quality=dict(exact_regulated_source_norm2=raw,retained_source_norm2=retained,svd_eps=float(error.eps),meaning='Tuple strength is exact finite-regulator raw Bpsi norm²; normalized returned MPS approximates compressed source. Restore sqrt(retained_source_norm2) for its unnormalized amplitudes, not sqrt(raw) if compression lost norm. Neither is a Born probability.')
        out.canonical_form(renormalize=False);out.norm/=np.sqrt(retained)
        return out,raw

    def observable(self,site,A):return self.mpo([self.term(1.,{site:A})])

    def current(self,edge):
        return self.mpo([(1j*direction*term[0],term[1]) for term,e,direction in self.hop_terms if e==edge])

    def contact(self,edge):
        return self.mpo([(-term[0],term[1]) for term,e,direction in self.hop_terms if e==edge])

    def lab_source(self,mode,other=None):
        """Physical -dH/da_mode; contact is +d²H/(da_mode da_other).

        Includes intra/intercell paths AND rotating-ion mechanical source.
        Derivatives act on the uncompressed multiplications, not cut Q/P.
        """
        cache={};terms=[]
        LR=[np.kron(x.toarray(),np.eye(self.nf)) for x in self.material['L']]
        I=self.parameters['material_inertia']
        for coefficient,factors in self.terms:
            rf=next((name for name,i in factors if i==self.rfsite),None)
            if rf is None:continue
            if rf not in cache:
                original_key=next((k for k,v in self.ops[self.rfsite].items() if v==rf),None)
                recipe=self.rfrecipes.get(original_key)
                if recipe is None:
                    cache[rf]=None;continue
                if recipe[0]=='phase':
                    _,s,t,axis,ell,dagger=recipe
                    phi=spatial.segment(self.lab[:,s],self.lab[:,t],self.modes)[0]
                    values=displacement(self.basis,-phi)*(-1j*phi[:,mode,None,None])
                    if other is not None:values*=(-1j*phi[:,other,None,None])
                    if axis>=0:values*=self.rotations[:,ell,axis,None,None]
                    op=self.project(values)
                    if dagger:op=op.conj().T
                elif recipe[0]=='cross':
                    if other is not None:
                        cache[rf]=None;continue
                    s=recipe[1];op=np.zeros((self.RFdim,)*2,complex)
                    for a in range(3):
                        lam=self.rscalar(self.connection[:,s,mode,a])
                        op-=(LR[a]@lam+lam@LR[a])/(2*I)
                else:
                    _,s,t=recipe
                    if other is None:
                        values=np.einsum('qa,qla,lij->qij',self.connection[:,s,mode],self.connection[:,t],self.Q)
                        values+=np.einsum('qla,qa,lij->qij',self.connection[:,s],self.connection[:,t,mode],self.Q)
                    else:
                        scalar=np.einsum('qa,qa->q',self.connection[:,s,mode],self.connection[:,t,other])
                        scalar+=np.einsum('qa,qa->q',self.connection[:,s,other],self.connection[:,t,mode])
                        values=scalar[:,None,None]*np.eye(self.nf)
                    op=self.project(values)/I
                cache[rf]=self.register(self.rfsite,op)
            if cache[rf] is not None:
                terms.append(((-1 if other is None else 1)*coefficient,[(cache[rf] if i==self.rfsite else name,i) for name,i in factors]))
        return self.mpo(terms)

    def measure(self,psi,energy=True):
        n=np.array([float(x.real) for x in psi.expectation_value('N',sites=self.fsites)]).reshape(4,8)
        left=[];r2=[];angular_boundary=[]
        for v in range(4):
            left.append([float(physical_element(psi,self.observable(self.gsites[v],a.toarray()),psi).real) for a in self.g['L']])
            r2.append(float(physical_element(psi,self.observable(self.rsites[v],np.diag(self.rnodes**2)),psi).real))
            boundary=np.diag([float(j>=self.parameters['gmax']-.5) for j,m,k in self.g['labels']])
            angular_boundary.append(float(physical_element(psi,self.observable(self.gsites[v],boundary),psi).real))
        fieldQ=[float(physical_element(psi,self.observable(self.rfsite,np.kron(np.eye(self.nR),q)),psi).real) for q in self.Q]
        result=dict(norm2=float(psi.overlap(psi).real),total_charge=float(self.ions.sum()-n.sum()),n_microsites=n.tolist(),n_cells=n.sum(axis=1).tolist(),local_LEFT=left,local_r2=r2,field_Q=fieldQ,local_top_two_j_shell_weights=angular_boundary,chi=max(psi.chi),charge_sewing=psi.get_total_charge(only_physical_legs=True).tolist())
        if energy:result['energy']=float(physical_element(psi,self.H,psi).real)
        return result
