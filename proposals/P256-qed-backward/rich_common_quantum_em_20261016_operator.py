"""ONE new rich-matter + live two-transverse Cartan Hamiltonian.
Only fundamental finite-matrix/source-selected matter and explicit flux cuts.
No old solved states, Maxwell B term, oscillator frequency, or Dyson join.
"""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from rich_composite_geometry_20261015_factors import PhiHamiltonian
from rich_common_quantum_em_20261016_geometry import MATTER_EDGES

INTEGER_CHARGES=(2,0,-1,-3)


def resolve(matrix,counts):
    unique=list(dict.fromkeys(map(tuple,counts)));where={c:i for i,c in enumerate(unique)};count_ids=np.asarray([where[tuple(c)]for c in counts]);lookup=np.full((len(unique),len(unique)),-1,dtype=np.int16);keys=[(edge,category,direction)for edge in range(5)for category in range(4)for direction in(1,-1)]
    for source,occupied in enumerate(unique):
        for code,(edge,category,direction)in enumerate(keys):
            start,end=MATTER_EDGES[edge];changed=list(occupied);changed[4*start+category]-=direction;changed[4*end+category]+=direction;target=where.get(tuple(changed))
            if target is not None:lookup[target,source]=code
    coo=matrix.tocoo();codes=lookup[count_ids[coo.row],count_ids[coo.col]]
    if np.any(codes<0):raise ValueError('Non-hop coefficient in actual hopping matrix; no element discarded')
    result={}
    for code,key in enumerate(keys):
        keep=codes==code
        if np.any(keep):result[key]=sparse.coo_matrix((coo.data[keep],(coo.row[keep],coo.col[keep])),shape=matrix.shape).tocsr()
    return result


class ChargeOperator(LinearOperator):
    """Compact tensor action with direct signed flux shifts, never powers of PUP."""
    def __init__(self,n,grid,hphi,flux,groups,free=True,field_energy=None):
        self.n=n;self.grid=np.asarray(grid);self.hphi=np.asarray(hphi);self.flux=np.asarray(flux);self.groups=groups;self.free=free;self.field_energy=np.zeros(len(flux))if field_energy is None else np.asarray(field_energy);self.where={tuple(m):i for i,m in enumerate(flux)};self.actions=[]
        for shift,(c0,cphi)in groups.items():
            source=[];target=[]
            for i,m in enumerate(flux):
                j=self.where.get(tuple(m+np.asarray(shift)))
                if j is not None:source.append(i);target.append(j)
            self.actions.append((np.asarray(source,dtype=np.int32),np.asarray(target,dtype=np.int32),c0,cphi))
        super().__init__(dtype=np.dtype(complex),shape=(n*len(grid)*len(flux),)*2)
    def action(self,value):
        value=np.asarray(value,dtype=complex);columns=1 if value.ndim==1 else value.shape[1];x=value.reshape(self.n,len(self.grid),len(self.flux),columns)
        if self.free:
            result=np.einsum('ab,nbmk->namk',self.hphi,x,optimize=True);result+=x*self.field_energy[None,None,:,None]
        else:result=np.zeros_like(x)
        for source,target,c0,cphi in self.actions:
            selected=x[:,:,source,:].reshape(self.n,-1);y=(c0@selected).reshape(self.n,len(self.grid),len(source),columns)
            if cphi.nnz:y+=(cphi@selected).reshape(y.shape)*self.grid[None,:,None,None]
            result[:,:,target,:]+=y
        return result.reshape(value.shape)
    def _matvec(self,value):return self.action(value)
    def _matmat(self,value):return self.action(value)
    def _rmatvec(self,value):return self.action(value)
    def _rmatmat(self,value):return self.action(value)
    def trace(self):
        pair=self.groups.get((0,0));body=0j if pair is None else len(self.flux)*(len(self.grid)*pair[0].diagonal().sum()+self.grid.sum()*pair[1].diagonal().sum())
        return body+(self.n*(len(self.flux)*np.trace(self.hphi)+len(self.grid)*self.field_energy.sum())if self.free else 0)


class Model:
    def __init__(self,prefix):
        self.prefix=prefix;self.body=json.loads(Path(prefix+'_body.json').read_text());self.geometry=json.loads(Path(prefix+'_geometry.json').read_text());self.labels=self.body['labels'];self.n=len(self.labels);self.counts=np.asarray([x['counts']for x in self.labels],int);self.occupation=self.counts.reshape(self.n,4,4);self.integer_q=2*self.occupation[:,:,0]-self.occupation[:,:,2]-3*self.occupation[:,:,3];self.q=self.integer_q/3.;self.quark_q=(2*self.occupation[:,:,0]-self.occupation[:,:,2])/3.;self.white_q=-self.occupation[:,:,3].astype(float)
        self.flux=np.asarray(self.geometry['electric_fluxes'],int);self.field_energy=np.asarray(self.geometry['bare_field_energies']);self.profile=np.asarray(self.geometry['matter_profiles'],int);self.lp=np.asarray(self.geometry['laplacian_pseudoinverse']);self.d=np.asarray(self.geometry['incidence']);self.t=np.asarray(self.geometry['profile']);self.gti=np.linalg.inv(np.asarray(self.geometry['profile_gram'],float));self.longitudinal=self.integer_q@self.lp@self.d;self.long_energy=.5*np.einsum('na,ab,nb->n',self.integer_q,self.lp,self.integer_q)
        with np.load(prefix+'_scalar.npz')as z:self.hphi=z['H'];self.grid=z['grid']
        inverse=sparse.load_npz(prefix+'_inversehalf.npz');static=sparse.load_npz(prefix+'_static.npz');self.static=(inverse@static@inverse).tocsr()+sparse.diags(self.long_energy);self.static=(self.static+self.static.conj().T)/2
        vector=sparse.load_npz(prefix+'_vector.npz')+sparse.load_npz(prefix+'_chord_vector.npz')
        # Scalar Hamiltonian array is distinct from scalar hopping sparse archive.
        scalar_hop=sparse.load_npz(prefix+'_scalar_hop.npz');scalar_hop+=sparse.load_npz(prefix+'_chord_scalar.npz');vparts=resolve(vector,self.counts);sparts=resolve(scalar_hop,self.counts);zero=sparse.csr_matrix((self.n,self.n),dtype=complex);self.parts=[];adjoint_errors={};ward=0.
        for edge in range(5):
            for category in range(4):
                key=(edge,category,1);back=(edge,category,-1);a=vparts.get(key,zero);b=sparts.get(key,zero);adjoint_errors[str(key)]=[float(np.linalg.norm((a.conj().T-vparts.get(back,zero)).data)),float(np.linalg.norm((b.conj().T-sparts.get(back,zero)).data))];a=(inverse@a@inverse).tocsr();b=(inverse@b@inverse).tocsr();charge=INTEGER_CHARGES[category];shift=tuple(map(int,charge*self.profile[edge]));self.parts.append({'edge':edge,'category':category,'integer_charge':charge,'physical_charge':charge/3.,'shift':shift,'c0':a,'cphi':b})
                start,end=MATTER_EDGES[edge]
                for matrix in(a,b):
                    coo=matrix.tocoo()
                    for site in range(4):ward=max(ward,float(np.linalg.norm(coo.data*(self.q[coo.row,site]-self.q[coo.col,site]-(charge/3.)*((site==end)-(site==start))))))
        self.receipt={'body_dimension':self.n,'Phi_dimension':len(self.grid),'transverse_flux_dimension':len(self.flux),'full_dimension':self.n*len(self.grid)*len(self.flux),'forward_plus_exact_adjoint_definition':True,'raw_forward_backward_projection_errors':adjoint_errors,'all_material_charge_current_Ward_residual':ward,'Gauss_integer_basis_max':float(np.max(abs(self.longitudinal@self.d.T-self.integer_q))),'Gauss_transverse_divergence_max':float(np.max(abs(self.d@self.t))),'physical_charge_unit':1/3,'physical_edge_flux':'E_Q=(D^T Lplus N+t G_T^-1 m)/3; D E_Q=Q','fundamental_finite_matrix_source_body_law':True,'old_solved_states_used':False}
    def groups(self,decoupled=False,theta=0.,pulse_species='white',pulse_edge=2,derivative=None,receiver_species=None,receiver_edge=None):
        zero=lambda:sparse.csr_matrix((self.n,self.n),dtype=complex);groups={}
        if derivative is None:groups[(0,0)]=[self.static.copy(),zero()]
        for part in self.parts:
            match=part['edge']==pulse_edge and(part['category']%2==1)==(pulse_species=='white');phase=np.exp(1j*part['physical_charge']*theta)if match else 1.;coefficient=phase
            if derivative is not None:
                kind,axis=derivative
                if kind in('current','contact'):
                    receiver=pulse_species if receiver_species is None else receiver_species;edge=pulse_edge if receiver_edge is None else receiver_edge
                    if not(part['edge']==edge and(part['category']%2==1)==(receiver=='white')):continue
                    coefficient*=(-1j*part['physical_charge'])if kind=='current'else -part['physical_charge']**2
                elif kind=='force':
                    if decoupled:continue
                    coefficient*=-1j*part['shift'][axis]
                elif kind=='field_contact':
                    if decoupled:continue
                    coefficient*=-part['shift'][axis[0]]*part['shift'][axis[1]]
                else:raise ValueError('Unknown SAME-H derivative')
            shift=(0,0)if decoupled else part['shift']
            for s,c,adjoint in((shift,coefficient,False),(tuple(-x for x in shift),np.conjugate(coefficient),True)):
                pair=groups.setdefault(s,[zero(),zero()])
                for i,key in enumerate(('c0','cphi')):pair[i]+=c*(part[key].conj().T if adjoint else part[key])
        return {key:(value[0].tocsr(),value[1].tocsr())for key,value in groups.items()}
    def hamiltonian(self,decoupled=False,theta=0.,pulse_species='white',pulse_edge=2):return ChargeOperator(self.n,self.grid,self.hphi,self.flux,self.groups(decoupled,theta,pulse_species,pulse_edge),field_energy=self.field_energy)
    def observer(self,kind,axis=None,theta=0.,species='white',edge=2,source_species='white',source_edge=2,decoupled=False):return ChargeOperator(self.n,self.grid,self.hphi,self.flux,self.groups(decoupled,theta,source_species,source_edge,(kind,axis),species,edge),free=False)
    def clamped(self,angles,order=None):
        zero=sparse.csr_matrix((self.n,self.n),dtype=complex);c0=self.static.copy()if order is None else zero.copy();cphi=zero.copy()
        for part in self.parts:
            phase=np.exp(1j*np.dot(part['shift'],angles));coefficient=phase
            if order is not None:
                for axis in order:coefficient*=1j*part['shift'][axis]
            c0+=coefficient*part['c0']+np.conjugate(coefficient)*part['c0'].conj().T;cphi+=coefficient*part['cphi']+np.conjugate(coefficient)*part['cphi'].conj().T
        return PhiHamiltonian(c0,cphi,self.hphi if order is None else np.zeros_like(self.hphi),self.grid)
    def field_shift(self,state,shift):
        x=np.asarray(state).reshape(self.n,len(self.grid),len(self.flux));result=np.zeros_like(x);where={tuple(m):i for i,m in enumerate(self.flux)}
        for i,m in enumerate(self.flux):
            j=where.get(tuple(m+np.asarray(shift)))
            if j is not None:result[:,:,j]=x[:,:,i]
        return result.reshape(np.asarray(state).shape)
    def loop(self,state,index,sine=True):
        shift=self.geometry['actual_cycle_profiles'][index];forward=self.field_shift(state,shift);backward=self.field_shift(state,[-x for x in shift]);return(forward-backward)/(2j)if sine else(forward+backward)/2
    def readout(self,state):
        x=np.asarray(state).reshape(self.n,len(self.grid),len(self.flux));prob=abs(x)**2;body=prob.sum(axis=(1,2));field=prob.sum(axis=(0,1));norm=float(field.sum());colours=self.occupation[:,:,0]+self.occupation[:,:,2];compact=np.all(np.sort(colours,axis=1)==np.array([6,6,6,9]),axis=1);compact_sites=compact[:,None]&(colours==9);boundary=np.max(abs(self.flux),axis=1)==6
        return{'norm':norm,'Q_site':(body@self.q).tolist(),'quark_Q_site':(body@self.quark_q).tolist(),'white_Q_site':(body@self.white_q).tolist(),'whole_quark_Q':float(body@self.quark_q.sum(axis=1)),'whole_white_Q':float(body@self.white_q.sum(axis=1)),'compact_quark_count_site_NOT_tagged_particle_PVM':(body@compact_sites).tolist(),'flux_mean':(field@self.flux).tolist(),'flux_second':np.einsum('n,na,nb->ab',field,self.flux,self.flux).tolist(),'flux_distribution':field.tolist(),'flux_boundary_weight':float(field[boundary].sum()),'transverse_electric_energy':float(field@self.field_energy),'longitudinal_integer_E_mean':(body@self.longitudinal).tolist(),'longitudinal_electric_energy':float(body@self.long_energy),'Phi':float(np.einsum('npm,p->',prob,self.grid)),'Phi2':float(np.einsum('npm,p->',prob,self.grid**2)),'loop_sine':[float(np.vdot(state,self.loop(state,i)).real)for i in range(2)],'loop_cosine':[float(np.vdot(state,self.loop(state,i,False)).real)for i in range(2)]}
