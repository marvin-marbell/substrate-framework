"""Microscopic connected local-rich H -> explicit Slater/finite-field functional.
No primitive-pair cut. Subsequent quadratic response is a DECLARED approximation.
"""
from itertools import product
import numpy as np
from scipy.linalg import eigh
from rich_connected_longwave_20261017_geometry import construct
from rich_colour_composite_20261014_model import loop_field,generators
from rich_spatial_colour_root_20261015_model import so3
from rich_common_colour_lepton_20261015_model import scalar_field

PAULI=(np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1.,-1.]))


def mode(s,f,spin,c):return 16*s+8*f+4*spin+c


class Functional:
    def __init__(self):
        self.g=construct();self.n=256;self.ops=[];self.opfields=[];self.oplabels=[];self.oplookup={};self.terms=[];self.bterms=[];self.loop=loop_field();self.ts=generators();self.fields=[]
        params={'scalar_points':7,'scalar_extent':4.,'scalar_metric':1.,'scalar_quartic':.2,'scalar_quadratic':1.,'orientation_azimuth_points':8,'orientation_polar_points':6};scalar=scalar_field(params);rotor=so3(params);self.phigrid=scalar['grid'];self.rotor=rotor
        for cell in range(4):
            self.fields.extend([{'cell':cell,'kind':'Phi','bare':scalar['kinetic']/2+2*scalar['potential']},{'cell':cell,'kind':'INTERNAL_R','bare':rotor['L2'].toarray()/6},{'cell':cell,'kind':'PW19','bare':.9*self.loop['C']-.25*self.loop['B']}])
        for field in self.fields:field['bare']=np.asarray(field['bare'],dtype=complex)
        self.d=np.asarray(self.g['incidence']);self.T=np.asarray(self.g['integer_particular_flow'],float);self.K=np.asarray(self.g['integer_cycle_matrix'],float);self.G=self.K.T@self.K;self.X=self.K.T@self.T;self.W=self.T.T@self.T;self.lp=np.asarray(self.g['laplacian_pseudoinverse']);self.P=np.asarray(self.g['transverse_projector']);self.cotree=np.asarray(self.g['cotree_links']);self.phase_index=np.full(96,-1,int);self.phase_index[self.cotree]=np.arange(81)
        self.site=np.repeat(np.arange(16),16);self.q=np.array([(2 if f==0 else-1)if c<3 else(0 if f==0 else-3)for s,f,spin,c in product(range(16),range(2),range(2),range(4))],float)
        self.coloured=np.array([mode(s,f,spin,c)for s,f,spin,c in product(range(16),range(2),range(2),range(3))]);self.groups=[np.array([mode(s,f,spin,c)for s,spin,c in product(range(16),range(2),range(3))])for f in range(2)]+[np.array([mode(s,f,spin,3)for s,spin in product(range(16),range(2))])for f in range(2)]
        self.Vabel=self.W[self.site[:,None],self.site[None,:]]*self.q[:,None]*self.q[None,:];self.Wcolour=1.8*self.W;self.selfdiag=.5*np.diag(self.W)[self.site]*self.q**2;self.selfdiag[self.coloured]+=.5*np.diag(self.Wcolour)[self.site[self.coloured]]*4/3
        self._hopping();self._colour();a=np.array(self.terms,dtype=object)
        self.rows=np.array(a[:,0],int);self.cols=np.array(a[:,1],int);self.coeff=np.array(a[:,2],complex);self.edges=np.array(a[:,3],int);self.ns=np.array(a[:,4],float);self.f1=np.array(a[:,5],int);self.f2=np.array(a[:,6],int);self.inter=np.array([e>=0 and self.g['links'][e]['intercell']for e in self.edges]);self.hop=self.edges>=0
        self.angles=np.where(self.hop,self.phase_index[np.maximum(self.edges,0)],-1);self.field_ids=np.array(self.opfields,int);self.opcount=len(self.ops)
        self.preparation_phase_mean=np.where((self.angles>=0)&(self.ns!=0),0.,1.)

    def op(self,field,label,matrix):
        key=(field,label)
        if key not in self.oplookup:
            self.oplookup[key]=len(self.ops);self.ops.append(np.asarray(matrix,complex));self.opfields.append(field);self.oplabels.append(label)
        return self.oplookup[key]

    def term(self,i,j,c,edge,n,factors):
        if c!=0:self.terms.append((i,j,complex(c),edge,n,factors[0]if factors else-1,factors[1]if len(factors)>1 else-1))

    def _hopping(self):
        chords={e:c for c,e in enumerate(self.g['colour_chords'])}
        for edge,l in enumerate(self.g['links']):
            s,t=l['start'],l['end'];d=np.asarray(l['displacement']);cells=list(dict.fromkeys([s//4,t//4]));weight=1/len(cells)
            for flavour,white in product(range(2),(False,True)):
                n=(0 if flavour==0 else-3)if white else(2 if flavour==0 else-1)
                for a,b in product(([3]if white else range(3)),repeat=2):
                    factors=[]
                    if white or edge not in chords:
                        if a!=b:continue
                    else:factors=[self.op(3*chords[edge]+2,f'Udag_{a}_{b}',self.loop['U'][b,a].conj().T)]
                    for cell in cells:
                        phi=self.op(3*cell,'Phi',np.diag(self.phigrid))
                        for spin in range(2):self.term(mode(t,flavour,spin,a),mode(s,flavour,spin,b),.65*weight,edge,n,[phi,*factors])
                        for axis,body in product(range(3),range(3)):
                            if abs(d[body])<1e-14:continue
                            r=self.op(3*cell+1,f'R_{axis}_{body}',self.rotor['coordinate'][axis][body].toarray())
                            for up,down in product(range(2),repeat=2):self.term(mode(t,flavour,up,a),mode(s,flavour,down,b),1j*.6*weight*d[body]*PAULI[axis][up,down],edge,n,[r,*factors])

    def _colour(self):
        for members in self.g['colour_subtrees']:
            percell=[]
            for cell in range(4):
                left=4*cell in members;right=4*cell+3 in members;ops=[]
                for axis in range(8):
                    m=int(left)*self.loop['L'][axis]+int(right)*self.loop['R'][axis]
                    self.fields[3*cell+2]['bare']+=.9*(m@m)
                    if not left and not right:continue
                    op=self.op(3*cell+2,f'cut_{int(left)}{int(right)}_{axis}',m);ops.append((axis,op))
                    for site,f,spin,a,b in product(members,range(2),range(2),range(3),range(3)):
                        self.term(mode(site,f,spin,a),mode(site,f,spin,b),.9*self.ts[axis][a,b],-1,0,[op])
                percell.append(dict(ops))
            for c,e in product(range(4),range(4)):
                if c>=e:continue
                for axis in set(percell[c])&set(percell[e]):self.bterms.append((1.8,percell[c][axis],percell[e][axis]))

    def expectations(self,psis):
        return np.array([np.vdot(psis[f],o@psis[f])for f,o in zip(self.opfields,self.ops)])

    def factors(self,expect):
        ext=np.r_[expect,1.+0j];return ext[self.f1]*ext[self.f2]

    def density(self,rho):
        return np.bincount(self.site,weights=(self.q*np.diag(rho).real),minlength=16)

    def interaction(self,rho,selfterm=False):
        rho=np.asarray(rho,dtype=complex)
        n=self.density(rho);out=-self.Vabel*rho;out[np.diag_indices(256)]+=self.q*(self.W@n)[self.site]
        r=rho[np.ix_(self.coloured,self.coloured)].reshape(16,4,3,16,4,3);tr=np.einsum('saitbi->satb',r);hc=-self.Wcolour[:,None,None,:,None,None]*(.5*tr[:,:,None,:,:,None]*np.eye(3)[None,None,:,None,None,:]-r/6)
        dd=np.array([sum((r[s,a,:,s,a,:]for a in range(4)),np.zeros((3,3),complex))for s in range(16)]);hh=.5*np.einsum('st,tij->sij',self.Wcolour,dd-np.trace(dd,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3)
        for s,a in product(range(16),range(4)):hc[s,a,:,s,a,:]+=hh[s]
        out[np.ix_(self.coloured,self.coloured)]+=hc.reshape(192,192)
        if selfterm:out[np.diag_indices(256)]+=self.selfdiag
        return(out+out.conj().T)/2

    def mean(self,rho,psis,pmean=None):
        ex=self.expectations(psis);fac=self.factors(ex);v=self.coeff*fac*self.preparation_phase_mean;h=self.interaction(rho,True);np.add.at(h,(self.rows,self.cols),v);np.add.at(h,(self.cols,self.rows),v.conj())
        if pmean is not None:h[np.diag_indices(256)]+=self.q*(self.X.T@pmean)[self.site]
        field=[x['bare'].copy()for x in self.fields];oc=np.zeros(self.opcount,complex);rv=self.coeff*rho[self.cols,self.rows]*self.preparation_phase_mean;ext=np.r_[ex,1.+0j]
        for fs,other in((self.f1,self.f2),(self.f2,self.f1)):
            good=fs>=0;np.add.at(oc,fs[good],rv[good]*ext[other[good]])
        for c,a,b in self.bterms:oc[a]+=.5*c*ex[b];oc[b]+=.5*c*ex[a]
        for k,(f,o)in enumerate(zip(self.opfields,self.ops)):field[f]+=oc[k]*o+oc[k].conjugate()*o.conj().T
        return h,field,ex

    def orbitals(self,h):
        occ=[];virt=[];eo=[];ev=[]
        for ids in self.groups:
            vals,vec=eigh(h[np.ix_(ids,ids)]);cut=len(ids)//2;u=np.zeros((256,cut),complex);v=u.copy();u[ids]=vec[:,:cut];v[ids]=vec[:,cut:];occ.append(u);virt.append(v);eo.extend(vals[:cut]);ev.extend(vals[cut:])
        return np.column_stack(occ),np.column_stack(virt),np.array(eo),np.array(ev)

    def initial(self):
        psis=[]
        for f in self.fields:
            if f['kind']=='Phi':v=np.zeros(7,complex);v[np.argmin(abs(self.phigrid+4/3))]=1
            elif f['kind']=='INTERNAL_R':v=np.zeros(10,complex);v[0]=1/np.sqrt(2);v[[1,5,9]]=1/np.sqrt(6)
            else:_,vec=eigh(f['bare']);v=vec[:,0]
            psis.append(v)
        h,_,_=self.mean(np.eye(256)/2,psis,np.zeros(81));u,_,_,_=self.orbitals(h);return u@u.conj().T,psis

    def stationary(self,iterations=160,tol=2e-9):
        rho,psis=self.initial();history=[];pmean=np.zeros(81)
        for step in range(iterations):
            h,fields,ex=self.mean(rho,psis,pmean);u,v,eo,ev=self.orbitals(h);rr=u@u.conj().T;new=[]
            for old,hh in zip(psis,fields):
                _,vec=eigh(hh);z=vec[:,0];z*=np.exp(-1j*np.angle(np.vdot(old,z)));z=.6*old+.4*z;new.append(z/np.linalg.norm(z))
            defect=max(np.linalg.norm(rr-rho),max(np.linalg.norm(a-b)for a,b in zip(new,psis)));history.append(float(defect));rho=.6*rho+.4*rr;psis=new
            if step%10==0:print(__import__('json').dumps({'stage':'FRESH_local_rich_variational_sea','iteration':step,'state_change':defect,'local_Phi':[float(np.vdot(psis[3*c],np.diag(self.phigrid)@psis[3*c]).real)for c in range(4)]}),flush=True)
            if defect<tol:break
        h,fields,_=self.mean(rho,psis,pmean);u,v,eo,ev=self.orbitals(h);rho=u@u.conj().T;h,fields,ex=self.mean(rho,psis,pmean)
        return {'rho':rho,'psis':psis,'U':u,'V':v,'eo':eo,'ev':ev,'pmean':pmean,'h':h,'field_h':fields,'expect':ex,'history':history,'fermion_stationarity_residual':float(np.linalg.norm(h@rho-rho@h)),'local_field_stationarity_residuals':[float(np.linalg.norm(hh@z-np.vdot(z,hh@z)*z))for hh,z in zip(fields,psis)]}
