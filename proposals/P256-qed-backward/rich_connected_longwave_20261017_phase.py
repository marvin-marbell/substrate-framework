"""Lie-Poisson/truncated-Wigner response of the local rich MICROSCOPIC H.
Trajectories are quasi-phase-space variables, NOT physical pure states/probabilities.
All local finite-field transitions and independent fermion fluctuations retained.
"""
from itertools import product
import numpy as np
from scipy.linalg import null_space
from rich_connected_longwave_20261017_functional import Functional


class Phase(Functional):
    def expectation_matrices(self,fields):return np.array([np.trace(fields[f]@o)for f,o in zip(self.opfields,self.ops)])

    def hartree(self,rho,p):
        n=self.density(rho);out=np.diag(self.q*(self.W@n+self.X.T@p)[self.site]).astype(complex);r=rho[np.ix_(self.coloured,self.coloured)].reshape(16,4,3,16,4,3);dd=np.array([sum((r[s,a,:,s,a,:]for a in range(4)),np.zeros((3,3),complex))for s in range(16)]);hh=.5*np.einsum('st,tij->sij',self.Wcolour,dd-np.trace(dd,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3)
        for s,f,spin in product(range(16),range(2),range(2)):
            ids=np.array([16*s+8*f+4*spin+c for c in range(3)]);out[np.ix_(ids,ids)]+=hh[s]
        return out,n,dd

    def evaluate(self,rho,fields,a,p,control=False,no_intercell=False):
        ex=self.expectation_matrices(fields);ext=np.r_[ex,1.+0j];phase=np.ones(len(self.terms),complex);use=self.angles>=0
        if not control:phase[use]=np.exp(1j*self.ns[use]*a[self.angles[use]])
        coefficient=self.coeff*phase
        if no_intercell:coefficient=coefficient*(~self.inter)
        fac=ext[self.f1]*ext[self.f2];value=coefficient*fac;h,n,dd=self.hartree(rho,p);np.add.at(h,(self.rows,self.cols),value);np.add.at(h,(self.cols,self.rows),value.conj());rv=coefficient*rho[self.cols,self.rows];oc=np.zeros(self.opcount,complex)
        for fs,other in((self.f1,self.f2),(self.f2,self.f1)):
            good=fs>=0;np.add.at(oc,fs[good],rv[good]*ext[other[good]])
        for c,i,j in self.bterms:oc[i]+=.5*c*ex[j];oc[j]+=.5*c*ex[i]
        hh=[f['bare'].copy()for f in self.fields]
        for k,(f,o)in enumerate(zip(self.opfields,self.ops)):hh[f]+=oc[k]*o+oc[k].conjugate()*o.conj().T
        current=np.bincount(self.edges[self.hop],weights=2*(self.ns[self.hop]/3)*(rv[self.hop]*fac[self.hop]).imag,minlength=96);pdot=np.zeros(81)
        if not control:np.add.at(pdot,self.angles[use],2*self.ns[use]*(rv[use]*fac[use]).imag)
        adot=self.G@p+self.X@n;efull=self.T@n+self.K@p
        colour=np.array([[np.trace(t@dd[s]).real for t in self.ts]for s in range(16)]);energy=float(.5*efull@efull+.5*np.einsum('sa,st,ta',colour,self.Wcolour,colour)+2*np.sum(rv*fac).real+sum(np.trace(q@f['bare']).real for q,f in zip(fields,self.fields))+sum(c*(ex[i]*ex[j]).real for c,i,j in self.bterms))
        return h,hh,adot,pdot,current,efull,n,energy,ex

    def derivative(self,rho,fields,a,p,control=False,no_intercell=False):
        h,hh,adot,pdot,current,e,n,energy,ex=self.evaluate(rho,fields,a,p,control,no_intercell);dr=np.zeros_like(rho)
        for ids in self.groups:
            rr=rho[np.ix_(ids,ids)];hd=h[np.ix_(ids,ids)];dr[np.ix_(ids,ids)]=-1j*(hd@rr-rr@hd)
        df=[-1j*(x@y-y@x)for x,y in zip(hh,fields)];return dr,df,adot,pdot

    def jet(self,state,tangent,control=False,no_intercell=False):
        rho,fields,a,p=state;dr,df,da,dp=tangent;h,hh,adot,pdot,j,e,n,energy,ex=self.evaluate(rho,fields,a,p,control,no_intercell);dex=self.expectation_matrices(df);ext=np.r_[ex,1.+0j];dext=np.r_[dex,0.+0j];fac=ext[self.f1]*ext[self.f2];dfac=dext[self.f1]*ext[self.f2]+ext[self.f1]*dext[self.f2];use=self.angles>=0;phase=np.ones(len(self.terms),complex);dphase=np.zeros(len(self.terms),complex)
        if not control:phase[use]=np.exp(1j*self.ns[use]*a[self.angles[use]]);dphase[use]=1j*self.ns[use]*da[self.angles[use]]*phase[use]
        c=self.coeff*phase;dc=self.coeff*dphase
        if no_intercell:c=c*(~self.inter);dc=dc*(~self.inter)
        dh,dn,_=self.hartree(dr,dp);value=dc*fac+c*dfac;np.add.at(dh,(self.rows,self.cols),value);np.add.at(dh,(self.cols,self.rows),value.conj());rv=c*rho[self.cols,self.rows];drv=dc*rho[self.cols,self.rows]+c*dr[self.cols,self.rows];oc=np.zeros(self.opcount,complex)
        for fs,other in((self.f1,self.f2),(self.f2,self.f1)):
            good=fs>=0;np.add.at(oc,fs[good],drv[good]*ext[other[good]]+rv[good]*dext[other[good]])
        for c0,i,k in self.bterms:oc[i]+=.5*c0*dex[k];oc[k]+=.5*c0*dex[i]
        dhh=[np.zeros_like(x)for x in hh]
        for k,(f,o)in enumerate(zip(self.opfields,self.ops)):dhh[f]+=oc[k]*o+oc[k].conjugate()*o.conj().T
        dcurrent=np.bincount(self.edges[self.hop],weights=2*(self.ns[self.hop]/3)*(drv[self.hop]*fac[self.hop]+rv[self.hop]*dfac[self.hop]).imag,minlength=96);dpdot=np.zeros(81)
        if not control:np.add.at(dpdot,self.angles[use],2*self.ns[use]*(drv[use]*fac[use]+rv[use]*dfac[use]).imag)
        dadot=self.G@dp+self.X@dn;br=np.zeros_like(rho);tr=np.zeros_like(rho)
        for ids in self.groups:
            rr=rho[np.ix_(ids,ids)];r1=dr[np.ix_(ids,ids)];h0=h[np.ix_(ids,ids)];h1=dh[np.ix_(ids,ids)];br[np.ix_(ids,ids)]=-1j*(h0@rr-rr@h0);tr[np.ix_(ids,ids)]=-1j*(h0@r1-r1@h0+h1@rr-rr@h1)
        bf=[-1j*(x@y-y@x)for x,y in zip(hh,fields)];tf=[-1j*(x@y-y@x+dx@r-r@dx)for x,y,dx,r in zip(hh,df,dhh,fields)]
        de=self.T@dn+self.K@dp;de_dot=self.T@self.density(tr)+self.K@dpdot;native_ward=float(np.max(abs(de_dot-3*dcurrent)));denergy=float(np.trace(h@dr).real+sum(np.trace(x@y).real for x,y in zip(hh,df))+adot@dp-pdot@da)
        return (br,bf,adot,pdot),(tr,tf,dadot,dpdot),{'current':dcurrent,'Eint':de,'Nint':dn,'energy':denergy,'native_full_H_field_current_Ward':native_ward,'baseline_energy':energy}

    def sample(self,reference,rng):
        rho=reference['rho'].copy();offset=0
        for ids in self.groups:
            k=len(ids)//2;u=reference['U'][:,offset:offset+k];v=reference['V'][:,offset:offset+k];z=(rng.standard_normal((k,k))+1j*rng.standard_normal((k,k)))/2;rho+=v@z@u.conj().T+u@z.conj().T@v.conj().T;offset+=k
        fields=[]
        for psi in reference['psis']:
            ex=null_space(psi.conj()[None,:]);z=(rng.standard_normal(ex.shape[1])+1j*rng.standard_normal(ex.shape[1]))/2;eta=ex@z;fields.append(np.outer(psi,psi.conj())+np.outer(eta,psi.conj())+np.outer(psi,eta.conj()))
        return rho,fields,rng.uniform(-np.pi,np.pi,81),np.zeros(81)

    def source(self,rho,strength):
        pos=np.asarray(self.g['positions']);profile=np.zeros(256);first=self.site//4==0;profile[first]=pos[self.site[first],0]*self.q[first]/3;u=np.exp(1j*strength*profile);return u[:,None]*rho*u.conj()[None,:]

    def probes(self,state,control=False,no_intercell=False):
        rho,fields,a,p=state;_,_,_,_,j,e,n,energy,ex=self.evaluate(rho,fields,a,p,control,no_intercell);et=self.P@e;el=e-et;positions=np.asarray(self.g['positions']);d=np.array([l['displacement']for l in self.g['links']]);mid=np.array([positions[l['start']]+np.array(l['displacement'])/2 for l in self.g['links']]);cells=(np.floor(mid[:,0]/np.sqrt(2)+.5).astype(int)%4);cellj=[];celle=[];cellt=[];celll=[];cellenergy=[]
        for c in range(4):
            select=cells==c;gram=d[select].T@d[select];inv=np.linalg.pinv(gram);cellj.append(d[select].T@j[select]);celle.append(inv@(d[select].T@e[select])/3);cellt.append(inv@(d[select].T@et[select])/3);celll.append(inv@(d[select].T@el[select])/3);cellenergy.append(.5*float(e[select]@e[select]))
        raw=np.asarray(self.g['probe_profiles_raw']);uniform=np.asarray(self.g['uniform_profiles']);return {'Qsite':n/3,'Qcell':n.reshape(4,4).sum(axis=1)/3,'current_edge':j,'Efull_edge':e/3,'Etransverse_edge':et/3,'Elong_edge':el/3,'cell_current_vector':np.array(cellj),'cell_Efull_vector':np.array(celle),'cell_Etransverse_vector':np.array(cellt),'cell_Elong_vector':np.array(celll),'cell_electric_energy':np.array(cellenergy),'q_current':raw.T@j,'q_Efull':raw.T@e/3,'q_ET':raw.T@et/3,'uniform_E':uniform.T@e/3,'Phi':np.array([np.trace(fields[3*c]@np.diag(self.phigrid)).real for c in range(4)]),'R_trace':np.array([sum(np.trace(fields[3*c+1]@self.rotor['coordinate'][i][i].toarray()).real for i in range(3))for c in range(4)]),'colour_ReTrV':np.array([np.trace(fields[3*c+2]@self.loop['B']).real for c in range(4)]),'energy':np.array(energy),'Gauss_max':np.array(np.max(abs(self.d@e-n))/3)}


def current_image(model,reference,edge,vector_only=False,replace_R_by_mean=False):
    """Exact native quantum J|product-ref> norm, including PH x R/PW x U.
    Rotor reference is |m=0>; each signed root harmonic is orthogonal exactly.
    """
    rho=reference['rho'];psis=reference['psis'];entries=[]
    for k in np.flatnonzero((model.edges==edge)&(model.ns!=0)):
        if vector_only and model.fields[model.opfields[model.f1[k]]]['kind']!='INTERNAL_R':continue
        factors=[int(f)for f in(model.f1[k],model.f2[k])if f>=0];c=-1j*model.ns[k]/3*model.coeff[k];shift=model.ns[k]if model.angles[k]>=0 else 0
        entries.append((int(model.rows[k]),int(model.cols[k]),c,[(f,False)for f in factors],shift));entries.append((int(model.cols[k]),int(model.rows[k]),c.conjugate(),[(f,True)for f in reversed(factors)],-shift))
    cache={}
    def field_product(factors):
        key=tuple(factors)
        if key in cache:return cache[key]
        per={};scalar=1.+0j
        for k,dag in factors:
            f=model.opfields[k];o=model.ops[k].conj().T if dag else model.ops[k]
            if replace_R_by_mean and model.fields[f]['kind']=='INTERNAL_R':scalar*=np.vdot(psis[f],o@psis[f]);continue
            per[f]=per.get(f,np.eye(len(psis[f])))@o
        for f,o in per.items():scalar*=np.vdot(psis[f],o@psis[f])
        cache[key]=scalar;return scalar
    norm=0.+0j;mean=0.+0j
    for i,j,c,ff,s in entries:
        if s==0:mean+=c*rho[j,i]*field_product(ff)
        for k,l,b,gg,t in entries:
            if s+t!=0:continue
            wick=rho[j,i]*rho[l,k]+rho[l,i]*((1 if j==k else 0)-rho[j,k]);norm+=c*b*wick*field_product(ff+gg)
    return {'edge':edge,'vector_only':vector_only,'replace_R_by_mean':replace_R_by_mean,'raw_native_J_image_norm2':float(norm.real),'imaginary_roundoff':float(norm.imag),'quantum_mean_current':float(mean.real),'raw_strength_NOT_probability':True,'includes_joint_PH_R_PW_and_own_integer_EM_harmonics':not replace_R_by_mean}
