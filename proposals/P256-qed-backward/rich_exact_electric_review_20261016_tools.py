"""Own fresh states, canonical CAR8 source and physical moving-basis observers."""
from typing import Any
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.special import eval_hermite,factorial
import rich_exact_electric_picture_20261016_model as picture
import rich_exact_electric_picture_20261016_fields as electric
import rich_spatial_frame_photon_20261016_fields as primitive


def neutral_state(model):
    matter=model.matter;mask=sum(int(value)<<site for site,value in enumerate(matter.geo['ions']));radial=np.exp(-2*(matter.rnodes-.9)**2);radial/=np.linalg.norm(radial)
    state=np.zeros(model.shape,complex);state[matter.sector.lookup[mask],0,:,0,model.basis.index((0,)*8)]=radial
    result=state.reshape(-1)
    if np.linalg.norm(result-model.preparation())>2e-13:raise AssertionError('Fresh analytic neutral ray differs from declared original preparation')
    return result


def canonical_car(target,reference,site,create):
    rows=[];cols=[];values=[]
    for column,mask in enumerate(reference.masks):
        occupied=(mask>>site)&1
        if occupied==int(create):continue
        changed=mask^(1<<site);sign=(-1)**((mask&((1<<site)-1)).bit_count())
        rows.append(target.lookup[changed]);cols.append(column);values.append(sign)
    return sparse.coo_matrix((values,(rows,cols)),shape=(target.dimension,reference.dimension)).tocoo()


def source(target,reference,state,component=0,create=True,target_basis=None,time=0.,impulse=None):
    """Fresh canonical CAR/sign + full rectangular Wilson multiplier before cut."""
    target_basis=target.basis if target_basis is None else target_basis;nt=len(target_basis);nr=reference.nfield
    tg=tuple(sorted({label[0] for label in target.matter.g['labels']}));rg=tuple(sorted({label[0] for label in reference.matter.g['labels']}));gs=picture.frozen.source_matrices(tg,rg,create)
    x=np.asarray(state).reshape(reference.shape);out=np.zeros((target.nf,target.ng,target.nr,target.nm,nt),complex)
    eps=np.zeros(8) if impulse is None else np.asarray(impulse)
    for site in range(8):
        car=canonical_car(target.matter.sector,reference.matter.sector,site,create)
        coefficient=(-1 if create else 1)*target.fixed.F[:,site]
        displacement=electric.displacement(target_basis,reference.basis,coefficient,time,target.parameters['cavity_inertia'],eps)
        rf=picture.frozen.mult(target.R,displacement).transpose(0,1,3,2).reshape(target.nm*nt,reference.nm*nr)
        y=(x.reshape(-1,reference.nm*nr)@rf.T).reshape(reference.nf,reference.ng,reference.nr,target.nm,nt);y=np.einsum('ij,fjrma->firma',gs[component,site],y,optimize=True)
        for row,column,value in zip(car.row,car.col,car.data):out[row]+=value*y[column]
    return out.reshape(-1)


def arnoldi(apply,state,interval,dimension=12):
    """General reorthogonalized Arnoldi exponential; no imported propagator/state."""
    norm=float(np.linalg.norm(state))
    if norm==0:raise ValueError('Physical science source must be nonzero')
    q=np.asarray(state)/norm;vectors=[];small=np.zeros((dimension+1,dimension),complex);last=0.
    for column in range(dimension):
        vectors.append(q);image=apply(q)
        for _ in range(2):
            for row,v in enumerate(vectors):
                coefficient=np.vdot(v,image);small[row,column]+=coefficient;image-=coefficient*v
        last=float(np.linalg.norm(image));small[column+1,column]=last
        if last<1e-13 or column==dimension-1:break
        q=image/last
    n=len(vectors);compressed=small[:n,:n];coefficients=expm(-1j*interval*compressed)[:,0];out=np.zeros_like(state,dtype=complex)
    for coefficient,v in zip(coefficients,vectors):out+=norm*coefficient*v
    receipt=dict(arnoldi_dimension=n,projected_H_antihermitian_norm=float(np.linalg.norm(compressed-compressed.conj().T)),terminal_image_diagnostic=float(last*abs(coefficients[-1])*norm),norm_defect=float(np.linalg.norm(out)-norm))
    return out,receipt


def propagate(model,state,impulse,time=.04,dt=.01,hopping=True,coupled=True):
    steps=int(np.ceil(time/dt));step=time/steps;value=state.copy();receipts=[]
    for index in range(steps):
        absolute=(index+.5)*step;value,record=arnoldi(lambda vector:model.apply(vector,absolute,impulse,hopping,coupled),value,step);record['absolute_midpoint']=absolute;receipts.append(record)
    return value,dict(method='Own Arnoldi12, two-pass full orthogonalization, exponential-midpoint time ordering',T=time,dt=step,steps=steps,records=receipts)


def field_overlap_matrix(basis,axis,strength):
    # Independent1D Gauss-Hermite integral for full W, spectators exact deltas.
    nodes,weights=np.polynomial.hermite.hermgauss(24);maximum=max(label[axis] for label in basis)
    polynomials=np.array([eval_hermite(n,nodes)/(np.sqrt(2.**n*factorial(n))*np.pi**.25) for n in range(maximum+1)])
    one=(polynomials*(weights*np.exp(1j*strength*nodes))[None,:])@polynomials.T
    out=np.zeros((len(basis),len(basis)),complex)
    for row,a in enumerate(basis):
        for col,b in enumerate(basis):
            if all(a[k]==b[k] for k in range(8) if k!=axis):out[row,col]=one[a[axis],b[axis]]
    return out


def observe(model,state,time,impulse,coupled=True,hopping=True,vertices=True):
    x=np.asarray(state).reshape(model.shape);prob=np.sum(abs(x)**2,axis=(1,2,3,4));rprob=np.sum(abs(x)**2,axis=(0,1,3,4))
    pi=np.array([np.vdot(state,model.field(state,axis,time,impulse)).real for axis in range(8)]);q=np.array([np.vdot(state,model.field(state,axis,time,impulse,'q')).real for axis in range(8)])
    hi=model.apply(state,time,impulse,hopping,coupled);h0=model.free_energy(state,impulse)
    spin=[]
    for op in model.matter.g['L']:
        image=np.einsum('ij,fjrma->firma',op.toarray(),x,optimize=True);spin.append(float(np.vdot(x,image).real))
    result:dict[str,Any]=dict(norm=float(np.linalg.norm(state)**2),Pi=pi.tolist(),a=q.tolist(),density=(prob@model.matter.sector.n).tolist(),total_charge=float(prob@model.fixed.Q.sum(axis=1)),r=float(rprob@model.matter.rnodes),LEFT=spin,physical_H_energy=float(np.vdot(state,hi+h0).real),HI_energy_NOT_conserved=float(np.vdot(state,hi).real),H0_basis_energy=float(np.vdot(state,h0).real))
    if vertices:result['current']={str(axis):float(-np.vdot(state,model.vertex(state,axis,time,impulse,hopping=hopping,coupled=coupled)).real) for axis in (0,2,4,6)}
    return result


def temporal(end,base,kick0,initial):
    return {key:(np.asarray(end[key])-base[key]-np.asarray(kick0[key])+initial[key]).tolist() for key in ('Pi','a','density','r','LEFT')}
