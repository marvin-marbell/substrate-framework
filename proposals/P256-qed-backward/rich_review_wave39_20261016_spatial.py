"""Fresh selected physical-source derivatives and finite-complement discrimination."""
import json
from pathlib import Path
import numpy as np
from rich_local_frame_carrier_20261015_model import Model,VERTICES
from rich_spatial_gauge_source_20261016_family import SpatialFamily
ROOT=Path(__file__).resolve().parent
P='rich_review_wave39_20261016_'

def main():
    interface=json.loads((ROOT/'rich_dressed_local_carrier_20261015_interface.json').read_text())
    neutral_model=Model(6,interface['parameters'])
    with np.load(ROOT/interface['state_file']) as data:neutral=data['neutral'].copy()
    m=Model(7,interface['parameters'])
    states=[m.source_from(neutral_model,0,'addition',spin,0)@neutral for spin in range(2)]
    states=[v/np.linalg.norm(v) for v in states];v=states[0]
    spinphase=float(np.linalg.norm((m.J[0]-1j*m.J[1])@v+states[1]))
    badphase=float(np.linalg.norm((m.J[0]-1j*m.J[1])@v-states[1]))
    del neutral_model
    direction=np.array([.2,-.3,.7]);direction/=np.linalg.norm(direction)
    fam=SpatialFamily(m,direction,'B',8);h=fam.h0
    dh,contact,hop,mech=fam.derivatives();hv=h@v;dv=dh@v;cv=contact@v
    eps=2e-4
    plus=fam.law(eps)@v;minus=fam.law(-eps)@v
    fd1=(plus-minus)/(2*eps);fd2=(plus-2*hv+minus)/eps**2
    dropped=float(np.linalg.norm(fd1-hop@v)/np.linalg.norm(fd1))
    early=float(np.linalg.norm(cv-fam.wrong_contact()@v)/np.linalg.norm(cv))
    result=dict(dimension=m.dimension,source='fresh bare dressed B(site0,orbital0,addition) on DECLARED neutral amplitude; not own stationary ground',spin_lowering_phase_residual=spinphase,bad_spin_phase_residual=badphase,derivative_relative=float(np.linalg.norm(fd1-dv)/np.linalg.norm(dv)),contact_relative=float(np.linalg.norm(fd2-cv)/np.linalg.norm(cv)),mechanical_norm=float(np.linalg.norm(mech@v)),hop_norm=float(np.linalg.norm(hop@v)),dropped_mechanical_relative=dropped,early_projected_contact_relative=early,magnetic_source_mean=float(-np.vdot(v,dv).real),scalar_z_spinflip_complex=[float(z) for z in [np.vdot(states[1],dv).real,np.vdot(states[1],dv).imag]])
    # Independent unprojected Q geometry: ions must rotate with the same positions.
    q=m.sector.charge;prob=np.sum(abs(v.reshape(m.sector.dimension,10,-1))**2,axis=(1,2));meanq=prob@q
    result['net_charge_mean']=meanq.tolist();result['total_net_charge']=float(meanq.sum())
    R,w,basis=fam.rotations,fam.weights,fam.basis
    x=np.einsum('qab,vb->qva',R,VERTICES)
    ownlambda=.5*(np.sum(x*x,axis=2)[:,:,None]*direction-x*np.einsum('qva,a->qv',x,direction)[:,:,None])
    result['own_geometric_connection_max']=float(np.max(abs(ownlambda-fam.connection)))
    # Pure-gradient London compression: full scalar phase is unitary but PUP is not.
    constant=SpatialFamily(m,direction,'constant',8);D=constant.dipole();dd,_,hh,mm=constant.derivatives()
    ward=1j*(D@(h@v)-h@(D@v));dconst=dd@v
    result['constant_Ward_relative']=float(np.linalg.norm(dconst-ward)/np.linalg.norm(dconst))
    result['constant_wrong_Ward_sign_relative']=float(np.linalg.norm(dconst+ward)/np.linalg.norm(dconst))
    oneq=np.array([1.,-1.,-1.]);chi=np.einsum('qva,a,v->q',x,direction,oneq)
    U=np.einsum('qi,q,qj->ij',basis,w*np.exp(1j*.4*chi),basis)
    result['finite_complement_unitarity_defect']=float(np.linalg.norm(U.conj().T@U-np.eye(10)))
    # Independent product-before-cut witness for the same selected charge geometry.
    lam=np.einsum('v,qva->qa',oneq,ownlambda)
    integrate=lambda f:np.einsum('qi,q,qj->ij',basis,w*f,basis)
    full=integrate(np.sum(lam*lam,axis=1));earlymat=sum(integrate(lam[:,a])@integrate(lam[:,a]) for a in range(3))
    result['own_Lambda_square_missing_complement_norm']=float(np.linalg.norm(full-earlymat))
    np.savez_compressed(ROOT/(P+'spatial_raw.npz'),own_lambda=ownlambda,weights=w,net_charges=q,source_sector_probabilities=prob,Lambda2_full=full,Lambda2_early=earlymat,compressed_gauge=U)
    (ROOT/(P+'spatial_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert result['derivative_relative']<1e-6 and result['contact_relative']<1e-4
    assert dropped>1e-3 and early>1e-3
    assert spinphase<1e-8 and badphase>1
    assert result['own_geometric_connection_max']<1e-12
    assert result['constant_Ward_relative']<1e-8
    assert result['finite_complement_unitarity_defect']>1e-3
if __name__=='__main__':main()
