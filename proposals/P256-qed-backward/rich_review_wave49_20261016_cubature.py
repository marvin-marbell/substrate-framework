"""Independent finite-q cubature audit at FIXED retained R2/field1, no solved state."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm,block_diag
import rich_gauge_band_frame_20261016_model as band
import rich_local_frame_spatial_carrier_20261016_model as carrier
import rich_spatial_frame_photon_20261016_kernel as kernel
import rich_spatial_frame_photon_20261016_fields as fields
from rich_spatial_frame_photon_20261016_model import rotations

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def material_grid(level,na,nb,ng):
    z,w=np.polynomial.legendre.leggauss(nb)
    points=np.array([(a,np.arccos(b),c) for a in np.arange(na)*2*np.pi/na for b in z for c in np.arange(ng)*2*np.pi/ng])
    weights=np.tile(np.repeat(w/(2*na*ng),ng),na);blocks=[]
    for j in range(level+1):
        J=band.jmat(j);m=np.arange(-j,j+1);d=np.array([expm(-1j*np.arccos(b)*J[1]) for b in z])
        D=np.exp(-1j*points[:,0,None,None]*m[None,:,None])*np.tile(np.repeat(d,ng,axis=0),(na,1,1))*np.exp(-1j*points[:,2,None,None]*m[None,None,:])
        blocks.append(np.sqrt(2*j+1)*D.reshape(len(points),-1))
    return dict(F=np.concatenate(blocks,axis=1),weights=weights,points=points)


def block(angular,body,Q,modes):
    F=angular['F'];w=angular['weights'];R=len(F[0]);rot=rotations(angular['points'])
    active=np.nonzero(Q)[0];x=np.einsum('qij,sj->qsi',rot,body[active]);f=kernel.profiles(x,modes)
    conn=np.einsum('s,qsla->qla',Q[active],np.cross(x[:,:,None,:],f))
    gram=(F.conj()[:,:,None]*F[:,None,:]*w[:,None,None]).reshape(len(w),R*R).T
    basis=fields.basis(1);ext=fields.basis(2);nf=len(basis)
    q=[fields.primitive(basis,basis,a,'q').toarray() for a in range(8)]
    qe=np.array([fields.primitive(ext,basis,a,'q').toarray() for a in range(8)])
    def project(v):return (gram@v.reshape(len(w),-1)).reshape(R,R,*v.shape[1:])
    L=[block_diag(*[np.kron(-band.jmat(j)[a].T,np.eye(2*j+1)) for j in range(3)]) for a in range(3)]
    cas=sum(v@v for v in L);out=np.kron(cas,np.eye(nf)).astype(complex)
    for axis in range(3):
        for mode in range(8):
            coefficient=project(conn[:,mode,axis]);out-=np.kron(L[axis]@coefficient+coefficient@L[axis],q[mode])
    image=np.einsum('qla,lek->qaek',conn,qe,optimize=True)
    square=np.einsum('qaei,qaej->qij',image.conj(),image,optimize=True)
    out+=project(square).transpose(0,2,1,3).reshape(R*nf,R*nf)
    return out/12,float(np.linalg.norm(F.conj().T@(w[:,None]*F)-np.eye(R)))


def run():
    body=(carrier.CENTRES[:,None,:]+.24*(band.geometry()['sites'][None,:,:]-.5)).reshape(32,3)
    Q=np.zeros(32);Q[0]=-1;Q[10]=1;Q[11]=-1
    modes=kernel.default_modes();modes[:,3:6]*=np.sinc(.08*np.linalg.norm(modes[:,:3],axis=1)/np.pi)[:,None]
    old,go=block(band.angular((0,1,2)),body,Q,modes)
    high,gh=block(material_grid(2,24,16,24),body,Q,modes)
    fine,gf=block(material_grid(2,28,18,28),body,Q,modes)
    coarse=float(np.linalg.norm(old-high));refined=float(np.linalg.norm(high-fine))
    result=dict(kind='OWN_ACTUAL47_R2_FIELD1_FINITE_Q_CUBATURE_COUNTERCASE',retained_R_dimension=35,retained_field_dimension=9,grids=['Band R2 alpha4pi14 betaGL8 gamma2pi14:7distinctSO3alpha','SO3 alpha24 betaGL16 gamma24','SO3 alpha28 betaGL18 gamma28'],Gram_errors=[go,gh,gf],old_vs_high_norm=coarse,old_vs_high_max=float(np.max(abs(old-high))),high_vs_fine_norm=refined,high_vs_fine_max=float(np.max(abs(high-fine))),limits='Actual47 shell-averaged plane profiles and 32microsite positions, one valid integer N17 charge configuration including a neutral pair. Fixed full R2/field1 rotational block; all Lambda² field products before cut. This is a cubature-discretization discriminator, not evolved full CAR32 evidence or a universal convergence proof.')
    np.savez_compressed(ROOT/(P+'cubature_raw.npz'),old=old,high=high,fine=fine,positions=body,charges=Q,modes=modes)
    (ROOT/(P+'cubature_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
