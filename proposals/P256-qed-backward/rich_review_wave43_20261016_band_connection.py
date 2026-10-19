"""Independent changed-layer consumer: actual quantum-frame basis/force connections."""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from rich_gauge_band_frame_20261016_model import Model,DEFAULTS,tensor

ROOT=Path(__file__).resolve().parent
P='rich_review_wave43_20261016_'


def main():
    cfg=json.loads((ROOT/(P+'band_connection_input.json')).read_text())
    ref=Model(4);m=Model(5);p=dict(DEFAULTS);k=np.array(p['k']);q=np.array(cfg['q'])
    x=m.source_from(ref,0,True)@ref.preparation();x/=np.linalg.norm(x)
    x=expm_multiply(1j*cfg['pulse']*m.E[2],x)
    x=expm_multiply(-1j*cfg['time']*m.H,x)
    # Build second actual H at changed momentum, not replayed producer delta.
    other=Model(5,{**p,'k':(k+q).tolist()})
    gamma=m.geo['gamma'];identity=sp.eye(len(m.rnodes))
    current=[]
    for a in range(3):
        axis=sp.csr_matrix(m.H.shape,dtype=complex)
        for i in range(3):
            for ell in range(3):axis+=tensor(m.sector.bilinear(gamma[i]),m.g['O'][ell][i],m.material['O'][ell][a],identity)
        current.append(2*np.sin(k[a]+q[a]/2)*axis)
    contraction=sum((2*np.sin(q[a]/2)*(current[a]@x) for a in range(3)),np.zeros_like(x))
    bare_delta=other.H@x-m.H@x
    repeats=int(np.prod(m.shape[1:]));positions=m.geo['sites']
    phase_k=np.repeat(np.exp(1j*m.sector.n@(positions@k)),repeats)
    phase_p=np.repeat(np.exp(1j*m.sector.n@(positions@(k+q))),repeats)
    rho=phase_p.conj()*phase_k
    # Consumer is represented in the periodic basis; density connects k and k+q.
    y=phase_k.conj()*x
    lhs=phase_p.conj()*(other.H@(phase_p*rho*y))-rho*phase_k.conj()*(m.H@(phase_k*y))
    rhs=phase_p.conj()*contraction
    wrong=phase_p.conj()*(other.H@(phase_p*y))-phase_k.conj()*(m.H@(phase_k*y))-phase_k.conj()*contraction
    # Independently differentiate genuine radial local field operator in fixed u(r) measure.
    r=m.rnodes;mass=p['radial_mass'];cas=m.g['casimir'].diagonal();beta=m.sector.bilinear(m.geo['beta']).diagonal()
    def local(shift):
        z=r+shift
        v=p['radial_stiffness']*(z-p['radial_centre'])**2/2+3/(8*mass*z*z)
        return v[None,None,None,:]+cas[None,:,None,None]/(2*mass*z[None,None,None,:]**2)+p['root_coupling']*beta[:,None,None,None]*z[None,None,None,:]
    eps=cfg['difference_step'];force=-(local(eps)-local(-eps))/(2*eps)
    fd=force*x.reshape(m.shape)
    missing_cone=3/(4*mass*r**3)+cas[:,None]/(mass*r[None,:]**3)
    missing_norm=np.linalg.norm(missing_cone[None,:,None,:]*x.reshape(m.shape))
    prob=abs(x.reshape(m.shape))**2
    gh=np.array([lab[0]==1.5 for lab in m.g['labels']]);rh=np.array([lab[0]==1 for lab in m.material['labels']])
    result=dict(kind='OWN_QUANTUM_BAND_BASIS_AND_RADIAL_SUPPLIER_CONSUMER',input=cfg,
        actual_H_midpoint_error=float(np.linalg.norm(bare_delta-contraction)),
        periodic_density_connection_Ward_error=float(np.linalg.norm(lhs-rhs)),
        omitted_density_connection_error=float(np.linalg.norm(wrong)),
        r_force_derivative_error=float(np.linalg.norm(fd.ravel()-m.force@x)),
        omitted_cone_frame_force_on_actual_state=float(missing_norm),
        g_highest_j_weight=float(prob[:,gh,:,:].sum()),R_highest_l_weight=float(prob[:,:,rh,:].sum()),
        radius_semantics='positive matrix norm/cone coordinate, NOT spatial extent',
        limits='NEW finite tree H only; explicit sizable angular-cut occupancy, no convergence or compressed Ward claim')
    (ROOT/(P+'band_connection_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(P+'band_connection_raw.npz'),state=x,force=fd.ravel(),ward_correct=lhs-rhs,ward_wrong=wrong)
    print(json.dumps(result,indent=2),flush=True)
    assert result['actual_H_midpoint_error']<1e-10
    assert result['periodic_density_connection_Ward_error']<1e-10
    assert result['omitted_density_connection_error']>.01
    assert result['r_force_derivative_error']<1e-7
    assert missing_norm>1


if __name__=='__main__':main()
