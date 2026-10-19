"""Independent NEW continuum-shell/cavity consumer; no preceding CLI replay."""
import os
for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm,solve
import rich_spinor_rot_20261014_model as law

def covariance(m):
    axis=np.array([2.,-1.,3.]);axis/=np.linalg.norm(axis);angle=.43
    k=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R=expm(angle*k);U=law.rotate(m,axis,angle)
    q=np.array([.43,-.37,.61]);ep=np.array([.21,.74,.17]);ep/=np.linalg.norm(ep)
    j=law.current(m,q,ep);jr=law.current(m,R@q,R@ep)
    return dict(finite_H_rotation_error=float(np.linalg.norm(U@m['H']@U.conj().T-m['H'])),finite_current_rotation_error=float(np.linalg.norm(U@j@U.conj().T-jr)))

def observe(p):
    m=law.solve(law.build(p));H=m['H'];e=m['energies'];v=m['vectors']
    source=law.vacuum_sources(m);z=.71+.083j
    direct=source.conj().T@solve(z*np.eye(len(H))-H,source)
    spectral=law.charged_green(m,.71,.083)
    qin=np.array([.21,.34,.49]);qout=np.array([-.31,.42,.18]);ei=np.array([1,1j,0])/np.sqrt(2);eo=np.array([0,1,0])
    ji=law.current(m,qin,ei);jo=law.current(m,qout,eo);co=law.contact(m,qout,eo,qin,ei)
    full=np.array([[law.amplitude(m,a,b,ji,jo,co,.53,.083) for a in range(2)] for b in range(2)])
    four=np.array([[law.amplitude(m,a,b,ji,jo,co,.53,.083,virtual=range(4)) for a in range(2)] for b in range(2)])
    return m,dict(parameters=m['parameters'],dimension=len(H),**covariance(m),G_spectral_vs_direct_error=float(np.linalg.norm(spectral-direct)),vacuum_source_norm_error=float(np.linalg.norm(source.conj().T@source-np.eye(source.shape[1]))),full_KH_rate=float(np.sum(abs(full)**2)),four_virtual_rate=float(np.sum(abs(four)**2)),full_vs_four_amplitude_distance=float(np.linalg.norm(full-four)))

def main():
    bare,b=observe(law.Parameters())
    coupled,c=observe(law.Parameters(coupling=.45,field_cutoff=1))
    # Retain field dimensions but remove lambda interactions for a SAME cavity law contrast.
    dim=len(bare['H']);f=coupled['field'];hf=np.kron(bare['H'],np.eye(coupled['field_dimension']))+np.kron(np.eye(dim),1.15*f['number'])
    g=coupled['vectors'][:,0];e=coupled['energies'][0]
    # Source and full state retain spin and neutral orientor, no single low quartet evolution.
    j=law.current(coupled,np.zeros(3),np.array([1.,0,0]));kick=j@g;kick/=np.linalg.norm(kick)
    values,vectors=np.linalg.eigh(hf);times=[0,.3,.7,1.1];motion=[]
    for t in times:
        state=vectors@(np.exp(-1j*values*t)*(vectors.conj().T@kick))
        motion.append(dict(time=t,energy_removed_coupling=float(np.vdot(state,hf@state).real),survival=float(abs(np.vdot(kick,state))**2)))
    out=dict(bare=b,coupled=c,removed_cavity_coupling_quench=motion,ground_energy_shift=float(coupled['energies'][0]-bare['energies'][0]),scope='Complete angular shells recover actual SO3 current covariance, unlike cubic predecessor. Labelled primitive isotropic3coordinate quantum cavity, not propagating photons or Gauss links. G is vacuum addition-only: no N2,holes,positrons. Quench is prepared state, not isolated-charge scattering or new ground.')
    Path(__file__).with_name('rich_review_wave2_20261014_rot_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
