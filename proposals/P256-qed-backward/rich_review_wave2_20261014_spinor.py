"""Independent literal current, magnetic source, rotation and virtual-sector probe."""
import os
for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
import rich_spinor_20261014_model as model

def main():
    sectors={n:model.solve(model.build_sector(number=n)) for n in (0,1,2)}
    s=sectors[1]; h=s['H'];v=s['vectors'];e=s['energies'];g=v[:,:2]
    q=np.array([.43,.67,.29]);pol=q/np.linalg.norm(q)
    r=s['geometry']['positions']
    rho=np.kron(model.lift(np.kron(np.diag(np.exp(1j*r@q)),np.eye(2)),s['states']),np.eye(4))
    j=model.current(s,q,pol)
    ward=np.linalg.norm(h@rho-rho@h-np.linalg.norm(q)*j)
    rotation=[];angular=s['angular']
    for a in range(3):
        rotation.append(dict(axis=a, H_commutator=float(np.linalg.norm(h@angular[a]-angular[a]@h))))
    algebra=[]
    for a,b,c in [(0,1,2),(1,2,0),(2,0,1)]:
        algebra.append(float(np.linalg.norm(angular[a]@angular[b]-angular[b]@angular[a]-1j*angular[c])))
    # Uniform magnetic Bz, symmetric gauge: each straight edge gets B(r_b x r_a)_z/2.
    def magnetic_field(B):
        one=np.zeros((12,12),complex)
        for a in range(6):
            for b in range(6):
                f=np.cross(r[b],r[a])[2]/2
                one[2*a:2*a+2,2*b:2*b+2]=s['orbital'][a,b]*np.exp(1j*B*f)*np.eye(2)
        zero=np.kron(s['orbital'],np.eye(2))
        return h+np.kron(model.lift(one-zero,s['states']),np.eye(4))
    step=1e-5;m=-(magnetic_field(step)-magnetic_field(-step))/(2*step)
    projected=g.conj().T@m@g
    moment=np.linalg.eigvalsh(projected)
    magnetic_splitting=[]
    for B in [-1e-4,1e-4]:
        en=np.linalg.eigvalsh(magnetic_field(B))
        magnetic_splitting.append(dict(B=B, split=float(en[1]-en[0])))
    qin=np.array([0,0,.27]);qout=np.array([.27,0,0]);epsin=np.array([1,1j,0])/np.sqrt(2)
    # Full vs FOUR-level virtual space is physically different, even when projected Clifford algebra is exact.
    full=[];four=[]
    for epsout in (np.array([0,1,0]),np.array([0,0,1])):
        full.append(np.array([[model.scattering(s,a,b,qin,epsin,qout,epsout,.47,.061) for a in range(2)] for b in range(2)]))
        four.append(np.array([[model.scattering(s,a,b,qin,epsin,qout,epsout,.47,.061,virtual=range(4)) for a in range(2)] for b in range(2)]))
    virtual_distance=float(np.linalg.norm(np.array(full)-np.array(four)))
    data=model.charged_spectral(sectors);gs=model.charged_green(data,.59,.061)
    direct=np.zeros((12,12),complex);equal=np.zeros_like(direct)
    for i in range(2):
        a=np.column_stack([model.car_source(s,sectors[2],mode,True)@v[:,i] for mode in range(12)])
        b=np.column_stack([model.car_source(s,sectors[0],mode,False)@v[:,i] for mode in range(12)])
        z=.59+.061j
        direct+=(a.conj().T@solve((z+e[i])*np.eye(len(sectors[2]['H']))-sectors[2]['H'],a)+(b.conj().T@solve((z-e[i])*np.eye(len(sectors[0]['H']))+sectors[0]['H'],b)).T)/2
        equal+=(a.conj().T@a+(b.conj().T@b).T)/2
    result=dict(law=s['law'],parameters=s['parameters'],ward_operator_error=float(ward),rotation=rotation,angular_algebra_errors=algebra, magnetic_projected_eigenvalues=moment.tolist(), magnetic_splitting=magnetic_splitting, full_vs_four_virtual_distance=virtual_distance,full_current_contact_rate=float(np.sum(abs(np.array(full))**2)), four_virtual_rate=float(np.sum(abs(np.array(four))**2)), spectral_vs_direct_G_error=float(np.linalg.norm(gs-direct)),CAR_equal_time_error=float(np.linalg.norm(equal-np.eye(12))), scope='Literal six-site Peierls current and orbital magnetic derivative; no supplied primitive spin magnetic term, no g2 claim. Angular operators do not generate an exact spatial SO3 law. Complete N0/N2 receivers are finite CAR charge sectors, not relativistic positrons. External probes do not establish propagating photon polarizations.')
    Path(__file__).with_name('rich_review_wave2_20261014_spinor_corrected_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
