"""Independent physical-source and finite-transfer discriminators; no producer replay."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from rich_gauge_band_frame_20261016_model import Model, angular, geometry, tensor
from rich_spatial_gauge_source_20261016_family import quadrature

ROOT=Path(__file__).resolve().parent
P='rich_review_wave45_20261016_'


def source_case():
    """Same declared source37 moving-position action on fresh Band40 material basis."""
    m=Model(5);ref=Model(4)
    x=m.source_from(ref,0,True)@ref.preparation()
    strength=float(np.linalg.norm(x)**2)
    if strength<1e-14:raise ValueError('declared source annihilates neutral preparation')
    x/=np.sqrt(strength)
    x=expm_multiply(-.17j*m.H,x)
    mat=m.material;R,w,_=quadrature(10)
    # Use Band40's actual normalized material wavefunctions, not source37 basis labels.
    a=angular((0,1));F=a['F'];weights=a['weights'];pts=a['points']
    # Reconstruct actual SO3 matrices on precisely its Euler grid.
    from rich_gauge_band_frame_20261016_model import jmat
    pauli=[2*z for z in jmat(.5)];rot=[]
    for aa,bb,cc in pts:
        u=expm(-.5j*aa*pauli[2])@expm(-.5j*bb*pauli[1])@expm(-.5j*cc*pauli[2])
        rot.append([[np.trace(pauli[i]@u@pauli[j]@u.conj().T).real/2 for j in range(3)] for i in range(3)])
    rot=np.array(rot);vertices=m.geo['sites']-.5
    positions=np.einsum('qij,vj->qvi',rot,vertices)
    B=np.array([.31,-.47,.79]);A=np.cross(B,positions)/2
    lam=np.cross(positions,A)
    # Net charges in this law are n-ions; source37 sign convention is separately declared.
    Q=m.sector.q
    values=np.einsum('fv,qva->fqa',Q,lam)
    fstate=x.reshape(m.shape);I=m.parameters['material_inertia']
    L=[z.toarray() for z in mat['L']]
    projected=np.einsum('qi,q,fqa,qj->faij',F.conj(),weights,values,F,optimize=True)
    full2=np.einsum('qi,q,fq,qj->fij',F.conj(),weights,np.sum(values**2,axis=2),F,optimize=True)
    compressed=np.einsum('faij,fajk->fik',projected,projected)
    current=np.zeros_like(fstate);contact=np.zeros_like(fstate);wrong=np.zeros_like(fstate)
    for f in range(m.sector.dimension):
        lin=sum((L[i]@projected[f,i]+projected[f,i]@L[i] for i in range(3)))/(2*I)
        current[f]=np.einsum('ij,gjr->gir',lin,fstate[f])
        contact[f]=np.einsum('ij,gjr->gir',full2[f]/I,fstate[f])
        wrong[f]=np.einsum('ij,gjr->gir',compressed[f]/I,fstate[f])
    # Rotation geometry derivative: dx/dphi = axis cross x, actual pointwise field pullback.
    axis=np.array([.2,.4,-.3]);step=1e-6
    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    fd=(np.einsum('ij,qvj->qvi',expm(step*cross),positions)-np.einsum('ij,qvj->qvi',expm(-step*cross),positions))/(2*step)
    result=dict(kind='FRESH_BAND40_SOURCE37_ACTION_DISCRIMINATOR',dimension=m.dimension,Gauss_error=m.sector.gauss_error,
        mechanical_current_norm=float(np.linalg.norm(current)),contact_norm=float(np.linalg.norm(contact)),
        prematurely_projected_contact_error=float(np.linalg.norm(contact-wrong)),rotation_geometry_derivative_error=float(np.max(abs(fd-np.cross(axis,positions)))),
        charge_convention='Band40 n-ions; source37 action used with this explicitly selected charge sign',
        limits='Diagnostic source action on frozen Band40, not new44 law or a dynamical photon. r is a matrix norm, not these fixed material vertices.')
    np.savez_compressed(ROOT/(P+'prelim_source_raw.npz'),state=x,current=current,contact=contact,wrong_contact=wrong,positions=positions,charges=Q)
    return result


def transfer_case():
    """Two-site/two-link compact cycle PH Hamiltonian with physical Gauss basis."""
    cap=3;beta=1.4;M=5;dt=beta/M;I=2.3;U=4.
    basis=[]
    for mask in range(16):
        n=np.array([(mask>>i)&1 for i in range(4)])
        if n[:2].sum()!=n[2:].sum():continue
        for e0 in range(-cap,cap+1):
            for e1 in range(-cap,cap+1):
                if e0-e1==n[0]-n[2]:basis.append((mask,e0,e1))
    lookup={b:i for i,b in enumerate(basis)};d=len(basis)
    E=np.diag([b[1] for b in basis]);HE=np.diag([(b[1]**2+b[2]**2)/(2*I) for b in basis])
    HF=np.zeros((d,d),complex)
    for col,(mask,e0,e1) in enumerate(basis):
        n=[(mask>>i)&1 for i in range(4)]
        HF[col,col]=sum(2*n[v]+2*n[v+2]-4*n[v]*n[v+2] for v in (0,1))
        for species in (0,1):
            for link,(tail,head) in enumerate(((0,1),(1,0))):
                a=tail+2*species;b=head+2*species
                if n[b] and not n[a]:
                    sign=(-1)**((mask&((1<<b)-1)).bit_count())
                    reduced=mask^(1<<b);sign*=(-1)**((reduced&((1<<a)-1)).bit_count())
                    electric=[e0,e1];electric[link]+=1 if species==0 else -1
                    row=lookup.get((reduced^(1<<a),*electric))
                    if row is not None:HF[row,col]-=sign;HF[col,row]-=sign
    K=expm(-dt*HE);V=expm(-dt*HF);T=K@V
    Z=np.trace(np.linalg.matrix_power(T,M)).real
    # Exact interval Frechet vertices by conjugation: all powers of HF retained.
    V1=1j*(E@V-V@E);V2=-(E@E@V-2*E@V@E+V@E@E)
    Tp=K@V1;Tpp=K@V2
    C=np.array([np.trace(E@np.linalg.matrix_power(T,l)@E@np.linalg.matrix_power(T,M-l))/Z for l in range(M)])
    hess=np.empty(M,complex)
    hess[0]=np.trace(Tpp@np.linalg.matrix_power(T,M-1))/Z
    for l in range(1,M):hess[l]=np.trace(Tp@np.linalg.matrix_power(T,l-1)@Tp@np.linalg.matrix_power(T,M-l-1))/Z
    # Wrong replacement is point-current insertion rather than interval derivative.
    J=-1j*(E@HF-HF@E);point=dt*K@J@V
    wrong=np.array([np.trace(point@np.linalg.matrix_power(T,l-1)@point@np.linalg.matrix_power(T,M-l-1))/Z if l else hess[0] for l in range(M)])
    freq=np.arange(M);lam=4*np.sin(np.pi*freq/M)**2
    ef=np.fft.fft(C);hf=np.fft.fft(hess);wf=np.fft.fft(wrong)
    ward=hf[1:]+lam[1:]*ef[1:]
    # Independent trace perturbation on one interval checks sign/contact.
    eps=2e-5
    def trace(a):
        G=expm(1j*a*E);return np.trace(K@G@V@G.conj().T@np.linalg.matrix_power(T,M-1))/Z
    contactfd=(trace(eps)+trace(-eps)-2*trace(0))/eps**2
    result=dict(kind='OWN_PHYSICAL_COMPACT_CYCLE_FINITE_TRANSFER',dimension=d,beta=beta,M=M,cap=cap,I=I,U=U,
        exact_discrete_Ward_error=float(np.max(abs(ward))),wrong_point_vertex_Ward_error=float(np.max(abs(wf[1:]+lam[1:]*ef[1:]))),
        interval_contact_finite_difference_error=float(abs(contactfd-hess[0])),zero_mode_EE=float(ef[0].real),source_Hessian_zero_mode=float(abs(hf[0])),
        current_contact_sumrule='Hessian n0 vanishes; independent electric zero mode does not. Finite transfer not continuum frequency.',
        limits='Two-site two-parallel-link actual Gauss physical Hamiltonian, Ecap3 regulator; no V64 or continuum thermodynamic conclusion.')
    np.savez_compressed(ROOT/(P+'prelim_transfer_raw.npz'),basis=np.array(basis),HE=HE,HF=HF,T=T,E=E,EE=C,Hessian=hess,wrong_Hessian=wrong,EE_frequency=ef,source_frequency=hf,ward=ward)
    return result


def main():
    result=dict(source=source_case(),transfer=transfer_case())
    (ROOT/(P+'prelim_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
