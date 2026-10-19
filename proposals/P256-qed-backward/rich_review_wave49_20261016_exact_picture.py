"""Own Fourier-Hermite quadrature and fresh-state exact-electric picture countercases.
No producer CLI, producer states, Model constructor, or cached carrier copy.
"""
from math import factorial
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from scipy.special import eval_hermite
import rich_exact_electric_picture_20261016_fields as api
import rich_spatial_frame_photon_20261016_fields as frozen
import rich_spatial_frame_photon_20261016_kernel as kernel
import rich_gauge_band_frame_20261016_model as band
from rich_review_wave49_20261016_spatial_benchmark import integral,skew

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def fourier_weyl(labels,c,time,inertia,impulse):
    """Integrate shifted physical momentum wavefunctions; no Laguerre/BCH formula."""
    nodes,weights=np.polynomial.hermite.hermgauss(96)
    maximum=max(map(max,labels));result=np.ones((len(labels),len(labels)),complex)
    for axis in range(8):
        u=nodes;c0=c[axis];tau=time/inertia
        table=np.empty((maximum+1,maximum+1),complex)
        for n in range(maximum+1):
            for m in range(maximum+1):
                integrand=eval_hermite(n,u+c0/2)*eval_hermite(m,u-c0/2)*np.exp(1j*tau*c0*(u+impulse[axis]))
                table[n,m]=(1j**n)*((-1j)**m)*np.exp(-c0*c0/4)*np.dot(weights,integrand)/np.sqrt(np.pi*2.**(n+m)*factorial(n)*factorial(m))
        result*=np.array([[table[n[axis],m[axis]] for m in labels] for n in labels])
    return result


def momentum_moment(labels,axis,power):
    nodes,weights=np.polynomial.hermite.hermgauss(64);out=np.zeros((len(labels),len(labels)),complex)
    for i,n in enumerate(labels):
        for j,m in enumerate(labels):
            if any(n[a]!=m[a] for a in range(8) if a!=axis):continue
            a,b=n[axis],m[axis]
            out[i,j]=(1j**a)*((-1j)**b)*np.dot(weights,eval_hermite(a,nodes)*eval_hermite(b,nodes)*nodes**power)/np.sqrt(np.pi*2.**(a+b)*factorial(a)*factorial(b))
    return out


def wave_fourier(x,maximum,time,inertia,impulse,derivative=False):
    nodes,weights=np.polynomial.hermite.hermgauss(160);u=np.sqrt(2)*nodes;p=u+impulse
    factor=np.exp(1j*np.outer(x,p)-.5j*time/inertia*p*p)
    if derivative:factor*=1j*p
    return np.array([((-1j)**n)*factor@(weights*eval_hermite(n,u))/(np.pi**.75*np.sqrt(2.**n*factorial(n))) for n in range(maximum+1)]).T


def run():
    modes=kernel.default_modes();cube=band.geometry()
    centres=np.array([[0,0,0],[1.1,.1,.2],[.2,1.,.1],[.1,.3,1.2]])-.35
    body=(centres[:,None,:]+.16*(cube['sites'][None,:,:]-.5)).reshape(32,3)
    R=expm(skew([.27,-.41,.19]));x=body@R.T
    # Physical radial Wilson-source path, not a spatially uniform rephasing.
    c=-integral(np.zeros(3),x[3],modes)
    eps=np.zeros(8);eps[0]=.23;eps[2]=-.17
    time=.3;inertia=2.;records={};arrays={'positions':x,'source_coefficients':c,'impulse':eps}
    rng=np.random.default_rng(490016)
    for cut in (1,2):
        labels=frozen.basis(cut);d=len(labels)
        exact=api.displacement(labels,labels,c,time,inertia,eps)
        independent=fourier_weyl(labels,c,time,inertia,eps)
        errors={'Fourier_momentum_Weyl':float(np.max(abs(exact-independent)))}
        h=2e-5
        for axis in (0,3,4,7):
            e=np.eye(8)[axis]*h
            fd=(fourier_weyl(labels,c+e,time,inertia,eps)-fourier_weyl(labels,c-e,time,inertia,eps))/(2*h)
            errors['coefficient_derivative_'+str(axis)]=float(np.max(abs(fd-api.coefficient_derivative(labels,labels,c,[axis],time,inertia,eps))))
        # LEFT/material direction derivative of actual finite-q source geometry.
        derivative=[]
        for axis in np.eye(3):
            cp=-integral(np.zeros(3),x[3]@expm(h*skew(axis)).T,modes)
            cm=-integral(np.zeros(3),x[3]@expm(-h*skew(axis)).T,modes)
            dc=(cp-cm)/(2*h)
            direct=(fourier_weyl(labels,cp,time,inertia,eps)-fourier_weyl(labels,cm,time,inertia,eps))/(2*h)
            chain=sum(dc[a]*api.coefficient_derivative(labels,labels,c,[a],time,inertia,eps) for a in range(8))
            derivative.append(float(np.max(abs(direct-chain))))
        psi=rng.normal(size=d)+1j*rng.normal(size=d);psi/=np.linalg.norm(psi)
        p=np.array([momentum_moment(labels,a,1) for a in range(8)])
        p2=np.array([momentum_moment(labels,a,2) for a in range(8)])
        means=np.einsum('i,aij,j->a',psi.conj(),p,psi).real
        shifted=means+eps
        H0=sum(p2)/(2*inertia)
        H0W=H0+sum(eps[a]*p[a]/inertia for a in range(8))+np.dot(eps,eps)*np.eye(d)/(2*inertia)
        work=float(np.vdot(psi,(H0W-H0)@psi).real)
        predicted=float(np.dot(eps,means)/inertia+np.dot(eps,eps)/(2*inertia))
        overlap=api.protocol_overlap(labels,labels,np.zeros(8),eps)
        independent_overlap=fourier_weyl(labels,eps,0,inertia,np.zeros(8))
        errors['mixed_protocol_overlap']=float(np.max(abs(overlap-independent_overlap)))
        for axis in range(8):
            errors['physical_Pi_quadrature_'+str(axis)]=float(np.max(abs(p[axis]-frozen.primitive(labels,labels,axis,'p').toarray())))
        q=np.array([frozen.primitive(labels,labels,a,'q').toarray() for a in range(8)])
        badpulse=expm(1j*np.einsum('a,aij->ij',eps,q));bad=badpulse@psi
        badmeans=np.einsum('i,aij,j->a',bad.conj(),p,bad).real
        badfree=expm(-1j*time*H0)@psi
        drift=np.einsum('i,aij,j->a',badfree.conj(),p,badfree).real-means
        wrong_pulse=float(np.linalg.norm((badmeans-means-eps)[[1,3,4,5,6,7]]))
        wrong_overlap=abs(np.vdot(psi,overlap@psi)-np.vdot(psi,psi))
        # Exact free evolution must leave every canonical Pi mean invariant.
        assert max(errors.values())<2e-8 and max(derivative)<2e-8
        assert abs(work-predicted)<2e-14
        assert wrong_pulse>1e-5 and np.linalg.norm(drift)>1e-5 and wrong_overlap>1e-4
        records[str(cut)]=dict(field_dimension=d,errors=errors,material_LEFT_direction_errors=derivative,initial_all8_Pi=means.tolist(),full_W_kicked_all8_Pi=shifted.tolist(),full_W_other_mode_Pi_shift=(shifted-means-eps)[[1,3,4,5,6,7]].tolist(),physical_free_Pi_drift=[0.]*8,full_W_work=work,full_W_work_formula_error=abs(work-predicted),WRONG_projected_kick_other_mode_shift_norm=wrong_pulse,WRONG_projected_free_H_Pi_drift_norm=float(np.linalg.norm(drift)),WRONG_coefficient_overlap_error=float(wrong_overlap),WRONG_time_zero_source_error=float(np.linalg.norm(exact-api.displacement(labels,labels,c,0,inertia,eps))))
        arrays.update({f'cut{cut}_labels':np.array(labels),f'cut{cut}_fresh_coefficients':psi,f'cut{cut}_source_API':exact,f'cut{cut}_source_Fourier':independent,f'cut{cut}_Pi':p,f'cut{cut}_Pi2':p2,f'cut{cut}_protocol_overlap':overlap})
    xs=np.linspace(-4,4,35)
    waves,_=api.wavefunctions(xs,3,time,inertia,.23);independent=wave_fourier(xs,3,time,inertia,.23)
    wave_error=float(np.max(abs(waves-independent)))
    assert wave_error<2e-12
    arrays.update(wave_x=xs,wave_API=waves,wave_Fourier=independent)
    np.savez_compressed(ROOT/(P+'exact_picture_raw.npz'),**arrays)
    result=dict(kind='OWN_FRESH_FIELD_STATES_AND_INDEPENDENT_FOURIER_SOURCE_KERNEL_NOT_COUPLED_BODY_EVOLUTION',source_geometry='32 retained actual-position microsites; independent benchmark geometry, site3 radial Wilson source; no producer state',absolute_time=time,inertia=inertia,records=records,free_Hermite_wave_Fourier_error=wave_error,semantics='Exact full W and U0 are represented by moving protocol basis and transformed observables before cut. Coefficients unchanged does not mean protocol states identical. Coefficient-c derivative is NOT external-A canonical source variation. Physical free Pi drift zeros above are analytic operator identities checked against wrong cut-H countercase, not numerically evolved coupled states.',not_claimed=['New actual47 trajectory','New coupled48 interaction-picture state or field2 convergence','Continuum photon/cavity closure','Projection becomes unitary or no angular exterior'])
    (ROOT/(P+'exact_picture_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
