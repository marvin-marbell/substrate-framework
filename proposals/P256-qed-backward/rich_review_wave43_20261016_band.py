"""Own source evolution and quantum-field derivative consumer of declared Band40 law."""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from rich_gauge_band_frame_20261016_model import Model, geometry, angular, DEFAULTS

ROOT=Path(__file__).resolve().parent
P='rich_review_wave43_20261016_'


def norm(x):return float(np.linalg.norm(x))


def expect(psi,op):return float(np.vdot(psi,op@psi).real)


def main():
    cfg=json.loads((ROOT/(P+'band_input.json')).read_text())
    p={**DEFAULTS,**cfg['parameters']}
    neutral=Model(4,p);omega=neutral.preparation()
    geo=geometry();spin=geo['spin'];eps=cfg['difference_step']
    # Raw field source c^dag scalars; physical rotation acts on argument g, not c.
    points=np.array([[.23,.61,.19],[1.1,.83,-.31]])
    field_derivative=[];wrong_spin_label=[]
    for a in range(3):
        for alpha,beta,gamma in points:
            D=expm(-1j*alpha*spin[2])@expm(-1j*beta*spin[1])@expm(-1j*gamma*spin[2])
            plus=expm(1j*eps*spin[a])@D
            minus=expm(-1j*eps*spin[a])@D
            derivative=-1j*(plus-minus)/(2*eps)
            field_derivative.append(norm(derivative-spin[a]@D))
            wrong_spin_label.append(norm(derivative))
    # Bare microsite band spin changes charge without electric transport.
    gauss_bad=[];wilson_good=[];wrong_path=[]
    for a in range(3):
        S=neutral.sector.bilinear(spin[a]).tocoo()
        for row,col,value in zip(S.row,S.col,S.data):
            dq=neutral.sector.q[row]-neutral.sector.q[col]
            de=neutral.sector.electric[row]-neutral.sector.electric[col]
            gauss_bad.append(abs(value)*norm(dq))
            wilson_good.append(abs(value)*norm(geo['incidence']@de-dq))
            wrong_path.append(abs(value)*norm(geo['incidence']@(-de)-dq))
    # Literal embedded position Peierls finite-q current, independently assembled.
    sites=geo['sites'];lookup={tuple(s):i for i,s in enumerate(sites)}
    def h(k):
        out=np.zeros((8,8),complex)
        for a in range(3):
            for i,s in enumerate(sites):
                dest=s.copy();dest[a]^=1;j=lookup[tuple(dest)];u=(-1.)**sum(s[:a])
                out[j,i]+=-u*np.exp(-1j*k[a]);out[i,j]+=-u*np.exp(1j*k[a])
        return out
    def vertex(k,q,a):
        out=np.zeros((8,8),complex)
        for i,s in enumerate(sites):
            dest=s.copy();dest[a]^=1;j=lookup[tuple(dest)];u=(-1.)**sum(s[:a]);p=k[a]+q[a]/2
            out[j,i]+=1j*u*np.exp(-1j*p);out[i,j]+=-1j*u*np.exp(1j*p)
        return out
    k=np.array(p['k']);q=np.array(cfg['q'])
    ward=norm(h(k+q)-h(k)-sum(2*np.sin(q[a]/2)*vertex(k,q,a) for a in range(3)))
    wrong_midpoint=norm(h(k+q)-h(k)-sum(2*np.sin(q[a]/2)*vertex(k,np.zeros(3),a) for a in range(3)))
    branches={};arrays={}
    for create in (True,False):
        model=Model(5 if create else 3,p)
        component=cfg['addition_component'] if create else cfg['removal_component']
        B=model.source_from(neutral,component,create)
        raw=B@omega;strength=norm(raw)**2
        if strength<1e-12:raise ValueError('declared source has no weight on occupied reference')
        psi=raw/np.sqrt(strength)
        eom=model.H@raw-B@(neutral.H@omega)
        # ALL genuine field kinetic derivatives of this g-dependent source.
        kin=model.Hg@raw-B@(neutral.Hg@omega)
        electric=model.HE@raw-B@(neutral.HE@omega)
        radial=model.Hradial@raw-B@(neutral.Hradial@omega)
        j2=sum((A@A for A in model.J),sp.csr_matrix(model.H.shape))
        # Own physical electric pulse: changes source phase, not occupation probability at t0.
        pulse=expm_multiply(1j*cfg['pulse']*model.E[cfg['pulse_edge']],psi)
        times=np.array(cfg['times'])
        evolved=np.stack([expm_multiply(-1j*t*model.H,psi) for t in times])
        driven=np.stack([expm_multiply(-1j*t*model.H,pulse) for t in times])
        obs=[model.r,model.E[cfg['pulse_edge']],model.currents[cfg['pulse_edge']],model.Jg[2],model.JR[2],model.force]
        base=np.array([[expect(v,o) for o in obs] for v in evolved])
        event=np.array([[expect(v,o) for o in obs] for v in driven])
        # Omitted g-source kinetic term must affect the actual source EOM, not just a matrix norm.
        missing_eom=eom-kin
        kinetic_energy_shift=float(np.vdot(raw,kin).real/strength)
        source_missing_response=norm(eom-missing_eom)
        # Full-space directional Peierls identity in reduced tree electric coordinates.
        e=model.E[cfg['pulse_edge']];chi=cfg['difference_step']
        test=expm_multiply(1j*chi*e,model.H@expm_multiply(-1j*chi*e,psi))
        testm=expm_multiply(-1j*chi*e,model.H@expm_multiply(1j*chi*e,psi))
        derivative=(test-testm)/(2*chi)
        current_error=norm(derivative+model.currents[cfg['pulse_edge']]@psi)
        Jbalance=norm((model.torque_g[2]+model.torque_R[2])@psi)
        label='addition' if create else 'removal'
        branches[label]=dict(N=model.sector.number,dimension=model.dimension,gauss_error=model.sector.gauss_error,
            boundary_flux=np.unique(model.sector.electric[:,-1]).astype(int).tolist(),source_strength=strength,
            J2=expect(psi,j2),source_energy=expect(psi,model.H),source_EOM_norm=norm(eom),
            omitted_g_kinetic_norm=source_missing_response,g_kinetic_source_energy=kinetic_energy_shift,
            electric_source_derivative_norm=norm(electric),radial_source_derivative_norm=norm(radial),
            current_derivative_error=current_error,angular_torque_balance=Jbalance,
            observables=['r matrixnorm','E tree','physical tree current','Lg_z','LR_z','radial supplier force'],
            base=base.tolist(),electric_pulse=event.tolist(),response=(event-base).tolist(),
            t0_density_change=max(abs(expect(pulse,n)-expect(psi,n)) for n in model.n),
            source_state_class='OWN declared neutral Haar/ion/cone preparation, not producer solution or ground',
            angular_cuts=dict(g=sorted(set(x[0] for x in model.g['labels'])),R=sorted(set(x[0] for x in model.material['labels']))))
        arrays[label+'_base']=base;arrays[label+'_driven']=event;arrays[label+'_source']=raw
        assert abs(branches[label]['J2']-.75)<1e-9
        assert source_missing_response > .01
        assert current_error<1e-7
        assert np.max(abs(event[-1]-base[-1]))>1e-5
    # Truncated S1 Wilson source is noncanonical on a boundary electric state.
    u=np.diag(np.ones(2),-1);boundary=np.array([0.,0.,1.])
    s1_defect=norm((u.conj().T@u-np.eye(3))@boundary)
    result=dict(kind='INDEPENDENT_BAND_FRAME_SOURCE_FIELD_RESPONSE',input=cfg,parameters=p,
        source_scope='NEW finite tree central sewing law, not old full cube or Rotor41 V64 state',
        primitive_physical_action='LEFT(g,R); scalar CAR labels; one field-derived halfspin, not extra halfspin tensor',
        raw_LEFT_derivative_error=max(field_derivative),wrong_no_field_derivative=max(wrong_spin_label),
        bare_bandS_gauss_defect=max(gauss_bad),unitary_Wilson_gauss_defect=max(wilson_good),wrong_Wilson_gauss_defect=max(wrong_path),
        midpoint_Ward_error=ward,wrong_q0_vertex_error=wrong_midpoint,S1_boundary_CAR_defect=s1_defect,
        branches=branches,limits=['fixed finite magnetic-cell tree; no independent EM loop degrees',
            'angular and radial Galerkin cuts; source image exact only on initial Haar angular state',
            'finite-time nonstationary source experiment, not mass, stability or bulk photon',
            'integer staggered ions break assumed simple C; addition/removal not asserted conjugate energies'])
    np.savez_compressed(ROOT/(P+'band_raw.npz'),times=times,**arrays)
    (ROOT/(P+'band_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    assert ward<1e-12 and wrong_midpoint>.001
    assert max(wilson_good)<1e-12 and max(gauss_bad)>.1 and max(wrong_path)>.1
    assert s1_defect==1.


if __name__=='__main__':main()
