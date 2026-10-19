"""NEW full-state mobile physical experiment, source propagator and receiver."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, expm_multiply, gmres, norm
from rich_frame_photon_20261014_model import Parameters
from rich_common_mobile_20261014_model import Mobile, MobileParameters


def expect(op,psi):
    return float(np.vdot(psi,op@psi).real)


def ground(h):
    rng=np.random.default_rng(71014)
    v=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0])
    e,u=eigsh(h,k=2,which='SA',v0=v,tol=2e-10,maxiter=18000)
    ix=np.argsort(e)
    return e[ix],u[:,ix],float(np.linalg.norm(h@u[:,ix[0]]-e[ix[0]]*u[:,ix[0]]))


def resolve(h,z,columns):
    a=z*sp.eye(h.shape[0],format='csr')-h
    answers=[];res=[]
    for v in columns.T:
        x,info=gmres(a,v,rtol=2e-9,atol=1e-11,restart=100,maxiter=600)
        if info:raise RuntimeError(('full resolvent',info))
        answers.append(x);res.append(float(np.linalg.norm(a@x-v)))
    return np.column_stack(answers),res


def evolve(h,psi,t):
    return expm_multiply(-1j*t*h,psi,traceA=-1j*t*h.diagonal().sum())


def pulse(op,psi,strength):
    return expm_multiply(-1j*strength*op,psi,traceA=-1j*strength*op.diagonal().sum())


def receiver(m,psi):
    axis=np.array([.31,-.49,.815]);axis/=np.linalg.norm(axis)
    sm=sum(axis[a]*sum(s[a] for s in m.spin) for a in range(3))
    vals,vec=eigh(sm)
    spin=[vec[:,vals>0]@vec[:,vals>0].conj().T,vec[:,vals<0]@vec[:,vals<0].conj().T]
    field=m.coordinates[0]*.6-m.coordinates[1]*.8
    matrix=psi.reshape(m.nfield,m.ns,64)
    probabilities=[]
    for cell in range(2):
        cellop=m.number[cell]
        for ps in spin:
            for sign in (1,-1):
                selected=matrix[field*sign>=0 if sign==1 else field*sign>0]
                op=cellop@ps
                probabilities.append(float(np.einsum('fmk,mn,fnk->',selected.conj(),op,selected).real))
    return probabilities


def snapshot(m,psi):
    return dict(norm=float(np.vdot(psi,psi).real),cell=[expect(x,psi) for x in m.N],
                position=[expect(x,psi) for x in m.X],spin=[expect(x,psi) for x in m.S],
                mode_current=[expect(x,psi) for x in m.J],field=[expect(x,psi) for x in m.A],
                electric=[expect(x,psi)/m.mass for x in m.P],energy=expect(m.H,psi),
                energy_components=dict(electric=expect(m.E,psi),longitudinal=expect(m.HL,psi),
                                       body=expect(m.Hbody,psi),links=expect(m.H-m.E-m.HL-m.Hbody-m.Hdiag,psi)),
                physical_current_q0=[expect(x,psi) for x in m.current_fourier(np.zeros(3))],
                cm_charge_current=[expect(x,psi) for x in m.cm_current()],
                inclusive_probabilities=receiver(m,psi))


def run(inp,out):
    data=json.loads(inp.read_text())
    p=MobileParameters(base=Parameters(**data['base']),centres=tuple(map(tuple,data['centres'])),cm_hopping=data['cm_hopping'],core_mass=data['core_mass'])
    sectors=[Mobile(p,n) for n in range(3)]
    n0,m,n2=sectors
    e0,u0,r0=ground(n0.H); e1,u1,r1=ground(m.H)
    vacuum=u0[:,0];stationary=u1[:,0]
    state=n0.source_to(m,0)@vacuum; state/=np.linalg.norm(state)
    field_generator=.6*m.P[0]-.8*m.P[1]
    prepared=pulse(field_generator,state,data['field_pulse'])
    event=[];states={};references={}
    for time in data['times']:
        ps=evolve(m.H,prepared,time);ref=evolve(m.H,state,time)
        event.append(dict(time=time,field_pulsed=snapshot(m,ps),unpulsed=snapshot(m,ref)))
        states[str(time)]=ps;references[str(time)]=ref
    final=states[str(data['times'][-1])]
    controls={}
    for label,kwargs in [('all_links_off',dict(links=False)),('current_coupling_off',dict(coupling=False)),('spin_metric_off',dict(metric=False))]:
        c=Mobile(p,1,**kwargs)
        cp=pulse(.6*c.P[0]-.8*c.P[1],state,data['field_pulse'])
        a=evolve(c.H,cp,data['times'][-1]);b=evolve(c.H,state,data['times'][-1])
        controls[label]=dict(pulsed=snapshot(c,a),reference=snapshot(c,b))
    wrong=pulse(m.S[0],state,data['field_pulse'])
    controls['wrong_Noether_spin_preparation']=snapshot(m,evolve(m.H,wrong,data['times'][-1]))
    current_prep=pulse(.6*m.J[0]-.8*m.J[1],state,data['current_pulse'])
    current_final=evolve(m.H,current_prep,data['times'][-1])
    q=np.array(data['wavevector']);rho=m.density_fourier(q);j=m.current_fourier(q)
    continuity=norm(m.H@rho-rho@m.H+sum(q[a]*j[a] for a in range(3)))
    step=2e-4
    derivatives=[]
    for a in range(2):
        v=np.zeros(2);v[a]=step
        hp,hm=m.shifted(v),m.shifted(-v)
        derivatives.append(dict(first=float(norm((hp-hm)/(2*step)+m.J[a])),second=float(norm((hp+hm-2*m.H)/step**2-m.K[a][a]))))
    uniform=[]
    for a in range(3):
        v=np.zeros(3);v[a]=step
        derivative=-(m.shifted(uniform=v)-m.shifted(uniform=-v))/(2*step)
        uniform.append(float(norm(derivative-m.current_fourier(np.zeros(3))[a])))
    cm_continuity=[float(norm(m.cm_current()[a]+1j*(m.H@m.X[a]-m.X[a]@m.H))) for a in range(3)]
    creators=[m.source_to(n2,i) for i in range(4)]
    annih=[n0.source_to(m,i).getH() for i in range(4)]
    car=[];charge_ward=[]
    for i in range(4):
        for j0 in range(4):
            car.append(float(norm(creators[i].getH()@creators[j0]+annih[j0].getH()@annih[i]-(m.I if i==j0 else 0))))
        form=np.exp(-1j*np.dot(p.centres[i//2],q)-p.base.core_width**2*np.dot(q,q)/2)
        charge_ward.append(float(norm(n2.density_fourier(q)@creators[i]-creators[i]@rho+form*creators[i])))
    plus=np.column_stack([c@stationary for c in creators]);minus=np.column_stack([c@stationary for c in annih])
    source_G=[]; field_D=[];residuals=[]
    for real,imag in data['frequencies']:
        z=complex(real,imag)
        gp,rp=resolve(n2.H,e1[0]+z,plus)
        gm,rm=resolve(-n0.H,z-e1[0],minus)
        source_G.append(dict(z=[real,imag],addition=encode(plus.conj().T@gp),removal=encode(minus.conj().T@gm)))
        columns=np.column_stack([a@stationary for a in m.A])
        xp,rrp=resolve(m.H,e1[0]+z,columns)
        xm,rrm=resolve(-m.H,z-e1[0],columns)
        field_D.append(dict(z=[real,imag],retarded=encode(columns.conj().T@xp-(columns.conj().T@xm).T)))
        residuals+=rp+rm+rrp+rrm
    f=m.geom['frame'];rawP=f['P']
    raw_gauss=max(float(np.linalg.norm(g@rawP)) for g in f['G'])
    raw_source=max(float(np.linalg.norm((np.eye(56)-rawP@rawP.conj().T)@c@rawP)) for c in f['raw_source'])
    gram=np.column_stack([a@stationary for a in m.J])
    coordinate=np.column_stack([a@stationary for a in m.A])
    result=dict(kind='NEW_ENTIRE_MOBILE_FRAME_PHOTON_COMMON_H',parameters=asdict(p),dimensions=[x.size for x in sectors],matter_dimensions=[x.ns for x in sectors],
                ground_energies=[e0.tolist(),e1.tolist()],ground_residuals=[r0,r1],source_CAR_max=max(car),source_charge_Ward=charge_ward,
                raw_local_Gauss=raw_gauss,raw_local_source_leakage=raw_source,physical_halfspin=float(norm(sum(s@s for s in m.S)-.75*m.I)),
                continuum_continuity=float(continuity),mode_derivatives=derivatives,uniform_A_current_derivative=uniform,
                field_preparation_CM_commutator=float(norm(field_generator@m.N[0]-m.N[0]@field_generator)),
                current_Gram_eigenvalues=np.linalg.eigvalsh(gram.conj().T@gram).tolist(),coordinate_Gram_eigenvalues=np.linalg.eigvalsh(coordinate.conj().T@coordinate).tolist(),
                cm_current_continuity=cm_continuity,
                event=event,controls=controls,current_insertion_event=snapshot(m,current_final),
                source_G=source_G,field_D=field_D,resolvent_residual_max=max(residuals),
                source_G_convention='addition_ab=<c_a R2 c_bdag>; raw removal_ab=<c_adag R0 c_b>. Canonical electron G_ab=addition_ab+removal_ba; no combined G is silently supplied.',
                total_charge_sectors=[float(x.density_fourier(np.zeros(3))[0,0].real) for x in sectors],
                scope='Two fixed translation cells and two global Gaussian collective profiles; no free COM, local causality, asymptotic photons, Maxwell B2, Lorentz mass or g2 claim. All64 supplier and complete mobile N0/1/2 states retained.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),initial=state,field_prepared=prepared,final=final,reference=references[str(data['times'][-1])],current_final=current_final,stationary=stationary,vacuum=vacuum,plus_sources=plus,minus_sources=minus)
    print(json.dumps({k:result[k] for k in ('dimensions','continuum_continuity','source_CAR_max','physical_halfspin','resolvent_residual_max')}))
    print(json.dumps(dict(final=result['event'][-1],controls=result['controls'])))


def encode(a):
    return dict(real=a.real.tolist(),imag=a.imag.tolist())


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
