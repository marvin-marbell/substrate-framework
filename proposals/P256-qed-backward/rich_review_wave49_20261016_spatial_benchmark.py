"""Own finite-q path/PZW/mechanical quantum benchmark using no solved producer state."""
from itertools import product
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
import rich_gauge_band_frame_20261016_model as band
import rich_spatial_frame_photon_20261016_kernel as kernel

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def integral(a,b,modes):
    t,w=np.polynomial.legendre.leggauss(48);t=(t+1)/2;w=w/2
    x=np.asarray(a)[...,None,:]+t[:,None]*(np.asarray(b)-a)[...,None,:]
    f=np.cos(x@modes[:,:3].T+modes[:,6])[...,None]*modes[:,3:6]
    return np.einsum('...tld,...d,t->...l',f,np.asarray(b)-a,w)


def field(cut,n=8):
    return [v for v in product(range(cut+1),repeat=n) if sum(v)<=cut]


def fieldop(target,reference,a,kind):
    out=np.zeros((len(target),len(reference)),complex);lookup={v:i for i,v in enumerate(target)}
    for j,v in enumerate(reference):
        for sign in (-1,1):
            w=list(v);w[a]+=sign;i=lookup.get(tuple(w))
            if i is not None:out[i,j]=np.sqrt((v[a]+(sign>0))/2)*(1j*sign if kind=='p' else 1)
    return out


def run():
    modes=kernel.default_modes();centres=np.array([[0,0,0],[1.1,.1,.2],[.2,1.,.1],[.1,.3,1.2]])-.35
    cube=band.geometry();body=(centres[:,None,:]+.16*(cube['sites'][None,:,:]-.5)).reshape(32,3)
    ions=np.tile(cube['ions'],4);n=ions.copy();n[0]=1;Q=ions-n
    # Include an independent neutral polarization pair, not a collapsed cell point.
    Q[10]+=1;Q[11]-=1
    edges=np.array([[0,10],[10,21],[21,0]])
    x=(expm(skew([.27,-.41,.19]))@body.T).T
    got=kernel.charge_kernel(x,Q,modes,edges)
    F=integral(np.zeros(3),x,modes);phi=integral(x[edges[:,0]],x[edges[:,1]],modes)
    h=2e-6;fd=[]
    for a in np.eye(3):
        xp=x@expm(h*skew(a)).T;xm=x@expm(-h*skew(a)).T
        fd.append((integral(np.zeros(3),xp,modes)-integral(np.zeros(3),xm,modes))/(2*h))
    fd=np.stack(fd,axis=-1)
    endpoint=[]
    for a in np.eye(3):endpoint.append((integral(np.zeros(3),x+h*a,modes)-integral(np.zeros(3),x-h*a,modes))/(2*h))
    endpoints=np.stack(endpoint,axis=-1)
    loop=phi.sum(axis=0);res=got['residual_flux'].sum(axis=0)
    zero=modes.copy();zero[:,:3]=0;z=kernel.charge_kernel(x,Q,zero,edges)
    assert np.linalg.norm(loop)>1e-4 and np.max(abs(loop-res))<1e-12
    assert np.max(abs(got['F']-F))<1e-11 and np.max(abs(got['drotation']-fd))<1e-8
    assert np.max(abs(got['gradF']-endpoints))<1e-8
    assert np.linalg.norm(z['residual_flux'])+np.linalg.norm(z['residual_Lambda'])<1e-11
    # Actual fresh quantum R and eight-coordinate field H, no bare q² energy.
    ang=band.angular((0,1));wave=ang['F'];weights=ang['weights'];L=[v.toarray() for v in ang['L']]
    rotations=[]
    for a,b,c in ang['points']:
        rotations.append(expm(skew([0,0,a]))@expm(skew([0,b,0]))@expm(skew([0,0,c])))
    pos=np.einsum('qij,sj->qsi',rotations,body)
    geom=[kernel.charge_kernel(v,Q,modes) for v in pos]
    polar=np.array([g['P'] for g in geom]);lam=np.array([g['Lambda'] for g in geom]);rla=np.array([g['residual_Lambda'] for g in geom])
    basis=field(1);ext=field(3);d=len(wave[0]);f=len(basis);I=3.1;If=2.4
    ident=np.eye(f);eye=np.eye(d)
    q=[fieldop(basis,basis,a,'q') for a in range(8)];p=[fieldop(basis,basis,a,'p') for a in range(8)]
    qe=[fieldop(ext,basis,a,'q') for a in range(8)];pe=[fieldop(ext,basis,a,'p') for a in range(8)]
    def project(v):return wave.conj().T@(weights[:,None]*v[:,None]*wave)
    Hrot=np.kron(sum(v@v for v in L)/(2*I),ident)
    Hfield=np.kron(eye,sum(v.conj().T@v for v in pe)/(2*If))
    for l in range(8):
        Hfield+=np.kron(project(polar[:,l]),p[l])/If
        for a in range(3):
            r=project(rla[:,l,a]);Hrot-=np.kron(L[a]@r+r@L[a],q[l])/(2*I)
        for k in range(8):
            Hrot+=np.kron(project(np.sum(rla[:,l]*rla[:,k],axis=-1)),qe[l].conj().T@qe[k])/(2*I)
    Hfield+=np.kron(project(np.sum(polar**2,axis=-1)),ident)/(2*If)
    H=Hrot+Hfield
    def source(a,b):
        deriv=np.zeros_like(H);contact=project(np.sum(lam[:,a]*lam[:,b],axis=-1))/I
        for axis in range(3):
            la=project(lam[:,a,axis]);deriv-=np.kron(L[axis]@la+la@L[axis],ident)/(2*I)
        for k in range(8):deriv+=np.kron(project(np.sum(rla[:,k]*lam[:,a],axis=-1)),q[k])/I
        return deriv,np.kron(contact,ident)
    da,caa=source(0,0);db,cab=source(4,0);_,cbb=source(4,4)
    def shifted(a,b):
        # Independent fixed-canonical minimal source: original Lambda, not residual.
        external=a*lam[:,0]+b*lam[:,4]
        out=Hfield+np.kron(sum(v@v for v in L)/(2*I),ident)
        for axis in range(3):
            coefficient=project(external[:,axis])
            out-=np.kron(L[axis]@coefficient+coefficient@L[axis],ident)/(2*I)
        out+=np.kron(project(np.sum(external**2,axis=-1)),ident)/(2*I)
        for l in range(8):
            for axis in range(3):
                coefficient=project(rla[:,l,axis])
                out-=np.kron(L[axis]@coefficient+coefficient@L[axis],q[l])/(2*I)
            out+=np.kron(project(np.sum(rla[:,l]*external,axis=-1)),q[l])/I
            for k in range(8):
                out+=np.kron(project(np.sum(rla[:,l]*rla[:,k],axis=-1)),qe[l].conj().T@qe[k])/(2*I)
        return out
    eps=2e-4
    contactfd=(shifted(eps,eps)-shifted(eps,-eps)-shifted(-eps,eps)+shifted(-eps,-eps))/(4*eps**2)
    # Wrong compressed product is a physical omitted-exterior contact countercase.
    wrong=sum(project(lam[:,0,a])@project(lam[:,0,a]) for a in range(3))/I
    contactdebt=np.linalg.norm(caa-np.kron(wrong,ident))
    state=np.zeros(d*f,complex);state[0]=1
    kick=expm(.19j*np.kron(eye,q[0])+.13j*np.kron(eye,q[4]));state=kick@state
    off=Hfield+np.kron(sum(v@v for v in L)/(2*I),ident)
    U=expm(-.37j*H);Uoff=expm(-.37j*off);final=U@state;foff=Uoff@state
    electric=[np.kron(eye,p[a])+np.kron(project(polar[:,a]),ident) for a in range(8)]
    def observe(v):return dict(norm=float(np.vdot(v,v).real),energy=float(np.vdot(v,H@v).real),physical_E=[float(np.vdot(v,o@v).real) for o in electric],material_LEFT=[float(np.vdot(v,np.kron(o,ident)@v).real) for o in L],physical_current_qin=float(-np.vdot(v,da@v).real))
    feedback=np.linalg.norm(final-foff)
    assert contactdebt>1e-6 and feedback>1e-5
    assert np.linalg.norm(H-H.conj().T)<1e-10
    fixed_current_fd=(shifted(eps,0)-shifted(-eps,0))/(2*eps)
    assert np.linalg.norm(fixed_current_fd-da)<1e-8
    assert np.linalg.norm(contactfd-cab)<1e-6
    # Exact outgoing FIELD image for the retained angular block, not angular closure.
    embedding=np.zeros((len(ext),f),complex)
    for j,v in enumerate(basis):embedding[ext.index(v),j]=1
    middle=field(2)
    qsecond=[[fieldop(ext,middle,l,'q')@fieldop(middle,basis,k,'q') for k in range(8)] for l in range(8)]
    psecond=sum(fieldop(ext,middle,l,'p')@fieldop(middle,basis,l,'p') for l in range(8))
    Himage=np.kron(sum(v@v for v in L)/(2*I),embedding)+np.kron(eye,psecond)/(2*If)
    Himage+=np.kron(project(np.sum(polar**2,axis=-1)),embedding)/(2*If)
    for l in range(8):
        Himage+=np.kron(project(polar[:,l]),pe[l])/If
        for axis in range(3):
            coefficient=project(rla[:,l,axis])
            Himage-=np.kron(L[axis]@coefficient+coefficient@L[axis],qe[l])/(2*I)
        for k in range(8):
            Himage+=np.kron(project(np.sum(rla[:,l]*rla[:,k],axis=-1)),qsecond[l][k])/(2*I)
    outgoing=(Himage@final).reshape(d,len(ext))
    outside=[j for j,v in enumerate(ext) if v not in basis]
    field_exterior=float(np.linalg.norm(outgoing[:,outside]))
    shell_weight=float(sum(np.linalg.norm(final.reshape(d,f)[:,j])**2 for j,v in enumerate(basis) if sum(v)==1))
    def expectation(v,o):return np.vdot(v,o@v)
    # Two time orders use the SAME fresh H, not a borrowed response state.
    t1=.11;t2=.23;V1=expm(-1j*t1*H);V2=expm(-1j*t2*H)
    order1=expectation(state,da@V1@db@V2);order2=expectation(state,db@V1@da@V2)
    result=dict(kind='OWN_EIGHT_REAL_FIELD_FINITE_Q_MECHANICAL_PZW_BENCHMARK',kernel_errors=dict(path=float(np.max(abs(got['F']-F))),rotation_derivative=float(np.max(abs(got['drotation']-fd))),endpoint_derivative=float(np.max(abs(got['gradF']-endpoints))),q0_residual=float(np.linalg.norm(z['residual_Lambda']))),magnetic_loop_flux=loop.tolist(),wrong_all_finite_q_is_pure_gauge_error=float(np.linalg.norm(loop)),full_contact_exterior_debt=float(contactdebt),mixed_contact_fd_error=float(np.linalg.norm(contactfd-cab)),source_feedback_state_difference=float(feedback),initial=observe(state),final=observe(final),wrong_omitted_residual_connection_final=observe(foff),time_orders=[[float(order1.real),float(order1.imag)],[float(order2.real),float(order2.imag)]],dimension=d*f,limits='Fresh quantum R l0,1 and eight real noncompact fields total shell1; full polynomial products before cuts. Fixed 32 microsite charge configuration with ions, NOT mobile CAR32 or integrated47. No photon gap, global SO3 closure, convergence, inertial mass, or localization claim. Field q/p CCR not exact in finite retained shell; no bare restoring frequency. Source variation fixes canonical length variables and fixed PZW basis, retaining ORIGINAL Lambda, not just residual Lambda.')
    result.update(fixed_canonical_current_fd_error=float(np.linalg.norm(fixed_current_fd-da)),field_exterior_H_norm=field_exterior,field_boundary_shell_weight=shell_weight,angular_exterior='Not bounded by this diagnostic; finite-q smooth coefficients have unlimited multipoles.',kick_work=float(np.vdot(state,H@state).real-H[0,0].real),same_H_energy_drift=float(np.vdot(final,H@final).real-np.vdot(state,H@state).real))
    np.savez_compressed(ROOT/(P+'spatial_benchmark_raw.npz'),H=H,Hrot=Hrot,Hfield=Hfield,initial_state=state,final_state=final,off_state=foff,current_in=-da,current_out=-db,contact=caa,mixed_contact=cab,positions=pos,charges=Q,modes=modes,polarization=polar,lambda_original=lam,lambda_residual=rla,loop_flux=loop,field_basis=np.array(basis),initial_kick=kick)
    (ROOT/(P+'spatial_benchmark_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
