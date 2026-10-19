"""Own physical two-site transfer and holonomy/winding countercases."""
import itertools
import numpy as np
from scipy.linalg import expm, expm_frechet
import rich_unitary_rotor_photon_20261016_thermal as rotor
import rich_collective_photon_response_20261016_model as model


def trace_jet(B,A,C,AC):
    P=np.eye(len(B[0]),dtype=complex);Pa=np.zeros_like(P);Pb=Pa.copy();Pab=Pa.copy()
    for b,a,c,ac in zip(B,A,C,AC):
        Pab,Pa,Pb,P=b@Pab+a@Pb+c@Pa+ac@P,b@Pa+a@P,b@Pb+c@P,b@P
    return np.trace(Pab)/np.trace(P)-np.trace(Pa)*np.trace(Pb)/np.trace(P)**2


def physical_probe():
    basis=[n for n in range(16) if n.bit_count()==2];index={n:i for i,n in enumerate(basis)}
    hop=np.zeros((6,6));E=np.zeros(6);ons=np.zeros(6)
    for k,n in enumerate(basis):
        n0=(n&1)+((n>>1)&1);E[k]=n0-1;ons[k]=4*(n0-1)**2
        for a,b in ((0,2),(2,0),(1,3),(3,1)):
            if n>>b&1 and not n>>a&1:
                mid=n^(1<<b);sign=(-1)**((n&((1<<b)-1)).bit_count()+(mid&((1<<a)-1)).bit_count())
                hop[index[mid|(1<<a)],k]-=sign
    electric=np.diag(E);J=-1j*(electric@hop-hop@electric);contact=-hop
    beta=.8;rows=[];raw={};I=16.;H=hop+np.diag(E**2/(2*I)+ons)
    for M in (2,4,8,16):
        dt=beta/M;X=-dt*hop;Y=dt*J;Z=-dt*contact;diag=np.diag(np.exp(-dt*(E**2/(2*I)+ons)))
        T=expm(X);Tp=expm_frechet(X,Y,compute_expm=False)
        # Exponential block [X,Y,Z/2;0,X,Y;0,0,X] gives half the second
        # derivative: includes Frechet(exp,X'') AND time-ordered double insertion.
        block=np.zeros((18,18),complex)
        for k in range(3):block[k*6:(k+1)*6,k*6:(k+1)*6]=X
        block[:6,6:12]=Y;block[6:12,12:]=Y;block[:6,12:]=Z/2
        Tpp=2*expm(block)[:6,12:];B=T@diag;A=Tp@diag;AC=Tpp@diag
        ZM=np.trace(np.linalg.matrix_power(B,M));C=np.array([np.trace(electric@np.linalg.matrix_power(B,k)@electric@np.linalg.matrix_power(B,M-k))/ZM for k in range(M)])
        Cn=np.fft.fft(C).real;exact=[];point=[];no_contact=[]
        for n in range(1,M):
            a=np.exp(2j*np.pi*n*np.arange(M)/M)/np.sqrt(M);b=a.conjugate();lam=4*np.sin(np.pi*n/M)**2
            exact.append(-trace_jet([B]*M,a[:,None,None]*A,b[:,None,None]*A,(a*b)[:,None,None]*AC).real/lam)
            pointA=dt*J@B;pointAC=(dt**2*J@J-dt*contact)@B
            point.append(-trace_jet([B]*M,a[:,None,None]*pointA,b[:,None,None]*pointA,(a*b)[:,None,None]*pointAC).real/lam)
            no_contact.append(-trace_jet([B]*M,a[:,None,None]*A,b[:,None,None]*A,np.zeros((M,6,6))).real/lam)
        rows.append(dict(M=M,trace=float(ZM.real),exact_discrete_identity_error=float(np.max(abs(np.array(exact)-Cn[1:]))),
                         wrong_point_vertex_error=float(np.max(abs(np.array(point)-Cn[1:]))),
                         wrong_missing_contact_error=float(np.max(abs(np.array(no_contact)-Cn[1:]))),
                         independent_zero_Matsubara=float(Cn[0]),zero_if_erased_error=float(abs(Cn[0]))))
        raw[f'EE_tau_M{M}']=C;raw[f'EE_n_M{M}']=Cn;raw[f'source_EE_M{M}']=np.r_[np.nan,exact]
    exact_trace=float(np.trace(expm(-beta*H)).real)
    # Own compact Gauss projection, M1, all HS signs summed. This is a new
    # quadrature, not a producer CLI invocation or stored-output replay.
    Q=128;angles=2*np.pi*np.arange(Q)/Q;alpha=np.array(list(itertools.product(angles,repeat=2)))
    K=np.exp(rotor.heat(alpha[:,0]-alpha[:,1],beta)[0]);lam=np.arccosh(np.exp(beta*2));T=expm([[0,beta],[beta,0]])
    projected=0.;no_holonomy=0.;wrong_pair=0.
    for signs in itertools.product((-1,1),repeat=2):
        B=T*np.exp(lam*np.array(signs))[None,:];scalar=np.exp(-beta*4-lam*sum(signs))/4
        det=np.linalg.det(np.eye(2)[None,:,:]+np.exp(-1j*alpha)[:,:,None]*B)
        projected+=float(np.mean(K*abs(det)**2)*scalar)
        no_holonomy+=float(abs(np.linalg.det(np.eye(2)+B))**2*scalar)
        wrong_pair+=float(np.mean(K*det*det).real*scalar)
    physical_M1=float(np.trace(expm(-beta*hop)@np.diag(np.exp(-beta*(E**2/(2*I)+ons)))).real)
    # Compact winding has a material effect for a hot/light free rotor. Its exact
    # E variance is the integer-spectrum trace; discarding images is rejected.
    bfree=12.;Ifree=.2;delta=np.pi*.73;dt=bfree/4
    m=np.arange(-40,41);weight=np.exp(-bfree*m*m/(2*Ifree));freeEE=float(np.sum(m*m*weight)/weight.sum())
    logK,score,second=rotor.heat(np.array(delta),dt,Ifree)
    exactK=np.sum(np.exp(-dt*m*m/(2*Ifree))*np.exp(1j*m*delta)).real
    one_image=np.sqrt(2*np.pi*Ifree/dt)*np.exp(-Ifree*delta**2/(2*dt))
    rng=np.random.default_rng(46001);pos,arcs,D=rotor.geometry(2);th=rng.normal(0,.2,(4,len(arcs)));al=rng.normal(0,.1,len(pos));s=rng.choice([-1,1],(4,len(pos)))
    data=model.path_data(th,al,s,arcs,D,.8);response=model.interval_response(th,al,s,data,pos,arcs,D,.8,qvalues=(np.pi,))
    stable=model.interval_response(th,al,s,data,pos,arcs,D,.8,qvalues=(np.pi,),stabilized=True,compare_step=True)
    # Independent centered source finite difference checks the Gaussian determinant
    # jet, distinct from the physical transfer Ward benchmark above.
    f,_=model.spatial_modes(pos,arcs,(np.pi,));n=1;eps=.0003;finite=0.
    for phase in (np.cos,np.sin):
        a=phase(2*np.pi*n*np.arange(4)/4)/2;logs=[]
        for e in (-eps,0,eps):
            dd=model.path_data(th+e*a[:,None]*f[0].real,al,s,arcs,D,.8)
            logs.append(2*np.linalg.slogdet(np.eye(len(pos))+dd['twist'][:,None]*dd['P'])[1])
        finite+=(logs[2]+logs[0]-2*logs[1])/eps**2+((logs[2]-logs[0])/(2*eps))**2
    image=np.arange(-40,41)
    wideK=np.sqrt(2*np.pi*Ifree/dt)*np.exp(-Ifree*(delta+2*np.pi*image)**2/(2*dt)).sum()
    result=dict(kind='OWN_ACTUAL_PHYSICAL_TRANSFER_HAMILTONIAN_AND_COMPACT_PROJECTOR',beta=beta,I=I,U=4,t=1,
                physical_dimension=6,exact_H_trace=exact_trace,finite_transfer=rows,
                holonomy=dict(Q=Q,projected_trace=projected,physical_M1=physical_M1,error=abs(projected-physical_M1),wrong_omitted_projector=no_holonomy,wrong_hole_pair=wrong_pair),
                winding=dict(beta=bfree,I=Ifree,integer_trace_EE=freeEE,compact_wide_image_error=abs(wideK-exactK),inadequate_five_image_error=abs(np.exp(logK)-exactK),wrong_one_image_error=abs(one_image-exactK),note='Light-rotor adverse control, not V64 parameter replacement; five-image V64 helper is inadequate in THIS control, wide images restore the integer trace.'),
                stable_vs_analytic_Hessian_error=float(np.max(abs(response['source_hessian']-stable['source_hessian']))),
                stable_step_halving_error=float(np.max(abs(stable['derivative_step_difference']))),
                independent_real_source_second_finite_difference=float(finite),
                independent_real_source_derivative_error=float(abs(finite-response['source_hessian'][0,n])),
                max_analytic_source_condition=float(response['max_source_matrix_condition']))
    raw.update(H=H,E=electric,J=J,contact=contact,analytic_source=response['source_hessian'],stable_source=stable['source_hessian'])
    assert max(r['exact_discrete_identity_error'] for r in rows)<1e-9,result
    assert rows[1]['wrong_point_vertex_error']>1e-3,result
    assert rows[1]['wrong_missing_contact_error']>1e-2,result
    assert abs(projected-physical_M1)<1e-8,result
    assert abs(no_holonomy-physical_M1)>.1,result
    assert result['stable_vs_analytic_Hessian_error']<1e-5,result
    assert result['independent_real_source_derivative_error']<1e-5,result
    return result,raw
