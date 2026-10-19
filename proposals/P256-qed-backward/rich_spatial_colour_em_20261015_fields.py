"""Actual 3D TE/TM fields, same-Gaussian charge smearing and pre-cut field products.
Oscillator shells are a rotationally complete BASIS, never a magnetic oscillator law.
"""
from itertools import product
from math import factorial
import numpy as np
from scipy import sparse
from scipy.special import erf, eval_genlaguerre


def electric_gram(width,epsilon):
    return np.diag([epsilon*np.pi**1.5*width**5/4]*3+[epsilon*5*np.pi**1.5*width**3/8]*3)


def profiles(points,width,epsilon,sigma=0.):
    """Gaussian convolution of both curl families; normalize by ORIGINAL electric Gram."""
    points=np.asarray(points,float);effective=np.sqrt(width**2+sigma**2)
    amplitude=(width**2/(width**2+sigma**2))**2.5
    r2=np.sum(points*points,axis=-1);f=amplitude*np.exp(-r2/(2*effective**2))
    te=np.stack([f[...,None]*np.cross(np.eye(3)[a],points)/2 for a in range(3)],axis=-2)
    tm=f[...,None,None]*((1-r2/(2*effective**2))[...,None,None]*np.eye(3)+points[...,None,:]*points[..., :,None]/(2*effective**2))
    return np.concatenate([te,tm],axis=-2)/np.sqrt(np.diag(electric_gram(width,epsilon)))[...,None]


def gaussian_green(distance,sigma,epsilon):
    """V=erf(r/(2sigma))/(4pi epsilon r), including its analytic r=0 limit."""
    distance=np.asarray(distance,float);result=np.empty_like(distance)
    np.divide(erf(distance/(2*sigma)),4*np.pi*epsilon*distance,out=result,where=distance!=0)
    result=np.where(distance==0,1/(4*np.pi**1.5*epsilon*sigma),result)
    return result


def field_shell(cut,scale):
    basis=[n for n in product(range(cut+1),repeat=6) if sum(n)<=cut]
    extended=[n for n in product(range(cut+3),repeat=6) if sum(n)<=cut+2]
    index={n:i for i,n in enumerate(extended)};selected=[index[n] for n in basis];annihilators=[]
    for axis in range(6):
        rows=[];cols=[];values=[]
        for column,occupation in enumerate(extended):
            if occupation[axis]:
                target=list(occupation);target[axis]-=1
                rows.append(index[tuple(target)]);cols.append(column);values.append(np.sqrt(occupation[axis]))
        annihilators.append(sparse.csr_matrix((values,(rows,cols)),shape=(len(extended),)*2,dtype=complex))
    coordinates=[scale*(a+a.conj().T)/np.sqrt(2) for a in annihilators]
    momenta=[-1j*(a-a.conj().T)/(scale*np.sqrt(2)) for a in annihilators]
    project=lambda matrix:matrix.tocsr()[selected][:,selected].toarray()
    q=[project(x) for x in coordinates];pi=[project(x) for x in momenta]
    qq=[[project(x@y) for y in coordinates] for x in coordinates]
    kinetic=project(sum(x@x/2 for x in momenta))
    angular=[]
    for axis in range(3):
        angular.append(project(sum(-1j*(annihilators[3*f+(axis+1)%3].conj().T@annihilators[3*f+(axis+2)%3]-annihilators[3*f+(axis+2)%3].conj().T@annihilators[3*f+(axis+1)%3]) for f in range(2))))
    parity=np.diag([(-1)**sum(n) for n in basis]);boundary=np.diag([float(sum(n)==cut) for n in basis])
    return dict(basis=basis,dimension=len(basis),Q=q,Pi=pi,QQ=qq,H_E=kinetic,F=angular,C=parity,boundary=boundary,scale=scale,cut=cut)


def phase_values(coefficients,q,field):
    """EXACT continuous-Q <m|exp(i q c.Q)|n>, not exp(i q c.PQP)."""
    coefficients=np.asarray(coefficients,float);s=q*field['scale']*coefficients
    basis=field['basis'];output=np.empty((len(coefficients),len(basis),len(basis)),complex)
    gaussian=np.exp(-np.sum(s*s,axis=1)/4)
    for row,m in enumerate(basis):
        for col,n in enumerate(basis):
            value=gaussian.astype(complex)
            for axis,(left,right) in enumerate(zip(m,n)):
                low,high=min(left,right),max(left,right)
                value=value*np.sqrt(factorial(low)/factorial(high))*(1j*s[:,axis]/np.sqrt(2))**(high-low)*eval_genlaguerre(low,high-low,s[:,axis]**2/2)
            output[:,row,col]=value
    return output


def route_coefficients(geometry,rotor,p):
    body=[]
    for route in geometry['quadrature']:
        values=profiles(route['points'],p['em_width'],p['epsilon'],p['charge_sigma'])
        body.append(np.einsum('nmi,ni->m',values,route['dl']))
    body=np.array(body)
    rotated=np.concatenate([np.einsum('qia,ea->qei',rotor['R'],body[:,start:start+3]) for start in [0,3]],axis=2)
    return dict(body=body,values=rotated,raw_electric_gram=electric_gram(p['em_width'],p['epsilon']))


def integrate_RQ(rotor,values):
    # The FULL multiplication product lives in values BEFORE either R or Q cut.
    matrix=np.einsum('qr,q,qef,qs->resf',rotor['basis'],rotor['weights'],values,rotor['basis'],optimize=True)
    dimension=rotor['dimension']*values.shape[1]
    return matrix.reshape(dimension,dimension)


def phase_kernel(rotor,geometry,coefficients,field,edge,q,derivative=(),spin_axis=None,external=None):
    c=coefficients['values'][:,edge];values=phase_values(c,q,field)
    factor=np.ones(len(c),complex)
    if external is not None:factor*=np.exp(1j*q*(c@external))
    for axis in derivative:factor*=1j*q*c[:,axis]
    if spin_axis is not None:factor*=geometry['tangents'][:,edge,spin_axis]
    return integrate_RQ(rotor,values*factor[:,None,None])


def endpoint_connections(geometry,rotor,p):
    points=np.einsum('qia,sa->qsi',rotor['R'],geometry['vertices'])
    values=profiles(points,p['em_width'],p['epsilon'],p['charge_sigma'])
    # Centre convection of isotropic Gaussian blobs. No invented internal spinning cloud.
    connection=np.cross(points[:,:,None,:],values).transpose(0,1,3,2)
    return dict(average=(connection[:,0]+connection[:,1])/2,difference=(connection[:,1]-connection[:,0])/2,site=connection)


def mechanical_kernels(rotor,field,connections,inertia,external=None,derivative=()):
    """(L-rho r cross A)^2 products, with QmuQnu projected BEFORE field cut."""
    nf=field['dimension'];identity=np.eye(nf);external=np.zeros(6) if external is None else np.asarray(external,float)
    q=[field['Q'][i]+external[i]*identity for i in range(6)]
    qq=[[field['QQ'][i][j]+external[i]*field['Q'][j]+external[j]*field['Q'][i]+external[i]*external[j]*identity for j in range(6)] for i in range(6)]
    families=[connections['average'],connections['difference']];size=rotor['dimension']*nf
    linear=[np.zeros((size,size),complex) for _ in range(2)];quadratic=[np.zeros((size,size),complex) for _ in range(3)]
    for family,c in enumerate(families):
        for axis,mu in product(range(3),range(6)):
            r=rotor['integrate'](c[:,axis,mu]).toarray();l=rotor['L'][axis].toarray()
            matrix=(l@r+r@l)/(2*inertia)
            if not derivative:linear[family]-=np.kron(matrix,q[mu])
            elif len(derivative)==1 and derivative[0]==mu:linear[family]-=np.kron(matrix,identity)
    for index,(left,right) in enumerate([(0,0),(0,1),(1,1)]):
        for mu,nu in product(range(6),repeat=2):
            values=np.sum(families[left][:,:,mu]*families[right][:,:,nu],axis=1)
            if left!=right:values+=np.sum(families[right][:,:,mu]*families[left][:,:,nu],axis=1)
            r=rotor['integrate'](values).toarray()/(2*inertia)
            if not derivative:matrix=qq[mu][nu]
            elif len(derivative)==1:
                k=derivative[0];matrix=(q[nu] if k==mu else 0*identity)+(q[mu] if k==nu else 0*identity)
            else:
                k,l=derivative;matrix=((k==mu and l==nu)+(k==nu and l==mu))*identity
            quadratic[index]+=np.kron(r,matrix)
    return dict(linear=linear,quadratic=quadratic)
