"""Finite-q real transverse plane profiles; hbar=charge unit=1, lab lengths.

Mode rows are (qx,qy,qz,ex,ey,ez,phase); f=e*cos(q.x+phase).
A=sum a_l f_l, Hkin=(L-sum a_l Lambda_l)^2/(2I).
For net charges Q=ions-fermions, P=sum Q F, U=exp(i a.P).
U†Hvelocity U has Pi+P, edge phi-(Fj-Fi), Lambda-d_R P.
No dynamics, state, cavity energy or charge convention is inferred here.
"""
import numpy as np


def default_modes(qin=(0.,0.,.7),qout=(.6,0.,.2)):
    rows=[]
    for q in (qin,qout):
        q=np.asarray(q,float);n=q/np.linalg.norm(q)
        seed=np.eye(3)[np.argmin(abs(n))];e=np.cross(n,seed);e/=np.linalg.norm(e)
        for p in (e,np.cross(n,e)):
            for phase in (0.,-np.pi/2):rows.append(np.r_[q,p,phase])
    return np.asarray(rows)


def profiles(points,modes):
    """Return (..., mode, xyz); arbitrary real mode amplitudes in e allowed."""
    x=np.asarray(points);m=np.asarray(modes)
    return np.cos(x@m[:,:3].T+m[:,6])[...,None]*m[:,3:6]


def profile_gradients(points,modes):
    """Return (..., mode, vector_component, derivative_component)."""
    x=np.asarray(points);m=np.asarray(modes)
    return -np.sin(x@m[:,:3].T+m[:,6])[...,None,None]*m[:,3:6,None]*m[:,None,:3]


def _entire(z):
    """C=int_0^1 exp(i z t)dt and dC/dz, stable at q=0."""
    z=np.asarray(z,float);small=abs(z)<1.e-4;w=np.where(small,1.,z)
    c=np.expm1(1j*w)/(1j*w)
    dc=np.exp(1j*w)/w+1j*np.expm1(1j*w)/w**2
    cs=np.zeros(z.shape,complex);ds=np.zeros(z.shape,complex)
    term=np.ones(z.shape,complex)
    for n in range(14):
        cs+=term/(n+1)
        ds+=1j*term/(n+2)
        term*=1j*z/(n+1)
    return np.where(small,cs,c),np.where(small,ds,dc)


def segment(start,end,modes):
    """Exact straight-path integral and endpoint gradients; shapes (...,m),(...,m,3)."""
    a=np.asarray(start);b=np.asarray(end);m=np.asarray(modes);d=b-a
    k=d@m[:,:3].T;ed=d@m[:,3:6].T;c,dc=_entire(k)
    phase=np.exp(1j*(a@m[:,:3].T+m[:,6]));v=phase*ed*c
    da=phase[...,None]*(-m[:,3:6]*c[...,None]+ed[...,None]*m[:,:3]*(1j*c-dc)[...,None])
    db=phase[...,None]*(m[:,3:6]*c[...,None]+ed[...,None]*m[:,:3]*dc[...,None])
    return np.real(v),np.real(da),np.real(db)


def path_integrals(vertices,modes):
    """Exact sum along declared piecewise straight paths (...,vertex,xyz)."""
    x=np.asarray(vertices)
    return segment(x[...,:-1,:],x[...,1:,:],modes)[0].sum(axis=-2)


def geometry(positions,modes,reference=(0.,0.,0.)):
    """Per-site F,dF/dx,Lambda,d_R F; rotation is physical LEFT delta x=axis cross x.

    F,gradF,connection,drotation have shapes (site,mode),(site,mode,3),
    (site,mode,axis),(site,mode,axis). Charge sums are deliberately caller-owned.
    reference is fixed in lab under material rotations, normally zero.
    """
    x=np.asarray(positions);f=profiles(x,modes)
    F,_,grad=segment(np.asarray(reference),x,modes)
    velocity=np.cross(np.eye(3)[:,None,:],x[None,:,:]).transpose(1,0,2)
    connection=np.einsum('sld,sad->sla',f,velocity)
    drotation=np.einsum('sld,sad->sla',grad,velocity)
    return dict(F=F,gradF=grad,profiles=f,connection=connection,drotation=drotation,
                residual_connection=connection-drotation)


def charge_kernel(positions,charges,modes,edges=None,paths=None,reference=(0.,0.,0.)):
    """Arbitrary scalar net charges; returns polarization and mechanical coefficients.

    Edge convention is path i->j for fermion hop i->j of charge -1;
    use actual charge times (phi-(Fj-Fi)) in its phase. Paths override straight edges.
    """
    g=geometry(positions,modes,reference);Q=np.asarray(charges)
    out=dict(g,P=np.einsum('s,sl->l',Q,g['F']),
             Lambda=np.einsum('s,sla->la',Q,g['connection']),
             dP=np.einsum('s,sla->la',Q,g['drotation']),
             residual_Lambda=np.einsum('s,sla->la',Q,g['residual_connection']))
    if edges is not None:
        ij=np.asarray(edges,int);x=np.asarray(positions)
        phi=segment(x[ij[:,0]],x[ij[:,1]],modes)[0] if paths is None else path_integrals(paths,modes)
        out['phi']=phi;out['residual_flux']=phi-(g['F'][ij[:,1]]-g['F'][ij[:,0]])
    return out
