"""Radial one-dimensional quadrature, exact SO3 tensor structure for both modes."""
import numpy as np


def shell_connections(points,radius,width,epsilon):
    """Sphere averages of A(X+b n) and b n cross A, retaining full TE/TM.

    Only scalar radial integrals are quadrature approximated. No fixed
    Cartesian direction grid can leak rotational anisotropy into the law.
    """
    r=np.linalg.norm(points,axis=1);z,w=np.polynomial.legendre.leggauss(64)
    s2=r[:,None]**2+radius**2+2*radius*r[:,None]*z[None,:]
    f=np.exp(-s2/(2*width**2));f0=f@(w/2);fz=f@(w*z/2);fzz=f@(w*z*z/2)
    perpendicular=(f0-fzz)/2;xx=points[:,:,None]*points[:,None,:]
    alpha_scalar=f0+radius*fz/r
    alpha_te=np.array([alpha_scalar[:,None]*np.cross(np.eye(3)[a],points)/2 for a in range(3)]).transpose(1,0,2)
    beta_scalar=radius/2*(r*fz+radius*(f0-perpendicular))
    beta_tensor=radius/2*(-fz/r-radius*(fzz-perpendicular)/r**2)
    beta_te=beta_scalar[:,None,None]*np.eye(3)[None,:,:]+beta_tensor[:,None,None]*xx
    tm_scalar=(f*(1-s2/(2*width**2)))@(w/2)+radius**2*perpendicular/(2*width**2)
    tm_tensor=(f0+2*radius*fz/r+radius**2*(fzz-perpendicular)/r**2)/(2*width**2)
    alpha_tm=tm_scalar[:,None,None]*np.eye(3)[None,:,:]+tm_tensor[:,None,None]*xx
    beta_tm_scalar=radius/r*((f*(1-s2/(2*width**2)))@(w*z/2))-radius**2*perpendicular/(2*width**2)
    beta_tm=np.array([beta_tm_scalar[:,None]*np.cross(points,np.eye(3)[a]) for a in range(3)]).transpose(1,0,2)
    metrics=np.array([epsilon*np.pi**1.5*width**5/4]*3+[epsilon*5*np.pi**1.5*width**3/8]*3)
    alpha=np.concatenate([alpha_te,alpha_tm],axis=1)/np.sqrt(metrics)[None,:,None]
    beta=np.concatenate([beta_te,beta_tm],axis=1)/np.sqrt(metrics)[None,:,None]
    return alpha,beta,metrics
