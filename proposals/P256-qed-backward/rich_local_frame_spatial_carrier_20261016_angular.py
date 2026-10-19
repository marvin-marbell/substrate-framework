"""Oversampled SO3 cubature and exact retained-PW product factorization.
The auxiliary l<=2L basis is EXACT for products of retained l<=L functions,
not an extra truncation of the nonpolynomial finite-q field profile.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from rich_gauge_band_frame_20261016_model import jmat
from rich_spatial_frame_photon_20261016_quadrature import material as retained_material
from rich_frame_gauss_20261014_model import cg


def values(js,points):
    beta,inverse=np.unique(points[:,1],return_inverse=True);blocks=[]
    for j in js:
        m=np.arange(-j,j+1);d=len(m);J=jmat(j)
        small=np.array([expm(-1j*b*J[1]) for b in beta])
        block=np.exp(-1j*points[:,0,None,None]*m[None,:,None])*small[inverse]*np.exp(-1j*points[:,2,None,None]*m[None,None,:])
        blocks.append(np.sqrt(d)*block.reshape(len(points),d*d))
    return np.concatenate(blocks,axis=1)


def rotations(points):
    a,b,c=points.T;ca,sa=np.cos(a),np.sin(a);cb,sb=np.cos(b),np.sin(b);cc,sc=np.cos(c),np.sin(c)
    R=np.empty((len(points),3,3))
    R[:,0,0]=ca*cb*cc-sa*sc;R[:,0,1]=-ca*cb*sc-sa*cc;R[:,0,2]=ca*sb
    R[:,1,0]=sa*cb*cc+ca*sc;R[:,1,1]=-sa*cb*sc+ca*cc;R[:,1,2]=sa*sb
    R[:,2,0]=-sb*cc;R[:,2,1]=sb*sc;R[:,2,2]=cb
    return R


@lru_cache(maxsize=8)
def material(maximum,nangle=24,nbeta=16):
    base=dict(retained_material(maximum,azimuth=nangle,polar=nbeta))
    points=base['points'];weights=base['weights'];F=base['F']
    expanded=values(range(2*maximum+1),points)
    retained=base['labels'];aux=[(j,m,n) for j in range(2*maximum+1) for m in range(-j,j+1) for n in range(-j,j+1)]
    index={x:i for i,x in enumerate(retained)};d=len(retained);gaunt=np.zeros((d*d,len(aux)),complex)
    for k,(ell,a,b) in enumerate(aux):
        for col,(j,m,n) in enumerate(retained):
            for jp in range(abs(int(j)-ell),min(int(j)+ell,maximum)+1):
                row=index.get((jp,m+a,n+b))
                if row is not None:gaunt[row*d+col,k]=np.sqrt((2*j+1)*(2*ell+1)/(2*jp+1))*cg(j,m,ell,a,jp)*cg(j,n,ell,b,jp)
    base.update(F=F,weights=weights,points=points,rotations=rotations(points),gaunt=gaunt,forward=(expanded.conj()*weights[:,None]).T.copy(),quadrature=dict(nangle=nangle,nbeta=nbeta),gram_error=float(np.linalg.norm(F.conj().T@(weights[:,None]*F)-np.eye(d))))
    return base


def project(material,field_values):
    nR=len(material['labels']);nf=field_values.shape[1]
    coefficients=material['forward']@field_values.reshape(len(field_values),nf*nf)
    a=material['gaunt']@coefficients
    return a.reshape(nR,nR,nf,nf).transpose(0,2,1,3).reshape(nR*nf,nR*nf)
