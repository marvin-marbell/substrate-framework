#!/usr/bin/env python3
"""Noncommuting log resolvents and explicit ORIGINALfield positive-range chart.

Matrices supplied here must be actual immersion crossGrams. A reduced Krylov
consumer is a NUMERICAL APPROXIMATION and must report FULLoperator residuals;
this module does not turn log(QdagAQ) into an exact log of the full operator.
The declared contact transport is ORIGINALfield Pancharatnam: unitary polar
of W(0)dagW(s), W(s)=V(s)(V(s)dagV(s))^-1/2. No null-energy extension is used.
CommonGLOBALneutral lambda is already frozen into V; no vacuum jets added.
"""
from __future__ import annotations
import numpy as np
import scipy.linalg
from numpy.polynomial.legendre import leggauss


def hermitian(a: np.ndarray) -> np.ndarray:
    return (a+a.conj().T)/2


def resolvent_log_jet(a: np.ndarray,ax: np.ndarray,ay: np.ndarray,axy: np.ndarray,
                       order: int=256,tail_span: float=40.) -> dict:
    """Actual noncommuting Frechetlog: BOTH ordered resolvent products."""
    values,frame=np.linalg.eigh(hermitian(a))
    if np.min(values)<=0:raise ArithmeticError('Positive-range A is not strictly positive; no spectralfloor')
    x=frame.conj().T@ax@frame;y=frame.conj().T@ay@frame;xy=frame.conj().T@axy@frame
    lower=float(np.log(values[0])-tail_span);upper=float(np.log(values[-1])+tail_span)
    nodes,weights=leggauss(order);grid=(upper-lower)*nodes/2+(upper+lower)/2;weights=weights*(upper-lower)/2
    jx=np.zeros_like(x);jy=np.zeros_like(y);mixed=np.zeros_like(xy)
    for u,w in zip(grid,weights,strict=True):
        t=np.exp(u);r=1/(values+t);rx=r[:,None]*x*r[None,:];ry=r[:,None]*y*r[None,:]
        jx+=w*t*rx;jy+=w*t*ry
        # R Axy R - R Ay R Ax R - R Ax R Ay R. Matrices do NOT commute.
        mixed+=w*t*(r[:,None]*xy*r[None,:]-(ry@x+rx@y)*r[None,:])
    loga=(frame*np.log(values))@frame.conj().T
    xmin=float(values[0]);tmin=np.exp(lower);tmax=np.exp(upper)
    nx=float(np.linalg.norm(ax,2));ny=float(np.linalg.norm(ay,2));nxy=float(np.linalg.norm(axy,2))
    return dict(log=hermitian(loga),x=hermitian(frame@jx@frame.conj().T),y=hermitian(frame@jy@frame.conj().T),
        xy=hermitian(frame@mixed@frame.conj().T),eigenvalues=values,
        first_tail_operator_bounds=[tmin*nx/xmin**2+nx/tmax,tmin*ny/xmin**2+ny/tmax],
        mixed_tail_operator_bound=tmin*(nxy/xmin**2+2*nx*ny/xmin**3)+nxy/tmax+nx*ny/tmax**2,
        integration=dict(order=order,log_t_bounds=[lower,upper],tail_span=tail_span,
            scope='Tail bounds refer only to supplied positive matrix; quadraturediscretization requires orderrefinement. No fulloperator/Krylov/complement uniformbound implied.'))


def physical_range_jets(a: np.ndarray,cx: np.ndarray,cy: np.ndarray,cxy: np.ndarray,
                        dxy: np.ndarray,ax: np.ndarray,ay: np.ndarray,axy: np.ndarray,
                        spacing: float=.01,order: int=256) -> dict:
    """Originalfield polar immersion, Pancharatnam horizontal current/contact.

    A=VdagV; Cx=VdagVx; Cxy=VdagVxy; Dxy=VxdagVy. Ax,Axy
    are complete derivatives, not pointwise scalarTaylor. CrossGrams retain
    ALLfield/Fock directions even when the supplied numerical source chart is
    finite. Missing fullmidpoint complement must be priced by the caller.
    """
    values,frame=np.linalg.eigh(hermitian(a))
    if np.min(values)<=0:raise ArithmeticError('No positive-range inverse without strictpositivity')
    s=(frame*np.sqrt(values))@frame.conj().T;si=(frame/np.sqrt(values))@frame.conj().T
    sx=scipy.linalg.solve_sylvester(s,s,ax);sy=scipy.linalg.solve_sylvester(s,s,ay)
    sxy=scipy.linalg.solve_sylvester(s,s,axy-sx@sy-sy@sx)
    ix=-si@sx@si;iy=-si@sy@si
    ixy=si@sx@si@sy@si+si@sy@si@sx@si-si@sxy@si
    ox=si@cx@si-sx@si;oy=si@cy@si-sy@si
    ox=(ox-ox.conj().T)/2;oy=(oy-oy.conj().T)/2
    fxy=si@cxy@si+si@cx@iy+si@cy@ix+s@ixy
    sigma=(fxy-fxy.conj().T)/2
    qxy=sigma+(ox@oy+oy@ox)/2
    log= resolvent_log_jet(a,ax,ay,axy,order=order)
    h=-log['log']/spacing;hx=-log['x']/spacing;hy=-log['y']/spacing;hxy=-log['xy']/spacing
    comm=lambda left,right:left@right-right@left
    physical_x=hermitian(hx+comm(ox,h));physical_y=hermitian(hy+comm(oy,h))
    physical_xy=hermitian(hxy+comm(ox,hy)+comm(oy,hx)+comm(sigma,h)+
                         (comm(ox,comm(oy,h))+comm(oy,comm(ox,h)))/2)
    # Horizontal positive-range immersion crossmetric, NOT an arbitrary
    # kernel-energy/log extension. Null-extension contact price is explicit.
    wxy=si@dxy@si+si@cx.conj().T@iy+ix@cy@si+ix@a@iy
    horizontal_xy=wxy-ox.conj().T@oy
    return dict(H=h,canonical_x=hx,canonical_y=hy,canonical_xy=hxy,
        physical_x=physical_x,physical_y=physical_y,physical_xy=physical_xy,
        Omega_x=ox,Omega_y=oy,Pancharatnam_Qxy=qxy,horizontal_range_crossmetric=horizontal_xy,
        ONE_kernel_energy_unit_mixed_contact_price=hermitian(horizontal_xy+horizontal_xy.conj().T),
        resolvent=log,connection='DECLARED ORIGINALfield positive-range Pancharatnam unitary-polar W0dagWs. No zero/nullenergyextension.',
        current_relation='Physical_j=canonical_hj+[Omega_j,H]. Second contact includes sigmaxy, bothfirstconnections, doublecommutators; sourcepulledjets Q(s)m(s) must also be differentiated.')
