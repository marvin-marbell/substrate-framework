"""Underlying commuting charge/phi/field-amplitude PVMs, compressed without postselection."""
from math import factorial, gamma
import numpy as np
from scipy.linalg import expm
from rich_spatial_colour_em_20261015_fields import phase_values


def field_halfline(field,axis):
    basis=field['basis'];size=len(basis);effect=np.zeros((size,size))
    for row,left in enumerate(basis):
        for col,right in enumerate(basis):
            if any(left[a]!=right[a] for a in range(6) if a!=axis):continue
            m,n=left[axis],right[axis]
            hm=np.polynomial.hermite.herm2poly(np.eye(m+1)[m]);hn=np.polynomial.hermite.herm2poly(np.eye(n+1)[n])
            polynomial=np.polynomial.polynomial.polymul(hm,hn)
            integral=sum(coefficient*gamma((power+1)/2)/2 for power,coefficient in enumerate(polynomial))
            effect[row,col]=integral/np.sqrt(2**(m+n)*factorial(m)*factorial(n)*np.pi)
    return [np.eye(size)-effect,effect]


def field_action(state,matter_dimension,nphi,nrotor,matrix):
    x=state.reshape(matter_dimension,nphi,nrotor,matrix.shape[1])
    return np.einsum('ab,mfrb->mfra',matrix,x,optimize=True).ravel()


def joint_receiver(state,matter_effects,scalar,rotor,field,field_axis=2):
    nm=matter_effects[0].shape[0];nf=len(scalar['grid']);nr=rotor['dimension'];ne=field['dimension']
    x=state.reshape(nm,nf,nr,ne);field_effects=field_halfline(field,field_axis);probability=np.zeros((len(matter_effects),2,2))
    for charge,effect in enumerate(matter_effects):
        charged=(effect@x.reshape(nm,-1)).reshape(x.shape)
        for branch,field_effect in enumerate(field_effects):
            applied=np.einsum('ab,mfrb->mfra',field_effect,charged,optimize=True)
            for side in range(2):
                selected=(scalar['grid']<0) if side==0 else (scalar['grid']>=0)
                probability[charge,side,branch]=float(np.vdot(x[:,selected],applied[:,selected]).real)
    return probability


def drive(model,name,state):
    field=model['field'];base=model['base'];nm=base['operators'][name]['Q'].shape[0];nf=len(base['scalar']['grid']);nr=base['rotor']['dimension'];p=model['parameters']
    vector=np.array(p['drive_profile'],float);generator=sum(coefficient*coordinate for coefficient,coordinate in zip(vector,field['Q']))
    unitary=expm(-1j*p['field_kick']*generator)
    kicked=field_action(state,nm,nf,nr,unitary)
    continuum_projection=phase_values(vector[None,:],-p['field_kick'],field)[0]
    projected=field_action(state,nm,nf,nr,continuum_projection)
    return kicked,dict(field_only_unitarity_error=float(abs(np.linalg.norm(kicked)-np.linalg.norm(state))),continuous_drive_omitted_norm_squared=float(np.linalg.norm(state)**2-np.linalg.norm(projected)**2),cut_unitary_vs_compressed_continuous_drive=float(np.linalg.norm(kicked-projected)),meaning='Actual finite field-only unitary exp(-i kick P Qdrive P), NOT a body-conditioned pulse. The exact compressed continuous-Q unitary is NONunitary because it loses higher shells; use it only to measure omitted preparation support, never renormalize/postselect it into the physical event.')
