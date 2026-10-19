"""Same-Gaussian spatial density/current, including charged mechanical convection.
These are full physical probe products before R/field cuts, not dipole-current guesses.
"""
from itertools import product
import numpy as np
from scipy import sparse
from rich_spatial_colour_em_20261015_model import FactorOperator
from rich_spatial_colour_em_20261015_fields import phase_values,integrate_RQ,phase_kernel,mechanical_kernels


def density(model,name,wavevector):
    base=model['base'];op=base['operators'][name];rotor=base['rotor'];field=model['field'];p=model['parameters'];k=np.asarray(wavevector,float)
    positions=np.einsum('qia,sa->qsi',rotor['R'],base['geometry']['vertices'])
    values=np.exp(-1j*np.einsum('qsi,i->qs',positions,k))*np.exp(-p['charge_sigma']**2*(k@k)/2)
    average=(values[:,0]+values[:,1])/2;difference=(values[:,1]-values[:,0])/2
    kernels=[sparse.kron(rotor['integrate'](value),sparse.eye(field['dimension'])) for value in [average,difference]]
    return FactorOperator(op['Q'].shape[0],base['scalar'],rotor['dimension']*field['dimension'],[(op['Q'],kernels[0],False),(op['density_delta'],kernels[1],False)],hermitian=False)


def spatial_current(model,name,wavevector,component,include_mechanical=True):
    base=model['base'];op=base['operators'][name];rotor=base['rotor'];field=model['field'];p=model['parameters'];geometry=base['geometry'];k=np.asarray(wavevector,float)
    nrq=rotor['dimension']*field['dimension'];terms=[];gaussian=np.exp(-p['charge_sigma']**2*(k@k)/2)
    for edge,route in enumerate(geometry['quadrature']):
        points=np.einsum('qia,na->qni',rotor['R'],route['points']);dl=np.einsum('qia,na->qni',rotor['R'],route['dl'])
        line=gaussian*np.sum(np.exp(-1j*np.einsum('qni,i->qn',points,k))*dl[:,:,component],axis=1)
        coefficient=model['coefficients']['values'][:,edge];weight=geometry['weights'][edge]
        for charge,matrices in op['hops'][edge]:
            plus=phase_values(coefficient,charge,field);minus=phase_values(coefficient,-charge,field)
            for spin in [None,0,1,2]:
                multiplier=line if spin is None else line*geometry['tangents'][:,edge,spin]
                kp=integrate_RQ(rotor,plus*multiplier[:,None,None]);km=integrate_RQ(rotor,minus*multiplier[:,None,None])
                if spin is None:
                    terms.extend([(-1j*charge*p['scalar_coupling']*weight*matrices[0],kp,True),(1j*charge*p['scalar_coupling']*weight*matrices[0].conj().T,km,True)])
                else:
                    terms.extend([(charge*p['vector_velocity']*weight*matrices[spin+1],kp,False),(charge*p['vector_velocity']*weight*matrices[spin+1].conj().T,km,False)])
    if not include_mechanical:
        return FactorOperator(op['Q'].shape[0],base['scalar'],nrq,terms,hermitian=False)
    # General external A_i(r)=delta exp(-ik.r) gives the mechanical current too.
    positions=np.einsum('qia,sa->qsi',rotor['R'],geometry['vertices'])
    multiplier=gaussian*np.exp(-1j*np.einsum('qsi,i->qs',positions,k))
    probe=np.stack([np.cross(np.eye(3)[axis],positions)[:,:,component]*multiplier for axis in range(3)],axis=2)
    w=[(probe[:,0]+probe[:,1])/2,(probe[:,1]-probe[:,0])/2]
    c=[model['connections']['average'],model['connections']['difference']]
    for charge,values in zip([op['Q'],op['density_delta']],w):
        kernel=np.zeros((nrq,nrq),complex)
        for axis in range(3):
            r=rotor['integrate'](values[:,axis]).toarray();l=rotor['L'][axis].toarray()
            kernel+=np.kron((l@r+r@l)/(2*p['orientation_inertia']),np.eye(field['dimension']))
        terms.append((charge,kernel,False))
    charge_products=[op['Q']@op['Q'],op['Q']@op['density_delta'],op['density_delta2']]
    for index,(left,right) in enumerate([(0,0),(0,1),(1,1)]):
        kernel=np.zeros((nrq,nrq),complex)
        for mu in range(6):
            values=np.sum(c[left][:,:,mu]*w[right],axis=1)
            if left!=right:values+=np.sum(c[right][:,:,mu]*w[left],axis=1)
            r=rotor['integrate'](values).toarray()/p['orientation_inertia']
            kernel-=np.kron(r,field['Q'][mu])
        terms.append((charge_products[index],kernel,False))
    return FactorOperator(op['Q'].shape[0],base['scalar'],nrq,terms,hermitian=False)


def mechanical_operator(model,name,derivative=()):
    """Actual covariant-rotor H term, or its physical profile-probe derivative."""
    base=model['base'];op=base['operators'][name];rotor=base['rotor'];field=model['field'];p=model['parameters']
    key=(tuple(derivative),None)
    if key not in model['mechanical_cache']:
        model['mechanical_cache'][key]=mechanical_kernels(rotor,field,model['connections'],p['orientation_inertia'],derivative=derivative)
    kernels=model['mechanical_cache'][key]
    charges=[op['Q'],op['density_delta'],op['Q']@op['Q'],op['Q']@op['density_delta'],op['density_delta2']]
    sign=-1 if len(derivative)==1 else 1
    terms=[(sign*a,b,False) for a,b in zip(charges,kernels['linear']+kernels['quadratic'])]
    return FactorOperator(op['Q'].shape[0],base['scalar'],rotor['dimension']*field['dimension'],terms)


def no_hop_law(model,name,full_h):
    """Remove material transfers ONLY; keep live charged-rotor EM/Coulomb/colour."""
    base=model['base'];op=base['operators'][name];rotor=base['rotor'];field=model['field'];p=model['parameters']
    nm=op['Q'].shape[0];nr=rotor['dimension'];ne=field['dimension'];identity=sparse.eye(nr*ne,format='csr');matter_identity=sparse.eye(nm,format='csr')
    coulomb=((model['V_self']+model['V_cross'])*(op['Q']@op['Q'])+(model['V_self']-model['V_cross'])*op['density_delta2'])/4
    bare_rotor=rotor['L2']/(2*p['orientation_inertia'])
    terms=[(op['electric']-p['plaquette']*op['B']+coulomb,identity,False),(matter_identity,sparse.kron(bare_rotor,sparse.eye(ne),format='csr')+sparse.kron(sparse.eye(nr),field['H_E'],format='csr'),False)]
    terms.extend(mechanical_operator(model,name).terms)
    return FactorOperator(nm,base['scalar'],nr*ne,terms,include_scalar=full_h.include_scalar)
