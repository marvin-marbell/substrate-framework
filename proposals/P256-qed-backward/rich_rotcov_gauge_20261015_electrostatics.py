"""Full shell/supplier/central-ion Coulomb cross density, same soft kernel."""
import numpy as np
from scipy.linalg import eigh
from rich_physical_noether_20261014_model import oscillator


def shell_kernel(distance,radius,softening):
    """Sphere average 1/sqrt(|R+b n|²+d²), stable also at R=0."""
    plus=np.sqrt((distance+radius)**2+softening**2)
    minus=np.sqrt((distance-radius)**2+softening**2)
    # Rationalized difference of square roots cancels R exactly.
    return 2/(plus+minus)

def scalar_angular_integral(operator,angular):
    """Exact Haar average in retained SO3 irreps, including multiplicities.

    This evaluates the scalar Coulomb integral; it neither removes a current
    vector nor adds a J penalty. No truncation of angular irreps is performed.
    """
    dimension=len(operator);casimir=sum(a@a for a in angular)
    jz=np.diag(angular[2]).real;lower=angular[0]-1j*angular[1];result=np.zeros_like(operator,complex)
    for j in range(int(round(max(jz)))+1):
        selected=np.flatnonzero(abs(jz-j)<1e-7)
        if not len(selected):continue
        values,vectors=eigh(casimir[np.ix_(selected,selected)])
        keep=abs(values-j*(j+1))<1e-7
        if not np.any(keep):continue
        highest=np.zeros((dimension,int(np.sum(keep))),complex);highest[selected]=vectors[:,keep]
        multiplet=[highest]
        for m in range(j,-j,-1):multiplet.append(lower@multiplet[-1]/np.sqrt(j*(j+1)-m*(m-1)))
        average=sum(u.conj().T@operator@u for u in multiplet)/(2*j+1)
        result+=sum(u@average@u.conj().T for u in multiplet)
    return result


def cross_density(p,osc,g,supplier_quadratures):
    """Return one geometry x supplier matrix for each breathing radius."""
    supplier_osc=oscillator(p['supplier_cut'])
    ng=len(osc['basis']);ns=len(supplier_quadratures[0]['u']);identity=np.eye(ns)
    density_g=np.einsum('xa,xb,x->xab',g['polys'].conj(),g['polys'],g['weights'])
    angular_g=[g['u'].conj().T@a@g['u'] for a in osc['L']]
    result=[]
    for radius in p['radii']:
        ion=p['ion_charge']*g['integrate'](shell_kernel(np.linalg.norm(g['points'],axis=1),radius,p['coulomb_softening']))
        operator=np.kron(ion,np.eye(ns*ns)).astype(complex)
        for flavour,q in enumerate(supplier_quadratures):
            displacement=g['points'][:,None,:]-q['points'][None,:,:]
            kernel=shell_kernel(np.linalg.norm(displacement,axis=2),radius,p['coulomb_softening'])
            density_s=np.einsum('xa,xb,x->xab',q['polys'].conj(),q['polys'],q['weights'])
            pair=np.einsum('xab,xy,ycd->acbd',density_g,kernel,density_s,optimize=True).reshape(ng*ns,ng*ns)
            angular_s=[q['u'].conj().T@a@q['u'] for a in supplier_osc['L']]
            angular=[np.kron(angular_g[a],identity)+np.kron(np.eye(ng),angular_s[a]) for a in range(3)]
            scalar=scalar_angular_integral(pair,angular)
            if flavour==0:operator-=np.kron(scalar,identity)
            else:operator-=np.einsum('acbd,ij->aicbjd',scalar.reshape(ng,ns,ng,ns),identity).reshape(ng*ns*ns,ng*ns*ns)
        result.append(p['charge']**2*operator/(4*np.pi*p['epsilon']))
    return result
