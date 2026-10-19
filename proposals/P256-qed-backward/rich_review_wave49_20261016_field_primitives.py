"""Independent Hermite-quadrature consumer of finite-q field products/exponentials."""
from pathlib import Path
from math import factorial
import json
import numpy as np
from scipy.linalg import expm
from scipy.special import eval_hermite
import rich_spatial_frame_photon_20261016_fields as fields

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    keep=fields.basis(2);target=fields.basis(3);coeff=np.array([.31,-.27,.17,.39,-.21,.13,.23,-.19])
    x,w=np.polynomial.hermite.hermgauss(64);w=w/np.sqrt(np.pi)
    def independent(axes):
        out=np.ones((len(target),len(keep)),complex)
        for a,c in enumerate(coeff):
            count=axes.count(a)
            poly=np.array([eval_hermite(n,x)/np.sqrt(2**n*factorial(n)) for n in range(4)])
            mat=np.einsum('ix,x,jx->ij',poly,w*np.exp(1j*c*x)*(1j*x)**count,poly)
            out*=mat[np.array(target)[:,a,None],np.array(keep)[None,:,a]]
        return out
    errors={};images={}
    for axes in [(),(0,),(4,),(0,4),(4,0),(0,0)]:
        got=fields.displacement(target,keep,coeff) if not axes else fields.displacement_derivative(target,keep,coeff,axes)
        exact=independent(axes);key=','.join(map(str,axes)) or 'value';errors[key]=float(np.max(abs(got-exact)));images[key]=got
    Q=[fields.primitive(keep,keep,a,'q').toarray() for a in range(8)]
    full=fields.displacement(keep,keep,coeff);wrong=expm(1j*sum(c*q for c,q in zip(coeff,Q)))
    compressed_error=float(np.linalg.norm(full-wrong))
    Pe=[fields.primitive(target,keep,a,'p') for a in range(8)]
    Pi=[fields.primitive(keep,keep,a,'p') for a in range(8)]
    kinetic=sum((v.conj().T@v).toarray() for v in Pe)/2
    wrong_kinetic=sum((v@v).toarray() for v in Pi)/2
    assert max(errors.values())<1e-12,errors
    assert compressed_error>.01 and np.linalg.norm(kinetic-wrong_kinetic)>1
    np.savez_compressed(ROOT/(P+'field_primitives_raw.npz'),coefficients=coeff,field_basis=np.array(keep),target_basis=np.array(target),full_displacement=full,wrong_compressed_exponential=wrong,kinetic=kinetic,wrong_compressed_kinetic=wrong_kinetic,**{'image_'+k.replace(',','_'):v for k,v in images.items()})
    result=dict(kind='OWN_HERMITE_QUADRATURE_OF_ACTIVE48_FIELD_API',errors=errors,wrong_exponential_after_projection_error=compressed_error,wrong_kinetic_after_projection_error=float(np.linalg.norm(kinetic-wrong_kinetic)),field_dimension=len(keep),limits='All eight real coordinates with full total shells2/3. This tests pure unbounded field kernels, not the active48 solved matter-field law or photon existence.')
    (ROOT/(P+'field_primitives_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
