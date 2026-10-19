"""Fresh physical LEFT and full-product audit of active47 pure declarations."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
import rich_local_frame_spatial_carrier_20261016_model as local
import rich_spatial_frame_photon_20261016_fields as fields

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    basis=local.field_basis(2);Q,QQ,PP=local.field_moments(basis)
    ref=[tuple(v) for v in basis];ext=fields.basis(3)
    fullQ=np.array([fields.primitive(ref,ref,a,'q').toarray() for a in range(8)])
    qext=[fields.primitive(ext,ref,a,'q') for a in range(8)]
    pext=[fields.primitive(ext,ref,a,'p') for a in range(8)]
    qerror=max(np.linalg.norm(Q[a]-fullQ[a]) for a in range(8))
    qqerror=max(np.linalg.norm(QQ[a,b]-(qext[a].conj().T@qext[b]).toarray()) for a in range(8) for b in range(8))
    pperror=max(np.linalg.norm(PP[a]-(pext[a].conj().T@pext[a]).toarray()) for a in range(8))
    coeff=np.array([[.13,-.21,.31,.17,-.29,.11,.27,-.19]])
    de=np.linalg.norm(local.displacement(basis,coeff)[0]-fields.displacement(ref,ref,coeff[0]))
    g=local.angular((0,.5,1,1.5,2));spin=local.geometry()['spin']
    D=np.array([expm(-1j*a*spin[2])@expm(-1j*b*spin[1])@expm(-1j*c*spin[2]) for a,b,c in g['points']])
    wave=g['F'];w=g['weights'];L=[a.toarray() for a in g['L']]
    operators=np.einsum('qi,q,qms,qj->msij',wave.conj(),w,D,wave,optimize=True)
    comm=[]
    for a in range(3):
        actual=np.einsum('ij,msjk->msik',L[a],operators)-np.einsum('msij,jk->msik',operators,L[a])
        expected=-np.einsum('mn,nsij->msij',spin[a],operators)
        comm.append(float(np.linalg.norm(actual-expected)))
    parity=np.array([(-1)**round(2*j) for j,m,n in g['labels']])
    seam=float(np.linalg.norm(parity[None,None,:,None]*operators+operators*parity[None,None,None,:]))
    # Full source multiplication is unitary at each g. P D†D P precedes projection.
    compressed=sum(operators[mu,0].conj().T@operators[mu,0] for mu in range(8))
    debt=np.eye(len(parity))-compressed
    eig=np.linalg.eigvalsh(debt)
    assert max(comm)<1e-9 and seam<1e-9
    assert min(eig)>-1e-10 and max(eig)>.1
    assert max(qerror,qqerror,pperror,de)<1e-12
    result=dict(kind='OWN_ACTIVE47_FULL_MULTIPLET_LOCAL_OPERATOR_AUDIT',full_g_dimension=len(parity),field_dimension=len(basis),field_kernel_errors=dict(Q=float(qerror),QQ=float(qqerror),PP=float(pperror),displacement=float(de)),physical_LEFT_source_commutator_errors=comm,local_central_sewing_error=seam,source_angular_exterior_eigenvalue_range=[float(min(eig)),float(max(eig))],source_angular_exterior_trace=float(np.trace(debt).real),limits='Pure active47 operators only; complete j0,.5,1,1.5,2 and full beforecut field shell2 products. Positive source exterior at top j is retained as debt, not a CAR failure of uncut D(g)c or localization gate. No evolved carrier/combined-H verdict from this primitive audit.')
    np.savez_compressed(ROOT/(P+'local_primitives_raw.npz'),g_labels=np.array(g['labels']),source_multipliers=operators,source_exterior_debt=debt,field_basis=basis,Q=Q,QQ=QQ,PP=PP,parity=parity)
    (ROOT/(P+'local_primitives_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
