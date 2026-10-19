"""Exact outgoing g/R/field polynomial images of retained packets, no evolution."""
import numpy as np
from scipy import sparse as sp
import rich_coupled_frame_photon_20261016_model as m


def full_image(model,state):
    hm=model.matter;nf,ng,nR,nr=hm.shape;p=model.parameters;level=p['angular_level']
    gs=m.band.angular(tuple(np.arange(hm.sector.number%2/2,level+hm.sector.number%2/2+1.1,1)))
    rs=m.band.angular(tuple(range(level+2)))
    ig=[gs['labels'].index(v) for v in hm.g['labels']];ir=[rs['labels'].index(v) for v in hm.material['labels']]
    eg=sp.csr_matrix(([1.]*ng,(ig,range(ng))),shape=(len(gs['labels']),ng))
    er=sp.csr_matrix(([1.]*nR,(ir,range(nR))),shape=(len(rs['labels']),nR))
    emb=m.band.tensor(sp.eye(nf),eg,er,sp.eye(nr))
    field=m.field_basis(p['field_total_cut']+2)
    ef=sp.csr_matrix(([1.]*model.nfield,([field.index(v) for v in model.basis],range(model.nfield))),shape=(len(field),model.nfield))
    x=state.reshape(model.shape)
    # Nonmultiplicative terms preserve angular labels. Hrelative alone reaches
    # gmax+1,Rmax+1; D_i reaches Rmax+1, D² is rotationally scalar.
    matter=emb@((hm.H-hm.Hrelative)@x)
    for i in range(3):
        for a in range(3):
            for ell in range(3):
                rect=m.band.tensor(hm.sector.bilinear(hm.geo['gamma'][i]),gs['O'][ell][i]@eg,rs['O'][ell][a]@er,sp.eye(nr))
                matter+=(-2*np.cos(p['k'][a]))*(rect@x)
    out=matter@ef.T
    inertia=p['cavity_inertia'];middle=m.field_basis(p['field_total_cut']+1)
    p2=sp.csr_matrix(sum(m.field_operator(field,middle,a,'p')@m.field_operator(middle,model.basis,a,'p') for a in range(3)))
    out+=(emb@x)@p2.T/(2*inertia)
    out+=(emb@(model.D2@x))@ef.T/(2*inertia)
    for a in range(3):
        drect=sum(m.band.tensor(sp.diags(model.body_dipole[:,b]),eg,rs['O'][a][b]@er,sp.eye(nr)) for b in range(3))
        out+=(drect@x)@m.field_operator(field,model.basis,a,'p').T/inertia
    # Orthogonal complement of the original product regulator, not a bound
    # inferred from its conserved projected energy.
    retained=emb.T@out@ef
    projected=(model.H@state).reshape(model.shape)
    total=float(np.linalg.norm(out)**2);inside=float(np.linalg.norm(retained)**2)
    angular_inside=emb.T@out
    angular_out=float(np.linalg.norm(out-emb@angular_inside))
    field_out=float(np.linalg.norm(angular_inside-retained@ef.T))
    return dict(full_H_norm2=total,retained_H_norm2=inside,all_exterior_H_norm=float(np.hypot(angular_out,field_out)),angular_exterior_H_norm=angular_out,field_exterior_H_norm=field_out,compression_identity_error=float(np.linalg.norm(retained-projected)),extended_shape=out.shape,retained_shape=model.shape)
