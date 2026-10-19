"""Rectangular primitive images: extend g/R/field BEFORE applying same law.
Finite-q exponentials have infinite images; these expanded shells are lower
bounds on omitted image, not a finite-shell equality or convergence certificate.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
import rich_spatial_frame_photon_20261016_model as m
import rich_spatial_frame_photon_20261016_fields as fields
import rich_spatial_frame_photon_20261016_kernel as kernel
ROOT=Path(__file__).resolve().parent
P='rich_spatial_frame_photon_20261016_'


def image(model,state,g_level=2,R_level=2,field_cut=2):
    p=model.parameters;number=model.matter.sector.number
    G=m.band.angular(tuple(np.arange(number%2/2,g_level+number%2/2+.1,1)))
    R=m.projection.material(R_level);rot=m.rotations(R['points'])
    gi=[G['labels'].index(v) for v in model.matter.g['labels']]
    ri=[R['labels'].index(v) for v in model.R['labels']]
    fb=fields.basis(field_cut);ni=len(fb);nm=len(R['labels']);ng=len(G['labels'])
    fi=[fb.index(v) for v in model.basis]
    shape=(model.nf,ng,model.nr,nm,ni)
    x=state.reshape(model.shape);out=np.zeros(shape,complex)
    Fr=R['F'][:,ri]
    gaunt=R['gaunt'].reshape(nm,nm,-1)[:,ri,:].reshape(nm*len(ri),-1)
    keep=np.flatnonzero(np.any(abs(gaunt)>1.e-15,axis=0))
    gaunt=gaunt[:,keep];forward=R['forward'][keep]
    def mult(values):
        values=np.asarray(values);s=values.shape[1:]
        out=(gaunt@(forward@values.reshape(len(values),-1))).reshape((nm,len(ri))+s)
        return out.transpose((0,)+tuple(range(2,out.ndim))+(1,))
    emr=np.eye(nm)[:,ri];emf=np.eye(ni)[:,fi]
    # Base acts only F,g,r and is unchanged; evaluate in old space then embed.
    base=(model.base@x.reshape(-1,model.nfield)).reshape(model.shape)
    for a,b in enumerate(gi):
        for c,d in enumerate(ri):out[:,b,:,d,fi]=base[:,a,:,c,:].transpose(2,0,1)
    middle=fields.basis(field_cut+1)
    pp=sum((fields.primitive(middle,fb,a,'p').conj().T@fields.primitive(middle,model.basis,a,'p')).toarray() for a in range(8))
    py=np.einsum('fgrma,ba->fgrmb',x,pp,optimize=True)/(2*p['cavity_inertia'])
    for a,b in enumerate(gi):
        for c,d in enumerate(ri):out[:,b,:,d,:]+=py[:,a,:,c,:]
    positions=np.einsum('tab,sb->tsa',rot,(model.matter.geo['sites']-.5)*p['material_scale'])
    conn=np.array([kernel.geometry(xx,model.modes)['connection'] for xx in positions])
    connQ=np.einsum('fs,tsla->ftla',model.Q,conn,optimize=True)
    qt=np.array([fields.primitive(middle,fb,a,'q').toarray() for a in range(8)])
    qr=np.array([fields.primitive(middle,model.basis,a,'q').toarray() for a in range(8)])
    qq=np.einsum('lei,mej->lmij',qt.conj(),qr,optimize=True)
    qcross=[fields.primitive(fb,model.basis,a,'q').toarray() for a in range(8)]
    blockbase=np.kron(R['casimir'].toarray()@emr,emf)
    for f,cq in enumerate(connQ):
        block=blockbase.astype(complex).copy()
        for a in range(3):
            for lam in range(8):
                M=mult(cq[:,lam,a]);L=R['L'][a].toarray();lr=model.R['L'][a].toarray()
                block-=np.kron(L@M+M@lr,qcross[lam])
        coefficients=np.einsum('tla,tma->tlm',cq,cq,optimize=True)
        block+=np.einsum('ilmj,lmab->iajb',mult(coefficients),qq,optimize=True).reshape(nm*ni,model.nm*model.nfield)
        y=(x[f].reshape(-1,model.nm*model.nfield)@block.T).reshape(model.ng,model.nr,nm,ni)/(2*p['material_inertia'])
        for a,b in enumerate(gi):out[f,b]+=y[a]
    # Rectangular exact coherent field displacement, independently extended g/R.
    for axis in range(3):
        for source in range(8):
            target=source^(1<<(2-axis))
            if target<source:continue
            gamma=model.matter.geo['gamma'][axis][target,source]
            if not abs(gamma):continue
            phi=kernel.segment(positions[:,source],positions[:,target],model.modes)[0]
            one=np.zeros((8,8),complex);one[target,source]=gamma
            hop=model.matter.sector.bilinear(one).tocoo()
            for ell in range(3):
                coeff=rot[:,ell,:]@(-2*np.cos(p['k']))
                for reverse in (False,True):
                    disp=fields.displacement(fb,model.basis,phi if reverse else -phi)
                    rf=mult(coeff[:,None,None]*disp).transpose(0,1,3,2).reshape(nm*ni,model.nm*model.nfield)
                    y=(x.reshape(-1,model.nm*model.nfield)@rf.T).reshape(model.nf,model.ng,model.nr,nm,ni)
                    g=G['O'][ell][axis].toarray()[:,gi]
                    y=np.einsum('ij,fjrma->firma',g,y,optimize=True)
                    rows,cols,values=(hop.col,hop.row,hop.data.conj()) if reverse else (hop.row,hop.col,hop.data)
                    for row,col,value in zip(rows,cols,values):out[row]+=value*y[col]
    compressed=out[:,gi,:,:,:][:,:,:,ri,:][...,fi]
    discrepancy=float(np.linalg.norm(compressed.reshape(-1)-model.apply(state)))
    exterior=out.copy()
    for a,b in enumerate(gi):
        for c,d in enumerate(ri):exterior[:,b,:,d,fi]=0
    record=dict(target_shape=shape,g_level=g_level,R_level=R_level,field_cut=field_cut,compression_identity_error=discrepancy,expanded_H_norm=float(np.linalg.norm(out)),omitted_image_lower_bound=float(np.linalg.norm(exterior)),claim='finite expanded-shell lower bound; nonpolynomial Peierls has infinite R/field image')
    return record,out,exterior


def main():
    result={}
    for number,name in [(4,'neutral'),(5,'addition'),(3,'removal')]:
        model=m.Model(number)
        source=P+'paired_neutral.npz' if number==4 else P+'paired_'+name+'.npz'
        raw=np.load(ROOT/source);state=raw['end' if number==4 else 'baseline']
        record,out,exterior=image(model,state)
        record.update(packet_source=source,time=.04)
        result[name]=record
        np.savez_compressed(ROOT/(P+name+'_primitive_images.npz'),expanded_H_image=out,exterior_image=exterior,unbounded_generator_F=model.F,unbounded_generator_connections=model.conn,unbounded_generator_charge=model.Q,modes=model.modes,primitive_field_basis=np.asarray(model.basis),primitive_q_image=model.qe)
        (ROOT/(P+'images_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(name,json.dumps(record),flush=True)


if __name__=='__main__':main()
