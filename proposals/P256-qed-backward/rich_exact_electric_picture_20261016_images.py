"""Stream full primitive images BEFORE each coefficient cut.
Hfull and moving-basis connection H0 are separate. Only H_I outgoing is the
moving-Galerkin residual. Expanded nonpolynomial shells give lower bounds.
"""
import gc
import json
from pathlib import Path
from typing import Any
import numpy as np
import rich_exact_electric_picture_20261016_model as m
import rich_exact_electric_picture_20261016_fields as f
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_picture_20261016_'


def rectangular_projection(R,reference_indices):
    nt=len(R['labels']);nr=len(reference_indices)
    gaunt=R['gaunt'].reshape(nt,nt,-1)[:,reference_indices,:].reshape(nt*nr,-1)
    keep=np.flatnonzero(np.any(gaunt!=0,axis=0));gaunt=gaunt[:,keep];forward=R['forward'][keep]
    def project(values):
        values=np.asarray(values);s=values.shape[1:]
        out=(gaunt@(forward@values.reshape(len(values),-1))).reshape((nt,nr)+s)
        return out.transpose((0,)+tuple(range(2,out.ndim))+(1,))
    return project,gaunt,forward


def displacement_projection(gaunt,forward,coefficients,scalar,target,reference,time,nRtarget,nRreference):
    # Full Weyl generating function; no 3GB point-by-field intermediate.
    accum=np.zeros((forward.shape[0],len(target)*len(reference)),dtype=np.complex128)
    for start in range(0,len(coefficients),128):
        stop=min(start+128,len(coefficients))
        D=f.displacement(target,reference,coefficients[start:stop],time)
        accum+=forward[:,start:stop]@(scalar[start:stop,None,None]*D).reshape(stop-start,-1)
    return (gaunt@accum).reshape(nRtarget,nRreference,len(target),len(reference)).transpose(0,2,1,3).reshape(nRtarget*len(target),nRreference*len(reference))


def image(model,state,time,angular=False):
    source=model.fixed;p=model.parameters;number=model.matter.sector.number
    R=m.frozen.projection.material(2) if angular else source.R
    G=m.frozen.band.angular(tuple(np.arange(number%2/2,2+number%2/2+.1,1))) if angular else model.matter.g
    gi=[G['labels'].index(v) for v in model.matter.g['labels']]
    ri=[R['labels'].index(v) for v in source.R['labels']]
    target=source.basis if angular else m.primitive.basis(p['field_total_cut']+2)
    fi=[target.index(v) for v in source.basis];nf=len(target);ng=len(G['labels']);nm=len(R['labels'])
    shape=(model.nf,ng,model.nr,nm,nf);x=state.reshape(model.shape)
    out=np.zeros(shape,dtype=np.complex128);connection=np.zeros(shape,dtype=np.complex128)
    project,gaunt,forward=rectangular_projection(R,ri)
    E=np.eye(nf)[:,fi];ER=np.eye(nm)[:,ri]
    base=(source.base@x.reshape(-1,model.nfield)).reshape(model.shape)
    for a,b in enumerate(gi):
        for c,d in enumerate(ri):out[:,b,:,d,fi]=base[:,a,:,c,:].transpose(2,0,1)
    # Full primitive products: intermediate refcut+1 is exact for b²/Pi².
    middle=m.primitive.basis(p['field_total_cut']+1)
    bt=f.primitive_images(middle,target,time)['b'];br=f.primitive_images(middle,source.basis,time)['b']
    qq=np.einsum('lei,mej->lmij',bt.conj(),br,optimize=True)
    bc=f.primitive_images(target,source.basis,time)['b']
    pt=np.asarray([m.primitive.primitive(middle,target,a,'p').toarray() for a in range(8)])
    pr=np.asarray([m.primitive.primitive(middle,source.basis,a,'p').toarray() for a in range(8)])
    p2=np.einsum('lei,lej->ij',pt.conj(),pr,optimize=True)/(2*p['cavity_inertia'])
    py=np.einsum('fgrma,ba->fgrmb',x,p2,optimize=True)
    for a,b in enumerate(gi):
        for c,d in enumerate(ri):connection[:,b,:,d]+=py[:,a,:,c]
    positions=np.einsum('tab,sb->tsa',m.frozen.rotations(R['points']),(source.matter.geo['sites']-.5)*p['material_scale'])
    conn=np.asarray([m.frozen.kernel.geometry(z,source.modes)['connection'] for z in positions])
    connQ=np.einsum('fs,tsla->ftla',source.Q,conn,optimize=True)
    rotations=m.frozen.rotations(R['points'])
    for sector,c in enumerate(connQ):
        M=project(c);K=np.asarray([sum(R['L'][a].toarray()@M[:,l,a,:]+M[:,l,a,:]@source.R['L'][a].toarray() for a in range(3)) for l in range(8)])
        C=project(np.einsum('tla,tma->tlm',c,c,optimize=True))
        block=np.kron(R['casimir'].toarray()@ER,E)-np.einsum('lij,lab->iajb',K,bc,optimize=True).reshape(nm*nf,model.nm*model.nfield)
        block+=np.einsum('ilmj,lmab->iajb',C,qq,optimize=True).reshape(nm*nf,model.nm*model.nfield)
        y=(x[sector].reshape(-1,model.nm*model.nfield)@block.T).reshape(model.ng,model.nr,nm,nf)/(2*p['material_inertia'])
        for a,b in enumerate(gi):out[sector,b]+=y[a]
    parity=np.array([(-1.)**sum(v) for v in target]);oldparity=np.array([(-1.)**sum(v) for v in source.basis])
    for axis in range(3):
        for site in range(8):
            dest=site^(1<<(2-axis))
            if dest<site:continue
            gamma=source.matter.geo['gamma'][axis][dest,site]
            if not abs(gamma):continue
            phi=m.frozen.kernel.segment(positions[:,site],positions[:,dest],source.modes)[0]
            one=np.zeros((8,8),dtype=np.complex128);one[dest,site]=gamma;hop=source.matter.sector.bilinear(one).tocoo()
            for ell in range(3):
                coefficient=rotations[:,ell,:]@(-2*np.cos(p['k']))
                rf=displacement_projection(gaunt,forward,-phi,coefficient,target,source.basis,time,nm,model.nm)
                reverse=rf.reshape(nm,nf,model.nm,model.nfield)*parity[None,:,None,None]*oldparity[None,None,None,:]
                for backwards,matrix in [(False,rf),(True,reverse.reshape(nm*nf,-1))]:
                    rows,cols,values=(hop.col,hop.row,hop.data.conj()) if backwards else (hop.row,hop.col,hop.data)
                    y=(x[cols].reshape(-1,model.nm*model.nfield)@matrix.T).reshape(len(cols),model.ng,model.nr,nm,nf)
                    g=G['O'][ell][axis].toarray()[:,gi]
                    y=np.einsum('ij,fjrma->firma',g,y,optimize=True)
                    for k,(row,value) in enumerate(zip(rows,values)):out[row]+=value*y[k]
    compressed=out[:,gi,:,:,:][:,:,:,ri,:][...,fi].reshape(-1)
    compressed_connection=connection[:,gi,:,:,:][:,:,:,ri,:][...,fi].reshape(-1)
    record: dict[str,Any]
    record=dict(target_shape=shape,absolute_time=time,angular_extension=angular,HI_compression_error=float(np.linalg.norm(compressed-model.apply(state,time))),H0_connection_compression_error=float(np.linalg.norm(compressed_connection-model.free_energy(state))),physical_H_compression_error=float(np.linalg.norm(compressed+compressed_connection-model.physical_energy(state,time))),claim='separate orthogonal angular or field expanded-shell lower bound, fixed positive-r5 regulator; not full infinite Peierls image norm')
    outside=out.copy();freeoutside=connection.copy()
    for a,b in enumerate(gi):
        for c,d in enumerate(ri):outside[:,b,:,d,fi]=0;freeoutside[:,b,:,d,fi]=0
    record.update(HI_outgoing_lower_bound=float(np.linalg.norm(outside)),H0_basis_connection_outside_norm_NOT_error=float(np.linalg.norm(freeoutside)),full_H_outside_norm_NOT_error=float(np.linalg.norm(outside+freeoutside)))
    if not angular:
        hi=model.apply(state,time);J=-model.vertex(state,0,time)
        pidot=1j*(model.apply(model.field(state,0,time),time)-model.field(hi,0,time))
        adot_debt=1j*(model.apply(model.field(state,0,time,kind='q'),time)-model.field(hi,0,time,kind='q'))
        record['interacting_mode0_canonical_debt']=dict(source_norm=float(np.linalg.norm(J)),Pi_dot_minus_source=float(np.linalg.norm(pidot-J)),a_dot_minus_Pi_over_I=float(np.linalg.norm(adot_debt)))
    return record,out,connection


def main():
    result={}
    for cut in (1,2):
        p=dict(m.DEFAULTS,field_total_cut=cut)
        for number,name in [(4,'neutral'),(5,'addition'),(3,'removal')]:
            model=m.Model(number,p);raw=np.load(ROOT/(P+f'field{cut}_'+name+'_raw.npz'));state=raw['baseline']
            result[f'field{cut}_{name}']={}
            for angular,label in [(False,'field'),(True,'angular')]:
                record,hi,h0=image(model,state,.04,angular)
                result[f'field{cut}_{name}'][label]=record
                np.savez_compressed(ROOT/(P+f'field{cut}_{name}_{label}_images.npz'),H_I_image=hi,H0_basis_connection_image=h0,full_H_image=hi+h0,unbounded_generator_F=model.fixed.F,unbounded_generator_charge=model.fixed.Q)
                (ROOT/(P+'images_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(cut,name,label,json.dumps(record),flush=True)
                del hi,h0;gc.collect()
            del model;gc.collect()


if __name__=='__main__':main()
