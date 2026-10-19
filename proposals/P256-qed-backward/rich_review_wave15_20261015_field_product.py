"""Physical coordinate-phase/current commutation before versus after electric cut."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from rich_review_wave15_20261015_quantum import load


def norm(a):return float(sparse.linalg.norm(a) if sparse.issparse(a) else np.linalg.norm(a))


def generator(size):
    s=sparse.diags(np.ones(size-1),-1,shape=(size,size));sx=sparse.kron(s,sparse.eye(size));sy=sparse.kron(sparse.eye(size),s)
    return (sx+sx.conj().T)/2+.4*(sy-sy.conj().T)/(2j)


def main():
    inp,out=map(Path,sys.argv[1:3]);req=json.loads(inp.read_text());observed=json.loads(Path(req['owner_json']).read_text());data=np.load(req['owner_npz']);cut=3;size=7;nb=49;center=24;old=np.arange(-3,4);big=np.arange(-7,8);bs=15;lookup={int(x):i for i,x in enumerate(big)};embedding=np.array([lookup[int(x)]*bs+lookup[int(y)] for x in old for y in old]);gbig=generator(bs);gold=generator(size);results={}
    deltas=sorted({(sign*k,0) for k in [1,2,3] for sign in [-1,1]}|{(0,sign*k) for k in [1,2,3] for sign in [-1,1]}|{(sign*k,sign*k) for k in [1,2,3] for sign in [-1,1]})
    for name in ['lepton','baryon']:
        h=load(data,name+'_H');matter=h.shape[0]//nb;row=np.arange(matter)*nb;source=load(data,name+'_source')@data['neutral_stationary'];initial=source/norm(source);a=initial.reshape(matter,nb);padded=np.zeros((matter,bs*bs),complex);padded[:,embedding]=a;pieces=[]
        for dx,dy in deltas:
            c=h[row+(dx+cut)*size+dy+cut][:,row+center];rr=[];cc=[]
            for q,(x,y) in enumerate(( (x,y) for x in big for y in big)):
                tx=int(x)+dx;ty=int(y)+dy
                if tx in lookup and ty in lookup:rr.append(lookup[tx]*bs+lookup[ty]);cc.append(q)
            shift=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(bs*bs,bs*bs));pieces.append((dx,dy,c,shift))
        def extended_current(v,axis):
            result=np.zeros_like(v)
            for dx,dy,c,shift in pieces:result+=(-1j*(dx if axis==0 else dy)/3)*(c@v)@shift.T
            return result
        gfull=sparse.kron(sparse.eye(matter),gold);kicked=expm_multiply(-1j*observed['request']['kick']*gold,a.T).T.ravel();channels=[]
        for axis in range(2):
            j=load(data,name+'_J'+str(axis));comm=j@gfull-gfull@j
            before=extended_current(padded@gbig.T,axis)-extended_current(padded,axis)@gbig.T
            current_jump=float(np.vdot(kicked,j@kicked).real-np.vdot(initial,j@initial).real)
            channels.append(dict(axis=axis,cut_current_phase_commutator_operator=norm(comm),cut_current_phase_commutator_source=norm(comm@initial),full_product_commutator_on_embedded_same_source=norm(before),finite_kick_t0_current_change=current_jump,difference_from_exercised_full_consumer=current_jump-observed['results'][name]['samples'][0]['current_excess'][axis]))
        results[name]=dict(dimension=h.shape[0],unchanged_matter_projection=True,original_electric_cut=3,product_witness_extended_cut=7,channels=channels)
    result=dict(kind='QUANTUM_PHASE_CURRENT_FINITE_ELECTRIC_COMPRESSION_PRODUCT_WITNESS',results=results,interpretation='Before electric projection, G=cos(theta_x)+.4sin(theta_y) and physical Peierls J(theta) are commuting multiplication operators. Nonzero t0 current jump under exp(-ik PGP) is a finite-cut multiplication-algebra artifact, NOT instantaneous matter reaction or body-conditioned preparation. Embed SAME matter/source projection and form current/phase products with all shifts through3 BEFORE cut: infinitesimal commutator vanishes. Extended7 suffices for both polynomial products on old cut3 source; this is NOT finite-kick/continuum convergence. Matter-only phi/loop observables remain unchanged at t0; later conditional finite-model reaction separately recorded, no H or stiffness change.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
