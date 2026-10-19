"""Uncompleted whole law: own occupied reference and exact degenerate bright V."""
import json
import sys
from pathlib import Path
import numpy as np
from rich_parity_sea_field_20261014_model import build, fourier_source
from rich_parity_sea_field_20261014_calculate import diagonalize, blocks, norm, enc


def run(config):
    root=Path(__file__).resolve().parent;p=json.loads((root/config['law_input']).read_text());m=build(p,mirror=False)
    sea=2*p['sites'];neutral=diagonalize(m,sea);charged=diagonalize(m,sea+1)
    eg=min(b['e'][0] for b in neutral);references=[];reference_m=[]
    for b in neutral:
        for i in np.flatnonzero(abs(b['e']-eg)<2e-8):
            references.append(b['U']@b['v'][:,i]);reference_m.append(b['m'])
    b=charged[0];sources=fourier_source(m,0);amps=[b['v'].conj().T@(b['U'].getH()@np.column_stack([a@r for a in sources])) for r in references]
    clusters=[]
    for start,end in blocks(b['e']):
        residue=sum(a[start:end].conj().T@a[start:end] for a in amps)/len(references)
        weight=float(np.trace(residue).real)
        if weight<1e-10:continue
        w=b['U']@b['v'][:,start:end];velocity=w.conj().T@(m['V']@w)
        clusters.append(dict(gap=float(b['e'][start]-eg),degeneracy=end-start,residue=enc(residue),total_residue=weight,
                             orthonormal_error=norm(w.conj().T@w-np.eye(end-start)),velocity=enc(velocity),velocity_eigenvalues=np.linalg.eigvalsh(velocity).tolist(),
                             current_norms=[norm(w.conj().T@(a@w)) for a in m['J']],stationary_entire_block=norm(velocity)<1e-8))
    refs=np.column_stack(references)
    result=dict(candidate=p['candidate']+'-without-mirror-routes',reference=dict(N=sea,E=float(eg),degeneracy=len(references),momenta=reference_m),
                P_H=norm(m['P']@m['H']@m['P'].getH()-m['H']),P_reference_projector=norm(m['P']@refs-refs@(refs.conj().T@(m['P']@refs))),
                first_m0_bright=clusters[0],dominant_m0_bright=max(clusters,key=lambda a:a['total_residue']),all_m0_bright_blocks=clusters,
                complete_sectors={str(sea):[len(a['e']) for a in neutral],str(sea+1):[len(a['e']) for a in charged]},
                source='literal own rebuilt H/ref/source with exact dense orthonormal eigenspectra; not low eigsh columns or completed-law sea',
                limits='Adverse finite branch only; no global impossibility theorem, continuous COM inertia or asymptotic state.')
    return result,dict(neutral_reference=refs,m0_N7_energies=b['e'],m0_N7_vectors=b['v'])


if __name__=='__main__':
    config=json.loads(Path(sys.argv[1]).read_text());result,states=run(config);stem=Path(sys.argv[2])
    stem.with_name(stem.name+'_result.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(stem.with_name(stem.name+'_states.npz'),**states)
    print(json.dumps({key:value for key,value in result.items() if key!='all_m0_bright_blocks'}))
