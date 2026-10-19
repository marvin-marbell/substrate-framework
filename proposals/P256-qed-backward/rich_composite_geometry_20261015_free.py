"""Exact full-CAR64 quadratic diagnostic, NOT a replacement candidate/threshold."""
from itertools import product
import json,resource,sys,time
from pathlib import Path
import numpy as np
from rich_composite_geometry_20261015_model import geometry,PAULI,EDGES,epsilon


def profile(projector,graph):
    colours=[c for c in product(range(3),repeat=3) if epsilon(*c)]
    positions=[];weights=[];radii=[]
    for pos in product(range(4),repeat=3):
        total=0j
        for spins in product(range(2),repeat=3):
            orbital=[2*v+s for v,s in zip(pos,spins)]
            for left,right in product(colours,repeat=2):
                gram=np.zeros((3,3),complex)
                for i,j in product(range(3),repeat=2):
                    if (i==2)==(j==2) and left[i]==right[j]:gram[i,j]=projector[orbital[i],orbital[j]]
                total+=epsilon(*left)*epsilon(*right)*np.linalg.det(gram)/6
        positions.append(pos);weights.append(float(total.real))
        radii.append(sum(np.dot(graph['vertices'][pos[i]]-graph['vertices'][pos[j]],graph['vertices'][pos[i]]-graph['vertices'][pos[j]]) for i in range(3) for j in range(i))/3)
    normalization=sum(weights)
    return {'positions':positions,'raw_spin_summed_source_weights':weights,'raw_sum':normalization,
        'normalized_source_probe_weights':(np.array(weights)/normalization).tolist(),
        'compact_source_profile_fraction':sum(w for pos,w in zip(positions,weights) if len(set(pos))==1)/normalization,
        'mean_relative_radius_squared':float(np.dot(weights,radii)/normalization),
        'meaning':'Ordered-position epsilon/colour and COMPLETE spin-summed THREE-ADDITION source spectral profile via exact Gaussian Wick determinant. Probe-normalized positive weights, not an added-particle position PVM or binding criterion.'}


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];p=config['parameters'];started=time.time()
    graph=geometry(p);phi=np.sqrt(p['scalar_quadratic']/(2*p['scalar_quartic']))
    one=np.zeros((8,8),complex)
    for edge,(start,end) in enumerate(EDGES):
        transport=(p['scalar_coupling']*phi*np.eye(2)+1j*p['vector_velocity']*sum(graph['axes'][edge,i]*PAULI[i] for i in range(3)))*graph['weights'][edge]
        one[2*end:2*end+2,2*start:2*start+2]+=transport
        one[2*start:2*start+2,2*end:2*end+2]+=transport.conj().T
    energies,vectors=np.linalg.eigh(one);occupied=vectors[:,:4]@vectors[:,:4].conj().T
    lowest=np.flatnonzero(abs(energies-energies[4])<1e-9)
    addition=vectors[:,lowest]@vectors[:,lowest].conj().T
    result={'kind':'ACTUAL_EXACT_CAR64_QUADRATIC_NONBINDING_CONTROL_NOT_CANDIDATE_FALLBACK',
        'parameters':p,'law_change':'Freeze phi to primitive bare potential minimum;R=I,U=I;NO electric/plaquette/quantumfield/rotor dynamics. Full64 canonical fermions, neutral occupied exact32-particle Slater sea. No proper angular claims for fixed orientation.',
        'one_body8_eigenvalues':energies.tolist(),'eight_internal_repeats':'2flavour x4members;64totalmodes',
        'neutral_N32_energy':float(8*sum(energies[:4])),
        'uud_addition_energy':float(3*energies[4]),
        'additivity':'Two up and one down quarks can occupy the same lowest unoccupied spatial/spin orbital with epsilon-contracted distinct colours. Addition energy is the SUM of constituent eigenenergies in THIS exactly quadratic control, not a lepton-gap/asymptotic threshold for the interacting candidate.',
        'lowest_addition_band_rank':len(lowest),'pole_profile':profile(addition,graph),
        'full_raw_source_profile':profile(np.eye(8)-occupied,graph),
        'irreducible_fermionic_cumulants':'Gaussian free vacuum obeys Wick; irreducible connected fermionic cumulants beyond2 vanish. Composite epsilon-source correlations still have nonzero disconnected/exchange Wick terms; colour singlet label alone is not confinement.',
        'resource':{'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(out+'_onebody.npz',H=one,energies=energies,vectors=vectors,occupied=occupied,lowest_addition=addition)
    print(json.dumps({'completed':'quadratic_control',**result['resource']}),flush=True)


if __name__=='__main__':main()
