"""Exact CAR64 Gaussian-control quark-count FCS, not3-added-particle positions.
Colour-white uud is epsilon superposition of3 equal-energy Slater components;
regional Ncolour commutes separate flavour/colour totals, so cross terms vanish.
"""
import json,resource,sys,time
from pathlib import Path
import numpy as np
from itertools import product


def fcs(p4,p5):
    count=13;character=np.zeros((count,)*3,complex)
    for a,b,c in product(range(count),repeat=3):
        phases=np.exp(2j*np.pi*np.array([a,b,c,0])/count);d=np.diag(np.repeat(phases,2))
        z4=np.linalg.det(np.eye(8)-p4+p4@d);z5=np.linalg.det(np.eye(8)-p5+p5@d)
        character[a,b,c]=z4**3*z5**3
    probabilities=np.fft.fftn(character).real/(count**3)
    compact=0.;fragmented=0.;normalization=0.;mean=np.zeros(4);second=np.zeros((4,4));outside=0.;minimum=1.
    for a,b,c in product(range(count),repeat=3):
        n=(a,b,c,27-a-b-c);value=float(probabilities[a,b,c])
        if not 0<=n[3]<=12:outside+=abs(value);continue
        minimum=min(minimum,value);normalization+=value;v=np.array(n)
        mean+=value*v;second+=value*np.outer(v,v)
        if sorted(n)==[6,6,6,9]:compact+=value
        if sorted(n)==[6,7,7,7]:fragmented+=value
    return {'compact_quark_occupation_channel':compact,'fragmented_quark_occupation_channel':fragmented,
        'ncolour_site_mean':mean.tolist(),'ncolour_site_covariance':(second-np.outer(mean,mean)).tolist(),
        'normalization':normalization,'signed_min_probability':minimum,'unphysical_fourth_count_alias_roundoff_abs':outside,
        'three_angle_grid':count,'exact_support':'Each siteNcolour0..12,total27;13-point Fourier in3 independent angles has no alias. NO clipping/forced renormalization.'},probabilities


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time();data=np.load(config['free_snapshot'])
    energies=data['energies'];vectors=data['vectors'];p4=data['occupied'];lowest=np.flatnonzero(abs(energies-energies[4])<1e-9)
    rows=[];tables=[]
    for index in lowest:
        vector=vectors[:,index];p5=p4+np.outer(vector,vector.conjugate());row,table=fcs(p4,p5)
        rows.append(row);tables.append(table)
    result={'kind':'ACTUAL_EXACT_FULL_OCCUPIED_FREE_CAR64_REGIONAL_QUARK_OCCUPATION_CONTROL',
        'law':'Same geometry/g/v exact quadratic control already captured:phi fixed at bare potential minimum,R=I,U=I,no electric/plaquette/field dynamics. Full occupiedN32sea plus colour-whiteuudN35,NOT projected interacting candidate or particle-only valence fallback.',
        'FCS_equation':'chi=det(I-P4+P4 D)^3 det(I-P5+P5 D)^3. Three internal flavour/colour modules contain one extra lowest-band fermion;three remainN4. Epsilon-colour offdiagonal Slater terms have different separately conserved flavour/colour totals and vanish for regional totalNcolour measurements.',
        'lowest_addition_energy':float(3*energies[4]),'threshold_scope':'Additive energy in THIS genuinely nonbinding quadratic control ONLY, not an interacting breakup/lepton threshold.',
        'degenerate_orbital_rows':rows,'max_degenerate_orbital_PVM_difference':max(float(np.max(abs(t-tables[0]))) for t in tables),
        'ensemble_compact_probability':float(np.mean([r['compact_quark_occupation_channel'] for r in rows])),
        'ensemble_fragmented_probability':float(np.mean([r['fragmented_quark_occupation_channel'] for r in rows])),
        'meaning':'Exact SAME quark-occupation channel protocol used by candidate, including whole occupied sea. These are NOT EM charge/colour8/Casimir or positions of3 added particles. Degenerate lowest-band orbitals all reported/equal ensemble, no preferred source fit. No properJ claim for pinnedR control.',
        'resource':{'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');np.savez(out+'_FCS.npz',tables=np.array(tables))
    print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
