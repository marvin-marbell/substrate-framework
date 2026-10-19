"""NEW own mobile N1 reference G plus shared-field/current inclusive event."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from rich_common_mobile_20261014_model import Mobile, MobileParameters
from rich_frame_photon_20261014_model import Parameters, spectrum, full_resolvent
from rich_review_wave7_20261014_mobile import norm, expectation, evolve

ROOT=Path(__file__).resolve().parent


def main():
    inputs=json.loads((ROOT/'rich_review_wave7_20261014_mobile_state_input.json').read_text())
    pp=inputs['parameters']
    p=MobileParameters(base=Parameters(**pp['base']),centres=tuple(tuple(row) for row in pp['centres']),cm_hopping=pp['cm_hopping'],core_mass=pp['core_mass'])
    neutral,mobile,pair=[Mobile(p,number) for number in (0,1,2)]
    energies,eigen,receipt=spectrum(mobile.H,count=6)
    reference=eigen[:,0];energy=energies[0]
    plus=np.column_stack([mobile.source_to(pair,mode)@reference for mode in range(4)])
    minus=np.column_stack([neutral.source_to(mobile,mode).getH()@reference for mode in range(4)])
    z=complex(*inputs['frequency'])
    addition,aresidual=full_resolvent(pair.H,energy+z,plus)
    removal,rresidual=full_resolvent(neutral.H,energy-z,minus)
    gplus=plus.conj().T@addition
    gminus=-(minus.conj().T@removal).T
    spin_internal=sum(value*sum(side[axis] for side in mobile.spin) for axis,value in enumerate(inputs['spin_axis']))
    vals,u=eigh(spin_internal);spinproject=u[:,vals>0]@u[:,vals>0].conj().T
    ps=mobile.mop(spinproject)
    fieldmask=(np.asarray(inputs['field_direction'])@mobile.coordinates>0).astype(float)
    pf=mobile.fop(sp.diags(fieldmask))
    incoming=evolve(sum(value*operator for value,operator in zip(inputs['field_direction'],mobile.P)),reference,-1j*inputs['pulse'])
    current=sum(value*operator for value,operator in zip(inputs['current_direction'],mobile.J))
    spin=sum(value*operator for value,operator in zip(inputs['spin_axis'],mobile.S))
    events={}
    for label,insert in (('no_insertion',mobile.I),('actual_same_current',current),('wrong_Noether_spin',spin)):
        state=insert@incoming;strength=float(np.linalg.norm(state)**2);state/=np.linalg.norm(state)
        outgoing=evolve(mobile.H,state,-1j*inputs['time'])
        probabilities=[float(np.linalg.norm(cell@sproj@fproj@outgoing)**2) for cell in (mobile.N[0],mobile.I-mobile.N[0]) for sproj in (ps,mobile.I-ps) for fproj in (pf,mobile.I-pf)]
        events[label]={'insertion_strength':strength,'probabilities_cell_spin_field':probabilities,'sum':sum(probabilities),
                       'cell0':expectation(mobile.N[0],outgoing),'spin':expectation(spin,outgoing),
                       'field':[expectation(a,outgoing) for a in mobile.A],'mode_current':[expectation(a,outgoing) for a in mobile.J]}
    reactions={}
    for label,law in (('same_H',mobile),('wrong_coupling_off',Mobile(p,1,coupling=False)),('metric_off',Mobile(p,1,metric=False))):
        field=evolve(sum(value*operator for value,operator in zip(inputs['field_direction'],law.P)),reference,-1j*inputs['pulse'])
        driven=evolve(law.H,field,-1j*inputs['time'])
        baseline=evolve(law.H,reference,-1j*inputs['time'])
        reactions[label]={'same_initial_state':True,'cell_response':expectation(law.N[0],driven)-expectation(law.N[0],baseline),
                          'spin_response':expectation(sum(value*a for value,a in zip(inputs['spin_axis'],law.S)),driven)-expectation(sum(value*a for value,a in zip(inputs['spin_axis'],law.S)),baseline),
                          'current_norms':[norm(a) for a in law.J]}
    result={'kind':'NEW_SAME_MOBILE_N1_REFERENCE_COMPLETE_ADJACENT_G_AND_QUANTUM_EVENT','inputs':inputs,
            'originating_reference':{'N':1,'energy':float(energy),'S2_error':norm(sum(s@s for s in mobile.S)-.75*mobile.I),**receipt},
            'dimensions':[neutral.size,mobile.size,pair.size],
            'source_Gram_CAR_error':float(np.linalg.norm(plus.conj().T@plus+(minus.conj().T@minus).T-np.eye(4))),
            'G_addition':[[str(value) for value in row] for row in gplus], 'G_removal_canonical_transposed':[[str(value) for value in row] for row in gminus],
            'G_complete':[[str(value) for value in row] for row in gplus+gminus], 'full_adjacent_resolvent_residuals':aresidual+rresidual,
            'same_reference_field_current_events':events,'physical_reactions':reactions,
            'scope':'Same complete whole-H N1 ground reference owns G and incoming independent FIELD pulse/current insertion/inclusive outgoing cell-spin-field receiver. FullN0/N2 retained, one active charge. Counterlaws use SAME initial state, not frozen/preselected virtuals. Finite global modes/two positions, no local causality/freeCOM/scattering asymptote.'}
    (ROOT/'rich_review_wave7_20261014_mobile_state_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
