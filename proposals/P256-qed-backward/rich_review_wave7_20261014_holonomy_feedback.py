"""NEW frozen-feedback/high-virtual falsifiers at actual filled Fock reference."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from rich_holonomy_em_20261014_model import build, sector
from rich_frame_photon_20261014_model import spectrum, full_resolvent
from rich_review_wave7_20261014_holonomy import norm, expectation, propagate

ROOT=Path(__file__).resolve().parent


def main():
    inputs=json.loads((ROOT/'rich_review_wave7_20261014_holonomy_feedback_input.json').read_text())
    p=inputs['parameters'];external=np.array(inputs['external']);model=build(p,external=external)
    ix=sector(model,4);h=model['H'][ix][:,ix]
    e,v,receipt=spectrum(h,count=6);reference=v[:,0]
    frozen_j=[np.zeros((256,256),complex) for _ in range(2)]
    frozen_k=[[np.zeros((256,256),complex) for _ in range(2)]for _ in range(2)]
    for forward,coeff in zip(model['forwards'],model['geometry']['line']):
        link=np.exp(-1j*p['charge']*coeff@external)*forward
        for a in range(2):
            frozen_j[a]+=p['charge']*coeff[a]*1j*(link-link.conj().T)
            for b in range(2):
                frozen_k[a][b]+=-p['charge']**2*coeff[a]*coeff[b]*(link+link.conj().T)
    frozen_j=[sp.kron(sp.csr_matrix(a),sp.eye(model['nf']),format='csr') for a in frozen_j]
    frozen_k=[[sp.kron(sp.csr_matrix(a),sp.eye(model['nf']),format='csr')for a in row]for row in frozen_k]
    actual_variance=[];frozen_variance=[]
    for actual,frozen in zip(model['J'],frozen_j):
        a=actual[ix][:,ix];b=frozen[ix][:,ix]
        actual_variance.append(expectation(a@a,reference)-expectation(a,reference)**2)
        frozen_variance.append(expectation(b@b,reference)-expectation(b,reference)**2)
    target=sector(model,5);targeth=model['H'][target][:,target]
    sources=np.column_stack([a[target][:,ix]@reference for a in model['source']]);z=complex(*inputs['frequency'])
    full,residuals=full_resolvent(targeth,e[0]+z,sources)
    lowe,lowv,lowreceipt=spectrum(targeth,count=inputs['wrong_low_virtual_count'])
    low=lowv@((1/(e[0]+z-lowe))[:,None]*(lowv.conj().T@sources))
    gf=sources.conj().T@full;gl=sources.conj().T@low
    field=sum(value*a for value,a in zip(inputs['field_direction'],model['P']))[ix][:,ix]
    prepared=propagate(field,reference,-1j*inputs['pulse'])
    receivers={}
    for label,operators in (('whole_quantum_current',model['J']),('wrong_frozen_Q0_current',frozen_j)):
        current=sum(value*a for value,a in zip(inputs['current_direction'],operators))[ix][:,ix]
        injected=current@prepared;weight=float(np.linalg.norm(injected)**2);injected/=np.linalg.norm(injected)
        outgoing=propagate(h,injected,-1j*inputs['time'])
        receivers[label]={'insertion_strength':weight,'field_Q':[expectation(a[ix][:,ix],outgoing) for a in model['Q']],
                          'actual_current':[expectation(a[ix][:,ix],outgoing) for a in model['J']],
                          'S2':expectation(model['S2'][ix][:,ix],outgoing)}
    result={'kind':'NEW_WHOLE_H_CURRENT_CONTACT_AND_FULL_VIRTUAL_FEEDBACK_FALSIFIER','inputs':inputs,'reference_receipt':receipt,
            'full_minus_frozen_J_norms':[norm(a-b) for a,b in zip(model['J'],frozen_j)],
            'full_minus_frozen_K_norms':[[norm(model['K'][a][b]-frozen_k[a][b])for b in range(2)]for a in range(2)],
            'actual_current_connected_variance':actual_variance,'wrong_frozen_connected_variance':frozen_variance,
            'full_G_norm':float(np.linalg.norm(gf)),'wrong_low_virtual_G_relative_error':float(np.linalg.norm(gl-gf)/np.linalg.norm(gf)),
            'discarded_source_norm':float(np.linalg.norm(sources-lowv@(lowv.conj().T@sources))),
            'full_resolvent_residuals':residuals,'wrong_low_virtual_receipt':lowreceipt,'field_current_receivers':receivers,
            'meaning':'Q0-frozen current/contact are explicit wrong operators, not the complete shared quantum field source. Lowest finite poles are only a wrong virtual comparator; actual fullN5 solve retains all states. Neutral receiver is a loop-current/field feedback event, not a charged spin-1/2 species assertion.'}
    (ROOT/'rich_review_wave7_20261014_holonomy_feedback_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
