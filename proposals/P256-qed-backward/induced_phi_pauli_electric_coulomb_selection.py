"""Actual primitive Pi/shared-Coulomb selection, no full-H replacement."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import block_diag

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_initial_moment as initial
pauli=initial.pauli
quantum=initial.quantum
SOURCE_BYTES={**initial.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
MOMENT_SHA='2be1430a2087bd625b11eb847ea4d46ede3b8453da5af08d94d9e2bbbecd1bc0'


def groups(native:dict[str,Any],state:dict[str,Any])->tuple[list[np.ndarray],np.ndarray]:
    B=state['B'];plus=B[:,:2];gamma=native['frame']/np.sqrt([native['g0']]+[native['gs']]*3)[:,None,None];gamma5=gamma[0]@gamma[1]@gamma[2]@gamma[3];tau=[-1j*plus.conj().T@gamma[j]@gamma5@plus for j in (1,2,3)]
    all_groups=[]
    for t in tau:
        blocks=[]
        for site in native['sites']:
            G=np.eye(2,dtype=complex)
            for ti,n in zip(tau,site):G=G@np.linalg.matrix_power(ti,int(n))
            u=G@(1j*t)@G.conj().T;blocks.append(B@np.asarray(block_diag(u,u))@B.conj().T)
        all_groups.append(np.asarray(block_diag(*blocks)))
    S=np.asarray(block_diag(*[(-1.)**int(sum(site))*gamma[0] for site in native['sites']]))
    return all_groups,S


def project(A:np.ndarray,rotations:list[np.ndarray],S:np.ndarray)->np.ndarray:
    spin=(A+sum(U@A@U.conj().T for U in rotations))/4
    return (spin+S@spin@S)/2


def physical(mean:dict[str,Any])->dict[str,Any]:
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];beta=F(1,2048);b=float(beta);g=float(native['UE']/native['a']);a=float(native['a']);s=2.;n=4;mode=pauli.magnetic_shape(native)
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0;f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4);dy=np.zeros(3*V);dy[1::3]=f
    lv,lb=np.linalg.eigh(native['Lplus']);A0map=lb[:,lv>1e-12]*np.sqrt(lv[lv>1e-12]);rng=np.random.default_rng(20261022);rows=[];structural=[]
    for sample in range(4):
        fractions=np.arange(2*n+1)/(2*n);middle=rng.normal(0,np.sqrt(s),M);difference=rng.normal(0,np.sqrt(2*g*b/(1+g*b/(2*s))),M);motion=np.vstack((np.zeros(M),np.cumsum(rng.normal(0,np.sqrt(g*b/n),(2*n,M)),axis=0)))
        fields=(middle+(fractions[:,None]-.5)*difference+motion-fractions[:,None]*motion[-1])@native['Q'].T;alphas=np.sqrt(2*g*b/n)*rng.normal(size=(n,V-1))@A0map.T
        states,Wk,Fk=pauli.endpoints(native,fields[0],mode);_,Wb,Fb=pauli.endpoints(native,fields[-1],mode);rotations,S=groups(native,states[0]);xsl=slice(4*x,4*x+4)
        for state,Fop,p in zip(states,Fk,native['Pprofiles']):
            P=state['P'];Q=np.eye(4*V)-P;D=Q[:,xsl]@Q[xsl,:]/2;H=quantum.matter_hamiltonian(native,p,fields[0]);H1=initial.one_link_Wilson(native,p,fields[0]);other=H-H1
            structural.append(dict(source_P_invariant_error=max(float(np.linalg.norm(U@P@U.conj().T-P)) for U in [*rotations,S]),ALL4hole_D_invariant_error=max(float(np.linalg.norm(U@D@U.conj().T-D)) for U in [*rotations,S]),Pauli_force_invariant_projection_norm=float(np.linalg.norm(project(Fop,rotations,S))),actual_fullCAR_other_operator_invariant_error=float(np.linalg.norm(other-project(other,rotations,S))),actual_one_link_Wilson_noninvariant_norm=float(np.linalg.norm(H1-project(H1,rotations,S)))))
        total=alphas.sum(axis=0);C=np.diag(np.repeat(np.exp(-1j*total),4));HSerror=max(float(np.linalg.norm(U@C-C@U)) for U in [*rotations,S]);phase=np.exp(1j*(128*total.sum()+total[ref]));scalar=(1+g*b/(2*s))**(-M/2)
        # Exact HE=g Pi² commutes HC=g rho Lplus rho. This is an OPERATOR
        # selection control, not a replacement model or dropped physical term.
        ec=scalar*phase*pauli.source_kernel(native,Wb,Wk,Fb,Fk,[C,C],x)
        words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)]
        for j in range(n):
            Dphase=np.repeat(np.exp(-1j*alphas[j]),4)
            for si,p in enumerate(native['Pprofiles']):
                half=quantum.half_heat(native,p,fields[2*j+1],b/n);words[si]=((half*Dphase[None,:])@half)@words[si]
        Efree=sum(p['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,p,np.zeros(3*V)))[:2*V].sum() for p in native['Pprofiles'])
        full=scalar*np.exp(b*Efree)*phase*pauli.source_kernel(native,Wb,Wk,Fb,Fk,words,x);Aket=float(dy@fields[0]/a);Abra=float(dy@fields[-1]/a)
        rows.append(dict(sample=sample,actual_sharedA0_word_group_commutator_error=HSerror,exact_HE_HC_decomposition_ONLY_no_CAR_vacuumoffset_RAW_phi_carrier_left_real=float(ec[0,1].real),exact_HE_HC_decomposition_ONLY_no_CAR_vacuumoffset_RAW_phi_carrier_left_imag=float(ec[0,1].imag),exact_HE_HC_decomposition_ONLY_RAW_phi_Ay_left_abs=float(abs(ec[0,1]*Aket)),exact_HE_HC_decomposition_ONLY_RAW_phi_Ay_right_abs=float(abs(ec[1,0]*Abra)),actual_fullH_samepath_SAMEneutraloffset_RAW_phi_Ay_real=float((full[0,1]*Aket).real),actual_fullH_samepath_SAMEneutraloffset_RAW_phi_Ay_imag=float((full[0,1]*Aket).imag)))
    structural_keys=('source_P_invariant_error','ALL4hole_D_invariant_error','Pauli_force_invariant_projection_norm','actual_fullCAR_other_operator_invariant_error')
    if max(r[k] for r in structural for k in structural_keys)>1e-9 or max(r['actual_sharedA0_word_group_commutator_error'] for r in rows)>1e-9 or max(r['exact_HE_HC_decomposition_ONLY_RAW_phi_Ay_left_abs'] for r in rows)>1e-7:raise ArithmeticError('Actual native Pi/Coulomb representation/sourceword selection failed')
    return dict(N=4,V=V,M=M,beta=str(beta),Strang_steps=n,samples=4,seed=20261022,ALLcharged_flavours=64,shared_REAL_A0_modes=V-1,g_Pi_AND_Coulomb=g,source_position=[1,1,0],actual_source_and_operator_representation_controls=structural,actual_Pi_sharedCoulomb_and_FULL_H_samepath_words=rows,selection_proof_for_parent_review='Fixed localspin Gx and Gamma0/sublattice S independentTheta. CanonicalY sourceSlater invariant (positive onsite-section, det occupied SU2 andS equal1); ALL4hole E reducedkernel invariant, compensating samefixedsource-colour rotation on Ref. Densityrho and primitivePi commutegroups. PauliF has NOinvariant component, actualtwirl projection zero. Hence <Fsource|exp[-beta(HE+HC)]|A-source>=0 ALLbeta (alsoinitialgenerator). SharedHS densityword and actualelectric endpoint pair executed; HEC commute and original Gaussian2 heat exact. Physicalone-linkWilsonΓ0 is noninvariant and actualFULLH includes it; samepath fullmatrix nonzero. No physicalterm removed or newlaw substituted.',scope='Native frozenPhi/referenceE0 only. OriginalfullPhi EU breaks preparation symmetry bytinycolourtransport, separatelypriced06db; no fullPhi/heavy/Noe evolutionzero claim. ChargedmatterPhi-gradient only, neutralheavyPHnoise6af1 remains. No completefinitebeta ensemblebridge/pole/commoncone/vacuum/QED promotion.')


def run(path:Path)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MOMENT_SHA:raise ArithmeticError('Frozen41d2/2be1 initialmoment epoch required')
    cap=json.loads(raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['beta']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['metric']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(moment=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],native_N=4,beta='1/2048',Strang_steps=4,samples=4,seed=20261022,source='UNCHANGEDoriginalcanonicalY/Gaussian2/AP222/all64/ALL4Refholes/Ebeforeheat. HE/HC word is namedoperator-decompositioncontrol, NOTreplacementlaw. SAMEpath fullHCAR+HE+HC alsoexecuted.',a='1/64',UEphys='1',g_Pi_AND_Coulomb='64',Lambda='2^8192',chi='2^-4096',original_clock='4096')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_native_Pauli_electric_Coulomb_selection=physical(mean),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--moment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.moment)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual Pi/Coulomb selection calculation')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_native_Pauli_electric_Coulomb_selection'],elapsed_seconds=r['elapsed_seconds']),indent=2))
