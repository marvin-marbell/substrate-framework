"""Independent arithmetic of NEW gauge source/current spectral capsule, no spectrum replay."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main():
    inp,out=map(Path,sys.argv[1:3]);p=json.loads(inp.read_text());state=Path(p['owner_npz']);result=Path(p['owner_json']);z=np.load(state);o=json.loads(result.read_text());energy=z['neutral_energies'];amplitude=z['neutral_current_Q_amplitudes'];gaps=energy-o['occupied']['energy'];keep=gaps>1e-8
    signed=-2*np.sum(abs(amplitude[keep,:6])**2/gaps[keep,None],axis=0)
    gram=z['addition_amplitudes'].conj().T@z['addition_amplitudes']+(z['removal_amplitudes'].conj().T@z['removal_amplitudes']).T-np.eye(4)
    evidence=dict(kind='INDEPENDENT_ARITHMETIC_AUDIT_OF_NEW_EXACT_ALL_RETAINED_SPECTRAL_CAPSULE',inputs={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [state,result]},neutral_virtual_states=len(energy),source_states_addition=len(z['addition_energies']),source_states_removal=len(z['removal_energies']),raw_spectral_CAR_error=float(np.linalg.norm(gram)),all_virtual_direct_signed=signed.tolist(),difference_from_owner=[float(signed[a]-o['response'][a]['direct_signed_all_virtual']) for a in range(6)],direct_contact_plus_paramagnetic=[float(signed[a]+o['response'][a]['physical_contact']) for a in range(6)],meaning='NEW whole-H eigenamplitudes, every retained virtual. Arithmetic audit, not independently recomputed spectrum or old G/D. Static sum is fixed-original-A response only; no covariant-current/direct-contact conflation or Maxwell pole claim.')
    out.write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence))


if __name__=='__main__':main()
