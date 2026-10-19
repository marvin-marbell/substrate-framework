"""NEW same-law physical vector-field receiver and full-source impulse dynamics.
Not a replay of actual1; no model/source bytes changed.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from rich_dirac_contact_20261014_model import Parameters,build,hamiltonian,spectrum,vertex
from rich_dirac_contact_20261014_calculate import serial
ROOT=Path(__file__).resolve().parent
PREFIX='rich_dirac_contact_20261014_actual2'


def response(e,v,a,b,state,z):
    aa=v.conj().T@a@v[:,state]; bb=v.conj().T@b@v[:,state]; de=e-e[state]
    return np.sum(aa.conj()*bb/(z-de)-bb.conj()*aa/(z+de))


def main():
    inputs=json.loads((ROOT/'rich_dirac_contact_20261014_input.json').read_text())
    result={'kind':'NEW_SAME_H_VECTOR_FIELD_CURRENT_RECEIVER','architectures':{}}; arrays={}
    for geometry in inputs['architectures']:
        model=build(Parameters(geometry=geometry,**inputs['parameters']))
        k=np.ones(3)*np.pi if geometry=='facing' else np.zeros(3)
        e,v=spectrum(model,k); state=3; current=np.array([vertex(model,k,k,d) for d in np.eye(3)])
        chi=np.array([[response(e,v,model['field_Q'][i],current[j],state,.37+.08j) for j in range(3)] for i in range(3)])
        # A specified impulsive source Hext=-lambda delta(t) Jx, exactly unitary.
        # This is not labelled a finite constant-A minimal-coupling quench.
        strength=.7; prepared=expm(1j*strength*current[0])@v[:,state]
        coefficients=v.conj().T@prepared
        rows=[]
        for time in np.linspace(0,24,97):
            psi=v@(np.exp(-1j*e*time)*coefficients)
            rows.append([time,*[np.vdot(psi,q@psi).real for q in model['field_Q']],*[np.vdot(psi,j@psi).real for j in current],np.vdot(psi,model['field_number']@psi).real,np.vdot(psi,model['spin'][2]@psi).real,np.vdot(psi,hamiltonian(model,k)@psi).real,np.vdot(psi,psi).real])
        rows=np.array(rows); dipoles=v[:,:2].conj().T@current[0]@v[:,2:4]
        result['architectures'][geometry]={'chi_Qi_Jj':chi,'source_impulse_strength':strength,'source_definition':'Hext=-lambda delta(t) Jx; prepare exp(i lambda Jx)|upper,+Sz>; full288 H evolution','max_Qx':np.max(np.abs(rows[:,1])),'Qx_peak_to_peak':np.ptp(rows[:,1]),'field_number_range':[rows[:,7].min(),rows[:,7].max()],'spin_z_range':[rows[:,8].min(),rows[:,8].max()],'energy_range':[rows[:,9].min(),rows[:,9].max()],'norm_range':[rows[:,10].min(),rows[:,10].max()],'upper_to_lower_xcurrent':dipoles,'transition_energy_upper_to_lower':np.mean(e[2:4])-np.mean(e[:2]),'stability_scope':'Upper doublet is an excited charged addition band above another charged doublet, not an absolute lowest-charge electron. Nonzero interdoublet current permits transitions to lower charged band if an accessible field receiver is supplied. No filled lower band is present. Supplied oscillator gap1.15 exceeds this local gap, not a proof of free-space stability.'}
        arrays[geometry+'_pulse']=rows
    result['columns']=['time','Qx','Qy','Qz','Jx','Jy','Jz','field_number','primitive_Sz','energy','norm']
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(serial(result),indent=2)+'\n')
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**arrays)
    print(json.dumps(serial(result),indent=2))
if __name__=='__main__': main()
