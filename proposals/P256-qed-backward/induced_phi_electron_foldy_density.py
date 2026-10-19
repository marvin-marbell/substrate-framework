"""Actual microscopic density/Foldy Darwin/spinorbit/latticeWard CONTROL.

Same1/2linkBloch/Peierls law as executedspin g2; scalarvertex is actualrho=I.
No original E SOURCE/primitiveinteractingpole/photon-normalizedeR retag.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_electron_spin_magnetic as spin
volume=spin.volume
pauli=spin.pauli
quantum=spin.quantum
SOURCE_BYTES={**spin.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
SPIN_SHA='4748e6ee0f9af1f7317b4f634be5cf3792604d432442bfc78e2a3d94d8598386'


def energy(profile:dict[str,Any],terms:list[dict[str,Any]],p:np.ndarray,a:float)->float:
    H=spin.h(profile,terms,p,a);return float(np.sqrt(np.trace(H@H).real/4))


def physical(mean:dict[str,Any],probes:list[float])->dict[str,Any]:
    native=quantum.build_native(mean,4);a=native['a'];gamma=native['frame']/np.sqrt([native['g0']]+[native['gs']]*3)[:,None,None]
    e0,U0=np.linalg.eigh(gamma[0]);B0=U0[:,e0>0];Sz=-1j*gamma[1]@gamma[2]/2
    _,Us=np.linalg.eigh(B0.conj().T@Sz@B0);B=B0@Us;Sigma=2*B.conj().T@Sz@B;Rminus=(np.eye(4)-gamma[0])/2;results=[]
    for profile in native['Pprofiles']:
        terms=volume.primitive(profile);M=energy(profile,terms,np.zeros(3),a);Hx=spin.rest_derivative(terms,a,0,1);Hy=spin.rest_derivative(terms,a,1,1)
        Dx=Rminus@Hx@B/(2*M);Dy=Rminus@Hy@B/(2*M)
        Darwin=.5*Dx.conj().T@Dx;d=float(np.trace(Darwin).real/2)
        SO=Dy.conj().T@Dx;so=float(np.real(np.trace(Sigma@SO)/(-2j)))
        if d<=0 or abs(so-2*d)>1e-10 or np.linalg.norm(Darwin-d*np.eye(2))>1e-10:raise ArithmeticError('Actual microscopic density Darwin/Thomas-spinorbit coefficients failed')
        rows=[]
        for q in probes:
            p=np.array([q/2,0.,0.]);Wp=spin.positive(profile,terms,p,a,B);Wm=spin.positive(profile,terms,-p,a,B);D=Wp.conj().T@Wm
            measured_d=(1-float(np.trace(D).real/2))/(q*q)
            pin=np.array([q,0.,0.]);pout=np.array([0.,q,0.]);Wi=spin.positive(profile,terms,pin,a,B);Wo=spin.positive(profile,terms,pout,a,B);density=Wo.conj().T@Wi;cross=float(np.cross(pout,pin)[2])
            measured_so=float(np.real(np.trace(Sigma@density)/(2j*cross)))
            rows.append(dict(physical_probe=q,actual_scalar_density_Darwin_vertex=spin.control.matrix(D),actual_measured_Darwin_coefficient=measured_d,actual_noncollinear_density_spinorbit_vertex=spin.control.matrix(density),actual_measured_spinorbit_coefficient=measured_so))
        # Exact finite-q latticeWard: linksourcecurrent uses qtilde=
        # (exp(i a q)-1)/(i a), NOT naiveq or shortpathcurrent.
        pairs=[(np.array([.17,.09,.12]),np.array([-.06,.025,.04])),(np.array([.32,-.17,.21]),np.array([.10,.07,-.13]))]
        controls=[]
        for pout,pin in pairs:
            q=pout-pin;Wi=spin.positive(profile,terms,pin,a,B);Wo=spin.positive(profile,terms,pout,a,B);density=Wo.conj().T@Wi;delta=energy(profile,terms,pout,a)-energy(profile,terms,pin,a)
            divergence=np.zeros((4,4),complex);wrong=np.zeros_like(divergence)
            for axis in range(3):
                Q=(np.exp(1j*a*q[axis])-1)/(1j*a);divergence+=Q*spin.current(terms,pout,pin,a,axis)
                # DiscriminatorONLY: keep H but incorrectlyshorten actual2linksum.
                for term in terms:
                    if term['axis']==axis:wrong+=Q*1j*a*term['length']*(term['hop']*np.exp(1j*term['length']*a*pin[axis])-term['hop'].conj().T*np.exp(-1j*term['length']*a*pout[axis]))
            literal=spin.h(profile,terms,pout,a)-spin.h(profile,terms,pin,a);error=float(np.linalg.norm(divergence-literal));projected=float(np.linalg.norm(Wo.conj().T@divergence@Wi-delta*density));wrong_error=float(np.linalg.norm(wrong-literal))
            if max(error,projected)>1e-10 or wrong_error<1e-10:raise ArithmeticError('Actual exactdensity/latticePeierlsWard consumer failed')
            controls.append(dict(physical_pout=pout.tolist(),physical_pin=pin.tolist(),actual_FULL1_2link_operator_Ward_error=error,actual_positiveband_density_energy_current_Ward_error=projected,WRONG_shortened_two_link_pathsum_Ward_defect=wrong_error))
        if abs(rows[-1]['actual_measured_Darwin_coefficient']-d)>2e-5*max(1,d) or abs(rows[-1]['actual_measured_spinorbit_coefficient']-so)>1e-6*max(1,so):raise ArithmeticError('Actual finiteprobe density Foldy coefficient consumer failed')
        results.append(dict(role=profile['label'],load=profile['load'],actual_rest_energy_M=M,actual_Darwin_coefficient=d,actual_spinorbit_coefficient=so,actual_Thomas_relation_spinorbit_over_Darwin=so/d,actual_Foldy_scalar_density_coefficients=spin.control.matrix(Darwin),actual_finiteprobe_density_rows=rows,actual_operator_and_positiveband_density_Ward_controls=controls))
    return dict(a=a,g0=native['g0'],gs=native['gs'],charged_flavours=64,actual_microscopic_density_Foldy_results=results,derivation='Literalrho=I4, soD(pout,pin)=Wplus(pout)^dagWplus(pin). Positive-restprojector derivativeD_iW=PminusHprime_i B/(2M), exactsamepositive-overlapFoldysection as4748. D=I-cm²|q|²/(8M²)+i cm² Sigma.(poutcrosspin)/(4M²)+O(p^4); actualdirectmatrixcoefficients d=.5Dx^dagDx, spinorbitfromDy^dagDx. SO=2Darwin, equivalentlymu/(2M), no empiricalThomasfactor inserted. Exact finitePeierlsWard SUM[(exp(i a q_j)-1)/(i a)]J_Aj=H(pout)-H(pin), projectedto(Eout-Ein)D. ALLlength2path phases/contact provenance retained; naivephysicalq/shortpathsum areNOTWard.',scope='DistinctmicroscopicbareBlochpositiveband/density control, NOToriginalcanonicalY/Gaussian2/E SOURCE orprimitivePi+Coul interactingelectronpole, no photon-normalizedeR/anomaly/commoncone/QEDpromotion. PositivebandD is NOT the densitybubble ofall64sea; that remainsseparate873/7cc/quantumdensityfeedbackbranch. Scalarpotentialoperator read directly fromvertex, no signconventionretag.')


def run(path:Path)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SPIN_SHA:raise ArithmeticError('Frozen cad/4748 spinmagnetic/Foldy epoch required')
    cap=json.loads(raw);mean=cap['input_descriptor']['actual_root_mean_input'];probes=cap['input_descriptor']['physical_q_probes']
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(spin=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),original_source_abi_UNCHANGED=cap['input_descriptor']['original_source_abi_UNCHANGED'],actual_root_mean_input=mean,actual_probes=probes,source='DistinctactualmicroscopicpositivebandFoldy/scalardensitycontrol; NOToriginalE SOURCE.',original_Lambda_UNCHANGED='2^8192',original_chi_UNCHANGED='2^-4096',original_clock_UNCHANGED='4096')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_electron_Foldy_density_Darwin_spinorbit_Ward=physical(mean,probes),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.spin)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actualdensityFoldyconsumer')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_electron_Foldy_density_Darwin_spinorbit_Ward'],elapsed_seconds=r['elapsed_seconds']),indent=2))
