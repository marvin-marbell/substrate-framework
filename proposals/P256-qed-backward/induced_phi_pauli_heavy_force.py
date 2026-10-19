"""Neutral-heavy -Lambda h Pauli Phi-gradient noise, not charged-only force."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from induced_phi_backreaction import PhiConfig,models,scalar_paths
from induced_phi_ir_well_transfer import rational
from induced_phi_em_ygraph_evolving_field_clock import sl

sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_source_metric as metric
pauli=metric.pauli
quantum=pauli.quantum
SOURCE_BYTES={**metric.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
METRIC_SHA='2586ec47ddeb662c414f6dda563f3216d5ed6b7681ba30ebdef1a917837a803e'


def physical(mean:dict[str,Any])->dict[str,Any]:
    native=quantum.build_native(mean,4);V=native['V'];mode=pauli.magnetic_shape(native)
    # Lambda=1 ONLY constructs bare h model; ACTUAL Hheavy=-2^8192 h.
    cfg=PhiConfig(Lambda=1.,**mean['microscopic_config']);load,model,role=models(cfg)[0]
    if load!=1 or role!='neutral heavy -Lambda' or model.improved:raise ArithmeticError('Actual neutral-heavy species/unimproved-law contract changed')
    onsite,coefficients=scalar_paths(model);paths=[]
    for displacement,c in coefficients:
        support=[path for path in native['Pprofiles'][0]['paths'] if np.array_equal(displacement,path['displacement'])]
        if len(support)!=V:raise ArithmeticError('Actual heavy path support mismatch')
        paths.extend({**path,'scalars':c,'hop':path['seam']*np.einsum('m,mij->ij',c,native['frame'])} for path in support)
    heavy=dict(model=model,onsite=onsite*native['frame'][0],paths=paths,load=1,label=role)
    theta=np.zeros(3*V);h=quantum.matter_hamiltonian(native,heavy,theta);gradient=-pauli.operator(native,heavy,theta,mode)
    values,vectors=np.linalg.eigh(h)
    if np.max(values[:2*V])>=0 or np.min(values[2*V:])<=0:raise ArithmeticError('Actual -Lambda h ground not upperfilled2V')
    empty=vectors[:,:2*V];occupied=vectors[:,2*V:];ph=empty.conj().T@gradient@occupied;weights=np.abs(ph)**2
    covariance=float(np.sum(weights));mu=complex(np.trace(occupied.conj().T@gradient@occupied));gaps=values[None,2*V:]-values[:2*V,None]
    if covariance<=1e-8 or abs(mu)>1e-8:raise ArithmeticError('Actual neutral-heavy Pauli force noise/mode mean mechanism failed')
    if any(int(np.sum(abs(d)))!=1 or complex(c[0]).imag or complex(c[0]).real>=0 for d,c in coefficients):raise ArithmeticError('Actual unimproved heavy Wilson nearest-mass contract changed')
    # Literal binary primitive W(k)>=onsite-2sum|c0|; no fittedgap or
    # identification of rounded coefficients with exact declared decimal m.
    massL=F.from_float(float(onsite))-2*sum(abs(F.from_float(float(complex(c[0]).real))) for _,c in coefficients)
    if massL<=0:raise ArithmeticError('Actual heavy primitive uniform mass lower lost')
    gapL=2*massL*sl(rational(mean['centre_enclosure']['g0_lower']))
    lambda_exponent=8192;beta=F(1,8192);heavy_decay_exponent_coefficient=beta*gapL
    norm_squared=float(np.sum(np.abs(gradient)**2));first_moment=float(np.sum(weights*gaps)/covariance)
    return dict(N=4,V=V,neutralheavy_species=1,neutralheavy_EMcharge=0,actual_bare_heavy_model=asdict(model),actual_Hheavy='-2^8192 h_heavy(Phi); upperpositiveh band filled2V; NOT canonicalY/chargednegativeband',heavy_particles=2*V,sourcePhi_probe='SAMECart-unit35 betaSigma_z/sinqx as b50, chargedsource/probability unchanged',actual_heavy_Pauli_gradient_mean_divided_by_Lambda_real=float(mu.real),actual_heavy_Pauli_gradient_mean_divided_by_Lambda_imag=float(mu.imag),actual_heavy_PH_force_covariance_divided_by_Lambda_squared=covariance,RAWcharged_SOURCE_heavy_gradient_Gram_divided_by_Lambda_squared=covariance/2,actual_uncharged_reference_onebody_Frobenius_squared=norm_squared,actual_heavy_PH_excitation_gap_divided_by_Lambda_min=float(np.min(gaps)),actual_heavy_PH_excitation_gap_divided_by_Lambda_max=float(np.max(gaps)),actual_heavy_PH_weighted_initial_energy_moment_divided_by_Lambda=first_moment,uniform_PH_gap_divided_by_Lambda_lower=dict(numerator=str(gapL.numerator),denominator=str(gapL.denominator)),heavy_normalized_heat_spectral_formula='sum_ab |<negative_a|(-delta h)|positive_b>|² exp[-beta*Lambda*(epsilon_b-epsilon_a)] / sum_ab |...|² ; exact frozenPhi heavyground vacuumoffset retained',beta=str(beta),normalized_heavy_PH_heat_upper_log_natural='<= -2^8192 * coefficient',normalized_heavy_PH_heat_log_coefficient_exact=dict(numerator=str(heavy_decay_exponent_coefficient.numerator),denominator=str(heavy_decay_exponent_coefficient.denominator)),SOURCE_heavy_gradient_Gram_Lambda_squared_exponent=2*lambda_exponent,source_probability_before_heat='1/2',source_tensor_factoring='At frozenPhi reference, heavy positivefilled ground factors from ALL64charged+Ref+EM. MeanheavyF=0; heavy/photon mixed heat EXACTzero. Total matter-gradient Phi/Phi heat = chargedPhi/Phi heat + RAWcreatorheat times heavyPHspectralheat (same offsets), but initial SOURCE Gram adds NONZERO Lambda²covariance. No replacement of charged-only b50 metric.',NO_generic_charged_Phi_clock_extension_to_heavy_PH='Applying heavy-gradient creates neutralheavy PH above -Lambda h ground. Existing c42/76ad chargedpacket Phi-ground clock bound CANNOT be reused for this source; needs its own fullPhi/heavy propagation.',scope='Native actualstatic heavyHamiltonian/source PH spectrum ONLY; notfullcanonicalPhi force. Boson/self/Noe kinetic gradients remain outside this observable, not assertedzero. FullPhi-source heavysection/boson kinetic time NOT numerically propagated; no totalforce magneticbridge promotion.')


def run(path:Path)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=METRIC_SHA:raise ArithmeticError('Frozen06db/2586 sourceMetric epoch required')
    cap=json.loads(raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(metric=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],native_N=4,source='UNCHANGEDoriginalPhi222/projectedneutralheavyground/chargedcanonicalY/all64/Gaussian2/AP222/ALL4Refholes. E BEFOREheat; sourceprobability1/2. No source/projector/filter/gate replacement.',actual_Lambda='2^8192',actual_chi='2^-4096',a='1/64',UEphys='1',neutralheavy_EMcharge='0',heavy_count=1,clock='4096',probe_scope='Add actualneutral-heavy potential-gradient contribution to physicallyspecifiedPhi Pauli direction; clarify charged-only b50 versus totalmattergradient. NOT claimfullcanonicalPhi force.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_neutralheavy_Pauli_Phi_force_noise=physical(mean),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metric',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.metric)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during physical heavyforce calculation')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_neutralheavy_Pauli_Phi_force_noise'],elapsed_seconds=r['elapsed_seconds']),indent=2))
