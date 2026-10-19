"""Exact leading neutral Pauli/photonic source moment from primitive loops."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import induced_reference_electric_graph_enclosure as loops
from induced_phi_ir_well_transfer import rational

sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_gaussian_ward as gaussian
pauli=gaussian.pauli
quantum=pauli.quantum
SOURCE_BYTES={**gaussian.SOURCE_BYTES,**loops.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
WARD_SHA='9953ff62d8b83d1adeecb00cffb873b4e6f91508fa578c155343b985346540fd'


def spin_sign(axes:list[int],insertion:int)->int:
    word=axes[:insertion]+[0,1]+axes[insertion:];mask=0;sign=1
    for axis in word:
        sign*=(-1)**((mask>>(axis+1)).bit_count());mask^=1<<axis
    if mask:raise ArithmeticError('Actual Pauli closed trace notscalar')
    return sign


def closed_four_link()->dict[str,Any]:
    result=defaultdict(int);nonzero=0;sequences=[]
    directions=[(axis,direction) for axis in range(3) for direction in (-1,1)]
    for kinetic in product(directions,repeat=2):
        endpoint=[0,0,0]
        for axis,d in kinetic:endpoint[axis]+=d
        if abs(endpoint[0])!=1 or abs(endpoint[1])!=1 or endpoint[2]!=0:continue
        for Aaxis in (0,1):
            Haxis=1-Aaxis;Adir=-endpoint[Aaxis];Hdir=-endpoint[Haxis]
            for insertion,weight in ((0,1),(1,2),(2,1)):
                position:tuple[int,...]=(0,0,0);flow={};Astart=0;Aend=0
                def step(axis:int,d:int)->None:
                    nonlocal position
                    coordinates=list(position);coordinates[axis]+=d;target=tuple(coordinates);origin=position if d==1 else target;loops.change(flow,origin,axis,d);position=target
                for j in range(3):
                    if j==insertion:Astart=position[0];step(Aaxis,Adir);Aend=position[0]
                    if j<2:step(*kinetic[j])
                step(Haxis,Hdir)
                if position!=(0,0,0):raise ArithmeticError('Actual primitive magnetic loop notclosed')
                # EXACT sin(pi*k/2) for N4; translation sum of unit sin/cos
                # modes gives half[sin(Astart-p)+sin(Aend-p)] on yflows.
                sins=(0,1,0,-1);wave=sum(v*(sins[(Astart-site[0])%4]+sins[(Aend-site[0])%4]) for (site,axis),v in flow.items() if axis==1)
                coefficient=spin_sign([a for a,_ in kinetic],insertion)*kinetic[0][1]*kinetic[1][1]*weight*wave;norm=sum(v*v for v in flow.values())
                result[norm]+=coefficient
                if coefficient:nonzero+=1;sequences.append(dict(kinetic=[list(k) for k in kinetic],Phi0_axis=Aaxis,Phi0_direction=Adir,H_Wilson_axis=Haxis,H_Wilson_direction=Hdir,insertion=insertion,Clifford_weight=weight,integer_flow=[list(k) for k in loops.flow_key(flow)],squared_flow_norm=norm,signed_photon_wave_coefficient=coefficient))
    if dict(result)!={0:0,4:32} or nonzero!=16:raise ArithmeticError('Actual signed four-link Wilson/Pauli Gaussian loop coefficient failed')
    return dict(source_N=4,kinetic_paths=2,Phi0_vertex_paths=1,H_Wilson_paths=1,nonzero_magnetic_plaquette_words=nonzero,signed_coefficients_by_squared_flow_norm=dict(result),primitive_words=sequences,Gaussian_weight='exp[-variance ||ell||²/2]=exp(-4), harmonic/coexact directions RETAINED; closed integerell divergence0 =>Q Qdag ell=ell.',Pauli_trace='Sigma_z=i tau1 tau2; Trspin closedword=2i*Cliffordsign. Two kinetic oriented(-i) signs and photonGaussian insertion i produce real signed coefficients, not an assumed magnetic term.')


def run(path:Path)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=WARD_SHA:raise ArithmeticError('Frozen1b4/9953 physicalWard epoch required')
    cap=json.loads(raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['moment']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['beta']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['metric']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record);native=quantum.build_native(mean,4);enc=mean['centre_enclosure'];iv=loops.iv;g0=iv.mpf([loops.exact(rational(enc['g0_lower'])).a,loops.exact(rational(enc['g0_upper'])).b]);gs=iv.mpf([loops.exact(rational(enc['gs_lower'])).a,loops.exact(rational(enc['gs_upper'])).b]);root0=iv.sqrt(g0);a=loops.exact(F(1,64));g=loops.exact(F(64));s=loops.exact(F(2));p=loops.exact(F(1,2));rows=[];total=iv.mpf(0)
    for profile in native['Pprofiles']:
        c=pauli.scalars(profile)
        if c['path'][2][1]!=0:raise ArithmeticError('Actual leading loop requires nearest-only primitivePhi5')
        c0=loops.exact(c['path'][1][0]);ci=loops.exact(abs(c['path'][1][1]));d=2*loops.exact(abs(c['onsite']))*root0+g
        contribution=p*loops.exact(profile['load'])*32*s/a/iv.sqrt(8)*c0*c0*root0*ci*ci*gs/(d*d)*iv.exp(-2*s);total+=contribution;rows.append(dict(load=profile['load'],actual_neutral_source_Y2_Pauli_Ay_moment=loops.bounds(contribution)))
    loop_control=closed_four_link();macro=total*iv.sin(2*iv.pi/262144)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(ward=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],source='UNCHANGEDoriginalcanonicalY/Gaussian2/ALL64/AP222/ALL4Refholes. Compute ONLY leading neutralcovariance contribution p=sumload Tr(PFQH1) atY²; no SOURCE approximation/cutover/filter.',native_N=4,macro_N=262144,a='1/64',UEphys='1',g_Pi_AND_Coulomb='64',Lambda='2^8192',chi='2^-4096',original_clock='4096')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_primitive_closed_four_link_supplier=loop_control,actual_exact_leading_neutral_source_N4_magnetic_moment=loops.bounds(total),actual_exact_leading_neutral_source_macro_q_magnetic_moment=loops.bounds(macro),all64_leading_neutral_contributions=rows,physical_formula='Mneutral^(Y²)=p sumload [32s/(a sqrt8)] c01² sqrt(g0) (|ci1|² gs)/(2|onsite|sqrt(g0)+UE/a)² sinq exp(-2s). Derived from TrR{Y,A}R{Y,M1} leadingR=I, traceY²AM+2YAYM+AY²M; allorderedstraightprimitivePhi0/kineticpaths and exactclosedflows. Nativeq=pi/2, macroq=2pi/N; no adjustedspeed/field term.',debt='This is an EXACTleading NEUTRAL source covariance term, NOT wholechargedsource moment. HigherR powers and localizedactive rank1 contribution remain unpriced here. FullSAMEsource9953 measuredCARmoment70.8588±9.4775 is separate, not deterministicwholemoment certificate. No derivative/phase/particle/pole/commoncone/retarded/scattering promotion.',scope='ChargedmatterPhi-gradient component only. HeavyLambda²Phi-force noise6af1 and boson/self/Noe gradients notzeroed; Phi/SU4/heavy time not propagated. Macrosine coefficient usesactualrootenclosure and lawspacing, not literalmacroquantumkernel.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ward',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.ward)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actualprimitive loop calculation')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],leading_N4=r['actual_exact_leading_neutral_source_N4_magnetic_moment'],leading_macro_q=r['actual_exact_leading_neutral_source_macro_q_magnetic_moment'],loop=r['actual_primitive_closed_four_link_supplier'],elapsed_seconds=r['elapsed_seconds']),indent=2))
