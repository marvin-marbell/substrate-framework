#!/usr/bin/env python3
"""NEW bare-electric C reflected primitive; frozen REFERENCE89 is unchanged.

F=[Jchart chi SF_C R_Phi,colour R_E G, sqrt(1-chi²)e^-Hout/2 G].
R_E contains ONLY real electric momentum heat, all coexact+harmonic modes,
plus the solved longitudinal Gauss Coulomb load. NO coexact A²/Mehler/RPA
operator is imported. Reference Hermite/Mehler coordinates remain a BASIS and
packet choice only. No photon canonical trace or harmonic ground is assumed.
Auxiliary scalar+valence spatial and mixed-temporal response must be measured
on this primitive, not inherited from positiveT4 or a supplier's beta label.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any
if __name__=='__main__':
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
import numpy as np
from scipy.linalg import eigh,expm
ROOT=Path(__file__).resolve().parent;SOURCE_BYTES=Path(__file__).read_bytes()
REFERENCE_HASH='89fb263bedabf82c64fb432bb66d672ffc5a4d612abf6ba237c9dc80fa49b6d2'
REFERENCE_CAPSULE_BYTES=(ROOT/'physical_c_real_cover_kernel_smooth_packet_capsule.json').read_bytes()
REFERENCE_BYTES=json.loads(REFERENCE_CAPSULE_BYTES)['execution']['source_bytes_at_IMPORT'].encode()
if hashlib.sha256(REFERENCE_BYTES).hexdigest()!=REFERENCE_HASH:
    raise ArithmeticError('Bare-electric cutover requires exact captured REFERENCE89 primitive')
_snapshot_directory=tempfile.TemporaryDirectory(prefix='physical-C-bare-electric-reference89-')
_snapshot=Path(_snapshot_directory.name)
(_snapshot/'physical_c_real_cover_kernel.py').write_bytes(REFERENCE_BYTES)
for name in ('physical_c_completed_ledger_same_halfstep_capsule.json','physical_alpha_compton_Cartesian_final_native_capsule.json'):
    (_snapshot/name).write_bytes((ROOT/name).read_bytes())
_spec=importlib.util.spec_from_file_location('_physical_C_bare_reference89',_snapshot/'physical_c_real_cover_kernel.py')
if _spec is None or _spec.loader is None:raise RuntimeError('Cannot load captured REFERENCE89')
reference=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=reference;_spec.loader.exec_module(reference)
alpha=reference.alpha


@dataclass(frozen=True)
class BareElectricParameters:
    photon_grade: int=1
    quadrature_order: int=2
    normal_order: int=3
    colour_order: int=1
    normal_sigma: float=.01
    colour_sigma: float=.003
    chart_radius: float=.02
    centre_penalty: float=10.
    electric_coefficient: float=.02
    outside_quartic: float=.02
    outside_couplings: tuple[float,float,float]=(.07,.02,.03)
    prefix_steps: int=4
    bare_mass: float=-2.0282053185612114
    spacing: float=.01


def reference_parameters(p: BareElectricParameters) -> Any:
    return reference.NativeParameters(photon_grade=p.photon_grade,quadrature_order=p.quadrature_order,
        normal_order=p.normal_order,colour_order=p.colour_order,normal_sigma=p.normal_sigma,
        colour_sigma=p.colour_sigma,chart_radius=p.chart_radius,centre_penalty=p.centre_penalty,
        harmonic_free_coefficient=p.electric_coefficient,outside_quartic=p.outside_quartic,
        outside_couplings=p.outside_couplings,prefix_steps=p.prefix_steps,
        critical_bare_mass=p.bare_mass,spacing=p.spacing)


def field_regulator(p: BareElectricParameters) -> dict[str,Any]:
    """Exact compressed FREE momentum heat on ALL real coordinates, no A²."""
    field=reference.field_regulator(reference_parameters(p))
    phdim=field['photon_dimension'];omega=2*math.asinh(1.)
    # The captured basis uses qcoexact=x/sqrt(sinh omega), qharm=x.
    # This normalization is retained ONLY as a basis/preparation choice.
    inverse_width2=np.r_[np.repeat(math.sinh(omega),2),np.ones(3)]
    a=1+p.electric_coefficient*inverse_width2/2
    b=1+p.electric_coefficient*inverse_width2
    vacuum=float(np.prod(a**(-.5)));vacuum_square=float(np.prod(b**(-.5)))
    diagonal=np.full(phdim,vacuum);full_square=np.full(phdim,vacuum_square)
    kinetic=np.full(phdim,p.electric_coefficient*float(sum(inverse_width2))/2)
    if p.photon_grade:
        diagonal[1:]/=a;full_square[1:]/=b
        kinetic[1:]+=p.electric_coefficient*inverse_width2
    yh,yp2=reference.oscillator_h(p.normal_order);xh,xp2=reference.oscillator_h(p.colour_order)
    field['R']=np.kron(np.kron(np.diag(diagonal),expm(-yh/2)),expm(-xh/2))
    free=np.diag(kinetic)
    field['H']=np.kron(np.kron(free,np.eye(p.normal_order)),np.eye(p.colour_order))
    field['H']+=np.kron(np.kron(np.eye(phdim),yh),np.eye(p.colour_order))
    field['H']+=np.kron(np.kron(np.eye(phdim),np.eye(p.normal_order)),xh)
    field['Houtside']=np.kron(np.kron(free,np.eye(p.normal_order)),np.eye(p.colour_order))
    field['Houtside']+=np.kron(np.kron(np.eye(phdim),yp2/2),np.eye(p.colour_order))
    field['Houtside']+=np.kron(np.kron(np.eye(phdim),np.eye(p.normal_order)),xp2/2)
    field['actual_halfheat_omitted_Hermite_operator_norm']=float(math.sqrt(max(full_square-diagonal**2)))
    field['EM_operator']='R_E=exp[-UE/2 sum_ALL5 real Pi²]; corresponding outsideTRANSFER kineticUE sumPi². NO coexactA², oscillatorΩN or RPA magnetic operator.'
    field['exact_free_halfheat_diagonal']=diagonal.tolist()
    field['exact_free_fullheat_compression_diagonal']=full_square.tolist()
    field['reference_coordinate_inverse_width_squared']=inverse_width2.tolist()
    return field


def native_factor(p: BareElectricParameters,number: int,centre: int,field: dict[str,Any],
                  directions: tuple[np.ndarray,np.ndarray] | None=None,
                  normal_reference: bool=False) -> dict[str,Any]:
    """Actual all-column NEW factors, neutral/addition/removal Ref compensation."""
    if number not in (3,4,5):raise ValueError('Native sectors are N3Ref1, N4Ref4/0 and N5Ref3')
    law,phi0,direction=reference.native_law(reference_parameters(p))
    graph=reference.real_graph();e=alpha.charge_contract(alpha.Parameters())['bare_e']
    _,masks=alpha.second_quantized(np.eye(8),number);size=len(masks)
    rho=np.array([[(mask&15).bit_count()-2,(mask>>4).bit_count()-(number-2)] for mask in masks],float)
    if np.max(abs(np.sum(rho,axis=1)))>0:raise ArithmeticError('Actual plus/minus Ref compensation fails real Gauss')
    coulomb=p.electric_coefficient*e**2*np.einsum('ni,ij,nj->n',rho,graph['Lplus'],rho)
    basis=field['basis'];keys=('',) if directions is None else ('','x','y','xy')
    chartjets={key:[] for key in keys};outsidejets={key:[] for key in keys};psis=[];chis=[]
    for cover,y,xi in field['points']:
        phases=(e*graph['Q']@cover).reshape(2,3)
        colour=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex).copy()
        colour[0,0]=np.exp(1j*np.pi*centre/2)*expm(1j*xi*alpha.matrix_unit_basis()[7]/math.sqrt(8))
        remainder=y*np.broadcast_to(direction,(2,4,4,4))
        potential=law.potential(phases,np.zeros_like(remainder) if normal_reference else remainder,colour,directions)
        generators=[]
        for key in keys:
            h,_=alpha.second_quantized(potential['h'+key],number)
            generators.append((h+potential['pref'+key]*np.eye(size))/2)
        exponentials=(expm(generators[0]),) if directions is None else alpha.matrix_exp_jet(*generators)
        chi=reference.cutoff(abs(y),p.chart_radius);chis.append(chi)
        for key,value in zip(keys,exponentials,strict=True):chartjets[key].append(chi*value)
        outside=law.outside_potential(phases,np.broadcast_to(phi0,(2,4,4,4))+remainder,colour,p.outside_couplings,directions)
        for key in keys:
            h,_=alpha.second_quantized(outside['H'+key],number)
            outsidejets[key].append(h+outside['offset'+key]*np.eye(size))
        psis.append(law.source_packet_jet(phases,colour,(np.zeros((2,3)),np.zeros((2,3))))['psi'])
    dim=field['dimension']*size
    r=np.kron(field['R'],np.diag(np.exp(-coulomb/2)))
    chart={key:reference.compress(basis,np.array(chartjets[key]))@r for key in keys}
    coupler=2*(p.outside_couplings[0]+3*sum(p.outside_couplings[1:]))
    young=3*coupler**(4/3)/(4**(4/3)*(p.outside_quartic/2)**(1/3))
    quartic=np.array([sum((np.linalg.norm(phi0+y*direction)**2/4)**2 for _ in range(2)) for _,y,_ in field['points']])
    h0=np.kron(field['Houtside'],np.eye(size))+reference.compress(basis,np.array(outsidejets['']))
    h0+=np.kron(basis.T@(p.outside_quartic*quartic[:,None]*basis),np.eye(size))+2*young*np.eye(dim)
    h0+=np.kron(np.eye(field['dimension']),np.diag(coulomb))
    generators=[-h0/2]+[-reference.compress(basis,np.array(outsidejets[key]))/2 for key in keys[1:]]
    exponents=(expm(generators[0]),) if directions is None else alpha.matrix_exp_jet(*generators)
    outside_weight=np.kron(basis.T@(np.sqrt(1-np.array(chis)**2)[:,None]*basis),np.eye(size))
    out={key:outside_weight@value for key,value in zip(keys,exponents,strict=True)}
    factors={key:np.hstack((chart[key],out[key])) for key in keys}
    k=factors['']@factors[''].conj().T;values,vectors=eigh((k+k.conj().T)/2)
    if values[0]<=0:raise ArithmeticError('Complete bare-electric reflected spectrum unresolved')
    energy=p.centre_penalty*(1-math.cos(np.pi*centre/2))
    result=dict(values=values*math.exp(-energy),vectors=vectors,masks=masks,centre_energy=energy,
        dimension=dim,complete_column_count=factors[''].shape[1],source_packet_blocks=np.array(psis),
        outside_H_min=float(eigh(h0,eigvals_only=True)[0]),outside_factor_norm=float(np.linalg.norm(out[''],'fro')),
        charge_branch='additionRef4to3' if number==5 else 'removalRef0to1' if number==3 else 'neutralRef4or0',
        actual_sumrho_zero=True,literal_REAL_cover=True)
    if directions is not None:result['F']=factors
    return result


def full_family_definition() -> dict[str,Any]:
    return dict(law='NEW_BARE_ELECTRIC_LEFT_C_SF_REFLECTED_TWO_CHANNEL',reference_epoch=REFERENCE_HASH,
        field_space='Same complete CartR60V/SU4^3V/REALR^(2V+1) and physical Gamma(V) Fock+Ref bundle. One compactSU4 Haar and EXACT realGauss quotient; no noncompact Haar/Gibbs trace.',
        primitive='Actual frozen56 Cartesian positiveB completion and intrinsic C-average hC, scalarpref; SF=exp((pref+dGammahC)/2). This is NOT assumed standard4DWilson. Whole scalar+valence staticcurl and mixed-sea-overlap temporal coefficients must be measured/derived. PositiveT4 is separate diagnostic comparator, not an inherited coefficient.',
        chart_EM='ONLY R_E=exp[-UE/2(sum_real_transversePi²+rho.Lplus.rho)], UE=.02 in physicaltheta units for the declared macro candidate. Coexact AND harmonic momentum heat; no coexact A², RPAΩN or imported magnetic action.',
        native_scaling='Native NF0 with effective chargee=.302822..., UE=.02 in native realcover coordinates, not simulation of literal macroQ,Naux,beta. Captured reference coordinates/Hermite widths are ONLY basis/preparation; exact FREE heat compression used on every retained coordinate.',
        baseline='MacroV2^72 even,Nt2^20, auxκ1e-250,Naux6.25e1148 even, active/Ref primitiveQ captured56. No largeNF matrices allocated. Beta16Nκ^4 is a TARGET to test, not a bare magnetic operator.',
        outside='Same positive fullCart quartic+linearPhi C-completed ALLFock law and retained2Eaux. OutsideTRANSFER EM form is UE(sumPi²+rhoLplusrho), so its halfstep matches R_E. NO coexactA²/oscillator. Positive Friedrichs semigroup, nonidentity outside.',
        exact_order='F=[Jchart chi SF R_Phi,colour R_E G,sqrt(1-chi²)exp(-HoutTRANSFER/2)G],K=FFdag. Same colour/normal source frame and connected diagonalZ4 cover, all cyclecharacters, coefficient10. No singleM for entireF.',
        normal_colour='Own massivegap1TRANSFER sigma1e-1500 BEFORE NF; normal tube1e-1000 and colourLie8sigma; physicalDminus=QdQdag. No originalLAW05 inheritance.',
        boundary='Actual normalized realGaussian/Hermite packet W=|Xi><Xi|, optionally smooth-reference spectral preparation with actualGram. Powers use operatornormlambda_N, no finite-volume harmonicground or globalTrK^p.',
        source='SAMEPsi=[exp(iQtheta00)U00;I]/2, addition Ref4to3 and dual removal Ref0to1. Ref charge compensation and centrecount preserved. ONE neutral lambda for both.',
        spectral_units='Hphysical=-log(K/lambdaN)/a. TRANSFERPe[0,4] is PHYSICAL[0,4/a]; PHYSICALPe[0,4] means TRANSFERE<=4a. No mass/pole/CCR inference from finite regulator.',
        physical_electric_magnetic_response_admitted=False)


def actual_current_probe() -> dict[str,Any]:
    p=BareElectricParameters(photon_grade=0,colour_order=1);field=field_regulator(p)
    lam=np.array([.4,-.3]);direction=(reference.real_graph()['D'].T@lam).reshape(2,3)
    rows=[];neutral_lambda=0.
    for number in (4,5,3):
        row=native_factor(p,number,0,field,(direction,direction))
        f,fx,fy,fxy=(row['F'][key] for key in ('','x','y','xy'))
        k=f@f.conj().T;j=fx@f.conj().T+f@fx.conj().T
        contact=fxy@f.conj().T+fx@fy.conj().T+fy@fx.conj().T+f@fxy.conj().T
        if number==4:neutral_lambda=float(eigh(k,eigvals_only=True)[-1])
        rho=np.array([[(mask&15).bit_count()-2,(mask>>4).bit_count()-(number-2)] for mask in row['masks']])
        g=np.kron(np.eye(field['dimension']),np.diag(rho@lam))
        h,hx,hxx=reference.physical_H_jet(k,j,contact,neutral_lambda,p.spacing)
        first=float(np.linalg.norm(hx-1j*(g@h-h@g),'fro')/np.linalg.norm(h,'fro'))
        second=float(np.linalg.norm(hxx+g@g@h-2*g@h@g+h@g@g,'fro')/np.linalg.norm(h,'fro'))
        if max(first,second)>2e-7:raise ArithmeticError('Actual bare-electric Klog spatial current/contact failed')
        rows.append(dict(number=number,actual_Klog_current_Ward=first,actual_Klog_contact_Ward=second,
            ONE_neutral_lambda_N=neutral_lambda,all_complete_columns=row['complete_column_count'],
            actual_outside_factor_norm=row['outside_factor_norm']))
    return dict(rows=rows,scope='Actual newbareelectric COMPLETEchart/outside Klog derivative, Refplus ANDdualminus, smallreal5-coordinate regulator; no h56 current substitution.')


def actual_source_consumer(p: BareElectricParameters) -> dict[str,Any]:
    field=field_regulator(p);law,_,_=reference.native_law(reference_parameters(p))
    mean=law.potential(np.zeros((2,3)),np.zeros((2,4,4,4)),np.broadcast_to(np.eye(4),(2,3,4,4)).copy())
    h,_=alpha.second_quantized(mean['h'],4);_,v=eigh(h);sea=v[:,-1]
    neutral=[];plus=[];minus=[]
    for z in range(4):
        neutral.append(native_factor(p,4,z,field));plus.append(native_factor(p,5,z,field));minus.append(native_factor(p,3,z,field))
    common=max(row['values'][-1] for row in neutral)
    targets=[]
    for n in neutral:
        xi=np.zeros(n['dimension'],complex);xi[:70]=sea/2
        targets.append(n['vectors']@((n['values']/common)**p.prefix_steps*(n['vectors'].conj().T@xi)))
    normalizer=sum(float(np.linalg.norm(s)**2) for s in targets)
    targets=[s/math.sqrt(normalizer) for s in targets]
    incoming=np.zeros((6,6));incoming[1,0]=1
    incoming=np.kron(np.kron(incoming,np.eye(p.normal_order)),np.eye(p.colour_order))
    photon=[(incoming@s.reshape(field['dimension'],70)).reshape(-1) for s in targets]
    gph=sum(float(np.linalg.norm(s)**2) for s in photon)
    rows=[]
    for label,sectors in [('addition',plus),('removal',minus)]:
        raw=half=moment=negative=transfer_band=physical_band=0.
        source_gap_branch=0.
        for n,c,s in zip(neutral,sectors,photon,strict=True):
            coefficients=reference.source_coefficients(field,n['source_packet_blocks'])
            if label=='addition':operator=reference.creation_matrices(n['masks'],c['masks'])
            else:
                operator=reference.creation_matrices(c['masks'],n['masks']).transpose(0,2,1)
                coefficients=coefficients.conj()
            source=reference.apply_source(coefficients,operator,s/math.sqrt(gph))
            energy=-np.log(c['values']/common);spectral=source@c['vectors'].conj()
            weights=np.sum(abs(spectral)**2,axis=0)*c['values']/common
            raw+=float(np.linalg.norm(source)**2);half+=float(sum(weights));moment+=float(weights@energy)
            transfer_band+=float(sum(weights[(energy>=0)&(energy<=4)]))
            physical_band+=float(sum(weights[(energy>=0)&(energy<=4*p.spacing)]))
            negative+=float(sum(weights[energy<0]))
            source_gap_branch+=float(sum(weights[energy<=min(energy)+1e-8]))
        gaps=[-math.log(c['values'][-1]/common) for c in sectors]
        rows.append(dict(branch=label,actual_raw_source_Gram=raw,actual_complete_half_Gram=half,
            actual_mean_TRANSFER=moment/half,actual_mean_PHYSICAL=moment/half/p.spacing,
            literal_TRANSFER_Pe0_4_Gram=transfer_band,literal_PHYSICAL_Pe0_4_Gram=physical_band,
            actual_negative_energy_weight=negative,actual_lowest_gap_branch_source_weight=source_gap_branch,
            actual_top_gaps_TRANSFER=gaps,actual_top_gaps_PHYSICAL=[g/p.spacing for g in gaps]))
    return dict(ONE_neutral_lambda_N=float(common),actual_log_tau=.5*math.log(normalizer),
        actual_incoming_projected_reference_packet_Gram=gph,branches=rows,
        exact_FREE_halfheat_regulator=field['actual_halfheat_omitted_Hermite_operator_norm'],
        no_coexact_quadratic_A2_or_RPA_operator=True,
        centre_spectra=[dict(centre=z,N4_count=len(n['values']),N5_count=len(c['values']),N3_count=len(d['values']),
            N4_columns=n['complete_column_count'],N5_columns=c['complete_column_count'],N3_columns=d['complete_column_count'],
            outside_N4_H_min=n['outside_H_min'],outside_N4_factor_norm=n['outside_factor_norm']) for z,(n,c,d) in enumerate(zip(neutral,plus,minus,strict=True))],
        scope='ActualNEWbare-electric real5mode/Naux0/normal3/colour1/all4centrecycles/all70N4 and56N3,N5 columns; bothchart/outside positivefactors. Reference photon packet basis ONLY, not already-derived Maxwell photons or physical Diracpole.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();start=time.time();p=BareElectricParameters()
    result=dict(full_family_definition=full_family_definition(),parameters=asdict(p),
        actual_new_bare_electric_source_consumer=actual_source_consumer(p),actual_current_contact=actual_current_probe())
    inputs=json.dumps(dict(parameters=asdict(p),reference_epoch=REFERENCE_HASH,NEW_bare_electric_only=True),sort_keys=True,separators=(',',':'))
    result['execution']=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        frozen_reference_sha256=REFERENCE_HASH,frozen_reference_bytes=REFERENCE_BYTES.decode(),
        frozen_Alpha56_sources=alpha.SOURCE_HASHES,input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),
        runtime_seconds=time.time()-start,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,platform=platform.platform()),
        own_startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],input_sha256=result['execution']['input_sha256'],
        consumer=result['actual_new_bare_electric_source_consumer'],runtime_seconds=result['execution']['runtime_seconds'])))
if __name__=='__main__':main()
