#!/usr/bin/env python3
"""Full-field physical charged trial cost from the exact neutral ground, no BO cut."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import expm

from full_band_gauss_section import FullBandGaussSection
from gauss_charge_intertwiner import boundary_word, rectangular_ladder
from pair_full_field_concentration import constants, ground_moment_bound, source_hashes as inherited_hashes
from quantum_su4_gauss import PrimitiveMatter, norm, require_small, sparse_sum
from rotation_band import RotationBand
from unrestricted_field_minimum import ALIGNMENT, BARE, STIFFNESS, matrices, pair_clifford_reference

ROOT=Path(__file__).resolve().parent


def source_hashes():
    return {**inherited_hashes(),**{
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('gauss_charge_intertwiner.md','pair_full_field_charge_transfer.py','pair_full_field_charge_transfer.md')}}


def lift(matter, h, species):
    terms=[]
    for cell in range(2):
        for i in range(4):
            for j in range(4):
                coefficient=h[4*cell+i,4*cell+j]
                if coefficient!=0:
                    terms.append(coefficient*matter.units[cell,species,i,j])
    for i in range(4):
        for j in range(4):
            forward=matter.forward[species,i,j]
            if h[i,4+j]!=0:
                terms.extend((h[i,4+j]*forward,h[4+j,i]*forward.getH()))
    return sparse_sum(terms).tocsr()


def reflection(code, matter):
    """Lift cell interchange with the ACTUAL marker-word Fock phases."""
    words=[boundary_word(code,i,True) for i in range(16)]
    def amplitude(orbitals):
        mask,value=0,1.+0j
        for orbital in reversed(orbitals):
            new=mask|(1<<orbital)
            value*=code.element(mask,new,words[orbital],sign=1)
            mask=new
        return mask,value
    rows,values=[],[]
    for mask in matter.masks:
        occupied=tuple(i for i in range(16) if mask>>i&1)
        _,original=amplitude(occupied)
        target,transformed=amplitude(tuple((i+8)%16 for i in occupied))
        rows.append(matter.index[target]);values.append(transformed/original)
    return sparse.csr_matrix((values,(rows,np.arange(matter.dimension))),
        shape=(matter.dimension,matter.dimension))


def charged_map(model, charge):
    matter=PrimitiveMatter(model.code,4+charge,model.generators)
    modes=tuple(range(4,8))+tuple(range(12,16))
    ladders=[rectangular_ladder(model.code,model.matter.masks,matter.masks,i,charge==1)
        for i in modes]
    columns=[sparse.csr_matrix(np.eye(4)[:,i:i+1]) for i in range(4)]
    mapping=sparse_sum(sparse.kron(ladders[4+i],columns[i],format='csr') for i in range(4))
    boundary=model.generators if charge==-1 else -model.generators.transpose(0,2,1)
    return matter,ladders,columns,mapping,boundary


def charge_bound(Lambda, law, *, stiffness, alignment) -> dict[str,Any]:
    moments=ground_moment_bound(Lambda,law,stiffness=stiffness,alignment=alignment)
    deviation=np.sqrt(moments['expected_average_norm_deviation_squared_upper'])
    mean=np.asarray(law['optimal_average_norms'])+deviation
    cost=float((law['light_u_coefficient']*np.sqrt(mean[0])
        +law['light_v_coefficient']*np.sqrt(mean[1]))/4)
    limiting=float((law['light_u_coefficient']*np.sqrt(law['optimal_average_norms'][0])
        +law['light_v_coefficient']*np.sqrt(law['optimal_average_norms'][1]))/4)
    return dict(Lambda=Lambda,neutral_ground_moment_upper=moments,
        neutral_average_u_v_upper=mean.tolist(),both_charged_minus_neutral_ground_energy_upper=cost,
        large_Lambda_charge_cost_upper_limit=limiting,
        exact_trial_cost_identity='Etrial(q=+/-1)-Eneutral=-<Hlight>neutral/4',
        exact_charge_trial_norm_squared=2.,rate='upper cost tends finite limit, correction O(Lambda^-1/12)',
        scope='full physical field/link/phase domains; charged-ground upper bound, not a lower mass gap or computed quantum eigenstate')


def run(*,Lambdas,stiffness=STIFFNESS,alignment=ALIGNMENT,ue4=.3,ue1=.4,magnetic=.7):
    if not all(np.isfinite(v) for v in (stiffness,alignment,ue4,ue1,magnetic)):
        raise ValueError('Finite law coefficients required')
    if min(stiffness,ue4,ue1)<=0 or min(alignment,magnetic)<0 or not Lambdas:
        raise ValueError('Positive stiffness/electrics, nonnegative J/K and nonempty Lambdas required')
    identities=source_hashes()
    model=FullBandGaussSection(RotationBand(**BARE),heavy_scale=1.)
    frame,reference=pair_clifford_reference(model,stiffness=stiffness)
    anchor=np.tile(frame,(2,1,1,1))
    law=constants(model,anchor,stiffness=stiffness,alignment=alignment,
        ue4=ue4,ue1=ue1,magnetic=magnetic,chart_radius=.001)
    parity=reflection(model.code,model.matter)
    identity=sparse.eye(model.matter.dimension,format='csr')
    parity_residual=norm(parity@parity-identity)
    require_small(parity_residual,'actual marker-word reflection involution')
    nright=4-model.matter.light_left
    sector_reports=[]
    rng=np.random.default_rng(222)
    samples=[]
    for _ in range(8):
        fields=matrices(rng.normal(size=(2,4,15))*.3,model.basis)
        U=expm(1j*np.einsum('a,aij->ij',rng.normal(size=15)*.2,model.generators))
        phases=rng.uniform(-np.pi,np.pi,2)
        reflected=fields[::-1].copy();reflected[:,1]*=-1
        h=model.onebody(fields,U,phases)
        reflected_h=model.onebody(reflected,U.conj().T,phases[::-1])
        for species in range(2):
            require_small(norm(parity@model.lift(h[species],species)@parity.getH()
                -model.lift(reflected_h[species],species)),'actual all-field native reflection',2e-8)
        samples.append(h)
    section=model.state(anchor,np.eye(4))
    light=model.lift(section.onebody[1],1)
    heavy=model.lift(section.onebody[0],0)
    probe_Lambda=1e6
    quartic=stiffness*(reference['temporal_amplitude']**4+reference['x_spatial_amplitude']**4)
    neutral_H=-probe_Lambda*heavy+light+probe_Lambda*quartic*identity
    neutral_energy=float(np.vdot(section.state,neutral_H@section.state).real)
    require_small(norm(neutral_H@section.state-neutral_energy*section.state),'actual frozen Pauli eigenstate',2e-8)
    require_small(abs(float(np.vdot(section.state,nright*section.state).real)-2.),'actual anchor right occupation')
    for charge in (-1,1):
        matter,ladders,columns,mapping,boundary=charged_map(model,charge)
        ib=sparse.eye(4,format='csr')
        im=sparse.eye(matter.dimension,format='csr')
        expected=4-nright if charge==1 else nright
        residuals=dict(norm_operator=norm(mapping.getH()@mapping-sparse.diags(expected,dtype=float)),
            colour_Gauss=0.,EM_Gauss=0.,wrong_boundary_sign=0.,omitted_boundary=0.,
            common_heavy=0.,light_commutator=0.)
        for cell in range(2):
            for a in range(15):
                target=sparse.kron(matter.Q[cell][a],ib,format='csr')
                b=sparse.kron(im,sparse.csr_matrix(boundary[a]),format='csr') if cell==1 else sparse.csr_matrix(target.shape)
                residuals['colour_Gauss']=max(residuals['colour_Gauss'],norm((target+b)@mapping-mapping@model.matter.Q[cell][a]))
                if cell==1:
                    residuals['wrong_boundary_sign']=max(residuals['wrong_boundary_sign'],norm((target-b)@mapping-mapping@model.matter.Q[cell][a]))
                    residuals['omitted_boundary']=max(residuals['omitted_boundary'],norm(target@mapping-mapping@model.matter.Q[cell][a]))
        for cell in range(2):
            source=model.matter.light_left-2 if cell==0 else nright-2
            target=matter.light_left-2 if cell==0 else matter.light_number-matter.light_left-2-charge
            residuals['EM_Gauss']=max(residuals['EM_Gauss'],norm(sparse.kron(sparse.diags(target,dtype=float),ib)@mapping-mapping@sparse.diags(source,dtype=float)))
        for hH,hL in samples:
            HH0=model.lift(hH,0);HHq=lift(matter,hH,0)
            HL0=model.lift(hL,1);HLq=lift(matter,hL,1)
            residuals['common_heavy']=max(residuals['common_heavy'],norm(sparse.kron(HHq,ib)@mapping-mapping@HH0))
            changes=[]
            for i in range(4):
                coefficients=hL[:,4+i] if charge==1 else -hL[4+i,:]
                component=sparse_sum(coefficients[j]*ladders[j] for j in range(8))
                changes.append(sparse.kron(component,columns[i],format='csr'))
            exact=sparse_sum(changes)
            residuals['light_commutator']=max(residuals['light_commutator'],norm(sparse.kron(HLq,ib)@mapping-mapping@HL0-exact))
        for label,value in residuals.items():
            if label not in ('wrong_boundary_sign','omitted_boundary'):
                require_small(value,label,2e-8)
        if min(residuals['wrong_boundary_sign'],residuals['omitted_boundary'])<1e-5:
            raise AssertionError('Actual external boundary adverse case vanished')
        HHq=lift(matter,section.onebody[0],0);HLq=lift(matter,section.onebody[1],1)
        charged_H=sparse.kron(-probe_Lambda*HHq+HLq+probe_Lambda*quartic*im,ib,format='csr')
        image=mapping@section.state
        weight=float(np.vdot(image,image).real)
        energy=float(np.vdot(image,charged_H@image).real)/weight
        light_energy=float(np.vdot(section.state,light@section.state).real)
        predicted=-light_energy/4
        require_small(abs(energy-neutral_energy-predicted),'correct normalized charged trial gap',2e-8)
        wrong=float(np.vdot(image,charged_H@image).real)-neutral_energy
        if abs(wrong-predicted)<1e5:
            raise AssertionError('False unit-norm creator control did not reach common-scale error')
        sector_reports.append(dict(charge=charge,boundary='4' if charge==-1 else 'bar4',
            original_Pauli_dimension=matter.dimension,with_boundary_dimension=4*matter.dimension,
            all_native_Nright_counts={str(n):int(np.count_nonzero(nright==n)) for n in range(5)},
            residuals=residuals,frozen_anchor_Lambda=probe_Lambda,norm_squared=weight,
            frozen_anchor_neutral_energy=neutral_energy,frozen_anchor_charged_trial_energy=energy,
            measured_frozen_trial_gap=energy-neutral_energy,exact_frozen_trial_gap=predicted,
            false_unit_norm_creator_gap=wrong,
            freeze_scope='native pointwise matter/creator control only, not the full quantum neutral ground'))
    rows=[charge_bound(Lambda,law,stiffness=stiffness,alignment=alignment) for Lambda in Lambdas]
    if source_hashes()!=identities:
        raise RuntimeError('Source/proof law changed during execution')
    return dict(source_sha256=identities,source_bindings_unchanged=True,
        governing_plan='issue-222-continuing-backward-20260930',work_item='campaign',
        bare_coefficients={**BARE,'stiffness':stiffness,'alignment':alignment,'ue4':ue4,'ue1':ue1,'magnetic':magnetic},
        exact_neutral_domain='physical neutral NH4 NL4 fullL2(R120 x SU4 x T2) x4900',
        exact_charged_domains='NH4 NL3/5 and genuine external fundamental/antifundamental at right; fullR120/SU4/T2, all3920x4 fibres',
        charge_extension='same full coordinate Hamiltonian; boundary only in Gauss constraints, no added boundary/tree q² Hamiltonian',
        reflection_control=dict(native_involution=parity_residual,arbitrary_all120_noncentral_phase_points=len(samples)),
        sector_controls=sector_reports,uniform_analytic_constants=law,rows=rows,all_checks_pass=True,
        limits=['Analytic exactground trial/cost bounds, not computed full quantum eigenstates or interval certificates.',
            'Constant creator intertwinning cancels common full Cartesian/link/phase/heavy operator; no scalar inertia or BO/light/field cut.',
            'q denotes light-number departure with fixed compensating colour/EM reference at R; no isolated or separated charge/reference energy bound.',
            'No positive charged lower gap, propagating packet, mechanical spin, bulk compact phase, radiative/Lorentz closure or campaign completion.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Lambda',type=float,nargs='+',default=[1e20,1e24,1e30,1e36,1e48])
    parser.add_argument('--stiffness',type=float,default=STIFFNESS)
    parser.add_argument('--J',type=float,default=ALIGNMENT)
    parser.add_argument('--ue4',type=float,default=.3)
    parser.add_argument('--ue1',type=float,default=.4)
    parser.add_argument('--magnetic',type=float,default=.7)
    parser.add_argument('--output',type=Path,default=ROOT/'pair_full_field_charge_transfer_receipt.json')
    args=parser.parse_args()
    result=run(Lambdas=args.Lambda,stiffness=args.stiffness,alignment=args.J,
        ue4=args.ue4,ue1=args.ue1,magnetic=args.magnetic)
    text=json.dumps(result,indent=2)
    args.output.write_text(text+'\n');print(text)


if __name__=='__main__':
    main()
