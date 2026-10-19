"""Actual residue-selected spatial charged poles, including dark lower virtuals."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye
from rich_spatial_holonomy_20261014_model import build, momentum_basis, fourier_source
from rich_spatial_holonomy_20261014_calculate import expect


def run(p,states_path,pole_path):
    model=build(p); reference=np.load(states_path)['reference'][:,0]; ref=json.loads(pole_path.read_text())['reference']; eg=ref['E']; refcurve=ref['smooth_branches'][0]['d2E_dphi2_all_virtual']; sea=2*p['sites']; length=model['geometry']['circumference']
    w=-p['charge']/p['sites']*sum(model['edgeJ']); contact=-p['charge']**2/p['sites']**2*sum(link+link.conj().T for link in model['links'])
    s2=model['S2']; ident=eye(model['H'].shape[0],format='csr'); half=(s2-3.75*ident)@(s2-8.75*ident)/24
    records=[]
    for n in (sea-1,sea+1):
        for source_m in range(p['sites']):
            target_m=source_m if n>sea else (-source_m)%p['sites']; u=momentum_basis(model,n,target_m)
            h=(u.conj().T@model['H']@u).toarray(); e,v=eigh(h,check_finite=False)
            source=fourier_source(model,source_m); source=source if n>sea else [a.conj().T for a in source]
            columns=np.column_stack([a@reference for a in source]); amp=v.conj().T@(u.conj().T@columns); weights=np.sum(abs(amp)**2,axis=1)
            first=np.flatnonzero(weights>1e-10)[0]; energy=e[first]; selected=np.flatnonzero(abs(e-energy)<1e-8); other=np.flatnonzero(abs(e-energy)>=1e-8)
            b=v[:,selected]; full=u@b; pole_amp=amp[selected]; density=pole_amp@pole_amp.conj().T; weight=float(np.trace(density).real)
            wb=u.conj().T@(w@full); one=b.conj().T@wb; coupling=v[:,other].conj().T@wb
            two=b.conj().T@(u.conj().T@(contact@full))+2*coupling.conj().T@(coupling/(energy-e[other])[:,None])
            slopes,rotation=eigh(one,check_finite=False); branches=[]; seen=set()
            for i,slope in enumerate(slopes):
                if i in seen: continue
                group=np.flatnonzero(abs(slopes-slope)<1e-9); seen.update(group.tolist()); g=rotation[:,group]
                curvatures,rv=eigh(g.conj().T@two@g,check_finite=False)
                for j,curv in enumerate(curvatures):
                    coeff=g@rv[:,j]; state=full@coeff; curvature=(curv-refcurve)*length**2/p['charge']**2
                    branches.append({'slope_flux':float(slope),'mechanical_group_velocity':float(-length/p['charge']*slope),'energy_flux_curvature_raw':float(curv),'source_gap_flux_inverse_inertia':float(curvature),'conditional_band_inertia':float(1/curvature) if curvature>0 else None,'source_residue_branch':float(np.vdot(coeff,density@coeff).real),'S2':expect(s2,state),'jhalf_receiver':expect(half,state),'physical_profile_current':expect(model['J'],state)})
            low=np.flatnonzero(abs(e-e[0])<1e-8)
            records.append({'N':n,'source_m':source_m,'target_m':target_m,'k_source':2*np.pi*source_m/length,'sector_minimum_gap':float(e[0]-eg),'sector_minimum_source_weight':float(sum(weights[low])),'first_actual_bright_gap_same_reference':float(energy-eg),'bright_pole_degeneracy':len(selected),'bright_pole_total_source_weight':weight,'bright_source_weighted_S2':float(np.trace(density@(full.conj().T@(s2@full))).real/weight),'bright_source_weighted_jhalf':float(np.trace(density@(full.conj().T@(half@full))).real/weight),'all_virtual_count':len(other),'lower_energy_dark_virtual_count':int(sum(e[other]<energy)),'smooth_branches':branches})
    return {'candidate':p['candidate'],'reference':{'N':sea,'E':eg,'flux_curvature':refcurve,'source':'exact actual1 reference/pole1 same-H neutral curvature'},'actual_source_pole_dispersion_and_inertia':records,'meaning':'Source-residue-selected actual finite spatial poles; sector minima at m±1 are dark and cannot be labelled electron-source dispersion. Every other finite state, INCLUDING lower dark states and all highfield states, enters analytic current/contact curvature. m0 moving doublet/negative excesscurvature is not positive rest mass. Smooth positive curvature away from k0 is local bandinertia only.','scope':p['scope']}

if __name__=='__main__':
    p=json.loads(Path(sys.argv[1]).read_text()); result=run(p,Path(sys.argv[2]),Path(sys.argv[3])); Path(sys.argv[4]).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
