"""Independent small continuum-product and same-fibre algebra falsifiers, not whole-body acceptance."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply
from rich_rotcov_induced_20261015_model import oscillator,quadrature,field_space,supplier
from rich_rotcov_gauge_20261015_model import london_profiles


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());p=json.loads(Path(request['owner_input']).read_text());osc=oscillator(2);ext=oscillator(4);q=quadrature(osc,p['length'],p['radial_quadrature']);slots=[ext['basis'].index(x) for x in osc['basis']]
    kin=q['u']@q['kinetic'](p['reduced_mass'])@q['u'].conj().T;exact=sum(a@a for a in ext['P'])[np.ix_(slots,slots)]/(2*p['reduced_mass']*p['length']**2);pp=sum(a@a for a in osc['P'])/(2*p['reduced_mass']*p['length']**2);diag=np.diag([(sum(x)+1.5)/(2*p['reduced_mass']*p['length']**2) for x in osc['basis']])
    sup=supplier(p);field=field_space(p['field_cut'],p['field_scale']);nf=len(field['basis']);n=4;qs=[quadrature(oscillator(1),l,p['radial_quadrature']) for l in p['supplier_lengths']];chis=[];squares=[];linear=[];quad=[];profiles=[]
    for f,qq in enumerate(qs):
        original,radial,chi,grad=london_profiles(qq['points'],p);chis.append(qq['integrate'](chi));squares.append(qq['integrate'](np.einsum('xa,xa->x',chi,chi)));linear.append(qq['linear'](radial,p['supplier_masses'][f],-p['charge']));quad.append(p['charge']**2/(2*p['supplier_masses'][f])*qq['integrate'](np.einsum('xai,xbi->xab',radial,radial)))
        _,constant,cc,_=london_profiles(qq['points'],dict(p,gauge_profile='constant'));profiles.append(dict(constant_radial_norm=float(np.linalg.norm(constant)),constant_nonzero_chi=float(np.linalg.norm(cc))))
    pair=lambda a,b:np.kron(a,np.eye(n))+np.kron(np.eye(n),b)
    chi=[pair(chis[0][:,:,a],chis[1][:,:,a]) for a in range(6)];square=pair(squares[0],squares[1])+2*sum(np.kron(chis[0][:,:,a],chis[1][:,:,a]) for a in range(6));bh=square-sum(x@x for x in chi)
    electric=np.kron(np.eye(16),field['E'])+p['charge']**2*np.kron(square,np.eye(nf))/2
    matter=np.kron(sup['H'],np.eye(nf));original_bare=matter.copy()
    for a in range(6):
        electric-=p['charge']*np.kron(chi[a],field['P'][a])
        matter+=np.kron(pair(linear[0][a],linear[1][a]),field['Q'][a])
        for b in range(6):matter+=np.kron(pair(quad[0][:,:,a,b],quad[1][:,:,a,b]),field['QQ'][a][b])
    h=matter+electric;e,u=eigh(h);v=u[:,0];kick=expm_multiply(-.31j*np.kron(np.eye(16),field['Q'][4]),v);baseline=expm_multiply(-.8j*h,v);event=expm_multiply(-.8j*h,kick);wrong=expm_multiply(-.8j*(h-p['charge']**2*np.kron(bh,np.eye(nf))/2),kick)
    mean=lambda state,op:float(np.vdot(state,op@state).real)
    r2=[np.kron(a,np.eye(nf)) for a in sup['r2']]
    # Exact two-mode CAR witness: original-positive = (c_plus†+c_minus†)/sqrt2.
    plus=np.zeros((4,4));minus=np.zeros((4,4))
    for mask in range(4):
        if not mask&1:plus[mask|1,mask]=1
        if not mask&2:minus[mask|2,mask]=(-1)**bool(mask&1)
    vacuum=np.eye(4)[:,0];physical=(plus+minus)/np.sqrt(2);inherited=plus/np.sqrt(2)
    result=dict(kind='BOUNDED_SUPPLIER_FIELD_OPERATOR_CONSUMER_NOT_FULL_CHARGED_BODY',gradient_cut2=dict(full_p2_error=float(np.linalg.norm(kin-exact)),old_Tdiag_error=float(np.linalg.norm(kin-diag)),PpP_squared_error=float(np.linalg.norm(kin-pp)),zero_two_entry=float(kin[0,osc['basis'].index((2,0,0))].real)),electric_projection=dict(BH_min=float(np.linalg.eigvalsh(bh).min()),BH_norm=float(np.linalg.norm(bh)),sum_chi_squared_vs_projected_square=float(np.linalg.norm(square-sum(x@x for x in chi))),field_E_vs_projected_P_squared=float(np.linalg.norm(field['E']-sum(x@x/2 for x in field['P'])))),constant_gradient=profiles,actual_supplier_field=dict(dimension=len(h),ground_energy=float(e[0]),field_only_t0_r2_change=[mean(kick,x)-mean(v,x) for x in r2],event_r2_excess=[mean(event,x)-mean(baseline,x) for x in r2],event_minimal_matter_energy_excess=mean(event,matter)-mean(baseline,matter),coordinate_bare_energy_excess=mean(event,original_bare)-mean(baseline,original_bare),omit_BH_event_distance=float(np.linalg.norm(event-wrong)),norm=float(np.linalg.norm(event))),original_positive_source_falsifier=dict(trigger='rotated-negative holes permitted by new Gamma population moves; two-mode vacuum is exact CAR witness, not claimed actual many-body state',full_source_strength=float(np.linalg.norm(physical@vacuum)**2),old_mirror_positive_strength=float(np.linalg.norm(inherited@vacuum)**2),omitted_component_norm=float(np.linalg.norm((physical-inherited)@vacuum))),limits='Small supplier/field block is independent actual physical reaction control, not whole charged-body/source spectrum. Gradient probe uses actual same helper at cut2. No finite oscillator CCR/Fourier regional proof, Maxwell or full pair closure claimed.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),ground=v,kicked=kick,event=event,baseline=baseline);print(json.dumps(result))


if __name__=='__main__':main()
