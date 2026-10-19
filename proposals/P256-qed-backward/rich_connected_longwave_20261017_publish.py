"""Package completed actual response and compact physics, never manufacture outcome."""
from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np


def load(path):return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    cfg=load(sys.argv[1]);out=cfg['out'];runs={p:load(p+'_exit.json')for p in cfg['required_runs']}
    for p,r in runs.items():
        if not r['natural_zero']or r['input_epoch_changed']:raise ValueError('No valid completed epoch '+p)
    p=load(cfg['preparation']+'_result.json');w=load(cfg['wave']+'_result.json');r=load(cfg['readout']+'_result.json');receipts=w['numerical_receipts'];full=w['responses']['full'];control=w['responses']['active_cotree_Uidentity'];paired=w['paired_active_control_contrasts'];last=len(w['times'])-1
    circ=load(cfg['readout']+'_local_circulation.json');opposite=np.array(circ['triangle_cells'])==2;curl=circ['circulation']
    def value(which,key,field='source_mean',cell=None):
        z=np.array(which[key][field])[last];return z if cell is None else z[cell]
    fmt=lambda x:np.array2string(np.asarray(x),precision=7,separator=',',max_line_width=10000)
    raw=np.load(cfg['wave']+'_sample_observers.npz');initialE=max(float(np.max(abs(raw[n+'_response_per_unit_source_Efull_edge'][:,0])))for n in('full','active_cotree_Uidentity'));initial2=max(float(np.max(abs(raw[n+'_response_per_unit_source_cell_current_vector'][:,0,2])))for n in('full','active_cotree_Uidentity'));energy=r['energy_exchange'];ecov=w['q_ET_covariance'];joints=p['native_current_images'];lines=[
      'COMPLETED connected local-rich TWA dipole case; INCONCLUSIVE for travelling-field evidence, NOT an exact quantum-photon result.',
      'Four-cell periodic FCC tube L=(4,1,1)*sqrt2,16anchored sites,CAR256,N128; ALL96matter/charge links,32intercell.',
      'Independent quantumPhi7/INTERNALR10/PW19 in EACHcell; endpoint averages onintercell hops are an explicit NEWorder-field rule.',
      'g=.65,v=.6,colour1.8/plaquette.25,negative scalar quadratic/volume2/inertia3 percell remain INPUTS; no B²/omega/BO/mass/spin fit.',
      'Colourtree Gauss derived with4retainedlocalchordholonomies; othercolourpaths follow chosen tree, not fullSU3spatial closure.',
      'Compact U1 microscopic law: Eint=T*Nint+K*m,81integercycle generators and2piZ81 angles; H_E=.5||TN+Km||².',
      'Charge-dependent cycle offsets/cross terms kept; physicalQ/E=Nint/Eint divided3. No substitution by a real longitudinal minimum.',
      'ALL81cycle directions retained, including uniformwinding; qX/4,qX/2,bothy/z sin/cos are OBSERVERS, not imposed globalfield ties.',
      f'Actual8probe rank={r["actual_probe_rank"]}, complementary near-field rank={r["complement_rank"]}, uniformrank={r["uniform_rank_INSIDE81"]} INSIDE81; source-retained projector error={r["source_retained_projector_difference_from_full_cycle"]:.4g}.',
      'No hardGLOBALone-particle-hole closure/sharedPhiRbus. Conserved species algebras U96/U96/U32/U32 allow independent sea fluctuations.',
      'Numerical route: full-dynamic Hamiltonian Lie-Poisson/TWA, exact finite local-operator matrices and actual rho*R*V/Phi*V current products.',
      'Initial fermion/local-field first and symmetrized second moments sampled; higher star-product/local-field moments are approximate.',
      'Trajectory quasi-density matrices may be nonpositive; NEVERclipped/called probabilities. Continuous TWA momenta are NOT integer spectra.',
      'Preparation HF uses Wick exchange/self; trajectory Hamiltonian uses bilinear-symbol Hartree squares plus sampled covariance, not second Fock correction.',
      'Initial compact m0 rotors have independent uniform81angles; n2/-1/-3 quantum harmonics are not silently cut or replaced by meanfields.',
      f'Fresh HOT PREPARED state, NOT vacuum/ground: fermion stationarity residual={p["fermion_stationarity_residual"]:.8g}; inhomogeneous localPhi={fmt(p["local_Phi"])}.',
      f'Prepared localR trace={fmt(p["local_R_trace"])}; legalq labels are NOT stationary Bloch dispersion branches.',
      f'Native vector J-image strengths={fmt([x["vector"]["raw_native_J_image_norm2"]for x in joints])}; R-mean replacement={fmt([x["vector_replace_R_by_mean"]["raw_native_J_image_norm2"]for x in joints])}.',
      f'Actual sampled joint-current second moments={fmt([x["TWA_vector_second_moment"]for x in p["joint_current_covariance_comparison"]])}, MCerrors={fmt([x["MonteCarlo_standard_error"]for x in p["joint_current_covariance_comparison"]])}; NOT probabilities/exact evolved quantum equality.',
      f'Bounded native disjoint-pulse commutator={p["bounded_native_cluster_probe"]["disjoint_density_pulses_commutator_norm"]:.4g}; absent-intercell-link Edot={p["bounded_native_cluster_probe"]["absent_intercell_link_Edot_max"]:.4g}.',
      f'Disjoint WHITE SCALAR +independentPhi flow commutators: rho={circ["native_disjoint_scalar_flow_probe"]["full_species_algebra_rho_commutator_norm"]:.4g}, Phi={circ["native_disjoint_scalar_flow_probe"]["independent_localPhi_flow_commutator_norm"]:.4g}; hard-global-onepair counterexample norm={circ["native_disjoint_scalar_flow_probe"]["hardglobal_onepair_sharedvacuum_COUNTEREXAMPLE_commutator_norm"]:.5g}.',
      f'Universal cell0 nonconstant physical dipole impulse f={w["source_strength"]}; exact Slater1-survival={p["exact_Slater_source_nontriviality_1minus_survival"]:.9g}; not neutralQcell identity/incomingphoton.',
      f'Actual {w["samples"]} matched samples, times={w["times"]}; analytic source tangent evolves ALONG each nonlinearbaseline. O(f²) omitted.',
      'Full/control start from SAME sampled physical input; control is NOT refitted. Baseline drift carried, not static Hessian propagation.',
      'Active control removes ONLY cotree quantum Peierls phases, retaining all matter/localPhiRcolour and full canonicalelectric/Gauss offsets.',
      'Residual control Coulomb/tree/material fields are NOT all transverse radiation; its field transport route deliberately differs from fullH.',
      f'Initial source fullE max={initialE:.4g}, oppositecell2 current max={initial2:.4g}; no initial global-profile field jump or receiver move.',
      'Current bins1/3 can already contain source-adjacent bonds at t0. Oppositecell2 is pinned; least-squares graph E-vector is not continuum Poynting.',
      f'At finaltime oppositecell2 FULL physicalE={fmt(value(full,"cell_Efull_vector",cell=2))}, MCerror={fmt(value(full,"cell_Efull_vector","MonteCarlo_standard_error",2))}.',
      'Helmholtz transverse projection is itself spatially NONLOCAL: distant ET can be compensated by EL. ET alone is NOT propagation evidence.',
      f'Oppositecell2 local CLOSED-triangle Ecirculations={fmt(np.array(curl["full"]["closed_local_cycle_response"]["source_mean"])[last,opposite])}, MCerror={fmt(np.array(curl["full"]["closed_local_cycle_response"]["MonteCarlo_standard_error"])[last,opposite])}.',
      'Local closed-cycle Ecirculation removes gradient/projection tails; it is still not MaxwellB, onephoton, unique radiation or an assigned velocity.',
      f'At t={w["times"][-1]} oppositecell2 FULL transverseE source={fmt(value(full,"cell_Etransverse_vector",cell=2))}, MCerror={fmt(value(full,"cell_Etransverse_vector","MonteCarlo_standard_error",2))}.',
      f'At same t oppositecell2 CONTROL transverseE={fmt(value(control,"cell_Etransverse_vector",cell=2))}, MCerror={fmt(value(control,"cell_Etransverse_vector","MonteCarlo_standard_error",2))}.',
      f'Paired FULL-minus-control oppositecell2 transverseE={fmt(np.array(paired["cell_Etransverse_vector"]["source_difference_full_minus_active_control"])[last,2])}, MCerror={fmt(np.array(paired["cell_Etransverse_vector"]["paired_MonteCarlo_standard_error"])[last,2])}.',
      f'At same t oppositecell2 FULL current={fmt(value(full,"cell_current_vector",cell=2))}, MCerror={fmt(value(full,"cell_current_vector","MonteCarlo_standard_error",2))}.',
      f'At same t FULL cellcharges={fmt(value(full,"Qcell"))}; charge transport/polarization is not an added-particle position probability.',
      f'FULL localPhi source means={fmt(value(full,"Phi"))}, MCerror={fmt(value(full,"Phi","MonteCarlo_standard_error"))}.',
      f'FULL localRtrace means={fmt(value(full,"R_trace"))}, MCerror={fmt(value(full,"R_trace","MonteCarlo_standard_error"))}; colourReTrV means={fmt(value(full,"colour_ReTrV"))}, MCerror={fmt(value(full,"colour_ReTrV","MonteCarlo_standard_error"))}.',
      f'FULL late source electric-energy change={energy["full"]["EM"]["source_mean"][-1]:.8g}, MCerror={energy["full"]["EM"]["MonteCarlo_standard_error"][-1]:.4g}; combined rich matter/fields budget exported.',
      f'At firstpostpulse t={w["times"][1]}, FULL EMenergy response={energy["full"]["EM"]["source_mean"][1]:.8g} +/-{energy["full"]["EM"]["MonteCarlo_standard_error"][1]:.6g}, exchangedwith combinedrichmatter/fields afterinitialprep.',
      f'CONTROL alsohas EMenergy response={energy["active_cotree_Uidentity"]["EM"]["source_mean"][1]:.8g} +/-{energy["active_cotree_Uidentity"]["EM"]["MonteCarlo_standard_error"][1]:.6g} atsame t: energyredistribution is NOT uniquely transverse radiation.',
      f'FULL q_ET covariance source-change Frobenius={np.linalg.norm(ecov["full"][-1]["source_q_ET_covariance_change_LINEAR"]):.8g}; this is approximate quantum covariance, NOT entanglement proof.',
      'Native full/longitudinal/transverse E, current/contact, all73complement channels, uniformresponses, cell/field energies and raw MCsamples exported.',
      'Supported q-dependent time traces and finite-window transforms use SAME generator; six-time window is NOT resolved poles/velocity/branch count.',
      f'Max numerical energy/tangent-energy drift={max(x["energy_drift_max"]for x in receipts):.4g}/{max(x["tangent_energy_drift_max"]for x in receipts):.4g}; trace/Casimir/Hermiticity receipts preserved.',
      f'Max full native field-current Ward={max(x["full_generator_field_current_Ward_max"]for x in receipts if x["full_generator_field_current_Ward_max"]is not None):.4g}; numerical conservation is NOT particle lifetime.',
      'Outcome: oppositecell physicalE is belowits sampling uncertainty; marginal correlated circulation components do NOT establish a travellingwave/front.',
      'Actual hot-state current/energy redistribution and covariance are retained, but this case cannot distinguish coherent radiation from screened/diffusive material response.',
      'No no-photon/pole/velocity inference. Charge3 Wilson electric-circulation input is ONLY a possible later discriminator; NOT launched and NOT a current assignment.',
      'SEVERE: TWA quantum corrections, finitePhi/R/PW/colourholonomy reduction,32sample uncertainty, weaksource/shorttube/time and hot inhomogeneous input.',
      'No exact fullcompact quantum propagation, full sea/vacuum, Lorentz/3Disotropy/Noether spin1, proton/masses/lifetimes or whole-issue closure claimed.',
      'Preserved two natural1 dtype constructor histories are technical failures, not physical no-go. Superseded globalpair/bus proposal was NOT executed as supplier.',
      'Equations/interface: rich_connected_longwave_20261017_equations.json / rich_connected_longwave_20261017_interface.json.'
    ]
    if len(lines)>76:raise ValueError('Plain physics exceeds80line envelope')
    Path(out+'_physics.txt').write_text('\n'.join(lines)+'\n');manifest={};used=set()
    historical=load('rich_connected_longwave_20261017_superseded_proposal.json')
    history_archive=Path('rich_connected_longwave_20261017_superseded_proposal.zip')
    if not history_archive.exists():
        with zipfile.ZipFile(history_archive,'w',zipfile.ZIP_DEFLATED)as h:
            h.write('rich_connected_longwave_20261017_superseded_proposal.json','UNEXECUTED_scope.json')
            for name in historical['files_to_archive_then_remove_from_live_code']:h.write(name,'UNEXECUTED_NOT_SUPPLIER/'+name)
    with zipfile.ZipFile(out+'_native.zip','w',allowZip64=True)as z:
        def add(name,payload):
            if name in used:return
            used.add(name);info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_STORED if name.endswith(('.npz','.zip','.pkl'))else zipfile.ZIP_DEFLATED;z.writestr(info,payload);manifest[name]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
        for prefix in cfg['required_runs']:
            with zipfile.ZipFile(prefix+'_source_inputs.zip')as source:
                for item in source.infolist():
                    name=Path(item.filename).name
                    if not item.is_dir()and name not in used:add(name,source.read(item))
        add(history_archive.name,history_archive.read_bytes())
        for pattern in cfg['native_patterns']:
            paths=sorted(Path.cwd().glob(pattern))
            if not paths:raise FileNotFoundError(pattern)
            for path in paths:
                if path.is_file()and path.name not in used:add(path.name,path.read_bytes())
    Path(out+'_native_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');native=digest(out+'_native.zip');mh=digest(out+'_native_manifest.json');Path(out+'_evidence.json').write_text(json.dumps({'completed_unchanged_natural0':runs,'preserved_technical_runs':{p:load(p+'_exit.json')for p in cfg['preserved_runs']},'native_sha256':native,'manifest_sha256':mh},indent=2)+'\n')
    with zipfile.ZipFile(out+'_capsule.zip','w',zipfile.ZIP_STORED,allowZip64=True)as z:
        for suffix in('_native.zip','_native_manifest.json','_physics.txt','_evidence.json'):z.write(out+suffix,Path(out+suffix).name)
        for prefix in cfg['required_runs']+cfg['preserved_runs']:z.write(prefix+'_capsule.zip',Path(prefix+'_capsule.zip').name)
        z.write(sys.argv[1],Path(sys.argv[1]).name);z.write(__file__,Path(__file__).name)
    capsule=digest(out+'_capsule.zip');receipt={'capsule_sha256':capsule,'native_sha256':native,'manifest_sha256':mh,'physics_lines':len(lines),'natural_runs':len(runs),'preserved_runs':len(cfg['preserved_runs'])};Path(out+'_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');Path(out+'_synthesis.txt').write_text('\n'.join(lines+['Native capsule SHA256 '+native,'Full provenance capsule SHA256 '+capsule,'Native manifest SHA256 '+mh])+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
