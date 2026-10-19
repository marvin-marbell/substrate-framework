"""Publish actual primitive canonical vertex: <=80plain lines,raw arrays separate."""
from pathlib import Path
import hashlib,json,sys,time,zipfile
ROOT=Path(__file__).resolve().parent
PREFIX='rich_common_source_vertex_20261016_'


def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb')as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def main():
    config=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);run=config['run'];started=time.time()
    if not run.startswith(PREFIX)or'/'in run or'\\'in run:raise ValueError('Owned run required')
    receipt=json.loads((ROOT/(run+'_exit.json')).read_text());data=json.loads((ROOT/(run+'_result.json')).read_text())
    if receipt['returncode']!=0 or not receipt['natural_exit']or receipt['source_input_before']!=receipt['source_input_after']:raise ValueError('Actual immutable natural0 required')
    synthesis_path=Path(str(output)+'_synthesis.txt');manifest_path=Path(str(output)+'_manifest.json');capsule_path=Path(str(output)+'_capsule.zip');receipt_path=Path(str(output)+'_receipt.json')
    for path in (synthesis_path,manifest_path,capsule_path,receipt_path):
        if path.exists():raise FileExistsError('Never overwrite scientific publication')
    field=data['field_word_receipt'];norms=data['raw_source_traces'];lines=[
        'PRIMITIVE CANONICAL SOURCE-CONNECTED CURRENT VERTEX: SAME GEOMETRY H/p/Omega',
        'Fresh original UNMASKED Bdag0 half-tripletM+.5 epsilon/sqrt6,charged DOWN-WHITEphysicalSPIN-UP Ldag2 coefficient1.',
        'rho0=OmegaOmega†,rhoB=Bdag0rho0B0,rhoL=Ldag2rho0L2,rhoBL=Bdag0Ldag2rho0L2B0; L acts FIRST.',
        'N32/N35/N33/N36 on ONE CAR64; jointJ1M1. No oldcompactB,Lsite0,or independently evolved native sector answers.',
        f'Original RAW source traces n0,nB,nL,nBL={[norms[x]for x in ("0","B","L","BL")]}; NOT normalized probabilities.',
        'Z(s,t;eta,theta,u)=Tr exp(i sum eta_a J_a) Utheta[rho0+s rhoB+t rhoL+st rhoBL]Utheta†.',
        'Mixedst log derivatives use ALL trace/partition terms. Mean coefficient=[st]M/n; NOT naive normalizedsectoralternating subtraction.',
        'Physical current component basis a=scalar,vectorx,y,z; total current is their coherent SUM.',
        'Full field-coordinate products and CAR words BEFORE R10/source projection; changed sea/core and rich field outputs KEPT.',
        'Same input Phi7/PW19 and occupiedonehopOmega. R input j<=1 plus2coordinate factors EXACTLY reachesj<=3=84functions.',
        'Fixed polynomial-degree/Gaunt certificate,NOT biggercap/precision/convergence campaign or importednewcoupling.',
        f'Input embedding/R10coordinate matching errors={field["input_embedding_error"]:.6g}/{field["root_R10_coordinate_error"]:.6g}; fullword HaarGramerror={field["Gram_error"]:.6g}.',
        'Disjoint quarkTREEedge0 and whiteedge2 even bilinears commute exactly; fullPhi/R multiplication commutes forANYstate.',
        'Thus primitive firstcrosscommutator=0; nonzero nativePAP firstslopes are a SEPARATE finite noncommuting projected model.',
        'Leadingretarded Delta<J>(u)=-u²/2 <V>+O(u³),V=[DeltaHsource,[Tphi+Trot,Jreceiver]],theta=.3 finite.',
        'SS fieldvertex=[Phi,[Tphi,Phi]] from ACTUALfinite7 matrices,NOT canonicalCCR by fiat; SV/VS=0.',
        'VV fieldvertex=sum_ab Iinv_ab D_a(Rd_source)_i D_b(Rd_receiver)_j,owning phases/spins/hops retained.',
        'For isotropicI: [(d_source.dot.d_receiver)deltaij-(R d_receiver)_i(R d_source)_j]/I.',
        'This supplies an unfitted spin-dependent current tensor,not inserted bare sigma.B or a pointparticle binder.',
        'All scalar/vector/SS/VV interference retained via complex27x27 outputGram,not sum of componentnorms.',
        'Current-dressed offdiagonal tensor is canonical source-connected Weyl current-covariance leadingu²,not S-matrix/amputation.'
    ]
    for direction in ('white','quark'):
        own=data['directions'][direction];con=own['canonical_connected'];control=own['active_source_control'];native=data['native_PAP_comparison'][direction];raw=con['raw']['BL'];tensor=con['connected_Weyl_current_covariance_u2_tensor'];maximum=max(abs(x)for row in tensor for x in row)
        lines.extend([
            f'{direction}pulse: primitive first fieldcomm inputnormmax={own["field_receipt"]["full_coordinate_first_commutator_input_norm_max"]:.6g}; exactCAR modeintersection empty.',
            f'{direction}: nativePAP i<comm>={native["PAP_first_slope"]:.12g},norm²={native["PAP_first_commutator_norm2"]:.12g}; nativeSSnorm²={native["parts_Gram"]["real"][3][3]:.12g}.',
            f'{direction}: rawtrace canonical weights [rho0,rhoB,rhoL,rhoBL]={[con["canonical_linear_trace_weights"][x]for x in ("0","B","L","BL")]}.',
            f'{direction}: CANONICAL connected meanu² total={con["connected_mean_u2_total"]:.12g}; components={con["connected_mean_u2_components"]}.',
            f'{direction}: connected Weylcovarianceu² max|tensor|={maximum:.12g},TOTALcurrentvariance coefficient={con["connected_Weyl_current_variance_u2_total"]:.12g}; fulltensor inrawNPZ.',
            f'{direction}: joint rich Vpsi norm²={raw["vertex_norm2"]:.12g},single-source-return norm²={raw["vertex_elastic_norm2"]:.12g},outside-return signednorm²={raw["vertex_outside_single_source_norm2_signed"]:.12g}.',
            f'{direction}: joint Vpsi R>j1 norm²={raw["vertex_R_above_j1_norm2_signed"]:.12g},newmaterialrawsupport norm²={raw["vertex_new_raw_material_support_norm2"]:.12g}; alloutputs saved,not discarded.',
            f'{direction}: actualsource DeltaH/current/contact norms²={[raw[x]for x in ("active_source_DeltaH_norm2","owning_source_current_norm2","owning_source_contact_norm2")]}.',
            f'{direction}: derivedrotationkinetic derivative-identity error={own["field_receipt"]["rotor_double_commutator_derivative_identity_error_max"]:.6g},axesdot={own["field_receipt"]["source_receiver_axis_dot"]:.6g}.',
            f'{direction}: ACTIVE-SOURCE mediatorOFF wordnorm={control["kinetic_word_norm"]:.6g}; unchanged DeltaHnorm² perrho={control["actual_active_source_DeltaH_norm2"]}.'
        ])
    lines.extend([
        f'ActualPhi7 doublecommutator eigenvalues={data["directions"]["white"]["field_receipt"]["Phi_double_commutator_eigenvalues"]},trace={data["directions"]["white"]["field_receipt"]["Phi_double_commutator_trace"]:.6g}.',
        'MediatorOFF removesONLYsharedPhi kinetic/Trot; ALLmatter source/hops,potential,plaquette,unsharedHE remain active.',
        'In commuting-coordinate law crossresponse then vanishes operatorially forANYstate; sourceDeltaHnormsNONZERO. Not sourceoff/null.',
        'Previous native kinetic-necessity claim rejected: native body/R PAP can respond without field kinetics; its positivefinite mechanism remains distinct.',
        'PW19 kept in reference/output; chosen tree0/white2 have identityU. HE leadingcross cancels by whiteCAR/Phi/R commutation/Jacobi.',
        'These are ACTUAL leadingretarded moments/tensors and fullrich word outputs,NOT an integrated finite-u pulse trajectory.',
        'NEW primitive mechanism WORKS at this order: nonzero canonical source-connected retarded mean in BOTH directions after fullcoordinate firstcomm vanishes.',
        'White-pulse connected currentvariance coefficient is nonzero; reverse variance is zero at rounding level and is PRESERVED,not retuned.',
        'Source-connected coefficients isolate the DECLARED canonical intensity cumulant,not asymptotic scattering particles or amputated coupling.',
        'Spin/charges/SU3/3+1/gauging,negative scalar quadratic/volume and mechanicaldensity INPUTS; no barecoupling/receiver tuning.',
        'Native EM is EXTERNAL MATERIAL probe; not Maxwell/Coulomb/dynamicalphoton/labrotational convection,proton/mass/binding/lifetime/freeCOM.',
        'Firstjoinedmissingpiece: controlled particle/exterior propagation and dynamicalEM embedding of this source-current link;47/48laws remain separate.',
        f'Ownconsumer NATURAL0,unchangedepoch,RSS={receipt["child_peak_RSS_KiB"]}KiB,physicsseconds={data["resource"]["elapsed_seconds"]:.6g}.',
        'Exact fieldlifts/CARdictionary/countpartition outputarchives/rawtensors/sourceinputs/exits separate; fresh review needs no old producer reconstruction.',
        'Actual rooted LSP/positiveAST exercised; no midflighttype-refactor/build/test/lint/formatter or allproject-cleanclaim.'
    ])
    if len(lines)>80:raise ValueError('Plain physics synthesis exceeds80lines')
    synthesis_path.write_text('\n'.join(lines)+'\n');scientific=ROOT/(run+'_capsule.zip');members=[synthesis_path,Path(__file__),Path(sys.argv[1]),scientific,ROOT/(PREFIX+'interface.json'),ROOT/(PREFIX+'language1.json'),ROOT/(PREFIX+'language_publish1.json')];manifest={'kind':'PRIMITIVE_CANONICAL_VERTEX_EXACT_PUBLICATION','synthesis_lines':len(lines),'members':{path.name:{'bytes':path.stat().st_size,'SHA256':sha(path)}for path in members},'scientific_raw_entries':receipt['files'],'source_input_epoch':receipt['source_input_before'],'scope':'Scientific capsule contains ALL exactwordrichoutputs,fieldlifts andCARcoordinates; no producer/sector Hamiltonian reconstruction required. Previousnative/capsules immutable,separate.'};manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');members.append(manifest_path)
    with zipfile.ZipFile(capsule_path,'x',allowZip64=True)as archive:
        for path in members:archive.write(path,path.name,compress_type=zipfile.ZIP_STORED if path.suffix=='.zip'else zipfile.ZIP_DEFLATED)
    delivered={'kind':'ACTUAL_PRIMITIVE_VERTEX_PUBLICATION_NOT_NEW_PHYSICS','capsule':str(capsule_path),'bytes':capsule_path.stat().st_size,'SHA256':sha(capsule_path),'synthesis':str(synthesis_path),'synthesis_lines':len(lines),'manifest_SHA256':sha(manifest_path),'elapsed_seconds':time.time()-started};receipt_path.write_text(json.dumps(delivered,indent=2)+'\n');print(json.dumps(delivered),flush=True)


if __name__=='__main__':main()
