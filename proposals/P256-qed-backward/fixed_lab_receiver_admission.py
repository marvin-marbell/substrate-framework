"""Outward native admission for the UNCHANGED fixed LAB receiver function.

A unit normal-linear/one-photon witness is integrated with Arb balls. This is
not a sampling norm estimate or normalization in the cached charged Gram.
The compact SOURCE comparison theorem is supplied by native_receiver_norm_certificate;
its mean LAPACK evaluation is not used in the positivity decision here.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from flint import arb, acb, acb_mat, ctx
from prepared_neutral_canonical_source import LABELS
from noncompact_screened_gauss_dressing import source_paths
from native_receiver_norm_certificate import compact_frame_constants


def ball(value: Any) -> acb:
    z=complex(value)
    return acb(arb(z.real),arb(z.imag))


def bound(value: arb | int) -> dict[str,str]:
    value=arb(value)
    return {'ball':str(value),'lower':str(value.lower()),'upper':str(value.upper())}


def admit_native(model: Any,data: dict[str,Any],f2: np.ndarray) -> dict[str,Any]:
    """Actual cutoff Hilbert-space rejection; all stored inputs interpreted exactly.

    Project only the MEAN comparison onto a unit n5 x hard2 witness. Reverse
    triangle inequality then pays the entire actual-minus-mean SOURCE image,
    the mean cutoff tail, and ALL original nonhard extras. Actual F2 is unchanged.
    """
    ctx.prec=128
    if model.law.size!=3 or model.sigma!=2.**-13 or model.radius!=.012:
        raise ValueError('This analytic support theorem is native N3 only')
    constants=compact_frame_constants(model)
    # The support theorem uses a positive diagonal Clifford mean, not a fit.
    from sea_selection import SELECTED
    for fields in model.chart.mean_phi:
        for m in range(4):
            a=(sum(ball(fields[m,i,j])*ball(SELECTED[m,j,i]) for i in range(4) for j in range(4))/4).real
            if not a*a>=arb('.06'):raise ValueError('Unpaid outward mean Clifford lower')
            if any(ball(fields[m,i,j])-a*ball(SELECTED[m,i,j])!=0 for i in range(4) for j in range(4)):
                raise ValueError('Mean is not EXACT diagonal Clifford; cannot bypass frame SVD')
    coordinates=model.charged_coordinates
    r=acb_mat([[ball(z) for z in row] for row in coordinates])*acb_mat([[ball(z)] for z in f2])
    def coefficient(t: int,w: int,b: int,n: int,k: int) -> acb:
        return r[((((t*4+w)*2+b)*7+n)*15+k),0]
    paths=source_paths(model.law,model.reference_site)
    tau=arb(36).tanh()
    profiles=list(model.source_bank.profiles.values())
    # Exact hard Hermite degree-one projection, including actual packet overlaps.
    packets=model.noether_bank.hard.packets
    labels=model.noether_bank.hard.labels
    target_mode=labels[2].index(1)
    hardnorm=sum((ball(z).conjugate()*ball(z) for z in packets[:,target_mode]),acb(0)).real.sqrt()
    hardoverlap={k:sum(ball(packets[i,target_mode]).conjugate()*ball(packets[i,label.index(1)]) for i in range(52))/hardnorm
                 for k,label in enumerate(labels) if sum(label)==1}
    # The runtime polynomial uses binary64 sqrt(2); bind even that scale.
    scale=arb(float(np.sqrt(2)))/arb(2).sqrt()
    if not abs(scale-1)<arb('1e-15'):raise ValueError('Runtime Hermite scale changed')
    hardoverlap={k:scale*v for k,v in hardoverlap.items()}
    directions=model.original.directions
    normalnorm=sum((arb(float(x))**2 for x in directions[:,4]),arb(0)).sqrt()
    normaloverlap={n:sum(arb(float(directions[i,4]))*arb(float(directions[i,n-1])) for i in range(len(directions)))/normalnorm for n in range(1,7)}
    normaloverlap={n:scale*v for n,v in normaloverlap.items()}
    # Mean BARE marks: no positive-band replacement. Seeded CAR projection occurs next.
    upper_raw=arb(0);marks={}
    for charge in range(2):
        for site in range(model.law.volume):
            scal=tau**len(paths[site])
            for i in range(4):
                for ref in range(4):
                    witness=acb(0)
                    for n in range(7):
                        for k in range(15):
                            value=acb(0)
                            for t,eta in enumerate(profiles):
                                for w in range(4):
                                    entry=ball(eta[site])*ball(LABELS[w,i,ref]) if charge==0 else -ball(eta[site]).conjugate()*ball(LABELS[w,ref,i]).conjugate()
                                    value+=scal*entry*coefficient(t,w,charge,n,k)
                            upper_raw+=value.real**2+value.imag**2
                            if n in normaloverlap and k in hardoverlap:witness+=normaloverlap[n]*hardoverlap[k]*value
                    marks[charge,site,i,ref]=witness
    # Pointwise raw SOURCE norm upper from Hermite modes degree<=2. Packet Gram
    # eigenvalues in [1-eps,1+eps] imply Fock Gram upper <=(1+eps)^2.
    packet_error=max(sum(abs(sum(ball(packets[i,a]).conjugate()*ball(packets[i,b]) for i in range(52))-(1 if a==b else 0)) for b in range(4)) for a in range(4))
    normal_error=max(sum(abs(sum(arb(float(directions[i,a]))*arb(float(directions[i,b])) for i in range(len(directions)))-(1 if a==b else 0)) for b in range(6)) for a in range(6))
    # Removal of mean centre attenuation only enlarges the value bound. 1e-10
    # also pays stored degree<=2 sqrt constants and seed contraction error.
    upper_raw*=(1+packet_error)**2*(1+normal_error)*arb('1.0000000001')/tau**(2*max(map(len,paths)))
    if not upper_raw<100:raise ValueError('Native comparison raw SOURCE upper100 unpaid')
    fibre=model.source_bank.seed_fibre
    norm=arb(0)
    declared_norm=arb(0)
    # Runtime declares the labelled CAR fibre orthonormal:
    # FullSlaterFibre.overlap returns I (slater_fibre.py156-157), and
    # source_vectors_from_marks272-276 supplies these exact coefficients.
    # ALSO pay the conservative raw-orbital interpretation explicitly.
    orbitals=acb_mat([[ball(z) for z in row] for row in fibre.orbitals])
    orbital_gram=orbitals.conjugate().transpose()*orbitals
    orbital_error=max(sum((abs(orbital_gram[a,b]-(1 if a==b else 0)) for b in range(fibre.n)),arb(0)) for a in range(fibre.n))
    if not orbital_error<arb('1e-10'):raise ValueError('Original full Slater conditioning debt unpaid')
    # Every light wedge has degree <= n. The normalized/unnormalized wedge
    # Gram comparison therefore costs at most this ratio, without replacing
    # any seed or normalizing the physical function.
    wedge_lower=((1-orbital_error)/(1+orbital_error))**fibre.n
    wedge_norm_upper=(((1+orbital_error)/(1-orbital_error))**fibre.n).sqrt()
    upper_raw*=wedge_norm_upper**2
    displacement=(model.law.sites-model.law.sites[model.reference_site])
    kernel=acb_mat([[(-arb(sum(int(displacement[x,a]-displacement[y,a])**2 for a in range(3)))/36).exp() for y in range(model.law.volume)] for x in range(model.law.volume)])
    seed_errors=[]
    for charge,seed in ((0,fibre.empty),(1,fibre.occupied[-1])):
        seedgram=acb_mat([[ball(z) for z in row] for row in seed]).conjugate().transpose()*acb_mat([[ball(z) for z in row] for row in seed])
        err=max(sum(abs(seedgram[a,b]-(1 if a==b else 0)) for b in range(seed.shape[1])) for a in range(seed.shape[1]))
        seed_errors.append(bound(err))
        if not err<arb('1e-10'):raise ValueError('Fixed seed CAR conditioning debt unpaid')
        rows=[]
        for site in range(model.law.volume):
            rows.append([sum(ball(seed[4*site+i,mode]).conjugate()*marks[charge,site,i,ref] if charge==0 else marks[charge,site,i,ref]*ball(seed[4*site+i,mode]) for i in range(4)) for ref in range(4) for mode in range(seed.shape[1])])
        C=acb_mat(rows);KC=kernel*C
        branch_norm=sum(((C[x,j].conjugate()*KC[x,j]).real for x in range(C.nrows()) for j in range(C.ncols())),arb(0))
        declared_norm+=branch_norm
        # E(EdaggerE)^-1 Edagger physical seed projection: lambda_max(G)
        # <=1+err, hence its squared image norm >=||Edagger marks||²/(1+err).
        norm+=branch_norm/(1+err)
    norm*=wedge_lower
    # Outward analytic constants, intentionally looser than the supplier's floats.
    core=arb('.006');delta=core/arb(2).sqrt();s0=2*arb('.06').sqrt();gap=arb('1.99')
    system=8*delta/(2*s0-delta);least=2*arb(2).sqrt()*system/gap
    L=arb('2.1')*2/(1-least)*2*arb(2).sqrt()/gap*8/(2*s0-delta)
    birth=L*arb(2)**-13*(arb('15.5')*upper_raw).sqrt()
    frame=L*delta;edge=2*frame+arb('.3').sqrt()*core
    score=4-2*arb(2).sqrt()*edge
    centre=max(map(len,paths))*(6*(-18*score).exp()+6*arb(-72).exp())*upper_raw.sqrt()
    D=model.chart.normal_dim;radius=2*core**2/(arb(2)**-13)**2
    logtail=-(radius-D-D*(radius/D).log())/2
    tail=upper_raw*arb(61).sqrt()*(logtail/2).exp()
    # ALL original nonhard terms, directly bounded with balls. |sin|, |cos|
    # <=1; Gaussian linear L2 norm ||v||/sqrt2, not scalar Gram sampling.
    c=model.noether_bank.coefficients
    def euclidean(values: Any) -> arb:
        return sum((abs(ball(z))**2 for z in values),arb(0)).sqrt()
    gamma=sum(euclidean(flow)/arb(2).sqrt()+abs(sum(ball(a)*ball(b) for a,b in zip(flow,phase)))/2
              for flow,phase in zip(c['coexact'],c['phases']))
    extra=arb(0)
    prepared=[euclidean(data['preparation_state14'][7*b:7*b+7])*arb(1+normal_error).sqrt() for b in range(2)]
    for t,eta in enumerate(profiles):
        eta_norm=euclidean(eta)
        for w in range(4):
            for b in range(2):
                common=arb(0);relative=arb(0)
                for row in range(len(c['phases'])):
                    rho=sum(ball(c['lambda_C'][row,x])*ball(model.noether_bank._neutral_rho[b,x]) for x in range(model.law.volume))
                    harmonic=euclidean(c['harmonic'][row])/arb(2).sqrt()
                    common+=abs(rho)+harmonic
                    rel=[ball(c['lambda_C'][row,x])-ball(c['lambda_C'][row,model.reference_site])-sum(ball(c['harmonic'][row,a])*arb(int(displacement[x,a]))/3 for a in range(3)) for x in range(model.law.volume)]
                    relative+=sum((abs(ball(eta[x])*rel[x])**2 for x in range(model.law.volume)),arb(0)).sqrt()
                offset=((t*4+w)*2+b)
                extra+=prepared[b]*(abs(r[1680+offset,0])*eta_norm*gamma+
                    abs(r[1680+16+offset,0])*eta_norm*common+
                    abs(r[1680+32+offset,0])*relative)
    extra*=wedge_norm_upper
    lower=(norm-tail).sqrt()-birth-centre-extra
    if not lower>1:raise ValueError('Witness does not reject; do not infer admission from a cached metric')
    upper=upper_raw.sqrt()+extra
    return dict(decision='REJECT_BINARY_EFFECT',actual_F2_unchanged=True,owner_decision_required=True,
        actual_original_F2_norm_lower=bound(lower),actual_original_F2_norm_squared_lower=bound(lower**2),
        actual_E2_operator_norm_upper=bound(upper**2),E0_negative_direction_expectation_upper=bound(1-lower**2),
        exact_declared_CAR_unit_mean_witness_norm_squared=bound(declared_norm),
        conditioned_unit_mean_witness_norm_squared_lower=bound(norm),mean_raw_SOURCE_HS_squared_upper=bound(upper_raw),
        full_actual_birth_difference_norm_upper=bound(birth),full_actual_centre_difference_norm_upper=bound(centre),
        mean_outside_cutoff_one_core_squared_upper=bound(tail),all_original_nonhard_norm_upper=bound(extra),
        nonhard_bound='Arb128 direct raw coefficients, all gamma/sine/density/harmonic terms; |sin|,|cos|<=1, exact Gaussian linear norm, no presumed observable norm1',
        seed_CAR_conditioning_balls=seed_errors,hard_packet_Gram_error=bound(packet_error),normal_Gram_error=bound(normal_error),
        full_original_orbital_Gram_error=bound(orbital_error),
        raw_Slater_wedge_squared_norm_lower_factor=bound(wedge_lower),
        raw_Slater_wedge_norm_upper_factor=bound(wedge_norm_upper),
        conditioning_semantics='Runtime FullSlaterFibre.overlap=I (156-157) and source_vectors_from_marks272-276 declare exact labelled-CAR coefficients. Lower additionally discounts each seed projection by1+Gram-error and pays full original orbital wedge/normalization comparison. No seed/state/function changed.',
        mean_witness='unit normal direction5 times unit hard onephoton label2; all harmonic phases exact Gaussian; full original seeded CAR',
        numerical_scope='Arb128 mean/raw/seed/Hermite/Weyl/constants/ALLextras; no mean frame LAPACK evaluation or sparse quadrature in decision',
        analytic_SOURCE_theorem=constants,probability_or_Born_claim=False,
        generic_auxiliary_observable_transfer='Use M=actual ||O|| upper. For E2 use displayed actual_E2_operator_norm_upper, never M=1.',
        LEGAL_effect_path='Separate original LEGAL TWO raw OUT functions jointly inverse-root-normalized in TRUE physical Gram; E2 rankone unitF2 and E0=I-E2. Native cached14Phi Cholesky factory is not this definition. LEGAL physical Gram/rank and endpoint errors must be paid in physical_fullfield_scattering.paired_physical_binary_enclosure; this native result is NOT a LEGAL disproof.')
