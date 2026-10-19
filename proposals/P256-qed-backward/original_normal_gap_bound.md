# Original whole-heavy finite-graph normal-gap bound

This is a **quadratic normal theorem**, not an exact interacting quantum ground calculation or completed low-energy operator embedding. It retains the original whole-half itinerant heavy sector, full60 fields and45 colour-link coordinates per cell, all15 SU4 Gauss tangents, all64 centre torons and every finite-grid Fourier momentum. No added27-flavour selector, angular cutoff or presumed identity toron enters. Numerical receipts estimate the theorem's exact constants; they do not outward-certify a Lambda onset or measure a quantum spectral gap.

## 1. Source, stationary frames and declared kinetic law

At fixed0<a<=1 the source is the unimproved `RotationBand(**BARE,a=a)`, m=.2,r=.12,v=.3, kappa=2/a. Its original endpoint average and whole2V occupation are owned by `original_charged_principal.py`/`bulk_quantum_selection.py`; the latter's initialized27-selector metadata is unused here. The full10 strictly convex positive Gram problem at each centre theta is solved, not the cubic two-parameter problem transported to a noncubic toron. Write

```
F_theta(G)=kappa ||G||_F^2/2 -2 average_k sqrt(x(k+theta/(Na))^T G x(k+theta/(Na))),
Phi_mu=sum_nu sqrt(G)_mu,nu Gamma_nu.
```

The analytic normal theorem assumes the exact positive Gram minimum, not the binary floating approximation. At that minimum the field gradient is zero. The actual endpoint average gives delta h=(1/2){T,partial_k h} for a uniform link variation. The metric-Clifford identity {h,partial_k h}=partial_k(x^T G x)I and TrT=0 make its sea derivative zero; the trace part vanishes as well. All link gradients therefore vanish. No Lambda^(-1/4) shift is inferred from an unsymmetrized vertex.

The **changed slow kinetic trajectory** is explicit:

```
Hkin_field = Lambda^(-1/2) p_Phi^2/2,
u4=.3 Lambda^(-1/2),
Hheavy has coefficient Lambda,
J=.2 Lambda^(5/4), K4=.7 Lambda^(5/4).
```

It is not the earlier fast field sqrt(Lambda) law or a fixed-u4 result. Factor out chi=Lambda^(-1/2): the constant bare cometric C0 is diag(I60,.6 I45). The physical coordinate complement is mass-horizontal, obtained from Z spanning ker((sqrt(C0^-1)G_gauge)^dagger), with all90 normal dimensions. The constant reduced matrices are

```
A=Z^dagger sqrt(C0) H0 sqrt(C0) Z,
B=Z^dagger sqrt(C0) H1 sqrt(C0) Z,
omega^2 = chi (Lambda A+Lambda^(5/4) B).
```

Stationarity/gauge-nullness is necessary before treating complements as physical Hessians. The executable checks the native source gradient at aH scale. It does not use a stiff-dominated relative H_Lambda residual to hide nonstationarity. The separate principal producer's off-stationary Euclidean-slice diagnostics are not sign certificates for this theorem.

## 2. Uniform upper bound on every source H0(q)

All norms are finite-dimensional operator norms unless labelled HS. Let g_min/g_max be Gram eigenvalues, Cx2=(|m|+6|r|/a)^2+3(v/a)^2 and

```
Delta=2|m| sqrt(g_min),
SF=sum_j ||F_j||_HS^2=(3r^2 G00+v^2 sum_j Gjj)/a^2,
SV=60 Cx2+30 SF.
```

At a positive metric-Clifford frame the exact full Bloch matrix has two positive and two negative levels +/-sqrt(x^T Gx). Thus the actual whole2V occupied cluster is separated from the rest by at least Delta; the rank2 per-block projector is a consequence at this frame, not imposed on general competing configurations.

For all q, a field vertex has coefficient (x_mu(k)+x_mu(k+q))/2 multiplying a basis matrix of HS norm2. Summing all60 vertex norms squared is at most60 Cx2. A link vertex is the endpoint anticommutator combination; ||{F,T}||_HS<=2||T||op||F||_HS<=sqrt(2)||F||_HS. Summing15 generators for three axes gives at most30 SF. The whole-half sea Hessian is a negative Gram sum with occupied/empty denominators at least Delta. Its trace-norm, hence operator norm, is bounded by SV/Delta. Both particle-hole orientations are already contained in the matrix HS sum; they are not counted a second time.

For the contact Hessian, averaged phase-weighted rank2 projectors have HS norm at mostsqrt(2). Each mixed field/link entry is bounded by4|scalar_mu|. Its two adjoint blocks contribute at most5400(r^2+v^2)/a^2 to the total squared Frobenius norm. Since ||{T_a,T_b}||op<=1, each link/link entry is bounded bysqrt(2)||F_j||_HS; all225 entries on each axis contribute450 SF. Consequently

```
||H_contact(q)|| <= sqrt(5400(r^2+v^2)/a^2+450 SF).
```

Writing the local field coordinates as X in R^(15x4), the quartic is kappa Tr(X^T X)^2/2 and its Hessian maps V to2kappa(VG+XV^T X+XX^T V). Therefore its norm is at most6kappa g_max. For every q,

```
||H0(q)|| <= L
 =6kappa g_max+SV/Delta+sqrt(5400(r^2+v^2)/a^2+450 SF).
```

Compression by Z and sqrt(C0), whose norms are at most1, preserves this upper bound. This L is deliberately conservative and source-derived; numerical sampling of H0 at a few q is not substituted for an all-q upper bound.

## 3. Exact normal counts and positive constants

At q0, covariant alignment restores all45 links. Removing the15 gauge field directions leaves precisely45 uniform field normals in kerB and45 stiff normals. At nonzero q, vanishing alignment implies deltaPhi=D omega and link increments (1-exp(iq_j))omega; these are exactly the15 gauge tangents. Thus B is positive on the full90-dimensional physical complement. The plaquette term is retained as an additional positive form, not silently discarded.

The uniform source normal Hessian is positive: the Gram functional is strictly convex in all10 metric directions. The remaining35 physical directions have strictly positive trace/half-sea slack curvature whenever the shifted coefficient quadratic design has rank10. At a metric-Clifford point the slack zero condition is that {h,delta h} be scalar for every sampled momentum. Full quadratic richness makes this the differentiated Clifford relation. After separating the10 metric changes, all derivations of Cl4=Mat4 are inner; precisely15 SU4 directions remain zero, not extra physical modes. N>=5 avoids degree2 Fourier aliasing for every centre shift. The producer checks the actual design rank for every class. The positive45-normal constant is alpha=min eig(A|kerB) at q0. Let b be the smallest positive eigenvalue of B over q0 and all nonzero momenta; b>0 on each fixed finite graph. Let Bmax=max_q ||B(q)||.

This positivity is about each centre-class stationary normal form. It is **not** a proof that a finite N centre class is globally minimal against all reducible/noncentral flat or nonuniform fields. The separate large-N flat-holonomy/coercivity argument is needed before a global scalar-well reduction.

## 4. Schur bound, gap powers and zero-point budget

For q0 decompose v=s+t with s in kerB and t perpendicular. Since ||A||<=L,

```
<v,(Lambda A+Lambda^(5/4)B)v>
 >= (Lambda alpha/2)||s||^2
    +[Lambda^(5/4)b-Lambda(L+2L^2/alpha)]||t||^2.
```

This uses2L||s||||t||<=(alpha/2)||s||^2+(2L^2/alpha)||t||^2; no assumption that A is positive on every link normal is made. If

```
Lambda^(1/4) >= 2(L+2L^2/alpha)/b,
```

then the t coefficient is at least Lambda^(5/4)b/2. At nonzero q, B>=b and the weaker condition Lambda^(1/4)>=2L/b already suffices. The single displayed threshold therefore controls **every** momentum and gives uniform-mode frequencies at leastsqrt(alpha/2) Lambda^(1/4), stiff frequencies at leastsqrt(b/2) Lambda^(3/8), and a full finite-graph quadratic normal gap at least their minimum.

All90V real physical normals are counted by the Fourier decomposition, including conjugate momenta without doubling. The oscillator zero-point upper bound is

```
E_zp <=45V sqrt(L Lambda^(1/2)+Bmax Lambda^(3/4))=O(Lambda^(3/8)).
```

These exact powers supply the scalar-well separation used in the independent BO candidate: a flat-adapted epsilon=Lambda^(-1/5) comparison costs O(epsilon E_zp)=O(Lambda^(7/40)), below the smallest normal gap Lambda^(1/4). The normal-bound theorem alone neither establishes that scalar localized spectral gap nor excludes an additional low-energy block in the full operator. In particular an image residual ||KW-WHeff|| going to zero is insufficient without a complementary lower bound. No norm-resolvent or full low-energy claim is promoted here.

## 5. Native current/contact and receipt meaning

The consumer additionally builds the actual4V-by4V original light matrix with native finite full Gram and centre seam links, arbitrary compact EM phases and its exact first/second endpoint phase derivatives. It checks J=i[Q,h], contact=-[Q,[Q,h]], and finite compact-U1 covariance. Omitting contact is a nonzero adverse case. This source check does not replace the finite full metric by isotropic `RotationBand` or imply photon quantization, a scattering amplitude or radiative agreement.

The default CLI executes all64 centres, allN^3 momenta at N5, source stationarity, whole-heavy gap/richness, complete physical normal counts and the analytic L formula. The native endpoint controls use identity, quarter and all-minus centre classes. Receipts retain energy/Gram differences, conservative Lambda estimates and isotropic debts. Binary alpha/b estimates and source stationarity residuals are explicitly diagnostics, not interval certification of exact minima, measured quantum eigenvalues or a practical universal onset.

No limit is uniform in a or N. No thermodynamic compact-U1/Z4 phase, positive charged mass bound, physical half-spin, radiative matching or continuum/UV completion is asserted. The campaign remains open.
