# A global all120-field pair bound, without a Clifford restriction

Consider the actual `FullBandGaussSection` two-cell owner, one active x path,
whole-heavy occupation four of eight, arbitrary traceless Hermitian fields
Phi_mu,L/R and arbitrary U in SU(4). Gauge-fix U to identity by transforming the
right fields too. Write its actual temporal onsite c and path coefficients
`p0=-d`, `p1=-i t`, with real c,d,t. Require **c>|d|**, kappa>0, J>=0.
The pair has c=m+r/a, not the three-axis open-cube m+3r/a.

The actual one-body block is

```
h = [c Phi0,L,  -d(Phi0,L+Phi0,R)/2 - i t(Phi1,L+Phi1,R)/2;
     adjoint,    c Phi0,R].
```

No assumptions about Clifford identities, homogeneity, spectral sign pairing,
per-block occupation or a Gram budget enter the lower bound. The full local
zero-preference potential is `kappa/2 sum_x ||G_x||_F²`, where
`G_x,mu,nu=Tr(Phi_mu,x Phi_nu,x)/4`; alignment is full J/8 W>=0.

## Ky Fan block-involution inequality

For any rank4 occupied projector P on C8, let Q=2P-I. Q is Hermitian, Q²=I,
TrQ=0. Since Trh=0, TrPh=TrQh/2. Write

```
Q=[A,B;B†,C], B=X+iY,
X=(B+B†)/2, Y=(B-B†)/(2i),
```

with A,C,X,Y Hermitian. Q²=I gives

- `||A||_HS²+||C||_HS²=8-2(||X||_HS²+||Y||_HS²)`;
- `AB+BC=0`, hence `Tr[(A+C)X]=0`;
- `||Y||_HS²<=||B||_HS²<=4`.

The two temporal field duals are `(cA-dX)/2`, `(cC-dX)/2`; the two spatial
duals are `-tY/2`. Therefore their squared HS norms sum to

```
T0=2c² - (c²-d²)||X||²/2 - c²||Y||²/2,
T1=t²||Y||²/2.
```

Put `u=(G_L,00+G_R,00)/2`, `v=(G_L,11+G_R,11)/2` and
`q=||Y||²/4` in [0,1]. Discarding the nonnegative X penalty is allowed precisely
by c²>d². Full HS Cauchy-Schwarz, with `||Phi_mu,x||_HS²=4G_x,mu,mu`, yields

```
TrPh <= 4c sqrt(u(1-q)) + 4|t| sqrt(v q)
      <= 4 sqrt(c²u+t²v).
```

Maximizing over P proves this inequality for the **actual top four eigenvalue
sum**, even when the signs are not four positive/four negative. Finite sample
checks are controls of this derived inequality, not its proof.

## Exact minimum and explicit nonnegative excess

The quartic satisfies

```
V >= kappa/2 sum_x (G_x,00²+G_x,11²) >= kappa(u²+v²).
```

Set `R*=[(c⁴+t⁴)/kappa]^(1/3)`, `u*=c²/(kappa R*)`,
`v*=t²/(kappa R*)`. The reduced scalar function is strongly convex in u,v:

```
kappa(u²+v²)-4 sqrt(c²u+t²v) >= -3R*.
```

Uniform fields `Phi0=b Gamma0`, `Phi1=s Gamma1`, Phi2=Phi3=0, with
b²=u*, s²=v*, Hermitian traceless anticommuting unit Gamma0,Gamma1, attain the
bound. Their eight eigenvalues are the four combinations `±R*±db`, each twice;
c>|d| implies R*>|d|b, so the actual largest four sum to4R*. Their alignment is
zero. This proves a **global classical minimum over all120 field coordinates
and the whole SU(4) link**, not merely within the exhibited Clifford family.
The existing reference supplier's narrower proof metadata remains attributable
to that supplier; this separate theorem supplies the additional global result.

More usefully, the actual classical excess `E+3R*` is exactly the sum of:

1. Full Gram remainder `kappa/2 sum_x ||G_x||² - kappa/2 sum_x(G_x,00²+G_x,11²)`;
2. Endpoint norm spread `kappa/2 sum_x[(G_x,00-u)²+(G_x,11-v)²]`;
3. Radial norm deviation `kappa[(u-u*)²+(v-v*)²]`;
4. Radial square `2[sqrt(c²u+t²v)-R*]²/R*`;
5. Ky Fan slack `4sqrt(c²u+t²v)-sum(top4 h)`;
6. The full-field nonnegative alignment J/8 W.

Every term is nonnegative. Term1 separately controls
`kappa/2 sum_x(G_x,22²+G_x,33²)` as well as all off-diagonal Gram squares.
Thus unused field directions were not projected away: they have global quartic
confinement, although they can lack quadratic restoring curvature at the minimum.
The exact radial-square identity follows by expanding the quartic about u*,v*
and using `c²u*+t²v*=R*²`.

The pointwise lower bound also bounds the full whole-heavy Pauli operator at
fixed NH4; adding nonnegative field/link kinetic terms preserves it on the
all120-field Hilbert space. It is **not** the exact quantum ground energy,
proof of concentration of that ground, or a difference of charged and neutral
quantum spectral infima. Light/photon terms must be treated separately before
any such comparison. No bulk or thermodynamic minimum, physical spin, compact
phase or full electron-photon outcome follows from this finite pair theorem.

## Actual discriminating controls

The CLI recomputes all eight modes, the maximizing Q identities, all remainder
terms and source-owner anchor energy at96 nonuniform all120-field/genuine
noncentral-link inputs. It checks equality at the exhibited minimum and actual
4900-state primitive heavy/light energy and Gauss-section derivatives there.

The necessary dropped-penalty condition is exercised at an actual wrong-domain
Wilson owner: m=-1.9,r=2,a1 gives c=.1,d=1. For uniform temporal field .3Gamma0,
the actual top4 sum is1.2, while the invalid claimed c-based bound is.12. The
coefficient acceptance boundary must reject this owner rather than silently
discard its negative X penalty. No negative-mass physical model is proposed.

```
python proposals/P256-qed-backward/pair_global_field_bound.py \
  --output proposals/P256-qed-backward/pair_global_field_bound_receipt.json
```
