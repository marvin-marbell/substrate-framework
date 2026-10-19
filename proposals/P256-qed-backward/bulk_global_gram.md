# Existing global Gram theorem, now at the actual periodic bulk supplier

This producer reuses `soft_budget.select_gram` and its existing convex theorem;
it does not establish a second Gram minimizer. Its added result is the explicit
source/input-bound transfer to the current **actual unimproved** periodic
whole-heavy law a1,m.2,r.12,v.3,kappa2,g0=0 at midpoint8/16/24/32. It compares all
three existing finite mean branches and arbitrary all60-coordinate competitors.

For any four traceless Hermitian matrices Phi_mu, let
`G_mu,nu=Tr(Phi_mu Phi_nu)/4`. G is positive semidefinite. In the flat identity-link
uniform sector the executing owner is `h(k)=sum x_mu(k) Phi_mu`. Hence
`Tr h(k)=0`, `Tr h(k)^2=4 x(k)^T G x(k)`.

## Fixed total half filling, not a block constraint

For **any** occupied subset in one block, its eigenvalue sum is at most the
sum of all positive eigenvalues. Trh=0 makes the latter `Tr|h|/2`. HS Cauchy
bounds it by `sqrt(Trh²)=2sqrt(x^TGx)`. Summing this **inequality** over blocks
bounds the actual largest2N eigenvalue sum of all4N modes. It is not an
absolute-value replacement of the physical energy or an assumption that the
occupied rank equals2 at every momentum.

Thus the actual whole-heavy/quartic uniform functional satisfies

```
E(Phi)>=F(G)=kappa/2 ||G||_F² -2 average_k sqrt(x(k)^T G x(k)).
```

For every PSD4-by-4 G, take an actual square root A with AA^T=G and
`Phi_mu=sum_nu A_mu,nu Gamma_nu` using the executing primitive four Clifford
matrix units. Then h(k)^2=(x^TGx)I. Trh=0 gives two positive/two negative levels
when the radius is nonzero; zero modes can be occupied at half filling too.
These fields achieve the bound with exactly the required total occupation.
Therefore minimization over **all60 uniform Cartesian field coordinates** is
exactly minimization of F over the full PSD cone. The lower bound never
restricted the matrices to Clifford form.

F is kappa-strongly convex in the HS metric on G. Cubic sign/permutation
symmetry of the actual midpoint distribution and uniqueness force
`G*=diag(g0,gS,gS,gS)`; it is not a hard Gram budget. For the present positive
mass/velocity data both scales are positive. The existing source solver
minimizes these two variables and refines actual stationarity. The full10
symmetric-matrix gradient is independently recomputed by this producer,
including off-diagonal entries. Strong convexity gives

```
F(G)>=F(G*)+kappa/2 ||G-G*||_F²,
inf F>=F(Ghat)-||gradient F(Ghat)||_F²/(2kappa).
```

The program evaluates the latter residual formula in floating arithmetic,
not as an interval-certified enclosure. Each quadrature has its own optimum.
No bound on the continuum optimum is inferred from mesh agreement alone.

At energy equality HS Cauchy requires each h(k)^2 to be scalar. The actual
quadratic coefficient design has rank10. Expanding h(k)^2 then forces all
individual squared/anticommutator coefficient identities
`{Phi_mu,Phi_nu}=2G*_mu,nu I`: the Clifford condition is selected, not installed
as a constraint. This rank requirement is recomputed at each executed mesh.
The same convex argument applies to the cubic continuum integral; the receipt
does not supply an interval evaluation of that integral or its unique metric.

## Actual adverse consumer

An explicit unrestricted temporal field `diag(5,1,1,-7)`, other components0,
is traceless but has three positive one-body eigenvalues at every block. The
largest temporal band is occupied at every momentum. The next two transfer
occupation toward larger x0(k), so globally filled blocks have ranks1 or3
(and a rank2 tie if necessary), not uniformly2. The program recomputes actual
global half filling and a genuinely wrong forced-two-per-block energy, and
requires a nonzero difference. This distinguishes the physical occupation
from an easier implementation at its actual assertion.

Uniform-sector globality says nothing about arbitrary spatially varying
fields or colour links. In particular, the open8-cell cube has graph-valued
noncommuting coefficient operators and its different U1 competitor does not
inherit this periodic Bloch proof. No quantum ground concentration, charged
lower gap, compact phase, physical spin, Lorentz closure or completed campaign
is claimed here. Historical finite-minimum receipts remain attributed to their
original narrower proofs; the new producer supplies this separate global
uniform transfer without rewriting their source/input bindings.

```
python proposals/P256-qed-backward/bulk_global_gram.py \
  --output proposals/P256-qed-backward/bulk_global_gram_receipt.json
```
