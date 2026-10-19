# Prepared neutral: exact all-mode Weyl and hard Fock core

This independent core extends the **basis**, not the once-prepared state. It owns no Phi/Haar Hamiltonian, SOURCE functions, Noether generator, preparation, filter, clock, or calibration. `prepared_neutral_ambient_weak_state.py` and its frozen A17 receipt/NPZ are unchanged. The native input is N=3, UE=2^-20, c=.125. These are diagnostic inputs, not admitted legal-N parameters or hard-mode IDs.

## Real chart, complex packets and normalized functions

There are d=2(V-1)=52 real coexact oscillator coordinates y. The inherited A17 chart is A_coexact=legs y, with `legs=vectors sqrt(2 UE/omega)` and omega=c sqrt(lambda). The normalized Gaussian is psi0(y)=pi^(-d/4) exp(-y.y/2), so its probability covariance is I/2. The three harmonic theta coordinates are excluded; their Hamiltonian is the separate free UE/N ptheta^2, not three more oscillators.

`native_packets(law,em)` constructs literal ez exp(±2pi i x_j/N)/sqrt(V) profiles for +x,-x,+y,-y and returns their full52 coefficients g=vectors.T profile. It verifies physical coexact reconstruction and g†g=I. No whitening, projection repair, renormalization or packet replacement is performed. At N3 all four omega=.125 sqrt(3). A diagonal free-energy basis additionally requires monochromatic packets, checked by diag(omega)g=g omega_packet.

Define a†(g_i)=sum_m g_mi a†_m and z_i=g_i.T y. Complex one- and two-photon wave polynomials are

- p0=1;
- p_i=sqrt(2) z_i;
- p_2i=[2 z_i^2-g_i.T g_i]/sqrt(2);
- p_ij=2 z_i z_j-g_i.T g_j for i!=j.

The subtraction is **bilinear**, not Hermitian. For these native planes g_+x.T g_+x=0 while g_+x.T g_-x=1, likewise for y. Treating every subtraction as 1 would even give a two-+x function a vacuum overlap of -1/sqrt(2). The implementation uses the exact creator recurrence [sqrt(2)(g.y)p-(g.grad p)/sqrt(2)]/sqrt(n_i+1), supporting higher finite cutoffs without a dense 52^r tensor.

The public `labels` are four occupation tuples, sorted by (sum, tuple). Total<=2 has 15 states; the vacuum is index0. Use `basis.index[occupation_tuple]`, not guessed one-photon column indices. The all52 Gaussian remains present even for functions depending only on four complex projections. `free_energy` is sum_i omega_i n_i plus **one** all52 zero point, sum_m omega_m/2=7.973272719851423. Do not add this zero point again, or retain a separately sampled coexact vacuum contribution in addition to it.

## Exact Weyl compression

For a real all52 displacement l, alpha_i=g_i†(i l/sqrt(2)). Normal ordering in the complete oscillator space gives

W_mn(l)=exp(-l.l/4) product_i sqrt(m_i! n_i!) sum_{k=0}^{min(m_i,n_i)} alpha_i^(m_i-k) (-alpha_i*)^(n_i-k) / [k! (m_i-k)! (n_i-k)!].

This is P exp(i l.y) P, not exp(i P(l.y)P). The common Debye factor uses **all52** l, including directions orthogonal to the four hard packets. The hard polynomial factor is finite; the Gaussian exponential is not truncated. Neither a hard-only Debye factor nor an exponential of compressed coordinate operators implements this matrix.

## Arbitrary finite Gaussian/Weyl-polynomial functions

`ProjectionPolynomial(projections,terms)` has projection rows v_i of shape (k,d) and a dictionary `{sorted_factor_index_tuple: complex_coefficient}`. The empty tuple denotes a constant and repeated indices denote powers. Thus terms `{(0,0):2,():-1}` with one row v represent 2(v.y)^2-1. `constant(d,value)`, `multiply(other)`, `conjugate()`, `directional_derivative(direction_d)`, and `evaluate(y_d)->(value,gradient_d)` are public. Multiplication joins small projection banks; it does not allocate a full mode tensor.

`fourier_gaussian_moment(projections,factors,displacement_d)` computes E[exp(i l.y) product_i(v_i.y)] using the common exp(-l.l/4), shifted mean i(v_i.l)/2, and bilinear covariance (v_i.v_j)/2. Memoized Wick recursion supports **every finite degree**, not only degree4. `polynomial_fourier_integral(polynomial,l)` sums these moments. No Gaussian quadrature is used by these APIs.

`GaussianFunction(polynomial,phase_d)` means psi0 p exp(i phase.y), with real all52 phase. Its `evaluate(y)` returns the physical wave and full first derivative, including -y p and i phase p. Its `directional_derivative(direction_d)` returns another exact GaussianFunction, so derivative overlaps use the same kernel.

`gaussian_kernel(left,right,displacement=None)` returns <left|exp(i displacement.y)|right>; the effective displacement is right.phase-left.phase+displacement. Arbitrary outside-coexact finite polynomial/Weyl partners can therefore be integrated by B without inventing a one-dimensional spin substitute. Finite linear combinations are integrated by summing these bilinear kernels. This core deliberately does not design or assert closure of any G.

## Exact creator boundary

`HardFockBasis(em,hard_packets,max_total=2)` exposes `packets`, `omega`, `packet_omega`, `bilinear`, `labels`, `index`, `dimension`, `creation[4,D,D]`, `annihilation[4,D,D]`, `zero_point`, `free_energy[D]`, and `coordinate_polynomials`.

- `polynomials(y_d)` returns polynomial values[D] and derivatives[d,D], divided by the common Gaussian.
- `evaluate(y_d)` returns `polynomial`, `polynomial_derivative`, physical `value`, and physical `first_derivative`.
- `function(index,phase=None)` returns the corresponding GaussianFunction.
- `weyl_matrix(displacement_all52)` returns exact D-by-D Weyl compression.
- `arbitrary_creator(packet_d)` does not require packet normalization. It returns `projected`, `hard_coefficients`, `outside_hard_packet`, `outside_hard_norm_squared`, `boundary_labels`, `boundary_map`, `outside_hard_map`, `outside_gram`, and `full_norm_gram`.

Writing h=g c+r with g†r=0, the projected creator is sum_i c_i P a†_i P. `boundary_map` maps highest retained occupations into all degree cutoff+1 hard occupations with coefficient c_i sqrt(n_i+1). `outside_hard_map` has original hard-label rows and coefficient ||r||; those rows represent one photon in residual packet r/||r|| (zero map if r=0). Its environmental direction is supplied separately by `outside_hard_packet`. The two outside ranges are orthogonal. Hence outside_gram=boundary_map†boundary_map+||r||^2 I. It retains coherent interference between hard creator terms and obeys projected†projected+outside_gram=||h||^2 I+projected projected†. For h=g_+x+.4 r_unit and occupation (2,0,0,0), outside norm squared is 3+.16=3.16. A creator norm1 shortcut or omega_min/mode-count surrogate loses this real boundary.

## Physical packet templates

`packet_templates(law,basis,time=6)` is explicitly native N3. It returns names incoming/forward/outgoing, positions (1,1,1)/(2,1,1)/(1,2,1), native wavevectors ±2pi/3 along x/y, `creator_hard_coefficients[3,4]`, their conjugate `annihilator_hard_coefficients`, `creator_all52[3,52]`, its conjugate `annihilator_all52`, outgoing free phase, time, scope, and `SOURCE_eta_common=True`.

Incoming has +x/-x weights (4,1)/sqrt(17) times exp(-ik.X). Forward uses the same weights and exp(-ik.(X+ex)), hence its relative geometry phase is exp(-ik_x). Outgoing uses +y/-y weights and exp(-ik.(X+ey)) exp(-i omega 6). These are creator coefficients; annihilator coefficients are conjugate. There is no legacy origin-parity phase. Source19 eta is the common 020fb-descriptor profile and is **not spatially shifted** by this API.

## Executed evidence

Natural CLI:

```
/home/administrator/substrate-framework/.venv/bin/python prepared_neutral_weyl_hard_fock.py --state-npz <local>/issue222-ambient-weak-state.npz --output-npz <local>/issue222-weyl-hard-fock.npz --source-archive <local>/issue222-weyl-hard-fock-frozen-sources.zip
```

Cwd is this proposal directory. Exact argv, native receipt, natural exit/walltime, stdout/stderr hashes, all loaded source hashes and pre-science frozen source archive are in `local://issue222-weyl-hard-fock-execution.json`. Data are `local://issue222-weyl-hard-fock.npz`. Current real Pyright evidence is `local://issue222-weyl-hard-fock-final-real-lsp.json`; it is source-bound rather than inferred from syntax. The positive AST query matched HardFockBasis. Pretyping attempts and their diagnostics are retained in separate local evidence archives, not silently relabeled final science.

The actual 52-mode N3 bank, four physical planes and frequency spectrum are constructed, not mocked. All 15-by-15 overlaps/Weyl matrices and a physical derivative-pair matrix are checked against independent order12 tensor Gauss-Hermite on four real effective projections (20736 nodes). Weyl error is 1.52e-14, derivative-pair error 1.55e-14; the independent general Hermite moment API agrees with normal ordering at 2.59e-15. A degree8 outside-mode Weyl polynomial is checked by independent order20 one-dimensional quadrature (error5.03e-16), and the full wave first derivative is checked against finite differences. Wrong exp(compressed A) has matrix defect .477745; missing outside-hard Debye has defect .313298; wrong complex Wick has vacuum overlap .707107. The degree2 creator boundary has actual missing norm squared3.16. These are decision-bearing rejected branches, not merely nonzero output assertions.

The exact frozen NPZ `neutral_state` is embedded coefficient-for-coefficient in the EM vacuum: `embedded[:,0]=neutral_state`, with every other photon column zero. Its original neutral metric norm and embedded norm are both1. The old raw preparation norm .8920522472031723 and normalized-neutral SOURCE norm squared .8718600287632787 remain immutable historical A17 facts; no second preparation/filter/normalization is run. NPZ input SHA256 is eb283954e7ebb13e1b5368deb058214443d9a839b8319161c0306afb2c8b53b6.

## Scope and unpaid dynamics

Exact Gaussian integration is exact for specified finite polynomial/Weyl functions on **all52 untruncated oscillators**. The retained hard Fock chart is finite Galerkin, not full-Fock dynamics. Its creator leakage is computed, not assumed absent. Actual additional physical Noether partners and their dynamics remain B/A integration work. This proof does not promote native IDs to legal-N photon modes, establish legal Xi, simulate all512 histories/full astronomical law, pay SOURCE/higher-graph normalization, or prove a physical closed rotation/Born result.

Analytically integrating EM may change the old vacuum Hamiltonian block because A17 used fourteen stochastic all-mode EM samples. The owner must explicitly report that change while preserving the exact old prepared coefficients; this core cannot infer equality of those compressions and does not refilter. Full Phi/Haar/metric, harmonic, actual full448 SOURCE quartet functions, Root-Q/fullaux and time-evolution/outside debts remain separate. No old supplier scientific CLI was rerun, and no sibling sources were edited.
