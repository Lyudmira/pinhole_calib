
## Variational and Perturbative Geometry of Three-Dimensional Manifolds Under Mis-Specified Intrinsics

## Identifiability landscape

The mathematically decisive fact behind your three questions is that uncalibrated image correspondences determine a reconstruction only up to a 3D projective transformation; passing from that projective class to a metric or Euclidean reconstruction requires additional calibration information, classically through the plane at infinity and the image of the absolute conic. In bundle adjustment language, this becomes a gauge problem: Euclidean reconstruction has a 7-DOF similarity gauge, whereas projective reconstruction has a 15-DOF projective gauge. In parallel, the Manhattan world assumption 1 imposes an orthogonality structure on normals or dominant directions, which is exactly the type of extra geometric information that can break part of the projective ambiguity. 2

For the intrinsic mismatch itself, it is useful to write
$$
H=\widehat{K}^{-1} K^{*}=\left[\begin{array}{ccc}
\alpha_{x} & 0 & \beta_{x} \\
0 & \alpha_{y} & \beta_{y} \\
0 & 0 & 1
\end{array}\right], \quad \alpha_{x}=\frac{f_{x}^{*}}{\hat{f}_{x}}, \quad \beta_{x}=\frac{c_{x}^{*}-\hat{c}_{x}}{\hat{f}_{x}},
$$
and likewise for $y$. Then, in camera coordinates,
$$
H[X, Y, Z]^{\top}=\left[\alpha_{x} X+\beta_{x} Z, \alpha_{y} Y+\beta_{y} Z, Z\right]^{\top} .
$$

So focal-length error is an anisotropic stretch in the image-plane directions, while principal-point error is a rank-one shear $s e_{3}^{\top}$ with $s=\left(\beta_{x}, \beta_{y}, 0\right)^{\top}$. In normalized image coordinates,
$$
\pi\left(H[X, Y, Z]^{\top}\right)=\left[\alpha_{x} X / Z+\beta_{x}, \alpha_{y} Y / Z+\beta_{y}\right]^{\top}
$$
so principal-point drift appears as a constant image-plane shift, while focal drift appears as an image-plane scaling. This is exactly why principal-point estimation is known to be delicate in vanishing-point-based calibration, and why calibration bias propagates directly into motion and structure estimates. 3

## Exact algebraic regime without priors

The literal statement in your question 1.1 is false if interpreted as a single-view problem with freely reassigned depths. Fix one true camera-frame point $p_{i}^{*}=T^{*} X_{i}^{*}$, and define the wrong-intrinsic ray
$$
\tilde{r}_{i}:=\frac{H p_{i}^{*}}{e_{3}^{\top} H p_{i}^{*}} \in \mathbb{R}^{3} .
$$

For any chosen $\widehat{T} \in S E(3)$ and any positive scalars $\rho_{i}$, let
$$
\widehat{X}_{i}:=\widehat{T}^{-1}\left(\rho_{i} \tilde{r}_{i}\right) .
$$

Then
$$
\pi\left(\widehat{T} \widehat{X}_{i}\right)=\pi\left(\rho_{i} \tilde{r}_{i}\right)=\pi\left(H p_{i}^{*}\right),
$$
so the reprojection error is exactly zero. There is not only non-uniqueness; there is in fact an infinite family of exact solutions. Thus "wrong principal point implies strictly positive residual" is not true in the single-view depth-reassignment sense. This is the same projective non-uniqueness that underlies the standard ambiguity of uncalibrated reconstruction.

What becomes true is the stronger, genuinely multiview statement: if one asks for a single common 3D structure and all cameras constrained to have the fixed wrong intrinsic matrix $\widehat{K}$, then exact zero residual means that the same projective reconstruction admits two Euclidean upgrades, one with $K^{*}$ and one with $\widehat{K}$. For generic nonplanar scenes and generic noncritical motion, self-calibration is locally unique up to a similarity transform; therefore the second upgrade cannot exist unless $\widehat{K}$ lies in the same metricupgrade class as $K^{*}$, which generically forces $\widehat{K}=K^{*}$. This is precisely the content of the self-calibration literature and the theory of critical motion sequences: wrong intrinsics can remain exactly compatible only in degenerate cases such as planar scenes, pure rotations, or other critical motions. 5

For your question 1.2, the clean algebraic object is the least-squares projection of the affine map $H$ onto the rigid group. Let $p_{i}^{*}=T^{*} X_{i}^{*}$ be camera-frame points for one view, let $\bar{p}=\frac{1}{N} \sum_{i} p_{i}^{*}$, and write centered coordinates $q_{i}=p_{i}^{*}-\bar{p}$. Put $H=I+E, R=I+[\delta \omega]_{\times}+O\left(\|E\|^{2}\right), t=\delta t+O\left(\|E\|^{2}\right)$. Minimizing
$$
\sum_{i}\left\|H p_{i}^{*}-\left(R p_{i}^{*}+t\right)\right\|^{2}
$$
gives, to first order,
$$
\delta t=E \bar{p}-\delta \omega \times \bar{p},
$$
and
$$
(\operatorname{tr} \Sigma I-\Sigma) \delta \omega=\frac{1}{N} \sum_{i} q_{i} \times E q_{i}, \quad \Sigma:=\frac{1}{N} \sum_{i} q_{i} q_{i}^{\top} .
$$

Equivalently, if $C=H \Sigma$, then the exact LS rotation is the polar factor
$$
\widehat{R}=\operatorname{polar}(C),
$$
and the linearized skew matrix $\Omega=[\delta \omega]_{\times}$solves the Sylvester equation
$$
\Omega \Sigma+\Sigma \Omega=E \Sigma-\Sigma E^{\top} .
$$

This is the exact algebraic structure of the rigid projection. It is not mysterious: one keeps the rigid part of the affine perturbation and discards the symmetric nonrigid part.

For pure principal-point shear,
$$
E=\beta_{x} e_{1} e_{3}^{\top}+\beta_{y} e_{2} e_{3}^{\top}
$$

Then
$$
\frac{1}{N}\sum_{i} q_{i} \times E q_{i}=\left[\begin{array}{c}
-\beta_{y} \Sigma_{z z} \\
\beta_{x} \Sigma_{z z} \\
\beta_{y} \Sigma_{x z}-\beta_{x} \Sigma_{y z}
\end{array}\right]
$$

Hence principal-point error excites mainly pitch and yaw, and excites roll only if the cloud has $x z / y z$ covariance. In the common symmetric case $\Sigma=\operatorname{diag}\left(\sigma_{x}^{2}, \sigma_{y}^{2}, \sigma_{z}^{2}\right)$,
$$
\delta \omega_{x}=-\beta_{y} \frac{\sigma_{z}^{2}}{\sigma_{y}^{2}+\sigma_{z}^{2}}, \quad \delta \omega_{y}=\beta_{x} \frac{\sigma_{z}^{2}}{\sigma_{x}^{2}+\sigma_{z}^{2}}, \quad \delta \omega_{z}=0
$$
with
$$
\delta t=E \bar{p}-\delta \omega \times \bar{p}
$$

So the off-diagonal entries of $H$ are absorbed explicitly as yaw/pitch proportional to the depth variance, while any residual nonrigid part passes into translation, scale, or pointwise depth reshaping. In the limiting case $\sigma_{z}^{2} \gg \sigma_{x}^{2}, \sigma_{y}^{2}$, the rigid absorber approaches
$$
\delta \omega \approx\left(-\beta_{y}, \beta_{x}, 0\right),
$$
which is exactly the infinitesimal camera rotation that mimics a principal-point shift near the optical axis.

## Lie-algebraic first-order solution

Let
$$
p_{t i}=R_{t}^{*} X_{i}^{*}+t_{t}^{*}=\left(X_{t i}, Y_{t i}, Z_{t i}\right)^{\top}, \quad x_{t i}:=\frac{X_{t i}}{Z_{t i}}, \quad y_{t i}:=\frac{Y_{t i}}{Z_{t i}}
$$

Using the left-multiplicative perturbation
$$
\widehat{T}_{t}=\exp \left(\left(\Delta \xi_{t}\right)^{\wedge}\right) T_{t}^{*}, \quad \Delta \xi_{t}=\left[\begin{array}{c}
\Delta \rho_{t} \\
\Delta \phi_{t}
\end{array}\right] \in \mathbb{R}^{6}
$$
and $\widehat{X}_{i}=X_{i}^{*}+\Delta X_{i}$, the predicted pixel location is
$$
z_{t i}\left(\theta_{K}, \xi_{t}, X_{i}\right)=\left[\begin{array}{c}
f_{x} X_{t i} / Z_{t i}+c_{x} \\
f_{y} Y_{t i} / Z_{t i}+c_{y}
\end{array}\right]
$$

The measurement Jacobians at the true state are
$$
\begin{gathered}
G_{p, t i}:=\frac{\partial z_{t i}}{\partial p_{t i}}=\left[\begin{array}{ccc}
f_{x} / Z_{t i} & 0 & -f_{x} X_{t i} / Z_{t i}^{2} \\
0 & f_{y} / Z_{t i} & -f_{y} Y_{t i} / Z_{t i}^{2}
\end{array}\right] \\
G_{\xi, t i}:=\frac{\partial z_{t i}}{\partial \Delta \xi_{t}}=G_{p, t i}\left[I-\left[p_{t i}\right]_{\times}\right], \quad G_{X, t i}:=\frac{\partial z_{t i}}{\partial \Delta X_{i}}=G_{p, t i} R_{t}^{*}
\end{gathered}
$$
and
$$
G_{K, t i}:=\frac{\partial z_{t i}}{\partial\left(f_{x}, f_{y}, c_{x}, c_{y}\right)}=\left[\begin{array}{cccc}
x_{t i} & 0 & 1 & 0 \\
0 & y_{t i} & 0 & 1
\end{array}\right] .
$$

Expanding $G_{\xi, t i}$ explicitly,
$$
G_{\xi, t i}=\left[\begin{array}{cccccc}
\frac{f_{x}}{Z} & 0 & -\frac{f_{x} X}{Z^{2}} & -\frac{f_{x} X Y}{Z^{2}} & f_{x}\left(1+\frac{X^{2}}{Z^{2}}\right) & -\frac{f_{x} Y}{Z} \\
0 & \frac{f_{y}}{Z} & -\frac{f_{y} Y}{Z^{2}} & -f_{y}\left(1+\frac{Y^{2}}{Z^{2}}\right) & \frac{f_{y} X Y}{Z^{2}} & \frac{f_{y} X}{Z}
\end{array}\right]_{(X, Y, Z)=\left(X_{t i}, Y_{t i}, Z_{t i}\right)} .
$$

These are the analytical Jacobians you asked for in 2.1. They are the standard manifold linearization ingredients used in bundle adjustment and on-manifold least-squares optimization. 8

If $r$ denotes the stacked reprojection residual vector, then at first order
$$
r \approx-G_{\Xi} \Delta \Xi-G_{X} \Delta X-G_{K} \Delta \theta_{K},
$$
where $\Delta \Xi$ stacks all pose increments and $\Delta X$ stacks all point increments. On a gauge-fixed slice of the state space, the implicit-function-theorem derivative of the stationary-point map is
$$
\frac{\partial(\Delta \Xi, \Delta X)}{\partial \theta_{K}}=-\mathcal{H}^{-1} \mathcal{B}_{K}
$$
with
$$
\mathcal{H}=\left[\begin{array}{cc}
U & W \\
W^{\top} & V
\end{array}\right]=\left[\begin{array}{cc}
G_{\Xi}^{\top} G_{\Xi} & G_{\Xi}^{\top} G_{X} \\
G_{X}^{\top} G_{\Xi} & G_{X}^{\top} G_{X}
\end{array}\right]+\lambda \nabla^{2} \mathcal{E}_{\text{prior}}, \quad \mathcal{B}_{K}=\left[\begin{array}{l}
G_{\Xi}^{\top} G_{K} \\
G_{X}^{\top} G_{K}
\end{array}\right] .
$$

Without gauge fixing one replaces $\mathcal{H}^{-1}$ by the Moore-Penrose pseudoinverse, exactly because of the similarity-gauge null space. This is the clean Riemannian-IFT statement of the first-order solution map.

Eliminating points by Schur complement gives your 2.2 formula in closed block form:
$$
\Delta \Xi=-\underbrace{\left(U-W V^{-1} W^{\top}\right)^{-1}}_{S^{-1}} \underbrace{\left(G_{\Xi}^{\top} G_{K}-W V^{-1} G_{X}^{\top} G_{K}\right)}_{\Gamma_{K}} \Delta \theta_{K} .
$$

Thus
$$
\Delta \xi_{t}=\mathcal{F}_{t}\left(T_{t}^{*}, \mathcal{X}^{*}, \Delta K\right)=-\left[e_{t}^{\top} S^{-1} \Gamma_{K}\right] \Delta \theta_{K}
$$
where $e_{t}$ selects the $t$-th 6-vector block. This is exactly the reduced camera system familiar from sparse bundle adjustment: all pointwise depth flexibility is compressed into the Schur term $W V^{-1} W^{\top}$, and only the part of the intrinsic forcing that survives point elimination drives the pose bias. 10

For your question 2.3, principal-point drift gives the clearest closed form. Put
$$
\beta_{x}=\frac{c_{x}^{*}-\hat{c}_{x}}{\hat{f}_{x}} \approx-\frac{\Delta c_{x}}{f_{x}^{*}}, \quad \beta_{y}=\frac{c_{y}^{*}-\hat{c}_{y}}{\hat{f}_{y}} \approx-\frac{\Delta c_{y}}{f_{y}^{*}} .
$$

Near the optical axis, and keeping only the dominant pose channels, the normalized-image perturbation model is
$$
\delta x_{i} \approx-\frac{\Delta t_{x}}{Z_{i}}+\omega_{y}, \quad \delta y_{i} \approx-\frac{\Delta t_{y}}{Z_{i}}-\omega_{x}
$$

So the two principal-point subproblems decouple:
$$
\min _{\Delta t_{x}, \omega_{y}} \sum_{i}\left(\beta_{x}+\frac{\Delta t_{x}}{Z_{i}}-\omega_{y}\right)^{2}, \quad \min _{\Delta t_{y}, \omega_{x}} \sum_{i}\left(\beta_{y}+\frac{\Delta t_{y}}{Z_{i}}+\omega_{x}\right)^{2} .
$$

The $x$-block Hessian is
$$
H_{x}=\left[\begin{array}{cc}
\sum_{i} Z_{i}^{-2} & -\sum_{i} Z_{i}^{-1} \\
-\sum_{i} Z_{i}^{-1} & N
\end{array}\right],
$$
and analogously for $H_{y}$. The exact normal-equation solution is
$$
\begin{array}{cc}
\omega_{y}=\beta_{x}, & \Delta t_{x}=0 \\
\omega_{x}=-\beta_{y}, & \Delta t_{y}=0
\end{array}
$$
provided $\operatorname{Var}\left(1 / Z_{i}\right)>0$. Therefore
$$
\frac{\partial \omega_{y}}{\partial \Delta c_{x}} \approx-\frac{1}{f_{x}^{*}}, \quad \frac{\partial \omega_{x}}{\partial \Delta c_{y}} \approx \frac{1}{f_{y}^{*}}, \quad \frac{\partial \Delta t_{x}}{\partial \Delta c_{x}} \approx \frac{\partial \Delta t_{y}}{\partial \Delta c_{y}} \approx 0
$$
in the generic nondegenerate-depth regime. Principal-point error is therefore first-order rotational, not translational, after point elimination. ${ }^{11}$

The degenerate case is exactly your constant-depth sheet. If $Z_{i} \equiv Z_{0}$, then
$$
\operatorname{det} H_{x}=N \sum_{i} Z_{i}^{-2}-\left(\sum_{i} Z_{i}^{-1}\right)^{2}=0,
$$
and the family of exact first-order compensators becomes
$$
\omega_{y}-\frac{\Delta t_{x}}{Z_{0}}=\beta_{x}, \quad-\omega_{x}-\frac{\Delta t_{y}}{Z_{0}}=\beta_{y}
$$

So a fronto-parallel constant-depth surface cannot distinguish principal-point shift from a coupled yaw/ translation or pitch/translation mode. This is the exact algebraic origin of the sensitivity explosion. For $Z \sim$ Uniform $\left[Z_{\text{min}}, Z_{\text{max}}\right]$,
$$
\mathbb{E}\left[Z^{-1}\right]=\frac{\ln \left(Z_{\max } / Z_{\min }\right)}{Z_{\max }-Z_{\min }}, \quad \mathbb{E}\left[Z^{-2}\right]=\frac{1}{Z_{\min } Z_{\max }},
$$
hence
$$
\operatorname{det}\left(H_{x}\right) / N^{2}=\frac{1}{Z_{\min } Z_{\max }}-\left(\frac{\ln \left(Z_{\max } / Z_{\min }\right)}{Z_{\max }-Z_{\min }}\right)^{2}>0
$$
so the degeneracy is broken as soon as the inverse-depth variance is nonzero. The soft eigen-direction is the mixed translation/rotation mode; in the constant-depth limit it converges to
$$
v_{\text{soft}}^{(x)} \propto\left(Z_{0}, 1\right), \quad v_{\text{soft}}^{(y)} \propto\left(Z_{0},-1\right),
$$
which is exactly the coupled ( $\Delta t_{x}, \omega_{y}$ ) and ( $\Delta t_{y}, \omega_{x}$ ) ambiguity you asked for.

## Manhattan-constrained plane normals

For one true plane in camera coordinates,
$$
\Pi_{\text{true}}: \quad n^{* T} X=d,
$$
the wrong-intrinsic, no-prior reconstruction remains exactly planar. If $X=Z K^{*-1} q$ is the true point on the ray $q$, then the wrong-ray reconstruction is $\widehat{X}=H X$, so the reconstructed plane is
$$
\widehat{\Pi}_{0}: \quad \tilde{n}^{T} \widehat{X}=d, \quad \widetilde{n}=H^{-T} n^{*}
$$

Since
$$
H^{-T}=\left[\begin{array}{ccc}
1 / \alpha_{x} & 0 & 0 \\
0 & 1 / \alpha_{y} & 0 \\
-\beta_{x} / \alpha_{x} & -\beta_{y} / \alpha_{y} & 1
\end{array}\right]
$$
the exact unconstrained normal is
$$
\widetilde{n}=\left[\begin{array}{c}
n_{x}^{*} / \alpha_{x} \\
n_{y}^{*} / \alpha_{y} \\
n_{z}^{*}-\beta_{x} n_{x}^{*} / \alpha_{x}-\beta_{y} n_{y}^{*} / \alpha_{y}
\end{array}\right], \quad \widehat{n}_{0}=\frac{\widetilde{n}}{\|\widetilde{n}\|}
$$

This is the closed-form answer to 3.1 before Manhattan coupling is enforced. It already shows the qualitative geometry: a fronto-parallel plane ( $n^{*}=e_{3}$ ) is unaffected by principal-point drift, while side walls or floors tilt because their normals have $x$ - or $y$-components. Manhattan methods rely exactly on such orthogonal dominant directions and plane normals.

Writing $\hat{f}_{x}=f_{x}^{*}+\Delta f_{x}, \hat{c}_{x}=c_{x}^{*}+\Delta c_{x}$, and similarly in $y$, one has
$$
\alpha_{x}=1-\frac{\Delta f_{x}}{f_{x}^{*}}+O\left(\|\Delta K\|^{2}\right), \quad \beta_{x}=-\frac{\Delta c_{x}}{f_{x}^{*}}+O\left(\|\Delta K\|^{2}\right)
$$
and therefore
$$
\widetilde{n}=n^{*}+\left[\begin{array}{c}
\left(\Delta f_{x} / f_{x}^{*}\right) n_{x}^{*} \\
\left(\Delta f_{y} / f_{y}^{*}\right) n_{y}^{*} \\
\left(\Delta c_{x} / f_{x}^{*}\right) n_{x}^{*}+\left(\Delta c_{y} / f_{y}^{*}\right) n_{y}^{*}
\end{array}\right]+O\left(\|\Delta K\|^{2}\right) .
$$

After renormalization,
$$
\widehat{n}_{0}=n^{*}+P_{n^{*}}^{\perp}\left[\begin{array}{c}
\left(\Delta f_{x} / f_{x}^{*}\right) n_{x}^{*} \\
\left(\Delta f_{y} / f_{y}^{*}\right) n_{y}^{*} \\
\left(\Delta c_{x} / f_{x}^{*}\right) n_{x}^{*}+\left(\Delta c_{y} / f_{y}^{*}\right) n_{y}^{*}
\end{array}\right]+O\left(\|\Delta K\|^{2}\right), \quad P_{n^{*}}^{\perp}=I-n^{*} n^{* T}
$$

So the principal-point component of the tilt enters through the $z$-component of the perturbed normal, while focal error rescales the in-plane components. That is the requested first-order implicit dependence of the reconstructed plane normal on $\left(\Delta c_{x}, \Delta c_{y}, \Delta f\right)$.

The Manhattan prior changes this only through coupling between planes. For an isolated plane there is no orthogonality penalty to apply. If one has a local Manhattan triad $N^{*}=\left[n_{1}^{*}, n_{2}^{*}, n_{3}^{*}\right] \in S O(3)$, the unconstrained transformed triad is $\widetilde{N}=H^{-T} N^{*}$. The stationarity equations for the orthogonalityregularized problem, with unit-norm constraints, are
$$
\widehat{n}_{k}-\widetilde{n}_{k}+\lambda \sum_{\ell \perp k}\left(\widehat{n}_{k}^{\top} \widehat{n}_{\ell}\right) \widehat{n}_{\ell}+\mu_{k} \widehat{n}_{k}=0, \quad\left\|\widehat{n}_{k}\right\|=1
$$

That is the exact implicit system. In the strong-prior regime, the explicit asymptotic solution is the nearest orthonormal frame:
$$
\widehat{N}=\operatorname{polar}(\widetilde{N})=\operatorname{polar}\left(H^{-T} N^{*}\right)+O\left(\|\Delta K\|^{2}, \lambda^{-1}\right) .
$$

Thus the Manhattan prior keeps only the orthogonal part of the normal distortion and rejects the nonorthogonal shear part. ${ }^{15}$

For pure principal-point shear, $\alpha_{x}=\alpha_{y}=1$, so $H^{-T}=I-F^{\top}$ with
$$
F=\left[\begin{array}{ccc}
0 & 0 & \beta_{x} \\
0 & 0 & \beta_{y} \\
0 & 0 & 0
\end{array}\right]
$$

The polar factor has infinitesimal rotation
$$
\omega_{\mathrm{MW}}=\left(-\frac{\beta_{y}}{2}, \frac{\beta_{x}}{2}, 0\right)=\left(\frac{\Delta c_{y}}{2 f_{y}^{*}},-\frac{\Delta c_{x}}{2 f_{x}^{*}}, 0\right)+O\left(\|\Delta K\|^{2}\right)
$$
so, in the strong-Manhattan limit, the optimizer converts the shear of the normal frame into a global pitchyaw rotation of the triad. That half-factor is not a contradiction with the $\omega_{y}=\beta_{x}, \omega_{x}=-\beta_{y}$ result above: the former is the closest orthogonal frame to the sheared normal matrix, while the latter is the closest reprojection compensator after point elimination. They are projections in two different metrics.

## Residual distribution: shear, averaging, and curvature

The decisive answer to your 3.2 needs one more distinction. Principal-point error by itself is only a shear-like affine distortion, and even incompatible viewwise affine distortions do not automatically imply curvature. They imply residual. Curvature appears only after the particular fusion energy distributes that residual spatially in a non-affine way.

For view $t$, the first-order world-space displacement induced by the wrong intrinsic matrix is
$$
u_{t}(X)=R_{t}^{* T}(H-I)\left(R_{t}^{*} X+t_{t}^{*}\right)=A_{t} X+b_{t} .
$$

So each individual view induces an affine field, not a curved one. For pure principal-point drift,
$$
H-I=s e_{3}^{\top}, \quad s=\left(\beta_{x}, \beta_{y}, 0\right)^{\top},
$$
hence
$$
A_{t}=R_{t}^{* T} s\left(e_{3}^{\top} R_{t}^{*}\right)=a_{t} r_{3, t}^{* T}, \quad a_{t}:=R_{t}^{* T} s
$$

This is a rank-one shear in world coordinates. If all views are compatible with one common affine field, equivalently if $A_{t}=A$ and $b_{t}=b$ for all $t$, which happens in the single-view case and in rare commuting-motion cases, then the reconstructed manifold is just globally sheared: planes stay planes, zero Gaussian curvature remains zero, and the error is not a bend.

Generic multiview motion breaks that compatibility. The matrices $A_{t}=R_{t}^{* T}(H-I) R_{t}^{*}$ depend on the camera orientation, so different views request different shear directions and different depth-dependent offsets. No single global affine deformation can satisfy all of them simultaneously. This is exactly the same kind of incompatibility that, in self-calibration terms, prevents a wrong $\widehat{K}$ from being a valid metric upgrade for generic motion. ${ }^{18}$ But incompatibility alone is not yet a curvature theorem.

Indeed, consider the constant-weight affine fusion energy
$$
E_{\mathrm{aff}}[u]=\int \sum_{t} w_{t}\left\|u(X)-u_{t}(X)\right\|^{2} d\mu(X), \quad w_{t}>0 \text{ constant.}
$$
The pointwise minimizer is
$$
u^{\star}(X)=\bar{A} X+\bar{b}, \quad \bar{A}=\frac{\sum_{t} w_{t} A_{t}}{\sum_{t} w_{t}}, \quad \bar{b}=\frac{\sum_{t} w_{t} b_{t}}{\sum_{t} w_{t}} .
$$
Thus constant-weight least-squares fusion of affine requests is still affine. It leaves nonzero residuals
$$
u^{\star}(X)-u_{t}(X)=\left(\bar{A}-A_{t}\right) X+\left(\bar{b}-b_{t}\right),
$$
but it does not bend a plane. This is the minimal counterexample showing that residual incompatibility is not sufficient for curvature.

This is not merely a pathological one-line counterexample; it is the whole constant-coefficient quadratic fusion class. More generally, if the reduced local energy has constant matrix weights,
$$
E_{\mathrm{const}}[u]=\int \sum_{t}\left(u(X)-u_{t}(X)\right)^{T} W_{t}\left(u(X)-u_{t}(X)\right) d\mu(X),
$$
with $W_{t}\succeq 0$ independent of $X$, then the pointwise minimizer is
$$
u^{\star}(X)=\left(\sum_{t} W_{t}\right)^{-1}\sum_{t} W_{t}\left(A_{t}X+b_{t}\right),
$$
whenever $\sum_{t}W_t$ is nonsingular. This is still affine. Thus residual incompatibility is not sufficient for curvature in any constant-weight first-order displacement fusion model.

Actual reprojection-based fusion is not usually in this constant-coefficient class. After linearization, a local displacement channel has the form
$$
E_{\mathrm{red}}[u]=\int \sum_{t}\left(u(s)-u_{t}(s)\right)^{T} W_{t}(s)\left(u(s)-u_{t}(s)\right) d s,
$$
where
$$
W_{t}(s)=J_{t}(s)^{T} J_{t}(s)
$$
is the reduced reprojection, Schur, confidence, and visibility weight seen by that surface coordinate. Even with constant feature confidence, $J_t(s)$ varies with depth, view angle, projection scale, occlusion, and the chosen surface parameterization. This is the non-affine response operator that the previous paragraph was missing.

Now write
$$
u_{t}(s)=b_{t}+a_{t}s+O(s^{2}),
$$
and
$$
W_{t}(s)=W_{t0}+W_{t1}s+\frac{1}{2}W_{t2}s^{2}+O(s^{3}).
$$
Define
$$
M_{k}=\sum_{t} W_{t k}, \quad q_{0}=\sum_{t} W_{t0}b_{t},
$$
$$
q_{1}=\sum_{t}\left(W_{t0}a_{t}+W_{t1}b_{t}\right),
$$
and
$$
q_{2}=\sum_{t}\left(W_{t1}a_{t}+\frac{1}{2}W_{t2}b_{t}\right).
$$
The pointwise minimizer satisfies
$$
u^{\star}(s)=M(s)^{-1}q(s)=c_{0}+c_{1}s+c_{2}s^{2}+O(s^{3}),
$$
with
$$
c_{0}=M_{0}^{-1}q_{0},
$$
$$
c_{1}=M_{0}^{-1}\left(q_{1}-M_{1}c_{0}\right),
$$
and
$$
c_{2}=M_{0}^{-1}\left(q_{2}-M_{1}c_{1}-\frac{1}{2}M_{2}c_{0}\right).
$$
This is the closed-form residual-distribution law for the local reduced quadratic model. The bending channel is $2c_{2}$. If $W_{t1}=W_{t2}=0$ for all $t$, then $c_{2}=0$ and the response is affine. If the spatial variation of the reduced weights is not aligned with the affine requests, $c_{2}$ is generically nonzero. In the algebraic sense, this genericity statement follows because $c_{2}$ is an analytic function of $\{W_{t0},W_{t1},W_{t2},a_t,b_t\}$ that is not identically zero; the zero set is therefore lower-dimensional unless additional symmetries force it.

The scalar visibility/confidence model is the commutative special case of this matrix formula. Let the normal/depth-channel request from view $t$ be
$$
f_{t}(s)=a_{t} s+b_{t}+O(s^{2}) .
$$
Consider the data energy
$$
E_{0}[\zeta]=\int \sum_{t} w_{t}(s)\left(\zeta(s)-f_{t}(s)\right)^{2} d s,
$$
where
$$
w_{t}(s)=w_{t 0}\left(1+\eta_{t} s+\frac{1}{2}\kappa_{t} s^{2}+O(s^{3})\right), \quad w_{t 0}>0 .
$$
With $\pi_{t}=w_{t 0} / \sum_{j} w_{j 0}$ and $\langle g\rangle=\sum_{t} \pi_{t} g_{t}$, the pointwise minimizer is
$$
\zeta^{\star}(s)=\frac{\sum_{t} w_{t}(s) f_{t}(s)}{\sum_{t} w_{t}(s)}=c_{0}+c_{1} s+c_{2} s^{2}+O(s^{3}),
$$
where
$$
c_{0}=\langle b\rangle,
$$
$$
c_{1}=\langle a\rangle+\operatorname{Cov}(b,\eta),
$$
and
$$
c_{2}=\operatorname{Cov}(a,\eta)+\frac{1}{2} \operatorname{Cov}(b,\kappa)-\langle\eta\rangle \operatorname{Cov}(b,\eta).
$$
Therefore the local bending response in this scalar channel is
$$
\zeta^{\star \prime \prime}(0)=2 c_{2}.
$$
This formula is the nondegenerate residual-distribution law. If all weights are constant, $\eta_{t}=\kappa_{t}=0$, then $c_{2}=0$ and the response is affine. Bending appears only when the spatial variation of the weights is correlated with the incompatible affine slopes or offsets. Equivalently, view incompatibility provides the forcing, while spatially varying visibility or confidence decides where that forcing is deposited.

If one adds a thin-plate or depth smoothness term,
$$
E_{\lambda}[\zeta]=\int \sum_{t} w_{t}(s)\left(\zeta(s)-f_{t}(s)\right)^{2} d s+\lambda \int\left|\zeta^{\prime \prime}(s)\right|^{2} d s,
$$
then the Euler-Lagrange equation is
$$
\lambda \zeta^{\prime \prime \prime \prime}(s)+W(s) \zeta(s)=R(s), \quad W(s)=\sum_{t} w_{t}(s), \quad R(s)=\sum_{t} w_{t}(s) f_{t}(s).
$$
For constant $W$ and affine $R$, the interior solution is affine, up to boundary-layer effects. Smoothness does not by itself create bending from affine requests; it propagates bending created by non-affine forcing, spatially varying weights, nonlinear projection operators, boundaries, occlusion, or regularization constraints.

The Manhattan-normal version has the same structure but on $S O(3)$. For one view, a local affine displacement $X \mapsto X+u_{t}(X)$ with gradient $A_{t}$ sends normals to
$$
N_{t}=\left(I-A_{t}^{T}\right) N^{*}+O\left(\|A_{t}\right\|^{2}).
$$
With a strong Manhattan prior, the pointwise normal-frame estimate solves
$$
N^{\star}(s)=\operatorname{polar}\left(M(s)\right), \quad M(s)=\frac{\sum_{t} w_{t}(s) N_{t}}{\sum_{t} w_{t}(s)} .
$$
To first order,
$$
N^{\star}(s)=\left(I+\left[\omega_{\mathrm{MW}}(s)\right]_{\times}\right) N^{*},
$$
where
$$
\left[\omega_{\mathrm{MW}}(s)\right]_{\times}=\operatorname{skew}\left(-\bar{A}_{w}(s)^{T}\right), \quad \bar{A}_{w}(s)=\frac{\sum_{t} w_{t}(s) A_{t}}{\sum_{t} w_{t}(s)} .
$$
Thus a constant weighted average $\bar{A}_{w}$ gives only a constant Manhattan-frame rotation. Curvature of a plane requires a spatially varying normal,
$$
\partial_{s} N^{\star}(0)=\left[\omega_{\mathrm{MW}}^{\prime}(0)\right]_{\times} N^{*},
$$
with
$$
\omega_{\mathrm{MW}}^{\prime}(0)=\sum_{t} \pi_{t}\left(\eta_{t}-\langle\eta\rangle\right) \omega_{\mathrm{MW}, t}.
$$
So in the Manhattan regime, bending is controlled by the covariance between the spatial weight gradients and the per-view shear-induced Manhattan rotations. If all views have constant relative weight, the normal frame is merely rotated and no plane curvature is created.

Now parametrize a surface patch by image coordinates $(u, v)$. For a single view with principal-point shift $\beta=\left(\beta_{x}, \beta_{y}\right)$, the wrong-ray depth field can be written locally as a coordinate transport,
$$
Z_{\beta}(u, v)=Z^{*}\left(u-\beta_{x}, v-\beta_{y}\right),
$$
so
$$
\delta Z_{\text{single}}(u, v)=-\beta \cdot \nabla Z^{*}(u, v)+O\left(\|\beta\|^{2}\right) .
$$
This is a first-order transport term, not intrinsic curvature. In other words, the depth error moves along the gradient; it does not yet bend the surface intrinsically. 19

The corrected geometric principle is therefore:
$$
\text{wrong intrinsics} \Rightarrow \text{viewwise affine shear requests } u_{t}(X)=A_{t} X+b_{t}.
$$
Unconstrained reconstruction can keep this as a projective or affine gauge change. Bundle adjustment with fixed Euclidean cameras projects part of it onto $\mathfrak{s e}(3)$, chiefly yaw and pitch for principal-point drift, and marginalizes the rest into structure. But the spatial distribution of the remaining residual is not determined by incompatibility alone. Constant-weight affine fusion gives an affine average and no bend. Bending requires a non-affine response operator: spatially varying visibility/confidence, nonlinear reprojection weights, occlusion boundaries, smoothness with non-affine forcing, or Manhattan-frame weights that vary across the patch. In short: single view or affine-compatible motion gives shear; generic multiview gives residual; residual becomes bend only through a specified non-affine fusion or regularization mechanism.
