# Intrinsic-Aware Foundation-Prior Bundle Adjustment

## Joint Pinhole Calibration, Multiview Geometry, and Bias-Corrected Learned Priors

### Abstract

Feed-forward 3D foundation models predict camera poses, depth maps, point maps, tracks, normals, and sometimes intrinsics directly from images. These predictions are powerful but geometrically fragile: many such models assume a pinhole camera whose principal point is at the image center, while real images may be asymmetrically cropped, stabilized, or otherwise represented by a non-centered pinhole camera. We introduce intrinsic-aware foundation-prior bundle adjustment, a joint optimization framework that estimates a shared pinhole camera, camera poses, and 3D structure while using foundation-model outputs as weak learned priors. The central difficulty is that learned outputs from a centered-principal-point model are not unbiased noisy measurements when the true principal point is off-center; they are biased responses to a wrong intrinsic model. We therefore derive the geometry needed to correct, gate, or delay these priors. Focal-length error acts as anisotropic stretch, while principal-point error acts as rank-one shear in camera coordinates. We derive how this shear is absorbed by rigid pose, Schur-complement bundle adjustment, pointwise depth, and Manhattan normal regularization. After point elimination, principal-point drift is generically absorbed as yaw and pitch rather than lateral translation, while constant-depth sheets expose a translation-rotation ambiguity. We further show that incompatible viewwise shears imply residual but not automatically curvature; bending arises only through a specified non-affine response operator, such as spatially varying reprojection weights, visibility, confidence, occlusion, or Manhattan normal coupling. The resulting method combines correspondence geometry, shared-intrinsic calibration, and bias-aware learned priors in a single robust optimization pipeline.

## 1. Introduction

### 1.1 Joint Geometry with Learned Priors

Classical multiview geometry can estimate camera intrinsics and poses from correspondences. It is reliable when there are enough static, well-distributed matches and nondegenerate camera motion, but it becomes ill-conditioned under sparse texture, narrow baseline, dynamic objects, low parallax, or weak constraints on principal point and focal length.

Recent 3D foundation models suggest a complementary strategy: use feed-forward predictions as weak priors while still enforcing multiview geometry. We propose the joint objective

$$
\min_{K,\{T_t\},\{X_j\},\{Z_t\}}
E_{\mathrm{corr}}
+
\lambda_{\mathrm{BA}}E_{\mathrm{reproj}}
+
\lambda_{\mathrm{FM}}E_{\mathrm{FM}}
+
\lambda_K E_K.
$$

Here \(E_{\mathrm{corr}}\) and \(E_{\mathrm{reproj}}\) enforce multiview geometry, \(E_K\) regularizes the shared pinhole intrinsics, and \(E_{\mathrm{FM}}\) pulls the solution toward foundation-model predictions:

$$
E_{\mathrm{FM}}
=
E_{\mathrm{pose}}(T_t,\widehat T_t)
+
E_{\mathrm{depth}}(Z_t,\widehat Z_t)
+
E_{\mathrm{normal}}(N_t,\widehat N_t)
+
E_{\mathrm{point}}(X_j,\widehat X_j).
$$

This pipeline is attractive because learned predictions can regularize ill-conditioned calibration and reconstruction, especially when correspondence geometry alone is noisy, sparse, or locally degenerate. But the learned outputs are useful only if they are valid weak priors. If the foundation model was run under a wrong camera assumption, they may not be.

### 1.2 The Prior-Validity Problem

Many feed-forward 3D systems assume a centered pinhole camera. In these systems, the intrinsic matrix is effectively restricted to

$$
K_0 =
\begin{bmatrix}
f_x & 0 & W/2 \\
0 & f_y & H/2 \\
0 & 0 & 1
\end{bmatrix}.
$$

The true pinhole camera may instead be

$$
K^* =
\begin{bmatrix}
f_x^* & 0 & c_x^* \\
0 & f_y^* & c_y^* \\
0 & 0 & 1
\end{bmatrix},
$$

with

$$
(c_x^*, c_y^*) \neq (W/2, H/2).
$$

If \(K^*\) is known, the natural engineering solution is simple: remap the input image to a centered canonical camera \(K_\star\), then run the centered-camera model. For a distortion-free pinhole camera this is a deterministic image warp:

$$
I_\star(q_\star) = I(K^* K_\star^{-1} q_\star).
$$

This requires no scene depth because it changes only the camera ray parameterization at the same optical center. But a practical joint system often starts from learned predictions on the original image before the correct \(K^*\) is known. Those predictions are conditioned on the model's centered-camera convention. Consequently,

$$
(\widehat T_t,\widehat Z_t,\widehat N_t,\widehat X_j)
$$

should not be interpreted as unbiased noisy measurements of the true geometry. They are wrong-intrinsic responses.

This paper addresses the geometric question needed to make learned-prior joint calibration well-posed:

> If a non-centered pinhole image is interpreted under a centered-principal-point camera, what systematic biases should we expect in rotation, translation, focal length, depth, plane normals, and multiview surface geometry?

This question is specific to learned-prior optimization: a purely correspondence-based optimizer can estimate geometry without knowing how a foundation model fails, but a learned-prior optimizer must understand the bias structure before trusting its priors.

### Contributions

1. We introduce intrinsic-aware foundation-prior bundle adjustment, a joint optimization framework for shared pinhole calibration, camera poses, 3D structure, and learned pose/depth/normal/point priors.
2. We formulate the prior-validity problem: foundation-model outputs are biased when generated under a centered-principal-point assumption but applied to non-centered pinhole imagery.
3. We express intrinsic mismatch as a camera-coordinate affine map \(H=\widehat K^{-1}K^*\), separating focal-length stretch from principal-point shear.
4. We show that single-view reprojection has an infinite family of exact wrong-intrinsic solutions when depth is freely reassigned, so single-view residual alone cannot diagnose the bias.
5. We derive the least-squares rigid projection of intrinsic shear onto \(SE(3)\), including closed-form yaw/pitch excitation for principal-point drift.
6. We derive the first-order bundle-adjustment response using on-manifold Jacobians and Schur complement point elimination, giving an explicit local map from intrinsic error to pose and structure bias.
7. We show that after point elimination and under nondegenerate depth variation, principal-point error is first-order rotational rather than translational; translation appears as a soft ambiguity in constant-depth scenes.
8. We derive the effect of wrong intrinsics on Manhattan plane normals and distinguish the reprojection-compensation metric from the Manhattan-frame metric.
9. We prove that incompatible viewwise affine shears do not by themselves imply curvature; constant-coefficient fusion gives an affine average. We then derive a local reduced quadratic model in which curvature appears through spatially varying reprojection, visibility, or confidence weights.
10. We translate these derivations into a practical prior policy: learned priors should be corrected, de-biased, gated, or delayed until the expected wrong-intrinsic response is modeled.

## 2. Problem Setup

Let \(K^*\) be the true pinhole intrinsics and \(\widehat K\) be the intrinsics assumed by a reconstruction system. We define

$$
H = \widehat K^{-1} K^*
=
\begin{bmatrix}
\alpha_x & 0 & \beta_x \\
0 & \alpha_y & \beta_y \\
0 & 0 & 1
\end{bmatrix},
$$

where

$$
\alpha_x = \frac{f_x^*}{\widehat f_x},
\qquad
\beta_x = \frac{c_x^*-\widehat c_x}{\widehat f_x},
$$

and analogously for \(y\). In camera coordinates,

$$
H
\begin{bmatrix}
X \\ Y \\ Z
\end{bmatrix}
=
\begin{bmatrix}
\alpha_x X+\beta_x Z \\
\alpha_y Y+\beta_y Z \\
Z
\end{bmatrix}.
$$

Thus focal-length error is an anisotropic stretch in the image-plane directions, while principal-point error is a rank-one shear:

$$
H-I = s e_3^T,
\qquad
s=(\beta_x,\beta_y,0)^T
$$

for pure principal-point drift. In normalized image coordinates,

$$
\pi(H[X,Y,Z]^T)
=
\begin{bmatrix}
\alpha_x X/Z + \beta_x \\
\alpha_y Y/Z + \beta_y
\end{bmatrix}.
$$

Principal-point drift therefore appears as a constant normalized image-plane shift, while focal drift appears as scaling. This simple observation is the seed of the entire difficulty.

## 3. Identifiability Landscape

The decisive fact is that uncalibrated image correspondences determine a reconstruction only up to a 3D projective transformation. Passing from this projective class to a metric or Euclidean reconstruction requires calibration information, classically through the plane at infinity and the image of the absolute conic.

In bundle-adjustment language this is a gauge problem:

- Euclidean reconstruction has a 7-DOF similarity gauge.
- Projective reconstruction has a 15-DOF projective gauge.

The Manhattan-world assumption imposes orthogonality on dominant directions or plane normals, and therefore provides exactly the kind of extra geometric information that can break part of the projective ambiguity.

### 3.1 Single-View Non-Uniqueness

The statement "wrong principal point implies positive residual" is false for a single view if depths may be freely reassigned.

Fix a true camera-frame point

$$
p_i^* = T^* X_i^*.
$$

Define the wrong-intrinsic ray

$$
\tilde r_i
:=
\frac{H p_i^*}{e_3^T H p_i^*}
\in \mathbb R^3.
$$

For any chosen \(\widehat T \in SE(3)\) and any positive scalars \(\rho_i\), define

$$
\widehat X_i
:=
\widehat T^{-1}(\rho_i \tilde r_i).
$$

Then

$$
\pi(\widehat T \widehat X_i)
=
\pi(\rho_i \tilde r_i)
=
\pi(Hp_i^*).
$$

Thus the reprojection error is exactly zero. There is not merely non-uniqueness; there is an infinite family of exact wrong-intrinsic single-view solutions. This is the same projective non-uniqueness that underlies uncalibrated reconstruction.

### 3.2 Multiview Metric Compatibility

The meaningful statement is multiview:

> If a single common 3D structure and all cameras are constrained to use the same fixed wrong intrinsics \(\widehat K\), exact zero residual means the same projective reconstruction admits two Euclidean upgrades, one with \(K^*\) and one with \(\widehat K\).

For generic nonplanar scenes and generic noncritical motion, self-calibration is locally unique up to a similarity transform. Therefore the second upgrade cannot exist unless \(\widehat K\) lies in the same metric-upgrade class as \(K^*\), which generically forces

$$
\widehat K = K^*.
$$

Wrong intrinsics can remain exactly compatible only in degenerate cases such as planar scenes, pure rotations, or other critical motion sequences.

## 4. Rigid Projection of Intrinsic Shear

We first ask how the affine map \(H\) is approximated by a rigid motion in one camera frame.

Let \(p_i^* = T^* X_i^*\) be camera-frame points. Define

$$
\bar p = \frac{1}{N}\sum_i p_i^*,
\qquad
q_i = p_i^*-\bar p.
$$

Let

$$
H=I+E,
\qquad
R=I+[\delta\omega]_\times+O(\|E\|^2),
\qquad
t=\delta t+O(\|E\|^2).
$$

Minimizing

$$
\sum_i \|H p_i^*-(R p_i^*+t)\|^2
$$

gives, to first order,

$$
\delta t = E\bar p - \delta\omega \times \bar p,
$$

and

$$
(\operatorname{tr}\Sigma I-\Sigma)\delta\omega
=
\frac{1}{N}\sum_i q_i \times E q_i,
$$

where

$$
\Sigma
:=
\frac{1}{N}\sum_i q_i q_i^T.
$$

Equivalently, if \(C=H\Sigma\), the exact least-squares rotation is the polar factor

$$
\widehat R = \operatorname{polar}(C).
$$

The linearized skew matrix \(\Omega=[\delta\omega]_\times\) solves

$$
\Omega\Sigma+\Sigma\Omega
=
E\Sigma-\Sigma E^T.
$$

This is the exact algebraic structure of the rigid projection: one keeps the rigid part of the affine perturbation and discards the symmetric nonrigid part.

### 4.1 Pure Principal-Point Shear

For pure principal-point shear,

$$
E=\beta_x e_1e_3^T+\beta_y e_2e_3^T.
$$

Then

$$
\frac{1}{N}\sum_i q_i\times E q_i
=
\begin{bmatrix}
-\beta_y \Sigma_{zz} \\
\beta_x \Sigma_{zz} \\
\beta_y \Sigma_{xz}-\beta_x \Sigma_{yz}
\end{bmatrix}.
$$

Hence principal-point error excites mainly pitch and yaw, and excites roll only if the point cloud has \(xz\) or \(yz\) covariance.

In the common symmetric case

$$
\Sigma=\operatorname{diag}(\sigma_x^2,\sigma_y^2,\sigma_z^2),
$$

we obtain

$$
\delta\omega_x
=
-\beta_y
\frac{\sigma_z^2}{\sigma_y^2+\sigma_z^2},
$$

$$
\delta\omega_y
=
\beta_x
\frac{\sigma_z^2}{\sigma_x^2+\sigma_z^2},
$$

and

$$
\delta\omega_z=0.
$$

The translation is

$$
\delta t = E\bar p-\delta\omega\times\bar p.
$$

In the limiting case

$$
\sigma_z^2 \gg \sigma_x^2,\sigma_y^2,
$$

the rigid absorber approaches

$$
\delta\omega\approx(-\beta_y,\beta_x,0),
$$

which is exactly the infinitesimal camera rotation that mimics a principal-point shift near the optical axis.

## 5. Bundle-Adjustment Response

Let

$$
p_{ti}
=
R_t^*X_i^*+t_t^*
=
(X_{ti},Y_{ti},Z_{ti})^T,
$$

and define

$$
x_{ti} = \frac{X_{ti}}{Z_{ti}},
\qquad
y_{ti} = \frac{Y_{ti}}{Z_{ti}}.
$$

Use a left-multiplicative pose perturbation

$$
\widehat T_t
=
\exp((\Delta\xi_t)^\wedge)T_t^*,
\qquad
\Delta\xi_t =
\begin{bmatrix}
\Delta\rho_t \\
\Delta\phi_t
\end{bmatrix}
\in \mathbb R^6,
$$

and let

$$
\widehat X_i=X_i^*+\Delta X_i.
$$

The predicted pixel is

$$
z_{ti}(\theta_K,\xi_t,X_i)
=
\begin{bmatrix}
f_x X_{ti}/Z_{ti}+c_x \\
f_y Y_{ti}/Z_{ti}+c_y
\end{bmatrix}.
$$

The measurement Jacobians at the true state are

$$
G_{p,ti}
:=
\frac{\partial z_{ti}}{\partial p_{ti}}
=
\begin{bmatrix}
f_x/Z_{ti} & 0 & -f_xX_{ti}/Z_{ti}^2 \\
0 & f_y/Z_{ti} & -f_yY_{ti}/Z_{ti}^2
\end{bmatrix},
$$

$$
G_{\xi,ti}
:=
\frac{\partial z_{ti}}{\partial\Delta\xi_t}
=
G_{p,ti}
\begin{bmatrix}
I & -[p_{ti}]_\times
\end{bmatrix},
$$

and

$$
G_{X,ti}
:=
\frac{\partial z_{ti}}{\partial\Delta X_i}
=
G_{p,ti}R_t^*.
$$

The intrinsic Jacobian is

$$
G_{K,ti}
:=
\frac{\partial z_{ti}}{\partial(f_x,f_y,c_x,c_y)}
=
\begin{bmatrix}
x_{ti} & 0 & 1 & 0 \\
0 & y_{ti} & 0 & 1
\end{bmatrix}.
$$

Expanding \(G_{\xi,ti}\), with \(X,Y,Z=(X_{ti},Y_{ti},Z_{ti})\),

$$
G_{\xi,ti}
=
\begin{bmatrix}
\frac{f_x}{Z} & 0 & -\frac{f_xX}{Z^2}
& -\frac{f_xXY}{Z^2}
& f_x\left(1+\frac{X^2}{Z^2}\right)
& -\frac{f_xY}{Z}
\\
0 & \frac{f_y}{Z} & -\frac{f_yY}{Z^2}
& -f_y\left(1+\frac{Y^2}{Z^2}\right)
& \frac{f_yXY}{Z^2}
& \frac{f_yX}{Z}
\end{bmatrix}.
$$

These are the analytical Jacobians used in on-manifold bundle adjustment.

### 5.1 First-Order Stationary-Point Map

Let \(r\) be the stacked reprojection residual. At first order,

$$
r \approx
-G_\Xi\Delta\Xi
-G_X\Delta X
-G_K\Delta\theta_K.
$$

On a gauge-fixed slice of the state space, the implicit-function-theorem derivative of the stationary-point map is

$$
\frac{\partial(\Delta\Xi,\Delta X)}
{\partial\theta_K}
=
-\mathcal H^{-1}\mathcal B_K,
$$

where

$$
\mathcal H =
\begin{bmatrix}
U & W \\
W^T & V
\end{bmatrix}
=
\begin{bmatrix}
G_\Xi^TG_\Xi & G_\Xi^TG_X \\
G_X^TG_\Xi & G_X^TG_X
\end{bmatrix}
+\lambda\nabla^2\mathcal E_{\text{prior}},
$$

and

$$
\mathcal B_K =
\begin{bmatrix}
G_\Xi^TG_K \\
G_X^TG_K
\end{bmatrix}.
$$

Without gauge fixing, \(\mathcal H^{-1}\) is replaced by the Moore-Penrose pseudoinverse because of the similarity-gauge null space.

Eliminating points by Schur complement gives

$$
\Delta\Xi
=
-
\underbrace{(U-WV^{-1}W^T)^{-1}}_{S^{-1}}
\underbrace{(G_\Xi^TG_K-WV^{-1}G_X^TG_K)}_{\Gamma_K}
\Delta\theta_K.
$$

Thus

$$
\Delta\xi_t
=
\mathcal F_t(T_t^*,\mathcal X^*,\Delta K)
=
-[e_t^T S^{-1}\Gamma_K]\Delta\theta_K,
$$

where \(e_t\) selects the \(t\)-th 6-vector pose block. This is the reduced camera system familiar from sparse bundle adjustment: pointwise depth flexibility is compressed into the Schur term \(WV^{-1}W^T\), and only the intrinsic forcing that survives point elimination drives pose bias.

### 5.2 Principal-Point Drift Near the Optical Axis

Let

$$
\beta_x
=
\frac{c_x^*-\widehat c_x}{\widehat f_x}
\approx
-\frac{\Delta c_x}{f_x^*},
$$

and

$$
\beta_y
=
\frac{c_y^*-\widehat c_y}{\widehat f_y}
\approx
-\frac{\Delta c_y}{f_y^*}.
$$

Near the optical axis, keeping only the dominant pose channels, the normalized-image perturbation model is

$$
\delta x_i
\approx
-\frac{\Delta t_x}{Z_i}
+\omega_y,
$$

$$
\delta y_i
\approx
-\frac{\Delta t_y}{Z_i}
-\omega_x.
$$

The principal-point subproblems decouple:

$$
\min_{\Delta t_x,\omega_y}
\sum_i
\left(
\beta_x+\frac{\Delta t_x}{Z_i}-\omega_y
\right)^2,
$$

and

$$
\min_{\Delta t_y,\omega_x}
\sum_i
\left(
\beta_y+\frac{\Delta t_y}{Z_i}+\omega_x
\right)^2.
$$

The \(x\)-block Hessian is

$$
H_x =
\begin{bmatrix}
\sum_i Z_i^{-2} & -\sum_i Z_i^{-1} \\
-\sum_i Z_i^{-1} & N
\end{bmatrix}.
$$

When

$$
\operatorname{Var}(1/Z_i)>0,
$$

the exact normal-equation solution is

$$
\omega_y=\beta_x,
\qquad
\Delta t_x=0,
$$

and

$$
\omega_x=-\beta_y,
\qquad
\Delta t_y=0.
$$

Therefore, in the generic nondegenerate-depth regime,

$$
\frac{\partial\omega_y}{\partial\Delta c_x}
\approx
-\frac{1}{f_x^*},
$$

$$
\frac{\partial\omega_x}{\partial\Delta c_y}
\approx
\frac{1}{f_y^*},
$$

and

$$
\frac{\partial\Delta t_x}{\partial\Delta c_x}
\approx
\frac{\partial\Delta t_y}{\partial\Delta c_y}
\approx
0.
$$

Thus principal-point error is first-order rotational, not translational, after point elimination under this near-axis reduced model.

### 5.3 Constant-Depth Degeneracy

If

$$
Z_i \equiv Z_0,
$$

then

$$
\det H_x
=
N\sum_i Z_i^{-2}
-
\left(\sum_i Z_i^{-1}\right)^2
=0.
$$

The family of exact first-order compensators becomes

$$
\omega_y-\frac{\Delta t_x}{Z_0}=\beta_x,
$$

and

$$
-\omega_x-\frac{\Delta t_y}{Z_0}=\beta_y.
$$

A fronto-parallel constant-depth sheet cannot distinguish principal-point shift from a coupled yaw/translation or pitch/translation mode.

For

$$
Z\sim\operatorname{Uniform}[Z_{\min},Z_{\max}],
$$

we have

$$
\mathbb E[Z^{-1}]
=
\frac{\ln(Z_{\max}/Z_{\min})}{Z_{\max}-Z_{\min}},
$$

and

$$
\mathbb E[Z^{-2}]
=
\frac{1}{Z_{\min}Z_{\max}}.
$$

Hence

$$
\frac{\det H_x}{N^2}
=
\frac{1}{Z_{\min}Z_{\max}}
-
\left(
\frac{\ln(Z_{\max}/Z_{\min})}{Z_{\max}-Z_{\min}}
\right)^2
>0.
$$

The degeneracy is broken as soon as inverse-depth variance is nonzero. The soft eigen-directions in the constant-depth limit are

$$
v_{\text{soft}}^{(x)}\propto(Z_0,1),
\qquad
v_{\text{soft}}^{(y)}\propto(Z_0,-1),
$$

corresponding to the coupled \((\Delta t_x,\omega_y)\) and \((\Delta t_y,\omega_x)\) ambiguities.

## 6. Manhattan-Constrained Plane Normals

For one true plane in camera coordinates,

$$
\Pi_{\text{true}}:\quad n^{*T}X=d,
$$

the wrong-intrinsic, no-prior reconstruction remains exactly planar. If

$$
X=ZK^{*-1}q
$$

is the true point on the ray \(q\), then the wrong-ray reconstruction is

$$
\widehat X=HX.
$$

The reconstructed plane is

$$
\widehat\Pi_0:\quad \tilde n^T\widehat X=d,
\qquad
\tilde n=H^{-T}n^*.
$$

Since

$$
H^{-T}
=
\begin{bmatrix}
1/\alpha_x & 0 & 0 \\
0 & 1/\alpha_y & 0 \\
-\beta_x/\alpha_x & -\beta_y/\alpha_y & 1
\end{bmatrix},
$$

the exact unconstrained normal is

$$
\tilde n
=
\begin{bmatrix}
n_x^*/\alpha_x \\
n_y^*/\alpha_y \\
n_z^*-\beta_x n_x^*/\alpha_x-\beta_y n_y^*/\alpha_y
\end{bmatrix},
$$

with normalized normal

$$
\widehat n_0
=
\frac{\tilde n}{\|\tilde n\|}.
$$

Thus a fronto-parallel plane \(n^*=e_3\) is unaffected by principal-point drift, while side walls or floors tilt because their normals have \(x\)- or \(y\)-components.

Writing

$$
\widehat f_x=f_x^*+\Delta f_x,
\qquad
\widehat c_x=c_x^*+\Delta c_x,
$$

we have

$$
\alpha_x
=
1-\frac{\Delta f_x}{f_x^*}
+O(\|\Delta K\|^2),
$$

and

$$
\beta_x
=
-\frac{\Delta c_x}{f_x^*}
+O(\|\Delta K\|^2),
$$

and similarly in \(y\). Therefore

$$
\tilde n
=
n^*
+
\begin{bmatrix}
(\Delta f_x/f_x^*)n_x^* \\
(\Delta f_y/f_y^*)n_y^* \\
(\Delta c_x/f_x^*)n_x^*
+
(\Delta c_y/f_y^*)n_y^*
\end{bmatrix}
+O(\|\Delta K\|^2).
$$

After renormalization,

$$
\widehat n_0
=
n^*
+
P_{n^*}^{\perp}
\begin{bmatrix}
(\Delta f_x/f_x^*)n_x^* \\
(\Delta f_y/f_y^*)n_y^* \\
(\Delta c_x/f_x^*)n_x^*
+
(\Delta c_y/f_y^*)n_y^*
\end{bmatrix}
+O(\|\Delta K\|^2),
$$

where

$$
P_{n^*}^{\perp}=I-n^*n^{*T}.
$$

The principal-point component of the tilt enters through the \(z\)-component of the perturbed normal, while focal error rescales the in-plane components.

### 6.1 Strong Manhattan Prior

For an isolated plane there is no orthogonality penalty to apply. If a local Manhattan triad is

$$
N^* = [n_1^*,n_2^*,n_3^*]\in SO(3),
$$

then the unconstrained transformed triad is

$$
\tilde N=H^{-T}N^*.
$$

The stationarity equations for the orthogonality-regularized problem, with unit-norm constraints, are

$$
\widehat n_k-\tilde n_k
+
\lambda
\sum_{\ell\perp k}
(\widehat n_k^T\widehat n_\ell)\widehat n_\ell
+
\mu_k\widehat n_k
=0,
$$

with

$$
\|\widehat n_k\|=1.
$$

This is the exact implicit system. In the strong-prior regime, the explicit asymptotic solution is the nearest orthonormal frame:

$$
\widehat N
=
\operatorname{polar}(\tilde N)
=
\operatorname{polar}(H^{-T}N^*)
+O(\|\Delta K\|^2,\lambda^{-1}).
$$

Thus the Manhattan prior keeps only the orthogonal part of the normal distortion and rejects the nonorthogonal shear part.

For pure principal-point shear,

$$
\alpha_x=\alpha_y=1,
$$

so

$$
H^{-T}=I-F^T,
$$

with

$$
F=
\begin{bmatrix}
0 & 0 & \beta_x \\
0 & 0 & \beta_y \\
0 & 0 & 0
\end{bmatrix}.
$$

The polar factor has infinitesimal rotation

$$
\omega_{\mathrm{MW}}
=
\left(-\frac{\beta_y}{2},\frac{\beta_x}{2},0\right)
=
\left(
\frac{\Delta c_y}{2f_y^*},
-\frac{\Delta c_x}{2f_x^*},
0
\right)
+O(\|\Delta K\|^2).
$$

In the strong-Manhattan limit, the optimizer converts the shear of the normal frame into a global pitch-yaw rotation of the triad. The half-factor is not a contradiction with the reprojection compensation

$$
(\omega_x,\omega_y)=(-\beta_y,\beta_x).
$$

The former is the closest orthogonal frame to the sheared normal matrix; the latter is the closest reprojection compensator after point elimination. They are projections in two different metrics.

## 7. Residual Distribution: Shear, Averaging, and Curvature

Principal-point error by itself is a shear-like affine distortion. Even incompatible viewwise affine distortions do not automatically imply curvature. They imply residual. Curvature appears only after the particular fusion energy distributes that residual spatially in a non-affine way.

For view \(t\), the first-order world-space displacement induced by the wrong intrinsic matrix is

$$
u_t(X)
=
R_t^{*T}(H-I)(R_t^*X+t_t^*)
=
A_tX+b_t.
$$

For pure principal-point drift,

$$
H-I=se_3^T,
\qquad
s=(\beta_x,\beta_y,0)^T,
$$

hence

$$
A_t
=
R_t^{*T}s(e_3^TR_t^*)
=
a_t r_{3,t}^{*T},
$$

where

$$
a_t:=R_t^{*T}s.
$$

This is a rank-one shear in world coordinates. If all views are compatible with one common affine field, then the reconstructed manifold is globally sheared: planes stay planes, zero Gaussian curvature remains zero, and the error is not a bend.

Generic multiview motion breaks compatibility because \(A_t=R_t^{*T}(H-I)R_t^*\) depends on camera orientation. Different views request different shear directions and depth-dependent offsets. No single global affine deformation can satisfy all of them simultaneously. However, incompatibility alone is not yet a curvature theorem.

### 7.1 A No-Bending Class

Consider the constant-weight affine fusion energy

$$
E_{\mathrm{aff}}[u]
=
\int
\sum_t
w_t\|u(X)-u_t(X)\|^2
d\mu(X),
$$

with \(w_t>0\) constant. The pointwise minimizer is

$$
u^\star(X)
=
\bar A X+\bar b,
$$

where

$$
\bar A
=
\frac{\sum_t w_tA_t}{\sum_t w_t},
\qquad
\bar b
=
\frac{\sum_t w_tb_t}{\sum_t w_t}.
$$

Thus constant-weight least-squares fusion of affine requests is still affine. It leaves residuals

$$
u^\star(X)-u_t(X)
=
(\bar A-A_t)X+(\bar b-b_t),
$$

but it does not bend a plane.

This is not a pathological counterexample. It is the whole constant-coefficient quadratic fusion class. More generally, if

$$
E_{\mathrm{const}}[u]
=
\int
\sum_t
(u(X)-u_t(X))^T W_t (u(X)-u_t(X))
d\mu(X),
$$

with \(W_t\succeq0\) independent of \(X\), then

$$
u^\star(X)
=
\left(\sum_t W_t\right)^{-1}
\sum_t W_t(A_tX+b_t),
$$

whenever \(\sum_t W_t\) is nonsingular. This is still affine. Therefore residual incompatibility is not sufficient for curvature in any constant-weight first-order displacement fusion model.

### 7.2 Reduced Reprojection Model

Actual reprojection-based fusion is not usually in the constant-coefficient class. After linearization, a local displacement channel has the form

$$
E_{\mathrm{red}}[u]
=
\int
\sum_t
(u(s)-u_t(s))^T
W_t(s)
(u(s)-u_t(s))
ds,
$$

where

$$
W_t(s)=J_t(s)^TJ_t(s)
$$

is the reduced reprojection, Schur, confidence, and visibility weight seen by that surface coordinate. Even with constant feature confidence, \(J_t(s)\) varies with depth, view angle, projection scale, occlusion, and surface parameterization. This is the non-affine response operator that was missing from the unconditional bending claim.

Write

$$
u_t(s)=b_t+a_ts+O(s^2),
$$

and

$$
W_t(s)
=
W_{t0}+W_{t1}s+\frac{1}{2}W_{t2}s^2+O(s^3).
$$

Define

$$
M_k=\sum_t W_{tk},
\qquad
q_0=\sum_t W_{t0}b_t,
$$

$$
q_1
=
\sum_t
(W_{t0}a_t+W_{t1}b_t),
$$

and

$$
q_2
=
\sum_t
\left(
W_{t1}a_t+\frac{1}{2}W_{t2}b_t
\right).
$$

The pointwise minimizer satisfies

$$
u^\star(s)
=
M(s)^{-1}q(s)
=
c_0+c_1s+c_2s^2+O(s^3),
$$

with

$$
c_0=M_0^{-1}q_0,
$$

$$
c_1=M_0^{-1}(q_1-M_1c_0),
$$

and

$$
c_2
=
M_0^{-1}
\left(
q_2-M_1c_1-\frac{1}{2}M_2c_0
\right).
$$

The bending channel is

$$
2c_2.
$$

If \(W_{t1}=W_{t2}=0\) for all views, then \(c_2=0\) and the response is affine. If the spatial variation of the reduced weights is not aligned with the affine requests, \(c_2\) is generically nonzero. Algebraically, \(c_2\) is an analytic function of

$$
\{W_{t0},W_{t1},W_{t2},a_t,b_t\}
$$

that is not identically zero; its zero set is lower-dimensional unless additional symmetries force it.

### 7.3 Scalar Visibility and Confidence Model

The scalar visibility/confidence model is the commutative special case of the matrix formula. Let

$$
f_t(s)=a_ts+b_t+O(s^2),
$$

and

$$
w_t(s)
=
w_{t0}
\left(
1+\eta_ts+\frac{1}{2}\kappa_ts^2+O(s^3)
\right),
$$

with \(w_{t0}>0\). Let

$$
\pi_t=\frac{w_{t0}}{\sum_j w_{j0}},
\qquad
\langle g\rangle=\sum_t\pi_tg_t.
$$

The pointwise minimizer is

$$
\zeta^\star(s)
=
\frac{\sum_t w_t(s)f_t(s)}
{\sum_t w_t(s)}
=
c_0+c_1s+c_2s^2+O(s^3),
$$

where

$$
c_0=\langle b\rangle,
$$

$$
c_1=\langle a\rangle+\operatorname{Cov}(b,\eta),
$$

and

$$
c_2
=
\operatorname{Cov}(a,\eta)
+
\frac{1}{2}\operatorname{Cov}(b,\kappa)
-
\langle\eta\rangle\operatorname{Cov}(b,\eta).
$$

Therefore

$$
\zeta^{\star\prime\prime}(0)=2c_2.
$$

Bending appears only when spatial variation in visibility or confidence is correlated with incompatible affine slopes or offsets.

#### Covariance Interpretation

The leading term in the local scalar model is

$$
2\operatorname{Cov}(a,\eta).
$$

Thus, to second order in this reduced model, bending is not caused by view incompatibility alone. It is caused by the covariance between:

- the shear slope requested by each view, \(a_t\);
- the local spatial gradient of that view's fusion weight, visibility, or confidence, \(\eta_t\).

Equivalently, curvature appears when the views that dominate the reconstruction change across the surface patch, and those changing views request different wrong-intrinsic shear slopes. If all views keep constant relative weights, the incompatible shear requests average to another affine field. If weights vary spatially but all views request the same slope, the result is also affine. The bending channel opens when slope disagreement and spatial dominance variation are coupled.

The full scalar second-order response is

$$
\zeta^{\star\prime\prime}(0)
=
2\operatorname{Cov}(a,\eta)
+
\operatorname{Cov}(b,\kappa)
-
2\langle\eta\rangle\operatorname{Cov}(b,\eta).
$$

Therefore the covariance term \(2\operatorname{Cov}(a,\eta)\) is the cleanest first-order visibility-gradient mechanism, while offset variation, second-order weight changes, boundaries, robust kernels, and smoothness operators contribute additional curvature channels.

### 7.4 Smoothness and Euler-Lagrange Response

If one adds a thin-plate or depth smoothness term,

$$
E_\lambda[\zeta]
=
\int
\sum_t
w_t(s)(\zeta(s)-f_t(s))^2
ds
+
\lambda
\int
|\zeta''(s)|^2ds,
$$

then the Euler-Lagrange equation is

$$
\lambda\zeta''''(s)+W(s)\zeta(s)=R(s),
$$

where

$$
W(s)=\sum_t w_t(s),
\qquad
R(s)=\sum_t w_t(s)f_t(s).
$$

For constant \(W\) and affine \(R\), the interior solution is affine, up to boundary-layer effects. Smoothness does not by itself create bending from affine requests. It propagates bending created by non-affine forcing, spatially varying weights, nonlinear projection operators, boundaries, occlusion, or regularization constraints.

### 7.5 Manhattan-Normal Residual Distribution

For one view, a local affine displacement

$$
X\mapsto X+u_t(X)
$$

with gradient \(A_t\) sends normals to

$$
N_t
=
(I-A_t^T)N^*
+O(\|A_t\|^2).
$$

With a strong Manhattan prior, the pointwise normal-frame estimate solves

$$
N^\star(s)
=
\operatorname{polar}(M(s)),
$$

where

$$
M(s)
=
\frac{\sum_t w_t(s)N_t}
{\sum_t w_t(s)}.
$$

To first order,

$$
N^\star(s)
=
(I+[\omega_{\mathrm{MW}}(s)]_\times)N^*,
$$

where

$$
[\omega_{\mathrm{MW}}(s)]_\times
=
\operatorname{skew}(-\bar A_w(s)^T),
$$

and

$$
\bar A_w(s)
=
\frac{\sum_t w_t(s)A_t}
{\sum_t w_t(s)}.
$$

A constant weighted average \(\bar A_w\) gives only a constant Manhattan-frame rotation. Curvature of a plane requires a spatially varying normal:

$$
\partial_sN^\star(0)
=
[\omega_{\mathrm{MW}}'(0)]_\times N^*.
$$

If

$$
w_t(s)=w_{t0}(1+\eta_ts+O(s^2)),
$$

then

$$
\omega_{\mathrm{MW}}'(0)
=
\sum_t
\pi_t
(\eta_t-\langle\eta\rangle)
\omega_{\mathrm{MW},t}.
$$

Thus, in the Manhattan regime, bending is controlled by the covariance between spatial weight gradients and per-view shear-induced Manhattan rotations. If all views have constant relative weight, the normal frame is merely rotated and no plane curvature is created.

### 7.6 Single-View Depth Transport

For a single view with principal-point shift

$$
\beta=(\beta_x,\beta_y),
$$

the wrong-ray depth field can be written locally as coordinate transport:

$$
Z_\beta(u,v)
=
Z^*(u-\beta_x,v-\beta_y).
$$

Thus

$$
\delta Z_{\text{single}}(u,v)
=
-\beta\cdot\nabla Z^*(u,v)
+O(\|\beta\|^2).
$$

This is a first-order transport term, not intrinsic curvature. The depth error moves along the gradient; it does not yet bend the surface intrinsically.

## 8. Intrinsic-Aware Foundation-Prior Bundle Adjustment

The geometric principle is now:

$$
\text{wrong intrinsics}
\Rightarrow
\text{viewwise affine shear requests } u_t(X)=A_tX+b_t.
$$

Unconstrained reconstruction can keep this as a projective or affine gauge change. Bundle adjustment with fixed Euclidean cameras projects part of it onto \(\mathfrak{se}(3)\), chiefly yaw and pitch for principal-point drift, and marginalizes the rest into structure. But the spatial distribution of the remaining residual is not determined by incompatibility alone. Constant-weight affine fusion gives an affine average and no bend. Bending requires a non-affine response operator: spatially varying visibility or confidence, nonlinear reprojection weights, occlusion boundaries, smoothness with non-affine forcing, or Manhattan-frame weights that vary across the patch.

This motivates a geometry-aware prior policy for intrinsic-aware foundation-prior bundle adjustment. Calibration-first canonicalization is the safest special case, but the broader requirement is that learned priors must be corrected, gated, or delayed according to the expected wrong-intrinsic response.

### 8.1 State Variables and Inputs

The optimization state is

$$
\Theta =
\left(
K,
\{T_t\}_{t=1}^{M},
\{X_j\}_{j=1}^{N},
\{Z_t\},
\{N_t\},
\mathcal S
\right),
$$

where

$$
K=(f_x,f_y,c_x,c_y)
$$

is shared by all frames, \(T_t\in SE(3)\) are camera poses, \(X_j\) are sparse or semi-dense 3D points, \(Z_t\) are optional depth maps, \(N_t\) are optional normal maps or Manhattan frames, and \(\mathcal S\) contains nuisance variables such as depth scale-shift, robust switches, confidence calibration, or per-sequence crop parameters. Distortion can be added later, but the core method first solves the non-centered pinhole problem.

The inputs are:

- image observations \(I_t\);
- sparse or dense correspondences \(x_{tj}\leftrightarrow x_{sj}\);
- track confidences and visibility masks;
- foundation-model predictions \(\widehat T_t,\widehat Z_t,\widehat P_t,\widehat N_t\);
- model confidences \(\widehat C_t\), if available;
- optional weak metadata priors on focal length, field of view, crop, or image resolution.

The learned predictions are not inserted directly. They are treated as outputs of a model evaluated under an assumed camera \(\widehat K\), typically centered-principal-point. The current calibration residual is

$$
\Delta K = K-\widehat K,
$$

or, equivalently, the normalized mismatch

$$
H=\widehat K^{-1}K.
$$

### 8.2 Full Objective

The proposed objective is

$$
\min_{\Theta}
E_{\mathrm{corr}}
+
\lambda_{\mathrm{repr}}E_{\mathrm{repr}}
+
\lambda_{\mathrm{FM}}E_{\mathrm{FM}}^{\mathrm{corr}}
+
\lambda_K E_K
+
\lambda_{\mathrm{reg}}E_{\mathrm{reg}}.
$$

The correspondence term can be written in ray form:

$$
E_{\mathrm{corr}}
=
\sum_{(s,t),i}
\rho_{\mathrm{epi}}
\left(
r_s^i(K)^T
[t_{st}]_\times
R_{st}
r_t^i(K)
\right),
$$

where

$$
r_t^i(K)=K^{-1}\tilde x_t^i.
$$

When triangulated points are available, the reprojection term is

$$
E_{\mathrm{repr}}
=
\sum_{t,j}
\rho_{\mathrm{repr}}
\left(
\pi_K(T_tX_j)-x_{tj}
\right).
$$

The intrinsic prior is weak and explicit:

$$
E_K
=
\rho_f(f_x,f_y)
+
\rho_c(c_x,c_y)
+
\rho_{\mathrm{shared}}(K).
$$

Here \(\rho_f\) can encode square-pixel or focal-range priors, while \(\rho_c\) prevents principal point from drifting in poorly constrained sequences. The shared-camera term ties all frames to one physical pinhole camera unless the data explicitly contains per-frame digital stabilization or crop.

The regularizer \(E_{\mathrm{reg}}\) may include depth smoothness, normal consistency, temporal depth consistency, or Manhattan orthogonality:

$$
E_{\mathrm{reg}}
=
E_{\mathrm{smooth}}
+
E_{\mathrm{normal}}
+
E_{\mathrm{MW}}
+
E_{\mathrm{switch}}.
$$

The distinctive term is \(E_{\mathrm{FM}}^{\mathrm{corr}}\): learned priors enter only after correction, marginalization, or gating by the wrong-intrinsic response.

### 8.3 Prior States

A foundation-model prior can be in one of three states.

**Corrected prior.** If \(K^*\) is known or has been estimated, select a centered target camera \(K_\star\), remap

$$
I_\star(q_\star)
=
I(K^*K_\star^{-1}q_\star).
$$

Then run the foundation model on \(I_\star\). The resulting learned predictions are the safest priors because the input satisfies the model's centered-principal-point convention.

**Debiased prior.** If the model has already been run on the original non-centered image, its outputs should be interpreted as wrong-intrinsic responses. The formulas above predict dominant bias channels: yaw/pitch pose bias, depth transport, plane-normal shear, and possible curvature through non-affine reduced weights. These predictions can be used to subtract or marginalize expected bias before the prior enters \(E_{\mathrm{FM}}\).

**Gated prior.** If \(K^*\) is too uncertain, the learned output should enter only through robust, low-weight, or switchable residuals. Priors whose residuals align with the predicted wrong-intrinsic response should be down-weighted rather than treated as independent evidence.

### 8.4 Estimating \(K^*\) Before Trusting Priors

If \(K^*\) is unknown, the remap cannot be performed directly. The calibration estimate must come first.

Possible solvers include:

- \(F\)-based self-calibration from 2D correspondences, using \(E(K)=K^TFK\) and essential-matrix constraints.
- PnP with unknown principal point when 2D-3D correspondences exist.
- Minimal absolute-pose solvers such as P4.5Pfuv and P5Pfuva.
- Non-minimal solvers with unknown principal point and radial distortion, such as P7Pfruv.
- Shared-\(K\) bundle adjustment, where principal point is refined only after the reconstruction is sufficiently constrained.

The safest practical stack is:

1. Estimate initial \(K\) using 2D-2D or 2D-3D geometry.
2. Keep \(c_x,c_y\) fixed or strongly regularized while pose and structure stabilize.
3. Refine shared \(c_x,c_y\) in global BA.
4. Canonicalize all frames to centered pinhole.
5. Run the feed-forward 3D model.
6. Only after canonicalization should model pose/depth enter as weak, robust, gated priors.

### 8.5 Bias-Aware Learned Prior

The naive learned-prior term is

$$
E_{\mathrm{FM}}
=
\sum_t
\rho_T(T_t,\widehat T_t)
+
\rho_Z(Z_t,\widehat Z_t)
+
\rho_N(N_t,\widehat N_t).
$$

The bias-aware version should instead use

$$
E_{\mathrm{FM}}^{\mathrm{corr}}
=
\sum_t
w_t^{\mathrm{FM}}
\rho_T(T_t,\mathcal C_T(\widehat T_t,\Delta K))
+
w_t^{Z}
\rho_Z(Z_t,\mathcal C_Z(\widehat Z_t,\Delta K))
+
w_t^{N}
\rho_N(N_t,\mathcal C_N(\widehat N_t,\Delta K)),
$$

where \(\mathcal C_T,\mathcal C_Z,\mathcal C_N\) are correction or marginalization maps derived from the geometric response model. At minimum, the prior weights \(w_t^{\mathrm{FM}},w_t^Z,w_t^N\) should decrease when \(\Delta K\) is large or when the predicted residual matches a known wrong-intrinsic bias channel.

The pose correction is based on the local BA response:

$$
\delta\xi_t^{K}
=
-[e_t^TS^{-1}\Gamma_K]\Delta\theta_K.
$$

If the model pose was produced under \(\widehat K\), then a first-order debiased pose prior can be written as

$$
\mathcal C_T(\widehat T_t,\Delta K)
=
\exp(-(\delta\xi_t^K)^\wedge)\widehat T_t.
$$

The sign convention depends on whether \(\widehat T_t\) is interpreted as the biased output or as the reference state; the implementation should verify the sign by synthetic principal-point sweeps.

For depth, the single-view transport approximation gives

$$
\mathcal C_Z(\widehat Z_t,\Delta K)(u,v)
\approx
\widehat Z_t(u,v)
+
\beta\cdot\nabla\widehat Z_t(u,v),
$$

when \(\widehat Z_t\) is the centered-camera response and the target is the true-camera depth. If the learned depth is affine-invariant or only relative, this correction should be combined with per-frame scale and shift:

$$
Z_t(u)
\approx
a_t\mathcal C_Z(\widehat Z_t,\Delta K)(u)+b_t.
$$

For normals, the pinhole shear model gives

$$
\tilde n = H^{-T}n.
$$

Thus a normal predicted under the wrong intrinsic model can be moved toward the true-intrinsic frame by the inverse local transform, followed by renormalization:

$$
\mathcal C_N(\widehat n,\Delta K)
=
\frac{H^{T}\widehat n}{\|H^{T}\widehat n\|},
$$

with an additional Manhattan-frame projection if a strong orthogonality prior is used.

### 8.6 Robust Gating and Switchable Priors

Even after first-order correction, learned priors should remain weak. We use switchable weights

$$
\gamma_{t}^{T},\gamma_t^Z,\gamma_t^N\in[0,1]
$$

and optimize

$$
E_{\mathrm{FM}}^{\mathrm{gated}}
=
\sum_t
\gamma_t^T
\rho_T(T_t,\mathcal C_T(\widehat T_t,\Delta K))
+
\gamma_t^Z
\rho_Z(Z_t,\mathcal C_Z(\widehat Z_t,\Delta K))
+
\gamma_t^N
\rho_N(N_t,\mathcal C_N(\widehat N_t,\Delta K))
+
\lambda_\gamma E_\gamma.
$$

The gate prior \(E_\gamma\) discourages trivially turning off every learned residual while allowing rejection when the learned output agrees more with a known wrong-intrinsic bias channel than with multiview geometry. A simple deterministic gate is

$$
\gamma_t
=
\sigma
\left(
\tau
-
\eta_K\|\Delta K\|
-
\eta_r R_t^{\mathrm{geom}}
\right),
$$

where \(R_t^{\mathrm{geom}}\) measures disagreement between the learned prior and correspondence geometry after applying the predicted correction. A learned confidence map can modulate \(\gamma_t\), but it should not replace the geometric gate.

### 8.7 Optimization Schedule

The full method uses a staged schedule:

1. **Geometry bootstrap.** Estimate an initial shared \(K\), poses, and sparse structure from correspondences. Keep \(c_x,c_y\) fixed or strongly regularized until the sequence is sufficiently constrained.
2. **Principal-point release.** Refine shared \(c_x,c_y\) only after pose and structure stabilize. Monitor conditioning or posterior covariance for principal point.
3. **Prior classification.** Classify foundation-model predictions as corrected, debiased, or gated according to the current uncertainty in \(K\).
4. **Bias-aware joint solve.** Optimize correspondences, reprojection, shared intrinsics, and corrected learned priors jointly.
5. **Canonical rerun when possible.** If the estimated \(K\) becomes reliable, canonicalize frames to centered pinhole and rerun the foundation model. Replace debiased raw-image priors with corrected-input priors.
6. **Final robust refinement.** Re-optimize all active terms with switchable priors and robust losses, keeping the shared camera model explicit.

## 9. Experimental Program

The derivations above suggest four validation stages.

### 9.1 Synthetic Pure Geometry

Generate known 3D geometry and sweep principal-point offset:

$$
\Delta c_x,\Delta c_y.
$$

Measure:

- rigid Procrustes yaw/pitch response,
- translation-rotation soft modes,
- focal bias,
- plane-normal tilt,
- constant-depth degeneracy.

### 9.2 Solver Benchmark

Compare:

- centered-principal-point PnP,
- DLT,
- P4.5Pfuv,
- P5Pfuva,
- P7Pfruv,
- \(F\)-based self-calibration,
- shared-\(K\) BA.

Metrics:

$$
\frac{|f-\widehat f|}{f},
\qquad
\|c-\widehat c\|,
\qquad
\text{rotation error},
\qquad
\text{translation direction error},
\qquad
\text{reprojection error}.
$$

### 9.3 Residual Distribution Test

Construct multiview affine shear requests \(A_tX+b_t\). Compare:

- constant-weight fusion, which should remain affine;
- spatially varying weight fusion, whose curvature should match \(2c_2\);
- Manhattan-normal fusion, whose normal variation should match \(\omega_{\mathrm{MW}}'(0)\).

This test prevents the problem from silently degenerating into a constant-weight affine average.

### 9.4 Foundation Model Response

Finally, feed synthetic off-center images into the foundation model and compare its predicted pose, depth, and point maps to the pure geometric response:

$$
g_{\mathrm{model}}
\quad\text{versus}\quad
g_{\mathrm{geometry}}.
$$

The model's output should not be treated as a strong prior until this comparison is understood.

### 9.5 Learned-Prior Ablation

Run the full joint objective with four prior policies:

- no foundation-model prior;
- naive prior on raw off-center model outputs;
- gated prior with weights reduced according to \(\|\Delta K\|\) and geometric residual alignment;
- corrected prior after canonicalization or response-based debiasing.

The expected failure mode is that the naive prior over-constrains the solution toward wrong-\(K\)-conditioned pose and depth. The expected success mode is that corrected or gated priors improve conditioning without locking the optimizer into the centered-principal-point bias.

## 10. Discussion

The key difficulty is not that non-centered principal points are hard to warp once \(K\) is known. The key difficulty is that \(K\) is often unknown, and the wrong \(K\) can be absorbed by pose, focal length, structure, and normal-frame regularization in different ways depending on the reconstruction energy.

The most important negative result is that affine incompatibility alone is not a curvature theorem. It creates residual, but the residual must be spatially distributed by a non-affine operator before bending appears. This matters for computer graphics pipelines because many reconstruction artifacts that look like curved walls or warped floors are not caused by the principal-point shear alone; they are produced by the interaction of that shear with view-dependent weights, visibility, occlusion, confidence, priors, and surface regularization.

## 11. Conclusion

Mis-specified pinhole intrinsics inject an affine perturbation

$$
H-I
$$

into the reconstruction problem. Principal-point error is a rank-one shear, not merely an image translation. Single-view reconstruction can absorb it exactly by reassigning depth. Generic multiview reconstruction cannot absorb all viewwise shears into one Euclidean structure. Bundle adjustment projects part of the shear into \(SE(3)\), chiefly yaw and pitch, and the remaining response depends on the reduced fusion operator. Manhattan priors project sheared normal frames back to \(SO(3)\), producing rotations in a different metric. Curvature is not automatic; it appears only when residual incompatibility is spatially distributed by non-affine weights, visibility, confidence, or regularization.

This analysis supports a geometry-aware learned-prior policy. The safest implementation is calibration-first: estimate the true pinhole \(K\), canonicalize images to centered principal point, then run centered-camera feed-forward 3D models. When this is not possible, learned pose, depth, point, and normal predictions should enter joint BA only through correction maps, robust gates, or low-confidence priors informed by the wrong-intrinsic response derived here.
