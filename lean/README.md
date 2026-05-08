# Lean check for principal-point compensation

This Lean project formalizes the small first-order least-squares model behind
the apparent tension between:

1. the reference paper's qualitative statement that small principal-point
   offsets can be partly absorbed by translation, and
2. the manuscript's statement that, after point elimination in a nondegenerate
   near-axis BA model, the zero first-order response is yaw/pitch rather than
   lateral translation.

The formalized x-channel residual is

```text
beta + t * a - omega,   where a = 1 / Z.
```

The y-channel is analogous up to the sign convention.

Main theorems live in `PinholeCalib/PrincipalPoint.lean`:

- `baObjective2_eq_zero_iff`: with two distinct inverse depths, zero BA
  residual forces `t = 0` and `omega = beta`.
- `constDepthObjective_eq_zero_iff`: at constant depth, there is a full
  zero-residual family `omega - t * a = beta`.
- `translation_only_exact_on_constant_depth`: if rotation is fixed to zero,
  translation alone exactly compensates constant depth.
- `translationOnly_complete_square`: if rotation is fixed to zero and depths
  differ, the best translation has residual proportional to `(a1 - a2)^2`.

So the two statements are compatible: translation compensation is exact in the
constant-depth/rotation-fixed model and approximate when depths are close, while
the free rotation+translation nondegenerate BA model selects pure rotation as
the unique zero-residual solution.

Additional modules extend the formalization:

- `PinholeCalib/SingleView.lean`: 3D affine-chart reprojection existence with
  arbitrary positive depth scaling.
- `PinholeCalib/JointChannels.lean`: coupled `(x, y)` objective and explicit
  roll-zero theorem in the diagonal near-axis model.
- `PinholeCalib/HxMatrix.lean`: explicit `2 x 2` Hessian block, determinant,
  variance, and finite-point rank criterion.
- `PinholeCalib/Normals.lean`: pure-shear normal transport and the skew-part
  `1 / 2` coefficient.
- `PinholeCalib/AffineFusion.lean`: affine fusion commutation and the discrete
  variable-weight finite-difference term.

Archival logs for this formalization pass are stored in `logs/`:

- `logs/build.log`: clean rebuild of the Lean package.
- `logs/*.inventory.log`: theorem/definition inventory for each new module.

I installed a local Elan toolchain in this directory under `.elan/`. To check
with that local toolchain:

```bash
cd /data/users/mia/current/pinhole_calib/lean
ELAN_HOME=$PWD/.elan PATH=$PWD/.elan/bin:$PATH lake build
```

、
关键结论：

- [baObjective2_eq_zero_iff](/data/users/mia/current/pinhole_calib/lean/PinholeCalib/PrincipalPoint.lean:67)：如果两个逆深度不同 `a1 ≠ a2`，零残差当且仅当 `t = 0 ∧ omega = beta`。这支持我们稿子里的“非退化消点 BA 模型中，主点误差一阶主要被 yaw/pitch 吸收，而不是横向平移”。

- [constDepthObjective_eq_zero_iff](/data/users/mia/current/pinhole_calib/lean/PinholeCalib/PrincipalPoint.lean:112)：恒定深度时，零残差是一整条族 `omega - t * a = beta`。所以令 `omega = 0` 时，平移可以精确补偿。

- [translationOnly_complete_square](/data/users/mia/current/pinhole_calib/lean/PinholeCalib/PrincipalPoint.lean:138)：如果把 rotation 固定为 0，只允许 translation 去吸收，那么最优平移后的剩余误差正比于 `(a1 - a2)^2`。也就是说深度越接近常数，translation-only 补偿越好。

结论：两边推导不矛盾，但必须把语境写清楚。ref 的说法对应“平移可部分补偿，尤其小偏移/近恒定深度/rotation 受限或被正则时”；我们稿子的说法对应“rotation 和 translation 都自由、逆深度非退化、消点后的近轴 BA 零残差解”。
