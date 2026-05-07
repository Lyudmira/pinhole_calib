
**我们需要的来自/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc的内容**
- 完整非中心 pinhole 相机：`fx, fy, cx, cy` 都是相机状态，projection matrix 也使用 `cx/cy` 偏移，而不是默认中心主点。见 [base.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/cameras/base.py:154)、[graphics_utils.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/utils/graphics_utils.py:118)。
- 可微内参优化：`VanillaDelta/RigDelta` 已经包含 `delta_focal_x/y, delta_cx/cy`，训练里也有 intrinsic optimizer。见 [delta.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/cameras/delta.py:72)、[intrinsics.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/cameras/optimizers/intrinsics.py:16)。
- shared-K 结构已经存在：vanilla 按 `camera_id` 共享内参，rig 按 `camera_name` 共享内参/相机到 rig 的外参。见 [vanilla.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/cameras/converters/vanilla.py:57)、[rig.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/cameras/converters/rig.py:211)。
- renderer 和几何函数都用完整 K：unproject/project/normal-from-depth 都走 `intrinsic_matrix`，PGSR 的 ray generation 明确用 `(u-cx)/fx`。见 [pgsr_renderer.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/renderers/gsplat_renderer/pgsr_renderer.py:60)、[projection.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/utils/projection.py:33)。
- COLMAP wrapper 已经打开 `BundleAdjustment.refine_principal_point 1`，说明工程上已经在尝试让主点被 BA refine。见 [run_colmap.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/toolbox/run_colmap.py:384)。
- 有已知 K/D 的 undistort/recenter 基础，测试里也把目标主点设到中心。见 [undistort_image.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/toolbox/undistort_image.py:242)、[test_undistortion.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/tests/test_undistortion.py:34)。

**最接近我们核心问题的原型**
- `contrib/ontheflynvs_dc/mini_ba.py` 有 Schur complement MiniBA：优化 pose、focal、3D points，带 Huber/MAD 和 LM damping。它还没有优化 `cx/cy`，但结构非常适合扩展。见 [mini_ba.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/contrib/ontheflynvs_dc/mini_ba.py:52)。
- `contrib/ontheflynvs_dc/Levenberg_Marquardt/ba_optimizer.py` 更近：它的参数字典已经包含 `focal_x, focal_y, cx, cy`，并对 pairwise triangulation/reprojection residual 做 jacobian 和 LM step。见 [ba_optimizer.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/contrib/ontheflynvs_dc/Levenberg_Marquardt/ba_optimizer.py:156)。
- `dc_reality/splatting/utils/moge/_utils.py` 有一个非常相关的小求解器：从 MoGe pointmap 拟合 `fx, fy, cx, cy` 和 depth shift `t`。这其实就是“foundation pointmap -> 反求相机内参”的雏形。见 [_utils.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/utils/moge/_utils.py:30)。
- `second_order_gradient` 和 `adapt_curvature` 已经在做 Hessian/曲率分析，参数列表也包含 `focal_x/y,cx,cy`。这对我们做“residual 如何在 pose/K/depth 间分配”的实证诊断很有用。见 [hessian_utils.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/contrib/fundamental/second_order_gradient/hessian_utils.py:21)。

**为了开始工程化我们的论文我们需要自己做的：还缺的关键层**
- 没有 Larsson 2018 的 P4.5Pfuv/P5Pfuva/P7Pfruv solver。
- 没有 PnPuv 那种“固定 pose/structure 时线性解 `cx,cy`”的 separable solver。
- 没有我们论文里要的 `g_geometry` / miscalibration response correction：即 foundation depth/pose/pointmap 在错误中心主点下如何被校正、降权或投到 compatible residual space。
- 现有 depth prior 只是对 rendered depth 和 target disparity 做 alignment/SSIM/normal consistency，并没有判断 target depth 是否因为错 K 而系统性弯曲。见 [depths.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/trainer/criteria/depths.py:60)。
- 当前 `cx/cy` 参数化是 `cx = _cx * exp(delta_cx)`，这对小扰动可用，但理论上不如 additive pixel offset 或 normalized offset 自然。见 [rig.py](/data/users/mia/current/pinhole_calib/from_dc/gaussian-splatting-dc/dc_reality/dc_reality/splatting/cameras/models/rig.py:114)。

**第一步**


1. 用现有 shared-K differentiable camera + COLMAP/rig BA 作为 bootstrap。
2. 把 `BAOptimizer`/`MiniBA` 改成正式的 shared `fx,fy,cx,cy` + pose + optional XYZ/depth 的 LM/Schur 求解层。
3. 加一个 separable `cx/cy` 或 `fx,cx/fy,cy` least-squares 子步骤，降低 ill-conditioning。
4. 在 depth/pose prior 进入 loss 前，加 miscalibration-aware correction/gating，而不是直接把 foundation output 当强 prior。

可以考虑的：重新按照/data/users/mia/current/pinhole_calib/ref/Camera Pose Estimation with Unknown Principal Point.md的内容实现他们用的求解器。
