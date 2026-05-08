import Mathlib

namespace PinholeCalib

noncomputable section

abbrev R := Real
abbrev Vec2 := Fin 2 → R
abbrev Vec3 := Fin 3 → R

def proj (p : Vec3) : Vec2 :=
  ![p 0 / p 2, p 1 / p 2]

def mismatchMatrix (αx αy βx βy : R) : Matrix (Fin 3) (Fin 3) R :=
  !![αx, 0, βx;
    0, αy, βy;
    0, 0, 1]

def normalizedMismatchRay (αx αy βx βy : R) (p : Vec3) : Vec3 :=
  (1 / p 2) • (mismatchMatrix αx αy βx βy).mulVec p

theorem mismatchMatrix_mulVec_third (αx αy βx βy : R) (p : Vec3) :
    (mismatchMatrix αx αy βx βy).mulVec p 2 = p 2 := by
  simp [mismatchMatrix, Matrix.mulVec, Fin.sum_univ_three]
  ring_nf

theorem proj_smul {ρ : R} (hρ : ρ ≠ 0) (p : Vec3) :
    proj (ρ • p) = proj p := by
  ext i <;> fin_cases i
  · unfold proj
    field_simp [hρ]
    ring_nf
  · unfold proj
    field_simp [hρ]
    ring_nf

theorem proj_normalizedMismatchRay (αx αy βx βy : R) {p : Vec3}
    (hZ : p 2 ≠ 0) :
    proj (normalizedMismatchRay αx αy βx βy p) =
      proj ((mismatchMatrix αx αy βx βy).mulVec p) := by
  unfold normalizedMismatchRay
  apply proj_smul
  exact one_div_ne_zero hZ

theorem reprojection_exists_for_any_positive_depth
    {World : Type*}
    (cameraOfWorld : World → Vec3)
    (worldOfCamera : Vec3 → World)
    (hcam : ∀ q, cameraOfWorld (worldOfCamera q) = q)
    (αx αy βx βy : R)
    (p : Vec3)
    {ρ : R}
    (hρ : 0 < ρ)
    (hZ : p 2 ≠ 0) :
    let Xhat := worldOfCamera (ρ • normalizedMismatchRay αx αy βx βy p)
    proj (cameraOfWorld Xhat) =
      proj ((mismatchMatrix αx αy βx βy).mulVec p) := by
  dsimp
  rw [hcam]
  exact (proj_smul (ne_of_gt hρ) _).trans (proj_normalizedMismatchRay _ _ _ _ hZ)

end

end PinholeCalib
