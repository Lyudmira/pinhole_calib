import Mathlib

open scoped BigOperators

namespace PinholeCalib

noncomputable section

abbrev R := Real
abbrev Vec3 := Fin 3 → R

def blend (w : R) (p q : Vec3) : Vec3 :=
  w • p + (1 - w) • q

theorem linearMap_sum_smul
    {ι : Type*}
    [Fintype ι]
    (L : Vec3 →ₗ[R] Vec3)
    (w : ι → R)
    (p : ι → Vec3) :
    L (∑ i, w i • p i) = ∑ i, w i • L (p i) := by
  simp

theorem linearMap_blend
    (L : Vec3 →ₗ[R] Vec3)
    (w : R)
    (p q : Vec3) :
    L (blend w p q) = blend w (L p) (L q) := by
  simp [blend]

theorem finiteDifference_blend
    (w0 w1 : R)
    (p0 p1 q0 q1 : Vec3) :
    blend w1 p1 q1 - blend w0 p0 q0 =
      w0 • (p1 - p0) +
      (1 - w0) • (q1 - q0) +
      (w1 - w0) • (p1 - q1) := by
  ext i
  simp [blend]
  ring

theorem finiteDifference_blend_after_linearMap
    (L : Vec3 →ₗ[R] Vec3)
    (w0 w1 : R)
    (p0 p1 q0 q1 : Vec3) :
    L (blend w1 p1 q1 - blend w0 p0 q0) =
      w0 • L (p1 - p0) +
      (1 - w0) • L (q1 - q0) +
      (w1 - w0) • L (p1 - q1) := by
  rw [finiteDifference_blend]
  simp

end

end PinholeCalib
