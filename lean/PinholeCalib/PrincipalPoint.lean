import Mathlib

/-!
# Principal point drift: two first-order compensation models

This file formalizes the small one-dimensional least-squares models behind the
apparent tension between two statements:

* a centered-principal-point absolute-pose solver can absorb a small principal
  point offset partly by translation;
* after point elimination in a nondegenerate near-axis BA model, the zero
  first-order compensation is rotational, not translational.

The point of the formalization is not to verify a full PnP polynomial solver.
It isolates the linearized residuals where the two explanations meet.
-/

namespace PinholeCalib

noncomputable section

abbrev R := Real

/- The normalized-image residual for the x-channel near the optical axis.

    beta + t / Z - omega

We write `a = 1 / Z`, so the residual is `beta + t * a - omega`.
The y-channel is the same calculation with the sign convention changed.
-/
def residual (beta a t omega : R) : R :=
  beta + t * a - omega

def baObjective2 (beta a1 a2 t omega : R) : R :=
  residual beta a1 t omega ^ 2 + residual beta a2 t omega ^ 2

def constDepthObjective (beta a t omega : R) : R :=
  residual beta a t omega ^ 2

def translationOnlyObjective2 (beta a1 a2 t : R) : R :=
  residual beta a1 t 0 ^ 2 + residual beta a2 t 0 ^ 2

def translationOnlyTstar (beta a1 a2 : R) : R :=
  -beta * (a1 + a2) / (a1 ^ 2 + a2 ^ 2)

private theorem two_sq_sum_eq_zero {x y : R}
    (h : x ^ 2 + y ^ 2 = 0) : x = 0 ∧ y = 0 := by
  have hx_nonneg : 0 <= x ^ 2 := sq_nonneg x
  have hy_nonneg : 0 <= y ^ 2 := sq_nonneg y
  have hx_sq : x ^ 2 = 0 := by nlinarith
  have hy_sq : y ^ 2 = 0 := by nlinarith
  exact ⟨sq_eq_zero_iff.mp hx_sq, sq_eq_zero_iff.mp hy_sq⟩

private theorem one_sq_eq_zero {x : R} (h : x ^ 2 = 0) : x = 0 := by
  exact sq_eq_zero_iff.mp h

/- In the nondegenerate two-depth BA residual model, zero residual forces the
translation increment to be zero and the yaw/pitch component to equal beta.

This is the formal version of the manuscript's statement:

    min_{t, omega} sum_i (beta + t / Z_i - omega)^2

has the exact zero-residual solution `omega = beta`, `t = 0` when the inverse
depths are not all equal.
-/
theorem baObjective2_eq_zero_iff {beta a1 a2 t omega : R}
    (hdepth : a1 ≠ a2) :
    baObjective2 beta a1 a2 t omega = 0 ↔ t = 0 ∧ omega = beta := by
  constructor
  · intro h
    unfold baObjective2 at h
    rcases two_sq_sum_eq_zero h with ⟨h1, h2⟩
    unfold residual at h1 h2
    have ht_mul : t * (a1 - a2) = 0 := by nlinarith
    have hdiff : a1 - a2 ≠ 0 := sub_ne_zero.mpr hdepth
    have ht : t = 0 := by
      rcases mul_eq_zero.mp ht_mul with ht | hbad
      · exact ht
      · exact False.elim (hdiff hbad)
    have h1' : beta - omega = 0 := by
      simpa [ht] using h1
    have homega : omega = beta := by linarith
    exact ⟨ht, homega⟩
  · intro h
    rcases h with ⟨rfl, rfl⟩
    simp [baObjective2, residual]

theorem baObjective2_solution_is_unique_minimizer {beta a1 a2 t omega : R}
    (hdepth : a1 ≠ a2)
    (hmin : baObjective2 beta a1 a2 t omega <=
      baObjective2 beta a1 a2 0 beta) :
    t = 0 ∧ omega = beta := by
  have hzero : baObjective2 beta a1 a2 0 beta = 0 := by
    simp [baObjective2, residual]
  have hle_zero : baObjective2 beta a1 a2 t omega <= 0 := by
    simpa [hzero] using hmin
  have hnonneg : 0 <= baObjective2 beta a1 a2 t omega := by
    unfold baObjective2
    nlinarith [sq_nonneg (residual beta a1 t omega),
      sq_nonneg (residual beta a2 t omega)]
  have hobj : baObjective2 beta a1 a2 t omega = 0 :=
    le_antisymm hle_zero hnonneg
  exact (baObjective2_eq_zero_iff hdepth).mp hobj

/- In the constant-depth model, there is a whole zero-residual affine line.
Equivalently, `omega - t * a = beta`.

This is the formal version of the manuscript's constant-depth degeneracy and
the place where a translation-only compensation becomes possible.
-/
theorem constDepthObjective_eq_zero_iff {beta a t omega : R} :
    constDepthObjective beta a t omega = 0 ↔
      omega - t * a = beta := by
  constructor
  · intro h
    unfold constDepthObjective at h
    have hres : residual beta a t omega = 0 := one_sq_eq_zero h
    unfold residual at hres
    linarith
  · intro h
    unfold constDepthObjective residual
    nlinarith

theorem translation_only_exact_on_constant_depth {beta a : R}
    (ha : a ≠ 0) :
    constDepthObjective beta a (-beta / a) 0 = 0 := by
  unfold constDepthObjective residual
  field_simp [ha]

/- If rotation is held fixed at zero, translation can still be the best
least-squares absorber. For two inverse depths, completing the square gives the
best translation and the irreducible residual.

The residual is proportional to `(a1 - a2)^2`, so translation is exact at
constant depth and only approximate when depths differ.
-/
theorem translationOnly_complete_square {beta a1 a2 t : R}
    (hden : a1 ^ 2 + a2 ^ 2 ≠ 0) :
    translationOnlyObjective2 beta a1 a2 t =
      (a1 ^ 2 + a2 ^ 2) *
          (t - translationOnlyTstar beta a1 a2) ^ 2
        + beta ^ 2 * (a1 - a2) ^ 2 / (a1 ^ 2 + a2 ^ 2) := by
  unfold translationOnlyObjective2 translationOnlyTstar residual
  field_simp [hden]
  ring

theorem translationOnly_min_value {beta a1 a2 : R}
    (hden : a1 ^ 2 + a2 ^ 2 ≠ 0) :
    translationOnlyObjective2 beta a1 a2
        (translationOnlyTstar beta a1 a2) =
      beta ^ 2 * (a1 - a2) ^ 2 / (a1 ^ 2 + a2 ^ 2) := by
  rw [translationOnly_complete_square (beta := beta) (a1 := a1)
    (a2 := a2) (t := translationOnlyTstar beta a1 a2) hden]
  simp

theorem translationOnlyTstar_is_minimizer {beta a1 a2 t : R}
    (hpos : 0 < a1 ^ 2 + a2 ^ 2) :
    translationOnlyObjective2 beta a1 a2
        (translationOnlyTstar beta a1 a2) <=
      translationOnlyObjective2 beta a1 a2 t := by
  have hden : a1 ^ 2 + a2 ^ 2 ≠ 0 := ne_of_gt hpos
  rw [translationOnly_complete_square (beta := beta) (a1 := a1)
      (a2 := a2) (t := translationOnlyTstar beta a1 a2) hden,
    translationOnly_complete_square (beta := beta) (a1 := a1)
      (a2 := a2) (t := t) hden]
  have hsq : 0 <=
      (a1 ^ 2 + a2 ^ 2) *
        (t - translationOnlyTstar beta a1 a2) ^ 2 := by
    exact mul_nonneg (le_of_lt hpos) (sq_nonneg _)
  nlinarith

/- A compact theorem statement of the non-contradiction:

* if translation and rotation are both free and inverse depths are not equal,
  the zero-residual solution is pure rotation in this linearized BA model;
* if rotation is fixed, translation is still the best available absorber, with
  residual controlled by inverse-depth variation.
-/
theorem no_contradiction_summary {beta a1 a2 t omega : R}
    (hdepth : a1 ≠ a2)
    (hba : baObjective2 beta a1 a2 t omega = 0) :
    t = 0 ∧ omega = beta := by
  exact (baObjective2_eq_zero_iff hdepth).mp hba

end

end PinholeCalib
