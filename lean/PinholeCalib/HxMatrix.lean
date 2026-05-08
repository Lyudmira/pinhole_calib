import Mathlib

open scoped BigOperators Matrix

namespace PinholeCalib

noncomputable section

abbrev R := Real

def sampleCount (n : Nat) : R :=
  (n + 1 : R)

def mean {n : Nat} (a : Fin (n + 1) → R) : R :=
  (∑ i, a i) / sampleCount n

def secondMoment {n : Nat} (a : Fin (n + 1) → R) : R :=
  (∑ i, a i ^ 2) / sampleCount n

def variance {n : Nat} (a : Fin (n + 1) → R) : R :=
  secondMoment a - mean a ^ 2

def centeredSquareSum {n : Nat} (a : Fin (n + 1) → R) : R :=
  ∑ i, (a i - mean a) ^ 2

def HxMatrix {n : Nat} (fx : R) (a : Fin (n + 1) → R) :
    Matrix (Fin 2) (Fin 2) R :=
  let nR := sampleCount n
  !![nR * fx ^ 2 * secondMoment a, nR * fx ^ 2 * mean a;
    nR * fx ^ 2 * mean a, nR * fx ^ 2]

theorem sampleCount_ne_zero (n : Nat) : sampleCount n ≠ 0 := by
  unfold sampleCount
  positivity

theorem sampleCount_pos (n : Nat) : 0 < sampleCount n := by
  unfold sampleCount
  positivity

theorem sum_eq_sampleCount_mul_mean {n : Nat} (a : Fin (n + 1) → R) :
    ∑ i, a i = sampleCount n * mean a := by
  unfold mean
  field_simp [sampleCount_ne_zero n]

theorem sum_sq_eq_sampleCount_mul_secondMoment {n : Nat} (a : Fin (n + 1) → R) :
    ∑ i, a i ^ 2 = sampleCount n * secondMoment a := by
  unfold secondMoment
  field_simp [sampleCount_ne_zero n]

theorem sampleCount_mul_variance_eq_centeredSquareSum {n : Nat}
    (a : Fin (n + 1) → R) :
    sampleCount n * variance a = centeredSquareSum a := by
  unfold variance centeredSquareSum mean secondMoment sampleCount
  field_simp
  ring_nf

theorem variance_eq_centeredSquareSum_div_sampleCount {n : Nat}
    (a : Fin (n + 1) → R) :
    variance a = centeredSquareSum a / sampleCount n := by
  have hmain := sampleCount_mul_variance_eq_centeredSquareSum a
  have hN : sampleCount n ≠ 0 := sampleCount_ne_zero n
  apply (eq_div_iff hN).2
  simpa [mul_comm] using hmain

theorem variance_nonneg {n : Nat} (a : Fin (n + 1) → R) :
    0 <= variance a := by
  rw [variance_eq_centeredSquareSum_div_sampleCount]
  exact div_nonneg
    (Finset.sum_nonneg fun i _ => sq_nonneg (a i - mean a))
    (le_of_lt (sampleCount_pos n))

theorem centeredSquareSum_eq_zero_iff_all_equal {n : Nat}
    (a : Fin (n + 1) → R) :
    centeredSquareSum a = 0 ↔ ∀ i j, a i = a j := by
  constructor
  · intro hzero i j
    have hterms :=
      (Finset.sum_eq_zero_iff_of_nonneg
        (fun k _ => sq_nonneg (a k - mean a))).mp hzero
    have hi_sq : (a i - mean a) ^ 2 = 0 := hterms i (Finset.mem_univ i)
    have hj_sq : (a j - mean a) ^ 2 = 0 := hterms j (Finset.mem_univ j)
    have hi : a i = mean a := by
      have : a i - mean a = 0 := sq_eq_zero_iff.mp hi_sq
      linarith
    have hj : a j = mean a := by
      have : a j - mean a = 0 := sq_eq_zero_iff.mp hj_sq
      linarith
    linarith
  · intro hall
    have hconst : ∀ i, a i = a 0 := fun i => hall i 0
    have hmean : mean a = a 0 := by
      unfold mean
      rw [show (∑ i, a i) = ∑ _i : Fin (n + 1), a 0 by
        apply Finset.sum_congr rfl
        intro i hi
        exact hconst i]
      field_simp [sampleCount, sampleCount_ne_zero n]
    apply Finset.sum_eq_zero
    intro i hi
    apply sq_eq_zero_iff.mpr
    rw [hconst i, hmean]
    ring

theorem variance_eq_zero_iff_all_equal {n : Nat}
    (a : Fin (n + 1) → R) :
    variance a = 0 ↔ ∀ i j, a i = a j := by
  constructor
  · intro hzero
    apply (centeredSquareSum_eq_zero_iff_all_equal a).mp
    rw [← sampleCount_mul_variance_eq_centeredSquareSum a, hzero]
    ring
  · intro hall
    rw [variance_eq_centeredSquareSum_div_sampleCount,
      (centeredSquareSum_eq_zero_iff_all_equal a).2 hall]
    simp

theorem variance_pos_iff_not_all_equal {n : Nat}
    (a : Fin (n + 1) → R) :
    0 < variance a ↔ ¬ ∀ i j, a i = a j := by
  have hnonneg : 0 <= variance a := variance_nonneg a
  constructor
  · intro hpos hall
    have : variance a = 0 := (variance_eq_zero_iff_all_equal a).2 hall
    linarith
  · intro hnot
    have hne : variance a ≠ 0 := by
      intro hzero
      exact hnot ((variance_eq_zero_iff_all_equal a).1 hzero)
    exact lt_of_le_of_ne (variance_nonneg a) (by simpa using hne.symm)

theorem det_HxMatrix {n : Nat} (fx : R) (a : Fin (n + 1) → R) :
    (HxMatrix fx a).det =
      (sampleCount n * fx ^ 2) ^ 2 * variance a := by
  have hN : sampleCount n ≠ 0 := sampleCount_ne_zero n
  unfold HxMatrix variance mean secondMoment
  simp [Matrix.det_fin_two]
  field_simp [hN]
  ring

theorem det_HxMatrix_ne_zero_iff {n : Nat} {fx : R} (a : Fin (n + 1) → R) :
    (HxMatrix fx a).det ≠ 0 ↔
      fx ≠ 0 ∧ ¬ ∀ i j, a i = a j := by
  rw [det_HxMatrix]
  have hN : sampleCount n ≠ 0 := sampleCount_ne_zero n
  have hscale : (sampleCount n * fx ^ 2) ^ 2 ≠ 0 ↔ fx ≠ 0 := by
    constructor
    · intro hpow hfx
      apply hpow
      simp [hfx]
    · intro hfx
      exact pow_ne_zero 2 (mul_ne_zero hN (pow_ne_zero 2 hfx))
  rw [mul_ne_zero_iff, hscale]
  constructor
  · rintro ⟨hfx, hvar⟩
    refine ⟨hfx, ?_⟩
    intro hall
    exact hvar ((variance_eq_zero_iff_all_equal a).2 hall)
  · rintro ⟨hfx, hneq⟩
    refine ⟨hfx, ?_⟩
    intro hzero
    exact hneq ((variance_eq_zero_iff_all_equal a).1 hzero)

end

end PinholeCalib
