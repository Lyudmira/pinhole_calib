import PinholeCalib.PrincipalPoint

namespace PinholeCalib

noncomputable section

abbrev R := Real

def jointObjective2
    (βx βy a1 a2 tx ty ωx ωy : R) : R :=
  baObjective2 βx a1 a2 tx ωy + baObjective2 βy a1 a2 ty (-ωx)

def jointObjective2WithRoll
    (βx βy a1 a2 tx ty ωx ωy ωz : R) : R :=
  jointObjective2 βx βy a1 a2 tx ty ωx ωy + ωz ^ 2

private theorem baObjective2_nonneg (beta a1 a2 t omega : R) :
    0 <= baObjective2 beta a1 a2 t omega := by
  unfold baObjective2
  nlinarith [sq_nonneg (residual beta a1 t omega),
    sq_nonneg (residual beta a2 t omega)]

theorem jointObjective2_eq_zero_iff
    {βx βy a1 a2 tx ty ωx ωy : R}
    (hdepth : a1 ≠ a2) :
    jointObjective2 βx βy a1 a2 tx ty ωx ωy = 0 ↔
      tx = 0 ∧ ty = 0 ∧ ωy = βx ∧ ωx = -βy := by
  constructor
  · intro h
    unfold jointObjective2 at h
    have hx_nonneg : 0 <= baObjective2 βx a1 a2 tx ωy :=
      baObjective2_nonneg _ _ _ _ _
    have hy_nonneg : 0 <= baObjective2 βy a1 a2 ty (-ωx) :=
      baObjective2_nonneg _ _ _ _ _
    have hx_zero : baObjective2 βx a1 a2 tx ωy = 0 := by nlinarith
    have hy_zero : baObjective2 βy a1 a2 ty (-ωx) = 0 := by nlinarith
    rcases (baObjective2_eq_zero_iff hdepth).mp hx_zero with ⟨htx, hωy⟩
    rcases (baObjective2_eq_zero_iff hdepth).mp hy_zero with ⟨hty, hnegωx⟩
    have hωx : ωx = -βy := by linarith
    exact ⟨htx, hty, hωy, hωx⟩
  · rintro ⟨rfl, rfl, rfl, rfl⟩
    simp [jointObjective2, baObjective2, residual]

theorem jointObjective2WithRoll_eq_zero_iff
    {βx βy a1 a2 tx ty ωx ωy ωz : R}
    (hdepth : a1 ≠ a2) :
    jointObjective2WithRoll βx βy a1 a2 tx ty ωx ωy ωz = 0 ↔
      tx = 0 ∧ ty = 0 ∧ ωy = βx ∧ ωx = -βy ∧ ωz = 0 := by
  constructor
  · intro h
    unfold jointObjective2WithRoll at h
    have hjoint_nonneg : 0 <= jointObjective2 βx βy a1 a2 tx ty ωx ωy := by
      unfold jointObjective2
      exact add_nonneg
        (baObjective2_nonneg _ _ _ _ _)
        (baObjective2_nonneg _ _ _ _ _)
    have hωz_nonneg : 0 <= ωz ^ 2 := sq_nonneg ωz
    have hjoint_zero : jointObjective2 βx βy a1 a2 tx ty ωx ωy = 0 := by
      nlinarith
    have hωz_zero : ωz ^ 2 = 0 := by
      nlinarith
    rcases (jointObjective2_eq_zero_iff hdepth).mp hjoint_zero with
      ⟨htx, hty, hωy, hωx⟩
    exact ⟨htx, hty, hωy, hωx, sq_eq_zero_iff.mp hωz_zero⟩
  · rintro ⟨rfl, rfl, rfl, rfl, rfl⟩
    simp [jointObjective2WithRoll, jointObjective2, baObjective2, residual]

end

end PinholeCalib
