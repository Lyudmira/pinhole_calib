import Mathlib

namespace PinholeCalib

noncomputable section

abbrev R := Real
abbrev Vec3 := Fin 3 → R

def pureShear (βx βy : R) : Matrix (Fin 3) (Fin 3) R :=
  !![1, 0, βx;
    0, 1, βy;
    0, 0, 1]

def pureShearInvT (βx βy : R) : Matrix (Fin 3) (Fin 3) R :=
  !![1, 0, 0;
    0, 1, 0;
    -βx, -βy, 1]

def shearError (βx βy : R) : Matrix (Fin 3) (Fin 3) R :=
  pureShear βx βy - 1

def skewPart (M : Matrix (Fin 3) (Fin 3) R) :
    Matrix (Fin 3) (Fin 3) R :=
  ((1 : R) / 2) • (M - M.transpose)

def planeValue (n p : Vec3) : R :=
  n 0 * p 0 + n 1 * p 1 + n 2 * p 2

theorem pureShear_plane_invariant (βx βy : R) (n p : Vec3) :
    planeValue ((pureShearInvT βx βy).mulVec n) ((pureShear βx βy).mulVec p) =
      planeValue n p := by
  simp [planeValue, pureShear, pureShearInvT, Matrix.mulVec, Fin.sum_univ_three]
  ring_nf

theorem shearError_skewPart (βx βy : R) :
    skewPart (shearError βx βy) =
      !![0, 0, βx / 2;
        0, 0, βy / 2;
        -βx / 2, -βy / 2, 0] := by
  ext i j <;> fin_cases i <;> fin_cases j <;>
    simp [skewPart, shearError, pureShear]
  all_goals ring_nf

end

end PinholeCalib
