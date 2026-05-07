import Lake
open Lake DSL

package "pinhole_calib" where
  moreLeanArgs := #["-DautoImplicit=false"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.14.0"

lean_lib PinholeCalib where
