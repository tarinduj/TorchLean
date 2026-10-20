import NN.IR.Semantics
import NN.Spec.Core.Context.Real
import Mathlib.Data.Fin.VecNotation

/-!
Loops over fixed-size tiles, reads tiles of input tensors, and writes tiles of an output tensor.
A tile's shape is its block-size list; loads and stores are masked by the loops' bounds, with one
bound and one tile index per block size.
-/

open Spec TorchLean

namespace TorchLean.Helion

variable {α : Type} [TorchLean.Storage α] {r : ℕ}

/-- For `B > 0`, the number of blocks of `B` positions needed to cover `n` positions.
For example, `n = 7` and `B = 4` gives two blocks. -/
def numTiles (n B : ℕ) : ℕ := (n + B - 1) / B

/-- Start with `init`. For each `t` from `0` to `T - 1`, replace the current value with
`body t current`. Return the final value. -/
def foldTiles {σ : Type} (T : ℕ) (init : σ) (body : ℕ → σ → σ) : σ :=
  (List.range T).foldl (fun s t => body t s) init

/-- Like `foldTiles`, but calculate the number of iterations from `n` and `B`. -/
def forTiles {σ : Type} (n B : ℕ) (init : σ) (body : ℕ → σ → σ) : σ :=
  foldTiles (numTiles n B) init body

/-- Read `x` at index list `idx`, or `fill` when `idx` is outside `x`. -/
def getOrFill {s : Shape} (fill : α) (x : Tensor α s) (idx : List ℕ) : α :=
  (getSpec x idx).getD fill

/-- Whether global index `g` is in the loop-bound mask: below `bounds` on every axis. -/
def InMask (bounds g : Fin r → ℕ) : Prop :=
  ∀ k, g k < bounds k

instance (bounds g : Fin r → ℕ) : Decidable (InMask bounds g) := by
  unfold InMask; infer_instance

/-- Read `x` at global index `g` when it is in the mask of `bounds`, `fill` otherwise. -/
def maskedGet {s : Shape} (bounds : Fin r → ℕ) (fill : α) (x : Tensor α s) (g : Fin r → ℕ) : α :=
  if InMask bounds g then
    getOrFill fill x (List.ofFn g)
  else
    fill

/-- Return the `k`-th index of position `c`. -/
def coordAt {s : Shape} (c : s.Coord) (k : Fin s.length) : ℕ :=
  (Shape.Coord.toList s c).get ⟨k.val, by simpa only [Shape.Coord.length_toList] using k.isLt⟩

/-- Global index of local position `localIdx` in tile `tileIdx`. -/
def globalIndex (blockSizes : List ℕ) (tileIdx localIdx : Fin blockSizes.length → ℕ) :
    Fin blockSizes.length → ℕ :=
  fun k => tileIdx k * blockSizes.get k + localIdx k

/-- Load tile `tileIdx` of `x`, a `blockSizes`-shaped tile masked by `bounds`. -/
def loadTile {s : Shape} (blockSizes : List ℕ) (bounds : Fin blockSizes.length → ℕ) (fill : α)
    (x : Tensor α s) (tileIdx : Fin blockSizes.length → ℕ) : Tensor α blockSizes :=
  TorchLean.Tensor.Internal.Rep.ofFn fun c =>
    maskedGet bounds fill x (globalIndex blockSizes tileIdx (coordAt c))

/-- Whether global index `g` belongs to tile `tileIdx`. -/
def InTile (blockSizes : List ℕ) (tileIdx g : Fin blockSizes.length → ℕ) : Prop :=
  ∀ k, tileIdx k * blockSizes.get k ≤ g k ∧ g k < tileIdx k * blockSizes.get k + blockSizes.get k

instance (blockSizes : List ℕ) (tileIdx g : Fin blockSizes.length → ℕ) :
    Decidable (InTile blockSizes tileIdx g) := by
  unfold InTile; infer_instance

/-- Store `v` into tile `tileIdx` of `out`; positions outside the tile or the bounds keep `out`, and
so does all of `out` when its rank differs from the tile's. -/
def storeTile {s : Shape} (blockSizes : List ℕ) (bounds : Fin blockSizes.length → ℕ)
    (out : Tensor α s) (tileIdx : Fin blockSizes.length → ℕ) (v : Tensor α blockSizes) :
    Tensor α s :=
  TorchLean.Tensor.Internal.Rep.ofFn fun c =>
    if h : s.length = blockSizes.length then
      let g : Fin blockSizes.length → ℕ := fun k => coordAt c (Fin.cast h.symm k)
      if InTile blockSizes tileIdx g ∧ InMask bounds g then
        getOrFill (out c) v (List.ofFn fun k => g k - tileIdx k * blockSizes.get k)
      else
        out c
    else
      out c

/-- Set the entries of tile `tileIdx` that lie outside the mask of `bounds` to `fill`: Helion's
`_mask_to(v, fill)` for a finite `fill`. -/
def maskTo (blockSizes : List ℕ) (bounds tileIdx : Fin blockSizes.length → ℕ) (fill : α)
    (v : Tensor α blockSizes) : Tensor α blockSizes :=
  TorchLean.Tensor.Internal.Rep.ofFn fun c =>
    if InMask bounds (globalIndex blockSizes tileIdx (coordAt c)) then
      v c
    else
      fill

/-- The largest entry along `axis` among those inside the mask of `bounds`: Helion's `amax` along
`axis` after `_mask_to(v, -inf)`. Where no entry along `axis` is inside the mask, this gives `0`
and Helion gives `-inf`. -/
def maskedMax [Max α] [Zero α] (blockSizes : List ℕ) (bounds tileIdx : Fin blockSizes.length → ℕ)
    (axis : Fin blockSizes.length) (v : Tensor α blockSizes) :
    Tensor α (Shape.eraseAxis blockSizes axis) :=
  TorchLean.Tensor.Internal.Rep.ofFn fun c =>
    let lane (j : ℕ) : List ℕ := (Shape.Coord.toList _ c).insertIdx axis j
    let inMask (j : ℕ) := InMask bounds (globalIndex blockSizes tileIdx fun k => (lane j).getD k 0)
    let entries := ((List.range (blockSizes.get axis)).filter inMask).map fun j =>
      getOrFill 0 v (lane j)
    entries.max?.getD 0

end TorchLean.Helion
