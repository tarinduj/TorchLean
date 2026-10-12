import NN.IR.Semantics
import NN.Spec.Core.Context.Real

/-!
Loops over fixed-size tiles, reads from an input tensor, and writes to an output tensor.
-/

open Spec TorchLean

namespace TorchLean.Helion

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

/-- Read `x[r, c]`, or `fill` when `(r, c)` is outside `x`. -/
def getOrFill {α : Type} [TorchLean.Storage α] {M N : ℕ}
    (fill : α) (x : Tensor α [M, N]) (r c : ℕ) : α :=
  if h : r < M ∧ c < N then
    Spec.get2 x ⟨r, h.1⟩ ⟨c, h.2⟩
  else
    fill

/-- Read `x[r, c]` when `(r, c)` is within the loop bounds `m, n`, and `fill` otherwise: Helion's
load mask. Positions within the bounds but outside `x` also give `fill`. -/
def maskedGet {α : Type} [TorchLean.Storage α] {M N : ℕ}
    (m n : ℕ) (fill : α) (x : Tensor α [M, N]) (r c : ℕ) : α :=
  if r < m ∧ c < n then
    getOrFill fill x r c
  else
    fill

/-- Return a `[B0, B1]` tensor. Its position `(i, j)` is `maskedGet m n fill x` at
`(tm * B0 + i, tn * B1 + j)`, where `m, n` are the bounds of the loops that produce the tiles. -/
def loadTile2D {α : Type} [TorchLean.Storage α] {M N : ℕ}
    (m n B0 B1 : ℕ) (fill : α) (x : Tensor α [M, N]) (tm tn : ℕ) : Tensor α [B0, B1] :=
  Tensor.dim fun i => Tensor.dim fun j =>
    Tensor.scalar (maskedGet m n fill x (tm * B0 + i) (tn * B1 + j))

/-- Return a new tensor equal to `out`, except that row `tm * B0 + i` gets `v[i]` wherever
that row exists in `out`. -/
def storeTile1D {α : Type} [TorchLean.Storage α] {M B0 : ℕ}
    (out : Tensor α [M]) (tm : ℕ) (v : Tensor α [B0]) : Tensor α [M] :=
  Tensor.dim fun r =>
    let start := tm * B0
    Tensor.scalar <|
      if h : start ≤ r.val ∧ r.val < start + B0 then
        Tensor.getScalar v ⟨r.val - start, by omega⟩
      else
        Tensor.getScalar out r

/-! ## Lemmas -/

/-- With zero iterations, `body` is never called and the result is `init`. -/
theorem foldTiles_zero {σ : Type} (init : σ) (body : ℕ → σ → σ) :
    foldTiles 0 init body = init := rfl

/-- To run `T + 1` iterations, run the first `T`, then call `body` with index `T` and their result. -/
theorem foldTiles_succ {σ : Type} (T : ℕ) (init : σ) (body : ℕ → σ → σ) :
    foldTiles (T + 1) init body = body T (foldTiles T init body) := by
  simp [foldTiles, List.range_succ, List.foldl_append]

/-- When `B > 0`, the blocks counted by `numTiles n B` have room for at least `n` positions.
For example, two blocks of size four have room for seven positions. -/
theorem le_numTiles_mul (n B : ℕ) (hB : 0 < B) : n ≤ numTiles n B * B := by
  have h1 := Nat.div_add_mod (n + B - 1) B
  have h2 := Nat.mod_lt (n + B - 1) hB
  unfold numTiles
  rw [Nat.mul_comm]
  generalize B * ((n + B - 1) / B) = p at *
  omega

/-- At position `(i, j)`, `loadTile2D` returns `maskedGet` at `(tm * B0 + i, tn * B1 + j)`. -/
theorem get2_loadTile2D {α : Type} [TorchLean.Storage α] {M N : ℕ}
    (m n B0 B1 : ℕ) (fill : α) (x : Tensor α [M, N]) (tm tn : ℕ) (i : Fin B0) (j : Fin B1) :
    Spec.get2 (loadTile2D m n B0 B1 fill x tm tn) i j =
      maskedGet m n fill x (tm * B0 + i) (tn * B1 + j) := by
  simp [loadTile2D, Spec.get2, Spec.get]

/-- A lane outside the loop bounds reads `fill`. -/
theorem maskedGet_masked {α : Type} [TorchLean.Storage α] {M N : ℕ}
    (m n : ℕ) (fill : α) (x : Tensor α [M, N]) (r c : ℕ) (h : ¬(r < m ∧ c < n)) :
    maskedGet m n fill x r c = fill := by
  simp [maskedGet, h]

/-- A lane within the loop bounds reads `x[r, c]`, when the bounds fit inside `x`. -/
theorem maskedGet_enabled {α : Type} [TorchLean.Storage α] {M N : ℕ}
    (m n : ℕ) (fill : α) (x : Tensor α [M, N]) (r c : ℕ) (hr : r < m) (hc : c < n)
    (hm : m ≤ M) (hn : n ≤ N) :
    maskedGet m n fill x r c = Spec.get2 x ⟨r, by omega⟩ ⟨c, by omega⟩ := by
  simp [maskedGet, getOrFill, hr, hc, show r < M by omega, show c < N by omega]

/-- When the mask bounds equal the tensor dimensions, `maskedGet` equals `getOrFill`. -/
theorem maskedGet_eq_getOrFill {α : Type} [TorchLean.Storage α] {M N : ℕ}
    (fill : α) (x : Tensor α [M, N]) (r c : ℕ) :
    maskedGet M N fill x r c = getOrFill fill x r c := by
  unfold maskedGet getOrFill
  split_ifs <;> simp_all

/-- After `storeTile1D`, row `r` equals `v[r - tm * B0]` if tile `tm` covers it; otherwise
it still equals `out[r]`. -/
theorem getScalar_storeTile1D {α : Type} [TorchLean.Storage α] {M B0 : ℕ}
    (out : Tensor α [M]) (tm : ℕ) (v : Tensor α [B0]) (r : Fin M) :
    Tensor.getScalar (storeTile1D out tm v) r =
      if h : tm * B0 ≤ r.val ∧ r.val < tm * B0 + B0 then
        Tensor.getScalar v ⟨r.val - tm * B0, by omega⟩
      else
        Tensor.getScalar out r := by
  simp [storeTile1D]

end TorchLean.Helion
