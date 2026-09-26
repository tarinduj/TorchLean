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

/-- Return a `[B0, B1]` tensor. Its position `(i, j)` reads `x[tm * B0 + i, tn * B1 + j]`,
or `fill` if that position is outside `x`. -/
def loadTile {α : Type} [TorchLean.Storage α] {M N : ℕ}
    (B0 B1 : ℕ) (fill : α) (x : Tensor α [M, N]) (tm tn : ℕ) : Tensor α [B0, B1] :=
  Tensor.dim fun i => Tensor.dim fun j =>
    Tensor.scalar (getOrFill fill x (tm * B0 + i) (tn * B1 + j))

/-- Return a new tensor equal to `out`, except that row `tm * B0 + i` gets `v[i]` wherever
that row exists in `out`. -/
def storeTile {α : Type} [TorchLean.Storage α] {M B0 : ℕ}
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

/-- At position `(i, j)`, `loadTile` returns `x[tm * B0 + i, tn * B1 + j]` when it exists,
and `fill` otherwise. -/
theorem get2_loadTile {α : Type} [TorchLean.Storage α] {M N : ℕ}
    (B0 B1 : ℕ) (fill : α) (x : Tensor α [M, N]) (tm tn : ℕ) (i : Fin B0) (j : Fin B1) :
    Spec.get2 (loadTile B0 B1 fill x tm tn) i j =
      getOrFill fill x (tm * B0 + i) (tn * B1 + j) := by
  simp [loadTile, Spec.get2, Spec.get]

/-- After `storeTile`, row `r` equals `v[r - tm * B0]` if tile `tm` covers it; otherwise
it still equals `out[r]`. -/
theorem getScalar_storeTile {α : Type} [TorchLean.Storage α] {M B0 : ℕ}
    (out : Tensor α [M]) (tm : ℕ) (v : Tensor α [B0]) (r : Fin M) :
    Tensor.getScalar (storeTile out tm v) r =
      if h : tm * B0 ≤ r.val ∧ r.val < tm * B0 + B0 then
        Tensor.getScalar v ⟨r.val - tm * B0, by omega⟩
      else
        Tensor.getScalar out r := by
  simp [storeTile]

/-- Storing tile `tm` and then a different tile `tm'` gives the same result in reverse order. -/
theorem storeTile_comm {α : Type} [TorchLean.Storage α] {M B0 : ℕ}
    (out : Tensor α [M]) (tm tm' : ℕ) (h : tm ≠ tm') (v v' : Tensor α [B0]) :
    storeTile (storeTile out tm v) tm' v' = storeTile (storeTile out tm' v') tm v := by
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨r, _⟩
  rw [← Tensor.getScalar_eq_apply, ← Tensor.getScalar_eq_apply]
  simp only [getScalar_storeTile]
  have hd : (tm + 1) * B0 ≤ tm' * B0 ∨ (tm' + 1) * B0 ≤ tm * B0 := by
    rcases Nat.lt_or_gt_of_ne h with h | h
    · exact .inl (Nat.mul_le_mul_right _ h)
    · exact .inr (Nat.mul_le_mul_right _ h)
  rw [Nat.succ_mul, Nat.succ_mul] at hd
  split_ifs <;> first | rfl | omega

end TorchLean.Helion
