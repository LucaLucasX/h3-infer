# Style-Preserving Merge — Design

**Date:** 2026-06-22
**Status:** FINAL DESIGN = the per-LoRA `preserve` flag only. The automatic `sum_preserve` mode (Fix 1) was
**reverted** — see the postmortem below. Default merging is unchanged from `main` (orthogonal → weighted_average /
SLERP); style preservation is opt-in via the flag.

> ## Postmortem — why automatic `sum_preserve` was reverted
>
> Fix 1 routed orthogonal groups to a bounded-additive `sum_preserve` mode. It fixed the 2-LoRA style+content case,
> but a real run of **3 character/concept LoRAs (no style)** then oversaturated badly (energy 2.52× vs `main`'s
> 1.02×) — additive stacking is exactly wrong for a balanced multi-LoRA blend. The root realization: **the analyzer
> cannot distinguish "preserve this style LoRA" from "blend these characters" — both are orthogonal.** Additive vs
> averaging is a question of *user intent*, not of any measurable signal (the deep-research wave already refuted
> auto-detecting "style" from weight statistics). So additive must NEVER be auto-selected; it has to be driven by the
> explicit `preserve` flag.
>
> **What shipped instead:** default routing reverted to `main` (weighted_average / SLERP for orthogonal). The
> `preserve` flag became a general overlay in `_merge_diffs`: blend the non-preserved LoRAs with the chosen mode,
> then add each preserved LoRA's full-strength delta on top (also exempt from sparsification + TIES sign-election).
> The standalone `sum_preserve` mode, its exact-linear handling, the `_ORTHO_SUM_HEADROOM` constant, and the `sum+`
> display were all removed. The auto-strength per-LoRA exclusion stays **descoped** (single uniform scalar baked into
> the linear path + group-cache replay; mild/floor-bounded).

---

> **NOTE:** the sections below are the ORIGINAL Fix-1 design (automatic `sum_preserve`). They are kept for history;
> the postmortem above supersedes them. The `preserve`-flag plumbing (Fix 2) shipped as described.
**Trigger:** Merging a style LoRA (e.g. `prodigy`) into a content/concept LoRA via `per_prefix`
makes the style vanish; `additive` keeps it but does no conflict resolution.

## Root cause (verified against a real run)

Real AutoTuner run, 2 LoRAs on Flux:

| LoRA | strength | rank | L2 norm |
|------|----------|------|---------|
| `bj_…epoch15` (concept) | 1.0 → 0.914 | 21 | 7.10 |
| `prodigy` (style) | 1.99 → 1.819 | 128 | 15.04 |

cos ≈ 0.002 (orthogonal), excess conflict 0%, sparsification **disabled**. All 112 groups route to
`weighted_average`. That path (`_merge_diffs`, mode `weighted_average`, line ~3578) computes
`Σ wᵢdᵢ / Σ|wᵢ|`, so the style lands at `1.819 / (0.914+1.819) = 0.666×` its delta — a **2.7× suppression**
vs `additive` (`weighted_sum`, full `1.819×`). The trap: `weighted_average` makes `strength` a *ratio* knob,
not an *amount* knob — `s/(s+s')` asymptotes to 1.0, so cranking the style strength can never push it past 1×.

This is NOT the literature's sign-election/trimming washout (that needs real conflict or TIES); the deep-research
wave (`wstby86xn`) confirmed: style is **not** a low-magnitude signal (prodigy is the *heavier* LoRA), and the
washout here is purely our `÷Σw` averaging.

## Fix 1 — bounded-additive `sum_preserve` mode (default behaviour)

New merge mode. `result = Σ wᵢdᵢ` (each LoRA at full strength), then a **per-group energy cap**:

```
max_single = max_i ‖wᵢ dᵢ‖                 # strongest single contribution in this group
cap        = _ORTHO_SUM_HEADROOM × max_single   # headroom = 1.5
if ‖result‖ > cap:  result *= cap / ‖result‖
```

- **N=2 orthogonal** (user case): `‖result‖ = √(E_s+E_c) ≈ 1.03×max_single < 1.5×` → no cap → prodigy at full 1.819×. ✅
- **N=3 equal orthogonal**: `‖result‖ = √3 ≈ 1.73×max_single > 1.5×` → scaled to 1.5× → bounded, no oversaturation. ✅

The cap is **per-group** and self-contained, so it does NOT depend on the global auto-strength floor (the thing
that made a naive `weighted_average→weighted_sum` flip oversaturate at N≥3 — see memory `autotuner-speed-batches`
bug 458f6dd and the orthogonal-floor clamp at `_compute_branch_auto_scale:5837`).

### Routing (`_decide_prefix_mode`, ~6229)

Orthogonal (`|cos| < orthogonal_cos_sim_max`), **non-opposing** (`cos ≥ 0`), `n_loras ≥ 2`: → **sum_preserve** in **all**
strategy sets.

> **Revised after a real run.** The first cut kept `full` → SLERP and only used `sum_preserve` for `no_slerp`/`basic`.
> But the AutoTuner's heuristic scorer (`_score_merge_result`: `effective_rank·0.4 + (1−norm_cv)·0.3 + sparsity_fit·0.3`)
> has **no magnitude/energy term** and rewards *uniform norms* — so it ranked the SLERP config (energy 1.22×) **above**
> the `sum_preserve` config (energy 1.91×) and applied it, leaving the style washed out by default. Since SLERP is
> strictly worse than `sum_preserve` for orthogonal groups (it rotates two independent directions into a 45° blend that
> loses both, vs keeping each in its own subspace), orthogonal groups now use `sum_preserve` everywhere and SLERP is
> retained only for **non-orthogonal, similar-direction** (cos ≈ 0.25–0.5) low-conflict groups in the `full` set.

Opposing groups keep `weighted_average` (cancellation is intended). Conflicted groups (excess conflict > threshold) keep TIES.

### Implementation points

- `_ORTHO_SUM_HEADROOM = 1.5` module constant.
- `_merge_diffs`: handle `mode == "sum_preserve"` (dense path) — additive sum + cap from actual tensor norms.
- `_build_exact_linear_patch`: handle `sum_preserve` so full-rank Flux patches stay **low-rank** (compute cap from
  Pass-1 `per_lora_norm_sq` + `pairwise_dots`, fold the scalar `c` into the fused factors). Falls back to dense
  (returns None) when stats unavailable (conflict modes / LoKr / missing prefix) — dense path computes the exact cap.
- `_merge_one_group` (~7327): add `sum_preserve` to the linear-eligible modes; thread the norm stats in.
- Display: map `sum_preserve` to the `====` (sum) glyph + a "sum+" / "sum-preserve" label in the block-strategy map
  and per-group counts.
- Scoring: ensure candidate scorer treats `sum_preserve` like a magnitude-preserving linear mode (no crash on the
  new string; reward energy preservation as for slerp).

## Fix 2 — per-LoRA `preserve` flag (explicit style protection)

For the harder cases the literature flags (real conflict / 3+ LoRAs / TIES), let the user tag a LoRA as a style/
preserve LoRA on the stack. A tagged LoRA:

1. **Exempt from TIES sign-election deletion** — in `_ties_disjoint_merge` (3056) its contribution is always added,
   even where it loses the majority-sign vote (no minority-direction zeroing).
2. **Exempt from sparsification** — its diff is not trimmed/DARE-dropped.
3. **Strength floor** — excluded from the global auto-strength reduction (kept at its set strength).

### Plumbing

- `LoRAStack` (458): add `preserve` (BOOLEAN) widget → dict key `"preserve"`.
- `LoRAStackDynamic` (521): add `preserve_{i}` per-slot widget (advanced mode) → carried in the tuple as a new
  trailing field; `configure()`/positional-restore migration per memory `stacker-positional-restore`.
- `_normalize_stack` (3923): carry `preserve` onto the normalized dict (both tuple + dict formats).
- `_prepare_group_diffs` (2158): emit a `preserve` map keyed by LoRA index alongside `eff_strengths`.
- `_merge_diffs`: new `preserve_flags=None` param (aligned with `diffs_with_weights`); honoured in the TIES and
  sparsification branches.
- `_compute_branch_auto_scale` (5775): preserved LoRAs excluded from the reduction (their effective strength is held).
- AutoTuner replay / tuner-data: persist the preserve flags so a replayed config behaves identically.

## Tests

- `sum_preserve` dense: N=2 orthogonal → both diffs at full weight (norm ≈ √(E1+E2), un-capped); N=3 equal orthogonal
  → capped at 1.5×single; opposing pair NOT routed to sum_preserve.
- `_build_exact_linear_patch` sum_preserve: low-rank patch reconstructs `c·Σ wᵢ BᵢAᵢ`; matches the dense path.
- `_decide_prefix_mode`: orthogonal no_slerp/basic → sum_preserve; full → slerp; opposing → weighted_average.
- preserve flag: tagged LoRA survives a TIES group where it holds the minority sign; survives sparsification; auto-
  strength leaves its strength unscaled.
- Stack nodes: `preserve` round-trips through both stack formats; positional-restore migration covers the new widget.

## Out of scope (noted, not built)

- Block-aware style protection (B-LoRA block 5 / SplitFlux blocks 30–57) — architecture-specific, brittle across the
  10+ supported architectures. Revisit if `sum_preserve` + preserve flag prove insufficient.
- Auto-detecting "this is a style LoRA" from weight stats — the low-magnitude premise was refuted, so detection is
  unreliable; explicit tagging is the robust path.
