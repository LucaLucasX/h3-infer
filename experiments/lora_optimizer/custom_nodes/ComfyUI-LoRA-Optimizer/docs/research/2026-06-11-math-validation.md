# AutoTuner Math Validation — Deep Research Report (2026-06-11)

Validation of the 10 mathematical foundations of the AutoTuner against published
literature and first-principles derivation. Sources: 30+ primary papers fetched and
claim-extracted; the Sheppard-formula cluster additionally passed 15/15 adversarial
verification votes (verified verbatim against source PDFs).

## Verdict table

| # | Claim | Verdict | Key source(s) |
|---|-------|---------|---------------|
| 1 | Sign-conflict baseline `arccos(ρ)/π` | **correct-with-caveats** — formula exact, but compared against the wrong statistic | Sheppard 1899; O'Donnell AoBF Thm 40 (arXiv 1205.0314, verified verbatim); Cho & Saul NeurIPS 2009; Giner et al. QF 2018 |
| 2 | Subsampling estimator (uniform w/ replacement, ×n/k) | **correct-with-caveats** — unbiased; suboptimal variance for heavy-tailed entries | Musco et al. VLDB 2024 (Threshold Sampling) |
| 3 | WA merge energy `(Σs²‖d‖² + 2Σss⟨d,d⟩)/(Σ\|s\|)²` | **correct** — exact expansion, internally consistent with the WA implementation | first-principles |
| 4 | Auto-strength "preserve strongest contributor energy" | **unprincipled-heuristic** (conservative/safe) | MetaGPT (closed-form λ∝‖τ‖²); Pico calibration (target = mean source norm, arXiv 2604.16826) |
| 5 | Effective rank (Roy–Vetterli) /40 as quality | **directionally supported, cap unprincipled** | Roy & Vetterli 2007; rank-collapse + stable-rank correlation papers; Iso-C ICML 2025; contradicted in heterogeneous regimes |
| 6 | Iterative pairwise SLERP | **correct for N=2; WRONG for N≥3** — order-dependent, collapses empirically at m≥5 | multi-SLERP analyses; Karcher-mean merging (0.239 vs 0.610 at m=5) |
| 7 | TIES/DARE/DELLA fidelity | **DELLA correct; DARE correct-but-misattributed; TIES deviates** | TIES NeurIPS 2023; DARE; DAREx ICLR 2025; DELLA/MAGPRUNE official impl |
| 8 | Subspace overlap `‖UᵀU‖²_F/k`, threshold 0.35 | **metric correct** (= mean squared canonical correlation); threshold unprincipled but conservative | published identically for LoRA merging; random baseline = k/D exactly |
| 9 | Composite score (rank/cv/energy/sparsity) | **unprincipled-heuristic** — cv & sparsity-fit have no literature support; energy does | Iso-C SAR; stable-rank correlation; Pico norm calibration |
| 10 | Decision tree, `(0.5+0.5·overlap)` modulation | **reasonable, one inconsistency** — multiplier floors at 0.5 with zero overlap | WUDI/subspace-interference theory; "in-the-wild" merging robustness studies |

## Detailed findings

### 1. Sign-conflict baseline — the weighted-vs-unweighted mismatch (HIGH priority)

The `arccos(ρ)/π` form is exactly Sheppard's theorem for ρ-correlated Gaussians
(verified verbatim: O'Donnell, *Analysis of Boolean Functions*, Thm 40) and — more
relevantly for deterministic weight diffs — it is the degree-0 arc-cosine kernel
result (Cho & Saul, NeurIPS 2009): for two FIXED vectors, the probability of sign
disagreement under a random Gaussian projection is exactly θ/π. It also holds
approximately on heavy-tailed real data (validated on 10y of S&P 500 returns,
Giner et al. 2018). The functional form is solid.

**The problem:** the measured statistic is min(|a|,|b|)-magnitude-weighted and
noise-floor-filtered, while the baseline is the UNWEIGHTED expectation. Cho & Saul's
degree-n kernels show magnitude-weighted sign agreement has a *different* angular
dependence (J₁(θ) = sinθ + (π−θ)cosθ vs J₀(θ) = π−θ) — the two statistics are not
interchangeable. For ρ>0, sign-mismatching positions tend to have smaller |a|,|b|
(mismatches concentrate where one variable fights the correlation near zero), so the
weighted conflict ratio is systematically BELOW the unweighted rate; the noise floor
truncation pushes the same direction. Net effect: `excess_conflict = weighted −
arccos(ρ)/π` is biased negative → **real conflict is under-detected for correlated
LoRAs** (frequently clamped to 0), which biases the decision tree away from
TIES/consensus toward weighted-average.

**Fix:** compute the unweighted mismatch fraction over the same (joint-nonzero,
noise-floored) positions and use THAT against the arccos/π baseline for
`excess_conflict`. Keep the weighted ratio as a separate magnitude-relevance signal
(it is still useful for the merge itself, just not against this baseline).
Optionally derive the weighted baseline via the J₁ machinery, but the unweighted
comparison is the simplest consistent estimator.

### 2. Subsampling estimator

Uniform-with-replacement sampling rescaled by n/k is an unbiased estimator of full
sums (elementary). The literature (Threshold Sampling, VLDB 2024) shows uniform
sampling is the weak choice for heavy-tailed entries — squared-magnitude-proportional
coordinated sampling achieves variance ≤ (2/m)·max(‖a_I‖²‖b‖², ‖a‖²‖b_I‖²) and
uniform is only equivalent when entries are similar in magnitude; even 2% outliers
makes weighted sampling clearly better. LoRA diffs are mildly heavy-tailed, k=100k is
large, and mixing the estimated dot with exact norms introduces no systematic bias —
so this is acceptable. If cross-term instability is ever observed, switch the dot
estimate to norm-proportional sampling.

### 3. Weighted-average energy — correct

The numerator is exactly ‖Σᵢ sᵢdᵢ‖² and the (Σ|sᵢ|)² denominator matches the
implementation's normalization (weights wᵢ = sᵢ/Σ|s|, signed numerator, absolute
denominator) — internally consistent including negative strengths. The
√(kept_fraction) scaling under subsampling assumes uniform energy per group;
every-Nth-from-conflict-sorted selection is stratified, making it roughly
representative — acceptable approximation, documented caveat.

### 4. Auto-strength target

No literature supports "scale combined energy to the strongest single contributor".
Published data-free alternatives: **MetaGPT** derives closed-form λ_t =
‖τ_t‖²/Σ_k‖τ_k‖² by minimizing average loss difference (valid under the
orthogonality assumption — which the AutoTuner explicitly measures!); **Pico-style
calibration** rescales the merged update so its Frobenius norm equals the MEAN of
the source norms (γ = avg‖ΔW_t‖/‖ΔW_merged‖). The current max-contributor target is
more permissive than mean-norm for heterogeneous stacks and behaves identically for
equal norms; it is safe but ad hoc. The 0.85 orthogonal floor is pure empiricism.
Candidate upgrade: mean-source-norm target, with MetaGPT λ as the per-LoRA weighting
when measured cos ≈ 0.

### 5. Effective rank as quality — genuinely contested in the literature

Support: rank collapse of merged task vectors is *provably inevitable* as experts
grow under task arithmetic; higher stable rank empirically correlates with better
merge performance across TA/TIES/Consensus; actively raising the spectrum
(SV clamping) improves merges; Iso-C (ICML 2025) flattens the spectrum (maximizing
effective rank) and wins on homogeneous benchmarks. Counter-evidence: in
heterogeneous "in-the-wild" merging, Iso-C degrades monotonically with expert count
(down to 0% win probability) — the rank→quality link is regime-dependent.

The /40 cap is indefensible as a constant: Roy–Vetterli erank is bounded by true
rank, which for merged LoRA patches is ≤ Σ ranks (e.g. 32 for two rank-16 LoRAs —
the score saturates differently per stack). **Fix:** normalize by achievable rank:
`eff_rank / min(Σᵢ rankᵢ, thin_dim)`, or use stable rank ‖A‖²_F/‖A‖²₂ which needs
no SVD and has the direct empirical correlation result behind it.

### 6. SLERP — replace with Karcher mean for N≥3 (HIGH priority)

(a) In d~10⁶⁻⁷, independent LoRA diffs concentrate near 90°; the sphere is locally
flat and SLERP ≈ LERP + norm correction — the real benefit IS the norm preservation,
which the code applies explicitly anyway. (b) Iterative pairwise SLERP with frac =
w_next/(w_acc+w_next) is a recognized "multi-SLERP" formulation but is
order-dependent (the descending-|weight| sort makes it deterministic, not
order-free) and is NOT a Fréchet mean. The principled N-way generalization is the
weighted **Karcher mean** on the sphere (closed-form log/exp-map iteration), and the
empirical case is stark: at m=5 models, multi-SLERP scores 0.239 vs Karcher 0.610 vs
plain LERP 0.542 — iterative SLERP collapses below even linear averaging. (c) The
magnitude claim is true: WA of N orthonormal vectors shrinks norm by 1/√N; geodesic
interpolation avoids the shrinkage (also documented as "representation collapse"
from Euclidean blending of far-apart checkpoints). For two-model merges, SLERP is
exactly the Fisher-Rao Karcher mean under a spherical proxy — keep it for N=2.

**Fix:** for n_diffs ≥ 3, replace the iterative loop with a weighted Karcher mean:
normalize to unit sphere, iterate (tangent-space log-map weighted average at current
estimate → exp-map back) 2–3 times from the normalized weighted LERP as init, then
apply the existing norm correction. Cost is a few extra dot products per group.

### 7. TIES / DARE / DELLA fidelity

- **TIES:** the paper's sign election is ONLY the magnitude-weighted total-mass vote
  γ = sgn(Σ τ̂) — there is no frequency vote in the paper, yet the code's *default*
  is `frequency` (switching to `total` only when magnitude ratio is high). The paper
  also trims *globally* at top-20% (density 0.2, validated to match full-density
  performance), while the code trims per-tensor with density clamped to 0.4–0.95 —
  i.e. trimming far less than the validated recipe (consistent with the earlier
  internal audit note "trims too little"). Disjoint-merge division by |A^p| matches.
  **Fix:** default sign election → `total`; allow/auto-select density down to ~0.2.
- **DARE:** canonical rescale is exactly 1/(1−p); the code reduces to it at
  dampening=0 ✓. But the dampening interpolation q = density + dampening·(1−density)
  is NOT DAREx — DAREx-q selects q empirically (validation grid search) or via
  analytical per-model solutions, never a closed-form interpolation. The mechanism
  (q > 1−p tames the variance blow-up that provably breaks DARE at high p) is
  directionally aligned with DAREx's diagnosis, but the docstring attribution is
  wrong, and dampening>0 makes the estimator intentionally biased (shrinks E[output]
  by density/q). **Fix:** comment correction + document the bias.
- **DELLA:** matches the official MAGPRUNE implementation in every checked detail —
  linear-in-rank drop probs over an ε-window centered on the target rate, per-row
  ranking, higher magnitude → lower drop, per-element 1/(1−pᵢ) rescale. ✓
- **Conflict-aware variants + >40% skip guard:** novel (no literature equivalent),
  but internally consistent with the Sheppard base rate (~50% at ρ=0) and with the
  TIES paper's observation that real trimmed task vectors show conflict rates far
  below 50%. Reasonable.

### 8. Subspace overlap

The metric (1/k)·‖Q₁ᵀQ₂‖²_F is exactly the mean squared canonical correlation (mean
cos² of principal angles) and appears with the identical formula in recent LoRA-
merging literature. The chance baseline for random k-dim subspaces of R^D is exactly
k/D (first-principles: for Haar-uniform subspaces E[Q₂Q₂ᵀ] = (k/D)·I; verified
numerically — NOT stated in the Pico paper, contrary to the original extract) —
≈0.025 at D=320, ≈0.0016 at D=5120 — so 0.35 is far above chance at
every layer size, and measured overlaps between independently trained adapters are
tiny (B-side ≈0.1 at small ranks). Consequences: (a) the threshold is "safe" but the
consensus gate (requires ≥0.35) is nearly unreachable for genuinely independent
LoRAs — by design it only fires for same-concept/retrained adapters; (b) literature
notes strong B-side (output) vs A-side asymmetry — averaging left/right loses
signal; (c) k≤8 truncation under-measures rank-32 adapters. **Fix (optional):**
k = min(rank_a, rank_b), chance-corrected score (overlap − k/D)/(1 − k/D), and
consider weighting the B-side higher.

### 9. Composite score

Data-free merge-quality proxies with literature support: **subspace alignment
ratio** between each source and the merged matrix (strong Pearson correlation with
per-task gains — Iso-C paper), **stable rank** (correlates with merge quality),
**norm/energy preservation** vs the mean-source-norm target (Pico calibration), and
MetaGPT's loss-difference objective. NOT supported anywhere: norm-CV across patches
and proximity-to-DARE-ideal-sparsity — both are internal inventions; sparsity-fit in
particular repurposes a sparsification hyperparameter as a quality target, which is
circular. A literature-grounded v3 composite: energy-preservation (vs mean-norm
target) + stable-rank ratio + per-source subspace alignment ratio, dropping cv and
sparsity-fit or demoting them to tiebreakers.

### 10. Decision tree

Gating on (conflict, cosine, subspace overlap) is consistent with interference
theory — interference happens in shared subspace, sign conflicts only matter where
subspaces intersect (WUDI's interference analysis; "directions shared across tasks
cause interference"). One inconsistency: `effective_conflict = excess·(0.5 +
0.5·overlap)` floors the multiplier at 0.5 even when overlap = 0 — under the
interference model, zero subspace intersection means excess sign-conflict cannot
cause interference, so the floor keeps half of a signal the theory says is inert.
Worth considering `(α + (1−α)·overlap)` with smaller α, or overlap-gating excess
directly. Also worth noting: heterogeneous-merging studies repeatedly find plain
task arithmetic with fixed coefficients more robust than TIES/Iso-C as expert count
grows — supporting the tree's bias toward weighted-average as the default and TIES
only under strong evidence.

## Prioritized fix list

1. **Claim 1 (quality, affects every decision):** use the unweighted mismatch
   fraction (same filtered positions) against the arccos/π baseline for
   `excess_conflict`. Bump ANALYSIS_CACHE_VERSION + AUTOTUNER_ALGO_VERSION.
2. **Claim 6 (quality for 3+ LoRA stacks):** weighted Karcher mean instead of
   iterative pairwise SLERP for n≥3 (keep SLERP for n=2).
3. **Claim 7 TIES (fidelity):** default sign election `total`; extend density range
   toward the paper-validated 0.2.
4. **Claims 5/9 (scoring):** replace eff_rank/40 with achievable-rank or stable-rank
   normalization; longer-term replace cv/sparsity-fit with subspace-alignment-ratio.
5. **Claim 8 (minor):** chance-corrected overlap, k=min(ranks).
6. **Claim 7 DARE (docs):** fix the DAREx attribution comment; document dampening bias.
7. **Claim 10 (tunable):** revisit the 0.5 floor in the conflict modulation.

## Provenance

Deep-research run wf_292bfb85-f65: 1 scope + 5 search + 47 fetch/extract agents
(121 unique central claims, multi-source) + 15/15 adversarial verification votes
upheld (Sheppard cluster, verified verbatim against arXiv 1205.0314 and
QF 10.1080/14697688.2017.1414510). Remaining votes skipped — key facts were
independently confirmed by 2–3 separate primary extracts each. Key sources:
arXiv 2306.01708 (TIES), 2311.03099 (DARE), 2406.11617 (DELLA), 2410.09344 (DAREx),
Iso-C ICML 2025, MetaGPT, Pico (2604.16826), KnOTS (2410.19735), WUDI, Threshold Sampling (VLDB 2024),
Roy & Vetterli 2007, Cho & Saul NeurIPS 2009, multi-SLERP/Karcher-merge analyses.

## Batch-2 adversarial verification (2026-06-12)

The 10 load-bearing claims above were re-verified with 2 adversarial votes each
(source-fidelity + independent refutation; run wf_807d440b-5f4, 20 agents).
Result: **7 upheld, 3 disputed — all shipped code changes survive.**

Disputes (attribution/framing, not substance):
- **karcher-principled:** the emergentmind source frames pairwise SLERP and the
  Karcher mean as coequal strategies, not approximation-vs-target. The math is
  independently established (Buss & Fillmore 2001: existence/uniqueness +
  convergent log/exp algorithm), so the Karcher implementation stands on the
  primary literature, not the blog framing.
- **subspace-kd:** the canonical-correlation identity and the k/D chance
  baseline are TRUE (proven first-principles + numerically in the vote:
  Björck–Golub principal angles; Haar-uniform expectation) but are NOT stated
  in the Pico paper — the original extract misattributed them. Citations
  corrected above.
- **mean-norm-calib:** the mean-source-Frobenius-norm rescale IS published
  verbatim — but in the Pico paper ("Crowded in B-Space", arXiv 2604.16826),
  not KnOTS (which only tunes a scalar coefficient on a held-out validation
  set and has no norm calibration). This report previously said "KnOTS
  calibration"; corrected to Pico throughout.

Notable upheld-with-caveats:
- **J1-weighted:** a 4M-sample Monte Carlo in the vote confirmed the
  min(|a|,|b|)-weighted mismatch deviates from arccos(ρ)/π exactly as claimed
  (ρ=0.5: 0.21 weighted vs 0.33 unweighted) and that the 0f8a97c fix is the
  correct apples-to-apples comparison. Caveat: the two laws coincide at ρ=0,
  so for near-orthogonal stacks the old bias was small; it matters as |ρ|
  grows.
- **multislerp-collapse:** the 0.239/0.610/0.542 numbers are verbatim-correct,
  but the paper's "(Multi-)SLERP" baseline is MergeKit's tangent-space
  barycentric multislerp, not iterative pairwise SLERP, and several Euclidean
  baselines collapse at m≈5 too (pool heterogeneity contributes). The vote
  nonetheless independently confirmed pairwise SLERP is order-dependent and
  non-Fréchet, and audited the Karcher implementation as faithful — "the code
  change stands on its own merits."
- **stable-rank-quality:** the correlation is real but NOT monotone/causal:
  CART (2412.12153) shows accuracy peaks at ~8% of full rank then declines;
  AdaRank (2503.22178) shows dominant singular components cause interference;
  stable rank is inflated by isotropic noise (e.g. DARE residue). The v2
  rank term is one-sided (capped at 1), which tolerates this — but do NOT
  evolve it into "maximize rank."
- **della-official:** local code follows the DELLA PAPER's MAGPRUNE convention
  (window width ε); the official repo uses width 2ε — an ε-scaling convention
  difference, not an error. Behavior also diverges at extreme density±ε (we
  clamp, repo raises).
- **darex-empirical:** the dampening interpolation is a defensible data-free
  proxy within the paper's admissible q ∈ (1−p, 1]; describe it as "inspired
  by DAREx", not an implementation of DAREx-q's selection rule.
