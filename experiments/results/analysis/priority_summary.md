# Priority 1 + Priority 2 Analysis Summary (CoNLL 2026)

This document consolidates the **Priority 1** and **Priority 2** analyses required by `CONLL_2026_COMPREHENSIVE_REVIEW.md`. All statements below are grounded in generated outputs.

---

## Priority 1 (Completed)

### 1) Morphology verification (claim check)
- Output: `experiments/results/analysis/morphology_verification/morphology_verification_summary.md`
- Result: **Morphology sensitivity claim is NOT supported** with current evidence (gap < 10% and exact‑match unseen count is tiny).
- Paper action: keep as a cautionary note; do not claim strong morphology effect.

### 2) Bootstrap confidence intervals
- Output: `experiments/results/analysis/finetuning_summary.md`
- Result: 95% bootstrap CIs included for key F1 results.
- Paper action: use CI‑formatted numbers (mean ± std + CI).

### 3) Full statistical tests table
- Output: `experiments/results/analysis/statistical_tests/paired_ttests_complete.*`
- Result: full pairwise comparisons with p‑values, Bonferroni, Cohen’s d.
- Paper action: include full table in supplementary material.

### 4) Power analysis statement
- Output: `experiments/results/analysis/statistical_tests/paired_ttests_complete.md`
- Result: detectable effect size (|d| ≥ 3.26 with n=3, 80% power).
- Paper action: add to Methods as limitation + power context.

### 5) 99.7% end‑boundary bias explanation
- Output: `experiments/results/analysis/boundary_bias_explanation.md`
- Result: four competing hypotheses + discriminative experiments.
- Paper action: include concise version in Discussion.

---

## Priority 2 (Completed)

### 1) Embedding space (CLS t‑SNE)
- Outputs:
  - Report: `experiments/results/analysis/embedding_space/embedding_space_report.md`
  - Figures: `paper/figures/embedding_space/*_tsne_seen_vs_unseen.png`
- Insight: seen vs unseen **overlap heavily** in CLS space (no clear separation), supporting robust **detection** generalization.

### 2) Attention pattern analysis (SPAN errors)
- Outputs:
  - `experiments/results/analysis/attention_patterns/attention_insights.md`
  - Figures: `paper/figures/attention_patterns/*_attention_ratio_correct_vs_partial_end.png`
- Insight: PARTIAL_END errors show **higher idiom‑focused attention** than PERFECT spans (NeoDictaBERT +29.6%, XLM‑R +43.9%), indicating **attention ≠ explanation** and boundary errors are not due to missing idiom focus.

### 3) Learning curves (TensorBoard)
- Outputs:
  - `experiments/results/analysis/learning_curves/learning_curves_summary.md`
  - Figures: `paper/figures/learning_curves/learning_curves_{cls,span}.png`
- Insight: span models reach **~0.99 F1 by epoch 3** for mBERT and XLM‑R; NeoDictaBERT span logs first eval at ~epoch 3.7, so “by epoch 3” is **not** supported for that model.

### 4) Qualitative error showcase (SPAN, unseen)
- Outputs:
  - `experiments/results/analysis/qualitative_errors/qualitative_errors.md`
  - Paper tables: `paper/tables/qualitative_errors.md`, `paper/tables/qualitative_errors.tex`
- Insight: representative errors are dominated by **PARTIAL_END** and **MISS** on unseen idioms, reinforcing the boundary truncation narrative.

---

## Integrated Story (Paper‑Ready)

**Detection generalizes; localization does not.**  
CLS embeddings show strong overlap between seen and unseen (t‑SNE), and classification gaps are small. Yet SPAN errors on unseen idioms are dominated by **boundary truncation** (Priority 1), even when attention is concentrated on idiom tokens (Priority 2). This indicates the failure is **not** detection, but **boundary decision**, consistent with the 99.7% end‑boundary bias. Learning curves show fast convergence on seen validation (span reaches ~0.99 quickly), which further suggests generalization—not optimization—is the bottleneck.

---

## Caveats (Do Not Overclaim)

- Morphology sensitivity is **not supported** with current evidence.
- Attention patterns are **correlational**, not causal.
- Learning curves are **aggregate validation F1**, not per‑idiom trajectories.

