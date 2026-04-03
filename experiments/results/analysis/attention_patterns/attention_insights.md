# Attention Pattern Insights (SPAN, Unseen)

**Purpose:** Explain what attention patterns reveal about boundary errors (PERFECT vs PARTIAL_END) in unseen idioms.

**Data Source:**
- `experiments/results/analysis/attention_patterns/*/unseen_test/attention_by_error.csv`
- Models: NeoDictaBERT, XLM‑RoBERTa (best seed by seen F1)

---

## Key Findings (Evidence‑Based)

### 1) Errors are **not** caused by lack of idiom focus

**NeoDictaBERT (SPAN, unseen):**
- PERFECT mean idiom‑attention ratio: **0.2320**
- PARTIAL_END mean idiom‑attention ratio: **0.3008**
- Relative change: **+29.6%** (errors show *more* idiom attention)

**XLM‑RoBERTa (SPAN, unseen):**
- PERFECT mean idiom‑attention ratio: **0.1778**
- PARTIAL_END mean idiom‑attention ratio: **0.2558**
- Relative change: **+43.9%**

**Interpretation:** The model strongly attends to idiom tokens even when it **fails** to localize boundaries. Boundary errors are not explained by “ignoring the idiom.”

---

### 2) Boundary failure is distinct from detection failure

The attention patterns align with the main dissociation:
- **CLS generalizes well** → detection is robust.
- **SPAN fails on unseen** → localization is fragile.

Even when attention is concentrated on idiom tokens, models still truncate idiom endings (PARTIAL_END), confirming that **boundary decision mechanisms** are the bottleneck, not idiom recognition.

---

### 3) Attention ≠ Explanation

High idiom‑focused attention **does not guarantee correct spans**. This supports the interpretation that **attention alone is not a faithful explanation** for span correctness, and that boundary tagging errors arise from deeper structural issues (IOB2 transition asymmetry, end‑boundary ambiguity).

---

## Paper‑Ready Wording (Suggested)

> “Attention analysis reveals a counter‑intuitive pattern: PARTIAL_END errors exhibit *higher* idiom‑token attention than correct spans (+29.6% for NeoDictaBERT, +43.9% for XLM‑R). This indicates that boundary failures are not due to missing idiom focus, but to the model’s inability to decide where the idiom terminates. The result strengthens the detection‑localization dissociation and supports the claim that attention is not a sufficient explanation for correct span localization.”

---

## Figures

- `paper/figures/attention_patterns/neodictabert_attention_ratio_correct_vs_partial_end.png`
- `paper/figures/attention_patterns/xlm-roberta-base_attention_ratio_correct_vs_partial_end.png`
