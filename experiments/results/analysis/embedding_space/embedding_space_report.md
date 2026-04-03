# Embedding Space Visualization (t-SNE)

**Task:** CLS (sentence classification)

**Why CLS:** We use fine-tuned CLS checkpoints to extract [CLS] embeddings, so the representation aligns with the sentence-level idiomaticity task. This analysis is not for SPAN.

**Data:** Seen test (`data/splits/test.csv`) vs. Unseen test (`data/splits/unseen_idiom_test.csv`)

**Method:**
- Extract [CLS] embeddings from the fine-tuned CLS model.
- Reduce with PCA (50 dims) then t-SNE (2D).
- Plot seen vs. unseen in the same embedding space.

---

## NeoDictaBERT (CLS)

**Observation:** Two large islands with heavy mixing of seen and unseen samples inside each region. No clear split boundary.

**Interpretation (safe):** The CLS embedding space does not separate seen vs. unseen idioms; unseen points sit in similar regions as seen ones.

**Implication:** Consistent with strong CLS generalization (small seen→unseen gap).

**Caveat:** t-SNE is qualitative; interpret overlap, not geometry.

---

## XLM-RoBERTa (CLS)

**Observation:** A curved manifold with substantial intermixing of seen and unseen across the arc. No distinct split clusters.

**Interpretation (safe):** Split-invariant CLS embeddings; unseen examples occupy the same regions as seen.

**Implication:** Supports CLS robustness on unseen idioms (small generalization gap).

**Caveat:** t-SNE is qualitative; interpret overlap, not geometry.

---

## Paper-Ready Takeaway

Both Hebrew-specific (NeoDictaBERT) and multilingual (XLM-R) models show **high overlap** between seen and unseen in CLS embedding space. This supports the broader finding that **detection generalizes**, even when **span localization does not**.
