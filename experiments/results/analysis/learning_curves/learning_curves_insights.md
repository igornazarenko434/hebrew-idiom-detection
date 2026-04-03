# Learning Curve Insights (Validation F1)

**Source:** TensorBoard logs from full fine-tuning runs  
**Script:** `scripts/analyze_learning_curves.py`  
**Outputs:** `learning_curves_summary.csv`, `learning_curves_summary_by_epoch3.csv`

---

## What this analysis measures

- Validation **F1 per epoch** from TensorBoard (`eval/f1` + `train/epoch`).
- Curves reflect **overall validation performance**, not per‑idiom F1 (per‑idiom curves would require per‑epoch predictions, which are not logged).

---

## Key Findings (Evidence‑Based)

### Span task (unseen generalization context)
From `learning_curves_summary_by_epoch3.csv`:

- **mBERT (span)** reaches **F1 ≈ 0.992** by epoch 3  
  → shows rapid convergence on span detection.

- **XLM‑RoBERTa (span)** reaches **F1 ≈ 0.991** by epoch 3  
  → similarly fast convergence.

- **NeoDictaBERT (span)**: first logged eval is at **epoch ~3.7**, so we cannot claim “by epoch 3” from available logs.

