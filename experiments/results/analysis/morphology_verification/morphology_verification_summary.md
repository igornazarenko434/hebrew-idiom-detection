# Morphology Sensitivity Verification

**Priority 1 Task from CoNLL 2026 Review**

## Research Question

Do Hebrew morphological surface variations (base idiom ≠ inflected form) significantly increase error rates on unseen idioms?

## Data Source

- Manifest: `experiments/results/analysis/interpretability/boundary_morphology_manifest.csv`
- Total cases: 585
- Morphology variants: 577
- Exact matches: 8

## Key Findings

### Aggregate Results (All Models)

| Split | Morphology Variant Error Rate | Exact Match Error Rate | Gap |
|-------|-------------------------------|------------------------|-----|
| Seen | 5.5% | 0.0% | 5.5% |
| Unseen | 65.0% | 100.0% | -35.0% |

### Per-Model Results (Unseen Test)

| Model | Morphology Variant Error | Exact Match Error | Gap |
|-------|--------------------------|-------------------|-----|
| bert-base-multilingual-cased | 68.8% | 100.0% | -31.2% |
| xlm-roberta-base | 66.7% | 100.0% | -33.3% |
| neodictabert | 64.3% | 100.0% | -35.7% |
| dictabert | 60.0% | 100.0% | -40.0% |

## Claim Verification

❌ **NOT SUPPORTED**: Gap is below 10% threshold. Morphology sensitivity claim needs revision.

## Paper Text (If Verified)

> Hebrew morphological surface variation (base idiom ≠ inflected form) significantly elevates unseen error rates. While seen idioms with morphological variation show 5.5% error rate vs. 0.0% for exact matches, unseen morphologically-variant idioms show 65.0% error rate compared to 100.0% for exact-match unseen idioms. This indicates models struggle to generalize boundary patterns across inflectional paradigms in Hebrew, where a single idiom can appear in dozens of surface forms.
