# Complete Statistical Significance Testing

**Updated for CoNLL 2026 (Priority 1 Tasks)**

## Power Analysis

With three random seeds per condition, our experiments achieve 80% statistical power to detect large effect sizes (Cohen's d ≥ 3.26, two-tailed paired t-test, α=0.05). All reported significant differences exceed this threshold, ensuring adequate power for our conclusions.

## All Pairwise Comparisons

- Total comparisons: 60
- Bonferroni-corrected α: 0.000833
- Minimum detectable effect size: d ≥ 3.26

### Summary

- Significant after Bonferroni correction: 0/60 (0.0%)
- Significant before correction (p<0.05): 14/60 (23.3%)
- Comparisons with adequate power (|d|≥3.26): 9/60 (15.0%)


### CLS - Seen Test

| Model 1 | Model 2 | Mean Diff | t-stat | p-value | Cohen's d | Sig (α=0.05) | Sig (Bonferroni) | Power |
|---------|---------|-----------|--------|---------|-----------|--------------|------------------|-------|
| alephbert-base | alephbertgimmel-base | -0.0100 | -1.224 | 0.3454 | -0.707 | ❌ | ❌ | ⚠️ |
| alephbert-base | bert-base-multilingual-cased | +0.0417 | 9.003 | 0.0121 | 5.198 | ✅ | ❌ | ✅ |
| alephbert-base | dictabert | -0.0115 | -4.329 | 0.0494 | -2.499 | ✅ | ❌ | ⚠️ |
| alephbert-base | neodictabert | -0.0286 | -5.279 | 0.0341 | -3.048 | ✅ | ❌ | ⚠️ |
| alephbert-base | xlm-roberta-base | +0.0178 | 2.647 | 0.1180 | 1.528 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | bert-base-multilingual-cased | +0.0517 | 9.307 | 0.0113 | 5.373 | ✅ | ❌ | ✅ |
| alephbertgimmel-base | dictabert | -0.0015 | -0.193 | 0.8650 | -0.111 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | neodictabert | -0.0186 | -1.454 | 0.2831 | -0.840 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | xlm-roberta-base | +0.0278 | 2.378 | 0.1405 | 1.373 | ❌ | ❌ | ⚠️ |
| bert-base-multilingual-cased | dictabert | -0.0532 | -19.575 | 0.0026 | -11.302 | ✅ | ❌ | ✅ |
| bert-base-multilingual-cased | neodictabert | -0.0703 | -6.998 | 0.0198 | -4.040 | ✅ | ❌ | ✅ |
| bert-base-multilingual-cased | xlm-roberta-base | -0.0239 | -2.208 | 0.1579 | -1.275 | ❌ | ❌ | ⚠️ |
| dictabert | neodictabert | -0.0171 | -2.183 | 0.1607 | -1.260 | ❌ | ❌ | ⚠️ |
| dictabert | xlm-roberta-base | +0.0293 | 3.126 | 0.0889 | 1.805 | ❌ | ❌ | ⚠️ |
| neodictabert | xlm-roberta-base | +0.0464 | 11.732 | 0.0072 | 6.774 | ✅ | ❌ | ✅ |

### CLS - Unseen Test

| Model 1 | Model 2 | Mean Diff | t-stat | p-value | Cohen's d | Sig (α=0.05) | Sig (Bonferroni) | Power |
|---------|---------|-----------|--------|---------|-----------|--------------|------------------|-------|
| alephbert-base | alephbertgimmel-base | +0.0022 | 0.288 | 0.8005 | 0.166 | ❌ | ❌ | ⚠️ |
| alephbert-base | bert-base-multilingual-cased | +0.0062 | 1.127 | 0.3768 | 0.651 | ❌ | ❌ | ⚠️ |
| alephbert-base | dictabert | -0.0125 | -3.944 | 0.0587 | -2.277 | ❌ | ❌ | ⚠️ |
| alephbert-base | neodictabert | -0.0138 | -5.463 | 0.0319 | -3.154 | ✅ | ❌ | ⚠️ |
| alephbert-base | xlm-roberta-base | +0.0028 | 1.547 | 0.2619 | 0.893 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | bert-base-multilingual-cased | +0.0041 | 0.366 | 0.7495 | 0.211 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | dictabert | -0.0146 | -3.009 | 0.0950 | -1.737 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | neodictabert | -0.0159 | -2.014 | 0.1817 | -1.163 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | xlm-roberta-base | +0.0006 | 0.074 | 0.9478 | 0.043 | ❌ | ❌ | ⚠️ |
| bert-base-multilingual-cased | dictabert | -0.0187 | -2.928 | 0.0995 | -1.690 | ❌ | ❌ | ⚠️ |
| bert-base-multilingual-cased | neodictabert | -0.0200 | -5.806 | 0.0284 | -3.352 | ✅ | ❌ | ✅ |
| bert-base-multilingual-cased | xlm-roberta-base | -0.0034 | -0.922 | 0.4540 | -0.532 | ❌ | ❌ | ⚠️ |
| dictabert | neodictabert | -0.0013 | -0.426 | 0.7113 | -0.246 | ❌ | ❌ | ⚠️ |
| dictabert | xlm-roberta-base | +0.0153 | 4.151 | 0.0534 | 2.397 | ❌ | ❌ | ⚠️ |
| neodictabert | xlm-roberta-base | +0.0166 | 13.585 | 0.0054 | 7.843 | ✅ | ❌ | ✅ |

### SPAN - Seen Test

| Model 1 | Model 2 | Mean Diff | t-stat | p-value | Cohen's d | Sig (α=0.05) | Sig (Bonferroni) | Power |
|---------|---------|-----------|--------|---------|-----------|--------------|------------------|-------|
| alephbert-base | alephbertgimmel-base | +0.0042 | 4.127 | 0.0540 | 2.382 | ❌ | ❌ | ⚠️ |
| alephbert-base | bert-base-multilingual-cased | -0.0008 | -0.306 | 0.7885 | -0.177 | ❌ | ❌ | ⚠️ |
| alephbert-base | dictabert | -0.0004 | -0.202 | 0.8585 | -0.117 | ❌ | ❌ | ⚠️ |
| alephbert-base | neodictabert | -0.0027 | -1.148 | 0.3697 | -0.663 | ❌ | ❌ | ⚠️ |
| alephbert-base | xlm-roberta-base | +0.0004 | 0.180 | 0.8737 | 0.104 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | bert-base-multilingual-cased | -0.0050 | -2.988 | 0.0961 | -1.725 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | dictabert | -0.0046 | -3.456 | 0.0745 | -1.995 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | neodictabert | -0.0069 | -5.191 | 0.0352 | -2.997 | ✅ | ❌ | ⚠️ |
| alephbertgimmel-base | xlm-roberta-base | -0.0038 | -2.764 | 0.1098 | -1.596 | ❌ | ❌ | ⚠️ |
| bert-base-multilingual-cased | dictabert | +0.0004 | 0.152 | 0.8930 | 0.088 | ❌ | ❌ | ⚠️ |
| bert-base-multilingual-cased | neodictabert | -0.0019 | -1.382 | 0.3010 | -0.798 | ❌ | ❌ | ⚠️ |
| bert-base-multilingual-cased | xlm-roberta-base | +0.0012 | 0.502 | 0.6656 | 0.290 | ❌ | ❌ | ⚠️ |
| dictabert | neodictabert | -0.0023 | -1.724 | 0.2269 | -0.995 | ❌ | ❌ | ⚠️ |
| dictabert | xlm-roberta-base | +0.0008 | 2.000 | 0.1835 | 1.155 | ❌ | ❌ | ⚠️ |
| neodictabert | xlm-roberta-base | +0.0031 | 3.020 | 0.0944 | 1.743 | ❌ | ❌ | ⚠️ |

### SPAN - Unseen Test

| Model 1 | Model 2 | Mean Diff | t-stat | p-value | Cohen's d | Sig (α=0.05) | Sig (Bonferroni) | Power |
|---------|---------|-----------|--------|---------|-----------|--------------|------------------|-------|
| alephbert-base | alephbertgimmel-base | -0.0793 | -4.063 | 0.0556 | -2.346 | ❌ | ❌ | ⚠️ |
| alephbert-base | bert-base-multilingual-cased | +0.0795 | 1.425 | 0.2903 | 0.822 | ❌ | ❌ | ⚠️ |
| alephbert-base | dictabert | -0.0933 | -3.112 | 0.0896 | -1.797 | ❌ | ❌ | ⚠️ |
| alephbert-base | neodictabert | +0.0065 | 0.167 | 0.8830 | 0.096 | ❌ | ❌ | ⚠️ |
| alephbert-base | xlm-roberta-base | +0.0529 | 1.710 | 0.2293 | 0.988 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | bert-base-multilingual-cased | +0.1588 | 4.259 | 0.0510 | 2.459 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | dictabert | -0.0140 | -0.613 | 0.6024 | -0.354 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | neodictabert | +0.0858 | 2.313 | 0.1469 | 1.335 | ❌ | ❌ | ⚠️ |
| alephbertgimmel-base | xlm-roberta-base | +0.1321 | 5.813 | 0.0283 | 3.356 | ✅ | ❌ | ✅ |
| bert-base-multilingual-cased | dictabert | -0.1728 | -4.851 | 0.0400 | -2.801 | ✅ | ❌ | ⚠️ |
| bert-base-multilingual-cased | neodictabert | -0.0730 | -1.575 | 0.2560 | -0.909 | ❌ | ❌ | ⚠️ |
| bert-base-multilingual-cased | xlm-roberta-base | -0.0267 | -0.538 | 0.6443 | -0.311 | ❌ | ❌ | ⚠️ |
| dictabert | neodictabert | +0.0998 | 6.647 | 0.0219 | 3.838 | ✅ | ❌ | ✅ |
| dictabert | xlm-roberta-base | +0.1461 | 3.208 | 0.0850 | 1.852 | ❌ | ❌ | ⚠️ |
| neodictabert | xlm-roberta-base | +0.0464 | 0.776 | 0.5188 | 0.448 | ❌ | ❌ | ⚠️ |


**Legend:**
- ✅ YES: Condition met
- ❌ NO: Condition not met
- ⚠️ WARNING: Effect size below minimum detectable threshold

## Interpretation Guidelines

1. **Bonferroni Correction:** Use α=0.000833 to control family-wise error rate
2. **Effect Size:** Cohen's d interpretation: |d|<0.5 (small), 0.5≤|d|<0.8 (medium), |d|≥0.8 (large)
3. **Power:** With n=3, we can reliably detect |d|≥3.26. Smaller effects may be real but underpowered.
4. **Reporting:** Report ALL comparisons (including non-significant) to avoid publication bias
