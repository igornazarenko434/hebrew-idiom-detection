# Comprehensive Fine-Tuning Analysis

## 1. In-Domain Performance (Seen Test)
Performance on idioms seen during training (split by sentences).
Reporting: Mean F1 ± Std (95% Bootstrap CI)

| task   | model                        |   mean |    std | ci               |
|:-------|:-----------------------------|-------:|-------:|:-----------------|
| cls    | neodictabert                 | 0.9583 | 0.0120 | [0.9514, 0.9722] |
| cls    | dictabert                    | 0.9413 | 0.0027 | [0.9397, 0.9444] |
| cls    | alephbertgimmel-base         | 0.9398 | 0.0123 | [0.9305, 0.9537] |
| cls    | alephbert-base               | 0.9298 | 0.0027 | [0.9282, 0.9328] |
| cls    | xlm-roberta-base             | 0.9119 | 0.0139 | [0.8981, 0.9259] |
| cls    | bert-base-multilingual-cased | 0.8880 | 0.0054 | [0.8818, 0.8912] |
| span   | neodictabert                 | 0.9965 | 0.0012 | [0.9954, 0.9977] |
| span   | bert-base-multilingual-cased | 0.9946 | 0.0024 | [0.9919, 0.9965] |
| span   | dictabert                    | 0.9942 | 0.0020 | [0.9931, 0.9965] |
| span   | alephbert-base               | 0.9938 | 0.0029 | [0.9908, 0.9965] |
| span   | xlm-roberta-base             | 0.9935 | 0.0018 | [0.9919, 0.9954] |
| span   | alephbertgimmel-base         | 0.9896 | 0.0012 | [0.9885, 0.9908] |

## 2. Generalization Performance (Unseen Test)
Performance on completely new idioms never seen during training (Zero-Shot Transfer).
Reporting: Mean F1 ± Std (95% Bootstrap CI)

| task   | model                        |   mean |    std | ci               |
|:-------|:-----------------------------|-------:|-------:|:-----------------|
| cls    | neodictabert                 | 0.9235 | 0.0032 | [0.9207, 0.9270] |
| cls    | dictabert                    | 0.9221 | 0.0044 | [0.9186, 0.9271] |
| cls    | alephbert-base               | 0.9097 | 0.0067 | [0.9020, 0.9145] |
| cls    | alephbertgimmel-base         | 0.9075 | 0.0119 | [0.8977, 0.9208] |
| cls    | xlm-roberta-base             | 0.9069 | 0.0053 | [0.9019, 0.9125] |
| cls    | bert-base-multilingual-cased | 0.9035 | 0.0073 | [0.8958, 0.9104] |
| span   | dictabert                    | 0.7610 | 0.0482 | [0.7054, 0.7901] |
| span   | alephbertgimmel-base         | 0.7470 | 0.0151 | [0.7348, 0.7639] |
| span   | alephbert-base               | 0.6677 | 0.0213 | [0.6466, 0.6892] |
| span   | neodictabert                 | 0.6612 | 0.0701 | [0.5831, 0.7188] |
| span   | xlm-roberta-base             | 0.6148 | 0.0363 | [0.5745, 0.6450] |
| span   | bert-base-multilingual-cased | 0.5882 | 0.0797 | [0.5280, 0.6786] |


## Statistical Significance - Seen Test Set

### Task: CLS (Seen)
**Best Model:** neodictabert (Mean F1: 0.9583)
**Bonferroni-corrected α:** 0.0100 (5 comparisons)

| Comparison | T-Stat | P-Value | Bonferroni | Cohen's d | Effect Size | Significant? |
|------------|--------|---------|------------|-----------|-------------|--------------|
| neodictabert vs dictabert | 2.183 | 0.1607 | 0.0100 | 1.958 | Large | ❌ NO |
| neodictabert vs alephbertgimmel-base | 1.454 | 0.2831 | 0.0100 | 1.527 | Large | ❌ NO |
| neodictabert vs alephbert-base | 5.279 | 0.0341 | 0.0100 | 3.279 | Large | ⚠️ YES* |
| neodictabert vs xlm-roberta-base | 11.732 | 0.0072 | 0.0100 | 3.568 | Large | ✅ YES** |
| neodictabert vs bert-base-multilingual-cased | 6.998 | 0.0198 | 0.0100 | 7.547 | Large | ⚠️ YES* |

### Task: SPAN (Seen)
**Best Model:** neodictabert (Mean F1: 0.9965)
**Bonferroni-corrected α:** 0.0100 (5 comparisons)

| Comparison | T-Stat | P-Value | Bonferroni | Cohen's d | Effect Size | Significant? |
|------------|--------|---------|------------|-----------|-------------|--------------|
| neodictabert vs bert-base-multilingual-cased | 1.382 | 0.3010 | 0.0100 | 1.018 | Large | ❌ NO |
| neodictabert vs dictabert | 1.724 | 0.2269 | 0.0100 | 1.410 | Large | ❌ NO |
| neodictabert vs alephbert-base | 1.148 | 0.3697 | 0.0100 | 1.215 | Large | ❌ NO |
| neodictabert vs xlm-roberta-base | 3.020 | 0.0944 | 0.0100 | 2.066 | Large | ❌ NO |
| neodictabert vs alephbertgimmel-base | 5.191 | 0.0352 | 0.0100 | 5.994 | Large | ⚠️ YES* |

**Legend:**
- ✅ YES**: Significant after Bonferroni correction (conservative)
- ⚠️ YES*: Significant without correction (p < 0.05), but NOT after Bonferroni
- ❌ NO: Not significant

## Statistical Significance - Unseen Test Set

### Task: CLS (Unseen)
**Best Model:** neodictabert (Mean F1: 0.9235)
**Bonferroni-corrected α:** 0.0100 (5 comparisons)

| Comparison | T-Stat | P-Value | Bonferroni | Cohen's d | Effect Size | Significant? |
|------------|--------|---------|------------|-----------|-------------|--------------|
| neodictabert vs dictabert | 0.426 | 0.7113 | 0.0100 | 0.340 | Small | ❌ NO |
| neodictabert vs alephbert-base | 5.463 | 0.0319 | 0.0100 | 2.614 | Large | ⚠️ YES* |
| neodictabert vs alephbertgimmel-base | 2.014 | 0.1817 | 0.0100 | 1.823 | Large | ❌ NO |
| neodictabert vs xlm-roberta-base | 13.585 | 0.0054 | 0.0100 | 3.775 | Large | ✅ YES** |
| neodictabert vs bert-base-multilingual-cased | 5.806 | 0.0284 | 0.0100 | 3.538 | Large | ⚠️ YES* |

### Task: SPAN (Unseen)
**Best Model:** dictabert (Mean F1: 0.7610)
**Bonferroni-corrected α:** 0.0100 (5 comparisons)

| Comparison | T-Stat | P-Value | Bonferroni | Cohen's d | Effect Size | Significant? |
|------------|--------|---------|------------|-----------|-------------|--------------|
| dictabert vs alephbertgimmel-base | 0.613 | 0.6024 | 0.0100 | 0.392 | Small | ❌ NO |
| dictabert vs alephbert-base | 3.112 | 0.0896 | 0.0100 | 2.503 | Large | ❌ NO |
| dictabert vs neodictabert | 6.647 | 0.0219 | 0.0100 | 1.658 | Large | ⚠️ YES* |
| dictabert vs xlm-roberta-base | 3.208 | 0.0850 | 0.0100 | 3.424 | Large | ❌ NO |
| dictabert vs bert-base-multilingual-cased | 4.851 | 0.0400 | 0.0100 | 2.624 | Large | ⚠️ YES* |

**Legend:**
- ✅ YES**: Significant after Bonferroni correction (conservative)
- ⚠️ YES*: Significant without correction (p < 0.05), but NOT after Bonferroni
- ❌ NO: Not significant

## 3. Executive Summary
- **Best In-Domain (CLS):** neodictabert (0.9583 ± 0.0120, 95% CI: [0.9514, 0.9722])
- **Best In-Domain (SPAN):** neodictabert (0.9965 ± 0.0012, 95% CI: [0.9954, 0.9977])
- **Best Generalization (CLS):** neodictabert (0.9235 ± 0.0032, 95% CI: [0.9207, 0.9270])
- **Best Generalization (SPAN):** dictabert (0.7610 ± 0.0482, 95% CI: [0.7054, 0.7901])
