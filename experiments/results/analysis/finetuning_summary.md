# Comprehensive Fine-Tuning Analysis

## 1. In-Domain Performance (Seen Test)
Performance on idioms seen during training (split by sentences).

| task   | model                        |   mean |    std |
|:-------|:-----------------------------|-------:|-------:|
| cls    | dictabert                    | 0.9483 | 0.0027 |
| cls    | alephbertgimmel-base         | 0.9468 | 0.0101 |
| cls    | alephbert-base               | 0.9421 | 0.0106 |
| cls    | xlm-roberta-base             | 0.9174 | 0.0142 |
| cls    | bert-base-multilingual-cased | 0.8758 | 0.0071 |
| span   | alephbert-base               | 0.9965 | 0.0011 |
| span   | bert-base-multilingual-cased | 0.9931 | 0.0020 |
| span   | xlm-roberta-base             | 0.9927 | 0.0024 |
| span   | alephbertgimmel-base         | 0.9912 | 0.0013 |
| span   | dictabert                    | 0.9912 | 0.0007 |

## 2. Generalization Performance (Unseen Test)
Performance on completely new idioms never seen during training (Zero-Shot Transfer).

| task   | model                        |   mean |    std |
|:-------|:-----------------------------|-------:|-------:|
| cls    | alephbertgimmel-base         | 0.9138 | 0.0048 |
| cls    | dictabert                    | 0.9108 | 0.0136 |
| cls    | alephbert-base               | 0.9062 | 0.0112 |
| cls    | bert-base-multilingual-cased | 0.9014 | 0.0048 |
| cls    | xlm-roberta-base             | 0.8986 | 0.0087 |
| span   | alephbertgimmel-base         | 0.7559 | 0.0140 |
| span   | dictabert                    | 0.7258 | 0.0897 |
| span   | alephbert-base               | 0.7248 | 0.0311 |
| span   | xlm-roberta-base             | 0.6318 | 0.0813 |
| span   | bert-base-multilingual-cased | 0.5799 | 0.0427 |


## Statistical Significance - Seen Test Set

### Task: CLS (Seen)
**Best Model:** dictabert (Mean F1: 0.9483)
**Bonferroni-corrected α:** 0.0125 (4 comparisons)

| Comparison | T-Stat | P-Value | Bonferroni | Cohen's d | Effect Size | Significant? |
|------------|--------|---------|------------|-----------|-------------|--------------|
| dictabert vs alephbertgimmel-base | 0.357 | 0.7551 | 0.0125 | 0.208 | Small | ❌ NO |
| dictabert vs alephbert-base | 0.836 | 0.4910 | 0.0125 | 0.797 | Medium | ❌ NO |
| dictabert vs xlm-roberta-base | 4.586 | 0.0444 | 0.0125 | 3.030 | Large | ⚠️ YES* |
| dictabert vs bert-base-multilingual-cased | 23.613 | 0.0018 | 0.0125 | 13.554 | Large | ✅ YES** |

### Task: SPAN (Seen)
**Best Model:** alephbert-base (Mean F1: 0.9965)
**Bonferroni-corrected α:** 0.0125 (4 comparisons)

| Comparison | T-Stat | P-Value | Bonferroni | Cohen's d | Effect Size | Significant? |
|------------|--------|---------|------------|-----------|-------------|--------------|
| alephbert-base vs bert-base-multilingual-cased | 2.592 | 0.1221 | 0.0125 | 2.122 | Large | ❌ NO |
| alephbert-base vs xlm-roberta-base | 2.301 | 0.1480 | 0.0125 | 2.046 | Large | ❌ NO |
| alephbert-base vs alephbertgimmel-base | 3.880 | 0.0605 | 0.0125 | 4.317 | Large | ❌ NO |
| alephbert-base vs dictabert | 5.294 | 0.0339 | 0.0125 | 5.725 | Large | ⚠️ YES* |

**Legend:**
- ✅ YES**: Significant after Bonferroni correction (conservative)
- ⚠️ YES*: Significant without correction (p < 0.05), but NOT after Bonferroni
- ❌ NO: Not significant

## Statistical Significance - Unseen Test Set

### Task: CLS (Unseen)
**Best Model:** alephbertgimmel-base (Mean F1: 0.9138)
**Bonferroni-corrected α:** 0.0125 (4 comparisons)

| Comparison | T-Stat | P-Value | Bonferroni | Cohen's d | Effect Size | Significant? |
|------------|--------|---------|------------|-----------|-------------|--------------|
| alephbertgimmel-base vs dictabert | 0.301 | 0.7917 | 0.0125 | 0.291 | Small | ❌ NO |
| alephbertgimmel-base vs alephbert-base | 2.043 | 0.1778 | 0.0125 | 0.885 | Large | ❌ NO |
| alephbertgimmel-base vs bert-base-multilingual-cased | 2.564 | 0.1244 | 0.0125 | 2.564 | Large | ❌ NO |
| alephbertgimmel-base vs xlm-roberta-base | 3.015 | 0.0947 | 0.0125 | 2.156 | Large | ❌ NO |

### Task: SPAN (Unseen)
**Best Model:** alephbertgimmel-base (Mean F1: 0.7559)
**Bonferroni-corrected α:** 0.0125 (4 comparisons)

| Comparison | T-Stat | P-Value | Bonferroni | Cohen's d | Effect Size | Significant? |
|------------|--------|---------|------------|-----------|-------------|--------------|
| alephbertgimmel-base vs dictabert | 0.678 | 0.5676 | 0.0125 | 0.469 | Small | ❌ NO |
| alephbertgimmel-base vs alephbert-base | 1.197 | 0.3540 | 0.0125 | 1.289 | Large | ❌ NO |
| alephbertgimmel-base vs xlm-roberta-base | 2.494 | 0.1301 | 0.0125 | 2.127 | Large | ❌ NO |
| alephbertgimmel-base vs bert-base-multilingual-cased | 8.896 | 0.0124 | 0.0125 | 5.537 | Large | ✅ YES** |

**Legend:**
- ✅ YES**: Significant after Bonferroni correction (conservative)
- ⚠️ YES*: Significant without correction (p < 0.05), but NOT after Bonferroni
- ❌ NO: Not significant

## 3. Executive Summary
- **Best In-Domain (CLS):** dictabert (0.9483 ± 0.0027)
- **Best In-Domain (SPAN):** alephbert-base (0.9965 ± 0.0011)
- **Best Generalization (CLS):** alephbertgimmel-base (0.9138 ± 0.0048)
- **Best Generalization (SPAN):** alephbertgimmel-base (0.7559 ± 0.0140)
