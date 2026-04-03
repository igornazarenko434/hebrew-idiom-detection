# Statistical Significance Testing (Best Model Comparisons)

Comparing best model vs. all others with Bonferroni correction and Cohen's d.

**Note:** This report shows only best-model comparisons. See `paired_ttests_complete.md` for ALL pairwise comparisons.

## Task: CLS

### Seen Test
**Best Model:** neodictabert

**Bonferroni α:** 0.010000

| Comparison | t-stat | p-value | Bonferroni | Cohen's d | Significant |
|------------|--------|---------|------------|-----------|-------------|
| neodictabert vs alephbert-base | 5.279 | 0.0341 | 0.0100 | 3.048 | ❌ NO |
| neodictabert vs alephbertgimmel-base | 1.454 | 0.2831 | 0.0100 | 0.840 | ❌ NO |
| neodictabert vs bert-base-multilingual-cased | 6.998 | 0.0198 | 0.0100 | 4.040 | ❌ NO |
| neodictabert vs dictabert | 2.183 | 0.1607 | 0.0100 | 1.260 | ❌ NO |
| neodictabert vs xlm-roberta-base | 11.732 | 0.0072 | 0.0100 | 6.774 | ✅ YES |

### Unseen Test
**Best Model:** neodictabert

**Bonferroni α:** 0.010000

| Comparison | t-stat | p-value | Bonferroni | Cohen's d | Significant |
|------------|--------|---------|------------|-----------|-------------|
| neodictabert vs alephbert-base | 5.463 | 0.0319 | 0.0100 | 3.154 | ❌ NO |
| neodictabert vs alephbertgimmel-base | 2.014 | 0.1817 | 0.0100 | 1.163 | ❌ NO |
| neodictabert vs bert-base-multilingual-cased | 5.806 | 0.0284 | 0.0100 | 3.352 | ❌ NO |
| neodictabert vs dictabert | 0.426 | 0.7113 | 0.0100 | 0.246 | ❌ NO |
| neodictabert vs xlm-roberta-base | 13.585 | 0.0054 | 0.0100 | 7.843 | ✅ YES |

## Task: SPAN

### Seen Test
**Best Model:** neodictabert

**Bonferroni α:** 0.010000

| Comparison | t-stat | p-value | Bonferroni | Cohen's d | Significant |
|------------|--------|---------|------------|-----------|-------------|
| neodictabert vs alephbert-base | 1.148 | 0.3697 | 0.0100 | 0.663 | ❌ NO |
| neodictabert vs alephbertgimmel-base | 5.191 | 0.0352 | 0.0100 | 2.997 | ❌ NO |
| neodictabert vs bert-base-multilingual-cased | 1.382 | 0.3010 | 0.0100 | 0.798 | ❌ NO |
| neodictabert vs dictabert | 1.724 | 0.2269 | 0.0100 | 0.995 | ❌ NO |
| neodictabert vs xlm-roberta-base | 3.020 | 0.0944 | 0.0100 | 1.743 | ❌ NO |

### Unseen Test
**Best Model:** dictabert

**Bonferroni α:** 0.010000

| Comparison | t-stat | p-value | Bonferroni | Cohen's d | Significant |
|------------|--------|---------|------------|-----------|-------------|
| dictabert vs alephbert-base | 3.112 | 0.0896 | 0.0100 | 1.797 | ❌ NO |
| dictabert vs alephbertgimmel-base | 0.613 | 0.6024 | 0.0100 | 0.354 | ❌ NO |
| dictabert vs bert-base-multilingual-cased | 4.851 | 0.0400 | 0.0100 | 2.801 | ❌ NO |
| dictabert vs neodictabert | 6.647 | 0.0219 | 0.0100 | 3.838 | ❌ NO |
| dictabert vs xlm-roberta-base | 3.208 | 0.0850 | 0.0100 | 1.852 | ❌ NO |

