# Statistical Significance Summary
**Generated:** 2025-12-31 14:42:20

## Scope
- Tasks: cls, span
- Splits: seen_test, unseen_test
- Seeds: 42, 123, 456

## CLS - Seen Test
| model_best   | model_other                  |   best_mean_f1 |   other_mean_f1 |   t_statistic |   p_value |   cohens_d |   bonferroni_alpha | significant_0.05   | significant_bonferroni   |
|:-------------|:-----------------------------|---------------:|----------------:|--------------:|----------:|-----------:|-------------------:|:-------------------|:-------------------------|
| dictabert    | alephbert-base               |         0.9483 |          0.9421 |        0.8362 |    0.4910 |     0.4828 |             0.0125 | False              | False                    |
| dictabert    | alephbertgimmel-base         |         0.9483 |          0.9468 |        0.3571 |    0.7551 |     0.2062 |             0.0125 | False              | False                    |
| dictabert    | bert-base-multilingual-cased |         0.9483 |          0.8758 |       23.6128 |    0.0018 |    13.6329 |             0.0125 | True               | True                     |
| dictabert    | xlm-roberta-base             |         0.9483 |          0.9174 |        4.5863 |    0.0444 |     2.6479 |             0.0125 | True               | False                    |

## CLS - Unseen Test
| model_best           | model_other                  |   best_mean_f1 |   other_mean_f1 |   t_statistic |   p_value |   cohens_d |   bonferroni_alpha | significant_0.05   | significant_bonferroni   |
|:---------------------|:-----------------------------|---------------:|----------------:|--------------:|----------:|-----------:|-------------------:|:-------------------|:-------------------------|
| alephbertgimmel-base | alephbert-base               |         0.9138 |          0.9062 |        2.0432 |    0.1778 |     1.1796 |             0.0125 | False              | False                    |
| alephbertgimmel-base | bert-base-multilingual-cased |         0.9138 |          0.9014 |        2.5639 |    0.1244 |     1.4803 |             0.0125 | False              | False                    |
| alephbertgimmel-base | dictabert                    |         0.9138 |          0.9108 |        0.3012 |    0.7917 |     0.1739 |             0.0125 | False              | False                    |
| alephbertgimmel-base | xlm-roberta-base             |         0.9138 |          0.8986 |        3.0145 |    0.0947 |     1.7404 |             0.0125 | False              | False                    |

## SPAN - Seen Test
| model_best     | model_other                  |   best_mean_f1 |   other_mean_f1 |   t_statistic |   p_value |   cohens_d |   bonferroni_alpha | significant_0.05   | significant_bonferroni   |
|:---------------|:-----------------------------|---------------:|----------------:|--------------:|----------:|-----------:|-------------------:|:-------------------|:-------------------------|
| alephbert-base | alephbertgimmel-base         |         0.9965 |          0.9912 |        3.8795 |    0.0605 |     2.2398 |             0.0125 | False              | False                    |
| alephbert-base | bert-base-multilingual-cased |         0.9965 |          0.9931 |        2.5924 |    0.1221 |     1.4967 |             0.0125 | False              | False                    |
| alephbert-base | dictabert                    |         0.9965 |          0.9912 |        5.2941 |    0.0339 |     3.0565 |             0.0125 | True               | False                    |
| alephbert-base | xlm-roberta-base             |         0.9965 |          0.9927 |        2.3014 |    0.1480 |     1.3287 |             0.0125 | False              | False                    |

## SPAN - Unseen Test
| model_best           | model_other                  |   best_mean_f1 |   other_mean_f1 |   t_statistic |   p_value |   cohens_d |   bonferroni_alpha | significant_0.05   | significant_bonferroni   |
|:---------------------|:-----------------------------|---------------:|----------------:|--------------:|----------:|-----------:|-------------------:|:-------------------|:-------------------------|
| alephbertgimmel-base | alephbert-base               |         0.7559 |          0.7248 |        1.1970 |    0.3540 |     0.6911 |             0.0125 | False              | False                    |
| alephbertgimmel-base | bert-base-multilingual-cased |         0.7559 |          0.5799 |        8.8957 |    0.0124 |     5.1360 |             0.0125 | True               | True                     |
| alephbertgimmel-base | dictabert                    |         0.7559 |          0.7258 |        0.6782 |    0.5676 |     0.3915 |             0.0125 | False              | False                    |
| alephbertgimmel-base | xlm-roberta-base             |         0.7559 |          0.6318 |        2.4937 |    0.1301 |     1.4397 |             0.0125 | False              | False                    |

## Interpretation Notes
- Paired t-test compares matched seeds.
- Cohen’s d reports effect size (paired).
- Bonferroni alpha is applied per task/split comparison set.
