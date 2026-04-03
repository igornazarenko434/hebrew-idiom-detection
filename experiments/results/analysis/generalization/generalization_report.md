# Generalization Analysis (Seen vs Unseen)

|                                          |   Seen |   Unseen |   gap_absolute |   gap_percent |
|:-----------------------------------------|-------:|---------:|---------------:|--------------:|
| ('neodictabert', 'cls')                  | 0.9583 |   0.9235 |         0.0349 |        3.6394 |
| ('dictabert', 'cls')                     | 0.9413 |   0.9221 |         0.0191 |        2.0322 |
| ('alephbert-base', 'cls')                | 0.9298 |   0.9097 |         0.0201 |        2.1599 |
| ('alephbertgimmel-base', 'cls')          | 0.9398 |   0.9075 |         0.0323 |        3.4333 |
| ('xlm-roberta-base', 'cls')              | 0.9119 |   0.9069 |         0.0051 |        0.5551 |
| ('bert-base-multilingual-cased', 'cls')  | 0.8880 |   0.9035 |        -0.0154 |       -1.7366 |
| ('dictabert', 'span')                    | 0.9942 |   0.7610 |         0.2332 |       23.4601 |
| ('alephbertgimmel-base', 'span')         | 0.9896 |   0.7470 |         0.2426 |       24.5187 |
| ('alephbert-base', 'span')               | 0.9938 |   0.6677 |         0.3261 |       32.8155 |
| ('neodictabert', 'span')                 | 0.9965 |   0.6612 |         0.3353 |       33.6471 |
| ('xlm-roberta-base', 'span')             | 0.9935 |   0.6148 |         0.3786 |       38.1107 |
| ('bert-base-multilingual-cased', 'span') | 0.9946 |   0.5882 |         0.4064 |       40.8621 |

**Note:** 'Gap' is the performance drop. Lower gap means better robustness.