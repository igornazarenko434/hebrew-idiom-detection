# Generalization Analysis (Seen vs Unseen)

|                                          |   Seen |   Unseen |   gap_absolute |   gap_percent |
|:-----------------------------------------|-------:|---------:|---------------:|--------------:|
| ('alephbertgimmel-base', 'cls')          | 0.9468 |   0.9138 |         0.0330 |        3.4831 |
| ('dictabert', 'cls')                     | 0.9483 |   0.9108 |         0.0375 |        3.9527 |
| ('alephbert-base', 'cls')                | 0.9421 |   0.9062 |         0.0360 |        3.8168 |
| ('bert-base-multilingual-cased', 'cls')  | 0.8758 |   0.9014 |        -0.0256 |       -2.9259 |
| ('xlm-roberta-base', 'cls')              | 0.9174 |   0.8986 |         0.0188 |        2.0495 |
| ('alephbertgimmel-base', 'span')         | 0.9912 |   0.7559 |         0.2353 |       23.7379 |
| ('dictabert', 'span')                    | 0.9912 |   0.7258 |         0.2654 |       26.7722 |
| ('alephbert-base', 'span')               | 0.9965 |   0.7248 |         0.2717 |       27.2661 |
| ('xlm-roberta-base', 'span')             | 0.9927 |   0.6318 |         0.3609 |       36.3551 |
| ('bert-base-multilingual-cased', 'span') | 0.9931 |   0.5799 |         0.4132 |       41.6041 |

**Note:** 'Gap' is the performance drop. Lower gap means better robustness.