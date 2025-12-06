# Mission 3.4: Zero-Shot Results Analysis

**Generated:** 2025-12-06 13:11

## 1. Executive Summary

This analysis covers the zero-shot evaluation of 5 models across two datasets (Seen 'test' and Unseen 'unseen_idiom_test').

- **Best Model (Task 1):** bert-base-multilingual-cased (F1: 0.5453)
- **Hebrew vs Multilingual:**
  - Avg Hebrew Model F1: 0.4239
  - Avg Multilingual Model F1: 0.4396
- **Task 2 Performance:** All models achieved ~0.0 F1 for the untrained span detection, establishing a valid lower bound.

## 2. Detailed Results Table

| Model | Dataset | Type | Task 1 Acc | Task 1 F1 | Task 2 Span F1 |
|-------|---------|------|------------|-----------|----------------|
| bert-base-multilingual-cased | test | Multilingual | 0.5023 | 0.5022 | 0.0393 |
| alephbert-base | test | Hebrew | 0.4838 | 0.4703 | 0.0177 |
| alephbertgimmel-base | test | Hebrew | 0.4653 | 0.4583 | 0.0382 |
| xlm-roberta-base | test | Multilingual | 0.5046 | 0.3514 | 0.0155 |
| dictabert | test | Hebrew | 0.4954 | 0.3392 | 0.0180 |
| bert-base-multilingual-cased | unseen_idiom_test | Multilingual | 0.5458 | 0.5453 | 0.0000 |
| alephbert-base | unseen_idiom_test | Hebrew | 0.5229 | 0.4980 | 0.0000 |
| alephbertgimmel-base | unseen_idiom_test | Hebrew | 0.4625 | 0.4545 | 0.0000 |
| xlm-roberta-base | unseen_idiom_test | Multilingual | 0.5104 | 0.3595 | 0.0000 |
| dictabert | unseen_idiom_test | Hebrew | 0.4771 | 0.3230 | 0.0000 |

## 3. Analysis

### 3.1 Comparison: Hebrew vs. Multilingual
- Comparison of average performance shows that pre-training language focus (Hebrew vs Multilingual) does not significantly differ for zero-shot classification using [CLS] prototypes.
- Fine-tuning will be required to see the true benefit of language-specific pre-training.

### 3.2 Task Difficulty
- **Task 1 (Classification):** Models perform around random chance (~50%), indicating that 'Literal' vs 'Figurative' are not linearly separable in the pre-trained embedding space without adaptation.
- **Task 2 (Span Detection):** The untrained baseline is effectively 0%, while the heuristic baseline (string match) is 100%. This massive gap confirms that the task requires learning specific idiom patterns and cannot be solved by the architecture structure alone.

### 3.3 Error Patterns
- **Class Collapse:** Some models (like DictaBERT and XLM-R) exhibited 'class collapse' in zero-shot, predicting one class almost exclusively (visible in Confusion Matrices).
- **Random Guessing:** Other models (mBERT, AlephBERT) showed more balanced but random predictions.

## 4. Conclusion
We have established a rigorous baseline. The 'floor' is set at random chance. Phase 4 (Fine-Tuning) will aim to close the gap towards the theoretical ceiling.
