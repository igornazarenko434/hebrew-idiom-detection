# Reproducing Experiments

This guide explains how to reproduce all experiments from the paper.

## Hardware Requirements

- **GPU**: NVIDIA RTX 4090 (24 GB VRAM) or equivalent
- **RAM**: 16 GB+ system memory
- **Storage**: 100-150 GB (models, cache, results)
- **Original training infrastructure**: [VAST.ai](https://vast.ai/) cloud GPU instances

## 1. Environment Setup

```bash
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git
cd hebrew-idiom-detection

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Verify Dataset

The dataset is included in the repository:

```bash
ls data/splits/
# Expected: train.csv, validation.csv, test.csv, unseen_idiom_test.csv
```

- **Train**: 3,456 sentences (54 seen idioms)
- **Validation**: 432 sentences (54 seen idioms)
- **Seen Test**: 432 sentences (54 seen idioms)
- **Unseen Test**: 480 sentences (6 held-out idioms)

## 3. Hyperparameter Optimization

We use Optuna with TPE sampling (15 trials per model-task combination):

```bash
python src/idiom_experiment.py \
    --mode hpo \
    --model_id onlplab/alephbert-base \
    --data data/expressions_data_with_splits.csv \
    --task cls \
    --device cuda
```

Repeat for all 6 models x 2 tasks = 12 HPO runs. Or batch:

```bash
bash scripts/run_all_hpo.sh
```

## 4. Full Fine-Tuning (3 Seeds)

```bash
python src/idiom_experiment.py \
    --mode full_finetune \
    --model_id onlplab/alephbert-base \
    --data data/expressions_data_with_splits.csv \
    --task cls \
    --seed 42 \
    --device cuda
```

Run for all 6 models x 2 tasks x 3 seeds (42, 123, 456) = 36 training runs. Or batch:

```bash
bash scripts/run_all_experiments.sh
```

**Models evaluated:**

| Model | HuggingFace ID |
|-------|---------------|
| AlephBERT | `onlplab/alephbert-base` |
| AlephBERTGimmel | `dicta-il/alephbertgimmel-base` |
| DictaBERT | `dicta-il/dictabert` |
| NeoDictaBERT | `dicta-il/neodictabert` |
| mBERT | `bert-base-multilingual-cased` |
| XLM-RoBERTa | `xlm-roberta-base` |

## 5. Evaluation

```bash
bash scripts/run_evaluation_batch.sh
```

This evaluates all trained models on both seen and unseen test sets, producing:
- `experiments/results/evaluation/seen_test/<model>/<task>/seed_<seed>/`
- `experiments/results/evaluation/unseen_test/<model>/<task>/seed_<seed>/`

## 6. Analysis

Run the analysis pipeline to generate all figures and statistics:

```bash
# Core analysis
python src/analyze_finetuning_results.py      # Summary tables + bootstrap CIs
python src/analyze_generalization.py           # Seen vs unseen gap
python src/analyze_error_distribution.py       # Error categorization

# Per-idiom analysis
python scripts/analyze_per_idiom_f1.py         # Per-idiom F1 heatmaps

# Statistical tests
python scripts/statistical_tests.py            # Paired t-tests + Bonferroni

# Interpretability
python scripts/analyze_token_importance.py     # Integrated Gradients + attention
python scripts/analyze_embedding_space.py      # t-SNE visualizations
python scripts/analyze_attention_patterns.py   # Attention head analysis
```

## 7. Expected Output Structure

```
experiments/results/
├── analysis/
│   ├── finetuning_summary.csv          # Aggregate F1/accuracy
│   ├── generalization/                 # Generalization gap analysis
│   ├── error_analysis/                 # Error distribution
│   ├── per_idiom_f1/                   # Per-idiom difficulty ranking
│   ├── statistical_tests/              # Paired t-tests
│   ├── token_importance/               # IG + attention heatmaps
│   ├── embedding_space/                # t-SNE plots
│   └── ...
├── evaluation/
│   ├── seen_test/                      # 36 evaluation result sets
│   └── unseen_test/                    # 36 evaluation result sets
└── full_fine-tuning/                   # 36 trained model checkpoints
```

## Pre-Computed Results

All experiment results are included in this repository under `experiments/results/`. You can skip steps 3-5 and go directly to step 6 (analysis) if you want to verify our analysis pipeline without retraining.

## Troubleshooting

- **CUDA out of memory**: Reduce batch size in `experiments/configs/training_config.yaml`
- **Model download fails**: Ensure HuggingFace Hub access; some models require `trust_remote_code=True`
- **NeoDictaBERT tokenizer issues**: The code includes automatic fallback handling (see recent commits)
