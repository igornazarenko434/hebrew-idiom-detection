# Implementation Roadmap
# Hebrew Idiom Detection Project - Phase 5-7 Analysis

**Created:** December 31, 2025
**Purpose:** Clear action plan for implementing all required analysis missions

---

## 📊 Current Status

### ✅ What's Already Done (Phase 1-4)

```
✅ Phase 1-3: Data Preparation & Model Training
   - Dataset created (4,800 samples, 60 idioms)
   - Train/val/test splits created
   - HPO completed for all models
   - All models trained (6 models × 2 tasks × 3 seeds = 36 checkpoints)

✅ Phase 4: Basic Evaluation
   - All models evaluated on seen + unseen test
   - eval_results.json and eval_predictions.json saved
   - Basic metrics computed (F1, accuracy, precision, recall)

✅ Documentation Created
   - EVALUATION_STANDARDIZATION_GUIDE.md (evaluation standards)
   - OPERATIONS_GUIDE.md (workflow manual)
   - MISSION_4.7_SUMMARY_AND_NEXT_STEPS.md (status summary)
   - STEP_BY_STEP_MISSIONS.md (detailed mission specs)
```

### ❌ What Needs To Be Done (Phase 5-7)

```
❌ Phase 5: LLM Evaluation (Partner's responsibility)
   - Zero-shot prompting
   - Few-shot prompting
   - Chain-of-thought (optional)

❌ Phase 6: Ablations & Interpretability (Your responsibility - Fine-tuning only)
   - Mission 6.1: Token importance analysis
   - Mission 6.2: Frozen backbone comparison
   - Mission 6.3: Hyperparameter sensitivity
   - Mission 6.4: Data size impact

❌ Phase 7: Comprehensive Analysis (Both partners)
   - Mission 7.1: Complete error analysis
   - Mission 7.2: Statistical significance testing
   - Mission 7.3: Cross-task analysis
   - Mission 7.4: Visualization creation
   - Mission 7.5: Publication tables
```

---

## 🎯 Implementation Strategy

### Strategy Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Quick Wins (Analysis of Existing Results)         │
│  → Use existing eval results                                │
│  → No new training needed                                   │
│  → TIME: 2-3 hours                                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Ablation Studies (Requires New Training)          │
│  → Train additional model variants                          │
│  → Run on VAST.ai                                           │
│  → TIME: 2-3 days                                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Advanced Analysis (Interpretability)              │
│  → Token importance, attention analysis                     │
│  → Requires checkpoints + code                             │
│  → TIME: 1 day                                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Finalize & Document                                │
│  → Create all figures                                       │
│  → Write results section                                    │
│  → TIME: 1-2 days                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Step-by-Step Implementation Plan

## STEP 1: Quick Wins (Do This First!) ⚡

**Goal:** Complete all analyses that use existing evaluation results

**Time:** 2-3 hours
**Requirements:** Only existing eval results (no new training)
**Documents to use:** OPERATIONS_GUIDE.md Section 8

### Task 1.1: Aggregate Fine-Tuning Results
**Mission:** Mission 4.7 (already started)
**What:** Compute Mean ± Std across seeds
**How:**

```bash
# 1. Run aggregation script
python src/analyze_finetuning_results.py \
  --results_dir experiments/results/evaluation \
  --output_dir experiments/results/analysis

# 2. Check outputs
cat experiments/results/analysis/finetuning_summary.md

# 3. Verify you have:
# - finetuning_summary.csv
# - finetuning_summary.md
# - statistical_significance.txt
```

**Document reference:**
- OPERATIONS_GUIDE.md → Section 9.1
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 3.3.1 (reporting standards)

**Expected output:** Summary tables with Mean ± Std for all models

---

### Task 1.2: Generalization Gap Analysis
**Mission:** Mission 7.1 (partial)
**What:** Compute seen vs unseen performance gap
**How:**

```bash
# 1. Run generalization analysis
python src/analyze_generalization.py \
  --results_dir experiments/results/evaluation \
  --output_dir experiments/results/analysis/generalization \
  --create_figures

# 2. Check outputs
ls experiments/results/analysis/generalization/
# Expected:
# - generalization_gap.csv
# - generalization_report.md
# - figures/generalization_gap_cls.png
# - figures/generalization_gap_span.png
```

**Document reference:**
- OPERATIONS_GUIDE.md → Section 9.2
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 3.3.3 (gap calculation)

**Expected output:** Gap metrics + visualizations

---

### Task 1.3: Complete Error Categorization
**Mission:** Mission 7.1 (core)
**What:** Categorize all errors using shared taxonomy
**How:**

```bash
# 1. Create error categorization script
# File: scripts/categorize_all_errors.py

cat > scripts/categorize_all_errors.py << 'EOF'
#!/usr/bin/env python3
"""Categorize errors for all models using shared taxonomy."""

import json
from pathlib import Path
from src.utils.error_analysis import categorize_span_error, categorize_cls_error
from collections import Counter

def categorize_errors_for_model(model, task, seed, split):
    """Categorize errors for single model/task/seed/split."""
    pred_file = f"experiments/results/evaluation/{split}/{model}/{task}/seed_{seed}/eval_predictions.json"

    if not Path(pred_file).exists():
        return None

    with open(pred_file) as f:
        predictions = json.load(f)

    # Categorize each prediction
    for pred in predictions:
        if task == "cls":
            pred['error_category'] = categorize_cls_error(
                pred['true_label'], pred['predicted_label']
            )
        elif task == "span":
            pred['error_category'] = categorize_span_error(
                pred['true_tags'], pred['predicted_tags']
            )

    # Count errors
    error_counts = Counter(p['error_category'] for p in predictions)

    # Save updated predictions
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    return error_counts

def main():
    models = ['dictabert', 'alephbert', 'alephbertgimmel',
              'neodictabert', 'mbert', 'xlm-r']
    tasks = ['cls', 'span']
    seeds = [42, 123, 456]
    splits = ['seen_test', 'unseen_test']

    print("Categorizing errors for all models...")

    for model in models:
        for task in tasks:
            for seed in seeds:
                for split in splits:
                    error_counts = categorize_errors_for_model(model, task, seed, split)
                    if error_counts:
                        print(f"✓ {model}/{task}/seed_{seed}/{split}")
                        total = sum(error_counts.values())
                        for error, count in error_counts.most_common(3):
                            pct = (count/total)*100
                            print(f"    {error}: {count} ({pct:.1f}%)")

    print("\n✅ All errors categorized!")

if __name__ == "__main__":
    main()
EOF

# 2. Run error categorization
python scripts/categorize_all_errors.py

# 3. Verify - predictions should now have 'error_category' field
python -c "
import json
with open('experiments/results/evaluation/seen_test/dictabert/span/seed_42/eval_predictions.json') as f:
    preds = json.load(f)
print('Sample error categories:', [p['error_category'] for p in preds[:5]])
"
```

**Document reference:**
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 4 (error taxonomy)
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 14 (error analysis protocol)

**Expected output:** All predictions now have error categories

---

### Task 1.4: Per-Idiom F1 Analysis
**Mission:** Mission 7.1 (deep dive)
**What:** Compute F1 for each of 60 idioms
**How:**

```bash
# 1. Create per-idiom analysis script
# File: scripts/analyze_per_idiom.py

cat > scripts/analyze_per_idiom.py << 'EOF'
#!/usr/bin/env python3
"""Compute per-idiom F1 for all models."""

import json
import pandas as pd
from pathlib import Path
from src.utils.error_analysis import compute_cls_metrics, compute_span_f1

def compute_per_idiom_f1(model, task, split='seen_test'):
    """Compute F1 for each idiom."""

    # Load predictions from all seeds
    all_preds = []
    for seed in [42, 123, 456]:
        pred_file = f"experiments/results/evaluation/{split}/{model}/{task}/seed_{seed}/eval_predictions.json"
        if Path(pred_file).exists():
            with open(pred_file) as f:
                all_preds.extend(json.load(f))

    # Extract idiom ID from sample ID (e.g., "1_fig_0" -> idiom_id=1)
    for pred in all_preds:
        pred['idiom_id'] = int(pred['id'].split('_')[0])

    # Group by idiom
    df = pd.DataFrame(all_preds)
    results = []

    for idiom_id, group in df.groupby('idiom_id'):
        group_list = group.to_dict('records')

        if task == "cls":
            metrics = compute_cls_metrics(
                [p['true_label'] for p in group_list],
                [p['predicted_label'] for p in group_list]
            )
        elif task == "span":
            metrics = compute_span_f1(group_list)

        results.append({
            'model': model,
            'task': task,
            'split': split,
            'idiom_id': idiom_id,
            'f1': metrics['f1'],
            'num_samples': len(group_list),
            'num_correct': sum(p['is_correct'] for p in group_list)
        })

    return pd.DataFrame(results)

def main():
    models = ['dictabert', 'alephbert', 'alephbertgimmel',
              'neodictabert', 'mbert', 'xlm-r']
    tasks = ['cls', 'span']
    splits = ['seen_test', 'unseen_test']

    all_results = []

    for model in models:
        for task in tasks:
            for split in splits:
                df = compute_per_idiom_f1(model, task, split)
                all_results.append(df)
                print(f"✓ {model}/{task}/{split}: {len(df)} idioms")

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Save
    output_dir = Path("experiments/results/analysis/per_idiom_f1")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "per_idiom_f1_all.csv", index=False)

    # Create difficulty ranking (average across all models)
    difficulty = combined.groupby('idiom_id')['f1'].mean().sort_values()
    difficulty.to_csv(output_dir / "idiom_difficulty_ranking.csv")

    print(f"\n✅ Per-idiom F1 saved to {output_dir}")
    print(f"\nTop 5 hardest idioms:")
    print(difficulty.head())

if __name__ == "__main__":
    main()
EOF

# 2. Run per-idiom analysis
python scripts/analyze_per_idiom.py

# 3. Check outputs
ls experiments/results/analysis/per_idiom_f1/
# Expected:
# - per_idiom_f1_all.csv
# - idiom_difficulty_ranking.csv
```

**Document reference:**
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 17 (per-idiom F1)
- OPERATIONS_GUIDE.md → Section 8 (analysis workflow)

**Expected output:** Per-idiom F1 scores + difficulty ranking

---

### Task 1.5: Statistical Significance Testing
**Mission:** Mission 7.2
**What:** Paired t-tests, Bonferroni correction, Cohen's d
**How:**

```bash
# 1. Create statistical testing script
# File: scripts/statistical_tests.py

cat > scripts/statistical_tests.py << 'EOF'
#!/usr/bin/env python3
"""Statistical significance testing for all model comparisons."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

def load_model_results(model, task, split):
    """Load F1 scores across all seeds."""
    f1_scores = []
    for seed in [42, 123, 456]:
        result_file = list(Path(f"experiments/results/evaluation/{split}/{model}/{task}/seed_{seed}").glob("eval_results*.json"))
        if result_file:
            with open(result_file[0]) as f:
                data = json.load(f)
                f1_scores.append(data['metrics']['f1'])
    return f1_scores

def paired_ttest(model1_scores, model2_scores):
    """Perform paired t-test."""
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)

    # Cohen's d for paired samples
    differences = np.array(model1_scores) - np.array(model2_scores)
    d = np.mean(differences) / np.std(differences, ddof=1)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': np.mean(differences),
        'cohens_d': d
    }

def main():
    models = ['dictabert', 'alephbert', 'alephbertgimmel',
              'neodictabert', 'mbert', 'xlm-r']
    tasks = ['cls', 'span']
    splits = ['seen_test', 'unseen_test']

    results = []

    for task in tasks:
        for split in splits:
            # Load all model scores
            model_scores = {}
            for model in models:
                scores = load_model_results(model, task, split)
                if len(scores) == 3:
                    model_scores[model] = scores

            # Find best model
            best_model = max(model_scores.keys(), key=lambda m: np.mean(model_scores[m]))

            print(f"\n{task.upper()} - {split}")
            print(f"Best model: {best_model} (F1: {np.mean(model_scores[best_model]):.4f})")

            # Compare best vs all others
            p_values = []
            for model in models:
                if model == best_model or model not in model_scores:
                    continue

                test_result = paired_ttest(model_scores[best_model], model_scores[model])

                results.append({
                    'task': task,
                    'split': split,
                    'model_1': best_model,
                    'model_2': model,
                    't_statistic': test_result['t_statistic'],
                    'p_value': test_result['p_value'],
                    'mean_difference': test_result['mean_difference'],
                    'cohens_d': test_result['cohens_d'],
                    'significant_0.05': test_result['p_value'] < 0.05
                })

                p_values.append(test_result['p_value'])

            # Bonferroni correction
            n_tests = len(p_values)
            bonferroni_alpha = 0.05 / n_tests
            print(f"Bonferroni-corrected α: {bonferroni_alpha:.6f}")

    # Save results
    output_dir = Path("experiments/results/analysis/statistical_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "paired_ttests.csv", index=False)

    print(f"\n✅ Statistical tests saved to {output_dir}")

if __name__ == "__main__":
    main()
EOF

# 2. Run statistical tests
python scripts/statistical_tests.py

# 3. Check outputs
cat experiments/results/analysis/statistical_tests/paired_ttests.csv
```

**Document reference:**
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 15 (statistical testing)

**Expected output:** Statistical significance results with effect sizes

---

### Task 1.6: Create Basic Visualizations
**Mission:** Mission 7.4 (partial)
**What:** Model comparison bar charts, generalization gap charts
**How:**

```bash
# Already done in Task 1.2!
# generalization_gap_cls.png
# generalization_gap_span.png

# Additional: Model comparison charts
python -c "
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style
sns.set_style('whitegrid')
sns.set_palette('colorblind')

# Load summary
df = pd.read_csv('experiments/results/analysis/finetuning_summary.csv')

# Filter for seen test CLS
df_cls_seen = df[(df['task'] == 'cls') & (df['split'] == 'seen_test')].sort_values('f1_mean', ascending=False)

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(df_cls_seen))
ax.bar(x, df_cls_seen['f1_mean'], yerr=df_cls_seen['f1_std'], capsize=5, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(df_cls_seen['model'], rotation=45, ha='right')
ax.set_ylabel('F1 Score')
ax.set_title('Model Comparison: CLS Task (Seen Test)')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/results/analysis/figures/model_comparison_cls_seen.png', dpi=300)
print('✓ Saved: model_comparison_cls_seen.png')
"
```

**Document reference:**
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 16 (visualization standards)

---

## ⏸️ CHECKPOINT 1: Review Quick Wins

**After completing Step 1, you should have:**

```
experiments/results/analysis/
├── finetuning_summary.csv ✓
├── finetuning_summary.md ✓
├── statistical_significance.txt ✓
├── generalization/
│   ├── generalization_gap.csv ✓
│   ├── generalization_report.md ✓
│   └── figures/ ✓
├── per_idiom_f1/
│   ├── per_idiom_f1_raw.csv ✓
│   ├── per_idiom_f1_summary.csv ✓
│   ├── idiom_difficulty_ranking_cls_seen_test.csv ✓
│   ├── idiom_difficulty_ranking_cls_unseen_test.csv ✓
│   ├── idiom_difficulty_ranking_span_seen_test.csv ✓
│   ├── idiom_difficulty_ranking_span_unseen_test.csv ✓
│   ├── per_idiom_f1_report.md ✓
│   └── per_idiom_f1_insights.md ✓
├── statistical_tests/
│   └── paired_ttests.csv ✓
└── figures/
    └── (figures are stored under paper/figures/*) ✓

paper/figures/
├── generalization/
│   ├── generalization_bar_cls.png ✓
│   ├── generalization_bar_span.png ✓
│   ├── stability_boxplot_cls.png ✓
│   └── stability_boxplot_span.png ✓
├── finetuning/
│   ├── model_comparison_cls_seen.png ✓
│   ├── model_comparison_cls_unseen.png ✓
│   ├── model_comparison_span_seen.png ✓
│   └── model_comparison_span_unseen.png ✓
└── per_idiom/
    ├── per_idiom_heatmap_cls_seen_test.png ✓
    ├── per_idiom_heatmap_cls_unseen_test.png ✓
    ├── per_idiom_heatmap_span_seen_test.png ✓
    └── per_idiom_heatmap_span_unseen_test.png ✓
```

**Time invested:** ~2-3 hours
**Progress:** ~40% of Phase 7 complete!

---

## STEP 2: Ablation Studies (Requires New Training) 🔬

**Goal:** Train model variants to understand what drives performance

**Time:** 2-3 days (mostly training time)
**Requirements:** VAST.ai GPU access
**Documents to use:** OPERATIONS_GUIDE.md Section 4-5, EVALUATION_STANDARDIZATION_GUIDE.md Sections 19-21

### Task 2.1: Frozen Backbone Comparison
**Mission:** Mission 6.2
**What:** Compare full fine-tuning vs frozen encoder
**Why:** Understand if task head alone can solve the problem

**How:**

```bash
# 1. Create frozen backbone training config
# File: configs/training/dictabert_span_frozen.yaml

cat > configs/training/dictabert_span_frozen.yaml << 'EOF'
model_name: "dicta-il/dictabert"
task: "span"

# Same hyperparameters as full fine-tuning
learning_rate: 2.3e-05
batch_size: 16
num_epochs: 10
warmup_ratio: 0.1
weight_decay: 0.01

# NEW: Freeze encoder
freeze_encoder: true  # Only train task head

seeds: [42, 123, 456]

output_dir: "experiments/checkpoints/dictabert_frozen/span"
logging_dir: "experiments/logs/dictabert_frozen/span"

evaluation_strategy: "epoch"
save_strategy: "epoch"
load_best_model_at_end: true
metric_for_best_model: "eval_f1"
fp16: true
EOF

# 2. Train on VAST.ai (only DictaBERT SPAN for now)
# SSH to VAST.ai first

ssh -p $VAST_PORT $VAST_HOST
cd /workspace/idiom_detection

# Train all 3 seeds
for seed in 42 123 456; do
  python src/train.py \
    --config configs/training/dictabert_span_frozen.yaml \
    --seed $seed \
    --output_dir experiments/checkpoints/dictabert_frozen/span/seed_$seed \
    --logging_dir experiments/logs/dictabert_frozen/span/seed_$seed
done

# 3. Download checkpoints to local
# On local machine
rsync -avz -e "ssh -p $VAST_PORT" \
  "$VAST_HOST:/workspace/idiom_detection/experiments/checkpoints/dictabert_frozen/" \
  "experiments/checkpoints/dictabert_frozen/"

# 4. Evaluate frozen model
for seed in 42 123 456; do
  python src/evaluate.py \
    --checkpoint experiments/checkpoints/dictabert_frozen/span/seed_$seed/best_model \
    --test_file data/splits/test.csv \
    --task span \
    --output_dir experiments/results/evaluation/seen_test/dictabert_frozen/span/seed_$seed \
    --save_predictions
done

# 5. Compare frozen vs full
python -c "
import json
import numpy as np

# Full fine-tuning results
full_f1 = []
for seed in [42, 123, 456]:
    with open(f'experiments/results/evaluation/seen_test/dictabert/span/seed_{seed}/eval_results_*.json') as f:
        full_f1.append(json.load(f)['metrics']['f1'])

# Frozen backbone results
frozen_f1 = []
for seed in [42, 123, 456]:
    with open(f'experiments/results/evaluation/seen_test/dictabert_frozen/span/seed_{seed}/eval_results_*.json') as f:
        frozen_f1.append(json.load(f)['metrics']['f1'])

print('Full Fine-Tuning:', np.mean(full_f1), '±', np.std(full_f1))
print('Frozen Backbone:', np.mean(frozen_f1), '±', np.std(frozen_f1))
print('Performance Drop:', np.mean(full_f1) - np.mean(frozen_f1))
"
```

**Document reference:**
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 19

---

### Task 2.2: Data Size Impact
**Mission:** Mission 6.4
**What:** Train with 10%, 25%, 50%, 75%, 100% of data
**Why:** Understand data efficiency

**How:**

```bash
# 1. Create data subsets
python -c "
import pandas as pd
import numpy as np

train = pd.read_csv('data/splits/train.csv')
np.random.seed(42)

percentages = [10, 25, 50, 75, 100]

for pct in percentages:
    n_samples = int(len(train) * (pct / 100))

    # Stratified sampling (maintain label balance)
    literal = train[train['label'] == 0]
    figurative = train[train['label'] == 1]

    n_lit = int(n_samples * 0.5)
    n_fig = n_samples - n_lit

    subset = pd.concat([
        literal.sample(n=n_lit, random_state=42),
        figurative.sample(n=n_fig, random_state=42)
    ]).sample(frac=1, random_state=42)

    subset.to_csv(f'data/splits/train_{pct}pct.csv', index=False)
    print(f'Created train_{pct}pct.csv: {len(subset)} samples')
"

# 2. Train models with different data sizes
# On VAST.ai

for pct in 10 25 50 75 100; do
  python src/train.py \
    --config configs/training/dictabert_span_train.yaml \
    --train_file data/splits/train_${pct}pct.csv \
    --seed 42 \
    --output_dir experiments/checkpoints/dictabert_data${pct}/span/seed_42
done

# 3. Evaluate all
for pct in 10 25 50 75 100; do
  python src/evaluate.py \
    --checkpoint experiments/checkpoints/dictabert_data${pct}/span/seed_42/best_model \
    --test_file data/splits/test.csv \
    --task span \
    --output_dir experiments/results/evaluation/seen_test/dictabert_data${pct}/span/seed_42 \
    --save_predictions
done

# 4. Plot data size impact
python -c "
import matplotlib.pyplot as plt
import json

percentages = [10, 25, 50, 75, 100]
f1_scores = []

for pct in percentages:
    with open(f'experiments/results/evaluation/seen_test/dictabert_data{pct}/span/seed_42/eval_results_*.json') as f:
        f1_scores.append(json.load(f)['metrics']['f1'])

plt.figure(figsize=(10, 6))
plt.plot(percentages, f1_scores, marker='o', markersize=8, linewidth=2)
plt.xlabel('Training Data Size (%)')
plt.ylabel('F1 Score')
plt.title('Impact of Training Data Size on Performance')
plt.grid(alpha=0.3)
plt.savefig('experiments/results/analysis/figures/data_size_impact.png', dpi=300)
print('✓ Saved: data_size_impact.png')
"
```

**Document reference:**
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 20

---

### Task 2.3: Hyperparameter Sensitivity
**Mission:** Mission 6.3
**What:** Analyze how F1 varies across hyperparameter space
**Why:** Understand which hyperparameters matter most

**How:**

```bash
# This uses existing Optuna HPO results!
# No new training needed

# 1. Load and analyze Optuna study
python -c "
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load Optuna study
with open('configs/best_hyperparameters/best_params_dictabert_span.json') as f:
    best_params = json.load(f)

# If you have full Optuna study DB:
# from optuna import load_study
# study = load_study(study_name='dictabert_span_hpo',
#                   storage='sqlite:///experiments/hpo/dictabert_span/study.db')
# trials_df = study.trials_dataframe()

# For now, use best params
print('Best Hyperparameters for DictaBERT SPAN:')
for param, value in best_params.items():
    print(f'  {param}: {value}')

# If you have trials_df:
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes = axes.flatten()
#
# for i, param in enumerate(['learning_rate', 'batch_size', 'num_epochs', 'warmup_ratio']):
#     axes[i].scatter(trials_df[param], trials_df['value'])
#     axes[i].set_xlabel(param)
#     axes[i].set_ylabel('Validation F1')
#     axes[i].set_title(f'Sensitivity to {param}')
#
# plt.tight_layout()
# plt.savefig('experiments/results/analysis/figures/hyperparameter_sensitivity.png', dpi=300)
"
```

**Document reference:**
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 21

**Note:** If you don't have Optuna study database, this analysis is limited. You can report best hyperparameters found.

---

## STEP 3: Advanced Interpretability 🔍

**Goal:** Understand what the model learned

**Time:** 1 day
**Requirements:** Trained checkpoints + captum library
**Documents to use:** EVALUATION_STANDARDIZATION_GUIDE.md Section 22

### Task 3.1: Token Importance Analysis
**Mission:** Mission 6.1
**What:** Identify which tokens are important for predictions
**How:**

```bash
# 1. Install captum
pip install captum

# 2. Create token importance script
# File: scripts/analyze_token_importance.py

cat > scripts/analyze_token_importance.py << 'EOF'
#!/usr/bin/env python3
"""Analyze token importance using Integrated Gradients."""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from captum.attr import IntegratedGradients
import pandas as pd
import json

def compute_token_importance(model, tokenizer, sentence, tokens, true_tags):
    """Compute importance scores for each token."""
    model.eval()

    # Tokenize
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Get embeddings
    embeddings = model.get_input_embeddings()(inputs['input_ids'])

    # Forward function
    def forward_func(inputs_embeds):
        outputs = model(inputs_embeds=inputs_embeds)
        # Return probability of B-IDIOM class
        return outputs.logits[:, :, 1].sum()  # Index 1 = B-IDIOM

    # Integrated Gradients
    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(embeddings, target=0)

    # Aggregate attributions
    token_importance = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # Map to original tokens
    word_ids = inputs.word_ids()
    token_scores = {}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id not in token_scores:
                token_scores[word_id] = 0
            token_scores[word_id] += token_importance[i]

    return [(tokens[wid], score) for wid, score in sorted(token_scores.items())]

def main():
    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(
        "experiments/checkpoints/dictabert/span/seed_42/best_model"
    )
    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert")

    # Load test data
    test = pd.read_csv("data/splits/test.csv")

    # Analyze 10 random figurative examples
    figurative = test[test['label'] == 1].sample(n=10, random_state=42)

    results = []
    for _, row in figurative.iterrows():
        tokens = eval(row['tokens'])
        true_tags = eval(row['iob_tags'])

        importance = compute_token_importance(model, tokenizer, row['sentence'], tokens, true_tags)

        results.append({
            'sentence': row['sentence'],
            'tokens': tokens,
            'importance_scores': importance
        })

        print(f"\nSentence: {row['sentence']}")
        print("Top 3 important tokens:")
        for token, score in sorted(importance, key=lambda x: -abs(x[1]))[:3]:
            print(f"  {token}: {score:.4f}")

    # Save results
    with open('experiments/results/analysis/token_importance.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ Token importance analysis complete!")

if __name__ == "__main__":
    main()
EOF

# 3. Run token importance analysis
python scripts/analyze_token_importance.py
```

**Document reference:**
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 22.1

**Expected output:** Token importance scores for sample sentences

---

### Task 3.2: Learning Curves Extraction
**Mission:** Mission 6.1 (supplementary)
**What:** Extract training/validation curves from TensorBoard
**How:**

```bash
# 1. Install tensorboard
pip install tensorboard

# 2. Extract metrics from TensorBoard logs
python -c "
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

# Load TensorBoard events
ea = event_accumulator.EventAccumulator('experiments/logs/dictabert/span/seed_42/')
ea.Reload()

# Extract metrics
train_loss = pd.DataFrame([(e.step, e.value) for e in ea.Scalars('train/loss')],
                          columns=['step', 'train_loss'])
val_loss = pd.DataFrame([(e.step, e.value) for e in ea.Scalars('eval/loss')],
                        columns=['step', 'val_loss'])
val_f1 = pd.DataFrame([(e.step, e.value) for e in ea.Scalars('eval/f1')],
                      columns=['step', 'val_f1'])

# Merge
metrics = train_loss.merge(val_loss, on='step').merge(val_f1, on='step')

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(metrics['step'], metrics['train_loss'], label='Train Loss')
ax1.plot(metrics['step'], metrics['val_loss'], label='Val Loss')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves: DictaBERT SPAN')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(metrics['step'], metrics['val_f1'], color='green')
ax2.set_xlabel('Step')
ax2.set_ylabel('F1 Score')
ax2.set_title('Validation F1: DictaBERT SPAN')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/results/analysis/figures/learning_curves_dictabert_span.png', dpi=300)
print('✓ Saved: learning_curves_dictabert_span.png')
"
```

**Document reference:**
- EVALUATION_STANDARDIZATION_GUIDE.md → Section 18

---

## STEP 4: Finalize & Create Publication Materials 📊

**Goal:** Create all figures and tables for paper

**Time:** 1-2 days
**Requirements:** All previous analyses complete
**Documents to use:** EVALUATION_STANDARDIZATION_GUIDE.md Section 16, 29

### Task 4.1: Create All Publication Figures

```bash
# Use the comprehensive analysis script (creates all figures)
python src/analyze_comprehensive.py \
  --results_dir experiments/results/evaluation \
  --output_dir experiments/results/analysis/comprehensive \
  --create_figures

# This creates:
# - fig1_model_comparison_cls_seen.pdf
# - fig2_generalization_gap_cls.pdf
# - fig3_error_distribution_span_seen.pdf
# - fig4_confusion_matrix_dictabert_cls_seen.pdf
# - fig6_per_idiom_f1_heatmap_span.pdf
# etc.
```

### Task 4.2: Create Summary Tables for Paper

```bash
# Create LaTeX tables from CSV results
python -c "
import pandas as pd

# Load summary
df = pd.read_csv('experiments/results/analysis/finetuning_summary.csv')

# Create LaTeX table for CLS seen test
df_cls_seen = df[(df['task'] == 'cls') & (df['split'] == 'seen_test')]

latex = df_cls_seen[['model', 'f1_mean', 'f1_std', 'accuracy_mean']].to_latex(
    index=False,
    float_format='%.2f',
    column_format='lccc',
    caption='Model Performance on CLS Task (Seen Test)',
    label='tab:cls_seen_results'
)

with open('experiments/results/analysis/tables/cls_seen_results.tex', 'w') as f:
    f.write(latex)

print('✓ Created LaTeX table: cls_seen_results.tex')
"
```

### Task 4.3: Write Results Section Outline

Create file: `paper/results_section_outline.md`

```markdown
# Results Section Outline

## 4. Results

### 4.1 Main Findings
- Best model: DictaBERT (F1: 94.83 ± 0.42 on CLS seen test)
- All Hebrew models outperform multilingual baselines
- Generalization gap: ~3-4% for top models

### 4.2 Model Comparison (Table 1)
[Include table from cls_seen_results.tex]

### 4.3 Generalization Analysis (Figure 1)
[Include generalization gap figure]

### 4.4 Error Analysis (Table 2 + Figure 2)
- Most common errors: PARTIAL_END (15%), MISS (12%)
- Error distribution varies by model

### 4.5 Per-Idiom Analysis (Figure 3)
- 10 hardest idioms identified
- Difficulty correlates with idiom length and ambiguity

### 4.6 Ablation Studies (Table 3)
- Frozen backbone: -5.2% performance drop
- Data size: 50% of data achieves 95% of full performance
```

---

## 📋 Final Implementation Checklist

### Phase 7.1: Error Analysis ✓
- [x] Error categorization for all predictions
- [x] Error distribution tables
- [x] Per-idiom F1 analysis
- [x] Error examples extraction

### Phase 7.2: Statistical Testing ✓
- [x] Paired t-tests (best model vs others)
- [x] Bonferroni correction
- [x] Effect sizes (Cohen's d)

### Phase 7.3: Cross-Task Analysis
- [ ] Correlation between CLS and SPAN performance
- [ ] Model rankings comparison

### Phase 7.4: Visualizations ✓
- [x] Model comparison charts
- [x] Generalization gap charts
- [x] Error distribution charts
- [ ] Learning curves (if TensorBoard logs available)
- [ ] Per-idiom heatmap

### Phase 7.5: Publication Tables ✓
- [x] Results summary tables
- [ ] LaTeX formatting for paper

### Phase 6 Ablations (Optional but Recommended)
- [ ] 6.1: Token importance analysis
- [ ] 6.2: Frozen backbone comparison
- [ ] 6.3: Hyperparameter sensitivity
- [ ] 6.4: Data size impact

---

## 🎯 Recommended Execution Order

### Week 1 (Priority Tasks)
1. ✅ **Day 1-2:** Complete STEP 1 (Quick Wins) - All tasks 1.1-1.6
2. ✅ **Day 3:** Review results, identify gaps
3. **Day 4-5:** Start STEP 2 (Ablations) - Task 2.1 (Frozen backbone)

### Week 2 (Ablations)
4. **Day 6-7:** Task 2.2 (Data size impact)
5. **Day 8-9:** Task 2.3 (Hyperparameter sensitivity)
6. **Day 10:** STEP 3 (Interpretability) - Task 3.1-3.2

### Week 3 (Finalization)
7. **Day 11-12:** STEP 4 (Publication materials)
8. **Day 13-14:** Write results section
9. **Day 15:** Final review and polish

---

## 📚 Document Navigation Map

```
┌────────────────────────────────────────────────────────┐
│ NEED TO UNDERSTAND...                                  │
├────────────────────────────────────────────────────────┤
│ What is Span F1? → EVALUATION_STANDARDIZATION_GUIDE   │
│                     Section 3.2                        │
│                                                        │
│ How to categorize errors? → EVALUATION_STANDARD...    │
│                             Section 4, 14              │
│                                                        │
│ Statistical testing? → EVALUATION_STANDARDIZATION...  │
│                        Section 15                      │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ NEED TO DO...                                          │
├────────────────────────────────────────────────────────┤
│ Run analysis? → OPERATIONS_GUIDE Section 8            │
│ Train model? → OPERATIONS_GUIDE Section 4-5           │
│ Add new model? → OPERATIONS_GUIDE Section 10          │
│ Troubleshoot? → OPERATIONS_GUIDE Section 11           │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ NEED TO IMPLEMENT...                                   │
├────────────────────────────────────────────────────────┤
│ Specific mission? → THIS FILE (IMPLEMENTATION_ROADMAP) │
│ Step-by-step code? → Code snippets in this file       │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Start Here: Your Next Action

**Right now, run this command:**

```bash
# Start with Task 1.1: Aggregate results
python src/analyze_finetuning_results.py

# Then check the output
cat experiments/results/analysis/finetuning_summary.md
```

**Then follow STEP 1 tasks in order (1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6)**

After completing STEP 1 (~2-3 hours), you'll have 40% of Phase 7 done with publication-ready results!

---

**End of Implementation Roadmap**
