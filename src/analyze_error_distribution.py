"""
Comprehensive Error Analysis Dashboard
Task 1.3 Enhancement: Visualize error distributions across models and tasks

Generates:
1. Error distribution stacked bar chart (CLS)
2. Error distribution grouped bar chart (SPAN - aggregated categories)
3. Error distribution heatmap (SPAN - all 12 categories)
4. Seen vs Unseen error shift comparison
5. Per-model error profiles (radar chart)
6. Error summary report
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter

# Configuration
EVAL_ROOT = Path("experiments/results/evaluation")
OUTPUT_DIR = Path("experiments/results/analysis/error_analysis")
FIGURES_DIR = Path("paper/figures/error_analysis")

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plotting configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9
})

# Color schemes
CLS_COLORS = {
    'CORRECT': '#2ecc71',      # Green
    'FALSE_POSITIVE': '#e74c3c',  # Red
    'FALSE_NEGATIVE': '#f39c12'   # Orange
}

SPAN_CATEGORY_GROUPS = {
    'PERFECT': ['PERFECT'],
    'BOUNDARY_ERRORS': ['PARTIAL_START', 'PARTIAL_END', 'PARTIAL_BOTH',
                        'EXTEND_START', 'EXTEND_END', 'EXTEND_BOTH'],
    'DETECTION_ERRORS': ['MISS', 'FALSE_POSITIVE'],
    'POSITION_ERRORS': ['SHIFT', 'WRONG_SPAN', 'MULTI_SPAN']
}

SPAN_GROUP_COLORS = {
    'PERFECT': '#2ecc71',           # Green
    'BOUNDARY_ERRORS': '#f39c12',   # Orange
    'DETECTION_ERRORS': '#e74c3c',  # Red
    'POSITION_ERRORS': '#9b59b6'    # Purple
}


def load_all_categorized_predictions():
    """Load all predictions with error_category field."""
    print("📊 Loading categorized predictions...")

    data = []
    models = ['alephbert-base', 'alephbertgimmel-base', 'dictabert',
              'bert-base-multilingual-cased', 'xlm-roberta-base']
    tasks = ['cls', 'span']
    seeds = [42, 123, 456]
    splits = ['seen_test', 'unseen_test']

    for split in splits:
        for model in models:
            for task in tasks:
                for seed in seeds:
                    pred_file = EVAL_ROOT / split / model / task / f"seed_{seed}" / "eval_predictions.json"

                    if not pred_file.exists():
                        continue

                    try:
                        with open(pred_file, 'r', encoding='utf-8') as f:
                            predictions = json.load(f)

                        for pred in predictions:
                            data.append({
                                'model': model,
                                'task': task,
                                'seed': seed,
                                'split': 'Seen' if split == 'seen_test' else 'Unseen',
                                'error_category': pred.get('error_category', 'UNKNOWN'),
                                'is_correct': pred.get('is_correct', None),
                                'id': pred.get('id', '')
                            })
                    except Exception as e:
                        print(f"⚠️  Error loading {pred_file}: {e}")

    df = pd.DataFrame(data)
    print(f"   Loaded {len(df):,} categorized predictions")
    print(f"   Models: {df['model'].nunique()}")
    print(f"   Tasks: {df['task'].unique().tolist()}")
    print(f"   Splits: {df['split'].unique().tolist()}")

    return df


def aggregate_error_stats(df):
    """Compute error percentages per model/task/split."""
    print("\n📈 Aggregating error statistics...")

    # Group by model, task, split, error_category
    stats = df.groupby(['model', 'task', 'split', 'error_category']).size().reset_index(name='count')

    # Calculate totals per model/task/split
    totals = df.groupby(['model', 'task', 'split']).size().reset_index(name='total')

    # Merge and calculate percentage
    stats = stats.merge(totals, on=['model', 'task', 'split'])
    stats['percentage'] = (stats['count'] / stats['total']) * 100

    return stats


def group_span_categories(df):
    """Group SPAN error categories into meaningful clusters."""
    def get_category_group(cat):
        for group, categories in SPAN_CATEGORY_GROUPS.items():
            if cat in categories:
                return group
        return 'OTHER'

    df['category_group'] = df['error_category'].apply(get_category_group)
    return df


def plot_error_distribution_cls(stats_df):
    """Generate stacked bar chart for CLS task error distribution."""
    print("\n📊 Plotting CLS error distribution...")

    # Filter CLS task
    cls_stats = stats_df[stats_df['task'] == 'cls'].copy()

    # Pivot for plotting
    pivot = cls_stats.pivot_table(
        index=['model', 'split'],
        columns='error_category',
        values='percentage',
        fill_value=0
    ).reset_index()

    # Ensure all categories exist
    for cat in CLS_COLORS.keys():
        if cat not in pivot.columns:
            pivot[cat] = 0

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, split in enumerate(['Seen', 'Unseen']):
        ax = axes[idx]
        data = pivot[pivot['split'] == split].sort_values('CORRECT', ascending=False)

        models = data['model'].values
        x = np.arange(len(models))
        width = 0.6

        # Plot stacked bars
        bottom = np.zeros(len(models))
        for category in ['CORRECT', 'FALSE_POSITIVE', 'FALSE_NEGATIVE']:
            values = data[category].values
            ax.bar(x, values, width, bottom=bottom, label=category.replace('_', ' ').title(),
                  color=CLS_COLORS[category], edgecolor='white', linewidth=0.5)

            # Add percentage labels for non-zero values
            for i, val in enumerate(values):
                if val > 2:  # Only label if > 2%
                    ax.text(i, bottom[i] + val/2, f'{val:.1f}%',
                           ha='center', va='center', fontsize=8, fontweight='bold', color='white')

            bottom += values

        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title(f'CLS Task - {split} Test', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('-base', '').replace('bert-', '') for m in models],
                          rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = FIGURES_DIR / 'error_distribution_cls.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_error_distribution_span_aggregated(stats_df):
    """Generate grouped bar chart for SPAN task with aggregated categories."""
    print("\n📊 Plotting SPAN error distribution (aggregated)...")

    # Filter SPAN task
    span_stats = stats_df[stats_df['task'] == 'span'].copy()

    # Group categories
    span_stats = group_span_categories(span_stats)

    # Aggregate by category group
    grouped = span_stats.groupby(['model', 'split', 'category_group'])['percentage'].sum().reset_index()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, split in enumerate(['Seen', 'Unseen']):
        ax = axes[idx]
        data = grouped[grouped['split'] == split]

        # Pivot for grouped bar chart
        pivot = data.pivot(index='model', columns='category_group', values='percentage').fillna(0)

        # Ensure all groups exist
        for group in SPAN_CATEGORY_GROUPS.keys():
            if group not in pivot.columns:
                pivot[group] = 0

        # Sort by PERFECT (descending)
        pivot = pivot.sort_values('PERFECT', ascending=False)

        # Plot grouped bars
        x = np.arange(len(pivot))
        width = 0.18

        for i, group in enumerate(['PERFECT', 'BOUNDARY_ERRORS', 'DETECTION_ERRORS', 'POSITION_ERRORS']):
            offset = (i - 1.5) * width
            values = pivot[group].values
            ax.bar(x + offset, values, width, label=group.replace('_', ' ').title(),
                  color=SPAN_GROUP_COLORS[group], edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title(f'SPAN Task - {split} Test', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('-base', '').replace('bert-', '') for m in pivot.index],
                          rotation=45, ha='right')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)

        # Add 100% reference line
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    save_path = FIGURES_DIR / 'error_distribution_span_aggregated.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_error_heatmap_span(stats_df):
    """Generate detailed heatmap for all 12 SPAN error categories."""
    print("\n🔥 Plotting SPAN error heatmap (all categories)...")

    # Filter SPAN task
    span_stats = stats_df[stats_df['task'] == 'span'].copy()

    # Create combined model+split labels
    span_stats['model_split'] = span_stats['model'] + ' (' + span_stats['split'] + ')'

    # Pivot for heatmap
    pivot = span_stats.pivot_table(
        index='model_split',
        columns='error_category',
        values='percentage',
        fill_value=0
    )

    # Order rows: Hebrew models first, then multilingual, seen before unseen
    model_order = []
    for model in ['dictabert', 'alephbert-base', 'alephbertgimmel-base',
                  'xlm-roberta-base', 'bert-base-multilingual-cased']:
        for split in ['Seen', 'Unseen']:
            label = f'{model} ({split})'
            if label in pivot.index:
                model_order.append(label)

    pivot = pivot.reindex(model_order)

    # Order columns by importance
    col_order = ['PERFECT', 'PARTIAL_END', 'PARTIAL_START', 'MISS',
                 'PARTIAL_BOTH', 'FALSE_POSITIVE', 'EXTEND_END',
                 'EXTEND_START', 'EXTEND_BOTH', 'MULTI_SPAN',
                 'WRONG_SPAN', 'SHIFT']
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use diverging colormap centered on median
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Percentage (%)'},
                linewidths=0.5, linecolor='white',
                vmin=0, vmax=100, ax=ax)

    ax.set_xlabel('Error Category', fontweight='bold', fontsize=11)
    ax.set_ylabel('Model (Test Set)', fontweight='bold', fontsize=11)
    ax.set_title('SPAN Task: Error Distribution Heatmap (All 12 Categories)',
                fontweight='bold', fontsize=13, pad=20)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    save_path = FIGURES_DIR / 'error_heatmap_span.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_seen_unseen_comparison(stats_df):
    """Compare error distributions between seen and unseen test sets."""
    print("\n📊 Plotting seen vs unseen error shift...")

    # Filter SPAN task and group categories
    span_stats = stats_df[stats_df['task'] == 'span'].copy()
    span_stats = group_span_categories(span_stats)

    # Aggregate across models and seeds
    comparison = span_stats.groupby(['split', 'category_group'])['percentage'].mean().reset_index()

    # Pivot for plotting
    pivot = comparison.pivot(index='category_group', columns='split', values='percentage').fillna(0)

    # Ensure all groups exist
    for group in SPAN_CATEGORY_GROUPS.keys():
        if group not in pivot.index:
            pivot.loc[group] = [0, 0]

    # Sort by unseen percentage (descending)
    pivot = pivot.sort_values('Unseen', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(pivot))
    width = 0.35

    seen_bars = ax.bar(x - width/2, pivot['Seen'], width, label='Seen Test',
                      color='#3498db', edgecolor='white', linewidth=1)
    unseen_bars = ax.bar(x + width/2, pivot['Unseen'], width, label='Unseen Test',
                        color='#e74c3c', edgecolor='white', linewidth=1)

    # Add value labels
    for bars in [seen_bars, unseen_bars]:
        for bar in bars:
            height = bar.get_height()
            if height > 1:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Error Category Group', fontweight='bold')
    ax.set_ylabel('Average Percentage (%)', fontweight='bold')
    ax.set_title('SPAN Task: Error Distribution Shift from Seen to Unseen Idioms',
                fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace('_', '\n') for g in pivot.index], fontsize=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = FIGURES_DIR / 'seen_unseen_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_model_error_profiles(stats_df):
    """Generate radar chart showing distinctive error signatures per model."""
    print("\n🕸️  Plotting model error profiles (radar chart)...")

    # Filter SPAN task, Unseen only (where differences are most visible)
    span_unseen = stats_df[(stats_df['task'] == 'span') & (stats_df['split'] == 'Unseen')].copy()

    # Get top error categories (excluding PERFECT)
    top_categories = (span_unseen[span_unseen['error_category'] != 'PERFECT']
                     .groupby('error_category')['percentage']
                     .mean()
                     .nlargest(6)
                     .index.tolist())

    # Filter to top categories + PERFECT
    categories = ['PERFECT'] + top_categories
    data = span_unseen[span_unseen['error_category'].isin(categories)]

    # Aggregate by model
    pivot = data.pivot_table(
        index='model',
        columns='error_category',
        values='percentage',
        aggfunc='mean',
        fill_value=0
    )[categories]  # Reorder columns

    # Radar chart setup
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Define colors for models
    model_colors = {
        'dictabert': '#e74c3c',
        'alephbert-base': '#3498db',
        'alephbertgimmel-base': '#2ecc71',
        'xlm-roberta-base': '#f39c12',
        'bert-base-multilingual-cased': '#9b59b6'
    }

    # Plot each model
    for model in pivot.index:
        values = pivot.loc[model].tolist()
        values += values[:1]  # Complete the circle

        label = model.replace('-base', '').replace('bert-', '')
        ax.plot(angles, values, 'o-', linewidth=2, label=label,
               color=model_colors.get(model, '#34495e'))
        ax.fill(angles, values, alpha=0.15, color=model_colors.get(model, '#34495e'))

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([cat.replace('_', '\n') for cat in categories], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.set_title('SPAN Task: Model Error Profiles on Unseen Idioms\n(Radar Chart)',
                fontweight='bold', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    plt.tight_layout()
    save_path = FIGURES_DIR / 'model_error_profiles.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def generate_error_summary_report(stats_df, df_raw):
    """Generate comprehensive error analysis report with full methodology."""
    print("\n📝 Generating error summary report...")

    # Save detailed statistics
    stats_csv = OUTPUT_DIR / 'error_distribution_detailed.csv'
    stats_df.to_csv(stats_csv, index=False)
    print(f"   ✅ Saved detailed stats: {stats_csv}")

    # Calculate summary statistics
    cls_stats = stats_df[stats_df['task'] == 'cls']
    span_stats = stats_df[stats_df['task'] == 'span'].copy()
    span_stats = group_span_categories(span_stats)

    cls_summary = cls_stats.pivot_table(
        index='error_category',
        columns='split',
        values='percentage',
        aggfunc='mean'
    ).round(2)

    span_summary = span_stats.groupby(['split', 'category_group'])['percentage'].mean().reset_index()
    span_pivot = span_summary.pivot(index='category_group', columns='split', values='percentage').round(2)

    cls_correct = cls_stats[cls_stats['error_category'] == 'CORRECT'].groupby('split')['percentage'].mean()
    span_perfect = span_stats[span_stats['category_group'] == 'PERFECT'].groupby('split')['percentage'].mean()

    span_unseen = span_stats[span_stats['split'] == 'Unseen']
    top_errors = (span_unseen[span_unseen['category_group'] != 'PERFECT']
                 .groupby('category_group')['percentage']
                 .mean()
                 .sort_values(ascending=False))

    # Generate comprehensive report
    models = sorted(stats_df['model'].unique())
    num_models = len(models)
    num_seeds = len(df_raw['seed'].unique())

    # Precompute counts for report (use raw data because stats_df has no seed column)
    file_count = df_raw.groupby(['model', 'task', 'seed', 'split']).size().shape[0]

    # Ensure expected rows exist to avoid KeyError in report rendering
    cls_summary = cls_summary.reindex(['CORRECT', 'FALSE_POSITIVE', 'FALSE_NEGATIVE'])
    span_pivot = span_pivot.reindex(['PERFECT', 'BOUNDARY_ERRORS', 'DETECTION_ERRORS', 'POSITION_ERRORS'])

    report = f"""# Error Analysis Summary Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Predictions Analyzed:** {len(df_raw):,}
**Scope:** {num_models} models × 2 tasks × {num_seeds} seeds × 2 splits × variable samples

---

## Methodology

### Data Aggregation Process

**1. Error Categorization (Step 1)**
- **Tool:** `scripts/categorize_all_errors.py`
- **Input:** {file_count} evaluation files ({num_models} models × 2 tasks × {num_seeds} seeds × 2 splits)
- **Process:** Applied standardized error taxonomy to all {len(df_raw):,} predictions
- **Output:** Added `error_category` field to all `eval_predictions.json` files
- **Taxonomy Source:** `src/utils/error_analysis.py` (categorize_span_error, categorize_cls_error)

**2. Aggregation Across Seeds (Step 2)**
- **Tool:** `src/analyze_error_distribution.py`
- **Aggregation Method:** Pooled all predictions across {num_seeds} seeds ({', '.join(map(str, sorted(df_raw['seed'].unique())))})
- **Rationale:** Provides robust error statistics by combining all runs
- **Total Samples per Model/Task/Split:**
  - CLS Seen: ~{len(df_raw[(df_raw['task']=='cls') & (df_raw['split']=='Seen')]) // num_models:,} predictions per model (across {num_seeds} seeds)
  - CLS Unseen: ~{len(df_raw[(df_raw['task']=='cls') & (df_raw['split']=='Unseen')]) // num_models:,} predictions per model
  - SPAN Seen: ~{len(df_raw[(df_raw['task']=='span') & (df_raw['split']=='Seen')]) // num_models:,} predictions per model
  - SPAN Unseen: ~{len(df_raw[(df_raw['task']=='span') & (df_raw['split']=='Unseen')]) // num_models:,} predictions per model

**3. Cross-Model Aggregation (Step 3)**
- **Method:** Averaged percentages across all {num_models} models
- **Purpose:** Report overall error distribution patterns
- **Models Included:**
{chr(10).join(f'  - {m}' for m in models)}

---

## Error Taxonomy

### CLS Task Categories (3 categories)
| Category | Description |
|----------|-------------|
| **CORRECT** | Predicted label matches ground truth (TP + TN) |
| **FALSE_POSITIVE** | Predicted Figurative, actually Literal |
| **FALSE_NEGATIVE** | Predicted Literal, actually Figurative |

### SPAN Task Categories (12 categories → 4 groups)

#### Group 1: PERFECT
**Exact Match**
- **PERFECT**: Predicted span boundaries match ground truth exactly

#### Group 2: BOUNDARY_ERRORS (6 categories)
**Span detected but boundaries incorrect**
- **PARTIAL_START**: Missing beginning token(s) of idiom
- **PARTIAL_END**: Missing ending token(s) of idiom
- **PARTIAL_BOTH**: Truncated on both start and end
- **EXTEND_START**: Extra token(s) at start of span
- **EXTEND_END**: Extra token(s) at end of span
- **EXTEND_BOTH**: Extended on both start and end

**Grouping Logic:**
```python
BOUNDARY_ERRORS = ['PARTIAL_START', 'PARTIAL_END', 'PARTIAL_BOTH',
                   'EXTEND_START', 'EXTEND_END', 'EXTEND_BOTH']
```

#### Group 3: DETECTION_ERRORS (2 categories)
**Failed to detect idiom or hallucinated non-existent idiom**
- **MISS**: No span predicted when ground truth has idiom
- **FALSE_POSITIVE**: Span predicted when no idiom exists

**Grouping Logic:**
```python
DETECTION_ERRORS = ['MISS', 'FALSE_POSITIVE']
```

#### Group 4: POSITION_ERRORS (3 categories)
**Span at wrong location or fragmented**
- **SHIFT**: Span overlaps but boundaries misaligned
- **WRONG_SPAN**: Completely different phrase tagged as idiom
- **MULTI_SPAN**: Multiple spans predicted (hallucination)

**Grouping Logic:**
```python
POSITION_ERRORS = ['SHIFT', 'WRONG_SPAN', 'MULTI_SPAN']
```

---

## CLS Task Error Distribution

{cls_summary.to_markdown()}

**Interpretation:**
- Models maintain high accuracy (~{cls_correct.get('Seen', 0):.0f}-{cls_correct.get('Unseen', 0):.0f}%) on both seen and unseen idioms
- False Positives {('increase' if cls_summary.loc['FALSE_POSITIVE', 'Unseen'] > cls_summary.loc['FALSE_POSITIVE', 'Seen'] else 'decrease')} on unseen idioms ({cls_summary.loc['FALSE_POSITIVE', 'Seen']:.2f}% → {cls_summary.loc['FALSE_POSITIVE', 'Unseen']:.2f}%), suggesting models {'over-predict' if cls_summary.loc['FALSE_POSITIVE', 'Unseen'] > cls_summary.loc['FALSE_POSITIVE', 'Seen'] else 'under-predict'} figurative meaning for novel idioms
- False Negatives {('increase' if cls_summary.loc['FALSE_NEGATIVE', 'Unseen'] > cls_summary.loc['FALSE_NEGATIVE', 'Seen'] else 'decrease')} on unseen idioms ({cls_summary.loc['FALSE_NEGATIVE', 'Seen']:.2f}% → {cls_summary.loc['FALSE_NEGATIVE', 'Unseen']:.2f}%)

---

## SPAN Task Error Distribution (Grouped)

{span_pivot.to_markdown()}

**Category Grouping Breakdown:**
- **PERFECT**: {len(SPAN_CATEGORY_GROUPS['PERFECT'])} category (exact matches)
- **BOUNDARY_ERRORS**: {len(SPAN_CATEGORY_GROUPS['BOUNDARY_ERRORS'])} categories (partial/extended spans)
- **DETECTION_ERRORS**: {len(SPAN_CATEGORY_GROUPS['DETECTION_ERRORS'])} categories (missed or hallucinated)
- **POSITION_ERRORS**: {len(SPAN_CATEGORY_GROUPS['POSITION_ERRORS'])} categories (wrong location)

**Interpretation:**
- **Dramatic Generalization Gap:** Perfect matches drop from {span_pivot.loc['PERFECT', 'Seen']:.1f}% (seen) to {span_pivot.loc['PERFECT', 'Unseen']:.1f}% (unseen)
- **Boundary Errors Dominate Unseen:** {span_pivot.loc['BOUNDARY_ERRORS', 'Unseen']:.2f}% boundary errors on unseen idioms vs {span_pivot.loc['BOUNDARY_ERRORS', 'Seen']:.2f}% on seen
  - Models can detect idioms but struggle with exact boundaries for novel expressions
- **Detection Failures:** {span_pivot.loc['DETECTION_ERRORS', 'Unseen']:.2f}% detection errors on unseen idioms
  - Models miss some unseen idioms entirely or hallucinate non-existent ones
- **Position Errors Rare:** Only {span_pivot.loc['POSITION_ERRORS', 'Unseen']:.2f}% on unseen idioms
  - When models detect idioms, they usually find the correct region

---

## Key Findings

### CLS Task Performance
- **Seen Test Accuracy:** {cls_correct.get('Seen', 0):.1f}% (averaged across {num_models} models, {num_seeds} seeds each)
- **Unseen Test Accuracy:** {cls_correct.get('Unseen', 0):.1f}%
- **Generalization Gap:** {(cls_correct.get('Seen', 0) - cls_correct.get('Unseen', 0)):.1f} percentage points
- **Dominant Error (Unseen):** {cls_summary.loc[cls_summary['Unseen'].idxmax(), 'Unseen']:.1f}% ({cls_summary['Unseen'].idxmax()})

### SPAN Task Performance
- **Seen Test Perfect Matches:** {span_perfect.get('Seen', 0):.1f}%
- **Unseen Test Perfect Matches:** {span_perfect.get('Unseen', 0):.1f}%
- **Generalization Gap:** {(span_perfect.get('Seen', 0) - span_perfect.get('Unseen', 0)):.1f} percentage points
- **Dominant Errors (Unseen):**
{chr(10).join(f'  {i}. **{error.replace("_", " ")}:** {pct:.1f}% ({SPAN_CATEGORY_GROUPS[error] if error in SPAN_CATEGORY_GROUPS else "various"})' for i, (error, pct) in enumerate(top_errors.head(3).items(), 1))}

### Critical Insights
1. **CLS generalizes well:** Only {(cls_correct.get('Seen', 0) - cls_correct.get('Unseen', 0)):.1f}% performance drop on unseen idioms
2. **SPAN struggles with generalization:** {(span_perfect.get('Seen', 0) - span_perfect.get('Unseen', 0)):.1f}% drop indicates exact boundary detection is harder for novel idioms
3. **Boundary detection is the bottleneck:** Models can often detect idioms but fail on precise token boundaries
4. **Seen idioms nearly perfect:** {span_perfect.get('Seen', 0):.1f}% perfect matches shows models learn seen idiom boundaries very well

---

## Visualizations Generated

1. **error_distribution_cls.png** - Stacked bar chart of CLS errors (CORRECT/FALSE_POSITIVE/FALSE_NEGATIVE)
2. **error_distribution_span_aggregated.png** - Grouped bar chart of 4 SPAN error groups
3. **error_heatmap_span.png** - Heatmap showing all 12 SPAN categories across models
4. **seen_unseen_comparison.png** - Comparison showing error shift from seen to unseen
5. **model_error_profiles.png** - Radar chart showing distinctive error patterns per model

All figures saved to: `paper/figures/error_analysis/` (300 DPI, publication-ready)

## Figures (Embedded)

![CLS Error Distribution](paper/figures/error_analysis/error_distribution_cls.png)
![SPAN Error Distribution (Grouped)](paper/figures/error_analysis/error_distribution_span_aggregated.png)
![SPAN Error Heatmap (All Categories)](paper/figures/error_analysis/error_heatmap_span.png)
![Seen vs Unseen Shift](paper/figures/error_analysis/seen_unseen_comparison.png)
![Model Error Profiles](paper/figures/error_analysis/model_error_profiles.png)
"""

    # Save report
    report_path = OUTPUT_DIR / 'error_analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"   ✅ Saved summary report: {report_path}")


def main():
    """Main analysis pipeline."""
    print("="*70)
    print("COMPREHENSIVE ERROR ANALYSIS DASHBOARD")
    print("="*70)

    # Load data
    df_raw = load_all_categorized_predictions()

    # Aggregate statistics
    stats_df = aggregate_error_stats(df_raw)

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_error_distribution_cls(stats_df)
    plot_error_distribution_span_aggregated(stats_df)
    plot_error_heatmap_span(stats_df)
    plot_seen_unseen_comparison(stats_df)
    plot_model_error_profiles(stats_df)

    # Generate report
    print("\n" + "="*70)
    print("GENERATING REPORTS")
    print("="*70)
    generate_error_summary_report(stats_df, df_raw)

    print("\n" + "="*70)
    print("✅ ERROR ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  Figures: {FIGURES_DIR}/")
    print(f"    - error_distribution_cls.png")
    print(f"    - error_distribution_span_aggregated.png")
    print(f"    - error_heatmap_span.png")
    print(f"    - seen_unseen_comparison.png")
    print(f"    - model_error_profiles.png")
    print(f"\n  Analysis: {OUTPUT_DIR}/")
    print(f"    - error_distribution_detailed.csv")
    print(f"    - error_analysis_report.md")


if __name__ == "__main__":
    main()
