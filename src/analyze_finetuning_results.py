"""
Enhanced Fine-Tuning Results Analysis
With Bonferroni correction, Cohen's d effect size, and LaTeX output
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
EVAL_ROOT = Path("experiments/results/evaluation")
OUTPUT_DIR = Path("experiments/results/analysis")
PAPER_TABLES_DIR = Path("paper/tables")
PAPER_FIGURES_DIR = Path("paper/figures/finetuning")

OUTPUT_MD = OUTPUT_DIR / "finetuning_summary.md"
OUTPUT_CSV = OUTPUT_DIR / "finetuning_summary.csv"
SIG_FILE = OUTPUT_DIR / "statistical_significance.txt"
LATEX_TABLE = PAPER_TABLES_DIR / "finetuning_results.tex"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plotting configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
})


def load_all_results():
    """Load all evaluation results from seen_test and unseen_test directories."""
    data = []
    test_types = ["seen_test", "unseen_test"]

    for test_type in test_types:
        search_path = EVAL_ROOT / test_type
        if not search_path.exists():
            print(f"⚠️  Warning: Path not found: {search_path}")
            continue

        print(f"  Loading {test_type}...")
        for f in search_path.rglob("eval_results*.json"):
            try:
                # Path structure: .../seen_test/MODEL/TASK/SEED/eval_results.json
                parts = f.parts
                seed_part = next((p for p in reversed(parts) if p.startswith("seed_")), None)
                if not seed_part:
                    continue

                seed_idx = parts.index(seed_part)
                task = parts[seed_idx - 1]
                model = parts[seed_idx - 2]
                seed = int(seed_part.replace("seed_", ""))

                with open(f, 'r') as json_file:
                    res = json.load(json_file)

                metrics = res.get("metrics", {})

                # Standardize metric names
                f1 = metrics.get("f1", metrics.get("eval_f1", 0))
                acc = metrics.get("accuracy", metrics.get("eval_accuracy", 0))
                prec = metrics.get("precision", metrics.get("eval_precision", 0))
                rec = metrics.get("recall", metrics.get("eval_recall", 0))

                data.append({
                    "model": model,
                    "task": task,
                    "seed": seed,
                    "test_set": "Seen" if test_type == "seen_test" else "Unseen",
                    "f1": f1,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec
                })
            except Exception as e:
                print(f"Error loading {f}: {e}")

    return pd.DataFrame(data)


def bootstrap_ci(scores, n_bootstrap=10000, confidence_level=0.95, random_state=42):
    """
    Compute bootstrap confidence interval for mean.

    Args:
        scores: Array of F1 scores (typically 3 seeds)
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        (ci_low, ci_high): Bootstrap confidence interval bounds
    """
    from scipy.stats import bootstrap

    if len(scores) < 2:
        return np.nan, np.nan

    # Bootstrap requires a callable that returns the statistic
    def mean_func(sample, axis):
        return np.mean(sample, axis=axis)

    # Perform bootstrap
    rng = np.random.RandomState(random_state)
    result = bootstrap(
        (scores,),
        mean_func,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        random_state=rng,
        method='percentile'
    )

    return result.confidence_interval.low, result.confidence_interval.high


def calculate_statistics(df):
    """
    Calculate Mean ± Std and Bootstrap 95% CI for all metrics.

    Updated for Priority 1 Task: Bootstrap CIs (CONLL_2026_COMPREHENSIVE_REVIEW.md)
    """
    # Group by Test Set, Task, Model
    grouped = df.groupby(["test_set", "task", "model"])

    results = []
    for (test_set, task, model), group in grouped:
        f1_scores = group["f1"].values

        # Compute statistics
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores, ddof=1)
        count = len(f1_scores)

        # Compute bootstrap CI
        ci_low, ci_high = bootstrap_ci(f1_scores)

        results.append({
            "test_set": test_set,
            "task": task,
            "model": model,
            "mean": mean_f1,
            "std": std_f1,
            "count": count,
            "ci_low": ci_low,
            "ci_high": ci_high
        })

    summary = pd.DataFrame(results)

    # Sort by Test Set, Task, then Mean F1 (Desc)
    summary = summary.sort_values(
        ["test_set", "task", "mean"],
        ascending=[True, True, False]
    )

    return summary


def cohens_d(x, y):
    """
    Calculate Cohen's d effect size.

    Args:
        x, y: Two groups of scores (numpy arrays)

    Returns:
        Cohen's d (float)
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
    )


def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def perform_ttests(df, summary):
    """
    Perform paired t-tests with Bonferroni correction and Cohen's d.
    """
    report_lines = []

    for test_set in ["Seen", "Unseen"]:
        report_lines.append(f"\n## Statistical Significance - {test_set} Test Set")

        for task in ["cls", "span"]:
            # Filter data for specific context
            subset_df = df[(df['test_set'] == test_set) & (df['task'] == task)]
            subset_summary = summary[(summary['test_set'] == test_set) & (summary['task'] == task)]

            if subset_summary.empty:
                continue

            # Identify Best Model
            best_model = subset_summary.iloc[0]['model']
            best_mean = subset_summary.iloc[0]['mean']
            best_scores = subset_df[subset_df['model'] == best_model]['f1'].values

            # Number of comparisons for Bonferroni correction
            other_models = subset_summary.iloc[1:]['model'].values
            num_comparisons = len(other_models)
            alpha = 0.05
            alpha_bonferroni = alpha / num_comparisons if num_comparisons > 0 else alpha

            report_lines.append(f"\n### Task: {task.upper()} ({test_set})")
            report_lines.append(f"**Best Model:** {best_model} (Mean F1: {best_mean:.4f})")
            report_lines.append(f"**Bonferroni-corrected α:** {alpha_bonferroni:.4f} ({num_comparisons} comparisons)")
            report_lines.append("\n| Comparison | T-Stat | P-Value | Bonferroni | Cohen's d | Effect Size | Significant? |")
            report_lines.append("|------------|--------|---------|------------|-----------|-------------|--------------|")

            # Compare best against all others
            for other in other_models:
                other_scores = subset_df[subset_df['model'] == other]['f1'].values

                # Valid T-Test requires same number of samples (3 seeds)
                if len(best_scores) == len(other_scores) == 3:
                    t_stat, p_val = stats.ttest_rel(best_scores, other_scores)
                    d = cohens_d(best_scores, other_scores)
                    effect = interpret_effect_size(d)

                    # Check both uncorrected and Bonferroni-corrected significance
                    sig_uncorrected = p_val < alpha
                    sig_bonferroni = p_val < alpha_bonferroni

                    if sig_bonferroni:
                        sig_marker = "✅ YES**"
                    elif sig_uncorrected:
                        sig_marker = "⚠️ YES*"
                    else:
                        sig_marker = "❌ NO"

                    report_lines.append(
                        f"| {best_model} vs {other} | {t_stat:.3f} | "
                        f"{p_val:.4f} | {alpha_bonferroni:.4f} | "
                        f"{d:.3f} | {effect} | {sig_marker} |"
                    )
                else:
                    report_lines.append(
                        f"| {best_model} vs {other} | N/A | N/A | N/A | N/A | N/A | Missing seeds |"
                    )

        # Add legend
        report_lines.append("\n**Legend:**")
        report_lines.append("- ✅ YES**: Significant after Bonferroni correction (conservative)")
        report_lines.append("- ⚠️ YES*: Significant without correction (p < 0.05), but NOT after Bonferroni")
        report_lines.append("- ❌ NO: Not significant")

    return "\n".join(report_lines)


def create_latex_tables(summary):
    """Create LaTeX tables for paper."""
    latex_lines = []

    latex_lines.append("% Fine-Tuning Results Tables")
    latex_lines.append("% Generated automatically by analyze_finetuning_results.py")
    latex_lines.append("")

    # Table 1: Seen Test Results
    latex_lines.append("% Table 1: In-Domain Performance (Seen Test)")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{In-Domain Performance on Seen Idioms}")
    latex_lines.append("\\label{tab:seen_performance}")
    latex_lines.append("\\begin{tabular}{llrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Task & Model & Mean F1 & Std \\\\")
    latex_lines.append("\\midrule")

    seen_summary = summary[summary['test_set'] == 'Seen']
    for task in ['cls', 'span']:
        task_data = seen_summary[seen_summary['task'] == task]
        task_label = "Classification" if task == "cls" else "Span Detection"

        for idx, row in task_data.iterrows():
            model_name = row['model'].replace('_', '\\_').replace('-', '-')
            if idx == task_data.index[0]:  # First row for this task
                latex_lines.append(
                    f"{task_label} & {model_name} & "
                    f"{row['mean']:.4f} & {row['std']:.4f} \\\\"
                )
            else:
                latex_lines.append(
                    f" & {model_name} & {row['mean']:.4f} & {row['std']:.4f} \\\\"
                )
        if task == 'cls':
            latex_lines.append("\\midrule")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    latex_lines.append("")

    # Table 2: Unseen Test Results
    latex_lines.append("% Table 2: Generalization Performance (Unseen Test)")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Zero-Shot Generalization on Unseen Idioms}")
    latex_lines.append("\\label{tab:unseen_performance}")
    latex_lines.append("\\begin{tabular}{llrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Task & Model & Mean F1 & Std \\\\")
    latex_lines.append("\\midrule")

    unseen_summary = summary[summary['test_set'] == 'Unseen']
    for task in ['cls', 'span']:
        task_data = unseen_summary[unseen_summary['task'] == task]
        task_label = "Classification" if task == "cls" else "Span Detection"

        for idx, row in task_data.iterrows():
            model_name = row['model'].replace('_', '\\_').replace('-', '-')
            if idx == task_data.index[0]:
                latex_lines.append(
                    f"{task_label} & {model_name} & "
                    f"{row['mean']:.4f} & {row['std']:.4f} \\\\"
                )
            else:
                latex_lines.append(
                    f" & {model_name} & {row['mean']:.4f} & {row['std']:.4f} \\\\"
                )
        if task == 'cls':
            latex_lines.append("\\midrule")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)

def plot_model_comparison(summary: pd.DataFrame, task: str, test_set: str, out_path: Path) -> None:
    """Create model comparison bar chart with mean ± std."""
    data = summary[(summary["task"] == task) & (summary["test_set"] == test_set)].copy()
    if data.empty:
        return

    data = data.sort_values("mean", ascending=False)
    labels = data["model"].tolist()
    means = data["mean"].tolist()
    stds = data["std"].fillna(0.0).tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.85, color=sns.color_palette("colorblind")[0])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Model Comparison: {task.upper()} ({test_set} Test)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def create_comparison_figures(summary: pd.DataFrame) -> None:
    """Generate comparison figures for CLS/SPAN across Seen/Unseen."""
    for task in ["cls", "span"]:
        for test_set in ["Seen", "Unseen"]:
            filename = f"model_comparison_{task}_{test_set.lower()}.png"
            out_path = PAPER_FIGURES_DIR / filename
            plot_model_comparison(summary, task, test_set, out_path)
    print(f"   ✅ Generated comparison figures in {PAPER_FIGURES_DIR}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="Fine-tuning results analysis")
    parser.add_argument(
        "--create_figures",
        action="store_true",
        help="Generate model comparison figures in paper/figures/finetuning"
    )
    args = parser.parse_args()

    print("📊 Loading ALL results (Seen + Unseen)...")
    df = load_all_results()

    if df.empty:
        print("❌ No data found.")
        return

    print(f"   Loaded {len(df)} total evaluation records.")

    print("📈 Calculating comprehensive statistics...")
    summary = calculate_statistics(df)

    # Save CSV
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"   ✅ Saved summary CSV: {OUTPUT_CSV}")

    print("🧪 Performing statistical tests (with Bonferroni + Cohen's d)...")
    ttest_report = perform_ttests(df, summary)

    # Write Markdown Report
    with open(OUTPUT_MD, 'w') as f:
        f.write("# Comprehensive Fine-Tuning Analysis\n\n")

        f.write("## 1. In-Domain Performance (Seen Test)\n")
        f.write("Performance on idioms seen during training (split by sentences).\n")
        f.write("Reporting: Mean F1 ± Std (95% Bootstrap CI)\n\n")
        seen_summary = summary[summary['test_set'] == 'Seen'][['task', 'model', 'mean', 'std', 'ci_low', 'ci_high']].copy()
        seen_summary['ci'] = seen_summary.apply(lambda r: f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1)
        f.write(seen_summary[['task', 'model', 'mean', 'std', 'ci']].to_markdown(index=False, floatfmt=".4f"))

        f.write("\n\n## 2. Generalization Performance (Unseen Test)\n")
        f.write("Performance on completely new idioms never seen during training (Zero-Shot Transfer).\n")
        f.write("Reporting: Mean F1 ± Std (95% Bootstrap CI)\n\n")
        unseen_summary = summary[summary['test_set'] == 'Unseen'][['task', 'model', 'mean', 'std', 'ci_low', 'ci_high']].copy()
        unseen_summary['ci'] = unseen_summary.apply(lambda r: f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1)
        f.write(unseen_summary[['task', 'model', 'mean', 'std', 'ci']].to_markdown(index=False, floatfmt=".4f"))

        f.write("\n\n" + ttest_report)

        # Executive Summary
        f.write("\n\n## 3. Executive Summary\n")
        best_seen_cls = summary[(summary['test_set']=='Seen') & (summary['task']=='cls')].iloc[0]
        best_seen_span = summary[(summary['test_set']=='Seen') & (summary['task']=='span')].iloc[0]
        best_unseen_cls = summary[(summary['test_set']=='Unseen') & (summary['task']=='cls')].iloc[0]
        best_unseen_span = summary[(summary['test_set']=='Unseen') & (summary['task']=='span')].iloc[0]

        f.write(f"- **Best In-Domain (CLS):** {best_seen_cls['model']} ({best_seen_cls['mean']:.4f} ± {best_seen_cls['std']:.4f}, 95% CI: [{best_seen_cls['ci_low']:.4f}, {best_seen_cls['ci_high']:.4f}])\n")
        f.write(f"- **Best In-Domain (SPAN):** {best_seen_span['model']} ({best_seen_span['mean']:.4f} ± {best_seen_span['std']:.4f}, 95% CI: [{best_seen_span['ci_low']:.4f}, {best_seen_span['ci_high']:.4f}])\n")
        f.write(f"- **Best Generalization (CLS):** {best_unseen_cls['model']} ({best_unseen_cls['mean']:.4f} ± {best_unseen_cls['std']:.4f}, 95% CI: [{best_unseen_cls['ci_low']:.4f}, {best_unseen_cls['ci_high']:.4f}])\n")
        f.write(f"- **Best Generalization (SPAN):** {best_unseen_span['model']} ({best_unseen_span['mean']:.4f} ± {best_unseen_span['std']:.4f}, 95% CI: [{best_unseen_span['ci_low']:.4f}, {best_unseen_span['ci_high']:.4f}])\n")

    # Save statistical significance log
    with open(SIG_FILE, 'w') as f:
        f.write(ttest_report)

    print(f"   ✅ Generated markdown report: {OUTPUT_MD}")
    print(f"   ✅ Generated significance log: {SIG_FILE}")

    # Generate LaTeX tables for paper
    print("📄 Generating LaTeX tables for paper...")
    latex_content = create_latex_tables(summary)
    with open(LATEX_TABLE, 'w') as f:
        f.write(latex_content)
    print(f"   ✅ Generated LaTeX tables: {LATEX_TABLE}")

    if args.create_figures:
        print("📊 Generating model comparison figures...")
        create_comparison_figures(summary)

    print("\n✅ Analysis complete!")
    print(f"\nOutputs:")
    print(f"  - Analysis: {OUTPUT_CSV}")
    print(f"  - Report: {OUTPUT_MD}")
    print(f"  - Statistics: {SIG_FILE}")
    print(f"  - Paper: {LATEX_TABLE}")


if __name__ == "__main__":
    main()
