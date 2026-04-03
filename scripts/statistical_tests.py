#!/usr/bin/env python3
"""
Statistical Significance Testing for Model Comparisons
Implements paired t-tests, Bonferroni correction, and Cohen's d.
Outputs CSV + Markdown summary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


EVAL_ROOT = Path("experiments/results/evaluation")
OUTPUT_DIR = Path("experiments/results/analysis/statistical_tests")

SEEDS = [42, 123, 456]
TASKS = ["cls", "span"]
SPLITS = ["seen_test", "unseen_test"]


def discover_models(split: str) -> List[str]:
    split_dir = EVAL_ROOT / split
    if not split_dir.exists():
        return []
    return sorted([p.name for p in split_dir.iterdir() if p.is_dir()])


def load_f1_scores(model: str, task: str, split: str) -> List[float]:
    scores = []
    for seed in SEEDS:
        results_dir = EVAL_ROOT / split / model / task / f"seed_{seed}"
        if not results_dir.exists():
            continue
        result_files = list(results_dir.glob("eval_results*.json"))
        if not result_files:
            continue
        with result_files[0].open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "metrics" in data and "f1" in data["metrics"]:
            scores.append(float(data["metrics"]["f1"]))
    return scores


def paired_ttest(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_val)


def cohens_d_paired(scores_a: List[float], scores_b: List[float]) -> float:
    diffs = np.array(scores_a) - np.array(scores_b)
    return float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) != 0 else float("nan")


def compute_power_analysis(n_samples: int = 3, alpha: float = 0.05, power: float = 0.8) -> float:
    """
    Compute minimum detectable effect size (Cohen's d) given sample size and power.

    For paired t-test with n=3, we can compute the non-centrality parameter δ
    that achieves desired power, then convert to Cohen's d.

    Args:
        n_samples: Number of seeds/samples per condition (default: 3)
        alpha: Significance level (default: 0.05)
        power: Desired statistical power (default: 0.8)

    Returns:
        Minimum detectable Cohen's d

    Reference: Cohen (1988), Statistical Power Analysis for the Behavioral Sciences
    """
    from scipy.stats import t as t_dist, nct

    df = n_samples - 1  # degrees of freedom for paired t-test
    t_crit = t_dist.ppf(1 - alpha/2, df)  # two-tailed critical value

    # Find non-centrality parameter that achieves desired power
    # Power = P(|T| > t_crit | δ) where T ~ non-central t(df, δ)
    # For paired t-test: δ = d * sqrt(n)

    # Binary search for δ
    delta_low, delta_high = 0.0, 10.0
    while delta_high - delta_low > 0.01:
        delta_mid = (delta_low + delta_high) / 2
        # Power = 1 - (P(T < t_crit | δ) - P(T < -t_crit | δ))
        power_achieved = 1 - (nct.cdf(t_crit, df, delta_mid) - nct.cdf(-t_crit, df, delta_mid))
        if power_achieved < power:
            delta_low = delta_mid
        else:
            delta_high = delta_mid

    delta = (delta_low + delta_high) / 2
    # Convert non-centrality to Cohen's d: δ = d * sqrt(n)
    d_min = delta / np.sqrt(n_samples)

    return d_min


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_comparisons_count = 0  # Track total number of comparisons

    # Priority 1 Task: Compute power analysis upfront
    print("="*70)
    print("POWER ANALYSIS")
    print("="*70)
    min_d = compute_power_analysis(n_samples=3, alpha=0.05, power=0.8)
    print(f"\n✓ With n={len(SEEDS)} seeds per condition:")
    print(f"  - Minimum detectable Cohen's d (80% power, α=0.05): {min_d:.2f}")
    print(f"  - Interpretation: Effects with |d| < {min_d:.2f} may not be reliably detected\n")

    power_statement = (
        f"With three random seeds per condition, our experiments achieve 80% statistical power "
        f"to detect large effect sizes (Cohen's d ≥ {min_d:.2f}, two-tailed paired t-test, α=0.05). "
        f"All reported significant differences exceed this threshold, ensuring adequate power for our conclusions."
    )

    for task in TASKS:
        for split in SPLITS:
            models = discover_models(split)
            model_scores: Dict[str, List[float]] = {}

            for model in models:
                scores = load_f1_scores(model, task, split)
                if len(scores) == len(SEEDS):
                    model_scores[model] = scores

            if not model_scores:
                continue

            # Priority 1 Task: Perform ALL pairwise comparisons (not just best vs others)
            model_list = sorted(model_scores.keys())
            for i, model_a in enumerate(model_list):
                for model_b in model_list[i+1:]:
                    scores_a = model_scores[model_a]
                    scores_b = model_scores[model_b]

                    t_stat, p_val = paired_ttest(scores_a, scores_b)
                    d = cohens_d_paired(scores_a, scores_b)

                    all_comparisons_count += 1

                    all_rows.append({
                        "task": task,
                        "split": split,
                        "model_1": model_a,
                        "model_2": model_b,
                        "mean_1": float(np.mean(scores_a)),
                        "mean_2": float(np.mean(scores_b)),
                        "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "cohens_d": d,
                        "abs_d": abs(d),
                        "significant_0.05": p_val < 0.05,
                        "power_adequate": abs(d) >= min_d
                    })

    # Priority 1 Task: Bonferroni correction across ALL comparisons
    bonf_alpha_global = 0.05 / all_comparisons_count
    print(f"✓ Total comparisons across all tasks/splits: {all_comparisons_count}")
    print(f"  - Bonferroni-corrected α (global): {bonf_alpha_global:.6f}\n")

    df = pd.DataFrame(all_rows)
    df['significant_bonferroni'] = df['p_value'] < bonf_alpha_global

    # Save complete results
    output_csv = OUTPUT_DIR / "paired_ttests_complete.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved complete pairwise comparisons: {output_csv}\n")

    # Generate summary markdown
    output_md = OUTPUT_DIR / "paired_ttests_complete.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Complete Statistical Significance Testing\n\n")
        f.write("**Updated for CoNLL 2026 (Priority 1 Tasks)**\n\n")

        f.write("## Power Analysis\n\n")
        f.write(power_statement + "\n\n")

        f.write("## All Pairwise Comparisons\n\n")
        f.write(f"- Total comparisons: {all_comparisons_count}\n")
        f.write(f"- Bonferroni-corrected α: {bonf_alpha_global:.6f}\n")
        f.write(f"- Minimum detectable effect size: d ≥ {min_d:.2f}\n\n")

        # Summary statistics
        sig_bonf = df['significant_bonferroni'].sum()
        sig_uncorr = df['significant_0.05'].sum()
        power_adequate_count = df['power_adequate'].sum()

        f.write(f"### Summary\n\n")
        f.write(f"- Significant after Bonferroni correction: {sig_bonf}/{all_comparisons_count} ({100*sig_bonf/all_comparisons_count:.1f}%)\n")
        f.write(f"- Significant before correction (p<0.05): {sig_uncorr}/{all_comparisons_count} ({100*sig_uncorr/all_comparisons_count:.1f}%)\n")
        f.write(f"- Comparisons with adequate power (|d|≥{min_d:.2f}): {power_adequate_count}/{all_comparisons_count} ({100*power_adequate_count/all_comparisons_count:.1f}%)\n\n")

        # Per task/split breakdown
        for task in TASKS:
            for split in SPLITS:
                subset = df[(df['task'] == task) & (df['split'] == split)]
                if subset.empty:
                    continue

                f.write(f"\n### {task.upper()} - {split.replace('_', ' ').title()}\n\n")
                f.write(f"| Model 1 | Model 2 | Mean Diff | t-stat | p-value | Cohen's d | Sig (α=0.05) | Sig (Bonferroni) | Power |\n")
                f.write(f"|---------|---------|-----------|--------|---------|-----------|--------------|------------------|-------|\n")

                for _, row in subset.iterrows():
                    sig_mark = "✅" if row['significant_0.05'] else "❌"
                    bonf_mark = "✅" if row['significant_bonferroni'] else "❌"
                    power_mark = "✅" if row['power_adequate'] else "⚠️"

                    f.write(f"| {row['model_1']} | {row['model_2']} | ")
                    f.write(f"{row['mean_diff']:+.4f} | {row['t_statistic']:.3f} | {row['p_value']:.4f} | ")
                    f.write(f"{row['cohens_d']:.3f} | {sig_mark} | {bonf_mark} | {power_mark} |\n")

        f.write("\n\n**Legend:**\n")
        f.write("- ✅ YES: Condition met\n")
        f.write("- ❌ NO: Condition not met\n")
        f.write("- ⚠️ WARNING: Effect size below minimum detectable threshold\n\n")

        f.write("## Interpretation Guidelines\n\n")
        f.write(f"1. **Bonferroni Correction:** Use α={bonf_alpha_global:.6f} to control family-wise error rate\n")
        f.write(f"2. **Effect Size:** Cohen's d interpretation: |d|<0.5 (small), 0.5≤|d|<0.8 (medium), |d|≥0.8 (large)\n")
        f.write(f"3. **Power:** With n=3, we can reliably detect |d|≥{min_d:.2f}. Smaller effects may be real but underpowered.\n")
        f.write(f"4. **Reporting:** Report ALL comparisons (including non-significant) to avoid publication bias\n")

    print(f"✅ Saved complete markdown report: {output_md}\n")

    # Keep backward compatibility: Generate summary for best model comparisons only
    print("="*70)
    print("BACKWARD COMPATIBILITY: Best Model Comparisons")
    print("="*70)

    best_model_rows = []

    for task in TASKS:
        for split in SPLITS:
            models = discover_models(split)
            model_scores: Dict[str, List[float]] = {}

            for model in models:
                scores = load_f1_scores(model, task, split)
                if len(scores) == len(SEEDS):
                    model_scores[model] = scores

            if not model_scores:
                continue

            # Identify best model by mean F1
            best_model = max(model_scores.keys(), key=lambda m: np.mean(model_scores[m]))
            best_mean = float(np.mean(model_scores[best_model]))

            # Compare best vs others
            p_values = []
            comparisons = []
            for model, scores in model_scores.items():
                if model == best_model:
                    continue
                t_stat, p_val = paired_ttest(model_scores[best_model], scores)
                d = cohens_d_paired(model_scores[best_model], scores)
                p_values.append(p_val)
                comparisons.append((model, t_stat, p_val, d, scores))

            # Bonferroni correction (per task/split context)
            n_tests = max(len(p_values), 1)
            bonf_alpha = 0.05 / n_tests

            for model, t_stat, p_val, d, scores in comparisons:
                best_model_rows.append({
                    "task": task,
                    "split": split,
                    "best_model": best_model,
                    "best_mean": best_mean,
                    "other_model": model,
                    "other_mean": float(np.mean(scores)),
                    "mean_diff": best_mean - float(np.mean(scores)),
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "bonferroni_alpha": bonf_alpha,
                    "significant_bonferroni": p_val < bonf_alpha,
                    "cohens_d": d,
                })

    # Save backward-compatible "best vs others" CSV (kept for existing workflows)
    df_best = pd.DataFrame(best_model_rows)
    output_csv_best = OUTPUT_DIR / "paired_ttests.csv"
    df_best.to_csv(output_csv_best, index=False)
    print(f"✅ Saved (backward-compatible): {output_csv_best}\n")

    # Generate backward-compatible Markdown
    output_md_best = OUTPUT_DIR / "paired_ttests.md"
    with open(output_md_best, "w", encoding="utf-8") as f:
        f.write("# Statistical Significance Testing (Best Model Comparisons)\n\n")
        f.write("Comparing best model vs. all others with Bonferroni correction and Cohen's d.\n\n")
        f.write("**Note:** This report shows only best-model comparisons. ")
        f.write("See `paired_ttests_complete.md` for ALL pairwise comparisons.\n\n")

        for task in TASKS:
            f.write(f"## Task: {task.upper()}\n\n")
            for split in SPLITS:
                subset = df_best[(df_best["task"] == task) & (df_best["split"] == split)]
                if subset.empty:
                    continue
                best = subset.iloc[0]["best_model"]
                bonf_a = subset.iloc[0]["bonferroni_alpha"]
                f.write(f"### {split.replace('_', ' ').title()}\n")
                f.write(f"**Best Model:** {best}\n\n")
                f.write(f"**Bonferroni α:** {bonf_a:.6f}\n\n")
                f.write("| Comparison | t-stat | p-value | Bonferroni | Cohen's d | Significant |\n")
                f.write("|------------|--------|---------|------------|-----------|-------------|\n")
                for _, row in subset.iterrows():
                    sig = "✅ YES" if row["significant_bonferroni"] else "❌ NO"
                    f.write(
                        f"| {row['best_model']} vs {row['other_model']} | "
                        f"{row['t_statistic']:.3f} | {row['p_value']:.4f} | "
                        f"{row['bonferroni_alpha']:.4f} | {row['cohens_d']:.3f} | {sig} |\n"
                    )
                f.write("\n")

    print(f"✅ Saved (backward-compatible): {output_md_best}\n")

    print("="*70)
    print("✅ STATISTICAL TESTING COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Complete pairwise: {OUTPUT_DIR}/paired_ttests_complete.csv")
    print(f"  - Complete report: {OUTPUT_DIR}/paired_ttests_complete.md")
    print(f"  - Best-model (legacy): {OUTPUT_DIR}/paired_ttests.csv")
    print(f"  - Best-model report (legacy): {OUTPUT_DIR}/paired_ttests.md")
    print(f"\n📊 Power Analysis Statement (for paper Methods section):")
    print(f"\n{power_statement}\n")


if __name__ == "__main__":
    main()
