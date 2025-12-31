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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

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

            # Bonferroni correction
            n_tests = max(len(p_values), 1)
            bonf_alpha = 0.05 / n_tests

            for model, t_stat, p_val, d, scores in comparisons:
                all_rows.append({
                    "task": task,
                    "split": split,
                    "model_best": best_model,
                    "model_other": model,
                    "best_mean_f1": best_mean,
                    "other_mean_f1": float(np.mean(scores)),
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "cohens_d": d,
                    "significant_0.05": p_val < 0.05,
                    "bonferroni_alpha": bonf_alpha,
                    "significant_bonferroni": p_val < bonf_alpha,
                })

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "paired_ttests.csv", index=False)

    # Markdown summary
    lines = []
    lines.append("# Statistical Significance Summary")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Tasks: {', '.join(TASKS)}")
    lines.append(f"- Splits: {', '.join(SPLITS)}")
    lines.append(f"- Seeds: {', '.join(map(str, SEEDS))}")
    lines.append("")

    if df.empty:
        lines.append("No complete model sets found (missing seeds or results).")
    else:
        for (task, split), group in df.groupby(["task", "split"]):
            lines.append(f"## {task.upper()} - {split.replace('_', ' ').title()}")
            lines.append(group[[
                "model_best", "model_other", "best_mean_f1", "other_mean_f1",
                "t_statistic", "p_value", "cohens_d", "bonferroni_alpha",
                "significant_0.05", "significant_bonferroni"
            ]].to_markdown(index=False, floatfmt=".4f"))
            lines.append("")

        lines.append("## Interpretation Notes")
        lines.append("- Paired t-test compares matched seeds.")
        lines.append("- Cohen’s d reports effect size (paired).")
        lines.append("- Bonferroni alpha is applied per task/split comparison set.")
        lines.append("")

    (OUTPUT_DIR / "paired_ttests.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"✅ Statistical tests saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
