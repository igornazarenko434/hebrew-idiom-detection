#!/usr/bin/env python3
"""
Per-Idiom F1 Analysis
Computes per-idiom F1 for all models, tasks, splits, and seeds.

Outputs:
  - experiments/results/analysis/per_idiom_f1/per_idiom_f1_raw.csv
  - experiments/results/analysis/per_idiom_f1/per_idiom_f1_summary.csv
  - experiments/results/analysis/per_idiom_f1/idiom_difficulty_ranking_{task}_{split}.csv
  - experiments/results/analysis/per_idiom_f1/idiom_metadata.csv
  - paper/figures/per_idiom/per_idiom_heatmap_{task}_{split}.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.error_analysis import compute_cls_metrics, compute_span_f1

# Paths
EVAL_ROOT = PROJECT_ROOT / "experiments/results/evaluation"
OUTPUT_DIR = PROJECT_ROOT / "experiments/results/analysis/per_idiom_f1"
FIGURES_DIR = PROJECT_ROOT / "paper/figures/per_idiom"

# Splits
SPLIT_FILES = {
    "seen_test": PROJECT_ROOT / "data/splits/test.csv",
    "unseen_test": PROJECT_ROOT / "data/splits/unseen_idiom_test.csv",
}

# Seeds used for fine-tuning
SEEDS = [42, 123, 456]

# Plot config
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
})


def load_split_metadata(split: str) -> pd.DataFrame:
    """Load split CSV and return id -> base_pie mapping with idiom_id."""
    df = pd.read_csv(SPLIT_FILES[split])
    df = df[["id", "base_pie"]].drop_duplicates()
    df["idiom_id"] = df["id"].str.split("_").str[0].astype(int)
    return df


def discover_models() -> List[str]:
    """Discover available model directories under evaluation splits."""
    models = set()
    for split in SPLIT_FILES.keys():
        split_dir = EVAL_ROOT / split
        if not split_dir.exists():
            continue
        for model_dir in split_dir.iterdir():
            if model_dir.is_dir():
                models.add(model_dir.name)
    return sorted(models)


def load_predictions(pred_file: Path) -> List[Dict]:
    with pred_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics_for_group(task: str, preds: List[Dict]) -> Dict[str, float]:
    if task == "cls":
        metrics = compute_cls_metrics(preds)
        return {
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
    if task == "span":
        metrics = compute_span_f1(preds)
        return {
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
    raise ValueError(f"Unsupported task: {task}")


def compute_per_idiom_f1(model: str, task: str, split: str, meta: pd.DataFrame) -> pd.DataFrame:
    """Compute per-idiom F1 for a model/task/split across available seeds."""
    all_rows: List[Dict] = []

    for seed in SEEDS:
        pred_file = EVAL_ROOT / split / model / task / f"seed_{seed}" / "eval_predictions.json"
        if not pred_file.exists():
            continue

        preds = load_predictions(pred_file)
        df = pd.DataFrame(preds)
        if df.empty or "id" not in df.columns:
            continue

        try:
            df["idiom_id"] = df["id"].str.split("_").str[0].astype(int)
        except Exception:
            continue

        if "idiom_id" not in df.columns:
            continue
        df = df.merge(meta[["id", "base_pie"]], on="id", how="left")

        for idiom_id, group in df.groupby("idiom_id"):
            group_list = group.to_dict("records")
            metrics = compute_metrics_for_group(task, group_list)
            base_pie = group["base_pie"].dropna().iloc[0] if group["base_pie"].notna().any() else ""

            all_rows.append({
                "model": model,
                "task": task,
                "split": split,
                "seed": seed,
                "idiom_id": idiom_id,
                "base_pie": base_pie,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "num_samples": len(group_list),
            })

    return pd.DataFrame(all_rows)


def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-idiom F1 across seeds."""
    summary = (
        raw_df
        .groupby(["model", "task", "split", "idiom_id", "base_pie"])
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            num_samples_total=("num_samples", "sum"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary["f1_std"] = summary["f1_std"].fillna(0.0)
    return summary


def build_idiom_metadata(meta_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Create a stable idiom_id -> base_pie mapping."""
    meta = pd.concat(meta_frames, ignore_index=True)
    meta = meta[["idiom_id", "base_pie"]].drop_duplicates()
    meta = meta.sort_values("idiom_id")
    return meta


def build_difficulty_ranking(summary: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Compute idiom difficulty ranking per task/split (avg across models)."""
    rankings = {}
    for (task, split), group in summary.groupby(["task", "split"]):
        ranked = (
            group.groupby(["idiom_id", "base_pie"])["f1_mean"]
            .mean()
            .sort_values()
            .reset_index()
        )
        rankings[(task, split)] = ranked
    return rankings


def plot_heatmap(summary: pd.DataFrame, task: str, split: str, idiom_order: List[int]) -> None:
    """Plot per-idiom F1 heatmap (models x idioms)."""
    subset = summary[(summary["task"] == task) & (summary["split"] == split)]
    if subset.empty:
        return

    pivot = subset.pivot_table(
        index="model",
        columns="idiom_id",
        values="f1_mean",
        aggfunc="mean",
        fill_value=0.0,
    )

    # Reorder idioms by difficulty
    pivot = pivot.reindex(columns=idiom_order)

    # Sort models by average performance
    model_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.reindex(model_order)

    # Plot
    plt.figure(figsize=(16, 6))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "F1 (Mean across seeds)"},
        linewidths=0.3,
        linecolor="white",
    )
    plt.title(f"Per-Idiom F1 Heatmap ({task.upper()}, {split.replace('_', ' ').title()})", fontweight="bold")
    plt.xlabel("Idiom ID (ordered by difficulty)")
    plt.ylabel("Model")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"per_idiom_heatmap_{task}_{split}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_summary_report(summary_df: pd.DataFrame,
                         rankings: Dict[Tuple[str, str], pd.DataFrame]) -> None:
    """Write a detailed markdown summary of per-idiom results."""
    report_path = OUTPUT_DIR / "per_idiom_f1_report.md"

    def top_bottom(table: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        top = table.head(n)
        bottom = table.tail(n)
        return top, bottom

    def load_span_unseen_predictions() -> List[Dict]:
        preds = []
        for model in discover_models():
            for seed in SEEDS:
                pred_file = EVAL_ROOT / "unseen_test" / model / "span" / f"seed_{seed}" / "eval_predictions.json"
                if pred_file.exists():
                    preds.extend(load_predictions(pred_file))
        return preds

    lines = []
    lines.append("# Per-Idiom F1 Summary Report")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Models: {summary_df['model'].nunique()}")
    lines.append(f"- Tasks: {', '.join(sorted(summary_df['task'].unique()))}")
    lines.append(f"- Splits: {', '.join(sorted(summary_df['split'].unique()))}")
    lines.append(f"- Idioms (Seen): {summary_df[summary_df['split']=='seen_test']['idiom_id'].nunique()}")
    lines.append(f"- Idioms (Unseen): {summary_df[summary_df['split']=='unseen_test']['idiom_id'].nunique()}")
    lines.append("")

    lines.append("## Methodology")
    lines.append("- Per-idiom F1 computed from `eval_predictions.json`.")
    lines.append("- CLS uses macro F1; SPAN uses exact span F1.")
    lines.append("- Aggregation: mean ± std across seeds (42, 123, 456).")
    lines.append("- Difficulty ordering: average F1 across all models (lower = harder).")
    lines.append("")

    lines.append("## How To Read The Heatmaps")
    lines.append("- Columns are **idiom IDs ordered by difficulty** (left = hardest).")
    lines.append("- Rows are models ordered by average performance.")
    lines.append("- Color reflects **mean F1 across seeds** for each model–idiom pair.")
    lines.append("")

    lines.append("## Difficulty Rankings (Per Task/Split)")
    for (task, split), ranked in rankings.items():
        lines.append(f"### {task.upper()} - {split.replace('_', ' ').title()}")
        top, bottom = top_bottom(ranked, n=5)
        lines.append("**Hardest 5 idioms (lowest F1):**")
        lines.append(top.to_markdown(index=False, floatfmt='.4f'))
        lines.append("")
        lines.append("**Easiest 5 idioms (highest F1):**")
        lines.append(bottom.to_markdown(index=False, floatfmt='.4f'))
        lines.append("")

    # Idiom 49 deep check (SPAN Unseen)
    span_unseen_rank = rankings.get(("span", "unseen_test"))
    if span_unseen_rank is not None and not span_unseen_rank.empty:
        idiom_49 = span_unseen_rank[span_unseen_rank["idiom_id"] == 49]
        if not idiom_49.empty:
            preds = load_span_unseen_predictions()
            idiom_49_preds = [p for p in preds if p.get("id", "").startswith("49_")]
            counts = pd.Series([p.get("error_category", "UNKNOWN") for p in idiom_49_preds]).value_counts()

            full_f1 = compute_span_f1(preds)["f1"] if preds else 0.0
            filtered_f1 = compute_span_f1([p for p in preds if not p.get("id", "").startswith("49_")])["f1"] if preds else 0.0

            lines.append("## Idiom 49 Deep Check (SPAN Unseen)")
            lines.append(f"- Idiom 49 average F1 across models: {idiom_49['f1_mean'].iloc[0]:.6f}")
            lines.append("- Error category distribution (all models, all seeds):")
            lines.append(counts.to_frame(name="count").to_markdown())
            lines.append("")
            lines.append("**Impact on overall SPAN Unseen F1:**")
            lines.append(f"- With idiom 49: {full_f1:.4f}")
            lines.append(f"- Without idiom 49: {filtered_f1:.4f}")
            lines.append(f"- Delta: {filtered_f1 - full_f1:.4f}")
            lines.append("")

    lines.append("## Figures")
    lines.append("- `paper/figures/per_idiom/per_idiom_heatmap_cls_seen_test.png`")
    lines.append("- `paper/figures/per_idiom/per_idiom_heatmap_cls_unseen_test.png`")
    lines.append("- `paper/figures/per_idiom/per_idiom_heatmap_span_seen_test.png`")
    lines.append("- `paper/figures/per_idiom/per_idiom_heatmap_span_unseen_test.png`")
    lines.append("")

    lines.append("## Output Files")
    lines.append("- `experiments/results/analysis/per_idiom_f1/per_idiom_f1_raw.csv`")
    lines.append("- `experiments/results/analysis/per_idiom_f1/per_idiom_f1_summary.csv`")
    lines.append("- `experiments/results/analysis/per_idiom_f1/idiom_metadata.csv`")
    lines.append("- `experiments/results/analysis/per_idiom_f1/idiom_difficulty_ranking_{task}_{split}.csv`")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models = discover_models()
    tasks = ["cls", "span"]

    # Load split metadata
    split_meta = {split: load_split_metadata(split) for split in SPLIT_FILES.keys()}
    idiom_meta = build_idiom_metadata(list(split_meta.values()))
    idiom_meta.to_csv(OUTPUT_DIR / "idiom_metadata.csv", index=False)

    # Compute raw per-idiom metrics
    all_raw = []
    for split, meta in split_meta.items():
        for model in models:
            for task in tasks:
                df = compute_per_idiom_f1(model, task, split, meta)
                if not df.empty:
                    all_raw.append(df)
                    print(f"✓ {model}/{task}/{split}: {df['idiom_id'].nunique()} idioms")

    if not all_raw:
        print("No prediction files found. Exiting.")
        return

    raw_df = pd.concat(all_raw, ignore_index=True)
    raw_df.to_csv(OUTPUT_DIR / "per_idiom_f1_raw.csv", index=False)

    # Aggregate across seeds
    summary_df = build_summary(raw_df)
    summary_df.to_csv(OUTPUT_DIR / "per_idiom_f1_summary.csv", index=False)

    # Difficulty rankings
    rankings = build_difficulty_ranking(summary_df)
    for (task, split), ranked in rankings.items():
        out_path = OUTPUT_DIR / f"idiom_difficulty_ranking_{task}_{split}.csv"
        ranked.to_csv(out_path, index=False)

        # Heatmap ordered by difficulty
        idiom_order = ranked["idiom_id"].tolist()
        plot_heatmap(summary_df, task, split, idiom_order)

    write_summary_report(summary_df, rankings)

    print(f"\n✅ Per-idiom analysis saved to {OUTPUT_DIR}")
    print(f"✅ Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
