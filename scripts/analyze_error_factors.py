#!/usr/bin/env python3
"""
Error factor analysis:
  1) Idiom length effect (SPAN): error rate by idiom length.
  2) Boundary directionality (SPAN): start vs end truncation bias.

Outputs:
  experiments/results/analysis/error_factors/
    - idiom_length_effect.csv
    - boundary_directionality.csv
    - figures/idiom_length_effect_<split>.png
    - figures/boundary_directionality_<split>.png
    - summary.md
  paper/figures/error_factors/
    - idiom_length_effect_<split>.png
    - boundary_directionality_<split>.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_dataset(split: str) -> pd.DataFrame:
    path = Path("data/splits/test.csv") if split == "seen_test" else Path("data/splits/unseen_idiom_test.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def select_best_seed(split: str, model: str, task: str) -> tuple[int, float]:
    base = Path(f"experiments/results/evaluation/{split}/{model}/{task}")
    best_seed = None
    best_f1 = -1.0
    if not base.exists():
        raise FileNotFoundError(f"No eval results at {base}")
    for seed_dir in base.glob("seed_*"):
        try:
            seed = int(seed_dir.name.split("_")[-1])
        except ValueError:
            continue
        for eval_file in seed_dir.glob("eval_results_*.json"):
            data = json.loads(eval_file.read_text(encoding="utf-8"))
            f1 = data.get("metrics", {}).get("f1")
            if f1 is None:
                continue
            if f1 > best_f1:
                best_f1 = f1
                best_seed = seed
    if best_seed is None:
        raise FileNotFoundError(f"No valid eval results for {model}/{task}/{split}")
    return best_seed, best_f1


LENGTH_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)]
BOUNDARY_CATEGORIES = ["PARTIAL_START", "PARTIAL_END"]


def get_models_with_span() -> List[str]:
    root = Path("experiments/results/full_fine-tuning")
    models = []
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        if (model_dir / "span").exists():
            models.append(model_dir.name)
    return sorted(models)


def load_predictions(split: str, model: str, seed: int) -> List[Dict]:
    path = Path(
        f"experiments/results/evaluation/{split}/{model}/span/seed_{seed}/eval_predictions.json"
    )
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def idiom_length(text: str) -> int:
    if not isinstance(text, str):
        return 0
    tokens = [t for t in text.strip().split() if t]
    return len(tokens)


def assign_length_bin(length: int) -> str:
    for low, high in LENGTH_BINS:
        if low <= length <= high:
            return f"{low}" if low == high else f"{low}+"
    return "unknown"


def plot_length_effect(df: pd.DataFrame, split: str, output_path: Path) -> None:
    if df.empty:
        return
    order = [f"{b[0]}" if b[0] == b[1] else f"{b[0]}+" for b in LENGTH_BINS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, group in df.groupby("model"):
        series = group.set_index("length_bin").reindex(order)
        ax.plot(order, series["error_rate"], marker="o", label=model)
    ax.set_title(f"Idiom Length Effect (SPAN) - {split}")
    ax.set_xlabel("Idiom length (tokens)")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_boundary_directionality(df: pd.DataFrame, split: str, output_path: Path) -> None:
    if df.empty:
        return
    models = sorted(df["model"].unique())
    start_vals = []
    end_vals = []
    for model in models:
        sub = df[df["model"] == model].set_index("category")
        start = sub.loc["PARTIAL_START", "count"] if "PARTIAL_START" in sub.index else 0
        end = sub.loc["PARTIAL_END", "count"] if "PARTIAL_END" in sub.index else 0
        denom = max(1, start + end)
        start_vals.append(start / denom)
        end_vals.append(end / denom)
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, start_vals, width, label="PARTIAL_START")
    ax.bar(x + width / 2, end_vals, width, label="PARTIAL_END")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Proportion of boundary errors")
    ax.set_title(f"Boundary Directionality (SPAN) - {split}")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Idiom length + boundary directionality analysis.")
    parser.add_argument("--splits", default="seen_test,unseen_test")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    models = get_models_with_span()

    out_root = Path("experiments/results/analysis/error_factors")
    fig_root = out_root / "figures"
    paper_root = Path("paper/figures/error_factors")
    out_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)
    paper_root.mkdir(parents=True, exist_ok=True)

    length_rows = []
    boundary_rows = []

    for split in splits:
        df = read_dataset(split)
        id_to_base = {str(r["id"]): r.get("base_pie", "") for _, r in df.iterrows()}
        for model in models:
            try:
                seed, _ = select_best_seed(split, model, "span")
            except Exception:
                continue
            preds = load_predictions(split, model, seed)
            if not preds:
                continue

            length_counts = defaultdict(lambda: {"total": 0, "errors": 0})
            boundary_counts = defaultdict(int)

            for p in preds:
                ex_id = str(p.get("id"))
                base = id_to_base.get(ex_id, "")
                length = idiom_length(base)
                if length == 0:
                    continue
                length_bin = assign_length_bin(length)
                is_error = p.get("error_category") not in {None, "PERFECT"}
                length_counts[length_bin]["total"] += 1
                length_counts[length_bin]["errors"] += 1 if is_error else 0
                cat = p.get("error_category")
                if cat in BOUNDARY_CATEGORIES:
                    boundary_counts[cat] += 1

            for length_bin, stats in length_counts.items():
                total = stats["total"]
                errors = stats["errors"]
                length_rows.append({
                    "model": model,
                    "split": split,
                    "length_bin": length_bin,
                    "total": total,
                    "errors": errors,
                    "error_rate": errors / max(1, total),
                })

            for cat in BOUNDARY_CATEGORIES:
                boundary_rows.append({
                    "model": model,
                    "split": split,
                    "category": cat,
                    "count": boundary_counts.get(cat, 0),
                })

    length_df = pd.DataFrame(length_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    if not length_df.empty:
        length_df.to_csv(out_root / "idiom_length_effect.csv", index=False)
    if not boundary_df.empty:
        boundary_df.to_csv(out_root / "boundary_directionality.csv", index=False)

    for split in splits:
        split_length = length_df[length_df["split"] == split]
        split_boundary = boundary_df[boundary_df["split"] == split]
        plot_length_effect(split_length, split, fig_root / f"idiom_length_effect_{split}.png")
        plot_boundary_directionality(
            split_boundary,
            split,
            fig_root / f"boundary_directionality_{split}.png",
        )
        for name in [
            f"idiom_length_effect_{split}.png",
            f"boundary_directionality_{split}.png",
        ]:
            src = fig_root / name
            if src.exists():
                dst = paper_root / name
                dst.write_bytes(src.read_bytes())

    summary_path = out_root / "summary.md"
    lines = [
        "# Error Factor Analysis Summary",
        "",
        "## Idiom Length Effect",
        "Error rate by idiom length (token count of base_pie).",
        "",
        "## Boundary Directionality",
        "Relative frequency of PARTIAL_START vs PARTIAL_END boundary errors.",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
