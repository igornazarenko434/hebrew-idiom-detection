#!/usr/bin/env python3
"""
Cross-task inconsistency analysis (CLS vs SPAN) on the same split/model.
Highlights cases where CLS is correct but SPAN fails (and vice-versa).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def find_best_seed(split: str, model: str, task: str) -> int | None:
    eval_root = Path(f"experiments/results/evaluation/{split}/{model}/{task}")
    if not eval_root.exists():
        return None
    best_seed = None
    best_f1 = -1.0
    for seed_dir in eval_root.glob("seed_*"):
        seed = int(seed_dir.name.replace("seed_", ""))
        checkpoint = Path(f"experiments/results/full_fine-tuning/{model}/{task}/seed_{seed}")
        if not checkpoint.exists():
            continue
        for p in seed_dir.glob("eval_results_*.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            f1 = data.get("metrics", {}).get("f1")
            if f1 is None:
                continue
            if f1 > best_f1:
                best_f1 = f1
                best_seed = seed
    return best_seed


def load_predictions(split: str, model: str, task: str, seed: int) -> List[Dict]:
    path = Path(
        f"experiments/results/evaluation/{split}/{model}/{task}/seed_{seed}/eval_predictions.json"
    )
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def get_models_with_both_tasks() -> List[str]:
    root = Path("experiments/results/full_fine-tuning")
    models = []
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        if (model_dir / "cls").exists() and (model_dir / "span").exists():
            models.append(model_dir.name)
    return sorted(models)


def categorize(cls_correct: bool, span_correct: bool) -> str:
    if cls_correct and span_correct:
        return "both_correct"
    if cls_correct and not span_correct:
        return "cls_only"
    if not cls_correct and span_correct:
        return "span_only"
    return "both_wrong"


def plot_consistency(df: pd.DataFrame, out_path: Path, title: str) -> None:
    order = ["both_correct", "cls_only", "span_only", "both_wrong"]
    pivot = df.pivot(index="model", columns="category", values="percent").fillna(0.0)
    pivot = pivot[order]
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 4), colormap="tab20c")
    ax.set_ylabel("Percentage of samples")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    output_root = Path("experiments/results/analysis/consistency")
    paper_root = Path("paper/figures/consistency")
    output_root.mkdir(parents=True, exist_ok=True)
    paper_root.mkdir(parents=True, exist_ok=True)

    rows = []
    example_rows = []
    models = get_models_with_both_tasks()
    splits = ["seen_test", "unseen_test"]

    for split in splits:
        for model in models:
            seed_cls = find_best_seed(split, model, "cls")
            seed_span = find_best_seed(split, model, "span")
            if seed_cls is None or seed_span is None:
                continue
            preds_cls = load_predictions(split, model, "cls", seed_cls)
            preds_span = load_predictions(split, model, "span", seed_span)
            if not preds_cls or not preds_span:
                continue

            cls_by_id = {str(p.get("id")): p for p in preds_cls}
            span_by_id = {str(p.get("id")): p for p in preds_span}
            common_ids = sorted(set(cls_by_id).intersection(span_by_id))

            counts = {k: 0 for k in ["both_correct", "cls_only", "span_only", "both_wrong"]}
            for ex_id in common_ids:
                cls_p = cls_by_id[ex_id]
                span_p = span_by_id[ex_id]
                cls_correct = bool(cls_p.get("is_correct", cls_p.get("true_label") == cls_p.get("predicted_label")))
                span_correct = span_p.get("error_category") == "PERFECT"
                cat = categorize(cls_correct, span_correct)
                counts[cat] += 1

            total = len(common_ids) if common_ids else 1
            for cat, cnt in counts.items():
                rows.append({
                    "split": split,
                    "model": model,
                    "category": cat,
                    "count": cnt,
                    "percent": 100.0 * cnt / total,
                })

            # Collect representative examples for cls_only and span_only
            for cat in ["cls_only", "span_only"]:
                examples = []
                for ex_id in common_ids:
                    cls_p = cls_by_id[ex_id]
                    span_p = span_by_id[ex_id]
                    cls_correct = bool(cls_p.get("is_correct", cls_p.get("true_label") == cls_p.get("predicted_label")))
                    span_correct = span_p.get("error_category") == "PERFECT"
                    if categorize(cls_correct, span_correct) == cat:
                        examples.append((ex_id, cls_p, span_p))
                # prefer high-confidence CLS for cls_only
                if cat == "cls_only":
                    examples.sort(key=lambda x: x[1].get("confidence", 0), reverse=True)
                for ex_id, cls_p, span_p in examples[:5]:
                    example_rows.append({
                        "split": split,
                        "model": model,
                        "category": cat,
                        "id": ex_id,
                        "sentence": cls_p.get("sentence") or span_p.get("sentence"),
                        "cls_true": cls_p.get("true_label"),
                        "cls_pred": cls_p.get("predicted_label"),
                        "cls_confidence": cls_p.get("confidence"),
                        "span_error_category": span_p.get("error_category"),
                    })

        # Plot per split
        split_df = pd.DataFrame([r for r in rows if r["split"] == split])
        if not split_df.empty:
            plot_consistency(
                split_df,
                paper_root / f"consistency_{split}.png",
                f"Cross-Task Consistency (CLS vs SPAN) — {split.replace('_', ' ').title()}",
            )

    if rows:
        pd.DataFrame(rows).to_csv(output_root / "cross_task_consistency.csv", index=False)
    if example_rows:
        pd.DataFrame(example_rows).to_csv(output_root / "cross_task_examples.csv", index=False)

    # Short summary
    summary = [
        "# Cross-Task Consistency Summary",
        "",
        "- Categories: both_correct, cls_only, span_only, both_wrong",
        "- CLS-only highlights detection without localization",
        "- SPAN-only highlights localization despite CLS failure",
        "",
        "Outputs:",
        "- experiments/results/analysis/consistency/cross_task_consistency.csv",
        "- experiments/results/analysis/consistency/cross_task_examples.csv",
        "- paper/figures/consistency/consistency_<split>.png",
    ]
    (output_root / "cross_task_summary.md").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
