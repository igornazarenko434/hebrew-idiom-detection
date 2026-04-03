#!/usr/bin/env python3
"""
Confidence calibration vs error type.
CLS: uses stored confidence from eval_predictions.json.
SPAN: recomputes confidence from best checkpoint logits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from analyze_token_importance import (
    compute_span_confidence,
    ensure_hf_cache,
    load_models,
    parse_tokens,
    read_dataset,
)


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


def get_models_with_tasks(task: str) -> List[str]:
    root = Path("experiments/results/full_fine-tuning")
    models = []
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        if (model_dir / task).exists():
            models.append(model_dir.name)
    return sorted(models)


def plot_confidence(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    df = df.sort_values("mean_confidence", ascending=False)
    plt.figure(figsize=(9, 4))
    plt.bar(df["error_category"], df["mean_confidence"], yerr=df["std_confidence"], color="#2c7fb8")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean confidence")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence calibration vs error type.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    ensure_hf_cache()

    out_root = Path("experiments/results/analysis/calibration")
    paper_root = Path("paper/figures/calibration")
    out_root.mkdir(parents=True, exist_ok=True)
    paper_root.mkdir(parents=True, exist_ok=True)

    splits = ["seen_test", "unseen_test"]

    # CLS confidence (from eval preds)
    for split in splits:
        for model in get_models_with_tasks("cls"):
            seed = find_best_seed(split, model, "cls")
            if seed is None:
                continue
            preds = load_predictions(split, model, "cls", seed)
            if not preds:
                continue
            rows = []
            for p in preds:
                cat = p.get("error_category")
                if not cat:
                    # derive if missing
                    true_label = p.get("true_label")
                    pred_label = p.get("predicted_label")
                    if true_label == pred_label:
                        cat = "CORRECT"
                    elif pred_label == 1:
                        cat = "FP"
                    else:
                        cat = "FN"
                rows.append({
                    "error_category": cat,
                    "confidence": p.get("confidence"),
                })
            df = pd.DataFrame(rows).dropna()
            if df.empty:
                continue
            agg = df.groupby("error_category")["confidence"].agg(["mean", "std", "count"]).reset_index()
            agg = agg.rename(columns={"mean": "mean_confidence", "std": "std_confidence"})
            out_dir = out_root / "cls" / model / split
            out_dir.mkdir(parents=True, exist_ok=True)
            agg.to_csv(out_dir / "confidence_by_error.csv", index=False)
            plot_confidence(
                agg,
                paper_root / "cls" / model / split / "confidence_by_error.png",
                f"CLS Confidence vs Error Type — {model} ({split})",
            )

    # SPAN confidence (recomputed)
    for split in splits:
        for model in get_models_with_tasks("span"):
            seed = find_best_seed(split, model, "span")
            if seed is None:
                continue
            preds = load_predictions(split, model, "span", seed)
            if not preds:
                continue
            df_data = read_dataset(split)
            token_by_id = {
                str(row["id"]): parse_tokens(row.get("tokens", []))
                for _, row in df_data.iterrows()
                if "id" in row
            }
            checkpoint = Path(f"experiments/results/full_fine-tuning/{model}/span/seed_{seed}")
            tokenizer, model_obj, label_maps = load_models(model, "span", checkpoint, args.device)
            label2id = label_maps[0] if label_maps else {}

            rows = []
            for p in preds:
                ex_id = str(p.get("id"))
                tokens = token_by_id.get(ex_id) or parse_tokens(p.get("tokens", []))
                if not tokens:
                    tokens = str(p.get("sentence", "")).split()
                conf = compute_span_confidence(
                    tokenizer,
                    model_obj,
                    tokens,
                    args.device,
                    label2id,
                    max_length=args.max_length,
                )
                cat = p.get("error_category") or "UNKNOWN"
                rows.append({"error_category": cat, "confidence": conf})

            df = pd.DataFrame(rows)
            if df.empty:
                continue
            agg = df.groupby("error_category")["confidence"].agg(["mean", "std", "count"]).reset_index()
            agg = agg.rename(columns={"mean": "mean_confidence", "std": "std_confidence"})
            out_dir = out_root / "span" / model / split
            out_dir.mkdir(parents=True, exist_ok=True)
            agg.to_csv(out_dir / "confidence_by_error.csv", index=False)
            plot_confidence(
                agg,
                paper_root / "span" / model / split / "confidence_by_error.png",
                f"SPAN Confidence vs Error Type — {model} ({split})",
            )

    summary = [
        "# Confidence Calibration Summary",
        "",
        "- CLS confidence uses stored prediction probabilities",
        "- SPAN confidence recomputed from logits per example",
        "",
        "Outputs:",
        "- experiments/results/analysis/calibration/cls/<model>/<split>/confidence_by_error.csv",
        "- experiments/results/analysis/calibration/span/<model>/<split>/confidence_by_error.csv",
        "- paper/figures/calibration/<task>/<model>/<split>/confidence_by_error.png",
    ]
    (out_root / "confidence_summary.md").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
