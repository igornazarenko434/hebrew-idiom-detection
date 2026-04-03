#!/usr/bin/env python3
"""Extract validation F1 learning curves from TensorBoard logs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

cache_root = Path("experiments/cache")
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


def find_event_files(log_dir: Path) -> List[Path]:
    return sorted(log_dir.glob("events.out.tfevents.*"))


def load_scalars(event_file: Path) -> Dict[str, List[tuple]]:
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        scalars[tag] = ea.Scalars(tag)
    return scalars


def extract_eval_f1(log_dir: Path) -> pd.DataFrame:
    event_files = find_event_files(log_dir)
    if not event_files:
        return pd.DataFrame()

    # Use the first file for scalar history; additional files may overlap.
    scalars = load_scalars(event_files[0])
    eval_f1 = scalars.get("eval/f1", [])
    train_epoch = scalars.get("train/epoch", [])

    if not eval_f1:
        return pd.DataFrame()

    epoch_by_step = {e.step: e.value for e in train_epoch}
    rows = []
    for e in eval_f1:
        steps = [s for s in epoch_by_step.keys() if s <= e.step]
        epoch = epoch_by_step[max(steps)] if steps else None
        rows.append(
            {
                "step": e.step,
                "epoch": epoch,
                "f1": e.value,
            }
        )
    df = pd.DataFrame(rows)
    df["epoch_round"] = df["epoch"].round(2)
    return df


def plot_curves(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, sub in df.groupby("model"):
        sub = sub.sort_values("epoch_round")
        ax.plot(sub["epoch_round"], sub["f1_mean"], marker="o", label=model)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning curves from TensorBoard logs.")
    parser.add_argument("--root", default="experiments/results/full_fine-tuning")
    parser.add_argument("--tasks", nargs="+", default=["cls", "span"])
    parser.add_argument("--epoch_threshold", type=float, default=3.0)
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path("experiments/results/analysis/learning_curves")
    paper_root = Path("paper/figures/learning_curves")
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    curve_rows = []

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        model = model_dir.name
        for task in args.tasks:
            task_dir = model_dir / task
            if not task_dir.exists():
                continue
            for seed_dir in sorted(task_dir.glob("seed_*")):
                log_dir = seed_dir / "logs"
                if not log_dir.exists():
                    continue
                df = extract_eval_f1(log_dir)
                if df.empty:
                    continue
                df["model"] = model
                df["task"] = task
                df["seed"] = seed_dir.name
                curve_rows.append(df)

    if not curve_rows:
        raise SystemExit("No TensorBoard logs with eval/f1 found.")

    curves = pd.concat(curve_rows, ignore_index=True)
    curves.to_csv(output_root / "learning_curves_raw.csv", index=False)

    agg = (
        curves.groupby(["model", "task", "epoch_round"])["f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "f1_mean", "std": "f1_std", "count": "n"})
    )
    agg.to_csv(output_root / "learning_curves_summary.csv", index=False)

    for task in args.tasks:
        sub = agg[agg["task"] == task]
        if sub.empty:
            continue
        fig_path = output_root / f"learning_curves_{task}.png"
        plot_curves(sub, fig_path, f"Validation F1 by Epoch ({task.upper()})")
        paper_path = paper_root / f"learning_curves_{task}.png"
        plot_curves(sub, paper_path, f"Validation F1 by Epoch ({task.upper()})")

    for (model, task), sub in agg.groupby(["model", "task"]):
        early = sub[sub["epoch_round"] <= args.epoch_threshold]
        reached = early["f1_mean"].max() if not early.empty else np.nan
        summary_rows.append(
            {
                "model": model,
                "task": task,
                "max_f1_by_epoch3": reached,
                "epochs_logged": int(sub["epoch_round"].nunique()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "learning_curves_summary_by_epoch3.csv", index=False)

    summary_md = output_root / "learning_curves_summary.md"
    lines = [
        "# Learning Curves Summary",
        "",
        "Validation F1 extracted from TensorBoard logs (train/epoch + eval/f1).",
        "",
        "## Max F1 by Epoch 3",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['model']} ({row['task']}): max F1 ≤ epoch 3 = {row['max_f1_by_epoch3']:.4f}"
        )
    summary_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
