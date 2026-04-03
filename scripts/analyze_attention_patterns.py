#!/usr/bin/env python3
"""Attention pattern analysis for SPAN errors (PERFECT vs PARTIAL_END)."""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

cache_root = Path("experiments/cache/huggingface")
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(cache_root))
os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModel

from src.idiom_experiment import BertCRFForTokenClassification, load_tokenizer_safe


SPLITS = {
    "unseen_test": Path("data/splits/unseen_idiom_test.csv"),
}


def load_eval_results(split: str, model: str, task: str) -> List[Tuple[int, Path, Dict]]:
    base = Path(f"experiments/results/evaluation/{split}/{model}/{task}")
    results = []
    if not base.exists():
        return results
    for seed_dir in sorted(base.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_")[-1])
        except ValueError:
            continue
        for eval_file in seed_dir.glob("eval_results_*.json"):
            data = json.loads(eval_file.read_text(encoding="utf-8"))
            results.append((seed, eval_file, data))
    return results


def select_best_seed(model: str, task: str, split: str = "seen_test") -> Tuple[int, float]:
    results = load_eval_results(split, model, task)
    if not results:
        raise FileNotFoundError(f"No eval results for {model}/{task}/{split}")
    best = max(results, key=lambda x: x[2]["metrics"].get("f1", -1))
    return best[0], best[2]["metrics"].get("f1", 0.0)


def available_seeds(model: str, task: str) -> List[int]:
    base = Path(f"experiments/results/full_fine-tuning/{model}/{task}")
    seeds = []
    for seed_dir in sorted(base.glob("seed_*")):
        try:
            seeds.append(int(seed_dir.name.split("_")[-1]))
        except ValueError:
            continue
    return seeds


def select_best_seed_available(model: str, task: str, split: str = "seen_test") -> Tuple[int, float]:
    seeds = set(available_seeds(model, task))
    results = [r for r in load_eval_results(split, model, task) if r[0] in seeds]
    if not results:
        raise FileNotFoundError(f"No matching eval results for available seeds: {model}/{task}/{split}")
    best = max(results, key=lambda x: x[2]["metrics"].get("f1", -1))
    return best[0], best[2]["metrics"].get("f1", 0.0)


def parse_tokens(value) -> List[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return str(value).split()


def load_dataset(split: str) -> Dict[str, List[str]]:
    path = SPLITS[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)
    token_map = {}
    for _, row in df.iterrows():
        token_map[str(row["id"])] = parse_tokens(row.get("tokens"))
    return token_map


def load_predictions(split: str, model: str, seed: int) -> List[Dict]:
    path = Path(
        f"experiments/results/evaluation/{split}/{model}/span/seed_{seed}/eval_predictions.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_predictions: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_span_model(model_name: str, checkpoint: Path, device: str):
    trust_remote_code = "neodictabert" in model_name
    tokenizer = load_tokenizer_safe(
        checkpoint,
        trust_remote_code=trust_remote_code,
        fix_mistral_regex="neodictabert" in model_name,
    )
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    base_model = AutoModel.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    label2id = config.label2id
    id2label = config.id2label
    num_labels = len(label2id)
    model = BertCRFForTokenClassification(base_model, num_labels, label2id, id2label)

    weights_path = checkpoint / "model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
    else:
        weights_path_bin = checkpoint / "pytorch_model.bin"
        if weights_path_bin.exists():
            state_dict = torch.load(weights_path_bin, map_location=device)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint}")

    model.to(device)
    model.eval()
    return tokenizer, model


def aggregate_word_attributions(tokens: List[str], word_ids: List[int], scores: np.ndarray) -> List[Tuple[str, float]]:
    word_scores = defaultdict(float)
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= len(tokens):
            continue
        word_scores[word_id] += float(scores[idx])
    return [(tokens[i], word_scores.get(i, 0.0)) for i in range(len(tokens))]


def compute_attention_scores(
    tokenizer,
    model,
    tokens: List[str],
    device: str,
    max_length: int = 128,
) -> List[float]:
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = tokenized.word_ids(batch_index=0) if hasattr(tokenized, "word_ids") else None
    inputs = tokenized.to(device)

    if hasattr(model, "transformer"):
        base_model = model.transformer
    elif hasattr(model, "base_model"):
        base_model = model.base_model
    elif hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model

    outputs = base_model(
        **inputs,
        output_attentions=True,
    )
    attentions = outputs.attentions
    if not attentions:
        return [0.0 for _ in tokens]

    att = torch.stack(attentions).mean(dim=0)  # [batch, heads, seq, seq]
    att = att.mean(dim=1)[0]  # [seq, seq]
    scores = att.mean(dim=0).detach().cpu().numpy()  # average attention received

    word_ids = word_ids if word_ids is not None else list(range(len(tokens)))
    if word_ids:
        token_scores = aggregate_word_attributions(tokens, word_ids, scores)
    else:
        token_scores = list(zip(tokens, scores[: len(tokens)]))
    return [float(s) for _, s in token_scores]


def compute_idiom_attention_ratio(att_scores: List[float], tags: List[str]) -> float:
    if not att_scores or not tags:
        return 0.0
    n = min(len(att_scores), len(tags))
    idiom_mask = [1 if tags[i].startswith("B") or tags[i].startswith("I") else 0 for i in range(n)]
    total = float(np.sum(att_scores[:n]))
    if total == 0.0:
        return 0.0
    idiom_att = float(np.sum([att_scores[i] for i in range(n) if idiom_mask[i] == 1]))
    return idiom_att / total


def plot_comparison(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    subset = df[df["error_category"].isin(["PERFECT", "PARTIAL_END"])]
    if subset.empty:
        return
    sns.barplot(
        data=subset,
        x="error_category",
        y="idiom_attention_ratio",
        hue="error_category",
        estimator=np.mean,
        errorbar="sd",
        ax=ax,
        palette="Set2",
        legend=False,
    )
    ax.set_title(title)
    ax.set_ylabel("Mean attention on idiom tokens")
    ax.set_xlabel("")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_by_error(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    order = df.groupby("error_category")["idiom_attention_ratio"].mean().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x="error_category",
        y="idiom_attention_ratio",
        order=order,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Attention on idiom tokens")
    ax.set_xlabel("Error category")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention pattern analysis for SPAN errors.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["neodictabert", "xlm-roberta-base"],
        help="Model names under experiments/results/full_fine-tuning/<model>/span",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    split = "unseen_test"
    token_map = load_dataset(split)

    output_root = Path("experiments/results/analysis/attention_patterns")
    paper_root = Path("paper/figures/attention_patterns")
    summary_rows = []

    for model in args.models:
        seed, f1 = select_best_seed_available(model, "span", split="seen_test")
        ckpt = Path(f"experiments/results/full_fine-tuning/{model}/span/seed_{seed}")
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        tokenizer, span_model = load_span_model(model, ckpt, args.device)
        preds = load_predictions(split, model, seed)

        rows = []
        mismatch = 0
        for pred in preds:
            ex_id = pred["id"]
            tokens = token_map.get(ex_id, [])
            true_tags = pred.get("true_tags", [])
            if not tokens or not true_tags:
                continue
            if len(tokens) != len(true_tags):
                mismatch += 1
            att_scores = compute_attention_scores(
                tokenizer, span_model, tokens, args.device, max_length=args.max_length
            )
            ratio = compute_idiom_attention_ratio(att_scores, true_tags)
            rows.append(
                {
                    "id": ex_id,
                    "error_category": pred.get("error_category", "UNKNOWN"),
                    "idiom_attention_ratio": ratio,
                }
            )

        df = pd.DataFrame(rows)
        out_dir = output_root / model / split
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "attention_example_scores.csv", index=False)

        agg = df.groupby("error_category")["idiom_attention_ratio"].agg(["mean", "std", "count"]).reset_index()
        agg.to_csv(out_dir / "attention_by_error.csv", index=False)

        fig1 = out_dir / "attention_ratio_correct_vs_partial_end.png"
        plot_comparison(df, fig1, f"{model} (SPAN, unseen) - Idiom Attention")
        fig2 = out_dir / "attention_ratio_by_error.png"
        plot_by_error(df, fig2, f"{model} (SPAN, unseen) - Attention by Error Type")

        paper_fig = paper_root / f"{model}_attention_ratio_correct_vs_partial_end.png"
        plot_comparison(df, paper_fig, f"{model} (SPAN) - Idiom Attention")

        if "PERFECT" in agg["error_category"].values and "PARTIAL_END" in agg["error_category"].values:
            perf = float(agg.loc[agg["error_category"] == "PERFECT", "mean"].values[0])
            pe = float(agg.loc[agg["error_category"] == "PARTIAL_END", "mean"].values[0])
            drop = (perf - pe) / perf if perf else 0.0
        else:
            perf, pe, drop = 0.0, 0.0, 0.0

        summary_rows.append(
            {
                "model": model,
                "seed": seed,
                "seen_f1_seed": f1,
                "perfect_mean": perf,
                "partial_end_mean": pe,
                "relative_drop": drop,
                "mismatch_count": mismatch,
                "n_examples": len(df),
            }
        )

        summary_md = out_dir / "summary.md"
        summary_md.write_text(
            "\n".join(
                [
                    "# Attention Pattern Analysis",
                    "",
                    f"- Model: {model}",
                    f"- Seed (best on seen): {seed}",
                    f"- Seen F1 (seed): {f1:.4f}",
                    f"- Examples analyzed: {len(df)}",
                    f"- Token length mismatches: {mismatch}",
                    "",
                    "## Key Comparison",
                    f"- PERFECT mean attention: {perf:.4f}",
                    f"- PARTIAL_END mean attention: {pe:.4f}",
                    f"- Relative drop: {drop:.1%}",
                    "",
                    "## Outputs",
                    f"- `attention_by_error.csv`",
                    f"- `attention_example_scores.csv`",
                    f"- `{fig1}`",
                    f"- `{fig2}`",
                ]
            )
            + "\n"
        )

    summary_df = pd.DataFrame(summary_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_root / "attention_summary.csv", index=False)

    summary_md = output_root / "attention_summary.md"
    summary_lines = ["# Attention Pattern Summary", ""]
    for row in summary_rows:
        summary_lines.append(
            f"- {row['model']} (seed {row['seed']}): PERFECT={row['perfect_mean']:.4f}, "
            f"PARTIAL_END={row['partial_end_mean']:.4f}, drop={row['relative_drop']:.1%}"
        )
    summary_md.write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
