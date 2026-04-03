#!/usr/bin/env python3
"""
Contextual distractors analysis for SPAN:
Find error cases where IG mass concentrates on non-idiom context tokens.

Outputs:
  experiments/results/analysis/contextual_distractors/
    - contextual_distractors.csv
    - selected_cases.csv
    - summary.md
    - figures/<model>/<split>/context_distractor_<id>.png
    - html/<model>/<split>/context_distractor_<id>.html
  paper/figures/interpretability/contextual_distractors/<model>/<split>/
    - context_distractor_<id>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from analyze_token_importance import (
    compute_attention_scores,
    compute_ig_scores,
    ensure_hf_cache,
    load_models,
    read_dataset,
    render_html,
    select_best_models,
    select_best_seed,
)


def load_predictions(split: str, model: str, seed: int) -> List[Dict]:
    path = Path(
        f"experiments/results/evaluation/{split}/{model}/span/seed_{seed}/eval_predictions.json"
    )
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def split_tokens(sentence: str) -> List[str]:
    return [t for t in str(sentence).split() if t]


def contextual_distractors(
    preds: List[Dict],
    model_name: str,
    split: str,
    tokenizer,
    model_obj,
    label2id: Dict[str, int],
    device: str,
    max_length: int,
    max_cases: int,
) -> Dict[str, List[Dict]]:
    rows = []
    selected = []
    for p in preds:
        if p.get("error_category") in {None, "PERFECT"}:
            continue
        tokens = split_tokens(p.get("sentence", ""))
        true_tags = p.get("true_tags") or []
        if not tokens or len(tokens) != len(true_tags):
            continue
        idiom_idx = {i for i, t in enumerate(true_tags) if t in {"B-IDIOM", "I-IDIOM"}}
        if not idiom_idx:
            continue
        toks, ig_scores = compute_ig_scores(
            tokenizer,
            model_obj,
            tokens,
            "span",
            label2id,
            device,
            max_length=max_length,
        )
        if len(toks) != len(tokens):
            # fall back to computed tokens if aligned
            tokens = toks
        abs_scores = np.abs(np.array(ig_scores))
        idiom_abs = float(abs_scores[list(idiom_idx)].sum())
        context_idx = [i for i in range(len(tokens)) if i not in idiom_idx]
        context_abs = float(abs_scores[context_idx].sum()) if context_idx else 0.0
        ratio = context_abs / max(1e-6, idiom_abs)
        top_idx = list(np.argsort(abs_scores)[::-1][:5])
        top_context = [tokens[i] for i in top_idx if i in context_idx]
        top_idiom = [tokens[i] for i in top_idx if i in idiom_idx]
        rows.append({
            "model": model_name,
            "split": split,
            "id": p.get("id"),
            "error_category": p.get("error_category"),
            "context_ratio": ratio,
            "context_ig_sum": context_abs,
            "idiom_ig_sum": idiom_abs,
            "top_context_tokens": " | ".join(top_context[:5]),
            "top_idiom_tokens": " | ".join(top_idiom[:5]),
            "sentence": p.get("sentence"),
        })

    rows = sorted(rows, key=lambda x: x["context_ratio"], reverse=True)
    for row in rows[:max_cases]:
        selected.append(row)
    return {"rows": rows, "selected": selected}


def save_heatmaps(
    selected: List[Dict],
    preds_by_id: Dict[str, Dict],
    model: str,
    split: str,
    tokenizer,
    model_obj,
    label2id: Dict[str, int],
    device: str,
    max_length: int,
    fig_dir: Path,
    html_dir: Path,
    paper_dir: Path,
):
    from analyze_token_importance import plot_token_heatmap

    fig_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        ex_id = str(row.get("id"))
        pred = preds_by_id.get(ex_id)
        if not pred:
            continue
        tokens = split_tokens(pred.get("sentence", ""))
        true_tags = pred.get("true_tags") or []
        if not tokens or len(tokens) != len(true_tags):
            continue
        toks, ig_scores = compute_ig_scores(
            tokenizer,
            model_obj,
            tokens,
            "span",
            label2id,
            device,
            max_length=max_length,
        )
        att_toks, att_scores = compute_attention_scores(
            tokenizer,
            model_obj,
            tokens,
            "span",
            device,
            max_length=max_length,
        )
        if len(toks) != len(att_toks):
            continue
        fig_path = fig_dir / f"context_distractor_{ex_id}.png"
        plot_token_heatmap(toks, ig_scores, att_scores, fig_path)
        html_path = html_dir / f"context_distractor_{ex_id}.html"
        html_path.write_text(render_html(toks, ig_scores), encoding="utf-8")
        (paper_dir / fig_path.name).write_bytes(fig_path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual distractors analysis (SPAN).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="unseen_test")
    parser.add_argument("--max_cases", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--models", default="", help="Comma-separated model names.")
    args = parser.parse_args()

    ensure_hf_cache()

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = []
        try:
            best = select_best_models()
            for key in [(args.split, "span", "hebrew"), (args.split, "span", "multilingual")]:
                if key in best:
                    models.append(best[key])
        except Exception:
            models = []
        if not models:
            # fallback: all span models
            root = Path("experiments/results/full_fine-tuning")
            models = [d.name for d in root.iterdir() if d.is_dir() and (d / "span").exists()]

    out_root = Path("experiments/results/analysis/contextual_distractors")
    fig_root = out_root / "figures"
    html_root = out_root / "html"
    paper_root = Path("paper/figures/interpretability/contextual_distractors")
    out_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)
    html_root.mkdir(parents=True, exist_ok=True)
    paper_root.mkdir(parents=True, exist_ok=True)

    df = read_dataset(args.split)
    id_to_base = {str(r["id"]): r.get("base_pie", "") for _, r in df.iterrows()}
    id_to_span = {str(r["id"]): r.get("pie_span", "") for _, r in df.iterrows()}

    all_rows = []
    selected_rows = []

    for model in models:
        try:
            seed, _ = select_best_seed(args.split, model, "span")
        except Exception:
            continue
        preds = load_predictions(args.split, model, seed)
        if not preds:
            continue
        checkpoint = Path(
            f"experiments/results/full_fine-tuning/{model}/span/seed_{seed}"
        )
        if not checkpoint.exists():
            continue
        tokenizer, model_obj, label_pair = load_models(model, "span", checkpoint, args.device)
        label2id = label_pair[0] if label_pair else {}

        result = contextual_distractors(
            preds,
            model,
            args.split,
            tokenizer,
            model_obj,
            label2id,
            args.device,
            args.max_length,
            args.max_cases,
        )
        for row in result["rows"]:
            ex_id = str(row.get("id"))
            row["base_pie"] = id_to_base.get(ex_id, "")
            row["pie_span"] = id_to_span.get(ex_id, "")
        for row in result["selected"]:
            ex_id = str(row.get("id"))
            row["base_pie"] = id_to_base.get(ex_id, "")
            row["pie_span"] = id_to_span.get(ex_id, "")
        all_rows.extend(result["rows"])
        selected_rows.extend(result["selected"])

        preds_by_id = {str(p.get("id")): p for p in preds}
        save_heatmaps(
            result["selected"],
            preds_by_id,
            model,
            args.split,
            tokenizer,
            model_obj,
            label2id,
            args.device,
            args.max_length,
            fig_root / model / args.split,
            html_root / model / args.split,
            paper_root / model / args.split,
        )

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_root / "contextual_distractors.csv", index=False)
    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(out_root / "selected_cases.csv", index=False)

    summary_lines = [
        "# Contextual Distractors Summary",
        "",
        "Selected cases where non-idiom context tokens dominate IG in SPAN errors.",
        "",
    ]
    if selected_rows:
        summary_lines.append("Top selected cases:")
        for row in selected_rows[:10]:
            summary_lines.append(
                f"- {row['model']} {row['split']} id={row['id']} "
                f"error={row['error_category']} context_ratio={row['context_ratio']:.2f}"
            )
    (out_root / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
