#!/usr/bin/env python3
"""
Boundary-type stratification + morphology sensitivity analysis.
Generates IG/ATTN heatmaps for SPAN errors (boundary categories) and
surface-form variants of idioms (base_pie vs pie_span).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from analyze_token_importance import (
    compute_attention_scores,
    compute_ig_scores,
    compute_span_confidence,
    ensure_hf_cache,
    load_models,
    plot_token_heatmap,
    read_dataset,
)


BOUNDARY_CATEGORIES = ["PARTIAL_START", "PARTIAL_END", "SHIFT", "WRONG_SPAN", "MULTI_SPAN", "MISS"]


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


def get_models_with_span() -> List[str]:
    root = Path("experiments/results/full_fine-tuning")
    models = []
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        if (model_dir / "span").exists():
            models.append(model_dir.name)
    return sorted(models)


def pick_examples(
    preds: List[Dict],
    df: pd.DataFrame,
    categories: List[str],
    per_category: int,
) -> Dict[str, List[Dict]]:
    id_to_base = {str(r["id"]): r.get("base_pie", "") for _, r in df.iterrows()}
    id_to_span = {str(r["id"]): r.get("pie_span", "") for _, r in df.iterrows()}
    out = {cat: [] for cat in categories}
    for p in preds:
        cat = p.get("error_category")
        if cat in out and len(out[cat]) < per_category:
            p = dict(p)
            ex_id = str(p.get("id"))
            p["base_pie"] = id_to_base.get(ex_id, "")
            p["pie_span"] = id_to_span.get(ex_id, "")
            out[cat].append(p)
    return out


def morphology_variants(
    preds: List[Dict],
    df: pd.DataFrame,
    per_idiom: int,
) -> Dict[str, List[Dict]]:
    id_to_base = {str(r["id"]): r.get("base_pie", "") for _, r in df.iterrows()}
    id_to_span = {str(r["id"]): r.get("pie_span", "") for _, r in df.iterrows()}
    groups = defaultdict(list)
    for p in preds:
        ex_id = str(p.get("id"))
        base = id_to_base.get(ex_id, "")
        span = id_to_span.get(ex_id, "")
        if base and span and base != span:
            p = dict(p)
            p["base_pie"] = base
            p["pie_span"] = span
            groups[base].append(p)

    # keep small, balanced samples per idiom
    result = {}
    for base, items in groups.items():
        correct = [p for p in items if p.get("error_category") in {None, "PERFECT"}]
        errors = [p for p in items if p.get("error_category") not in {None, "PERFECT"}]
        selected = []
        selected.extend(correct[:per_idiom])
        selected.extend(errors[:per_idiom])
        result[base] = selected[: 2 * per_idiom]
    return result


def save_heatmaps(
    model: str,
    split: str,
    out_dir: Path,
    examples: List[Dict],
    tokenizer,
    model_obj,
    label2id: Dict[str, int],
    device: str,
    max_length: int,
    tag: str,
) -> List[Dict]:
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for ex in examples:
        ex_id = str(ex.get("id"))
        tokens = ex.get("tokens")
        if not tokens:
            tokens = str(ex.get("sentence", "")).split()
        toks, scores = compute_ig_scores(
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
        if len(scores) == len(att_scores):
            heat_path = out_dir / f"{tag}_{ex_id}.png"
            plot_token_heatmap(toks, scores, att_scores, heat_path)
        rows.append({
            "model": model,
            "split": split,
            "id": ex_id,
            "error_category": ex.get("error_category"),
            "base_pie": ex.get("base_pie"),
            "pie_span": ex.get("pie_span"),
            "sentence": ex.get("sentence"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary + morphology interpretability analysis.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--per_category", type=int, default=3)
    parser.add_argument("--per_idiom", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    ensure_hf_cache()

    out_root = Path("experiments/results/analysis/interpretability")
    paper_root = Path("paper/figures/interpretability")
    out_root.mkdir(parents=True, exist_ok=True)
    paper_root.mkdir(parents=True, exist_ok=True)

    models = get_models_with_span()
    splits = ["seen_test", "unseen_test"]

    summary_rows = []

    for model in models:
        for split in splits:
            seed = find_best_seed(split, model, "span")
            if seed is None:
                continue
            preds = load_predictions(split, model, "span", seed)
            if not preds:
                continue
            df = read_dataset(split)

            checkpoint = Path(f"experiments/results/full_fine-tuning/{model}/span/seed_{seed}")
            tokenizer, model_obj, label_maps = load_models(model, "span", checkpoint, args.device)
            label2id = label_maps[0] if label_maps else {}

            # Boundary-type stratification
            cat_examples = pick_examples(preds, df, BOUNDARY_CATEGORIES, args.per_category)
            boundary_dir = out_root / "boundary" / model / split
            paper_boundary_dir = paper_root / "boundary" / model / split
            for cat, items in cat_examples.items():
                if not items:
                    continue
                cat_dir = boundary_dir / cat.lower()
                paper_dir = paper_boundary_dir / cat.lower()
                rows = save_heatmaps(
                    model,
                    split,
                    cat_dir,
                    items,
                    tokenizer,
                    model_obj,
                    label2id,
                    args.device,
                    args.max_length,
                    f"{cat.lower()}",
                )
                for r in rows:
                    summary_rows.append({**r, "analysis": "boundary", "category": cat})
                if cat_dir.exists():
                    paper_dir.mkdir(parents=True, exist_ok=True)
                    for p in cat_dir.glob("*.png"):
                        (paper_dir / p.name).write_bytes(p.read_bytes())

            # Morphology sensitivity
            morph_groups = morphology_variants(preds, df, args.per_idiom)
            morph_dir = out_root / "morphology" / model / split
            paper_morph_dir = paper_root / "morphology" / model / split
            for base, items in morph_groups.items():
                if not items:
                    continue
                base_dir = morph_dir / base.replace(" ", "_")
                paper_dir = paper_morph_dir / base.replace(" ", "_")
                rows = save_heatmaps(
                    model,
                    split,
                    base_dir,
                    items,
                    tokenizer,
                    model_obj,
                    label2id,
                    args.device,
                    args.max_length,
                    "morph",
                )
                for r in rows:
                    summary_rows.append({**r, "analysis": "morphology", "category": base})
                if base_dir.exists():
                    paper_dir.mkdir(parents=True, exist_ok=True)
                    for p in base_dir.glob("*.png"):
                        (paper_dir / p.name).write_bytes(p.read_bytes())

    if summary_rows:
        out_csv = out_root / "boundary_morphology_manifest.csv"
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
