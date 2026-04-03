#!/usr/bin/env python3
"""Generate qualitative error showcase table for SPAN unseen test."""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SPLIT = "unseen_test"
DATA_PATH = Path("data/splits/unseen_idiom_test.csv")


def parse_tokens(value) -> List[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return str(value).split()


def load_dataset() -> Dict[str, Dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    rows = {}
    for _, row in df.iterrows():
        rows[str(row["id"])] = {
            "sentence": row["sentence"],
            "tokens": parse_tokens(row.get("tokens")),
            "base_pie": row.get("base_pie", ""),
            "pie_span": row.get("pie_span", ""),
            "label": row.get("label", None),
        }
    return rows


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


def load_predictions(model: str, seed: int) -> List[Dict]:
    path = Path(f"experiments/results/evaluation/{SPLIT}/{model}/span/seed_{seed}/eval_predictions.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_predictions: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def spans_from_tags(tags: List[str]) -> List[Tuple[int, int]]:
    spans = []
    start = None
    for i, tag in enumerate(tags):
        if tag.startswith("B"):
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag.startswith("I"):
            continue
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def span_text(tokens: List[str], spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return ""
    parts = []
    for start, end in spans:
        parts.append(" ".join(tokens[start:end]))
    return " | ".join(parts)


def explain_error(error_category: str, gold: str, pred: str, missing: List[str], extra: List[str]) -> str:
    if error_category == "MISS":
        return "Model predicted no span despite gold idiom."
    if error_category == "PARTIAL_END":
        detail = f" Missing tokens: {', '.join(missing)}." if missing else ""
        return f"Model truncated the idiom ending (end‑boundary miss).{detail}"
    if error_category == "PARTIAL_START":
        detail = f" Missing tokens: {', '.join(missing)}." if missing else ""
        return f"Model missed idiom start boundary.{detail}"
    if error_category == "SHIFT":
        detail = f" Missing tokens: {', '.join(missing)}; extra tokens: {', '.join(extra)}." if missing or extra else ""
        return f"Predicted span shifted from gold idiom.{detail}"
    if error_category == "WRONG_SPAN":
        detail = f" Predicted different tokens: {', '.join(extra)}." if extra else ""
        return f"Predicted a different span than the gold idiom.{detail}"
    if error_category == "MULTI_SPAN":
        return "Predicted multiple spans for a single idiom."
    if error_category == "EXTEND_END":
        detail = f" Extra tokens: {', '.join(extra)}." if extra else ""
        return f"Span extends beyond the gold idiom boundary.{detail}"
    return "Boundary mismatch between gold and predicted span."


def select_errors(
    preds: List[Dict],
    max_total: int,
    max_per_category: int,
    id_to_base: Dict[str, str],
) -> List[Dict]:
    desired_order = ["PARTIAL_END", "MISS", "WRONG_SPAN", "SHIFT", "MULTI_SPAN", "EXTEND_END", "PARTIAL_START"]
    buckets = defaultdict(list)
    for p in preds:
        cat = p.get("error_category", "UNKNOWN")
        if cat != "PERFECT":
            buckets[cat].append(p)

    selected = []
    used_idioms = set()
    for cat in desired_order:
        count = 0
        for item in buckets.get(cat, []):
            if count >= max_per_category:
                break
            idiom = id_to_base.get(item.get("id", ""), "")
            if idiom in used_idioms:
                continue
            used_idioms.add(idiom)
            if len(selected) >= max_total:
                return selected
            selected.append(item)
            count += 1

    if len(selected) < max_total:
        for cat in desired_order:
            for item in buckets.get(cat, []):
                if len(selected) >= max_total:
                    return selected
                if item in selected:
                    continue
                selected.append(item)
    return selected[:max_total]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative error showcase table.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["neodictabert", "xlm-roberta-base"],
        help="Models to include in the showcase.",
    )
    parser.add_argument("--max_errors", type=int, default=10)
    parser.add_argument("--max_per_category", type=int, default=2)
    args = parser.parse_args()

    dataset = load_dataset()
    rows = []

    for model in args.models:
        seed, f1 = select_best_seed_available(model, "span", split="seen_test")
        preds = load_predictions(model, seed)
        selected = select_errors(
            preds,
            args.max_errors // len(args.models) or 1,
            args.max_per_category,
            {k: v.get("base_pie", "") for k, v in dataset.items()},
        )

        for pred in selected:
            ex_id = pred["id"]
            meta = dataset.get(ex_id, {})
            tokens = meta.get("tokens", [])
            true_tags = pred.get("true_tags", [])
            pred_tags = pred.get("predicted_tags", [])
            gold_spans = spans_from_tags(true_tags)
            pred_spans = spans_from_tags(pred_tags)
            gold_text = span_text(tokens, gold_spans)
            pred_text = span_text(tokens, pred_spans)
            gold_tokens = gold_text.split() if gold_text else []
            pred_tokens = pred_text.split() if pred_text else []
            missing = [t for t in gold_tokens if t not in pred_tokens]
            extra = [t for t in pred_tokens if t not in gold_tokens]

            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "seen_f1_seed": f"{f1:.4f}",
                    "id": ex_id,
                    "base_pie": meta.get("base_pie", ""),
                    "sentence": meta.get("sentence", pred.get("sentence", "")),
                    "gold_span": gold_text,
                    "predicted_span": pred_text,
                    "error_category": pred.get("error_category", "UNKNOWN"),
                    "explanation": explain_error(
                        pred.get("error_category", "UNKNOWN"),
                        gold_text,
                        pred_text,
                        missing,
                        extra,
                    ),
                }
            )

    if not rows:
        raise SystemExit("No errors found to showcase.")

    out_root = Path("experiments/results/analysis/qualitative_errors")
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "qualitative_errors.csv", index=False)

    md_lines = [
        "# Qualitative Error Showcase (SPAN, unseen)",
        "",
        df.to_markdown(index=False),
        "",
    ]
    (out_root / "qualitative_errors.md").write_text("\n".join(md_lines))

    paper_dir = Path("paper/tables")
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "qualitative_errors.md").write_text("\n".join(md_lines))

    # Minimal LaTeX table
    latex = df.rename(
        columns={
            "id": "ID",
            "base_pie": "Idiom",
            "gold_span": "Gold",
            "predicted_span": "Pred",
            "error_category": "Error",
            "explanation": "Explanation",
        }
    )[["model", "ID", "Idiom", "Gold", "Pred", "Error", "Explanation"]].to_latex(
        index=False, escape=True
    )
    (paper_dir / "qualitative_errors.tex").write_text(latex)


if __name__ == "__main__":
    main()
