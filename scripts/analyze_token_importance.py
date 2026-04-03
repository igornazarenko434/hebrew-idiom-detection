#!/usr/bin/env python3
"""
Token importance analysis for best Hebrew + multilingual models.

Outputs:
  experiments/results/analysis/token_importance/
    - token_importance_summary.md
    - token_importance_<model>_<task>_<split>.json
    - token_importance_top_tokens_<model>_<task>_<split>.csv
    - figures/token_importance_<model>_<task>_<split>_top_tokens.png
    - html/token_importance_<model>_<task>_<split>_<example_id>.html
  paper/figures/token_importance/
    - token_importance_<model>_<task>_<split>_top_tokens.png
"""

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
from scipy.stats import spearmanr

try:
    from captum.attr import IntegratedGradients
except ImportError as exc:  # pragma: no cover - runtime-only
    raise SystemExit("captum is required. Install with: pip install captum") from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

plt.rcParams["font.family"] = "DejaVu Sans"

from src.idiom_experiment import (  # noqa: E402
    BertCRFForTokenClassification,
    load_tokenizer_safe,
)

from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)

try:
    from bidi.algorithm import get_display
except ImportError:
    get_display = None

def ensure_hf_cache():
    cache_root = Path(".").resolve()
    cache_dir = cache_root / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(cache_dir)
    if not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)


HEBREW_MODELS = {
    "alephbert-base",
    "alephbertgimmel-base",
    "dictabert",
    "neodictabert",
}
MULTILINGUAL_MODELS = {
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
}

SPLITS = {
    "seen_test": Path("data/splits/test.csv"),
    "unseen_test": Path("data/splits/unseen_idiom_test.csv"),
}

TASKS = ["cls", "span"]


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


def select_best_seed(split: str, model: str, task: str) -> Tuple[int, float]:
    results = load_eval_results(split, model, task)
    if not results:
        raise FileNotFoundError(f"No eval results for {model}/{task}/{split}")
    best = max(results, key=lambda x: x[2]["metrics"].get("f1", -1))
    return best[0], best[2]["metrics"].get("f1", 0.0)


def select_best_models() -> Dict[Tuple[str, str, str], str]:
    summary_path = Path("experiments/results/analysis/finetuning_summary.csv")
    if not summary_path.exists():
        raise FileNotFoundError("finetuning_summary.csv not found. Run analysis first.")

    df = pd.read_csv(summary_path)
    best = {}
    for split_label, split_key in [("Seen", "seen_test"), ("Unseen", "unseen_test")]:
        for task in TASKS:
            sub = df[(df["test_set"] == split_label) & (df["task"] == task)]
            hebrew = sub[sub["model"].isin(HEBREW_MODELS)].sort_values("mean", ascending=False)
            multilingual = sub[sub["model"].isin(MULTILINGUAL_MODELS)].sort_values("mean", ascending=False)
            if not hebrew.empty:
                best[(split_key, task, "hebrew")] = hebrew.iloc[0]["model"]
            if not multilingual.empty:
                best[(split_key, task, "multilingual")] = multilingual.iloc[0]["model"]
    return best


def read_dataset(split: str) -> pd.DataFrame:
    path = SPLITS[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)
    return df


def parse_tokens(value) -> List[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return str(value).split()


def load_predictions(split: str, model: str, task: str, seed: int) -> List[Dict]:
    path = Path(
        f"experiments/results/evaluation/{split}/{model}/{task}/seed_{seed}/eval_predictions.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_predictions: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_examples(
    preds: List[Dict],
    df: pd.DataFrame,
    task: str,
    max_examples: int = 15,
    target_expressions: List[str] | None = None,
    confidence_map: Dict[str, float] | None = None,
    min_error_examples: int = 5,
    min_correct_examples: int = 5,
) -> List[Dict]:
    target_expressions = target_expressions or []
    id_to_base = {str(r["id"]): r.get("base_pie", "") for _, r in df.iterrows()}
    id_to_span = {str(r["id"]): r.get("pie_span", "") for _, r in df.iterrows()}

    selected = []
    used_ids = set()
    error_added = 0
    correct_added = 0

    def add_example(example, reason):
        ex_id = str(example.get("id"))
        if ex_id in used_ids:
            return False
        entry = dict(example)
        entry["selection_reason"] = reason
        entry["base_pie"] = id_to_base.get(ex_id, "")
        entry["pie_span"] = id_to_span.get(ex_id, "")
        selected.append(entry)
        used_ids.add(ex_id)
        return True

    def is_error(entry: Dict) -> bool:
        if task == "cls":
            return entry.get("true_label") != entry.get("predicted_label")
        return entry.get("error_category") not in {None, "PERFECT"}

    if task == "cls":
        confidences = [p.get("confidence") for p in preds if p.get("confidence") is not None]
        if confidences:
            high_thr = np.quantile(confidences, 0.9)
            low_thr = np.quantile(confidences, 0.1)
        else:
            high_thr, low_thr = 0.9, 0.6

        # High-confidence errors
        for p in preds:
            if not p.get("is_correct") and p.get("confidence", 0) >= high_thr:
                if error_added >= min_error_examples:
                    continue
                if add_example(p, "high_conf_error"):
                    error_added += 1

        # Low-confidence correct
        for p in preds:
            if p.get("is_correct") and p.get("confidence", 1) <= low_thr:
                if correct_added >= min_correct_examples:
                    continue
                if add_example(p, "low_conf_correct"):
                    correct_added += 1

        # Frequent misclassified idioms
        idiom_counts = defaultdict(int)
        for p in preds:
            if not p.get("is_correct"):
                idiom_counts[id_to_base.get(str(p.get("id")), "")] += 1
        for idiom, _ in sorted(idiom_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            if not idiom:
                continue
            for p in preds:
                if not p.get("is_correct") and id_to_base.get(str(p.get("id")), "") == idiom:
                    if add_example(p, "frequent_idiom_error"):
                        error_added += 1
                    break

        # Ensure enough error cases
        if error_added < min_error_examples:
            error_candidates = [p for p in preds if not p.get("is_correct")]
            error_candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            for p in error_candidates:
                if add_example(p, "ensure_error_case"):
                    error_added += 1
                if error_added >= min_error_examples:
                    break

        # Ensure enough correct cases
        if correct_added < min_correct_examples:
            correct_candidates = [p for p in preds if p.get("is_correct")]
            correct_candidates.sort(key=lambda x: x.get("confidence", 1))
            for p in correct_candidates:
                if add_example(p, "ensure_correct_case"):
                    correct_added += 1
                if correct_added >= min_correct_examples:
                    break

    else:
        # Span: use confidence_map (computed from logits)
        if confidence_map is None:
            confidence_map = {}
        conf_values = list(confidence_map.values())
        if conf_values:
            high_thr = np.quantile(conf_values, 0.9)
            low_thr = np.quantile(conf_values, 0.1)
        else:
            high_thr, low_thr = 0.9, 0.6

        for p in preds:
            ex_id = str(p.get("id"))
            conf = confidence_map.get(ex_id, 0.0)
            if not p.get("is_correct") and conf >= high_thr:
                if error_added >= min_error_examples:
                    continue
                if add_example(p, "high_conf_error"):
                    error_added += 1
            if p.get("is_correct") and conf <= low_thr:
                if correct_added >= min_correct_examples:
                    continue
                if add_example(p, "low_conf_correct"):
                    correct_added += 1

        idiom_counts = defaultdict(int)
        for p in preds:
            if p.get("error_category") and p.get("error_category") != "PERFECT":
                idiom_counts[id_to_base.get(str(p.get("id")), "")] += 1
        for idiom, _ in sorted(idiom_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            if not idiom:
                continue
            for p in preds:
                if p.get("error_category") != "PERFECT" and id_to_base.get(str(p.get("id")), "") == idiom:
                    if add_example(p, "frequent_idiom_error"):
                        error_added += 1
                    break

        # Ensure enough error cases
        if error_added < min_error_examples:
            error_candidates = [p for p in preds if p.get("error_category") not in {None, "PERFECT"}]
            error_candidates.sort(key=lambda x: confidence_map.get(str(x.get("id")), 0.0), reverse=True)
            for p in error_candidates:
                if add_example(p, "ensure_error_case"):
                    error_added += 1
                if error_added >= min_error_examples:
                    break

        # Ensure enough correct cases
        if correct_added < min_correct_examples:
            correct_candidates = [p for p in preds if p.get("error_category") in {None, "PERFECT"}]
            correct_candidates.sort(key=lambda x: confidence_map.get(str(x.get("id")), 1.0))
            for p in correct_candidates:
                if add_example(p, "ensure_correct_case"):
                    correct_added += 1
                if correct_added >= min_correct_examples:
                    break

    # Specific expressions
    for expr in target_expressions:
        for p in preds:
            if id_to_base.get(str(p.get("id")), "") == expr:
                add_example(p, "target_expression")
                break

    # Fill with category coverage if needed
    if len(selected) < max_examples:
        if task == "cls":
            cats = {
                "TP": [p for p in preds if p.get("true_label") == 1 and p.get("predicted_label") == 1],
                "TN": [p for p in preds if p.get("true_label") == 0 and p.get("predicted_label") == 0],
                "FP": [p for p in preds if p.get("true_label") == 0 and p.get("predicted_label") == 1],
                "FN": [p for p in preds if p.get("true_label") == 1 and p.get("predicted_label") == 0],
            }
            for cat, items in cats.items():
                for p in items:
                    add_example(p, f"category_{cat}")
                    if len(selected) >= max_examples:
                        break
        else:
            ordered = ["PERFECT", "MISS", "FALSE_POSITIVE", "PARTIAL_BOTH", "SHIFT", "WRONG_SPAN"]
            for cat in ordered:
                for p in preds:
                    if p.get("error_category") == cat:
                        add_example(p, f"category_{cat}")
                        if len(selected) >= max_examples:
                            break

    return selected[:max_examples]


DEFAULT_UNSEEN_EXPRESSIONS = [
    "רץ אחרי הזנב של עצמו",
    "נשאר מאחור",
    "חצה קו אדום",
    "שבר שתיקה",
    "חתך פינה",
    "איבד את הראש",
]

DEFAULT_SEEN_CLS_EXPRESSIONS = [
    "עשה סצנה",
    "ירה לכל הכיוונים",
    "קיפל את הזנב",
    "החזיק אצבעות",
    "קבר את עצמו",
    "ירד לו האסימון",
]

DEFAULT_SEEN_SPAN_EXPRESSIONS = [
    "הרים את הראש",
    "ירד לו האסימון",
    "הניף דגל לבן",
    "נתן גז",
    "נכנס מתחת לאלונקה",
    "משך בחוטים",
]


def build_target_expression_map(
    user_targets: List[str],
    use_default: bool = True,
) -> Dict[Tuple[str, str], List[str]]:
    if user_targets:
        return {
            ("seen_test", "cls"): user_targets,
            ("seen_test", "span"): user_targets,
            ("unseen_test", "cls"): user_targets,
            ("unseen_test", "span"): user_targets,
        }
    if not use_default:
        return {}
    return {
        ("seen_test", "cls"): DEFAULT_SEEN_CLS_EXPRESSIONS,
        ("seen_test", "span"): DEFAULT_SEEN_SPAN_EXPRESSIONS,
        ("unseen_test", "cls"): DEFAULT_UNSEEN_EXPRESSIONS,
        ("unseen_test", "span"): DEFAULT_UNSEEN_EXPRESSIONS,
    }


def load_models(
    model_name: str,
    task: str,
    checkpoint: Path,
    device: str
):
    trust_remote_code = "neodictabert" in model_name
    tokenizer = load_tokenizer_safe(
        checkpoint,
        trust_remote_code=trust_remote_code,
        fix_mistral_regex="neodictabert" in model_name
    )

    if task == "cls":
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code
        )
        model.to(device)
        model.eval()
        return tokenizer, model, None

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
    return tokenizer, model, (label2id, id2label)


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
    task: str,
    device: str,
    max_length: int = 128,
) -> Tuple[List[str], List[float]]:
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

    if task == "span" and hasattr(model, "transformer"):
        base_model = model.transformer
    elif task == "span" and hasattr(model, "base_model"):
        base_model = model.base_model
    elif task == "span" and hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model

    outputs = base_model(
        **inputs,
        output_attentions=True,
    )
    attentions = outputs.attentions
    if not attentions:
        return tokens, [0.0 for _ in tokens]

    # attentions: tuple of (batch, heads, seq, seq)
    att = torch.stack(attentions).mean(dim=0)  # [batch, heads, seq, seq]
    att = att.mean(dim=1)[0]  # [seq, seq]

    if task == "cls":
        scores = att[0]  # CLS to tokens
    else:
        scores = att.mean(dim=0)  # average attention received

    scores = scores.detach().cpu().numpy()
    word_ids = word_ids if word_ids is not None else list(range(len(tokens)))
    if word_ids:
        token_scores = aggregate_word_attributions(tokens, word_ids, scores)
    else:
        token_scores = list(zip(tokens, scores[: len(tokens)]))
    return [t for t, _ in token_scores], [float(s) for _, s in token_scores]


def compute_span_confidence(
    tokenizer,
    model,
    tokens: List[str],
    device: str,
    label2id: Dict[str, int],
    max_length: int = 128,
) -> float:
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

    if hasattr(model, "transformer") and hasattr(model, "classifier"):
        base = model.transformer
        outputs = base(**inputs)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        logits = model.classifier(hidden)
    else:
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)[0]  # [seq, num_labels]
    if not word_ids:
        return float(probs.max().item())

    word_conf = defaultdict(float)
    word_count = defaultdict(int)
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        max_prob = float(probs[idx].max().item())
        word_conf[word_id] += max_prob
        word_count[word_id] += 1

    confs = []
    for wid, total in word_conf.items():
        confs.append(total / max(1, word_count[wid]))
    if not confs:
        return float(probs.max().item())
    return float(np.mean(confs))


def compute_ig_scores(
    tokenizer,
    model,
    tokens: List[str],
    task: str,
    label2id: Dict[str, int] | None,
    device: str,
    max_length: int = 128,
) -> Tuple[List[str], List[float]]:
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

    def try_resolve(obj):
        if isinstance(obj, torch.nn.Embedding):
            return obj
        if hasattr(obj, "get_input_embeddings"):
            try:
                return obj.get_input_embeddings()
            except NotImplementedError:
                pass
        if hasattr(obj, "encoder") and isinstance(getattr(obj, "encoder"), torch.nn.Embedding):
            return getattr(obj, "encoder")
        if hasattr(obj, "_input_embed_layer"):
            layer = getattr(obj, "_input_embed_layer")
            if isinstance(layer, str):
                if hasattr(obj, layer):
                    return getattr(obj, layer)
            elif layer is not None:
                return layer
        return None

    def resolve_embedding_layer(obj):
        resolved = try_resolve(obj)
        if resolved is not None:
            return resolved
        for attr in ("base_model", "model", "transformer", "bert", "roberta", "encoder"):
            if hasattr(obj, attr):
                candidate = getattr(obj, attr)
                if isinstance(candidate, torch.nn.Embedding):
                    return candidate
                resolved = try_resolve(candidate)
                if resolved is not None:
                    return resolved
        raise RuntimeError("Could not resolve input embedding layer for this model.")

    embed_layer = resolve_embedding_layer(model)

    input_embeds = embed_layer(inputs["input_ids"])
    baseline = torch.zeros_like(input_embeds)

    def forward_cls(embeds, attention_mask, token_type_ids=None):
        out = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out.logits[:, 1].sum().unsqueeze(0)

    def forward_span(embeds, attention_mask, token_type_ids=None):
        if hasattr(model, "transformer") and hasattr(model, "classifier"):
            if token_type_ids is not None:
                out = model.transformer(
                    input_ids=None,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    inputs_embeds=embeds,
                )
            else:
                out = model.transformer(
                    input_ids=None,
                    attention_mask=attention_mask,
                    inputs_embeds=embeds,
                )
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            logits = model.classifier(hidden)
        else:
            out = model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = out.logits

        b_id = label2id.get("B-IDIOM", 1)
        i_id = label2id.get("I-IDIOM", 2)
        return logits[:, :, [b_id, i_id]].sum().unsqueeze(0)

    ig = IntegratedGradients(forward_cls if task == "cls" else forward_span)
    additional_args = (inputs["attention_mask"], inputs.get("token_type_ids"))
    attributions = ig.attribute(
        input_embeds,
        baselines=baseline,
        additional_forward_args=additional_args,
        n_steps=32,
    )

    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    word_ids = word_ids if word_ids is not None else list(range(len(tokens)))
    if word_ids:
        token_scores = aggregate_word_attributions(tokens, word_ids, scores)
    else:
        token_scores = list(zip(tokens, scores[: len(tokens)]))
    return [t for t, _ in token_scores], [float(s) for _, s in token_scores]


def render_html(tokens: List[str], scores: List[float]) -> str:
    max_abs = max(1e-8, max(abs(s) for s in scores))
    spans = []
    for token, score in zip(tokens, scores):
        alpha = min(0.85, abs(score) / max_abs)
        if score >= 0:
            color = f"rgba(66, 133, 244, {alpha})"
        else:
            color = f"rgba(219, 68, 55, {alpha})"
        spans.append(f"<span style='background:{color}; padding:2px 3px; margin:1px; border-radius:3px;'>{token}</span>")
    return (
        "<div dir='rtl' style='unicode-bidi: plaintext; line-height:1.8; font-family:Arial, sans-serif;'>"
        + " ".join(spans)
        + "</div>"
    )


def contains_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in text)


def format_token_label(token: str) -> str:
    if contains_hebrew(token) and get_display:
        return get_display(token)
    return token


def plot_top_tokens(token_scores: Dict[str, List[float]], output_path: Path, title: str, top_k: int = 15):
    agg = {tok: float(np.mean([abs(s) for s in scores])) for tok, scores in token_scores.items()}
    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_k]
    if not top:
        return
    tokens, values = zip(*top)
    tokens = [format_token_label(t) for t in tokens]
    plt.figure(figsize=(10, 5))
    plt.bar(tokens, values, color="#2c7fb8")
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Mean |IG Attribution|")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_token_heatmap(tokens: List[str], ig_scores: List[float], att_scores: List[float], output_path: Path):
    tokens = [format_token_label(t) for t in tokens]
    data = np.array([ig_scores, att_scores], dtype=float)
    # Normalize per row to reveal variation when raw scales differ.
    for idx in range(data.shape[0]):
        row = data[idx]
        std = np.std(row)
        if std < 1e-6:
            data[idx] = np.zeros_like(row)
        else:
            data[idx] = (row - np.mean(row)) / std
    fig, ax = plt.subplots(figsize=(min(14, 0.6 * len(tokens) + 2), 2.8))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["IG", "ATTN"])
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Token importance analysis for best models.")
    parser.add_argument("--device", default="cpu", help="Device: cpu, mps, cuda")
    parser.add_argument("--examples_per_category", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--target_expressions", type=str, default="", help="Comma-separated base_pie expressions.")
    parser.add_argument("--target_expressions_file", type=str, default="", help="Path to text/JSON list of expressions.")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available; falling back to CPU.")
        args.device = "cpu"
    ensure_hf_cache()

    output_root = Path("experiments/results/analysis/token_importance")
    figs_dir = output_root / "figures"
    html_dir = output_root / "html"
    paper_figs = Path("paper/figures/token_importance")
    interpret_figs = output_root / "figures" / "interpretability"
    paper_interpret = Path("paper/figures/interpretability")
    output_root.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    paper_figs.mkdir(parents=True, exist_ok=True)
    interpret_figs.mkdir(parents=True, exist_ok=True)
    paper_interpret.mkdir(parents=True, exist_ok=True)

    best_models = select_best_models()
    target_expressions = []
    if args.target_expressions:
        target_expressions = [e.strip() for e in args.target_expressions.split(",") if e.strip()]
    elif args.target_expressions_file:
        p = Path(args.target_expressions_file)
        if p.exists():
            if p.suffix.lower() in {".json"}:
                target_expressions = json.loads(p.read_text(encoding="utf-8"))
            else:
                target_expressions = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    target_expression_map = build_target_expression_map(target_expressions, use_default=True)

    summary_lines = [
        "# Token Importance Analysis (Mission 3.1)",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Selected Best Models",
    ]

    heatmap_error_budget = 5
    heatmap_correct_budget = 5
    heatmap_counts = defaultdict(lambda: {"error": 0, "correct": 0})

    for (split, task, group), model in best_models.items():
        seed, f1 = select_best_seed(split, model, task)
        summary_lines.append(f"- {split} | {task} | {group}: {model} (seed {seed}, F1={f1:.4f})")

    summary_lines.append("")
    summary_lines.append("## Attribution Examples")

    # Build selection anchor: best Hebrew model per split/task
    selection_anchor = {
        (split, task): model
        for (split, task, group), model in best_models.items()
        if group == "hebrew"
    }

    selected_cases_registry = []

    for (split, task, group), model in best_models.items():
        seed, f1 = select_best_seed(split, model, task)
        checkpoint = Path(f"experiments/results/full_fine-tuning/{model}/{task}/seed_{seed}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        tokenizer, model_obj, label_maps = load_models(model, task, checkpoint, args.device)
        label2id = label_maps[0] if label_maps else None

        df = read_dataset(split)
        token_by_id = {
            str(row["id"]): parse_tokens(row.get("tokens", []))
            for _, row in df.iterrows()
            if "id" in row
        }

        # Build selection set once per split/task using anchor model
        anchor_model = selection_anchor.get((split, task), model)
        anchor_seed, _ = select_best_seed(split, anchor_model, task)
        anchor_preds = load_predictions(split, anchor_model, task, anchor_seed)

        confidence_map = None
        if task == "span":
            confidence_map = {}
            anchor_checkpoint = Path(
                f"experiments/results/full_fine-tuning/{anchor_model}/{task}/seed_{anchor_seed}"
            )
            anchor_tok, anchor_model_obj, anchor_label_maps = load_models(
                anchor_model, task, anchor_checkpoint, args.device
            )
            anchor_label2id = anchor_label_maps[0] if anchor_label_maps else None
            for p in anchor_preds:
                ex_id = str(p.get("id"))
                toks = token_by_id.get(ex_id, parse_tokens(p.get("tokens", [])))
                if not toks:
                    toks = str(p.get("sentence", "")).split()
                confidence_map[ex_id] = compute_span_confidence(
                    anchor_tok,
                    anchor_model_obj,
                    toks,
                    args.device,
                    anchor_label2id,
                    max_length=args.max_length,
                )

        target_for_split_task = target_expression_map.get((split, task), [])
        examples = select_examples(
            anchor_preds,
            df,
            task,
            max_examples=15,
            target_expressions=target_for_split_task,
            confidence_map=confidence_map,
        )

        for ex in examples:
            selected_cases_registry.append({
                "split": split,
                "task": task,
                "anchor_model": anchor_model,
                "id": ex.get("id"),
                "selection_reason": ex.get("selection_reason"),
                "base_pie": ex.get("base_pie"),
                "pie_span": ex.get("pie_span"),
                "sentence": ex.get("sentence"),
                "target_expressions": ", ".join(target_for_split_task),
            })

        token_scores_agg = defaultdict(list)
        att_scores_agg = defaultdict(list)
        example_outputs = []

        for ex in examples:
            ex_id = str(ex.get("id", "unknown"))
            tokens = token_by_id.get(ex_id, parse_tokens(ex.get("tokens", [])))
            if not tokens:
                tokens = str(ex.get("sentence", "")).split()

            try:
                toks, scores = compute_ig_scores(
                    tokenizer,
                    model_obj,
                    tokens,
                    task,
                    label2id,
                    args.device,
                    max_length=args.max_length
                )
            except Exception as exc:
                raise RuntimeError(
                    f"IG failed for model={model}, task={task}, split={split}, id={ex_id}"
                ) from exc

            att_toks, att_scores = compute_attention_scores(
                tokenizer,
                model_obj,
                tokens,
                task,
                args.device,
                max_length=args.max_length,
            )

            for t, s in zip(toks, scores):
                token_scores_agg[t].append(s)
            for t, s in zip(att_toks, att_scores):
                att_scores_agg[t].append(s)

            html = render_html(toks, scores)
            html_path = html_dir / f"token_importance_{model}_{task}_{split}_{ex_id}.html"
            html_path.write_text(html, encoding="utf-8")

            corr = None
            if len(scores) == len(att_scores):
                corr = spearmanr(scores, att_scores).correlation

            example_outputs.append({
                "id": ex_id,
                "category": (
                    ex.get("error_category")
                    if task == "span"
                    else (
                        "TP" if ex.get("true_label") == 1 and ex.get("predicted_label") == 1
                        else "TN" if ex.get("true_label") == 0 and ex.get("predicted_label") == 0
                        else "FP" if ex.get("true_label") == 0 and ex.get("predicted_label") == 1
                        else "FN" if ex.get("true_label") == 1 and ex.get("predicted_label") == 0
                        else "NA"
                    )
                ),
                "sentence": ex.get("sentence", ""),
                "tokens": toks,
                "scores": scores,
                "attention_scores": att_scores,
                "attention_tokens": att_toks,
                "spearman_corr": corr,
                "selection_reason": ex.get("selection_reason"),
                "base_pie": ex.get("base_pie"),
                "pie_span": ex.get("pie_span"),
                "true_label": ex.get("true_label"),
                "predicted_label": ex.get("predicted_label"),
                "true_tags": ex.get("true_tags"),
                "predicted_tags": ex.get("predicted_tags"),
                "html_path": str(html_path)
            })

            heatmap_key = (model, task, split)
            is_error = (
                (task == "cls" and ex.get("true_label") != ex.get("predicted_label"))
                or (task == "span" and ex.get("error_category") not in {None, "PERFECT"})
            )
            if len(scores) == len(att_scores):
                bucket = "error" if is_error else "correct"
                if (
                    (bucket == "error" and heatmap_counts[heatmap_key]["error"] < heatmap_error_budget)
                    or (bucket == "correct" and heatmap_counts[heatmap_key]["correct"] < heatmap_correct_budget)
                ):
                    model_dir = interpret_figs / model / task / split
                    paper_dir = paper_interpret / model / task / split
                    model_dir.mkdir(parents=True, exist_ok=True)
                    paper_dir.mkdir(parents=True, exist_ok=True)
                    heat_path = model_dir / f"heatmap_{ex_id}_{bucket}.png"
                    plot_token_heatmap(toks, scores, att_scores, heat_path)
                    if heat_path.exists():
                        (paper_dir / heat_path.name).write_bytes(heat_path.read_bytes())
                    heatmap_counts[heatmap_key][bucket] += 1

        out_json = output_root / f"token_importance_{model}_{task}_{split}.json"
        out_json.write_text(json.dumps(example_outputs, ensure_ascii=False, indent=2), encoding="utf-8")

        csv_path = output_root / f"token_importance_top_tokens_{model}_{task}_{split}.csv"
        rows = [
            {"token": tok, "mean_abs_importance": float(np.mean([abs(s) for s in vals]))}
            for tok, vals in token_scores_agg.items()
        ]
        if rows:
            pd.DataFrame(rows).sort_values("mean_abs_importance", ascending=False).to_csv(csv_path, index=False)

        fig_path = figs_dir / f"token_importance_{model}_{task}_{split}_top_tokens.png"
        plot_top_tokens(
            token_scores_agg,
            fig_path,
            title=f"Top Tokens by |IG| - {model} ({task}, {split})",
            top_k=15,
        )
        if fig_path.exists():
            (paper_figs / fig_path.name).write_bytes(fig_path.read_bytes())

        att_fig_path = figs_dir / f"attention_importance_{model}_{task}_{split}_top_tokens.png"
        plot_top_tokens(
            att_scores_agg,
            att_fig_path,
            title=f"Top Tokens by Attention - {model} ({task}, {split})",
            top_k=15,
        )
        if att_fig_path.exists():
            (paper_figs / att_fig_path.name).write_bytes(att_fig_path.read_bytes())

        summary_lines.append(
            f"- {split} | {task} | {group}: {model} seed {seed} (F1={f1:.4f}) "
            f"→ {out_json.name}"
        )

    summary_path = output_root / "token_importance_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    # Save selected cases registry
    registry_path = output_root / "selected_cases.csv"
    if selected_cases_registry:
        pd.DataFrame(selected_cases_registry).drop_duplicates().to_csv(registry_path, index=False)

    # Interpretability report
    report_lines = [
        "# Interpretability Analysis (Mission 6.1)",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Selection Criteria",
        "- High-confidence errors",
        "- Low-confidence correct",
        "- Frequent misclassified idioms",
    ]
    if target_expression_map:
        report_lines.append("- Target expressions per split/task:")
        for key in sorted(target_expression_map.keys()):
            report_lines.append(f"  - {key[0]} | {key[1]}: {', '.join(target_expression_map[key])}")
    else:
        report_lines.append("- Target expressions: not provided")
    report_lines.append("")
    report_lines.append("## Files")
    report_lines.append(f"- Selected cases: {registry_path}")
    report_lines.append(f"- Token importance summary: {summary_path}")

    report_path = Path("experiments/results/interpretability_analysis.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"✓ Token importance analysis complete: {summary_path}")


if __name__ == "__main__":
    main()
