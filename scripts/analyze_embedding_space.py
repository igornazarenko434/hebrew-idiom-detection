#!/usr/bin/env python3
"""t-SNE embedding space visualization for seen vs unseen idioms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

cache_root = Path("experiments/cache/huggingface")
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(cache_root))
os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))

from transformers import AutoModel, AutoTokenizer




def load_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"id", "sentence", "base_pie", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return df


def select_best_seed(model: str, split: str = "seen_test") -> Tuple[int, float]:
    base = Path("experiments/results/evaluation") / split / model / "cls"
    best_seed = None
    best_f1 = -1.0
    for seed_dir in sorted(base.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_")[-1])
        except ValueError:
            continue
        for eval_file in seed_dir.glob("eval_results_*.json"):
            data = json.loads(eval_file.read_text())
            f1 = float(data.get("metrics", {}).get("f1", -1.0))
            if f1 > best_f1:
                best_f1 = f1
                best_seed = seed
    if best_seed is None:
        raise FileNotFoundError(f"No eval results found for {model} cls {split}")
    return best_seed, best_f1


def load_model_and_tokenizer(model_path: Path, trust_remote_code: bool) -> Tuple[AutoModel, AutoTokenizer]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, fix_mistral_regex=False
        )
    model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return model, tokenizer


@torch.no_grad()
def compute_cls_embeddings(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    sentences: List[str],
    batch_size: int,
    device: str,
    max_length: int,
) -> np.ndarray:
    model.eval()
    model.to(device)
    embeddings: List[np.ndarray] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        cls_vec = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        embeddings.append(cls_vec)
    return np.vstack(embeddings)


def run_tsne(
    embeddings: np.ndarray,
    n_components: int,
    pca_dim: int,
    perplexity: float,
    random_state: int,
) -> np.ndarray:
    pca_dim = min(pca_dim, embeddings.shape[1])
    reduced = PCA(n_components=pca_dim, random_state=random_state).fit_transform(embeddings)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(reduced)


def plot_tsne(points: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    seen_mask = labels == "seen"
    unseen_mask = labels == "unseen"
    ax.scatter(points[seen_mask, 0], points[seen_mask, 1], s=18, alpha=0.7, label="Seen")
    ax.scatter(points[unseen_mask, 0], points[unseen_mask, 1], s=18, alpha=0.7, label="Unseen")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE embedding space visualization (CLS).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["neodictabert", "xlm-roberta-base"],
        help="Model names under experiments/results/full_fine-tuning/<model>/cls",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    seen_df = load_split(Path("data/splits/test.csv"))
    unseen_df = load_split(Path("data/splits/unseen_idiom_test.csv"))

    output_root = Path("experiments/results/analysis/embedding_space")
    paper_root = Path("paper/figures/embedding_space")

    for model in args.models:
        seed, f1 = select_best_seed(model, split="seen_test")
        ckpt = Path(f"experiments/results/full_fine-tuning/{model}/cls/seed_{seed}")
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        trust_remote_code = "neodictabert" in model
        mdl, tok = load_model_and_tokenizer(ckpt, trust_remote_code=trust_remote_code)

        seen_emb = compute_cls_embeddings(
            mdl, tok, seen_df["sentence"].tolist(), args.batch_size, args.device, args.max_length
        )
        unseen_emb = compute_cls_embeddings(
            mdl, tok, unseen_df["sentence"].tolist(), args.batch_size, args.device, args.max_length
        )

        all_emb = np.vstack([seen_emb, unseen_emb])
        labels = np.array(["seen"] * len(seen_emb) + ["unseen"] * len(unseen_emb))
        points = run_tsne(
            all_emb,
            n_components=2,
            pca_dim=args.pca_dim,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )

        out_dir = output_root / model
        out_dir.mkdir(parents=True, exist_ok=True)
        points_df = pd.DataFrame(
            {
                "id": pd.concat([seen_df["id"], unseen_df["id"]], ignore_index=True),
                "split": labels,
                "label": pd.concat([seen_df["label"], unseen_df["label"]], ignore_index=True),
                "base_pie": pd.concat([seen_df["base_pie"], unseen_df["base_pie"]], ignore_index=True),
                "tsne_x": points[:, 0],
                "tsne_y": points[:, 1],
            }
        )
        points_df.to_csv(out_dir / "tsne_points.csv", index=False)

        title = f"{model} (CLS) | seen vs unseen | seed {seed} | F1={f1:.4f}"
        fig_path = out_dir / "tsne_seen_vs_unseen.png"
        plot_tsne(points, labels, fig_path, title)

        paper_path = paper_root / f"{model}_tsne_seen_vs_unseen.png"
        plot_tsne(points, labels, paper_path, f"{model} (CLS) embedding space")

        summary = out_dir / "summary.md"
        summary.write_text(
            "\n".join(
                [
                    "# Embedding Space Visualization",
                    "",
                    f"- Model: {model}",
                    f"- Seed: {seed}",
                    f"- Seen F1 (seed): {f1:.4f}",
                    f"- Seen samples: {len(seen_df)}",
                    f"- Unseen samples: {len(unseen_df)}",
                    f"- Perplexity: {args.perplexity}",
                    f"- PCA dim: {args.pca_dim}",
                    f"- Output: {fig_path}",
                    f"- Paper figure: {paper_path}",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
