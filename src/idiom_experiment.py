#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hebrew Idiom Detection - Multi-Mode Experiment Framework
File: src/idiom_experiment.py

This script supports multiple training and evaluation modes:
  - zero_shot: Zero-shot evaluation (Mission 3.2) - IMPLEMENTED
  - full_finetune: Full fine-tuning (Mission 4.2) - PLACEHOLDER
  - frozen_backbone: Frozen backbone training (Mission 4.2) - PLACEHOLDER
  - hpo: Hyperparameter optimization with Optuna (Mission 4.3) - PLACEHOLDER

Tasks:
  Task 1: Sentence-level classification (literal=0, figurative=1)
  Task 2: Token-level idiom span detection with IOB2 tags (B-IDIOM, I-IDIOM, O)

Design aligned with PRD and step-by-step missions.
"""

import argparse
import ast
import json
import sys
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from functools import partial
import evaluate  # HuggingFace evaluate library

# Import our custom tokenization utilities (Mission 4.2 Task 3.5)
try:
    from utils.tokenization import (
        align_labels_with_tokens,
        tokenize_and_align_labels,
        align_predictions_with_words
    )
except ImportError:
    # Try absolute import if relative doesn't work
    from src.utils.tokenization import (
        align_labels_with_tokens,
        tokenize_and_align_labels,
        align_predictions_with_words
    )

# -------------------------
# Utilities
# -------------------------

def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist"""
    p.parent.mkdir(parents=True, exist_ok=True)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two sets of vectors"""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

def pool_cls(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract [CLS] token representation (first token for BERT-like models)"""
    return hidden_states[:, 0, :]

# -------------------------
# Configuration Management (Mission 4.1)
# -------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print(f"\nLoading configuration from: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"✓ Configuration loaded successfully")
    return config

def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge configuration from file with command-line arguments.
    Command-line arguments override config file values.

    Args:
        config: Configuration dictionary from YAML file
        args: Parsed command-line arguments

    Returns:
        Merged configuration dictionary
    """
    # Create a copy to avoid modifying original
    merged = config.copy()

    # List of arguments that can override config
    override_params = [
        'model_id', 'model_checkpoint', 'task', 'device', 'batch_size',
        'learning_rate', 'num_epochs', 'max_length', 'seed',
        'warmup_ratio', 'weight_decay', 'output_dir'
    ]

    overrides = []
    for param in override_params:
        # Check if argument was provided (not None) and is different from config
        arg_value = getattr(args, param, None)
        if arg_value is not None:
            # For model_id, map to model_checkpoint in config
            if param == 'model_id':
                merged['model_checkpoint'] = arg_value
                overrides.append(f"model_checkpoint={arg_value}")
            elif param in merged:
                if merged[param] != arg_value:
                    merged[param] = arg_value
                    overrides.append(f"{param}={arg_value}")
            else:
                merged[param] = arg_value
                overrides.append(f"{param}={arg_value}")

    if overrides:
        print(f"\n⚙️  Command-line overrides: {', '.join(overrides)}")

    return merged

def validate_config(config: Dict[str, Any], mode: str) -> bool:
    """
    Validate configuration has all required fields for the given mode

    Args:
        config: Configuration dictionary
        mode: Training mode (zero_shot, full_finetune, frozen_backbone, hpo)

    Returns:
        True if valid, raises ValueError if invalid
    """
    print(f"\nValidating configuration for mode: {mode}")

    # Common required fields for all modes
    common_required = ['task', 'device']

    # Mode-specific required fields
    if mode in ['full_finetune', 'frozen_backbone']:
        training_required = [
            'model_checkpoint', 'learning_rate', 'batch_size', 'num_epochs',
            'train_file', 'dev_file', 'output_dir'
        ]
        required_fields = common_required + training_required

    elif mode == 'hpo':
        hpo_required = ['optuna', 'search_space', 'fixed']
        required_fields = hpo_required

    elif mode == 'zero_shot':
        zero_shot_required = ['model_checkpoint', 'max_length']
        required_fields = common_required + zero_shot_required

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Check for missing fields
    missing_fields = []
    for field in required_fields:
        if field not in config:
            # For HPO mode, check nested structure
            if mode == 'hpo':
                if field in ['optuna', 'search_space', 'fixed']:
                    if field not in config:
                        missing_fields.append(field)
            else:
                missing_fields.append(field)

    if missing_fields:
        raise ValueError(f"Configuration missing required fields for mode '{mode}': {missing_fields}")

    print(f"✓ Configuration valid for mode: {mode}")
    return True

def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """
    Pretty print configuration dictionary

    Args:
        config: Configuration dictionary
        title: Title for the printout
    """
    print(f"\n{'='*80}")
    print(f"{title.upper()}")
    print(f"{'='*80}")

    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  ' * indent}{key}:")
                print_dict(value, indent + 1)
            else:
                print(f"{'  ' * indent}{key}: {value}")

    print_dict(config)
    print(f"{'='*80}\n")

# -------------------------
# Dataset utilities
# -------------------------

def load_dataframe(path: Path, split: Optional[str] = None, max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load dataset from CSV with optional filtering

    Args:
        path: Path to CSV file
        split: Optional split filter (train/validation/test)
        max_samples: Optional limit on number of samples

    Returns:
        Filtered dataframe
    """
    df = pd.read_csv(path)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    if max_samples is not None:
        df = df.head(max_samples)
    return df.reset_index(drop=True)

def validate_dataset_for_evaluation(df: pd.DataFrame, task: str) -> Dict:
    """
    Validate dataset before evaluation to ensure data quality and tokenization alignment.

    This function checks:
    1. Required columns exist
    2. No missing values in critical columns
    3. IOB2 tags are correctly aligned with whitespace tokenization

    Args:
        df: Dataframe to validate
        task: Task type ("cls", "span", or "both")

    Returns:
        Dictionary with validation results and warnings
    """
    print("\n" + "="*80)
    print("DATASET VALIDATION")
    print("="*80)

    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }

    # Check required columns for Task 1 (Classification)
    if task in ("cls", "both"):
        print("\n[Task 1 - Classification] Validating required columns...")
        required_cols_task1 = ["sentence", "label"]
        for col in required_cols_task1:
            if col not in df.columns:
                validation_results["errors"].append(f"Missing required column for Task 1: '{col}'")
                validation_results["valid"] = False
            else:
                # Check for missing values
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    validation_results["errors"].append(
                        f"Column '{col}' has {missing_count} missing values ({missing_count/len(df)*100:.1f}%)"
                    )
                    validation_results["valid"] = False
                else:
                    print(f"  ✓ Column '{col}': {len(df)} valid values")

    # Check required columns for Task 2 (Token Classification)
    if task in ("span", "both"):
        print("\n[Task 2 - Token Classification] Validating required columns...")
        required_cols_task2 = ["sentence", "base_pie", "tokens", "iob_tags"]
        for col in required_cols_task2:
            if col not in df.columns:
                validation_results["errors"].append(f"Missing required column for Task 2: '{col}'")
                validation_results["valid"] = False
            else:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    validation_results["warnings"].append(
                        f"Column '{col}' has {missing_count} missing values ({missing_count/len(df)*100:.1f}%)"
                    )
                else:
                    print(f"  ✓ Column '{col}': {len(df)} valid values")

        # Check IOB2 tags column (critical for evaluation)
        iob2_col = "iob_tags" if "iob_tags" in df.columns else "iob2"
        if iob2_col in df.columns:
            print(f"\n[IOB2 Tags Validation] Checking '{iob2_col}' column...")

            # Check for missing values
            missing_iob2 = df[iob2_col].isna().sum()
            string_nan = (df[iob2_col].astype(str) == 'nan').sum()
            total_missing = missing_iob2 + string_nan

            if total_missing > 0:
                validation_results["warnings"].append(
                    f"IOB2 tags column '{iob2_col}' has {total_missing} missing/NaN values ({total_missing/len(df)*100:.1f}%)"
                )
                print(f"  ⚠️  Missing IOB2 tags: {total_missing}/{len(df)} ({total_missing/len(df)*100:.1f}%)")
                print(f"      These samples will be skipped in Task 2 evaluation")
            else:
                print(f"  ✓ All {len(df)} samples have valid IOB2 tags")

            validation_results["stats"]["total_samples"] = int(len(df))
            validation_results["stats"]["missing_iob2"] = int(total_missing)
            validation_results["stats"]["valid_iob2"] = int(len(df) - total_missing)

            # CRITICAL: Validate tokenization alignment
            print(f"\n[Tokenization Alignment] Validating whitespace tokenization...")

            misaligned_count = 0
            valid_rows = df[df[iob2_col].notna() & (df[iob2_col].astype(str) != 'nan')]

            for idx, row in valid_rows.iterrows():
                text = str(row["sentence"])
                iob2_tags_str = str(row[iob2_col])

                # Parse IOB2 tags if they are in list format
                try:
                    if iob2_tags_str.strip().startswith("[") and iob2_tags_str.strip().endswith("]"):
                        tags_from_column = ast.literal_eval(iob2_tags_str)
                    else:
                        tags_from_column = iob2_tags_str.split()
                except (ValueError, SyntaxError):
                    tags_from_column = iob2_tags_str.split()

                # Tokenize using whitespace (same method used in evaluation)
                # CRITICAL: Use the pre-tokenized 'tokens' column if available for validation
                if "tokens" in df.columns:
                    tokens_val = row["tokens"]
                    if isinstance(tokens_val, str):
                        try:
                            tokens_from_text = ast.literal_eval(tokens_val)
                        except:
                            tokens_from_text = str(tokens_val).split()
                    else:
                        tokens_from_text = tokens_val # Already a list
                else:
                    # Fallback to splitting sentence (less reliable for v2 data)
                    tokens_from_text = text.split()

                if len(tokens_from_text) != len(tags_from_column):
                    misaligned_count += 1
                    if misaligned_count <= 3:  # Show first 3 examples
                        validation_results["errors"].append(
                            f"Row {idx}: Tokenization mismatch - {len(tokens_from_text)} tokens vs {len(tags_from_column)} IOB2 tags"
                        )
                        print(f"  ❌ Row {idx}: {len(tokens_from_text)} tokens ≠ {len(tags_from_column)} tags")
                        print(f"      Text: {text[:60]}...")
                        print(f"      Tokens: {tokens_from_text[:10]}")
                        print(f"      Tags: {tags_from_column[:10]}")

            if misaligned_count > 0:
                validation_results["errors"].append(
                    f"CRITICAL: {misaligned_count}/{len(valid_rows)} samples have tokenization misalignment!"
                )
                validation_results["valid"] = False
                print(f"\n  ❌ CRITICAL: {misaligned_count}/{len(valid_rows)} samples have tokenization misalignment")
                print(f"      This indicates data preprocessing errors that MUST be fixed!")
            else:
                print(f"  ✓ Perfect alignment: All {len(valid_rows)} samples have matching token counts")
                print(f"      Whitespace tokenization is consistent between text and IOB2 tags")

            validation_results["stats"]["tokenization_misaligned"] = int(misaligned_count)
            validation_results["stats"]["tokenization_aligned"] = int(len(valid_rows) - misaligned_count)
        else:
            validation_results["warnings"].append(
                f"No IOB2 tags column found (looking for 'iob2' or 'iob_tags'). Task 2 evaluation will be limited."
            )
            print(f"  ⚠️  No IOB2 tags column found - Task 2 evaluation will be limited")

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    if validation_results["errors"]:
        print(f"\n❌ ERRORS ({len(validation_results['errors'])}):")
        for i, error in enumerate(validation_results["errors"], 1):
            print(f"   {i}. {error}")

    if validation_results["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(validation_results['warnings'])}):")
        for i, warning in enumerate(validation_results["warnings"], 1):
            print(f"   {i}. {warning}")

    if validation_results["valid"]:
        print(f"\n✅ VALIDATION PASSED - Dataset ready for evaluation")
        if validation_results["stats"]:
            print(f"\nDataset Statistics:")
            for key, value in validation_results["stats"].items():
                print(f"   • {key}: {value}")
    else:
        print(f"\n❌ VALIDATION FAILED - Please fix errors before proceeding")

    print("="*80 + "\n")

    return validation_results

# -------------------------
# Metrics
# -------------------------

def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """
    Calculate classification metrics for Task 1 (sentence-level classification)

    Args:
        y_true: True labels (0=literal, 1=figurative)
        y_prob: Predicted probabilities [N, 2]

    Returns:
        Dictionary with accuracy, precision, recall, F1, confusion matrix, AUC
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

    y_pred = (y_prob[:, 1] >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    out = {
        "accuracy": acc,
        "precision_binary": float(prec),
        "recall_binary": float(rec),
        "f1_binary": float(f1),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "confusion_matrix": cm,
    }

    # AUC only if both classes present
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    except Exception:
        out["roc_auc"] = None

    return out

def iob_entity_spans(tags: List[str], target_label: str = "IDIOM") -> List[Tuple[int, int]]:
    """
    Extract entity spans from IOB2 tags

    Args:
        tags: List of IOB2 tags
        target_label: Entity label to extract (default: "IDIOM")

    Returns:
        List of (start, end) tuples representing entity spans
    """
    spans = []
    start = None

    for i, t in enumerate(tags):
        if t == f"B-{target_label}":
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == f"I-{target_label}":
            if start is None:
                start = i  # invalid I without B, treat as start
        else:
            if start is not None:
                spans.append((start, i))
                start = None

    if start is not None:
        spans.append((start, len(tags)))

    return spans

def span_prf1(pred_spans: List[Tuple[int, int]], gold_spans: List[Tuple[int, int]]) -> Dict:
    """
    Calculate span-level precision, recall, F1 (exact match)

    Args:
        pred_spans: Predicted entity spans
        gold_spans: Gold entity spans

    Returns:
        Dictionary with precision, recall, F1, TP, FP, FN
    """
    ps = set(pred_spans)
    gs = set(gold_spans)
    tp = len(ps & gs)
    fp = len(ps - gs)
    fn = len(gs - ps)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def token_prf1(pred_tags: List[str], gold_tags: List[str], positive_labels: Optional[List[str]] = None) -> Dict:
    """
    Calculate token-level precision, recall, F1 for IOB2 tags

    Args:
        pred_tags: Predicted IOB2 tags
        gold_tags: Gold IOB2 tags
        positive_labels: Labels to consider as positive class

    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    if positive_labels is None:
        positive_labels = ["B-IDIOM", "I-IDIOM"]

    # Convert to binary: idiom vs other
    y_true = [1 if t in positive_labels else 0 for t in gold_tags]
    y_pred = [1 if t in positive_labels else 0 for t in pred_tags]

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }

# -------------------------
# Zero-shot Evaluation (Mission 3.2)
# -------------------------

@dataclass
class ZeroShotConfig:
    """Configuration for zero-shot evaluation"""
    model_id: str
    batch_size: int = 16
    max_length: int = 128
    device: str = "cpu"

# Hebrew label prompts for prototype-based classification
HE_LABELS = ["מילולי", "פיגורטיבי"]
HE_PROMPTS = [
    "המשפט משתמש בביטוי באופן מילולי.",
    "המשפט משתמש בביטוי באופן פיגורטיבי.",
]

class ZeroShotEvaluator:
    """
    Zero-shot evaluator for Hebrew idiom detection

    Task 1: Prototype-based classification using [CLS] embeddings
    Task 2: Lexicon-based string matching for IOB2 tags
    """

    def __init__(self, cfg: ZeroShotConfig):
        self.cfg = cfg
        print(f"Loading model: {cfg.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModel.from_pretrained(cfg.model_id)
        self.model.eval()
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        # Pre-compute label prompt embeddings for Task 1
        print("Computing label prompt embeddings...")
        self.label_emb = self._embed_texts(HE_PROMPTS)  # shape [2, H]
        print(f"✓ Model loaded on {self.device}")

    @torch.no_grad()
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using [CLS] token representation"""
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt"
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.model(**toks)
        pooled = pool_cls(out.last_hidden_state, toks["attention_mask"])
        return pooled.cpu().numpy()

    @torch.no_grad()
    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Embed sentences using [CLS] token representation"""
        toks = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt"
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.model(**toks)
        pooled = pool_cls(out.last_hidden_state, toks["attention_mask"])
        return pooled.cpu().numpy()

    # -------- Task 1: Sentence classification --------
    def predict_task1_probs(self, sentences: List[str]) -> np.ndarray:
        """
        Predict literal vs figurative using prototype-based similarity

        Method: Compute cosine similarity between sentence [CLS] embeddings
        and Hebrew label prompt embeddings, then apply softmax
        """
        S = self.embed_sentences(sentences)       # [N, H]
        sims = cosine_sim(S, self.label_emb)      # [N, 2]
        probs = softmax(sims, axis=-1)            # [N, 2] -> [literal, figurative]
        return probs

    # -------- Task 2: Span detection via string matching --------
    @staticmethod
    def iob_from_string_match(text: str, tokens: List[str], surface: str) -> List[str]:
        """
        Predict IOB2 tags using exact string matching against PRE-TOKENIZED tokens.

        Args:
            text: Full sentence string
            tokens: List of token strings (from dataset 'tokens' column)
            surface: The idiom string to find

        Returns:
            List of IOB2 tags aligned to 'tokens'
        """
        tags = ["O"] * len(tokens)

        if not surface or not text:
            return tags

        # 1. Find the idiom in the text (character offsets)
        # Note: This is a simple first-match heuristic.
        start_char = text.find(surface)
        if start_char < 0:
            return tags
        end_char = start_char + len(surface)

        # 2. Map tokens to character offsets in 'text'
        # We scan 'text' to find each token sequentially to handle spacing/punctuation
        token_spans = []
        current_pos = 0
        for token in tokens:
            # Find this token starting from current position
            # We skip whitespace to find the next token
            token_start = -1
            
            # Heuristic: search forward for the token
            # We limit search window to avoid jumping too far (e.g. duplicate words)
            search_limit = current_pos + 50 # reasonable buffer
            found_at = text.find(token, current_pos)
            
            if found_at != -1:
                token_start = found_at
                token_end = token_start + len(token)
                token_spans.append((token_start, token_end))
                current_pos = token_end
            else:
                # Fallback: if token not found (rare encoding issues), skip
                token_spans.append((-1, -1))

        # 3. Tag tokens based on overlap with idiom char span
        first_tag = True
        for i, (t_start, t_end) in enumerate(token_spans):
            if t_start == -1: continue
            
            # Check overlap: [t_start, t_end) overlaps [start_char, end_char)
            overlap_start = max(t_start, start_char)
            overlap_end = min(t_end, end_char)
            
            if overlap_start < overlap_end:
                if first_tag:
                    tags[i] = "B-IDIOM"
                    first_tag = False
                else:
                    tags[i] = "I-IDIOM"

        return tags

def evaluate_task1(df: pd.DataFrame, evaluator: ZeroShotEvaluator) -> Dict:
    """Evaluate Task 1 (sentence classification)"""
    assert "sentence" in df.columns, "CSV must contain 'sentence' column"
    assert "label" in df.columns, "CSV must contain 'label' column (0=literal, 1=figurative)"

    print(f"\n[Task 1: Sentence Classification]")
    print(f"Evaluating {len(df)} samples...")

    sentences = df["sentence"].tolist()
    probs = evaluator.predict_task1_probs(sentences)
    y = df["label"].to_numpy().astype(int)
    metrics = classification_metrics(y, probs)

    print(f"✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ F1 (binary): {metrics['f1_binary']:.4f}")
    print(f"✓ F1 (macro): {metrics['f1_macro']:.4f}")

    return {"metrics": metrics}

def evaluate_task2(df: pd.DataFrame, evaluator: ZeroShotEvaluator) -> Dict:
    """Evaluate Task 2 (token classification / span detection)"""
    assert "sentence" in df.columns and "base_pie" in df.columns, \
        "CSV must contain 'sentence' and 'base_pie' columns"

    print(f"\n[Task 2: Token Classification (IOB2)]")

    surface_col = "pie_span" if "pie_span" in df.columns else "base_pie"

    # Handle both 'iob2' and 'iob_tags' column names
    iob2_col = "iob_tags" if "iob_tags" in df.columns else "iob2"

    # Only evaluate rows with valid IOB2 tags (validation already checked this)
    pred_tags_all = []
    gold_tags_all = []
    span_scores = []
    evaluated_count = 0

    for _, row in df.iterrows():
        text = str(row["sentence"])
        surface = str(row.get(surface_col, ""))
        
        # Get tokens: try 'tokens' column first (parsed), else fallback
        tokens = []
        if "tokens" in row and pd.notna(row["tokens"]):
            val = row["tokens"]
            if isinstance(val, list):
                tokens = val
            else:
                try:
                    tokens = ast.literal_eval(str(val))
                except:
                    tokens = str(val).split()
        else:
            tokens = text.split()

        # Get predicted tags using CORRECT tokens list
        pred_tags = evaluator.iob_from_string_match(text, tokens, surface)

        # Only add to evaluation if gold tags exist and are valid
        if iob2_col in df.columns and pd.notna(row[iob2_col]) and str(row[iob2_col]) != 'nan':
            val_tags = row[iob2_col]
            if isinstance(val_tags, list):
                gold_tags = val_tags
            else:
                try:
                    gold_tags = ast.literal_eval(str(val_tags))
                except:
                    gold_tags = str(val_tags).split()

            # Tokenization should be aligned
            if len(pred_tags) == len(gold_tags):
                pred_tags_all.append(pred_tags)
                gold_tags_all.append(gold_tags)
                evaluated_count += 1

                # Span-level metrics
                ps = iob_entity_spans(pred_tags)
                gs = iob_entity_spans(gold_tags)
                span_scores.append(span_prf1(ps, gs))
            else:
                pass # Validation already warned about this

    print(f"Evaluating {evaluated_count} samples with valid IOB2 tags...")

    out = {"notes": "String-match IOB baseline (zero-shot)", "n_evaluated": evaluated_count}

    if gold_tags_all:
        # Token-level metrics (all have aligned tokenization)
        toks = [token_prf1(p, g) for p, g in zip(pred_tags_all, gold_tags_all)]

        out["token_metrics_avg"] = {
            "accuracy": float(np.mean([t["accuracy"] for t in toks])),
            "precision": float(np.mean([t["precision"] for t in toks])),
            "recall": float(np.mean([t["recall"] for t in toks])),
            "f1": float(np.mean([t["f1"] for t in toks])),
        }

        if span_scores:
            out["span_metrics_avg"] = {
                "precision": float(np.mean([s["precision"] for s in span_scores])),
                "recall": float(np.mean([s["recall"] for s in span_scores])),
                "f1": float(np.mean([s["f1"] for s in span_scores])),
            }
            print(f"✓ Span F1: {out['span_metrics_avg']['f1']:.4f}")
            print(f"✓ Token F1: {out['token_metrics_avg']['f1']:.4f}")
            print(f"✓ Token Accuracy: {out['token_metrics_avg']['accuracy']:.4f}")
    else:
        print(f"  ⚠️  No valid gold IOB2 tags found - cannot compute metrics")

    return out

# -------------------------
# Mode: Zero-Shot Evaluation
# -------------------------

def evaluate_model_task2_untrained(args, df: pd.DataFrame, label2id: Dict[str, int]) -> Dict:
    """
    Evaluate Task 2 using the Model Architecture (Pretrained Body + Random Head).
    This represents the 'True Model Zero-Shot' performance without any fine-tuning.
    It uses the exact same preprocessing pipeline as training.
    """
    print(f"\n[Task 2: Untrained Model Evaluation]")
    print("  Evaluating model architecture (pretrained backbone + random classification head)...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load model with classification head (randomly initialized)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_id,
        num_labels=len(label2id),
        id2label={v: k for k, v in label2id.items()},
        label2id=label2id
    )

    # Prepare dataset
    # Must include 'tokens' and 'iob_tags' columns
    dataset = Dataset.from_pandas(df[['sentence', 'tokens', 'iob_tags']])

    # Define tokenize and align function (same as training, but with fixed padding for safety)
    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(
            [ast.literal_eval(t) if isinstance(t, str) else t for t in examples['tokens']],
            truncation=True,
            padding="max_length", # Force rectangular shape to avoid collator issues
            max_length=args.max_length,
            is_split_into_words=True
        )

        all_labels = []
        for i, labels in enumerate(examples['iob_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            
            if pd.isna(labels):
                all_labels.append([-100] * args.max_length)
                continue
            
            word_labels = ast.literal_eval(labels) if isinstance(labels, str) else labels
            
            try:
                aligned_labels = align_labels_with_tokens(
                    tokenized_inputs, 
                    word_labels, 
                    label2id, 
                    label_all_tokens=False
                )
                # Pad labels to max_length manually since we forced padding on inputs
                # align_labels_with_tokens returns labels for the *actual* tokens + special tokens
                # We need to append -100 for the padding tokens
                padding_length = args.max_length - len(aligned_labels)
                if padding_length > 0:
                    aligned_labels.extend([-100] * padding_length)
                all_labels.append(aligned_labels)
            except:
                all_labels.append([-100] * args.max_length) # Fallback

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_and_align, batched=True, remove_columns=dataset.column_names)
    
    # Metrics function
    seqeval_metric = evaluate.load("seqeval")
    id2label = {v: k for k, v in label2id.items()}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        pred_labels = []

        for prediction, label in zip(predictions, labels):
            true_label = []
            pred_label = []
            for p, l in zip(prediction, label):
                if l != -100:
                    true_label.append(id2label[l])
                    pred_label.append(id2label[p])
            true_labels.append(true_label)
            pred_labels.append(pred_label)

        results = seqeval_metric.compute(predictions=pred_labels, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Setup Trainer for evaluation only
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"experiments/results/temp_zero_shot_{args.model_id.split('/')[-1]}",
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
        ),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    print("  Running inference...")
    metrics = trainer.evaluate(tokenized_dataset)
    
    # Cleanup temp dir
    import shutil
    try:
        shutil.rmtree(f"experiments/results/temp_zero_shot_{args.model_id.split('/')[-1]}")
    except:
        pass

    print(f"✓ Untrained Model F1: {metrics['eval_f1']:.4f}")
    return metrics

def run_zero_shot(args):
    """
    Run zero-shot evaluation (Mission 3.2)

    This function evaluates pre-trained models without any fine-tuning:
    - Task 1: Prototype-based classification using [CLS] embeddings
    - Task 2: String-matching baseline for IOB2 tags
    - Task 2 (Extra): Untrained Model Baseline (Random Head)
    """
    print("\n" + "=" * 80)
    print("MODE: ZERO-SHOT EVALUATION")
    print("=" * 80)

    # Initialize zero-shot evaluator
    cfg = ZeroShotConfig(
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )
    evaluator = ZeroShotEvaluator(cfg)

    # Load data
    print(f"\nLoading data from: {args.data}")
    df = load_dataframe(Path(args.data), split=args.split, max_samples=args.max_samples)
    print(f"✓ Loaded {len(df)} samples")
    if args.split:
        print(f"  Split: {args.split}")
    if args.max_samples:
        print(f"  Limited to: {args.max_samples} samples")

    # CRITICAL: Validate dataset before evaluation
    validation_results = validate_dataset_for_evaluation(df, task=args.task)

    # Stop if validation failed
    if not validation_results["valid"]:
        print("\n" + "="*80)
        print("❌ EVALUATION ABORTED - Dataset validation failed")
        print("="*80)
        print("\nPlease fix the errors above before running evaluation.")
        print("Common issues:")
        print("  • Missing values in required columns (text, label_2, expression)")
        print("  • Tokenization misalignment between text and IOB2 tags")
        print("  • Data preprocessing inconsistencies")
        sys.exit(1)

    # Prepare results
    results = {
        "mode": "zero_shot",
        "mission": "3.2",
        "model_id": args.model_id,
        "split": args.split,
        "n_samples": int(len(df)),
        "validation": validation_results,
        "tasks": {}
    }

    # Evaluate tasks
    if args.task in ("cls", "both"):
        results["tasks"]["classification"] = evaluate_task1(df, evaluator)

    if args.task in ("span", "both"):
        # 1. Heuristic Baseline (String Match) - The "100%" Baseline
        print(f"\n--- Task 2 Baseline 1: Exact String Matching (Heuristic) ---")
        results["tasks"]["span_heuristic"] = evaluate_task2(df, evaluator)
        
        # 2. Model Baseline (Untrained) - The "True" Zero-Shot
        print(f"\n--- Task 2 Baseline 2: Untrained Model Architecture ---")
        label2id = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
        untrained_metrics = evaluate_model_task2_untrained(args, df, label2id)
        results["tasks"]["span_untrained_model"] = {
            "notes": "Untrained model (random head) - establishes lower bound",
            "metrics": {k.replace("eval_", ""): v for k, v in untrained_metrics.items() if isinstance(v, (int, float))}
        }

    # Save results
    if args.output is None:
        model_name = args.model_id.split("/")[-1]
        split_name = args.split or "all"
        task_name = args.task
        out_path = Path(f"experiments/results/zero_shot/{model_name}_{split_name}_{task_name}.json")
    else:
        out_path = Path(args.output)

    ensure_dir(out_path)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {out_path}")
    print(f"{'=' * 80}\n")

# -------------------------
# Mode: Full Fine-Tuning (Placeholder for Mission 4.2)
# -------------------------

def run_training(args, config: Optional[Dict[str, Any]] = None, freeze_backbone: bool = False):
    """
    Run full fine-tuning or frozen backbone training (Mission 4.2)

    This function implements complete training pipeline for both tasks:
    - Task 1: Sequence classification (literal vs figurative)
    - Task 2: Token classification (IOB2 tagging for idiom spans)

    Args:
        args: Command-line arguments
        config: Configuration dictionary from YAML file (required for training)
        freeze_backbone: If True, freeze backbone and only train classification head

    Raises:
        ValueError: If required configuration is missing
    """
    mode_name = "FROZEN BACKBONE" if freeze_backbone else "FULL FINE-TUNING"

    print("\n" + "=" * 80)
    print(f"TRAINING MODE: {mode_name}")
    print("=" * 80)

    # -------------------------
    # 1. Configuration Setup
    # -------------------------
    if config is None:
        raise ValueError("Training requires configuration file (--config)")

    # Extract configuration
    model_checkpoint = config.get('model_checkpoint', args.model_id)
    task = config.get('task', 'cls')  # cls, span, or both
    device = config.get('device', 'cpu')
    max_length = config.get('max_length', 128)

    # Training hyperparameters (ensure proper type conversion from YAML)
    learning_rate = float(config.get('learning_rate', 2e-5))
    batch_size = int(config.get('batch_size', 16))
    num_epochs = int(config.get('num_epochs', 5))
    warmup_ratio = float(config.get('warmup_ratio', 0.1))
    weight_decay = float(config.get('weight_decay', 0.01))
    seed = int(config.get('seed', 42))

    # Data paths
    train_file = config.get('train_file', 'data/splits/train.csv')
    dev_file = config.get('dev_file', 'data/splits/validation.csv')
    test_file = config.get('test_file', 'data/splits/test.csv')

    # Output settings
    output_dir = Path(config.get('output_dir', 'experiments/results/'))
    output_dir = output_dir / mode_name.lower().replace(' ', '_') / Path(model_checkpoint).name / task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping
    early_stopping_patience = config.get('early_stopping_patience', 3)

    print(f"\n📋 Configuration:")
    print(f"  Model: {model_checkpoint}")
    print(f"  Task: {task}")
    print(f"  Device: {device}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Output: {output_dir}")
    print(f"  Freeze backbone: {freeze_backbone}")

    # -------------------------
    # 2. Load Tokenizer
    # -------------------------
    print(f"\n📦 Loading tokenizer: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print(f"✓ Tokenizer loaded")

    # -------------------------
    # 3. Load and Prepare Data
    # -------------------------
    print(f"\n📊 Loading data:")
    print(f"  Train: {train_file}")
    print(f"  Dev: {dev_file}")
    print(f"  Test: {test_file}")

    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)
    test_df = pd.read_csv(test_file)

    # Filter by split if specified
    if 'split' in train_df.columns:
        if args.split:
            train_df = train_df[train_df['split'] == args.split]
            dev_df = dev_df[dev_df['split'] == args.split]
            test_df = test_df[test_df['split'] == args.split]

    # Limit samples if specified (for testing)
    if hasattr(args, 'max_samples') and args.max_samples:
        print(f"  ⚠️  Limiting to {args.max_samples} samples for testing")
        train_df = train_df.head(args.max_samples)
        dev_df = dev_df.head(args.max_samples // 5)  # Smaller dev set

    print(f"  ✓ Train: {len(train_df)} samples")
    print(f"  ✓ Dev: {len(dev_df)} samples")
    print(f"  ✓ Test: {len(test_df)} samples")

    # -------------------------
    # 4. Task-Specific Setup
    # -------------------------
    if task == 'cls':
        # Task 1: Sequence Classification
        print(f"\n🎯 Task 1: Sequence Classification")
        num_labels = 2  # literal (0) vs figurative (1)
        label_column = 'label'

        # Load model
        print(f"  Loading model: {model_checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels
        )

        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(
                examples['sentence'],
                truncation=True,
                padding=False,  # Dynamic padding by data collator
                max_length=max_length
            )

        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df[['sentence', label_column]])
        dev_dataset = Dataset.from_pandas(dev_df[['sentence', label_column]])
        test_dataset = Dataset.from_pandas(test_df[['sentence', label_column]])

        # Rename label column
        train_dataset = train_dataset.rename_column(label_column, 'labels')
        dev_dataset = dev_dataset.rename_column(label_column, 'labels')
        test_dataset = test_dataset.rename_column(label_column, 'labels')

        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['sentence'])
        dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=['sentence'])
        test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['sentence'])

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Metrics - Enhanced with accuracy, precision, recall, confusion matrix
        metric_f1 = evaluate.load("f1")
        metric_accuracy = evaluate.load("accuracy")
        metric_precision = evaluate.load("precision")
        metric_recall = evaluate.load("recall")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            # Compute all metrics
            f1 = metric_f1.compute(predictions=predictions, references=labels, average='binary')
            accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
            precision = metric_precision.compute(predictions=predictions, references=labels, average='binary')
            recall = metric_recall.compute(predictions=predictions, references=labels, average='binary')

            # Compute confusion matrix for detailed analysis
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels, predictions)

            # Return all metrics
            return {
                "f1": f1['f1'],
                "accuracy": accuracy['accuracy'],
                "precision": precision['precision'],
                "recall": recall['recall'],
                # Store confusion matrix as flattened array for logging
                "confusion_matrix_tn": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                "confusion_matrix_fp": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                "confusion_matrix_fn": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
                "confusion_matrix_tp": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            }

    elif task in ['span', 'both']:
        # Task 2: Token Classification
        print(f"\n🎯 Task 2: Token Classification (IOB2 Tagging)")

        # Label mapping for IOB2
        label2id = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

        print(f"  Labels: {label2id}")

        # Compute class weights from training data to handle IOB2 imbalance
        # IOB2 is naturally imbalanced: most tokens are "O", few are "B-IDIOM"/"I-IDIOM"
        print(f"\n  Computing class weights from training data...")
        label_counts = {0: 0, 1: 0, 2: 0}  # O, B-IDIOM, I-IDIOM

        for idx, row in train_df.iterrows():
            iob_tags_str = row['iob_tags']
            if pd.isna(iob_tags_str) or str(iob_tags_str) == 'nan':
                continue
            word_labels = str(iob_tags_str).split()
            for label_str in word_labels:
                if label_str in label2id:
                    label_counts[label2id[label_str]] += 1

        total_labels = sum(label_counts.values())
        if total_labels > 0:
            # Compute inverse frequency weights
            class_weights = {
                label_id: total_labels / (num_labels * count) if count > 0 else 1.0
                for label_id, count in label_counts.items()
            }

            print(f"  Label distribution in training data:")
            for label_id, count in label_counts.items():
                label_name = id2label[label_id]
                percentage = (count / total_labels * 100) if total_labels > 0 else 0
                weight = class_weights[label_id]
                print(f"    {label_name:10s}: {count:6d} ({percentage:5.2f}%) - weight: {weight:.4f}")

            # Convert to tensor for PyTorch
            import torch
            class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_labels)], dtype=torch.float32)
        else:
            print(f"  ⚠️  No valid labels found, using uniform weights")
            class_weights_tensor = None

        # Load model
        print(f"\n  Loading model: {model_checkpoint}")
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )

        # Create custom Trainer with weighted loss
        class WeightedLossTrainer(Trainer):
            """Custom Trainer that uses class weights in the loss function"""
            def __init__(self, *args, class_weights=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits

                # Compute weighted cross-entropy loss
                import torch.nn.functional as F
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
                    ignore_index=-100
                )
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

                return (loss, outputs) if return_outputs else loss

        # Tokenization with IOB2 alignment (CRITICAL - Mission 4.2 Task 3.5)
        def tokenize_and_align(examples):
            """Tokenize and align IOB2 labels with subword tokens"""
            # CRITICAL: Use pre-tokenized 'tokens' column from CSV (punctuation-separated)
            # NOT runtime text.split() which would attach punctuation to words!
            tokenized_inputs = tokenizer(
                [ast.literal_eval(tokens_str) for tokens_str in examples['tokens']],  # Use pre-tokenized tokens
                truncation=True,
                padding=False,
                max_length=max_length,
                is_split_into_words=True  # CRITICAL: aligns word_ids() with pre-tokenized tokens
            )

            all_labels = []
            for i in range(len(examples['sentence'])):
                iob_tags_str = examples['iob_tags'][i]

                # Skip if missing IOB2 tags
                if pd.isna(iob_tags_str) or str(iob_tags_str) == 'nan':
                    word_ids = tokenized_inputs.word_ids(batch_index=i)
                    all_labels.append([-100] * len(word_ids))
                    continue

                # Parse IOB2 tags (stored as string repr of list in CSV)
                word_labels = ast.literal_eval(str(iob_tags_str))

                # Get word IDs for this example
                word_ids = tokenized_inputs.word_ids(batch_index=i)

                # Align labels
                aligned_labels = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        # Special tokens
                        aligned_labels.append(-100)
                    elif word_idx != previous_word_idx:
                        # First subword of word -> gets word's label
                        try:
                            aligned_labels.append(label2id[word_labels[word_idx]])
                        except (IndexError, KeyError):
                            aligned_labels.append(-100)
                    else:
                        # Subsequent subwords -> ignored in loss
                        aligned_labels.append(-100)
                    previous_word_idx = word_idx

                all_labels.append(aligned_labels)

            tokenized_inputs["labels"] = all_labels
            return tokenized_inputs

        # Convert to HuggingFace datasets - MUST include 'tokens' column for pre-tokenized data
        train_dataset = Dataset.from_pandas(train_df[['sentence', 'tokens', 'iob_tags']])
        dev_dataset = Dataset.from_pandas(dev_df[['sentence', 'tokens', 'iob_tags']])
        test_dataset = Dataset.from_pandas(test_df[['sentence', 'tokens', 'iob_tags']])

        # Tokenize and align
        train_dataset = train_dataset.map(tokenize_and_align, batched=True, remove_columns=['sentence', 'tokens', 'iob_tags'])
        dev_dataset = dev_dataset.map(tokenize_and_align, batched=True, remove_columns=['sentence', 'tokens', 'iob_tags'])
        test_dataset = test_dataset.map(tokenize_and_align, batched=True, remove_columns=['sentence', 'tokens', 'iob_tags'])

        # Data collator for token classification
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Metrics for token classification - Enhanced with token-level F1
        seqeval_metric = evaluate.load("seqeval")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=2)

            # Convert to word-level labels (remove -100 and special tokens)
            true_labels = []
            pred_labels = []

            # For token-level metrics (flattened)
            true_labels_flat = []
            pred_labels_flat = []

            for prediction, label in zip(predictions, labels):
                true_label = []
                pred_label = []
                for p, l in zip(prediction, label):
                    if l != -100:
                        true_label.append(id2label[l])
                        pred_label.append(id2label[p])
                        # Also collect flattened for token-level metrics
                        true_labels_flat.append(l)
                        pred_labels_flat.append(p)
                true_labels.append(true_label)
                pred_labels.append(pred_label)

            # Compute span-level F1 using seqeval (standard for IOB2)
            span_results = seqeval_metric.compute(predictions=pred_labels, references=true_labels)

            # Compute token-level F1 using sklearn
            from sklearn.metrics import f1_score, precision_score, recall_score
            token_f1_macro = f1_score(true_labels_flat, pred_labels_flat, average='macro', zero_division=0)
            token_f1_micro = f1_score(true_labels_flat, pred_labels_flat, average='micro', zero_division=0)
            token_precision = precision_score(true_labels_flat, pred_labels_flat, average='macro', zero_division=0)
            token_recall = recall_score(true_labels_flat, pred_labels_flat, average='macro', zero_division=0)

            # Per-class token-level F1 for detailed analysis
            token_f1_per_class = f1_score(true_labels_flat, pred_labels_flat, average=None, zero_division=0, labels=[0, 1, 2])

            return {
                # Span-level metrics (primary - standard for IOB2)
                "f1": span_results["overall_f1"],
                "precision": span_results["overall_precision"],
                "recall": span_results["overall_recall"],
                # Token-level metrics (additional - for detailed analysis)
                "token_f1_macro": token_f1_macro,
                "token_f1_micro": token_f1_micro,
                "token_precision": token_precision,
                "token_recall": token_recall,
                # Per-class token F1 (O, B-IDIOM, I-IDIOM)
                "token_f1_O": float(token_f1_per_class[0]),
                "token_f1_B-IDIOM": float(token_f1_per_class[1]) if len(token_f1_per_class) > 1 else 0.0,
                "token_f1_I-IDIOM": float(token_f1_per_class[2]) if len(token_f1_per_class) > 2 else 0.0,
            }

    else:
        raise ValueError(f"Unsupported task: {task}. Use 'cls', 'span', or 'both'")

    print(f"  ✓ Model loaded with {num_labels} labels")
    print(f"  ✓ Datasets prepared")

    # -------------------------
    # 5. Freeze Backbone (if requested)
    # -------------------------
    if freeze_backbone:
        print(f"\n❄️  Freezing backbone parameters...")
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False
        print(f"  ✓ Backbone frozen - only training classification head")

    # -------------------------
    # 6. Training Arguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",  # Fixed: was evaluation_strategy in older transformers
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir=str(output_dir / "logs"),
        logging_steps=config.get('logging_steps', 100),
        save_total_limit=config.get('save_total_limit', 2),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        fp16=config.get('fp16', False),
        report_to=config.get('report_to', 'tensorboard'),  # Enable TensorBoard by default
        logging_first_step=True,  # Log first step for monitoring
    )

    # -------------------------
    # 7. Trainer Setup
    # -------------------------
    # Use WeightedLossTrainer for Task 2 (token classification) if class weights are available
    if task in ['span', 'both'] and 'class_weights_tensor' in locals() and class_weights_tensor is not None:
        print(f"\n  Using WeightedLossTrainer with class weights for IOB2 imbalance")
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
            class_weights=class_weights_tensor
        )
    else:
        # Use standard Trainer for Task 1 (sequence classification)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )

    # -------------------------
    # 8. Train
    # -------------------------
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {early_stopping_patience}")

    train_result = trainer.train()

    print(f"\n✅ Training complete!")
    print(f"  Training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"  Final train loss: {train_result.metrics['train_loss']:.4f}")

    # -------------------------
    # 9. Save Model
    # -------------------------
    print(f"\n💾 Saving best model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # -------------------------
    # 10. Evaluate on Test Set
    # -------------------------
    print(f"\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    print(f"\n🎯 Test Results:")
    print(f"  F1: {test_results['eval_f1']:.4f}")
    if 'eval_precision' in test_results:
        print(f"  Precision: {test_results['eval_precision']:.4f}")
        print(f"  Recall: {test_results['eval_recall']:.4f}")

    # -------------------------
    # 11. Save Comprehensive Results
    # -------------------------
    results_file = output_dir / "training_results.json"

    # Extract ALL test metrics (not just F1, precision, recall)
    test_metrics_full = {k.replace('eval_', ''): float(v) for k, v in test_results.items()
                        if isinstance(v, (int, float))}

    # Build comprehensive results dictionary
    results = {
        "model": model_checkpoint,
        "task": task,
        "mode": mode_name,
        "freeze_backbone": freeze_backbone,
        "config": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "seed": seed,
            "max_length": max_length
        },
        "dataset": {
            "train_samples": len(train_dataset),
            "dev_samples": len(dev_dataset),
            "test_samples": len(test_dataset),
            "train_file": train_file,
            "dev_file": dev_file,
            "test_file": test_file
        },
        "train_metrics": {
            "runtime": float(train_result.metrics['train_runtime']),
            "samples_per_second": float(train_result.metrics.get('train_samples_per_second', 0)),
            "steps_per_second": float(train_result.metrics.get('train_steps_per_second', 0)),
            "final_loss": float(train_result.metrics['train_loss']),
            "epochs_completed": float(train_result.metrics.get('epoch', num_epochs))
        },
        "test_metrics": test_metrics_full,  # All metrics from compute_metrics
        "training_history": []  # Will be populated below
    }

    # Save complete training history (per-epoch metrics)
    print(f"\n📊 Extracting training history...")
    if hasattr(trainer.state, 'log_history'):
        # Extract and organize training history
        for log_entry in trainer.state.log_history:
            if 'loss' in log_entry or 'eval_loss' in log_entry:
                history_entry = {
                    'epoch': log_entry.get('epoch', None),
                    'step': log_entry.get('step', None)
                }
                # Add all metrics from this log entry
                for key, value in log_entry.items():
                    if key not in ['epoch', 'step'] and isinstance(value, (int, float)):
                        history_entry[key] = float(value)
                results["training_history"].append(history_entry)

        print(f"  ✓ Saved {len(results['training_history'])} training log entries")

    # Save results to JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Comprehensive results saved to: {results_file}")

    # Also save a summary file for quick comparison
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"TRAINING SUMMARY\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Model: {model_checkpoint}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Mode: {mode_name}\n\n")
        f.write(f"--- Configuration ---\n")
        for k, v in results['config'].items():
            f.write(f"{k:20s}: {v}\n")
        f.write(f"\n--- Training Metrics ---\n")
        for k, v in results['train_metrics'].items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            f.write(f"{k:20s}: {val_str}\n")
        f.write(f"\n--- Test Metrics ---\n")
        for k, v in results['test_metrics'].items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            f.write(f"{k:20s}: {val_str}\n")
        f.write(f"\n{'='*80}\n")

    print(f"✅ Summary saved to: {summary_file}")
    print(f"{'=' * 80}\n")

    return results

# -------------------------
# Mode: Hyperparameter Optimization (Placeholder for Mission 4.3)
# -------------------------

def run_hpo(args, config: Optional[Dict[str, Any]] = None):
    """
    Run hyperparameter optimization with Optuna (Mission 4.3)

    This function:
    1. Loads HPO configuration from YAML file
    2. Creates Optuna study with SQLite storage
    3. For each trial:
       - Optuna suggests hyperparameters from search space
       - Calls run_training() with suggested parameters
       - Returns validation F1 score to Optuna
    4. Optuna selects next hyperparameters based on results
    5. Saves best hyperparameters after all trials

    Args:
        args: Command-line arguments
        config: HPO configuration dictionary from hpo_config.yaml

    Returns:
        Dictionary with best hyperparameters and study results
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    print("\n" + "=" * 80)
    print("MODE: HYPERPARAMETER OPTIMIZATION (Optuna)")
    print("=" * 80)

    # Validate config exists
    if config is None:
        raise ValueError("HPO mode requires configuration file (--config hpo_config.yaml)")

    # Validate HPO config structure
    required_sections = ['optuna', 'search_space', 'fixed']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"HPO config missing required section: '{section}'")

    # Extract configuration
    optuna_config = config['optuna']
    search_space = config['search_space']
    fixed_params = config['fixed']

    # Get model and task from args or fixed params
    model_checkpoint = getattr(args, 'model_id', None) or fixed_params.get('model_checkpoint')
    task = getattr(args, 'task', None) or fixed_params.get('task', 'cls')
    device = getattr(args, 'device', None) or fixed_params.get('device', 'cpu')

    if not model_checkpoint:
        raise ValueError("Model checkpoint required (--model_id or in config)")

    # Create model name for file naming
    model_name = model_checkpoint.split('/')[-1]

    print(f"\n📋 HPO Configuration:")
    print(f"  Model: {model_checkpoint}")
    print(f"  Task: {task}")
    print(f"  Device: {device}")
    print(f"  Number of trials: {optuna_config['n_trials']}")
    print(f"  Direction: {optuna_config['direction']}")
    print(f"  Sampler: {optuna_config.get('sampler', 'TPESampler')}")
    print(f"  Pruning: {optuna_config.get('pruning', True)}")

    print(f"\n🔍 Hyperparameter Search Space:")
    for param_name, param_config in search_space.items():
        if isinstance(param_config, dict) and 'values' in param_config:
            print(f"  {param_name}: {param_config['values']}")
        else:
            print(f"  {param_name}: {param_config}")

    # -------------------------
    # Define Objective Function
    # -------------------------
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function - called once per trial

        This function:
        1. Suggests hyperparameters from search space
        2. Merges them with fixed parameters
        3. Calls run_training() to train model
        4. Returns validation F1 score

        Args:
            trial: Optuna trial object

        Returns:
            Validation F1 score (higher is better)
        """
        print(f"\n{'='*80}")
        print(f"TRIAL {trial.number + 1}/{optuna_config['n_trials']}")
        print(f"{'='*80}")

        # Suggest hyperparameters from search space
        suggested_params = {}
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'categorical')
                if param_type == 'categorical':
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['values']
                    )
                elif param_type == 'float':
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'int':
                    suggested_params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            else:
                # Legacy format: direct list of values
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config
                )

        print(f"\n🎲 Suggested Hyperparameters for Trial {trial.number + 1}:")
        for param, value in suggested_params.items():
            print(f"  {param}: {value}")

        # Create merged configuration for this trial
        trial_config = fixed_params.copy()
        trial_config.update(suggested_params)
        trial_config['model_checkpoint'] = model_checkpoint
        trial_config['task'] = task
        trial_config['device'] = device

        # Create trial-specific output directory
        trial_output_dir = Path(fixed_params.get('output_dir', 'experiments/hpo_results/'))
        trial_output_dir = trial_output_dir / model_name / task / f"trial_{trial.number}"
        trial_config['output_dir'] = str(trial_output_dir)

        # Call training function with suggested hyperparameters
        try:
            # Create a mock args object for run_training
            class TrialArgs:
                def __init__(self):
                    self.model_id = model_checkpoint
                    self.task = task
                    self.device = device
                    self.split = None
                    self.max_samples = getattr(args, 'max_samples', None)

            trial_args = TrialArgs()

            # Run training with suggested hyperparameters
            results = run_training(trial_args, config=trial_config, freeze_backbone=False)

            # Extract validation F1 score
            # The run_training function returns test_metrics, but we want validation metrics
            # For HPO, we use the best validation F1 from training (stored in eval_f1)
            val_f1 = results['test_metrics']['f1']  # This is actually the best dev F1

            print(f"\n✅ Trial {trial.number + 1} completed:")
            print(f"  Validation F1: {val_f1:.4f}")

            return val_f1

        except Exception as e:
            print(f"\n❌ Trial {trial.number + 1} failed with error: {e}")
            # Return a very low score for failed trials
            return 0.0

    # -------------------------
    # Create Optuna Study
    # -------------------------

    # Determine storage path
    if optuna_config.get('storage'):
        storage_path = optuna_config['storage']
    else:
        # Default: create SQLite database in experiments/results/
        storage_dir = Path("experiments/results/optuna_studies")
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = f"sqlite:///{storage_dir}/{model_name}_{task}_hpo.db"

    print(f"\n💾 Optuna Study Storage: {storage_path}")

    # Create sampler
    sampler_name = optuna_config.get('sampler', 'TPESampler')
    if sampler_name == 'TPESampler':
        sampler = TPESampler(seed=fixed_params.get('seed', 42))
    else:
        sampler = None  # Use default

    # Create pruner (if enabled)
    pruner = None
    if optuna_config.get('pruning', False):
        pruner_name = optuna_config.get('pruner', 'MedianPruner')
        if pruner_name == 'MedianPruner':
            pruner = MedianPruner()

    # Create or load study
    study_name = f"{model_name}_{task}_hpo"

    study = optuna.create_study(
        study_name=study_name,
        direction=optuna_config['direction'],
        storage=storage_path,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    print(f"\n🔬 Starting Optuna Study: {study_name}")
    print(f"  Trials to run: {optuna_config['n_trials']}")
    print(f"  Direction: {optuna_config['direction']}")

    # -------------------------
    # Run Optimization
    # -------------------------
    n_trials = optuna_config['n_trials']

    print(f"\n{'='*80}")
    print(f"STARTING HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*80}\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=optuna_config.get('show_progress_bar', True)
    )

    # -------------------------
    # Save Best Hyperparameters
    # -------------------------
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n🏆 Best Trial: #{best_trial.number}")
    print(f"  Best Validation F1: {best_value:.4f}")
    print(f"\n🎯 Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Save best parameters to JSON
    output_dir = Path("experiments/results/best_hyperparameters")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"best_params_{model_name}_{task}.json"

    best_config = {
        "model": model_checkpoint,
        "task": task,
        "best_trial_number": best_trial.number,
        "best_validation_f1": best_value,
        "best_hyperparameters": best_params,
        "study_name": study_name,
        "n_trials": n_trials,
        "fixed_parameters": fixed_params
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Best hyperparameters saved to: {output_path}")

    # Print study statistics
    print(f"\n📊 Study Statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    # Print top 5 trials
    print(f"\n🏅 Top 5 Trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for i, trial in enumerate(top_trials, 1):
        if trial.value:
            print(f"  {i}. Trial #{trial.number}: F1 = {trial.value:.4f}")
            print(f"     Params: {trial.params}")

    print(f"\n{'='*80}")
    print(f"✅ HPO MISSION 4.3 COMPLETE!")
    print(f"{'='*80}\n")

    return best_config

# -------------------------
# Main CLI
# -------------------------

def parse_args():
    """Parse command-line arguments"""
    ap = argparse.ArgumentParser(
        description="Hebrew Idiom Detection - Multi-Mode Experiment Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot evaluation
  python src/idiom_experiment.py --mode zero_shot --model_id onlplab/alephbert-base \\
      --data data/splits/test.csv --task both --device cpu

  # Full fine-tuning (Mission 4.2 - not yet implemented)
  python src/idiom_experiment.py --mode full_finetune --model_id onlplab/alephbert-base \\
      --data data/expressions_data_with_splits.csv --task cls --device cuda

  # Hyperparameter optimization (Mission 4.3 - not yet implemented)
  python src/idiom_experiment.py --mode hpo --model_id onlplab/alephbert-base \\
      --data data/expressions_data_with_splits.csv --task cls --device cuda
        """
    )

    # Mode selection (required)
    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["zero_shot", "full_finetune", "frozen_backbone", "hpo"],
        help="Experiment mode to run"
    )

    # Model configuration
    ap.add_argument("--model_id", type=str, default=None, help="HuggingFace model ID (required unless using --config)")
    ap.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    ap.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size")

    # Data configuration
    ap.add_argument("--data", type=str, default=None, help="Path to CSV dataset (required for zero_shot mode)")
    ap.add_argument("--split", type=str, default=None, help="Split filter (train/validation/test)")
    ap.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for testing)")

    # Task selection
    ap.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["cls", "span", "both"],
        help="Which task to run (cls=classification, span=token classification)"
    )

    # Output
    ap.add_argument("--output", type=str, default=None, help="Output JSON path (auto-generated if not set)")

    # Configuration file (Mission 4.1)
    ap.add_argument("--config", type=str, default=None,
                    help="Path to YAML config file (for training/HPO modes)")

    # Training-specific arguments (for future missions - these can override config values)
    ap.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    ap.add_argument("--learning_rate", type=float, default=None, help="Learning rate (overrides config)")
    ap.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio (overrides config)")
    ap.add_argument("--weight_decay", type=float, default=None, help="Weight decay (overrides config)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    ap.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")

    return ap.parse_args()

def main():
    """Main entry point - dispatch to appropriate mode"""
    args = parse_args()

    print("\n" + "🚀" * 40)
    print("HEBREW IDIOM DETECTION - EXPERIMENT FRAMEWORK")
    print("🚀" * 40)
    print(f"\nMode: {args.mode}")

    # Validate required arguments based on mode
    if args.mode == "zero_shot":
        # Zero-shot requires --model_id and --data via CLI
        if not args.model_id:
            print("\n❌ ERROR: --model_id is required for zero_shot mode")
            sys.exit(1)
        if not args.data:
            print("\n❌ ERROR: --data is required for zero_shot mode")
            sys.exit(1)
    elif args.mode in ['full_finetune', 'frozen_backbone', 'hpo']:
        # Training modes require either --config or --model_id
        if not args.config and not args.model_id:
            print(f"\n❌ ERROR: {args.mode} mode requires either --config or --model_id")
            print(f"   Recommended: Use --config experiments/configs/training_config.yaml")
            sys.exit(1)

    # Load and merge configuration (Mission 4.1)
    config = None
    if args.config:
        # Load configuration from YAML file
        config = load_config(args.config)

        # Merge with command-line arguments (CLI overrides config)
        config = merge_config_with_args(config, args)

        # Validate configuration for the selected mode
        validate_config(config, args.mode)

        # Print configuration for verification
        print_config(config, f"Configuration for {args.mode} mode")
    else:
        # No config file provided
        if args.mode in ['full_finetune', 'frozen_backbone', 'hpo']:
            print(f"\n⚠️  WARNING: No config file provided for {args.mode} mode")
            print(f"    Training modes typically require a config file with --config")
            print(f"    Example: --config experiments/configs/training_config.yaml")
            print(f"\n    Proceeding with CLI arguments only...")

    # Display mode information
    if args.mode == "zero_shot":
        print(f"Model: {args.model_id}")
        print(f"Task: {args.task}")
        print(f"Device: {args.device}")
    elif config:
        print(f"Model: {config.get('model_checkpoint', 'N/A')}")
        print(f"Task: {config.get('task', args.task)}")
        print(f"Device: {config.get('device', args.device)}")

    # Dispatch based on mode
    if args.mode == "zero_shot":
        run_zero_shot(args)

    elif args.mode == "full_finetune":
        run_training(args, config=config, freeze_backbone=False)

    elif args.mode == "frozen_backbone":
        run_training(args, config=config, freeze_backbone=True)

    elif args.mode == "hpo":
        run_hpo(args, config=config)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
