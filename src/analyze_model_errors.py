#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Model Errors on Unseen Idioms
File: src/analyze_model_errors.py

This script performs a detailed error analysis on the unseen idiom test set:
1. Breaks down performance by Idiom (base_pie).
2. Categorizes errors into types (Missed, Partial, Boundary, Invalid Sequence).
3. Generates a summary report.
"""

import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ast import literal_eval
from tqdm import tqdm
from collections import defaultdict, Counter
import argparse

# Import utilities
try:
    from src.utils.tokenization import align_predictions_with_words
except ImportError:
    sys.path.append('.')
    from src.utils.tokenization import align_predictions_with_words

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "experiments/results/full_fine-tuning/alephbert-base/span"
DEVICE = "cpu"
MAX_LENGTH = 128

# Label mapping
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
ID2LABEL = {0: "O", 1: "B-IDIOM", 2: "I-IDIOM"}

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze model errors")
    parser.add_argument("--data", type=str, default="data/splits/unseen_idiom_test.csv", help="Path to dataset CSV")
    return parser.parse_args()

def load_resources(data_path):
    print(f"📦 Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    
    print(f"📊 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    return model, tokenizer, df

def predict_single(model, tokenizer, tokens, true_tags):
    # Tokenize pre-tokenized input
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=False
    )
    
    word_ids = encoding.word_ids(0)
    inputs = {k: v.to(DEVICE) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    
    # Align
    aligned_preds_ids = []
    prev_word_idx = None
    
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            aligned_preds_ids.append(predictions[i])
            prev_word_idx = word_idx
            
    pred_tags = [ID2LABEL[p] for p in aligned_preds_ids]
    
    # Truncate
    min_len = min(len(pred_tags), len(true_tags))
    return pred_tags[:min_len], true_tags[:min_len]

def categorize_error(true_tags, pred_tags):
    """
    Classify the type of error between true and predicted sequences.
    """
    if true_tags == pred_tags:
        return "Correct"
        
    true_has_idiom = any(t != "O" for t in true_tags)
    pred_has_idiom = any(t != "O" for t in pred_tags)
    
    if not true_has_idiom and not pred_has_idiom:
        return "Correct (Both O)"
        
    if true_has_idiom and not pred_has_idiom:
        return "Complete Miss"
        
    if not true_has_idiom and pred_has_idiom:
        return "Hallucination"
    
    # Check for Invalid Sequence (I without B)
    has_invalid_seq = False
    in_idiom = False
    for t in pred_tags:
        if t == "B-IDIOM":
            in_idiom = True
        elif t == "I-IDIOM":
            if not in_idiom:
                has_invalid_seq = True
        else: # O
            in_idiom = False
            
    if has_invalid_seq:
        # Check if it's ONLY missing the B (e.g. O I I vs B I I)
        # Simplify: just mark as Structural Error
        return "Structural Error (e.g. Missing B-tag)"

    # Overlap analysis
    true_indices = {i for i, t in enumerate(true_tags) if t != "O"}
    pred_indices = {i for i, t in enumerate(pred_tags) if t != "O"}
    
    intersection = true_indices.intersection(pred_indices)
    
    if not intersection:
        return "Wrong Location" # Predicted idiom but in completely wrong place
        
    if pred_indices < true_indices:
        return "Partial Span (Undershoot)"
        
    if true_indices < pred_indices:
        return "Partial Span (Overshoot)"
        
    return "Boundary Mismatch" # Overlapping but neither is subset of other

def main():
    args = parse_args()
    model, tokenizer, df = load_resources(args.data)
    
    # Storage for stats
    idiom_stats = defaultdict(lambda: {"total": 0, "correct": 0, "errors": Counter()})
    
    print(f"\n🚀 Analyzing {len(df)} samples...")
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if pd.isna(row['tokens']) or pd.isna(row['iob_tags']):
                continue
            
            tokens = literal_eval(row['tokens'])
            true_tags = literal_eval(row['iob_tags'])
            base_pie = row.get('base_pie', 'Unknown')
            
            pred_tags, true_tags = predict_single(model, tokenizer, tokens, true_tags)
            
            error_type = categorize_error(true_tags, pred_tags)
            
            idiom_stats[base_pie]["total"] += 1
            idiom_stats[base_pie]["errors"][error_type] += 1
            
            if error_type == "Correct":
                idiom_stats[base_pie]["correct"] += 1
                
        except Exception as e:
            print(f"Error row {i}: {e}")
            continue
            
    # Print Report
    print(f"\n{'='*100}")
    print(f"{ 'IDIOM ANALYSIS REPORT':^100}")
    print(f"{ '='*100}")
    
    # Convert to DataFrame for easier sorting/display
    report_data = []
    for idiom, stats in idiom_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        row_data = {
            "Idiom": idiom,
            "Total": total,
            "Accuracy": f"{accuracy:.1f}%",
            "Missed": stats["errors"]["Complete Miss"],
            "Partial": stats["errors"]["Partial Span (Undershoot)"] + stats["errors"]["Partial Span (Overshoot)"] + stats["errors"]["Boundary Mismatch"],
            "Structure": stats["errors"]["Structural Error (e.g. Missing B-tag)"],
            "Hallucination": stats["errors"]["Hallucination"]
        }
        report_data.append(row_data)
        
    if not report_data:
        print("No results to report.")
        return

    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values(by="Accuracy", ascending=True)
    
    # Formatting for output
    print(f"{ 'Idiom':<30} | {'Total':<5} | {'Acc':<6} | {'Miss':<5} | {'Part':<5} | {'Struct':<6} | {'Halluc':<6}")
    print("-" * 100)
    
    for _, row in report_df.iterrows():
        # Reverse Hebrew string for display
        idiom_disp = row['Idiom'][::-1]
        print(f"{idiom_disp:>30} | {row['Total']:<5} | {row['Accuracy']:<6} | {row['Missed']:<5} | {row['Partial']:<5} | {row['Structure']:<6} | {row['Hallucination']:<6}")
        
    print("-" * 100)
    print("Legend:")
    print("  Miss:      Complete failure to identify the idiom.")
    print("  Part:      Found part of the idiom but missed words or included extras.")
    print("  Struct:    Found words but sequence is invalid (e.g. I-IDIOM without B-IDIOM).")
    print("  Halluc:    Predicted an idiom where there was none.")

if __name__ == "__main__":
    main()
