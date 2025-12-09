#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Dive Analysis of Idiom Detection Model
File: src/analyze_deep_dive.py

This script performs advanced analysis on model predictions:
1. Token-level Confusion Matrix (B vs I vs O)
2. Confidence Analysis (Are errors low confidence?)
3. Performance by Idiom Length
4. Performance by Sentence Length
5. Performance by Context (Literal vs Figurative)
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
from sklearn.metrics import confusion_matrix
import warnings

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
ID2LABEL = {0: "O", 1: "B-IDIOM", 2: "I-IDIOM"}
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}

def parse_args():
    parser = argparse.ArgumentParser(description="Deep dive analysis")
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

def predict_with_confidence(model, tokenizer, tokens, true_tags):
    # Tokenize
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
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=2)
        
        # Get predictions and their confidence
        confidence_scores, predictions = torch.max(probs, dim=2)
        confidence_scores = confidence_scores[0].cpu().numpy()
        predictions = predictions[0].cpu().numpy()
    
    # Align to words (first subword strategy)
    aligned_preds = []
    aligned_conf = []
    prev_word_idx = None
    
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            aligned_preds.append(predictions[i])
            aligned_conf.append(confidence_scores[i])
            prev_word_idx = word_idx
            
    # Convert IDs to labels
    pred_tags = [ID2LABEL[p] for p in aligned_preds]
    
    # Truncate
    min_len = min(len(pred_tags), len(true_tags))
    return pred_tags[:min_len], aligned_conf[:min_len]

def main():
    args = parse_args()
    model, tokenizer, df = load_resources(args.data)
    
    # Storage
    all_true_tokens = []
    all_pred_tokens = []
    
    conf_correct = []
    conf_error = []
    
    length_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    sent_len_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    print(f"\n🚀 Running deep dive analysis on {len(df)} samples...")
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if pd.isna(row['tokens']) or pd.isna(row['iob_tags']):
                continue
            
            tokens = literal_eval(row['tokens'])
            true_tags = literal_eval(row['iob_tags'])
            
            pred_tags, conf_scores = predict_with_confidence(model, tokenizer, tokens, true_tags)
            
            # 1. Confusion Matrix Data
            all_true_tokens.extend(true_tags[:len(pred_tags)])
            all_pred_tokens.extend(pred_tags)
            
            # 2. Confidence Analysis
            for t, p, c in zip(true_tags[:len(pred_tags)], pred_tags, conf_scores):
                if t == p:
                    conf_correct.append(c)
                else:
                    conf_error.append(c)
            
            # Sentence Level Correctness
            is_sentence_correct = (pred_tags == true_tags[:len(pred_tags)])
            
            # 3. Idiom Length Analysis
            idiom_len = sum(1 for t in true_tags if t != "O")
            if idiom_len > 0:
                length_stats[idiom_len]["total"] += 1
                if is_sentence_correct:
                    length_stats[idiom_len]["correct"] += 1
                    
            # 4. Sentence Length Analysis
            sent_len = len(tokens)
            # Binning: <10, 10-20, 20-30, 30+
            if sent_len < 10: bin_sl = "<10"
            elif sent_len < 20: bin_sl = "10-20"
            elif sent_len < 30: bin_sl = "20-30"
            else: bin_sl = "30+"
            
            sent_len_stats[bin_sl]["total"] += 1
            if is_sentence_correct:
                sent_len_stats[bin_sl]["correct"] += 1
                
            # 5. Literal vs Figurative
            label_type = "Figurative" if row['label'] == 1 else "Literal"
            type_stats[label_type]["total"] += 1
            if is_sentence_correct:
                type_stats[label_type]["correct"] += 1
                
        except Exception as e:
            print(f"Error row {i}: {e}")
            continue

    # --- REPORT GENERATION ---
    print(f"\n{'='*80}")
    print(f"{ 'DEEP DIVE ANALYSIS REPORT':^80}")
    print(f"{'='*80}")

    # 1. Confusion Matrix
    labels = ["O", "B-IDIOM", "I-IDIOM"]
    cm = confusion_matrix(all_true_tokens, all_pred_tokens, labels=labels)
    
    print("\n1. Token-Level Confusion Matrix:")
    print(f"{ '':>10} | {'Pred O':>8} | {'Pred B':>8} | {'Pred I':>8}")
    print("-" * 46)
    for i, label in enumerate(labels):
        print(f"{ 'True ' + label:>10} | {cm[i][0]:>8} | {cm[i][1]:>8} | {cm[i][2]:>8}")
    
    print("\n   Interpretation:")
    print(f"   - True B misclassified as O: {cm[1][0]} (Missed Idiom Start)")
    print(f"   - True I misclassified as O: {cm[2][0]} (Missed Idiom Part)")
    print(f"   - True B misclassified as I: {cm[1][2]} (Structure Error)")

    # 2. Confidence
    avg_conf_correct = np.mean(conf_correct) * 100 if conf_correct else 0
    avg_conf_error = np.mean(conf_error) * 100 if conf_error else 0
    print(f"\n2. Model Confidence:")
    print(f"   - Average Confidence on Correct Predictions: {avg_conf_correct:.2f}%")
    print(f"   - Average Confidence on Errors:              {avg_conf_error:.2f}%")
    print(f"   - Delta: {avg_conf_correct - avg_conf_error:.2f}% (Higher delta = Model knows when it's unsure)")

    # 3. Idiom Length
    print(f"\n3. Performance by Idiom Length (Tokens):")
    print(f"   {'Length':<10} | {'Count':<10} | {'Accuracy':<10}")
    print("-" * 40)
    for length in sorted(length_stats.keys()):
        stats = length_stats[length]
        acc = (stats["correct"] / stats["total"] * 100)
        print(f"   {length:<10} | {stats['total']:<10} | {acc:.1f}%")

    # 4. Sentence Length
    print(f"\n4. Performance by Sentence Length:")
    print(f"   {'Length':<10} | {'Count':<10} | {'Accuracy':<10}")
    print("-" * 40)
    # Sort bins
    order = ["<10", "10-20", "20-30", "30+"]
    for length in order:
        stats = sent_len_stats[length]
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"   {length:<10} | {stats['total']:<10} | {acc:.1f}%")

    # 5. Context Type
    print(f"\n5. Performance by Context Type:")
    print(f"   {'Type':<15} | {'Count':<10} | {'Accuracy':<10}")
    print("-" * 45)
    for c_type in ["Literal", "Figurative"]:
        stats = type_stats[c_type]
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"   {c_type:<15} | {stats['total']:<10} | {acc:.1f}%")

if __name__ == "__main__":
    main()
