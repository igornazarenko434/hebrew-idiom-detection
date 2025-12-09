#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Position Bias in Model Predictions
File: src/analyze_position_bias.py

This script investigates if the model's performance is biased by the position
of the idiom in the sentence. It calculates the relative position of the idiom
and correlates it with prediction accuracy.
"""

import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ast import literal_eval
from tqdm import tqdm
from collections import defaultdict
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
ID2LABEL = {0: "O", 1: "B-IDIOM", 2: "I-IDIOM"}

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze position bias")
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

def get_idiom_position_ratio(row, tokens):
    """Calculate relative position of idiom start (0.0 to 1.0)"""
    # Use 'start_token' from CSV if available and valid
    if 'start_token' in row and not pd.isna(row['start_token']):
        start_idx = int(row['start_token'])
        # Verify it's within bounds
        if 0 <= start_idx < len(tokens):
            return start_idx / len(tokens)
            
    # Fallback: Find first B-IDIOM or I-IDIOM in tags
    try:
        tags = literal_eval(row['iob_tags'])
        for i, tag in enumerate(tags):
            if tag != 'O':
                return i / len(tokens)
    except:
        pass
        
    return -1.0 # Unknown

def main():
    args = parse_args()
    model, tokenizer, df = load_resources(args.data)
    
    # Bins for position: 0-0.2 (Beginning), 0.2-0.8 (Middle), 0.8-1.0 (End)
    # Or more granular: 0.1 bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["0-20% (Start)", "20-40%", "40-60% (Mid)", "60-80%", "80-100% (End)"]
    
    position_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    print(f"\n🚀 Analyzing position bias in {len(df)} samples...")
    
    valid_samples = 0
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if pd.isna(row['tokens']) or pd.isna(row['iob_tags']):
                continue
                
            tokens = literal_eval(row['tokens'])
            true_tags = literal_eval(row['iob_tags'])
            
            # Get Position Ratio
            ratio = get_idiom_position_ratio(row, tokens)
            if ratio < 0:
                continue
                
            valid_samples += 1
            
            # Determine Bin
            bin_idx = -1
            for b_i in range(len(bins) - 1):
                if bins[b_i] <= ratio < bins[b_i+1]:
                    bin_idx = b_i
                    break
            
            if bin_idx == -1: continue
            bin_name = bin_labels[bin_idx]
            
            # Predict
            pred_tags, true_tags = predict_single(model, tokenizer, tokens, true_tags)
            
            # Check correctness (Strict exact match)
            is_correct = (pred_tags == true_tags)
            
            position_stats[bin_name]["total"] += 1
            if is_correct:
                position_stats[bin_name]["correct"] += 1
                
        except Exception as e:
            print(f"Error row {i}: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"{'.':^80}")
    print(f"{'.':^80}")
    print(f"Total Valid Samples Analyzed: {valid_samples}")
    
    print(f"\n{'Position (Sentence %)':<25} | {'Count':<10} | {'Accuracy':<10}")
    print("-" * 60)
    
    for bin_name in bin_labels:
        stats = position_stats[bin_name]
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total * 100) if total > 0 else 0.0
        
        print(f"{bin_name:<25} | {total:<10} | {acc:.1f}%")
        
    print("-" * 60)
    print("Interpretation:")
    print(" - If accuracy is significantly higher in '0-20% (Start)', the model")
    print("   might be biased towards finding idioms at the beginning.")
    print(" - Uniform accuracy suggests the model is robust to position.")

if __name__ == "__main__":
    main()
