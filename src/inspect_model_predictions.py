#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect Idiom Model Predictions
File: src/inspect_model_predictions.py

This script:
1. Loads the trained AlephBERT span detection model
2. Runs inference on the unseen idiom test set
3. Prints detailed examples of Correct vs Incorrect predictions
4. Shows the original sentence, true tags, and predicted tags
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ast import literal_eval
from tqdm import tqdm

# Import utilities
try:
    from src.utils.tokenization import align_predictions_with_words
except ImportError:
    # Handle if running from root
    sys.path.append('.')
    from src.utils.tokenization import align_predictions_with_words

import argparse

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "experiments/results/full_fine-tuning/alephbert-base/span"
DEVICE = "cpu"  # Use CPU for simple inference inspection
MAX_LENGTH = 128

# Label mapping (must match training)
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
ID2LABEL = {0: "O", 1: "B-IDIOM", 2: "I-IDIOM"}

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect model predictions")
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

# ... (rest of the file)

def main():
    args = parse_args()
    model, tokenizer, df = load_resources(args.data)
    
    # ... (rest of main function)

def predict_single(model, tokenizer, tokens, true_tags):
    """Run prediction on a pre-tokenized sentence"""
    # Tokenize pre-tokenized input
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=False
    )
    
    # Get word IDs BEFORE moving to device
    word_ids = encoding.word_ids(0)
    
    inputs = {k: v.to(DEVICE) for k, v in encoding.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    
    # Align subword predictions to original words
    aligned_preds_ids = []
    prev_word_idx = None
    
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # We only care about the first subword for each original word
        if word_idx != prev_word_idx:
            aligned_preds_ids.append(predictions[i])
            prev_word_idx = word_idx
            
    # Convert IDs to labels
    pred_tags = [ID2LABEL[p] for p in aligned_preds_ids]
    
    # Truncate if lengths mismatch (due to tokenizer truncation)
    min_len = min(len(pred_tags), len(true_tags))
    pred_tags = pred_tags[:min_len]
    true_tags = true_tags[:min_len]
    tokens = tokens[:min_len]
    
    return pred_tags, true_tags, tokens

def main():
    args = parse_args()
    model, tokenizer, df = load_resources(args.data)
    
    correct_examples = []
    incorrect_examples = []
    
    print(f"\n🚀 Running inference on {len(df)} samples...")
    
    total_correct = 0
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Parse stringified lists
            # Handle possible float/nan values if bad data
            if pd.isna(row['tokens']) or pd.isna(row['iob_tags']):
                continue
                
            tokens = literal_eval(row['tokens'])
            true_tags = literal_eval(row['iob_tags'])
            
            pred_tags, true_tags, tokens = predict_single(model, tokenizer, tokens, true_tags)
            
            # Check correctness
            is_correct = (pred_tags == true_tags)
            
            # Store result
            result = {
                "id": i,
                "sentence": row['sentence'],
                "tokens": tokens,
                "true_tags": true_tags,
                "pred_tags": pred_tags,
                "idiom": row.get('base_pie', 'N/A')
            }
            
            if is_correct:
                total_correct += 1
                if len(correct_examples) < 5:  # Keep 5 examples
                    correct_examples.append(result)
            else:
                if len(incorrect_examples) < 10:  # Keep 10 examples
                    incorrect_examples.append(result)
                    
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    accuracy = total_correct / len(df)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{ '='*80}")
    print(f"Total Samples: {len(df)}")
    print(f"Perfect Sentence Matches: {total_correct} ({accuracy:.2%})")
    print(f"Incorrect Sentence Matches: {len(df) - total_correct} ({1-accuracy:.2%})")
    
    print(f"\n{'='*80}")
    print(f"✅ CORRECT EXAMPLES (First 3)")
    print(f"{ '='*80}")
    for ex in correct_examples[:3]:
        print(f"\nSentence: {ex['sentence']}")
        print(f"Idiom:    {ex['idiom']}")
        
        # Print side by side
        print(f"{ 'Word':<20} {'True':<10} {'Pred':<10}")
        print("-" * 45)
        for w, t, p in zip(ex['tokens'], ex['true_tags'], ex['pred_tags']):
            # Reverse Hebrew word for display
            w_rev = w[::-1]
            print(f"{w_rev:<20} {t:<10} {p:<10}")
            
    print(f"\n{'='*80}")
    print(f"❌ INCORRECT EXAMPLES (First 5)")
    print(f"{ '='*80}")
    for ex in incorrect_examples[:5]:
        print(f"\nSentence: {ex['sentence']}")
        print(f"Idiom:    {ex['idiom']}")
        
        # Highlight mismatches
        print(f"{ 'Word':<20} {'True':<10} {'Pred':<10} {'Status'}")
        print("-" * 55)
        for w, t, p in zip(ex['tokens'], ex['true_tags'], ex['pred_tags']):
            status = "✅" if t == p else "❌"
            w_rev = w[::-1]
            print(f"{w_rev:<20} {t:<10} {p:<10} {status}")

if __name__ == "__main__":
    main()