import os
import json
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from tabulate import tabulate

# Configuration
OUTPUT_DIR = Path("experiments/results/analysis/deep_dive")
FIGURES_DIR = Path("paper/figures/deep_dive")

# Ensure output dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 300, 'font.family': 'serif'})

def load_predictions(file_path):
    """Load eval_predictions.json."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def analyze_task1_cls(df, model_name, dataset_name):
    """Deep analysis for Sequence Classification (Task 1)."""
    print(f"\n🔍 Analyzing Task 1 (Classification) for {model_name} on {dataset_name}...")
    
    # 1. Error Categorization
    df['error_type'] = 'Correct'
    df.loc[(df['true_label'] == 0) & (df['predicted_label'] == 1), 'error_type'] = 'False Positive (Literal->Figurative)'
    df.loc[(df['true_label'] == 1) & (df['predicted_label'] == 0), 'error_type'] = 'False Negative (Figurative->Literal)'
    
    # Count errors
    error_counts = df['error_type'].value_counts()
    print("\nError Distribution:")
    print(tabulate(error_counts.to_frame(), headers=["Type", "Count"], tablefmt="github"))
    
    # 2. Confidence Analysis
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="confidence", hue="is_correct", multiple="stack", bins=20, palette={True: "green", False: "red"})
    plt.title(f"Confidence Distribution (Correct vs Incorrect) - {model_name}")
    plt.xlabel("Model Confidence")
    plt.ylabel("Count")
    plt.savefig(FIGURES_DIR / f"confidence_dist_{model_name}_{dataset_name}.png")
    plt.close()
    
    # 3. Hardest Idioms
    print("\n🔥 Hardest Idioms (Highest Error Rate):")
    idiom_stats = df.groupby('expression').agg(
        total=('id', 'count'),
        errors=('is_correct', lambda x: (~x).sum())
    )
    idiom_stats['error_rate'] = idiom_stats['errors'] / idiom_stats['total']
    hardest = idiom_stats.sort_values('error_rate', ascending=False).head(10)
    print(tabulate(hardest, headers=["Idiom", "Total", "Errors", "Error Rate"], tablefmt="github", floatfmt=".2f"))
    
    # Save hardest idioms to CSV
    hardest.to_csv(OUTPUT_DIR / f"hardest_idioms_{model_name}_{dataset_name}.csv")

    # 4. Save Top Errors (High Confidence but Wrong)
    top_errors = df[~df['is_correct']].sort_values("confidence", ascending=False).head(20)
    top_errors_path = OUTPUT_DIR / f"top_errors_{model_name}_{dataset_name}.csv"
    top_errors[['sentence', 'expression', 'true_label', 'predicted_label', 'confidence', 'error_type']].to_csv(top_errors_path, index=False)
    print(f"\nSaved top 20 high-confidence errors to: {top_errors_path}")

def analyze_task2_span(df, model_name, dataset_name):
    """Deep analysis for Token Classification/Span (Task 2)."""
    print(f"\n🔍 Analyzing Task 2 (Span) for {model_name} on {dataset_name}...")
    
    span_errors = []
    
    for idx, row in df.iterrows():
        true_tags = row['true_tags']
        pred_tags = row['predicted_tags']
        sentence = row['sentence']
        
        # Extract spans (indices of B-IDIOM ... I-IDIOM)
        def get_spans(tags):
            spans = []
            start = -1
            for i, tag in enumerate(tags):
                if tag == "B-IDIOM":
                    if start != -1: spans.append((start, i-1)) # Close previous
                    start = i
                elif tag == "O" and start != -1:
                    spans.append((start, i-1))
                    start = -1
            if start != -1: spans.append((start, len(tags)-1))
            return spans

        true_spans = get_spans(true_tags)
        pred_spans = get_spans(pred_tags)
        
        # Classify Errors
        # Simple heuristic matching
        matched_pred = [False] * len(pred_spans)
        
        for t_start, t_end in true_spans:
            found_match = False
            for j, (p_start, p_end) in enumerate(pred_spans):
                if matched_pred[j]: continue
                
                # Check Overlap
                overlap = max(0, min(t_end, p_end) - max(t_start, p_start) + 1)
                if overlap > 0:
                    found_match = True
                    matched_pred[j] = True
                    
                    if t_start == p_start and t_end == p_end:
                        error_type = "EXACT_MATCH"
                    elif t_start == p_start:
                        error_type = "BOUNDARY_END_ERROR"
                    elif t_end == p_end:
                        error_type = "BOUNDARY_START_ERROR"
                    else:
                        error_type = "PARTIAL_OVERLAP"
                        
                    span_errors.append({
                        "id": row.get("id", idx),
                        "sentence": sentence,
                        "true_span": f"{t_start}-{t_end}",
                        "pred_span": f"{p_start}-{p_end}",
                        "error_type": error_type
                    })
                    break
            
            if not found_match:
                span_errors.append({
                    "id": row.get("id", idx),
                    "sentence": sentence,
                    "true_span": f"{t_start}-{t_end}",
                    "pred_span": "NONE",
                    "error_type": "MISSING_SPAN"
                })
                
        # Check for spurious (hallucinated) spans
        for j, (p_start, p_end) in enumerate(pred_spans):
            if not matched_pred[j]:
                span_errors.append({
                    "id": row.get("id", idx),
                    "sentence": sentence,
                    "true_span": "NONE",
                    "pred_span": f"{p_start}-{p_end}",
                    "error_type": "SPURIOUS_SPAN"
                })

    # Analysis of Span Errors
    err_df = pd.DataFrame(span_errors)
    
    # Filter out exact matches to see error distribution
    errors_only = err_df[err_df['error_type'] != "EXACT_MATCH"]
    
    print("\nSpan Error Distribution:")
    if not errors_only.empty:
        counts = errors_only['error_type'].value_counts()
        print(tabulate(counts.to_frame(), headers=["Error Type", "Count"], tablefmt="github"))
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.countplot(y="error_type", data=errors_only, palette="viridis")
        plt.title(f"Span Error Types - {model_name}")
        plt.savefig(FIGURES_DIR / f"span_errors_{model_name}_{dataset_name}.png")
        plt.close()
        
        # Save detailed error log
        err_path = OUTPUT_DIR / f"span_error_log_{model_name}_{dataset_name}.csv"
        errors_only.to_csv(err_path, index=False)
        print(f"\nSaved detailed span error log to: {err_path}")
    else:
        print("  No span errors found! Perfect performance?")

def main():
    parser = argparse.ArgumentParser(description="Deep Comprehensive Analysis of Model Predictions")
    parser.add_argument("--prediction_file", type=str, required=True, help="Path to eval_predictions.json")
    parser.add_argument("--task", type=str, required=True, choices=["cls", "span"], help="Task type")
    parser.add_argument("--model_name", type=str, default="UnknownModel", help="Name of model for reporting")
    parser.add_argument("--dataset_name", type=str, default="UnknownData", help="Name of dataset (Seen/Unseen)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.prediction_file):
        print(f"❌ File not found: {args.prediction_file}")
        return

    df = load_predictions(args.prediction_file)
    
    if args.task == "cls":
        analyze_task1_cls(df, args.model_name, args.dataset_name)
    else:
        analyze_task2_span(df, args.model_name, args.dataset_name)

    print(f"\n✅ Deep analysis complete. Check {OUTPUT_DIR} and {FIGURES_DIR}")

if __name__ == "__main__":
    main()
