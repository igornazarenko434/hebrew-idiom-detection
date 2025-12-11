import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configuration
RESULTS_DIR = Path("experiments/results/full_fine-tuning")
OUTPUT_DIR = Path("experiments/results/analysis")
FIGURES_DIR = Path("paper/figures/finetuning")

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 300})

def load_results():
    """Recursively load all training_results.json and trainer_state.json files."""
    data = []
    
    # Find all training_results.json files
    result_files = list(RESULTS_DIR.rglob("training_results.json"))
    
    print(f"Found {len(result_files)} result files.")
    
    for res_file in result_files:
        try:
            # Extract metadata from path
            # Expected structure: .../model/task/seed_X/training_results.json
            parts = res_file.parts
            seed_part = parts[-2]  # "seed_42"
            task_part = parts[-3]  # "cls"
            model_part = parts[-4] # "alephbert-base"
            
            if not seed_part.startswith("seed_"):
                continue
                
            seed = int(seed_part.split("_")[1])
            model = model_part
            task = task_part
            
            # Load Metrics
            with open(res_file, 'r') as f:
                metrics = json.load(f)
            
            # Load Training History (Loss Curves)
            history = []
            trainer_state_path = res_file.parent / "trainer_state.json"
            if trainer_state_path.exists():
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                    history = state.get("log_history", [])
            
            # Get key metric based on task
            f1 = metrics.get("eval_f1", 0)
            precision = metrics.get("eval_precision", 0)
            recall = metrics.get("eval_recall", 0)
            accuracy = metrics.get("eval_accuracy", 0)
            
            # Append to list
            data.append({
                "model": model,
                "task": task,
                "seed": seed,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "history": history,
                "train_runtime": metrics.get("train_runtime", 0),
                "samples_per_second": metrics.get("train_samples_per_second", 0)
            })
            
        except Exception as e:
            print(f"Error reading {res_file}: {e}")
            
    return pd.DataFrame(data)

def analyze_performance(df):
    """Calculate Mean +/- Std for each model/task."""
    print("\n📊 Aggregating Results...")
    
    # Group by model and task
    agg = df.groupby(["model", "task"]).agg({
        "f1": ["mean", "std", "count"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "accuracy": ["mean", "std"]
    }).reset_index()
    
    # Flatten columns
    agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    
    # Sort by Task then F1 Mean
    agg = agg.sort_values(by=["task", "f1_mean"], ascending=[True, False])
    
    # Save Summary Table
    csv_path = OUTPUT_DIR / "finetuning_summary.csv"
    agg.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")
    
    # Create formatted Markdown Table
    md_path = OUTPUT_DIR / "finetuning_summary.md"
    with open(md_path, "w") as f:
        f.write("# Final Fine-Tuning Results Summary\n\n")
        f.write(agg.to_markdown(index=False, floatfmt=".4f"))
    
    return agg

def plot_learning_curves(df):
    """Plot Training Loss and Validation F1 over time."""
    print("\n📈 Plotting Learning Curves...")
    
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        
        # 1. Validation F1 over Steps
        plt.figure(figsize=(12, 6))
        for model in task_df['model'].unique():
            model_df = task_df[task_df['model'] == model]
            
            # Aggregate histories across seeds
            all_steps = []
            all_f1s = []
            
            for _, row in model_df.iterrows():
                history = row['history']
                # Filter for eval steps
                eval_steps = [x for x in history if 'eval_f1' in x]
                steps = [x['step'] for x in eval_steps]
                f1s = [x['eval_f1'] for x in eval_steps]
                plt.plot(steps, f1s, alpha=0.3, linewidth=1) # Plot individual seeds faintly
                
                # Collect for mean line (simplified approach)
                # In robust research we'd interpolate, but here we just plot all points
            
        plt.title(f"Validation F1 per Step - Task: {task.upper()}")
        plt.xlabel("Training Steps")
        plt.ylabel("F1 Score")
        plt.legend(task_df['model'].unique())
        plt.savefig(FIGURES_DIR / f"learning_curve_f1_{task}.png")
        plt.close()

def plot_performance_comparison(df, agg_df):
    """Bar chart with Error Bars for F1 Score."""
    print("\n📊 Plotting Performance Comparison...")
    
    for task in df['task'].unique():
        plt.figure(figsize=(10, 6))
        
        subset = df[df['task'] == task]
        
        # Create bar plot with error bars (ci='sd' means standard deviation)
        sns.barplot(data=subset, x="model", y="f1", capsize=.1, errorbar="sd", palette="viridis")
        
        plt.title(f"Model Performance (F1 Score) - Task: {task.upper()}")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.5, 1.0) # Zoom in on the top half
        plt.ylabel("F1 Score")
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"performance_comparison_{task}.png")
        plt.close()

def check_significance(df):
    """Perform Paired T-Tests between best model and others."""
    print("\nhai Performing Statistical Significance Tests...")
    
    with open(OUTPUT_DIR / "statistical_significance.txt", "w") as f:
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            
            # Find best model based on mean F1
            means = task_df.groupby("model")["f1"].mean()
            best_model_name = means.idxmax()
            best_scores = task_df[task_df["model"] == best_model_name]["f1"].values
            
            f.write(f"\nTask: {task}\n")
            f.write(f"Best Model: {best_model_name} (Mean F1: {means[best_model_name]:.4f})\n")
            f.write("-