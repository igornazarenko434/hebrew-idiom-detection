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
plt.rcParams.update({'figure.dpi': 300, 'font.family': 'serif'})

def load_results():
    """Recursively load results, logs, and configurations."""
    data = []
    
    # Find all training_results.json files
    result_files = list(RESULTS_DIR.rglob("training_results.json"))
    
    print(f"Found {len(result_files)} result files.")
    
    for res_file in result_files:
        try:
            # Robustly find the seed part in the path
            parts = res_file.parts
            seed_part = next((p for p in reversed(parts) if p.startswith("seed_")), None)
            if not seed_part: continue
                
            seed_idx = parts.index(seed_part)
            seed = int(seed_part.split("_")[1])
            task = parts[seed_idx - 1]
            model = parts[seed_idx - 2]
            
            # 1. Load Metrics
            with open(res_file, 'r') as f:
                metrics = json.load(f)
            
            # 2. Load Training History (Loss Curves)
            history = []
            trainer_state_path = res_file.parent / "trainer_state.json"
            if trainer_state_path.exists():
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                    history = state.get("log_history", [])
            
            # 3. Load Hyperparameters (from config.json if available)
            # Note: config.json usually stores model config, not training args.
            # Training args are often in training_args.bin (binary) or inside trainer_state.json/training_results.json
            # training_results.json usually has some args like batch size
            
            # Try to find learning rate from history or results
            lr = metrics.get("learning_rate", "N/A")
            batch_size = metrics.get("train_batch_size", "N/A")
            epochs = metrics.get("epoch", "N/A")
            
            # Get key metric
            f1 = metrics.get("eval_f1", 0)
            precision = metrics.get("eval_precision", 0)
            recall = metrics.get("eval_recall", 0)
            accuracy = metrics.get("eval_accuracy", 0)
            
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
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs
            })
            
        except Exception as e:
            print(f"Error reading {res_file}: {e}")
            
    return pd.DataFrame(data)

def analyze_performance(df):
    """Calculate Mean +/- Std for each model/task and save summary."""
    print("\n📊 Aggregating Results...")
    
    # Group by model and task
    # We also include hyperparameters in the grouping (assuming they are same per model/task)
    # If they vary by seed (which they shouldn't), this will split them.
    # To be safe, we aggregate stats first.
    
    agg = df.groupby(["model", "task"]).agg({
        "f1": ["mean", "std", "count"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "accuracy": ["mean", "std"],
        "train_runtime": ["mean"],
        "learning_rate": ["first"], # Just take the first one found
        "batch_size": ["first"]
    }).reset_index()
    
    # Flatten columns
    agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    
    # Calculate Coefficient of Variation (Stability)
    # CV = (Std / Mean) * 100. Lower is better/more stable.
    agg["f1_cv"] = (agg["f1_std"] / agg["f1_mean"]) * 100
    
    # Sort by Task then F1 Mean
    agg = agg.sort_values(by=["task", "f1_mean"], ascending=[True, False])
    
    # Clean up column names for display
    agg.rename(columns={"learning_rate_first": "lr", "batch_size_first": "bs"}, inplace=True)
    
    # Save Summary Table
    csv_path = OUTPUT_DIR / "finetuning_summary.csv"
    agg.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")
    
    # Create formatted Markdown Table
    md_path = OUTPUT_DIR / "finetuning_summary.md"
    with open(md_path, "w") as f:
        f.write("# Final Fine-Tuning Results Summary\n\n")
        f.write(agg.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n**Note:** `f1_cv` is Coefficient of Variation (%). Lower means more stable across seeds.")
    
    return agg

def plot_learning_curves(df):
    """Plot Training Loss AND Validation F1 over time."""
    print("\n📈 Plotting Learning Curves (Loss & F1)...")
    
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        
        # Create a figure with 2 subplots side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot 1: Training Loss
        for model in task_df['model'].unique():
            model_df = task_df[task_df['model'] == model]
            
            # Collect all loss points
            for _, row in model_df.iterrows():
                history = row['history']
                loss_steps = [x for x in history if 'loss' in x]
                if not loss_steps: continue
                
                steps = [x['step'] for x in loss_steps]
                losses = [x['loss'] for x in loss_steps]
                
                axes[0].plot(steps, losses, alpha=0.4, linewidth=1, label=model if _ == 0 else "")
        
        axes[0].set_title(f"Training Loss - Task: {task.upper()}")
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Loss")
        # De-duplicate legend
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys())

        # Plot 2: Validation F1
        for model in task_df['model'].unique():
            model_df = task_df[task_df['model'] == model]
            
            for _, row in model_df.iterrows():
                history = row['history']
                eval_steps = [x for x in history if 'eval_f1' in x]
                if not eval_steps: continue
                
                steps = [x['step'] for x in eval_steps]
                f1s = [x['eval_f1'] for x in eval_steps]
                
                axes[1].plot(steps, f1s, alpha=0.4, linewidth=1, label=model if _ == 0 else "")

        axes[1].set_title(f"Validation F1 - Task: {task.upper()}")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("F1 Score")
        # De-duplicate legend
        handles, labels = axes[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[1].legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"learning_curves_{task}.png")
        plt.close()

def plot_performance_comparison(df, agg_df):
    """Bar chart with Error Bars for F1 Score."""
    print("\n📊 Plotting Performance Comparison...")
    
    for task in df['task'].unique():
        plt.figure(figsize=(10, 6))
        subset = df[df['task'] == task]
        
        sns.barplot(data=subset, x="model", y="f1", capsize=.1, errorbar="sd", palette="viridis")
        
        plt.title(f"Model Performance (F1 Score) - Task: {task.upper()}")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.5, 1.0)
        plt.ylabel("F1 Score")
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"performance_comparison_{task}.png")
        plt.close()

def check_significance(df):
    """Perform Paired T-Tests between best model and others."""
    print("\n✅ Performing Statistical Significance Tests...")
    
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