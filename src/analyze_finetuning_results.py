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
                data_json = json.load(f)
            
            # Extract nested dictionaries
            test_metrics = data_json.get("test_metrics", {})
            config = data_json.get("config", {})
            history = data_json.get("training_history", [])
            
            # Fallback to trainer_state.json if history is missing in main file
            if not history:
                trainer_state_path = res_file.parent / "trainer_state.json"
                if trainer_state_path.exists():
                    with open(trainer_state_path, 'r') as f:
                        state = json.load(f)
                        history = state.get("log_history", [])

            # 2. Extract Hyperparameters
            lr = config.get("learning_rate")
            batch_size = config.get("batch_size", "N/A")
            epochs = config.get("num_epochs")
            
            # Ensure LR and Epochs are floats for consistency
            if isinstance(lr, (int, float)):
                lr = float(lr)
            else:
                lr = np.nan # Use NaN if not a valid number
            
            # Ensure LR and Epochs are floats for consistency
            if isinstance(epochs, (int, float)):
                epochs = float(epochs)
            else:
                epochs = np.nan
            
            # Debug print to see the actual LR being extracted
            print(f"DEBUG: Processing {res_file.name}. Extracted LR: {lr} (type: {type(lr)})")
            
            # Get key metric based on task
            f1 = test_metrics.get("f1", test_metrics.get("eval_f1", 0))
            precision = test_metrics.get("precision", test_metrics.get("eval_precision", 0))
            recall = test_metrics.get("recall", test_metrics.get("eval_recall", 0))
            accuracy = test_metrics.get("accuracy", test_metrics.get("eval_accuracy", 0))
            
            # Extract Confusion Matrix (if available)
            tn = test_metrics.get("confusion_matrix_tn", np.nan)
            fp = test_metrics.get("confusion_matrix_fp", np.nan)
            fn = test_metrics.get("confusion_matrix_fn", np.nan)
            tp = test_metrics.get("confusion_matrix_tp", np.nan)
            
            # Get runtime from train_metrics if available
            train_metrics = data_json.get("train_metrics", {})
            train_runtime = train_metrics.get("runtime", 0)
            
            # Append to list
            data.append({
                "model": model,
                "task": task,
                "seed": seed,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "history": history,
                "train_runtime": train_runtime,
                "learning_rate": lr, # Now this should be float
                "batch_size": batch_size,
                "epochs": epochs
            })
            
        except Exception as e:
            print(f"Error reading {res_file}: {e}. Extracted LR: {lr} (type: {type(lr)})")
            
    return pd.DataFrame(data)

def analyze_performance(df):
    """Calculate Mean +/- Std for each model/task and save summary."""
    print("\n📊 Aggregating Results...")
    
    # 1. Save Detailed (Per-Seed) CSV
    detail_csv_path = OUTPUT_DIR / "finetuning_seeds_detail.csv"
    # Drop history column for CSV readability
    df_detail = df.drop(columns=["history"])
    df_detail.to_csv(detail_csv_path, index=False)
    print(f"Saved detailed seed report to {detail_csv_path}")

    # 2. Aggregation
    agg = df.groupby(["model", "task"]).agg({
        "f1": ["mean", "std", "count"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "accuracy": ["mean", "std"],
        "train_runtime": ["mean"],
        "learning_rate": ["first"],
        "batch_size": ["first"],
        "tp": ["mean"], "tn": ["mean"], "fp": ["mean"], "fn": ["mean"]
    }).reset_index()
    
    agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    
    # Derived Metrics
    # Stability: CV = (Std / Mean) * 100
    agg["f1_cv"] = (agg["f1_std"] / agg["f1_mean"]) * 100
    
    # Efficiency: F1 per minute of training
    agg["efficiency_score"] = agg["f1_mean"] / (agg["train_runtime_mean"] / 60)
    
    agg = agg.sort_values(by=["task", "f1_mean"], ascending=[True, False])
    agg.rename(columns={"learning_rate_first": "lr", "batch_size_first": "bs"}, inplace=True)
    
    # Format LR to scientific notation string for consistent display in markdown
    def format_lr(x):
        try:
            return f"{float(x):.1e}"
        except (ValueError, TypeError):
            return str(x)

    agg["lr"] = pd.to_numeric(agg["lr"], errors='coerce').apply(lambda x: f"{x:.1e}" if pd.notna(x) else x)
    
    # Save Summary Table
    csv_path = OUTPUT_DIR / "finetuning_summary.csv"
    agg.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")
    
    # Create formatted Markdown Table
    md_path = OUTPUT_DIR / "finetuning_summary.md"
    with open(md_path, "w") as f:
        f.write("# Final Fine-Tuning Results Summary\n\n")
        f.write(agg.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n**Notes:**\n")
        f.write("- `f1_cv`: Coefficient of Variation (%). Lower is more stable.\n")
        f.write("- `efficiency_score`: F1 score per minute of training time.\n")
        f.write("- `tp_mean`, `fp_mean`, etc.: Average raw counts from Confusion Matrix.\n")
    
    return agg

def plot_learning_curves(df):
    """Plot Training Loss AND Validation F1 over time."""
    print("\n📈 Plotting Learning Curves (Loss & F1)...")
    
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot 1: Training Loss
        for model in task_df['model'].unique():
            model_df = task_df[task_df['model'] == model]
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
            f.write("-" * 50 + "\n")
            
            for model in task_df['model'].unique():
                if model == best_model_name:
                    continue
                
                compare_scores = task_df[task_df["model"] == model]["f1"].values
                
                if len(best_scores) == len(compare_scores) and len(best_scores) > 1:
                    t_stat, p_val = stats.ttest_rel(best_scores, compare_scores)
                    is_sig = "SIGNIFICANT" if p_val < 0.05 else "Not Significant"
                    f.write(f"vs {model}: p={p_val:.4f} ({is_sig})\n")
                else:
                    f.write(f"vs {model}: Cannot compute paired t-test (unequal samples)\n")

def main():
    print("Starting Deep Analysis of Fine-Tuning Results...")
    df = load_results()
    
    if df.empty:
        print("❌ No results found! Make sure you ran download_results_for_analysis.sh first.")
        return

    # Basic Analysis
    agg_df = analyze_performance(df)
    
    # Visualizations
    plot_performance_comparison(df, agg_df)
    plot_learning_curves(df)
    
    # Statistics
    check_significance(df)
    
    print("\n✅ Analysis Complete!")
    print(f"Results: {OUTPUT_DIR}")
    print(f"Figures: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
