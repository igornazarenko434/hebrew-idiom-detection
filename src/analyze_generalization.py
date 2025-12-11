import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Configuration
EVAL_DIR = Path("experiments/results/evaluation")
OUTPUT_DIR = Path("experiments/results/analysis/generalization")
FIGURES_DIR = Path("paper/figures/generalization")

# Ensure output dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 300, 'font.family': 'serif'})

def load_eval_data():
    """Load all evaluation JSONs for both Seen and Unseen sets."""
    data = []
    
    # We expect two main subfolders: 'seen_test' and 'unseen_test'
    test_types = ["seen_test", "unseen_test"]
    
    for test_type in test_types:
        search_path = EVAL_DIR / test_type
        if not search_path.exists():
            print(f"⚠️  Warning: Path not found: {search_path}")
            continue
            
        json_files = list(search_path.rglob("eval_results*.json"))
        print(f"Found {len(json_files)} result files in {test_type}")
        
        for f in json_files:
            try:
                # Path structure: .../seen_test/MODEL/TASK/SEED/eval_results.json
                # We need to extract metadata robustly
                parts = f.parts
                # Go backwards to find key components
                seed_part = next((p for p in reversed(parts) if p.startswith("seed_")), None)
                if not seed_part: continue
                
                seed_idx = parts.index(seed_part)
                task = parts[seed_idx - 1]
                model = parts[seed_idx - 2]
                
                with open(f, 'r') as json_file:
                    res = json.load(json_file)
                
                # Extract F1 (handle metrics structure differences)
                f1 = res.get("eval_f1", res.get("f1", 0))
                
                data.append({
                    "model": model,
                    "task": task,
                    "seed": int(seed_part.replace("seed_", "")),
                    "test_set": "Seen" if test_type == "seen_test" else "Unseen",
                    "f1": f1
                })
            except Exception as e:
                print(f"Error parsing {f}: {e}")
                
    return pd.DataFrame(data)

def analyze_gap(df):
    """Calculate the Generalization Gap (Seen - Unseen)."""
    print("\n📊 Calculating Generalization Gap...")
    
    # Pivot table to get Mean F1 for Seen/Unseen per Model/Task
    agg = df.groupby(["model", "task", "test_set"])["f1"].mean().unstack()
    
    # agg structure:
    # test_set        Seen    Unseen
    # model   task
    # aleph   cls     0.95    0.70
    
    if "Seen" not in agg.columns or "Unseen" not in agg.columns:
        print("❌ Error: Need both Seen and Unseen results to calculate gap.")
        return None

    agg["gap_absolute"] = agg["Seen"] - agg["Unseen"]
    agg["gap_percent"] = (agg["gap_absolute"] / agg["Seen"]) * 100
    
    # Sort by Unseen F1 (Performance on the hard task)
    agg = agg.sort_values("Unseen", ascending=False)
    
    # Save table
    csv_path = OUTPUT_DIR / "generalization_gap.csv"
    agg.to_csv(csv_path)
    
    # Markdown Report
    md_path = OUTPUT_DIR / "generalization_report.md"
    with open(md_path, "w") as f:
        f.write("# Generalization Analysis (Seen vs Unseen)\n\n")
        f.write(agg.to_markdown(floatfmt=".4f"))
        f.write("\n\n**Note:** 'Gap' is the performance drop. Lower gap means better robustness.")
        
    print(f"Saved analysis to {md_path}")
    return agg

def plot_generalization(df):
    """Generate side-by-side bar chart."""
    print("\n📈 Plotting Generalization Comparison...")
    
    for task in df['task'].unique():
        plt.figure(figsize=(12, 6))
        task_df = df[df['task'] == task]
        
        # Bar chart with hue=test_set
        sns.barplot(
            data=task_df, 
            x="model", 
            y="f1", 
            hue="test_set", 
            palette={"Seen": "#3498db", "Unseen": "#e74c3c"},
            capsize=0.1,
            errorbar="sd"
        )
        
        plt.title(f"Generalization Performance: {task.upper()} Task")
        plt.ylabel("F1 Score")
        plt.xlabel("Model")
        plt.ylim(0.4, 1.0) # Focus on relevant range
        plt.legend(title="Test Set")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = FIGURES_DIR / f"generalization_bar_{task}.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close()

def plot_stability(df):
    """Generate boxplots to show variance across seeds on Unseen data."""
    print("\n📈 Plotting Stability (Variance)...")
    
    # Filter only for Unseen data (where stability matters most)
    unseen_df = df[df['test_set'] == "Unseen"]
    
    if unseen_df.empty:
        return

    for task in unseen_df['task'].unique():
        plt.figure(figsize=(10, 6))
        subset = unseen_df[unseen_df['task'] == task]
        
        sns.boxplot(data=subset, x="model", y="f1", palette="Set3")
        sns.stripplot(data=subset, x="model", y="f1", color="black", alpha=0.5) # Show individual points
        
        plt.title(f"Stability on Unseen Idioms (Variance across seeds) - {task.upper()}")
        plt.ylabel("F1 Score (Unseen)")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = FIGURES_DIR / f"stability_boxplot_{task}.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close()

def main():
    print("Starting Deep Generalization Analysis...")
    df = load_eval_data()
    
    if df.empty:
        print("❌ No evaluation data found. Run 'scripts/run_evaluation_batch.sh' first.")
        return

    # 1. Table Analysis
    analyze_gap(df)
    
    # 2. Visualizations
    plot_generalization(df)
    plot_stability(df)
    
    print("\n✅ Analysis Complete!")

if __name__ == "__main__":
    main()
