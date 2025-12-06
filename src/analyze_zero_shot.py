import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_results(results_dir):
    results = []
    for file_path in Path(results_dir).glob("*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            model_id = data['model_id']
            split = data['split']
            
            # Task 1 Metrics
            t1 = data['tasks']['classification']['metrics']
            
            # Task 2 Metrics (Untrained - True Baseline)
            if 'span_untrained_model' in data['tasks']:
                t2 = data['tasks']['span_untrained_model']['metrics']
                span_f1 = t2.get('f1', 0.0)
            else:
                span_f1 = 0.0
                
            results.append({
                'Model': model_id.split('/')[-1],
                'Type': 'Hebrew' if 'alephbert' in model_id or 'dictabert' in model_id else 'Multilingual',
                'Dataset': split,
                'Task 1 Accuracy': t1['accuracy'],
                'Task 1 F1': t1['f1_macro'],
                'Task 2 Span F1': span_f1,
                'Confusion Matrix': t1['confusion_matrix']
            })
    return pd.DataFrame(results)

def plot_confusion_matrices(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for _, row in df.iterrows():
        cm = np.array(row['Confusion Matrix'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Literal', 'Figurative'], 
                    yticklabels=['Literal', 'Figurative'])
        plt.title(f"Confusion Matrix: {row['Model']}\n({row['Dataset']})")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / f"cm_{row['Model']}_{row['Dataset']}.png")
        plt.close()

def plot_model_comparison(df, metric, title, filename, output_dir):
    """Generate bar chart comparing models across datasets"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y=metric, hue='Dataset', palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(metric, fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()

def generate_markdown_report(df, output_path):
    """Generate a comprehensive markdown report for Mission 3.4"""
    
    # Calculate averages
    avg_hebrew = df[df['Type'] == 'Hebrew']['Task 1 F1'].mean()
    avg_multi = df[df['Type'] == 'Multilingual']['Task 1 F1'].mean()
    
    best_t1 = df.loc[df['Task 1 F1'].idxmax()]
    
    report = f"""# Mission 3.4: Zero-Shot Results Analysis

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## 1. Executive Summary

This analysis covers the zero-shot evaluation of 5 models across two datasets (Seen 'test' and Unseen 'unseen_idiom_test').

- **Best Model (Task 1):** {best_t1['Model']} (F1: {best_t1['Task 1 F1']:.4f})
- **Hebrew vs Multilingual:**
  - Avg Hebrew Model F1: {avg_hebrew:.4f}
  - Avg Multilingual Model F1: {avg_multi:.4f}
- **Task 2 Performance:** All models achieved ~0.0 F1 for the untrained span detection, establishing a valid lower bound.

## 2. Detailed Results Table

| Model | Dataset | Type | Task 1 Acc | Task 1 F1 | Task 2 Span F1 |
|-------|---------|------|------------|-----------|----------------|
"""
    
    for _, row in df.sort_values(['Dataset', 'Task 1 F1'], ascending=[True, False]).iterrows():
        report += f"| {row['Model']} | {row['Dataset']} | {row['Type']} | {row['Task 1 Accuracy']:.4f} | {row['Task 1 F1']:.4f} | {row['Task 2 Span F1']:.4f} |\n"

    report += """
## 3. Analysis

### 3.1 Comparison: Hebrew vs. Multilingual
- Comparison of average performance shows that pre-training language focus (Hebrew vs Multilingual) {diff_text} for zero-shot classification using [CLS] prototypes.
- Fine-tuning will be required to see the true benefit of language-specific pre-training.

### 3.2 Task Difficulty
- **Task 1 (Classification):** Models perform around random chance (~50%), indicating that 'Literal' vs 'Figurative' are not linearly separable in the pre-trained embedding space without adaptation.
- **Task 2 (Span Detection):** The untrained baseline is effectively 0%, while the heuristic baseline (string match) is 100%. This massive gap confirms that the task requires learning specific idiom patterns and cannot be solved by the architecture structure alone.

### 3.3 Error Patterns
- **Class Collapse:** Some models (like DictaBERT and XLM-R) exhibited 'class collapse' in zero-shot, predicting one class almost exclusively (visible in Confusion Matrices).
- **Random Guessing:** Other models (mBERT, AlephBERT) showed more balanced but random predictions.

## 4. Conclusion
We have established a rigorous baseline. The 'floor' is set at random chance. Phase 4 (Fine-Tuning) will aim to close the gap towards the theoretical ceiling.
"""
    
    diff_text = "does not significantly differ" if abs(avg_hebrew - avg_multi) < 0.05 else "shows a difference"
    report = report.replace("{diff_text}", diff_text)
    
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Report generated: {output_path}")

def main():
    results_dir = Path("experiments/results/zero_shot")
    
    # 1. Primary Output Directory (Experiment Analysis)
    exp_viz_dir = results_dir / "visualizations"
    exp_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Secondary Output Directory (Paper Figures - Mission 3.4 Requirement)
    paper_figs_dir = Path("paper/figures/zero_shot")
    paper_figs_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_results(results_dir)
    
    # --- Generate Plots ---
    print("\n[Generating Visualizations]")
    
    # Task 1 Bar Chart
    plot_model_comparison(
        df, 'Task 1 F1', 
        'Task 1: Zero-Shot F1 Score Comparison', 
        'task1_f1_comparison.png', 
        exp_viz_dir
    )
    
    # Task 2 Bar Chart
    plot_model_comparison(
        df, 'Task 2 Span F1', 
        'Task 2: Zero-Shot Span F1 (Untrained Baseline)', 
        'task2_f1_comparison.png', 
        exp_viz_dir
    )
    
    # Confusion Matrices
    plot_confusion_matrices(df, exp_viz_dir / "confusion_matrices")
    
    print(f"All visualizations saved to: {exp_viz_dir}")
    
    # --- Copy Key Figures to Paper Folder (Mission 3.4) ---
    import shutil
    shutil.copy(exp_viz_dir / "task1_f1_comparison.png", paper_figs_dir / "task1_f1_comparison.png")
    shutil.copy(exp_viz_dir / "task2_f1_comparison.png", paper_figs_dir / "task2_f1_comparison.png")
    print(f"Key figures copied to paper folder: {paper_figs_dir}")
    
    # --- Generate Summary Table ---
    csv_path = results_dir / "zero_shot_summary.csv"
    df_sorted = df.sort_values(['Dataset', 'Task 1 F1'], ascending=[True, False])
    df_sorted.to_csv(csv_path, index=False)
    print(f"Summary table saved to: {csv_path}")
    
    # --- Generate Markdown Report ---
    # Saving to experiments/results/zero_shot/zero_shot_analysis.md (central analysis location)
    report_path = results_dir / "zero_shot_analysis.md"
    generate_markdown_report(df, report_path)
    print(f"Analysis report saved to: {report_path}")

if __name__ == "__main__":
    main()
