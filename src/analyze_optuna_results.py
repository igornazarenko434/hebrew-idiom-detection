#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna Study Analysis
File: src/analyze_optuna_results.py

This script analyzes the Optuna HPO results stored in SQLite databases.
It generates visualizations and summaries for each study (Mission 4.5 Task 6).

Usage:
    python src/analyze_optuna_results.py --db_dir experiments/results/optuna_studies
"""

import os
import argparse
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from optuna.visualization import matplotlib as optuna_plot

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def analyze_study(db_path, output_base_dir):
    """Analyze a single Optuna study database."""
    study_name = Path(db_path).stem.replace("_hpo", "") # Remove _hpo suffix
    # The actual study name in DB might differ, so we list them
    storage_url = f"sqlite:///{db_path}"
    
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        if not summaries:
            print(f"⚠️  No studies found in {db_path}")
            return

        # Usually 1 study per DB file in our setup
        study_summary = summaries[0]
        actual_study_name = study_summary.study_name
        
        print(f"\n🔍 Analyzing Study: {actual_study_name} (File: {Path(db_path).name})")
        study = optuna.load_study(study_name=actual_study_name, storage=storage_url)
        
        # Create output directory for this study
        study_out_dir = output_base_dir / study_name
        ensure_dir(study_out_dir)
        
        # 1. Best Trial Summary
        if study.best_trial:
            print(f"  🏆 Best F1: {study.best_value:.4f}")
            print(f"  📝 Best Params: {study.best_params}")
            
            # Save summary text
            with open(study_out_dir / "best_params.txt", "w") as f:
                f.write(f"Study: {actual_study_name}\n")
                f.write(f"Best Trial ID: {study.best_trial.number}\n")
                f.write(f"Best Value (F1): {study.best_value:.4f}\n")
                f.write("Params:\n")
                for k, v in study.best_params.items():
                    f.write(f"  {k}: {v}\n")
        else:
            print("  ⚠️  No completed trials found.")
            return

        # 2. Visualizations (Static Matplotlib)
        print("  📊 Generating plots...")
        
        # Optimization History
        try:
            fig = optuna_plot.plot_optimization_history(study)
            fig.figure.savefig(study_out_dir / "optimization_history.png", dpi=300, bbox_inches='tight')
            plt.close(fig.figure) # Close to free memory
        except Exception as e:
            print(f"    - Failed to plot history: {e}")

        # Param Importances
        try:
            fig = optuna_plot.plot_param_importances(study)
            fig.figure.savefig(study_out_dir / "param_importance.png", dpi=300, bbox_inches='tight')
            plt.close(fig.figure)
        except Exception as e:
            print(f"    - Failed to plot importance (maybe too few trials): {e}")
            
        # Slice Plot (Individual Params)
        try:
            fig = optuna_plot.plot_slice(study)
            fig.figure.savefig(study_out_dir / "param_slice.png", dpi=300, bbox_inches='tight')
            plt.close(fig.figure)
        except Exception as e:
            print(f"    - Failed to plot slice: {e}")

        # 3. Export Trials to CSV
        df = study.trials_dataframe()
        df.to_csv(study_out_dir / "all_trials.csv", index=False)
        print(f"  💾 Saved all trials to {study_out_dir / 'all_trials.csv'}")

    except Exception as e:
        print(f"❌ Error processing {db_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna HPO Results")
    parser.add_argument("--db_dir", type=str, default="experiments/results/optuna_studies",
                        help="Directory containing .db files")
    parser.add_argument("--output_dir", type=str, default="experiments/results/hpo_analysis",
                        help="Directory to save analysis results")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    output_dir = Path(args.output_dir)
    
    if not db_dir.exists():
        print(f"❌ Error: Database directory not found: {db_dir}")
        print("Make sure you have synced the results from Google Drive/Vast.ai!")
        return

    db_files = list(db_dir.glob("*.db"))
    
    if not db_files:
        print(f"⚠️  No .db files found in {db_dir}")
        return

    print(f"found {len(db_files)} database files. Starting analysis...")
    ensure_dir(output_dir)

    for db_file in db_files:
        analyze_study(db_file, output_dir)

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
