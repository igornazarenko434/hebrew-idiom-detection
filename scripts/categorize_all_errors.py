#!/usr/bin/env python3
"""
Categorize errors for all models using shared taxonomy.
Task 1.3: Complete Error Categorization
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.error_analysis import categorize_span_error, categorize_cls_error


def categorize_errors_for_model(model, task, seed, split):
    """Categorize errors for single model/task/seed/split."""
    pred_file = Path(f"experiments/results/evaluation/{split}/{model}/{task}/seed_{seed}/eval_predictions.json")

    if not pred_file.exists():
        return None

    # Load predictions
    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # Categorize each prediction
    categorized_count = 0
    for pred in predictions:
        if task == "cls":
            # CLS task: needs true_label and predicted_label
            if 'true_label' in pred and 'predicted_label' in pred:
                pred['error_category'] = categorize_cls_error(
                    pred['true_label'],
                    pred['predicted_label']
                )
                categorized_count += 1
        elif task == "span":
            # SPAN task: needs true_tags and predicted_tags
            if 'true_tags' in pred and 'predicted_tags' in pred:
                pred['error_category'] = categorize_span_error(
                    pred['true_tags'],
                    pred['predicted_tags']
                )
                categorized_count += 1

    # Count errors
    error_counts = Counter(p.get('error_category', 'UNKNOWN') for p in predictions)

    # Save updated predictions
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    return {
        'total': len(predictions),
        'categorized': categorized_count,
        'error_counts': error_counts
    }


def main():
    """Main categorization pipeline."""
    models = [
        'alephbert-base',
        'alephbertgimmel-base',
        'dictabert',
        'neodictabert',
        'bert-base-multilingual-cased',
        'xlm-roberta-base'
    ]
    tasks = ['cls', 'span']
    seeds = [42, 123, 456]
    splits = ['seen_test', 'unseen_test']

    print("="*70)
    print("ERROR CATEGORIZATION - Task 1.3")
    print("="*70)
    print(f"\nCategorizing errors for:")
    print(f"  - {len(models)} models")
    print(f"  - {len(tasks)} tasks (cls, span)")
    print(f"  - {len(seeds)} seeds (42, 123, 456)")
    print(f"  - {len(splits)} splits (seen_test, unseen_test)")
    print(f"  - Total: {len(models) * len(tasks) * len(seeds) * len(splits)} files")
    print()

    total_processed = 0
    total_categorized = 0
    failed = []

    for model in models:
        for task in tasks:
            for seed in seeds:
                for split in splits:
                    result = categorize_errors_for_model(model, task, seed, split)
                    if result:
                        total_processed += result['total']
                        total_categorized += result['categorized']

                        # Show top 3 error categories
                        top_errors = result['error_counts'].most_common(3)
                        error_str = ", ".join([f"{err}: {cnt}" for err, cnt in top_errors])

                        print(f"✓ {model:30s} {task:4s} seed_{seed} {split:12s} - {error_str}")
                    else:
                        failed.append(f"{model}/{task}/seed_{seed}/{split}")
                        print(f"✗ {model:30s} {task:4s} seed_{seed} {split:12s} - FILE NOT FOUND")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total predictions processed: {total_processed:,}")
    print(f"Successfully categorized: {total_categorized:,}")
    print(f"Failed files: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f}")

    print("\n✅ Error categorization complete!")
    print("\nNext step: Verify by checking a sample file:")
    print("  python -c \"import json; preds = json.load(open('experiments/results/evaluation/seen_test/dictabert/span/seed_42/eval_predictions.json')); print('Sample categories:', [p.get('error_category') for p in preds[:5]])\"")


if __name__ == "__main__":
    main()
