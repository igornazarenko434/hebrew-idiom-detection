#!/usr/bin/env python3
"""
Verify Morphology Sensitivity Claims for CoNLL 2026 Paper
Analyzes boundary_morphology_manifest.csv to compute:
- Error rates for morphology variants vs exact matches
- Seen vs unseen split comparison
- Per-model breakdown

Priority 1 Task from CONLL_2026_COMPREHENSIVE_REVIEW.md Section II.7
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
MANIFEST_FILE = Path("experiments/results/analysis/interpretability/boundary_morphology_manifest.csv")
OUTPUT_DIR = Path("experiments/results/analysis/morphology_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_manifest():
    """Load the morphology manifest."""
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    df = pd.read_csv(MANIFEST_FILE)
    print(f"✓ Loaded manifest: {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    return df

def categorize_morphology(df):
    """Categorize cases as morphology variant or exact match."""
    # Morphology variant: base_pie != pie_span
    df['is_morphology_variant'] = df['base_pie'] != df['pie_span']

    # Error: not PERFECT
    df['is_error'] = df['error_category'] != 'PERFECT'

    return df

def compute_error_rates(df):
    """Compute error rates by split and morphology type."""
    results = []

    for split in ['seen_test', 'unseen_test']:
        split_df = df[df['split'] == split]

        # Morphology variants
        morph_variants = split_df[split_df['is_morphology_variant']]
        morph_error_rate = morph_variants['is_error'].mean() if len(morph_variants) > 0 else 0.0

        # Exact matches
        exact_matches = split_df[~split_df['is_morphology_variant']]
        exact_error_rate = exact_matches['is_error'].mean() if len(exact_matches) > 0 else 0.0

        results.append({
            'split': split,
            'morphology_variant_count': len(morph_variants),
            'morphology_variant_errors': morph_variants['is_error'].sum(),
            'morphology_variant_error_rate': morph_error_rate,
            'exact_match_count': len(exact_matches),
            'exact_match_errors': exact_matches['is_error'].sum(),
            'exact_match_error_rate': exact_error_rate,
            'gap': morph_error_rate - exact_error_rate
        })

    return pd.DataFrame(results)

def compute_per_model_breakdown(df):
    """Compute error rates per model, split, and morphology type."""
    results = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        for split in ['seen_test', 'unseen_test']:
            split_df = model_df[model_df['split'] == split]

            if len(split_df) == 0:
                continue

            # Morphology variants
            morph_variants = split_df[split_df['is_morphology_variant']]
            morph_error_rate = morph_variants['is_error'].mean() if len(morph_variants) > 0 else 0.0

            # Exact matches
            exact_matches = split_df[~split_df['is_morphology_variant']]
            exact_error_rate = exact_matches['is_error'].mean() if len(exact_matches) > 0 else 0.0

            results.append({
                'model': model,
                'split': split,
                'morphology_variant_count': len(morph_variants),
                'morphology_variant_error_rate': morph_error_rate,
                'exact_match_count': len(exact_matches),
                'exact_match_error_rate': exact_error_rate,
                'gap': morph_error_rate - exact_error_rate
            })

    return pd.DataFrame(results)

def analyze_morphology_error_types(df):
    """Analyze error types for morphology variants."""
    results = []

    for split in ['seen_test', 'unseen_test']:
        split_df = df[(df['split'] == split) & (df['is_morphology_variant'])]

        if len(split_df) == 0:
            continue

        error_counts = split_df['error_category'].value_counts()

        for error_type, count in error_counts.items():
            results.append({
                'split': split,
                'error_category': error_type,
                'count': count,
                'percentage': (count / len(split_df)) * 100
            })

    return pd.DataFrame(results)

def main():
    print("="*70)
    print("MORPHOLOGY SENSITIVITY VERIFICATION")
    print("="*70)

    # Load data
    df = load_manifest()

    # Categorize
    df = categorize_morphology(df)

    print(f"\n✓ Categorization complete:")
    print(f"  Morphology variants: {df['is_morphology_variant'].sum()}")
    print(f"  Exact matches: {(~df['is_morphology_variant']).sum()}")

    # Compute aggregate error rates
    print("\n" + "="*70)
    print("AGGREGATE ERROR RATES (All Models Combined)")
    print("="*70)

    aggregate_results = compute_error_rates(df)
    print("\n" + aggregate_results.to_string(index=False))

    # Save aggregate results
    aggregate_file = OUTPUT_DIR / "morphology_aggregate_error_rates.csv"
    aggregate_results.to_csv(aggregate_file, index=False)
    print(f"\n✓ Saved: {aggregate_file}")

    # Compute per-model breakdown
    print("\n" + "="*70)
    print("PER-MODEL BREAKDOWN")
    print("="*70)

    per_model_results = compute_per_model_breakdown(df)

    # Show unseen results (most important for paper)
    unseen_results = per_model_results[per_model_results['split'] == 'unseen_test'].sort_values('gap', ascending=False)
    print("\nUnseen Test (sorted by gap):")
    print(unseen_results.to_string(index=False))

    # Save per-model results
    per_model_file = OUTPUT_DIR / "morphology_per_model_error_rates.csv"
    per_model_results.to_csv(per_model_file, index=False)
    print(f"\n✓ Saved: {per_model_file}")

    # Analyze error types for morphology variants
    print("\n" + "="*70)
    print("ERROR TYPE DISTRIBUTION (Morphology Variants Only)")
    print("="*70)

    error_types = analyze_morphology_error_types(df)
    print("\n" + error_types.to_string(index=False))

    # Save error type distribution
    error_types_file = OUTPUT_DIR / "morphology_error_types.csv"
    error_types.to_csv(error_types_file, index=False)
    print(f"\n✓ Saved: {error_types_file}")

    # Generate summary report
    print("\n" + "="*70)
    print("SUMMARY FOR PAPER")
    print("="*70)

    # Get key numbers for paper
    seen_agg = aggregate_results[aggregate_results['split'] == 'seen_test'].iloc[0]
    unseen_agg = aggregate_results[aggregate_results['split'] == 'unseen_test'].iloc[0]

    print(f"\n✓ KEY FINDING:")
    print(f"  Seen Test:")
    print(f"    - Morphology variants: {seen_agg['morphology_variant_error_rate']:.1%} error rate ({seen_agg['morphology_variant_errors']:.0f}/{seen_agg['morphology_variant_count']:.0f})")
    print(f"    - Exact matches: {seen_agg['exact_match_error_rate']:.1%} error rate ({seen_agg['exact_match_errors']:.0f}/{seen_agg['exact_match_count']:.0f})")
    print(f"    - Gap: {seen_agg['gap']:.1%}")

    print(f"\n  Unseen Test:")
    print(f"    - Morphology variants: {unseen_agg['morphology_variant_error_rate']:.1%} error rate ({unseen_agg['morphology_variant_errors']:.0f}/{unseen_agg['morphology_variant_count']:.0f})")
    print(f"    - Exact matches: {unseen_agg['exact_match_error_rate']:.1%} error rate ({unseen_agg['exact_match_errors']:.0f}/{unseen_agg['exact_match_count']:.0f})")
    print(f"    - Gap: {unseen_agg['gap']:.1%}")

    # Generate summary markdown
    summary_md = OUTPUT_DIR / "morphology_verification_summary.md"
    with open(summary_md, 'w') as f:
        f.write("# Morphology Sensitivity Verification\n\n")
        f.write("**Priority 1 Task from CoNLL 2026 Review**\n\n")
        f.write("## Research Question\n\n")
        f.write("Do Hebrew morphological surface variations (base idiom ≠ inflected form) ")
        f.write("significantly increase error rates on unseen idioms?\n\n")

        f.write("## Data Source\n\n")
        f.write(f"- Manifest: `{MANIFEST_FILE}`\n")
        f.write(f"- Total cases: {len(df)}\n")
        f.write(f"- Morphology variants: {df['is_morphology_variant'].sum()}\n")
        f.write(f"- Exact matches: {(~df['is_morphology_variant']).sum()}\n\n")

        f.write("## Key Findings\n\n")
        f.write("### Aggregate Results (All Models)\n\n")
        f.write("| Split | Morphology Variant Error Rate | Exact Match Error Rate | Gap |\n")
        f.write("|-------|-------------------------------|------------------------|-----|\n")
        f.write(f"| Seen | {seen_agg['morphology_variant_error_rate']:.1%} | {seen_agg['exact_match_error_rate']:.1%} | {seen_agg['gap']:.1%} |\n")
        f.write(f"| Unseen | {unseen_agg['morphology_variant_error_rate']:.1%} | {unseen_agg['exact_match_error_rate']:.1%} | {unseen_agg['gap']:.1%} |\n\n")

        f.write("### Per-Model Results (Unseen Test)\n\n")
        f.write("| Model | Morphology Variant Error | Exact Match Error | Gap |\n")
        f.write("|-------|--------------------------|-------------------|-----|\n")
        for _, row in unseen_results.iterrows():
            f.write(f"| {row['model']} | {row['morphology_variant_error_rate']:.1%} | {row['exact_match_error_rate']:.1%} | {row['gap']:.1%} |\n")

        f.write("\n## Claim Verification\n\n")

        if unseen_agg['gap'] > 0.10:  # 10% threshold
            f.write("✅ **VERIFIED**: Morphology variants show substantially higher error rates ")
            f.write(f"on unseen idioms ({unseen_agg['gap']:.1%} gap). This supports the claim that ")
            f.write("Hebrew morphological variation drives SPAN errors on novel idioms.\n\n")
        else:
            f.write("❌ **NOT SUPPORTED**: Gap is below 10% threshold. ")
            f.write("Morphology sensitivity claim needs revision.\n\n")

        f.write("## Paper Text (If Verified)\n\n")
        f.write("> Hebrew morphological surface variation (base idiom ≠ inflected form) ")
        f.write(f"significantly elevates unseen error rates. While seen idioms with morphological ")
        f.write(f"variation show {seen_agg['morphology_variant_error_rate']:.1%} error rate ")
        f.write(f"vs. {seen_agg['exact_match_error_rate']:.1%} for exact matches, unseen morphologically-variant ")
        f.write(f"idioms show {unseen_agg['morphology_variant_error_rate']:.1%} error rate ")
        f.write(f"compared to {unseen_agg['exact_match_error_rate']:.1%} for exact-match unseen idioms. ")
        f.write("This indicates models struggle to generalize boundary patterns across inflectional ")
        f.write("paradigms in Hebrew, where a single idiom can appear in dozens of surface forms.\n")

    print(f"\n✓ Saved summary: {summary_md}")

    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review summary in morphology_verification_summary.md")
    print("2. If verified, include claim in paper Section 7 (Discussion)")
    print("3. Reference aggregate_error_rates.csv for exact numbers")

if __name__ == "__main__":
    main()
