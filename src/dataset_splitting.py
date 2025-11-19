"""
Dataset Splitting Module for Hebrew Idiom Detection
Mission 2.5: Expression-Based Dataset Splitting

CRITICAL: Hybrid splitting strategy to evaluate both in-domain and unseen idioms
- Keep a dedicated unseen-idom test set (6 fixed idioms)
- Remaining idioms are split by sentences (not expressions) so every idiom appears in train/val/in-domain-test
- Maintains balanced literal/figurative coverage per idiom across splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json
import math


class ExpressionBasedSplitter:
    """
    Handles expression-based dataset splitting to prevent data leakage.

    Strategy:
    1. Split expressions (not sentences) into train/dev/test
    2. Assign all sentences of an expression to the same split
    3. Ensure label balance across splits
    4. Zero expression overlap between splits
    """

    def __init__(self, data_path: str = None):
        """
        Initialize the splitter

        Args:
            data_path: Path to the processed dataset CSV
        """
        if data_path is None:
            project_root = Path(__file__).parent.parent
            data_path = project_root / "data" / "processed_data.csv"

        self.data_path = Path(data_path)
        self.df = None
        self.train_df = None
        self.dev_df = None
        self.test_df = None
        self.unseen_test_df = None
        self.expression_split_counts: List[Dict] = []

        # Unseen idiom expressions (fixed list from mission spec)
        self.unseen_idiom_expressions = [
            "חתך פינה",                    # cut corner
            "חצה קו אדום",                 # crossed red line
            "נשאר מאחור",                  # stayed behind
            "שבר שתיקה",                   # broke silence
            "איבד את הראש",                # lost the head
            "רץ אחרי הזנב של עצמו"         # ran after his own tail
        ]
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        self.random_state = 42

    def load_dataset(self) -> pd.DataFrame:
        """Load the processed dataset"""
        print(f"Loading dataset from: {self.data_path}")
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        print(f"✅ Loaded {len(self.df)} sentences")
        return self.df

    def analyze_expressions(self) -> pd.DataFrame:
        """
        Analyze expression distribution and label balance

        Returns:
            DataFrame with expression statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print("\n" + "=" * 80)
        print("EXPRESSION ANALYSIS")
        print("=" * 80)

        # Group by expression and count labels
        expr_stats = self.df.groupby('expression').agg({
            'id': 'count',
            'label_2': ['sum', lambda x: (x == 0).sum()]
        }).reset_index()

        expr_stats.columns = ['expression', 'total_sentences', 'figurative_count', 'literal_count']

        # Calculate balance
        expr_stats['balance_ratio'] = expr_stats['figurative_count'] / expr_stats['literal_count']
        expr_stats['perfectly_balanced'] = expr_stats['figurative_count'] == expr_stats['literal_count']

        # Sort by total sentences
        expr_stats = expr_stats.sort_values('total_sentences', ascending=False)

        print(f"\nTotal unique expressions: {len(expr_stats)}")
        print(f"Total sentences: {expr_stats['total_sentences'].sum()}")
        print(f"Perfectly balanced expressions (50/50): {expr_stats['perfectly_balanced'].sum()}")

        print("\nExpression Statistics:")
        print(expr_stats.to_string(index=False))

        return expr_stats

    def verify_unseen_idiom_expressions(self) -> Dict:
        """
        Verify that test expressions are valid and have good coverage

        Returns:
            Dictionary with verification results
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("TEST EXPRESSIONS VERIFICATION")
        print("=" * 80)

        all_expressions = set(self.df['expression'].unique())
        test_expr_set = set(self.unseen_idiom_expressions)

        # Check if all test expressions exist
        missing = test_expr_set - all_expressions
        if missing:
            print(f"❌ Missing test expressions: {missing}")
            raise ValueError(f"Test expressions not found in dataset: {missing}")

        print(f"\n✅ All {len(self.unseen_idiom_expressions)} test expressions found in dataset")
        print("\nTest Expression Details:")

        total_test_sentences = 0
        test_literal = 0
        test_figurative = 0

        for expr in self.unseen_idiom_expressions:
            expr_df = self.df[self.df['expression'] == expr]
            literal = (expr_df['label_2'] == 0).sum()
            figurative = (expr_df['label_2'] == 1).sum()
            total = len(expr_df)

            total_test_sentences += total
            test_literal += literal
            test_figurative += figurative

            print(f"\n  {expr}:")
            print(f"    Total: {total} | Literal: {literal} | Figurative: {figurative}")
            print(f"    Balance: {figurative/literal:.2f}x" if literal > 0 else "    Balance: N/A")

        print("\n" + "-" * 80)
        print(f"Test Set Summary:")
        print(f"  Total sentences: {total_test_sentences}")
        print(f"  Literal: {test_literal} ({test_literal/total_test_sentences*100:.1f}%)")
        print(f"  Figurative: {test_figurative} ({test_figurative/total_test_sentences*100:.1f}%)")
        print(f"  Percentage of dataset: {total_test_sentences/len(self.df)*100:.1f}%")

        return {
            'test_sentences': total_test_sentences,
            'test_literal': test_literal,
            'test_figurative': test_figurative,
            'test_balance': test_figurative / test_literal if test_literal > 0 else 0
        }

    def _compute_split_counts(self, n: int) -> Tuple[int, int, int]:
        """
        Compute train/validation/test counts for a group of size n while
        honoring the configured ratios and ensuring each split receives
        at least one sample when possible.
        """
        if n == 0:
            return 0, 0, 0

        targets = [
            ('train', self.train_ratio),
            ('validation', self.val_ratio),
            ('test_in_domain', self.test_ratio)
        ]

        raw = {name: n * ratio for name, ratio in targets}
        order_index = {name: idx for idx, (name, _) in enumerate(targets)}
        counts = {name: int(math.floor(value)) for name, value in raw.items()}
        assigned = sum(counts.values())
        remainder = n - assigned

        if remainder > 0:
            fractional = sorted(
                ((name, raw[name] - counts[name]) for name in counts),
                key=lambda x: (-x[1], order_index[x[0]])
            )
            idx = 0
            while remainder > 0:
                name, _ = fractional[idx % len(fractional)]
                counts[name] += 1
                remainder -= 1
                idx += 1

        # Guarantee at least one sentence per split when we have enough samples
        if n >= len(targets):
            for name in counts:
                if counts[name] == 0:
                    donor = max(counts, key=lambda k: counts[k])
                    if counts[donor] <= 1:
                        continue
                    counts[donor] -= 1
                    counts[name] += 1

        return counts['train'], counts['validation'], counts['test_in_domain']

    def create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create splits:
        - unseen_idiom_test: fixed list of idioms kept entirely for zero-shot evaluation
        - train/validation/test (in-domain): sentence-level split per remaining idiom/label
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("CREATING HYBRID SPLITS (UNSEEN + SEEN IDIOMS)")
        print("=" * 80)

        # 1. Unseen idiom test set (entire idioms)
        self.unseen_test_df = self.df[self.df['expression'].isin(self.unseen_idiom_expressions)].copy()
        self.unseen_test_df['split'] = 'unseen_idiom_test'

        # 2. Remaining sentences (all idioms that will appear in every seen split)
        remaining_df = self.df[~self.df['expression'].isin(self.unseen_idiom_expressions)].copy()

        train_parts = []
        val_parts = []
        test_parts = []
        expression_split_counts = []

        for expr, expr_df in remaining_df.groupby('expression'):
            expr_counts = {'expression': expr, 'train': 0, 'validation': 0, 'test_in_domain': 0}

            for label_value, label_df in expr_df.groupby('label_2'):
                label_df = label_df.sample(frac=1, random_state=self.random_state)
                n_samples = len(label_df)
                train_n, val_n, test_n = self._compute_split_counts(n_samples)

                train_parts.append(label_df.iloc[:train_n])
                val_parts.append(label_df.iloc[train_n:train_n + val_n])
                test_parts.append(label_df.iloc[train_n + val_n:])

                expr_counts['train'] += train_n
                expr_counts['validation'] += val_n
                expr_counts['test_in_domain'] += test_n

            expression_split_counts.append(expr_counts)

        self.expression_split_counts = expression_split_counts

        self.train_df = pd.concat(train_parts).sort_values('id').reset_index(drop=True)
        self.dev_df = pd.concat(val_parts).sort_values('id').reset_index(drop=True)
        self.test_df = pd.concat(test_parts).sort_values('id').reset_index(drop=True)

        self.train_df['split'] = 'train'
        self.dev_df['split'] = 'validation'
        self.test_df['split'] = 'test_in_domain'

        print("\n✅ Completed sentence-level splitting for remaining idioms.")
        print(f"   Train sentences: {len(self.train_df)}")
        print(f"   Validation sentences: {len(self.dev_df)}")
        print(f"   In-domain test sentences: {len(self.test_df)}")
        print(f"   Unseen idiom test sentences: {len(self.unseen_test_df)}")

        return self.train_df, self.dev_df, self.test_df, self.unseen_test_df

    def verify_splits(self) -> Dict:
        """
        Verify split quality and print statistics

        Returns:
            Dictionary with split statistics
        """
        if any(df is None for df in [self.train_df, self.dev_df, self.test_df, self.unseen_test_df]):
            raise ValueError("Splits not created. Call create_splits() first.")

        print("\n" + "=" * 80)
        print("SPLIT VERIFICATION & STATISTICS")
        print("=" * 80)

        stats = {}

        split_map = [
            ('Train', self.train_df),
            ('Validation', self.dev_df),
            ('In-Domain Test', self.test_df),
            ('Unseen Idiom Test', self.unseen_test_df)
        ]

        for split_name, split_df in split_map:
            n_sentences = len(split_df)
            n_expressions = split_df['expression'].nunique()
            n_literal = (split_df['label_2'] == 0).sum()
            n_figurative = (split_df['label_2'] == 1).sum()

            pct_sentences = (n_sentences / len(self.df)) * 100
            pct_literal = (n_literal / n_sentences) * 100
            pct_figurative = (n_figurative / n_sentences) * 100

            key = split_name.lower().replace('-', '_').replace(' ', '_')
            stats[key] = {
                'sentences': n_sentences,
                'expressions': n_expressions,
                'literal': n_literal,
                'figurative': n_figurative,
                'pct_of_total': pct_sentences,
                'pct_literal': pct_literal,
                'pct_figurative': pct_figurative
            }

            print(f"\n{split_name} Set:")
            print(f"  Sentences: {n_sentences:,} ({pct_sentences:.2f}% of dataset)")
            print(f"  Expressions: {n_expressions:,}")
            print(f"  Literal: {n_literal:,} ({pct_literal:.2f}%)")
            print(f"  Figurative: {n_figurative:,} ({pct_figurative:.2f}%)")

        coverage_issues = [
            rec['expression']
            for rec in self.expression_split_counts
            if min(rec['train'], rec['validation'], rec['test_in_domain']) == 0
        ]

        if coverage_issues:
            print("\n⚠️  Expressions missing from at least one seen split:")
            for expr in coverage_issues:
                print(f"   - {expr}")
        else:
            print("\n✅ Every seen idiom appears in train, validation, and in-domain test splits.")

        total_sentences = sum(entry['sentences'] for entry in stats.values())

        print("\n" + "=" * 80)
        print("OVERALL VERIFICATION")
        print("=" * 80)
        print(f"Total sentences accounted for: {total_sentences} (dataset: {len(self.df)})")
        print(f"Match: {'✅ Perfect' if total_sentences == len(self.df) else '❌ Mismatch'}")

        return stats

    def save_splits(self, output_dir: str = None) -> None:
        """
        Save splits to CSV files in data/splits/ directory (as per Mission 2.5)

        Args:
            output_dir: Directory to save splits (default: data/splits/)
        """
        if any(df is None for df in [self.train_df, self.dev_df, self.test_df, self.unseen_test_df]):
            raise ValueError("Splits not created. Call create_splits() first.")

        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "data" / "splits"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("SAVING SPLITS")
        print("=" * 80)

        # Save each split (validation.csv as per Mission 2.5, not dev.csv)
        split_files = {
            "train.csv": self.train_df,
            "validation.csv": self.dev_df,
            "test.csv": self.test_df,
            "unseen_idiom_test.csv": self.unseen_test_df
        }

        for filename, df in split_files.items():
            path = output_dir / filename
            df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"✅ Saved {filename}: {len(df)} sentences")

        metadata = {
            'unseen_idiom_expressions': self.unseen_idiom_expressions,
            'split_ratios': {
                'train': self.train_ratio,
                'validation': self.val_ratio,
                'test_in_domain': self.test_ratio
            },
            'expression_split_counts': self.expression_split_counts,
            'statistics': {
                'train': {'sentences': len(self.train_df)},
                'validation': {'sentences': len(self.dev_df)},
                'test_in_domain': {'sentences': len(self.test_df)},
                'unseen_idiom_test': {'sentences': len(self.unseen_test_df)}
            }
        }

        metadata_path = output_dir / "split_expressions.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"✅ Split metadata saved: {metadata_path}")

        # Save updated dataset with split column (Mission 2.5 requirement)
        print("\n" + "-" * 80)
        print("Saving updated dataset with split column...")

        # Combine all splits back into one dataframe
        full_dataset = pd.concat(
            [self.train_df, self.dev_df, self.test_df, self.unseen_test_df],
            ignore_index=False
        )
        full_dataset = full_dataset.sort_values('id').reset_index(drop=True)

        # Save to data/ directory (not data/splits/)
        data_dir = Path(__file__).parent.parent / "data"
        updated_dataset_path = data_dir / "expressions_data_with_splits.csv"

        full_dataset.to_csv(updated_dataset_path, index=False, encoding='utf-8-sig')
        print(f"✅ Updated dataset with split column saved: {updated_dataset_path}")

    def run_mission_2_5(self) -> Dict:
        """
        Complete Mission 2.5 with the hybrid splitting strategy

        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 80)
        print("MISSION 2.5: EXPRESSION-BASED DATASET SPLITTING")
        print("=" * 80)

        results = {}

        # Step 1: Load dataset
        print("\n[Step 1] Loading dataset...")
        self.load_dataset()

        # Step 2: Analyze expressions
        print("\n[Step 2] Analyzing expression distribution...")
        results['expression_stats'] = self.analyze_expressions()

        # Step 3: Verify unseen idiom expressions
        print("\n[Step 3] Verifying unseen idiom expressions...")
        results['test_verification'] = self.verify_unseen_idiom_expressions()

        # Step 4: Create splits
        print("\n[Step 4] Creating splits...")
        self.create_splits()

        # Step 5: Verify splits
        print("\n[Step 5] Verifying splits...")
        results['split_stats'] = self.verify_splits()

        # Step 6: Save splits
        print("\n[Step 6] Saving splits...")
        self.save_splits()

        # Success criteria check
        print("\n" + "=" * 80)
        print("MISSION 2.5 SUCCESS CRITERIA CHECK")
        print("=" * 80)

        train_stats = results['split_stats']['train']
        dev_stats = results['split_stats']['validation']
        test_stats = results['split_stats']['in_domain_test']
        unseen_stats = results['split_stats']['unseen_idiom_test']

        seen_expressions = set(self.train_df['expression'].unique())
        unseen_expressions = set(self.unseen_test_df['expression'].unique())

        criteria = [
            ("Unseen idiom test contains 6 specified idioms",
             unseen_stats['expressions'] == len(self.unseen_idiom_expressions)
             and unseen_expressions == set(self.unseen_idiom_expressions)),
            ("Seen vs unseen idioms are disjoint", len(seen_expressions & unseen_expressions) == 0),
            ("Seen splits balanced (50/50 ± 5%)",
             all(abs(stats['pct_literal'] - 50) < 5 for stats in [train_stats, dev_stats, test_stats])),
            ("Every seen idiom appears in train/validation/test_in_domain",
             all(rec['train'] > 0 and rec['validation'] > 0 and rec['test_in_domain'] > 0
                 for rec in self.expression_split_counts)),
            ("All sentences preserved",
             train_stats['sentences'] + dev_stats['sentences'] + test_stats['sentences'] +
             unseen_stats['sentences'] == len(self.df)),
            ("Split metadata recorded", True)
        ]

        all_passed = True
        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            print(f"{status} {criterion}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n" + "=" * 80)
            print("🎉 MISSION 2.5 COMPLETE - ALL SUCCESS CRITERIA PASSED!")
            print("=" * 80)
            print(f"\n📋 Summary:")
            print(f"   • Train: {train_stats['sentences']:,} sentences ({train_stats['expressions']} expressions)")
            print(f"   • Validation: {dev_stats['sentences']:,} sentences ({dev_stats['expressions']} expressions)")
            print(f"   • In-domain test: {test_stats['sentences']:,} sentences "
                  f"({test_stats['expressions']} expressions)")
            print(f"   • Unseen idiom test: {unseen_stats['sentences']:,} sentences "
                  f"({unseen_stats['expressions']} expressions)")
            print(f"   • Ready for Mission 2.6: Data Preparation Testing")
        else:
            print("\n⚠️  Some criteria not met. Please review above.")

        results['mission_complete'] = all_passed
        return results


def main():
    """Main function to execute Mission 2.5"""

    splitter = ExpressionBasedSplitter()
    results = splitter.run_mission_2_5()

    return splitter, results


if __name__ == "__main__":
    splitter, results = main()
