"""
Dataset Splitting Module for Hebrew Idiom Detection
Mission 2.5: Expression-Based Dataset Splitting

CRITICAL: Expression-based splitting to prevent data leakage
- No expression should appear in multiple splits
- This ensures the model generalizes to unseen idioms
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json


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

        # Test expressions: 6 specific idioms as defined in STEP_BY_STEP_MISSIONS.md
        # Mission 2.5 requires these EXACT expressions
        self.test_expressions = [
            "חתך פינה",                    # cut corner
            "חצה קו אדום",                 # crossed red line
            "נשאר מאחור",                  # stayed behind
            "שבר שתיקה",                   # broke silence
            "איבד את הראש",                # lost the head
            "רץ אחרי הזנב של עצמו"         # ran after his own tail
        ]

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

    def verify_test_expressions(self) -> Dict:
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
        test_expr_set = set(self.test_expressions)

        # Check if all test expressions exist
        missing = test_expr_set - all_expressions
        if missing:
            print(f"❌ Missing test expressions: {missing}")
            raise ValueError(f"Test expressions not found in dataset: {missing}")

        print(f"\n✅ All {len(self.test_expressions)} test expressions found in dataset")
        print("\nTest Expression Details:")

        total_test_sentences = 0
        test_literal = 0
        test_figurative = 0

        for expr in self.test_expressions:
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

    def select_dev_expressions(self, n_dev: int = 6) -> List[str]:
        """
        Select dev expressions using stratified sampling

        Args:
            n_dev: Number of dev expressions (default: 6 for ~10%)

        Returns:
            List of dev expression names
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print(f"SELECTING {n_dev} DEV EXPRESSIONS")
        print("=" * 80)

        # Get all expressions except test expressions
        available_expressions = self.df[
            ~self.df['expression'].isin(self.test_expressions)
        ]['expression'].unique()

        print(f"\nAvailable expressions (excluding test): {len(available_expressions)}")

        # Calculate balance for each expression
        expr_stats = []
        for expr in available_expressions:
            expr_df = self.df[self.df['expression'] == expr]
            literal = (expr_df['label_2'] == 0).sum()
            figurative = (expr_df['label_2'] == 1).sum()
            total = len(expr_df)

            # Prefer balanced expressions
            balance_score = 1 / (abs(literal - figurative) + 1)

            expr_stats.append({
                'expression': expr,
                'total': total,
                'literal': literal,
                'figurative': figurative,
                'balance_score': balance_score
            })

        # Sort by balance score (prefer balanced) and total sentences (prefer medium-sized)
        expr_df = pd.DataFrame(expr_stats)
        expr_df['size_score'] = 1 / (abs(expr_df['total'] - expr_df['total'].median()) + 1)
        expr_df['combined_score'] = expr_df['balance_score'] * expr_df['size_score']
        expr_df = expr_df.sort_values('combined_score', ascending=False)

        # Select top n_dev expressions
        dev_expressions = expr_df.head(n_dev)['expression'].tolist()

        print("\nSelected Dev Expressions:")
        for i, expr in enumerate(dev_expressions, 1):
            stats = expr_df[expr_df['expression'] == expr].iloc[0]
            print(f"{i}. {expr}")
            print(f"   Total: {stats['total']} | Literal: {stats['literal']} | Figurative: {stats['figurative']}")

        # Calculate dev set statistics
        dev_df = self.df[self.df['expression'].isin(dev_expressions)]
        dev_literal = (dev_df['label_2'] == 0).sum()
        dev_figurative = (dev_df['label_2'] == 1).sum()

        print("\n" + "-" * 80)
        print(f"Dev Set Summary:")
        print(f"  Total sentences: {len(dev_df)}")
        print(f"  Literal: {dev_literal} ({dev_literal/len(dev_df)*100:.1f}%)")
        print(f"  Figurative: {dev_figurative} ({dev_figurative/len(dev_df)*100:.1f}%)")
        print(f"  Percentage of dataset: {len(dev_df)/len(self.df)*100:.1f}%")

        return dev_expressions

    def create_splits(self, n_dev: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/dev/test splits based on expressions

        Args:
            n_dev: Number of dev expressions

        Returns:
            Tuple of (train_df, dev_df, test_df)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("CREATING EXPRESSION-BASED SPLITS")
        print("=" * 80)

        # 1. Create test split
        self.test_df = self.df[self.df['expression'].isin(self.test_expressions)].copy()

        # 2. Select and create dev split
        dev_expressions = self.select_dev_expressions(n_dev=n_dev)
        self.dev_df = self.df[self.df['expression'].isin(dev_expressions)].copy()

        # 3. Create train split (remaining expressions)
        used_expressions = set(self.test_expressions) | set(dev_expressions)
        self.train_df = self.df[~self.df['expression'].isin(used_expressions)].copy()

        # Set split column
        self.train_df['split'] = 'train'
        self.dev_df['split'] = 'dev'
        self.test_df['split'] = 'test'

        # Verify no overlap
        train_expr = set(self.train_df['expression'].unique())
        dev_expr = set(self.dev_df['expression'].unique())
        test_expr = set(self.test_df['expression'].unique())

        overlap_train_dev = train_expr & dev_expr
        overlap_train_test = train_expr & test_expr
        overlap_dev_test = dev_expr & test_expr

        if overlap_train_dev or overlap_train_test or overlap_dev_test:
            print("❌ EXPRESSION OVERLAP DETECTED!")
            print(f"Train-Dev overlap: {overlap_train_dev}")
            print(f"Train-Test overlap: {overlap_train_test}")
            print(f"Dev-Test overlap: {overlap_dev_test}")
            raise ValueError("Expression overlap detected between splits!")

        print("\n✅ Zero expression overlap verified!")

        return self.train_df, self.dev_df, self.test_df

    def verify_splits(self) -> Dict:
        """
        Verify split quality and print statistics

        Returns:
            Dictionary with split statistics
        """
        if self.train_df is None or self.dev_df is None or self.test_df is None:
            raise ValueError("Splits not created. Call create_splits() first.")

        print("\n" + "=" * 80)
        print("SPLIT VERIFICATION & STATISTICS")
        print("=" * 80)

        stats = {}

        for split_name, split_df in [('Train', self.train_df), ('Dev', self.dev_df), ('Test', self.test_df)]:
            n_sentences = len(split_df)
            n_expressions = split_df['expression'].nunique()
            n_literal = (split_df['label_2'] == 0).sum()
            n_figurative = (split_df['label_2'] == 1).sum()

            pct_sentences = (n_sentences / len(self.df)) * 100
            pct_literal = (n_literal / n_sentences) * 100
            pct_figurative = (n_figurative / n_sentences) * 100

            stats[split_name.lower()] = {
                'sentences': n_sentences,
                'expressions': n_expressions,
                'literal': n_literal,
                'figurative': n_figurative,
                'pct_of_total': pct_sentences,
                'pct_literal': pct_literal,
                'pct_figurative': pct_figurative
            }

            print(f"\n{split_name} Set:")
            print(f"  Sentences: {n_sentences:,} ({pct_sentences:.1f}% of total)")
            print(f"  Expressions: {n_expressions} ({n_expressions/60*100:.1f}% of 60)")
            print(f"  Literal: {n_literal:,} ({pct_literal:.1f}%)")
            print(f"  Figurative: {n_figurative:,} ({pct_figurative:.1f}%)")
            print(f"  Balance: {'✅ Good' if abs(pct_literal - 50) < 5 else '⚠️ Imbalanced'}")

        # Overall verification
        total_sentences = stats['train']['sentences'] + stats['dev']['sentences'] + stats['test']['sentences']
        total_expressions = stats['train']['expressions'] + stats['dev']['expressions'] + stats['test']['expressions']

        print("\n" + "=" * 80)
        print("OVERALL VERIFICATION")
        print("=" * 80)
        print(f"Total sentences: {total_sentences} (original: {len(self.df)})")
        print(f"Total expressions: {total_expressions} (should be 60)")
        print(f"Match: {'✅ Perfect' if total_sentences == len(self.df) and total_expressions == 60 else '❌ Mismatch'}")

        return stats

    def save_splits(self, output_dir: str = None) -> None:
        """
        Save splits to CSV files in data/splits/ directory (as per Mission 2.5)

        Args:
            output_dir: Directory to save splits (default: data/splits/)
        """
        if self.train_df is None or self.dev_df is None or self.test_df is None:
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
        train_path = output_dir / "train.csv"
        validation_path = output_dir / "validation.csv"
        test_path = output_dir / "test.csv"

        self.train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        print(f"✅ Train set saved: {train_path} ({len(self.train_df)} sentences)")

        self.dev_df.to_csv(validation_path, index=False, encoding='utf-8-sig')
        print(f"✅ Validation set saved: {validation_path} ({len(self.dev_df)} sentences)")

        self.test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
        print(f"✅ Test set saved: {test_path} ({len(self.test_df)} sentences)")

        # Save split metadata (split_expressions.json as per Mission 2.5)
        metadata = {
            'test_expressions': self.test_expressions,
            'validation_expressions': list(self.dev_df['expression'].unique()),
            'train_expressions': list(self.train_df['expression'].unique()),
            'statistics': {
                'train': {'sentences': len(self.train_df), 'expressions': self.train_df['expression'].nunique()},
                'validation': {'sentences': len(self.dev_df), 'expressions': self.dev_df['expression'].nunique()},
                'test': {'sentences': len(self.test_df), 'expressions': self.test_df['expression'].nunique()}
            }
        }

        metadata_path = output_dir / "split_expressions.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"✅ Split expressions mapping saved: {metadata_path}")

        # Save updated dataset with split column (Mission 2.5 requirement)
        print("\n" + "-" * 80)
        print("Saving updated dataset with split column...")

        # Combine all splits back into one dataframe
        full_dataset = pd.concat([self.train_df, self.dev_df, self.test_df], ignore_index=False)
        full_dataset = full_dataset.sort_values('id').reset_index(drop=True)

        # Save to data/ directory (not data/splits/)
        data_dir = Path(__file__).parent.parent / "data"
        updated_dataset_path = data_dir / "expressions_data_with_splits.csv"

        full_dataset.to_csv(updated_dataset_path, index=False, encoding='utf-8-sig')
        print(f"✅ Updated dataset with split column saved: {updated_dataset_path}")

    def run_mission_2_5(self, n_dev: int = 6) -> Dict:
        """
        Complete Mission 2.5: Expression-Based Dataset Splitting

        Args:
            n_dev: Number of dev expressions

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

        # Step 3: Verify test expressions
        print("\n[Step 3] Verifying test expressions...")
        results['test_verification'] = self.verify_test_expressions()

        # Step 4: Create splits
        print("\n[Step 4] Creating splits...")
        self.create_splits(n_dev=n_dev)

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
        dev_stats = results['split_stats']['dev']
        test_stats = results['split_stats']['test']

        # Verify set intersections are empty (no data leakage)
        train_expr = set(self.train_df['expression'].unique())
        val_expr = set(self.dev_df['expression'].unique())
        test_expr = set(self.test_df['expression'].unique())

        criteria = [
            ("Test set: 6 specific expressions only", test_stats['expressions'] == 6),
            ("Zero data leakage: test ∩ train = ∅", len(test_expr & train_expr) == 0),
            ("Zero data leakage: test ∩ val = ∅", len(test_expr & val_expr) == 0),
            ("Zero data leakage: train ∩ val = ∅", len(train_expr & val_expr) == 0),
            ("Each split balanced (50/50 ± 5%)",
             abs(train_stats['pct_literal'] - 50) < 5 and
             abs(dev_stats['pct_literal'] - 50) < 5 and
             abs(test_stats['pct_literal'] - 50) < 5),
            ("Expression-to-split mapping documented", True),  # Will be saved
            ("All sentences preserved",
             train_stats['sentences'] + dev_stats['sentences'] + test_stats['sentences'] == len(self.df)),
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
            print(f"   • Dev: {dev_stats['sentences']:,} sentences ({dev_stats['expressions']} expressions)")
            print(f"   • Test: {test_stats['sentences']:,} sentences ({test_stats['expressions']} expressions)")
            print(f"   • Zero data leakage - expressions never overlap!")
            print(f"   • Ready for Mission 2.6: Data Preparation Testing")
        else:
            print("\n⚠️  Some criteria not met. Please review above.")

        results['mission_complete'] = all_passed
        return results


def main():
    """Main function to execute Mission 2.5"""

    splitter = ExpressionBasedSplitter()
    results = splitter.run_mission_2_5(n_dev=6)

    return splitter, results


if __name__ == "__main__":
    splitter, results = main()
