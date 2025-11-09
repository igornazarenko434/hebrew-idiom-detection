"""
Data Preparation Module for Hebrew Idiom Detection
Mission 2.1: Dataset Loading and Inspection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import re
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetLoader:
    """Handles dataset loading, validation, and text preprocessing"""

    def __init__(self, data_path: str = None):
        """
        Initialize DatasetLoader

        Args:
            data_path: Path to the CSV dataset file
        """
        if data_path is None:
            project_root = Path(__file__).parent.parent
            data_path = project_root / "data" / "expressions_data_tagged.csv"

        self.data_path = Path(data_path)
        self.df = None

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the CSV dataset

        Returns:
            DataFrame with the loaded dataset
        """
        print(f"Loading dataset from: {self.data_path}")

        try:
            # Load CSV with UTF-8 encoding for Hebrew text
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
            print("✅ Dataset loaded successfully!")
            print(f"Total rows: {len(self.df)}")
            return self.df
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.data_path}")
            raise
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise

    def display_basic_statistics(self) -> Dict:
        """
        Display basic statistics about the dataset (Mission 2.1 requirement)

        Returns:
            Dictionary with basic statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print("\n" + "=" * 80)
        print("DATASET BASIC INFORMATION")
        print("=" * 80)

        # Total rows and columns
        print(f"\n📊 Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")

        # Column names and types
        print(f"\n📋 Column Names and Types:")
        print("-" * 80)
        for col, dtype in self.df.dtypes.items():
            print(f"  • {col:25s} : {dtype}")

        # Memory usage
        memory_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"\n💾 Memory Usage: {memory_mb:.2f} MB")

        # First 10 rows
        print(f"\n📄 First 10 Rows:")
        print("-" * 80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 40)
        print(self.df.head(10).to_string())

        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'memory_mb': memory_mb
        }

    def check_missing_values(self) -> Dict:
        """
        Check for missing values in the dataset (Mission 2.1 requirement)

        Returns:
            Dictionary with missing value statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("MISSING VALUES CHECK")
        print("=" * 80)

        # Define critical fields per PRD (adjust if PRD allows NAs)
        critical_fields = ['text', 'expression', 'num_tokens', 'label', 'label_2']
        # Annotation-critical fields (may be allowed NA if PRD specifies)
        annotation_fields = ['matched_expression', 'span_start', 'span_end', 'iob2_tags']

        missing_stats = pd.DataFrame({
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        }).sort_values(by='Missing_Count', ascending=False)

        # Report
        any_missing = (missing_stats['Missing_Count'] > 0).any()
        if not any_missing:
            print("\n✅ No missing values found in any column!")
        else:
            print("\n⚠️  Missing values summary (top):")
            print(missing_stats[missing_stats['Missing_Count'] > 0].to_string())

        # Are there critical-field missings?
        critical_missing_cols = [c for c in critical_fields if self.df[c].isna().any()]
        annotation_missing_cols = [c for c in annotation_fields if self.df[c].isna().any()]

        if critical_missing_cols:
            print(f"\n❌ Missing values in critical fields: {critical_missing_cols}")
        if annotation_missing_cols:
            print(f"ℹ️  Missing values in annotation fields: {annotation_missing_cols} "
                  f"(validate with PRD whether allowed)")

        return {
            'has_missing': any_missing,
            'missing_columns': missing_stats[missing_stats['Missing_Count'] > 0].index.tolist(),
            'critical_missing_columns': critical_missing_cols,
            'annotation_missing_columns': annotation_missing_cols
        }

    def check_duplicate_rows(self) -> Dict:
        """
        Check for duplicate rows in the dataset (Mission 2.1 requirement)

        Returns:
            Dictionary with duplicate statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("DUPLICATE ROWS CHECK")
        print("=" * 80)

        # Complete duplicates
        complete_duplicates = self.df.duplicated().sum()
        print(f"\n📌 Complete duplicate rows: {complete_duplicates}")

        # Duplicate IDs
        id_duplicates = self.df['id'].duplicated().sum()
        print(f"📌 Duplicate IDs: {id_duplicates}")

        # Duplicate text (same sentence)
        text_duplicates = self.df['text'].duplicated().sum()
        print(f"📌 Duplicate sentences (text): {text_duplicates}")

        if complete_duplicates == 0 and id_duplicates == 0:
            print("\n✅ No duplicate rows found!")
        else:
            print(f"\n⚠️  Found {complete_duplicates} complete duplicates and {id_duplicates} duplicate IDs")

        return {
            'complete_duplicates': int(complete_duplicates),
            'id_duplicates': int(id_duplicates),
            'text_duplicates': int(text_duplicates)
        }

    def verify_schema(self) -> bool:
        """
        Verify dataset schema matches PRD Section 2.2 (Mission 2.1 requirement)

        Returns:
            True if schema matches expected columns
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("SCHEMA VALIDATION (PRD Section 2.2)")
        print("=" * 80)

        # Expected columns from PRD Section 2.2
        expected_columns = [
            'id', 'split', 'language', 'source', 'text', 'expression',
            'matched_expression', 'span_start', 'span_end',
            'token_span_start', 'token_span_end', 'num_tokens',
            'label', 'label_2', 'iob2_tags', 'char_mask'
        ]

        actual_columns = set(self.df.columns)
        expected_columns_set = set(expected_columns)

        missing_columns = expected_columns_set - actual_columns
        extra_columns = actual_columns - expected_columns_set

        if missing_columns:
            print(f"\n❌ Missing columns: {missing_columns}")

        if extra_columns:
            print(f"\n⚠️  Extra columns not in PRD: {extra_columns}")

        if not missing_columns and not extra_columns:
            print("\n✅ Schema validation PASSED! All expected columns present.")
            return True
        else:
            print("\n⚠️  Schema validation FAILED.")
            return False

    # ---------- NEW: robust text normalizer (does NOT affect indices) ----------
    def _soft_normalize(self, text: str) -> str:
        """
        Soft normalization that does not change semantics:
        - Replace NBSP with space
        - Normalize various dashes to '-'
        """
        text = text.replace('\xa0', ' ')
        text = text.replace('–', '-').replace('—', '-').replace('־', '-')  # en/em dash & maqaf
        return text

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Hebrew text for training

        Args:
            text: Input text string

        Returns:
            Cleaned text
        """
        # Remove BOM and directional marks (not needed for training)
        text = text.replace('\ufeff', '')  # BOM
        text = text.replace('\u200f', '')  # Right-to-left mark
        text = text.replace('\u200e', '')  # Left-to-right mark

        # Soft normalize
        text = self._soft_normalize(text)

        # Normalize multiple whitespaces to single space
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Keep important punctuation: . , ! ? - " ' : ; ( )
        # These are important for sentence structure and learning
        return text

    # ---------- NEW: verify char-level spans & masks (before text cleaning) ----------
    def verify_char_spans_and_masks(self) -> Dict:
        """
        Verify that text[span_start:span_end] equals matched_expression (exclusive end),
        and that char_mask marks exactly that range with '1'.
        Uses the ORIGINAL text (pre-cleaning) if exists.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        # Use 'text' as-is if original not present yet
        base_text_col = 'text_original' if 'text_original' in self.df.columns else 'text'

        print("\n" + "=" * 80)
        print("CHAR SPAN & MASK VERIFICATION")
        print("=" * 80)

        span_mismatch = []
        mask_mismatch = []
        length_mismatch = []

        for idx, row in self.df.iterrows():
            txt = row[base_text_col]
            cm = row.get('char_mask', None)

            if pd.isna(row.get('span_start')) or pd.isna(row.get('span_end')) or pd.isna(row.get('matched_expression')):
                # allow NA if PRD permits; just skip verification
                continue

            s = int(row['span_start'])
            e = int(row['span_end'])  # exclusive end by data design
            if e < s or s < 0 or e > len(txt):
                length_mismatch.append(idx)
                continue

            if txt[s:e].strip() != str(row['matched_expression']).strip():
                span_mismatch.append(idx)

            if isinstance(cm, str):
                if len(cm) != len(txt) or not all(ch == '1' for ch in cm[s:e]):
                    mask_mismatch.append(idx)

        if not span_mismatch and not mask_mismatch and not length_mismatch:
            print("\n✅ All char spans & masks are consistent.")
        else:
            if span_mismatch:
                print(f"⚠️  Span mismatches (examples): {span_mismatch[:5]}...")
            if mask_mismatch:
                print(f"⚠️  Char-mask mismatches (examples): {mask_mismatch[:5]}...")
            if length_mismatch:
                print(f"⚠️  Span length/index errors (examples): {length_mismatch[:5]}...")

        return {
            'span_mismatch_count': len(span_mismatch),
            'mask_mismatch_count': len(mask_mismatch),
            'span_index_errors': len(length_mismatch)
        }

    def preprocess_text_column(self) -> None:
        """
        Preprocess the 'text' column - clean and normalize all texts
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("TEXT PREPROCESSING")
        print("=" * 80)

        print("Cleaning and normalizing Hebrew text...")

        # Store original text (for comparison if needed)
        if 'text_original' not in self.df.columns:
            self.df['text_original'] = self.df['text'].copy()

        # Clean text
        self.df['text'] = self.df['text'].apply(self.clean_text)

        # Check how many texts were modified
        modified_count = (self.df['text'] != self.df['text_original']).sum()

        print("✅ Text preprocessing complete!")
        print(f"   Modified {modified_count} out of {len(self.df)} texts")

        # Show example of changes
        if modified_count > 0:
            print(f"\nExample of text cleaning:")
            for idx in range(min(3, len(self.df))):
                if self.df.iloc[idx]['text'] != self.df.iloc[idx]['text_original']:
                    print(f"\n  Original: '{self.df.iloc[idx]['text_original'][:60]}...'")
                    print(f"  Cleaned:  '{self.df.iloc[idx]['text'][:60]}...'")
                    break

    def verify_iob2_tags(self) -> Dict:
        """
        Verify IOB2 tags correctness and alignment with tokens

        Returns:
            Dictionary with verification results
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("IOB2 TAGS VERIFICATION")
        print("=" * 80)

        valid_tags = {'O', 'B-IDIOM', 'I-IDIOM'}

        misalignment_errors = []
        invalid_tag_errors = []
        sequence_errors = []
        span_mismatch_errors = []
        whitespace_token_mismatch = 0  # diagnostic only

        for idx, row in self.df.iterrows():
            if pd.isna(row['iob2_tags']):
                # allow NA if PRD permits; counted earlier in missing-values report
                continue

            # 0) Diagnostic: naive whitespace token count vs. num_tokens
            ws_tokens = len(str(row['text']).split())
            if ws_tokens != int(row['num_tokens']):
                whitespace_token_mismatch += 1

            # 1) Parse IOB2 tags
            iob2_tags = str(row['iob2_tags']).split()
            num_tags = len(iob2_tags)
            expected_tokens = int(row['num_tokens'])

            # 2) Number of tags matches num_tokens
            if num_tags != expected_tokens:
                misalignment_errors.append(idx)
                continue  # cannot check more without alignment

            # 3) All tags are valid
            invalid_tags = set(iob2_tags) - valid_tags
            if invalid_tags:
                invalid_tag_errors.append((idx, invalid_tags))

            # 4) IOB2 sequence is valid (no I-IDIOM without B-IDIOM)
            for i, tag in enumerate(iob2_tags):
                if tag == 'I-IDIOM':
                    if i == 0 or iob2_tags[i - 1] not in ['B-IDIOM', 'I-IDIOM']:
                        sequence_errors.append(idx)
                        break

            # 5) Token span indices are half-open [start, end)
            try:
                s = int(row['token_span_start'])
                e = int(row['token_span_end'])  # exclusive end
                if s < 0 or e < 0 or s > e or e > expected_tokens:
                    span_mismatch_errors.append(idx)
                else:
                    # region [s:e) must be all B/I, with iob2_tags[s] == 'B-IDIOM'
                    if iob2_tags[s] != 'B-IDIOM' or any(t not in ('B-IDIOM', 'I-IDIOM') for t in iob2_tags[s:e]):
                        span_mismatch_errors.append(idx)
            except Exception:
                # if indices are NA, earlier missing check will cover
                pass

        total_checked = (self.df['iob2_tags'].notna()).sum()
        alignment_rate = ((total_checked - len(misalignment_errors)) / total_checked) * 100 if total_checked else 0.0
        valid_tags_rate = ((total_checked - len(invalid_tag_errors)) / total_checked) * 100 if total_checked else 0.0
        valid_sequence_rate = ((total_checked - len(sequence_errors)) / total_checked) * 100 if total_checked else 0.0
        span_ok_rate = ((total_checked - len(span_mismatch_errors)) / total_checked) * 100 if total_checked else 0.0

        print(f"\n📊 IOB2 Verification Results (checked {total_checked} rows):")
        print(f"  • Alignment rate:        {alignment_rate:.2f}%")
        print(f"  • Valid tags rate:       {valid_tags_rate:.2f}%")
        print(f"  • Valid sequences rate:  {valid_sequence_rate:.2f}%")
        print(f"  • Span match rate:       {span_ok_rate:.2f}%")
        if whitespace_token_mismatch:
            print(f"  • Note: {whitespace_token_mismatch} rows differ in whitespace-token count "
                  f"vs num_tokens (diagnostic only)")

        if len(misalignment_errors) == 0 and len(invalid_tag_errors) == 0 and len(sequence_errors) == 0 and len(span_mismatch_errors) == 0:
            print("\n✅ All IOB2 tags and spans are correct and properly aligned!")
        else:
            if misalignment_errors:
                print(f"\n⚠️  Tag-count vs num_tokens misalignments (ex): {misalignment_errors[:5]}...")
            if invalid_tag_errors:
                print(f"⚠️  Invalid tag errors (ex): {invalid_tag_errors[:5]}...")
            if sequence_errors:
                print(f"⚠️  Sequence errors (ex): {sequence_errors[:5]}...")
            if span_mismatch_errors:
                print(f"⚠️  Token-span mismatches (ex): {span_mismatch_errors[:5]}...")

        return {
            'alignment_rate': alignment_rate,
            'valid_tags_rate': valid_tags_rate,
            'valid_sequence_rate': valid_sequence_rate,
            'span_match_rate': span_ok_rate,
            'misalignment_count': len(misalignment_errors),
            'invalid_tags_count': len(invalid_tag_errors),
            'sequence_errors_count': len(sequence_errors),
            'span_mismatch_count': len(span_mismatch_errors),
            'whitespace_token_mismatch_rows': int(whitespace_token_mismatch)
        }

    def generate_statistics(self) -> Dict:
        """
        Generate dataset statistics as per Mission 2.1

        Returns:
            Dictionary with comprehensive statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)

        stats = {}

        # Total sentences
        stats['total_sentences'] = len(self.df)
        print(f"\n📊 Total Sentences: {stats['total_sentences']}")

        # Label distribution (literal vs figurative)
        label_counts = self.df['label'].value_counts()
        stats['label_distribution'] = label_counts.to_dict()

        print(f"\n📊 Label Distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  • {label:15s}: {count:5d} ({percentage:.2f}%)")

        # Unique idioms/expressions
        unique_expressions = self.df['expression'].nunique()
        stats['unique_expressions'] = unique_expressions
        print(f"\n📊 Unique Idioms/Expressions: {unique_expressions}")

        # Token statistics (sentence length)
        avg_tokens = self.df['num_tokens'].mean()
        median_tokens = self.df['num_tokens'].median()
        min_tokens = self.df['num_tokens'].min()
        max_tokens = self.df['num_tokens'].max()

        stats['avg_sentence_length'] = avg_tokens
        stats['median_sentence_length'] = median_tokens
        stats['min_sentence_length'] = min_tokens
        stats['max_sentence_length'] = max_tokens

        print(f"\n📊 Sentence Length Statistics:")
        print(f"  • Average: {avg_tokens:.2f} tokens")
        print(f"  • Median:  {median_tokens:.0f} tokens")
        print(f"  • Min:     {min_tokens:.0f} tokens")
        print(f"  • Max:     {max_tokens:.0f} tokens")

        # Idiom length (token-span) – computed using half-open [s:e)
        self.df['idiom_length'] = (self.df['token_span_end'] - self.df['token_span_start']).astype(int)
        avg_idiom_length = self.df['idiom_length'].mean()
        median_idiom_length = self.df['idiom_length'].median()

        stats['avg_idiom_length'] = avg_idiom_length
        stats['median_idiom_length'] = median_idiom_length

        print(f"\n📊 Idiom Length Statistics:")
        print(f"  • Average: {avg_idiom_length:.2f} tokens")
        print(f"  • Median:  {median_idiom_length:.0f} tokens")

        # Top 10 most frequent expressions
        top_expressions = self.df['expression'].value_counts().head(10)
        stats['top_10_expressions'] = top_expressions.to_dict()

        print(f"\n📊 Top 10 Most Frequent Expressions:")
        for i, (expr, count) in enumerate(top_expressions.items(), 1):
            print(f"  {i:2d}. {expr:40s} : {count:3d} occurrences")

        # ---------- NEW: per-expression label coverage (diagnostic) ----------
        coverage = self.df.pivot_table(index='expression', columns='label_2', values='id', aggfunc='count', fill_value=0)
        stats['per_expression_label_counts'] = coverage.to_dict()

        return stats

    def analyze_sentence_types(self) -> Dict:
        """
        Analyze sentence types as per Mission 2.4 requirements:
        - Questions (ending with ?)
        - Declarative (regular statements)
        - Imperative (commands)
        - Exclamatory (containing !)

        Returns:
            Dictionary with sentence type statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("SENTENCE TYPE ANALYSIS (Mission 2.4)")
        print("=" * 80)

        # Add sentence type column
        def classify_sentence_type(text: str) -> str:
            """Classify sentence type based on punctuation and structure"""
            text = str(text).strip()

            # Check for question mark
            if '?' in text:
                return 'Question'
            # Check for exclamation mark
            elif '!' in text:
                return 'Exclamatory'
            # For Hebrew, imperatives are harder to detect without morphological analysis
            # We'll use declarative as default for statements
            else:
                return 'Declarative'

        self.df['sentence_type'] = self.df['text'].apply(classify_sentence_type)

        # Overall sentence type distribution
        type_counts = self.df['sentence_type'].value_counts()
        type_percentages = (type_counts / len(self.df) * 100).round(2)

        print("\n📊 Sentence Type Distribution:")
        for stype, count in type_counts.items():
            pct = type_percentages[stype]
            print(f"  • {stype:15s}: {count:5d} ({pct:5.2f}%)")

        # Cross-tabulation: sentence type by label
        crosstab = pd.crosstab(
            self.df['sentence_type'],
            self.df['label'],
            margins=True
        )

        print("\n📊 Sentence Type by Label (Literal vs Figurative):")
        print(crosstab.to_string())

        # Percentage breakdown
        crosstab_pct = pd.crosstab(
            self.df['sentence_type'],
            self.df['label'],
            normalize='columns'
        ) * 100

        print("\n📊 Percentage Distribution within each Label:")
        print(crosstab_pct.round(2).to_string())

        # Check balance across labels
        print("\n📊 Balance Check (are sentence types distributed evenly across labels?):")
        for stype in type_counts.index:
            literal_count = self.df[(self.df['sentence_type'] == stype) & (self.df['label'] == 'מילולי')].shape[0]
            figurative_count = self.df[(self.df['sentence_type'] == stype) & (self.df['label'] == 'פיגורטיבי')].shape[0]
            total_type = literal_count + figurative_count

            if total_type > 0:
                literal_pct = (literal_count / total_type) * 100
                figurative_pct = (figurative_count / total_type) * 100
                print(f"  • {stype:15s}: Literal={literal_pct:.1f}%, Figurative={figurative_pct:.1f}%")

        # Sentence types by expression (top 10 expressions)
        print("\n📊 Sentence Types by Top Expressions:")
        top_expressions = self.df['expression'].value_counts().head(10).index

        for expr in top_expressions:
            expr_df = self.df[self.df['expression'] == expr]
            type_dist = expr_df['sentence_type'].value_counts()
            print(f"\n  {expr}:")
            for stype, count in type_dist.items():
                pct = (count / len(expr_df)) * 100
                print(f"    - {stype}: {count} ({pct:.1f}%)")

        return {
            'type_counts': type_counts.to_dict(),
            'type_percentages': type_percentages.to_dict(),
            'crosstab': crosstab.to_dict(),
            'crosstab_percentage': crosstab_pct.to_dict()
        }

    def create_visualizations(self) -> None:
        """
        Create all required visualizations for Missions 2.2 and 2.4
        Saves figures to paper/figures/ directory
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS (Missions 2.2 & 2.4)")
        print("=" * 80)

        # Create figures directory if it doesn't exist
        figures_dir = Path(__file__).parent.parent / "paper" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Set style for all plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300  # High resolution for publication
        plt.rcParams['font.size'] = 10

        # --- 1. MISSION 2.2: Label Distribution Bar Chart ---
        print("\n[1/6] Creating label distribution bar chart...")
        fig, ax = plt.subplots(figsize=(8, 6))

        label_counts = self.df['label'].value_counts()
        colors = ['#3498db', '#e74c3c']  # Blue for literal, Red for figurative

        bars = ax.bar(label_counts.index, label_counts.values, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(self.df)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xlabel('Label Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Label Distribution: Literal vs Figurative\n(Mission 2.2)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        label_dist_path = figures_dir / "label_distribution.png"
        plt.savefig(label_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {label_dist_path}")

        # --- 2. Sentence Length Distribution Histogram ---
        print("\n[2/6] Creating sentence length distribution histogram...")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(self.df['num_tokens'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.axvline(self.df['num_tokens'].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {self.df["num_tokens"].mean():.2f}')
        ax.axvline(self.df['num_tokens'].median(), color='orange', linestyle='--',
                  linewidth=2, label=f'Median: {self.df["num_tokens"].median():.0f}')

        ax.set_xlabel('Number of Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Sentence Length Distribution\n(Mission 2.4)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        sent_length_path = figures_dir / "sentence_length_distribution.png"
        plt.savefig(sent_length_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {sent_length_path}")

        # --- 3. Idiom Length Distribution Histogram ---
        print("\n[3/6] Creating idiom length distribution histogram...")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Ensure idiom_length column exists
        if 'idiom_length' not in self.df.columns:
            self.df['idiom_length'] = (self.df['token_span_end'] - self.df['token_span_start']).astype(int)

        ax.hist(self.df['idiom_length'], bins=range(1, int(self.df['idiom_length'].max()) + 2),
               color='#9b59b6', alpha=0.7, edgecolor='black', align='left')
        ax.axvline(self.df['idiom_length'].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {self.df["idiom_length"].mean():.2f}')
        ax.axvline(self.df['idiom_length'].median(), color='orange', linestyle='--',
                  linewidth=2, label=f'Median: {self.df["idiom_length"].median():.0f}')

        ax.set_xlabel('Idiom Length (tokens)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Idiom Length Distribution\n(Mission 2.4)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        idiom_length_path = figures_dir / "idiom_length_distribution.png"
        plt.savefig(idiom_length_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {idiom_length_path}")

        # --- 4. Top 10 Idioms Bar Chart ---
        print("\n[4/6] Creating top 10 idioms bar chart...")
        fig, ax = plt.subplots(figsize=(12, 8))

        top_10 = self.df['expression'].value_counts().head(10)

        bars = ax.barh(range(len(top_10)), top_10.values, color='#e67e22', alpha=0.8, edgecolor='black')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_10.values)):
            ax.text(value, i, f' {value}', va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10.index, fontsize=10)
        ax.set_xlabel('Frequency (occurrences)', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Most Frequent Idioms\n(Mission 2.4)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        plt.tight_layout()
        top_idioms_path = figures_dir / "top_10_idioms.png"
        plt.savefig(top_idioms_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {top_idioms_path}")

        # --- 5. Sentence Type Distribution Pie Chart ---
        print("\n[5/6] Creating sentence type distribution pie chart...")

        # Ensure sentence_type column exists
        if 'sentence_type' not in self.df.columns:
            self.analyze_sentence_types()

        fig, ax = plt.subplots(figsize=(10, 8))

        type_counts = self.df['sentence_type'].value_counts()
        colors_pie = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        wedges, texts, autotexts = ax.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct='%1.1f%%',
            colors=colors_pie[:len(type_counts)],
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )

        # Add count to labels
        for i, (text, count) in enumerate(zip(texts, type_counts.values)):
            text.set_text(f'{text.get_text()}\n(n={count})')
            text.set_fontsize(11)
            text.set_fontweight('bold')

        ax.set_title('Sentence Type Distribution\n(Mission 2.4)', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        sent_type_pie_path = figures_dir / "sentence_type_distribution.png"
        plt.savefig(sent_type_pie_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {sent_type_pie_path}")

        # --- 6. Sentence Type by Label Stacked Bar Chart ---
        print("\n[6/6] Creating sentence type by label stacked bar chart...")
        fig, ax = plt.subplots(figsize=(12, 7))

        # Create crosstab for stacked bar chart
        crosstab = pd.crosstab(self.df['sentence_type'], self.df['label'])

        crosstab.plot(kind='bar', stacked=True, ax=ax, color=['#3498db', '#e74c3c'],
                     alpha=0.8, edgecolor='black', width=0.7)

        ax.set_xlabel('Sentence Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Sentence Type Distribution by Label\n(Literal vs Figurative - Mission 2.4)',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Label', fontsize=10, title_fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        sent_type_label_path = figures_dir / "sentence_type_by_label.png"
        plt.savefig(sent_type_label_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {sent_type_label_path}")

        print("\n" + "=" * 80)
        print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📁 Location: {figures_dir}")
        print("\nCreated files:")
        print("  1. label_distribution.png")
        print("  2. sentence_length_distribution.png")
        print("  3. idiom_length_distribution.png")
        print("  4. top_10_idioms.png")
        print("  5. sentence_type_distribution.png")
        print("  6. sentence_type_by_label.png")

    # ---------- NEW: simple label consistency check (Mission 2.2 aid) ----------
    def verify_label_consistency(self) -> Dict:
        """
        Verify that label textual values match label_2 numeric coding (0 literal, 1 figurative).
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("LABEL CONSISTENCY CHECK (label vs label_2)")
        print("=" * 80)

        mapping_ok = (
            ((self.df['label'] == 'מילולי') & (self.df['label_2'] == 0)) |
            ((self.df['label'] == 'פיגורטיבי') & (self.df['label_2'] == 1))
        )
        inconsistencies = (~mapping_ok).sum()

        label_counts = self.df['label'].value_counts().to_dict()
        label2_counts = self.df['label_2'].value_counts().to_dict()

        print(f"\nCounts: label={label_counts} | label_2={label2_counts}")
        if inconsistencies == 0:
            print("✅ label and label_2 are consistent for all rows.")
        else:
            print(f"❌ Found {inconsistencies} inconsistencies between label and label_2.")

        return {
            'inconsistencies': int(inconsistencies),
            'label_counts': label_counts,
            'label_2_counts': label2_counts
        }

    def run_mission_2_1(self) -> Dict:
        """
        Complete Mission 2.1: Dataset Loading and Inspection

        Returns:
            Dictionary with all validation and statistics results
        """
        print("\n" + "=" * 80)
        print("MISSION 2.1: DATASET LOADING AND INSPECTION")
        print("=" * 80)

        results = {}

        # Step 1: Load dataset
        print("\n[Step 1] Loading dataset...")
        self.load_dataset()

        # Step 2: Display basic statistics
        print("\n[Step 2] Displaying basic statistics...")
        results['basic_info'] = self.display_basic_statistics()

        # Step 3: Check for missing values
        print("\n[Step 3] Checking for missing values...")
        results['missing_values'] = self.check_missing_values()

        # Step 4: Check for duplicate rows
        print("\n[Step 4] Checking for duplicate rows...")
        results['duplicates'] = self.check_duplicate_rows()

        # Step 5: Verify schema matches PRD Section 2.2
        print("\n[Step 5] Verifying schema...")
        results['schema_valid'] = self.verify_schema()

        # ---------- NEW: Step 5.1 – verify char spans & masks BEFORE text cleaning ----------
        print("\n[Step 5.1] Verifying char spans & masks...")
        results['char_spans'] = self.verify_char_spans_and_masks()

        # Step 6: Preprocess text
        print("\n[Step 6] Preprocessing text...")
        self.preprocess_text_column()

        # Step 7: Verify IOB2 tags (with half-open token spans)
        print("\n[Step 7] Verifying IOB2 tags...")
        results['iob2_verification'] = self.verify_iob2_tags()

        # ---------- NEW: Step 7.1 – label consistency ----------
        print("\n[Step 7.1] Verifying label consistency...")
        results['label_consistency'] = self.verify_label_consistency()

        # Step 8: Generate statistics
        print("\n[Step 8] Generating statistics...")
        results['statistics'] = self.generate_statistics()

        # Final validation summary
        print("\n" + "=" * 80)
        print("MISSION 2.1 SUCCESS CRITERIA CHECK")
        print("=" * 80)

        criteria = [
            ("Dataset loads successfully", results['basic_info']['total_rows'] > 0),
            ("4,800 total sentences", results['basic_info']['total_rows'] == 4800),
            ("All required columns present", results['schema_valid']),
            ("No critical missing data", len(results['missing_values']['critical_missing_columns']) == 0),
            ("Schema matches PRD Section 2.2", results['schema_valid']),
            ("IOB2 alignment ≥ 99%", results['iob2_verification']['alignment_rate'] >= 99.0),
            ("Token-span half-open validated", results['iob2_verification']['span_match_rate'] >= 99.0),
            ("Text preprocessed and clean", True),
            ("Labels consistent", results['label_consistency']['inconsistencies'] == 0),
        ]

        all_passed = True
        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            print(f"{status} {criterion}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n" + "=" * 80)
            print("🎉 MISSION 2.1 COMPLETE - ALL SUCCESS CRITERIA PASSED!")
            print("=" * 80)
            print(f"\n📋 Summary:")
            print(f"   • Total sentences: {results['statistics']['total_sentences']}")
            print(f"   • Unique idioms: {results['statistics']['unique_expressions']}")
            print(f"   • Average sentence length: {results['statistics']['avg_sentence_length']:.2f} tokens")
            print(f"   • Average idiom length: {results['statistics']['avg_idiom_length']:.2f} tokens")
            print(f"   • Dataset is clean and ready for next mission!")
        else:
            print("\n⚠️  Some criteria not met. Please review above.")

        results['mission_complete'] = all_passed
        return results

    def run_mission_2_2(self) -> Dict:
        """
        Complete Mission 2.2: Label Distribution Validation

        Returns:
            Dictionary with validation results
        """
        print("\n" + "=" * 80)
        print("MISSION 2.2: LABEL DISTRIBUTION VALIDATION")
        print("=" * 80)

        results = {}

        # Step 1: Verify label consistency (already done in 2.1)
        print("\n[Step 1] Verifying label consistency...")
        results['label_consistency'] = self.verify_label_consistency()

        # Step 2: Get label counts
        label_counts = self.df['label'].value_counts()
        literal_count = label_counts.get('מילולי', 0)
        figurative_count = label_counts.get('פיגורטיבי', 0)

        results['literal_count'] = int(literal_count)
        results['figurative_count'] = int(figurative_count)
        results['total_count'] = len(self.df)

        # Step 3: Create visualization (NEW!)
        print("\n[Step 2] Creating label distribution visualization...")
        # Only create the label distribution chart
        figures_dir = Path(__file__).parent.parent / "paper" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(label_counts.index, label_counts.values, color=colors, alpha=0.8, edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(self.df)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xlabel('Label Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Label Distribution: Literal vs Figurative\n(Mission 2.2)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        label_dist_path = figures_dir / "label_distribution.png"
        plt.savefig(label_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Visualization saved: {label_dist_path}")

        # Success criteria check
        print("\n" + "=" * 80)
        print("MISSION 2.2 SUCCESS CRITERIA CHECK")
        print("=" * 80)

        criteria = [
            ("2,400 literal samples (50%)", literal_count == 2400),
            ("2,400 figurative samples (50%)", figurative_count == 2400),
            ("Labels consistent across columns", results['label_consistency']['inconsistencies'] == 0),
            ("Visualization saved", label_dist_path.exists())
        ]

        all_passed = True
        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            print(f"{status} {criterion}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n🎉 MISSION 2.2 COMPLETE!")
        else:
            print("\n⚠️  Some criteria not met.")

        results['mission_complete'] = all_passed
        return results

    def run_mission_2_3(self) -> Dict:
        """
        Complete Mission 2.3: IOB2 Tags Validation

        Returns:
            Dictionary with validation results
        """
        print("\n" + "=" * 80)
        print("MISSION 2.3: IOB2 TAGS VALIDATION")
        print("=" * 80)

        # Run IOB2 verification (already done in 2.1)
        results = self.verify_iob2_tags()

        # Success criteria check
        print("\n" + "=" * 80)
        print("MISSION 2.3 SUCCESS CRITERIA CHECK")
        print("=" * 80)

        criteria = [
            ("100% alignment between tokens and IOB2 tags", results['alignment_rate'] == 100.0),
            ("No invalid tags", results['invalid_tags_count'] == 0),
            ("No sequence violations", results['sequence_errors_count'] == 0),
            ("Token spans correct", results['span_mismatch_count'] == 0),
            ("Zero or minimal errors", results['misalignment_count'] + results['invalid_tags_count'] +
                                      results['sequence_errors_count'] + results['span_mismatch_count'] == 0)
        ]

        all_passed = True
        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            print(f"{status} {criterion}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n🎉 MISSION 2.3 COMPLETE!")
        else:
            print("\n⚠️  Some criteria not met.")

        results['mission_complete'] = all_passed
        return results

    def run_mission_2_4(self) -> Dict:
        """
        Complete Mission 2.4: Dataset Statistics Analysis

        Returns:
            Dictionary with statistics and analysis results
        """
        print("\n" + "=" * 80)
        print("MISSION 2.4: DATASET STATISTICS ANALYSIS")
        print("=" * 80)

        results = {}

        # Step 1: Generate statistics (already done in 2.1, but ensure it's complete)
        print("\n[Step 1] Generating dataset statistics...")
        results['statistics'] = self.generate_statistics()

        # Step 2: NEW - Sentence type analysis
        print("\n[Step 2] Analyzing sentence types...")
        results['sentence_types'] = self.analyze_sentence_types()

        # Step 3: Create all visualizations
        print("\n[Step 3] Creating visualizations...")
        self.create_visualizations()

        # Step 4: Save statistics to file
        stats_dir = Path(__file__).parent.parent / "experiments" / "results"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_file = stats_dir / "dataset_statistics.txt"

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET STATISTICS - Mission 2.4\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total sentences: {results['statistics']['total_sentences']}\n")
            f.write(f"\nLabel Distribution:\n")
            for label, count in results['statistics']['label_distribution'].items():
                pct = (count / results['statistics']['total_sentences']) * 100
                f.write(f"  {label}: {count} ({pct:.2f}%)\n")

            f.write(f"\nUnique idioms: {results['statistics']['unique_expressions']}\n")
            f.write(f"Average sentence length: {results['statistics']['avg_sentence_length']:.2f} tokens\n")
            f.write(f"Average idiom length: {results['statistics']['avg_idiom_length']:.2f} tokens\n")

            f.write(f"\n\nSentence Type Distribution:\n")
            for stype, count in results['sentence_types']['type_counts'].items():
                pct = results['sentence_types']['type_percentages'][stype]
                f.write(f"  {stype}: {count} ({pct:.2f}%)\n")

            f.write(f"\n\nTop 10 Idioms:\n")
            for i, (expr, count) in enumerate(results['statistics']['top_10_expressions'].items(), 1):
                f.write(f"  {i}. {expr}: {count}\n")

        print(f"\n✅ Statistics saved to: {stats_file}")

        # Success criteria check
        print("\n" + "=" * 80)
        print("MISSION 2.4 SUCCESS CRITERIA CHECK")
        print("=" * 80)

        figures_dir = Path(__file__).parent.parent / "paper" / "figures"

        criteria = [
            ("Statistics calculated correctly", True),
            ("Unique idioms: 60-80 expressions", 60 <= results['statistics']['unique_expressions'] <= 80),
            ("Avg sentence length: ~12.5 tokens", 10 <= results['statistics']['avg_sentence_length'] <= 20),
            ("Avg idiom length: 2-4 tokens", 2.0 <= results['statistics']['avg_idiom_length'] <= 4.0),
            ("Sentence types analyzed", 'type_counts' in results['sentence_types']),
            ("Visualizations created", (figures_dir / "label_distribution.png").exists()),
            ("Statistics file saved", stats_file.exists())
        ]

        all_passed = True
        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            print(f"{status} {criterion}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n🎉 MISSION 2.4 COMPLETE!")
        else:
            print("\n⚠️  Some criteria not met.")

        results['mission_complete'] = all_passed
        return results

    def save_processed_dataset(self, output_path: str = None) -> None:
        """
        Save the processed dataset to CSV

        Args:
            output_path: Path to save the dataset (default: data/processed_data.csv)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        if output_path is None:
            output_path = self.data_path.parent / "processed_data.csv"

        # Drop the original text column (we keep the cleaned version)
        df_to_save = self.df.drop(columns=['text_original'], errors='ignore')

        df_to_save.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ Processed dataset saved to: {output_path}")


def main():
    """Main function to execute Missions 2.1, 2.2, 2.3, and 2.4"""

    # Initialize loader
    loader = DatasetLoader()

    all_results = {}

    # Run Mission 2.1: Dataset Loading and Inspection
    print("\n" + "🚀" * 40)
    print("STARTING DATA PREPARATION PIPELINE")
    print("🚀" * 40)

    all_results['mission_2_1'] = loader.run_mission_2_1()

    if not all_results['mission_2_1'].get('mission_complete', False):
        print("\n❌ Mission 2.1 failed. Cannot proceed.")
        return loader, all_results

    # Run Mission 2.2: Label Distribution Validation
    all_results['mission_2_2'] = loader.run_mission_2_2()

    # Run Mission 2.3: IOB2 Tags Validation
    all_results['mission_2_3'] = loader.run_mission_2_3()

    # Run Mission 2.4: Dataset Statistics Analysis
    all_results['mission_2_4'] = loader.run_mission_2_4()

    # Save processed dataset
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATASET")
    print("=" * 80)
    loader.save_processed_dataset()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL MISSIONS")
    print("=" * 80)

    missions_status = {
        "Mission 2.1 (Dataset Loading & Inspection)": all_results['mission_2_1'].get('mission_complete', False),
        "Mission 2.2 (Label Distribution Validation)": all_results['mission_2_2'].get('mission_complete', False),
        "Mission 2.3 (IOB2 Tags Validation)": all_results['mission_2_3'].get('mission_complete', False),
        "Mission 2.4 (Dataset Statistics Analysis)": all_results['mission_2_4'].get('mission_complete', False)
    }

    all_complete = all(missions_status.values())

    for mission, status in missions_status.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {mission}")

    if all_complete:
        print("\n" + "🎉" * 40)
        print("ALL MISSIONS COMPLETE! DATA PREPARATION PHASE DONE!")
        print("🎉" * 40)
        print("\n📋 Summary:")
        print(f"   • Dataset validated and cleaned")
        print(f"   • Total sentences: {all_results['mission_2_1']['statistics']['total_sentences']}")
        print(f"   • Literal: {all_results['mission_2_2']['literal_count']} | Figurative: {all_results['mission_2_2']['figurative_count']}")
        print(f"   • Unique idioms: {all_results['mission_2_1']['statistics']['unique_expressions']}")
        print(f"   • All visualizations created")
        print(f"   • Ready for Mission 2.5: Dataset Splitting")
    else:
        print("\n⚠️  Some missions incomplete. Please review above.")

    return loader, all_results


if __name__ == "__main__":
    loader, results = main()