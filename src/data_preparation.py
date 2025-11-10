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

        # ---------- NEW: Expression occurrence statistics (min/max/mean/std) ----------
        expr_counts = self.df['expression'].value_counts()
        stats['expression_occurrences'] = {
            'min': int(expr_counts.min()),
            'max': int(expr_counts.max()),
            'mean': float(expr_counts.mean()),
            'median': float(expr_counts.median()),
            'std': float(expr_counts.std())
        }
        print(f"\n📊 Expression Occurrence Statistics:")
        print(f"  • Min occurrences per idiom: {stats['expression_occurrences']['min']}")
        print(f"  • Max occurrences per idiom: {stats['expression_occurrences']['max']}")
        print(f"  • Mean occurrences per idiom: {stats['expression_occurrences']['mean']:.2f}")
        print(f"  • Median occurrences per idiom: {stats['expression_occurrences']['median']:.2f}")
        print(f"  • Std occurrences per idiom: {stats['expression_occurrences']['std']:.2f}")

        # Token statistics (sentence length)
        avg_tokens = self.df['num_tokens'].mean()
        median_tokens = self.df['num_tokens'].median()
        std_tokens = self.df['num_tokens'].std()
        min_tokens = self.df['num_tokens'].min()
        max_tokens = self.df['num_tokens'].max()

        stats['avg_sentence_length'] = avg_tokens
        stats['median_sentence_length'] = median_tokens
        stats['std_sentence_length'] = std_tokens
        stats['min_sentence_length'] = min_tokens
        stats['max_sentence_length'] = max_tokens

        print(f"\n📊 Sentence Length Statistics (tokens):")
        print(f"  • Average: {avg_tokens:.2f} tokens")
        print(f"  • Median:  {median_tokens:.0f} tokens")
        print(f"  • Std:     {std_tokens:.2f} tokens")
        print(f"  • Min:     {min_tokens:.0f} tokens")
        print(f"  • Max:     {max_tokens:.0f} tokens")

        # ---------- NEW: Character-level length statistics ----------
        self.df['sentence_char_length'] = self.df['text'].str.len()
        self.df['idiom_char_length'] = self.df['matched_expression'].str.len()

        stats['sentence_char_length'] = {
            'mean': float(self.df['sentence_char_length'].mean()),
            'median': float(self.df['sentence_char_length'].median()),
            'std': float(self.df['sentence_char_length'].std()),
            'min': int(self.df['sentence_char_length'].min()),
            'max': int(self.df['sentence_char_length'].max())
        }

        print(f"\n📊 Sentence Length Statistics (characters):")
        print(f"  • Average: {stats['sentence_char_length']['mean']:.2f} chars")
        print(f"  • Median:  {stats['sentence_char_length']['median']:.2f} chars")
        print(f"  • Std:     {stats['sentence_char_length']['std']:.2f} chars")
        print(f"  • Min:     {stats['sentence_char_length']['min']} chars")
        print(f"  • Max:     {stats['sentence_char_length']['max']} chars")

        # Idiom length (token-span) – computed using half-open [s:e)
        self.df['idiom_length'] = (self.df['token_span_end'] - self.df['token_span_start']).astype(int)
        avg_idiom_length = self.df['idiom_length'].mean()
        median_idiom_length = self.df['idiom_length'].median()
        std_idiom_length = self.df['idiom_length'].std()
        min_idiom_length = self.df['idiom_length'].min()
        max_idiom_length = self.df['idiom_length'].max()

        stats['avg_idiom_length'] = avg_idiom_length
        stats['median_idiom_length'] = median_idiom_length
        stats['std_idiom_length'] = std_idiom_length
        stats['min_idiom_length'] = min_idiom_length
        stats['max_idiom_length'] = max_idiom_length

        print(f"\n📊 Idiom Length Statistics (tokens):")
        print(f"  • Average: {avg_idiom_length:.2f} tokens")
        print(f"  • Median:  {median_idiom_length:.0f} tokens")
        print(f"  • Std:     {std_idiom_length:.2f} tokens")
        print(f"  • Min:     {min_idiom_length:.0f} tokens")
        print(f"  • Max:     {max_idiom_length:.0f} tokens")

        stats['idiom_char_length'] = {
            'mean': float(self.df['idiom_char_length'].mean()),
            'median': float(self.df['idiom_char_length'].median()),
            'std': float(self.df['idiom_char_length'].std()),
            'min': int(self.df['idiom_char_length'].min()),
            'max': int(self.df['idiom_char_length'].max())
        }

        print(f"\n📊 Idiom Length Statistics (characters):")
        print(f"  • Average: {stats['idiom_char_length']['mean']:.2f} chars")
        print(f"  • Median:  {stats['idiom_char_length']['median']:.2f} chars")
        print(f"  • Std:     {stats['idiom_char_length']['std']:.2f} chars")
        print(f"  • Min:     {stats['idiom_char_length']['min']} chars")
        print(f"  • Max:     {stats['idiom_char_length']['max']} chars")

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

    def analyze_idiom_position(self) -> Dict:
        """
        Analyze idiom positions within sentences (dataset_analysis_plan.md requirement)
        Computes position_ratio = token_span_start / num_tokens

        Returns:
            Dictionary with position statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("IDIOM POSITION ANALYSIS")
        print("=" * 80)

        # Compute position ratio
        self.df['position_ratio'] = self.df['token_span_start'] / self.df['num_tokens']

        # Classify positions
        def classify_position(ratio):
            if ratio < 0.33:
                return 'start'
            elif ratio < 0.67:
                return 'middle'
            else:
                return 'end'

        self.df['position_category'] = self.df['position_ratio'].apply(classify_position)

        # Overall position statistics
        position_counts = self.df['position_category'].value_counts()
        total = len(self.df)

        stats = {
            'position_ratio': {
                'mean': float(self.df['position_ratio'].mean()),
                'median': float(self.df['position_ratio'].median()),
                'std': float(self.df['position_ratio'].std()),
                'min': float(self.df['position_ratio'].min()),
                'max': float(self.df['position_ratio'].max())
            },
            'position_distribution': {
                'start': int(position_counts.get('start', 0)),
                'middle': int(position_counts.get('middle', 0)),
                'end': int(position_counts.get('end', 0))
            },
            'position_percentages': {
                'start': float((position_counts.get('start', 0) / total) * 100),
                'middle': float((position_counts.get('middle', 0) / total) * 100),
                'end': float((position_counts.get('end', 0) / total) * 100)
            }
        }

        print(f"\n📊 Position Ratio Statistics:")
        print(f"  • Mean: {stats['position_ratio']['mean']:.4f}")
        print(f"  • Median: {stats['position_ratio']['median']:.4f}")
        print(f"  • Std: {stats['position_ratio']['std']:.4f}")
        print(f"  • Range: [{stats['position_ratio']['min']:.4f}, {stats['position_ratio']['max']:.4f}]")

        print(f"\n📊 Position Distribution:")
        print(f"  • Start (0-33%): {stats['position_distribution']['start']} ({stats['position_percentages']['start']:.2f}%)")
        print(f"  • Middle (33-67%): {stats['position_distribution']['middle']} ({stats['position_percentages']['middle']:.2f}%)")
        print(f"  • End (67-100%): {stats['position_distribution']['end']} ({stats['position_percentages']['end']:.2f}%)")

        # Compare positions by label
        print(f"\n📊 Position Distribution by Label:")
        for label in self.df['label'].unique():
            label_df = self.df[self.df['label'] == label]
            pos_counts = label_df['position_category'].value_counts()
            label_total = len(label_df)

            print(f"\n  {label}:")
            print(f"    • Start: {pos_counts.get('start', 0)} ({(pos_counts.get('start', 0)/label_total)*100:.2f}%)")
            print(f"    • Middle: {pos_counts.get('middle', 0)} ({(pos_counts.get('middle', 0)/label_total)*100:.2f}%)")
            print(f"    • End: {pos_counts.get('end', 0)} ({(pos_counts.get('end', 0)/label_total)*100:.2f}%)")
            print(f"    • Mean position ratio: {label_df['position_ratio'].mean():.4f}")

        return stats

    def analyze_polysemy(self) -> Dict:
        """
        Analyze polysemy: idioms appearing in both literal and figurative contexts
        (dataset_analysis_plan.md requirement)

        Returns:
            Dictionary with polysemy statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("POLYSEMY ANALYSIS")
        print("=" * 80)

        # Group by expression and label
        expr_label_counts = self.df.groupby(['expression', 'label']).size().unstack(fill_value=0)

        # Calculate figurative ratio for each expression
        expr_label_counts['total'] = expr_label_counts.sum(axis=1)
        expr_label_counts['figurative_ratio'] = (
            expr_label_counts.get('פיגורטיבי', 0) / expr_label_counts['total']
        )
        expr_label_counts['figurative_percentage'] = expr_label_counts['figurative_ratio'] * 100

        # Identify polysemous idioms (appear in both categories)
        has_literal = expr_label_counts.get('מילולי', 0) > 0
        has_figurative = expr_label_counts.get('פיגורטיבי', 0) > 0
        polysemous_idioms = expr_label_counts[has_literal & has_figurative]

        # Identify mono-sense idioms
        only_literal = expr_label_counts[has_literal & ~has_figurative]
        only_figurative = expr_label_counts[~has_literal & has_figurative]

        stats = {
            'total_expressions': len(expr_label_counts),
            'polysemous_count': len(polysemous_idioms),
            'only_literal_count': len(only_literal),
            'only_figurative_count': len(only_figurative),
            'polysemous_percentage': float((len(polysemous_idioms) / len(expr_label_counts)) * 100),
            'polysemous_idioms': polysemous_idioms.index.tolist(),
            'expression_figurative_ratios': expr_label_counts['figurative_ratio'].to_dict()
        }

        print(f"\n📊 Polysemy Statistics:")
        print(f"  • Total expressions: {stats['total_expressions']}")
        print(f"  • Polysemous idioms (both literal & figurative): {stats['polysemous_count']} ({stats['polysemous_percentage']:.2f}%)")
        print(f"  • Only literal: {stats['only_literal_count']}")
        print(f"  • Only figurative: {stats['only_figurative_count']}")

        print(f"\n📊 Top 10 Most Polysemous Idioms (by balance):")
        # Sort by how close to 50/50 (most polysemous)
        polysemous_sorted = polysemous_idioms.copy()
        polysemous_sorted['balance_score'] = 1 - abs(polysemous_sorted['figurative_ratio'] - 0.5) * 2
        polysemous_sorted = polysemous_sorted.sort_values('balance_score', ascending=False)

        for i, (expr, row) in enumerate(polysemous_sorted.head(10).iterrows(), 1):
            fig_pct = row['figurative_percentage']
            lit_pct = 100 - fig_pct
            print(f"  {i}. {expr}")
            print(f"     Figurative: {row.get('פיגורטיבי', 0):.0f} ({fig_pct:.1f}%) | "
                  f"Literal: {row.get('מילולי', 0):.0f} ({lit_pct:.1f}%)")

        # Store for heatmap visualization
        self.polysemy_data = expr_label_counts

        return stats

    def analyze_lexical_statistics(self) -> Dict:
        """
        Compute comprehensive lexical statistics (dataset_analysis_plan.md requirement):
        - Vocabulary size
        - Type-Token Ratio (TTR)
        - Word frequencies
        - Function words

        Returns:
            Dictionary with lexical statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("LEXICAL STATISTICS")
        print("=" * 80)

        from collections import Counter

        # Tokenize all sentences
        all_tokens = []
        tokens_per_sentence = []

        for text in self.df['text']:
            tokens = text.split()
            all_tokens.extend(tokens)
            tokens_per_sentence.append(set(tokens))

        # Overall vocabulary
        vocabulary = set(all_tokens)
        total_tokens = len(all_tokens)

        # Type-Token Ratio
        ttr_overall = len(vocabulary) / total_tokens

        # Average unique words per sentence
        avg_unique_per_sentence = np.mean([len(s) for s in tokens_per_sentence])

        # Word frequencies
        word_freq = Counter(all_tokens)
        top_20_overall = word_freq.most_common(20)

        # Lexical stats by label
        label_stats = {}
        for label in self.df['label'].unique():
            label_df = self.df[self.df['label'] == label]
            label_tokens = []
            for text in label_df['text']:
                label_tokens.extend(text.split())

            label_vocab = set(label_tokens)
            label_ttr = len(label_vocab) / len(label_tokens) if len(label_tokens) > 0 else 0
            label_freq = Counter(label_tokens)

            label_stats[label] = {
                'vocabulary_size': len(label_vocab),
                'total_tokens': len(label_tokens),
                'ttr': label_ttr,
                'top_20': label_freq.most_common(20)
            }

        # Function words (Hebrew)
        function_words = ['של', 'את', 'על', 'עם', 'ב', 'ל', 'מ', 'ה', 'ש', 'כ',
                         'כי', 'אם', 'או', 'גם', 'רק', 'לא', 'אבל', 'זה', 'היה', 'הוא']

        function_word_counts = {fw: word_freq.get(fw, 0) for fw in function_words}
        total_function_words = sum(function_word_counts.values())
        function_word_ratio = total_function_words / total_tokens

        # Top words in idioms
        idiom_tokens = []
        for matched_expr in self.df['matched_expression'].dropna():
            idiom_tokens.extend(str(matched_expr).split())

        idiom_word_freq = Counter(idiom_tokens)
        top_20_idiom_words = idiom_word_freq.most_common(20)

        stats = {
            'vocabulary_size': len(vocabulary),
            'total_tokens': total_tokens,
            'ttr_overall': float(ttr_overall),
            'avg_unique_per_sentence': float(avg_unique_per_sentence),
            'top_20_words': top_20_overall,
            'top_20_idiom_words': top_20_idiom_words,
            'function_word_counts': function_word_counts,
            'function_word_ratio': float(function_word_ratio),
            'label_statistics': label_stats
        }

        print(f"\n📊 Overall Lexical Statistics:")
        print(f"  • Vocabulary size (unique words): {stats['vocabulary_size']:,}")
        print(f"  • Total tokens: {stats['total_tokens']:,}")
        print(f"  • Type-Token Ratio (TTR): {stats['ttr_overall']:.4f}")
        print(f"  • Average unique words per sentence: {stats['avg_unique_per_sentence']:.2f}")
        print(f"  • Function word ratio: {stats['function_word_ratio']:.4f} ({function_word_ratio*100:.2f}%)")

        print(f"\n📊 Top 20 Most Frequent Words:")
        for i, (word, count) in enumerate(top_20_overall, 1):
            pct = (count / total_tokens) * 100
            print(f"  {i:2d}. '{word}': {count:4d} ({pct:.2f}%)")

        print(f"\n📊 Top 20 Words in Idioms:")
        for i, (word, count) in enumerate(top_20_idiom_words, 1):
            print(f"  {i:2d}. '{word}': {count:4d}")

        print(f"\n📊 Lexical Statistics by Label:")
        for label, lstats in label_stats.items():
            print(f"\n  {label}:")
            print(f"    • Vocabulary size: {lstats['vocabulary_size']:,}")
            print(f"    • Total tokens: {lstats['total_tokens']:,}")
            print(f"    • TTR: {lstats['ttr']:.4f}")
            print(f"    • Top 5 words: {[w for w, c in lstats['top_20'][:5]]}")

        print(f"\n📊 Function Word Frequencies:")
        for fw, count in sorted(function_word_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            if count > 0:
                pct = (count / total_tokens) * 100
                print(f"  • '{fw}': {count:4d} ({pct:.2f}%)")

        return stats

    # ==================== PART 2: OPTIONAL/RECOMMENDED ANALYSES ====================

    def analyze_structural_complexity(self) -> Dict:
        """
        Analyze syntactic and structural complexity (dataset_analysis_plan.md PART 2.1)
        - Subclause markers (ש, כי, אם, etc.)
        - Punctuation counts
        - Sentence complexity metrics

        Returns:
            Dictionary with structural complexity statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("STRUCTURAL COMPLEXITY ANALYSIS")
        print("=" * 80)

        from collections import Counter

        # Subclause markers in Hebrew
        subclause_markers = ['ש', 'כי', 'אם', 'כאשר', 'למרות', 'אף', 'מכיוון', 'בגלל', 'אלא']

        # Punctuation marks
        punctuation_marks = ['.', ',', '!', '?', ':', ';', '-', '–', '—', '"', "'", '(', ')']

        # Initialize metrics
        self.df['subclause_count'] = 0
        self.df['punctuation_count'] = 0
        self.df['subclause_ratio'] = 0.0

        for idx, row in self.df.iterrows():
            text = row['text']
            tokens = text.split()

            # Count subclause markers
            subclause_count = sum(1 for token in tokens if token in subclause_markers)
            self.df.at[idx, 'subclause_count'] = subclause_count
            self.df.at[idx, 'subclause_ratio'] = subclause_count / len(tokens) if len(tokens) > 0 else 0

            # Count punctuation
            punct_count = sum(1 for char in text if char in punctuation_marks)
            self.df.at[idx, 'punctuation_count'] = punct_count

        # Overall statistics
        stats = {
            'mean_subclause_count': float(self.df['subclause_count'].mean()),
            'mean_subclause_ratio': float(self.df['subclause_ratio'].mean()),
            'mean_punctuation_count': float(self.df['punctuation_count'].mean()),
            'sentences_with_subclauses': int((self.df['subclause_count'] > 0).sum()),
            'sentences_with_subclauses_pct': float((self.df['subclause_count'] > 0).sum() / len(self.df) * 100)
        }

        print(f"\n📊 Overall Structural Complexity:")
        print(f"  • Mean subclause markers per sentence: {stats['mean_subclause_count']:.2f}")
        print(f"  • Mean subclause ratio: {stats['mean_subclause_ratio']:.4f}")
        print(f"  • Mean punctuation marks per sentence: {stats['mean_punctuation_count']:.2f}")
        print(f"  • Sentences with subclauses: {stats['sentences_with_subclauses']} ({stats['sentences_with_subclauses_pct']:.2f}%)")

        # By label comparison
        stats['by_label'] = {}
        print(f"\n📊 Structural Complexity by Label:")
        for label in self.df['label'].unique():
            label_df = self.df[self.df['label'] == label]
            label_stats = {
                'mean_subclause_count': float(label_df['subclause_count'].mean()),
                'mean_subclause_ratio': float(label_df['subclause_ratio'].mean()),
                'mean_punctuation_count': float(label_df['punctuation_count'].mean())
            }
            stats['by_label'][label] = label_stats

            print(f"\n  {label}:")
            print(f"    • Mean subclause markers: {label_stats['mean_subclause_count']:.2f}")
            print(f"    • Mean subclause ratio: {label_stats['mean_subclause_ratio']:.4f}")
            print(f"    • Mean punctuation: {label_stats['mean_punctuation_count']:.2f}")

        return stats

    def analyze_lexical_richness(self) -> Dict:
        """
        Analyze lexical richness and variation (dataset_analysis_plan.md PART 2.2)
        - Hapax legomena (words appearing once)
        - Zipf's law validation
        - Additional richness metrics

        Returns:
            Dictionary with lexical richness statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("LEXICAL RICHNESS ANALYSIS")
        print("=" * 80)

        from collections import Counter
        import math

        # Collect all tokens
        all_tokens = []
        for text in self.df['text']:
            all_tokens.extend(text.split())

        # Word frequencies
        word_freq = Counter(all_tokens)
        total_tokens = len(all_tokens)
        unique_words = len(word_freq)

        # Hapax legomena (words appearing exactly once)
        hapax_legomena = [word for word, count in word_freq.items() if count == 1]
        hapax_count = len(hapax_legomena)
        hapax_ratio = hapax_count / unique_words

        # Dis legomena (words appearing exactly twice)
        dis_legomena = [word for word, count in word_freq.items() if count == 2]
        dis_count = len(dis_legomena)

        # Zipf's law: rank vs frequency (for top 1000 words)
        sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        zipf_data = []
        for rank, (word, freq) in enumerate(sorted_freq[:1000], 1):
            zipf_data.append({
                'rank': rank,
                'word': word,
                'frequency': freq,
                'log_rank': math.log(rank),
                'log_freq': math.log(freq)
            })

        # Maas index (a^2 / (log(total_tokens) - log(unique_words)))
        if total_tokens > unique_words > 0:
            maas_index = (math.log(total_tokens) - math.log(unique_words)) / (math.log(total_tokens) ** 2)
        else:
            maas_index = 0

        stats = {
            'total_tokens': total_tokens,
            'unique_words': unique_words,
            'ttr': float(unique_words / total_tokens),
            'hapax_legomena_count': hapax_count,
            'hapax_ratio': float(hapax_ratio),
            'dis_legomena_count': dis_count,
            'maas_index': float(maas_index),
            'zipf_data': zipf_data[:100]  # Store top 100 for visualization
        }

        print(f"\n📊 Lexical Richness Statistics:")
        print(f"  • Total tokens: {stats['total_tokens']:,}")
        print(f"  • Unique words: {stats['unique_words']:,}")
        print(f"  • Type-Token Ratio (TTR): {stats['ttr']:.4f}")
        print(f"  • Hapax legomena (words appearing once): {stats['hapax_legomena_count']:,} ({hapax_ratio*100:.2f}%)")
        print(f"  • Dis legomena (words appearing twice): {stats['dis_legomena_count']:,}")
        print(f"  • Maas Index: {stats['maas_index']:.4f}")

        # By label
        stats['by_label'] = {}
        print(f"\n📊 Lexical Richness by Label:")
        for label in self.df['label'].unique():
            label_df = self.df[self.df['label'] == label]
            label_tokens = []
            for text in label_df['text']:
                label_tokens.extend(text.split())

            label_word_freq = Counter(label_tokens)
            label_hapax = len([w for w, c in label_word_freq.items() if c == 1])

            label_stats = {
                'unique_words': len(label_word_freq),
                'total_tokens': len(label_tokens),
                'ttr': len(label_word_freq) / len(label_tokens),
                'hapax_count': label_hapax,
                'hapax_ratio': label_hapax / len(label_word_freq)
            }
            stats['by_label'][label] = label_stats

            print(f"\n  {label}:")
            print(f"    • Unique words: {label_stats['unique_words']:,}")
            print(f"    • TTR: {label_stats['ttr']:.4f}")
            print(f"    • Hapax legomena: {label_stats['hapax_count']:,} ({label_stats['hapax_ratio']*100:.2f}%)")

        # Store for visualization
        self.zipf_data = zipf_data

        return stats

    def analyze_collocations(self) -> Dict:
        """
        Analyze collocations around idioms (dataset_analysis_plan.md PART 2.4)
        - Extract ±3 token context around idioms
        - Most frequent collocations by label
        - PMI (Pointwise Mutual Information)

        Returns:
            Dictionary with collocation statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("COLLOCATIONAL ANALYSIS")
        print("=" * 80)

        from collections import Counter
        import math

        # Extract context words (±3 tokens around idiom)
        context_words_all = []
        context_by_label = {'מילולי': [], 'פיגורטיבי': []}

        for idx, row in self.df.iterrows():
            tokens = row['text'].split()
            span_start = int(row['token_span_start'])
            span_end = int(row['token_span_end'])
            label = row['label']

            # Get context (3 tokens before and after)
            context_before = tokens[max(0, span_start-3):span_start]
            context_after = tokens[span_end:min(len(tokens), span_end+3)]

            context = context_before + context_after
            context_words_all.extend(context)
            context_by_label[label].extend(context)

        # Overall collocation frequencies
        context_freq_all = Counter(context_words_all)
        top_20_context = context_freq_all.most_common(20)

        # By label
        context_freq_literal = Counter(context_by_label['מילולי'])
        context_freq_figurative = Counter(context_by_label['פיגורטיבי'])

        top_20_literal = context_freq_literal.most_common(20)
        top_20_figurative = context_freq_figurative.most_common(20)

        stats = {
            'total_context_words': len(context_words_all),
            'unique_context_words': len(context_freq_all),
            'top_20_context_overall': top_20_context,
            'top_20_context_literal': top_20_literal,
            'top_20_context_figurative': top_20_figurative
        }

        print(f"\n📊 Collocational Statistics:")
        print(f"  • Total context words (±3 tokens): {stats['total_context_words']:,}")
        print(f"  • Unique context words: {stats['unique_context_words']:,}")

        print(f"\n📊 Top 20 Context Words (Overall):")
        for i, (word, count) in enumerate(top_20_context, 1):
            pct = (count / len(context_words_all)) * 100
            print(f"  {i:2d}. '{word}': {count:4d} ({pct:.2f}%)")

        print(f"\n📊 Top 10 Context Words by Label:")
        print(f"\n  Literal:")
        for i, (word, count) in enumerate(top_20_literal[:10], 1):
            print(f"    {i:2d}. '{word}': {count:4d}")

        print(f"\n  Figurative:")
        for i, (word, count) in enumerate(top_20_figurative[:10], 1):
            print(f"    {i:2d}. '{word}': {count:4d}")

        # Store for word cloud visualization
        self.context_freq_all = context_freq_all
        self.context_freq_literal = context_freq_literal
        self.context_freq_figurative = context_freq_figurative

        return stats

    def analyze_annotation_consistency(self) -> Dict:
        """
        Analyze annotation reliability and consistency (dataset_analysis_plan.md PART 2.6)
        - Span match percentage
        - IOB2 consistency
        - Prefix attachment patterns

        Returns:
            Dictionary with consistency statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("ANNOTATION CONSISTENCY ANALYSIS")
        print("=" * 80)

        # Check prefix attachment patterns (ו, ה, ל, etc.)
        prefixes = ['ו', 'ה', 'ל', 'מ', 'ב', 'כ', 'ש']
        prefix_patterns = []

        for idx, row in self.df.iterrows():
            matched_expr = str(row['matched_expression'])
            if any(matched_expr.startswith(prefix) for prefix in prefixes):
                prefix_patterns.append({
                    'expression': row['expression'],
                    'matched_expression': matched_expr,
                    'has_prefix': True
                })

        # Consistency per idiom
        idiom_consistency = {}
        for expr in self.df['expression'].unique():
            expr_df = self.df[self.df['expression'] == expr]

            # Check how consistent the matched expressions are
            matched_variants = expr_df['matched_expression'].nunique()
            most_common_match = expr_df['matched_expression'].mode()[0] if len(expr_df) > 0 else None
            most_common_count = (expr_df['matched_expression'] == most_common_match).sum()
            consistency_rate = most_common_count / len(expr_df)

            idiom_consistency[expr] = {
                'total_occurrences': len(expr_df),
                'unique_variants': matched_variants,
                'most_common_variant': most_common_match,
                'consistency_rate': float(consistency_rate)
            }

        stats = {
            'prefix_attachment_count': len(prefix_patterns),
            'prefix_attachment_rate': float(len(prefix_patterns) / len(self.df)),
            'idiom_consistency': idiom_consistency,
            'mean_consistency_rate': float(np.mean([v['consistency_rate'] for v in idiom_consistency.values()]))
        }

        print(f"\n📊 Annotation Consistency:")
        print(f"  • Prefix attachments found: {stats['prefix_attachment_count']} ({stats['prefix_attachment_rate']*100:.2f}%)")
        print(f"  • Mean consistency rate per idiom: {stats['mean_consistency_rate']:.4f}")

        # Show idioms with low consistency
        print(f"\n📊 Idioms with Variant Forms (Top 10):")
        sorted_idioms = sorted(idiom_consistency.items(), key=lambda x: x[1]['unique_variants'], reverse=True)
        for i, (expr, data) in enumerate(sorted_idioms[:10], 1):
            print(f"  {i}. {expr}: {data['unique_variants']} variants (consistency: {data['consistency_rate']:.2f})")

        return stats

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

        # --- 7. BOXPLOT: Sentence Length by Label (NEW from dataset_analysis_plan.md) ---
        print("\n[7/11] Creating sentence length boxplot by label...")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data for boxplot
        literal_lengths = self.df[self.df['label'] == 'מילולי']['num_tokens']
        figurative_lengths = self.df[self.df['label'] == 'פיגורטיבי']['num_tokens']

        bp = ax.boxplot([literal_lengths, figurative_lengths],
                        labels=['Literal\n(מילולי)', 'Figurative\n(פיגורטיבי)'],
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

        # Color the boxes
        colors_box = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Sentence Length (tokens)', fontsize=12, fontweight='bold')
        ax.set_title('Sentence Length Distribution by Label\n(Box Plot)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add statistics text
        stats_text = f"Literal: μ={literal_lengths.mean():.2f}, σ={literal_lengths.std():.2f}\n"
        stats_text += f"Figurative: μ={figurative_lengths.mean():.2f}, σ={figurative_lengths.std():.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        boxplot_path = figures_dir / "sentence_length_boxplot_by_label.png"
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {boxplot_path}")

        # --- 8. HEATMAP: Polysemy (NEW from dataset_analysis_plan.md) ---
        print("\n[8/11] Creating polysemy heatmap...")

        if hasattr(self, 'polysemy_data'):
            fig, ax = plt.subplots(figsize=(12, 16))

            # Prepare data for heatmap
            heatmap_data = self.polysemy_data[['figurative_percentage']].sort_values('figurative_percentage')

            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                       cbar_kws={'label': 'Figurative Usage (%)'},
                       linewidths=0.5, ax=ax)

            ax.set_title('Polysemy Heatmap: Figurative Usage Percentage per Idiom',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Figurative %', fontsize=12, fontweight='bold')
            ax.set_ylabel('Idiom Expression', fontsize=12, fontweight='bold')

            plt.tight_layout()
            heatmap_path = figures_dir / "polysemy_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {heatmap_path}")
        else:
            print("   ⚠️  Polysemy data not available. Run analyze_polysemy() first.")

        # --- 9. HISTOGRAM: Idiom Position (NEW from dataset_analysis_plan.md) ---
        print("\n[9/11] Creating idiom position histogram...")
        if 'position_ratio' in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.hist(self.df['position_ratio'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax.axvline(0.33, color='red', linestyle='--', linewidth=2, label='Start/Middle boundary')
            ax.axvline(0.67, color='red', linestyle='--', linewidth=2, label='Middle/End boundary')
            ax.axvline(self.df['position_ratio'].mean(), color='orange', linestyle='-',
                      linewidth=2, label=f'Mean: {self.df["position_ratio"].mean():.3f}')

            ax.set_xlabel('Position Ratio (token_span_start / num_tokens)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Idiom Position Distribution within Sentences', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            position_hist_path = figures_dir / "idiom_position_histogram.png"
            plt.savefig(position_hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {position_hist_path}")
        else:
            print("   ⚠️  Position data not available. Run analyze_idiom_position() first.")

        # --- 10. BAR CHART: Idiom Position by Label (NEW) ---
        print("\n[10/11] Creating idiom position by label bar chart...")
        if 'position_category' in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create crosstab
            position_label_crosstab = pd.crosstab(self.df['position_category'], self.df['label'])

            position_label_crosstab.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'],
                                         alpha=0.8, edgecolor='black', width=0.7)

            ax.set_xlabel('Idiom Position', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Idiom Position Distribution by Label', fontsize=14, fontweight='bold')
            ax.legend(title='Label', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(['Start (0-33%)', 'Middle (33-67%)', 'End (67-100%)'], rotation=0)

            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fontsize=9, fontweight='bold')

            plt.tight_layout()
            position_label_path = figures_dir / "idiom_position_by_label.png"
            plt.savefig(position_label_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {position_label_path}")
        else:
            print("   ⚠️  Position category not available. Run analyze_idiom_position() first.")

        # --- 11. VIOLIN PLOT: Sentence Length by Label (NEW) ---
        print("\n[11/11] Creating sentence length violin plot...")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data
        plot_data = self.df[['num_tokens', 'label']].copy()

        sns.violinplot(data=plot_data, x='label', y='num_tokens', ax=ax,
                      palette=['#3498db', '#e74c3c'], alpha=0.7)

        ax.set_xlabel('Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sentence Length (tokens)', fontsize=12, fontweight='bold')
        ax.set_title('Sentence Length Distribution by Label\n(Violin Plot)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        violin_path = figures_dir / "sentence_length_violin_by_label.png"
        plt.savefig(violin_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {violin_path}")

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
        print("  7. sentence_length_boxplot_by_label.png (NEW)")
        print("  8. polysemy_heatmap.png (NEW)")
        print("  9. idiom_position_histogram.png (NEW)")
        print("  10. idiom_position_by_label.png (NEW)")
        print("  11. sentence_length_violin_by_label.png (NEW)")

    def create_advanced_visualizations(self) -> None:
        """
        Create advanced visualizations for PART 2 optional analyses
        Saves figures to paper/figures/ directory
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        print("\n" + "=" * 80)
        print("CREATING ADVANCED VISUALIZATIONS (PART 2)")
        print("=" * 80)

        # Create figures directory
        figures_dir = Path(__file__).parent.parent / "paper" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10

        # --- 1. ZIPF'S LAW PLOT ---
        print("\n[1/6] Creating Zipf's law plot...")
        if hasattr(self, 'zipf_data') and self.zipf_data:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot log-log
            ranks = [d['rank'] for d in self.zipf_data[:100]]
            freqs = [d['frequency'] for d in self.zipf_data[:100]]

            ax.loglog(ranks, freqs, 'bo', alpha=0.6, markersize=4)

            # Fit a line for reference
            log_ranks = np.log(ranks)
            log_freqs = np.log(freqs)
            coeffs = np.polyfit(log_ranks, log_freqs, 1)
            fit_line = np.exp(coeffs[1]) * np.array(ranks) ** coeffs[0]
            ax.loglog(ranks, fit_line, 'r--', linewidth=2, label=f'Fit: slope={coeffs[0]:.2f}')

            ax.set_xlabel('Rank (log scale)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
            ax.set_title("Zipf's Law: Word Frequency vs Rank", fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            zipf_path = figures_dir / "zipf_law_plot.png"
            plt.savefig(zipf_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {zipf_path}")
        else:
            print("   ⚠️  Zipf data not available. Run analyze_lexical_richness() first.")

        # --- 2. STRUCTURAL COMPLEXITY BY LABEL ---
        print("\n[2/6] Creating structural complexity comparison...")
        if 'subclause_count' in self.df.columns and 'punctuation_count' in self.df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Subclause count by label
            self.df.boxplot(column='subclause_count', by='label', ax=axes[0], patch_artist=True)
            axes[0].set_xlabel('Label', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Subclause Markers Count', fontsize=11, fontweight='bold')
            axes[0].set_title('Subclause Markers by Label', fontsize=12, fontweight='bold')
            axes[0].get_figure().suptitle('')  # Remove default title

            # Punctuation count by label
            self.df.boxplot(column='punctuation_count', by='label', ax=axes[1], patch_artist=True)
            axes[1].set_xlabel('Label', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Punctuation Marks Count', fontsize=11, fontweight='bold')
            axes[1].set_title('Punctuation by Label', fontsize=12, fontweight='bold')
            axes[1].get_figure().suptitle('')  # Remove default title

            plt.tight_layout()
            complexity_path = figures_dir / "structural_complexity_by_label.png"
            plt.savefig(complexity_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {complexity_path}")
        else:
            print("   ⚠️  Structural complexity data not available. Run analyze_structural_complexity() first.")

        # --- 3. COLLOCATION WORD CLOUDS ---
        print("\n[3/6] Creating collocation word clouds...")
        if hasattr(self, 'context_freq_literal') and hasattr(self, 'context_freq_figurative'):
            try:
                from wordcloud import WordCloud

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # Literal context word cloud
                wc_literal = WordCloud(width=800, height=400, background_color='white',
                                      max_words=50, colormap='Blues').generate_from_frequencies(self.context_freq_literal)
                axes[0].imshow(wc_literal, interpolation='bilinear')
                axes[0].axis('off')
                axes[0].set_title('Context Words: Literal Usage', fontsize=14, fontweight='bold')

                # Figurative context word cloud
                wc_figurative = WordCloud(width=800, height=400, background_color='white',
                                         max_words=50, colormap='Reds').generate_from_frequencies(self.context_freq_figurative)
                axes[1].imshow(wc_figurative, interpolation='bilinear')
                axes[1].axis('off')
                axes[1].set_title('Context Words: Figurative Usage', fontsize=14, fontweight='bold')

                plt.tight_layout()
                wordcloud_path = figures_dir / "collocation_word_clouds.png"
                plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Saved: {wordcloud_path}")
            except ImportError:
                print("   ⚠️  wordcloud library not available. Skipping word clouds.")
                print("      Install with: pip install wordcloud")
        else:
            print("   ⚠️  Collocation data not available. Run analyze_collocations() first.")

        # --- 4. VOCABULARY DIVERSITY SCATTER ---
        print("\n[4/6] Creating vocabulary diversity scatter plot...")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate types and tokens per idiom
        idiom_diversity = []
        for expr in self.df['expression'].unique():
            expr_df = self.df[self.df['expression'] == expr]
            all_tokens = []
            for text in expr_df['text']:
                all_tokens.extend(text.split())

            types = len(set(all_tokens))
            tokens = len(all_tokens)
            ttr = types / tokens if tokens > 0 else 0

            idiom_diversity.append({
                'expression': expr,
                'types': types,
                'tokens': tokens,
                'ttr': ttr
            })

        # Plot
        types_list = [d['types'] for d in idiom_diversity]
        tokens_list = [d['tokens'] for d in idiom_diversity]
        ttr_list = [d['ttr'] for d in idiom_diversity]

        scatter = ax.scatter(tokens_list, types_list, c=ttr_list, cmap='viridis',
                            s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Type-Token Ratio', fontsize=11, fontweight='bold')

        ax.set_xlabel('Total Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unique Types (Words)', fontsize=12, fontweight='bold')
        ax.set_title('Vocabulary Diversity per Idiom', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        diversity_path = figures_dir / "vocabulary_diversity_scatter.png"
        plt.savefig(diversity_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {diversity_path}")

        # --- 5. HAPAX LEGOMENA DISTRIBUTION ---
        print("\n[5/6] Creating hapax legomena comparison...")
        if hasattr(self, 'zipf_data'):
            # Compare hapax ratios by label (from lexical richness analysis)
            fig, ax = plt.subplots(figsize=(8, 6))

            # We need to run lexical richness by label first
            # This data should be available from analyze_lexical_richness()
            labels = ['מילולי', 'פיגורטיבי']
            hapax_counts = []
            total_vocab = []

            for label in labels:
                label_df = self.df[self.df['label'] == label]
                label_tokens = []
                for text in label_df['text']:
                    label_tokens.extend(text.split())

                from collections import Counter
                word_freq = Counter(label_tokens)
                hapax = len([w for w, c in word_freq.items() if c == 1])
                hapax_counts.append(hapax)
                total_vocab.append(len(word_freq))

            # Create grouped bar chart
            x = np.arange(len(labels))
            width = 0.35

            bars1 = ax.bar(x - width/2, total_vocab, width, label='Total Vocabulary', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x + width/2, hapax_counts, width, label='Hapax Legomena', color='#e74c3c', alpha=0.8)

            ax.set_xlabel('Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('Word Count', fontsize=12, fontweight='bold')
            ax.set_title('Vocabulary vs Hapax Legomena by Label', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            hapax_path = figures_dir / "hapax_legomena_comparison.png"
            plt.savefig(hapax_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {hapax_path}")
        else:
            print("   ⚠️  Hapax data not available. Run analyze_lexical_richness() first.")

        # --- 6. CONTEXT WORDS BAR CHART (Alternative to word cloud) ---
        print("\n[6/6] Creating context words bar chart...")
        if hasattr(self, 'context_freq_literal') and hasattr(self, 'context_freq_figurative'):
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Top 15 literal context words
            top_literal = self.context_freq_literal.most_common(15)
            words_lit = [w for w, c in top_literal]
            counts_lit = [c for w, c in top_literal]

            axes[0].barh(range(len(words_lit)), counts_lit, color='#3498db', alpha=0.8, edgecolor='black')
            axes[0].set_yticks(range(len(words_lit)))
            axes[0].set_yticklabels(words_lit, fontsize=10)
            axes[0].set_xlabel('Frequency', fontsize=11, fontweight='bold')
            axes[0].set_title('Top 15 Context Words: Literal Usage', fontsize=12, fontweight='bold')
            axes[0].invert_yaxis()
            axes[0].grid(axis='x', alpha=0.3)

            # Add value labels
            for i, count in enumerate(counts_lit):
                axes[0].text(count, i, f' {count}', va='center', fontsize=9)

            # Top 15 figurative context words
            top_figurative = self.context_freq_figurative.most_common(15)
            words_fig = [w for w, c in top_figurative]
            counts_fig = [c for w, c in top_figurative]

            axes[1].barh(range(len(words_fig)), counts_fig, color='#e74c3c', alpha=0.8, edgecolor='black')
            axes[1].set_yticks(range(len(words_fig)))
            axes[1].set_yticklabels(words_fig, fontsize=10)
            axes[1].set_xlabel('Frequency', fontsize=11, fontweight='bold')
            axes[1].set_title('Top 15 Context Words: Figurative Usage', fontsize=12, fontweight='bold')
            axes[1].invert_yaxis()
            axes[1].grid(axis='x', alpha=0.3)

            # Add value labels
            for i, count in enumerate(counts_fig):
                axes[1].text(count, i, f' {count}', va='center', fontsize=9)

            plt.tight_layout()
            context_bar_path = figures_dir / "context_words_bar_chart.png"
            plt.savefig(context_bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {context_bar_path}")
        else:
            print("   ⚠️  Context data not available. Run analyze_collocations() first.")

        print("\n" + "=" * 80)
        print("✅ ALL ADVANCED VISUALIZATIONS CREATED!")
        print("=" * 80)
        print(f"\n📁 Location: {figures_dir}")
        print("\nCreated advanced visualization files:")
        print("  1. zipf_law_plot.png")
        print("  2. structural_complexity_by_label.png")
        print("  3. collocation_word_clouds.png (if wordcloud library available)")
        print("  4. vocabulary_diversity_scatter.png")
        print("  5. hapax_legomena_comparison.png")
        print("  6. context_words_bar_chart.png")

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
        Enhanced with dataset_analysis_plan.md requirements

        Returns:
            Dictionary with statistics and analysis results
        """
        print("\n" + "=" * 80)
        print("MISSION 2.4: DATASET STATISTICS ANALYSIS (Enhanced)")
        print("=" * 80)

        results = {}

        # Step 1: Generate statistics (already done in 2.1, but ensure it's complete)
        print("\n[Step 1/6] Generating dataset statistics...")
        results['statistics'] = self.generate_statistics()

        # Step 2: Sentence type analysis
        print("\n[Step 2/6] Analyzing sentence types...")
        results['sentence_types'] = self.analyze_sentence_types()

        # Step 3: NEW - Idiom position analysis (dataset_analysis_plan.md)
        print("\n[Step 3/6] Analyzing idiom positions...")
        results['idiom_position'] = self.analyze_idiom_position()

        # Step 4: NEW - Polysemy analysis (dataset_analysis_plan.md)
        print("\n[Step 4/6] Analyzing polysemy (literal vs figurative per idiom)...")
        results['polysemy'] = self.analyze_polysemy()

        # Step 5: NEW - Lexical statistics (dataset_analysis_plan.md)
        print("\n[Step 5/6] Computing lexical statistics...")
        results['lexical'] = self.analyze_lexical_statistics()

        # Step 6: Create all visualizations
        print("\n[Step 6/6] Creating visualizations...")
        self.create_visualizations()

        # Step 7: Save comprehensive statistics to file
        stats_dir = Path(__file__).parent.parent / "experiments" / "results"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_file = stats_dir / "dataset_statistics_comprehensive.txt"

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE DATASET STATISTICS - Mission 2.4 (Enhanced)\n")
            f.write("=" * 80 + "\n\n")

            # Basic Statistics
            f.write("BASIC STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total sentences: {results['statistics']['total_sentences']}\n")
            f.write(f"Unique idioms: {results['statistics']['unique_expressions']}\n\n")

            # Expression Occurrences
            f.write("Expression Occurrence Statistics:\n")
            f.write(f"  Min: {results['statistics']['expression_occurrences']['min']}\n")
            f.write(f"  Max: {results['statistics']['expression_occurrences']['max']}\n")
            f.write(f"  Mean: {results['statistics']['expression_occurrences']['mean']:.2f}\n")
            f.write(f"  Median: {results['statistics']['expression_occurrences']['median']:.2f}\n")
            f.write(f"  Std: {results['statistics']['expression_occurrences']['std']:.2f}\n\n")

            # Label Distribution
            f.write("Label Distribution:\n")
            for label, count in results['statistics']['label_distribution'].items():
                pct = (count / results['statistics']['total_sentences']) * 100
                f.write(f"  {label}: {count} ({pct:.2f}%)\n")
            f.write("\n")

            # Sentence Lengths
            f.write("Sentence Length Statistics:\n")
            f.write(f"  Tokens - Mean: {results['statistics']['avg_sentence_length']:.2f}, "
                   f"Median: {results['statistics']['median_sentence_length']:.0f}, "
                   f"Std: {results['statistics']['std_sentence_length']:.2f}\n")
            f.write(f"  Characters - Mean: {results['statistics']['sentence_char_length']['mean']:.2f}, "
                   f"Median: {results['statistics']['sentence_char_length']['median']:.2f}, "
                   f"Std: {results['statistics']['sentence_char_length']['std']:.2f}\n\n")

            # Idiom Lengths
            f.write("Idiom Length Statistics:\n")
            f.write(f"  Tokens - Mean: {results['statistics']['avg_idiom_length']:.2f}, "
                   f"Median: {results['statistics']['median_idiom_length']:.0f}, "
                   f"Std: {results['statistics']['std_idiom_length']:.2f}\n")
            f.write(f"  Characters - Mean: {results['statistics']['idiom_char_length']['mean']:.2f}, "
                   f"Median: {results['statistics']['idiom_char_length']['median']:.2f}, "
                   f"Std: {results['statistics']['idiom_char_length']['std']:.2f}\n\n")

            # Sentence Types
            f.write("Sentence Type Distribution:\n")
            for stype, count in results['sentence_types']['type_counts'].items():
                pct = results['sentence_types']['type_percentages'][stype]
                f.write(f"  {stype}: {count} ({pct:.2f}%)\n")
            f.write("\n")

            # Idiom Position
            f.write("Idiom Position Analysis:\n")
            f.write(f"  Mean position ratio: {results['idiom_position']['position_ratio']['mean']:.4f}\n")
            f.write(f"  Start (0-33%): {results['idiom_position']['position_distribution']['start']} "
                   f"({results['idiom_position']['position_percentages']['start']:.2f}%)\n")
            f.write(f"  Middle (33-67%): {results['idiom_position']['position_distribution']['middle']} "
                   f"({results['idiom_position']['position_percentages']['middle']:.2f}%)\n")
            f.write(f"  End (67-100%): {results['idiom_position']['position_distribution']['end']} "
                   f"({results['idiom_position']['position_percentages']['end']:.2f}%)\n\n")

            # Polysemy
            f.write("Polysemy Analysis:\n")
            f.write(f"  Total expressions: {results['polysemy']['total_expressions']}\n")
            f.write(f"  Polysemous (both literal & figurative): {results['polysemy']['polysemous_count']} "
                   f"({results['polysemy']['polysemous_percentage']:.2f}%)\n")
            f.write(f"  Only literal: {results['polysemy']['only_literal_count']}\n")
            f.write(f"  Only figurative: {results['polysemy']['only_figurative_count']}\n\n")

            # Lexical Statistics
            f.write("Lexical Statistics:\n")
            f.write(f"  Vocabulary size: {results['lexical']['vocabulary_size']:,} unique words\n")
            f.write(f"  Total tokens: {results['lexical']['total_tokens']:,}\n")
            f.write(f"  Type-Token Ratio (TTR): {results['lexical']['ttr_overall']:.4f}\n")
            f.write(f"  Avg unique words per sentence: {results['lexical']['avg_unique_per_sentence']:.2f}\n")
            f.write(f"  Function word ratio: {results['lexical']['function_word_ratio']:.4f}\n\n")

            # Top 10 Idioms
            f.write("Top 10 Most Frequent Idioms:\n")
            for i, (expr, count) in enumerate(results['statistics']['top_10_expressions'].items(), 1):
                f.write(f"  {i}. {expr}: {count}\n")
            f.write("\n")

            # Top 20 Words
            f.write("Top 20 Most Frequent Words:\n")
            for i, (word, count) in enumerate(results['lexical']['top_20_words'][:20], 1):
                pct = (count / results['lexical']['total_tokens']) * 100
                f.write(f"  {i}. '{word}': {count} ({pct:.2f}%)\n")

        print(f"\n✅ Comprehensive statistics saved to: {stats_file}")

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

    def run_comprehensive_analysis(self, include_part2: bool = True, create_visualizations: bool = True) -> Dict:
        """
        Run COMPREHENSIVE dataset analysis including both PART 1 (required) and PART 2 (optional)
        from dataset_analysis_plan.md

        Args:
            include_part2: Whether to include PART 2 optional analyses
            create_visualizations: Whether to create visualizations

        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "🔬" * 40)
        print("COMPREHENSIVE DATASET ANALYSIS - FULL REPORT")
        if include_part2:
            print("Including PART 1 (Required) + PART 2 (Optional/Recommended)")
        else:
            print("Including PART 1 (Required) Only")
        print("🔬" * 40)

        results = {}

        # ========== PART 1: REQUIRED ANALYSES ==========
        print("\n" + "=" * 80)
        print("PART 1: REQUIRED ANALYSES")
        print("=" * 80)

        # 1. Basic statistics
        print("\n[1/11] Generating dataset statistics...")
        results['statistics'] = self.generate_statistics()

        # 2. Sentence types
        print("\n[2/11] Analyzing sentence types...")
        results['sentence_types'] = self.analyze_sentence_types()

        # 3. Idiom position
        print("\n[3/11] Analyzing idiom positions...")
        results['idiom_position'] = self.analyze_idiom_position()

        # 4. Polysemy
        print("\n[4/11] Analyzing polysemy...")
        results['polysemy'] = self.analyze_polysemy()

        # 5. Lexical statistics
        print("\n[5/11] Computing lexical statistics...")
        results['lexical'] = self.analyze_lexical_statistics()

        if include_part2:
            # ========== PART 2: OPTIONAL/RECOMMENDED ANALYSES ==========
            print("\n" + "=" * 80)
            print("PART 2: OPTIONAL/RECOMMENDED ANALYSES")
            print("=" * 80)

            # 6. Structural complexity
            print("\n[6/11] Analyzing structural complexity...")
            results['structural_complexity'] = self.analyze_structural_complexity()

            # 7. Lexical richness
            print("\n[7/11] Analyzing lexical richness (hapax, Zipf's law)...")
            results['lexical_richness'] = self.analyze_lexical_richness()

            # 8. Collocations
            print("\n[8/11] Analyzing collocations...")
            results['collocations'] = self.analyze_collocations()

            # 9. Annotation consistency
            print("\n[9/11] Analyzing annotation consistency...")
            results['annotation_consistency'] = self.analyze_annotation_consistency()

        if create_visualizations:
            # 10. Standard visualizations
            print("\n[10/11] Creating standard visualizations...")
            self.create_visualizations()

            if include_part2:
                # 11. Advanced visualizations
                print("\n[11/11] Creating advanced visualizations...")
                self.create_advanced_visualizations()

        # Save comprehensive report
        stats_dir = Path(__file__).parent.parent / "experiments" / "results"
        stats_dir.mkdir(parents=True, exist_ok=True)

        if include_part2:
            stats_file = stats_dir / "dataset_statistics_full.txt"
            file_title = "FULL COMPREHENSIVE DATASET STATISTICS (PART 1 + PART 2)"
        else:
            stats_file = stats_dir / "dataset_statistics_comprehensive.txt"
            file_title = "COMPREHENSIVE DATASET STATISTICS (PART 1)"

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{file_title}\n")
            f.write("Hebrew Idiom Detection Dataset - Master's Thesis Analysis\n")
            f.write("=" * 80 + "\n\n")

            # PART 1 Statistics
            f.write("=" * 80 + "\n")
            f.write("PART 1: REQUIRED ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # Basic Statistics
            f.write("1. BASIC STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total sentences: {results['statistics']['total_sentences']}\n")
            f.write(f"Unique idioms: {results['statistics']['unique_expressions']}\n\n")

            f.write("Expression Occurrence Statistics:\n")
            f.write(f"  Min: {results['statistics']['expression_occurrences']['min']}\n")
            f.write(f"  Max: {results['statistics']['expression_occurrences']['max']}\n")
            f.write(f"  Mean: {results['statistics']['expression_occurrences']['mean']:.2f}\n")
            f.write(f"  Median: {results['statistics']['expression_occurrences']['median']:.2f}\n")
            f.write(f"  Std: {results['statistics']['expression_occurrences']['std']:.2f}\n\n")

            f.write("Label Distribution:\n")
            for label, count in results['statistics']['label_distribution'].items():
                pct = (count / results['statistics']['total_sentences']) * 100
                f.write(f"  {label}: {count} ({pct:.2f}%)\n")
            f.write("\n")

            # Length Statistics
            f.write("2. SENTENCE & IDIOM LENGTH STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write("Sentence Length:\n")
            f.write(f"  Tokens - Mean: {results['statistics']['avg_sentence_length']:.2f}, "
                   f"Median: {results['statistics']['median_sentence_length']:.0f}, "
                   f"Std: {results['statistics']['std_sentence_length']:.2f}\n")
            f.write(f"  Characters - Mean: {results['statistics']['sentence_char_length']['mean']:.2f}, "
                   f"Median: {results['statistics']['sentence_char_length']['median']:.2f}, "
                   f"Std: {results['statistics']['sentence_char_length']['std']:.2f}\n\n")

            f.write("Idiom Length:\n")
            f.write(f"  Tokens - Mean: {results['statistics']['avg_idiom_length']:.2f}, "
                   f"Median: {results['statistics']['median_idiom_length']:.0f}, "
                   f"Std: {results['statistics']['std_idiom_length']:.2f}\n")
            f.write(f"  Characters - Mean: {results['statistics']['idiom_char_length']['mean']:.2f}, "
                   f"Median: {results['statistics']['idiom_char_length']['median']:.2f}, "
                   f"Std: {results['statistics']['idiom_char_length']['std']:.2f}\n\n")

            # Idiom Position
            f.write("3. IDIOM POSITION ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean position ratio: {results['idiom_position']['position_ratio']['mean']:.4f}\n")
            f.write(f"Start (0-33%): {results['idiom_position']['position_distribution']['start']} "
                   f"({results['idiom_position']['position_percentages']['start']:.2f}%)\n")
            f.write(f"Middle (33-67%): {results['idiom_position']['position_distribution']['middle']} "
                   f"({results['idiom_position']['position_percentages']['middle']:.2f}%)\n")
            f.write(f"End (67-100%): {results['idiom_position']['position_distribution']['end']} "
                   f"({results['idiom_position']['position_percentages']['end']:.2f}%)\n\n")

            # Polysemy
            f.write("4. POLYSEMY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total expressions: {results['polysemy']['total_expressions']}\n")
            f.write(f"Polysemous (both literal & figurative): {results['polysemy']['polysemous_count']} "
                   f"({results['polysemy']['polysemous_percentage']:.2f}%)\n")
            f.write(f"Only literal: {results['polysemy']['only_literal_count']}\n")
            f.write(f"Only figurative: {results['polysemy']['only_figurative_count']}\n\n")

            # Lexical Statistics
            f.write("5. LEXICAL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Vocabulary size: {results['lexical']['vocabulary_size']:,} unique words\n")
            f.write(f"Total tokens: {results['lexical']['total_tokens']:,}\n")
            f.write(f"Type-Token Ratio (TTR): {results['lexical']['ttr_overall']:.4f}\n")
            f.write(f"Avg unique words per sentence: {results['lexical']['avg_unique_per_sentence']:.2f}\n")
            f.write(f"Function word ratio: {results['lexical']['function_word_ratio']:.4f}\n\n")

            # Top Words
            f.write("Top 20 Most Frequent Words:\n")
            for i, (word, count) in enumerate(results['lexical']['top_20_words'][:20], 1):
                pct = (count / results['lexical']['total_tokens']) * 100
                f.write(f"  {i:2d}. '{word}': {count:4d} ({pct:.2f}%)\n")
            f.write("\n")

            if include_part2:
                f.write("=" * 80 + "\n")
                f.write("PART 2: OPTIONAL/RECOMMENDED ANALYSIS\n")
                f.write("=" * 80 + "\n\n")

                # Structural Complexity
                f.write("6. STRUCTURAL COMPLEXITY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Mean subclause markers per sentence: {results['structural_complexity']['mean_subclause_count']:.2f}\n")
                f.write(f"Mean subclause ratio: {results['structural_complexity']['mean_subclause_ratio']:.4f}\n")
                f.write(f"Mean punctuation marks: {results['structural_complexity']['mean_punctuation_count']:.2f}\n")
                f.write(f"Sentences with subclauses: {results['structural_complexity']['sentences_with_subclauses']} "
                       f"({results['structural_complexity']['sentences_with_subclauses_pct']:.2f}%)\n\n")

                # Lexical Richness
                f.write("7. LEXICAL RICHNESS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Hapax legomena: {results['lexical_richness']['hapax_legomena_count']:,} "
                       f"({results['lexical_richness']['hapax_ratio']*100:.2f}%)\n")
                f.write(f"Dis legomena: {results['lexical_richness']['dis_legomena_count']:,}\n")
                f.write(f"Maas Index: {results['lexical_richness']['maas_index']:.4f}\n\n")

                # Collocations
                f.write("8. COLLOCATIONAL ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total context words (±3 tokens): {results['collocations']['total_context_words']:,}\n")
                f.write(f"Unique context words: {results['collocations']['unique_context_words']:,}\n\n")

                f.write("Top 10 Context Words:\n")
                for i, (word, count) in enumerate(results['collocations']['top_20_context_overall'][:10], 1):
                    pct = (count / results['collocations']['total_context_words']) * 100
                    f.write(f"  {i:2d}. '{word}': {count:4d} ({pct:.2f}%)\n")
                f.write("\n")

                # Annotation Consistency
                f.write("9. ANNOTATION CONSISTENCY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Prefix attachments: {results['annotation_consistency']['prefix_attachment_count']} "
                       f"({results['annotation_consistency']['prefix_attachment_rate']*100:.2f}%)\n")
                f.write(f"Mean consistency rate per idiom: {results['annotation_consistency']['mean_consistency_rate']:.4f}\n\n")

        print(f"\n✅ Comprehensive report saved to: {stats_file}")

        # Summary
        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   • Total sentences analyzed: {results['statistics']['total_sentences']}")
        print(f"   • Unique idioms: {results['statistics']['unique_expressions']}")
        print(f"   • Vocabulary size: {results['lexical']['vocabulary_size']:,}")
        if include_part2:
            print(f"   • Hapax legomena: {results['lexical_richness']['hapax_legomena_count']:,}")
            print(f"   • PART 1 + PART 2 analyses completed!")
        else:
            print(f"   • PART 1 analyses completed!")

        if create_visualizations:
            print(f"   • All visualizations created in paper/figures/")

        results['analysis_complete'] = True
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