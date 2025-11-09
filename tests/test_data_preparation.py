"""
Unit Tests for Data Preparation Module
Mission 2.6: Data Preparation Testing

This test suite covers:
- Mission 2.1: Dataset loading and inspection
- Mission 2.2: Label distribution validation
- Mission 2.3: IOB2 tags validation
- Mission 2.4: Dataset statistics analysis
- Mission 2.5: Placeholder tests for splitting (to be completed when Mission 2.5 is done)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_preperation import DatasetLoader  # Note: typo in filename


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def data_path():
    """Return path to the test dataset"""
    project_root = Path(__file__).parent.parent
    return project_root / "data" / "expressions_data_tagged.csv"


@pytest.fixture(scope="session")
def loader(data_path):
    """Create a DatasetLoader instance and load the dataset"""
    loader = DatasetLoader(data_path=str(data_path))
    loader.load_dataset()
    return loader


@pytest.fixture(scope="session")
def loaded_df(loader):
    """Return the loaded DataFrame"""
    return loader.df


# ============================================================================
# MISSION 2.1: DATASET LOADING AND INSPECTION
# ============================================================================

class TestMission21DatasetLoading:
    """Tests for Mission 2.1: Dataset Loading and Inspection"""

    def test_dataset_file_exists(self, data_path):
        """Test that the dataset file exists"""
        assert data_path.exists(), f"Dataset file not found at {data_path}"

    def test_dataset_loads_successfully(self, loader):
        """Test that dataset loads without errors"""
        assert loader.df is not None, "Dataset failed to load"
        assert isinstance(loader.df, pd.DataFrame), "Loaded data is not a DataFrame"

    def test_dataset_has_correct_shape(self, loaded_df):
        """Test that dataset has expected number of rows (4,800)"""
        assert len(loaded_df) == 4800, f"Expected 4,800 rows, got {len(loaded_df)}"
        assert len(loaded_df.columns) >= 16, f"Expected at least 16 columns, got {len(loaded_df.columns)}"

    def test_required_columns_present(self, loaded_df):
        """Test that all required columns from PRD Section 2.2 are present"""
        required_columns = [
            'id', 'split', 'language', 'source', 'text', 'expression',
            'matched_expression', 'span_start', 'span_end',
            'token_span_start', 'token_span_end', 'num_tokens',
            'label', 'label_2', 'iob2_tags', 'char_mask'
        ]

        missing_columns = set(required_columns) - set(loaded_df.columns)
        assert len(missing_columns) == 0, f"Missing required columns: {missing_columns}"

    def test_no_critical_missing_values(self, loaded_df):
        """Test that critical fields have no missing values"""
        critical_fields = ['text', 'expression', 'num_tokens', 'label', 'label_2']

        for field in critical_fields:
            missing_count = loaded_df[field].isna().sum()
            assert missing_count == 0, f"Critical field '{field}' has {missing_count} missing values"

    def test_no_duplicate_ids(self, loaded_df):
        """Test that there are no duplicate IDs"""
        duplicate_count = loaded_df['id'].duplicated().sum()
        assert duplicate_count == 0, f"Found {duplicate_count} duplicate IDs"

    def test_schema_validation(self, loader):
        """Test that schema matches PRD Section 2.2"""
        assert loader.verify_schema() is True, "Schema validation failed"

    def test_basic_statistics_generation(self, loader):
        """Test that basic statistics can be generated"""
        stats = loader.display_basic_statistics()

        assert 'total_rows' in stats, "Missing 'total_rows' in statistics"
        assert stats['total_rows'] == 4800, f"Expected 4800 rows, got {stats['total_rows']}"
        assert 'total_columns' in stats, "Missing 'total_columns' in statistics"


# ============================================================================
# MISSION 2.2: LABEL DISTRIBUTION VALIDATION
# ============================================================================

class TestMission22LabelDistribution:
    """Tests for Mission 2.2: Label Distribution Validation"""

    def test_label_column_exists(self, loaded_df):
        """Test that label columns exist"""
        assert 'label' in loaded_df.columns, "Column 'label' not found"
        assert 'label_2' in loaded_df.columns, "Column 'label_2' not found"

    def test_label_values_are_valid(self, loaded_df):
        """Test that label column contains only valid values"""
        valid_labels = {'מילולי', 'פיגורטיבי'}
        unique_labels = set(loaded_df['label'].unique())

        assert unique_labels.issubset(valid_labels), \
            f"Invalid labels found: {unique_labels - valid_labels}"

    def test_label_2_values_are_valid(self, loaded_df):
        """Test that label_2 column contains only 0 or 1"""
        valid_label_2 = {0, 1}
        unique_label_2 = set(loaded_df['label_2'].unique())

        assert unique_label_2.issubset(valid_label_2), \
            f"Invalid label_2 values found: {unique_label_2 - valid_label_2}"

    def test_label_distribution_balanced(self, loaded_df):
        """Test that dataset has balanced labels (50/50 split)"""
        label_counts = loaded_df['label'].value_counts()

        literal_count = label_counts.get('מילולי', 0)
        figurative_count = label_counts.get('פיגורטיבי', 0)

        assert literal_count == 2400, f"Expected 2400 literal samples, got {literal_count}"
        assert figurative_count == 2400, f"Expected 2400 figurative samples, got {figurative_count}"

    def test_label_consistency(self, loaded_df):
        """Test that label and label_2 are consistent"""
        # label = 'מילולי' should correspond to label_2 = 0
        # label = 'פיגורטיבי' should correspond to label_2 = 1

        literal_mask = loaded_df['label'] == 'מילולי'
        figurative_mask = loaded_df['label'] == 'פיגורטיבי'

        literal_label_2_ok = (loaded_df[literal_mask]['label_2'] == 0).all()
        figurative_label_2_ok = (loaded_df[figurative_mask]['label_2'] == 1).all()

        assert literal_label_2_ok, "label='מילולי' does not match label_2=0"
        assert figurative_label_2_ok, "label='פיגורטיבי' does not match label_2=1"

    def test_label_distribution_visualization_created(self, loader):
        """Test that label distribution visualization can be created"""
        figures_dir = Path(__file__).parent.parent / "paper" / "figures"

        # Run Mission 2.2 to create visualization
        results = loader.run_mission_2_2()

        # Check that visualization file was created
        viz_path = figures_dir / "label_distribution.png"
        assert viz_path.exists(), f"Label distribution visualization not created at {viz_path}"


# ============================================================================
# MISSION 2.3: IOB2 TAGS VALIDATION
# ============================================================================

class TestMission23IOB2Validation:
    """Tests for Mission 2.3: IOB2 Tags Validation"""

    def test_iob2_tags_column_exists(self, loaded_df):
        """Test that iob2_tags column exists"""
        assert 'iob2_tags' in loaded_df.columns, "Column 'iob2_tags' not found"

    def test_iob2_tags_are_strings(self, loaded_df):
        """Test that iob2_tags are strings (space-separated)"""
        non_na_tags = loaded_df['iob2_tags'].dropna()

        for idx, tags in non_na_tags.head(10).items():
            assert isinstance(tags, str), f"Row {idx}: iob2_tags is not a string"

    def test_iob2_tags_have_valid_labels(self, loaded_df):
        """Test that all IOB2 tags are valid (O, B-IDIOM, I-IDIOM)"""
        valid_tags = {'O', 'B-IDIOM', 'I-IDIOM'}

        for idx, tags_str in loaded_df['iob2_tags'].dropna().head(100).items():
            tags = tags_str.split()
            unique_tags = set(tags)

            invalid_tags = unique_tags - valid_tags
            assert len(invalid_tags) == 0, \
                f"Row {idx}: Invalid tags found: {invalid_tags}"

    def test_iob2_tags_count_matches_num_tokens(self, loaded_df):
        """Test that number of IOB2 tags matches num_tokens"""
        mismatches = []

        for idx, row in loaded_df.head(100).iterrows():
            if pd.isna(row['iob2_tags']):
                continue

            tags = row['iob2_tags'].split()
            num_tags = len(tags)
            expected_tokens = int(row['num_tokens'])

            if num_tags != expected_tokens:
                mismatches.append((idx, num_tags, expected_tokens))

        assert len(mismatches) == 0, \
            f"Found {len(mismatches)} tag-count mismatches. Examples: {mismatches[:5]}"

    def test_iob2_sequence_validity(self, loaded_df):
        """Test that IOB2 sequences are valid (no I-IDIOM without B-IDIOM)"""
        sequence_errors = []

        for idx, tags_str in loaded_df['iob2_tags'].dropna().head(100).items():
            tags = tags_str.split()

            for i, tag in enumerate(tags):
                if tag == 'I-IDIOM':
                    # I-IDIOM must be preceded by B-IDIOM or I-IDIOM
                    if i == 0 or tags[i - 1] not in ['B-IDIOM', 'I-IDIOM']:
                        sequence_errors.append(idx)
                        break

        assert len(sequence_errors) == 0, \
            f"Found {len(sequence_errors)} sequence errors. Rows: {sequence_errors[:5]}"

    def test_token_span_indices_valid(self, loaded_df):
        """Test that token_span_start and token_span_end are valid indices"""
        span_errors = []

        for idx, row in loaded_df.head(100).iterrows():
            if pd.isna(row['token_span_start']) or pd.isna(row['token_span_end']):
                continue

            start = int(row['token_span_start'])
            end = int(row['token_span_end'])
            num_tokens = int(row['num_tokens'])

            # Check: 0 <= start < end <= num_tokens
            if not (0 <= start < end <= num_tokens):
                span_errors.append((idx, start, end, num_tokens))

        assert len(span_errors) == 0, \
            f"Found {len(span_errors)} invalid token spans. Examples: {span_errors[:5]}"

    def test_token_span_matches_iob2_tags(self, loaded_df):
        """Test that token_span indices match B-IDIOM and I-IDIOM positions"""
        mismatch_errors = []

        for idx, row in loaded_df.head(100).iterrows():
            if pd.isna(row['iob2_tags']) or pd.isna(row['token_span_start']):
                continue

            tags = row['iob2_tags'].split()
            start = int(row['token_span_start'])
            end = int(row['token_span_end'])

            # Tags in range [start:end) should start with B-IDIOM
            if tags[start] != 'B-IDIOM':
                mismatch_errors.append((idx, 'First tag not B-IDIOM'))
                continue

            # All tags in [start:end) should be B-IDIOM or I-IDIOM
            span_tags = tags[start:end]
            if not all(t in ('B-IDIOM', 'I-IDIOM') for t in span_tags):
                mismatch_errors.append((idx, 'Non-idiom tags in span'))

        assert len(mismatch_errors) == 0, \
            f"Found {len(mismatch_errors)} span-tag mismatches. Examples: {mismatch_errors[:5]}"

    def test_iob2_verification_function(self, loader):
        """Test that IOB2 verification function works"""
        results = loader.verify_iob2_tags()

        assert 'alignment_rate' in results, "Missing 'alignment_rate' in results"
        assert 'valid_tags_rate' in results, "Missing 'valid_tags_rate' in results"
        assert 'valid_sequence_rate' in results, "Missing 'valid_sequence_rate' in results"

        # Expect high alignment rates (>95%)
        assert results['alignment_rate'] >= 95.0, \
            f"Low alignment rate: {results['alignment_rate']:.2f}%"


# ============================================================================
# MISSION 2.4: DATASET STATISTICS ANALYSIS
# ============================================================================

class TestMission24Statistics:
    """Tests for Mission 2.4: Dataset Statistics Analysis"""

    def test_unique_expressions_count(self, loaded_df):
        """Test that there are 60-80 unique idiom expressions"""
        unique_expr = loaded_df['expression'].nunique()
        assert 60 <= unique_expr <= 80, \
            f"Expected 60-80 unique expressions, got {unique_expr}"

    def test_average_sentence_length(self, loaded_df):
        """Test that average sentence length is reasonable (~12.5 tokens)"""
        avg_length = loaded_df['num_tokens'].mean()
        assert 10 <= avg_length <= 20, \
            f"Expected avg sentence length 10-20 tokens, got {avg_length:.2f}"

    def test_average_idiom_length(self, loaded_df):
        """Test that average idiom length is reasonable (~3.2 tokens)"""
        idiom_lengths = (loaded_df['token_span_end'] - loaded_df['token_span_start']).dropna()
        avg_idiom_length = idiom_lengths.mean()

        assert 2.0 <= avg_idiom_length <= 4.0, \
            f"Expected avg idiom length 2-4 tokens, got {avg_idiom_length:.2f}"

    def test_statistics_generation_function(self, loader):
        """Test that generate_statistics function works"""
        stats = loader.generate_statistics()

        # Check required keys
        required_keys = [
            'total_sentences',
            'unique_expressions',
            'avg_sentence_length',
            'avg_idiom_length'
        ]

        for key in required_keys:
            assert key in stats, f"Missing key '{key}' in statistics"

        # Check values
        assert stats['total_sentences'] == 4800, \
            f"Expected 4800 sentences, got {stats['total_sentences']}"
        assert 60 <= stats['unique_expressions'] <= 80, \
            f"Expected 60-80 expressions, got {stats['unique_expressions']}"

    def test_sentence_type_analysis(self, loader):
        """Test that sentence type analysis works"""
        results = loader.analyze_sentence_types()

        assert 'type_counts' in results, "Missing 'type_counts' in results"
        assert 'type_percentages' in results, "Missing 'type_percentages' in results"

        # Check that at least some sentence types are detected
        assert len(results['type_counts']) > 0, "No sentence types detected"

    def test_visualizations_creation(self, loader):
        """Test that all required visualizations can be created"""
        figures_dir = Path(__file__).parent.parent / "paper" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Create visualizations
        loader.create_visualizations()

        # Check that all required visualizations exist
        required_figs = [
            "label_distribution.png",
            "sentence_length_distribution.png",
            "idiom_length_distribution.png",
            "top_10_idioms.png",
            "sentence_type_distribution.png",
            "sentence_type_by_label.png"
        ]

        for fig_name in required_figs:
            fig_path = figures_dir / fig_name
            assert fig_path.exists(), f"Visualization '{fig_name}' not created"

    def test_statistics_file_saved(self, loader):
        """Test that statistics can be saved to file"""
        stats_dir = Path(__file__).parent.parent / "experiments" / "results"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # Run Mission 2.4 which saves statistics
        results = loader.run_mission_2_4()

        # Check that statistics file was created
        stats_file = stats_dir / "dataset_statistics.txt"
        assert stats_file.exists(), f"Statistics file not created at {stats_file}"


# ============================================================================
# MISSION 2.5: PLACEHOLDER TESTS (TO BE COMPLETED)
# ============================================================================

class TestMission25Splitting:
    """Tests for Mission 2.5: Expression-Based Dataset Splitting"""

    def test_splitting_module_exists(self):
        """Test that dataset_splitting module exists"""
        try:
            from dataset_splitting import ExpressionBasedSplitter
            assert True
        except ImportError:
            pytest.fail("dataset_splitting module not found")

    def test_splitter_can_be_instantiated(self):
        """Test that ExpressionBasedSplitter can be instantiated"""
        from dataset_splitting import ExpressionBasedSplitter

        splitter = ExpressionBasedSplitter()
        assert splitter is not None

    def test_split_data_file_exists(self):
        """Test that split data file exists"""
        project_root = Path(__file__).parent.parent
        split_file = project_root / "data" / "expressions_data_with_splits.csv"

        assert split_file.exists(), f"Split data file not found at {split_file}"

    def test_expression_based_split_no_data_leakage(self):
        """Test that expression-based splitting has no data leakage"""
        project_root = Path(__file__).parent.parent
        split_file = project_root / "data" / "expressions_data_with_splits.csv"

        if not split_file.exists():
            pytest.skip("Split data file not created yet")

        df = pd.read_csv(split_file, encoding='utf-8-sig')

        # Check that split column exists
        assert 'split' in df.columns, "Split column not found"

        # Get expressions in each split
        train_expr = set(df[df['split'] == 'train']['expression'].unique())
        dev_expr = set(df[df['split'] == 'dev']['expression'].unique())
        test_expr = set(df[df['split'] == 'test']['expression'].unique())

        # CRITICAL: No expression should appear in multiple splits
        assert len(train_expr & dev_expr) == 0, \
            f"Expression overlap between train and dev: {train_expr & dev_expr}"
        assert len(train_expr & test_expr) == 0, \
            f"Expression overlap between train and test: {train_expr & test_expr}"
        assert len(dev_expr & test_expr) == 0, \
            f"Expression overlap between dev and test: {dev_expr & test_expr}"

        # Verify splits are balanced (within 10% of 50/50)
        for split_name in ['train', 'dev', 'test']:
            split_df = df[df['split'] == split_name]
            literal_pct = (split_df['label_2'] == 0).sum() / len(split_df) * 100

            # Allow some flexibility for test set
            margin = 15 if split_name == 'test' else 10
            assert abs(literal_pct - 50) < margin, \
                f"{split_name} set imbalanced: {literal_pct:.1f}% literal"

    def test_test_set_expressions(self):
        """Test that test set contains expected expressions"""
        project_root = Path(__file__).parent.parent
        split_file = project_root / "data" / "expressions_data_with_splits.csv"

        if not split_file.exists():
            pytest.skip("Split data file not created yet")

        df = pd.read_csv(split_file, encoding='utf-8-sig')
        test_df = df[df['split'] == 'test']

        # Test set should have exactly 6 expressions
        test_expressions = test_df['expression'].nunique()
        assert test_expressions == 6, \
            f"Test set should have 6 expressions, got {test_expressions}"

        # Test set should have both literal and figurative examples
        literal_count = (test_df['label_2'] == 0).sum()
        figurative_count = (test_df['label_2'] == 1).sum()

        assert literal_count > 0, "Test set has no literal examples"
        assert figurative_count > 0, "Test set has no figurative examples"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete pipeline"""

    def test_complete_mission_2_1_pipeline(self, loader):
        """Test that Mission 2.1 can be completed end-to-end"""
        results = loader.run_mission_2_1()

        assert 'mission_complete' in results, "Missing 'mission_complete' flag"
        assert results['mission_complete'] is True, "Mission 2.1 not completed successfully"

    def test_complete_mission_2_2_pipeline(self, loader):
        """Test that Mission 2.2 can be completed end-to-end"""
        results = loader.run_mission_2_2()

        assert 'mission_complete' in results, "Missing 'mission_complete' flag"
        assert results['mission_complete'] is True, "Mission 2.2 not completed successfully"

    def test_complete_mission_2_3_pipeline(self, loader):
        """Test that Mission 2.3 can be completed end-to-end"""
        results = loader.run_mission_2_3()

        assert 'mission_complete' in results, "Missing 'mission_complete' flag"
        assert results['mission_complete'] is True, "Mission 2.3 not completed successfully"

    def test_complete_mission_2_4_pipeline(self, loader):
        """Test that Mission 2.4 can be completed end-to-end"""
        results = loader.run_mission_2_4()

        assert 'mission_complete' in results, "Missing 'mission_complete' flag"
        assert results['mission_complete'] is True, "Mission 2.4 not completed successfully"

    def test_data_can_be_saved(self, loader):
        """Test that processed dataset can be saved"""
        output_path = Path(__file__).parent.parent / "data" / "test_processed_data.csv"

        # Save dataset
        loader.save_processed_dataset(output_path=str(output_path))

        # Verify file created
        assert output_path.exists(), "Processed dataset file not created"

        # Cleanup
        output_path.unlink()

    def test_complete_mission_2_5_pipeline(self):
        """Test that Mission 2.5 splitting completed successfully"""
        project_root = Path(__file__).parent.parent
        split_file = project_root / "data" / "expressions_data_with_splits.csv"

        if not split_file.exists():
            pytest.skip("Mission 2.5 not completed yet")

        # Verify split file can be loaded
        df = pd.read_csv(split_file, encoding='utf-8-sig')

        # Verify all splits exist
        assert 'split' in df.columns, "Split column not found"
        splits = set(df['split'].unique())
        assert 'train' in splits, "Train split not found"
        assert 'dev' in splits or 'validation' in splits, "Dev/validation split not found"
        assert 'test' in splits, "Test split not found"

        # Verify total count matches original
        assert len(df) == 4800, f"Expected 4800 rows, got {len(df)}"


# ============================================================================
# HELPER FUNCTIONS FOR FUTURE USE
# ============================================================================

def enable_mission_2_5_tests():
    """
    Helper function to enable Mission 2.5 tests once ready.

    Instructions:
    1. Remove @pytest.mark.skip decorators from TestMission25SplittingPlaceholder
    2. Implement the test logic in each placeholder test
    3. Run: pytest tests/test_data_preparation.py::TestMission25SplittingPlaceholder -v
    """
    pass


# ============================================================================
# MAIN EXECUTION (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
