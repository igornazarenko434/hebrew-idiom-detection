"""
Error Analysis Utilities for Hebrew Idiom Detection
Standardized error categorization for both Fine-Tuning and Prompting methods

Version: 1.0
Date: December 30, 2025
"""

from typing import List, Tuple, Optional, Dict
from collections import Counter
import pandas as pd


# ============================================================================
# SPAN DETECTION ERROR CATEGORIZATION
# ============================================================================

def get_span_indices(tags: List[str]) -> Optional[List[Tuple[int, int]]]:
    """
    Extract all span boundaries from IOB tags.

    Args:
        tags: List of IOB2 tags ['O', 'B-IDIOM', 'I-IDIOM', ...]

    Returns:
        List of (start_idx, end_idx) tuples (exclusive end), or None if no spans

    Example:
        tags = ['O', 'B-IDIOM', 'I-IDIOM', 'O', 'B-IDIOM', 'O']
        returns: [(1, 3), (4, 5)]
    """
    if not tags:
        return None

    spans = []
    start = None

    for i, tag in enumerate(tags):
        if tag == 'B-IDIOM':
            # Close previous span if exists
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == 'O' and start is not None:
            # Close current span
            spans.append((start, i))
            start = None
        # I-IDIOM continues the span, no action needed

    # Close final span if sentence ends with idiom
    if start is not None:
        spans.append((start, len(tags)))

    return spans if spans else None


def categorize_span_error(true_tags: List[str], pred_tags: List[str]) -> str:
    """
    Categorize span detection errors into standardized taxonomy.

    Error Categories:
        PERFECT: Exact match
        MISS: No span predicted when ground truth has span
        FALSE_POSITIVE: Span predicted when ground truth has no span
        PARTIAL_START: Missing beginning token(s)
        PARTIAL_END: Missing ending token(s)
        PARTIAL_BOTH: Truncated on both ends
        EXTEND_START: Extra token(s) at start
        EXTEND_END: Extra token(s) at end
        EXTEND_BOTH: Extended on both ends
        SHIFT: Span at wrong position (offset)
        WRONG_SPAN: Completely different phrase tagged
        MULTI_SPAN: Multiple spans predicted (hallucination)

    Args:
        true_tags: Ground truth IOB tags
        pred_tags: Predicted IOB tags

    Returns:
        Error category code (str)

    Example:
        >>> categorize_span_error(
        ...     ['O', 'B-IDIOM', 'I-IDIOM', 'O'],
        ...     ['O', 'B-IDIOM', 'O', 'O']
        ... )
        'PARTIAL_END'
    """
    # Validation
    if len(true_tags) != len(pred_tags):
        return "ERROR_LENGTH_MISMATCH"

    # Extract span boundaries
    true_spans = get_span_indices(true_tags)
    pred_spans = get_span_indices(pred_tags)

    # Perfect match
    if true_tags == pred_tags:
        return "PERFECT"

    # No ground truth span (literal sentence)
    if true_spans is None:
        if pred_spans is not None:
            return "FALSE_POSITIVE"
        else:
            return "PERFECT"  # Both empty

    # Ground truth has span, but nothing predicted
    if pred_spans is None:
        return "MISS"

    # Multiple spans predicted (only designed for single span GT)
    if len(pred_spans) > 1:
        return "MULTI_SPAN"

    # Assume single span for now (most common case)
    # TODO: Extend for multi-span support if needed
    true_start, true_end = true_spans[0]
    pred_start, pred_end = pred_spans[0]

    # Exact boundary match (should be caught earlier, but safety check)
    if true_start == pred_start and true_end == pred_end:
        return "PERFECT"

    # Check for overlap
    has_overlap = not (pred_end <= true_start or pred_start >= true_end)

    if not has_overlap:
        return "WRONG_SPAN"  # Completely different region

    # Analyze overlap patterns
    # Missing start (pred starts later, same end)
    if pred_start > true_start and pred_end == true_end:
        return "PARTIAL_START"

    # Missing end (same start, pred ends earlier)
    if pred_start == true_start and pred_end < true_end:
        return "PARTIAL_END"

    # Truncated both ends
    if pred_start > true_start and pred_end < true_end:
        return "PARTIAL_BOTH"

    # Extended start (pred starts earlier, same end)
    if pred_start < true_start and pred_end == true_end:
        return "EXTEND_START"

    # Extended end (same start, pred extends further)
    if pred_start == true_start and pred_end > true_end:
        return "EXTEND_END"

    # Extended both (pred longer on both sides)
    if pred_start < true_start and pred_end > true_end:
        return "EXTEND_BOTH"

    # Shifted (overlapping but misaligned boundaries)
    return "SHIFT"


# ============================================================================
# CLASSIFICATION ERROR CATEGORIZATION
# ============================================================================

def categorize_cls_error(true_label: int, pred_label: int) -> str:
    """
    Categorize classification errors.

    Error Categories:
        CORRECT: Prediction matches ground truth
        FALSE_POSITIVE: Predicted Figurative, actually Literal
        FALSE_NEGATIVE: Predicted Literal, actually Figurative

    Args:
        true_label: Ground truth label (0=Literal, 1=Figurative)
        pred_label: Predicted label (0=Literal, 1=Figurative)

    Returns:
        Error category code (str)

    Example:
        >>> categorize_cls_error(0, 1)
        'FALSE_POSITIVE'
    """
    if true_label == pred_label:
        return "CORRECT"
    elif true_label == 0 and pred_label == 1:
        return "FALSE_POSITIVE"  # Predicted Figurative, actually Literal
    elif true_label == 1 and pred_label == 0:
        return "FALSE_NEGATIVE"  # Predicted Literal, actually Figurative
    else:
        return "ERROR_INVALID_LABELS"


# ============================================================================
# BATCH ERROR ANALYSIS
# ============================================================================

def analyze_span_errors(predictions: List[Dict]) -> pd.DataFrame:
    """
    Analyze all span detection errors and return detailed DataFrame.

    Args:
        predictions: List of prediction dicts with keys:
            - 'id': str
            - 'sentence': str
            - 'true_tags': List[str]
            - 'predicted_tags': List[str]
            - 'is_correct': bool

    Returns:
        DataFrame with columns: ['id', 'sentence', 'error_type', ...]

    Example:
        >>> predictions = [
        ...     {'id': '1_lit_1', 'sentence': '...', 'true_tags': [...], 'predicted_tags': [...]}
        ... ]
        >>> df = analyze_span_errors(predictions)
        >>> print(df['error_type'].value_counts())
    """
    results = []

    for pred in predictions:
        error_type = categorize_span_error(
            pred['true_tags'],
            pred['predicted_tags']
        )

        results.append({
            'id': pred['id'],
            'sentence': pred.get('sentence', ''),
            'error_type': error_type,
            'true_tags': str(pred['true_tags']),
            'predicted_tags': str(pred['predicted_tags']),
            'is_correct': pred.get('is_correct', False)
        })

    df = pd.DataFrame(results)
    return df


def analyze_cls_errors(predictions: List[Dict]) -> pd.DataFrame:
    """
    Analyze all classification errors and return detailed DataFrame.

    Args:
        predictions: List of prediction dicts with keys:
            - 'id': str
            - 'sentence': str
            - 'true_label': int
            - 'predicted_label': int
            - 'is_correct': bool

    Returns:
        DataFrame with columns: ['id', 'sentence', 'error_type', ...]
    """
    results = []

    for pred in predictions:
        error_type = categorize_cls_error(
            pred['true_label'],
            pred['predicted_label']
        )

        results.append({
            'id': pred['id'],
            'sentence': pred.get('sentence', ''),
            'error_type': error_type,
            'true_label': pred['true_label'],
            'predicted_label': pred['predicted_label'],
            'is_correct': pred.get('is_correct', False)
        })

    df = pd.DataFrame(results)
    return df


def generate_error_summary(error_df: pd.DataFrame, task: str = 'span') -> Dict:
    """
    Generate summary statistics for errors.

    Args:
        error_df: DataFrame from analyze_span_errors() or analyze_cls_errors()
        task: 'span' or 'cls'

    Returns:
        Dictionary with error counts and percentages

    Example:
        >>> summary = generate_error_summary(df, task='span')
        >>> print(summary['error_distribution'])
    """
    total = len(error_df)
    correct = (error_df['error_type'] == 'PERFECT').sum() if task == 'span' else (error_df['error_type'] == 'CORRECT').sum()
    errors = total - correct

    error_counts = error_df['error_type'].value_counts().to_dict()

    # Calculate percentages
    error_pct = {}
    for error_type, count in error_counts.items():
        if task == 'span' and error_type == 'PERFECT':
            continue
        if task == 'cls' and error_type == 'CORRECT':
            continue
        error_pct[error_type] = {
            'count': count,
            'pct_of_errors': (count / errors * 100) if errors > 0 else 0,
            'pct_of_total': (count / total * 100)
        }

    return {
        'total_samples': total,
        'correct': correct,
        'errors': errors,
        'accuracy': (correct / total * 100) if total > 0 else 0,
        'error_distribution': error_pct
    }


def extract_error_examples(error_df: pd.DataFrame,
                           error_type: str,
                           n: int = 5,
                           seed: int = 42) -> pd.DataFrame:
    """
    Extract random examples of a specific error type.

    Args:
        error_df: DataFrame from analyze_*_errors()
        error_type: Error category code (e.g., 'PARTIAL_END')
        n: Number of examples to extract
        seed: Random seed for reproducibility

    Returns:
        DataFrame with n examples of the error type
    """
    subset = error_df[error_df['error_type'] == error_type]

    if len(subset) == 0:
        return pd.DataFrame()

    sample_size = min(n, len(subset))
    return subset.sample(n=sample_size, random_state=seed)


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_span_f1(predictions: List[Dict]) -> Dict[str, float]:
    """
    Compute Span F1 (exact match) for span detection.

    CRITICAL: This is EXACT SPAN MATCHING, not token-level F1.
    A span is correct only if start AND end match exactly.

    Args:
        predictions: List of dicts with 'true_tags' and 'predicted_tags'

    Returns:
        Dict with 'f1', 'precision', 'recall'

    Example:
        >>> preds = [
        ...     {'true_tags': ['O', 'B-IDIOM', 'I-IDIOM'], 'predicted_tags': ['O', 'B-IDIOM', 'I-IDIOM']},
        ...     {'true_tags': ['B-IDIOM', 'O'], 'predicted_tags': ['O', 'O']}
        ... ]
        >>> metrics = compute_span_f1(preds)
        >>> print(f"F1: {metrics['f1']:.4f}")
    """
    correct_spans = 0
    total_pred_spans = 0
    total_true_spans = 0

    for pred in predictions:
        true_spans = get_span_indices(pred['true_tags'])
        pred_spans = get_span_indices(pred['predicted_tags'])

        # Convert None to empty list
        true_spans = true_spans or []
        pred_spans = pred_spans or []

        # Count exact matches
        for span in pred_spans:
            if span in true_spans:
                correct_spans += 1

        total_pred_spans += len(pred_spans)
        total_true_spans += len(true_spans)

    # Calculate metrics
    precision = correct_spans / total_pred_spans if total_pred_spans > 0 else 0.0
    recall = correct_spans / total_true_spans if total_true_spans > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'correct_spans': correct_spans,
        'total_pred_spans': total_pred_spans,
        'total_true_spans': total_true_spans
    }


def compute_cls_metrics(predictions: List[Dict]) -> Dict[str, float]:
    """
    Compute classification metrics (macro F1, accuracy, etc.).

    Args:
        predictions: List of dicts with 'true_label' and 'predicted_label'

    Returns:
        Dict with 'f1', 'accuracy', 'precision', 'recall', confusion matrix
    """
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

    true_labels = [p['true_label'] for p in predictions]
    pred_labels = [p['predicted_label'] for p in predictions]

    # Compute metrics
    f1 = f1_score(true_labels, pred_labels, average='macro')
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    return {
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix_tn': int(tn),
        'confusion_matrix_fp': int(fp),
        'confusion_matrix_fn': int(fn),
        'confusion_matrix_tp': int(tp)
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Span detection error analysis
    print("=" * 80)
    print("SPAN DETECTION ERROR CATEGORIZATION EXAMPLES")
    print("=" * 80)

    test_cases = [
        # (true_tags, pred_tags, expected_error)
        (
            ['O', 'B-IDIOM', 'I-IDIOM', 'O'],
            ['O', 'B-IDIOM', 'I-IDIOM', 'O'],
            'PERFECT'
        ),
        (
            ['O', 'B-IDIOM', 'I-IDIOM', 'O'],
            ['O', 'O', 'O', 'O'],
            'MISS'
        ),
        (
            ['O', 'O', 'O', 'O'],
            ['O', 'B-IDIOM', 'O', 'O'],
            'FALSE_POSITIVE'
        ),
        (
            ['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O'],
            ['O', 'O', 'B-IDIOM', 'I-IDIOM', 'O'],
            'PARTIAL_START'
        ),
        (
            ['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O'],
            ['O', 'B-IDIOM', 'I-IDIOM', 'O', 'O'],
            'PARTIAL_END'
        ),
        (
            ['B-IDIOM', 'I-IDIOM', 'O'],
            ['B-IDIOM', 'I-IDIOM', 'I-IDIOM'],
            'EXTEND_END'
        ),
    ]

    for true_tags, pred_tags, expected in test_cases:
        result = categorize_span_error(true_tags, pred_tags)
        status = "✅" if result == expected else "❌"
        print(f"{status} Expected: {expected:20} | Got: {result:20}")
        if result != expected:
            print(f"   True: {true_tags}")
            print(f"   Pred: {pred_tags}")

    print("\n" + "=" * 80)
    print("CLASSIFICATION ERROR CATEGORIZATION EXAMPLES")
    print("=" * 80)

    cls_tests = [
        (0, 0, 'CORRECT'),
        (1, 1, 'CORRECT'),
        (0, 1, 'FP'),
        (1, 0, 'FN'),
    ]

    for true_label, pred_label, expected in cls_tests:
        result = categorize_cls_error(true_label, pred_label)
        status = "✅" if result == expected else "❌"
        print(f"{status} True: {true_label} | Pred: {pred_label} | Expected: {expected} | Got: {result}")

    print("\n" + "=" * 80)
    print("SPAN F1 CALCULATION EXAMPLE")
    print("=" * 80)

    example_predictions = [
        {'true_tags': ['O', 'B-IDIOM', 'I-IDIOM', 'O'], 'predicted_tags': ['O', 'B-IDIOM', 'I-IDIOM', 'O']},
        {'true_tags': ['B-IDIOM', 'I-IDIOM', 'O'], 'predicted_tags': ['B-IDIOM', 'O', 'O']},
        {'true_tags': ['O', 'B-IDIOM', 'O'], 'predicted_tags': ['O', 'B-IDIOM', 'O']},
    ]

    metrics = compute_span_f1(example_predictions)
    print(f"Correct Spans: {metrics['correct_spans']}")
    print(f"Total Predicted: {metrics['total_pred_spans']}")
    print(f"Total True: {metrics['total_true_spans']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Span F1: {metrics['f1']:.4f}")
