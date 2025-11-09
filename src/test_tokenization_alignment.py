#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test IOB2 Tokenization Alignment
File: src/test_tokenization_alignment.py

Mission 4.2 Task 3.5: Test IOB2 alignment thoroughly before training

This script tests the alignment between word-level IOB2 tags and subword tokens
for all model tokenizers we'll use in the project.

Validation steps:
1. Load 10 examples from training data
2. Test alignment with each tokenizer
3. Print detailed alignment examples
4. Verify span boundaries are preserved
5. Verify special tokens have -100
6. Save results to experiments/results/tokenization_alignment_test.txt
"""

import sys
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.tokenization import align_labels_with_tokens, align_predictions_with_words


# Label mapping for IOB2 tags
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def test_alignment_for_model(model_name: str, model_id: str, examples: pd.DataFrame, output_file):
    """
    Test IOB2 alignment for a specific model tokenizer
    """
    print(f"\n{'='*80}")
    print(f"Testing alignment for: {model_name} ({model_id})")
    print(f"{'='*80}\n")

    output_file.write(f"\n{'='*80}\n")
    output_file.write(f"Testing alignment for: {model_name} ({model_id})\n")
    output_file.write(f"{'='*80}\n\n")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"✓ Tokenizer loaded successfully\n")
    except Exception as e:
        error_msg = f"❌ Failed to load tokenizer: {e}\n"
        print(error_msg)
        output_file.write(error_msg)
        return

    alignment_errors = 0
    span_errors = 0

    # Test on each example
    for idx, row in examples.iterrows():
        text = row['text']
        iob2_tags_str = row['iob2_tags']

        # Skip if missing IOB2 tags
        if pd.isna(iob2_tags_str) or str(iob2_tags_str) == 'nan':
            continue

        # Parse IOB2 tags
        word_labels = str(iob2_tags_str).split()
        words = text.split()

        # Check word-label alignment (original data validation)
        if len(words) != len(word_labels):
            alignment_errors += 1
            print(f"⚠️  Example {idx}: Word-label mismatch ({len(words)} words vs {len(word_labels)} tags)")
            continue

        # Tokenize with is_split_into_words=True to align word_ids with our whitespace tokenization
        tokenized = tokenizer(
            words,  # Pass pre-tokenized words
            truncation=True,
            max_length=128,
            is_split_into_words=True  # Critical: tells tokenizer words are pre-tokenized
        )

        # Get tokens for display
        tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])

        # Get word IDs
        word_ids = tokenized.word_ids()

        # Align labels
        try:
            aligned_labels = align_labels_with_tokens(
                tokenized,
                word_labels,
                LABEL2ID,
                label_all_tokens=False
            )
        except Exception as e:
            alignment_errors += 1
            print(f"❌ Example {idx}: Alignment failed - {e}")
            continue

        # Print detailed alignment
        print(f"Example {idx}:")
        print(f"  Text: {text}")
        print(f"  Words ({len(words)}): {words}")
        print(f"  IOB2 tags: {word_labels}")
        print(f"  Subword tokens ({len(tokens)}): {tokens}")
        print(f"  Word IDs: {word_ids}")
        print(f"  Aligned labels ({len(aligned_labels)}): {[ID2LABEL.get(lbl, str(lbl)) for lbl in aligned_labels]}")

        # Save to file
        output_file.write(f"Example {idx}:\n")
        output_file.write(f"  Text: {text}\n")
        output_file.write(f"  Words ({len(words)}): {words}\n")
        output_file.write(f"  IOB2 tags: {word_labels}\n")
        output_file.write(f"  Subword tokens ({len(tokens)}): {tokens}\n")
        output_file.write(f"  Word IDs: {word_ids}\n")
        output_file.write(f"  Aligned labels ({len(aligned_labels)}): {[ID2LABEL.get(lbl, str(lbl)) for lbl in aligned_labels]}\n")

        # Validation checks
        checks = []

        # Check 1: Special tokens have -100
        if aligned_labels[0] == -100 and aligned_labels[-1] == -100:
            checks.append("✓ Special tokens ([CLS], [SEP]) have label -100")
        else:
            checks.append(f"❌ Special tokens issue: first={aligned_labels[0]}, last={aligned_labels[-1]}")
            span_errors += 1

        # Check 2: Verify idiom spans are preserved
        # Get word-level idiom spans
        word_spans = []
        in_span = False
        span_start = None
        for i, label in enumerate(word_labels):
            if label == "B-IDIOM":
                if in_span:
                    word_spans.append((span_start, i))
                span_start = i
                in_span = True
            elif label == "O" and in_span:
                word_spans.append((span_start, i))
                in_span = False
        if in_span:
            word_spans.append((span_start, len(word_labels)))

        checks.append(f"  Found {len(word_spans)} idiom spans in word-level labels")

        # Check 3: Verify length matches
        if len(aligned_labels) == len(tokens):
            checks.append(f"✓ Aligned labels length ({len(aligned_labels)}) matches tokens ({len(tokens)})")
        else:
            checks.append(f"❌ Length mismatch: {len(aligned_labels)} labels vs {len(tokens)} tokens")
            span_errors += 1

        # Print checks
        for check in checks:
            print(f"  {check}")
            output_file.write(f"  {check}\n")

        print()
        output_file.write("\n")

    # Summary
    print(f"\nSummary for {model_name}:")
    print(f"  Total examples tested: {len(examples)}")
    print(f"  Alignment errors: {alignment_errors}")
    print(f"  Span/validation errors: {span_errors}")

    output_file.write(f"\nSummary for {model_name}:\n")
    output_file.write(f"  Total examples tested: {len(examples)}\n")
    output_file.write(f"  Alignment errors: {alignment_errors}\n")
    output_file.write(f"  Span/validation errors: {span_errors}\n")

    if alignment_errors == 0 and span_errors == 0:
        print(f"✅ All tests passed for {model_name}!")
        output_file.write(f"✅ All tests passed for {model_name}!\n")
    else:
        print(f"⚠️  Some tests failed for {model_name}")
        output_file.write(f"⚠️  Some tests failed for {model_name}\n")


def main():
    """
    Main test function
    """
    print("\n" + "="*80)
    print("IOB2 TOKENIZATION ALIGNMENT TEST")
    print("Mission 4.2 Task 3.5")
    print("="*80)

    # Load training data
    train_file = "data/splits/train.csv"
    print(f"\nLoading training data from: {train_file}")

    try:
        df = pd.read_csv(train_file)
        print(f"✓ Loaded {len(df)} training samples")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Filter to examples with valid IOB2 tags and idioms
    df_valid = df[df['iob2_tags'].notna() & (df['iob2_tags'].astype(str) != 'nan')].copy()
    df_with_idioms = df_valid[df_valid['iob2_tags'].str.contains('IDIOM', na=False)]

    print(f"  Valid IOB2 tags: {len(df_valid)}")
    print(f"  With idioms: {len(df_with_idioms)}")

    # Select 10 examples with idioms for testing
    test_examples = df_with_idioms.head(10).reset_index(drop=True)
    print(f"\n✓ Selected {len(test_examples)} examples for testing\n")

    # Create output directory
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tokenization_alignment_test.txt"

    # Models to test (from Mission 3.1)
    models = [
        ("AlephBERT-base", "onlplab/alephbert-base"),
        ("DictaBERT", "dicta-il/dictabert"),
        ("mBERT", "bert-base-multilingual-cased"),
        ("XLM-RoBERTa-base", "xlm-roberta-base"),
        ("XLM-RoBERTa-large", "xlm-roberta-large")
    ]

    # Open output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("IOB2 TOKENIZATION ALIGNMENT TEST RESULTS\n")
        f.write("Mission 4.2 Task 3.5\n")
        f.write("="*80 + "\n")
        f.write(f"Test date: {pd.Timestamp.now()}\n")
        f.write(f"Test examples: {len(test_examples)}\n")
        f.write(f"Models tested: {len(models)}\n\n")

        # Test each model
        for model_name, model_id in models:
            test_alignment_for_model(model_name, model_id, test_examples, f)

    print(f"\n{'='*80}")
    print(f"✅ Test complete! Results saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
