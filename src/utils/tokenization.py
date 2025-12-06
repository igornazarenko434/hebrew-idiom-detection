#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenization Utilities for Hebrew Idiom Detection
File: src/utils/tokenization.py

This module handles alignment between word-level IOB2 tags and subword tokenization.

CRITICAL: Transformer tokenizers (mBERT, XLM-R, AlephBERT, DictaBERT) split words
into subwords, but our IOB2 tags are aligned with word-level tokens (whitespace-split).
This alignment is ESSENTIAL for Task 2 (Token Classification) to work correctly.

Example:
    Word-level (dataset):
        Words:     ["הוא", "שבר", "את", "הראש"]
        IOB2 tags: ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM"]

    After mBERT tokenization (subwords):
        Subwords:  ["[CLS]", "הוא", "##ש", "##בר", "את", "##ה", "##ראש", "[SEP]"]

    Aligned labels (what we need for training):
        Labels:    [-100, 0, -100, -100, 1, -100, 2, -100]
        Where: -100 = ignored in loss calculation
               0 = "O", 1 = "B-IDIOM", 2 = "I-IDIOM"
"""

from typing import List, Dict, Optional, Tuple
import torch


def align_labels_with_tokens(
    tokenized_inputs,
    word_labels: List[str],
    label2id: Dict[str, int],
    label_all_tokens: bool = False
) -> List[int]:
    """
    Align word-level IOB2 labels with subword tokens.

    Strategy:
    - Use tokenizer's word_ids() to track which subword belongs to which word
    - First subword of each word gets the word's IOB2 label
    - Subsequent subwords of same word:
      * If label_all_tokens=True: Get same label (for some NER approaches)
      * If label_all_tokens=False: Get -100 (ignored in loss) - DEFAULT and RECOMMENDED
    - Special tokens ([CLS], [SEP], [PAD]) always get -100

    Args:
        tokenized_inputs: Output from tokenizer (single example)
                         Must have been tokenized with word_ids tracking
        word_labels: List of IOB2 label strings aligned with word-level tokens
                    Example: ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM"]
        label2id: Dictionary mapping label strings to IDs
                 Example: {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
        label_all_tokens: If True, label all subword tokens (not recommended for IOB2)
                         If False, only first subword gets label, rest get -100 (default)

    Returns:
        aligned_labels: List of label IDs for each subword token
                       -100 for tokens that should be ignored in loss

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        >>> text = "הוא שבר את הראש"
        >>> word_labels = ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM"]
        >>> label2id = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
        >>>
        >>> # Tokenize with word_ids tracking
        >>> tokenized = tokenizer(text, return_offsets_mapping=False)
        >>> aligned = align_labels_with_tokens(tokenized, word_labels, label2id)
        >>> # Result: [-100, 0, -100, -100, 1, -100, 2, -100]
        >>> #          [CLS]  הוא  ##ש  ##בר  את  ##ה ##ראש [SEP]
    """
    aligned_labels = []
    word_ids = tokenized_inputs.word_ids()  # Maps each token to its word index

    previous_word_idx = None
    for word_idx in word_ids:
        # Special tokens have word_idx = None -> always -100
        if word_idx is None:
            aligned_labels.append(-100)
        # First subword of a new word -> gets word's label
        elif word_idx != previous_word_idx:
            aligned_labels.append(label2id[word_labels[word_idx]])
        # Subsequent subwords of the same word
        else:
            if label_all_tokens:
                # Option 1: Give same label to all subwords (some NER approaches)
                aligned_labels.append(label2id[word_labels[word_idx]])
            else:
                # Option 2 (DEFAULT): Ignore subsequent subwords in loss
                # This is recommended for IOB2 tagging
                aligned_labels.append(-100)

        previous_word_idx = word_idx

    return aligned_labels


def align_predictions_with_words(
    predictions: List[int],
    word_ids: List[Optional[int]],
    ignore_label: int = -100
) -> List[int]:
    """
    Convert subword-level predictions back to word-level predictions.

    During evaluation, the model outputs predictions for each subword token.
    We need to convert these back to word-level to match our ground truth IOB2 tags.

    Strategy:
    - Group subword predictions by word_id
    - For each word, take the FIRST subword's prediction as the word's prediction
    - Ignore special tokens (word_id = None)

    Args:
        predictions: List of predicted label IDs for each subword token
                    Example: [-100, 0, -100, -100, 1, -100, 2, -100]
        word_ids: List of word indices from tokenizer.word_ids()
                 Example: [None, 0, 1, 1, 2, 3, 3, None]
        ignore_label: Label ID to ignore (default: -100)

    Returns:
        word_predictions: List of predicted label IDs for each word
                         Length matches number of words in original sentence

    Example:
        >>> predictions = [-100, 0, -100, -100, 1, -100, 2, -100]
        >>> word_ids = [None, 0, 1, 1, 2, 3, 3, None]
        >>> word_preds = align_predictions_with_words(predictions, word_ids)
        >>> # Result: [0, -100, 1, 2]
        >>> # Word 0: "הוא" -> 0 (O)
        >>> # Word 1: "שבר" -> -100 (first subword was -100, skip)
        >>> # Word 2: "את" -> 1 (B-IDIOM)
        >>> # Word 3: "הראש" -> 2 (I-IDIOM)
    """
    word_predictions = []
    previous_word_idx = None

    for word_idx, pred in zip(word_ids, predictions):
        # Skip special tokens
        if word_idx is None:
            continue

        # For each new word, take the first subword's prediction
        if word_idx != previous_word_idx:
            word_predictions.append(pred)
            previous_word_idx = word_idx
        # Skip subsequent subwords of the same word

    return word_predictions


def tokenize_and_align_labels(
    examples: Dict,
    tokenizer,
    label2id: Dict[str, int],
    text_column: str = "sentence",
    label_column: str = "iob_tags",
    max_length: int = 128,
    label_all_tokens: bool = False
) -> Dict:
    """
    Tokenize text and align IOB2 labels for batch processing with HuggingFace datasets.

    This function is designed to be used with datasets.map() for efficient batch processing.

    Args:
        examples: Batch of examples from HuggingFace dataset
                 Must have text_column and label_column
        tokenizer: HuggingFace tokenizer
        label2id: Dictionary mapping label strings to IDs
        text_column: Name of text column (default: "text")
        label_column: Name of IOB2 tags column (default: "iob2_tags")
        max_length: Maximum sequence length (default: 128)
        label_all_tokens: Whether to label all subword tokens (default: False)

    Returns:
        tokenized: Dictionary with input_ids, attention_mask, and labels

    Example usage with HuggingFace datasets:
        >>> from datasets import Dataset
        >>> from functools import partial
        >>>
        >>> dataset = Dataset.from_pandas(df)
        >>> label2id = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
        >>>
        >>> tokenize_fn = partial(
        ...     tokenize_and_align_labels,
        ...     tokenizer=tokenizer,
        ...     label2id=label2id,
        ...     max_length=128
        ... )
        >>>
        >>> tokenized_dataset = dataset.map(
        ...     tokenize_fn,
        ...     batched=True,
        ...     remove_columns=dataset.column_names
        ... )
    """
    # Tokenize texts
    tokenized_inputs = tokenizer(
        examples[text_column],
        truncation=True,
        padding=False,  # Padding will be done by data collator
        max_length=max_length,
        # Don't return offsets mapping in final output, but need word_ids
        return_offsets_mapping=False,
        is_split_into_words=False  # Our text is not pre-tokenized
    )

    all_labels = []

    # Process each example in the batch
    for i, text in enumerate(examples[text_column]):
        # Get IOB2 tags for this example
        iob2_tags_str = examples[label_column][i]

        # Skip if missing IOB2 tags
        if pd.isna(iob2_tags_str) or str(iob2_tags_str) == 'nan':
            # Create all -100 labels for this example
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            all_labels.append([-100] * len(word_ids))
            continue

        # Parse IOB2 tags (space-separated string)
        word_labels = str(iob2_tags_str).split()

        # Get word_ids for this example
        # We need to extract the tokenized inputs for this specific example
        example_input_ids = tokenized_inputs["input_ids"][i]

        # Tokenize this single example to get word_ids
        single_tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=False
        )

        # Align labels with tokens
        try:
            aligned_labels = align_labels_with_tokens(
                single_tokenized,
                word_labels,
                label2id,
                label_all_tokens=label_all_tokens
            )
            all_labels.append(aligned_labels)
        except (IndexError, KeyError) as e:
            # Handle alignment errors (e.g., mismatch between words and tags)
            print(f"Warning: Alignment error for example {i}: {e}")
            # Create all -100 labels as fallback
            word_ids = single_tokenized.word_ids()
            all_labels.append([-100] * len(word_ids))

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


# Import pandas for NaN checking
import pandas as pd
