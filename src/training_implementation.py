#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Pipeline Implementation for Mission 4.2
This file contains the complete run_training() function implementation.
Will be integrated into idiom_experiment.py
"""

# This is the complete implementation that will replace the placeholder run_training() function

def run_training(args, config: Optional[Dict[str, Any]] = None, freeze_backbone: bool = False):
    """
    Run full fine-tuning or frozen backbone training (Mission 4.2)

    This function implements complete training pipeline for both tasks:
    - Task 1: Sequence classification (literal vs figurative)
    - Task 2: Token classification (IOB2 tagging for idiom spans)

    Args:
        args: Command-line arguments
        config: Configuration dictionary from YAML file (required for training)
        freeze_backbone: If True, freeze backbone and only train classification head

    Raises:
        ValueError: If required configuration is missing
    """
    mode_name = "FROZEN BACKBONE" if freeze_backbone else "FULL FINE-TUNING"

    print("\n" + "=" * 80)
    print(f"TRAINING MODE: {mode_name}")
    print("=" * 80)

    # -------------------------
    # 1. Configuration Setup
    # -------------------------
    if config is None:
        raise ValueError("Training requires configuration file (--config)")

    # Extract configuration
    model_checkpoint = config.get('model_checkpoint', args.model_id)
    task = config.get('task', 'cls')  # cls, span, or both
    device = config.get('device', 'cpu')
    max_length = config.get('max_length', 128)

    # Training hyperparameters
    learning_rate = config.get('learning_rate', 2e-5)
    batch_size = config.get('batch_size', 16)
    num_epochs = config.get('num_epochs', 5)
    warmup_ratio = config.get('warmup_ratio', 0.1)
    weight_decay = config.get('weight_decay', 0.01)
    seed = config.get('seed', 42)

    # Data paths
    train_file = config.get('train_file', 'data/splits/train.csv')
    dev_file = config.get('dev_file', 'data/splits/validation.csv')
    test_file = config.get('test_file', 'data/splits/test.csv')

    # Output settings
    output_dir = Path(config.get('output_dir', 'experiments/results/'))
    output_dir = output_dir / mode_name.lower().replace(' ', '_') / Path(model_checkpoint).name / task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping
    early_stopping_patience = config.get('early_stopping_patience', 3)

    print(f"\n📋 Configuration:")
    print(f"  Model: {model_checkpoint}")
    print(f"  Task: {task}")
    print(f"  Device: {device}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Output: {output_dir}")
    print(f"  Freeze backbone: {freeze_backbone}")

    # -------------------------
    # 2. Load Tokenizer
    # -------------------------
    print(f"\n📦 Loading tokenizer: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print(f"✓ Tokenizer loaded")

    # -------------------------
    # 3. Load and Prepare Data
    # -------------------------
    print(f"\n📊 Loading data:")
    print(f"  Train: {train_file}")
    print(f"  Dev: {dev_file}")
    print(f"  Test: {test_file}")

    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)
    test_df = pd.read_csv(test_file)

    # Filter by split if specified
    if 'split' in train_df.columns:
        if args.split:
            train_df = train_df[train_df['split'] == args.split]
            dev_df = dev_df[dev_df['split'] == args.split]
            test_df = test_df[test_df['split'] == args.split]

    # Limit samples if specified (for testing)
    if hasattr(args, 'max_samples') and args.max_samples:
        print(f"  ⚠️  Limiting to {args.max_samples} samples for testing")
        train_df = train_df.head(args.max_samples)
        dev_df = dev_df.head(args.max_samples // 5)  # Smaller dev set

    print(f"  ✓ Train: {len(train_df)} samples")
    print(f"  ✓ Dev: {len(dev_df)} samples")
    print(f"  ✓ Test: {len(test_df)} samples")

    # -------------------------
    # 4. Task-Specific Setup
    # -------------------------
    if task == 'cls':
        # Task 1: Sequence Classification
        print(f"\n🎯 Task 1: Sequence Classification")
        num_labels = 2  # literal (0) vs figurative (1)
        label_column = 'label_2'

        # Load model
        print(f"  Loading model: {model_checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels
        )

        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=False,  # Dynamic padding by data collator
                max_length=max_length
            )

        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df[['text', label_column]])
        dev_dataset = Dataset.from_pandas(dev_df[['text', label_column]])
        test_dataset = Dataset.from_pandas(test_df[['text', label_column]])

        # Rename label column
        train_dataset = train_dataset.rename_column(label_column, 'labels')
        dev_dataset = dev_dataset.rename_column(label_column, 'labels')
        test_dataset = test_dataset.rename_column(label_column, 'labels')

        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Metrics
        metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            f1 = metric.compute(predictions=predictions, references=labels, average='binary')
            return {"f1": f1['f1']}

    elif task in ['span', 'both']:
        # Task 2: Token Classification
        print(f"\n🎯 Task 2: Token Classification (IOB2 Tagging)")

        # Label mapping for IOB2
        label2id = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

        print(f"  Labels: {label2id}")

        # Load model
        print(f"  Loading model: {model_checkpoint}")
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )

        # Tokenization with IOB2 alignment (CRITICAL - Mission 4.2 Task 3.5)
        def tokenize_and_align(examples):
            """Tokenize and align IOB2 labels with subword tokens"""
            tokenized_inputs = tokenizer(
                [text.split() for text in examples['text']],  # Pre-tokenized by whitespace
                truncation=True,
                padding=False,
                max_length=max_length,
                is_split_into_words=True  # CRITICAL: aligns word_ids() with whitespace tokens
            )

            all_labels = []
            for i, text in enumerate(examples['text']):
                iob2_tags_str = examples['iob2_tags'][i]

                # Skip if missing IOB2 tags
                if pd.isna(iob2_tags_str) or str(iob2_tags_str) == 'nan':
                    word_ids = tokenized_inputs.word_ids(batch_index=i)
                    all_labels.append([-100] * len(word_ids))
                    continue

                # Parse IOB2 tags
                word_labels = str(iob2_tags_str).split()

                # Get word IDs for this example
                word_ids = tokenized_inputs.word_ids(batch_index=i)

                # Align labels
                aligned_labels = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        # Special tokens
                        aligned_labels.append(-100)
                    elif word_idx != previous_word_idx:
                        # First subword of word -> gets word's label
                        try:
                            aligned_labels.append(label2id[word_labels[word_idx]])
                        except (IndexError, KeyError):
                            aligned_labels.append(-100)
                    else:
                        # Subsequent subwords -> ignored in loss
                        aligned_labels.append(-100)
                    previous_word_idx = word_idx

                all_labels.append(aligned_labels)

            tokenized_inputs["labels"] = all_labels
            return tokenized_inputs

        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df[['text', 'iob2_tags']])
        dev_dataset = Dataset.from_pandas(dev_df[['text', 'iob2_tags']])
        test_dataset = Dataset.from_pandas(test_df[['text', 'iob2_tags']])

        # Tokenize and align
        train_dataset = train_dataset.map(tokenize_and_align, batched=True, remove_columns=['text', 'iob2_tags'])
        dev_dataset = dev_dataset.map(tokenize_and_align, batched=True, remove_columns=['text', 'iob2_tags'])
        test_dataset = test_dataset.map(tokenize_and_align, batched=True, remove_columns=['text', 'iob2_tags'])

        # Data collator for token classification
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Metrics for token classification
        seqeval_metric = evaluate.load("seqeval")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=2)

            # Convert to word-level labels (remove -100 and special tokens)
            true_labels = []
            pred_labels = []

            for prediction, label in zip(predictions, labels):
                true_label = []
                pred_label = []
                for p, l in zip(prediction, label):
                    if l != -100:
                        true_label.append(id2label[l])
                        pred_label.append(id2label[p])
                true_labels.append(true_label)
                pred_labels.append(pred_label)

            results = seqeval_metric.compute(predictions=pred_labels, references=true_labels)
            return {
                "f1": results["overall_f1"],
                "precision": results["overall_precision"],
                "recall": results["overall_recall"]
            }

    else:
        raise ValueError(f"Unsupported task: {task}. Use 'cls', 'span', or 'both'")

    print(f"  ✓ Model loaded with {num_labels} labels")
    print(f"  ✓ Datasets prepared")

    # -------------------------
    # 5. Freeze Backbone (if requested)
    # -------------------------
    if freeze_backbone:
        print(f"\n❄️  Freezing backbone parameters...")
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False
        print(f"  ✓ Backbone frozen - only training classification head")

    # -------------------------
    # 6. Training Arguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir=str(output_dir / "logs"),
        logging_steps=config.get('logging_steps', 100),
        save_total_limit=config.get('save_total_limit', 2),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        fp16=config.get('fp16', False),
        report_to="none",  # Disable wandb/tensorboard
    )

    # -------------------------
    # 7. Trainer Setup
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    # -------------------------
    # 8. Train
    # -------------------------
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {early_stopping_patience}")

    train_result = trainer.train()

    print(f"\n✅ Training complete!")
    print(f"  Training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"  Final train loss: {train_result.metrics['train_loss']:.4f}")

    # -------------------------
    # 9. Save Model
    # -------------------------
    print(f"\n💾 Saving best model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # -------------------------
    # 10. Evaluate on Test Set
    # -------------------------
    print(f"\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    print(f"\n🎯 Test Results:")
    print(f"  F1: {test_results['eval_f1']:.4f}")
    if 'eval_precision' in test_results:
        print(f"  Precision: {test_results['eval_precision']:.4f}")
        print(f"  Recall: {test_results['eval_recall']:.4f}")

    # -------------------------
    # 11. Save Results
    # -------------------------
    results_file = output_dir / "training_results.json"
    results = {
        "model": model_checkpoint,
        "task": task,
        "mode": mode_name,
        "freeze_backbone": freeze_backbone,
        "config": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay
        },
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "test_samples": len(test_dataset),
        "train_metrics": {
            "runtime": float(train_result.metrics['train_runtime']),
            "loss": float(train_result.metrics['train_loss'])
        },
        "test_metrics": {
            "f1": float(test_results['eval_f1']),
            "precision": float(test_results.get('eval_precision', 0)),
            "recall": float(test_results.get('eval_recall', 0))
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")
    print(f"{'=' * 80}\n")

    return results
