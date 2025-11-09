# Product Requirements Document (PRD)
# Hebrew Idiom Detection: Dataset Creation & Model Benchmarking

---

**Document Version:** 2.1 (VAST.ai + PyCharm Setup)
**Last Updated:** November 7, 2025
**Project Type:** Master's Thesis Research
**Primary Researcher:** [Your Name]
**Institution:** [Your University]
**Duration:** 12 weeks  

---

## Executive Summary

This research project presents **the first comprehensive Hebrew idiom dataset** with dual-task annotations and establishes performance benchmarks across encoder-based transformer models and large language models (LLMs).

**Core Innovation:** A balanced dataset of 4,800 Hebrew sentences enabling:
1. Sentence-level classification (literal vs. figurative)
2. Token-level span identification (IOB2 tagging)

**Research Goals:** Systematically compare 5 fine-tuned transformer models against LLM prompting approaches to determine optimal strategies for Hebrew figurative language understanding.

**Expected Impact:** First systematic benchmark for Hebrew idiom detection with implications for low-resource language NLP and cross-lingual transfer.

---

## 1. Project Overview

### 1.1 Problem Statement

Hebrew idioms present unique NLP challenges:
- **Context-dependent ambiguity**: Same expression can be literal or figurative
- **Multi-word expressions**: Precise boundary detection required
- **Resource scarcity**: No existing Hebrew idiom datasets with dual annotations
- **Model comparison gap**: Unclear whether fine-tuning or prompting is superior

**Example:**
```
Sentence: "הוא שבר את הראש על הבעיה"
          (He broke the head on the problem)

Context 1 (Figurative): Mental effort → "thought hard"
Context 2 (Literal): Physical injury → actual head injury

Challenge: Models must use context to distinguish meaning.
```

### 1.2 Research Questions

1. **Dataset Quality:**
   - Is the dataset balanced and representative?
   - Do IOB2 annotations align correctly with tokens?

2. **Model Performance:**
   - How do Hebrew-specific models (AlephBERT, DictaBERT) compare to multilingual (mBERT, XLM-R)?
   - What is the performance gap between sentence classification and span detection?
   - Do fine-tuned models outperform zero-shot approaches?

3. **LLM Evaluation:**
   - Can open-access LLMs compete with fine-tuned models via prompting?
   - Zero-shot vs. few-shot prompting: What's the improvement?

4. **Training Strategies:**
   - Which hyperparameters matter most for each task?
   - Full fine-tuning vs. frozen backbone: Cost-benefit analysis?

5. **Error Patterns:**
   - What types of idioms are hardest to detect?
   - What contextual patterns cause failures?

### 1.3 Success Criteria

**Minimum Viable Product (MVP):**
- ✅ Dataset validated and splits created
- ✅ 5 encoder models trained and evaluated on both tasks
- ✅ 1 LLM evaluated via prompting
- ✅ Comprehensive results comparison
- ✅ Academic paper draft complete

**Performance Targets:**
- Sentence Classification: F1 > 85%
- Token Classification: F1 > 80%
- LLM competitive within 5% of best fine-tuned model

**Publication Goal:**
- Submit to ACL/EMNLP or similar tier-1 venue
- Dataset released publicly on HuggingFace

---

## 2. Dataset Specification

### 2.1 Dataset Overview

**Name:** Hebrew-Idioms-4800  
**Size:** 4,800 manually annotated sentences  
**Language:** Hebrew  
**Annotation Levels:** 2 (sentence + token)  
**Balance:** 2,400 literal + 2,400 figurative  
**Status:** ✅ Complete and validated  

### 2.2 Data Schema

```python
{
    # Identifiers
    "id": int,                      # Unique identifier
    "split": str,                   # "train", "validation", or "test"
    
    # Text Data
    "text": str,                    # Full sentence
    "expression": str,              # Idiom (normalized form)
    "matched_expression": str,      # Idiom as it appears in text
    "language": str,                # "he" (Hebrew)
    "source": str,                  # Data source
    
    # Sentence-Level Label (Task 1)
    "label": str,                   # "מילולי" or "פיגורטיבי"
    "label_2": int,                 # 0 = literal, 1 = figurative
    
    # Token-Level Annotations (Task 2)
    "iob2_tags": str,              # Space-separated IOB2 tags
    "num_tokens": int,             # Total tokens in sentence
    "token_span_start": int,       # Token start position
    "token_span_end": int,         # Token end position
    
    # Character-Level (Auxiliary)
    "span_start": int,             # Character start
    "span_end": int,               # Character end
    "char_mask": str,              # Binary mask (0/1)
}
```

**Example Entry:**
```json
{
    "id": 42,
    "split": "train",
    "text": "הוא שבר את הראש על הבעיה המורכבת",
    "expression": "שבר את הראש",
    "matched_expression": "שבר את הראש",
    "label": "פיגורטיבי",
    "label_2": 1,
    "iob2_tags": "O B-IDIOM I-IDIOM I-IDIOM O O O",
    "num_tokens": 7,
    "token_span_start": 1,
    "token_span_end": 4
}
```

### 2.3 Dataset Statistics

| Metric | Value          |
|--------|----------------|
| Total sentences | 4,800          |
| Literal samples | 2,400 (50%)    |
| Figurative samples | 2,400 (50%)    |
| Unique idioms | 60 expressions |
| Avg sentence length | 12.5 tokens    |
| Avg idiom length | 3.2 tokens     |
| Train split | 3,360 (70%)    |
| Validation split | 720 (15%)      |
| Test split | 720 (15%)      |

### 2.4 Split Strategy

**Standard Stratified Split:**
- 70% train / 15% validation / 15% test
- Stratified by label (maintain 50/50 balance)
- Stratified by idiom (ensure all idioms represented)

**Quality Control:**
- ✅ Manually verified by native Hebrew speakers
- ✅ IOB2 tags validated for alignment
- ✅ No duplicate sentences
- ✅ Grammatically correct

---

## 3. Task Definitions

### 3.1 Task 1: Sentence Classification

**Objective:** Classify sentence as literal (0) or figurative (1)

**Input:**
```python
text = "הוא שבר את הראש על הבעיה"
```

**Output:**
```python
label = 1  # Figurative
probability = 0.89
```

**Evaluation Metrics:**
- Accuracy
- Macro F1-score
- Precision & Recall (per class)
- ROC-AUC
- Confusion Matrix

**Baseline Performance:**
- Random: 50%
- Majority class: 50%
- Expected fine-tuned: 85-92%

### 3.2 Task 2: Token Classification (IOB2)

**Objective:** Identify idiom span using BIO tagging

**Input:**
```python
tokens = ["הוא", "שבר", "את", "הראש", "על", "הבעיה"]
```

**Output:**
```python
labels = ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM", "O", "O"]
```

**Label Scheme:**
- `O`: Outside idiom
- `B-IDIOM`: Beginning of idiom
- `I-IDIOM`: Inside idiom (continuation)

**Evaluation Metrics:**
- Token-level F1 (macro)
- Span-level F1 (exact match)
- Precision & Recall (per class)
- Boundary accuracy

**Baseline Performance:**
- Random: 33% (3 classes)
- Expected fine-tuned: 80-90%

#### 3.2.1 Subword Tokenization Alignment (CRITICAL Implementation Detail)

**Challenge:** Transformer tokenizers (especially multilingual models like mBERT and XLM-RoBERTa) split Hebrew words into subword units, but our IOB2 tags are aligned with word-level tokens (whitespace-separated).

**Example of the problem:**
```python
# Original word-level tokens and IOB2 tags
Word tokens: ["הוא", "שבר", "את", "הראש", "על", "הבעיה"]
IOB2 tags:   ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM", "O", "O"]

# After mBERT/XLM-R tokenization (subwords)
Subword tokens: ["הוא", "##ש", "##בר", "את", "##ה", "##ראש", "על", "##ה", "##בעיה"]
# How to align IOB2 tags to these 9 subword tokens?
```

**Solution Strategy:**
1. Use HuggingFace tokenizer's `word_ids()` method to track which subword belongs to which original word
2. Align IOB2 labels to subwords:
   - First subword of each word gets the word's original IOB2 tag
   - Subsequent subwords of the same word get label `-100` (ignored in loss computation)
   - Special tokens ([CLS], [SEP], [PAD]) get label `-100`
3. During training: Model learns from first subword of each word
4. During evaluation: Aggregate subword predictions back to word-level before computing metrics

**Implementation:**
- Location: `src/utils/tokenization.py`
- Key function: `align_labels_with_tokens(tokenized_inputs, word_labels)`
- Used in: Data collation for Task 2 (Token Classification)

**Validation:**
- Must verify alignment preserves span boundaries
- Test on all model tokenizers (especially mBERT, XLM-R which aggressively split Hebrew)
- Print alignment examples during first training epoch

**Note:** This alignment is mandatory for Task 2. Without it, the model will learn incorrect label-token associations and produce meaningless results.

---

## 4. Model Specifications

### 4.1 Encoder Models (Fine-Tuning)

**Models to Evaluate (5 total):**

| Model | ID | Params | Type | Priority |
|-------|-----|--------|------|----------|
| **AlephBERT-base** | onlplab/alephbert-base | 110M | Hebrew | ⭐⭐⭐ |
| **AlephBERT-Gimmel** | [model-id] | 110M | Hebrew | ⭐⭐⭐ |
| **DictaBERT** | dicta-il/dictabert | 110M | Hebrew | ⭐⭐⭐ |
| **mBERT** | bert-base-multilingual-cased | 110M | Multilingual | ⭐⭐⭐ |
| **XLM-RoBERTa-base** | xlm-roberta-base | 125M | Multilingual | ⭐⭐⭐ |

**Selection Rationale:**
- Hebrew-specific models for language-specific performance
- Multilingual models for comparison and transfer learning
- Similar sizes (110-125M params) for fair comparison

### 4.2 LLM Model (Prompting)

**Target LLM (1 model):**

**Options:**
1. **Llama 3.1 70B** (via Azure/Together AI) - Recommended
2. **Mistral Large** (via Azure)
3. **GPT-3.5-Turbo** (via Azure OpenAI)

**Selection Criteria:**
- Open-access or affordable API
- Strong multilingual capabilities
- Hebrew language support

**Evaluation Approach:**
- Zero-shot prompting
- Few-shot prompting (3-5 examples)
- Structured output format (JSON)

### 4.3 Architecture Configuration

**For Sequence Classification:**
```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,  # Binary: literal vs. figurative
)
```

**For Token Classification:**
```python
model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    num_labels=3,  # IOB2: O, B-IDIOM, I-IDIOM
    id2label={0: "O", 1: "B-IDIOM", 2: "I-IDIOM"},
    label2id={"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
)
```

---

## 5. Training Methodology

### 5.1 Training Configuration

**Optimizer:** AdamW  
**Scheduler:** Linear decay with warmup  
**Early Stopping:** Patience = 2-3 epochs  
**Metric for Best Model:** Validation F1  

**Hyperparameter Search Space:**

| Parameter | Range | Default |
|-----------|-------|---------|
| Learning rate | [1e-5, 2e-5, 3e-5, 5e-5] | 2e-5 |
| Batch size | [8, 16, 32] | 16 |
| Epochs | [3, 5, 8] | 5 |
| Warmup ratio | [0.0, 0.1, 0.2] | 0.1 |
| Weight decay | [0.0, 0.01, 0.05] | 0.01 |
| Gradient accumulation | [1, 2, 4] | 1 |

**Hyperparameter Optimization:**
- Method: Optuna (10-15 trials per model)
- Objective: Maximize validation F1
- Pruning: Enabled (Successive Halving)

### 5.2 Training Modes

**Mode 1: Zero-Shot Evaluation**
- No training
- Direct evaluation on test set
- Baseline for comparison

**Mode 2: Full Fine-Tuning** ⭐ **PRIMARY**
- Train all model parameters
- Best expected performance
- ~20-30 minutes per model

**Mode 3: Frozen Backbone (Optional)**
- Train only classification head
- Fast baseline (~5 minutes)
- Use for comparison only

### 5.3 Cross-Seed Validation

**Seeds:** 42, 123, 456 (minimum 3)  
**Reporting:** Mean ± Standard Deviation  
**Statistical Testing:** Paired t-test (α = 0.05)

---

## 6. LLM Prompting Strategy

### 6.1 Prompting Approaches

**Strategy 1: Zero-Shot**
```
Analyze the following Hebrew sentence and determine if the expression 
"{expression}" is used literally or figuratively.

Sentence: {text}
Expression: {expression}

Answer with JSON:
{
  "classification": "literal" or "figurative",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
```

**Strategy 2: Few-Shot (3 examples)**
```
Examples:
1. "הוא שבר את הכוס" → literal (actually broke a cup)
2. "הוא שבר את הראש על הבעיה" → figurative (thought hard)
3. "היא שברה שיא עולמי" → figurative (broke a record)

Now analyze:
Sentence: {text}
Expression: {expression}
[Same JSON output format]
```

**Strategy 3: Chain-of-Thought**
```
Analyze step by step:
1. Identify the expression: {expression}
2. Look at the surrounding context
3. Determine if physical or metaphorical
4. Provide classification

[JSON output]
```

### 6.2 LLM Evaluation

**Metrics:**
- Accuracy vs. fine-tuned models
- API cost per prediction
- Inference latency (seconds)
- Reasoning quality (manual review of 50 samples)

**Cost Estimation:**
- ~4,800 test samples × 2 tasks = 9,600 API calls
- Estimated cost: $50-200 (depending on model)

---

## 7. Compute Infrastructure & Development Environment

### 7.1 Development Environment

**IDE:** PyCharm Professional/Community Edition

**Local Development:**
- Code development and testing in PyCharm
- Version control (Git) integration
- Project structure and file management
- All source code maintained in PyCharm project

**Note:** All code must be compatible with remote execution on VAST.ai servers

### 7.2 Compute Infrastructure (VAST.ai)

**Training Platform:** VAST.ai GPU Cloud

**Recommended Instance Configuration:**
- GPU: NVIDIA RTX 3090 / RTX 4090 / A5000
- VRAM: Minimum 24GB
- RAM: 32GB+
- Storage: 100GB+ NVMe SSD
- Cost: ~$0.20-0.60/hour (significantly cheaper than Azure)

**VAST.ai Selection Criteria:**
- Reliability score: >98%
- DLPerf score: >50
- CUDA version: 11.8+ or 12.x
- PyTorch/Transformers pre-installed (or Docker container support)

**Training Time Estimate:**
- Single model fine-tuning: 20-40 minutes per task
- Total training time: 10-15 hours across all experiments
- Estimated cost: $3-9 total (vs. $80+ on Azure)

### 7.3 Storage Infrastructure

**Google Drive (Primary Storage):**
- Available space: 2TB
- Storage allocation:
  - Dataset: ~5 MB
  - Model checkpoints: ~1 GB per model × 5 × 2 tasks = 10 GB
  - Experiment results/logs: ~1 GB
  - Intermediate files: ~2 GB
  - Total: ~15 GB (less than 1% of available space)

**Storage Strategy:**
- Raw dataset: Google Drive + GitHub
- Training checkpoints: Google Drive (synced from VAST.ai)
- Final models: Google Drive + HuggingFace Hub
- Results/logs: Google Drive
- Code repository: GitHub + Google Drive backup

**Data Transfer:**
- Upload dataset to VAST.ai instance at start of training
- Download model checkpoints to Google Drive after each experiment
- Use `rclone` or Google Drive CLI for automated sync

### 7.4 Budget Estimate

| Item | Cost |
|------|------|
| VAST.ai Compute (15 hours @ $0.40/hr avg) | $6 |
| Google Drive Storage (included, 2TB available) | $0 |
| LLM API Calls | $50-100 |
| **Total** | **~$56-106** |

**Cost Advantages:**
- 85-90% savings vs. Azure for GPU compute
- No storage costs (existing Google Drive)
- Pay-per-minute billing on VAST.ai

### 7.5 VAST.ai Setup Instructions

**Initial Setup:**
```bash
# 1. Create VAST.ai account at https://vast.ai

# 2. Search for instances (via Web UI or CLI)
#    - Filter: RTX 3090/4090, 24GB+ VRAM, >98% reliability
#    - Sort by: $/hr (ascending)

# 3. Rent instance and SSH connect
ssh root@<instance-ip> -p <port> -L 8080:localhost:8080

# 4. Install dependencies on instance
pip install transformers datasets torch accelerate optuna wandb scikit-learn

# 5. Upload dataset from Google Drive
# Option A: Direct download
wget --no-check-certificate '<google-drive-share-link>' -O expressions_data_tagged.csv

# Option B: Using gdown
pip install gdown
gdown <file-id>

# Option C: Mount Google Drive (if supported)
pip install google-colab
from google.colab import drive
drive.mount('/content/drive')

# 6. Clone code repository
git clone <your-repo-url>
cd hebrew-idiom-detection

# 7. Run training
python src/idiom_experiment.py --model alephbert-base --task sequence_classification

# 8. Download results to Google Drive after training
# Use gcloud CLI, rclone, or manual download via Jupyter notebook
```

**VAST.ai Workflow:**
1. Develop and test code locally in PyCharm
2. Push code to GitHub repository
3. Rent VAST.ai instance for training session
4. Clone repository and upload dataset to instance
5. Run training experiments
6. Download model checkpoints and results to Google Drive
7. Terminate instance (pay only for usage)
8. Continue development locally, repeat as needed

**Compatibility Requirements:**
- Code must work in both local PyCharm environment and VAST.ai remote servers
- Use relative paths for data/model loading
- Support command-line arguments for configuration
- Implement checkpointing for interrupted training sessions
- Use Docker containers (optional but recommended) for reproducibility

---

## 8. Experimental Design

### 8.1 Experiment Matrix

| Experiment | Models | Training | Seeds | Tasks | Total Runs |
|------------|--------|----------|-------|-------|------------|
| **E1: Zero-Shot** | 5 | None | 1 | 2 | 10 |
| **E2: Fine-Tuned** | 5 | Full | 3 | 2 | 30 |
| **E3: LLM Prompting** | 1 | N/A | N/A | 2 | 6 |
| **E4: Ablations** | 1 (best) | Variants | 1 | 2 | 6 |
| **Total** | - | - | - | - | **52 runs** |

### 8.2 Timeline (12 Weeks)

**Week 1: Setup & Validation**
- Azure environment setup
- Data validation and preparation
- Framework testing

**Week 2-3: Zero-Shot Baseline**
- Evaluate all 5 models (zero-shot)
- Establish baseline performance

**Week 4-6: Full Fine-Tuning**
- HPO for all 5 models (both tasks)
- Train final models with best hyperparameters
- Cross-seed validation

**Week 7: LLM Evaluation**
- Design prompting strategies
- Run LLM evaluations
- Compare with fine-tuned models

**Week 8: Ablation Studies**
- Frozen vs. full fine-tuning
- Hyperparameter sensitivity
- Data size impact

**Week 9: Analysis**
- Error analysis
- Statistical tests
- Visualization

**Week 10-11: Writing**
- Draft paper/thesis
- Results tables and figures
- Discussion section

**Week 12: Finalization**
- Revisions
- Final submission

---

## 9. Evaluation Protocol

### 9.1 Metrics Suite

**Classification Metrics (Task 1):**
- Accuracy
- Macro F1-score
- Precision (per class)
- Recall (per class)
- Confusion Matrix
- ROC-AUC

**Token Classification Metrics (Task 2):**
- Token-level F1 (macro)
- Span-level F1 (exact match)
- Precision/Recall (per class)
- Boundary accuracy

**LLM-Specific Metrics:**
- Cost per prediction (USD)
- Latency (seconds)
- Reasoning quality score (1-5)

### 9.2 Statistical Testing

**Method:** Paired t-test for cross-seed comparisons  
**Significance Level:** α = 0.05  
**Multiple Comparisons:** Bonferroni correction  

**Reporting Format:**
```
Model A: 89.2 ± 1.3% F1
Model B: 85.7 ± 1.8% F1
Difference: 3.5% (p = 0.018 < 0.05) ✓ Significant
```

### 9.3 Error Analysis

**Categories:**
1. False Positives (Literal → Figurative)
2. False Negatives (Figurative → Literal)
3. High-confidence errors (>0.8 probability)
4. Boundary errors (B/I confusion)

**Analysis Dimensions:**
- By idiom type
- By sentence length
- By context complexity
- By idiom frequency

---

## 10. Deliverables

### 10.1 Code Repository

**Development Environment:** PyCharm IDE (Professional/Community Edition)

**Project Structure:**
```
hebrew-idiom-detection/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── docker/                          # Docker configs for VAST.ai
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── expressions_data_tagged.csv
│   ├── splits/
│   └── README.md                    # Data documentation
├── src/
│   ├── __init__.py
│   ├── data_preparation.py          # Dataset loading and splitting
│   ├── idiom_experiment.py          # Main training script (CLI support - all modes: zero_shot, full_finetune, frozen_backbone, hpo)
│   ├── llm_evaluation.py            # LLM prompting experiments
│   ├── visualization.py             # Results visualization
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py               # Evaluation metrics
│       ├── tokenization.py          # IOB2 alignment for subword tokenization (CRITICAL for Task 2)
│       ├── storage.py               # Google Drive sync utilities
│       └── vast_utils.py            # VAST.ai helper functions
├── experiments/
│   ├── configs/                     # YAML/JSON experiment configs
│   │   ├── training_config.yaml     # Base training configuration template
│   │   └── hpo_config.yaml          # Optuna hyperparameter search space
│   ├── results/                     # Synced to Google Drive
│   └── logs/                        # Training logs
├── models/                          # Model checkpoints (local cache)
│   └── .gitkeep                     # Note: Actual models in Google Drive
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_llm_evaluation.ipynb
│   └── 04_error_analysis.ipynb
├── scripts/
│   ├── setup_vast_instance.sh       # Automate VAST.ai setup
│   ├── sync_to_gdrive.sh            # Google Drive upload script (rclone or manual - see Mission 4.4)
│   ├── download_from_gdrive.sh      # Google Drive download script (gdown)
│   ├── run_all_hpo.sh               # Batch runner for all 10 HPO studies (Mission 4.5, optional)
│   └── run_all_experiments.sh       # Batch runner for all 30 final training runs (Mission 4.6)
├── tests/                           # Unit tests (pytest)
│   ├── test_data_preparation.py
│   └── test_metrics.py
└── paper/
    ├── figures/
    ├── tables/
    └── main.tex
```

**PyCharm Configuration:**
- Python interpreter: Virtual environment (Python 3.9+)
- Code style: PEP 8
- Git integration enabled
- Remote interpreter support for VAST.ai (optional)

**VAST.ai Compatibility:**
- All scripts support command-line arguments
- Environment variables for paths (no hardcoded paths)
- Checkpointing enabled for all training scripts
- Google Drive sync integrated into training workflow
- Docker support for reproducible environments

**Key Scripts:**
- `src/idiom_experiment.py`: Main training script with CLI interface
- `scripts/setup_vast_instance.sh`: One-command VAST.ai setup
- `scripts/sync_to_gdrive.sh`: Automated model upload to Google Drive
- `requirements.txt`: All dependencies for local and remote environments

### 10.2 Dataset Release

**Package Contents:**
- CSV file with all annotations
- Data schema documentation
- Split files (train/val/test)
- Statistics report
- Annotation guidelines

**Distribution:**
- GitHub repository
- HuggingFace Datasets
- DOI via Zenodo

### 10.3 Academic Paper

**Target Venue:** ACL/EMNLP/NAACL  
**Format:** 8 pages + references  
**Submission Deadline:** [Target conference deadline]  

**Structure:**
1. Abstract (250 words)
2. Introduction (1.5 pages)
3. Dataset (2 pages)
4. Methodology (2 pages)
5. Results (2 pages)
6. Analysis (1.5 pages)
7. Conclusion (0.5 pages)

### 10.4 Thesis Document

**Length:** 60-80 pages  
**Language:** Hebrew with English abstract  
**Chapters:** 8-10  
**Defense Date:** [Target date]  

---

## 11. Risk Management

### 11.1 Technical Risks

| Risk | Probability | Mitigation |
|------|------------|------------|
| Azure compute unavailable | Low | Book instances in advance, use Spot as backup |
| Models don't converge | Low | Use proven architectures, careful HPO |
| LLM API rate limits | Medium | Batch requests, implement retry logic |
| Data quality issues | Low | Already validated, double-check before training |

### 11.2 Timeline Risks

| Risk | Probability | Mitigation |
|------|------------|------------|
| Training takes longer | Medium | Prioritize key models, parallel runs on Azure |
| LLM costs exceed budget | Low | Start with small test set, scale if affordable |
| Writing delays | Medium | Start early, weekly progress targets |

### 11.3 Research Risks

| Risk | Probability | Mitigation |
|------|------------|------------|
| Results not significant | Low | Multiple models, robust evaluation protocol |
| Paper rejection | Medium | Target multiple venues, iterate based on feedback |
| LLM underperforms | Low | Still valuable negative result for paper |

---

## 12. Success Metrics

### 12.1 Performance Targets

**Minimum Acceptable:**
- Sequence Classification F1 > 80%
- Token Classification F1 > 75%
- At least 3 models complete

**Target:**
- Sequence Classification F1 > 85%
- Token Classification F1 > 80%
- All 5 models + LLM evaluated
- Paper submitted

**Stretch:**
- Sequence Classification F1 > 90%
- Token Classification F1 > 85%
- Paper accepted at top venue
- Dataset widely adopted (>100 downloads in 6 months)

### 12.2 Publication Metrics

**Primary Goal:** Paper accepted at ACL/EMNLP/NAACL  
**Alternative:** Workshop paper or regional conference  
**Thesis:** Successfully defend and graduate  

### 12.3 Impact Metrics (Post-Publication)

- Dataset downloads > 100 (Year 1)
- GitHub stars > 25
- Citations > 5 (Year 1)
- Paper views > 300

---

## 13. References

**Key Papers:**
- Seker et al. (2022) - AlephBERT: Pre-trained Hebrew Language Models
- Dicta AI Labs (2023) - DictaBERT: Hebrew NLU Transformer
- Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers
- Conneau et al. (2020) - Unsupervised Cross-lingual Representation Learning (XLM-R)
- Brown et al. (2020) - Language Models are Few-Shot Learners (GPT-3)
- Wei et al. (2022) - Chain-of-Thought Prompting
- Liu & Hwa (2016) - Phrase Embeddings for Idioms
- Gharbieh et al. (2016) - Token-level Idiom Detection

---

## 14. Approval & Sign-Off

### 14.1 Stakeholders

| Role | Name | Approval Date |
|------|------|---------------|
| Primary Researcher | [Your Name] | [Date] |
| Thesis Advisor | [Advisor Name] | [ ] |
| Second Reader | [Reader Name] | [ ] |

### 14.2 Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-06 | Initial comprehensive PRD |
| 2.0 | 2025-11-06 | Hybrid version - streamlined, Azure-focused |
| 2.1 | 2025-11-07 | Updated infrastructure: VAST.ai compute, PyCharm IDE, Google Drive storage |

---

**Document Status:** ✅ APPROVED FOR EXECUTION  
**Next Review:** After Week 4 (checkpoint)  
**Contact:** [your.email@university.edu]  

---

**END OF PRD**

*This document serves as the master specification for the Hebrew Idiom Detection research project. All implementation work should follow these specifications.*
