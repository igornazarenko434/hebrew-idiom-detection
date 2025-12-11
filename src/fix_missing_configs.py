import os
import json
from pathlib import Path
from transformers import AutoConfig

# Configuration
RESULTS_DIR = Path("experiments/results/full_fine-tuning")

# Model mapping (Folder Name -> HF ID)
MODEL_MAP = {
    "alephbert-base": "onlplab/alephbert-base",
    "alephbertgimmel-base": "dicta-il/alephbertgimmel-base",
    "dictabert": "dicta-il/dictabert",
    "bert-base-multilingual-cased": "bert-base-multilingual-cased",
    "xlm-roberta-base": "xlm-roberta-base"
}

# Task 2 Label Map
ID2LABEL = {0: "O", 1: "B-IDIOM", 2: "I-IDIOM"}
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}

print("🔧 Starting Config Fixer for Task 2 (Span)...\n")

count = 0
for model_dir in RESULTS_DIR.iterdir():
    if not model_dir.is_dir() or model_dir.name not in MODEL_MAP:
        continue
        
    model_name = model_dir.name
    hf_id = MODEL_MAP[model_name]
    
    # Check 'span' task folder
    span_dir = model_dir / "span"
    if not span_dir.exists():
        continue
        
    for seed_dir in span_dir.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
            
        config_path = seed_dir / "config.json"
        
        if not config_path.exists():
            print(f"🛠️  Fixing missing config for: {model_name} | span | {seed_dir.name}\n")
            
            # Load base config
            try:
                config = AutoConfig.from_pretrained(hf_id)
                
                # Update with Task 2 specifics
                config.id2label = ID2LABEL
                config.label2id = LABEL2ID
                config.num_labels = 3
                
                # Save
                config.save_pretrained(seed_dir)
                count += 1
            except Exception as e:
                print(f"❌ Failed to fix {seed_dir}: {e}\n")

print(f"\n✅ Fixed {count} missing config files.")
