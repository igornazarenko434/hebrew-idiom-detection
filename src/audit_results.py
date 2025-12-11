import os
from pathlib import Path

# Define expected structure
models = [
    "alephbert-base", 
    "alephbertgimmel-base", 
    "dictabert", 
    "bert-base-multilingual-cased", 
    "xlm-roberta-base"
]
tasks = ["cls", "span"]
seeds = [42, 123, 456]

base_dir = Path("experiments/results/full_fine-tuning")

print(f"{'Model':<30} | {'Task':<5} | {'Seed':<4} | {'Status':<10} | {'Model File':<20} | {'Checkpoints?'}")
print("-" * 100)

for model in models:
    for task in tasks:
        for seed in seeds:
            # Construct path (accounting for nested structure if present)
            # Based on your ls output: .../seed_XXX/full_fine-tuning/model/task/...
            seed_dir = base_dir / model / task / f"seed_{seed}"
            
            # Find where the model file actually is (handling recursion)
            model_file = None
            has_results = False
            checkpoints = []
            
            if seed_dir.exists():
                # Look for model file recursively in this seed dir
                found_models = list(seed_dir.rglob("model.safetensors")) + list(seed_dir.rglob("pytorch_model.bin"))
                # Filter out ones inside checkpoint dirs for the "final" check
                final_models = [p for p in found_models if "checkpoint-" not in str(p)]
                
                if final_models:
                    model_file = final_models[0].name
                
                # Check for results json
                found_results = list(seed_dir.rglob("training_results.json"))
                if found_results:
                    has_results = True
                    
                # Check for checkpoint dirs
                checkpoints = list(seed_dir.rglob("checkpoint-*"))

            # Determine Status
            status = "MISSING"
            if model_file and has_results:
                status = "✅ DONE"
            elif model_file:
                status = "⚠️ NO JSON"
            elif seed_dir.exists():
                status = "❌ EMPTY"
            else:
                status = "❌ NO DIR"

            # Print row
            ckpt_str = f"Yes ({len(checkpoints)})" if checkpoints else "No"
            print(f"{model:<30} | {task:<5} | {seed:<4} | {status:<10} | {str(model_file):<20} | {ckpt_str}")
