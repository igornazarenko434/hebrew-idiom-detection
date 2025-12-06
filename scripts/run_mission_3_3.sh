#!/bin/bash
# Mission 3.3: Zero-Shot Evaluation for All Models
# Runs src/idiom_experiment.py in zero_shot mode for 5 models x 2 datasets

# Models to evaluate
MODELS=(
    "onlplab/alephbert-base"
    "dicta-il/alephbertgimmel-base"
    "dicta-il/dictabert"
    "bert-base-multilingual-cased"
    "xlm-roberta-base"
)

# Datasets to evaluate on (Path and Split Name)
# Format: "PATH|SPLIT_NAME"
DATASETS=(
    "data/splits/test.csv|test"
    "data/splits/unseen_idiom_test.csv|unseen_idiom_test"
)

# Create output directory if it doesn't exist
mkdir -p experiments/results/zero_shot

echo "========================================================"
echo "STARTING MISSION 3.3: ZERO-SHOT EVALUATION"
echo "Models: ${#MODELS[@]}"
echo "Datasets: ${#DATASETS[@]}"
echo "Device: mps"
echo "========================================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "--------------------------------------------------------"
    echo "🤖 Processing Model: $model"
    echo "--------------------------------------------------------"

    for dataset_info in "${DATASETS[@]}"; do
        # Split dataset info string
        IFS="|" read -r data_path split_name <<< "$dataset_info"
        
        echo "  📂 Dataset: $split_name ($data_path)"
        
        # Run experiment
        # We rely on idiom_experiment.py to handle validation, metric calc, and saving
        .venv/bin/python src/idiom_experiment.py \
            --mode zero_shot \
            --model_id "$model" \
            --data "$data_path" \
            --split "$split_name" \
            --task both \
            --device mps
            
        if [ $? -eq 0 ]; then
            echo "  ✅ Completed $split_name for $model"
        else
            echo "  ❌ FAILED $split_name for $model"
        fi
    done
done

echo ""
echo "========================================================"
echo "MISSION 3.3 COMPLETE"
echo "Check results in: experiments/results/zero_shot/"
echo "========================================================"
