"""
Mission 3.1: Model Selection and Download
Download and verify all 5 encoder models for Hebrew idiom detection
"""

import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json


class ModelDownloader:
    """Download and verify Hebrew language models"""

    def __init__(self):
        """Initialize with model list from PRD Section 4.1"""
        self.models = {
            'alephbert-base': 'onlplab/alephbert-base',
            'alephbert-gimmel': 'dicta-il/alephbertgimmel-base',  # Official AlephBERT Gimmel (128K vocab, SOTA)
            'dictabert': 'dicta-il/dictabert',
            'mbert': 'bert-base-multilingual-cased',
            'xlm-roberta-base': 'xlm-roberta-base'
        }

        self.model_info = {}

        # Sample Hebrew sentence for testing
        self.test_sentence = "הוא שבר את הראש כדי למצוא פתרון לבעיה."

    def download_and_verify_model(self, model_name: str, model_id: str) -> dict:
        """
        Download model and tokenizer, verify they work

        Args:
            model_name: Short name for the model
            model_id: HuggingFace model ID

        Returns:
            Dictionary with model information
        """
        print(f"\n{'='*80}")
        print(f"Processing: {model_name}")
        print(f"Model ID: {model_id}")
        print(f"{'='*80}")

        try:
            # Download tokenizer
            print(f"[1/5] Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            print(f"✅ Tokenizer downloaded")

            # Download model
            print(f"[2/5] Downloading model...")
            model = AutoModel.from_pretrained(model_id)
            print(f"✅ Model downloaded")

            # Count parameters
            print(f"[3/5] Counting parameters...")
            num_params = sum(p.numel() for p in model.parameters())
            num_params_m = num_params / 1_000_000
            print(f"✅ Parameters: {num_params:,} ({num_params_m:.1f}M)")

            # Verify parameter count is in expected range
            if not (100 <= num_params_m <= 150):
                print(f"⚠️  Warning: Parameter count {num_params_m:.1f}M outside expected range 100-150M")

            # Test tokenization on Hebrew
            print(f"[4/5] Testing Hebrew tokenization...")
            tokens = tokenizer(self.test_sentence, return_tensors='pt')
            print(f"✅ Tokenized successfully")
            print(f"   Input IDs shape: {tokens['input_ids'].shape}")
            print(f"   Sample tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0][:10])}")

            # Test forward pass
            print(f"[5/5] Testing forward pass...")
            with torch.no_grad():
                outputs = model(**tokens)
            print(f"✅ Forward pass successful")
            print(f"   Output shape: {outputs.last_hidden_state.shape}")

            # Gather model info
            info = {
                'model_id': model_id,
                'num_parameters': int(num_params),
                'num_parameters_millions': round(num_params_m, 2),
                'vocab_size': tokenizer.vocab_size,
                'max_length': tokenizer.model_max_length,
                'hidden_size': model.config.hidden_size,
                'num_layers': model.config.num_hidden_layers,
                'num_attention_heads': model.config.num_attention_heads,
                'test_tokenization_length': tokens['input_ids'].shape[1],
                'status': 'verified'
            }

            print(f"\n✅ {model_name.upper()} - ALL CHECKS PASSED")
            print(f"   Vocab size: {info['vocab_size']:,}")
            print(f"   Max length: {info['max_length']}")
            print(f"   Hidden size: {info['hidden_size']}")
            print(f"   Layers: {info['num_layers']}")
            print(f"   Attention heads: {info['num_attention_heads']}")

            return info

        except Exception as e:
            print(f"\n❌ ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_id': model_id,
                'status': 'failed',
                'error': str(e)
            }

    def download_all_models(self) -> dict:
        """Download and verify all models"""
        print("\n" + "🚀" * 40)
        print("MISSION 3.1: MODEL SELECTION AND DOWNLOAD")
        print("🚀" * 40)

        print(f"\nModels to download: {len(self.models)}")
        for name, model_id in self.models.items():
            print(f"  • {name}: {model_id}")

        print(f"\nTest sentence: {self.test_sentence}")

        # Download each model
        for model_name, model_id in self.models.items():
            info = self.download_and_verify_model(model_name, model_id)
            self.model_info[model_name] = info

        return self.model_info

    def save_model_info(self, output_path: str = None):
        """Save model information to JSON file"""
        if output_path is None:
            output_path = Path(__file__).parent.parent / "experiments" / "configs" / "model_info.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_info, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Model info saved to: {output_path}")

    def verify_success(self) -> bool:
        """Verify all models downloaded successfully"""
        print("\n" + "="*80)
        print("MISSION 3.1 SUCCESS CRITERIA CHECK")
        print("="*80)

        criteria = []

        # Check each model
        for model_name, info in self.model_info.items():
            success = info.get('status') == 'verified'
            criteria.append((f"{model_name} downloaded and verified", success))

        # Overall checks
        all_verified = all(info.get('status') == 'verified' for info in self.model_info.values())
        criteria.append(("All 5 models downloaded", len(self.model_info) == 5))
        criteria.append(("All models verified", all_verified))
        criteria.append(("Model info documented", len(self.model_info) > 0))

        all_passed = True
        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            print(f"{status} {criterion}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n" + "="*80)
            print("🎉 MISSION 3.1 COMPLETE - ALL MODELS READY!")
            print("="*80)
            print(f"\n📋 Summary:")
            for model_name, info in self.model_info.items():
                if info.get('status') == 'verified':
                    print(f"   • {model_name}: {info['num_parameters_millions']}M params, vocab={info['vocab_size']:,}")
            print(f"\n✅ Ready for Mission 3.2: Zero-Shot Evaluation")
        else:
            print("\n⚠️  Some models failed. Please check errors above.")

        return all_passed

    def run_mission_3_1(self):
        """Complete Mission 3.1"""
        # Download all models
        self.download_all_models()

        # Save model info
        self.save_model_info()

        # Verify success
        success = self.verify_success()

        return success


def main():
    """Main function"""
    downloader = ModelDownloader()
    success = downloader.run_mission_3_1()
    return downloader, success


if __name__ == "__main__":
    downloader, success = main()
