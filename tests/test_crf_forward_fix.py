import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Mock torchcrf if not installed, or use real one
try:
    from torchcrf import CRF
except ImportError:
    # Minimal Mock for CRF
    class CRF(nn.Module):
        def __init__(self, num_tags, batch_first=True):
            super().__init__()
            self.num_tags = num_tags
        def forward(self, emissions, tags, mask=None, reduction='mean'):
            return torch.tensor(10.0) # Dummy log likelihood
        def decode(self, emissions, mask=None):
            # Dummy Viterbi: return hardcoded path [0, 1, 2] for testing
            batch_size, seq_len, _ = emissions.shape
            return [[0, 1, 2] + [0]*(seq_len-3) for _ in range(batch_size)]

# ---------------------------------------------------------
# The Class Under Test (Copied from src/idiom_experiment.py)
# ---------------------------------------------------------
class BertCRFForTokenClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        # Mock transformer and classifier
        self.transformer = MagicMock()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.use_crf = True
        self.config = MagicMock()
        self.config.hidden_size = 768

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # 1. Mock Transformer Output
        batch_size, seq_len = input_ids.shape
        # Create dummy hidden states
        hidden_states = torch.randn(batch_size, seq_len, 768)
        
        # Mocking the transformer return
        outputs = MagicMock()
        outputs.last_hidden_state = hidden_states
        outputs.__getitem__ = lambda self, x: hidden_states
        
        self.transformer.return_value = outputs

        # --- Actual Logic from src/idiom_experiment.py ---
        
        # Get transformer outputs
        if token_type_ids is not None:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Get hidden states
        sequence_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if not self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss, logits
            else:
                return logits

        # CRF forward
        # Always compute Viterbi decoding for prediction/eval
        mask = attention_mask.bool()
        viterbi_predictions = self.crf.decode(logits, mask=mask)
        
        # Convert list of lists to padded tensor
        max_len = logits.size(1)
        predictions_padded = []
        for pred in viterbi_predictions:
            padded = pred + [-100] * (max_len - len(pred)) # Pad with -100
            predictions_padded.append(padded)
        predictions_tensor = torch.tensor(predictions_padded, device=logits.device)

        if labels is not None:
            # Training/Evaluation: compute CRF loss
            crf_mask = (labels != -100) & (attention_mask.bool())
            crf_mask[:, 0] = True 
            labels_cleaned = labels.clone()
            labels_cleaned[labels == -100] = 0

            log_likelihood = self.crf(logits, labels_cleaned, mask=crf_mask, reduction='mean')
            loss = -log_likelihood 

            return loss, predictions_tensor # Return Viterbi predictions, not logits
        else:
            return predictions_tensor

# ---------------------------------------------------------
# Test Execution
# ---------------------------------------------------------
def test_forward_with_labels():
    print("🧪 Testing forward() with labels (Evaluation Mode)...")
    
    model = BertCRFForTokenClassification(num_labels=3)
    
    # Inputs
    input_ids = torch.randint(0, 100, (2, 5)) # Batch=2, Seq=5
    mask = torch.ones(2, 5)
    labels = torch.tensor([[0, 1, 2, 0, -100], [0, 0, 0, 0, 0]]) # Dummy labels
    
    # Call forward
    output = model(input_ids, mask, labels=labels)
    
    # Check 1: Return type
    if not isinstance(output, tuple):
        print("❌ FAILED: Expected tuple (loss, preds), got single value")
        return
        
    loss, preds = output
    
    # Check 2: Second element is NOT logits
    # Logits shape would be (2, 5, 3)
    # Viterbi Preds shape would be (2, 5)
    
    print(f"  Output[1] shape: {preds.shape}")
    
    if len(preds.shape) == 2:
        print("✅ PASSED: Output[1] is 2D tensor (Viterbi Predictions)")
    elif len(preds.shape) == 3:
        print("❌ FAILED: Output[1] is 3D tensor (Logits). Fix did not work.")
    else:
        print(f"❌ FAILED: Unexpected shape {preds.shape}")

def test_forward_without_labels():
    print("\n🧪 Testing forward() without labels (Inference Mode)...")
    
    model = BertCRFForTokenClassification(num_labels=3)
    input_ids = torch.randint(0, 100, (2, 5))
    mask = torch.ones(2, 5)
    
    # Call forward
    output = model(input_ids, mask, labels=None)
    
    # Check: Return type should be tensor (predictions)
    if isinstance(output, tuple):
        print("❌ FAILED: Expected single tensor, got tuple")
    elif len(output.shape) == 2:
        print("✅ PASSED: Output is 2D tensor (Viterbi Predictions)")
    else:
        print(f"❌ FAILED: Unexpected output {output}")

if __name__ == "__main__":
    test_forward_with_labels()
    test_forward_without_labels()
