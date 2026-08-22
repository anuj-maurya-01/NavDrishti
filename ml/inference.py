import os
import sys
import json
import numpy as np
import torch
import cv2

# Ensure config and ml modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from ml.feature_extraction import FeatureExtractor
from ml.model import ISLAttentionBiLSTM, ISLBaselineModel

class ISLInferenceManager:
    def __init__(self, model_type="temporal", model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Inference] Using device: {self.device}")
        
        # Paths
        labels_path = os.path.join(config.MODELS_DIR, "labels.json")
        if model_path is None:
            model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
            
        # Verify files exist
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Label map not found at {labels_path}. Train the model first.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
            
        # Load labels
        with open(labels_path, "r") as f:
            mapping = json.load(f)
            self.idx_to_label = mapping["idx_to_label"]
            self.label_to_idx = mapping["label_to_idx"]
            
        self.num_classes = len(self.idx_to_label)
        
        # Load model architecture
        if model_type == "baseline":
            self.model = ISLBaselineModel(input_dim=config.INPUT_DIM, num_classes=self.num_classes)
        else:
            self.model = ISLAttentionBiLSTM(
                input_dim=config.INPUT_DIM,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.LSTM_LAYERS,
                num_classes=self.num_classes,
                dropout=config.DROPOUT
            )
            
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[Inference] Successfully loaded model from {model_path}")
        
        # Initialize FeatureExtractor
        self.extractor = FeatureExtractor(static_image_mode=False)
        
        # Sequence buffer for temporal inference
        self.buffer = []
        self.sequence_length = config.SEQUENCE_LENGTH
        
    def reset_buffer(self):
        """
        Clears the temporal frame buffer.
        """
        self.buffer = []
        print("[Inference] Sequence buffer cleared.")

    def process_frame(self, frame):
        """
        Processes a single BGR frame.
        - Extracts MediaPipe landmarks & hand-crafted cues.
        - Updates the temporal buffer.
        - Runs inference if we have enough frames.
        Returns:
            prediction: str (or None if buffer not ready/no face detected)
            confidence: float (or 0.0)
            cues: dict (facial expression and pose states)
        """
        # Extract features
        feats, cues = self.extractor.extract_features(frame)
        
        # Check if face was detected
        # The face features are from indices 18 to 372
        face_detected = np.any(feats[18:378] != 0)
        
        # If face detected, push to sequence buffer
        if face_detected:
            self.buffer.append(feats)
        elif len(self.buffer) > 0:
            # Carry forward the last valid features if face is temporarily occluded
            self.buffer.append(self.buffer[-1])
        else:
            # Buffer is empty and no face detected: do not push anything, return zeros
            pass
            
        # Limit buffer length
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
            
        # If buffer is not full, pad it by copying the first valid features
        if len(self.buffer) == 0:
            return None, 0.0, cues, []
            
        # We can predict immediately by padding the buffer with copies of the oldest item
        padded_buffer = list(self.buffer)
        while len(padded_buffer) < self.sequence_length:
            padded_buffer.insert(0, padded_buffer[0])
            
        # Convert buffer to tensor shape (1, 30, 534)
        seq_array = np.array(padded_buffer, dtype=np.float32)
        seq_tensor = torch.tensor(seq_array).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, attn_weights = self.model(seq_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(1)
            top_k = min(3, probs.shape[1])
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
            
        pred_label = self.idx_to_label[str(pred_idx.item())]
        pred_conf = conf.item()
        top_predictions = [
            {
                "label": self.idx_to_label[str(idx.item())],
                "confidence": float(prob.item())
            }
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]
        
        # Add attention weights to cues for debugging/explainability
        if attn_weights is not None:
            cues["attention_distribution"] = attn_weights[0].cpu().numpy().tolist()
            
        return pred_label, pred_conf, cues, top_predictions

    def close(self):
        self.extractor.close()

if __name__ == "__main__":
    # Test inference locally
    print("Testing Inference Manager...")
    try:
        manager = ISLInferenceManager()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(dummy_frame, (320, 240), 100, (255, 255, 255), -1) # fake face
        
        pred, conf, cues = manager.process_frame(dummy_frame)
        print("Prediction:", pred)
        print("Confidence:", conf)
        print("Cues:", list(cues.keys()))
        manager.close()
    except Exception as e:
        print("Inference test error:", e)
