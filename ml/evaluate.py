import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure config and ml modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from ml.dataset_loader import get_data_loaders
from ml.model import ISLAttentionBiLSTM, ISLBaselineModel

def evaluate_model(model_type="temporal", split_mode="stratified", model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Evaluating on device: {device}")
    
    # Load loaders
    train_loader, val_loader, test_loader, label_to_idx, idx_to_label = get_data_loaders(
        split_mode=split_mode, batch_size=config.BATCH_SIZE
    )

    if split_mode == "full":
        eval_loader = train_loader
        eval_split_name = "full dataset"
    else:
        eval_loader = test_loader
        eval_split_name = "test split"
    
    # Check test set size
    if len(eval_loader.dataset) == 0:
        print("[Error] No evaluation samples found in features folder! Preprocess the dataset first.")
        return False
        
    num_classes = len(label_to_idx)
    
    # Load model configuration
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
        
    if not os.path.exists(model_path):
        print(f"[Error] Trained model file not found at '{model_path}'. Train a model first.")
        return False
        
    # Initialize correct architecture
    if model_type == "baseline":
        model = ISLBaselineModel(input_dim=config.INPUT_DIM, num_classes=num_classes)
    else:
        model = ISLAttentionBiLSTM(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.LSTM_LAYERS,
            num_classes=num_classes,
            dropout=config.DROPOUT
        )
        
    # Load state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model weights from: {model_path}")
    except Exception as e:
        print(f"[Error] Failed to load model weights: {e}")
        return False
        
    model = model.to(device)
    model.eval()
    
    # Inference lists
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Performance profiling
    inference_times = []
    
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x = batch_x.to(device)
            
            # Start timer per batch
            start_t = time.time()
            logits, _ = model(batch_x)
            end_t = time.time()
            
            # Batch average time per sample
            batch_time = (end_t - start_t) / batch_x.size(0)
            inference_times.extend([batch_time] * batch_x.size(0))
            
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate performance metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    
    # Calculate top-3 accuracy if suitable
    top3_correct = 0
    for i in range(len(all_targets)):
        top3_indices = np.argsort(all_probs[i])[-3:]
        if all_targets[i] in top3_indices:
            top3_correct += 1
    top3_accuracy = top3_correct / len(all_targets)
    
    # Calculate model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / (avg_inference_time + 1e-6)
    
    print("\n" + "=" * 60)
    print("      GESCOM: PERFORMANCE EVALUATION DASHBOARD")
    print("=" * 60)
    print(f"Evaluation Split       : {eval_split_name}")
    print(f"Overall Test Accuracy: {accuracy:.4f} (Top-1)")
    print(f"Overall Top-3 Accuracy: {top3_accuracy:.4f}")
    print(f"Macro Precision       : {precision:.4f}")
    print(f"Macro Recall          : {recall:.4f}")
    print(f"Macro F1-Score        : {f1:.4f}")
    print("-" * 60)
    print(f"Average Inference Latency : {avg_inference_time*1000:.2f} ms / sample")
    print(f"Real-time Throughput (FPS): {fps:.2f} frames/sec")
    print(f"Model File Size           : {model_size_mb:.2f} MB")
    print("=" * 60)
    
    # Write report
    report_dict = {
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_inference_time_ms": float(avg_inference_time * 1000),
        "fps": float(fps),
        "model_size_mb": float(model_size_mb)
    }
    
    report_path = os.path.join(config.MODELS_DIR, f"evaluation_{model_type}.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
        
    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(15, 12))
    # Plot top subset if num_classes is too large (like 100)
    if num_classes > 25:
        # Plot a heatmap of the confusion matrix without full text labels to keep it clean
        sns.heatmap(cm, cmap="Blues", cbar=True)
        plt.title("Confusion Matrix (101 Spoken Language sentences)")
    else:
        labels = [idx_to_label[str(i)] for i in range(num_classes)]
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    cm_path = os.path.join(config.MODELS_DIR, f"confusion_matrix_{model_type}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[Success] Saved confusion matrix visualization to {cm_path}")
    print(f"[Success] Saved evaluation metrics to {report_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="temporal", choices=["temporal", "baseline"])
    parser.add_argument("--split", type=str, default="full", choices=["signer", "stratified", "full"])
    args = parser.parse_args()
    
    evaluate_model(model_type=args.model, split_mode=args.split)
