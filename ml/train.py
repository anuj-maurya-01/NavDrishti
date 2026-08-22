import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure config and ml modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from ml.dataset_loader import get_data_loaders
from ml.model import ISLAttentionBiLSTM, ISLBaselineModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(model_type="temporal", split_mode="stratified", epochs=config.EPOCHS, 
                batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE, patience=8):
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    
    # Load loaders
    train_loader, val_loader, _, label_to_idx, _ = get_data_loaders(
        split_mode=split_mode, batch_size=batch_size
    )
    
    # Check if we have preprocessed data
    if len(train_loader.dataset) == 0:
        print("[Error] No preprocessed feature files found in data/features! Run preprocessing first.")
        return False
        
    num_classes = len(label_to_idx)
    
    # Create model
    if model_type == "baseline":
        print("[Train] Initializing Baseline Model (1D CNN + MLP)")
        model = ISLBaselineModel(input_dim=config.INPUT_DIM, num_classes=num_classes)
    else:
        print("[Train] Initializing Main Temporal Model (BiLSTM + Attention)")
        model = ISLAttentionBiLSTM(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.LSTM_LAYERS,
            num_classes=num_classes,
            dropout=config.DROPOUT
        )
        
    model = model.to(device)
    
    # Loss, Optimizer, and LR Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    monitor_on_train = split_mode == "full"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Checkpoints directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
    final_model_path = os.path.join(config.MODELS_DIR, f"final_model_{model_type}.pth")
    
    # Save feature configuration for model loading
    feature_config_path = os.path.join(config.MODELS_DIR, "feature_config.json")
    with open(feature_config_path, "w") as f:
        json.dump({
            "input_dim": config.INPUT_DIM,
            "sequence_length": config.SEQUENCE_LENGTH,
            "model_type": model_type
        }, f, indent=4)
        
    # Training Loop
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                
                running_val_loss += loss.item() * batch_x.size(0)
                _, predicted = logits.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
                
        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        monitored_loss = epoch_train_loss if monitor_on_train else epoch_val_loss
        
        # Log to history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch:2d}/{epochs:2d} | "
              f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.4f}")
              
        # LR Scheduler step
        scheduler.step(monitored_loss)
        
        # Checkpoint: Save best model
        if monitored_loss < best_val_loss:
            best_val_loss = monitored_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f" ==> Saved new best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            
        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs of no improvement.")
            break
            
    # Save final model
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save history to file
    history_path = os.path.join(config.MODELS_DIR, f"history_{model_type}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print("Training process finished successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="temporal", choices=["temporal", "baseline"])
    parser.add_argument("--split", type=str, default="full", choices=["signer", "stratified", "full"])
    parser.add_argument("--epochs", type=int, default=3) # Small number for local testing
    args = parser.parse_args()
    
    train_model(model_type=args.model, split_mode=args.split, epochs=args.epochs)
