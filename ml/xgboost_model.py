import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from ml.dataset_loader import get_data_loaders


def summarize_sequence(sequence):
    """Convert a (timesteps, features) sequence into fixed temporal descriptors."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3:
        raise ValueError(f"Expected a 3D sequence batch, got shape {sequence.shape}")

    deltas = np.diff(sequence, axis=1, prepend=sequence[:, :1, :])
    return np.concatenate([
        sequence.mean(axis=1),
        sequence.std(axis=1),
        sequence.min(axis=1),
        sequence.max(axis=1),
        np.abs(deltas).mean(axis=1),
    ], axis=1)


def loader_to_arrays(loader):
    sequences = []
    labels = []
    for batch_x, batch_y in loader:
        sequences.append(batch_x.numpy())
        labels.append(batch_y.numpy())
    return summarize_sequence(np.concatenate(sequences)), np.concatenate(labels)


def train_xgboost(split_mode="full", estimators=250):
    train_loader, val_loader, test_loader, label_to_idx, idx_to_label = get_data_loaders(
        split_mode=split_mode, batch_size=config.BATCH_SIZE
    )
    if len(train_loader.dataset) == 0:
        raise RuntimeError("No preprocessed feature files found in data/features")

    train_x, train_y = loader_to_arrays(train_loader)
    eval_loader = val_loader if split_mode == "full" else test_loader
    eval_x, eval_y = loader_to_arrays(eval_loader)

    model = XGBClassifier(
        n_estimators=estimators,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.7,
        objective="multi:softprob",
        num_class=len(label_to_idx),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        random_state=42,
    )
    print(f"[XGBoost] Training on {len(train_y)} samples with {train_x.shape[1]} temporal features.")
    model.fit(train_x, train_y, eval_set=[(eval_x, eval_y)], verbose=False)

    predictions = model.predict(eval_x).astype(int)
    probabilities = model.predict_proba(eval_x)
    top_k = min(3, probabilities.shape[1])
    top3 = np.argsort(probabilities, axis=1)[:, -top_k:]
    top3_accuracy = float(np.mean([target in choices for target, choices in zip(eval_y, top3)]))
    metrics = {
        "model": "xgboost",
        "split": split_mode,
        "samples": int(len(eval_y)),
        "accuracy": float(accuracy_score(eval_y, predictions)),
        "top3_accuracy": top3_accuracy,
        "macro_f1": float(f1_score(eval_y, predictions, average="macro", zero_division=0)),
        "feature_count": int(train_x.shape[1]),
    }

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, "xgboost_model.json")
    model.save_model(model_path)
    with open(os.path.join(config.MODELS_DIR, "evaluation_xgboost.json"), "w") as handle:
        json.dump(metrics, handle, indent=4)
    with open(os.path.join(config.MODELS_DIR, "xgboost_config.json"), "w") as handle:
        json.dump({"feature_count": int(train_x.shape[1]), "model_type": "xgboost"}, handle, indent=4)

    print(json.dumps(metrics, indent=4))
    print(f"[XGBoost] Saved model to {model_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["signer", "stratified", "full"], default="full")
    parser.add_argument("--estimators", type=int, default=250)
    args = parser.parse_args()
    train_xgboost(split_mode=args.split, estimators=args.estimators)
