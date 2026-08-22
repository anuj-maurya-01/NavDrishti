import os
import json
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Ensure config is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def get_signer_id(filepath):
    """
    Extracts a signer identifier based on file name patterns.
    - 'are you free today.mp4' -> signer_1
    - 'free (2).MP4' -> signer_2
    - 'free (3).MP4' -> signer_3
    - 'free (4).MP4' -> signer_4
    - 'free (5).MP4' -> signer_5
    - 'MVI_xxxx.MP4' -> signer_6 (canon camera)
    - Others -> signer_7
    """
    bn = os.path.basename(filepath)
    if "(2)" in bn:
        return "signer_2"
    elif "(3)" in bn:
        return "signer_3"
    elif "(4)" in bn:
        return "signer_4"
    elif "(5)" in bn:
        return "signer_5"
    elif "(1)" in bn or "(8)" in bn:
        return "signer_6"
    elif "MVI_" in bn:
        return "signer_6"
    else:
        # Default or root names
        return "signer_1"

def load_label_mapping():
    """
    Creates and saves label mapping.
    """
    excel_path = config.EXCEL_DETAILS_PATH
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Details excel sheet not found at {excel_path}")
        
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['Sentences'])
    df = df[df['File location'].apply(lambda x: isinstance(x, str) and not x.startswith('>>>'))]
    
    unique_sentences = sorted(df['Sentences'].unique())
    label_to_idx = {sentence: idx for idx, sentence in enumerate(unique_sentences)}
    idx_to_label = {idx: sentence for idx, sentence in enumerate(unique_sentences)}
    
    # Save to models directory
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    labels_file = os.path.join(config.MODELS_DIR, "labels.json")
    with open(labels_file, "w") as f:
        json.dump({
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label
        }, f, indent=4)
        
    print(f"[Info] Saved label mapping with {len(label_to_idx)} classes to {labels_file}")
    return label_to_idx, idx_to_label

class ISLDataset(Dataset):
    def __init__(self, metadata_df, label_to_idx, augment=False):
        self.df = metadata_df
        self.label_to_idx = label_to_idx
        self.augment = augment
        
        # Filter rows to only those whose .npy files actually exist
        self.records = []
        for idx, row in self.df.iterrows():
            rel_path = row['File location']
            safe_name = rel_path.replace("\\", "_").replace("/", "_").replace(" ", "_")
            feature_filename = f"{safe_name}.npy"
            feature_filepath = os.path.join(config.FEATURES_DIR, feature_filename)
            
            if os.path.exists(feature_filepath):
                self.records.append({
                    "feature_path": feature_filepath,
                    "label": row['Sentences']
                })
                
        print(f"[Dataset] Loaded {len(self.records)} valid samples out of {len(self.df)} metadata entries.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        feature_path = record["feature_path"]
        label_str = record["label"]
        label_idx = self.label_to_idx[label_str]
        
        # Load pre-extracted feature sequence (SEQUENCE_LENGTH, INPUT_DIM)
        sequence = np.load(feature_path).astype(np.float32)
        
        # Augmentation on feature sequence (only during training if augment=True)
        if self.augment:
            # 1. Add small Gaussian noise to features
            noise = np.random.normal(0, 0.005, sequence.shape).astype(np.float32)
            sequence += noise
            
            # 2. Temporal Jitter (slightly shift sequence frames in time)
            if np.random.rand() > 0.5:
                # Roll sequence along time axis by -1, 0, or 1 frames
                shift = np.random.choice([-1, 1])
                sequence = np.roll(sequence, shift, axis=0)
                
        return torch.tensor(sequence), torch.tensor(label_idx, dtype=torch.long)

def get_data_loaders(split_mode="signer", batch_size=config.BATCH_SIZE, val_signer="signer_5", test_signer="signer_6"):
    """
    Splits the dataset and returns Train, Val, and Test PyTorch DataLoaders.
    """
    excel_path = config.EXCEL_DETAILS_PATH
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Details excel sheet not found at {excel_path}")
        
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['Sentences'])
    df = df[df['File location'].apply(lambda x: isinstance(x, str) and not x.startswith('>>>'))].copy()
    
    # Extract signers
    df['Signer'] = df['File location'].apply(get_signer_id)
    
    # Load label mapping
    label_to_idx, idx_to_label = load_label_mapping()
    
    if split_mode == "full":
        print("[Split] Performing Full-Dataset Training Split.")
        print("        Train set    : all available samples")
        print("        Val set      : mirrored train set for checkpoint monitoring")
        print("        Test set     : empty")

        train_df = df.copy()
        val_df = df.copy()
        test_df = df.iloc[0:0].copy()

    elif split_mode == "signer":
        print(f"[Split] Performing Signer-Independent Split.")
        print(f"        Train signers: All except {val_signer} and {test_signer}")
        print(f"        Val signer   : {val_signer}")
        print(f"        Test signer  : {test_signer}")
        
        train_df = df[(df['Signer'] != val_signer) & (df['Signer'] != test_signer)]
        val_df = df[df['Signer'] == val_signer]
        test_df = df[df['Signer'] == test_signer]
        
    else:  # Class-balanced random split
        print("[Split] Performing Class-Balanced Random Split.")
        train_rows = []
        val_rows = []
        test_rows = []
        
        for sentence, group in df.groupby('Sentences'):
            # Shuffle group samples
            group_shuffled = group.sample(frac=1, random_state=42)
            n = len(group_shuffled)
            if n == 1:
                train_rows.append(group_shuffled.iloc[0])
            elif n == 2:
                train_rows.append(group_shuffled.iloc[0])
                test_rows.append(group_shuffled.iloc[1])
            else:
                val_rows.append(group_shuffled.iloc[0])
                test_rows.append(group_shuffled.iloc[1])
                for i in range(2, n):
                    train_rows.append(group_shuffled.iloc[i])
                    
        train_df = pd.DataFrame(train_rows)
        val_df = pd.DataFrame(val_rows)
        test_df = pd.DataFrame(test_rows)
        
    print(f"[Split] Metadata splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = ISLDataset(train_df, label_to_idx, augment=True)
    val_dataset = ISLDataset(val_df, label_to_idx, augment=False)
    test_dataset = ISLDataset(test_df, label_to_idx, augment=False)
    
    # Create DataLoaders
    # Note: drop_last=False to make sure we don't drop evaluation samples
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, label_to_idx, idx_to_label

if __name__ == "__main__":
    # Test DataLoaders locally
    print("Testing Dataset Loader...")
    try:
        train_loader, val_loader, test_loader, l2i, i2l = get_data_loaders(split_mode="stratified", batch_size=2)
        print("Success! Data loaders initialized.")
        if len(train_loader) > 0:
            seq_batch, label_batch = next(iter(train_loader))
            print("Sequence batch shape:", seq_batch.shape)
            print("Label batch shape:", label_batch.shape)
    except Exception as e:
        print("Error during test:", e)
