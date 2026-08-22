import os
import sys
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Ensure config and ml modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from ml.feature_extraction import FeatureExtractor

def process_videos(limit=None):
    print("=" * 60)
    print("      GESCOM: FEATURE EXTRACTION & PREPROCESSING PIPELINE")
    print("=" * 60)
    
    excel_path = config.EXCEL_DETAILS_PATH
    if not os.path.exists(excel_path):
        print(f"Error: Labeled Excel sheet not found at '{excel_path}'")
        return False
        
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['Sentences'])
    df = df[df['File location'].apply(lambda x: isinstance(x, str) and not x.startswith('>>>'))]
    
    total_videos = len(df)
    print(f"Total videos to process: {total_videos}")
    
    # Initialize extractor
    extractor = FeatureExtractor(static_image_mode=False)
    
    # Track statistics
    skipped_count = 0
    processed_count = 0
    failed_count = 0
    no_face_warning_count = 0
    
    # Process videos
    for idx, row in tqdm(df.iterrows(), total=total_videos, desc="Processing videos"):
        if limit is not None and processed_count >= limit:
            print(f"Reached processing limit of {limit} videos.")
            break
            
        sentence = row['Sentences']
        rel_path = row['File location']
        
        # Build paths
        full_path = os.path.join(os.path.dirname(config.DATASET_PATH), rel_path)
        if not os.path.exists(full_path):
            full_path = os.path.join(config.DATASET_PATH, "..", rel_path)
            if not os.path.exists(full_path):
                print(f"\n[Warning] File not found: {rel_path}")
                failed_count += 1
                continue
                
        # Unique name for output feature file
        safe_name = rel_path.replace("\\", "_").replace("/", "_").replace(" ", "_")
        feature_filename = f"{safe_name}.npy"
        feature_filepath = os.path.join(config.FEATURES_DIR, feature_filename)
        
        # CHECKPOINTING: Skip if already processed
        if os.path.exists(feature_filepath):
            skipped_count += 1
            continue
            
        # Extract features from video
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            print(f"\n[Warning] Failed to open video: {rel_path}")
            failed_count += 1
            cap.release()
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"\n[Warning] Empty or corrupted video (0 frames): {rel_path}")
            failed_count += 1
            cap.release()
            continue
            
        # Determine sampling indices
        indices = np.linspace(0, total_frames - 1, config.SEQUENCE_LENGTH, dtype=int)
        
        # Read frames and extract features
        seq_features = []
        frame_idx = 0
        success = True
        
        # Store decoded frames at indices to avoid reading all if sequence length is small
        sampled_frames = {}
        for idx_to_read in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_to_read)
            ret, frame = cap.read()
            if ret:
                sampled_frames[idx_to_read] = frame
                
        cap.release()
        
        if len(sampled_frames) < config.SEQUENCE_LENGTH / 2:
            print(f"\n[Warning] Could not read enough frames from: {rel_path}")
            failed_count += 1
            continue
            
        # Run landmark extractor on sampled frames
        no_face_in_any = True
        last_valid_features = np.zeros(config.INPUT_DIM, dtype=np.float32)
        
        for step, idx_to_read in enumerate(indices):
            frame = sampled_frames.get(idx_to_read, None)
            if frame is None:
                # Pad with last valid feature
                seq_features.append(last_valid_features)
                continue
                
            feats, cues = extractor.extract_features(frame)
            
            # Check if face was detected (non-zero face mesh features)
            # The custom features are from index 0 to 18, and face features start from index 18
            # If all face features are zero, face was not detected
            face_detected = np.any(feats[18:378] != 0)
            
            if face_detected:
                no_face_in_any = False
                last_valid_features = feats
                seq_features.append(feats)
            else:
                # Face missing in this frame: carry forward the last valid face frame features
                seq_features.append(last_valid_features)
                
        seq_features = np.array(seq_features, dtype=np.float32)
        
        if no_face_in_any:
            no_face_warning_count += 1
            
        # Save sequence features
        np.save(feature_filepath, seq_features)
        processed_count += 1
        
    extractor.close()
    
    print("\n" + "=" * 40)
    print("Preprocessing Statistics:")
    print("=" * 40)
    print(f"Total Videos Checked: {total_videos}")
    print(f"Already Processed (Skipped): {skipped_count}")
    print(f"Newly Processed: {processed_count}")
    print(f"Failed/Missing: {failed_count}")
    print(f"No Face Detected in Video (Warnings): {no_face_warning_count}")
    print(f"Feature sequences stored in: {config.FEATURES_DIR}")
    print("=" * 40)
    return True

if __name__ == "__main__":
    # Process all videos in the dataset
    process_videos(limit=None)
