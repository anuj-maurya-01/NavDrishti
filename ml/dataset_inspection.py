import os
import sys
import pandas as pd
import cv2
import numpy as np

# Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def inspect_dataset():
    print("=" * 60)
    print("      GESCOM: DATASET INSPECTION & STATISTICS REPORT")
    print("=" * 60)
    
    dataset_path = config.DATASET_PATH
    print(f"Dataset root: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return False
        
    # Count files and folders
    total_files = 0
    video_count = 0
    image_count = 0
    txt_count = 0
    xlsx_count = 0
    csv_count = 0
    other_count = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            total_files += 1
            ext = f.lower()
            if ext.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_count += 1
            elif ext.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_count += 1
            elif ext.endswith('.txt'):
                txt_count += 1
            elif ext.endswith('.xlsx'):
                xlsx_count += 1
            elif ext.endswith('.csv'):
                csv_count += 1
            else:
                other_count += 1
                
    print(f"Total files in dataset folder: {total_files}")
    print(f"  Videos: {video_count}")
    print(f"  Images: {image_count}")
    print(f"  Excel sheets: {xlsx_count}")
    print(f"  CSV files: {csv_count}")
    print(f"  Text files: {txt_count}")
    print(f"  Other files: {other_count}")
    
    # Inspect Excel mapping
    excel_path = config.EXCEL_DETAILS_PATH
    if not os.path.exists(excel_path):
        print(f"Error: Details excel sheet '{excel_path}' not found.")
        return False
        
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['Sentences'])
    # Filter out trash rows (like the trailing >>>)
    df = df[df['File location'].apply(lambda x: isinstance(x, str) and not x.startswith('>>>'))]
    
    total_samples = len(df)
    unique_classes = df['Sentences'].nunique()
    
    print("\n" + "-" * 40)
    print("Class & Sample Statistics")
    print("-" * 40)
    print(f"Total labeled video samples in details Excel: {total_samples}")
    print(f"Total unique classes (Sentences): {unique_classes}")
    
    # Class distribution
    class_counts = df['Sentences'].value_counts()
    print("\nSample Distribution (Top 10 Classes):")
    for i, (cls_name, count) in enumerate(class_counts.head(10).items()):
        print(f"  {i+1:2d}. {cls_name:<40} : {count} samples")
        
    print("\nSample Distribution (Bottom 5 Classes):")
    for i, (cls_name, count) in enumerate(class_counts.tail(5).items()):
        print(f"  {i+1:2d}. {cls_name:<40} : {count} samples")
        
    # Analyze Video Resolutions, FPS and Duration
    print("\n" + "-" * 40)
    print("Video Attributes & Formats")
    print("-" * 40)
    
    resolutions = []
    fps_list = []
    durations = []
    frame_counts = []
    corrupted_files = []
    
    # Sample up to 30 videos to inspect performance
    sample_paths = df['File location'].sample(min(30, len(df)), random_state=42).tolist()
    
    for item in sample_paths:
        full_path = os.path.join(os.path.dirname(dataset_path), item)
        if not os.path.exists(full_path):
            # Try joining directly
            full_path = os.path.join(dataset_path, "..", item)
            if not os.path.exists(full_path):
                corrupted_files.append(item)
                continue
                
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            corrupted_files.append(item)
            cap.release()
            continue
            
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if frames > 0 and fps > 0:
            duration = frames / fps
            durations.append(duration)
            resolutions.append(f"{int(w)}x{int(h)}")
            fps_list.append(fps)
            frame_counts.append(frames)
            
        cap.release()
        
    if resolutions:
        unique_resolutions = set(resolutions)
        print(f"Detected Resolutions: {', '.join(unique_resolutions)}")
        print(f"Average Frame Rate (FPS): {np.mean(fps_list):.2f} (Min: {np.min(fps_list):.2f}, Max: {np.max(fps_list):.2f})")
        print(f"Average Duration: {np.mean(durations):.2f} seconds (Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s)")
        print(f"Average Frame Count per Video: {np.mean(frame_counts):.1f} (Min: {np.min(frame_counts)}, Max: {np.max(frame_counts)})")
    else:
        print("Could not analyze sample videos.")
        
    print(f"Corrupted or Missing videos detected: {len(corrupted_files)}")
    for f in corrupted_files[:5]:
        print(f"  Missing: {f}")
        
    # Write a dataset report to docs/dataset.md
    docs_dir = os.path.join(config.BASE_DIR, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, "dataset.md")
    
    with open(report_path, "w") as rf:
        rf.write("# Dataset Statistics Report - ISL_CSLRT_Corpus\n\n")
        rf.write("## Overview\n")
        rf.write(f"- **Dataset Root Path**: `{dataset_path}`\n")
        rf.write(f"- **Total Scan Files**: {total_files}\n")
        rf.write(f"- **Total Video Files**: {video_count}\n")
        rf.write(f"- **Total Image Files (Frames)**: {image_count}\n")
        rf.write(f"- **Excel Index Sheets**: {xlsx_count}\n")
        rf.write(f"- **CSV Sign Gloss Files**: {csv_count}\n\n")
        rf.write("## Annotation Metadata\n")
        rf.write(f"- **Total Video Samples Listed**: {total_samples}\n")
        rf.write(f"- **Total Class Labels (Sentences)**: {unique_classes}\n")
        rf.write(f"- **Average Video FPS**: {np.mean(fps_list):.2f} Hz\n")
        rf.write(f"- **Average Duration**: {np.mean(durations):.2f} seconds\n")
        rf.write(f"- **Average Frame Count**: {np.mean(frame_counts):.1f} frames\n")
        rf.write(f"- **Typical Video Resolution**: {', '.join(unique_resolutions) if resolutions else 'Unknown'}\n\n")
        rf.write("## Class Distribution (Top 10 classes)\n")
        rf.write("| Sentence Label | Number of Video Samples |\n")
        rf.write("| --- | --- |\n")
        for cls_name, count in class_counts.head(10).items():
            rf.write(f"| {cls_name} | {count} |\n")
            
    print(f"\n[Success] Dataset inspection complete. Statistics report saved to {report_path}")
    return True

if __name__ == "__main__":
    inspect_dataset()
