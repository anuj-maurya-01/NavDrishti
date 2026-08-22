import subprocess
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def log_header(title):
    print("=" * 70)
    print(f"   {title.upper()}")
    print("=" * 70)

def run_step(command, desc):
    log_header(f"Starting: {desc}")
    start_t = time.time()
    
    # Run the command and stream output
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=BASE_DIR
    )
    
    # Print output line by line
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        
    process.stdout.close()
    return_code = process.wait()
    
    duration = time.time() - start_t
    if return_code == 0:
        print(f"\n[Success] {desc} completed in {duration/60:.2f} minutes.\n")
        return True
    else:
        print(f"\n[Error] {desc} failed with exit code {return_code}.\n")
        return False

def main():
    print("=" * 70)
    print("      GESCOM: FULL END-TO-END PIPELINE RUNNER")
    print("=" * 70)
    
    # Step 1: Preprocess all videos
    success = run_step("python ml/preprocessing.py", "Dataset Preprocessing (Landmark Extraction)")
    if not success:
        print("Pipeline aborted due to preprocessing failure.")
        return
        
    # Step 2: Train model on all features
    success = run_step("python ml/train.py --epochs 35 --split full", "Neural Network Training On Full ISL_CSLRT_Corpus")
    if not success:
        print("Pipeline aborted due to training failure.")
        return
        
    # Step 3: Evaluate model
    success = run_step("python ml/evaluate.py --model temporal --split full", "Model Evaluation On Full ISL_CSLRT_Corpus")
    if not success:
        print("Pipeline aborted due to evaluation failure.")
        return
        
    # Step 4: Restart Flask Backend
    log_header("Restarting Flask Backend Server")
    print("Starting Flask app.py in background...")
    
    # Run app.py in background
    # Since we want it to run as a separate process, we start it without waiting
    subprocess.Popen("python app.py", shell=True, cwd=BASE_DIR)
    print("[Success] Flask backend restarted successfully with the new model weights.")
    print("Open http://127.0.0.1:5000 in your browser to test the full model!")
    print("=" * 70)

if __name__ == "__main__":
    main()
