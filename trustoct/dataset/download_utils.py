import os
import sys
import shutil

def download_kermany_dataset(target_dir="./data/OCT2017"):
    """
    Downloads the Kermany/Mendeley OCT2017 dataset directly into Colab's disk.
    No Google Drive storage needed — uses Colab's free ~100GB VM disk.
    
    Priority order:
      1. Check if dataset already exists locally
      2. Try kagglehub (simplest, auto-handles auth)
      3. Try Kaggle CLI API
      4. Manual instructions fallback
    """
    target_dir = os.path.abspath(target_dir)
    
    # Check if already downloaded
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        train_count = sum(len(os.listdir(os.path.join(train_dir, c))) 
                         for c in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, c)))
        if train_count > 100:
            print(f"[Dataset] Already exists at: {target_dir} ({train_count} train images)")
            return target_dir

    print("[Dataset] Kermany OCT dataset not found. Downloading...")
    os.makedirs(target_dir, exist_ok=True)

    # =========================================================
    # Method 1: kagglehub (simplest — works in Colab by default)
    # =========================================================
    try:
        import kagglehub
        print("[Dataset] Method 1: Downloading via kagglehub...")
        print("[Dataset] This may take 5-15 minutes depending on connection speed.")
        path = kagglehub.dataset_download("paultimothymooney/kermany2018")
        print(f"[Dataset] Downloaded to cache: {path}")
        
        # Find the actual OCT2017 directory inside the download
        oct_dir = _find_oct_directory(path)
        if oct_dir:
            _copy_dataset(oct_dir, target_dir)
            print(f"[Dataset] ✅ Dataset ready at: {target_dir}")
            return target_dir
        else:
            print("[Dataset] Warning: Could not find OCT2017 structure in download.")
    except ImportError:
        print("[Dataset] kagglehub not installed. Trying Method 2...")
    except Exception as e:
        print(f"[Dataset] kagglehub error: {e}. Trying Method 2...")

    # =========================================================
    # Method 2: Kaggle CLI API
    # =========================================================
    try:
        print("[Dataset] Method 2: Downloading via Kaggle CLI...")
        
        # Check if running in Colab
        in_colab = 'google.colab' in sys.modules
        
        if in_colab:
            print("[Dataset] Detected Google Colab environment.")
            print("[Dataset] Setting up Kaggle API...")
            # In Colab, user needs to upload kaggle.json or set env vars
            _setup_kaggle_colab()
        
        import subprocess
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'paultimothymooney/kermany2018', 
             '-p', target_dir, '--unzip'],
            capture_output=True, text=True, timeout=1200
        )
        
        if result.returncode == 0:
            oct_dir = _find_oct_directory(target_dir)
            if oct_dir and oct_dir != target_dir:
                _copy_dataset(oct_dir, target_dir)
            print(f"[Dataset] ✅ Dataset ready at: {target_dir}")
            return target_dir
        else:
            print(f"[Dataset] Kaggle CLI error: {result.stderr}")
    except FileNotFoundError:
        print("[Dataset] Kaggle CLI not found.")
    except Exception as e:
        print(f"[Dataset] Kaggle CLI error: {e}")

    # =========================================================
    # Method 3: Manual instructions
    # =========================================================
    print("\n" + "="*60)
    print(" MANUAL DATASET SETUP REQUIRED")
    print("="*60)
    print("""
    The automated download didn't work. Please download manually:
    
    Option A — In Colab (recommended):
    
      # Step 1: Upload your kaggle.json (get it from kaggle.com → Account → API)
      from google.colab import files
      files.upload()  # Upload kaggle.json
      
      # Step 2: Setup and download
      !mkdir -p ~/.kaggle
      !cp kaggle.json ~/.kaggle/
      !chmod 600 ~/.kaggle/kaggle.json
      !kaggle datasets download -d paultimothymooney/kermany2018 -p data/ --unzip
    
    Option B — Direct URL:
    
      Download from: https://data.mendeley.com/datasets/rscbjbr9sj/2
      Unzip into: data/OCT2017/ with train/, test/, val/ subdirectories
    """)
    print("="*60)
    
    return target_dir


def _find_oct_directory(base_path):
    """Recursively search for the OCT2017 directory structure (train/test with class folders)."""
    # Check if base_path itself has train/test
    for name in ['', 'OCT2017', 'OCT2017 ', 'kermany2018', 'oct2017']:
        candidate = os.path.join(base_path, name) if name else base_path
        train_check = os.path.join(candidate, 'train')
        test_check = os.path.join(candidate, 'test')
        if os.path.exists(train_check) and os.path.exists(test_check):
            return candidate
    
    # Deep search (up to 3 levels)
    for root, dirs, files in os.walk(base_path):
        depth = root.replace(base_path, '').count(os.sep)
        if depth > 3:
            continue
        if 'train' in dirs and 'test' in dirs:
            return root
    
    return None


def _copy_dataset(src_dir, dst_dir):
    """Copy train/test/val directories to target location."""
    for split in ['train', 'test', 'val']:
        src = os.path.join(src_dir, split)
        dst = os.path.join(dst_dir, split)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"[Dataset] Copying {split}/...")
            shutil.copytree(src, dst)
        elif os.path.exists(src) and os.path.exists(dst):
            print(f"[Dataset] {split}/ already exists, skipping.")


def _setup_kaggle_colab():
    """Guide user to set up Kaggle API in Colab."""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print("[Dataset] Kaggle API key found.")
        return
    
    # Check if kaggle.json exists in current directory (user uploaded it)
    if os.path.exists("kaggle.json"):
        os.makedirs(kaggle_dir, exist_ok=True)
        shutil.copy("kaggle.json", kaggle_json)
        os.chmod(kaggle_json, 0o600)
        print("[Dataset] Kaggle API key configured from uploaded file.")
        return
    
    print("[Dataset] No Kaggle API key found.")
    print("[Dataset] Please run this in a Colab cell first:")
    print('  from google.colab import files; files.upload()  # upload kaggle.json')


if __name__ == "__main__":
    download_kermany_dataset()
