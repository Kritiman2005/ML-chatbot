import os
from pathlib import Path

def ensure_directories(*dirs):
    """Ensure that directories exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file, save_dir="sample_docs"):
    """Save uploaded file to local directory."""
    ensure_directories(save_dir)
    save_path = os.path.join(save_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    return save_path
