"""
Verify that downloaded tensor cache / checkpoint files actually load, not just
that they exist on disk. download_files.py only guards against truncation
mid-download (it deletes partial files on failure) — this catches anything
that finished writing but is still corrupt (bad blob upload, disk error,
silent truncation the HTTP client didn't flag).

Run in bdt-env (needs torch), after download_files.py finishes:
    conda activate bdt-env
    python verify_downloads.py
"""
import json
import pickle
import sys
import time
from pathlib import Path

import torch

CACHE_DIR = Path(r"C:\Users\seif\neuro_dt\tensor_cache")
CKPT_DIR = Path(r"C:\Users\seif\neuro_dt\checkpoints")
EXPECTED_TENSOR_COUNT = 1549


def check_pt_files(directory, pattern, label):
    files = sorted(directory.glob(pattern))
    print(f"\nChecking {len(files)} {label} file(s) in {directory}...")
    bad = []
    start = time.time()
    for i, f in enumerate(files, 1):
        try:
            torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  ✗ CORRUPT: {f.name} ({e})")
            bad.append(f.name)
        if i % 200 == 0:
            elapsed = time.time() - start
            print(f"  {i}/{len(files)} checked | {elapsed/60:.1f} min elapsed")
    return files, bad


def check_pickle(path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            pickle.load(f)
        return True
    except Exception as e:
        print(f"  ✗ CORRUPT: {path.name} ({e})")
        return False


def check_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            json.load(f)
        return True
    except Exception as e:
        print(f"  ✗ CORRUPT: {path.name} ({e})")
        return False


if __name__ == "__main__":
    tensor_files, bad_tensors = check_pt_files(CACHE_DIR, "*.pt", "tensor cache")
    ckpt_files, bad_ckpts = check_pt_files(CKPT_DIR, "*.pth", "checkpoint")

    print("\nChecking other checkpoint artifacts...")
    markov_ok = check_pickle(CKPT_DIR / "markov_matrices.pkl")
    auc_ok = check_json(CKPT_DIR / "auc_results.json")
    ablation_path = CKPT_DIR / "ablation_results.json"
    ablation_ok = check_json(ablation_path)

    print(f"\n{'='*60}")
    print(f"Tensor cache : {len(tensor_files)}/{EXPECTED_TENSOR_COUNT} files present, {len(bad_tensors)} corrupt")
    print(f"Checkpoints  : {len(ckpt_files)} .pth files present, {len(bad_ckpts)} corrupt")
    print(f"markov_matrices.pkl : {'OK' if markov_ok else ('MISSING' if markov_ok is None else 'CORRUPT')}")
    print(f"auc_results.json    : {'OK' if auc_ok else ('MISSING' if auc_ok is None else 'CORRUPT')}")
    if ablation_ok is None:
        print("ablation_results.json : MISSING — CPU ablation results (A0, A1, B1-B4) not on this machine,")
        print("                         Cell A2 will re-run them instead of skipping. Check the Azure container.")
    else:
        print(f"ablation_results.json : {'OK' if ablation_ok else 'CORRUPT'}")

    if len(tensor_files) < EXPECTED_TENSOR_COUNT:
        print(f"\n⚠ Only {len(tensor_files)}/{EXPECTED_TENSOR_COUNT} tensor cache files present — download isn't finished yet.")

    corrupt = bad_tensors + bad_ckpts
    if corrupt:
        print("\n⚠ Delete these and rerun download_files.py to re-fetch them:")
        for name in corrupt:
            print(f"    {name}")
        sys.exit(1)
    elif len(tensor_files) == EXPECTED_TENSOR_COUNT and not bad_ckpts:
        print("\n✓ All present files loaded successfully.")
